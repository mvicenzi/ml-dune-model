"""Pixel-level PID k-NN — the cheap non-parametric tracking metric.

The complement to `probe_pid.py`: instead of training a head, it asks whether a
pixel's nearest neighbours in cosine feature space already carry its class. Fast,
no fitting — useful as a per-epoch curve while a run trains.

Classes are the 7-class pixel taxonomy with Background (label 0) excluded, so
this scores motifs 1-6 only. It reports majority-vote accuracy (per class =
recall) plus a confusion matrix, and optionally neighbourhood purity at several
k. Student and teacher features are scored side by side.

Writes JSON and nothing else. `probes/plot_probes.py` draws the per-class recall
and the confusion matrix from it.

Ported from `dino/diagnostics/plot_knn_pixel.py`, which is left untouched so its
published numbers stay reproducible. Two differences here:

  1. **A per-image cap on the pools** (`--max_pixels_per_image`). The quota alone
     does not spread a pool over events: abundant classes reach it almost
     immediately, and on prod-jay at 5000/class the old pools came from 8 events
     for Track and 2 for Shower (188 for Other). A 2-event pool does not
     represent its class, and the k-NN degenerates towards "are pixels of this
     one shower near each other" — which is why the uncapped Track recall was
     0.777 against a leakage-free probe's 0.394. With the cap every pool spans
     hundreds of events and the metric agrees with the probe to ~0.02 overall.
     Pass a negative value for the old uncapped behaviour; the two are NOT
     numerically comparable.
  2. **JSON output** keyed like every other probe, so results merge into
     `probes.merge` tables alongside the trained-head metrics.

Note this metric has no train/val split — queries and neighbours come from one
pool — so it is a *relative* tracking curve, not a leakage-free score. Pool
selection depends only on labels and offsets (never on features) and is seeded,
so it is identical for every checkpoint and safe to compare across epochs. Do not
quote it beside `probe_pid` numbers as if the protocols matched.

Usage:
    python -m probes.probe_knn_pid FEATURES.npz --out knn_pixel.json
    python -m probes.probe_knn_pid FEATURES.npz --max_pixels_per_class=50000 \\
        --knn_k=5 --device=cuda --with_purity
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from probes.features import load_features
from probes.probe_pid import PID_NAMES
from probes.results import run_header, run_label, write_json

# Motifs 1-6; Background (label 0) is excluded from this metric entirely.
PIXEL_CLASS_NAMES = PID_NAMES[1:]

# Number of events a class pool should ideally be spread over.
_SPREAD_TARGET_EVENTS = 200


def _label_to_class(labels: np.ndarray) -> np.ndarray:
    """Truth labels (int8 0..6) -> class index 0..5; -1 for Background/no-truth."""
    out = labels.astype(np.int32) - 1
    out[labels == 0] = -1
    return out


def auto_per_image_cap(max_pixels_per_class: int, n_images: int) -> int:
    """Per-image cap that spreads a pool over ~`_SPREAD_TARGET_EVENTS` events.

    Adapts to the file: when it holds fewer events than the target, the cap grows
    so the quota can still be filled from what is there (a 200-event extraction
    should not be forced into 200-event-wide pools it cannot supply). Always >= 1.
    """
    target = max(1, min(int(n_images), _SPREAD_TARGET_EVENTS))
    return max(1, -(-int(max_pixels_per_class) // target))    # ceil division


# ---------------------------------------------------------------------------
# Stratified pixel collection
# ---------------------------------------------------------------------------

def collect(
    feats_by_src: dict,
    global_cls: np.ndarray,
    offsets: np.ndarray,
    max_pixels_per_class: int,
    seed: int = 42,
    max_pixels_per_image: int = 0,
) -> tuple:
    """Visit images in random order, filling per-class pools.

    Takes at most `max_pixels_per_image` pixels of one class from any single
    image, so the pool spans many events (see the module docstring for why).

    max_pixels_per_image: per class, per image. 0 = auto (see
        `auto_per_image_cap`); negative = uncapped, the pre-fix behaviour.

    A class present in too few events may fall short of its quota; that is
    reported, not an error.

    `feats_by_src` maps a branch name ("student" / "teacher") to its feature
    array; every branch is pooled on the SAME pixel choices, and a file that
    carries only one branch simply passes one entry.

    Returns (pix_by_src, pix_cls, counts, n_events, cap_used).
    """
    rng = np.random.default_rng(seed)
    n_images  = len(offsets) - 1
    n_classes = len(PIXEL_CLASS_NAMES)

    if max_pixels_per_image == 0:
        cap = auto_per_image_cap(max_pixels_per_class, n_images)
    elif max_pixels_per_image < 0:
        cap = None
    else:
        cap = int(max_pixels_per_image)

    pools = {src: [[] for _ in range(n_classes)] for src in feats_by_src}
    counts   = np.zeros(n_classes, dtype=np.int64)
    n_events = np.zeros(n_classes, dtype=np.int64)

    for img_idx in rng.permutation(n_images):
        if counts.min() >= max_pixels_per_class:
            break

        sl  = slice(int(offsets[img_idx]), int(offsets[img_idx + 1]))
        cls = global_cls[sl]
        sliced = {src: X[sl] for src, X in feats_by_src.items()}

        for c in range(n_classes):
            need = max_pixels_per_class - int(counts[c])
            if need <= 0:
                continue
            mask    = cls == c
            n_avail = int(mask.sum())
            if n_avail == 0:
                continue
            take = min(need, n_avail) if cap is None else min(need, n_avail, cap)
            sel = (np.where(mask)[0] if take == n_avail
                   else rng.choice(np.where(mask)[0], size=take, replace=False))
            for src, X in sliced.items():
                pools[src][c].append(X[sel])
            counts[c]   += take
            n_events[c] += 1

    parts = {src: [] for src in feats_by_src}
    lbl_parts = []
    first = next(iter(feats_by_src))
    for c in range(n_classes):
        if not pools[first][c]:
            continue
        for src in feats_by_src:
            parts[src].append(np.concatenate(pools[src][c]))
        lbl_parts.append(np.full(len(parts[first][-1]), c, dtype=np.int64))
    if not lbl_parts:
        raise SystemExit("no labelled pixels collected — is pixel_labels present?")

    pix_by_src = {src: np.concatenate(v) for src, v in parts.items()}
    return pix_by_src, np.concatenate(lbl_parts), counts, n_events, cap


# ---------------------------------------------------------------------------
# Batched k-NN
# ---------------------------------------------------------------------------

def _l2_normalise(X: torch.Tensor) -> torch.Tensor:
    return X / X.norm(dim=1, keepdim=True).clamp(min=1e-8)


def knn_purity(feats, labels, ks, device, batch_size) -> dict:
    """{k: (overall_purity, per_class[n_classes], per_sample[N])}."""
    n_classes = len(PIXEL_CLASS_NAMES)
    X    = _l2_normalise(torch.from_numpy(feats.astype(np.float32)).to(device))
    N    = X.shape[0]
    lbls = torch.from_numpy(labels.astype(np.int64)).to(device)
    max_k = min(max(ks), N - 1)

    nn_labels = torch.empty(N, max_k, dtype=torch.int64, device=device)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B   = end - start
        sim = X[start:end] @ X.T
        sim[torch.arange(B, device=device),
            torch.arange(start, end, device=device)] = -torch.inf
        _, idx = sim.topk(max_k, dim=1)
        nn_labels[start:end] = lbls[idx]

    out = {}
    for k in ks:
        k_eff = min(k, N - 1)
        same  = (nn_labels[:, :k_eff] == lbls[:, None]).float().mean(dim=1)
        per_class = np.full(n_classes, np.nan)
        for c in np.unique(labels):
            per_class[c] = float(same[lbls == c].mean())
        out[k] = (float(same.mean()), per_class, same.cpu().numpy())
    return out


def knn_predict(feats, labels, k, device, batch_size) -> np.ndarray:
    """Majority-vote k-NN predictions [N], self excluded."""
    n_classes = len(PIXEL_CLASS_NAMES)
    X     = _l2_normalise(torch.from_numpy(feats.astype(np.float32)).to(device))
    N     = X.shape[0]
    lbls  = torch.from_numpy(labels.astype(np.int64)).to(device)
    k_eff = min(k, N - 1)

    preds = torch.empty(N, dtype=torch.int64, device=device)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        B   = end - start
        sim = X[start:end] @ X.T
        sim[torch.arange(B, device=device),
            torch.arange(start, end, device=device)] = -torch.inf
        _, idx  = sim.topk(k_eff, dim=1)
        nn_lbls = lbls[idx]
        off     = torch.arange(B, device=device).unsqueeze(1) * n_classes
        flat    = (nn_lbls + off).reshape(-1)
        cnt     = torch.bincount(flat, minlength=B * n_classes).reshape(B, n_classes)
        preds[start:end] = cnt.argmax(dim=1)
    return preds.cpu().numpy()


def accuracy(preds: np.ndarray, labels: np.ndarray) -> tuple:
    """(overall accuracy, per-type recall, per-type F1, macro-F1).

    Per-type accuracy is recall, which is blind to false alarms: a type that gets
    over-predicted inflates its own number. F1 adds the precision side and is
    free here — the majority vote already produced hard labels. Macro-F1 is the
    figure comparable in form (not in protocol) with the trained PID readout.
    """
    from sklearn.metrics import f1_score

    n_classes = len(PIXEL_CLASS_NAMES)
    per_class = np.full(n_classes, np.nan)
    for c in range(n_classes):
        m = labels == c
        if m.any():
            per_class[c] = float((preds[m] == c).mean())
    f1 = f1_score(labels, preds, average=None, labels=list(range(n_classes)),
                  zero_division=0)
    present = [c for c in range(n_classes) if (labels == c).any()]
    macro_f1 = float(np.mean([f1[c] for c in present])) if present else float("nan")
    return float((preds == labels).mean()), per_class, np.asarray(f1, float), macro_f1


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_one(npz_path: Path, args) -> dict:
    # Which branches the file carries. `load_features` takes one at a time, so
    # peek at the archive index first (np.load is lazy — this reads no arrays).
    sources = [b for b in ("student", "teacher")
               if f"{b}_features" in np.load(npz_path, allow_pickle=True).files]
    if not sources:
        raise SystemExit(f"{npz_path.name} has no student_features or "
                         f"teacher_features.")

    # Same loader as every other probe: it resolves the pixel<->event mapping and
    # rejects a file whose offsets and features disagree, per branch.
    fxs = {src: load_features(npz_path, source=src) for src in sources}
    fx0 = fxs[sources[0]]
    fx0.require("pixel_labels")

    print(f"\n=== {run_label(npz_path, sources[0])} ===")
    feats_by_src = {src: fx.feat for src, fx in fxs.items()}
    offsets = fx0.offsets
    pixel_labels = fx0.truth["pixel_labels"]
    n_events = fx0.n_events
    n_truth = int((pixel_labels != 0).sum())
    print(f"  events={n_events}  pixels={fx0.n_pixels}  "
          f"with truth={n_truth} ({100*n_truth/fx0.n_pixels:.1f}%)  "
          f"D={fx0.feat.shape[1]}  branches={sources}")

    pix_by_src, pix_cls, counts, ev_per_class, cap = collect(
        feats_by_src, _label_to_class(pixel_labels), offsets,
        max_pixels_per_class=args.max_pixels_per_class, seed=args.seed,
        max_pixels_per_image=args.max_pixels_per_image)

    print(f"  pixels per class per image: "
          f"{'uncapped (legacy)' if cap is None else cap}"
          f"{' (auto)' if args.max_pixels_per_image == 0 else ''}")
    for c, name in enumerate(PIXEL_CLASS_NAMES):
        short = "  << short of quota" if counts[c] < args.max_pixels_per_class else ""
        print(f"    {name:<9} {int(counts[c]):>8,} from {int(ev_per_class[c]):>5,} "
              f"events{short}")
    print(f"  total sampled: {len(pix_cls):,}")
    if ev_per_class.min() < 10:
        print("  WARNING: a class pool comes from fewer than 10 events; its score "
              "reflects those events, not the class")

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    k_eff = min(args.knn_k, len(pix_cls) - 1)
    preds = {src: knn_predict(X, pix_cls, k_eff, device, args.batch_size)
             for src, X in pix_by_src.items()}
    accs = {src: accuracy(pr, pix_cls) for src, pr in preds.items()}

    print(f"  k-NN majority vote (k={args.knn_k}), recall / F1 per type:")
    for c, name in enumerate(PIXEL_CLASS_NAMES):
        print(f"    {name:<9} " + "  ".join(
            f"{src} {accs[src][1][c]:.3f}/{accs[src][2][c]:.3f}" for src in sources))
    print(f"    {'Overall':<9} " + "  ".join(
        f"{src} acc {accs[src][0]:.3f}  macro-F1 {accs[src][3]:.3f}"
        for src in sources))

    from sklearn.metrics import confusion_matrix
    common = {
        "knn_k": int(args.knn_k),
        # Kept inside the metric, not hoisted to the entry, because `compare`
        # merges every metric for one checkpoint into a single entry: a top-level
        # `seed` or `pool_per_class` here would silently overwrite PID's, which
        # uses different values by design.
        "seed": int(args.seed),
        "n_pixels_scored": int(len(pix_cls)),
        "max_pixels_per_class": int(args.max_pixels_per_class),
        "max_pixels_per_image": ("uncapped" if cap is None else int(cap)),
        "classes": list(PIXEL_CLASS_NAMES),
        "class_counts": {n: int(counts[i]) for i, n in enumerate(PIXEL_CLASS_NAMES)},
        "events_per_class": {n: int(ev_per_class[i])
                             for i, n in enumerate(PIXEL_CLASS_NAMES)},
        "leakage_free": False,   # one pool, no train/val split — relative metric
    }
    entries = {}
    for src in sources:
        acc, pr = accs[src], preds[src]
        # The same provenance block every probe writes, minus the two run
        # settings that are per-metric (see `common` above).
        header = run_header(fxs[src], args.seed, args.max_pixels_per_class)
        header.pop("seed", None)
        header.pop("pool_per_class", None)
        entries[run_label(npz_path, src)] = {**header, "knn_pixel": {
            **common,
            "overall_accuracy": acc[0],
            "macro_f1": acc[3],
            "per_class_accuracy": {n: (None if not np.isfinite(acc[1][i]) else float(acc[1][i]))
                                   for i, n in enumerate(PIXEL_CLASS_NAMES)},
            "per_class_f1": {n: float(acc[2][i])
                             for i, n in enumerate(PIXEL_CLASS_NAMES)},
            "confusion": confusion_matrix(
                pix_cls, pr, labels=list(range(len(PIXEL_CLASS_NAMES)))).tolist(),
        }}

    if args.with_purity:
        ks = [int(k) for k in args.ks.split(",")]
        pur = {src: knn_purity(X, pix_cls, ks, device, args.batch_size)
               for src, X in pix_by_src.items()}
        for k in ks:
            print(f"    purity k={k:<3d} " + "  ".join(
                f"{src} {pur[src][k][0]:.3f}" for src in sources))
        for src in sources:
            entries[run_label(npz_path, src)]["knn_pixel"]["purity"] = {
                str(k): pur[src][k][0] for k in ks}

    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features", nargs="+", help="feature .npz file(s)")
    ap.add_argument("--out", default="pixel_knn.json", help="output JSON path")
    ap.add_argument("--max_pixels_per_class", type=int, default=50_000,
                    help="max pixels sampled per class (default: 50000)")
    ap.add_argument("--max_pixels_per_image", type=int, default=0,
                    help="max pixels of one class from a single image, so each pool "
                         "spans many events (default: 0 = auto, spreading over ~200 "
                         "events or all of them if the file has fewer; negative = "
                         "uncapped, the legacy behaviour, not comparable)")
    ap.add_argument("--knn_k", type=int, default=5, help="k for the majority vote")
    ap.add_argument("--ks", default="1,5,10,20", help="k values for --with_purity")
    ap.add_argument("--with_purity", action="store_true",
                    help="also compute neighbourhood label purity")
    ap.add_argument("--device", default="", help="torch device (default: cuda if available)")
    ap.add_argument("--batch_size", type=int, default=2048, help="k-NN query batch size")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_path = Path(args.out)
    results = {}
    for p in args.features:
        path = Path(p).resolve()
        if not path.exists():
            print(f"[skip] {path}: not found")
            continue
        results.update(run_one(path, args))
        write_json(results, out_path)

    if not results:
        sys.exit("no features scored")
    write_json(results, out_path)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
