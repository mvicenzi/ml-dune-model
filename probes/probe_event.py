"""Event-level flavor k-NN on mean-pooled frozen features.

Mean-pools each event's pixel features into one vector and asks whether events
of the same interaction flavor (numuCC / nueCC / NC) are neighbours in cosine
space. Pooling runs over a capped sample of each event's pixels
(`--max_pixels_per_event`), the same way `probe_knn_pid` caps per image.

Reported as neighbour-label purity and majority-vote accuracy at several k,
against two references:

  chance     what the degenerate answers score on this exact population —
             guess uniformly, and always predict the most common flavor. Both
             are measured rather than assumed, since neither is 1/3 once the
             flavors are unequal.
  raw        the same [channel, tick, log charge] raw-charge input the model
  charge     itself is fed, mean-pooled per event and scored identically; guards
             against a number that only reflects where an event sits or how much
             charge it deposited.

Scored on the natural event population: nothing is balanced here, because no
head is trained and so there is no prior for a readout to learn. The flavor mix
is recorded next to every score, which is what makes accuracy interpretable.

This is a coarse, cheap tracking metric: it has been seen to move very little
across ablations whose representations had visibly collapsed on other metrics, so
treat it as a sanity curve rather than a decider. `dino/diagnostics/plot_knn.py`
produces the plotted version of the same idea; this one writes JSON for
run-to-run tables.

Usage:
  python -m probes.probe_event FEATURES.npz [MORE.npz ...] --out event_probe.json
"""

import argparse
import time
from pathlib import Path

import numpy as np

from probes.features import raw_charge, load_features
from probes.results import run_header, run_label, write_json

# Event flavor classes (per-event `labels`). Defined here because this is the
# module that gives them meaning — the same convention `probe_pid` follows for the
# pixel taxonomy.
FLAVOR_NAMES = ["numuCC", "nueCC", "NC"]

KS = (1, 5, 10, 20)


def _flavor(c: int) -> str:
    return FLAVOR_NAMES[c] if c < len(FLAVOR_NAMES) else str(c)


# Pixels drawn per event before pooling; the cap is what keeps this probe off the
# whole feature array.
DEFAULT_MAX_PIXELS_PER_EVENT = 2000


def sample_pixels(offsets: np.ndarray, max_per_event: int, seed: int):
    """Up to `max_per_event` pixel indices per event, drawn without replacement.

    Returns the flat pixel indices and the CSR offsets of the sample. Events with
    fewer pixels than the cap keep all of them, so only the large events are
    thinned — which also evens out how precisely each event's mean is estimated,
    since event sizes here span 0 to ~52 000 pixels.
    """
    rng = np.random.RandomState(seed)
    counts = np.diff(offsets).astype(np.int64)
    take = np.minimum(counts, max_per_event)
    idx = np.empty(int(take.sum()), dtype=np.int64)
    pos = 0
    for e in range(len(counts)):
        n, t, lo = int(counts[e]), int(take[e]), int(offsets[e])
        if t == 0:
            continue
        idx[pos:pos + t] = (np.arange(lo, lo + n) if t == n
                            else lo + rng.choice(n, size=t, replace=False))
        pos += t
    return idx, np.concatenate([[0], np.cumsum(take)]).astype(np.int64)


def mean_pool(X: np.ndarray, offsets: np.ndarray,
              max_per_event: int = DEFAULT_MAX_PIXELS_PER_EVENT,
              seed: int = 0) -> np.ndarray:
    """Per-event mean over a capped pixel sample → [n_events, D], in float32.

    Only the sample is upcast, which is what every other probe does too — see
    `load_features`: consumers index a pool first so the cast lands on the small
    slice. Pooling the whole array instead would upcast all 55 M pixels, 13 GiB on
    top of the 6.6 GiB already held.

    Empty events yield NaN and are dropped by the caller.
    """
    n_events = len(offsets) - 1
    idx, off = sample_pixels(offsets, max_per_event, seed)
    counts = np.diff(off).astype(np.float64)
    sums = np.zeros((n_events, X.shape[1]), dtype=np.float32)
    # Feed reduceat only the non-empty events. Their starts are then strictly
    # increasing and all in range, so each segment is exactly one event: an empty
    # event contributes no pixels, so the next non-empty start still equals the
    # previous event's end. Passing the empty ones instead puts a start one past
    # the end when the last event is empty, and clamping that index would
    # silently steal a pixel from its neighbour.
    nz = np.nonzero(counts > 0)[0]
    if len(nz):
        sums[nz] = np.add.reduceat(np.asarray(X[idx], dtype=np.float32),
                                   off[nz].astype(np.intp), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = sums / counts[:, None]
    pooled[counts == 0] = np.nan
    assert len(pooled) == n_events
    return pooled


def trivial_scores(y: np.ndarray, n_classes: int, seed: int) -> dict:
    """What the degenerate answers score on this exact event population.

    Two of them, because they differ and neither is 1/3 in general: `uniform`
    guesses at random, and `majority` always predicts the most common flavor —
    so its accuracy is that flavor's share and its macro-F1 a third of a single
    F1, the other two being zero. `majority_tied` flags an arbitrary argmax
    tie-break.
    """
    from sklearn.metrics import f1_score

    counts = np.bincount(y, minlength=n_classes)
    maj = int(counts.argmax())
    rng = np.random.RandomState(seed + 7)
    out = {
        "majority_fraction": float(counts[maj] / len(y)),
        "majority_class": _flavor(maj),
        "majority_tied": bool((counts == counts[maj]).sum() > 1),
    }
    labels = list(range(n_classes))
    for name, pred in (("uniform", rng.randint(0, n_classes, len(y))),
                       ("majority", np.full(len(y), maj))):
        out[name] = {
            "accuracy": float((pred == y).mean()),
            "macro_f1": float(f1_score(y, pred, average="macro", labels=labels,
                                       zero_division=0)),
        }
    return out


def knn_curve(F: np.ndarray, y: np.ndarray, n_classes: int, ks=KS,
              device: str = "cpu") -> dict:
    """Cosine kNN purity, majority-vote accuracy and macro-F1 at each k.

    Self is excluded. Macro-F1 is worth having next to accuracy because the three
    flavors are not equally common: accuracy can be carried by the majority class,
    F1 averaged over classes cannot.
    """
    import torch
    from sklearn.metrics import f1_score

    dev = torch.device(device)
    X = torch.from_numpy(np.ascontiguousarray(F, dtype=np.float32)).to(dev)
    X = X / X.norm(dim=1, keepdim=True).clamp(min=1e-8)
    lab = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64)).to(dev)

    sim = X @ X.T
    sim.fill_diagonal_(-2.0)
    kmax = min(max(ks), len(X) - 1)
    nn_lab = lab[sim.topk(kmax, dim=1).indices]        # [N, kmax]

    labels = list(range(n_classes))
    out = {}
    for k in ks:
        k_eff = min(k, kmax)
        lab_k = nn_lab[:, :k_eff]
        purity = float((lab_k == lab[:, None]).float().mean())
        votes = torch.nn.functional.one_hot(lab_k, n_classes).sum(1)
        pred = votes.argmax(1)
        acc = float((pred == lab).float().mean())
        macro_f1 = float(f1_score(lab.cpu().numpy(), pred.cpu().numpy(),
                                  average="macro", labels=labels, zero_division=0))
        out[str(k)] = {"purity": purity, "accuracy": acc, "macro_f1": macro_f1}
    return out


def run_one(path: Path, args) -> dict:
    fx = load_features(path, source=args.source)
    print(f"\n=== {run_label(path, args.source)} ===")

    # Both sides pool the SAME sampled pixels, so the comparison is not confounded
    # by which pixels each side happened to see.
    cap, seed = args.max_pixels_per_event, args.seed
    pooled_feat = mean_pool(fx.feat, fx.offsets, cap, seed)
    pooled_raw = mean_pool(raw_charge(fx), fx.offsets, cap, seed)
    y = fx.labels

    keep = (y >= 0) & np.isfinite(pooled_feat).all(1) & np.isfinite(pooled_raw).all(1)
    n_unknown = int((y < 0).sum())
    n_bad = int(len(y) - keep.sum() - n_unknown)
    if keep.sum() < max(KS) + 2:
        return {"error": f"only {int(keep.sum())} usable events"}

    y = y[keep]
    pooled_feat, pooled_raw = pooled_feat[keep], pooled_raw[keep]
    n_classes = max(len(FLAVOR_NAMES), int(y.max()) + 1)
    hist = {_flavor(int(c)): int((y == c).sum()) for c in np.unique(y)}

    # `compare` merges every metric for one checkpoint into a single entry, so a
    # top-level `seed` or `pool_per_class` here would silently overwrite PID's,
    # which uses different values by design. Same treatment as probe_knn_pid.
    entry = run_header(fx, args.seed, 0)
    entry.pop("seed", None)
    entry.pop("pool_per_class", None)

    chance = trivial_scores(y, n_classes, args.seed)
    print(f"  events={len(y)} ({n_unknown} unknown, {n_bad} empty/non-finite dropped)")
    print(f"  classes={hist}  majority={chance['majority_class']} "
          f"({chance['majority_fraction']:.4f})")

    t0 = time.time()
    feat = knn_curve(pooled_feat, y, n_classes, device=args.device)
    raw = knn_curve(pooled_raw, y, n_classes, device=args.device)
    metric = {
        "seed": int(args.seed),
        "max_pixels_per_event": int(cap),
        "n_events_used": int(len(y)),
        "class_counts": hist,
        "chance": chance,
        "n_unknown_dropped": n_unknown,
        "n_nonfinite_dropped": n_bad,
        "feat": feat,
        "raw": raw,
        "delta_accuracy": {k: feat[k]["accuracy"] - raw[k]["accuracy"] for k in feat},
        "delta_macro_f1": {k: feat[k]["macro_f1"] - raw[k]["macro_f1"] for k in feat},
    }

    for k in (str(x) for x in KS):
        f, r = feat[k], raw[k]
        print(f"  k={k:>2}  acc feat {f['accuracy']:.4f} (raw {r['accuracy']:.4f}, "
              f"delta {f['accuracy']-r['accuracy']:+.4f}, vs majority "
              f"{f['accuracy']-chance['majority_fraction']:+.4f})  macro-F1 feat "
              f"{f['macro_f1']:.4f} (raw {r['macro_f1']:.4f}, uniform "
              f"{chance['uniform']['macro_f1']:.4f})  purity {f['purity']:.4f}")
    print(f"  [{time.time()-t0:.0f}s]")

    entry["event_knn"] = metric
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features", nargs="+", help="feature .npz file(s)")
    ap.add_argument("--out", default="event_probe.json", help="output JSON path")
    ap.add_argument("--source", default="student", choices=["student", "teacher"])
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds the pixel sample and the uniform guess")
    ap.add_argument("--max_pixels_per_event", type=int,
                    default=DEFAULT_MAX_PIXELS_PER_EVENT,
                    help=f"pixels pooled per event (default {DEFAULT_MAX_PIXELS_PER_EVENT})")
    ap.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    args = ap.parse_args()

    results = {}
    for p in args.features:
        path = Path(p).resolve()
        if not path.exists():
            print(f"[skip] {path}: not found")
            continue
        results[run_label(path, args.source)] = run_one(path, args)
        write_json(results, args.out)

    write_json(results, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
