"""Overlap: can a readout tell that a pixel's charge is shared?

A pixel's `pixel_energyfrac` is its leading contributor's share of the deposited
energy, so `overlap = 1 - pixel_energyfrac` is contamination: 0 when one particle
owns the pixel outright, 0.5 when two contribute equally. The probe asks a yes/no
question about that number — is this pixel more contaminated than `t`? — and
scores it exactly like PID: two fixed heads, the raw-charge inputs as the number
to beat, and the degenerate answers measured on the same pixels.

The threshold is swept rather than chosen. `t = 0.2` is the headline, but 0.1 and
0.3 are scored alongside it, so a reader can see whether the answer depends on
where the line was drawn. Each threshold is its own task with its own prevalence
and its own chance level, all of which are reported.

Scored on the **natural** validation population, not a balanced one. That is what
doc 12 rule 3 asks for, and unlike PID it is available here: the contaminated
class is 14% of truth pixels, not 0.6%, so precision is not a prior artefact. It
also closes a confound — contamination rates differ 4x across particle types
(Blip 8.7%, DeltaRay 35.4%), but every type is minority-contaminated, so on the
natural population a head that knows only the particle type scores F1 = 0. On a
balanced pool it would not, and some of the score would be PID leaking in.

Only pixels carrying truth take part; contamination is undefined without it.

Usage:
  python -m probes.probe_overlap FEATURES.npz [MORE.npz ...] --out overlap.json
"""

import argparse
import time
from pathlib import Path

import numpy as np

from probes.class_balancing import balanced_pool, pixel_split
from probes.features import Features, load_features, raw_charge
from probes.linear_heads import fit_mlp, fit_svm
from probes.probe_pid import PID_HEADLINE, PID_NAMES
from probes.results import run_header, run_label, write_json

# Swept thresholds on `overlap`; the first is the headline. Contamination above
# ~0.5 is rare (1.6% of truth pixels), so the sweep stays in the range where both
# classes are well populated.
THRESHOLDS = (0.2, 0.1, 0.3)

DEFAULT_TRAIN_PER_CLASS = 20_000
# The validation population is the natural one, drawn uniformly so prevalence is
# preserved. Capped only to bound memory: all validation truth pixels would be
# ~8.4 M rows, and a float32 copy of those is 2.1 GB on top of the feature array.
DEFAULT_VAL_PIXELS = 2_000_000


def truth_mask(fx: Features) -> np.ndarray:
    """Pixels carrying truth: label != 0 (0 is Background/no-truth)."""
    return fx.truth["pixel_labels"].astype(np.int64) != 0


def overlap_score(fx: Features) -> np.ndarray:
    """Contamination per pixel, 0 = pure.

    `pixel_energyfrac` is the leading contributor's share of the pixel's energy,
    so `1 - frac` is what the other contributors deposited. Pixels without truth
    have no meaningful value and are forced to 0; callers must mask them out
    rather than read that 0 as "pure".
    """
    ov = 1.0 - fx.truth["pixel_energyfrac"].astype(np.float64)
    ov[~truth_mask(fx)] = 0.0
    return np.clip(ov, 0.0, 1.0)


def binary_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Precision, recall and F1 for the positive (contaminated) class.

    Degenerate cases score 0.0 rather than NaN: a head that never predicts the
    positive class has no precision to speak of, and 0 is the honest reading of
    it — the same convention `probe_pid` uses per class.
    """
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1),
            "accuracy": float(np.mean(y_pred == y_true)),
            "predicted_positive_rate": float(np.mean(y_pred == 1))}


def trivial_scores(y_true: np.ndarray, seed: int) -> dict:
    """What the degenerate answers score on this exact population.

    `majority` is always "pure" here, since contamination is the minority at every
    swept threshold — so it scores F1 = 0 by construction, and its accuracy is
    just the prevalence restated. `uniform` guesses at random. Both are the same
    arithmetic as the real scores.
    """
    rng = np.random.RandomState(seed + 7)
    maj = int(np.mean(y_true) > 0.5)
    return {
        "majority": {"predicts": "contaminated" if maj else "pure",
                     **binary_scores(y_true, np.full(len(y_true), maj))},
        "uniform": binary_scores(y_true, rng.randint(0, 2, len(y_true))),
    }


def _fit_and_score(Xtr, ytr, Xva, yva, seed: int, device: str):
    """Both fixed heads on one (train, val) pair. Returns (scores, mlp predictions).

    The predictions come back so the per-type breakdown can slice the very
    predictions the headline scored, instead of refitting an identical head.
    """
    from sklearn.preprocessing import StandardScaler

    # Fitted on the training rows only — the same step every other head gets.
    # Without it the raw baseline's channel/tick columns of magnitude 1e3 go
    # straight into Adam at lr 5e-3 and the head collapses to a constant.
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)
    out = {"svm": binary_scores(yva, fit_svm(Xtr_s, ytr, Xva_s, seed))}
    pred, _ = fit_mlp(Xtr_s, ytr, Xva_s, n_classes=2, seed=seed, device=device)
    out["mlp"] = binary_scores(yva, pred)
    return out, pred


def overlap_metric(fx: Features, raw: np.ndarray, is_train: np.ndarray, seed: int,
                   per_class: int, val_pixels: int, device: str) -> dict:
    """Score the contaminated/pure call at every swept threshold."""
    ov = overlap_score(fx)
    tm = truth_mask(fx)
    pid = fx.truth["pixel_labels"].astype(np.int64)

    # One validation population for every threshold: drawn uniformly from the
    # validation truth pixels, so it carries the natural prevalence, and shared
    # across thresholds so the sweep compares tasks rather than samples.
    va_all = np.where(~is_train & tm)[0]
    if len(va_all) > val_pixels:
        va_all = np.random.RandomState(seed + 1).choice(va_all, val_pixels,
                                                        replace=False)
    tr_cand = np.where(is_train & tm)[0]
    if len(tr_cand) == 0 or len(va_all) == 0:
        return {"error": "no truth pixels on one side of the split"}

    res = {"thresholds": [float(t) for t in THRESHOLDS],
           "headline_threshold": float(THRESHOLDS[0]),
           "n_truth_pixels": int(tm.sum()),
           "train_per_class": int(per_class),
           "n_val": int(len(va_all)),
           "val_population": "natural"}

    Xva = {"feat": np.asarray(fx.feat[va_all], dtype=np.float32),
           "raw": np.asarray(raw[va_all], dtype=np.float32)}

    sweep = {}
    headline_pred = headline_y = None
    for t in THRESHOLDS:
        yva = (ov[va_all] > t).astype(np.int64)
        y_all = (ov > t).astype(np.int64)
        # Balance the TRAINING pool only (rule 3): the head must not learn the
        # prior, but the score must face the real one.
        tr = balanced_pool(tr_cand, y_all, [0, 1], per_class, seed)
        ytr = y_all[tr]
        entry = {"prevalence_val": float(yva.mean()),
                 "n_train": int(len(tr)),
                 "train_counts": {"pure": int((ytr == 0).sum()),
                                  "contaminated": int((ytr == 1).sum())},
                 "chance": trivial_scores(yva, seed)}
        for src in ("feat", "raw"):
            X = fx.feat if src == "feat" else raw
            heads, pred = _fit_and_score(np.asarray(X[tr], dtype=np.float32), ytr,
                                         Xva[src], yva, seed, device)
            for head, vals in heads.items():
                entry[f"{head}_{src}"] = vals
            if t == THRESHOLDS[0] and src == "feat":
                headline_pred, headline_y = pred, yva
        for head in ("svm", "mlp"):
            entry[f"delta_f1_{head}"] = (entry[f"{head}_feat"]["f1"]
                                         - entry[f"{head}_raw"]["f1"])
        sweep[f"{t:g}"] = entry
    res["sweep"] = sweep

    # Headline, flat, so `compare` can reach it without knowing the sweep shape.
    head_t = f"{THRESHOLDS[0]:g}"
    h = sweep[head_t]
    for head in ("svm", "mlp"):
        res[f"f1_{head}_feat"] = h[f"{head}_feat"]["f1"]
        res[f"f1_{head}_raw"] = h[f"{head}_raw"]["f1"]
        res[f"delta_f1_{head}"] = h[f"delta_f1_{head}"]
    res["recall_mlp_feat"] = h["mlp_feat"]["recall"]
    res["precision_mlp_feat"] = h["mlp_feat"]["precision"]
    res["prevalence_val"] = h["prevalence_val"]

    # Per particle type, at the headline threshold. The base rate is what makes
    # these readable: recall at a 35% prior (DeltaRay) means something different
    # from recall at 9% (Blip).
    res["per_type"] = _per_type(pid[va_all], headline_y, headline_pred)
    return res


def _per_type(pid_va: np.ndarray, yva: np.ndarray, pred: np.ndarray) -> dict:
    """Recall and F1 within each particle type, at the headline threshold.

    Refits nothing: this slices the very predictions the headline scored, so it
    cannot disagree with it. Types with too few validation pixels report only
    their count, rather than a number resting on a handful of rows.
    """
    out = {}
    for c in PID_HEADLINE:
        m = pid_va == c
        entry = {"n": int(m.sum())}
        if m.sum() >= 500:
            entry["base_rate"] = float(yva[m].mean())
            entry.update(binary_scores(yva[m], pred[m]))
        out[PID_NAMES[c]] = entry
    return out


def run_one(path: Path, args) -> dict:
    fx = load_features(path, source=args.source)
    fx.require("pixel_labels", "pixel_energyfrac")
    print(f"\n=== {run_label(path, args.source)} ===")

    is_train = pixel_split(fx, args.seed)
    entry = run_header(fx, args.seed, args.train_per_class)
    entry.pop("seed", None)
    entry.pop("pool_per_class", None)

    t0 = time.time()
    m = overlap_metric(fx, raw_charge(fx), is_train, args.seed,
                       args.train_per_class, args.val_pixels, args.device)
    m["seed"] = int(args.seed)
    entry["overlap"] = m
    if "error" in m:
        print(f"  [error] {m['error']}")
        return entry

    print(f"  truth pixels={m['n_truth_pixels']}  scored={m['n_val']} (natural)")
    for t in (f"{x:g}" for x in THRESHOLDS):
        s = m["sweep"][t]
        print(f"  t={t:>4}  prevalence {s['prevalence_val']:.4f}  "
              f"F1 mlp {s['mlp_feat']['f1']:.4f} (raw {s['mlp_raw']['f1']:.4f}, "
              f"delta {s['delta_f1_mlp']:+.4f})  svm {s['svm_feat']['f1']:.4f}  "
              f"recall {s['mlp_feat']['recall']:.4f}  "
              f"precision {s['mlp_feat']['precision']:.4f}")
    print(f"  [{time.time()-t0:.0f}s]")
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features", nargs="+", help="feature .npz file(s)")
    ap.add_argument("--out", default="overlap.json", help="output JSON path")
    ap.add_argument("--source", default="student", choices=["student", "teacher"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train_per_class", type=int, default=DEFAULT_TRAIN_PER_CLASS,
                    help=f"balanced training pixels per class "
                         f"(default {DEFAULT_TRAIN_PER_CLASS})")
    ap.add_argument("--val_pixels", type=int, default=DEFAULT_VAL_PIXELS,
                    help=f"validation pixels, drawn uniformly so the natural "
                         f"prevalence is kept (default {DEFAULT_VAL_PIXELS})")
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
