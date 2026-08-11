"""Instance: do a pixel's neighbours belong to the same particle?

For each pixel, look at its `--knn_k` nearest neighbours in cosine feature space
*within its own event*, take the most common particle among them, and ask whether
that is the pixel's own particle. Scored as the fraction of pixels where it is —
a prediction compared with truth, the same shape as PID and kNN-PID. Nothing is
trained.

Averaged **per pixel**, not per instance. That is a deliberate break from the
fork, which averaged per instance: half of all instances here have 3 pixels or
fewer, so a per-instance average is dominated by particles that hold 2% of the
charge, and it also averaged over a different population than it sampled. Per
pixel, the population sampled and the population averaged are the same one.

The cost of that choice is that big particles dominate — 85% of truthed pixels
belong to particles of 100+ pixels — so the headline mostly describes tracks and
showers. The per-size breakdown is what keeps the small ones visible, exactly as
PID's per-class table sits under its macro average.

Instance ids are per-event labels, so unlike PID there is no shared label space
and no confusion matrix to build: `abs(pixel_trackid) == 7` in one event has
nothing to do with the same value in another. Only the accuracy is poolable.

Usage:
  python -m probes.probe_instance FEATURES.npz [MORE.npz ...] --out instance.json
"""

import argparse
import time
from pathlib import Path

import numpy as np

from probes.features import Features, load_features, raw_charge
from probes.results import run_header, run_label, write_json

# Bins over the size of a pixel's OWN particle. Singletons get their own bin
# because they are the one group that cannot be scored at all.
SIZE_BINS = ((1, 1), (2, 3), (4, 9), (10, 99), (100, 999), (1000, 10 ** 9))
SIZE_BIN_NAMES = ("1", "2-3", "4-9", "10-99", "100-999", "1000+")

DEFAULT_KNN_K = 5
# Query pixels drawn globally, not per event: a per-event cap would give a
# 50 000-pixel event the same say as a 500-pixel one, which is the per-instance
# weighting bug in another costume. Drawn uniformly, so each event contributes in
# proportion to its size and the average really is "a randomly chosen pixel".
DEFAULT_MAX_QUERIES = 100_000


def instance_truth(fx: Features):
    """(instance id per pixel, mask of pixels that carry instance truth).

    `pixel_trackid` is 0 where there is no truth, and `abs()` would fuse every
    such pixel into one enormous pseudo-instance whose members all neighbour each
    other — straight inflation. Those pixels are excluded, not relabelled.
    """
    tid = fx.truth["pixel_trackid"]
    lab = fx.truth["pixel_labels"].astype(np.int64)
    return np.abs(tid.astype(np.int64)), (lab != 0) & (tid != 0)


def majority_vote(ids: np.ndarray):
    """Most common value per row of `ids` [n, k], with its count.

    Plurality, not a strict majority — the same rule `probe_pid`'s kNN and
    `probe_event` use (`argmax` over votes). Ties break towards the smaller id
    via the sort, which is arbitrary but deterministic, so two checkpoints scored
    on the same pool break them identically.
    """
    s = np.sort(ids, axis=1)
    best = s[:, 0].copy()
    best_n = np.ones(len(s), dtype=np.int64)
    cur, cnt = s[:, 0].copy(), np.ones(len(s), dtype=np.int64)
    for j in range(1, s.shape[1]):
        same = s[:, j] == cur
        cnt = np.where(same, cnt + 1, 1)
        cur = s[:, j]
        upd = cnt > best_n
        best = np.where(upd, cur, best)
        best_n = np.where(upd, cnt, best_n)
    return best, best_n


def _random_neighbours(n: int, pos: np.ndarray, k: int, rng) -> np.ndarray:
    """k neighbour indices per query, drawn uniformly from the event, self excluded.

    This is the chance level, and it is not small: a particle that owns half its
    event is the plurality among random neighbours most of the time. Without it a
    big-particle score cannot be read at all. Drawn with replacement across the k,
    which is immaterial except in events barely larger than k.
    """
    draw = rng.randint(0, n - 1, size=(len(pos), k))
    return draw + (draw >= pos[:, None])          # shift past self


def _event_predictions(feats_by_src, inst_e, pos, k, device, rng):
    """Plurality instance prediction for one event's queries, per source."""
    import torch

    dev = torch.device(device)
    qk = torch.from_numpy(pos).long().to(dev)
    out = {}
    for src, X in feats_by_src.items():
        fe = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(dev)
        fe = fe / fe.norm(dim=1, keepdim=True).clamp(min=1e-8)
        sims = fe[qk] @ fe.T
        sims[torch.arange(len(pos), device=dev), qk] = -2.0     # never your own row
        nn = sims.topk(k, dim=1).indices.cpu().numpy()
        out[src], _ = majority_vote(inst_e[nn])
    out["chance"], _ = majority_vote(
        inst_e[_random_neighbours(len(inst_e), pos, k, rng)])
    return out


def instance_metric(fx: Features, raw: np.ndarray, seed: int, k: int,
                    max_queries: int, device: str) -> dict:
    """Per-pixel majority-vote instance accuracy, overall and per particle size."""
    inst, tm = instance_truth(fx)
    truth_idx = np.where(tm)[0]
    if len(truth_idx) == 0:
        return {"error": "no pixels carry instance truth"}

    rng = np.random.RandomState(seed)
    queries = (truth_idx if len(truth_idx) <= max_queries
               else np.sort(rng.choice(truth_idx, max_queries, replace=False)))

    ev_of_q = fx.pixel_event[queries]
    order = np.argsort(ev_of_q, kind="stable")
    queries, ev_of_q = queries[order], ev_of_q[order]
    bounds = np.searchsorted(ev_of_q, np.unique(ev_of_q), side="left")
    bounds = np.append(bounds, len(queries))

    SOURCES = ("feat", "raw", "chance")
    correct = {src: [] for src in SOURCES}
    sizes, n_events, n_skipped_q = [], 0, 0

    for i in range(len(bounds) - 1):
        q = queries[bounds[i]:bounds[i + 1]]
        ev = int(ev_of_q[bounds[i]])
        a, b = int(fx.offsets[ev]), int(fx.offsets[ev + 1])
        rows = a + np.where(tm[a:b])[0]          # truthed pixels of this event
        if len(rows) < k + 1:
            n_skipped_q += len(q)                # too few neighbours to vote
            continue
        inst_e = inst[rows]
        pos = np.searchsorted(rows, q)
        # Instance size within this event, for the pixel's own particle.
        uniq, counts = np.unique(inst_e, return_counts=True)
        size_of = dict(zip(uniq.tolist(), counts.tolist()))

        preds = _event_predictions(
            {"feat": fx.feat[rows], "raw": raw[rows]}, inst_e, pos, k, device, rng)
        truth_q = inst_e[pos]
        for src in SOURCES:
            correct[src].append(preds[src] == truth_q)
        sizes.append(np.array([size_of[v] for v in truth_q.tolist()]))
        n_events += 1

    if not sizes:
        return {"error": "no event had enough truthed pixels to vote"}

    sizes = np.concatenate(sizes)
    correct = {src: np.concatenate(v) for src, v in correct.items()}

    res = {"knn_k": int(k), "seed": int(seed),
           "n_queries_scored": int(len(sizes)),
           "n_queries_dropped_small_event": int(n_skipped_q),
           "n_events_used": int(n_events),
           "n_truth_pixels": int(tm.sum()),
           "averaged_over": "pixels"}

    # A pixel whose particle is a singleton has no mate that could be voted for,
    # so it is wrong by construction. That is the ceiling, and it is a property
    # of the truth, not of the features.
    can = sizes >= 2
    res["ceiling"] = float(can.mean())
    res["singleton_fraction"] = float((sizes == 1).mean())

    # Pooled over every scored pixel. Kept, but NOT the headline: chance runs from
    # 0.00 in the small bins to 0.72 in the largest, and 52% of pixels sit in that
    # largest bin, so the pooled figure is mostly chance and answers the opposite
    # question to the per-bin evidence (measured: pooled says the features beat the
    # raw charge by +0.016, the macro below says they lose by -0.057).
    pooled = {}
    for src in SOURCES:
        pooled[f"{src}_accuracy"] = float(correct[src].mean())
    pooled["delta_accuracy"] = pooled["feat_accuracy"] - pooled["raw_accuracy"]
    pooled["margin_feat"] = pooled["feat_accuracy"] - pooled["chance_accuracy"]
    res["pooled"] = pooled

    per_size = {}
    for (lo, hi), name in zip(SIZE_BINS, SIZE_BIN_NAMES):
        m = (sizes >= lo) & (sizes <= hi)
        e = {"n_pixels": int(m.sum()), "pixel_fraction": float(m.mean())}
        if m.sum():
            e["ceiling"] = float(can[m].mean())
            for src in SOURCES:
                e[f"{src}_accuracy"] = float(correct[src][m].mean())
            # The margin over chance is the readable quantity: a raw accuracy of
            # 0.79 is excellent in a bin where chance is 0.01 and unremarkable in
            # one where chance is 0.72.
            for src in ("feat", "raw"):
                e[f"{src}_margin"] = e[f"{src}_accuracy"] - e["chance_accuracy"]
            e["delta_margin"] = e["feat_margin"] - e["raw_margin"]
        per_size[name] = e
    res["per_size"] = per_size

    # Headline: equal weight per particle size, on the margin over chance. Bins
    # with a zero ceiling are excluded — a singleton has no mate that could be
    # voted for, so it contributes a guaranteed 0 that says nothing about the
    # representation. Same reasoning that keeps Background out of PID's macro-F1.
    used = [n for n in SIZE_BIN_NAMES
            if per_size[n].get("ceiling", 0.0) > 0.0 and per_size[n]["n_pixels"] >= 100]
    res["macro_bins_used"] = used
    if used:
        for src in ("feat", "raw"):
            res[f"macro_margin_{src}"] = float(
                np.mean([per_size[n][f"{src}_margin"] for n in used]))
        res["delta_macro_margin"] = res["macro_margin_feat"] - res["macro_margin_raw"]
    res["headline"] = "macro_margin_feat"
    return res


def run_one(path: Path, args) -> dict:
    fx = load_features(path, source=args.source)
    fx.require("pixel_labels", "pixel_trackid")
    print(f"\n=== {run_label(path, args.source)} ===")

    entry = run_header(fx, args.seed, 0)
    entry.pop("seed", None)
    entry.pop("pool_per_class", None)

    t0 = time.time()
    m = instance_metric(fx, raw_charge(fx), args.seed, args.knn_k,
                        args.max_queries, args.device)
    entry["instance"] = m
    if "error" in m:
        print(f"  [error] {m['error']}")
        return entry

    print(f"  queries={m['n_queries_scored']} over {m['n_events_used']} events  "
          f"k={m['knn_k']}  ceiling={m['ceiling']:.4f} "
          f"({100*m['singleton_fraction']:.2f}% singletons)")
    if "macro_margin_feat" in m:
        print(f"  HEADLINE macro margin over chance, equal weight per size "
              f"({'/'.join(m['macro_bins_used'])}):")
        print(f"    feat {m['macro_margin_feat']:+.4f}   "
              f"raw {m['macro_margin_raw']:+.4f}   "
              f"feat-raw {m['delta_macro_margin']:+.4f}")
    p = m["pooled"]
    print(f"  pooled over pixels (not the headline): feat {p['feat_accuracy']:.4f}  "
          f"raw {p['raw_accuracy']:.4f}  chance {p['chance_accuracy']:.4f}")
    print(f"    {'size':>8s} {'%px':>7s} {'chance':>8s} {'feat':>8s} {'raw':>8s}"
          f" {'feat-ch':>9s} {'raw-ch':>9s}")
    for name in SIZE_BIN_NAMES:
        b = m["per_size"][name]
        if "feat_accuracy" in b:
            print(f"    {name:>8s} {100*b['pixel_fraction']:6.2f}% "
                  f"{b['chance_accuracy']:8.4f} {b['feat_accuracy']:8.4f} "
                  f"{b['raw_accuracy']:8.4f} {b['feat_margin']:+9.4f} "
                  f"{b['raw_margin']:+9.4f}"
                  + ("" if b["ceiling"] > 0 else "   [ceiling 0, excluded]"))
    print(f"  [{time.time()-t0:.0f}s]")
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features", nargs="+", help="feature .npz file(s)")
    ap.add_argument("--out", default="instance.json", help="output JSON path")
    ap.add_argument("--source", default="student", choices=["student", "teacher"])
    ap.add_argument("--seed", type=int, default=0, help="seeds the query sample")
    ap.add_argument("--knn_k", type=int, default=DEFAULT_KNN_K,
                    help=f"neighbours voting on each pixel (default {DEFAULT_KNN_K})")
    ap.add_argument("--max_queries", type=int, default=DEFAULT_MAX_QUERIES,
                    help=f"query pixels, drawn uniformly over all events "
                         f"(default {DEFAULT_MAX_QUERIES})")
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
