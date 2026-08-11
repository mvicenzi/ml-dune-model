"""Vertex: can a readout tell that a pixel sits close to the interaction point?

The 3D true vertex is projected into this view and every pixel gets its distance
to it in pixel space. The probe asks a yes/no question about that number — is this
pixel within `r` pixels of the vertex? — and scores it exactly like the overlap
probe: a fixed head, the raw-charge inputs as the number to beat, and the
degenerate answers measured on the same pixels.

The radius is swept rather than chosen. `r = 20 px` is the headline (the fork's
convention), but 10 and 30 are scored alongside it, so a reader can see whether
the answer depends on where the line was drawn. Each radius is its own task with
its own prevalence and its own chance level, all of which are reported.

Scored on the **natural** validation population, not a balanced one — near-vertex
pixels are ~5% of the image, so precision means something. Training is balanced,
because a head must not learn the prior; scoring is not, because the score must
face it. That is doc 12 rule 3.

Reported as precision / recall / F1, the same quantities as overlap and PID, so
one number means one thing across the suite. AP and AUROC survive only as
secondary keys: they are what the fork's `rich_pixel_probe.py` quoted, and doc 12
rule 4 keeps normalised scores off a headline.

The split is by *event* (`class_balancing.pixel_split`). A per-pixel split would
leak badly here — distance to the vertex is a smooth function of position within
an event, so pixels of the same event on both sides would hand either head that
event's position-to-distance map.

Every pixel takes part, truthed or not: distance to the vertex is geometry, and a
noise pixel beside the vertex is still beside the vertex. Needs `apa`/`view`
provenance so the projection knows which wire plane it is looking at. Runs on CPU;
--device cuda only speeds up the head.

Usage:
  python -m probes.probe_vertex FEATURES.npz [MORE.npz ...] --out vertex.json
"""

import argparse
import time
from pathlib import Path

import numpy as np

from probes.class_balancing import balanced_pool, pixel_split
from probes.features import Features, load_features, raw_charge
from probes.linear_heads import fit_mlp
from probes.probe_overlap import binary_scores
from probes.results import run_header, run_label, write_json

# Swept radii in pixels; the first is the headline. 20 px is the fork's
# convention, which is why it leads. Note channel pitch and tick spacing are
# different physical scales (~0.5 cm vs 0.321 cm), so each disc is anisotropic in
# cm — as the fork's was.
RADII_PX = (20.0, 10.0, 30.0)

DEFAULT_TRAIN_PER_CLASS = 50_000
# Validation pixels, drawn uniformly so the natural prevalence is preserved.
# Capped only to bound memory and head time.
DEFAULT_VAL_PIXELS = 200_000

# The drift-to-tick offset is a stated constant inherited from the fork, not zero
# and not a fit performed at run time.
#
# The projection is `tick = drift_cm / 0.321126 + t0`; the channel half needs no
# parameter. t0 absorbs the frame reference time, WireCell's response-plane
# offset, any tick cropping, and the pixel-bin indexing convention, so it is a
# property of how the images were made rather than a physical constant — which is
# why it is named here, overridable, and recorded in every result.
#
# The value is the fork's: its build_vertex_labels.py self-calibrates t0 per pack
# from muon endpoints and stores the result in each .vtx.npz sidecar; all 18 in
# its data_sdcc/ give -0.567 +- 0.007 (U -0.572..-0.578, V -0.559..-0.567,
# W -0.556..-0.564). Adopting it keeps one constant across both implementations
# and needs no genealogy tier our containers lack.
#
# Cross-checked, not assumed. Projecting charged-track 3D endpoints (mcpart
# start/end_xyzts joined to their footprint pixels through track_ids) against the
# pixels' actual tick, over 2116 measurements in 282 events of
# prod-jay-100k-truth-2026-06-11, gives -0.649 ticks, 95% CI [-0.729, -0.596]
# (event-clustered bootstrap). That is 0.08 tick = 0.026 cm = 0.08 px from the
# fork's value: a real difference in production timing (the CIs do not overlap)
# but far below anything this metric resolves — switching t0 by ten times as much
# moves only 0.089% of pixels across the 20 px radius. Our fit is what rules out
# the earlier t0=0 (which was wrong by 0.65 tick) and what makes the fork's number
# usable here rather than merely inherited.
#
# Most of the offset is a BINNING convention, not timing: a stored pixel tick is
# an integer bin index (dtype int32) whose centre sits at index+0.5, while the
# projection is continuous, so index-minus-continuous earns -0.5 for free. Both
# fits carry it (fork packs are int32 (channel,tick) too); refitting ours against
# bin centres leaves -0.149 [-0.229, -0.096], so the genuine frame/response-plane
# term is ~0.15 tick, ~0.07 on the fork's production. The value to USE is the
# index one, because the metric compares projected ticks against integer pixel
# coords.
#
# A charge-density scan (median distance from the projected vertex to the nearest
# charge pixel) is flat within +-2 ticks and rises beyond it, so that observable
# rules out multi-tick errors but cannot resolve this value — see probes/README.md.
#
# Override with --vertex_t0_ticks for a production that registers differently, or
# with -0.649 to use our own prod-jay fit instead. Vertex numbers recorded before
# 2026-08-05 were taken at t0=0 (README records the size of the difference).
DEFAULT_VERTEX_T0_TICKS = -0.567


def vertex_distance(fx: Features, t0_ticks: float):
    """Per-pixel distance to the projected true vertex, and which pixels have one.

    Projects each event's `vertex_xyz` to this view's (channel, tick) with
    loader/wire_geometry.py, then measures pixel-space distance. Events whose
    vertex falls outside the wire volume are excluded rather than clamped, so a
    mis-projected vertex drops out instead of piling every pixel at one edge.
    """
    from loader.wire_geometry import WireGeometry   # pure numpy, no warpconvnet

    prov = fx.provenance
    if "view" not in prov or "apa" not in prov:
        raise SystemExit(
            "vertex metric needs `apa`/`view` provenance in the features file; "
            "re-extract so they are recorded.")
    view, apa = str(prov["view"]), int(prov["apa"])

    geom = WireGeometry.load(t0_ticks=t0_ticks)
    ymin, ymax, zmin, zmax = geom.apa_bbox(apa)

    dist = np.full(fx.n_pixels, np.nan, dtype=np.float32)
    n_ok = n_outside = 0
    for ev in range(fx.n_events):
        a, b = int(fx.offsets[ev]), int(fx.offsets[ev + 1])
        if b == a:
            continue
        xyz = fx.vertex_xyz[ev]
        if not (ymin - 5 <= xyz[1] <= ymax + 5 and zmin - 5 <= xyz[2] <= zmax + 5
                and abs(xyz[0]) < 360.0):
            n_outside += 1
            continue
        _, u, v, w, tick = geom.project(xyz, apa=apa)
        ch = float(geom.channel_for_view(view, np.array([u, v, w])))
        pos = fx.positions[a:b]
        dist[a:b] = np.hypot(pos[:, 0].astype(np.float64) - ch,
                             pos[:, 1].astype(np.float64) - tick).astype(np.float32)
        n_ok += 1

    return dist, np.isfinite(dist), {"n_events_projected": n_ok,
                                     "n_events_vertex_outside_volume": n_outside}


def _fit_and_score(Xtr, ytr, Xva, yva, seed: int, device: str):
    """The fixed MLP head on one (train, val) pair, scored and ranked.

    Returns precision/recall/F1 plus AP and AUROC off the same probabilities, so
    the secondary ranking numbers cannot disagree with the headline — they are
    that head's own scores, thresholded one way and ranked the other.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import average_precision_score, roc_auc_score

    # Fitted on the training rows only. Without it the raw baseline's channel/tick
    # columns of magnitude 1e3 go straight into Adam at lr 5e-3 and the head
    # collapses to a constant — the failure doc 12 defect 6 records.
    sc = StandardScaler().fit(Xtr)
    pred, prob = fit_mlp(sc.transform(Xtr), ytr, sc.transform(Xva),
                         n_classes=2, seed=seed, device=device)
    out = binary_scores(yva, pred)
    if 0 < int(yva.sum()) < len(yva):
        out["ap"] = float(average_precision_score(yva, prob[:, 1]))
        out["auroc"] = float(roc_auc_score(yva, prob[:, 1]))
    return out


def vertex_metric(fx: Features, raw: np.ndarray, is_train: np.ndarray, seed: int,
                  t0_ticks: float, per_class: int, val_pixels: int,
                  device: str) -> dict:
    """Score the near/far call at every swept radius."""
    dist, valid, info = vertex_distance(fx, t0_ticks)
    if info["n_events_projected"] == 0:
        return {"error": "no event vertex projected inside the wire volume", **info}

    # One validation population for every radius: drawn uniformly from the
    # validation pixels, so it carries the natural prevalence, and shared across
    # radii so the sweep compares tasks rather than samples.
    va_all = np.where(~is_train & valid)[0]
    if len(va_all) > val_pixels:
        va_all = np.random.RandomState(seed + 1).choice(va_all, val_pixels,
                                                        replace=False)
    tr_cand = np.where(is_train & valid)[0]
    if len(tr_cand) == 0 or len(va_all) == 0:
        return {"error": "no projected pixels on one side of the split", **info}

    res = {"radii_px": [float(r) for r in RADII_PX],
           "headline_radius_px": float(RADII_PX[0]),
           "t0_ticks_assumed": float(t0_ticks),
           # Interaction vertex only. The fork's sidecar carries secondary
           # vertices too, which is why its prevalence is ~11% against our ~5%:
           # a denser task with a different chance level, so its AP is not ours.
           "vertex_kind": "interaction_only",
           "train_per_class": int(per_class),
           "n_val": int(len(va_all)),
           "val_population": "natural",
           "seed": int(seed), **info}

    Xva = {"feat": np.asarray(fx.feat[va_all], dtype=np.float32),
           "raw": np.asarray(raw[va_all], dtype=np.float32)}

    sweep = {}
    for r in RADII_PX:
        y_all = np.zeros(len(dist), dtype=np.int64)
        y_all[valid & (dist <= r)] = 1
        yva = y_all[va_all]
        # Balance the TRAINING pool only (rule 3): the head must not learn the
        # prior, but the score must face the real one.
        tr = balanced_pool(tr_cand, y_all, [0, 1], per_class, seed)
        ytr = y_all[tr]
        # A tighter radius has fewer near pixels to draw on, on both sides of the
        # split. Both counts are recorded because the sweep is only comparable
        # while every radius trains on a full pool and scores on enough positives
        # for precision to mean anything — at 20 px the near class is ~5% of the
        # image, and 10 px is ~4x thinner than that.
        entry = {"prevalence_val": float(yva.mean()),
                 "n_val_near": int(yva.sum()),
                 "n_train": int(len(tr)),
                 "train_counts": {"far": int((ytr == 0).sum()),
                                  "near": int((ytr == 1).sum())},
                 "train_pool_short": bool(min((ytr == 0).sum(),
                                              (ytr == 1).sum()) < per_class),
                 # A seeded coin flip on the same pixels. Not something the head
                 # is asked to beat — the raw charge inputs are what it is
                 # measured against — but a head that has collapsed to a constant
                 # cannot clear it, so it is what separates a weak score from a
                 # broken one. Same arithmetic as the real scores.
                 "chance": binary_scores(
                     yva, np.random.RandomState(seed + 7).randint(0, 2, len(yva)))}
        if ytr.min() == ytr.max() or yva.min() == yva.max():
            entry["error"] = "one class empty at this radius"
            sweep[f"{r:g}"] = entry
            continue
        for src in ("feat", "raw"):
            X = fx.feat if src == "feat" else raw
            entry[f"mlp_{src}"] = _fit_and_score(
                np.asarray(X[tr], dtype=np.float32), ytr, Xva[src], yva,
                seed, device)
        entry["delta_f1_mlp"] = entry["mlp_feat"]["f1"] - entry["mlp_raw"]["f1"]
        sweep[f"{r:g}"] = entry
    res["sweep"] = sweep

    # Headline, flat, so `compare` can reach it without knowing the sweep shape.
    # Same key names as probe_overlap, because it is the same call on a different
    # question and a reader should not have to learn two spellings.
    h = sweep[f"{RADII_PX[0]:g}"]
    if "error" not in h:
        res["f1_mlp_feat"] = h["mlp_feat"]["f1"]
        res["f1_mlp_raw"] = h["mlp_raw"]["f1"]
        res["delta_f1_mlp"] = h["delta_f1_mlp"]
        res["recall_mlp_feat"] = h["mlp_feat"]["recall"]
        res["precision_mlp_feat"] = h["mlp_feat"]["precision"]
        res["prevalence_val"] = h["prevalence_val"]
        # Secondary, not the headline: what the fork quoted (its best was AP
        # 0.492, on the denser taxonomy above). Kept so that line stays alive.
        res["fork_comparable"] = {
            "feat_ap": h["mlp_feat"].get("ap"), "raw_ap": h["mlp_raw"].get("ap"),
            "feat_auroc": h["mlp_feat"].get("auroc"),
            "raw_auroc": h["mlp_raw"].get("auroc")}
    return res


def run_one(path: Path, args) -> dict:
    fx = load_features(path, source=args.source)
    print(f"\n=== {run_label(path, args.source)} ===")

    is_train = pixel_split(fx, args.seed)
    entry = run_header(fx, args.seed, args.train_per_class)
    entry.pop("pool_per_class", None)
    print(f"  event split: {int(is_train[fx.offsets[:-1]].sum())}/{fx.n_events} events "
          f"train ({is_train.sum()}/{len(is_train)} pixels)")

    t0 = time.time()
    m = vertex_metric(fx, raw_charge(fx), is_train, args.seed,
                      args.vertex_t0_ticks, args.train_per_class,
                      args.val_pixels, args.device)
    entry["vertex"] = m
    if "error" in m:
        print(f"  [error] {m['error']}")
        return entry

    print(f"  scored={m['n_val']} pixels (natural)  "
          f"{m['n_events_projected']} events projected, "
          f"{m['n_events_vertex_outside_volume']} vertices outside volume")
    for r in (f"{x:g}" for x in RADII_PX):
        s = m["sweep"][r]
        if "error" in s:
            print(f"  r={r:>4} px  skipped: {s['error']}")
            continue
        f, w = s["mlp_feat"], s["mlp_raw"]
        print(f"  r={r:>4} px  near-vertex rate {s['prevalence_val']:.4f}  "
              f"F1 feat {f['f1']:.4f} (raw {w['f1']:.4f}, "
              f"delta {s['delta_f1_mlp']:+.4f})  "
              f"of those it called near, {100*f['precision']:.1f}% were; "
              f"of those that were, it found {100*f['recall']:.1f}%")
        short = ("  [training pool short: "
                 f"{s['train_counts']['near']} near vs {m['train_per_class']} asked]"
                 if s["train_pool_short"] else "")
        print(f"            {s['n_val_near']} near pixels scored;  "
              f"coin-flip F1 {s['chance']['f1']:.4f}{short}")
    print(f"  [t0={m['t0_ticks_assumed']} ticks, {time.time()-t0:.0f}s]")
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features", nargs="+",
                    help="feature .npz file(s) from probes.extract_features")
    ap.add_argument("--out", default="vertex.json", help="output JSON path")
    ap.add_argument("--source", default="student", choices=["student", "teacher"],
                    help="which feature set to score (default: student)")
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds the event split, the pools and the head init (default: 0)")
    ap.add_argument("--train_per_class", type=int, default=DEFAULT_TRAIN_PER_CLASS,
                    help=f"balanced training pixels per class "
                         f"(default {DEFAULT_TRAIN_PER_CLASS})")
    ap.add_argument("--val_pixels", type=int, default=DEFAULT_VAL_PIXELS,
                    help=f"validation pixels, drawn uniformly so the natural "
                         f"near-vertex rate is kept (default {DEFAULT_VAL_PIXELS})")
    ap.add_argument("--device", default="cpu",
                    help="torch device for the MLP head (default: cpu)")
    ap.add_argument("--vertex_t0_ticks", type=float, default=DEFAULT_VERTEX_T0_TICKS,
                    help=f"drift-to-tick offset for the vertex projection (default: "
                         f"{DEFAULT_VERTEX_T0_TICKS}, the fork's calibrated value; "
                         f"pass -0.649 for our own prod-jay fit -- see "
                         f"DEFAULT_VERTEX_T0_TICKS). Recorded in the output JSON.")
    args = ap.parse_args()

    results = {}
    for p in args.features:
        path = Path(p).resolve()
        if not path.exists():
            print(f"[skip] {path}: not found")
            continue
        results[run_label(path, args.source)] = run_one(path, args)
        write_json(results, args.out)   # incremental: a long run stays crash-safe

    write_json(results, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
