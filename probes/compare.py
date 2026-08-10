"""Tabulate probe results across checkpoints and across trainings.

Every probe writes JSON keyed by `<run>:<epoch tag>:<feature source>`, so result
files from different epochs, different runs and different objectives merge into
one table without any bookkeeping: point this at as many JSONs as you like
(pixel_probe and event_probe outputs mix freely, since they share the key scheme).

  # one training's trajectory over epochs
  python -m probes.compare CONDOR_OUT/mae_baseline_mixed_b100/probes/*.json

  # several trainings side by side, saved for a doc or a plot
  python -m probes.compare run_a/probes.json run_b/probes.json --csv table.csv

  # PID curves over epochs, with the input baseline and chance drawn in
  python -m probes.compare '.../probes/pid_ep*.json' --plot figures/

Rows sort by run then epoch. Missing metrics show as "-" rather than breaking
the table, so a partial run still tabulates.
"""

import argparse
import csv
import json
import re
from pathlib import Path

# (column, dotted path into the entry, decimals). Deltas are feat - base: the
# part of the score the representation is responsible for. A tuple of paths is
# tried in order, which is how results recorded under superseded key names still
# tabulate alongside current ones.
COLUMNS = [
    ("events",     "n_events",                          0),
    ("pid_svm",    ("pid.svm_feat", "pid7.svm_feat"),   4),
    ("pid_mlp",    ("pid.mlp_feat", "pid7.mlp_feat"),   4),
    ("pid_base",   ("pid.mlp_base", "pid.mlp_raw",
                    "pid7.mlp_raw"),                    4),
    ("d_pid",      ("pid.delta_mlp", "pid7.delta_mlp"), 4),
    ("pid_miou",   "pid.miou_mlp_feat",                 4),
    ("ov_mae",     "overlap.feat_mae",                  4),
    ("ov_mae_base", ("overlap.base_mae", "overlap.raw_mae"),   4),
    ("d_ov_mae",   "overlap.delta_mae",                 4),
    ("inst",       "instance.feat_inst_all",            4),
    ("inst_base",  ("instance.base_inst_all",
                    "instance.raw_inst_all"),           4),
    ("d_inst",     "instance.delta_inst",               4),
    ("inst_top1",  "instance.feat_top1_all",            4),
    ("d_inst_top1", "instance.delta_top1",              4),
    ("vtx_ap",     "vertex.feat_ap",                    4),
    ("vtx_ap_base", ("vertex.base_ap", "vertex.raw_ap"),  4),
    ("d_vtx_ap",   "vertex.delta_ap",                   4),
    ("knn10",      "feat.10.accuracy",                  4),
    ("d_knn10",    "delta_accuracy.10",                 4),
    ("knn10_f1",   "feat.10.macro_f1",                  4),
    # Non-parametric pixel k-NN (probes.probe_knn_pid). Not leakage-free —
    # a relative tracking curve, not comparable with the trained-head columns.
    ("knnpix",     "knn_pixel.overall_accuracy",        4),
    ("knnpix_f1",  "knn_pixel.macro_f1",                4),
]


def dig(entry: dict, path):
    """Follow a dotted path; None if any step is missing or not a mapping.

    `path` may be a tuple of alternatives, tried in order — used for keys that
    were renamed, so old and new result files merge into one table.
    """
    if isinstance(path, tuple):
        for alt in path:
            v = dig(entry, alt)
            if v is not None:
                return v
        return None
    cur = entry
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def epoch_of(label: str) -> int:
    """Sort key from the epoch tag (`ep100` -> 100); -1 when unparseable."""
    m = re.search(r"ep(\d+)", label)
    return int(m.group(1)) if m else -1


def load_all(paths) -> dict:
    """Merge result files. Later files win per (label, metric), so re-running one
    metric into a new JSON updates the table without discarding the others."""
    merged = {}
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"[skip] {path}: not found")
            continue
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"[skip] {path}: not a probe result JSON")
            continue
        for label, entry in data.items():
            if isinstance(entry, dict):
                merged.setdefault(label, {}).update(entry)
    return merged


def check_comparability(merged: dict) -> None:
    """Warn when merged entries were scored against different input baselines.

    A `log10_1p` baseline (feature files extracted before charge-transform
    provenance existed) and a `trained` one are different baselines, so their
    feat-base deltas cannot be compared even for the same checkpoint. Merging is
    still allowed — the feat columns remain meaningful — but silently tabulating
    both would defeat the point of the table.
    """
    kinds = {}
    for label, entry in merged.items():
        k = entry.get("input_baseline_transform") or entry.get("raw_floor_transform")
        if k:
            kinds.setdefault(k, []).append(label)
    if len(kinds) > 1:
        print("[warn] mixed input-baseline transforms in this table — the base and "
              "delta columns are NOT comparable across these groups:")
        for k, labels in sorted(kinds.items()):
            print(f"         {k}: {len(labels)} run(s), e.g. {labels[0]}")
        print("       Re-extract the older features to put everything on the "
              "trained transform.\n")

    # Pool size changes what a balanced score means, so a table mixing them is
    # comparing two different measurements.
    pools = {}
    for label, entry in merged.items():
        p = entry.get("pool_per_class")
        if p:
            pools.setdefault(int(p), []).append(label)
    if len(pools) > 1:
        print("[warn] mixed pool_per_class in this table: "
              + ", ".join(f"{p} ({len(v)} run(s))" for p, v in sorted(pools.items()))
              + " — balanced scores are not comparable across these.\n")

    # And so does the number of events behind the extraction.
    sizes = {}
    for label, entry in merged.items():
        n = entry.get("n_events")
        if n:
            sizes.setdefault(int(n), []).append(label)
    if len(sizes) > 1:
        print("[warn] mixed extraction sizes (n_events): "
              + ", ".join(f"{n} ({len(v)} run(s))" for n, v in sorted(sizes.items()))
              + " — rare types rest on very different event counts.\n")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

# Okabe-Ito, a colourblind-safe categorical set, assigned in this fixed order to
# runs — never cycled by rank, so adding a run cannot repaint the others. Yellow
# (#F0E442) is deliberately absent: too low-contrast on a white surface.
RUN_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
              "#000000"]

# One encoding per question: colour = which run, linestyle = features or the
# input baseline, panel = which head.
FEAT_STYLE = dict(linestyle="-", marker="o", markersize=4, linewidth=1.8)
BASE_STYLE = dict(linestyle="--", marker="s", markersize=3.5, linewidth=1.3,
                  alpha=0.75)


def series_of(merged: dict, path):
    """{(run, source): [(epoch, value), ...]} for one dotted path, epoch-sorted.

    Entries whose epoch tag does not parse are dropped: a curve needs an x.
    """
    out = {}
    for label, entry in merged.items():
        v = dig(entry, path)
        ep = epoch_of(label)
        if v is None or ep < 0:
            continue
        parts = label.split(":")
        out.setdefault((parts[0], parts[-1]), []).append((ep, float(v)))
    return {k: sorted(v) for k, v in sorted(out.items())}


def _curve_label(key, n_sources: int) -> str:
    run, source = key
    return f"{run} ({source})" if n_sources > 1 else run


def _draw_panel(ax, merged, feat_path, base_path, chance_path, colors,
                n_sources: int) -> bool:
    """One panel: feature curve, input-baseline curve, chance line. True if drawn."""
    feat = series_of(merged, feat_path)
    if not feat:
        return False
    base = series_of(merged, base_path) if base_path else {}
    for key, pts in feat.items():
        c = colors[key]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c,
                label=_curve_label(key, n_sources), **FEAT_STYLE)
        if key in base:
            bp = base[key]
            ax.plot([p[0] for p in bp], [p[1] for p in bp], color=c, **BASE_STYLE)
    # Chance is a property of the scored population, so it is measured per
    # checkpoint rather than assumed; they agree to within sampling, so one line
    # (the mean) is drawn and its spread reported in the label if it moves.
    if chance_path:
        vals = [v for pts in series_of(merged, chance_path).values()
                for _, v in pts]
        if vals:
            lo, hi = min(vals), max(vals)
            spread = f" ±{(hi - lo) / 2:.3f}" if hi - lo > 5e-4 else ""
            ax.axhline(sum(vals) / len(vals), color="#666666", linestyle=":",
                       linewidth=1.2, label=f"chance{spread}")
    ax.grid(alpha=0.3, linewidth=0.6)
    return True


def _outer_labels(axes, xlabel: str, ylabel: str, n_used: int = None) -> None:
    """Label only the outer edge of a shared-axis grid.

    Repeating the label on every panel of a `sharex`/`sharey` grid puts an axis
    title under panels whose tick labels are suppressed, which reads as a broken
    axis. The x label goes on the lowest *used* panel of each column, so it stays
    correct when the last row is partly switched off.
    """
    import numpy as _np

    grid = _np.atleast_2d(axes)
    nrow, ncol = grid.shape
    used = nrow * ncol if n_used is None else n_used
    for r in range(nrow):
        grid[r, 0].set_ylabel(ylabel)
    for c in range(ncol):
        rows = [r for r in range(nrow) if r * ncol + c < used]
        if rows:
            grid[max(rows), c].set_xlabel(xlabel)


def make_plots(merged: dict, out_dir: Path) -> None:
    """PID curves over epochs: score, the input baseline, chance, and per type.

    Only PID is plotted. The other metrics are still being settled (docs 12/13),
    and a plot that outlives the metric it draws is worse than no plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    if not series_of(merged, "pid.mlp_feat"):
        print("[skip] --plot: no PID results in these files")
        return

    runs = sorted({(l.split(":")[0], l.split(":")[-1]) for l in merged})
    colors = {k: RUN_COLORS[i % len(RUN_COLORS)] for i, k in enumerate(runs)}
    n_sources = len({k[1] for k in runs})
    written = []

    def finish(fig, name, note=""):
        fig.tight_layout()
        path = out_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(name + (f"  ({note})" if note else ""))

    # 1. Macro-F1 and macro-IoU per head on the balanced pool. There is only one
    # population: the natural-prevalence block was removed once measurement showed
    # it added nothing beyond a fixed prior-mismatch artefact (see probe_pid).
    specs = []
    for head in ("svm", "mlp"):
        specs.append((f"{head} macro-F1", f"pid.{head}_feat",
                      (f"pid.{head}_base", f"pid.{head}_raw"),
                      "pid.chance.uniform.m_f1"))
        specs.append((f"{head} macro-IoU", f"pid.miou_{head}_feat",
                      (f"pid.miou_{head}_base", f"pid.miou_{head}_raw"), None))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    any_drawn = False
    for ax, (title, fp, bp, cp) in zip(axes.ravel(), specs):
        drawn = _draw_panel(ax, merged, fp, bp, cp, colors, n_sources)
        any_drawn |= drawn
        ax.set_title(title, fontsize=11)
        if not drawn:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="#999999")
    if any_drawn:
        _outer_labels(axes, "epoch", "score")
        axes.ravel()[0].legend(fontsize=8)
        fig.suptitle("PID readout vs epoch — balanced pool\n"
                     "solid = frozen features, dashed = input baseline "
                     "(channel, tick, log charge), dotted = uniform guess",
                     fontsize=12)
        finish(fig, "pid_balanced.png")
    else:
        plt.close(fig)

    # 3. The delta. Its own figure because it is the number that answers "what did
    # the backbone add", and because it is far less seed-sensitive than either
    # absolute curve (the two sides share a sample).
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    drew_delta = False
    for ax, head in zip(axes, ("svm", "mlp")):
        d = series_of(merged, (f"pid.delta_{head}", f"pid7.delta_{head}"))
        for key, pts in d.items():
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=colors[key], label=_curve_label(key, n_sources),
                    **FEAT_STYLE)
            drew_delta = True
        ax.axhline(0.0, color="#666666", linewidth=1.2)
        ax.set_title(f"{head}: macro-F1 (features) − macro-F1 (input baseline)",
                     fontsize=11)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3, linewidth=0.6)
    if drew_delta:
        axes[0].set_ylabel("delta")
        axes[0].legend(fontsize=8)
        fig.suptitle("What the backbone added, over the input baseline "
                     "(above zero = the features help)", fontsize=12)
        finish(fig, "pid_delta.png")
    else:
        plt.close(fig)

    # 4. Per type. The macro average hides the whole story here: the rare motifs
    # are where every reference model collapses, and they are what moves the mean.
    types = None
    for entry in merged.values():
        types = dig(entry, "pid.headline_classes") or types
    if types:
        ncol = 3
        nrow = (len(types) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow),
                                 sharex=True, sharey=True)
        flat = axes.ravel() if hasattr(axes, "ravel") else [axes]
        for ax, t in zip(flat, types):
            _draw_panel(ax, merged,
                        f"pid.per_class_f1.mlp_feat.{t}",
                        (f"pid.per_class_f1.mlp_base.{t}",
                         f"pid.per_class_f1.mlp_raw.{t}"),
                        None, colors, n_sources)
            ax.set_title(t, fontsize=11)
        for ax in flat[len(types):]:
            ax.axis("off")
        _outer_labels(axes, "epoch", "F1", n_used=len(types))
        flat[0].legend(fontsize=8)
        fig.suptitle("Per-type F1 vs epoch, MLP head, balanced pool "
                     "(solid = features, dashed = input baseline)", fontsize=12)
        finish(fig, "pid_per_class.png",
               "check the event counts behind the rare types")

    for name in written:
        print(f"  saved {out_dir / name.split(' ')[0]}"
              + (f"   {name.split('  ', 1)[1]}" if "  " in name else ""))


def build_rows(merged: dict):
    """Keep only columns that at least one run populated."""
    active = [(name, path, nd) for name, path, nd in COLUMNS
              if any(dig(e, path) is not None for e in merged.values())]
    labels = sorted(merged, key=lambda s: (s.split(":")[0], epoch_of(s), s))
    rows = []
    for label in labels:
        entry = merged[label]
        row = {"run": label}
        for name, path, nd in active:
            v = dig(entry, path)
            row[name] = "-" if v is None else (f"{v:.{nd}f}" if nd else f"{int(v)}")
        rows.append(row)
    return ["run"] + [c[0] for c in active], rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="probe result JSON file(s)")
    ap.add_argument("--csv", default="", help="also write the table as CSV")
    ap.add_argument("--markdown", action="store_true",
                    help="emit a markdown table (pipe-separated) instead of aligned text")
    ap.add_argument("--plot", default="",
                    help="directory to write PID curves to (score, input "
                         "baseline, chance, per type)")
    args = ap.parse_args()

    merged = load_all(args.results)
    if not merged:
        raise SystemExit("no results found")
    check_comparability(merged)
    header, rows = build_rows(merged)

    widths = {c: max(len(c), *(len(r[c]) for r in rows)) for c in header}
    if args.markdown:
        print("| " + " | ".join(c.ljust(widths[c]) for c in header) + " |")
        print("|" + "|".join("-" * (widths[c] + 2) for c in header) + "|")
        for r in rows:
            print("| " + " | ".join(r[c].ljust(widths[c]) for c in header) + " |")
    else:
        print("  ".join(c.ljust(widths[c]) for c in header))
        for r in rows:
            print("  ".join(r[c].ljust(widths[c]) for c in header))

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}")

    if args.plot:
        print()
        make_plots(merged, Path(args.plot))


if __name__ == "__main__":
    main()
