"""Plot probe trajectories over epochs, from the CSV `probes.merge` writes.

The headline curves come from the table, so the figures show exactly the numbers
the table shows and there is one place a column is defined: add it to
`merge.COLUMNS` and it becomes plottable here.

Some things cannot live in a flat table — per-class F1 is seven numbers per head
per epoch, and each metric's chance level sits inside a nested sweep. Pass the
result JSONs with `--json` and those are read straight from them, as reference
lines on the trajectory panels and as a second per-class figure. Rows are matched
to entries by the `<run>:<epoch tag>:<source>` label the CSV already carries, so
the two sources cannot drift apart.

  python -m probes.merge .../probes/*_ep*.json --csv table.csv
  python -m probes.plot_probes table.csv --out_dir figures/
  python -m probes.plot_probes table.csv --json '.../probes/*_ep*.json' \
      --out_dir figures/

One panel per metric, one colour per run, solid for the frozen features, dashed
for the raw charge inputs, dotted for chance. Panels whose columns are absent
from the CSV are skipped, so a partial suite still plots.
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

from probes.merge import load_all

# Okabe-Ito, a colourblind-safe categorical set, assigned in this fixed order to
# runs — never cycled by rank, so adding a run cannot repaint the others. Yellow
# (#F0E442) is deliberately absent: too low-contrast on a white surface.
RUN_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
              "#000000"]

FEAT_STYLE = dict(linestyle="-", marker="o", markersize=4, linewidth=1.8)
RAW_STYLE = dict(linestyle="--", marker="s", markersize=3.5, linewidth=1.3,
                 alpha=0.75)


def walk(entry, keys):
    """Follow an explicit list of keys; None if any step is missing.

    A list rather than `merge.dig`'s dotted string because the sweep keys are
    numbers: overlap's headline threshold is the literal key "0.2", which a
    dotted path would split into "0" and "2".
    """
    cur = entry
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def _chance_pid(entry):
    return walk(entry, ["pid", "chance", "uniform", "m_f1"])


def _chance_overlap(entry):
    t = walk(entry, ["overlap", "headline_threshold"])
    return None if t is None else walk(
        entry, ["overlap", "sweep", f"{t:g}", "chance", "uniform", "f1"])


def _chance_vertex(entry):
    r = walk(entry, ["vertex", "headline_radius_px"])
    return None if r is None else walk(
        entry, ["vertex", "sweep", f"{r:g}", "chance", "f1"])


def _chance_event(entry):
    return walk(entry, ["chance"])


# (title, feature column, raw column or None, chance accessor or None). A margin
# panel takes `zero` because the chance level is already subtracted out of the
# quantity — the reference a reader needs there is 0, not a measured rate.
PANELS = [
    ("PID macro-F1 (MLP)",        "pid_mlp",    "pid_raw",       _chance_pid),
    ("PID macro-F1 (SVM)",        "pid_svm",    None,            _chance_pid),
    ("PID macro-IoU (MLP)",       "pid_miou",   None,            None),
    ("Overlap F1 (t = 0.2)",      "ov_f1",      "ov_f1_raw",     _chance_overlap),
    ("Instance macro margin",     "inst_mgn",   "inst_mgn_raw",  "zero"),
    ("Vertex F1 (r = 20 px)",     "vtx_f1",     "vtx_f1_raw",    _chance_vertex),
    ("Event flavor acc (k = 10)", "knn10",      None,            _chance_event),
    ("kNN PID accuracy",          "knnpix",     None,            None),
]


def epoch_of(label: str) -> int:
    """Epoch from the run label (`...:ep100:student` -> 100); -1 if unparseable."""
    m = re.search(r"ep(\d+)", label)
    return int(m.group(1)) if m else -1


def read_table(path: Path):
    """(rows, columns) from the merge CSV. Values stay strings; "-" means absent."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def series(rows, column: str):
    """{(run, source): [(epoch, value), ...]} for one column, epoch-sorted."""
    out = defaultdict(list)
    for r in rows:
        raw = r.get(column, "-")
        ep = epoch_of(r["run"])
        if raw in ("-", "", None) or ep < 0:
            continue
        parts = r["run"].split(":")
        out[(parts[0], parts[-1])].append((ep, float(raw)))
    return {k: sorted(v) for k, v in sorted(out.items())}


def json_series(merged, accessor):
    """{(run, source): [(epoch, value), ...]} pulled from the result JSONs."""
    out = defaultdict(list)
    for label, entry in merged.items():
        ep = epoch_of(label)
        v = accessor(entry)
        if v is None or ep < 0:
            continue
        parts = label.split(":")
        out[(parts[0], parts[-1])].append((ep, float(v)))
    return {k: sorted(v) for k, v in sorted(out.items())}


def _run_keys(labels):
    return sorted({(l.split(":")[0], l.split(":")[-1]) for l in labels
                   if epoch_of(l) >= 0})


def _draw_chance(ax, values):
    """One dotted line for chance, with its spread if it moves across epochs.

    Chance is a property of the scored population rather than of the model, so it
    is measured per checkpoint and should agree to within sampling. Drawing the
    mean and reporting the spread makes a disagreement visible instead of hiding
    it behind one of several lines.
    """
    if not values:
        return
    lo, hi = min(values), max(values)
    spread = f" ±{(hi - lo) / 2:.3f}" if hi - lo > 5e-4 else ""
    ax.axhline(sum(values) / len(values), color="#666666", linestyle=":",
               linewidth=1.2, label=f"chance{spread}")


def plot_trajectories(rows, columns, merged, out_dir: Path, stem: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [p for p in PANELS if p[1] in columns and series(rows, p[1])]
    if not panels:
        print("[skip] none of the plottable columns are populated in this CSV")
        return

    runs = _run_keys([r["run"] for r in rows])
    colors = {k: RUN_COLORS[i % len(RUN_COLORS)] for i, k in enumerate(runs)}
    n_sources = len({k[1] for k in runs})

    ncol = 2 if len(panels) > 1 else 1
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.0 * ncol, 3.6 * nrow),
                             sharex=True, squeeze=False)
    flat = axes.ravel()

    for ax, (title, feat_col, raw_col, chance) in zip(flat, panels):
        feat = series(rows, feat_col)
        raw = series(rows, raw_col) if raw_col and raw_col in columns else {}
        for key, pts in feat.items():
            run, source = key
            label = f"{run} ({source})" if n_sources > 1 else run
            ax.plot([p[0] for p in pts], [p[1] for p in pts],
                    color=colors[key], label=label, **FEAT_STYLE)
            if key in raw:
                ax.plot([p[0] for p in raw[key]], [p[1] for p in raw[key]],
                        color=colors[key], **RAW_STYLE)
        if chance == "zero":
            ax.axhline(0.0, color="#666666", linestyle=":", linewidth=1.2,
                       label="chance (subtracted)")
        elif chance is not None and merged:
            _draw_chance(ax, [v for pts in json_series(merged, chance).values()
                              for _, v in pts])
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.3, linewidth=0.6)
        ax.set_ylabel("score")

    for ax in flat[len(panels):]:
        ax.axis("off")

    # Label x only on the lowest *used* panel of each column: on a sharex grid the
    # others have their tick labels suppressed, and a title under them reads as a
    # broken axis.
    for c in range(ncol):
        used = [r for r in range(nrow) if r * ncol + c < len(panels)]
        if used:
            axes[max(used)][c].set_xlabel("epoch")

    flat[0].legend(fontsize=8)
    fig.suptitle("Probe scores vs epoch — solid = frozen features, "
                 "dashed = raw charge (channel, tick, log charge), "
                 "dotted = chance", fontsize=12)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}  ({len(panels)} panels, {len(runs)} run(s))")


def plot_pid_per_class(merged, out_dir: Path, stem: str) -> None:
    """Per-type PID F1 over epochs. JSON only — seven numbers per head per epoch
    is not something a flat table should carry.

    The macro average hides the whole story here: the rare motifs are where a
    representation collapses, and they are what moves the mean.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    types = None
    for entry in merged.values():
        types = walk(entry, ["pid", "headline_classes"]) or types
    if not types:
        return

    runs = _run_keys(merged)
    colors = {k: RUN_COLORS[i % len(RUN_COLORS)] for i, k in enumerate(runs)}
    n_sources = len({k[1] for k in runs})

    ncol = 3
    nrow = (len(types) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    flat = axes.ravel()
    drew = False

    for ax, t in zip(flat, types):
        for src, style in (("mlp_feat", FEAT_STYLE), ("mlp_raw", RAW_STYLE)):
            s = json_series(merged,
                            lambda e, t=t, src=src: walk(
                                e, ["pid", "per_class_f1", src, t]))
            for key, pts in s.items():
                run, source = key
                label = (f"{run} ({source})" if n_sources > 1 else run) \
                    if src == "mlp_feat" else None
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        color=colors[key], label=label, **style)
                drew = True
        ax.set_title(t, fontsize=11)
        ax.grid(alpha=0.3, linewidth=0.6)

    if not drew:
        plt.close(fig)
        return

    for ax in flat[len(types):]:
        ax.axis("off")
    for r in range(nrow):
        axes[r][0].set_ylabel("F1")
    for c in range(ncol):
        used = [r for r in range(nrow) if r * ncol + c < len(types)]
        if used:
            axes[max(used)][c].set_xlabel("epoch")
    flat[0].legend(fontsize=8)
    fig.suptitle("Per-type PID F1 vs epoch, MLP head, balanced pool "
                 "(solid = features, dashed = raw charge)", fontsize=12)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}  ({len(types)} types)")


def plot_knn_pid(merged, out_dir: Path, stem: str) -> None:
    """kNN-PID per-class recall and the confusion matrix, from the JSON.

    One row per entry: with several epochs or runs in the merge this draws each
    of them, which is the point — a confusion matrix is read by comparing it with
    another one.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    have = [(l, e["knn_pixel"]) for l, e in sorted(merged.items())
            if isinstance(e.get("knn_pixel"), dict)
            and e["knn_pixel"].get("confusion")]
    if not have:
        return

    fig, axes = plt.subplots(len(have), 2, figsize=(11, 4.2 * len(have)),
                             squeeze=False)
    for row, (label, kp) in enumerate(have):
        classes = kp.get("classes") or []
        # Per-class recall, next to the macro and overall figures.
        ax = axes[row][0]
        acc = kp.get("per_class_accuracy") or {}
        vals = [acc.get(c) for c in classes]
        ok = [(c, v) for c, v in zip(classes, vals) if v is not None]
        if ok:
            ax.bar([c for c, _ in ok], [v for _, v in ok], color="#0072B2")
            for i, (_, v) in enumerate(ok):
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        ax.axhline(kp.get("overall_accuracy", 0.0), color="#666666",
                   linestyle=":", linewidth=1.2,
                   label=f"overall {kp.get('overall_accuracy', float('nan')):.3f}")
        ax.set_ylabel("recall")
        ax.set_title(f"{label}\nper-class recall  (macro-F1 "
                     f"{kp.get('macro_f1', float('nan')):.3f}, k={kp.get('knn_k')})",
                     fontsize=10)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)

        # Row-normalised confusion: each row is "of the pixels really this class,
        # where did they go", which is what makes the diagonal comparable across
        # classes drawn from pools of different size.
        ax = axes[row][1]
        cm = kp["confusion"]
        norm = []
        for r in cm:
            tot = sum(r)
            norm.append([v / tot if tot else 0.0 for v in r])
        im = ax.imshow(norm, cmap="viridis", vmin=0.0, vmax=1.0)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title("confusion (row-normalised)", fontsize=10)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}  ({len(have)} entrie(s))")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="table CSV written by `python -m probes.merge --csv`")
    ap.add_argument("--json", nargs="*", default=[],
                    help="result JSONs, for the chance lines and the per-class "
                         "figure — the parts a flat table cannot carry")
    ap.add_argument("--out_dir", default="figures",
                    help="directory to write the PNGs to (default: figures)")
    ap.add_argument("--name", default="probe_trajectories",
                    help="output file stem (default: probe_trajectories)")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"{path}: not found")
    rows, columns = read_table(path)
    if not rows:
        raise SystemExit(f"{path}: no rows")
    print(f"{len(rows)} row(s), {len(columns)} column(s) from {path}")

    merged = load_all(args.json) if args.json else {}
    if args.json:
        print(f"{len(merged)} entrie(s) from {len(args.json)} JSON path(s) "
              f"for chance and per-class")
    else:
        print("no --json given: chance lines and the per-class figure are skipped")

    plot_trajectories(rows, columns, merged, Path(args.out_dir), args.name)
    if merged:
        plot_pid_per_class(merged, Path(args.out_dir), "pid_per_class")
        plot_knn_pid(merged, Path(args.out_dir), "knn_pid")


if __name__ == "__main__":
    main()
