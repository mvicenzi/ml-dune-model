"""Merge probe result JSONs into one table, across checkpoints and trainings.

Every probe writes JSON keyed by `<run>:<epoch tag>:<feature source>`, so result
files from different epochs, different runs and different objectives merge into
one table without any bookkeeping: point this at as many JSONs as you like.

  # one training's trajectory over epochs
  python -m probes.merge CONDOR_OUT/mae_baseline_mixed_b100/probes/*.json

  # several trainings side by side, saved for a doc or a plot
  python -m probes.merge run_a/probes.json run_b/probes.json --csv table.csv

Rows sort by run then epoch. Missing metrics show as "-" rather than breaking
the table, so a partial run still tabulates.

This produces **no new data**: every number in the table is read straight out of
the JSONs, which are the durable result. The table is a view, rebuildable in
seconds at any time — which is why the condor job treats this step as non-fatal.

Plotting lives in `probes/plot_probes.py`, which reads the CSV this writes rather
than the JSONs, so the figures and the table can never disagree.
"""

import argparse
import csv
import json
import re
from pathlib import Path

# (column, dotted path into the entry, decimals). Deltas are feat - raw: the
# part of the score the representation is responsible for. A tuple of paths is
# tried in order, which is how results recorded under superseded key names still
# tabulate alongside current ones.
COLUMNS = [
    ("events",     "n_events",                          0),
    ("pid_svm",    ("pid.svm_feat", "pid7.svm_feat"),   4),
    ("pid_mlp",    ("pid.mlp_feat", "pid7.mlp_feat"),   4),
    ("pid_raw",    ("pid.mlp_raw", "pid.mlp_base",
                    "pid7.mlp_raw"),                    4),
    ("d_pid",      ("pid.delta_mlp", "pid7.delta_mlp"), 4),
    ("pid_miou",   "pid.miou_mlp_feat",                 4),
    # Overlap: F1 on the contaminated call at the headline threshold (0.2).
    # Alternative paths (the tuple form above) are for RENAMED keys only. Never
    # point a column at two different measurements — an MAE and an F1 answer
    # different questions, and a column that silently holds either is unreadable.
    # Same rule for instance and vertex below.
    ("ov_f1",      "overlap.f1_mlp_feat",               4),
    ("ov_f1_raw",  "overlap.f1_mlp_raw",                4),
    ("d_ov_f1",    "overlap.delta_f1_mlp",              4),
    # Instance: macro over particle-size bins of the margin over chance. NOT an
    # accuracy — chance runs from 0.00 in the small bins to 0.72 in the largest,
    # so the raw accuracy is not comparable across bins and the pooled figure
    # answers the opposite question to the per-bin evidence.
    ("inst_mgn",   "instance.macro_margin_feat",        4),
    ("inst_mgn_raw", "instance.macro_margin_raw",       4),
    ("d_inst_mgn", "instance.delta_macro_margin",       4),
    # Vertex: F1 on the near/far call at the headline radius (20 px).
    ("vtx_f1",     "vertex.f1_mlp_feat",                4),
    ("vtx_f1_raw", "vertex.f1_mlp_raw",                 4),
    ("d_vtx_f1",   "vertex.delta_f1_mlp",               4),
    ("knn10",      "feat.10.accuracy",                  4),
    ("d_knn10",    "delta_accuracy.10",                 4),
    ("knn10_f1",   "feat.10.macro_f1",                  4),
    # Non-parametric pixel k-NN (probes.probe_knn_pid). Not leakage-free —
    # a relative tracking curve, not comparable with the trained-head columns.
    ("knnpix",     "knn_pixel.overall_accuracy",        4),
    ("knnpix_f1",  "knn_pixel.macro_f1",                4),
]


# (display name, the namespaces a probe writes under, the column-path prefixes
# that read it). Tells "this probe did not run" apart from "this probe ran and
# COLUMNS does not match what it writes" — the second is silent otherwise.
# Several namespaces are listed where a metric has been written under more than
# one, so an older file (`pid7`) still counts as "the probe ran", not as a gap.
METRIC_PROBES = [
    ("PID",           ("pid", "pid7"),     ("pid.", "pid7.")),
    ("overlap",       ("overlap",),        ("overlap.",)),
    ("instance",      ("instance",),       ("instance.",)),
    ("vertex",        ("vertex",),         ("vertex.",)),
    ("event flavor",  ("delta_accuracy",), ("feat.", "delta_accuracy.")),
    ("kNN PID",       ("knn_pixel",),      ("knn_pixel.",)),
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
    """Warn when merged entries were scored against different raw-charge inputs.

    A `log10_1p` raw-charge input (feature files extracted before charge-
    transform provenance existed) and a `trained` one differ, so their
    feat-raw deltas cannot be compared even for the same checkpoint. Merging is
    still allowed — the feat columns remain meaningful — but silently tabulating
    both would defeat the point of the table.
    """
    kinds = {}
    for label, entry in merged.items():
        k = (entry.get("raw_charge_transform")
             or entry.get("input_baseline_transform")
             or entry.get("raw_floor_transform"))
        if k:
            kinds.setdefault(k, []).append(label)
    if len(kinds) > 1:
        print("[warn] mixed raw-charge transforms in this table — the raw and "
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

    # The threshold metrics define their positive class with a parameter, so two
    # entries scored at different settings are answering different questions and
    # their F1 columns line up in the table while meaning different things.
    for path, what in (("overlap.headline_threshold", "overlap threshold"),
                       ("vertex.headline_radius_px", "vertex radius (px)"),
                       ("vertex.t0_ticks_assumed", "vertex t0 (ticks)")):
        vals = {}
        for label, entry in merged.items():
            v = dig(entry, path)
            if v is not None:
                vals.setdefault(float(v), []).append(label)
        if len(vals) > 1:
            print(f"[warn] mixed {what}: "
                  + ", ".join(f"{v:g} ({len(ls)} run(s))"
                              for v, ls in sorted(vals.items()))
                  + " — these rows define the positive class differently.\n")


def report_coverage(merged: dict, active) -> None:
    """Name any metric present in the files that produced no column.

    The table keeps only populated columns, so a probe whose result keys do not
    match COLUMNS vanishes from it entirely rather than showing "-", which is
    indistinguishable from never having run the probe. Naming the two cases apart
    is what stops a metric going quietly unreported.
    """
    paths = []
    for _, path, _ in active:
        paths.extend(path if isinstance(path, tuple) else [path])

    missing, mismatched = [], []
    for label, namespaces, prefixes in METRIC_PROBES:
        present = any(ns in e for e in merged.values() for ns in namespaces)
        shown = any(p.startswith(prefixes) for p in paths)
        if present and not shown:
            mismatched.append(label)
        elif not present:
            missing.append(label)

    if mismatched:
        print("[warn] results present but no column matched, so these metrics are "
              "NOT in the table below: " + ", ".join(mismatched))
        print("       COLUMNS in probes/merge.py is out of date with what the "
              "probe writes.\n")
    if missing:
        print(f"[note] no results for: {', '.join(missing)}\n")


def build_rows(merged: dict):
    """Keep only columns that at least one run populated."""
    active = [(name, path, nd) for name, path, nd in COLUMNS
              if any(dig(e, path) is not None for e in merged.values())]
    report_coverage(merged, active)
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


if __name__ == "__main__":
    main()
