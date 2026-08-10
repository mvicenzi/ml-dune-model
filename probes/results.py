"""Result-file conventions: how an entry is keyed, and what provenance it carries.

Every probe/metrics keys its JSON `<run>:<epoch tag>:<source>`.
This allows results files from different metrics, epochs and trainings to be merged into one table
with no bookkeeping.
"""

from pathlib import Path

from probes.features import Features, raw_charge_kind


def run_label(path, source: str) -> str:
    """`<run>:<epoch tag>:<source>`, e.g. `mae_baseline_mixed_b100:ep100:student`.

    Derived from the conventional layout
    `<CONDOR_OUT>/<run>/checkpoints/features_ep<N>.npz`.
    """
    path = Path(path)
    stem = path.stem
    tag = stem[len("features_"):] if stem.startswith("features_") else stem
    run = path.parents[1].name if len(path.parents) >= 2 else path.parent.name
    return f"{run}:{tag}:{source}"


def write_json(results: dict, out_path) -> None:
    """Write results incrementally, so a long multi-checkpoint run is crash-safe."""
    import json
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)


def run_header(fx: Features, seed: int, per_class: int) -> dict:
    """Provenance block recorded next to every metric in the output JSON."""
    return {
        "features_file": str(fx.path),
        "feature_source": fx.source,
        "n_events": fx.n_events,
        "n_pixels": fx.n_pixels,
        "feature_dim": int(fx.feat.shape[1]),
        "truth_channels": sorted(fx.truth),
        "seed": seed,
        "pool_per_class": per_class,
        "raw_charge_transform": raw_charge_kind(fx),
        "provenance": fx.provenance,
    }
