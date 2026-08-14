"""Score several probes against one feature file in a single process.

Each probe used to run as its own `python -m probes.probe_*` invocation, so a
six-stage sweep read and inflated the same ~7 GB .npz six times. The file lives
on GPFS and `np.savez_compressed` is read back through `zipfile` in small
chunks, so those repeated reads dominated: a four-stage job measured 8 minutes
of user CPU against 87 minutes of system time.

This runner calls each probe's own `main()` in one interpreter. `load_features`
caches per (file, source), so the first stage pays for the read and the rest get
the in-memory object. Probes treat `Features` as read-only, so sharing it is
safe.

Every stage runs even if an earlier one fails, and the exit status is non-zero
if any failed -- a sweep should not lose five completed measurements because the
sixth aborted, nor report success when it is missing metrics.

Usage:
  python -m probes.run_probes <features.npz> --out_dir=DIR --epoch=N \
      [--stages=pid,knn,overlap,instance,vertex,event] [common args...]

Per-probe knobs come from the environment, matching what submit_probes.sh
forwards: PID_EXTRA_ARGS, KNN_EXTRA_ARGS, OVERLAP_EXTRA_ARGS,
INSTANCE_EXTRA_ARGS, VERTEX_EXTRA_ARGS, EVENT_EXTRA_ARGS, EMBED_EXTRA_ARGS and
KNN_MAX_PER_CLASS. Each *_EXTRA_ARGS REPLACES the common args for that probe
rather than adding to them.
"""

import argparse
import importlib
import os
import shlex
import sys
import time
import traceback

# stage -> (module, output template, uses the common extra args as its fallback)
#
# `knn` and `embed` take no fallback on purpose: the common args are documented
# as carrying --device, which probe_knn_pid is given explicitly below and
# plot_embedding does not accept at all -- either would abort the stage.
STAGES = {
    "pid":      ("probes.probe_pid",       "pid_ep{epoch}.json",      True),
    "knn":      ("probes.probe_knn_pid",   "pixelknn_ep{epoch}.json", False),
    "overlap":  ("probes.probe_overlap",   "overlap_ep{epoch}.json",  True),
    "instance": ("probes.probe_instance",  "instance_ep{epoch}.json", True),
    "vertex":   ("probes.probe_vertex",    "vertex_ep{epoch}.json",   True),
    "event":    ("probes.probe_event",     "event_ep{epoch}.json",    True),
    "embed":    ("probes.plot_embedding",  "ep{epoch}",               False),
}

STAGE_TITLES = {
    "pid":      "probe_pid (particle type, trained head)",
    "knn":      "probe_knn_pid (particle type, untrained k-NN)",
    "overlap":  "probe_overlap (is a pixel's charge shared?)",
    "instance": "probe_instance (do neighbours share a particle?)",
    "vertex":   "probe_vertex (is a pixel near the interaction point?)",
    "event":    "probe_event (interaction flavor, pooled k-NN)",
    "embed":    "plot_embedding (2-D map, pixel PID + event flavor)",
}

DEFAULT_STAGES = "pid,knn,overlap,instance,vertex,event"


def stage_argv(stage, features, out_dir, epoch, common):
    """The argv a stage's own main() would have received from probesjob.sh."""
    module, out_template, takes_common = STAGES[stage]
    out = os.path.join(out_dir, out_template.format(epoch=epoch))
    env_extra = os.environ.get(f"{stage.upper()}_EXTRA_ARGS", "")

    argv = [features]
    if stage == "embed":
        argv += ["--mode=both", f"--out_dir={out}"]
    else:
        argv += [f"--out={out}"]

    if stage == "knn":
        # All-pairs over the pool, so cost grows quadratically and this job has
        # no GPU. Raise KNN_MAX_PER_CLASS only where --device=cuda is available.
        argv += [f"--max_pixels_per_class={os.environ.get('KNN_MAX_PER_CLASS', '10000')}",
                 "--device=cpu"]

    if env_extra:
        argv += shlex.split(env_extra)
    elif takes_common:
        argv += common
    return module, argv


def run_stage(stage, features, out_dir, epoch, common):
    """Call one probe's main() in this process. Returns True when it succeeded."""
    module_name, argv = stage_argv(stage, features, out_dir, epoch, common)
    # Reported separately because it sits OUTSIDE the `total for <stage>` timer
    # below: a cold sklearn+torch import off the GPFS venv measured 64 s (cluster
    # 1697), and only the first stage of a job pays it.
    t_imp = time.time()
    module = importlib.import_module(module_name)
    t_imp = time.time() - t_imp
    if t_imp >= 1.0:
        print(f"  [import {module_name} {t_imp:.0f}s]", flush=True)

    saved = sys.argv
    sys.argv = [module_name.rsplit(".", 1)[-1]] + argv
    t0 = time.time()
    try:
        module.main()
        return True
    except SystemExit as exc:
        # Probes raise SystemExit for a missing truth channel or a bad flag, and
        # argparse does too. A zero code still means success.
        if not exc.code:
            return True
        print(f"  [{stage} FAILED] {exc}", flush=True)
        return False
    except Exception:
        traceback.print_exc()
        print(f"  [{stage} FAILED] see traceback above", flush=True)
        return False
    finally:
        sys.argv = saved
        print(f"  [{time.time() - t0:.0f}s total for {stage}]\n", flush=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features", help="extract_features .npz to score")
    ap.add_argument("--out_dir", required=True, help="directory for the result JSONs")
    ap.add_argument("--epoch", required=True, help="epoch number, used in output names")
    ap.add_argument("--stages", default=DEFAULT_STAGES,
                    help=f"comma-separated subset of {','.join(STAGES)} "
                         f"(default: {DEFAULT_STAGES})")
    args, common = ap.parse_known_args()

    requested = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in requested if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s) {unknown}; known: {sorted(STAGES)}")

    # Fixed order regardless of how --stages was written, so a sweep's output
    # reads the same way every time.
    selected = [s for s in STAGES if s in requested]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Runner: {len(selected)} stage(s) in one process: {','.join(selected)}")
    if common:
        print(f"Common args: {' '.join(common)}")
    print("")

    failed = []
    for i, stage in enumerate(selected, 1):
        print(f"--- [{i}/{len(selected)}] {STAGE_TITLES[stage]} ---")
        if not run_stage(stage, args.features, args.out_dir, args.epoch, common):
            failed.append(stage)

    if failed:
        print(f"=== {len(selected) - len(failed)}/{len(selected)} stages succeeded; "
              f"FAILED: {','.join(failed)} ===")
        return 1
    print(f"=== all {len(selected)} stage(s) succeeded ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
