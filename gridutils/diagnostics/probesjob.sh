#!/bin/bash
#
# Probe-suite job script — runs the frozen-feature probes for one epoch.
# Called by submit_probes.sh; runs on a Condor worker.
#
# The probes read an extracted features .npz and never import warpconvnet or a
# dataset reader, so this job asks for no GPU (see submit_probes.sh).
#
# Args (positional):
#   $1 codedir    -- path to ml-dune-model repo root
#   $2 pyenv      -- path to uv virtual environment to activate
#   $3 run_name   -- run name (subdirectory under CONDOR_OUT)
#   $4 epoch      -- epoch number
#   $5 condor_out -- base CONDOR_OUT directory
#   $6 features   -- absolute path to the features .npz to score
#   $7+ extra_args -- forwarded to EVERY probe, so it may only carry flags all of
#                     them accept: --seed and --device. (--source is taken by all
#                     but probe_knn_pid, which scores student and teacher together.)
#
# A probe-specific knob (--pool_per_class, --train_per_class, --max_queries,
# --vertex_t0_ticks, ...) is not accepted by the others and would abort them, so
# pass those through the per-probe variables below. Each REPLACES extra_args for
# that one probe rather than adding to it:
#
#   PID_EXTRA_ARGS  KNN_EXTRA_ARGS  OVERLAP_EXTRA_ARGS
#   INSTANCE_EXTRA_ARGS  VERTEX_EXTRA_ARGS  EVENT_EXTRA_ARGS  EMBED_EXTRA_ARGS
#
# Output: ${condor_out}/${run_name}/probes/{pid,pixelknn,overlap,instance,vertex,
# event}_ep<N>.json, plus a merged table.txt over every JSON accumulated there.
# The `embed` stage instead writes PNGs and cached 2-D points to probes/ep<N>/.
#
# PROBE_STAGES selects which probes run, comma-separated; default is every
# measurement, which is all of them but `embed`.
#   PROBE_STAGES=pid                      -> PID only
#   PROBE_STAGES=overlap,instance,vertex  -> the per-pixel metrics only
#   PROBE_STAGES=embed                    -> the t-SNE/UMAP pictures only
# `merge` always runs (non-fatally), over whatever result JSONs are in the directory.

set -euo pipefail

codedir=$1
pyenv=$2
run_name=$3
epoch=$4
condor_out=$5
features=$6
shift 6
extra_args="${*}"

echo "Running $CLUSTER_ID.$JOB_ID on $(hostname)"
echo ""
echo "JOB CONFIGURATION:"
echo "  codedir=${codedir}"
echo "  run_name=${run_name}"
echo "  epoch=${epoch}"
echo "  features=${features}"
echo "  extra_args=${extra_args}"
echo ""

if [[ ! -f "$features" ]]; then
    echo "ERROR: features file not found: $features"
    exit 1
fi

echo "Activating python environment..."
source "${pyenv}/bin/activate"

OUT_DIR="${condor_out}/${run_name}/probes"
mkdir -p "$OUT_DIR"

PROBE_STAGES="${PROBE_STAGES:-pid,knn,overlap,instance,vertex,event}"
echo "Stages: ${PROBE_STAGES}"

# Pin the BLAS/torch thread pools to the CPUs condor actually allocated.
# submit_probes.sh forwards PROBE_THREADS=request_cpus so the two always agree.
# Without this, torch and OpenBLAS size their pools from the *machine's* core
# count and oversubscribe a 1-CPU slot, which is slower than single-threaded and
# steals cycles from other jobs on the node.
PROBE_THREADS="${PROBE_THREADS:-1}"
export OMP_NUM_THREADS="$PROBE_THREADS"
export MKL_NUM_THREADS="$PROBE_THREADS"
export OPENBLAS_NUM_THREADS="$PROBE_THREADS"
export NUMEXPR_NUM_THREADS="$PROBE_THREADS"
echo "Threads: ${PROBE_THREADS} (OMP/MKL/OpenBLAS/numexpr)"

export PYTHONPATH="$codedir${PYTHONPATH:+:$PYTHONPATH}"

echo "=== Probes for run: ${run_name}  epoch: ${epoch} ==="
echo "    Output dir: ${OUT_DIR}"
echo ""

# One process for every stage. The runner owns the per-probe flags and reads the
# same *_EXTRA_ARGS / KNN_MAX_PER_CLASS variables this script used to expand, so
# submit_probes.sh forwards them unchanged.
#
# Why one process: each probe calls `load_features`, and as separate invocations
# they each read and inflate the whole ~7 GB .npz off GPFS. That repetition, not
# the arithmetic, is what the job spends its time on -- a four-stage job measured
# 8 minutes of user CPU against 87 minutes of system time. In one process the
# first stage pays for the read and the rest reuse the cached object.
#
# Held non-fatal so `merge` still runs over whatever completed: the runner keeps
# going after a failed stage and reports which ones failed in its exit status,
# which this script re-raises at the end.
probe_status=0
python -u -m probes.run_probes "$features" \
    --out_dir="$OUT_DIR" \
    --epoch="$epoch" \
    --stages="$PROBE_STAGES" \
    $extra_args || probe_status=$?
echo ""


# Merge every result JSON in the directory, so the table grows into a trajectory
# as more epochs land. Written via a per-job temp then renamed: several epochs are
# usually in flight at once, and concurrent writers on one path would interleave.
# Last finisher wins, which is what we want — it sees the most epochs.
#
# nullglob so a stage that was never run drops out of the list instead of handing
# merge a literal `*.json` pattern that does not exist.
#
# NON-FATAL, deliberately. This stage produces no data — every number in the table
# is read back out of the per-probe JSONs, which are already on disk and are the
# real result. Letting a rendering step fail the job would throw away hours of
# completed probe work and force a rerun that regenerates identical files.
echo "--- merge (all epochs present so far) ---"
shopt -s nullglob
result_files=(
    "${OUT_DIR}"/pid_ep*.json
    "${OUT_DIR}"/pixelknn_ep*.json
    "${OUT_DIR}"/overlap_ep*.json
    "${OUT_DIR}"/instance_ep*.json
    "${OUT_DIR}"/vertex_ep*.json
    "${OUT_DIR}"/event_ep*.json
)
shopt -u nullglob

if [ ${#result_files[@]} -eq 0 ]; then
    echo "no result JSONs in ${OUT_DIR}; nothing to merge"
else
    tmp_txt="${OUT_DIR}/.table.${epoch}.$$.txt"
    tmp_csv="${OUT_DIR}/.table.${epoch}.$$.csv"
    if python -u -m probes.merge "${result_files[@]}" \
            --csv="$tmp_csv" | tee "$tmp_txt"; then
        mv -f "$tmp_txt" "${OUT_DIR}/table.txt"
        mv -f "$tmp_csv" "${OUT_DIR}/table.csv"
    else
        echo "[warn] merge failed; the probe JSONs above are unaffected and the"
        echo "       table can be rebuilt at any time with:"
        echo "         python -m probes.merge ${OUT_DIR}/*_ep*.json --csv table.csv"
        rm -f "$tmp_txt" "$tmp_csv"
    fi
fi
echo ""

# The merge above is deliberately non-fatal, but a failed probe is not: exiting
# non-zero is what stops a sweep from looking successful while missing metrics.
if [ "$probe_status" -ne 0 ]; then
    echo "=== Probes FAILED (see the runner's summary above) ==="
    exit "$probe_status"
fi

echo "=== Probes complete! ==="
