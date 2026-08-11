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
stage_on() { [[ ",${PROBE_STAGES}," == *",$1,"* ]]; }
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

if stage_on pid; then
    echo "--- [1/8] probe_pid (particle type, trained head) ---"
    python -u -m probes.probe_pid "$features" \
        --out="${OUT_DIR}/pid_ep${epoch}.json" \
        ${PID_EXTRA_ARGS:-$extra_args}
    echo ""
fi


# Pool size is modest on purpose: this k-NN is all-pairs over the pool, so cost
# grows quadratically and this job has no GPU. 10k/class (60k pixels) is a couple
# of minutes on 4 cores; the 50k/class default would be ~an hour. Raise
# KNN_MAX_PER_CLASS only when running somewhere with a GPU (--device=cuda).
if stage_on knn; then
    echo "--- [2/8] probe_knn_pid (particle type, untrained k-NN) ---"
    python -u -m probes.probe_knn_pid "$features" \
        --out="${OUT_DIR}/pixelknn_ep${epoch}.json" \
        --max_pixels_per_class="${KNN_MAX_PER_CLASS:-10000}" \
        --device=cpu \
        ${KNN_EXTRA_ARGS:-}
    echo ""
fi


if stage_on overlap; then
    echo "--- [3/8] probe_overlap (is a pixel's charge shared?) ---"
    python -u -m probes.probe_overlap "$features" \
        --out="${OUT_DIR}/overlap_ep${epoch}.json" \
        ${OVERLAP_EXTRA_ARGS:-$extra_args}
    echo ""
fi


if stage_on instance; then
    echo "--- [4/8] probe_instance (do neighbours share a particle?) ---"
    python -u -m probes.probe_instance "$features" \
        --out="${OUT_DIR}/instance_ep${epoch}.json" \
        ${INSTANCE_EXTRA_ARGS:-$extra_args}
    echo ""
fi


# Needs `apa`/`view` provenance in the features file to project the vertex; an
# extraction predating those aborts here with a message saying so.
if stage_on vertex; then
    echo "--- [5/8] probe_vertex (is a pixel near the interaction point?) ---"
    python -u -m probes.probe_vertex "$features" \
        --out="${OUT_DIR}/vertex_ep${epoch}.json" \
        ${VERTEX_EXTRA_ARGS:-$extra_args}
    echo ""
fi


if stage_on event; then
    echo "--- [6/8] probe_event (interaction flavor, pooled k-NN) ---"
    python -u -m probes.probe_event "$features" \
        --out="${OUT_DIR}/event_ep${epoch}.json" \
        ${EVENT_EXTRA_ARGS:-$extra_args}
    echo ""
fi


# Off by default (see PROBE_STAGES above): this one produces a picture, not a
# number, so it does not belong in every sweep. `--mode both` draws the pixel-PID
# and the event-flavor view from a single read of the features file, which is
# minutes of single-threaded zlib and dwarfs the reducing itself.
#
# `$extra_args` is deliberately NOT the fallback here, unlike pid/overlap/instance
# /vertex/event: it is documented as carrying --device, which plot_embedding does
# not accept and would abort on. Same reasoning as the knn stage.
#
# Per-epoch outputs, matching where the k-NN figures already go:
#   probes/ep<N>/embedding_{pid,event}.png
#   probes/ep<N>/embedding_{pid,event}_<features stem>.npz   <- redraw from this
if stage_on embed; then
    echo "--- [7/8] plot_embedding (2-D map, pixel PID + event flavor) ---"
    python -u -m probes.plot_embedding "$features" \
        --mode=both \
        --out_dir="${OUT_DIR}/ep${epoch}" \
        ${EMBED_EXTRA_ARGS:-}
    echo ""
fi


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
echo "--- [8/8] merge (all epochs present so far) ---"
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

echo "=== Probes complete! ==="
