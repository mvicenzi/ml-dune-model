#!/bin/bash
#
# Diagnostics job script — runs plot_histories + plot_knn_pixel for one epoch.
# Called by submit_diagnostics.sh; runs on a Condor worker.
#
# Args (positional):
#   $1 codedir    -- path to ml-dune-model repo root
#   $2 pyenv      -- path to uv virtual environment to activate
#   $3 run_name   -- run name (subdirectory under CONDOR_OUT)
#   $4 epoch      -- epoch number
#   $5 condor_out -- base CONDOR_OUT directory
#   $6+ extra_args -- forwarded to plot_knn_pixel (e.g. --max_pixels_per_class=30000)

set -euo pipefail

codedir=$1
pyenv=$2
run_name=$3
epoch=$4
condor_out=$5
shift 5
extra_args="${*}"

echo "Running $CLUSTER_ID.$JOB_ID on $(hostname)"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo ""
echo "JOB CONFIGURATION:"
echo "  codedir=${codedir}"
echo "  run_name=${run_name}"
echo "  epoch=${epoch}"
echo "  condor_out=${condor_out}"
echo "  extra_args=${extra_args}"
echo ""

echo "Activating python environment..."
source "${pyenv}/bin/activate"

HISTORIES="${condor_out}/${run_name}/debug/histories.json"
FEATURES="${condor_out}/${run_name}/checkpoints/features_ep${epoch}.npz"
OUT_DIR="${condor_out}/${run_name}/checkpoints/ep${epoch}"

if [[ ! -f "$HISTORIES" ]]; then
    echo "ERROR: histories file not found: $HISTORIES"
    exit 1
fi
if [[ ! -f "$FEATURES" ]]; then
    echo "ERROR: features file not found: $FEATURES"
    exit 1
fi

mkdir -p "$OUT_DIR"

echo "=== Running diagnostics for run: ${run_name}  epoch: ${epoch} ==="
echo "    Output dir: ${OUT_DIR}"
echo ""

echo "--- [1/2] plot_histories ---"
PYTHONPATH="$codedir${PYTHONPATH:+:$PYTHONPATH}" \
    python -m dino.diagnostics.plot_histories "$HISTORIES" --no_cov_plots
echo ""

echo "--- [2/2] plot_knn_pixel (ep${epoch}) ---"
PYTHONPATH="$codedir${PYTHONPATH:+:$PYTHONPATH}" \
    python -m dino.diagnostics.plot_knn_pixel "$FEATURES" \
        --out_dir="$OUT_DIR" \
        --device=cuda \
        $extra_args
echo ""

echo "=== Diagnostics complete! ==="
