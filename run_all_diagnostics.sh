#!/usr/bin/env bash
# Run all diagnostics for a given run name and epoch.
#
# Usage:
#   ./run_all_diagnostics.sh <runname> [epoch]
#
#   epoch defaults to 10.
#
# Expects (relative to this script's directory):
#   dino_debug/<runname>/histories.json
#   dino_checkpoints/<runname>/features_ep<epoch>.npz
#
# Outputs are written to:
#   dino_checkpoints/<runname>/ep<epoch>/

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <runname> [epoch]"
    exit 1
fi

RUNNAME="$1"
EPOCH="${2:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HISTORIES="$SCRIPT_DIR/dino_debug/${RUNNAME}/histories.json"
FEATURES="$SCRIPT_DIR/dino_checkpoints/${RUNNAME}/features_ep${EPOCH}.npz"
CHECKPOINT="$SCRIPT_DIR/dino_checkpoints/${RUNNAME}/checkpoint_epoch${EPOCH}.pt"
OUT_DIR="$SCRIPT_DIR/dino_checkpoints/${RUNNAME}/ep${EPOCH}"

# Validate inputs
if [[ ! -f "$HISTORIES" ]]; then
    echo "Error: histories file not found: $HISTORIES"
    exit 1
fi
if [[ ! -f "$FEATURES" ]]; then
    echo "Features file not found: $FEATURES"
    if [[ ! -f "$CHECKPOINT" ]]; then
        echo "Error: checkpoint not found either: $CHECKPOINT"
        exit 1
    fi
    echo "Extracting features from checkpoint: $CHECKPOINT"
    python -m dino.diagnostics.extract_features "$CHECKPOINT" --pixel_truth
    echo
fi

mkdir -p "$OUT_DIR"

echo "=== Running diagnostics for run: ${RUNNAME}  epoch: ${EPOCH} ==="
echo "    Output dir: ${OUT_DIR}"
echo

echo "--- [1/2] plot_histories ---"
python -m dino.diagnostics.plot_histories "$HISTORIES" --no_cov_plots
echo

# echo "--- plot_features (ep${EPOCH}) ---"
# python -m dino.diagnostics.plot_features "$FEATURES" --n_dominant=5 --n_samples=30 --out_dir="$OUT_DIR"
# echo

#echo "--- [2/3] plot_knn (ep${EPOCH}) ---"
#python -m dino.diagnostics.plot_knn "$FEATURES" --out_dir="$OUT_DIR"
#echo

echo "--- [2/2] plot_knn_pixel (ep${EPOCH}) ---"
python -m dino.diagnostics.plot_knn_pixel "$FEATURES" --out_dir="$OUT_DIR"
echo

echo "=== Done. ==="
