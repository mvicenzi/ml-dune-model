#!/usr/bin/env bash
# Run all diagnostics for a given run name.
#
# Usage:
#   ./run_all_diagnostics.sh <runname>
#
# Expects (relative to this script's directory):
#   dino_debug/<runname>/histories.json
#   dino_checkpoints/<runname>/features_ep10.npz
#
# Outputs are written alongside each input file.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <runname>"
    exit 1
fi

RUNNAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HISTORIES="$SCRIPT_DIR/dino_debug/${RUNNAME}/histories.json"
FEATURES="$SCRIPT_DIR/dino_checkpoints/${RUNNAME}/features_ep10.npz"
CHECKPOINT="$SCRIPT_DIR/dino_checkpoints/${RUNNAME}/checkpoint_epoch10.pt"

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
    python -m dino.diagnostics.extract_features "$CHECKPOINT"
    echo
fi

echo "=== Running diagnostics for run: ${RUNNAME} ==="
echo

echo "--- [1/3] plot_histories ---"
python -m dino.diagnostics.plot_histories "$HISTORIES"
echo

echo "--- [2/3] plot_features (ep10) ---"
python -m dino.diagnostics.plot_features "$FEATURES" --n_dominant=5 --n_samples=30
echo


echo "--- [3/3] plot_knn (ep10, pixel k-NN) ---"
python -m dino.diagnostics.plot_knn "$FEATURES" --pixel_knn --n_pixel_samples=50000
echo

echo "=== Done. ==="
