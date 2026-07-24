#!/bin/bash
#
# DINO feature-extraction script.
# Runs on the Condor worker; called by submit_extract.sh.
#
# Args (positional):
#   $1 codedir    -- path to ml-dune-model repo root
#   $2 pyenv      -- path to uv virtual environment to activate
#   $3 checkpoint -- absolute path to the .pt checkpoint on GPFS
#   $4 cache_dir  -- cache base for warpconvnet benchmark files
#   $5+ extra_args -- forwarded to extract_features (e.g. --pixel_truth --max_images=5000)
#
# Output: features_ep<N>.npz written alongside the checkpoint on GPFS.
# Override with --output=<path> in extra_args.

set -euo pipefail

codedir=$1
pyenv=$2
checkpoint=$3
cache_dir=$4
shift 4
extra_args="${*}"

wp_cache_gpfs="${cache_dir}/warpconvnet"
wp_cache="${_CONDOR_SCRATCH_DIR}/warpconvnet"
mkdir -p "$wp_cache_gpfs"

echo "Copying WarpConvNet benchmark cache to local scratch..."
cp -r "$wp_cache_gpfs" "$wp_cache"

echo "Running $CLUSTER_ID.$JOB_ID on $(hostname)"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  _CONDOR_SCRATCH_DIR=${_CONDOR_SCRATCH_DIR}"
echo ""

echo "JOB CONFIGURATION:"
echo "  codedir=${codedir}"
echo "  pyenv=${pyenv}"
echo "  checkpoint=${checkpoint}"
echo "  cache_dir=${cache_dir}"
echo "  extra_args=${extra_args}"
echo ""

echo "WarpConNet overrides:"
export WARPCONVNET_USE_FP16_ACCUM=false
export WARPCONVNET_BENCHMARK_CACHE_DIR="$wp_cache"
echo "  WARPCONVNET_USE_FP16_ACCUM=${WARPCONVNET_USE_FP16_ACCUM}"
echo "  WARPCONVNET_BENCHMARK_CACHE_DIR=${WARPCONVNET_BENCHMARK_CACHE_DIR}"
echo ""

echo "Activating python environment..."
source "${pyenv}/bin/activate"

echo "Extracting features from: ${checkpoint}"

data_cache="${cache_dir}/data"
mkdir -p "$data_cache"

PYTHONPATH="$codedir${PYTHONPATH:+:$PYTHONPATH}" \
    python -u -m dino.diagnostics.extract_features \
        "$checkpoint" \
        --pixel_truth \
        --device=cuda \
        --cache_dir="$data_cache" \
        $extra_args

echo "Syncing WarpConvNet benchmark cache back to GPFS..."
rsync -a "${wp_cache}/" "${wp_cache_gpfs}/" || true

echo "Feature extraction complete!"
