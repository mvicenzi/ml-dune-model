#!/bin/bash
#
# Runner for test scripts that need a GPU (and flash_attn).
# Runs on the Condor worker; the login node has neither.
#
# Test scripts are plain python with their own runner and exit code -- the cluster
# environment has no pytest.
#
# Args (positional):
#   $1 codedir   -- path to ml-dune-model repo root
#   $2 pyenv     -- path to uv virtual environment to activate
#   $3 cache_dir -- cache base for warpconvnet benchmark files
#   $4  script   -- test script to run, e.g. tests/test_attention_pe.py
#
# Exit code is the script's, so a failing check fails the job.

set -euo pipefail

codedir=$1
pyenv=$2
cache_dir=$3
shift 3
script="$1"

wp_cache_gpfs="${cache_dir}/warpconvnet"
wp_cache="${_CONDOR_SCRATCH_DIR}/warpconvnet"
mkdir -p "$wp_cache_gpfs"

echo "Copying WarpConvNet benchmark cache to local scratch..."
cp -r "$wp_cache_gpfs" "$wp_cache"

echo "Running $CLUSTER_ID.$JOB_ID on $(hostname)"
# Default: a CPU-only test slot has no CUDA_VISIBLE_DEVICES, and set -u would abort.
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<none>}"
echo "  _CONDOR_SCRATCH_DIR=${_CONDOR_SCRATCH_DIR}"
echo ""

echo "JOB CONFIGURATION:"
echo "  codedir=${codedir}"
echo "  pyenv=${pyenv}"
echo "  cache_dir=${cache_dir}"
echo "  script=${script}"
echo ""

# --- NCCL on PCIe-only L40S (multi-GPU tests) --------------------------------
# sgpu0003/4 have no NVLink and their GPU-to-GPU P2P transport hangs at this
# driver/NCCL level: the process group initialises and the startup broadcasts
# succeed, but the first AllReduce never returns and the job sits until the
# watchdog kills it. Forcing the host shared-memory transport fixes it at no real
# cost (SHM bandwidth matches PCIe-P2P and our gradients are small). Harmless for
# single-GPU tests, which never touch NCCL.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
echo "  NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}  NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"

echo "WarpConNet overrides:"
export WARPCONVNET_USE_FP16_ACCUM=false
export WARPCONVNET_BENCHMARK_CACHE_DIR="$wp_cache"
echo "  WARPCONVNET_USE_FP16_ACCUM=${WARPCONVNET_USE_FP16_ACCUM}"
echo "  WARPCONVNET_BENCHMARK_CACHE_DIR=${WARPCONVNET_BENCHMARK_CACHE_DIR}"
echo ""

echo "Activating python environment..."
source "${pyenv}/bin/activate"

status=0
PYTHONPATH="$codedir${PYTHONPATH:+:$PYTHONPATH}" \
    python -u "${codedir}/${script}" || status=$?

echo "Syncing WarpConvNet benchmark cache back to GPFS..."
rsync -a "${wp_cache}/" "${wp_cache_gpfs}/" || true

echo "test exit status: ${status}"
exit $status
