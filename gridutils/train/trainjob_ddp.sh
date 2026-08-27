#!/bin/bash
#
# Multi-GPU (DDP) DINO training script.
# Runs on the Condor worker; called by submit.sh when request_gpus > 1.
# Single-node only: launches one rank per assigned GPU via torchrun.
#
# Args (positional):
#   $1 codedir  -- path to ml-dune-model repo root
#   $2 pyenv    -- path to uv virtual environment to activate
#   $3 config   -- path to run_config.json
#   $4 outdir   -- path to output directory on GPFS (where outputs are rsynced back)
#   $5 cache_dir -- general cache base; ${cache_dir}/warpconvnet and ${cache_dir}/data are used
#   $6 run_name -- run/training name
#
# I/O strategy: write everything to $_CONDOR_SCRATCH_DIR (fast local disk on
# the worker), rsync to GPFS at the end via an EXIT trap so partial outputs
# survive failures and preemption.

set -euo pipefail

codedir=$1
pyenv=$2
config=$3
outdir=$4
cache_dir=$5
run_name=$6

wp_cache_gpfs="${cache_dir}/warpconvnet"
wp_cache="${_CONDOR_SCRATCH_DIR}/warpconvnet"
data_cache="${cache_dir}/data"
mkdir -p "$wp_cache_gpfs" "$data_cache"

echo "Copying WarpConvNet benchmark cache to local scratch..."
cp -r "$wp_cache_gpfs" "$wp_cache"

# One rank per GPU Condor assigned to this slot (comma-separated list; works
# for both index and GPU-UUID forms).
ngpus=$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")

echo "Running $CLUSTER_ID.$JOB_ID on $(hostname)"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (ngpus=${ngpus})"
echo "  _CONDOR_SCRATCH_DIR=${_CONDOR_SCRATCH_DIR}"
echo ""

echo "JOB CONFIGURATION:"
echo "  run_name=${run_name}"
echo "  codedir=${codedir}"
echo "  pyenv=${pyenv}"
echo "  config=${config}"
echo "  outdir=${outdir}"
echo "  cache_dir=${cache_dir}"
echo ""

# --- NCCL on PCIe-only L40S ---------------------------------------------------
# sgpu0003/4 have no NVLink and their GPU-to-GPU P2P transport hangs at this
# driver/NCCL level: the process group initialises and the startup broadcasts
# succeed, but the first AllReduce never returns and the job sits until the
# watchdog kills it. Forcing the host shared-memory transport fixes it at no real
# cost (SHM bandwidth matches PCIe-P2P and our gradients are small).
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

# Stage all outputs on local scratch.  main() will append /${run_name} under
# each base, so the actual write dirs are $scratch_ckpt/$run_name and
# $scratch_dbg/$run_name.
scratch_ckpt=${_CONDOR_SCRATCH_DIR}/checkpoints
scratch_dbg=${_CONDOR_SCRATCH_DIR}/debug
mkdir -p "$scratch_ckpt" "$scratch_dbg"

sync_back() {
  echo "Syncing ${_CONDOR_SCRATCH_DIR} -> ${outdir}"
  mkdir -p "${outdir}/checkpoints" "${outdir}/debug"
  # Trailing slash on source flattens the inner /${run_name} dir, so the GPFS
  # layout is ${outdir}/{checkpoints,debug}/... without a redundant nest.
  rsync -a "${scratch_ckpt}/${run_name}/" "${outdir}/checkpoints/" || true
  rsync -a "${scratch_dbg}/${run_name}/"  "${outdir}/debug/"       || true
  # Merge benchmark cache back so new entries (new op shapes, updated WarpConvNet)
  # are available to future jobs. rsync uses atomic temp-file writes so concurrent
  # syncs from multiple jobs are safe.
  rsync -a "${wp_cache}/" "${wp_cache_gpfs}/" || true
}
# Periodic mid-run sync: save_every-epoch checkpoints land in scratch as soon
# as they're written, but via the EXIT trap alone they only reach GPFS when the
# whole job ends -- invisible to monitoring/probing for the entire run. A
# background loop re-runs the same idempotent sync_back every SYNC_INTERVAL
# seconds. rsync writes each destination file to a temp name and renames, so
# GPFS-visible files appear atomically; a checkpoint caught mid-write on the
# source transfers truncated but is repaired on the next tick (and by the final
# flush), so treat a checkpoint on GPFS as settled once a later one exists or
# the mtime is > SYNC_INTERVAL old.
SYNC_INTERVAL="${SYNC_INTERVAL:-300}"
( while sleep "$SYNC_INTERVAL"; do sync_back; done ) &
sync_loop_pid=$!
stop_sync_loop() { kill "$sync_loop_pid" 2>/dev/null || true; }

trap 'stop_sync_loop; sync_back' EXIT              # flush scratch -> GPFS on any normal/error exit
trap 'stop_sync_loop; sync_back; exit 143' SIGTERM # on scheduler kill: flush, then exit 128+15 (SIGTERM)

echo "Executing train_dino.py under torchrun (${ngpus} ranks) ..."

# --standalone: single-node rendezvous on a free localhost port, so concurrent
# jobs on the same worker cannot collide on a fixed master port.
PYTHONPATH="$codedir${PYTHONPATH:+:$PYTHONPATH}" \
PYTHONUNBUFFERED=1 \
    torchrun --standalone --nnodes=1 --nproc_per_node="$ngpus" \
        -m dino.train_dino from_config \
        --config_path="$config" \
        --output_dir="$scratch_ckpt" \
        --debug_dir="$scratch_dbg" \
        --cache_dir="$data_cache" \
        --device=cuda

echo "Training complete!"
