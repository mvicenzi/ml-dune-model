#!/bin/bash
#
# DINO training script.  
# Runs on the Condor worker; called by submit.sh.
#
# Args (positional):
#   $1 codedir  -- path to ml-dune-model repo root
#   $2 pyenv    -- path to uv virtual environment to activate
#   $3 config   -- path to run_config.json
#   $4 outdir   -- path to output directory on GPFS (where outputs are rsynced back)
#   $5 wp_cache -- path to warpconvnet cache dir
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
wp_cache=$5
run_name=$6

echo "[job.sh] codedir=${codedir}"
echo "[job.sh] pyenv=${pyenv}"
echo "[job.sh] config=${config}"
echo "[job.sh] outdir=${outdir}"
echo "[job.sh] wp_cache=${wp_cache}"
echo "[job.sh] run_name=${run_name}"
echo "[job.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[job.sh] _CONDOR_SCRATCH_DIR=${_CONDOR_SCRATCH_DIR}"

source "${pyenv}/bin/activate"
export WARPCONVNET_BENCHMARK_CACHE_DIR="$wp_cache"

# Stage all outputs on local scratch.  main() will append /${run_name} under
# each base, so the actual write dirs are $scratch_ckpt/$run_name and
# $scratch_dbg/$run_name.
scratch_ckpt=${_CONDOR_SCRATCH_DIR}/checkpoints
scratch_dbg=${_CONDOR_SCRATCH_DIR}/debug
mkdir -p "$scratch_ckpt" "$scratch_dbg"

sync_back() {
  echo "[job.sh] syncing ${_CONDOR_SCRATCH_DIR} -> ${outdir}"
  mkdir -p "${outdir}/checkpoints" "${outdir}/debug"
  # Trailing slash on source flattens the inner /${run_name} dir, so the GPFS
  # layout is ${outdir}/{checkpoints,debug}/... without a redundant nest.
  rsync -a "${scratch_ckpt}/${run_name}/" "${outdir}/checkpoints/" || true
  rsync -a "${scratch_dbg}/${run_name}/"  "${outdir}/debug/"       || true
}
trap sync_back EXIT                      # flush scratch -> GPFS on any normal/error exit
trap 'sync_back; exit 143' SIGTERM       # on scheduler kill: flush, then exit 128+15 (SIGTERM)

PYTHONPATH="$codedir${PYTHONPATH:+:$PYTHONPATH}" \
    python -m dino.train_dino from_config "$config" \
        --output_dir="$scratch_ckpt" \
        --debug_dir="$scratch_dbg" \
        --device=cuda

echo "[job.sh] training complete"
