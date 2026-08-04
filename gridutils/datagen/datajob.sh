#!/bin/bash
#
# Dataset-generation job (CPU-only). Runs on the Condor worker; called by
# submit_datagen.sh. Executes an arbitrary loader module (create_shards,
# pack_dataset, ...) with the given arguments.
#
# Args (positional):
#   $1  codedir -- path to ml-dune-model repo root
#   $2  pyenv   -- path to uv virtual environment to activate
#   $3+ module and its arguments, e.g.
#         loader.create_shards --datadir ... --apa 0 --view W --outdir ...
#
# The loader scripts write their outputs (shards/pack) directly to the GPFS
# paths given in their arguments; shard creation is resumable, so a killed
# job can simply be resubmitted.

set -euo pipefail

codedir=$1
pyenv=$2
shift 2

echo "Running ${CLUSTER_ID:-?}.${JOB_ID:-?} on $(hostname)"
echo "  _CONDOR_SCRATCH_DIR=${_CONDOR_SCRATCH_DIR}"
echo ""
echo "JOB CONFIGURATION:"
echo "  codedir=${codedir}"
echo "  pyenv=${pyenv}"
echo "  command=python -u -m $*"
echo ""

# Importing loader/ initializes warpconvnet; keep its benchmark cache on
# local scratch (this is a CPU job — nothing worth merging back).
export WARPCONVNET_BENCHMARK_CACHE_DIR="${_CONDOR_SCRATCH_DIR}/warpconvnet"

echo "Activating python environment..."
source "${pyenv}/bin/activate"

PYTHONPATH="$codedir${PYTHONPATH:+:$PYTHONPATH}" \
    python -u -m "$@"

echo "Data generation complete!"
