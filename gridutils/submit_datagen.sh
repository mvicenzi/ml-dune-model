#!/bin/bash
#
# Submit a CPU-only dataset-generation job (shard/pack creation) to Condor.
#
# Usage:
#   bash gridutils/submit_datagen.sh <job_name> <module> [args...]
#
# Examples (DATA = production root; outputs under /gpfs01/lbne/users/fm/$USER):
#   DATA=/gpfs01/lbne/users/bnayak/cffm-data/prod-jay-100k-truth-2026-06-11
#   OUT=/gpfs01/lbne/users/fm/${USER}/cffm-data
#
#   # full training shards (full truth; extraction reads the same set):
#   bash gridutils/submit_datagen.sh shards_mixed_apa0W \
#       loader.create_shards --datadir $DATA --apa 0 --view W \
#       --outdir $OUT/shards_prod-jay-2026-06-11_mixed_apa0W \
#       --cache_dir /gpfs01/lbne/users/fm/${USER}/cache/data \
#       --with_extra_truth --num_workers 8
#
#   # packed .npz (RAM-heavy: ~45 GB peak at 200k events):
#   REQUEST_MEMORY=64000 bash gridutils/submit_datagen.sh pack_mixed_apa0W \
#       loader.pack_dataset --datadir $DATA --apa 0 --view W \
#       --out_path $OUT/packed/prod-jay-2026-06-11_mixed_apa0W.npz \
#       --cache_dir /gpfs01/lbne/users/fm/${USER}/cache/data --num_workers 8
#
# Note: jobs sharing a --cache_dir each scan the dataset if the index cache
# is missing (the write is atomic, so concurrent jobs are safe — just
# wasteful). If the cache doesn't exist yet, let one job build it before
# submitting the others.

set -euo pipefail

# ---- User-overridable env --------------------------------------------------
CONDOR_OUT="${CONDOR_OUT:-/gpfs01/lbne/users/fm/${USER}/CONDOR_OUT}"
REPODIR="${REPODIR:-${HOME}/ml-dune-model}"
PYENV="${PYENV:-/gpfs01/lbne/users/fm/${USER}/uvenv}"

# CPU-only job requirements (pack creation needs REQUEST_MEMORY=64000).
REQUEST_MEMORY="${REQUEST_MEMORY:-16000}"
REQUEST_CPUS="${REQUEST_CPUS:-8}"

# ---- Args / validation -----------------------------------------------------
if [ $# -lt 2 ]; then
  echo "usage: $0 <job_name> <module> [args...]" >&2
  exit 2
fi

job_name=$1
shift

out_dir="${CONDOR_OUT}/datagen/${job_name}"
if [ -d "$out_dir" ]; then
  echo "ERROR: ${out_dir} already exists." >&2
  echo "       Choose a new job_name or delete the directory and retry." >&2
  exit 1
fi

# ---- Layout ----------------------------------------------------------------
echo "Creating job directory: ${out_dir}"
mkdir -p "$out_dir"

subfile="${out_dir}/${job_name}.sub"
cat > "$subfile" <<EOF
universe                = vanilla
notification            = never
executable              = ${REPODIR}/gridutils/datajob.sh
arguments               = ${REPODIR} ${PYENV} $*
environment             = "CLUSTER_ID=\$(ClusterId) JOB_ID=\$(ProcId)"
output                  = ${out_dir}/\$(ClusterId).\$(ProcId).out
error                   = ${out_dir}/\$(ClusterId).\$(ProcId).err
log                     = ${out_dir}/\$(ClusterId).\$(ProcId).log
getenv                  = False
request_memory          = ${REQUEST_MEMORY}
request_cpus            = ${REQUEST_CPUS}
should_transfer_files   = NO
queue 1
EOF

echo "Submitting ${subfile}"
condor_submit "$subfile"
