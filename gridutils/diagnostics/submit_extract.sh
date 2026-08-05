#!/bin/bash
#
# DINO feature-extraction submission to SDCC GPU pool.
#
# Usage:
#   bash gridutils/diagnostics/submit_extract.sh <run_name> [epoch...] [extra_args...]
#
# Examples:
#   submit_extract.sh myrun                          # all checkpoints in the run
#   submit_extract.sh myrun 10                       # epoch 10 only
#   submit_extract.sh myrun 10 50 100                # epochs 10, 50, and 100
#   submit_extract.sh myrun 10 --max_images=5000     # epoch 10, limit images
#
# Epoch numbers (bare integers) and --flag style args are told apart automatically.
# Extra flags are forwarded verbatim to probes.extract_features
# (in addition to --pixel_truth which is always passed; see extractjob.sh).
# One Condor job is submitted per checkpoint; logs go to ${CONDOR_OUT}/<run_name>_extract/.

set -euo pipefail

# ---- User-overridable env --------------------------------------------------

CONDOR_OUT="${CONDOR_OUT:-/gpfs01/lbne/users/fm/${USER}/CONDOR_OUT}"
REPODIR="${REPODIR:-${HOME}/ml-dune-model}"
PYENV="${PYENV:-/gpfs01/lbne/users/fm/${USER}/uvenv}"
CACHE_DIR="${CACHE_DIR:-/gpfs01/lbne/users/fm/${USER}/cache}"

# Extraction reads the full training shard set (full truth) deterministically.
# Automatically appended to extra_args when the directory exists and not already overridden.
TRUTH_SHARDS_DIR="${TRUTH_SHARDS_DIR:-/gpfs01/lbne/users/fm/cffm-data/shards_prod-jay-2026-06-11_mixed_apa0W}"

REQUEST_MEMORY="${REQUEST_MEMORY:-32000}"
REQUEST_GPUS="${REQUEST_GPUS:-1}"
REQUEST_CPUS="${REQUEST_CPUS:-4}"
GPU_REQUIREMENTS="${GPU_REQUIREMENTS:-(GPUs_DeviceName == \"NVIDIA L40S\") && (GPUs_Capability == 8.9)}"

# ---- Args / validation -----------------------------------------------------
if [ $# -lt 1 ]; then
  echo "usage: $0 <run_name> [epoch...] [--max_images=N] [--batch_size=N]" >&2
  exit 2
fi

run_name=$1; shift

# Separate bare integers (epochs) from --flag style args (extra_args)
epochs=()
extra_args=()
for arg in "$@"; do
  if [[ "$arg" =~ ^[0-9]+$ ]]; then
    epochs+=("$arg")
  else
    extra_args+=("$arg")
  fi
done

checkpoints_dir="${CONDOR_OUT}/${run_name}/checkpoints"
if [ ! -d "$checkpoints_dir" ]; then
  echo "ERROR: checkpoints directory not found: $checkpoints_dir" >&2
  exit 1
fi

# Resolve checkpoint paths
checkpoints=()
if [ ${#epochs[@]} -eq 0 ]; then
  # No epochs specified: find all checkpoint_epoch*.pt, version-sorted
  while IFS= read -r -d '' f; do
    checkpoints+=("$f")
  done < <(find "$checkpoints_dir" -maxdepth 1 -name 'checkpoint_epoch*.pt' -print0 | sort -zV)
  if [ ${#checkpoints[@]} -eq 0 ]; then
    echo "ERROR: no checkpoint_epoch*.pt files found in ${checkpoints_dir}" >&2
    exit 1
  fi
else
  for ep in "${epochs[@]}"; do
    ckpt="${checkpoints_dir}/checkpoint_epoch${ep}.pt"
    if [ ! -f "$ckpt" ]; then
      echo "ERROR: checkpoint not found: $ckpt" >&2
      exit 1
    fi
    checkpoints+=("$ckpt")
  done
fi

echo "Checkpoints to process (${#checkpoints[@]}):"
for c in "${checkpoints[@]}"; do echo "  $c"; done
echo ""

log_dir="${CONDOR_OUT}/${run_name}_extract"
mkdir -p "$log_dir"
echo "Condor logs -> ${log_dir}"

# Build the per-job checkpoint list for Condor's queue-from syntax
checkpoint_list=""
for c in "${checkpoints[@]}"; do
  checkpoint_list+="  ${c}"$'\n'
done

# Auto-inject --truth_shards_dir when the standard path exists and not already specified.
if [ -d "$TRUTH_SHARDS_DIR" ]; then
  already_set=false
  for a in "${extra_args[@]}"; do
    [[ "$a" == --truth_shards_dir* ]] && already_set=true && break
  done
  if ! $already_set; then
    extra_args+=("--truth_shards_dir=${TRUTH_SHARDS_DIR}")
    echo "Truth shards  -> ${TRUTH_SHARDS_DIR}"
  fi
fi

# Auto-inject --extra_truth when the shard set carries the extra per-pixel tiers
# (pixel_energyfrac / pixel_trackid / pixel_truth_q). The instance, charge and
# overlap-strata probes need them; asking for them against a shard set that lacks
# them would abort the job, hence the metadata check rather than a blind default.
# Set EXTRA_TRUTH=0 to opt out.
EXTRA_TRUTH="${EXTRA_TRUTH:-1}"
meta="${TRUTH_SHARDS_DIR}/metadata.json"
if [ "$EXTRA_TRUTH" = "1" ] && [ -f "$meta" ] && grep -q '"extra_truth"[[:space:]]*:[[:space:]]*true' "$meta"; then
  already_set=false
  for a in "${extra_args[@]}"; do
    [[ "$a" == --extra_truth* ]] && already_set=true && break
  done
  if ! $already_set; then
    extra_args+=("--extra_truth")
    echo "Extra truth   -> enabled (shard metadata reports extra_truth: true)"
  fi
fi

# Flatten extra args to a single string (safe: all are --flag style, no spaces)
extra_args_str=""
if [ ${#extra_args[@]} -gt 0 ]; then
  extra_args_str="${extra_args[*]}"
fi

subfile="${log_dir}/${run_name}_extract.sub"
cat > "$subfile" <<EOF
universe                = vanilla
notification            = never
executable              = ${REPODIR}/gridutils/diagnostics/extractjob.sh
arguments               = ${REPODIR} ${PYENV} \$(CHECKPOINT) ${CACHE_DIR} ${extra_args_str}
environment             = "CLUSTER_ID=\$(ClusterId) JOB_ID=\$(ProcId)"
+JobBatchName           = "extract-${run_name}"
output                  = ${log_dir}/\$(ClusterId).\$(ProcId).out
error                   = ${log_dir}/\$(ClusterId).\$(ProcId).err
log                     = ${log_dir}/\$(ClusterId).\$(ProcId).log
getenv                  = False
request_memory          = ${REQUEST_MEMORY}
request_cpus            = ${REQUEST_CPUS}
request_gpus            = ${REQUEST_GPUS}
Requirements            = ${GPU_REQUIREMENTS}
should_transfer_files   = NO
queue CHECKPOINT from (
${checkpoint_list})
EOF

echo "Submitting ${subfile}"
condor_submit "$subfile"
