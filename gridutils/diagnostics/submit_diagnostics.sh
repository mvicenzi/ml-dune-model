#!/bin/bash
#
# Diagnostics submission to SDCC GPU pool.
#
# Usage:
#   bash gridutils/diagnostics/submit_diagnostics.sh <run_name> [epoch...] [extra_args...]
#
# Examples:
#   submit_diagnostics.sh myrun                          # epoch 100 (default)
#   submit_diagnostics.sh myrun 10 50 100                # epochs 10, 50, and 100
#   submit_diagnostics.sh myrun 100 --max_pixels_per_class=30000
#
# Epoch numbers (bare integers) and --flag style args are told apart automatically.
# Extra flags are forwarded verbatim to plot_knn_pixel.
# One Condor job is submitted per epoch; logs go to ${CONDOR_OUT}/<run_name>_diag/.

set -euo pipefail

# ---- User-overridable env --------------------------------------------------
CONDOR_OUT="${CONDOR_OUT:-/gpfs01/lbne/users/fm/${USER}/CONDOR_OUT}"
REPODIR="${REPODIR:-${HOME}/ml-dune-model}"
PYENV="${PYENV:-/gpfs01/lbne/users/fm/${USER}/uvenv}"

REQUEST_MEMORY="${REQUEST_MEMORY:-32000}"
REQUEST_GPUS="${REQUEST_GPUS:-1}"
REQUEST_CPUS="${REQUEST_CPUS:-4}"
GPU_REQUIREMENTS="${GPU_REQUIREMENTS:-(GPUs_DeviceName == \"NVIDIA L40S\") && (GPUs_Capability == 8.9)}"
# ---------------------------------------------------------------------------

if [ $# -lt 1 ]; then
  echo "usage: $0 <run_name> [epoch...] [--extra_arg=val ...]" >&2
  exit 2
fi

run_name=$1; shift

# Separate bare integers (epochs) from --flag style args
epochs=()
extra_args=()
for arg in "$@"; do
  if [[ "$arg" =~ ^[0-9]+$ ]]; then
    epochs+=("$arg")
  else
    extra_args+=("$arg")
  fi
done

# Default to epoch 100
if [ ${#epochs[@]} -eq 0 ]; then
  epochs=(100)
fi

run_dir="${CONDOR_OUT}/${run_name}"
if [ ! -d "$run_dir" ]; then
  echo "ERROR: run directory not found: $run_dir" >&2
  exit 1
fi

# Validate that features files exist for requested epochs
for ep in "${epochs[@]}"; do
  feat="${run_dir}/checkpoints/features_ep${ep}.npz"
  if [ ! -f "$feat" ]; then
    echo "ERROR: features file not found: $feat" >&2
    echo "       Run submit_extract.sh first." >&2
    exit 1
  fi
done

log_dir="${CONDOR_OUT}/${run_name}_diag"
mkdir -p "$log_dir"
echo "Condor logs -> ${log_dir}"

# Flatten extra args
extra_args_str=""
if [ ${#extra_args[@]} -gt 0 ]; then
  extra_args_str="${extra_args[*]}"
fi

# Build epoch list for Condor queue-from syntax
epoch_list=""
for ep in "${epochs[@]}"; do
  epoch_list+="  ${ep}"$'\n'
done

echo "Epochs to process (${#epochs[@]}): ${epochs[*]}"
echo ""

subfile="${log_dir}/${run_name}_diag.sub"
cat > "$subfile" <<EOF
universe                = vanilla
notification            = never
executable              = ${REPODIR}/gridutils/diagnostics/diagnosticsjob.sh
arguments               = ${REPODIR} ${PYENV} ${run_name} \$(EPOCH) ${CONDOR_OUT} ${extra_args_str}
environment             = "CLUSTER_ID=\$(ClusterId) JOB_ID=\$(ProcId)"
+JobBatchName           = "diag-${run_name}"
output                  = ${log_dir}/\$(ClusterId).\$(ProcId).out
error                   = ${log_dir}/\$(ClusterId).\$(ProcId).err
log                     = ${log_dir}/\$(ClusterId).\$(ProcId).log
getenv                  = False
request_memory          = ${REQUEST_MEMORY}
request_cpus            = ${REQUEST_CPUS}
request_gpus            = ${REQUEST_GPUS}
Requirements            = ${GPU_REQUIREMENTS}
should_transfer_files   = NO
queue EPOCH from (
${epoch_list})
EOF

echo "Submitting ${subfile}"
condor_submit "$subfile"
