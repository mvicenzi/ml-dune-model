#!/bin/bash
#
# Probe-suite submission to the SDCC pool.
#
# Usage:
#   bash gridutils/diagnostics/submit_probes.sh <run_name> [epoch...] [extra_args...]
#
# Examples:
#   submit_probes.sh myrun                        # epoch 100 (default)
#   submit_probes.sh myrun 50 100                 # epochs 50 and 100
#   submit_probes.sh myrun 100 --pool_per_class=10000
#   FEATURES_PREFIX=features_probe_ submit_probes.sh myrun 100
#   PROBE_STAGES=embed submit_probes.sh myrun 100   # the 2-D maps, nothing else
#
# Epoch numbers (bare integers) and --flag style args are told apart automatically.
# Extra flags are forwarded verbatim to every probe, so they may only be flags all
# of them accept (--seed, --device). Per-probe knobs go through the *_EXTRA_ARGS
# variables listed below, which probesjob.sh documents.
#
# NOTE: no GPU is requested. The probes consume an extracted features .npz and
# import neither warpconvnet nor a dataset reader, so they are CPU-only work —
# which also means they skip the GPU queue entirely.

set -euo pipefail

# ---- User-overridable env --------------------------------------------------
CONDOR_OUT="${CONDOR_OUT:-/gpfs01/lbne/users/fm/${USER}/CONDOR_OUT}"
REPODIR="${REPODIR:-${HOME}/ml-dune-model}"
PYENV="${PYENV:-/gpfs01/lbne/users/fm/${USER}/uvenv}"

# Basename prefix of the features files to score. Override when scoring a
# separate probe-ready extraction (e.g. features_probe_ep100.npz).
FEATURES_PREFIX="${FEATURES_PREFIX:-features_}"

# Measured, not estimated: the runner scores every stage in one process, and
# condor reported MemoryUsage 17090 MB for a 4-stage sweep and 14649 MB for a
# 6-stage one. That is above the ~12 GB a single stage needs, because the ~7 GB
# loaded file stays resident for the whole sweep while python does not return
# each stage's freed working set to the OS. 24 GB is ~40% over the observed peak.
#
# Do not drop this to 16000 on the reasoning that one stage fits there — that
# sizes the old one-process-per-stage layout, and the 4-stage sweep above would
# be killed. Raise it if a heavier stage is added.
#
# The CPU count matters more than the memory for matching: 32000 MB *and* 4 CPUs
# matched 4 slots in the whole pool and sat idle, while at Cpus>=1 there are ~33
# slots at 24000 MB. Check with:
#   condor_status -const 'Memory >= 24000 && Cpus >= 1' -af Name | wc -l
REQUEST_MEMORY="${REQUEST_MEMORY:-24000}"

# One CPU, because the work is single-threaded and asking for more does not just
# waste the allocation — it starves the job. This pool is carved into Cpus=1
# dynamic slots, so at 16 GB there are ~35 unclaimed slots at Cpus>=1 against ~4
# at Cpus>=4, and a 4-CPU request sits idle indefinitely.
#
# Nothing here parallelises: `LinearSVC` (liblinear) is single-threaded by
# construction, the MLP's per-batch matmul is 256x64 @ 64x128 — far too small to
# thread, measured no faster at 2-8 threads than at 1 — and the actual bottleneck
# is single-threaded zlib decompression of the ~7 GB feature file.
#
# Raise it only for a probe that genuinely threads (probe_knn_pid does all-pairs
# k-NN); PROBE_THREADS follows this value so the process never oversubscribes its
# allocation.
REQUEST_CPUS="${REQUEST_CPUS:-1}"

# Which probes to run, comma-separated (see probesjob.sh). The default is every
# measurement; `embed` is opt-in because it draws a picture rather than producing
# a number, and is usually wanted for one epoch rather than a whole sweep:
#
#   PROBE_STAGES=embed submit_probes.sh myrun 100
#
# That one stage IS the "embedding job" — same features file, same CPU-only
# profile, same memory, so it needs no submit script of its own.
PROBE_STAGES="${PROBE_STAGES:-pid,knn,overlap,instance,vertex,event}"
# ---------------------------------------------------------------------------

if [ $# -lt 1 ]; then
  echo "usage: $0 <run_name> [epoch...] [--extra_arg=val ...]" >&2
  exit 2
fi

run_name=$1; shift

epochs=()
extra_args=()
for arg in "$@"; do
  if [[ "$arg" =~ ^[0-9]+$ ]]; then
    epochs+=("$arg")
  else
    extra_args+=("$arg")
  fi
done

if [ ${#epochs[@]} -eq 0 ]; then
  epochs=(100)
fi

run_dir="${CONDOR_OUT}/${run_name}"
if [ ! -d "$run_dir" ]; then
  echo "ERROR: run directory not found: $run_dir" >&2
  exit 1
fi

# Validate features exist, and warn when they predate the extra per-pixel truth
# tiers (the instance / charge / strata metrics need them).
for ep in "${epochs[@]}"; do
  feat="${run_dir}/checkpoints/${FEATURES_PREFIX}ep${ep}.npz"
  if [ ! -f "$feat" ]; then
    echo "ERROR: features file not found: $feat" >&2
    echo "       Run submit_extract.sh first (with --extra_truth for the full suite)." >&2
    exit 1
  fi
done

log_dir="${CONDOR_OUT}/${run_name}_probes"
mkdir -p "$log_dir"
echo "Condor logs -> ${log_dir}"

extra_args_str=""
if [ ${#extra_args[@]} -gt 0 ]; then
  extra_args_str="${extra_args[*]}"
fi

epoch_list=""
for ep in "${epochs[@]}"; do
  epoch_list+="  ${ep}"$'\n'
done

echo "Epochs to process (${#epochs[@]}): ${epochs[*]}"
echo "Features prefix: ${FEATURES_PREFIX}"
echo "Stages:          ${PROBE_STAGES}"
echo ""

# `getenv = False`, so anything probesjob.sh reads from the environment has to be
# forwarded explicitly or it silently takes its default on the worker. These are
# the per-probe knobs probesjob.sh documents; values must not contain spaces or
# quotes (condor's environment syntax is space-separated).
job_env_extra=""
for var in PID_EXTRA_ARGS KNN_EXTRA_ARGS OVERLAP_EXTRA_ARGS \
           INSTANCE_EXTRA_ARGS VERTEX_EXTRA_ARGS EVENT_EXTRA_ARGS \
           EMBED_EXTRA_ARGS KNN_MAX_PER_CLASS; do
  val="${!var:-}"
  if [ -n "$val" ]; then
    case "$val" in
      *[[:space:]\"\']*)
        echo "ERROR: ${var} contains a space or quote, which condor's environment" >&2
        echo "       syntax cannot carry: ${var}=${val}" >&2
        exit 2 ;;
    esac
    job_env_extra+=" ${var}=${val}"
    echo "Forwarding:      ${var}=${val}"
  fi
done

subfile="${log_dir}/${run_name}_probes.sub"
cat > "$subfile" <<EOF
universe                = vanilla
notification            = never
executable              = ${REPODIR}/gridutils/diagnostics/probesjob.sh
arguments               = ${REPODIR} ${PYENV} ${run_name} \$(EPOCH) ${CONDOR_OUT} ${run_dir}/checkpoints/${FEATURES_PREFIX}ep\$(EPOCH).npz ${extra_args_str}
environment             = "CLUSTER_ID=\$(ClusterId) JOB_ID=\$(ProcId) PROBE_STAGES=${PROBE_STAGES} PROBE_THREADS=${REQUEST_CPUS}${job_env_extra}"
+JobBatchName           = "probes-${run_name}"
output                  = ${log_dir}/\$(ClusterId).\$(ProcId).out
error                   = ${log_dir}/\$(ClusterId).\$(ProcId).err
log                     = ${log_dir}/\$(ClusterId).\$(ProcId).log
getenv                  = False
request_memory          = ${REQUEST_MEMORY}
request_cpus            = ${REQUEST_CPUS}
should_transfer_files   = NO
queue EPOCH from (
${epoch_list})
EOF

echo "Submitting ${subfile}"
condor_submit "$subfile"
