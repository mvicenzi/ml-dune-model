#!/bin/bash
#
# Drive extraction + probes over a run's checkpoints, waiting for each stage.
#
# Unlike its submit_*.sh siblings this script does NOT submit a Condor job and
# return: it runs on the login node and blocks, submitting extraction and probe
# jobs as each checkpoint becomes available. It is meant to be left running
# against a training job that is still producing checkpoints.
#
# Usage:
#   bash gridutils/diagnostics/collect_probes.sh <run_name> [epoch...] [options]
#
# Examples:
#   collect_probes.sh myrun                            # every checkpoint present now
#   collect_probes.sh myrun 60 70 80 90 100            # named epochs, waiting for each
#   collect_probes.sh myrun 100 --max_images=10000 --features_prefix=features_10k_
#   collect_probes.sh myrun 50 100 --parallel=2 --stages=pid,overlap
#
# Detached, which is the usual way to leave it against a live run (note the
# explicit `bash` — the login shell is tcsh, where `VAR=x script` is a syntax
# error, so every knob here is a flag rather than an environment variable):
#   nohup bash gridutils/diagnostics/collect_probes.sh myrun 60 70 80 90 100 \
#       --max_images=10000 --features_prefix=features_10k_ > collect.log 2>&1 &
#
# Bare integers are epochs. The options below are consumed here; every other
# --flag is forwarded verbatim to submit_extract.sh.
#
#   --features_prefix=P  basename prefix of the features file (default features_).
#                        Also supplied to extraction as --output_prefix, so the
#                        two cannot drift apart.
#   --stages=a,b,c       forwarded to submit_probes.sh as PROBE_STAGES.
#   --parallel=N         probe jobs allowed in the queue at once (default 1).
#   --timeout=S          give up waiting for one epoch after S seconds
#                        (default 43200 = 12 h). 0 disables the deadline.
#   --poll=S             seconds between checks while waiting (default 120).
#   --settle=S           re-check gap for the .npz integrity test (default 90).
#   --force              re-probe an epoch whose result JSONs already exist.
#   --dry_run            report what would be submitted, submit nothing.
#
# Serialisation is deliberate. Two probe sweeps running at once contend on GPFS
# badly enough to matter: probe_overlap measured 2905 s and 4043 s with two jobs
# in flight against 1963 s for the same stage running alone. --parallel=1 is
# therefore the default even though the pool would happily run more.

set -euo pipefail

# ---- User-overridable env --------------------------------------------------
CONDOR_OUT="${CONDOR_OUT:-/gpfs01/lbne/users/fm/${USER}/CONDOR_OUT}"
REPODIR="${REPODIR:-${HOME}/ml-dune-model}"

# Members every extraction writes regardless of its flags, used to tell a
# complete .npz from a valid-but-differently-configured one. The conditional
# tiers (--extra_truth's pixel_*, --head_features) are deliberately absent, and
# the feature array itself is matched by suffix because its name carries the
# source (student_features / teacher_features).
REQUIRE_KEYS="${REQUIRE_KEYS:-labels,positions,charges,offsets,event_key,epoch,source,checkpoint_path,extraction_source,feature_dim}"
# ---------------------------------------------------------------------------

if [ $# -lt 1 ]; then
  echo "usage: $0 <run_name> [epoch...] [--parallel=N] [--extract_flag=val ...]" >&2
  exit 2
fi

run_name=$1; shift

features_prefix="features_"
stages=""
parallel=1
timeout_s=43200
poll_s=120
settle_s=90
force=false
dry_run=false

epochs=()
extract_args=()
for arg in "$@"; do
  case "$arg" in
    --features_prefix=*) features_prefix="${arg#*=}" ;;
    --stages=*)          stages="${arg#*=}" ;;
    --parallel=*)        parallel="${arg#*=}" ;;
    --timeout=*)         timeout_s="${arg#*=}" ;;
    --poll=*)            poll_s="${arg#*=}" ;;
    --settle=*)          settle_s="${arg#*=}" ;;
    --force)             force=true ;;
    --dry_run)           dry_run=true ;;
    *)
      if [[ "$arg" =~ ^[0-9]+$ ]]; then
        epochs+=("$arg")
      else
        extract_args+=("$arg")
      fi ;;
  esac
done

run_dir="${CONDOR_OUT}/${run_name}"
ckpt_dir="${run_dir}/checkpoints"
probe_dir="${run_dir}/probes"

if [ ! -d "$run_dir" ]; then
  echo "ERROR: run directory not found: $run_dir" >&2
  exit 1
fi

# No epochs named: take the checkpoints that exist now. Waiting for a checkpoint
# that has not been written yet only makes sense when the caller says which.
if [ ${#epochs[@]} -eq 0 ]; then
  while IFS= read -r f; do
    ep="${f##*checkpoint_epoch}"
    epochs+=("${ep%.pt}")
  done < <(find "$ckpt_dir" -maxdepth 1 -name 'checkpoint_epoch*.pt' | sort -V)
  if [ ${#epochs[@]} -eq 0 ]; then
    echo "ERROR: no checkpoint_epoch*.pt in ${ckpt_dir} and no epochs given" >&2
    exit 1
  fi
fi

# The features prefix drives extraction's output name too, unless the caller
# already passed one. Setting one without the other is the classic way to end up
# probing a file that does not exist.
already_prefixed=false
for a in ${extract_args[@]+"${extract_args[@]}"}; do
  [[ "$a" == --output_prefix* ]] && already_prefixed=true && break
done
if ! $already_prefixed && [ "$features_prefix" != "features_" ]; then
  extract_args+=("--output_prefix=${features_prefix}")
fi

echo "Run:             ${run_name}"
echo "Epochs (${#epochs[@]}):     ${epochs[*]}"
echo "Features prefix: ${features_prefix}"
echo "Parallel probes: ${parallel}"
echo "Extraction args: ${extract_args[*]-<none>}"
$dry_run && echo "DRY RUN — nothing will be submitted"
echo ""

# ---- Helpers ---------------------------------------------------------------

# Full command line of every queued job, one per line. `condor_q -nobatch`
# cannot be used for this: it prints the executable's basename followed by
# arguments truncated to the terminal width, and a checkpoint path sits ~140
# characters into extractjob.sh's arguments, so it is cut off and never matches.
queued_cmdlines() {
  condor_q -af Cmd Args 2>/dev/null || true
}

extract_queued_for() {   # is an extraction for this checkpoint already in flight?
  queued_cmdlines | grep -q "extractjob\.sh.*checkpoint_epoch${1}\.pt"
}

# Every probe sweep in the queue, whoever submitted it: GPFS contention does not
# care which run a competing job belongs to, so the --parallel gate counts all.
probe_jobs_queued() {
  queued_cmdlines | grep -c "probesjob\.sh" || true
}

# Only this run's sweeps. probesjob.sh takes <repodir> <pyenv> <run_name> ...,
# so the run name is a whitespace-delimited token on the argument line. Used for
# the final wait, which must not block on an unrelated run's job.
probe_jobs_queued_this_run() {
  queued_cmdlines | grep -c "probesjob\.sh.*[[:space:]]${run_name}[[:space:]]" || true
}

npz_ok() {   # a complete features file: readable zip directory + expected members
  python3 - "$1" "$REQUIRE_KEYS" <<'PY' 2>/dev/null
import sys, zipfile
try:
    names = set(zipfile.ZipFile(sys.argv[1]).namelist())
except Exception:
    sys.exit(1)
required = {k + ".npy" for k in sys.argv[2].split(",") if k}
if not required <= names:
    sys.exit(1)
# The feature array is named for its source, so match it by suffix.
sys.exit(0 if any(n.endswith("_features.npy") for n in names) else 1)
PY
}

# Every requested stage already has its result JSON. The stage names and their
# output filenames come from STAGES in probes/run_probes.py — note `knn` writes
# pixelknn_ep<N>.json and `embed` writes a directory, so the mapping is not
# uniform. The fallback list must track submit_probes.sh's PROBE_STAGES default.
probes_done_for() {
  local ep=$1 stage_list="${stages:-pid,knn,overlap,instance,vertex,event}"
  local stage json
  for stage in ${stage_list//,/ }; do
    case "$stage" in
      knn)   json="${probe_dir}/pixelknn_ep${ep}.json" ;;
      embed) json="${probe_dir}/ep${ep}" ;;
      *)     json="${probe_dir}/${stage}_ep${ep}.json" ;;
    esac
    [ -e "$json" ] || return 1
  done
  return 0
}

# Blocks until `cond ep` succeeds. Returns 1 on deadline so the caller can skip
# the epoch and carry on rather than the whole sweep hanging on one bad number.
wait_for() {
  local cond=$1 ep=$2 what=$3 start
  start=$(date +%s)
  until "$cond" "$ep"; do
    if [ "$timeout_s" -gt 0 ] && [ $(( $(date +%s) - start )) -ge "$timeout_s" ]; then
      echo "  TIMEOUT after ${timeout_s}s waiting for ${what} (ep${ep}) — skipping" >&2
      return 1
    fi
    sleep "$poll_s"
  done
  return 0
}

# A checkpoint is usable once it has stopped growing. Training rsyncs to GPFS on
# a 300 s tick, so a file younger than that may still be mid-copy.
checkpoint_ready() {
  local f="${ckpt_dir}/checkpoint_epoch${1}.pt"
  [ -s "$f" ] && [ $(( $(date +%s) - $(stat -c %Y "$f") )) -gt 360 ]
}

# Integrity has to hold across two checks, because a duplicate extraction
# overwriting the file in place looks complete both before and after but not
# during. The queue check on its own is not enough: a job can start between the
# check and the read.
npz_settled() {
  local f="${ckpt_dir}/${features_prefix}ep${1}.npz"
  if extract_queued_for "$1" || ! npz_ok "$f"; then return 1; fi
  sleep "$settle_s"
  if extract_queued_for "$1" || ! npz_ok "$f"; then return 1; fi
  return 0
}

probe_slot_free() {   # ignores its epoch argument; wait_for's signature
  [ "$(probe_jobs_queued)" -lt "$parallel" ]
}

# ---- Collect ---------------------------------------------------------------

skipped=()      # nothing to do — results were already on disk
abandoned=()    # gave up waiting; the caller must not treat these as collected
submitted=()

for ep in "${epochs[@]}"; do
  echo "=== ep${ep} ==="

  if ! $force && probes_done_for "$ep"; then
    echo "  probe results already present — skipping (use --force to redo)"
    skipped+=("$ep")
    continue
  fi

  if ! wait_for checkpoint_ready "$ep" "checkpoint_epoch${ep}.pt"; then
    abandoned+=("$ep"); continue
  fi

  npz="${ckpt_dir}/${features_prefix}ep${ep}.npz"
  # Older than its checkpoint means the run was resumed and rewrote that epoch:
  # the file is a perfectly valid .npz of a model that no longer exists there.
  if npz_ok "$npz" && [ "$npz" -ot "${ckpt_dir}/checkpoint_epoch${ep}.pt" ]; then
    echo "  features are older than the checkpoint — re-extracting"
    if $dry_run; then
      echo "  would submit: submit_extract.sh ${run_name} ${ep} ${extract_args[*]-}"
    else
      (cd "$REPODIR" && bash gridutils/diagnostics/submit_extract.sh \
          "$run_name" "$ep" ${extract_args[@]+"${extract_args[@]}"} 2>&1 | tail -2)
    fi
  elif npz_ok "$npz"; then
    echo "  features already extracted"
  elif extract_queued_for "$ep"; then
    echo "  extraction already in flight"
  elif $dry_run; then
    echo "  would submit: submit_extract.sh ${run_name} ${ep} ${extract_args[*]-}"
  else
    echo "  submitting extraction"
    (cd "$REPODIR" && bash gridutils/diagnostics/submit_extract.sh \
        "$run_name" "$ep" ${extract_args[@]+"${extract_args[@]}"} 2>&1 | tail -2)
  fi

  if $dry_run; then
    echo "  would submit probes once ${npz##*/} validates"
    submitted+=("$ep"); continue
  fi

  if ! wait_for npz_settled "$ep" "a stable ${features_prefix}ep${ep}.npz"; then
    abandoned+=("$ep"); continue
  fi
  echo "  features validated"

  # On deadline here we submit anyway: a busy queue is a reason to wait, not a
  # reason to drop the epoch's measurement.
  wait_for probe_slot_free "$ep" "a free probe slot" || true
  echo "  submitting probes"
  # `env` rather than a bare assignment prefix, because a word produced by
  # expansion (the optional PROBE_STAGES) is not parsed as an assignment.
  (cd "$REPODIR" && env FEATURES_PREFIX="$features_prefix" \
      ${stages:+PROBE_STAGES="$stages"} \
      bash gridutils/diagnostics/submit_probes.sh "$run_name" "$ep" 2>&1 | tail -2)
  submitted+=("$ep")
done

# Exit only once this run's sweeps have left the queue, so that a caller chaining
# a merge or a plot sees every JSON on disk. Deadlined like every other wait: a
# held job must not hang an unattended collector forever.
if ! $dry_run && [ ${#submitted[@]} -gt 0 ]; then
  echo ""
  echo "Waiting for this run's probe jobs to finish ..."
  wait_start=$(date +%s)
  while [ "$(probe_jobs_queued_this_run)" -gt 0 ]; do
    if [ "$timeout_s" -gt 0 ] && [ $(( $(date +%s) - wait_start )) -ge "$timeout_s" ]; then
      echo "  TIMEOUT after ${timeout_s}s — probe jobs still queued, check condor_q" >&2
      abandoned+=("still-queued")
      break
    fi
    sleep "$poll_s"
  done
fi

echo ""
echo "=== collection finished ==="
$dry_run && echo "would submit:   ${submitted[*]-<none>}" \
         || echo "submitted:      ${submitted[*]-<none>}"
echo "already done:   ${skipped[*]-<none>}"
echo "abandoned:      ${abandoned[*]-<none>}"

# Non-zero when an epoch was given up on, so that a merge or a plot chained
# after this script does not quietly tabulate an incomplete sweep.
if [ ${#abandoned[@]} -gt 0 ]; then
  echo "ERROR: ${#abandoned[@]} epoch(s) were not collected" >&2
  exit 1
fi
