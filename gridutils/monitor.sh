#!/bin/bash
#
# monitor.sh -- one-shot status snapshot for a set of training runs:
#   condor state (via the +JobBatchName tag written by train/submit.sh),
#   epoch progress (last `[timing] epoch=` line in the streamed condor .out),
#   and the latest checkpoint synced to GPFS (visible mid-run thanks to
#   trainjob.sh's periodic sync).
#
# Usage (run on the submit node):
#   bash gridutils/monitor.sh                      # all dino-* batches in the queue
#   bash gridutils/monitor.sh run_name1 run_name2  # explicit run_name list
#
# This is a point-in-time read of condor_q + GPFS; there is no daemon.
# Re-run it for a fresh snapshot, or wrap it: watch bash gridutils/monitor.sh

set -uo pipefail

CONDOR_OUT="${CONDOR_OUT:-/gpfs01/lbne/users/fm/${USER}/CONDOR_OUT}"

RUNS=("$@")

# One queue read, reused for every row. Jobs submitted by train/submit.sh carry
# JobBatchName = "dino-<run_name>"; older jobs are matched by their trailing
# run_name argument as a fallback.
Q_BATCH=$(condor_q "$USER" -af:j JobBatchName JobStatus 2>/dev/null || true)
Q_ARGS=$(condor_q  "$USER" -af:j JobStatus Args        2>/dev/null || true)

if [ ${#RUNS[@]} -eq 0 ]; then
  mapfile -t RUNS < <(printf '%s\n' "$Q_BATCH" | awk '$2 ~ /^dino-/ {sub(/^dino-/,"",$2); print $2}' | sort -u)
fi
if [ ${#RUNS[@]} -eq 0 ]; then
  echo "No dino-* batches in the queue and no run names given."
  echo "usage: $0 [run_name ...]"
  exit 0
fi

status_name() {
  case "$1" in
    1) echo idle ;; 2) echo running ;; 3) echo removed ;;
    4) echo done ;; 5) echo HELD ;; 6) echo xfer ;; 7) echo suspended ;;
    *) echo "status$1" ;;
  esac
}

printf "%-40s %-15s %-9s %s\n" "run_name" "condor" "epoch" "last_ckpt_synced"
printf "%-40s %-15s %-9s %s\n" "--------" "------" "-----" "----------------"

for run in "${RUNS[@]}"; do
  # -- condor state: exact batch-name match first, args fallback for old jobs --
  line=$(printf '%s\n' "$Q_BATCH" | awk -v b="dino-${run}" '$2==b {print $1, $3; exit}')
  if [ -z "$line" ]; then
    line=$(printf '%s\n' "$Q_ARGS" | awk -v r="$run" '$NF==r {print $1, $2; exit}')
  fi
  if [ -n "$line" ]; then
    cluster=${line%% *}; cluster=${cluster%.*}
    state="$(status_name "${line##* }")(${cluster})"
  else
    state="not-queued"
  fi

  rundir="${CONDOR_OUT}/${run}"

  # -- epoch progress from the streamed condor .out ([timing] lines) --
  epoch="-"
  out=$(ls -t "${rundir}"/*.out 2>/dev/null | head -1)
  if [ -n "$out" ]; then
    last=$(grep -E '^\[timing\] epoch=' "$out" 2>/dev/null | tail -1)
    [ -n "$last" ] && epoch=$(sed -E 's/^\[timing\] epoch=([0-9]+).*/\1/' <<<"$last")
  fi
  total=$(jq -r '.epochs // empty' "${rundir}/debug/run_config.json" 2>/dev/null || true)
  [ "$epoch" != "-" ] && [ -n "$total" ] && epoch="${epoch}/${total}"

  # -- latest checkpoint synced to GPFS --
  ck=$(ls -t "${rundir}"/checkpoints/checkpoint_epoch*.pt 2>/dev/null | head -1)
  if [ -n "$ck" ]; then
    ck_info="ep$(basename "$ck" | grep -oE '[0-9]+')  $(date -r "$ck" '+%Y-%m-%d %H:%M')"
  else
    ck_info="(none yet)"
  fi

  printf "%-40s %-15s %-9s %s\n" "$run" "$state" "$epoch" "$ck_info"
done
