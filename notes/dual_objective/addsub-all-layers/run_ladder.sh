#!/usr/bin/env bash
# The probe ladder, unattended. Runs the trials in order, sequentially, each under
# `timeout -k` (a timeout or hang = "does not fit"), skips trials whose result file exists,
# keeps the XLA compilation cache on the volume (env.sh), and writes ONE markdown table.
#
#     source env.sh && ./run_ladder.sh --detach          # survives shell disconnect
#     tail -f $DATA_ROOT/ladder/driver.log
#     cat $DATA_ROOT/ladder/summary.md                   # paste this back
#
# Default order (Antoine 2026-09-08: hard ~1 h target, so no 128/96 fallback rung):
#   1. mesh-c   replicate 2 / fsdp 4 / zero1, batch 256/128, 30 steps, step-0 slow eval on
#   2. mesh-b   replicate 1 / fsdp 8 / zero1, same
#   3. smoke-abgrid-<best>   AB-grid smoke at the fastest fitting mesh (every 10 / slow 20,
#                            floor 0.0, grid [1,10]^2; checkpointing ON)
#   4. smoke-resume-<best>   SIGTERM after the step-20 log -> save -> relaunch -> finish
# `--extra <trial>` appends manifest trials (e.g. fallback-128x96-c). `--only <trial>` runs one.
# Rerun after a fix: delete $DATA_ROOT/ladder/<trial>.rc AND the trial's run dir
# ($DATA_ROOT/runs/<run id>, see trials/manifest.tsv) — the run id is fixed, and a leftover
# run dir would either resume or refuse on the pinned config.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:?source env.sh first}" "${DATA_ROOT:?source env.sh first}" "${VENV_PY:?source env.sh first}"
LADDER="$DATA_ROOT/ladder"; TRIALS="$HERE/trials"; MANIFEST="$TRIALS/manifest.tsv"
mkdir -p "$LADDER"

EXTRA=(); ONLY=""; DETACH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --detach) DETACH=1 ;;
    --extra) EXTRA+=("$2"); shift ;;
    --only) ONLY="$2"; shift ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac; shift
done
if [ "$DETACH" -eq 1 ]; then
  ARGS=(); [ -n "$ONLY" ] && ARGS+=(--only "$ONLY"); for e in "${EXTRA[@]:-}"; do [ -n "$e" ] && ARGS+=(--extra "$e"); done
  setsid nohup "$HERE/run_ladder.sh" "${ARGS[@]}" >> "$LADDER/driver.log" 2>&1 < /dev/null &
  echo "ladder driver pid $! ; log $LADDER/driver.log ; summary -> $LADDER/summary.md"; exit 0
fi

manifest_field() {  # manifest_field <trial> <col: 2 run_id | 3 config | 4 kind | 5 timeout>
  awk -F'\t' -v n="$1" -v c="$2" '$1==n{print $c; exit}' "$MANIFEST"
}
reap_gpus() {  # a killed hung trial can hold HBM for minutes; the next trial must not start on it
  for _ in $(seq 1 30); do
    PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' || true)
    [ -z "$PIDS" ] && return 0
    echo "GPU compute processes still present ($PIDS) — kill -9 and wait"; for p in $PIDS; do kill -9 "$p" 2>/dev/null || true; done
    sleep 10
  done
  echo "=== GPUs still busy after 5 min; giving up ==="; return 1
}
run_trial() {  # run_trial <trial>   -> $LADDER/<trial>.rc = "<rc> <seconds>"
  local NAME="$1" RUN_ID CONFIG TIMEOUT LOG T0 RC
  if [ -f "$LADDER/$NAME.rc" ]; then echo "[$NAME] done already ($(cat "$LADDER/$NAME.rc")) — skip"; return 0; fi
  RUN_ID=$(manifest_field "$NAME" 2); CONFIG="$TRIALS/$(manifest_field "$NAME" 3)"; TIMEOUT=$(manifest_field "$NAME" 5)
  [ -n "$RUN_ID" ] || { echo "[$NAME] not in manifest"; return 2; }
  LOG="$LADDER/$NAME.log"
  echo "[$NAME] $(date -Is) start: $CONFIG run_id=$RUN_ID timeout=${TIMEOUT}s log=$LOG"
  reap_gpus || true
  T0=$(date +%s)
  set +e
  PD_LOG="$LOG" WATCHDOG_FUSE="${WATCHDOG_FUSE:-2700}" timeout -k 600 "$TIMEOUT" "$HERE/pd_run.sh" "$CONFIG" "$RUN_ID" >> "$LOG" 2>&1
  RC=$?
  set -e
  echo "$RC $(( $(date +%s) - T0 ))" > "$LADDER/$NAME.rc"
  echo "[$NAME] $(date -Is) rc=$RC after $(( ($(date +%s) - T0) / 60 )) min"
}
run_resume_trial() {  # two legs on one run id; SIGTERM the trainer once step 20 is logged
  local NAME="$1" RUN_ID CONFIG TIMEOUT LOG T0 RC1 RC2 RUN_DIR PIDF
  if [ -f "$LADDER/$NAME.rc" ]; then echo "[$NAME] done already ($(cat "$LADDER/$NAME.rc")) — skip"; return 0; fi
  RUN_ID=$(manifest_field "$NAME" 2); CONFIG="$TRIALS/$(manifest_field "$NAME" 3)"; TIMEOUT=$(manifest_field "$NAME" 5)
  LOG="$LADDER/$NAME.log"; RUN_DIR="$DATA_ROOT/runs/$RUN_ID"; PIDF="$DATA_ROOT/pids/$RUN_ID.trainer.pid"
  echo "[$NAME] $(date -Is) leg 1 start (will SIGTERM after the step-20 log)"
  reap_gpus || true
  T0=$(date +%s)
  set +e
  PD_LOG="$LOG" WATCHDOG_FUSE="${WATCHDOG_FUSE:-2700}" timeout -k 600 "$TIMEOUT" "$HERE/pd_run.sh" "$CONFIG" "$RUN_ID" >> "$LOG" 2>&1 &
  local LEG1=$!
  # wait for the step-20 train record (log every 10; save_every 20 has just written a ckpt)
  # The step-20 record is written right before the step-20 save; a TERM landing during
  # that save sets the trainer's flag, step 21 runs, then it saves again and exits — so the
  # resume starts from 21 (or 20 if the flag beat the save), never from 30.
  while kill -0 "$LEG1" 2>/dev/null; do
    if [ -f "$RUN_DIR/metrics.jsonl" ] && grep -q '"step": 20,' "$RUN_DIR/metrics.jsonl"; then
      echo "[$NAME] step 20 logged -> SIGTERM trainer $(cat "$PIDF" 2>/dev/null)"
      kill -TERM "$(cat "$PIDF")" 2>/dev/null || true
      break
    fi
    sleep 2
  done
  wait "$LEG1"; RC1=$?
  echo "[$NAME] leg 1 rc=$RC1; ckpts: $(ls "$RUN_DIR/ckpts" 2>/dev/null | tr '\n' ' ')"
  echo "[$NAME] $(date -Is) leg 2 start (resume on the same run id)"
  reap_gpus || true
  PD_LOG="$LOG" WATCHDOG_FUSE="${WATCHDOG_FUSE:-2700}" timeout -k 600 "$TIMEOUT" "$HERE/pd_run.sh" "$CONFIG" "$RUN_ID" >> "$LOG" 2>&1
  RC2=$?
  set -e
  RC=$(( RC1 != 0 ? RC1 : RC2 ))
  echo "$RC $(( $(date +%s) - T0 ))" > "$LADDER/$NAME.rc"
  echo "[$NAME] $(date -Is) leg 2 rc=$RC2 (combined $RC) after $(( ($(date +%s) - T0) / 60 )) min"
}

echo "=== ladder $(date -Is) repo $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo tarball) data_root $DATA_ROOT ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | head -8

if [ -n "$ONLY" ]; then
  case "$ONLY" in smoke-resume-*) run_resume_trial "$ONLY" ;; *) run_trial "$ONLY" ;; esac
else
  run_trial mesh-c
  run_trial mesh-b
  BEST=$("$VENV_PY" "$HERE/summarize_ladder.py" --data-root "$DATA_ROOT" --choose-mesh)
  echo "=== fastest fitting mesh: $BEST ==="
  if [ "$BEST" = "none" ]; then
    echo "=== neither mesh fits at 256/128; the smokes need a shape. Run by hand:"
    echo "    ./run_ladder.sh --only fallback-128x96-c   (or -b), then --only smoke-abgrid-<m>"
  else
    run_trial "smoke-abgrid-$BEST"
    run_resume_trial "smoke-resume-$BEST"
  fi
  for e in "${EXTRA[@]:-}"; do
    [ -n "$e" ] || continue
    case "$e" in smoke-resume-*) run_resume_trial "$e" ;; *) run_trial "$e" ;; esac
  done
fi
"$VENV_PY" "$HERE/summarize_ladder.py" --data-root "$DATA_ROOT" --write "$LADDER/summary.md" > /dev/null
echo "=== ladder done $(date -Is); summary: $LADDER/summary.md ==="
cat "$LADDER/summary.md"
