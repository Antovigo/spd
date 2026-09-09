#!/usr/bin/env bash
# The probe ladder, unattended. Runs trials in order, sequentially, each under `timeout -k`
# (a timeout or hang counts as "does not fit"), skips trials whose result file exists, keeps
# the XLA compilation cache on the volume (env.sh), and writes ONE markdown table.
#
#     source env.sh && ./run_ladder.sh --profile 4xh100 --detach
#     tail -f $DATA_ROOT/ladder-4xh100/driver.log
#     cat $DATA_ROOT/ladder-4xh100/summary.md          # paste this back
#
# A PROFILE is one pod shape (see PROFILES in make_trials.py, the single source of truth for
# which config, meshes, trials dir, ladder dir and run ids it uses):
#     8xh100   addsub-all-layers-sota.yaml         meshes c (2x4) and b (1x8), batch 256/128
#     4xh100   addsub-all-layers-4xh100-sota.yaml  meshes f4 (1x4) and r2f2 (2x2), batch 128/128
#
# Order:
#   1. every default mesh at the config's AUTHORED batch, 30 steps, step-0 slow eval ON
#   2. if none of them fit, every default mesh again at the FALLBACK batch (128/96), because
#      "does not fit" is a memory answer and shrinking the broad stream is the lever
#   3. AB-grid smoke at the winning (mesh, batch): every 10 / slow 20, mean_ci_floor 0.0,
#      grid [1,10]^2, checkpointing ON. Pass = step_20.js written, saved_components > 0, no
#      traceback
#   4. SIGTERM/resume smoke at the same shape: save_every 20, TERM after the step-20 log,
#      relaunch on the same run id, must resume and finish at 30
#
# `--extra <trial>` appends any manifest trial (e.g. mesh-ddp-b128x128). `--only <trial>`
# runs exactly one. Rerun after a fix: delete `<ladder>/<trial>.rc` AND the trial's run dir
# (`$DATA_ROOT/runs/<run id>`, ids in the profile's manifest.tsv) — run ids are fixed, so a
# leftover run dir would either resume or refuse on the pinned config.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:?source env.sh first}" "${DATA_ROOT:?source env.sh first}" "${VENV_PY:?source env.sh first}"

PROFILE=8xh100; EXTRA=(); ONLY=""; DETACH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift ;;
    --detach) DETACH=1 ;;
    --extra) EXTRA+=("$2"); shift ;;
    --only) ONLY="$2"; shift ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac; shift
done

# Ask make_trials.py where this profile lives, so the mapping has ONE definition.
read -r TRIALS_SUB LADDER_SUB <<<"$("$VENV_PY" -c "
import sys; sys.path.insert(0, '$HERE')
from make_trials import PROFILES
p = PROFILES['$PROFILE']; print(p['out'], p['ladder'])")"
TRIALS="$HERE/$TRIALS_SUB"; LADDER="$DATA_ROOT/$LADDER_SUB"; MANIFEST="$TRIALS/manifest.tsv"
[ -f "$MANIFEST" ] || { echo "no manifest at $MANIFEST — run: $VENV_PY make_trials.py --profile $PROFILE"; exit 2; }
mkdir -p "$LADDER"

if [ "$DETACH" -eq 1 ]; then
  ARGS=(--profile "$PROFILE"); [ -n "$ONLY" ] && ARGS+=(--only "$ONLY")
  for e in "${EXTRA[@]:-}"; do [ -n "$e" ] && ARGS+=(--extra "$e"); done
  setsid nohup "$HERE/run_ladder.sh" "${ARGS[@]}" >> "$LADDER/driver.log" 2>&1 < /dev/null &
  echo "ladder driver pid $! ; log $LADDER/driver.log ; summary -> $LADDER/summary.md"; exit 0
fi

meta() { awk -F'\t' -v k="$1" '$0 ~ /^#/ { sub(/^# */, ""); if ($1 == k) { print $2; exit } }' "$MANIFEST"; }
field() { awk -F'\t' -v n="$1" -v c="$2" '$1==n{print $c; exit}' "$MANIFEST"; }
AUTHORED_BATCH=$(meta authored_batch); FALLBACK_BATCH=$(meta fallback_batch)
IFS=',' read -r -a MESHES <<< "$(meta default_meshes)"

summarize() { "$VENV_PY" "$HERE/summarize_ladder.py" --profile "$PROFILE" --data-root "$DATA_ROOT" "$@"; }

reap_gpus() {  # a killed hung trial can hold HBM for minutes; the next trial must not start on it
  for _ in $(seq 1 30); do
    PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' || true)
    [ -z "$PIDS" ] && return 0
    echo "GPU compute processes still present ($PIDS) — kill -9 and wait"
    for p in $PIDS; do kill -9 "$p" 2>/dev/null || true; done
    sleep 10
  done
  echo "=== GPUs still busy after 5 min; giving up ==="; return 1
}

run_trial() {  # run_trial <trial>  -> <ladder>/<trial>.rc = "<rc> <seconds>"
  local NAME="$1" RUN_ID CONFIG TIMEOUT LOG T0 RC
  if [ -f "$LADDER/$NAME.rc" ]; then echo "[$NAME] done already ($(cat "$LADDER/$NAME.rc")) — skip"; return 0; fi
  RUN_ID=$(field "$NAME" 2); CONFIG="$TRIALS/$(field "$NAME" 3)"; TIMEOUT=$(field "$NAME" 5)
  [ -n "$RUN_ID" ] || { echo "[$NAME] not in $MANIFEST"; return 2; }
  LOG="$LADDER/$NAME.log"
  echo "[$NAME] $(date -Is) start: $CONFIG run_id=$RUN_ID timeout=${TIMEOUT}s log=$LOG"
  reap_gpus || true
  T0=$(date +%s)
  set +e
  PD_LOG="$LOG" WATCHDOG_FUSE="${WATCHDOG_FUSE:-2700}" \
    timeout -k 600 "$TIMEOUT" "$HERE/pd_run.sh" "$CONFIG" "$RUN_ID" >> "$LOG" 2>&1
  RC=$?
  set -e
  echo "$RC $(( $(date +%s) - T0 ))" > "$LADDER/$NAME.rc"
  echo "[$NAME] $(date -Is) rc=$RC after $(( ($(date +%s) - T0) / 60 )) min"
}

run_resume_trial() {  # two legs on one run id; SIGTERM the trainer once step 20 is logged
  local NAME="$1" RUN_ID CONFIG TIMEOUT LOG T0 RC RC1 RC2 RUN_DIR PIDF LEG1
  if [ -f "$LADDER/$NAME.rc" ]; then echo "[$NAME] done already ($(cat "$LADDER/$NAME.rc")) — skip"; return 0; fi
  RUN_ID=$(field "$NAME" 2); CONFIG="$TRIALS/$(field "$NAME" 3)"; TIMEOUT=$(field "$NAME" 5)
  LOG="$LADDER/$NAME.log"; RUN_DIR="$DATA_ROOT/runs/$RUN_ID"; PIDF="$DATA_ROOT/pids/$RUN_ID.trainer.pid"
  echo "[$NAME] $(date -Is) leg 1 start (will SIGTERM after the step-20 log)"
  reap_gpus || true
  T0=$(date +%s)
  set +e
  PD_LOG="$LOG" WATCHDOG_FUSE="${WATCHDOG_FUSE:-2700}" \
    timeout -k 600 "$TIMEOUT" "$HERE/pd_run.sh" "$CONFIG" "$RUN_ID" >> "$LOG" 2>&1 &
  LEG1=$!
  # The step-20 record is written right before the step-20 save; a TERM landing during that
  # save sets the trainer's flag, step 21 runs, then it saves again and exits — so the
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
  PD_LOG="$LOG" WATCHDOG_FUSE="${WATCHDOG_FUSE:-2700}" \
    timeout -k 600 "$TIMEOUT" "$HERE/pd_run.sh" "$CONFIG" "$RUN_ID" >> "$LOG" 2>&1
  RC2=$?
  set -e
  RC=$(( RC1 != 0 ? RC1 : RC2 ))
  echo "$RC $(( $(date +%s) - T0 ))" > "$LADDER/$NAME.rc"
  echo "[$NAME] $(date -Is) leg 2 rc=$RC2 (combined $RC) after $(( ($(date +%s) - T0) / 60 )) min"
}

dispatch() { case "$1" in smoke-resume-*) run_resume_trial "$1" ;; *) run_trial "$1" ;; esac; }

echo "=== ladder $(date -Is) profile=$PROFILE repo $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo tarball) data_root $DATA_ROOT ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | head -8

if [ -n "$ONLY" ]; then
  dispatch "$ONLY"
else
  for m in "${MESHES[@]}"; do run_trial "mesh-$m-$AUTHORED_BATCH"; done
  WINNER=$(summarize --choose-mesh --batch-tag "$AUTHORED_BATCH")
  if [ "$WINNER" = "none" ]; then
    echo "=== nothing fits at the authored batch $AUTHORED_BATCH — escalating to $FALLBACK_BATCH ==="
    for m in "${MESHES[@]}"; do run_trial "mesh-$m-$FALLBACK_BATCH"; done
    WINNER=$(summarize --choose-mesh --batch-tag "$FALLBACK_BATCH")
  fi
  echo "=== winning (mesh, batch): $WINNER ==="
  if [ "$WINNER" = "none" ]; then
    echo "=== nothing fits at either batch. Read $LADDER/summary.md; the next levers are"
    echo "    a smaller nontarget batch, or eval.batch_size / PGDReconLoss.n_batches. ==="
  else
    run_trial "smoke-abgrid-$WINNER"
    run_resume_trial "smoke-resume-$WINNER"
  fi
  for e in "${EXTRA[@]:-}"; do [ -n "$e" ] && dispatch "$e"; done
fi
summarize --write "$LADDER/summary.md" > /dev/null
echo "=== ladder done $(date -Is); summary: $LADDER/summary.md ==="
cat "$LADDER/summary.md"
