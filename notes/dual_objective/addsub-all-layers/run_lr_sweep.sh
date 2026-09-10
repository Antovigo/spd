#!/usr/bin/env bash
# The LR sweep, unattended, followed by the REAL RUN on the winner.
#
#     source env.sh && ./run_lr_sweep.sh [--arms "lr0.5x lr2x lr0.125x"] [--detach]
#     tail -f $DATA_ROOT/lr-sweep/driver.log
#     cat $DATA_ROOT/lr-sweep/summary.md
#
# For each arm, in order: launch it (or adopt it if already running), wait until it reaches
# PROBE_STEPS, SIGTERM it so it saves, and score it. Arms are authored with `steps: 40000`
# and interrupted, so the winner is simply RESUMED — the pinned launch_config byte-compares
# and the schedules are the real ones. When every arm is scored the winner is relaunched and
# runs to 40000 as the official run; that resume compiles a distinct executable (~25 min,
# measured 2026-09-10) and then caches it.
#
# SCORE = mean of the last `SCORE_WINDOW` logged `train/loss/total` values at or before
# PROBE_STEPS. A window, not the single value at PROBE_STEPS, because step-to-step loss is
# noisy; lower is better. Every arm shares seed 0, so the only difference is the LR.
#
# Survives an arm crashing: that arm scores `nan`, is skipped, and the sweep continues. If
# NO arm scores, nothing is launched — deliberately, since that means something is broken.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:?source env.sh first}" "${DATA_ROOT:?source env.sh first}" "${VENV_PY:?source env.sh first}"

ARMS="lr0.5x lr2x lr0.125x"
PROBE_STEPS="${PROBE_STEPS:-2000}"
SCORE_WINDOW="${SCORE_WINDOW:-5}"
ARM_FUSE="${ARM_FUSE:-18000}"          # 5 h per arm: ~95 min compile + ~77 min to step 2000
DETACH=0
while [ $# -gt 0 ]; do
  case "$1" in
    --arms) ARMS="$2"; shift ;;
    --detach) DETACH=1 ;;
    --probe-steps) PROBE_STEPS="$2"; shift ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac; shift
done

SWEEP_DIR="$DATA_ROOT/lr-sweep"; mkdir -p "$SWEEP_DIR"
DRIVER_LOG="$SWEEP_DIR/driver.log"

if [ "$DETACH" = 1 ]; then
  setsid nohup "$0" --arms "$ARMS" --probe-steps "$PROBE_STEPS" > "$DRIVER_LOG" 2>&1 < /dev/null &
  echo "sweep driver pid $! ; log $DRIVER_LOG ; summary -> $SWEEP_DIR/summary.md"
  exit 0
fi

say() { echo "[$(date -u +%H:%M:%S)] $*" ; }
arm_cfg()    { echo "$HERE/arms-4xh100/$1.yaml"; }
arm_run_id() { awk '/^# run id/{print $4; exit}' "$(arm_cfg "$1")"; }
max_step() {
  "$VENV_PY" - "$1" <<'PY' 2>/dev/null || echo 0
import json, sys
best = 0
try:
    for line in open(sys.argv[1]):
        try: best = max(best, json.loads(line).get("step") or 0)
        except Exception: pass
except FileNotFoundError: pass
print(best)
PY
}
score_arm() {
  "$VENV_PY" - "$1" "$PROBE_STEPS" "$SCORE_WINDOW" <<'PY' 2>/dev/null || echo nan
import json, sys
path, probe, window = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
vals = []
try:
    for line in open(path):
        try: d = json.loads(line)
        except Exception: continue
        s, t = d.get("step"), d.get("train/loss/total")
        if s is not None and t is not None and s <= probe: vals.append((s, t))
except FileNotFoundError: pass
vals.sort()
tail = [t for _, t in vals[-window:]]
print(f"{sum(tail)/len(tail):.6f}" if tail else "nan")
PY
}

say "sweep start: arms='$ARMS' probe=$PROBE_STEPS window=$SCORE_WINDOW"
declare -A SCORES
for arm in $ARMS; do
  cfg="$(arm_cfg "$arm")"; rid="$(arm_run_id "$arm")"; m="$DATA_ROOT/runs/$rid/metrics.jsonl"
  if [ ! -f "$cfg" ]; then say "$arm: NO CONFIG at $cfg — skipped"; continue; fi
  pidf="$DATA_ROOT/pids/$rid.runner.pid"

  if [ "$(max_step "$m")" -ge "$PROBE_STEPS" ]; then
    say "$arm ($rid): already at step $(max_step "$m") — not relaunching"
  else
    if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
      say "$arm ($rid): already running as pid $(cat "$pidf") — adopting"
    else
      say "$arm ($rid): launching"
      CONFIG="$cfg" RUN_ID="$rid" WATCHDOG_FUSE=14400 "$HERE/launch_run.sh" --profile 4xh100 \
        >> "$DRIVER_LOG" 2>&1 || { say "$arm: launch FAILED"; SCORES[$arm]=nan; continue; }
      sleep 20
    fi
    deadline=$(( $(date +%s) + ARM_FUSE ))
    while :; do
      s="$(max_step "$m")"
      [ "${s:-0}" -ge "$PROBE_STEPS" ] && { say "$arm: reached step $s"; break; }
      if [ ! -f "$pidf" ] || ! kill -0 "$(cat "$pidf" 2>/dev/null)" 2>/dev/null; then
        say "$arm: runner exited at step ${s:-0} before $PROBE_STEPS"; break
      fi
      [ "$(date +%s)" -ge "$deadline" ] && { say "$arm: FUSE ${ARM_FUSE}s at step ${s:-0}"; break; }
      sleep 60
    done
  fi

  if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then
    say "$arm: SIGTERM (saves and exits)"
    kill -TERM "$(cat "$pidf")" 2>/dev/null
    for _ in $(seq 1 40); do kill -0 "$(cat "$pidf")" 2>/dev/null || break; sleep 15; done
  fi
  SCORES[$arm]="$(score_arm "$m")"
  say "$arm: score=${SCORES[$arm]} (mean of last $SCORE_WINDOW totals <= $PROBE_STEPS)"
done

{
  echo "# LR sweep — profile 4xh100, probe $PROBE_STEPS steps"
  echo
  echo "score = mean of the last $SCORE_WINDOW logged \`train/loss/total\` at or before step $PROBE_STEPS; lower is better."
  echo
  echo "| arm | run id | score | ckpt |"
  echo "|---|---|---|---|"
  for arm in $ARMS; do
    rid="$(arm_run_id "$arm")"
    echo "| $arm | $rid | ${SCORES[$arm]:-not run} | $(ls "$DATA_ROOT/runs/$rid/ckpts" 2>/dev/null | tr '\n' ' ') |"
  done
} > "$SWEEP_DIR/summary.md"

WINNER=""; BEST=""
for arm in $ARMS; do
  s="${SCORES[$arm]:-nan}"
  [ "$s" = nan ] && continue
  if [ -z "$BEST" ] || awk "BEGIN{exit !($s < $BEST)}"; then BEST="$s"; WINNER="$arm"; fi
done

if [ -z "$WINNER" ]; then
  say "NO arm produced a score — not launching the real run. Inspect $DRIVER_LOG."
  echo >> "$SWEEP_DIR/summary.md"; echo "**No winner: every arm failed to score.**" >> "$SWEEP_DIR/summary.md"
  exit 1
fi

say "WINNER: $WINNER (score $BEST)"

# The official run DOUBLES C (all 32 blocks — C is tiled by schema, see make_lr_arm.py), so
# it cannot resume the winning arm: every parameter shape changes. It starts fresh on the
# profile's own run id and pays one cold compile (~95 min).
FINAL_CFG="$HERE/arms-4xh100/final-C2x-$WINNER.yaml"
FINAL_ID="p-b4132c01"
if [ ! -f "$FINAL_CFG" ]; then
  say "no doubled-C config for $WINNER at $FINAL_CFG — falling back to RESUMING the arm"
  CONFIG="$(arm_cfg "$WINNER")" RUN_ID="$(arm_run_id "$WINNER")" WATCHDOG_FUSE=14400 \
    "$HERE/launch_run.sh" --profile 4xh100 >> "$DRIVER_LOG" 2>&1
  say "official run (C1x, resumed) launched on $(arm_run_id "$WINNER")"; exit 0
fi

say "launching the official run: $WINNER at DOUBLED C, run id $FINAL_ID"
{ echo; echo "**Winner: \`$WINNER\`** (score $BEST). Official run = that LR at **doubled C** (all 32 blocks), run id \`$FINAL_ID\`."; } >> "$SWEEP_DIR/summary.md"
CONFIG="$FINAL_CFG" RUN_ID="$FINAL_ID" WATCHDOG_FUSE=14400 \
  "$HERE/launch_run.sh" --profile 4xh100 >> "$DRIVER_LOG" 2>&1

# WALK-BACK: doubled C is a memory gamble (est. 47-50 GB/rank of 80, from 34.5 measured at
# C1x). If it dies of OOM, resume the winning arm at the authored C instead of losing the
# night. Watch long enough to cover the cold compile plus the first steps.
FINAL_LOG="$DATA_ROOT/logs/$(awk -F': *' '/^run_name:/{print $2; exit}' "$FINAL_CFG").latest.log"
deadline=$(( $(date +%s) + 10800 ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  sleep 120
  pidf="$DATA_ROOT/pids/$FINAL_ID.runner.pid"
  if [ ! -f "$pidf" ] || ! kill -0 "$(cat "$pidf" 2>/dev/null)" 2>/dev/null; then
    if grep -qE "RESOURCE_EXHAUSTED|Out of memory|OOM|failed to allocate" "$FINAL_LOG" 2>/dev/null; then
      say "official run OOMed at doubled C — WALKING BACK to the authored C and resuming $WINNER"
      { echo; echo "**Doubled C OOMed; walked back to the authored C** (resumed \`$WINNER\`)."; } >> "$SWEEP_DIR/summary.md"
      CONFIG="$(arm_cfg "$WINNER")" RUN_ID="$(arm_run_id "$WINNER")" WATCHDOG_FUSE=14400 \
        "$HERE/launch_run.sh" --profile 4xh100 >> "$DRIVER_LOG" 2>&1
      say "official run (C1x, resumed) launched on $(arm_run_id "$WINNER")"
    else
      say "official run exited without an OOM signature — NOT relaunching; inspect $FINAL_LOG"
    fi
    exit 0
  fi
  if [ "$(max_step "$DATA_ROOT/runs/$FINAL_ID/metrics.jsonl")" -ge 100 ]; then
    say "official run is training at doubled C — walk-back watch ends"; exit 0
  fi
done
say "walk-back watch timed out; official run still up"
