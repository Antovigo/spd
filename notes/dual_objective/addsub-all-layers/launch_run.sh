#!/usr/bin/env bash
# Launch (or RESUME) the real run, detached: survives closing the shell and the laptop.
#     source env.sh && ./launch_run.sh
# Fixed run id p-a1132b01 (the trainer enforces `p-<8hex>`): the second and every later
# invocation resumes from the newest checkpoint in $DATA_ROOT/runs/p-a1132b01/ckpts — the
# pinned launch_config.yaml is byte-compared, so the config file must be unchanged.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:?source env.sh first}" "${DATA_ROOT:?source env.sh first}"
RUN_ID="${RUN_ID:-p-a1132b01}"
CONFIG="${CONFIG:-$HERE/../addsub-all-layers-sota.yaml}"
RUN_NAME="$(awk -F': *' '/^run_name:/{print $2; exit}' "$CONFIG")"
LOG="$DATA_ROOT/logs/$RUN_NAME.$(date +%Y%m%d-%H%M%S).log"

if [ -f "$DATA_ROOT/pids/$RUN_ID.runner.pid" ] && kill -0 "$(cat "$DATA_ROOT/pids/$RUN_ID.runner.pid")" 2>/dev/null; then
  echo "already running: runner pid $(cat "$DATA_ROOT/pids/$RUN_ID.runner.pid")"; exit 1
fi
if [ -d "$DATA_ROOT/runs/$RUN_ID/ckpts" ]; then
  echo "RESUME: checkpoints present: $(ls "$DATA_ROOT/runs/$RUN_ID/ckpts" | tr '\n' ' ')"
fi
export PD_LOG="$LOG" WATCHDOG_FUSE="${WATCHDOG_FUSE:-3600}"
setsid nohup "$HERE/pd_run.sh" "$CONFIG" "$RUN_ID" > "$LOG" 2>&1 < /dev/null &
echo $! > "$DATA_ROOT/pids/$RUN_ID.runner.pid"
ln -sfn "$LOG" "$DATA_ROOT/logs/$RUN_NAME.latest.log"
echo "launched runner pid $(cat "$DATA_ROOT/pids/$RUN_ID.runner.pid"); log: $LOG"
echo "follow:  tail -f $DATA_ROOT/logs/$RUN_NAME.latest.log"
echo "stop:    $HERE/stop_run.sh        (SIGTERM -> checkpoint save -> exit)"
