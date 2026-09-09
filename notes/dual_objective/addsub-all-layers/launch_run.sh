#!/usr/bin/env bash
# Launch (or RESUME) the real run, detached: survives closing the shell and the laptop.
#     source env.sh && ./launch_run.sh [--profile 4xh100]
# Each profile has ONE fixed run id (the trainer enforces `p-<8hex>`): the second and every
# later invocation resumes from the newest checkpoint in $DATA_ROOT/runs/<run id>/ckpts —
# the pinned launch_config.yaml is byte-compared, so the config file must be unchanged.
#     8xh100  addsub-all-layers-sota.yaml         -> p-a1132b01
#     4xh100  addsub-all-layers-4xh100-sota.yaml  -> p-b4132c01
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:?source env.sh first}" "${DATA_ROOT:?source env.sh first}" "${VENV_PY:?source env.sh first}"

# `--profile <name>` picks the pod shape; PROFILES in make_trials.py is the single source of
# truth for which config and run id each one uses. RUN_ID / CONFIG in the environment still
# override, for a one-off.
PROFILE=8xh100
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac; shift
done
read -r P_SOTA P_RUN_ID <<<"$("$VENV_PY" -c "
import sys; sys.path.insert(0, '$HERE')
from make_trials import PROFILES
p = PROFILES['$PROFILE']; print(p['sota'], p['run_id'])")"
RUN_ID="${RUN_ID:-$P_RUN_ID}"
CONFIG="${CONFIG:-$HERE/../$P_SOTA}"
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
echo "stop:    $HERE/stop_run.sh --profile $PROFILE   (SIGTERM -> checkpoint save -> exit)"
