#!/usr/bin/env bash
# Clean stop: SIGTERM the runner, which forwards it to the trainer, which saves a checkpoint
# at the end of the current step and exits. Waits for it. Resume with launch_run.sh.
set -euo pipefail
: "${DATA_ROOT:?source env.sh first}"
RUN_ID="${RUN_ID:-p-a1132b01}"
PIDFILE="$DATA_ROOT/pids/$RUN_ID.runner.pid"
[ -f "$PIDFILE" ] || { echo "no runner pid file for $RUN_ID"; exit 1; }
PID=$(cat "$PIDFILE")
kill -TERM "$PID" 2>/dev/null || { echo "runner $PID not running"; rm -f "$PIDFILE"; exit 0; }
echo "SIGTERM sent to runner $PID; waiting for the trainer to save and exit (a 32-block save is ~20 GB)"
while kill -0 "$PID" 2>/dev/null; do sleep 10; printf .; done
echo; rm -f "$PIDFILE"
echo "stopped. newest checkpoints: $(ls "$DATA_ROOT/runs/$RUN_ID/ckpts" 2>/dev/null | tr '\n' ' ')"
