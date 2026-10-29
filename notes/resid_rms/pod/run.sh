#!/usr/bin/env bash
# Run on the POD, detached: a 2,000-prompt pilot (checked against the reference KLs), then the full
# collection. Stops at the first failure.
#     source /workspace/spd/notes/resid_rms/pod/env.sh && [BATCH=200] ./run.sh
# Follow: tail -f $DATA_ROOT/logs/resid-rms.latest.log
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${RUN_DIR:?source env.sh first}" "${VENV_PY:?source env.sh first}"
if [ "${FOREGROUND:-0}" != 1 ]; then
  LOG="$DATA_ROOT/logs/resid-rms.$(date +%Y%m%d-%H%M%S).log"
  ln -sfn "$LOG" "$DATA_ROOT/logs/resid-rms.latest.log"
  FOREGROUND=1 nohup "$0" "$@" > "$LOG" 2>&1 < /dev/null &
  echo "started pid $! -> $LOG"; exit 0
fi
OUT="$RUN_DIR/analysis/resid_rms"
cd "$HERE"
echo "$(date -u +%FT%TZ) commit $(git -C "$REPO" log -1 --format=%h)"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
if [ "${SKIP_PILOT:-0}" != 1 ]; then
  rm -rf "$OUT-pilot"
  "$VENV_PY" -u collect.py --limit 2000 --batch_size "${BATCH:-200}" --out "$OUT-pilot"
fi
"$VENV_PY" -u collect.py --batch_size "${BATCH:-200}" --out "$OUT"
echo "$(date -u +%FT%TZ) all done"
