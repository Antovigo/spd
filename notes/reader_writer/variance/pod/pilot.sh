#!/usr/bin/env bash
# Run on the POD, detached: the plan's pilot (§10) on block 0, plus a timing profile of the
# extraction's forward pieces. Writes under $OUT/pilot_src/ and $OUT/profile_tmp/.
#     source /workspace/spd/notes/reader_writer/variance/pod/env.sh && ./pilot.sh
# Follow: tail -f $DATA_ROOT/logs/rwv-pilot.latest.log
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${RUN_DIR:?source env.sh first}" "${VENV_PY:?source env.sh first}"
if [ "${FOREGROUND:-0}" != 1 ]; then
  LOG="$DATA_ROOT/logs/rwv-pilot.$(date +%Y%m%d-%H%M%S).log"
  ln -sfn "$LOG" "$DATA_ROOT/logs/rwv-pilot.latest.log"
  FOREGROUND=1 nohup "$0" "$@" > "$LOG" 2>&1 &
  echo "started pid $! -> $LOG"; exit 0
fi
OUT="$RUN_DIR/analysis/reader_writer/variance"
cd "$HERE"
echo "$(date -u +%FT%TZ) commit $(git -C "$REPO" log -1 --format=%h)"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
"$VENV_PY" -u extract_alive_only.py --profile --out "$OUT/profile_tmp"
"$VENV_PY" -u extract_alive_only.py --limit 6000 --out "$OUT/pilot_src/alive_only"
"$VENV_PY" -u variance.py --layer 0 --pilot --src "$OUT/pilot_src"
echo "$(date -u +%FT%TZ) pilot done"
