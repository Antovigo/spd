#!/usr/bin/env bash
# Run on the POD, detached: the alive-only extraction (skipped once alive_only/meta.json exists),
# then variance.py for blocks 0..31 (each skips the positions already published, and resumes a
# half-done position from its saved accumulators). Stops at the first failure.
#     source /workspace/spd/notes/reader_writer/variance/pod/env.sh && ./run_all.sh [LAYERS...]
# Follow: tail -f $DATA_ROOT/logs/rwv.latest.log
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${RUN_DIR:?source env.sh first}" "${VENV_PY:?source env.sh first}"
if [ "${FOREGROUND:-0}" != 1 ]; then
  LOG="$DATA_ROOT/logs/rwv.$(date +%Y%m%d-%H%M%S).log"
  ln -sfn "$LOG" "$DATA_ROOT/logs/rwv.latest.log"
  FOREGROUND=1 nohup "$0" "$@" > "$LOG" 2>&1 &
  echo "started pid $! -> $LOG"; exit 0
fi
LAYERS=("$@"); [ ${#LAYERS[@]} -gt 0 ] || LAYERS=($(seq 0 31))
OUT="$RUN_DIR/analysis/reader_writer/variance"
cd "$HERE"
echo "$(date -u +%FT%TZ) commit $(git -C "$REPO" log -1 --format=%h)"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
if [ ! -f "$OUT/alive_only/meta.json" ]; then
  echo "$(date -u +%FT%TZ) extraction"
  "$VENV_PY" -u extract_alive_only.py --out "$OUT/alive_only"
fi
for L in "${LAYERS[@]}"; do
  echo "$(date -u +%FT%TZ) block $L"
  "$VENV_PY" -u variance.py --layer "$L" --src "$OUT"
done
echo "$(date -u +%FT%TZ) all done"
