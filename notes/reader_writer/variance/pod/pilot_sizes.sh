#!/usr/bin/env bash
# Run on the POD after pilot.sh: variance.py --pilot on blocks 0, 31 and 18 at several batch sizes
# (throughput and checks per size; the pilot pool is 6000 prompts, so chunk * batch must stay
# well below it for any chunk after the first, which carries the compile, to be timed), against pilot.sh's 6000-prompt extraction.
#     source /workspace/spd/notes/reader_writer/variance/pod/env.sh && ./pilot_sizes.sh ["L B jB wc K" ...]
# Follow: tail -f $DATA_ROOT/logs/rwv-pilot-sizes.latest.log
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${RUN_DIR:?source env.sh first}" "${VENV_PY:?source env.sh first}"
if [ "${FOREGROUND:-0}" != 1 ]; then
  LOG="$DATA_ROOT/logs/rwv-pilot-sizes.$(date +%Y%m%d-%H%M%S).log"
  ln -sfn "$LOG" "$DATA_ROOT/logs/rwv-pilot-sizes.latest.log"
  FOREGROUND=1 nohup "$0" "$@" > "$LOG" 2>&1 < /dev/null &
  echo "started pid $! -> $LOG"; exit 0
fi
P="$RUN_DIR/analysis/reader_writer/variance/pilot_src"
cd "$HERE"
echo "commit $(git -C "$REPO" log -1 --format=%h)"
CFGS=("31 128 32 16 8" "31 64 32 32 8" "18 128 32 16 8")  # layer batch jbatch wc chunk
[ $# -gt 0 ] && CFGS=("$@")
for cfg in "${CFGS[@]}"; do
  set -- $cfg
  echo "CFG layer=$1 batch=$2 jbatch=$3 wc=$4 chunk=$5"
  rm -rf "$P/pilot/L$1"
  "$VENV_PY" -u variance.py --layer "$1" --pilot --src "$P" --batch "$2" --jbatch "$3" --wc "$4" --chunk "$5" 2>&1 \
    | grep -vE "^WARNING|^W1" || echo "CFG FAILED"
done
echo PILOTS_DONE
