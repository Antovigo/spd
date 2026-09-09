#!/usr/bin/env bash
# One-screen status of a probe ladder. Read-only; safe to run while the ladder is going.
#
#     source env.sh && ./ladder_status.sh [--profile 4xh100]
#
# The live trial is identified from the RUNNING PROCESS, not from which files exist: logs of
# killed trials stay on disk, and an earlier version of this check reported three trials as
# "running" at once because of them.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${DATA_ROOT:?source env.sh first}" "${VENV_PY:?source env.sh first}"
PROFILE=8xh100
while [ $# -gt 0 ]; do
  case "$1" in --profile) PROFILE="$2"; shift ;; *) echo "unknown arg $1"; exit 2 ;; esac; shift
done
read -r TRIALS_SUB LADDER_SUB <<<"$("$VENV_PY" -c "
import sys; sys.path.insert(0, '$HERE')
from make_trials import PROFILES
p = PROFILES['$PROFILE']; print(p['out'], p['ladder'])")"
TRIALS="$HERE/$TRIALS_SUB"; L="$DATA_ROOT/$LADDER_SUB"

LIVE=$(pgrep -af "run_targeted|run_with_stackdump" | sed -n "s#.*/$TRIALS_SUB/\([^ ]*\)\.yaml.*#\1#p" | head -1)
echo "profile $PROFILE   live trial: ${LIVE:-none}   $(date -Is)"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed 's/^/  gpu /'
printf '%-30s %-9s %s\n' TRIAL STATUS DETAIL
awk -F'\t' '!/^#/{print $1"\t"$2}' "$TRIALS/manifest.tsv" | while IFS=$'\t' read -r name rid; do
  if [ -f "$L/$name.rc" ]; then
    read -r rc secs < "$L/$name.rc"
    printf '%-30s %-9s rc=%s after %smin\n' "$name" done "$rc" "$((secs / 60))"
  elif [ "$name" = "$LIVE" ]; then
    step=$(tail -1 "$DATA_ROOT/runs/$rid/metrics.jsonl" 2>/dev/null | sed -n 's/.*"step": *\([0-9]*\).*/\1/p')
    age=$(( $(date +%s) - $(stat -c %Y "$L/$name.log" 2>/dev/null || date +%s) ))
    printf '%-30s %-9s step=%s, log idle %ss\n' "$name" RUNNING "${step:-startup}" "$age"
  elif [ -f "$L/$name.log" ]; then
    printf '%-30s %-9s (log from a previous attempt)\n' "$name" stale
  else
    printf '%-30s %-9s\n' "$name" pending
  fi
done
