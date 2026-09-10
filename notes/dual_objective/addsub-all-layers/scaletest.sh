#!/usr/bin/env bash
# Run ONE short trial at a reduced block count, to find where a full-network run stops
# working. Diagnostic, not part of the ladder.
#
#     source env.sh && ./scaletest.sh --profile 4xh100 --blocks 4
#     ./scaletest.sh --profile 4xh100 --blocks 8 --start 18
#     ./scaletest.sh --blocks 4 --env NCCL_DEBUG=INFO --env NCCL_DEBUG_SUBSYS=INIT,COLL
#     kill -USR1 $(pgrep -f run_with_stackdump) # dump every thread's stack into the log
#
# WHY. The 4xh100 ladder's 32-block trials stalled about a minute in, after allocating the
# memory pool, with no CPU, no GPU and no output — and a bare multi-GPU JAX collective ran
# fine on the same pod, so the machine is not the problem. This bisects the remaining
# variable: 4 blocks (28 sites) is a shape that has completed real runs, 32 (224 sites) is
# the one that stalls. Everything else is held byte-identical to the profile's config.
#
# The generated config differs from the production one ONLY in: layers, run_name, pd.steps
# (30), train_log_every (10), no wandb, no checkpointing. It runs through
# run_with_stackdump.py, so `kill -USR1` gives a Python stack without ptrace.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:?source env.sh first}" "${DATA_ROOT:?source env.sh first}" "${VENV_PY:?source env.sh first}"

# --start defaults to 18, matching the reference recipes (addsub-L18-*, addsub-4L18-21-*),
# so a small --blocks reproduces a shape that has completed real runs. Pass --start
# explicitly to place the window elsewhere; the tail below the window is what the frozen
# suffix has to run on every masked forward, so lower windows cost more.
PROFILE=4xh100; BLOCKS=4; START=18; STEPS=30; ENV=()
while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; shift ;;
    --blocks) BLOCKS="$2"; shift ;;
    --start) START="$2"; shift ;;
    --steps) STEPS="$2"; shift ;;
    --env) ENV+=("$2"); shift ;;   # K=V, repeatable; lands in runtime.launch_env.env
    *) echo "unknown arg $1"; exit 2 ;;
  esac; shift
done

OUT="$DATA_ROOT/scaletest"; mkdir -p "$OUT"
CFG="$OUT/${BLOCKS}L.yaml"; LOG="$OUT/${BLOCKS}L.log"
RUN_ID=$("$VENV_PY" -c "
import hashlib
print('p-' + hashlib.blake2b(f'scaletest/$PROFILE/$BLOCKS/$START'.encode(), digest_size=4).hexdigest())")

"$VENV_PY" "$HERE/scaletest_config.py" "$PROFILE" "$BLOCKS" "$START" "$STEPS" "$CFG" "${ENV[@]:-}"

N_DEV=$("$VENV_PY" -c "
import yaml; r = yaml.safe_load(open('$CFG'))['runtime']; print(r['replicate'] * r['fsdp'] * r['tp'])")
cd "$REPO"
echo "scaletest: $BLOCKS blocks, $N_DEV GPU, run_id $RUN_ID"
echo "config $CFG"
echo "log    $LOG"
rm -rf "$DATA_ROOT/runs/$RUN_ID"
JAX_LOG_COMPILES=1 setsid nohup "$VENV_PY" "$HERE/run_with_stackdump.py" \
  "$CFG" "$DATA_ROOT" "$N_DEV" --run_id "$RUN_ID" > "$LOG" 2>&1 &
echo "pid $! ; follow with: tail -f $LOG"
echo "if it stalls:  kill -USR1 \$(pgrep -f run_with_stackdump | head -1)  then re-read the log"
