#!/usr/bin/env bash
# Run on the POD. Launch the pruning stages detached (closing the shell does not stop them).
#     source /workspace/spd/notes/dual_objective/addsub-all-layers/env.sh
#     ./run.sh [--stages select,validate,...]      # default: every stage up to `prune`
# Follow: tail -f $DATA_ROOT/logs/pruning.latest.log
set -euo pipefail
: "${REPO:?source env.sh first}" "${DATA_ROOT:?source env.sh first}" "${VENV_PY:?source env.sh first}"
NAME="${NAME:-addsub-08-prune}"
STAGES="select,validate,references,screen,combine,estimator_check"
[ "${1:-}" = --stages ] && STAGES="$2"
LOG="$DATA_ROOT/logs/pruning.$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$DATA_ROOT/logs"
cd "$REPO"
echo "code: $(cat "$REPO/COMMIT")" > "$LOG"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv >> "$LOG"
setsid nohup "$VENV_PY" -m param_decomp.pruning.run --config "$DATA_ROOT/pruning/config.yaml" \
  --data_root "$DATA_ROOT" --name "$NAME" --stages "$STAGES" >> "$LOG" 2>&1 < /dev/null &
ln -sfn "$LOG" "$DATA_ROOT/logs/pruning.latest.log"
echo "launched pid $!; log $LOG"
