#!/usr/bin/env bash
# Run ON the pod: the whole integer-only pipeline (notes/integer_only/plan.md), stages 1 to 6.
#     source /workspace/spd/notes/dual_objective/addsub-all-layers/env.sh
#     /workspace/spd/notes/integer_only/pod/run.sh              # detaches; prints the log path
#     /workspace/spd/notes/integer_only/pod/run.sh --foreground # same, in this shell
# Env overrides: RUN_ID (default p-512d73ab), CONFIG (default the addsub-answers-09 config).
# Each stage writes $RUN/integer_only/status/<stage>.done or .failed; a done stage is skipped,
# so rerunning this script resumes. Stage 3 runs one process per λ, λ number i on GPU i. The
# script stops at the first failed stage (a failed `pick` means no λ met the rule: read
# $RUN/integer_only/sweep/summary.tsv). It never touches the pod's other processes.
set -euo pipefail
: "${REPO:?source env.sh first}" "${DATA_ROOT:?source env.sh first}" "${VENV_PY:?source env.sh first}"
RUN_ID="${RUN_ID:-p-512d73ab}"
RUN="$DATA_ROOT/runs/$RUN_ID"
CONFIG="${CONFIG:-notes/integer_only/addsub-answers-09.yaml}"
LOGS="$RUN/integer_only/logs"
mkdir -p "$LOGS"

if [ "${1:-}" != --foreground ]; then
  LOG="$LOGS/pipeline.$(date -u +%Y%m%d-%H%M%S).log"
  setsid nohup "$0" --foreground >> "$LOG" 2>&1 < /dev/null &
  ln -sfn "$LOG" "$LOGS/pipeline.latest.log"
  echo "launched pid $!; log $LOG"
  exit 0
fi

# Bitwise-reproducible GPU kernels: without it the run's bf16 CI fn gives a different max CI on
# the same prompts in another process (smoke: 457 of ~30,000 (position, component) pairs flip at
# 0.1); with it, two processes agree exactly (and the CI pass ran 12% faster on an L40).
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_deterministic_ops=true"
cd "$REPO"
echo "code: $(git rev-parse HEAD)  run: $RUN  config: $CONFIG  start: $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
N_GPU=$(nvidia-smi -L | wc -l)
read -r -a LAMBDAS <<< "$("$VENV_PY" -c "
from param_decomp.integer_only.config import IntegerOnlyConfig
print(*IntegerOnlyConfig.from_file('$CONFIG').lambdas)")"
[ "${#LAMBDAS[@]}" -le "$N_GPU" ] || { echo "${#LAMBDAS[@]} λ values but $N_GPU GPUs"; exit 1; }

stage() {  # stage <name> <CUDA_VISIBLE_DEVICES> [extra args]: one stage, its own log
  local name=$1 gpus=$2 log
  shift 2
  log="$LOGS/$name${1:+_$2}.log"
  echo "=== $name $* on GPU [$gpus] $(date -u +%FT%TZ) -> $log"
  CUDA_VISIBLE_DEVICES="$gpus" "$VENV_PY" -m param_decomp.integer_only.run "$name" \
    --run_dir "$RUN" --data_root "$DATA_ROOT" --config "$CONFIG" "$@" >> "$log" 2>&1
}

stage candidates 0
stage problem 0
PIDS=()
for i in "${!LAMBDAS[@]}"; do
  stage learn "$i" --lam "${LAMBDAS[$i]}" &
  PIDS+=("$!")
done
FAILED=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAILED=1; done
[ "$FAILED" = 0 ] || { echo "a learn process failed: see $LOGS/learn_*.log"; exit 1; }
stage pick 0
stage collect 0
JAX_PLATFORMS=cpu stage applet ""
echo "=== all stages done $(date -u +%FT%TZ): $RUN/integer_only/README.md"
