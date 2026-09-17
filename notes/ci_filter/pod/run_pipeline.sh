#!/usr/bin/env bash
# CI filter pipeline on ONE GPU pod: smoke -> objective 1 -> objective 2, detached.
#     source /workspace/spd/notes/dual_objective/addsub-all-layers/env.sh
#     ./run_pipeline.sh [--no-smoke] [--only obj1|obj2] [--init-id cf-xxxxxxxx]
# Stages run in order and stop at the first failure. The smoke writes to the fixed id
# `cf-smoke` and is deleted when it passes. Objective 2 starts from objective 1's CI fn: from
# the id this pipeline just produced, or from --init-id when run with `--only obj2`.
# Hardware knobs (env): MICRO, EVAL_BATCH, GRID_CHUNK — see make_config.py.
# Filter ids (output dir + wandb run name/id): OBJ1_ID, OBJ2_ID; they fail closed if taken.
# Follow: tail -f $DATA_ROOT/logs/ci_filter.latest.log
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:?source env.sh first}" "${DATA_ROOT:?source env.sh first}" "${VENV_PY:?source env.sh first}"

SMOKE=1; ONLY=""; INIT_ID=""
while [ $# -gt 0 ]; do
  case "$1" in
    --no-smoke) SMOKE=0 ;;
    --only) ONLY="$2"; shift ;;
    --init-id) INIT_ID="$2"; shift ;;
    --foreground) FOREGROUND=1 ;;
    *) echo "unknown arg $1"; exit 2 ;;
  esac; shift
done

RUN_ID=p-ba5a0c05
OBJ1_ID="${OBJ1_ID:-addsub-05-filter-last-pos}"
OBJ2_ID="${OBJ2_ID:-addsub-05-filter-integers}"
STEP=40000
OUT_ROOT="$DATA_ROOT/runs/$RUN_ID/analysis/ci_filter/step_$STEP"
CONFIGS="$DATA_ROOT/ci_filter/configs"
mkdir -p "$CONFIGS"

if [ "${FOREGROUND:-0}" != 1 ]; then
  LOG="$DATA_ROOT/logs/ci_filter.$(date +%Y%m%d-%H%M%S).log"
  mkdir -p "$DATA_ROOT/logs"
  ARGS=(--foreground)
  [ "$SMOKE" = 0 ] && ARGS+=(--no-smoke)
  [ -n "$ONLY" ] && ARGS+=(--only "$ONLY")
  [ -n "$INIT_ID" ] && ARGS+=(--init-id "$INIT_ID")
  setsid nohup "$0" "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
  ln -sfn "$LOG" "$DATA_ROOT/logs/ci_filter.latest.log"
  echo "launched pid $!; log $LOG"
  echo "follow: tail -f $DATA_ROOT/logs/ci_filter.latest.log"
  exit 0
fi

test -f "$DATA_ROOT/runs/$RUN_ID/launch_config.yaml" || { echo "run data missing: push it first (README step 3)"; exit 1; }
test -d "$DATA_ROOT/runs/$RUN_ID/ckpts/$STEP/decomposition" || { echo "checkpoint $STEP missing"; exit 1; }
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
echo "code: $(git -C "$REPO" log -1 --oneline)"

run_stage() {  # <stage> <filter id> [--init-id id]
  local stage="$1" id="$2"; shift 2
  "$VENV_PY" "$HERE/make_config.py" "$stage" "$CONFIGS/$stage-$id.yaml" "$@"
  echo "=== $stage $id start $(date -Is)"
  (cd "$REPO" && "$VENV_PY" -m param_decomp.ci_filter.scripts.run_ci_filter \
    --config "$CONFIGS/$stage-$id.yaml" --data_root "$DATA_ROOT" --filter_id "$id")
  echo "=== $stage $id done $(date -Is)"
}

if [ "$SMOKE" = 1 ] && [ -z "$ONLY" ]; then
  rm -rf "$OUT_ROOT/cf-smoke"
  run_stage smoke cf-smoke
  rm -rf "$OUT_ROOT/cf-smoke"
fi
if [ -z "$ONLY" ] || [ "$ONLY" = obj1 ]; then
  echo "$OBJ1_ID" > "$DATA_ROOT/ci_filter/obj1.id"
  run_stage obj1 "$OBJ1_ID"
  INIT_ID="$OBJ1_ID"
fi
if [ -z "$ONLY" ] || [ "$ONLY" = obj2 ]; then
  : "${INIT_ID:?objective 2 needs --init-id, the objective-1 filter id}"
  echo "$OBJ2_ID" > "$DATA_ROOT/ci_filter/obj2.id"
  run_stage obj2 "$OBJ2_ID" --init-id "$INIT_ID"
fi
echo "=== pipeline finished $(date -Is); outputs under $OUT_ROOT"
