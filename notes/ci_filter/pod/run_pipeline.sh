#!/usr/bin/env bash
# CI filter pipeline on ONE GPU pod: smoke -> objective 1 -> objective 2, detached.
#     source /workspace/spd/notes/dual_objective/addsub-all-layers/env.sh
#     ./run_pipeline.sh [--no-smoke] [--only obj1|obj2|obj3] [--init-id cf-xxxxxxxx]
# TASK=mult filters the MULTIPLICATION decomposition (p-3c1a0c06) instead of addsub;
# only objective 1 has a mult seat. RUN_ID and STEP override the task's defaults.
# Stages run in order and stop at the first failure; `--only obj1` keeps the smoke (skip it with
# --no-smoke), `--only obj2` never runs it. The smoke writes to the fixed id
# `cf-smoke` and is deleted when it passes. Objective 2 starts from objective 1's CI fn: from
# the id this pipeline just produced, or from --init-id when run with `--only obj2`.
# Hardware knobs (env): MICRO, EVAL_BATCH, GRID_CHUNK — see make_config.py.
# Filter ids (output dir + wandb run name/id): OBJ1_ID, OBJ2_ID, OBJ3_ID (the answer-CE
# objective, which only runs with `--only obj3`); they fail closed if taken.
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

# TASK selects the decomposition: `addsub` (default, p-ba5a0c05) or `mult` (p-3c1a0c06).
# It must agree with the seat make_config.py picks, so it is exported, not just read here.
export TASK="${TASK:-addsub}"
case "$TASK" in
  addsub) DEFAULT_RUN_ID=p-ba5a0c05; PREFIX=addsub-05 ;;
  mult)   DEFAULT_RUN_ID=p-3c1a0c06; PREFIX=mult-06 ;;
  *) echo "TASK must be addsub or mult, got '$TASK'"; exit 2 ;;
esac
# EXPORTED, both of them: make_config.py writes them into the generated config, so the
# checkpoint this script preflights is always the one the job actually loads. Without the
# export the overrides would move the checks and the output dir while the job silently ran
# the template's hard-coded run and step.
export RUN_ID="${RUN_ID:-$DEFAULT_RUN_ID}"
export STEP="${STEP:-40000}"
OBJ1_ID="${OBJ1_ID:-$PREFIX-filter-last-pos}"
OBJ2_ID="${OBJ2_ID:-$PREFIX-filter-integers}"
OBJ3_ID="${OBJ3_ID:-$PREFIX-filter-answer-ce}"
# Only objective 1 has a multiplication seat, and make_config.py refuses the others. Default
# to it rather than letting the bare invocation spend ~3 GPU-hours on obj1 and THEN die in
# obj2's config generation. An explicit `--only` still wins (and still fails loudly if it
# names a stage mult has no seat for).
if [ "$TASK" = mult ] && [ -z "$ONLY" ]; then
  ONLY=obj1
  echo "TASK=mult: only objective 1 is authored for multiplication; running --only obj1"
fi
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

if [ "$SMOKE" = 1 ] && [ "$ONLY" != obj2 ] && [ "$ONLY" != obj3 ]; then
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
if [ "$ONLY" = obj3 ]; then
  : "${INIT_ID:?objective 3 needs --init-id, the filter it narrows}"
  echo "$OBJ3_ID" > "$DATA_ROOT/ci_filter/obj3.id"
  run_stage obj3 "$OBJ3_ID" --init-id "$INIT_ID"
fi
echo "=== pipeline finished $(date -Is); outputs under $OUT_ROOT"
