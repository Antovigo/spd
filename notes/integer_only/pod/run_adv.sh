#!/usr/bin/env bash
# Run ON the pod: the integer-only-adv stages (notes/integer_only/plan_adv.md): `train` on all
# visible GPUs, `alive`, `score` and `collect` on GPU 0 (the factored model is single-device, and
# one device for both CI passes keeps them one program family, as integer-only stages 1 and 5),
# `applet` on the CPU. `run.sh` calls this after the global-mask stages; it can also run alone
# (in the foreground, from any directory) once the integer-only stages `candidates` and
# `problem` are done:
#     source /workspace/spd/notes/dual_objective/addsub-all-layers/env.sh
#     /workspace/spd/notes/integer_only/pod/run_adv.sh
# Env overrides: RUN_ID (default p-512d73ab), ADV_CONFIG (default the addsub-answers-09-adv
# config). Each stage writes $RUN/integer_only_adv/status/<stage>.done or .failed; a done stage
# is skipped and `train` resumes from its last saved state, so rerunning resumes. Stops at the
# first failed stage (exit 1). Prints "=== adv stages done" at the end, never "all stages done".
set -euo pipefail
: "${REPO:?source env.sh first}" "${DATA_ROOT:?source env.sh first}" "${VENV_PY:?source env.sh first}"
RUN_ID="${RUN_ID:-p-512d73ab}"
RUN="$DATA_ROOT/runs/$RUN_ID"
ADV_CONFIG="${ADV_CONFIG:-notes/integer_only/addsub-answers-09-adv.yaml}"
LOGS="$RUN/integer_only_adv/logs"
mkdir -p "$LOGS"
# Bitwise-reproducible GPU kernels (plan.md, "Reproducibility of the CI"; run.sh sets it too),
# and no CUDA command buffers, as the run's own launch_env (on an L40 / driver 535 the
# training step's capture failed: "Failed to add memset node to a CUDA graph").
for FLAG in --xla_gpu_deterministic_ops=true --xla_gpu_enable_command_buffer=; do
  case " ${XLA_FLAGS:-} " in
    *" $FLAG "*) ;;
    *) export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }$FLAG" ;;
  esac
done
cd "$REPO"
ALL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)
echo "=== integer-only-adv: code $(git rev-parse HEAD)  config: $ADV_CONFIG  GPUs [$ALL_GPUS] $(date -u +%FT%TZ)"

adv_stage() {  # adv_stage <name> <CUDA_VISIBLE_DEVICES>: one stage, its own log
  local name=$1 gpus=$2 log="$LOGS/$1.log"
  echo "=== adv $name on GPU [$gpus] $(date -u +%FT%TZ) -> $log"
  CUDA_VISIBLE_DEVICES="$gpus" "$VENV_PY" -m param_decomp.integer_only.adv.run "$name" \
    --run_dir "$RUN" --data_root "$DATA_ROOT" --config "$ADV_CONFIG" >> "$log" 2>&1
}

adv_stage train "$ALL_GPUS"
adv_stage alive 0
adv_stage score 0
adv_stage collect 0
JAX_PLATFORMS=cpu adv_stage applet ""
echo "=== adv stages done $(date -u +%FT%TZ): $RUN/integer_only_adv/README.md"
