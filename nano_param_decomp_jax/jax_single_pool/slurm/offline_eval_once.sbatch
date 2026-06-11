#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --qos=opportunistic
#SBATCH --output=/mnt/data/artifacts/mechanisms/param-decomp/slurm_logs/slurm-%j.out
# Offline eval of ONE checkpoint, push-triggered by the trainer right after the
# save (run.py `_submit_offline_eval`): jsp-export (JAX venv) -> pd-offline-eval
# (torch venv, logs eval/* + slow_eval/* into the run's wandb) -> delete the export.
# The trainer submits with `-J jsp-oeval-<run> --dependency=singleton`, so per-run
# evals serialize and never overlap. `<run_dir>/.offline_eval_done` dedups across
# requeue-resubmissions; `<run_dir>/offline_eval_args` (optional, one line) carries
# run-specific pd-offline-eval flags (e.g. `--micro-batch-size 4` for C=49152,
# where the default mb8 OOMs a B200).
# Usage (normally via the trainer):
#   sbatch -J jsp-oeval-<run> --dependency=singleton --comment="..." \
#     slurm/offline_eval_once.sbatch <run_dir> <step>
set -uo pipefail
RUN_DIR="$1"
STEP="$2"
JAX_REPO=/mnt/home/oli/pd-nano-jax-jaxsp/nano_param_decomp_jax
TORCH_REPO=/mnt/home/oli/param-decomp
MARKER="$RUN_DIR/.offline_eval_done"
touch "$MARKER"

if grep -qx "$STEP" "$MARKER"; then
  echo "step $STEP already evaluated"
  exit 0
fi
EXTRA_EVAL_ARGS=()
if [[ -f "$RUN_DIR/offline_eval_args" ]]; then
  read -r -a EXTRA_EVAL_ARGS < "$RUN_DIR/offline_eval_args"
fi
export_file="$RUN_DIR/export/model_${STEP}.safetensors"
(cd "$JAX_REPO" && .venv-cuda/bin/jsp-export "$RUN_DIR" --step "$STEP")
(cd "$TORCH_REPO" && .venv/bin/pd-offline-eval "$export_file" "$RUN_DIR/torch_config.yaml" "${EXTRA_EVAL_ARGS[@]}")
echo "$STEP" >> "$MARKER"
rm -f "$export_file"
