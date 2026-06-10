#!/bin/bash
#SBATCH --job-name=jsp-offline-eval-loop
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --qos=opportunistic
#SBATCH --dependency=singleton
#SBATCH --output=/mnt/data/artifacts/mechanisms/param-decomp/slurm_logs/slurm-%j.out
# Recurring offline eval of a live JAX run: for every committed orbax checkpoint not
# yet evaluated, jsp-export (JAX venv) -> pd-offline-eval (torch venv, logs eval/* +
# slow_eval/* into the run's wandb) -> delete the ~42 GB export. Self-resubmits until
# the run reaches its final step or goes quiet for 24 h. Failures (e.g. a checkpoint
# caught mid-write) are retried on the next cycle, not marked done.
# Usage:
#   sbatch --comment="..." slurm/offline_eval_loop.sbatch <run_dir> <torch_eval_yaml> <final_step>
set -uo pipefail
RUN_DIR="$1"
TORCH_YAML="$2"
FINAL_STEP="$3"
JAX_REPO=/mnt/home/oli/pd-nano-jax-jaxsp/nano_param_decomp_jax
TORCH_REPO=/mnt/home/oli/param-decomp
MARKER="$RUN_DIR/.offline_eval_done"
touch "$MARKER"

for step in $(ls "$RUN_DIR/ckpts" | grep -E '^[0-9]+$' | sort -n); do
  [[ "$step" -eq 0 ]] && continue
  grep -qx "$step" "$MARKER" && continue
  echo "=== exporting + evaluating step $step ==="
  export_file="$RUN_DIR/export/model_${step}.safetensors"
  if ! (cd "$JAX_REPO" && .venv-cuda/bin/jsp-export "$RUN_DIR" --step "$step"); then
    echo "export of step $step failed (mid-write checkpoint?) — retrying next cycle"
    continue
  fi
  if (cd "$TORCH_REPO" && .venv/bin/pd-offline-eval "$export_file" "$TORCH_YAML"); then
    echo "$step" >> "$MARKER"
  else
    echo "offline eval of step $step failed — retrying next cycle"
  fi
  rm -f "$export_file"
done

if grep -qx "$FINAL_STEP" "$MARKER"; then
  echo "final step $FINAL_STEP evaluated — loop done"
  exit 0
fi
quiet_seconds=$(( $(date +%s) - $(stat -c %Y "$RUN_DIR/metrics.jsonl") ))
if (( quiet_seconds > 86400 )); then
  echo "run quiet for ${quiet_seconds}s — not resubmitting"
  exit 0
fi
sbatch --begin=now+40minutes --comment="offline-eval loop for $(basename "$RUN_DIR") (self-resubmitting)" \
  "$0" "$RUN_DIR" "$TORCH_YAML" "$FINAL_STEP"
