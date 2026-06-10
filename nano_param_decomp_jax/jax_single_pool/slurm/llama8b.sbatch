#!/bin/bash
#SBATCH --job-name=jsp-train
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --requeue
#SBATCH --signal=B:TERM@300
#SBATCH --output=/mnt/data/artifacts/mechanisms/param-decomp/slurm_logs/slurm-%j.out
# Usage:
#   sbatch --nodes=16 --time=3-00:00:00 --qos=opportunistic \
#     --comment="JAX single-pool VPD Llama8B L18 B=512 (SPEC-compliant; jsp-train)" \
#     slurm/llama8b.sbatch configs/llama8b_l18_b512.yaml
# Requeue-resume: SIGTERM (preemption / time limit) triggers an on-loop checkpoint
# save; the requeued job restores the latest checkpoint and continues (SPEC S22).
set -euo pipefail
CONFIG="$1"
REPO=/mnt/home/oli/pd-nano-jax-jaxsp/nano_param_decomp_jax
cd "$REPO"
srun --kill-on-bad-exit=1 --ntasks-per-node=8 --distribution=block:block bash -c "
  source .venv-cuda/bin/activate
  exec jsp-train jax_single_pool/$CONFIG
"
