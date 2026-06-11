#!/bin/bash
#SBATCH --job-name=jsp-train
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --requeue
# TERM must reach the srun-launched python ranks (the trainer's SIGTERM-save
# handler) — `B:` would deliver it to the batch shell ONLY, which is how job 50818
# lost its preemption grace window and restarted from step 0.
#SBATCH --signal=TERM@300
#SBATCH --output=/mnt/data/artifacts/mechanisms/param-decomp/slurm_logs/slurm-%j.out
# Usage:
#   sbatch --nodes=16 --time=3-00:00:00 --qos=opportunistic \
#     --comment="JAX single-pool VPD Llama8B L18 B=512 (SPEC-compliant; jsp-train)" \
#     slurm/llama8b.sbatch configs/llama8b_l18_b512.yaml
# Requeue-resume: preemption / time limit delivers SIGTERM to the ranks (see
# --signal above), the trainer saves a checkpoint inside the grace window, and the
# requeued job restores it and continues (SPEC S22). No in-job retry loop: the one
# recurring app-level crash class (XLA CUDA-graph capture) is eliminated at the
# source below, and a loop can neither distinguish deterministic failures (it burned
# 5 attempts on each OOM) nor escape a bad node (it reuses the allocation).
set -uo pipefail
CONFIG="$1"
REPO=/mnt/home/oli/pd-nano-jax-jaxsp/nano_param_decomp_jax
cd "$REPO"
srun --kill-on-bad-exit=1 --ntasks-per-node=8 --cpus-per-task=8 --distribution=block:block bash -c "
  source .venv-cuda/bin/activate
  # default 0.75 caps the XLA pool at ~135 GiB on a B200 — the C30k 16-GPU step's
  # 108 GiB arena + params + warmup/harvest buffers needs more (OOM, job 50644)
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
  # CUDA-graph capture (XLA command buffers) intermittently dies with
  # CUDA_ERROR_STREAM_CAPTURE_INVALIDATED — killed jobs 50453/50525/50676/50743 on
  # DISJOINT allocations (not a bad node). Torch never exercises this path. Disabling
  # measured ~0% cost at the production step (8,007 vs 8,015 tok/s/GPU).
  export XLA_FLAGS=\"--xla_gpu_enable_command_buffer=\"
  exec jsp-train jax_single_pool/$CONFIG
"
