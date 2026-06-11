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
set -uo pipefail
CONFIG="$1"
REPO=/mnt/home/oli/pd-nano-jax-jaxsp/nano_param_decomp_jax
cd "$REPO"
# In-job retry: SLURM --requeue does not catch app-level crashes (e.g. the intermittent
# XLA CUDA-graph capture failure that killed job 50453 at step 280). Each retry resumes
# from the latest orbax checkpoint; persistent failures still fail the job.
for attempt in 1 2 3 4 5; do
  srun --kill-on-bad-exit=1 --ntasks-per-node=8 --cpus-per-task=8 --distribution=block:block bash -c "
    source .venv-cuda/bin/activate
    # default 0.75 caps the XLA pool at ~135 GiB on a B200 — the C30k 16-GPU step's
    # 108 GiB arena + params + warmup/harvest buffers needs more (OOM, job 50644)
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
    # CUDA-graph capture (XLA command buffers) intermittently dies with
    # CUDA_ERROR_STREAM_CAPTURE_INVALIDATED — killed jobs 50453/50525/50676/50743 on
    # DISJOINT allocations (not a bad node). Torch never exercises this path. Disabling
    # costs ~1-3% launch overhead; measured vs not-training-at-all, an easy trade.
    export XLA_FLAGS="--xla_gpu_enable_command_buffer="
    exec jsp-train jax_single_pool/$CONFIG
  " && exit 0
  echo "=== train crashed (attempt $attempt); resuming from latest checkpoint ==="
  sleep 30
done
exit 1
