#!/usr/bin/env bash
# Generic JAX-spike SLURM job. Override scale on the sbatch CLI:
#   sbatch --nodes=2 --ntasks-per-node=8 --partition=h200-reserved-default remote/job.sbatch
# Always launches via srun (1 task per GPU). Each task pins to its local GPU and runs
# _remote_cmd.sh; distributed scripts call jax.distributed.initialize() (SLURM auto-detect).
#SBATCH --job-name=jax-spike
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/polished-lake/home/oli/pd-nano-jax/jax_spike/logs/%j.out
set -euo pipefail
cd ~/pd-nano-jax/jax_spike
mkdir -p logs

echo "=== job $SLURM_JOB_ID | nodes=$SLURM_NNODES ntasks=$SLURM_NTASKS | $(date) ==="
echo "nodelist: $SLURM_JOB_NODELIST"

srun --kill-on-bad-exit=1 bash -c '
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID   # one GPU per task
  cd ~/pd-nano-jax/jax_spike
  source .venv-cuda/bin/activate
  bash _remote_cmd.sh
'
