#!/usr/bin/env bash
# Torch DDP bench job. Override scale on the sbatch CLI (--nodes / --ntasks-per-node / --partition).
# Each task = 1 GPU; reads SLURM env for NCCL rendezvous. Runs _torch_cmd.sh via the torch venv.
#SBATCH --job-name=pd-torch-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/polished-lake/home/oli/pd-nano-jax/jax_spike/logs/%j.out
set -euo pipefail
cd ~/pd-nano-jax
mkdir -p jax_spike/logs
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=29513
echo "=== torch job $SLURM_JOB_ID | nodes=$SLURM_NNODES ntasks=$SLURM_NTASKS | master=$MASTER_ADDR | $(date) ==="

srun --gres=gpu:8 --kill-on-bad-exit=1 bash -c '
  export RANK=$SLURM_PROCID WORLD_SIZE=$SLURM_NTASKS LOCAL_RANK=$SLURM_LOCALID
  export MASTER_ADDR='"$MASTER_ADDR"' MASTER_PORT='"$MASTER_PORT"'
  cd ~/pd-nano-jax
  PYTHONPATH=. ~/param-decomp/.venv/bin/python jax_spike/$(cat jax_spike/_torch_cmd.sh)
'
