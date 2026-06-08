#!/usr/bin/env bash
# Generic JAX-spike SLURM job. Override scale on the sbatch CLI:
#   sbatch --nodes=2 --ntasks-per-node=8 --partition=h200-reserved-default remote/job.sbatch
# Always launches via srun (1 task per GPU). Each task pins to its local GPU and runs
# _remote_cmd.sh; distributed scripts call jax.distributed.initialize() (SLURM auto-detect).
#SBATCH --job-name=jax-spike
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/polished-lake/home/oli/pd-nano-jax/jax_spike/logs/%j.out
set -euo pipefail
cd ~/pd-nano-jax/jax_spike
mkdir -p logs

echo "=== job $SLURM_JOB_ID | nodes=$SLURM_NNODES ntasks=$SLURM_NTASKS | $(date) ==="
echo "nodelist: $SLURM_JOB_NODELIST"

# All 8 GPUs visible to every task (--gres=gpu:8, no per-task binding); each process
# claims local_device_ids=[SLURM_LOCALID] in init_distributed(). This is the canonical
# JAX-on-SLURM-GPU recipe (1 process per GPU). jax.distributed auto-detects the
# coordinator / num_processes / process_id from the SLURM env.
srun --kill-on-bad-exit=1 bash -c '
  cd ~/pd-nano-jax/jax_spike
  source .venv-cuda/bin/activate
  bash _remote_cmd.sh
'
