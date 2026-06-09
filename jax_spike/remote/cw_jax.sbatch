#!/usr/bin/env bash
# CoreWeave B200 JAX-spike job. The python script + args are baked into the job at submit
# time (no shared _remote_cmd.sh file -> safe to queue both stage12 and stage13 concurrently):
#   sbatch remote/cw_jax.sbatch stage12_gpt2xl_1pool.py --per_gpu_batch 2 --steps 30
#   sbatch remote/cw_jax.sbatch stage13_gpt2xl_2pool.py --global_batch 16 --steps 30
# srun launches 1 task per GPU; each process claims local_device_ids=[SLURM_LOCALID] in
# init_distributed(); jax.distributed auto-detects coordinator/num_processes/process_id.
#SBATCH --job-name=jax-cw-gpt2xl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=00:25:00
#SBATCH --comment="4way gpt2xl b200 throughput bench (jax cell) - oli, lore handoff 2026-06-09"
#SBATCH --output=/mnt/home/oli/pd-nano-jax/jax_spike/logs/%j.out
set -euo pipefail
cd ~/pd-nano-jax/jax_spike
mkdir -p logs

echo "=== job $SLURM_JOB_ID | nodes=$SLURM_NNODES ntasks=$SLURM_NTASKS | $(date) ==="
echo "nodelist: $SLURM_JOB_NODELIST | cmd: python $* ==="

srun --kill-on-bad-exit=1 bash -c "
  cd ~/pd-nano-jax/jax_spike
  source .venv-cuda/bin/activate
  python $*
"
