#!/usr/bin/env bash
# Multi-GPU JAX Llama-8B single-pool PD throughput bench. 1 task/GPU; each process claims
# local_device_ids=[SLURM_LOCALID] in init_distributed(); jax.distributed auto-detects the mesh.
# Submit: sbatch llama8b_slurm.sbatch <args to llama8b_real>
#   e.g.  sbatch llama8b_slurm.sbatch --shard --per_gpu_batch 8 --steps 12 --n_warmup 2
#SBATCH --job-name=jax-llama8b-pd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --qos=opportunistic
#SBATCH --time=00:30:00
#SBATCH --comment="jax single-pool Llama-8B L18-MLP PD throughput/MFU bench (jax-vs-torch) - oli"
#SBATCH --output=/mnt/home/oli/pd-nano-jax-jaxsp/nano_param_decomp_jax/jax_single_pool/experiments/logs/%j.out
set -euo pipefail
WT=/mnt/home/oli/pd-nano-jax-jaxsp
cd "$WT/nano_param_decomp_jax"
mkdir -p jax_single_pool/experiments/logs

echo "=== job $SLURM_JOB_ID | nodes=$SLURM_NNODES ntasks=$SLURM_NTASKS | $(date) ==="
echo "nodelist: $SLURM_JOB_NODELIST | args: $* ==="

srun --kill-on-bad-exit=1 --nodes="$SLURM_NNODES" --ntasks-per-node=8 --distribution=block bash -c "
  cd $WT/nano_param_decomp_jax
  source .venv-cuda/bin/activate
  export PYTHONPATH=$WT/jax_spike
  python -m jax_single_pool.experiments.llama8b_real $*
"
