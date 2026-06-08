#!/usr/bin/env bash
# Single-node torch bench via torchrun (clean 8-GPU A/B). Reads the module+args to run
# from jax_spike/_torch_run_cmd.sh and launches:
#   torchrun --standalone --nproc_per_node=<ntasks> -m <module> <config...>
# Under torchrun each worker has WORLD_SIZE set, so pd-lm / pd-lm-2pool fall through to the
# training path (no self-submission). One torchrun launcher per node spawns one worker per GPU.
#SBATCH --job-name=pd-torch-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/polished-lake/home/oli/pd-nano-jax/jax_spike/logs/%j.out
set -euo pipefail
cd ~/pd-nano-jax
mkdir -p jax_spike/logs
NPROC="${NPROC:-8}"
echo "=== torch job $SLURM_JOB_ID | node=$SLURM_JOB_NODELIST nproc=$NPROC | $(date) ==="
PYTHONPATH=. ~/pd-nano-jax/.venv/bin/torchrun --standalone --nproc_per_node="$NPROC" \
  -m $(cat jax_spike/_torch_run_cmd.sh)
