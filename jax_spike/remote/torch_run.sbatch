#!/usr/bin/env bash
# Single-node torch bench via torchrun (clean 8-GPU A/B). Pass the module + config as args:
#   sbatch ... remote/torch_run.sbatch param_decomp_lab.experiments.lm.run path/to/config.yaml
#   sbatch ... remote/torch_run.sbatch param_decomp_lab.experiments.lm.two_pool_run path/to/config.yaml
# Launches: torchrun --standalone --nproc_per_node=<NPROC> -m <args...>. Under torchrun each
# worker has WORLD_SIZE set, so pd-lm / pd-lm-2pool fall through to the training path (no
# self-submission). One torchrun launcher per node spawns one worker per GPU. Args are baked into
# the job at submit time, so multiple jobs can be queued concurrently without a shared command file.
#SBATCH --job-name=pd-torch-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=00:18:00
#SBATCH --output=/mnt/polished-lake/home/oli/pd-nano-jax/jax_spike/logs/%j.out
set -euo pipefail
cd ~/pd-nano-jax
mkdir -p jax_spike/logs
NPROC="${NPROC:-8}"
echo "=== torch job $SLURM_JOB_ID | node=$SLURM_JOB_NODELIST nproc=$NPROC | cmd: $* | $(date) ==="
PYTHONPATH=. ~/pd-nano-jax/.venv/bin/torchrun --standalone --nproc_per_node="$NPROC" -m "$@"
