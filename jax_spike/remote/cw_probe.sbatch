#!/usr/bin/env bash
# Single-process multi-GPU jax probe (one python process sees all local GPUs via jax.devices()).
#   sbatch --export=ALL,JAX_VENV=.venv-cuda13 remote/cw_probe.sbatch probe_reshard.py
#SBATCH --job-name=jax-probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=00:15:00
#SBATCH --comment="jax sharding/reshard/zero data-gathering probe - oli perf spike"
#SBATCH --output=/mnt/home/oli/pd-nano-jax/jax_spike/logs/%j.out
set -euo pipefail
cd ~/pd-nano-jax/jax_spike
mkdir -p logs
echo "=== probe job $SLURM_JOB_ID | $(date) | cmd: python $* ==="
source "${JAX_VENV:-.venv-cuda13}/bin/activate"
python $*
