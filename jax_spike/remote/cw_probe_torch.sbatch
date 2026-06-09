#!/usr/bin/env bash
# Single-GPU torch probe (torch venv at repo root, PYTHONPATH=repo root).
#   sbatch remote/cw_probe_torch.sbatch jax_spike/probe_roofline_torch.py --model llama8b
#SBATCH --job-name=torch-probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --comment="torch+compile roofline probe (per-GPU-batch MFU knee) - oli perf spike"
#SBATCH --output=/mnt/home/oli/pd-nano-jax/jax_spike/logs/%j.out
set -euo pipefail
cd ~/pd-nano-jax
mkdir -p jax_spike/logs
echo "=== torch probe $SLURM_JOB_ID | $(date) | cmd: $* ==="
source .venv/bin/activate
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}_probe"
PYTHONPATH=. python "$@"
