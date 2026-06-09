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
#SBATCH --comment="4way gpt2xl b200 throughput bench (torch cell) - oli, lore handoff 2026-06-09"
#SBATCH --output=%x-%j.out
set -euo pipefail
# REPO_DIR lets a perf-experiment agent run its OWN isolated clone+venv (default: the main checkout).
REPO_DIR="${REPO_DIR:-$HOME/pd-nano-jax}"
cd "$REPO_DIR"
mkdir -p jax_spike/logs
NPROC="${NPROC:-8}"
echo "=== torch job $SLURM_JOB_ID | repo=$REPO_DIR node=$SLURM_JOB_NODELIST nproc=$NPROC | cmd: $* | $(date) ==="
PYTHONPATH=. "$REPO_DIR/.venv/bin/torchrun" --standalone --nproc_per_node="$NPROC" -m "$@"
