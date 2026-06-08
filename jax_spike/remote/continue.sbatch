#!/bin/bash
# Launch an autonomous Claude on the cluster to continue the 4-way benchmark.
# Submit:  sbatch ~/pd-nano-jax/jax_spike/remote/continue.sbatch
#SBATCH --job-name=claude-continue
#SBATCH --partition=h200-reserved-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/polished-lake/home/oli/pd-nano-jax/jax_spike/logs/claude-continue-%j.out
set -uo pipefail
cd ~/pd-nano-jax
mkdir -p jax_spike/logs
CLAUDE=/mnt/polished-lake/home/oli/.local/bin/claude
echo "=== claude-continue on $(hostname) @ $(date) ==="
# `danger` expanded: --dangerously-skip-permissions + the minimal system prompt; plus the --add-dir
# from the user's claude() shell function.
"$CLAUDE" --dangerously-skip-permissions \
  --add-dir /mnt/polished-lake/shared/goodfire-orchestrate \
  --system-prompt "$(~/.claude/minimal-prompt.sh)" \
  -p "$(cat ~/pd-nano-jax/jax_spike/remote/continue_prompt.md)"
echo "=== claude-continue exited @ $(date) ==="
