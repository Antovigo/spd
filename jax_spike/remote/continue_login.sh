#!/bin/bash
# Fallback: run the autonomous Claude on the LOGIN node (proven internet + can submit SLURM),
# detached so it survives logout. Logs to jax_spike/logs/claude-continue-login.out.
#   bash ~/pd-nano-jax/jax_spike/remote/continue_login.sh
set -uo pipefail
cd ~/pd-nano-jax
mkdir -p jax_spike/logs
CLAUDE=/mnt/polished-lake/home/oli/.local/bin/claude
LOG=~/pd-nano-jax/jax_spike/logs/claude-continue-login.out
setsid nohup "$CLAUDE" --dangerously-skip-permissions \
  --add-dir /mnt/polished-lake/shared/goodfire-orchestrate \
  --system-prompt "$(~/.claude/minimal-prompt.sh)" \
  -p "$(cat ~/pd-nano-jax/jax_spike/remote/continue_prompt.md)" \
  > "$LOG" 2>&1 &
echo "launched detached claude on login node, pid $!, log: $LOG"
