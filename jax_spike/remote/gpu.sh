#!/usr/bin/env bash
# Local -> Andromeda GPU runner for the JAX spike.
# Usage: remote/gpu.sh "python stage6_pgd.py --timing"
# Syncs jax_spike/ to the cluster, submits a 1-GPU SLURM job, waits, prints the log.
set -euo pipefail

CMD="${*:?usage: remote/gpu.sh <command to run on the GPU node>}"
SSH_OPTS="-o RemoteCommand=none -o RequestTTY=no"
REMOTE=a-login
HERE="$(cd "$(dirname "$0")/.." && pwd)"  # the jax_spike/ dir

cd "$HERE"
echo "$CMD" > _remote_cmd.sh

echo "[sync] jax_spike/ -> $REMOTE:~/jax_spike/"
rsync -az --delete \
  --exclude '.venv*' --exclude '__pycache__' --exclude 'logs' --exclude '.git' \
  -e "ssh $SSH_OPTS" ./ "$REMOTE:jax_spike/"

ssh $SSH_OPTS $REMOTE 'mkdir -p ~/jax_spike/logs'
JOBID=$(ssh $SSH_OPTS $REMOTE 'cd ~/jax_spike && sbatch --parsable remote/job.sbatch')
echo "[submit] job $JOBID"

# wait for completion
while :; do
  ST=$(ssh $SSH_OPTS $REMOTE "squeue -j $JOBID -h -o %T" 2>/dev/null || true)
  [ -z "$ST" ] && break
  echo "[wait] job $JOBID: $ST"
  sleep 8
done

echo "[done] ===== log for job $JOBID ====="
ssh $SSH_OPTS $REMOTE "cat ~/jax_spike/logs/$JOBID.out"
