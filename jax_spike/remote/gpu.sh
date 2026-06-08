#!/usr/bin/env bash
# Local -> Andromeda GPU runner (git-based). Push local commits, pull on cluster,
# submit a SLURM job (1 task per GPU via srun), wait, print the log.
#
# Usage:
#   remote/gpu.sh "python stage6_pgd.py --timing"                  # 1 node, 8 GPUs
#   NODES=2 GPN=8 remote/gpu.sh "python stage8_train.py --steps 50" # 2 nodes, 16 GPUs
#   NODES=1 GPN=1 remote/gpu.sh "python stage6_pgd.py"             # 1 GPU
#
# Env knobs: NODES (default 1), GPN gpus/tasks per node (default 8),
#            PART partition (default h200-reserved-default), BRANCH (default feature/nano-pd-jax)
set -euo pipefail

CMD="${*:?usage: [NODES=N GPN=G PART=p] remote/gpu.sh <command>}"
NODES="${NODES:-1}"; GPN="${GPN:-8}"; PART="${PART:-h200-reserved-default}"
BRANCH="${BRANCH:-feature/nano-pd-jax}"
SSH="ssh -o RemoteCommand=none -o RequestTTY=no a-login"
WT='~/pd-nano-jax/jax_spike'

# 1. push local branch (must be committed)
echo "[push] $BRANCH"
git push -q origin "$BRANCH"

# 2. on cluster: pull, write the command, submit
echo "[remote] pull + submit ($NODES node(s) x $GPN gpu = $((NODES*GPN)) GPUs, part=$PART)"
JOBID=$($SSH "
  set -e
  cd $WT/..
  git fetch -q origin $BRANCH && git reset -q --hard origin/$BRANCH
  cd $WT && mkdir -p logs
  printf '%s\n' \"$CMD\" > _remote_cmd.sh
  sbatch --parsable --nodes=$NODES --ntasks-per-node=$GPN --partition=$PART remote/job.sbatch
")
echo "[submit] job $JOBID"

# 3. wait
while :; do
  ST=$($SSH "squeue -j $JOBID -h -o %T" 2>/dev/null || true)
  [ -z "$ST" ] && break
  echo "[wait] $JOBID: $ST"; sleep 8
done

echo "[done] ===== log $JOBID ====="
$SSH "cat $WT/logs/$JOBID.out"
