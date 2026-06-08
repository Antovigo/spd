#!/usr/bin/env bash
# Local -> Andromeda torch-DDP runner (mirrors gpu.sh but for the torch bench).
#   NODES=2 GPN=8 remote/torch_gpu.sh "stage9_torch_bench.py --steps 12"
# Arg is the python invocation relative to jax_spike/ (no leading 'python').
set -euo pipefail
CMD="${*:?usage: [NODES=N GPN=G PART=p] remote/torch_gpu.sh <script.py args...>}"
NODES="${NODES:-2}"; GPN="${GPN:-8}"; PART="${PART:-h200-reserved-default}"
BRANCH="${BRANCH:-feature/nano-pd-jax}"
SSH="ssh -o RemoteCommand=none -o RequestTTY=no a-login"
WT='~/pd-nano-jax'

git push -q origin "$BRANCH"
echo "[remote] pull + submit torch ($NODES x $GPN = $((NODES*GPN)) GPU, part=$PART)"
JOBID=$($SSH "
  set -e
  cd $WT && git fetch -q origin $BRANCH && git reset -q --hard origin/$BRANCH
  mkdir -p jax_spike/logs
  printf '%s\n' \"$CMD\" > jax_spike/_torch_cmd.sh
  sbatch --parsable --nodes=$NODES --ntasks-per-node=$GPN --partition=$PART jax_spike/remote/torch_job.sbatch
")
echo "[submit] job $JOBID"
while :; do
  ST=$($SSH "squeue -j $JOBID -h -o %T" 2>/dev/null || true)
  [ -z "$ST" ] && break
  echo "[wait] $JOBID: $ST"; sleep 8
done
echo "[done] ===== log $JOBID ====="
$SSH "cat $WT/jax_spike/logs/$JOBID.out"
