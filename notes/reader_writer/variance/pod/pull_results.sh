#!/usr/bin/env bash
# Run on the CLUSTER. Pull the analysis outputs back into the cluster copy of the run: results/,
# logs, the extraction's meta, KL and comps (PULL_INNER=1 also pulls the ~70 GB of activations).
# Each piece is tried even if an earlier one fails; the exit code is the last failure.
#     POD_HOST=<ip> POD_PORT=<port> KEY=~/.ssh/runpod [NAME=<name>] ./pull_results.sh   (NAME: variance_<name>/)
set -uo pipefail
: "${POD_HOST:?}" "${POD_PORT:?}" "${KEY:?}"
REMOTE="root@$POD_HOST:/workspace/data/runs/p-ba8b0c08/analysis/reader_writer/variance${NAME:+_$NAME}"
LOCAL="$HOME/out/pod-backup/p-ba8b0c08/analysis/reader_writer/variance${NAME:+_$NAME}"
SSH="ssh -p $POD_PORT -i $KEY -o StrictHostKeyChecking=accept-new"
mkdir -p "$LOCAL/logs/pod"
rc=0
rsync -a -e "$SSH" "$REMOTE/results" "$LOCAL/" || rc=$?
rsync -a -e "$SSH" --include='*/' --include='meta.json' --include='kl_last.npy' --include='comps.npz' \
  --exclude='*' "$REMOTE/alive_only" "$LOCAL/" || rc=$?
rsync -a -e "$SSH" "root@$POD_HOST:/workspace/data/logs/" "$LOCAL/logs/pod/" || rc=$?
if [ -n "${PULL_INNER:-}" ]; then  # the activations, one rsync per block, 8 at a time
  seq 0 31 | xargs -P 8 -I{} rsync -a --mkpath -e "$SSH" "$REMOTE/alive_only/L{}/inner.npy" \
    "$LOCAL/alive_only/L{}/" || rc=$?
fi
exit $rc  # nonzero while results/ does not exist yet (before block 0 publishes)
