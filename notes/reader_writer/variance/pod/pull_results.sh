#!/usr/bin/env bash
# Run on the CLUSTER. Pull the analysis outputs back into the cluster copy of the run: results/,
# logs, the extraction's meta, KL and comps (PULL_INNER=1 also pulls the ~70 GB of activations).
#     POD_HOST=<ip> POD_PORT=<port> KEY=~/.ssh/runpod ./pull_results.sh
set -euo pipefail
: "${POD_HOST:?}" "${POD_PORT:?}" "${KEY:?}"
REMOTE="root@$POD_HOST:/workspace/data/runs/p-ba8b0c08/analysis/reader_writer/variance"
LOCAL="$HOME/out/pod-backup/p-ba8b0c08/analysis/reader_writer/variance"
SSH="ssh -p $POD_PORT -i $KEY -o StrictHostKeyChecking=accept-new"
mkdir -p "$LOCAL/logs/pod"
rsync -a -e "$SSH" "$REMOTE/results" "$LOCAL/"
rsync -a -e "$SSH" --include='*/' --include='meta.json' --include='kl_last.npy' --include='comps.npz' \
  ${PULL_INNER:+--include='inner.npy'} --exclude='*' "$REMOTE/alive_only" "$LOCAL/"
rsync -a -e "$SSH" "root@$POD_HOST:/workspace/data/logs/" "$LOCAL/logs/pod/"
