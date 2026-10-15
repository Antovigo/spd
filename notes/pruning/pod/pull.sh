#!/usr/bin/env bash
# Run on the CLUSTER. Pull the pruning outputs and logs back into the run's cluster copy.
#     POD_HOST=<ip> POD_PORT=<port> KEY=~/.ssh/runpod ./pull.sh
set -euo pipefail
: "${POD_HOST:?}" "${POD_PORT:?}" "${KEY:?}"
RUN_ID="${RUN_ID:-p-ba8b0c08}"
STEP="${STEP:-40000}"
DEST="${DEST:-$HOME/out/pod-backup/$RUN_ID}"
SSH="ssh -p $POD_PORT -i $KEY -o StrictHostKeyChecking=accept-new"
rsync -a --mkpath -e "$SSH" "root@$POD_HOST:/workspace/data/runs/$RUN_ID/analysis/pruning/" "$DEST/analysis/pruning/"
rsync -a --mkpath -e "$SSH" "root@$POD_HOST:/workspace/data/logs/" "$DEST/analysis/pruning/step_$STEP/logs/" --include "pruning.*" --exclude "*"
