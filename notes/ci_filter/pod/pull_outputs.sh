#!/usr/bin/env bash
# Run on the CLUSTER. Pull every CI filter output from the pod into the run's own analysis
# dir, next to anything already there. Each filter's `training/ci_fn/` is the fp32 CI fn
# (~3.9 GB); pass --no-ci-fn to skip it.
#     POD_HOST=<ip> POD_PORT=<port> KEY=~/.ssh/<key> ./pull_outputs.sh [--no-ci-fn]
set -euo pipefail
: "${POD_HOST:?}" "${POD_PORT:?}" "${KEY:?}"
EXCLUDE=()
[ "${1:-}" = --no-ci-fn ] && EXCLUDE=(--exclude "training/ci_fn/")
DEST="${DEST:-$HOME/out/pod-backup/p-ba5a0c05/analysis/ci_filter/}"
SSH="ssh -p $POD_PORT -i $KEY -o StrictHostKeyChecking=accept-new"
rsync -av --mkpath "${EXCLUDE[@]}" -e "$SSH" \
  "root@$POD_HOST:/workspace/data/runs/p-ba5a0c05/analysis/ci_filter/" "$DEST"
rsync -av --mkpath -e "$SSH" "root@$POD_HOST:/workspace/data/logs/ci_filter.*" "$DEST/pod-logs/" || true
