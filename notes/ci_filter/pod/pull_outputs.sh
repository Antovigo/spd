#!/usr/bin/env bash
# Run on the CLUSTER. Pull every CI filter dir (`<run>/<id>-<step>/`, logs included) from the pod
# into the run's own dir, next to anything already there. Each filter's `training/ci_fn/` is the
# fp32 CI fn (~3.9 GB); pass --no-ci-fn to skip it.
#     POD_HOST=<ip> POD_PORT=<port> KEY=~/.ssh/<key> [RUN_ID=p-...] [STEP=40000] ./pull_outputs.sh [--no-ci-fn]
set -euo pipefail
: "${POD_HOST:?}" "${POD_PORT:?}" "${KEY:?}"
EXCLUDE=(--exclude "cf-smoke-*/")
[ "${1:-}" = --no-ci-fn ] && EXCLUDE+=(--exclude "training/ci_fn/")
RUN_ID="${RUN_ID:-p-ba5a0c05}"
STEP="${STEP:-40000}"
DEST="${DEST:-$HOME/out/pod-backup/$RUN_ID/}"
SSH="ssh -p $POD_PORT -i $KEY -o StrictHostKeyChecking=accept-new"
rsync -av --mkpath "${EXCLUDE[@]}" --include "*-$STEP/***" --exclude "*" -e "$SSH" \
  "root@$POD_HOST:/workspace/data/runs/$RUN_ID/" "$DEST"
