#!/usr/bin/env bash
# Run on the CLUSTER. Pull analysis/resid_rms (and the pilot) and the pod logs back.
#     POD_HOST=<ip> POD_PORT=<port> KEY=~/.ssh/runpod ./pull.sh
set -uo pipefail
: "${POD_HOST:?}" "${POD_PORT:?}" "${KEY:?}"
REMOTE="root@$POD_HOST:/workspace/data/runs/p-ba8b0c08/analysis"
LOCAL="$HOME/out/pod-backup/p-ba8b0c08/analysis"
SSH="ssh -p $POD_PORT -i $KEY -o StrictHostKeyChecking=accept-new"
mkdir -p "$LOCAL/resid_rms/logs/pod"
rc=0
rsync -a -e "$SSH" "$REMOTE/resid_rms" "$REMOTE/resid_rms-pilot" "$LOCAL/" || rc=$?
rsync -a -e "$SSH" "root@$POD_HOST:/workspace/data/logs/" "$LOCAL/resid_rms/logs/pod/" || rc=$?
exit $rc
