#!/usr/bin/env bash
# Run on the CLUSTER. Push the decomposition the CI filter reads to the pod: the pinned
# launch_config.yaml and ONLY the `decomposition` item of checkpoint 40000 (~10 GB; the 43 GB
# `training` item, i.e. optimizer state, is never read by the filter).
#     POD_HOST=<ip> POD_PORT=<port> KEY=~/.ssh/<key> ./push_run_data.sh
set -euo pipefail
: "${POD_HOST:?}" "${POD_PORT:?}" "${KEY:?}"
SRC="${SRC:-$HOME/out/pod-backup/p-ba5a0c05}"
DST="root@$POD_HOST:/workspace/data/runs/p-ba5a0c05"
SSH="ssh -p $POD_PORT -i $KEY -o StrictHostKeyChecking=accept-new"
rsync -av --mkpath -e "$SSH" "$SRC/launch_config.yaml" "$DST/"
rsync -av --mkpath -e "$SSH" "$SRC/ckpts/40000/_CHECKPOINT_METADATA" "$SRC/ckpts/40000/decomposition" "$DST/ckpts/40000/"
