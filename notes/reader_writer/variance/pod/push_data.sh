#!/usr/bin/env bash
# Run on the CLUSTER. Push what the analysis reads to the pod (~29 GB): the run's pinned
# launch_config.yaml, ONLY the `decomposition` item of checkpoint 40000 (10 GB), the CI filter's
# config, alive set and trained CI fn (3.4 GB), and the Llama-3.1-8B snapshot (15 GB; the pod
# runs with HF_HUB_OFFLINE=1, so no HF token is needed there).
#     POD_HOST=<ip> POD_PORT=<port> KEY=~/.ssh/runpod ./push_data.sh
set -euo pipefail
: "${POD_HOST:?}" "${POD_PORT:?}" "${KEY:?}"
RUN=p-ba8b0c08
SRC="$HOME/out/pod-backup/$RUN"
FILTER=addsub-08-last-pos-40000
DST="root@$POD_HOST:/workspace/data/runs/$RUN"
SSH="ssh -p $POD_PORT -i $KEY -o StrictHostKeyChecking=accept-new"
rsync -a --mkpath -e "$SSH" "$SRC/launch_config.yaml" "$DST/"
rsync -a --mkpath -e "$SSH" "$SRC/ckpts/40000/_CHECKPOINT_METADATA" "$SRC/ckpts/40000/decomposition" "$DST/ckpts/40000/"
rsync -a --mkpath -e "$SSH" "$SRC/$FILTER/config.yaml" "$SRC/$FILTER/alive" "$DST/$FILTER/"
rsync -a --mkpath -e "$SSH" "$SRC/$FILTER/training/ci_fn" "$DST/$FILTER/training/"
HUB="$HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B"
SNAP=$(basename "$(ls -d "$HUB"/snapshots/* | tail -1)")
rsync -aL --mkpath -e "$SSH" "$HUB/snapshots/$SNAP/" \
  "root@$POD_HOST:/workspace/hf/hub/models--meta-llama--Llama-3.1-8B/snapshots/$SNAP/"
rsync -a --mkpath -e "$SSH" "$HUB/refs" "root@$POD_HOST:/workspace/hf/hub/models--meta-llama--Llama-3.1-8B/"
echo "pushed run data and Llama snapshot $SNAP"
