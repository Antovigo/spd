#!/usr/bin/env bash
# Run on the CLUSTER. Push what the analysis reads to the pod (~29 GB): the run's pinned
# launch_config.yaml, ONLY the `decomposition` item of checkpoint 40000 (10 GB), the CI filter's
# config, alive set and trained CI fn (3.4 GB), and the Llama-3.1-8B snapshot (15 GB; the pod
# runs with HF_HUB_OFFLINE=1, so no HF token is needed there).
# One rsync per file, 8 at a time: a single ssh stream from the cluster runs at ~15 MB/s, eight
# reach ~105 MB/s (2026-10-15). -rltR, not -a: the pod volume (MooseFS) refuses chown.
#     POD_HOST=<ip> POD_PORT=<port> KEY=~/.ssh/runpod ./push_data.sh
set -euo pipefail
: "${POD_HOST:?}" "${POD_PORT:?}" "${KEY:?}"
RUN=p-ba8b0c08
SRC="$HOME/out/pod-backup/$RUN"
FILTER=addsub-08-last-pos-40000
HUB="$HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B"
SNAP=$(basename "$(ls -d "$HUB"/snapshots/* | tail -1)")
export SSH="ssh -p $POD_PORT -i $KEY -o StrictHostKeyChecking=accept-new -c aes128-gcm@openssh.com"
push() {  # push <root> <dest>: every file under <root>/./ listed on stdin, relative paths kept
  xargs -P 8 -I{} rsync -rLtR -e "$SSH" "{}" "root@$POD_HOST:$1"
}
{
  echo "$SRC/./launch_config.yaml"
  echo "$SRC/./ckpts/40000/_CHECKPOINT_METADATA"
  (cd "$SRC" && find ckpts/40000/decomposition "$FILTER/alive" "$FILTER/training/ci_fn" -type f \
     | sed "s|^|$SRC/./|")
  echo "$SRC/./$FILTER/config.yaml"
} | push "/workspace/data/runs/$RUN/"
{
  (cd "$HUB" && find "snapshots/$SNAP" refs | sed "s|^|$HUB/./|")
} | grep -v "/\.$" | push "/workspace/hf/hub/models--meta-llama--Llama-3.1-8B/"
echo "pushed run data and Llama snapshot $SNAP"
