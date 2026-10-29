#!/usr/bin/env bash
# Run on the CLUSTER. Push what collect.py reads to the pod (~29 GB): the run's launch_config.yaml,
# ONLY the `decomposition` item of checkpoint 40000, the CI filter's config, alive set, trained CI
# fn, the two reference KL files the run checks against, and the Llama-3.1-8B snapshot.
# One rsync per file, 8 at a time (one ssh stream ~15 MB/s, eight ~105 MB/s). -rltR: MooseFS refuses chown.
#     POD_HOST=<ip> POD_PORT=<port> KEY=~/.ssh/runpod ./push_data.sh
set -euo pipefail
: "${POD_HOST:?}" "${POD_PORT:?}" "${KEY:?}"
RUN=p-ba8b0c08
SRC="$HOME/out/pod-backup/$RUN"
FILTER=addsub-08-last-pos-40000
HUB="$HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B"
SNAP=$(basename "$(ls -d "$HUB"/snapshots/* | tail -1)")
export SSH="ssh -p $POD_PORT -i $KEY -o StrictHostKeyChecking=accept-new -c aes128-gcm@openssh.com"
push() {  # push <dest>: every file under <root>/./ listed on stdin, relative paths kept
  local files; files=$(cat)
  sed 's|.*/\./||' <<<"$files" | xargs -d "\n" dirname | sort -u | sed "s|^|$1|" \
    | $SSH "root@$POD_HOST" 'xargs -d "\n" mkdir -p'
  xargs -P 8 -I{} rsync -rLtR -e "$SSH" "{}" "root@$POD_HOST:$1" <<<"$files"
}
{
  echo "$SRC/./launch_config.yaml"
  echo "$SRC/./ckpts/40000/_CHECKPOINT_METADATA"
  (cd "$SRC" && find ckpts/40000/decomposition "$FILTER/alive" "$FILTER/training/ci_fn" -type f \
     | sed "s|^|$SRC/./|")
  echo "$SRC/./$FILTER/config.yaml"
  echo "$SRC/./$FILTER/dataset/kl_last.npy"
  echo "$SRC/./analysis/reader_writer/variance/alive_only/kl_last.npy"
} | push "/workspace/data/runs/$RUN/"
{
  (cd "$HUB" && find "snapshots/$SNAP" refs -not -type d | sed "s|^|$HUB/./|")
} | push "/workspace/hf/hub/models--meta-llama--Llama-3.1-8B/"
echo "pushed run data and Llama snapshot $SNAP"
