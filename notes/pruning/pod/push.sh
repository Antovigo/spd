#!/usr/bin/env bash
# Run on the CLUSTER (rsync -rtl: a network-FS /workspace rejects the chown of -a). Ship everything the pruning run reads to a fresh GPU pod:
#   - the code: `git archive` of the worktree's HEAD (committed state only, no push needed);
#   - the run: launch_config.yaml + checkpoint 40000's `decomposition` item (~10 GB);
#   - the ceiling CI filter: config.yaml, alive/ and training/ci_fn/ (~3.4 GB);
#   - the Llama-3.1-8B HF snapshot from the cluster cache (~15 GB; no HF token on the pod);
#   - the pruning config (CONFIG, default ~/pd_scratch/pruning/addsub-08-prune.yaml).
#     POD_HOST=<ip> POD_PORT=<port> KEY=~/.ssh/runpod ./push.sh
set -euo pipefail
: "${POD_HOST:?}" "${POD_PORT:?}" "${KEY:?}"
RUN_ID="${RUN_ID:-p-ba8b0c08}"
FILTER="${FILTER:-addsub-08-last-pos}"
STEP="${STEP:-40000}"
SRC="${SRC:-$HOME/out/pod-backup/$RUN_ID}"
CONFIG="${CONFIG:-$HOME/pd_scratch/pruning/addsub-08-prune.yaml}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
SSH="ssh -p $POD_PORT -i $KEY -o StrictHostKeyChecking=accept-new"
POD="root@$POD_HOST"
DST="$POD:/workspace/data/runs/$RUN_ID"

echo "code: $(git -C "$REPO" log -1 --oneline)"
git -C "$REPO" archive --format=tar HEAD | $SSH "$POD" 'mkdir -p /workspace/spd && tar -x -C /workspace/spd'
$SSH "$POD" "echo $(git -C "$REPO" rev-parse HEAD) > /workspace/spd/COMMIT"
rsync -rtl --mkpath -e "$SSH" "$CONFIG" "$POD:/workspace/data/pruning/config.yaml"
rsync -rtl --mkpath -e "$SSH" "$SRC/launch_config.yaml" "$DST/"
rsync -rtl --mkpath -e "$SSH" "$SRC/ckpts/$STEP/_CHECKPOINT_METADATA" "$SRC/ckpts/$STEP/decomposition" "$DST/ckpts/$STEP/"
F="$SRC/$FILTER-$STEP"
rsync -rtl --mkpath -e "$SSH" "$F/config.yaml" "$F/alive" "$F/training" "$DST/$FILTER-$STEP/"
rsync -rtl --mkpath -e "$SSH" "$HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B" "$POD:/workspace/hf/hub/"
echo "pushed"
