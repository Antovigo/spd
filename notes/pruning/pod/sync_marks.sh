#!/usr/bin/env bash
# Run on the CLUSTER, from this worktree. Every INTERVAL seconds (default 300), pull the pod's
# learned masks (`learn/`) into the cluster copy and rewrite the ab-grid applet's `pruned.js`, so
# its "pruning mask" menu lists every run in the folder. A static page cannot list a folder, so
# this loop is what keeps the menu current.
#     POD_HOST=<ip> POD_PORT=<port> KEY=~/.ssh/runpod ./sync_marks.sh
# Top-level `learn/lam_*` (the constant-LR sweep, obsolete) and `learn/kl3000` (abandoned at
# step 400 with an empty mask) are not pulled.
set -uo pipefail
: "${POD_HOST:?}" "${POD_PORT:?}" "${KEY:?}"
RUN_ID="${RUN_ID:-p-ba8b0c08}"
STEP="${STEP:-40000}"
NAME="${NAME:-addsub-08-prune}"
FILTER="${FILTER:-addsub-08-last-pos}"
DEFAULT="${DEFAULT:-cos3000/lam_2e-05}"
INTERVAL="${INTERVAL:-300}"
RUN="$HOME/out/pod-backup/$RUN_ID"
PRUNE="$RUN/analysis/pruning/step_$STEP/$NAME"
REMOTE="/workspace/data/runs/$RUN_ID/analysis/pruning/step_$STEP/$NAME/learn/"
SSH="ssh -p $POD_PORT -i $KEY -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20"
cd "$(dirname "$0")/../../.."
while true; do
  rsync -rtl --mkpath -e "$SSH" --exclude "/lam_*" --exclude "/kl3000" \
    "root@$POD_HOST:$REMOTE" "$PRUNE/learn/" \
    && .venv/bin/python -m param_decomp.pruning.marks --prune_dir "$PRUNE" \
      --grids_dir "$RUN/$FILTER-$STEP/ab_grids" --default "$PRUNE/learn/$DEFAULT" 2>&1 \
      | grep -v absl | tail -1
  sleep "$INTERVAL"
done
