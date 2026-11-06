#!/usr/bin/env bash
# Run on the cluster. Copies what DAS reads to a pod, at the same home-relative paths as the settings
# file, and installs the repo at the pushed branch.
#     POD_HOST=<ip> POD_PORT=<port> notes/das/launch/set_up_runpod_pod.sh
set -euo pipefail
: "${POD_HOST:?}" "${POD_PORT:?}"
KEY=~/.ssh/runpod
SSH="ssh -p $POD_PORT -i $KEY -o StrictHostKeyChecking=accept-new -c aes128-gcm@openssh.com"
RUN=out/pod-backup/p-ba8b0c08
PRUNING=$RUN/analysis/pruning/step_40000/addsub-08-prune
WEIGHTS=.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
$SSH "root@$POD_HOST" "command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync >/dev/null)"
cd ~
rsync -rLtR -e "$SSH" "$RUN/prompts/tokenized.npz" "$PRUNING/references.npz" "$PRUNING/config.yaml" \
  "$RUN/analysis/das/prompt_pairs" "root@$POD_HOST:/root/"
find "$WEIGHTS" -type f | grep -E 'json$|safetensors$' \
  | xargs -P 4 -I{} rsync -rLtR -e "$SSH" {} "root@$POD_HOST:/root/"
$SSH "root@$POD_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
[ -d spd ] || git clone -q --branch experiment/das https://github.com/Antovigo/spd.git spd
cd spd && git pull -q
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$DRIVER" -ge 580 ]; then EXTRA=cuda13; else EXTRA=cuda; fi
uv sync --frozen --no-dev --extra "$EXTRA" 2>&1 | tail -1
.venv/bin/python -c "import jax; print(jax.devices())"
REMOTE
