#!/usr/bin/env bash
# Run on the cluster. Installs Feucht et al.'s repository at its pinned dependencies on a pod, with the
# Llama-3.1-8B weights, on the pod's volume so they survive a stop.
#     POD_HOST=<ip> POD_PORT=<port> notes/das/replicate_feucht/set_up_pod.sh
set -euo pipefail
: "${POD_HOST:?}" "${POD_PORT:?}"
KEY=~/.ssh/runpod
SSH="ssh -p $POD_PORT -i $KEY -o StrictHostKeyChecking=accept-new -c aes128-gcm@openssh.com"
WEIGHTS=$HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
$SSH "root@$POD_HOST" "command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync >/dev/null); mkdir -p /workspace/models/Llama-3.1-8B"
find -L "$WEIGHTS" -type f | grep -E 'json$|safetensors$' \
  | xargs -P 4 -I{} rsync -Lt -e "$SSH" {} "root@$POD_HOST:/workspace/models/Llama-3.1-8B/"
$SSH "root@$POD_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=/workspace/uv-cache
cd /workspace
[ -d arithmetic-wild ] || git clone -q https://github.com/goodfire-ai/arithmetic-wild
cd arithmetic-wild
git log -1 --format='%h %s'
uv sync --frozen 2>&1 | tail -1
.venv/bin/python -c "import torch, pyvene, causalab; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
REMOTE
