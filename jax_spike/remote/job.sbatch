#!/usr/bin/env bash
#SBATCH --job-name=jax-spike
#SBATCH --partition=h200-dev
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/polished-lake/home/oli/jax_spike/logs/%j.out
set -euo pipefail
cd ~/jax_spike

# idempotent CUDA venv (persists across syncs; excluded from rsync --delete)
if [ ! -d .venv-cuda ]; then
  echo "[setup] creating .venv-cuda + installing jax[cuda12]..."
  uv venv .venv-cuda --python 3.13 >/dev/null
  source .venv-cuda/bin/activate
  uv pip install -q "jax[cuda12]" numpy
else
  source .venv-cuda/bin/activate
fi

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import jax; print('jax', jax.__version__, '|', jax.devices())"
echo "=== running command ==="
cat _remote_cmd.sh
echo "---"
bash _remote_cmd.sh
