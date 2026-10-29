# Pod environment for the residual-RMS collection. Source it in every shell:
#     source /workspace/spd/notes/resid_rms/pod/env.sh
export VOLUME="${VOLUME:-/workspace}"
export REPO="${REPO:-$VOLUME/spd}"                 # clone of Antovigo/spd @ experiment/arith_representations
export DATA_ROOT="${DATA_ROOT:-$VOLUME/data}"
export RUN_DIR="${RUN_DIR:-$DATA_ROOT/runs/p-ba8b0c08}"
export VENV_PY="${VENV_PY:-$REPO/.venv/bin/python}"
export HF_HOME="$VOLUME/hf" HF_HUB_CACHE="$VOLUME/hf/hub" HF_HUB_OFFLINE=1   # weights pushed from the cluster
export UV_CACHE_DIR="$VOLUME/uv-cache" TMPDIR="$DATA_ROOT/tmp"
export WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
mkdir -p "$DATA_ROOT"/{runs,tmp,logs} "$HF_HUB_CACHE" "$UV_CACHE_DIR"
