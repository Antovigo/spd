# Pod environment for the addsub-all-layers line. Source it in every shell and every script:
#     source /workspace/spd/notes/dual_objective/addsub-all-layers/env.sh
# Everything that must survive a pod restart lives under $VOLUME (the Runpod NETWORK volume,
# mounted at /workspace); the container disk is ephemeral. Override any variable by
# exporting it BEFORE sourcing. Secrets are NOT here: they live in $VOLUME/secrets.env
# (chmod 600, never committed) and are sourced last if present.

export VOLUME="${VOLUME:-/workspace}"
export REPO="${REPO:-$VOLUME/spd}"                 # git clone of Antovigo/spd @ feature/dual_obj_jax
export DATA_ROOT="${DATA_ROOT:-$VOLUME/data}"      # runs/ datasets/ neuron_ranks/ wandb/ tmp/ logs/ ladder/
export VENV_PY="${VENV_PY:-$REPO/.venv/bin/python}"

# The library does not read PARAM_DECOMP_OUT_DIR (data_root is the positional argument);
# exported for parity with the cluster scripts and for anything you paste from them.
export PARAM_DECOMP_OUT_DIR="$DATA_ROOT"
export WANDB_DIR="$DATA_ROOT/wandb"
export HF_HOME="$VOLUME/hf"                        # hub cache = $HF_HOME/hub (what hf_snapshot_dir reads)
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_ENABLE_HF_TRANSFER=0                 # plain downloads; hf_transfer is not in the lock
export TMPDIR="$DATA_ROOT/tmp"
export UV_CACHE_DIR="$VOLUME/uv-cache"             # wheels cache; the container disk is small
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"  # the config's
# launch_env re-exports the same value inside the process; this copy is for ad-hoc shells.

mkdir -p "$DATA_ROOT"/{runs/by-name,datasets,neuron_ranks,wandb,tmp,logs,ladder,pids} "$HF_HUB_CACHE" "$UV_CACHE_DIR"

# runtime.compilation_cache_dir is `~/.cache/param-decomp/xla` (a per-user name, per
# CONFIGS.md rule 5); point it at the volume so compiles survive a pod restart.
if [ ! -e "$HOME/.cache/param-decomp" ]; then
  mkdir -p "$HOME/.cache" "$VOLUME/xla-cache"
  ln -sfn "$VOLUME/xla-cache" "$HOME/.cache/param-decomp"
fi

# HF_TOKEN + WANDB_API_KEY (see the guide, "Secrets").
if [ -f "$VOLUME/secrets.env" ]; then
  # shellcheck disable=SC1091
  set -a; . "$VOLUME/secrets.env"; set +a
fi
