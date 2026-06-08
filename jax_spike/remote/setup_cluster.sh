#!/usr/bin/env bash
# One-time cluster setup: create the feature/nano-pd-jax worktree at ~/pd-nano-jax
# and build the CUDA JAX venv (on the login node; wheels install fine without a GPU).
set -euo pipefail
SSH="ssh -o RemoteCommand=none -o RequestTTY=no a-login"
BRANCH="${BRANCH:-feature/nano-pd-jax}"

$SSH "
  set -e
  cd ~/param-decomp
  git fetch -q origin $BRANCH
  if [ ! -d ~/pd-nano-jax ]; then
    git worktree add ~/pd-nano-jax $BRANCH
  else
    git -C ~/pd-nano-jax fetch -q origin $BRANCH && git -C ~/pd-nano-jax reset -q --hard origin/$BRANCH
  fi
  cd ~/pd-nano-jax/jax_spike
  if [ ! -d .venv-cuda ]; then
    echo '[setup] building .venv-cuda (jax[cuda12])...'
    uv venv .venv-cuda --python 3.13 >/dev/null
    source .venv-cuda/bin/activate
    uv pip install -q 'jax[cuda12]' numpy
  fi
  source .venv-cuda/bin/activate
  python -c 'import jax; print(\"jax\", jax.__version__, \"installed\")'
  echo '[setup] done'
"
