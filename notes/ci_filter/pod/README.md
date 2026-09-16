# CI filter on one H100 (Runpod)

Runs both CI filter objectives (`param_decomp/ci_filter`) on `addsub-all-layers-4xh100-05`
(`p-ba5a0c05`, step 40000): a 10-minute smoke, objective 1 (last-position KL), then objective
2 (last-position integer KL) starting from objective 1's CI fn. Outputs land in
`$DATA_ROOT/runs/p-ba5a0c05/analysis/ci_filter/step_40000/<cf-id>/` and are pulled back into
the cluster copy of the run.

## Why one H100 and not the L40s

One L40 does not fit at any microbatch (128/256/512 all OOM once the gradient accumulator is
live; jobs 11804, 11810, 11812, 2026-09-16). Static memory on one device is ~31 GB (Llama bf16
16, prepared components 3.7, CI fn fp32 3.6, Adam 7.2) plus 3.6 GB of gradients per
microbatch. Measured on an L40 without accumulation: a 512-prompt microbatch fits in a
43.8 GB pool with an 8.6 GiB step arena, 1024 does not. On an 80 GB H100 (76 GB pool) the
whole 1024-prompt batch should run as ONE microbatch (no accumulator at all), at an estimated
~50-55 GB peak; the pool evaluation at 1000 prompts and grid chunks of 1000 should also fit.
These are extrapolations — the smoke checks them before the long stages.

## 1. Pod

- 1x H100 80 GB (SXM or PCIe: single GPU, no collectives). Host driver >= r570 (CUDA 12.8).
- Volume: >= 60 GB (venv ~7, Llama weights 16, checkpoint 10, XLA cache, outputs ~10 incl.
  two saved CI fns at ~3.9 GB each).
- Expected time: ~2-3 h per objective on H100 (L40 measured 3.0 s per 1024-prompt step at
  microbatch 512; H100 is typically 2-3x faster), plus a few minutes per pool evaluation (6 per objective) and ~15-30 min for the grid.

## 2. Code, environment, weights, secrets

Same as `notes/dual_objective/addsub-all-layers-4xh100-guide.md` sections 2, 3b and 4, with
the branch changed:

```bash
cd /workspace
git clone -b feature/ci_filter https://github.com/Antovigo/spd.git spd
cd spd && git log -1 --oneline
curl -LsSf https://astral.sh/uv/install.sh | sh && source "$HOME/.local/bin/env"
source notes/dual_objective/addsub-all-layers/env.sh
uv python install 3.12
uv sync --frozen --no-dev --extra cuda
.venv/bin/python -c "import jax; print(jax.__version__, jax.devices())"   # 1x CudaDevice
# HF_TOKEN in /workspace/secrets.env (guide section 4), then the weights (guide 3b):
source notes/dual_objective/addsub-all-layers/env.sh
$VENV_PY - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("meta-llama/Llama-3.1-8B",
      allow_patterns=["*.safetensors", "*.json", "tokenizer*"], ignore_patterns=["original/*"]))
PY
```

No datasets and no neuron-rank artifact are needed (checked: the run's deliverable resolves
against an empty data root).

If a pod from the all-layers run is still around with its volume, only `git fetch && git
checkout feature/ci_filter && uv sync --frozen --no-dev --extra cuda` is needed.

## 3. Run data (from the cluster)

```bash
cd ~/Code/param-decomp/ci_filter/notes/ci_filter/pod
POD_HOST=<ip> POD_PORT=<port> KEY=~/.ssh/<key> ./push_run_data.sh    # ~10 GB
```

It sends `launch_config.yaml` and only the `decomposition` item of `ckpts/40000` (the 43 GB
`training` item is optimizer state the filter never reads).

## 4. Launch (on the pod)

```bash
source /workspace/spd/notes/dual_objective/addsub-all-layers/env.sh
cd /workspace/spd/notes/ci_filter/pod
./run_pipeline.sh                          # smoke -> obj1 -> obj2, detached
tail -f $DATA_ROOT/logs/ci_filter.latest.log
```

The whole pipeline is one detached process; closing the shell does not stop it. It stops at
the first failing stage. What to look for in the log:

- `eval @ 0`: pre-training pool scores. Reference on this checkpoint (full-vocab KL, last
  position): continuous CI 0.058, rounded CI 0.051, alive set (12,473 components) 0.086,
  all on 0.0245. The smoke's pool (1..30) gives different numbers.
- `step N: ... X s/step`: the first logged step includes compilation.
- `RESOURCE_EXHAUSTED` at the first step: rerun with a microbatch, e.g.
  `MICRO=512 ./run_pipeline.sh`. At an evaluation: `EVAL_BATCH=500`. In the grid pass:
  `GRID_CHUNK=500`. Knobs are read when each stage's config is generated
  (`$DATA_ROOT/ci_filter/configs/`).

Restarting a single stage: `./run_pipeline.sh --only obj1`, or
`./run_pipeline.sh --only obj2 --init-id <objective-1 id>` (ids in `$DATA_ROOT/ci_filter/obj1.id`
and `obj2.id`). A failed stage leaves its output dir behind; delete it before rerunning with
the same id (new ids are minted on every launch, so this only matters for `cf-smoke`, which the
pipeline removes itself).

## 5. Pull the outputs back (from the cluster)

```bash
POD_HOST=<ip> POD_PORT=<port> KEY=~/.ssh/<key> ./pull_outputs.sh     # add --no-ci-fn to skip the ~3.9 GB CI fns
```

They land in `~/out/pod-backup/p-ba5a0c05/analysis/ci_filter/step_40000/<cf-id>/` with the pod
logs in `analysis/ci_filter/pod-logs/`. Open `<cf-id>/ab_grids/index.html` over `file://`.
