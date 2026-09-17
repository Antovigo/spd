# CI filter on one H100 (Runpod)

Runs both CI filter objectives (`param_decomp/ci_filter`) on `addsub-all-layers-4xh100-05`
(`p-ba5a0c05`, step 40000): a 10-minute smoke, objective 1 (last-position KL), then objective
2 (last-position integer KL) starting from objective 1's CI fn. Outputs land in
`$DATA_ROOT/runs/p-ba5a0c05/analysis/ci_filter/step_40000/<cf-id>/` and are pulled back into
the cluster copy of the run.

## Recipe

The template seats (`param_decomp/ci_filter/configs/`) train with the decomposition's own
target-pass recon, `recon: {kind: merged_stochastic_ppgd}` (SPEC S34: stochastic-subset draws plus
a 1/3 persistent-PGD adversarial share, WEIGHT DELTA ON; the adversary ascends once per step, from
the main backward — `n_warmup_steps: 0`, where -05 used 2, since the 20k-prompt pool repeats), scored by the filter objective at the last position. Evaluations stay deterministic
(CI masks, delta off). Every run logs live to the wandb project `param-decomp-llama` (needs
`WANDB_API_KEY` in `/workspace/secrets.env`).

## Why one H100 and not the L40s

A 1x L40 only fits the deterministic recon (`ci_masked`) at microbatch 256 with the gradient sum
held on the host, at 12.8 s/step (~18 h per objective), and the persistent adversary does not fit
beside it. Static memory on one device is ~31 GB (Llama bf16 16, prepared components 3.7, CI fn
fp32 3.6, Adam 7.2); the adversary adds ~3.9 GB (fp32 sources + Adam moments for 1024 x 5 x
62,688) and its source gradient 1.3 GB. Measured on an L40, one 512-prompt forward/backward needs
an 8.6 GiB arena, so a 1024-prompt step needs ~17 GB; the warmup ascents and the main step are
separate jits, so they do not stack. On an 80 GB H100 (76 GB pool) that is ~55-60 GB at the
whole batch in one microbatch — an extrapolation the smoke checks first. If it OOMs: `MICRO=512`
(the adversary is sliced per microbatch; every ascent still updates the whole bundle once).

## 1. Pod

- 1x H100 80 GB (SXM or PCIe: single GPU, no collectives). Host driver >= r570 (CUDA 12.8).
- Volume: >= 60 GB (venv ~7, Llama weights 16, checkpoint 10, XLA cache, outputs ~10 incl.
  two saved CI fns at ~3.9 GB each).
- Expected time (estimate): with no warmup ascents a step is one clean forward plus one masked
  forward/backward, like the deterministic step (L40: 3.0 s per 1024-prompt step at microbatch
  512; H100 is typically 2-3x faster), so ~1.5 s/step, ~2-2.5 h per objective, plus a few minutes
  per pool evaluation (6 per objective), ~15-30 min for the grid and ~5 min for the two PGD evals.
  Each warmup step (`n_warmup_steps`) adds another masked forward/backward.

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
  all on 0.0245 (0.0139 over all positions = the decomposition's UnmaskedReconLoss). The
  smoke's pool (1..30) gives different numbers.
- `step N: ...`: `train/adv_fraction` ramps 0 -> 1/3 over the first 5%, `train/source_lr`
  0 -> 0.01 over the first 1.25% (wandb).
- `step N: ... X s/step`: the first logged step includes compilation.
- `RESOURCE_EXHAUSTED` at the first step: rerun with a microbatch, e.g.
  `MICRO=512 ./run_pipeline.sh`. At an evaluation: `EVAL_BATCH=500`. In the grid pass:
  `GRID_CHUNK=500`. Knobs are read when each stage's config is generated
  (`$DATA_ROOT/ci_filter/configs/`).

The two filters are named `addsub-05-filter-last-pos` and `addsub-05-filter-integers` (output dir
and wandb run; override with `OBJ1_ID` / `OBJ2_ID`). Restarting a single stage:
`./run_pipeline.sh --only obj1`, or `./run_pipeline.sh --only obj2 --init-id addsub-05-filter-last-pos`.
Output dirs fail closed: delete a failed stage's dir before rerunning it under the same name (its
wandb run can stay or go; every launch gets a fresh wandb id). The smoke (`cf-smoke`) does not
log to wandb and removes its own dir.

## 5. Pull the outputs back (from the cluster)

```bash
POD_HOST=<ip> POD_PORT=<port> KEY=~/.ssh/<key> ./pull_outputs.sh     # add --no-ci-fn to skip the ~3.9 GB CI fns
```

They land in `~/out/pod-backup/p-ba5a0c05/analysis/ci_filter/step_40000/<cf-id>/` with the pod
logs in `analysis/ci_filter/pod-logs/`. Open `<cf-id>/ab_grids/index.html` over `file://`.
