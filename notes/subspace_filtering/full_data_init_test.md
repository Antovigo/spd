# Full-data init test: coupled vs kaiming on pile_llama 4L

Compare **coupled** vs **kaiming** (default) component initialization on the standard
full-data `pile_llama_simple_mlp-4L` decomposition. Too big for the L40 cluster → run on
a **runpod 8×H100** node. **No wandb logging** — everything is written locally to
`metrics.jsonl` and pulled back for plotting.

## Design

6 conditions = `{coupled, kaiming}` × seeds `{0, 1, 2}`. Each condition is measured at
three points:

| Point | Meaning | How it's captured |
|---|---|---|
| **(a) init** | raw init, before warmup | `raw` pass: `warmup=0, steps=1`, metrics @ step 0 |
| **(b) post-warmup** | after 100 faithfulness-warmup steps | `train` pass, metrics @ step 0 |
| **(c) trained** | after 200 main steps | `train` pass, metrics @ step 200 |

- **Warmup = 100** (not the config's 400), then **200 main steps** — as requested.
- The `raw` pass takes one optimizer step at step 0 (`steps` is a `PositiveInt`, so 0 is
  invalid), but the step is bounded by `grad_clip_norm=0.01` → `‖Δ‖ ≤ lr·0.01 = 5e-7`,
  i.e. negligible. The **exact** raw-init weight faithfulness is
  `train/loss/FaithfulnessLoss` @ step 0 of the `raw` pass — computed on the pre-step
  weights.
- **Data is held constant** across all conditions (`DATA_SEED=0`); only `pd.seed` varies.
  So the three seeds are three draws of the initialization / stochastic training on
  identical data — init is the only thing being compared.
- Everything else (losses, optimizers, `C` values, target modules, eval metrics) is the
  canonical `pile_llama_simple_mlp-4L.yaml`, unchanged. The driver overrides only
  `wandb` (→ off), `weight_init`, `seed`, `faithfulness_warmup_steps`, `steps`, and the
  eval/log cadence (so metrics land at step 0 and 200).

**Load once:** the driver (`full_data_init_test_driver.py`) builds the target model and
the train/eval data pipeline **once**, then loops all 12 passes (6 conditions × {raw,
train}) in a single process. No reloading between conditions.

Runs on `feature/subspace_restriction` because coupled init lives only there
(`PDConfig.weight_init`, `init_coupled_`) and that branch also carries the
`broadcast_buffers=False` DDP fix the multi-forward pile step needs on 8 GPUs. No core
`param_decomp/` changes were needed.

Output: **12 local run dirs** at `$PARAM_DECOMP_OUT_DIR/runs/cinit-<scheme>-s<seed>-<raw|train>/`,
each with a `metrics.jsonl`.

---

## Step 1 — locally: commit + push

Run from the `subspace_restriction` worktree
(`/mnt/nw/home/a.vigouroux/Code/param-decomp/subspace_restriction`):

```bash
source .venv/bin/activate
git status                                   # confirm branch = feature/subspace_restriction
git add notes/subspace_filtering/full_data_init_test_driver.py \
        notes/subspace_filtering/full_data_init_test_plots.py \
        notes/subspace_filtering/full_data_init_test.md
git commit -m "experiment(subspace): coupled-vs-kaiming init test on full-data pile 4L"
git push
```

---

## Step 2 — on the runpod pod: setup

Fresh 8×H100 pod. Pick a **persistent volume** path (survives pod restarts) for outputs
and the HF cache — below assumes `/workspace`.

```bash
# --- clone just this branch, shallow (fast: skips other branches + history) ---
cd /workspace
git clone --branch feature/subspace_restriction --single-branch --depth 1 \
    https://github.com/goodfire-ai/param-decomp.git
cd param-decomp

# --- install (uv workspace, both packages, no dev deps) ---
pip install -U uv            # if uv not already present
make install-lab             # == uv sync --all-packages --no-dev
source .venv/bin/activate

# --- credentials ---
cp .env.example .env
#   edit .env: set WANDB_API_KEY and WANDB_ENTITY.
#   Even though we DON'T log to wandb, the key is still needed to DOWNLOAD the target
#   model, which is pulled from goodfire/spd/runs/t-9d2b8f02 (target.spec.run_path).
#   The key must have read access to goodfire/spd.
```

## Step 3 — on the pod: launch

```bash
# persistent output + HF cache so results/data survive pod restarts
export PARAM_DECOMP_OUT_DIR=/workspace/pd_out
export HF_HOME=/workspace/hf_cache

# H100 headroom + the DDP env the repo normally sets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HF_HUB_ETAG_TIMEOUT=30
export HF_HUB_DOWNLOAD_TIMEOUT=30

# 8-GPU single-node run (all 12 passes, one process)
torchrun --standalone --nproc_per_node=8 \
    notes/subspace_filtering/full_data_init_test_driver.py \
    2>&1 | tee /workspace/full_data_init_test.log
```

To detach, wrap the `torchrun` in `tmux`/`nohup`. Progress: 12 passes; the 6 `raw` passes
are near-instant (1 step), the 6 `train` passes are 100 warmup + 200 steps each. Results
land in `$PARAM_DECOMP_OUT_DIR/runs/cinit-*/metrics.jsonl`.

---

## Step 4 — retrieve the data to your laptop (before shutting the pod down)

The `metrics.jsonl` + `experiment_config.yaml` files are tiny; checkpoints
(`model_*.pth`) are large and not needed for plots. Pull just the metrics from the pod.
Fill in your pod's SSH host/port (runpod shows these under "Connect"):

```bash
# on your LAPTOP, from wherever you want the data:
rsync -avz -e "ssh -p <POD_PORT>" \
    --prune-empty-dirs \
    --include='*/' \
    --include='metrics.jsonl' \
    --include='experiment_config.yaml' \
    --exclude='*' \
    root@<POD_HOST>:/workspace/pd_out/runs/ \
    ./full_data_init_test_runs/
```

To also grab the checkpoints (large — only if you want offline recompute), rerun without
the `--include/--exclude` filters, or add `--include='model_*.pth'` before the
`--exclude='*'` line. Confirm you pulled all 12:

```bash
ls -d ./full_data_init_test_runs/cinit-*   # expect 12 dirs
```

Now it's safe to shut the pod down.

## Step 5 — plot (on your laptop)

Runs anywhere with the repo checked out + matplotlib (your laptop; **not** the GF cluster
login node). Reads only the local `metrics.jsonl` files — no wandb, no GPU.

```bash
python notes/subspace_filtering/full_data_init_test_plots.py ./full_data_init_test_runs
# -> ./full_data_init_test_runs/figures/{faithfulness,recon,l0,ce_kl}.png + summary.csv
```

The script writes one figure **per group** — coupled (red `#d62728`) vs kaiming (grey
`#555555`) across init → post-warmup → trained, mean over seeds + min-max ribbon — plus a
tidy `summary.csv` (all scalar keys, one row per scheme/seed/phase). Grouping is
**explicit**, keyed off the verified metric-key formats this config emits (each group is
asserted non-empty, so a key-format change fails loudly rather than mis-grouping):

- **`recon.png`** — the aggregate recon scalars (`eval/loss/PGDReconLoss`,
  `eval/loss/CIHiddenActsReconLoss`, `eval/loss/StochasticHiddenActsReconLoss`) on **one
  shared log y-axis**, pow10 limits (`10**floor(log10(min>0))` .. `10**ceil(log10(max))`)
  pooled over the group. Per-module recon keys (`eval/loss/<Class>/<module>`) are
  intentionally excluded — the aggregate is the headline.
- **`l0.png`** — every `eval/l0/*` panel (total + per-layer) on **another shared log
  y-axis**, same pow10 rule.
- **`faithfulness.png`** — `train/loss/FaithfulnessLoss` on its own log panel (the
  headline weight-space metric).
- **`ce_kl.png`** — the CI-masked CE/KL headline (`kl_ci_masked`,
  `ce_unrecovered_ci_masked`, `ce_difference_ci_masked`), each on its own linear axis.

### What the plots answer

- **`train/loss/FaithfulnessLoss` — the headline** (weight-space faithfulness):
  - **init:** does coupled start far more faithful than kaiming?
  - **init → post-warmup:** does 100 warmup steps *substitute* for coupled init (kaiming
    catches up), or only recover part of the gain?
  - **post-warmup → trained:** does any init advantage persist / compound over 200 steps?
- **Eval metrics** (`CEandKLLosses`, `*ReconLoss`, `CI_L0`, `CIMeanPerComponent`, …) tell
  the same story in activation space. Note: at raw **init** the CI function is untrained
  (random), so CI-dependent metrics at that point are noisy — read the clean init signal
  off the weight-space `FaithfulnessLoss`; the post-warmup and trained points are the
  meaningful activation-metric comparisons.
