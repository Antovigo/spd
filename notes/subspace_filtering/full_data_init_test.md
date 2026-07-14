# Full-data init test: coupled vs kaiming on pile_llama 4L

Compare **coupled** vs **kaiming** (default) component initialization on the standard
full-data `pile_llama_simple_mlp-4L` decomposition. Too big for the L40 cluster → run on
a **runpod 8×H100** node. **No wandb logging** — everything is written locally to
`metrics.jsonl` and pulled back for plotting.

## Design

6 conditions = `{coupled, kaiming}` × seeds `{0, 1, 2}`, each run through **three phases**
(`raw`, `train`, `nofaith`) = **18 passes**, giving four measurement points:

| Point | Meaning | How it's captured |
|---|---|---|
| **(a) init** | raw init, before warmup | `raw` pass (`warmup=0, steps=1`), metrics @ step 0 |
| **(b) post-warmup** | after 100 faithfulness-warmup steps | `train` pass, metrics @ step 0 |
| **(c) trained** | after 200 main steps, faithfulness on | `train` pass, metrics @ step 200 |
| **(d) trained-nofaith** | after 200 main steps, **no faithfulness** | `nofaith` pass (`warmup=0, steps=200`), metrics @ step 200 |

The `nofaith` phase drops faithfulness entirely — no warmup and `FaithfulnessLoss` removed
from the training loss — to see how the decomposition behaves with no faithfulness
pressure. Weight faithfulness is still **observed** there via an eval probe (the same
`FaithfulnessLoss` run as an eval metric → `eval/loss/FaithfulnessLoss`), logged at init
and step 200. The plots coalesce that with the loss-side `train/loss/FaithfulnessLoss` from
the other phases into one `faithfulness` panel.

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
the train/eval data pipeline **once**, then loops all 18 passes (6 conditions × {raw,
train, nofaith}) in a single process. No reloading between conditions. Passes whose final
step is already in `metrics.jsonl` are **skipped**, so re-launching after a partial or
earlier run only does what's missing. **Re-export the Step 3 env vars (especially
`PARAM_DECOMP_OUT_DIR`) in the shell you re-launch from** — a fresh terminal loses them,
and the driver would then read/write a different `./out` dir and re-run everything.

Runs on `feature/subspace_restriction` because coupled init lives only there
(`PDConfig.weight_init`, `init_coupled_`) and that branch also carries the
`broadcast_buffers=False` DDP fix the multi-forward pile step needs on 8 GPUs. No core
`param_decomp/` changes were needed.

Output: **18 local run dirs** at `$PARAM_DECOMP_OUT_DIR/runs/cinit-<scheme>-s<seed>-<raw|train|nofaith>/`,
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

## Step 2a — provision the RunPod pod

| Choice | What to pick | Why |
|---|---|---|
| **GPU** | **8 × H100 80GB SXM** | Matches `--nproc_per_node=8`. SXM (NVLink) gives faster DDP all-reduce than PCIe. Fewer GPUs also work if they divide the global batch of 64 (1/2/4/8) — just edit `--nproc_per_node`. |
| **Template / image** | Any maintained **CUDA 12.x** image with `git` (e.g. a RunPod PyTorch template) | uv installs Python 3.13 + the pinned torch itself, so the image's Python/torch versions don't matter — it only needs a recent H100 driver (CUDA 12.x) and git. |
| **Deploy type** | **On-Demand** | The run is short (200 steps × 6), but a Spot preemption mid-run would waste the setup. On-Demand avoids that. |

**Storage** — the run writes the venv, HF cache, target model, and `metrics.jsonl` under
`/workspace`, so that must be the persistent volume. The driver **skips checkpointing**
(`_skip_checkpoint`), so there are no multi-GB `model_/training_.pth` files — disk stays small:

| Disk | Size | Notes |
|---|---|---|
| **Volume disk** (persistent, mount `/workspace`) | **~50 GB** | Survives stop/restart. Holds the `.venv` + uv cache (~20 GB), HF cache + target model (~15 GB), and the tiny `metrics.jsonl` logs. No checkpoints are written. |
| **Container disk** (ephemeral) | **~40 GB** | Just the base image / OS. Keep the uv cache off it (`UV_CACHE_DIR` below) so large torch wheels don't fill it. |

## Step 2b — on the pod: setup

Below assumes the persistent volume is mounted at `/workspace`.

```bash
# --- clone just this branch, shallow (fast: skips other branches + history) ---
cd /workspace
git clone --branch feature/subspace_restriction --single-branch --depth 1 \
    https://github.com/goodfire-ai/param-decomp.git
cd param-decomp

# --- install (uv workspace, both packages, no dev deps) ---
export UV_CACHE_DIR=/workspace/uv_cache   # keep torch wheels off the ephemeral container disk
pip install -U uv            # if uv not already present
make install-lab             # == uv sync --all-packages --no-dev (uv fetches Python 3.13)
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

# 8-GPU single-node run (all 18 passes, one process), detached so it survives the
# web terminal disconnecting. nohup inherits the exports above — keep it in this shell.
nohup torchrun --standalone --nproc_per_node=8 \
    notes/subspace_filtering/full_data_init_test_driver.py \
    > /workspace/full_data_init_test.log 2>&1 &
```

Monitor / manage (the web terminal has no tmux; nohup needs none):

```bash
tail -f /workspace/full_data_init_test.log   # Ctrl-C stops watching, not the job
ps aux | grep torchrun                       # or `nvidia-smi` — busy GPUs = running
pkill -f full_data_init_test_driver          # to stop it
```

Progress: 18 passes; the 6 `raw` passes are near-instant (1 step), the 6 `train` passes are
100 warmup + 200 steps, the 6 `nofaith` passes are 200 steps. Results land in
`$PARAM_DECOMP_OUT_DIR/runs/cinit-*/metrics.jsonl`.

---

## Step 4 — retrieve the data to your laptop (before shutting the pod down)

The driver writes no checkpoints, so each run dir holds just the tiny `metrics.jsonl` +
`experiment_config.yaml`.

**Recommended: `runpodctl send/receive`** — peer-to-peer, no SSH keys (rsync/scp need a
public key installed on the pod + the *direct-TCP* endpoint, and error out with a password
prompt otherwise, since `root` has no password).

```bash
# on the POD: tar the metrics, then send (prints a one-time code)
cd /workspace/pd_out/runs
tar czf /workspace/init_test_metrics.tgz cinit-*/metrics.jsonl cinit-*/experiment_config.yaml
runpodctl send /workspace/init_test_metrics.tgz
```

```bash
# on your LAPTOP (install runpodctl first; e.g. brew install runpod/runpodctl/runpodctl)
runpodctl receive <code-printed-by-send>
mkdir -p full_data_init_test_runs && tar xzf init_test_metrics.tgz -C full_data_init_test_runs
```

<details><summary>Alternative: rsync (needs SSH key set up)</summary>

Add your `~/.ssh/id_ed25519.pub` in RunPod → Settings → SSH Public Keys, **restart the pod**
(keys inject at start), and use the **direct-TCP** connect string (`root@<ip> -p <port>`,
*not* the `ssh.runpod.io` proxy — the proxy doesn't support rsync/scp):

```bash
rsync -avz -e "ssh -p <POD_PORT> -i ~/.ssh/id_ed25519" \
    --prune-empty-dirs --include='*/' \
    --include='metrics.jsonl' --include='experiment_config.yaml' --exclude='*' \
    root@<POD_HOST>:/workspace/pd_out/runs/ ./full_data_init_test_runs/
```
</details>

Confirm you pulled all 18, then it's safe to shut the pod down:

```bash
ls -d ./full_data_init_test_runs/cinit-*   # expect 18 dirs
```

## Step 5 — plot (on your laptop)

Runs anywhere with the repo checked out + matplotlib (your laptop; **not** the GF cluster
login node). Reads only the local `metrics.jsonl` files — no wandb, no GPU.

```bash
python notes/subspace_filtering/full_data_init_test_plots.py ./full_data_init_test_runs
# -> ./full_data_init_test_runs/figures/{faith,nofaith}_{faithfulness,recon,l0,ce_kl}.png
#    + summary.csv
```

The runs split into **two comparison series**, each its own set of figures
(`<series>_<group>.png`), sharing the `init` start point:

- **`faith_*.png`** — the faithfulness arm: **init → post-warmup → trained** (faithfulness
  warmup + loss on).
- **`nofaith_*.png`** — the no-faithfulness arm: **init → trained-nofaith** (no warmup, no
  faithfulness loss).

Within each series the script writes one figure **per group** — coupled (red `#d62728`) vs
kaiming (grey `#555555`), mean over seeds + min-max ribbon — plus a shared tidy
`summary.csv` (all scalar keys, one row per scheme/seed/phase, all four phases). Each series
is **self-scaled**: shared log limits are pooled over that series' own phases. Grouping is
**explicit**, keyed off the verified metric-key formats this config emits (each group is
asserted non-empty, so a key-format change fails loudly rather than mis-grouping):

- **`<series>_recon.png`** — the aggregate recon scalar (`eval/loss/PGDReconLoss`) on a
  **shared log y-axis**, pow10 limits (`10**floor(log10(min>0))` .. `10**ceil(log10(max))`)
  pooled over the series. Per-module recon keys (`eval/loss/<Class>/<module>`) **and** the
  hidden-acts recon metrics (`CIHiddenActsReconLoss`, `StochasticHiddenActsReconLoss`) are
  intentionally excluded — only the PGD recon aggregate is the headline.
- **`<series>_l0.png`** — every `eval/l0/*` panel (total + per-layer) on **another shared
  log y-axis**, same pow10 rule.
- **`<series>_faithfulness.png`** — the coalesced `faithfulness` key on its own log panel
  (the headline weight-space metric): `train/loss/FaithfulnessLoss` for the faith phases,
  `eval/loss/FaithfulnessLoss` (the eval probe) for `trained-nofaith`.
- **`<series>_ce_kl.png`** — the CI-masked CE/KL headline (`kl_ci_masked`,
  `ce_unrecovered_ci_masked`, `ce_difference_ci_masked`), each on its own linear axis.

### What the plots answer

- **`faith_faithfulness.png` — the headline** (weight-space faithfulness, faith arm):
  - **init:** does coupled start far more faithful than kaiming?
  - **init → post-warmup:** does 100 warmup steps *substitute* for coupled init (kaiming
    catches up), or only recover part of the gain?
  - **post-warmup → trained:** does any init advantage persist / compound over 200 steps?
- **`nofaith_faithfulness.png`** (the no-faithfulness arm) — from the same `init`, with no
  faithfulness pressure, how far does faithfulness drift over 200 steps, and does coupled
  init hold it closer than kaiming without the loss enforcing it?
- **Eval metrics** (`CEandKLLosses`, `*ReconLoss`, `CI_L0`, `CIMeanPerComponent`, …) tell
  the same story in activation space. Note: at raw **init** the CI function is untrained
  (random), so CI-dependent metrics at that point are noisy — read the clean init signal
  off the weight-space `FaithfulnessLoss`; the post-warmup and trained points are the
  meaningful activation-metric comparisons.
