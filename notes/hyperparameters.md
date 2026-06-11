# Hyperparameter log — Llama-3.1-8B addition tPD

Targeted decomposition of `model.layers.18.mlp.{gate,up,down}_proj` on the addition task
(`a+b=`, a,b ∈ [1,100]). Base config:
`param_decomp_lab/experiments/lm/llama-3.1-8b_addition_targeted.yaml`.

## Setup

- Hardware: 1 node, 8×L40 (48 GB each), SLURM partition `compute`. Using **2 GPUs**.
- Hardware update: using **4 GPUs** (was 2). Launcher `~/pd_scratch/run_ddp.sbatch`
  derives `--nproc_per_node` from `$CUDA_VISIBLE_DEVICES`, so `sbatch --gpus=N` overrides.
- Launch: own sbatch → `torchrun --standalone --nproc_per_node=$NGPU -m
  param_decomp_lab.experiments.lm.run <cfg>` (the repo `--dp` path snapshots code +
  under-requests memory, so we drive torchrun directly with explicit `--mem`/`--cpus`).
- DDP: config `batch_size` is the **global** batch; per-GPU = batch_size / world_size
  (`data.py::rank_batch_size`). So global batch must be divisible by 4.
- GPU cap (8 total, job 330 holds 1): only **one** 4-GPU job at a time → 8B sweeps run
  sequentially.
- Scratch configs (wandb off, short runs) live in `~/pd_scratch/`; metrics read from the
  run's local `metrics.jsonl`. `PARAM_DECOMP_OUT_DIR=~/out`.

## Goal of tuning

tPD tug-of-war: faithfulness/reconstruction (recon losses, KL to target) vs sparsity
(CI L0). Read the **start of training** (first few hundred steps) to judge: (a) does it
run, (b) does recon drop, (c) is the LR stable.

## Batch-size search (2×L40)

| global batch | per-GPU | result | peak mem/GPU | notes |
|---|---|---|---|---|
| _tbd_ | | | | |

## LR / hyperparameter trials (8B, batch 192, C 512)

Throughput ~3.13 s/it at batch 192 → 20k steps ≈ 17.4 h (fits 24 h QOS).
Base HPs were a hybrid (LR 1e-4 + recon 1.0 from 4L-targeted; structure from 8B-full;
impmin 1e-4 = own guess). The 8B full-data run used LR 5e-5 / impmin 1e-6 — more
conservative. User: targeted opt is simpler → expect higher LR is fine.

| LR | job | steps | train/loss/total trend | stable? | verdict |
|---|---|---|---|---|---|
| 1e-4 | 355 | 100 | 0.47 → 0.024 @100 | yes | healthy baseline |
| 3e-4 | 356 | — | INVALID | — | infra OOM: leftover proc from killed 355 on same GPUs |
| 1e-3 | 357 | 150 | 0.47→0.038→0.016@100→0.013@125 | yes | **best**: fastest, recon 0.008, L0~9 |
| 3e-3 | 358 | 150 | 0.47→**4.82@25**→0.28@100→0.14@150 | no | overshoots, slow recovery |

**Chosen LR = 1e-3** (20× the 8B full-data 5e-5; confirms targeted opt tolerates higher
LR). 3e-3 too hot (spike), 1e-4 slower. Did NOT tune impmin (1e-4) — watch L0 vs recon.
Lesson: after `scancel`, GPU mem isn't freed instantly — don't relaunch on same node
immediately (caused the 356 OOM).

## Serious run #2 (job 359, run_id llama8b-add-02) — CURRENT

4×L40, global batch 192, C=512, **LR 1e-3**, impmin 1e-4, 20k steps (~17.4 h),
save_every 2000, full eval, wandb param-decomp-llama. Monitoring start-of-training.
- Start-of-training healthy: step-0 full eval cleared (no OOM); train/loss/total
  0.47→0.012@100→0.014@200 (matches 1e-3 test). ~3.3 s/it → ~18-19 h for 20k + eval.
  Watching for crash (→ resume from latest model_<step>.pth) / completion.

## Log

- Setup: `uv sync --all-packages` (plain `uv sync` misses the lab pkg incl. `fire`).
  WandB authenticates automatically via `~/.netrc`. HF login required for the gated
  Llama-3.1-8B (`hf auth login`).
- Base config validation caught: `TargetedCIHeatmap.prompts_file` must equal
  `data.prompts_file` (probes = target distribution) → dropped that eval metric (10k
  prompts → 10k-row figure). Base config now validates.
- 8B blocked on HF gating (token valid, repo access not yet granted). To unblock infra
  work, made an ungated twin: `llama_pile_12L_addition_targeted.yaml` — pretrained pile
  Llama 12L (`t-f99617bb`), target `h.8.mlp.{c_fc,down_proj}` (layer 8 < 12), gpt-neox
  tokenizer, nontarget = Pile (pre-tokenized). Same small CI transformer + targeted
  loss/eval set. Validates.
- Scratch tooling: `~/pd_scratch/make_variant.py --base {8b,12l}` emits variants;
  `~/pd_scratch/run_2gpu.sbatch <cfg> [run_id]` → `torchrun --nproc_per_node=2`
  (partition compute, 2 gpus, 16 cpu, 120G), logs peak GPU mem.
- Infra smoke job 332 died at launch (logs/ dir missing for `--output`); fixed.
- **Infra smoke PASSED (job 333, run pile12l-smoke-02)**: 12L, global batch 16, 30
  steps, light eval, 2×L40, `exit_code=0`. DDP + target/nontarget passes + losses +
  eval + checkpoint + wandb all work. Training healthy (CE/KL diff ↓, target recon ↓,
  ~101/200 components active). Sampler reported peak 32.7 GB but it queried all node
  GPUs → untrusted; fixed sampler to `-i $CUDA_VISIBLE_DEVICES`.
- srun interactive launch is broken on this cluster (Communication connection failure)
  → use sbatch only.

## Batch-size search (2×L40, 12L infra model)

Probes: short runs, peak GPU mem per-GPU via `nvidia-smi -i $CUDA_VISIBLE_DEVICES`.
No GPU isolation on this node (job sees all 8) → the `-i` filter is essential.

| global batch | eval | peak/GPU (MiB) | result |
|---|---|---|---|
| 32  | none | 3066  | fits |
| 96  | none | 6640  | fits |
| 192 | none | 11890 | fits |
| 512 | full | 32400 | fits |
| 640 | full | 40290 | **fits — safe max (~87%)** |
| 704 | full | 44174 | fits, ~no headroom |
| 768 | full | 43302 | fits, ~no headroom |
| 1024| full | OOM   | — |

~55 MiB/global-batch-unit + ~1.3 GB fixed. **Chosen: global batch 640** for 12L runs.
NB: 12L memory profile does NOT transfer to 8B — re-probe 8B after HF access.

## Pivot to 8B (HF access granted)

- Llama-3.1-8B access granted; weights pre-downloaded to HF cache (15 GB, no GPU).
- 12L LR sweep (jobs 344-346) cancelled before completion (not informative) and all
  12L run dirs removed from `~/out/runs` per request. Infra workflow validated on 12L:
  DDP + tPD pipeline + batch probing + sweep launching all work.
- Now: 8B smoke → re-probe 8B max batch (expect far smaller: 16 GB frozen weights/GPU +
  ~176 M delta params for layer-18 MLP) → tune 8B LR on start-of-training.

## 8B fp32→bf16 fix (critical)

First 8B smoke (job 347) OOM'd at global batch 8 (45.7 GB/GPU): `build_target` called
`from_pretrained` with no dtype → weights loaded in **fp32 (32 GB)**. `autocast_bf16`
only casts compute, not storage. Fix: added `dtype` field to `HFTarget` (default
`"auto"` = checkpoint dtype) + set `bfloat16` in the 8B config. Re-smoke (job 348):
`exit_code=0`, peak **19.4 GB/GPU** at batch 8. Headroom restored.

## Batch-size search (4×L40, 8B real model, bf16)

| global batch | eval | peak/GPU (MiB) | result |
|---|---|---|---|
| 8   | none  | 19438 | fits |
| 32  | full  | 21610 | fits |
| 64  | full  | 24378 | fits |
| 128 | full  | 30204 | fits |
| 192 | full  | 35792 | **fits — chosen (78%)** |
| 256 | full  | 41254 | fits but 90%, tight for long run |

~90 MiB/global-batch-unit + ~16 GB bf16 weights. **Chosen: global batch 192.**
C is cheap in memory (low-rank components) → spent headroom on subcomponents instead of
a bigger batch: **C 200 → 512** per matrix (adds <1 GB).

## Serious run (job 355, run_id llama8b-add-01)

- 4×L40, global batch 192, C=512, 20k steps, LR 1e-4 (base), impmin 1e-4, save_every
  2000, wandb on (param-decomp-llama). Walltime 24 h (QOS `normal` cap; resumable via
  checkpoints if 20k needs longer).
- Cluster note: QOS max wall = 24 h; partition `compute` MaxTime unlimited but QOS binds.
- Monitoring start-of-training for divergence / recon trend / throughput.
