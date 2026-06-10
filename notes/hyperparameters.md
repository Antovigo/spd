# Hyperparameter log — Llama-3.1-8B addition tPD

Targeted decomposition of `model.layers.18.mlp.{gate,up,down}_proj` on the addition task
(`a+b=`, a,b ∈ [1,100]). Base config:
`param_decomp_lab/experiments/lm/llama-3.1-8b_addition_targeted.yaml`.

## Setup

- Hardware: 1 node, 8×L40 (48 GB each), SLURM partition `compute`. Using **2 GPUs**.
- Launch: own sbatch → `torchrun --standalone --nproc_per_node=2 -m
  param_decomp_lab.experiments.lm.run <cfg>` (the repo `--dp` path snapshots code +
  under-requests memory, so we drive torchrun directly with explicit `--mem`/`--cpus`).
- DDP: config `batch_size` is the **global** batch; per-GPU = batch_size / world_size
  (`data.py::rank_batch_size`). So global batch must be divisible by 2.
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

## LR / hyperparameter trials

| # | change vs base | steps | recon trend | KL | CI L0 | verdict |
|---|---|---|---|---|---|---|
| _tbd_ | | | | | | |

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

## Batch-size search (2×L40, 8B real model)

| global batch | eval | peak/GPU | result |
|---|---|---|---|
| _tbd_ | | | |
