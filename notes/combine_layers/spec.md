# Spec — combine_layers scripts

Goal: combine several single-block targeted decompositions of Llama-3.1-8B (addsub)
into one model whose eligible layers are all replaced by their decomposed versions,
while the PD objectives stay satisfied.

Test runs: `addsub-L16-04-init-proj`, `addsub-L17-04-init-proj`,
`addsub-L18-05-coupled`, `addsub-L19-05` (latest `model_*.pth` of each).

## Core addition: `grouped_global` CI mode (`param_decomp/ci_fns.py`)

`GroupedGlobalCiConfig` (`mode: grouped_global`) + `GroupedGlobalCiFnWrapper`: several
independent global CI fns, one per named *group* of decomposition targets. Groups are
fnmatch patterns over resolved module paths; they must partition the targets (each
target matches exactly one group, each group matches ≥1 target). Each group's CI fn
only ever sees its own group's inputs, so it is weight-compatible with the
`_global_ci_fn` of a run that decomposed exactly those modules. State-dict prefix:
`ci_fn._group_ci_fns.<group>.*`. This is a full `CiConfig` union member: YAML-parseable
and trainable by `Trainer` (needed for objective 2).

Tests: `param_decomp/tests/test_grouped_ci_fn.py` (group assignment invariants;
grouped wrapper == single global CI fn on the same weights).

## `param_decomp_lab/combine/assembly.py`

- `load_source_runs(runs)` — resolve run ids/paths, one `SourceRun` per run. Derives
  the group name `layers<N>` from the (asserted unique) block index in the run's
  decomposition targets. Asserts the runs are combinable: same target spec, CI config,
  sigmoid type, delta usage, sampling; distinct layers.
- `build_combined_component_model(sources, target_model, run_batch, device)` — builds a
  `ComponentModel` over the union of all sources' decomposition targets with the
  grouped CI config, then loads each source checkpoint's `_components.*` tensors and
  remaps its `ci_fn._global_ci_fn.*` to `ci_fn._group_ci_fns.layers<N>.*`. Checkpoints
  are mmap-loaded, so their embedded 16 GB `target_model.*` copies are never read; the
  caller's target model is used instead (identical frozen weights).

## `param_decomp_lab/combine/eval_combined.py` (objective 1)

Standalone eval of assembled decompositions, no training. Mirrors the Trainer's eval
pass (same metric classes, same context construction, bf16 autocast).

- Subjects: each run alone (`--include_singles`, default true; identical code path,
  sanity-checks the assembly against end-of-training logs) + all runs `combined`.
- Target-distribution metrics (addsub prompts, eval split seed): `TargetReconLoss`
  (stochastic / ci_masked / rounded / delta_only recon + total_l0), `PGDReconLoss`
  (config copied from the runs' own eval config: 20 steps, step 0.1,
  shared_across_batch), `CI_L0` per block group at `--ci_alive_thr`.
- Nontarget metrics (FineWeb stream, under `delta_override(1.0)`):
  `NontargetReconLoss`.
- `--ci_thr` (default **0.01**, matching the training logs' rounded recon threshold) is
  the rounding threshold: components with CI > thr get mask 1, else 0. `--ci_alive_thr`
  (default 0.1) only affects L0 counts.
- Seed is re-applied before each subject so stochastic masks / PGD are comparable.
- Output: JSON `{meta, results: {subject: {metric: value}}}`.

## `param_decomp_lab/combine/finetune.py` (objectives 2 and 3)

Fine-tunes the assembled decomposition with the full targeted `Trainer` loop
(nontarget FineWeb pass included). Builds the combined `LMExperimentConfig` in code
from the source runs (written to the run dir as usual, so `SavedLMRun` reloads work):

- Union of the sources' decomposition targets; data / eval / nontarget blocks copied
  from the (identical) source configs.
- Importance-minimality pinned to its **end-of-training state**: the sources anneal
  coeff ×2→×1 and p 2→0.5 across training, so fine-tuning uses constant base coeff
  (default: min over sources = 3e-5) at p = 0.5 — continuing the converged objective,
  not replaying the anneal.
- LR schedules keep the sources' shape (cosine, final 0.1×) with fine-tune start
  values: `--components_lr` (default 1e-4), `--ci_fn_lr` (default 5e-5).
- `--ci_fn_mode=grouped` (default, objective 2): per-block CI fns loaded from the
  sources. `--ci_fn_mode=global_fresh` (objective 3): ONE randomly-initialised global
  CI fn over all matrices; `--ci_d_model` / `--ci_n_blocks` override its size;
  components still initialised from the sources.
- `--freeze_ci_fns` / `--freeze_components`: effective freeze via `FROZEN_LR=1e-12`
  (schedules require start_val > 0; hard `requires_grad_(False)` after DDP
  construction would stall the reducer; Adam's step is ≈lr regardless of grad scale).
- `--wandb=False` + `--nontarget_batch_size` for cheap probes.
- Weights are loaded into `trainer.component_model` after `Trainer` construction
  (all DDP ranks load identical files, so no desync). `combine_provenance.json` in the
  run dir records the source checkpoints and freeze flags.

DDP: `torchrun --standalone --nproc_per_node=N -m param_decomp_lab.combine.finetune`
(note: at 32/rank on L40s, DDP gradient buckets OOM — production runs are
single-GPU at global batch 32).

`eval_combined` also accepts `--finetuned=<ids>`: fine-tuned combined runs
(grouped_global CI) evaluated as additional subjects from their own checkpoints,
through the identical metric loop.

## Plotting

- `param_decomp_lab/combine/plot_obj1.py` — obj-1 figures: per-batch recon dot plot
  (singles vs combined, target + nontarget panels, log x) and recon-vs-blocks
  scaling chain.
- `param_decomp_lab/combine/plot_obj2.py` — obj-2 figures: merged dot plot
  (singles / raw combined / fine-tuned) and fine-tuning trajectory (recon + L0 vs
  step, single-block range as reference band).
