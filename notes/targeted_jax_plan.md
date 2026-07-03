# Targeted Parameter Decomposition (tPD) — JAX Implementation Plan

Porting **targeted PD** onto the JAX single-pool engine (`feature/jax`). Branch:
`feature/targeted-jax`. Method: *Targeted Recovery of Weight-Space Mechanisms From Neural
Networks* (Vigouroux & Sharkey, 2026). Torch reference: `origin/feature/targeted` (core
commit `af31c9957`; latest `origin/experiment/8B_targeted` adds only analysis tooling).
The torch design doc (`notes/targeted_implementation_main.md` on that branch) is the
conceptual source; this doc is the JAX-native re-derivation.

Semantics are pinned in `param_decomp/SPEC.md` §11 (S-tPD-*). Every change is checked
against SPEC by invariant ID (the repo's "one rule").

## Concept

Decompose only the mechanisms causally important on a narrow **target** dataset, while a
broad **non-target** stream keeps behavior faithful off-target. Each optimizer step runs
two recon passes:

- **Target pass** — narrow prompts; delta adversarially ablated (force output from the
  rank-1 subcomponents); **no faithfulness** (delta must stay nonzero); subcomponents need
  not sum to `W`.
- **Non-target pass** — broad distribution; delta pinned m_δ = 1.0 so `components + Δ`
  reconstruct exactly; stochastic component masks; impmin coeff ×ratio (~2); PPGD /
  unmasked / hidden-acts losses dropped.

Mechanisms firing only off-target are absorbed into Δ; subcomponents specialize to the
target and stay inert elsewhere.

## The key porting insight (torch → JAX)

The torch impl pinned the delta mask with a process-global `ContextVar` read at mask
construction, and ran the two passes as **two sequential `.backward()`s** — dragging in the
DDP `find_unused_parameters` coverage saga (2-rank smoke test, deferred `no_sync()`).

JAX is simpler and different:
- **No ContextVar** (nothing survives a `jit` boundary; the HLO-baking rule wants explicit
  args). The delta value is a plain scalar threaded into the two mask-construction
  functions (`delta_override`), already scaffolded default-off.
- **One `value_and_grad`, not two backwards.** Both passes' scalar losses are computed
  inside one `loss_fn` and summed; `jax.grad` differentiates once. All the DDP coverage
  machinery is moot.
- **`model` and `batch` are already jit ARGS** (`train.py::step`), so a second non-target
  batch threads in cleanly.

The delta is already the **C+1-th adversarial source channel** (`adversary.source_masks`
→ `delta_masks[site] = source[..., -1]`; `components.site_out` applies
`delta_mask[...,None] * (x@W.T − xV@U)`). So the *target* pass needs no new delta handling
— the persistent-PGD adversary already ablates it. Only the *non-target* pass needs the
`delta_override=1.0` pin.

---

## Phase 0 — DONE (this branch, scaffold)

- **Delta seams (default-off, behavior-preserving):** `delta_override: float | None = None`
  added to `param_decomp/adversary.py::source_masks` and
  `param_decomp/train.py::stochastic_entry_masks`. `None` = today's behavior byte-identical;
  a float pins every delta mask to that constant. (SPEC S-tPD-1.)
- **Lab module skeleton:** `param_decomp_lab/experiments/lm_targeted/`
  (`config.py`, `data.py`, `run.py`, `launch.py`, `prompts/`, `configs/`).
- **Console script:** `pd-lm-targeted` → `experiments.lm_targeted.launch:cli`.
- **Config classes:** `TargetPromptsDataConfig`, `NontargetConfig`,
  `LMTargetedExperimentConfig` (+ validators: no FaithfulnessLoss, `faithfulness_warmup_steps
  == 0`). `EXCLUDED_NONTARGET_LOSS_CONFIGS` listed.
- **SPEC §11** draft + **example YAML** ported from the torch numpy/pandas config.

Everything below is the fill-in.

---

## Phase 1 — engine: the non-target pass (core, additive, default-off)

`param_decomp/` — keep the engine generic (no lab import; pinned by
`test_runtime_standalone`). Add ONE optional bundle threaded through, default `None`.

1. **`NontargetPass` dataclass** (frozen; in `run.py` or `built_run.py`):
   ```python
   @dataclass(frozen=True)
   class NontargetPass:
       sample_batch: Callable[[int], Any]        # broad-stream per-step batch
       loss_metrics: list[LossMetricConfig]       # filtered non-target loss set
       impmin_coeff_ratio: float                  # already folded into the coeffs? decide
   ```
   Decision: fold `impmin_coeff_ratio` into the loss configs lab-side
   (`build_nontarget_loss_metrics`) so the engine stays ratio-agnostic; then the field is
   informational only (or dropped).

2. **`run_decomposition_training(..., nontarget: NontargetPass | None = None)`** — new
   trailing optional param. Pass it to `make_train_step`. When `None`, the step is
   byte-identical to today.

3. **`make_train_step(..., nontarget=None)`** builds a second set of recon loss terms
   (`build_loss_terms(nontarget.loss_metrics, lm.site_names)`, closed over static). Inside
   `loss_fn`, after the target recon grid:
   - fetch the non-target batch (thread it in as a second jit arg — see below),
   - run the CI fn on it (its own `ci.lower`),
   - build masks with `stochastic_entry_masks(..., delta_override=1.0)` /
     `source_masks(..., delta_override=1.0)`,
   - run the non-target recon grid, sum its coeff-weighted terms into `total_loss`.
   One `value_and_grad`. (SPEC S-tPD-2/3.)

4. **Threading the second batch.** `step(model, state, batch, key)` becomes
   `step(model, state, batch, nontarget_batch, key)` (or `batch` becomes a
   `(target, nontarget)` tuple). The engine loop calls both `sample_batch(step)` and
   `nontarget.sample_batch(step)`. Prefer an explicit extra arg over a tuple so the
   untargeted path's signature is unchanged (make the extra arg optional / a separate jitted
   step chosen at factory time).
   > Design choice to settle here: (a) one `step` with an optional second batch (branch on
   > a static `has_nontarget`), or (b) two step factories. (a) keeps one code path; (b)
   > keeps the untargeted `step` byte-identical. Lean (a) with a static flag.

5. **Persistent adversary.** The non-target pass uses NO PPGD (excluded), so it does not
   touch `state.adversaries`. Only the target pass drives the adversary lifecycle
   (warmup/final ascent) exactly as today. Assert the non-target loss set contains no
   PersistentPGD (lab-side, `build_nontarget_loss_metrics`).

6. **SPEC §11** — promote S-tPD-* from DRAFT once wired; cite IDs in the commit.

## Phase 2 — config + build (lab)

`param_decomp_lab/experiments/lm_targeted/config.py`:
- **`build_nontarget_loss_metrics`** — filter `EXCLUDED_NONTARGET_LOSS_CONFIGS`; for
  `ImportanceMinimalityLossConfig`, `model_copy(update={"coeff": coeff * ratio})`; assert a
  full-model stochastic recon loss remains (S-tPD-2).
- **`build_targeted_from_schema`** — mirror `lm.config.build_from_schema`:
  `LMTargetedExperimentConfig(**raw)` → `build_experiment_config`-style `BuiltRun`, but the
  engine's `data` is the NON-TARGET parquet `DataConfig` (reuse `lm.config._data` on
  `cfg.nontarget.data`); the target prompts + the non-target pass ride to the composition
  root separately (return `(BuiltRun, tPD extras)` or a small `TargetedBuiltRun` wrapper).
- Reconcile the example YAML's `ci_config` / loss-config field names against the real JAX
  schema (the torch names may differ, e.g. `StochasticReconSubsetLoss` routing).

## Phase 3 — target-prompt loader (lab)

`param_decomp_lab/experiments/lm_targeted/data.py`:
- **`load_prompt_tokens`** — tokenize the prompts file once; pad to `max_seq_len`; RAISE on
  over-length. Decide the pad token and whether target recon scores only the final position
  (the paper reconstructs the last-token completion — likely add a `recon_positions` notion,
  mirroring the torch `recon_positions` commit).
- **`TargetPromptServer`** — per-step batch matching `ShardServer`'s contract: whole pool if
  `batch_size >= n_prompts`, else seeded sample-without-replacement; shard over the mesh via
  the same `_global_token_batch` helper.

## Phase 4 — composition root (lab)

`param_decomp_lab/experiments/lm_targeted/run.py`:
- **`train_targeted`** — mirror `lm.run.train`: build the non-target parquet loader
  (`ShardServer` at `cfg.nontarget.batch_size`) AND the target loader
  (`TargetPromptServer`); build `eval_fn`; call `run_decomposition_training(...,
  nontarget=NontargetPass(nontarget_sample_batch, build_nontarget_loss_metrics(cfg), ...))`
  where the engine's own `sample_batch` = the TARGET stream.
- **`main`** — reuse `lm.run.main`'s process setup (sigterm/init_distributed/XLA cache/HF
  hardening/mesh/config pin), then `train_targeted`. Factor the shared setup or duplicate
  per the additive-merge preference.

## Phase 5 — tPD eval metrics (JAX-native rewrites)

The torch `Metric` impls were dropped in the JAX migration, so these are rewrites in the
JAX in-loop eval style (`eval.py` fast tier / `slow_eval.py` plot tier), not ports:
- **`TargetReconLoss` / `NontargetReconLoss`** — recon under 4 mask strategies (stochastic,
  CI-masked, rounded, delta-only) at delta 0.0 (target) / 1.0 (non-target). The non-target
  one runs under `delta_override=1.0`.
- **`TargetedCIHeatmap`** — target-prompt CI row over a non-target CI row (probe forward on
  the prompts, tokenized in the metric's construction).
- **`WeightMagnitude`** — per-site sorted `‖V_c‖·‖U_c‖`, weights-only (no batch).
- Register in the eval config union + the LM `_eval` conversion / slow-tier set.

## Phase 6 — launcher + smoke (lab)

- **`launch.py`** — reuse `experiments.lm.launch` almost verbatim; validate
  `LMTargetedExperimentConfig`; point the rank command at `experiments.lm_targeted.run`.
- **Smoke:** `runtime.dp: null` inline run of the numpy/pandas config for a few steps;
  confirm the non-target loss logs, non-target CI L0 drops, delta stays ~1 on non-target,
  tPD figures render. Then a toy convergence check (optionally port the torch TMS/ResidMLP
  `active_indices` convergence tests, which passed at ratio 20 / 20k steps for TMS).

---

## Files (merge surface)

**New (this branch):** `param_decomp_lab/experiments/lm_targeted/{__init__,config,data,run,
launch}.py`, `prompts/numpy_and_pandas.txt`, `configs/numpy_pandas_4L_targeted.yaml`;
`notes/targeted_jax_plan.md`; `pd-lm-targeted` in `param_decomp_lab/pyproject.toml`.

**Edited — core (additive, default-off):** `adversary.py` (+`delta_override`), `train.py`
(+`delta_override`; Phase 1 non-target pass), `run.py` (+`NontargetPass` param; loop),
`built_run.py` (maybe `NontargetPass`), `SPEC.md` (§11), `configs.py` (Phase 5 eval configs),
`eval.py`/`slow_eval.py` (Phase 5 metrics).

**Untouched:** `components.py` (delta application already generic), `ci_fn.py`, the
persistent adversary lifecycle (target-pass only).

## Open decisions

1. Second-batch threading: optional extra `step` arg + static `has_nontarget` flag (lean)
   vs. two step factories vs. `(target, nontarget)` tuple batch.
2. Whether `BuiltRun` gains a `nontarget` field or the composition root passes
   `NontargetPass` to the engine directly (keeps `BuiltRun` LM-agnostic — lean).
3. Target recon over final position only vs. all positions (paper: last-token completion).
4. tPD eval metric surface — how much of the torch set to port vs. minimal (Target/Nontarget
   recon + heatmap are the load-bearing ones).
