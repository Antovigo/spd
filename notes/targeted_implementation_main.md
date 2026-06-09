# Targeted Parameter Decomposition (tPD) — Implementation Plan

Add **targeted parameter decomposition** to the `param_decomp` codebase. The codebase is split into a
shared core package `param_decomp/` (trainer, masks, loss metrics, configs) and an application package
`param_decomp_lab/` (experiments, datasets, eval metrics, plotting, wandb). This plan exploits that
split to keep the core nearly untouched.

Correctness is defined by the behavior described in **Concept** + the override semantics below, and is
pinned by the convergence tests in **Tests**.

## Goals
1. Core targeted training, runnable on toy models (TMS, ResidMLP) and LMs.
2. tPD evals: target/nontarget recon, targeted CI heatmap, nontarget CI-mean plot; plus the generic
   `WeightMagnitude` plot.
3. Maximal isolation: confine new logic to new modules + lab-side hooks so upstream merges stay clean —
   except where consistency wins out. The eval path deliberately widens core `EvalLoop` + the eval
   driver (small, additive, default-off) so the new eval metrics mirror existing ones exactly rather
   than inventing a metric-owned-iterator pattern. Toy and LM share one code path.

## Concept

A normal PD run decomposes a target model's weights into components plus a learned residual ("delta"),
training so that stochastically masked subsets of components reconstruct the target model's output.
Causal-importance (CI) values gate each component.

tPD additionally decomposes using **narrow target inputs** while training on a **broad nontarget
distribution** to preserve behavior. Each training step does two passes:

1. **Target pass** — a broad/narrow *target* batch through the normal PD losses (the delta is masked
   stochastically as usual).
2. **Nontarget pass** — a *nontarget* batch with the delta component **forced fully on** (`delta`
   mask = `1.0`), so `components + delta` reconstruct the target model exactly. The reconstruction
   losses then penalize any component that changes the output on nontarget data, and
   importance-minimality pushes CI → 0. Together these drive components to be **inactive** on nontarget
   data while remaining free to specialize on target data.

The two passes accumulate gradients into one optimizer step.

---

## Design decisions (read first)

The isolation goal is achievable cleanly because the application layer is already separate from the
stable kernel. Five decisions follow:

1. **Pin the delta mask with a context-managed override (a `ContextVar`), not a threaded argument.**
   Per-step state is otherwise threaded explicitly through a frozen `MetricContext`
   (`param_decomp/metrics/context.py`), and core has **no** existing `ContextVar`. A contextvar is the
   one mechanism that lets the delta mask be pinned without editing `context.py` or any recon-loss
   metric body: it is *read* at two core sites (`masks.py`, `pgd_utils.py`) — plus *asserted-absent* at
   a third (`persistent_pgd_state.py`, the PPGD path that targeted never uses) — is default-off, and
   leaves the high-churn metric files byte-identical. (A `MetricContext.delta_override` field instead
   would touch `context.py` plus every recon metric's `update()`.)

2. **All new config lives in lab.** The core `PDConfig` carries no task/data config — per-experiment
   data config lives on lab's `ExperimentConfig` (`cfg.data`). So `nontarget_data`, `active_indices`,
   `prompts_file`, the nontarget batch sizes, the impmin ratio, and the targeted validators all go on
   the **lab** side. Core `PDConfig` is **untouched**.

3. **The nontarget loss set is built in lab and handed to the trainer.** `build_nontarget_loss_configs`
   operates on core config *types* but is a lab function. The lab experiment passes the filtered config
   list + the nontarget loader into `Trainer.run(...)`. Core never learns about "targeted" — it runs
   whatever nontarget metrics it is given, under `delta_override(1.0)`.

4. **Metric short names are class vars, auto-collected.** The wandb short-name map is built by reading
   `cls.short_name` off each metric class (`param_decomp_lab/infra/wandb.py::_build_short_names`).
   Setting `short_name` on each new metric is sufficient; there is no separate registry to edit.

5. **The eval driver feeds nontarget data; the new metrics stay byte-for-byte standard.** No existing
   eval metric owns a dataloader or does I/O in `compute()` — the driver loops the loader, builds one
   `ctx` per batch via `_build_metric_context`, and every metric just does `reset()` → `update(ctx)`
   ×`n_steps` → `compute()`. To keep the new evals maximally consistent with that, **widen `EvalLoop`**
   with an optional `nontarget_loader` + `nontarget_metrics` list, and have the driver run a
   *mirror-image* second eval loop over the nontarget loader (wrapped in `delta_override(1.0)`), feeding
   `nontarget_metrics` exactly as the first loop feeds `metrics`. The new metrics then accumulate in
   `update(ctx)` and reduce in `compute()` like every existing metric — none holds an iterator, none
   does I/O in `compute()`. This costs a small, additive, backward-compatible widening of core
   `EvalLoop` + the `Trainer.run` eval driver (both default-off when `nontarget_loader is None`), which
   the maintainer has explicitly chosen over the earlier "metric-owned iterator" scheme **because eval
   consistency outranks keeping those two core files untouched**. Probe-only inputs (the heatmap's
   target row: one-hots over `active_indices` / the LM prompts) follow the existing
   `IdentityCIError`/`UVPlots` precedent — cache metadata in `update`, synthesize the probe and run a
   forward on `self.model` in `compute`.

**Net core surface:** new `param_decomp/targeted.py` (contextvar only) + 1 read in `masks.py` + 2 reads
in `pgd_utils.py` + 1 fail-fast assert in `persistent_pgd_state.py` + the guarded nontarget pass in
`optimize.py` + the `EvalLoop` widening and mirror nontarget eval loop in `optimize.py`. Everything
else is lab.

> **Three delta-mask sites, not two.** The delta/residual mask is built at *three* core sites:
> `masks.py` (stochastic), `pgd_utils.py` (per-step PGD), and `persistent_pgd_state.py` (PPGD). Only
> the first two are instrumented to read the override; the PPGD site is left alone because PPGD is
> excluded from the nontarget loss set (§ "Losses excluded"). That exclusion is the *only* thing
> keeping a process-global override from silently producing wrong masks in the PPGD path, so it is
> backstopped with a fail-fast assert at the PPGD site rather than trusted implicitly.

### Relevant APIs
- Trainer: `param_decomp/optimize.py::Trainer.run(train_loader, sink, cadence, eval_loop=None)`. One
  forward + one backward per step. `_build_metric_context(batch, ...)` (same file) runs the DDP
  forward, computes CI, and returns a frozen `MetricContext`; it calls `move_batch_to_device`.
  `weight_deltas = self.component_model.calc_weight_deltas()` is computed fresh each step *outside*
  `bf16_autocast`.
- Delta mask (stochastic): `param_decomp/masks.py::calc_stochastic_component_mask_info(causal_importances,
  component_mask_sampling, weight_deltas, router)` — the residual mask is `torch.rand(leading_dims, ...)`
  built per layer when `weight_deltas is not None` (the only random-delta site). `make_mask_infos`
  (same file) assembles `ComponentsMaskInfo` from explicit component/delta masks.
- Delta mask (PGD): `param_decomp/metrics/pgd_utils.py::_init_adv_sources` (`mask_c = module_c if
  weight_deltas is None else module_c + 1`) and `_construct_mask_infos_from_adv_sources` (last channel
  `[..., -1]` is the delta source, indexed with `batch_dims`). PPGD lives in
  `param_decomp/metrics/persistent_pgd_recon.py`.
- Loss metrics are `Metric[Cfg]` subclasses (`param_decomp/metrics/base.py`) with
  `update(ctx)->Tensor|None`, dispatched from `LOSS_METRIC_CLASSES` /
  `instantiate_metrics(pd_config, component_model, device)` in `param_decomp/metrics/dispatch.py`.
  Loss configs: `AnyLossMetricConfig` in `param_decomp/configs.py`. `StochasticReconLoss`
  (`param_decomp/metrics/stochastic_recon.py`) is the reference for stochastic mask-based recon.
- Eval metrics: `EVAL_METRIC_CLASSES` / `AnyEvalMetricConfig` in
  `param_decomp_lab/eval_metrics/__init__.py`. Base `Metric` exposes `reset/update/compute/bind`,
  `log_namespace`, `slow`, `short_name`. `CIMeanPerComponent`
  (`param_decomp_lab/eval_metrics/ci_mean_per_component.py`) is the reference for a CI-aggregating
  plot metric.
- Datasets: `SparseFeatureDataset` (`param_decomp_lab/experiments/tms/data.py`), `ResidMLPDataset`
  (`param_decomp_lab/experiments/resid_mlp/data.py`, subclass). LM data:
  `param_decomp_lab/experiments/lm/data.py::create_lm_data_loader(cfg, *, split, batch_size, seed, ...)`
  → `(loader, tokenizer)`.
- Experiment entry points (build loaders + `Trainer`): `param_decomp_lab/experiments/{tms,resid_mlp,lm}/run.py`
  (`main`; `_fresh_main` + `_resume_main` for LM). `ExperimentConfig[T, D]` lives in
  `param_decomp_lab/experiments/utils.py`.
- Plotting: `param_decomp_lab/eval_metrics/plotting.py` (has `_render_figure`,
  `plot_mean_component_cis_both_scales`, …). No plotting in core.

---

## Delta-override mechanism

Rather than thread a delta-value argument through every mask-construction call site, pin the residual
("delta") mask to a constant within a `with delta_override(v):` scope, read at the two construction
sites. No signatures change; `context.py` and every recon-loss metric stay byte-identical.

```python
# param_decomp/targeted.py  (core — must be importable by masks.py / pgd_utils.py)
import contextlib
from collections.abc import Iterator
from contextvars import ContextVar

# None  -> normal random/adversarial delta mask (default everywhere)
# float -> delta mask pinned to this constant within the `with` scope
_DELTA_OVERRIDE: ContextVar[float | None] = ContextVar("delta_override", default=None)

def get_delta_override() -> float | None:
    return _DELTA_OVERRIDE.get()

@contextlib.contextmanager
def delta_override(value: float) -> Iterator[None]:
    """Pin the delta-component mask to `value` for all mask construction in this scope."""
    token = _DELTA_OVERRIDE.set(value)
    try:
        yield
    finally:
        _DELTA_OVERRIDE.reset(token)  # re-entrant + exception-safe, like torch.no_grad()
```

DP runs via `DistributedDataParallel` (one process per rank, single-threaded train loop), so a
process-global contextvar is safe — set only inside the trainer's nontarget block and the targeted
eval metrics, read only at the two sites below.

| Family | Site | Used on nontarget by | Action |
|---|---|---|---|
| Stochastic | `param_decomp/masks.py::calc_stochastic_component_mask_info` | `StochasticReconLoss{,Layerwise,Subset}`, the stochastic eval strategy | pin `torch.rand(leading_dims,…)` → `torch.full(leading_dims, override,…)` |
| PGD | `param_decomp/metrics/pgd_utils.py::_init_adv_sources` + `_construct_mask_infos_from_adv_sources` | `PGDReconLoss` (LM) | drop the optimized delta slot (`mask_c`) **and** pin the last-channel value |
| PPGD | `param_decomp/metrics/persistent_pgd_state.py::get_ppgd_mask_infos` (third delta-mask site, `expanded_adv_sources[k][..., -1]`) | excluded from the nontarget pass | **not** instrumented — but add a fail-fast `assert get_delta_override() is None` so the exclusion is enforced loudly, not implicitly |

**Losses excluded from the nontarget pass (decided):**
- `UnmaskedReconLoss`, both PPGD losses, and `StochasticHiddenActsReconLoss` are dropped from the
  nontarget loss set (the first is meaningless with a forced delta; PPGD is a stateful per-step
  adversary whose third delta-mask site the override deliberately does not touch — and is asserted
  unreachable there; hidden-acts recon is a target-only diagnostic).
- CIMasked recon losses build masks with `weight_deltas_and_masks=None`, so the override cannot reach
  them; no targeted config uses them on the nontarget pass — documented, not handled.

---

## Step-by-step

### Phase 0 — new modules
- **`param_decomp/targeted.py`** (core, ~18 lines): only the contextvar + `get_delta_override` +
  `delta_override` above. Nothing else core needs to import lives here.
- **`param_decomp_lab/targeted.py`** (lab): orchestration that may import lab + core freely.
  - `build_nontarget_loss_configs(loss_metrics: list[AnyLossMetricConfig], impmin_ratio: float)
    -> list[AnyLossMetricConfig]`: drop `UnmaskedReconLossConfig`, both PPGD configs, and
    `StochasticHiddenActsReconLossConfig`; scale `ImportanceMinimalityLossConfig.coeff` by
    `impmin_ratio` (guard `coeff is not None`). Returns a fresh list (pydantic `model_copy`).
  - `split_eval_metrics(metrics: list[Metric]) -> tuple[list[Metric], list[Metric]]`: partition
    instantiated eval metrics into `(target_metrics, nontarget_metrics)` by class (the three
    nontarget-data metrics go right), for `EvalLoop.metrics` / `EvalLoop.nontarget_metrics` (see Phase 5).

### Phase 1 — core hooks

**`param_decomp/masks.py`** — in `calc_stochastic_component_mask_info`, read the override for the
delta mask:
```python
from param_decomp.targeted import get_delta_override
...
override = get_delta_override()
weight_deltas_and_masks[layer] = (
    weight_deltas[layer],
    torch.full(leading_dims, override, device=device, dtype=dtype)
    if override is not None
    else torch.rand(leading_dims, device=device, dtype=dtype),
)
```

**`param_decomp/metrics/pgd_utils.py`** — two reads (fold into the existing `match weight_deltas` arm
in `_construct_mask_infos_from_adv_sources`):
```python
# _init_adv_sources: no optimized slot for a pinned delta
mask_c = module_c if (weight_deltas is None or get_delta_override() is not None) else module_c + 1
# _construct_mask_infos_from_adv_sources: pin instead of using the last source
override = get_delta_override()
case dict():
    if override is not None:
        weight_deltas_and_masks = {
            k: (weight_deltas[k], torch.full(batch_dims, override, device=device)) for k in weight_deltas
        }
        adv_sources_components = expanded_adv_sources           # all C channels are components now
    else:
        weight_deltas_and_masks = {k: (weight_deltas[k], expanded_adv_sources[k][..., -1]) for k in weight_deltas}
        adv_sources_components = {k: v[..., :-1] for k, v in expanded_adv_sources.items()}
```

**`param_decomp/metrics/persistent_pgd_state.py`** — in `get_ppgd_mask_infos`, add a single fail-fast
guard (PPGD is excluded from the nontarget pass, so the override must never reach this third
delta-mask site):
```python
from param_decomp.targeted import get_delta_override
...
assert get_delta_override() is None, "delta override must not reach the PPGD mask path"
```
`persistent_pgd_recon.py` itself stays unchanged.

**`param_decomp/optimize.py::Trainer.run`** — add the second pass. Signature:
```python
def run(self, train_loader, sink, cadence, eval_loop=None, *,
        nontarget_loader: DataLoader[Any] | None = None,
        nontarget_loss_configs: list[LossMetricConfig] | None = None) -> None:
```
- Assert `(nontarget_loader is None) == (nontarget_loss_configs is None)`.
- When set: `nt_pd = self.pd_config.model_copy(update={"loss_metrics": nontarget_loss_configs})`;
  `nontarget_metrics, _ = instantiate_metrics(nt_pd, self.component_model, device)`;
  `nontarget_iterator = loop_dataloader(nontarget_loader)`. (Not snapshotted — only `update()` return
  values are used; `compute()` is never called for these.)
- In the step loop, **after** `total_loss.backward()` and the `after_backward()` hooks:
  ```python
  if nontarget_metrics is not None:
      nt_weight_deltas = self.component_model.calc_weight_deltas()  # fresh: target backward freed the graph
      with bf16_autocast(enabled=runtime_config.autocast_bf16):
          nt_ctx = _build_metric_context(
              next(nontarget_iterator), step=step, is_eval=False, device=device,
              wrapped_model=self._wrapped_model, component_model=self.component_model,
              config=pd_config, reconstruction_loss=self.reconstruction_loss,
              weight_deltas=nt_weight_deltas,
          )
          with delta_override(1.0):
              nt_losses = {n: m.update(nt_ctx) for n, m in nontarget_metrics.items()}
      nt_total = torch.zeros((), device=device)
      for name, lv in nt_losses.items():
          if lv is None:
              continue
          cfg = cast(LossMetricConfig, nontarget_metrics[name].cfg)
          assert cfg.coeff is not None
          nt_total = nt_total + cfg.coeff * lv
          batch_log_data[f"nontarget/loss/{type(nontarget_metrics[name]).__name__}"] = lv.item()
      assert torch.isfinite(nt_total).all()
      nt_total.backward()                                           # accumulates into the same .grad
      batch_log_data["nontarget/loss/total"] = nt_total.item()
  ```
  Logged keys are prefixed `train/` by the existing logging block → `train/nontarget/loss/*`. Add
  per-layer `train/nontarget/l0/<layer>` from `nt_ctx.ci` if desired. The optimizer step at the bottom
  of the loop consumes the summed gradients — one step, two backwards.
- `_build_metric_context` sets `use_delta_component=pd_config.use_delta_component`; targeted runs
  require it `True` (validated lab-side), so `delta_override(1.0)` pins an *active* delta.

> **Why a sequential two-backward step (not one fused forward).** Each pass builds its graph, backs
> through it, and frees it before the next pass starts, so **only one activation graph is ever
> resident** — the lowest possible peak, and the reason each distribution can use the full memory
> budget independently (matters for LMs). Accumulated gradients live in parameter-sized `.grad`
> buffers, not batch-sized, so accumulation does not raise the peak. The two distributions carry
> genuinely different losses (importance-minimality coeff differs; `StochasticHiddenActsReconLoss` is
> target-only), which this scheme handles for free by running a different metric set per pass; it also
> keeps the scalar `delta_override` valid, since each forward runs in its own scope. A single fused
> *batch* (concatenating target + nontarget rows into one forward) is rejected: it would need a
> per-row delta mask (the scalar override cannot express "1.0 on nontarget rows, random on target
> rows") and distribution-aware slicing inside every loss, breaking metric isolation.
>
> **DDP.** Two backwards trigger two gradient all-reduces by default. The accumulated gradient is
> still correct (averaging an already-averaged tensor with equal counts is idempotent), so **v1 keeps
> both all-reduces** — two syncs per step, accepted as the simple, correct baseline.
>
> The single-all-reduce optimization (wrap the *target* backward in `self._wrapped_model.no_sync()` so
> only the nontarget backward synchronizes the summed gradient) is **deferred**, because it is unsafe
> with the current DDP construction. DDP is built at `optimize.py:329/333` with the **default**
> reducer (`find_unused_parameters=False`, no `static_graph`): the synchronizing backward must produce
> a gradient for *every* parameter that requires grad, or the reducer waits forever for buckets that
> never become ready. Under `no_sync()` the **nontarget** backward becomes the only synchronizing one,
> and its loss set is deliberately *restricted* (no faithfulness / unmasked-recon / PPGD / hidden-acts)
> — so any component parameter those dropped losses alone would have touched is now never marked ready
> on the syncing pass, deadlocking multi-rank runs (invisible in single-GPU testing). Before enabling
> `no_sync` later, either prove the restricted nontarget set covers the full parameter set, or
> construct DDP with `find_unused_parameters=True` for targeted runs. v1 takes neither risk.

**Eval driver — mirror nontarget loop.** Widen `EvalLoop` (frozen dataclass in `optimize.py`) with two
optional, default-off fields:
```python
nontarget_loader: DataLoader[Any] | None = None
nontarget_metrics: list[Metric[Any]] = field(default_factory=list)
# __post_init__: assert (nontarget_loader is None) == (not nontarget_metrics)
```
In the eval block of `Trainer.run`, after the existing target loop reduces `metrics`, run a structurally
identical second loop over `nontarget_loader` — reusing the same `eval_weight_deltas`, the same
`n_steps`, the same slow/active gating — wrapped in `with delta_override(1.0):` so the whole nontarget
eval pass sees the residual forced on (mirroring the *training* nontarget block exactly):
```python
if eval_loop.nontarget_loader is not None:
    nt_active = [m for m in eval_loop.nontarget_metrics if not (m.slow and not slow_step)]
    for m in nt_active:
        m.reset()
    with delta_override(1.0):
        for _ in range(eval_loop.n_steps):
            nt_ctx = _build_metric_context(next(nt_eval_iterator), step=step, is_eval=True, ...,
                                           weight_deltas=eval_weight_deltas)
            for m in nt_active:
                m.update(nt_ctx)
    nt_metrics = collect_metric_outputs(nt_active)
    sink.log({f"eval/{k}": v for k, v in nt_metrics.items()}, step=step)
```
The two loops are byte-for-byte the same shape; the only delta is the loader and the surrounding
`delta_override(1.0)`. `TargetReconLoss` (target distribution, delta off) stays in `metrics` and pins
its own `delta_override(0.0)` internally around its stochastic strategy — the target loop is the normal
mixed-metric pass and must **not** force a delta on its other metrics. Everything else is lab (Phase 5).

### Phase 2 — config (all lab; core `PDConfig` untouched)

**`param_decomp_lab/experiments/utils.py::ExperimentConfig[T, D]`** — add:
```python
nontarget_data: D | None = None
nontarget_batch_size: PositiveInt | None = None
nontarget_eval_batch_size: PositiveInt | None = None
nontarget_impmin_coeff_ratio: NonNegativeFloat = 1.0
```
After-validator, when `nontarget_data is not None`:
- require `nontarget_batch_size` and `nontarget_eval_batch_size`;
- require `pd.use_delta_component is True` (the mechanism forces the delta on);
- require **no** `FaithfulnessLossConfig` in `pd.loss_metrics` **and** `pd.faithfulness_warmup_steps == 0`
  (both drive the delta → 0; targeted needs the delta nonzero to carry nontarget behavior, and a
  warmup that zeroes the delta right before targeted training would have to immediately re-grow it —
  so warmup is disallowed, not just the ongoing loss);
- (toy) require target and nontarget describe the same input space (same `n_features`, enforced where
  both are known).

**Per-experiment data configs (lab):**
- `TMSDataConfig` / `ResidMLPDataConfig`: `active_indices: list[NonNegativeInt] | None = None`
  (upper bound `< n_features` asserted in the dataset).
- `LMDataConfig`: add `prompts_file: str | None = None`; relax `dataset_name: str | None = None`; add
  an after-validator requiring exactly one of `{dataset_name, prompts_file}` (safe — all current LM
  yamls set `dataset_name`). Target data uses `prompts_file`; nontarget data uses `dataset_name`.

Because `nontarget_data: D` reuses the same per-experiment data type, toy and LM share the field with
no per-domain split.

### Phase 3 — datasets (lab)

- `param_decomp_lab/experiments/tms/data.py::SparseFeatureDataset`: add `active_indices: list[int] | None
  = None` to `__init__`. When set, only those feature columns may be nonzero in a generated batch:
  - in `_generate_n_feature_active_batch`: draw each row's active set from `active_indices` only
    (instead of all `n_features`);
  - in `_masked_batch_generator`: after the probabilistic mask, zero every column **not** in
    `active_indices`;
  - assert `all(0 <= i < n_features for i in active_indices)` in `__init__`.
  - Leave `_generate_multi_feature_batch_no_zero_samples` unchanged (not used by targeted).
- `param_decomp_lab/experiments/resid_mlp/data.py::ResidMLPDataset`: add `active_indices` to `__init__`
  and forward via `super().__init__(...)`.
- `param_decomp_lab/experiments/lm/prompts_dataset.py` (new) — a static, file-backed LM target loader:
  - `load_prompts_dataset(prompts_file: str, tokenizer, max_seq_len: int) -> Tensor`: read one prompt
    per non-empty line; tokenize each; **pad** short sequences to `max_seq_len`; **raise** if any
    tokenized prompt exceeds `max_seq_len` (no silent truncation). Return a `[n_prompts, max_seq_len]`
    `input_ids` tensor.
  - `StaticBatchLoader`: an iterable over a fixed in-memory prompt pool that, each iteration,
    **randomly samples** rows from the pool to fill a batch of size `min(batch_size, n_prompts)`
    (sampled without replacement within a batch; `batch_size >= n_prompts` just yields the whole pool,
    optionally reshuffled). This gives the target stream real per-step variation when the pool exceeds
    one batch, rather than pinning the exact same rows every step. Seed the sampler for reproducibility.
  - `create_prompts_data_loader(cfg: LMDataConfig, *, batch_size, device, seed, ...) -> tuple[DataLoader,
    tokenizer]`: build the tokenizer (`AutoTokenizer.from_pretrained(cfg.tokenizer_name)`), tokenize
    `cfg.prompts_file` via `load_prompts_dataset`, wrap in `StaticBatchLoader` (with `batch_size` +
    `seed`), and return `(loader, tokenizer)` to mirror `create_lm_data_loader`'s return shape.

### Phase 4 — loader wiring (lab `run.py`, guarded by `cfg.nontarget_data is not None`)

**`tms/run.py` and `resid_mlp/run.py`:**
- `build_tms_loader` / `build_resid_mlp_loader`: pass `active_indices=data_cfg.active_indices` into the
  dataset (target loader).
- In `main`: when `cfg.nontarget_data is not None`, build a second loader from `cfg.nontarget_data`
  (full distribution; `active_indices=None`) at `cfg.nontarget_batch_size`, and call
  ```python
  trainer.run(
      train_loader, sink, cfg.cadence, eval_loop,
      nontarget_loader=nontarget_loader,
      nontarget_loss_configs=build_nontarget_loss_configs(cfg.pd.loss_metrics, cfg.nontarget_impmin_coeff_ratio),
  )
  ```

**`lm/run.py`:**
- `build_lm_loader`: if `data_cfg.prompts_file` is set, use `create_prompts_data_loader(...)`; else the
  existing `create_lm_data_loader(...)` (unchanged default path).
- `_fresh_main`: build the nontarget loader from `cfg.nontarget_data` via `create_lm_data_loader` at
  `cfg.nontarget_batch_size`; pass `nontarget_loader` + `nontarget_loss_configs` to `trainer.run`.
- `_resume_main`: thread the same nontarget loader/configs (the resume path reconstructs the loader
  identically — see checklist).

> Keep each block ~10 guarded lines. A shared `param_decomp_lab/targeted.py::build_nontarget_loaders`
> can absorb the common shape later, but per-experiment dataset construction differs enough that
> inlining is clearer now.

### Phase 5 — new evals (lab)

All new eval metrics are `Metric` subclasses in `param_decomp_lab/eval_metrics/`, registered in that
package's `__init__.py` (add to `AnyEvalMetricConfig` union **and** `EVAL_METRIC_CLASSES`), each with a
`short_name` class var (auto-collected for wandb).

**Generic:**
- `weight_magnitude.py` — `WeightMagnitude`: accumulate CI in `update`; in `compute` plot, per
  `LinearComponents` layer, the product of singular-vector norms `‖V‖·‖U‖` (component weight
  magnitude). Add `plot_weight_magnitude` + grid helpers `_parse_layer_grid`,
  `_setup_layer_grid_labels` to `param_decomp_lab/eval_metrics/plotting.py` (`_render_figure` already
  exists there). Pure `ctx` consumer — constructed from config alone.

**tPD-specific.** Recon strategies build mask infos with `make_mask_infos` / `ComponentsMaskInfo`. Each
metric evaluates reconstruction under four masking strategies, where the **component mask** `m_c` and
**delta mask** `m_δ` per layer are:

| Strategy | component mask `m_c` | delta mask `m_δ` |
|---|---|---|
| `stochastic` | sampled as in `StochasticReconLoss`, averaged over `n_mask_samples` | inherited from the surrounding `delta_override(v)` scope (= `v`) |
| `CImasked` | `ci.lower_leaky` (CI values used directly as the gate) | `torch.full(..., v)` |
| `rounded` | `(ci.lower_leaky > rounding_threshold).float()` (hard 0/1 gate) | `torch.full(..., v)` |
| `delta_only` | all-zeros (components off) | `torch.full(..., 1.0)` |

`v` is the delta value for the data distribution: `0.0` for target recon (components must do the work),
`1.0` for nontarget recon (the residual carries the output). For the **nontarget** metrics `v = 1.0`
comes for free from the driver's `delta_override(1.0)` scope around the whole nontarget eval loop — the
`stochastic` strategy just calls `calc_stochastic_component_mask_info` and inherits the pinned delta,
exactly as a normal stochastic recon metric would. The **target** `TargetReconLoss` sets
`delta_override(0.0)` itself around its `stochastic` strategy (its loop has no surrounding scope).
`delta_only` is a fixed sanity strategy (delta alone should reproduce the target output) regardless of
distribution. Recon error is computed with `ctx.reconstruction_loss` (the `(pred, target) -> (sum, n)`
protocol on the context).

All three nontarget-data metrics are **standard accumulate-in-`update` / reduce-in-`compute` metrics**
(no self-owned iterator, no I/O in `compute`); the driver's mirror nontarget loop feeds them `ctx` per
batch under `delta_override(1.0)`. They go in `EvalLoop.nontarget_metrics`. `TargetReconLoss` and
`WeightMagnitude` consume the normal target `ctx` and stay in `EvalLoop.metrics`.

- `targeted_recon_loss.py` — `TargetReconLoss`, `NontargetReconLoss`, modelled on the multi-strategy
  `CEandKLLosses`/recon-eval template: `reset` allocates per-strategy scalar accumulators; `update(ctx)`
  builds the four strategy masks off `ctx.ci`, runs masked forwards on `self.model`, scores each with
  `ctx.reconstruction_loss`, and accumulates (detached); `compute` all-reduces and emits the four
  strategy losses + `total_l0` (summed CI L0). `TargetReconLoss` consumes the target `ctx` with
  `v = 0.0` (wrapping its `stochastic` strategy in `delta_override(0.0)`); `NontargetReconLoss` consumes
  the nontarget `ctx` with `v = 1.0` inherited from the driver scope.
- `nontarget_ci_mean_per_component.py` (new sibling; `ci_mean_per_component.py` untouched): a near-copy
  of `CIMeanPerComponent` — `update(ctx)` sums `ctx.ci.lower_leaky` per component and counts examples;
  `compute` all-reduces and emits `nontarget_ci_mean_per_component[_log]` via the existing
  `plot_mean_component_cis_both_scales`. The only difference from its sibling is which loop feeds it.
- `targeted_ci_heatmap.py` — `TargetedCIHeatmap` + `plot_targeted_ci_heatmaps`: one figure, a target row
  over a nontarget row of per-component CI heatmaps. The **nontarget row** is accumulated in
  `update(ctx)` (the nontarget loop), mirroring `CIMeanPerComponent`. The **target row** follows the
  `IdentityCIError`/`UVPlots` precedent — `update` caches only metadata; `compute` synthesizes the
  target probes (one-hots over `active_indices` for toy, the tokenized prompts for LM, carried on the
  metric config) and runs a `cache_type="input"` forward + `calc_causal_importances` on `self.model`,
  then renders both rows. Uses `CIOutputs` + `move_batch_to_device`.

**Nontarget metrics → `EvalLoop.nontarget_metrics` (no injected iterators).** `_build_eval_loop` (each
lab `run.py`) partitions the configured eval metrics by which distribution they consume — the three
nontarget-data metrics (`NontargetReconLoss`, `NontargetCIMeanPerComponent`, `TargetedCIHeatmap`) go
into `nontarget_metrics`, everything else (including `TargetReconLoss` / `WeightMagnitude`) into
`metrics` — and builds a nontarget eval loader from `cfg.nontarget_data` at
`cfg.nontarget_eval_batch_size` to pass as `EvalLoop.nontarget_loader`. The partition is a small lab
helper (`param_decomp_lab/targeted.py::split_eval_metrics`, keyed off the metric class). The driver's
mirror loop (above) does the rest; no metric receives an iterator. `TargetedCIHeatmap` additionally
receives its static target-probe spec (`active_indices` / tokenized prompts) on its config.

Config knobs: recon configs carry `rounding_threshold`. The nontarget batch count is the shared
`EvalLoop.n_steps` (no per-metric `n_nontarget_batches`); the nontarget eval batch size is
`cfg.nontarget_eval_batch_size`.

### Phase 6 — example config + smoke test
Add `param_decomp_lab/experiments/tms/<name>_targeted.yaml`: a copy of the tms_5-2 PD config with
`data.active_indices` set, a `nontarget_data` (full distribution), `nontarget_batch_size` /
`nontarget_eval_batch_size`, `nontarget_impmin_coeff_ratio`, and the new eval metrics enabled. Smoke:
run the TMS entry point for a handful of steps; confirm `train/nontarget/*` logs appear, nontarget CI
L0 drops, and the new figures render.

---

## Tests

Provide a small CI-pattern test helper (`param_decomp_lab/tests/_targeted_ci_solutions.py` or
`param_decomp/utils/`):
- `IdentityCIPattern` — exactly one distinct active component per input;
- `DenseCIPattern(k)` — at most `k` components active;
- `TargetCISolution.distance_from(ci, tolerance)` — distance of an observed CI pattern from a target
  pattern.
"Active" means `ci.lower_leaky > ci_alive_threshold`. Core-only tests (contextvar + masks/pgd override)
go in `param_decomp/tests/test_targeted.py`; dataset/config/eval-wiring/convergence tests go under
`param_decomp_lab/tests/` (next to `test_tms.py`).

**A. Unit (fast, no training):**
1. `delta_override` — None default; `v` inside scope; resets on exit/exception; nests.
2. `calc_stochastic_component_mask_info` — delta mask constant under override, random `[0,1)` without;
   component masks unaffected; **no-op when override None**.
3. PGD — `mask_c == module_c` under override (else `+1`); delta pinned in construction; a PGD recon
   forward inside `delta_override(1.0)` runs (shape-consistent, all-C-component path). PPGD — calling
   `get_ppgd_mask_infos` inside `delta_override(1.0)` **raises** (the third-site guard fires).
4. `build_nontarget_loss_configs` — drops Unmasked/PPGD/HiddenActs; scales impmin; keeps stochastic recon.
5. Validators (lab `ExperimentConfig`) — missing batch sizes / `use_delta_component=False` /
   FaithfulnessLoss-in-targeted / `faithfulness_warmup_steps > 0`-in-targeted raise; `LMDataConfig`
   both-or-neither of `dataset_name`/`prompts_file` raises, exactly-one passes.
6. `active_indices` — only listed columns nonzero; out-of-range raises; None unchanged.
7. `prompts_dataset` — pads to max_seq_len, raises over-length; `StaticBatchLoader` yields
   `min(batch_size, n_prompts)` rows drawn from the pool, varies across iterations when the pool
   exceeds one batch, is reproducible under a fixed seed, and yields the whole pool when
   `batch_size >= n_prompts`.
8. Faithfulness invariant — component masks = 1 + delta mask = 1 ⇒ exact target output (guards the
   delta path through `make_mask_infos`).
9. Eval wiring — `split_eval_metrics` partitions the 3 nontarget metrics into `nontarget_metrics`;
   `EvalLoop.__post_init__` asserts the `nontarget_loader`/`nontarget_metrics` both-or-neither
   invariant; driving the nontarget loop emits the expected keys (`NontargetReconLoss` → 4 strategies +
   `total_l0`) and `delta_override` is back to `None` after the loop (no leak).

**B. Convergence (slow, `@pytest.mark.slow`):** load a pretrained target, `active_indices` = 3 seeded
random features (the only features active in target inputs), nontarget = same task full distribution;
build the trainer + nontarget loader; `run` (seeded, min steps that converge); compute CI on target
one-hots and a nontarget batch; assert:
- **TMS_40-10-id**: target — `linear1`,`linear2` = `IdentityCIPattern`, hidden = `DenseCIPattern(k=5)`;
  nontarget — `linear1`,`linear2` zero active.
- **resid_mlp1**: target — `mlp_in` one/input, `mlp_out` small `DenseCIPattern`; nontarget — `mlp_in`
  zero active.

Use `TargetCISolution.distance_from(..., tolerance)`; nontarget checks `(ci > thr).sum() == 0`. Seed
everything, keep steps low; skip-if-target-unavailable (or use a tiny in-fixture target to avoid wandb
in CI).

**C. Other:** isolation/regression (`nontarget_data=None` ⇒ no `train/nontarget/*` keys; existing
non-targeted tests unchanged); `get_delta_override() is None` after a nontarget step (no contextvar
leak); nontarget L0 ≈ 0 at end of the convergence run; fast `steps≈3` targeted wiring smoke (emits
`train/nontarget/loss/total` + new evals, no NaNs).

---

## Files touched (merge surface)

**New:**
- `param_decomp/targeted.py` (core, contextvar only)
- `param_decomp_lab/targeted.py` (lab orchestration)
- `param_decomp_lab/experiments/lm/prompts_dataset.py`
- `param_decomp_lab/eval_metrics/{weight_magnitude,targeted_recon_loss,nontarget_ci_mean_per_component,targeted_ci_heatmap}.py`
- example targeted yaml; `param_decomp/tests/test_targeted.py` + `param_decomp_lab/tests/test_targeted*.py`
  (+ CI-pattern test helper)

**Edited — core (tiny, default-off):**
- `param_decomp/masks.py` — 1 override read
- `param_decomp/metrics/pgd_utils.py` — 2 override reads
- `param_decomp/metrics/persistent_pgd_state.py` — 1 fail-fast `assert get_delta_override() is None`
- `param_decomp/optimize.py` — 2 keyword loader/config params on `Trainer.run` + 1 guarded nontarget
  train pass; `EvalLoop` gains `nontarget_loader` + `nontarget_metrics` (default-off) + the mirror
  nontarget eval loop in the eval driver

**Edited — lab:**
- `param_decomp_lab/experiments/utils.py` — `nontarget_*` fields + targeted validators on `ExperimentConfig`
- `param_decomp_lab/experiments/{tms,resid_mlp}/data.py` — `active_indices`
- `param_decomp_lab/experiments/lm/data.py` — `prompts_file`, relaxed `dataset_name`, validator
- `param_decomp_lab/experiments/{tms,resid_mlp,lm}/run.py` — guarded nontarget train loader; `_build_eval_loop` builds the nontarget eval loader + partitions metrics into `EvalLoop.nontarget_metrics`
- `param_decomp_lab/eval_metrics/__init__.py` — 5 union entries + 5 `EVAL_METRIC_CLASSES` entries
- `param_decomp_lab/eval_metrics/plotting.py` — `plot_weight_magnitude` (+helpers), `plot_targeted_ci_heatmaps`

**Untouched (isolation win):** core `PDConfig` and `param_decomp/configs.py`, `metrics/context.py`,
`metrics/dispatch.py`, `metrics/persistent_pgd_recon.py` (its state helper gets a one-line assert),
every recon-loss `metrics/*.py`
body, `param_decomp_lab/eval_metrics/ci_mean_per_component.py`, `infra/wandb.py` (short names
auto-collected via `short_name`). (`EvalLoop` is now lightly widened — see Edited/core — a deliberate
trade of two touched core symbols for eval metrics that mirror the existing ones exactly.)

## Checklist
- [ ] `_build_metric_context` already calls `move_batch_to_device` for all nontarget batches (toy + LM).
- [ ] Nontarget pass recomputes `weight_deltas` (target backward freed the graph).
- [ ] DDP: v1 lets **both** backwards all-reduce (correct, two syncs/step). Do **not** add
      `no_sync()` — DDP is built with `find_unused_parameters=False`, and the restricted nontarget
      loss set may not mark every parameter ready on the lone syncing backward → deadlock. Single
      all-reduce is a deferred optimization, gated on `find_unused_parameters=True` or a proof of full
      parameter coverage.
- [ ] `active_indices` honored in the per-n-active + masked generators only (not the no-zero generator);
      `0 <= i < n_features` assert.
- [ ] `TargetedCIHeatmap` LM path needs `tokenizer_name` — present on `LMDataConfig`.
- [ ] After `dataset_name` becomes optional, ensure `build_lm_loader` takes the `prompts_file` branch
      when it's set, and `_resume_main` rebuilds the nontarget loader identically.
- [ ] Nontarget loss metrics are not snapshotted (only `update()` return is used) — confirm resume is
      unaffected.
- [ ] `make check` (basedpyright + ruff) passes.
- [ ] Add a "Targeted Decomposition" section to the relevant `CLAUDE.md` pointing at
      `param_decomp/targeted.py` + `param_decomp_lab/targeted.py`.
