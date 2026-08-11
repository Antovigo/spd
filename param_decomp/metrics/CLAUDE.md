# `param_decomp/metrics/`

Loss `Metric` classes plus the dispatch wiring that turns a `PDConfig.loss_metrics` YAML
entry into a bound, runnable `Metric` instance.

Loss metrics are **canonical and curated** — adding one is a deliberate change to the
core library. For eval metrics (user-extensible, lab-side), see
[`param_decomp_lab/eval_metrics/CLAUDE.md`](../../param_decomp_lab/eval_metrics/CLAUDE.md).

## File map

| File | Purpose |
|---|---|
| `base.py` | `Metric` ABC (lifecycle: `__init__(cfg)` → `bind` → `update` → `compute`) + `NamedMetricConfig` / `LossMetricConfig` bases + `instance_key` / `key_prefix` + `before_backward` / `after_backward` hooks |
| `context.py` | `MetricContext` — the per-step bundle every `Metric.update(ctx)` receives |
| `dispatch.py` | `LOSS_METRIC_CLASSES` type→class table + `instantiate_metrics(...)` |
| `<loss_name>.py` | One file per metric: `<Name>Loss` class + `<Name>LossConfig` config side-by-side |
| `persistent_pgd_state.py` | PPGD adversarial-source state machine (shared by `persistent_pgd_recon.py`) |
| `pgd_utils.py` | Shared PGD helpers; `pgd_masked_objective_update` runs PGD against any mask-consuming objective (output recon and per-site activation error are two such) |
| `hidden_acts.py` | Shared per-site (hidden-activation) relative-error machinery: clean targets, squared-error accumulation, DDP reduction, site filtering, and the `SiteInputs` chained/local dispatch |
| `output.py` | Shared output-extraction helpers used across recon losses |

## Adding a loss metric

1. Define `<Name>Loss(Metric[<Name>LossConfig])` and its `<Name>LossConfig(LossMetricConfig)`
   in `<name>.py`. The config must carry a unique `type: Literal["<Name>Loss"]` discriminator.
2. Append the config to `AnyLossMetricConfig` in `param_decomp/configs.py`.
3. Append the class to `LOSS_METRIC_CLASSES` in `dispatch.py`.

The pydantic discriminated union validates `pd.loss_metrics` YAML entries without any
custom validator. `instantiate_loss_metrics()` builds and `bind()`s one instance per
entry. Duplicate `type` literals in a single config are rejected.

A metric that wants to manipulate state coupled to backward overrides `before_backward`
and/or `after_backward` (see PPGD for the canonical example).

## Importance-minimality variants

Two interchangeable CI-sparsity penalties, same `sum + beta·entropy` shape:

- `ImportanceMinimalityLoss` (`importance_minimality.py`) — `L_p`: `(c + eps)^p`, `p`
  annealed toward 0. Gradient `p·c^(p-1)` blows up as `c→0` for `p<1`.
- `SmoothL0ImportanceMinimalityLoss` (`smooth_l0_importance_minimality.py`) — bounded
  Geman–McClure `c²/(c²+γ²)`: flat at 0, saturating to 1, bounded gradient `~0.65/γ` near
  `c≈γ`. Drop-in alternative avoiding the `L_p` cliff; `γ` anneals like `p`. Self-contained.

Both also carry the same **coefficient schedule** (`coeff_warmup_frac`,
`coeff_peak_multiplier`, `coeff_anneal_start_frac`, `coeff_anneal_end_frac`): ramp
`0 → peak` over the warmup, hold, then ramp `peak → 1.0` across the anneal window.
Defaults are a no-op. It scales the **live training loss only** — `compute()`'s logged
sparsity proxy is unscaled, so train and eval log keys differ by the multiplier while the
schedule is above 1.0. `build_nontarget_loss_configs` copies the whole config, so a
targeted run's nontarget impmin inherits the same schedule on top of its `impmin_ratio`.

Both are registered eval-side (`EVAL_METRIC_CLASSES` + `AnyEvalMetricConfig`), so a run driven
by one can log the other as an eval-only sparsity proxy. The directly comparable cross-run
sparsity number is `CI_L0`; each penalty's own `_no_beta` proxy is on its own scale.

## Metric identity (`instance_key`) and same-class loss + eval

Metric instances are keyed everywhere — instance dicts, state-dict, and log-key
suffixes — by `Metric.instance_key`, which defaults to the class name. Any
`NamedMetricConfig` can override it by setting `name` (`LossMetricConfig` extends it, so
loss configs have it too). This is what lets the
*same* metric class appear under both `pd.loss_metrics` and `eval.metrics`: without a
distinct `name` their instance keys collide and `instantiate_metrics` rejects the
overlap. Example — a 1-step PGD training loss plus a 20-step PGD eval probe:

```yaml
pd:
  loss_metrics:
    - type: PGDReconLoss        # instance_key "PGDReconLoss", auto-evaluated too
      coeff: 0.5
      n_steps: 1
eval:
  metrics:
    - type: PGDReconLoss        # distinct instance_key -> no collision
      name: PGDReconLoss_20step
      n_steps: 20
```

`name` disambiguates scalar-output metrics (the log key is `{log_namespace}/{instance_key}`).
A dict-returning metric flattens under its own internal keys, so two instances of one class
would still collide; they prefix their keys with `Metric.key_prefix`, which is `"<name>/"`
when `name` is set and `""` otherwise — so a single-instance run's log keys are unchanged.
`NAlive`, `CI_L0`, `CIMeanPerComponent` and `CIHistograms` do this, which is what lets a
dual-CI run log one of each per CI net.

## Config placement rule

The default home for a config is `param_decomp/configs.py`. Move a config next to its
implementation only when leaving it in `configs.py` would close an import cycle —
concretely, when the implementation module `M` is also (transitively) imported by
`configs.py` (usually via the metric union). Then `M → configs` would loop; put the
config in `M` and update callers to import it from `M` directly.

Configs currently kept next to their implementation for this reason:

- `ScheduleConfig` → `param_decomp.schedule`
- `DecompositionTargetConfig` → `param_decomp.decomposition_targets`
- `CiConfig` family (`LayerwiseCiConfig`, `AttnConfig`, `GlobalSharedTransformerCiConfig`,
  `GlobalCiConfig`) → `param_decomp.ci_fns`
- `SamplingType`, `SubsetRoutingType` + members → `param_decomp.masks`
- Each loss metric's `LossMetricConfig` subclass → `param_decomp/metrics/<name>.py`

Never use `if TYPE_CHECKING:` + forward-reference strings to paper over a cycle. If
you're reaching for that, the config placement is wrong; move the config instead.

## Sources vs masks (PGD terminology)

These two concepts both show up in the PGD metrics and are easy to confuse:

- **Sources** (`adv_sources`, `PPGDSources`, `self.sources`) — the raw values PGD
  optimizes adversarially. They get interpolated with CI to produce component masks:
  `mask = ci + (1 - ci) * source`. Used in `pgd_utils.py` (regular PGD) and
  `persistent_pgd_state.py` (PPGD).
- **Masks** (`component_masks`, `RoutingMasks`, `make_mask_infos`, `n_mask_samples`) —
  the materialized per-component masks consumed by forward passes. Produced from
  sources (in PGD) or from stochastic sampling (otherwise). This is the general PD
  concept — sources are a PGD-internal stepping stone.

## PPGD note

PPGD's state machine lives in `persistent_pgd_state.py` (shared); its `Metric`
classes + configs live in `persistent_pgd_recon.py`. The split is so the subset
variant (`PersistentPGDReconSubsetLoss`) can reuse the same state machine.

`PersistentPGDState` does not know what it is attacking: `warmup` and `compute_sum_and_n`
take a `PGDObjective` per call, closing over the live batch, so the state owns only the
sources and their optimizer. `_PersistentPGDReconBase._objective` is the single overridable
seam — the base attacks output reconstruction, `PersistentPGDHiddenActsReconLoss` attacks the
relative per-site activation error through the truncated `site_outputs` forward.

**Each metric instance owns its own adversary.** The state is built lazily per instance and
the trainer snapshots metric state keyed by `instance_key`, so a dual-CI run listing both a
`PersistentPGDReconLoss` and a `PersistentPGDHiddenActsReconLoss` gets two independent
source tensors, two optimizer states, separately checkpointed and resumed. PPGD configs
carry `ci_role`, so each adversary attacks the CI net that owns its objective — without
that, the output net would face a persistent adversary while the hidden net faced only
stochastic masks, which would confound any comparison of their densities.

All PPGD losses are excluded from the nontarget pass: with the delta forced fully on, the
adversary's objective is degenerate.

## Hidden-acts recon: fused aux vs standalone

`StochasticReconSubsetLossConfig.hidden_acts_recon` (`HiddenActsReconAux`, in
`stochastic_recon_subset.py`) adds `coeff * MSE(masked site output, frozen x@W + b)`
riding the host's stochastically-masked forwards — the frozen targets are recomputed
from the step's cached clean input acts, so it costs **no extra forward** (unlike the
standalone `StochasticHiddenActsReconLoss`, which runs its own clean + masked passes
per step and survives only for older configs). The aux folds into the host's returned
loss at the `aux.coeff / host.coeff` ratio (the trainer multiplies by the host coeff);
eval logs it separately as `loss/StochasticReconSubsetLoss/hidden_acts`.

## Dual CI networks (`CIRole`)

`pd.dual_hidden_ci` builds a **second** CI fn on `ComponentModel` (`ci_fn_hidden`) of the
same architecture as `ci_config`. Both nets score the *same* pool of subcomponents; they
differ only in the reconstruction loss that trains them:

- `"output"` — importance for reconstructing the target model's final output. The existing
  net; every metric defaults to this role, so all pre-existing configs are unchanged.
- `"hidden"` — importance for reconstructing the decomposed sites' activations.

`CIRole` lives in `param_decomp/ci_fns.py`. `MetricContext` carries `ci` and
`ci_hidden: CIOutputs | None`; metrics that can read either take a `ci_role` config field
and go through `ctx.ci_for(role)`, which asserts the net exists. Both nets' parameters sit
in the single `ci_fn_optimizer` — Adam is per-parameter and the nets are disjoint, so a
shared optimizer is mathematically identical to two with the same hyperparameters.

### Ordering the two nets: `hidden_ci_floor` and `HiddenCIShortfallLoss`

A subcomponent reaches the model's output only through the output of the matrix it lives in
— exactly what the hidden objective measures — so output-important should imply
hidden-important, while the converse rightly fails for a component cancelled downstream. Two
ways to act on that, one structural and one as a penalty:

| | `pd.hidden_ci_floor` | `HiddenCIShortfallLoss` |
|---|---|---|
| where | `ComponentModel.calc_causal_importances` | a loss metric (also registered eval-side) |
| effect | `CI_hidden >= CI_output` by construction | penalises `relu(CI_out - CI_hidden)` |
| diagnostic left | none — the shortfall is ~0 by design | the shortfall itself, still logged |

The floor is a smooth `max` **in logit space** (`floor_hidden_ci_logits`, `ci_fns.py`):
`z_hidden = z_out.detach() + softplus(z_hidden_raw - z_out.detach(), beta=sharpness)`. Logit
space is the right place because both squashes are monotone, so one ordering there is
inherited by the mask *and* the minimality penalty at once. The smooth max rather than a hard
one because a hard max zeroes the gradient wherever the floor binds; rather than a plain
`z_out + softplus(h)` because both readout heads init to logit 0.5, which that form would turn
into a hidden CI pinned at 1 with no mask gradient before step 1. Its limits are recorded in
`floor_hidden_ci_logits`' docstring and in `test_hidden_ci_floor.py`: the escape gradient
decays as `exp(beta*gap)` and is effectively gone past `gap ≈ -2`.

`z_out` is detached in both mechanisms. Otherwise the hidden objective's own sparsity pressure
would satisfy the constraint by pushing the *output* net's logits down — the hidden impmin
acting on the output CI, confounding the comparison the dual setup exists to make.

The floor lives on the model, not in the trainer, so `role="hidden"` returns the floored value
to every caller — eval metrics, `find_alive_subcomponents`, harvest, the app. It costs one
extra output-net forward per hidden CI call, run under `no_grad`; on the L18 shape that is
~0.25% of one target-model forward.

`HiddenCIShortfallLoss` is normalised like the importance-minimality losses (sum over
subcomponents, mean over positions, summed over sites) rather than as a plain mean, so its
coefficient is directly comparable to impmin's. That is not cosmetic: violations occupy well
under 1% of entries on the `addsub-L18-11` runs, so a mean over all entries would divide the
signal by the ~6000 non-violating subcomponents and leave a per-entry gradient orders of
magnitude below the sparsity pressure it has to overcome.

Two knock-on rules:

- `ci_scaled_component_weight_decay` takes the **max over both nets'** batch CI max. A
  subcomponent alive only in the hidden net is carrying interference, i.e. doing real work;
  decaying it would fight the hidden-acts loss.
- Anything keyed by metric *class name* breaks once one class has two instances. Loss log
  keys, nontarget log keys, and the hidden-acts result dicts are all keyed by
  `Metric.instance_key` instead (see `NamedMetricConfig` in `base.py`).

## Hidden-activation reconstruction

Three metrics measure the same quantity — the **relative** per-site error
`Σ(out − tgt)² / Σ tgt²`, averaged over sites — under three different masks:

| metric | mask | where |
|---|---|---|
| `StochasticHiddenReconSubsetLoss` | stochastic subset ablation | core, training loss |
| `CIHiddenActsReconLoss` | CI itself | lab, eval only |
| `PGDHiddenActsReconLoss` | adversarial (`n_steps` of sign-PGD) | core, eval only |

Relative rather than raw MSE so that sites with very different activation scales (an MLP
`down_proj` against an attention `q_proj`) weigh equally and the coefficient transfers
across blocks. Numerator and denominator are accumulated and DDP-reduced separately — the
result is a ratio of sums, never a mean of per-batch or per-rank ratios.

Targets are the frozen model's own site outputs, recomputed as `F.linear(x_clean, W, b)`
from the clean input activations the step already cached: **no extra forward pass**.

`site_patterns` (fnmatch, e.g. `["*.mlp.down_proj", "*.self_attn.o_proj"]`) restricts which
sites the error is *measured* at; masking always covers every decomposed site.

### `routing` — how far ablation damage travels

`StochasticHiddenReconSubsetLoss.routing` selects a `Router` (`masks.py`). The choice decides
how much *compounding* the chained formulation actually sees, which is easy to miss:

- `uniform_k_subset` / `static_probability` — each position routes to a subset of the
  matrices, the rest running frozen. A site is scored only where it is itself routed, and at
  those positions each upstream site is routed only `E[k]/n_modules` of the time (4/7 on the
  seven-site L18 shape). So roughly 43% of the upstream chain is frozen wherever the error is
  read, and a downstream matrix mostly sees clean inputs.
- `{type: all}` — every position routes to every matrix. Nothing runs frozen weights, so a
  downstream site inherits the full damage of everything above it. This is already what
  `PersistentPGDHiddenActsReconLoss` does (`_router_for_cfg`'s `case _` is `AllLayersRouter`),
  so before this option existed the adversarial and stochastic halves of the hidden objective
  disagreed about the routing regime.

Switching a hidden loss from a subsetting router to `all` raises the measured error — most at
the sites furthest downstream — and scores every position instead of the routed subset, so its
numbers are **not** comparable across the change. It costs nothing extra: the forward pass ran
in full either way.

### `site_inputs` — chained vs local

An orthogonal axis, on all four of the above plus `PersistentPGDHiddenActsReconLoss`:

| `site_inputs` | each site's components run on | its error is |
|---|---|---|
| `"masked_forward"` (default) | the input the masked forward produced | its own error **+** what it inherited from upstream sites |
| `"clean"` | the input the frozen model gave it | its own error alone |

Both compare against the same frozen targets, so the two are subtractable:
`masked_forward − clean` is the inherited (compounding) part, per site. Running one instance
of each — distinguished by `name`, as everywhere else — is how that gets logged.

`"masked_forward"` goes through `ComponentModel.site_outputs`, which aborts the forward once
every hooked site has been cached (a private sentinel exception, since a forward hook cannot
ask PyTorch to stop). Everything past the last decomposition target would otherwise be
wasted compute *and* retained-for-backward activations.

`"clean"` needs **no forward pass at all**: with every site handed its own cached input the
sites are independent, so it is matmuls per site over tensors `_build_metric_context` already
produced. It also retains nothing through the frozen model for backward, and computes only
the measured sites.

Measured on the `addsub-L18-11` shape (7 sites on one Llama-3.1-8B block, 2048 tokens),
MACs/token summed over the sites:

| | MACs/token | mask-dependent |
|---|---:|---|
| `V^T x` | 35.6 M | no |
| `(acts · mask) @ U` | 41.4 M | yes |
| `x @ delta.T` (only with `use_delta_component`) | 218.1 M | no |
| **`"clean"` total** | **295.1 M** | |
| `"masked_forward"` (19 blocks + the component work) | 4439 M | |

So **~15x cheaper per mask sample**, not the order of magnitude a naive "no forward pass"
reading suggests: with the delta component on, `x @ delta.T` is a dense `d_in x d_out` matmul —
as expensive as running the frozen matrix — and dominates the local path. Two optimisations
are available and not yet taken: the delta term is derivable from the target already in hand
(`x @ delta.T == (target - bias) - acts @ U`, exact and gradient-exact since `W` is frozen),
and everything except `(acts · mask) @ U` is mask-independent and so hoistable out of the
`n_mask_samples` and PGD loops. Together they would give ~37x on the first sample and ~107x
per extra one. Worth doing before `n_mask_samples` is raised or the adversary moves into
training, and not before: at `n_mask_samples: 1` the hoist buys nothing.

Two restrictions come with `"clean"`, both asserted at bind:

- **It cannot measure readout sites** (`resolve_measured_sites`): a readout target is a point
  in the residual stream, so feeding every matrix its clean input leaves it unchanged and its
  error identically zero. Give the `"clean"` instance a `site_patterns` covering the
  decomposed sites and read the readouts from a separate `"masked_forward"` instance.
- **A PGD or PPGD instance must measure every decomposed site**
  (`assert_sources_reach_every_site`). Those metrics allocate one adversarial source per
  decomposed site and differentiate w.r.t. all of them at once (`allow_unused=False`).
  Chained, a source at an unmeasured site still reaches the loss by perturbing the sites
  downstream of it; locally the sites are independent, so it reaches nothing and
  `torch.autograd.grad` raises mid-step. The stochastic and CI-masked probes never
  differentiate w.r.t. sources and so may narrow freely.

One semantic shift worth knowing: under `"clean"` a subset router no longer routes anything —
with no chain it only decides which positions `site_squared_errors` scores. A `"clean"`
instance on `uniform_k_subset` therefore scores the same position count as its chained
counterpart (which is what keeps the two comparable) but gains nothing from the subsetting.

### Readout sites — measuring off the decomposed matrices

`pd.hidden_readout_sites` (`{measurement_name: module_path}`) adds measurement points that
are not a decomposed matrix's output. The named module's **input** is captured, clean and
masked, and joins the decomposed sites in `ComponentModel.measurement_sites` — the set
`site_patterns` selects from. Nothing else changes: the same targets / squared-error /
DDP-reduction path serves both kinds.

The residual stream is the motivating case. In a Llama block, hook
`post_attention_layernorm` for the post-attention stream and the *next* block's
`input_layernorm` for the post-MLP one:

```yaml
pd:
  hidden_readout_sites:
    resid_post_attn: model.layers.18.post_attention_layernorm
    resid_post_mlp: model.layers.19.input_layernorm
  loss_metrics:
    - type: StochasticHiddenReconSubsetLoss
      site_patterns: ["resid_*"]
```

Three things follow from a readout's target being the stream rather than a write:

- Its denominator `Σ tgt²` is dominated by the frozen incoming stream, so the same relative
  error is numerically far smaller than at a matrix output — which reprices that site
  against the shared importance-minimality coefficient.
- It is measured at **every position**, having no routing mask of its own. This is
  required, not a shortcut: attention mixes positions, so a position routed to no
  component still receives error from the routed positions it attends to.
- Readout sites are cached on the clean `cache_type="input"` pass, so they land in
  `pre_weight_acts` alongside the pre-weight activations. The CI fn is defined over the
  decomposition targets only, so `calc_causal_importances` selects those entries rather
  than consuming the whole cache — the global CI-fn wrapper transforms every key it is
  handed and would otherwise `KeyError` on a readout name.

Cadence: the CI-masked and stochastic probes cost one truncated forward per eval batch and
run on the **fast** cadence (`eval.every`). `PGDHiddenActsReconLoss` costs `n_steps + 1`
truncated forwards *per eval batch* — note the site count does **not** multiply this, since
one forward captures every site — so its price is set by `n_steps`, and `slow` is therefore a
config decision rather than a class one: mix in `EvalCadenceConfig` and set `slow: true` for a
high-`n_steps` instance. The useful pattern is a cheap frequent probe (`n_steps: 5`) to see
early whether the adversary finds materially more error than sampled masks, plus an occasional
20-step instance under a distinct `name` for the definitive worst case.

Note the older `StochasticHiddenActsReconLoss` (raw MSE, its own clean + masked passes) and
`StochasticReconSubsetLossConfig.hidden_acts_recon` still exist for older configs.
