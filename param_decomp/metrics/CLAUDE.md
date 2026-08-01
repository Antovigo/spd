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
| `hidden_acts.py` | Shared per-site (hidden-activation) relative-error machinery: clean targets, squared-error accumulation, DDP reduction, site filtering |
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
from the clean input activations the step already cached: **no extra forward pass**, and it
measures accumulated drift from the target model rather than each site's local error given
an already-perturbed input.

All three use `ComponentModel.site_outputs`, which aborts the forward once every hooked
site has been cached (a private sentinel exception, since a forward hook cannot ask PyTorch
to stop). Everything past the last decomposition target would otherwise be wasted compute
*and* retained-for-backward activations.

`site_patterns` (fnmatch, e.g. `["*.mlp.down_proj", "*.self_attn.o_proj"]`) restricts which
sites the error is *measured* at; masking always covers every decomposed site.

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
  `pre_weight_acts` alongside the pre-weight activations. CI fns index that dict by their
  own layer names, so the extra keys are inert.

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
