"""The torch-free pydantic config schema for the algorithm core.

Every algorithm-level config class lives here (or in the sibling `base_config` /
`schedule` modules): routing, the explicit (toy) site spec, loss-metric configs,
eval-metric configs, the top-level `PDConfig` / `Cadence`, the placement table the
engine resolves, and the `wandb.config` shaping helpers. Depends only on pydantic /
numpy / pyyaml / annotated-types (via `base_config`), so non-trainer consumers validate
the same YAML run configs without pulling jax/wandb.

Experiment-level schema (the `ExperimentConfig` base and its LM / TMS / ResidMLP
subclasses, each binding concrete `target`/`decomposition`/`data` sections) lives
lab-side under `param_decomp/experiments/` — including the authored
`decomposition.ci` configs, the tiled LM site specs, and the LM's compute substrate
(`runtime:`), which speak each domain's vocabulary. Core carries only the RESOLVED CI-fn
arches (`ci_fn.py`) and the resolved flat sites.
"""

import copy
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import (
    AliasChoices,
    BeforeValidator,
    Discriminator,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from param_decomp.core.axes import MeshAxis, SemanticAxis
from param_decomp.core.base_config import BaseConfig, Probability
from param_decomp.core.nonlinearity import NonlinearityUnitKind
from param_decomp.core.schedule import ScheduleConfig

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class UniformKSubsetRoutingConfig(BaseConfig):
    """Route each position to a uniformly-sized random subset."""

    type: Literal["uniform_k_subset"] = "uniform_k_subset"


class StaticProbabilityRoutingConfig(BaseConfig):
    """Each position independently routes to each module with probability `p`."""

    type: Literal["static_probability"] = "static_probability"
    p: Probability


class AllRoutingConfig(BaseConfig):
    """Route every position to every module (the `"all"` fast path)."""

    type: Literal["all"] = "all"


# Discriminated union over the subset-routing configs (keyed by ``type``).
SubsetRoutingType = UniformKSubsetRoutingConfig | StaticProbabilityRoutingConfig | AllRoutingConfig


# ---------------------------------------------------------------------------
# Decomposition site (C) specs
# ---------------------------------------------------------------------------


class ExplicitSite(BaseConfig):
    name: str
    C: PositiveInt


class ExplicitCSpec(BaseConfig):
    """Arbitrary named sites with per-site C — the positionless toys (no arch grid, and the
    MLP CI fns have no chunk-homogeneity requirement)."""

    kind: Literal["explicit"] = "explicit"
    sites: list[ExplicitSite] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Loss-metric configs
# ---------------------------------------------------------------------------


type LossCoeff = float | ScheduleConfig
"""A loss coefficient over training: a bare float IS the constant, a `ScheduleConfig` is
evaluated at the current step (the `gamma` pattern) — so warmups, anneals, and
0-until-step activation gates are authorable per term. The float arm is not a parse-time
spelling of the constant schedule: it also carries the values a schedule's positive
`max_val` cannot (a plain 0.0), so consumers resolve via `losses.coeff_at`."""


class LossMetricConfig(BaseConfig):
    """Pydantic config for a metric that can also be used as a training loss.

    `coeff` is required when this metric is listed under `loss_metrics` and must be null
    when listed under `eval.metrics` — both directions are asserted
    (`PDConfig.validate_loss_metrics`;
    `param_decomp.experiments.eval_config.validate_eval_metrics`). It is a `LossCoeff`:
    a bare float or a step-evaluated `ScheduleConfig`.

    `name` overrides the class name as this instance's identity (`Metric.instance_key`),
    letting the same metric class appear under both `loss_metrics` and `eval.metrics`
    with different settings — e.g. a 1-step PGD training loss alongside a 20-step PGD
    eval probe. Leave `None` (the default) and the class name is used.
    """

    coeff: LossCoeff | None = None
    name: str | None = None


class HiddenActsReconstruction(BaseConfig):
    """The auxiliary relative-MSE part of one recon loss (SPEC S35): how hard, and measured
    where. Both are required together — a strength with nowhere to measure, or measurement
    points nothing pulls on, are equally meaningless, so they are one object rather than two
    optional fields. Unlike the eval-only hidden-acts metrics, this compares configured activation
    points by relative error and contributes to the optimization objective."""

    coeff: PositiveFloat | ScheduleConfig = Field(
        ...,
        description=(
            "Strength RELATIVE to the e2e loss: each forward uses "
            "`e2e + coeff * mean_points(relative squared error)`. A training term's outer "
            "`coeff` scales that sum; the eval probe reports it directly (and, having no "
            "training step to read, takes only the float arm). A `ScheduleConfig` is "
            "evaluated at the current step like every other loss coefficient."
        ),
    )
    points: tuple[str, ...] = Field(
        ...,
        description=(
            "Activations compared between the masked and clean forwards, named in the TARGET's "
            "own tap vocabulary — e.g. `resid.19` for the residual stream leaving block 18. "
            "There is no default: which internal activations matter is a question about the "
            "experiment, not something the trainer should guess. Each unique selected physical "
            "value is retained once; the target owns how those values are materialized."
        ),
    )

    @model_validator(mode="after")
    def validate_points(self) -> Self:
        assert self.points, "hidden_acts_reconstruction.points must name at least one activation"
        assert len(set(self.points)) == len(self.points), f"duplicate points: {self.points}"
        return self


class HiddenActsReconstructionMixin(BaseConfig):
    """Adds an optional auxiliary relative-MSE term to a recon loss (SPEC S35), pulling each
    masked forward toward the clean forward at named internal activations rather than only at
    the output. Per-point division by the clean activation's own squared scale keeps points of
    different magnitude and width comparable; the mean over points keeps the coefficient's
    meaning stable as the point count changes.

    Self-contained on the loss, like every other loss-specific input (routing, `n_samples`, the
    PPGD optimizer): a recon term carries everything needed to compute it. In a transformer the
    points are typically residual-stream boundaries after each block, but the loss itself is
    architecture-neutral. Adversarial recon losses ascend this same combined objective. `None`
    (default) disables the auxiliary part for this loss."""

    hidden_acts_reconstruction: HiddenActsReconstruction | None = None


class FaithfulnessLossConfig(LossMetricConfig):
    """Mean per-site squared weight error, relative to each frozen target matrix."""

    type: Literal["FaithfulnessLoss"] = "FaithfulnessLoss"


class FrequencyMinimalityConfig(BaseConfig):
    """The frequency-minimality penalty riding on an imp-min term: a component's per-token
    firing frequency `f_c` (over the whole global batch) penalized by
    `f_c * log2(1 + reference_datapoint_count * f_c)`, summed over components and scaled by
    `coeff`.

    `reference_datapoint_count` (`a'`) is the datapoint count the penalty is normalized against, so
    the curvature is invariant to batch size at a fixed firing rate. Setting it to the run's
    global `batch_size * seq_len` reproduces the implicit `B*T` the old rolled `beta` term
    baked inside its `log2`; coefficients then transfer as `coeff = old imp.coeff * old
    beta`. The `f=0 -> 0` cutoff is inherent to the form.

    `ema_halflife_steps` (when set) evaluates the penalty at a debiased exponential moving
    average of `f_c` across steps instead of the noisy single-batch estimate, with the
    gradient kept at the single-batch scale so `coeff` transfers between the two modes
    (SPEC S8''). Capped at `1e6`: a halflife past the run's length already degenerates
    to a debiased running mean, and fp32 rounding drift in the recurrence grows with the
    halflife (pinned by `test_ema_long_scan_rounding_bounded`). While frequencies move
    faster than the halflife the smoothed penalty lags the batch diagnostic (logged
    alongside as `FrequencyMinimalityLoss_batch`) — estimator convergence, not
    instability (S8'').
    """

    coeff: NonNegativeFloat | ScheduleConfig
    reference_datapoint_count: PositiveInt | Literal["auto"] = Field(
        validation_alias=AliasChoices("reference_datapoint_count", "reference_token_count")
    )
    """`"auto"` resolves `a'` to each pass's OWN `B·T`, so `a' · f_c` is the raw firing
    count — the torch oracle's form, and the only spelling correct on both tPD passes at
    once: SPEC T6 shares one frequency block across the target and non-target passes, whose
    `B·T` differ (a 5-token prompt stream vs a 64-token corpus stream), so any literal is
    right for one and off by their ratio on the other. Prefer it over restating the
    arithmetic, which also goes stale the moment a batch size changes."""
    ema_halflife_steps: PositiveFloat | None = Field(default=None, allow_inf_nan=False, le=1e6)


class ImportanceMinimalityLossConfig(LossMetricConfig):
    """Geman–McClure smooth-L0 importance-minimality penalty on upper-leaky CI values.

    Per-value penalty `phi_gamma(c) = c^2 / (c^2 + gamma^2)` — a smooth approximation to
    the active-component count `1[c>0]`, exact only as `gamma -> 0` — summed over the
    per-component mean activities, with an optional `frequency` penalty over the same
    values. `phi'(0) = 0` and `|phi'| <= 0.65/gamma` everywhere, so there is no singularity
    at the origin (no `eps` floor, no aggressive grad clip) — the gradient is localized
    on the threshold band `c ~ gamma/sqrt(3)` and redescends for clearly-on components.

    `gamma` is the width's full schedule (SPEC S9); annealing it down (knots with
    decreasing `frac`) sharpens the count. Its knots must keep `frac > 0` (asserted
    where the term is built) — a `gamma` touching 0 is never intended.
    """

    type: Literal["ImportanceMinimalityLoss"] = "ImportanceMinimalityLoss"
    gamma: ScheduleConfig
    frequency: FrequencyMinimalityConfig | None = None
    normalize_at_one: bool = False
    """Rescale `phi` so `c = 1` contributes exactly 1 at every gamma (`losses.py`)."""


class NonlinearityLocalityLossConfig(LossMetricConfig):
    """Concentrate each component's write vector on fewer nonlinearity-facing units
    (SPEC S36).

    `relative_threshold` is relative to a uniform unit fraction; annealing it down sharpens
    the soft count. `unit_kind_coefficients` weights each unit kind's component mean and
    must name every kind the target's partitions declare (asserted at step build); a
    `None` entry excludes that kind from the objective outright — no reduction is built,
    which is why weights are strictly positive rather than zero-able.
    """

    type: Literal["NonlinearityLocalityLoss"] = "NonlinearityLocalityLoss"
    coeff: NonNegativeFloat | ScheduleConfig | None = None
    relative_threshold: ScheduleConfig
    unit_kind_coefficients: dict[NonlinearityUnitKind, PositiveFloat | None]

    @model_validator(mode="after")
    def validate_relative_threshold(self) -> Self:
        assert all(knot.frac > 0.0 for knot in self.relative_threshold.points), (
            "relative_threshold knots must all keep frac > 0"
        )
        return self

    @model_validator(mode="after")
    def validate_unit_kind_coefficients(self) -> Self:
        assert any(w is not None for w in self.unit_kind_coefficients.values()), (
            "unit_kind_coefficients must train at least one unit kind"
        )
        return self


class CIMaskedReconLossConfig(LossMetricConfig, HiddenActsReconstructionMixin):
    slow: ClassVar[bool] = False
    type: Literal["CIMaskedReconLoss"] = "CIMaskedReconLoss"


class CIMaskedReconSubsetLossConfig(LossMetricConfig, HiddenActsReconstructionMixin):
    type: Literal["CIMaskedReconSubsetLoss"] = "CIMaskedReconSubsetLoss"
    routing: Annotated[SubsetRoutingType, Field(discriminator="type")] = (
        UniformKSubsetRoutingConfig()
    )


class StochasticReconLossConfig(LossMetricConfig, HiddenActsReconstructionMixin):
    type: Literal["StochasticReconLoss"] = "StochasticReconLoss"
    n_mask_samples: PositiveInt = 1


class StochasticReconSubsetLossConfig(LossMetricConfig, HiddenActsReconstructionMixin):
    type: Literal["StochasticReconSubsetLoss"] = "StochasticReconSubsetLoss"
    routing: Annotated[SubsetRoutingType, Field(discriminator="type")] = (
        UniformKSubsetRoutingConfig()
    )
    n_mask_samples: PositiveInt = 1


class UnmaskedReconLossConfig(LossMetricConfig, HiddenActsReconstructionMixin):
    type: Literal["UnmaskedReconLoss"] = "UnmaskedReconLoss"


PGDInitStrategy = Literal["random", "ones", "zeroes"]

SourceShape = Literal["c", "bc", "sc", "bsc"]
"""The stored adversarial-source shape, spelled over the waist axes in tensor order —
each letter names an axis the source keeps FULL; a missing letter is a size-1 broadcast
axis (the rank always matches the waist, so the elementwise combine broadcasts):

  positionless target:  `c (1, C)` · `bc (B, C)`   (`sc`/`bsc` are invalid — they
                        name a position axis the target lacks)
  positioned target:    `c (1, 1, C)` · `bc (B, 1, C)` — one source per batch
                        element shared over positions — · `sc (1, P, C)` ·
                        `bsc (B, P, C)`

These are the component-source shapes. Each site also carries a distinct delta source
with the same leading shape and no C axis.

One vocabulary for BOTH adversaries: persistent PGD implements all four; per-step
(fresh) PGD implements `c`/`bc`/`bsc` and rejects `sc` at validation."""


class PGDConfig(LossMetricConfig, HiddenActsReconstructionMixin):
    """Shared base for per-step PGD loss configs."""

    init: PGDInitStrategy
    step_size: PositiveFloat
    n_steps: NonNegativeInt
    source_shape: SourceShape

    @model_validator(mode="after")
    def validate_source_shape_implemented(self) -> Self:
        if self.source_shape == "sc":
            raise ValueError("per-step PGD does not implement `sc` (persistent PGD does)")
        return self


class PGDReconLossConfig(PGDConfig):
    """Slow-tier by declaration: each evaluation is `n_steps` full forward+backward
    ascents per batch, and a targeted dual run fans it out over stream x CI-role — at the
    fast cadence it dominated eval wall-clock (~10 min per eval on the L18 8B runs)."""

    slow: ClassVar[bool] = True
    type: Literal["PGDReconLoss"] = "PGDReconLoss"
    n_batches: PositiveInt | None = None
    """Cap on how many eval batches the probe consumes (None = all `eval.n_steps`).
    The reported loss averages over the batches actually run."""


class PGDReconSubsetLossConfig(PGDConfig):
    type: Literal["PGDReconSubsetLoss"] = "PGDReconSubsetLoss"
    routing: Annotated[SubsetRoutingType, Field(discriminator="type")] = (
        UniformKSubsetRoutingConfig()
    )


class AdamPGDConfig(BaseConfig):
    """Adam-style PGD optimizer config — the only implemented persistent-PGD optimizer."""

    type: Literal["adam"] = "adam"
    beta1: Probability = Field(default=0.9, description="Adam beta1 for masks")
    beta2: Probability = Field(default=0.999, description="Adam beta2 for masks")
    eps: NonNegativeFloat = Field(default=1e-8, description="Adam epsilon for masks")
    lr_schedule: ScheduleConfig


class PersistentPGDLossConfig(LossMetricConfig, HiddenActsReconstructionMixin):
    """Shared adversary fields for the persistent-PGD loss terms (SPEC §4.4–4.5): the
    Adam-ascended source bundle's optimizer, stored shape, dtype, and warmup. Sources are
    clamped to `[0, 1]` after each step — the only implemented parameterization."""

    optimizer: AdamPGDConfig
    source_shape: SourceShape
    source_dtype: Literal["float32", "bfloat16"] = "float32"
    """Storage dtype for the persistent PPGD source tensors AND their Adam moments
    (`m`/`v`). `float32` (default) is SPEC N1 (fp32 SRC_STEP moments) and the only
    oracle-parity path. `bfloat16` halves the resident source+moment footprint (it scales
    with total source elements — dominant at large site counts) at some numerical risk: the
    second-moment `v` accumulates squared grads, which can underflow in bf16 for small
    grads — opt in only as an experiment."""
    n_warmup_steps: NonNegativeInt = Field(
        default=0,
        description=(
            "Extra inner PGD source-optimization steps on each train batch before the final loss"
            " computation."
        ),
    )
    adversary_objective: Literal["term", "e2e"] = "e2e"
    """Objective the persistent sources ascend. `e2e` excludes hidden-activation
    reconstruction from source ascents while keeping it in the outer components/CI
    objective; `term` makes the sources ascend the complete loss."""


class PersistentPGDReconLossConfig(PersistentPGDLossConfig):
    """Persistent-PGD recon loss: the adversary's masks routed to all sites every
    forward."""

    type: Literal["PersistentPGDReconLoss"] = "PersistentPGDReconLoss"


class MergedStochasticSubsetPPGDReconLossConfig(PersistentPGDLossConfig):
    """ONE masked forward serving both recon pressures (SPEC S10' variation): each batch
    element is assigned adversarial with probability `adv_fraction` (mask sources = the
    persistent-PGD adversary's, every site routed) or stochastic otherwise (fresh
    `U[0,1]` sources, routed per `routing`) — the whole sequence takes one family, so no
    sample's loss is scored against a mixed-family attention context. `adv_fraction` is a
    `ScheduleConfig`, evaluated per step like the imp-min gamma — a constant is the plain
    merge; a ramp anneals the adversarial share over training. `coeff` is the TOTAL:
    coeff 1.0 + constant adv_fraction 0.5 replaces the canonical 0.5 stochastic + 0.5
    persistent-PGD pair in expectation. Carries the persistent-adversary fields; one
    source bundle feeds this one term (SPEC S23) and the S14' final ascent rides its
    backward — no extra forward."""

    type: Literal["MergedStochasticSubsetPPGDReconLoss"] = "MergedStochasticSubsetPPGDReconLoss"
    adv_fraction: ScheduleConfig
    routing: SubsetRoutingType = Field(default_factory=UniformKSubsetRoutingConfig)

    @model_validator(mode="after")
    def validate_adv_fraction_is_probability(self) -> Self:
        # `frac` knots live in [0, 1] with the peak frac=1.0 attained, so the curve's
        # range is exactly [min_frac, 1.0] · max_val — the peak check is sufficient.
        if self.adv_fraction.max_val > 1.0:
            raise ValueError(f"adv_fraction must stay within [0, 1]: {self.adv_fraction}")
        return self


# ---------------------------------------------------------------------------
# Eval-metric configs
#
# Every eval metric declares `slow: ClassVar[bool]` beside its own definition. An eval
# costs what it costs wherever it is bound, so the tier travels WITH the metric rather
# than being assigned by whichever family binds it. `ClassVar` is what makes it
# unauthorable: pydantic ignores it, and `extra="forbid"` then refuses a YAML `slow:` key.
# No default, so a new metric cannot skip the decision — swept at import in
# `experiments.eval_config`, which also refuses a tier that is merely inherited.
# ---------------------------------------------------------------------------


class CIHistogramsConfig(BaseConfig):
    """The two value histograms bin exactly, over ONE eval batch — counts from batches binned
    against their own min/max sit on different edges and cannot be summed — so
    `n_batches_accum` may only be None or 1. `density_heatmap_n_bins`
    opts into the per-token per-component CI density heatmap (an on-device bincount into that
    many log-spaced `[1e-9, 1]` bands sharing the same forward, accumulated over EVERY batch);
    `None` disables it."""

    slow: ClassVar[bool] = True
    type: Literal["CIHistograms"] = "CIHistograms"
    n_batches_accum: PositiveInt | None
    density_heatmap_n_bins: PositiveInt | None = None


class CI_L0Config(BaseConfig):
    """`groups` maps `{group_name: [fnmatch-style layer pattern, ...]}`.

    Matching layers' L0s are summed into the group and logged under the group's name.
    """

    slow: ClassVar[bool] = False
    type: Literal["CI_L0"] = "CI_L0"
    groups: dict[str, list[str]] | None
    ci_alive_threshold: float = 0.0


class CIMeanPerComponentConfig(BaseConfig):
    slow: ClassVar[bool] = True
    type: Literal["CIMeanPerComponent"] = "CIMeanPerComponent"


class ComponentActivationDensityConfig(BaseConfig):
    slow: ClassVar[bool] = True
    type: Literal["ComponentActivationDensity"] = "ComponentActivationDensity"
    ci_alive_threshold: float = 0.0


class WeightMagnitudeConfig(BaseConfig):
    """Per-site `‖V_c‖·‖U_c‖` scatter in descending magnitude order, log y.

    Reads the trained V/U alone — no forward pass and no eval batch."""

    slow: ClassVar[bool] = True
    type: Literal["WeightMagnitude"] = "WeightMagnitude"


class IdentityCITargetSpec(BaseConfig):
    """A layer expected to produce an Identity CI pattern over `n_features` features."""

    layer_pattern: str
    n_features: PositiveInt


class DenseCITargetSpec(BaseConfig):
    """A layer expected to produce a Dense CI pattern with `k` active components."""

    layer_pattern: str
    k: PositiveInt


class IdentityCIErrorConfig(BaseConfig):
    """`identity_ci` / `dense_ci` list layers expected to produce Identity / Dense patterns."""

    slow: ClassVar[bool] = True
    type: Literal["IdentityCIError"] = "IdentityCIError"
    identity_ci: list[IdentityCITargetSpec] | None
    dense_ci: list[DenseCITargetSpec] | None


class _PermutationPlotsBaseConfig(BaseConfig):
    """fnmatch patterns for layers permuted to align with the corresponding target solution.

    `identity_patterns` and `dense_patterns` are matched separately against the model.
    """

    identity_patterns: list[str] | None
    dense_patterns: list[str] | None


class PermutedCIPlotsConfig(_PermutationPlotsBaseConfig):
    slow: ClassVar[bool] = True
    type: Literal["PermutedCIPlots"] = "PermutedCIPlots"


class UVPlotsConfig(_PermutationPlotsBaseConfig):
    slow: ClassVar[bool] = True
    type: Literal["UVPlots"] = "UVPlots"


class WellTemperednessConfig(BaseConfig):
    """Whether higher causal importance preactivations mean greater ablation effects.

    `groups` maps names to fnmatch-style site patterns. Every region always schedules
    `n_locations * n_components_per_region` solo ablations: a sparse region pads its quota
    with out-of-region components whose damage is computed and discarded.
    """

    slow: ClassVar[bool] = True
    type: Literal["WellTemperedness"] = "WellTemperedness"
    groups: dict[str, list[str]] | None
    n_locations: PositiveInt
    n_components_per_region: PositiveInt
    ablations_per_forward: PositiveInt


# ---------------------------------------------------------------------------
# Top-level PD configs
# ---------------------------------------------------------------------------


class AdamWOptimizerConfig(BaseConfig):
    type: Literal["adamw"] = "adamw"
    lr_schedule: ScheduleConfig = Field(..., description="Learning rate schedule")
    weight_decay: NonNegativeFloat = Field(default=0.0, description="AdamW weight decay")
    betas: tuple[Probability, Probability] = Field(
        default=(0.9, 0.999), description="AdamW (beta1, beta2)"
    )
    grad_clip_norm: PositiveFloat | None = Field(
        default=None,
        description="If set, clip the grad norm of this group's parameters to this value",
    )


class MuonOptimizerConfig(BaseConfig):
    """Muon (`optax.contrib.muon`): Newton-Schulz-orthogonalized momentum for the group's
    matrix leaves; the rest fall back to Adam(0.9, 0.999) at the same LR. Experimental
    (non-canonical). Which leaves are matrices is per-group (`run_state.build_optimizers`):
    the V/U components tree is all-2D (fallback never fires); the chunkwise CI fn is
    per-chunk stacks, so its 3D leaves are muon'd over the trailing two axes (chunk axis
    batched) and its 2D bias stacks take the fallback; the MLP CI fns use the plain 2D rule."""

    type: Literal["muon"]
    lr_schedule: ScheduleConfig = Field(..., description="Learning rate schedule")
    beta: Probability = Field(
        default=0.95, description="Momentum decay for the orthogonalized update"
    )
    consistent_rms: PositiveFloat | None = Field(
        default=None,
        description=(
            "If set, scale updates by `sqrt(max(fan_in, fan_out)) * consistent_rms` so update"
            " RMS is shape-independent (0.2 ~ AdamW's empirical RMS, making the AdamW LR"
            " transferable). If None, optax's width scaling `sqrt(max(1, fan_out / fan_in))`."
        ),
    )
    weight_decay: NonNegativeFloat = Field(default=0.0, description="Weight decay")
    grad_clip_norm: PositiveFloat | None = Field(
        default=None,
        description="If set, clip the grad norm of this group's parameters to this value",
    )
    impl: Literal["optax", "stacked"] = Field(
        default="optax",
        description=(
            "NS implementation. `optax` = per-leaf `optax.contrib.muon` (the reference"
            " semantics, SPEC S20). `stacked` = one batched NS per semantic kind's"
            " stack, executed at the placement table's `ns_compute` waypoint row —"
            " device-local orthogonalization, no per-iteration collectives"
            " (`muon_stacked.py`); same trajectory up to float reassociation (the"
            " SPEC D4 tolerance class)."
        ),
    )
    ns_steps: PositiveInt = Field(
        default=5, description="Newton-Schulz iterations (optax default 5; fewer = cheaper/looser)"
    )
    ns_dtype: Literal["float32", "bfloat16"] = Field(
        default="float32",
        description=(
            "Dtype of the NS orthogonalization only (masters/momentum stay fp32 per N1);"
            " bfloat16 halves NS compute+comm (the Kimi recipe). `stacked` impl only."
        ),
    )


def _default_optimizer_type_adamw(data: object) -> object:
    """AdamW is the canonical optimizer, so a config without `type` (every config predating
    the muon gate, and the common case going forward) discriminates to it."""
    if isinstance(data, dict) and "type" not in data:
        return {**data, "type": "adamw"}
    return data


AnyOptimizerConfig = Annotated[
    AdamWOptimizerConfig | MuonOptimizerConfig,
    Discriminator("type"),
    BeforeValidator(_default_optimizer_type_adamw),
]


ReconLossMetricConfigs = (
    CIMaskedReconLossConfig
    | CIMaskedReconSubsetLossConfig
    | MergedStochasticSubsetPPGDReconLossConfig
    | PersistentPGDReconLossConfig
    | PGDReconLossConfig
    | PGDReconSubsetLossConfig
    | StochasticReconLossConfig
    | StochasticReconSubsetLossConfig
    | UnmaskedReconLossConfig
)
"""The full recon vocabulary as a PLAIN assignment union. A PEP 695 `type` alias is not a
runtime Union, and pydantic's `Discriminator` keys on one — so the members are spelled here
once and both the `type` alias below and every `Annotated[..., Discriminator]` use point at
this, rather than a second copy of the member list drifting out of sync."""

type AnyReconLossMetricConfig = ReconLossMetricConfigs


AnyLossMetricConfig = Annotated[
    AnyReconLossMetricConfig
    | FaithfulnessLossConfig
    | ImportanceMinimalityLossConfig
    | NonlinearityLocalityLossConfig,
    Discriminator("type"),
]
"""The trainable losses. The hidden-acts metrics are EVAL vocabulary
(`AnyEvalMetricConfig`, SPEC S31) — hidden-acts pressure on TRAINING rides a recon
term's `hidden_acts_reconstruction` (SPEC S35), never a standalone term."""


TargetedLossMetricConfig = Annotated[
    AnyReconLossMetricConfig | ImportanceMinimalityLossConfig | NonlinearityLocalityLossConfig,
    Discriminator("type"),
]
"""The loss types a tPD TARGET pass admits (SPEC T3/S36): the full recon vocabulary
(adversaries run in the target pass, T7) + importance-minimality + the optional
nonlinearity-locality prior — no `FaithfulnessLossConfig` member, so a targeted config
cannot spell a faithfulness role, and no eval-only hidden-acts type.

The nonlinearity term is admitted here even though it belongs to no PASS: it is a
weight-space prior on `U` alone (S36 — CI-ungated, data-independent), so it is scored
once against the components and added once to the step's total, exactly as in plain PD.
It is authored on the target list because that is the tPD config's one loss list, not
because the target stream scores it. NOTE (T3 interaction): with no faithfulness role the
weight delta is unpenalized, so concentration pressure on `U` has an escape valve plain PD
does not give it — expect a weaker effect per unit coefficient than the plain-PD seats."""


class UnmaskedNoDeltaReconLossConfig(LossMetricConfig):
    """The tPD non-target pass's unmasked reconstruction term — T4's one delta-OFF arm:
    every component mask `1.0` and every weight-delta mask `0.0`, so the FULL component
    sum alone must reconstruct the frozen output. Prevents components that never activate
    from interfering with the reconstruction (the tPD paper's CSS-only unmasked recon
    term, Method details). Fully determined: no routing, sampling, or optional
    hidden-activation reconstruction fields exist here, and it is non-target-only — the plain and target-pass unions have no
    member for it."""

    slow: ClassVar[bool] = False
    type: Literal["UnmaskedNoDeltaReconLoss"] = "UnmaskedNoDeltaReconLoss"


NontargetReconLossMetricConfig = (
    CIMaskedReconLossConfig
    | CIMaskedReconSubsetLossConfig
    | StochasticReconLossConfig
    | StochasticReconSubsetLossConfig
    | UnmaskedNoDeltaReconLossConfig
)
"""The CLOSED non-target recon set (SPEC T5): the stochastic/constant-source ones
(delta pinned fully ON, T4) plus `UnmaskedNoDeltaReconLoss` — T4's one enumerated
delta-OFF exception. The non-target HIDDEN pass admits exactly this set; the OUTPUT
pass's union below additionally admits the persistent adversarial types (T5/T7
amended 2026-08-19)."""


NontargetOutputReconLossMetricConfig = (
    NontargetReconLossMetricConfig
    | PersistentPGDReconLossConfig
    | MergedStochasticSubsetPPGDReconLossConfig
)
"""The non-target OUTPUT pass's vocabulary (SPEC T5/T7 amended 2026-08-19): the closed
non-target set plus the two persistent-source-carrying recon types. Their masks compose
from the bundle's COMPONENT channels with every weight-delta mask pinned to 1.0 (T4 —
the source's delta channel is ignored), so the probed worst case is a partial ablation
of components on data they should be inactive on. Each such term gets its own bundle
(S23), sized off the BROAD stream's geometry."""


NontargetHiddenReconLossMetricConfig = (
    NontargetReconLossMetricConfig | MergedStochasticSubsetPPGDReconLossConfig
)
"""The non-target HIDDEN pass's vocabulary (SPEC T5/T12 amended 2026-08-20): the closed
set plus the MERGED term ONLY — the one adversarial type whose masked forward REPLACES a
stochastic term's (S34), so admitting it costs zero extra forwards on the broad stream.
The standalone persistent-PGD type stays refused here: it would add a masked forward per
step, and hidden-acts robustness off-target is not worth dedicated compute (the
pgd_nt_plan scope decision — this amendment admits only the free adversary). Delta
pinned per T4 like every non-target forward; own bundle per term (S23), sized off the
BROAD stream's geometry like the output pass's."""


HiddenReconLossMetricConfig = ReconLossMetricConfigs
"""The recon types the hidden pass admits on the TARGET stream (SPEC T12): the same full
vocabulary as the output pass, adversaries included — the hidden objective is an ordinary
reconstruction objective that happens to be measured at internal activations rather than at
the output, so nothing about the mask-source algebra changes."""


class HiddenPassConfig(BaseConfig):
    """The hidden role's surface (SPEC T12): a second reconstruction objective, scored at named
    internal activations instead of the model output, whose masks come from the CI fn's SECOND
    readout head (S37).

    `points` lives HERE rather than on each term — the pass IS "reconstruct these activations",
    so per-term points would be re-authoring the objective per term. It is also shared with the
    non-target stream's hidden surface, the same way T6 shares one importance-minimality config
    across passes: WHAT is being reconstructed is a property of the experiment, not of a stream.
    Only coefficients differ per pass.

    Requires `decomposition.ci.dual` — without a second head there is no hidden CI to shape."""

    points: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description=(
            "Activations the hidden objective reconstructs, in the TARGET's own tap "
            "vocabulary — e.g. `layers.19.mlp.down_proj.out` for block 19's MLP output. Each "
            "point's error is normalized by its own clean squared scale and the pass takes the "
            "mean, so points of different magnitude and width stay comparable. No default: "
            "which activations matter is the experiment's question."
        ),
    )
    impmin_coeff: NonNegativeFloat | ScheduleConfig = Field(
        ...,
        description=(
            "The hidden head's importance-minimality COEFFICIENT. The penalty's shape and "
            "anneal are the target pass's, shared by construction (T6's rule extended to the "
            "hidden role) — only the strength is authored here."
        ),
    )
    recon: list[Annotated[HiddenReconLossMetricConfig, Discriminator("type")]] = Field(
        ..., min_length=1
    )

    @model_validator(mode="after")
    def validate_hidden_entries(self) -> Self:
        seen: set[str] = set()
        assert len(set(self.points)) == len(self.points), f"duplicate points: {self.points}"
        for cfg in self.recon:
            assert cfg.coeff is not None, f"hidden.recon {cfg.type!r} must set `coeff`"
            name = cfg.name if cfg.name is not None else cfg.type
            assert name not in seen, f"duplicate hidden loss {name!r}"
            seen.add(name)
            assert cfg.hidden_acts_reconstruction is None, (
                f"hidden.recon {cfg.type!r}: the S35 rider has no place on a hidden-pass term — "
                "the pass already scores exactly these activations via `hidden.points`, and a "
                "rider would add a SECOND hidden-acts term on top of its own"
            )
        return self


class NontargetHiddenConfig(BaseConfig):
    """The hidden role on the NON-TARGET stream (SPEC T12/T5-amended).

    Structurally the non-target pass's twin: its own coefficients, the closed non-target recon
    vocabulary plus the zero-extra-forward MERGED adversarial term (T5/T12 amended
    2026-08-20), and the measurement points of the target-stream `hidden:` block. Delta
    pinned fully on, as every non-target forward is (T4)."""

    impmin_coeff: NonNegativeFloat | ScheduleConfig
    recon: list[Annotated[NontargetHiddenReconLossMetricConfig, Discriminator("type")]] = Field(
        ..., min_length=1
    )

    @model_validator(mode="after")
    def validate_nontarget_hidden_entries(self) -> Self:
        seen: set[str] = set()
        for cfg in self.recon:
            assert cfg.coeff is not None, f"nontarget.hidden.recon {cfg.type!r} must set `coeff`"
            name = cfg.name if cfg.name is not None else cfg.type
            assert name not in seen, f"duplicate non-target hidden loss {name!r}"
            seen.add(name)
            if not isinstance(cfg, UnmaskedNoDeltaReconLossConfig):
                assert cfg.hidden_acts_reconstruction is None, (
                    f"nontarget.hidden.recon {cfg.type!r}: the S35 rider has no place here — "
                    "the hidden pass already scores `hidden.points`"
                )
        return self


class NontargetConfig(BaseConfig):
    """The tPD non-target pass, authored directly (SPEC T5) — never derived from the
    target pass's loss list.

    `batch_size` is the broad stream's GLOBAL batch; `pd.batch_size` stays the target
    stream's. A non-target OUTPUT recon term may carry a persistent adversary (T5/T7
    amended 2026-08-19), whose bundle sizes off THIS pass's geometry — `batch_size` x
    the broad dataset's own seq len — never `pd.batch_size`. `impmin_coeff` is the non-target
    pass's own importance-minimality coefficient (a bare float or a step-evaluated
    schedule) — the penalty's shape and anneal are the target pass's, shared by
    construction (`objective.build_targeted_objective`)."""

    batch_size: PositiveInt
    impmin_coeff: NonNegativeFloat | ScheduleConfig
    recon: list[Annotated[NontargetOutputReconLossMetricConfig, Discriminator("type")]] = Field(
        ..., min_length=1
    )
    hidden: NontargetHiddenConfig | None = Field(
        default=None,
        description=(
            "The hidden role on this stream (SPEC T12). Requires `pd.hidden`, whose `points` "
            "it measures at. None (the default) leaves the hidden head shaped by the target "
            "stream alone."
        ),
    )

    @model_validator(mode="after")
    def validate_nontarget_entries(self) -> Self:
        """Per-entry facts the shared recon classes can spell but the non-target pass
        refuses — caught at parse (seat authoring, submit validation), not at objective
        build on the GPUs."""
        seen: set[str] = set()
        for cfg in self.recon:
            assert cfg.coeff is not None, f"nontarget.recon {cfg.type!r} must set `coeff`"
            name = cfg.name if cfg.name is not None else cfg.type
            assert name not in seen, f"duplicate non-target loss {name!r}"
            seen.add(name)
            # `UnmaskedNoDeltaReconLoss` has no `hidden_acts_reconstruction` field
            # (fully determined; `extra="forbid"` refuses it at parse), so only the
            # shared classes need the check.
            if not isinstance(cfg, UnmaskedNoDeltaReconLossConfig):
                assert cfg.hidden_acts_reconstruction is None, (
                    f"nontarget.recon {cfg.type!r}: hidden_acts_reconstruction has no place on "
                    "the non-target pass (SPEC T5) — with the delta pinned on, "
                    "internal-activation matching would constrain exactly the behavior tPD "
                    "deliberately declines to decompose"
                )
        return self


RuleConfig = dict[SemanticAxis, MeshAxis | list[MeshAxis] | None]
"""One placement row as configured: semantic axis name -> mesh axis, ordered mesh axes,
or null (replicate). Both name vocabularies are the closed `axes.py` Literals, so an
axis name outside them is refused at parse; rules construction additionally fails
closed on any in-vocabulary key no tensor consumes at that row (a typo'd axis would
silently replicate) and on mesh axes the run's bound mesh does not declare."""


class ComponentsPlacementConfig(BaseConfig):
    """The component V/U lifecycle rows of an explicit placement table. ONE set of rows
    places every semantic group — there are no per-group fallback rows; a group these
    rows cannot place refuses at rules construction."""

    optimizer_state: RuleConfig
    compute_weights: RuleConfig
    faithfulness_weights: RuleConfig
    faithfulness_deltas: RuleConfig
    operands: RuleConfig
    ns_compute: RuleConfig


class CIWeightPlacementConfig(BaseConfig):
    """The lifecycle rows for one CI weight family. `ns_compute` is the muon NS staging
    waypoint (`stack` only; matrices whole per device — `placement.ns_staging_sharding`)."""

    optimizer_state: RuleConfig
    compute_weights: RuleConfig
    operands: RuleConfig
    ns_compute: RuleConfig


class CIFnPlacementConfig(BaseConfig):
    """Chunkwise CI-transformer weight and activation roles."""

    attention: CIWeightPlacementConfig
    ffn: CIWeightPlacementConfig
    input: CIWeightPlacementConfig
    output: CIWeightPlacementConfig
    vectors: RuleConfig
    activations: RuleConfig


class ActivationsPlacementConfig(BaseConfig):
    """The public and component-internal activation placement rows."""

    external: RuleConfig
    component: RuleConfig


TargetActivationRef = Literal["external", "intermediate"]


class TargetLinearPlacementConfig(BaseConfig):
    """One frozen-target linear declaration."""

    persist: RuleConfig
    operand: RuleConfig
    input: TargetActivationRef
    output: TargetActivationRef


class TargetComponentLinearPlacementConfig(BaseConfig):
    """The component-replaced linear's public activation contract."""

    input: TargetActivationRef
    output: TargetActivationRef


class TargetWeightPlacementConfig(BaseConfig):
    """A frozen target weight's resting and execution layouts."""

    persist: RuleConfig
    operand: RuleConfig


class TargetPlacementConfig(BaseConfig):
    """Every frozen-target weight role and its execution contract."""

    embedding: TargetWeightPlacementConfig
    normalization: RuleConfig
    position_encoding: RuleConfig
    column: TargetLinearPlacementConfig
    row: TargetLinearPlacementConfig
    output: TargetWeightPlacementConfig
    intermediate: RuleConfig
    component: TargetComponentLinearPlacementConfig


class PlacementTableConfig(BaseConfig):
    """An explicit placement table (`runtime.sharding`), mirroring the typed
    `placement.PlacementRules`. The row vocabulary is CLOSED (extra keys are a parse
    error); rule values are free-form axis-name -> mesh-axes mappings, where YAML list
    order is semantics (nested-axis linearization — PLACEMENT_DESIGN.md invariant 5)."""

    components: ComponentsPlacementConfig
    ci_fn: CIFnPlacementConfig
    activations: ActivationsPlacementConfig
    target: TargetPlacementConfig


class NamedNeuronRanks(BaseConfig):
    """A neuron-ranking artifact in the store: `<data_root>/neuron_ranks/<name>` — what the
    LM `initialization: neuron_aligned_targeted` starts from (SPEC T13). Names are
    immutable versions — a re-harvest is a new name."""

    kind: Literal["name"] = "name"
    name: str

    @model_validator(mode="after")
    def _flat_name(self) -> Self:
        assert self.name and "/" not in self.name and "*" not in self.name, (
            f"neuron-ranks names are flat store names: {self.name!r}"
        )
        return self


class NeuronRanksDir(BaseConfig):
    """Ad-hoc escape hatch: an explicit artifact directory. Machine-specific by nature,
    so the path must be absolute; a named store artifact is the portable form."""

    kind: Literal["dir"] = "dir"
    dir: Path

    @model_validator(mode="after")
    def _absolute(self) -> Self:
        assert self.dir.is_absolute(), f"ad-hoc neuron-ranks dirs are absolute paths: {self.dir}"
        return self


NeuronRanksRef = Annotated[NamedNeuronRanks | NeuronRanksDir, Field(discriminator="kind")]


class PDConfigBase(BaseConfig):
    """The algorithm sections shared by every run shape: seed, losses, optimizers, sizes.

    Domain-agnostic — the target-coupled apparatus (which sites to decompose + the CI-fn
    arch) lives in the per-domain `decomposition` section, not here. Flipping any field here
    changes what algorithm runs. The concrete shapes are `PDConfig` (plain VPD: the full
    loss vocabulary + the faithfulness warmup) and `TargetedPDConfig` (tPD, SPEC §11: the
    faithfulness-free loss vocabulary, no warmup fields at all — a targeted config cannot
    SPELL a faithfulness role, T3). Pair with `Cadence` (when to emit) when running the
    trainer (`param_decomp.core.run`); the compute substrate reaches the engine unpacked
    into primitives, never as a config object.
    """

    seed: int = Field(
        default=0,
        description="Random seed for reproducibility, including LM dataset shuffling.",
    )
    components_optimizer: AnyOptimizerConfig = Field(
        ..., description="Optimizer config for the component (LinearComponent etc.) parameters"
    )
    ci_fn_optimizer: AnyOptimizerConfig = Field(
        ..., description="Optimizer config for the CI function parameters"
    )
    steps: PositiveInt = Field(..., description="Total number of optimisation steps")
    batch_size: PositiveInt = Field(
        ...,
        description=(
            "Global batch size (may be divided across multiple devices). For a targeted "
            "run this is the TARGET stream's batch (T2); the broad stream's lives on "
            "`nontarget.batch_size`."
        ),
    )


class PDConfig(PDConfigBase):
    """The plain-VPD algorithm shape: the full loss vocabulary + the faithfulness warmup."""

    type: Literal["faithful"] = "faithful"

    loss_metrics: list[AnyLossMetricConfig] = Field(
        ...,
        min_length=1,
        description=(
            "Training-loss metrics. Each entry's `type` field selects the concrete metric; "
            "`coeff` weights it in the total training loss. Active loss metrics are automatically"
            " also evaluated."
        ),
    )

    # --- Faithfulness Warmup ---
    faithfulness_warmup_steps: NonNegativeInt = Field(
        default=0,
        description="Number of warmup steps to optimize faithfulness loss before main training",
    )
    faithfulness_warmup_lr: PositiveFloat = Field(
        default=0.001,
        description="Learning rate for warmup phase (optimizing faithfulness loss only)",
    )
    faithfulness_warmup_weight_decay: NonNegativeFloat = Field(
        default=0.0,
        description="Weight decay for warmup phase optimizer",
    )

    @model_validator(mode="after")
    def validate_loss_metrics(self) -> Self:
        _validate_training_losses(self.loss_metrics)
        faith_terms = [cfg for cfg in self.loss_metrics if isinstance(cfg, FaithfulnessLossConfig)]
        assert len(faith_terms) == 1, f"need exactly one FaithfulnessLoss, got {len(faith_terms)}"
        return self


class TargetedPDConfig(PDConfigBase):
    """The tPD algorithm shape (SPEC §11): the faithfulness-free loss vocabulary, and no
    faithfulness-warmup fields at all — warmup drives the weight delta to zero, and tPD
    needs the delta free to carry off-target behavior (T3), so the knobs do not exist
    here rather than being validated to zero. `batch_size` is the TARGET stream's (T2)."""

    loss_metrics: list[TargetedLossMetricConfig] = Field(
        ...,
        min_length=1,
        description=(
            "TARGET-pass training losses; the non-target pass is authored separately on "
            "`nontarget:` and never derived from this list."
        ),
    )

    hidden: HiddenPassConfig | None = Field(
        default=None,
        description=(
            "The hidden role's TARGET-stream surface (SPEC T12): a second reconstruction "
            "objective scored at internal activations, whose masks come from the CI fn's "
            "second readout head. Requires `decomposition.ci.dual`. None (the default) is an "
            "ordinary single-objective tPD run, byte-inert."
        ),
    )

    ci_scaled_weight_decay: PositiveFloat | None = Field(
        default=None,
        description=(
            "CI-scaled weight decay on the subcomponent V/U vectors (SPEC T11): after each "
            "optimizer step every subcomponent's V column and U row scale by "
            "`1 - lr*wd*(1 - max CI)` with the max over BOTH streams' batches, so dead "
            "components — never important on either stream — get dragged to zero. None "
            "disables (the default; the term is an optional auxiliary)."
        ),
    )

    @model_validator(mode="after")
    def validate_loss_metrics(self) -> Self:
        _validate_training_losses(self.loss_metrics)
        for metric in self.loss_metrics:
            match metric:
                case ImportanceMinimalityLossConfig(frequency=frequency) if frequency is not None:
                    assert frequency.ema_halflife_steps is None, (
                        "frequency.ema_halflife_steps is not implemented for the targeted "
                        "(tPD) objective: the EMA carries one frequency stream per site, and "
                        "the two-pass step takes the penalty on two independent streams "
                        "(SPEC S8'' — plain PD only)"
                    )
                case _:
                    pass
        return self

    @model_validator(mode="after")
    def validate_hidden_pass(self) -> Self:
        """Identities are unique ACROSS passes: every term's log key is `loss/<name>`, so a
        hidden term sharing a target term's default name (its type literal) would silently
        overwrite it in the metrics. Refuse at parse rather than lose a curve."""
        if self.hidden is None:
            return self
        target_names = {cfg.name if cfg.name is not None else cfg.type for cfg in self.loss_metrics}
        for cfg in self.hidden.recon:
            name = cfg.name if cfg.name is not None else cfg.type
            assert name not in target_names, (
                f"hidden.recon {name!r} collides with a target-pass loss identity: both log "
                f"under `loss/{name}`. Give one of them an explicit `name`."
            )
        return self


AnyPDConfig = PDConfig | TargetedPDConfig
"""The closed set of run-shape algorithm configs — consumers that read `loss_metrics`
off a shape-blind `pd` take this union; base-field-only consumers take `PDConfigBase`."""


def _validate_training_losses(loss_metrics: Sequence[AnyLossMetricConfig]) -> None:
    """The role/identity facts every training-loss list must satisfy, refused at parse:
    coefficients set, identities unique (`name or type` — the logged instance key),
    exactly one importance-minimality term, at least one recon term, and at most one
    nonlinearity term. Faithfulness multiplicity is the plain shape's own claim (the
    targeted union has no member)."""
    seen: set[str] = set()
    for cfg in loss_metrics:
        assert cfg.coeff is not None, f"loss_metrics.{cfg.type!r} must set `coeff`"
        name = cfg.name if cfg.name is not None else cfg.type
        assert name not in seen, f"duplicate loss instance_key {name!r}"
        seen.add(name)
    imp_terms = [cfg for cfg in loss_metrics if isinstance(cfg, ImportanceMinimalityLossConfig)]
    assert len(imp_terms) == 1, f"need exactly one importance-minimality term, got {len(imp_terms)}"
    recon_terms = [
        cfg
        for cfg in loss_metrics
        if not isinstance(
            cfg,
            FaithfulnessLossConfig
            | ImportanceMinimalityLossConfig
            | NonlinearityLocalityLossConfig,
        )
    ]
    assert recon_terms, "need at least one recon loss term"
    nonlinearity_terms = [
        cfg for cfg in loss_metrics if isinstance(cfg, NonlinearityLocalityLossConfig)
    ]
    assert len(nonlinearity_terms) <= 1, (
        f"need at most one nonlinearity-locality term, got {len(nonlinearity_terms)}"
    )


class DenseLogPhase(BaseConfig):
    """Denser train-log period for the first `until_step` steps, then `Cadence`'s
    steady `train_log_every` takes over. Early training has the fastest dynamics
    (faithfulness warmup, the sharp initial loss drop), so denser sampling there is
    where the signal is; the per-log overhead is a few scalar cross-pool reductions,
    negligible against the step. `until_step` is exclusive."""

    every: PositiveInt
    until_step: PositiveInt


class KeepLastNCheckpoints(BaseConfig):
    """Keep only the `n` most recent `ckpts/<step>/` directories; every save deletes
    whatever falls out of that window."""

    kind: Literal["keep_last"] = "keep_last"
    n: PositiveInt


class KeepAllCheckpoints(BaseConfig):
    """Keep every checkpoint the run writes — the whole trajectory stays on disk."""

    kind: Literal["keep_all"] = "keep_all"


CheckpointRetention = Annotated[KeepLastNCheckpoints | KeepAllCheckpoints, Discriminator("kind")]
"""Which of a periodic run's written checkpoints survive on disk. Retention only prunes
what has already been written; writing nothing at all is `NoCheckpointing` — a sibling
`Checkpointing` arm, deliberately not a keep-nothing retention kind."""


class PeriodicCheckpointing(BaseConfig):
    """Checkpoint on `save_every`, on SIGTERM, and at the final step — the resumable run
    shape (SPEC S22). Under `keep_last` retention the retained window always contains the
    newest checkpoint, so the final-step one is never pruned."""

    kind: Literal["periodic"] = "periodic"
    save_every: PositiveInt
    retention: CheckpointRetention


class NoCheckpointing(BaseConfig):
    """The run never writes `ckpts/` — no periodic saves, no final-step save, and no
    SIGTERM save either: a preemption or cancel exits without writing, so an interrupted
    run costs zero checkpoint bytes. With nothing to resume from, the run id is
    single-entry — re-entering it is refused at engine startup. For probe/measurement
    runs whose trajectory is throwaway; the trained decomposition is unrecoverable."""

    kind: Literal["none"] = "none"


Checkpointing = Annotated[PeriodicCheckpointing | NoCheckpointing, Discriminator("kind")]
"""Whether (and on what rhythm) the trainer writes `ckpts/<step>/` checkpoints."""


class Cadence(BaseConfig):
    """Rhythm of non-eval loop emissions: the train-log period and checkpointing.

    Held separately from `RunSink` so the sink only owns *where* output goes; `Cadence`
    owns *when* train logs and checkpoints fire. Eval timing lives on `EvalLoop`,
    alongside the runtime objects it depends on.
    """

    train_log_every: PositiveInt
    checkpointing: Checkpointing
    dense_log_phase: DenseLogPhase | None = None
    """Optional denser logging for early training; `None` means a flat `train_log_every`."""

    def should_log_train(self, step: int) -> bool:
        if self.dense_log_phase is not None and step < self.dense_log_phase.until_step:
            return step % self.dense_log_phase.every == 0
        return step % self.train_log_every == 0


# ---------------------------------------------------------------------------
# Run-level config (logging + fine-tune lineage)
# ---------------------------------------------------------------------------


class WandbConfig(BaseConfig):
    """Wandb logging settings. Presence on `ExperimentConfig` opts in; omit to skip wandb."""

    project: str
    entity: str | None = None
    group: str | None = None
    """Wandb UI group; None = ungrouped."""
    tags: list[str] = Field(default_factory=list)
    """Wandb tags; empty = untagged."""


class ResumeProvenance(BaseConfig):
    """Fine-tune lineage: a fresh run initialized from a PARENT decomposition. Lives on
    `ExperimentConfig`.

    A fine-tune run gets its own `run_id` / `launch_config.yaml` / `ckpts/`; this records the
    parent it forked from. The JAX trainer (SPEC S33) loads the parent checkpoint's
    V/U + ci_fn onto a fresh reference state and trains a clean schedule from step 0
    (fresh optimizer / sources) under the new config — only when the run's own `ckpts/`
    is empty (a subsequent SLURM requeue resumes from the run's own dir, ignoring
    provenance). The structure (sites / C / ci-fn arch) must match the parent; only
    LR / coeffs / gamma / seq / batch / steps may change. Provenance flows into
    `launch_config.yaml` and `wandb.config` so the lineage is visible in the wandb UI. A run with
    `resume_provenance is None` is a fresh-from-init run.
    """

    parent_run_dir: Path
    """Path to the parent run's directory (the dir that contains `ckpts/<parent_step>/`)."""

    parent_step: int
    """The parent's orbax `ckpts/<step>/` checkpoint step to initialize V/U + ci_fn from."""


# ---------------------------------------------------------------------------
# wandb.config shaping
# ---------------------------------------------------------------------------

IMP_MIN_METRIC_NAMES: dict[str, str] = {
    "imp": "ImportanceMinimalityLoss",
    "imp_smooth_l0": "SmoothL0ImportanceMinimalityLoss",
    "freq": "FrequencyMinimalityLoss",
    "freq_batch": "FrequencyMinimalityLoss_batch",
}
"""Step-record short key -> logged loss name, for the terms whose record key is not
already the term's class name. Shared so the target stream (expanded by `run._METRIC_KEYS`)
and the non-target stream (keyed in `train.make_targeted_train_step`) cannot drift apart:
both streams must spell the same quantity the same way."""


METRIC_SHORT_NAMES: dict[str, str] = {
    "CIMaskedReconLoss": "CIMaskRecon",
    "CIMaskedReconSubsetLoss": "CIMaskReconSub",
    "FaithfulnessLoss": "Faith",
    "NonlinearityLocalityLoss": "Nonlinearity",
    "ImportanceMinimalityLoss": "ImpMin",
    "PersistentPGDReconLoss": "PersistPGDRecon",
    "PGDReconLoss": "PGDRecon",
    "PGDReconSubsetLoss": "PGDReconSub",
    "StochasticReconLoss": "StochRecon",
    "StochasticReconSubsetLoss": "StochReconSub",
    "UnmaskedReconLoss": "UnmaskedRecon",
    "ABGridDataset": "ABGrids",
    "CEandKLLosses": "CEandKL",
    "CIHistograms": "CIHist",
    "CI_L0": "CI_L0",
    "CIMaskedAttnPatternsReconLoss": "CIAttnRecon",
    "CIMeanPerComponent": "CIMeanPerComp",
    "ComponentActivationDensity": "CompActDens",
    "IdentityCIError": "IdCIErr",
    "PermutedCIPlots": "PermCIPlots",
    "StochasticAttnPatternsReconLoss": "StochAttnRecon",
    "UVPlots": "UVPlots",
}


def flatten_typed_lists(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested lists-of-typed-dicts (loss/eval metric lists) into queryable flat
    keys addressed by metric `short_name`, returning a copy with the raw lists dropped.

    Example: `pd.loss_metrics: [{type: "ImportanceMinimalityLoss", coeff: 0.1}]`
    becomes `pd.loss_metrics.ImpMin.coeff: 0.1`, and the raw `pd.loss_metrics` list is
    removed so wandb doesn't also log it as an opaque JSON blob. A metric type with no
    entry in `METRIC_SHORT_NAMES` falls back to its raw type string.
    """
    flattened: dict[str, Any] = {}

    def is_typed_list(obj: Any) -> bool:
        return (
            isinstance(obj, list)
            and len(obj) > 0
            and all(isinstance(x, dict) and "type" in x for x in obj)
        )

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                child = obj[key]
                child_path = f"{path}.{key}" if path else key
                if is_typed_list(child):
                    for entry in child:
                        short = METRIC_SHORT_NAMES.get(entry["type"], entry["type"])
                        for k, v in entry.items():
                            if k == "type":
                                continue
                            flattened[f"{child_path}.{short}.{k}"] = v
                    del obj[key]
                else:
                    walk(child, child_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                walk(item, f"{path}.{i}")

    out = copy.deepcopy(config_dict)  # walk dels nested keys; never mutate the caller's dict
    walk(out, "")
    out.update(flattened)
    return out
