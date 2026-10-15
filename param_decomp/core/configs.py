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
from itertools import pairwise
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
from param_decomp.metric_schema import MetricSchema

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
`max_val` cannot (a plain 0.0), so the trainer carries it as a `RuntimeSchedule` with a traced magnitude."""


class TrainingLossConfig(BaseConfig):
    """A loss weighted into the training objective by `coeff`.

    `name` overrides the serialized `type` as this instance's logged identity, so two
    instances of one loss type can train side by side. Leave `None` (the default) and the
    loss's tag is used."""

    name: str | None = None
    coeff: LossCoeff


class BatchHiddenActsNormalization(BaseConfig):
    """One ratio per point over the whole batch: `Σ_{b,t,d}(m−c)² / Σ_{b,t,d}c²` (the
    original relative-squared-error form). Every position's ABSOLUTE squared error is pooled
    against the batch's total clean energy, so a position holding most of that energy (the BOS
    massive activation at Llama's first and last MLP outputs) is the whole measurement and the
    others are silenced."""

    kind: Literal["batch"] = "batch"


class PerPositionHiddenActsNormalization(BaseConfig):
    """One ratio per (batch, position) entry, then the mean over entries:
    `mean_{b,t} ‖m−c‖²_{b,t} / (‖c‖²_{b,t} + floor_fraction · median_{b,t}‖c‖²)` (local
    amendment 2026-09-11). Each position is measured against its own clean scale, so no
    position can silence another, and the mean over entries keeps the per-token weight at
    `1/(B·T)` — the same normalization the KL and importance-minimality terms carry, so the
    recon/sparsity balance stays batch- and stream-invariant. The floor is a fraction of the
    MEDIAN per-entry clean energy (the median, not the mean, so an outlier position cannot set
    it) and guards near-silent positions whose own ratio would otherwise explode."""

    kind: Literal["per_position"] = "per_position"
    floor_fraction: PositiveFloat = Field(
        default=0.01,
        description=(
            "Fraction of the median per-entry clean energy added to every entry's denominator. "
            "Strictly positive: a zero floor lets a silent position divide by zero."
        ),
    )


HiddenActsNormalization = Annotated[
    BatchHiddenActsNormalization | PerPositionHiddenActsNormalization, Discriminator("kind")
]
"""How a relative-squared-error comparison pools over positions."""

BATCH_HIDDEN_ACTS_NORMALIZATION = BatchHiddenActsNormalization()
"""The default (batch-pooled) normalization, as a module-level singleton for argument defaults."""


class CaptureReconstruction(BaseConfig):
    """One target-owned capture and the numerical distance applied to its paired values.

    `normalization` refines `relative_squared_error` only: `batch` (the default) is one ratio
    over the whole batch, `per_position` one ratio per (batch, position) entry, averaged."""

    capture: str = Field(min_length=1)
    distance: Literal["relative_squared_error", "categorical_kl_from_logits"]
    normalization: HiddenActsNormalization = Field(default_factory=BatchHiddenActsNormalization)

    @model_validator(mode="after")
    def validate_normalization_distance(self) -> Self:
        assert self.distance == "relative_squared_error" or isinstance(
            self.normalization, BatchHiddenActsNormalization
        ), f"{self.capture}: `normalization` refines relative_squared_error only"
        return self


class AuxiliaryReconstructionConfig(BaseConfig):
    """Add `coeff * mean(comparisons)` to the same masked output reconstruction."""

    name: str = Field(min_length=1, pattern=r"^[^/]+$")
    coeff: NonNegativeFloat | ScheduleConfig
    comparisons: tuple[CaptureReconstruction, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_comparisons(self) -> Self:
        assert self.name != "e2e", "e2e names the model-output comparison"
        captures = tuple(comparison.capture for comparison in self.comparisons)
        assert len(set(captures)) == len(captures), f"duplicate comparison captures: {captures}"
        return self


class ReconstructionAuxiliariesMixin(BaseConfig):
    """Named comparison groups sharing their reconstruction term's masks and draws."""

    auxiliaries: tuple[AuxiliaryReconstructionConfig, ...] = ()

    @model_validator(mode="after")
    def validate_auxiliary_names(self) -> Self:
        names = tuple(auxiliary.name for auxiliary in self.auxiliaries)
        assert len(set(names)) == len(names), f"duplicate auxiliary names: {names}"
        return self


class FaithfulnessLossConfig(TrainingLossConfig):
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
    gradient kept at the single-batch scale so `coeff` transfers between the two modes. Capped
    at `1e6`: a halflife past the run's length already degenerates
    to a debiased running mean, and fp32 rounding drift in the recurrence grows with the
    halflife (pinned by `test_ema_long_scan_rounding_bounded`). While frequencies move
    faster than the halflife the smoothed penalty lags the batch diagnostic (logged
    alongside as `FrequencyMinimalityLoss_batch`) — estimator convergence, not
    instability.
    """

    coeff: NonNegativeFloat | ScheduleConfig
    reference_datapoint_count: PositiveInt | Literal["auto"] = Field(
        validation_alias=AliasChoices("reference_datapoint_count", "reference_token_count")
    )
    """`"auto"` resolves `a'` to each pass's OWN `B·T`, so `a' · f_c` is the raw firing
    count — the torch oracle's form, and the only spelling correct on both tPD passes at
    once: the target and non-target passes share one frequency block, whose `B·T` differ (a
    5-token prompt stream vs a 64-token corpus stream), so any literal is right for one and
    off by their ratio on the other."""
    ema_halflife_steps: PositiveFloat | None = Field(default=None, allow_inf_nan=False, le=1e6)


class ImportanceMinimalityLossConfig(TrainingLossConfig):
    """Smooth-L0 activity and optional frequency penalties on upper-leaky CI values.

    The per-value penalty is `c^2 / (c^2 + gamma^2)` (Geman–McClure), summed over
    per-component mean activity. It approaches the active-component count as
    positive `gamma` anneals toward zero. Its derivative is zero at the origin and bounded by
    `0.65 / gamma`; no epsilon floor or gradient clip is needed. Every gamma
    schedule knot must have `frac > 0`; zero collapses the threshold band.

    `normalize_at_one` multiplies by `1 + gamma^2`, so CI 1 contributes exactly 1
    throughout the schedule. Without it, saturated components contribute more as
    `gamma` decreases. The optional frequency penalty uses the same activities."""

    type: Literal["ImportanceMinimalityLoss"] = "ImportanceMinimalityLoss"
    gamma: ScheduleConfig
    frequency: FrequencyMinimalityConfig | None = None
    normalize_at_one: bool = False


class NonlinearityLocalityLossConfig(TrainingLossConfig):
    """Concentrate each component's write vector on fewer nonlinearity-facing units.

    `relative_threshold` is relative to a uniform unit fraction; annealing it down sharpens
    the soft count. With `normalize_at_one`, the count is rescaled by
    `1 + relative_threshold / unit_count`, so an aligned vector concentrated in one unit contributes
    exactly one use (or the partition's use multiplicity for shared heads) throughout the schedule.
    Otherwise its cost grows as the threshold anneals down. `unit_kind_coefficients` weights
    each unit kind's component mean and must name every kind the target's partitions declare
    (asserted at step build); a `None` entry excludes that kind from the objective outright —
    no reduction is built, which is why weights are strictly positive rather than zero-able.
    """

    type: Literal["NonlinearityLocalityLoss"] = "NonlinearityLocalityLoss"
    coeff: NonNegativeFloat | ScheduleConfig
    relative_threshold: ScheduleConfig
    unit_kind_coefficients: dict[NonlinearityUnitKind, PositiveFloat | None]
    normalize_at_one: bool = False

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


class CIMaskedReconLossConfig(TrainingLossConfig, ReconstructionAuxiliariesMixin):
    type: Literal["CIMaskedReconLoss"] = "CIMaskedReconLoss"


class CIMaskedReconSubsetLossConfig(TrainingLossConfig, ReconstructionAuxiliariesMixin):
    type: Literal["CIMaskedReconSubsetLoss"] = "CIMaskedReconSubsetLoss"
    routing: Annotated[SubsetRoutingType, Field(discriminator="type")] = (
        UniformKSubsetRoutingConfig()
    )


class StochasticReconLossConfig(TrainingLossConfig, ReconstructionAuxiliariesMixin):
    type: Literal["StochasticReconLoss"] = "StochasticReconLoss"


class StochasticReconSubsetLossConfig(TrainingLossConfig, ReconstructionAuxiliariesMixin):
    type: Literal["StochasticReconSubsetLoss"] = "StochasticReconSubsetLoss"
    routing: Annotated[SubsetRoutingType, Field(discriminator="type")] = (
        UniformKSubsetRoutingConfig()
    )


class UnmaskedReconLossConfig(TrainingLossConfig, ReconstructionAuxiliariesMixin):
    type: Literal["UnmaskedReconLoss"] = "UnmaskedReconLoss"


PGDInitStrategy = Literal["random", "ones", "zeroes"]

BatchSourceShape = Literal["bc", "bsc"]
"""Training sources keep the full batch axis: each row owns its adversarial state.

`bc` shares a row's source across positions; `bsc` keeps every position and requires
a positioned target. Component sources also keep C; delta sources omit C.
"""

SourceShape = Literal["c", "bc", "bsc"]
"""Fresh PGD also admits batch-shared `c` sources for evaluation.

Every source has the waist's rank. Omitted axes are size-1 broadcast axes:
positionless `c (1, C)` / `bc (B, C)`; positioned `c (1, 1, C)` /
`bc (B, 1, C)` / `bsc (B, P, C)`.
"""


class PGDConfig(ReconstructionAuxiliariesMixin):
    """Reconstruction and ascent settings shared by training losses and eval probes."""

    step_size: PositiveFloat


class TrainingPGDConfig(PGDConfig, TrainingLossConfig):
    """Fresh training adversaries own independent sources for every batch row."""

    n_steps: NonNegativeInt
    init: PGDInitStrategy
    source_shape: BatchSourceShape


class PGDReconLossConfig(TrainingPGDConfig):
    type: Literal["PGDReconLoss"] = "PGDReconLoss"


class PGDReconSubsetLossConfig(TrainingPGDConfig):
    type: Literal["PGDReconSubsetLoss"] = "PGDReconSubsetLoss"
    routing: Annotated[SubsetRoutingType, Field(discriminator="type")] = (
        UniformKSubsetRoutingConfig()
    )


class EvalPGDConfig(PGDConfig):
    """Fresh evaluation probes optimize randomly initialized, batch-shared sources. They
    carry no weight and no name: the read-out after `n` steps logs as `<type>_<n>step`."""

    init: Literal["random"]
    source_shape: Literal["c"]
    n_batches: PositiveInt | None = None
    """Cap on how many eval batches the probe consumes (None = all `eval.n_steps`). The
    reported loss averages over the batches actually run."""


class EvalPGDReconLossConfig(EvalPGDConfig):
    """A fresh ascent read out once, after `n_steps`."""

    slow: ClassVar[bool] = False
    type: Literal["PGDReconLoss"] = "PGDReconLoss"
    n_steps: NonNegativeInt

    @property
    def read_out_steps(self) -> tuple[int, ...]:
        return (self.n_steps,)


class SlowPGDReconLossConfig(EvalPGDConfig):
    """A ladder: ONE fresh ascent read out after each of `read_out_steps`."""

    slow: ClassVar[bool] = True
    type: Literal["SlowPGDReconLoss"] = "SlowPGDReconLoss"
    read_out_steps: tuple[NonNegativeInt, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_read_out_steps_ascend(self) -> Self:
        steps = self.read_out_steps
        assert all(a < b for a, b in pairwise(steps)), (
            f"SlowPGDReconLoss read_out_steps must strictly ascend, got {steps}"
        )
        return self


type AnyPGDEvalConfig = EvalPGDReconLossConfig | SlowPGDReconLossConfig
type AnyPGDEvalType = Literal["PGDReconLoss", "SlowPGDReconLoss"]


class AdamPGDConfig(BaseConfig):
    """Adam source ascent with persistent first and second moments."""

    type: Literal["adam"] = "adam"
    beta1: Probability = Field(default=0.9, description="Adam beta1 for masks")
    beta2: Probability = Field(default=0.999, description="Adam beta2 for masks")
    eps: NonNegativeFloat = Field(default=1e-8, description="Adam epsilon for masks")
    lr_schedule: ScheduleConfig


class SgdPGDConfig(BaseConfig):
    """Projected SGD source ascent without optimizer buffers.

    Omitting moment buffers keeps per-datapoint `bsc` sources affordable at large
    component counts."""

    type: Literal["sgd"] = "sgd"
    lr_schedule: ScheduleConfig


class MomentumSgdPGDConfig(BaseConfig):
    """Projected momentum-SGD source ascent with one persistent velocity buffer.

    `v = momentum * v + grad; sources += lr * v`, followed by clipping to [0, 1].
    Velocity uses `adversary.velocity_dtype`: bf16 for either 16-bit source format,
    fp32 for fp32 sources."""

    type: Literal["momentum_sgd"] = "momentum_sgd"
    momentum: Probability
    lr_schedule: ScheduleConfig


class PersistentAdversaryLossConfig(TrainingLossConfig, ReconstructionAuxiliariesMixin):
    """Shared optimizer and lifecycle fields for persistent adversarial sources."""

    optimizer: Annotated[
        AdamPGDConfig | SgdPGDConfig | MomentumSgdPGDConfig, Field(discriminator="type")
    ]
    source_dtype: Literal["float32", "bfloat16", "uint16"] = "float32"
    """Storage representation for the persistent PPGD source VALUES; adam moments follow
    it, the momentum velocity takes its float sibling (`adversary.velocity_dtype`).
    `float32` (default) keeps source optimizer moments in fp32 and is the oracle-parity
    path. `bfloat16` halves the resident footprint at ~2^-8 resolution near 1.0.
    `uint16` is the unit-interval FIXED-POINT representation (`value = u/65535`,
    `adversary.UINT16_UNIT_SCALE`): same bytes as bf16, uniform 1/65535 resolution across
    [0,1]; ascents update the fp32 view and store back with stochastic rounding
    (unbiased — round-to-nearest would permanently stall sub-half-step updates). uint16
    pairs with the float-buffered or stateless source optimizers; adam refuses it (its moments
    follow the storage dtype, and integer moments are meaningless)."""

    @model_validator(mode="after")
    def validate_storage_supports_optimizer(self) -> Self:
        if self.source_dtype == "uint16" and isinstance(self.optimizer, AdamPGDConfig):
            raise ValueError(
                "uint16 fixed-point source storage pairs with source optimizer sgd/momentum_sgd; "
                "adam stores its moments at the source dtype"
            )
        return self

    n_warmup_steps: NonNegativeInt = Field(
        default=0,
        description=(
            "Extra inner PGD source-optimization steps on each train batch before the final loss"
            " computation."
        ),
    )
    adversary_objective: Literal["term", "e2e"] = "e2e"
    """Objective the persistent sources ascend. `e2e` excludes reconstruction
    auxiliaries from source ascents while keeping them in the outer components/CI
    objective; `term` makes the sources ascend the complete loss."""


class PersistentPGDLossConfig(PersistentAdversaryLossConfig):
    """A persistent adversary with independent state for every batch row."""

    source_shape: BatchSourceShape


class PersistentPGDReconLossConfig(PersistentPGDLossConfig):
    """Persistent-PGD recon loss: the adversary's masks routed to all sites every
    forward."""

    type: Literal["PersistentPGDReconLoss"] = "PersistentPGDReconLoss"


class SourcePoolConfig(BaseConfig):
    """Each batch element owns this many persistent cross-site particles."""

    size_per_batch_element: PositiveInt


class MergedStochasticSubsetPooledPPGDReconLossConfig(PersistentAdversaryLossConfig):
    """The merged stochastic/PPGD term with persistent cross-site source particles.

    Each batch element samples one particle from its configured pool and broadcasts it over
    positions. Warmup ascents, the main forward, and the source-gradient retake share
    the sampled indices. Each site stores ``batch_size * pool.size_per_batch_element``
    particles, each containing a component vector and a delta scalar.
    """

    type: Literal["MergedStochasticSubsetPooledPPGDReconLoss"] = (
        "MergedStochasticSubsetPooledPPGDReconLoss"
    )
    pool: SourcePoolConfig
    adv_fraction: ScheduleConfig
    routing: SubsetRoutingType = Field(default_factory=UniformKSubsetRoutingConfig)

    @model_validator(mode="after")
    def validate_adv_fraction_is_probability(self) -> Self:
        if self.adv_fraction.max_val > 1.0:
            raise ValueError(f"adv_fraction must stay within [0, 1]: {self.adv_fraction}")
        return self


class MergedStochasticSubsetPPGDReconLossConfig(PersistentPGDLossConfig):
    """Combine persistent-adversarial and stochastic reconstruction in one forward.

    Each batch element uses persistent sources at every site with probability
    `adv_fraction`; otherwise it uses fresh uniform sources and the configured
    routing. The choice is shared across its sequence. `adv_fraction` may be
    scheduled, and `coeff` weights the combined loss: coefficient 1 with an
    adversarial fraction of 0.5 matches a 0.5-weighted pair in expectation.
    The persistent sources receive their final ascent gradient from this forward."""

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


class CIMaskedReconEvalConfig(BaseConfig):
    """The CI-masked (constant-source-0) KL probe, per stream and CI role. In a dual run it
    is what measures the hidden head's arm; `CEandKLLosses` already emits the output arm
    under the same `ce_kl/kl_ci_masked` key."""

    slow: ClassVar[bool] = False
    type: Literal["CIMaskedReconLoss"] = "CIMaskedReconLoss"


class UnmaskedNoDeltaReconEvalConfig(BaseConfig):
    """The unmasked-no-delta KL probe on the optimized stream (the non-target pass's own
    training term, measured on eval batches)."""

    slow: ClassVar[bool] = False
    type: Literal["UnmaskedNoDeltaReconLoss"] = "UnmaskedNoDeltaReconLoss"


class WeightMagnitudeConfig(BaseConfig):
    """Per-site `‖V_c‖·‖U_c‖` scatter in descending magnitude order, log y.

    Reads the trained V/U alone — no forward pass and no eval batch."""

    slow: ClassVar[bool] = True
    type: Literal["WeightMagnitude"] = "WeightMagnitude"


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
    ci_alive_threshold: Probability = 0.0


class CIMeanPerComponentConfig(BaseConfig):
    slow: ClassVar[bool] = True
    type: Literal["CIMeanPerComponent"] = "CIMeanPerComponent"


class ComponentActivationDensityConfig(BaseConfig):
    slow: ClassVar[bool] = True
    type: Literal["ComponentActivationDensity"] = "ComponentActivationDensity"
    ci_alive_threshold: Probability = 0.0


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
    """Muon: Newton-Schulz-orthogonalized momentum for the group's matrix leaves; the rest
    fall back to Adam(0.9, 0.999) at the same LR. Experimental (non-canonical). The
    semantics are `optax.contrib.muon`'s; the NS runs batched per semantic
    kind at the placement table's `ns_compute` waypoint (`muon_stacked.py`). Which leaves
    are matrices is per-group (`run_state.build_optimizers`):
    the V/U components tree is all-2D (fallback never fires); the chunkwise CI fn is
    per-chunk stacks, so its 3D leaves are muon'd over the trailing two axes (chunk axis
    batched) and its 2D bias stacks take the fallback; the MLP CI fns use the plain 2D rule."""

    type: Literal["muon"]
    lr_schedule: ScheduleConfig = Field(..., description="Learning rate schedule")
    beta: Probability = Field(
        default=0.95, description="Momentum decay for the orthogonalized update"
    )
    # 0.2 approximately matches AdamW's empirical update RMS, following Moonshot Muon
    # (https://arxiv.org/abs/2502.16982), so matrix learning rates use a comparable scale.
    consistent_rms: PositiveFloat | None = Field(
        default=0.2,
        description=(
            "If set, scale updates by `sqrt(max(fan_in, fan_out)) * consistent_rms` for"
            " approximately shape-independent update RMS. If None, use optax's width"
            " scaling `sqrt(max(1, fan_out / fan_in))`."
        ),
    )
    weight_decay: NonNegativeFloat = Field(default=0.0, description="Weight decay")
    grad_clip_norm: PositiveFloat | None = Field(
        default=None,
        description="If set, clip the grad norm of this group's parameters to this value",
    )
    ns_steps: PositiveInt = Field(
        default=5, description="Newton-Schulz iterations (optax default 5; fewer = cheaper/looser)"
    )
    ns_dtype: Literal["float32", "bfloat16"] = Field(
        default="bfloat16",
        description="Dtype of the NS orthogonalization only; masters and momentum stay float32.",
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
    | MergedStochasticSubsetPooledPPGDReconLossConfig
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
"""Trainable losses; hidden-activation reconstruction is a reconstruction auxiliary,
not a standalone loss."""


TargetedLossMetricConfig = Annotated[
    AnyReconLossMetricConfig | ImportanceMinimalityLossConfig | NonlinearityLocalityLossConfig,
    Discriminator("type"),
]
"""Target-pass losses: reconstruction (including adversaries), importance-minimality, and
the optional nonlinearity-locality prior. Faithfulness is excluded so the delta can carry
off-target behavior.

The nonlinearity term is admitted here even though it belongs to no PASS (local
amendment): it is a weight-space prior on `U` alone — CI-ungated, data-independent — so it
is scored once against the components and added once to the step's total, exactly as in
plain PD. It is authored on the target list because that is the tPD config's one loss
list, not because the target stream scores it. With no faithfulness role the weight delta
is unpenalized, so concentration pressure on `U` has an escape valve plain PD does not give
it — expect a weaker effect per unit coefficient than plain-PD runs."""


class UnmaskedNoDeltaReconLossConfig(TrainingLossConfig):
    """Reconstruct non-target outputs using the full component sum without the delta.

    Component masks are 1 and there is no weight delta. This constrains even
    components that never activate, so they cannot interfere with reconstruction."""

    type: Literal["UnmaskedNoDeltaReconLoss"] = "UnmaskedNoDeltaReconLoss"


NontargetReconLossMetricConfig = (
    CIMaskedReconLossConfig
    | CIMaskedReconSubsetLossConfig
    | StochasticReconLossConfig
    | StochasticReconSubsetLossConfig
    | UnmaskedNoDeltaReconLossConfig
)
"""The CLOSED non-target reconstruction set: stochastic or constant component sources with
the delta on, or the full component sum with the delta off. The non-target HIDDEN pass
admits this set plus the merged term; the OUTPUT pass's union below additionally admits
the persistent adversarial types (local amendments 2026-08-19/20)."""


NontargetOutputReconLossMetricConfig = (
    NontargetReconLossMetricConfig
    | PersistentPGDReconLossConfig
    | MergedStochasticSubsetPPGDReconLossConfig
)
"""The non-target OUTPUT pass's vocabulary: the closed non-target set plus the two
persistent-source-carrying recon types. Their masks compose from the bundle's COMPONENT
channels with every weight-delta mask pinned to 1.0 (the source's delta channel is
ignored), so the probed worst case is a partial ablation of components on data they should
be inactive on. Each such term gets its own bundle, sized off the BROAD stream's geometry."""


NontargetHiddenReconLossMetricConfig = (
    NontargetReconLossMetricConfig | MergedStochasticSubsetPPGDReconLossConfig
)
"""The non-target HIDDEN pass's vocabulary: the closed set plus the MERGED term ONLY — the
one adversarial type whose masked forward REPLACES a stochastic term's, so admitting it
costs zero extra forwards on the broad stream. The standalone persistent-PGD type stays
refused here: it would add a masked forward per step. Delta pinned like every non-target
forward; own bundle per term, sized off the BROAD stream's geometry."""


HiddenReconLossMetricConfig = ReconLossMetricConfigs
"""The recon types the hidden pass admits on the TARGET stream: the same full vocabulary as
the output pass, adversaries included — the hidden objective is an ordinary reconstruction
objective measured at internal activations rather than at the output."""


class HiddenPassConfig(BaseConfig):
    """The hidden role's surface: a second reconstruction objective, scored at named
    internal activations instead of the model output, whose masks come from the CI fn's
    SECOND readout head.

    `points` lives HERE rather than on each term — the pass IS "reconstruct these
    activations". It is shared with the non-target stream's hidden surface, the way one
    importance-minimality config is shared across passes: WHAT is being reconstructed is a
    property of the experiment, not of a stream. Only coefficients differ per pass.

    Requires `decomposition.ci.dual` — without a second head there is no hidden CI to shape."""

    points: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description=(
            "Activations the hidden objective reconstructs, in the TARGET's own capture "
            "vocabulary — e.g. `layers.19.mlp.down_proj.out` for block 19's MLP output. Each "
            "point's error is normalized by its own clean squared scale (see `normalization`) "
            "and the pass takes the MEAN over points, so each point's weight is 1/P: adding "
            "points dilutes the others unless the term coefficients are rescaled."
        ),
    )
    normalization: HiddenActsNormalization = Field(
        default_factory=BatchHiddenActsNormalization,
        description=(
            "How each point's relative squared error pools over positions. `batch`: one ratio "
            "over the whole batch — a position holding most of the point's clean energy "
            "(Llama's BOS massive activation at layers 0, 1, 31) IS the measurement. "
            "`per_position`: each (batch, position) entry against its own clean scale, then "
            "the mean over entries. Shared with the non-target hidden pass, like `points`."
        ),
    )
    impmin_coeff: NonNegativeFloat | ScheduleConfig = Field(
        ...,
        description=(
            "The hidden head's importance-minimality COEFFICIENT. The penalty's shape and "
            "anneal are the target pass's, shared by construction — only the strength is "
            "authored here."
        ),
    )
    recon: list[Annotated[HiddenReconLossMetricConfig, Discriminator("type")]] = Field(
        ..., min_length=1
    )
    output_coeff: LossCoeff | None = Field(
        default=None,
        description=(
            "OPTIONAL end-to-end KL rider: each hidden term scores "
            "`mean_points(hidden error) + output_coeff * kl_per_position` on the SAME masked "
            "forward, so the hidden head is also asked to keep the components the model "
            "output needs. `None` (default) is the pure hidden pass: no e2e term, "
            "`recon_loss_fn` never called on a hidden forward. Its adversaries ascend the "
            "complete objective including the rider."
        ),
    )
    log_alternate_normalization: bool = Field(
        default=False,
        description=(
            "DIAGNOSTIC ONLY: also score every point under the OTHER `normalization` and log "
            "it beside the real one (`<point>__altnorm`), so the ratio between the two can be "
            "read off a live run. Merged after the pass total, so the loss is untouched. "
            "Shared with the non-target hidden pass, like `points`."
        ),
    )

    @model_validator(mode="after")
    def validate_hidden_entries(self) -> Self:
        seen: set[str] = set()
        assert len(set(self.points)) == len(self.points), f"duplicate points: {self.points}"
        for cfg in self.recon:
            name = cfg.name if cfg.name is not None else cfg.type
            assert name not in seen, f"duplicate hidden loss {name!r}"
            seen.add(name)
            assert not cfg.auxiliaries, (
                f"hidden.recon {cfg.type!r}: reconstruction auxiliaries have no place on a "
                "hidden-pass term — the pass already scores exactly `hidden.points`"
            )
        return self


class NontargetHiddenConfig(BaseConfig):
    """The hidden role on the NON-TARGET stream: the non-target pass's twin with its own
    coefficients, the closed non-target recon vocabulary plus the zero-extra-forward MERGED
    adversarial term, and the measurement points of the target-stream `hidden:` block.
    Delta pinned fully on, as every non-target forward is."""

    impmin_coeff: NonNegativeFloat | ScheduleConfig
    recon: list[Annotated[NontargetHiddenReconLossMetricConfig, Discriminator("type")]] = Field(
        ..., min_length=1
    )
    output_coeff: LossCoeff | None = Field(
        default=None,
        description=(
            "OPTIONAL end-to-end KL rider on the non-target hidden pass, as `pd.hidden."
            "output_coeff`. Costs the full-vocabulary logits and their backward on every "
            "broad-stream hidden forward."
        ),
    )

    @model_validator(mode="after")
    def validate_nontarget_hidden_entries(self) -> Self:
        seen: set[str] = set()
        for cfg in self.recon:
            name = cfg.name if cfg.name is not None else cfg.type
            assert name not in seen, f"duplicate non-target hidden loss {name!r}"
            seen.add(name)
            if not isinstance(cfg, UnmaskedNoDeltaReconLossConfig):
                assert not cfg.auxiliaries, (
                    f"nontarget.hidden.recon {cfg.type!r}: reconstruction auxiliaries have no "
                    "place here — the hidden pass already scores `hidden.points`"
                )
        return self


class NontargetConfig(BaseConfig):
    """The tPD non-target pass, authored directly — never derived from the
    target pass's loss list.

    `batch_size` is the broad stream's GLOBAL batch; `pd.batch_size` stays the target
    stream's. A non-target OUTPUT recon term may carry a persistent adversary, whose
    bundle sizes off THIS pass's geometry — `batch_size` x the broad dataset's own seq
    len — never `pd.batch_size`. `impmin_coeff` is the non-target
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
            "The hidden role on this stream. Requires `pd.hidden`, whose `points` it measures "
            "at. None (the default) leaves the hidden head shaped by the target stream alone."
        ),
    )

    @model_validator(mode="after")
    def validate_nontarget_entries(self) -> Self:
        """Per-entry facts the shared recon classes can spell but the non-target pass
        refuses — caught at parse (config authoring and validation), not at objective
        build on the GPUs."""
        seen: set[str] = set()
        for cfg in self.recon:
            name = cfg.name if cfg.name is not None else cfg.type
            assert name not in seen, f"duplicate non-target loss {name!r}"
            seen.add(name)
            if not isinstance(cfg, UnmaskedNoDeltaReconLossConfig):
                assert not cfg.auxiliaries, (
                    f"nontarget.recon {cfg.type!r}: auxiliary reconstruction is target-pass-only "
                    "— with the delta pinned on, internal-activation matching would "
                    "constrain the behavior tPD declines to decompose"
                )
        return self


RuleConfig = dict[SemanticAxis, MeshAxis | list[MeshAxis] | None]
"""One placement row as configured: semantic axis name -> mesh axis, ordered mesh axes,
or null (replicate). The three value spellings are authoring sugar for ONE in-code value
(`axes.MeshAssignment`, an ordered tuple; `null` = `()`), folded at the placement parse
boundary. A `null` entry is a LISTED axis — unlike an omitted one it still meets the
row-key check. Both name vocabularies are the closed `axes.py` Literals, so an axis
name outside them is refused at parse; rules construction additionally fails closed on
any in-vocabulary key no tensor consumes at that row (a typo'd axis would silently
replicate) and on a mesh whose axes are not the table's vocabulary."""


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


PlacementPresetName = Literal[
    "owner",
    "zero1",
    "zero1-replicated-resident",
    "owner-replicated-resident",
    "zero1-replicated-resident-moe",
    "owner-replicated-resident-moe",
    "zero1-replicated-resident-moe-replicated-ns",
    "ddp",
]
"""The built-in placement tables by name (`placement.PRESETS` holds the tables and asserts
it enumerates exactly these names). Each CI architecture maps these same names to its own
rows."""


class PlacementTableConfig(BaseConfig):
    """An explicit placement table (`runtime.sharding`), mirroring the typed
    `placement.PlacementRules`. The row vocabulary is CLOSED (extra keys are a parse
    error); rule values are free-form axis-name -> mesh-axes mappings, where YAML list
    order is semantics (nested-axis linearization — PLACEMENT_DESIGN.md invariant 5).
    CI rows are the CI architecture's own: `ci_fn` names the preset it takes them from."""

    components: ComponentsPlacementConfig
    ci_fn: PlacementPresetName
    activations: ActivationsPlacementConfig
    target: TargetPlacementConfig


PlacementSpec = PlacementPresetName | PlacementTableConfig
"""What a run or consumer writes down as its placement: a preset name or an explicit table."""


SequenceSharding = Literal["replicate", "sequence_parallel"]
"""How masked forwards place the between-blocks residual over tp. `replicate` keeps the
position axis replicated (the standing spelling). `sequence_parallel` shards the position
axis over tp between the blocks of masked forwards; block interiors run at full width,
so each block boundary's tp-axis activation reduction becomes a reduce-scatter +
all-gather pair at 1/tp resident width. Clean forwards, CI-fn taps and the output edge
keep the replicated residual. Sequence length must tile tp, and only targets implementing
it accept it (currently qwen36_moe; others refuse at their masked forward)."""


class HsdpMeshShape(BaseConfig):
    """The three-axis `(replicate, fsdp, tp)` logical mesh every non-resident placement
    runs on — the `runtime.mesh:` value of those runs."""

    replicate: PositiveInt = Field(
        description="Logical model-replication and cross-replica ownership axis.",
    )
    fsdp: PositiveInt = Field(
        description="Logical parameter-sharding axis; also shards the data batch.",
    )
    tp: PositiveInt = Field(
        description="Logical Megatron tensor-parallel axis; batch activations replicate over it.",
    )

    @property
    def world_size(self) -> int:
        return self.replicate * self.fsdp * self.tp


class ResidentMeshShape(BaseConfig):
    """The two-axis `(data, tp)` logical mesh of the `*-replicated-resident` placements —
    the whole ÷tp working copy is resident, so no fsdp axis exists to spell."""

    data: PositiveInt = Field(
        description="Logical data-parallel axis: shards the batch and the ÷N optimizer masters.",
    )
    tp: PositiveInt = Field(
        description="Logical Megatron tensor-parallel axis; batch activations replicate over it.",
    )

    @property
    def world_size(self) -> int:
        return self.data * self.tp


MeshShape = HsdpMeshShape | ResidentMeshShape
"""Structurally discriminated: the two shapes' required keys are disjoint and extra keys
refuse (`extra="forbid"`), so a `runtime.mesh:` mapping parses to exactly one arm."""


class NamedNeuronRanks(BaseConfig):
    """A neuron-ranking artifact in the store: `<data_root>/neuron_ranks/<name>` — what the
    LM `initialization: neuron_aligned_targeted` starts from. Names are immutable versions —
    a re-harvest is a new name."""

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
    loss vocabulary + the faithfulness warmup) and `TargetedPDConfig` (tPD, the
    faithfulness-free loss vocabulary, no warmup fields at all — a targeted config cannot
    SPELL a faithfulness role). Pair with `Cadence` (when to emit) when running the
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
            "run this is the TARGET stream's batch; the broad stream's lives on "
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


type WeightDecayStreams = Literal["target", "nontarget", "all"]
"""Which streams' CI the CI-scaled decay reads (`TargetedPDConfig.ci_scaled_weight_decay_stream`)."""

type WeightDecayRoles = Literal["output", "hidden", "all"]
"""Which CI READOUT HEAD the CI-scaled decay reads (`TargetedPDConfig.ci_scaled_weight_decay_role`):
`output` the end-to-end head every run carries, `hidden` the internal-activations head
(present only when the run carries a hidden pass)."""


class TargetedPDConfig(PDConfigBase):
    """Targeted-PD losses and updates, without faithfulness or faithfulness warmup.

    The weight delta must remain free to carry off-target behavior; these fields
    are absent rather than accepted at zero. `batch_size` sizes the target stream."""

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
            "The hidden role's TARGET-stream surface: a second reconstruction objective "
            "scored at internal activations, whose masks come from the CI fn's second "
            "readout head. Requires `decomposition.ci.dual`. None (the default) is an "
            "ordinary single-objective tPD run, byte-inert."
        ),
    )

    ci_scaled_weight_decay: PositiveFloat | None = Field(
        default=None,
        description=(
            "CI-scaled weight decay on the subcomponent V/U vectors: after each "
            "optimizer step every subcomponent's V column and U row scale by "
            "`1 - lr*wd*(1 - max CI)`, the max taken over every pass's batch by default — "
            "both streams and both CI heads — so dead components, never important anywhere, "
            "get dragged to zero. None disables (the default; the term is an optional "
            "auxiliary). The three `ci_scaled_weight_decay_*` fields below narrow WHICH CI "
            "that max reads."
        ),
    )
    ci_scaled_weight_decay_stream: WeightDecayStreams = Field(
        default="all",
        description=(
            "Which streams' CI counts as evidence a component is alive: `all` (the default, "
            "the original rule), `target` (the narrow stream only — a component load-bearing "
            "ONLY on the broad stream is then decayed away), or `nontarget`. Only meaningful "
            "with `ci_scaled_weight_decay` set."
        ),
    )
    ci_scaled_weight_decay_role: WeightDecayRoles = Field(
        default="all",
        description=(
            "Which CI READOUT HEAD the decay reads: `all` (the default, the original rule), "
            "`output` (the end-to-end reconstruction head only — components that only ever "
            "matter to the hidden-activations objective are then decayed away), or `hidden` "
            "(the internal-activations head only, which requires a hidden pass). Only "
            "meaningful with `ci_scaled_weight_decay` set."
        ),
    )
    ci_scaled_weight_decay_quantile: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description=(
            "The per-component batch statistic the decay reads: 1.0 (the default) is the MAX "
            "— the original rule, byte-inert and gather-free — and a smaller value takes that "
            "quantile over the pooled batch and position axes instead, so a component alive "
            "on only a few points no longer fully escapes the decay. Applied per selected "
            "pass; the passes still combine by max. Only meaningful with "
            "`ci_scaled_weight_decay` set."
        ),
    )

    @model_validator(mode="after")
    def validate_loss_metrics(self) -> Self:
        _validate_training_losses(self.loss_metrics)
        return self

    @model_validator(mode="after")
    def validate_ci_scaled_weight_decay(self) -> Self:
        """The selectors describe a decay; without one they would be a config that reads as
        if it changed something. Fail closed rather than ignoring them."""
        if self.ci_scaled_weight_decay is None:
            assert (
                self.ci_scaled_weight_decay_stream == "all"
                and self.ci_scaled_weight_decay_role == "all"
                and self.ci_scaled_weight_decay_quantile == 1.0
            ), (
                "ci_scaled_weight_decay_{stream,role,quantile} select which CI the CI-scaled "
                "weight decay reads; they need `ci_scaled_weight_decay` set to mean anything"
            )
        return self

    @model_validator(mode="after")
    def validate_hidden_pass(self) -> Self:
        """Identities are unique ACROSS passes: every term's log key is `loss/<name>`, and
        persistent terms key their adversary state by name, so a hidden term sharing a target
        term's default name (its type literal) would silently overwrite its curve and collide
        in the adversary table. Refuse at parse rather than lose a curve."""
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
    identities unique (`name or type` — the logged instance key),
    exactly one importance-minimality term, at least one recon term, and at most one
    nonlinearity term. Faithfulness multiplicity is the plain shape's own claim
    (the targeted union has no member)."""
    seen: set[str] = set()
    for cfg in loss_metrics:
        if isinstance(cfg, ReconstructionAuxiliariesMixin):
            for auxiliary in cfg.auxiliaries:
                if isinstance(auxiliary.coeff, float):
                    assert auxiliary.coeff > 0.0, (
                        f"loss_metrics.{cfg.type!r}.auxiliaries.{auxiliary.name}.coeff must be positive; "
                        "zero is reserved for eval-only measurement"
                    )
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
    shape. Under `keep_last` retention the retained window always contains the
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

    train_log_every: PositiveInt = 20
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
    metric_schema: MetricSchema = "legacy"
    """W&B key layout, fixed for the lifetime of a run. Local JSONL keys stay unchanged."""
    entity: str | None = None
    group: str | None = None
    """Wandb UI group; None = ungrouped."""
    tags: list[str] = Field(default_factory=list)
    """Wandb tags; empty = untagged."""


class ResumeProvenance(BaseConfig):
    """Fine-tune lineage: a fresh run initialized from a PARENT decomposition. Lives on
    `ExperimentConfig`.

    A fine-tune run gets its own `run_id` / `launch_config.yaml` / `ckpts/`; this records the
    parent it forked from. The JAX trainer loads the parent checkpoint's
    V/U + ci_fn onto a fresh reference state and trains a clean schedule from step 0
    (fresh optimizer / sources) under the new config — only when the run's own `ckpts/`
    is empty (a subsequent restart resumes from the run's own dir, ignoring
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
    "freq": "FrequencyMinimalityLoss",
    "freq_batch": "FrequencyMinimalityLoss_batch",
}
"""Step-record short key -> logged loss name, for the terms whose record key is not
already the term's class name. Shared so the target stream (expanded by the run loop) and
the other passes (keyed in `train.make_targeted_train_step`) cannot drift apart: every
pass must spell the same quantity the same way."""


METRIC_SHORT_NAMES: dict[str, str] = {
    "CIMaskedReconLoss": "CIMaskRecon",
    "CIMaskedReconSubsetLoss": "CIMaskReconSub",
    "FaithfulnessLoss": "Faith",
    "ImportanceMinimalityLoss": "ImpMin",
    "NonlinearityLocalityLoss": "Nonlinearity",
    "PersistentPGDReconLoss": "PersistPGDRecon",
    "PGDReconLoss": "PGDRecon",
    "PGDReconSubsetLoss": "PGDReconSub",
    "StochasticReconLoss": "StochRecon",
    "StochasticReconSubsetLoss": "StochReconSub",
    "UnmaskedReconLoss": "UnmaskedRecon",
    "ArithmeticCIGrid": "ArithCIGrid",
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
