"""Authored metric schemas whose semantics require an LM target."""

from collections.abc import Sequence
from typing import Annotated, ClassVar, Literal, Self

from pydantic import (
    Discriminator,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from param_decomp.core.base_config import BaseConfig, Probability
from param_decomp.core.configs import (
    AnyLossMetricConfig,
    MergedStochasticSubsetPPGDReconLossConfig,
    PersistentPGDReconLossConfig,
    ReconstructionAuxiliariesMixin,
    TargetedLossMetricConfig,
)


class CEandKLLossesConfig(BaseConfig):
    """Token-level CE/KL metrics for categorical LM outputs."""

    slow: ClassVar[bool] = False
    type: Literal["CEandKLLosses"] = "CEandKLLosses"
    rounding_threshold: Probability


class CIActiveCountsPerPositionConfig(BaseConfig):
    """`CI_L0`'s active-component count at every position of the eval window, summed over
    all sites, for CI > 0, > 0.01 and > 0.1: one native W&B line chart, `eval/l0/per_position`."""

    slow: ClassVar[bool] = True
    type: Literal["CIActiveCountsPerPosition"] = "CIActiveCountsPerPosition"


class CIMaskedAttnPatternsReconLossConfig(BaseConfig):
    """Materializes [B, H, T, T] attention maps even when the forward uses flash."""

    slow: ClassVar[bool] = False
    type: Literal["CIMaskedAttnPatternsReconLoss"] = "CIMaskedAttnPatternsReconLoss"


class StochasticAttnPatternsReconLossConfig(BaseConfig):
    """Materializes attention maps for each mask sample; memory grows quadratically in T."""

    slow: ClassVar[bool] = False
    type: Literal["StochasticAttnPatternsReconLoss"] = "StochasticAttnPatternsReconLoss"


class CIMaskedStrategy(BaseConfig):
    """Masks = the CI envelope's lower values; no weight-delta correction."""

    kind: Literal["ci_masked"] = "ci_masked"


class StochasticStrategy(BaseConfig):
    """One draw of `ci + (1 - ci)·U[0,1]` with random weight-delta masks."""

    kind: Literal["stochastic"] = "stochastic"


class FreshPGDStrategy(BaseConfig):
    """Fresh sign-PGD sources (random init, one source per component shared by every
    token) ascended `n_steps` times against the output KL, composed with the CI."""

    kind: Literal["fresh_pgd"] = "fresh_pgd"
    n_steps: NonNegativeInt
    step_size: PositiveFloat


class PersistentStrategy(BaseConfig):
    """The run's persistent-PGD adversary sources, composed with the CI. `state_key` is
    the `name` (instance key) of the persistent training term whose sources are read. A
    `c`/`sc` source applies to the eval batch as it is; a `bc`/`bsc` source's rows index
    training samples, so each eval sequence draws one row (`masking.sample_source_rows`)
    — in expectation the eval batch sees the adversary the training batch saw."""

    kind: Literal["persistent"] = "persistent"
    state_key: str


RouterDivergenceStrategy = Annotated[
    CIMaskedStrategy | StochasticStrategy | FreshPGDStrategy | PersistentStrategy,
    Discriminator("kind"),
]


class RouterDivergenceConfig(BaseConfig):
    """Per MoE layer, how far the masked model's expert router drifts from the target's:
    the target's router applied to the masked residual, read out at every layer under the
    target's own expert selection pinned everywhere, against the target's routing at that
    layer. Masking and mixing-weight changes can still propagate through the residual.
    Each strategy's masks run their own masked forwards per batch; `kind` names the strategy in the log keys, so each kind
    appears once."""

    slow: ClassVar[bool] = False
    type: Literal["RouterDivergence"] = "RouterDivergence"
    strategies: tuple[RouterDivergenceStrategy, ...]

    @model_validator(mode="after")
    def validate_strategies(self) -> Self:
        kinds = [strategy.kind for strategy in self.strategies]
        assert kinds, "RouterDivergence needs at least one strategy"
        assert len(kinds) == len(set(kinds)), (
            f"RouterDivergence strategies repeat a kind: {kinds} — a strategy is logged under "
            "its kind, so each kind appears once"
        )
        return self


def assert_router_divergence_persistent_terms_exist(
    metric: RouterDivergenceConfig,
    loss_metrics: Sequence[AnyLossMetricConfig | TargetedLossMetricConfig],
) -> None:
    """Every `persistent` strategy names one of the run's persistent-PGD training terms
    (by instance key, `name or type` — the key its adversary lives under in the training
    state)."""
    persistent_terms = {
        (term.name if term.name is not None else term.type)
        for term in loss_metrics
        if isinstance(
            term, (PersistentPGDReconLossConfig, MergedStochasticSubsetPPGDReconLossConfig)
        )
    }
    for strategy in metric.strategies:
        match strategy:
            case PersistentStrategy(state_key=state_key):
                assert state_key in persistent_terms, (
                    f"RouterDivergence persistent strategy names state_key {state_key!r}, but "
                    f"the run's persistent-PGD terms are {sorted(persistent_terms)}"
                )
            case CIMaskedStrategy() | StochasticStrategy() | FreshPGDStrategy():
                pass


class WellTemperednessConfig(BaseConfig):
    """Whether higher causal importance preactivations mean greater ablation effects.

    Components are ablated one at a time at sampled token positions of an LM. `groups`
    maps names to fnmatch-style site patterns. Every region always schedules
    `n_locations * n_components_per_region` solo ablations: a sparse region pads its quota
    with out-of-region components whose damage is computed and discarded.
    """

    slow: ClassVar[bool] = True
    type: Literal["WellTemperedness"] = "WellTemperedness"
    groups: dict[str, list[str]] | None
    n_locations: PositiveInt
    n_components_per_region: PositiveInt
    ablations_per_forward: PositiveInt


class ArithmeticCEKLConfig(BaseConfig):
    rounding_threshold: Probability


class ArithmeticCIL0Config(BaseConfig):
    ci_alive_threshold: Probability
    groups: dict[str, list[str]] | None


class ArithmeticFreshPGDConfig(ReconstructionAuxiliariesMixin):
    n_steps: NonNegativeInt
    step_size: PositiveFloat


class ArithmeticProbeMetrics(BaseConfig):
    """Scalar operations evaluated on the arithmetic grid rather than corpus batches."""

    ce_kl: ArithmeticCEKLConfig
    ci_l0: ArithmeticCIL0Config
    fresh_pgd: ArithmeticFreshPGDConfig | None


class ArithmeticCIGridConfig(BaseConfig):
    """Per-component causal-importance heatmaps over an arithmetic operand grid."""

    slow: ClassVar[bool] = True
    type: Literal["ArithmeticCIGrid"] = "ArithmeticCIGrid"
    probe_metrics: ArithmeticProbeMetrics
    operation: Literal["add", "sub", "mul"] = "add"
    a_range: tuple[int, int] = (1, 100)
    b_range: tuple[int, int] = (1, 100)
    thresholds: list[Probability] = Field(default_factory=lambda: [0.1])
    top_k: PositiveInt = 24


class ABGridDatasetConfig(BaseConfig):
    """Per-component CI + inner-activation snapshots over an arithmetic operand grid,
    written under the run dir with the applet that browses them.

    Never runs on the first eval pass, whatever `eval.slow_on_first_step` says — an
    untrained decomposition saves every component and shows nothing
    (`experiments.eval_config.schedule_for`)."""

    slow: ClassVar[bool] = True
    type: Literal["ABGridDataset"] = "ABGridDataset"
    mean_ci_floor: Probability
    """Full `a x b` grids are stored only for components whose prompt-mean CI reaches this
    at SOME recorded position; the mean-CI vector is stored for every component, so the cut
    stays visible in the applet. It is the ONLY bound on snapshot size — ~40 KB per saved
    component per recorded position at a 100x100 grid (u8 CI + f16 inner, base64), scaling
    with the cell count — so read `eval/ab_grids/saved_components/total` before trusting a
    floor at production C. A grid past 100 on either side is written split (thumbnail index +
    one full-resolution file per module; `ab_grid_dataset.THUMBNAIL_MAX_SIDE`)."""
    operation: Literal["add", "sub", "mul"] = "add"
    a_range: tuple[int, int] = (1, 100)
    b_range: tuple[int, int] = (1, 100)
    positions: list[int] | None = None
    """Prompt positions to record (negatives allowed). `null` records the answer (last)
    position alone. Each extra position multiplies both the gathered columns and the
    snapshot on disk."""
    chunk_prompts: PositiveInt | None = None
    """Rows per forward through the grid. `null` sends all `|a_range| x |b_range|` prompts
    in ONE forward — 10000 prompts needs ~20GiB and OOMs a 45GB L40. Chunking bounds the
    forward without shrinking the grid (the collector forwards each chunk twice — sums, then
    saved columns — so only one chunk's grids are ever on device); the CI / inner columns are
    concatenated across chunks and the mean CI sums over them, so WHICH components are saved
    is chunk-count-invariant and their values match the unchunked pass up to float
    reassociation (SPEC D4). Prefer a multiple of the device count so no chunk carries
    sharding pad."""


class TwoStreamCIMeanPerComponentConfig(BaseConfig):
    """Both streams' mean CI per component on ONE axis per site, ordered by descending
    TARGET mean and coloured by stream.

    Computes `CIMeanPerComponent`'s reduction on both streams, so authoring both pays for
    the nontarget pass twice. Refuses on a plain run, which has no target stream."""

    slow: ClassVar[bool] = True
    type: Literal["TwoStreamCIMeanPerComponent"] = "TwoStreamCIMeanPerComponent"
