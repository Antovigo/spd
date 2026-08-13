"""Authored metric schemas whose semantics require an LM target."""

from typing import ClassVar, Literal

from pydantic import Field, NonNegativeInt, PositiveFloat, PositiveInt

from param_decomp.core.base_config import BaseConfig
from param_decomp.core.configs import HiddenActsReconstruction


class CEandKLLossesConfig(BaseConfig):
    """Token-level CE/KL metrics for categorical LM outputs."""

    slow: ClassVar[bool] = False
    type: Literal["CEandKLLosses"] = "CEandKLLosses"
    rounding_threshold: float


class CIMaskedAttnPatternsReconLossConfig(BaseConfig):
    slow: ClassVar[bool] = False
    type: Literal["CIMaskedAttnPatternsReconLoss"] = "CIMaskedAttnPatternsReconLoss"


class StochasticAttnPatternsReconLossConfig(BaseConfig):
    slow: ClassVar[bool] = False
    type: Literal["StochasticAttnPatternsReconLoss"] = "StochasticAttnPatternsReconLoss"
    n_mask_samples: PositiveInt = 1


class ArithmeticCEKLConfig(BaseConfig):
    rounding_threshold: float


class ArithmeticCIL0Config(BaseConfig):
    ci_alive_threshold: float
    groups: dict[str, list[str]] | None


class ArithmeticFreshPGDConfig(BaseConfig):
    name: str | None = None
    n_steps: NonNegativeInt
    step_size: PositiveFloat
    hidden_acts_reconstruction: HiddenActsReconstruction | None = None


class ArithmeticProbeMetrics(BaseConfig):
    """Scalar operations evaluated on the arithmetic grid rather than corpus batches."""

    ce_kl: ArithmeticCEKLConfig
    ci_l0: ArithmeticCIL0Config
    fresh_pgd: ArithmeticFreshPGDConfig | None


class ArithmeticCIGridConfig(BaseConfig):
    """Per-component causal-importance heatmaps over an arithmetic operand grid."""

    slow: ClassVar[bool] = True
    type: Literal["ArithmeticCIGrid"] = "ArithmeticCIGrid"
    probe_metrics: ArithmeticProbeMetrics | None
    """`null` runs the heatmaps ALONE. The scalar probes cost a second full-grid forward
    (logits over every prompt), which is what makes a 100x100 grid unaffordable on a 45GB
    card. On a targeted run `TargetPoolScalars` already reports CE/KL and CI-L0 over the
    run's own prompt pool, so these are redundant there."""
    operation: Literal["add", "sub", "mul"] = "add"
    a_range: tuple[int, int] = (1, 100)
    b_range: tuple[int, int] = (1, 100)
    thresholds: list[float] = Field(default_factory=lambda: [0.1])
    top_k: PositiveInt = 24
    chunk_prompts: PositiveInt | None = None
    """Rows per forward through the grid. `null` sends all `|a_range| x |b_range|` prompts
    in ONE forward, which is what the grid step did unconditionally — 10000 prompts needs
    ~20GiB and OOMs a 45GB L40. Chunking bounds the forward without shrinking the grid; the
    CI / x@V columns are concatenated across chunks and the per-component max is a running
    max, so which components are plotted is chunk-count-invariant and their values match the
    unchunked pass up to float reassociation (SPEC D4). Prefer a multiple of the device count
    so no chunk carries sharding pad."""


class TargetPoolFreshPGDConfig(BaseConfig):
    n_steps: NonNegativeInt
    step_size: PositiveFloat


class TargetPoolScalarsConfig(BaseConfig):
    """CI-L0 and a fresh-PGD recon probe on the tPD run's OWN `prompts:` pool.

    A targeted run's `eval.metrics` measure `data.eval` — the broad NON-TARGET stream (T2),
    since the target pool has no held-out split. This metric is the target-side twin: the
    same kernels over pool batches drawn exactly like a training target batch. Unlike
    `ArithmeticCIGrid`, which builds its own single-operation grid, this reads the run's
    authored pool, so a mixed add/sub pool is measured whole. Logged under
    `eval/target_pool/`. Refuses on a plain (non-targeted) run, which has no pool."""

    slow: ClassVar[bool] = False
    type: Literal["TargetPoolScalars"] = "TargetPoolScalars"
    ci_alive_threshold: float = 0.0
    ci_l0_groups: dict[str, list[str]] | None = None
    fresh_pgd: TargetPoolFreshPGDConfig | None


class TwoStreamCIMeanPerComponentConfig(BaseConfig):
    """Both streams' mean CI per component on ONE axis per site, ordered by descending
    TARGET mean and coloured by stream.

    Supersedes authoring `CIMeanPerComponent` on a targeted run: it computes the same broad-
    stream reduction plus the target-pool one, so authoring both would pay for the broad
    pass twice. Refuses on a plain run, which has no target stream."""

    slow: ClassVar[bool] = True
    type: Literal["TwoStreamCIMeanPerComponent"] = "TwoStreamCIMeanPerComponent"
