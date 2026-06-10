"""Loss-metric configs.

One config per loss `Metric` in `param_decomp/metrics/` (plus the lab-side chunkwise
subset recon). Each carries a unique `type: Literal["<ClassName>"]` discriminator;
`AnyLossMetricConfig` in `param_decomp_config.pd` unions them for YAML validation.
"""

from typing import Annotated, Literal

from pydantic import Field, NonNegativeFloat, NonNegativeInt, PositiveInt

from param_decomp_config.base import BaseConfig, Probability
from param_decomp_config.routing import SubsetRoutingType, UniformKSubsetRoutingConfig
from param_decomp_config.schedule import ScheduleConfig


class LossMetricConfig(BaseConfig):
    """Pydantic config for a metric that can also be used as a training loss.

    `coeff` is required when this metric is listed under `loss_metrics` (asserted by
    `PDConfig`'s field validator); ignored for eval-only instances.
    """

    coeff: float | None = None


class FaithfulnessLossConfig(LossMetricConfig):
    type: Literal["FaithfulnessLoss"] = "FaithfulnessLoss"


class ImportanceMinimalityLossConfig(LossMetricConfig):
    """Config for the `L_p`-style importance-minimality penalty on upper-leaky CI values.

    `pnorm` is the initial `p`; `beta` weights the entropy-like `mean * log2(1 + sum)`
    term added on top of the `L_p` term. `pnorm` is linearly annealed toward
    `p_anneal_final_p` between `p_anneal_start_frac` and `p_anneal_end_frac` of training
    (no-op when `p_anneal_final_p is None` or `p_anneal_start_frac == 1.0`).
    """

    type: Literal["ImportanceMinimalityLoss"] = "ImportanceMinimalityLoss"
    pnorm: NonNegativeFloat
    beta: NonNegativeFloat
    p_anneal_start_frac: Probability = 1.0
    p_anneal_final_p: NonNegativeFloat | None = None
    p_anneal_end_frac: Probability = 1.0
    eps: NonNegativeFloat = 1e-12


class CIMaskedReconLossConfig(LossMetricConfig):
    type: Literal["CIMaskedReconLoss"] = "CIMaskedReconLoss"


class CIMaskedReconLayerwiseLossConfig(LossMetricConfig):
    type: Literal["CIMaskedReconLayerwiseLoss"] = "CIMaskedReconLayerwiseLoss"


class CIMaskedReconSubsetLossConfig(LossMetricConfig):
    type: Literal["CIMaskedReconSubsetLoss"] = "CIMaskedReconSubsetLoss"
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]


class StochasticReconLossConfig(LossMetricConfig):
    type: Literal["StochasticReconLoss"] = "StochasticReconLoss"


class StochasticReconLayerwiseLossConfig(LossMetricConfig):
    type: Literal["StochasticReconLayerwiseLoss"] = "StochasticReconLayerwiseLoss"


class StochasticReconSubsetLossConfig(LossMetricConfig):
    type: Literal["StochasticReconSubsetLoss"] = "StochasticReconSubsetLoss"
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]


class StochasticHiddenActsReconLossConfig(LossMetricConfig):
    type: Literal["StochasticHiddenActsReconLoss"] = "StochasticHiddenActsReconLoss"


class UnmaskedReconLossConfig(LossMetricConfig):
    type: Literal["UnmaskedReconLoss"] = "UnmaskedReconLoss"


class ChunkwiseSubsetReconLossConfig(LossMetricConfig):
    """Reconstruction loss that mirrors the 3-pool / 2-pool chunkwise subset recon.

    The decomposed sites (`model.target_module_paths`, in order) are grouped into
    chunks of `sites_per_chunk`; each chunk runs `SubsetReconPlan(routing, n_samples)`
    — one masked suffix forward per generated routing, all the chunk's sites swapped in
    with a per-position routing draw — and the recon is the fused-linear-KL against the
    clean logits (when `use_fused_kl`). The total is the mean over all chunk forwards of
    `recon_loss / n_positions`, matching the 2-pool's per-step recon.

    The impl (`param_decomp_lab.metrics.chunkwise_subset_recon`) lives lab-side: it
    needs the vendored `LMComponentModel` (fused-KL LM-head bypass) and the lab
    recon-plan machinery. The lab FSDP trainer dispatches this `type` to the lab class.
    """

    type: Literal["ChunkwiseSubsetReconLoss"] = "ChunkwiseSubsetReconLoss"
    sites_per_chunk: PositiveInt
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]
    n_samples: PositiveInt = 1
    use_fused_kl: bool = True


PGDInitStrategy = Literal["random", "ones", "zeroes"]
MaskScope = Literal["unique_per_datapoint", "shared_across_batch"]


class PGDConfig(LossMetricConfig):
    """Shared base for per-step PGD loss configs."""

    init: PGDInitStrategy
    step_size: float
    n_steps: int
    mask_scope: MaskScope


class PGDReconLossConfig(PGDConfig):
    type: Literal["PGDReconLoss"] = "PGDReconLoss"


class PGDReconLayerwiseLossConfig(PGDConfig):
    type: Literal["PGDReconLayerwiseLoss"] = "PGDReconLayerwiseLoss"


class PGDReconSubsetLossConfig(PGDConfig):
    type: Literal["PGDReconSubsetLoss"] = "PGDReconSubsetLoss"
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]


class SignPGDConfig(BaseConfig):
    """Sign-PGD optimizer config (adds `lr * sign(grad)` to sources)."""

    type: Literal["sign"] = "sign"
    lr_schedule: ScheduleConfig


class AdamPGDConfig(BaseConfig):
    """Adam-style PGD optimizer config."""

    type: Literal["adam"] = "adam"
    beta1: Probability = Field(default=0.9, description="Adam beta1 for masks")
    beta2: Probability = Field(default=0.999, description="Adam beta2 for masks")
    eps: NonNegativeFloat = Field(default=1e-8, description="Adam epsilon for masks")
    lr_schedule: ScheduleConfig


PGDOptimizerConfig = SignPGDConfig | AdamPGDConfig


class SingleSourceScope(BaseConfig):
    """PPGD source scope: one shared source vector across the whole batch."""

    type: Literal["single_source"] = "single_source"


class BroadcastAcrossBatchScope(BaseConfig):
    """PPGD source scope: shared across batch elements but free along other batch dims."""

    type: Literal["broadcast_across_batch"] = "broadcast_across_batch"


class RepeatAcrossBatchScope(BaseConfig):
    """PPGD source scope: `n_sources` source vectors tiled along the batch dim.

    `n_sources` must divide the per-rank batch size.
    """

    type: Literal["repeat_across_batch"] = "repeat_across_batch"
    n_sources: PositiveInt


class PerBatchPerPositionScope(BaseConfig):
    """PPGD source scope: an independent source per batch element and position.

    Skips cross-rank synchronization of source state.
    """

    type: Literal["per_batch_per_position"] = "per_batch_per_position"


PersistentPGDSourceScope = Annotated[
    SingleSourceScope
    | BroadcastAcrossBatchScope
    | RepeatAcrossBatchScope
    | PerBatchPerPositionScope,
    Field(discriminator="type"),
]


class _PersistentPGDBaseConfig(LossMetricConfig):
    """Shared fields for persistent PGD configs.

    `update()` returns `None` before `start_frac` of training. Under
    `use_sigmoid_parameterization=True` sources are unconstrained and read via sigmoid;
    otherwise sources are clamped to `[0, 1]` after each step.
    """

    optimizer: Annotated[PGDOptimizerConfig, Field(discriminator="type")]
    scope: PersistentPGDSourceScope
    use_sigmoid_parameterization: bool = False
    n_warmup_steps: NonNegativeInt = Field(
        default=0,
        description=(
            "Extra inner PGD source-optimization steps on each train batch before the final loss"
            " computation."
        ),
    )
    start_frac: Probability = 0.0
    n_samples: PositiveInt = 1


class PersistentPGDReconLossConfig(_PersistentPGDBaseConfig):
    type: Literal["PersistentPGDReconLoss"] = "PersistentPGDReconLoss"


class PersistentPGDReconSubsetLossConfig(_PersistentPGDBaseConfig):
    type: Literal["PersistentPGDReconSubsetLoss"] = "PersistentPGDReconSubsetLoss"
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]
