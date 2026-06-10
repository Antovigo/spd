"""`ChunkwiseSubsetReconLossConfig` — config for the flat (single-pool) chunkwise
subset recon loss whose `Metric` impl lives lab-side.

The config lives in core so it joins `AnyLossMetricConfig` and validates in a YAML
`pd.loss_metrics`; the impl (`param_decomp_lab.metrics.chunkwise_subset_recon`) needs
the vendored `LMComponentModel` (fused-KL LM-head bypass) and the lab recon-plan
machinery, so it cannot live in core. The lab FSDP trainer dispatches this `type` to
the lab class.
"""

from typing import Annotated, Literal

from pydantic import Field, PositiveInt

from param_decomp.masks import SubsetRoutingType, UniformKSubsetRoutingConfig
from param_decomp.metrics.base import LossMetricConfig


class ChunkwiseSubsetReconLossConfig(LossMetricConfig):
    """Reconstruction loss that mirrors the 3-pool / 2-pool chunkwise subset recon.

    The decomposed sites (`model.target_module_paths`, in order) are grouped into
    chunks of `sites_per_chunk`; each chunk runs `SubsetReconPlan(routing, n_samples)`
    — one masked suffix forward per generated routing, all the chunk's sites swapped in
    with a per-position routing draw — and the recon is the fused-linear-KL against the
    clean logits (when `use_fused_kl`). The total is the mean over all chunk forwards of
    `recon_loss / n_positions`, matching the 2-pool's per-step recon.
    """

    type: Literal["ChunkwiseSubsetReconLoss"] = "ChunkwiseSubsetReconLoss"
    sites_per_chunk: PositiveInt
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]
    n_samples: PositiveInt = 1
    use_fused_kl: bool = True
