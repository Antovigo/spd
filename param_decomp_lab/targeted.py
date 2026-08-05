"""Lab-side orchestration for targeted parameter decomposition (tPD)."""

from typing import Any

from pydantic import NonNegativeFloat, PositiveInt

from param_decomp.base_config import BaseConfig
from param_decomp.configs import AnyLossMetricConfig
from param_decomp.distributed import get_distributed_state
from param_decomp.metrics.base import Metric
from param_decomp.metrics.importance_minimality import ImportanceMinimalityLossConfig
from param_decomp.metrics.persistent_pgd_recon import (
    PersistentPGDHiddenActsReconLossConfig,
    PersistentPGDReconLossConfig,
    PersistentPGDReconSubsetLossConfig,
    validate_pgd_scope,
)
from param_decomp.metrics.smooth_l0_importance_minimality import (
    SmoothL0ImportanceMinimalityLossConfig,
)
from param_decomp.metrics.stochastic_hidden_acts_recon import StochasticHiddenActsReconLossConfig
from param_decomp.metrics.unmasked_recon import UnmaskedReconLossConfig


class NontargetConfig[D: BaseConfig](BaseConfig):
    """Nontarget-pass settings for a targeted decomposition run.

    `data` reuses the experiment's data-config type and describes the broad nontarget
    distribution; `impmin_coeff_ratio` scales the importance-minimality coeff on the
    nontarget pass relative to the target pass.
    """

    data: D
    batch_size: PositiveInt
    eval_batch_size: PositiveInt
    impmin_coeff_ratio: NonNegativeFloat = 1.0


def _nontarget_variant(cfg: AnyLossMetricConfig, impmin_ratio: float) -> AnyLossMetricConfig | None:
    """The nontarget-pass form of one target-pass loss config, or `None` to drop it."""
    match cfg:
        case UnmaskedReconLossConfig() | StochasticHiddenActsReconLossConfig():
            return None
        case (
            PersistentPGDReconLossConfig()
            | PersistentPGDReconSubsetLossConfig()
            | PersistentPGDHiddenActsReconLossConfig()
        ):
            if cfg.nontarget_coeff is None:
                return None
            return cfg.model_copy(update={"coeff": cfg.nontarget_coeff})
        case ImportanceMinimalityLossConfig() | SmoothL0ImportanceMinimalityLossConfig():
            assert cfg.coeff is not None
            return cfg.model_copy(update={"coeff": cfg.coeff * impmin_ratio})
        case _:
            return cfg.model_copy()


def build_nontarget_loss_configs(
    loss_metrics: list[AnyLossMetricConfig],
    impmin_ratio: float,
    *,
    nontarget_batch_size: int,
) -> list[AnyLossMetricConfig]:
    """Derive the nontarget-pass loss set from the target-pass loss configs.

    Per-loss handling is in `_nontarget_variant`. Two things it encodes that aren't obvious:
    `StochasticHiddenReconSubsetLoss` is deliberately kept, because with the delta forced on
    it drives the components to be inactive at each *site* on the nontarget distribution — a
    more local version of the pressure the output recon applies; and the result must retain a
    full-model recon loss so the nontarget backward grads every parameter (a DDP requirement
    under the default reducer).
    """
    out = [
        variant
        for cfg in loss_metrics
        if (variant := _nontarget_variant(cfg, impmin_ratio)) is not None
    ]

    dist_state = get_distributed_state()
    validate_pgd_scope(
        out,
        batch_size=nontarget_batch_size,
        world_size=dist_state.world_size if dist_state is not None else 1,
    )
    return out


def split_eval_metrics(
    metrics: list[Metric[Any]],
) -> tuple[list[Metric[Any]], list[Metric[Any]]]:
    """Partition instantiated eval metrics into `(target_metrics, nontarget_metrics)`.

    Routing reads each metric's `eval_distribution` marker: `"nontarget"` metrics go
    right (fed by the mirror nontarget eval loop under `delta_override(1.0)`); everything
    else goes left (normal target eval pass). New nontarget metrics just set the marker —
    no edit here is needed.
    """
    target_metrics = [m for m in metrics if m.eval_distribution != "nontarget"]
    nontarget_metrics: list[Metric[Any]] = [
        m for m in metrics if m.eval_distribution == "nontarget"
    ]
    return target_metrics, nontarget_metrics
