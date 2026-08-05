from typing import Any, Literal, override

import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.batch_and_loss_fns import ReconstructionLoss
from param_decomp.component_model import ComponentModel
from param_decomp.distributed import all_reduce
from param_decomp.masks import AllLayersRouter
from param_decomp.metrics.base import EvalCadenceConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.pgd_utils import PGDConfig, pgd_masked_recon_loss_update


class PGDReconLossConfig(PGDConfig):
    type: Literal["PGDReconLoss"] = "PGDReconLoss"


class NontargetPGDReconLossConfig(PGDConfig, EvalCadenceConfig):
    """Eval-only, so `slow` is a config decision: a 20-step attack costs `n_steps + 1`
    full forwards per eval batch and belongs on the slow cadence."""

    type: Literal["NontargetPGDReconLoss"] = "NontargetPGDReconLoss"


def pgd_recon_loss(
    *,
    model: ComponentModel,
    batch: Any,
    target_out: Tensor,
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    pgd_config: PGDConfig,
    reconstruction_loss: ReconstructionLoss,
) -> Float[Tensor, ""]:
    """Compute PGD masked recon loss directly (helper for tests/notebooks)."""
    sum_loss, n = pgd_masked_recon_loss_update(
        model=model,
        batch=batch,
        ci=ci,
        weight_deltas=weight_deltas,
        target_out=target_out,
        router=AllLayersRouter(),
        pgd_config=pgd_config,
        reconstruction_loss=reconstruction_loss,
    )
    return sum_loss / n


class _PGDReconLossBase[TConfig: PGDConfig](Metric[TConfig]):
    """Recon loss with adversarially-optimised masks routing to all component layers.

    Runs `cfg.n_steps` of per-step PGD on fresh adversarial sources each batch (no
    cross-step persistence).
    """

    log_namespace = "loss"

    @override
    def reset(self) -> None:
        self.sum_loss = torch.zeros((), device=self.device)
        self.n_examples = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        wd = ctx.weight_deltas if ctx.use_delta_component else None
        sum_loss, n = pgd_masked_recon_loss_update(
            model=self.model,
            batch=ctx.batch,
            ci=ctx.ci.lower_leaky,
            weight_deltas=wd,
            target_out=ctx.target_out,
            router=AllLayersRouter(),
            pgd_config=self.cfg,
            reconstruction_loss=ctx.reconstruction_loss,
        )
        self.sum_loss += sum_loss.detach()
        self.n_examples += n
        return sum_loss / n

    @override
    def compute(self) -> MetricResult:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return sum_loss / n_examples


class PGDReconLoss(_PGDReconLossBase[PGDReconLossConfig]):
    short_name = "PGDRecon"


class NontargetPGDReconLoss(_PGDReconLossBase[NontargetPGDReconLossConfig]):
    """Worst-case recon error on the *nontarget* distribution, delta pinned on.

    The adversary can only push masks up from CI toward 1, and a mask of 1 restores the
    target model exactly — so the worst case is the most damaging partial ablation of the
    components, and the CI-masked error is its floor. Eval-only: the training-loss
    counterpart is a PPGD instance with `nontarget_coeff` set.
    """

    eval_distribution = "nontarget"
    short_name = "NontargetPGDRecon"
