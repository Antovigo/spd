from typing import Annotated, Any, Literal, override

import torch
import torch.nn.functional as F
from jaxtyping import Float
from pydantic import Field, NonNegativeFloat
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.base_config import BaseConfig
from param_decomp.batch_and_loss_fns import ReconstructionLoss
from param_decomp.component_model import ComponentModel
from param_decomp.components import LinearComponents
from param_decomp.distributed import all_reduce
from param_decomp.masks import (
    Router,
    SamplingType,
    SubsetRoutingType,
    UniformKSubsetRoutingConfig,
    calc_stochastic_component_mask_info,
    get_subset_router,
)
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp.torch_helpers import get_obj_device


class HiddenActsReconAux(BaseConfig):
    """Auxiliary hidden-activation reconstruction riding on the host recon loss.

    Adds `coeff * MSE(masked site output, frozen x@W + b)` per decomposed linear site,
    sharing the host's stochastically-masked forwards — no extra forward: the frozen
    targets are recomputed from the step's already-cached clean input activations. The
    MSE covers exactly the sites masked in each draw, at the positions routed to
    components. Eval-logged as `<Host>/hidden_acts`; `coeff: 0.0` measures without
    affecting training.
    """

    coeff: NonNegativeFloat


class StochasticReconSubsetLossConfig(LossMetricConfig):
    type: Literal["StochasticReconSubsetLoss"] = "StochasticReconSubsetLoss"
    routing: Annotated[
        SubsetRoutingType, Field(discriminator="type", default=UniformKSubsetRoutingConfig())
    ]
    hidden_acts_recon: HiddenActsReconAux | None = None


def _stochastic_recon_subset_loss_update(
    model: ComponentModel,
    sampling: SamplingType,
    n_mask_samples: int,
    batch: Any,
    target_out: Tensor,
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    router: Router,
    reconstruction_loss: ReconstructionLoss,
    hidden_acts_targets: dict[str, Float[Tensor, "..."]] | None = None,
) -> tuple[Float[Tensor, ""], int, Float[Tensor, ""], int]:
    """Returns `(sum_recon_loss, n_examples, sum_hidden_acts_sq_err, n_hidden_elems)`;
    the last two are zero unless `hidden_acts_targets` is given."""
    assert ci, "Empty ci"
    device = get_obj_device(ci)
    sum_loss = torch.zeros((), device=device)
    sum_sq_err = torch.zeros((), device=device)
    n_examples = 0
    n_elems = 0
    stoch_mask_infos_list = [
        calc_stochastic_component_mask_info(
            causal_importances=ci,
            component_mask_sampling=sampling,
            weight_deltas=weight_deltas,
            router=router,
        )
        for _ in range(n_mask_samples)
    ]
    for stoch_mask_infos in stoch_mask_infos_list:
        if hidden_acts_targets is None:
            out = model(batch, mask_infos=stoch_mask_infos)
        else:
            result = model(batch, mask_infos=stoch_mask_infos, cache_type="output")
            out = result.output
            for module_name, info in stoch_mask_infos.items():
                target = hidden_acts_targets[module_name]
                diff = result.cache[module_name] - target
                if isinstance(info.routing_mask, Tensor):
                    diff = diff[info.routing_mask]
                sum_sq_err = sum_sq_err + diff.float().pow(2).sum()
                n_elems += diff.numel()
        loss, batch_n = reconstruction_loss(out, target_out)
        sum_loss = sum_loss + loss
        n_examples += batch_n
    return sum_loss, n_examples, sum_sq_err, n_elems


def stochastic_recon_subset_loss(
    model: ComponentModel,
    sampling: SamplingType,
    n_mask_samples: int,
    batch: Any,
    target_out: Tensor,
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    routing: SubsetRoutingType,
    reconstruction_loss: ReconstructionLoss,
) -> Float[Tensor, ""]:
    """Compute stochastic subset recon loss directly (helper for tests/notebooks)."""
    sum_loss, n, _, _ = _stochastic_recon_subset_loss_update(
        model=model,
        sampling=sampling,
        n_mask_samples=n_mask_samples,
        batch=batch,
        target_out=target_out,
        ci=ci,
        weight_deltas=weight_deltas,
        router=get_subset_router(routing, device=get_obj_device(model)),
        reconstruction_loss=reconstruction_loss,
    )
    return sum_loss / n


class StochasticReconSubsetLoss(Metric[StochasticReconSubsetLossConfig]):
    """Stochastic recon loss with masks applied only on a routed subset of layers.

    Subset chosen per `cfg.routing`. Sums recon loss across `ctx.n_mask_samples` draws.
    """

    log_namespace = "loss"
    short_name = "StochReconSub"

    @override
    def bind(self, *, model: ComponentModel, device: str) -> None:
        super().bind(model=model, device=device)
        self.router = get_subset_router(self.cfg.routing, device)

    @override
    def reset(self) -> None:
        self.sum_loss = torch.zeros((), device=self.device)
        self.n_examples = torch.zeros((), device=self.device, dtype=torch.long)
        self.sum_sq_err = torch.zeros((), device=self.device)
        self.n_elems = torch.zeros((), device=self.device, dtype=torch.long)

    def _hidden_acts_targets(self, ctx: MetricContext) -> dict[str, Tensor]:
        """Frozen per-site clean outputs `x@W + b` from the cached clean input acts."""
        targets: dict[str, Tensor] = {}
        for module_name, x in ctx.pre_weight_acts.items():
            comp = self.model.components[module_name]
            assert isinstance(comp, LinearComponents), (
                f"hidden_acts_recon supports linear sites only, got {type(comp).__name__} "
                f"for {module_name}"
            )
            w = self.model.target_weight(module_name)
            targets[module_name] = F.linear(x.detach().to(w.dtype), w, comp.bias)
        return targets

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        wd = ctx.weight_deltas if ctx.use_delta_component else None
        aux = self.cfg.hidden_acts_recon
        sum_loss, n, sum_sq_err, n_elems = _stochastic_recon_subset_loss_update(
            model=self.model,
            sampling=ctx.sampling,
            n_mask_samples=ctx.n_mask_samples,
            batch=ctx.batch,
            target_out=ctx.target_out,
            ci=ctx.ci.lower_leaky,
            weight_deltas=wd,
            router=self.router,
            reconstruction_loss=ctx.reconstruction_loss,
            hidden_acts_targets=self._hidden_acts_targets(ctx) if aux is not None else None,
        )
        self.sum_loss += sum_loss.detach()
        self.n_examples += n
        loss = sum_loss / n
        if aux is not None:
            assert n_elems > 0, "hidden_acts_recon configured but no masked site produced acts"
            self.sum_sq_err += sum_sq_err.detach()
            self.n_elems += n_elems
            assert self.cfg.coeff is not None and self.cfg.coeff > 0, (
                "hidden_acts_recon rides the host loss coeff; host coeff must be > 0"
            )
            # total_loss applies the host coeff to this return value, so fold the aux in
            # at the coefficient ratio: host_coeff * (kl + ratio * mse) = host_coeff * kl
            # + aux_coeff * mse.
            loss = loss + (aux.coeff / self.cfg.coeff) * (sum_sq_err / n_elems)
        return loss

    @override
    def compute(self) -> MetricResult:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        recon = sum_loss / n_examples
        if self.cfg.hidden_acts_recon is None:
            return recon
        sum_sq_err = all_reduce(self.sum_sq_err, op=ReduceOp.SUM)
        n_elems = all_reduce(self.n_elems, op=ReduceOp.SUM)
        name = type(self).__name__
        return {name: recon, f"{name}/hidden_acts": sum_sq_err / n_elems.float()}
