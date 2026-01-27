from typing import Any, ClassVar, Literal, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.models.components import WeightDeltaAndMask, make_mask_infos
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import calc_sum_recon_loss_lm


def _ci_masked_recon_loss_update(
    model: ComponentModel,
    output_loss_type: Literal["mse", "kl"],
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None = None,
    force_delta_mask_one: bool = False,
) -> tuple[Float[Tensor, ""], int]:
    weight_deltas_and_masks: dict[str, WeightDeltaAndMask] | None = None
    if weight_deltas is not None:
        leading_dims = next(iter(ci.values())).shape[:-1]
        delta_mask_value = 1.0 if force_delta_mask_one else 0.0
        weight_deltas_and_masks = {
            layer: (
                weight_deltas[layer],
                torch.full(leading_dims, delta_mask_value, device=batch.device),
            )
            for layer in ci
        }
    mask_infos = make_mask_infos(ci, weight_deltas_and_masks=weight_deltas_and_masks)
    out = model(batch, mask_infos=mask_infos)
    loss_type = output_loss_type
    loss = calc_sum_recon_loss_lm(pred=out, target=target_out, loss_type=loss_type)
    return loss, out.shape.numel() if loss_type == "mse" else out.shape[:-1].numel()


def _ci_masked_recon_loss_compute(
    sum_loss: Float[Tensor, ""], n_examples: Int[Tensor, ""] | int
) -> Float[Tensor, ""]:
    return sum_loss / n_examples


def ci_masked_recon_loss(
    model: ComponentModel,
    output_loss_type: Literal["mse", "kl"],
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    target_out: Float[Tensor, "... vocab"],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None = None,
    force_delta_mask_one: bool = False,
) -> Float[Tensor, ""]:
    sum_loss, n_examples = _ci_masked_recon_loss_update(
        model=model,
        output_loss_type=output_loss_type,
        batch=batch,
        target_out=target_out,
        ci=ci,
        weight_deltas=weight_deltas,
        force_delta_mask_one=force_delta_mask_one,
    )
    return _ci_masked_recon_loss_compute(sum_loss, n_examples)


class CIMaskedReconLoss(Metric):
    """Recon loss when masking with CI values directly on all component layers."""

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        output_loss_type: Literal["mse", "kl"],
        use_delta_component: bool = False,
    ) -> None:
        self.model = model
        self.output_loss_type: Literal["mse", "kl"] = output_loss_type
        self.use_delta_component = use_delta_component
        self.sum_loss = torch.tensor(0.0, device=device)
        self.n_examples = torch.tensor(0, device=device)

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: CIOutputs,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None = None,
        **_: Any,
    ) -> None:
        sum_loss, n_examples = _ci_masked_recon_loss_update(
            model=self.model,
            output_loss_type=self.output_loss_type,
            batch=batch,
            target_out=target_out,
            ci=ci.lower_leaky,
            weight_deltas=weight_deltas if self.use_delta_component else None,
        )
        self.sum_loss += sum_loss
        self.n_examples += n_examples

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_loss = all_reduce(self.sum_loss, op=ReduceOp.SUM)
        n_examples = all_reduce(self.n_examples, op=ReduceOp.SUM)
        return _ci_masked_recon_loss_compute(sum_loss, n_examples)
