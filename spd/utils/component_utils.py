import torch
import torch.distributed as dist
from jaxtyping import Float
from torch import Tensor

from spd.configs import SamplingType
from spd.models.components import (
    Components,
    ComponentsMaskInfo,
    WeightDeltaAndMask,
    make_mask_infos,
)
from spd.routing import Router
from spd.utils.distributed_utils import all_reduce, is_distributed


def calc_stochastic_component_mask_info(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    component_mask_sampling: SamplingType,
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    router: Router,
    force_delta: float | None = None,
) -> dict[str, ComponentsMaskInfo]:
    ci_sample = next(iter(causal_importances.values()))
    leading_dims = ci_sample.shape[:-1]
    device = ci_sample.device
    dtype = ci_sample.dtype

    component_masks: dict[str, Float[Tensor, "... C"]] = {}
    for layer, ci in causal_importances.items():
        match component_mask_sampling:
            case "binomial":
                stochastic_source = torch.randint(0, 2, ci.shape, device=device).float()
            case "continuous":
                stochastic_source = torch.rand_like(ci)
        component_masks[layer] = ci + (1 - ci) * stochastic_source

    weight_deltas_and_masks: dict[str, WeightDeltaAndMask] | None = None
    if weight_deltas is not None:
        weight_deltas_and_masks = {}
        for layer in causal_importances:
            delta_mask = (
                torch.full(leading_dims, force_delta, device=device, dtype=dtype)
                if force_delta is not None
                else torch.rand(leading_dims, device=device, dtype=dtype)
            )
            weight_deltas_and_masks[layer] = (weight_deltas[layer], delta_mask)

    routing_masks = router.get_masks(
        module_names=list(causal_importances.keys()),
        mask_shape=leading_dims,
    )

    return make_mask_infos(
        component_masks=component_masks,
        weight_deltas_and_masks=weight_deltas_and_masks,
        routing_masks=routing_masks,
    )


def calc_ci_l_zero(ci: Float[Tensor, "... C"], threshold: float) -> float:
    return (ci > threshold).float().sum(-1).mean().item()


@torch.no_grad()  # pyright: ignore[reportUntypedFunctionDecorator]
def apply_ci_scaled_weight_decay(
    components: dict[str, Components],
    step_max_ci: dict[str, Float[Tensor, " C"]] | None,
    lr: float,
    weight_decay: float,
) -> None:
    """Apply weight decay with sqrt for linear decay on W = V @ U.

    When step_max_ci is provided, decay is CI-scaled: scale = sqrt(1 - lr * wd * (1 - max_ci)).
    When step_max_ci is None, uniform decay: scale = sqrt(1 - lr * wd).
    Applied to both V and U, so effective decay on W is scale².
    """
    if step_max_ci is not None and is_distributed():
        for layer_name in step_max_ci:
            all_reduce(step_max_ci[layer_name], op=dist.ReduceOp.MAX)

    for layer_name, comps in components.items():
        if step_max_ci is not None:
            decay_factors = 1.0 - step_max_ci[layer_name].clamp(0.0, 1.0)  # (C,)
        else:
            decay_factors = torch.ones(comps.C, device=comps.V.device)
        scale = (1.0 - lr * weight_decay * decay_factors).sqrt()
        comps.V.data.mul_(scale[None, :])  # V: (v_dim, C)
        comps.U.data.mul_(scale[:, None])  # U: (C, u_dim)
