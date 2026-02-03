import torch
from jaxtyping import Float
from torch import Tensor

from spd.configs import SamplingType
from spd.models.components import ComponentsMaskInfo, WeightDeltaAndMask, make_mask_infos
from spd.routing import Router


def calc_stochastic_component_mask_info(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    component_mask_sampling: SamplingType,
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    router: Router,
    force_delta: float | None = None,
) -> dict[str, ComponentsMaskInfo]:
    """Calculate stochastic component mask info for reconstruction losses.

    Args:
        force_delta: If None, use random mask for delta component. If a float (e.g., 0.0 or 1.0),
            use that value as the delta mask. Use 1.0 for nontarget data during training,
            None for target data, and 0.0 to evaluate active components only.
    """
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
            if force_delta is not None:
                delta_mask = torch.full(leading_dims, force_delta, device=device, dtype=dtype)
            else:
                delta_mask = torch.rand(leading_dims, device=device, dtype=dtype)
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
