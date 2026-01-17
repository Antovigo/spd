import torch
from jaxtyping import Bool, Float
from torch import Tensor

from spd.configs import SamplingType
from spd.models.components import ComponentsMaskInfo, WeightDeltaAndMask, make_mask_infos
from spd.routing import Router


def calc_stochastic_component_mask_info(
    causal_importances: dict[str, Float[Tensor, "... C"]],
    component_mask_sampling: SamplingType,
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    router: Router,
    is_target: Bool[Tensor, "..."] | None = None,
) -> dict[str, ComponentsMaskInfo]:
    """Calculate stochastic component mask info for SPD forward passes.

    Args:
        causal_importances: CI values per layer with shape [..., C]
        component_mask_sampling: Sampling strategy for component masks
        weight_deltas: Delta weights per layer (if using delta component)
        router: Router for determining which layers to mask
        is_target: Boolean mask indicating target samples. For targeted decomposition:
            - Target samples (True): delta mask is stochastic (standard SPD ablation)
            - Non-target samples (False): delta mask is always 1.0 (delta fully active)
            If None, all samples use stochastic delta masks (standard SPD behavior).
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
            delta_mask = torch.rand(leading_dims, device=device, dtype=dtype)

            if is_target is not None: # set delta_mask to one on nontarget samples
                delta_mask = torch.where(is_target, delta_mask, torch.ones_like(delta_mask))

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
