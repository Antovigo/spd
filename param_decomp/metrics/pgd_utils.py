from collections.abc import Callable
from typing import Any, Literal

import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.batch_and_loss_fns import ReconstructionLoss
from param_decomp.component_model import ComponentModel
from param_decomp.distributed import all_reduce, broadcast_tensor
from param_decomp.masks import (
    ComponentsMaskInfo,
    Router,
    RoutingMasks,
    WeightDeltaAndMask,
    interpolate_component_mask,
    make_mask_infos,
)
from param_decomp.metrics.base import LossMetricConfig
from param_decomp.targeted import get_delta_override

PGDInitStrategy = Literal["random", "ones", "zeroes"]
MaskScope = Literal["unique_per_datapoint", "shared_across_batch"]


class PGDConfig(LossMetricConfig):
    """Shared base for per-step PGD loss configs."""

    init: PGDInitStrategy
    step_size: float
    n_steps: int
    mask_scope: MaskScope


def get_pgd_init_tensor(
    init: PGDInitStrategy,
    shape: tuple[int, ...] | torch.Size,
    device: torch.device | str,
) -> Float[Tensor, "... shape"]:
    match init:
        case "random":
            return torch.rand(shape, device=device)
        case "ones":
            return torch.ones(shape, device=device)
        case "zeroes":
            return torch.zeros(shape, device=device)


def _init_adv_sources(
    model: ComponentModel,
    batch_dims: tuple[int, ...],
    device: torch.device | str,
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    pgd_config: PGDConfig,
) -> dict[str, Float[Tensor, "*batch_dims mask_c"]]:
    adv_sources: dict[str, Float[Tensor, "*batch_dims mask_c"]] = {}
    for module_name in model.target_module_paths:
        module_c = model.module_to_c[module_name]
        mask_c = (
            module_c
            if (weight_deltas is None or get_delta_override() is not None)
            else module_c + 1
        )
        match pgd_config.mask_scope:
            case "unique_per_datapoint":
                shape = torch.Size([*batch_dims, mask_c])
                source = get_pgd_init_tensor(pgd_config.init, shape, device)
            case "shared_across_batch":
                singleton_batch_dims = [1 for _ in batch_dims]
                shape = torch.Size([*singleton_batch_dims, mask_c])
                source = broadcast_tensor(get_pgd_init_tensor(pgd_config.init, shape, device))
        adv_sources[module_name] = source.requires_grad_(True)
    return adv_sources


def _run_pgd_loop(
    adv_sources: dict[str, Float[Tensor, "..."]],
    pgd_config: PGDConfig,
    fwd_fn: Callable[[], tuple[Float[Tensor, ""], int]],
) -> tuple[Float[Tensor, ""], int]:
    for _ in range(pgd_config.n_steps):
        assert all(adv.grad is None for adv in adv_sources.values())
        with torch.enable_grad():
            sum_loss, n_examples = fwd_fn()
            loss = sum_loss / n_examples
        grads = torch.autograd.grad(loss, list(adv_sources.values()))
        match pgd_config.mask_scope:
            case "shared_across_batch":
                adv_sources_grads = {
                    k: all_reduce(g, op=ReduceOp.AVG)
                    for k, g in zip(adv_sources.keys(), grads, strict=True)
                }
            case "unique_per_datapoint":
                adv_sources_grads = dict(zip(adv_sources.keys(), grads, strict=True))
        with torch.no_grad():
            for k in adv_sources:
                adv_sources[k].add_(pgd_config.step_size * adv_sources_grads[k].sign())
                adv_sources[k].clamp_(0.0, 1.0)

    return fwd_fn()


def construct_mask_infos_from_adv_sources(
    expanded_adv_sources: dict[str, Float[Tensor, "*batch_dims mask_c"]],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    routing_masks: RoutingMasks,
) -> dict[str, ComponentsMaskInfo]:
    """Turn adversarial sources already broadcast to the batch shape into mask infos.

    Under a `delta_override` the delta mask is pinned to the override value and the sources
    carry no delta channel at all: on the nontarget pass the residual has to stay fully on
    for the pass to mean anything, so the adversary attacks components only. Shared with
    PPGD, which differs only in how its persistent sources reach the batch shape.
    """
    batch_dims = next(iter(expanded_adv_sources.values())).shape[:-1]
    adv_sources_components: dict[str, Float[Tensor, "*batch_dims C"]]
    weight_deltas_and_masks: dict[str, WeightDeltaAndMask] | None
    match weight_deltas:
        case None:
            adv_sources_components, weight_deltas_and_masks = expanded_adv_sources, None
        case dict():
            override = get_delta_override()
            if override is not None:
                pinned = torch.full(
                    batch_dims, override, device=next(iter(expanded_adv_sources.values())).device
                )
                adv_sources_components = expanded_adv_sources
                weight_deltas_and_masks = {k: (weight_deltas[k], pinned) for k in weight_deltas}
            else:
                adv_sources_components = {k: v[..., :-1] for k, v in expanded_adv_sources.items()}
                weight_deltas_and_masks = {
                    k: (weight_deltas[k], expanded_adv_sources[k][..., -1]) for k in weight_deltas
                }

    return make_mask_infos(
        component_masks=interpolate_component_mask(ci, adv_sources_components),
        weight_deltas_and_masks=weight_deltas_and_masks,
        routing_masks=routing_masks,
    )


PGDObjective = Callable[[dict[str, ComponentsMaskInfo]], tuple[Float[Tensor, ""], int]]
"""Maps the mask payload built from the current adversarial sources to `(sum, n_examples)`.

PGD ascends `sum / n_examples`. Output reconstruction is one such objective; per-site
activation error is another.
"""


def pgd_masked_objective_update(
    model: ComponentModel,
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    router: Router,
    pgd_config: PGDConfig,
    objective: PGDObjective,
) -> tuple[Float[Tensor, ""], int]:
    """Per-step PGD against an arbitrary mask-consuming objective.

    Inits fresh adversarial sources, runs `pgd_config.n_steps` of inner sign-PGD ascending
    `objective`, and returns it evaluated at the final sources. Callers wanting output
    reconstruction should use `pgd_masked_recon_loss_update`.
    """
    batch_dims = next(iter(ci.values())).shape[:-1]
    device = next(iter(ci.values())).device
    routing_masks = router.get_masks(module_names=model.target_module_paths, mask_shape=batch_dims)
    adv_sources = _init_adv_sources(model, batch_dims, device, weight_deltas, pgd_config)

    def forward_at_current_sources() -> tuple[Float[Tensor, ""], int]:
        return objective(
            construct_mask_infos_from_adv_sources(
                expanded_adv_sources={k: v.expand(*batch_dims, -1) for k, v in adv_sources.items()},
                ci=ci,
                weight_deltas=weight_deltas,
                routing_masks=routing_masks,
            )
        )

    return _run_pgd_loop(adv_sources, pgd_config, forward_at_current_sources)


def pgd_masked_recon_loss_update(
    model: ComponentModel,
    batch: Any,
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
    target_out: Tensor,
    router: Router,
    pgd_config: PGDConfig,
    reconstruction_loss: ReconstructionLoss,
) -> tuple[Float[Tensor, ""], int]:
    """Per-step PGD masked recon.

    Inits fresh adversarial sources, runs `pgd_config.n_steps` of inner sign-PGD against
    the recon objective, returns `(sum_loss, n_examples)` evaluated at the final sources.
    """

    def recon_objective(
        mask_infos: dict[str, ComponentsMaskInfo],
    ) -> tuple[Float[Tensor, ""], int]:
        return reconstruction_loss(model(batch, mask_infos=mask_infos), target_out)

    return pgd_masked_objective_update(
        model=model,
        ci=ci,
        weight_deltas=weight_deltas,
        router=router,
        pgd_config=pgd_config,
        objective=recon_objective,
    )
