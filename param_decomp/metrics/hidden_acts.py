"""Shared pieces of the per-site (hidden-activation) reconstruction error.

Used by the `StochasticHiddenReconSubsetLoss` training loss and by the lab-side
hidden-acts eval probes, so all of them measure the same quantity.
"""

from fnmatch import fnmatch

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.component_model import ComponentModel
from param_decomp.components import LinearComponents
from param_decomp.distributed import all_reduce
from param_decomp.masks import ComponentsMaskInfo

SiteErrors = dict[str, tuple[Float[Tensor, ""], Float[Tensor, ""]]]
"""Per site: `(summed squared error, summed squared target)`.

Numerator and denominator stay separate all the way through accumulation and the DDP
reduction: the relative error is a ratio of sums over the whole eval pass, never a mean of
per-batch or per-rank ratios.
"""


def select_sites(all_sites: list[str], patterns: list[str] | None) -> list[str]:
    """Sites matching any fnmatch pattern, or all of them when `patterns` is `None`.

    Patterns keep model-specific site names (`"*.mlp.down_proj"`) out of the core code
    while still allowing measurement to be restricted to, say, the residual-stream writes.
    """
    if patterns is None:
        return list(all_sites)
    selected = [site for site in all_sites if any(fnmatch(site, p) for p in patterns)]
    assert selected, f"site_patterns {patterns} matched none of {all_sites}"
    return selected


def clean_site_outputs(
    model: ComponentModel,
    pre_weight_acts: dict[str, Float[Tensor, "... d_in"]],
    sites: list[str],
) -> dict[str, Float[Tensor, "..."]]:
    """Frozen per-site outputs `x @ W.T + b`, recomputed from the cached clean input acts.

    These are what the target model itself produces at each site, so they cost no extra
    forward pass — the clean `cache_type="input"` pass every step already runs has all the
    inputs. Detached: targets, never a gradient path.
    """
    targets: dict[str, Float[Tensor, ...]] = {}
    for site in sites:
        components = model.components[site]
        assert isinstance(components, LinearComponents), (
            f"hidden-acts reconstruction supports linear sites only, got "
            f"{type(components).__name__} for {site}"
        )
        w = model.target_weight(site)
        targets[site] = F.linear(pre_weight_acts[site].detach().to(w.dtype), w, components.bias)
    return targets


def site_squared_errors(
    site_outputs: dict[str, Float[Tensor, "..."]],
    targets: dict[str, Float[Tensor, "..."]],
    mask_infos: dict[str, ComponentsMaskInfo],
) -> SiteErrors:
    """Squared error and squared target mass per site, over positions routed to components.

    Positions not routed to components ran the frozen module untouched, so their error is
    identically zero; including them would only dilute the ratio.
    """
    out: SiteErrors = {}
    for site, target in targets.items():
        predicted = site_outputs[site]
        assert predicted.shape == target.shape, f"{site}: {predicted.shape} vs {target.shape}"
        routing_mask = mask_infos[site].routing_mask
        if isinstance(routing_mask, Tensor):
            print(
                f"DEBUG site={site} routing_mask.shape={tuple(routing_mask.shape)} "
                f"routing_mask.sum={routing_mask.sum().item()} predicted.shape={tuple(predicted.shape)} "
                f"target.shape={tuple(target.shape)} target.abs().max()={target.abs().max().item()}",
                flush=True,
            )
            predicted = predicted[routing_mask]
            target = target[routing_mask]
        # Upcast before subtracting: under bf16 autocast these two are close and large, so
        # a bf16 difference discards most of the significant bits, and a bf16 reduction over
        # ~1e7 elements is worthless besides. `copy=True` is load-bearing — plain `.float()`
        # returns `predicted` itself when it is already fp32, and the in-place `sub_` would
        # then corrupt a tensor the graph still needs. Subtracting in place into that fresh
        # copy keeps only one fp32 buffer live per site.
        out[site] = (
            predicted.to(torch.float32, copy=True).sub_(target).pow_(2).sum(),
            target.float().pow(2).sum(),
        )
    return out


def add_site_errors(accum: SiteErrors, new: SiteErrors) -> None:
    """In place: add `new`'s numerators and denominators into `accum`."""
    for site, (sq_err, sq_target) in new.items():
        if site in accum:
            prev_err, prev_target = accum[site]
            accum[site] = (prev_err + sq_err, prev_target + sq_target)
        else:
            accum[site] = (sq_err, sq_target)


def detached_site_errors(errors: SiteErrors) -> SiteErrors:
    """Copy with both entries detached, for accumulators that outlive the autograd graph."""
    return {site: (err.detach(), target.detach()) for site, (err, target) in errors.items()}


def mean_relative_error(errors: SiteErrors) -> Float[Tensor, ""]:
    """Mean over sites of `Σ(out - tgt)² / Σ tgt²`.

    Relative per site so that sites with wildly different activation scales (an MLP
    `down_proj` against an attention `q_proj`) weigh equally, and so the coefficient
    transfers across blocks. A site whose clean output is identically zero over the routed
    positions would divide by zero; the trainer's `isfinite` assertion on every loss
    catches that without a per-step device sync here.
    """
    assert errors, "no sites measured"
    return torch.stack([sq_err / sq_target for sq_err, sq_target in errors.values()]).mean()


def reduced_relative_errors(errors: SiteErrors, name: str) -> dict[str, Float[Tensor, ""]]:
    """DDP-reduced per-site relative errors plus their mean, keyed under `name`.

    Both halves of every ratio are reduced in one collective and only then divided, so the
    result is the ratio over all ranks' data rather than an average of per-rank ratios.
    """
    assert errors, "no sites accumulated"
    sites = sorted(errors)  # sorted so every rank issues identically ordered collectives
    stacked = torch.stack([torch.stack(errors[site]) for site in sites])  # [n_sites, 2]
    reduced = all_reduce(stacked, op=ReduceOp.SUM)
    denominators = reduced[:, 1]
    # Once per eval pass, so this device sync is free in context (unlike inside the PGD loop).
    assert (denominators > 0).all(), (
        "clean site output is identically zero for "
        f"{[s for s, d in zip(sites, denominators.tolist(), strict=True) if d <= 0]}"
    )
    per_site = reduced[:, 0] / denominators
    out: dict[str, Float[Tensor, ""]] = {name: per_site.mean()}
    for i, site in enumerate(sites):
        out[f"{name}/{site}"] = per_site[i]
    return out
