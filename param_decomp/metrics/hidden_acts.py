"""Shared pieces of the per-site (hidden-activation) reconstruction error.

Used by the `StochasticHiddenReconSubsetLoss` training loss and by the lab-side
hidden-acts eval probes, so all of them measure the same quantity.
"""

from fnmatch import fnmatch
from typing import Any, Literal

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

SiteInputs = Literal["clean", "masked_forward"]
"""Which input each measured site's components are run on.

`"masked_forward"` runs the model with every decomposed site replaced, so a site downstream
of another receives an input the upstream replacement has already perturbed: its error is
its own approximation error *plus* whatever it inherited. `"clean"` hands every site the
input the frozen model gave it, so each site's error is its own alone — which also makes the
sites independent, and therefore needs no forward pass at all.

Both compare against the same frozen targets (`clean_site_outputs`), so the two are
subtractable: `masked_forward - clean` is the inherited (compounding) part of the error.
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


def resolve_measured_sites(
    model: ComponentModel,
    patterns: list[str] | None,
    site_inputs: SiteInputs,
) -> list[str]:
    """`select_sites` plus the restriction that `"clean"` cannot measure a readout site.

    A readout site's target is a point in the residual stream, whose value is a property of
    the whole chain rather than of any one matrix. Hand every matrix its clean input and the
    stream is unchanged by construction, so the readout's error would be identically zero —
    a silently meaningless measurement, hence the assert.
    """
    sites = select_sites(model.measurement_sites, patterns)
    if site_inputs == "clean":
        readouts = [site for site in sites if site in model.hidden_readout_sites]
        assert not readouts, (
            f"site_inputs='clean' cannot measure readout sites {sorted(readouts)}: their "
            "error is identically zero when every matrix is fed its clean input. Restrict "
            "them to a separate site_inputs='masked_forward' instance via site_patterns."
        )
    return sites


def masked_site_outputs(
    model: ComponentModel,
    batch: Any,
    pre_weight_acts: dict[str, Float[Tensor, "... d_in"]],
    mask_infos: dict[str, ComponentsMaskInfo],
    sites: list[str],
    site_inputs: SiteInputs,
) -> dict[str, Float[Tensor, "..."]]:
    """Masked outputs at `sites`, from either the chained forward or each site's clean input.

    Masking always covers every decomposed site — `mask_infos` is unfiltered — while only
    `sites` are returned, since those are the ones an error is read at.
    """
    match site_inputs:
        case "masked_forward":
            cache = model.site_outputs(batch, mask_infos)
            return {site: cache[site] for site in sites}
        case "clean":
            return _local_site_outputs(model, pre_weight_acts, mask_infos, sites)


def _local_site_outputs(
    model: ComponentModel,
    pre_weight_acts: dict[str, Float[Tensor, "... d_in"]],
    mask_infos: dict[str, ComponentsMaskInfo],
    sites: list[str],
) -> dict[str, Float[Tensor, "..."]]:
    """Each site's components run on its own cached clean input — no forward pass.

    With every site fed the frozen model's own input to it, no site depends on any other, so
    there is nothing to run *through*: this is a pair of matmuls per site over tensors the
    step's `cache_type="input"` pass already produced.

    Positions the routing mask excludes are computed and then dropped by
    `site_squared_errors`, exactly as in the chained path, where they are likewise scored
    only over routed positions.
    """
    outputs: dict[str, Float[Tensor, ...]] = {}
    for site in sites:
        components = model.components[site]
        assert isinstance(components, LinearComponents), (
            f"hidden-acts reconstruction supports linear sites only, got "
            f"{type(components).__name__} for {site}"
        )
        mask_info = mask_infos[site]
        outputs[site] = components(
            pre_weight_acts[site],
            mask=mask_info.component_mask,
            weight_delta_and_mask=mask_info.weight_delta_and_mask,
        )
    return outputs


def clean_site_outputs(
    model: ComponentModel,
    pre_weight_acts: dict[str, Float[Tensor, "... d_in"]],
    sites: list[str],
) -> dict[str, Float[Tensor, "..."]]:
    """Frozen per-site targets, recomputed from the cached clean input acts.

    A decomposed site's target is its own output `x @ W.T + b`; a readout site's target is
    the captured clean tensor itself, which the same `cache_type="input"` pass already put
    in `pre_weight_acts`. Either way this costs no extra forward pass. Detached: targets,
    never a gradient path.
    """
    targets: dict[str, Float[Tensor, ...]] = {}
    for site in sites:
        if site in model.hidden_readout_sites:
            targets[site] = pre_weight_acts[site].detach()
            continue
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

    A readout site has no routing mask and is measured over every position. It must be:
    attention mixes positions, so a position routed to nothing still sees error arriving
    from the routed positions it attends to, and restricting would discard it.
    """
    out: SiteErrors = {}
    for site, target in targets.items():
        predicted = site_outputs[site]
        assert predicted.shape == target.shape, f"{site}: {predicted.shape} vs {target.shape}"
        routing_mask = mask_infos[site].routing_mask if site in mask_infos else "all"
        if isinstance(routing_mask, Tensor):
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
