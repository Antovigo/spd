"""Shared pieces of the per-site (hidden-activation) reconstruction error.

Used by the `StochasticHiddenReconSubsetLoss` training loss and by the lab-side
hidden-acts eval probes, so all of them measure the same quantity.
"""

from fnmatch import fnmatch
from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.base_config import BaseConfig
from param_decomp.component_model import ComponentModel
from param_decomp.components import LinearComponents
from param_decomp.distributed import all_reduce
from param_decomp.masks import ComponentsMaskInfo
from param_decomp.metrics.context import MetricContext

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


class HiddenActsSitesConfig(BaseConfig):
    """Where a hidden-activation metric measures, and on which inputs.

    Mixed into every metric in the family so the pair is declared once. `site_patterns`
    (fnmatch, e.g. `["*.mlp.down_proj"]`) filters `ComponentModel.measurement_sites`;
    `None` measures all of them. Masking always covers every decomposed site regardless —
    only measurement is filtered.
    """

    site_patterns: list[str] | None = None
    site_inputs: SiteInputs = "masked_forward"


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


def resolve_measured_sites(model: ComponentModel, cfg: HiddenActsSitesConfig) -> list[str]:
    """`select_sites` plus the two restrictions `"clean"` carries.

    A readout site's target is a point in the residual stream, whose value is a property of
    the whole chain rather than of any one matrix, so feeding every matrix its clean input
    leaves it unchanged and its error identically zero. Silently measuring nothing is the
    dangerous outcome, hence the assert.

    Every measured site must also be linear, which `clean_site_outputs` needs for the target
    anyway; failing here reports it at bind rather than on the first step.
    """
    sites = select_sites(model.measurement_sites, cfg.site_patterns)
    if cfg.site_inputs == "clean":
        readouts = [site for site in sites if site in model.hidden_readout_sites]
        assert not readouts, (
            f"site_inputs='clean' cannot measure readout sites {sorted(readouts)}: their "
            "error is identically zero when every matrix is fed its clean input. Restrict "
            "this instance to the decomposed sites with site_patterns, and read the "
            "readouts from a separate site_inputs='masked_forward' instance."
        )
    for site in sites:
        if site in model.hidden_readout_sites:
            continue
        components = model.components[site]
        assert isinstance(components, LinearComponents), (
            f"hidden-acts reconstruction supports linear sites only, got "
            f"{type(components).__name__} for {site}"
        )
    return sites


def assert_sources_reach_every_site(model: ComponentModel, sites: list[str], why: str) -> None:
    """Guard the PGD metrics against `"clean"` leaving an adversarial source out of the graph.

    PGD and PPGD allocate one adversarial source per *decomposed* site and differentiate the
    objective with respect to all of them at once (`allow_unused=False`). Chained, a source
    at an unmeasured site still reaches the loss by perturbing measured sites downstream of
    it. Locally the sites are independent, so a source at a site this instance does not
    measure reaches nothing and `torch.autograd.grad` raises mid-step.

    Called by the PGD metrics only: the stochastic and CI-masked probes never differentiate
    with respect to sources, so they are free to narrow `site_patterns` under `"clean"`.
    """
    unmeasured = [path for path in model.target_module_paths if path not in set(sites)]
    assert not unmeasured, (
        f"{why} measures {sorted(sites)} under site_inputs='clean', leaving decomposed "
        f"sites {sorted(unmeasured)} with adversarial sources that cannot reach the loss. "
        "Widen site_patterns to cover every decomposed site, or use "
        "site_inputs='masked_forward' for this instance."
    )


def masked_site_outputs(
    ctx: MetricContext,
    mask_infos: dict[str, ComponentsMaskInfo],
    sites: list[str],
    site_inputs: SiteInputs,
) -> dict[str, Float[Tensor, "..."]]:
    """Masked outputs at `sites`, from either the chained forward or each site's clean input.

    Masking always covers every decomposed site — `mask_infos` is unfiltered — while only
    `sites` are returned, since those are the ones an error is read at.

    The `"clean"` branch runs no forward: with every site handed the frozen model's own input
    to it, no site depends on any other, so this is a pair of matmuls per site over tensors
    the step's `cache_type="input"` pass already produced. Positions the routing mask excludes
    are computed and then dropped by `site_squared_errors`, exactly as in the chained path,
    where they are likewise scored only over routed positions.
    """
    match site_inputs:
        case "masked_forward":
            cache = ctx.model.site_outputs(ctx.batch, mask_infos)
            return {site: cache[site] for site in sites}
        case "clean":
            return {
                site: ctx.model.components[site](
                    ctx.pre_weight_acts[site],
                    mask=mask_infos[site].component_mask,
                    weight_delta_and_mask=mask_infos[site].weight_delta_and_mask,
                )
                for site in sites
            }


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
