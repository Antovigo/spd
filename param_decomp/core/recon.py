"""Recon plans for the VPD objective.

A recon term describes a closed traversal over live-site groups, routing draws, and mask-source
strategies. This module knows nothing about the objective's faithfulness or importance terms;
`objective.py` composes those with the recon terms into the complete loss surface.
"""

from collections.abc import Callable
from dataclasses import dataclass

from jax import random
from jaxtyping import Array, PRNGKeyArray

from param_decomp.core.configs import (
    AllRoutingConfig,
    MergedStochasticSubsetPPGDReconLossConfig,
    PersistentPGDReconLossConfig,
    PGDInitStrategy,
    SourceShape,
    StaticProbabilityRoutingConfig,
    SubsetRoutingType,
    UniformKSubsetRoutingConfig,
)
from param_decomp.core.model import chunk_sites

Routes = dict[str, Array] | None
RoutingSampler = Callable[[PRNGKeyArray, tuple[int, ...]], tuple[Routes, ...]]
"""`(key, leading_shape) -> (routes, ...)` — a STATICALLY-sized family of routing draws,
each `{site: bool[*leading]}` (or None = route everywhere) becoming ONE forward. The torch
`Router.get_masks` made pure: fresh draws per step require the key threaded in —
samplers run INSIDE the jitted step, so they must be traceable (SPEC R1). Returning
several draws from one invocation enables JOINTLY-sampled families (independent
repeats, antithetic/complementary subsets, per-step random covers) that duplicated
plan entries with independent keys cannot express. The plan's structure — live-sets,
sampler identities, family sizes — is static; only the key varies per step."""


# ───────────────────────────── mask-source strategies ─────────────────────────────


@dataclass(frozen=True)
class StochasticSources:
    """Fresh per-draw sources: components `U[0,1]`, delta `U[0,1]`."""


@dataclass(frozen=True)
class ConstantSources:
    """`mask = ci + (1-ci)*value`: 0.0 = CI-masked, 1.0 = unmasked. `delta_mask = 0`
    (torch passes no delta path at all for these; multiplying by zero is
    mathematically identical, it just pays the delta matmul)."""

    value: float


@dataclass(frozen=True)
class FreshPGDSources:
    """Per-step sign-PGD-ascended sources (torch `PGDRecon*` as TRAINING losses): init
    per `init`, `n_steps` of `step_size * sign(grad)` with clamp to [0,1], no state
    across steps. The entry's routing is drawn ONCE per step and shared by every
    ascent and the final loss forward (SPEC S24, torch parity)."""

    init: PGDInitStrategy
    n_steps: int
    step_size: float
    source_shape: SourceShape


@dataclass(frozen=True)
class PersistentSources:
    """Sources living in `TrainState.adversaries[state_key]` across steps (PPGD). Carries
    the shared `PersistentPGDReconLossConfig` so the term is self-describing — the step
    reads its scope/optimizer/warmup straight off `cfg`. `state_key` indexes
    `TrainState.adversaries` (one key per persistent term, SPEC S23)."""

    state_key: str
    cfg: PersistentPGDReconLossConfig


@dataclass(frozen=True)
class MixedPersistentStochasticSources:
    """The merged stochastic+PPGD strategy: per batch element, the persistent bundle's
    sources (probability `cfg.adv_fraction`, routed all-live) or fresh `U[0,1]` (routed
    per the entry's sampler). `state_key` indexes `TrainState.adversaries` like
    `PersistentSources`."""

    state_key: str
    cfg: "MergedStochasticSubsetPPGDReconLossConfig"


MaskSourceStrategy = (
    StochasticSources
    | ConstantSources
    | FreshPGDSources
    | PersistentSources
    | MixedPersistentStochasticSources
)


@dataclass(frozen=True)
class ReconForward:
    """One plan entry: which sites run their decomposed path (`live_sites` — everything
    else takes the frozen `x @ W` path, the ~9x-cheaper non-decomposed matmul), a
    sampler producing this entry's family of routing draws, and the strategy that
    generates each draw's mask/delta sources. `has_delta` (static, derived from the
    strategy) skips the `x @ Δ` matmul for constant-source entries."""

    live_sites: tuple[str, ...]
    sample_routing: RoutingSampler
    sources: MaskSourceStrategy

    @property
    def has_delta(self) -> bool:
        """`ConstantSources` carries no delta path (torch passes no `weight_deltas` for the
        Unmasked/CIMasked losses); its `delta_mask` would be a constant 0, so the `x @ Δ`
        matmul is skipped entirely (static, retrace-safe — LOSS_PARITY_DESIGN §4b). Every
        other strategy drives a live delta mask."""
        return not isinstance(self.sources, ConstantSources)


ReconPlan = tuple[ReconForward, ...]


@dataclass(frozen=True)
class ReconLossTerm:
    """One coefficiented recon loss: mean over ALL draws of ALL plan entries of
    `kl_per_position` (SPEC S10'). `name` is the torch `instance_key` (`cfg.name` or
    the type literal) — the metric log key is `loss/<name>`."""

    name: str
    coeff: float
    plan: ReconPlan


# ───────────────────────────── routing samplers ─────────────────────────────


def uniform_k_subset_routes(
    key: PRNGKeyArray, live_sites: tuple[str, ...], leading_shape: tuple[int, ...]
) -> dict[str, Array]:
    """Per position: `k ~ U{1..|live|}`, then a uniform k-subset of the live sites
    routes True (SPEC S11). Distributionally identical to torch's double-argsort ranks."""
    n_sites = len(live_sites)
    k_key, perm_key = random.split(key)
    k = random.randint(k_key, leading_shape, 1, n_sites + 1)
    perms = random.uniform(perm_key, (n_sites, *leading_shape)).argsort(axis=0)
    routed = perms < k
    return {name: routed[j] for j, name in enumerate(live_sites)}


def uniform_k_routing(live_sites: tuple[str, ...], n_draws: int) -> RoutingSampler:
    """`n_draws` independent per-position uniform-k-subset draws over `live_sites`."""

    def sample(key: PRNGKeyArray, leading_shape: tuple[int, ...]) -> tuple[Routes, ...]:
        return tuple(
            uniform_k_subset_routes(draw_key, live_sites, leading_shape)
            for draw_key in random.split(key, n_draws)
        )

    return sample


def static_probability_routing(
    live_sites: tuple[str, ...], p: float, n_draws: int
) -> RoutingSampler:
    """`n_draws` independent draws routing each position to each live site with
    probability `p` (torch `StaticProbabilityRouter`)."""

    def sample(key: PRNGKeyArray, leading_shape: tuple[int, ...]) -> tuple[Routes, ...]:
        return tuple(
            {
                name: random.bernoulli(random.fold_in(draw_key, j), p, leading_shape)
                for j, name in enumerate(live_sites)
            }
            for draw_key in random.split(key, n_draws)
        )

    return sample


def route_all_n(n_draws: int) -> RoutingSampler:
    """`n_draws` forwards, each routing every position to every live site (`AllRoutingConfig`)."""

    def sample(_key: PRNGKeyArray, _leading_shape: tuple[int, ...]) -> tuple[Routes, ...]:
        return (None,) * n_draws

    return sample


def routing_sampler_from_config(
    routing: SubsetRoutingType, live_sites: tuple[str, ...], n_draws: int
) -> RoutingSampler:
    match routing:
        case UniformKSubsetRoutingConfig():
            return uniform_k_routing(live_sites, n_draws)
        case StaticProbabilityRoutingConfig():
            return static_probability_routing(live_sites, routing.p, n_draws)
        case AllRoutingConfig():
            return route_all_n(n_draws)


# ───────────────────────────── live-set helpers ─────────────────────────────

LiveSet = tuple[str, ...]
"""The sites that run their decomposed path in one forward, everything else on the
frozen `x@W` path (SPEC S2)."""


def all_sites_live(sites: tuple[str, ...]) -> list[LiveSet]:
    """The whole model live in a single forward (torch `all_sites`)."""
    return [tuple(sites)]


def each_site_live(sites: tuple[str, ...]) -> list[LiveSet]:
    """One single-site live-set per site (torch `*Layerwise`)."""
    return [(s,) for s in sites]


def live_groups(sites: tuple[str, ...], k: int) -> list[LiveSet]:
    """Sequential size-`k` groups in canonical site order (torch chunkwise; SPEC S10)."""
    return list(chunk_sites(sites, k))


# ───────────────────────────── the plan constructor ─────────────────────────────


def make_plan(
    live_sets: list[LiveSet],
    routing: SubsetRoutingType,
    sources: MaskSourceStrategy,
    n_samples: int,
) -> ReconPlan:
    """One `ReconForward` per live-set: those sites live, the rest frozen `x@W` (SPEC S2),
    with `n_samples` routing draws from `routing` over the live-set's own sites (SPEC S11)
    and the shared `sources`. The live-set choice (`all_sites_live`/`each_site_live`/
    `live_groups`) and the routing/source choices are orthogonal — see LOSS_PARITY_DESIGN.md."""
    return tuple(
        ReconForward(
            live_sites=live_set,
            sample_routing=routing_sampler_from_config(routing, live_set, n_samples),
            sources=sources,
        )
        for live_set in live_sets
    )


def subset_chunk_plan(
    site_names: tuple[str, ...],
    sites_per_chunk: int,
    n_samples: int,
    sources: MaskSourceStrategy,
) -> ReconPlan:
    """The production plan: `n_samples` uniform-k forwards per chunk (torch
    `SubsetReconPlan` over `ThreePoolTopology` chunks)."""
    return make_plan(
        live_groups(site_names, sites_per_chunk), UniformKSubsetRoutingConfig(), sources, n_samples
    )


# ───────────────────────────── shared-config -> flat terms ─────────────────────────────


def persistent_configs(
    recon_terms: tuple[ReconLossTerm, ...],
) -> "dict[str, PersistentPGDReconLossConfig | MergedStochasticSubsetPPGDReconLossConfig]":
    """`state_key -> config` for every persistent-source-carrying recon term (SPEC S23:
    each key feeds exactly one term). Derived from the terms, not stored separately — the
    config rides each `PersistentSources` / `MixedPersistentStochasticSources` strategy;
    both carry the same adversary fields (optimizer/scope/source_dtype/n_warmup_steps)."""
    out: dict[str, PersistentPGDReconLossConfig | MergedStochasticSubsetPPGDReconLossConfig] = {}
    for term in recon_terms:
        for entry in term.plan:
            if isinstance(entry.sources, (PersistentSources, MixedPersistentStochasticSources)):
                assert entry.sources.state_key not in out, entry.sources.state_key
                out[entry.sources.state_key] = entry.sources.cfg
    return out
