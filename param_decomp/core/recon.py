"""Recon plans for the VPD objective.

A recon term describes a closed traversal over live-site groups, routing draws, and mask-source
strategies. This module knows nothing about the objective's faithfulness or importance terms;
`objective.py` composes those with the recon terms into the complete loss surface.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
from jax import random
from jax.sharding import Mesh
from jaxtyping import Array, Float, PRNGKeyArray

from param_decomp.core.configs import (
    AllRoutingConfig,
    HiddenActsReconstruction,
    LossCoeff,
    MergedStochasticSubsetPPGDReconLossConfig,
    PersistentPGDReconLossConfig,
    PGDInitStrategy,
    SourceShape,
    StaticProbabilityRoutingConfig,
    SubsetRoutingType,
    UniformKSubsetRoutingConfig,
)
from param_decomp.core.model import (
    CaptureKeys,
    DecomposedModel,
    ForwardResult,
    chunk_sites,
    select_captures,
)
from param_decomp.core.sharding import batch_shard_leading

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
class UnmaskedNoDeltaSources:
    """Every component mask `1.0`, every weight-delta mask `0.0` — the full component sum
    alone reconstructs. The tPD non-target pass's one delta-OFF arm (SPEC T4's enumerated
    exception): the polarity rides this type, never a flag on the delta-pinned strategies.
    Deterministic — no sources are drawn."""


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
    | UnmaskedNoDeltaSources
    | FreshPGDSources
    | PersistentSources
    | MixedPersistentStochasticSources
)


@dataclass(frozen=True)
class ReconForward[SourcesT: MaskSourceStrategy]:
    """One plan entry: which sites run their decomposed path (`live_sites` — everything
    else takes the frozen `x @ W` path, the ~9x-cheaper non-decomposed matmul), a
    sampler producing this entry's family of routing draws, and the strategy that
    generates each draw's mask/delta sources. `SourcesT` narrows which strategies a plan
    can carry — the tPD non-target pass's plans admit only the enumerated non-target
    strategies IN THE TYPE (SPEC T5). `uses_weight_deltas` (static,
    derived from the strategy) skips the `x @ Δ` matmul for constant-source entries."""

    live_sites: tuple[str, ...]
    sample_routing: RoutingSampler
    sources: SourcesT

    @property
    def uses_weight_deltas(self) -> bool:
        """`ConstantSources` carries no delta path (torch passes no `weight_deltas` for the
        Unmasked/CIMasked losses); its `delta_mask` would be a constant 0, so the `x @ Δ`
        matmul is skipped entirely (static, retrace-safe — LOSS_PARITY_DESIGN §4b).
        `UnmaskedNoDeltaSources` carries a materialized delta mask pinned to 0 (SPEC T4's
        exception); every other strategy drives a live delta mask."""
        return not isinstance(self.sources, ConstantSources)


type ReconPlan[SourcesT: MaskSourceStrategy] = tuple[ReconForward[SourcesT], ...]


@dataclass(frozen=True)
class OutputOnlyReconstruction:
    """A reconstruction specification that compares only the model output."""


@dataclass(frozen=True)
class OutputAndHiddenActsReconstruction:
    """A reconstruction specification that also compares named hidden activations.

    Value-level: `coeff` is the rider's strength AT one step — a literal for static
    contexts (eval probes), a traced scalar when the step resolved a schedule
    (`losses.reconstruction_spec_at`) — never a schedule object, so the loss math
    downstream stays step-blind."""

    coeff: Float[Array, ""] | float
    points: tuple[str, ...]


@dataclass(frozen=True)
class HiddenActsOnlyReconstruction:
    """A reconstruction specification that compares ONLY named hidden activations — no
    end-to-end output term at all (SPEC T12).

    This is the hidden pass's whole comparison, and it is why the hidden role is a PASS rather
    than S35's per-term rider: the rider adds `coeff · hidden` ON TOP of a term's e2e loss, so
    its CI is still shaped mostly by the output objective. The hidden pass instead asks its own
    question — "which subcomponents matter for reproducing these internal activations?" — and
    carries no output term to dilute it. There is no `coeff` here either: the pass's strength
    lives on its recon terms' own coefficients, one level up, exactly like the output pass's."""

    points: tuple[str, ...]


type ReconstructionSpec = (
    OutputOnlyReconstruction | OutputAndHiddenActsReconstruction | HiddenActsOnlyReconstruction
)


class ForwardObservations(NamedTuple):
    """The output and named activations one reconstruction comparison consumes."""

    output: Any
    hidden_acts_by_point: dict[str, Array]


def reconstruction_observations(
    result: ForwardResult,
    *,
    hidden_acts_capture_keys: CaptureKeys,
    mesh: Mesh | None,
) -> ForwardObservations:
    """Convert one forward result into the exact view a reconstruction consumes."""
    output = jax.tree.map(lambda value: batch_shard_leading(value, mesh), result.output)
    return ForwardObservations(
        output,
        select_captures(result.captures, hidden_acts_capture_keys),
    )


def resolve_reconstruction_spec(
    hidden_acts_reconstruction: HiddenActsReconstruction | None,
) -> ReconstructionSpec:
    """Resolve the authored optional into one explicit reconstruction specification —
    STATIC contexts only (eval probes): a scheduled rider coeff needs the step threaded
    in (`losses.reconstruction_spec_at`), which an eval probe deliberately lacks."""
    if hidden_acts_reconstruction is None:
        return OutputOnlyReconstruction()
    coeff = hidden_acts_reconstruction.coeff
    assert isinstance(coeff, float), (
        f"an eval probe's hidden-acts rider coeff must be a constant float, got {coeff}"
    )
    return OutputAndHiddenActsReconstruction(coeff, hidden_acts_reconstruction.points)


def hidden_acts_capture_keys(reconstruction: ReconstructionSpec) -> CaptureKeys:
    """Return the hidden activations required before evaluating this specification."""
    match reconstruction:
        case OutputOnlyReconstruction():
            return frozenset()
        case (
            OutputAndHiddenActsReconstruction(points=points)
            | HiddenActsOnlyReconstruction(points=points)
        ):
            return frozenset(points)


@dataclass(frozen=True)
class ReconLossTerm[SourcesT: MaskSourceStrategy]:
    """One coefficiented recon loss: mean over ALL draws of ALL plan entries of
    `kl_per_position` (SPEC S10'). `name` is the torch `instance_key` (`cfg.name` or
    the type literal) — the metric log key is `loss/<name>`. `coeff` and the S35 rider's
    coeff may be schedules, so the term stays a static description; the step resolves
    both to per-step values (`losses.coeff_at` / `losses.reconstruction_spec_at`)."""

    name: str
    coeff: LossCoeff
    plan: ReconPlan[SourcesT]
    hidden_acts_reconstruction: HiddenActsReconstruction | None

    @property
    def hidden_acts_capture_keys(self) -> CaptureKeys:
        match self.hidden_acts_reconstruction:
            case None:
                return frozenset()
            case HiddenActsReconstruction(points=points):
                return frozenset(points)


AnyReconLossTerm = ReconLossTerm[MaskSourceStrategy]
"""The width-erased term — what heterogeneous storage (the plain objective, `_StepAtoms`)
holds. Machinery that must preserve a narrower width is generic over `SourcesT` instead
(dataclass type params are invariant on 3.13: the synthesized `__replace__` puts them in
parameter position)."""


@dataclass(frozen=True)
class ResolvedReconstructionTerms:
    terms: tuple[AnyReconLossTerm, ...]
    hidden_acts_capture_keys_by_term: dict[str, CaptureKeys]
    hidden_acts_capture_keys: CaptureKeys
    persistent_term_by_key: dict[str, AnyReconLossTerm]


def index_persistent_terms(
    terms: tuple[AnyReconLossTerm, ...], *, into: dict[str, AnyReconLossTerm]
) -> None:
    """Index each term's persistent bundles by state key, into a table that may already hold
    other passes' (SPEC S23). Adversary state keys are a GLOBAL namespace — they index
    `TrainState.adversaries` — so a multi-pass objective accumulates into one table and the
    "one key feeds one term" invariant is checked across all of them.

    Exhaustive over the source union on purpose: a new mask-source strategy must be classified
    here deliberately rather than fall through a catch-all and silently lose its bundle."""
    for term in terms:
        for entry in term.plan:
            match entry.sources:
                case (
                    PersistentSources(state_key=state_key)
                    | MixedPersistentStochasticSources(state_key=state_key)
                ):
                    previous = into.setdefault(state_key, term)
                    assert previous is term, f"persistent source {state_key!r} feeds multiple terms"
                case (
                    StochasticSources()
                    | ConstantSources()
                    | UnmaskedNoDeltaSources()
                    | FreshPGDSources()
                ):
                    pass


def resolve_reconstruction_terms(
    model: DecomposedModel, terms: tuple[AnyReconLossTerm, ...]
) -> ResolvedReconstructionTerms:
    """Validate and index the static facts shared by every reconstruction draw."""
    hidden_acts_capture_keys_by_term = {term.name: term.hidden_acts_capture_keys for term in terms}
    hidden_acts_capture_keys = frozenset(
        key for term_keys in hidden_acts_capture_keys_by_term.values() for key in term_keys
    )
    model.assert_hidden_acts_reconstruction_points(tuple(sorted(hidden_acts_capture_keys)))

    persistent_term_by_key: dict[str, AnyReconLossTerm] = {}
    index_persistent_terms(terms, into=persistent_term_by_key)

    return ResolvedReconstructionTerms(
        terms, hidden_acts_capture_keys_by_term, hidden_acts_capture_keys, persistent_term_by_key
    )


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


def make_plan[SourcesT: MaskSourceStrategy](
    live_sets: list[LiveSet],
    routing: SubsetRoutingType,
    sources: SourcesT,
    n_samples: int,
) -> ReconPlan[SourcesT]:
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


def subset_chunk_plan[SourcesT: MaskSourceStrategy](
    site_names: tuple[str, ...],
    sites_per_chunk: int,
    n_samples: int,
    sources: SourcesT,
) -> ReconPlan[SourcesT]:
    """The production plan: `n_samples` uniform-k forwards per chunk (torch
    `SubsetReconPlan` over `ThreePoolTopology` chunks)."""
    return make_plan(
        live_groups(site_names, sites_per_chunk), UniformKSubsetRoutingConfig(), sources, n_samples
    )


# ───────────────────────────── shared-config -> flat terms ─────────────────────────────


def persistent_configs(
    recon_terms: tuple[AnyReconLossTerm, ...],
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
