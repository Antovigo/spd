"""The closed VPD training objectives — plain and targeted.

Authored loss metrics become explicit objective roles: the plain objective is exactly one
faithfulness term, one CI-minimality term, a non-empty ordered tuple of recon
terms, and at most one nonlinearity-locality term; the targeted (tPD)
objective is a faithfulness-free target-pass surface plus a directly-authored
non-target pass (delta-pinned recon + minimality with its own activity coefficient).
The recon vocabulary (routing samplers, mask-source strategies) lives in `recon.py`;
this module alone composes it with the other objective roles.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple, Protocol

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float32

from param_decomp.core.ci_fn.interface import CI
from param_decomp.core.components import ComponentStacks, SiteSpec, nonlinearity_alignments
from param_decomp.core.configs import (
    AllRoutingConfig,
    AnyLossMetricConfig,
    AnyReconLossMetricConfig,
    CIMaskedReconLossConfig,
    CIMaskedReconSubsetLossConfig,
    FaithfulnessLossConfig,
    HiddenActsNormalization,
    HiddenPassConfig,
    ImportanceMinimalityLossConfig,
    LossCoeff,
    MergedStochasticSubsetPooledPPGDReconLossConfig,
    MergedStochasticSubsetPPGDReconLossConfig,
    NonlinearityLocalityLossConfig,
    NontargetConfig,
    NontargetHiddenReconLossMetricConfig,
    NontargetOutputReconLossMetricConfig,
    NontargetReconLossMetricConfig,
    PersistentPGDReconLossConfig,
    PGDReconLossConfig,
    PGDReconSubsetLossConfig,
    StochasticReconLossConfig,
    StochasticReconSubsetLossConfig,
    SubsetRoutingType,
    TargetedLossMetricConfig,
    UnmaskedNoDeltaReconLossConfig,
    UnmaskedReconLossConfig,
)
from param_decomp.core.losses import (
    FrequencyEstimator,
    FrequencyPenalty,
    activity_sum,
    activity_sum_from_ci,
    nonlinearity_loss,
    resolve_reference_datapoint_count,
)
from param_decomp.core.nonlinearity import NonlinearityAlignment, NonlinearityUnitKind
from param_decomp.core.recon import (
    AnyReconLossTerm,
    AuxiliaryReconstructionTerm,
    ConstantSources,
    FreshPGDSources,
    MaskSourceStrategy,
    MixedPersistentStochasticSources,
    PersistentSourcePool,
    PersistentSources,
    ReconLossTerm,
    StochasticSources,
    UnmaskedNoDeltaSources,
    routing_sampler_from_config,
)
from param_decomp.core.runtime_schedule import RuntimeSchedule, scheduled_value_at
from param_decomp.core.schedule import ScheduleConfig


class FaithfulnessTerm(eqx.Module):
    name: str = eqx.field(static=True)
    coeff: RuntimeSchedule


class ComponentFrequencies(Protocol):
    def __call__(
        self, ci: CI, gamma: Float32[Array, ""], *, normalize_at_one: bool
    ) -> dict[str, Float32[Array, " _"]]: ...


class MinimalityResult(NamedTuple):
    weighted_loss: Float32[Array, ""]
    activity: Float32[Array, ""]
    frequency: FrequencyPenalty | None
    gamma: Float32[Array, ""]


class FrequencyMinimalityTerm(eqx.Module):
    coeff: RuntimeSchedule
    reference_datapoint_count: int | Literal["auto"] = eqx.field(static=True)


class MinimalityTerm(eqx.Module):
    """CI activity and an optional component-frequency penalty."""

    name: str = eqx.field(static=True)
    activity_coeff: RuntimeSchedule
    gamma: ScheduleConfig = eqx.field(static=True)
    normalize_at_one: bool = eqx.field(static=True)
    frequency: FrequencyMinimalityTerm | None


def evaluate_minimality(
    term: MinimalityTerm,
    estimator: FrequencyEstimator,
    ci: CI,
    train_frac: Float32[Array, ""],
    component_frequencies: ComponentFrequencies,
) -> tuple[MinimalityResult, FrequencyEstimator]:
    gamma = scheduled_value_at(train_frac, term.gamma)
    match term.frequency:
        case None:
            # Selected CI needs no component scatter when only the activity is read.
            activity = activity_sum_from_ci(ci.upper, gamma, normalize_at_one=term.normalize_at_one)
            frequency = None
            updated_estimator = estimator
            frequency_loss = jnp.zeros((), jnp.float32)
        case FrequencyMinimalityTerm() as frequency_term:
            frequencies = component_frequencies(ci, gamma, normalize_at_one=term.normalize_at_one)
            activity = activity_sum(frequencies)
            frequency, updated_estimator = estimator.evaluate(
                frequencies,
                resolve_reference_datapoint_count(
                    ci.upper, frequency_term.reference_datapoint_count
                ),
            )
            frequency_loss = frequency_term.coeff.at(train_frac) * frequency.value
    return MinimalityResult(
        term.activity_coeff.at(train_frac) * activity + frequency_loss, activity, frequency, gamma
    ), updated_estimator


class NonlinearityResult(NamedTuple):
    weighted_loss: Float32[Array, ""]
    loss: Float32[Array, ""]
    by_kind: dict[NonlinearityUnitKind, Float32[Array, ""]]
    relative_threshold: Float32[Array, ""]


class NonlinearityTerm(eqx.Module):
    """Weight-space concentration over declared nonlinearity-facing units."""

    name: str = eqx.field(static=True)
    coeff: RuntimeSchedule
    relative_threshold: ScheduleConfig = eqx.field(static=True)
    unit_kind_coefficients: dict[NonlinearityUnitKind, float | None] = eqx.field(static=True)
    normalize_at_one: bool = eqx.field(static=True)


@dataclass(frozen=True)
class ResolvedNonlinearity:
    """A `NonlinearityTerm` joined with the target's declared partitions: the closed
    unit-kind check happens at `resolve`, once, and `None`-weighted kinds are filtered
    out here so no loss math ever sees an excluded kind."""

    term: NonlinearityTerm
    trained_alignments: dict[str, NonlinearityAlignment]
    kind_coefficients: dict[NonlinearityUnitKind, float]

    @staticmethod
    def resolve(term: NonlinearityTerm, sites: tuple[SiteSpec, ...]) -> "ResolvedNonlinearity":
        alignments = nonlinearity_alignments(sites)
        assert alignments, "NonlinearityLocalityLoss needs a partitioned site"
        declared_kinds = {a.partition.unit_kind for a in alignments.values()}
        authored = term.unit_kind_coefficients
        assert authored.keys() == declared_kinds, (
            f"unit_kind_coefficients must name exactly the target's partitioned kinds: "
            f"authored {sorted(authored)}, declared {sorted(declared_kinds)}"
        )
        kind_coefficients: dict[NonlinearityUnitKind, float] = {
            kind: w for kind, w in authored.items() if w is not None
        }
        return ResolvedNonlinearity(
            term,
            {
                name: alignment
                for name, alignment in alignments.items()
                if alignment.partition.unit_kind in kind_coefficients
            },
            kind_coefficients,
        )

    def evaluate(
        self, train_frac: Float32[Array, ""], components: ComponentStacks
    ) -> NonlinearityResult:
        threshold = scheduled_value_at(train_frac, self.term.relative_threshold)
        value, by_kind = nonlinearity_loss(
            components,
            self.trained_alignments,
            threshold,
            self.kind_coefficients,
            normalize_at_one=self.term.normalize_at_one,
        )
        return NonlinearityResult(self.term.coeff.at(train_frac) * value, value, by_kind, threshold)


class PDObjective(eqx.Module):
    """Faithfulness, CI minimality, reconstruction, and optional nonlinearity losses."""

    faith: FaithfulnessTerm
    minimality: MinimalityTerm
    recon: tuple[AnyReconLossTerm, ...]
    nonlinearity: NonlinearityTerm | None


class TargetPass(eqx.Module):
    """Target-stream reconstruction and CI-minimality losses.

    Faithfulness is excluded so the delta remains free to carry off-target behavior.
    Delta sources follow plain VPD: stochastic terms sample uniformly on [0, 1],
    and adversarial terms ascend and project their delta channel over [0, 1].
    This makes reconstruction from components alone an adversarial worst case
    while also covering hybrid ablations of components and delta; pinning the
    target delta off would omit those hybrids."""

    minimality: MinimalityTerm
    recon: tuple[AnyReconLossTerm, ...]


NontargetClosedSources = StochasticSources | ConstantSources | UnmaskedNoDeltaSources
"""The CLOSED non-target strategy set — the non-adversarial arms of both non-target passes."""

NontargetHiddenSources = NontargetClosedSources | MixedPersistentStochasticSources
"""The non-target HIDDEN pass's strategies (local amendment 2026-08-20): the closed set plus
the MERGED strategy only — the one adversarial arm that rides an existing masked forward,
so it costs the broad stream nothing. The standalone persistent bundle stays
unrepresentable here (it would be an extra forward per step)."""

NontargetOutputSources = NontargetHiddenSources | PersistentSources
"""The non-target OUTPUT pass's strategies (local amendment 2026-08-19): the closed set plus
the persistent adversarial pair, whose masks the step composes DELTA-PINNED (the bundle's
delta channel is ignored)."""


class NontargetPass(eqx.Module):
    """Broad-stream reconstruction and CI-minimality terms.

    Stochastic and constant-source terms keep the weight delta on; the unmasked
    term keeps it off and scores the full component sum alone; the persistent and merged
    arms compose their bundle's component channels with the delta pinned on. Minimality
    uses the target pass's penalty configuration with this pass's own activity
    coefficient."""

    recon: tuple[ReconLossTerm[NontargetOutputSources], ...]
    """Fresh-PGD stays unrepresentable off-target."""
    minimality: MinimalityTerm


class HiddenPass[S: MaskSourceStrategy](eqx.Module):
    """The HIDDEN role's surface on one stream.

    Structurally a pass like any other — importance-minimality at its own coefficient plus a
    recon grid — differing in exactly two ways: its masks come from the CI fn's SECOND
    readout head, and its recon comparison is the point mean at `points` rather than the
    model output (`recon.HiddenReconstruction`), so it carries no end-to-end term unless
    `output_coeff` adds one.

    `points` is the pass's, not each term's: the pass IS "reconstruct these activations".
    The minimality's penalty config (shape, anneal, frequency block) is the target pass's,
    shared by construction; only the activity coefficient is this pass's.

    Generic over `S` so the non-target stream's hidden pass keeps its narrowing in the TYPE:
    the closed set plus the merged strategy only."""

    minimality: MinimalityTerm
    recon: tuple[ReconLossTerm[S], ...]
    points: tuple[str, ...] = eqx.field(static=True)
    normalization: HiddenActsNormalization = eqx.field(static=True)
    """How each point pools over positions — the pass's, shared with the non-target hidden
    pass exactly like `points`."""
    output_coeff: RuntimeSchedule | None
    """The optional e2e KL rider; `None` is the pure hidden pass. Per pass, not shared."""
    log_alternate_normalization: bool = eqx.field(static=True, default=False)
    """Diagnostic: also log each point under the other normalization (loss untouched)."""


class TargetedPDObjective(eqx.Module):
    """The complete tPD objective. Every pass sums into ONE gradient — whether that is one
    `value_and_grad` or a sequence of per-pass ones whose gradients are added is a
    scheduling choice with identical arithmetic (`runtime.sequential_passes`).

    Up to FOUR passes, the cartesian product of {target, non-target} x {output, hidden}. The
    two hidden passes are `None` in an ordinary single-objective tPD run, which is then
    exactly the two-pass objective."""

    target: TargetPass
    nontarget: NontargetPass
    hidden: HiddenPass[MaskSourceStrategy] | None = None
    nontarget_hidden: HiddenPass[NontargetHiddenSources] | None = None
    nonlinearity: NonlinearityTerm | None = None
    """The optional weight-space concentration prior. Deliberately NOT a member of any pass:
    it reads `U` alone — no CI, no activations, no stream — so it is scored once per step and
    added once to the total, whatever the pass count."""

    def __check_init__(self) -> None:
        assert self.nontarget_hidden is None or self.hidden is not None, (
            "a non-target hidden pass needs the target-stream hidden pass, whose `points` it "
            "measures at"
        )


def _minimality_term(
    cfg: ImportanceMinimalityLossConfig, name: str, activity_coeff: LossCoeff
) -> MinimalityTerm:
    assert all(k.frac > 0 for k in cfg.gamma.points), (
        f"gamma knots must all keep frac > 0, got {cfg.gamma.points}: a zero "
        "width collapses the smooth-L0 threshold band the gradient lives on"
    )
    return MinimalityTerm(
        name,
        RuntimeSchedule.from_coeff(activity_coeff),
        cfg.gamma,
        cfg.normalize_at_one,
        FrequencyMinimalityTerm(
            RuntimeSchedule.from_coeff(cfg.frequency.coeff),
            cfg.frequency.reference_datapoint_count,
        )
        if cfg.frequency is not None
        else None,
    )


def _collect_terms(
    loss_metrics: Sequence[AnyLossMetricConfig],
    sites: tuple[SiteSpec, ...],
) -> tuple[
    FaithfulnessTerm | None,
    MinimalityTerm | None,
    tuple[AnyReconLossTerm, ...],
    NonlinearityTerm | None,
]:
    """One pass over an authored loss list into its objective roles, names unique across
    all roles. Completeness (which roles must be present) is each objective builder's own
    claim, not this walk's.

    Recon-term order follows the authored list and is semantically load-bearing: per-term
    RNG keys derive from the recon index.
    """
    faith: FaithfulnessTerm | None = None
    minimality: MinimalityTerm | None = None
    recon_terms: list[AnyReconLossTerm] = []
    nonlinearity: NonlinearityTerm | None = None

    def unique_name(cfg: AnyLossMetricConfig) -> str:
        # Only committed terms are in `taken`, so persistent terms may call this once for
        # their state key and again inside `recon` without colliding with themselves.
        name = cfg.name if cfg.name is not None else cfg.type
        taken = {term.name for term in recon_terms}
        if faith is not None:
            taken.add(faith.name)
        if minimality is not None:
            taken.add(minimality.name)
        if nonlinearity is not None:
            taken.add(nonlinearity.name)
        assert name not in taken, f"duplicate loss instance_key {name!r}"
        return name

    def recon(
        cfg: AnyReconLossMetricConfig,
        routing: SubsetRoutingType,
        sources: MaskSourceStrategy,
    ) -> AnyReconLossTerm:
        # `sources` sits in parameter position, so the term is built width-erased
        # directly (storage is width-erased; narrower widths are the builders' concern).
        return ReconLossTerm(
            unique_name(cfg),
            RuntimeSchedule.from_coeff(cfg.coeff),
            routing_sampler_from_config(routing, tuple(site.name for site in sites)),
            sources,
            tuple(
                AuxiliaryReconstructionTerm(
                    auxiliary.name,
                    RuntimeSchedule.from_coeff(auxiliary.coeff),
                    auxiliary.comparisons,
                )
                for auxiliary in cfg.auxiliaries
            ),
        )

    for cfg in loss_metrics:
        match cfg:
            case FaithfulnessLossConfig():
                assert faith is None
                faith = FaithfulnessTerm(unique_name(cfg), RuntimeSchedule.from_coeff(cfg.coeff))
            case ImportanceMinimalityLossConfig():
                assert minimality is None
                minimality = _minimality_term(cfg, unique_name(cfg), cfg.coeff)
            case UnmaskedReconLossConfig():
                recon_terms.append(recon(cfg, AllRoutingConfig(), ConstantSources(1.0)))
            case (
                CIMaskedReconLossConfig()
                | CIMaskedReconSubsetLossConfig()
                | StochasticReconLossConfig()
                | StochasticReconSubsetLossConfig()
            ):
                routing, sources = _nontarget_recon_parts(cfg)
                recon_terms.append(recon(cfg, routing, sources))
            case PGDReconLossConfig() | PGDReconSubsetLossConfig():
                sources = FreshPGDSources(cfg.init, cfg.n_steps, cfg.step_size, cfg.source_shape)
                routing = (
                    cfg.routing if isinstance(cfg, PGDReconSubsetLossConfig) else AllRoutingConfig()
                )
                recon_terms.append(recon(cfg, routing, sources))
            case MergedStochasticSubsetPPGDReconLossConfig():
                key = unique_name(cfg)
                sources = MixedPersistentStochasticSources(state_key=key, cfg=cfg)
                recon_terms.append(recon(cfg, cfg.routing, sources))
            case MergedStochasticSubsetPooledPPGDReconLossConfig():
                key = unique_name(cfg)
                sources = PersistentSourcePool(state_key=key, cfg=cfg)
                recon_terms.append(recon(cfg, cfg.routing, sources))
            case PersistentPGDReconLossConfig():
                key = unique_name(cfg)
                sources = PersistentSources(state_key=key, cfg=cfg)
                recon_terms.append(recon(cfg, AllRoutingConfig(), sources))
            case NonlinearityLocalityLossConfig():
                assert nonlinearity is None
                nonlinearity = NonlinearityTerm(
                    unique_name(cfg),
                    RuntimeSchedule.from_coeff(cfg.coeff),
                    cfg.relative_threshold,
                    cfg.unit_kind_coefficients,
                    cfg.normalize_at_one,
                )

    return faith, minimality, tuple(recon_terms), nonlinearity


def build_objective(
    loss_metrics: Sequence[AnyLossMetricConfig],
    sites: tuple[SiteSpec, ...],
) -> PDObjective:
    """Build the closed plain-VPD objective, rejecting incomplete authored surfaces."""
    faith, minimality, recon_terms, nonlinearity = _collect_terms(loss_metrics, sites)
    assert faith is not None and minimality is not None, (
        f"need FaithfulnessLoss + ImportanceMinimalityLoss, got {[m.type for m in loss_metrics]}"
    )
    assert recon_terms, "no recon loss terms configured"
    return PDObjective(faith, minimality, recon_terms, nonlinearity)


def build_recon_terms(
    loss_metrics: Sequence[AnyLossMetricConfig],
    sites: tuple[SiteSpec, ...],
    hidden: HiddenPassConfig | None = None,
) -> tuple[AnyReconLossTerm, ...]:
    """Just the recon Σ of an authored loss list — the persistent-source layout derives
    from these (`recon.persistent_configs`), so state init shares this walk with both
    objective builders instead of demanding one builder's completeness rules.

    A target-stream hidden pass's terms are included: they carry adversaries of their own,
    sized off the target stream like the output pass's, so their persistent bundles must be
    allocated alongside the output pass's or the state would be missing the keys the step
    asks for."""
    terms = _collect_terms(loss_metrics, sites)[2]
    if hidden is not None:
        terms = terms + _collect_terms(hidden.recon, sites)[2]
    return terms


def build_targeted_objective(
    loss_metrics: Sequence[TargetedLossMetricConfig],
    nontarget: NontargetConfig,
    sites: tuple[SiteSpec, ...],
    hidden: HiddenPassConfig | None = None,
) -> TargetedPDObjective:
    """Build the closed two-pass tPD objective.

    `loss_metrics` authors the TARGET pass — typed by `TargetedLossMetricConfig`, which
    has no faithfulness member (the delta is the unpenalized off-target escape valve,
    so a targeted config cannot spell a faithfulness role). The non-target pass is
    authored directly on `nontarget` — never derived from the target list — and its
    minimality shares the target's penalty config (shape + anneal) by construction,
    with the non-target pass's own activity coefficient."""
    faith, minimality, recon_terms, nonlinearity = _collect_terms(loss_metrics, sites)
    # The library boundary for lists built outside the schema; unreachable for a parsed
    # TargetedPDConfig.
    assert faith is None, "a targeted loss list carried a FaithfulnessLossConfig"
    assert minimality is not None, (
        f"need a ImportanceMinimalityLoss, got {[m.type for m in loss_metrics]}"
    )
    assert recon_terms, "no recon loss terms configured"

    (minimality_config,) = (
        cfg for cfg in loss_metrics if isinstance(cfg, ImportanceMinimalityLossConfig)
    )
    nt_minimality = _minimality_term(minimality_config, "nontarget/impmin", nontarget.impmin_coeff)
    nt_terms = build_nontarget_output_terms(nontarget.recon, sites)

    hidden_pass = None
    nontarget_hidden_pass = None
    if hidden is not None:
        # The hidden pass's terms go through the SAME walk as the target pass's: a hidden recon
        # term is an ordinary recon term whose comparison happens to be internal activations,
        # so the mask-source algebra (stochastic, PPGD, mixed) is unchanged. What makes it
        # hidden is the pass it lives in — the step reads its mask off the hidden CI head and
        # scores it against `points`.
        h_faith, h_minimality, h_recon, h_nonlinearity = _collect_terms(hidden.recon, sites)
        assert h_nonlinearity is None, (
            "the nonlinearity prior is weight-space and pass-less: author it once on the "
            "target loss list, never on a hidden pass"
        )
        assert h_faith is None and h_minimality is None, (
            "a hidden pass authors recon terms only: its importance-minimality is the "
            "`impmin_coeff` scalar (the penalty config is the target pass's)"
        )
        assert h_recon, "hidden pass has no recon terms"
        hidden_pass = HiddenPass(
            minimality=_minimality_term(minimality_config, "hidden/impmin", hidden.impmin_coeff),
            recon=h_recon,
            points=hidden.points,
            normalization=hidden.normalization,
            output_coeff=_optional_schedule(hidden.output_coeff),
            log_alternate_normalization=hidden.log_alternate_normalization,
        )
        if nontarget.hidden is not None:
            nontarget_hidden_pass = HiddenPass(
                minimality=_minimality_term(
                    minimality_config, "nontarget_hidden/impmin", nontarget.hidden.impmin_coeff
                ),
                recon=build_nontarget_hidden_terms(nontarget.hidden.recon, sites),
                points=hidden.points,
                normalization=hidden.normalization,
                output_coeff=_optional_schedule(nontarget.hidden.output_coeff),
                log_alternate_normalization=hidden.log_alternate_normalization,
            )
    else:
        assert nontarget.hidden is None, (
            "nontarget.hidden needs pd.hidden — the non-target hidden pass measures at the "
            "target-stream hidden pass's `points`"
        )
    return TargetedPDObjective(
        target=TargetPass(minimality=minimality, recon=recon_terms),
        nontarget=NontargetPass(minimality=nt_minimality, recon=nt_terms),
        hidden=hidden_pass,
        nontarget_hidden=nontarget_hidden_pass,
        nonlinearity=nonlinearity,
    )


def _optional_schedule(coeff: LossCoeff | None) -> RuntimeSchedule | None:
    return None if coeff is None else RuntimeSchedule.from_coeff(coeff)


def build_nontarget_output_terms(
    cfgs: Sequence[NontargetOutputReconLossMetricConfig], sites: tuple[SiteSpec, ...]
) -> tuple[ReconLossTerm[NontargetOutputSources], ...]:
    """Build the non-target OUTPUT pass's recon terms: the closed non-target vocabulary plus
    the persistent adversarial types, whose masks the step composes delta-pinned (the
    bundle's delta channel is ignored). Public because state init shares this walk: each
    bundle here sizes off the BROAD stream's geometry, so state init must see the same state
    keys the step will ask for. The `nontarget/` key prefix keeps the global adversary
    namespace disjoint from the target-stream passes', whose keys are bare loss names."""
    terms: list[ReconLossTerm[NontargetOutputSources]] = []
    site_names = tuple(site.name for site in sites)
    for cfg in cfgs:
        name = cfg.name if cfg.name is not None else cfg.type
        assert name not in {t.name for t in terms}, f"duplicate non-target loss {name!r}"
        sources: NontargetOutputSources
        routing: SubsetRoutingType
        match cfg:
            case PersistentPGDReconLossConfig():
                routing = AllRoutingConfig()
                sources = PersistentSources(state_key=f"nontarget/{name}", cfg=cfg)
            case MergedStochasticSubsetPPGDReconLossConfig():
                routing = cfg.routing
                sources = MixedPersistentStochasticSources(state_key=f"nontarget/{name}", cfg=cfg)
            case _:
                routing, sources = _nontarget_recon_parts(cfg)
        terms.append(
            ReconLossTerm(
                name,
                RuntimeSchedule.from_coeff(cfg.coeff),
                routing_sampler_from_config(routing, site_names),
                sources,
                (),
            )
        )
    return tuple(terms)


def build_nontarget_hidden_terms(
    cfgs: Sequence[NontargetHiddenReconLossMetricConfig], sites: tuple[SiteSpec, ...]
) -> tuple[ReconLossTerm[NontargetHiddenSources], ...]:
    """Build the non-target HIDDEN pass's recon terms: the closed vocabulary plus the
    zero-extra-forward MERGED term, delta-pinned like every non-target forward. Public
    because state init shares this walk (as it does the output pass's): a merged bundle here
    sizes off the BROAD stream's geometry. The `nontarget_hidden/` key prefix keeps the
    namespace disjoint from the output pass's `nontarget/` and the target-stream passes'
    bare names."""
    terms: list[ReconLossTerm[NontargetHiddenSources]] = []
    site_names = tuple(site.name for site in sites)
    for cfg in cfgs:
        name = cfg.name if cfg.name is not None else cfg.type
        assert name not in {t.name for t in terms}, f"duplicate non-target hidden loss {name!r}"
        sources: NontargetHiddenSources
        routing: SubsetRoutingType
        match cfg:
            case MergedStochasticSubsetPPGDReconLossConfig():
                routing = cfg.routing
                sources = MixedPersistentStochasticSources(
                    state_key=f"nontarget_hidden/{name}", cfg=cfg
                )
            case _:
                routing, sources = _nontarget_recon_parts(cfg)
        terms.append(
            ReconLossTerm(
                name,
                RuntimeSchedule.from_coeff(cfg.coeff),
                routing_sampler_from_config(routing, site_names),
                sources,
                (),
            )
        )
    return tuple(terms)


def _nontarget_recon_parts(
    cfg: NontargetReconLossMetricConfig,
) -> tuple[SubsetRoutingType, StochasticSources | ConstantSources | UnmaskedNoDeltaSources]:
    """Resolve a non-target reconstruction config into its routing and source strategy."""
    match cfg:
        case CIMaskedReconLossConfig():
            return AllRoutingConfig(), ConstantSources(0.0)
        case CIMaskedReconSubsetLossConfig():
            return cfg.routing, ConstantSources(0.0)
        case StochasticReconLossConfig():
            return AllRoutingConfig(), StochasticSources()
        case StochasticReconSubsetLossConfig():
            return cfg.routing, StochasticSources()
        case UnmaskedNoDeltaReconLossConfig():
            return AllRoutingConfig(), UnmaskedNoDeltaSources()
