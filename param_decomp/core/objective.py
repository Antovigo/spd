"""The closed VPD training objectives — plain and targeted.

Authored loss metrics become explicit objective roles: the plain objective is exactly one
faithfulness term, one importance-minimality term, a non-empty ordered tuple of recon
terms, and at most one nonlinearity-locality term; the targeted (tPD, SPEC §11)
objective is a faithfulness-free target-pass surface plus a directly-authored
non-target pass (delta-pinned recon + importance-minimality at its own coefficient).
The recon vocabulary (routing samplers, mask-source strategies) lives in `recon.py`;
this module alone composes it with the other objective roles.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from jaxtyping import Array

from param_decomp.core.components import ComponentStacks, SiteSpec, nonlinearity_partitions
from param_decomp.core.configs import (
    AllRoutingConfig,
    AnyLossMetricConfig,
    AnyReconLossMetricConfig,
    CIMaskedReconLossConfig,
    CIMaskedReconSubsetLossConfig,
    FaithfulnessLossConfig,
    HiddenPassConfig,
    ImportanceMinimalityLossConfig,
    LossCoeff,
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
from param_decomp.core.losses import coeff_at, nonlinearity_loss, scheduled_value_at
from param_decomp.core.nonlinearity import NonlinearityPartition, NonlinearityUnitKind
from param_decomp.core.recon import (
    AnyReconLossTerm,
    ConstantSources,
    FreshPGDSources,
    MaskSourceStrategy,
    MixedPersistentStochasticSources,
    PersistentSources,
    ReconLossTerm,
    StochasticSources,
    UnmaskedNoDeltaSources,
    routing_sampler_from_config,
)


@dataclass(frozen=True)
class FaithfulnessTerm:
    name: str
    coeff: LossCoeff


@dataclass(frozen=True)
class ImportanceMinimalityTerm:
    """CI-space importance-minimality plus optional frequency penalty."""

    name: str
    coeff: LossCoeff
    cfg: ImportanceMinimalityLossConfig


@dataclass(frozen=True)
class NonlinearityTerm:
    """Weight-space concentration over authored nonlinearity-facing output units."""

    name: str
    coeff: LossCoeff
    cfg: NonlinearityLocalityLossConfig


@dataclass(frozen=True)
class ResolvedNonlinearity:
    """A `NonlinearityTerm` joined with the target's declared partitions: the closed
    unit-kind check happens at `resolve`, once, and `None`-weighted kinds are filtered
    out here so no loss math ever sees an excluded kind."""

    term: NonlinearityTerm
    trained_partitions: dict[str, NonlinearityPartition]
    kind_coefficients: dict[NonlinearityUnitKind, float]

    @staticmethod
    def resolve(term: NonlinearityTerm, sites: tuple[SiteSpec, ...]) -> "ResolvedNonlinearity":
        partitions = nonlinearity_partitions(sites)
        assert partitions, "NonlinearityLocalityLoss needs a partitioned site"
        declared_kinds = {p.unit_kind for p in partitions.values()}
        authored = term.cfg.unit_kind_coefficients
        assert authored.keys() == declared_kinds, (
            f"unit_kind_coefficients must name exactly the target's partitioned kinds: "
            f"authored {sorted(authored)}, declared {sorted(declared_kinds)}"
        )
        kind_coefficients: dict[NonlinearityUnitKind, float] = {
            kind: w for kind, w in authored.items() if w is not None
        }
        return ResolvedNonlinearity(
            term,
            {name: p for name, p in partitions.items() if p.unit_kind in kind_coefficients},
            kind_coefficients,
        )

    def weighted_loss_and_metrics(
        self, train_frac: Array, components: ComponentStacks
    ) -> tuple[Array, dict[str, Array]]:
        """The term's coefficient-weighted step value plus its ready-to-log metrics."""
        threshold = scheduled_value_at(train_frac, self.term.cfg.relative_threshold)
        value, by_kind = nonlinearity_loss(
            components, self.trained_partitions, threshold, self.kind_coefficients
        )
        metrics: dict[str, Array] = {
            f"loss/{self.term.name}": value,
            "nonlinearity_relative_threshold": threshold,
            **{f"loss/{self.term.name}_{kind}": v for kind, v in by_kind.items()},
        }
        return coeff_at(train_frac, self.term.coeff) * value, metrics


@dataclass(frozen=True)
class LossSurface:
    """`L = c·faith + c·importance + Σ c·recon [+ c·nonlinearity]`."""

    faith: FaithfulnessTerm
    imp: ImportanceMinimalityTerm
    recon: tuple[AnyReconLossTerm, ...]
    nonlinearity: NonlinearityTerm | None


@dataclass(frozen=True)
class TargetPass:
    """The tPD target-pass surface (SPEC T3/T7): the full decomposition objective minus
    faithfulness — the delta is the off-target escape valve and must never be penalized,
    so a targeted objective has no faithfulness role at all."""

    imp: ImportanceMinimalityTerm
    recon: tuple[AnyReconLossTerm, ...]


NontargetClosedSources = StochasticSources | ConstantSources | UnmaskedNoDeltaSources
"""The CLOSED non-target strategy set (SPEC T5) — the non-adversarial arms of both
non-target passes."""

NontargetHiddenSources = NontargetClosedSources | MixedPersistentStochasticSources
"""The non-target HIDDEN pass's strategies (T5/T12 amended 2026-08-20): the closed set
plus the MERGED strategy only — the one adversarial arm that rides an existing masked
forward (S34), so it costs the broad stream nothing. The standalone persistent bundle
stays unrepresentable here (it would be an extra forward per step)."""

NontargetOutputSources = NontargetHiddenSources | PersistentSources
"""The non-target OUTPUT pass's strategies (T5/T7 amended 2026-08-19): the closed set
plus the persistent adversarial pair, whose masks the step composes DELTA-PINNED (T4 —
the bundle's delta channel is ignored)."""


@dataclass(frozen=True)
class NontargetPass:
    """The tPD non-target-pass surface, complete (SPEC T4/T5): with the delta mask pinned
    fully on, the broad stream judges only what the components must not disturb — so its
    whole objective is delta-pinned reconstruction against the frozen output plus
    importance-minimality at its own coefficient (T4's one enumerated exception is the
    unmasked-no-delta term, whose delta is pinned OFF). `imp.cfg` IS the target pass's
    config (penalty shape, anneal, frequency block shared by construction); only the
    coefficient is the non-target pass's own."""

    recon: tuple[ReconLossTerm[NontargetOutputSources], ...]
    """The non-target OUTPUT vocabulary in the type (SPEC T5, amended 2026-08-19): the
    delta-pinned stochastic/constant pair, the delta-off unmasked arm, and the two
    persistent adversarial strategies — every arm delta-pinned per T4, the persistent
    bundles' delta channels ignored. Fresh-PGD stays unrepresentable off-target."""
    impmin_coeff: LossCoeff
    """The non-target pass's importance-minimality COEFFICIENT — the penalty config
    (shape, anneal, frequency block) is the target pass's, structurally: this pass
    cannot carry its own (SPEC T6)."""


@dataclass(frozen=True)
class HiddenPass[S: MaskSourceStrategy]:
    """The HIDDEN role's surface on one stream (SPEC T12).

    Structurally a pass like any other — importance-minimality at its own coefficient plus a
    recon grid — differing in exactly two ways: its masks come from the CI fn's SECOND readout
    head (S37), and its recon comparison is `HiddenActsOnlyReconstruction(points)` rather than
    the model output, so it carries no end-to-end term at all.

    `points` is the pass's, not each term's: the pass IS "reconstruct these activations". The
    imp-min CONFIG (penalty shape, anneal, frequency block) is the target pass's, shared by
    construction — T6's rule extended to the hidden role; only the coefficient is this pass's.

    Generic over `S` so the non-target stream's hidden pass keeps T5's narrowing in the
    TYPE: the closed set plus the merged strategy only (T5/T12 amended 2026-08-20) — a
    standalone persistent bundle stays representable on the non-target OUTPUT pass alone."""

    recon: tuple[ReconLossTerm[S], ...]
    impmin_coeff: LossCoeff
    points: tuple[str, ...]


@dataclass(frozen=True)
class TargetedObjective:
    """The complete tPD objective. Every pass sums into ONE gradient (SPEC T1) — whether that
    is one `value_and_grad` or a sequence of per-pass ones whose gradients are added is a
    scheduling choice with identical arithmetic (`runtime.sequential_passes`).

    Up to FOUR passes, the cartesian product of {target, non-target} x {output, hidden}. The
    two hidden passes are `None` in an ordinary single-objective tPD run, which is then exactly
    the pre-T12 objective."""

    target: TargetPass
    nontarget: NontargetPass
    hidden: HiddenPass[MaskSourceStrategy] | None = None
    nontarget_hidden: HiddenPass[NontargetHiddenSources] | None = None
    nonlinearity: NonlinearityTerm | None = None
    """The optional weight-space concentration prior (SPEC S36). Deliberately NOT a member
    of any pass: it reads `U` alone — no CI, no activations, no stream — so it is scored
    once per step and added once to the total, whatever the pass count."""

    def __post_init__(self) -> None:
        assert self.nontarget_hidden is None or self.hidden is not None, (
            "a non-target hidden pass needs the target-stream hidden pass, whose `points` it "
            "measures at (SPEC T12)"
        )

    @property
    def hidden_points(self) -> tuple[str, ...]:
        """The activations BOTH hidden passes reconstruct; empty when there is no hidden pass."""
        return () if self.hidden is None else self.hidden.points


def _collect_terms(
    loss_metrics: Sequence[AnyLossMetricConfig],
    site_names: tuple[str, ...],
) -> tuple[
    FaithfulnessTerm | None,
    ImportanceMinimalityTerm | None,
    tuple[AnyReconLossTerm, ...],
    NonlinearityTerm | None,
]:
    """One pass over an authored loss list into its objective roles, names unique across
    all roles. Completeness (which roles must be present) is each objective builder's own
    claim, not this walk's.

    Recon-term order follows the authored list and is semantically load-bearing: per-term
    RNG keys derive from the recon index (SPEC R1).
    """
    faith: FaithfulnessTerm | None = None
    imp: ImportanceMinimalityTerm | None = None
    recon_terms: list[AnyReconLossTerm] = []
    nonlinearity: NonlinearityTerm | None = None

    def unique_name(cfg: AnyLossMetricConfig) -> str:
        # Only committed terms are in `taken`, so persistent terms may call this once for
        # their state key and again inside `recon` without colliding with themselves.
        name = cfg.name if cfg.name is not None else cfg.type
        taken = {term.name for term in recon_terms}
        if faith is not None:
            taken.add(faith.name)
        if imp is not None:
            taken.add(imp.name)
        if nonlinearity is not None:
            taken.add(nonlinearity.name)
        assert name not in taken, f"duplicate loss instance_key {name!r}"
        return name

    def recon(
        cfg: AnyReconLossMetricConfig,
        routing: SubsetRoutingType,
        sources: MaskSourceStrategy,
        n_samples: int,
    ) -> AnyReconLossTerm:
        # `sources` sits in parameter position, so the term is built width-erased
        # directly (storage is width-erased; narrower widths are the builders' concern).
        assert cfg.coeff is not None
        return ReconLossTerm(
            unique_name(cfg),
            cfg.coeff,
            routing_sampler_from_config(routing, site_names, n_samples),
            sources,
            cfg.hidden_acts_reconstruction,
        )

    for cfg in loss_metrics:
        assert cfg.coeff is not None, f"{cfg.type}: training losses need a coeff"
        match cfg:
            case FaithfulnessLossConfig():
                assert faith is None
                faith = FaithfulnessTerm(unique_name(cfg), cfg.coeff)
            case ImportanceMinimalityLossConfig():
                assert imp is None
                assert all(k.frac > 0 for k in cfg.gamma.points), (
                    f"gamma knots must all keep frac > 0, got {cfg.gamma.points}: a zero "
                    "width collapses the smooth-L0 threshold band the gradient lives on"
                )
                imp = ImportanceMinimalityTerm(unique_name(cfg), cfg.coeff, cfg)
            case UnmaskedReconLossConfig():
                recon_terms.append(
                    recon(cfg, AllRoutingConfig(), ConstantSources(1.0), n_samples=1)
                )
            case (
                CIMaskedReconLossConfig()
                | CIMaskedReconSubsetLossConfig()
                | StochasticReconLossConfig()
                | StochasticReconSubsetLossConfig()
            ):
                routing, sources, n_samples = _nontarget_recon_parts(cfg)
                recon_terms.append(recon(cfg, routing, sources, n_samples))
            case PGDReconLossConfig() | PGDReconSubsetLossConfig():
                sources = FreshPGDSources(cfg.init, cfg.n_steps, cfg.step_size, cfg.source_shape)
                routing = (
                    cfg.routing if isinstance(cfg, PGDReconSubsetLossConfig) else AllRoutingConfig()
                )
                recon_terms.append(recon(cfg, routing, sources, n_samples=1))
            case MergedStochasticSubsetPPGDReconLossConfig():
                key = unique_name(cfg)
                sources = MixedPersistentStochasticSources(state_key=key, cfg=cfg)
                recon_terms.append(recon(cfg, cfg.routing, sources, n_samples=1))
            case PersistentPGDReconLossConfig():
                key = unique_name(cfg)
                sources = PersistentSources(state_key=key, cfg=cfg)
                recon_terms.append(recon(cfg, AllRoutingConfig(), sources, n_samples=1))
            case NonlinearityLocalityLossConfig():
                assert nonlinearity is None
                nonlinearity = NonlinearityTerm(unique_name(cfg), cfg.coeff, cfg)

    return faith, imp, tuple(recon_terms), nonlinearity


def build_objective(
    loss_metrics: Sequence[AnyLossMetricConfig],
    site_names: tuple[str, ...],
) -> LossSurface:
    """Build the closed plain-VPD objective, rejecting incomplete authored surfaces."""
    faith, imp, recon_terms, nonlinearity = _collect_terms(loss_metrics, site_names)
    assert faith is not None and imp is not None, (
        f"need FaithfulnessLoss + ImportanceMinimalityLoss, got {[m.type for m in loss_metrics]}"
    )
    assert recon_terms, "no recon loss terms configured"
    return LossSurface(faith, imp, recon_terms, nonlinearity)


def build_recon_terms(
    loss_metrics: Sequence[AnyLossMetricConfig],
    site_names: tuple[str, ...],
    hidden: HiddenPassConfig | None = None,
) -> tuple[AnyReconLossTerm, ...]:
    """Just the recon Σ of an authored loss list — the persistent-source layout derives
    from these (`recon.persistent_configs`), so state init shares this walk with both
    objective builders instead of demanding one builder's completeness rules.

    A hidden pass's terms are included: they carry adversaries of their own (T7 keeps
    adversaries on the target STREAM, and the hidden pass is a target-stream pass), so their
    persistent bundles must be allocated alongside the output pass's or the state would be
    missing the keys the step asks for."""
    terms = _collect_terms(loss_metrics, site_names)[2]
    if hidden is not None:
        terms = terms + _collect_terms(hidden.recon, site_names)[2]
    return terms


def build_targeted_objective(
    loss_metrics: Sequence[TargetedLossMetricConfig],
    nontarget: NontargetConfig,
    site_names: tuple[str, ...],
    hidden: HiddenPassConfig | None = None,
) -> TargetedObjective:
    """Build the closed two-pass tPD objective (SPEC §11).

    `loss_metrics` authors the TARGET pass — typed by `TargetedLossMetricConfig`, which
    has no faithfulness member (T3: the delta is the unpenalized off-target escape valve,
    so a targeted config cannot spell a faithfulness role). The non-target pass is
    authored directly on `nontarget` — never derived from the target list — and its
    importance-minimality shares the target's penalty config (shape + anneal) by
    construction, at the non-target pass's own coefficient."""
    faith, imp, recon_terms, nonlinearity = _collect_terms(loss_metrics, site_names)
    # The library boundary for lists built outside the schema; unreachable for a parsed
    # TargetedPDConfig.
    assert faith is None, "a targeted loss list carried a FaithfulnessLossConfig (SPEC T3)"
    assert imp is not None, (
        f"need an ImportanceMinimalityLoss, got {[m.type for m in loss_metrics]}"
    )
    assert imp.cfg.frequency is None or imp.cfg.frequency.ema_halflife_steps is None, (
        "frequency.ema_halflife_steps is not implemented for the targeted (tPD) objective "
        "(SPEC S8'' — plain PD only; the TargetedPDConfig validator carries the why)"
    )
    assert recon_terms, "no recon loss terms configured"

    nt_terms = build_nontarget_output_terms(nontarget.recon, site_names)

    hidden_pass = None
    nontarget_hidden_pass = None
    if hidden is not None:
        # The hidden pass's terms go through the SAME walk as the target pass's: a hidden recon
        # term is an ordinary recon term whose comparison happens to be internal activations,
        # so the mask-source algebra (stochastic, PPGD, mixed) is unchanged. What makes it
        # hidden is the pass it lives in — the step reads its mask off the hidden CI head and
        # scores it against `points` (T12).
        h_faith, h_imp, h_recon, h_nonlinearity = _collect_terms(hidden.recon, site_names)
        assert h_nonlinearity is None, (
            "the nonlinearity prior is weight-space and pass-less (SPEC S36): author it once "
            "on the target loss list, never on a hidden pass"
        )
        assert h_faith is None and h_imp is None, (
            "a hidden pass authors recon terms only: its importance-minimality is the "
            "`impmin_coeff` scalar (the penalty config is the target pass's, T6/T12)"
        )
        assert h_recon, "hidden pass has no recon terms"
        hidden_pass = HiddenPass(
            recon=h_recon, impmin_coeff=hidden.impmin_coeff, points=hidden.points
        )
        if nontarget.hidden is not None:
            nontarget_hidden_pass = HiddenPass(
                recon=build_nontarget_hidden_terms(nontarget.hidden.recon, site_names),
                impmin_coeff=nontarget.hidden.impmin_coeff,
                points=hidden.points,
            )
    else:
        assert nontarget.hidden is None, (
            "nontarget.hidden needs pd.hidden — the non-target hidden pass measures at the "
            "target-stream hidden pass's `points` (SPEC T12)"
        )

    return TargetedObjective(
        target=TargetPass(imp=imp, recon=recon_terms),
        nontarget=NontargetPass(recon=nt_terms, impmin_coeff=nontarget.impmin_coeff),
        hidden=hidden_pass,
        nontarget_hidden=nontarget_hidden_pass,
        nonlinearity=nonlinearity,
    )


def build_nontarget_output_terms(
    cfgs: Sequence[NontargetOutputReconLossMetricConfig], site_names: tuple[str, ...]
) -> tuple[ReconLossTerm[NontargetOutputSources], ...]:
    """Build the non-target OUTPUT pass's recon terms (SPEC T5/T7 amended 2026-08-19):
    the closed non-target vocabulary plus the persistent adversarial types, whose masks
    the step composes delta-pinned (T4 — the bundle's delta channel is ignored). Public
    because state init shares this walk: each bundle here sizes off the BROAD stream's
    geometry (T2), so `init_train_state` must see the same state keys the step will ask
    for. The `nontarget/` key prefix keeps the global adversary namespace disjoint from
    the target-stream passes', whose keys are bare loss names."""
    terms: list[ReconLossTerm[NontargetOutputSources]] = []
    for cfg in cfgs:
        assert cfg.coeff is not None  # non-None at parse (NontargetConfig); narrows the type
        name = cfg.name if cfg.name is not None else cfg.type
        assert name not in {t.name for t in terms}, f"duplicate non-target loss {name!r}"
        sources: NontargetOutputSources
        routing: SubsetRoutingType
        match cfg:
            case PersistentPGDReconLossConfig():
                routing = AllRoutingConfig()
                sources = PersistentSources(state_key=f"nontarget/{name}", cfg=cfg)
                n_samples = 1
            case MergedStochasticSubsetPPGDReconLossConfig():
                routing = cfg.routing
                sources = MixedPersistentStochasticSources(state_key=f"nontarget/{name}", cfg=cfg)
                n_samples = 1
            case _:
                # Width-erased rebuild: dataclass type params are invariant, so the
                # closed-set family widens explicitly through the annotations above.
                routing, sources, n_samples = _nontarget_recon_parts(cfg)
        terms.append(
            ReconLossTerm(
                name,
                cfg.coeff,
                routing_sampler_from_config(routing, site_names, n_samples),
                sources,
                None,
            )
        )
    return tuple(terms)


def build_nontarget_hidden_terms(
    cfgs: Sequence[NontargetHiddenReconLossMetricConfig], site_names: tuple[str, ...]
) -> tuple[ReconLossTerm[NontargetHiddenSources], ...]:
    """Build the non-target HIDDEN pass's recon terms (T5/T12 amended 2026-08-20): the
    closed vocabulary plus the zero-extra-forward MERGED term, delta-pinned like every
    non-target forward (T4). Public because state init shares this walk (as it does the
    output pass's): a merged bundle here sizes off the BROAD stream's geometry (T2). The
    `nontarget_hidden/` key prefix keeps the namespace disjoint from the output pass's
    `nontarget/` and the target-stream passes' bare names."""
    terms: list[ReconLossTerm[NontargetHiddenSources]] = []
    for cfg in cfgs:
        assert cfg.coeff is not None  # non-None at parse (NontargetHiddenConfig); narrows the type
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
                n_samples = 1
            case _:
                # Width-erased rebuild, as `build_nontarget_output_terms` does.
                routing, sources, n_samples = _nontarget_recon_parts(cfg)
        # `hidden_acts_reconstruction=None` structurally: the S35 rider is refused on every
        # non-target term at parse — a hidden pass's points are the pass's own (T5/T12).
        terms.append(
            ReconLossTerm(
                name,
                cfg.coeff,
                routing_sampler_from_config(routing, site_names, n_samples),
                sources,
                None,
            )
        )
    return tuple(terms)


def _nontarget_recon_parts(
    cfg: NontargetReconLossMetricConfig,
) -> tuple[SubsetRoutingType, StochasticSources | ConstantSources | UnmaskedNoDeltaSources, int]:
    """The `(routing, sources, n_samples)` family of one recon config the non-target pass
    admits (SPEC T5): the stochastic/constant-source types — shared verbatim with the
    plain objective's arms, which widen through `recon` — plus the non-target-only
    unmasked-no-delta term (T4's one delta-off exception)."""
    match cfg:
        case CIMaskedReconLossConfig():
            return AllRoutingConfig(), ConstantSources(0.0), 1
        case CIMaskedReconSubsetLossConfig():
            return cfg.routing, ConstantSources(0.0), 1
        case StochasticReconLossConfig():
            return AllRoutingConfig(), StochasticSources(), cfg.n_mask_samples
        case StochasticReconSubsetLossConfig():
            return cfg.routing, StochasticSources(), cfg.n_mask_samples
        case UnmaskedNoDeltaReconLossConfig():
            return AllRoutingConfig(), UnmaskedNoDeltaSources(), 1
