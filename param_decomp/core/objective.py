"""The closed VPD training objectives — plain and targeted.

Authored loss metrics become explicit objective roles: the plain objective is exactly one
faithfulness term, one importance-minimality term, and a non-empty ordered tuple of recon
terms; the targeted (tPD, SPEC §11) objective is a faithfulness-free target-pass surface
plus a directly-authored non-target pass (delta-pinned recon + importance-minimality at
its own coefficient). Recon planning lives in `recon.py`; this module alone composes those
plans with the other objective roles.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from param_decomp.core.configs import (
    AllRoutingConfig,
    AnyImportanceMinimalityLossConfig,
    AnyLossMetricConfig,
    AnyReconLossMetricConfig,
    ChunkwiseSubsetReconLossConfig,
    CIMaskedReconLayerwiseLossConfig,
    CIMaskedReconLossConfig,
    CIMaskedReconSubsetLossConfig,
    FaithfulnessLossConfig,
    HiddenPassConfig,
    ImportanceMinimalityLossConfig,
    LossCoeff,
    MergedStochasticSubsetPPGDReconLossConfig,
    NontargetConfig,
    NontargetOutputReconLossMetricConfig,
    NontargetReconLossMetricConfig,
    PersistentPGDReconLossConfig,
    PGDReconLayerwiseLossConfig,
    PGDReconLossConfig,
    PGDReconSubsetLossConfig,
    SmoothL0ImportanceMinimalityLossConfig,
    StochasticReconLayerwiseLossConfig,
    StochasticReconLossConfig,
    StochasticReconSubsetLossConfig,
    SubsetRoutingType,
    TargetedLossMetricConfig,
    UnmaskedNoDeltaReconLossConfig,
    UnmaskedReconLossConfig,
)
from param_decomp.core.recon import (
    AnyReconLossTerm,
    ConstantSources,
    FreshPGDSources,
    LiveSet,
    MaskSourceStrategy,
    MixedPersistentStochasticSources,
    PersistentSources,
    ReconForward,
    ReconLossTerm,
    ReconPlan,
    StochasticSources,
    UnmaskedNoDeltaSources,
    all_sites_live,
    each_site_live,
    live_groups,
    make_plan,
)


@dataclass(frozen=True)
class FaithfulnessTerm:
    """Weight-space term: `Σ_s ‖Δ_s‖² / Σ_s numel`."""

    name: str
    coeff: LossCoeff


@dataclass(frozen=True)
class ImportanceMinimalityTerm:
    """CI-space importance-minimality plus optional frequency penalty."""

    name: str
    coeff: LossCoeff
    cfg: AnyImportanceMinimalityLossConfig


@dataclass(frozen=True)
class LossSurface:
    """`L = c·faith + c·importance + Σ c·recon`, with every role explicit."""

    faith: FaithfulnessTerm
    imp: ImportanceMinimalityTerm
    recon: tuple[AnyReconLossTerm, ...]


@dataclass(frozen=True)
class TargetPass:
    """The tPD target-pass surface (SPEC T3/T7): the full decomposition objective minus
    faithfulness — the delta is the off-target escape valve and must never be penalized,
    so a targeted objective has no faithfulness role at all."""

    imp: ImportanceMinimalityTerm
    recon: tuple[AnyReconLossTerm, ...]


NontargetHiddenSources = StochasticSources | ConstantSources | UnmaskedNoDeltaSources
"""The CLOSED non-target strategy set (SPEC T5) — the hidden non-target pass's whole
vocabulary, and the non-adversarial arms of the output pass's."""

NontargetOutputSources = (
    NontargetHiddenSources | PersistentSources | MixedPersistentStochasticSources
)
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
    head (S36), and its recon comparison is `HiddenActsOnlyReconstruction(points)` rather than
    the model output, so it carries no end-to-end term at all.

    `points` is the pass's, not each term's: the pass IS "reconstruct these activations". The
    imp-min CONFIG (penalty shape, anneal, frequency block) is the target pass's, shared by
    construction — T6's rule extended to the hidden role; only the coefficient is this pass's.

    Generic over `S` so the non-target stream's hidden pass keeps T5's narrowing in the
    TYPE: the CLOSED set, no adversarial or mixed sources — the 2026-08-19 amendment
    admits those on the non-target OUTPUT pass only."""

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
) -> tuple[FaithfulnessTerm | None, ImportanceMinimalityTerm | None, tuple[AnyReconLossTerm, ...]]:
    """One pass over an authored loss list into its objective roles, names unique across
    all roles. Completeness (which roles must be present) is each objective builder's own
    claim, not this walk's.

    Recon-term order follows the authored list and is semantically load-bearing: per-term
    RNG keys derive from the recon index (SPEC R1).
    """
    faith: FaithfulnessTerm | None = None
    imp: ImportanceMinimalityTerm | None = None
    recon_terms: list[AnyReconLossTerm] = []

    def unique_name(cfg: AnyLossMetricConfig) -> str:
        # Only committed terms are in `taken`, so persistent terms may call this once for
        # their state key and again inside `recon` without colliding with themselves.
        name = cfg.name if cfg.name is not None else cfg.type
        taken = {term.name for term in recon_terms}
        if faith is not None:
            taken.add(faith.name)
        if imp is not None:
            taken.add(imp.name)
        assert name not in taken, f"duplicate loss instance_key {name!r}"
        return name

    def recon[S: MaskSourceStrategy](
        cfg: AnyReconLossMetricConfig, plan: ReconPlan[S]
    ) -> AnyReconLossTerm:
        assert cfg.coeff is not None
        # Storage is width-erased; dataclass type params are invariant (3.13 synthesizes
        # `__replace__`), so widening is an explicit rebuild rather than an upcast.
        wide_plan: ReconPlan[MaskSourceStrategy] = tuple(
            ReconForward[MaskSourceStrategy](e.live_sites, e.sample_routing, e.sources)
            for e in plan
        )
        return ReconLossTerm(
            unique_name(cfg),
            cfg.coeff,
            wide_plan,
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
                assert all(k.frac > 0 for k in cfg.pnorm.points), (
                    f"pnorm knots must all keep frac > 0, got {cfg.pnorm.points}: at p = 0 "
                    "every (c + eps)^p is 1, so the penalty goes flat and stops teaching"
                )
                imp = ImportanceMinimalityTerm(unique_name(cfg), cfg.coeff, cfg)
            case SmoothL0ImportanceMinimalityLossConfig():
                assert imp is None
                assert all(k.frac > 0 for k in cfg.gamma.points), (
                    f"gamma knots must all keep frac > 0, got {cfg.gamma.points}: a zero "
                    "width collapses the smooth-L0 threshold band the gradient lives on"
                )
                imp = ImportanceMinimalityTerm(unique_name(cfg), cfg.coeff, cfg)
            case UnmaskedReconLossConfig():
                plan = make_plan(
                    all_sites_live(site_names),
                    AllRoutingConfig(),
                    ConstantSources(1.0),
                    n_samples=1,
                )
                recon_terms.append(recon(cfg, plan))
            case (
                CIMaskedReconLossConfig()
                | CIMaskedReconSubsetLossConfig()
                | CIMaskedReconLayerwiseLossConfig()
                | StochasticReconLossConfig()
                | StochasticReconSubsetLossConfig()
                | StochasticReconLayerwiseLossConfig()
                | ChunkwiseSubsetReconLossConfig()
            ):
                recon_terms.append(recon(cfg, _nontarget_recon_plan(cfg, site_names)))
            case PGDReconLossConfig() | PGDReconSubsetLossConfig():
                sources = FreshPGDSources(cfg.init, cfg.n_steps, cfg.step_size, cfg.source_shape)
                routing = (
                    cfg.routing if isinstance(cfg, PGDReconSubsetLossConfig) else AllRoutingConfig()
                )
                recon_terms.append(
                    recon(cfg, make_plan(all_sites_live(site_names), routing, sources, n_samples=1))
                )
            case PGDReconLayerwiseLossConfig():
                sources = FreshPGDSources(cfg.init, cfg.n_steps, cfg.step_size, cfg.source_shape)
                recon_terms.append(
                    recon(
                        cfg,
                        make_plan(
                            each_site_live(site_names), AllRoutingConfig(), sources, n_samples=1
                        ),
                    )
                )
            case MergedStochasticSubsetPPGDReconLossConfig():
                key = unique_name(cfg)
                sources = MixedPersistentStochasticSources(state_key=key, cfg=cfg)
                recon_terms.append(
                    recon(
                        cfg,
                        make_plan(all_sites_live(site_names), cfg.routing, sources, n_samples=1),
                    )
                )
            case PersistentPGDReconLossConfig():
                key = unique_name(cfg)
                sources = PersistentSources(state_key=key, cfg=cfg)
                recon_terms.append(
                    recon(
                        cfg,
                        make_plan(
                            all_sites_live(site_names), AllRoutingConfig(), sources, n_samples=1
                        ),
                    )
                )

    for term in recon_terms:
        for entry in term.plan:
            assert entry.live_sites and set(entry.live_sites) <= set(site_names), entry
    return faith, imp, tuple(recon_terms)


def build_objective(
    loss_metrics: Sequence[AnyLossMetricConfig],
    site_names: tuple[str, ...],
) -> LossSurface:
    """Build the closed plain-VPD objective, rejecting incomplete authored surfaces."""
    faith, imp, recon_terms = _collect_terms(loss_metrics, site_names)
    assert faith is not None and imp is not None, (
        f"need FaithfulnessLoss + ImportanceMinimalityLoss, got {[m.type for m in loss_metrics]}"
    )
    assert recon_terms, "no recon loss terms configured"
    return LossSurface(faith, imp, recon_terms)


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
    faith, imp, recon_terms = _collect_terms(loss_metrics, site_names)
    # The library boundary for lists built outside the schema; unreachable for a parsed
    # TargetedPDConfig.
    assert faith is None, "a targeted loss list carried a FaithfulnessLossConfig (SPEC T3)"
    assert imp is not None, (
        f"need an ImportanceMinimalityLoss, got {[m.type for m in loss_metrics]}"
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
        h_faith, h_imp, h_recon = _collect_terms(hidden.recon, site_names)
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
                recon=_nontarget_terms(nontarget.hidden.recon, site_names, "non-target hidden"),
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
        plan: ReconPlan[NontargetOutputSources]
        match cfg:
            case PersistentPGDReconLossConfig():
                sources: NontargetOutputSources = PersistentSources(
                    state_key=f"nontarget/{name}", cfg=cfg
                )
                plan = make_plan(all_sites_live(site_names), AllRoutingConfig(), sources, 1)
            case MergedStochasticSubsetPPGDReconLossConfig():
                sources = MixedPersistentStochasticSources(state_key=f"nontarget/{name}", cfg=cfg)
                plan = make_plan(all_sites_live(site_names), cfg.routing, sources, 1)
            case _:
                # Width-erased rebuild, as `_collect_terms.recon` does: dataclass type
                # params are invariant, so the closed-set plan widens explicitly.
                plan = tuple(
                    ReconForward[NontargetOutputSources](e.live_sites, e.sample_routing, e.sources)
                    for e in _nontarget_recon_plan(cfg, site_names)
                )
        terms.append(ReconLossTerm(name, cfg.coeff, plan, None))
    return tuple(terms)


def _nontarget_terms(
    cfgs: Sequence[NontargetReconLossMetricConfig], site_names: tuple[str, ...], what: str
) -> tuple[ReconLossTerm[NontargetHiddenSources], ...]:
    """Build the non-target HIDDEN pass's recon terms from the CLOSED vocabulary (T5/T12);
    `what` only names the pass in the error."""
    terms: list[ReconLossTerm[NontargetHiddenSources]] = []
    for cfg in cfgs:
        assert cfg.coeff is not None  # non-None at parse (NontargetConfig); narrows the type
        name = cfg.name if cfg.name is not None else cfg.type
        assert name not in {t.name for t in terms}, f"duplicate {what} loss {name!r}"
        # `hidden_acts_reconstruction=None` structurally: the S35 rider is refused on every
        # non-target term at parse — an output pass has no points, and a hidden pass's are the
        # pass's own (T5/T12).
        terms.append(ReconLossTerm(name, cfg.coeff, _nontarget_recon_plan(cfg, site_names), None))
    return tuple(terms)


def _nontarget_recon_plan(
    cfg: NontargetReconLossMetricConfig, site_names: tuple[str, ...]
) -> ReconPlan[StochasticSources | ConstantSources | UnmaskedNoDeltaSources]:
    """Plan construction for the recon types the non-target pass admits (SPEC T5): the
    stochastic/constant-source types — shared verbatim with the plain objective's arms,
    which widen via `recon` — plus the non-target-only unmasked-no-delta term (T4's one
    delta-off exception)."""

    def narrow(
        live_sets: list[LiveSet],
        routing: SubsetRoutingType,
        sources: StochasticSources | ConstantSources | UnmaskedNoDeltaSources,
        n_samples: int,
    ) -> ReconPlan[StochasticSources | ConstantSources | UnmaskedNoDeltaSources]:
        # The parameter annotation solves `make_plan`'s SourcesT at the pass's width —
        # values widen into parameters; invariant containers don't.
        return make_plan(live_sets, routing, sources, n_samples)

    match cfg:
        case CIMaskedReconLossConfig():
            return narrow(all_sites_live(site_names), AllRoutingConfig(), ConstantSources(0.0), 1)
        case CIMaskedReconSubsetLossConfig():
            return narrow(all_sites_live(site_names), cfg.routing, ConstantSources(0.0), 1)
        case CIMaskedReconLayerwiseLossConfig():
            return narrow(each_site_live(site_names), AllRoutingConfig(), ConstantSources(0.0), 1)
        case StochasticReconLossConfig():
            return narrow(
                all_sites_live(site_names),
                AllRoutingConfig(),
                StochasticSources(),
                cfg.n_mask_samples,
            )
        case StochasticReconSubsetLossConfig():
            return narrow(
                all_sites_live(site_names), cfg.routing, StochasticSources(), cfg.n_mask_samples
            )
        case StochasticReconLayerwiseLossConfig():
            return narrow(
                each_site_live(site_names),
                AllRoutingConfig(),
                StochasticSources(),
                cfg.n_mask_samples,
            )
        case ChunkwiseSubsetReconLossConfig():
            return narrow(
                live_groups(site_names, cfg.sites_per_chunk),
                cfg.routing,
                StochasticSources(),
                cfg.n_samples,
            )
        case UnmaskedNoDeltaReconLossConfig():
            return narrow(
                all_sites_live(site_names), AllRoutingConfig(), UnmaskedNoDeltaSources(), 1
            )
