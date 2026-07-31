"""The closed VPD training objective.

Authored loss metrics become exactly one faithfulness term, one importance-minimality term,
and a non-empty ordered tuple of recon terms. Recon planning lives in `recon.py`; this module
alone composes those plans with the other objective roles.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from param_decomp.core.configs import (
    AllRoutingConfig,
    AnyImportanceMinimalityLossConfig,
    AnyLossMetricConfig,
    ChunkwiseSubsetReconLossConfig,
    CIMaskedReconLayerwiseLossConfig,
    CIMaskedReconLossConfig,
    CIMaskedReconSubsetLossConfig,
    FaithfulnessLossConfig,
    ImportanceMinimalityLossConfig,
    MergedStochasticSubsetPPGDReconLossConfig,
    PersistentPGDReconLossConfig,
    PGDReconLayerwiseLossConfig,
    PGDReconLossConfig,
    PGDReconSubsetLossConfig,
    SmoothL0ImportanceMinimalityLossConfig,
    StochasticHiddenActsReconLossConfig,
    StochasticReconLayerwiseLossConfig,
    StochasticReconLossConfig,
    StochasticReconSubsetLossConfig,
    UnmaskedReconLossConfig,
)
from param_decomp.core.recon import (
    ConstantSources,
    FreshPGDSources,
    MixedPersistentStochasticSources,
    PersistentSources,
    ReconLossTerm,
    ReconPlan,
    StochasticSources,
    all_sites_live,
    each_site_live,
    live_groups,
    make_plan,
)


@dataclass(frozen=True)
class FaithfulnessTerm:
    """Weight-space term: `Σ_s ‖Δ_s‖² / Σ_s numel`."""

    name: str
    coeff: float


@dataclass(frozen=True)
class ImportanceMinimalityTerm:
    """CI-space importance-minimality plus optional frequency penalty."""

    name: str
    coeff: float
    cfg: AnyImportanceMinimalityLossConfig


@dataclass(frozen=True)
class LossSurface:
    """`L = c·faith + c·importance + Σ c·recon`, with every role explicit."""

    faith: FaithfulnessTerm
    imp: ImportanceMinimalityTerm
    recon: tuple[ReconLossTerm, ...]


def build_objective(
    loss_metrics: Sequence[AnyLossMetricConfig],
    site_names: tuple[str, ...],
) -> LossSurface:
    """Build the closed objective, rejecting unsupported or incomplete authored surfaces.

    Recon-term order follows the authored list and is semantically load-bearing: per-term
    RNG keys derive from the recon index (SPEC R1).
    """
    faith: FaithfulnessTerm | None = None
    imp: ImportanceMinimalityTerm | None = None
    recon_terms: list[ReconLossTerm] = []

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

    def recon(cfg: AnyLossMetricConfig, plan: ReconPlan) -> ReconLossTerm:
        assert cfg.coeff is not None
        return ReconLossTerm(unique_name(cfg), cfg.coeff, plan)

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
            case UnmaskedReconLossConfig() | CIMaskedReconLossConfig():
                value = 1.0 if isinstance(cfg, UnmaskedReconLossConfig) else 0.0
                plan = make_plan(
                    all_sites_live(site_names),
                    AllRoutingConfig(),
                    ConstantSources(value),
                    n_samples=1,
                )
                recon_terms.append(recon(cfg, plan))
            case CIMaskedReconSubsetLossConfig():
                plan = make_plan(
                    all_sites_live(site_names), cfg.routing, ConstantSources(0.0), n_samples=1
                )
                recon_terms.append(recon(cfg, plan))
            case CIMaskedReconLayerwiseLossConfig():
                plan = make_plan(
                    each_site_live(site_names), AllRoutingConfig(), ConstantSources(0.0), 1
                )
                recon_terms.append(recon(cfg, plan))
            case StochasticHiddenActsReconLossConfig():
                # Deliberately eval-only / keep-on-bridge (SPEC S31); this is a known
                # unsupported training arm, not an unrecognized config variant.
                raise AssertionError(f"{cfg.type} is an eval metric, not a JAX training loss")
            case StochasticReconLossConfig():
                plan = make_plan(
                    all_sites_live(site_names),
                    AllRoutingConfig(),
                    StochasticSources(),
                    n_samples=cfg.n_mask_samples,
                )
                recon_terms.append(recon(cfg, plan))
            case StochasticReconSubsetLossConfig():
                plan = make_plan(
                    all_sites_live(site_names),
                    cfg.routing,
                    StochasticSources(),
                    n_samples=cfg.n_mask_samples,
                )
                recon_terms.append(recon(cfg, plan))
            case StochasticReconLayerwiseLossConfig():
                plan = make_plan(
                    each_site_live(site_names),
                    AllRoutingConfig(),
                    StochasticSources(),
                    n_samples=cfg.n_mask_samples,
                )
                recon_terms.append(recon(cfg, plan))
            case ChunkwiseSubsetReconLossConfig():
                plan = make_plan(
                    live_groups(site_names, cfg.sites_per_chunk),
                    cfg.routing,
                    StochasticSources(),
                    n_samples=cfg.n_samples,
                )
                recon_terms.append(recon(cfg, plan))
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

    assert faith is not None and imp is not None, (
        f"need FaithfulnessLoss + ImportanceMinimalityLoss, got {[m.type for m in loss_metrics]}"
    )
    assert recon_terms, "no recon loss terms configured"
    for term in recon_terms:
        for entry in term.plan:
            assert entry.live_sites and set(entry.live_sites) <= set(site_names), entry
    return LossSurface(faith, imp, tuple(recon_terms))
