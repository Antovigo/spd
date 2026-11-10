"""Persistent adversaries on the tPD NON-TARGET OUTPUT pass (SPEC T5/T7 amended 2026-08-19).

The amendment's load-bearing claims, each pinned here on the TMS fixture:

- a `PersistentPGDReconLoss` / `MergedStochasticSubsetPPGDReconLoss` in `nontarget.recon`
  builds, steps, and REPORTS under the non-target namespace;
- the bundle actually trains: sources move and the SRC_STEP moments advance by
  `n_warmup + 1` per step (S13'), through the shared backward's final ascent (S14');
- the non-target HIDDEN vocabulary admits the MERGED term only (T5/T12 amended
  2026-08-20 — the zero-extra-forward adversary): a standalone persistent type there is
  a parse error, not a filtered no-op, while a merged term builds, trains its
  `nontarget_hidden/`-prefixed bundle, and reports under the composed namespace;
- fused and sequential schedulings still score the same objective with the new arms in
  the graph (T1).
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array
from pydantic import ValidationError

from param_decomp.core.adversary import (
    PersistentAdversary,
    SourceStacks,
    init_persistent_sources,
    init_sources_opt_state,
)
from param_decomp.core.ci_fn.implementations.layerwise_mlp import LayerwiseMLPCIFnArch
from param_decomp.core.components import SiteC, init_component_stacks
from param_decomp.core.configs import (
    AdamPGDConfig,
    AdamWOptimizerConfig,
    HiddenPassConfig,
    ImportanceMinimalityLossConfig,
    MergedStochasticSubsetPPGDReconLossConfig,
    NontargetConfig,
    NontargetHiddenConfig,
    PersistentPGDReconLossConfig,
    StochasticReconLossConfig,
    TargetedLossMetricConfig,
)
from param_decomp.core.losses import init_frequency_estimator
from param_decomp.core.model import PlacedModel
from param_decomp.core.objective import build_targeted_objective
from param_decomp.core.recon import MixedPersistentStochasticSources, PersistentSources
from param_decomp.core.run_state import _adamw_optimizer
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.train import (
    Decomposition,
    ForwardSubstrate,
    TargetedPDTrainingState,
    TrainState,
    make_targeted_train_step,
)
from param_decomp.targets.tms import (
    TMSConfig,
    init_tms_target,
    sample_sparse_features,
    scatter_features,
    site_input_tap_keys,
    site_specs,
    tms_decomposed_model,
)

SITES = (SiteC("linear1", 8), SiteC("linear2", 6))
N_WARMUP = 1
NONTARGET_BATCH = 16
TOTAL_STEPS = 20


def _adversary_optimizer() -> AdamPGDConfig:
    return AdamPGDConfig(lr_schedule=ScheduleConfig.constant(0.05))


def _persistent_cfg() -> PersistentPGDReconLossConfig:
    return PersistentPGDReconLossConfig(
        coeff=0.5,
        optimizer=_adversary_optimizer(),
        source_shape="bc",
        n_warmup_steps=N_WARMUP,
    )


def _merged_cfg() -> MergedStochasticSubsetPPGDReconLossConfig:
    return MergedStochasticSubsetPPGDReconLossConfig(
        coeff=1.0,
        optimizer=_adversary_optimizer(),
        source_shape="bc",
        n_warmup_steps=0,
        adv_fraction=ScheduleConfig.constant(0.5),
    )


def _loss_metrics() -> tuple[TargetedLossMetricConfig, ...]:
    return (
        ImportanceMinimalityLossConfig(coeff=3e-3, gamma=ScheduleConfig.constant(1.0)),
        StochasticReconLossConfig(coeff=1.0),
    )


def _setup(
    nontarget: NontargetConfig, *, sequential: bool = False, hidden: HiddenPassConfig | None = None
):
    cfg = TMSConfig(n_features=5, n_hidden=2)
    sites = site_specs(cfg, SITES)
    target = init_tms_target(cfg, jax.random.PRNGKey(0))
    model = PlacedModel(model=tms_decomposed_model(cfg, target, sites), placement=None)
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = LayerwiseMLPCIFnArch(
        hidden_dims=(16,),
        has_position_axis=False,
        input_names=site_input_tap_keys(tuple(s.name for s in sites)),
        dual=hidden is not None,
    ).initialize(sites, None, jax.random.PRNGKey(2))
    loss_metrics = _loss_metrics()
    objective = build_targeted_objective(loss_metrics, nontarget, sites, hidden=hidden)
    # The step asks the training state's adversaries for every plan's state key — build
    # each non-target bundle at ITS pass's geometry: positionless `bc` is `(B,)` over the
    # broad stream's batch (T2).
    nt_terms = objective.nontarget.recon + (
        objective.nontarget_hidden.recon if objective.nontarget_hidden is not None else ()
    )
    adversaries: dict[str, PersistentAdversary] = {}
    for term in nt_terms:
        match term.sources:
            case PersistentSources() | MixedPersistentStochasticSources():
                strategy = term.sources
            case _:
                continue
        sources = init_persistent_sources(
            sites, (NONTARGET_BATCH,), jnp.float32, jax.random.PRNGKey(3)
        )
        adversaries[strategy.state_key] = PersistentAdversary(
            sources=sources,
            opt_state=init_sources_opt_state(strategy.cfg.optimizer, sources),
            state_key=strategy.state_key,
            optimizer=strategy.cfg.optimizer,
            n_warmup=strategy.cfg.n_warmup_steps,
        )
    optimizer_config = AdamWOptimizerConfig(lr_schedule=ScheduleConfig.constant(1e-3))
    opt_vu = _adamw_optimizer(optimizer_config, TOTAL_STEPS)
    opt_ci = _adamw_optimizer(optimizer_config, TOTAL_STEPS)
    importance = next(
        term for term in loss_metrics if isinstance(term, ImportanceMinimalityLossConfig)
    )

    def frequency():
        return init_frequency_estimator(importance.frequency, sites)

    state = TrainState(
        decomposition=Decomposition[Array](components=vu, ci_fn=ci_fn),
        training=TargetedPDTrainingState(
            objective=objective,
            target_frequency=frequency(),
            nontarget_frequency=frequency(),
            hidden_frequency=frequency() if objective.hidden is not None else None,
            nontarget_hidden_frequency=(
                frequency() if objective.nontarget_hidden is not None else None
            ),
            ci_scaled_weight_decay=None,
            components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries=adversaries,
            step=jnp.zeros((), jnp.int32),
        ),
    )
    step = jax.jit(
        make_targeted_train_step(
            model,
            substrate=ForwardSubstrate.of(
                model,
                remat_recon_forwards=False,
                remat_ci_fn=False,
                ci_capture_keys=ci_fn.capture_keys,
            ),
            components_optimizer=opt_vu,
            ci_fn_optimizer=opt_ci,
            total_steps=TOTAL_STEPS,
            sequential_passes=sequential,
        )
    )
    return cfg, state, step


def _host_sources(sources: SourceStacks) -> list[np.ndarray]:
    """The stored source leaves, copied off-device (the step donates its input state)."""
    return [np.asarray(leaf) for leaf in jax.tree.leaves(sources)]


def _sources_moved(before: list[np.ndarray], after: SourceStacks) -> bool:
    return any(not np.allclose(b, a) for b, a in zip(before, _host_sources(after), strict=True))


def _assert_projected(sources: SourceStacks) -> None:
    for arr in _host_sources(sources):
        assert arr.min() >= 0.0 and arr.max() <= 1.0, "S15: sources must stay projected"


def _model_of():
    cfg = TMSConfig(n_features=5, n_hidden=2)
    sites = site_specs(cfg, SITES)
    return PlacedModel(
        model=tms_decomposed_model(cfg, init_tms_target(cfg, jax.random.PRNGKey(0)), sites),
        placement=None,
    )


def _batches(cfg: TMSConfig, i: int = 0) -> tuple[Array, Array]:
    narrow = sample_sparse_features(
        jax.random.fold_in(jax.random.PRNGKey(10), i), 16, 2, 0.3, "exactly_one_active"
    )
    target_batch = scatter_features(narrow, (0, 1), cfg.n_features)
    broad = sample_sparse_features(
        jax.random.fold_in(jax.random.PRNGKey(11), i),
        NONTARGET_BATCH,
        5,
        0.3,
        "at_least_zero_active",
    )
    return target_batch, broad


def test_nontarget_persistent_term_trains_its_adversary():
    """The bundle updates S13'-style: `n_warmup + 1` moment steps per training step, the
    final one riding the shared backward (S14') — off-target, delta pinned (T4)."""
    nontarget = NontargetConfig(
        batch_size=NONTARGET_BATCH,
        impmin_coeff=6e-3,
        recon=[StochasticReconLossConfig(coeff=1.0), _persistent_cfg()],
    )
    cfg, state, step = _setup(nontarget)
    state_key = "nontarget/PersistentPGDReconLoss"
    before = _host_sources(state.training.adversaries[state_key].sources)

    target_batch, broad = _batches(cfg)
    new_state, metrics = step(_model_of(), state, target_batch, broad, jax.random.PRNGKey(7))
    adv = new_state.training.adversaries[state_key]

    assert int(adv.opt_state.step_count) == N_WARMUP + 1
    # Every stored channel (component and weight-delta sources, every site's stack).
    moved = _sources_moved(before, adv.sources)
    assert moved, "the non-target adversary's sources never moved"
    _assert_projected(adv.sources)
    assert "nontarget_data/loss/PersistentPGDReconLoss" in metrics
    assert "nontarget_data/loss/StochasticReconLoss" in metrics

    # A second step keeps advancing the same persistent state.
    target_batch2, broad2 = _batches(cfg, 1)
    newer, _ = step(_model_of(), new_state, target_batch2, broad2, jax.random.PRNGKey(8))
    assert int(newer.training.adversaries[state_key].opt_state.step_count) == 2 * (N_WARMUP + 1)


def test_nontarget_merged_term_steps_with_zero_warmup():
    """The S34 merge off-target: one masked forward, no warmup ascents — the bundle's one
    update per step is the S14' final ascent, gated to adversarial-assigned samples."""
    nontarget = NontargetConfig(
        batch_size=NONTARGET_BATCH, impmin_coeff=6e-3, recon=[_merged_cfg()]
    )
    cfg, state, step = _setup(nontarget)
    state_key = "nontarget/MergedStochasticSubsetPPGDReconLoss"
    before = _host_sources(state.training.adversaries[state_key].sources)

    target_batch, broad = _batches(cfg)
    new_state, metrics = step(_model_of(), state, target_batch, broad, jax.random.PRNGKey(7))
    adv = new_state.training.adversaries[state_key]

    assert int(adv.opt_state.step_count) == 1  # final ascent only: n_warmup == 0
    # Every stored channel (component and weight-delta sources, every site's stack).
    moved = _sources_moved(before, adv.sources)
    assert moved
    assert "nontarget_data/loss/MergedStochasticSubsetPPGDReconLoss" in metrics


@pytest.mark.parametrize("make_cfg", [_persistent_cfg, _merged_cfg])
def test_sequential_and_fused_agree_with_nontarget_adversaries(
    make_cfg: Callable[
        [], PersistentPGDReconLossConfig | MergedStochasticSubsetPPGDReconLossConfig
    ],
):
    """T1 with the amended arms in the graph: identical loss scalars either scheduling."""
    nontarget = NontargetConfig(
        batch_size=NONTARGET_BATCH,
        impmin_coeff=6e-3,
        recon=[StochasticReconLossConfig(coeff=1.0), make_cfg()],
    )
    cfg, state_a, fused = _setup(nontarget, sequential=False)
    _, state_b, seq = _setup(nontarget, sequential=True)
    target_batch, broad = _batches(cfg)
    _, fused_metrics = fused(_model_of(), state_a, target_batch, broad, jax.random.PRNGKey(7))
    _, seq_metrics = seq(_model_of(), state_b, target_batch, broad, jax.random.PRNGKey(7))
    assert set(fused_metrics) == set(seq_metrics)
    for name in sorted(k for k in fused_metrics if not k.startswith("grad_norms/")):
        assert fused_metrics[name] == seq_metrics[name], name


def test_nontarget_hidden_refuses_standalone_persistent_type():
    """T5/T12 as amended 2026-08-20: the non-target HIDDEN vocabulary admits the merged
    term only — the standalone persistent type (an extra forward per step) stays a parse
    error, while the merged config (same adversary, zero extra forwards) parses."""
    with pytest.raises(ValidationError):
        NontargetHiddenConfig.model_validate(
            {
                "impmin_coeff": 6e-3,
                "recon": [
                    {
                        "type": "PersistentPGDReconLoss",
                        "coeff": 0.5,
                        "optimizer": {
                            "type": "adam",
                            "lr_schedule": {"max_val": 0.05, "points": [{"at": 0.0, "frac": 1.0}]},
                        },
                        "source_shape": "bc",
                    }
                ],
            }
        )
    parsed = NontargetHiddenConfig.model_validate(
        {
            "impmin_coeff": 6e-3,
            "recon": [
                {
                    "type": "MergedStochasticSubsetPPGDReconLoss",
                    "coeff": 1.0,
                    "adv_fraction": 0.5,
                    "n_warmup_steps": 0,
                    "optimizer": {
                        "type": "adam",
                        "lr_schedule": {
                            "max_val": 0.05,
                            "points": [{"at": 0.0, "frac": 1.0}, {"at": 1.0, "frac": 1.0}],
                        },
                    },
                    "source_shape": "bc",
                }
            ],
        }
    )
    assert isinstance(parsed.recon[0], MergedStochasticSubsetPPGDReconLossConfig)


HIDDEN_POINTS = ("linear1.out", "linear2.out")
"""Both decomposed sites' linear outputs — downstream of the masking, so the pass has a
gradient to give (as in `test_dual_objective`)."""


def _target_hidden() -> HiddenPassConfig:
    return HiddenPassConfig(
        points=HIDDEN_POINTS,
        impmin_coeff=5e-3,
        recon=[StochasticReconLossConfig(coeff=2.0, name="HiddenStochasticRecon")],
    )


def test_nontarget_hidden_merged_term_trains_its_adversary():
    """The S34 merge on the non-target HIDDEN pass (T5/T12 amended 2026-08-20): the
    `nontarget_hidden/`-prefixed bundle updates through the S14' final ascent alone
    (n_warmup 0), stays projected (S15), and reports under the composed
    stream x role namespace."""
    nontarget = NontargetConfig(
        batch_size=NONTARGET_BATCH,
        impmin_coeff=6e-3,
        recon=[StochasticReconLossConfig(coeff=1.0)],
        hidden=NontargetHiddenConfig(impmin_coeff=6e-3, recon=[_merged_cfg()]),
    )
    cfg, state, step = _setup(nontarget, hidden=_target_hidden())
    state_key = "nontarget_hidden/MergedStochasticSubsetPPGDReconLoss"
    before = _host_sources(state.training.adversaries[state_key].sources)

    target_batch, broad = _batches(cfg)
    new_state, metrics = step(_model_of(), state, target_batch, broad, jax.random.PRNGKey(7))
    adv = new_state.training.adversaries[state_key]

    assert int(adv.opt_state.step_count) == 1  # final ascent only: n_warmup == 0
    # Every stored channel (component and weight-delta sources, every site's stack).
    moved = _sources_moved(before, adv.sources)
    assert moved, "the non-target hidden adversary's sources never moved"
    _assert_projected(adv.sources)
    assert "nontarget_data/hidden_ci/loss/MergedStochasticSubsetPPGDReconLoss" in metrics


def test_sequential_and_fused_agree_with_nontarget_hidden_merged_term():
    """T1 with the hidden-pass merged arm in the graph: identical loss scalars either
    scheduling. Every per-term and per-pass loss is compared for EXACT equality; the one
    cross-pass composite (`total`) gets a ~1-ULP fp32 tolerance — with this graph shape
    the fused scheduling's cross-pass fusion clusters round the target pass's (unlogged)
    contribution one bit differently under the suite's pinned `--xla_cpu_max_isa=AVX2`
    codegen, while every logged component stays bit-equal (a dropped or double-counted
    pass would move `total` by far more than rounding)."""
    nontarget = NontargetConfig(
        batch_size=NONTARGET_BATCH,
        impmin_coeff=6e-3,
        recon=[StochasticReconLossConfig(coeff=1.0)],
        hidden=NontargetHiddenConfig(impmin_coeff=6e-3, recon=[_merged_cfg()]),
    )
    cfg, state_a, fused = _setup(nontarget, sequential=False, hidden=_target_hidden())
    _, state_b, seq = _setup(nontarget, sequential=True, hidden=_target_hidden())
    target_batch, broad = _batches(cfg)
    _, fused_metrics = fused(_model_of(), state_a, target_batch, broad, jax.random.PRNGKey(7))
    _, seq_metrics = seq(_model_of(), state_b, target_batch, broad, jax.random.PRNGKey(7))
    assert set(fused_metrics) == set(seq_metrics)
    for name in sorted(k for k in fused_metrics if not k.startswith("grad_norms/")):
        if name == "total":
            assert np.isclose(fused_metrics[name], seq_metrics[name], rtol=1e-6, atol=0.0), name
        else:
            assert fused_metrics[name] == seq_metrics[name], name
