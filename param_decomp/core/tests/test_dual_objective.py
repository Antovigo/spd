"""The four-pass tPD objective (SPEC T12) and the two ways it reaches one gradient (T1).

The load-bearing claim is that `sequential_passes` is a SCHEDULING choice and nothing else:
scoring the passes one at a time and adding their gradients must give exactly what fusing
them gives, because adding per-pass gradients is what the fused backward does internally.
That is checked as an exact-structure, tight-tolerance equality rather than a smoke test —
if it ever drifts, the memory-saving path is silently training a different objective.

The TMS target is a fixture here, as in `experiments/tms/test_targeted_tms.py`; there is no
shipped toy tPD run shape.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jaxtyping import Array

from param_decomp.core.ci_fn import LayerwiseMLPCIArch, LayerwiseMLPCIFn, build_ci_fn
from param_decomp.core.components import SiteC, init_component_stacks
from param_decomp.core.configs import (
    HiddenPassConfig,
    ImportanceMinimalityLossConfig,
    NonlinearityLocalityLossConfig,
    NontargetConfig,
    NontargetHiddenConfig,
    StochasticReconLossConfig,
    TargetedLossMetricConfig,
)
from param_decomp.core.model import PlacedModel
from param_decomp.core.objective import build_targeted_objective
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.train import (
    Decomposition,
    ForwardSubstrate,
    TrainingItem,
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

HIDDEN_POINTS = ("linear1.out", "linear2.out")
"""Both decomposed sites' linear outputs — downstream of the masking, so the pass has a
gradient to give (the trainer refuses points masking cannot reach)."""


def _loss_metrics() -> tuple[TargetedLossMetricConfig, ...]:
    return (
        ImportanceMinimalityLossConfig(coeff=3e-3, pnorm=ScheduleConfig.constant(1.0)),
        StochasticReconLossConfig(coeff=1.0),
    )


def _nonlinearity_cfg(coeff: float = 0.25) -> NonlinearityLocalityLossConfig:
    """TMS declares a `Neurons` partition on `linear2` only, so `neuron` is the one unit
    kind the coefficients may name."""
    return NonlinearityLocalityLossConfig(
        coeff=coeff,
        relative_threshold=ScheduleConfig.constant(4.0),
        unit_kind_coefficients={"neuron": 1.0},
    )


def _hidden() -> tuple[HiddenPassConfig, NontargetConfig]:
    hidden = HiddenPassConfig(
        points=HIDDEN_POINTS,
        impmin_coeff=5e-3,
        # An explicit name: the identity is unique across passes because both this and the
        # target pass would otherwise default to the same type literal.
        recon=[StochasticReconLossConfig(coeff=2.0, name="HiddenStochasticRecon")],
    )
    nontarget = NontargetConfig(
        batch_size=32,
        impmin_coeff=6e-3,
        recon=[StochasticReconLossConfig(coeff=1.0)],
        hidden=NontargetHiddenConfig(
            impmin_coeff=6e-3, recon=[StochasticReconLossConfig(coeff=1.0)]
        ),
    )
    return hidden, nontarget


def _setup(*, dual: bool, sequential: bool, nonlinearity_coeff: float | None = None):
    cfg = TMSConfig(n_features=5, n_hidden=2)
    sites = site_specs(cfg, (SiteC("linear1", 8), SiteC("linear2", 6)))
    target = init_tms_target(cfg, jax.random.PRNGKey(0))
    model = PlacedModel(model=tms_decomposed_model(cfg, target, sites), placement=None)
    vu = init_component_stacks(sites, jax.random.PRNGKey(1))
    ci_fn = build_ci_fn(
        LayerwiseMLPCIArch(
            hidden_dims=(16,),
            has_position_axis=False,
            input_names=site_input_tap_keys(tuple(s.name for s in sites)),
            dual=dual,
        ),
        sites,
        jax.random.PRNGKey(2),
    )
    opt_vu = optax.adamw(1e-3, weight_decay=0.0)
    opt_ci = optax.adamw(1e-3, weight_decay=0.0)
    state = TrainState(
        decomposition=Decomposition(components=vu, ci_fn=ci_fn),
        training=TrainingItem(
            components_opt_state=opt_vu.init(eqx.filter(vu, eqx.is_array)),
            ci_fn_opt_state=opt_ci.init(eqx.filter(ci_fn, eqx.is_array)),
            adversaries={},
            freq_ema=None,
            step=jnp.zeros((), jnp.int32),
        ),
    )
    if dual:
        hidden, nontarget = _hidden()
    else:
        hidden, nontarget = (
            None,
            NontargetConfig(
                batch_size=32, impmin_coeff=6e-3, recon=[StochasticReconLossConfig(coeff=1.0)]
            ),
        )
    loss_metrics = _loss_metrics()
    if nonlinearity_coeff is not None:
        loss_metrics = (*loss_metrics, _nonlinearity_cfg(nonlinearity_coeff))
    objective = build_targeted_objective(loss_metrics, nontarget, model.site_names, hidden=hidden)
    step = make_targeted_train_step(
        model_static=model,
        substrate=ForwardSubstrate.of(
            model,
            remat_recon_forwards=False,
            remat_ci_fn=False,
            ci_capture_keys=ci_fn.capture_keys,
            ci_placement=None,
        ),
        objective=objective,
        ci_scaled_weight_decay=None,
        components_optimizer=opt_vu,
        ci_fn_optimizer=opt_ci,
        total_steps=20,
        sequential_passes=sequential,
    )
    return cfg, state, step


def _batches(cfg: TMSConfig, i: int = 0) -> tuple[Array, Array]:
    """The two streams: a NARROW two-feature grid (the behavior being decomposed) and the
    broad five-feature distribution (T2)."""
    narrow = sample_sparse_features(
        jax.random.fold_in(jax.random.PRNGKey(10), i), 16, 2, 0.3, "exactly_one_active"
    )
    target_batch = scatter_features(narrow, (0, 1), cfg.n_features)
    broad = sample_sparse_features(
        jax.random.fold_in(jax.random.PRNGKey(11), i), 16, 5, 0.3, "at_least_zero_active"
    )
    return target_batch, broad


def test_sequential_and_fused_passes_give_the_same_step():
    """T1. The two schedulings must score the SAME objective — the sequential path exists to
    bound memory, and a difference in what it optimizes would make it silently wrong.

    Graded by what each quantity can distinguish:

    - every LOSS scalar is compared for EXACT equality. The forward is identical, so anything
      other than bit-equality here means a pass was dropped, double-counted, or fed the wrong
      CI head.
    - the resulting DECOMPOSITION is compared tightly. It cannot be exact at bf16 compute:
      the two schedulings accumulate the same mathematical gradient in a different order, and
      bf16 has ~8 mantissa bits. Forcing `precision.COMPUTE_DT` to fp32 makes this comparison
      BIT-EXACT and collapses the grad-norm gap below to ~1e-7, which is how we know the
      residual is rounding rather than a different objective.
    - the grad-norm DIAGNOSTICS get bf16 width. They are norms of the accumulated gradient,
      so they inherit that rounding directly.
    """
    cfg, state_a, fused_step = _setup(dual=True, sequential=False)
    _, state_b, sequential_step = _setup(dual=True, sequential=True)
    target_batch, broad = _batches(cfg)
    key = jax.random.PRNGKey(7)

    model = _model_of()
    fused_state, fused_metrics = fused_step(model, state_a, target_batch, broad, key)
    seq_state, seq_metrics = sequential_step(model, state_b, target_batch, broad, key)

    shared = set(fused_metrics) & set(seq_metrics)
    assert shared == set(fused_metrics) == set(seq_metrics), "the two paths logged different keys"
    # Every pass must be represented, or "equal" would be cheap.
    for expected in (
        "total",
        "loss/StochasticReconLoss",
        "hidden_ci/loss/total",
        "nontarget_data/loss/total",
        "nontarget_data/hidden_ci/loss/total",
    ):
        assert expected in shared, sorted(shared)

    losses = sorted(k for k in shared if not k.startswith("grad_norms/"))
    for name in losses:
        assert fused_metrics[name] == seq_metrics[name], (
            f"{name} differs between the fused and sequential pass schedules: "
            f"{fused_metrics[name]} vs {seq_metrics[name]}"
        )

    fused_leaves = jax.tree.leaves(eqx.filter(fused_state.decomposition, eqx.is_inexact_array))
    seq_leaves = jax.tree.leaves(eqx.filter(seq_state.decomposition, eqx.is_inexact_array))
    for got, want in zip(seq_leaves, fused_leaves, strict=True):
        assert jnp.allclose(got, want, rtol=1e-5, atol=1e-7), (
            "the sequential pass schedule moved the decomposition somewhere else"
        )

    for name in sorted(k for k in shared if k.startswith("grad_norms/")):
        assert jnp.allclose(fused_metrics[name], seq_metrics[name], rtol=2e-3, atol=1e-6), name


def _model_of():
    cfg = TMSConfig(n_features=5, n_hidden=2)
    sites = site_specs(cfg, (SiteC("linear1", 8), SiteC("linear2", 6)))
    return PlacedModel(
        model=tms_decomposed_model(cfg, init_tms_target(cfg, jax.random.PRNGKey(0)), sites),
        placement=None,
    )


@pytest.mark.parametrize("sequential", [False, True])
def test_hidden_pass_trains_the_hidden_head_and_reports_its_own_losses(sequential: bool):
    """T12/S37: the hidden pass moves the hidden head, and logs under its own namespace."""
    cfg, state, step = _setup(dual=True, sequential=sequential)
    target_batch, broad = _batches(cfg)
    before = state.decomposition.ci_fn
    assert isinstance(before, LayerwiseMLPCIFn)
    # COPIED off-device before the step: `state` is donated, so its buffers are gone after.
    before_heads = {
        site: np.asarray(mlp.hidden_head[0])
        for site, mlp in before.site_mlps.items()
        if mlp.hidden_head is not None
    }
    before_trunks = {site: np.asarray(mlp.weights[0]) for site, mlp in before.site_mlps.items()}

    new_state, metrics = step(_model_of(), state, target_batch, broad, jax.random.PRNGKey(7))
    after = new_state.decomposition.ci_fn
    assert isinstance(after, LayerwiseMLPCIFn)

    for site in ("linear1", "linear2"):
        head_after = after.site_mlps[site].hidden_head
        assert head_after is not None
        assert not np.allclose(before_heads[site], np.asarray(head_after[0])), (
            f"the hidden head at {site} did not move — the hidden pass reached nothing"
        )
        # And the trunk moved too: both objectives shape one representation (S37).
        assert not np.allclose(before_trunks[site], np.asarray(after.site_mlps[site].weights[0]))

    assert "hidden_ci/loss/total" in metrics
    assert "hidden_ci/loss/HiddenStochasticRecon" in metrics
    assert "nontarget_data/hidden_ci/loss/total" in metrics
    # The target-OUTPUT pass keeps the unprefixed keys every tPD run has always logged.
    assert "loss/StochasticReconLoss" in metrics and "total" in metrics
    for point in HIDDEN_POINTS:
        key = f"hidden_ci/loss/HiddenStochasticRecon/hidden_acts_reconstruction/{point}"
        assert key in metrics, sorted(k for k in metrics if "hidden_acts" in k)
    # The hidden pass has NO end-to-end term, so it must not report one.
    assert "hidden_ci/loss/HiddenStochasticRecon/e2e" not in metrics


def test_hidden_pass_and_dual_ci_must_agree():
    """A dual CI fn with no hidden pass would train a head against nothing, and a hidden pass
    with a single-role fn has no head to train. Both are refused, at trace."""
    cfg, state, step = _setup(dual=False, sequential=False)
    dual_fn = build_ci_fn(
        LayerwiseMLPCIArch(
            hidden_dims=(16,),
            has_position_axis=False,
            input_names=site_input_tap_keys(("linear1", "linear2")),
            dual=True,
        ),
        site_specs(TMSConfig(n_features=5, n_hidden=2), (SiteC("linear1", 8), SiteC("linear2", 6))),
        jax.random.PRNGKey(2),
    )
    mismatched = TrainState(
        decomposition=Decomposition(components=state.decomposition.components, ci_fn=dual_fn),
        training=state.training,
    )
    target_batch, broad = _batches(cfg)
    with pytest.raises(AssertionError, match="head count and the objective's pass roles"):
        step(_model_of(), mismatched, target_batch, broad, jax.random.PRNGKey(7))


def test_nontarget_hidden_requires_the_target_hidden_pass():
    """T12: the non-target hidden pass measures at the target block's points, so it cannot
    exist without it — refused at objective build, not at the GPUs."""
    nontarget = NontargetConfig(
        batch_size=32,
        impmin_coeff=6e-3,
        recon=[StochasticReconLossConfig(coeff=1.0)],
        hidden=NontargetHiddenConfig(
            impmin_coeff=6e-3, recon=[StochasticReconLossConfig(coeff=1.0)]
        ),
    )
    with pytest.raises(AssertionError, match="nontarget.hidden needs pd.hidden"):
        build_targeted_objective(_loss_metrics(), nontarget, ("linear1", "linear2"))


def test_the_nonlinearity_prior_is_scored_once_whatever_the_pass_count() -> None:
    """SPEC S36 as amended for tPD: the prior is weight-space and belongs to no pass, so a
    four-pass run must charge it exactly once — not once per pass.

    Measured as the term's own contribution (total WITH minus total WITHOUT) rather than by
    counting passes: a per-pass application would make that contribution scale with the pass
    count, so comparing the two-pass and four-pass shapes is what catches it."""

    def contribution(coeff: float, *, dual: bool) -> float:
        cfg, state, step = _setup(dual=dual, sequential=False)
        target_batch, broad = _batches(cfg)
        _, without = step(_model_of(), state, target_batch, broad, jax.random.PRNGKey(3))
        _cfg, state, step = _setup(dual=dual, sequential=False, nonlinearity_coeff=coeff)
        _, with_term = step(_model_of(), state, target_batch, broad, jax.random.PRNGKey(3))
        return float(with_term["total"]) - float(without["total"])

    two_pass = contribution(0.25, dual=False)
    four_pass = contribution(0.25, dual=True)
    assert two_pass > 0.0, two_pass
    # It reads `U` alone, so its contribution does not depend on how many passes exist.
    assert two_pass == pytest.approx(four_pass, rel=1e-5), (two_pass, four_pass)
    # ... and it is linear in the coefficient, i.e. charged exactly once.
    assert contribution(0.5, dual=True) == pytest.approx(2.0 * four_pass, rel=1e-5)


def test_a_hidden_pass_cannot_spell_the_nonlinearity_prior() -> None:
    """It is authored on the ONE target loss list; a hidden pass carrying it would imply a
    per-pass weight-space term, which S36 says does not exist. Refused at PARSE — the
    hidden pass's recon union is the recon vocabulary only — with a library-boundary assert
    in `build_targeted_objective` behind it for programmatically-built passes."""
    hidden, _nontarget = _hidden()
    with pytest.raises(Exception, match="NonlinearityLocalityLoss"):
        HiddenPassConfig(
            points=HIDDEN_POINTS,
            impmin_coeff=hidden.impmin_coeff,
            # The type error IS the claim: the union has no nonlinearity member, so this
            # is unrepresentable statically as well as at parse.
            recon=[*hidden.recon, _nonlinearity_cfg()],  # pyright: ignore[reportArgumentType]
        )
