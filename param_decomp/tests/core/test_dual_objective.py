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
from pydantic import ValidationError

from param_decomp.core.ci_fn import LayerwiseMLPCIArch, LayerwiseMLPCIFn, build_ci_fn
from param_decomp.core.components import SiteC, init_component_stacks
from param_decomp.core.configs import (
    CIMaskedReconLossConfig,
    HiddenActsNormalization,
    HiddenPassConfig,
    ImportanceMinimalityLossConfig,
    NonlinearityLocalityLossConfig,
    NontargetConfig,
    NontargetHiddenConfig,
    PerPositionHiddenActsNormalization,
    StochasticReconLossConfig,
    TargetedLossMetricConfig,
    UnmaskedReconLossConfig,
)
from param_decomp.core.model import PlacedModel
from param_decomp.core.objective import build_targeted_objective
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.train import (
    CIScaledWeightDecay,
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
        ImportanceMinimalityLossConfig(coeff=3e-3, gamma=ScheduleConfig.constant(1.0)),
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


def _hidden(
    normalization: HiddenActsNormalization | None = None,
    output_coeff: float | None = None,
) -> tuple[HiddenPassConfig, NontargetConfig]:
    hidden = HiddenPassConfig(
        points=HIDDEN_POINTS,
        **({} if normalization is None else {"normalization": normalization}),
        output_coeff=output_coeff,
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


def _setup(
    *,
    dual: bool,
    sequential: bool,
    hidden_only: bool = False,
    mask_free_target_output: bool = False,
    ci_masked_target_output: bool = False,
    nonlinearity_coeff: float | None = None,
    normalization: HiddenActsNormalization | None = None,
    output_coeff: float | None = None,
    ci_scaled_weight_decay: CIScaledWeightDecay | None = None,
):
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
        hidden, nontarget = _hidden(normalization, output_coeff)
        if hidden_only:
            # The hidden-only objective: the broad stream authors NO output recon terms, so
            # the step must not build a non-target output pass at all.
            nontarget = NontargetConfig(
                batch_size=nontarget.batch_size,
                impmin_coeff=nontarget.impmin_coeff,
                recon=[],
                hidden=nontarget.hidden,
            )
    else:
        assert not hidden_only, "hidden-only needs the hidden pass"
        hidden, nontarget = (
            None,
            NontargetConfig(
                batch_size=32, impmin_coeff=6e-3, recon=[StochasticReconLossConfig(coeff=1.0)]
            ),
        )
    loss_metrics = _loss_metrics()
    if mask_free_target_output:
        # The production hidden-only shape: the target OUTPUT pass keeps importance-minimality
        # and the mask-free faithfulness anchor, and no CI-masked term at all.
        loss_metrics = (loss_metrics[0], UnmaskedReconLossConfig(coeff=0.5))
    if ci_masked_target_output:
        loss_metrics = (loss_metrics[0], CIMaskedReconLossConfig(coeff=1.0))
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
        ci_scaled_weight_decay=ci_scaled_weight_decay,
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


@pytest.mark.parametrize("sequential", [False, True])
def test_hidden_pass_per_position_normalization_traces_and_moves_the_hidden_head(
    sequential: bool,
):
    """S35 amended 2026-09-11: the per-position form (a median inside the pass's graph)
    traces under jit on both scheduling paths, gives finite per-point losses that differ from
    the batch form's, and still reaches the hidden head."""
    per_position = PerPositionHiddenActsNormalization(floor_fraction=0.01)
    cfg, state, step = _setup(dual=True, sequential=sequential, normalization=per_position)
    target_batch, broad = _batches(cfg)
    before = state.decomposition.ci_fn
    assert isinstance(before, LayerwiseMLPCIFn)
    before_heads = {
        site: np.asarray(mlp.hidden_head[0])
        for site, mlp in before.site_mlps.items()
        if mlp.hidden_head is not None
    }
    new_state, metrics = step(_model_of(), state, target_batch, broad, jax.random.PRNGKey(7))
    after = new_state.decomposition.ci_fn
    assert isinstance(after, LayerwiseMLPCIFn)
    for site in ("linear1", "linear2"):
        head_after = after.site_mlps[site].hidden_head
        assert head_after is not None
        assert not np.allclose(before_heads[site], np.asarray(head_after[0]))
    per_point = {
        point: float(
            metrics[f"hidden_ci/loss/HiddenStochasticRecon/hidden_acts_reconstruction/{point}"]
        )
        for point in HIDDEN_POINTS
    }
    assert all(np.isfinite(v) for v in per_point.values()), per_point

    # The batch form on the same draws scores differently: the two are distinct objectives.
    _, state_b, step_b = _setup(dual=True, sequential=sequential)
    _, metrics_b = step_b(_model_of(), state_b, target_batch, broad, jax.random.PRNGKey(7))
    per_point_b = {
        point: float(
            metrics_b[f"hidden_ci/loss/HiddenStochasticRecon/hidden_acts_reconstruction/{point}"]
        )
        for point in HIDDEN_POINTS
    }
    assert any(not np.isclose(per_point[p], per_point_b[p], rtol=1e-3) for p in HIDDEN_POINTS), (
        per_point,
        per_point_b,
    )


@pytest.mark.parametrize("sequential", [False, True])
def test_hidden_pass_output_rider_adds_the_kl_to_the_hidden_objective(sequential: bool):
    """T12 amended 2026-09-11: with `output_coeff` the hidden pass scores
    `mean_points + output_coeff * KL` on its own forward, logs the bare KL as `e2e`, and the
    target-OUTPUT pass is untouched. Without it no `e2e` key exists on the hidden pass."""
    kappa = 0.5
    cfg, state, step = _setup(dual=True, sequential=sequential, output_coeff=kappa)
    target_batch, broad = _batches(cfg)
    _, metrics = step(_model_of(), state, target_batch, broad, jax.random.PRNGKey(7))
    prefix = "hidden_ci/loss/HiddenStochasticRecon"
    e2e = float(metrics[f"{prefix}/e2e"])
    assert np.isfinite(e2e) and e2e > 0.0
    per_point = [float(metrics[f"{prefix}/hidden_acts_reconstruction/{p}"]) for p in HIDDEN_POINTS]
    assert float(metrics[prefix]) == pytest.approx(np.mean(per_point) + kappa * e2e, rel=1e-5)
    # The rider is per pass: the non-target hidden pass authored none, so it logs no e2e.
    nt_prefix = "nontarget_data/hidden_ci/loss/StochasticReconLoss"
    assert nt_prefix in metrics and f"{nt_prefix}/e2e" not in metrics
    # And the pure hidden pass is unchanged: same draws, no e2e key, hidden term = point mean.
    _, state_b, step_b = _setup(dual=True, sequential=sequential)
    _, metrics_b = step_b(_model_of(), state_b, target_batch, broad, jax.random.PRNGKey(7))
    assert f"{prefix}/e2e" not in metrics_b
    per_point_b = [
        float(metrics_b[f"{prefix}/hidden_acts_reconstruction/{p}"]) for p in HIDDEN_POINTS
    ]
    assert float(metrics_b[prefix]) == pytest.approx(np.mean(per_point_b), rel=1e-5)


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


@pytest.mark.parametrize("sequential", [False, True])
def test_hidden_only_objective_builds_no_nontarget_output_pass(sequential: bool):
    """The hidden-only arm: `nontarget.recon: []` next to `nontarget.hidden` means the broad
    stream is judged at the hidden points ALONE. The pass must be absent, not zeroed — a
    pass built with a zero coefficient would still run its masked forward — so the proof is
    that its whole metric namespace is missing while the hidden one is present."""
    cfg, state, step = _setup(dual=True, sequential=sequential, hidden_only=True)
    target_batch, broad = _batches(cfg)
    _, metrics = step(_model_of(), state, target_batch, broad, jax.random.PRNGKey(7))

    assert "nontarget_data/hidden_ci/loss/total" in metrics
    assert not [
        k
        for k in metrics
        if k.startswith("nontarget_data/") and not k.startswith("nontarget_data/hidden_ci/")
    ], sorted(k for k in metrics if k.startswith("nontarget_data/"))
    # The target-OUTPUT pass is never skipped: it carries the mask-free anchor and owns the
    # unprefixed keys, so `passes[0]` stays put and no other metric key moves.
    assert "loss/StochasticReconLoss" in metrics and "total" in metrics
    assert "hidden_ci/loss/total" in metrics


@pytest.mark.parametrize("sequential", [False, True])
def test_hidden_only_target_output_pass_may_be_mask_free(sequential: bool):
    """The production hidden-only shape end to end: the target OUTPUT pass holds
    importance-minimality plus `UnmaskedReconLoss` and NOTHING CI-masked (so it carries no
    adversary to ascend), there is no non-target output pass, and both hidden passes run.
    The step must still trace, and the only masked forwards must be the hidden ones."""
    cfg, state, step = _setup(
        dual=True, sequential=sequential, hidden_only=True, mask_free_target_output=True
    )
    target_batch, broad = _batches(cfg)
    _, metrics = step(_model_of(), state, target_batch, broad, jax.random.PRNGKey(7))

    assert "loss/UnmaskedReconLoss" in metrics
    assert "hidden_ci/loss/total" in metrics
    assert "nontarget_data/hidden_ci/loss/total" in metrics
    assert not [
        k
        for k in metrics
        if k.startswith("nontarget_data/") and not k.startswith("nontarget_data/hidden_ci/")
    ]
    # The mask-free pass takes NO penalty on CI: importance-minimality and the frequency
    # term are penalties on the head it would otherwise push down, and the trunk is SHARED
    # (S37), so that gradient would reach the hidden head through the trunk and smuggle a
    # second CI objective into a run that claims to have one.
    assert float(metrics["imp"]) == 0.0
    assert float(metrics["freq"]) == 0.0


@pytest.mark.parametrize("sequential", [False, True])
def test_hidden_only_leaves_the_output_head_exactly_where_it_started(sequential: bool):
    """The claim `shapes_ci` exists to make true: with the target-output pass mask-free, the
    OUTPUT readout receives an exactly-zero cotangent — `ConstantSources` is
    `ci + (1-ci)*1.0`, whose derivative in `ci` is exactly zero — so it never moves, and at
    `zero_init_readout` (W = 0) it therefore contributes exactly zero trunk gradient. The
    hidden head and the shared trunk must still move: the hidden passes are the objective."""
    cfg, state, step = _setup(
        dual=True, sequential=sequential, hidden_only=True, mask_free_target_output=True
    )
    target_batch, broad = _batches(cfg)
    before = state.decomposition.ci_fn
    assert isinstance(before, LayerwiseMLPCIFn)
    # The OUTPUT head is the stack's final layer; `hidden_head` is the second one (S37).
    before_output = {s: np.asarray(m.weights[-1]) for s, m in before.site_mlps.items()}
    before_hidden = {
        s: np.asarray(m.hidden_head[0])
        for s, m in before.site_mlps.items()
        if m.hidden_head is not None
    }
    before_trunk = {s: np.asarray(m.weights[0]) for s, m in before.site_mlps.items()}

    new_state, _ = step(_model_of(), state, target_batch, broad, jax.random.PRNGKey(7))
    after = new_state.decomposition.ci_fn
    assert isinstance(after, LayerwiseMLPCIFn)
    for site in ("linear1", "linear2"):
        np.testing.assert_array_equal(
            before_output[site],
            np.asarray(after.site_mlps[site].weights[-1]),
            err_msg=f"the output head at {site} moved — something is still shaping it",
        )
        head_after = after.site_mlps[site].hidden_head
        assert head_after is not None
        assert not np.allclose(before_hidden[site], np.asarray(head_after[0]))
        assert not np.allclose(before_trunk[site], np.asarray(after.site_mlps[site].weights[0]))


@pytest.mark.parametrize("sequential", [False, True])
def test_a_ci_masked_only_pass_keeps_its_minimality(sequential: bool):
    """`CIMaskedReconLoss` carries `ConstantSources(0.0)` — `mask = ci + (1-ci)*0.0`, i.e.
    `mask = ci` — so it is as CI-dependent as a stochastic term even though its source type
    is the same one the unmasked arm uses. A pass built from those ALONE must keep its
    importance-minimality and frequency penalties: classifying by TYPE rather than by VALUE
    would silently disarm an existing objective."""
    cfg, state, step = _setup(dual=True, sequential=sequential, ci_masked_target_output=True)
    target_batch, broad = _batches(cfg)
    _, metrics = step(_model_of(), state, target_batch, broad, jax.random.PRNGKey(7))
    assert float(metrics["imp"]) > 0.0
    assert "loss/CIMaskedReconLoss" in metrics


def test_empty_nontarget_recon_needs_a_hidden_pass() -> None:
    """An empty output list with no hidden pass would leave the broad stream carrying
    importance-minimality and nothing to reconstruct — refused at parse."""
    with pytest.raises(ValidationError, match="nontarget.recon is empty"):
        NontargetConfig(batch_size=32, impmin_coeff=6e-3, recon=[])


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


def _pin_heads(state: TrainState, *, output_ci: float, hidden_ci: float) -> TrainState:
    """Freeze the dual CI landscape so T11's statistic is known exactly: every trunk weight
    and both readout heads' weights zeroed, each head's BIAS pinned to saturation, so every
    component reads CI `output_ci` from the output head and `hidden_ci` from the hidden one
    — independent of the batch, the stream, and the site."""
    ci_fn = jax.tree.map(jnp.zeros_like, state.decomposition.ci_fn)
    assert isinstance(ci_fn, LayerwiseMLPCIFn)
    logit = {1.0: 30.0, 0.0: -30.0}  # saturates the CI squashing in either direction
    pinned = ci_fn
    for site in ci_fn.site_mlps:
        mlp = ci_fn.site_mlps[site]
        assert mlp.hidden_head is not None, "the dual fn carries a second readout head (S37)"
        pinned = eqx.tree_at(
            lambda f, site=site: (
                f.site_mlps[site].biases[-1],
                f.site_mlps[site].hidden_head[1],
            ),
            pinned,
            (
                jnp.full_like(mlp.biases[-1], logit[output_ci]),
                jnp.full_like(mlp.hidden_head[1], logit[hidden_ci]),
            ),
        )
    return TrainState(
        decomposition=Decomposition(components=state.decomposition.components, ci_fn=pinned),
        training=state.training,
    )


@pytest.mark.parametrize(
    ("role", "expected_decay"),
    [("all", 0.0), ("hidden", 0.0), ("output", 0.2)],
)
def test_ci_scaled_weight_decay_role_selects_which_head_counts_as_alive(
    role: str, expected_decay: float
) -> None:
    """T11 amended: `role` decides WHICH readout head's CI keeps a component alive.

    The CI fn is pinned so every component is DEAD to the output head and SATURATED on the
    hidden head — a hidden-reconstruction-only component, the shape the default rule
    protects. `all` and `hidden` therefore decay nothing; `output` decays everything at the
    full `lr·wd` rate, which is the point of the knob: it is how you ask for a decay that
    ignores the hidden objective."""
    wd = CIScaledWeightDecay(
        coeff=0.2,
        components_lr=ScheduleConfig.constant(1.0),
        role=role,  # pyright: ignore[reportArgumentType]
    )
    cfg, state, step = _setup(dual=True, sequential=False, ci_scaled_weight_decay=wd)
    state = _pin_heads(state, output_ci=0.0, hidden_ci=1.0)
    target_batch, broad = _batches(cfg)
    _, metrics = step(_model_of(), state, target_batch, broad, jax.random.PRNGKey(5))
    assert float(metrics["ci_scaled_weight_decay/max"]) == pytest.approx(expected_decay, abs=1e-6)
    assert float(metrics["ci_scaled_weight_decay/mean"]) == pytest.approx(expected_decay, abs=1e-6)


def test_ci_scaled_weight_decay_refuses_a_role_the_run_does_not_carry() -> None:
    """The selection is resolved against the passes the objective built, at construction:
    asking a single-head run for the hidden head would otherwise leave an empty reduction
    (or, worse, a silent full-rate decay of every component)."""
    wd = CIScaledWeightDecay(coeff=0.2, components_lr=ScheduleConfig.constant(1.0), role="hidden")
    with pytest.raises(AssertionError, match="selects no pass"):
        _setup(dual=False, sequential=False, ci_scaled_weight_decay=wd)
