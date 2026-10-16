"""The pruning engine on a tiny all-sites decomposed GLU: the factored forward against the
decomposition's own masked forward, the dense reference against the clean forward, partial
forwards from every block, flips, the mask gradient, and the estimator math."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from param_decomp.ci_filter.step import all_logits, lm_batch, masked_output
from param_decomp.core.components import ComponentStacks, SiteC
from param_decomp.core.init_placed import (
    init_model_component_stacks_placed,
    random_component_initializer,
)
from param_decomp.core.model import MaterializedMasking
from param_decomp.core.placement import from_config
from param_decomp.core.precision import COMPUTE_DT
from param_decomp.core.sharding import place_target, single_device_mesh
from param_decomp.experiments.lm.load_run import PlacedTransformer
from param_decomp.pruning import model as pm
from param_decomp.pruning.config import PruneConfig
from param_decomp.pruning.engine import (
    flip_on_groups,
    flip_on_sample,
    kl_int_prompts,
    make_sample_cache,
    mask_gradient,
    score_prompts,
    target_logp,
)
from param_decomp.pruning.problem import Problem
from param_decomp.pruning.prune import PruneState, run_blockwise, run_rounds, score_set
from param_decomp.pruning.run import combine
from param_decomp.pruning.screen import (
    Context,
    Precisions,
    lean_screen,
    rich_screen,
    sample_caches,
    target_on_sample,
)
from param_decomp.pruning.stats import (
    QUANTITIES,
    kl,
    log_softmax32,
    shift_sums,
    stratified,
    weighted_moments,
)
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm
from param_decomp.targets.transformer import (
    KIND_ORDER,
    TransformerDecomposedModel,
    glu_site_specs,
    site_name,
)

C = 8
CE = QUANTITIES.index("ce")
CE_INT = QUANTITIES.index("ce_int")

type Setup = tuple[
    Mesh,
    PlacedTransformer,
    ComponentStacks,
    dict[str, np.ndarray],
    pm.Arch,
    pm.Shared,
    pm.Factors,
    pm.DenseWeights,
    pm.Slots,
    jax.Array,
]


@pytest.fixture(scope="module")
def setup() -> Setup:
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(
        cfg,
        tuple(
            SiteC(site_name(layer, kind), C) for layer in range(cfg.n_layer) for kind in KIND_ORDER
        ),
    )
    mesh = single_device_mesh()
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    rules = from_config("ddp", mesh, sites)
    placed: PlacedTransformer = place_target(model, rules)
    with jax.set_mesh(mesh):
        components: ComponentStacks = init_model_component_stacks_placed(
            placed, jax.random.PRNGKey(1), rules, random_component_initializer
        )
    rng = np.random.default_rng(0)
    alive = {s.name: rng.random(C) < 0.7 for s in sites}
    for name in alive:  # at least one per site, and one site with every component
        alive[name][0] = True
    alive[site_name(3, "down")][:] = True
    assert isinstance(placed.model, TransformerDecomposedModel)
    arch, shared, factors, dense, slots = pm.build(placed.model, components, alive)
    tokens = jnp.asarray(rng.integers(1, 64, size=(6, 5)), jnp.int32)
    return mesh, placed, components, alive, arch, shared, factors, dense, slots, tokens


def _flip_entry(slots: pm.Slots, a: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    return (
        jnp.asarray(slots.kind[a], jnp.int32),
        jnp.asarray(slots.layer[a], jnp.int32),
        jnp.asarray(slots.slot[a], jnp.int32),
    )


def test_factored_forward_matches_the_decomposition(setup: Setup) -> None:
    mesh, placed, components, alive, arch, shared, factors, dense, slots, tokens = setup
    on = np.ones(slots.n, bool)
    mask = slots.mask(on, arch.n_layer)
    ours = pm.factored_logits(arch, shared, factors, mask, pm.Flips.none(6), tokens, "fp32", "fp32")
    with jax.set_mesh(mesh):
        prepared = placed.prepare_compute_weights(components)
        masks = {
            name: jnp.broadcast_to(jnp.asarray(keep, COMPUTE_DT), (*tokens.shape, C))
            for name, keep in alive.items()
        }
        theirs = all_logits(
            masked_output(
                placed,
                prepared,
                lm_batch(tokens),
                MaterializedMasking(component_masks=masks),
                remat=False,
            )
        )[:, -1, :]
        clean = all_logits(placed.clean_forward(lm_batch(tokens)).output)[:, -1, :]
    # The decomposition computes in bf16; ours is the same model in fp32.
    gap = kl(log_softmax32(ours), log_softmax32(theirs))
    assert float(jnp.max(gap)) < 1e-3, gap
    target = pm.dense_logits(arch, shared, dense, tokens, "fp32", "fp32")
    assert float(jnp.max(kl(log_softmax32(target), log_softmax32(clean)))) < 1e-3
    # Components drawn at random are not the model.
    assert float(jnp.mean(kl(log_softmax32(target), log_softmax32(ours)))) > 1e-3


def test_partial_forward_and_flips(setup: Setup) -> None:
    *_, arch, shared, factors, _dense, slots, tokens = setup
    on = np.ones(slots.n, bool)
    mask = slots.mask(on, arch.n_layer)
    none = pm.Flips.none(6)
    inputs, final = pm.block_inputs(arch, shared, factors, mask, tokens, "fp32")
    full = pm.factored_logits(arch, shared, factors, mask, none, tokens, "fp32", "fp32")
    np.testing.assert_allclose(
        np.asarray(pm.last_logits(arch, shared, final, "fp32")), np.asarray(full), atol=1e-5
    )
    for start in range(arch.n_layer):
        resid = pm.run_blocks(arch, shared, factors, mask, none, inputs[start], start, "fp32")
        np.testing.assert_allclose(
            np.asarray(pm.last_logits(arch, shared, resid, "fp32")), np.asarray(full), atol=1e-5
        )
    # Removing component `a` by a flip from its own block = zeroing it in the global mask.
    for a in (0, slots.n // 2, slots.n - 1):
        kind, layer, slot = _flip_entry(slots, a)
        flips = pm.Flips.same(6, kind, layer, slot, jnp.asarray(-1.0))
        resid = pm.run_blocks(arch, shared, factors, mask, flips, inputs[int(layer)], layer, "fp32")
        without = on.copy()
        without[a] = False
        refold = pm.factored_logits(
            arch, shared, factors, slots.mask(without, arch.n_layer), none, tokens, "fp32", "fp32"
        )
        np.testing.assert_allclose(
            np.asarray(pm.last_logits(arch, shared, resid, "fp32")), np.asarray(refold), atol=1e-5
        )
        assert not np.allclose(np.asarray(refold), np.asarray(full), atol=1e-4)


def test_flip_passes_agree_and_sign_convention(setup: Setup) -> None:
    *_, arch, shared, factors, dense, slots, tokens = setup
    rng = np.random.default_rng(1)
    on = rng.random(slots.n) < 0.8
    mask = slots.mask(on, arch.n_layer)
    y = jnp.asarray(rng.integers(0, 64, size=6), jnp.int32)
    r = jnp.asarray(rng.integers(0, 10, size=6), jnp.int32)
    # The tiny vocab has 64 tokens: map "integers" 0..999 onto them (only the arithmetic matters).
    int_ids = jnp.asarray(np.arange(1000) % 64, jnp.int32)
    logp_t = target_logp(arch, shared, dense, tokens, "fp32", "fp32")
    cache = make_sample_cache(
        arch, shared, factors, mask, logp_t, tokens, y, r, int_ids, "fp32", "fp32"
    )
    base_ce = score_prompts(
        arch, shared, factors, mask, None, tokens, y, r, int_ids, "fp32", "fp32"
    )[CE]
    for a in (int(np.nonzero(on)[0][0]), int(np.nonzero(~on)[0][0])):
        kind, layer, slot = _flip_entry(slots, a)
        sign = jnp.asarray(-1.0 if on[a] else 1.0)
        res = flip_on_sample(
            arch, shared, factors, mask, cache, kind, layer, slot, sign, int_ids, "fp32", "fp32"
        )
        without = on.copy()
        without[a] = False
        with_c = on.copy()
        with_c[a] = True
        ce_without = score_prompts(
            arch, shared, factors, slots.mask(without, arch.n_layer), None, tokens, y, r,
            int_ids, "fp32", "fp32",
        )[CE]  # fmt: skip
        ce_with = score_prompts(
            arch, shared, factors, slots.mask(with_c, arch.n_layer), None, tokens, y, r,
            int_ids, "fp32", "fp32",
        )[CE]  # fmt: skip
        np.testing.assert_allclose(
            np.asarray(res.delta[CE]), np.asarray(ce_without - ce_with), atol=1e-4
        )
        np.testing.assert_allclose(np.asarray(res.without[CE]), np.asarray(ce_without), atol=1e-4)
        # The grouped H_c pass (full forwards, per-row flips) gives the same per-prompt delta.
        group = flip_on_groups(
            arch, shared, factors, dense, mask,
            tokens[None], y[None], r[None], jnp.ones((1, 6)),
            kind[None], layer[None], slot[None], sign[None], int_ids, "fp32", "fp32", "fp32",
        )  # fmt: skip
        np.testing.assert_allclose(
            np.asarray(group.delta[0, CE]), np.asarray(res.delta[CE]), atol=1e-4
        )
        np.testing.assert_allclose(np.asarray(group.dq_x[0, 0]), np.asarray(res.dq_x), atol=1e-5)
        np.testing.assert_allclose(
            np.asarray(group.sums[0, 0, 1, CE, 0]), float(jnp.sum(res.delta[CE])), rtol=1e-4
        )
    del base_ce


def test_mask_gradient_is_the_derivative(setup: Setup) -> None:
    *_, arch, shared, factors, _dense, slots, tokens = setup
    rng = np.random.default_rng(2)
    on = rng.random(slots.n) < 0.8
    mask = slots.mask(on, arch.n_layer)
    y = jnp.asarray(rng.integers(0, 64, size=6), jnp.int32)
    weight = jnp.ones((6,))
    int_ids = jnp.asarray(np.arange(1000) % 64, jnp.int32)
    r = jnp.asarray(rng.integers(0, 64, size=6), jnp.int32)
    h = 2e-2  # central difference: O(h²) bias, and fp32 sums need a step well above 1e-3
    reference = slots.mask(np.ones(slots.n, bool), arch.n_layer)
    for objective in ("ce", "ce_int", "kl_int"):
        grad = slots.gather(
            mask_gradient(
                arch, shared, factors, mask, reference, tokens, y, r, int_ids, weight, "fp32",
                objective,
            )
        )  # fmt: skip
        for a in (0, int(np.nonzero(~on)[0][0]), slots.n - 1):
            kind = KIND_ORDER[slots.kind[a]]
            bumped = {k: v for k, v in mask.items()}
            lo = {k: v for k, v in mask.items()}
            bumped[kind] = mask[kind].at[slots.layer[a], slots.slot[a]].add(h)
            lo[kind] = mask[kind].at[slots.layer[a], slots.slot[a]].add(-h)

            def f(m: pm.Mask, objective: str = objective) -> float:
                if objective == "kl_int":
                    return float(
                        jnp.sum(
                            kl_int_prompts(
                                arch, shared, factors, m, reference, tokens, int_ids, "fp32"
                            )
                        )
                    )
                quantity = QUANTITIES.index(objective)
                return float(
                    jnp.sum(
                        score_prompts(
                            arch, shared, factors, m, None, tokens, y, r, int_ids, "fp32", "fp32"
                        )[quantity]
                    )
                )

            np.testing.assert_allclose(grad[a], (f(bumped) - f(lo)) / (2 * h), rtol=2e-2, atol=1e-4)


def test_shift_sums_hand_case() -> None:
    dq = jnp.zeros((2, 1000)).at[0, 7].set(0.5).at[1, 3].set(-0.25)
    r = jnp.asarray([5, 3])
    by_x, by_rel = shift_sums(dq, r, jnp.asarray([1.0, 2.0]))
    assert float(by_x[7]) == 0.5 and float(by_x[3]) == -0.5
    assert float(by_rel[999 + 2]) == 0.5  # x - r = 7 - 5
    assert float(by_rel[999 + 0]) == -0.5  # 3 - 3, weight 2


def test_stratified_estimator_is_unbiased() -> None:
    """A population with a heavy stratum H: the estimator's mean over many draws of R matches
    the population mean, and its SE matches the spread of the estimates."""
    rng = np.random.default_rng(3)
    n_total, n_h, n_r = 5000, 50, 400
    population = rng.normal(0.0, 1.0, n_total)
    h = np.arange(n_h)
    population[h] += 30.0
    truth = population.mean()
    sum_h = np.array([np.sum(population[h] ** k) for k in (1, 2, 3)])
    estimates, ses = [], []
    for _ in range(2000):
        sample = rng.choice(n_total, n_r, replace=False)
        in_h = sample < n_h
        sum_r = np.array([np.sum(population[sample] ** k) for k in (1, 2, 3)])
        sum_rh = np.array([np.sum(population[sample[in_h]] ** k) for k in (1, 2, 3)])
        mean, se = stratified(n_total, sum_h, sum_r, sum_rh, n_h, np.asarray(n_r - in_h.sum()))
        estimates.append(float(mean))
        ses.append(float(se))
    assert abs(np.mean(estimates) - truth) < 3 * np.std(estimates) / np.sqrt(len(estimates))
    assert abs(np.mean(ses) / np.std(estimates) - 1.0) < 0.1
    mean, sd, _ = weighted_moments(
        n_total, sum_h, sum_h * 0 + np.array([np.sum(population[n_h:n_h + n_r] ** k) for k in (1, 2, 3)]),
        np.zeros(3), n_h, np.asarray(n_r),
    )  # fmt: skip
    assert abs(float(mean) - truth) < 0.2 and abs(float(sd) - population.std()) < 0.5


# ------------------------------- screens, table, rounds -------------------------------


def _context(setup: Setup, sample_all: bool) -> Context:
    *_, arch, shared, factors, dense, slots, _tokens = setup
    rng = np.random.default_rng(5)
    n = 48
    r = rng.integers(1, 30, size=n)
    int_ids = np.arange(1000, dtype=np.int32) % 64
    problem = Problem(
        tokens=rng.integers(1, 64, size=(n, 5)).astype(np.int32),
        pool_rows=np.arange(n),
        op=np.zeros(n, np.int64),
        a=r,
        b=np.zeros(n, np.int64),
        r=r,
        y=int_ids[r],
        int_ids=int_ids,
    )
    sample = np.arange(n) if sample_all else np.sort(rng.choice(n, 16, replace=False))
    top = np.stack([rng.choice(n, 4, replace=False) for _ in range(slots.n)])
    return Context(
        arch=arch, shared=shared, factors=factors, dense=dense, slots=slots, problem=problem,
        sample=sample, top=top, precisions=Precisions("fp32", "fp32", "fp32"),
        batch_size=16, group_size=3,
    )  # fmt: skip


def _exact_delta(ctx: Context, on: np.ndarray, a: int) -> float:
    """`CE(S without a) − CE(S with a)` over every prompt."""
    without, with_a = on.copy(), on.copy()
    without[a], with_a[a] = False, True
    everything = np.arange(ctx.problem.n)
    n_layer = ctx.arch.n_layer
    lo = score_set(ctx, ctx.slots.mask(without, n_layer), everything, "fp32", False)[CE]
    hi = score_set(ctx, ctx.slots.mask(with_a, n_layer), everything, "fp32", False)[CE]
    return float(np.mean(lo - hi))


def test_lean_screen_is_exact_when_the_sample_is_everything(setup: Setup) -> None:
    """With `R = P` the stratified weight is 1 and the estimate is the exact mean, for removals
    and additions alike; the SE is then the finite-population SE (not checked)."""
    ctx = _context(setup, sample_all=True)
    on = np.random.default_rng(6).random(ctx.slots.n) < 0.7
    mask = ctx.slots.mask(on, ctx.arch.n_layer)
    caches = sample_caches(ctx, mask, target_on_sample(ctx))
    comps = np.asarray([int(np.nonzero(on)[0][0]), int(np.nonzero(~on)[0][0]), ctx.slots.n - 1])
    est, se = lean_screen(ctx, mask, on, comps, caches)
    for i, a in enumerate(comps):
        np.testing.assert_allclose(est[i], _exact_delta(ctx, on, int(a)), atol=1e-5)
    assert (se >= 0).all()


def test_rich_screen_combine_and_rounds(setup: Setup, tmp_path: Path) -> None:
    ctx = _context(setup, sample_all=True)
    on0 = np.ones(ctx.slots.n, bool)
    mask0 = ctx.slots.mask(on0, ctx.arch.n_layer)
    caches = sample_caches(ctx, mask0, target_on_sample(ctx))
    rich_screen(ctx, mask0, on0, tmp_path / "screen", caches)
    combine(ctx, tmp_path)
    table = np.genfromtxt(
        tmp_path / "components.tsv", delimiter="\t", names=True, dtype=None, encoding=None
    )
    assert table.shape == (ctx.slots.n,)
    for a in (0, ctx.slots.n // 2):
        np.testing.assert_allclose(table["d_ce"][a], _exact_delta(ctx, on0, a), atol=1e-5)
    vectors = np.load(tmp_path / "vectors.npz")
    assert vectors["dq_x"].shape == (ctx.slots.n, 1000)
    assert vectors["rel_mod_20"].shape == (ctx.slots.n, 20)
    # Δq is a difference of distributions: it sums to zero over x and over d.
    np.testing.assert_allclose(vectors["dq_x"].sum(axis=1), 0.0, atol=1e-5)
    np.testing.assert_allclose(vectors["dq_rel"].sum(axis=1), 0.0, atol=1e-5)

    config = PruneConfig.model_validate(
        {
            "run": {"kind": "id", "id": "p-test"},
            "step": 0,
            "filter_id": "cf-test",
            "pool": {},
            "epsilon": 1e-3,
            "max_rounds": 4,
            "max_fresh_screens": 1,
            "grad_batch_size": 16,
            "objective": "ce_int",
        }
    )
    check = np.arange(ctx.problem.n)
    ce_s0 = float(np.mean(score_set(ctx, mask0, check, "fp32", False)[CE_INT]))
    state = PruneState(
        on=on0.copy(),
        est=np.asarray(table["d_ce_int"], np.float64),
        se=np.asarray(table["se_ce_int"], np.float64),
        readded=np.full(ctx.slots.n, -1),
        removed=np.full(ctx.slots.n, -1),
    )
    final = run_rounds(config, ctx, state, check, ce_s0, tmp_path)
    rounds = (tmp_path / "rounds.jsonl").read_text().splitlines()
    assert rounds and (tmp_path / "sets").exists()
    ce_final = float(
        np.mean(
            score_set(ctx, ctx.slots.mask(final.on, ctx.arch.n_layer), check, "fp32", False)[CE_INT]
        )
    )
    assert ce_final - ce_s0 <= config.global_budget + 1e-6
    # A resumed run continues the numbering instead of overwriting the first rounds.
    n_sets = len(list((tmp_path / "sets").glob("S_*.npz")))
    assert final.next_round == n_sets == len(rounds)
    resumed = run_rounds(
        config.model_copy(update={"max_rounds": final.next_round + 1}),
        ctx,
        final,
        check,
        ce_s0,
        tmp_path,
    )
    assert resumed.next_round <= final.next_round + 1
    records = [json.loads(line) for line in (tmp_path / "rounds.jsonl").read_text().splitlines()]
    assert [r["round"] for r in records] == list(range(len(records)))


def test_engine_runs_under_the_consumer_mesh(setup: Setup) -> None:
    """`run.py` drives every pass inside `jax.set_mesh` (the restored run's explicit mesh)."""
    mesh = setup[0]
    with jax.set_mesh(mesh):
        test_lean_screen_is_exact_when_the_sample_is_everything(setup)
        test_mask_gradient_is_the_derivative(setup)


def test_build_and_select_under_a_zero1_placement() -> None:
    """The restored run's layout (`zero1`, explicit fsdp/tp-typed arrays): building the factored
    model and selecting `H_c` with a real CI fn both run, and the selection is the top CI."""
    from dataclasses import replace

    from param_decomp.ci_filter.step import UNCONSTRAINED, gather_batch, output_ci
    from param_decomp.pruning.select import select_top_ci
    from param_decomp.targets.testing import tiny_glu_chunkwise_ci_fn_arch
    from param_decomp.tests.placed_ci_fn import placed_ci_fn

    cfg = tiny_glu_cfg()
    sites = glu_site_specs(
        cfg,
        tuple(
            SiteC(site_name(layer, kind), C) for layer in range(cfg.n_layer) for kind in KIND_ORDER
        ),
    )
    mesh = single_device_mesh()
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    rules = from_config("zero1", mesh, sites)
    placed: PlacedTransformer = place_target(model, rules)
    arch_ci = replace(tiny_glu_chunkwise_ci_fn_arch(model, n_blocks=1), dual=True)
    rng = np.random.default_rng(0)
    alive = {s.name: rng.random(C) < 0.6 for s in sites}
    for name in alive:
        alive[name][0] = True
    tokens = rng.integers(1, 64, size=(20, 5)).astype(np.int32)
    with jax.set_mesh(mesh):
        ci_fn = placed_ci_fn(arch_ci, sites, jax.random.PRNGKey(3), mesh, rules)
        components = init_model_component_stacks_placed(
            placed, jax.random.PRNGKey(1), rules, random_component_initializer
        )
        prepared = placed.prepare_compute_weights(components)
        assert isinstance(placed.model, TransformerDecomposedModel)
        arch, shared, factors, _dense, slots = pm.build(placed.model, components, alive)
        mask = slots.mask(np.ones(slots.n, bool), arch.n_layer)
        logits = pm.factored_logits(
            arch, shared, factors, mask, pm.Flips.none(4), jnp.asarray(tokens[:4]), "fp32", "fp32"
        )
        assert bool(jnp.all(jnp.isfinite(logits)))
        top, values = select_top_ci(
            placed, prepared, ci_fn, UNCONSTRAINED, tokens, slots, k=3, batch_size=8
        )
        clean = placed.clean_forward(
            gather_batch(jnp.asarray(tokens), jnp.arange(20)), ci_fn.capture_keys
        )
        lower = output_ci(placed, prepared, ci_fn, UNCONSTRAINED, clean, remat=False).lower
    a = 5
    ci = np.asarray(lower[slots.sites[a]], np.float32)[:, :, slots.index[a]].max(axis=1)
    np.testing.assert_allclose(np.sort(values[a])[::-1], np.sort(ci)[::-1][:3], rtol=1e-2)
    assert set(top[a].tolist()) <= set(range(20))


def test_blockwise_pruning(setup: Setup, tmp_path: Path) -> None:
    ctx = _context(setup, sample_all=True)
    n = ctx.slots.n
    on0 = np.ones(n, bool)
    mask0 = ctx.slots.mask(on0, ctx.arch.n_layer)
    config = PruneConfig.model_validate(
        {
            "run": {"kind": "id", "id": "p-test"},
            "step": 0,
            "filter_id": "cf-test",
            "pool": {},
            "epsilon": 1e-3,
            "objective": "ce_int",
            "mode": "blockwise",
            "max_rounds": 3,
            "round_precision": "tf32",
        }
    )
    check = np.arange(ctx.problem.n)
    ce_s0 = float(np.mean(score_set(ctx, mask0, check, "fp32", False)[CE_INT]))
    state = PruneState(
        on=on0.copy(), est=np.zeros(n), se=np.zeros(n),
        readded=np.full(n, -1), removed=np.full(n, -1),
    )  # fmt: skip
    final = run_blockwise(config, ctx, state, check, ce_s0, tmp_path)
    records = [json.loads(line) for line in (tmp_path / "rounds.jsonl").read_text().splitlines()]
    assert [r["block"] for r in records] == list(range(ctx.arch.n_layer))
    assert final.next_round == ctx.arch.n_layer
    assert not (tmp_path / "sets" / "partial.npz").exists()
    # Removals only ever happen in the block being pruned.
    was_removed = final.removed >= 0
    np.testing.assert_array_equal(final.removed[was_removed], ctx.slots.layer[was_removed])
    on_mask = ctx.slots.mask(final.on, ctx.arch.n_layer)
    ce_final = float(np.mean(score_set(ctx, on_mask, check, "fp32", False)[CE_INT]))
    assert ce_final - ce_s0 <= config.global_budget + 1e-6


def test_temperature_fit() -> None:
    """A pure temperature change is fully explained (R² 1, the right β); a random change is not."""
    from param_decomp.pruning.engine import temperature_fit

    rng = np.random.default_rng(7)
    logits = jnp.asarray(rng.normal(0, 2, size=(3, 50)), jnp.float32)
    logp_b = log_softmax32(logits)
    fit = temperature_fit(logp_b, log_softmax32(1.3 * logits))
    np.testing.assert_allclose(np.asarray(fit.r2), 1.0, atol=1e-4)
    np.testing.assert_allclose(np.asarray(fit.beta), 1.3, atol=1e-3)
    assert float(jnp.min(fit.kl)) > 0
    other = temperature_fit(logp_b, log_softmax32(logits + rng.normal(0, 2, size=(3, 50))))
    assert float(jnp.max(other.r2)) < 0.5


def test_learned_mask(setup: Setup, tmp_path: Path) -> None:
    """A large λ prunes components, the run resumes from `state.npz`, and `final.json` scores
    the last mask on the check set."""
    from param_decomp.pruning.learn import kl_table, learn_mask, run_learn

    ctx = _context(setup, sample_all=True)
    check = np.arange(0, ctx.problem.n, 2)
    config = PruneConfig.model_validate(
        {
            "run": {"kind": "id", "id": "p-test"},
            "step": 0,
            "filter_id": "cf-test",
            "pool": {},
            "full_precision": "fp32",
            "grad_batch_size": 8,
            "learn_lambdas": [1.0],
            "learn_steps": 4,
            "learn_batch": 8,
            "learn_lr": 0.5,
            "learn_eval_every": 2,
            "learn_eval_size": 8,
            "learn_objective": "kl_int",
        }
    )
    run_learn(config, ctx, check, tmp_path)
    d = tmp_path / "learn" / "lam_1"
    final = json.loads((d / "final.json").read_text())
    assert final["size"] < ctx.slots.n
    assert final["prompts"] == check.size
    assert [json.loads(r)["step"] for r in (d / "log.jsonl").read_text().splitlines()] == [2, 4]
    assert (tmp_path / "learn" / "baseline.json").exists()
    assert "kl_int" in final["all"]
    on = learn_mask(config, ctx, check, 1.0, tmp_path)
    assert int(on.sum()) == final["size"]
    cosine = config.model_copy(
        update={"learn_schedule": "cosine", "learn_name": "cos", "learn_sampling": True}
    )
    run_learn(cosine, ctx, check, tmp_path)
    assert (tmp_path / "learn" / "cos" / "lam_1" / "final.json").exists()
    screen = {"d_ce_int": np.linspace(-1, 1, ctx.slots.n)}
    with (tmp_path / "components.tsv").open("w") as f:
        f.write("d_ce_int\n" + "\n".join(f"{v}" for v in screen["d_ce_int"]) + "\n")
    gated = config.model_copy(
        update={"learn_name": "gated", "learn_max_screen_d_ce_int": 0.0, "learn_sampling": True}
    )
    run_learn(gated, ctx, check, tmp_path)
    kept = np.load(tmp_path / "learn" / "gated" / "lam_1" / "state.npz")["logit"] > 0
    assert kept[screen["d_ce_int"] > 0].all()
    kl_table(config, ctx, check, tmp_path)
    table = json.loads((tmp_path / "learn" / "kl_table.json").read_text())
    assert table["s0"]["kl_int_s0"] == pytest.approx(0.0, abs=1e-6)
    assert table["s0"]["kl_full_s0"] == pytest.approx(0.0, abs=1e-6)
    assert set(table) == {"s0", "lam_1", "cos/lam_1", "gated/lam_1"}
    changed = config.model_copy(update={"learn_lr": 0.1})
    with pytest.raises(AssertionError, match="learn settings differ"):
        learn_mask(changed, ctx, check, 1.0, tmp_path)
