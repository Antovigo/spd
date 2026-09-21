"""The CI filter's objective math, pool, template configs, and its jitted step, evaluation,
grid collection and CI-fn checkpoint on a tiny placed GLU over a one-device mesh (the
explicit-sharding path the 8B run takes)."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from param_decomp.ci_filter.ablation import MASKINGS, ablation_rows
from param_decomp.ci_filter.attribution import (
    answer_targets,
    make_ablation_metrics,
    make_ablation_scores,
    make_ablation_state,
    make_attribution_batch,
)
from param_decomp.ci_filter.checkpoint import (
    restore_ci_fn,
    save_ci_fn,
    save_kept,
    starting_ci,
    trained_ci,
)
from param_decomp.ci_filter.config import (
    ArithmeticPoolConfig,
    CIFilterConfig,
    InitFromCIFilter,
    LastPositionAnswerCE,
    LastPositionIntegerKL,
    MergedPPGDRecon,
    PGDEvalConfig,
)
from param_decomp.ci_filter.grid import collect_operation, write_applet, write_operation
from param_decomp.ci_filter.nontarget import (
    make_probe_baseline,
    make_probe_ci,
    make_probe_component,
)
from param_decomp.ci_filter.objective import kl_rows, objective_rows, row_scores
from param_decomp.ci_filter.paths import CIFilterOutputs
from param_decomp.ci_filter.pool import answer_token_ids, build_pool
from param_decomp.ci_filter.ppgd import PPGDStep, init_adversary
from param_decomp.ci_filter.step import (
    UNCONSTRAINED,
    CIConstraints,
    _capped,
    batch_gradients,
    ci_masked_micro_call,
    index_batches,
    make_apply_update,
    make_eval_batch,
    make_micro_grads,
    make_optimizer,
    make_pgd_eval,
    make_score_fixed_masks,
    pgd_eval_batches,
    remove_components,
    sample_batch,
    trainable,
)
from param_decomp.core.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    ChunkwiseTransformerCIFn,
    MHACIAttention,
    PlacedCIFn,
    build_ci_fn,
    resolve_ci_placement,
)
from param_decomp.core.components import SiteC
from param_decomp.core.configs import AdamPGDConfig
from param_decomp.core.init_placed import (
    init_model_component_stacks_placed,
    random_component_initializer,
)
from param_decomp.core.model import PlacedModel, prepare_compute_weights
from param_decomp.core.placement import from_config
from param_decomp.core.precision import COMPUTE_DT, cast_floating
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.sharding import hsdp_mesh, place_target, single_device_mesh
from param_decomp.targets.glu_transformer import KIND_ORDER, glu_site_specs, site_name
from param_decomp.targets.testing import tiny_glu_cfg, tiny_glu_decomposed_lm

CONFIGS = Path(__file__).parents[2] / "ci_filter" / "configs"


class FakeTokenizer:
    """64-token vocab: 0 BOS, 5 '+', 6 '-', 7 '=', 10+n the integer n (n <= 20), 40 '00'."""

    def encode(self, text: str, /, *, add_special_tokens: bool) -> list[int]:
        out = [0] if add_special_tokens else []
        number = ""
        for ch in text:
            if ch.isdigit():
                number += ch
                continue
            if number:
                out.append(10 + int(number))
                number = ""
            out.append({"+": 5, "-": 6, "=": 7}[ch])
        if number:
            out.append(10 + int(number))
        return out

    def decode(self, token_ids: list[int], /) -> str:
        (i,) = token_ids
        if 10 <= i <= 30:
            return str(i - 10)
        return {0: "<s>", 5: "+", 6: "-", 7: "=", 40: "00"}.get(i, f"<{i}>")

    def __len__(self) -> int:
        return 64


# ------------------------------------ pure math ------------------------------------


def test_kl_rows_and_restricted_objective() -> None:
    rng = np.random.default_rng(0)
    clean = jnp.asarray(rng.normal(size=(3, 64)), jnp.float32)
    masked = jnp.asarray(rng.normal(size=(3, 64)), jnp.float32)
    np.testing.assert_allclose(np.asarray(kl_rows(clean, clean)), 0.0, atol=1e-6)

    ids = jnp.asarray([10, 11, 12, 6])
    # The restricted KL ignores every logit outside the answer set...
    moved = masked.at[:, 50].add(100.0)
    objective = LastPositionIntegerKL()
    targets = jnp.asarray([0, 1, 3])
    a = objective_rows(objective, masked, clean, ids, targets)
    b = objective_rows(objective, moved, clean, ids, targets)
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5)
    # ...and is invariant to a shared shift of the answer logits (renormalization).
    shifted = masked.at[:, ids].add(3.0)
    np.testing.assert_allclose(
        np.asarray(objective_rows(objective, shifted, clean, ids, targets)),
        np.asarray(a),
        rtol=1e-4,
    )
    scores = row_scores(masked, clean, ids, targets)
    assert scores.kl.shape == scores.integer_kl.shape == (3,)

    # The answer CE is the restricted, renormalized negative log-probability of the TRUE answer
    # (it ignores the clean model entirely) and its argmax defines `correct`.
    ce = objective_rows(LastPositionAnswerCE(), masked, clean, ids, targets)
    logp = jax.nn.log_softmax(np.asarray(masked)[:, np.asarray(ids)], axis=-1)
    np.testing.assert_allclose(
        np.asarray(ce), -logp[np.arange(3), np.asarray(targets)], rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(np.asarray(scores.answer_ce), np.asarray(ce), rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(
        np.asarray(scores.correct), logp.argmax(axis=-1) == np.asarray(targets)
    )
    unchanged = objective_rows(LastPositionAnswerCE(), masked, moved, ids, targets)
    np.testing.assert_allclose(np.asarray(unchanged), np.asarray(ce), rtol=1e-6)
    # Mass moved onto NON-answer tokens does not count: the CE is renormalized over the answer
    # set, so only the relative distribution among integers can change it.
    elsewhere = objective_rows(LastPositionAnswerCE(), moved, clean, ids, targets)
    np.testing.assert_allclose(np.asarray(elsewhere), np.asarray(ce), rtol=1e-6)
    assert bool(jnp.all(row_scores(clean, clean, ids, targets).integer_top1))


def test_pool_blocks_labels_and_answer_tokens() -> None:
    tok = FakeTokenizer()
    pool = build_pool(ArithmeticPoolConfig(a_range=(1, 3), b_range=(2, 4)), tok)
    assert pool.tokens.shape == (18, 5)
    assert [(b.operation, b.start, b.stop) for b in pool.blocks] == [("add", 0, 9), ("sub", 9, 18)]
    assert pool.tokens[9 + 1].tolist() == tok.encode("1-3=", add_special_tokens=True)
    assert pool.position_labels == ("<BOS>", "a", "op", "b", "=")
    ids = answer_token_ids(tok, include_minus=True)
    assert ids.tolist() == [6, *range(10, 31), 40]
    assert 6 not in answer_token_ids(tok, include_minus=False).tolist()


def test_capped_ci_value_and_gradient() -> None:
    """`min(x, cap)`; at or under the cap the gradient passes, over it only a lowering one."""
    x = jnp.array([0.2, 0.5, 0.9, 0.9])
    cap = jnp.array([0.5, 0.5, 0.4, 0.4])
    np.testing.assert_allclose(np.asarray(_capped(x, cap)), [0.2, 0.5, 0.4, 0.4])
    weights = jnp.array([1.0, -1.0, -1.0, 1.0])
    grad_x, grad_cap = jax.grad(lambda x, c: jnp.sum(weights * _capped(x, c)), argnums=(0, 1))(
        x, cap
    )
    np.testing.assert_allclose(np.asarray(grad_x), [1.0, -1.0, 0.0, 1.0])
    np.testing.assert_allclose(np.asarray(grad_cap), 0.0)


def test_batch_indexing() -> None:
    batches = index_batches(7, 3)
    assert [real for _, real in batches] == [3, 3, 1]
    assert batches[-1][0].tolist() == [6, 0, 0]
    idx = sample_batch(20, 8, seed=0, step=3)
    assert len(set(idx.tolist())) == 8
    assert idx.tolist() == sample_batch(20, 8, seed=0, step=3).tolist()


@pytest.mark.parametrize(
    "name",
    ["last_position_kl.yaml", "last_position_integer_kl.yaml", "last_position_answer_ce.yaml"],
)
def test_template_configs_parse(name: str, tmp_path: Path) -> None:
    config = CIFilterConfig.from_file(CONFIGS / name)
    assert config.n_microbatches == 1
    config.to_file(tmp_path / name)
    assert CIFilterConfig.from_file(tmp_path / name) == config
    if name != "last_position_kl.yaml":
        assert isinstance(config.init, InitFromCIFilter)
    assert config.ci_ceiling and config.prune_dead  # the adopted default mechanism


# ------------------------------------ placed run ------------------------------------


def _config(micro: int) -> CIFilterConfig:
    raw = CIFilterConfig.from_file(CONFIGS / "last_position_kl.yaml").model_dump(mode="json")
    raw |= {
        "steps": 4,
        "batch_size": 2 * micro,
        "microbatch_size": micro,
        "eval_batch_size": micro,
        "pool": {"operations": ["add", "sub"], "a_range": [1, 3], "b_range": [1, 3]},
        "compilation_cache_dir": None,
        "wandb": None,
        "recon": {"kind": "ci_masked", "coeff": 1.5},
    }
    return CIFilterConfig.model_validate(raw)


C = 8


def _setup(mesh: Mesh, sharding: str):
    cfg = tiny_glu_cfg()
    sites = glu_site_specs(
        cfg, tuple(SiteC(site_name(layer, kind), C) for layer in (1, 2) for kind in KIND_ORDER)
    )
    model = tiny_glu_decomposed_lm(cfg, sites, jax.random.PRNGKey(0))
    rules = from_config(sharding, mesh, sites)
    placed: PlacedModel = place_target(model, rules)
    arch = ChunkwiseTransformerCIArch(
        chunks=(Chunk(input_taps=("resid.1",), output_sites=model.site_names),),
        input_dim=cfg.n_embd,
        d_model=16,
        n_blocks=1,
        attention=MHACIAttention(n_heads=2),
        ffn_hidden=32,
        ffn_kind="gelu",
        learned_norm_scale=False,
        dual=True,
    )
    fn = build_ci_fn(arch, model.sites, jax.random.PRNGKey(3))
    assert isinstance(fn, ChunkwiseTransformerCIFn)
    ci_fn = PlacedCIFn(fn=fn, placement=resolve_ci_placement(arch, rules))
    return rules, placed, sites, ci_fn


def test_nontarget_probe_isolates_one_component() -> None:
    """Subtracting no component reproduces the frozen model exactly; subtracting one moves the
    prediction somewhere."""
    mesh = single_device_mesh()
    rules, placed, sites, ci_fn = _setup(mesh, "ddp")
    tokens = jnp.asarray(np.array([[0, 11, 5, 12, 7], [0, 12, 6, 11, 7]], np.int32))
    with jax.set_mesh(mesh):
        components = init_model_component_stacks_placed(
            placed, jax.random.PRNGKey(1), rules, random_component_initializer
        )
        prepared = prepare_compute_weights(placed, components)
        ones = {s.name: jnp.ones(s.C, jnp.float32) for s in sites}
        selection = ((placed.site_names[0], 0), (placed.site_names[1], 1))
        ci = make_probe_ci(selection)(placed, ci_fn, tokens)
        assert np.asarray(ci).shape == (*tokens.shape, len(selection))
        assert np.all((np.asarray(ci) >= 0.0) & (np.asarray(ci) <= 1.0))
        baseline = make_probe_baseline()(placed, prepared, tokens, ones)
        # Subtracting NOTHING is the frozen model itself: the delta carries `W - UV`.
        np.testing.assert_allclose(np.asarray(baseline.clean_kl), 0.0, atol=1e-4)
        probe = make_probe_component()
        same = probe(placed, prepared, tokens, baseline, ones)
        np.testing.assert_allclose(np.asarray(same["kl"]), 0.0, atol=1e-4)
        np.testing.assert_array_equal(np.asarray(same["top1"]), np.asarray(baseline.top1))
        site = placed.site_names[0]
        keep = dict(ones)
        keep[site] = ones[site].at[0].set(0.0)
        ablated = probe(placed, prepared, tokens, baseline, keep)
        assert float(np.asarray(ablated["kl"]).max()) > 0.0


def test_ablation_sweep_metrics_match_a_single_ablation() -> None:
    """The sweep's hoisted state + metrics reproduce the one-at-a-time ablation's score, and an
    all-ones keep vector is the un-ablated baseline."""
    mesh = single_device_mesh()
    rules, placed, sites, ci_fn = _setup(mesh, "ddp")
    pool = build_pool(ArithmeticPoolConfig(a_range=(1, 3), b_range=(1, 3)), FakeTokenizer())
    answers = answer_token_ids(FakeTokenizer(), include_minus=True)
    targets = answer_targets(pool, FakeTokenizer(), answers)
    with jax.set_mesh(mesh):
        components = init_model_component_stacks_placed(
            placed, jax.random.PRNGKey(1), rules, random_component_initializer
        )
        prepared = prepare_compute_weights(placed, components)
        tokens_all = jax.sharding.reshard(jnp.asarray(pool.tokens), P())
        answer_ids = jax.sharding.reshard(jnp.asarray(answers), P())
        target_all = jax.sharding.reshard(jnp.asarray(targets.indices), P())
        idx = jnp.asarray(np.arange(4, dtype=np.int32))

        state = make_ablation_state()(placed, ci_fn, UNCONSTRAINED, tokens_all, idx, target_all)
        metrics = make_ablation_metrics()
        ones = {s.name: jnp.ones(s.C, jnp.float32) for s in sites}
        base = metrics(placed, prepared, state, answer_ids, ones)
        assert set(base) == {"kl", "integer_kl", "answer_ce", "accuracy"}
        assert float(base["kl"]) >= -1e-6

        site = placed.site_names[0]
        keep = dict(ones)
        keep[site] = ones[site].at[3].set(0.0)
        ablated = metrics(placed, prepared, state, answer_ids, keep)
        scores, _ = make_ablation_scores()(
            placed, prepared, ci_fn, UNCONSTRAINED, tokens_all, idx, answer_ids, target_all, keep
        )
        np.testing.assert_allclose(
            float(ablated["answer_ce"]), -float(np.mean(np.asarray(scores))), rtol=1e-4, atol=1e-5
        )


def test_train_eval_grid_and_checkpoint_run_placed(tmp_path: Path) -> None:
    _exercise_placed(tmp_path, single_device_mesh(), "ddp")


@pytest.mark.multidevice
def test_train_eval_grid_and_checkpoint_run_on_a_zero1_fsdp_mesh(tmp_path: Path) -> None:
    """The consumer layout of a multi-GPU filter: `zero1` over an all-devices fsdp axis."""
    _exercise_placed(tmp_path, hsdp_mesh(1, jax.device_count(), 1), "zero1")


def _exercise_placed(tmp_path: Path, mesh: Mesh, sharding: str) -> None:
    rules, placed, sites, ci_fn = _setup(mesh, sharding)
    pool = build_pool(ArithmeticPoolConfig(a_range=(1, 3), b_range=(1, 3)), FakeTokenizer())
    answers = answer_token_ids(FakeTokenizer(), include_minus=True)
    micro = mesh.size if mesh.size > 1 else 2
    config = _config(micro)
    with jax.set_mesh(mesh):
        # Training and every read below run under both constraints: the starting CI fn caps the
        # trained one (`ci_ceiling`) and every odd component is removed (`prune_dead`).
        start_ci_fn = ci_fn
        kept = {s.name: (jnp.arange(s.C) % 2 == 0).astype(jnp.float32) for s in sites}
        constraints = CIConstraints(ceilings=(cast_floating(ci_fn, COMPUTE_DT),), kept=kept)
        full_components = init_model_component_stacks_placed(
            placed, jax.random.PRNGKey(1), rules, random_component_initializer
        )
        components = remove_components(full_components, kept)
        for site in placed.site_names:
            V, U = (np.asarray(x) for x in (components.site(site).V, components.site(site).U))
            full_V, full_U = (
                np.asarray(x) for x in (full_components.site(site).V, full_components.site(site).U)
            )
            np.testing.assert_array_equal(V[:, 1::2], 0.0)
            np.testing.assert_array_equal(U[1::2], 0.0)
            np.testing.assert_array_equal(V[:, ::2], full_V[:, ::2])
            np.testing.assert_array_equal(U[::2], full_U[::2])
        del full_components
        prepared = prepare_compute_weights(placed, components)
        v_norms = {
            site: jax.sharding.reshard(jnp.linalg.norm(components.site(site).V, axis=0), P())
            for site in placed.site_names
        }
        tokens_all = jax.sharding.reshard(jnp.asarray(pool.tokens), P())
        answer_ids = jax.sharding.reshard(jnp.asarray(answers), P())
        targets = answer_targets(pool, FakeTokenizer(), answers)
        target_all = jax.sharding.reshard(jnp.asarray(targets.indices), P())

        optimizer = make_optimizer(config)
        opt_state = optimizer.init(trainable(ci_fn))
        micro_grads = make_micro_grads(config)
        apply_update = make_apply_update(optimizer)
        before = jax.tree.leaves(eqx.filter(ci_fn, eqx.is_inexact_array))
        before = [np.asarray(x).copy() for x in before]

        def step_grads(i: int, on_host: bool):
            idx = sample_batch(pool.n_prompts, config.batch_size, config.seed, i)
            frac = jnp.float32(i / (config.steps - 1))
            call = ci_masked_micro_call(
                micro_grads,
                placed,
                prepared,
                ci_fn,
                constraints,
                tokens_all,
                answer_ids,
                target_all,
                frac,
            )
            return batch_gradients(call, idx, micro, on_host)

        # The merged persistent-PGD step (delta on): warmup + main + final ascent, and a split
        # into microbatches leaves the adversary's trajectory unchanged (adv_fraction ~0 makes
        # the main term's source gradient zero, so only the deterministic warmups move it).
        idx_all = sample_batch(pool.n_prompts, config.batch_size, config.seed, 0)
        batch = config.batch_size
        adversaries = {}
        start = init_adversary(placed, batch, pool.seq_len, jax.random.PRNGKey(5))
        for split in (1, 2):
            ppgd_config = config.model_copy(
                update={
                    "microbatch_size": batch // split,
                    # eps 1e-4: Adam turns a float-noise gradient (removed components'
                    # neighbours sit near 0) into a full-lr step whose sign depends on the split.
                    "recon": MergedPPGDRecon(
                        adv_fraction=ScheduleConfig.constant(1e-12),
                        n_warmup_steps=2,
                        optimizer=AdamPGDConfig(
                            beta1=0.5,
                            beta2=0.99,
                            eps=1e-4,
                            lr_schedule=ScheduleConfig.constant(0.01),
                        ),
                    ),
                }
            )
            ppgd_grads, ppgd_metrics, ppgd_schedules, adversary = PPGDStep.build(ppgd_config)(
                placed,
                prepared,
                ci_fn,
                constraints,
                tokens_all,
                idx_all,
                answer_ids,
                target_all,
                jnp.float32(0.5),
                start,
                jax.random.PRNGKey(6),
                batch // split,
                False,
            )
            assert float(adversary.opt_state.step_count) == 3.0  # 2 warmups + the final ascent
            assert all(np.isfinite(float(v)) for v in ppgd_metrics.values())
            assert {"adv_fraction", "source_lr"} <= set(ppgd_schedules)
            assert all(np.isfinite(np.asarray(g)).all() for g in jax.tree.leaves(ppgd_grads))
            for new_s, old_s in zip(
                jax.tree.leaves(adversary.sources), jax.tree.leaves(start.sources), strict=True
            ):
                assert float(new_s.min()) >= 0.0 and float(new_s.max()) <= 1.0
                assert new_s.shape == old_s.shape
            adversaries[split] = adversary
        moved = any(
            not np.allclose(np.asarray(a), np.asarray(b))
            for a, b in zip(
                jax.tree.leaves(adversaries[1].sources), jax.tree.leaves(start.sources), strict=True
            )
        )
        assert moved
        for a, b in zip(
            jax.tree.leaves(adversaries[1].sources),
            jax.tree.leaves(adversaries[2].sources),
            strict=True,
        ):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-3, atol=1e-4)

        # The host-held running sum is the device sum.
        on_device, _, _ = step_grads(0, on_host=False)
        on_host, _, _ = step_grads(0, on_host=True)
        for a, b in zip(jax.tree.leaves(on_device), jax.tree.leaves(on_host), strict=True):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-7)
            assert jax.typeof(a).sharding == jax.typeof(b).sharding
        del on_device, on_host

        schedules: dict[str, jax.Array] = {}
        for i in range(config.steps):
            grads, metrics, schedules = step_grads(i, on_host=i % 2 == 1)
            ci_fn, opt_state, grad_norm = apply_update(ci_fn, opt_state, grads)
            assert all(np.isfinite(float(v)) for v in metrics.values())
            assert np.isfinite(float(grad_norm)) and float(grad_norm) > 0.0
        assert float(schedules["gamma"]) == pytest.approx(0.01)
        after = [np.asarray(x) for x in jax.tree.leaves(eqx.filter(ci_fn, eqx.is_inexact_array))]
        assert any(not np.array_equal(a, b) for a, b in zip(after, before, strict=True))

        eval_idx = jnp.asarray(index_batches(pool.n_prompts, micro)[-1][0])
        eval_batch = make_eval_batch(config)
        rows, site_max, l0 = eval_batch(
            placed, prepared, ci_fn, constraints, tokens_all, eval_idx, answer_ids, target_all
        )
        # The start under its OWN constraints is the ceiling (same jit signature: one compile).
        _, start_max, _ = eval_batch(
            placed, prepared, start_ci_fn, constraints, tokens_all, eval_idx, answer_ids, target_all
        )
        for site in site_max:
            np.testing.assert_array_less(
                np.asarray(site_max[site]), np.asarray(start_max[site]) + 1e-6
            )
            np.testing.assert_array_equal(np.asarray(site_max[site])[1::2], 0.0)
        assert set(rows) == {"ci", "rounded"}
        assert np.asarray(rows["ci"]["kl"]).shape == (micro,)
        assert set(site_max) == set(placed.site_names)
        assert np.asarray(l0["l0_per_token"]).shape == (micro,)
        pgd_config = config.model_copy(
            update={"pgd_eval": PGDEvalConfig(n_steps=2, n_batches=2, batch_size=micro)}
        )
        pgd_batches = pgd_eval_batches(pool.n_prompts, pgd_config)
        assert len(pgd_batches) == 2 and len(set(np.concatenate(pgd_batches).tolist())) == 2 * micro
        pgd = make_pgd_eval(pgd_config)(
            placed,
            prepared,
            ci_fn,
            constraints,
            tokens_all,
            jnp.asarray(pgd_batches[0]),
            answer_ids,
            target_all,
            jax.random.PRNGKey(0),
        )
        assert np.isfinite(float(pgd)) and float(pgd) >= 0.0
        alive = {s.name: (jnp.arange(s.C) % 2).astype(jnp.float32) for s in sites}
        ablations = ablation_rows(
            placed, prepared, ci_fn, constraints, alive, tokens_all, eval_idx, 0.01
        )
        assert set(ablations) == set(MASKINGS)
        for reductions in ablations.values():
            assert np.asarray(reductions["last"]).shape == (micro,)
            assert np.all(np.asarray(reductions["all_positions"]) >= -1e-6)
        on = {s.name: jnp.ones(s.C, jnp.float32) for s in sites}
        scored = make_score_fixed_masks(config)(
            placed, prepared, on, tokens_all, eval_idx, answer_ids, target_all
        )
        assert np.all(np.asarray(scored["kl"]) >= -1e-6)

        # Attribution: the first-order effect predicts a SMALL real mask change (the gradient
        # check that makes the screen meaningful), and ablating a component is its keep vector.
        assert targets.values.shape == (pool.n_prompts,)
        assert np.array_equal(answers[targets.indices], targets.token_ids)
        rows = make_attribution_batch(False)(
            placed, prepared, ci_fn, constraints, tokens_all, eval_idx, answer_ids, target_all
        )
        assert np.asarray(rows.score).shape == (micro,)
        assert np.all(np.asarray(rows.score) <= 1e-6) and np.all(
            np.isfinite(np.asarray(rows.score))
        )
        assert set(rows.effect) == set(placed.site_names)
        for site in placed.site_names:
            np.testing.assert_array_equal(np.asarray(rows.effect[site])[:, 1::2], 0.0)

        ablation = make_ablation_scores()
        ones = {s.name: jnp.ones(s.C, jnp.float32) for s in sites}
        base, base_correct = ablation(
            placed, prepared, ci_fn, constraints, tokens_all, eval_idx, answer_ids, target_all, ones
        )
        np.testing.assert_allclose(np.asarray(base), np.asarray(rows.score), rtol=1e-4, atol=1e-5)
        assert np.all(np.isin(np.asarray(base_correct), (0.0, 1.0)))
        site0 = placed.site_names[0]
        eps = 0.01
        scaled = dict(ones)
        scaled[site0] = ones[site0].at[0].set(1.0 - eps)
        nudged, _ = ablation(
            placed,
            prepared,
            ci_fn,
            constraints,
            tokens_all,
            eval_idx,
            answer_ids,
            target_all,
            scaled,
        )
        predicted = eps * np.asarray(rows.effect[site0])[:, 0]
        np.testing.assert_allclose(
            np.asarray(nudged) - np.asarray(base), predicted, rtol=0.05, atol=1e-3
        )

        # Floor 0 saves every component; each slice's mean CI is its saved columns' mean.
        block = pool.blocks[1]
        slices = collect_operation(
            placed,
            prepared,
            ci_fn,
            constraints,
            v_norms,
            tokens_all,
            block,
            pool.seq_len,
            micro,
            0.0,
        )
        assert len(slices.snapshots) == pool.seq_len
        site = placed.site_names[0]
        for snapshot in slices.snapshots:
            assert snapshot.saved[site].tolist() == list(range(C))
            assert snapshot.ci_columns["output"][site].shape == (9, 1, C)
            assert snapshot.inner_columns[site].shape == (9, 1, C)
            np.testing.assert_allclose(
                snapshot.mean_ci["output"][site][0],
                snapshot.ci_columns["output"][site][:, 0, :].mean(axis=0),
                rtol=1e-5,
                atol=1e-6,
            )
        counts = write_operation(tmp_path / "grids", slices, pool, 4, 0.0)
        write_applet(tmp_path / "grids", pool, "test")
        assert set(counts) == {f"sub_pos{p}.js" for p in range(pool.seq_len)}
        assert (tmp_path / "grids" / "index.html").exists()
        assert "sub_pos4.js" in (tmp_path / "grids" / "manifest.js").read_text()

        save_ci_fn(tmp_path / "ci_fn", ci_fn)
        restored = restore_ci_fn(tmp_path / "ci_fn", _setup(mesh, sharding)[3])
        for a, b in zip(jax.tree.leaves(restored), jax.tree.leaves(ci_fn), strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

        # Constraint resolution: a capped, pruned objective 1 from the run, then objective 2
        # from it (one more ceiling; the removal is inherited).
        run_dir = tmp_path / "run"
        run_ci_fn = _setup(mesh, sharding)[3]
        obj1 = CIFilterOutputs.for_run(run_dir, 0, "obj1")
        obj1.create()
        obj1_config = config.model_copy(update={"ci_ceiling": True, "prune_dead": True})
        obj1_config.to_file(obj1.config)
        save_ci_fn(obj1.ci_fn, ci_fn)
        save_kept(obj1.kept, {site: np.asarray(k) > 0 for site, k in kept.items()})
        bare = config.model_copy(update={"ci_ceiling": False, "prune_dead": False})
        start, unconstrained = starting_ci(bare, run_dir, 0, run_ci_fn)
        assert start is run_ci_fn and unconstrained == UNCONSTRAINED
        start, capped_by = starting_ci(obj1_config, run_dir, 0, run_ci_fn)
        assert start is run_ci_fn and len(capped_by.ceilings) == 1 and capped_by.kept is None
        obj2_config = config.model_copy(
            update={"ci_ceiling": True, "init": InitFromCIFilter(id="obj1")}
        )
        trained, inherited = trained_ci(run_dir, 0, "obj1", run_ci_fn)
        assert len(inherited.ceilings) == 1 and inherited.kept is not None
        for site, k in kept.items():
            np.testing.assert_array_equal(np.asarray(inherited.kept[site]), np.asarray(k))
        start, capped_by = starting_ci(obj2_config, run_dir, 0, run_ci_fn)
        assert len(capped_by.ceilings) == 2 and capped_by.kept is not None
        for a, b in zip(jax.tree.leaves(start), jax.tree.leaves(trained), strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(
            jax.tree.leaves(capped_by.ceilings[1]), jax.tree.leaves(ci_fn), strict=True
        ):
            assert a.dtype == COMPUTE_DT or not jnp.issubdtype(b.dtype, jnp.floating)
