"""The integer-only-adv filter's pieces on a tiny all-sites decomposed GLU with a dual CI fn (CPU):
the restricted KL it trains against `KL_int` and a hand computation, the merged persistent-PGD
term with the weight delta off, an adversary with fewer rows than the batch, the ceiling with
the prune off, the final-step coefficients, the restore of a run's adversary from a training
checkpoint, a resumed fine-tune, the alive list and the collection's stored pairs."""

import json
from dataclasses import replace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest
from jax.sharding import PartitionSpec as P

from param_decomp.ci_filter.checkpoint import starting_ci
from param_decomp.ci_filter.config import CIFilterConfig, LastPositionIntegerKL
from param_decomp.ci_filter.objective import objective_rows
from param_decomp.ci_filter.ppgd import (
    Adversary,
    PPGDStep,
    _masking,
    init_adversary,
    make_warmup_grads,
    row_ids,
    sum_rows,
)
from param_decomp.ci_filter.step import (
    UNCONSTRAINED,
    FilterCIFn,
    gather_batch,
    output_ci,
    prepare_components,
)
from param_decomp.core.ci_fn.interface import is_dual
from param_decomp.core.components import ComponentStacks, SiteC
from param_decomp.core.configs import TargetedPDConfig
from param_decomp.core.init_placed import (
    init_model_component_stacks_placed,
    random_component_initializer,
)
from param_decomp.core.masking import source_masking
from param_decomp.core.placement import from_config
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.sharding import place_target, single_device_mesh
from param_decomp.experiments.lm.load_run import PlacedTransformer
from param_decomp.integer_only import model as im
from param_decomp.integer_only.adv.adversary import restore_run_adversary, stored_tree
from param_decomp.integer_only.adv.coefficients import (
    TARGET_TERM,
    filter_config,
    resolve_final,
    value_at_end,
)
from param_decomp.integer_only.adv.config import IntegerOnlyAdvConfig
from param_decomp.integer_only.adv.layout import AdvLayout
from param_decomp.integer_only.adv.run import alive_document
from param_decomp.integer_only.adv.train import Problem, fresh_state, train
from param_decomp.integer_only.ci_pass import chunk_ci_columns, chunk_max_ci
from param_decomp.integer_only.collect import collect_dataset, unpack_column
from param_decomp.integer_only.objective import int_log_q, kl_int, log_softmax32
from param_decomp.integer_only.pick import read_mask_json
from param_decomp.integer_only.pool import PromptPool
from param_decomp.targets.testing import (
    tiny_glu_cfg,
    tiny_glu_chunkwise_ci_fn_arch,
    tiny_glu_decomposed_lm,
)
from param_decomp.targets.transformer import KIND_ORDER, glu_site_specs, site_name
from param_decomp.tests.placed_ci_fn import placed_ci_fn

C = 8
VOCAB = 64
T = 5
N_PROMPTS = 24
N_INT = 20
"""The toy's integer tokens: token `10 + x` is the integer `x`, for `x` in 0..19."""

type Toy = tuple[PlacedTransformer, ComponentStacks, FilterCIFn, PromptPool]


@pytest.fixture(scope="module")
def toy() -> Toy:
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
    arch = replace(tiny_glu_chunkwise_ci_fn_arch(model, n_blocks=1), dual=True)
    with jax.set_mesh(mesh):
        components = init_model_component_stacks_placed(
            placed, jax.random.PRNGKey(1), rules, random_component_initializer
        )
        ci_fn = placed_ci_fn(arch, sites, jax.random.PRNGKey(3), mesh, rules)
    assert is_dual(ci_fn)
    rng = np.random.default_rng(0)
    a = rng.integers(0, 10, N_PROMPTS)
    b = rng.integers(0, 10, N_PROMPTS)
    tokens = rng.integers(1, VOCAB, size=(N_PROMPTS, T)).astype(np.int32)
    tokens[:, 1], tokens[:, 3] = 10 + a, 10 + b
    pool = PromptPool(
        tokens=tokens, ops=("add",), op=np.zeros(N_PROMPTS, np.int8), a=a, b=b, r=a + b,
        position_labels=("x", "a", "x", "b", "x"), sha256="toy",
    )  # fmt: skip
    return placed, components, ci_fn, pool


def _problem(pool: PromptPool, train_rows: np.ndarray) -> Problem:
    with jax.set_mesh(single_device_mesh()):
        return Problem(
            tokens_all=jax.sharding.reshard(jnp.asarray(pool.tokens), P()),
            answer_ids=jax.sharding.reshard(jnp.arange(10, 10 + N_INT, dtype=jnp.int32), P()),
            targets=jax.sharding.reshard(jnp.asarray(pool.r.astype(np.int32)), P()),
            train=train_rows,
        )


def _pd(steps: int, batch: int) -> TargetedPDConfig:
    """A run's `pd` section: schedules whose final values differ from their peaks."""
    return TargetedPDConfig.model_validate(
        {
            "steps": steps,
            "batch_size": batch,
            "components_optimizer": {
                "type": "adamw",
                "lr_schedule": {"max_val": 1e-3, "points": [{"at": 0, "frac": 1}, {"at": 1, "frac": 1}]},
            },
            "ci_fn_optimizer": {
                "type": "adamw",
                "betas": [0.8, 0.95],
                "lr_schedule": {"max_val": 1e-3, "points": [{"at": 0, "frac": 1}, {"at": 1, "frac": 0.1}]},
            },
            "loss_metrics": [
                {
                    "type": "ImportanceMinimalityLoss",
                    "coeff": {"max_val": 2e-4, "points": [{"at": 0, "frac": 1}, {"at": 0.25, "frac": 0.25}, {"at": 1, "frac": 0.25}]},
                    "normalize_at_one": True,
                    "gamma": {"max_val": 1.0, "points": [{"at": 0, "frac": 1}, {"at": 0.75, "frac": 1}, {"at": 1, "frac": 0.01}]},
                    "frequency": {"coeff": 1e-4, "reference_datapoint_count": "auto"},
                },
                {
                    "type": "MergedStochasticSubsetPPGDReconLoss",
                    "coeff": 1.5,
                    "adv_fraction": {"max_val": 0.9, "points": [{"at": 0, "frac": 0}, {"at": 0.25, "frac": 1}, {"at": 1, "frac": 1}]},
                    "n_warmup_steps": 2,
                    "routing": {"type": "uniform_k_subset"},
                    "optimizer": {"type": "adam", "beta1": 0.5, "beta2": 0.99, "eps": 1e-8, "lr_schedule": {"max_val": 0.01, "points": [{"at": 0, "frac": 0}, {"at": 0.0125, "frac": 1}, {"at": 1, "frac": 0.5}]}},
                    "source_shape": "bsc",
                },
                {"type": "UnmaskedReconLoss", "coeff": 0.5},
            ],
        }
    )  # fmt: skip


def _filter_config(steps: int, batch: int, micro: int, tmp: Path) -> CIFilterConfig:
    coefficients = resolve_final(_pd(steps=11, batch=2), T)
    config = IntegerOnlyAdvConfig(step=0, steps=steps, batch=batch, microbatch=micro, save_every=2)
    return filter_config(coefficients, config, tmp)


# ------------------------------------ objective ------------------------------------


def test_restricted_kl_is_kl_int_by_hand() -> None:
    """The objective the filter trains (the restricted KL over the 1000 integer tokens) is
    integer_only's `KL_int`, and both are the hand-computed KL of the renormalized `q`."""
    rng = np.random.default_rng(1)
    clean = rng.normal(size=(4, VOCAB)).astype(np.float32)
    masked = rng.normal(size=(4, VOCAB)).astype(np.float32)
    ids = np.arange(10, 10 + N_INT)
    trained = objective_rows(
        LastPositionIntegerKL(include_minus=False), jnp.asarray(masked), jnp.asarray(clean),
        jnp.asarray(ids), jnp.zeros(4, jnp.int32),
    )  # fmt: skip
    io = kl_int(
        int_log_q(log_softmax32(jnp.asarray(clean)), jnp.asarray(ids)),
        int_log_q(log_softmax32(jnp.asarray(masked)), jnp.asarray(ids)),
    )

    def q(logits: np.ndarray) -> np.ndarray:
        p = np.exp(logits - logits.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        return p[:, ids] / p[:, ids].sum(axis=1, keepdims=True)

    q_ref, q_m = q(clean.astype(np.float64)), q(masked.astype(np.float64))
    hand = (q_ref * (np.log(q_ref) - np.log(q_m))).sum(axis=1)
    np.testing.assert_allclose(np.asarray(trained), hand, rtol=1e-4)
    np.testing.assert_allclose(np.asarray(io), hand, rtol=1e-4)


# ------------------------------------ coefficients ------------------------------------


def test_final_step_coefficients(tmp_path: Path) -> None:
    """Every schedule read at the run's last step `S - 1` (run fraction 1), `auto` `a'` as the
    run's `B·T`, the CI-fn-independent term dropped, and the filter config built from them."""
    steps, batch = 11, 4
    pd = _pd(steps, batch)
    c = resolve_final(pd, T)
    assert c.impmin_coeff == pytest.approx(5e-5)
    assert c.gamma == pytest.approx(0.01)
    assert c.frequency_coeff == pytest.approx(1e-4)
    assert c.reference_datapoint_count == batch * T
    assert c.adv_fraction == pytest.approx(0.9)
    assert c.source_lr == pytest.approx(0.005)
    assert c.recon_coeff == 1.5 and c.n_warmup_steps == 2 and c.run_batch == batch
    assert (c.source_beta1, c.source_beta2, c.source_eps) == (0.5, 0.99, 1e-8)
    assert c.ci_betas == (0.8, 0.95) and c.ci_grad_clip_norm is None
    assert "UnmaskedReconLoss" in c.dropped
    early = value_at_end(
        ScheduleConfig.model_validate(
            {"max_val": 1.0, "points": [{"at": 0, "frac": 1}, {"at": 1, "frac": 0.01}]}
        ),
        steps,
    )
    assert early == pytest.approx(0.01)
    assert value_at_end(0.25, steps) == 0.25

    fc = filter_config(c, IntegerOnlyAdvConfig(step=7, steps=5, batch=8, microbatch=4), tmp_path)
    assert fc.prune is False and fc.recon.kind == "merged_stochastic_ppgd"
    assert fc.recon.weight_delta == "off"
    assert fc.imp_min.reference_datapoint_count == batch * T
    for schedule in (fc.lr_schedule, fc.imp_min.coeff, fc.imp_min.gamma, fc.recon.adv_fraction):
        assert schedule.is_constant
    assert fc.lr_schedule.max_val == 8e-5 and fc.betas == (0.8, 0.95)
    fc.to_file(tmp_path / "fc.yaml")
    assert CIFilterConfig.from_file(tmp_path / "fc.yaml") == fc
    assert CIFilterConfig.model_fields["prune"].default is True


# ------------------------------------ the PPGD term ------------------------------------


def test_weight_delta_off_in_the_merged_term(toy: Toy, tmp_path: Path) -> None:
    """With the delta off the materialized masks carry no delta, the warmup gradient on the
    delta sources is exactly zero (on: nonzero), and the full step runs."""
    placed, components, ci_fn, pool = toy
    fc = _filter_config(steps=2, batch=4, micro=2, tmp=tmp_path)
    on = fc.model_copy(update={"recon": fc.recon.model_copy(update={"weight_delta": "on"})})
    mesh = single_device_mesh()
    problem = _problem(pool, np.arange(N_PROMPTS))
    with jax.set_mesh(mesh):
        prepared, _ = prepare_components(placed, components)
        adversary = init_adversary(placed, 4, T, jax.random.PRNGKey(5))
        idx = jnp.arange(4, dtype=jnp.int32)
        clean = placed.clean_forward(gather_batch(problem.tokens_all, idx), ci_fn.capture_keys)
        lower = output_ci(placed, prepared, ci_fn, UNCONSTRAINED, clean, remat=False).lower
        masking = source_masking(lower, adversary.sources.per_site())
        assert _masking(masking, "off").weight_delta_masks is None
        assert _masking(masking, "on").weight_delta_masks is not None
        grads = {}
        for name, config in (("off", fc), ("on", on)):
            grads[name] = make_warmup_grads(config)(
                placed, prepared, ci_fn, UNCONSTRAINED, problem.tokens_all, idx,
                problem.answer_ids, problem.targets, adversary.sources,
            )  # fmt: skip
        for stack in grads["off"].stacks.values():
            np.testing.assert_array_equal(np.asarray(stack.delta), 0.0)
        assert any(float(jnp.abs(s.delta).max()) > 0 for s in grads["on"].stacks.values())
        assert any(float(jnp.abs(s.components).max()) > 0 for s in grads["off"].stacks.values())
        _, metrics, _, moved = PPGDStep.build(fc)(
            placed, prepared, ci_fn, UNCONSTRAINED, problem.tokens_all, np.arange(4, dtype=np.int32),
            problem.answer_ids, problem.targets, jnp.float32(1.0), adversary,
            jax.random.PRNGKey(6), 2, False,
        )  # fmt: skip
    assert float(moved.opt_state.step_count) == 3.0
    assert all(np.isfinite(float(v)) for v in metrics.values())
    for new, old in zip(
        (s.delta for s in moved.sources.stacks.values()),
        (s.delta for s in adversary.sources.stacks.values()),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(new), np.asarray(old))  # no delta gradient


def test_adversary_rows_serve_every_rth_batch_element(toy: Toy, tmp_path: Path) -> None:
    """Element `j` reads row `j mod R`; the per-row gradient sums its elements', so the
    adversary's trajectory does not depend on the microbatch split."""
    assert row_ids(6, 4, 4).tolist() == [2, 3, 0, 1]
    like = init_adversary(toy[0], 2, T, jax.random.PRNGKey(0)).sources
    parts = [jax.tree.map(lambda x, k=k: jnp.ones_like(x[:, :1]) * (k + 1), like) for k in range(3)]
    total = sum_rows(parts, [np.array([0]), np.array([1]), np.array([0])], like)
    leaf = np.asarray(jax.tree.leaves(total)[0])
    np.testing.assert_allclose(leaf[:, 0], 4.0)
    np.testing.assert_allclose(leaf[:, 1], 2.0)

    placed, components, ci_fn, pool = toy
    problem = _problem(pool, np.arange(N_PROMPTS))
    idx = np.arange(8, dtype=np.int32)
    results = {}
    with jax.set_mesh(single_device_mesh()):
        prepared, _ = prepare_components(placed, components)
        start = init_adversary(placed, 3, T, jax.random.PRNGKey(5))
        for micro in (8, 4, 2):
            fc = _filter_config(steps=2, batch=8, micro=micro, tmp=tmp_path)
            # adv_fraction ~0: only the deterministic warmups move the sources.
            fc = fc.model_copy(
                update={
                    "recon": fc.recon.model_copy(
                        update={"adv_fraction": ScheduleConfig.constant(1e-12)}
                    )
                }
            )
            _, _, _, results[micro] = PPGDStep.build(fc)(
                placed, prepared, ci_fn, UNCONSTRAINED, problem.tokens_all, idx,
                problem.answer_ids, problem.targets, jnp.float32(1.0), start,
                jax.random.PRNGKey(6), micro, False,
            )  # fmt: skip
    for micro in (4, 2):
        for a, b in zip(
            jax.tree.leaves(results[8].sources),
            jax.tree.leaves(results[micro].sources),
            strict=True,
        ):
            assert a.shape == b.shape and a.shape[1] == 3
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-3, atol=1e-4)
    moved = any(
        not np.allclose(np.asarray(a), np.asarray(b))
        for a, b in zip(
            jax.tree.leaves(results[8].sources), jax.tree.leaves(start.sources), strict=True
        )
    )
    assert moved


# ------------------------------------ constraints ------------------------------------


def test_ceiling_on_prune_off(toy: Toy, tmp_path: Path) -> None:
    """The filter starts from the run's CI fn under ONE ceiling (the run's own CI fn) and no
    removal; a moved CI fn's CI stays at or under the run's entrywise."""
    placed, components, ci_fn, pool = toy
    fc = _filter_config(steps=2, batch=4, micro=2, tmp=tmp_path)
    with jax.set_mesh(single_device_mesh()):
        start, constraints = starting_ci(fc, tmp_path, 0, ci_fn)
        assert start is ci_fn and constraints.kept is None and len(constraints.ceilings) == 1
        prepared, _ = prepare_components(placed, components)
        moved = jax.tree.map(lambda x: x * 1.7 + 0.05 if eqx.is_inexact_array(x) else x, ci_fn)
        tokens_all = jax.sharding.reshard(jnp.asarray(pool.tokens), P())
        idx = jnp.arange(8, dtype=jnp.int32)
        sites = {s: jnp.arange(C, dtype=jnp.int32) for s in placed.model.site_names}
        capped = np.asarray(
            chunk_ci_columns(placed, prepared, moved, constraints, tokens_all, idx, sites)
        )
        free = np.asarray(
            chunk_ci_columns(placed, prepared, moved, UNCONSTRAINED, tokens_all, idx, sites)
        )
        run = np.asarray(
            chunk_ci_columns(placed, prepared, ci_fn, UNCONSTRAINED, tokens_all, idx, sites)
        )
    assert np.all(capped <= run + 1e-2)
    np.testing.assert_allclose(capped, np.minimum(free, run), atol=1e-2)
    assert not np.allclose(free, run, atol=1e-2)
    assert capped.min() >= 0.0 and (capped > 0).any()


# ------------------------------------ adversary restore ------------------------------------


def test_run_adversary_restores_from_a_training_checkpoint(toy: Toy, tmp_path: Path) -> None:
    """The target-stream adversary is read out of a checkpoint's `training` item (other items
    and adversaries left unread) into the filter's structure, bit for bit; a shape mismatch
    fails before reading."""
    placed = toy[0]
    mesh = single_device_mesh()
    with jax.set_mesh(mesh):
        saved = init_adversary(placed, 3, T, jax.random.PRNGKey(11))
        saved = Adversary(
            sources=saved.sources,
            opt_state=replace(
                saved.opt_state,
                m=jax.tree.map(lambda x: x + 0.25, saved.opt_state.m),
                step_count=jnp.asarray(17.0),
            ),
        )
        other = init_adversary(placed, 3, T, jax.random.PRNGKey(12))
        like = init_adversary(placed, 3, T, jax.random.PRNGKey(13))
    run_dir = tmp_path / "run"
    training = {
        "adversaries": {TARGET_TERM: stored_tree(saved), "Hidden": stored_tree(other)},
        "ci_fn_opt_state": {"count": jnp.asarray(4)},
    }
    with ocp.CheckpointManager((run_dir / "ckpts").resolve()) as mgr:
        mgr.save(
            40,
            args=ocp.args.Composite(
                decomposition=ocp.args.StandardSave({"x": jnp.zeros(2)}),
                training=ocp.args.StandardSave(training),
            ),
        )
        mgr.wait_until_finished()
    got = restore_run_adversary(run_dir, 40, TARGET_TERM, like, mesh)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(saved), strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert float(got.opt_state.step_count) == 17.0
    with jax.set_mesh(mesh):
        wrong = init_adversary(placed, 2, T, jax.random.PRNGKey(13))
    with pytest.raises(AssertionError, match="stored"):
        restore_run_adversary(run_dir, 40, TARGET_TERM, wrong, mesh)


# ------------------------------------ the fine-tune ------------------------------------


def test_fine_tune_resumes_on_the_same_trajectory(
    toy: Toy, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Four steps straight equal a run interrupted after its step-2 save and resumed."""
    import param_decomp.integer_only.adv.train as train_module

    placed, components, ci_fn, pool = toy
    problem = _problem(pool, np.arange(N_PROMPTS))
    config = IntegerOnlyAdvConfig(step=0, steps=4, batch=4, microbatch=2, save_every=2)
    fc = _filter_config(steps=4, batch=4, micro=2, tmp=tmp_path)
    real = train_module.batch_rows

    def interrupted(problem: Problem, config: IntegerOnlyAdvConfig, step: int) -> np.ndarray:
        if step == 2:
            raise KeyboardInterrupt
        return real(problem, config, step)

    finals = []
    with jax.set_mesh(single_device_mesh()):
        prepared, _ = prepare_components(placed, components)
        _, constraints = starting_ci(fc, tmp_path, 0, ci_fn)
        for name in ("straight", "resumed"):
            layout = AdvLayout(tmp_path / name)
            layout.training.mkdir(parents=True)
            adversary = init_adversary(placed, 2, T, jax.random.PRNGKey(5))
            start = fresh_state(ci_fn, fc, adversary)
            if name == "resumed":
                monkeypatch.setattr(train_module, "batch_rows", interrupted)
                with pytest.raises(KeyboardInterrupt):
                    train(config, fc, layout, placed, prepared, constraints, problem, start)
                monkeypatch.setattr(train_module, "batch_rows", real)
                assert json.loads(layout.latest_state.read_text()) == {"step": 2}
                assert [d.name for d in layout.training.glob("state_*")] == ["state_2"]
            trained, facts = train(
                config, fc, layout, placed, prepared, constraints, problem, start
            )
            assert facts["resumed_at"] == (2 if name == "resumed" else 0)
            finals.append(
                [np.asarray(x) for x in jax.tree.leaves(eqx.filter(trained, eqx.is_array))]
            )
    for a, b in zip(*finals, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
    start_leaves = jax.tree.leaves(eqx.filter(ci_fn, eqx.is_array))
    assert any(
        not np.allclose(a, np.asarray(b)) for a, b in zip(finals[0], start_leaves, strict=True)
    )


# ------------------------------------ outputs ------------------------------------


def test_alive_list_schema(tmp_path: Path) -> None:
    sizes = {"layers.0.mlp.up_proj": 4, "layers.0.mlp.down_proj": 3}
    max_ci = {
        "layers.0.mlp.up_proj": np.array([[0.0, 0.2, 0.05, 0.1], [0.11, 0.0, 0.0, 0.0]]),
        "layers.0.mlp.down_proj": np.zeros((2, 3)),
    }
    doc = alive_document(max_ci, 0.1, sizes, rule="r", source="s")
    assert set(doc) == {"rule", "source", "threshold", "n_alive", "n_components", "components"}
    assert doc["n_alive"] == 2 and doc["n_components"] == 7 and doc["threshold"] == 0.1
    assert doc["components"] == {"layers.0.mlp.down_proj": [], "layers.0.mlp.up_proj": [0, 1]}
    path = tmp_path / "integer-only-adv.json"
    path.write_text(json.dumps(doc))
    assert read_mask_json(path) == {"layers.0.mlp.down_proj": [], "layers.0.mlp.up_proj": [0, 1]}


def test_collection_stores_only_ever_on_pairs(toy: Toy, tmp_path: Path) -> None:
    """Under the ceiling, the stored pairs are exactly the (position, component) pairs ON on
    some prompt, and the stored bits are the CI pass's."""
    placed, components, ci_fn, pool = toy
    threshold = 0.5
    mesh = single_device_mesh()
    fc = _filter_config(steps=2, batch=4, micro=2, tmp=tmp_path)
    with jax.set_mesh(mesh):
        _, constraints = starting_ci(fc, tmp_path, 0, ci_fn)
        moved = jax.tree.map(lambda x: x * 0.9 if eqx.is_inexact_array(x) else x, ci_fn)
        prepared, v_norms = prepare_components(placed, components)
        tokens_all = jax.sharding.reshard(jnp.asarray(pool.tokens), P())
        idx = jnp.arange(N_PROMPTS, dtype=jnp.int32)
        max_ci = {
            k: np.asarray(v)
            for k, v in chunk_max_ci(
                placed, prepared, moved, constraints, tokens_all, idx, jnp.asarray(N_PROMPTS)
            ).items()
        }
        sites = sorted(max_ci)
        keep = {s: np.asarray((max_ci[s] > threshold).any(axis=0)) for s in sites}
        assert sum(int(k.sum()) for k in keep.values()) > 0
        arch, shared, _ = im.build_target(placed.model)  # pyright: ignore[reportArgumentType]
        factors, slots = im.build_factors(components, keep, arch.n_layer)
        cols = {s: jnp.asarray(np.flatnonzero(keep[s]).astype(np.int32)) for s in sites}
        ci = np.asarray(
            chunk_ci_columns(placed, prepared, moved, constraints, tokens_all, idx, cols)
        )
        facts = collect_dataset(
            placed, prepared, v_norms, moved, constraints, mesh, arch, shared, factors, slots,
            pool, max_ci, threshold, 8, "fp32", tmp_path / "datasets",
            lambda *args: f"readme {args}",
        )  # fmt: skip
    index = np.load(tmp_path / "datasets" / "index.npz")
    on = ci > threshold  # (N, T, n) in slot order
    want = {(t, j) for t in range(T) for j in range(slots.n) if on[:, t, j].any()}
    got = set()
    for t, site, comp in zip(
        index["col_position"], index["col_site"], index["col_component"], strict=True
    ):
        j = next(k for k in range(slots.n) if slots.sites[k] == site and slots.index[k] == comp)
        got.add((int(t), j))
    assert got == want and facts["n_columns"] == len(want)
    for t in range(T):
        sel = np.flatnonzero(index["col_position"] == t)
        packed = np.load(tmp_path / "datasets" / f"ci_on_pos{t}.npy")
        for k, col in enumerate(sel):
            j = next(
                m
                for m in range(slots.n)
                if slots.sites[m] == index["col_site"][col]
                and slots.index[m] == index["col_component"][col]
            )
            np.testing.assert_array_equal(unpack_column(packed[:, k], N_PROMPTS), on[:, t, j])
    assert (tmp_path / "datasets" / "README.md").read_text().startswith("readme")
