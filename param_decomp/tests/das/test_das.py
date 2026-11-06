"""The DAS forward, patch, pruning search and pair sampling on a tiny random Llama."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.das.build_prompt_pairs import TRAIN, sample_pairs, split_prompts
from param_decomp.das.patched_llama_forward import Llama, logits_from, patch_stream, stream_at
from param_decomp.das.prune_das_subspaces import (
    lowest_passing_dimension,
    next_target,
    remove_least_harmful,
    rotate_by_energy,
    round_passes,
    search_position,
)
from param_decomp.das.run_das_pilot import require_matching_settings
from param_decomp.das.train_das_subspaces import (
    EVALUATION_CHUNK,
    PairStreams,
    PrincipalComponents,
    SiteFunctions,
    advanced_base,
    columns_of,
    complete_to_orthogonal,
    dimension_mask,
    evaluate,
    householder_columns,
    principal_components,
    random_pca_basis,
    train_subspaces,
)
from param_decomp.pruning.model import Arch, DenseWeights, Shared, dense_logits

D = 32
VOCAB = 50
N_LAYER = 3
T = 5


def tiny_model() -> Llama:
    keys = iter(jax.random.split(jax.random.key(0), 40))

    def normal(*shape: int) -> jax.Array:
        return 0.2 * jax.random.normal(next(keys), shape, jnp.float32)

    arch = Arch(n_layer=N_LAYER, n_head=4, n_kv_head=2, head_dim=8, eps=1e-5)
    shared = Shared(
        embed=normal(VOCAB, D),
        ln1=jnp.ones((N_LAYER, D)),
        ln2=jnp.ones((N_LAYER, D)),
        norm=jnp.ones((D,)),
        head=normal(VOCAB, D),
        inv_freq=jnp.linspace(1.0, 0.01, 4),
    )
    blocks = tuple(
        {
            "q": normal(32, D),
            "k": normal(16, D),
            "v": normal(16, D),
            "o": normal(D, 32),
            "gate": normal(64, D),
            "up": normal(64, D),
            "down": normal(D, 64),
        }
        for _ in range(N_LAYER)
    )
    return Llama(arch, shared, blocks)


TOKENS = jnp.asarray(np.random.default_rng(0).integers(0, VOCAB, (6, T)))
SOURCE_TOKENS = (
    jnp.asarray(np.random.default_rng(1).integers(0, VOCAB, (6, T))).at[:, 0].set(TOKENS[:, 0])
)


def random_orthonormal(seed: int) -> jax.Array:
    return jnp.linalg.qr(jax.random.normal(jax.random.key(seed), (4, D, 8)))[0]


@pytest.mark.parametrize("t", range(2 * N_LAYER))
def test_running_from_any_stream_point_reproduces_the_full_forward(t: int) -> None:
    model = tiny_model()
    dense = DenseWeights(
        W={kind: jnp.stack([b[kind] for b in model.blocks]) for kind in model.blocks[0]}
    )
    full = dense_logits(model.arch, model.shared, dense, TOKENS, "bf16", "bf16")
    resumed = logits_from(model, stream_at(model, TOKENS, t), t)
    np.testing.assert_allclose(resumed, full, atol=1e-5)


@pytest.mark.parametrize("t", (0, 3))
def test_identity_patch_gives_the_source_and_zero_mask_gives_the_base(t: int) -> None:
    model = tiny_model()
    base, source = stream_at(model, TOKENS, t), stream_at(model, SOURCE_TOKENS, t)
    identity = jnp.broadcast_to(jnp.eye(D), (4, D, D))
    patched = patch_stream(base, source, identity, jnp.ones((4, D)))
    np.testing.assert_allclose(
        logits_from(model, patched, t), logits_from(model, source, t), atol=1e-5
    )
    unpatched = patch_stream(base, source, identity, jnp.zeros((4, D)))
    np.testing.assert_allclose(unpatched, base)


def test_rotation_by_energy_leaves_the_patch_unchanged() -> None:
    model = tiny_model()
    t = 3
    base, source = stream_at(model, TOKENS, t), stream_at(model, SOURCE_TOKENS, t)
    Q = random_orthonormal(1)
    dims = (5, 8, 3, 0)
    mask = dimension_mask(dims, 8)
    pairs = PairStreams(base, source, jnp.zeros(6, jnp.int32), jnp.zeros(6, jnp.int32))
    rotated = rotate_by_energy(Q, 1, dims[1], pairs)
    np.testing.assert_allclose(
        patch_stream(base, source, rotated, mask), patch_stream(base, source, Q, mask), atol=1e-4
    )


def test_householder_columns_start_at_the_base_and_later_reflectors_leave_earlier_columns() -> None:
    rng = np.random.default_rng(0)
    basis = np.asarray(random_orthonormal(3))
    full_base = complete_to_orthogonal(basis, (8, 8, 8, 8), rng)
    assert full_base.shape == (4, D, D)
    np.testing.assert_allclose(columns_of(full_base, 8), basis, atol=1e-5)
    base = full_base[0]
    np.testing.assert_allclose(base.T @ base, jnp.eye(D), atol=1e-5)
    np.testing.assert_allclose(householder_columns(base, jnp.zeros((D, 8))), basis[0], atol=1e-5)
    reflectors = 1e-2 * jax.random.normal(jax.random.key(4), (D, 8))
    columns = householder_columns(base, reflectors)
    np.testing.assert_allclose(columns.T @ columns, jnp.eye(8), atol=1e-5)
    changed_later = reflectors.at[:, 3:].set(0.5)
    np.testing.assert_allclose(
        householder_columns(base, changed_later)[:, :3], columns[:, :3], atol=1e-5
    )


def test_a_reflector_step_moves_the_columns_twice_as_far() -> None:
    base = complete_to_orthogonal(
        np.asarray(random_orthonormal(5)), (8, 8, 8, 8), np.random.default_rng(1)
    )[0]
    reflectors = jnp.tril(1e-4 * jax.random.normal(jax.random.key(6), (D, 8)), -1)
    moved = float(jnp.linalg.norm(householder_columns(base, reflectors) - base[:, :8]))
    assert moved == pytest.approx(2 * float(jnp.linalg.norm(reflectors)), rel=0.1)


def test_advanced_base_continues_from_the_trained_columns_and_stays_orthogonal() -> None:
    base = complete_to_orthogonal(
        np.asarray(random_orthonormal(7)), (8, 8, 8, 8), np.random.default_rng(2)
    )[0]
    reflectors = 0.05 * jax.random.normal(jax.random.key(8), (D, 8))
    advanced = advanced_base(base, reflectors)
    np.testing.assert_allclose(advanced[:, :8], householder_columns(base, reflectors), atol=1e-4)
    np.testing.assert_allclose(advanced.T @ advanced, jnp.eye(D), atol=1e-4)


def test_random_pca_basis_stays_within_the_components_and_pads_when_too_few() -> None:
    rng = np.random.default_rng(3)
    top = np.linalg.qr(rng.standard_normal((D, 5), np.float32))[0]
    empty = np.zeros((D, 0), np.float32)
    components = PrincipalComponents((top, top, empty, top), np.ones(4, bool))
    basis = random_pca_basis(components, (3, 7, 2, 0), 8, rng)
    assert random_pca_basis(
        PrincipalComponents((empty,) * 4, np.zeros(4, bool)), (0, 0, 0, 0), 8, rng
    ).shape == (4, D, 8)
    for position in range(4):
        np.testing.assert_allclose(basis[position].T @ basis[position], np.eye(8), atol=1e-4)
    inside = top.T @ basis[0, :, :3]
    np.testing.assert_allclose(inside.T @ inside, np.eye(3), atol=1e-4)
    np.testing.assert_allclose(basis[1, :, :5], top, atol=1e-5)
    np.testing.assert_allclose(top.T @ basis[1, :, 5:7], np.zeros((5, 2)), atol=1e-5)


def test_principal_components_explain_the_base_stream_and_flag_unchanged_positions() -> None:
    rng = np.random.default_rng(4)
    direction = np.linalg.qr(rng.standard_normal((D, 2), np.float32))[0]
    base = np.zeros((64, T, D), np.float32)
    base[:, 1] = rng.standard_normal((64, 2), np.float32) @ direction.T
    source = base.copy()
    source[:, 1] += 1.0
    pairs = PairStreams(
        jnp.asarray(base), jnp.asarray(source), jnp.zeros(64, jnp.int32), jnp.zeros(64, jnp.int32)
    )
    components = principal_components(pairs, 64)
    assert components.differs.tolist() == [True, False, False, False]
    assert 1 <= components.top[0].shape[1] <= 2
    assert components.top[1].shape[1] == 0
    spanned = direction @ (direction.T @ components.top[0])
    np.testing.assert_allclose(spanned, components.top[0], atol=1e-4)


def test_training_keeps_the_subspace_orthonormal_changes_only_active_columns_and_lowers_the_loss() -> (
    None
):
    model = tiny_model()
    t = 3
    integers = jnp.arange(VOCAB)
    functions = SiteFunctions(model, t, integers, 1e-2)
    rng = np.random.default_rng(5)
    batch_tokens = jnp.asarray(rng.integers(0, VOCAB, (256, T)))
    source_tokens = jnp.asarray(rng.integers(0, VOCAB, (256, T))).at[:, 0].set(batch_tokens[:, 0])
    source = stream_at(model, source_tokens, t)
    answers = jnp.argmax(logits_from(model, source, t), -1)
    pairs = PairStreams(stream_at(model, batch_tokens, t), source, answers, answers)
    dims = (0, 0, 0, 4)
    mask = dimension_mask(dims, 8)
    start = np.asarray(random_orthonormal(9))
    full_base = complete_to_orthogonal(start, dims, rng)
    before = evaluate(functions, columns_of(full_base, 8), mask, pairs)
    trained = train_subspaces(functions, full_base, mask, pairs, 16, 60, 0)
    after = evaluate(functions, columns_of(trained, 8), mask, pairs)
    assert after.cross_entropy < before.cross_entropy
    np.testing.assert_allclose(trained[3].T @ trained[3], jnp.eye(D), atol=1e-3)
    np.testing.assert_allclose(trained[3, :, :4].T @ trained[3, :, :4], jnp.eye(4), atol=1e-3)
    np.testing.assert_allclose(trained[0], jnp.eye(D), atol=1e-5)


def test_removal_keeps_the_least_harmful_columns_first_and_q_orthonormal() -> None:
    Q = random_orthonormal(2)
    scores = jnp.asarray([5.0, 1.0, 3.0, 2.0])  # removing column 1 then 3 hurts least
    pruned = remove_least_harmful(Q, 2, 4, 2, scores)
    np.testing.assert_allclose(pruned[2, :, 0], Q[2, :, 0])
    np.testing.assert_allclose(pruned[2, :, 1], Q[2, :, 2])
    np.testing.assert_allclose(pruned[2].T @ pruned[2], jnp.eye(8), atol=1e-5)


def test_evaluation_counts_only_the_real_pairs_of_a_short_last_chunk() -> None:
    model = tiny_model()
    t = 3
    functions = SiteFunctions(model, t, jnp.arange(VOCAB), 1e-3)
    rng = np.random.default_rng(6)
    n = EVALUATION_CHUNK + 5
    tokens = jnp.asarray(rng.integers(0, VOCAB, (n, T)))
    stream = stream_at(model, tokens, t)
    top = jnp.argmax(logits_from(model, stream, t), -1)
    # The source is the base, so the patch changes nothing, and only the last pair's answer is right.
    wrong = (top + 1) % VOCAB
    answers = wrong.at[-1].set(top[-1])
    pairs = PairStreams(stream, stream, answers, answers)
    metrics = evaluate(functions, random_orthonormal(10), dimension_mask((8, 8, 8, 8), 8), pairs)
    assert metrics.iia == pytest.approx(1 / n)
    assert metrics.full_vocabulary_iia == pytest.approx(1 / n)


def test_results_from_other_settings_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    stale = tmp_path / "start_64_dims.npz"
    stale.touch()
    with pytest.raises(AssertionError, match="unknown settings"):
        require_matching_settings(path, {"learning_rate": 1e-4}, [stale])
    require_matching_settings(path, {"learning_rate": 1e-4}, [])
    require_matching_settings(path, {"learning_rate": 1e-4}, [stale])
    with pytest.raises(AssertionError, match="other settings"):
        require_matching_settings(path, {"learning_rate": 2e-4}, [stale])


def test_a_round_that_improves_on_the_reference_passes() -> None:
    assert round_passes(0.65, 0.60, 0.02)
    assert round_passes(0.585, 0.60, 0.02)
    assert not round_passes(0.57, 0.60, 0.02)


def test_schedule_halves_then_steps_and_bisects_after_a_failure() -> None:
    assert next_target("halve_then_single_steps", 64, None) == 32
    assert next_target("halve_then_single_steps", 32, None) == 16
    assert next_target("halve_then_single_steps", 16, None) == 15
    assert next_target("single_steps", 64, None) == 63
    assert next_target("halve_then_single_steps", 64, 32) == 48
    assert next_target("halve_then_single_steps", 17, 16) is None
    assert next_target("halve_then_single_steps", 0, None) is None

    assert next_target("drop_then_bisect", 64, None) == 0
    assert next_target("drop_then_bisect", 64, 0) == 32

    tried: list[int] = []

    def passes_down_to_20(target: int) -> bool:
        tried.append(target)
        return target >= 20

    assert search_position("halve_then_single_steps", 64, passes_down_to_20) == 20
    assert tried == [32, 16, 24, 20, 18, 19]
    assert search_position("single_steps", 3, lambda target: target >= 2) == 2
    assert search_position("drop_then_bisect", 64, lambda target: target == 0) == 0
    assert search_position("drop_then_bisect", 64, lambda target: target >= 20) == 20


def test_fresh_search_tries_every_step_and_keeps_the_lowest_pass() -> None:
    tried: list[int] = []

    def fails_at_48_then_passes_down_to_8(target: int) -> bool:
        tried.append(target)
        return 8 <= target < 48

    ladder = (48, 32, 16, 8, 4, 0)
    assert lowest_passing_dimension(ladder, 64, fails_at_48_then_passes_down_to_8) == 8
    assert tried == list(ladder)
    assert lowest_passing_dimension(ladder, 64, lambda target: False) == 64


@pytest.mark.parametrize("sign", (1, -1))
def test_pairs_meet_every_requirement_and_splits_share_no_prompts(sign: int) -> None:
    rng = np.random.default_rng(0)
    a_grid, b_grid = np.meshgrid(np.arange(1, 41), np.arange(1, 41), indexing="ij")
    correct = (a_grid + sign * b_grid >= 1) & (rng.random(a_grid.shape) < 0.9)
    splits = split_prompts(correct, 0.6, 0.2, rng)
    assert ((splits >= 0) == correct).all()
    a, b, a1, b1 = sample_pairs(splits, TRAIN, sign, 512, rng)
    for x, y in ((a, b), (a1, b1), (a1, b), (a, b1)):
        assert (splits[x - 1, y - 1] == TRAIN).all()
        assert (x + sign * y >= 1).all()
    answers = np.stack([a + sign * b, a1 + sign * b, a + sign * b1, a1 + sign * b1])
    for i in range(4):
        for j in range(i + 1, 4):
            assert (answers[i] != answers[j]).all()
