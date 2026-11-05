"""The DAS forward, patch, pruning search and pair sampling on a tiny random Llama."""

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
from param_decomp.das.train_das_subspaces import PairStreams, dimension_mask, orthonormal
from param_decomp.pruning.model import Arch, DenseWeights, Shared, dense_logits

D = 32
VOCAB = 50
N_LAYER = 3
T = 5


def tiny_model() -> Llama:
    keys = iter(jax.random.split(jax.random.key(0), 20))

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
    dense = DenseWeights(
        W={
            "q": normal(N_LAYER, 32, D),
            "k": normal(N_LAYER, 16, D),
            "v": normal(N_LAYER, 16, D),
            "o": normal(N_LAYER, D, 32),
            "gate": normal(N_LAYER, 64, D),
            "up": normal(N_LAYER, 64, D),
            "down": normal(N_LAYER, D, 64),
        }
    )
    return Llama(arch, shared, dense)


TOKENS = jnp.asarray(np.random.default_rng(0).integers(0, VOCAB, (6, T)))
SOURCE_TOKENS = (
    jnp.asarray(np.random.default_rng(1).integers(0, VOCAB, (6, T))).at[:, 0].set(TOKENS[:, 0])
)


@pytest.mark.parametrize("t", range(2 * N_LAYER))
def test_running_from_any_stream_point_reproduces_the_full_forward(t: int) -> None:
    model = tiny_model()
    full = dense_logits(model.arch, model.shared, model.dense, TOKENS, "bf16", "bf16")
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
    Q = orthonormal(jax.random.normal(jax.random.key(1), (4, D, 8)))
    dims = (5, 8, 3, 0)
    mask = dimension_mask(dims, 8)
    pairs = PairStreams(base, source, jnp.zeros(6, jnp.int32), jnp.zeros(6, jnp.int32))
    rotated = rotate_by_energy(Q, 1, dims[1], pairs)
    np.testing.assert_allclose(
        patch_stream(base, source, rotated, mask), patch_stream(base, source, Q, mask), atol=1e-4
    )


def test_removal_keeps_the_least_harmful_columns_first_and_q_orthonormal() -> None:
    Q = orthonormal(jax.random.normal(jax.random.key(2), (4, D, 8)))
    scores = jnp.asarray([5.0, 1.0, 3.0, 2.0])  # removing column 1 then 3 hurts least
    pruned = remove_least_harmful(Q, 2, 4, 2, scores)
    np.testing.assert_allclose(pruned[2, :, 0], Q[2, :, 0])
    np.testing.assert_allclose(pruned[2, :, 1], Q[2, :, 2])
    np.testing.assert_allclose(pruned[2].T @ pruned[2], jnp.eye(8), atol=1e-5)


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

    tried: list[int] = []

    def passes_down_to_20(target: int) -> bool:
        tried.append(target)
        return target >= 20

    assert search_position("halve_then_single_steps", 64, passes_down_to_20) == 20
    assert tried == [32, 16, 24, 20, 18, 19]
    assert search_position("single_steps", 3, lambda target: target >= 2) == 2


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
