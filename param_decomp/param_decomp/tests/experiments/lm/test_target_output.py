"""`experiments.lm.target_output`: the last-position integer KL on both output edges."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from param_decomp.experiments.lm.target_output import (
    FullOutputKL,
    LastPositionIntegerKL,
    integer_token_ids,
    last_position_subset_kl,
    target_recon_loss_fn,
)
from param_decomp.targets.lm_output import StreamedLinearOutput


def _reference(masked: np.ndarray, clean: np.ndarray, ids: np.ndarray) -> float:
    def log_softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=-1, keepdims=True)
        return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))

    log_q = log_softmax(masked[:, -1, ids].astype(np.float64))
    log_p = log_softmax(clean[:, -1, ids].astype(np.float64))
    return float((np.exp(log_p) * (log_p - log_q)).sum(axis=-1).mean())


def test_last_position_subset_kl_matches_numpy_on_both_edges():
    key_a, key_b, key_h = jax.random.split(jax.random.PRNGKey(0), 3)
    batch, seq, d, vocab = 3, 4, 8, 32
    ids = np.asarray([3, 7, 11, 20], np.int32)
    head = jax.random.normal(key_h, (vocab, d))
    act_m = jax.random.normal(key_a, (batch, seq, d))
    act_c = jax.random.normal(key_b, (batch, seq, d))
    logits_m, logits_c = act_m @ head.T, act_c @ head.T
    want = _reference(np.asarray(logits_m), np.asarray(logits_c), ids)
    fn = last_position_subset_kl(ids)
    assert float(fn(logits_m, logits_c)) == pytest.approx(want, rel=1e-5)
    streamed = fn(
        StreamedLinearOutput(activations=act_m, head=head, n_chunks=4),
        StreamedLinearOutput(activations=act_c, head=head, n_chunks=4),
    )
    assert float(streamed) == pytest.approx(want, rel=1e-5)
    # Earlier positions and tokens outside the subset do not enter it.
    edited = logits_m.at[:, :-1].set(0.0).at[:, -1, 0].set(50.0)
    assert float(fn(edited, logits_c)) == pytest.approx(want, rel=1e-5)
    assert float(fn(logits_c, logits_c)) == pytest.approx(0.0, abs=1e-6)


class _Digits:
    """Every integer below 1000 is its own token; anything else is two."""

    def encode(self, text: str, /, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return [int(text) + 10] if text.isdigit() and int(text) < 1000 else [1, 2]


def test_integer_token_ids_and_config_dispatch():
    np.testing.assert_array_equal(integer_token_ids(_Digits(), 0, 3), [10, 11, 12, 13])
    with pytest.raises(AssertionError, match="not a single token"):
        integer_token_ids(_Digits(), 998, 1000)
    assert target_recon_loss_fn(FullOutputKL(), _Digits()) is None
    fn = target_recon_loss_fn(LastPositionIntegerKL(integers=(0, 3)), _Digits())
    assert fn is not None
    x = jnp.zeros((1, 2, 20))
    assert float(fn(x, x)) == 0.0
