"""The TARGET stream's output objective of a tPD LM run (`LMTargetedExperimentConfig.target_output`).

`full_kl` (the default) is the target's own comparison: full-vocabulary KL at every position.
`last_position_integer_kl` scores only the LAST position, and only the distribution among the
single tokens of the integers `lo..hi`: both outputs' logits restricted to those tokens and
renormalized, `KL(clean || masked)`, averaged over the batch. Every prompt of the pool must end
at its last position (equal-length rows, no padding). The non-target stream is never affected.
"""

from collections.abc import Callable
from typing import Annotated, Literal, Self

import jax
import jax.numpy as jnp
import numpy as np
from param_decomp_formats.config import BaseConfig
from pydantic import Field, model_validator

from param_decomp.experiments.lm.arithmetic_probe import PromptEncoder
from param_decomp.targets.lm_output import LMOutput, StreamedLinearOutput


class FullOutputKL(BaseConfig):
    kind: Literal["full_kl"] = "full_kl"


class LastPositionIntegerKL(BaseConfig):
    kind: Literal["last_position_integer_kl"] = "last_position_integer_kl"
    integers: tuple[int, int] = (0, 999)
    """Inclusive range; each integer must be ONE token (no leading space)."""

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        lo, hi = self.integers
        assert 0 <= lo <= hi, self.integers
        return self


TargetOutputConfig = Annotated[FullOutputKL | LastPositionIntegerKL, Field(discriminator="kind")]


def integer_token_ids(tokenizer: PromptEncoder, lo: int, hi: int) -> np.ndarray:
    """The single token of every integer `lo..hi` (each asserted single and distinct)."""
    ids = []
    for x in range(lo, hi + 1):
        encoded = [int(t) for t in tokenizer.encode(str(x), add_special_tokens=False)]
        assert len(encoded) == 1, f"{x} is not a single token: {encoded}"
        ids.append(encoded[0])
    assert len(set(ids)) == len(ids)
    return np.asarray(ids, np.int32)


def last_position_logits(output: LMOutput, token_ids: jax.Array) -> jax.Array:
    """`[*batch, K]` fp32 logits of `token_ids` at the last position of `[*batch, T, ...]`."""
    match output:
        case StreamedLinearOutput(activations=activations, head=head):
            last = activations[..., -1, :].astype(jnp.float32)
            return last @ jnp.take(head, token_ids, axis=0).astype(jnp.float32).T
        case jax.Array():
            return jnp.take(output[..., -1, :], token_ids, axis=-1).astype(jnp.float32)


def last_position_subset_kl(
    token_ids: np.ndarray,
) -> Callable[[LMOutput, LMOutput], jax.Array]:
    """The mean over the batch of `KL(clean || masked)` among `token_ids` at the last position."""
    ids = np.asarray(token_ids, np.int32)

    def recon_loss_fn(masked: LMOutput, clean: LMOutput) -> jax.Array:
        token_ids = jnp.asarray(ids)
        log_q = jax.nn.log_softmax(last_position_logits(masked, token_ids), axis=-1)
        log_p = jax.nn.log_softmax(last_position_logits(clean, token_ids), axis=-1)
        return jnp.sum(jnp.exp(log_p) * (log_p - log_q), axis=-1).mean()

    return recon_loss_fn


def target_recon_loss_fn(
    config: FullOutputKL | LastPositionIntegerKL, tokenizer: PromptEncoder
) -> Callable[[LMOutput, LMOutput], jax.Array] | None:
    """The engine's `target_recon_loss_fn` for `config`; None keeps the target's own."""
    match config:
        case FullOutputKL():
            return None
        case LastPositionIntegerKL(integers=(lo, hi)):
            return last_position_subset_kl(integer_token_ids(tokenizer, lo, hi))
