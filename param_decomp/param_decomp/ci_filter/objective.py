"""Pure last-position output scores: full-vocabulary and answer-restricted KL, top-1, and the
cross-entropy of the TRUE arithmetic answer (the target the model itself may get wrong)."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from param_decomp.ci_filter.config import (
    LastPositionAnswerCE,
    LastPositionIntegerKL,
    LastPositionKL,
    Objective,
)


def kl_rows(
    masked_logits: Float[Array, "B vocab"], clean_logits: Float[Array, "B vocab"]
) -> Float[Array, " B"]:
    """Per-prompt `KL(softmax(clean) || softmax(masked))`, fp32."""
    log_q = jax.nn.log_softmax(masked_logits.astype(jnp.float32), axis=-1)
    log_p = jax.nn.log_softmax(clean_logits.astype(jnp.float32), axis=-1)
    return jnp.sum(jnp.exp(log_p) * (log_p - log_q), axis=-1)


def restrict(logits: Float[Array, "B vocab"], token_ids: Int[Array, " K"]) -> Float[Array, "B K"]:
    """The logits of `token_ids` only; a softmax over them is the renormalized distribution."""
    return jnp.take(logits, token_ids, axis=-1)


def answer_logprob_rows(
    logits: Float[Array, "B vocab"],
    answer_ids: Int[Array, " K"],
    target_idx: Int[Array, " B"],
) -> Float[Array, " B"]:
    """`log p(correct answer)` under the answer-restricted, renormalized distribution. The
    target is the true result's first token, NOT the model's own prediction."""
    logp = jax.nn.log_softmax(restrict(logits, answer_ids).astype(jnp.float32), axis=-1)
    return jnp.take_along_axis(logp, target_idx[:, None], axis=1)[:, 0]


class RowScores(NamedTuple):
    kl: Float[Array, " B"]
    integer_kl: Float[Array, " B"]
    top1: Bool[Array, " B"]
    integer_top1: Bool[Array, " B"]
    answer_ce: Float[Array, " B"]
    correct: Bool[Array, " B"]


def row_scores(
    masked_logits: Float[Array, "B vocab"],
    clean_logits: Float[Array, "B vocab"],
    answer_ids: Int[Array, " K"],
    target_idx: Int[Array, " B"],
) -> RowScores:
    masked_int = restrict(masked_logits, answer_ids)
    clean_int = restrict(clean_logits, answer_ids)
    return RowScores(
        kl=kl_rows(masked_logits, clean_logits),
        integer_kl=kl_rows(masked_int, clean_int),
        top1=jnp.argmax(masked_logits, axis=-1) == jnp.argmax(clean_logits, axis=-1),
        integer_top1=jnp.argmax(masked_int, axis=-1) == jnp.argmax(clean_int, axis=-1),
        answer_ce=-answer_logprob_rows(masked_logits, answer_ids, target_idx),
        correct=jnp.argmax(masked_int, axis=-1) == target_idx,
    )


def objective_rows(
    objective: Objective,
    masked_logits: Float[Array, "B vocab"],
    clean_logits: Float[Array, "B vocab"],
    answer_ids: Int[Array, " K"],
    target_idx: Int[Array, " B"],
) -> Float[Array, " B"]:
    """The per-prompt value of the training objective. The KL arms score faithfulness to the
    target model; the CE arm scores the TRUE answer and so may leave the model behind."""
    match objective:
        case LastPositionKL():
            return kl_rows(masked_logits, clean_logits)
        case LastPositionIntegerKL():
            return kl_rows(restrict(masked_logits, answer_ids), restrict(clean_logits, answer_ids))
        case LastPositionAnswerCE():
            return -answer_logprob_rows(masked_logits, answer_ids, target_idx)
