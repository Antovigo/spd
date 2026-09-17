"""Which components help the TRUE arithmetic answer and which interfere with it.

The score of a prompt is the log-probability its masked forward gives the correct result's
first token, renormalized over the integer answer set (objective 2's distribution). Two
readings of one component's role, both signed the same way — POSITIVE means "switching this
component off RAISES the correct answer's probability", i.e. the component interferes:

- `attribution_batch` (screen, every component at once): the first-order effect of ablating
  it, `-sum_t m * d(score)/d(m)` at its current CI. One forward/backward per prompt batch;
- `ablation_scores` (causal, one component at a time): the REAL change in score when that
  component's mask is zeroed at every position, one forward per component.

The first-order reading is exact only to the linearization; the screen ranks, the ablation
decides (`scripts/run_attribution.py` runs the screen over the pool, then verifies the
extremes)."""

from collections.abc import Callable
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from param_decomp.ci_filter.objective import answer_logprob_rows, restrict
from param_decomp.ci_filter.pool import ArithmeticPool, Tokenizer
from param_decomp.ci_filter.step import CIConstraints, Prepared, gather_rows, output_ci
from param_decomp.core.ci_fn import PlacedCIFn
from param_decomp.core.model import MaterializedMasking, PlacedModel
from param_decomp.core.precision import COMPUTE_DT

VALUE_OF = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b}


@dataclass(frozen=True)
class Targets:
    """Per-prompt supervision: the correct result, the token it starts with, and that token's
    index in the answer set (the restricted distribution's coordinate)."""

    values: np.ndarray
    """`(N,)` int: `a + b` or `a - b`."""
    token_ids: np.ndarray
    """`(N,)` int32: the first token of the result's decimal form (`-` for a negative one)."""
    indices: np.ndarray
    """`(N,)` int32: position of that token within `answer_ids`."""


def answer_targets(pool: ArithmeticPool, tokenizer: Tokenizer, answer_ids: np.ndarray) -> Targets:
    """The true answer of every pool prompt, in pool row order."""
    values: list[int] = []
    for block in pool.blocks:
        value_of = VALUE_OF[block.operation]
        values.extend(value_of(a, b) for a in block.grid.a_values for b in block.grid.b_values)
    first: list[int] = []
    for value in values:
        encoded = [int(t) for t in tokenizer.encode(str(value), add_special_tokens=False)]
        assert encoded, f"empty encoding for {value}"
        first.append(encoded[0])
    token_ids = np.asarray(first, dtype=np.int32)
    indices = np.searchsorted(answer_ids, token_ids).astype(np.int32)
    found = answer_ids[np.clip(indices, 0, answer_ids.size - 1)] == token_ids
    assert found.all(), (
        f"{int((~found).sum())} answers start with a token outside the answer set, "
        f"e.g. {token_ids[~found][:5].tolist()}"
    )
    return Targets(values=np.asarray(values), token_ids=token_ids, indices=indices)


class AttributionRows(eqx.Module):
    """One batch's per-prompt scores (replicated) and per-(prompt, component) effects."""

    score: Float[Array, " B"]
    """`log p(correct)` of the CI-masked forward."""
    clean_score: Float[Array, " B"]
    """The same for the unmasked target model."""
    correct: Float[Array, " B"]
    """1.0 where the masked forward's restricted argmax IS the correct answer."""
    clean_correct: Float[Array, " B"]
    """The same for the target model — the ceiling this analysis can explain."""
    effect: dict[str, Float[Array, "B C"]]
    """First-order `score(ablated) - score` per component, summed over positions: POSITIVE =
    interferes (removing it helps), negative = helps."""
    active: dict[str, Float[Array, "B C"]]
    """Max CI of the component over the prompt's positions (which prompts it can act on)."""


type AttributionBatch = Callable[
    [
        PlacedModel,
        Prepared,
        PlacedCIFn,
        CIConstraints,
        Int[Array, "N T"],
        Int[Array, " B"],
        Int[Array, " K"],
        Int[Array, " N"],
    ],
    AttributionRows,
]


def make_attribution_batch(remat: bool) -> AttributionBatch:
    """`(..., pool, idx, answer_ids, target_indices) -> AttributionRows` for one batch."""

    @eqx.filter_jit
    def attribution_batch(
        placed: PlacedModel,
        prepared: Prepared,
        ci_fn: PlacedCIFn,
        constraints: CIConstraints,
        tokens_all: Int[Array, "N T"],
        idx: Int[Array, " B"],
        answer_ids: Int[Array, " K"],
        target_all: Int[Array, " N"],
    ) -> AttributionRows:
        tokens = gather_rows(tokens_all, idx)
        target_idx = target_all[idx]
        clean = placed.clean_forward(tokens, capture_keys=ci_fn.capture_keys)
        masks = output_ci(placed, ci_fn, constraints, clean.captures, remat=False).lower

        def score_of(masks: dict[str, Array]) -> Float[Array, " B"]:
            logits = placed.masked_forward(
                prepared, tokens, masking=MaterializedMasking(component_masks=masks), remat=remat
            ).output[:, -1, :]
            return answer_logprob_rows(logits, answer_ids, target_idx)

        score, pullback = jax.vjp(score_of, masks)
        (grads,) = pullback(jnp.ones_like(score))
        logits = placed.masked_forward(
            prepared, tokens, masking=MaterializedMasking(component_masks=masks), remat=False
        ).output[:, -1, :]
        restricted = restrict(logits, answer_ids)
        clean_restricted = restrict(clean.output[:, -1, :], answer_ids)
        rows = {
            "score": score,
            "clean_score": answer_logprob_rows(clean.output[:, -1, :], answer_ids, target_idx),
            "correct": (jnp.argmax(restricted, axis=-1) == target_idx).astype(jnp.float32),
            "clean_correct": (jnp.argmax(clean_restricted, axis=-1) == target_idx).astype(
                jnp.float32
            ),
        }
        return AttributionRows(
            **{k: jax.sharding.reshard(v, P()) for k, v in rows.items()},
            effect={
                site: jax.sharding.reshard(
                    -jnp.sum(masks[site].astype(jnp.float32) * g.astype(jnp.float32), axis=1), P()
                )
                for site, g in grads.items()
            },
            active={
                site: jax.sharding.reshard(jnp.max(v.astype(jnp.float32), axis=1), P())
                for site, v in masks.items()
            },
        )

    return attribution_batch


type AblationScores = Callable[
    [
        PlacedModel,
        Prepared,
        PlacedCIFn,
        CIConstraints,
        Int[Array, "N T"],
        Int[Array, " B"],
        Int[Array, " K"],
        Int[Array, " N"],
        dict[str, Array],
    ],
    tuple[Float[Array, " B"], Float[Array, " B"]],
]


def make_ablation_scores() -> AblationScores:
    """`(..., keep) -> (score, correct)` of the CI masks scaled by a per-site `(C,)` `keep`
    vector: all ones is the baseline, one zero entry ablates that component everywhere. The
    keep vector is traced, so scanning components costs ONE compile."""

    @eqx.filter_jit
    def ablation_scores(
        placed: PlacedModel,
        prepared: Prepared,
        ci_fn: PlacedCIFn,
        constraints: CIConstraints,
        tokens_all: Int[Array, "N T"],
        idx: Int[Array, " B"],
        answer_ids: Int[Array, " K"],
        target_all: Int[Array, " N"],
        keep: dict[str, Array],
    ) -> tuple[Float[Array, " B"], Float[Array, " B"]]:
        tokens = gather_rows(tokens_all, idx)
        target_idx = target_all[idx]
        clean = placed.clean_forward(tokens, capture_keys=ci_fn.capture_keys)
        masks = output_ci(placed, ci_fn, constraints, clean.captures, remat=False).lower
        masked = {site: v * keep[site].astype(COMPUTE_DT) for site, v in masks.items()}
        logits = placed.masked_forward(
            prepared, tokens, masking=MaterializedMasking(component_masks=masked), remat=False
        ).output[:, -1, :]
        score = answer_logprob_rows(logits, answer_ids, target_idx)
        correct = (jnp.argmax(restrict(logits, answer_ids), axis=-1) == target_idx).astype(
            jnp.float32
        )
        return jax.sharding.reshard(score, P()), jax.sharding.reshard(correct, P())

    return ablation_scores
