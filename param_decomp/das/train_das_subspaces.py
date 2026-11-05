"""One DAS fit at one site: data on the device, SVD initialization, training and evaluation."""

import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array

from param_decomp.das.build_prompt_pairs import N_OPERAND, sign_of
from param_decomp.das.load_das_experiment_config import Task, Variable
from param_decomp.das.patched_llama_forward import (
    N_PATCHED,
    Llama,
    logits_from,
    patch_stream,
    stream_at,
)

MAX_DIMS = 64
EVALUATION_CHUNK = 256


def expected_answer(
    task: Task, variable: Variable, a: np.ndarray, b: np.ndarray, a1: np.ndarray, b1: np.ndarray
) -> np.ndarray:
    """The answer predicted when `variable` of base `(a, b)` is set from source `(a1, b1)`."""
    sign = sign_of(task)
    match variable:
        case "first_operand":
            return a1 + sign * b
        case "second_operand":
            return a + sign * b1
        case "result":
            return a1 + sign * b1


@dataclass(frozen=True)
class PairStreams:
    """One split's pairs at one stream point, on the device."""

    base: Array  # (N, T, d)
    source: Array  # (N, T, d)
    expected_integer: Array  # (N,)
    expected_token: Array  # (N,)

    @property
    def n(self) -> int:
        return int(self.base.shape[0])

    def head(self, n: int) -> "PairStreams":
        return PairStreams(
            self.base[:n], self.source[:n], self.expected_integer[:n], self.expected_token[:n]
        )


class SiteFunctions:
    """The compiled functions of one stream point."""

    def __init__(self, model: Llama, t: int, integer_tokens: Array, learning_rate: float) -> None:
        self.t = t
        self.integer_tokens = integer_tokens
        self.optimizer = optax.adam(learning_rate)

        # The weights are jit arguments: closing over them would embed 15 GB of constants in every
        # compiled function.
        def cross_entropy(
            weights: Llama, columns: Array, mask: Array, base: Array, source: Array, y: Array
        ) -> Array:
            basis = jax.vmap(lambda w: jnp.linalg.qr(w)[0])(columns)
            logits = logits_from(weights, patch_stream(base, source, basis, mask), t)
            log_probs = jax.nn.log_softmax(logits, -1)
            return -jnp.take_along_axis(log_probs, y[:, None], axis=1).mean()

        def train_step(
            weights: Llama,
            columns: Array,
            opt_state: optax.OptState,
            mask: Array,
            base: Array,
            source: Array,
            y: Array,
        ) -> tuple[Array, optax.OptState, Array]:
            loss, gradient = jax.value_and_grad(cross_entropy, argnums=1)(
                weights, columns, mask, base, source, y
            )
            updates, opt_state = self.optimizer.update(gradient, opt_state)
            return cast(Array, optax.apply_updates(columns, updates)), opt_state, loss

        def evaluate_chunk(
            weights: Llama, basis: Array, mask: Array, base: Array, source: Array, y: Array
        ) -> tuple[Array, Array, Array]:
            """The top integer and the top token of the full vocabulary per pair, and the summed
            cross-entropy."""
            logits = logits_from(weights, patch_stream(base, source, basis, mask), t)
            log_probs = jax.nn.log_softmax(logits, -1)
            cross_entropy_sum = -jnp.take_along_axis(log_probs, y[:, None], axis=1).sum()
            return (
                jnp.argmax(logits[:, integer_tokens], -1),
                jnp.argmax(logits, -1),
                cross_entropy_sum,
            )

        self.train_step = partial(jax.jit(train_step), model)
        self.evaluate_chunk = partial(jax.jit(evaluate_chunk), model)
        self.stream_chunk = partial(
            jax.jit(lambda weights, tokens: stream_at(weights, tokens, t)), model
        )


def load_split(
    functions: SiteFunctions,
    tokenized_prompts: Path,
    prompt_pairs: Path,
    task: Task,
    variable: Variable,
    split: int,
    n_pairs: int,
) -> PairStreams:
    pairs = np.load(prompt_pairs / f"{task}_prompt_pairs.npz")
    selected = np.flatnonzero(pairs["split"] == split)[:n_pairs]
    assert selected.size == n_pairs and n_pairs % EVALUATION_CHUNK == 0, (selected.size, n_pairs)
    prompts = np.load(tokenized_prompts)
    base_rows = pairs["base_prompt_row"][selected]
    source_rows = pairs["source_prompt_row"][selected]
    answers = expected_answer(
        task,
        variable,
        prompts["a"][base_rows],
        prompts["b"][base_rows],
        prompts["a"][source_rows],
        prompts["b"][source_rows],
    )
    assert answers.min() >= 1 and answers.max() <= 2 * N_OPERAND

    def streams(rows: np.ndarray) -> Array:
        chunks = [
            functions.stream_chunk(jnp.asarray(prompts["tokens"][rows[s : s + EVALUATION_CHUNK]]))
            for s in range(0, n_pairs, EVALUATION_CHUNK)
        ]
        return jnp.concatenate(chunks)

    expected_integer = jnp.asarray(answers)
    return PairStreams(
        base=streams(base_rows),
        source=streams(source_rows),
        expected_integer=expected_integer,
        expected_token=functions.integer_tokens[expected_integer],
    )


def svd_initialization(train: PairStreams, n_pairs: int) -> tuple[Array, Array]:
    """`initial_basis` `(4, d, MAX_DIMS)`: top right singular vectors of source minus base per position, and
    `differs` `(4,)` bool: whether the position has any base-source difference at this point."""
    difference = train.source[:n_pairs, 1:] - train.base[:n_pairs, 1:]
    columns, differs = [], []
    for position in range(N_PATCHED):
        _, _, vh = jnp.linalg.svd(difference[:, position], full_matrices=False)
        columns.append(vh[:MAX_DIMS].T)
        differs.append(jnp.abs(difference[:, position]).max() > 0)
    return jnp.stack(columns), jnp.stack(differs)


def dimension_mask(dims: tuple[int, ...], n_columns: int = MAX_DIMS) -> Array:
    """`(4, C)`: the first `dims[p]` columns on at position `p`."""
    assert len(dims) == N_PATCHED
    return jnp.stack([(jnp.arange(n_columns) < k).astype(jnp.float32) for k in dims])


def orthonormal(columns: Array) -> Array:
    return jax.vmap(lambda w: jnp.linalg.qr(w)[0])(columns)


def train_subspaces(
    functions: SiteFunctions,
    columns: Array,
    mask: Array,
    train: PairStreams,
    batch_size: int,
    n_steps: int,
    seed: int,
) -> Array:
    """Train the masked columns of `columns`, restarting Adam, and return orthonormal `basis`."""
    opt_state = functions.optimizer.init(columns)
    rng = np.random.default_rng(seed)
    order = rng.permutation(train.n)
    cursor = 0
    for _ in range(n_steps):
        if cursor + batch_size > train.n:
            order = rng.permutation(train.n)
            cursor = 0
        batch = jnp.asarray(order[cursor : cursor + batch_size])
        cursor += batch_size
        columns, opt_state, _ = functions.train_step(
            columns,
            opt_state,
            mask,
            train.base[batch],
            train.source[batch],
            train.expected_token[batch],
        )
    return orthonormal(columns)


@dataclass(frozen=True)
class Metrics:
    iia: float
    full_vocabulary_iia: float
    cross_entropy: float


def evaluate(functions: SiteFunctions, basis: Array, mask: Array, pairs: PairStreams) -> Metrics:
    integer_hits = token_hits = 0
    cross_entropy_sum = 0.0
    for s in range(0, pairs.n, EVALUATION_CHUNK):
        chunk = jnp.arange(s, s + EVALUATION_CHUNK)
        top_integer, top_token, ce = functions.evaluate_chunk(
            basis, mask, pairs.base[chunk], pairs.source[chunk], pairs.expected_token[chunk]
        )
        integer_hits += int((top_integer == pairs.expected_integer[chunk]).sum())
        token_hits += int((top_token == pairs.expected_token[chunk]).sum())
        cross_entropy_sum += float(ce)
    return Metrics(integer_hits / pairs.n, token_hits / pairs.n, cross_entropy_sum / pairs.n)


def timed_full_fit(
    functions: SiteFunctions,
    initial_basis: Array,
    dims: tuple[int, ...],
    train: PairStreams,
    batch_size: int,
    epochs: int,
    seed: int,
) -> tuple[Array, float, int]:
    """A full fit from the SVD initialization: `(basis, seconds, steps)`."""
    n_steps = epochs * train.n // batch_size
    started = time.monotonic()
    basis = train_subspaces(
        functions, initial_basis, dimension_mask(dims), train, batch_size, n_steps, seed
    )
    basis.block_until_ready()
    return basis, time.monotonic() - started, n_steps
