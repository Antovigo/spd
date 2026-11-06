"""One DAS fit at one site: data on the device, Feucht et al.'s initialization and parametrization, training
and evaluation.

Initialization: random within the principal components of the base stream explaining 90% of its variance.
Parametrization: `torch.nn.utils.parametrizations.orthogonal` on a tall matrix, as they use it. Each position
has a `d`×`d` orthogonal `base`, whose first columns are the current subspace, and Householder reflectors
that start at zero, with Adam on the reflectors. Reflector `j` changes only columns `j` and later, so a
position's first `k` columns depend only on its first `k` reflectors, and masking later columns is exact."""

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
MIN_EXPLAINED_VARIANCE = 0.9
MAX_COMPONENTS = 500
METHOD = "feucht_random_pca_householder"
"""Names the initialization and parametrization, so results from any other method are not reused."""


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


def householder_columns(base: Array, reflectors: Array) -> Array:
    """The columns of one position, as in `torch`'s Householder map: `(d, d)` `base` and `(d, C)` reflectors,
    of which only the strictly lower part is used. Zero reflectors give `base[:, :C]`."""
    d, n_columns = reflectors.shape
    below = jnp.tril(reflectors, -1)
    taus = 2.0 / (1.0 + (below * below).sum(0))
    columns = jnp.eye(d, n_columns, dtype=jnp.float32)
    for j in range(n_columns - 1, -1, -1):
        v = below[:, j].at[j].set(1.0)
        columns = columns - taus[j] * v[:, None] * (v[None, :] @ columns)
    return base @ (-columns)


def advanced_base(base: Array, reflectors: Array) -> Array:
    """The `(d, d)` orthogonal base whose first `C` columns are `householder_columns(base, reflectors)`, so
    training can continue from it with zero reflectors."""
    n_columns = reflectors.shape[1]
    below = jnp.tril(reflectors, -1)
    taus = 2.0 / (1.0 + (below * below).sum(0))
    for j in range(n_columns):
        v = below[:, j].at[j].set(1.0)
        base = base - taus[j] * (base @ v)[:, None] * v[None, :]
    return base.at[:, :n_columns].multiply(-1.0)


class SiteFunctions:
    """The compiled functions of one stream point."""

    def __init__(self, model: Llama, t: int, integer_tokens: Array, learning_rate: float) -> None:
        self.t = t
        self.integer_tokens = integer_tokens
        self.optimizer = optax.adam(learning_rate)

        # The weights are jit arguments: closing over them would embed 15 GB of constants in every
        # compiled function.
        def cross_entropy(
            weights: Llama,
            reflectors: Array,
            full_base: Array,
            mask: Array,
            base: Array,
            source: Array,
            y: Array,
        ) -> Array:
            basis = jax.vmap(householder_columns)(full_base, reflectors)
            logits = logits_from(weights, patch_stream(base, source, basis, mask), t)
            log_probs = jax.nn.log_softmax(logits, -1)
            return -jnp.take_along_axis(log_probs, y[:, None], axis=1).mean()

        def train_step(
            weights: Llama,
            reflectors: Array,
            opt_state: optax.OptState,
            full_base: Array,
            mask: Array,
            base: Array,
            source: Array,
            y: Array,
        ) -> tuple[Array, optax.OptState, Array]:
            loss, gradient = jax.value_and_grad(cross_entropy, argnums=1)(
                weights, reflectors, full_base, mask, base, source, y
            )
            updates, opt_state = self.optimizer.update(gradient, opt_state)
            return cast(Array, optax.apply_updates(reflectors, updates)), opt_state, loss

        def evaluate_chunk(
            weights: Llama, basis: Array, mask: Array, base: Array, source: Array, y: Array
        ) -> tuple[Array, Array, Array]:
            """The top integer, the top token of the full vocabulary and the cross-entropy, per pair."""
            logits = logits_from(weights, patch_stream(base, source, basis, mask), t)
            log_probs = jax.nn.log_softmax(logits, -1)
            return (
                jnp.argmax(logits[:, integer_tokens], -1),
                jnp.argmax(logits, -1),
                -jnp.take_along_axis(log_probs, y[:, None], axis=1)[:, 0],
            )

        self.train_step = partial(jax.jit(train_step), model)
        self.advance_base = jax.jit(jax.vmap(advanced_base))
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


@dataclass(frozen=True)
class PrincipalComponents:
    """One site's initialization data, per position 1..4."""

    top: tuple[np.ndarray, ...]
    """`(d, n)`: the top components explaining 90% of the base stream's variance."""
    differs: np.ndarray
    """`(4,)` bool: whether the position has any base-source difference at this point."""


def principal_components(train: PairStreams, n_pairs: int) -> PrincipalComponents:
    """Their `random_subspace_in_pca`'s components, from the base stream of the first `n_pairs` pairs."""
    base = np.asarray(train.base[:n_pairs, 1:])
    source = np.asarray(train.source[:n_pairs, 1:])
    top: list[np.ndarray] = []
    for position in range(N_PATCHED):
        features = base[:, position]
        centered = features - features.mean(0, keepdims=True)
        _, singular, vh = np.linalg.svd(centered, full_matrices=False)
        energy = (singular**2).sum()
        if energy == 0:
            top.append(np.zeros((features.shape[1], 0), np.float32))
            continue
        needed = int(np.searchsorted(np.cumsum(singular**2 / energy), MIN_EXPLAINED_VARIANCE) + 1)
        n = min(needed, MAX_COMPONENTS, *features.shape)
        top.append(vh[:n].T)
    return PrincipalComponents(tuple(top), np.abs(source - base).max((0, 2)) > 0)


def random_pca_basis(
    components: PrincipalComponents,
    dims: tuple[int, ...],
    n_columns: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """`(4, d, n_columns)` orthonormal. A position's first `dims[p]` columns are random within its top
    components, plus random orthogonal columns if there are too few components, as theirs. Its later
    columns are inactive random filler."""

    def orthonormal_complement(columns: np.ndarray, n: int) -> np.ndarray:
        extra = rng.standard_normal((columns.shape[0], n), np.float32)
        extra = extra - columns @ (columns.T @ extra)
        return np.linalg.qr(extra)[0]

    positions = []
    for top, k in zip(components.top, dims, strict=True):
        n_top = top.shape[1]
        if k <= n_top:
            active = top @ np.linalg.qr(rng.standard_normal((n_top, k), np.float32))[0]
        else:
            active = np.concatenate([top, orthonormal_complement(top, k - n_top)], axis=1)
        filler = orthonormal_complement(active, n_columns - k)
        positions.append(np.concatenate([active, filler], axis=1))
    return np.stack(positions).astype(np.float32)


def complete_to_orthogonal(
    basis: np.ndarray, dims: tuple[int, ...], rng: np.random.Generator
) -> Array:
    """`(4, d, d)` orthogonal with `basis` as first columns, completed by QR of small noise, as `torch`'s
    dynamic trivialization does. A position with no active column gets the identity."""
    d, n_columns = basis.shape[1:]
    bases = []
    for columns, k in zip(basis, dims, strict=True):
        if k == 0:
            bases.append(np.eye(d, dtype=np.float32))
            continue
        noise = rng.normal(0.0, 1e-2, (d, d - n_columns)).astype(np.float32)
        q, r = np.linalg.qr(np.concatenate([columns, noise], axis=1))
        bases.append(q * np.sign(np.diag(r))[None, :])
    return jnp.asarray(np.stack(bases), jnp.float32)


def columns_of(full_base: Array, n_columns: int = MAX_DIMS) -> Array:
    """The subspace columns `(4, d, n_columns)` of `(4, d, d)` bases."""
    return full_base[:, :, :n_columns]


def dimension_mask(dims: tuple[int, ...], n_columns: int = MAX_DIMS) -> Array:
    """`(4, C)`: the first `dims[p]` columns on at position `p`."""
    assert len(dims) == N_PATCHED
    return jnp.stack([(jnp.arange(n_columns) < k).astype(jnp.float32) for k in dims])


def train_subspaces(
    functions: SiteFunctions,
    full_base: Array,
    mask: Array,
    train: PairStreams,
    batch_size: int,
    n_steps: int,
    seed: int,
) -> Array:
    """Train the masked columns from zero reflectors, restarting Adam, and return the new `(4, d, d)` base."""
    reflectors = jnp.zeros((full_base.shape[0], full_base.shape[1], mask.shape[1]), jnp.float32)
    opt_state = functions.optimizer.init(reflectors)
    rng = np.random.default_rng(seed)
    order = rng.permutation(train.n)
    cursor = 0
    for _ in range(n_steps):
        if cursor + batch_size > train.n:
            order = rng.permutation(train.n)
            cursor = 0
        batch = jnp.asarray(order[cursor : cursor + batch_size])
        cursor += batch_size
        reflectors, opt_state, _ = functions.train_step(
            reflectors,
            opt_state,
            full_base,
            mask,
            train.base[batch],
            train.source[batch],
            train.expected_token[batch],
        )
    assert bool(jnp.isfinite(reflectors).all()), "training produced non-finite subspaces"
    return functions.advance_base(full_base, reflectors)


@dataclass(frozen=True)
class Metrics:
    iia: float
    full_vocabulary_iia: float
    cross_entropy: float


def evaluate(functions: SiteFunctions, basis: Array, mask: Array, pairs: PairStreams) -> Metrics:
    integer_hits = token_hits = 0
    cross_entropy_sum = 0.0
    for s in range(0, pairs.n, EVALUATION_CHUNK):
        # A short last chunk is padded to the compiled shape, and only its real pairs are counted.
        n_real = min(EVALUATION_CHUNK, pairs.n - s)
        chunk = jnp.minimum(jnp.arange(s, s + EVALUATION_CHUNK), pairs.n - 1)
        top_integer, top_token, cross_entropy = functions.evaluate_chunk(
            basis, mask, pairs.base[chunk], pairs.source[chunk], pairs.expected_token[chunk]
        )
        real = chunk[:n_real]
        integer_hits += int((top_integer[:n_real] == pairs.expected_integer[real]).sum())
        token_hits += int((top_token[:n_real] == pairs.expected_token[real]).sum())
        cross_entropy_sum += float(cross_entropy[:n_real].sum())
    return Metrics(integer_hits / pairs.n, token_hits / pairs.n, cross_entropy_sum / pairs.n)


def timed_full_fit(
    functions: SiteFunctions,
    components: PrincipalComponents,
    dims: tuple[int, ...],
    train: PairStreams,
    batch_size: int,
    epochs: int,
    seed: int,
) -> tuple[Array, float, int]:
    """A full fit from their initialization: `(4, d, d)` base, seconds, steps."""
    n_steps = epochs * train.n // batch_size
    started = time.monotonic()
    rng = np.random.default_rng(seed)
    full_base = complete_to_orthogonal(random_pca_basis(components, dims, MAX_DIMS, rng), dims, rng)
    full_base = train_subspaces(
        functions, full_base, dimension_mask(dims), train, batch_size, n_steps, seed
    )
    full_base.block_until_ready()
    return full_base, time.monotonic() - started, n_steps
