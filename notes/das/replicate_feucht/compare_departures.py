"""Which departure from Feucht et al.'s DAS makes ours need more steps?

Reimplements their fit in our JAX forward, at the last token before layer 18's MLP (their IIA: first operand
61.7%, second operand 63.8%, result 87.6%), then changes one setting at a time to ours:

- parametrization: `torch`'s Householder map (theirs), or re-orthonormalizing columns after each update (ours);
- initialization: random within the principal components explaining 90% of the variance (theirs), or the top
  singular vectors of source minus base (ours);
- pairs: their 4,096 pairs split 80/20 (theirs), or pairs from prompts with all expected answers correct and
  no prompt shared between splits (ours).

Every fit is scored on both held-out sets, by the top token over the full vocabulary, as their metric does.

    python compare_departures.py --weights <dir> --tokenized <npz> --their-pairs <json> \
        --our-pairs <npz> --pca-features <safetensors> --out <csv> [--arms name ...]
"""

import argparse
import csv
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array
from safetensors.numpy import load_file
from transformers import AutoTokenizer

from param_decomp.das.build_prompt_pairs import N_OPERAND, TRAIN, VALIDATION
from param_decomp.das.patched_llama_forward import Llama, load_llama, logits_from, stream_at
from param_decomp.pruning.problem import integer_token_ids

STREAM_POINT = 37
LAST_TOKEN = 4
BATCH_SIZE = 16
EPOCHS = 8
THEIR_TRAIN_PAIRS = 3276
CHUNK = 256
SEEDS = (0, 1, 2)
DIMENSIONS = {"first_operand": 16, "second_operand": 15, "result": 16}


@dataclass(frozen=True)
class Arm:
    parametrization: str  # "householder" or "reorthonormalize"
    initialization: str  # "random_pca" or "source_minus_base"
    pairs: str  # "theirs" or "ours"
    learning_rate: float = 1e-4


THEIRS = Arm("householder", "random_pca", "theirs")
ARMS = {
    "theirs_in_jax": THEIRS,
    "our_parametrization": Arm("reorthonormalize", "random_pca", "theirs"),
    "our_parametrization_learning_rate_2e-4": Arm("reorthonormalize", "random_pca", "theirs", 2e-4),
    "our_initialization": Arm("householder", "source_minus_base", "theirs"),
    "our_pairs": Arm("householder", "random_pca", "ours"),
    "all_ours": Arm("reorthonormalize", "source_minus_base", "ours"),
    "all_ours_learning_rate_2e-4": Arm("reorthonormalize", "source_minus_base", "ours", 2e-4),
}


def orthonormal(columns: Array) -> Array:
    """Our earlier parametrization, kept here for this comparison: Cholesky QR, applied twice."""

    def once(w: Array) -> Array:
        factor = jnp.linalg.cholesky(jnp.swapaxes(w, -1, -2) @ w)
        solved = jax.lax.linalg.triangular_solve(
            factor, jnp.swapaxes(w, -1, -2), left_side=True, lower=True
        )
        return jnp.swapaxes(solved, -1, -2)

    return once(once(columns))


@dataclass(frozen=True)
class Operands:
    a: np.ndarray
    b: np.ndarray
    a1: np.ndarray
    b1: np.ndarray

    def __getitem__(self, selected: slice | np.ndarray) -> "Operands":
        return Operands(self.a[selected], self.b[selected], self.a1[selected], self.b1[selected])


def their_pairs(path: Path) -> Operands:
    data = json.loads(path.read_text())
    return Operands(
        np.asarray([int(x["input"]) for x in data["input"]]),
        np.asarray([int(x["offset"]) for x in data["input"]]),
        np.asarray([int(x[0]["input"]) for x in data["counterfactual_inputs"]]),
        np.asarray([int(x[0]["offset"]) for x in data["counterfactual_inputs"]]),
    )


def our_pairs(path: Path, tokenized: np.lib.npyio.NpzFile) -> tuple[Operands, np.ndarray]:
    pairs = np.load(path)
    base, source = pairs["base_prompt_row"], pairs["source_prompt_row"]
    operands = Operands(
        tokenized["a"][base], tokenized["b"][base], tokenized["a"][source], tokenized["b"][source]
    )
    return operands, pairs["split"]


def expected(variable: str, x: Operands) -> np.ndarray:
    match variable:
        case "first_operand":
            return x.a1 + x.b
        case "second_operand":
            return x.a + x.b1
        case "result":
            return x.a1 + x.b1
        case _:
            raise ValueError(variable)


class Streams:
    """One set of pairs at the stream point, on the device."""

    def __init__(
        self,
        stream_chunk: Callable[[Array], Array],
        tokens: np.ndarray,
        x: Operands,
        integer_tokens: Array,
    ) -> None:
        def streams(a: np.ndarray, b: np.ndarray) -> Array:
            rows = (a - 1) * N_OPERAND + (b - 1)
            chunks = []
            for s in range(0, rows.size, CHUNK):
                part = rows[s : s + CHUNK]
                padded = np.concatenate([part, np.full(CHUNK - part.size, part[0])])
                chunks.append(stream_chunk(jnp.asarray(tokens[padded]))[: part.size])
            return jnp.concatenate(chunks)

        self.x = x
        self.base = streams(x.a, x.b)
        self.source = streams(x.a1, x.b1)
        self.integer_tokens = integer_tokens
        self.n = x.a.size

    def tokens_of(self, variable: str) -> Array:
        return self.integer_tokens[jnp.asarray(expected(variable, self.x))]

    def head(self, n: int) -> "Streams":
        out = object.__new__(Streams)
        out.x, out.base, out.source = self.x[:n], self.base[:n], self.source[:n]
        out.integer_tokens, out.n = self.integer_tokens, n
        return out


def random_pca_columns(
    features: np.ndarray, k: int, rng: np.random.Generator
) -> tuple[np.ndarray, int]:
    """Their `random_subspace_in_pca`: random `k` columns within the top components explaining 90%."""
    centered = features - features.mean(0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    ratio = singular**2 / (singular**2).sum()
    n_components = int(np.searchsorted(np.cumsum(ratio), 0.9) + 1)
    top = vh[:n_components].T
    assert k <= n_components, (k, n_components)
    q, _ = np.linalg.qr(rng.standard_normal((n_components, k)))
    return (top @ q).astype(np.float32), n_components


def source_minus_base_columns(train: Streams, k: int) -> np.ndarray:
    difference = np.asarray(train.source[:, LAST_TOKEN] - train.base[:, LAST_TOKEN])
    _, _, vh = np.linalg.svd(difference, full_matrices=False)
    return vh[:k].T


def full_base(columns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The `d`×`d` base of `torch`'s dynamic trivialization: the columns, completed by QR."""
    d, k = columns.shape
    extra = rng.normal(0.0, 1e-2, (d, d - k)).astype(np.float32)
    q, r = np.linalg.qr(np.concatenate([columns, extra], axis=1))
    return (q * np.sign(np.diag(r))[None, :]).astype(np.float32)


def householder_columns(base: Array, reflectors: Array) -> Array:
    """`torch`'s Householder map, from the strictly lower part of `reflectors`, with the start at the
    base's own columns: `reflectors` = 0 gives `base[:, :k]`."""
    d, k = reflectors.shape
    below = jnp.tril(reflectors, -1)
    columns = jnp.eye(d, k, dtype=jnp.float32)
    taus = 2.0 / (1.0 + (below * below).sum(0))
    for j in range(k - 1, -1, -1):
        v = below[:, j].at[j].set(1.0)
        columns = columns - taus[j] * v[:, None] * (v[None, :] @ columns)
    return base @ (-columns)


def make_functions(arm: Arm):  # noqa: ANN201
    optimizer = optax.adam(arm.learning_rate)

    def basis_of(parameters: Array, base: Array) -> Array:
        match arm.parametrization:
            case "householder":
                return householder_columns(base, parameters)
            case "reorthonormalize":
                return orthonormal(parameters)
            case _:
                raise ValueError(arm.parametrization)

    def patched_logits(weights: Llama, basis: Array, x_base: Array, x_source: Array) -> Array:
        difference = x_source[:, LAST_TOKEN] - x_base[:, LAST_TOKEN]
        patched = x_base.at[:, LAST_TOKEN].add((difference @ basis) @ basis.T)
        return logits_from(weights, patched, STREAM_POINT)

    def loss_fn(
        parameters: Array, weights: Llama, base: Array, x_base: Array, x_source: Array, y: Array
    ) -> Array:
        logits = patched_logits(weights, basis_of(parameters, base), x_base, x_source)
        log_probs = jax.nn.log_softmax(logits, -1)
        return -jnp.take_along_axis(log_probs, y[:, None], axis=1).mean()

    def step(
        parameters: Array,
        opt_state: optax.OptState,
        weights: Llama,
        base: Array,
        x_base: Array,
        x_source: Array,
        y: Array,
    ) -> tuple[Array, optax.OptState, Array]:
        loss, gradient = jax.value_and_grad(loss_fn)(parameters, weights, base, x_base, x_source, y)
        updates, opt_state = optimizer.update(gradient, opt_state)
        parameters = cast(Array, optax.apply_updates(parameters, updates))
        if arm.parametrization == "reorthonormalize":
            parameters = orthonormal(parameters)
        return parameters, opt_state, loss

    def predict(weights: Llama, basis: Array, x_base: Array, x_source: Array) -> Array:
        return jnp.argmax(patched_logits(weights, basis, x_base, x_source), -1)

    return optimizer, jax.jit(step), jax.jit(predict), jax.jit(basis_of)


def score(
    predict: Callable[[Llama, Array, Array, Array], Array],
    model: Llama,
    basis: Array,
    pairs: Streams,
    variable: str,
) -> float:
    expected_tokens = pairs.tokens_of(variable)
    hits = 0
    for s in range(0, pairs.n, CHUNK):
        chunk = jnp.arange(s, min(s + CHUNK, pairs.n))
        padded = jnp.concatenate([chunk, jnp.full(CHUNK - chunk.size, chunk[0])])
        top = predict(model, basis, pairs.base[padded], pairs.source[padded])[: chunk.size]
        hits += int((top == expected_tokens[chunk]).sum())
    return hits / pairs.n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--tokenized", type=Path, required=True)
    parser.add_argument("--their-pairs", type=Path, required=True)
    parser.add_argument("--our-pairs", type=Path, required=True)
    parser.add_argument("--pca-features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    arguments = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(str(arguments.weights))
    integer_tokens = jnp.asarray(integer_token_ids(tokenizer))
    model = load_llama(arguments.weights)
    tokenized = np.load(arguments.tokenized)
    tokens = tokenized["tokens"]
    stream_chunk = jax.jit(lambda weights, t: stream_at(weights, t, STREAM_POINT))

    def streams(x: Operands) -> Streams:
        return Streams(lambda t: stream_chunk(model, t), tokens, x, integer_tokens)

    theirs = their_pairs(arguments.their_pairs)
    ours, our_split = our_pairs(arguments.our_pairs, tokenized)
    train_sets = {
        "theirs": streams(theirs[:THEIR_TRAIN_PAIRS]),
        "ours": streams(ours[our_split == TRAIN][:THEIR_TRAIN_PAIRS]),
    }
    their_test = streams(theirs[THEIR_TRAIN_PAIRS:])
    our_validation = streams(ours[our_split == VALIDATION])
    features = load_file(str(arguments.pca_features))["features"][:THEIR_TRAIN_PAIRS].astype(
        np.float32
    )
    print(f"their test {their_test.n} pairs, our validation {our_validation.n} pairs", flush=True)

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["arm", "variable", "seed", "their_test_iia", "our_validation_iia", "train_iia",
              "components_for_90_percent", "steps", "seconds"]  # fmt: skip
    new_file = not arguments.out.exists()
    with arguments.out.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for name in arguments.arms:
            arm = ARMS[name]
            optimizer, step, predict, basis_of = make_functions(arm)
            train = train_sets[arm.pairs]
            for variable, k in DIMENSIONS.items():
                y_all = train.tokens_of(variable)
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    n_components = 0
                    match arm.initialization:
                        case "random_pca":
                            columns, n_components = random_pca_columns(features, k, rng)
                        case "source_minus_base":
                            columns = source_minus_base_columns(train, k)
                        case _:
                            raise ValueError(arm.initialization)
                    base = jnp.asarray(full_base(columns, rng))
                    parameters = (
                        jnp.zeros((columns.shape[0], k), jnp.float32)
                        if arm.parametrization == "householder"
                        else jnp.asarray(columns)
                    )
                    opt_state = optimizer.init(parameters)
                    started = time.monotonic()
                    steps = 0
                    for _ in range(EPOCHS):
                        order = rng.permutation(train.n)
                        for s in range(0, train.n - BATCH_SIZE + 1, BATCH_SIZE):
                            batch = jnp.asarray(order[s : s + BATCH_SIZE])
                            parameters, opt_state, _loss = step(
                                parameters, opt_state, model, base, train.base[batch],
                                train.source[batch], y_all[batch],
                            )  # fmt: skip
                            steps += 1
                    basis = basis_of(parameters, base)
                    basis.block_until_ready()
                    seconds = time.monotonic() - started
                    row = {
                        "arm": name,
                        "variable": variable,
                        "seed": seed,
                        "their_test_iia": score(predict, model, basis, their_test, variable),
                        "our_validation_iia": score(
                            predict, model, basis, our_validation, variable
                        ),
                        "train_iia": score(predict, model, basis, train, variable),
                        "components_for_90_percent": n_components,
                        "steps": steps,
                        "seconds": seconds,
                    }
                    writer.writerow(row)
                    stream.flush()
                    print(row, flush=True)
    print("departures finished", flush=True)


if __name__ == "__main__":
    main()
