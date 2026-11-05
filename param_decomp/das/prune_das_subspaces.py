"""The per-position minimum by iterative pruning, and the fresh-fit search it is compared with
(plan sections 7.1 and 7.3)."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array

from param_decomp.das.patched_llama_forward import N_PATCHED
from param_decomp.das.train_das_subspaces import (
    EVALUATION_CHUNK,
    PairStreams,
    SiteFunctions,
    dimension_mask,
    evaluate,
    timed_full_fit,
    train_subspaces,
)

type Schedule = Literal["halve_then_single_steps", "single_steps"]


def round_passes(iia: float, reference_iia: float, position_margin: float) -> bool:
    """A round that improves on the reference passes."""
    return iia >= reference_iia - position_margin


def next_target(schedule: Schedule, current: int, failed: int | None) -> int | None:
    """The next dimension to try at a position, or None when its search is over. `current` is the
    last passing dimension, `failed` the lowest-gap failing one, if any."""
    if current == 0:
        return None
    if failed is not None:
        return None if current - failed == 1 else (current + failed) // 2
    match schedule:
        case "single_steps":
            return current - 1
        case "halve_then_single_steps":
            return current // 2 if current > 16 else current - 1


def search_position(schedule: Schedule, start: int, try_target: Callable[[int], bool]) -> int:
    """The lowest passing dimension reached from `start`; `try_target` runs one round."""
    current, failed = start, None
    while (target := next_target(schedule, current, failed)) is not None:
        if try_target(target):
            current = target
        else:
            failed = target
    return current


def lowest_passing_dimension(
    ladder: tuple[int, ...], start: int, try_target: Callable[[int], bool]
) -> int:
    """Try every dimension of `ladder` and return the lowest that passes, or `start` if none does."""
    return min((target for target in ladder if try_target(target)), default=start)


def rotate_by_energy(basis: Array, position: int, k: int, train: PairStreams) -> Array:
    """Rotate a position's active columns so they are ordered by source-minus-base energy. The span,
    and so the patch, does not change."""
    if k == 0:
        return basis
    active = basis[position, :, :k]
    difference = train.source[:, position + 1] - train.base[:, position + 1]
    projected = difference @ active
    _, vectors = jnp.linalg.eigh(projected.T @ projected)  # ascending eigenvalues
    return basis.at[position, :, :k].set(active @ vectors[:, ::-1])


def score_removals(
    functions: SiteFunctions,
    basis: Array,
    dims: tuple[int, ...],
    position: int,
    scoring: PairStreams,
) -> Array:
    """`(dims[position],)`: the mean cross-entropy on `scoring` with each column removed, untrained."""
    assert scoring.n == EVALUATION_CHUNK
    mask = dimension_mask(dims)
    scores = []
    for column in range(dims[position]):
        *_, cross_entropy_sum = functions.evaluate_chunk(
            basis,
            mask.at[position, column].set(0.0),
            scoring.base,
            scoring.source,
            scoring.expected_token,
        )
        scores.append(cross_entropy_sum / scoring.n)
    return jnp.stack(scores)


def remove_least_harmful(basis: Array, position: int, k: int, target: int, scores: Array) -> Array:
    """Reorder a position's columns: the `target` kept columns first, then the removed ones, which
    stay as inactive filler so `basis` keeps its shape and full rank."""
    ranked = jnp.argsort(scores)  # least harmful removal first
    removed = ranked[: k - target]
    kept = ranked[k - target :]
    order = jnp.concatenate([jnp.sort(kept), removed, jnp.arange(k, basis.shape[2])])
    return basis.at[position].set(basis[position][:, order])


def rank_positions(
    functions: SiteFunctions,
    basis: Array,
    dims: tuple[int, ...],
    positions: tuple[int, ...],
    scoring: PairStreams,
) -> tuple[int, ...]:
    """`positions` (indices 0..3 for token positions 1..4) by the rise in the training objective when
    each is zeroed, untrained, smallest first."""
    assert scoring.n == EVALUATION_CHUNK
    mask = dimension_mask(dims)
    rises = []
    for position in positions:
        *_, cross_entropy_sum = functions.evaluate_chunk(
            basis,
            mask.at[position].set(0.0),
            scoring.base,
            scoring.source,
            scoring.expected_token,
        )
        rises.append(float(cross_entropy_sum))
    return tuple(p for _, p in sorted(zip(rises, positions, strict=True)))


@dataclass(frozen=True)
class SearchRound:
    position: int
    """Token position being reduced, 1..4."""
    dims: tuple[int, ...]
    """`k_p` at token positions 1..4 tried in this round."""
    validation_iia: float
    validation_full_vocabulary_iia: float
    validation_cross_entropy: float
    passed: bool
    training_steps: int
    seconds: float


def prune_search(
    functions: SiteFunctions,
    start_basis: Array,
    start_dims: tuple[int, ...],
    positions: tuple[int, ...],
    train: PairStreams,
    validation: PairStreams,
    scoring: PairStreams,
    schedule: Schedule,
    finetune_steps: int,
    batch_size: int,
    position_margin: float,
    reference_iia: float,
    seed: int,
    report: Callable[[SearchRound], None],
) -> tuple[Array, tuple[int, ...]]:
    """Prune `positions` in order, each from where the previous one ended; fine-tune all active
    columns after every removal, and restore the last passing state when a round fails."""
    basis, dims = start_basis, start_dims
    round_index = 0

    def prune_position(position: int) -> None:
        def try_target(target: int) -> bool:
            nonlocal basis, dims, round_index
            round_index += 1
            started = time.monotonic()
            k = dims[position]
            rotated = rotate_by_energy(basis, position, k, train)
            scores = score_removals(functions, rotated, dims, position, scoring)
            candidate = remove_least_harmful(rotated, position, k, target, scores)
            candidate_dims = tuple(target if p == position else dims[p] for p in range(N_PATCHED))
            mask = dimension_mask(candidate_dims)
            candidate = train_subspaces(
                functions, candidate, mask, train, batch_size, finetune_steps, seed + round_index
            )
            metrics = evaluate(functions, candidate, mask, validation)
            passed = round_passes(metrics.iia, reference_iia, position_margin)
            report(
                SearchRound(
                    position=position + 1,
                    dims=candidate_dims,
                    validation_iia=metrics.iia,
                    validation_full_vocabulary_iia=metrics.full_vocabulary_iia,
                    validation_cross_entropy=metrics.cross_entropy,
                    passed=passed,
                    training_steps=finetune_steps,
                    seconds=time.monotonic() - started,
                )
            )
            if passed:
                basis, dims = candidate, candidate_dims
            return passed

        search_position(schedule, dims[position], try_target)

    for position in positions:
        prune_position(position)
    return basis, dims


def fresh_fit_search(
    functions: SiteFunctions,
    initial_basis: Array,
    start_dims: tuple[int, ...],
    positions: tuple[int, ...],
    ladder: tuple[int, ...],
    train: PairStreams,
    validation: PairStreams,
    batch_size: int,
    epochs: int,
    position_margin: float,
    reference_iia: float,
    seed: int,
    report: Callable[[SearchRound], None],
) -> tuple[int, ...]:
    """The same search with a full fit from the SVD initialization at every step. Every step of the
    ladder is fitted, since a smaller dimension can pass after a larger one fails, and the position
    keeps its lowest passing dimension."""
    dims = start_dims
    for position in positions:

        def try_target(target: int, position: int = position, dims: tuple[int, ...] = dims) -> bool:
            candidate_dims = tuple(target if p == position else dims[p] for p in range(N_PATCHED))
            basis, seconds, steps = timed_full_fit(
                functions, initial_basis, candidate_dims, train, batch_size, epochs, seed
            )
            metrics = evaluate(functions, basis, dimension_mask(candidate_dims), validation)
            passed = round_passes(metrics.iia, reference_iia, position_margin)
            report(
                SearchRound(
                    position=position + 1,
                    dims=candidate_dims,
                    validation_iia=metrics.iia,
                    validation_full_vocabulary_iia=metrics.full_vocabulary_iia,
                    validation_cross_entropy=metrics.cross_entropy,
                    passed=passed,
                    training_steps=steps,
                    seconds=seconds,
                )
            )
            return passed

        lowest = lowest_passing_dimension(ladder, dims[position], try_target)
        dims = tuple(lowest if p == position else dims[p] for p in range(N_PATCHED))
    return dims
