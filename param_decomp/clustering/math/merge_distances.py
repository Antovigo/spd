import multiprocessing
import os
from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np

from param_decomp.clustering.math.matching_dist import matching_dist, matching_dist_vec
from param_decomp.clustering.math.perm_invariant_hamming import (
    perm_invariant_hamming_matrix,
)
from param_decomp.clustering.types import (
    DistancesArray,
    DistancesMethod,
    MergesArray,
)

_T = TypeVar("_T")
_R = TypeVar("_R")

_WORKER_CONTEXT = multiprocessing.get_context("forkserver")
"""NOT the platform default (`fork` on Linux). Every caller reaches here with JAX
imported, and JAX runs ~100 threads: `fork(2)` hands the child a copy of a mutex whose
owning thread does not exist on the other side, so the child blocks in futex forever and
the parent blocks in `wait(2)`. `forkserver` forks from a thread-free server process."""


def _run_parallel(func: Callable[[_T], _R], items: Sequence[_T]) -> list[_R]:
    assert items, "nothing to distribute"
    # sched_getaffinity, not cpu_count: under a SLURM/cgroup allocation the machine's core
    # count is not ours to spend.
    with _WORKER_CONTEXT.Pool(min(len(items), len(os.sched_getaffinity(0)))) as pool:
        return pool.map(func, items)


def compute_distances(
    normalized_merge_array: MergesArray,
    method: DistancesMethod,
) -> DistancesArray:
    match method:
        case "perm_invariant_hamming":
            pairwise_distances = perm_invariant_hamming_matrix
        case "matching_dist":
            pairwise_distances = matching_dist
        case "matching_dist_vec":
            pairwise_distances = matching_dist_vec
    n_iters = normalized_merge_array.shape[1]
    labels_per_iteration = [normalized_merge_array[:, i, :] for i in range(n_iters)]
    return np.stack(_run_parallel(pairwise_distances, labels_per_iteration), axis=0)
