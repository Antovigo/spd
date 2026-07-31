import numpy as np

from param_decomp.clustering.math.merge_distances import _WORKER_CONTEXT, compute_distances
from param_decomp.clustering.types import MergesArray


def test_worker_pool_does_not_fork():
    """Every caller reaches `compute_distances` with JAX imported, so a `fork`-based pool
    hands its children a mutex no surviving thread can release and both sides hang."""
    assert _WORKER_CONTEXT.get_start_method() != "fork"


def test_compute_distances_over_the_worker_pool():
    """Exercises the pool itself on a shape small enough for the default suite: three
    iterations of a two-member ensemble over three components."""
    merges: MergesArray = np.array(
        [
            [[0, 0, 0], [0, 0, 1], [0, 1, 2]],
            [[0, 0, 0], [0, 1, 1], [0, 1, 2]],
        ],
        dtype=np.int32,
    )
    distances = compute_distances(merges, method="perm_invariant_hamming")
    assert distances.shape == (3, 2, 2)
    # Strict-lower-triangular convention: only [1, 0] is defined.
    assert distances[0, 1, 0] == 0.0  # iteration 0: both members put everything in one group
    assert distances[1, 1, 0] == 1.0  # iteration 1: the members split a different component
    assert np.isnan(distances[:, 0, 1]).all()
