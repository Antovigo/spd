"""Checkpoint / resume of the generic trainer's `TrainState` (SPEC S22).

The whole trajectory — V/U + CI masters, both optimizer states, the persistent
adversary (sources + its Adam moments), and the step counter — lives in `TrainState`
as one pytree, so save/resume is a flat pytree serialization. The frozen target is NOT
saved (SPEC §3): resume rebuilds it from HF and loads only the trajectory.

Uses numpy `.npz` over the flattened leaves keyed by tree position, which keeps the
file framework-light (no equinox-version coupling). All `TrainState` leaves are
fp32/int (SPEC N1), so numpy round-trips them exactly.
"""

from pathlib import Path

import jax
import numpy as np

from jax_single_pool.train import TrainState


def save_state(path: Path, state: TrainState) -> None:
    leaves = jax.tree.leaves(state)
    arrays = {str(i): np.asarray(leaf) for i, leaf in enumerate(leaves)}
    np.savez(path, **arrays)  # pyright: ignore[reportArgumentType] (numpy savez **kwds stub is strict)


def load_state(path: Path, reference: TrainState) -> TrainState:
    """Reload into the structure of `reference` (a freshly-initialised `TrainState`
    with identical shapes/dtypes — its treedef defines the layout)."""
    ref_leaves, treedef = jax.tree.flatten(reference)
    npz = np.load(path)
    assert len(npz.files) == len(ref_leaves), (
        f"checkpoint has {len(npz.files)} leaves, reference TrainState has {len(ref_leaves)}"
    )
    leaves = [np.asarray(npz[str(i)]) for i in range(len(npz.files))]
    return jax.tree.unflatten(treedef, leaves)
