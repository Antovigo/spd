"""Checkpoint / resume of the single-pool `TrainState`.

The whole adversary trajectory (PGD sources + Adam moments) lives in `TrainState`
as a pytree, so save/resume is a flat pytree serialization — no bespoke
state-dict plumbing (contrast the torch PPGD `state_dict`/`load_state_dict`). The
frozen `W_target` round-trips with the rest; downstream tooling that wants only
the trainable surface can read `state.decomp.V/U` + `state.ci`.

Uses numpy `.npz` over the flattened leaves keyed by their tree path, which keeps
the file self-describing and framework-light (no equinox-version coupling).
"""

from pathlib import Path

import jax
import numpy as np

from jax_single_pool.step import TrainState


def save_state(path: Path, state: TrainState) -> None:
    leaves = jax.tree.leaves(state)
    arrays = {str(i): np.asarray(leaf) for i, leaf in enumerate(leaves)}
    np.savez(path, **arrays)  # pyright: ignore[reportArgumentType] (numpy savez **kwds stub is strict)


def load_state(path: Path, reference: TrainState) -> TrainState:
    """Reload into the structure of `reference` (same shapes/dtypes). `reference`
    is a freshly-initialised `TrainState` — its treedef defines the layout."""
    _, treedef = jax.tree.flatten(reference)
    npz = np.load(path)
    leaves = [np.asarray(npz[str(i)]) for i in range(len(npz.files))]
    return jax.tree.unflatten(treedef, leaves)
