"""Save and restore a fine-tuned CI fn as an orbax tree of its array leaves.

The tree is keyed by leaf index; the restore target is the decomposition's own CI fn, whose
treedef, shapes, dtypes and shardings the saved leaves must match."""

from pathlib import Path

import equinox as eqx
import jax
import orbax.checkpoint as ocp
from jaxtyping import Array

from param_decomp.core.ci_fn import PlacedCIFn


def save_ci_fn(path: Path, ci_fn: PlacedCIFn) -> None:
    leaves: list[Array] = jax.tree.leaves(eqx.filter(ci_fn, eqx.is_array))
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(path.resolve(), {f"{i:05d}": leaf for i, leaf in enumerate(leaves)})
        checkpointer.wait_until_finished()


def restore_ci_fn(path: Path, like: PlacedCIFn) -> PlacedCIFn:
    """`like`'s structure with the saved leaves, placed like `like`'s leaves."""
    arrays, static = eqx.partition(like, eqx.is_array)
    leaves: list[Array]
    leaves, treedef = jax.tree.flatten(arrays)
    abstract = {
        f"{i:05d}": jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=leaf.sharding)
        for i, leaf in enumerate(leaves)
    }
    with ocp.StandardCheckpointer() as checkpointer:
        restored = checkpointer.restore(path.resolve(), abstract)
    assert sorted(restored) == sorted(abstract), "saved CI fn has a different leaf count"
    arrays = jax.tree.unflatten(treedef, [restored[k] for k in sorted(restored)])
    return eqx.combine(arrays, static)
