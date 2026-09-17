"""Save and restore a fine-tuned CI fn as an orbax tree of its array leaves, and resolve the CI
a filter starts from and the ceilings it is capped by (`CIFilterConfig.ci_ceiling`).

The tree is keyed by leaf index; the restore target is the decomposition's own CI fn, whose
treedef, shapes, dtypes and shardings the saved leaves must match."""

from pathlib import Path

import equinox as eqx
import jax
import orbax.checkpoint as ocp
from jaxtyping import Array

from param_decomp.ci_filter.config import CIFilterConfig, InitFromCIFilter, InitFromRun
from param_decomp.ci_filter.paths import CIFilterOutputs
from param_decomp.ci_filter.step import Ceilings
from param_decomp.core.ci_fn import PlacedCIFn
from param_decomp.core.precision import COMPUTE_DT, cast_floating


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


def starting_ci(
    config: CIFilterConfig, run_dir: Path, step: int, run_ci_fn: PlacedCIFn
) -> tuple[PlacedCIFn, Ceilings]:
    """The CI fn a filter trains from and the ceilings capping it: `()` without `ci_ceiling`,
    else the starting point's own ceilings plus a frozen copy of the starting CI fn."""
    match config.init:
        case InitFromRun():
            ci_fn, inherited = run_ci_fn, ()
        case InitFromCIFilter(id=source_id):
            ci_fn, inherited = trained_ci(run_dir, step, source_id, run_ci_fn)
    if not config.ci_ceiling:
        return ci_fn, ()
    return ci_fn, (*inherited, cast_floating(ci_fn, COMPUTE_DT))


def trained_ci(
    run_dir: Path, step: int, filter_id: str, run_ci_fn: PlacedCIFn
) -> tuple[PlacedCIFn, Ceilings]:
    """A finished filter's trained CI fn with the ceilings it trained under (read off its
    pinned config), i.e. everything needed to reproduce the CI it reports."""
    outputs = CIFilterOutputs.for_run(run_dir, step, filter_id)
    _, ceilings = starting_ci(CIFilterConfig.from_file(outputs.config), run_dir, step, run_ci_fn)
    return restore_ci_fn(outputs.ci_fn, run_ci_fn), ceilings
