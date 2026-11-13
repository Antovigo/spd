"""The run's target-stream persistent adversary, read from its training checkpoint.

The run's merged term (`coefficients.TARGET_TERM`, `source_shape: bsc`) keeps, per site group
(matrix kind), component sources `(L, B, T, C)` and weight-delta sources `(L, B, T)` with their
Adam moments `m`, `v` and one ascent counter: `L` (integer) blocks, `B` (integer) the run's
target batch, one row per batch slot, `T` (integer) prompt positions, `C` (integer) the kind's
components per block. The checkpoint stores it under `training/adversaries/<term>/` with the
same tree as `ci_filter.ppgd.Adversary` (`sources.stacks.<kind>.{components, delta}`,
`opt_state.{m, v}.stacks...`, `opt_state.step_count`).

`restore_run_adversary` reads it unchanged, all `B` rows. The filter's batch (4096 prompts) is
larger than `B` (128): batch element `j` reads row `j mod B` (`ci_filter.ppgd`), so each trained
row keeps attacking `batch / B` prompts per step and its gradient is the sum over them. A row is
not tied to a prompt in the run either (its batches are fresh draws every step)."""

from pathlib import Path

import jax
import orbax.checkpoint as ocp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from param_decomp.ci_filter.ppgd import Adversary
from param_decomp.core.adversary import SourcesAdamState, SourceStack, SourceStacks
from param_decomp.core.checkpoint import make_read_only_checkpoint_manager


def _stacks_tree(stacks: SourceStacks) -> dict[str, object]:
    out: dict[str, dict[str, object]] = {}
    for kind, stack in stacks.stacks.items():
        assert isinstance(stack.components, jax.Array), "dense component sources only"
        out[kind] = {"components": stack.components, "delta": stack.delta}
    return {"stacks": out}


def stored_tree(adversary: Adversary) -> dict[str, object]:
    """The adversary in the checkpoint's nested-dict form."""
    state = adversary.opt_state
    return {
        "sources": _stacks_tree(adversary.sources),
        "opt_state": {
            "m": _stacks_tree(state.m),
            "v": _stacks_tree(state.v),
            "step_count": state.step_count,
        },
    }


def _stacks_of(tree: dict[str, object], like: SourceStacks) -> SourceStacks:
    stacks = tree["stacks"]
    assert isinstance(stacks, dict)
    return SourceStacks(
        stacks={
            kind: SourceStack(components=stacks[kind]["components"], delta=stacks[kind]["delta"])
            for kind in like.stacks
        },
        site_stack_indices=like.site_stack_indices,
    )


def adversary_of(tree: dict[str, object], like: Adversary) -> Adversary:
    """`stored_tree`'s inverse, the static structure taken from `like`."""
    sources, state = tree["sources"], tree["opt_state"]
    assert isinstance(sources, dict) and isinstance(state, dict)
    return Adversary(
        sources=_stacks_of(sources, like.sources),
        opt_state=SourcesAdamState(
            m=_stacks_of(state["m"], like.opt_state.m),
            v=_stacks_of(state["v"], like.opt_state.v),
            step_count=state["step_count"],
        ),
    )


def restore_run_adversary(
    run_dir: Path, step: int, term: str, like: Adversary, mesh: jax.sharding.Mesh
) -> Adversary:
    """Checkpoint `step`'s adversary of the target-stream term `term`, replicated over `mesh`,
    in `like`'s structure; every stored leaf must have `like`'s shape and dtype."""
    ckpts = run_dir / "ckpts"
    with ocp.StandardCheckpointer() as checkpointer:
        stored = checkpointer.metadata(ckpts / str(step) / "training")
    item_metadata = stored.item_metadata
    assert item_metadata is not None
    stored_flat = ocp.tree.to_flat_dict(dict(item_metadata)["adversaries"][term])
    replicated = NamedSharding(mesh, P())
    abstract = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=replicated), stored_tree(like)
    )
    want = ocp.tree.to_flat_dict(abstract)
    assert set(stored_flat) == set(want), (sorted(stored_flat), sorted(want))
    for path, leaf in want.items():
        got = stored_flat[path]
        assert tuple(got.shape) == tuple(leaf.shape) and got.dtype == leaf.dtype, (
            f"{'.'.join(path)}: stored {got.shape} {got.dtype}, filter wants "
            f"{leaf.shape} {leaf.dtype}"
        )
    item = {"adversaries": {term: abstract}}
    restore_args = ocp.checkpoint_utils.construct_restore_args(item)
    mgr = make_read_only_checkpoint_manager(ckpts)
    with jax.set_mesh(None):
        composite = mgr.restore(
            step,
            args=ocp.args.Composite(
                training=ocp.args.PyTreeRestore(
                    item=item, restore_args=restore_args, partial_restore=True
                )
            ),
        )
    return adversary_of(composite["training"]["adversaries"][term], like)
