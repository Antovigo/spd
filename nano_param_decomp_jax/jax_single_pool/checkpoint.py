"""Checkpoint / resume of the generic trainer's `TrainState` via orbax (SPEC S22).

The whole trajectory — V/U + CI masters, both optimizer states, the persistent
adversary (sources + its Adam moments), and the step counter — lives in `TrainState`
as one pytree; orbax saves it **sharded** (every process writes its own shards, no
full-gather on the training loop) and restores it onto the reference state's
shardings. The frozen target is NOT saved (SPEC §3): resume rebuilds it from HF and
loads only the trajectory.

Synchronous saves (no async): a SIGTERM-triggered save must be on disk before the
process exits for SLURM requeue-resume.
"""

from pathlib import Path
from typing import cast

import jax
import orbax.checkpoint as ocp

from jax_single_pool.train import TrainState


def make_checkpoint_manager(ckpt_dir: Path, keep_last: int) -> ocp.CheckpointManager:
    return ocp.CheckpointManager(
        ckpt_dir.resolve(),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=keep_last,
            enable_async_checkpointing=False,
        ),
    )


def save_state(mgr: ocp.CheckpointManager, step: int, state: TrainState) -> None:
    mgr.save(step, args=ocp.args.StandardSave(state))  # pyright: ignore[reportCallIssue]
    mgr.wait_until_finished()


def restore_latest(
    mgr: ocp.CheckpointManager, reference: TrainState
) -> tuple[TrainState, int] | None:
    """Restore the newest checkpoint onto `reference`'s shapes/dtypes/shardings
    (a freshly-initialised, correctly-placed `TrainState`). None if no checkpoint."""
    step = mgr.latest_step()
    if step is None:
        return None
    abstract = jax.tree.map(ocp.utils.to_shape_dtype_struct, reference)
    restored = mgr.restore(step, args=ocp.args.StandardRestore(abstract))  # pyright: ignore[reportCallIssue]
    return cast(TrainState, restored), step
