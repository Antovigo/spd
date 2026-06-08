"""DDP utilities for the core trainer.

Process-group bring-up/teardown lives in `param_decomp_lab.distributed` — core only
reads cached state and runs collectives.
"""

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ReduceOp
from torch.types import Number


@dataclass(frozen=True, slots=True)
class DistributedState:
    rank: int
    world_size: int
    local_rank: int
    backend: Literal["nccl", "gloo"]


# Module-level cached state used as a single source of truth.
# Written by `param_decomp_lab.distributed.init_distributed/cleanup_distributed`.
_state: DistributedState | None = None

_SHOULD_GET_INITIALIZED: bool = os.environ.get("WORLD_SIZE") is not None


def get_distributed_state() -> DistributedState | None:
    """Return the cached distributed state for this process, or None when not distributed.

    Whether the process is distributed is decided once at import time from the
    `WORLD_SIZE` env var. In a distributed setting the state must have been initialized
    by `param_decomp_lab.distributed` before this is called; otherwise it must remain
    unset. Both invariants are asserted.
    """
    if _SHOULD_GET_INITIALIZED:
        assert _state is not None
        return _state
    else:
        assert _state is None
        return None


def is_distributed() -> bool:
    state = get_distributed_state()
    return state is not None


def is_main_process() -> bool:
    """True on global rank 0, or always in non-distributed runs."""
    state = get_distributed_state()
    if state is None:
        return True
    return state.rank == 0


def is_local_main_process() -> bool:
    """True on local rank 0 (one process per node in multi-node setups)."""
    state = get_distributed_state()
    if state is None:
        return True
    return state.local_rank == 0


def sync_across_processes() -> None:
    """Block until every rank reaches this point; no-op outside distributed mode."""
    if is_distributed():
        dist.barrier()


# Reduction-scope contextvar: when set, all_reduce / broadcast_tensor /
# sum_metrics_across_ranks operate on this subgroup instead of the world. Used by
# 3-pool eval to confine reductions to the PPGD pool while CI + LW pools barrier
# elsewhere. Default None = global group, no behavior change for 1-pool callers.
_reduction_group: ContextVar["dist.ProcessGroup | None"] = ContextVar(
    "_reduction_group", default=None
)


@contextmanager
def use_reduction_group(group: "dist.ProcessGroup | None") -> Iterator[None]:
    """Scope a `dist.ProcessGroup` for collective ops in `param_decomp.distributed`.

    Inside the `with` block, `all_reduce`, `broadcast_tensor`,
    `sum_metrics_across_ranks`, and `avg_metrics_across_ranks` use `group` instead
    of the default global group. Outside the block, behavior is unchanged.

    Pass `group=None` to keep the global group (useful as a no-op default).
    """
    token = _reduction_group.set(group)
    try:
        yield
    finally:
        _reduction_group.reset(token)


def active_reduction_group() -> "dist.ProcessGroup | None":
    """The reduction group set by ``use_reduction_group``, or ``None`` (global group)."""
    return _reduction_group.get()


def all_reduce(
    tensor: torch.Tensor, op: dist.ReduceOp.RedOpType = dist.ReduceOp.SUM
) -> torch.Tensor:
    """All-reduce `tensor` across ranks in place; no-op in non-distributed mode.

    Honors `use_reduction_group(...)` if active — otherwise uses the global group.
    """
    if is_distributed():
        dist.all_reduce(tensor, op=op, group=_reduction_group.get())
    return tensor


def broadcast_tensor(tensor: Tensor) -> Tensor:
    """Broadcast `tensor` from rank 0 to every other rank in place.

    Honors `use_reduction_group(...)` if active — `src=0` refers to the first rank
    of the active group when one is set.
    """
    if is_distributed():
        group = _reduction_group.get()
        src = 0 if group is None else dist.get_global_rank(group, 0)
        dist.broadcast(tensor, src=src, group=group)
    return tensor


def sum_metrics_across_ranks(
    metrics: Mapping[str, Number], device: str | torch.device
) -> Mapping[str, float]:
    """Sum each metric value across all ranks. All ranks must pass the same keys."""
    assert is_distributed(), "Can only sum metrics across ranks if running in distributed mode"
    metric_values = torch.tensor([metrics[k] for k in metrics], device=device)
    metric_values = all_reduce(metric_values, op=ReduceOp.SUM)
    return {k: metric_values[i].item() for i, k in enumerate(metrics)}


def avg_metrics_across_ranks(
    metrics: Mapping[str, Number], device: str | torch.device
) -> Mapping[str, float]:
    """Average each metric value across all ranks.

    All ranks must pass the same keys; non-distributed runs return `metrics` unchanged.
    Honors `use_reduction_group(...)` for the divisor.
    """
    state = get_distributed_state()
    if state is None:
        return metrics
    group = _reduction_group.get()
    n = state.world_size if group is None else dist.get_world_size(group)
    assert n > 0, "Reduction-group size must be greater than 0"
    sum_metrics = sum_metrics_across_ranks(metrics, device)
    return {k: v / n for k, v in sum_metrics.items()}


def gather_all_tensors(tensor: Tensor) -> list[Tensor]:
    """Gather `tensor` from every rank into a list indexed by rank.

    Requires identical shapes across ranks. The local rank's entry is replaced with the
    original tensor to preserve autograd through this rank's contribution. In
    non-distributed mode returns `[tensor]`.

    Honors `use_reduction_group(...)` — when set, gather across that subgroup's
    ranks instead of the world. The returned list's local-rank index follows the
    GROUP-local rank, not the global rank.
    """
    state = get_distributed_state()
    if state is None:
        return [tensor]

    tensor = tensor.contiguous()
    group = _reduction_group.get()
    n = state.world_size if group is None else dist.get_world_size(group)
    local_rank = state.rank if group is None else dist.get_group_rank(group, state.rank)

    gathered = [torch.zeros_like(tensor) for _ in range(n)]
    torch.distributed.all_gather(gathered, tensor, group=group)

    # Replace our rank's entry with the original to preserve autograd
    gathered[local_rank] = tensor

    return gathered


def seed_all_ranks(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def seed_per_rank(base_seed: int) -> None:
    """Seed the global RNG with `base_seed * world_size + rank` to diverge ops across ranks.

    Non-distributed: just `base_seed`.
    """
    dist_state = get_distributed_state()
    world_size = dist_state.world_size if dist_state is not None else 1
    rank = dist_state.rank if dist_state is not None else 0
    seed = base_seed * world_size + rank
    seed_all_ranks(seed)
