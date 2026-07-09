"""Shared projection / flavor / hook machinery for the subspace-filtering scripts."""

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

FLAVORS = ("raw", "centered", "bias")


def orthonormal_basis(vectors: Tensor) -> Tensor:
    """Orthonormal basis `Q [d, r]` of the span of the columns of `vectors [d, n]`."""
    left, sv, _ = torch.linalg.svd(vectors, full_matrices=False)
    keep = sv > sv[0] * 1e-5
    return left[:, keep].contiguous()


def project_onto(x: Tensor, basis: Tensor | None) -> Tensor:
    """Project `x [..., d]` onto span(`basis [d, r]`); `None` = zero span."""
    if basis is None:
        return torch.zeros_like(x)
    return (x @ basis) @ basis.T


def project_rows(x: Tensor, bases: list[Tensor | None]) -> Tensor:
    """Project each leading-dim slice of `x [b, ..., d]` onto its own `bases[i]`."""
    assert len(bases) == x.shape[0], f"{len(bases)} bases for batch {x.shape[0]}"
    out = torch.empty_like(x)
    for i, basis in enumerate(bases):
        out[i] = 0.0 if basis is None else (x[i] @ basis) @ basis.T
    return out


def apply_flavor(x: Tensor, mu: Tensor, flavor: str, proj_fn: Callable[[Tensor], Tensor]) -> Tensor:
    """One projection flavor on `x [..., d]` with site mean `mu [d]` (all float32)."""
    match flavor:
        case "raw":
            return proj_fn(x)
        case "centered":
            return mu + proj_fn(x - mu)
        case "bias":
            return x - mu + proj_fn(mu.expand_as(x))
        case _:
            raise AssertionError(f"unknown flavor {flavor!r}")


def projection_pre_hook(transform: Callable[[Tensor], Tensor]) -> Callable[..., Any]:
    """Forward pre-hook applying `transform` to the last position of the module input."""

    def hook(_mod: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        x = args[0].clone()
        with torch.autocast("cuda", enabled=False):
            x[:, -1] = transform(x[:, -1].float()).to(x.dtype)
        return (x, *args[1:])

    return hook


def projection_out_hook(transform: Callable[[Tensor], Tensor]) -> Callable[..., Any]:
    """Forward hook applying `transform` to the last position of the module output."""

    def hook(_mod: Any, _args: tuple[Any, ...], output: Tensor) -> Tensor:
        out = output.clone()
        with torch.autocast("cuda", enabled=False):
            out[:, -1] = transform(out[:, -1].float()).to(out.dtype)
        return out

    return hook


def circuit_hook(v: Tensor, u: Tensor, mask: Tensor) -> Callable[..., Any]:
    """Replace the module's last-position output with the masked subcomponent sum.

    `v [d_in, C]`, `u [C, d_out]` float32; `mask` `[C]` (static) or `[b, C]` (per prompt).
    Earlier positions keep the original-weight output.
    """

    def hook(_mod: Any, args: tuple[Any, ...], output: Tensor) -> Tensor:
        out = output.clone()
        with torch.autocast("cuda", enabled=False):
            acts = (args[0][:, -1].float() @ v) * mask
            out[:, -1] = (acts @ u).to(out.dtype)
        return out

    return hook


def full_projection_pre_hook(transform: Callable[[Tensor], Tensor]) -> Callable[..., Any]:
    """Forward pre-hook applying `transform` to the whole `[b, seq, d]` module input."""

    def hook(_mod: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        x = args[0]
        with torch.autocast("cuda", enabled=False):
            x = transform(x.float()).to(x.dtype)
        return (x, *args[1:])

    return hook


def full_projection_out_hook(transform: Callable[[Tensor], Tensor]) -> Callable[..., Any]:
    """Forward hook applying `transform` to the whole `[b, seq, d]` module output."""

    def hook(_mod: Any, _args: tuple[Any, ...], output: Tensor) -> Tensor:
        with torch.autocast("cuda", enabled=False):
            return transform(output.float()).to(output.dtype)

    return hook


def full_circuit_hook(v: Tensor, u: Tensor, mask: Tensor) -> Callable[..., Any]:
    """Replace the module's output at every position with the masked subcomponent sum.

    `v [d_in, C]`, `u [C, d_out]` float32; `mask` `[C]`, `[b, C]` (per prompt, broadcast
    over positions), or `[b, seq, C]` (per position).
    """

    def hook(_mod: Any, args: tuple[Any, ...], output: Tensor) -> Tensor:
        with torch.autocast("cuda", enabled=False):
            acts = args[0].float() @ v  # [b, seq, C]
            acts = acts * (mask[:, None, :] if mask.ndim == 2 else mask)
            return (acts @ u).to(output.dtype)

    return hook


def project_pairs(x: Tensor, q_pad: Tensor) -> Tensor:
    """Project each (row, position) of `x [b, seq, d]` onto its own padded basis.

    `q_pad [b*seq, d, r]` holds one orthonormal basis per flattened (row, position) pair,
    zero-padded along `r` (zero columns leave the projection unchanged; an all-zero slice
    means the zero span).
    """
    b, seq, d = x.shape
    assert q_pad.shape[:2] == (b * seq, d), f"{tuple(q_pad.shape)} vs {(b * seq, d)}"
    flat = x.reshape(b * seq, d)
    coef = torch.einsum("bdr,bd->br", q_pad, flat)
    return torch.einsum("bdr,br->bd", q_pad, coef).reshape(b, seq, d)
