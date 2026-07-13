"""Tests for the `span_proj` init projection onto each target's row/col space."""

import torch

from param_decomp.components import LinearComponents
from param_decomp.optimize import (
    compute_target_space_bases,
    project_components_to_target_spaces_,
)


def _rank3_weight(d_out: int, d_in: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    q_out, _ = torch.linalg.qr(torch.randn(d_out, 3, generator=gen))
    q_in, _ = torch.linalg.qr(torch.randn(d_in, 3, generator=gen))
    return q_out @ torch.diag(torch.tensor([5.0, 2.0, 1.0])) @ q_in.T


def _residual_parts(
    comp: LinearComponents, q_in: torch.Tensor, q_out: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    v_res = comp.V - q_in @ (q_in.T @ comp.V)
    u_res = comp.U - (comp.U @ q_out) @ q_out.T
    return v_res, u_res


def test_compute_target_space_bases_ranks():
    w = _rank3_weight(10, 6)
    bases = compute_target_space_bases({"m": w}, rank_threshold=1e-5)
    q_in, q_out = bases["m"]
    assert q_in.shape == (6, 3) and q_out.shape == (10, 3)
    assert torch.allclose(q_in.T @ q_in, torch.eye(3), atol=1e-5)
    assert torch.allclose(q_out.T @ q_out, torch.eye(3), atol=1e-5)


def test_project_init_removes_exactly_the_out_of_span_mass():
    w = _rank3_weight(10, 6)
    bases = compute_target_space_bases({"m": w}, rank_threshold=1e-5)
    comp = LinearComponents(C=4, d_in=6, d_out=10)
    v0, u0 = comp.V.clone(), comp.U.clone()

    project_components_to_target_spaces_({"m": comp}, bases)

    q_in, q_out = bases["m"]
    v_res, u_res = _residual_parts(comp, q_in, q_out)
    assert v_res.abs().max() < 1e-5 and u_res.abs().max() < 1e-5
    # The in-span part is untouched.
    assert torch.allclose(comp.V, q_in @ (q_in.T @ v0), atol=1e-5)
    assert torch.allclose(comp.U, (u0 @ q_out) @ q_out.T, atol=1e-5)
