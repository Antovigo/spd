"""Tests for the soft legality pressure (init projection + illegal-mass decay)."""

import torch

from param_decomp.components import LinearComponents
from param_decomp.optimize import (
    apply_illegal_mass_decay_,
    compute_legality_bases,
    project_components_to_target_spaces_,
)


def _rank3_weight(d_out: int, d_in: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    q_out, _ = torch.linalg.qr(torch.randn(d_out, 3, generator=gen))
    q_in, _ = torch.linalg.qr(torch.randn(d_in, 3, generator=gen))
    return q_out @ torch.diag(torch.tensor([5.0, 2.0, 1.0])) @ q_in.T


def _illegal_parts(
    comp: LinearComponents, q_in: torch.Tensor, q_out: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    v_ill = comp.V - q_in @ (q_in.T @ comp.V)
    u_ill = comp.U - (comp.U @ q_out) @ q_out.T
    return v_ill, u_ill


def test_compute_legality_bases_ranks():
    w = _rank3_weight(10, 6)
    bases = compute_legality_bases({"m": w}, rank_threshold=1e-5)
    q_in, q_out = bases["m"]
    assert q_in.shape == (6, 3) and q_out.shape == (10, 3)
    assert torch.allclose(q_in.T @ q_in, torch.eye(3), atol=1e-5)
    assert torch.allclose(q_out.T @ q_out, torch.eye(3), atol=1e-5)


def test_project_init_removes_exactly_the_illegal_mass():
    w = _rank3_weight(10, 6)
    bases = compute_legality_bases({"m": w}, rank_threshold=1e-5)
    comp = LinearComponents(C=4, d_in=6, d_out=10)
    v0, u0 = comp.V.clone(), comp.U.clone()

    project_components_to_target_spaces_({"m": comp}, bases)

    q_in, q_out = bases["m"]
    v_ill, u_ill = _illegal_parts(comp, q_in, q_out)
    assert v_ill.abs().max() < 1e-5 and u_ill.abs().max() < 1e-5
    # The legal part is untouched.
    assert torch.allclose(comp.V, q_in @ (q_in.T @ v0), atol=1e-5)
    assert torch.allclose(comp.U, (u0 @ q_out) @ q_out.T, atol=1e-5)


def test_illegal_mass_decay_shrinks_only_illegal_part():
    w = _rank3_weight(10, 6)
    bases = compute_legality_bases({"m": w}, rank_threshold=1e-5)
    comp = LinearComponents(C=4, d_in=6, d_out=10)
    q_in, q_out = bases["m"]
    v_ill0, u_ill0 = _illegal_parts(comp, q_in, q_out)
    v_leg0 = comp.V - v_ill0

    apply_illegal_mass_decay_({"m": comp}, bases, coeff=0.5, lr=0.1)

    v_ill1, u_ill1 = _illegal_parts(comp, q_in, q_out)
    assert torch.allclose(v_ill1, v_ill0 * 0.95, atol=1e-6)
    assert torch.allclose(u_ill1, u_ill0 * 0.95, atol=1e-6)
    assert torch.allclose(comp.V - v_ill1, v_leg0, atol=1e-6)

    for _ in range(200):
        apply_illegal_mass_decay_({"m": comp}, bases, coeff=0.5, lr=0.1)
    v_ill, u_ill = _illegal_parts(comp, q_in, q_out)
    assert v_ill.norm() < 1e-4 * v_ill0.norm()
    assert u_ill.norm() < 1e-4 * u_ill0.norm()
    assert torch.allclose(comp.V - v_ill, v_leg0, atol=1e-5)
