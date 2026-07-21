"""Tests for the `coupled` and `within_span` component initializations."""

import einops
import torch

from param_decomp.components import LinearComponents
from param_decomp.optimize import init_coupled_, init_within_span_


def _random_weight(d_out: int, d_in: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(0)
    return torch.randn(d_out, d_in, generator=gen)


def _component_sum(comp: LinearComponents) -> torch.Tensor:
    return einops.einsum(comp.V, comp.U, "d_in C, C d_out -> d_out d_in")


def _col_space_residual(u: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    q_out, _, _ = torch.linalg.svd(w, full_matrices=False)
    return u - (u @ q_out) @ q_out.T


def _row_space_residual(v: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    _, _, vh = torch.linalg.svd(w, full_matrices=False)
    q_in = vh.T
    return v - q_in @ (q_in.T @ v)


def test_coupled_seeds_are_unit_norm_and_derived_side_is_raw_w_image():
    w = _random_weight(10, 6)  # d_in < d_out: V seeded, U derived
    comp = LinearComponents(C=8, d_in=6, d_out=10)
    init_coupled_({"m": comp}, {"m": w}, seed=0)
    assert torch.allclose(comp.V.norm(dim=0), torch.ones(8), atol=1e-5)
    assert torch.allclose(comp.U, (w @ comp.V).T, atol=1e-6)
    assert _col_space_residual(comp.U, w).abs().max() < 1e-5

    w_t = w.T.contiguous()  # d_in > d_out: U seeded, V derived
    comp_t = LinearComponents(C=8, d_in=10, d_out=6)
    init_coupled_({"m": comp_t}, {"m": w_t}, seed=0)
    assert torch.allclose(comp_t.U.norm(dim=1), torch.ones(8), atol=1e-5)
    assert torch.allclose(comp_t.V, w_t.T @ comp_t.U.T, atol=1e-6)
    assert _row_space_residual(comp_t.V, w_t).abs().max() < 1e-5


def test_coupled_sum_is_w_restricted_to_seed_span():
    w = _random_weight(10, 6)
    comp = LinearComponents(C=4, d_in=6, d_out=10)
    init_coupled_({"m": comp}, {"m": w}, seed=0)
    # sum_c u_c v_c^T = W V V^T exactly.
    assert torch.allclose(_component_sum(comp), w @ comp.V @ comp.V.T, atol=1e-5)


def test_coupled_is_seed_deterministic():
    w = _random_weight(10, 6)
    comp_a = LinearComponents(C=8, d_in=6, d_out=10)
    comp_b = LinearComponents(C=8, d_in=6, d_out=10)
    init_coupled_({"m": comp_a}, {"m": w}, seed=7)
    init_coupled_({"m": comp_b}, {"m": w}, seed=7)
    assert torch.equal(comp_a.V, comp_b.V) and torch.equal(comp_a.U, comp_b.U)


def test_within_span_sides_are_in_span_and_uncoupled():
    w = _random_weight(10, 6)  # d_in < d_out
    comp = LinearComponents(C=8, d_in=6, d_out=10)
    init_within_span_({"m": comp}, {"m": w}, seed=0)
    assert _row_space_residual(comp.V, w).abs().max() < 1e-5
    assert _col_space_residual(comp.U, w).abs().max() < 1e-5
    # not the coupled init: U is not the W-image of V
    u_coupled = (w @ comp.V).T
    assert not torch.allclose(comp.U / comp.U.norm(dim=1, keepdim=True),
                              u_coupled / u_coupled.norm(dim=1, keepdim=True), atol=1e-2)  # fmt: skip

    w_t = w.T.contiguous()  # d_in > d_out
    comp_t = LinearComponents(C=8, d_in=10, d_out=6)
    init_within_span_({"m": comp_t}, {"m": w_t}, seed=0)
    assert _row_space_residual(comp_t.V, w_t).abs().max() < 1e-5
    assert _col_space_residual(comp_t.U, w_t).abs().max() < 1e-5


def test_within_span_narrow_side_unit_norm_wide_side_w_image_scale():
    w = _random_weight(10, 6)
    comp = LinearComponents(C=8, d_in=6, d_out=10)
    init_within_span_({"m": comp}, {"m": w}, seed=0)
    assert torch.allclose(comp.V.norm(dim=0), torch.ones(8), atol=1e-5)
    # wide side = W h with |h| = 1, so its norm lies within W's singular-value range
    s = torch.linalg.svdvals(w)
    u_norms = comp.U.norm(dim=1)
    assert ((u_norms >= s[-1] - 1e-5) & (u_norms <= s[0] + 1e-5)).all()


def test_within_span_is_seed_deterministic():
    w = _random_weight(10, 6)
    comp_a = LinearComponents(C=8, d_in=6, d_out=10)
    comp_b = LinearComponents(C=8, d_in=6, d_out=10)
    init_within_span_({"m": comp_a}, {"m": w}, seed=7)
    init_within_span_({"m": comp_b}, {"m": w}, seed=7)
    assert torch.equal(comp_a.V, comp_b.V) and torch.equal(comp_a.U, comp_b.U)
