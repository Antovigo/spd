"""Tests for the rowcombo / coupled component initializations."""

import einops
import torch

from param_decomp.components import LinearComponents
from param_decomp.optimize import init_coupled_, init_coupled_unit_, init_rowcombo_


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


def test_rowcombo_lands_in_legal_spans():
    w = _random_weight(10, 6)
    comp = LinearComponents(C=64, d_in=6, d_out=10)
    init_rowcombo_({"m": comp}, {"m": w}, seed=0)
    # Rows of a 10x6 W span all of R^6, so only the U side has a nontrivial constraint.
    assert _col_space_residual(comp.U, w).abs().max() < 1e-5

    w_t = w.T.contiguous()
    comp_t = LinearComponents(C=64, d_in=10, d_out=6)
    init_rowcombo_({"m": comp_t}, {"m": w_t}, seed=0)
    assert _row_space_residual(comp_t.V, w_t).abs().max() < 1e-5


def test_rowcombo_matches_dense_norm_statistics():
    w = _random_weight(24, 16)
    comp = LinearComponents(C=4096, d_in=16, d_out=24)
    init_rowcombo_({"m": comp}, {"m": w}, seed=0)
    v_col_sq = comp.V.pow(2).sum(dim=0)
    u_row_sq = comp.U.pow(2).sum(dim=1)
    assert abs(v_col_sq.mean().item() - 1.0) < 0.1
    assert abs(u_row_sq.mean().item() - 24 / 4096) / (24 / 4096) < 0.1


def test_rowcombo_is_seed_deterministic():
    w = _random_weight(10, 6)
    comp_a = LinearComponents(C=8, d_in=6, d_out=10)
    comp_b = LinearComponents(C=8, d_in=6, d_out=10)
    init_rowcombo_({"m": comp_a}, {"m": w}, seed=7)
    init_rowcombo_({"m": comp_b}, {"m": w}, seed=7)
    assert torch.equal(comp_a.V, comp_b.V) and torch.equal(comp_a.U, comp_b.U)


def test_coupled_wide_output_derives_u_and_sums_to_w():
    w = _random_weight(10, 6)  # d_in < d_out: V seeded, U derived
    comp = LinearComponents(C=8192, d_in=6, d_out=10)
    v_seed = comp.V.clone()
    init_coupled_({"m": comp}, {"m": w})

    assert torch.equal(comp.V, v_seed)
    assert torch.allclose(comp.U, (w @ v_seed).T * (6 / 8192), atol=1e-6)
    assert _col_space_residual(comp.U, w).abs().max() < 1e-5
    # E[sum of components] = W; sample error ~ sqrt(d_in / C).
    rel_err = (_component_sum(comp) - w).norm() / w.norm()
    assert rel_err < 0.2


def test_coupled_unit_seeds_are_unit_norm_and_derived_side_is_raw_w_image():
    w = _random_weight(10, 6)  # d_in < d_out: V seeded, U derived
    comp = LinearComponents(C=8, d_in=6, d_out=10)
    init_coupled_unit_({"m": comp}, {"m": w}, seed=0)
    assert torch.allclose(comp.V.norm(dim=0), torch.ones(8), atol=1e-5)
    assert torch.allclose(comp.U, (w @ comp.V).T, atol=1e-6)
    assert _col_space_residual(comp.U, w).abs().max() < 1e-5

    w_t = w.T.contiguous()  # d_in > d_out: U seeded, V derived
    comp_t = LinearComponents(C=8, d_in=10, d_out=6)
    init_coupled_unit_({"m": comp_t}, {"m": w_t}, seed=0)
    assert torch.allclose(comp_t.U.norm(dim=1), torch.ones(8), atol=1e-5)
    assert torch.allclose(comp_t.V, w_t.T @ comp_t.U.T, atol=1e-6)
    assert _row_space_residual(comp_t.V, w_t).abs().max() < 1e-5


def test_coupled_unit_sum_is_w_restricted_to_seed_span():
    w = _random_weight(10, 6)
    comp = LinearComponents(C=4, d_in=6, d_out=10)
    init_coupled_unit_({"m": comp}, {"m": w}, seed=0)
    # sum_c u_c v_c^T = W V V^T exactly.
    assert torch.allclose(_component_sum(comp), w @ comp.V @ comp.V.T, atol=1e-5)


def test_coupled_wide_input_derives_v_and_sums_to_w():
    w = _random_weight(6, 10)  # d_in > d_out: U seeded, V derived
    comp = LinearComponents(C=8192, d_in=10, d_out=6)
    u_seed = comp.U.clone()
    init_coupled_({"m": comp}, {"m": w})

    assert torch.equal(comp.U, u_seed)
    assert torch.allclose(comp.V, w.T @ u_seed.T, atol=1e-6)
    assert _row_space_residual(comp.V, w).abs().max() < 1e-5
    rel_err = (_component_sum(comp) - w).norm() / w.norm()
    assert rel_err < 0.2
