"""Tests for `SVDLinearComponents` — the SVD-coordinate parameterization."""

import pytest
import torch
from torch import nn

from param_decomp.components import (
    LinearComponents,
    SVDLinearComponents,
    make_components,
)


def _random_weight(d_out: int, d_in: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(d_out, d_in, generator=gen)


def _spectrum_weight(spectrum: list[float], d_out: int, d_in: int) -> torch.Tensor:
    """Weight with a prescribed singular spectrum (descending)."""
    gen = torch.Generator().manual_seed(1)
    q_out, _ = torch.linalg.qr(torch.randn(d_out, len(spectrum), generator=gen))
    q_in, _ = torch.linalg.qr(torch.randn(d_in, len(spectrum), generator=gen))
    return q_out @ torch.diag(torch.tensor(spectrum)) @ q_in.T


def test_v_in_row_space_and_u_in_col_space():
    w = _spectrum_weight([5.0, 2.0, 1.0], d_out=10, d_in=6)
    # 0.0 would keep the float-noise directions of this exactly-rank-3 matrix.
    comp = SVDLinearComponents(C=4, target_weight=w, rank_threshold=1e-5)
    assert comp.r == 3

    row_basis, _ = torch.linalg.qr(w.T)  # [d_in, d_out->6], row space in first 3 cols
    col_basis, _ = torch.linalg.qr(w)
    row_p = row_basis[:, :3] @ row_basis[:, :3].T
    col_p = col_basis[:, :3] @ col_basis[:, :3].T

    v, u = comp.V, comp.U
    assert torch.allclose(row_p @ v, v, atol=1e-5)
    assert torch.allclose(u @ col_p, u, atol=1e-5)


def test_forward_matches_dense_on_full_rank_square():
    d = 8
    w = _random_weight(d, d)  # full rank a.s.
    bias = torch.randn(d)
    svd_comp = SVDLinearComponents(C=5, target_weight=w, rank_threshold=0.0, bias=bias)
    dense_comp = LinearComponents(C=5, d_in=d, d_out=d, bias=bias)
    assert svd_comp.Q_in is not None and svd_comp.Q_out is not None
    with torch.no_grad():
        dense_comp.V.copy_(svd_comp.Q_in @ svd_comp.A)
        dense_comp.U.copy_(svd_comp.B @ svd_comp.Q_out.T)

    x = torch.randn(3, 7, d)
    mask = torch.rand(3, 7, 5)
    assert torch.allclose(svd_comp(x, mask), dense_comp(x, mask), atol=1e-5)
    assert torch.allclose(
        svd_comp.get_component_acts(x), dense_comp.get_component_acts(x), atol=1e-5
    )
    assert torch.allclose(svd_comp.weight, dense_comp.weight, atol=1e-5)


def test_get_component_acts_matches_v_property():
    w = _random_weight(10, 6)
    comp = SVDLinearComponents(C=4, target_weight=w, rank_threshold=0.0)
    x = torch.randn(5, 6)
    assert torch.allclose(comp.get_component_acts(x), x @ comp.V, atol=1e-5)


def test_rank_truncation_and_tail_in_delta():
    w = _spectrum_weight([10.0, 5.0, 1e-4], d_out=12, d_in=7)
    comp = SVDLinearComponents(C=4, target_weight=w, rank_threshold=1e-2)
    assert comp.r == 2
    assert comp.Q_out is not None

    kept_p = comp.Q_out @ comp.Q_out.T
    weight = comp.weight
    assert torch.allclose(kept_p @ weight, weight, atol=1e-5)
    # The dropped singular direction of `w` is untouchable: it survives in the delta
    # exactly, for any A/B.
    delta = w - weight
    dropped = (torch.eye(12) - kept_p) @ w
    assert torch.allclose((torch.eye(12) - kept_p) @ delta, dropped, atol=1e-6)


def test_scale_subcomponents_matches_dense_semantics():
    w = _random_weight(9, 6)
    comp = SVDLinearComponents(C=4, target_weight=w, rank_threshold=0.0)
    keep = torch.tensor([1.0, 0.5, 0.0, 0.9])
    v_before, u_before = comp.V.clone(), comp.U.clone()
    with torch.no_grad():
        comp.scale_subcomponents_(keep)
    assert torch.allclose(comp.V, v_before * keep[None, :], atol=1e-6)
    assert torch.allclose(comp.U, u_before * keep[:, None], atol=1e-6)


def test_grads_reach_coordinates_only():
    w = _random_weight(9, 6)
    comp = SVDLinearComponents(C=4, target_weight=w, rank_threshold=0.0)
    param_names = {name for name, _ in comp.named_parameters()}
    assert param_names == {"A", "B"}

    comp(torch.randn(2, 6)).sum().backward()
    assert comp.A.grad is not None and comp.A.grad.abs().sum() > 0
    assert comp.B.grad is not None and comp.B.grad.abs().sum() > 0
    assert comp.Q_in is not None and not comp.Q_in.requires_grad
    assert comp.Q_out is not None and not comp.Q_out.requires_grad


def test_state_dict_round_trip_restores_basis():
    w = _random_weight(9, 6, seed=2)
    comp = SVDLinearComponents(C=4, target_weight=w, rank_threshold=0.0)
    sd = comp.state_dict()
    assert {"A", "B", "Q_in", "Q_out", "singular_values"} <= set(sd.keys())

    fresh = SVDLinearComponents(C=4, target_weight=w, rank_threshold=0.0)
    assert fresh.Q_in is not None
    with torch.no_grad():
        fresh.A.mul_(0.0)
        fresh.Q_in.mul_(-1.0)  # simulate a different SVD sign convention
    fresh.load_state_dict(sd)
    assert torch.allclose(fresh.V, comp.V, atol=1e-6)
    assert torch.allclose(fresh.U, comp.U, atol=1e-6)


def test_constrain_in_only():
    w = _spectrum_weight([5.0, 2.0, 1.0], d_out=10, d_in=6)
    comp = SVDLinearComponents(C=4, target_weight=w, rank_threshold=1e-5, constrain="in")
    assert comp.Q_in is not None and comp.Q_out is None
    assert comp.A.shape == (3, 4)
    assert comp.B.shape == (4, 10)
    assert comp.U is comp.B

    row_basis, _ = torch.linalg.qr(w.T)
    row_p = row_basis[:, :3] @ row_basis[:, :3].T
    assert torch.allclose(row_p @ comp.V, comp.V, atol=1e-5)

    x = torch.randn(5, 6)
    assert torch.allclose(comp.get_component_acts(x), x @ comp.V, atol=1e-5)
    assert torch.allclose(comp(x), (x @ comp.V) @ comp.U, atol=1e-5)


def test_constrain_out_only():
    w = _spectrum_weight([5.0, 2.0, 1.0], d_out=10, d_in=6)
    comp = SVDLinearComponents(C=4, target_weight=w, rank_threshold=1e-5, constrain="out")
    assert comp.Q_in is None and comp.Q_out is not None
    assert comp.A.shape == (6, 4)
    assert comp.B.shape == (4, 3)
    assert comp.V is comp.A

    col_basis, _ = torch.linalg.qr(w)
    col_p = col_basis[:, :3] @ col_basis[:, :3].T
    assert torch.allclose(comp.U @ col_p, comp.U, atol=1e-5)

    x = torch.randn(5, 6)
    assert torch.allclose(comp(x), (x @ comp.V) @ comp.U, atol=1e-5)


def test_make_components_dispatch():
    model = nn.Sequential(nn.Linear(6, 9, bias=False))
    comps = make_components(model, {"0": 4}, svd_rank_threshold=0.0)
    assert isinstance(comps["0"], SVDLinearComponents)
    comps_dense = make_components(model, {"0": 4})
    assert isinstance(comps_dense["0"], LinearComponents)

    embed_model = nn.Sequential(nn.Embedding(11, 5))
    with pytest.raises(AssertionError):
        make_components(embed_model, {"0": 4}, svd_rank_threshold=0.0)
