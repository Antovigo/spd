import torch

from param_decomp.components import LinearComponents


def test_frozen_subcomponents_block_grads_and_decay():
    comp = LinearComponents(C=4, d_in=3, d_out=2, bias=None)
    frozen = torch.tensor([True, False, True, False])
    v_before, u_before = comp.V.detach().clone(), comp.U.detach().clone()
    comp.freeze_subcomponents(frozen)

    (comp.V.sum() + comp.U.sum()).backward()
    assert comp.V.grad is not None and comp.U.grad is not None
    assert torch.all(comp.V.grad[:, frozen] == 0) and torch.all(comp.U.grad[frozen, :] == 0)
    assert torch.all(comp.V.grad[:, ~frozen] == 1) and torch.all(comp.U.grad[~frozen, :] == 1)

    # Mirror _apply_ci_scaled_weight_decay: keep < 1 everywhere, forced to 1 when frozen.
    keep = torch.full((4,), 0.5)
    assert comp.frozen_subcomponents is not None
    keep = keep.masked_fill(comp.frozen_subcomponents, 1.0)
    with torch.no_grad():
        comp.V.mul_(keep[None, :])
        comp.U.mul_(keep[:, None])
    assert torch.equal(comp.V[:, frozen], v_before[:, frozen])
    assert torch.equal(comp.U[frozen, :], u_before[frozen, :])
    assert torch.allclose(comp.V[:, ~frozen], v_before[:, ~frozen] * 0.5)
