import torch

from param_decomp.components import Components, LinearComponents
from param_decomp.optimize import init_coupled_, init_coupled_zero_u_


def _components_and_weights() -> tuple[dict[str, Components], dict[str, torch.Tensor]]:
    """One site per orientation, since `init_coupled_` branches on `d_in <= d_out`."""
    components: dict[str, Components] = {
        "wide": LinearComponents(C=6, d_in=4, d_out=8),
        "narrow": LinearComponents(C=6, d_in=8, d_out=4),
    }
    weights = {"wide": torch.randn(8, 4), "narrow": torch.randn(4, 8)}
    return components, weights


def test_components_start_silent_but_v_matches_coupled() -> None:
    components, weights = _components_and_weights()
    init_coupled_zero_u_(components, weights, seed=0)

    coupled, _ = _components_and_weights()
    init_coupled_(coupled, weights, seed=0)

    for name, comp in components.items():
        assert torch.equal(comp.U, torch.zeros_like(comp.U))
        assert torch.equal(comp.weight, torch.zeros_like(comp.weight))
        assert torch.equal(comp.V, coupled[name].V), f"{name}: V diverged from coupled init"
        assert comp.V.abs().max() > 0.0, f"{name}: V is degenerate, CI nets would see nothing"


def test_ci_input_is_live_and_u_has_gradient_at_step_zero() -> None:
    components, weights = _components_and_weights()
    init_coupled_zero_u_(components, weights, seed=0)

    comp = components["wide"]
    assert isinstance(comp, LinearComponents)
    x = torch.randn(3, comp.d_in)
    acts = comp.get_component_acts(x)
    assert acts.abs().max() > 0.0

    comp(x).square().sum().backward()
    assert comp.U.grad is not None and comp.U.grad.abs().max() == 0.0, (
        "at zero U the output is zero, so the squared-output objective has no gradient; "
        "use a target-matching objective to exercise the live path"
    )

    comp.zero_grad()
    target = torch.randn(3, comp.d_out)
    (comp(x) - target).square().sum().backward()
    assert comp.U.grad is not None and comp.U.grad.abs().max() > 0.0, "U is a dead parameter"
    assert comp.V.grad is not None and comp.V.grad.abs().max() == 0.0, (
        "V's gradient is proportional to U, so it must be zero on the first step"
    )
