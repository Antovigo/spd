"""The `hidden_ci_floor` parameterization and the `HiddenCIShortfallLoss` that measures it."""

from typing import override

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from param_decomp.ci_fns import HiddenCIFloorConfig, LayerwiseCiConfig, floor_hidden_ci_logits
from param_decomp.component_model import ComponentModel
from param_decomp.decomposition_targets import DecompositionTarget
from param_decomp.masks import (
    AllRoutingConfig,
    UniformKSubsetRoutingConfig,
    get_router,
)
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.hidden_ci_shortfall import (
    HiddenCIShortfallLoss,
    HiddenCIShortfallLossConfig,
)
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse, run_batch_passthrough


def _model(*, floor: HiddenCIFloorConfig | None, dual: bool = True) -> ComponentModel:
    target = nn.Linear(3, 4, bias=False)
    target.requires_grad_(False)

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = target

        @override
        def forward(self, x: Tensor) -> Tensor:
            return self.fc(x)

    return ComponentModel(
        target_model=Wrapper(),
        run_batch=run_batch_passthrough,
        decomposition_targets=[DecompositionTarget(module_path="fc", C=5)],
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[4]),
        sigmoid_type="leaky_hard",
        dual_hidden_ci=dual,
        hidden_ci_floor=floor,
    )


class TestFloorLogits:
    def test_result_is_never_below_the_floor(self):
        torch.manual_seed(0)
        hidden = {"m": torch.randn(64) * 3}
        output = {"m": torch.randn(64) * 3}
        floored = floor_hidden_ci_logits(hidden, output, HiddenCIFloorConfig())
        assert (floored["m"] >= output["m"]).all()

    def test_passes_through_well_above_the_floor(self):
        """Far above the floor the smooth max is the hidden logit itself, so an
        unconstrained hidden net is left alone where the constraint does not bind."""
        hidden = {"m": torch.tensor([5.0, 8.0])}
        output = {"m": torch.tensor([0.0, 0.5])}
        floored = floor_hidden_ci_logits(hidden, output, HiddenCIFloorConfig(sharpness=10.0))
        torch.testing.assert_close(floored["m"], hidden["m"], atol=1e-5, rtol=0)

    def test_gradient_survives_across_the_operating_range(self):
        """The reason for a *smooth* max: a hard one zeroes the gradient wherever the floor
        binds, stranding any hidden logit that fell under it. Logits live near the `[0, 1]`
        sigmoid window, so this checks the band that range can reach."""
        hidden = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], requires_grad=True)
        floored = floor_hidden_ci_logits(
            {"m": hidden}, {"m": torch.zeros(5)}, HiddenCIFloorConfig()
        )
        floored["m"].sum().backward()
        assert hidden.grad is not None
        assert (hidden.grad > 1e-5).all(), f"gradient collapsed in-range: {hidden.grad}"

    def test_gradient_underflows_far_below_the_floor(self):
        """Records the limit of the escape hatch rather than leaving it implicit: past
        `gap ≈ -87/sharpness` the gradient is exactly zero and the subcomponent is pinned to
        the output net's CI permanently."""
        hidden = torch.tensor([-20.0], requires_grad=True)
        floored = floor_hidden_ci_logits(
            {"m": hidden}, {"m": torch.zeros(1)}, HiddenCIFloorConfig()
        )
        floored["m"].sum().backward()
        assert hidden.grad is not None and hidden.grad.item() == 0.0

    def test_init_offset_is_small(self):
        """Both readout heads init to logit 0.5, so the floor must be near-identity there —
        otherwise a floored run starts somewhere the baseline never visits."""
        equal = torch.full((3,), 0.5)
        floored = floor_hidden_ci_logits({"m": equal}, {"m": equal}, HiddenCIFloorConfig())
        assert (floored["m"] - equal).max() < 0.07


class TestFloorOnTheModel:
    def test_hidden_ci_is_ordered_above_output_ci(self):
        torch.manual_seed(0)
        model = _model(floor=HiddenCIFloorConfig())
        acts = {"fc": torch.randn(8, 3)}
        out = model.calc_causal_importances(acts, sampling="continuous", role="output")
        hidden = model.calc_causal_importances(acts, sampling="continuous", role="hidden")
        assert (hidden.pre_sigmoid["fc"] >= out.pre_sigmoid["fc"]).all()
        assert (hidden.upper_leaky["fc"] >= out.upper_leaky["fc"]).all()
        assert (hidden.lower_leaky["fc"] >= out.lower_leaky["fc"]).all()

    def test_output_role_is_untouched_by_the_floor(self):
        torch.manual_seed(0)
        acts = {"fc": torch.randn(8, 3)}
        floored = _model(floor=HiddenCIFloorConfig())
        plain = _model(floor=None)
        plain.load_state_dict(floored.state_dict())
        torch.testing.assert_close(
            floored.calc_causal_importances(acts, sampling="continuous", role="output").pre_sigmoid[
                "fc"
            ],
            plain.calc_causal_importances(acts, sampling="continuous", role="output").pre_sigmoid[
                "fc"
            ],
        )

    def test_hidden_backward_does_not_reach_the_output_net(self):
        """The floor must not become a path for the hidden objective's sparsity pressure to
        drag the output CI down."""
        torch.manual_seed(0)
        model = _model(floor=HiddenCIFloorConfig())
        assert model.ci_fn_hidden is not None
        ci = model.calc_causal_importances(
            {"fc": torch.randn(8, 3)}, sampling="continuous", role="hidden"
        )
        ci.lower_leaky["fc"].sum().backward()
        assert any(p.grad is not None for p in model.ci_fn_hidden.parameters())
        assert all(p.grad is None for p in model.ci_fn.parameters())

    def test_floor_without_a_hidden_net_is_rejected(self):
        with pytest.raises(AssertionError, match="dual_hidden_ci"):
            _model(floor=HiddenCIFloorConfig(), dual=False)


def _ctx(model: ComponentModel, ci_out: Tensor, ci_hidden: Tensor) -> MetricContext:
    """A context whose two CI nets are replaced by explicit values, so the metric is tested
    against a planted violation rather than whatever the untrained nets happen to emit."""
    from param_decomp.component_model import CIOutputs

    def outputs(values: Tensor) -> CIOutputs:
        return CIOutputs(
            lower_leaky={"fc": values},
            upper_leaky={"fc": values},
            pre_sigmoid={"fc": values},
        )

    return MetricContext(
        model=model,
        batch=torch.randn(ci_out.shape[0], 3),
        target_out=torch.randn(ci_out.shape[0], 4),
        pre_weight_acts={"fc": torch.randn(ci_out.shape[0], 3)},
        ci=outputs(ci_out),
        ci_hidden=outputs(ci_hidden),
        weight_deltas={},
        step=0,
        total_steps=10,
        use_delta_component=False,
        sampling="continuous",
        n_mask_samples=1,
        reconstruction_loss=recon_loss_mse,
        is_eval=False,
    )


class TestShortfallLoss:
    def _metric(self) -> tuple[HiddenCIShortfallLoss, ComponentModel]:
        model = _model(floor=None)
        metric = HiddenCIShortfallLoss(HiddenCIShortfallLossConfig(coeff=1.0))
        metric.bind(model=model, device="cpu")
        return metric, model

    def test_zero_when_hidden_is_everywhere_at_least_output(self):
        metric, model = self._metric()
        ci_out = torch.full((4, 5), 0.3)
        loss = metric.update(_ctx(model, ci_out, ci_out + 0.2))
        assert loss.item() == 0.0

    def test_counts_only_the_hidden_below_output_direction(self):
        """A hidden CI *above* the output CI is the expected case and must not be penalised."""
        metric, model = self._metric()
        ci_out = torch.zeros(4, 5)
        assert metric.update(_ctx(model, ci_out, torch.full((4, 5), 0.9))).item() == 0.0

    def test_value_is_the_summed_shortfall_per_position(self):
        """Normalised like impmin — summed over subcomponents, averaged over positions — so
        its coefficient is on the same scale as the sparsity penalty it has to overcome."""
        metric, model = self._metric()
        ci_out = torch.full((4, 5), 0.5)
        ci_hidden = torch.full((4, 5), 0.2)
        loss = metric.update(_ctx(model, ci_out, ci_hidden))
        assert loss.item() == pytest.approx(0.3 * 5)

    def test_gradient_pushes_the_hidden_ci_up_and_leaves_the_output_ci_alone(self):
        metric, model = self._metric()
        ci_out = torch.full((4, 5), 0.5, requires_grad=True)
        ci_hidden = torch.full((4, 5), 0.2, requires_grad=True)
        metric.update(_ctx(model, ci_out, ci_hidden)).backward()
        assert ci_out.grad is None or torch.count_nonzero(ci_out.grad) == 0
        assert ci_hidden.grad is not None and (ci_hidden.grad < 0).all()


class TestAllRouting:
    def test_all_routing_routes_every_position(self):
        router = get_router(AllRoutingConfig(), device="cpu")
        assert router.get_masks(["a", "b"], (4, 3)) == "all"

    def test_subset_routing_leaves_positions_on_the_frozen_path(self):
        """The contrast the `all` option exists to remove: under subsetting a site is frozen
        at many positions, so downstream sites inherit far less of its damage."""
        torch.manual_seed(0)
        masks = get_router(UniformKSubsetRoutingConfig(), device="cpu").get_masks(
            [f"m{i}" for i in range(7)], (64, 8)
        )
        assert isinstance(masks, dict)
        routed_frac = torch.stack([m.float().mean() for m in masks.values()]).mean()
        assert 0.4 < routed_frac.item() < 0.7
