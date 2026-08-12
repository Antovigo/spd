"""The two CI-ordering constraints and the `HiddenCIShortfallLoss` that measures them.

`hidden_ci_floor` and `output_ci_cap` enforce the same inequality (`CI_hidden >= CI_output`)
but move different nets, so the tests here care as much about *where the gradient goes* as
about the inequality holding.
"""

from typing import override

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from param_decomp.ci_fns import (
    AttnConfig,
    CIRole,
    GlobalCiConfig,
    GlobalSharedTransformerCiConfig,
    HiddenCIFloorConfig,
    LayerwiseCiConfig,
    OutputCICapConfig,
    cap_output_ci_logits,
    floor_hidden_ci_logits,
)
from param_decomp.component_model import CIOutputs, ComponentModel
from param_decomp.decomposition_targets import DecompositionTarget
from param_decomp.masks import AllRoutingConfig, get_router
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.hidden_ci_shortfall import (
    HiddenCIShortfallLoss,
    HiddenCIShortfallLossConfig,
)
from param_decomp.metrics.stochastic_hidden_recon_subset import (
    StochasticHiddenReconSubsetLoss,
    StochasticHiddenReconSubsetLossConfig,
)
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse, run_batch_passthrough


class _Wrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 4, bias=False)
        self.requires_grad_(False)

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


def _model(*, floor: HiddenCIFloorConfig | None, dual: bool) -> ComponentModel:
    """Two independent layerwise CI nets."""
    return ComponentModel(
        target_model=_Wrapper(),
        run_batch=run_batch_passthrough,
        decomposition_targets=[DecompositionTarget(module_path="fc", C=5)],
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[4]),
        sigmoid_type="leaky_hard",
        dual_hidden_ci=dual,
        hidden_ci_floor=floor,
    )


def _shared_trunk_model(
    *, floor: HiddenCIFloorConfig | None, cap: OutputCICapConfig | None = None
) -> ComponentModel:
    """One trunk, two private readout heads — the configuration these constraints ship with.

    The transformer CI fn needs a sequence axis, so callers must feed `[batch, pos, d_in]`.
    """
    return ComponentModel(
        target_model=_Wrapper(),
        run_batch=run_batch_passthrough,
        decomposition_targets=[DecompositionTarget(module_path="fc", C=5)],
        ci_config=GlobalCiConfig(
            fn_type="global_shared_transformer",
            simple_transformer_ci_cfg=GlobalSharedTransformerCiConfig(
                d_model=8,
                n_blocks=1,
                mlp_hidden_dim=[16],
                attn_config=AttnConfig(n_heads=2, max_len=16),
            ),
        ),
        sigmoid_type="leaky_hard",
        dual_hidden_ci=True,
        dual_hidden_ci_shared_trunk=True,
        hidden_ci_floor=floor,
        output_ci_cap=cap,
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
        sigmoid window, so this checks the band that range can reach. Past `gap ~ -87/beta`
        the gradient underflows to exactly zero — outside the operating range, and recorded
        in `floor_hidden_ci_logits`' docstring rather than tested, since that limit is a
        property of `softplus`, not of this repo."""
        hidden = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], requires_grad=True)
        floored = floor_hidden_ci_logits(
            {"m": hidden}, {"m": torch.zeros(5)}, HiddenCIFloorConfig()
        )
        floored["m"].sum().backward()
        assert hidden.grad is not None
        assert (hidden.grad > 1e-5).all(), f"gradient collapsed in-range: {hidden.grad}"

    def test_init_offset_is_small(self):
        """Both readout heads init to logit 0.5, so the floor must be near-identity there —
        otherwise a floored run starts somewhere the baseline never visits. Guards the
        default `sharpness` against being lowered without noticing the cost."""
        equal = torch.full((3,), 0.5)
        floored = floor_hidden_ci_logits({"m": equal}, {"m": equal}, HiddenCIFloorConfig())
        assert (floored["m"] - equal).max() < 0.07


class TestCapLogits:
    def test_result_never_exceeds_the_cap(self):
        torch.manual_seed(0)
        output = {"m": torch.randn(64) * 3}
        hidden = {"m": torch.randn(64) * 3}
        capped = cap_output_ci_logits(output, hidden, OutputCICapConfig())
        assert (capped["m"] <= hidden["m"]).all()

    def test_passes_through_well_below_the_cap(self):
        output = {"m": torch.tensor([-4.0, -1.0])}
        hidden = {"m": torch.tensor([3.0, 5.0])}
        capped = cap_output_ci_logits(output, hidden, OutputCICapConfig(sharpness=10.0))
        torch.testing.assert_close(capped["m"], output["m"], atol=1e-5, rtol=0)

    def test_binding_redirects_gradient_into_the_hidden_logit(self):
        """The mechanism. Where the output net asks for more than the hidden net supports,
        its gradient must reach the *hidden* logit — that is what forces the output
        reconstruction to justify a subcomponent at the site level before it can use it."""
        output = torch.tensor([5.0], requires_grad=True)  # far above the cap: binding
        hidden = torch.tensor([0.0], requires_grad=True)
        cap_output_ci_logits({"m": output}, {"m": hidden}, OutputCICapConfig())[
            "m"
        ].sum().backward()
        assert output.grad is not None and hidden.grad is not None
        assert output.grad.item() < 1e-3, (
            f"gradient should not stay on the output logit: {output.grad}"
        )
        assert hidden.grad.item() > 0.99, (
            f"gradient should transfer to the hidden logit: {hidden.grad}"
        )

    def test_slack_cap_leaves_gradient_on_the_output_logit(self):
        output = torch.tensor([-5.0], requires_grad=True)  # far below the cap: slack
        hidden = torch.tensor([0.0], requires_grad=True)
        cap_output_ci_logits({"m": output}, {"m": hidden}, OutputCICapConfig())[
            "m"
        ].sum().backward()
        assert output.grad is not None and hidden.grad is not None
        assert output.grad.item() > 0.99
        assert hidden.grad.item() < 1e-3


class TestCapOnTheModel:
    def test_ordering_holds_under_the_cap(self):
        torch.manual_seed(0)
        model = _shared_trunk_model(floor=None, cap=OutputCICapConfig())
        out, hidden = model.calc_causal_importances_both_roles(
            {"fc": torch.randn(4, 8, 3)}, sampling="binomial", detach_inputs=False
        )
        assert (hidden.lower_leaky["fc"] >= out.lower_leaky["fc"]).all()
        assert (hidden.upper_leaky["fc"] >= out.upper_leaky["fc"]).all()

    def test_output_backward_reaches_the_hidden_head(self):
        """Under a shared trunk the trunk is trained by both roles anyway, so the invariant
        worth asserting is that the *hidden head* — private to the hidden role — receives
        gradient from an output-role backward. That path exists only via the cap."""
        torch.manual_seed(0)
        model = _shared_trunk_model(floor=None, cap=OutputCICapConfig())
        assert model.ci_fn_hidden is not None
        out, _ = model.calc_causal_importances_both_roles(
            {"fc": torch.randn(4, 8, 3)}, sampling="continuous", detach_inputs=False
        )
        out.lower_leaky["fc"].sum().backward()
        hidden_head = [p for n, p in model.ci_fn_hidden.named_parameters() if "_output_head" in n]
        assert hidden_head, "expected a private hidden readout head"
        assert any(p.grad is not None for p in hidden_head), (
            "the cap must carry output-role gradient into the hidden net"
        )

    def test_floor_and_cap_together_are_rejected(self):
        with pytest.raises(AssertionError, match="circular|same inequality"):
            _shared_trunk_model(floor=HiddenCIFloorConfig(), cap=OutputCICapConfig())


class TestFloorOnTheModel:
    def test_hidden_ci_is_ordered_above_output_ci(self):
        torch.manual_seed(0)
        model = _model(floor=HiddenCIFloorConfig(), dual=True)
        acts = {"fc": torch.randn(8, 3)}
        out = model.calc_causal_importances(acts, sampling="continuous", role="output")
        hidden = model.calc_causal_importances(acts, sampling="continuous", role="hidden")
        assert (hidden.pre_sigmoid["fc"] >= out.pre_sigmoid["fc"]).all()
        assert (hidden.upper_leaky["fc"] >= out.upper_leaky["fc"]).all()
        assert (hidden.lower_leaky["fc"] >= out.lower_leaky["fc"]).all()

    def test_output_role_is_untouched_by_the_floor(self):
        torch.manual_seed(0)
        acts = {"fc": torch.randn(8, 3)}
        floored = _model(floor=HiddenCIFloorConfig(), dual=True)
        plain = _model(floor=None, dual=True)
        plain.load_state_dict(floored.state_dict())
        torch.testing.assert_close(
            floored.calc_causal_importances(acts, sampling="continuous", role="output").pre_sigmoid[
                "fc"
            ],
            plain.calc_causal_importances(acts, sampling="continuous", role="output").pre_sigmoid[
                "fc"
            ],
        )

    def test_hidden_backward_does_not_reach_the_output_head(self):
        """The floor must not become a path for the hidden objective's sparsity pressure to
        drag the output CI down.

        Scoped to the output *head*, not to `ci_fn` as a whole: under
        `dual_hidden_ci_shared_trunk` most of `ci_fn.parameters()` is the shared trunk, which
        the hidden objective legitimately trains through its own forward. The head is the
        only part that is private to the output role, so it is the only part the floor could
        wrongly leak into. See `test_shared_trunk_floor_leaves_the_output_head_clean`.
        """
        torch.manual_seed(0)
        model = _model(floor=HiddenCIFloorConfig(), dual=True)
        assert model.ci_fn_hidden is not None
        ci = model.calc_causal_importances(
            {"fc": torch.randn(8, 3)}, sampling="continuous", role="hidden"
        )
        ci.lower_leaky["fc"].sum().backward()
        assert any(p.grad is not None for p in model.ci_fn_hidden.parameters())
        assert all(p.grad is None for p in model.ci_fn.parameters())

    def test_shared_trunk_floor_leaves_the_output_head_clean(self):
        """The configuration the floor actually ships with: one trunk, two heads.

        The trunk *does* receive gradient from the hidden objective — that is the point of
        sharing it — so the invariant worth asserting is narrower: nothing reaches the output
        role's private readout head.
        """
        torch.manual_seed(0)
        model = _shared_trunk_model(floor=HiddenCIFloorConfig())
        assert model.ci_fn_hidden is not None
        ci = model.calc_causal_importances(
            {"fc": torch.randn(2, 4, 3)}, sampling="continuous", role="hidden"
        )
        ci.lower_leaky["fc"].sum().backward()
        head = dict(model.ci_fn.named_parameters())
        head_grads = {n: p.grad for n, p in head.items() if "_output_head" in n}
        assert head_grads, "expected a private output head to check"
        assert all(g is None for g in head_grads.values()), (
            f"floor leaked gradient into the output head: {sorted(head_grads)}"
        )
        assert any(p.grad is not None for p in model.ci_fn_hidden.parameters())

    def test_binomial_jitter_does_not_invert_the_order(self):
        """The production config runs `sampling: binomial`, which mixes `-0.05 * rand_like`
        into the lower-leaky branch. Drawn independently per role that noise inverts the
        ordering wherever the logit gap is under 0.0476 — i.e. exactly where the floor binds.
        `calc_causal_importances_both_roles` shares one draw, which is what makes the
        guarantee survive onto the branch that actually masks components.
        """
        torch.manual_seed(0)
        model = _shared_trunk_model(floor=HiddenCIFloorConfig())
        out, hidden = model.calc_causal_importances_both_roles(
            {"fc": torch.randn(4, 8, 3)}, sampling="binomial", detach_inputs=False
        )
        assert (hidden.lower_leaky["fc"] >= out.lower_leaky["fc"]).all()
        assert (hidden.upper_leaky["fc"] >= out.upper_leaky["fc"]).all()

    def test_both_roles_matches_the_single_role_path_without_jitter(self):
        """The joint path is an optimisation plus the shared draw — not different maths."""
        torch.manual_seed(0)
        model = _shared_trunk_model(floor=HiddenCIFloorConfig())
        acts = {"fc": torch.randn(4, 8, 3)}
        out, hidden = model.calc_causal_importances_both_roles(
            acts, sampling="continuous", detach_inputs=False
        )
        roles: list[tuple[CIRole, CIOutputs]] = [("output", out), ("hidden", hidden)]
        for role, joint in roles:
            single = model.calc_causal_importances(acts, sampling="continuous", role=role)
            torch.testing.assert_close(joint.pre_sigmoid["fc"], single.pre_sigmoid["fc"])

    def test_floor_without_a_hidden_net_is_rejected(self):
        with pytest.raises(AssertionError, match="dual_hidden_ci"):
            _model(floor=HiddenCIFloorConfig(), dual=False)


def _ctx(
    model: ComponentModel, ci_out: Tensor, ci_hidden: Tensor, *, is_eval: bool = False
) -> MetricContext:
    """A context whose two CI nets are replaced by explicit values, so the metric is tested
    against a planted violation rather than whatever the untrained nets happen to emit."""

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
        is_eval=is_eval,
    )


class TestShortfallLoss:
    def _metric(self) -> tuple[HiddenCIShortfallLoss, ComponentModel]:
        model = _model(floor=None, dual=True)
        metric = HiddenCIShortfallLoss(HiddenCIShortfallLossConfig(coeff=1.0))
        metric.bind(model=model, device="cpu")
        return metric, model

    def test_counts_only_the_hidden_below_output_direction(self):
        """A hidden CI *above* the output CI is the expected case and must not be penalised."""
        metric, model = self._metric()
        ci_out = torch.zeros(4, 5)
        assert metric.update(_ctx(model, ci_out, torch.full((4, 5), 0.9))).item() == 0.0

    def test_requires_a_hidden_net(self):
        metric = HiddenCIShortfallLoss(HiddenCIShortfallLossConfig(coeff=1.0))
        with pytest.raises(AssertionError, match="dual_hidden_ci"):
            metric.bind(model=_model(floor=None, dual=False), device="cpu")

    def test_value_is_the_summed_shortfall_per_position(self):
        """Normalised like impmin — summed over subcomponents, averaged over positions — so
        its coefficient is on the same scale as the sparsity penalty it has to overcome."""
        metric, model = self._metric()
        ci_out = torch.full((4, 5), 0.5)
        ci_hidden = torch.full((4, 5), 0.2)
        loss = metric.update(_ctx(model, ci_out, ci_hidden))
        assert loss.item() == pytest.approx(0.3 * 5)

    def test_gradient_pushes_hidden_up_and_output_down(self):
        """Both directions, deliberately. A violation is evidence about both nets, and the
        reading worth having is that the output reconstruction should stop leaning on a
        subcomponent the hidden net says does no work — so the penalty must be able to lower
        `ci_out`, not only raise `ci_hidden`."""
        metric, model = self._metric()
        ci_out = torch.full((4, 5), 0.5, requires_grad=True)
        ci_hidden = torch.full((4, 5), 0.2, requires_grad=True)
        metric.update(_ctx(model, ci_out, ci_hidden)).backward()
        assert ci_out.grad is not None and (ci_out.grad > 0).all(), "should push ci_out down"
        assert ci_hidden.grad is not None and (ci_hidden.grad < 0).all(), "should push ci_hidden up"
        # Equal and opposite: the correction is split evenly between the two nets.
        torch.testing.assert_close(ci_out.grad, -ci_hidden.grad)

    def test_no_gradient_where_the_ordering_already_holds(self):
        metric, model = self._metric()
        ci_out = torch.full((4, 5), 0.2, requires_grad=True)
        ci_hidden = torch.full((4, 5), 0.5, requires_grad=True)
        metric.update(_ctx(model, ci_out, ci_hidden)).backward()
        assert ci_out.grad is not None and torch.count_nonzero(ci_out.grad) == 0
        assert ci_hidden.grad is not None and torch.count_nonzero(ci_hidden.grad) == 0

    def test_compute_divides_by_the_pooled_position_and_entry_counts(self):
        """`compute()` carries the only non-obvious arithmetic in this metric: two different
        denominators over accumulators packed into one collective."""
        metric, model = self._metric()
        # 4 positions x 5 components, shortfall 0.3 everywhere, over two eval batches.
        ci_out = torch.full((4, 5), 0.5)
        ci_hidden = torch.full((4, 5), 0.2)
        for _ in range(2):
            metric.update(_ctx(model, ci_out, ci_hidden, is_eval=True))
        out = metric.compute()
        assert isinstance(out, dict)
        # Summed over the 5 components, averaged over positions — batch count divides out.
        assert out["HiddenCIShortfallLoss"].item() == pytest.approx(0.3 * 5)
        assert out["HiddenCIShortfallLoss/fc"].item() == pytest.approx(0.3 * 5)
        # 0.3 clears the 0.1 reporting threshold, so every entry counts as violating.
        assert out["HiddenCIShortfallLoss/violating_frac"].item() == pytest.approx(1.0)

    def test_violating_frac_ignores_shortfalls_under_the_threshold(self):
        metric, model = self._metric()
        ci_out = torch.full((4, 5), 0.15)
        metric.update(_ctx(model, ci_out, torch.full((4, 5), 0.1), is_eval=True))
        out = metric.compute()
        assert isinstance(out, dict)
        assert out["HiddenCIShortfallLoss"].item() == pytest.approx(0.05 * 5, abs=1e-6)
        assert out["HiddenCIShortfallLoss/violating_frac"].item() == 0.0


class TestAllRouting:
    def test_all_routing_routes_every_position(self):
        """`"all"` is a sentinel, not a mask dict — the downstream consumers branch on it to
        skip the per-position `torch.where` and the masked gather entirely."""
        router = get_router(AllRoutingConfig(), device="cpu")
        assert router.get_masks(["a", "b"], (4, 3)) == "all"

    def test_hidden_recon_loss_accepts_all_routing(self):
        """The one config this option exists for. `StochasticHiddenReconSubsetLoss` is the
        only routing-bearing metric with no already-existing route-everywhere sibling."""
        cfg = StochasticHiddenReconSubsetLossConfig(coeff=1.0, routing=AllRoutingConfig())
        metric = StochasticHiddenReconSubsetLoss(cfg)
        metric.bind(model=_model(floor=None, dual=True), device="cpu")
        assert metric.router.get_masks(["fc"], (4, 3)) == "all"
