"""Dual CI nets, the early-exit `site_outputs` forward, and the relative site error."""

from typing import override

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from param_decomp.ci_fns import LayerwiseCiConfig
from param_decomp.component_model import ComponentModel
from param_decomp.decomposition_targets import DecompositionTarget
from param_decomp.masks import ComponentsMaskInfo, make_mask_infos
from param_decomp.metrics.context import MetricContext
from param_decomp.metrics.hidden_acts import (
    clean_site_outputs,
    mean_relative_error,
    select_sites,
    site_squared_errors,
)
from param_decomp.metrics.pgd_hidden_acts_recon import (
    PGDHiddenActsReconLoss,
    PGDHiddenActsReconLossConfig,
)
from param_decomp.tests.metrics.fixtures import make_two_layer_component_model
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse, run_batch_passthrough


def _ones_mask_infos(model: ComponentModel) -> dict[str, ComponentsMaskInfo]:
    return make_mask_infos(
        {path: torch.ones(model.module_to_c[path]) for path in model.target_module_paths}
    )


class TestDualCINets:
    def _model(self, *, dual: bool) -> ComponentModel:
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
            decomposition_targets=[DecompositionTarget(module_path="fc", C=2)],
            ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[2]),
            sigmoid_type="leaky_hard",
            dual_hidden_ci=dual,
        )

    def test_single_ci_has_no_hidden_net(self):
        model = self._model(dual=False)
        assert model.ci_fn_hidden is None
        with pytest.raises(AssertionError, match="dual_hidden_ci"):
            model.calc_causal_importances(
                pre_weight_acts={"fc": torch.randn(2, 3)}, sampling="continuous", role="hidden"
            )

    def test_hidden_net_is_separate_and_independently_trainable(self):
        model = self._model(dual=True)
        assert model.ci_fn_hidden is not None
        output_params = {id(p) for p in model.ci_fn.parameters()}
        hidden_params = {id(p) for p in model.ci_fn_hidden.parameters()}
        assert not (output_params & hidden_params), "the two CI nets must not share parameters"

        acts = {"fc": torch.randn(2, 3)}
        ci_hidden = model.calc_causal_importances(
            pre_weight_acts=acts, sampling="continuous", role="hidden"
        )
        ci_hidden.lower_leaky["fc"].sum().backward()
        assert all(p.grad is not None for p in model.ci_fn_hidden.parameters())
        assert all(p.grad is None for p in model.ci_fn.parameters()), (
            "a hidden-role backward must not touch the output net"
        )

    def test_state_dict_carries_the_hidden_net(self):
        keys = self._model(dual=True).state_dict().keys()
        assert any(k.startswith("ci_fn_hidden.") for k in keys)
        assert not any(k.startswith("ci_fn_hidden.") for k in self._model(dual=False).state_dict())


class TestSiteOutputs:
    def test_matches_the_full_forward_cache(self):
        model = make_two_layer_component_model(torch.randn(4, 3), torch.randn(2, 4))
        batch = torch.randn(5, 3)
        mask_infos = _ones_mask_infos(model)

        torch.manual_seed(0)
        full = model(batch, cache_type="output", mask_infos=mask_infos)
        torch.manual_seed(0)
        truncated = model.site_outputs(batch, mask_infos)

        assert truncated.keys() == set(model.target_module_paths)
        for path in model.target_module_paths:
            assert not isinstance(full, Tensor)
            torch.testing.assert_close(truncated[path], full.cache[path])

    def test_stops_before_running_the_tail(self):
        """The forward must abort after the last decomposed site, not merely discard the rest."""
        model = make_two_layer_component_model(torch.randn(4, 3), torch.randn(2, 4))
        # fc2 is the last decomposed site, so nothing after it should execute. Register a
        # hook on the target model's root forward output to detect completion.
        completed = []
        model.target_model.register_forward_hook(lambda *_: completed.append(True))

        model.site_outputs(torch.randn(5, 3), _ones_mask_infos(model))
        assert not completed, "site_outputs ran the model to completion instead of exiting early"

    def test_keeps_the_autograd_graph(self):
        model = make_two_layer_component_model(torch.randn(4, 3), torch.randn(2, 4))
        outputs = model.site_outputs(torch.randn(5, 3), _ones_mask_infos(model))
        outputs["fc1"].sum().backward()
        assert model.components["fc1"].V.grad is not None


class TestRelativeSiteError:
    def test_zero_when_components_reproduce_the_target(self):
        weight1, weight2 = torch.randn(4, 3), torch.randn(2, 4)
        model = make_two_layer_component_model(weight1, weight2)
        batch = torch.randn(6, 3)
        clean = model(batch, cache_type="input")
        assert not isinstance(clean, Tensor)

        # Components exactly reproduce W on both sites, delta absorbing the remainder.
        mask_infos = make_mask_infos(
            {p: torch.ones(model.module_to_c[p]) for p in model.target_module_paths},
            weight_deltas_and_masks={
                p: (delta, torch.ones(batch.shape[0]))
                for p, delta in model.calc_weight_deltas().items()
            },
        )
        targets = clean_site_outputs(model, clean.cache, model.target_module_paths)
        errors = site_squared_errors(model.site_outputs(batch, mask_infos), targets, mask_infos)
        assert mean_relative_error(errors).item() == pytest.approx(0.0, abs=1e-10)

    def test_one_when_the_site_output_is_fully_ablated(self):
        model = make_two_layer_component_model(torch.randn(4, 3), torch.randn(2, 4))
        batch = torch.randn(6, 3)
        clean = model(batch, cache_type="input")
        assert not isinstance(clean, Tensor)
        targets = clean_site_outputs(model, clean.cache, ["fc1"])
        # All components off and no delta: fc1's output is identically zero, so the
        # relative error against its clean output is exactly 1.
        mask_infos = make_mask_infos(
            {p: torch.zeros(model.module_to_c[p]) for p in model.target_module_paths}
        )
        errors = site_squared_errors(model.site_outputs(batch, mask_infos), targets, mask_infos)
        assert mean_relative_error(errors).item() == pytest.approx(1.0, rel=1e-6)

    def test_backward_reaches_upstream_components(self):
        """Downstream site error must grad upstream components — the point of the loss.

        Also the regression test for measuring several chained sites in one backward: an
        in-place fp32 subtraction on a graph tensor corrupted this, and it is invisible
        unless the test actually calls `backward` in fp32 with `routing_mask == "all"`.
        """
        model = make_two_layer_component_model(torch.randn(6, 4), torch.randn(3, 6))
        batch = torch.randn(8, 4)
        clean = model(batch, cache_type="input")
        assert not isinstance(clean, Tensor)
        mask_infos = _ones_mask_infos(model)

        def grad_norms(measured: list[str]) -> tuple[float, float]:
            model.zero_grad(set_to_none=True)
            targets = clean_site_outputs(model, clean.cache, measured)
            errors = site_squared_errors(model.site_outputs(batch, mask_infos), targets, mask_infos)
            mean_relative_error(errors).backward()

            def norm(path: str) -> float:
                grad = model.components[path].V.grad
                return 0.0 if grad is None else grad.norm().item()

            return norm("fc1"), norm("fc2")

        upstream_from_downstream, _ = grad_norms(["fc2"])
        assert upstream_from_downstream > 0, (
            "error at the downstream site must reach the upstream site's components"
        )
        _, downstream_from_upstream = grad_norms(["fc1"])
        assert downstream_from_upstream == 0.0, "gradient must not flow forwards"
        both_upstream, both_downstream = grad_norms(["fc1", "fc2"])
        assert both_upstream > 0 and both_downstream > 0

    def test_targets_match_the_frozen_model(self):
        """`clean_site_outputs` must reproduce what the target model itself computes."""
        weight1, weight2 = torch.randn(4, 3), torch.randn(2, 4)
        model = make_two_layer_component_model(weight1, weight2)
        batch = torch.randn(6, 3)
        clean_in = model(batch, cache_type="input")
        clean_out = model(batch, cache_type="output")
        assert not isinstance(clean_in, Tensor) and not isinstance(clean_out, Tensor)
        targets = clean_site_outputs(model, clean_in.cache, model.target_module_paths)
        for path in model.target_module_paths:
            torch.testing.assert_close(targets[path], clean_out.cache[path])


class TestEvalDoesNotPerturbTraining:
    """The eval loop runs *after* backward and *before* the optimizer step.

    So an eval probe that leaked into `.grad` would silently corrupt the update. The PGD
    probe runs an inner ascent with its own backwards, which makes this worth pinning —
    especially now that it fires on the fast cadence.
    """

    def test_pgd_hidden_probe_leaves_gradients_bitwise_unchanged(self):
        model = make_two_layer_component_model(torch.randn(6, 4), torch.randn(3, 6))
        batch = torch.randn(8, 4)
        clean = model(batch, cache_type="input")
        assert not isinstance(clean, Tensor)
        ci = model.calc_causal_importances(clean.cache, sampling="continuous")

        # Populate .grad the way a training step leaves it just before eval.
        model.site_outputs(batch, _ones_mask_infos(model))["fc2"].sum().backward()
        before = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        assert before, "test needs some populated gradients to be meaningful"

        cfg = PGDHiddenActsReconLossConfig(
            init="random",
            step_size=0.1,
            n_steps=5,
            mask_scope="shared_across_batch",
            ci_role="output",
        )
        metric = PGDHiddenActsReconLoss(cfg)
        metric.bind(model=model, device="cpu")
        with torch.no_grad():  # as the trainer invokes eval
            metric.update(
                MetricContext(
                    model=model,
                    batch=batch,
                    target_out=clean.output,
                    pre_weight_acts=clean.cache,
                    ci=ci,
                    ci_hidden=None,
                    weight_deltas=model.calc_weight_deltas(),
                    step=0,
                    total_steps=1,
                    use_delta_component=True,
                    sampling="continuous",
                    n_mask_samples=1,
                    reconstruction_loss=recon_loss_mse,
                    is_eval=True,
                )
            )
        after = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        assert set(before) == set(after)
        assert all(torch.equal(before[n], after[n]) for n in before), (
            "a PGD eval probe modified training gradients"
        )

    def test_slow_is_overridable_per_instance(self):
        def probe(slow: bool | None) -> PGDHiddenActsReconLoss:
            return PGDHiddenActsReconLoss(
                PGDHiddenActsReconLossConfig(
                    init="random",
                    step_size=0.1,
                    n_steps=5,
                    mask_scope="shared_across_batch",
                    slow=slow,
                )
            )

        assert probe(None).is_slow is False, "fast by default"
        assert probe(True).is_slow is True, "config overrides the class default"
        assert probe(False).is_slow is False


class TestSelectSites:
    def test_none_keeps_every_site(self):
        assert select_sites(["a.mlp.down_proj", "a.self_attn.o_proj"], None) == [
            "a.mlp.down_proj",
            "a.self_attn.o_proj",
        ]

    def test_patterns_filter_and_preserve_order(self):
        sites = ["l18.mlp.gate_proj", "l18.mlp.down_proj", "l18.self_attn.o_proj"]
        assert select_sites(sites, ["*.mlp.down_proj", "*.self_attn.o_proj"]) == [
            "l18.mlp.down_proj",
            "l18.self_attn.o_proj",
        ]

    def test_rejects_patterns_matching_nothing(self):
        with pytest.raises(AssertionError, match="matched none"):
            select_sites(["l18.mlp.gate_proj"], ["*.nonexistent"])
