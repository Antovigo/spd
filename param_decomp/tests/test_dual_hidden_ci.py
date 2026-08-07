"""Dual CI nets, the early-exit `site_outputs` forward, and the relative site error."""

from typing import override

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from param_decomp.ci_fns import (
    AttnConfig,
    GlobalCiConfig,
    GlobalSharedTransformerCiConfig,
    LayerwiseCiConfig,
    share_transformer_trunk,
)
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
from param_decomp.metrics.persistent_pgd_recon import (
    PersistentPGDHiddenActsReconLoss,
    PersistentPGDHiddenActsReconLossConfig,
    PersistentPGDReconLoss,
    PersistentPGDReconLossConfig,
)
from param_decomp.metrics.persistent_pgd_state import AdamPGDConfig, PerBatchPerPositionScope
from param_decomp.metrics.pgd_hidden_acts_recon import (
    PGDHiddenActsReconLoss,
    PGDHiddenActsReconLossConfig,
)
from param_decomp.schedule import ScheduleConfig
from param_decomp.tests.metrics.fixtures import make_two_layer_component_model
from param_decomp_lab.batch_and_loss_fns import recon_loss_mse, run_batch_passthrough
from param_decomp_lab.component_model_io import _validate_checkpoint_trunk_sharing


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


def _transformer_dual_model(*, shared_trunk: bool) -> ComponentModel:
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
        ci_config=GlobalCiConfig(
            fn_type="global_shared_transformer",
            simple_transformer_ci_cfg=GlobalSharedTransformerCiConfig(
                d_model=8,
                n_blocks=2,
                mlp_hidden_dim=[16],
                attn_config=AttnConfig(n_heads=2, max_len=4),
            ),
        ),
        sigmoid_type="leaky_hard",
        dual_hidden_ci=True,
        dual_hidden_ci_shared_trunk=shared_trunk,
    )


class TestSharedTrunk:
    """One trunk, one readout head per role — `pd.dual_hidden_ci_shared_trunk`."""

    def _heads_and_trunk(self, model: ComponentModel) -> tuple[set[int], set[int], set[int]]:
        assert model.ci_fn_hidden is not None
        output_head = {id(p) for n, p in model.ci_fn.named_parameters() if "_output_head" in n}
        hidden_head = {
            id(p) for n, p in model.ci_fn_hidden.named_parameters() if "_output_head" in n
        }
        trunk = {id(p) for n, p in model.ci_fn.named_parameters() if "_output_head" not in n}
        return output_head, hidden_head, trunk

    def test_trunk_is_one_set_of_parameters_and_heads_are_private(self):
        model = _transformer_dual_model(shared_trunk=True)
        assert model.ci_fn_hidden is not None
        output_head, hidden_head, trunk = self._heads_and_trunk(model)
        hidden_trunk = {
            id(p) for n, p in model.ci_fn_hidden.named_parameters() if "_output_head" not in n
        }
        assert trunk == hidden_trunk, "the two nets must reach the very same trunk parameters"
        assert not (output_head & hidden_head), "readout heads must stay private per role"
        assert trunk, "test is vacuous without trunk parameters"

    def test_independent_nets_share_nothing(self):
        model = _transformer_dual_model(shared_trunk=False)
        assert model.ci_fn_hidden is not None
        assert not (
            {id(p) for p in model.ci_fn.parameters()}
            & {id(p) for p in model.ci_fn_hidden.parameters()}
        )

    def test_optimizer_sees_each_parameter_once(self):
        shared = _transformer_dual_model(shared_trunk=True)
        named = shared.ci_fn_named_parameters()
        assert len({id(p) for _, p in named}) == len(named), "a shared parameter was counted twice"
        assert any(n.startswith("ci_fn_hidden.") for n, _ in named), "hidden head must be trainable"

        independent = _transformer_dual_model(shared_trunk=False)
        _, _, trunk = self._heads_and_trunk(shared)
        assert len(independent.ci_fn_named_parameters()) == len(named) + len(trunk)

    def test_both_objectives_reach_the_trunk_but_only_their_own_head(self):
        model = _transformer_dual_model(shared_trunk=True)
        assert model.ci_fn_hidden is not None
        acts = {"fc": torch.randn(2, 4, 3)}
        # Zero-init readouts make the trunk gradient vanish at step 0, so perturb first.
        for _, param in model.ci_fn_named_parameters():
            with torch.no_grad():
                param.add_(torch.randn_like(param) * 0.1)

        model.calc_causal_importances(
            pre_weight_acts=acts, sampling="continuous", role="hidden"
        ).lower_leaky["fc"].sum().backward()

        def grads(module: nn.Module, head: bool) -> list[Tensor | None]:
            return [p.grad for n, p in module.named_parameters() if ("_output_head" in n) is head]

        assert all(g is not None and g.abs().sum() > 0 for g in grads(model.ci_fn, head=False)), (
            "the hidden objective must train the shared trunk"
        )
        assert all(g is not None for g in grads(model.ci_fn_hidden, head=True)), (
            "the hidden objective must train its own head"
        )
        assert all(g is None for g in grads(model.ci_fn, head=True)), (
            "the hidden objective must not touch the output net's head"
        )

    def test_state_dict_keys_match_independent_nets(self):
        """A shared trunk must stay checkpoint-key-compatible with an independent pair."""
        shared = _transformer_dual_model(shared_trunk=True).state_dict()
        independent = _transformer_dual_model(shared_trunk=False).state_dict()
        assert shared.keys() == independent.keys()

    def test_loading_an_independent_checkpoint_under_the_flag_is_refused(self):
        """Key-identical state dicts make this the only thing standing between a
        mismatched flag and the output net silently running the hidden net's trunk."""
        _validate_checkpoint_trunk_sharing(
            _transformer_dual_model(shared_trunk=True).state_dict(), shared_trunk=True
        )
        with pytest.raises(AssertionError, match="different trunk weights"):
            _validate_checkpoint_trunk_sharing(
                _transformer_dual_model(shared_trunk=False).state_dict(), shared_trunk=True
            )

    def test_layerwise_ci_fns_have_no_trunk_to_share(self):
        layerwise = TestDualCINets()._model(dual=True)
        assert layerwise.ci_fn_hidden is not None
        with pytest.raises(AssertionError, match="global CI fns"):
            share_transformer_trunk(source=layerwise.ci_fn, target=layerwise.ci_fn_hidden)


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


class TestPersistentPGDPerObjectiveSources:
    """Each reconstruction objective must own its own persistent adversary."""

    def _ctx(self, model: ComponentModel, batch: Tensor) -> MetricContext:
        clean = model(batch, cache_type="input")
        assert not isinstance(clean, Tensor)
        return MetricContext(
            model=model,
            batch=batch,
            target_out=clean.output,
            pre_weight_acts=clean.cache,
            ci=model.calc_causal_importances(clean.cache, sampling="continuous"),
            ci_hidden=model.calc_causal_importances(clean.cache, sampling="continuous"),
            weight_deltas=model.calc_weight_deltas(),
            step=0,
            total_steps=10,
            use_delta_component=True,
            sampling="continuous",
            n_mask_samples=1,
            reconstruction_loss=recon_loss_mse,
            is_eval=False,
        )

    def test_sources_are_independent_and_separately_checkpointed(self):
        model = make_two_layer_component_model(torch.randn(6, 4), torch.randn(3, 6))
        optimizer = AdamPGDConfig(lr_schedule=ScheduleConfig(fn_type="constant", start_val=0.01))
        out_loss = PersistentPGDReconLoss(
            PersistentPGDReconLossConfig(
                coeff=0.5, optimizer=optimizer, scope=PerBatchPerPositionScope(), n_warmup_steps=1
            )
        )
        hid_loss = PersistentPGDHiddenActsReconLoss(
            PersistentPGDHiddenActsReconLossConfig(
                coeff=0.5, optimizer=optimizer, scope=PerBatchPerPositionScope(), n_warmup_steps=1
            )
        )
        for m in (out_loss, hid_loss):
            m.bind(model=model, device="cpu")

        assert out_loss.instance_key != hid_loss.instance_key, "state would collide in the snapshot"
        ctx = self._ctx(model, torch.randn(8, 4))
        out_loss.update(ctx)
        hid_loss.update(ctx)

        assert out_loss.state is not None and hid_loss.state is not None
        for name in model.target_module_paths:
            a, b = out_loss.state.sources[name], hid_loss.state.sources[name]
            assert a is not b, "the two objectives are sharing a source tensor"
            assert not torch.equal(a, b), "sources should diverge once each adversary has stepped"

        # Distinct instance keys mean the trainer's snapshot stores them separately.
        snapshot = {m.instance_key: m.state_dict() for m in (out_loss, hid_loss)}
        assert len(snapshot) == 2
        assert not torch.equal(
            snapshot[out_loss.instance_key]["sources"]["fc1"],
            snapshot[hid_loss.instance_key]["sources"]["fc1"],
        )

    def test_hidden_variant_attacks_the_hidden_net_by_default(self):
        assert (
            PersistentPGDHiddenActsReconLossConfig(
                coeff=0.5,
                optimizer=AdamPGDConfig(
                    lr_schedule=ScheduleConfig(fn_type="constant", start_val=0.01)
                ),
                scope=PerBatchPerPositionScope(),
            ).ci_role
            == "hidden"
        )


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
