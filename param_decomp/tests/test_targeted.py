"""Core tests for the targeted-decomposition delta-mask override."""

import pytest
import torch

from param_decomp.masks import AllLayersRouter, calc_stochastic_component_mask_info, make_mask_infos
from param_decomp.metrics.persistent_pgd_state import get_ppgd_mask_infos
from param_decomp.metrics.pgd_utils import (
    PGDConfig,
    _construct_mask_infos_from_adv_sources,
    _init_adv_sources,
    pgd_masked_recon_loss_update,
)
from param_decomp.targeted import delta_override, get_delta_override
from param_decomp.tests.metrics.fixtures import (
    make_one_layer_component_model,
    make_two_layer_component_model,
)


class _PGDTestConfig(PGDConfig):
    type: str = "pgd_test"


def _mse_sum_and_n(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, int]:
    return ((pred - target) ** 2).sum(), pred.shape[0]


class TestDeltaOverride:
    def test_default_is_none(self):
        assert get_delta_override() is None

    def test_value_inside_scope(self):
        with delta_override(0.5):
            assert get_delta_override() == 0.5
        assert get_delta_override() is None

    def test_resets_on_exception(self):
        with pytest.raises(RuntimeError), delta_override(1.0):
            raise RuntimeError("boom")
        assert get_delta_override() is None

    def test_nests(self):
        with delta_override(1.0):
            with delta_override(0.0):
                assert get_delta_override() == 0.0
            assert get_delta_override() == 1.0
        assert get_delta_override() is None


class TestStochasticMaskOverride:
    def _make_inputs(self):
        ci = {"fc": torch.rand(64, 3)}
        weight_deltas = {"fc": torch.randn(4, 5)}
        return ci, weight_deltas

    def test_delta_mask_pinned_under_override(self):
        ci, weight_deltas = self._make_inputs()
        with delta_override(0.7):
            infos = calc_stochastic_component_mask_info(
                ci, "continuous", weight_deltas, AllLayersRouter()
            )
        assert infos["fc"].weight_delta_and_mask is not None
        _, delta_mask = infos["fc"].weight_delta_and_mask
        assert torch.equal(delta_mask, torch.full((64,), 0.7))

    def test_delta_mask_random_without_override(self):
        ci, weight_deltas = self._make_inputs()
        infos = calc_stochastic_component_mask_info(
            ci, "continuous", weight_deltas, AllLayersRouter()
        )
        assert infos["fc"].weight_delta_and_mask is not None
        _, delta_mask = infos["fc"].weight_delta_and_mask
        assert delta_mask.shape == (64,)
        assert (delta_mask >= 0).all() and (delta_mask < 1).all()
        assert delta_mask.unique().numel() > 1

    def test_component_masks_unaffected(self):
        ci, weight_deltas = self._make_inputs()
        with delta_override(1.0):
            infos = calc_stochastic_component_mask_info(
                ci, "continuous", weight_deltas, AllLayersRouter()
            )
        mask = infos["fc"].component_mask
        assert mask.shape == (64, 3)
        # mask = ci + (1 - ci) * source >= ci, and is not constant
        assert (mask >= ci["fc"] - 1e-6).all()
        assert mask.unique().numel() > 1

    def test_noop_when_no_weight_deltas(self):
        ci, _ = self._make_inputs()
        with delta_override(1.0):
            infos = calc_stochastic_component_mask_info(ci, "continuous", None, AllLayersRouter())
        assert infos["fc"].weight_delta_and_mask is None


class TestPGDOverride:
    def _pgd_config(self) -> _PGDTestConfig:
        return _PGDTestConfig(
            init="random", step_size=0.1, n_steps=1, mask_scope="unique_per_datapoint"
        )

    def test_mask_c_drops_delta_slot_under_override(self):
        model = make_one_layer_component_model(torch.randn(4, 5), C=3)
        weight_deltas = model.calc_weight_deltas()
        sources = _init_adv_sources(model, (8,), "cpu", weight_deltas, self._pgd_config())
        assert sources["fc"].shape == (8, 4)  # C + 1 without override
        with delta_override(1.0):
            sources = _init_adv_sources(model, (8,), "cpu", weight_deltas, self._pgd_config())
        assert sources["fc"].shape == (8, 3)  # all channels are components

    def test_construction_pins_delta(self):
        ci = {"fc": torch.rand(8, 3)}
        weight_deltas = {"fc": torch.randn(4, 5)}
        adv_sources = {"fc": torch.rand(8, 3)}
        with delta_override(1.0):
            infos = _construct_mask_infos_from_adv_sources(
                adv_sources, ci, weight_deltas, "all", (8,)
            )
        assert infos["fc"].weight_delta_and_mask is not None
        _, delta_mask = infos["fc"].weight_delta_and_mask
        assert torch.equal(delta_mask, torch.ones(8))
        assert infos["fc"].component_mask.shape == (8, 3)

    def test_pgd_recon_forward_runs_under_override(self):
        model = make_one_layer_component_model(torch.randn(4, 5), C=3)
        batch = torch.randn(8, 5)
        target_out = model.target_model(batch)
        ci = {"fc": torch.rand(8, 3)}
        weight_deltas = model.calc_weight_deltas()
        with delta_override(1.0):
            sum_loss, n = pgd_masked_recon_loss_update(
                model=model,
                batch=batch,
                ci=ci,
                weight_deltas=weight_deltas,
                target_out=target_out,
                router=AllLayersRouter(),
                pgd_config=self._pgd_config(),
                reconstruction_loss=_mse_sum_and_n,
            )
        assert get_delta_override() is None
        assert torch.isfinite(sum_loss)
        assert n == 8

    def test_ppgd_raises_under_override(self):
        ci = {"fc": torch.rand(8, 3)}
        weight_deltas = {"fc": torch.randn(4, 5)}
        ppgd_sources = {"fc": torch.rand(1, 4)}
        with pytest.raises(AssertionError, match="PPGD"), delta_override(1.0):
            get_ppgd_mask_infos(ci, weight_deltas, ppgd_sources, "all", (8,))


class TestFaithfulnessInvariant:
    def test_full_masks_reproduce_target_output(self):
        """Component masks = 1 + delta mask = 1 ⇒ exact target output."""
        torch.manual_seed(0)
        model = make_two_layer_component_model(torch.randn(4, 5), torch.randn(3, 4))
        batch = torch.randn(16, 5)
        weight_deltas = model.calc_weight_deltas()
        mask_infos = make_mask_infos(
            component_masks={"fc1": torch.ones(16, 1), "fc2": torch.ones(16, 1)},
            weight_deltas_and_masks={
                name: (weight_deltas[name], torch.ones(16)) for name in ("fc1", "fc2")
            },
        )
        out = model(batch, mask_infos=mask_infos)
        target_out = model.target_model(batch)
        torch.testing.assert_close(out, target_out, atol=1e-5, rtol=1e-5)
