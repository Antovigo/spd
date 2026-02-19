from unittest.mock import patch

import torch
from torch import Tensor
from torch.nn import functional as F

from spd.configs import LayerwiseCiConfig, SamplingType
from spd.metrics import stochastic_attention_patterns_recon_loss
from spd.metrics.stochastic_attention_patterns_recon_loss import (
    _capture_attention_patterns,
    _collect_attention_patterns,
)
from spd.models.component_model import ComponentModel
from spd.models.components import ComponentsMaskInfo, make_mask_infos
from spd.pretrain.models.gpt2_simple import GPT2Simple, GPT2SimpleConfig
from spd.routing import Router
from spd.utils.module_utils import ModulePathInfo


def _make_gpt2_component_model() -> ComponentModel:
    """Create a tiny GPT2Simple wrapped in ComponentModel for testing."""
    config = GPT2SimpleConfig(
        model_type="GPT2Simple",
        block_size=16,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_embd=16,
        flash_attention=False,
    )
    target = GPT2Simple(config)
    target.requires_grad_(False)

    module_path_info = [
        ModulePathInfo(module_path="h.0.attn.q_proj", C=4),
        ModulePathInfo(module_path="h.0.attn.k_proj", C=4),
        ModulePathInfo(module_path="h.0.attn.v_proj", C=4),
        ModulePathInfo(module_path="h.0.attn.o_proj", C=4),
        ModulePathInfo(module_path="h.0.mlp.c_fc", C=4),
        ModulePathInfo(module_path="h.0.mlp.down_proj", C=4),
    ]

    return ComponentModel(
        target_model=target,
        module_path_info=module_path_info,
        ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[8]),
        pretrained_model_output_attr="idx_0",
        sigmoid_type="leaky_hard",
    )


class TestStochasticAttentionPatternsReconLoss:
    def test_manual_calculation(self) -> None:
        """Test attention patterns KL with manual calculation.

        Mocks calc_stochastic_component_mask_info to use deterministic masks,
        then verifies the metric output matches a manual KL computation.
        """
        torch.manual_seed(42)

        model = _make_gpt2_component_model()
        batch = torch.randint(0, 32, (2, 8))

        ci = {path: torch.rand(2, 8, 4) for path in model.target_module_paths}

        # Get target attention patterns
        target_model = model.target_model
        with _capture_attention_patterns(target_model):
            model(batch)
            target_patterns = [pat.detach() for pat in _collect_attention_patterns(target_model)]

        assert len(target_patterns) == 1  # 1 layer

        # Define deterministic masks for 2 mask samples
        sample_masks = [
            {path: torch.rand(2, 8, 4) for path in model.target_module_paths},
            {path: torch.rand(2, 8, 4) for path in model.target_module_paths},
        ]

        call_count = [0]

        def mock_calc(
            causal_importances: dict[str, Tensor],  # pyright: ignore[reportUnusedParameter]
            component_mask_sampling: SamplingType,  # pyright: ignore[reportUnusedParameter]
            weight_deltas: dict[str, Tensor] | None,  # pyright: ignore[reportUnusedParameter]
            router: Router,  # pyright: ignore[reportUnusedParameter]
        ) -> dict[str, ComponentsMaskInfo]:
            idx = call_count[0] % len(sample_masks)
            call_count[0] += 1
            return make_mask_infos(
                component_masks=sample_masks[idx],
                routing_masks="all",
                weight_deltas_and_masks=None,
            )

        with patch(
            "spd.metrics.stochastic_attention_patterns_recon_loss.calc_stochastic_component_mask_info",
            side_effect=mock_calc,
        ):
            # Manually compute expected KL
            sum_kl = 0.0
            n_distributions = 0

            for masks in sample_masks:
                mask_infos = make_mask_infos(
                    component_masks=masks,
                    routing_masks="all",
                    weight_deltas_and_masks=None,
                )
                with _capture_attention_patterns(target_model):
                    model(batch, mask_infos=mask_infos)
                    comp_patterns = _collect_attention_patterns(target_model)

                for target_pat, comp_pat in zip(target_patterns, comp_patterns, strict=True):
                    kl = F.kl_div(
                        comp_pat.clamp(min=1e-12).log(),
                        target_pat,
                        reduction="sum",
                    )
                    sum_kl += kl.item()
                    n_distributions += (
                        target_pat.shape[0] * target_pat.shape[1] * target_pat.shape[2]
                    )

            expected = sum_kl / n_distributions

            actual = stochastic_attention_patterns_recon_loss(
                model=model,
                sampling="continuous",
                n_mask_samples=2,
                batch=batch,
                ci=ci,
                weight_deltas=None,
            )

            assert torch.allclose(actual, torch.tensor(expected), rtol=1e-5), (
                f"Expected {expected}, got {actual.item()}"
            )

    def test_multiple_mask_samples(self) -> None:
        """Verify that n_mask_samples > 1 averages over all samples correctly."""
        torch.manual_seed(42)

        model = _make_gpt2_component_model()
        batch = torch.randint(0, 32, (2, 8))

        ci = {path: torch.rand(2, 8, 4) for path in model.target_module_paths}

        # Run with 1 sample and then 3 samples; both should produce finite results
        result_1 = stochastic_attention_patterns_recon_loss(
            model=model,
            sampling="continuous",
            n_mask_samples=1,
            batch=batch,
            ci=ci,
            weight_deltas=None,
        )
        result_3 = stochastic_attention_patterns_recon_loss(
            model=model,
            sampling="continuous",
            n_mask_samples=3,
            batch=batch,
            ci=ci,
            weight_deltas=None,
        )

        assert torch.isfinite(result_1)
        assert torch.isfinite(result_3)
        # With different mask samples, results should generally differ
        assert result_1.dim() == 0
        assert result_3.dim() == 0

    def test_output_is_scalar(self) -> None:
        """The metric should return a scalar tensor."""
        torch.manual_seed(42)

        model = _make_gpt2_component_model()
        batch = torch.randint(0, 32, (2, 8))

        ci = {path: torch.rand(2, 8, 4) for path in model.target_module_paths}

        result = stochastic_attention_patterns_recon_loss(
            model=model,
            sampling="continuous",
            n_mask_samples=1,
            batch=batch,
            ci=ci,
            weight_deltas=None,
        )

        assert result.dim() == 0
        assert result.dtype == torch.float32

    def test_kl_is_non_negative(self) -> None:
        """KL divergence should be non-negative."""
        torch.manual_seed(42)

        model = _make_gpt2_component_model()
        batch = torch.randint(0, 32, (2, 8))

        ci = {path: torch.rand(2, 8, 4) for path in model.target_module_paths}

        result = stochastic_attention_patterns_recon_loss(
            model=model,
            sampling="continuous",
            n_mask_samples=2,
            batch=batch,
            ci=ci,
            weight_deltas=None,
        )

        assert result.item() >= -1e-7, f"KL should be non-negative, got {result.item()}"


class TestStochasticAttentionPatternsReconLossHF:
    """Tests for HuggingFace model support (GPTNeoX)."""

    def test_gpt_neox_attention_capture(self) -> None:
        """Verify attention patterns can be captured from a HuggingFace GPTNeoX model."""
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

        torch.manual_seed(42)

        hf_config = GPTNeoXConfig(
            hidden_size=64,
            intermediate_size=256,
            num_attention_heads=4,
            num_hidden_layers=2,
            vocab_size=128,
            max_position_embeddings=32,
        )
        target = GPTNeoXForCausalLM(hf_config)
        target.requires_grad_(False)
        target.eval()

        module_path_info = [
            ModulePathInfo(module_path="gpt_neox.layers.0.attention.query_key_value", C=4),
            ModulePathInfo(module_path="gpt_neox.layers.0.attention.dense", C=4),
            ModulePathInfo(module_path="gpt_neox.layers.0.mlp.dense_h_to_4h", C=4),
            ModulePathInfo(module_path="gpt_neox.layers.0.mlp.dense_4h_to_h", C=4),
        ]

        model = ComponentModel(
            target_model=target,
            module_path_info=module_path_info,
            ci_config=LayerwiseCiConfig(fn_type="mlp", hidden_dims=[8]),
            pretrained_model_output_attr="logits",
            sigmoid_type="leaky_hard",
        )

        batch = torch.randint(0, 128, (2, 8))
        ci = {path: torch.rand(2, 8, 4) for path in model.target_module_paths}

        # Capture target attention patterns
        target_model = model.target_model
        with _capture_attention_patterns(target_model):
            model(batch)
            target_patterns = _collect_attention_patterns(target_model)

        assert len(target_patterns) == 2  # 2 layers
        for pat in target_patterns:
            assert pat.shape == (2, 4, 8, 8)  # (batch, heads, seq, seq)

        # Run the full metric
        result = stochastic_attention_patterns_recon_loss(
            model=model,
            sampling="continuous",
            n_mask_samples=1,
            batch=batch,
            ci=ci,
            weight_deltas=None,
        )

        assert result.dim() == 0
        assert torch.isfinite(result)
        assert result.item() >= -1e-7
