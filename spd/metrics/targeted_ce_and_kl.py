"""Targeted CE and KL losses for comparing target vs nontarget data."""

from collections.abc import Iterator
from typing import Any, ClassVar, override

import einops
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import SamplingType
from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.models.components import WeightDeltaAndMask, make_mask_infos
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import calc_kl_divergence_lm, extract_batch_data


class TargetedCEandKL(Metric):
    """CE and KL losses for different masking strategies on target and nontarget data.

    For BOTH target and nontarget data, computes:

    KL Metrics (prefix = `target/` or `nontarget/`):
    - kl_components_only_ci_masked: CI values as mask, force_delta=0.0
    - kl_components_only_rounded_masked: CI > threshold as mask, force_delta=0.0
    - kl_components_only_unmasked: All 1s as mask, force_delta=0.0
    - kl_full_ci_masked: CI values as mask, force_delta=1.0
    - kl_full_rounded_masked: CI > threshold as mask, force_delta=1.0
    - kl_delta_only: All 0s as mask, force_delta=1.0 (delta only)

    CE Metrics:
    - ce_baseline: masks=0, force_delta=0.0 (zeroed model, no delta)
    - ce_component_model: Target data uses force_delta=0.0, nontarget uses force_delta=1.0

    NOTE: Assumes all batches and sequences are the same size.
    """

    metric_section: ClassVar[str] = "targeted_ce_kl"

    loss_keys: list[str] = [
        "kl_components_only_ci_masked",
        "kl_components_only_rounded_masked",
        "kl_components_only_unmasked",
        "kl_full_ci_masked",
        "kl_full_rounded_masked",
        "kl_delta_only",
        "ce_baseline",
        "ce_component_model",
    ]

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        sampling: SamplingType,
        rounding_threshold: float,
        nontarget_eval_iterator: Iterator[
            Int[Tensor, "..."] | tuple[Float[Tensor, "..."], Float[Tensor, "..."]]
        ],
    ) -> None:
        self.model = model
        self.sampling: SamplingType = sampling
        self.rounding_threshold = rounding_threshold
        self.device = device
        self.nontarget_eval_iterator = nontarget_eval_iterator

        # Store target batches for later nontarget processing
        self.target_batches: list[Tensor] = []
        self.target_outs: list[Tensor] = []
        self.target_cis: list[dict[str, Tensor]] = []
        self.target_weight_deltas: list[dict[str, Float[Tensor, "d_out d_in"]]] = []

        self.target_loss_sums: dict[str, Tensor] = {
            key: torch.tensor(0.0, device=device) for key in self.loss_keys
        }
        self.target_n_positions: Int[Tensor, ""] = torch.tensor(0, device=device)

    def _get_mask_infos(
        self,
        component_masks: dict[str, Float[Tensor, "... C"]],
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
        force_delta: float,
    ) -> dict[str, Any]:
        """Create mask infos with specified force_delta value."""
        leading_dims = next(iter(component_masks.values())).shape[:-1]
        weight_deltas_and_masks: dict[str, WeightDeltaAndMask] = {
            layer: (
                weight_deltas[layer],
                torch.full(leading_dims, force_delta, device=self.device),
            )
            for layer in component_masks
        }
        return make_mask_infos(component_masks, weight_deltas_and_masks=weight_deltas_and_masks)

    def _compute_losses(
        self,
        batch: Tensor,
        target_out: Tensor,
        ci: dict[str, Tensor],
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
        is_target: bool,
    ) -> dict[str, float]:
        """Compute all CE/KL losses for a batch."""
        assert batch.ndim == 2, "Batch must be 2D (batch, seq_len)"
        masked_batch = batch.clone()
        masked_batch[:, 0] = -100
        flat_masked_batch = masked_batch.flatten()

        def ce_vs_labels(logits: Tensor) -> float:
            flat_logits = einops.rearrange(logits, "b seq_len vocab -> (b seq_len) vocab")
            return F.cross_entropy(
                flat_logits[:-1], flat_masked_batch[1:], ignore_index=-100
            ).item()

        def kl_vs_target(logits: Tensor) -> float:
            return calc_kl_divergence_lm(pred=logits, target=target_out).item()

        out: dict[str, float] = {}

        # KL: components only (no delta), CI masked
        mask_infos = self._get_mask_infos(ci, weight_deltas, force_delta=0.0)
        logits = self.model(batch, mask_infos=mask_infos)
        out["kl_components_only_ci_masked"] = kl_vs_target(logits)

        # KL: components only (no delta), rounded masked
        rounded_ci = {k: (v > self.rounding_threshold).float() for k, v in ci.items()}
        mask_infos = self._get_mask_infos(rounded_ci, weight_deltas, force_delta=0.0)
        logits = self.model(batch, mask_infos=mask_infos)
        out["kl_components_only_rounded_masked"] = kl_vs_target(logits)

        # KL: components only (no delta), unmasked (all 1s)
        ones_ci = {k: torch.ones_like(v) for k, v in ci.items()}
        mask_infos = self._get_mask_infos(ones_ci, weight_deltas, force_delta=0.0)
        logits = self.model(batch, mask_infos=mask_infos)
        out["kl_components_only_unmasked"] = kl_vs_target(logits)

        # KL: full model (with delta), CI masked
        mask_infos = self._get_mask_infos(ci, weight_deltas, force_delta=1.0)
        logits = self.model(batch, mask_infos=mask_infos)
        out["kl_full_ci_masked"] = kl_vs_target(logits)

        # KL: full model (with delta), rounded masked
        mask_infos = self._get_mask_infos(rounded_ci, weight_deltas, force_delta=1.0)
        logits = self.model(batch, mask_infos=mask_infos)
        out["kl_full_rounded_masked"] = kl_vs_target(logits)

        # KL: delta only (all components zeroed, delta enabled)
        zeros_ci = {k: torch.zeros_like(v) for k, v in ci.items()}
        mask_infos = self._get_mask_infos(zeros_ci, weight_deltas, force_delta=1.0)
        logits = self.model(batch, mask_infos=mask_infos)
        out["kl_delta_only"] = kl_vs_target(logits)

        # CE baseline: all zeroed, no delta
        mask_infos = self._get_mask_infos(zeros_ci, weight_deltas, force_delta=0.0)
        logits = self.model(batch, mask_infos=mask_infos)
        out["ce_baseline"] = ce_vs_labels(logits)

        # CE component model: target uses no delta, nontarget uses delta
        ce_force_delta = 0.0 if is_target else 1.0
        mask_infos = self._get_mask_infos(ci, weight_deltas, force_delta=ce_force_delta)
        logits = self.model(batch, mask_infos=mask_infos)
        out["ce_component_model"] = ce_vs_labels(logits)

        return out

    @override
    def update(
        self,
        *,
        batch: Tensor,
        target_out: Tensor,
        ci: CIOutputs,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
        **_: Any,
    ) -> None:
        """Accumulate target data losses."""
        losses = self._compute_losses(
            batch=batch,
            target_out=target_out,
            ci=ci.lower_leaky,
            weight_deltas=weight_deltas,
            is_target=True,
        )

        assert batch.ndim == 2, "Batch must be 2D (batch, seq_len)"
        n_positions_in_batch = batch.shape[0] * batch.shape[1]

        for key in self.loss_keys:
            self.target_loss_sums[key] += losses[key] * n_positions_in_batch
        self.target_n_positions += n_positions_in_batch

        # Store batch info for nontarget processing in compute()
        self.target_batches.append(batch.detach())
        self.target_outs.append(target_out.detach())
        self.target_cis.append({k: v.detach() for k, v in ci.lower_leaky.items()})
        self.target_weight_deltas.append(weight_deltas)

    @override
    def compute(self) -> dict[str, float]:
        """Compute averaged losses for both target and nontarget data."""
        out: dict[str, float] = {}

        # Finalize target losses
        target_n_positions_reduced = all_reduce(self.target_n_positions, op=ReduceOp.SUM).item()
        for key in self.loss_keys:
            summed_loss = all_reduce(self.target_loss_sums[key], op=ReduceOp.SUM).item()
            out[f"target/{key}"] = summed_loss / target_n_positions_reduced

        # Process nontarget batches
        nontarget_loss_sums: dict[str, Tensor] = {
            key: torch.tensor(0.0, device=self.device) for key in self.loss_keys
        }
        nontarget_n_positions = torch.tensor(0, device=self.device)

        # Reuse the stored weight_deltas (consistent across batches)
        weight_deltas = self.target_weight_deltas[0] if self.target_weight_deltas else {}

        for _ in range(len(self.target_batches)):
            batch_raw = next(self.nontarget_eval_iterator)
            batch = extract_batch_data(batch_raw).to(self.device)

            with torch.no_grad():
                target_output = self.model(batch, cache_type="input")
                target_out = target_output.output
                ci = self.model.calc_causal_importances(
                    pre_weight_acts=target_output.cache,
                    detach_inputs=False,
                    sampling=self.sampling,
                )

            losses = self._compute_losses(
                batch=batch,
                target_out=target_out,
                ci=ci.lower_leaky,
                weight_deltas=weight_deltas,
                is_target=False,
            )

            assert batch.ndim == 2, "Batch must be 2D (batch, seq_len)"
            n_positions_in_batch = batch.shape[0] * batch.shape[1]

            for key in self.loss_keys:
                nontarget_loss_sums[key] += losses[key] * n_positions_in_batch
            nontarget_n_positions += n_positions_in_batch

        # Finalize nontarget losses
        nontarget_n_positions_reduced = all_reduce(nontarget_n_positions, op=ReduceOp.SUM).item()
        for key in self.loss_keys:
            summed_loss = all_reduce(nontarget_loss_sums[key], op=ReduceOp.SUM).item()
            out[f"nontarget/{key}"] = summed_loss / nontarget_n_positions_reduced

        return out
