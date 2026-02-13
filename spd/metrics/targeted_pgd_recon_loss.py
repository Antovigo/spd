"""Targeted PGD reconstruction loss for comparing target vs nontarget data."""

from collections.abc import Iterator
from typing import Any, ClassVar, Literal, override

import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp

from spd.configs import PGDConfig, SamplingType
from spd.metrics.base import Metric
from spd.metrics.pgd_utils import pgd_masked_recon_loss_update
from spd.models.component_model import CIOutputs, ComponentModel
from spd.routing import AllLayersRouter
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import extract_batch_data


class TargetedPGDReconLoss(Metric):
    """PGD reconstruction loss on target and nontarget data.

    On target data: force_delta=0.0 (components only, delta ablated).
    On nontarget data: force_delta=1.0 (delta always on).
    """

    metric_section: ClassVar[str] = "targeted_ce_kl"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        output_loss_type: Literal["mse", "kl"],
        pgd_config: PGDConfig,
        use_delta_component: bool,
        sampling: SamplingType,
        nontarget_eval_iterator: Iterator[
            Int[Tensor, "..."] | tuple[Float[Tensor, "..."], Float[Tensor, "..."]]
        ],
    ) -> None:
        self.model = model
        self.device = device
        self.output_loss_type: Literal["mse", "kl"] = output_loss_type
        self.pgd_config: PGDConfig = pgd_config
        self.use_delta_component = use_delta_component
        self.sampling: SamplingType = sampling
        self.nontarget_eval_iterator = nontarget_eval_iterator

        self.target_sum_loss = torch.tensor(0.0, device=device)
        self.target_n_examples = torch.tensor(0, device=device)
        self.n_target_batches = 0
        self.weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None = None

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        target_out: Float[Tensor, "... vocab"],
        ci: CIOutputs,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
        **_: Any,
    ) -> None:
        """Run PGD on target batch with force_delta=0.0."""
        effective_deltas = weight_deltas if self.use_delta_component else None
        sum_loss, n_examples = pgd_masked_recon_loss_update(
            model=self.model,
            batch=batch,
            ci=ci.lower_leaky,
            weight_deltas=effective_deltas,
            target_out=target_out,
            output_loss_type=self.output_loss_type,
            router=AllLayersRouter(),
            pgd_config=self.pgd_config,
            force_delta=0.0,
        )
        self.target_sum_loss += sum_loss
        self.target_n_examples += n_examples
        self.n_target_batches += 1
        # Store weight_deltas for nontarget processing (consistent across batches)
        if self.weight_deltas is None:
            self.weight_deltas = effective_deltas

    @override
    def compute(self) -> dict[str, float]:
        """Process nontarget batches with force_delta=1.0, return target and nontarget losses."""
        # Finalize target loss
        target_sum = all_reduce(self.target_sum_loss, op=ReduceOp.SUM).item()
        target_n = all_reduce(self.target_n_examples, op=ReduceOp.SUM).item()

        # Process nontarget batches (same count as target batches)
        nontarget_sum_loss = torch.tensor(0.0, device=self.device)
        nontarget_n_examples = torch.tensor(0, device=self.device)

        for _ in range(self.n_target_batches):
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

            sum_loss, n_examples = pgd_masked_recon_loss_update(
                model=self.model,
                batch=batch,
                ci=ci.lower_leaky,
                weight_deltas=self.weight_deltas,
                target_out=target_out,
                output_loss_type=self.output_loss_type,
                router=AllLayersRouter(),
                pgd_config=self.pgd_config,
                force_delta=1.0,
            )
            nontarget_sum_loss += sum_loss
            nontarget_n_examples += n_examples

        nontarget_sum = all_reduce(nontarget_sum_loss, op=ReduceOp.SUM).item()
        nontarget_n = all_reduce(nontarget_n_examples, op=ReduceOp.SUM).item()

        return {
            "target/pgd_recon_loss": target_sum / target_n,
            "nontarget/pgd_recon_loss": nontarget_sum / nontarget_n,
        }
