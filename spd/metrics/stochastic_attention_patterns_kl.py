from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, ClassVar, override

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from torch.distributed import ReduceOp
from torch.nn import functional as F

from spd.configs import SamplingType
from spd.metrics.base import Metric
from spd.models.component_model import CIOutputs, ComponentModel
from spd.routing import AllLayersRouter
from spd.utils.component_utils import calc_stochastic_component_mask_info
from spd.utils.distributed_utils import all_reduce
from spd.utils.general_utils import get_obj_device


def _get_attn_modules(model: nn.Module) -> list[nn.Module]:
    return [m for m in model.modules() if hasattr(m, "store_attention_patterns")]


@contextmanager
def _capture_attention_patterns(model: nn.Module) -> Generator[None]:
    attn_modules = _get_attn_modules(model)
    assert attn_modules, "No attention modules with store_attention_patterns found"
    for m in attn_modules:
        object.__setattr__(m, "store_attention_patterns", True)
    try:
        yield
    finally:
        for m in attn_modules:
            object.__setattr__(m, "store_attention_patterns", False)
            object.__setattr__(m, "_attention_patterns", None)


def _collect_attention_patterns(model: nn.Module) -> list[Tensor]:
    patterns: list[Tensor] = []
    for m in _get_attn_modules(model):
        pat: Tensor | None = getattr(m, "_attention_patterns")  # noqa: B009
        assert pat is not None, "Attention pattern not stored after forward pass"
        patterns.append(pat)
        object.__setattr__(m, "_attention_patterns", None)
    return patterns


def _update(
    model: ComponentModel,
    sampling: SamplingType,
    n_mask_samples: int,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
) -> tuple[Float[Tensor, ""], int]:
    assert ci, "Empty ci"
    device = get_obj_device(ci)
    sum_kl = torch.tensor(0.0, device=device)
    n_distributions = 0

    target_model = model.target_model

    with _capture_attention_patterns(target_model):
        model(batch)
        target_patterns = [pat.detach() for pat in _collect_attention_patterns(target_model)]

    stoch_mask_infos_list = [
        calc_stochastic_component_mask_info(
            causal_importances=ci,
            component_mask_sampling=sampling,
            weight_deltas=weight_deltas,
            router=AllLayersRouter(),
        )
        for _ in range(n_mask_samples)
    ]

    for stoch_mask_infos in stoch_mask_infos_list:
        with _capture_attention_patterns(target_model):
            model(batch, mask_infos=stoch_mask_infos)
            comp_patterns = _collect_attention_patterns(target_model)

        assert len(comp_patterns) == len(target_patterns)
        for target_pat, comp_pat in zip(target_patterns, comp_patterns, strict=True):
            kl = F.kl_div(
                comp_pat.clamp(min=1e-12).log(),
                target_pat,
                reduction="sum",
            )
            sum_kl = sum_kl + kl
            # Each (batch, head, query_position) is one distribution
            n_distributions += target_pat.shape[0] * target_pat.shape[1] * target_pat.shape[2]

    return sum_kl, n_distributions


def _compute(
    sum_kl: Float[Tensor, ""], n_distributions: Int[Tensor, ""] | int
) -> Float[Tensor, ""]:
    return sum_kl / n_distributions


def stochastic_attention_patterns_kl(
    model: ComponentModel,
    sampling: SamplingType,
    n_mask_samples: int,
    batch: Int[Tensor, "..."] | Float[Tensor, "..."],
    ci: dict[str, Float[Tensor, "... C"]],
    weight_deltas: dict[str, Float[Tensor, "d_out d_in"]] | None,
) -> Float[Tensor, ""]:
    sum_kl, n_distributions = _update(
        model,
        sampling,
        n_mask_samples,
        batch,
        ci,
        weight_deltas,
    )
    return _compute(sum_kl, n_distributions)


class StochasticAttentionPatternsKL(Metric):
    """KL divergence between target and stochastic component attention patterns."""

    metric_section: ClassVar[str] = "loss"

    def __init__(
        self,
        model: ComponentModel,
        device: str,
        sampling: SamplingType,
        use_delta_component: bool,
        n_mask_samples: int,
    ) -> None:
        self.model = model
        self.sampling: SamplingType = sampling
        self.use_delta_component = use_delta_component
        self.n_mask_samples = n_mask_samples
        self.sum_kl = torch.tensor(0.0, device=device)
        self.n_distributions = torch.tensor(0, device=device)

    @override
    def update(
        self,
        *,
        batch: Int[Tensor, "..."] | Float[Tensor, "..."],
        ci: CIOutputs,
        weight_deltas: dict[str, Float[Tensor, "d_out d_in"]],
        **_: Any,
    ) -> None:
        sum_kl, n_distributions = _update(
            model=self.model,
            sampling=self.sampling,
            n_mask_samples=self.n_mask_samples,
            batch=batch,
            ci=ci.lower_leaky,
            weight_deltas=weight_deltas if self.use_delta_component else None,
        )
        self.sum_kl += sum_kl
        self.n_distributions += n_distributions

    @override
    def compute(self) -> Float[Tensor, ""]:
        sum_kl = all_reduce(self.sum_kl, op=ReduceOp.SUM)
        n_distributions = all_reduce(self.n_distributions, op=ReduceOp.SUM)
        return _compute(sum_kl, n_distributions)
