"""Bounded smooth-L0 (Geman–McClure) importance-minimality penalty on CI values.

A drop-in alternative to the `L_p` `ImportanceMinimalityLoss`: the per-CI-value penalty is
`φ(c) = c² / (c² + γ²)` instead of `(c + eps)^p`. It is flat at 0 (`φ'(0)=0`) and has a
bounded gradient (`~0.65/γ` near the threshold `c≈γ`), so there is no gradient cliff at the
accumulation point where most components live (the `L_p` gradient `p·c^(p-1)` blows up as
`c→0` for `p<1`). The per-component sum saturates to ≈ the active-component count. Self-
contained (the entropy term and DDP-`world_size` handling mirror the `L_p` variant) so it adds
no coupling to `importance_minimality.py`.
"""

from typing import Literal, override

import torch
from jaxtyping import Float
from pydantic import NonNegativeFloat, PositiveFloat, model_validator
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.base_config import Probability
from param_decomp.ci_fns import CIRole
from param_decomp.distributed import all_reduce, get_distributed_state
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext


class SmoothL0ImportanceMinimalityLossConfig(LossMetricConfig):
    """Config for the bounded smooth-L0 (Geman–McClure) importance-minimality penalty.

    `gamma` is the initial threshold `γ` in `φ(c) = c² / (c² + γ²)`; `beta` weights the same
    entropy-like `mean * log2(1 + sum)` term used by the `L_p` variant, now over `φ`. `gamma`
    is linearly annealed toward `gamma_final` between `gamma_anneal_start_frac` and
    `gamma_anneal_end_frac` of training (no-op when `gamma_final is None` or
    `gamma_anneal_start_frac == 1.0`).

    The training loss is scaled by a coefficient multiplier with a warmup-then-anneal
    schedule (`_get_coeff_multiplier`), mirroring the `L_p` variant: ramp
    `0 -> coeff_peak_multiplier` over `coeff_warmup_frac`, hold at the peak, then ramp
    `coeff_peak_multiplier -> 1.0` between `coeff_anneal_start_frac` and
    `coeff_anneal_end_frac`. Defaults are a no-op (constant multiplier of `1.0`). The
    multiplier scales the live loss only; the value logged by `compute()` (the sparsity
    proxy) is unaffected.

    `ci_role` picks which CI net is penalised; a dual-CI run lists this loss twice, once per
    net, with distinct `name`s.
    """

    type: Literal["SmoothL0ImportanceMinimalityLoss"] = "SmoothL0ImportanceMinimalityLoss"
    ci_role: CIRole = "output"
    gamma: PositiveFloat
    beta: NonNegativeFloat
    gamma_anneal_start_frac: Probability = 1.0
    gamma_final: PositiveFloat | None = None
    gamma_anneal_end_frac: Probability = 1.0
    normalize_at_one: bool = False
    """Rescale `φ` by `(1 + γ²)` so a fully-active component (`c = 1`) always contributes
    exactly 1, removing the implicit ~2x coefficient ramp across the γ anneal."""
    coeff_warmup_frac: Probability = 0.0
    coeff_peak_multiplier: NonNegativeFloat = 1.0
    coeff_anneal_start_frac: Probability = 1.0
    coeff_anneal_end_frac: Probability = 1.0

    @model_validator(mode="after")
    def validate_scheduling_fracs(self) -> "SmoothL0ImportanceMinimalityLossConfig":
        assert self.coeff_warmup_frac <= self.coeff_anneal_start_frac, (
            f"coeff_warmup_frac ({self.coeff_warmup_frac}) must be <= "
            f"coeff_anneal_start_frac ({self.coeff_anneal_start_frac})"
        )
        assert self.coeff_anneal_end_frac >= self.coeff_anneal_start_frac, (
            f"coeff_anneal_end_frac ({self.coeff_anneal_end_frac}) must be >= "
            f"coeff_anneal_start_frac ({self.coeff_anneal_start_frac})"
        )
        assert self.gamma_anneal_end_frac >= self.gamma_anneal_start_frac, (
            f"gamma_anneal_end_frac ({self.gamma_anneal_end_frac}) must be >= "
            f"gamma_anneal_start_frac ({self.gamma_anneal_start_frac})"
        )
        return self


def _get_coeff_multiplier(
    current_frac_of_training: float,
    coeff_warmup_frac: float,
    coeff_peak_multiplier: float,
    coeff_anneal_start_frac: float,
    coeff_anneal_end_frac: float,
) -> float:
    """Coefficient multiplier with warmup then anneal-to-1.0.

    - `[0, coeff_warmup_frac)`: linearly ramp `0 -> coeff_peak_multiplier`
    - `[coeff_warmup_frac, coeff_anneal_start_frac)`: constant `coeff_peak_multiplier`
    - `[coeff_anneal_start_frac, coeff_anneal_end_frac)`: linearly ramp `coeff_peak_multiplier -> 1.0`
    - `[coeff_anneal_end_frac, 1.0]`: constant `1.0`
    """
    if current_frac_of_training < coeff_warmup_frac:
        return coeff_peak_multiplier * current_frac_of_training / coeff_warmup_frac

    if current_frac_of_training < coeff_anneal_start_frac:
        return coeff_peak_multiplier

    if current_frac_of_training >= coeff_anneal_end_frac:
        return 1.0

    progress = (current_frac_of_training - coeff_anneal_start_frac) / (
        coeff_anneal_end_frac - coeff_anneal_start_frac
    )
    return coeff_peak_multiplier + (1.0 - coeff_peak_multiplier) * progress


def _get_linear_annealed_gamma(
    current_frac_of_training: float,
    initial_gamma: float,
    gamma_anneal_start_frac: float,
    gamma_final: float | None,
    gamma_anneal_end_frac: float,
) -> float:
    if gamma_final is None or gamma_anneal_start_frac >= 1.0:
        return initial_gamma
    assert gamma_anneal_end_frac >= gamma_anneal_start_frac, (
        f"gamma_anneal_end_frac ({gamma_anneal_end_frac}) must be >= "
        f"gamma_anneal_start_frac ({gamma_anneal_start_frac})"
    )
    if current_frac_of_training < gamma_anneal_start_frac:
        return initial_gamma
    elif current_frac_of_training >= gamma_anneal_end_frac:
        return gamma_final
    progress = (current_frac_of_training - gamma_anneal_start_frac) / (
        gamma_anneal_end_frac - gamma_anneal_start_frac
    )
    return initial_gamma + (gamma_final - initial_gamma) * progress


def _per_component_sums(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    gamma: float,
    normalize_at_one: bool,
) -> tuple[dict[str, Float[Tensor, " C"]], int]:
    assert ci_upper_leaky, "Empty ci_upper_leaky"
    assert gamma > 0.0, f"gamma must be positive, got {gamma}"
    gamma_sq = gamma * gamma
    scale = 1.0 + gamma_sq if normalize_at_one else 1.0
    out: dict[str, Float[Tensor, " C"]] = {}
    for layer_name, layer_ci in ci_upper_leaky.items():
        ci_sq = layer_ci * layer_ci
        phi = ci_sq * scale / (ci_sq + gamma_sq)
        out[layer_name] = phi.sum(dim=tuple(range(phi.dim() - 1)))
    n_examples = next(iter(ci_upper_leaky.values())).shape[:-1].numel()
    return out, n_examples


def _smooth_l0_and_entropy_terms(
    per_component_sums: dict[str, Float[Tensor, " C"]],
    n_examples: int,
    world_size: int,
) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
    """The two additive parts of the loss, summed over components: `(smooth_l0, entropy)`.

    Full loss is `smooth_l0 + beta * entropy`; `smooth_l0` alone (≈ the active-component
    count) is the beta-independent sparsity proxy.
    """
    device = next(iter(per_component_sums.values())).device
    smooth_l0 = torch.zeros((), device=device)
    entropy = torch.zeros((), device=device)
    for layer_sums in per_component_sums.values():
        per_component_mean = layer_sums / n_examples
        smooth_l0 = smooth_l0 + per_component_mean.sum()
        entropy = entropy + (per_component_mean * torch.log2(1 + layer_sums * world_size)).sum()
    return smooth_l0, entropy


def _finalize(
    per_component_sums: dict[str, Float[Tensor, " C"]],
    n_examples: int,
    beta: float,
    world_size: int,
) -> Float[Tensor, ""]:
    smooth_l0, entropy = _smooth_l0_and_entropy_terms(per_component_sums, n_examples, world_size)
    return smooth_l0 + beta * entropy


def smooth_l0_importance_minimality_loss(
    ci_upper_leaky: dict[str, Float[Tensor, "... C"]],
    current_frac_of_training: float,
    gamma: float,
    beta: float,
    gamma_anneal_start_frac: float,
    gamma_final: float | None,
    gamma_anneal_end_frac: float,
    normalize_at_one: bool = False,
    coeff_warmup_frac: float = 0.0,
    coeff_peak_multiplier: float = 1.0,
    coeff_anneal_start_frac: float = 1.0,
    coeff_anneal_end_frac: float = 1.0,
) -> Float[Tensor, ""]:
    """Compute the smooth-L0 importance-minimality loss directly (helper for external callers)."""
    annealed_gamma = _get_linear_annealed_gamma(
        current_frac_of_training=current_frac_of_training,
        initial_gamma=gamma,
        gamma_anneal_start_frac=gamma_anneal_start_frac,
        gamma_final=gamma_final,
        gamma_anneal_end_frac=gamma_anneal_end_frac,
    )
    per_component_sums, n_examples = _per_component_sums(
        ci_upper_leaky=ci_upper_leaky, gamma=annealed_gamma, normalize_at_one=normalize_at_one
    )
    dist_state = get_distributed_state()
    world_size = dist_state.world_size if dist_state is not None else 1
    loss = _finalize(
        per_component_sums=per_component_sums,
        n_examples=n_examples,
        beta=beta,
        world_size=world_size,
    )
    coeff_multiplier = _get_coeff_multiplier(
        current_frac_of_training=current_frac_of_training,
        coeff_warmup_frac=coeff_warmup_frac,
        coeff_peak_multiplier=coeff_peak_multiplier,
        coeff_anneal_start_frac=coeff_anneal_start_frac,
        coeff_anneal_end_frac=coeff_anneal_end_frac,
    )
    return loss * coeff_multiplier


class SmoothL0ImportanceMinimalityLoss(Metric[SmoothL0ImportanceMinimalityLossConfig]):
    """Bounded smooth-L0 (Geman–McClure) penalty driving CI sparsity.

    `c² / (c² + γ²)` summed across components plus a `beta`-weighted
    `mean * log2(1 + sum)` term.
    """

    log_namespace = "loss"
    short_name = "SmoothL0ImpMin"

    @override
    def reset(self) -> None:
        self.per_component_sums: dict[str, Float[Tensor, " C"]] = {}
        self.n_examples = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        gamma = _get_linear_annealed_gamma(
            current_frac_of_training=ctx.current_frac_of_training,
            initial_gamma=self.cfg.gamma,
            gamma_anneal_start_frac=self.cfg.gamma_anneal_start_frac,
            gamma_final=self.cfg.gamma_final,
            gamma_anneal_end_frac=self.cfg.gamma_anneal_end_frac,
        )
        per_component_sums, n = _per_component_sums(
            ci_upper_leaky=ctx.ci_for(self.cfg.ci_role).upper_leaky,
            gamma=gamma,
            normalize_at_one=self.cfg.normalize_at_one,
        )
        for layer_name, layer_sums in per_component_sums.items():
            if layer_name not in self.per_component_sums:
                self.per_component_sums[layer_name] = torch.zeros_like(layer_sums)
            self.per_component_sums[layer_name] += layer_sums.detach()
        self.n_examples += n

        dist_state = get_distributed_state()
        world_size = dist_state.world_size if dist_state is not None else 1
        loss = _finalize(
            per_component_sums=per_component_sums,
            n_examples=n,
            beta=self.cfg.beta,
            world_size=world_size,
        )
        coeff_multiplier = _get_coeff_multiplier(
            current_frac_of_training=ctx.current_frac_of_training,
            coeff_warmup_frac=self.cfg.coeff_warmup_frac,
            coeff_peak_multiplier=self.cfg.coeff_peak_multiplier,
            coeff_anneal_start_frac=self.cfg.coeff_anneal_start_frac,
            coeff_anneal_end_frac=self.cfg.coeff_anneal_end_frac,
        )
        return loss * coeff_multiplier

    @override
    def compute(self) -> MetricResult:
        reduced_sums = {
            k: all_reduce(v, op=ReduceOp.SUM) for k, v in self.per_component_sums.items()
        }
        n_examples = int(all_reduce(self.n_examples, op=ReduceOp.SUM))
        smooth_l0, entropy = _smooth_l0_and_entropy_terms(reduced_sums, n_examples, world_size=1)
        name = self.instance_key
        return {
            name: smooth_l0 + self.cfg.beta * entropy,
            f"{name}_no_beta": smooth_l0,
        }
