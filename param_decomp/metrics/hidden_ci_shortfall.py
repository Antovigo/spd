"""How far the hidden CI net falls below the output CI net, as a penalty and a diagnostic."""

from typing import Literal, override

import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.distributed import all_reduce
from param_decomp.metrics.base import LossMetricConfig, Metric, MetricResult
from param_decomp.metrics.context import MetricContext


class HiddenCIShortfallLossConfig(LossMetricConfig):
    """Config for the hidden-below-output CI penalty.

    No `ci_role`: this metric is about the relationship between the two nets, so it always
    reads both.
    """

    type: Literal["HiddenCIShortfallLoss"] = "HiddenCIShortfallLoss"
    violation_threshold: float = 0.1
    """Shortfall above which an entry counts toward the logged violating fraction. Purely a
    reporting cut — the penalty itself is linear from zero and ignores it."""


def _shortfall(
    ci_out: dict[str, Float[Tensor, "... C"]],
    ci_hidden: dict[str, Float[Tensor, "... C"]],
) -> dict[str, Float[Tensor, "... C"]]:
    """Elementwise `relu(CI_out - CI_hidden)`, per module.

    `ci_out` is detached here rather than by the caller: every use of this shortfall — loss
    and diagnostic alike — wants the constraint to act on the hidden net only, never to pull
    the output CI down to meet it.
    """
    assert ci_out.keys() == ci_hidden.keys(), (
        f"the two CI nets disagree on modules: {sorted(ci_out)} vs {sorted(ci_hidden)}"
    )
    return {
        name: (ci_out[name].detach() - hidden).clamp(min=0) for name, hidden in ci_hidden.items()
    }


class HiddenCIShortfallLoss(Metric[HiddenCIShortfallLossConfig]):
    """`relu(CI_output - CI_hidden)`, summed over subcomponents and averaged over positions.

    A subcomponent influences the model's output only through the output of the matrix it
    lives in, which is exactly what the hidden objective measures — so anything the output
    net considers important should be at least as important to the hidden net, while the
    converse may fail freely for a component whose contribution is cancelled downstream.
    This penalises the direction that should not happen.

    Three choices worth their reasons:

    - **Normalised like the importance-minimality losses** (sum over components, mean over
      positions, summed over sites) rather than as a plain mean. Violations occupy a
      vanishing fraction of entries — on the `addsub-L18-11` runs, well under 1% — so a mean
      over all entries would divide the signal by the ~6000 subcomponents that are not
      violating and leave a per-entry gradient orders of magnitude below the sparsity
      pressure it has to overcome. Sharing impmin's normalisation makes the two
      coefficients directly comparable instead.
    - **Linear, not squared.** The gradient must not fade out on the small violations a
      constraint is meant to remove, and the linear form's value is the shortfall itself.
    - **Measured on `lower_leaky`** — the branch that actually masks — whose backward leaks
      below zero exactly when the gradient is negative, which is the direction this penalty
      pushes, so it can revive a hidden CI that has saturated at zero. `upper_leaky` is flat
      there and could not. The raw logit would be wrong outright: a logit gap between two
      values that both clamp to 1 is not a violation.

    Also registered as an eval metric, so runs that do not penalise the shortfall can still
    log it — the unconstrained baseline, and `pd.hidden_ci_floor` runs where it should read
    ~0 and is a self-check on the floor.
    """

    log_namespace = "loss"
    short_name = "HiddenCIShortfall"

    @override
    def reset(self) -> None:
        self._shortfall_sums: dict[str, Tensor] = {}
        self._violating: dict[str, Tensor] = {}
        self._n_positions = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        shortfall = _shortfall(ctx.ci_for("output").lower_leaky, ctx.ci_for("hidden").lower_leaky)
        n_positions = next(iter(shortfall.values())).shape[:-1].numel()
        per_component_sums = {
            name: values.sum(dim=tuple(range(values.dim() - 1)))
            for name, values in shortfall.items()
        }

        if ctx.is_eval:  # `compute()` is eval-only, and each eval pass `reset()`s first
            for name, sums in per_component_sums.items():
                if name not in self._shortfall_sums:
                    self._shortfall_sums[name] = torch.zeros_like(sums)
                    self._violating[name] = torch.zeros((), device=self.device)
                self._shortfall_sums[name] += sums.detach()
                self._violating[name] += (
                    shortfall[name].detach() > self.cfg.violation_threshold
                ).sum()
            self._n_positions += n_positions

        return torch.stack([sums.sum() for sums in per_component_sums.values()]).sum() / n_positions

    @override
    def compute(self) -> MetricResult:
        assert self._shortfall_sums, "no batches accumulated"
        sites = sorted(self._shortfall_sums)  # sorted: every rank issues the same collectives
        # Numerators and the position count reduce separately and divide only afterwards, so
        # each result is a ratio over all ranks' data, not an average of per-rank ratios.
        n_positions = all_reduce(self._n_positions, op=ReduceOp.SUM)
        stacked = torch.stack(
            [
                torch.stack([self._shortfall_sums[site].sum(), self._violating[site]])
                for site in sites
            ]
        )  # [n_sites, 2]
        reduced = all_reduce(stacked, op=ReduceOp.SUM)
        n_components = torch.tensor(
            [self._shortfall_sums[site].numel() for site in sites], device=self.device
        )
        per_site_summed = reduced[:, 0] / n_positions
        per_site_violating = reduced[:, 1] / (n_positions * n_components)

        name = self.instance_key
        out: dict[str, Float[Tensor, ""]] = {
            name: per_site_summed.sum(),
            f"{name}/violating_frac": per_site_violating.mean(),
        }
        for i, site in enumerate(sites):
            out[f"{name}/{site}"] = per_site_summed[i]
        return out
