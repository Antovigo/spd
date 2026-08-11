"""How far the hidden CI net falls below the output CI net, as a penalty and a diagnostic."""

from typing import Literal, override

import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributed import ReduceOp

from param_decomp.component_model import ComponentModel
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
    exactly 0 and is a self-check on the floor. That self-check is only meaningful because
    the two roles share one binomial jitter draw
    (`ComponentModel.calc_causal_importances_both_roles`); with independent draws this metric
    reads the noise floor rather than the violation, which on the L18 shape is `6144 * 0.0083
    ≈ 51` — two orders above the real signal.
    """

    log_namespace = "loss"
    short_name = "HiddenCIShortfall"

    @override
    def bind(self, *, model: ComponentModel, device: str) -> None:
        assert model.ci_fn_hidden is not None, (
            "HiddenCIShortfallLoss compares the two CI nets; set pd.dual_hidden_ci"
        )
        super().bind(model=model, device=device)

    @override
    def reset(self) -> None:
        # Sorted so every rank builds the same layout and issues identically ordered
        # collectives; fixed at reset, so `compute()` needs no agreement protocol.
        self._sites = sorted(self.model.target_module_paths)
        self._n_components = torch.tensor(
            [self.model.module_to_c[site] for site in self._sites], device=self.device
        )
        # fp32 regardless of the CI dtype: under autocast these are bf16 sums of ~1e-5 terms
        # over millions of near-zero entries, which bf16 accumulation would quantise away.
        self._shortfall_sum = torch.zeros(len(self._sites), device=self.device)
        self._violating = torch.zeros(len(self._sites), device=self.device, dtype=torch.long)
        self._n_positions = torch.zeros((), device=self.device, dtype=torch.long)

    @override
    def update(self, ctx: MetricContext) -> Tensor:
        shortfall = _shortfall(ctx.ci_for("output").lower_leaky, ctx.ci_for("hidden").lower_leaky)
        n_positions = next(iter(shortfall.values())).shape[:-1].numel()
        per_site = torch.stack([shortfall[site].sum() for site in self._sites])

        if ctx.is_eval:  # `compute()` is eval-only, and each eval pass `reset()`s first
            self._shortfall_sum += per_site.detach().float()
            self._violating += torch.stack(
                [
                    (shortfall[site].detach() > self.cfg.violation_threshold).sum()
                    for site in self._sites
                ]
            )
            self._n_positions += n_positions

        return per_site.sum() / n_positions

    @override
    def compute(self) -> MetricResult:
        assert self._n_positions > 0, "no batches accumulated"
        # One collective over a fresh tensor: `all_reduce` is in place, and reducing the
        # accumulators directly would double-count them on a second `compute()`.
        packed = torch.cat(
            [
                self._shortfall_sum,
                self._violating.float(),
                self._n_positions.float().reshape(1),
            ]
        )
        reduced = all_reduce(packed, op=ReduceOp.SUM)
        n = len(self._sites)
        # Sums and the position count reduce together and divide only afterwards, so each
        # result is a ratio over all ranks' data, not an average of per-rank ratios.
        n_positions = reduced[-1]
        per_site_summed = reduced[:n] / n_positions
        # Pooled over entries rather than an unweighted mean of per-site rates: sites carry
        # different subcomponent counts (1024 vs 512 on the L18 shape), so a mean of rates
        # would over-weight the small ones and disagree with the summed headline above it.
        violating_frac = reduced[n : 2 * n].sum() / (n_positions * self._n_components.sum())

        name = self.instance_key
        out: dict[str, Float[Tensor, ""]] = {
            name: per_site_summed.sum(),
            f"{name}/violating_frac": violating_frac,
        }
        for i, site in enumerate(self._sites):
            out[f"{name}/{site}"] = per_site_summed[i]
        return out
