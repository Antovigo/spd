"""Training-time period-mixing score from inner activations over the (a, b) addition grid."""

from dataclasses import dataclass
from typing import Literal, override

import matplotlib.pyplot as plt
import numpy as np
import torch
from pydantic import NonNegativeFloat, PositiveInt
from torch import Tensor
from transformers import AutoTokenizer

from param_decomp.base_config import BaseConfig, Probability
from param_decomp.masks import SamplingType
from param_decomp.metrics.base import Metric, MetricResult
from param_decomp.metrics.context import MetricContext
from param_decomp_lab.eval_metrics.plotting import _render_figure
from param_decomp_lab.experiments.lm.prompts_dataset import load_prompts_dataset
from param_decomp_lab.period_orbits import count_periods, period_class_shares


class PeriodSeparationConfig(BaseConfig):
    """`prompts_file` must be a full `a+b=` grid (e.g. `addition_1-100.txt`) so the
    2D FFT sees every cell; the run's own (mixed-op) prompt pool is not suitable."""

    type: Literal["PeriodSeparation"] = "PeriodSeparation"
    prompts_file: str
    tokenizer_name: str
    ci_gate: NonNegativeFloat = 0.1
    theta: Probability = 0.2
    module_grep: str = "mlp"
    batch_size: PositiveInt = 512
    top_k_plot: PositiveInt = 20


@dataclass(frozen=True)
class ComponentPeriods:
    """Period-class shares of one CI-gated subcomponent's inner-activation grid."""

    module: str
    component: int
    mean_ci: float
    grid: np.ndarray
    shares: dict[int, float]


class PeriodSeparation(Metric[PeriodSeparationConfig]):
    """Period mixing of the answer-position inner activations `x·V_c / ‖V_c‖`.

    Only subcomponents with mean CI > `ci_gate` over the grid prompts are scored. Each
    scored grid's 2D spectrum is decomposed into canonical period classes
    (`period_orbits.period_class_shares`); a class is *present* when its share ≥ `theta`.
    `n_periods` = number of present classes: 0 = aperiodic (fine), 1 = clean single
    period (fine, even when used on both operands — the blob grid), ≥ 2 = mixing —
    the one failure mode being measured.

    Logged: `n_active` (gated count), `periodic_frac` (share of gated components with
    ≥ 1 present class), `mixed_frac` (share of *periodic* components with ≥ 2 —
    aperiodic components do not affect it), `excess_periods` (mean `n_periods − 1` over
    periodic components), a per-period component census, and an AB-heatmap-style figure
    of the inner-activation grids (top `top_k_plot` components per matrix by mean CI).

    Self-probing: eval batches are ignored except to learn the run's mask-sampling type;
    values are deterministic given the weights, so every rank computes the same numbers.
    """

    log_namespace = "period_separation"
    slow = True

    def __init__(self, cfg: PeriodSeparationConfig) -> None:
        super().__init__(cfg)
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name)
        self.prompt_ids = load_prompts_dataset(cfg.prompts_file, tokenizer)
        with open(cfg.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
        operands = []
        for prompt in prompts:
            body, _, _ = prompt.partition("=")
            a_str, _, b_str = body.partition("+")
            operands.append((int(a_str), int(b_str)))
        self.a_idx = torch.tensor([a - 1 for a, _ in operands])
        self.b_idx = torch.tensor([b - 1 for _, b in operands])
        self.n = int(max(max(a, b) for a, b in operands))
        assert len(operands) == self.n * self.n, (
            f"prompts_file must cover the full {self.n}×{self.n} grid, got {len(operands)} prompts"
        )

    @override
    def reset(self) -> None:
        self.sampling: SamplingType | None = None

    @override
    def update(self, ctx: MetricContext) -> None:
        if self.sampling is None:
            self.sampling = ctx.sampling
        return None

    def _answer_position_ci_and_inner(self) -> dict[str, tuple[Tensor, Tensor]]:
        """Per matched module: `(mean CI [C], inner activations [n_prompts, C])`, both at
        the answer (last) position; inner activations are V-column-normalised."""
        assert self.sampling is not None, "haven't seen an eval batch yet"
        ci_chunks: dict[str, list[Tensor]] = {}
        inner_chunks: dict[str, list[Tensor]] = {}
        with torch.no_grad():
            for start in range(0, self.prompt_ids.shape[0], self.cfg.batch_size):
                batch = self.prompt_ids[start : start + self.cfg.batch_size].to(self.device)
                pre_weight_acts = self.model(batch, cache_type="input").cache
                ci = self.model.calc_causal_importances(
                    pre_weight_acts=pre_weight_acts,
                    detach_inputs=True,
                    sampling=self.sampling,
                )
                for module_name, ci_vals in ci.lower_leaky.items():
                    if self.cfg.module_grep not in module_name:
                        continue
                    assert ci_vals.ndim == 3, f"expected [batch, seq, C], got {ci_vals.shape}"
                    ci_chunks.setdefault(module_name, []).append(ci_vals[:, -1, :].float().cpu())
                    components = self.model.components[module_name]
                    inner = components.get_component_acts(pre_weight_acts[module_name][:, -1, :])
                    inner = inner.float() / components.V.norm(dim=0).float().clamp_min(1e-8)
                    inner_chunks.setdefault(module_name, []).append(inner.cpu())
        return {
            name: (torch.cat(ci_chunks[name]).mean(dim=0), torch.cat(inner_chunks[name]))
            for name in ci_chunks
        }

    def _scored_components(self) -> list[ComponentPeriods]:
        scored = []
        for module_name, (mean_ci, inner) in self._answer_position_ci_and_inner().items():
            for comp in torch.nonzero(mean_ci > self.cfg.ci_gate).flatten().tolist():
                grid = torch.zeros(self.n, self.n)
                grid[self.b_idx, self.a_idx] = inner[:, comp]
                grid_np = grid.numpy()
                scored.append(
                    ComponentPeriods(
                        module=module_name,
                        component=comp,
                        mean_ci=float(mean_ci[comp]),
                        grid=grid_np,
                        shares=period_class_shares(grid_np),
                    )
                )
        return scored

    def _plot(self, scored: list[ComponentPeriods]) -> "plt.Figure":
        """AB-heatmap-style panel: matrices down the rows, top-`top_k_plot` (by mean CI)
        inner-activation grids across the columns."""
        modules = sorted({s.module for s in scored})
        by_module = {
            m: sorted((s for s in scored if s.module == m), key=lambda s: s.mean_ci, reverse=True)[
                : self.cfg.top_k_plot
            ]
            for m in modules
        }
        n_cols = max(len(v) for v in by_module.values())
        fig, axes = plt.subplots(
            len(modules),
            n_cols,
            figsize=(1.1 * n_cols, 1.35 * len(modules)),
            squeeze=False,
        )
        for row, module in enumerate(modules):
            for col in range(n_cols):
                ax = axes[row][col]
                ax.set_xticks([])
                ax.set_yticks([])
                if col >= len(by_module[module]):
                    ax.axis("off")
                    continue
                s = by_module[module][col]
                vmax = float(np.abs(s.grid).max()) or 1.0
                ax.imshow(s.grid, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
                present = sorted(
                    (period for period, sh in s.shares.items() if sh >= self.cfg.theta),
                    reverse=True,
                )
                label = ",".join(str(period) for period in present) if present else "-"
                ax.set_title(f"{s.component} [{label}]", fontsize=5)
            axes[row][0].set_ylabel(module.rsplit(".", 1)[-1], fontsize=6)
        fig.suptitle(
            f"answer-position inner activations (x·V/‖V‖) over a+b grid; "
            f"[periods with share ≥ {self.cfg.theta}]",
            fontsize=7,
        )
        fig.tight_layout()
        return fig

    @override
    def compute(self) -> MetricResult:
        scored = self._scored_components()
        result: MetricResult = {"n_active": float(len(scored))}
        if not scored:
            return result
        n_periods = [count_periods(s.shares, self.cfg.theta) for s in scored]
        periodic = [n for n in n_periods if n >= 1]
        result["periodic_frac"] = len(periodic) / len(scored)
        if periodic:
            result["mixed_frac"] = sum(1 for n in periodic if n >= 2) / len(periodic)
            result["excess_periods"] = sum(n - 1 for n in periodic) / len(periodic)
        # The θ cut is sharp in practice (many components hold a secondary period at
        # 10-25% power), so log the thresholded view at fixed side θs plus a θ-free
        # mixing intensity: the mean share of the second-strongest period class.
        for side_theta in (0.1, 0.3):
            nps = [count_periods(s.shares, side_theta) for s in scored]
            per = [n for n in nps if n >= 1]
            if per:
                tag = f"t{int(side_theta * 100):02d}"
                result[f"mixed_frac_{tag}"] = sum(1 for n in per if n >= 2) / len(per)
        result["secondary_share"] = float(
            sum(sorted(s.shares.values())[-2] for s in scored) / len(scored)
        )
        census: dict[int, int] = {}
        for comp in scored:
            for period, share in comp.shares.items():
                if share >= self.cfg.theta:
                    census[period] = census.get(period, 0) + 1
        for period in sorted(census):
            result[f"census/T{period}"] = float(census[period])
        fig = self._plot(scored)
        result["inner_acts_heatmap"] = _render_figure(fig)
        plt.close(fig)
        return result
