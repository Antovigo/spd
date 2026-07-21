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
from param_decomp_lab.period_orbits import (
    CANONICAL_PERIODS,
    class_bin_powers,
    count_periods,
    period_class_shares,
)


class PeriodSeparationConfig(BaseConfig):
    """`prompts_file` must be a full `a+b=` grid (e.g. `addition_1-100.txt`) so the
    2D FFT sees every cell; the run's own (mixed-op) prompt pool is not suitable."""

    type: Literal["PeriodSeparation"] = "PeriodSeparation"
    prompts_file: str
    tokenizer_name: str
    ci_gate: NonNegativeFloat = 0.1
    snr_thr: NonNegativeFloat = 20.0
    n_null_dirs: PositiveInt = 256
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
    snr: dict[int, float]  # class power / random-direction null median


class PeriodSeparation(Metric[PeriodSeparationConfig]):
    """Period mixing of the answer-position inner activations `x·V_c / ‖V_c‖`.

    Only subcomponents with mean CI > `ci_gate` over the grid prompts are scored. Each
    scored grid's 2D spectrum is decomposed into canonical period classes; a class is
    *present* when its bin power is `snr_thr`× the median of the same quantity over
    `n_null_dirs` fixed random unit read-directions through the same module input.
    That null is the point: the input is itself periodic, so a spectral-floor test
    fires on everything while a share-relative test misses small genuine periods —
    "above what a random direction would read" is the meaningful zero. (`theta` only
    labels the share columns.)
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

    def _answer_position_ci_and_inner(self) -> dict[str, tuple[Tensor, Tensor, Tensor]]:
        """Per matched module: `(mean CI [C], inner activations [n_prompts, C],
        null inner activations [n_prompts, n_null_dirs])`, all at the answer (last)
        position. Inner activations are V-column-normalised; the null columns are fixed
        random unit directions through the same module input — the presence null."""
        assert self.sampling is not None, "haven't seen an eval batch yet"
        ci_chunks: dict[str, list[Tensor]] = {}
        inner_chunks: dict[str, list[Tensor]] = {}
        null_chunks: dict[str, list[Tensor]] = {}
        null_dirs: dict[str, Tensor] = {}
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
                    x_pos = pre_weight_acts[module_name][:, -1, :]
                    inner = components.get_component_acts(x_pos)
                    inner = inner.float() / components.V.norm(dim=0).float().clamp_min(1e-8)
                    inner_chunks.setdefault(module_name, []).append(inner.cpu())
                    if module_name not in null_dirs:
                        gen = torch.Generator(device="cpu").manual_seed(0)
                        dirs = torch.randn(x_pos.shape[-1], self.cfg.n_null_dirs, generator=gen)
                        null_dirs[module_name] = (dirs / dirs.norm(dim=0)).to(
                            device=x_pos.device, dtype=x_pos.dtype
                        )
                    null = (x_pos @ null_dirs[module_name]).float()
                    null_chunks.setdefault(module_name, []).append(null.cpu())
        return {
            name: (
                torch.cat(ci_chunks[name]).mean(dim=0),
                torch.cat(inner_chunks[name]),
                torch.cat(null_chunks[name]),
            )
            for name in ci_chunks
        }

    def scored_components(self) -> list[ComponentPeriods]:
        scored = []
        for module_name, (mean_ci, inner, null) in self._answer_position_ci_and_inner().items():
            null_powers: dict[int, list[float]] = {}
            for j in range(null.shape[1]):
                grid = torch.zeros(self.n, self.n)
                grid[self.b_idx, self.a_idx] = null[:, j]
                for period, p in class_bin_powers(grid.numpy()).items():
                    null_powers.setdefault(period, []).append(p)
            null_median = {
                period: max(float(np.median(powers)), 1e-30)
                for period, powers in null_powers.items()
            }
            for comp in torch.nonzero(mean_ci > self.cfg.ci_gate).flatten().tolist():
                grid = torch.zeros(self.n, self.n)
                grid[self.b_idx, self.a_idx] = inner[:, comp]
                grid_np = grid.numpy()
                powers = class_bin_powers(grid_np)
                scored.append(
                    ComponentPeriods(
                        module=module_name,
                        component=comp,
                        mean_ci=float(mean_ci[comp]),
                        grid=grid_np,
                        shares=period_class_shares(grid_np),
                        snr={p: powers[p] / null_median[p] for p in powers},
                    )
                )
        return scored

    def plot_panel(self, scored: list[ComponentPeriods]) -> "plt.Figure":
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
                    (period for period, v in s.snr.items() if v >= self.cfg.snr_thr),
                    reverse=True,
                )
                label = ",".join(str(period) for period in present) if present else "-"
                ax.set_title(f"{s.component} [{label}]", fontsize=5)
            axes[row][0].set_ylabel(module.rsplit(".", 1)[-1], fontsize=6)
        fig.suptitle(
            f"answer-position inner activations (x·V/‖V‖) over a+b grid; "
            f"[periods with SNR ≥ {self.cfg.snr_thr:g}]",
            fontsize=7,
        )
        fig.tight_layout()
        return fig

    @override
    def compute(self) -> MetricResult:
        scored = self.scored_components()
        result: MetricResult = {"n_active": float(len(scored))}
        if not scored:
            return result
        n_periods = [count_periods(s.snr, self.cfg.snr_thr, CANONICAL_PERIODS) for s in scored]
        periodic = [n for n in n_periods if n >= 1]
        result["periodic_frac"] = len(periodic) / len(scored)
        if periodic:
            result["mixed_frac"] = sum(1 for n in periodic if n >= 2) / len(periodic)
            result["excess_periods"] = sum(n - 1 for n in periodic) / len(periodic)
        # Absolute detection is threshold-dependent too — log side-λ views for
        # robustness, plus the θ-free share of the second-strongest class.
        for side_thr in (10.0, 100.0):
            nps = [count_periods(s.snr, side_thr, CANONICAL_PERIODS) for s in scored]
            per = [n for n in nps if n >= 1]
            if per:
                result[f"mixed_frac_snr{int(side_thr)}"] = sum(1 for n in per if n >= 2) / len(per)
        result["secondary_share"] = float(
            sum(sorted(s.shares.values())[-2] for s in scored) / len(scored)
        )
        # Roadmap criteria: every canonical period present in every matrix (coverage),
        # with as few components per (matrix, period) cell as possible (redundancy).
        # The per-matrix census covers ALL classes (T=4/25/100 as diagnostics).
        matrices = sorted({s.module for s in scored})
        present_cells = 0
        presences = 0
        for module in matrices:
            comps = [s for s in scored if s.module == module]
            for period in CANONICAL_PERIODS:
                count = sum(1 for s in comps if s.snr[period] >= self.cfg.snr_thr)
                if count > 0:
                    present_cells += 1
                    presences += count
        result["coverage_frac"] = present_cells / (len(matrices) * len(CANONICAL_PERIODS))
        if present_cells:
            result["components_per_period"] = presences / present_cells
        # A component is PURE-T when T is its only detected canonical period and it has
        # no T=4/25 extras (genuine additional periodicities); T=100 detections are
        # ignored (trend-confounded). n_pure_periods = canonical periods owning at
        # least one pure component — the headline separation score (target: 5).
        pure_census: dict[int, int] = {}
        for s_comp in scored:
            detected = {p for p, v in s_comp.snr.items() if v >= self.cfg.snr_thr and p != 100}
            if len(detected) == 1 and (period := next(iter(detected))) in CANONICAL_PERIODS:
                pure_census[period] = pure_census.get(period, 0) + 1
        result["n_pure_periods"] = float(len(pure_census))
        for period in sorted(pure_census):
            result[f"pure_census/T{period}"] = float(pure_census[period])
        for module in matrices:
            comps = [s for s in scored if s.module == module]
            short = module.rsplit(".", 1)[-1]
            for period in sorted(comps[0].snr):
                count = sum(1 for s in comps if s.snr[period] >= self.cfg.snr_thr)
                if count > 0:
                    result[f"census/{short}/T{period}"] = float(count)
        fig = self.plot_panel(scored)
        result["inner_acts_heatmap"] = _render_figure(fig)
        plt.close(fig)
        return result
