"""Detect each alive subcomponent's activation period along the a and b operand axes.

Reads the per-(component, prompt) inner activations from `collect_inner_activations`
(`inner_activations_<op>.tsv`), forms each subcomponent's `[N, N]` inner-activation grid,
and measures periodicity of the two marginals — `f(a)` (mean over b) and `f(b)` (mean over
a) — with two independent metrics:
- **autocorrelation** — best lag in `1..N//2`; score = the (unit-`r(0)`) autocorrelation there,
- **Fourier transform** — peak nonzero frequency `k`; period = `round(N/k)`, score = that
  frequency's fraction of total (DC-removed) power.

A single representative `period` / `period_axis` (the FFT axis with the stronger peak) is
also stored for the downstream sorting in the cosine heatmaps and the neuron explorer.

CPU-only — no model loaded.

Usage:
    python -m param_decomp_lab.scripts.validation.compute_subcomp_periods \
        <inner_activations_tsv> [--output=PATH]

Output (default `subcomp_periods_<op>.tsv` in the run folder): one row per subcomponent
with `layer, matrix, component`, the four `{metric}_{axis}_period/score` columns, and the
representative `period` / `period_axis`.
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import fire
import numpy as np
from numpy.typing import NDArray

from param_decomp.log import logger
from param_decomp_lab.scripts.validation.common import read_alive_components

_FIELDS = [
    "layer",
    "matrix",
    "component",
    "autocorr_a_period",
    "autocorr_a_score",
    "autocorr_b_period",
    "autocorr_b_score",
    "fft_a_period",
    "fft_a_score",
    "fft_b_period",
    "fft_b_score",
    "period",
    "period_axis",
]


def _fft_period(marginal: NDArray[np.float32]) -> tuple[int, float]:
    """Peak nonzero-frequency period and its fraction of total DC-removed power."""
    centered = marginal - marginal.mean()
    power = np.abs(np.fft.rfft(centered))[1:] ** 2  # drop DC
    total = float(power.sum())
    if total < 1e-12 or power.size == 0:
        return 0, 0.0
    k = int(np.argmax(power)) + 1
    return int(round(len(marginal) / k)), round(float(power.max()) / total, 4)


def _autocorr_period(marginal: NDArray[np.float32]) -> tuple[int, float]:
    """Best lag in `1..N//2` by unit-`r(0)` autocorrelation, and that autocorrelation."""
    x = marginal - marginal.mean()
    var = float((x**2).sum())
    if var < 1e-12:
        return 0, 0.0
    acf = np.correlate(x, x, mode="full")[len(x) - 1 :] / var  # r(0)=1 at lag 0
    lags = acf[1 : len(x) // 2 + 1]
    if lags.size == 0:
        return 0, 0.0
    best = int(np.argmax(lags))
    return best + 1, round(float(lags[best]), 4)


def _infer_op(tsv_path: Path) -> str:
    stem = tsv_path.stem  # inner_activations_<op>
    assert stem.startswith("inner_activations_"), f"unexpected TSV name: {tsv_path.name}"
    return stem.removeprefix("inner_activations_")


def compute_subcomp_periods(inner_activations_tsv: str, output: str | None = None) -> Path:
    tsv_path = Path(inner_activations_tsv).expanduser()
    assert tsv_path.exists(), f"missing inner-activations TSV: {tsv_path}"
    op = _infer_op(tsv_path)
    run_dir = tsv_path.parent

    # (proj, component) -> full (layer, matrix) from the filtered-alive list.
    alive = read_alive_components(run_dir / f"alive_filtered_{op}.tsv")
    meta = {(a.proj, a.component): (a.layer, a.matrix) for a in alive}

    # Reconstruct each subcomponent's [N, N] inner-act grid indexed [a-1, b-1].
    cells: dict[tuple[str, int], list[tuple[int, int, float]]] = defaultdict(list)
    n = 0
    with tsv_path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            a, b = int(row["a"]), int(row["b"])
            key = (row["matrix"], int(row["subcomponent"]))
            cells[key].append((a, b, float(row["inner_act"])))
            n = max(n, a, b)

    rows: list[dict[str, Any]] = []
    for (proj, component), entries in cells.items():
        assert (proj, component) in meta, f"{proj}/{component} not in alive_filtered_{op}.tsv"
        layer, matrix = meta[(proj, component)]
        grid = np.zeros((n, n), dtype=np.float32)
        for a, b, val in entries:
            grid[a - 1, b - 1] = val
        marg_a = grid.mean(axis=1)  # f(a), averaged over b
        marg_b = grid.mean(axis=0)  # f(b), averaged over a

        ac_a, fft_a = _autocorr_period(marg_a), _fft_period(marg_a)
        ac_b, fft_b = _autocorr_period(marg_b), _fft_period(marg_b)
        # Representative period: the FFT axis with the stronger normalized peak.
        period, axis = (fft_a[0], "a") if fft_a[1] >= fft_b[1] else (fft_b[0], "b")
        rows.append(
            {
                "layer": layer,
                "matrix": matrix,
                "component": component,
                "autocorr_a_period": ac_a[0],
                "autocorr_a_score": ac_a[1],
                "autocorr_b_period": ac_b[0],
                "autocorr_b_score": ac_b[1],
                "fft_a_period": fft_a[0],
                "fft_a_score": fft_a[1],
                "fft_b_period": fft_b[0],
                "fft_b_score": fft_b[1],
                "period": period,
                "period_axis": axis,
            }
        )
    rows.sort(key=lambda r: (r["layer"], r["matrix"], r["component"]))

    out_path = Path(output).expanduser() if output else run_dir / f"subcomp_periods_{op}.tsv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"{len(rows)} subcomponents (N={n}) → {out_path}")
    return out_path


if __name__ == "__main__":
    fire.Fire(compute_subcomp_periods)
