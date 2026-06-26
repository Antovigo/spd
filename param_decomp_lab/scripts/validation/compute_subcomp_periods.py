"""Detect each alive subcomponent's activation period along the a and b operand axes.

Reads the per-(component, prompt) inner activations from `collect_inner_activations`
(`inner_activations_<op>.tsv`), forms each subcomponent's `[N, N]` inner-activation grid, and
measures periodicity of the two marginals — `f(a)` (mean over b) and `f(b)` (mean over a) —
two ways:

- **additive** (for add/sub): integer period of the marginal as a function of the operand,
  via autocorrelation (best lag) and FFT (peak frequency → `N/k`).
- **logarithmic** (for mult): the marginal is roughly periodic in `log(operand)` — it repeats
  each time the operand grows by a fixed multiplicative ratio `r`. We fit a sinusoid in
  `log(operand)` over `operand > threshold` (since the periodicity is usually only resolved
  above some value), choosing the period with the most **cross-validated** evidence (fit on
  half the points, score on the held-out half — so a few high-value points lining up by chance
  don't pass) and the lowest threshold that keeps a good held-out fit. Detected ratios are
  clustered (in log-ratio space) so they snap to a handful of canonical periods.

`period_type` (additive / log / none) is decided by comparing the additive and log
cross-validated R² on a common footing; `period` holds the representative additive integer and
`log_period` the representative (clustered) multiplicative ratio.

CPU-only.

Usage:
    python -m param_decomp_lab.scripts.validation.compute_subcomp_periods \
        <inner_activations_tsv> [--log-bar=0.45] [--output=PATH]

Output (default `subcomp_periods_<op>.tsv` beside the input TSV, in `analysis/datasets/`):
one row per subcomponent.
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import fire
import numpy as np
from numpy.typing import NDArray
from scipy.cluster.hierarchy import fcluster, linkage

from param_decomp.log import logger
from param_decomp_lab.scripts.validation.common import read_alive_components

_FIELDS = [
    "layer", "matrix", "component",
    "autocorr_a_period", "autocorr_a_score", "autocorr_b_period", "autocorr_b_score",
    "fft_a_period", "fft_a_score", "fft_b_period", "fft_b_score",
    "period", "period_axis",
    "log_a_ratio", "log_a_thr", "log_a_cvr2", "log_b_ratio", "log_b_thr", "log_b_cvr2",
    "log_period", "log_axis", "log_threshold",
    "additive_cvr2", "log_cvr2", "period_type",
]  # fmt: skip

# Candidate log-periods P (natural-log-operand units; multiplicative ratio = e^P) and the
# lower-cutoff thresholds scanned for where the periodicity becomes resolvable.
_LOG_PERIODS = np.exp(np.linspace(np.log(0.08), np.log(2.5), 60))
_LOG_THRS = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40)


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


def _cv_r2(x: NDArray[np.float64], y: NDArray[np.float64], w: float, nsplit: int = 4) -> float:
    """Held-out R² of the period-`2π/w` sinusoid, averaged over random half-splits."""
    rng = np.random.default_rng(0)
    m = len(y)
    if m < 12:
        return -1.0
    design = np.c_[np.cos(w * x), np.sin(w * x), np.ones(m)]  # built once; splits just slice rows
    scores: list[float] = []
    for _ in range(nsplit):
        idx = rng.permutation(m)
        tr, te = idx[: m // 2], idx[m // 2 :]
        coef, *_ = np.linalg.lstsq(design[tr], y[tr], rcond=None)
        pred = design[te] @ coef
        sst = float(((y[te] - y[te].mean()) ** 2).sum())
        scores.append(1.0 - float(((y[te] - pred) ** 2).sum()) / sst if sst > 1e-12 else -1.0)
    return float(np.mean(scores))


def _detect_log(marginal: NDArray[np.float32], operands: NDArray[np.int64], bar: float):
    """Best log-period for `f(operand)`: `(ratio, threshold, cvR², P)` or `None`.

    Scans (period, threshold) and scores each by held-out R², picking the period with the most
    cross-validated evidence anywhere, then reports the lowest threshold whose held-out fit
    still clears `bar` (the period is genuine over `operand > threshold`).
    """
    x_all = np.log(operands.astype(np.float64))
    y_all = marginal.astype(np.float64)

    def valid(p: float, t: int) -> NDArray[np.bool_] | None:
        mask = operands >= t
        if int(mask.sum()) < 12 or (x_all[mask].max() - x_all[mask].min()) / p < 2:
            return None
        return mask

    best_cv, best_p = -1.0, 0.0
    for p in _LOG_PERIODS:
        w = 2 * np.pi / p
        for t in _LOG_THRS:
            mask = valid(p, t)
            if mask is None:
                continue
            cv = _cv_r2(x_all[mask], y_all[mask], w)
            if cv > best_cv:
                best_cv, best_p = cv, p
    if best_cv < bar:
        return None
    w = 2 * np.pi / best_p
    thr = next(
        (
            t
            for t in _LOG_THRS
            if (m := valid(best_p, t)) is not None and _cv_r2(x_all[m], y_all[m], w) >= bar
        ),
        _LOG_THRS[0],
    )
    return round(float(np.exp(best_p)), 3), thr, round(best_cv, 3), float(best_p)


def _cluster_log_periods(ps: list[float], rel_thr: float = 0.15) -> dict[float, float]:
    """Map each detected log-period P to its cluster's geometric-mean center (in log-P space)."""
    arr = np.array(sorted(set(ps)))
    if arr.size == 0:
        return {}
    if arr.size == 1:
        return {float(arr[0]): float(arr[0])}
    labels = fcluster(
        linkage(np.log(arr).reshape(-1, 1), method="average"), t=rel_thr, criterion="distance"
    )
    centers = {lab: float(np.exp(np.log(arr[labels == lab]).mean())) for lab in set(labels)}
    return {float(p): centers[lab] for p, lab in zip(arr, labels, strict=True)}


def _infer_op(tsv_path: Path) -> str:
    stem = tsv_path.stem  # inner_activations_<op>
    assert stem.startswith("inner_activations_"), f"unexpected TSV name: {tsv_path.name}"
    return stem.removeprefix("inner_activations_")


def compute_subcomp_periods(
    inner_activations_tsv: str, log_bar: float = 0.45, output: str | None = None
) -> Path:
    tsv_path = Path(inner_activations_tsv).expanduser()
    assert tsv_path.exists(), f"missing inner-activations TSV: {tsv_path}"
    op = _infer_op(tsv_path)
    data_dir = tsv_path.parent  # the inner-activations TSV lives in <run>/analysis/datasets/

    alive = read_alive_components(data_dir / f"alive_filtered_{op}.tsv")
    meta = {(a.proj, a.component): (a.layer, a.matrix) for a in alive}

    cells: dict[tuple[str, int], list[tuple[int, int, float]]] = defaultdict(list)
    n = 0
    with tsv_path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            a, b = int(row["a"]), int(row["b"])
            cells[(row["matrix"], int(row["subcomponent"]))].append((a, b, float(row["inner_act"])))
            n = max(n, a, b)
    operands = np.arange(1, n + 1, dtype=np.int64)

    rows: list[dict[str, Any]] = []
    raw_ps: list[float] = []  # every detected log-period P, for clustering
    for (proj, component), entries in cells.items():
        assert (proj, component) in meta, f"{proj}/{component} not in alive_filtered_{op}.tsv"
        layer, matrix = meta[(proj, component)]
        grid = np.zeros((n, n), dtype=np.float32)
        for a, b, val in entries:
            grid[a - 1, b - 1] = val
        marg_a, marg_b = grid.mean(axis=1), grid.mean(axis=0)

        ac_a, fft_a = _autocorr_period(marg_a), _fft_period(marg_a)
        ac_b, fft_b = _autocorr_period(marg_b), _fft_period(marg_b)
        add_period, add_axis = (fft_a[0], "a") if fft_a[1] >= fft_b[1] else (fft_b[0], "b")
        # CV-R² of the additive fit (linear sinusoid at the representative period), to compare
        # against the log evidence on a common footing.
        add_marg = marg_a if add_axis == "a" else marg_b
        additive_cvr2 = (
            round(
                _cv_r2(
                    operands.astype(np.float64), add_marg.astype(np.float64), 2 * np.pi / add_period
                ),
                3,
            )
            if add_period >= 2
            else 0.0
        )

        log_a = _detect_log(marg_a, operands, log_bar)
        log_b = _detect_log(marg_b, operands, log_bar)
        for det in (log_a, log_b):
            if det is not None:
                raw_ps.append(det[3])
        log_axis = ""
        log_cvr2 = 0.0
        if log_a or log_b:
            cva, cvb = (log_a[2] if log_a else -1), (log_b[2] if log_b else -1)
            log_axis, log_cvr2 = ("a", cva) if cva >= cvb else ("b", cvb)

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
                "period": add_period,
                "period_axis": add_axis,
                "log_a_ratio": log_a[0] if log_a else "",
                "log_a_thr": log_a[1] if log_a else "",
                "log_a_cvr2": log_a[2] if log_a else "",
                "log_b_ratio": log_b[0] if log_b else "",
                "log_b_thr": log_b[1] if log_b else "",
                "log_b_cvr2": log_b[2] if log_b else "",
                "additive_cvr2": additive_cvr2,
                "log_cvr2": round(float(log_cvr2), 3),
                "_log_a_P": log_a[3] if log_a else None,
                "_log_b_P": log_b[3] if log_b else None,
                "_log_axis": log_axis,
            }
        )

    # Cluster the detected log-periods so they snap to a handful of canonical ratios.
    canon = _cluster_log_periods(raw_ps)
    for r in rows:
        ax = r.pop("_log_axis")
        rep_p = {"a": r.pop("_log_a_P"), "b": r.pop("_log_b_P"), "": None}[ax]
        r["log_period"] = round(float(np.exp(canon[rep_p])), 3) if rep_p is not None else ""
        r["log_axis"] = ax
        r["log_threshold"] = r[f"log_{ax}_thr"] if ax else ""
        # period_type: whichever fit cross-validates better. Log gets a small margin, since for
        # long periods the linear and log sinusoids are nearly degenerate and the log reading is
        # the meaningful one on mult (additive must clearly beat it to win).
        is_log = r["log_cvr2"] >= log_bar and r["log_cvr2"] + 0.1 >= r["additive_cvr2"]
        is_add = r["additive_cvr2"] >= log_bar and not is_log
        r["period_type"] = "log" if is_log else "additive" if is_add else "none"
    rows.sort(key=lambda r: (r["layer"], r["matrix"], r["component"]))

    out_path = Path(output).expanduser() if output else data_dir / f"subcomp_periods_{op}.tsv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    ntype = {t: sum(r["period_type"] == t for r in rows) for t in ("additive", "log", "none")}
    logger.info(f"{len(rows)} subcomponents (N={n}) → {out_path}  types={ntype}")
    return out_path


if __name__ == "__main__":
    fire.Fire(compute_subcomp_periods)
