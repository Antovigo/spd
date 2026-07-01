"""Fit the circular ("Fourier") features Llama-3.1-8B uses around layer 18's MLP.

Feucht et al. (2026, https://arxiv.org/pdf/2605.01148v1) show that the model represents each
operand as a circular feature in the activations *entering* L18's MLP, and writes the task
result as a circular feature in what the MLP *adds back* to the residual stream. This script
replicates their probing strategy to recover, for each canonical period, the plane in activation
space that the circle lives in and the circle's center.

A circular feature is the generative fit (`x` = activation vector)

    x ≈ offset + cos(θ)·cos_vec + sin(θ)·sin_vec

solved by least squares. The angle depends on the value `v` and the `--space`:
- **linear** (add/sub): `θ = 2π v / T`, `period` = integer `T`.
- **log** (mult): `θ = 2π·log(v)/log(r)`, `period` = multiplicative ratio `r` — one full turn each
  time `v` scales by `r`. Multiplication is periodic in `log v` (a helix in log space). The default
  ratios are the canonical log-periods the period analysis already found, read from the sibling
  `subcomp_periods_mult.tsv` (≈×1.27 is the dominant one; cross-checked by `find_log_periods`).
  `space` defaults to `log` for mult and `linear` otherwise.

To isolate the probed variable from the nuisance operand, the fit is on the mean activation per
distinct probed value (equal weight per value), matching the paper's probing. `offset` is the
center of the circle; `(cos_vec, sin_vec)` span its plane. `r2` is the fraction of that conditional
mean's variance the circle explains — the diagnostic for whether the feature is really there at
that period (the mean is a sum over several periods, so each single one explains only a fraction).

Sides and their probed variables, matching the paper:
- **input** — the post-RMSNorm activation entering the MLP (`mlp_input`), probed for each
  operand `a` and `b` separately.
- **output** — what the MLP writes to the residual (`mlp_output`), probed for the task result
  (`a+b` / `a-b` / `a×b`).

Consumes the `hidden_activations_<op>.npz` grid from `collect_hidden_activations` (so no GPU /
forward pass here); run that once per task first. Fit separately per task.

Usage:
    python -m param_decomp_lab.scripts.validation.find_fourier_features <hidden_activations_npz> \
        [--periods=2,5,10,...] [--space=linear|log] [--output=PATH]

Output (default `<PARAM_DECOMP_OUT_DIR>/runs/fourier_features/coordinates_<op>.json`): run/op
metadata (incl. `space`) plus `features[side][variable][period] = {period, r2, offset, cos, sin}`,
each vector a `d_model`-long list. For log space `period` holds the ratio `r`.
"""

import json
from pathlib import Path
from typing import Any

import fire
import numpy as np
from numpy.typing import NDArray

from param_decomp.log import logger
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.scripts.validation.common import op_symbol, read_subcomp_period_groups

# Linear space (add/sub): integer periods `T`, θ = 2πv/T.
_CANONICAL_PERIODS = (2, 5, 10, 20, 50, 100)
# Log space (mult): multiplicative ratios `r`, θ = 2π·log_r(v) (one turn each time v scales by r).
# Fallback only — the default log ratios are read from the run's `subcomp_periods_mult.tsv` (the
# frequencies the period analysis already found across mult runs; ≈×1.27 is the dominant one).
_CANONICAL_RATIOS = (1.26,)
# side -> the npz activation grid it probes (input = post-RMSNorm, output = residual write).
_SIDE_GRID = {"input": "mlp_input", "output": "mlp_output"}


def _space_for_op(op: str) -> str:
    """Multiplication is periodic in `log v` (helix in log space); add/sub in `v` itself."""
    return "log" if op == "mult" else "linear"


def _log_ratios_from_periods_tsv(npz_path: Path, op: str) -> tuple[float, ...]:
    """The distinct clustered log-ratios the period analysis found, from the sibling
    `subcomp_periods_<op>.tsv`; falls back to `_CANONICAL_RATIOS` if that file is absent."""
    tsv = npz_path.with_name(f"subcomp_periods_{op}.tsv")
    if not tsv.exists():
        logger.info(f"no {tsv.name} beside the npz; using fallback ratios {_CANONICAL_RATIOS}")
        return _CANONICAL_RATIOS
    ratios = sorted({g.value for g in read_subcomp_period_groups(tsv).values() if g.kind == "log"})
    assert ratios, f"no log-periodic components in {tsv}"
    return tuple(ratios)


def _result(op: str, a: NDArray[np.int64], b: NDArray[np.int64]) -> NDArray[np.int64]:
    match op:
        case "add":
            return a + b
        case "sub":
            return a - b
        case "mult":
            return a * b
        case _:
            raise AssertionError(f"unknown operation {op!r}")


def _conditional_means(
    acts: NDArray[np.float32], values: NDArray[np.int64]
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """`(unique_values, x̄(v))`: the mean activation per distinct probed value (equal weight per
    value) — this isolates the probed variable's signal from the nuisance operand and is
    period-independent, so it's computed once per (side, variable)."""
    unique = np.unique(values)
    means = np.stack([acts[values == v].mean(axis=0) for v in unique]).astype(np.float64)  # [k, d]
    return unique, means


def _fit_circle(
    unique: NDArray[np.int64], means: NDArray[np.float64], period: float, space: str
) -> dict[str, Any]:
    """Least-squares generative fit `x̄(v) ≈ offset + cos(θ)·cos_vec + sin(θ)·sin_vec`.

    The angle `θ` is `2πv/T` in linear space (`period` = integer `T`) or `2π·log(v)/log(r)` in log
    space (`period` = multiplicative ratio `r`, so one turn per `×r`), fit on the conditional means
    (see `_conditional_means`). Returns the three fitted `d`-vectors plus the fraction of the
    conditional mean's variance the circle explains (`r2`).
    """
    if space == "log":
        assert unique.min() > 0, f"log space needs positive values, got min {unique.min()}"
        assert period > 0.0 and period != 1.0, f"log ratio must be positive and != 1, got {period}"
        theta = 2.0 * np.pi * np.log(unique.astype(np.float64)) / np.log(period)
    else:
        theta = 2.0 * np.pi * unique.astype(np.float64) / period
    design = np.stack([np.ones_like(theta), np.cos(theta), np.sin(theta)], axis=1)  # [k, 3]
    coeffs, _, _, _ = np.linalg.lstsq(design, means, rcond=None)  # [3, d]
    offset, cos_vec, sin_vec = coeffs

    pred = design @ coeffs
    ss_res = float(((means - pred) ** 2).sum())
    ss_tot = float(((means - means.mean(axis=0)) ** 2).sum())
    assert ss_tot > 0.0, "conditional means are constant across the probed value; r2 undefined"
    r2 = 1.0 - ss_res / ss_tot

    return {
        "period": period,
        "r2": round(r2, 6),
        "offset": [round(x, 6) for x in offset.tolist()],
        "cos": [round(x, 6) for x in cos_vec.tolist()],
        "sin": [round(x, 6) for x in sin_vec.tolist()],
    }


def find_fourier_features(
    hidden_activations_npz: str,
    periods: tuple[float, ...] | float | None = None,
    space: str | None = None,
    output: str | None = None,
) -> Path:
    npz_path = Path(hidden_activations_npz).expanduser()
    assert npz_path.exists(), f"missing hidden-activations npz: {npz_path}"
    data = np.load(npz_path)
    op = str(data["op"])
    layer = int(data["layer"])

    space = space if space is not None else _space_for_op(op)
    assert space in ("linear", "log"), f"space must be 'linear' or 'log', got {space!r}"
    # In log space `periods` are multiplicative ratios; in linear space, integer periods.
    if periods is None:
        periods = (
            _log_ratios_from_periods_tsv(npz_path, op) if space == "log" else _CANONICAL_PERIODS
        )
    elif not isinstance(periods, (list, tuple)):
        periods = (periods,)  # fire parses a single `--periods=1.26` as a scalar

    a_axis, b_axis = data["a"], data["b"]
    n = a_axis.shape[0]
    assert b_axis.shape[0] == n, "non-square operand grid"
    # grids are [a-1, b-1, d]; flatten with `indexing="ij"` so sample k has a=aa[k], b=bb[k].
    aa, bb = np.meshgrid(a_axis.astype(np.int64), b_axis.astype(np.int64), indexing="ij")
    variable_values = {
        "input": {"a": aa.reshape(-1), "b": bb.reshape(-1)},
        "output": {"result": _result(op, aa, bb).reshape(-1)},
    }

    features: dict[str, dict[str, dict[str, Any]]] = {}
    for side, grid_key in _SIDE_GRID.items():
        grid = data[grid_key]
        assert grid.ndim == 3 and grid.shape[:2] == (n, n), (
            f"unexpected {grid_key} shape {grid.shape}"
        )
        acts = grid.reshape(n * n, grid.shape[-1]).astype(np.float32)
        features[side] = {}
        for variable, values in variable_values[side].items():
            unique, means = _conditional_means(acts, values)  # period-independent
            features[side][variable] = {
                str(period): _fit_circle(unique, means, period, space) for period in periods
            }
            r2s = {p: features[side][variable][str(p)]["r2"] for p in periods}
            logger.info(f"{op} {side}/{variable} ({space}): r2 by period {r2s}")

    payload = {
        "op": op,
        "symbol": op_symbol(op),
        "layer": layer,
        "space": space,
        "source": str(npz_path),
        "n_prompts": int(n * n),
        "grid_size": int(n),
        "periods": list(periods),
        "features": features,
    }

    out_path = (
        Path(output).expanduser()
        if output
        else PARAM_DECOMP_OUT_DIR / "runs" / "fourier_features" / f"coordinates_{op}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"wrote Fourier features for {op} ({size_mb:.1f} MB) → {out_path}")
    return out_path


if __name__ == "__main__":
    fire.Fire(find_fourier_features)
