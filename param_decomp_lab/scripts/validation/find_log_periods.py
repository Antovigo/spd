"""Find the log-space periods of the circular features around L18's MLP (multiplication).

For multiplication the model is hypothesised to represent an operand `v` as a *circular* feature
in activation space whose **phase advances with `log v`** (so `v → v·r` rotates it by a fixed
angle). This script finds those circles and reads off their periods **without scanning any
frequency grid**:

1. `x̄(v)` — the activation averaged over the nuisance operand, one vector per value `v`.
2. Remove DC + the linear-in-`log v` trend (the number-magnitude direction), leaving oscillation.
3. SVD of `x̄(v)` over `v`: a circular feature is a near-degenerate singular-value **pair** whose
   two score patterns are a `cos`/`sin` of the same phase. Each consecutive pair `(2k, 2k+1)` is a
   candidate plane.
4. Project onto the plane → trajectory `(s1(v), s2(v))`; take the **signed angle increment**
   between consecutive values (robust, no global unwrapping) divided by `Δ log v` → angular
   velocity `ω`. The **log-period is `P = 2π / median(ω)`**, multiplicative ratio `r = e^P`.

Diagnostics per plane say whether the circle is real: `sv_ratio` (≈1 for a degenerate pair),
`radius_cv` (0 = perfect circle), `omega_cv` (0 = phase exactly linear in `log v`), `var_share`.

Only values `v ≥ --v-min` are used: at small `v`, `log v` is sampled too coarsely and the phase
step exceeds the Nyquist limit `π` (periods below `~2·log((v_min+1)/v_min)` alias — raise
`--v-min` to reach shorter periods at the cost of fewer points).

CPU-only; consumes the `hidden_activations_mult.npz` grid from `collect_hidden_activations`.

Usage:
    python -m param_decomp_lab.scripts.validation.find_log_periods <hidden_activations_npz> \
        [--v-min=10] [--n-planes=3] [--output=PATH]

Output (default `<PARAM_DECOMP_OUT_DIR>/runs/fourier_features/log_periods_<op>.png` + `.json`):
a per-variable figure (top-plane circle coloured by `log v`, plus phase vs `log v`) and a JSON of
the detected planes with their periods and diagnostics.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import fire
import numpy as np
from numpy.typing import NDArray

from param_decomp.log import logger
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR

_SIDE_GRID = {"input": "mlp_input", "output": "mlp_output"}


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


@dataclass(frozen=True)
class Plane:
    """One candidate circular feature: SVD pair `(rank, rank+1)` of the detrended `x̄(log v)`."""

    rank: int  # index of the first of the two paired singular vectors
    log_period: float  # P: change in log v for one full turn (2π)
    ratio: float  # e^P: the multiplicative factor that advances the phase by one period
    var_share: float  # fraction of post-detrend variance in this plane
    sv_ratio: float  # S[rank+1] / S[rank] — ≈1 for a clean (degenerate) circle
    radius_cv: float  # std/mean of the trajectory radius — 0 for a perfect circle
    omega_cv: float  # MAD/|median| of angular velocity — 0 when phase is linear in log v


def _fit_planes(
    means: NDArray[np.float64], log_v: NDArray[np.float64], n_planes: int
) -> list[Plane]:
    order = np.argsort(log_v)
    log_v, means = log_v[order], means[order]

    trend = np.stack([np.ones_like(log_v), log_v], axis=1)  # [k, 2]
    coeffs, _, _, _ = np.linalg.lstsq(trend, means, rcond=None)
    resid = means - trend @ coeffs  # DC + linear-in-log-v removed

    u, s, _ = np.linalg.svd(resid, full_matrices=False)
    scores = u * s  # [k, k]; column j is that singular direction's pattern over v
    total_var = float((s**2).sum())
    d_log_v = np.diff(log_v)

    planes: list[Plane] = []
    for rank in range(0, 2 * n_planes, 2):
        s1, s2 = scores[:, rank], scores[:, rank + 1]
        # signed angle between consecutive trajectory points, in (-π, π] — no global unwrapping.
        d_theta = np.arctan2(
            s1[:-1] * s2[1:] - s2[:-1] * s1[1:], s1[:-1] * s1[1:] + s2[:-1] * s2[1:]
        )
        omega = d_theta / d_log_v
        med_omega = float(np.median(omega))
        radius = np.sqrt(s1**2 + s2**2)
        planes.append(
            Plane(
                rank=rank,
                log_period=round(2 * np.pi / abs(med_omega), 4),
                ratio=round(float(np.exp(2 * np.pi / abs(med_omega))), 4),
                var_share=round((float(s[rank]) ** 2 + float(s[rank + 1]) ** 2) / total_var, 4),
                sv_ratio=round(float(s[rank + 1]) / float(s[rank]), 4),
                radius_cv=round(float(radius.std() / radius.mean()), 4),
                omega_cv=round(float(np.median(np.abs(omega - med_omega)) / abs(med_omega)), 4),
            )
        )
    return planes


def _plot(
    fig_path: Path,
    per_variable: dict[str, tuple[NDArray[np.float64], NDArray[np.float64], list[Plane]]],
) -> None:
    from matplotlib import pyplot as plt

    n = len(per_variable)
    fig, axes = plt.subplots(n, 2, figsize=(9, 3.6 * n), squeeze=False)
    for row, (name, (means, log_v, planes)) in enumerate(per_variable.items()):
        order = np.argsort(log_v)
        log_v, means = log_v[order], means[order]
        trend = np.stack([np.ones_like(log_v), log_v], axis=1)
        coeffs, _, _, _ = np.linalg.lstsq(trend, means, rcond=None)
        u, s, _ = np.linalg.svd(means - trend @ coeffs, full_matrices=False)
        scores = u * s
        top = planes[0]
        s1, s2 = scores[:, top.rank], scores[:, top.rank + 1]

        ax = axes[row][0]
        sc = ax.scatter(s1, s2, c=log_v, cmap="viridis", s=14)
        ax.set_aspect("equal")
        ax.set_title(f"{name} · plane {top.rank}: P={top.log_period} (×{top.ratio})")
        ax.set_xlabel("score 1")
        ax.set_ylabel("score 2")
        fig.colorbar(sc, ax=ax, label="log v", fraction=0.046, pad=0.04)

        ax = axes[row][1]
        theta = np.unwrap(np.arctan2(s2, s1))
        # signed slope (rad per log v) so the reference line overlays the data's winding direction.
        d_theta = np.arctan2(
            s1[:-1] * s2[1:] - s2[:-1] * s1[1:], s1[:-1] * s1[1:] + s2[:-1] * s2[1:]
        )
        slope = float(np.median(d_theta / np.diff(log_v)))
        ax.scatter(log_v, theta, s=12, color="C0")
        ax.plot(log_v, theta[0] + slope * (log_v - log_v[0]), "0.5", ls="--")
        ax.set_title(f"phase vs log v (var {top.var_share:.0%}, ω-cv {top.omega_cv})")
        ax.set_xlabel("log v")
        ax.set_ylabel("unwrapped phase")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)


def find_log_periods(
    hidden_activations_npz: str,
    v_min: int = 10,
    n_planes: int = 3,
    output: str | None = None,
) -> Path:
    npz_path = Path(hidden_activations_npz).expanduser()
    assert npz_path.exists(), f"missing hidden-activations npz: {npz_path}"
    data = np.load(npz_path)
    op = str(data["op"])
    layer = int(data["layer"])

    a_axis, b_axis = data["a"], data["b"]
    n = a_axis.shape[0]
    assert b_axis.shape[0] == n, "non-square operand grid"
    aa, bb = np.meshgrid(a_axis.astype(np.int64), b_axis.astype(np.int64), indexing="ij")
    variable_values = {
        ("input", "a"): aa.reshape(-1),
        ("input", "b"): bb.reshape(-1),
        ("output", "result"): _result(op, aa, bb).reshape(-1),
    }

    features: dict[str, dict[str, list[dict[str, float | int]]]] = {}
    per_variable: dict[str, tuple[NDArray[np.float64], NDArray[np.float64], list[Plane]]] = {}
    for (side, variable), values in variable_values.items():
        acts = data[_SIDE_GRID[side]].reshape(n * n, -1).astype(np.float64)
        unique = np.unique(values)
        unique = unique[unique >= v_min]
        assert unique.size > 2 * n_planes, f"too few values ≥ {v_min} for {side}/{variable}"
        means = np.stack([acts[values == v].mean(axis=0) for v in unique])  # [k, d]
        log_v = np.log(unique.astype(np.float64))
        planes = _fit_planes(means, log_v, n_planes)
        features.setdefault(side, {})[variable] = [asdict(p) for p in planes]
        per_variable[f"{side}/{variable}"] = (means, log_v, planes)
        best = planes[0]
        logger.info(
            f"{op} {side}/{variable}: plane0 P={best.log_period} (×{best.ratio}) "
            f"var={best.var_share} sv_ratio={best.sv_ratio} radius_cv={best.radius_cv} "
            f"omega_cv={best.omega_cv}"
        )

    out_base = (
        Path(output).expanduser()
        if output
        else PARAM_DECOMP_OUT_DIR / "runs" / "fourier_features" / f"log_periods_{op}.png"
    )
    out_base.parent.mkdir(parents=True, exist_ok=True)
    _plot(out_base, per_variable)
    json_path = out_base.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "op": op,
                "layer": layer,
                "source": str(npz_path),
                "v_min": v_min,
                "features": features,
            },
            indent=2,
        )
    )
    logger.info(f"wrote log-period planes → {json_path} and figure → {out_base}")
    return out_base


if __name__ == "__main__":
    fire.Fire(find_log_periods)
