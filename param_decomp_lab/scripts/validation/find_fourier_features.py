"""Fit the circular ("Fourier") probes Llama-3.1-8B uses around layer 18's MLP.

Feucht et al. (2026, "Arithmetic in the Wild", https://arxiv.org/abs/2605.01148) train linear
probes that read a number's circular (Fourier) representation off the model's activations. For a
period `T` and integer variable `v` (operand `a`, operand `b`, or the result), the probe predicts
the two Fourier coordinates from the activation `x` (their Eq. 9, bias included):

    cos(θ) ≈ w_cos · x + b_cos ,    sin(θ) ≈ w_sin · x + b_sin

fit by least squares over the individual prompts. `θ = 2πv/T` in **linear** space (add/sub) or
`θ = 2π·log(v)/log(r)` in **log** space (mult, `period` = multiplicative ratio `r`; the ratios are
the canonical log-periods from the sibling `subcomp_periods_mult.tsv`). This replicates their
probing exactly.

Probes are fit at two **sites**, selectable in the applet:
- **mlp** — `a`, `b` at `mlp_input` (post-RMSNorm, where the gate/up components read); the result at
  `mlp_output` (the MLP write, where the down components write).
- **resid** — `a`, `b` at `resid_pre_mlp` (the residual stream entering the MLP, Feucht's site); the
  result at `resid_pre_mlp + mlp_output` (the residual after the MLP has written).

Weights are fit on a train split; `r2` is the held-out (test) fraction of variance explained,
averaged over the cos and sin probes — the diagnostic for whether the feature is present at `T`
(for period 2 `sin(2πv/2)=0` identically, so only the cos probe contributes).

Consumes the `hidden_activations_<op>.npz` grid from `collect_hidden_activations` (no GPU / forward
pass here); run that once per task first. Fit separately per task.

Usage:
    python -m param_decomp_lab.scripts.validation.find_fourier_features <hidden_activations_npz> \
        [--periods=2,5,10,...] [--space=linear|log] [--output=PATH]

Output (default `<PARAM_DECOMP_OUT_DIR>/runs/fourier_features/coordinates_<op>.json`): run/op
metadata (incl. `space`, `sites`) plus
`features[site][operand][period] = {period, r2, w_cos, b_cos, w_sin, b_sin}`, each `w_*` a
`d_model`-long list. For log space `period` holds the ratio `r`.
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
# Log space (mult): multiplicative ratios `r`, θ = 2π·log_r(v). Fallback only — the default ratios
# are read from the run's `subcomp_periods_mult.tsv` (the frequencies the period analysis found).
_CANONICAL_RATIOS = (1.26,)
_SITES = ("mlp", "resid")
_OPERANDS = ("a", "b", "result")
_TRAIN_FRAC = 0.8
_SPLIT_SEED = 0


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


def _site_grids(data: Any, n: int) -> dict[str, dict[str, NDArray[np.float32]]]:
    """Per-site, per-operand `[n*n, d]` activation grid the probe reads (see module docstring)."""

    def flat(name: str) -> NDArray[np.float32]:
        return data[name].reshape(n * n, -1).astype(np.float32)

    mlp_in, mlp_out = flat("mlp_input"), flat("mlp_output")
    resid_pre = flat("resid_pre_mlp")
    resid_post = resid_pre + mlp_out
    return {
        "mlp": {"a": mlp_in, "b": mlp_in, "result": mlp_out},
        "resid": {"a": resid_pre, "b": resid_pre, "result": resid_post},
    }


def _theta(values: NDArray[np.int64], period: float, space: str) -> NDArray[np.float64]:
    v = values.astype(np.float64)
    if space == "log":
        assert period > 0.0 and period != 1.0, f"log ratio must be positive and != 1, got {period}"
        assert v.min() > 0, f"log space needs positive values, got min {v.min()}"
        return 2.0 * np.pi * np.log(v) / np.log(period)
    return 2.0 * np.pi * v / period


def _fit_probes(
    x: NDArray[np.float32],
    values: NDArray[np.int64],
    periods: tuple[float, ...],
    space: str,
    train: NDArray[np.int64],
    test: NDArray[np.int64],
) -> dict[str, Any]:
    """Feucht's linear Fourier probes `[cos θ, sin θ] ≈ x·W + b` (bias) for every period at once —
    one least squares over the train prompts with the `[N, 2·P]` stacked cos/sin targets (the design
    is shared). Per period: the two weight vectors + biases and the held-out (test) R², averaged
    over the cos and sin probes (only cos where sin is degenerate, e.g. period 2)."""
    cols = [
        np.stack([np.cos(t), np.sin(t)], axis=1)
        for t in (_theta(values, p, space) for p in periods)
    ]
    y = np.concatenate(cols, axis=1)  # [N, 2P]: (cos_p0, sin_p0, cos_p1, ...)
    design = np.concatenate([x, np.ones((x.shape[0], 1), np.float32)], axis=1)  # [N, d+1]
    # Normal equations (float64) — a 4098×4098 solve beats lstsq's full SVD of an 8000×4097 matrix
    # by orders of magnitude on CPU; the probe design is well-conditioned so squaring is harmless.
    xtr = design[train].astype(np.float64)
    coef = np.linalg.solve(xtr.T @ xtr, xtr.T @ y[train])  # [d+1, 2P]

    pred = design[test] @ coef
    ss_res = ((y[test] - pred) ** 2).sum(axis=0)  # [2P]
    ss_tot = ((y[test] - y[test].mean(axis=0)) ** 2).sum(axis=0)  # [2P]
    out: dict[str, Any] = {}
    for i, period in enumerate(periods):
        c, s = 2 * i, 2 * i + 1
        valid = ss_tot[[c, s]] > 1e-9  # sin constant (0) at period 2 → excluded from R²
        assert valid.any(), f"both cos and sin targets constant at period {period}"
        r2 = (1.0 - ss_res[[c, s]][valid] / ss_tot[[c, s]][valid]).mean()
        out[str(period)] = {
            "period": period,
            "r2": round(float(r2), 6),
            "w_cos": [round(float(t), 6) for t in coef[:-1, c]],
            "b_cos": round(float(coef[-1, c]), 6),
            "w_sin": [round(float(t), 6) for t in coef[:-1, s]],
            "b_sin": round(float(coef[-1, s]), 6),
        }
    return out


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
    if periods is None:
        periods = (
            _log_ratios_from_periods_tsv(npz_path, op) if space == "log" else _CANONICAL_PERIODS
        )
    elif not isinstance(periods, (list, tuple)):
        periods = (periods,)  # fire parses a single `--periods=1.26` as a scalar
    periods = tuple(periods)

    a_axis, b_axis = data["a"], data["b"]
    n = a_axis.shape[0]
    assert b_axis.shape[0] == n, "non-square operand grid"
    # grids are [a-1, b-1, d]; flatten with `indexing="ij"` so sample k has a=aa[k], b=bb[k].
    aa, bb = np.meshgrid(a_axis.astype(np.int64), b_axis.astype(np.int64), indexing="ij")
    operand_values = {
        "a": aa.reshape(-1),
        "b": bb.reshape(-1),
        "result": _result(op, aa, bb).reshape(-1),
    }
    site_grids = _site_grids(data, n)

    # One fixed train/test split shared by every probe (deterministic).
    perm = np.random.default_rng(_SPLIT_SEED).permutation(n * n)
    n_train = int(_TRAIN_FRAC * n * n)
    train, test = perm[:n_train], perm[n_train:]

    features: dict[str, dict[str, dict[str, Any]]] = {}
    for site in _SITES:
        features[site] = {}
        for operand in _OPERANDS:
            x = site_grids[site].pop(operand)  # drop as we go to keep peak memory down
            features[site][operand] = _fit_probes(
                x, operand_values[operand], periods, space, train, test
            )
            del x
            r2s = {p: features[site][operand][str(p)]["r2"] for p in periods}
            logger.info(f"{op} {site}/{operand} ({space}): held-out r2 by period {r2s}")

    payload = {
        "op": op,
        "symbol": op_symbol(op),
        "layer": layer,
        "space": space,
        "sites": list(_SITES),
        "operands": list(_OPERANDS),
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
    logger.info(f"wrote Fourier probes for {op} ({size_mb:.1f} MB) → {out_path}")
    return out_path


if __name__ == "__main__":
    fire.Fire(find_fourier_features)
