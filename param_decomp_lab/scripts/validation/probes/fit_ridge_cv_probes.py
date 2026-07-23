"""Ridge Fourier probes: rotating-block CV picks λ, full-range refit, permutation-null gate.

For each residual-stream position (`resid_<pos>` grid in a `collect_resid_stream` npz),
integer variable `v ∈ {a, b, a+b, a-b}`, and period `T`, fits

    cos(2πv/T) ≈ w·x + b ,   sin(2πv/T) ≈ w·x + b

by closed-form ridge on the raw per-prompt activations. Three-stage protocol:

1. **Selection** — the unique values of `v` are split into `n_folds` contiguous blocks
   (default 5, i.e. each fold holds out two contiguous deciles of the value range: every
   prompt whose `v` falls in the block). Ridge is fit on the remaining folds at every λ
   in the grid; `cv_r2` is the fold-mean held-out R² (mean of cos/sin columns, sklearn
   convention) maximised over λ. A value-memorising probe cannot earn this score — its
   held-out values never appear in training — so `cv_r2` measures generalizable circular
   structure, not linear separability.
2. **Refit** — the reported probe is refit on the *full* value range at the selected
   relative λ (λ = `lambda_rel` × mean eigenvalue of the fit's own train Gram), so no
   value range is fitted less accurately than the rest; `block_r2` records the refit
   probe's in-sample R² within each value block as a homogeneity check.
3. **Null gate** — the identical procedure (same value-block folds, λ grid, selection)
   reruns `n_perm` times with `v` randomly permuted across prompts;
   `p_value = (1 + #{null cv_r2 ≥ cv_r2}) / (1 + n_perm)`. Only probes clearing the
   null indicate real structure.

Ridge is solved in the eigenbasis of the train Gram matrix — one eigh per (position,
variable, fold, permutation), then every (period, λ) is a rescale + matvec — in fp32.
GPU strongly recommended (~3k eighs of `d_model`² at the defaults); pass `--slurm`.

Usage:
    python -m param_decomp_lab.scripts.validation.probes.fit_ridge_cv_probes \
        <resid_stream_<op>.npz> [--periods=2,5,10,20,25,33,50,100] [--n-folds=5] \
        [--n-perm=20] [--seed=0] [--output=PATH] [--slurm [--slurm-time=8:00:00 ...]]

Output (default `ridge_cv_probes_<op>.json` beside the npz): `meta` (fold value ranges
per variable, λ grid, …) + `results[<pos>][variable][period]` = `{lambda_rel, cv_r2,
fold_r2, cv_angerr_deg, null_cv_r2, p_value, full_r2, block_r2, w_cos, b_cos, w_sin,
b_sin}` (`*_sin` null at period 2, where sin ≡ 0).
"""

import json
import time
from pathlib import Path
from typing import Any

import fire
import numpy as np
import torch
from jaxtyping import Float, Int
from numpy.typing import NDArray

from param_decomp.log import logger
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME
from param_decomp_lab.scripts.validation.common import SlurmOptions, submit_self_to_slurm

_MODULE = "param_decomp_lab.scripts.validation.probes.fit_ridge_cv_probes"
_PERIODS = (2, 5, 10, 20, 25, 33, 50, 100)
_VARIABLES = ("a", "b", "a+b", "a-b")
_LAMBDAS_REL = tuple(float(x) for x in np.geomspace(1e-6, 1e3, 19))


def _variable_values(
    aa: Int[np.ndarray, " n"], bb: Int[np.ndarray, " n"], variable: str
) -> Int[np.ndarray, " n"]:
    match variable:
        case "a":
            return aa
        case "b":
            return bb
        case "a+b":
            return aa + bb
        case "a-b":
            return aa - bb
        case _:
            raise ValueError(f"unknown variable {variable!r}; expected one of {_VARIABLES}")


def _fold_test_masks(values: Int[np.ndarray, " n"], n_folds: int) -> list[NDArray[np.bool_]]:
    """Rotating contiguous value blocks: fold k holds out all prompts whose value falls in
    the k-th contiguous block of the sorted unique values (~1/n_folds of the range each)."""
    blocks = np.array_split(np.unique(values), n_folds)
    masks = [np.isin(values, block) for block in blocks]
    assert all(m.any() and not m.all() for m in masks)
    return masks


def _targets(
    values: Int[np.ndarray, " n"], periods: tuple[int, ...]
) -> tuple[Float[np.ndarray, "n 2p"], list[bool]]:
    """Columns (cos, sin) per period; `sin_valid[j]` False where sin is constant (T=2)."""
    cols: list[np.ndarray] = []
    sin_valid: list[bool] = []
    for period in periods:
        angle = 2.0 * np.pi * values / period
        cols += [np.cos(angle), np.sin(angle)]
        sin_valid.append(bool(np.sin(angle).std() > 1e-6))
    return np.stack(cols, axis=1).astype(np.float32), sin_valid


def _per_period_r2(
    r2_cols: Float[torch.Tensor, " 2p"], sin_valid: list[bool]
) -> Float[torch.Tensor, " p"]:
    cos_r2, sin_r2 = r2_cols[0::2], r2_cols[1::2]
    valid = torch.tensor(sin_valid, device=r2_cols.device)
    return torch.where(valid, (cos_r2 + sin_r2) / 2, cos_r2)


class _RidgeBasis:
    """Eigenbasis of one train split's centred Gram; solves every (target, λ) cheaply."""

    def __init__(self, x: Float[torch.Tensor, "n d"], tr: NDArray[np.bool_]):
        xtr = x[torch.from_numpy(tr).to(x.device)]
        self.mu = xtr.mean(dim=0)
        self.xc = xtr - self.mu
        gram = self.xc.T @ self.xc
        gram = (gram + gram.T) / 2
        eigvals, self.eigvecs = torch.linalg.eigh(gram)
        self.eigvals = eigvals.clamp_min(0.0)
        self.mean_eig = float(self.eigvals.mean())

    def coef_eig(
        self, y_centered_tr: Float[torch.Tensor, "ntr 2p"], lam: float
    ) -> Float[torch.Tensor, "d 2p"]:
        bt = self.eigvecs.T @ (self.xc.T @ y_centered_tr)
        return bt / (self.eigvals + lam).unsqueeze(1)

    def project(self, x: Float[torch.Tensor, "m d"]) -> Float[torch.Tensor, "m d"]:
        return (x - self.mu) @ self.eigvecs


def _cv_scores(
    x: Float[torch.Tensor, "n d"],
    values: Int[np.ndarray, " n"],
    periods: tuple[int, ...],
    lambdas_rel: tuple[float, ...],
    n_folds: int,
    with_angerr: bool,
) -> tuple[Float[np.ndarray, "f p l"], Float[np.ndarray, "f p l"]]:
    """Held-out R² (and mean angular error, degrees) per (fold, period, λ)."""
    device = x.device
    y_np, sin_valid = _targets(values, periods)
    y = torch.from_numpy(y_np).to(device)
    n_periods, n_lam = len(periods), len(lambdas_rel)
    r2 = np.full((n_folds, n_periods, n_lam), np.nan)
    angerr = np.full((n_folds, n_periods, n_lam), np.nan)
    for fold, te_mask in enumerate(_fold_test_masks(values, n_folds)):
        basis = _RidgeBasis(x, ~te_mask)
        te = torch.from_numpy(te_mask).to(device)
        yte = y[te]
        ym = y[~te].mean(dim=0)
        zte = basis.project(x[te])
        bt = basis.eigvecs.T @ (basis.xc.T @ (y[~te] - ym))
        ss_tot = ((yte - yte.mean(dim=0)) ** 2).sum(dim=0).clamp_min(1e-8)
        true_angle = torch.atan2(yte[:, 1::2], yte[:, 0::2]) if with_angerr else None
        for il, rel in enumerate(lambdas_rel):
            coef = bt / (basis.eigvals + rel * basis.mean_eig).unsqueeze(1)
            pred = zte @ coef + ym
            r2_cols = 1.0 - ((yte - pred) ** 2).sum(dim=0) / ss_tot
            r2[fold, :, il] = _per_period_r2(r2_cols, sin_valid).cpu().numpy()
            if true_angle is not None:
                pred_angle = torch.atan2(pred[:, 1::2], pred[:, 0::2])
                diff = torch.remainder(pred_angle - true_angle + torch.pi, 2 * torch.pi) - torch.pi
                fold_err = torch.rad2deg(diff.abs().mean(dim=0)).cpu().numpy()
                angerr[fold, :, il] = np.where(sin_valid, fold_err, np.nan)
    return r2, angerr


def _masked_r2(
    y_true: Float[torch.Tensor, "n 2"],
    y_pred: Float[torch.Tensor, "n 2"],
    mask: NDArray[np.bool_],
    n_cols: int,
) -> float:
    m = torch.from_numpy(mask).to(y_true.device)
    ss_res = ((y_true[m] - y_pred[m]) ** 2).sum(dim=0)
    ss_tot = ((y_true[m] - y_true[m].mean(dim=0)) ** 2).sum(dim=0).clamp_min(1e-8)
    cols = 1.0 - ss_res / ss_tot
    return float(cols[:n_cols].mean())


def _selected_cv_r2(
    r2: Float[np.ndarray, "f p l"],
) -> tuple[Float[np.ndarray, " p"], Int[np.ndarray, " p"]]:
    """Per period: the λ index maximising the fold-mean R², and that maximum."""
    fold_mean = r2.mean(axis=0)
    best = fold_mean.argmax(axis=1)
    return fold_mean[np.arange(fold_mean.shape[0]), best], best


def fit_ridge_cv_probes(
    resid_stream_npz: str,
    periods: tuple[int, ...] = _PERIODS,
    variables: tuple[str, ...] = _VARIABLES,
    n_folds: int = 5,
    n_perm: int = 20,
    seed: int = 0,
    output: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "8:00:00",
    slurm_mem: str | None = None,
    dependency: str | None = None,
) -> Path | None:
    periods = _as_tuple(periods)
    variables = _as_tuple(variables)
    npz_path = Path(resid_stream_npz).expanduser()
    if slurm:
        argv = [
            str(npz_path),
            f"--periods={','.join(str(p) for p in periods)}",
            f"--variables={','.join(variables)}",
            f"--n-folds={n_folds}",
            f"--n-perm={n_perm}",
            f"--seed={seed}",
        ]
        if output is not None:
            argv.append(f"--output={Path(output).expanduser()}")
        opts = SlurmOptions(
            partition=partition,
            gpus=gpus,
            slurm_time=slurm_time,
            slurm_mem=slurm_mem,
            dependency=dependency,
        )
        submit_self_to_slurm(
            _MODULE, argv, opts, job_name=f"val-ridge-cv-{npz_path.stem.split('_')[-1]}"
        )
        return None

    assert npz_path.exists(), f"missing resid stream npz: {npz_path}"
    t0 = time.time()
    data = np.load(npz_path)
    op = str(data["op"])
    positions = [str(p) for p in data["positions"]]
    a_axis, b_axis = data["a"].astype(np.int64), data["b"].astype(np.int64)
    aa, bb = (g.reshape(-1) for g in np.meshgrid(a_axis, b_axis, indexing="ij"))
    n = len(aa)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)
    logger.info(f"{npz_path.name}: op={op}, positions {positions}, {n} prompts, device {device}")

    block_ranges = {
        v: [
            [int(block[0]), int(block[-1])]
            for block in np.array_split(np.unique(_variable_values(aa, bb, v)), n_folds)
        ]
        for v in variables
    }
    results: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for pos in positions:
        x = torch.from_numpy(np.asarray(data[f"resid_{pos}"], dtype=np.float32)).to(device)
        x = x.reshape(n, -1)
        full_basis = _RidgeBasis(x, np.ones(n, dtype=bool))
        pos_results: dict[str, dict[str, dict[str, Any]]] = {}
        for variable in variables:
            values = _variable_values(aa, bb, variable)
            r2, angerr = _cv_scores(x, values, periods, _LAMBDAS_REL, n_folds, with_angerr=True)
            cv_r2, best_il = _selected_cv_r2(r2)

            null_cv_r2 = np.empty((n_perm, len(periods)))
            for perm in range(n_perm):
                null_r2, _ = _cv_scores(
                    x, rng.permutation(values), periods, _LAMBDAS_REL, n_folds, with_angerr=False
                )
                null_cv_r2[perm], _ = _selected_cv_r2(null_r2)
            p_value = (1 + (null_cv_r2 >= cv_r2).sum(axis=0)) / (1 + n_perm)

            y_np, sin_valid = _targets(values, periods)
            y = torch.from_numpy(y_np).to(device)
            ym = y.mean(dim=0)
            yc = y - ym
            fold_masks = _fold_test_masks(values, n_folds)
            pos_results[variable] = {}
            for j, period in enumerate(periods):
                rel = _LAMBDAS_REL[int(best_il[j])]
                coef = full_basis.coef_eig(yc[:, 2 * j : 2 * j + 2], rel * full_basis.mean_eig)
                w = full_basis.eigvecs @ coef
                bias = ym[2 * j : 2 * j + 2] - full_basis.mu @ w
                pred = x @ w + bias
                yp = y[:, 2 * j : 2 * j + 2]
                n_cols = 1 + int(sin_valid[j])
                sel_angerr = (
                    float(np.nanmean(angerr[:, j, int(best_il[j])]))
                    if sin_valid[j]
                    else float("nan")
                )
                pos_results[variable][str(period)] = {
                    "lambda_rel": rel,
                    "cv_r2": round(float(cv_r2[j]), 5),
                    "fold_r2": [round(float(v), 5) for v in r2[:, j, int(best_il[j])]],
                    "cv_angerr_deg": None if np.isnan(sel_angerr) else round(sel_angerr, 2),
                    "null_cv_r2": [round(float(v), 5) for v in null_cv_r2[:, j]],
                    "p_value": round(float(p_value[j]), 5),
                    "full_r2": round(_masked_r2(yp, pred, np.ones(n, dtype=bool), n_cols), 5),
                    "block_r2": [round(_masked_r2(yp, pred, m, n_cols), 5) for m in fold_masks],
                    "w_cos": [round(float(v), 6) for v in w[:, 0].cpu().numpy()],
                    "b_cos": round(float(bias[0]), 6),
                    "w_sin": [round(float(v), 6) for v in w[:, 1].cpu().numpy()]
                    if sin_valid[j]
                    else None,
                    "b_sin": round(float(bias[1]), 6) if sin_valid[j] else None,
                }
            top = max(pos_results[variable].items(), key=lambda kv: kv[1]["cv_r2"])
            logger.info(
                f"[{time.time() - t0:6.0f}s] {pos}/{variable}: best T={top[0]} "
                f"cv_r2={top[1]['cv_r2']:.3f} (p={top[1]['p_value']:.3f})"
            )
        results[pos] = pos_results
        del x, full_basis

    payload = {
        "meta": {
            "source": str(npz_path),
            "op": op,
            "positions": positions,
            "variables": list(variables),
            "periods": list(periods),
            "lambdas_rel": [float(f"{x:.4e}") for x in _LAMBDAS_REL],
            "n_folds": n_folds,
            "n_perm": n_perm,
            "seed": seed,
            "n_prompts": n,
            "max_value": int(data["max_value"]),
            "block_ranges": block_ranges,
            "r2_convention": "held-out R2, sklearn (test-mean) baseline, mean of cos/sin cols",
        },
        "results": results,
    }
    out_path = (
        Path(output).expanduser() if output else npz_path.with_name(f"ridge_cv_probes_{op}.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    logger.info(
        f"[{time.time() - t0:6.0f}s] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)"
    )
    return out_path


def _as_tuple(x: "tuple[Any, ...] | Any") -> tuple[Any, ...]:
    return tuple(x) if isinstance(x, (list, tuple)) else (x,)


if __name__ == "__main__":
    fire.Fire(fit_ridge_cv_probes)
