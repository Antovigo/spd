"""Fit Feucht et al.'s exact Fourier probes on the collected resid-layer-output activations.

Replicates their fitting strategy verbatim. For each period `T` and integer variable `v` — operand
`a`, operand `b`, or the sum `a+b` (their input / offset / output) — it trains two linear probes

    cos(2πv/T) ≈ w_cos·x + b_cos ,   sin(2πv/T) ≈ w_sin·x + b_sin

each an `nn.Linear(d_model, 1)` (bias) optimised by Adam (lr 1e-3) against MSE for 500 epochs on the
**raw, un-standardised** activations, over an 80/20 split (`sklearn.train_test_split`,
`test_size=0.2`, `random_state=42`, run order `a` outer / `b` inner — identical to their prompt
order, so the split matches theirs). `r2_*` is the held-out (test) R². For period 2 `sin(2πv/2)=0`
identically, so its sin probe is skipped (`w_sin=0`, `r2_sin=null`).

By default it sweeps Feucht's **full per-variable spectrum** — every period `2..min(v.max()//2,
max_period)` (`max_period=150`; so `a`,`b` → 2..100, `a+b` → 2..150). This is a control: `R²` should
spike only at the true periods and stay low elsewhere (a fake period gives a blob, not a circle).
`--periods` overrides with an explicit shared list.

Consumes `resid_activations.npz` from `collect_resid_activations`. GPU strongly recommended (500
epochs × ~700 probes for the full sweep); pass `--slurm` to submit as a single-GPU job.

Usage:
    python -m param_decomp_lab.scripts.validation.probes.fit_fourier_probes <resid_activations.npz> \
        [--max-period=150 | --periods=2,5,10,...] [--output=PATH] [--slurm ...]

Output (default `probes.json` beside the npz): metadata (`periods_by_variable`, `max_period`) +
`probes[variable][period] = {w_cos, b_cos, w_sin, b_sin, r2_cos, r2_sin}`, weight vectors
`d_model`-long.
"""

import json
from pathlib import Path
from typing import Any, cast, override

import fire
import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from torch import nn

from param_decomp.log import logger
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME
from param_decomp_lab.scripts.validation.common import SlurmOptions, submit_self_to_slurm

_MODULE = "param_decomp_lab.scripts.validation.probes.fit_fourier_probes"
_LR = 1e-3
_EPOCHS = 500
_TEST_SIZE = 0.2
_RANDOM_STATE = 42


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def _train_probe(
    probe: LinearProbe,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
) -> tuple[NDArray[np.float32], float, float] | None:
    """Feucht's `train_probe` (Adam/MSE/500 epochs, full-batch); returns `(w, b, held-out R²)` or
    None for a degenerate (constant) target — e.g. the period-2 sin."""
    if float(y_train.std()) < 1e-6:
        return None
    optimizer = torch.optim.Adam(probe.parameters(), lr=_LR)
    loss_fn = nn.MSELoss()
    for _ in range(_EPOCHS):
        probe.train()
        optimizer.zero_grad()
        loss_fn(probe(x_train), y_train).backward()
        optimizer.step()
    probe.eval()
    with torch.no_grad():
        pred = probe(x_test)
        ss_res = float(((y_test - pred) ** 2).sum())
        ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-8 else float("nan")
    w = probe.linear.weight.detach().squeeze(0).cpu().numpy()
    b = float(probe.linear.bias.detach().reshape(()).cpu())
    return w, b, r2


def fit_fourier_probes(
    resid_activations_npz: str,
    periods: tuple[int, ...] | int | None = None,
    max_period: int = 150,
    output: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "2:00:00",
    slurm_mem: str | None = None,
) -> Path | None:
    npz_path = Path(resid_activations_npz).expanduser()
    if slurm:
        argv = [str(npz_path), f"--max-period={max_period}"]
        if periods is not None:
            argv.append(f"--periods={','.join(str(p) for p in _as_tuple(periods))}")
        if output is not None:
            argv.append(f"--output={Path(output).expanduser()}")
        opts = SlurmOptions(
            partition=partition, gpus=gpus, slurm_time=slurm_time, slurm_mem=slurm_mem
        )
        submit_self_to_slurm(_MODULE, argv, opts, job_name="val-fit-probes")
        return None

    assert npz_path.exists(), f"missing resid activations: {npz_path}"
    # None → Feucht's full per-variable sweep `2..min(v.max()//2, max_period)`; else a shared list.
    explicit_periods = _as_tuple(periods) if periods is not None else None
    data = np.load(npz_path)
    grid = data["resid"]  # [G, G, d] fp16, indexed [a-1, b-1]
    g = int(grid.shape[0])
    a_axis, b_axis = data["a"].astype(np.int64), data["b"].astype(np.int64)
    # flatten a-outer / b-inner (matches Feucht's prompt order so the split coincides)
    aa, bb = np.meshgrid(a_axis, b_axis, indexing="ij")
    variables = {"a": aa.reshape(-1), "b": bb.reshape(-1), "a+b": (aa + bb).reshape(-1)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_all = torch.tensor(grid.reshape(g * g, -1), dtype=torch.float32)
    split = train_test_split(
        x_all, np.arange(g * g), test_size=_TEST_SIZE, random_state=_RANDOM_STATE
    )
    x_train, x_test = (
        cast(torch.Tensor, split[0]).to(device),
        cast(torch.Tensor, split[1]).to(device),
    )
    idx_train, idx_test = cast("NDArray[np.int64]", split[2]), cast("NDArray[np.int64]", split[3])
    logger.info(
        f"fitting on {device}: {x_train.shape[0]} train / {x_test.shape[0]} test, d={x_all.shape[1]}"
    )

    torch.manual_seed(_RANDOM_STATE)
    if device == "cuda":
        torch.cuda.manual_seed_all(_RANDOM_STATE)

    probes: dict[str, dict[str, Any]] = {}
    periods_by_variable: dict[str, list[int]] = {}
    for var_name, values in variables.items():
        var_periods = (
            explicit_periods
            if explicit_periods is not None
            else tuple(range(2, min(int(values.max()) // 2, max_period) + 1))
        )
        periods_by_variable[var_name] = list(var_periods)
        probes[var_name] = {}
        for period in var_periods:
            entry: dict[str, Any] = {}
            for func_name, func in (("cos", np.cos), ("sin", np.sin)):
                target = func(2.0 * np.pi * values / period).astype(np.float32)
                y_train = torch.tensor(target[idx_train]).to(device)
                y_test = torch.tensor(target[idx_test]).to(device)
                fit = _train_probe(
                    LinearProbe(x_all.shape[1]).to(device), x_train, y_train, x_test, y_test
                )
                if fit is None:  # constant target (sin at period 2)
                    w, b, r2 = np.zeros(x_all.shape[1], np.float32), 0.0, None
                else:
                    w, b, r2 = fit[0], fit[1], round(fit[2], 6)
                entry[f"w_{func_name}"] = [round(float(t), 6) for t in w]
                entry[f"b_{func_name}"] = round(float(b), 6)
                entry[f"r2_{func_name}"] = r2
            probes[var_name][str(period)] = entry
        r2_cos = [probes[var_name][str(p)]["r2_cos"] for p in var_periods]
        logger.info(
            f"{var_name}: {len(var_periods)} periods {var_periods[0]}..{var_periods[-1]}, "
            f"cos R² max {max(r for r in r2_cos if r is not None):.3f}"
        )

    payload = {
        "model": "meta-llama/Llama-3.1-8B",
        "layer": int(data["layer"]),
        "max_value": int(data["max_value"]),
        "site": "resid_layer_output",
        "variables": list(variables),
        "periods_by_variable": periods_by_variable,
        "max_period": max_period,
        "lr": _LR,
        "epochs": _EPOCHS,
        "test_size": _TEST_SIZE,
        "random_state": _RANDOM_STATE,
        "source": str(npz_path),
        "probes": probes,
    }
    out_path = Path(output).expanduser() if output else npz_path.with_name("probes.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload))
    logger.info(f"wrote probes ({out_path.stat().st_size / 1e6:.1f} MB) → {out_path}")
    return out_path


def _as_tuple(periods: "tuple[int, ...] | int") -> tuple[int, ...]:
    return periods if isinstance(periods, (list, tuple)) else (periods,)


if __name__ == "__main__":
    fire.Fire(fit_fourier_probes)
