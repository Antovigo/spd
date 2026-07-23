"""Scatter the activations projected onto the ridge-CV probe planes.

Two modes, both coloring a subsample of per-prompt activations at
`(x·w_cos + b_cos, x·w_sin + b_sin)` — the probe's predicted (cos, sin) plane — by
`v mod T` on a cyclic colormap, so a genuine circular representation shows as an
ordered color wheel. Period 2 (sin ≡ 0) is drawn as a 1-D strip with jitter.

- Default: one summary figure per op — rows = variables, cols = periods, each panel the
  **best accepted layer** for that cell; cells with no accepted layer render "gated".
- `--all-layers`: one figure per (op, variable) — rows = **every** stream position in
  order (after L14 … after L20), cols = periods — so the layer at which a
  representation arrives (operands) or is generated (results) is visible in the point
  clouds. Every cell is drawn from its own full-range-refit probe; cells failing the
  permutation gate are tinted grey (their blob is the point: nothing decodable there).

Usage:
    python -m param_decomp_lab.scripts.validation.probes.plot_probe_projections \
        <ridge_cv_probes_<op>.json> [--all-layers] [--resid-stream-npz=PATH] \
        [--alpha=0.05] [--n-show=3000] [--seed=0] [--output=PATH] [--slurm [--gpus=0 ...]]

Output (default beside the json): `probe_projections_<op>.png`, or per-variable
`probe_projections_<op>_<a|b|sum|diff>.png` with `--all-layers` (`--output` then must be
omitted). The npz defaults to the json's `meta.source` (the npz the probes were fit on).
"""

import json
from pathlib import Path
from typing import Any

import fire
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray

from param_decomp.log import logger
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME
from param_decomp_lab.scripts.validation.common import SlurmOptions, submit_self_to_slurm
from param_decomp_lab.scripts.validation.probes.fit_ridge_cv_probes import _variable_values

_MODULE = "param_decomp_lab.scripts.validation.probes.plot_probe_projections"
_CYCLIC_CMAP = "twilight"
_GATED_TINT = "#ececef"
_VAR_SLUG = {"a": "a", "b": "b", "a+b": "sum", "a-b": "diff"}


def _project(x: NDArray[np.float32], cell: dict[str, Any]) -> NDArray[np.float32]:
    """[n, 2] predicted (cos, sin) — or [n, 1] for a cos-only (period-2) probe."""
    w_cos = np.asarray(cell["w_cos"], dtype=np.float32)
    cos_pred = x @ w_cos + cell["b_cos"]
    if cell["w_sin"] is None:
        return cos_pred[:, None]
    w_sin = np.asarray(cell["w_sin"], dtype=np.float32)
    return np.stack([cos_pred, x @ w_sin + cell["b_sin"]], axis=1)


def _draw_panel(
    ax: Axes,
    p: NDArray[np.float32],
    values: NDArray[np.int64],
    period: int,
    rng: np.random.Generator,
) -> None:
    color = (values % period) / period
    common: dict[str, Any] = {
        "c": color,
        "cmap": _CYCLIC_CMAP,
        "vmin": 0,
        "vmax": 1,
        "s": 2,
        "alpha": 0.5,
        "linewidths": 0,
    }
    if p.shape[1] == 1:  # period-2: 1-D probe, jitter the empty axis
        ax.scatter(p[:, 0], rng.uniform(-0.5, 0.5, len(p)), **common)
    else:
        ax.scatter(p[:, 0], p[:, 1], **common)
        ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="#999999", lw=0.6, ls=":"))
        ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cccccc")


def plot_probe_projections(
    ridge_cv_json: str,
    all_layers: bool = False,
    resid_stream_npz: str | None = None,
    alpha: float = 0.05,
    n_show: int = 3000,
    seed: int = 0,
    output: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 0,
    slurm_time: str = "1:00:00",
    slurm_mem: str | None = "64G",
    dependency: str | None = None,
) -> Path | list[Path] | None:
    json_path = Path(ridge_cv_json).expanduser()
    if slurm:
        argv = [str(json_path), f"--alpha={alpha}", f"--n-show={n_show}", f"--seed={seed}"]
        if all_layers:
            argv.append("--all-layers")
        if resid_stream_npz is not None:
            argv.append(f"--resid-stream-npz={Path(resid_stream_npz).expanduser()}")
        if output is not None:
            argv.append(f"--output={Path(output).expanduser()}")
        opts = SlurmOptions(
            partition=partition,
            gpus=gpus,
            slurm_time=slurm_time,
            slurm_mem=slurm_mem,
            dependency=dependency,
        )
        submit_self_to_slurm(_MODULE, argv, opts, job_name="val-probe-proj")
        return None

    assert json_path.exists(), f"missing ridge-CV probes json: {json_path}"
    payload = json.loads(json_path.read_text())
    meta, results = payload["meta"], payload["results"]
    op = meta["op"]
    variables: list[str] = meta["variables"]
    periods: list[int] = meta["periods"]
    npz_path = Path(resid_stream_npz).expanduser() if resid_stream_npz else Path(meta["source"])
    assert npz_path.exists(), f"missing resid stream npz: {npz_path}"
    data = np.load(npz_path)
    a_axis, b_axis = data["a"].astype(np.int64), data["b"].astype(np.int64)
    aa, bb = (g.reshape(-1) for g in np.meshgrid(a_axis, b_axis, indexing="ij"))
    n = len(aa)
    rng = np.random.default_rng(seed)
    show = rng.choice(n, size=min(n_show, n), replace=False)
    values_by_var = {v: _variable_values(aa, bb, v)[show] for v in variables}

    def accepted(cell: dict[str, Any]) -> bool:
        return cell["p_value"] <= alpha and cell["cv_r2"] > 0

    if all_layers:
        assert output is None, "--output is ambiguous with --all-layers (one file per variable)"
        layer_keys = [str(p) for p in meta["positions"]]
        proj: dict[tuple[str, str, int], NDArray[np.float32]] = {}
        for layer_key in layer_keys:
            x = np.asarray(data[f"resid_{layer_key}"], dtype=np.float32).reshape(n, -1)[show]
            for variable in variables:
                for period in periods:
                    cell = results[layer_key][variable][str(period)]
                    proj[(layer_key, variable, period)] = _project(x, cell)
            del x
        out_paths: list[Path] = []
        for variable in variables:
            fig, axes = plt.subplots(
                len(layer_keys),
                len(periods),
                figsize=(2.1 * len(periods) + 0.7, 2.1 * len(layer_keys) + 0.7),
                squeeze=False,
                constrained_layout=True,
            )
            for row_i, layer_key in enumerate(layer_keys):
                for col_i, period in enumerate(periods):
                    ax = axes[row_i][col_i]
                    cell = results[layer_key][variable][str(period)]
                    _draw_panel(
                        ax,
                        proj[(layer_key, variable, period)],
                        values_by_var[variable],
                        period,
                        rng,
                    )
                    if not accepted(cell):
                        ax.set_facecolor(_GATED_TINT)
                    tag = f"R²={cell['cv_r2']:.2f}" if accepted(cell) else "gated"
                    ax.text(
                        0.02,
                        0.98,
                        tag,
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=7,
                        color="#555555",
                    )
                    if row_i == 0:
                        ax.set_title(f"T={period}", fontsize=10)
                    if col_i == 0:
                        ax.set_ylabel(f"after {layer_key}", fontsize=9)
            fig.suptitle(
                f"{op} / {variable}: probe-plane projections down the stream "
                f"(color = {variable} mod T, {len(show)} prompts; grey tint = fails null gate)",
                fontsize=11,
            )
            out_path = json_path.parent / f"probe_projections_{op}_{_VAR_SLUG[variable]}.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            logger.info(f"wrote {out_path}")
            out_paths.append(out_path)
        return out_paths

    def best_layer(variable: str, period: int) -> str | None:
        cells = {
            layer_key: cell[variable][str(period)]
            for layer_key, cell in results.items()
            if accepted(cell[variable][str(period)])
        }
        return max(cells, key=lambda k: cells[k]["cv_r2"]) if cells else None

    chosen = {
        (v, t): layer_key
        for v in variables
        for t in periods
        if (layer_key := best_layer(v, t)) is not None
    }
    best_proj: dict[tuple[str, int], NDArray[np.float32]] = {}
    for layer_key in sorted(set(chosen.values())):
        x = np.asarray(data[f"resid_{layer_key}"], dtype=np.float32).reshape(n, -1)[show]
        for (variable, period), lk in chosen.items():
            if lk == layer_key:
                best_proj[(variable, period)] = _project(
                    x, results[layer_key][variable][str(period)]
                )
        del x

    fig, axes = plt.subplots(
        len(variables),
        len(periods),
        figsize=(2.3 * len(periods) + 0.8, 2.3 * len(variables) + 0.7),
        squeeze=False,
        constrained_layout=True,
    )
    for row_i, variable in enumerate(variables):
        for col_i, period in enumerate(periods):
            ax = axes[row_i][col_i]
            if row_i == 0:
                ax.set_title(f"T={period}", fontsize=10)
            if col_i == 0:
                ax.set_ylabel(variable, fontsize=11)
            if (variable, period) not in best_proj:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_color("#cccccc")
                ax.set_facecolor(_GATED_TINT)
                ax.text(
                    0.5,
                    0.5,
                    "gated",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#8e8e93",
                )
                continue
            _draw_panel(ax, best_proj[(variable, period)], values_by_var[variable], period, rng)
            cell = results[chosen[(variable, period)]][variable][str(period)]
            ax.text(
                0.02,
                0.98,
                f"{chosen[(variable, period)]}  R²={cell['cv_r2']:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                color="#555555",
            )
    fig.suptitle(
        f"{op}: activations on the probe planes (predicted cos/sin), color = value mod T "
        f"({_CYCLIC_CMAP}); best accepted layer per cell, {len(show)} prompts",
        fontsize=11,
    )
    out_path = (
        Path(output).expanduser() if output else json_path.parent / f"probe_projections_{op}.png"
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info(f"wrote {out_path} ({len(best_proj)}/{len(variables) * len(periods)} panels drawn)")
    return out_path


if __name__ == "__main__":
    fire.Fire(plot_probe_projections)
