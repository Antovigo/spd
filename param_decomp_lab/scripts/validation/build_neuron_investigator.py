"""Build a GPU-free HTML app to investigate which neurons take part in an arithmetic task.

For L18's MLP it cross-tabulates every filtered-alive subcomponent against the hidden
neurons it couples to, via the per-(subcomponent, neuron) **coefficient of interaction**
(the unit-normalized read/write weight, always ≥ 0):
- gate / up (pre-SwiGLU, *write* to neurons): `|U[c, j]| / ||U_c||`,
- down (post-SwiGLU, *read* from neurons): `|V[j, c]| / ||V_c||`.

The applet's left half is a neuron × subcomponent heatmap (subcomponents sorted by period
then mean CI; neurons sorted by total coefficient across all matrices, paged). Write
coefficients render blue, read coefficients red — done by flipping the sign of down (read)
columns and colouring with a diverging RdBu scale. Clicking a cell selects that
(neuron, subcomponent) pair; the right half then shows the subcomponent's inner-activation
`(a, b)` heatmap alongside the neuron's up / gate / post-SwiGLU-output `(a, b)` heatmaps.

Only the top-`top_neurons` neurons by total coefficient are kept — their up / gate grids
(needed for the right panel) are the bulk of the payload, so the cap bounds the file size.

CPU-only: reads the checkpoint U/V (mmap) plus, from `<run>/analysis/datasets/`, the
filtered-alive list (`alive_filtered_<op>.tsv`, for mean CI), the periods
(`subcomp_periods_<op>.tsv`), the inner-activation TSV (`inner_activations_<op>.tsv`), and the
hidden-activation npz (`hidden_activations_<op>.npz`, for the neuron grids). No forward pass.

Usage:
    python -m param_decomp_lab.scripts.validation.build_neuron_investigator <model_path> \
        [--op=add] [--top-neurons=512] [--output-dir=PATH]

Output: `<run_dir>/analysis/neuron_investigator_<op>/{index.html,data.js}`.
"""

import base64
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import fire
import numpy as np
from numpy.typing import NDArray

from param_decomp.log import logger
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.scripts.validation.common import (
    MLP_MATRICES,
    AliveComponent,
    analysis_datasets_dir,
    analysis_dir,
    load_component_uv,
    op_symbol,
    read_alive_components,
    read_subcomp_periods,
)

_APP_TEMPLATE = Path(__file__).with_name("neuron_investigator_app.html")


def _read_mean_ci(tsv_path: Path) -> dict[tuple[str, int], float]:
    """`(proj, component) -> mean CI` from an `alive_filtered_<op>.tsv`."""
    out: dict[tuple[str, int], float] = {}
    with tsv_path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[(row["matrix"].split(".")[-1], int(row["component"]))] = float(row["mean_ci"])
    return out


def _coeff_vector(
    a: AliveComponent, uv: dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]]
) -> NDArray[np.float32]:
    """Per-neuron coefficient of interaction (unit-normalized, ≥ 0) for one subcomponent."""
    v, u = uv[a.proj]
    # down (post-SwiGLU): read weights |V[:,c]|/||V_c||; gate/up (pre-SwiGLU): write |U[c,:]|/||U_c||.
    vec = v[:, a.component] if a.proj == "down_proj" else u[a.component, :]
    return (np.abs(vec) / max(float(np.linalg.norm(vec)), 1e-12)).astype(np.float32)


def _inner_grids(
    tsv_path: Path, alive_keys: set[tuple[str, int]], n: int
) -> dict[tuple[str, int], NDArray[np.float32]]:
    """Dense `(a, b)` inner-activation grid (row-major `[a, b]`) per filtered-alive subcomponent."""
    grids = {key: np.zeros((n, n), dtype=np.float32) for key in alive_keys}
    with tsv_path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = (row["matrix"], int(row["subcomponent"]))
            if key in grids:
                grids[key][int(row["a"]) - 1, int(row["b"]) - 1] = float(row["inner_act"])
    return grids


def _dense_b64(grid: NDArray[Any]) -> str:
    """Row-major fp16 grid as base64 (keeps signed values)."""
    return base64.b64encode(np.ascontiguousarray(grid, dtype=np.float16).tobytes()).decode("ascii")


def build_neuron_investigator(
    model_path: ModelPath,
    op: str = "add",
    top_neurons: int = 512,
    output_dir: str | None = None,
) -> Path:
    checkpoint = Path(model_path).expanduser()
    assert checkpoint.exists(), f"checkpoint not found: {checkpoint}"
    run_dir = checkpoint.parent
    data_dir = analysis_datasets_dir(run_dir)

    alive = read_alive_components(data_dir / f"alive_filtered_{op}.tsv", keep_projs=MLP_MATRICES)
    mean_ci = _read_mean_ci(data_dir / f"alive_filtered_{op}.tsv")
    periods = read_subcomp_periods(data_dir / f"subcomp_periods_{op}.tsv")
    layer = alive[0].layer
    uv = load_component_uv(checkpoint, layer, MLP_MATRICES)

    npz_path = data_dir / f"hidden_activations_{op}.npz"
    assert npz_path.exists(), f"missing {npz_path.name}; run collect_hidden_activations first"
    hidden = np.load(npz_path, allow_pickle=True)
    n = int(hidden["a"].shape[0])
    up_grid = hidden["up_preact"]  # [N, N, d_int] float16
    gate_grid = hidden["gate_preact"]

    # Horizontal axis order: by period, then strongest mean CI first.
    alive = sorted(
        alive, key=lambda a: (periods[(a.proj, a.component)], -mean_ci[(a.proj, a.component)])
    )
    coeff = np.stack([_coeff_vector(a, uv) for a in alive])  # [n_subcomps, d_int], ≥ 0
    is_read = np.array([a.proj == "down_proj" for a in alive])

    # Vertical axis: top-K neurons by total coefficient across every subcomponent / matrix.
    total = coeff.sum(axis=0)  # [d_int]
    neuron_ids = np.argsort(-total)[:top_neurons]
    logger.info(
        f"{len(alive)} subcomponents, top {len(neuron_ids)}/{coeff.shape[1]} neurons by total coefficient"
    )

    # Signed coefficient matrix [subcomp, neuron]: write (gate/up) +, read (down) − → RdBu.
    signed = coeff[:, neuron_ids] * np.where(is_read, -1.0, 1.0)[:, None]

    alive_keys = {(a.proj, a.component) for a in alive}
    inner_grids = _inner_grids(data_dir / f"inner_activations_{op}.tsv", alive_keys, n)

    # Neuron up / gate grids reindexed to [K, N, N] (row-major [a, b]) for the right panel.
    up_k = np.transpose(up_grid[:, :, neuron_ids], (2, 0, 1)).astype(np.float16)
    gate_k = np.transpose(gate_grid[:, :, neuron_ids], (2, 0, 1)).astype(np.float16)

    payload: dict[str, Any] = {
        "meta": {
            "op": op,
            "symbol": op_symbol(op),
            "n": n,
            "layer": layer,
            "n_subcomps": len(alive),
            "n_neurons": len(neuron_ids),
            "page_size": len(alive),  # "as many neurons as fit" ≈ subcomponent count
        },
        "subcomps": [
            {
                "proj": a.proj,
                "c": a.component,
                "period": periods[(a.proj, a.component)],
                "mean_ci": round(mean_ci[(a.proj, a.component)], 4),
                "is_read": bool(a.proj == "down_proj"),
                "inner": _dense_b64(inner_grids[(a.proj, a.component)]),
            }
            for a in alive
        ],
        "neuron_ids": [int(j) for j in neuron_ids],
        "neuron_totals": [round(float(total[j]), 4) for j in neuron_ids],
        "coeff": _dense_b64(signed),  # [n_subcomps, n_neurons]
        "neuron_up": _dense_b64(up_k),  # [n_neurons, N, N]
        "neuron_gate": _dense_b64(gate_k),
    }

    out_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else analysis_dir(run_dir) / f"neuron_investigator_{op}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.js").write_text(
        f"window.PD_DATA = {json.dumps(payload, separators=(',', ':'))};\n"
    )
    assert _APP_TEMPLATE.exists(), f"app template missing: {_APP_TEMPLATE}"
    shutil.copyfile(_APP_TEMPLATE, out_dir / "index.html")

    size_mb = (out_dir / "data.js").stat().st_size / 1e6
    logger.info(f"wrote investigator to {out_dir} (data.js {size_mb:.1f} MB) — open index.html")
    return out_dir


if __name__ == "__main__":
    fire.Fire(build_neuron_investigator)
