"""Build a GPU-free HTML app to investigate which neurons take part in an arithmetic task.

For L18's MLP it cross-tabulates every filtered-alive subcomponent against the hidden
neurons it couples to, via the per-(subcomponent, neuron) **interaction score** — the
standard deviation, over the target grid, of what the subcomponent writes to / reads from
that neuron (always ≥ 0):
- gate / up (pre-SwiGLU, *write*): `std(inner_act_c) · ||V_c|| · |U[c, j]|` — the std over the
  100×100 grid of the contribution `(x·V_c)·U[c,j]` the subcomponent adds to neuron `j`,
- down (post-SwiGLU, *read*): `std_grid(silu(gate_j)·up_j) · |V[j, c]| / ||V_c||` — the neuron's
  post-SwiGLU-activation std times the subcomponent's unit read weight.

The applet's left half is a neuron × subcomponent heatmap. Subcomponents (columns) are
ordered by period group (additive `p10` < multiplicative `×1.27` < none), then matrix
(gate > up > down), then the confidence the period is correct (the chosen fit's CV R²) — with
thick delimiters between period groups and thin ones between matrices, and period band labels
above the names. Neurons (rows) are ordered by total
interaction score per frequency (grouped by the period they couple to most strongly, then by
that coupling), paged 50 at a time. A neuron filter hides those whose total interaction score —
over input (gate/up), output (down), or all subcomponents per a dropdown — is below a typed
threshold. Write scores render blue, read scores red — done by flipping the sign of down (read)
columns and colouring with a diverging RdBu scale.
Clicking a cell selects that (neuron, subcomponent) pair; the right half then lays the
subcomponent's inner-activation `(a, b)` heatmap and the neuron's up / gate / post-SwiGLU-output
`(a, b)` heatmaps, each with its own colour scale. For a *write* (gate/up) subcomponent it adds
two more `(a, b)` grids: the subcomponent's contribution to the neuron's gate / up preactivation
(`inner_act_c · ||V_c|| · U[c, j]`) and the counterfactual preactivation with that contribution
removed (final − contribution) — i.e. what the neuron's gate / up would be without the
subcomponent. (Not drawn for *read* (down) subcomponents, where it isn't defined.) An operation
toggle re-renders every grid on a *different* task's activations (defaulting to the build op,
switchable to any operation with a saved npz) — to see how the same neuron / subcomponent
responds to add / sub / mult.

Only the top-`top_neurons` neurons by total interaction score are kept — their per-op up / gate
grids (needed for the right panel) are the bulk of the payload, so the cap bounds the file size.

CPU-only: reads the checkpoint U/V (mmap) plus, from `<run>/analysis/datasets/`, the
filtered-alive list (`alive_filtered_<op>.tsv`, for mean CI), the periods
(`subcomp_periods_<op>.tsv`), the build op's inner-activation TSV (`inner_activations_<op>.tsv`,
for the score), and one hidden-activation npz per operation (`hidden_activations_<o>.npz`, for
the per-op neuron / subcomponent grids). No forward pass.

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
    PeriodGroup,
    analysis_datasets_dir,
    analysis_dir,
    load_component_uv,
    op_symbol,
    read_alive_components,
    read_subcomp_period_groups,
)

_APP_TEMPLATE = Path(__file__).with_name("neuron_investigator_app.html")


def _read_mean_ci(tsv_path: Path) -> dict[tuple[str, int], float]:
    """`(proj, component) -> mean CI` from an `alive_filtered_<op>.tsv`."""
    out: dict[tuple[str, int], float] = {}
    with tsv_path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out[(row["matrix"].split(".")[-1], int(row["component"]))] = float(row["mean_ci"])
    return out


def _read_period_confidence(tsv_path: Path) -> dict[tuple[str, int], float]:
    """`(proj, component) -> confidence the detected period is real`, from `subcomp_periods`.

    The cross-validated R² of the chosen periodic fit (`additive_cvr2` / `log_cvr2` by
    `period_type`; the larger of the two when no type wins, which reads as low confidence).
    """
    out: dict[tuple[str, int], float] = {}
    with tsv_path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            add_r2, log_r2 = float(row["additive_cvr2"] or 0), float(row["log_cvr2"] or 0)
            kind = row.get("period_type") or "additive"
            conf = (
                add_r2 if kind == "additive" else log_r2 if kind == "log" else max(add_r2, log_r2)
            )
            out[(row["matrix"].split(".")[-1], int(row["component"]))] = conf
    return out


def _silu(x: NDArray[np.float32]) -> NDArray[np.float32]:
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def _period_label(pg: PeriodGroup) -> str:
    """Compact band label: `p10` (additive), `×1.27` (log multiplicative), `—` (none)."""
    if pg.kind == "additive":
        return f"p{int(pg.value)}"
    if pg.kind == "log":
        return f"×{pg.value:g}"
    return "—"


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


def _op_panel_grids(
    npz_path: Path,
    neuron_ids: NDArray[np.intp],
    alive: list[Any],
    uv: dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]],
    n: int,
) -> dict[str, str]:
    """Right-panel grids for one operation: the kept neurons' up/gate `(a, b)` grids and every
    build-op subcomponent's inner-activation `(a, b)` grid, computed on *this* op's activations
    (so the toggle shows how the same neuron / subcomponent responds to a different task).

    inner = (op input) · (V_c / ||V_c||): the post-RMSNorm MLP input for write (up/gate)
    subcomponents, the post-SwiGLU activation `silu(gate)·up` for read (down) ones.
    """
    z = np.load(npz_path, allow_pickle=True)
    up, gate, mlp_in = z["up_preact"], z["gate_preact"], z["mlp_input"]
    assert int(z["a"].shape[0]) == n, f"{npz_path.name}: grid side {z['a'].shape[0]} != {n}"
    d_int = up.shape[-1]
    up_k = np.transpose(up[:, :, neuron_ids], (2, 0, 1)).astype(np.float16)
    gate_k = np.transpose(gate[:, :, neuron_ids], (2, 0, 1)).astype(np.float16)

    mlp_in32 = mlp_in.astype(np.float32)  # [N, N, d_model] — write subcomponents read this
    # post-SwiGLU activation [N*N, d_int] — read (down) subcomponents read this; only built if needed.
    swiglu = (
        _silu(gate.reshape(n * n, d_int).astype(np.float32))
        * up.reshape(n * n, d_int).astype(np.float32)
        if any(a.proj == "down_proj" for a in alive)
        else None
    )
    inner = np.zeros((len(alive), n, n), dtype=np.float32)
    for i, a in enumerate(alive):
        v, _ = uv[a.proj]
        vn = v[:, a.component].astype(np.float32)
        vn = vn / max(float(np.linalg.norm(vn)), 1e-12)
        if a.proj == "down_proj":
            assert swiglu is not None
            inner[i] = (swiglu @ vn).reshape(n, n)
        else:
            inner[i] = mlp_in32 @ vn
    return {"up": _dense_b64(up_k), "gate": _dense_b64(gate_k), "inner": _dense_b64(inner)}


_OP_CANDIDATES = ("add", "sub", "mult")


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

    # Operations the right-panel toggle can switch to: those with a saved hidden-activation npz.
    available_ops = [
        o for o in _OP_CANDIDATES if (data_dir / f"hidden_activations_{o}.npz").exists()
    ]
    assert op in available_ops, f"build op {op!r} has no hidden_activations_{op}.npz in {data_dir}"

    alive = read_alive_components(data_dir / f"alive_filtered_{op}.tsv", keep_projs=MLP_MATRICES)
    mean_ci = _read_mean_ci(data_dir / f"alive_filtered_{op}.tsv")
    periods = read_subcomp_period_groups(data_dir / f"subcomp_periods_{op}.tsv")
    confidence = _read_period_confidence(data_dir / f"subcomp_periods_{op}.tsv")
    layer = alive[0].layer
    uv = load_component_uv(checkpoint, layer, MLP_MATRICES)

    npz_path = data_dir / f"hidden_activations_{op}.npz"
    assert npz_path.exists(), f"missing {npz_path.name}; run collect_hidden_activations first"
    hidden = np.load(npz_path, allow_pickle=True)
    n = int(hidden["a"].shape[0])
    up_grid = hidden["up_preact"]  # [N, N, d_int] float16
    gate_grid = hidden["gate_preact"]

    # Horizontal axis order: by period group, then strongest mean CI first.
    alive = sorted(
        alive,
        key=lambda a: (periods[(a.proj, a.component)].sort_key, -mean_ci[(a.proj, a.component)]),
    )
    is_read = np.array([a.proj == "down_proj" for a in alive])
    alive_keys = {(a.proj, a.component) for a in alive}
    inner_grids = _inner_grids(data_dir / f"inner_activations_{op}.tsv", alive_keys, n)

    # Interaction score [n_subcomps, d_int], ≥ 0: the fraction of the *target's* variance over the
    # grid that this subcomponent↔neuron rank-1 term explains, so writes (into a neuron preact) and
    # reads (from a neuron into a subcomponent inner act) are on one comparable scale. Write term
    # std = std((x·V_c)·U[c,j]) ÷ std(neuron j's own gate/up preactivation); read term std =
    # std(swiglu_j·V[j,c]) ÷ std(the down read total) = read magnitude ÷ std(inner_act_c).
    d_int = up_grid.shape[-1]
    gate_flat = gate_grid.reshape(n * n, d_int).astype(np.float32)
    up_flat = up_grid.reshape(n * n, d_int).astype(np.float32)
    neuron_h_std = (_silu(gate_flat) * up_flat).std(axis=0)  # [d_int]; post-SwiGLU read magnitude
    preact_std = {
        "gate_proj": np.maximum(gate_flat.std(axis=0), 1e-12),
        "up_proj": np.maximum(up_flat.std(axis=0), 1e-12),
    }
    score = np.zeros((len(alive), d_int), dtype=np.float32)
    # Signed per-neuron write weight ||V_c||·U[c,j] for input (gate/up) subcomponents (0 for reads):
    # the subcomponent's contribution to neuron j is inner_act_c (normalised) · this weight.
    write_weight = np.zeros((len(alive), d_int), dtype=np.float32)
    for i, a in enumerate(alive):
        v, u = uv[a.proj]
        v_norm = max(float(np.linalg.norm(v[:, a.component])), 1e-12)
        inner_std = max(float(inner_grids[(a.proj, a.component)].std()), 1e-12)
        if a.proj == "down_proj":  # read: neuron contribution to inner_c ÷ std(inner_c)
            score[i] = neuron_h_std * (np.abs(v[:, a.component]) / v_norm) / inner_std
        else:  # write: contribution to neuron j's preact ÷ std of that neuron's preactivation
            write_weight[i] = v_norm * u[a.component, :]
            score[i] = inner_std * np.abs(write_weight[i]) / preact_std[a.proj]

    # Vertical axis: top-K neurons by total interaction score across every subcomponent / matrix.
    total = score.sum(axis=0)  # [d_int]
    neuron_ids = np.argsort(-total)[:top_neurons]
    logger.info(
        f"{len(alive)} subcomponents, top {len(neuron_ids)}/{d_int} neurons by total interaction score"
    )
    del gate_flat, up_flat  # free the [N*N, d_int] buffers before the per-op panel pass

    # Signed score matrix [subcomp, neuron]: write (gate/up) +, read (down) − → RdBu.
    signed = score[:, neuron_ids] * np.where(is_read, -1.0, 1.0)[:, None]
    write_weight_k = write_weight[:, neuron_ids]  # [subcomp, neuron] signed ||V_c||·U[c,j]

    # Right-panel grids per available op (the build op plus any other task with a saved npz), so
    # the toggle can show how the focused neuron / subcomponent responds to a different operation.
    ops_grids = {
        o: _op_panel_grids(data_dir / f"hidden_activations_{o}.npz", neuron_ids, alive, uv, n)
        for o in available_ops
    }

    payload: dict[str, Any] = {
        "meta": {
            "op": op,
            "symbol": op_symbol(op),
            "n": n,
            "layer": layer,
            "n_subcomps": len(alive),
            "n_neurons": len(neuron_ids),
            "page_size": 50,  # neurons (rows) per page
            "ops": [{"op": o, "symbol": op_symbol(o)} for o in available_ops],
        },
        "subcomps": [
            {
                "proj": a.proj,
                "c": a.component,
                "period_label": _period_label(periods[(a.proj, a.component)]),
                # one sortable scalar per period group: additive < log < none, then by value.
                "period_sort": periods[(a.proj, a.component)].sort_key[0] * 1e6
                + periods[(a.proj, a.component)].sort_key[1],
                "mean_ci": round(mean_ci[(a.proj, a.component)], 4),
                "confidence": round(confidence[(a.proj, a.component)], 4),
                "is_read": bool(a.proj == "down_proj"),
            }
            for a in alive
        ],
        "neuron_ids": [int(j) for j in neuron_ids],
        "neuron_totals": [round(float(total[j]), 4) for j in neuron_ids],
        "score": _dense_b64(signed),  # [n_subcomps, n_neurons] signed interaction score
        # signed write weight ||V_c||·U[c,j]; contribution grid = (per-op inner) · this (input only).
        "write_weight": _dense_b64(write_weight_k),
        # per op: {up, gate: [n_neurons, N, N]; inner: [n_subcomps, N, N]} — the right panel.
        "ops": ops_grids,
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
