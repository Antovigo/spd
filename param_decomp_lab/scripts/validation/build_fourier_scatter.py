"""Build a GPU-free HTML applet comparing subcomponents/neurons to the Fourier features.

For each canonical period of a chosen **basis task** (add / sub / mult) and **operand** (first
input operand `a`, second input operand `b`, or the output `result`), the applet scatters the
activations of a chosen **activation task** projected onto that period's 2D Fourier plane — so you
can e.g. plot subtraction activations on addition's basis. One plot per period, side by side.
Points colour (viridis) by `a`, `b`, `a+b`, `a-b`, `a×b`, the base model's **accuracy** (1 if the
argmax next token is the correct answer) or **P(correct)** (probability on the correct token) from
the shared `arithmetic_map`, or the **selected subcomponent's CI** — the arithmetic ones optionally
reduced by a `mod` + `offset` form (like the subspace-scatter applet) to `(value − offset) mod m`;
scroll to zoom, drag to pan.

Points are Feucht's probe coordinates: `(w_cos·x + b_cos, w_sin·x + b_sin)` = the predicted
`(cos, sin)`, so a clean feature traces the unit circle (for a degenerate sin axis — period 2 — the
2nd axis falls back to the top activation-variance direction ⊥ the cos direction). A **site**
dropdown picks where the probe was read: `mlp` (a/b at `mlp_input`, result at `mlp_output` — the
spaces the SPD subcomponents read/write) or `resid` (the residual stream, reproducing Feucht).

At the MLP site only, the subcomponent **unit** directions (gate/up `V` for input operands, down
`U` for the result) — and, via an overlay toggle, **individual neuron** directions (gate/up read
rows / down write columns of the frozen target weight) — are drawn as arrows in the same frame;
only those whose in-plane norm clears a typed threshold show. Clicking a subcomponent or neuron
arrowhead draws its **angle to every Fourier plane** and (for a subcomponent) opens its
inner-activation `(a, b)` heatmaps.

The probes are read from `find_fourier_features`' output (`coordinates_<task>.json` under
`<PARAM_DECOMP_OUT_DIR>/runs/fourier_features/` by default). Activation grids, the alive set,
periods (for the colour-mod options) and inner activations come from the run's `analysis/datasets/`.

CPU-only (no forward pass). Usage:
    python -m param_decomp_lab.scripts.validation.build_fourier_scatter <model_path> \
        [--coordinates-dir=PATH] [--ops=add,sub,mult] [--arrow-floor=0.1] [--output-dir=PATH]

Output: `<run_dir>/analysis/fourier_scatter/{index.html,data.js}`.
"""

import base64
import csv
import json
from pathlib import Path
from typing import Any

import fire
import numpy as np
from numpy.typing import NDArray

from param_decomp.log import logger
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.scripts.validation.common import (
    MLP_MATRICES,
    analysis_datasets_dir,
    analysis_dir,
    load_component_uv,
    load_target_mlp_weights,
    op_symbol,
    read_alive_components,
    read_subcomp_period_groups,
)

_APP_TEMPLATE = Path(__file__).with_name("fourier_scatter_app.html")
_CANDIDATE_OPS = ("add", "sub", "mult")
# map_arithmetic condition (in the shared arithmetic_map/results.tsv) whose prompt symbol matches
# each op, for the "model accuracy" (P of the correct answer token) colour option.
_ACCURACY_CONDITION = {"add": "digit_add_plus", "sub": "digit_sub_minus", "mult": "digit_mul_times"}
_OPERANDS = ("a", "b", "result")
# activation grid each operand is probed at, per site (matches find_fourier_features._site_grids).
_SITE_GRID = {
    "mlp": {"a": "mlp_input", "b": "mlp_input", "result": "mlp_output"},
    "resid": {"a": "resid_pre_mlp", "b": "resid_pre_mlp", "result": "resid_post"},
}
# subcomponent / neuron proj set whose direction each operand's arrows use (MLP site only): input
# operands read via gate/up `V`, the result writes via down `U`.
_OPERAND_PROJS = {
    "a": ("gate_proj", "up_proj"),
    "b": ("gate_proj", "up_proj"),
    "result": ("down_proj",),
}


def _silu(x: NDArray[np.float32]) -> NDArray[np.float32]:
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def _b64(arr: NDArray[Any]) -> str:
    """Row-major fp16 array as base64 (keeps signed values)."""
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float16).tobytes()).decode("ascii")


def _read_ci_grids(
    tsv_path: Path, wanted: set[tuple[str, int]], n: int
) -> dict[tuple[str, int], NDArray[np.float32]]:
    """Per-(proj, component) last-token CI `(a, b)` grid from an `inner_activations_<op>.tsv`
    `ci` column, for the wanted keys. Returns `{}` if the file is absent or predates the `ci`
    column (so the applet's CI colouring degrades gracefully)."""
    if not tsv_path.exists():
        return {}
    # NaN = not recorded for this task (component filtered out of its TSV) → greyed in the applet,
    # distinct from a genuine CI of 0.
    grids = {k: np.full((n, n), np.nan, dtype=np.float32) for k in wanted}
    with tsv_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "ci" not in (reader.fieldnames or []):
            return {}
        for row in reader:
            key = (row["matrix"], int(row["subcomponent"]))
            if key in grids:
                grids[key][int(row["a"]) - 1, int(row["b"]) - 1] = float(row["ci"])
    return grids


def _accuracy_by_op(
    op_list: list[str], a_grid: NDArray[np.integer], b_grid: NDArray[np.integer]
) -> dict[str, dict[str, list[float]]]:
    """Per-op base-model performance over the (a, b) grid, read from the shared
    `arithmetic_map/results.tsv` and aligned to the applet's point order via `(a, b)` lookup:
    `accuracy` (1 if the argmax next token is the correct answer, else 0) and `p_correct` (the
    probability mass on the correct token). Empty for ops whose condition is absent."""
    results = PARAM_DECOMP_OUT_DIR / "arithmetic_map" / "results.tsv"
    if not results.exists():
        return {}
    wanted = {_ACCURACY_CONDITION[op]: op for op in op_list if op in _ACCURACY_CONDITION}
    cells: dict[str, dict[tuple[int, int], tuple[float, float]]] = {
        op: {} for op in wanted.values()
    }
    with results.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            op = wanted.get(row["condition"])
            if op is not None:
                cells[op][(int(row["a"]), int(row["b"]))] = (
                    float(row["correct"]),
                    float(row["p_correct"]),
                )
    ab = list(zip(a_grid, b_grid, strict=True))
    return {
        op: {
            "accuracy": [round(grid[(int(a), int(b))][0], 4) for a, b in ab],
            "p_correct": [round(grid[(int(a), int(b))][1], 4) for a, b in ab],
        }
        for op, grid in cells.items()
        if grid
    }


def _probe_plane(probe: dict[str, Any], fallback_acts: NDArray[np.float32]) -> dict[str, Any]:
    """Projection frame for one Feucht probe `[cos,sin] ≈ x·[w_cos,w_sin] + [b_cos,b_sin]`.

    `mode="pred"`: point/direction coords are the predicted `(cos, sin)` (`x·w (+b)`), so a clean
    feature traces the unit circle — Feucht's plot. `mode="ortho"`: an orthonormal fallback used
    when the sin axis is degenerate (period 2, `sin(2πv/2)=0` so `w_sin≈0` and the circle collapses
    to the `e1` line) — `e2` is then the direction of most `fallback_acts` variance ⊥ `e1`. `e1,e2`
    are always an orthonormal basis of the plane, used for the in-plane norm and plane angles."""
    w_cos = np.asarray(probe["w_cos"], dtype=np.float32)
    w_sin = np.asarray(probe["w_sin"], dtype=np.float32)
    n1 = float(np.linalg.norm(w_cos))
    e1 = (w_cos / max(n1, 1e-12)).astype(np.float32)
    resid = w_sin - float(w_sin @ e1) * e1
    if float(np.linalg.norm(resid)) > 1e-3 * max(n1, 1e-12):
        e2 = (resid / np.linalg.norm(resid)).astype(np.float32)
        return {
            "mode": "pred",
            "w_cos": w_cos,
            "b_cos": float(probe["b_cos"]),
            "w_sin": w_sin,
            "b_sin": float(probe["b_sin"]),
            "e1": e1,
            "e2": e2,
        }
    xc = fallback_acts - fallback_acts.mean(axis=0)
    xc = xc - np.outer(xc @ e1, e1)  # variance orthogonal to e1
    top = np.linalg.eigh(xc.T @ xc)[1][:, -1].astype(np.float32)
    top = top - float(top @ e1) * e1
    e2 = (top / max(float(np.linalg.norm(top)), 1e-12)).astype(np.float32)
    # The fallback axis (top activation variance) usually dwarfs the cos-axis spread, hiding the
    # residue separation. Rescale the drawn e2 coord to the cos-axis spread so both stay visible.
    s1 = float(np.std(fallback_acts @ e1))
    s2 = float(np.std(fallback_acts @ e2))
    return {"mode": "ortho", "e1": e1, "e2": e2, "e2scale": s1 / max(s2, 1e-12)}


def _project_points(plane: dict[str, Any], x: NDArray[np.float32]) -> NDArray[np.float32]:
    if plane["mode"] == "pred":
        return np.stack(
            [x @ plane["w_cos"] + plane["b_cos"], x @ plane["w_sin"] + plane["b_sin"]], axis=1
        ).astype(np.float32)
    return np.stack([x @ plane["e1"], (x @ plane["e2"]) * plane["e2scale"]], axis=1).astype(
        np.float32
    )


def _project_dirs(
    plane: dict[str, Any], dirs: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Unit-normalise `dirs`; return drawn coords (in the plane's mode, sharing the point frame) and
    the orthonormal in-plane norm `sqrt((·e1)²+(·e2)²) ∈ [0,1]` (for the arrow floor / angle — this
    stays the true orthonormal norm even when the drawn e2 coord is rescaled)."""
    unit = dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    inplane = np.sqrt((unit @ plane["e1"]) ** 2 + (unit @ plane["e2"]) ** 2).astype(np.float32)
    if plane["mode"] == "pred":
        coords = np.stack([unit @ plane["w_cos"], unit @ plane["w_sin"]], axis=1).astype(np.float32)
    else:
        coords = np.stack(
            [unit @ plane["e1"], (unit @ plane["e2"]) * plane["e2scale"]], axis=1
        ).astype(np.float32)
    return coords, inplane


def _resolve_ops(data_dir: Path, coord_dir: Path, ops: str | tuple[str, ...] | None) -> list[str]:
    if ops is not None:
        # fire parses `--ops=add,sub` as a tuple and `--ops=add` as a str.
        return list(ops) if isinstance(ops, (list, tuple)) else str(ops).split(",")
    detected = [
        op
        for op in _CANDIDATE_OPS
        if (data_dir / f"hidden_activations_{op}.npz").exists()
        and (coord_dir / f"coordinates_{op}.json").exists()
    ]
    assert detected, (
        f"no task with both hidden_activations_<op>.npz ({data_dir}) and a Fourier basis ({coord_dir})"
    )
    return detected


def build_fourier_scatter(
    model_path: ModelPath,
    coordinates_dir: str | None = None,
    ops: str | tuple[str, ...] | None = None,
    arrow_floor: float = 0.1,
    output_dir: str | None = None,
) -> Path:
    checkpoint = Path(model_path).expanduser()
    assert checkpoint.exists(), f"checkpoint not found: {checkpoint}"
    run_dir = checkpoint.parent
    data_dir = analysis_datasets_dir(run_dir)
    coord_dir = (
        Path(coordinates_dir).expanduser()
        if coordinates_dir
        else PARAM_DECOMP_OUT_DIR / "runs" / "fourier_features"
    )
    op_list = _resolve_ops(data_dir, coord_dir, ops)
    logger.info(f"tasks: {op_list} (bases from {coord_dir})")

    bases = {op: json.loads((coord_dir / f"coordinates_{op}.json").read_text()) for op in op_list}
    alive_by_op = {
        op: read_alive_components(data_dir / f"alive_filtered_{op}.tsv", keep_projs=MLP_MATRICES)
        for op in op_list
    }
    layer = alive_by_op[op_list[0]][0].layer
    for op in op_list:
        assert "sites" in bases[op] and "space" in bases[op], (
            f"coordinates_{op}.json is the old schema; refit find_fourier_features"
        )
        assert bases[op]["layer"] == layer, (
            f"coordinates_{op}.json was fit at layer {bases[op]['layer']} but the checkpoint's "
            f"alive components are layer {layer}; the projection plane would be from the wrong space"
        )
    uv = load_component_uv(checkpoint, layer, MLP_MATRICES)
    weights = load_target_mlp_weights(checkpoint, layer, MLP_MATRICES)

    # Per-task colour-modulo options (matching the subspace-scatter applet): integer residues, or
    # the detected log ratios for a task whose subcomponents are mostly log-periodic (mult).
    task_mods: dict[str, dict[str, Any]] = {}
    for op in op_list:
        ppath = data_dir / f"subcomp_periods_{op}.tsv"
        groups = list(read_subcomp_period_groups(ppath).values()) if ppath.exists() else []
        n_log = sum(g.kind == "log" for g in groups)
        if n_log > sum(g.kind == "additive" for g in groups):
            task_mods[op] = {
                "kind": "log",
                "values": sorted({round(g.value, 2) for g in groups if g.kind == "log"}),
            }
        else:
            task_mods[op] = {"kind": "additive", "values": [2, 5, 10, 20, 25, 50, 100]}

    # Activation grids per task: mlp_input / mlp_output flattened, plus the post-SwiGLU neuron
    # output (for down-subcomponent inner activations).
    hidden = {
        op: np.load(data_dir / f"hidden_activations_{op}.npz", allow_pickle=True) for op in op_list
    }
    n = int(hidden[op_list[0]]["a"].shape[0])
    # Post-SwiGLU output is only needed for down-subcomponent inner activations; skip it entirely
    # (a per-task [n*n, d_int] fp32 buffer) when no down_proj component is alive.
    need_swiglu = any(a.proj == "down_proj" for comps in alive_by_op.values() for a in comps)
    acts: dict[str, dict[str, NDArray[np.float32]]] = {}
    for op in op_list:
        z = hidden[op]
        assert int(z["a"].shape[0]) == n, f"grid size mismatch for {op}"
        mlp_output = z["mlp_output"].reshape(n * n, -1).astype(np.float32)
        resid_pre = z["resid_pre_mlp"].reshape(n * n, -1).astype(np.float32)
        acts[op] = {
            "mlp_input": z["mlp_input"].reshape(n * n, -1).astype(np.float32),
            "mlp_output": mlp_output,
            "resid_pre_mlp": resid_pre,
            "resid_post": resid_pre + mlp_output,  # residual after this layer's MLP write
        }
        if need_swiglu:
            d_int = z["up_preact"].shape[-1]
            acts[op]["swiglu"] = _silu(
                z["gate_preact"].reshape(n * n, d_int).astype(np.float32)
            ) * z["up_preact"].reshape(n * n, d_int).astype(np.float32)

    # Union of alive subcomponents across tasks, split by which vector defines their direction.
    union = sorted({(a.proj, a.component) for comps in alive_by_op.values() for a in comps})
    sub_id = {key: i for i, key in enumerate(union)}
    # Unit direction of each subcomponent (V for gate/up, U for down) in its side's space.
    sub_dir = {}
    for proj, c in union:
        v, u = uv[proj]
        vec = v[:, c] if proj != "down_proj" else u[c, :]
        sub_dir[(proj, c)] = vec.astype(np.float32)

    sites = list(bases[op_list[0]]["sites"])
    kept_subs: set[int] = set()  # subcomponents that clear the floor in at least one plot
    shown_neurons: set[tuple[str, int]] = set()  # (proj, idx) drawn as an arrow in some plot
    plane_cache: dict[
        tuple[str, str, str], dict[str, Any]
    ] = {}  # MLP-site planes for arrows/angles
    periods_out: dict[str, dict[str, list[float]]] = {}
    points_out: dict[str, dict[str, dict[str, dict[str, list[str]]]]] = {s: {} for s in sites}
    centers_out: dict[str, dict[str, dict[str, list[list[float]]]]] = {s: {} for s in sites}
    subcomp_out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    neurons_out: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for basis_op in op_list:
        periods_out[basis_op] = {}
        subcomp_out[basis_op], neurons_out[basis_op] = {}, {}
        for s in sites:
            points_out[s][basis_op], centers_out[s][basis_op] = {}, {}
        for operand in _OPERANDS:
            feats_site = {s: bases[basis_op]["features"][s][operand] for s in sites}
            period_keys = sorted(feats_site[sites[0]], key=float)
            periods_out[basis_op][operand] = [float(p) for p in period_keys]

            # Points + circle centre at every site (both reproduce Feucht's plot / the coherent-MLP
            # plot); the basis task's own activations seed the degenerate-plane fallback axis.
            for s in sites:
                grid_key = _SITE_GRID[s][operand]
                basis_mean = acts[basis_op][grid_key].mean(axis=0, keepdims=True)
                points_out[s][basis_op][operand] = {op: [] for op in op_list}
                centers_out[s][basis_op][operand] = []
                for pk in period_keys:
                    plane = _probe_plane(feats_site[s][pk], acts[basis_op][grid_key])
                    if s == "mlp":
                        plane_cache[(basis_op, operand, pk)] = plane
                    centers_out[s][basis_op][operand].append(
                        [round(float(c), 4) for c in _project_points(plane, basis_mean)[0]]
                    )
                    for op in op_list:
                        points_out[s][basis_op][operand][op].append(
                            _b64(_project_points(plane, acts[op][grid_key]))
                        )

            # Subcomponent / neuron arrows: MLP site only (their V/U directions live in MLP space).
            subcomp_out[basis_op][operand] = []
            neurons_out[basis_op][operand] = []
            projs = _OPERAND_PROJS[operand]
            keys = [(p, c) for (p, c) in union if p in projs]
            for pk in period_keys:
                plane = plane_cache[(basis_op, operand, pk)]
                if keys:
                    coords, inplane = _project_dirs(plane, np.stack([sub_dir[k] for k in keys]))
                    mask = inplane >= arrow_floor
                    ids = [sub_id[keys[i]] for i in np.nonzero(mask)[0]]
                else:
                    ids, coords, inplane, mask = (
                        [],
                        np.zeros((0, 2), np.float32),
                        np.zeros(0),
                        np.zeros(0, bool),
                    )
                kept_subs.update(ids)
                subcomp_out[basis_op][operand].append(
                    {"ids": ids, "xy": _b64(coords[mask]), "norm": _b64(inplane[mask])}
                )

                per_proj: dict[str, Any] = {}
                for proj in projs:
                    w = weights[proj]  # [d_out, d_in]
                    ndirs = w if proj != "down_proj" else w.T  # neuron dir per row
                    coords, inplane = _project_dirs(plane, ndirs)
                    mask = inplane >= arrow_floor
                    shown = [int(i) for i in np.nonzero(mask)[0]]
                    shown_neurons.update((proj, i) for i in shown)
                    per_proj[proj] = {
                        "ids": shown,
                        "xy": _b64(coords[mask]),
                        "norm": _b64(inplane[mask]),
                    }
                neurons_out[basis_op][operand].append(per_proj)

    kept = sorted(kept_subs)
    kept_pos = {sid: i for i, sid in enumerate(kept)}

    # Angle (deg, 0 = lies in the plane) from each selectable direction to every MLP-site Fourier
    # plane, keyed by "<basis>|<operand>" → [angle per period] (aligned with periods_out order).
    # Shown on selection; uses the full (un-floored) projection so orthogonal planes are included.
    neuron_dir = {
        (proj, idx): (weights[proj][idx] if proj != "down_proj" else weights[proj].T[idx]).astype(
            np.float32
        )
        for proj, idx in shown_neurons
    }
    subcomp_angles: dict[str, dict[str, list[float]]] = {}
    neuron_angles: dict[str, dict[str, list[float]]] = {}
    for basis_op in op_list:
        for operand in _OPERANDS:
            series = f"{basis_op}|{operand}"
            projs = _OPERAND_PROJS[operand]
            period_keys = sorted(bases[basis_op]["features"]["mlp"][operand], key=float)
            sub_keys = [k for k in union if k[0] in projs and sub_id[k] in kept_subs]
            neur_keys = [k for k in shown_neurons if k[0] in projs]
            sub_arr = np.stack([sub_dir[k] for k in sub_keys]) if sub_keys else None
            neur_arr = np.stack([neuron_dir[k] for k in neur_keys]) if neur_keys else None
            for pk in period_keys:
                plane = plane_cache[(basis_op, operand, pk)]
                for keys, arr, table, idfmt in (
                    (sub_keys, sub_arr, subcomp_angles, lambda k: str(sub_id[k])),
                    (neur_keys, neur_arr, neuron_angles, lambda k: f"{k[0]}:{k[1]}"),
                ):
                    if arr is None:
                        continue
                    _, inplane = _project_dirs(plane, arr)
                    deg = np.degrees(np.arccos(np.clip(inplane, 0.0, 1.0)))
                    for k, d in zip(keys, deg, strict=True):
                        table.setdefault(idfmt(k), {}).setdefault(series, []).append(
                            round(float(d), 1)
                        )

    # Inner-activation (a, b) grids for the kept subcomponents, one stack per task (click panel).
    inner: dict[str, str] = {}
    for op in op_list:
        stack = np.zeros((len(kept), n, n), dtype=np.float32)
        for sid in kept:
            proj, c = union[sid]
            v, _ = uv[proj]
            vn = v[:, c].astype(np.float32)
            vn = vn / max(float(np.linalg.norm(vn)), 1e-12)
            src = acts[op]["swiglu"] if proj == "down_proj" else acts[op]["mlp_input"]
            stack[kept_pos[sid]] = (src @ vn).reshape(n, n)
        inner[op] = _b64(stack)

    # Per-task CI (a, b) grid per kept subcomponent (for the "CI of selected" point colouring),
    # read from the inner-activations TSVs' `ci` column. Per-task: a task whose TSV predates the
    # column is simply omitted (the applet greys CI colouring for it) rather than disabling it for
    # every task. Missing components within a task are NaN (greyed), distinct from a true CI of 0.
    kept_keys = {union[sid] for sid in kept}
    ci_by_op: dict[str, str] = {}
    for op in op_list:
        grids = _read_ci_grids(data_dir / f"inner_activations_{op}.tsv", kept_keys, n)
        if not grids:
            continue
        stack = np.stack([grids[union[sid]] for sid in kept]) if kept else np.zeros((0, n, n))
        ci_by_op[op] = _b64(stack)
    logger.info(f"CI colour data present for: {', '.join(sorted(ci_by_op)) or 'no task'}")

    a_grid = np.repeat(np.arange(1, n + 1), n).astype(np.int32)
    b_grid = np.tile(np.arange(1, n + 1), n).astype(np.int32)

    acc_by_op = _accuracy_by_op(op_list, a_grid, b_grid)
    logger.info(
        f"model-accuracy colour data present for: {', '.join(sorted(acc_by_op)) or 'no task'}"
    )

    payload = {
        "meta": {
            "tasks": op_list,
            "n": n,
            "layer": int(layer),
            "symbol": {op: op_symbol(op) for op in op_list},
            "operands": list(_OPERANDS),
            "sites": sites,
            "arrow_floor": arrow_floor,
            "task_mods": task_mods,
            "has_ci": bool(ci_by_op),
            "has_accuracy": bool(acc_by_op),
            "space": {op: bases[op]["space"] for op in op_list},
            "a": [int(v) for v in a_grid],
            "b": [int(v) for v in b_grid],
            "accuracy": {op: d["accuracy"] for op, d in acc_by_op.items()},
            "p_correct": {op: d["p_correct"] for op, d in acc_by_op.items()},
        },
        "periods": periods_out,
        "points": points_out,
        "centers": centers_out,
        "subcomp": subcomp_out,
        "neurons": neurons_out,
        "subcomp_list": [
            {
                "id": sid,
                "proj": union[sid][0],
                "c": union[sid][1],
                "label": f"{union[sid][0][0]}{union[sid][1]}",
            }
            for sid in kept
        ],
        "subcomp_pos": {str(sid): kept_pos[sid] for sid in kept},
        "inner": inner,
        "ci": ci_by_op,
        "subcomp_angles": subcomp_angles,
        "neuron_angles": neuron_angles,
    }

    out_dir = (
        Path(output_dir).expanduser() if output_dir else analysis_dir(run_dir) / "fourier_scatter"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.js").write_text(
        f"window.PD_DATA = {json.dumps(payload, separators=(',', ':'))};\n"
    )
    assert _APP_TEMPLATE.exists(), f"app template missing: {_APP_TEMPLATE}"
    (out_dir / "index.html").write_text(_APP_TEMPLATE.read_text())

    size_mb = (out_dir / "data.js").stat().st_size / 1e6
    logger.info(
        f"wrote fourier-scatter applet ({', '.join(op_list)}; {len(kept)} kept subcomps) "
        f"→ {out_dir} (data.js {size_mb:.1f} MB)"
    )
    return out_dir


if __name__ == "__main__":
    fire.Fire(build_fourier_scatter)
