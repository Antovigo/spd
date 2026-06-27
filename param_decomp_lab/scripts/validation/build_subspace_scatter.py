"""Interactive 3D scatter of L18 MLP activations in a user-picked 3-subcomponent subspace.

A GPU-free HTML applet spanning every available task (add / sub / mult). The user picks up to
3 subcomponents from a thumbnail list — organised into high-level **task** sections, then by
period — and a points selector chooses which single task's last-token activations are
scattered (as a rotatable 3D plot) onto those 3 directions:

- **input**: the post-RMSNorm MLP input (`mlp_input`) projected onto the unit V directions of
  the up/gate subcomponents (`x · V̂_c`).
- **pre-nonlinearity**: each up/gate subcomponent's own preactivation (`up_preact` / `gate_preact`)
  projected onto its unit U direction — what the matrix writes to the neurons, pre-SwiGLU.
- **post-nonlinearity**: the post-SwiGLU neuron output (`silu(gate)·up`) projected onto the unit
  V directions of the down subcomponents — what the down matrix reads.
- **output**: the MLP output (`mlp_output`) projected onto the unit U directions of the down
  subcomponents (`y · Û_c`).

Tasks are auto-detected from the run's `analysis/datasets/` (those with a
`hidden_activations_<op>.npz` and `alive_filtered_<op>.tsv`). Periods come from
`subcomp_periods_<op>.tsv` when present;
subcomponents without an assigned period are grouped under "no period". Each direction's sign
(an arbitrary gauge) is flipped so the median projection is positive (its arrow points toward
the data). The directions are kept at their true mutual angles in the applet (Cholesky
embedding) and a dark-grey shadow is the points flattened onto the box floor.

CPU-only. Usage:
    python -m param_decomp_lab.scripts.validation.build_subspace_scatter <model_path> \
        [--ops=add,mult] [--output-dir=PATH]

Output: `<run_dir>/analysis/subspace_scatter/index.html` (self-contained Plotly applet).
"""

import base64
import io
import json
from pathlib import Path
from typing import Any

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.offline as pyo  # noqa: E402
from numpy.typing import NDArray  # noqa: E402

from param_decomp.log import logger  # noqa: E402
from param_decomp_lab.infra.paths import ModelPath  # noqa: E402
from param_decomp_lab.scripts.validation.common import (  # noqa: E402
    MLP_MATRICES,
    PeriodGroup,
    analysis_datasets_dir,
    analysis_dir,
    load_component_uv,
    read_alive_components,
    read_subcomp_period_groups,
)

_NO_PERIOD = PeriodGroup("none", 0.0, "no period", (2, 0.0))

_APP_TEMPLATE = Path(__file__).with_name("subspace_scatter_app.html")
_CANDIDATE_OPS = ("add", "sub", "mult")
# Per-task result of the operation, used as a colour option.
_RESULT = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mult": lambda a, b: a * b}
# side -> (which vector defines the direction, the matrices the picks come from). Which activation
# each direction is projected onto is resolved per (side, matrix) by `_side_acts`, in MLP order:
# input → pre-nonlinearity → post-nonlinearity → output.
_SIDES = {
    "input": ("V", ("gate_proj", "up_proj")),
    "pre-nonlinearity": ("U", ("gate_proj", "up_proj")),
    "post-nonlinearity": ("V", ("down_proj",)),
    "output": ("U", ("down_proj",)),
}


def _silu(x: NDArray[np.float32]) -> NDArray[np.float32]:
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def _side_acts(side: str, hidden: Any, n: int) -> dict[str, NDArray[np.float32]]:
    """Per matrix, the last-token activation a subcomponent direction is projected onto:
    `input` → post-RMSNorm MLP input (up/gate V is d_model); `output` → MLP output (down U is
    d_model); `pre-nonlinearity` → the matrix's own preactivation (up/gate U is d_int);
    `post-nonlinearity` → the post-SwiGLU neuron output `silu(gate)·up` (down V is d_int).
    """

    def flat(key: str) -> NDArray[np.float32]:
        return hidden[key].reshape(n * n, -1).astype(np.float32)

    match side:
        case "input":
            g = flat("mlp_input")
            return {"gate_proj": g, "up_proj": g}
        case "output":
            return {"down_proj": flat("mlp_output")}
        case "pre-nonlinearity":
            return {"up_proj": flat("up_preact"), "gate_proj": flat("gate_preact")}
        case "post-nonlinearity":
            return {"down_proj": _silu(flat("gate_preact")) * flat("up_preact")}
        case _:
            raise AssertionError(f"unknown side {side!r}")


def _thumbnail(grid: NDArray[np.float32]) -> str:
    """A small signed-diverging heatmap of the (a, b) pattern as a base64 PNG data URI."""
    lim = float(np.abs(grid).max()) or 1.0
    fig, ax = plt.subplots(figsize=(1.0, 1.0))
    ax.imshow(grid.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _resolve_ops(data_dir: Path, ops: str | tuple[str, ...] | None) -> list[str]:
    if ops is not None:
        return list(ops) if isinstance(ops, (list, tuple)) else str(ops).split(",")
    detected = [
        op
        for op in _CANDIDATE_OPS
        if (data_dir / f"hidden_activations_{op}.npz").exists()
        and (data_dir / f"alive_filtered_{op}.tsv").exists()
    ]
    assert detected, (
        f"no tasks with hidden_activations_<op>.npz + alive_filtered_<op>.tsv in {data_dir}"
    )
    return detected


def build_subspace_scatter(
    model_path: ModelPath, ops: str | None = None, output_dir: str | None = None
) -> Path:
    checkpoint = Path(model_path).expanduser()
    assert checkpoint.exists(), f"checkpoint not found: {checkpoint}"
    run_dir = checkpoint.parent
    data_dir = analysis_datasets_dir(run_dir)
    op_list = _resolve_ops(data_dir, ops)
    logger.info(f"tasks: {op_list}")

    # Per-task: alive set, (optional) periods, and the stored activations.
    per_op: dict[str, dict[str, Any]] = {}
    n = 0
    for op in op_list:
        alive = read_alive_components(
            data_dir / f"alive_filtered_{op}.tsv", keep_projs=MLP_MATRICES
        )
        periods_path = data_dir / f"subcomp_periods_{op}.tsv"
        periods = read_subcomp_period_groups(periods_path) if periods_path.exists() else {}
        hidden = np.load(data_dir / f"hidden_activations_{op}.npz", allow_pickle=True)
        nn = int(hidden["a"].shape[0])
        assert n in (0, nn), f"grid size mismatch across tasks ({n} vs {nn})"
        n = nn
        per_op[op] = {"alive": alive, "periods": periods, "hidden": hidden}

    layer = per_op[op_list[0]]["alive"][0].layer
    uv = load_component_uv(checkpoint, layer, MLP_MATRICES)

    # Shared point metadata: every task's full (a, b) grid, concatenated.
    a_grid = np.repeat(np.arange(1, n + 1), n)
    b_grid = np.tile(np.arange(1, n + 1), n)
    point_task = [ti for ti in range(len(op_list)) for _ in range(n * n)]
    point_a = [int(v) for _ in op_list for v in a_grid]
    point_b = [int(v) for _ in op_list for v in b_grid]
    point_result = [
        int(_RESULT[op](int(a), int(b)))
        for op in op_list
        for a, b in zip(a_grid, b_grid, strict=True)
    ]

    # Per-task modulo options for the colour control: integer residues (add/sub) or — when the
    # task's subcomponents are mostly log-periodic (mult) — the detected log ratios, so colour
    # can show the multiplicative phase that matches an actual subcomponent frequency.
    task_mods: list[dict[str, Any]] = []
    for op in op_list:
        groups = list(per_op[op]["periods"].values())
        n_log = sum(g.kind == "log" for g in groups)
        if n_log > sum(g.kind == "additive" for g in groups):
            ratios = sorted({round(g.value, 2) for g in groups if g.kind == "log"})
            task_mods.append({"kind": "log", "values": ratios})
        else:
            task_mods.append({"kind": "additive", "values": [2, 5, 10, 20, 25, 50, 100]})

    sides: dict[str, Any] = {}
    for side, (which, projs) in _SIDES.items():
        acts = {
            op: _side_acts(side, per_op[op]["hidden"], n) for op in op_list
        }  # [op][proj] -> [n*n, dim]
        # Union of components alive on any task for this side -> one direction each.
        union = sorted(
            {
                (a.proj, a.component)
                for op in op_list
                for a in per_op[op]["alive"]
                if a.proj in projs
            }
        )
        dir_index = {key: i for i, key in enumerate(union)}
        dir_label: list[str] = []
        dir_proj: list[NDArray[np.float32]] = []
        dirs: list[NDArray[np.float32]] = []
        for proj, c in union:
            v, u = uv[proj]
            d = v[:, c] if which == "V" else u[c, :]
            d = (d / max(float(np.linalg.norm(d)), 1e-12)).astype(np.float32)
            proj_all = np.concatenate([acts[op][proj] @ d for op in op_list]).astype(np.float32)
            if float(np.median(proj_all)) < 0:  # gauge: arrow points toward the data
                d, proj_all = -d, -proj_all
            dirs.append(d)
            dir_proj.append(proj_all)
            dir_label.append(f"{proj[0]}{c}")
        gram = np.stack(dirs) @ np.stack(dirs).T

        # Picks: one card per (task, alive component on this side), sorted by period (none last).
        picks: list[dict[str, Any]] = []
        for ti, op in enumerate(op_list):
            periods = per_op[op]["periods"]
            comps = sorted(
                {(a.proj, a.component) for a in per_op[op]["alive"] if a.proj in projs},
                key=lambda pc, p=periods: (p.get(pc, _NO_PERIOD).sort_key, pc[0], pc[1]),
            )
            offset = ti * n * n
            for proj, c in comps:
                di = dir_index[(proj, c)]
                thumb_grid = dir_proj[di][offset : offset + n * n].reshape(n, n)
                picks.append(
                    {
                        "dir": di,
                        "task": op,
                        "group": periods.get((proj, c), _NO_PERIOD).label,
                        "label": dir_label[di],
                        "thumb": _thumbnail(thumb_grid),
                    }
                )
        sides[side] = {
            "gram": [[round(float(g), 4) for g in row] for row in gram],
            "dir_proj": [[round(float(x), 4) for x in p] for p in dir_proj],
            "dir_label": dir_label,
            "picks": picks,
        }
        logger.info(
            f"{side}: {len(union)} directions, {len(picks)} pickable cards across {len(op_list)} tasks"
        )

    payload = {
        "meta": {
            "tasks": op_list,
            "point_task": point_task,
            "a": point_a,
            "b": point_b,
            "result": point_result,
            "task_mods": task_mods,
        },
        "sides": sides,
    }
    out_dir = (
        Path(output_dir).expanduser() if output_dir else analysis_dir(run_dir) / "subspace_scatter"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    assert _APP_TEMPLATE.exists(), f"app template missing: {_APP_TEMPLATE}"
    template = _APP_TEMPLATE.read_text()
    for marker in ("/*__PLOTLY_JS__*/", "/*__PD_DATA__*/"):
        assert marker in template, f"template missing injection marker {marker}"
    html = template.replace("/*__PLOTLY_JS__*/", pyo.get_plotlyjs()).replace(
        "/*__PD_DATA__*/", json.dumps(payload, separators=(",", ":"))
    )
    (out_dir / "index.html").write_text(html)
    logger.info(f"wrote subspace-scatter applet ({', '.join(op_list)}) → {out_dir}")
    return out_dir


if __name__ == "__main__":
    fire.Fire(build_subspace_scatter)
