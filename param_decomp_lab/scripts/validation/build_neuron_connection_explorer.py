"""Build a GPU-free HTML explorer of subcomponent <-> neuron connections in L18's MLP.

For a chosen operand pair `(a, b)` it shows, left to right:
- (left)   the gate / up subcomponents causally important on that prompt (up on top, then
           gate; each sorted by activation period),
- (center) the neurons those subcomponents write to / read from, kept only if some active
           subcomponent reaches |connection strength| above an adjustable threshold, sorted
           so each neuron sits near the gate/up subcomponent that drives it hardest,
- (right)  the down subcomponents causally important on that prompt (sorted by period).

Connection strength uses the V-unit normalization (V -> V/||V||, U -> U*||V||, outer product
unchanged):
- gate / up (pre-SwiGLU): strength to neuron j = `U[c,j]*||V_c||` (what is written to neuron j
  when the normalized inner activation is 1),
- down (post-SwiGLU): strength for neuron j = `V[j,c]/||V_c||` (the normalized read weight).

Lines between subcomponents and neurons are coloured by connection strength (red positive,
blue negative). Hovering a subcomponent shows its causal-importance `(a, b)` heatmap;
hovering a subcomponent shows its `(a, b)` heatmap — CI *or* normalized inner activation per
the page toggle; hovering a neuron shows its up / gate / post-SwiGLU output for the prompt.

CPU-only: reads the checkpoint U/V (mmap), the filtered-alive list (`alive_filtered_<op>.tsv`),
the periods, the `find_alive_components` per-position JSON (`alive_components_per_position.json`
— op-agnostic, filtered here; CI patterns + per-prompt activity), the `collect_inner_activations`
TSV (inner-activation patterns), and the `collect_hidden_activations` npz (neuron values). No
forward pass.

Usage:
    python -m param_decomp_lab.scripts.validation.build_neuron_connection_explorer <model_path> \
        [--op=add] [--conn-floor=0.1] [--top-neurons=60] [--output-dir=PATH]

Output: `<run_dir>/analysis/neuron_explorer_<op>/{index.html,data.js}`.
"""

import base64
import csv
import json
import re
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
    analysis_datasets_dir,
    analysis_dir,
    load_component_uv,
    op_symbol,
    read_alive_components,
    read_subcomp_periods,
)

_APP_TEMPLATE = Path(__file__).with_name("neuron_connection_explorer_app.html")


def _conn_vector(
    proj: str, c: int, uv: dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]]
) -> NDArray[np.float32]:
    """Unit-normalised per-neuron connection direction of subcomponent `c`: the read direction
    `V[:,c]` (down) or write direction `U[c,:]` (gate/up), each scaled to L2 norm 1. So an edge
    value is that neuron's share of the subcomponent's connection energy (∑_j w² = 1) and one
    threshold / colour ramp means the same on read and write sides."""
    v, u = uv[proj]
    vec = v[:, c] if proj == "down_proj" else u[c, :]
    return vec / max(float(np.linalg.norm(vec)), 1e-12)


def _ci_grids(
    json_path: Path, op: str, alive_keys: set[tuple[str, int]], n: int
) -> dict[tuple[str, int], NDArray[np.float32]]:
    """Last-position CI `(a, b)` grid for each filtered-alive subcomponent, this op only.

    The `find_alive_components` per-position JSON is op-agnostic (it can hold `a+b=` and
    `a×b=` prompts together), so we match on this op's exact symbol and assert ≥1 prompt hit —
    a wrong/missing JSON fails loudly instead of yielding silent all-zero grids.
    """
    data: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = json.loads(json_path.read_text())
    pattern = re.compile(rf"^(\d+){re.escape(op_symbol(op))}(\d+)=$")
    grids = {key: np.zeros((n, n), dtype=np.float32) for key in alive_keys}
    matched = 0
    for prompt, per_pos in data.items():
        m = pattern.match(prompt)
        if m is None:
            continue
        matched += 1
        a, b = int(m.group(1)), int(m.group(2))
        last = max(per_pos, key=int)
        for module, comps in per_pos[last].items():
            proj = module.split(".")[-1]
            for entry in comps:
                key = (proj, entry["component"])
                if key in grids:
                    grids[key][a - 1, b - 1] = entry["ci"]
    assert matched > 0, f"no '{op}' ({op_symbol(op)}) prompts in {json_path.name}"
    return grids


def _inner_grids(
    tsv_path: Path, alive_keys: set[tuple[str, int]], n: int
) -> dict[tuple[str, int], NDArray[np.float32]]:
    """Dense `(a, b)` normalized-inner-activation grid per filtered-alive subcomponent."""
    grids = {key: np.zeros((n, n), dtype=np.float32) for key in alive_keys}
    with tsv_path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = (row["matrix"], int(row["subcomponent"]))
            if key in grids:
                grids[key][int(row["a"]) - 1, int(row["b"]) - 1] = float(row["inner_act"])
    return grids


def _sparse_grid(grid: NDArray[np.float32]) -> dict[str, str]:
    """Active cells of a CI grid as base64 (uint16 flat [a,b] index + uint8 CI*255)."""
    flat = grid.ravel()
    idx = np.nonzero(flat)[0].astype(np.uint16)
    val = np.clip(np.rint(flat[idx] * 255.0), 0, 255).astype(np.uint8)
    return {
        "idx": base64.b64encode(idx.tobytes()).decode("ascii"),
        "val": base64.b64encode(val.tobytes()).decode("ascii"),
    }


def _dense_b64(grid: NDArray[np.float32]) -> str:
    """Dense `[N, N]` grid (row-major `[a, b]`) as fp16 base64 — keeps signed values."""
    return base64.b64encode(grid.astype(np.float16).tobytes()).decode("ascii")


def build_neuron_connection_explorer(
    model_path: ModelPath,
    op: str = "add",
    conn_floor: float = 0.1,
    top_neurons: int = 60,
    output_dir: str | None = None,
) -> Path:
    checkpoint = Path(model_path).expanduser()
    assert checkpoint.exists(), f"checkpoint not found: {checkpoint}"
    run_dir = checkpoint.parent
    data_dir = analysis_datasets_dir(run_dir)

    alive = read_alive_components(data_dir / f"alive_filtered_{op}.tsv", keep_projs=MLP_MATRICES)
    periods = read_subcomp_periods(data_dir / f"subcomp_periods_{op}.tsv")
    layer = alive[0].layer
    uv = load_component_uv(checkpoint, layer, MLP_MATRICES)

    npz_path = data_dir / f"hidden_activations_{op}.npz"
    assert npz_path.exists(), f"missing {npz_path.name}; run collect_hidden_activations first"
    hidden = np.load(npz_path, allow_pickle=True)
    n = int(hidden["a"].shape[0])
    up_grid = hidden["up_preact"]  # [N, N, d_int] float16
    gate_grid = hidden["gate_preact"]

    alive_keys = {(a.proj, a.component) for a in alive}
    # CI patterns come from the (unsuffixed, op-agnostic) find_alive_components output; the
    # signed inner-activation patterns from this op's collect_inner_activations TSV.
    ci_grids = _ci_grids(data_dir / "alive_components_per_position.json", op, alive_keys, n)
    inner_grids = _inner_grids(data_dir / f"inner_activations_{op}.tsv", alive_keys, n)

    # Per subcomponent: select its strongest neurons (|conn| > floor, top-K), accumulate the
    # neuron universe (union across all subcomponents and matrices).
    selected: dict[tuple[str, int], list[tuple[int, float]]] = {}
    universe: set[int] = set()
    for a_comp in alive:
        key = (a_comp.proj, a_comp.component)
        w = _conn_vector(a_comp.proj, a_comp.component, uv)
        keep = np.nonzero(np.abs(w) > conn_floor)[0]
        keep = keep[np.argsort(-np.abs(w[keep]))[:top_neurons]]
        selected[key] = [(int(j), round(float(w[j]), 4)) for j in keep]
        universe.update(int(j) for j in keep)

    neuron_ids = sorted(universe)
    u_index = {nid: i for i, nid in enumerate(neuron_ids)}
    logger.info(
        f"{len(alive)} subcomponents, {len(neuron_ids)} neurons in universe (floor {conn_floor}, top {top_neurons})"
    )

    def _subcomp_entries(proj_filter: tuple[str, ...]) -> list[dict[str, Any]]:
        comps = sorted(
            (a for a in alive if a.proj in proj_filter),
            key=lambda a: (periods[(a.proj, a.component)], a.proj, a.component),
        )
        return [
            {
                "proj": a.proj,
                "c": a.component,
                "period": periods[(a.proj, a.component)],
                "ci": _sparse_grid(ci_grids[(a.proj, a.component)]),
                "inner": _dense_b64(inner_grids[(a.proj, a.component)]),
                "conn": [[u_index[j], w] for j, w in selected[(a.proj, a.component)]],
            }
            for a in comps
        ]

    # Neuron up / gate values over the (a, b) grid, restricted to the universe, fp16 base64.
    universe_arr = np.array(neuron_ids, dtype=np.int64)
    up_u = up_grid[:, :, universe_arr].astype(np.float16)
    gate_u = gate_grid[:, :, universe_arr].astype(np.float16)

    payload = {
        "meta": {
            "op": op,
            "n": n,
            "layer": layer,
            "conn_floor": conn_floor,
            "top_neurons": top_neurons,
        },
        "subcomps": {
            "up": _subcomp_entries(("up_proj",)),
            "gate": _subcomp_entries(("gate_proj",)),
            "down": _subcomp_entries(("down_proj",)),
        },
        "neuron_ids": neuron_ids,
        "n_universe": len(neuron_ids),
        "neuron_up": base64.b64encode(up_u.tobytes()).decode("ascii"),
        "neuron_gate": base64.b64encode(gate_u.tobytes()).decode("ascii"),
    }

    out_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else analysis_dir(run_dir) / f"neuron_explorer_{op}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.js").write_text(
        f"window.PD_DATA = {json.dumps(payload, separators=(',', ':'))};\n"
    )
    assert _APP_TEMPLATE.exists(), f"app template missing: {_APP_TEMPLATE}"
    shutil.copyfile(_APP_TEMPLATE, out_dir / "index.html")

    size_mb = (out_dir / "data.js").stat().st_size / 1e6
    logger.info(f"wrote explorer to {out_dir} (data.js {size_mb:.1f} MB) — open index.html")
    return out_dir


if __name__ == "__main__":
    fire.Fire(build_neuron_connection_explorer)
