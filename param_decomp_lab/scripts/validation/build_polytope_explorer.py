"""Build a GPU-free HTML applet mapping an operation's (a, b) grid into gate-sign polytopes.

A SwiGLU MLP is piecewise (approximately) linear: within a region of input space where every
gate preactivation keeps its sign, the MLP applies one roughly fixed linear map (ignoring the
negative silu bump near zero). The applet colours the operation's `1..N × 1..N` operand grid
by *which combination of alive gates is positive* — each colour is one such polytope — to
answer "which prompts activate the same combination of gates?". A second colouring mode uses
the combination of causally-important subcomponents (CI > threshold) instead, answering the
same question for subcomponents.

**Alive gates** are the L18 MLP neurons whose gate preactivation takes both signs across the
op's grid (a gate that never flips contributes no polytope boundary there). Most of `d_int`
flips somewhere, so only the `--top-gates` most *output-relevant* alive gates are stored —
ranked by `std over the grid of silu(gate_j)·up_j` times `||down column j||`, the size of the
neuron's contribution to the MLP output. In the applet, top-k controls plus per-item
checkboxes choose which stored gates (or, in CI mode, which subcomponents) form the
combination; combinations are coloured by frequency, with the rarest pooled into grey. Hovering a map pixel (click to pin) shows which
gates are positive and which subcomponents are causally important on that prompt, against a
panel of per-gate preactivation and per-subcomponent CI / inner-activation `(a, b)` heatmaps;
hovering a legend row highlights its polytope on the map.

CPU-only: per op it reads `hidden_activations_<op>.npz` (gate/up grids),
`alive_filtered_<op>.tsv` + `inner_activations_<op>.tsv` (subcomponent set, CI + inner grids
— the TSV must have the `ci` column), and the checkpoint's target down-projection weight
(mmap). No forward pass. Every op with all three inputs present is included; the applet has
an operation selector.

Usage:
    python -m param_decomp_lab.scripts.validation.build_polytope_explorer <model_path> \
        [--ops=add,mult] [--top-gates=64] [--output-dir=PATH]

Output: `<run_dir>/analysis/polytope_explorer/{index.html,data.js}`.
"""

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
    b64_f16,
    load_target_mlp_weights,
    op_symbol,
    read_alive_components,
    read_mean_ci,
)

_APP_TEMPLATE = Path(__file__).with_name("polytope_explorer_app.html")
_OP_CANDIDATES = ("add", "sub", "mult")
_MATRIX_RANK = {"gate_proj": 0, "up_proj": 1, "down_proj": 2}
_OP_INPUTS = (
    "hidden_activations_{op}.npz",
    "alive_filtered_{op}.tsv",
    "inner_activations_{op}.tsv",
)


def _silu(x: NDArray[np.float32]) -> NDArray[np.float32]:
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def _resolve_ops(data_dir: Path, ops: str | tuple[str, ...] | None) -> list[str]:
    if ops is not None:
        return list(ops) if isinstance(ops, (list, tuple)) else str(ops).split(",")
    detected = [
        op
        for op in _OP_CANDIDATES
        if all((data_dir / f.format(op=op)).exists() for f in _OP_INPUTS)
    ]
    assert detected, f"no op with {[f.format(op='<op>') for f in _OP_INPUTS]} in {data_dir}"
    return detected


def _sub_grids(
    tsv_path: Path, subs: list[AliveComponent], n: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """`[S, N, N]` CI and inner-activation grids (row-major `[a-1, b-1]`) per subcomponent."""
    index = {(s.proj, s.component): i for i, s in enumerate(subs)}
    ci = np.zeros((len(subs), n, n), dtype=np.float32)
    inner = np.zeros((len(subs), n, n), dtype=np.float32)
    with tsv_path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        assert reader.fieldnames is not None and "ci" in reader.fieldnames, (
            f"{tsv_path.name} lacks a `ci` column; rerun collect_inner_activations"
        )
        for row in reader:
            i = index.get((row["matrix"], int(row["subcomponent"])))
            if i is None:
                continue
            a, b = int(row["a"]) - 1, int(row["b"]) - 1
            ci[i, a, b] = float(row["ci"])
            inner[i, a, b] = float(row["inner_act"])
    return ci, inner


def build_polytope_explorer(
    model_path: ModelPath,
    ops: str | None = None,
    top_gates: int = 64,
    output_dir: str | None = None,
) -> Path:
    checkpoint = Path(model_path).expanduser()
    assert checkpoint.exists(), f"checkpoint not found: {checkpoint}"
    run_dir = checkpoint.parent
    data_dir = analysis_datasets_dir(run_dir)
    op_list = _resolve_ops(data_dir, ops)

    layer, n = -1, -1
    down_norm = np.zeros(0, dtype=np.float32)  # ||down column j|| per neuron, set on first op
    ops_payload: dict[str, Any] = {}
    ops_meta: list[dict[str, Any]] = []
    for op in op_list:
        z = np.load(data_dir / f"hidden_activations_{op}.npz", allow_pickle=True)
        if layer < 0:
            layer, n = int(z["layer"]), int(z["a"].shape[0])
            down = load_target_mlp_weights(checkpoint, layer, ("down_proj",))["down_proj"]
            down_norm = np.linalg.norm(down, axis=0).astype(np.float32)
        assert int(z["layer"]) == layer and int(z["a"].shape[0]) == n
        gate = z["gate_preact"].astype(np.float32)  # [N, N, d_int], indexed [a-1, b-1]
        up = z["up_preact"].astype(np.float32)
        d_int = gate.shape[-1]
        assert down_norm.shape == (d_int,)

        gate_flat = gate.reshape(n * n, d_int)
        pos_frac = (gate_flat > 0).mean(axis=0)
        flips = (pos_frac > 0) & (pos_frac < 1)  # the alive gates: both signs on this op's grid
        assert flips.any(), f"{op}: no gate takes both signs over the grid"
        relevance = (_silu(gate_flat) * up.reshape(n * n, d_int)).std(axis=0) * down_norm
        kept = np.argsort(-np.where(flips, relevance, -np.inf))[: min(top_gates, int(flips.sum()))]
        gate_grids = np.transpose(gate[:, :, kept], (2, 0, 1))  # [K, N, N]
        del gate, up, gate_flat

        mean_ci = read_mean_ci(data_dir / f"alive_filtered_{op}.tsv")
        subs = read_alive_components(data_dir / f"alive_filtered_{op}.tsv", keep_projs=MLP_MATRICES)
        subs = sorted(subs, key=lambda s: (_MATRIX_RANK[s.proj], -mean_ci[(s.proj, s.component)]))
        ci_grids, inner_grids = _sub_grids(data_dir / f"inner_activations_{op}.tsv", subs, n)

        ops_payload[op] = {
            "symbol": op_symbol(op),
            "gates": [
                {
                    "id": int(j),
                    "rel": round(float(relevance[j]), 4),
                    "pos_frac": round(float(pos_frac[j]), 4),
                }
                for j in kept
            ],
            "gate_grids": b64_f16(gate_grids),
            "subs": [
                {
                    "proj": s.proj,
                    "c": s.component,
                    "mean_ci": round(mean_ci[(s.proj, s.component)], 4),
                }
                for s in subs
            ],
            "ci_grids": b64_f16(ci_grids),
            "inner_grids": b64_f16(inner_grids),
        }
        ops_meta.append(
            {
                "op": op,
                "symbol": op_symbol(op),
                "n_flipping": int(flips.sum()),
                "n_kept": int(kept.size),
                "n_subs": len(subs),
            }
        )
        logger.info(
            f"{op}: {int(flips.sum())}/{d_int} gates flip sign, kept top {kept.size} by output "
            f"relevance; {len(subs)} subcomponents"
        )

    payload: dict[str, Any] = {
        "meta": {
            "layer": layer,
            "n": n,
            "d_int": int(down_norm.shape[0]),
            "default_top_gates": 8,  # gates initially in the combination (top-20 combos cover ~70%)
            "default_top_subs": 8,  # subcomponents (by mean CI) initially in the combination
            "ops": ops_meta,
        },
        "ops": ops_payload,
    }

    out_dir = (
        Path(output_dir).expanduser() if output_dir else analysis_dir(run_dir) / "polytope_explorer"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.js").write_text(
        f"window.PD_DATA = {json.dumps(payload, separators=(',', ':'))};\n"
    )
    assert _APP_TEMPLATE.exists(), f"app template missing: {_APP_TEMPLATE}"
    shutil.copyfile(_APP_TEMPLATE, out_dir / "index.html")

    size_mb = (out_dir / "data.js").stat().st_size / 1e6
    logger.info(
        f"wrote polytope explorer to {out_dir} (data.js {size_mb:.1f} MB) — open index.html"
    )
    return out_dir


if __name__ == "__main__":
    fire.Fire(build_polytope_explorer)
