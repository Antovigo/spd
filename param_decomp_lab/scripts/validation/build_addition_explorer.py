"""Build a GPU-free interactive HTML explorer for `a+b=` arithmetic decompositions.

Answers two questions about a targeted MLP decomposition run on addition prompts:
1. Which periodic *bases* do components encode (odd/even, mod 5, mod 10, ...)?
2. Which intermediate *neurons* carry each base, and how do the gate / up / down
   projections cooperate?

It reads the `find_alive_components` per-position JSON (causal importances over the
(a, b) operand grid) and the decomposed checkpoint's component weights (U / V, loaded via
mmap so only the tiny per-component tensors touch RAM). No model forward pass, so it runs
on the GPU-less login node. Note: neuron *involvement* is read off the weights (which
neurons a component writes to / reads from), not from measured activations — exact
per-neuron activations would need a forward pass.

For every component it computes, per token position:
- the dense (a, b) causal-importance grid (stored sparsely),
- marginal CI along a, b, a+b, a-b, and their FFT power spectra,
- a residue-variance (eta-squared) fingerprint over moduli 2..`max_modulus`,
- a detected base = the integer period of the most periodic axis.
From the weights it computes each component's top intermediate neurons and the
cross-matrix neuron overlaps (gate<->up shared neurons, gate/up -> down read-write).

The applet (`index.html` + `data.js`, both `file://`-openable, no server / CDN / GPU) has
a sortable component gallery, a per-component detail panel, a per-base summary, and a
gate/up/down interplay + (a, b) cell inspector.

Usage:
    python -m param_decomp_lab.scripts.validation.build_addition_explorer <model_path> \
        [--op=+] [--ci-thr=0.1] [--positions=1,3,4] [--max-modulus=25] [--top-k=10] \
        [--no-weights] [--output-dir=PATH]

Output: `<run_dir>/analysis/addition_explorer/{index.html,data.js}`.
"""

import base64
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fire
import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import NDArray

from param_decomp.log import logger
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.scripts.validation.common import (
    analysis_datasets_dir,
    analysis_dir,
    parse_module_name,
)

_APP_TEMPLATE = Path(__file__).with_name("addition_explorer_app.html")
_SHORT = {
    "model.layers.18.mlp.gate_proj": "gate_proj",
    "model.layers.18.mlp.up_proj": "up_proj",
    "model.layers.18.mlp.down_proj": "down_proj",
}
# prompt -> position(str) -> module -> [{"component": int, "ci": float}]
PerPosition = dict[str, dict[str, dict[str, list[dict[str, Any]]]]]


def _parse_ab(data: PerPosition, op: str) -> tuple[dict[str, tuple[int, int]], int, int]:
    pattern = re.compile(rf"^(\d+){re.escape(op)}(\d+)=$")
    ab: dict[str, tuple[int, int]] = {}
    for prompt in data:
        match = pattern.match(prompt)
        if match is not None:
            ab[prompt] = (int(match.group(1)), int(match.group(2)))
    assert ab, f"no prompts of the form 'a{op}b=' found in the JSON"
    a_max = max(a for a, _ in ab.values())
    b_max = max(b for _, b in ab.values())
    assert a_max == b_max, f"expected a square (a, b) grid, got {a_max}x{b_max}"
    return ab, a_max, b_max


def _short_module(module: str) -> str:
    return _SHORT.get(module, module.rsplit(".", 1)[-1])


def _build_grids(
    data: PerPosition, ab: dict[str, tuple[int, int]], positions: list[int], n: int
) -> dict[int, dict[str, NDArray[np.float32]]]:
    """pos -> short_module -> [C, n, n] CI grid (axis order [component, b-1, a-1])."""
    n_comp = 128
    grids: dict[int, dict[str, NDArray[np.float32]]] = {
        pos: {m: np.zeros((n_comp, n, n), dtype=np.float32) for m in _SHORT.values()}
        for pos in positions
    }
    for prompt, (a, b) in ab.items():
        per_position = data[prompt]
        for pos in positions:
            per_module = per_position.get(str(pos), {})
            for module, comps in per_module.items():
                short = _short_module(module)
                grid = grids[pos][short]
                for entry in comps:
                    grid[entry["component"], b - 1, a - 1] = entry["ci"]
    return grids


def _eta_squared_spectrum(
    flat_vals: Float[NDArray[np.float32], " cells"],
    flat_key: Float[NDArray[np.int64], " cells"],
    max_modulus: int,
) -> list[float]:
    """eta^2 (between-residue-class variance / total) of CI grouped by `flat_key mod m`."""
    total = float(((flat_vals - flat_vals.mean()) ** 2).sum())
    if total < 1e-12:
        return [0.0] * (max_modulus - 1)
    out: list[float] = []
    for m in range(2, max_modulus + 1):
        residue = flat_key % m
        between = 0.0
        for r in range(m):
            sel = flat_vals[residue == r]
            if sel.size:
                between += sel.size * (sel.mean() - flat_vals.mean()) ** 2
        out.append(round(float(between) / total, 4))
    return out


def _fft_power(marginal: Float[NDArray[np.float32], " length"]) -> tuple[list[float], int, float]:
    """Normalized power at frequencies k=1..len//2; plus (peak_period, peak_confidence)."""
    centered = marginal - marginal.mean()
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    power = spectrum[1:]  # drop DC
    total = float(power.sum())
    if total < 1e-12 or power.size == 0:
        return [0.0] * power.size, 0, 0.0
    norm = power / total
    peak_k = int(np.argmax(power)) + 1
    period = int(round(len(marginal) / peak_k))
    return [round(float(p), 4) for p in norm], period, round(float(norm.max()), 4)


def _fundamental_base(eta: list[float], max_modulus: int, plateau: float) -> tuple[int, float]:
    """From an eta^2-by-modulus spectrum, the smallest modulus reaching the plateau.

    eta^2(m) is (weakly) monotone up across multiples of a true period, so the maximum lands
    on a large modulus; the *fundamental* base is the smallest m whose eta^2 is within
    `plateau` of that maximum. Returns (base_modulus, plateau_eta_squared).
    """
    best = max(eta)
    for i, m in enumerate(range(2, max_modulus + 1)):
        if eta[i] >= plateau * best:
            return m, round(float(best), 4)
    return 2, round(float(best), 4)


def _analyse_component(
    grid: Float[NDArray[np.float32], "n n"], max_modulus: int
) -> dict[str, Any] | None:
    """Per-component spectral + residue analysis at one position, or None if never active."""
    if grid.max() <= 0.0:
        return None
    n = grid.shape[0]
    # grid axis order is [b-1, a-1].
    marg_a = grid.mean(axis=0)  # over b -> f(a)
    marg_b = grid.mean(axis=1)  # over a -> f(b)

    a_idx = np.arange(1, n + 1)
    aa, bb = np.meshgrid(a_idx, a_idx, indexing="xy")  # aa varies along axis=1 (a), bb axis=0 (b)
    sum_key = (aa + bb).ravel()
    diff_key = (aa - bb).ravel()
    flat = grid.ravel().astype(np.float32)

    sum_vals = np.array([flat[sum_key == s].mean() for s in range(2, 2 * n + 1)], dtype=np.float32)
    diff_vals = np.array([flat[diff_key == d].mean() for d in range(-(n - 1), n)], dtype=np.float32)

    fft: dict[str, list[float]] = {}
    eta: dict[str, list[float]] = {}
    base_per_axis: dict[str, tuple[int, float]] = {}
    for name, marg, key in (
        ("a", marg_a, aa.ravel().astype(np.int64)),
        ("b", marg_b, bb.ravel().astype(np.int64)),
        ("sum", sum_vals, sum_key.astype(np.int64)),
        ("diff", diff_vals, diff_key.astype(np.int64)),
    ):
        fft[name] = _fft_power(marg.astype(np.float32))[0]
        eta[name] = _eta_squared_spectrum(flat, key, max_modulus)
        base_per_axis[name] = _fundamental_base(eta[name], max_modulus, plateau=0.85)

    # Base = periodic residue structure (eta^2), not magnitude trend, so a ramp in a/b/sum
    # reads as "broad" rather than a spurious long period. Pick the axis with the strongest η².
    best_axis = max(base_per_axis, key=lambda ax: base_per_axis[ax][1])
    base_period, base_strength = base_per_axis[best_axis]

    def q(arr: NDArray[np.float32]) -> list[float]:
        return [round(float(x), 3) for x in arr]

    return {
        "frac": round(float((grid > 0).mean()), 4),
        "maxci": round(float(grid.max()), 3),
        "marg": {"a": q(marg_a), "b": q(marg_b), "sum": q(sum_vals), "diff": q(diff_vals)},
        "fft": fft,
        "eta": eta,
        "base": {"axis": best_axis, "period": base_period, "confidence": base_strength},
    }


def _sparse_grid(grid: Float[NDArray[np.float32], "n n"]) -> dict[str, str]:
    """Active cells as base64 (uint16 flat index + uint8 CI*255), row-major over [b, a]."""
    flat = grid.ravel()
    idx = np.nonzero(flat)[0].astype(np.uint16)
    val = np.clip(np.rint(flat[idx] * 255.0), 0, 255).astype(np.uint8)
    return {
        "idx": base64.b64encode(idx.tobytes()).decode("ascii"),
        "val": base64.b64encode(val.tobytes()).decode("ascii"),
    }


@dataclass
class _Weights:
    gate_write: Float[NDArray[np.float32], "c d_int"]
    up_write: Float[NDArray[np.float32], "c d_int"]
    down_read: Float[NDArray[np.float32], "c d_int"]


def _load_weights(checkpoint: Path, layer: int) -> _Weights:
    sd = torch.load(checkpoint, map_location="cpu", mmap=True)
    p = f"_components.model-layers-{layer}-mlp"

    def get(name: str) -> NDArray[np.float32]:
        return sd[f"{p}-{name}"].float().numpy()

    return _Weights(
        gate_write=get("gate_proj.U"),  # [C, d_int]
        up_write=get("up_proj.U"),  # [C, d_int]
        down_read=get("down_proj.V").T.copy(),  # V is [d_int, C] -> [C, d_int]
    )


def _unit_rows(mat: NDArray[np.float32]) -> NDArray[np.float32]:
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return mat / norm


def _top_partners(
    src: NDArray[np.float32], dst: NDArray[np.float32], top_k: int
) -> list[list[dict[str, float]]]:
    """For each src row, the top_k dst rows by cosine similarity (signed)."""
    cos = _unit_rows(src) @ _unit_rows(dst).T  # [C_src, C_dst]
    out: list[list[dict[str, float]]] = []
    for row in cos:
        order = np.argsort(-np.abs(row))[:top_k]
        out.append([{"c": int(j), "cos": round(float(row[j]), 3)} for j in order])
    return out


def _neuron_fingerprints(write: NDArray[np.float32], top_k: int) -> list[dict[str, list[float]]]:
    """Top neurons (by |weight|) each component writes to / reads from."""
    out: list[dict[str, list[float]]] = []
    for row in write:
        order = np.argsort(-np.abs(row))[:top_k]
        out.append(
            {
                "idx": [int(j) for j in order],
                "w": [round(float(row[j]), 3) for j in order],
            }
        )
    return out


@dataclass
class _Payload:
    meta: dict[str, Any]
    components: dict[str, list[dict[str, Any] | None]] = field(default_factory=dict)
    neurons: dict[str, Any] = field(default_factory=dict)
    overlap: dict[str, Any] = field(default_factory=dict)


def build_addition_explorer(
    model_path: ModelPath,
    op: str = "+",
    ci_thr: float = 0.1,
    positions: str = "1,3,4",
    max_modulus: int = 25,
    top_k: int = 10,
    no_weights: bool = False,
    output_dir: str | None = None,
) -> Path:
    """Write the interactive explorer (`index.html` + `data.js`). Returns the output folder."""
    checkpoint = Path(model_path).expanduser()
    assert checkpoint.exists(), f"checkpoint not found: {checkpoint}"
    run_dir = checkpoint.parent
    json_path = analysis_datasets_dir(run_dir) / "alive_components_per_position.json"
    assert json_path.exists(), (
        f"missing {json_path.name}; run find_alive_components first (its CI threshold becomes "
        "this explorer's noise floor)"
    )
    pos_list = [int(p) for p in str(positions).split(",")]
    data: PerPosition = json.loads(json_path.read_text())
    ab, n, _ = _parse_ab(data, op)
    assert min(a for a, _ in ab.values()) == 1, "expected operands to start at 1"
    layer = int(parse_module_name(next(iter(_SHORT)))[0])

    logger.info(f"building grids for positions {pos_list} over {len(ab)} prompts ({n}x{n})")
    grids = _build_grids(data, ab, pos_list, n)

    payload = _Payload(
        meta={
            "run_id": run_dir.name,
            "op": op,
            "n": n,
            "ci_thr": ci_thr,
            "positions": pos_list,
            "position_labels": {1: "after a", 2: "the + sign", 3: "after b", 4: "= (answer)"},
            "matrices": list(_SHORT.values()),
            "max_modulus": max_modulus,
            "has_weights": not no_weights,
        }
    )

    for short in _SHORT.values():
        per_component: list[dict[str, Any] | None] = []
        for c in range(128):
            entry: dict[str, Any] = {}
            for pos in pos_list:
                grid = grids[pos][short][c]
                analysis = _analyse_component(grid, max_modulus)
                if analysis is None:
                    continue
                analysis["sparse"] = _sparse_grid(grid)
                entry[str(pos)] = analysis
            per_component.append(entry if entry else None)
        payload.components[short] = per_component

    if not no_weights:
        logger.info("loading component weights (mmap) for neuron-overlap analysis")
        w = _load_weights(checkpoint, layer)
        payload.neurons = {
            "gate_proj": _neuron_fingerprints(w.gate_write, top_k),
            "up_proj": _neuron_fingerprints(w.up_write, top_k),
            "down_proj": _neuron_fingerprints(w.down_read, top_k),
        }
        payload.overlap = {
            "gate_up": _top_partners(w.gate_write, w.up_write, top_k),
            "gate_down": _top_partners(w.gate_write, w.down_read, top_k),
            "up_down": _top_partners(w.up_write, w.down_read, top_k),
        }

    out_dir = (
        Path(output_dir).expanduser() if output_dir else analysis_dir(run_dir) / "addition_explorer"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    data_js = out_dir / "data.js"
    payload_json = json.dumps(
        {
            "meta": payload.meta,
            "components": payload.components,
            "neurons": payload.neurons,
            "overlap": payload.overlap,
        },
        separators=(",", ":"),
    )
    data_js.write_text(f"window.PD_DATA = {payload_json};\n")
    assert _APP_TEMPLATE.exists(), f"app template missing: {_APP_TEMPLATE}"
    shutil.copyfile(_APP_TEMPLATE, out_dir / "index.html")

    size_mb = data_js.stat().st_size / 1e6
    logger.info(f"wrote explorer to {out_dir} (data.js {size_mb:.1f} MB) — open index.html")
    return out_dir


if __name__ == "__main__":
    fire.Fire(build_addition_explorer)
