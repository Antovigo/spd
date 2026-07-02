"""Build a GPU-free HTML applet: the activations in the basis of Feucht's fitted Fourier probes.

Projects the collected resid-layer-output activations (the `1..max_value` `a+b=` grid) onto each
fitted probe's **predicted `(cos, sin)`** plane (`w_cos·x + b_cos`, `w_sin·x + b_sin`) — a clean
feature traces the unit circle. One plot per period for a chosen **basis variable** (`a`, `b`, or
`a+b`); colour the points by `a`, `b`, or `a+b`, optionally reduced to `(value − offset) mod m`.
Scroll to zoom, drag to pan. When the sin axis is degenerate (period 2) the 2nd axis falls back to
the top activation-variance direction ⊥ the cos direction (rescaled so the residue split shows).

Consumes `resid_activations.npz` + `probes.json` from `collect_resid_activations` /
`fit_fourier_probes`. CPU-only (no forward pass).

Usage:
    python -m param_decomp_lab.scripts.validation.probes.build_probe_scatter <probes.json> \
        [--activations=PATH] [--output-dir=PATH]

Output: `<out-dir>/probe_scatter/{index.html,data.js}` (default beside `probes.json`).
"""

import base64
import json
import shutil
from pathlib import Path
from typing import Any

import fire
import numpy as np
from numpy.typing import NDArray

from param_decomp.log import logger

_APP_TEMPLATE = Path(__file__).with_name("probe_scatter_app.html")


def _b64_f16(arr: NDArray[Any]) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.float16).tobytes()).decode("ascii")


def _b64_i16(arr: NDArray[Any]) -> str:
    return base64.b64encode(np.ascontiguousarray(arr, dtype=np.int16).tobytes()).decode("ascii")


def _plane(probe: dict[str, Any], fallback: NDArray[np.float32]) -> dict[str, Any]:
    """Predicted-`(cos, sin)` projection frame for one probe, with an orthonormal fallback when the
    sin axis is degenerate (period 2) — `e2` = top activation-variance direction ⊥ `w_cos`, its
    drawn coord rescaled to the cos-axis spread so the two residue classes stay visible."""
    w_cos = np.asarray(probe["w_cos"], np.float32)
    w_sin = np.asarray(probe["w_sin"], np.float32)
    n1 = float(np.linalg.norm(w_cos))
    e1 = w_cos / max(n1, 1e-12)
    resid = w_sin - float(w_sin @ e1) * e1
    if float(np.linalg.norm(resid)) > 1e-3 * max(n1, 1e-12):
        return {
            "mode": "pred",
            "w_cos": w_cos,
            "b_cos": float(probe["b_cos"]),
            "w_sin": w_sin,
            "b_sin": float(probe["b_sin"]),
        }
    xc = fallback - fallback.mean(axis=0)
    xc = xc - np.outer(xc @ e1, e1)
    top = np.linalg.eigh(xc.T @ xc)[1][:, -1].astype(np.float32)
    top = top - float(top @ e1) * e1
    e2 = top / max(float(np.linalg.norm(top)), 1e-12)
    s1, s2 = float(np.std(fallback @ e1)), float(np.std(fallback @ e2))
    return {"mode": "ortho", "e1": e1.astype(np.float32), "e2": e2, "scale": s1 / max(s2, 1e-12)}


def _project(plane: dict[str, Any], x: NDArray[np.float32]) -> NDArray[np.float32]:
    if plane["mode"] == "pred":
        return np.stack(
            [x @ plane["w_cos"] + plane["b_cos"], x @ plane["w_sin"] + plane["b_sin"]], axis=1
        ).astype(np.float32)
    return np.stack([x @ plane["e1"], (x @ plane["e2"]) * plane["scale"]], axis=1).astype(
        np.float32
    )


def build_probe_scatter(
    probes_json: str,
    activations: str | None = None,
    output_dir: str | None = None,
) -> Path:
    probes_path = Path(probes_json).expanduser()
    assert probes_path.exists(), f"missing probes json: {probes_path}"
    payload = json.loads(probes_path.read_text())
    probes = payload["probes"]
    variables = payload["variables"]
    periods = [str(p) for p in payload["periods"]]

    npz_path = (
        Path(activations).expanduser() if activations else Path(payload["source"]).expanduser()
    )
    assert npz_path.exists(), f"missing resid activations: {npz_path}"
    data = np.load(npz_path)
    grid = data["resid"]
    g = int(grid.shape[0])
    x = grid.reshape(g * g, -1).astype(np.float32)
    aa, bb = np.meshgrid(data["a"].astype(np.int64), data["b"].astype(np.int64), indexing="ij")
    a_flat, b_flat = aa.reshape(-1), bb.reshape(-1)

    points_out: dict[str, list[str]] = {}
    centers_out: dict[str, list[list[float]]] = {}
    r2_out: dict[str, list[float | None]] = {}
    for var in variables:
        points_out[var], centers_out[var], r2_out[var] = [], [], []
        for pk in periods:
            plane = _plane(probes[var][pk], x)
            points_out[var].append(_b64_f16(_project(plane, x)))
            centers_out[var].append(
                [round(float(c), 4) for c in _project(plane, x.mean(axis=0, keepdims=True))[0]]
            )
            r2s = [
                probes[var][pk][k] for k in ("r2_cos", "r2_sin") if probes[var][pk][k] is not None
            ]
            r2_out[var].append(round(float(np.mean(r2s)), 4) if r2s else None)

    out = {
        "meta": {
            "variables": variables,
            "periods": [int(p) for p in payload["periods"]],
            "max_value": payload["max_value"],
            "layer": payload["layer"],
            "n": g * g,
            "r2": r2_out,
        },
        "a": _b64_i16(a_flat),
        "b": _b64_i16(b_flat),
        "points": points_out,
        "centers": centers_out,
    }

    out_dir = Path(output_dir).expanduser() if output_dir else probes_path.parent / "probe_scatter"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.js").write_text(
        f"window.PD_DATA = {json.dumps(out, separators=(',', ':'))};\n"
    )
    assert _APP_TEMPLATE.exists(), f"app template missing: {_APP_TEMPLATE}"
    shutil.copyfile(_APP_TEMPLATE, out_dir / "index.html")
    size_mb = (out_dir / "data.js").stat().st_size / 1e6
    logger.info(f"wrote probe-scatter applet ({size_mb:.1f} MB) → {out_dir} — open index.html")
    return out_dir


if __name__ == "__main__":
    fire.Fire(build_probe_scatter)
