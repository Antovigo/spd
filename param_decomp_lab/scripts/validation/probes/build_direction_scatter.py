"""Build a GPU-free HTML applet: probe-scatter point clouds with neuron / subcomponent arrows.

Like `build_probe_scatter`, but tied to a *decomposition* run (its checkpoint) and dropped in that
run's `analysis/` folder. On top of each Fourier-plane point cloud it overlays arrows for the
directions of MLP **neurons** or the run's **subcomponents** (dropdown):

- **read** directions (`gate`/`up`), shown on the operand planes (`a`, `b`) — how strongly the
  unit reads the operand's period-T circular feature. Tagged `g` / `u`.
- **write** directions (`down`), shown on the sum plane (`a+b`) — how strongly the unit writes it.
  Tagged `d`. This is Feucht Fig 9c (down_proj rows onto the T-Fourier plane).

Fig 9c scale: everything is projected onto the **unit-normalised** probe directions `d = w/‖w‖`,
and the activation cloud is recomputed in that same normalised, mean-centred frame — so an arrow
`v·d` and a centred activation `x·d − mean` share one scale, and arrows read as increments from the
ring centre. Neuron read/write are the raw gate/up row / down column (Fig 9c). Subcomponents scale
the residual-space vector by the norm of the component's *other* (14336-d) vector, symmetric across
read and write: write = `U[c]·‖V[:,c]‖`, read = `V[:,c]·‖U[c]‖`. This is gauge-invariant (a rank-1
`u vᵀ` is free under `u→αu, v→v/α`) and equals the component's residual move per one std of its
activation — the same quantity a neuron's raw row/column is, so the two coexist on one scale.
Only the top-`top_k` units by projected 2D norm are shipped per plane; the threshold slider filters.

Consumes a decomposition `<run>/model_<step>.pth` plus the shared `resid_activations.npz` +
`probes_<site>.json` (from `collect_resid_activations` / `fit_fourier_probes`; the probes' target
model must be the same base model the checkpoint decomposes). CPU-only.

Usage:
    python -m param_decomp_lab.scripts.validation.probes.build_direction_scatter <checkpoint.pth> \
        <resid_activations.npz> [--layer=18] [--top-k=100] [--n-show=10000] [--output-dir=PATH]

Output: `<run>/analysis/direction_scatter/{index.html,data.js}` (default).
"""

import json
import shutil
from pathlib import Path
from typing import Any

import fire
import numpy as np
from numpy.typing import NDArray

from param_decomp.log import logger
from param_decomp_lab.scripts.validation.common import (
    RESID_SITES,
    analysis_dir,
    b64_f16,
    b64_i8,
    b64_i16,
    load_component_uv,
    load_target_mlp_weights,
)

_APP_TEMPLATE = Path(__file__).with_name("direction_scatter_app.html")
_MAT_CODE = {"g": 0, "u": 1, "d": 2}  # gate-read / up-read / down-write
_KINDS = ("neurons", "subcomponents")
_MLP = ("gate_proj", "up_proj", "down_proj")


def _frame(
    probe: dict[str, Any], x: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """`(axis_cos, axis_sin, center)` for the normalised, mean-centred plane (Fig 9c scale).

    Axes are the unit probe directions `w/‖w‖`. When the sin axis is degenerate (period 2) it
    falls back to the top activation-variance direction ⊥ cos, rescaled to the cos-axis spread.
    `center` is the plane-mean of the activations, subtracted to seat the ring at the origin.
    """
    w_cos = np.asarray(probe["w_cos"], np.float32)
    w_sin = np.asarray(probe["w_sin"], np.float32)
    n_cos = float(np.linalg.norm(w_cos))
    axis_c = w_cos / max(n_cos, 1e-12)
    n_sin = float(np.linalg.norm(w_sin))
    if n_sin > 1e-6:
        axis_s = w_sin / n_sin
    else:  # period-2 sin ≡ 0: orthonormal fallback, rescaled so the residue split stays visible
        xc = x - x.mean(axis=0)
        xc = xc - np.outer(xc @ axis_c, axis_c)
        e2 = np.linalg.eigh(xc.T @ xc)[1][:, -1].astype(np.float32)
        e2 = e2 / max(float(np.linalg.norm(e2)), 1e-12)
        s1, s2 = float(np.std(x @ axis_c)), float(np.std(x @ e2))
        axis_s = e2 * (s1 / max(s2, 1e-12))
    center = np.array([float((x @ axis_c).mean()), float((x @ axis_s).mean())], np.float32)
    return axis_c, axis_s, center


def _read_candidates(
    kind: str, weights: dict[str, NDArray[np.float32]], uv: dict[str, tuple[Any, Any]]
) -> tuple[NDArray[np.float32], NDArray[np.int8], NDArray[np.int64]]:
    """Residual-space read directions `[M, d_model]` + `g`/`u` matrix tag + unit index, gate then up."""
    if kind == "neurons":
        g_dirs, u_dirs = weights["gate_proj"], weights["up_proj"]  # (d_ff, d_model) rows = reads
    else:  # subcomponent read V[:,c] · ‖U[c]‖ (× neuron-space output norm) — symmetric with write
        (v_g, u_g), (v_u, u_u) = uv["gate_proj"], uv["up_proj"]  # V (d_model, C), U (C, d_ff)
        g_dirs = (v_g * np.linalg.norm(u_g, axis=1)).T  # (d_model, C) -> (C, d_model)
        u_dirs = (v_u * np.linalg.norm(u_u, axis=1)).T
    ids = np.arange(g_dirs.shape[0])
    dirs = np.concatenate([g_dirs, u_dirs], axis=0).astype(np.float32)
    mats = np.concatenate([np.full(len(ids), _MAT_CODE["g"]), np.full(len(ids), _MAT_CODE["u"])])
    return dirs, mats.astype(np.int8), np.concatenate([ids, ids])


def _write_candidates(
    kind: str, weights: dict[str, NDArray[np.float32]], uv: dict[str, tuple[Any, Any]]
) -> tuple[NDArray[np.float32], NDArray[np.int8], NDArray[np.int64]]:
    """Residual-space write directions `[M, d_model]` + `d` tag + unit index (down only)."""
    if kind == "neurons":
        dirs = weights["down_proj"].T  # (d_model, d_ff) columns = write dirs -> (d_ff, d_model)
    else:
        v_down, u_down = uv["down_proj"]  # V (d_ff, C), U (C, d_model)
        dirs = u_down * np.linalg.norm(v_down, axis=0)[:, None]  # U[c] · ‖V[:,c]‖ (input-vec norm)
    dirs = dirs.astype(np.float32)
    mats = np.full(dirs.shape[0], _MAT_CODE["d"], np.int8)
    return dirs, mats, np.arange(dirs.shape[0])


def build_direction_scatter(
    checkpoint: str,
    activations_npz: str,
    layer: int = 18,
    top_k: int = 100,
    n_show: int = 10000,
    output_dir: str | None = None,
) -> Path:
    ck = Path(checkpoint).expanduser()
    assert ck.exists(), f"missing checkpoint: {ck}"
    npz_path = Path(activations_npz).expanduser()
    assert npz_path.exists(), f"missing resid activations: {npz_path}"
    data = np.load(npz_path)
    g = int(data["a"].shape[0])
    site_payloads = {
        s: json.loads(p.read_text())
        for s in RESID_SITES
        if (p := npz_path.with_name(f"probes_{s}.json")).exists()
    }
    assert site_payloads, f"no probes_<site>.json beside {npz_path}"
    first = next(iter(site_payloads.values()))
    variables = first["variables"]
    periods_by_var = {v: [int(p) for p in first["periods_by_variable"][v]] for v in variables}

    aa, bb = np.meshgrid(data["a"].astype(np.int64), data["b"].astype(np.int64), indexing="ij")
    a_flat, b_flat = aa.reshape(-1), bb.reshape(-1)
    n_show = min(n_show, g * g)
    subset = np.sort(np.random.default_rng(0).choice(g * g, size=n_show, replace=False))

    weights = load_target_mlp_weights(ck, layer, _MLP)
    uv = load_component_uv(ck, layer, _MLP)
    d_model = weights["down_proj"].shape[0]
    read_cand = {k: _read_candidates(k, weights, uv) for k in _KINDS}
    write_cand = {k: _write_candidates(k, weights, uv) for k in _KINDS}

    points_out: dict[str, dict[str, list[str]]] = {}
    r2_out: dict[str, dict[str, list[float | None]]] = {}
    arrows_out: dict[str, dict[str, dict[str, list[dict[str, str]]]]] = {k: {} for k in _KINDS}
    all_norms: list[NDArray[np.float32]] = []
    for site, payload in site_payloads.items():
        probes = payload["probes"]
        x = data[f"resid_{site}"].reshape(g * g, -1).astype(np.float32)
        points_out[site], r2_out[site] = {}, {}
        for k in _KINDS:
            arrows_out[k][site] = {}
        for var in variables:
            ps = periods_by_var[var]
            points_out[site][var], r2_out[site][var] = [], []
            axes = np.empty((d_model, 2 * len(ps)), np.float32)  # [.. axis_cos_j, axis_sin_j ..]
            for j, p in enumerate(ps):
                probe = probes[var][str(p)]
                axis_c, axis_s, center = _frame(probe, x)
                axes[:, 2 * j], axes[:, 2 * j + 1] = axis_c, axis_s
                cloud = np.stack([x @ axis_c - center[0], x @ axis_s - center[1]], axis=1)
                points_out[site][var].append(b64_f16(cloud[subset]))
                r2s = [probe[c] for c in ("r2_cos", "r2_sin") if probe[c] is not None]
                r2_out[site][var].append(round(float(np.mean(r2s)), 4) if r2s else None)
            cand = read_cand if var in ("a", "b") else write_cand
            for k in _KINDS:
                dirs, mats, idx = cand[k]
                proj = dirs @ axes  # (M, 2P); NOT centred — arrows are increments from ring centre
                arrows_out[k][site][var] = []
                for j, p in enumerate(ps):
                    c, s = proj[:, 2 * j], proj[:, 2 * j + 1]
                    norm = np.sqrt(c * c + s * s)
                    kk = min(top_k, norm.shape[0])
                    top = np.argpartition(norm, -kk)[-kk:]
                    top = top[np.argsort(norm[top])[::-1]]
                    arrows_out[k][site][var].append(
                        {
                            "mat": b64_i8(mats[top]),
                            "idx": b64_i16(idx[top].astype(np.int16)),
                            "cs": b64_f16(np.stack([c[top], s[top]], axis=1)),
                        }
                    )
                    all_norms.append(norm[top])
                    if site == "post" and var == "a+b" and p == 10 and k == "neurons":
                        logger.info(
                            f"sanity post/a+b/T10 top-8 write neurons: {idx[top][:8].tolist()}"
                        )
            logger.info(f"{site}/{var}: {len(ps)} planes")

    norms = np.concatenate(all_norms)
    out = {
        "meta": {
            "run": ck.parent.name,
            "sites": list(site_payloads),
            "variables": variables,
            "periods": periods_by_var,
            "r2": r2_out,
            "canonical": [2, 5, 10, 20, 50, 100],
            "kinds": list(_KINDS),
            "mat_labels": {str(v): k for k, v in _MAT_CODE.items()},
            "top_k": top_k,
            "arrow_norm_hi": round(float(np.quantile(norms, 0.995)), 4),
            "arrow_norm_default": round(float(np.quantile(norms, 0.9)), 4),
            "max_value": first["max_value"],
            "max_period": first["max_period"],
            "layer": first["layer"],
            "n_total": g * g,
            "n_show": n_show,
        },
        "a": b64_i16(a_flat[subset]),
        "b": b64_i16(b_flat[subset]),
        "points": points_out,
        "arrows": arrows_out,
    }

    run_dir = ck.parent
    out_dir = (
        Path(output_dir).expanduser() if output_dir else analysis_dir(run_dir) / "direction_scatter"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.js").write_text(
        f"window.PD_DATA = {json.dumps(out, separators=(',', ':'))};\n"
    )
    assert _APP_TEMPLATE.exists(), f"app template missing: {_APP_TEMPLATE}"
    shutil.copyfile(_APP_TEMPLATE, out_dir / "index.html")
    size_mb = (out_dir / "data.js").stat().st_size / 1e6
    logger.info(f"wrote direction-scatter applet ({size_mb:.1f} MB) → {out_dir} — open index.html")
    return out_dir


if __name__ == "__main__":
    fire.Fire(build_direction_scatter)
