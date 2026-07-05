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

Selecting an arrow opens a right panel: its **angle** to the current Fourier plane (0° = the direction
lies in the plane), and — for a subcomponent — the **`dir · activation(a, b)` heatmap** (its inner
activation for reads, write-projection for writes). The heatmap is reconstructed in-browser from a
per-site low-rank basis of the activation grid (`heat_rank` components), so any of the ~1.4k
subcomponents works without shipping a grid each. A **colour-by CI** mode tints the points by the
selected subcomponent's causal importance, read from the `inner_activations_add.tsv` `ci` column
(a, b ≤ 100; points outside are greyed). Neuron heatmaps are omitted (too many directions to basis).

Consumes a decomposition `<run>/model_<step>.pth` plus the shared `resid_activations.npz` +
`probes_<site>.json` (from `collect_resid_activations` / `fit_fourier_probes`; the probes' target
model must be the same base model the checkpoint decomposes) and, for CI colouring, the run's
`analysis/datasets/inner_activations_add.tsv`. CPU-only.

Usage:
    python -m param_decomp_lab.scripts.validation.probes.build_direction_scatter <checkpoint.pth> \
        <resid_activations.npz> [--layer=18] [--top-k=100] [--n-show=8000] [--heat-rank=48] \
        [--alive-tsv=<run>/analysis/datasets/alive_components.tsv] [--output-dir=PATH]

`--alive-tsv` restricts the subcomponent arrows / heatmaps to the alive components listed in
the TSV (neurons are unaffected); indices keep their original numbering.

Output: `<run>/analysis/direction_scatter/{index.html,data.js}` (default).
"""

import csv
import json
import shutil
from pathlib import Path
from typing import Any

import fire
import numpy as np
from numpy.typing import NDArray
from sklearn.utils.extmath import randomized_svd

from param_decomp.log import logger
from param_decomp_lab.scripts.validation.common import (
    RESID_SITES,
    analysis_datasets_dir,
    analysis_dir,
    b64_f16,
    b64_i8,
    b64_i16,
    b64_u8,
    load_component_uv,
    load_target_mlp_weights,
    read_alive_components,
)

_APP_TEMPLATE = Path(__file__).with_name("direction_scatter_app.html")
_MAT_CODE = {"g": 0, "u": 1, "d": 2}  # gate-read / up-read / down-write
_KINDS = ("neurons", "subcomponents")
_MLP = ("gate_proj", "up_proj", "down_proj")
_PROJ_MAT = {"gate_proj": "g", "up_proj": "u", "down_proj": "d"}
_HEAT_RANK = 48  # low-rank of each site's activation grid; heatmaps reconstruct in this subspace
_CI_N = 100  # inner_activations harvest grid side (a, b ∈ 1..100)


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
    kind: str,
    weights: dict[str, NDArray[np.float32]],
    uv: dict[str, tuple[Any, Any]],
    keep: "dict[str, NDArray[np.int64]] | None",
) -> tuple[NDArray[np.float32], NDArray[np.int8], NDArray[np.int64]]:
    """Residual-space read directions `[M, d_model]` + `g`/`u` matrix tag + unit index, gate then up.

    `keep` (subcomponents only) restricts each projection to those component indices; the
    returned indices stay original so arrow labels keep matching the harvest."""
    if kind == "neurons":
        g_dirs, u_dirs = weights["gate_proj"], weights["up_proj"]  # (d_ff, d_model) rows = reads
        g_ids, u_ids = np.arange(g_dirs.shape[0]), np.arange(u_dirs.shape[0])
    else:  # subcomponent read V[:,c] · ‖U[c]‖ (× neuron-space output norm) — symmetric with write
        (v_g, u_g), (v_u, u_u) = uv["gate_proj"], uv["up_proj"]  # V (d_model, C), U (C, d_ff)
        g_dirs = (v_g * np.linalg.norm(u_g, axis=1)).T  # (d_model, C) -> (C, d_model)
        u_dirs = (v_u * np.linalg.norm(u_u, axis=1)).T
        g_ids = np.arange(g_dirs.shape[0]) if keep is None else keep["gate_proj"]
        u_ids = np.arange(u_dirs.shape[0]) if keep is None else keep["up_proj"]
        g_dirs, u_dirs = g_dirs[g_ids], u_dirs[u_ids]
    dirs = np.concatenate([g_dirs, u_dirs], axis=0).astype(np.float32)
    mats = np.concatenate(
        [np.full(len(g_ids), _MAT_CODE["g"]), np.full(len(u_ids), _MAT_CODE["u"])]
    )
    return dirs, mats.astype(np.int8), np.concatenate([g_ids, u_ids])


def _write_candidates(
    kind: str,
    weights: dict[str, NDArray[np.float32]],
    uv: dict[str, tuple[Any, Any]],
    keep: "dict[str, NDArray[np.int64]] | None",
) -> tuple[NDArray[np.float32], NDArray[np.int8], NDArray[np.int64]]:
    """Residual-space write directions `[M, d_model]` + `d` tag + unit index (down only)."""
    if kind == "neurons":
        dirs = weights["down_proj"].T  # (d_model, d_ff) columns = write dirs -> (d_ff, d_model)
        ids = np.arange(dirs.shape[0])
    else:
        v_down, u_down = uv["down_proj"]  # V (d_ff, C), U (C, d_model)
        dirs = u_down * np.linalg.norm(v_down, axis=0)[:, None]  # U[c] · ‖V[:,c]‖ (input-vec norm)
        ids = np.arange(dirs.shape[0]) if keep is None else keep["down_proj"]
        dirs = dirs[ids]
    dirs = dirs.astype(np.float32)
    mats = np.full(dirs.shape[0], _MAT_CODE["d"], np.int8)
    return dirs, mats, ids


def _ortho_plane(probe: dict[str, Any]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Orthonormal basis `(e1, e2)` of the probe plane `span(w_cos, w_sin)`, for the direction→plane
    angle. `e2 = 0` when the plane is degenerate (period-2 `sin ≡ 0`), collapsing to the cos axis."""
    w_cos = np.asarray(probe["w_cos"], np.float32)
    w_sin = np.asarray(probe["w_sin"], np.float32)
    e1 = w_cos / max(float(np.linalg.norm(w_cos)), 1e-12)
    perp = w_sin - (w_sin @ e1) * e1
    n_perp = float(np.linalg.norm(perp))
    e2 = perp / n_perp if n_perp > 1e-6 else np.zeros_like(e1)
    return e1, e2


def _subcomp_dirs(
    uv: dict[str, tuple[Any, Any]],
    keep: "dict[str, NDArray[np.int64]] | None",
) -> tuple[list[str], NDArray[np.float32]]:
    """Raw residual-space direction per subcomponent (its inner-activation / write vector): gate/up
    read `V[:,c]`, down write `U[c]`. Returns matids (`"g0"`, `"u3"`, `"d7"`, …) and `[N, d_model]`.
    Unscaled by the gauge norm — the heatmap is `dir · activation`, whose (a, b) pattern is
    invariant to that per-component constant. `keep` restricts to those component indices."""
    matids: list[str] = []
    dirs: list[NDArray[np.float32]] = []
    for proj in ("gate_proj", "up_proj"):
        v = uv[proj][0]  # (d_model, C)
        ids = np.arange(v.shape[1]) if keep is None else keep[proj]
        dirs.append(v.T[ids].astype(np.float32))
        matids += [f"{_PROJ_MAT[proj]}{c}" for c in ids]
    u_down = uv["down_proj"][1]  # (C, d_model)
    ids = np.arange(u_down.shape[0]) if keep is None else keep["down_proj"]
    dirs.append(u_down[ids].astype(np.float32))
    matids += [f"d{c}" for c in ids]
    return matids, np.concatenate(dirs, axis=0)


def _read_ci_grids(tsv: Path) -> dict[str, NDArray[np.float32]]:
    """`{matid: (n, n) CI grid}` from an `inner_activations_<op>.tsv` `ci` column (a, b ∈ 1..n). `{}`
    if the file / column is absent, so CI colouring degrades gracefully. Each present subcomponent
    is assumed fully covered (the harvest sweeps the whole grid)."""
    if not tsv.exists():
        return {}
    grids: dict[str, NDArray[np.float32]] = {}
    with tsv.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        if "ci" not in (reader.fieldnames or []):
            return {}
        for row in reader:
            a, b = int(row["a"]), int(row["b"])
            if a > _CI_N or b > _CI_N:
                continue
            matid = _PROJ_MAT[row["matrix"].split(".")[-1]] + row["subcomponent"]
            grids.setdefault(matid, np.zeros((_CI_N, _CI_N), np.float32))[a - 1, b - 1] = float(
                row["ci"]
            )
    return grids


def build_direction_scatter(
    checkpoint: str,
    activations_npz: str,
    layer: int = 18,
    top_k: int = 100,
    n_show: int = 8000,
    heat_rank: int = _HEAT_RANK,
    alive_tsv: str | None = None,
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

    keep: dict[str, NDArray[np.int64]] | None = None
    if alive_tsv is not None:
        alive = read_alive_components(Path(alive_tsv).expanduser(), keep_projs=_MLP)
        keep = {
            proj: np.array(sorted({c.component for c in alive if c.proj == proj}), np.int64)
            for proj in _MLP
        }
        assert all(len(ids) for ids in keep.values()), f"a proj has no alive rows in {alive_tsv}"
        logger.info("alive filter: " + ", ".join(f"{p} {len(keep[p])}" for p in _MLP))

    weights = load_target_mlp_weights(ck, layer, _MLP)
    uv = load_component_uv(ck, layer, _MLP)
    d_model = weights["down_proj"].shape[0]
    read_cand = {k: _read_candidates(k, weights, uv, keep) for k in _KINDS}
    write_cand = {k: _write_candidates(k, weights, uv, keep) for k in _KINDS}
    dnorm = {  # ‖dir‖ per candidate row, for the direction→plane angle (uses the unit direction)
        "read": {k: np.linalg.norm(read_cand[k][0], axis=1) for k in _KINDS},
        "write": {k: np.linalg.norm(write_cand[k][0], axis=1) for k in _KINDS},
    }
    sub_matids, sub_dirs = _subcomp_dirs(uv, keep)  # raw residual dir per subcomp, for heatmaps

    points_out: dict[str, dict[str, list[str]]] = {}
    r2_out: dict[str, dict[str, list[float | None]]] = {}
    arrows_out: dict[str, dict[str, dict[str, list[dict[str, str]]]]] = {k: {} for k in _KINDS}
    heat_p: dict[str, str] = {}  # (n_total, R) low-rank spatial basis per site
    heat_qd: dict[str, str] = {}  # (N_sub, R) direction coords in that basis
    heat_m: dict[str, str] = {}  # (N_sub,) direction · site-mean
    all_norms: list[NDArray[np.float32]] = []
    for site, payload in site_payloads.items():
        probes = payload["probes"]
        x = data[f"resid_{site}"].reshape(g * g, -1).astype(np.float32)
        # Low-rank of the (mean-centred) activation grid: dir·act(a,b) ≈ (dir·mean) + P @ (Vᵀ·dir),
        # so the browser reconstructs any subcomponent's dot-product heatmap from a shared per-site
        # basis P and a small per-direction vector. Exact within the top-`heat_rank` subspace.
        mean_s = x.mean(axis=0)
        u_svd, s_svd, vt_svd = randomized_svd(x - mean_s, n_components=heat_rank, random_state=0)
        p_lr = (u_svd * s_svd).astype(np.float32)
        m_s = (sub_dirs @ mean_s).astype(np.float32)
        heat_p[site] = b64_f16(p_lr)
        heat_qd[site] = b64_f16((sub_dirs @ vt_svd.T).astype(np.float32))
        heat_m[site] = b64_f16(m_s)
        if site == "post":
            approx = m_s[0] + p_lr @ (sub_dirs[0] @ vt_svd.T)
            corr = float(np.corrcoef(approx, x @ sub_dirs[0])[0, 1])
            logger.info(f"heatmap low-rank R={heat_rank}: {sub_matids[0]} recon corr={corr:.4f}")
        points_out[site], r2_out[site] = {}, {}
        for k in _KINDS:
            arrows_out[k][site] = {}
        for var in variables:
            ps = periods_by_var[var]
            points_out[site][var], r2_out[site][var] = [], []
            axes = np.empty((d_model, 2 * len(ps)), np.float32)  # [.. axis_cos_j, axis_sin_j ..]
            ortho = np.empty(
                (d_model, 2 * len(ps)), np.float32
            )  # orthonormal plane basis, for angle
            for j, p in enumerate(ps):
                probe = probes[var][str(p)]
                axis_c, axis_s, center = _frame(probe, x)
                axes[:, 2 * j], axes[:, 2 * j + 1] = axis_c, axis_s
                ortho[:, 2 * j], ortho[:, 2 * j + 1] = _ortho_plane(probe)
                cloud = np.stack([x @ axis_c - center[0], x @ axis_s - center[1]], axis=1)
                points_out[site][var].append(b64_f16(cloud[subset]))
                r2s = [probe[c] for c in ("r2_cos", "r2_sin") if probe[c] is not None]
                r2_out[site][var].append(round(float(np.mean(r2s)), 4) if r2s else None)
            role = "read" if var in ("a", "b") else "write"
            cand = read_cand if role == "read" else write_cand
            for k in _KINDS:
                dirs, mats, idx = cand[k]
                proj = dirs @ axes  # (M, 2P); NOT centred — arrows are increments from ring centre
                oproj = dirs @ ortho  # (M, 2P) onto the orthonormal plane basis, for the angle
                dn = dnorm[role][k]
                arrows_out[k][site][var] = []
                for j, p in enumerate(ps):
                    c, s = proj[:, 2 * j], proj[:, 2 * j + 1]
                    norm = np.sqrt(c * c + s * s)
                    kk = min(top_k, norm.shape[0])
                    top = np.argpartition(norm, -kk)[-kk:]
                    top = top[np.argsort(norm[top])[::-1]]
                    inplane = np.sqrt(oproj[top, 2 * j] ** 2 + oproj[top, 2 * j + 1] ** 2)
                    ang = np.degrees(np.arccos(np.clip(inplane / np.maximum(dn[top], 1e-12), 0, 1)))
                    arrows_out[k][site][var].append(
                        {
                            "mat": b64_i8(mats[top]),
                            "idx": b64_i16(idx[top].astype(np.int16)),
                            "cs": b64_f16(np.stack([c[top], s[top]], axis=1)),
                            "ang": b64_f16(ang.astype(np.float32)),
                        }
                    )
                    all_norms.append(norm[top])
                    if site == "post" and var == "a+b" and p == 10 and k == "neurons":
                        logger.info(
                            f"sanity post/a+b/T10 top-8 write neurons: {idx[top][:8].tolist()}"
                        )
            logger.info(f"{site}/{var}: {len(ps)} planes")

    run_dir = ck.parent
    ci_grids = _read_ci_grids(analysis_datasets_dir(run_dir) / "inner_activations_add.tsv")
    ci_out = {mid: b64_u8(np.clip(grid, 0.0, 1.0) * 255.0) for mid, grid in ci_grids.items()}
    logger.info(f"CI colour data for {len(ci_out)} subcomponents (a, b ≤ {_CI_N})")

    norms = np.concatenate(all_norms)
    out = {
        "meta": {
            "run": run_dir.name,
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
            "heat_rank": heat_rank,
            "has_ci": bool(ci_out),
            "ci_n": _CI_N,
            # per-variable held-out value ranges of the probe fit (block-held-out probes
            # only; None for full-fit probes) — drives the fit-vs-held-out colour mode
            "block_ranges": first.get("block_ranges"),
            # alive-subcomponent counts per tag when --alive-tsv filtered the arrows
            "alive": None if keep is None else {_PROJ_MAT[p]: len(keep[p]) for p in _MLP},
        },
        "a": b64_i16(a_flat[subset]),
        "b": b64_i16(b_flat[subset]),
        "points": points_out,
        "arrows": arrows_out,
        "heat": {"matids": sub_matids, "p": heat_p, "qd": heat_qd, "m": heat_m},
        "ci": ci_out,
    }

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
