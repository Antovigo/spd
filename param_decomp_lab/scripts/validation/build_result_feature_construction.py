"""Before/after-MLP18 Fourier-plane scatter with in-browser neuron / subcomponent ablation.

Does MLP 18 build the result (`a+b`) circular features from scratch, or add to structure
already present in the residual stream? And which neurons / subcomponents of a decomposition
build that structure? The applet shows one plot per canonical period in the probes' predicted
`(cos, sin)` frame, five rows sharing one zoomable view per column:

1. residual **before** the MLP, on the probes fit **before** it (`probes_pre.json`, site `pre`)
2. residual **before** the MLP, on the probes fit **after** it (`probes_post.json`, site `post`)
3. residual **after** the MLP, on the post probes
4. residual after the MLP on the post probes, **with one neuron or subcomponent ablated**
5. residual after the **alive-components-only MLP** on the post probes: the MLP recomputed
   with every decomposed matrix reconstructed from just the applet's alive subcomponents
   (binary mask, delta off) — how much of the representation the kept circuit rebuilds

Rows 1 vs 2 separate what the pre-MLP structure looks like in its own best frame from how much
of it already lies in the *final* representation's frame; rows 2 vs 3 show what the MLP adds in
that frame. Hovering a point highlights the same prompt in every plot.

The ablated row handles **one item at a time**, exactly on the full grid:

- **neurons** — the MLP output is additive over neurons, so ablating one removes
  `act_j · (w · W_down[:, j])` from the post projection. Only census neurons whose measured
  ablation `max_kl` exceeds `--kl-thr` are selectable; their post-SwiGLU activation grids ship
  fp16.
- **down subcomponents** — removing rank-1 `U_c V_c^T` from `W_down` subtracts
  `(h · V_c) · (w · U_c)`.
- **gate/up subcomponents** — the SwiGLU is re-evaluated at build time with the component's
  rank-1 term removed from its preactivation; the exact full-grid projected delta ships per
  component.

CPU-only (no forward pass; mmap for weights). The op is fixed to `add` — the probes' result
variable is `a+b`.

Usage:
    python -m param_decomp_lab.scripts.validation.build_result_feature_construction \
        <model_path> [--probes-dir=.../fourier_probes] [--census-dir=PATH] [--kl-thr=0.01] \
        [--last-ci-thr=0.01] [--periods=2,5,10,20,50,100] [--output-dir=PATH]

Reads: `probes_post.json` + `probes_pre.json` (default dir
`<PARAM_DECOMP_OUT_DIR>/runs/fourier_probes/`), the run's `hidden_activations_add.npz` /
`alive_filtered_add.tsv` / `subcomp_periods_add.tsv` / `inner_activations_add.tsv` (whose
last-token `ci` column drops alive components never causally important at the `=` position:
max CI < `--last-ci-thr`), the census `candidates.tsv` (+ `ablation_full_add.npz` when
present, overriding the screen KL), and the checkpoint U/V + frozen MLP weights (mmap).

Output: `<run_dir>/analysis/result_feature_construction/{index.html,data.js}`.
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
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.scripts.validation.common import (
    MLP_MATRICES,
    analysis_datasets_dir,
    analysis_dir,
    b64_f16,
    load_component_uv,
    load_target_mlp_weights,
    read_mean_ci,
    read_subcomp_period_groups,
)
from param_decomp_lab.scripts.validation.neurons.common import NEURONS_DIR, silu_combine

_APP_TEMPLATE = Path(__file__).with_name("result_feature_construction_app.html")
_PROJ_TAG = {"gate_proj": "g", "up_proj": "u", "down_proj": "d"}
_DEFAULT_PERIODS = "2,5,10,20,50,100"
_CIRCUIT_CHUNK = 2000  # prompts per chunk for the alive-only MLP row (bounds the [.., d_int] mats)


def _probe_axes(
    probes: dict[str, Any], periods: list[int], ref_cloud: NDArray[np.float32]
) -> tuple[NDArray[np.float32], NDArray[np.float32], list[float]]:
    """Per period, the predicted-`(cos, sin)` frame: `W [d_model, 2P]`, `bias [2P]`, mean r².

    A degenerate sin probe (period 2, `sin ≡ 0`) falls back to the top variance direction of
    `ref_cloud` (the residual cloud the probes were fit on) ⊥ the cos axis, rescaled to the
    cos-prediction spread and mean-centred, so the residue split stays a 2D plot (as in
    `build_fourier_scatter`).
    """
    d_model = ref_cloud.shape[1]
    w = np.zeros((d_model, 2 * len(periods)), np.float32)
    b = np.zeros(2 * len(periods), np.float32)
    r2s: list[float] = []
    for j, t in enumerate(periods):
        probe = probes[str(t)]
        w_cos = np.asarray(probe["w_cos"], np.float32)
        w[:, 2 * j] = w_cos
        b[2 * j] = float(probe["b_cos"])
        w_sin = None if probe["w_sin"] is None else np.asarray(probe["w_sin"], np.float32)
        if w_sin is not None and float(np.linalg.norm(w_sin)) > 1e-6:
            w[:, 2 * j + 1] = w_sin
            b[2 * j + 1] = float(probe["b_sin"])
        else:
            e1 = w_cos / max(float(np.linalg.norm(w_cos)), 1e-12)
            xc = ref_cloud - ref_cloud.mean(axis=0)
            xc = xc - np.outer(xc @ e1, e1)
            e2 = np.linalg.eigh(xc.T @ xc)[1][:, -1].astype(np.float32)
            e2 /= max(float(np.linalg.norm(e2)), 1e-12)
            scale = float(np.std(ref_cloud @ w_cos + b[2 * j])) / max(
                float(np.std(ref_cloud @ e2)), 1e-12
            )
            w[:, 2 * j + 1] = e2 * scale
            b[2 * j + 1] = -float((ref_cloud @ e2).mean()) * scale
        r2 = [probe[k] for k in ("r2_cos", "r2_sin") if probe.get(k) is not None]
        r2s.append(round(float(np.mean(r2)), 4))
    return w, b, r2s


def _max_last_token_ci(inner_tsv: Path) -> dict[tuple[str, int], float]:
    """`(proj, component) -> max last-token CI over the grid`, from the `ci` column of an
    `inner_activations_<op>.tsv`."""
    assert inner_tsv.exists(), f"missing inner activations: {inner_tsv}"
    out: dict[tuple[str, int], float] = {}
    with inner_tsv.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        assert "ci" in (reader.fieldnames or []), f"no ci column in {inner_tsv}"
        for row in reader:
            key = (row["matrix"].split(".")[-1], int(row["subcomponent"]))
            ci = float(row["ci"])
            if ci > out.get(key, 0.0):
                out[key] = ci
    return out


def _neuron_kl(census_dir: Path) -> dict[int, float]:
    """Neuron → measured ablation max KL on addition: the screen value from `candidates.tsv`,
    overridden by the full-grid ablation npz where it covers the neuron (unaliased)."""
    cand_path = census_dir / "candidates.tsv"
    assert cand_path.exists(), f"missing census candidates: {cand_path}"
    with cand_path.open() as f:
        kl = {int(r["neuron"]): float(r["max_kl_add"]) for r in csv.DictReader(f, delimiter="\t")}
    full_path = census_dir / "ablation_full_add.npz"
    if full_path.exists():
        full = np.load(full_path)
        for nid, grid in zip(full["neuron_ids"], full["kl"], strict=True):
            kl[int(nid)] = float(grid.astype(np.float32).max())
    else:
        logger.info(f"no {full_path.name}; neuron KL uses the stride-5 screen values only")
    return kl


def _gu_dact(
    gate: NDArray[np.float32],
    up: NDArray[np.float32],
    act: NDArray[np.float32],
    proj: str,
    inner: NDArray[np.float32],
    u_row: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Exact post-SwiGLU activation delta `[N, d_int]` for ablating one gate/up component:
    the SwiGLU re-evaluated with the component's rank-1 term removed from its preactivation."""
    d = np.outer(inner, u_row)
    if proj == "gate_proj":
        return silu_combine(gate - d, up) - act
    return silu_combine(gate, up - d) - act


def build_result_feature_construction(
    model_path: str,
    probes_dir: str | None = None,
    census_dir: str | None = None,
    kl_thr: float = 0.01,
    last_ci_thr: float = 0.01,
    periods: str = _DEFAULT_PERIODS,
    output_dir: str | None = None,
) -> Path:
    ck = Path(model_path).expanduser()
    assert ck.exists(), f"missing checkpoint: {ck}"
    run_dir = ck.parent
    datasets = analysis_datasets_dir(run_dir)
    period_list = [int(p) for p in str(periods).split(",")]

    pdir = (
        Path(probes_dir).expanduser()
        if probes_dir
        else PARAM_DECOMP_OUT_DIR / "runs" / "fourier_probes"
    )
    probe_payloads: dict[str, dict[str, Any]] = {}
    for site in ("post", "pre"):
        path = pdir / f"probes_{site}.json"
        assert path.exists(), f"missing probes JSON: {path}"
        probe_payloads[site] = json.loads(path.read_text())
        assert probe_payloads[site]["site"] == site, (
            f"{path.name} carries site {probe_payloads[site]['site']!r}, expected {site!r}"
        )

    npz_path = datasets / "hidden_activations_add.npz"
    assert npz_path.exists(), f"missing hidden activations: {npz_path}"
    d = np.load(npz_path)
    assert str(d["op"]) == "add", f"hidden activations are for op {d['op']}, need add"
    layer = int(d["layer"])
    for site in ("post", "pre"):
        assert layer == int(probe_payloads[site]["layer"]), (
            f"probe layer {probe_payloads[site]['layer']} ({site}) != run layer {layer}"
        )
    n = int(d["a"].shape[0])
    n_prompts = n * n
    resid_pre = d["resid_pre_mlp"].reshape(n_prompts, -1).astype(np.float32)
    mlp_out = d["mlp_output"].reshape(n_prompts, -1).astype(np.float32)
    x = d["mlp_input"].reshape(n_prompts, -1).astype(np.float32)
    gate = d["gate_preact"].reshape(n_prompts, -1).astype(np.float32)
    up = d["up_preact"].reshape(n_prompts, -1).astype(np.float32)
    resid_post = resid_pre + mlp_out

    weights = load_target_mlp_weights(ck, layer, MLP_MATRICES)
    uv = load_component_uv(ck, layer, MLP_MATRICES)
    act = silu_combine(gate, up)

    # the post frame carries the ablation math (rows 2-5); the pre frame only projects row 1
    w_post, b_post, r2_post = _probe_axes(
        probe_payloads["post"]["probes"]["a+b"], period_list, resid_post
    )
    w_pre, b_pre, r2_pre = _probe_axes(
        probe_payloads["pre"]["probes"]["a+b"], period_list, resid_pre
    )
    wd = weights["down_proj"].T @ w_post  # [d_int, 2P] — each neuron's write, probe-projected
    recon_err = float(np.abs(act @ wd - mlp_out @ w_post).max() / (mlp_out @ w_post).std())
    assert recon_err < 0.1, f"neuron-sum vs mlp_output projection mismatch: {recon_err:.3f}"
    for site, r2s in (("post", r2_post), ("pre", r2_pre)):
        logger.info(
            f"probe planes [{site}]: "
            + ", ".join(f"T={t} r2={r:.2f}" for t, r in zip(period_list, r2s, strict=True))
        )

    # --- neurons: measured-causal set, exact activation grids -------------------------------
    kl = _neuron_kl(Path(census_dir).expanduser() if census_dir else NEURONS_DIR)
    neuron_ids = sorted((j for j, k in kl.items() if k > kl_thr), key=lambda j: -kl[j])
    assert neuron_ids, f"no census neuron clears max_kl > {kl_thr}"
    logger.info(f"{len(neuron_ids)} neurons with measured ablation max KL > {kl_thr}")

    # --- subcomponents: alive on add AND causally important at the last token ----------------
    # The alive filter may include components acting only at operand positions; this applet
    # reads/ablates at the `=` token, so those are dropped via the per-prompt last-token CI.
    mean_ci = read_mean_ci(datasets / "alive_filtered_add.tsv")
    period_groups = read_subcomp_period_groups(datasets / "subcomp_periods_add.tsv")
    max_ci = _max_last_token_ci(datasets / "inner_activations_add.tsv")
    alive_all = sorted(
        ((proj, c) for (proj, c) in mean_ci if proj in MLP_MATRICES),
        key=lambda pc: (MLP_MATRICES.index(pc[0]), pc[1]),
    )
    alive = [pc for pc in alive_all if max_ci.get(pc, 0.0) >= last_ci_thr]
    assert alive, f"no alive component reaches last-token CI ≥ {last_ci_thr}"
    gu_comps = [(p, c) for p, c in alive if p != "down_proj"]
    down_comps = [(p, c) for p, c in alive if p == "down_proj"]
    logger.info(
        f"alive subcomponents: {len(gu_comps)} gate/up + {len(down_comps)} down "
        f"({len(alive_all) - len(alive)} dropped: max last-token CI < {last_ci_thr})"
    )

    inner_grids: dict[tuple[str, int], NDArray[np.float32]] = {}
    for proj, c in gu_comps:
        inner_grids[(proj, c)] = x @ uv[proj][0][:, c]
    for proj, c in down_comps:
        inner_grids[(proj, c)] = act @ uv[proj][0][:, c]

    # exact full-grid single-component deltas (gate/up; down deltas are exact in-browser)
    exact_single: dict[tuple[str, int], NDArray[np.float32]] = {}
    for proj, c in gu_comps:
        dact = _gu_dact(gate, up, act, proj, inner_grids[(proj, c)], uv[proj][1][c])
        exact_single[(proj, c)] = (dact @ wd).astype(np.float32)
        del dact
        logger.info(f"exact single delta {_PROJ_TAG[proj]}{c}")

    # --- circuit row: the MLP with only the alive subcomponents on, delta off ----------------
    # Each decomposed matrix is reconstructed from just the kept components (binary mask), so
    # preactivations are `(x·V) U` over the alive gate/up sets and the output goes through the
    # alive down set only. Chunked over prompts to bound the [chunk, d_int] intermediates.
    def stack_v(proj: str, comps: list[tuple[str, int]]) -> NDArray[np.float32]:
        return np.stack([uv[proj][0][:, c] for _, c in comps], axis=1)

    def stack_u(proj: str, comps: list[tuple[str, int]]) -> NDArray[np.float32]:
        return np.stack([uv[proj][1][c] for _, c in comps])

    gate_alive = [pc for pc in gu_comps if pc[0] == "gate_proj"]
    up_alive = [pc for pc in gu_comps if pc[0] == "up_proj"]
    assert gate_alive and up_alive and down_comps, "circuit row needs alive comps in every proj"
    vg, ug = stack_v("gate_proj", gate_alive), stack_u("gate_proj", gate_alive)
    vu, uu = stack_v("up_proj", up_alive), stack_u("up_proj", up_alive)
    vd = stack_v("down_proj", down_comps)
    ud_w = stack_u("down_proj", down_comps) @ w_post  # [Cd, 2P]
    y_circuit = np.empty((n_prompts, w_post.shape[1]), np.float32)
    for s in range(0, n_prompts, _CIRCUIT_CHUNK):
        sl = slice(s, min(s + _CIRCUIT_CHUNK, n_prompts))
        gate_m = (x[sl] @ vg) @ ug
        up_m = (x[sl] @ vu) @ uu
        y_circuit[sl] = (silu_combine(gate_m, up_m) @ vd) @ ud_w
    proj_post_on_post = resid_post @ w_post + b_post
    proj_circuit = resid_pre @ w_post + b_post + y_circuit
    dev = float(np.linalg.norm(proj_circuit - proj_post_on_post, axis=1).mean())
    logger.info(f"circuit row: mean |Δ| to the true post projection {dev:.3f}")

    # --- alignment grids: activation · unit direction, per item ------------------------------
    # Read side dots the activation the direction actually reads (MLP input `x` for gate/up V
    # and neuron gate/up rows; post-SwiGLU `h` for down V). Write side dots the space the
    # direction writes into (gate/up preactivations for gate/up U; the post residual for down U
    # and neuron down columns). Signed, colour-by options in the applet.
    def unit(v: NDArray[np.float32]) -> NDArray[np.float32]:
        return (v / np.linalg.norm(v)).astype(np.float32)

    align_in_sub = np.stack(
        [(x if p != "down_proj" else act) @ unit(uv[p][0][:, c]) for p, c in gu_comps + down_comps]
    )
    write_base = {"gate_proj": gate, "up_proj": up}
    align_out_sub = np.stack(
        [write_base[p] @ unit(uv[p][1][c]) for p, c in gu_comps]
        + [resid_post @ unit(uv[p][1][c]) for p, c in down_comps]
    )
    align_in_g = (gate[:, neuron_ids] / np.linalg.norm(weights["gate_proj"][neuron_ids], axis=1)).T
    align_in_u = (up[:, neuron_ids] / np.linalg.norm(weights["up_proj"][neuron_ids], axis=1)).T
    dcols = weights["down_proj"][:, neuron_ids]
    align_out_d = (resid_post @ (dcols / np.linalg.norm(dcols, axis=0))).T
    logger.info("alignment grids computed")

    # --- ablation-direction arrows -----------------------------------------------------------
    # Unit residual-space directions mapped through the same linear map as the points: read
    # directions (gate/up V columns, neuron gate/up rows — what the unit reads from the
    # residual) onto the pre frame; write directions (down U rows, neuron down columns — what
    # it writes) onto the post frame. RMSNorm sits between the pre residual and the actual
    # gate/up input; the probes absorb it linearly, so raw directions are used (as in
    # `build_direction_scatter`).
    def unit_rows(mat: NDArray[np.float32]) -> NDArray[np.float32]:
        return mat / np.linalg.norm(mat, axis=1, keepdims=True)

    sub_arrow_rows = [
        (uv[p][0][:, c] / np.linalg.norm(uv[p][0][:, c])) @ w_pre for p, c in gu_comps
    ] + [(uv[p][1][c] / np.linalg.norm(uv[p][1][c])) @ w_post for p, c in down_comps]
    neuron_arrow = {
        "g": unit_rows(weights["gate_proj"][neuron_ids]) @ w_pre,
        "u": unit_rows(weights["up_proj"][neuron_ids]) @ w_pre,
        "d": unit_rows(weights["down_proj"][:, neuron_ids].T) @ w_post,
    }

    # --- payload -----------------------------------------------------------------------------
    def comp_entry(proj: str, c: int) -> dict[str, Any]:
        group = period_groups.get((proj, c))
        return {
            "id": f"{_PROJ_TAG[proj]}{c}",
            "proj": _PROJ_TAG[proj],
            "ci": round(mean_ci[(proj, c)], 3),
            "period": group.label if group is not None else "?",
        }

    payload = {
        "meta": {
            "run": run_dir.name,
            "layer": layer,
            "n": n,
            "kl_thr": kl_thr,
            "last_ci_thr": last_ci_thr,
            "periods": period_list,
            "r2": {"post": r2_post, "pre": r2_pre},
            "mods": sorted({p for p in period_list if p > 1}),
        },
        # the four distinct clouds behind the five rows (row 4 = row 3 + ablation delta)
        "proj_pre_on_pre": b64_f16(resid_pre @ w_pre + b_pre),
        "proj_pre_on_post": b64_f16(resid_pre @ w_post + b_post),
        "proj_post_on_post": b64_f16(proj_post_on_post),
        "proj_circuit_on_post": b64_f16(proj_circuit),
        "neurons": [{"id": j, "kl": round(kl[j], 4)} for j in neuron_ids],
        "neuron_act": b64_f16(act[:, neuron_ids].T),
        "neuron_wd": b64_f16(wd[neuron_ids]),
        "subcomps": [comp_entry(p, c) for p, c in gu_comps + down_comps],
        "sub_arrow": b64_f16(np.stack(sub_arrow_rows)),
        "neuron_arrow_g": b64_f16(neuron_arrow["g"]),
        "neuron_arrow_u": b64_f16(neuron_arrow["u"]),
        "neuron_arrow_d": b64_f16(neuron_arrow["d"]),
        "sub_inner": b64_f16(np.stack([inner_grids[pc] for pc in gu_comps + down_comps])),
        "align_in_sub": b64_f16(align_in_sub),
        "align_out_sub": b64_f16(align_out_sub),
        "align_in_g": b64_f16(align_in_g),
        "align_in_u": b64_f16(align_in_u),
        "align_out_d": b64_f16(align_out_d),
        "gu_exact_delta": b64_f16(np.stack([exact_single[pc] for pc in gu_comps]))
        if gu_comps
        else "",
        "down_wu": b64_f16(np.stack([uv[p][1][c] @ w_post for p, c in down_comps]))
        if down_comps
        else "",
    }

    out_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else analysis_dir(run_dir) / "result_feature_construction"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.js").write_text(
        f"window.PD_DATA = {json.dumps(payload, separators=(',', ':'))};\n"
    )
    assert _APP_TEMPLATE.exists(), f"app template missing: {_APP_TEMPLATE}"
    shutil.copyfile(_APP_TEMPLATE, out_dir / "index.html")
    size_mb = (out_dir / "data.js").stat().st_size / 1e6
    logger.info(f"wrote result-feature-construction applet ({size_mb:.1f} MB) → {out_dir}")
    return out_dir


if __name__ == "__main__":
    fire.Fire(build_result_feature_construction)
