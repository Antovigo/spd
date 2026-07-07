"""Before/after-MLP18 Fourier-plane scatter with in-browser neuron / subcomponent ablation.

Does MLP 18 build the result (`a+b`) circular features from scratch, or add to structure
already present in the residual stream? And which neurons / subcomponents of a decomposition
build that structure? The applet shows one plot per canonical period, projected on the Fourier
probes fit to the residual stream around the MLP — a **basis** dropdown switches between the
probes fit **after** the MLP (`probes_post.json`, site `post`) and **before** it
(`probes_pre.json`, site `pre`) — in the probes' predicted `(cos, sin)` frame. Three rows: the
residual **before** the MLP, **after** the MLP, and after the MLP **with a user-picked set of
neurons or subcomponents ablated**.

The ablated row is emulated inside the applet (no forward pass, arbitrary simultaneous sets):

- **neurons** — exact on the full grid: the MLP output is additive over neurons, so ablating a
  set removes `Σ_j act_j · (w · W_down[:, j])` from the post projection. Only census neurons
  whose measured ablation `max_kl` exceeds `--kl-thr` are selectable; their post-SwiGLU
  activation grids ship exactly (fp16).
- **down subcomponents** — exact on the full grid: removing rank-1 `U_c V_c^T` from `W_down`
  subtracts `(h · V_c) · (w · U_c)`, additive over components.
- **gate/up subcomponents** — ablating these perturbs every neuron's preactivation (the alive
  components' couplings are dense), so the applet re-evaluates the full 14336-neuron SwiGLU on
  a `--stride`-strided subgrid from shipped low-rank preactivation factors (rank `--rank` SVD,
  neuron columns weighted by their max probe-projection norm over the bases). Single-component
  ablations ship their **exact** full-grid deltas per basis; multi-component ablations use the
  emulator with a control-variate correction (`emu(S) − Σ_c emu({c}) + Σ_c exact({c})`), so
  only the *interaction* between components is approximated. Build-time fidelity numbers per
  basis ship in the payload and are displayed in the applet.

The neuron activations and SVD factors are basis-independent and shipped once; a basis only
adds its projected clouds, per-unit projected write vectors, and exact single deltas.

CPU-only (no forward pass; mmap for weights). The op is fixed to `add` — the probes' result
variable is `a+b`.

Usage:
    python -m param_decomp_lab.scripts.validation.build_result_feature_construction \
        <model_path> [--probes-dir=.../fourier_probes] [--bases=post,pre] [--census-dir=PATH] \
        [--kl-thr=0.01] [--periods=2,5,10,20,50,100] [--rank=64] [--stride=2] [--output-dir=PATH]

Reads: `probes_<basis>.json` per basis (default dir
`<PARAM_DECOMP_OUT_DIR>/runs/fourier_probes/`), the run's `hidden_activations_add.npz` /
`alive_filtered_add.tsv` / `subcomp_periods_add.tsv`, the census `candidates.tsv`
(+ `ablation_full_add.npz` when present, overriding the screen KL), and the checkpoint U/V +
frozen MLP weights (mmap).

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
_RESID_BASES = ("post", "pre")  # probe sites on the residual stream (block output / pre-MLP)
# columns with a tiny probe-projection norm can't influence the projected output; the weight
# floor keeps the SVD from spending rank on them without dividing by ~0 on the way back
_WEIGHT_FLOOR = 0.05


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
    sel: list[tuple[str, NDArray[np.float32], NDArray[np.float32]]],
) -> NDArray[np.float32]:
    """Exact post-SwiGLU activation delta `[N, d_int]` for ablating gate/up components together.

    `sel` rows are `(proj, inner [N], U_row [d_int])`; the SwiGLU is re-evaluated with each
    component's rank-1 term removed from its preactivation. Basis-independent — project onto a
    basis's `wd` to get that basis's output delta.
    """
    dg = np.zeros_like(gate)
    du = np.zeros_like(up)
    for proj, inner, u_row in sel:
        (dg if proj == "gate_proj" else du)[...] += np.outer(inner, u_row)
    return silu_combine(gate - dg, up - du) - act


def build_result_feature_construction(
    model_path: str,
    probes_dir: str | None = None,
    bases: str = "post,pre",
    census_dir: str | None = None,
    kl_thr: float = 0.01,
    periods: str = _DEFAULT_PERIODS,
    rank: int = 64,
    stride: int = 2,
    output_dir: str | None = None,
) -> Path:
    ck = Path(model_path).expanduser()
    assert ck.exists(), f"missing checkpoint: {ck}"
    run_dir = ck.parent
    datasets = analysis_datasets_dir(run_dir)
    period_list = [int(p) for p in str(periods).split(",")]
    base_list = [b.strip() for b in str(bases).split(",")]
    assert base_list and all(b in _RESID_BASES for b in base_list), (
        f"bases must be drawn from {_RESID_BASES}, got {base_list}"
    )

    pdir = (
        Path(probes_dir).expanduser()
        if probes_dir
        else PARAM_DECOMP_OUT_DIR / "runs" / "fourier_probes"
    )
    probe_payloads: dict[str, dict[str, Any]] = {}
    for b in base_list:
        path = pdir / f"probes_{b}.json"
        assert path.exists(), f"missing probes JSON: {path}"
        probe_payloads[b] = json.loads(path.read_text())
        assert probe_payloads[b]["site"] == b, (
            f"{path.name} carries site {probe_payloads[b]['site']!r}, expected {b!r}"
        )

    npz_path = datasets / "hidden_activations_add.npz"
    assert npz_path.exists(), f"missing hidden activations: {npz_path}"
    d = np.load(npz_path)
    assert str(d["op"]) == "add", f"hidden activations are for op {d['op']}, need add"
    layer = int(d["layer"])
    for b in base_list:
        assert layer == int(probe_payloads[b]["layer"]), (
            f"probe layer {probe_payloads[b]['layer']} ({b}) != run layer {layer}"
        )
    n = int(d["a"].shape[0])
    n_prompts = n * n
    resid_pre = d["resid_pre_mlp"].reshape(n_prompts, -1).astype(np.float32)
    mlp_out = d["mlp_output"].reshape(n_prompts, -1).astype(np.float32)
    x = d["mlp_input"].reshape(n_prompts, -1).astype(np.float32)
    gate = d["gate_preact"].reshape(n_prompts, -1).astype(np.float32)
    up = d["up_preact"].reshape(n_prompts, -1).astype(np.float32)
    d_int = gate.shape[1]
    resid_post = resid_pre + mlp_out
    ref_cloud = {"post": resid_post, "pre": resid_pre}

    weights = load_target_mlp_weights(ck, layer, MLP_MATRICES)
    uv = load_component_uv(ck, layer, MLP_MATRICES)
    act = silu_combine(gate, up)

    # per-basis probe frames + projected write vectors
    axes: dict[str, dict[str, Any]] = {}
    for b in base_list:
        w_ax, b_ax, r2s = _probe_axes(probe_payloads[b]["probes"]["a+b"], period_list, ref_cloud[b])
        wd = weights["down_proj"].T @ w_ax  # [d_int, 2P] — each neuron's write, probe-projected
        recon_err = float(np.abs(act @ wd - mlp_out @ w_ax).max() / (mlp_out @ w_ax).std())
        assert recon_err < 0.1, (
            f"neuron-sum vs mlp_output projection mismatch ({b}): {recon_err:.3f}"
        )
        axes[b] = {"w": w_ax, "bias": b_ax, "r2": r2s, "wd": wd}
        logger.info(
            f"probe planes [{b}]: "
            + ", ".join(f"T={t} r2={r:.2f}" for t, r in zip(period_list, r2s, strict=True))
        )

    # --- neurons: measured-causal set, exact activation grids -------------------------------
    kl = _neuron_kl(Path(census_dir).expanduser() if census_dir else NEURONS_DIR)
    neuron_ids = sorted((j for j, k in kl.items() if k > kl_thr), key=lambda j: -kl[j])
    assert neuron_ids, f"no census neuron clears max_kl > {kl_thr}"
    logger.info(f"{len(neuron_ids)} neurons with measured ablation max KL > {kl_thr}")

    # --- subcomponents: alive on add, split by side -----------------------------------------
    mean_ci = read_mean_ci(datasets / "alive_filtered_add.tsv")
    period_groups = read_subcomp_period_groups(datasets / "subcomp_periods_add.tsv")
    alive = sorted(
        ((proj, c) for (proj, c) in mean_ci if proj in MLP_MATRICES),
        key=lambda pc: (MLP_MATRICES.index(pc[0]), pc[1]),
    )
    gu_comps = [(p, c) for p, c in alive if p != "down_proj"]
    down_comps = [(p, c) for p, c in alive if p == "down_proj"]
    logger.info(f"alive subcomponents: {len(gu_comps)} gate/up + {len(down_comps)} down")

    inner_grids: dict[tuple[str, int], NDArray[np.float32]] = {}
    for proj, c in gu_comps:
        inner_grids[(proj, c)] = x @ uv[proj][0][:, c]
    for proj, c in down_comps:
        inner_grids[(proj, c)] = act @ uv[proj][0][:, c]

    # exact full-grid single-component deltas per basis (gate/up; down is exact in-browser).
    # The expensive SwiGLU re-evaluation is basis-independent; each basis is one projection.
    exact_single: dict[str, dict[tuple[str, int], NDArray[np.float32]]] = {b: {} for b in base_list}
    for proj, c in gu_comps:
        dact = _gu_dact(gate, up, act, [(proj, inner_grids[(proj, c)], uv[proj][1][c])])
        for b in base_list:
            exact_single[b][(proj, c)] = (dact @ axes[b]["wd"]).astype(np.float32)
        del dact
        logger.info(f"exact single delta {_PROJ_TAG[proj]}{c}")

    # --- emulator: low-rank preactivation factors on the strided subgrid --------------------
    # One shared factor set: the SVD column weight is the max projection norm over the bases.
    ii = np.arange(0, n, stride)
    sub_idx = (ii[:, None] * n + ii[None, :]).ravel()
    m = len(sub_idx)
    wcol = np.max(
        np.stack([np.linalg.norm(axes[b]["wd"], axis=1) for b in base_list]), axis=0
    ).astype(np.float32)
    wcol = np.maximum(wcol / wcol.mean(), _WEIGHT_FLOOR)
    factors: dict[str, NDArray[np.float32]] = {}
    recon: dict[str, NDArray[np.float32]] = {}
    for name, pre in (("g", gate), ("u", up)):
        gsub = pre[sub_idx]
        mean = gsub.mean(axis=0)
        u_svd, s_svd, vt_svd = np.linalg.svd((gsub - mean) * wcol, full_matrices=False)
        z = (u_svd[:, :rank] * s_svd[:rank]).astype(np.float32)
        coef = (vt_svd[:rank] / wcol).astype(np.float32)
        # coefficients ship neuron-major ([d_int, rank]) — the browser reconstructs per neuron
        factors[f"z{name}"], factors[f"c{name}"], factors[f"m{name}"] = z, coef.T.copy(), mean
        recon[name] = mean + z @ coef
        logger.info(f"{name}: rank-{rank} SVD on the {m}-prompt subgrid")

    # fidelity: exact vs control-variate emulation for representative multi-component sets
    gate_s, up_s, act_s = gate[sub_idx], up[sub_idx], act[sub_idx]
    act_r = silu_combine(recon["g"], recon["u"])
    emu_single_dact = {
        (proj, c): _gu_dact(
            recon["g"],
            recon["u"],
            act_r,
            [(proj, inner_grids[(proj, c)][sub_idx], uv[proj][1][c])],
        )
        for proj, c in gu_comps
    }
    fidelity: dict[str, list[dict[str, Any]]] = {b: [] for b in base_list}
    checks = [("all gate+up", gu_comps)]
    if len(gu_comps) >= 4:
        checks.append(("4 components", gu_comps[:4]))
    for label, sel in checks:
        rows = [(p, inner_grids[(p, c)][sub_idx], uv[p][1][c]) for p, c in sel]
        dact_ex = _gu_dact(gate_s, up_s, act_s, rows)
        dact_em = _gu_dact(recon["g"], recon["u"], act_r, rows)
        for b in base_list:
            wd = axes[b]["wd"]
            dex = dact_ex @ wd
            cv = (
                dact_em @ wd
                - sum(emu_single_dact[pc] @ wd for pc in sel)
                + sum(exact_single[b][pc][sub_idx] for pc in sel)
            )
            mag = float(np.linalg.norm(dex, axis=1).mean())
            rel = float(np.linalg.norm(cv - dex, axis=1).mean()) / max(mag, 1e-9)
            fidelity[b].append({"config": label, "n_comps": len(sel), "rel_rms": round(rel, 4)})
            logger.info(f"emulator fidelity [{b}][{label}]: rel-rms {rel:.4f} (|delta| {mag:.3f})")
        del dact_ex, dact_em

    # --- payload -----------------------------------------------------------------------------
    def comp_entry(proj: str, c: int) -> dict[str, Any]:
        group = period_groups.get((proj, c))
        return {
            "id": f"{_PROJ_TAG[proj]}{c}",
            "proj": _PROJ_TAG[proj],
            "ci": round(mean_ci[(proj, c)], 3),
            "period": group.label if group is not None else "?",
        }

    basis_payloads: dict[str, dict[str, Any]] = {}
    for b in base_list:
        w_ax, b_ax, wd = axes[b]["w"], axes[b]["bias"], axes[b]["wd"]
        basis_payloads[b] = {
            "proj_pre": b64_f16(resid_pre @ w_ax + b_ax),
            "proj_post": b64_f16(resid_post @ w_ax + b_ax),
            "neuron_wd": b64_f16(wd[neuron_ids]),
            "wd_all": b64_f16(wd),
            "gu_exact_delta": b64_f16(np.stack([exact_single[b][pc] for pc in gu_comps]))
            if gu_comps
            else "",
            "down_wu": b64_f16(np.stack([uv[p][1][c] @ w_ax for p, c in down_comps]))
            if down_comps
            else "",
        }

    payload = {
        "meta": {
            "run": run_dir.name,
            "layer": layer,
            "n": n,
            "stride": stride,
            "rank": rank,
            "kl_thr": kl_thr,
            "periods": period_list,
            "bases": base_list,
            "r2": {b: axes[b]["r2"] for b in base_list},
            "mods": sorted({p for p in period_list if p > 1}),
            "fidelity": fidelity,
        },
        "bases": basis_payloads,
        "neurons": [{"id": j, "kl": round(kl[j], 4)} for j in neuron_ids],
        "neuron_act": b64_f16(act[:, neuron_ids].T),
        "subcomps": [comp_entry(p, c) for p, c in gu_comps + down_comps],
        "sub_inner": b64_f16(np.stack([inner_grids[pc] for pc in gu_comps + down_comps])),
        "gu_coupling": b64_f16(np.stack([uv[p][1][c] for p, c in gu_comps])) if gu_comps else "",
        "down_vcoupling": b64_f16(np.stack([uv[p][0][:, c] for p, c in down_comps]))
        if down_comps
        else "",
        "emul": {k: b64_f16(v) for k, v in factors.items()},
    }
    assert d_int == axes[base_list[0]]["wd"].shape[0]

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
