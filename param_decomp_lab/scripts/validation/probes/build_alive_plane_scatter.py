"""Applet: residual stream on the probe planes, for the real model and alive-only circuits.

Extends `final_plane_scatter` with extra data sources computed fresh alongside the original:
the model with only output-alive components active (delta off), and — on a `dual_hidden_ci`
checkpoint — with only hidden-alive components active too, the same "components must do the
work" masking `find_alive_subcomponents.py` uses for its sweep. On a non-dual checkpoint
there's no hidden-alive list, so only two sources are shipped (real model, output-alive-only);
on a dual checkpoint all three are — always captured over the *same* sampled `a<op>b=` grid
points in one run, so per-point comparisons across sources are exact. Running this on both a
baseline (non-dual) and a dual run of the same decomposition is how you see what adding the
hidden-CI net actually changed.

Two rows (`a @ layer`, `b @ layer`, the layer-of-interest's own probes) plus a result row
fixed to one probe layer at a time (`--probe-layers`, default `18`, the decomposed layer;
pass a comma-separated list — e.g. `18,20` — to prepare more than one, and the applet gets a dropdown to switch
between them, defaulting to L18 when present) — the `<result> @ layer` own-probe row of
`final_plane_scatter` is dropped. Every requested probe layer's projection is derived from
the *same* captured activations (the ridge-CV fit already has a probe per layer; adding more
probe layers costs extra CPU-side projection, not extra GPU forward passes). Color by `value
mod T`, Δ from the previous position (in-plane or full-resid), by
**distortion** — a point's plane-projected distance between the selected source and the
`original` source (disabled while viewing `original` itself, where it is trivially zero) — or
by the **causal importance** of one selected alive subcomponent (role + matrix + component
pickers; CI is a single value per prompt, read at the `=` token from the *original* forward,
so it colors every panel identically regardless of which source is displayed).

**Frame** picks what a panel's origin means. `plane` (default) projects onto the probe
plane's *orthonormal* basis with the probe bias dropped, so the origin is the true
activation-space zero and panel angles are in-plane angles; `pred` reproduces the probe's
predicted `(cos, sin)`, whose origin is the *mean* activation and whose axes are skewed by
`w_cos`/`w_sin` being neither orthogonal nor equal-norm. `plane` is the frame the arrows
mean something in — the target model is bias-free, so a component's read `x·V` vanishes on a
hyperplane through the true zero (RMSNorm rescales but never recentres), and an arrow drawn
from that zero makes the honest angle with each point. `pred` keeps the ring ~unit-radius
and is the better view for judging probe fit. Both come from one shipped projection: the
per-plane transform in `ops[op].xf` rebuilds `pred` client-side, since duplicating the
points would double an already ~100 MB `data.js`. Switching frames needs no regeneration.

**Component arrows** overlay each alive subcomponent's own direction on the plane, drawn from
the origin as the displacement it contributes (`dir·e1, dir·e2`, bias-free). A dropdown
picks the `U` vectors of the components that write to the stream or the `V` vectors of those
that read from it; directions are gauge-invariant (`V·‖U‖` / `U·‖V‖`, as in
`build_direction_scatter`) and reads absorb their RMSNorm gain, since the component sees the
normalised stream while the panels plot the raw one. Candidates are the **union** of the alive
lists. Each arrow is gated to the single stream position its matrix actually touches —
`_arrow_site` derives that from `(layer, matrix)`, so decompositions spanning many layers work
— and two sliders control a `|proj|` floor and a shared length multiplier. Hovering an
arrowhead names the subcomponent and gives its in-plane length and angle to the plane.

Every batch is built from prompts of identical token length (no padding / attention mask —
`ComponentModel`'s masked forward doesn't support either), so the sampled grid is bucketed by
each prompt's natural (non-zero-padded) token length before batching. This keeps tokenization
identical to what `collect_resid_stream.py` used to fit the probes being reused here, so
projecting fresh activations through their stored `w_cos`/`w_sin` weights stays valid.

Only the projected 2D (or 1D) points are ever written out — never the underlying `d_model`
activations — plus per-position full-resid-norm deltas (scalars), CI (base64 uint8, one byte
per shown point per selected component), and the arrows' own 2D projections (base64 float16),
so `data.js` stays commensurate with `final_plane_scatter`'s even with two or three sources
instead of one. Site gating keeps the arrow block small: a component only needs the planes its
own column can display, not one per stream position.

An 8B forward needs a GPU; pass `--slurm` to submit this invocation as a single-GPU SLURM job.

Usage:
    python -m param_decomp_lab.scripts.validation.probes.build_alive_plane_scatter \
        <model_path> <ridge_cv_probes_add.json> [<ridge_cv_probes_sub.json> ...] \
        [--probe-layers=18|18,20] [--n-show=2000] [--seed=0] [--alpha=0.05] \
        [--batch-size=256] [--output-dir=DIR] \
        [--slurm [--gpus=1 --slurm-time=2:00:00 --slurm-mem=...]]

Output (default `alive_plane_scatter/` under the run's `analysis/`): `index.html` + `data.js`.
Requires `find_alive_subcomponents` already run (reads `analysis/datasets/alive_subcomponents.tsv`,
plus `_hidden.tsv` when the checkpoint has a hidden CI net). Smoke-test with the parent
folder's `headless_check.py`.
"""

import json
import shutil
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import fire
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from param_decomp.component_model import ComponentModel
from param_decomp.log import logger
from param_decomp.masks import make_mask_infos
from param_decomp.torch_helpers import bf16_autocast
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME
from param_decomp_lab.scripts.validation.common import (
    AliveComponent,
    SlurmOptions,
    analysis_datasets_dir,
    analysis_dir,
    b64_f16,
    b64_u8,
    load_lm_run,
    op_symbol,
    probe_plane_basis,
    read_alive_components,
    submit_self_to_slurm,
)
from param_decomp_lab.scripts.validation.probes.fit_ridge_cv_probes import _variable_values

_MODULE = "param_decomp_lab.scripts.validation.probes.build_alive_plane_scatter"
_TEMPLATE = Path(__file__).with_name("alive_plane_scatter_app.html")
_OWN_VARIABLES = ("a", "b")
_RESULT_VARIABLE = {"add": "a+b", "sub": "a-b"}
# Periods the fitted grid has but this applet doesn't show.
_EXCLUDED_PERIODS = (25, 33)
Role = Literal["output", "hidden"]
ArrowRole = Literal["read", "write"]


def _layers_from_positions(positions: list[str]) -> list[int]:
    return sorted({int(p[1:].removesuffix("att")) for p in positions})


@contextmanager
def _capture_positions(hf_model: nn.Module, layers: list[int]) -> Generator[dict[str, Tensor]]:
    """Mirrors `collect_resid_stream.py`'s hooks: block output + pre-MLP-norm input, last token."""
    captured: dict[str, Tensor] = {}

    def make_block_hook(pos: str):  # noqa: ANN202
        def hook(_m: Any, _i: Any, out: Any) -> None:
            hidden = out[0] if isinstance(out, tuple) else out
            captured[pos] = hidden[:, -1].float()

        return hook

    def make_att_hook(pos: str):  # noqa: ANN202
        def hook(_m: Any, args: Any) -> None:
            captured[pos] = args[0][:, -1].float()

        return hook

    handles = [
        hf_model.get_submodule(f"model.layers.{i}").register_forward_hook(make_block_hook(f"L{i}"))
        for i in layers
    ] + [
        hf_model.get_submodule(
            f"model.layers.{i}.post_attention_layernorm"
        ).register_forward_pre_hook(make_att_hook(f"L{i}att"))
        for i in layers[1:]
    ]
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


def _flat(p: np.ndarray) -> list[float]:
    """Flattened, at ~6 significant digits.

    Decimals come from the array's own scale rather than a fixed count: activation-frame
    coords carry whatever scale the residual stream happens to have in that plane, unlike
    the probe's predictions, which were always ~unit.
    """
    flat = p[:, 0] if p.shape[1] == 1 else p.reshape(-1)
    peak = float(np.abs(flat).max())
    decimals = 6 if peak == 0.0 else int(np.clip(6 - np.floor(np.log10(peak)) - 1, 0, 12))
    return [round(float(c), decimals) for c in flat]


def _project_plane(x: NDArray[np.float32], cell: dict[str, Any]) -> NDArray[np.float32]:
    """`[n, 2]` activation on the probe plane's orthonormal basis — `[n, 1]` when degenerate.

    Linear, not affine: the probe bias is *not* applied, so the panel origin is the
    activation-space zero. That is the only origin the arrows can honestly start from —
    the target model is bias-free, so a subcomponent's read `x·V` vanishes on a hyperplane
    through the true zero, and RMSNorm (which rescales but never recentres) preserves it.
    An orthonormal basis additionally makes panel angles equal in-plane angles, which the
    raw `(w_cos, w_sin)` frame does not: those two are neither orthogonal nor equal-norm.
    """
    w_cos = np.asarray(cell["w_cos"], np.float32)
    w_sin = None if cell["w_sin"] is None else np.asarray(cell["w_sin"], np.float32)
    e1, e2 = probe_plane_basis(w_cos, w_sin)
    p1 = x @ e1
    return p1[:, None] if w_sin is None else np.stack([p1, x @ e2], axis=1)


def _frame_transform(cell: dict[str, Any]) -> dict[str, float]:
    """Coefficients taking orthonormal plane coords back to the probe's predicted (cos, sin).

    `pred_cos = k·p1 + bc`, `pred_sin = m1·p1 + m2·p2 + bs`. Exact, because `w_cos = k·e1`
    and `w_sin` lies in the plane by construction — so shipping one projection is enough
    for the applet to offer both frames, instead of doubling an already ~100 MB `data.js`.

    Kept at full precision, unlike the calibration scalars: these multiply every point, and
    `m1·p1` can cancel against `m2·p2` on a near-degenerate plane, so rounding them shows up
    in the reconstructed frame. There are only a handful per plane.
    """
    w_cos = np.asarray(cell["w_cos"], np.float32)
    w_sin = None if cell["w_sin"] is None else np.asarray(cell["w_sin"], np.float32)
    e1, e2 = probe_plane_basis(w_cos, w_sin)
    out = {"k": float(np.linalg.norm(w_cos)), "bc": cell["b_cos"]}
    if w_sin is not None:
        out |= {"m1": float(w_sin @ e1), "m2": float(w_sin @ e2), "bs": cell["b_sin"]}
    return out


def _comma_list(x: str | tuple[Any, ...]) -> list[str]:
    """Split a `--flag=a,b,c` value into parts.

    Fire parses an unquoted comma-separated CLI value into a tuple rather than a string
    (see the identical guard around `--ks` in `find_alive_subcomponents.py`).
    """
    return x.split(",") if isinstance(x, str) else [str(v) for v in x]


def _arrow_site(layer: int, proj: str) -> tuple[str, str | None, ArrowRole]:
    """The one captured stream position a decomposed matrix touches, its RMSNorm, and its role.

    Positions follow `_capture_positions`: `L{i}` is block `i`'s output, `L{i}att` the residual
    after block `i`'s attention (the pre-MLP-norm input). A read also names the RMSNorm whose
    gain it sees before the weight; a write applies none.
    """
    match proj:
        case "gate_proj" | "up_proj":
            return f"L{layer}att", f"model.layers.{layer}.post_attention_layernorm", "read"
        case "q_proj" | "k_proj" | "v_proj":
            return f"L{layer - 1}", f"model.layers.{layer}.input_layernorm", "read"
        case "down_proj":
            return f"L{layer}", None, "write"
        case "o_proj":
            return f"L{layer}att", None, "write"
        case _:
            raise AssertionError(f"no residual-stream site known for matrix {proj!r}")


@dataclass(frozen=True)
class ArrowCandidate:
    component: AliveComponent
    role: ArrowRole
    site: str
    norm_module: str | None
    alive_roles: tuple[Role, ...]


def _arrow_candidates(
    alive: dict[Role, list[AliveComponent]], stream_positions: list[str]
) -> list[ArrowCandidate]:
    """The union of the alive lists, each tagged with the stream position it can touch.

    A component whose site falls outside the captured positions has no panel to draw on and is
    dropped. `alive_roles` records which lists it came from, in `alive`'s order.
    """
    roles: dict[tuple[int, str, int], list[Role]] = {}
    components: dict[tuple[int, str, int], AliveComponent] = {}
    for role, acs in alive.items():
        for ac in acs:
            key = (ac.layer, ac.matrix, ac.component)
            components[key] = ac
            roles.setdefault(key, []).append(role)
    out: list[ArrowCandidate] = []
    for key in sorted(components):
        ac = components[key]
        site, norm_module, arrow_role = _arrow_site(ac.layer, ac.proj)
        if site not in stream_positions:
            continue
        out.append(ArrowCandidate(ac, arrow_role, site, norm_module, tuple(roles[key])))
    return out


def _arrow_directions(
    model: ComponentModel, candidates: list[ArrowCandidate], d_model: int
) -> NDArray[np.float32]:
    """`[n_candidates, d_model]` residual-space direction per candidate.

    Reads use `V[:, c]·‖U[c]‖`, writes `U[c]·‖V[:, c]‖` — `build_direction_scatter`'s product
    form, invariant to the rank-1 gauge `u → αu, v → v/α` and equal to the residual move per
    one std of the unit's activation. A read absorbs its RMSNorm gain (`γ ⊙ V`), since the
    component sees the normalised stream while the panels plot the raw one.
    """
    positions_by_module: dict[str, list[int]] = {}
    for i, candidate in enumerate(candidates):
        positions_by_module.setdefault(candidate.component.module, []).append(i)
    out = np.zeros((len(candidates), d_model), np.float32)
    for module, positions in positions_by_module.items():
        candidate = candidates[positions[0]]
        components = model.components[module]
        v, u = components.V.detach().float(), components.U.detach().float()
        cs = torch.tensor([candidates[i].component.component for i in positions], device=v.device)
        if candidate.role == "read":
            assert candidate.norm_module is not None
            norm_layer = model.target_model.get_submodule(candidate.norm_module)
            gain = norm_layer.get_parameter("weight").detach().float()
            dirs = v[:, cs].T * u[cs].norm(dim=1)[:, None] * gain
        else:
            dirs = u[cs] * v[:, cs].norm(dim=0)[:, None]
        assert dirs.shape == (len(positions), d_model), (
            f"{module} {candidate.role} directions are {tuple(dirs.shape)}, "
            f"expected [{len(positions)}, {d_model}]"
        )
        out[positions] = dirs.cpu().numpy()
    return out


def _plane_arrows(
    dirs: NDArray[np.float32], cell: dict[str, Any]
) -> tuple[dict[str, str], dict[str, NDArray[np.float32]]]:
    """Per-direction in-plane increment, its angle to the plane, and its norm in each frame.

    Shipped in the same orthonormal frame as the points, so an arrow's drawn length *is* its
    true in-plane norm and the angle it makes with a point *is* the in-plane angle between
    that direction and the activation. The probe bias never applies to an arrow whichever
    frame is displayed: it is a displacement, and a panel maps `x → x·w + b`, so adding
    `dir` to `x` moves the point by `dir·w` regardless of `b`. The second axis is all-zero
    on a degenerate (period-2) plane, matching how the panel draws it. The angle is 0° for a
    direction lying in the plane and 90° for one orthogonal to it, independent of scale.
    """
    w_cos = np.asarray(cell["w_cos"], np.float32)
    w_sin = None if cell["w_sin"] is None else np.asarray(cell["w_sin"], np.float32)
    e1, e2 = probe_plane_basis(w_cos, w_sin)
    a1, a2 = dirs @ e1, dirs @ e2
    in_plane = np.hypot(a1, a2)
    angle = np.degrees(
        np.arccos(np.clip(in_plane / np.maximum(np.linalg.norm(dirs, axis=1), 1e-12), 0.0, 1.0))
    )
    payload = {"cs": b64_f16(np.stack([a1, a2], axis=1)), "ang": b64_f16(angle)}
    xf = _frame_transform(cell)
    pred_sin = np.zeros_like(a1) if w_sin is None else xf["m1"] * a1 + xf["m2"] * a2
    return payload, {"plane": in_plane, "pred": np.hypot(xf["k"] * a1, pred_sin)}


def _sig(x: float) -> float:
    return float(f"{x:.4g}")


def build_alive_plane_scatter(
    model_path: ModelPath,
    *ridge_cv_jsons: str,
    probe_layers: str = "18",
    n_show: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    batch_size: int = 256,
    output_dir: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 1,
    slurm_time: str = "2:00:00",
    slurm_mem: str | None = None,
    dependency: str | None = None,
) -> Path | None:
    assert ridge_cv_jsons, "pass at least one ridge_cv_probes_<op>.json"
    json_paths = [Path(p).expanduser() for p in ridge_cv_jsons]
    if slurm:
        argv = [
            str(Path(model_path).expanduser()),
            *[str(p) for p in json_paths],
            f"--probe-layers={','.join(_comma_list(probe_layers))}",
            f"--n-show={n_show}",
            f"--seed={seed}",
            f"--alpha={alpha}",
            f"--batch-size={batch_size}",
        ]
        if output_dir is not None:
            argv.append(f"--output-dir={Path(output_dir).expanduser()}")
        opts = SlurmOptions(
            partition=partition,
            gpus=gpus,
            slurm_time=slurm_time,
            slurm_mem=slurm_mem,
            dependency=dependency,
        )
        submit_self_to_slurm(_MODULE, argv, opts, job_name="val-alive-plane")
        return None

    run = load_lm_run(model_path)
    model, cfg, device, tokenizer = run.model, run.cfg, run.device, run.tokenizer
    dual = model.ci_fn_hidden is not None

    data_dir = analysis_datasets_dir(run.run_dir)
    alive: dict[Role, list[AliveComponent]] = {
        "output": read_alive_components(data_dir / "alive_subcomponents.tsv"),
    }
    if dual:
        alive["hidden"] = read_alive_components(data_dir / "alive_subcomponents_hidden.tsv")
    sources: tuple[str, ...] = (
        ("original", "hidden_alive", "output_alive") if dual else ("original", "output_alive")
    )

    modules = sorted(model.components.keys())
    n_comp = {m: model.components[m].V.shape[1] for m in modules}
    weight_deltas = model.calc_weight_deltas()
    dtype = next(model.parameters()).dtype

    probe_layer_list = sorted({int(x) for x in _comma_list(probe_layers)})
    probe_keys = [f"L{pl}" for pl in probe_layer_list]

    # Load probe jsons; every op must share the same grid (positions/periods/max_value).
    shared_meta: dict[str, Any] | None = None
    op_payloads: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for json_path in json_paths:
        assert json_path.exists(), f"missing ridge-CV probes json: {json_path}"
        payload = json.loads(json_path.read_text())
        meta, results = payload["meta"], payload["results"]
        layer_keys = [str(p) for p in meta["positions"]]
        for pk in probe_keys:
            assert pk in layer_keys, f"probe layer {pk} not in {layer_keys}"
        grid = {"positions": layer_keys, "periods": meta["periods"], "max_value": meta["max_value"]}
        if shared_meta is None:
            shared_meta = grid
        else:
            assert shared_meta == grid, "jsons disagree on the grid"
        op_payloads.append((meta["op"], meta, results))
    assert shared_meta is not None
    layer_keys = cast(list[str], shared_meta["positions"])
    periods = [p for p in cast(list[int], shared_meta["periods"]) if p not in _EXCLUDED_PERIODS]
    max_value = cast(int, shared_meta["max_value"])
    layers = _layers_from_positions(layer_keys)

    alive_components_meta = {
        role: [{"matrix": ac.matrix, "component": ac.component, "layer": ac.layer} for ac in acs]
        for role, acs in alive.items()
    }

    arrow_candidates = _arrow_candidates(alive, layer_keys)
    # The probe weights define the space the arrows are projected into, so they set `d_model`.
    a_probe = op_payloads[0][2][layer_keys[0]][_OWN_VARIABLES[0]][str(periods[0])]
    d_model = len(cast(list[float], a_probe["w_cos"]))
    arrow_dirs = _arrow_directions(model, arrow_candidates, d_model)
    arrow_sites: dict[str, list[int]] = {}
    for i, candidate in enumerate(arrow_candidates):
        arrow_sites.setdefault(candidate.site, []).append(i)
    logger.info(
        f"arrows: {len(arrow_candidates)} alive subcomponents over "
        f"{ {site: len(idxs) for site, idxs in sorted(arrow_sites.items())} }"
    )
    arrow_norms: dict[str, list[NDArray[np.float32]]] = {"plane": [], "pred": []}

    rng = np.random.default_rng(seed)
    ops_data: dict[str, Any] = {}
    for op, _meta, results in op_payloads:
        sym = op_symbol(op)
        result_variable = _RESULT_VARIABLE[op]

        aa_full, bb_full = (
            g.reshape(-1)
            for g in np.meshgrid(
                np.arange(1, max_value + 1), np.arange(1, max_value + 1), indexing="ij"
            )
        )
        n_total = len(aa_full)
        show = rng.choice(n_total, size=min(n_show, n_total), replace=False)
        aa, bb = aa_full[show], bb_full[show]
        n_shown = len(show)
        prompts = [f"{a}{sym}{b}=" for a, b in zip(aa.tolist(), bb.tolist(), strict=True)]
        token_ids = [tokenizer.encode(p) for p in prompts]
        buckets: dict[int, list[int]] = {}
        for local_i, ids in enumerate(token_ids):
            buckets.setdefault(len(ids), []).append(local_i)

        captured_by_source: dict[str, dict[str, np.ndarray]] = {s: {} for s in sources}
        ci_raw: dict[Role, dict[tuple[str, int], np.ndarray]] = {
            role: {(ac.module, ac.component): np.zeros(n_shown, np.float32) for ac in acs}
            for role, acs in alive.items()
        }

        with torch.no_grad(), bf16_autocast(enabled=cfg.runtime.autocast_bf16):
            for length, local_idxs in buckets.items():
                for start in range(0, len(local_idxs), batch_size):
                    idxs = local_idxs[start : start + batch_size]
                    chunk = torch.tensor(
                        [token_ids[i] for i in idxs], dtype=torch.long, device=device
                    )
                    b = chunk.shape[0]

                    for source in sources:
                        if source == "original":
                            with _capture_positions(model.target_model, layers) as captured:
                                out = model(chunk, cache_type="input")
                            cache = out.cache
                            for role, acs in alive.items():
                                ci = model.calc_causal_importances(
                                    cache, sampling="continuous", role=role
                                )
                                for ac in acs:
                                    vals = (
                                        ci.lower_leaky[ac.module][:, -1, ac.component]
                                        .float()
                                        .cpu()
                                        .numpy()
                                    )
                                    ci_raw[role][(ac.module, ac.component)][idxs] = vals
                        else:
                            role: Role = "hidden" if source == "hidden_alive" else "output"
                            enabled = {
                                m: torch.zeros(n_comp[m], device=device, dtype=dtype)
                                for m in modules
                            }
                            for ac in alive[role]:
                                enabled[ac.module][ac.component] = 1.0
                            delta_mask = torch.zeros(b, length, device=device, dtype=dtype)
                            masks = {m: enabled[m].expand(b, length, n_comp[m]) for m in modules}
                            infos = make_mask_infos(
                                masks,
                                weight_deltas_and_masks={
                                    m: (weight_deltas[m], delta_mask) for m in modules
                                },
                            )
                            with _capture_positions(model.target_model, layers) as captured:
                                model(chunk, mask_infos=infos, cache_type="none")

                        for pos in layer_keys:
                            arr = captured[pos].cpu().numpy().astype(np.float32)  # [b, d_model]
                            store = captured_by_source[source]
                            if pos not in store:
                                store[pos] = np.zeros((n_shown, arr.shape[-1]), np.float32)
                            store[pos][idxs] = arr

        sources_out: dict[str, Any] = {}
        for source in sources:
            own: dict[str, Any] = {
                v: {
                    str(t): {
                        # `w_sin is None` (period-2 probes) is a property of the period, not
                        # the layer, so any valid layer key gives the same answer here.
                        "one_d": results[layer_keys[0]][v][str(t)]["w_sin"] is None,
                        "layers": {},
                        "prev": {},
                        "stats": {
                            lk: {
                                "cv_r2": results[lk][v][str(t)]["cv_r2"],
                                "p_value": results[lk][v][str(t)]["p_value"],
                                "accepted": bool(
                                    results[lk][v][str(t)]["p_value"] <= alpha
                                    and results[lk][v][str(t)]["cv_r2"] > 0
                                ),
                            }
                            for lk in layer_keys
                        },
                    }
                    for t in periods
                }
                for v in _OWN_VARIABLES
            }
            planes: dict[str, Any] = {
                str(pl): {
                    result_variable: {
                        str(t): {
                            "one_d": results[pk][result_variable][str(t)]["w_sin"] is None,
                            "stats": {
                                "cv_r2": results[pk][result_variable][str(t)]["cv_r2"],
                                "p_value": results[pk][result_variable][str(t)]["p_value"],
                                "accepted": bool(
                                    results[pk][result_variable][str(t)]["p_value"] <= alpha
                                    and results[pk][result_variable][str(t)]["cv_r2"] > 0
                                ),
                            },
                            "layers": {},
                        }
                        for t in periods
                    }
                }
                for pl, pk in zip(probe_layer_list, probe_keys, strict=True)
            }
            resid_delta: dict[str, list[float]] = {}
            prev_x: np.ndarray | None = None
            for lk in layer_keys:
                x = captured_by_source[source][lk]
                if prev_x is not None:
                    resid_delta[lk] = [
                        round(float(d), 2) for d in np.linalg.norm(x - prev_x, axis=1)
                    ]
                for pl, pk in zip(probe_layer_list, probe_keys, strict=True):
                    for t in periods:
                        pcell = results[pk][result_variable][str(t)]
                        planes[str(pl)][result_variable][str(t)]["layers"][lk] = _flat(
                            _project_plane(x, pcell)
                        )
                for v in _OWN_VARIABLES:
                    for t in periods:
                        ocell = results[lk][v][str(t)]
                        own[v][str(t)]["layers"][lk] = _flat(_project_plane(x, ocell))
                        if prev_x is not None:
                            own[v][str(t)]["prev"][lk] = _flat(_project_plane(prev_x, ocell))
                prev_x = x
            sources_out[source] = {"own": own, "planes": planes, "resid_delta": resid_delta}

        # One transform per displayed plane, shared by every source (it depends only on the
        # probe), letting the applet rebuild the predicted-(cos, sin) frame client-side. Own
        # rows read their own layer's probe, so those are keyed by layer too.
        xf_out: dict[str, dict[str, float]] = {}
        for lk in layer_keys:
            for v in _OWN_VARIABLES:
                for t in periods:
                    xf_out[f"{v}|{lk}|{t}"] = _frame_transform(results[lk][v][str(t)])
        for pl, pk in zip(probe_layer_list, probe_keys, strict=True):
            for t in periods:
                xf_out[f"res|{pl}|{t}"] = _frame_transform(results[pk][result_variable][str(t)])

        # Arrows are gated to the one stream position each component touches, so a site only
        # needs the planes its own column can display: that position's own `a`/`b` probes, and
        # the result probe of every prepared probe layer.
        arrows_out: dict[str, dict[str, dict[str, str]]] = {}
        for site, positions in arrow_sites.items():
            site_dirs = arrow_dirs[positions]
            site_planes: dict[str, dict[str, str]] = {}
            for v in _OWN_VARIABLES:
                for t in periods:
                    cell = results[site][v][str(t)]
                    site_planes[f"{v}|{t}"], norms = _plane_arrows(site_dirs, cell)
                    for frame, vals in norms.items():
                        arrow_norms[frame].append(vals)
            for pl, pk in zip(probe_layer_list, probe_keys, strict=True):
                for t in periods:
                    cell = results[pk][result_variable][str(t)]
                    site_planes[f"res|{pl}|{t}"], norms = _plane_arrows(site_dirs, cell)
                    for frame, vals in norms.items():
                        arrow_norms[frame].append(vals)
            arrows_out[site] = site_planes

        ci_out: dict[str, str] = {}
        for role, acs in alive.items():
            for ac in acs:
                vals = ci_raw[role][(ac.module, ac.component)]
                ci_out[f"{role}|{ac.matrix}|{ac.component}"] = b64_u8(
                    np.clip(vals, 0.0, 1.0) * 255.0
                )

        ab_flat: list[int] = []
        for i in range(n_shown):
            ab_flat += [int(aa[i]), int(bb[i])]
        ops_data[op] = {
            "ab": ab_flat,
            "result_variable": result_variable,
            "values": {
                v: [int(x) for x in _variable_values(aa, bb, v)]
                for v in (*_OWN_VARIABLES, result_variable)
            },
            "sources": sources_out,
            "ci": ci_out,
            "arrows": arrows_out,
            "xf": xf_out,
        }
        logger.info(
            f"{op}: captured {n_shown} prompts x {len(layer_keys)} positions x {len(sources)} sources"
        )

    out_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else analysis_dir(run.run_dir) / "alive_plane_scatter"
    )
    # Calibrated per frame, since an arrow's length is frame-dependent and the two differ by
    # orders of magnitude: in `pred` a probe maps a `d_model` activation to a ~unit cosine, so
    # `‖w‖` is tiny and raw projections are far shorter than the cloud. `mult_default` puts the
    # 99th-percentile arrow at one data unit as the slider's midpoint.
    arrow_calib = {
        frame: {
            "norm_hi": _sig(float(np.quantile(cat, 0.999))),
            "norm_default": _sig(float(np.quantile(cat, 0.9))),
            "mult_default": _sig(1.0 / max(float(np.quantile(cat, 0.99)), 1e-12)),
        }
        for frame, cat in ((f, np.concatenate(v)) for f, v in arrow_norms.items())
    }
    payload_js = {
        "meta": {
            "positions": layer_keys,
            "periods": periods,
            "probe_layers": probe_layer_list,
            "n_show": n_show,
            "max_value": max_value,
            "sources": list(sources),
            "alive_components": alive_components_meta,
            "arrows": {
                "components": [
                    {
                        "layer": c.component.layer,
                        "matrix": c.component.matrix,
                        "component": c.component.component,
                        "role": c.role,
                        "site": c.site,
                        "alive": "+".join(c.alive_roles),
                    }
                    for c in arrow_candidates
                ],
                "sites": arrow_sites,
                "calib": arrow_calib,
            },
        },
        "ops": ops_data,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.js").write_text("window.ALIVE_PLANE_DATA = " + json.dumps(payload_js) + ";")
    shutil.copy(_TEMPLATE, out_dir / "index.html")
    logger.info(
        f"wrote {out_dir}/index.html + data.js "
        f"({(out_dir / 'data.js').stat().st_size / 1e6:.1f} MB)"
    )
    return out_dir


if __name__ == "__main__":
    fire.Fire(build_alive_plane_scatter)
