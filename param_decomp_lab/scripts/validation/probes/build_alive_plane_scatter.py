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
fixed to one probe layer at a time (`--probe-layers`, default `20`; pass a comma-separated
list — e.g. `18,19,20` — to prepare more than one, and the applet gets a dropdown to switch
between them, defaulting to L20 when present) — the `<result> @ layer` own-probe row of
`final_plane_scatter` is dropped. Every requested probe layer's projection is derived from
the *same* captured activations (the ridge-CV fit already has a probe per layer; adding more
probe layers costs extra CPU-side projection, not extra GPU forward passes). Color by `value
mod T`, Δ from the previous position (in-plane or full-resid), by
**distortion** — a point's plane-projected distance between the selected source and the
`original` source (disabled while viewing `original` itself, where it is trivially zero) — or
by the **causal importance** of one selected alive subcomponent (role + matrix + component
pickers; CI is a single value per prompt, read at the `=` token from the *original* forward,
so it colors every panel identically regardless of which source is displayed).

Every batch is built from prompts of identical token length (no padding / attention mask —
`ComponentModel`'s masked forward doesn't support either), so the sampled grid is bucketed by
each prompt's natural (non-zero-padded) token length before batching. This keeps tokenization
identical to what `collect_resid_stream.py` used to fit the probes being reused here, so
projecting fresh activations through their stored `w_cos`/`w_sin` weights stays valid.

Only the projected 2D (or 1D) points are ever written out — never the underlying `d_model`
activations — plus per-position full-resid-norm deltas (scalars) and CI (base64 uint8, one
byte per shown point per selected component), so `data.js` stays commensurate with
`final_plane_scatter`'s even with two or three sources instead of one.

An 8B forward needs a GPU; pass `--slurm` to submit this invocation as a single-GPU SLURM job.

Usage:
    python -m param_decomp_lab.scripts.validation.probes.build_alive_plane_scatter \
        <model_path> <ridge_cv_probes_add.json> [<ridge_cv_probes_sub.json> ...] \
        [--probe-layers=20|18,19,20] [--n-show=2000] [--seed=0] [--alpha=0.05] \
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
from pathlib import Path
from typing import Any, Literal, cast

import fire
import numpy as np
import torch
from torch import Tensor, nn

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
    b64_u8,
    load_lm_run,
    op_symbol,
    read_alive_components,
    submit_self_to_slurm,
)
from param_decomp_lab.scripts.validation.probes.fit_ridge_cv_probes import _variable_values
from param_decomp_lab.scripts.validation.probes.plot_probe_projections import _project

_MODULE = "param_decomp_lab.scripts.validation.probes.build_alive_plane_scatter"
_TEMPLATE = Path(__file__).with_name("alive_plane_scatter_app.html")
_OWN_VARIABLES = ("a", "b")
_RESULT_VARIABLE = {"add": "a+b", "sub": "a-b"}
Role = Literal["output", "hidden"]


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
    return [round(float(c), 4) for c in (p[:, 0] if p.shape[1] == 1 else p.reshape(-1))]


def build_alive_plane_scatter(
    model_path: ModelPath,
    *ridge_cv_jsons: str,
    probe_layers: str = "20",
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
            f"--probe-layers={probe_layers}",
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

    probe_layer_list = sorted({int(x) for x in probe_layers.split(",")})
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
    periods = cast(list[int], shared_meta["periods"])
    max_value = cast(int, shared_meta["max_value"])
    layers = _layers_from_positions(layer_keys)

    alive_components_meta = {
        role: [{"matrix": ac.matrix, "component": ac.component, "layer": ac.layer} for ac in acs]
        for role, acs in alive.items()
    }

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
                            _project(x, pcell)
                        )
                for v in _OWN_VARIABLES:
                    for t in periods:
                        ocell = results[lk][v][str(t)]
                        own[v][str(t)]["layers"][lk] = _flat(_project(x, ocell))
                        if prev_x is not None:
                            own[v][str(t)]["prev"][lk] = _flat(_project(prev_x, ocell))
                prev_x = x
            sources_out[source] = {"own": own, "planes": planes, "resid_delta": resid_delta}

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
        }
        logger.info(
            f"{op}: captured {n_shown} prompts x {len(layer_keys)} positions x {len(sources)} sources"
        )

    out_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else analysis_dir(run.run_dir) / "alive_plane_scatter"
    )
    payload_js = {
        "meta": {
            "positions": layer_keys,
            "periods": periods,
            "probe_layers": probe_layer_list,
            "n_show": n_show,
            "max_value": max_value,
            "sources": list(sources),
            "alive_components": alive_components_meta,
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
