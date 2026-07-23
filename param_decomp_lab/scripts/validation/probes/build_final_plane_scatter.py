"""Applet: every residual-stream position projected onto the final (L20) probe planes.

Fixes the probe to `--probe-layer`'s full-range-refit plane for each (variable, period)
and projects the stream at *every* collected position onto it. The applet (vanilla-JS
canvas, `file://`, no CDN) shows all positions **side by side** so the construction of
the final state reads left to right; color by `value mod T` or by each point's Δ from
the previous position — in-plane (computed client-side from consecutive projections) or
full-residual norm (`resid_delta`, precomputed here) — on one shared scale across
panels, all on viridis; plus displacement arrows and hover tooltips.

Usage:
    python -m param_decomp_lab.scripts.validation.probes.build_final_plane_scatter \
        <ridge_cv_probes_add.json> [<ridge_cv_probes_sub.json> ...] [--probe-layer=20] \
        [--n-show=2000] [--seed=0] [--alpha=0.05] [--output-dir=DIR] [--slurm [--gpus=0 ...]]

Output (default `final_plane_scatter/` beside the first json): `index.html` + `data.js`.
Each json's npz comes from its `meta.source`. Smoke-test with the parent folder's
`headless_check.py`.
"""

import json
import shutil
from pathlib import Path
from typing import Any

import fire
import numpy as np

from param_decomp.log import logger
from param_decomp_lab.infra.settings import DEFAULT_PARTITION_NAME
from param_decomp_lab.scripts.validation.common import SlurmOptions, submit_self_to_slurm
from param_decomp_lab.scripts.validation.probes.fit_ridge_cv_probes import _variable_values
from param_decomp_lab.scripts.validation.probes.plot_probe_projections import _project

_MODULE = "param_decomp_lab.scripts.validation.probes.build_final_plane_scatter"
_TEMPLATE = Path(__file__).parent / "final_plane_scatter_app.html"


def build_final_plane_scatter(
    *ridge_cv_jsons: str,
    probe_layer: int = 20,
    n_show: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    output_dir: str | None = None,
    slurm: bool = False,
    partition: str | None = DEFAULT_PARTITION_NAME,
    gpus: int = 0,
    slurm_time: str = "1:00:00",
    slurm_mem: str | None = "64G",
    dependency: str | None = None,
) -> Path | None:
    assert ridge_cv_jsons, "pass at least one ridge_cv_probes_<op>.json"
    json_paths = [Path(p).expanduser() for p in ridge_cv_jsons]
    if slurm:
        argv = [str(p) for p in json_paths] + [
            f"--probe-layer={probe_layer}",
            f"--n-show={n_show}",
            f"--seed={seed}",
            f"--alpha={alpha}",
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
        submit_self_to_slurm(_MODULE, argv, opts, job_name="val-final-plane")
        return None

    out_dir = (
        Path(output_dir).expanduser()
        if output_dir
        else json_paths[0].parent / "final_plane_scatter"
    )
    rng = np.random.default_rng(seed)
    shared_meta: dict[str, Any] | None = None
    ops_data: dict[str, Any] = {}
    for json_path in json_paths:
        assert json_path.exists(), f"missing ridge-CV probes json: {json_path}"
        payload = json.loads(json_path.read_text())
        meta, results = payload["meta"], payload["results"]
        op = meta["op"]
        layer_keys = [f"L{i}" for i in meta["layers"]]
        probe_key = f"L{probe_layer}"
        assert probe_key in layer_keys, f"probe layer {probe_key} not in {layer_keys}"
        if shared_meta is None:
            shared_meta = {
                "layers": layer_keys,
                "variables": meta["variables"],
                "periods": meta["periods"],
            }
        else:
            assert shared_meta == {
                "layers": layer_keys,
                "variables": meta["variables"],
                "periods": meta["periods"],
            }, "jsons disagree on the grid"

        npz_path = Path(meta["source"])
        assert npz_path.exists(), f"missing resid stream npz: {npz_path}"
        data = np.load(npz_path)
        a_axis, b_axis = data["a"].astype(np.int64), data["b"].astype(np.int64)
        aa, bb = (g.reshape(-1) for g in np.meshgrid(a_axis, b_axis, indexing="ij"))
        n = len(aa)
        show = rng.choice(n, size=min(n_show, n), replace=False)

        planes: dict[str, dict[str, Any]] = {
            v: {
                str(t): {
                    "one_d": results[probe_key][v][str(t)]["w_sin"] is None,
                    "l20": {
                        "cv_r2": results[probe_key][v][str(t)]["cv_r2"],
                        "p_value": results[probe_key][v][str(t)]["p_value"],
                        "accepted": bool(
                            results[probe_key][v][str(t)]["p_value"] <= alpha
                            and results[probe_key][v][str(t)]["cv_r2"] > 0
                        ),
                    },
                    "layers": {},
                }
                for t in meta["periods"]
            }
            for v in meta["variables"]
        }
        resid_delta: dict[str, list[float]] = {}
        prev: np.ndarray | None = None
        for layer_key in layer_keys:
            x = np.asarray(data[f"resid_{layer_key}"], dtype=np.float32).reshape(n, -1)[show]
            if prev is not None:
                delta = np.linalg.norm(x - prev, axis=1)
                resid_delta[layer_key] = [round(float(d), 2) for d in delta]
            for variable in meta["variables"]:
                for period in meta["periods"]:
                    cell = results[probe_key][variable][str(period)]
                    p = _project(x, cell)
                    flat = p[:, 0] if p.shape[1] == 1 else p.reshape(-1)
                    planes[variable][str(period)]["layers"][layer_key] = [
                        round(float(c), 4) for c in flat
                    ]
            prev = x
        ab_flat: list[int] = []
        for i in show:
            ab_flat += [int(aa[i]), int(bb[i])]
        ops_data[op] = {
            "ab": ab_flat,
            "values": {
                v: [int(x) for x in _variable_values(aa, bb, v)[show]] for v in meta["variables"]
            },
            "planes": planes,
            "resid_delta": resid_delta,
        }
        logger.info(f"{op}: projected {len(show)} prompts x {len(layer_keys)} positions")

    assert shared_meta is not None
    payload_js = {
        "meta": {
            **shared_meta,
            "probe_layer": probe_layer,
            "n_show": n_show,
            "alpha": alpha,
        },
        "ops": ops_data,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.js").write_text("window.FINAL_PLANE_DATA = " + json.dumps(payload_js) + ";")
    shutil.copy(_TEMPLATE, out_dir / "index.html")
    logger.info(
        f"wrote {out_dir}/index.html + data.js "
        f"({(out_dir / 'data.js').stat().st_size / 1e6:.1f} MB)"
    )
    return out_dir


if __name__ == "__main__":
    fire.Fire(build_final_plane_scatter)
