"""Export the ridge-CV probes in Feucht's `probes_<site>.json` format, one file per layer.

Pure format transform of `ridge_cv_probes_<op>.json` (from `fit_ridge_cv_probes`): the
shipped weights are that pipeline's full-range refit at the CV-selected λ — nothing is
refit here. Feucht's per-column `r2_cos` / `r2_sin` don't exist for the CV protocol, so
both carry the combined `cv_r2` (fold-mean held-out R² over rotating value blocks);
`cv_r2`, `p_value`, `lambda_rel`, and `accepted` (p ≤ alpha and cv_r2 > 0) ride along
per probe. Period-2 sin follows Feucht: `w_sin` zeros, `r2_sin` null.

Usage:
    python -m param_decomp_lab.scripts.validation.probes.export_ridge_cv_planes \
        <ridge_cv_probes_<op>.json> [--alpha=0.05] [--output-dir=DIR]

Output (default `ridge_cv_planes_<op>/` beside the json): `probes_L<i>.json` per layer,
`site` = `L<i>`.
"""

import json
from pathlib import Path
from typing import Any

import fire

from param_decomp.log import logger

_MODULE = "param_decomp_lab.scripts.validation.probes.export_ridge_cv_planes"


def export_ridge_cv_planes(
    ridge_cv_json: str,
    alpha: float = 0.05,
    output_dir: str | None = None,
) -> list[Path]:
    json_path = Path(ridge_cv_json).expanduser()
    assert json_path.exists(), f"missing ridge-CV probes json: {json_path}"
    payload = json.loads(json_path.read_text())
    meta, results = payload["meta"], payload["results"]
    op = meta["op"]
    out_dir = (
        Path(output_dir).expanduser() if output_dir else json_path.parent / f"ridge_cv_planes_{op}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    d_model = len(
        next(iter(next(iter(results.values())).values()))[str(meta["periods"][0])]["w_cos"]
    )
    out_paths: list[Path] = []
    for layer in meta["layers"]:
        probes: dict[str, dict[str, Any]] = {}
        for variable in meta["variables"]:
            probes[variable] = {}
            for period in meta["periods"]:
                cell = results[f"L{layer}"][variable][str(period)]
                sin_ok = cell["w_sin"] is not None
                probes[variable][str(period)] = {
                    "w_cos": cell["w_cos"],
                    "b_cos": cell["b_cos"],
                    "r2_cos": cell["cv_r2"],
                    "w_sin": cell["w_sin"] if sin_ok else [0.0] * d_model,
                    "b_sin": cell["b_sin"] if sin_ok else 0.0,
                    "r2_sin": cell["cv_r2"] if sin_ok else None,
                    "cv_r2": cell["cv_r2"],
                    "p_value": cell["p_value"],
                    "lambda_rel": cell["lambda_rel"],
                    "accepted": bool(cell["p_value"] <= alpha and cell["cv_r2"] > 0),
                }
        out = out_dir / f"probes_L{layer}.json"
        out.write_text(
            json.dumps(
                {
                    "model": "meta-llama/Llama-3.1-8B",
                    "layer": layer,
                    "max_value": meta["max_value"],
                    "site": f"L{layer}",
                    "op": op,
                    "variables": meta["variables"],
                    "periods_by_variable": {v: meta["periods"] for v in meta["variables"]},
                    "max_period": max(meta["periods"]),
                    "method": "closed-form ridge, lambda by rotating-value-block CV, "
                    "full-range refit; r2_cos = r2_sin = cv_r2 (fold-mean held-out R2); "
                    f"gated by {meta['n_perm']}-perm null at alpha={alpha}",
                    "source": str(json_path),
                    "probes": probes,
                }
            )
        )
        out_paths.append(out)
    logger.info(f"exported {len(out_paths)} layer files → {out_dir}")
    return out_paths


if __name__ == "__main__":
    fire.Fire(export_ridge_cv_planes)
