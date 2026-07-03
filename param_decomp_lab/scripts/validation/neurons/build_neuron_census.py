"""Build the neuron-census applet: an explorable HTML view of the L18 neuron data.

Aggregates everything the census pipeline produced — ablation screens / full grids,
activations, baselines, periodicity scores, subspace projections, candidates — into a
self-contained `index.html` + `data.js` (vanilla JS, canvas, opens from `file://`).

Views: a sortable/filterable candidate table (ablation stats, top periodicity lags, probe-
plane fractions); a lag-vs-KL scatter; and a per-neuron panel with the KL / Δcorrect-logprob
/ activation `(a, b)` heatmaps per op, the full (Δa, Δb) periodicity-score matrix per channel,
the answer-offset logprob profile, an error-mode explorer (what the ablated argmax becomes,
with residue-class conditioning on a and b), and an in-browser local windowed periodicity map.

Grids ship uint8-quantized (per-grid min/max) base64; only the top `--top-k` candidates by
overall max KL carry full per-neuron grids (payload control) — the table still lists every
candidate. Uses full-grid ablation npzs when present, else falls back to the 41×41 screens.

CPU-only. Smoke-test with `headless_check.py`.

Usage:
    python -m param_decomp_lab.scripts.validation.neurons.build_neuron_census \
        [--census-dir=PATH] [--top-k=200] [--model-path=PATH] [--output-dir=PATH]

`model_path` (any run over the base model) is only used to locate the tokenizer, which
decodes ablated answer tokens to numbers for the error-mode views.

Output: `<census_dir>/applet/{index.html,data.js}`.
"""

import base64
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import fire
import numpy as np
from transformers import AutoTokenizer

from param_decomp.log import logger
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.scripts.validation.neurons.common import (
    NEURON_OPS,
    NEURONS_DIR,
    OFFSETS,
    PERIODS,
    correct_answer_grid,
    silu_combine,
    token_value_map,
    translation_lags,
)

_APP_TEMPLATE = Path(__file__).parent / "neuron_census_app.html"
_VALUE_SENTINEL = -30000  # ablated argmax isn't a number (e.g. bare `-`)


def _q8(grid: np.ndarray) -> dict[str, Any]:
    """Quantize a float grid to uint8 + base64 with per-grid scale."""
    g = grid.astype(np.float32)
    lo, hi = float(np.nanmin(g)), float(np.nanmax(g))
    scale = (hi - lo) or 1.0
    q = np.clip(np.round((g - lo) / scale * 255.0), 0, 255).astype(np.uint8)
    return {"b64": base64.b64encode(q.tobytes()).decode(), "lo": lo, "hi": hi}


def _i16_b64(grid: np.ndarray) -> str:
    return base64.b64encode(grid.astype("<i2").tobytes()).decode()


def build_neuron_census(
    model_path: ModelPath,
    census_dir: str | None = None,
    top_k: int = 200,
    output_dir: str | None = None,
) -> Path:
    root = Path(census_dir).expanduser() if census_dir else NEURONS_DIR
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    from param_decomp_lab.experiments.lm.run import SavedLMRun

    tokenizer = AutoTokenizer.from_pretrained(
        SavedLMRun.from_path(model_path).cfg.data.tokenizer_name
    )
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    with open(root / "candidates.tsv") as f:
        cand_rows = list(csv.DictReader(f, delimiter="\t"))
    cand_ids = [int(r["neuron"]) for r in cand_rows]
    logger.info(f"{len(cand_ids)} candidates in table; shipping grids for top {top_k}")

    lags = translation_lags()
    periodicity = {op: np.load(root / f"periodicity_{op}.npz") for op in NEURON_OPS}
    subspace = np.load(root / "subspace.npz")

    abl: dict[str, Any] = {}
    for op in NEURON_OPS:
        full = sorted(root.glob(f"ablation_full_{op}*.npz"))
        if full:
            parts = [np.load(p) for p in full]
            abl[op] = {
                "kl": np.concatenate([p["kl"] for p in parts]),
                "abl_token": np.concatenate([p["abl_token"] for p in parts]),
                "delta_correct_logprob": np.concatenate(
                    [p["delta_correct_logprob"] for p in parts]
                ),
                "offset_logprob": np.concatenate([p["offset_logprob"] for p in parts])
                if "offset_logprob" in parts[0]
                else None,
                "clean_offset_logprob": parts[0].get("clean_offset_logprob"),
                "neuron_ids": np.concatenate([p["neuron_ids"] for p in parts]),
                "a": parts[0]["a"],
                "orig_token": parts[0]["orig_token"],
                "stride": int(parts[0]["stride"]),
            }
        else:
            p = np.load(root / f"ablation_screen_{op}.npz")
            abl[op] = {
                "kl": p["kl"],
                "abl_token": p["abl_token"],
                "delta_correct_logprob": p["delta_correct_logprob"],
                "offset_logprob": None,
                "clean_offset_logprob": None,
                "neuron_ids": p["neuron_ids"],
                "a": p["a"],
                "orig_token": p["orig_token"],
                "stride": int(p["stride"]),
            }
        abl[op]["row_of"] = {int(n): i for i, n in enumerate(abl[op]["neuron_ids"])}

    acts = {op: np.load(root / f"activations_{op}.npz") for op in NEURON_OPS}
    baseline = {op: np.load(root / f"baseline_{op}.npz") for op in NEURON_OPS}

    value_maps = {op: token_value_map(tokenizer, abl[op]["abl_token"]) for op in NEURON_OPS}

    chan_names = [str(c) for c in periodicity[NEURON_OPS[0]]["channels"]]
    neurons_meta: list[dict[str, Any]] = []
    for row in cand_rows:
        nid = int(row["neuron"])
        meta: dict[str, Any] = {"id": nid}
        for k, v in row.items():
            if k != "neuron":
                meta[k] = float(v) if "." in v or "e" in v.lower() else int(v)
        for op in NEURON_OPS:
            sc = periodicity[op]["score"][nid]  # [3, n_lags]
            meta[f"pscore_{op}"] = np.round(sc, 3).tolist()
        meta["read_frac"] = np.round(subspace["read_frac"][nid], 3).tolist()
        meta["write_frac"] = np.round(subspace["write_frac"][nid], 3).tolist()
        neurons_meta.append(meta)

    ship_ids = cand_ids[:top_k]
    grids: dict[str, dict[str, Any]] = {}
    for nid in ship_ids:
        entry: dict[str, Any] = {}
        for op in NEURON_OPS:
            a_row = abl[op]["row_of"].get(nid)
            if a_row is None:
                continue
            kl = abl[op]["kl"][a_row].astype(np.float32)
            dlp = abl[op]["delta_correct_logprob"][a_row].astype(np.float32)
            tok = abl[op]["abl_token"][a_row]
            true_ans = correct_answer_grid(op)[:: abl[op]["stride"], :: abl[op]["stride"]]
            vmap = value_maps[op]
            dval = np.full(tok.shape, _VALUE_SENTINEL, dtype=np.int16)
            for tid, val in vmap.items():
                sel = tok == tid
                dval[sel] = np.clip(val - true_ans[sel], _VALUE_SENTINEL + 1, 30000)
            gate = acts[op]["gate_preact"][:, :, nid]
            up = acts[op]["up_preact"][:, :, nid]
            comb = silu_combine(gate, up)
            op_entry: dict[str, Any] = {
                "stride": abl[op]["stride"],
                "kl": _q8(kl),
                "dlp": _q8(dlp),
                "dval": _i16_b64(dval),
                "gate": _q8(gate.astype(np.float32)),
                "up": _q8(up.astype(np.float32)),
                "comb": _q8(comb),
            }
            if abl[op]["offset_logprob"] is not None:
                off = abl[op]["offset_logprob"][a_row].astype(np.float32)  # [n,n,16]
                clean = abl[op]["clean_offset_logprob"].astype(np.float32)
                delta = off - clean
                flips = tok != abl[op]["orig_token"]
                op_entry["offset_mean"] = np.round(delta.mean(axis=(0, 1)), 3).tolist()
                op_entry["offset_mean_flip"] = (
                    np.round(delta[flips].mean(axis=0), 3).tolist() if flips.any() else None
                )
            entry[op] = op_entry
        grids[str(nid)] = entry

    data = {
        "ops": list(NEURON_OPS),
        "n_values": int(acts[NEURON_OPS[0]]["a"].shape[0]),
        "lags": lags.tolist(),
        "periods": list(PERIODS),
        "offsets": list(OFFSETS),
        "channels": chan_names,
        "variables": [str(v) for v in subspace["variables"]],
        "read_r2": np.round(subspace["read_r2"], 3).tolist(),
        "write_r2": np.round(subspace["write_r2"], 3).tolist(),
        "baseline": {
            op: {
                "is_correct": _q8(baseline[op]["is_correct"].astype(np.float32)),
                "correct_prob": _q8(baseline[op]["correct_prob"].astype(np.float32)),
                "accuracy": float(baseline[op]["is_correct"].mean()),
            }
            for op in NEURON_OPS
        },
        "null_kl_max": {
            op: float(np.load(root / f"ablation_screen_{op}.npz")["null_kl"].max())
            for op in NEURON_OPS
            if (root / f"ablation_screen_{op}.npz").exists()
        },
        "neurons": neurons_meta,
        "grids": grids,
    }

    out_root = Path(output_dir).expanduser() if output_dir else root / "applet"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "data.js").write_text("window.DATA = " + json.dumps(data) + ";")
    shutil.copy(_APP_TEMPLATE, out_root / "index.html")
    size_mb = (out_root / "data.js").stat().st_size / 1e6
    logger.info(f"wrote {out_root}/index.html + data.js ({size_mb:.0f} MB)")
    return out_root / "index.html"


if __name__ == "__main__":
    fire.Fire(build_neuron_census)
