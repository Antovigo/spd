"""Build the subcomponent-census applet for a decomposition run (addition grid).

The subcomponent counterpart of `build_neuron_census`, dropped in the run's
`analysis/subcomp_census/`. Three views:

1. **Component table + detail** — measured ablation KL stats per (matrix, component), the
   inner-activation grid, periodicity profile, error modes (ablated answer − truth with
   residue conditioning), answer-offset profile, and the component's top coupled neurons
   (by `std(inner_c)·|U[c, j]|` write strength / `|V[j, c]|·std(act_j)` read strength),
   each tagged with the neuron's own ablation KL and period chips.
2. **Connection matrix** — candidate neurons (rows, grouped by dominant combined-activation
   period) × subcomponents (cols, grouped by matrix then dominant inner-act period), cell =
   signed functional coupling. Multi-period neurons are flagged; period-matched blocks are
   the thing to look for.
3. **Explanation view** — per candidate neuron, measured neuron ablation KL vs the R² of its
   gate/up preactivation reconstructed from only the *measured-causal* components.
   Bottom-right = causally-important neuron the decomposition does not explain. Clicking
   reconstructs actual / explained / residual grids in-browser from the shipped inner grids
   and couplings.

Inputs: the run's `subcomp_ablation_screen_add.npz` (+ `subcomp_ablation_full_add.npz` when
present, preferred), `subcomp_neuron_links_add.npz`, and the census dir (neuron candidates,
periodicity, activations). CPU-only; smoke-test with `headless_check.py`.

Usage:
    python -m param_decomp_lab.scripts.validation.neurons.build_subcomp_census \
        <model_path> [--census-dir=PATH] [--output-dir=PATH]

Output: `<run>/analysis/subcomp_census/{index.html,data.js}`.
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
from param_decomp_lab.experiments.lm.run import SavedLMRun
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.scripts.validation.common import analysis_datasets_dir, analysis_dir
from param_decomp_lab.scripts.validation.neurons.common import (
    NEURONS_DIR,
    OFFSETS,
    PERIODS,
    correct_answer_grid,
    silu_combine,
    token_value_map,
    translation_lags,
)

_APP_TEMPLATE = Path(__file__).parent / "subcomp_census_app.html"
_VALUE_SENTINEL = -30000
MLP_PROJS = ("gate_proj", "up_proj", "down_proj")


def _q8(grid: np.ndarray, symmetric: bool = False) -> dict[str, Any]:
    g = grid.astype(np.float32)
    if symmetric:
        m = float(np.abs(g).max()) or 1.0
        lo, hi = -m, m
    else:
        lo, hi = float(g.min()), float(g.max())
    scale = (hi - lo) or 1.0
    q = np.clip(np.round((g - lo) / scale * 255.0), 0, 255).astype(np.uint8)
    return {"b64": base64.b64encode(q.tobytes()).decode(), "lo": lo, "hi": hi}


def _i16_b64(grid: np.ndarray) -> str:
    return base64.b64encode(grid.astype("<i2").tobytes()).decode()


def build_subcomp_census(
    model_path: ModelPath,
    census_dir: str | None = None,
    output_dir: str | None = None,
) -> Path:
    saved = SavedLMRun.from_path(model_path)
    run_dir = saved.checkpoint_path.parent
    datasets = analysis_datasets_dir(run_dir)
    census = Path(census_dir).expanduser() if census_dir else NEURONS_DIR
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    tokenizer = AutoTokenizer.from_pretrained(saved.cfg.data.tokenizer_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    links = np.load(datasets / "subcomp_neuron_links_add.npz")
    full_path = datasets / "subcomp_ablation_full_add.npz"
    abl = np.load(full_path if full_path.exists() else datasets / "subcomp_ablation_screen_add.npz")
    stride = int(abl["stride"])
    abl_matrix = [str(m) for m in abl["matrix"]]
    abl_comp = abl["component"]
    abl_row: dict[tuple[str, int], int] = {
        (m, int(c)): i for i, (m, c) in enumerate(zip(abl_matrix, abl_comp, strict=True))
    }

    cand = links["candidate_neurons"]
    lags = translation_lags()
    acts = np.load(census / "activations_add.npz")
    neuron_period = np.load(census / "periodicity_add.npz")["score"]  # [d_int, 3, n_lags]
    with open(census / "candidates.tsv") as f:
        cand_stats = {int(r["neuron"]): r for r in csv.DictReader(f, delimiter="\t")}

    vmap = token_value_map(tokenizer, abl["abl_token"])
    true_ans = correct_answer_grid("add")[::stride, ::stride]
    kl_all = abl["kl"].astype(np.float32)
    flip_all = abl["answer_flip"]
    dlp_all = abl["delta_correct_logprob"].astype(np.float32)

    comps_meta: list[dict[str, Any]] = []
    comp_grids: dict[str, Any] = {}
    for proj in MLP_PROJS:
        pscore = links[f"period_score_{proj}"]  # [C, n_lags]
        inner = links[f"inner_{proj}"]  # [C, N, N] f16
        inner_std = links[f"inner_std_{proj}"]
        causal = links[f"causal_{proj}"]
        max_kl_grid = links[f"max_kl_{proj}"]
        for c in range(inner.shape[0]):
            row = abl_row.get((proj, c))
            if row is None:
                continue
            kl = kl_all[row]
            meta = {
                "matrix": proj,
                "comp": c,
                "max_kl": round(float(kl.max()), 5),
                "mean_kl": round(float(kl.mean()), 6),
                "n_flip": int(flip_all[row].sum()),
                "min_dlp": round(float(dlp_all[row].min()), 4),
                "causal": bool(causal[c]),
                "screen_max_kl": round(float(max_kl_grid[c]), 5),
                "inner_std": round(float(inner_std[c]), 4),
                "pscore": np.round(pscore[c], 3).tolist(),
            }
            comps_meta.append(meta)
            tok = abl["abl_token"][row]
            dval = np.full(tok.shape, _VALUE_SENTINEL, dtype=np.int16)
            for tid, val in vmap.items():
                sel = tok == tid
                dval[sel] = np.clip(val - true_ans[sel], _VALUE_SENTINEL + 1, 30000)
            entry: dict[str, Any] = {
                "stride": stride,
                "kl": _q8(kl),
                "dlp": _q8(dlp_all[row]),
                "dval": _i16_b64(dval),
                "inner": _q8(inner[c].astype(np.float32), symmetric=True),
            }
            if "offset_logprob" in abl:
                delta = abl["offset_logprob"][row].astype(np.float32) - abl[
                    "clean_offset_logprob"
                ].astype(np.float32)
                flips = flip_all[row]
                entry["offset_mean"] = np.round(delta.mean(axis=(0, 1)), 3).tolist()
                entry["offset_mean_flip"] = (
                    np.round(delta[flips].mean(axis=0), 3).tolist() if flips.any() else None
                )
            comp_grids[f"{proj}:{c}"] = entry

    neurons_meta: list[dict[str, Any]] = []
    neuron_grids: dict[str, Any] = {}
    for i, nid in enumerate(cand.tolist()):
        stats = cand_stats.get(nid, {})
        gate = acts["gate_preact"][:, :, nid].astype(np.float32)
        up = acts["up_preact"][:, :, nid].astype(np.float32)
        neurons_meta.append(
            {
                "id": nid,
                "max_kl_add": float(stats.get("max_kl_add", 0)),
                "source": stats.get("source", ""),
                "pscore": np.round(neuron_period[nid, 2], 3).tolist(),  # combined channel
                "r2_all": np.round(links["r2_all"][i], 3).tolist(),
                "r2_causal": np.round(links["r2_causal"][i], 3).tolist(),
            }
        )
        neuron_grids[str(nid)] = {
            "gate": _q8(gate, symmetric=True),
            "up": _q8(up, symmetric=True),
            "comb": _q8(silu_combine(gate, up), symmetric=True),
        }

    data = {
        "n_values": int(acts["a"].shape[0]),
        "lags": lags.tolist(),
        "periods": list(PERIODS),
        "offsets": list(OFFSETS),
        "matrices": list(MLP_PROJS),
        "subcomp_kl_thr": float(links["subcomp_kl_thr"]),
        "comps": comps_meta,
        "comp_grids": comp_grids,
        "neurons": neurons_meta,
        "neuron_grids": neuron_grids,
        "coupling": {
            "u_gate": _q8(links["u_gate"].astype(np.float32), symmetric=True),
            "u_up": _q8(links["u_up"].astype(np.float32), symmetric=True),
            "v_down": _q8(links["v_down"].astype(np.float32).T, symmetric=True),
            "shape": [int(links["u_gate"].shape[0]), len(cand)],
        },
    }

    out_root = (
        Path(output_dir).expanduser() if output_dir else analysis_dir(run_dir) / "subcomp_census"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "data.js").write_text("window.DATA = " + json.dumps(data) + ";")
    shutil.copy(_APP_TEMPLATE, out_root / "index.html")
    size_mb = (out_root / "data.js").stat().st_size / 1e6
    logger.info(
        f"{len(comps_meta)} components, {len(cand)} neurons; "
        f"wrote {out_root}/index.html + data.js ({size_mb:.0f} MB)"
    )
    return out_root / "index.html"


if __name__ == "__main__":
    fire.Fire(build_subcomp_census)
