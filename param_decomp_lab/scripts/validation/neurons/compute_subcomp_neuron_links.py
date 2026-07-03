"""Subcomponent ↔ neuron coupling, subcomponent periodicity, and how much of each causal
neuron's activation the decomposition explains — all on the 0..200 addition grid, CPU-only.

Reads the census `activations_add.npz` (`mlp_input`, `gate_preact`, `up_preact`) plus the
run's checkpoint (V/U per L18 MLP matrix, mmap) and produces, in one pass:

1. **Subcomponent inner-activation grids** `[C, 201, 201]` per matrix — `x · V_c` with x the
   MLP input (gate/up) or the post-SwiGLU neuron acts (down) — and their translation-
   invariance periodicity scores (same lag set as the neurons).
2. **Coupling data**: for the census candidate neurons, the raw coupling weights
   `U[c, j]` (gate/up: what component c writes into neuron j's preactivation) and
   `V[j, c]` (down: how strongly component c reads neuron j), plus each component's
   inner-activation std over the grid — `std(inner_c)·|U[c, j]|` is the functional
   interaction strength, computable downstream from these pieces.
3. **Explanation R²** per candidate neuron and channel (gate, up): variance of the neuron's
   preactivation grid explained by (a) the sum of ALL components (the residual is the run's
   weight delta) and (b) only the *causally relevant* components — those whose measured
   last-position ablation KL from `subcomp_ablation_screen_add.npz` exceeds
   `--subcomp-kl-thr` (measured, not the learned CI). A causally-important neuron with low
   (b) is a neuron the decomposition's causal components do NOT explain.

Usage:
    python -m param_decomp_lab.scripts.validation.neurons.compute_subcomp_neuron_links \
        <model_path> [--acts-npz=PATH] [--candidates-tsv=PATH] [--subcomp-screen-npz=PATH] \
        [--subcomp-kl-thr=0.01] [--layer=18] [--output=PATH]

Output (default `subcomp_neuron_links_add.npz` in the run's `analysis/datasets/`): per matrix
`inner_<proj>` fp16 `[C, 201, 201]`, `inner_std_<proj>`, `period_score_<proj>` `[C, n_lags]`,
`causal_<proj>` bool (screen KL > thr anywhere) + `max_kl_<proj>`; coupling `u_gate` / `u_up`
`[C, n_cand]` and `v_down` `[n_cand, C]`; `r2_all` / `r2_causal` `[n_cand, 2]`;
`candidate_neurons`, `lags`, `a`, `b`, `layer`.
"""

import csv
from pathlib import Path
from typing import Any

import fire
import numpy as np

from param_decomp.log import logger
from param_decomp_lab.experiments.lm.run import SavedLMRun
from param_decomp_lab.infra.paths import ModelPath
from param_decomp_lab.scripts.validation.common import (
    analysis_datasets_dir,
    load_component_uv,
    load_target_mlp_weights,
)
from param_decomp_lab.scripts.validation.neurons.common import (
    NEURONS_DIR,
    silu_combine,
    translation_lags,
    translation_scores,
)

MLP_PROJS = ("gate_proj", "up_proj", "down_proj")


def _r2(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-column R² of pred vs target, both `[n_points, n_neurons]`."""
    resid = ((target - pred) ** 2).sum(axis=0)
    var = ((target - target.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = 1.0 - resid / var
    return np.where(var > 1e-6, r2, 0.0).astype(np.float32)


def compute_subcomp_neuron_links(
    model_path: ModelPath,
    acts_npz: str | None = None,
    candidates_tsv: str | None = None,
    subcomp_screen_npz: str | None = None,
    subcomp_kl_thr: float = 0.01,
    layer: int = 18,
    output: str | None = None,
) -> Path:
    saved = SavedLMRun.from_path(model_path)
    run_dir = saved.checkpoint_path.parent
    uv = load_component_uv(saved.checkpoint_path, layer, MLP_PROJS)
    weights = load_target_mlp_weights(saved.checkpoint_path, layer, MLP_PROJS)

    acts_path = Path(acts_npz).expanduser() if acts_npz else NEURONS_DIR / "activations_add.npz"
    acts = np.load(acts_path)
    x = acts["mlp_input"].astype(np.float32)  # [N, N, d_model]
    gate = acts["gate_preact"]
    up = acts["up_preact"]
    n_side = x.shape[0]
    x_flat = x.reshape(-1, x.shape[-1])
    neuron_acts_flat = silu_combine(gate, up).reshape(-1, gate.shape[-1])

    cand_path = (
        Path(candidates_tsv).expanduser() if candidates_tsv else NEURONS_DIR / "candidates.tsv"
    )
    with open(cand_path) as f:
        cand = np.array([int(r["neuron"]) for r in csv.DictReader(f, delimiter="\t")])
    logger.info(f"{len(cand)} candidate neurons")

    screen_path = (
        Path(subcomp_screen_npz).expanduser()
        if subcomp_screen_npz
        else analysis_datasets_dir(run_dir) / "subcomp_ablation_screen_add.npz"
    )
    screen = np.load(screen_path)
    screen_matrix = screen["matrix"]
    screen_kl = screen["kl"].astype(np.float32).reshape(len(screen_matrix), -1)
    screen_comp = screen["component"]

    lags = translation_lags()
    arrays: dict[str, Any] = {
        "candidate_neurons": cand.astype(np.int32),
        "lags": lags,
        "a": acts["a"],
        "b": acts["b"],
        "layer": layer,
        "subcomp_kl_thr": subcomp_kl_thr,
    }
    causal_by_proj: dict[str, np.ndarray] = {}
    for proj in MLP_PROJS:
        v, u = uv[proj]  # V [d_in, C], U [C, d_out]
        source = neuron_acts_flat if proj == "down_proj" else x_flat
        inner = source @ v  # [n_points, C]
        inner_grids = inner.T.reshape(v.shape[1], n_side, n_side)
        arrays[f"inner_{proj}"] = inner_grids.astype(np.float16)
        arrays[f"inner_std_{proj}"] = inner.std(axis=0).astype(np.float32)
        arrays[f"period_score_{proj}"] = translation_scores(inner_grids, lags)

        mask = screen_matrix == proj
        assert mask.any(), f"no {proj} rows in {screen_path}"
        max_kl = np.zeros(v.shape[1], dtype=np.float32)
        max_kl[screen_comp[mask]] = screen_kl[mask].max(axis=1)
        causal_by_proj[proj] = max_kl > subcomp_kl_thr
        arrays[f"max_kl_{proj}"] = max_kl
        arrays[f"causal_{proj}"] = causal_by_proj[proj]
        logger.info(
            f"{proj}: C={v.shape[1]}, causal (screen KL > {subcomp_kl_thr}): "
            f"{int(causal_by_proj[proj].sum())}"
        )

    arrays["u_gate"] = uv["gate_proj"][1][:, cand].astype(np.float16)  # [C, n_cand]
    arrays["u_up"] = uv["up_proj"][1][:, cand].astype(np.float16)
    arrays["v_down"] = uv["down_proj"][0][cand].astype(np.float16)  # [n_cand, C]

    r2_all = np.zeros((len(cand), 2), dtype=np.float32)
    r2_causal = np.zeros((len(cand), 2), dtype=np.float32)
    targets = {"gate_proj": gate, "up_proj": up}
    for ci, proj in enumerate(("gate_proj", "up_proj")):
        v, u = uv[proj]
        target = targets[proj][:, :, cand].reshape(-1, len(cand)).astype(np.float32)
        pred_all = x_flat @ (v @ u[:, cand])
        causal = causal_by_proj[proj]
        pred_causal = x_flat @ (v[:, causal] @ u[causal][:, cand])
        r2_all[:, ci] = _r2(target, pred_all)
        r2_causal[:, ci] = _r2(target, pred_causal)
        w_row_norm = np.linalg.norm(weights[proj][cand], axis=1)
        w_hat_norm = np.linalg.norm((v @ u[:, cand]).T, axis=1)
        logger.info(
            f"{proj}: r2_all mean {r2_all[:, ci].mean():.3f}, "
            f"r2_causal mean {r2_causal[:, ci].mean():.3f}, "
            f"median |W_hat|/|W| {np.median(w_hat_norm / w_row_norm):.3f}"
        )
    arrays["r2_all"] = r2_all
    arrays["r2_causal"] = r2_causal

    out_path = (
        Path(output).expanduser()
        if output
        else analysis_datasets_dir(run_dir) / "subcomp_neuron_links_add.npz"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    logger.info(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")
    return out_path


if __name__ == "__main__":
    fire.Fire(compute_subcomp_neuron_links)
