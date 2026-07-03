"""Pick the causally-relevant neuron candidates from the dense ablation screens.

Selection is the union of two nets, both feeding the stride-1 full-grid ablation (which is
the unaliased ground truth):

- **screen**: max last-position ablation KL over any screened prompt of any op exceeds
  `--kl-thr`. The threshold must clear the measured null-patch noise floor by
  `--floor-margin`x (asserted), so "candidate" always means "distinguishable from patching
  nothing". Argmax flips alone do NOT qualify — on uncertain prompts the argmax flips
  between near-tied tokens at negligible KL (~10k neurons "flip" somewhere).
- **bound**: the screen strides the grid (a, b ≡ 0 mod stride), which phase-aliases exactly
  the periodic neurons this census is after — a neuron mattering only at `a ≡ 2 (mod 5)` is
  never sampled. Safety net: per op, the top `--bound-top` neurons by the full-grid
  perturbation bound `max |silu(gate)·up| · ‖down column‖` (from `activations_<op>.npz` +
  `subspace.npz` norms) join regardless of screen KL.

CPU-only.

Usage:
    python -m param_decomp_lab.scripts.validation.neurons.select_candidate_neurons \
        [--census-dir=PATH] [--kl-thr=0.01] [--floor-margin=3.0] [--bound-top=256] \
        [--output=PATH]

Output (default `candidates.tsv` in the census dir): one row per candidate neuron, sorted by
overall max KL — `neuron`, `source` (screen / bound / both), per op `max_kl_<op>`,
`mean_kl_<op>`, `n_flip_<op>`, `min_dlp_<op>` (most negative correct-logprob shift),
`bound_<op>`.
"""

import csv
from pathlib import Path

import fire
import numpy as np

from param_decomp.log import logger
from param_decomp_lab.scripts.validation.neurons.common import (
    NEURON_OPS,
    NEURONS_DIR,
    silu_combine,
)


def _perturbation_bound(acts_path: Path, down_norms: np.ndarray) -> np.ndarray:
    acts = np.load(acts_path)
    combined = silu_combine(acts["gate_preact"], acts["up_preact"])  # [N, N, d_int]
    return np.abs(combined).max(axis=(0, 1)) * down_norms


def select_candidate_neurons(
    census_dir: str | None = None,
    kl_thr: float = 0.01,
    floor_margin: float = 3.0,
    bound_top: int = 256,
    output: str | None = None,
) -> Path:
    root = Path(census_dir).expanduser() if census_dir else NEURONS_DIR
    screens = {op: root / f"ablation_screen_{op}.npz" for op in NEURON_OPS}
    screens = {op: p for op, p in screens.items() if p.exists()}
    assert screens, f"no ablation_screen_<op>.npz found in {root}"

    per_op: dict[str, dict[str, np.ndarray]] = {}
    neuron_ids: np.ndarray | None = None
    for op, path in screens.items():
        data = np.load(path)
        floor = float(data["null_kl"].max())
        assert kl_thr >= floor_margin * floor, (
            f"{op}: kl_thr {kl_thr} is under {floor_margin}x the null floor {floor:.2e}"
        )
        nid = data["neuron_ids"]
        if neuron_ids is None:
            neuron_ids = nid
        else:
            assert np.array_equal(neuron_ids, nid)
        kl = data["kl"].astype(np.float32).reshape(len(nid), -1)
        flip = data["answer_flip"].reshape(len(nid), -1)
        dlp = data["delta_correct_logprob"].astype(np.float32).reshape(len(nid), -1)
        per_op[op] = {
            "max_kl": kl.max(axis=1),
            "mean_kl": kl.mean(axis=1),
            "n_flip": flip.sum(axis=1),
            "min_dlp": dlp.min(axis=1),
        }
        logger.info(f"{op}: screen {path.name}, null floor {floor:.2e}")
    assert neuron_ids is not None

    down_norms = np.load(root / "subspace.npz")["norms"][:, 2]
    bounds: dict[str, np.ndarray] = {}
    bound_set: set[int] = set()
    for op in per_op:
        acts_path = root / f"activations_{op}.npz"
        assert acts_path.exists(), f"bound net needs {acts_path} (run collect_neuron_activations)"
        bounds[op] = _perturbation_bound(acts_path, down_norms)
        top = np.argsort(-bounds[op])[:bound_top]
        bound_set.update(int(i) for i in top)
        logger.info(f"{op}: bound net top-{bound_top}, min kept bound {bounds[op][top[-1]]:.3g}")

    overall_max = np.max([s["max_kl"] for s in per_op.values()], axis=0)
    screen_keep = overall_max > kl_thr
    keep = [
        int(i)
        for i in np.argsort(-overall_max)
        if screen_keep[i] or int(neuron_ids[i]) in bound_set
    ]
    n_both = sum(1 for i in keep if screen_keep[i] and int(neuron_ids[i]) in bound_set)
    logger.info(
        f"{len(keep)} candidates: {int(screen_keep.sum())} by screen KL > {kl_thr}, "
        f"{len(bound_set)} by bound, {n_both} both"
    )

    out_path = Path(output).expanduser() if output else root / "candidates.tsv"
    fields = (
        ["neuron", "source"]
        + [f"{k}_{op}" for op in per_op for k in ("max_kl", "mean_kl", "n_flip", "min_dlp")]
        + [f"bound_{op}" for op in per_op]
    )
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for i in keep:
            nid_i = int(neuron_ids[i])
            in_screen, in_bound = bool(screen_keep[i]), nid_i in bound_set
            row: dict[str, object] = {
                "neuron": nid_i,
                "source": "both" if in_screen and in_bound else "screen" if in_screen else "bound",
            }
            for op, s in per_op.items():
                row[f"max_kl_{op}"] = f"{s['max_kl'][i]:.4g}"
                row[f"mean_kl_{op}"] = f"{s['mean_kl'][i]:.4g}"
                row[f"n_flip_{op}"] = int(s["n_flip"][i])
                row[f"min_dlp_{op}"] = f"{s['min_dlp'][i]:.4g}"
                row[f"bound_{op}"] = f"{bounds[op][i]:.4g}"
            writer.writerow(row)
    logger.info(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    fire.Fire(select_candidate_neurons)
