"""Pick the causally-relevant neuron candidates from the dense ablation screens.

Reads `ablation_screen_<op>.npz` for every op present and keeps a neuron when its max
last-position ablation KL over any screened prompt of any op exceeds `--kl-thr`, or when it
flips the model's argmax answer anywhere. The threshold must clear the measured null-patch
noise floor by `--floor-margin`x (asserted), so "candidate" always means "distinguishable
from patching nothing".

CPU-only.

Usage:
    python -m param_decomp_lab.scripts.validation.neurons.select_candidate_neurons \
        [--census-dir=PATH] [--kl-thr=0.01] [--floor-margin=5.0] [--output=PATH]

Output (default `candidates.tsv` in the census dir): one row per candidate neuron, sorted by
overall max KL — `neuron`, then per op `max_kl_<op>`, `mean_kl_<op>`, `n_flip_<op>`,
`min_dlp_<op>` (most negative correct-logprob shift).
"""

import csv
from pathlib import Path

import fire
import numpy as np

from param_decomp.log import logger
from param_decomp_lab.scripts.validation.neurons.common import NEURON_OPS, NEURONS_DIR


def select_candidate_neurons(
    census_dir: str | None = None,
    kl_thr: float = 0.01,
    floor_margin: float = 5.0,
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
    overall_max = np.max([s["max_kl"] for s in per_op.values()], axis=0)
    any_flip = np.sum([s["n_flip"] for s in per_op.values()], axis=0) > 0
    keep = (overall_max > kl_thr) | any_flip
    order = np.argsort(-overall_max)
    kept = [int(i) for i in order if keep[i]]
    logger.info(
        f"{len(kept)}/{len(neuron_ids)} candidates (kl_thr {kl_thr}; "
        f"{int(any_flip.sum())} flip somewhere)"
    )

    out_path = Path(output).expanduser() if output else root / "candidates.tsv"
    fields = ["neuron"] + [
        f"{k}_{op}" for op in per_op for k in ("max_kl", "mean_kl", "n_flip", "min_dlp")
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for i in kept:
            row: dict[str, object] = {"neuron": int(neuron_ids[i])}
            for op, s in per_op.items():
                row[f"max_kl_{op}"] = f"{s['max_kl'][i]:.4g}"
                row[f"mean_kl_{op}"] = f"{s['mean_kl'][i]:.4g}"
                row[f"n_flip_{op}"] = int(s["n_flip"][i])
                row[f"min_dlp_{op}"] = f"{s['min_dlp'][i]:.4g}"
            writer.writerow(row)
    logger.info(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    fire.Fire(select_candidate_neurons)
