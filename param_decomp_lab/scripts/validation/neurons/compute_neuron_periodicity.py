"""Periodicity expansion: translation-invariance score per neuron × channel × (Δa, Δb) lag.

Reads an `activations_<op>.npz` (from `collect_neuron_activations`) and scores, for every
neuron and each channel (`gate` / `up` / `combined` = post-SwiGLU `silu(gate)·up`), how
translation-invariant its `[201, 201]` activation grid is at every lag in `translation_lags()`
— pure-a `(p, 0)`, pure-b `(0, p)`, and all mixed `(p, ±q)` for p, q in `PERIODS`
(2, 5, 10, 20, 25, 33, 50, 100). The score is a mean-centred Pearson correlation between the
grid and its shifted self over the overlap: no sinusoid assumption, and mixed lags catch
checkerboard / diagonal patterns a marginal-based detector misses. A true period-p pattern
also scores high at multiples of p — consumers should read profiles, not argmaxes.

With `--ablation-npz` it additionally scores each candidate neuron's KL grid from a (merged)
full ablation npz, storing it as an extra channel block keyed by that file's `neuron_ids`.

CPU-only.

Usage:
    python -m param_decomp_lab.scripts.validation.neurons.compute_neuron_periodicity \
        <activations_npz> [--ablation-npz=PATH] [--output=PATH]

Output (default `periodicity_<op>.npz` beside the input):
- `score`      — `[14336, 3, n_lags]` fp32, channels ordered (gate, up, combined)
- `lags`       — `[n_lags, 2]` int32 `(Δa, Δb)`
- `kl_score`   / `kl_neuron_ids` — `[n_cand, n_lags]` fp32 (only with `--ablation-npz`)
- `channels`, `periods`, `op`
"""

from pathlib import Path
from typing import Any

import fire
import numpy as np

from param_decomp.log import logger
from param_decomp_lab.scripts.validation.neurons.common import (
    PERIODS,
    silu_combine,
    translation_lags,
    translation_scores,
)

CHANNELS = ("gate", "up", "combined")


def compute_neuron_periodicity(
    activations_npz: str,
    ablation_npz: str | None = None,
    output: str | None = None,
) -> Path:
    acts_path = Path(activations_npz).expanduser()
    data = np.load(acts_path)
    op = str(data["op"])
    gate = data["gate_preact"]  # [N, N, d_int] f16
    up = data["up_preact"]
    n_side, _, d_int = gate.shape
    assert data["a"].shape[0] == n_side
    lags = translation_lags()

    scores = np.zeros((d_int, len(CHANNELS), len(lags)), dtype=np.float32)
    for ci, channel in enumerate(CHANNELS):
        match channel:
            case "gate":
                grids = np.transpose(gate, (2, 0, 1))
            case "up":
                grids = np.transpose(up, (2, 0, 1))
            case "combined":
                grids = np.transpose(silu_combine(gate, up), (2, 0, 1))
        scores[:, ci] = translation_scores(grids, lags)
        logger.info(f"{op}/{channel}: scored {d_int} neurons x {len(lags)} lags")

    extra: dict[str, Any] = {}
    if ablation_npz is not None:
        abl = np.load(Path(ablation_npz).expanduser())
        assert int(abl["stride"]) == 1, "KL periodicity needs the full-grid ablation npz"
        kl_grids = abl["kl"].astype(np.float32)  # [n_cand, N, N]
        extra = {
            "kl_score": translation_scores(kl_grids, lags),
            "kl_neuron_ids": abl["neuron_ids"],
        }
        logger.info(f"{op}/kl: scored {kl_grids.shape[0]} candidate neurons")

    out_path = Path(output).expanduser() if output else acts_path.parent / f"periodicity_{op}.npz"
    arrays: dict[str, Any] = {
        "score": scores,
        "lags": lags,
        "channels": np.array(CHANNELS),
        "periods": np.array(PERIODS, dtype=np.int32),
        "op": op,
        **extra,
    }
    np.savez_compressed(out_path, **arrays)
    logger.info(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")
    return out_path


if __name__ == "__main__":
    fire.Fire(compute_neuron_periodicity)
