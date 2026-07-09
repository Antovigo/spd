"""Battery divergences per input feature — localizes which TMS subcomponents are wrong.

For each battery experiment, plots the mean per-sample MSE as a function of the input
feature index (a sample contributes to feature `i` when feature `i` is active in it),
one line per flavor (raw / centered / bias) plus the circuit baseline as a green dashed
line. A bump at feature `i` in a span test — but not in the baseline — points at the
subcomponent(s) serving feature `i` reading/writing outside the selected subspaces,
e.g. exactly where the CI heatmap deviates from the identity pattern.

Usage:
    python -m param_decomp_lab.scripts.validation.subspace_filtering.plot_mse_by_feature_tms \
        <run>/analysis/datasets/subspace_filtering/mse_tms.tsv [--single-active]

`--single-active` restricts the aggregation to samples with exactly one active feature
(cleaner attribution, fewer samples per point).

Output: `<run>/analysis/subspace_filtering/mse_by_feature_tms[_single_active].png`.
"""

import csv
from pathlib import Path

import fire
import matplotlib
import numpy as np

from param_decomp.log import logger
from param_decomp_lab.scripts.validation.common import run_dir_of_dataset

_FLAVOR_COLORS = {"raw": "tab:red", "centered": "tab:blue", "bias": "tab:orange"}
_BASELINE_KEY = "circuit_baseline"
_FLOOR = 1e-12


def plot_mse_by_feature_tms(mse_tsv: str, single_active: bool = False) -> Path:
    mse_path = Path(mse_tsv).expanduser()
    run_dir = run_dir_of_dataset(mse_path.parent)
    x = np.load(mse_path.parent / "features_tms.npz")["x"]
    active = x != 0.0  # [n_samples, n_features]
    n_samples, n_features = active.shape
    if single_active:
        keep = active.sum(axis=1) == 1
        logger.info(f"single-active filter: {int(keep.sum())}/{n_samples} samples kept")
    else:
        keep = np.ones(n_samples, bool)

    mse: dict[tuple[str, str], np.ndarray] = {}
    with mse_path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            key = (row["experiment"], row["flavor"])
            mse.setdefault(key, np.zeros(n_samples, np.float32))[int(row["sample"])] = float(
                row["mse"]
            )
    experiments = [e for e in dict.fromkeys(e for e, _ in mse) if e != _BASELINE_KEY]

    def per_feature_mean(vals: np.ndarray) -> np.ndarray:
        return np.array(
            [
                vals[keep & active[:, i]].mean() if (keep & active[:, i]).any() else np.nan
                for i in range(n_features)
            ]
        )

    base_curve = np.maximum(per_feature_mean(mse[(_BASELINE_KEY, "none")]), _FLOOR)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 2
    nrows = -(-len(experiments) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(7.5 * ncols, 3.4 * nrows), squeeze=False, sharex=True, sharey=True
    )
    feat = np.arange(n_features)
    for ei, exp in enumerate(experiments):
        ax = axes[ei // ncols][ei % ncols]
        for flavor, color in _FLAVOR_COLORS.items():
            curve = np.maximum(per_feature_mean(mse[(exp, flavor)]), _FLOOR)
            ax.plot(feat, curve, color=color, lw=1.2, marker=".", ms=4, label=flavor)
        ax.plot(feat, base_curve, color="tab:green", ls="--", lw=1.2, label="circuit baseline")
        ax.set_yscale("log")
        ax.set_title(exp, fontsize=10)
        ax.grid(True, axis="x", alpha=0.2)
    for j in range(len(experiments), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    axes[0][0].legend(fontsize=8)
    for r in range(nrows):
        axes[r][0].set_ylabel("mean MSE (active samples)")
    for c in range(ncols):
        axes[nrows - 1][c].set_xlabel("input feature index")
    scope = "samples with exactly one active feature" if single_active else "all active samples"
    fig.suptitle(f"{run_dir.name}: battery MSE by input feature ({scope})")
    fig.tight_layout()

    suffix = "_single_active" if single_active else ""
    out_path = run_dir / "analysis" / "subspace_filtering" / f"mse_by_feature_tms{suffix}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"wrote {out_path}")
    return out_path


if __name__ == "__main__":
    fire.Fire(plot_mse_by_feature_tms)
