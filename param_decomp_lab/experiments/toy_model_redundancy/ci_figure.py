"""Subcomponent-CI grid figure for toy-model-redundancy decompositions (shared by the
plot_ci script and the ToyRedundancyCIPlot eval metric)."""

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from param_decomp_lab.toy_models.target_ci import permute_to_identity  # noqa: E402


def plot_subcomponent_grid(cis: dict[str, np.ndarray]) -> Figure:
    """`[matrix, block]` grid of CI heatmaps, square cells, all positions shown.

    `cis` maps `blocks.<b>.<matrix>` to `[vocab, seq, C]` CI arrays. ALL `C`
    subcomponents and ALL `seq` positions are shown: positions are stacked
    vertically within each panel (separated by a line), tokens on the y axis.
    Columns are Hungarian-permuted toward the identity of the per-position max —
    a display ordering only; every plotted value is a raw CI.
    """
    canonical = ("mlp_in", "mlp_out", "q", "k", "v", "o")
    present = {module.rsplit(".", 1)[-1] for module in cis}
    assert present <= set(canonical), f"unknown matrices: {present}"
    matrices = tuple(m for m in canonical if m in present)
    blocks_present = sorted({int(module.split(".")[1]) for module in cis})
    assert len(cis) == len(blocks_present) * len(matrices), f"unexpected modules: {sorted(cis)}"
    vocab, seq, _ = next(iter(cis.values())).shape
    max_c = max(ci.shape[2] for ci in cis.values())
    panel_rows = seq * vocab
    cell = 0.15
    fig, axes = plt.subplots(
        len(matrices),
        len(blocks_present),
        figsize=(
            len(blocks_present) * (max_c * cell + 0.5) + 0.8,
            len(matrices) * (panel_rows * cell + 0.8),
        ),
        squeeze=False,
        facecolor="white",
        sharey=True,
        gridspec_kw={"hspace": 0.8 / (panel_rows * cell), "wspace": 0.5 / (max_c * cell)},
    )
    im = None
    for module, ci in cis.items():
        _, block_str, matrix = module.split(".")
        ax = axes[matrices.index(matrix), blocks_present.index(int(block_str))]
        vocab, seq, _ = ci.shape
        _, perm = permute_to_identity(torch.from_numpy(ci.max(axis=1)))
        stacked = np.concatenate([ci[:, pos][:, perm.numpy()] for pos in range(seq)], axis=0)
        im = ax.imshow(stacked, aspect="equal", cmap="RdPu", vmin=0, vmax=1, interpolation="none")
        for pos in range(1, seq):
            ax.axhline(pos * vocab - 0.5, color="black", linewidth=0.6)
        for pos in range(seq):
            ax.text(
                len(perm) - 0.5,
                pos * vocab - 0.5,
                f"pos {pos}",
                fontsize=8,
                color="gray",
                ha="right",
                va="top",
            )
        ax.set_yticks(
            range(seq * vocab), [str(t) for _ in range(seq) for t in range(vocab)], fontsize=5
        )
        ax.set_title(module, fontsize=9)
        ax.set_xticks(range(len(perm)), [str(int(c)) for c in perm], fontsize=5, rotation=90)
        ax.tick_params(labelsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel("subcomponent", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("input token", fontsize=8)
    assert im is not None
    fig.colorbar(im, ax=axes, label="causal importance", fraction=0.02)
    for ax in axes.flat:
        ax.set_anchor("NW")
    return fig
