"""Reader activations at one site of the alive-only model, as functions of the prompt.

A site is (block b, read point, token position t): the readers of block b that read the
residual stream before its attention ("attn": q, k, v; stream position l = 2b) or before its
MLP ("mlp": gate, up; l = 2b + 1), restricted to the readers ACTIVE at t (original-model CI
> 0.01 somewhere on the (a, b) grid at t). Activations are the alive-only model's inner
activations in the ||V|| = 1 gauge (`virtual_weights/alive_only`).

At t = 1 (the "a" token) every activation is a function of a alone (causal attention), so a
site is a (100, n_readers) matrix over a = 1..100.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN = Path.home() / "out/pod-backup/p-ba5a0c05"
DATASET = RUN / "analysis/ci_filter/step_40000/addsub-05-filter-last-pos-ceiling/dataset"
VW = RUN / "analysis/virtual_weights"
OUT = RUN / "analysis/quantities"
READ_KINDS = {"attn": ("q", "k", "v"), "mlp": ("gate", "up")}


@dataclass
class Site:
    block: int
    point: str  # "attn" | "mlp"
    pos: int
    names: list[str]
    cols: np.ndarray  # dataset columns
    X: np.ndarray  # (100, n): activation of reader j at a = 1..100

    @property
    def label(self) -> str:
        return f"L{self.block}.{self.point}-in@pos{self.pos}"


def load_site(block: int, point: str, pos: int = 1) -> Site:
    assert pos == 1, "functions of a alone only at the a token"
    ix = np.load(DATASET / "index.npz")
    kind = np.array([k.split(".")[-1].replace("_proj", "") for k in ix["comp_kind"]])
    act = np.load(VW / "ci_positions.npz")["ci_max"][:, pos] > 0.01
    sel = np.flatnonzero((ix["comp_layer"] == block) & np.isin(kind, READ_KINDS[point]) & act)
    compact = VW / "alive_only/inner_token_a.npy"  # (100, A): inner[ROWS, 1], for pods
    if compact.exists():
        assert pos == 1
        inner = np.load(compact)[:, None, :].repeat(2, 1)  # rows indexed below as [rows, 1]
        rows_c = np.arange(100)
        X = np.asarray(inner[rows_c, pos][:, sel], np.float64) / ix["comp_v_norm"][sel]
        names = [f"L{block}.{kind[c]}.c{int(ix['comp_index'][c])}" for c in sel]
        return Site(block, point, pos, names, sel, X)
    inner = np.load(VW / "alive_only/inner.npy", mmap_mode="r")
    rows = (
        np.arange(100) * 100
    )  # op = +, b = 1, a = 1..100 (prompt i = op*10000 + (a-1)*100 + (b-1))
    X = np.asarray(inner[rows, pos][:, sel], np.float64) / ix["comp_v_norm"][sel]
    names = [f"L{block}.{kind[c]}.c{int(ix['comp_index'][c])}" for c in sel]
    return Site(block, point, pos, names, sel, X)


def plot_curves(curves: dict[str, np.ndarray], names: list[str], path: Path, ncol: int = 6,
                title: str = "") -> None:  # fmt: skip
    """One panel per reader; `curves` maps a legend label to a (100, n) matrix."""
    n = len(names)
    nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 1.9 * nrow), squeeze=False)
    a = np.arange(1, 101)
    styles = ["k-", "C1-", "C3-", "C0-"]
    for j in range(nrow * ncol):
        ax = axes.flat[j]
        if j >= n:
            ax.axis("off")
            continue
        for (lab, Y), st in zip(curves.items(), styles, strict=False):
            ax.plot(a, Y[:, j], st, lw=0.9, label=lab, ms=2)
        ax.set_title(names[j], fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        ax.grid(alpha=0.3, lw=0.4)
    axes.flat[0].legend(fontsize=6)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)
