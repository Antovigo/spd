"""Decomposed vs original inner activation of every alive component, one panel per matrix kind.

    python notes/ci_filter/plot_inner_scatter.py <dataset dir>

`<dataset dir>` is a `collect_dataset.py` output. A component's point is its inner activation
`x @ V_c` at ONE (prompt, position) entry drawn at random (seed 0) among those where it is ON in
the decomposed model (original-model CI > the dataset threshold), in the original model (x) and
in the decomposed model (y). Writes `<dataset dir>/figures/inner_scatter.png` and the per-component values to
`<dataset dir>/figures/inner_scatter.tsv`."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KINDS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
CHUNK = 500


def on_sample(
    dataset: Path, threshold: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`(original, decomposed, prompt, position)` per component: its inner activation at ONE
    (prompt, position) entry drawn uniformly from those where it is on (reservoir over chunks:
    every on entry gets a uniform key, the largest key wins)."""
    rng = np.random.default_rng(seed)
    ci = np.load(dataset / "original" / "ci.npy", mmap_mode="r")
    inner = {
        m: np.load(dataset / m / "inner.npy", mmap_mode="r") for m in ("original", "decomposed")
    }
    n_prompts, n_pos, n_comp = ci.shape
    best = np.full(n_comp, -1.0)
    picked = {m: np.zeros(n_comp) for m in inner}
    prompt, position = np.zeros(n_comp, np.int64), np.zeros(n_comp, np.int64)
    cols = np.arange(n_comp)
    for start in range(0, n_prompts, CHUNK):
        on = np.asarray(ci[start : start + CHUNK], np.float32) > threshold
        keys = np.where(on, rng.random(on.shape), -1.0).reshape(-1, n_comp)
        arg = keys.argmax(axis=0)
        win = keys[arg, cols] > best
        best[win] = keys[arg, cols][win]
        for m, values in inner.items():
            flat = np.asarray(values[start : start + CHUNK]).reshape(-1, n_comp)
            picked[m][win] = flat[arg, cols][win]
        prompt[win] = start + arg[win] // n_pos
        position[win] = arg[win] % n_pos
    assert (best >= 0).all(), f"{(best < 0).sum()} alive components never on"
    return picked["original"], picked["decomposed"], prompt, position


def main() -> None:
    dataset = Path(sys.argv[1])
    meta = json.loads((dataset / "meta.json").read_text())
    index = np.load(dataset / "index.npz")
    kind, layer = index["comp_kind"], index["comp_layer"]
    x, y, prompt, position = on_sample(dataset, meta["threshold"], seed=0)

    out = dataset / "figures"
    out.mkdir(exist_ok=True)
    rows = ["site\tcomponent\tprompt\tposition\toriginal\tdecomposed"]
    rows += [
        f"{s}\t{c}\t{p}\t{t}\t{a:.6g}\t{b:.6g}"
        for s, c, p, t, a, b in zip(
            index["comp_site"], index["comp_index"], prompt, position, x, y, strict=True
        )
    ]
    (out / "inner_scatter.tsv").write_text("\n".join(rows) + "\n")

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), constrained_layout=True)
    norm = plt.Normalize(0, meta["n_layer"] - 1)
    for ax, k in zip(axes.flat, KINDS, strict=False):
        sel = kind == k
        lo = min(x[sel].min(), y[sel].min())
        hi = max(x[sel].max(), y[sel].max())
        pad = 0.05 * (hi - lo)
        lims = (lo - pad, hi + pad)
        ax.plot(lims, lims, "k--", lw=1, zorder=1)
        order = np.argsort(layer[sel])
        points = ax.scatter(
            x[sel][order],
            y[sel][order],
            c=layer[sel][order],
            cmap="viridis",
            norm=norm,
            s=14,
            edgecolors="black",
            linewidths=0.3,
            zorder=2,
        )
        ax.set(xlim=lims, ylim=lims, aspect="equal", title=f"{k}  (n={sel.sum()})")
        ax.set_xlabel("original model")
        ax.set_ylabel("decomposed model")
        ax.grid(alpha=0.25, lw=0.5)
    axes.flat[-1].set_visible(False)
    fig.colorbar(points, ax=axes, label="layer", shrink=0.6)
    fig.suptitle(
        f"Inner activation x·V_c at one random (prompt, position) where the component is on — {meta['source']}, "
        f"{meta['n_alive']} alive components, {meta['n_prompts']} prompts"
    )
    fig.savefig(out / "inner_scatter.png", dpi=150)
    print(f"-> {out / 'inner_scatter.png'}")


if __name__ == "__main__":
    main()
