"""Which alive components never reach CI = 1, and what the most active of them look like.

CI is the filter's output CI on the original model (`dataset/original/ci.npy`, (N, T, A) fp16).
For every alive component: max CI over all prompts and positions, max CI at the last position,
mean CI over all prompts and positions. Components with max CI < 1 - TOL are the ones a
"reaches CI = 1 somewhere on the grid" criterion would exclude; the top 10 of them by mean CI
are drawn (alive-only inner activation in the ||V|| = 1 gauge and CI, a + b and a - b) at the
token position where their mean CI is largest.

Writes OUT/{stats.json, top10_excluded.png}.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from param_decomp.arith_repr.vectors.common import DATASET, RUN, comp_table, name

OUT = RUN / "analysis/virtual_weights/ci_max"
POS = ("<BOS>", "a", "op", "b", "=")
TOL = 1e-3


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ci = np.load(DATASET / "original/ci.npy")  # (N, T, A) float16
    mx = ci.max((0, 1)).astype(np.float32)
    mx_last = ci[:, -1].max(0).astype(np.float32)
    mean = ci.mean((0, 1), dtype=np.float32)
    mean_pos = ci.mean(0, dtype=np.float32)  # (T, A)
    comps = comp_table()
    excl = np.flatnonzero(mx < 1 - TOL)
    top = excl[np.argsort(-mean[excl])][:10]
    stats = {
        "n_alive": int(ci.shape[2]),
        "n_max_ci_below_1": int(len(excl)),
        "n_max_ci_last_pos_below_1": int((mx_last < 1 - TOL).sum()),
        "n_max_ci_exactly_1": int((mx == 1.0).sum()),
        "excluded_by_kind": {k: int((comps["kind"][excl] == k).sum()) for k in np.unique(comps["kind"])},
        "max_ci_quantiles_of_excluded": np.quantile(mx[excl], [0.1, 0.5, 0.9]).tolist(),
        "top10": [
            {"col": int(c), "name": name(comps, int(c)), "max_ci": float(mx[c]),
             "max_ci_last": float(mx_last[c]), "mean_ci": float(mean[c]),
             "mean_ci_by_pos": mean_pos[:, c].tolist()}
            for c in top
        ],
    }  # fmt: skip
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "top10"}, indent=1), flush=True)
    for t in stats["top10"]:
        print(t["name"], f"max {t['max_ci']:.3f} mean {t['mean_ci']:.4f}", flush=True)

    inner = np.load(RUN / "analysis/virtual_weights/alive_only/inner.npy", mmap_mode="r")
    vnorm = np.load(DATASET / "index.npz")["comp_v_norm"]
    fig, axes = plt.subplots(10, 4, figsize=(11, 27), squeeze=False)
    for row, c in enumerate(top):
        p = int(np.argmax(mean_pos[:, c]))
        x = np.asarray(inner[:, p, c], np.float32).reshape(2, 100, 100) / vnorm[c]
        y = ci[:, p, c].astype(np.float32).reshape(2, 100, 100)
        s = np.abs(x).max()
        for op in range(2):
            ax = axes[row, op]
            ax.imshow(x[op], cmap="RdBu_r", vmin=-s, vmax=s)
            ax.set_title(f"inner, a {'+-'[op]} b (±{s:.2g})", fontsize=8)
            ax = axes[row, 2 + op]
            ax.imshow(y[op], cmap="RdPu", vmin=0, vmax=1)
            ax.set_title(f"CI, a {'+-'[op]} b (max {y[op].max():.2f})", fontsize=8)
        axes[row, 0].set_ylabel(
            f"{name(comps, int(c))}\npos {p} ({POS[p]})\nmean CI {mean[c]:.3f}", fontsize=8
        )
        for ax in axes[row]:
            ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle("Top 10 alive components with max CI < 1 (by mean CI); grids: a = 1..100 down, b = 1..100 across",
                 fontsize=10)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(OUT / "top10_excluded.png", dpi=90)
    print("figure written", flush=True)


if __name__ == "__main__":
    main()
