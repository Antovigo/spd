"""Worst sites of the V3 patching test: which readers carry the residual where they are causally
important, and what their reads look like.

For a site with readers r, raw reads Y (100 x n), reconstruction Yhat and CI (100 x n): the
CI-weighted residual of reader r is sum_a CI(a, r) (Y - Yhat)(a, r)^2. Plots the 6 readers with the
largest, with the read, the reconstruction and the CI over a.

    python v3_inspect.py L0.mlp L16.attn ...
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from v3 import load

HERE = Path(__file__).parent
A = np.arange(1, 101)


def main() -> None:
    R = np.load(HERE / "v3_recon.npz")
    runs = {s["site"]: s for s in json.loads((HERE / "v3_token_a.json").read_text())}
    sites = sys.argv[1:]
    fig, axes = plt.subplots(len(sites), 6, figsize=(22, 2.6 * len(sites)), squeeze=False)
    for i, label in enumerate(sites):
        b, pt = int(label[1:].split(".")[0]), label.split(".")[1]
        site = load(b, pt)
        Y, Yh = R[label + "/Y"], R[label + "/Yhat"]
        E = Y - Yh
        w = (site.CI * E**2).sum(0)
        tot = (site.CI * (Y - Y.mean(0)) ** 2).sum(0) + 1e-30
        order = np.argsort(-w)
        print(f"== {label}: accepted {[a['name'] for a in runs[label]['accepted']]}")
        print(f"   CI-weighted residual share of the site: {w.sum() / tot.sum():.3f}; top readers carry "
              f"{w[order[:6]].sum() / w.sum():.2f} of it")  # fmt: skip
        for j, r in enumerate(order[:6]):
            ax = axes[i, j]
            ax.plot(A, Y[:, r], "k-", lw=0.9, label="raw read")
            ax.plot(A, Yh[:, r], "C1-", lw=0.9, label="reconstruction")
            ax2 = ax.twinx()
            ax2.bar(A, site.CI[:, r], color="C0", alpha=0.25, width=1.0)
            ax2.set_ylim(0, 1.05)
            ax2.tick_params(labelsize=5)
            share = w[r] / w.sum()
            ax.set_title(f"{label} {site.names[r]}: {share:.2f} of the site's CI-weighted residual\n"
                         f"reader R2 on CI-weighted entries {1 - w[r] / tot[r]:.2f}", fontsize=7)  # fmt: skip
            ax.tick_params(labelsize=6)
            imp = np.flatnonzero(site.CI[:, r] > 0.01) + 1
            top = np.argsort(-site.CI[:, r] * E[:, r] ** 2)[:6] + 1
            print(f"   {site.names[r]:<16} share {share:.2f}  important at {len(imp)} values; largest weighted "
                  f"errors at a = {top.tolist()}")  # fmt: skip
        axes[i, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(HERE / "figures/v3_worst_sites.png", dpi=80)


if __name__ == "__main__":
    main()
