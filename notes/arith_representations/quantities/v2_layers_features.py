"""Figure for the two features found in the layer sweep's residuals (2-adic valuation, repdigits):
the residual's top singular function averaged over sites (sign-aligned), and each feature's share
of the residual per site against its permutation null (from v2_feature_test.py).

    python v2_layers_features.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from v2 import A
from v2_sets import v2_adic

HERE = Path(__file__).parent


def mean_u1(sites: list[dict], ref_site: str) -> np.ndarray:
    ref = np.array(next(o for o in sites if o["site"] == ref_site)["resid_u1"])
    us = [np.array(o["resid_u1"]) for o in sites]
    us = [u * np.sign(np.corrcoef(u, ref)[0, 1]) / np.linalg.norm(u) for u in us]
    return np.mean(us, 0)


def main() -> None:
    base = json.loads((HERE / "layers_base.json").read_text())
    blk = lambda o: int(o["site"][1:].split(".")[0])  # noqa: E731
    late = mean_u1([o for o in base if blk(o) >= 14], "L26.mlp")
    early = mean_u1([o for o in base if 1 <= blk(o) <= 13], "L4.mlp")
    late *= np.sign(late[47])  # positive at a = 48
    early *= np.sign(early[21])  # positive at a = 22
    v = np.minimum(v2_adic(A), 4)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    ax = axes[0]
    cols = ["0.75", "C0", "C2", "C3", "C3"]
    ax.bar(A, late, color=[cols[k] for k in v])
    ax.set_title("residual top singular function, mean over sites L14-L31 (L0 set): colour = 2-adic valuation "
                 "(grey odd, blue 2 mod 4, green 4 mod 8, red 0 mod 8)", fontsize=8)  # fmt: skip
    ax.set_xticks(range(0, 101, 4))
    ax.tick_params(labelsize=6)
    ax = axes[1]
    rep = (A % 11 == 0) & (A <= 99)
    ax.bar(A, early, color=["C3" if r else "0.6" for r in rep])
    ax.set_title(
        "residual top singular function, mean over sites L1-L13 (L0 set): red = repdigits 11, 22, ..., 99",
        fontsize=8,
    )
    ax.set_xticks(range(0, 101, 11))
    ax.tick_params(labelsize=6)
    ax = axes[2]
    for name, c in (("two_adic", "C3"), ("repdigit", "C0")):
        ft = json.loads((HERE / f"feature_test_L0_set_{name}.json").read_text())
        x = np.arange(len(ft))
        ax.plot(
            x,
            [o["share"] for o in ft],
            ".-",
            color=c,
            lw=0.8,
            label=f"{name}: share of the residual",
        )
        ax.plot(
            x,
            [o["null_median"] for o in ft],
            ":",
            color=c,
            lw=0.8,
            label=f"{name}: permutation null (median)",
        )
        sig = [i for i, o in enumerate(ft) if o["p"] < 0.01]
        ax.plot(sig, [ft[i]["share"] for i in sig], "o", mfc="none", color=c, ms=6)
    ax.set_xticks(x, [o["site"] for o in ft], rotation=90, fontsize=6)
    ax.set_title("share of the residual (reader metric) along each feature's part outside the L0 set's span; "
                 "circles: p < 0.01 over 500 permutations", fontsize=8)  # fmt: skip
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "figures/v2_layers_new_features.png", dpi=90)


if __name__ == "__main__":
    main()
