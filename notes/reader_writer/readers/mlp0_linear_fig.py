"""Figure of the notebook page "Mod 10 representations", section 7 (from mlp0_linear.py's outputs).

python mlp0_linear_fig.py --out <notebook dir>
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mlp0_mod10 import MOD10, OUT
from mod10_L0 import A

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)
R = A % 10


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    js = json.loads((OUT / "linear.json").read_text())
    z = np.load(OUT / "linear.npz")
    dC, Wx = z["dC"], z["W_x"]
    names = [f"{n.split('.')[0][0]}{n.split('.')[1]}" for n in MOD10]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))

    h = np.linspace(-2.5, 2.0, 400)
    lo, hi = js["gate_h_range_top200_neurons"]
    ax[0].axvspan(lo, hi, color="#2a5db0", alpha=0.1, label="range of h_g on the data")
    ax[0].plot(h, h / (1 + np.exp(-h)), "k", label="SiLU(h)")
    ax[0].plot(h, np.maximum(h, 0), "--", color="#999", lw=1, label="ReLU(h)")
    ax[0].plot(h, h / 2, ":", color="#c0392b", lw=1, label="h / 2 (slope at 0)")
    ax[0].set_xlabel("gate pre-activation h")
    ax[0].set_title("(a) the gates are not in a linear regime")
    ax[0].legend(frameon=False, fontsize=7)

    x = np.arange(10)
    ax[1].bar(
        x, js["t_energy_on_main_residue"], color="#2a5db0", label="share of Σ_a t_c² on its residue"
    )
    ax[1].plot(x, [js["per_reader"][n]["r2_secant"] for n in MOD10], "o", color="#c0392b",
               label="r² of Δ_c ≈ κ_c t_c")  # fmt: skip
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(
        [f"{n}\nr={r}" for n, r in zip(names, js["t_main_residue"], strict=True)], fontsize=7
    )
    ax[1].set_ylim(0, 1.3)
    ax[1].set_title("(b) one-hot readers make each path linear")
    ax[1].legend(frameon=False, fontsize=7, loc="upper left", ncol=2)

    def amps(X: np.ndarray) -> np.ndarray:
        mu = np.stack([X[r == R].mean(0) for r in range(10)])
        return np.asarray(
            [np.linalg.norm(np.exp(-2j * np.pi * k * np.arange(10) / 10) @ mu) for k in range(1, 6)]
        )

    k = np.arange(1, 6)
    ax[2].bar(k - 0.2, amps(dC), 0.4, color="#1d1d1b", label="Δ_C (the period-10 path)")
    ax[2].bar(k + 0.2, amps(Wx), 0.4, color="#2a5db0", label="W x (one linear map)")
    ax[2].set_xticks(k)
    ax[2].set_xlabel("frequency k")
    ax[2].set_ylabel("‖Ψ_k‖")
    ax[2].set_title("(c) Fourier amplitudes of the output")
    ax[2].legend(frameon=False, fontsize=7, loc="upper center")
    ax[2].set_ylim(0, 1.3)

    n_dc = np.linalg.norm(dC, axis=1)
    ax[3].plot(A, n_dc, color="#1d1d1b", lw=0.8, label="‖Δ_C(a)‖")
    ax[3].plot(
        A, np.linalg.norm(dC - Wx, axis=1), color="#c0392b", lw=0.8, label="‖Δ_C(a) − W x(a)‖"
    )
    ax[3].set_xlabel("a")
    ax[3].set_title(f"(d) per prompt: r² = {js['r2_DeltaC_vs_Wx']:.3f}")
    ax[3].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out / "fig" / "mod10_mlp_linear.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
