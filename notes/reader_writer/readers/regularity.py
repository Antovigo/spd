"""How regular is the arrangement of the mod-10 directions inside D? Residue means and read directions.

Nothing is centred. Notation (also on the page):
  y_r       = Pi_D mu_r expressed in an orthonormal basis of D (9 numbers), r = 0..9
  C(r, r')  = cos(y_r, y_r'); circulant part: the mean of C over pairs with the same Delta = r' - r mod 10
  S         the 9x9 linear map with S y_r ~= y_{r+1} (least squares over the 10 residues): the "shift by one"
  R_k       rotation by 2 pi k / 10 in plane k (k = 1..4) and sign flip on the line k = 5, in the Fourier basis:
            the ideal shift if every plane were a perfect circle
  q_c       a reader's read direction inside D (Pi_D s w / ||Pi_D s w||, in the same 9-dim coordinates)

    python regularity.py --out <page dir> --neuron_data <dir>
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lobes import COMPONENTS, MIXED, NEURONS
from mod10_L0 import HUB, SRC, A, N, bf16_tensor, kept_readers
from neuron_page import fourier_planes, orth

plt.rcParams.update(
    {"figure.dpi": 110, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
)


def unit(X: np.ndarray, axis: int = -1) -> np.ndarray:
    return X / np.linalg.norm(X, axis=axis, keepdims=True)


def analyse(mu: np.ndarray, rows: np.ndarray, names: list[str]) -> dict:
    P = fourier_planes(mu)
    B = np.concatenate(
        [P[k] for k in range(1, 6)], 1
    )  # (4096, 9) Fourier-ordered basis (orthonormal per block)
    D = orth(B)
    y = mu @ D  # (10, 9)
    out: dict = {}
    C = unit(y) @ unit(y).T
    d = (np.arange(10)[:, None] - np.arange(10)[None, :]) % 10
    off = d != 0
    circ = np.zeros_like(C)
    for dd in range(1, 10):
        circ[d == dd] = C[d == dd].mean()
    out["norms"] = np.linalg.norm(y, axis=1).tolist()
    out["circulant_r2_offdiag"] = float(
        1 - ((C - circ)[off] ** 2).sum() / ((C[off] - C[off].mean()) ** 2).sum()
    )
    out["circ_resid_abs_max"] = float(np.abs(C - circ)[off].max())
    # best linear shift S: y_{r+1} = S y_r
    X, Y = y.T, np.roll(y, -1, axis=0).T  # (9, 10)
    S = Y @ np.linalg.pinv(X)
    out["shift_rel_err"] = float(np.linalg.norm(S @ X - Y) / np.linalg.norm(Y))
    sv = np.linalg.svd(S, compute_uv=False)
    out["shift_singular_values"] = sv.tolist()
    ev = np.linalg.eigvals(S)
    out["shift_eig_abs"] = np.sort(np.abs(ev)).tolist()
    out["shift_eig_angle_x10_over_2pi"] = np.sort(np.abs(np.angle(ev)) * 10 / (2 * np.pi)).tolist()
    S10 = np.linalg.matrix_power(S, 10)
    out["shift10_dist_from_identity"] = float(np.linalg.norm(S10 - np.eye(9)) / 3.0)
    # ideal rotation in the plane coordinates: coordinates of y_r in each P_k should be a circle at angle 2 pi k r / 10
    plane_fit = {}
    for k in range(1, 6):
        c = mu @ P[k]  # (10, d)
        if k < 5:
            z = c[:, 0] + 1j * c[:, 1]
            # best complex-linear fit z_r = A e^{2pi i k r/10} + Bc e^{-2 pi i k r/10} (ellipse); circle if |Bc| << |A|
            E = np.stack(
                [
                    np.exp(2j * np.pi * k * np.arange(10) / 10),
                    np.exp(-2j * np.pi * k * np.arange(10) / 10),
                ],
                1,
            )
            coef, *_ = np.linalg.lstsq(E, z, rcond=None)
            res = z - E @ coef
            plane_fit[str(k)] = {
                "circle_amp": float(max(abs(coef[0]), abs(coef[1]))),
                "other_amp": float(min(abs(coef[0]), abs(coef[1]))),
                "resid_rel": float(np.linalg.norm(res) / np.linalg.norm(z)),
            }
        else:
            s = c[:, 0]
            alt = (-1.0) ** np.arange(10)
            a_ = (s @ alt) / 10
            plane_fit[str(k)] = {
                "circle_amp": float(abs(a_)),
                "other_amp": 0.0,
                "resid_rel": float(np.linalg.norm(s - a_ * alt) / np.linalg.norm(s)),
            }
    out["plane_fit"] = plane_fit
    # readers: direction inside D, mapped by the fitted shift
    q, r0s = [], []
    for w in rows:
        m = mu @ unit(w)
        r0 = int(np.argmax(np.abs(m)))
        qd = np.sign(m[r0]) * (w @ D)
        q.append(unit(qd))
        r0s.append(r0)
    q = np.stack(q)
    r0s = np.asarray(r0s)
    Q = unit(q) @ unit(q).T
    out["r0"] = r0s.tolist()
    # transport: cos(S^Delta q_i, q_j) for readers with residues r_i, r_j
    pairs = []
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            dd = (r0s[j] - r0s[i]) % 10
            Sd = np.linalg.matrix_power(S, dd)
            pairs.append(
                (names[i], names[j], int(dd), float(Q[i, j]), float(unit(Sd @ q[i]) @ q[j]))
            )
    out["pairs"] = pairs
    out["Q"] = Q.tolist()
    # cosine of readers vs residue-mean cosine at the same Delta
    out["C"] = C.tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--neuron_data", type=Path, required=True)
    args = ap.parse_args()
    rr = A % 10
    x = np.load(args.neuron_data / "l0_input.npz")["x"].astype(np.float64)
    mu_n = np.stack([x[rr == r].mean(axis=0) for r in range(10)])
    (snap,) = glob.glob(HUB)
    idx = json.loads(Path(snap, "model.safetensors.index.json").read_text())["weight_map"]
    clean = [n for n in NEURONS if n not in MIXED]
    rows = []
    for nm in clean:
        k, i = nm.split("#")
        key = f"model.layers.0.mlp.{ {'g': 'gate', 'u': 'up'}[k] }_proj.weight"
        rows.append(bf16_tensor(snap + idx[key], key, np.asarray([int(i)]))[0])
    neu = analyse(mu_n, np.stack(rows).astype(np.float64), clean)
    cols, cnames, V = kept_readers()
    inner = np.load(SRC / "inner.npy", mmap_mode="r")
    T = np.asarray(inner[(A - 1) * N, 0, :][:, cols], np.float64)
    xv = (V @ np.linalg.pinv(V.T @ V) @ T.T).T
    mu_c = np.stack([xv[rr == r].mean(axis=0) for r in range(10)])
    com = analyse(mu_c, np.stack([V[:, cnames.index(c)] for c in COMPONENTS]), COMPONENTS)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.0))
    # (1) residue-mean cosines vs Delta: every pair
    shift = np.array([0, 1, 2, 3, 4, 5, -4, -3, -2, -1])
    for t, (lab, res_, col) in enumerate(
        (("components", com, "#2a5db0"), ("neurons", neu, "#b5442d"))
    ):
        C = np.array(res_["C"])
        d = (np.arange(10)[None, :] - np.arange(10)[:, None]) % 10
        m = d != 0
        axs[0].scatter(shift[d[m]] + (t - 0.5) * 0.15, C[m], s=10, color=col, label=lab, alpha=0.7)
    axs[0].axhline(0, color="#ccc", lw=0.6)
    axs[0].set_xticks(range(-4, 6))
    axs[0].set_xlabel("Δ = r′ − r")
    axs[0].set_ylabel("cos(Π_D μ_r, Π_D μ_r′)")
    axs[0].set_title("residue means: every pair (dots), by Δ")
    axs[0].legend(frameon=False, fontsize=8)
    # (2) eigenvalues of the fitted shift
    for t, (lab, res_, col) in enumerate(
        (("components", com, "#2a5db0"), ("neurons", neu, "#b5442d"))
    ):
        ang = np.array(res_["shift_eig_angle_x10_over_2pi"])
        ab = np.array(res_["shift_eig_abs"])
        axs[1].scatter(ang + (t - 0.5) * 0.06, ab, color=col, s=25, label=lab)
    for k in range(0, 6):
        axs[1].axvline(k, color="#ddd", lw=0.6)
    axs[1].axhline(1, color="#888", ls="--", lw=0.7)
    axs[1].set_ylim(0.9, 1.02)
    axs[1].set_xlabel("eigenvalue angle × 10 / 2π (ideal: k = 1..5)")
    axs[1].set_ylabel("|eigenvalue|")
    axs[1].set_title("eigenvalues of the fitted shift S (y_r → y_{r+1})")
    axs[1].legend(frameon=False, fontsize=8)
    # (3) readers: cos(q_i, q_j) and cos(S^Δ q_i, q_j) by Delta
    for t, (lab, res_, col) in enumerate(
        (("components", com, "#2a5db0"), ("clean neurons", neu, "#b5442d"))
    ):
        pr = res_["pairs"]
        dd = np.array([shift[p[2]] for p in pr])
        raw = np.array([p[3] for p in pr])
        tr = np.array([p[4] for p in pr])
        axs[2].scatter(dd + (t - 0.5) * 0.2 - 0.05, raw, s=8, color=col, alpha=0.35, marker="o")
        axs[2].scatter(
            dd + (t - 0.5) * 0.2 + 0.05,
            tr,
            s=10,
            color=col,
            alpha=0.9,
            marker="x",
            label=f"{lab}: cos(S^Δ q_i, q_j)",
        )
    axs[2].axhline(0, color="#ccc", lw=0.6)
    axs[2].set_xticks(range(-4, 6))
    axs[2].set_xlabel("Δ = r₀(j) − r₀(i)")
    axs[2].set_ylabel("cosine")
    axs[2].set_title("read directions in D: cos(q_i, q_j) (dots), cos(S^Δ q_i, q_j) (crosses)")
    axs[2].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(args.out / "fig" / "regularity.png")
    plt.close(fig)

    summ = {"components": com, "neurons": neu}
    (args.out / "regularity.json").write_text(json.dumps(summ, indent=1))
    for lab, res_ in summ.items():
        print(lab)
        for k in (
            "norms",
            "circulant_r2_offdiag",
            "circ_resid_abs_max",
            "shift_rel_err",
            "shift_singular_values",
            "shift_eig_abs",
            "shift_eig_angle_x10_over_2pi",
            "shift10_dist_from_identity",
            "plane_fit",
        ):
            v = res_[k]
            print(
                " ",
                k,
                np.round(v, 3).tolist()
                if isinstance(v, list)
                else (v if isinstance(v, dict) else round(v, 4)),
            )
        pr = res_["pairs"]
        same = [p for p in pr if p[2] == 0]
        diff = [p for p in pr if p[2] != 0]
        print("  same-residue pairs raw cos", np.round([p[3] for p in same], 2).tolist())
        print(
            "  diff-residue: raw cos range",
            round(min(p[3] for p in diff), 2),
            round(max(p[3] for p in diff), 2),
            "transported cos median",
            round(float(np.median([p[4] for p in diff])), 3),
            "min",
            round(min(p[4] for p in diff), 2),
        )
        print(
            "  diff-residue raw cos at delta3",
            np.round(sorted(p[3] for p in diff if p[2] in (3, 7)), 2).tolist()[:6],
            "...",
        )


if __name__ == "__main__":
    main()
