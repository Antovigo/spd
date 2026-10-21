"""Is the block-0 MLP's action on the a mod 10 representation a rotation (plus scale)? Position 1, alive-only.

Uses the rebuilt block 0 of mlp0_mod10.py (fields.npz). Nothing is centred.

Notation (a = integer operand 1..400; r = a mod 10 in 0..9; k = integer frequency 0..5):
  r_mid(a), x(a), m(a)   residual entering the MLP, MLP input (normed), MLP output (R^4096)
  D_x                    9-dim span of the frequency-1..5 Fourier planes of x's residue means
  C                      the 10 mod-10 readers (u31 left out)
  m_C(a)                 = m(a) - m^{-C}(a), where m^{-C} is the MLP output when the readers in C read
                         x - Pi_{D_x} x instead of x: the part of m carried by C reading D_x
  mu_r, v_r, w_r         residue means (over the 40 a with a mod 10 = r) of r_mid, m, m_C
  M_in = [mu_r]_r        10 x 4096; outputs M_out = [mu_r + v_r] (whole block) or [mu_r + w_r] (C path)
  G = M M^T              10 x 10 Gram matrix; F the unitary 10-point DFT, F[k, r] = exp(-2 pi i k r/10)/sqrt(10)
A map of the ten residue points is a rotation times a scale s exactly when G_out = s^2 G_in.
Gains g_j: square roots of the eigenvalues of G_in^{-1/2} G_out G_in^{-1/2} (the singular values of the
unique linear map of span{mu_r} that sends each mu_r to its output); all equal <=> rotation x scale.

    python mlp0_rotation.py
"""

import json

import numpy as np
from mlp0_mod10 import MOD10, OUT, SRC, mlp
from mod10_L0 import A

R = A % 10


def class_means(X: np.ndarray) -> np.ndarray:
    return np.stack([X[r == R].mean(0) for r in range(10)])


def r2_by_residue(X: np.ndarray) -> float:
    """Uncentred share of sum_a ||X(a)||^2 reproduced by X's residue means."""
    return float(1 - ((X - class_means(X)[R]) ** 2).sum() / (X**2).sum())


def gram_report(Min: np.ndarray, Mout: np.ndarray) -> dict:
    Gi, Go = Min @ Min.T, Mout @ Mout.T
    w, Q = np.linalg.eigh(Gi)
    Gi_mh = Q @ np.diag(w**-0.5) @ Q.T
    gains = np.sqrt(np.clip(np.linalg.eigvalsh(Gi_mh @ Go @ Gi_mh), 0, None))[::-1]
    s2 = float((Gi * Go).sum() / (Gi * Gi).sum())  # least-squares scale^2
    Fm = np.exp(-2j * np.pi * np.outer(np.arange(10), np.arange(10)) / 10) / np.sqrt(10)
    Fi, Fo = Fm @ Gi @ Fm.conj().T, Fm @ Go @ Fm.conj().T
    off = lambda H: float((np.abs(H - np.diag(np.diag(H))) ** 2).sum() / (np.abs(H) ** 2).sum())  # noqa: E731
    lam_i, lam_o = np.diag(Fi).real, np.diag(Fo).real
    # best rotation x scale (orthogonal Procrustes inside the joint span), relative residual
    u, sv, vt = np.linalg.svd(Mout.T @ Min, full_matrices=False)
    Rot = u @ vt
    s = sv.sum() / (Min**2).sum()
    proc = float(np.linalg.norm(Mout - s * Min @ Rot.T) / np.linalg.norm(Mout))
    return {
        "gains_sorted": gains.tolist(),
        "gain_spread_max_over_min": float(gains.max() / gains.min()),
        "scale_from_gram": float(np.sqrt(s2)),
        "gram_rel_err_vs_scaled_input": float(np.linalg.norm(Go - s2 * Gi) / np.linalg.norm(Go)),
        "procrustes_scale": float(s),
        "procrustes_rel_residual": proc,
        "fourier_gain_per_k": {k: float(np.sqrt(lam_o[k] / lam_i[k])) for k in range(6)},
        "fourier_energy_in": {
            k: float(lam_i[k] + (lam_i[10 - k] if 0 < k < 5 else 0)) for k in range(6)
        },
        "fourier_energy_out": {
            k: float(lam_o[k] + (lam_o[10 - k] if 0 < k < 5 else 0)) for k in range(6)
        },
        "offdiag_share_in": off(Fi),
        "offdiag_share_out": off(Fo),
        "rank_out": int(np.linalg.matrix_rank(Mout, tol=1e-6 * np.linalg.norm(Mout))),
    }


def main() -> None:
    f = np.load(OUT / "fields.npz")
    r_mid, x, m, Dx = (f[k].astype(np.float64) for k in ("r_mid", "x", "m", "Dx"))
    comps = {k: v.astype(np.float64) for k, v in np.load(SRC / "comps.npz").items() if k[0] in "UV"}
    z = np.load(SRC / "comps.npz")
    names = [f"{kd}.{int(i)}" for kd, i in zip(z["kind"], z["index"], strict=True) if kd != "down"]
    widx = {int(i): j for j, i in enumerate(z["index"][z["kind"] == "down"])}
    ten = np.asarray([names.index(n) for n in MOD10])
    m_full, _, _, td_full = mlp(x, comps)
    assert np.allclose(m_full, m)
    m_noC, _, _, td_noC = mlp(x, comps, Dx, ten)
    mC = m - m_noC
    res: dict = {}
    Min = class_means(r_mid)
    res["block"] = gram_report(Min, Min + class_means(m))
    res["C_path_with_residual"] = gram_report(Min, Min + class_means(mC))
    res["C_path_alone_vs_input"] = gram_report(Min, class_means(mC))
    res["energy"] = {
        "r_mid": float((r_mid**2).sum()), "m": float((m**2).sum()), "m_C": float((mC**2).sum()),
        "m_rest": float(((m - mC) ** 2).sum()),
    }  # fmt: skip
    res["residue_r2"] = {"r_mid": r2_by_residue(r_mid), "x": r2_by_residue(x), "m": r2_by_residue(m),
                         "m_C": r2_by_residue(mC), "m_rest": r2_by_residue(m - mC),
                         "x_in_Dx": r2_by_residue(x @ Dx)}  # fmt: skip
    # per reader: single-reader removal, the writers it moves, and the residue profile of that move
    per = {}
    sum_single = np.zeros_like(m)
    for c, name in zip(ten, MOD10, strict=True):
        mc, _, _, tdc = mlp(x, comps, Dx, np.asarray([c]))
        sum_single += m - mc
        dt = class_means(td_full - tdc)  # (10, n_writers): residue profile of each writer's change
        big = np.argsort(-np.abs(dt).max(0))[:3]
        main = big[0]
        prof = dt[:, main]
        r0 = int(np.argmax(np.abs(prof)))
        per[name] = {
            "main_writer": int(z["index"][z["kind"] == "down"][main]),
            "profile_main_writer": prof.tolist(),
            "residue": r0,
            "side_over_main": float(np.abs(np.delete(prof, r0)).max() / abs(prof[r0])),
            "share_of_change_energy_main_writer": float(
                (dt[:, main] ** 2).sum()
                * (comps["U_down"][main] ** 2).sum()
                / (class_means(m - mc) ** 2).sum()  # approximate (writers not orthogonal)
            ),
            "next_writers": [int(z["index"][z["kind"] == "down"][j]) for j in big[1:]],
        }
    res["per_reader"] = per
    res["additivity_rel_err"] = float(np.linalg.norm(sum_single - mC) / np.linalg.norm(mC))
    del widx
    (OUT / "rotation.json").write_text(json.dumps(res, indent=1))
    for key in ("block", "C_path_with_residual", "C_path_alone_vs_input"):
        r = res[key]
        print(key, "gains", np.round(r["gains_sorted"], 3), "spread", round(r["gain_spread_max_over_min"], 2),
              "gram_err", round(r["gram_rel_err_vs_scaled_input"], 3), "procrustes", round(r["procrustes_rel_residual"], 3),
              "\n   fourier gain", {k: round(v, 3) for k, v in r["fourier_gain_per_k"].items()},
              "offdiag in/out", round(r["offdiag_share_in"], 4), round(r["offdiag_share_out"], 4))  # fmt: skip
    print(
        "energy",
        res["energy"],
        "\nresidue_r2",
        res["residue_r2"],
        "\nadditivity",
        res["additivity_rel_err"],
    )
    for n, p in per.items():
        print(n, p["main_writer"], p["residue"], round(p["side_over_main"], 2), p["next_writers"],
              np.round(np.asarray(p["profile_main_writer"]) / np.abs(p["profile_main_writer"]).max(), 2))  # fmt: skip


if __name__ == "__main__":
    main()
