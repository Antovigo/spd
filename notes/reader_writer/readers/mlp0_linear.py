"""Does the block-0 MLP act LINEARLY on the period-10 representation? Position 1, alive-only model.

Closed form tested (nothing centred; zero is the activation zero):
  t_c(a) = x(a) . V_c                       inner activation of reader c (x: MLP input, a = 1..400)
  h_g,n = sum_c t_c U_g,c,n,  h_u,n = sum_c t_c U_u,c,n,  o_n = SiLU(h_g,n) h_u,n,  m = sum_n o_n omega_n
  omega_n = V_down[n, :] U_down             neuron n's write vector (R^4096)
Split the readers into C (the 10 period-10 readers, u31 left out) and the rest; m^0(a) is the MLP output
with t_c set to 0 for every c in C (all other readers unchanged), and Delta_C(a) = m(a) - m^0(a).
If each t_c is one-hot in the residue (about 0 off its residue r_c) and no two readers of C share a
residue, any function of (t_c)_{c in C} that vanishes at 0 equals sum_c kappa_c t_c on the data, so
  Delta_C(a) ~= sum_c kappa_c t_c(a) = W x(a),   W = sum_c kappa_c V_c^T   (rank <= 10)
with kappa_c in R^4096 the least-squares secant of Delta_c(a) = m(a) - m^{c->0}(a) on t_c(a).
Measured: r2 = 1 - sum_a ||Y - Yhat||^2 / sum_a ||Y||^2 (uncentred) for each step.

    python mlp0_linear.py
"""

import json

import numpy as np
from mlp0_mod10 import MOD10, OUT, SRC
from mod10_L0 import A

R = A % 10


def silu(h: np.ndarray) -> np.ndarray:
    return h / (1 + np.exp(-h))


def r2(Y: np.ndarray, Yh: np.ndarray) -> float:
    return float(1 - ((Y - Yh) ** 2).sum() / (Y**2).sum())


def fourier(X: np.ndarray) -> np.ndarray:
    mu = np.stack([X[r == R].mean(0) for r in range(10)])
    return np.stack([np.exp(-2j * np.pi * k * np.arange(10) / 10) @ mu for k in range(1, 6)])


def main() -> None:
    f = np.load(OUT / "fields.npz")
    x, r_mid, Dx = (f[k].astype(np.float64) for k in ("x", "r_mid", "Dx"))
    z = np.load(SRC / "comps.npz")
    c = {k: z[k].astype(np.float64) for k in z.files if k[0] in "UV"}
    kind, idx = z["kind"], z["index"]
    gid, uid = list(idx[kind == "gate"]), list(idx[kind == "up"])
    T = {"gate": x @ c["V_gate"], "up": x @ c["V_up"]}
    where = [
        (k, (gid if k == "gate" else uid).index(int(i))) for k, i in (n.split(".") for n in MOD10)
    ]

    def out(zero: list[tuple[str, int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tg, tu = T["gate"].copy(), T["up"].copy()
        for k, j in zero:
            (tg if k == "gate" else tu)[:, j] = 0.0
        hg, hu = tg @ c["U_gate"], tu @ c["U_up"]
        return ((silu(hg) * hu) @ c["V_down"]) @ c["U_down"], hg, hu

    m, hg, hu = out([])
    m0 = out(where)[0]
    dC = m - m0
    res: dict = {"energy": {"m": float((m**2).sum()), "Delta_C": float((dC**2).sum()),
                            "r_mid": float((r_mid**2).sum())}}  # fmt: skip
    # one-hotness of each t_c: share of sum_a t_c^2 on its main residue
    tC = np.stack([T[k][:, j] for k, j in where], 1)  # (400, 10)
    main_r = np.asarray(
        [np.argmax([np.abs(tC[r == R, i]).mean() for r in range(10)]) for i in range(10)]
    )
    res["t_main_residue"] = main_r.tolist()
    res["t_energy_on_main_residue"] = [
        float((tC[main_r[i] == R, i] ** 2).sum() / (tC[:, i] ** 2).sum()) for i in range(10)
    ]
    res["t_offres_rms_over_onres_rms"] = [
        float(np.sqrt((tC[main_r[i] != R, i] ** 2).mean() / (tC[main_r[i] == R, i] ** 2).mean()))
        for i in range(10)
    ]
    # per-reader secants
    kappa, per = [], {}
    sum_single = np.zeros_like(m)
    for i, (name, wh) in enumerate(zip(MOD10, where, strict=True)):
        d = m - out([wh])[0]
        sum_single += d
        t = tC[:, i]
        kap = (d * t[:, None]).sum(0) / (t**2).sum()
        kappa.append(kap)
        per[name] = {"r2_secant": r2(d, np.outer(t, kap)), "energy": float((d**2).sum())}
    K = np.stack(kappa, 1)  # (4096, 10)
    Vc = np.stack([c[f"V_{k}"][:, j] for k, j in where], 1)  # (4096, 10)
    W_x = tC @ K.T  # = x @ Vc @ K.T
    res["per_reader"] = per
    res["r2_DeltaC_vs_sum_single"] = r2(dC, sum_single)
    res["r2_DeltaC_vs_Wx"] = r2(dC, W_x)
    res["r2_DeltaC_vs_W_PiD_x"] = r2(dC, ((x @ Dx) @ Dx.T) @ Vc @ K.T)
    res["share_Vc_in_D"] = (
        np.linalg.norm(Dx.T @ Vc, axis=0) ** 2 / np.linalg.norm(Vc, axis=0) ** 2
    ).tolist()
    # period-10 content: Fourier components of Delta_C and of W x
    F1, F2 = fourier(dC), fourier(W_x)
    res["fourier_overlap_DeltaC_Wx"] = {
        k + 1: float((np.vdot(F1[k], F2[k]) / np.vdot(F1[k], F1[k])).real) for k in range(5)
    }
    res["fourier_amp_ratio_Wx_over_DeltaC"] = {
        k + 1: float(np.linalg.norm(F2[k]) / np.linalg.norm(F1[k])) for k in range(5)
    }
    # SiLU regime of the neurons C drives: how far from linear in h
    hgr = hg[:, np.abs(c["U_gate"][[j for k, j in where if k == "gate"]]).max(0).argsort()[-200:]]
    res["gate_h_range_top200_neurons"] = [float(hgr.min()), float(hgr.max())]
    res["silu_vs_half_h_plus_quarter_h2_maxerr_on_range"] = float(
        np.abs(
            silu(np.linspace(hgr.min(), hgr.max(), 1001))
            - (lambda h: h / 2 + h**2 / 4)(np.linspace(hgr.min(), hgr.max(), 1001))
        ).max()
    )
    # residual connection: x = gamma * r_mid / rho(a); spread of rho over a
    rho = np.sqrt((r_mid**2).mean(1) + 1e-5)
    res["rho_min_max_over_mean"] = [float(rho.min() / rho.mean()), float(rho.max() / rho.mean())]
    res["W_singular_values"] = np.linalg.svd(Vc @ K.T, compute_uv=False)[:12].tolist()
    np.savez(OUT / "linear.npz", K=K, Vc=Vc, main_r=main_r, dC=dC, W_x=W_x, tC=tC)
    (OUT / "linear.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
