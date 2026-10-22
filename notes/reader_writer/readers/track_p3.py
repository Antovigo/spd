"""Track the b mod 10 representation at position 3 (the token b) through blocks 0..LAST: the position-1 analyses
(track_analyze.py, track_residues.py, track_separability.py) on track_build_p3.py's stream.

Prompts: 8,000 = op in (+, -) x a in A_SUB (ten values, one per residue) x b = 1..400. Nothing is centred.
Notation as in track_analyze.py, with the residue r = b mod 10 (each residue class has 800 prompts, over all a and op):
  Psi_k^X, P_k^X, D^X   from X's residue means over b mod 10;  Delta = S(X) - S(X - Pi_D X) (only position 3's
                        input changed: for attention, positions 0..2 keep their own stream)
Also, per boundary: the a mod 10 and op structure at position 3 (Fourier amplitudes of the means over a mod 10;
norm of the difference of the two op means), and per block the attention weight of position 3 on positions 0..3.

    python track_p3.py
"""

import json

import numpy as np
from scipy.optimize import linprog
from track_build import LAST, Block, load_blocks
from track_build_p3 import OUT, attn_last

ST = np.load(OUT / "stream.npz")
RB = ST["b"] % 10
RA = ST["a"] % 10
OP = ST["op"]
CTX = ST["ctx"]
# the per-reader mechanism runs on 1,600 prompts: a in {7, 171} (residues 7 and 1), every b, both ops
MECH = np.isin(ST["a"], (7, 171))


def r2(Y: np.ndarray, Yh: np.ndarray) -> float:
    return float(1 - ((Y - Yh) ** 2).sum() / (Y**2).sum())


def psi(X: np.ndarray, lab: np.ndarray = RB) -> np.ndarray:
    mu = np.stack([X[lab == r].mean(0) for r in range(10)])
    return np.stack([np.exp(-2j * np.pi * k * np.arange(10) / 10) @ mu for k in range(1, 6)])


def plane(p: np.ndarray) -> np.ndarray:
    u, s, _ = np.linalg.svd(np.stack([p.real, p.imag], 1), full_matrices=False)
    return u[:, s > 1e-9 * s.max()]


def basis(X: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    P = [plane(p) for p in psi(X)]
    return P, np.linalg.qr(np.concatenate(P, 1))[0]


def cos(P: np.ndarray, Q: np.ndarray) -> list[float]:
    return np.linalg.svd(P.T @ Q, compute_uv=False).tolist()


def offdiag(mu: np.ndarray) -> float:
    F = np.exp(-2j * np.pi * np.outer(np.arange(10), np.arange(10)) / 10) / np.sqrt(10)
    H = (F @ (mu @ mu.T) @ F.conj().T)[1:, 1:]
    return float((np.abs(H - np.diag(np.diag(H))) ** 2).sum() / (np.abs(H) ** 2).sum())


def clouds(z: np.ndarray) -> dict:
    c = np.stack([z[r == RB].mean(0) for r in range(10)])
    s = np.asarray([np.sqrt(((z[r == RB] - c[r]) ** 2).sum(1).mean()) for r in range(10)])
    sep = np.full((10, 10), np.inf)
    for r in range(10):
        for q in range(10):
            if q != r:
                sep[r, q] = np.linalg.norm(c[r] - c[q]) / np.sqrt((s[r] ** 2 + s[q] ** 2) / 2)
    n = np.bincount(RB, minlength=10)
    loo = c[RB] * 0  # own-class mean without the point itself
    loo = (c[RB] * n[RB, None] - z) / (n[RB, None] - 1)
    d = ((z[:, None, :] - c[None]) ** 2).sum(-1)
    d[np.arange(len(z)), RB] = ((z - loo) ** 2).sum(-1)
    i, j = np.unravel_index(np.argmin(sep), sep.shape)
    # spread split: how much of each cloud's spread is explained by the context (op, a)
    within_ctx = np.mean([((z[(r == RB) & (k == CTX)] - z[(r == RB) & (k == CTX)].mean(0)) ** 2).sum(1).mean()
                          for r in range(10) for k in np.unique(CTX)])  # fmt: skip
    return {
        "radius": np.linalg.norm(c, axis=1).tolist(), "spread": s.tolist(), "sep_min": float(sep.min()),
        "sep_min_pair": [int(i), int(j)], "sep_median": float(np.median(sep[np.isfinite(sep)])),
        "acc_loo": float((d.argmin(1) == RB).mean()),
        "spread_share_within_context": float(within_ctx / np.mean(s**2)),
    }  # fmt: skip


def separable(z: np.ndarray, y: np.ndarray, offset: bool) -> bool:
    Z = np.concatenate([z, np.ones((len(z), 1))], 1) if offset else z
    res = linprog(np.zeros(Z.shape[1]), A_ub=-(y[:, None] * Z), b_ub=-np.ones(len(z)),
                  bounds=[(None, None)] * Z.shape[1], method="highs")  # fmt: skip
    return res.status == 0


def boundary(X: np.ndarray, prev: list[np.ndarray] | None, emb: list[np.ndarray]) -> dict:
    ps, (P, D) = psi(X), basis(X)
    mu = np.stack([X[r == RB].mean(0) for r in range(10)])
    u = np.linalg.svd(mu.T, full_matrices=False)[0][:, :10]
    pa = psi(X, RA)
    out = {
        "amp": [float(np.linalg.norm(p)) for p in ps],
        "round": [
            float(
                (lambda s: s[-1] / s[0])(
                    np.linalg.svd(np.stack([p.real, p.imag], 1), compute_uv=False)
                )
            )
            for p in ps[:4]
        ],  # fmt: skip
        "offdiag": offdiag(mu),
        "D_share_of_energy": float(((X @ D) ** 2).sum() / (X**2).sum()),
        "cos_emb": [cos(P[k], emb[k]) for k in range(5)],
        "a_amp": [float(np.linalg.norm(p)) for p in pa],
        "a_in_D_share": [
            float(
                (np.linalg.norm(D.T @ p.real) ** 2 + np.linalg.norm(D.T @ p.imag) ** 2)
                / np.linalg.norm(p) ** 2
            )
            for p in pa
        ],  # fmt: skip
        "op_diff_norm": float(np.linalg.norm(X[OP == 0].mean(0) - X[OP == 1].mean(0))),
        "clouds": clouds(X @ D),
        "sep_D_affine_ovr": sum(
            separable(X @ D, np.where(r == RB, 1.0, -1.0), True) for r in range(10)
        ),
        "sep_M_zero_ovr": sum(
            separable(X @ u, np.where(r == RB, 1.0, -1.0), False) for r in range(10)
        ),
    }
    if prev is not None:
        out["cos_prev"] = [cos(P[k], prev[k]) for k in range(5)]
    return out


def common(X: np.ndarray, Y: np.ndarray, Yd: np.ndarray, D: np.ndarray) -> tuple[dict, np.ndarray]:
    delta = Y - Yd
    z = X @ D
    L, *_ = np.linalg.lstsq(z, delta, rcond=None)
    pin, pd, py = psi(X), psi(delta), psi(Y)
    return {
        "r2_lin": r2(delta, z @ L),
        "delta_over_input_D": float(np.sqrt((delta**2).sum() / ((X @ D) ** 2).sum())),
        "out_gain": [float(np.linalg.norm(py[k]) / np.linalg.norm(pin[k])) for k in range(5)],
        "out_share_from_D": [
            float((np.vdot(py[k], pd[k]) / np.vdot(py[k], py[k])).real) for k in range(5)
        ],
        "beta_re": [
            float((np.vdot(pin[k], pd[k]) / np.vdot(pin[k], pin[k])).real) for k in range(5)
        ],
    }, delta


def mlp_mech(blk: Block, X: np.ndarray, Xd: np.ndarray, delta: np.ndarray, lab: np.ndarray) -> dict:
    tg, tu = blk.readers(X)
    tgd, tud = blk.readers(Xd)
    hg, hu = tg @ blk.m["U_gate"], tu @ blk.m["U_up"]
    sg = hg / (1 + np.exp(-hg))
    Vd, Ud = blk.m["V_down"], blk.m["U_down"]
    o = sg * hu
    names = [f"{k[0]}{int(i)}" for k, i in zip(blk.kind, blk.index, strict=True) if k != "down"]
    G = tg.shape[1]
    rows, approx, ssum = [], np.zeros_like(delta), np.zeros_like(delta)
    for j in range(len(names)):
        if j < G:
            dt = tg[:, j] - tgd[:, j]
            h2 = hg - np.outer(dt, blk.m["U_gate"][j])
            dc = ((o - (h2 / (1 + np.exp(-h2))) * hu) @ Vd) @ Ud
        else:
            dt = tu[:, j - G] - tud[:, j - G]
            dc = ((sg * np.outer(dt, blk.m["U_up"][j - G])) @ Vd) @ Ud
        if (dt**2).sum() == 0:
            continue
        kap = (dc * dt[:, None]).sum(0) / (dt**2).sum()
        approx += np.outer(dt, kap)
        ssum += dc
        e = np.asarray([(dt[lab == r] ** 2).sum() for r in range(10)])
        en = float((dc**2).sum())
        rows.append({"reader": names[j], "energy": en, "r2_secant": r2(dc, np.outer(dt, kap)) if en > 0 else None,
                     "main_residue": int(e.argmax()), "onehot": float(e.max() / e.sum())})  # fmt: skip
    rows.sort(key=lambda d: -d["energy"])
    tot, cum, top = sum(d["energy"] for d in rows), 0.0, []
    for d in rows:
        top.append(d)
        cum += d["energy"]
        if cum >= 0.9 * tot:
            break
    return {"r2_secant_sum": r2(delta, approx), "r2_sum_single": r2(delta, ssum), "top90": top}


def main() -> None:
    blocks, _, _ = load_blocks()
    emb = basis(ST["pre_0"].astype(np.float64))[0]
    res: dict = {"boundaries": {}, "blocks": {}}
    prev = None
    names = [f"{t}_{b}" for b in range(LAST + 1) for t in ("pre", "mid")] + [f"pre_{LAST + 1}"]
    for n in names:
        X = ST[n].astype(np.float64)
        res["boundaries"][n] = boundary(X, prev, emb)
        prev = basis(X)[0]
        c = res["boundaries"][n]
        print(n, "amp", np.round(c["amp"], 2), "offdiag", round(c["offdiag"], 3), "a_amp", np.round(c["a_amp"], 2),
              "sep", round(c["clouds"]["sep_min"], 2), "acc", round(c["clouds"]["acc_loo"], 3),
              "ctx share", round(c["clouds"]["spread_share_within_context"], 2), "ovr", c["sep_D_affine_ovr"],
              c["sep_M_zero_ovr"], flush=True)  # fmt: skip
    for blk in blocks:
        b = blk.b
        out: dict = {}
        X = ST[f"pre_{b}"].astype(np.float64)
        ctx = ST[f"ctx_pre_{b}"].astype(np.float64)[CTX]
        _, D = basis(X)
        Xd = X - (X @ D) @ D.T
        Y, p = attn_last(blk, ctx, X)
        Yd, pd_ = attn_last(blk, ctx, Xd)
        Yv, _ = attn_last(blk, ctx, X, value_from=Xd)
        c, delta = common(X, Y, Yd, D)
        out["attn"] = c | {"r2_value_path": r2(delta, Y - Yv), "pattern_change_max": float(np.abs(p - pd_).max()),
                           "weight_by_position": p.mean((0, 1)).tolist()}  # fmt: skip
        X = ST[f"mid_{b}"].astype(np.float64)
        _, D = basis(X)
        Xd = X - (X @ D) @ D.T
        Y, Yd = blk.mlp(X), blk.mlp(Xd)
        c, delta = common(X, Y, Yd, D)
        mech = mlp_mech(blk, X[MECH], Xd[MECH], delta[MECH], RB[MECH])
        out["mlp"] = c | {
            "mechanism": mech | {"prompts": "a in {7, 171}, b = 1..400, op +/-: 1,600"}
        }
        res["blocks"][b] = out
        a, m = out["attn"], out["mlp"]
        print(b, "attn gain", np.round(a["out_gain"], 2), "fromD", np.round(a["out_share_from_D"], 2), "w_pos",
              np.round(a["weight_by_position"], 3), "lin", round(a["r2_lin"], 2), "| mlp gain", np.round(m["out_gain"], 2),
              "lin", round(m["r2_lin"], 2), "sec", round(m["mechanism"]["r2_secant_sum"], 2), "add",
              round(m["mechanism"]["r2_sum_single"], 2), flush=True)  # fmt: skip
        (OUT / "analysis.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
