"""Track the a mod 10 representation at position 1 through blocks 0..LAST of the alive-only model.

Nothing is centred: zero is the activation-space zero. Input: track_build.py's stream.npz.

Notation (a = integer operand 1..400; r = a mod 10; k = integer frequency 1..5; i = imaginary unit):
  X(a)        a residual-stream field (R^4096) at one sublayer boundary; S a sublayer (attention or MLP of a
              block), Y = S(X) its output, added to the stream
  Psi_k^X     = sum_r mu_r exp(-2 pi i k r / 10), mu_r the mean of X over the 40 a with a mod 10 = r
  P_k^X       orthonormal basis of span{Re Psi_k^X, Im Psi_k^X} (a line for k = 5); D^X = span of P_1..P_5 (9 dims)
  z(a)        = D^T X(a), the 9 coordinates of X(a) in D^X (D an orthonormal basis)
  Delta(a)    = S(X)(a) - S(X - Pi_D X)(a): what S writes because its input contains D^X; S(X - Pi_D X) is
              what S writes without D^X (its period-10 content is generated from other features of a)
  r2(Y, Yh)   = 1 - sum_a ||Y - Yh||^2 / sum_a ||Y||^2   (uncentred)
Per sublayer:
  linear fit  Delta ~= L z with one 4096 x 9 matrix L (least squares, no intercept): r2_lin
  mechanism   MLP: per reader c, dt_c = t_c(X) - t_c(X - Pi_D X), Delta_c = m with only reader c's input
              changed, kappa_c the least-squares secant of Delta_c on dt_c; r2 of Delta ~= sum_c kappa_c dt_c
              attention: value path Delta_val = out(pattern of X, values of X - Pi_D X) vs Delta
Per boundary: amplitude ||Psi_k||, roundness (minor/major axis of the residue ellipse), principal cosines
of P_k with the previous boundary and with the embedding.

    python track_analyze.py
"""

import json

import numpy as np
from mod10_L0 import A
from track_build import LAST, OUT, Block, load_blocks

R = A % 10


def r2(Y: np.ndarray, Yh: np.ndarray) -> float:
    return float(1 - ((Y - Yh) ** 2).sum() / (Y**2).sum())


def psi(X: np.ndarray) -> np.ndarray:
    mu = np.stack([X[r == R].mean(0) for r in range(10)])
    return np.stack([np.exp(-2j * np.pi * k * np.arange(10) / 10) @ mu for k in range(1, 6)])


def plane(p: np.ndarray) -> np.ndarray:
    u, s, _ = np.linalg.svd(np.stack([p.real, p.imag], 1), full_matrices=False)
    return u[:, s > 1e-9 * s.max()]


def basis(X: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    P = [plane(p) for p in psi(X)]
    return P, np.linalg.qr(np.concatenate(P, 1))[0]


def cosines(P: np.ndarray, Q: np.ndarray) -> list[float]:
    return np.linalg.svd(P.T @ Q, compute_uv=False).tolist()


def boundary(X: np.ndarray, prev: list[np.ndarray] | None, emb: list[np.ndarray]) -> dict:
    ps, (P, D) = psi(X), basis(X)
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
        "D_share_of_energy": float(((X @ D) ** 2).sum() / (X**2).sum()),
        "mean_norm": float(np.linalg.norm(X, axis=1).mean()),
        "cos_emb": [cosines(P[k], emb[k]) for k in range(5)],
    }
    if prev is not None:
        out["cos_prev"] = [cosines(P[k], prev[k]) for k in range(5)]
    return out


def sublayer_common(
    X: np.ndarray, Y: np.ndarray, Yd: np.ndarray, D: np.ndarray, P: list[np.ndarray]
) -> tuple[dict, np.ndarray]:
    delta = Y - Yd
    z = X @ D
    L, *_ = np.linalg.lstsq(z, delta, rcond=None)
    pin, pd, py, pyd = psi(X), psi(delta), psi(Y), psi(Yd)
    return {
        "out_gain": [float(np.linalg.norm(py[k]) / np.linalg.norm(pin[k])) for k in range(5)],
        "out_share_from_D": [
            float((np.vdot(py[k], pd[k]) / np.vdot(py[k], py[k])).real) for k in range(5)
        ],
        "out_amp_without_D_over_out": [
            float(np.linalg.norm(pyd[k]) / np.linalg.norm(py[k])) for k in range(5)
        ],
        "r2_lin": r2(delta, z @ L),
        "delta_over_out": float(np.sqrt((delta**2).sum() / (Y**2).sum())),
        "delta_over_input_D": float(np.sqrt((delta**2).sum() / ((X @ D) ** 2).sum())),
        "fourier_gain": [float(np.linalg.norm(pd[k]) / np.linalg.norm(pin[k])) for k in range(5)],
        "beta_re": [
            float((np.vdot(pin[k], pd[k]) / np.vdot(pin[k], pin[k])).real) for k in range(5)
        ],
        "in_input_plane": [
            float(
                (
                    np.linalg.norm(P[k].T @ pd[k].real) ** 2
                    + np.linalg.norm(P[k].T @ pd[k].imag) ** 2
                )
                / np.linalg.norm(pd[k]) ** 2
            )
            for k in range(5)
        ],  # fmt: skip
    }, delta


def main_residue(v: np.ndarray) -> tuple[int, float]:
    e = np.asarray([(v[r == R] ** 2).sum() for r in range(10)])
    return int(e.argmax()), float(e.max() / e.sum())


def mlp_mechanism(blk: Block, X: np.ndarray, Xd: np.ndarray, delta: np.ndarray) -> dict:
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
        en = float((dc**2).sum())
        if (dt**2).sum() == 0:
            continue
        kap = (dc * dt[:, None]).sum(0) / (dt**2).sum()
        approx += np.outer(dt, kap)
        ssum += dc
        r0, share = main_residue(dt)
        rows.append({"reader": names[j], "energy": en, "r2_secant": r2(dc, np.outer(dt, kap)) if en > 0 else None,
                     "main_residue": r0, "onehot": share})  # fmt: skip
    rows.sort(key=lambda d: -d["energy"])
    tot = sum(d["energy"] for d in rows)
    cum, top = 0.0, []
    for d in rows:
        top.append(d)
        cum += d["energy"]
        if cum >= 0.9 * tot:
            break
    return {"r2_sum_single": r2(delta, ssum), "r2_secant_sum": r2(delta, approx), "n_readers": len(rows),
            "top90": top}  # fmt: skip


def attn_mechanism(
    blk: Block, bos: np.ndarray, X: np.ndarray, Xd: np.ndarray, delta: np.ndarray
) -> dict:
    p0, v0, v1 = blk.attn_parts(bos, X)
    p0d, _, v1d = blk.attn_parts(bos, Xd)
    full = blk.attn_out(p0, v0, v1)
    dval = full - blk.attn_out(p0, v0, v1d)
    per_head = []
    for h in range(p0.shape[1]):
        m = np.zeros_like(p0)
        m[:, h] = 1
        hv = blk.attn_out(p0 * m, v0 * m[..., None], v1 * m[..., None]) - blk.attn_out(
            p0 * m, v0 * m[..., None], v1d * m[..., None]
        )
        per_head.append(float((hv**2).sum()))
    order = np.argsort(per_head)[::-1][:4]
    return {
        "r2_value_path": r2(delta, dval) if (delta**2).sum() > 0 else None,
        "pattern_change_max": float(np.abs(p0 - p0d).max()),
        "p_self_mean": float((1 - p0).mean()),
        "top_heads": [[int(h), per_head[h] / max(sum(per_head), 1e-30)] for h in order],
    }


def main() -> None:
    st = np.load(OUT / "stream.npz")
    blocks, _, _ = load_blocks()
    emb = basis(st["pre_0"].astype(np.float64))[0]
    res: dict = {"boundaries": {}, "blocks": {}}
    prev = None
    for b in range(LAST + 2):
        for tag in ("pre", "mid") if b <= LAST else ("pre",):
            X = st[f"{tag}_{b}"].astype(np.float64)
            res["boundaries"][f"{tag}_{b}"] = boundary(X, prev, emb)
            prev = basis(X)[0]
    for blk in blocks:
        b = blk.b
        out: dict = {}
        for name, Xk, bosk in (
            ("attn", f"pre_{b}", f"bos_pre_{b}"),
            ("mlp", f"mid_{b}", f"bos_mid_{b}"),
        ):
            X, bos = st[Xk].astype(np.float64), st[bosk].astype(np.float64)
            P, D = basis(X)
            Xd = X - (X @ D) @ D.T
            f = (lambda U, blk=blk, bos=bos: blk.attn(bos, U)) if name == "attn" else blk.mlp
            Y, Yd = f(X), f(Xd)
            common, delta = sublayer_common(X, Y, Yd, D, P)
            mech = (
                attn_mechanism(blk, bos, X, Xd, delta)
                if name == "attn"
                else mlp_mechanism(blk, X, Xd, delta)
            )
            out[name] = common | {"mechanism": mech}
            print(b, name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in common.items()
                            if k in ("r2_lin", "delta_over_out", "delta_over_input_D")},
                  {k: v for k, v in mech.items() if k != "top90" and k != "top_heads"}, flush=True)  # fmt: skip
        res["blocks"][b] = out
        (OUT / "analysis.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
