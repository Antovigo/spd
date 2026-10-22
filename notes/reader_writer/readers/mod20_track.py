"""Track the period-20 representation of a at position 1 through blocks 0..LAST, in the alive-only model
(components, track_build.py) or in the original Llama-3.1-8B (neurons, track_build_orig.py).

Nothing is centred: zero is the activation-space zero (the network has no biases). Input: the model's stream.npz.

Notation (a = integer operand 1..400; i = imaginary unit):
  s           = a mod 20, integer 0..19 (20 values of a per residue); r = a mod 10, integer 0..9
  h           = floor(a / 10) mod 2, the parity of the tens (0 for s in 0..9, 1 for s in 10..19)
  g           = floor((a + 5) / 10) mod 2, the parity of a rounded to the nearest ten (0 for s in 0..4 and 15..19)
  Q           = floor(s / 5), integer 0..3, the quarter of the 20-cycle; (h, g) = (0,0), (0,1), (1,1), (1,0) for Q = 0..3
  X(a)        residual stream (R^4096) at one boundary (pre_l: entering block l; mid_l: after its attention add)
  mu_s        mean of X over the 20 a with a mod 20 = s
  Psi_j       = sum_s mu_s exp(-2 pi i j s / 20), j = 1..10 (frequency j turns j times per 20; its exact period is
              20 / gcd(j, 20); odd j are the frequencies that differ between s and s + 10)
  null_j      99th percentile of ||Psi_j|| over 100 random relabellings of the 400 a (seed 0)
  P           orthonormal basis (4096 x 2) of span{Re Psi_1, Im Psi_1}: the frequency-1 plane of period 20
  z(a)        = P^T X(a) in R^2;  m_s = P^T mu_s
  gap_s       = ||m_{s+1} - m_s|| (s + 1 taken mod 20): spacing of consecutive residue means along the ring;
              gap ratio = mean of the four gaps at the quarter edges (s = 4, 9, 14, 19) / mean of the other 16
  eta_v       uncentred share of sum_a ||z(a)||^2 explained by the class means of variable v (s, Q, h or g)
  sep(s, s')  ||c_s - c_s'|| / sqrt((sp_s^2 + sp_s'^2) / 2) with c_s the mean of z over residue s and sp_s the
              root-mean-square distance of those z to c_s; also for the 4 quarters
  acc_*       two-fold cross-validated nearest-class-mean accuracies in the plane (plane and means fitted on
              a <= 200, tested on a > 200, and the reverse): s (20 classes), Q (4), and h given r (choose between the
              means of s = r and s = r + 10)
  probe_h     two-fold cross-validated accuracy of an L2 logistic regression (C = 1, inputs divided by one global
              scale) predicting h from: the whole stream X, z alone, or X - Pi_P X (the stream without the plane); the
              plane used for z and for the removal is fitted on the training half only
Per sublayer S (attention or MLP of one block, including its RMSNorm; it adds S(X) to the stream):
  Delta(a)    = S(X)(a) - S(X - Pi_P X)(a): what S writes because its input contains the plane
  gain        ||Psi_1(S(X))|| / ||Psi_1(X)||;  share_from_P = Re <Psi_1(S(X)), Psi_1(Delta)> / ||Psi_1(S(X))||^2;
              without_P = ||Psi_1(S(X - Pi_P X))|| / ||Psi_1(S(X))|| (period-20 output made from other features)
  r2(Y, Yh)   = 1 - sum_a ||Y - Yh||^2 / sum_a ||Y||^2 (uncentred)
  r2_lin      two-fold cross-validated r2 of Delta ~= L z (one 4096 x 2 matrix, least squares, no intercept)
  r2_gated    the same with one matrix L_r per r = a mod 10: Delta ~= L_{a mod 10} z
  map         odd parts o_s = (m_s - m_{s+10}) / 2, s = 0..9, at the input (in P) and at the output boundary (in its own
              plane P'); r2_map is the r2 of o_out ~= o_in M with one 2 x 2 matrix M
  MLP readers c (components: kept gate / up components; neurons: rows of gate_proj / up_proj):
    dt_c      = t_c(X) - t_c(X - Pi_P X): the change of reader c's activation when the plane is removed
    Delta_c   the change of the MLP output when only reader c's activation changes that way
    kappa_c   least-squares secant of Delta_c on dt_c (in R^4096); kappa_{c,r} the same within each r = a mod 10
    secant r2 r2 of Delta ~= sum_c kappa_c dt_c; gated secant r2 the same with kappa_{c, a mod 10};
              additivity r2: r2 of Delta ~= sum_c Delta_c
    resp_c    uncentred r2 of t_c(a) ~= t_c restricted to the plane: reader c applied to RMSNorm-gain * Pi_P X(a) / rms(X(a))
    j1_c      share of sum_a t_c(a)^2 in frequency j = 1 of s: 2 |sum_s m_c(s) exp(-2 pi i s / 20)|^2 / sum_a t_c^2,
              m_c(s) the mean of t_c over the 20 a with a mod 20 = s

    python mod20_track.py --model alive|orig
"""

import argparse
import json

import numpy as np
from mod10_L0 import RUN, A
from sklearn.linear_model import LogisticRegression
from track_build import LAST, silu

S20 = A % 20
R10 = A % 10
H = (A // 10) % 2
G = ((A + 5) // 10) % 2
Q = S20 // 5
HALF = A <= 200
OUT = RUN / "analysis/reader_writer/track_mod20"
EDGES = (4, 9, 14, 19)


def r2(Y: np.ndarray, Yh: np.ndarray) -> float:
    return float(1 - ((Y - Yh) ** 2).sum() / (Y**2).sum())


def means(X: np.ndarray, lab: np.ndarray, n: int) -> np.ndarray:
    return np.stack([X[lab == q].mean(0) for q in range(n)])


def psi(X: np.ndarray, j: int, sel: np.ndarray | None = None) -> np.ndarray:
    sel = np.ones(len(X), bool) if sel is None else sel
    mu = means(X[sel], S20[sel], 20)
    return np.exp(-2j * np.pi * j * np.arange(20) / 20) @ mu


def plane(X: np.ndarray, sel: np.ndarray | None = None) -> np.ndarray:
    p = psi(X, 1, sel)
    u, s, _ = np.linalg.svd(np.stack([p.real, p.imag], 1), full_matrices=False)
    return u


def cosines(P: np.ndarray, Q_: np.ndarray) -> list[float]:
    return np.linalg.svd(P.T @ Q_, compute_uv=False).tolist()


def seps(z: np.ndarray, lab: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    c = means(z, lab, n)
    sp = np.asarray([np.sqrt(((z[lab == q] - c[q]) ** 2).sum(1).mean()) for q in range(n)])
    S = np.linalg.norm(c[:, None] - c[None], axis=-1) / np.sqrt(
        (sp[:, None] ** 2 + sp[None] ** 2) / 2
    )
    np.fill_diagonal(S, np.inf)
    return S, sp


def probe(X: np.ndarray, y: np.ndarray, part: str) -> float:
    """part: "stream" (X), "plane" (z) or "without_plane" (X - Pi_P X); P fitted on the training half only."""
    ok = 0
    for tr, te in ((HALF, ~HALF), (~HALF, HALF)):
        P = plane(X, tr)
        F = {"stream": X, "plane": X @ P, "without_plane": X - (X @ P) @ P.T}[part]
        F = F / F[tr].std()
        clf = LogisticRegression(C=1.0, max_iter=5000).fit(F[tr], y[tr])
        ok += int((clf.predict(F[te]) == y[te]).sum())
    return ok / len(y)


def boundary(X: np.ndarray, rng: np.random.Generator) -> dict:
    amp = [float(np.linalg.norm(psi(X, j))) for j in range(1, 11)]
    nulls = []
    for _ in range(100):
        lab = rng.permutation(S20)
        mu = means(X, lab, 20)
        nulls.append(
            [
                np.linalg.norm(np.exp(-2j * np.pi * j * np.arange(20) / 20) @ mu)
                for j in range(1, 11)
            ]
        )
    null = np.quantile(np.asarray(nulls), 0.99, axis=0).tolist()
    P = plane(X)
    z = X @ P
    m = means(z, S20, 20)
    gaps = np.linalg.norm(np.roll(m, -1, 0) - m, axis=1)
    edge = np.isin(np.arange(20), EDGES)
    tot = (z**2).sum()

    def eta(lab: np.ndarray, n: int) -> float:
        return float(1 - ((z - means(z, lab, n)[lab]) ** 2).sum() / tot)

    S, sp = seps(z, S20, 20)
    SQ, _ = seps(z, Q, 4)
    anti = [float(S[s, s + 10]) for s in range(10)]
    acc = {"s": 0, "Q": 0, "h_given_r": 0}
    for tr, te in ((HALF, ~HALF), (~HALF, HALF)):
        Pt = plane(X, tr)
        zt, ze = X[tr] @ Pt, X[te] @ Pt
        cs, cq = means(zt, S20[tr], 20), means(zt, Q[tr], 4)
        d = ((ze[:, None] - cs[None]) ** 2).sum(-1)
        acc["s"] += int((d.argmin(1) == S20[te]).sum())
        acc["Q"] += int((((ze[:, None] - cq[None]) ** 2).sum(-1).argmin(1) == Q[te]).sum())
        r = R10[te]
        acc["h_given_r"] += int(
            ((d[np.arange(len(r)), r + 10] < d[np.arange(len(r)), r]) == (H[te] == 1)).sum()
        )
    pe = np.stack([psi(X, 1).real, psi(X, 1).imag], 1)
    sv = np.linalg.svd(pe, compute_uv=False)
    return {
        "amp": amp,
        "null99": null,
        "roundness": float(sv[1] / sv[0]),
        "gaps": gaps.tolist(),
        "gap_ratio": float(gaps[edge].mean() / gaps[~edge].mean()),
        "residue_means_plane": m.tolist(),
        "eta": {"s": eta(S20, 20), "Q": eta(Q, 4), "h": eta(H, 2), "g": eta(G, 2)},
        "sep_min": float(S.min()),
        "sep_min_pair": [int(x) for x in np.unravel_index(np.argmin(S), S.shape)],
        "sep_median": float(np.median(S[np.isfinite(S)])),
        "sep_antipodal": anti,
        "sep_quarters_min": float(SQ.min()),
        "spread_mean": float(sp.mean()),
        "radius_mean": float(np.linalg.norm(means(z, S20, 20), axis=1).mean()),
        "offset_norm": float(np.linalg.norm(z.mean(0))),
        "acc": {k: v / 400 for k, v in acc.items()},
        "probe_h": {k: probe(X, H, k) for k in ("stream", "plane", "without_plane")},
        "_P": P,
    }


def cv_fit(Delta: np.ndarray, F: np.ndarray) -> float:
    num = den = 0.0
    for tr, te in ((HALF, ~HALF), (~HALF, HALF)):
        L, *_ = np.linalg.lstsq(F[tr], Delta[tr], rcond=None)
        num += ((Delta[te] - F[te] @ L) ** 2).sum()
        den += (Delta[te] ** 2).sum()
    return float(1 - num / den)


def odd_part(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    m = means(X @ P, S20, 20)
    return (m[:10] - m[10:]) / 2


def sublayer_common(
    X: np.ndarray, Y: np.ndarray, Yd: np.ndarray, P: np.ndarray, P_out: np.ndarray
) -> tuple[dict, np.ndarray]:
    delta = Y - Yd
    z = X @ P
    zg = np.concatenate([z * (r == R10)[:, None] for r in range(10)], 1)
    py, pd, pyd, px = psi(Y, 1), psi(delta, 1), psi(Yd, 1), psi(X, 1)
    o_in, o_out = odd_part(X, P), odd_part(X + Y, P_out)
    M, *_ = np.linalg.lstsq(o_in, o_out, rcond=None)
    w = means(Y, S20, 20)
    return {
        "gain": float(np.linalg.norm(py) / np.linalg.norm(px)),
        "share_from_P": float((np.vdot(py, pd) / np.vdot(py, py)).real),
        "without_P": float(np.linalg.norm(pyd) / np.linalg.norm(py)),
        "delta_over_input_plane": float(np.sqrt((delta**2).sum() / (z**2).sum())),
        "r2_lin": cv_fit(delta, z),
        "r2_gated": cv_fit(delta, zg),
        "r2_map": r2(o_out, o_in @ M),
        "write_in_plane": (w @ P).tolist(),
        "write_in_out_plane": (w @ P_out).tolist(),
    }, delta


def j1_share(T: np.ndarray) -> np.ndarray:
    m = means(T, S20, 20)
    M1 = np.exp(-2j * np.pi * np.arange(20) / 20) @ m
    return 2 * np.abs(M1) ** 2 / (T**2).sum(0)


def mech_neurons(blk, X: np.ndarray, Xd: np.ndarray, P: np.ndarray, delta: np.ndarray) -> dict:
    """Readers are neurons: gate row i changes only silu(g_i) u_i, up row i only silu(g_i) u_i; both write along
    down_proj column i, so Delta_c = s_c(a) w_i with a scalar s_c."""
    g, u = blk.readers(X)
    gd, ud = blk.readers(Xd)
    sg = silu(g)
    s_gate = (sg - silu(gd)) * u
    s_up = sg * (u - ud)
    out: dict = {}
    Wd = blk.Wd
    wn2 = (Wd**2).sum(0)
    xp = (
        blk.g2 * (X @ P @ P.T) / np.sqrt((X**2).mean(-1, keepdims=True) + 1e-5)
    )  # the plane's part of x
    rows = []
    approx = np.zeros_like(delta)
    gapprox = np.zeros_like(delta)
    ssum = np.zeros_like(delta)
    for name, sc, dt, Wr, T in (("g", s_gate, g - gd, blk.Wg, g), ("u", s_up, u - ud, blk.Wu, u)):
        k = (sc * dt).sum(0) / np.maximum((dt**2).sum(0), 1e-300)
        kr = np.stack(
            [
                (sc * dt)[r == R10].sum(0) / np.maximum((dt[r == R10] ** 2).sum(0), 1e-300)
                for r in range(10)
            ]
        )
        approx += (k * dt) @ Wd.T
        gapprox += (kr[R10] * dt) @ Wd.T
        ssum += sc @ Wd.T
        en = (sc**2).sum(0) * wn2
        r2_sec = 1 - ((sc - k * dt) ** 2).sum(0) / np.maximum((sc**2).sum(0), 1e-300)
        r2_gat = 1 - ((sc - kr[R10] * dt) ** 2).sum(0) / np.maximum((sc**2).sum(0), 1e-300)
        tp = xp @ Wr.T
        resp = 1 - ((T - tp) ** 2).sum(0) / (T**2).sum(0)
        j1 = j1_share(T)
        for i in range(T.shape[1]):
            rows.append((name, i, en[i], r2_sec[i], r2_gat[i], resp[i], j1[i]))
        out[f"resp_{name}"] = resp
        out[f"j1_{name}"] = j1
    return finish(rows, delta, approx, gapprox, ssum, out)


def finish(rows, delta, approx, gapprox, ssum, extra) -> dict:
    rows.sort(key=lambda d: -d[2])
    tot = sum(d[2] for d in rows)
    cum, top = 0.0, []
    for d in rows:
        top.append({"reader": f"{d[0]}{d[1]}", "share": d[2] / tot, "r2_secant": d[3], "r2_gated": d[4],
                    "resp_plane": d[5], "j1_share": d[6]})  # fmt: skip
        cum += d[2]
        if cum >= 0.9 * tot or len(top) >= 40:
            break
    n90 = 0
    cum = 0.0
    for d in rows:
        n90 += 1
        cum += d[2]
        if cum >= 0.9 * tot:
            break
    res = {"r2_secant_sum": r2(delta, approx), "r2_gated_secant_sum": r2(delta, gapprox),
           "r2_additivity": r2(delta, ssum), "n_readers": len(rows), "n_readers_90": n90, "top": [
               {k: (float(v) if not isinstance(v, str) else v) for k, v in t.items()} for t in top]}  # fmt: skip
    res["_extra"] = extra
    return res


def mech_components(blk, X: np.ndarray, Xd: np.ndarray, P: np.ndarray, delta: np.ndarray) -> dict:
    tg, tu = blk.readers(X)
    tgd, tud = blk.readers(Xd)
    hg, hu = tg @ blk.m["U_gate"], tu @ blk.m["U_up"]
    sg = silu(hg)
    Vd, Ud = blk.m["V_down"], blk.m["U_down"]
    o = sg * hu
    names = [(k[0], int(i)) for k, i in zip(blk.kind, blk.index, strict=True) if k != "down"]
    Gn = tg.shape[1]
    T = np.concatenate([tg, tu], 1)
    rmsx = np.sqrt((X**2).mean(-1, keepdims=True) + 1e-5)
    xp = blk.g2 * (X @ P @ P.T) / rmsx
    tp = np.concatenate([xp @ blk.m["V_gate"], xp @ blk.m["V_up"]], 1)
    resp = 1 - ((T - tp) ** 2).sum(0) / (T**2).sum(0)
    j1 = j1_share(T)
    rows = []
    approx, gapprox, ssum = np.zeros_like(delta), np.zeros_like(delta), np.zeros_like(delta)
    for j, (kind, idx) in enumerate(names):
        if j < Gn:
            dt = tg[:, j] - tgd[:, j]
            h2 = hg - np.outer(dt, blk.m["U_gate"][j])
            dc = ((o - silu(h2) * hu) @ Vd) @ Ud
        else:
            dt = tu[:, j - Gn] - tud[:, j - Gn]
            dc = ((sg * np.outer(dt, blk.m["U_up"][j - Gn])) @ Vd) @ Ud
        if (dt**2).sum() == 0:
            continue
        kap = (dc * dt[:, None]).sum(0) / (dt**2).sum()
        kr = np.stack(
            [(dc * dt[:, None])[r == R10].sum(0) / (dt[r == R10] ** 2).sum() for r in range(10)]
        )
        a1, a2 = np.outer(dt, kap), kr[R10] * dt[:, None]
        approx += a1
        gapprox += a2
        ssum += dc
        en = float((dc**2).sum())
        rows.append(
            (
                kind,
                idx,
                en,
                r2(dc, a1) if en > 0 else 0.0,
                r2(dc, a2) if en > 0 else 0.0,
                resp[j],
                j1[j],
            )
        )
    extra = {"resp_g": resp[:Gn], "resp_u": resp[Gn:], "j1_g": j1[:Gn], "j1_u": j1[Gn:],
             "ids_g": blk.index[blk.kind == "gate"], "ids_u": blk.index[blk.kind == "up"]}  # fmt: skip
    return finish(rows, delta, approx, gapprox, ssum, extra)


def attn_mechanism(blk, bos: np.ndarray, X: np.ndarray) -> dict:
    p0, _, _ = blk.attn_parts(bos, X)
    return {"p_self_mean": float((1 - p0).mean())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=("alive", "orig"), required=True)
    args = ap.parse_args()
    if args.model == "alive":
        from track_build import OUT as SRC
        from track_build import load_blocks

        blocks = load_blocks()[0]
        get = lambda b: blocks[b]  # noqa: E731
        mech = mech_components
    else:
        from track_build_orig import OUT as SRC
        from track_build_orig import load_block

        get = load_block
        mech = mech_neurons
    out_dir = OUT / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    st = np.load(SRC / "stream.npz")
    rng = np.random.default_rng(0)
    names = [f"{t}_{b}" for b in range(LAST + 1) for t in ("pre", "mid")] + [f"pre_{LAST + 1}"]
    res: dict = {"boundaries": {}, "sublayers": {}}
    Ps = {}
    for n in names:
        bd = boundary(st[n].astype(np.float64), rng)
        Ps[n] = bd.pop("_P")
        res["boundaries"][n] = bd
    emb = Ps["pre_0"]
    prev = None
    for n in names:
        res["boundaries"][n]["cos_emb"] = cosines(Ps[n], emb)
        if prev is not None:
            res["boundaries"][n]["cos_prev"] = cosines(Ps[n], Ps[prev])
        prev = n
        b = res["boundaries"][n]
        print(n, f"j1/null {b['amp'][0] / b['null99'][0]:.2f}", f"gap {b['gap_ratio']:.2f}",
              f"eta_s {b['eta']['s']:.2f} Q {b['eta']['Q']:.2f}", "acc", {k: round(v, 2) for k, v in b["acc"].items()},
              "probe", {k: round(v, 2) for k, v in b["probe_h"].items()}, flush=True)  # fmt: skip
    readers = {}
    for b in range(LAST + 1):
        blk = get(b)
        for kind, n0, n1 in (
            ("attn", f"pre_{b}", f"mid_{b}"),
            ("mlp", f"mid_{b}", names[names.index(f"mid_{b}") + 1]),
        ):
            X = st[n0].astype(np.float64)
            bos = st[f"bos_{n0}"].astype(np.float64)
            P = Ps[n0]
            Xd = X - (X @ P) @ P.T
            f = (lambda U, blk=blk, bos=bos: blk.attn(bos, U)) if kind == "attn" else blk.mlp
            Y, Yd = f(X), f(Xd)
            common, delta = sublayer_common(X, Y, Yd, P, Ps[n1])
            m = attn_mechanism(blk, bos, X) if kind == "attn" else mech(blk, X, Xd, P, delta)
            if kind == "mlp":
                readers[b] = m.pop("_extra")
            res["sublayers"][f"{kind}_{b}"] = common | {"mechanism": m}
            print(b, kind, {k: round(v, 3) for k, v in common.items() if isinstance(v, float)},
                  {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items() if k != "top"}, flush=True)  # fmt: skip
        del blk
        (out_dir / "analysis.json").write_text(json.dumps(res, indent=1))
    np.savez(out_dir / "planes.npz", **{n: Ps[n] for n in names})
    np.savez(
        out_dir / "readers.npz",
        **{f"L{b}_{k}": v for b, d in readers.items() for k, v in d.items()},
    )


if __name__ == "__main__":
    main()
