"""Numbers of report sections 2.1-2.2: how the copy heads route, what they copy, and why a and b
land apart at `=`.

    python -m param_decomp.arith_repr.vectors.routing   # stdout (needs qk.npz, frames, attn_patterns)

1. Routing pairs: query / key features by position, the RoPE factor <R_4 U_q[h], R_s U_k[kv]> per
   source position, attention from `=`, and the sublayer decomposition (full model, mean stream)
   of each key feature at the a / b / op positions.
2. Slot units: alive down / o comps on at exactly one operand position, per layer; L0/L1
   attention from positions a and b.
3. OV gain (model weights) of the copy heads on operand codes, lin, slot and op directions,
   relative to random directions.
4. Separation: principal cosines between the two heads' output spaces and between the planes a
   code lands in through each head; the adder units' V energy in the two output spaces."""

import ml_dtypes
import numpy as np

from param_decomp.arith_repr.autointerp.llama_weights import Weights, llama3_inv_freq
from param_decomp.arith_repr.vectors.common import (
    AUTOINTERP,
    OUT,
    RESID,
    comp_table,
    load_uv,
    name,
    period,
    point_names,
)
from param_decomp.arith_repr.vectors.load import load_reads
from param_decomp.arith_repr.vectors.qk import rope
from param_decomp.arith_repr.vectors.storage import load_frame

PAIRS = ((0, 23, 17, 6), (15, 13, 138, 48), (16, 21, 136, 5), (18, 30, 104, 95))
POS = ("BOS", "a", "op", "b", "=")


def col(comps: dict[str, np.ndarray], layer: int, kind: str, cidx: int) -> int:
    sel = (comps["layer"] == layer) & (comps["kind"] == kind) & (comps["cidx"] == cidx)
    return int(np.flatnonzero(sel)[0])


def sublayers(v: np.ndarray, mean: np.ndarray, rms: float) -> np.ndarray:
    """Contribution of the embedding and of every sublayer step to v . mean / rms."""
    return np.r_[v @ mean[0], (np.diff(mean, axis=0) @ v)] / rms


def routing(comps: dict[str, np.ndarray], st: dict[str, np.ndarray]) -> None:
    hm = np.load(OUT / "qk.npz")["hmean"]
    ap = np.load(AUTOINTERP / "attn_patterns.npy", mmap_mode="r")
    inv = llama3_inv_freq(Weights().config)
    ln1 = np.load(RESID / "norms.npz")["ln1"]
    names = ["emb", *point_names()[1:]]
    for L, H, qc, kc in PAIRS:
        q, k = col(comps, L, "q", qc), col(comps, L, "k", kc)
        _, Uq = load_uv(comps, np.array([q]))
        Vk, Uk = load_uv(comps, np.array([k]))
        uq, uk = Uq[0].reshape(32, 128)[H], Uk[0].reshape(8, 128)[H // 4]
        f = [rope(uq, 4, inv) @ rope(uk, s, inv) / np.sqrt(128) for s in range(5)]
        att = [
            np.asarray(ap[L, o * 10000 : (o + 1) * 10000 : 10, H, 4, :], np.float32).mean(0)
            for o in (0, 1)
        ]
        print(
            f"L{L}H{H}: attention from '=' add {np.round(att[0], 2)} sub {np.round(att[1], 2)} ({'/'.join(POS)})"
        )
        print(f"   q c{qc} by position {np.round(hm[q, :, 0], 1)}; k c{kc} {np.round(hm[k, :, 0], 1)}; "
              f"RoPE factor {np.round(f, 3)}")  # fmt: skip
        v = Vk[0] * ln1[L]
        for p, lab in ((0, "a"), (2, "b"), (1, "op")):
            c = sublayers(v, st["mean"][: 2 * L + 1, p, 0], float(st["rms"][2 * L, p, 0]))
            top = np.argsort(-np.abs(c))[:5]
            print(
                f"   key at {lab}: total {c.sum():+.1f}; "
                + ", ".join(f"{names[t]} {c[t]:+.1f}" for t in top)
            )


def slots(comps: dict[str, np.ndarray]) -> None:
    hm = np.load(OUT / "qk.npz")["hmean"]
    ap = np.load(AUTOINTERP / "attn_patterns.npy", mmap_mode="r")
    for L in (0, 1):
        A = np.asarray(ap[L, 0:10000:20], np.float32).mean(0)
        for p, lab in ((1, "a"), (3, "b")):
            heads = [f"H{h}->{POS[int(np.argmax(A[h, p, 1:])) + 1]} {A[h, p, 1:].max():.2f}"
                     for h in range(32) if A[h, p, 1:].max() > 0.3]  # fmt: skip
            print(f"L{L} attention from {lab}: {np.round(A[:, p].mean(0), 2)}; " + ", ".join(heads))
    for layer in range(16):
        sel = np.flatnonzero((comps["layer"] == layer) & np.isin(comps["kind"], ("down", "o")))
        ma, mb = hm[sel, 1, 0], hm[sel, 3, 0]
        a_only = sel[(np.abs(ma) > 1) & (np.abs(mb) < 0.1)]
        b_only = sel[(np.abs(mb) > 1) & (np.abs(ma) < 0.1)]
        print(
            f"L{layer} slot units: a-only {[name(comps, c) for c in a_only]} b-only {[name(comps, c) for c in b_only]}"
        )


def ov(W: Weights, L: int, h: int) -> np.ndarray:
    ln1 = np.load(RESID / "norms.npz")["ln1"]
    Wv = W.get(f"model.layers.{L}.self_attn.v_proj.weight")[(h // 4) * 128 : (h // 4 + 1) * 128]
    Wo = W.get(f"model.layers.{L}.self_attn.o_proj.weight")[:, h * 128 : (h + 1) * 128]
    return Wo @ (Wv * ln1[L][None])


def gain(M: np.ndarray, x: np.ndarray) -> float:
    xr, xi = np.real(x), np.imag(x)
    num = np.linalg.norm(M @ xr) ** 2 + np.linalg.norm(M @ xi) ** 2
    return float(np.sqrt(num / (np.linalg.norm(xr) ** 2 + np.linalg.norm(xi) ** 2)))


def plane(z: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(np.stack([np.real(z), np.imag(z)], 1))
    return Q


def pcos(za: np.ndarray, zb: np.ndarray) -> float:
    return float(np.linalg.svd(plane(za).T @ plane(zb), compute_uv=False)[0])


def ov_and_separation(comps: dict[str, np.ndarray], st: dict[str, np.ndarray]) -> None:
    assert ml_dtypes.bfloat16  # imported so numpy knows bfloat16 (safetensors read)
    W = Weights()
    fre = np.load(OUT / "frames_re.npy", mmap_mode="r")
    fim = np.load(OUT / "frames_im.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    M13, M21 = ov(W, 15, 13), ov(W, 16, 21)
    for M, L, p, line, lab in ((M13, 15, 2, 1, "b"), (M21, 16, 0, 0, "a")):
        t = 2 * L
        F = load_frame(fre, fim, st, t)
        base = float(np.median([gain(M, rng.standard_normal(4096)) for _ in range(200)]))
        mean = st["mean"][t]
        items = {f"{lab} mod {period(k)}": F[p, 0, line, k - 1] for k in (1, 2, 5, 10, 20, 50)}
        items[f"lin({lab})"] = st["slope_b" if line else "slope_a"][t, p, 0]
        items["slot"] = mean[2, 0] - mean[0, 0]
        items["op flag"] = mean[p, 0] - mean[p, 1]
        items["bulk"] = mean[p, 0]
        print(
            f"L{L} OV gain vs random: "
            + ", ".join(f"{n} {gain(M, x) / base:.1f}x" for n, x in items.items())
        )
    u13 = np.linalg.svd(M13, full_matrices=False)[0][:, :16]
    u21 = np.linalg.svd(M21, full_matrices=False)[0][:, :16]
    print(
        "output spaces H13 vs H21, principal cos:",
        np.round(np.linalg.svd(u13.T @ u21, compute_uv=False)[:6], 2),
    )
    Fb = load_frame(fre, fim, st, 30)[2, 0, 1]
    Fa = load_frame(fre, fim, st, 32)[0, 0, 0]
    for k in (1, 2, 5, 10, 20):
        i = k - 1
        print(f"k{k}: source a vs b {pcos(Fa[i], Fb[i]):.2f}; H13(b) vs H21(a) {pcos(M13 @ Fb[i], M21 @ Fa[i]):.2f}; "
              f"b through H13 vs H21 {pcos(M13 @ Fb[i], M21 @ Fb[i]):.2f}")  # fmt: skip
    # the adders' reads of the two output spaces
    ln2 = np.load(RESID / "norms.npz")["ln2"]
    r = load_reads()
    Q13 = np.linalg.qr(W.get("model.layers.15.self_attn.o_proj.weight")[:, 13 * 128 : 14 * 128])[0]
    Q21 = np.linalg.qr(W.get("model.layers.16.self_attn.o_proj.weight")[:, 21 * 128 : 22 * 128])[0]
    for layer, units in ((16, (10, 29, 43, 106, 76, 94)), (17, (52, 39, 49, 86, 11)),
                     (18, (23, 12, 21, 22, 27, 26, 24, 32, 88, 18))):  # fmt: skip
        for kind in ("gate", "up"):
            cols = np.array([col(comps, layer, kind, u) for u in units])
            V, _ = load_uv(comps, cols)
            Vm = np.stack(V) * ln2[layer][None]
            e13 = ((Vm @ Q13) ** 2).sum(1) / (Vm**2).sum(1)
            e21 = ((Vm @ Q21) ** 2).sum(1) / (Vm**2).sum(1)
            ea, eb = r.energy[cols, 3, 0, 0].sum(-1), r.energy[cols, 3, 0, 1].sum(-1)
            print(f"L{layer} {kind} {list(units)}: V in H13 space {np.median(e13):.2f}, H21 {np.median(e21):.2f}; "
                  f"a-share {np.round(ea / (ea + eb), 2)}")  # fmt: skip


def main() -> None:
    comps = comp_table()
    st = dict(np.load(OUT / "frames_stats.npz"))
    routing(comps, st)
    slots(comps)
    ov_and_separation(comps, st)


if __name__ == "__main__":
    main()
