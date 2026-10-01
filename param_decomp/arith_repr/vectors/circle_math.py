"""How the (a + b) mod 50 circle after the L18 MLP is assembled from the stream the MLP reads and the
MLP's write, in the plane of that circle (addition prompts, position '=').

    python -m param_decomp.arith_repr.vectors.circle_math   # -> stdout + OUT/circle_math.npz

Notation. a, b: operands, integers 1..100; s = (a + b) mod 50. theta_q = 2 pi q / 50 for an integer q.
x_in: the stream after L18 attention; w: the L18 MLP's write; x_out = x_in + w. e1, e2: orthonormal
basis of the stream's (a + b) mod 50 plane after L18 (from the frames). For a vector v, its point
in the plane is the complex number z = (v . e1) + i (v . e2), i the imaginary unit, prompt mean
removed.

1. Exact orthogonal split of any z(a, b) on the full grid:
       z = A(a) + B(b) + S(s) + R(a, b)
   with A, B, S the class means of z by a, by b and by s (each centred) and R the rest. On the
   balanced grid these four parts are orthogonal (a function of s averages to zero over b at fixed
   a, and vice versa). For each part P, |P_out|^2 = |P_in|^2 + |P_w|^2 + 2 <P_in, P_w>, where
   |P|^2 = mean over prompts of |P|^2 and <P, Q> = mean of Re(P conj(Q)).
2. How the write acts on the operand parts: the best 2 x 2 real matrix M_B with B_w(b) ~ M_B B_in(b)
   over the 100 values of b (likewise M_A), and its share of B_w's variance explained.
3. Period-50 waves: for q in {a, b, a + b, a - b}, z's coefficients on exp(+i theta_q) (turning with
   q) and exp(-i theta_q) (turning against it): c+ = mean z exp(-i theta_q), c- = mean z exp(+i theta_q)."""

from typing import Any

import ml_dtypes
import numpy as np

from param_decomp.arith_repr.autointerp.llama_weights import Weights
from param_decomp.arith_repr.vectors.case_k2 import FILE as K2
from param_decomp.arith_repr.vectors.case_k2 import plane
from param_decomp.arith_repr.vectors.common import DATASET, OUT, point_names

L, T = 18, 50
i_ = np.arange(10000)
A_, B_ = i_ // 100 + 1, i_ % 100 + 1
S_ = (A_ + B_) % T


def cmeans(z: np.ndarray, g: np.ndarray, n: int) -> np.ndarray:
    m = np.zeros(n, complex)
    np.add.at(m, g, z)
    return m / np.bincount(g, minlength=n)


def split(z: np.ndarray) -> dict[str, np.ndarray]:
    z = z - z.mean()
    A = cmeans(z, A_ - 1, 100)[A_ - 1]
    B = cmeans(z, B_ - 1, 100)[B_ - 1]
    S = cmeans(z, S_, T)[S_]
    return {"A": A, "B": B, "S": S, "R": z - A - B - S, "all": z}


def ip(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.mean(np.real(p * np.conj(q))))


def fit_map(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares 2 x 2 real M with dst ~ M src (columns: points as (x, y)); returns M and R^2."""
    X = np.stack([np.real(src), np.imag(src)], 1)
    Y = np.stack([np.real(dst), np.imag(dst)], 1)
    M = np.linalg.lstsq(X, Y, rcond=None)[0].T
    r2 = 1 - ((Y - X @ M.T) ** 2).sum() / (Y**2).sum()
    return M, float(r2)


def polar(M: np.ndarray) -> str:
    """M = R(phi) S with R a rotation, S symmetric: report phi and S's eigenvalues."""
    U, sv, Vt = np.linalg.svd(M)
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        return f"includes a reflection; singular values {np.round(sv, 3)}"
    S = Vt.T @ np.diag(sv) @ Vt
    phi = np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0]))
    return f"rotation {phi:+.0f} deg, then stretch with eigenvalues {np.round(np.linalg.eigvalsh(S), 3)}"


def who(act: np.ndarray, PW: np.ndarray, parts: dict[str, dict[str, np.ndarray]], top: np.ndarray,
        out: dict[str, Any]) -> None:  # fmt: skip
    """4. Per neuron n (all 14,336): its share of the circle, <S_n, S_w> / |S_w|^2, and its share of
    the opposition to the stream's operand content, <A_n, A_in> / <A_w, A_in> (likewise B), where
    A_n, B_n, S_n are the parts of neuron n's own write in the plane. Each set of shares sums to 1."""
    n_neur = act.shape[1]
    Ain, Bin, Sw = parts["in"]["A"], parts["in"]["B"], parts["write"]["S"]
    a_in = cmeans(Ain, A_ - 1, 100)  # (100,) class means
    b_in = cmeans(Bin, B_ - 1, 100)
    s_w = cmeans(Sw, S_, T)
    shS, shA, shB = (np.zeros(n_neur) for _ in range(3))
    for c in range(0, n_neur, 1024):
        zc = (
            act[:, c : c + 1024] * PW[0, c : c + 1024]
            + 1j * act[:, c : c + 1024] * PW[1, c : c + 1024]
        )
        zc = zc - zc.mean(0)
        g = zc.reshape(100, 100, -1)
        An = g.mean(1)  # (100 a values, n)
        Bn = g.mean(0)  # (100 b values, n)
        Sn = np.stack([zc[q == S_].mean(0) for q in range(T)])  # (50, n)
        # balanced grid: <P_n, P_ref> over prompts = mean over class values of the class means
        shA[c : c + 1024] = np.mean(np.real(An * np.conj(a_in)[:, None]), 0)
        shB[c : c + 1024] = np.mean(np.real(Bn * np.conj(b_in)[:, None]), 0)
        shS[c : c + 1024] = np.mean(np.real(Sn * np.conj(s_w)[:, None]), 0)
    shA, shB, shS = shA / shA.sum(), shB / shB.sum(), shS / shS.sum()
    print("\n4. who does what (shares sum to 1 over all 14,336 neurons)")
    print(f"   the seven circle neurons {list(map(int, top))}:")
    print(
        f"      circle {shS[top].sum():.3f} | opposition to a {shA[top].sum():.3f} | to b {shB[top].sum():.3f}"
    )
    for n in top:
        print(f"      n{n}: circle {shS[n]:+.3f}  a {shA[n]:+.3f}  b {shB[n]:+.3f}")
    for nm, sh in (("a", shA), ("b", shB)):
        order = np.argsort(-sh)
        k = int(np.searchsorted(np.cumsum(sh[order]), 0.8)) + 1
        print(f"   opposition to {nm}: top 10 {sh[order[:10]].sum():.3f}, {k} neurons reach 0.8; top 10 = "
              + ", ".join(f"n{i} {sh[i]:.3f} (circle {shS[i]:+.3f})" for i in order[:10]))  # fmt: skip
    others = np.setdiff1d(np.arange(n_neur), top)
    print(
        f"   all other neurons: circle {shS[others].sum():.3f}, a {shA[others].sum():.3f}, b {shB[others].sum():.3f}"
    )
    out.update(share_circle=shS, share_a=shA, share_b=shB)


def main() -> None:
    assert ml_dtypes.bfloat16
    z7 = dict(np.load(K2))
    top = z7["top"]
    names = point_names()
    t_in, t_out = names.index(f"L{L}.attn"), names.index(f"L{L}.mlp")
    fre = np.load(OUT / "frames_re.npy", mmap_mode="r")
    fim = np.load(OUT / "frames_im.npy", mmap_mode="r")
    E = plane(
        fre[t_out, 3, 0, 2, 1].astype(np.float32) + 1j * fim[t_out, 3, 0, 2, 1].astype(np.float32)
    )
    resid = np.load(DATASET / "original/resid.npy", mmap_mode="r")
    x_in = np.asarray(resid[t_in, :10000, 4], np.float32) @ E.T
    x_out = np.asarray(resid[t_out, :10000, 4], np.float32) @ E.T
    Wd = Weights().get(f"model.layers.{L}.mlp.down_proj.weight").astype(np.float32)
    g = np.asarray(
        np.load(DATASET / "original/mlp_gate.npy", mmap_mode="r")[L, :10000, 4], np.float32
    )
    u = np.asarray(
        np.load(DATASET / "original/mlp_up.npy", mmap_mode="r")[L, :10000, 4], np.float32
    )
    act = g / (1 + np.exp(-g)) * u
    del g, u
    PW = E @ Wd  # (2, 14336): every neuron's write in the plane
    w_all = act @ PW.T
    w7 = act[:, top] @ PW[:, top].T
    cz = lambda y: y[:, 0] + 1j * y[:, 1]  # noqa: E731
    Z = {
        "in": cz(x_in),
        "write": cz(w_all),
        "write7": cz(w7),
        "out": cz(x_out),
        "in+write7": cz(x_in + w7),
    }
    parts = {k: split(v) for k, v in Z.items()}
    tot_out = ip(parts["out"]["all"], parts["out"]["all"])
    print("check: out - (in + write), relative:",
          np.sqrt(ip(parts["out"]["all"] - parts["in"]["all"] - parts["write"]["all"],
                     parts["out"]["all"] - parts["in"]["all"] - parts["write"]["all"]) / tot_out))  # fmt: skip
    print("\n1. parts (squared size, as a fraction of x_out's total in the plane)")
    for k in Z:
        print(
            f"   {k:10s} "
            + "  ".join(
                f"{p} {ip(parts[k][p], parts[k][p]) / tot_out:6.3f}"
                for p in ("A", "B", "S", "R", "all")
            )
        )
    for w in ("write", "write7"):
        print(
            f"   cosine between the stream's part and the {w}'s part (-1 = the write cancels it):"
        )
        print("      " + "  ".join(
            f"{p} {ip(parts['in'][p], parts[w][p]) / np.sqrt(ip(parts['in'][p], parts['in'][p]) * ip(parts[w][p], parts[w][p])):+.3f}"
            for p in ("A", "B", "S", "R")))  # fmt: skip
    print("\n2. the write's operand parts as a 2 x 2 map of the stream's")
    out: dict[str, Any] = {}
    for w in ("write", "write7"):
        for p, g_, n in (("A", A_ - 1, 100), ("B", B_ - 1, 100)):
            src = cmeans(parts["in"][p], g_, n)
            dst = cmeans(parts[w][p], g_, n)
            M, r2 = fit_map(src, dst)
            print(f"   {w:7s} {p}: M = {np.round(M, 3).tolist()}, R^2 {r2:.3f}; {polar(M)}")
            out[f"M_{w}_{p}"] = M
    print("\n3. period-50 waves of z (c+ turns with q, c- against it): |c|, angle in degrees")
    th = lambda q: 2 * np.pi * q / T  # noqa: E731
    Q = {"a": A_, "b": B_, "a+b": A_ + B_, "a-b": A_ - B_}
    for k in ("in", "write", "write7", "out"):
        zz = parts[k]["all"]
        cells = []
        for qn, q in Q.items():
            cp = np.mean(zz * np.exp(-1j * th(q)))
            cm = np.mean(zz * np.exp(1j * th(q)))
            cells.append(
                f"{qn}: +{abs(cp):.3f}@{np.degrees(np.angle(cp)):+4.0f} -{abs(cm):.3f}@{np.degrees(np.angle(cm)):+4.0f}"
            )
            out[f"c_{k}_{qn}"] = np.array([cp, cm])
        print(f"   {k:7s} " + " | ".join(cells))
    print("   (scale: rms size of x_out in the plane =", f"{np.sqrt(tot_out):.3f})")
    who(act, PW, parts, top, out)
    np.savez(OUT / "circle_math.npz", **out)


if __name__ == "__main__":
    main()
