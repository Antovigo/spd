"""Read and write directions of the core flip components against the quantity codes of the residual
stream, in the alive-only model.

    python -m param_decomp.arith_repr.vectors.flip_dirs   # CPU, stdout + OUT/flip/dirs.json

Codes. From the base alive-only run's group sums at `=` (eval_alive_base.npz), at a stream point x:
* op flag d = xbar_add - xbar_sub (the op-conditional means);
* the code of quantity q in {a, b, a + b, a - b}, op o, harmonic k = 1..50: the complex vector
  F_{q,o}(k) = (1/100) sum_v ybar_o(v) e^{-2 pi i k v / 100}, ybar_o(v) the mean stream over the 100
  prompts with q = v (v = 1..100 for a and b, the residue mod 100 for a + b and a - b), centred per
  op. Its plane is spanned by Re F and Im F; the quantity's class means trace an ellipse in it;
* the flip plane of b at k: O(k) = (F_{b,add}(k) - F_{b,sub}(k)) / 2.
Points: IN = the stream entering L15's MLP (x15); OUT = the stream entering L16's MLP (x16), i.e. after
L15's MLP and L16's attention (where the adder reads).

Alignment of a direction u with a plane (orthonormal basis e1, e2): sqrt((u.e1)^2 + (u.e2)^2) / |u|; a
random direction gives about sqrt(2 / 4096) = 0.022 (`gain` = alignment / 0.022). Where the read or
write peaks along the quantity: the value v* at which Re(F . u e^{2 pi i k v / 100}) is largest.

Per component (L15 gate c0, c72; up c34, c117; down c16, c4, and the other flip writers down c21, c19,
c35, c72, c67, c64):
* gate / up: the read direction r = V * ln2 (norm gain folded) against the IN codes: cosine with the op
  flag, the best planes, the share of |r|^2 inside the span of all codes; the neurons its U drives,
  weighted by how much a unit change of its inner moves the neuron's act on the op where it is
  active (original model's L15 activations at `=`), and the down components reading those neurons;
* down: the neurons its V reads; its write U against the OUT codes (op flag, best planes, flip
  planes, share inside the span); the L16-L18 gate / up readers with the largest |U . r_reader|, and
  what each of those readers reads (its best plane in the stream entering its own MLP)."""

import json
from typing import Any

import numpy as np

from param_decomp.arith_repr.vectors.common import AUTOINTERP, DATASET, RESID, comp_table
from param_decomp.arith_repr.vectors.flip_components import DIR
from param_decomp.arith_repr.vectors.flip_model import FLIP
from param_decomp.arith_repr.vectors.flip_set import _site_order

L = 15
GROUPS = ("b", "a", "a+b", "a-b")  # row blocks of the 800 group columns
SHIFT = {"b": 1, "a": 1, "a+b": 0, "a-b": 0}  # value of row 0 of each block
RAND = np.sqrt(2 / 4096)
SWITCHES = ("gate.c0", "gate.c72", "up.c34", "up.c117")
WRITERS = (
    "down.c16",
    "down.c4",
    "down.c21",
    "down.c19",
    "down.c35",
    "down.c72",
    "down.c67",
    "down.c64",
)


def codes(x: np.ndarray) -> tuple[np.ndarray, dict[tuple[str, int], np.ndarray]]:
    """x (800, d) group sums -> op flag d (d,) and F[(q, o)] (50, d) complex codes."""
    y = x.reshape(4, 2, 100, -1) / 100.0
    mean = y[0].mean(1)  # (2, d)
    F = {}
    k = np.arange(1, 51)
    for g, q in enumerate(GROUPS):
        for o in (0, 1):
            yc = y[g, o] - mean[o]
            f = np.fft.fft(yc, axis=0)[1:51] / 100  # row i <-> value i + SHIFT
            F[(q, o)] = f * np.exp(-2j * np.pi * k * SHIFT[q] / 100)[:, None]
    return mean[0] - mean[1], F


def plane(z: np.ndarray) -> np.ndarray:
    """(2, d) orthonormal basis of span(Re z, Im z)."""
    e1 = np.real(z) / np.linalg.norm(np.real(z))
    s = np.imag(z) - (np.imag(z) @ e1) * e1
    return np.stack([e1, s / max(np.linalg.norm(s), 1e-12)])


def peak(F: np.ndarray, u: np.ndarray, k: int) -> float:
    c = F @ u  # complex: the signal along u is Re(c e^{2 pi i k v / 100}) * 2
    return float(np.mod(-np.angle(c) * 100 / (2 * np.pi * k), 100 / k))


def span_basis(d: np.ndarray, F: dict[tuple[str, int], np.ndarray]) -> np.ndarray:
    """(d, r) orthonormal basis of the span of the op flag and every code plane. The add and sub codes
    nearly coincide, so the span is taken from an SVD of the unit basis vectors, keeping singular
    values above 0.1 (a QR would add spurious directions from nearly dependent columns)."""
    cols = [d / np.linalg.norm(d)] + [b for f in F.values() for z in f for b in plane(z)]
    Uu, S, _ = np.linalg.svd(np.stack(cols, 1), full_matrices=False)
    return Uu[:, S > 0.1]


def profile(u: np.ndarray, d: np.ndarray, F: dict[tuple[str, int], np.ndarray], Q: np.ndarray,
            n_top: int = 5, extra: dict[str, np.ndarray] | None = None) -> dict[str, Any]:  # fmt: skip
    """Alignment of u with the op flag, the best code planes (any q, op, k), and the share of |u|^2 in
    the span Q of all planes and the op flag."""
    un = u / np.linalg.norm(u)
    rows = []
    for (q, o), f in F.items():
        for k in range(1, 51):
            pb = plane(f[k - 1])
            a = float(np.linalg.norm(pb @ un))
            rows.append((a, q, o, k, peak(f[k - 1], un, k)))
    for nm, z in (extra or {}).items():
        for k in range(1, 51):
            pb = plane(z[k - 1])
            rows.append((float(np.linalg.norm(pb @ un)), nm, -1, k, float("nan")))
    rows.sort(reverse=True)
    span = float(np.linalg.norm(Q.T @ un) ** 2)
    return {
        "cos op flag": float(un @ d / np.linalg.norm(d)),
        "span share": span,
        "top": [
            (
                round(a, 3),
                round(a / RAND, 1),
                q,
                ("add", "sub", "-")[o],
                k,
                100 // np.gcd(k, 100),
                None if np.isnan(p) else round(p, 1),
            )
            for a, q, o, k, p in rows[:n_top]
        ],  # fmt: skip
    }


def fmt(p: dict[str, Any]) -> str:
    top = "; ".join(f"{q} {o} k{k} (T{T}) {a:.2f} ({g:.0f}x)" + ("" if pk is None else f" @{pk}")
                    for a, g, q, o, k, T, pk in p["top"])  # fmt: skip
    return f"cos(op flag) {p['cos op flag']:+.2f}, span {p['span share']:.2f} | {top}"


def main() -> None:
    z = np.load(DIR / "eval_alive_base.npz")
    d_in, F_in = codes(z["x15"])
    Q_in = span_basis(d_in, F_in)
    at = {li: codes(z[f"x{li}"]) for li in (16, 17, 18)}  # streams entering the L16-L18 MLPs
    Q_at = {li: span_basis(*at[li]) for li in at}
    d_out, F_out = at[16]
    flip_out = {"flip b": (F_out[("b", 0)] - F_out[("b", 1)]) / 2}
    print(f"span dims: IN {Q_in.shape[1]}, L16 {Q_at[16].shape[1]}", flush=True)
    uv = np.load(AUTOINTERP / "uv_alive.npz")
    ln2 = np.load(RESID / "norms.npz")["ln2"]
    comps = comp_table()

    def vec(layer: int, kind: str) -> tuple[list[str], np.ndarray, np.ndarray]:
        suffix = {"gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}[kind]
        key = f"layers.{layer}.{suffix}"
        order = _site_order(layer, suffix, uv[key + ".ids"])
        lcols = np.flatnonzero(comps["layer"] == layer)
        sel = np.flatnonzero(comps["site"][lcols] == key)
        names = [f"c{int(comps['cidx'][lcols[s]])}" for s in sel]
        return (
            names,
            uv[key + ".V"][:, order].astype(np.float64),
            uv[key + ".U"][order].astype(np.float64),
        )

    sites = {k: vec(L, k) for k in ("gate", "up", "down")}
    readers = []  # (name, read direction) of the L16-L18 gate / up components
    for layer in (16, 17, 18):
        for kind in ("gate", "up"):
            names, Vr, _ = vec(layer, kind)
            for j, nm in enumerate(names):
                readers.append((f"L{layer}.{kind}.{nm}", Vr[:, j] * ln2[layer], layer))
    Rmat = np.stack([r for _, r, _ in readers], 1)  # (d, n_readers)
    # neuron sensitivities on the original model at `=`: d act / d g = silu'(g) u, d act / d u = silu(g)
    g_all = np.asarray(
        np.load(DATASET / "original/mlp_gate.npy", mmap_mode="r")[L, :, 4], np.float64
    )
    u_all = np.asarray(np.load(DATASET / "original/mlp_up.npy", mmap_mode="r")[L, :, 4], np.float64)
    sig = 1 / (1 + np.exp(-g_all))
    dsilu = sig * (1 + g_all * (1 - sig))
    sens = {"gate": [np.abs(dsilu * u_all)[o * 10000 : (o + 1) * 10000].mean(0) for o in (0, 1)],
            "up": [np.abs(g_all * sig)[o * 10000 : (o + 1) * 10000].mean(0) for o in (0, 1)]}  # fmt: skip
    h_mean = {}
    for kind in ("gate", "up"):
        names = sites[kind][0]
        H = z[f"h{L}.{kind}"][4, :200] / 100.0  # (200, n) group means by (op, b)
        h_mean[kind] = {nm: (H[:100, j].mean(), H[100:, j].mean()) for j, nm in enumerate(names)}
    _, Vd, _ = sites["down"]
    dnames = sites["down"][0]
    out: dict[str, Any] = {}

    print(
        "codes: IN = stream entering L15's MLP at `=`, OUT = stream entering L16's MLP at `=` (alive-only model)"
    )
    print(f"|op flag| IN {np.linalg.norm(d_in):.2f}, OUT {np.linalg.norm(d_out):.2f}\n")
    for nm in SWITCHES:
        kind, c = nm.split(".")
        names, V, U = sites[kind]
        j = names.index(c)
        r = V[:, j] * ln2[L]
        p = profile(r, d_in, F_in, Q_in)
        m_add, m_sub = h_mean[kind][c]
        o = 0 if abs(m_add) >= abs(m_sub) else 1
        drive = U[j] * sens[kind][o]  # effect on each neuron's act per unit inner, on op o
        topn = np.argsort(-np.abs(drive))[:6]
        dshare = np.abs(drive[FLIP]).sum() / np.abs(drive).sum()
        via = (drive[:, None] * Vd).sum(0)  # (n_down,) effect on each down component's inner
        topd = np.argsort(-np.abs(via))[:6]
        print(f"L15.{nm}: inner mean add {m_add:+.2f} sub {m_sub:+.2f}")
        print(f"   READ  {fmt(p)}")
        print(f"   DRIVES neurons ({('add', 'sub')[o]}): " + ", ".join(f"{int(n)}{'*' if n in FLIP else ''} {drive[n]:+.3f}" for n in topn)
              + f"; |drive| share on the 7 flip neurons {dshare:.3f}")  # fmt: skip
        print("   -> down comps: " + ", ".join(f"{dnames[i]} {via[i]:+.3f}" for i in topd))
        out[nm] = {"read": p, "inner_mean": [m_add, m_sub], "top_neurons": [int(n) for n in topn],
                   "flip_share": float(dshare), "down": [(dnames[i], float(via[i])) for i in topd]}  # fmt: skip
    print()
    _, _, Ud = sites["down"]
    for nm in WRITERS:
        _, c = nm.split(".")
        j = dnames.index(c)
        v = Vd[:, j]
        topn = np.argsort(-(v**2))[:4]
        p = profile(Ud[j], d_out, F_out, Q_at[16], extra=flip_out)
        rd = Ud[j] @ Rmat
        topr = np.argsort(-np.abs(rd))[:5]
        print(f"L15.{nm}: reads neurons " + ", ".join(f"{int(n)}{'*' if n in FLIP else ''} {v[n] ** 2 / (v**2).sum():.2f}" for n in topn))  # fmt: skip
        print(f"   WRITE {fmt(p)}")
        rtxt = []
        for i in topr:
            li_r = readers[i][2]
            rp = profile(readers[i][1], at[li_r][0], at[li_r][1], Q_at[li_r], n_top=1)
            _, g, q, o, k, T, _ = rp["top"][0]
            rtxt.append(f"{readers[i][0]} {rd[i]:+.2f} (reads {q} {o} k{k} T{T} {g:.0f}x)")
        print("   READ BY " + "; ".join(rtxt))
        out[nm] = {"write": p, "readers": [(readers[i][0], float(rd[i])) for i in topr]}
    (DIR / "dirs.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
