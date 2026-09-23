"""How the L15 MLP mirrors b on subtraction (numbers of report section 5, "How the L15 MLP builds the mirror").

    python -m param_decomp.arith_repr.vectors.mirror   # prints the tables; needs frames + mirror/L15_neurons.npz

1. Reflection centre per harmonic: R_sub = lambda conj(R_add) over the L16-L19 gate/up readers,
   b -> c - b with c = -arg(lambda) 100 / (2 pi k).
2. Reader-space geometry: E = (F_add + F_sub)/2, O = (F_add - F_sub)/2 of b's code at `=`, with only
   the L15H13 copy (stream before the L15 MLP) and the L15 MLP's write, seen through the L16-L18 gate/up
   V (norm weight folded). Mirror <=> E on one axis e (cos), O on one axis w ⟂ e (sin).
3. Neurons: the odd write along w, split over all L15 neurons (W_down act_b), and each neuron's odd
   coefficient split into A = gate-mean switch x up's b-code and B = up-mean sign x silu(g)'s b-code."""

import numpy as np

from param_decomp.arith_repr.vectors.common import OUT, RESID, comp_table, load_uv, peak_value
from param_decomp.arith_repr.vectors.load import load_reads
from param_decomp.arith_repr.vectors.storage import load_frame

KS = (1, 2, 3, 10, 20)


def axis(z: np.ndarray) -> np.ndarray:
    """Real unit direction carrying most of a complex vector's power."""
    _, _, vt = np.linalg.svd(np.stack([np.real(z), np.imag(z)]), full_matrices=False)
    return vt[0]


def one_axis(z: np.ndarray, a: np.ndarray) -> float:
    return float(abs(a @ z) ** 2 / (np.abs(z) ** 2).sum())


def reflection_centres(comps: dict[str, np.ndarray]) -> None:
    r = load_reads()
    for layer in (16, 17, 18, 19):
        sel = np.flatnonzero((comps["layer"] == layer) & np.isin(comps["kind"], ("gate", "up")))
        row = []
        for k in (1, 2, 3, 4, 5, 10, 20, 25):
            a, s = r.R[sel, 3, 0, 1, k - 1], r.R[sel, 3, 1, 1, k - 1]
            lam = (s * a).sum() / (np.abs(a) ** 2).sum()
            c = np.mod(-np.angle(lam) * 100 / (2 * np.pi * k), 100 / k)
            row.append(f"k{k} c={c:.2f}/{100 / k:.0f}")
        print(f"L{layer} reflection centres:", " | ".join(row))


def geometry(comps: dict[str, np.ndarray], F31: np.ndarray, F32: np.ndarray) -> None:
    ln2 = np.load(RESID / "norms.npz")["ln2"]
    for layer in (16, 17, 18):
        cols = np.flatnonzero((comps["layer"] == layer) & np.isin(comps["kind"], ("gate", "up")))
        V, _ = load_uv(comps, cols)
        Vm = np.stack(V) * ln2[layer][None]
        for k in KS:
            i = k - 1
            P = Vm @ ((F31[0, i] + F31[1, i]) / 2)
            Ma, Ms = Vm @ (F32[0, i] - F31[0, i]), Vm @ (F32[1, i] - F31[1, i])
            E, Od = P + (Ma + Ms) / 2, (Ma - Ms) / 2
            e, w = axis(E), axis(Od)
            Pp = P - e * (e @ P)
            MEp = (Ma + Ms) / 2 - e * (e @ ((Ma + Ms) / 2))
            canc = (MEp * np.conj(Pp)).sum().real / (np.abs(Pp) ** 2).sum()
            Ra, Rs = P + Ma, P + Ms
            den = np.linalg.norm(Ra) * np.linalg.norm(Rs)
            print(
                f"L{layer} k{k:2d}: E one-axis {one_axis(E, e):.2f} (copy alone {one_axis(P, axis(P)):.2f}), "
                f"MLP even write on copy off-axis {canc:+.2f}, O one-axis {one_axis(Od, w):.2f}, "
                f"|cos(e,w)| {abs(e @ w):.2f}, phase(O)-phase(E) {np.degrees(np.angle((w @ Od) / (e @ E))):+.0f}, "
                f"|O|/|E| {abs(w @ Od) / abs(e @ E):.2f}, same/refl "
                f"{abs((Rs * np.conj(Ra)).sum()) / den:.2f}/{abs((Rs * Ra).sum()) / den:.2f}"
            )


def neurons(comps: dict[str, np.ndarray]) -> None:
    z = dict(np.load(OUT / "mirror/L15_neurons.npz"))
    Wd = z["W_down"]
    ln2 = np.load(RESID / "norms.npz")["ln2"]
    cols = np.flatnonzero((comps["layer"] == 16) & np.isin(comps["kind"], ("gate", "up")))
    V, _ = load_uv(comps, cols)
    VW = (np.stack(V) * ln2[16][None]) @ Wd
    dcols = np.flatnonzero((comps["layer"] == 15) & (comps["kind"] == "down"))
    Vd, _ = load_uv(comps, dcols)
    Vdm = np.stack(Vd, 1)
    sm = [z[f"s_mean_o{o}"] for o in (0, 1)]
    um = [z[f"u_mean_o{o}"] for o in (0, 1)]
    for k in (2, 10, 20):
        i = k - 1
        act = [z[f"act_b_o{o}"][i] for o in (0, 1)]
        On = (act[0] - act[1]) / 2
        Or = VW @ On
        w = axis(Or)
        wO = w @ Or
        wn = w @ VW
        contrib = (wn * On * np.conj(wO)).real / abs(wO) ** 2
        ub = [z[f"u_b_o{o}"][i] for o in (0, 1)]
        sb = [z[f"s_b_o{o}"][i] for o in (0, 1)]
        A = (sm[0] * ub[0] - sm[1] * ub[1]) / 2
        B = (um[0] * sb[0] - um[1] * sb[1]) / 2

        sa, sb_ = (float(((wn * X * np.conj(wO)).real / abs(wO) ** 2).sum()) for X in (A, B))
        print(f"k{k}: odd write mode A {sa:+.2f}, mode B {sb_:+.2f}; "
              f"top-3 neurons {np.sort(contrib)[::-1][:3].sum():.2f}")  # fmt: skip
        for n in np.argsort(-contrib)[:3]:
            d = dcols[int(np.argmax(np.abs(Vdm[n])))]
            print(
                f"   neuron {n}: share {contrib[n]:+.2f}; g mean add/sub {z['g_mean_o0'][n]:+.2f}/"
                f"{z['g_mean_o1'][n]:+.2f}, u mean {um[0][n]:+.2f}/{um[1][n]:+.2f}; b read peak: up "
                f"{peak_value(z['u_b_o0'][i][n], k):.1f}, gate {peak_value(z['g_b_o0'][i][n], k):.1f} "
                f"(period {100 / np.gcd(k, 100):.0f}); read by down c{comps['cidx'][d]}"
            )


def main() -> None:
    comps = comp_table()
    fre = np.load(OUT / "frames_re.npy", mmap_mode="r")
    fim = np.load(OUT / "frames_im.npy", mmap_mode="r")
    st = dict(np.load(OUT / "frames_stats.npz"))
    F31 = load_frame(fre, fim, st, 31)[3][:, 1]  # b line at '=' before the L15 MLP
    F32 = load_frame(fre, fim, st, 32)[3][:, 1]  # after it
    reflection_centres(comps)
    geometry(comps, F31, F32)
    neurons(comps)


if __name__ == "__main__":
    main()
