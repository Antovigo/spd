"""Per-period summary of the `mlp_periods` bookkeeping: which MLP writes each result period, how,
and through which neurons / components.

    python -m param_decomp.arith_repr.vectors.periods_summary   # -> OUT/periods/summary.npz + stdout

Mechanism groups (of the terms in `mlp_periods`): X = a-code x b-code across gate and up (X_gu),
Sx = a-code x b-code inside silu(gate) (S_x), M = result x result harmonic mixing (M_gu + S_m),
P = result code already in the MLP's input, passed through (P_u + S_lin), O = the rest.
Per (op, period, layer): the write power (share of the stream's variance at `=` right after the MLP,
summed over the harmonics of the period) and the group shares (power-weighted over the harmonics)."""

import numpy as np

from param_decomp.arith_repr.vectors.common import OUT, comp_table, load_uv, peak_value, period
from param_decomp.arith_repr.vectors.load import Reads, load_reads

FAMILIES = (100, 50, 25, 20, 10, 5, 4, 2)
LAYERS = tuple(range(14, 32))
GROUPS = {"X": (2,), "Sx": (3,), "M": (4, 5), "P": (0, 1), "O": (6, 7)}
PER = np.array([period(k) for k in range(1, 51)])
W50 = np.r_[np.full(49, 2.0), 1.0]


def load(layer: int) -> dict[str, np.ndarray]:
    return dict(np.load(OUT / f"periods/L{layer}.npz"))


def table() -> tuple[np.ndarray, np.ndarray]:
    """power (2, F, L), groups (2, F, L, 5)."""
    st = np.load(OUT / "frames_stats.npz")
    power = np.zeros((2, len(FAMILIES), len(LAYERS)))
    groups = np.zeros((2, len(FAMILIES), len(LAYERS), len(GROUPS)))
    for j, layer in enumerate(LAYERS):
        z = load(layer)
        for o in range(2):
            P = (np.abs(z[f"T_o{o}"]) ** 2).sum(1) * W50 / st["tvar"][2 + 2 * layer, 3, o]
            sh = z[f"share_o{o}"]
            for f, per in enumerate(FAMILIES):
                m = per == PER
                w = P[m]
                power[o, f, j] = w.sum()
                fr = (sh[m, :8] * w[:, None]).sum(0) / max(w.sum(), 1e-12)
                groups[o, f, j] = [fr[list(ix)].sum() for ix in GROUPS.values()]
    return power, groups


def unit_of(comps: dict[str, np.ndarray], layer: int) -> tuple[np.ndarray, np.ndarray]:
    """For every neuron: the alive down comp reading it most (-1 if none reads it in its top 8)."""
    dcols = np.flatnonzero((comps["layer"] == layer) & (comps["kind"] == "down"))
    V, _ = load_uv(comps, dcols)
    Vd = np.stack(V, 1)  # (14336, n_down)
    top8 = np.argsort(-np.abs(Vd), axis=0)[:8]
    owner = np.full(Vd.shape[0], -1)
    for j in range(Vd.shape[1]):
        for n in top8[:, j]:
            if owner[n] < 0 or abs(Vd[n, j]) > abs(Vd[n, owner[n]]):
                owner[n] = j
    return owner, dcols


def feeders(comps: dict[str, np.ndarray], layer: int, n: int, k: int, o: int, reads: Reads) -> str:
    """The alive gate / up comps carrying neuron n's a- and b-line content at harmonic k."""
    r = reads
    out = []
    for kind in ("gate", "up"):
        cols = np.flatnonzero((comps["layer"] == layer) & (comps["kind"] == kind))
        _, U = load_uv(comps, cols)
        Un = np.stack(U)[:, n]
        for li, q in ((0, "a"), (1, "b")):
            c = Un * r.R[cols, 3, o, li, k - 1]
            j = int(np.argmax(np.abs(c)))
            if abs(c[j]) > 0.02:
                out.append(f"{kind} c{comps['cidx'][cols[j]]} reads {q} @{peak_value(c[j], k):.1f}")
    return "; ".join(out)


def main() -> None:
    comps = comp_table()
    power, groups = table()
    names = list(GROUPS)
    for o, op in enumerate(("add", "sub")):
        print(f"== {op}: write power x1000 / dominant mechanism")
        print("       " + " ".join(f"L{v:<6d}" for v in LAYERS))
        for f, per in enumerate(FAMILIES):
            cells = []
            for j in range(len(LAYERS)):
                g = names[int(np.argmax(groups[o, f, j]))]
                cells.append(
                    f"{power[o, f, j] * 1000:5.1f}{g:<2s}" if power[o, f, j] > 2e-4 else "   .   "
                )
            print(f"mod{per:<4d}" + " ".join(cells))
    reads = load_reads()
    # creation cells: for each (op, period) the layers where X / Sx / M carry >= 40 % of a write
    # holding >= 25 % of the period's peak write power
    for o, op in enumerate(("add", "sub")):
        for f, per in enumerate(FAMILIES):
            peak = power[o, f].max()
            for j, layer in enumerate(LAYERS):
                cre = groups[o, f, j, :3]
                if power[o, f, j] < 0.25 * peak or cre.max() < 0.4:
                    continue
                g = int(np.argmax(cre))
                z = load(layer)
                ks = np.flatnonzero(per == PER) + 1
                P = (np.abs(z[f"T_o{o}"][ks - 1]) ** 2).sum(1)
                kk = int(ks[np.argmax(P)])
                term = {0: [2], 1: [3], 2: [4, 5]}[g]
                nsh = z[f"neuron_o{o}"][:, kk - 1, term].sum(1)
                tot = z[f"share_o{o}"][kk - 1, term].sum()
                owner, dcols = unit_of(comps, layer)
                alive = nsh[owner >= 0].sum() / tot
                top = np.argsort(-nsh)[:4]
                print(f"\n{op} mod {per} L{layer}: {names[g]} {cre[g]:.2f} of a {power[o, f, j] * 1000:.1f}e-3 write "
                      f"(main k={kk}); neurons behind alive down comps carry {alive:.2f} of it")  # fmt: skip
                for n in top:
                    unit = (
                        f"down c{comps['cidx'][dcols[owner[n]]]}"
                        if owner[n] >= 0
                        else "no alive down"
                    )
                    print(
                        f"   neuron {n} ({unit}): {nsh[n] / tot:.2f} | {feeders(comps, layer, int(n), kk, o, reads)}"
                    )
                if g == 2:
                    mix = z[f"mix_gu_o{o}"][kk - 1] + z[f"mix_s_o{o}"][kk - 1]
                    pp = np.argsort(-mix)[:4]
                    print("   partners (p + (k-p)): " + ", ".join(
                        f"{p if p <= 50 else p - 100}+{(kk - p) % 100 if (kk - p) % 100 <= 50 else (kk - p) % 100 - 100}"
                        f" {mix[p] / max(mix.sum(), 1e-12):.2f}" for p in pp))  # fmt: skip
    np.savez(OUT / "periods/summary.npz", power=power, groups=groups, families=np.array(FAMILIES),
             layers=np.array(LAYERS), group_names=np.array(names))  # fmt: skip


if __name__ == "__main__":
    main()
