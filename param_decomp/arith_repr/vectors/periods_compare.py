"""Does the per-period MLP bookkeeping survive when the MLP is rebuilt from components only?

    python -m param_decomp.arith_repr.vectors.periods_compare   # stdout + OUT/periods/compare.npz

For every (op, period, layer) cell, model vs `comp` vs `comp_mean` (see `mlp_periods`):
write power, complex cosine between the model's and the rebuilt result write (power-weighted over
the period's harmonics), the mechanism-group shares, and the overlap of the top-4 creating neurons."""

import numpy as np

from param_decomp.arith_repr.vectors.common import OUT
from param_decomp.arith_repr.vectors.periods_summary import FAMILIES, GROUPS, LAYERS, PER, W50

MODES = ("model", "comp", "comp_mean")


def load(layer: int, mode: str) -> dict[str, np.ndarray]:
    name = f"L{layer}" if mode == "model" else f"L{layer}_{mode}"
    return dict(np.load(OUT / f"periods/{name}.npz"))


def main() -> None:
    st = np.load(OUT / "frames_stats.npz")
    nf, nl = len(FAMILIES), len(LAYERS)
    power = np.zeros((3, 2, nf, nl))
    groups = np.zeros((3, 2, nf, nl, len(GROUPS)))
    cos = np.zeros((2, 2, nf, nl))  # comp / comp_mean vs model
    overlap = np.zeros((2, 2, nf, nl))
    for j, layer in enumerate(LAYERS):
        z = [load(layer, m) for m in MODES]
        for o in range(2):
            T = [zz[f"T_o{o}"] for zz in z]
            for mi in range(3):
                P = (np.abs(T[mi]) ** 2).sum(1) * W50 / st["tvar"][2 + 2 * layer, 3, o]
                sh = z[mi][f"share_o{o}"]
                for f, per in enumerate(FAMILIES):
                    m = per == PER
                    w = P[m]
                    power[mi, o, f, j] = w.sum()
                    fr = (sh[m, :8] * w[:, None]).sum(0) / max(w.sum(), 1e-12)
                    groups[mi, o, f, j] = [fr[list(ix)].sum() for ix in GROUPS.values()]
            for ci in (1, 2):
                c = np.abs((T[ci] * np.conj(T[0])).sum(1)) / (
                    np.linalg.norm(T[ci], axis=1) * np.linalg.norm(T[0], axis=1) + 1e-12
                )
                w0 = (np.abs(T[0]) ** 2).sum(1)
                for f, per in enumerate(FAMILIES):
                    m = per == PER
                    cos[ci - 1, o, f, j] = (c[m] * w0[m]).sum() / max(w0[m].sum(), 1e-12)
                    k = int(np.flatnonzero(m)[np.argmax(w0[m])])
                    tops = [
                        set(np.argsort(-z[mi][f"neuron_o{o}"][:, k, 2:6].sum(1))[:4].tolist())
                        for mi in (0, ci)
                    ]
                    overlap[ci - 1, o, f, j] = len(tops[0] & tops[1]) / 4
    names = list(GROUPS)
    for o, op in enumerate(("add", "sub")):
        print(f"== {op}: per cell  model -> comp -> comp_mean : power x1000 (dominant mechanism), "
              "cos(write, model), top-4 creating neurons shared")  # fmt: skip
        for f, per in enumerate(FAMILIES):
            for j, layer in enumerate(LAYERS):
                if power[0, o, f, j] < 0.25 * power[0, o, f].max() or power[0, o, f, j] < 1e-3:
                    continue
                cells = []
                for mi in range(3):
                    g = names[int(np.argmax(groups[mi, o, f, j]))]
                    cre = groups[mi, o, f, j, :3].sum()
                    cells.append(f"{power[mi, o, f, j] * 1000:5.1f} {g:<2s} (new {cre:.2f})")
                print(f"  mod {per:3d} L{layer}: " + " -> ".join(cells)
                      + f" | cos {cos[0, o, f, j]:.2f}/{cos[1, o, f, j]:.2f}"
                      + f" | top-4 {overlap[0, o, f, j]:.2f}/{overlap[1, o, f, j]:.2f}")  # fmt: skip
    # overall: power-weighted agreement of the dominant mechanism, and mean cos
    for ci, mode in ((1, "comp"), (2, "comp_mean")):
        w = power[0]
        same = np.argmax(groups[0], -1) == np.argmax(groups[ci], -1)
        print(f"{mode}: dominant mechanism agrees on {(same * w).sum() / w.sum():.2f} of the write "
              f"power; power-weighted cos {(cos[ci - 1] * w).sum() / w.sum():.2f}; "
              f"power ratio {power[ci].sum() / power[0].sum():.2f}")  # fmt: skip
    np.savez(OUT / "periods/compare.npz", power=power, groups=groups, cos=cos, overlap=overlap)


if __name__ == "__main__":
    main()
