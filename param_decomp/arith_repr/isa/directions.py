"""Stream directions of the ISA outputs and groups, and how they sit against the Fourier code planes.

    python -m param_decomp.arith_repr.isa.directions <layer> <add|both>
        # needs isa_L<l>_<ops>_unmix.npz -> ISA_OUT/isa_L<l>_<ops>_dirs.{npz,json}

Every ISA output is a linear read of the raw stream `x` at the MLP's input point: the reader c
computes `h_c = (x / rms) . (g * V_c)` with `g` the RMSNorm gain, the pipeline multiplies back the
RMS, and whitening and ICA are linear, so `S_j = (x - mean x) . f_j` with the

* filter  `f_j = sum_c unmix[c, j] g * V_c`: what the readers combine to read output j;
* pattern `p_j = E[(x - mean x) S_j]`: the direction along which the stream moves with output j.

With correlated stream noise the two differ; the pattern locates the representation, the filter
says how it is read. A group's subspace is the span of its outputs' patterns.

Checks against the vectors analysis (labels only enter here through the frames): principal cosines
between each group's pattern subspace and every code plane `F(q, k)` of the stream at the same point
(position `=`, addition), plus the linear directions `lin(a)`, `lin(b)` and the op-flag direction."""

import json
import sys

import numpy as np

from param_decomp.arith_repr.isa.pipeline import ISA_OUT, OPS, POS
from param_decomp.arith_repr.vectors.common import (
    DATASET,
    LINES,
    OUT,
    RESID,
    comp_table,
    load_uv,
    period,
    read_point,
)
from param_decomp.arith_repr.vectors.storage import load_frame

MIN_ENERGY = 0.002
"""Code planes carrying less than this share of the stream variance are not matched."""


def orth(A: np.ndarray) -> np.ndarray:
    """Orthonormal basis (columns) of the column span of A, rank-cut."""
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    return U[:, s > 1e-6 * s[0]]


def pcos(Qa: np.ndarray, Qb: np.ndarray) -> np.ndarray:
    return np.linalg.svd(Qa.T @ Qb, compute_uv=False)


def main() -> None:
    layer, ops = int(sys.argv[1]), sys.argv[2]
    z = np.load(ISA_OUT / f"isa_L{layer}_{ops}_unmix.npz")
    unmix, gid, cols, S = z["unmix"], z["gid"], z["cols"], z["S"].astype(np.float64)
    comps = comp_table()
    V, _ = load_uv(comps, cols)
    gain = np.load(RESID / "norms.npz")["ln2"][layer]
    R = np.stack(V) * gain[None]  # (n, 4096): reader c's direction on the raw stream
    filters = unmix.T @ R  # (r, 4096)

    t = read_point(layer, "gate")
    x = np.asarray(
        np.load(DATASET / "original/resid.npy", mmap_mode="r")[t, OPS[ops], POS], np.float64
    )
    xc = x - x.mean(0)
    S_hat = xc @ filters.T
    check = [float(np.corrcoef(S_hat[:, j], S[:, j])[0, 1]) for j in range(S.shape[1])]
    patterns = (xc.T @ S / len(S)).T  # (r, 4096)
    print(f"L{layer} {ops}: filter check min corr {min(check):.5f}", flush=True)

    # cosine between each output's filter and pattern
    fp = [float(abs(filters[j] @ patterns[j]) / np.linalg.norm(filters[j]) / np.linalg.norm(patterns[j]))
          for j in range(S.shape[1])]  # fmt: skip
    G = int(gid.max() + 1)
    bases = [orth(patterns[gid == g].T) for g in range(G)]
    between = np.zeros((G, G))
    for i in range(G):
        for j in range(G):
            between[i, j] = pcos(bases[i], bases[j])[0] if i != j else 1.0
    # variance of the stream carried by each group's pattern subspace
    tvar = float((xc**2).sum(1).mean())
    gvar = [float(((xc @ Q) ** 2).sum(1).mean() / tvar) for Q in bases]

    # code planes of the stream at the same point (position '=', addition), ramps removed
    fre = np.load(OUT / "frames_re.npy", mmap_mode="r")
    fim = np.load(OUT / "frames_im.npy", mmap_mode="r")
    st = dict(np.load(OUT / "frames_stats.npz"))
    F = load_frame(fre, fim, st, t)[3, 0]  # (4 lines, 50, 4096)
    energy = np.load(OUT / "storage.npz")["energy"][t, 3, 0]  # (4, 50)
    planes = []
    for li in range(4):
        for k0 in range(50):
            if energy[li, k0] >= MIN_ENERGY:
                Fk = F[li, k0]
                planes.append((f"{LINES[li]} k{k0 + 1} (mod {period(k0 + 1)})", float(energy[li, k0]),
                               orth(np.stack([Fk.real, Fk.imag], 1))))  # fmt: skip
    ones = {"lin(a)": st["slope_a"][t, 3, 0], "lin(b)": st["slope_b"][t, 3, 0],
            "op flag": st["mean"][t, 3, 0] - st["mean"][t, 3, 1]}  # fmt: skip
    for nm, v in ones.items():
        planes.append((nm, float("nan"), (v / np.linalg.norm(v))[:, None]))

    groups = []
    for g in range(G):
        match = []
        overlap = []
        for nm, e, Q in planes:
            c = pcos(bases[g], Q)
            match.append({"plane": nm, "energy": e, "cos1": float(c[0]),
                          "cos2": float(c[1]) if c.size > 1 else None})  # fmt: skip
            # share of the plane (or line) that lies inside the group's pattern subspace
            overlap.append(round(float((c**2).sum() / Q.shape[1]), 3))
        match.sort(key=lambda m: -(m["cos1"] + (m["cos2"] or 0.0)))
        # how much of the group's subspace the matched code planes span together
        top = [Q for nm, _, Q in planes if any(m["plane"] == nm for m in match[:6])]
        cover = pcos(bases[g], orth(np.concatenate(top, 1))) if top else np.zeros(1)
        groups.append({"g": g, "dims": int(bases[g].shape[1]), "stream_var": gvar[g],
                       "top_planes": match[:8], "overlap": overlap, "cover_by_top6": np.round(cover, 3).tolist()})  # fmt: skip
        print(f"g{g} ({bases[g].shape[1]} dims, {100 * gvar[g]:.1f}% of stream var): "
              + "; ".join(f"{m['plane']} {m['cos1']:.2f}/{(m['cos2'] or 0):.2f}" for m in match[:4]),
              flush=True)  # fmt: skip

    np.savez(ISA_OUT / f"isa_L{layer}_{ops}_dirs.npz", filters=filters.astype(np.float32),
             patterns=patterns.astype(np.float32), gid=gid, stream_point=t)  # fmt: skip
    res = {"layer": layer, "ops": ops, "stream_point": int(t), "filter_check_min": min(check),
           "filter_pattern_cos": np.round(fp, 3).tolist(), "between_groups": np.round(between, 3).tolist(),
           "planes": [nm for nm, _, _ in planes], "groups": groups}  # fmt: skip
    (ISA_OUT / f"isa_L{layer}_{ops}_dirs.json").write_text(json.dumps(res))
    print("saved", ISA_OUT / f"isa_L{layer}_{ops}_dirs.json", flush=True)


if __name__ == "__main__":
    main()
