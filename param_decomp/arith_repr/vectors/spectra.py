"""Code frames of the raw stream, and the line spectra of every component's read (V) and write (U).

    python -m param_decomp.arith_repr.vectors.spectra frames   # -> OUT/frames_{re,im}.npy, frames_stats.npz
    python -m param_decomp.arith_repr.vectors.spectra reads    # -> OUT/read_spec.npz
    python -m param_decomp.arith_repr.vectors.spectra writes   # -> OUT/write_spec.npz (needs frames)

frames: for each of the 65 raw points (`common.point_names`), positions 1..4 (a, op, b, =), op
(add, sub) the line coefficients F (4 lines, 50 k, 4096) of the RAW residual over the full 100 x 100
grid (sub includes a < b), stored float16; plus per (point, pos, op) the mean, the linear slopes
on a and on b (4096 each), the total variance, the mean RMS, and the add - sub mean difference.

reads: the same line DFT of every alive component's inner activation `x_norm . V_c` over the grid:
R (A, 4 pos, 2 op, 4 lines, 50) complex, and mean / variance / linear slopes of the inner.

writes: for every residual writer (o, down) the projection of U_c on the frames at the point right
after its add: `U_c . F` (Aw, 4, 2, 4, 50) complex, `U_c . slope_a|b`, `U_c . (mean_add - mean_sub)`.
"""

import sys

import numpy as np

from param_decomp.arith_repr.vectors.common import (
    DATASET,
    OUT,
    comp_table,
    line_dft,
    load_uv,
    write_point,
)


def grid(x: np.ndarray) -> np.ndarray:
    return x.reshape(100, 100, *x.shape[1:])


def slopes(xg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    v = np.arange(1, 101, dtype=np.float32) - 50.5
    den = 100 * (v**2).sum()
    sa = np.tensordot(v, xg.sum(1), axes=(0, 0)) / den
    sb = np.tensordot(v, xg.sum(0), axes=(0, 0)) / den
    return sa, sb


def frames() -> None:
    mm = np.load(DATASET / "original/resid.npy", mmap_mode="r")
    n_pt = mm.shape[0]
    OUT.mkdir(parents=True, exist_ok=True)
    shape = (n_pt, 4, 2, 4, 50, 4096)
    fre = np.lib.format.open_memmap(OUT / "frames_re.npy", "w+", np.float16, shape)
    fim = np.lib.format.open_memmap(OUT / "frames_im.npy", "w+", np.float16, shape)
    mean = np.zeros((n_pt, 4, 2, 4096), np.float32)
    sa = np.zeros_like(mean)
    sb = np.zeros_like(mean)
    tvar = np.zeros((n_pt, 4, 2), np.float32)
    rms = np.zeros((n_pt, 4, 2), np.float32)
    for t in range(n_pt):
        block = np.asarray(mm[t])  # (N, T, d) f16
        for pi, p in enumerate(range(1, 5)):
            for o in range(2):
                x = block[o * 10000 : (o + 1) * 10000, p].astype(np.float32)
                rms[t, pi, o] = np.sqrt((x**2).mean(1)).mean()
                mean[t, pi, o] = x.mean(0)
                xc = x - mean[t, pi, o]
                tvar[t, pi, o] = (xc**2).sum(1).mean()
                xg = grid(xc)
                sa[t, pi, o], sb[t, pi, o] = slopes(xg)
                F = line_dft(xg)
                fre[t, pi, o] = np.real(F).astype(np.float16)
                fim[t, pi, o] = np.imag(F).astype(np.float16)
        print("frames", t, flush=True)
    fre.flush()
    fim.flush()
    np.savez(OUT / "frames_stats.npz", mean=mean, slope_a=sa, slope_b=sb, tvar=tvar, rms=rms)


def reads() -> None:
    mm = np.load(DATASET / "original/inner.npy", mmap_mode="r")
    A = mm.shape[2]
    R = np.zeros((A, 4, 2, 4, 50), np.complex64)
    mean = np.zeros((A, 4, 2), np.float32)
    var = np.zeros_like(mean)
    sa = np.zeros_like(mean)
    sb = np.zeros_like(mean)
    for pi, p in enumerate(range(1, 5)):
        for o in range(2):
            h = np.asarray(mm[o * 10000 : (o + 1) * 10000, p, :], np.float32)
            mean[:, pi, o] = h.mean(0)
            hc = h - mean[:, pi, o]
            var[:, pi, o] = (hc**2).mean(0)
            hg = grid(hc)
            sa[:, pi, o], sb[:, pi, o] = slopes(hg)
            R[:, pi, o] = np.moveaxis(line_dft(hg), -1, 0)
            print("reads", p, o, flush=True)
    np.savez(OUT / "read_spec.npz", R=R, mean=mean, var=var, slope_a=sa, slope_b=sb)


def writes() -> None:
    comps = comp_table()
    cols = np.flatnonzero(np.isin(comps["kind"], ("o", "down")))
    _, U = load_uv(comps, cols)
    Um = np.stack(U)  # (Aw, 4096)
    fre = np.load(OUT / "frames_re.npy", mmap_mode="r")
    fim = np.load(OUT / "frames_im.npy", mmap_mode="r")
    st = np.load(OUT / "frames_stats.npz")
    wp = np.array([write_point(comps["layer"][c], comps["kind"][c]) for c in cols])
    W = np.zeros((cols.size, 4, 2, 4, 50), np.complex64)
    WA = np.zeros((cols.size, 4, 2), np.float32)
    WB = np.zeros_like(WA)
    WOP = np.zeros((cols.size, 4), np.float32)
    for t in np.unique(wp):
        sel = np.flatnonzero(wp == t)
        Fr = np.asarray(fre[t], np.float32)  # (4, 2, 4, 50, 4096)
        Fi = np.asarray(fim[t], np.float32)
        W[sel] = np.einsum("cd,polkd->cpolk", Um[sel], Fr) + 1j * np.einsum(
            "cd,polkd->cpolk", Um[sel], Fi
        )
        WA[sel] = np.einsum("cd,pod->cpo", Um[sel], st["slope_a"][t])
        WB[sel] = np.einsum("cd,pod->cpo", Um[sel], st["slope_b"][t])
        WOP[sel] = np.einsum("cd,pd->cp", Um[sel], st["mean"][t, :, 0] - st["mean"][t, :, 1])
        print("writes", t, sel.size, flush=True)
    np.savez(
        OUT / "write_spec.npz", cols=cols, W=W, slope_a=WA, slope_b=WB, opdiff=WOP,
        unorm=np.linalg.norm(Um, axis=1),
    )  # fmt: skip


if __name__ == "__main__":
    {"frames": frames, "reads": reads, "writes": writes}[sys.argv[1]]()
