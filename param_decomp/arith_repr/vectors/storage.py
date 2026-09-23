"""Where each code lives in the raw stream, how stable its directions are, and whether positions share them.

    python -m param_decomp.arith_repr.vectors.storage   # -> OUT/storage.npz

From the frames (`spectra frames`), with the linear part of a and b removed from the a / b lines
(the ramp q - 50.5 has a 1/k Fourier tail that would otherwise sit on k = 1, 2, 3):

* `energy` (65, 4 pos, 2 op, 4 lines, 50): share of the stream's total variance carried by harmonic
  k of line q (2 |F|^2 / tvar; |F|^2 at k = 50); `lin` (65, 4, 2, 2): share of the linear a / b parts.
* `cos_next` (64, 4, 2, 4, 50): |<F_t, F_t+1>| / |F_t| |F_t+1| (complex cosine; 1 = same plane and
  same phase convention) between consecutive stream points.
* `cos_pos`: (65, 2 op, 50) complex cosine between a's code at position 1 and b's code at position 3
  (does the model store both operands in the same directions?), and `cos_pos_phase` its argument.
* `cos_op` (65, 4, 4, 50): complex cosine between the add and sub frames of the same line
  (`<F_sub, F_add>`), and `cos_op_conj` against the mirror image (`<F_sub, conj F_add>`, i.e. q -> -q)."""

import numpy as np

from param_decomp.arith_repr.vectors.common import OUT


def ramp_dft() -> np.ndarray:
    q = np.arange(1, 101)
    k = np.arange(1, 51)[:, None]
    return ((q - 50.5)[None] * np.exp(-2j * np.pi * k * q / 100)).mean(1)  # (50,)


def ccos(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    num = (x * np.conj(y)).sum(-1)
    return num / (np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1) + 1e-12)


def load_frame(fre: np.ndarray, fim: np.ndarray, st: dict[str, np.ndarray], t: int) -> np.ndarray:
    F = np.asarray(fre[t], np.float32) + 1j * np.asarray(fim[t], np.float32)  # (4,2,4,50,d)
    c = ramp_dft()
    F[:, :, 0] -= c[None, None, :, None] * st["slope_a"][t][:, :, None, :]
    F[:, :, 1] -= c[None, None, :, None] * st["slope_b"][t][:, :, None, :]
    return F


def main() -> None:
    fre = np.load(OUT / "frames_re.npy", mmap_mode="r")
    fim = np.load(OUT / "frames_im.npy", mmap_mode="r")
    st = dict(np.load(OUT / "frames_stats.npz"))
    n_pt = fre.shape[0]
    w = np.full(50, 2.0)
    w[-1] = 1.0
    energy = np.zeros((n_pt, 4, 2, 4, 50), np.float32)
    lin = np.zeros((n_pt, 4, 2, 2), np.float32)
    cos_next = np.zeros((n_pt - 1, 4, 2, 4, 50), np.complex64)
    cos_pos = np.zeros((n_pt, 2, 50), np.complex64)
    cos_op = np.zeros((n_pt, 4, 4, 50), np.complex64)
    cos_op_conj = np.zeros_like(cos_op)
    prev = None
    for t in range(n_pt):
        F = load_frame(fre, fim, st, t)
        tv = st["tvar"][t][:, :, None, None]
        energy[t] = (np.abs(F) ** 2).sum(-1) * w / tv
        lin[t, :, :, 0] = (st["slope_a"][t] ** 2).sum(-1) * 833.25 / st["tvar"][t]
        lin[t, :, :, 1] = (st["slope_b"][t] ** 2).sum(-1) * 833.25 / st["tvar"][t]
        if prev is not None:
            cos_next[t - 1] = ccos(prev, F)
        cos_pos[t] = ccos(F[0, :, 0], F[2, :, 1])
        cos_op[t] = ccos(F[:, 1], F[:, 0])
        cos_op_conj[t] = ccos(F[:, 1], np.conj(F[:, 0]))
        prev = F
        print(t, flush=True)
    np.savez(
        OUT / "storage.npz", energy=energy, lin=lin, cos_next=cos_next, cos_pos=cos_pos,
        cos_op=cos_op, cos_op_conj=cos_op_conj,
    )  # fmt: skip


if __name__ == "__main__":
    main()
