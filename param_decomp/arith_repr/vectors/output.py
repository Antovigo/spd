"""From the result codes to the number token: the unembedding's own Fourier planes.

    python -m param_decomp.arith_repr.vectors.output   # -> OUT/output.npz

The unembedding rows of the number tokens, with the final-norm weight folded in (`WU_numbers_g.npy`,
tokens "0".."199", "-"), have a line code of their own: `G(k) = mean_{n<100} W[n] e^{-2 pi i k n/100}`
(and `G100(k)` the same over n = 100..199). A result code F(k) in the last residual (at `=`) adds
`2 Re(F(k) . conj G(k) e^{2 pi i k (res - n)/100}) / rms` to the logit of token n: a vote peaked at
n = res (mod 100/k) when `arg(F . conj G) = 0`. Stored:

* `vote` (65 points, 2 op, 50): F_t(res line, k) . conj G(k) / rms_t — the logit-lens vote of every
  harmonic of the result code at every stream point (res = sum on add, diff on sub);
* `vote100` the same against the 100..199 rows; `hund_dir` = mean W[100..199] - mean W[0..99];
* `wvote` (Aw, 2 op, 50): each residual writer's own vote R_c(res, k) (U_c . conj G(k)) / rms_final;
* `whund` (Aw, 2 op): U_c . hund_dir / rms_final times the covariance of c's inner with
  [a + b >= 100] (add) or [a < b] (sub) — the writer's push towards the hundreds / negative tokens;
* `G`, `G100`, `rms_final`."""

import numpy as np

from param_decomp.arith_repr.vectors.common import DATASET, OUT, comp_table, load_uv
from param_decomp.arith_repr.vectors.load import load_reads
from param_decomp.arith_repr.vectors.storage import load_frame


def main() -> None:
    W = np.load(OUT / "WU_numbers_g.npy")  # (202, 4096)
    n = np.arange(100)
    e = np.exp(-2j * np.pi * np.arange(1, 51)[:, None] * n[None] / 100)  # (50, 100)
    W0 = W[:100] - W[:200].mean(0)
    W1 = W[100:200] - W[:200].mean(0)
    G = e @ W0 / 100
    G100 = e @ W1 / 100
    hund = W[100:200].mean(0) - W[:100].mean(0)
    fre = np.load(OUT / "frames_re.npy", mmap_mode="r")
    fim = np.load(OUT / "frames_im.npy", mmap_mode="r")
    st = dict(np.load(OUT / "frames_stats.npz"))
    n_pt = fre.shape[0]
    vote = np.zeros((n_pt, 2, 50), np.complex64)
    vote100 = np.zeros_like(vote)
    for t in range(n_pt):
        F = load_frame(fre, fim, st, t)  # (4, 2, 4, 50, d)
        for o in range(2):
            Fr = F[3, o, 2 + o]  # (50, d): pos '=', res line
            vote[t, o] = (Fr * np.conj(G)).sum(-1) / st["rms"][t, 3, o]
            vote100[t, o] = (Fr * np.conj(G100)).sum(-1) / st["rms"][t, 3, o]
    rms_final = st["rms"][-1, 3]
    comps = comp_table()
    reads = load_reads()
    cols = np.flatnonzero(np.isin(comps["kind"], ("o", "down")))
    _, U = load_uv(comps, cols)
    Um = np.stack(U)
    ug = Um @ np.conj(G).T  # (Aw, 50)
    wvote = np.zeros((cols.size, 2, 50), np.complex64)
    for o in range(2):
        wvote[:, o] = reads.R[cols, 3, o, 2 + o] * ug / rms_final[o]
    inner = np.load(DATASET / "original/inner.npy", mmap_mode="r")
    ix = np.load(DATASET / "index.npz")
    a, b = ix["a"], ix["b"]
    whund = np.zeros((cols.size, 2), np.float32)
    for o in range(2):
        sl = slice(o * 10000, (o + 1) * 10000)
        h = np.asarray(inner[sl, 4][:, cols], np.float32)
        ind = ((a[sl] + b[sl] >= 100) if o == 0 else (a[sl] < b[sl])).astype(np.float32)
        cov = ((h - h.mean(0)) * (ind - ind.mean())[:, None]).mean(0) / ind.std()
        whund[:, o] = cov * (Um @ hund) / rms_final[o]
    np.savez(
        OUT / "output.npz", vote=vote, vote100=vote100, wvote=wvote, whund=whund, cols=cols, G=G,
        G100=G100, hund_dir=hund, rms_final=rms_final,
    )  # fmt: skip


if __name__ == "__main__":
    main()
