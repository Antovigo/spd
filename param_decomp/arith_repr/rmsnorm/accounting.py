"""Where does the difference between the decomposed and the original residual stream come from?

At the attention norm of block L (norm point t = 2L), position `=`, the difference d = x_dec - x_orig is
the sum over blocks l < L of e_l = out_dec_l(x_dec_l) - out_orig_l(x_orig_l), and each term splits as

    e_l = -trunc_l + prop_l
    trunc_l = out_dense_l(x_dec_l) - out_dec_l(x_dec_l)   what block l's OFF components would have written
                                                         on the decomposed model's own input (local removal)
    prop_l  = out_dense_l(x_dec_l) - out_orig_l(x_orig_l) how the original block's output moves because its
                                                         input is already different (drift from upstream)

trunc_l is measured by running the decomposed model with block l dense and reading the stream after block l.
Per prompt we store the inner products needed to express the rms change and the alpha / stray split of
each part against x_orig.

    python -m param_decomp.arith_repr.rmsnorm.accounting"""

import numpy as np

from param_decomp.arith_repr.rmsnorm.model import NormModel
from param_decomp.arith_repr.rmsnorm.run import OUT, log, savez, subset

EQ = 4
N_BLOCK = 32
N_PROMPT = 1000
STATS = ("oo", "To", "Po", "TT", "PP", "TP", "err")


def run(M: NormModel) -> None:
    rows_all = subset(N_PROMPT, seed=7)
    out = np.zeros((len(STATS), N_BLOCK, len(rows_all)), np.float64)
    s0 = 0
    for rows, n in M.chunks(rows_all):
        even = {2 * b for b in range(N_BLOCK)}
        _, _, xo = M.forward_chunk(rows, False, capture=even)
        _, _, xd = M.forward_chunk(rows, True, capture=even)
        xo = {t: np.asarray(v[:n, EQ], np.float64) for t, v in xo.items()}
        xd = {t: np.asarray(v[:n, EQ], np.float64) for t, v in xd.items()}
        cum_t = np.zeros_like(xo[0])
        cum_p = np.zeros_like(xo[0])
        for b in range(N_BLOCK):
            if b > 0:
                d = xd[2 * b] - xo[2 * b]
                oo = (xo[2 * b] ** 2).sum(-1)
                sl = slice(s0, s0 + n)
                out[0, b, sl] = oo
                out[1, b, sl] = (cum_t * xo[2 * b]).sum(-1)
                out[2, b, sl] = (cum_p * xo[2 * b]).sum(-1)
                out[3, b, sl] = (cum_t**2).sum(-1)
                out[4, b, sl] = (cum_p**2).sum(-1)
                out[5, b, sl] = (cum_t * cum_p).sum(-1)
                out[6, b, sl] = ((d - cum_t - cum_p) ** 2).sum(-1)
            if b == N_BLOCK - 1:
                break
            decs = [True] * M.n_layer
            decs[b] = False
            _, _, xh = M.forward_chunk(rows, decs, capture={2 * b + 2})
            xh_next = np.asarray(xh[2 * b + 2][:n, EQ], np.float64)
            trunc = xh_next - xd[2 * b + 2]
            prop = (xh_next - xd[2 * b]) - (xo[2 * b + 2] - xo[2 * b])
            cum_t -= trunc
            cum_p += prop
        s0 += n
        log(f"accounting {s0}/{len(rows_all)}")
    OUT.mkdir(parents=True, exist_ok=True)
    savez(OUT / "accounting.npz", rows=rows_all, **dict(zip(STATS, out, strict=True)))


if __name__ == "__main__":
    M = NormModel()
    log("model loaded")
    run(M)
