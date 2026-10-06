"""Per-head statistics of the pattern comparison (`extract.py` outputs) -> OUT/summary.npz.

Shapes: L = 32 blocks, H = 32 heads, T = 5 positions (query or key), N = 20000 prompts.
* `meanP_<c>` (L, H, T, T): pattern averaged over prompts, c in orig / dec / loc.
* `tv_<c>`, `tv90_<c>` (L, H, T): mean and 90th percentile over prompts of the total variation
  distance TV = 1/2 sum_key |P_c - P_orig| of each query row, c in dec / loc.
* `r2_<c>` (L, H, T): share of the ORIGINAL pattern's prompt-to-prompt variance that P_c reproduces,
  1 - sum (P_c - P_orig)^2 / sum (P_orig - mean_prompts P_orig)^2 (sums over prompts and keys).
* `var_orig` (L, H, T): that prompt-to-prompt variance of the original row (mean over prompts of
  sum_key (P_orig - mean P_orig)^2), i.e. how prompt-dependent the head is; `var_<c>` the same for
  P_c (c in dec / loc).
* `W_<c>` (L, H, T): mean norm of the head's write at the query position; `dW` (L, H, T): mean norm
  of the change of the decomposed head's write when its pattern is swapped for the original one.
* `tvw_<c>` (L, T): TV averaged over heads and prompts, weighted by the original write norm;
  `tvwd_<c>` the same weighted by the decomposed write norm (the heads the decomposition uses).
* `corr_<c>` (L, H, T, T): Pearson correlation over prompts between P_c[query, key] and
  P_orig[query, key] (NaN where either is constant).
* `ctv_<c>` (L, H, T): TV between the rows restricted to the non-<BOS> keys and renormalised, i.e.
  the disagreement on WHERE the head looks once the <BOS> sink is set aside (query b and =).
* `uniform_dec` (L, H, T): share of prompts whose decomposed row is uniform over the causal keys
  (max |P - 1/(q+1)| < 0.01), the signature of a query or key with no active component.
* `n_active` (L, 4, T): mean number of active (masked-in) q / k / v / o components per position."""

from typing import Any, cast

import numpy as np

from param_decomp.arith_repr.attn_patterns.extract import CONDS, KINDS, OUT, H, L, T
from param_decomp.arith_repr.vectors.common import RUN, comp_table


def main() -> None:
    P = {c: np.load(OUT / f"{c}_P.npy", mmap_mode="r") for c in CONDS}
    Wn = {c: np.load(OUT / f"{c}_W.npy", mmap_mode="r") for c in CONDS}
    dW = np.load(OUT / "dec_dW.npy", mmap_mode="r")
    comps = comp_table()
    res: dict[str, np.ndarray] = {}

    def put(key: str, li: int, val: np.ndarray, shape: tuple[int, ...]) -> None:
        res.setdefault(key, np.zeros(shape, np.float32))[li] = val

    causal = np.tril(np.ones((T, T), bool))
    unif = np.where(causal, 1.0 / causal.sum(1, keepdims=True), 0.0)
    for li in range(L):
        po = np.asarray(P["orig"][li], np.float32)  # (N, H, T, T)
        wo = np.asarray(Wn["orig"][li], np.float32).transpose(0, 2, 1)  # (N, H, T)
        mo = po.mean(0)
        dev = ((po - mo) ** 2).sum(-1)  # (N, H, T)
        put("var_orig", li, dev.mean(0), (L, H, T))
        for c in CONDS:
            pc = po if c == "orig" else np.asarray(P[c][li], np.float32)
            put(f"meanP_{c}", li, pc.mean(0), (L, H, T, T))
            put(f"W_{c}", li, np.asarray(Wn[c][li], np.float32).mean(0).T, (L, H, T))
            if c == "orig":
                continue
            tv = 0.5 * np.abs(pc - po).sum(-1)  # (N, H, T)
            put(f"tv_{c}", li, tv.mean(0), (L, H, T))
            put(f"tv90_{c}", li, np.quantile(tv, 0.9, axis=0), (L, H, T))
            err = ((pc - po) ** 2).sum(-1).sum(0)
            put(f"r2_{c}", li, 1 - err / np.maximum(dev.sum(0), 1e-12), (L, H, T))
            put(f"tvw_{c}", li, (tv * wo).sum((0, 1)) / wo.sum((0, 1)), (L, T))
            wd = np.asarray(Wn["dec"][li], np.float32).transpose(0, 2, 1)
            put(f"tvwd_{c}", li, (tv * wd).sum((0, 1)) / np.maximum(wd.sum((0, 1)), 1e-12), (L, T))
            do, dc = po - mo, pc - pc.mean(0)
            cov = (do * dc).mean(0)
            with np.errstate(invalid="ignore", divide="ignore"):
                put(f"corr_{c}", li, cov / (do.std(0) * dc.std(0)), (L, H, T, T))
            ro = po[..., 1:] / np.maximum(po[..., 1:].sum(-1, keepdims=True), 1e-12)
            rc = pc[..., 1:] / np.maximum(pc[..., 1:].sum(-1, keepdims=True), 1e-12)
            put(f"var_{c}", li, (dc**2).sum(-1).mean(0), (L, H, T))
            put(f"ctv_{c}", li, 0.5 * np.abs(rc - ro).sum(-1).mean(0), (L, H, T))
            if c == "dec":
                u = np.abs(pc - unif).max(-1) < 0.01
                put("uniform_dec", li, u.mean(0), (L, H, T))
        put("dW", li, np.asarray(dW[li], np.float32).mean(0).T, (L, H, T))
        masks = np.load(RUN / "analysis/arith_repr/atlas/masks" / f"L{li}.npy", mmap_mode="r")
        cols = np.flatnonzero(comps["layer"] == li)
        m = np.asarray(masks[::10], np.float32)  # (N/10, T, n)
        na = [
            m[:, :, comps["site"][cols] == f"layers.{li}.{s}"].sum(-1).mean(0)
            for s in KINDS.values()
        ]
        put("n_active", li, np.stack(na), (L, 4, T))
        put(
            "n_alive",
            li,
            np.array([(comps["site"][cols] == f"layers.{li}.{s}").sum() for s in KINDS.values()]),
            (L, 4),
        )
        print(li, flush=True)
    np.savez(OUT / "summary.npz", **cast(dict[str, Any], res))


if __name__ == "__main__":
    main()
