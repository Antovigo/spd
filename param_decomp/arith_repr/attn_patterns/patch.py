"""Causal check: does giving the decomposed model the ORIGINAL attention patterns repair it?

The decomposed forward of `ComponentsModel` (masked alive components, no delta), with the softmax
pattern of chosen blocks replaced by the original model's pattern on the same prompt
(`orig_P.npy` from `extract.py`); values, o and everything else stay decomposed. Measured:
KL(original || variant) of the last-position next-token distribution, per prompt, against the
original model's log-probs (recomputed with the dense model of `rmsnorm.model.NormModel`).

Variants: "dec" (no patch), "all" (every block patched), "L<l>" (block l only), "upto<l>"
(blocks 0..l), "from<l>" (blocks l..31). Prompts: every STRIDE-th row of the grid.
Output: OUT/patch.npz (`rows`, one KL vector per variant).

`heads` mode (argv[1] == "heads") -> OUT/patch_heads.npz: single heads "L<l>H<h>" of blocks 15, 16, 18
patched with the original pattern, their union "L15-16 top", and two prompt-mean controls on every
block: "orig_mean" (the original pattern averaged over prompts) and "dec_mean" (the decomposed
pattern averaged over prompts, i.e. the decomposed model with its prompt dependence removed)."""

import sys
import time
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.arith_repr.attn_patterns.extract import OUT, mix, pattern
from param_decomp.arith_repr.isa.components_model import T, _logprobs, _project, _rms, _write
from param_decomp.arith_repr.rmsnorm.model import NormModel, kl_rows

STRIDE = 10
HEADS = {15: (13, 3, 28, 14), 16: (21, 22, 3, 30), 18: (30, 31, 16, 18)}
Patch = dict[int, tuple[tuple[int, ...] | None, str]]  # block -> (heads or None = all, source)


class PatchModel(NormModel):
    def __init__(self) -> None:
        super().__init__(dense=True)
        self.orig_P = np.load(OUT / "orig_P.npy", mmap_mode="r")
        S = np.load(OUT / "summary.npz")
        self.mean_P = {"orig_mean": S["meanP_orig"], "dec_mean": S["meanP_dec"]}

    def source(self, li: int, rows: np.ndarray, src: str) -> jax.Array:
        if src == "orig":
            return jnp.asarray(self.orig_P[li, rows], jnp.float32)
        return jnp.broadcast_to(
            jnp.asarray(self.mean_P[src][li]), (len(rows),) + self.mean_P[src][li].shape
        )

    def run(self, rows: np.ndarray, patched: Patch) -> jax.Array:
        """Last-position log-probs of the decomposed model with the patterns in `patched` forced."""
        x = jnp.asarray(np.stack([[self.embed[int(t)] for t in toks] for toks in self.tokens[rows]]), jnp.float32)  # fmt: skip
        B = len(rows)
        for li in range(self.n_layer):
            m = jnp.asarray(self.masks(li, rows))

            def site(kind: str, xin: jax.Array) -> jax.Array:
                s = self.sites[li][kind]  # noqa: B023
                return _write(_project(xin, s.V), m[:, :, s.cols], s.U)  # noqa: B023

            xin = x / _rms(x, self.eps)[..., None] * self.ln[0][li]
            q, k, v = site("q", xin), site("k", xin), site("v", xin)
            pat = pattern(q, k)
            if li in patched:
                heads, src = patched[li]
                forced = self.source(li, rows, src)
                pat = (
                    forced if heads is None else pat.at[:, list(heads)].set(forced[:, list(heads)])
                )
            att = mix(pat, v).reshape(B, T, -1)
            x = x + site("o", att)
            xin = x / _rms(x, self.eps)[..., None] * self.ln[1][li]
            x = x + site("down", jax.nn.silu(site("gate", xin)) * site("up", xin))
        return _logprobs(x[:, -1], self.final_j, self.unembed)


def main() -> None:
    pm = PatchModel()
    rows_all = np.arange(0, len(pm.tokens), STRIDE)
    heads_mode = len(sys.argv) > 1 and sys.argv[1] == "heads"

    def blocks(ls: range | list[int], src: str = "orig") -> Patch:
        return {li: (None, src) for li in ls}

    variants: dict[str, Patch] = {"dec": {}}
    if heads_mode:
        variants |= {f"L{li}H{h}": {li: ((h,), "orig")} for li, hs in HEADS.items() for h in hs}
        variants["L15-16 top"] = {15: ((13, 3), "orig"), 16: ((21, 3), "orig")}
        variants["orig_mean"] = blocks(range(32), "orig_mean")
        variants["dec_mean"] = blocks(range(32), "dec_mean")
    else:
        variants["all"] = blocks(range(32))
        variants |= {f"L{li}": blocks([li]) for li in range(32)}
        variants |= {f"upto{li}": blocks(range(li + 1)) for li in range(0, 32, 4)}
        variants |= {f"from{li}": blocks(range(li, 32)) for li in range(0, 32, 4)}
    orig = []
    for r, n in pm.chunks(rows_all):
        lp, _, _ = pm.forward_chunk(r, dec=False)
        orig.append(np.asarray(lp[:n]))
    print("orig done", flush=True)
    res = {"rows": rows_all}
    t0 = time.time()
    for name, patched in variants.items():
        kls = []
        for (r, n), o in zip(pm.chunks(rows_all), orig, strict=True):
            kls.append(np.asarray(kl_rows(jnp.asarray(o), pm.run(r, patched)[:n])))
        res[name] = np.concatenate(kls)
        print(f"{name} KL={res[name].mean():.4f} {time.time() - t0:.0f}s", flush=True)
    np.savez(OUT / ("patch_heads.npz" if heads_mode else "patch.npz"), **cast(dict[str, Any], res))


if __name__ == "__main__":
    main()
    sys.exit(0)
