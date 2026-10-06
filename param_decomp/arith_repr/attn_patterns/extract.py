"""Attention patterns of every head, original Llama-3.1-8B vs the rounded-CI decomposed model of
p-ba5a0c05, on the 20000 addsub prompts (5 positions: <BOS>, a, op, b, =).

Three conditions per block l, all from the dataset's stored raw residual streams:
* "orig": dense weights on the original stream.
* "dec": the decomposed model, i.e. the masked alive components (`m = 1[CI > 0.01]` on the original
  model's activations, no delta) on the decomposed model's own stream.
* "loc": the same masked components on the ORIGINAL stream, which isolates what the block's own
  q/k/v/o components change from what the upstream drift of the stream changes.

Per condition, outputs in OUT:
* `<cond>_P.npy` (L, N, H, T, T) float16: the softmax pattern of each head (query row, key column).
* `<cond>_W.npy` (L, N, T, H) float16: the norm of head h's write into the residual stream at each
  query position (its o-projection share: the dense W_o slice, or the masked o components).
* `dec_dW.npy` (L, N, T, H) float16: the norm of the change of the decomposed head's write when its
  own pattern is replaced by the original one (values and o unchanged) — how much the pattern
  disagreement moves what the decomposed head writes.
* `check.json`: max relative error of the recomputed q/k/o inner activations against the dataset's
  stored `inner.npy` for both models (validates norms, RoPE, GQA and the patterns)."""

import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.arith_repr.autointerp.llama_weights import Weights, llama3_inv_freq
from param_decomp.arith_repr.vectors.common import AUTOINTERP, DATASET, RESID, RUN, comp_table

jax.config.update("jax_default_matmul_precision", "highest")

OUT = RUN / "analysis/attn_patterns"
L, H, KV, HD, T, D = 32, 32, 8, 128, 5, 4096
CH = 1000
KINDS = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj"}  # fmt: skip
CONDS = ("orig", "dec", "loc")
N_CHECK = 1000


def rope() -> tuple[jax.Array, jax.Array]:
    inv = llama3_inv_freq(Weights().config)
    e = np.arange(T, dtype=np.float32)[:, None] * inv[None]
    e = np.concatenate([e, e], -1)
    return jnp.asarray(np.cos(e)), jnp.asarray(np.sin(e))


COS, SIN = rope()


def rot(x: jax.Array) -> jax.Array:
    h = HD // 2
    return (
        x * COS[None, :, None] + jnp.concatenate([-x[..., h:], x[..., :h]], -1) * SIN[None, :, None]
    )


@jax.jit
def pattern(q: jax.Array, k: jax.Array) -> jax.Array:
    """q (B, T, H*HD), k (B, T, KV*HD) -> (B, H, T, T) causal softmax pattern."""
    B = q.shape[0]
    q = rot(q.reshape(B, T, H, HD))
    k = jnp.repeat(rot(k.reshape(B, T, KV, HD)), H // KV, axis=2)
    sc = jnp.einsum("bqhd,bkhd->bhqk", q, k) / jnp.sqrt(HD)
    sc = jnp.where(jnp.tril(jnp.ones((T, T), bool))[None, None], sc, -jnp.inf)
    return jax.nn.softmax(sc, -1)


@jax.jit
def mix(P: jax.Array, v: jax.Array) -> jax.Array:
    """P (B, H, T, T), v (B, T, KV*HD) -> per-head mixed values z (B, T, H, HD)."""
    B = v.shape[0]
    v = jnp.repeat(v.reshape(B, T, KV, HD), H // KV, axis=2)
    return jnp.einsum("bhqk,bkhd->bqhd", P, v)


@jax.jit
def write_dense(z: jax.Array, Wo: jax.Array) -> jax.Array:
    """z (B, T, H, HD), Wo (D, H*HD) -> per-head writes (B, T, H, D)."""
    return jnp.einsum("bqhd,ehd->bqhe", z, Wo.reshape(D, H, HD))


@jax.jit
def write_comp(z: jax.Array, V: jax.Array, m: jax.Array, U: jax.Array) -> jax.Array:
    """Masked o components, split by head: (z_h @ V[h rows]) * m @ U -> (B, T, H, D)."""
    inner = jnp.einsum("bqhd,hdc->bqhc", z, V.reshape(H, HD, -1))
    return jnp.einsum("bqhc,bqc,ce->bqhe", inner, m, U)


@jax.jit
def norm(x: jax.Array, g: jax.Array, eps: float) -> jax.Array:
    return x / jnp.sqrt((x * x).mean(-1, keepdims=True) + eps) * g


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    w = Weights()
    comps = comp_table()
    uv = np.load(AUTOINTERP / "uv_alive.npz")
    norms = np.load(RESID / "norms.npz")
    ln1, eps = norms["ln1"], float(norms["eps"])
    resid = {
        m: np.load(DATASET / m / "resid.npy", mmap_mode="r") for m in ("original", "decomposed")
    }
    inner = {
        m: np.load(DATASET / m / "inner.npy", mmap_mode="r") for m in ("original", "decomposed")
    }
    N = resid["original"].shape[1]
    arr = {}
    for c in CONDS:
        arr[c + "_P"] = np.lib.format.open_memmap(
            OUT / f"{c}_P.npy", "w+", np.float16, (L, N, H, T, T)
        )
        arr[c + "_W"] = np.lib.format.open_memmap(
            OUT / f"{c}_W.npy", "w+", np.float16, (L, N, T, H)
        )
    arr["dec_dW"] = np.lib.format.open_memmap(OUT / "dec_dW.npy", "w+", np.float16, (L, N, T, H))
    check: dict[str, dict[str, float]] = {}
    t0 = time.time()
    for li in range(L):
        dense = {k: jnp.asarray(w.get(f"model.layers.{li}.{s}.weight")) for k, s in KINDS.items()}
        cols_l = np.flatnonzero(comps["layer"] == li)
        masks = np.load(RUN / "analysis/arith_repr/atlas/masks" / f"L{li}.npy", mmap_mode="r")
        site = {}
        for kind, suffix in KINDS.items():
            key = f"layers.{li}.{suffix}"
            sel = np.flatnonzero(comps["site"][cols_l] == key)  # columns in this layer's mask array
            ids = {int(i): j for j, i in enumerate(uv[key + ".ids"])}
            order = [ids[int(comps["cidx"][cols_l[s]])] for s in sel]
            site[kind] = (jnp.asarray(uv[key + ".V"][:, order]), jnp.asarray(uv[key + ".U"][order]), sel, cols_l[sel])  # fmt: skip
        g = jnp.asarray(ln1[li])
        errs = {}
        for s in range(0, N, CH):
            rows = slice(s, min(s + CH, N))
            m = jnp.asarray(masks[rows].astype(np.float32))
            xs = {"original": norm(jnp.asarray(resid["original"][2 * li, rows], jnp.float32), g, eps),
                  "decomposed": norm(jnp.asarray(resid["decomposed"][2 * li, rows], jnp.float32), g, eps)}  # fmt: skip

            def comp(kind: str, x: jax.Array) -> tuple[jax.Array, jax.Array]:
                V, U, sel, _ = site[kind]  # noqa: B023
                h = x @ V
                return h, (h * m[:, :, sel]) @ U  # noqa: B023

            res = {}
            # orig
            q, k, v = (xs["original"] @ dense[n].T for n in "qkv")
            pat = pattern(q, k)
            z = mix(pat, v)
            res["orig"] = (pat, write_dense(z, dense["o"]), None)
            inner_o = {"original": z.reshape(z.shape[0], T, -1) @ site["o"][0]}
            inner_qk = {"original": {n: xs["original"] @ site[n][0] for n in "qk"}}
            # dec and loc
            for cond, src in (("dec", "decomposed"), ("loc", "original")):
                (hq, q), (hk, k), (_, v) = (comp(n, xs[src]) for n in "qkv")
                pat = pattern(q, k)
                z = mix(pat, v)
                Vo, Uo, sel_o, _ = site["o"]
                mo = m[:, :, sel_o]
                res[cond] = (pat, write_comp(z, Vo, mo, Uo), z)
                if cond == "dec":
                    inner_o["decomposed"] = z.reshape(z.shape[0], T, -1) @ Vo
                    inner_qk["decomposed"] = {"q": hq, "k": hk}
                    z_swap = mix(res["orig"][0], v)
                    dW = write_comp(z_swap, Vo, mo, Uo) - res["dec"][1]
                    arr["dec_dW"][li, rows] = np.asarray(jnp.linalg.norm(dW, axis=-1), np.float16)
            for cond in CONDS:
                arr[cond + "_P"][li, rows] = np.asarray(res[cond][0], np.float16)
                arr[cond + "_W"][li, rows] = np.asarray(
                    jnp.linalg.norm(res[cond][1], axis=-1), np.float16
                )
            if s < N_CHECK:
                for model in ("original", "decomposed"):
                    for n, got in (
                        ("q", inner_qk[model]["q"]),
                        ("k", inner_qk[model]["k"]),
                        ("o", inner_o[model]),
                    ):
                        ref = np.asarray(inner[model][rows][:, :, site[n][3]], np.float32)
                        e = float(np.abs(np.asarray(got) - ref).max() / (np.abs(ref).max() + 1e-6))
                        errs[f"{model}.{n}"] = max(errs.get(f"{model}.{n}", 0.0), e)
        check[f"L{li}"] = errs
        print(f"L{li} {time.time() - t0:.0f}s {errs}", flush=True)
    for a in arr.values():
        a.flush()
    (OUT / "check.json").write_text(json.dumps(check, indent=1))


if __name__ == "__main__":
    main()
    sys.exit(0)
