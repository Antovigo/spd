"""Attention patterns of the ORIGINAL model on the addsub pool, from the stored residual stream.

    python -m param_decomp.arith_repr.autointerp.attn_patterns --resid <original-resid dir> --out attn.npy

Recomputes q, k (with the llama3 RoPE) from each layer's post-norm input and writes the softmax
pattern `(n_layer, N, n_head, T, T)` float16 (query, key), causal."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.arith_repr.autointerp.llama_weights import Weights, llama3_inv_freq
from param_decomp.arith_repr.resid import load_read_input


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resid", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    w = Weights()
    cfg = w.config
    n_layer, n_head, n_kv = (
        cfg["num_hidden_layers"],
        cfg["num_attention_heads"],
        cfg["num_key_value_heads"],
    )
    hd = cfg["hidden_size"] // n_head
    T = 5
    inv = llama3_inv_freq(cfg)
    freqs = np.arange(T, dtype=np.float32)[:, None] * inv[None]
    emb = np.concatenate([freqs, freqs], -1)
    cos, sin = jnp.asarray(np.cos(emb)), jnp.asarray(np.sin(emb))

    def rot(x: jax.Array) -> jax.Array:  # (N, T, H, hd)
        x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
        rh = jnp.concatenate([-x2, x1], -1)
        return x * cos[None, :, None] + rh * sin[None, :, None]

    @jax.jit
    def pattern(x: jax.Array, wq: jax.Array, wk: jax.Array) -> jax.Array:
        n = x.shape[0]
        q = (x @ wq.T).reshape(n, T, n_head, hd)
        k = (x @ wk.T).reshape(n, T, n_kv, hd)
        q, k = rot(q), rot(k)
        k = jnp.repeat(k, n_head // n_kv, axis=2)
        s = jnp.einsum("nqhd,nkhd->nhqk", q, k) / np.sqrt(hd)
        mask = jnp.tril(jnp.ones((T, T), bool))
        s = jnp.where(mask[None, None], s, -jnp.inf)
        return jax.nn.softmax(s, -1).astype(jnp.float16)

    out = None
    for li in range(n_layer):
        x = np.stack([load_read_input(args.resid, f"attn_in.{li}", p) for p in range(T)], 1)
        wq = jnp.asarray(w.get(f"model.layers.{li}.self_attn.q_proj.weight"))
        wk = jnp.asarray(w.get(f"model.layers.{li}.self_attn.k_proj.weight"))
        pats = []
        for lo in range(0, x.shape[0], 5000):
            pats.append(np.asarray(pattern(jnp.asarray(x[lo : lo + 5000]), wq, wk)))
        pat = np.concatenate(pats, 0)
        if out is None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            out = np.lib.format.open_memmap(args.out, "w+", np.float16, (n_layer, *pat.shape))
        out[li] = pat
        print(
            f"layer {li}: mean pattern at '=' (head-avg)",
            pat[:, :, 4].astype(np.float32).mean((0, 1)),
            flush=True,
        )
    assert out is not None
    out.flush()


if __name__ == "__main__":
    main()
