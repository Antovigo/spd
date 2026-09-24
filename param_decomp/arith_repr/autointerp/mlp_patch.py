"""Causal test of `mlp_linearity`: run the frozen model with some MLPs, at some positions, replaced
by their linearised versions, and measure what the answer loses.

    python -m param_decomp.arith_repr.autointerp.mlp_patch --run <run_dir> --resid <dir>
        --start <layer> --tag <name> <variant> [<variant> ...]

A variant is `clean` or `<kind>:<positions>:<layers>` (positions from a, b, =, comma-separated;
layers `all`, `l` or `l0-l1`). Kinds, with the population statistics of `mlp_lin/L<l>.npz`
(so every replacement reproduces the MLP's mean output on the pool):

* `lin`: the affine map h = h_bar + s_bar (u - u_bar) + u_bar beta (g - g_bar) — the MLP as one
  linear transformation of its (normed) input; nothing switches and nothing multiplies.
* `sl`: silu replaced by each neuron's least-squares line, h = (alpha + beta g) u + c — the gates
  never switch, but gate x up is kept.
* `mean`: h = h_bar — the MLP's input-dependent write removed (the scale of the effects).

Every variant starts from the recorded residual entering block `start` (so `clean` is the model
from there on, in float32) on a fixed sample of the pool, and runs to the logits at `=`.
Writes `mlp_patch/<tag>.npz`: per variant the per-prompt KL(clean || variant) over the full
vocabulary, argmax, and log-prob of the answer's first token; plus the prompt labels."""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np

from param_decomp.arith_repr.autointerp.llama_weights import (
    Weights,
    llama3_inv_freq,
    number_token_ids,
)
from param_decomp.arith_repr.autointerp.mechanisms import AI
from param_decomp.arith_repr.resid import load_raw

POS = {"a": 1, "b": 3, "=": 4}
T = 5
CHUNK = 500


def parse(variant: str, n_layer: int) -> tuple[str, dict[tuple[int, int], str]]:
    """variant -> (name, {(layer, position): kind})."""
    if variant == "clean":
        return variant, {}
    kind, poss, layers = variant.split(":")
    if layers == "all":
        ls = range(n_layer)
    elif "-" in layers:
        lo, hi = layers.split("-")
        ls = range(int(lo), int(hi) + 1)
    else:
        ls = range(int(layers), int(layers) + 1)
    return variant, {(li, POS[p]): kind for li in ls for p in poss.split(",")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--resid", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("variants", nargs="+")
    args = parser.parse_args()
    w = Weights()
    cfg = w.config
    n_layer, n_head, n_kv = (
        cfg["num_hidden_layers"],
        cfg["num_attention_heads"],
        cfg["num_key_value_heads"],
    )
    eps = float(cfg["rms_norm_eps"])
    hd = cfg["hidden_size"] // n_head
    variants = [parse(v, n_layer) for v in args.variants]
    assert variants[0][0] == "clean"
    lin_dir = args.run / AI / "mech" / "mlp_lin"
    out_dir = args.run / AI / "mech" / "mlp_patch"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((args.resid / "pool.json").read_text())
    labels = np.array([(0 if o == "add" else 1, a, b) for o, a, b in meta["labels"]])
    rng = np.random.default_rng(0)
    idx = np.sort(
        np.concatenate(
            [
                rng.choice(np.flatnonzero(labels[:, 0] == o), args.n // 2, replace=False)
                for o in (0, 1)
            ]
        )
    )
    labels = labels[idx]
    res = np.where(labels[:, 0] == 0, labels[:, 1] + labels[:, 2], labels[:, 1] - labels[:, 2])
    num_ids, minus = number_token_ids(200)
    answer = np.where(res >= 0, num_ids[np.clip(res, 0, 200)], minus)

    h0 = np.stack([load_raw(args.resid, f"resid.{args.start}", p)[idx] for p in range(T)], 1)
    V, N = len(variants), len(idx)
    state = np.repeat(h0[None], V, 0)  # (V, N, T, d)

    inv = llama3_inv_freq(cfg)
    freqs = np.arange(T, dtype=np.float32)[:, None] * inv[None]
    emb = np.concatenate([freqs, freqs], -1)
    cos, sin = jnp.asarray(np.cos(emb)), jnp.asarray(np.sin(emb))

    def rms(x: jax.Array, wt: jax.Array) -> jax.Array:
        return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + eps) * wt

    def rot(x: jax.Array) -> jax.Array:
        x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
        return x * cos[None, :, None] + jnp.concatenate([-x2, x1], -1) * sin[None, :, None]

    @jax.jit
    def attn(h: jax.Array, W: dict[str, jax.Array]) -> jax.Array:
        n = h.shape[0]
        x = rms(h, W["ln1"])
        q = rot((x @ W["q"].T).reshape(n, T, n_head, hd))
        k = rot((x @ W["k"].T).reshape(n, T, n_kv, hd))
        v = (x @ W["v"].T).reshape(n, T, n_kv, hd)
        k = jnp.repeat(k, n_head // n_kv, axis=2)
        v = jnp.repeat(v, n_head // n_kv, axis=2)
        s = jnp.einsum("nqhd,nkhd->nhqk", q, k) / np.sqrt(hd)
        s = jnp.where(jnp.tril(jnp.ones((T, T), bool))[None, None], s, -jnp.inf)
        o = jnp.einsum("nhqk,nkhd->nqhd", jax.nn.softmax(s, -1), v).reshape(n, T, -1)
        return h + o @ W["o"].T

    def mlp_fn(kinds: tuple[str | None, ...]):  # noqa: ANN202
        @jax.jit
        def f(h: jax.Array, W: dict[str, jax.Array], S: dict[str, jax.Array]) -> jax.Array:
            y = rms(h, W["ln2"])
            g = y @ W["g"].T
            u = y @ W["u"].T
            hid = jax.nn.silu(g) * u
            cols = []
            for p in range(T):
                kind = kinds[p]
                gp, up = g[:, p], u[:, p]
                if kind is None:
                    cols.append(hid[:, p])
                elif kind == "lin":
                    cols.append(
                        S[f"{p}.h_bar"]
                        + S[f"{p}.s_bar"] * (up - S[f"{p}.u_bar"])
                        + S[f"{p}.u_bar"] * S[f"{p}.beta"] * (gp - S[f"{p}.g_bar"])
                    )
                elif kind == "sl":
                    cols.append((S[f"{p}.alpha"] + S[f"{p}.beta"] * gp) * up + S[f"{p}.c_sl"])
                elif kind == "mean":
                    cols.append(jnp.broadcast_to(S[f"{p}.h_bar"], hid[:, p].shape))
                else:
                    raise ValueError(kind)
            return h + jnp.stack(cols, 1) @ W["d"].T

        return f

    mlps: dict[tuple[str | None, ...], Callable[..., jax.Array]] = {}
    for li in range(args.start, n_layer):
        pre = f"model.layers.{li}."
        raw = {
            "ln1": w.get(pre + "input_layernorm.weight"),
            "ln2": w.get(pre + "post_attention_layernorm.weight"),
            "q": w.get(pre + "self_attn.q_proj.weight"),
            "k": w.get(pre + "self_attn.k_proj.weight"),
            "v": w.get(pre + "self_attn.v_proj.weight"),
            "o": w.get(pre + "self_attn.o_proj.weight"),
            "g": w.get(pre + "mlp.gate_proj.weight"),
            "u": w.get(pre + "mlp.up_proj.weight"),
            "d": w.get(pre + "mlp.down_proj.weight"),
        }
        W = {k: jnp.asarray(v) for k, v in raw.items()}
        S = {}
        if any(li == key[0] for _, pt in variants for key in pt):
            st = np.load(lin_dir / f"L{li}.npz")
            for pname, p in POS.items():
                for k in ("h_bar", "s_bar", "u_bar", "g_bar", "beta", "alpha", "c_sl"):
                    S[f"{p}.{k}"] = jnp.asarray(st[f"{pname}.{k}"])
        for vi, (_, pt) in enumerate(variants):
            kinds = tuple(pt.get((li, p)) for p in range(T))
            if kinds not in mlps:
                mlps[kinds] = mlp_fn(kinds)
            f = mlps[kinds]
            for lo in range(0, N, CHUNK):
                h = attn(jnp.asarray(state[vi, lo : lo + CHUNK]), W)
                state[vi, lo : lo + CHUNK] = np.asarray(f(h, W, S))
        print(f"layer {li} done", flush=True)

    final = jnp.asarray(w.get("model.norm.weight"))
    head = jnp.asarray(w.get("lm_head.weight"))

    @jax.jit
    def logprobs_fn(h: jax.Array, final: jax.Array, head: jax.Array) -> jax.Array:
        return jax.nn.log_softmax(rms(h, final) @ head.T, -1)

    def logprobs(h: jax.Array) -> jax.Array:
        return logprobs_fn(h, final, head)

    out: dict[str, np.ndarray] = {"labels": labels, "answer": answer, "idx": idx}
    lp_clean = [
        np.asarray(logprobs(jnp.asarray(state[0, lo : lo + CHUNK, 4]))) for lo in range(0, N, CHUNK)
    ]
    for vi, (name, _) in enumerate(variants):
        kl, top, ans = [], [], []
        for ci, lo in enumerate(range(0, N, CHUNK)):
            lp = np.asarray(logprobs(jnp.asarray(state[vi, lo : lo + CHUNK, 4])))
            lc = lp_clean[ci]
            kl.append((np.exp(lc) * (lc - lp)).sum(-1))
            top.append(lp.argmax(-1))
            ans.append(lp[np.arange(len(lp)), answer[lo : lo + CHUNK]])
        out[f"{name}.kl"] = np.concatenate(kl)
        out[f"{name}.top"] = np.concatenate(top)
        out[f"{name}.ans_lp"] = np.concatenate(ans)
        agree = (out[f"{name}.top"] == out["clean.top"]).mean()
        acc = (out[f"{name}.top"] == answer).mean()
        print(
            f"{name}: KL {out[f'{name}.kl'].mean():.4f} agree {agree:.3f} acc {acc:.3f} "
            f"ans_lp {out[f'{name}.ans_lp'].mean():.3f}",
            flush=True,
        )
    np.savez(out_dir / f"{args.tag}.npz", **cast(dict[str, Any], out))


if __name__ == "__main__":
    main()
