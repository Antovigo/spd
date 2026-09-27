"""The components-only model as an explicit JAX forward, with hooks for interventions.

Every site is replaced by its alive components, masked as in the dataset's decomposed run:
`out = ((x @ V) * m) @ U`, with `m = 1[CI > 0.01 on the original model]` (`ATLAS/masks`), no weight
delta. Only the alive components' U and V (`uv_alive.npz`), the token embeddings, the norm gains,
the final norm and the unembedding are needed.

Prompts run in chunks of CHUNK rows (the last one padded), so every shape is compiled once. Hooks
(all optional) get `idx`, the positions of the chunk's rows in the `rows` passed to `forward`:
* `resid(t, x, idx)`: the raw stream at point t (t = 2l before block l's attention, 2l + 1 before
  its MLP), (B, T, d) -> (B, T, d); the returned stream is carried on.
* `inner(layer, kind, h, rms, idx)`: a site's inner activations before masking, (B, T, n) ->
  (B, T, n); `rms` is the per-token RMS of the stream the site reads (B, T), so `h * rms` is the
  linear read of the raw stream used by the atlas."""

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from param_decomp.arith_repr.autointerp.llama_weights import (
    Weights,
    llama3_inv_freq,
    number_token_ids,
)
from param_decomp.arith_repr.isa.atlas import ATLAS
from param_decomp.arith_repr.vectors.common import AUTOINTERP, DATASET, RESID, comp_table

KINDS = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
         "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}  # fmt: skip
T = 5
CHUNK = 500
BF16 = (
    ml_dtypes.bfloat16
)  # importing ml_dtypes registers bfloat16 with numpy, for the safetensors reads

ResidHook = Callable[[int, jax.Array, np.ndarray], jax.Array]
InnerHook = Callable[[int, str, jax.Array, jax.Array, np.ndarray], jax.Array]


@dataclass
class Site:
    V: jax.Array  # (d_in, n)
    U: jax.Array  # (n, d_out)
    cols: np.ndarray  # columns of this site in the layer's mask array
    names: list[str]


@jax.jit
def _project(x: jax.Array, V: jax.Array) -> jax.Array:
    return x @ V


@jax.jit
def _write(h: jax.Array, m: jax.Array, U: jax.Array) -> jax.Array:
    return (h * m) @ U


@jax.jit
def _rms(x: jax.Array, eps: float = 1e-5) -> jax.Array:
    return jnp.sqrt((x * x).mean(-1) + eps)


@jax.jit
def _attention(
    q: jax.Array, k: jax.Array, v: jax.Array, cos: jax.Array, sin: jax.Array
) -> jax.Array:
    """q (B, T, H, hd), k and v (B, T, KV, hd) -> (B, T, H * hd), causal, RoPE on q and k."""
    hd = q.shape[-1]

    def rot(x: jax.Array) -> jax.Array:
        h = hd // 2
        return (
            x * cos[None, :, None]
            + jnp.concatenate([-x[..., h:], x[..., :h]], -1) * sin[None, :, None]
        )

    rep = q.shape[2] // k.shape[2]
    q, k = rot(q), jnp.repeat(rot(k), rep, axis=2)
    v = jnp.repeat(v, rep, axis=2)
    sc = jnp.einsum("nqhd,nkhd->nhqk", q, k) / jnp.sqrt(hd)
    Tq = q.shape[1]
    sc = jnp.where(jnp.tril(jnp.ones((Tq, Tq), bool))[None, None], sc, -jnp.inf)
    att = jnp.einsum("nhqk,nkhd->nqhd", jax.nn.softmax(sc, -1), v)
    return att.reshape(q.shape[0], Tq, -1)


@jax.jit
def _logprobs(x_last: jax.Array, final: jax.Array, unembed: jax.Array) -> jax.Array:
    xf = x_last / _rms(x_last)[:, None] * final
    return jax.nn.log_softmax(xf @ unembed.T, -1)


class ComponentsModel:
    def __init__(self) -> None:
        w = Weights()
        cfg = w.config
        self.n_layer, self.n_head, self.n_kv = (
            cfg["num_hidden_layers"],
            cfg["num_attention_heads"],
            cfg["num_key_value_heads"],
        )
        self.hd = cfg["hidden_size"] // self.n_head
        norms = np.load(RESID / "norms.npz")
        self.ln1, self.ln2, self.final, self.eps = (
            norms["ln1"],
            norms["ln2"],
            norms["final"],
            float(norms["eps"]),
        )
        ix = np.load(DATASET / "index.npz")
        self.tokens = ix["tokens"]
        self.a, self.b, self.op = ix["a"].astype(int), ix["b"].astype(int), ix["op"].astype(int)
        vocab = np.unique(self.tokens)
        emb = w.get_rows("model.embed_tokens.weight", vocab)
        self.embed = {int(t): emb[i] for i, t in enumerate(vocab)}
        self.unembed = jnp.asarray(w.get("lm_head.weight"))  # (V, d)
        self.final_j = jnp.asarray(self.final)
        comps = comp_table()
        uv = np.load(AUTOINTERP / "uv_alive.npz")
        self.sites: list[dict[str, Site]] = []
        for li in range(self.n_layer):
            cols = np.flatnonzero(comps["layer"] == li)
            layer = {}
            for kind, suffix in KINDS.items():
                key = f"layers.{li}.{suffix}"
                sel = np.flatnonzero(comps["site"][cols] == key)
                ids = {int(i): j for j, i in enumerate(uv[key + ".ids"])}
                order = [ids[int(comps["cidx"][cols[s]])] for s in sel]
                layer[kind] = Site(jnp.asarray(uv[key + ".V"][:, order], jnp.float32), jnp.asarray(uv[key + ".U"][order], jnp.float32),
                                   sel, [f"L{li}.{kind}.c{int(comps['cidx'][cols[s]])}" for s in sel])  # fmt: skip
            self.sites.append(layer)
        inv = llama3_inv_freq(cfg)
        freqs = np.arange(T, dtype=np.float32)[:, None] * inv[None]
        e = np.concatenate([freqs, freqs], -1)
        self.cos, self.sin = jnp.asarray(np.cos(e)), jnp.asarray(np.sin(e))
        self.num_ids, self.minus = number_token_ids(200)
        res = np.where(self.op == 0, self.a + self.b, self.a - self.b)
        self.result = res
        self.answer = np.where(res >= 0, self.num_ids[np.clip(res, 0, 200)], self.minus)
        self._masks: dict[int, np.ndarray] = {}

    def masks(self, layer: int, rows: np.ndarray) -> np.ndarray:
        if layer not in self._masks:
            self._masks[layer] = np.load(ATLAS / "masks" / f"L{layer}.npy")
        return self._masks[layer][rows].astype(np.float32)

    def forward(self, rows: np.ndarray, resid: ResidHook | None = None, inner: InnerHook | None = None,
                capture: set[int] | None = None) -> tuple[np.ndarray, dict[int, np.ndarray]]:  # fmt: skip
        """Last-position log-probs (B, vocab) of the prompts `rows`, and the raw streams at the points
        in `capture` (float16, as in the dataset)."""
        outs, caps = [], {t: [] for t in capture or ()}
        for s in range(0, len(rows), CHUNK):
            idx = np.arange(s, min(s + CHUNK, len(rows)))
            n = len(idx)
            idx = np.concatenate([idx, np.full(CHUNK - n, idx[0])])
            lp, cp = self._chunk(rows[idx], idx, resid, inner, capture)
            outs.append(lp[:n])
            for t in cp:
                caps[t].append(cp[t][:n])
        return np.concatenate(outs), {t: np.concatenate(v) for t, v in caps.items()}

    def _chunk(self, rows: np.ndarray, idx: np.ndarray, resid: ResidHook | None, inner: InnerHook | None,
               capture: set[int] | None) -> tuple[np.ndarray, dict[int, np.ndarray]]:  # fmt: skip
        x = jnp.asarray(
            np.stack([[self.embed[int(t)] for t in toks] for toks in self.tokens[rows]]),
            jnp.float32,
        )
        B = len(rows)
        caps: dict[int, np.ndarray] = {}

        def site(li: int, kind: str, xin: jax.Array, rms: jax.Array, m: jax.Array) -> jax.Array:
            s = self.sites[li][kind]
            h = _project(xin, s.V)
            if inner is not None:
                h = inner(li, kind, h, rms, idx)
            return _write(h, m[:, :, s.cols], s.U)

        for li in range(self.n_layer):
            m = jnp.asarray(self.masks(li, rows))
            for t_off in (0, 1):
                t = 2 * li + t_off
                if resid is not None:
                    x = resid(t, x, idx)
                if capture is not None and t in capture:
                    caps[t] = np.asarray(x.astype(jnp.float16))
                rms = _rms(x, self.eps)
                xin = x / rms[..., None] * jnp.asarray(self.ln1[li] if t_off == 0 else self.ln2[li])
                if t_off == 0:
                    q = site(li, "q", xin, rms, m).reshape(B, T, self.n_head, self.hd)
                    k = site(li, "k", xin, rms, m).reshape(B, T, self.n_kv, self.hd)
                    v = site(li, "v", xin, rms, m).reshape(B, T, self.n_kv, self.hd)
                    att = _attention(q, k, v, self.cos, self.sin)
                    x = x + site(li, "o", att, _rms(att, self.eps), m)
                else:
                    g = site(li, "gate", xin, rms, m)
                    u = site(li, "up", xin, rms, m)
                    act = jax.nn.silu(g) * u
                    x = x + site(li, "down", act, _rms(act, self.eps), m)
        if capture is not None and 2 * self.n_layer in capture:
            caps[2 * self.n_layer] = np.asarray(x.astype(jnp.float16))
        return np.asarray(_logprobs(x[:, -1], self.final_j, self.unembed)), caps

    def value(self, tokens: np.ndarray) -> np.ndarray:
        """The number a predicted token stands for: 0..200, -1 for "-", -2 for anything else."""
        lut = {int(t): v for v, t in enumerate(self.num_ids)} | {int(self.minus): -1}
        return np.array([lut.get(int(t), -2) for t in tokens])

    def column(self, layer: int, name: str) -> tuple[str, int]:
        """(kind, column in that site) of the component called `name` (e.g. "L15.gate.c9")."""
        kind = name.split(".")[1]
        return kind, self.sites[layer][kind].names.index(name)


def kl(p_log: np.ndarray, q_log: np.ndarray) -> np.ndarray:
    """KL(p || q) per row, from log-probs."""
    return (np.exp(p_log) * (p_log - q_log)).sum(-1)
