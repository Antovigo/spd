"""Tiny pre-norm transformer for multi-site decomposition at attention scale.

Per-example module (single sequence in, single logits out); the trainer vmaps
over batch. All `Linear` calls are the mask-tolerant shim so substitution swaps
them in place. Per-site masks have shape `[seq, C]` — they're applied
per-position by vmapping the layer call over the sequence axis.

Decomposable sites per block: attn.{q,k,v,o}, mlp.{up,down}. Plus a final
`unembed`. Embedding is not decomposed in v1.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from nano_pd_jax.linear import Linear


def rms_norm(x: Float[Array, "seq d"], eps: float = 1e-6) -> Float[Array, "seq d"]:
    rms = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms


def _apply_linear_over_seq(
    layer: Linear,
    x: Float[Array, "seq d_in"],
    mask: Float[Array, "seq C"] | None,
) -> tuple[Float[Array, "seq d_out"], Float[Array, "seq d_in"]]:
    if mask is None:
        return jax.vmap(lambda t: layer(t, None))(x)
    return jax.vmap(layer)(x, mask)


class Attention(eqx.Module):
    q: Linear
    k: Linear
    v: Linear
    o: Linear
    n_heads: int = eqx.field(static=True)
    d_head: int = eqx.field(static=True)

    def __init__(self, d_model: int, n_heads: int, d_head: int, *, key: PRNGKeyArray):
        assert d_model == n_heads * d_head
        kq, kk, kv, ko = jax.random.split(key, 4)
        self.q = Linear(d_model, n_heads * d_head, use_bias=False, key=kq)
        self.k = Linear(d_model, n_heads * d_head, use_bias=False, key=kk)
        self.v = Linear(d_model, n_heads * d_head, use_bias=False, key=kv)
        self.o = Linear(n_heads * d_head, d_model, use_bias=False, key=ko)
        self.n_heads = n_heads
        self.d_head = d_head

    def __call__(
        self,
        x: Float[Array, "seq d_model"],
        m_q: Float[Array, "seq C"] | None,
        m_k: Float[Array, "seq C"] | None,
        m_v: Float[Array, "seq C"] | None,
        m_o: Float[Array, "seq C"] | None,
    ) -> tuple[
        Float[Array, "seq d_model"],
        dict[str, Float[Array, "seq d_in"]],
    ]:
        seq = x.shape[0]
        q, a_q = _apply_linear_over_seq(self.q, x, m_q)
        k, a_k = _apply_linear_over_seq(self.k, x, m_k)
        v, a_v = _apply_linear_over_seq(self.v, x, m_v)

        q = q.reshape(seq, self.n_heads, self.d_head).transpose(1, 0, 2)
        k = k.reshape(seq, self.n_heads, self.d_head).transpose(1, 0, 2)
        v = v.reshape(seq, self.n_heads, self.d_head).transpose(1, 0, 2)

        scores = jnp.einsum("hqd,hkd->hqk", q, k) / jnp.sqrt(self.d_head)
        causal = jnp.tril(jnp.ones((seq, seq), dtype=bool))
        scores = jnp.where(causal[None, :, :], scores, -jnp.inf)
        attn = jax.nn.softmax(scores, axis=-1)
        ctx = jnp.einsum("hqk,hkd->hqd", attn, v)
        ctx = ctx.transpose(1, 0, 2).reshape(seq, self.n_heads * self.d_head)

        out, a_o = _apply_linear_over_seq(self.o, ctx, m_o)
        return out, {"q": a_q, "k": a_k, "v": a_v, "o": a_o}


class MLP(eqx.Module):
    up: Linear
    down: Linear

    def __init__(self, d_model: int, d_ff: int, *, key: PRNGKeyArray):
        ku, kd = jax.random.split(key)
        self.up = Linear(d_model, d_ff, use_bias=False, key=ku)
        self.down = Linear(d_ff, d_model, use_bias=False, key=kd)

    def __call__(
        self,
        x: Float[Array, "seq d_model"],
        m_up: Float[Array, "seq C"] | None,
        m_down: Float[Array, "seq C"] | None,
    ) -> tuple[
        Float[Array, "seq d_model"],
        dict[str, Float[Array, "seq d_in"]],
    ]:
        h, a_up = _apply_linear_over_seq(self.up, x, m_up)
        h = jax.nn.gelu(h)
        out, a_down = _apply_linear_over_seq(self.down, h, m_down)
        return out, {"up": a_up, "down": a_down}


class Block(eqx.Module):
    attn: Attention
    mlp: MLP

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_head: int,
        d_ff: int,
        *,
        key: PRNGKeyArray,
    ):
        ka, km = jax.random.split(key)
        self.attn = Attention(d_model, n_heads, d_head, key=ka)
        self.mlp = MLP(d_model, d_ff, key=km)

    def __call__(
        self,
        x: Float[Array, "seq d_model"],
        masks: dict[str, Float[Array, "seq C"]],
        prefix: str,
    ) -> tuple[
        Float[Array, "seq d_model"],
        dict[str, Float[Array, "seq d_in"]],
    ]:
        attn_out, a_attn = self.attn(
            rms_norm(x),
            masks.get(f"{prefix}.attn.q"),
            masks.get(f"{prefix}.attn.k"),
            masks.get(f"{prefix}.attn.v"),
            masks.get(f"{prefix}.attn.o"),
        )
        x = x + attn_out
        mlp_out, a_mlp = self.mlp(
            rms_norm(x),
            masks.get(f"{prefix}.mlp.up"),
            masks.get(f"{prefix}.mlp.down"),
        )
        x = x + mlp_out
        acts = {
            f"{prefix}.attn.q": a_attn["q"],
            f"{prefix}.attn.k": a_attn["k"],
            f"{prefix}.attn.v": a_attn["v"],
            f"{prefix}.attn.o": a_attn["o"],
            f"{prefix}.mlp.up": a_mlp["up"],
            f"{prefix}.mlp.down": a_mlp["down"],
        }
        return x, acts


class TinyTransformer(eqx.Module):
    embed: eqx.nn.Embedding
    blocks: tuple[Block, ...]
    unembed: Linear
    n_layers: int = eqx.field(static=True)

    def __init__(
        self,
        vocab: int,
        d_model: int,
        n_heads: int,
        d_head: int,
        d_ff: int,
        n_layers: int,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, n_layers + 2)
        self.embed = eqx.nn.Embedding(vocab, d_model, key=keys[0])
        self.blocks = tuple(
            Block(d_model, n_heads, d_head, d_ff, key=keys[i + 1]) for i in range(n_layers)
        )
        self.unembed = Linear(d_model, vocab, use_bias=False, key=keys[-1])
        self.n_layers = n_layers

    def __call__(
        self,
        tokens: Int[Array, " seq"],
        masks: dict[str, Float[Array, "seq C"]] | None = None,
    ) -> tuple[
        Float[Array, "seq vocab"],
        dict[str, Float[Array, "seq d_in"]],
    ]:
        m = masks if masks is not None else {}
        x = jax.vmap(self.embed)(tokens)
        acts: dict[str, Float[Array, "seq d_in"]] = {}
        for i, block in enumerate(self.blocks):
            x, block_acts = block(x, m, f"blocks.{i}")
            acts.update(block_acts)
        x = rms_norm(x)
        logits, a_unembed = _apply_linear_over_seq(self.unembed, x, m.get("unembed"))
        acts["unembed"] = a_unembed
        return logits, acts
