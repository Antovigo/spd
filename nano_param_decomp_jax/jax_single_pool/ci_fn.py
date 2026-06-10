"""The `global_shared_transformer` CI fn for the Llama-8B target.

ONE shared transformer over ALL decomposed sites (matching torch
`GlobalSharedTransformerCiFn`): the per-site clean inputs (3 per decomposed layer:
gate_in / up_in / down_in) are rms-normed, concatenated along the feature dim,
projected to `d_model`; a stack of bidirectional-RoPE transformer blocks then a head
emits `3*n_layers*C` logits. Following torch (`sigmoid_type="leaky_hard"`), the SAME
logits are squashed two ways: a **lower-leaky-hard** sigmoid feeds the recon / PPGD
component masks (bounded above by 1), and an **upper-leaky-hard** sigmoid feeds the
importance-minimality penalty (bounded below by 0). Both are returned as
`{kind: (b, t, L, C)}` (a `CIValues` pair) so the step can mask layer i's `kind` site
with `lower[kind][:, :, i]` and penalize `upper[kind][:, :, i]`.

Mirrors the torch `GlobalSharedTransformerCiConfig` used in the llama8b configs
(d_model 4096, 4 blocks, 64 heads, mlp 16384).
"""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from vendored_jax.llama import apply_rope, rms_norm, rope_cos_sin

from jax_single_pool.llama8b import DT, KINDS


class CIValues(NamedTuple):
    """The two squashed views of the CI-fn logits (torch `CIOutputs`, sans pre_sigmoid).

    `lower` (lower-leaky-hard) gates component contributions in recon / PPGD;
    `upper` (upper-leaky-hard) is penalized by importance-minimality. Each is
    `{kind: (b, t, L, C)}`.
    """

    lower: dict[str, Array]
    upper: dict[str, Array]


@jax.custom_vjp
def lower_leaky_hard_sigmoid(x):
    return jnp.clip(x, 0.0, 1.0)


def _lhs_f(x):
    return jnp.clip(x, 0.0, 1.0), x


def _lhs_b(x, g):
    leak = jnp.where(g < 0, 0.01 * g, 0.0)
    return (jnp.where(x <= 0, leak, jnp.where(x <= 1, g, 0.0)),)


lower_leaky_hard_sigmoid.defvjp(_lhs_f, _lhs_b)


def upper_leaky_hard_sigmoid(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """`x>1 ? 1+alpha*(x-1) : clamp(x,0,1)` — torch `upper_leaky_hard_sigmoid`.

    Ordinary (non-custom-vjp) op: the autodiff of this exact forward matches torch's,
    which here builds its backward from the same `where` expression rather than a
    custom Function."""
    alpha = 0.01
    return jnp.where(x > 1, 1 + alpha * (x - 1), jnp.clip(x, 0.0, 1.0))


class CIBlock(eqx.Module):
    ln1: Array
    ln2: Array
    wq: Array
    wk: Array
    wv: Array
    wo: Array
    w1: Array
    w2: Array
    n_head: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __call__(self, x: Float[Array, "b t d"], inv_freq: Array) -> Array:
        b, t, d = x.shape
        h = rms_norm(x, self.ln1, self.eps)
        q = (h @ self.wq.T).reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = (h @ self.wk.T).reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = (h @ self.wv.T).reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        cos, sin = rope_cos_sin(inv_freq, t, x.dtype)
        q, k = apply_rope(q, k, cos, sin)
        qt, kt, vt = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        y = jax.nn.dot_product_attention(qt, kt, vt, is_causal=False)  # bidirectional
        y = y.reshape(b, t, d)
        x = x + y @ self.wo.T
        h = rms_norm(x, self.ln2, self.eps)
        return x + (jax.nn.gelu(h @ self.w1) @ self.w2)


class CIFn(eqx.Module):
    in_proj: Float[Array, "total_in d_model"]
    blocks: list  # CIBlock
    out_head: Float[Array, "d_model total_c"]
    inv_freq: Array
    C: int = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def __call__(self, site_inputs: list) -> CIValues:
        """`site_inputs`: flat list of 3*n_layers clean inputs in (layer, kind) order.

        Returns a `CIValues(lower, upper)` pair, each `{kind: (b, t, n_layers, C)}` —
        the two squashings of the SAME logits (torch's `lower_leaky` / `upper_leaky`)."""
        assert len(site_inputs) == 3 * self.n_layers, (
            f"expected {3 * self.n_layers} site inputs, got {len(site_inputs)}"
        )
        normed = [rms_norm(s, jnp.ones((s.shape[-1],), DT), self.eps) for s in site_inputs]
        x = jax.nn.relu(jnp.concatenate(normed, axis=-1) @ self.in_proj)
        for blk in self.blocks:
            x = blk(x, self.inv_freq)
        flat = x @ self.out_head  # (b, t, 3*n_layers*C)
        b, t, _ = flat.shape
        # logits are laid out site-major in (layer, kind) order — reshape to (b,t,L,3,C)
        per_site = flat.reshape(b, t, self.n_layers, len(KINDS), self.C)
        lower = lower_leaky_hard_sigmoid(per_site)
        upper = upper_leaky_hard_sigmoid(per_site)
        return CIValues(
            lower={kind: lower[:, :, :, j] for j, kind in enumerate(KINDS)},
            upper={kind: upper[:, :, :, j] for j, kind in enumerate(KINDS)},
        )


class CIFnDims(NamedTuple):
    d_model: int
    n_blocks: int
    n_heads: int
    mlp_hidden: int
    total_in: int
    C: int
    n_layers: int


def init_ci_fn(dims: CIFnDims, key) -> CIFn:
    ks = iter(jax.random.split(key, dims.n_blocks * 8 + 4))
    hd = dims.d_model // dims.n_heads

    def n(shape, s):
        return (jax.random.normal(next(ks), shape) * s).astype(DT)

    def block() -> CIBlock:
        return CIBlock(
            ln1=jnp.ones((dims.d_model,), DT),
            ln2=jnp.ones((dims.d_model,), DT),
            wq=n((dims.d_model, dims.d_model), dims.d_model**-0.5),
            wk=n((dims.d_model, dims.d_model), dims.d_model**-0.5),
            wv=n((dims.d_model, dims.d_model), dims.d_model**-0.5),
            wo=n((dims.d_model, dims.d_model), dims.d_model**-0.5),
            w1=n((dims.d_model, dims.mlp_hidden), dims.d_model**-0.5),
            w2=n((dims.mlp_hidden, dims.d_model), dims.mlp_hidden**-0.5),
            n_head=dims.n_heads,
            head_dim=hd,
            eps=1e-5,
        )

    inv_freq = 1.0 / (10000.0 ** (jnp.arange(0, hd, 2, dtype=jnp.float32) / hd))
    total_c = len(KINDS) * dims.n_layers * dims.C
    return CIFn(
        in_proj=n((dims.total_in, dims.d_model), dims.total_in**-0.5),
        blocks=[block() for _ in range(dims.n_blocks)],
        out_head=n((dims.d_model, total_c), dims.d_model**-0.5),
        inv_freq=inv_freq,
        C=dims.C,
        n_layers=dims.n_layers,
        eps=1e-5,
    )
