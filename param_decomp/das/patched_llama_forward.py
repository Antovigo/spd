"""Llama-3.1-8B resumable at any stream point, with the DAS interchange patch.

Stream point `t`: `t = 2l` is the raw residual stream before block `l`'s attention norm, `t = 2l + 1`
before its MLP norm. Everything after `t` reads only the stream at `t`.

The patch at positions 1..4 is `h <- h + Q m Qᵀ (h_source - h)`, with `Q` `(4, d, C)` orthonormal
columns per position and `m` `(4, C)` a 0/1 mask. A position's subspace is its masked columns, so array
shapes, and therefore compiled functions, do not change as dimensions are removed."""

import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int
from safetensors import safe_open

from param_decomp.arith_repr.autointerp.llama_weights import llama3_inv_freq
from param_decomp.pruning.model import (
    Arch,
    Shared,
    _block,
    _dense_site,
    attention_half,
    embed,
    last_logits,
    mlp_half,
)

PRECISION = "bf16"
N_PATCHED = 4
"""Positions 1..4; position 0 is `<BOS>`, identical in every prompt."""
SITE_NAMES = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}


class Llama(eqx.Module):
    arch: Arch
    shared: Shared
    blocks: tuple[dict[str, Array], ...]
    """Each block's site matrices, torch layout `(d_out, d_in)`."""


def load_llama(weights_dir: Path) -> Llama:
    """The target's weights from its HF safetensors shards, without any decomposition."""
    config = json.loads((weights_dir / "config.json").read_text())
    index = json.loads((weights_dir / "model.safetensors.index.json").read_text())["weight_map"]

    def get(name: str) -> Array:
        with safe_open(str(weights_dir / index[name]), framework="flax") as shards:
            return jnp.asarray(np.asarray(shards.get_tensor(name), np.float32), jnp.bfloat16)

    n_layer = config["num_hidden_layers"]
    layers = range(n_layer)
    arch = Arch(
        n_layer=n_layer,
        n_head=config["num_attention_heads"],
        n_kv_head=config["num_key_value_heads"],
        head_dim=config["hidden_size"] // config["num_attention_heads"],
        eps=config["rms_norm_eps"],
    )
    shared = Shared(
        embed=get("model.embed_tokens.weight"),
        ln1=jnp.stack([get(f"model.layers.{li}.input_layernorm.weight") for li in layers]),
        ln2=jnp.stack([get(f"model.layers.{li}.post_attention_layernorm.weight") for li in layers]),
        norm=get("model.norm.weight"),
        head=get("lm_head.weight"),
        inv_freq=jnp.asarray(llama3_inv_freq(config)),
    )
    blocks = tuple(
        {kind: get(f"model.layers.{li}.{name}.weight") for kind, name in SITE_NAMES.items()}
        for li in layers
    )
    return Llama(arch, shared, blocks)


def stream_at(model: Llama, tokens: Int[Array, "B T"], t: int) -> Float[Array, "B T d"]:
    """The raw residual stream at stream point `t`."""
    arch, shared = model.arch, model.shared
    resid = embed(shared, tokens)
    for block in range(t // 2):
        site = _dense_site(model.blocks[block], PRECISION)
        resid = _block(
            arch, shared.ln1[block], shared.ln2[block], shared.inv_freq, resid, site, PRECISION
        )
    if t % 2 == 1:
        block = t // 2
        site = _dense_site(model.blocks[block], PRECISION)
        resid = attention_half(arch, shared.ln1[block], shared.inv_freq, resid, site, PRECISION)
    return resid


def logits_from(model: Llama, resid: Float[Array, "B T d"], t: int) -> Float[Array, "B vocab"]:
    """Last-position logits from the stream at `t`, differentiable in `resid`. A plain loop over the
    blocks, because the reverse pass of a `scan` over stacked weights keeps a copy of them (26 GiB)."""
    arch, shared = model.arch, model.shared
    start = (t + 1) // 2
    if t % 2 == 1:
        block = t // 2
        resid = mlp_half(
            arch, shared.ln2[block], resid, _dense_site(model.blocks[block], PRECISION)
        )
    for block in range(start, arch.n_layer):
        site = _dense_site(model.blocks[block], PRECISION)
        resid = _block(
            arch, shared.ln1[block], shared.ln2[block], shared.inv_freq, resid, site, PRECISION
        )
    return last_logits(arch, shared, resid, PRECISION)


def patch_stream(
    base: Float[Array, "B T d"],
    source: Float[Array, "B T d"],
    Q: Float[Array, "4 d C"],
    mask: Float[Array, "4 C"],
) -> Float[Array, "B T d"]:
    """`base` with its masked subspace at positions 1..4 replaced by `source`'s."""
    difference = source[:, 1:] - base[:, 1:]
    coordinates = jnp.einsum("bpd,pdc->bpc", difference, Q) * mask
    return base.at[:, 1:].add(jnp.einsum("bpc,pdc->bpd", coordinates, Q))
