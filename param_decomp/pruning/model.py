"""The submodel `M_S` of a decomposition as a plain Llama forward in FACTORED form.

Every linear site of every block is replaced by the sum of its components, `y = ((x V) ⊙ m) U`,
with `V` `(d_in, C)` the components' input vectors as columns, `U` `(C, d_out)` their output
vectors as rows and `m` `(C,)` one global mask entry per component (the weight delta is OFF).
Embedding, norms, attention and unembedding are the target's. The residual stream, norms,
attention and softmax run in fp32; the site and head matmuls take a `Precision` (`fp32` exact,
`tf32`, or `bf16` inputs with fp32 accumulation).

Only the components of a fixed candidate set `S₀` are stored (`Factors`), per kind padded to the
widest layer; a padding slot has zero `V`/`U` and is never on. On top of the global mask a batch
row may FLIP one component (`Flips`): remove an on component or add an off one, so one forward
scores many single-component changes (the screen) or one change over many prompts.

`run_blocks` starts at any block from a cached block-input residual (`block_inputs`), so a flip in
block `l` reruns blocks `l..L-1` only. `DenseWeights` is the target model itself (Llama), run
through the same block equations as the reference distribution."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from param_decomp.core.components import ComponentStacks
from param_decomp.target_ports.llama import apply_rope, repeat_kv, rope_cos_sin
from param_decomp.targets.transformer import (
    KIND_ORDER,
    GatedMLP,
    TiedHead,
    TransformerDecomposedModel,
    parse_site_name,
    site_name,
)

type Precision = Literal["fp32", "tf32", "bf16"]

KIND_ID = {kind: i for i, kind in enumerate(KIND_ORDER)}
NO_FLIP = -1
"""`Flips.kind` of a row that flips nothing."""


def matmul(x: Array, w: Array, precision: Precision) -> Float[Array, "..."]:
    """`x @ w` with fp32 output."""
    match precision:
        case "fp32":
            return jnp.matmul(
                x.astype(jnp.float32), w.astype(jnp.float32), precision=jax.lax.Precision.HIGHEST
            )
        case "tf32":
            return jnp.matmul(
                x.astype(jnp.float32), w.astype(jnp.float32), precision=jax.lax.Precision.DEFAULT
            )
        case "bf16":
            return jnp.matmul(
                x.astype(jnp.bfloat16), w.astype(jnp.bfloat16), preferred_element_type=jnp.float32
            )


class Arch(eqx.Module):
    n_layer: int = eqx.field(static=True)
    n_head: int = eqx.field(static=True)
    n_kv_head: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)


class Shared(eqx.Module):
    """The target's non-site weights, shared by `M_S` and the dense target."""

    embed: Float[Array, "vocab d"]
    ln1: Float[Array, "L d"]
    ln2: Float[Array, "L d"]
    norm: Float[Array, " d"]
    head: Float[Array, "vocab d"]
    inv_freq: Float[Array, " hd2"]


class Factors(eqx.Module):
    """`S₀`'s components per kind, stacked over layers and padded to the widest layer."""

    V: dict[str, Float[Array, "L d_in C"]]
    U: dict[str, Float[Array, "L C d_out"]]


class DenseWeights(eqx.Module):
    """The target's site matrices, torch layout `(d_out, d_in)`, stacked over layers."""

    W: dict[str, Float[Array, "L d_out d_in"]]


type Mask = dict[str, Float[Array, "L C"]]
"""Global mask per kind: 1.0 on, 0.0 off; padding slots are always 0."""


class Flips(eqx.Module):
    """Per batch row, at most one component changed on top of the global mask: the component at
    `(kind, layer, slot)` gets `sign` added to its mask (`-1` removes an on component, `+1` adds
    an off one). `kind == NO_FLIP` leaves the row alone."""

    kind: Int[Array, " B"]
    layer: Int[Array, " B"]
    slot: Int[Array, " B"]
    sign: Float[Array, " B"]

    @staticmethod
    def none(batch: int) -> "Flips":
        zeros = jnp.zeros((batch,), jnp.int32)
        return Flips(kind=zeros + NO_FLIP, layer=zeros, slot=zeros, sign=jnp.zeros((batch,)))

    @staticmethod
    def same(batch: int, kind: Array, layer: Array, slot: Array, sign: Array) -> "Flips":
        """Every row flips the same component (scalars broadcast over the batch)."""
        ones = jnp.ones((batch,), jnp.int32)
        return Flips(
            kind=ones * kind, layer=ones * layer, slot=ones * slot, sign=jnp.ones((batch,)) * sign
        )


@dataclass(frozen=True)
class Slots:
    """Where every component of `S₀` sits in `Factors`: the canonical component order is the
    sites sorted by name, then component index ascending (the ci_filter dataset's order)."""

    sites: tuple[str, ...]
    """`(A,)` site name of each component."""
    index: np.ndarray
    """`(A,)` int: the component's index within its site's decomposition."""
    layer: np.ndarray
    kind: np.ndarray
    """`(A,)` int: `KIND_ID` of the site's kind."""
    slot: np.ndarray
    """`(A,)` int: its column in `Factors.V[kind][layer]`."""
    widths: dict[str, int]
    """Padded `C` per kind."""

    @property
    def n(self) -> int:
        return int(self.index.size)

    def mask(self, on: np.ndarray, n_layer: int) -> Mask:
        """The global mask of the components with `on[a]` true (`on` in canonical order)."""
        out = {kind: np.zeros((n_layer, width), np.float32) for kind, width in self.widths.items()}
        for kind, kind_id in KIND_ID.items():
            sel = (self.kind == kind_id) & on
            out[kind][self.layer[sel], self.slot[sel]] = 1.0
        return {kind: jnp.asarray(v) for kind, v in out.items()}

    def gather(self, per_kind: Mapping[str, Array | np.ndarray]) -> np.ndarray:
        """`(A,)` values of a per-kind `(L, C)` array (e.g. a mask gradient), canonical order."""
        out = np.zeros(self.n, np.float64)
        for kind, kind_id in KIND_ID.items():
            sel = self.kind == kind_id
            values = np.asarray(per_kind[kind])
            out[sel] = values[self.layer[sel], self.slot[sel]]
        return out


def slots_of(alive: Mapping[str, np.ndarray]) -> Slots:
    """The `Slots` of `{site: (C,) bool}`: every site of every layer must be present."""
    sites: list[str] = []
    index: list[int] = []
    layer: list[int] = []
    kind: list[int] = []
    slot: list[int] = []
    widths = dict.fromkeys(KIND_ORDER, 1)
    for site in sorted(alive):
        site_layer, site_kind = parse_site_name(site)
        ids = np.nonzero(alive[site])[0]
        widths[site_kind] = max(widths[site_kind], int(ids.size))
        sites.extend([site] * ids.size)
        index.extend(int(i) for i in ids)
        layer.extend([site_layer] * ids.size)
        kind.extend([KIND_ID[site_kind]] * ids.size)
        slot.extend(range(ids.size))
    return Slots(
        sites=tuple(sites),
        index=np.asarray(index, np.int64),
        layer=np.asarray(layer, np.int64),
        kind=np.asarray(kind, np.int64),
        slot=np.asarray(slot, np.int64),
        widths=widths,
    )


def build(
    model: TransformerDecomposedModel,
    components: ComponentStacks,
    alive: Mapping[str, np.ndarray],
) -> tuple[Arch, Shared, Factors, DenseWeights, Slots]:
    """The factored `S₀` model, the dense target and the slot table, from a restored
    decomposition. Every block must be decomposed at every site (an all-layers run)."""
    n_layer = model.n_layer
    expected = {site_name(layer, kind) for layer in range(n_layer) for kind in KIND_ORDER}
    assert set(model.site_names) == expected, "every site of every block must be decomposed"
    assert set(alive) == expected, sorted(expected - set(alive))[:5]
    assert model.stacked_prefix is None and model.stacked_tail is None, "split model"
    stacked = model.stacked
    assert isinstance(stacked.mlp, GatedMLP), "gated MLP (Llama) only"
    attn = stacked.attn
    arch = Arch(
        n_layer=n_layer,
        n_head=attn.n_head,
        n_kv_head=attn.n_kv_head,
        head_dim=attn.head_dim,
        eps=model.eps,
    )
    head = model.embed if isinstance(model.lm_head, TiedHead) else model.lm_head
    shared = Shared(
        embed=model.embed,
        ln1=stacked.ln1,
        ln2=stacked.ln2,
        norm=model.norm,
        head=head,
        inv_freq=model.inv_freq,
    )
    dense = DenseWeights(
        W={
            "q": attn.wq,
            "k": attn.wk,
            "v": attn.wv,
            "o": attn.wo,
            "gate": stacked.mlp.Wg,
            "up": stacked.mlp.Wu,
            "down": stacked.mlp.Wd,
        }
    )
    # Off the restored run's explicit (fsdp/tp-typed) mesh: every pruning array is a plain
    # single-device array, so the engine's gathers need no sharding annotations.
    device = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    shared, dense = jax.device_put((shared, dense), device)
    slots = slots_of(alive)
    V: dict[str, Array] = {}
    U: dict[str, Array] = {}
    for kind in KIND_ORDER:
        width = slots.widths[kind]
        v_layers, u_layers = [], []
        for layer in range(n_layer):
            site = site_name(layer, kind)
            ids = jnp.asarray(np.nonzero(alive[site])[0])
            factors = components.site(site)
            v = jnp.take(jax.device_put(factors.V, device).astype(jnp.float32), ids, axis=1)
            u = jnp.take(jax.device_put(factors.U, device).astype(jnp.float32), ids, axis=0)
            pad = width - int(ids.size)
            v_layers.append(jnp.pad(v, ((0, 0), (0, pad))))
            u_layers.append(jnp.pad(u, ((0, pad), (0, 0))))
        V[kind] = jnp.stack(v_layers)
        U[kind] = jnp.stack(u_layers)
    return arch, shared, Factors(V=V, U=U), dense, slots


# ----------------------------------- forward -----------------------------------


def rms_norm32(x: Float[Array, "... d"], weight: Array, eps: float) -> Float[Array, "... d"]:
    x = x.astype(jnp.float32)
    var = jnp.mean(x * x, axis=-1, keepdims=True)
    return weight.astype(jnp.float32) * (x * jax.lax.rsqrt(var + eps))


type LayerXs = tuple[Array, Array, Array, dict[str, Array], dict[str, Array], dict[str, Array]]
"""Per-block scan inputs: block index, the two norms, `V`, `U` and the mask per kind."""
type SiteFn = Callable[[str, Float[Array, "B T d_in"]], Float[Array, "B T d_out"]]


def _block(
    arch: Arch,
    ln1: Array,
    ln2: Array,
    inv_freq: Array,
    resid: Float[Array, "B T d"],
    site: SiteFn,
    attention_precision: Precision,
) -> Float[Array, "B T d"]:
    """One Llama block in fp32; `site(kind, x)` computes each linear site."""
    b, t, _ = resid.shape
    h = rms_norm32(resid, ln1, arch.eps)
    q = site("q", h).reshape(b, t, arch.n_head, arch.head_dim).transpose(0, 2, 1, 3)
    k = site("k", h).reshape(b, t, arch.n_kv_head, arch.head_dim).transpose(0, 2, 1, 3)
    v = site("v", h).reshape(b, t, arch.n_kv_head, arch.head_dim).transpose(0, 2, 1, 3)
    positions = jnp.broadcast_to(jnp.arange(t, dtype=jnp.int32), (b, t))
    cos, sin = rope_cos_sin(inv_freq, positions, jnp.float32)
    q, k = apply_rope(q, k, cos, sin)
    n_rep = arch.n_head // arch.n_kv_head
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)
    hp = jax.lax.Precision.HIGHEST if attention_precision == "fp32" else None
    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k, precision=hp) / arch.head_dim**0.5
    causal = jnp.tril(jnp.ones((t, t), bool))
    weights = jax.nn.softmax(jnp.where(causal, scores, -jnp.inf), axis=-1)
    attended = jnp.einsum("bhqk,bhkd->bhqd", weights, v, precision=hp)
    attended = attended.transpose(0, 2, 1, 3).reshape(b, t, arch.n_head * arch.head_dim)
    resid = resid + site("o", attended)
    h = rms_norm32(resid, ln2, arch.eps)
    hidden = jax.nn.silu(site("gate", h)) * site("up", h)
    return resid + site("down", hidden)


def _factored_site(
    V: Mapping[str, Array],
    U: Mapping[str, Array],
    mask: Mapping[str, Array],
    flips: Flips,
    layer: Array,
    precision: Precision,
) -> SiteFn:
    """`site(kind, x) = ((x V) ⊙ (m + flip)) U` at block `layer` (`V`, `U`, `mask` already
    sliced to that block)."""

    def site(kind: str, x: Array) -> Array:
        width = V[kind].shape[-1]
        here = (flips.kind == KIND_ID[kind]) & (flips.layer == layer)
        flip = (
            jax.nn.one_hot(flips.slot, width, dtype=jnp.float32)
            * jnp.where(here, flips.sign, 0.0)[:, None]
        )
        coeff = mask[kind][None, :] + flip  # (B, C)
        xv = matmul(x, V[kind], precision)
        return matmul(xv * coeff[:, None, :], U[kind], precision)

    return site


def _dense_site(W: Mapping[str, Array], precision: Precision) -> SiteFn:
    def site(kind: str, x: Array) -> Array:
        return matmul(x, W[kind].T, precision)

    return site


def embed(shared: Shared, tokens: Int[Array, "B T"]) -> Float[Array, "B T d"]:
    return shared.embed[tokens].astype(jnp.float32)


def last_logits(
    arch: Arch, shared: Shared, resid: Float[Array, "B T d"], precision: Precision
) -> Float[Array, "B vocab"]:
    """The unembedding at the last position only."""
    h = rms_norm32(resid[:, -1, :], shared.norm, arch.eps)
    return matmul(h, shared.head.T, precision)


def run_blocks(
    arch: Arch,
    shared: Shared,
    factors: Factors,
    mask: Mask,
    flips: Flips,
    resid: Float[Array, "B T d"],
    start: Int[Array, ""] | int,
    precision: Precision,
) -> Float[Array, "B T d"]:
    """Blocks `start..L-1` of `M_S` from the residual entering block `start` (forward only: the
    trip count may be traced)."""

    def body(layer: Array, resid: Array) -> Array:
        take = lambda a: jax.lax.dynamic_index_in_dim(a, layer, keepdims=False)  # noqa: E731
        site = _factored_site(
            {k: take(v) for k, v in factors.V.items()},
            {k: take(v) for k, v in factors.U.items()},
            {k: take(v) for k, v in mask.items()},
            flips,
            layer,
            precision,
        )
        return _block(
            arch, take(shared.ln1), take(shared.ln2), shared.inv_freq, resid, site, precision
        )

    return jax.lax.fori_loop(start, arch.n_layer, body, resid)


def factored_logits(
    arch: Arch,
    shared: Shared,
    factors: Factors,
    mask: Mask,
    flips: Flips,
    tokens: Int[Array, "B T"],
    site_precision: Precision,
    head_precision: Precision,
) -> Float[Array, "B vocab"]:
    """`M_S`'s last-position logits from the tokens, differentiable in `mask` (a `scan`)."""

    def body(resid: Array, xs: LayerXs) -> tuple[Array, None]:
        layer, ln1, ln2, V, U, m = xs
        site = _factored_site(V, U, m, flips, layer, site_precision)
        return _block(arch, ln1, ln2, shared.inv_freq, resid, site, site_precision), None

    xs = (
        jnp.arange(arch.n_layer),
        shared.ln1,
        shared.ln2,
        factors.V,
        factors.U,
        mask,
    )
    resid, _ = jax.lax.scan(body, embed(shared, tokens), xs)
    return last_logits(arch, shared, resid, head_precision)


def block_inputs(
    arch: Arch,
    shared: Shared,
    factors: Factors,
    mask: Mask,
    tokens: Int[Array, "B T"],
    precision: Precision,
) -> tuple[Float[Array, "L B T d"], Float[Array, "B T d"]]:
    """The residual entering every block of `M_S` (no flips) and the final residual."""
    flips = Flips.none(tokens.shape[0])

    def body(resid: Array, xs: LayerXs) -> tuple[Array, Array]:
        layer, ln1, ln2, V, U, m = xs
        site = _factored_site(V, U, m, flips, layer, precision)
        return _block(arch, ln1, ln2, shared.inv_freq, resid, site, precision), resid

    xs = (jnp.arange(arch.n_layer), shared.ln1, shared.ln2, factors.V, factors.U, mask)
    final, inputs = jax.lax.scan(body, embed(shared, tokens), xs)
    return inputs, final


def dense_logits(
    arch: Arch,
    shared: Shared,
    dense: DenseWeights,
    tokens: Int[Array, "B T"],
    site_precision: Precision,
    head_precision: Precision,
) -> Float[Array, "B vocab"]:
    """The target model's last-position logits through the same block equations."""

    def body(resid: Array, xs: tuple[Array, Array, dict[str, Array]]) -> tuple[Array, None]:
        ln1, ln2, W = xs
        site = _dense_site(W, site_precision)
        return _block(arch, ln1, ln2, shared.inv_freq, resid, site, site_precision), None

    resid, _ = jax.lax.scan(body, embed(shared, tokens), (shared.ln1, shared.ln2, dense.W))
    return last_logits(arch, shared, resid, head_precision)


def on_mask(mask: Mask, slots: Slots) -> Bool[np.ndarray, " A"]:
    """`(A,)` true where the global mask has the component on."""
    return slots.gather(mask) > 0.5
