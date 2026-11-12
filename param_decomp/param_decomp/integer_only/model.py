"""A decomposition's model under one GLOBAL component mask, as a plain Llama forward in FACTORED
form, and the dense target through the same block equations.

Every linear site of every block is the sum of its candidate components, `y = ((x V) ⊙ m) U`,
with `x` `(B, T, d_in)` the site input, `V` `(d_in, C)` the components' input vectors as
columns, `U` `(C, d_out)` their output vectors as rows and `m` `(C,)` one mask entry per
component, shared by every prompt and position (the weight delta is OFF). `x V_c` is component
`c`'s INNER activation. Embedding, norms, attention and unembedding are the target's. The
residual stream, norms, attention and softmax run in fp32; the site and head matmuls take a
`Precision` (`fp32` exact, `tf32`, or `bf16` inputs with fp32 accumulation).

Only the components of a fixed set (`Slots`, e.g. the candidates) are stored (`Factors`), per
kind padded to the widest block; a padding slot has zero `V`/`U` and mask 0. Every component
outside the set is permanently off."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from param_decomp.core.components import ComponentStacks
from param_decomp.integer_only.config import Precision
from param_decomp.target_ports.llama import apply_rope, repeat_kv, rope_cos_sin
from param_decomp.targets.transformer import (
    KIND_ORDER,
    GatedMLP,
    TiedHead,
    TransformerDecomposedModel,
    parse_site_name,
    site_name,
)

KIND_ID = {kind: i for i, kind in enumerate(KIND_ORDER)}


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
    """The target's non-site weights, shared by the factored model and the dense target."""

    embed: Float[Array, "vocab d"]
    ln1: Float[Array, "L d"]
    ln2: Float[Array, "L d"]
    norm: Float[Array, " d"]
    head: Float[Array, "vocab d"]
    inv_freq: Float[Array, " hd2"]


class Factors(eqx.Module):
    """The stored components per kind, stacked over blocks and padded to the widest block."""

    V: dict[str, Float[Array, "L d_in W"]]
    U: dict[str, Float[Array, "L W d_out"]]


class DenseWeights(eqx.Module):
    """The target's site matrices, torch layout `(d_out, d_in)`, stacked over blocks."""

    W: dict[str, Float[Array, "L d_out d_in"]]


type Mask = dict[str, Float[Array, "L W"]]
"""Global mask per kind: 1.0 on, 0.0 off; padding slots are always 0."""


@dataclass(frozen=True)
class Slots:
    """Where every stored component sits in `Factors`. The canonical component order (index
    `j`, `0 <= j < n`) is the sites sorted by name, then component index ascending."""

    sites: tuple[str, ...]
    """`(n,)` site name of each component."""
    index: np.ndarray
    """`(n,)` int: the component's index within its site's decomposition."""
    layer: np.ndarray
    kind: np.ndarray
    """`(n,)` int: `KIND_ID` of the site's kind."""
    slot: np.ndarray
    """`(n,)` int: its column in `Factors.V[kind][layer]`."""
    widths: dict[str, int]
    """Padded width `W` per kind."""

    @property
    def n(self) -> int:
        return int(self.index.size)

    def mask(self, on: np.ndarray, n_layer: int) -> Mask:
        """The global mask with component `j` on iff `on[j]` (`on` in canonical order)."""
        assert on.shape == (self.n,), (on.shape, self.n)
        out = {kind: np.zeros((n_layer, width), np.float32) for kind, width in self.widths.items()}
        for kind, kind_id in KIND_ID.items():
            sel = (self.kind == kind_id) & on
            out[kind][self.layer[sel], self.slot[sel]] = 1.0
        return {kind: jnp.asarray(v) for kind, v in out.items()}

    def gather(self, per_kind: Mapping[str, Array | np.ndarray]) -> np.ndarray:
        """`(n,)` values of a per-kind `(L, W)` array (e.g. a mask gradient), canonical order."""
        out = np.zeros(self.n, np.float64)
        for kind, kind_id in KIND_ID.items():
            sel = self.kind == kind_id
            values = np.asarray(per_kind[kind])
            out[sel] = values[self.layer[sel], self.slot[sel]]
        return out

    def components(self) -> dict[str, list[int]]:
        """`{site: [component indices]}` of the stored components."""
        out: dict[str, list[int]] = {}
        for site, c in zip(self.sites, self.index.tolist(), strict=True):
            out.setdefault(site, []).append(int(c))
        return out


def slots_of(keep: Mapping[str, np.ndarray]) -> Slots:
    """The `Slots` of `{site: (C,) bool}`, every site of every block present."""
    sites: list[str] = []
    index: list[int] = []
    layer: list[int] = []
    kind: list[int] = []
    slot: list[int] = []
    widths = dict.fromkeys(KIND_ORDER, 1)
    for site in sorted(keep):
        site_layer, site_kind = parse_site_name(site)
        ids = np.flatnonzero(keep[site])
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


def build_target(model: TransformerDecomposedModel) -> tuple[Arch, Shared, DenseWeights]:
    """The target's architecture, shared weights and site matrices, on the process's first
    device. Every block must be decomposed at every site (an all-blocks run)."""
    n_layer = model.n_layer
    expected = {site_name(layer, kind) for layer in range(n_layer) for kind in KIND_ORDER}
    assert set(model.site_names) == expected, "every site of every block must be decomposed"
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
    # Off the restored run's explicit (fsdp/tp-typed) mesh: every array here is a plain
    # single-device array, so the passes need no sharding annotations.
    device = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    shared, dense = jax.device_put((shared, dense), device)
    return arch, shared, dense


def build_factors(
    components: ComponentStacks, keep: Mapping[str, np.ndarray], n_layer: int
) -> tuple[Factors, Slots]:
    """The fp32 factors of the components `keep` selects (`{site: (C,) bool}`), single device."""
    device = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    slots = slots_of(keep)
    V: dict[str, Array] = {}
    U: dict[str, Array] = {}
    for kind in KIND_ORDER:
        width = slots.widths[kind]
        v_layers, u_layers = [], []
        for layer in range(n_layer):
            site = site_name(layer, kind)
            ids = jnp.asarray(np.flatnonzero(keep[site]))
            factors = components.site(site)
            v = jnp.take(jax.device_put(factors.V, device).astype(jnp.float32), ids, axis=1)
            u = jnp.take(jax.device_put(factors.U, device).astype(jnp.float32), ids, axis=0)
            pad = width - int(ids.size)
            v_layers.append(jnp.pad(v, ((0, 0), (0, pad))))
            u_layers.append(jnp.pad(u, ((0, pad), (0, 0))))
        V[kind] = jnp.stack(v_layers)
        U[kind] = jnp.stack(u_layers)
    return Factors(V=V, U=U), slots


# ----------------------------------- forward -----------------------------------


def rms_norm32(x: Float[Array, "... d"], weight: Array, eps: float) -> Float[Array, "... d"]:
    x = x.astype(jnp.float32)
    var = jnp.mean(x * x, axis=-1, keepdims=True)
    return weight.astype(jnp.float32) * (x * jax.lax.rsqrt(var + eps))


type SiteFn = Callable[[str, Float[Array, "B T d_in"]], Float[Array, "B T d_out"]]
type LayerXs = tuple[Array, Array, dict[str, Array], dict[str, Array], dict[str, Array]]
"""Per-block scan inputs: the two norms, and `V`, `U` and the mask per kind."""


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
    precision: Precision,
    record: dict[str, Array] | None = None,
) -> SiteFn:
    """`site(kind, x) = ((x V) ⊙ m) U` at one block (`V`, `U`, `mask` sliced to it); with
    `record`, each site's inner activations `x V` land in it by kind."""

    def site(kind: str, x: Array) -> Array:
        xv = matmul(x, V[kind], precision)
        if record is not None:
            record[kind] = xv
        return matmul(xv * mask[kind], U[kind], precision)

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


def factored_logits(
    arch: Arch,
    shared: Shared,
    factors: Factors,
    mask: Mask,
    tokens: Int[Array, "B T"],
    site_precision: Precision,
    head_precision: Precision,
) -> Float[Array, "B vocab"]:
    """The masked model's last-position logits, differentiable in `mask` (a `scan`; each block
    is rematerialized on the backward, which otherwise keeps every block's fp32 MLP
    activations)."""

    @jax.checkpoint
    def body(resid: Array, xs: LayerXs) -> tuple[Array, None]:
        ln1, ln2, V, U, m = xs
        site = _factored_site(V, U, m, site_precision)
        return _block(arch, ln1, ln2, shared.inv_freq, resid, site, site_precision), None

    xs = (shared.ln1, shared.ln2, factors.V, factors.U, mask)
    resid, _ = jax.lax.scan(body, embed(shared, tokens), xs)
    return last_logits(arch, shared, resid, head_precision)


def factored_inner(
    arch: Arch,
    shared: Shared,
    factors: Factors,
    mask: Mask,
    tokens: Int[Array, "B T"],
    precision: Precision,
) -> dict[str, Float[Array, "L B T W"]]:
    """Every stored component's inner activation `x V_c` in the masked model, per kind."""

    def body(resid: Array, xs: LayerXs) -> tuple[Array, dict[str, Array]]:
        ln1, ln2, V, U, m = xs
        record: dict[str, Array] = {}
        site = _factored_site(V, U, m, precision, record)
        resid = _block(arch, ln1, ln2, shared.inv_freq, resid, site, precision)
        return resid, record

    xs = (shared.ln1, shared.ln2, factors.V, factors.U, mask)
    _, inner = jax.lax.scan(body, embed(shared, tokens), xs)
    return inner


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
