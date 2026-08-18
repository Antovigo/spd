"""CI-fn interface + the chunkwise-transformer impl.

A CI fn maps named INPUT taps to a `CI` bundle over OUTPUT sites:
`dict[InputTap, Array] -> CI` (preactivations + the two squashings). The input keyspace (opaque
tap keys — the lab authors them, the target resolves and captures them) is
independent of the output keyspace (the decomposition sites). The output sites MUST
partition the model's sites — every site needs exactly one CI value — asserted at
construction. Core treats both keyspaces as OPAQUE dict keys: look up inputs, scatter
outputs, validate the partition. It never parses a key.

The SAME preactivations are squashed two ways (SPEC S5/S6) in ONE place (`CI.from_preactivations`):
`lower` (clip[0,1], leaky-below) feeds recon / PPGD / routing masks; `upper`
(leaky-above-1) feeds importance-minimality. `preactivations` is kept too — the CI histograms /
heatmaps plot the pre-squash view. Params are fp32 masters (SPEC N1); the trainer casts
for bf16 compute.

The chunkwise-transformer (`ChunkwiseTransformerCIFn`) is the LM impl: each chunk reads
one or more residual taps (RMS-normed per tap, then concatenated) and emits CI for the
matrix sites it covers, via an independent pre-norm bidirectional-RoPE transformer. The
per-chunk transformers are stacked along a leading `n_chunks` axis and run under a
`jax.lax.scan` over that axis (so the chunk iteration lowers as a loop — one chunk's FSDP
weight gather live at a time, not all `n_chunks` hoisted into the flat entry computation).
The positionless toys use the MLP impls below (`LayerwiseMLPCIFn` /
`GlobalMLPCIFn`); every impl satisfies the same `CIFn` protocol and is equally core — the
architectures differ by domain (sequence vs positionless), not by status. A POSITIONED target
that cannot afford attention over its positions runs the same chunkwise impl at `n_blocks=0`,
which is position-local by construction (see `ChunkwiseTransformerCIArch`).
"""

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, PRNGKeyArray

from param_decomp.core.components import SiteSpec
from param_decomp.core.model import CaptureKeys
from param_decomp.core.precision import COMPUTE_DT, cast_floating
from param_decomp.vendored_jax.llama import (
    apply_rope,
    attn_implementation,
    rms_norm,
    rope_cos_sin,
)

CI_FN_RMS_EPS = float(jnp.finfo(jnp.float32).eps)
"""Matches torch's `F.rms_norm` default eps (`finfo(fp32).eps` ~1.19e-7); RMS upcasts to
fp32 internally, so this is the dtype that governs (SPEC S4)."""

SiteDict = dict[str, Float[Array, "*leading C"]]
"""Per-output-site tensor keyed by OUTPUT site name."""


# ----------------------------- squashings (SPEC S5/S6) -----------------------------


@jax.custom_vjp
def lower_leaky_hard_sigmoid(x: Array) -> Array:
    return jnp.clip(x, 0.0, 1.0)


def _lhs_f(x: Array) -> tuple[Array, Array]:
    return jnp.clip(x, 0.0, 1.0), x


def _lhs_b(x: Array, g: Array) -> tuple[Array]:
    leak = jnp.where(g < 0, 0.01 * g, 0.0)
    return (jnp.where(x <= 0, leak, jnp.where(x <= 1, g, 0.0)),)


lower_leaky_hard_sigmoid.defvjp(_lhs_f, _lhs_b)


def upper_leaky_hard_sigmoid(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """`x>1 ? 1+alpha*(x-1) : clamp(x,0,1)` — ordinary autodiff of this expression
    (torch builds its backward the same way; only the lower squashing is a custom VJP)."""
    alpha = 0.01
    return jnp.where(x > 1, 1 + alpha * (x - 1), jnp.clip(x, 0.0, 1.0))


# ----------------------------- the CI bundle + protocol -----------------------------


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CI:
    """The CI fn output: raw preactivations + both squashings, all keyed by output site. `preactivations`
    is kept (a consumed view — the histograms / heatmaps plot pre-squash). The squashing
    lives only in `from_preactivations`, so no impl re-triplicates it."""

    preactivations: SiteDict
    lower: SiteDict
    upper: SiteDict

    @staticmethod
    def from_preactivations(preactivations: SiteDict) -> "CI":
        return CI(
            preactivations=preactivations,
            lower={k: lower_leaky_hard_sigmoid(v) for k, v in preactivations.items()},
            upper={k: upper_leaky_hard_sigmoid(v) for k, v in preactivations.items()},
        )


CIRole = Literal["output", "hidden"]
"""Which reconstruction objective a CI value scores for (SPEC S36). `output` is the
model-output reconstruction every VPD run has always had; `hidden` scores the SAME pool of
subcomponents for reconstructing named internal activations instead. A single-role run has
only `output` and never mentions the vocabulary."""

DUAL_CI_ROLES: tuple[CIRole, ...] = ("output", "hidden")
"""Role order — the head order in a dual CI fn and the pass order in a dual objective."""


def roles_for(dual: bool) -> tuple[CIRole, ...]:
    """The readout vocabulary a `dual` flag implies — the ONE place that correspondence
    lives, so the arches, the inits and the eval fan-out cannot drift apart on it."""
    return DUAL_CI_ROLES if dual else ("output",)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DualCI:
    """Two `CI` bundles read off ONE trunk (SPEC S36): `output` feeds the output-reconstruction
    objective, `hidden` the hidden-acts one. Both score the same subcomponent pool over the same
    inputs — only the readout head differs, so a component can be important for one objective and
    not the other.

    A pytree whose leaves are the two bundles', so `eqx.filter_vjp` over the CI fn takes a
    `DualCI` cotangent and the trunk's parameter gradient is the SUM of the two heads'
    contributions in ONE pullback — the trunk is never forward- or backward-run twice."""

    output: CI
    hidden: CI


AnyCI = CI | DualCI
"""What a CI fn returns: single-role runs keep the bare `CI` they always had."""


def output_ci(ci: AnyCI) -> CI:
    """The output-role bundle of whatever a CI fn returned. The spelling exists so a
    single-role consumer SAYS which role it means instead of reaching through a union — a dual
    fn's `.lower` is otherwise an easy silent mistake (it would have to pick a head for you)."""
    return ci_for_role(ci, "output")


def _bundle(roles: tuple[CIRole, ...], per_role: tuple[SiteDict, ...]) -> AnyCI:
    """Squash each role's preactivations into its bundle, returning the shape `roles` implies —
    a bare `CI` for a single-role fn, so nothing downstream of a plain run ever sees `DualCI`."""
    assert len(roles) == len(per_role), (roles, len(per_role))
    match roles:
        case ("output",):
            return CI.from_preactivations(per_role[0])
        case ("output", "hidden"):
            return DualCI(
                output=CI.from_preactivations(per_role[0]),
                hidden=CI.from_preactivations(per_role[1]),
            )
        case _:
            raise AssertionError(f"unknown CI role tuple {roles}")


def ci_for_role(ci: AnyCI, role: CIRole) -> CI:
    """The bundle scoring `role`. A single-role `CI` answers only to `output` — asking a
    single-role run for the hidden bundle is a wiring bug, not a fallback."""
    match ci:
        case DualCI():
            match role:
                case "output":
                    return ci.output
                case "hidden":
                    return ci.hidden
        case CI():
            assert role == "output", (
                f"a single-role CI fn has no {role!r} head; set `decomposition.ci.dual` to build one"
            )
            return ci


@runtime_checkable
class CIFn(Protocol):
    """`dict[InputTap, Array] -> CI | DualCI`. `output_names` partition the model sites (asserted
    at construction); input taps are unconstrained. `has_position_axis` must equal the
    paired `DecomposedModel.has_position_axis` (asserted at trainer construction).

    `roles` is the readout vocabulary: `("output",)` for the single-head CI fn every plain run
    uses — which returns a bare `CI` — and `DUAL_CI_ROLES` for a shared-trunk dual fn, which
    returns `DualCI`. The role count is STATIC (it selects the return type), so a step traced
    against a dual fn never branches on it."""

    @property
    def capture_keys(self) -> CaptureKeys: ...

    output_names: tuple[str, ...]
    has_position_axis: bool
    roles: tuple[CIRole, ...]

    def __call__(self, taps: dict[str, Array], *, remat: bool) -> AnyCI: ...

    def shardings(self, mesh: Mesh) -> "CIFn":
        """Per-leaf `dp` placement matching this CI fn's pytree structure (each array leaf
        → a `NamedSharding`; `P()` to replicate). Asserts every declared shard axis tiles
        the mesh. Applied via `jax.jit(init, out_shardings=...)`."""
        ...


def is_dual(ci_fn: CIFn) -> bool:
    """Whether this CI fn reads two roles off one trunk — a STATIC property of the built fn."""
    return len(ci_fn.roles) > 1


def evaluate_ci(ci_fn: CIFn, taps: dict[str, Array], *, remat: bool) -> AnyCI:
    """Run fp32-master CI parameters and their captured inputs in compute precision."""
    compute_ci_fn = cast_floating(ci_fn, COMPUTE_DT)
    compute_taps = cast_floating(taps, COMPUTE_DT)
    return compute_ci_fn(compute_taps, remat=remat)


def evaluate_ci_role(ci_fn: CIFn, taps: dict[str, Array], *, remat: bool, role: CIRole) -> CI:
    """One role's bundle — the seam for consumers (evals, probes) that score a single role."""
    return ci_for_role(evaluate_ci(ci_fn, taps, remat=remat), role)


def ci_preactivations(
    ci_fn: CIFn, taps: dict[str, Array], *, remat: bool, role: CIRole
) -> SiteDict:
    """Evaluate CI in compute precision and expose fp32 preactivations for metric reductions.

    `role` is REQUIRED, unlike the step-builder seams that default it: this is the lowest
    CI read in the tree, so a default here is how a dual run ends up reporting one head's
    numbers under both names — the exact failure S36 names."""
    ci = evaluate_ci_role(ci_fn, taps, remat=remat, role=role)
    return cast_floating(ci.preactivations, jnp.float32)


# ----------------------------- transformer building blocks -----------------------------


def _weightless_rms_norm(x: Array, eps: float) -> Array:
    return rms_norm(x, jnp.ones((x.shape[-1],), x.dtype), eps)


def _rms_norm_maybe_scaled(x: Array, scale: Array | None, eps: float) -> Array:
    """`scale is None` is the weightless norm — `ones` in x's dtype, i.e. today's numerics
    exactly (and no bf16→fp32 promotion, which an fp32 scale leaf would cause)."""
    if scale is None:
        return _weightless_rms_norm(x, eps)
    return rms_norm(x, scale, eps)


@dataclass(frozen=True)
class MHACIAttention:
    """Every query head carries its own K/V head."""

    n_heads: int

    @property
    def n_kv_heads(self) -> int:
        return self.n_heads


@dataclass(frozen=True)
class GQACIAttention:
    """`n_heads // n_kv_heads` query heads share each K/V head, so `wk`/`wv` narrow to
    `n_kv_heads * head_dim`. head_dim, the RoPE tables, `wq`/`wo` and every sharding are
    identical to MHA — only the K/V projections change."""

    n_heads: int
    n_kv_heads: int

    def __post_init__(self) -> None:
        assert self.n_heads % self.n_kv_heads == 0, (
            "n_heads must be divisible by n_kv_heads (each K/V head serves an equal group "
            f"of query heads): {self.n_heads} % {self.n_kv_heads}"
        )
        assert self.n_kv_heads < self.n_heads, (
            f"n_kv_heads == n_heads ({self.n_heads}) is MHA — use MHACIAttention rather than "
            "a degenerate GQA"
        )


CIAttention = MHACIAttention | GQACIAttention
"""The CI transformer's attention. Both arms answer `n_heads` and `n_kv_heads`, so the call
site never dispatches — but MHA derives its K/V count from the TYPE instead of leaving
`n_kv_heads == n_heads` as a convention a reader has to know, and cannot carry an explicit
one. GQA's grouping invariant is checked at construction, not at init."""


CIFfnKind = Literal["gelu", "swiglu"]
"""`gelu`: `Linear+b → GELU → Linear+b`. `swiglu`: a second projection gates the first —
`silu(h@wg + bg) * (h@w1 + b1) → Linear+b`. SwiGLU is a THIRD matrix, so it grows the MLP
~50% at a fixed `ffn_hidden`; iso-param means setting `ffn_hidden` to ~2/3. Nothing here
rescales it — the width is the config author's to state."""


class CIBlock(eqx.Module):
    """Pre-norm block: RMSNorm → bidirectional RoPE attention → residual;
    RMSNorm → FFN (`gelu` or `swiglu`) → residual.

    `attention` is the resolved variant: under `GQACIAttention` the K/V projections narrow to
    `n_kv_heads * head_dim` and `jax.nn.dot_product_attention` broadcasts each K/V head over
    its group of query heads. Both arms answer `n_kv_heads`, so nothing here dispatches.

    `gate is None` ⟺ the GELU FFN; a present gate ⟺ SwiGLU. The gate's `(w, b)` ride in one
    optional tuple because they vary together, and its presence IS the FFN discriminator — so
    there's no separate tag to desync from the params.

    `norm_scales is None` ⟺ the weightless norms (today's behaviour); present ⟺ learned
    per-channel scales, `(pre-attn, pre-MLP)`."""

    wq: Array
    wk: Array
    wv: Array
    wo: Array
    w1: Array
    b1: Array
    w2: Array
    b2: Array
    gate: tuple[Array, Array] | None
    norm_scales: tuple[Array, Array] | None
    attention: CIAttention = eqx.field(static=True)
    eps: float = eqx.field(static=True)

    def shardings(self, mesh: Mesh) -> "CIBlock":
        """ZeRO-1 PERSISTENCE layout (master + Adam m/v); leading `n_chunks` axis (axis 0)
        UNSHARDED. Attention is FSDP-on-d_model (tp-replicated — v1 skips attn Megatron); the
        MLP is Megatron-on-ffn_hidden including the `tp` axis (so it shards ÷N, not ÷(rep·fsdp)):

        - qkv (`[nc, head_out, d_model_in]`): d_model (axis2) ÷(rep·fsdp), head + tp replicated.
          `head_out` is `n_kv_head * head_dim` for k/v under GQA — the sharded axis is the
          d_model INPUT, so a narrower K/V head_out changes no sharding.
        - out-proj (`[nc, d_model_out, head_in]`): d_model (axis1) ÷(rep·fsdp), head + tp replicated.
        - w1 up-proj (`[nc, d_model_in, ffn_hidden_out]`): **ffn_hidden (axis2) ÷N** (incl tp),
          d_model replicated (column-parallel).
        - w2 down-proj (`[nc, ffn_hidden_in, d_model_out]`): **ffn_hidden (axis1) ÷N** (incl tp),
          d_model replicated (row-parallel → in-node reduce over fsdp·tp).

        Biases replicate. COMPUTE re-pins to `fsdp` (attn d_model) / `fsdp·tp` (MLP ffn_hidden)
        before the chunk scan (`ChunkwiseTransformerCIFn.__call__`), all intra-node NVLink.

        Fsdp-major linearization (compute axes first, `replicate` last) so the ÷N→compute
        reconstruct is a pure all-gather over `replicate` — see `placement._ZERO1_DATA`."""
        data = ("fsdp", "replicate")
        full = ("fsdp", "tp", "replicate")
        attn_in = NamedSharding(mesh, P(None, None, data))  # qkv: d_model (axis2) ÷(rep·fsdp)
        attn_out = NamedSharding(mesh, P(None, data, None))  # wo: d_model (axis1) ÷(rep·fsdp)
        ffn_in = NamedSharding(mesh, P(None, None, full))  # w1: ffn_hidden (axis2) ÷N
        ffn_out = NamedSharding(mesh, P(None, full, None))  # w2: ffn_hidden (axis1) ÷N
        repl = NamedSharding(mesh, P())
        n_data = mesh.shape["replicate"] * mesh.shape["fsdp"]
        n_full = mesh.devices.size
        for w in (self.wq, self.wk, self.wv):
            assert w.shape[2] % n_data == 0, f"CIBlock qkv d_model {w.shape[2]} not ÷ {n_data}"
        assert self.wo.shape[1] % n_data == 0, (
            f"CIBlock wo d_model {self.wo.shape[1]} not ÷ {n_data}"
        )
        assert self.w1.shape[2] % n_full == 0, (
            f"CIBlock w1 ffn_hidden {self.w1.shape[2]} not ÷ N={n_full}"
        )
        assert self.w2.shape[1] % n_full == 0, (
            f"CIBlock w2 ffn_hidden {self.w2.shape[1]} not ÷ N={n_full}"
        )
        placed = eqx.tree_at(
            lambda b: (b.wq, b.wk, b.wv, b.wo, b.w1, b.b1, b.w2, b.b2),
            self,
            (attn_in, attn_in, attn_in, attn_out, ffn_in, repl, ffn_out, repl),
        )
        if self.gate is not None:
            # The swiglu gate is a second `[nc, d_model, ffn_hidden]` up-proj: same
            # Megatron-on-ffn_hidden placement as w1, same ÷N divisibility requirement.
            assert self.gate[0].shape[2] % n_full == 0, (
                f"CIBlock swiglu gate ffn_hidden {self.gate[0].shape[2]} not ÷ N={n_full}"
            )
            placed = eqx.tree_at(lambda b: b.gate, placed, (ffn_in, repl))
        if self.norm_scales is not None:
            placed = eqx.tree_at(lambda b: b.norm_scales, placed, (repl, repl))
        return placed

    def __call__(self, x: Float[Array, "b t d"], inv_freq: Array) -> Array:
        t = x.shape[1]
        attn_scale, mlp_scale = (None, None) if self.norm_scales is None else self.norm_scales
        h = _rms_norm_maybe_scaled(x, attn_scale, self.eps)

        def heads(w: Array, n_head: int) -> Array:  # [b, t, d] -> [b, nh, t, hd]  (RoPE layout)
            proj = einops.einsum(h, w, "b t i, o i -> b t o")
            return einops.rearrange(proj, "b t (nh hd) -> b nh t hd", nh=n_head)

        q = heads(self.wq, self.attention.n_heads)
        kv = self.attention.n_kv_heads
        k, v = heads(self.wk, kv), heads(self.wv, kv)
        cos, sin = rope_cos_sin(inv_freq, t, x.dtype)
        q, k = apply_rope(q, k, cos, sin)  # cos/sin broadcast over the head axis: any count
        qt, kt, vt = (einops.rearrange(a, "b nh t hd -> b t nh hd") for a in (q, k, v))
        # cuDNN flash on GPU (its partitioner requires device-local heads — true here, no
        # head-sharding); XLA elsewhere (CPU tests have no cuDNN). Bidirectional. Fewer K/V
        # heads than query heads is GQA, grouped natively by dot_product_attention.
        impl = attn_implementation(jax.default_backend(), qt.dtype, t)
        y = jax.nn.dot_product_attention(qt, kt, vt, is_causal=False, implementation=impl)
        x = x + einops.einsum(
            einops.rearrange(y, "b t nh hd -> b t (nh hd)"), self.wo, "b t i, o i -> b t o"
        )
        h = _rms_norm_maybe_scaled(x, mlp_scale, self.eps)
        up = einops.einsum(h, self.w1, "b t i, i o -> b t o") + self.b1
        if self.gate is None:
            hidden = jax.nn.gelu(up, approximate=False)
        else:
            w_gate, b_gate = self.gate
            gate = einops.einsum(h, w_gate, "b t i, i o -> b t o") + b_gate
            hidden = jax.nn.silu(gate) * up
        return x + einops.einsum(hidden, self.w2, "b t i, i o -> b t o") + self.b2


# ----------------------------- chunkwise transformer -----------------------------


@dataclass(frozen=True)
class Chunk:
    """One resolved chunk: the input taps to concatenate → CI for a group of output sites.
    Authored lab-side (from `blocks_per_chunk` + topology); core treats both keyspaces as
    opaque keys. `input_taps` may name several residual taps (e.g. the residual entering the
    chunk plus earlier read points) — RMS-normed per tap and concatenated as the input."""

    input_taps: tuple[str, ...]
    output_sites: tuple[str, ...]


@dataclass(frozen=True)
class _ChunkMeta:
    """Per-chunk static routing, index-aligned with the stacked `chunks` leading axis."""

    input_taps: tuple[str, ...]  # taps to RMS-norm + concatenate as this chunk's input
    output_sites: tuple[str, ...]  # output sites this chunk scores, in C-per-slot order


@dataclass(frozen=True)
class ChunkwiseTransformerCIArch:
    """Resolved chunkwise-transformer arch: explicit chunks + the CI transformer's dims.

    `input_dim` is the per-chunk concatenated input width — a plain linear-layer input
    dimension. The lab computes it from the taps it authored (their widths summed); core
    stays agnostic to what the taps mean, so no transformer concept (residual width) leaks
    in. All chunks share one `input_dim` (the vmap homogeneity requirement).

    `attention` is the resolved variant (the schema's `attention` union, translated).

    `n_blocks=0` degenerates to `RMS-normed taps → in_proj → per-site output heads`: the FFN
    lives inside the block alongside attention, so dropping blocks leaves an affine map on the
    NORMALIZED tap — a direction-only probe, with no learned nonlinearity, no hidden layer, and
    no sensitivity to tap magnitude at all. It is position-local (blocks are the only thing
    reading ACROSS positions) and it runs, so it serves as a cheap baseline, but a positioned
    target that wants a real per-position CI fn wants `LayerwiseMLPCIArch(has_position_axis=True)`.
    Pinned by `core/tests/test_ci_fn_zero_blocks.py` (locality) and
    `core/tests/test_ci_fn_positioned_mlp.py` (the magnitude contrast)."""

    chunks: tuple[Chunk, ...]
    input_dim: int
    d_model: int
    n_blocks: int
    attention: CIAttention
    ffn_hidden: int
    ffn_kind: CIFfnKind
    learned_norm_scale: bool
    dual: bool = False
    """Build a SECOND readout head on this same trunk (SPEC S36), so the CI fn scores both
    the output and the hidden reconstruction objectives. An ARCH property, not a build-time
    flag: it changes the parameter tree, so it must ride the arch through checkpointing and
    the run bundle rather than being re-supplied at every construction site."""

    @property
    def capture_keys(self) -> CaptureKeys:
        """The activation taps consumed by any chunk."""
        return frozenset(tap for chunk in self.chunks for tap in chunk.input_taps)


class ChunkTransformer(eqx.Module):
    """ONE chunk: its (already RMS-normed, concatenated) input `[*leading, total_d_in]` →
    a TUPLE of per-output-site preactivations (`out` of `[*leading, C_j]` per site-slot j), via
    in_proj → RoPE blocks → one output head PER site-slot.

    One head per site-slot (`out_ws[j] [d_model, C_j]` / `out_bs[j] [C_j]`) instead of a
    single glued `[d_model, ΣC]` head: each head's output IS that site's CI, born already
    split per site (matching `x@V` / the mask, SPEC §4.1 `site_out`). Under pure HSDP the C
    axis is replicated (not sharded), so the split is a pure layout convenience; it was
    load-bearing under the prior TP layout (a tp-sharded glued ΣC axis sliced mid-site),
    and is kept harmlessly.

    In the bundle every array below carries a leading `n_chunks` axis and the module is
    run under `jax.lax.scan` over that axis, so this body is written for a single chunk."""

    in_proj_w: Float[Array, "total_d_in d_model"]
    in_proj_b: Float[Array, " d_model"]
    blocks: list[CIBlock]
    out_ws: tuple[Float[Array, "d_model _C"], ...]
    out_bs: tuple[Float[Array, " _C"], ...]
    hidden_out_ws: tuple[Float[Array, "d_model _C"], ...]
    hidden_out_bs: tuple[Float[Array, " _C"], ...]
    """The SECOND readout head (SPEC S36), empty tuples in a single-role fn. `in_proj_*` and
    `blocks` — everything before the readout — are the TRUNK and are shared by construction:
    there is one set of trunk arrays in the pytree, so both roles read one representation and
    one `eqx.filter_vjp` pullback sums both heads' trunk gradients. This is the JAX analogue of
    the torch `GlobalSharedTransformerCiFn.adopt_trunk` module-identity sharing, minus its two
    traps: the trunk cannot be double-counted by an optimizer (it appears once in the tree) and
    a shared-trunk checkpoint cannot be confused with an independent-pair one (the pytree shape
    differs), so no load-time value comparison is needed."""

    def shardings(self, mesh: Mesh) -> "ChunkTransformer":
        """True ÷N ZeRO-1 PERSISTENCE layout (master + Adam shard over the FULL mesh); leading
        `n_chunks` axis (axis 0) UNSHARDED. `in_proj_w [nc, total_d_in, d_model]`: d_model over
        the data axes, total_d_in replicated (column-parallel output — no weight gather). Each
        `out_ws[j] [nc, d_model, C_j]`: d_model over the data axes, **C_j over `tp`** (Megatron-C)
        → ÷N total, and the CI output C is `tp`-sharded so it dovetails with V/U's C-on-tp (the
        `mask · xV` multiply stays local). Blocks delegate to `CIBlock.shardings`; biases
        replicate. COMPUTE re-pins to `fsdp` (d) × `tp` (C) before the chunk scan
        (`ChunkwiseTransformerCIFn.__call__`). Fsdp-major linearization (`replicate` last)
        so the reconstruct is a pure all-gather over `replicate` — see
        `placement._ZERO1_DATA`."""
        data = ("fsdp", "replicate")
        in_proj_sh = NamedSharding(
            mesh, P(None, "tp", data)
        )  # in_proj: total_d_in ÷tp, d_model ÷(rep·fsdp) → ÷N (row-parallel)
        out_ws_sh = NamedSharding(
            mesh, P(None, data, "tp")
        )  # out_ws: d_model ÷(rep·fsdp), C ÷tp → ÷N
        repl = NamedSharding(mesh, P())
        n_data = mesh.shape["replicate"] * mesh.shape["fsdp"]
        n_tp = mesh.shape["tp"]
        assert self.in_proj_w.shape[1] % n_tp == 0, (
            f"ChunkTransformer in_proj_w total_d_in {self.in_proj_w.shape[1]} not ÷ tp={n_tp}"
        )
        assert self.in_proj_w.shape[2] % n_data == 0, (
            f"ChunkTransformer in_proj_w d_model {self.in_proj_w.shape[2]} not ÷ {n_data}"
        )
        for head, ws in (("out_ws", self.out_ws), ("hidden_out_ws", self.hidden_out_ws)):
            for slot, w in enumerate(ws):
                assert w.shape[1] % n_data == 0, (
                    f"ChunkTransformer {head}[{slot}] d_model {w.shape[1]} not ÷ {n_data}"
                )
                assert w.shape[2] % n_tp == 0, (
                    f"ChunkTransformer {head}[{slot}] C {w.shape[2]} not ÷ tp={n_tp}"
                )
        return eqx.tree_at(
            lambda ct: (
                ct.in_proj_w,
                ct.in_proj_b,
                ct.blocks,
                ct.out_ws,
                ct.out_bs,
                ct.hidden_out_ws,
                ct.hidden_out_bs,
            ),
            self,
            (
                in_proj_sh,
                repl,
                [b.shardings(mesh) for b in self.blocks],
                tuple(out_ws_sh for _ in self.out_ws),
                tuple(repl for _ in self.out_bs),
                # The hidden head shards exactly like the output head: both are
                # `[nc, d_model, C_j]` readouts off the same trunk.
                tuple(out_ws_sh for _ in self.hidden_out_ws),
                tuple(repl for _ in self.hidden_out_bs),
            ),
        )

    def __call__(
        self,
        x: Float[Array, "*leading total_d_in"],
        inv_freq: Array,
    ) -> tuple[tuple[Float[Array, "*leading _C"], ...], ...]:
        """`(per-slot output preacts, per-slot hidden preacts)` — the hidden tuple EMPTY in a
        single-role fn. The trunk below runs ONCE; the heads are the only per-role work."""
        x = einops.einsum(x, self.in_proj_w, "... i, i o -> ... o") + self.in_proj_b
        for block in self.blocks:
            x = block(x, inv_freq)

        def readout(
            ws: tuple[Array, ...], bs: tuple[Array, ...]
        ) -> tuple[Float[Array, "*leading _C"], ...]:
            return tuple(
                einops.einsum(x, w, "... i, i o -> ... o") + b for w, b in zip(ws, bs, strict=True)
            )

        return (readout(self.out_ws, self.out_bs), readout(self.hidden_out_ws, self.hidden_out_bs))


def _reconstruct_ci_compute_weights(chunks: "ChunkTransformer") -> "ChunkTransformer":
    """The ZeRO-1 reconstruction for the CI fn: the stacked per-chunk weights arrive with
    their `d_model` dim sharded ÷N over the FULL mesh (the master is `P(..., ("replicate",
    "fsdp"), ...)`); reconstruct them to the `fsdp`-sharded (÷fsdp) COMPUTE layout here,
    BEFORE the chunk scan, so the cross-`replicate` gather runs ONCE per step in ENTRY
    (landing a SMALL ÷fsdp-resident weight stack, NOT the full CI fn) and the per-chunk scan
    body gathers only on `fsdp` (intra-node NVLink), transiently. Cast to bf16 here so the
    ÷fsdp-resident stack is half-size (no f32 full copy). Mirrors the `.shardings` axis
    positions (leading `n_chunks` axis unsharded) with `"fsdp"` in place of the full-mesh
    tuple. No-op off-mesh."""
    if jax.sharding.get_abstract_mesh().empty:
        return chunks
    d_axis2 = P(None, None, "fsdp")  # attn d_model axis2 (gathered), tp-replicated
    d_axis1 = P(None, "fsdp", None)  # attn d_model axis1 (gathered), tp-replicated
    out_ws_axis = P(None, "fsdp", "tp")  # out_ws [nc, d_model÷fsdp, C÷tp] — d gathered, C Megatron
    ffn_in_c = P(None, None, ("fsdp", "tp"))  # w1 [nc, d_model, ffn_hidden÷(fsdp·tp)] — Megatron
    ffn_out_c = P(None, ("fsdp", "tp"), None)  # w2 [nc, ffn_hidden÷(fsdp·tp), d_model] — Megatron
    in_proj_c = P(None, "tp", "fsdp")  # in_proj [nc, total_d_in÷tp, d_model÷fsdp] — row-parallel
    repl_c = P()  # the swiglu gate bias / norm scales: [nc, d]-shaped, replicated like b1/b2

    def pin(x: Array, spec: "P") -> Array:
        # optimization_barrier: cast bf16 BEFORE the gather (else XLA sinks the convert past
        # the all-gather and moves the f32 master — 2x the comm).
        return jax.lax.with_sharding_constraint(
            jax.lax.optimization_barrier(x.astype(jnp.bfloat16)), spec
        )

    def pin_block(blk: CIBlock) -> CIBlock:
        pinned = eqx.tree_at(
            lambda b: (b.wq, b.wk, b.wv, b.wo, b.w1, b.w2),
            blk,
            (pin(blk.wq, d_axis2), pin(blk.wk, d_axis2), pin(blk.wv, d_axis2),
             pin(blk.wo, d_axis1), pin(blk.w1, ffn_in_c), pin(blk.w2, ffn_out_c)),
        )  # fmt: skip
        if blk.gate is not None:
            # The swiglu gate MUST be cast with the rest: left as an fp32 master it would
            # promote the whole `silu(gate) * up` product (and everything after it) back to
            # fp32 — a silent numerics + memory regression that still trains.
            pinned = eqx.tree_at(
                lambda b: b.gate, pinned, (pin(blk.gate[0], ffn_in_c), pin(blk.gate[1], repl_c))
            )
        if blk.norm_scales is not None:
            # Same trap, sharper: `rms_norm` returns `weight * x`, so an fp32 scale promotes
            # the block's bf16 residual stream to fp32. The weightless path dodges it only
            # because it builds its `ones` in x's own dtype.
            pinned = eqx.tree_at(
                lambda b: b.norm_scales,
                pinned,
                tuple(pin(s, repl_c) for s in blk.norm_scales),
            )
        return pinned

    pinned_blocks = [pin_block(blk) for blk in chunks.blocks]
    return eqx.tree_at(
        lambda ct: (ct.in_proj_w, ct.blocks, ct.out_ws, ct.hidden_out_ws),
        chunks,
        (
            pin(chunks.in_proj_w, in_proj_c),  # [nc, total_d_in÷tp, d_model÷fsdp] — row-parallel
            pinned_blocks,
            tuple(pin(w, out_ws_axis) for w in chunks.out_ws),  # [nc, d_model÷fsdp, C÷tp]
            # The hidden head is a readout like the output head — same layout, same reason.
            # Empty in a single-role fn, where this is a no-op on an empty tuple.
            tuple(pin(w, out_ws_axis) for w in chunks.hidden_out_ws),
        ),
    )


class ChunkwiseTransformerCIFn(eqx.Module):
    """Per-chunk `ChunkTransformer`s stacked along a leading `n_chunks` axis, iterated by a
    `jax.lax.scan` over that axis (lowers as a loop so one chunk's FSDP weight gather is live
    at a time, not all `n_chunks` at once). Each chunk's input is its `chunk_input_taps`
    RMS-normed per tap and concatenated. Requires homogeneous chunks (equal total input width
    and an identical per-slot C tuple — same C-per-output-site ORDER) so the stack, including
    the per-slot output heads, is rectangular — asserted at init."""

    chunks: ChunkTransformer  # arrays stacked along leading n_chunks
    inv_freq: Array  # shared across chunks (RoPE buffer); NOT mapped

    capture_keys: CaptureKeys = eqx.field(static=True)
    output_names: tuple[str, ...] = eqx.field(static=True)  # all sites, flat
    chunk_meta: tuple[_ChunkMeta, ...] = eqx.field(static=True)  # per-chunk routing
    eps: float = eqx.field(static=True)
    has_position_axis: bool = eqx.field(static=True)
    roles: tuple[CIRole, ...] = eqx.field(static=True)

    def shardings(self, mesh: Mesh) -> "ChunkwiseTransformerCIFn":
        """The stacked per-chunk transformer's HSDP layout (`ChunkTransformer.shardings`,
        leading `n_chunks` axis un-sharded); `inv_freq` (a 1-D RoPE buffer) replicates."""
        return eqx.tree_at(
            lambda f: (f.chunks, f.inv_freq),
            self,
            (self.chunks.shardings(mesh), NamedSharding(mesh, P())),
        )

    def __call__(self, taps: dict[str, Array], *, remat: bool) -> AnyCI:
        per_chunk_in = [
            jnp.concatenate(
                [_weightless_rms_norm(taps[k], self.eps) for k in m.input_taps], axis=-1
            )
            for m in self.chunk_meta
        ]
        stacked_in = jnp.stack(per_chunk_in, axis=0)  # [n_chunks, *leading, total_d_in]
        inv_freq = jax.lax.stop_gradient(self.inv_freq)
        # ZeRO-1 reconstruction: the master shards d_model ÷N over the FULL mesh; pin the
        # compute weights `fsdp`-ONLY here, BEFORE the chunk scan, so GSPMD gathers the
        # `replicate` shard ONCE per step in ENTRY (off the hot path) and the per-chunk scan
        # body gathers only on `fsdp` (intra-node NVLink). No-op off-mesh.
        chunks = _reconstruct_ci_compute_weights(self.chunks)
        # `lax.scan` (not `filter_vmap`) over the leading `n_chunks` axis so XLA lowers the
        # chunk iteration as a loop: one chunk's FSDP weight all-gather (∝ ΣC/tp) is live at
        # a time, then freed, instead of every chunk's gathered weights materialized at once
        # (the vmap unrolls, hoisting all n_chunks gathers into the flat entry computation).
        # Same math as the vmap — scan stacks per-iteration outputs exactly as vmap maps
        # them; results match up to fp32 reassociation (XLA picks different matmul layouts).
        chunk_arrays, chunk_static = eqx.partition(chunks, eqx.is_array)

        def run_chunk(
            _: None, scanned: tuple[ChunkTransformer, Array]
        ) -> tuple[None, tuple[tuple[Array, ...], ...]]:
            chunk_array, chunk_input = scanned
            chunk = eqx.combine(chunk_array, chunk_static)
            return None, chunk(chunk_input, inv_freq)

        # Per-CHUNK remat: checkpoint the scan BODY so the backward recomputes one chunk at a
        # time, keeping only the carry — NOT all `n_chunks` chunks' attention scores + MLP
        # hidden states stacked `[n_chunks, ...]`. (Whole-CI-fn checkpointing does not bound
        # the scan: the recompute still stacks every chunk — the `[n_chunks, *, seq, seq]`
        # f32 score slab that dominated the full-model step. Same fix shape as the target's
        # per-layer remat.)
        # Each per-slot head stacks over the chunk axis: `stacked_per_slot[j]` is
        # `[n_chunks, *leading, C_j]`. No glued ΣC axis, so no slice — site `(chunk i, slot j)`
        # is `stacked_per_slot[j][i]` directly (chunks are slot-homogeneous in C-per-site
        # ORDER, asserted at init, so slot j carries one C_j across every chunk).
        # Per-CHUNK checkpoint of the scan BODY in BOTH modes — `remat` controls ONLY whether
        # the chunk ACTIVATIONS are recomputed; it NEVER controls the ÷fsdp→full weight gather.
        # `remat=True` → nothing_saveable: recompute activations AND re-gather (min memory, the
        # `[n_chunks, *, seq, seq]` f32 score slab never stacks). `remat=False` → dots_saveable:
        # SAVE the activation matmuls, still re-gather the weights (a collective, not a dot) — i.e.
        # plain FSDP. WITHOUT any checkpoint the backward would instead stack every chunk's full
        # gathered weights `[n_chunks, …]` as residuals → DDP-stack OOM, so we always checkpoint.
        policy = (
            jax.checkpoint_policies.nothing_saveable
            if remat
            else jax.checkpoint_policies.dots_saveable
        )

        body = jax.checkpoint(run_chunk, policy=policy)
        _, stacked_per_role = jax.lax.scan(body, None, (chunk_arrays, stacked_in))

        def scatter(stacked_per_slot: tuple[Array, ...]) -> SiteDict:
            preactivations: SiteDict = {}
            for chunk_idx, m in enumerate(self.chunk_meta):
                for slot, site in enumerate(m.output_sites):
                    preactivations[site] = stacked_per_slot[slot][chunk_idx]
            return preactivations

        # ONE scan produced every head: the trunk ran once, and the heads' slot tuples come
        # back stacked side by side. `_bundle` is the single place that decides what a role
        # tuple maps to, so this impl cannot drift from the MLP ones.
        return _bundle(self.roles, tuple(scatter(s) for s in stacked_per_role[: len(self.roles)]))


def _init_chunk_transformer(
    arch: ChunkwiseTransformerCIArch,
    total_d_in: int,
    slot_cs: tuple[int, ...],
    key: PRNGKeyArray,
    *,
    dual: bool = False,
) -> ChunkTransformer:
    """One chunk's params, same Kaiming scheme as the old global transformer: relu-gain
    (√2) on in_proj / MLP-in, linear gain (1) on out / MLP-out, PyTorch-default
    `U(±1/√fan_in)` on the attention projections, zero biases.

    The per-site output heads are SLICES of a single glued `[d, ΣC]` Kaiming draw (drawn with
    the same `out_key`, `gain 1`): head j = columns `[offset_j : offset_j + C_j]`. This keeps
    the RNG consumption (one `(d, ΣC)` normal + one `(ΣC,)` zero bias) and the values bit-for-
    bit identical to the old single glued head, so the equivalence goldens are unchanged —
    the math is the same, only the partitioning differs.

    Each consumer takes its OWN explicit key — the split count lives next to its use
    (`n_blocks + 2` at the top = in_proj + out + one per block; 6 within a block, 7 with
    swiglu's gate), so it can't silently drift out of sync with the number of draws."""
    relu_gain = 2.0**0.5
    d, ffn = arch.d_model, arch.ffn_hidden
    d_kv = (d // arch.attention.n_heads) * arch.attention.n_kv_heads  # narrower under GQA

    def kaiming(k: PRNGKeyArray, shape: tuple[int, ...], fan_in: int, gain: float) -> Array:
        return jax.random.normal(k, shape) * (gain / fan_in**0.5)

    def attn_default(k: PRNGKeyArray, shape: tuple[int, ...], fan_in: int) -> Array:
        bound = 1.0 / fan_in**0.5
        return jax.random.uniform(k, shape, minval=-bound, maxval=bound)

    def block(bkey: PRNGKeyArray) -> CIBlock:
        # 6 draws for gelu, 7 for swiglu's extra gate — NOT 7 unconditionally: the split
        # count determines every derived key, so widening it would silently redraw every
        # gelu param and move the equivalence goldens.
        match arch.ffn_kind:
            case "gelu":
                kq, kk, kv, ko, k1, k2 = jax.random.split(bkey, 6)
                gate = None
            case "swiglu":
                kq, kk, kv, ko, k1, k2, kg = jax.random.split(bkey, 7)
                gate = (kaiming(kg, (d, ffn), d, relu_gain), jnp.zeros((ffn,)))
        norm_scales = (jnp.ones((d,)), jnp.ones((d,))) if arch.learned_norm_scale else None
        return CIBlock(
            wq=attn_default(kq, (d, d), d), wk=attn_default(kk, (d_kv, d), d),
            wv=attn_default(kv, (d_kv, d), d), wo=attn_default(ko, (d, d), d),
            w1=kaiming(k1, (d, ffn), d, relu_gain), b1=jnp.zeros((ffn,)),
            w2=kaiming(k2, (ffn, d), ffn, 1.0), b2=jnp.zeros((d,)),
            gate=gate, norm_scales=norm_scales,
            attention=arch.attention, eps=CI_FN_RMS_EPS,
        )  # fmt: skip

    in_key, out_key, *block_keys = jax.random.split(key, arch.n_blocks + 2)
    c_chunk = sum(slot_cs)
    offsets = [0]
    for c in slot_cs:
        offsets.append(offsets[-1] + c)

    def head(head_key: PRNGKeyArray) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
        glued_w = kaiming(head_key, (d, c_chunk), d, 1.0)
        glued_b = jnp.zeros((c_chunk,))
        return (
            tuple(glued_w[:, offsets[j] : offsets[j + 1]] for j in range(len(slot_cs))),
            tuple(glued_b[offsets[j] : offsets[j + 1]] for j in range(len(slot_cs))),
        )

    out_ws, out_bs = head(out_key)
    # The hidden head folds off `out_key` rather than widening the split: the split COUNT
    # determines every derived key, so taking one more would redraw the trunk and move the
    # equivalence goldens. Folding leaves a single-role fn bit-identical AND makes a dual
    # run's trunk and output head bit-identical to a single-role run at the same seed — so
    # the two topologies are comparable from step 0.
    hidden_out_ws, hidden_out_bs = head(jax.random.fold_in(out_key, 1)) if dual else ((), ())
    return ChunkTransformer(
        in_proj_w=kaiming(in_key, (total_d_in, d), total_d_in, relu_gain),
        in_proj_b=jnp.zeros((d,)),
        blocks=[block(bk) for bk in block_keys],
        out_ws=out_ws,
        out_bs=out_bs,
        hidden_out_ws=hidden_out_ws,
        hidden_out_bs=hidden_out_bs,
    )


def init_chunkwise_transformer_ci_fn(
    arch: ChunkwiseTransformerCIArch,
    sites: tuple[SiteSpec, ...],
    key: PRNGKeyArray,
    *,
    dual: bool = False,
) -> ChunkwiseTransformerCIFn:
    """Validate the output partition + chunk homogeneity, then build STACKED chunk params.

    - partition: the chunks' output sites are disjoint and cover every model site.
    - homogeneity: equal tap count (→ equal total input width) and an identical per-SLOT C
      tuple (same C-per-output-site in the same ORDER) across every chunk, so the per-chunk
      params — including the per-slot output heads — stack rectangularly along the scanned
      `n_chunks` axis. The per-slot heads stack slot-by-slot, so a mismatched C ORDER would
      silently misalign sites across chunks: fail fast.
    """
    site_c = {s.name: s.C for s in sites}
    covered = [name for ch in arch.chunks for name in ch.output_sites]
    assert sorted(covered) == sorted(s.name for s in sites), "chunks must partition sites"
    assert len(covered) == len(set(covered)), "chunks overlap on an output site"
    slot_cs_per_chunk = {tuple(site_c[n] for n in ch.output_sites) for ch in arch.chunks}
    assert len(slot_cs_per_chunk) == 1, (
        f"chunks not homogeneous in per-slot C tuple (the per-slot heads stack slot-by-slot "
        f"across chunks — equal C-per-site ORDER required): {slot_cs_per_chunk}"
    )
    (slot_cs,) = slot_cs_per_chunk
    assert all(ch.input_taps for ch in arch.chunks), "each chunk needs at least one input tap"
    # Per-chunk cat width must equal `arch.input_dim` (lab guarantees it; the runtime
    # `jnp.stack` / in_proj einsum fails loud if a chunk's taps don't sum to it).

    assert arch.n_blocks >= 0, (
        f"n_blocks must be >= 0 ({arch.n_blocks}); 0 is the legitimate position-local arch — "
        "in_proj + output heads, no attention — see ChunkwiseTransformerCIArch"
    )
    n_heads = arch.attention.n_heads
    hd = arch.d_model // n_heads
    assert arch.d_model % n_heads == 0 and hd % 2 == 0, (arch.d_model, n_heads)
    inv_freq = 1.0 / (10000.0 ** (jnp.arange(0, hd, 2, dtype=jnp.float32) / hd))

    # vmap over the per-chunk keys instead of unrolling n_chunks python-side inits and
    # stacking: bit-identical draws (same fold_in key per chunk), same stacked layout, but
    # the init graph is ONE chunk's RNG body — the unrolled form's XLA compile time grows
    # with chunk count (multi-minute at tens of chunks).
    chunk_keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(len(arch.chunks)))
    stacked: ChunkTransformer = eqx.filter_vmap(
        lambda k: _init_chunk_transformer(arch, arch.input_dim, slot_cs, k, dual=dual)
    )(chunk_keys)

    return ChunkwiseTransformerCIFn(
        chunks=stacked,
        inv_freq=inv_freq,
        capture_keys=arch.capture_keys,
        output_names=tuple(name for ch in arch.chunks for name in ch.output_sites),
        chunk_meta=tuple(_ChunkMeta(ch.input_taps, ch.output_sites) for ch in arch.chunks),
        eps=CI_FN_RMS_EPS,
        has_position_axis=True,
        roles=roles_for(dual),
    )


# ------------- per-site / global MLPs (pointwise over every leading axis) -------------


# The MLP arches bind their config to a target at the composition root. Their input taps
# (`input_names` / `input_taps`) are therefore resolved exactly once, like the chunkwise
# architecture's authored tap union, and every downstream consumer reads the same
# authoritative field.
@dataclass(frozen=True)
class LayerwiseMLPCIArch:
    """Hidden widths shared by every per-site MLP.

    `has_position_axis` is the TARGET's shape, not a property of the MLP: the stack is
    pointwise over every leading axis, so the same weights serve `[batch, d]` and
    `[batch, position, d]` alike. It is declared here so the CI fn and the model can be
    checked to agree (`core.run_state.init_decomposition`)."""

    hidden_dims: tuple[int, ...]
    has_position_axis: bool
    input_names: tuple[str, ...]
    dual: bool = False
    """Build a SECOND readout head on this same trunk (SPEC S36), so the CI fn scores both
    the output and the hidden reconstruction objectives. An ARCH property, not a build-time
    flag: it changes the parameter tree, so it must ride the arch through checkpointing and
    the run bundle rather than being re-supplied at every construction site."""

    @property
    def capture_keys(self) -> CaptureKeys:
        return frozenset(self.input_names)


class SiteMLP(eqx.Module):
    """`hidden_dims` Linear+GELU layers then a linear head: Kaiming-`relu` (`gain √2`)
    hidden layers with zero bias, linear-gain (`1`) final head."""

    weights: list[Float[Array, "d_in d_out"]]
    biases: list[Float[Array, " d_out"]]
    hidden_head: tuple[Float[Array, "d_in C"], Float[Array, " C"]] | None
    """The SECOND readout head (SPEC S36) — a replacement for the FINAL layer only, `None` in a
    single-role fn. The GELU stack up to it is the TRUNK and is shared by construction: it
    appears once in the pytree, so both roles read one representation and one pullback sums
    both heads' trunk gradients. Mirrors `ChunkTransformer`'s split — the head is the only
    part that must be private, because it is where "how much does this subcomponent matter FOR
    THIS OBJECTIVE" lives."""

    def shardings(self, mesh: Mesh) -> "SiteMLP":
        """Each `[d_in, d_out]` weight shards its OUTPUT axis (axis 1) ÷N over the FULL mesh
        (`("replicate","fsdp")`) — the master + Adam state shard ÷N. 1-D biases replicate.
        The toy MLP is single-shot (no scan), so there is no compute reconstruction; GSPMD
        gathers as needed (trivial at the toy's small device count). Asserts every output dim
        tiles the device count."""
        shard_out = NamedSharding(mesh, P(None, ("replicate", "fsdp")))
        repl = NamedSharding(mesh, P())
        n = mesh.devices.size
        for layer_idx, w in enumerate(self.weights):
            assert w.shape[1] % n == 0, (
                f"SiteMLP.weights[{layer_idx}].d_out {w.shape[1]} not ÷ N={n}"
            )
        sharded = eqx.tree_at(
            lambda m: (m.weights, m.biases),
            self,
            ([shard_out] * len(self.weights), [repl] * len(self.biases)),
        )
        if self.hidden_head is not None:
            assert self.hidden_head[0].shape[1] % n == 0, (
                f"SiteMLP.hidden_head C {self.hidden_head[0].shape[1]} not ÷ N={n}"
            )
            sharded = eqx.tree_at(
                lambda m: m.hidden_head, sharded, (shard_out, repl), is_leaf=lambda x: x is None
            )
        return sharded

    def __call__(self, x: Float[Array, "*leading d_in"]) -> Float[Array, "*leading C"]:
        return self.role_preactivations(x)[0]

    def role_preactivations(
        self, x: Float[Array, "*leading d_in"]
    ) -> tuple[Float[Array, "*leading C"], ...]:
        """One preactivation per role: `(output,)`, or `(output, hidden)` when a hidden head
        exists. The trunk runs ONCE and both heads read its output."""
        n_hidden = len(self.weights) - 1
        for layer_idx, (w, b) in enumerate(zip(self.weights[:-1], self.biases[:-1], strict=True)):
            x = einops.einsum(x, w, "... i, i o -> ... o") + b
            if layer_idx < n_hidden:
                x = jax.nn.gelu(x, approximate=False)

        def readout(w: Array, b: Array) -> Array:
            return einops.einsum(x, w, "... i, i o -> ... o") + b

        output = readout(self.weights[-1], self.biases[-1])
        if self.hidden_head is None:
            return (output,)
        return (output, readout(*self.hidden_head))


class LayerwiseMLPCIFn(eqx.Module):
    """One MLP per site, with input taps aligned to output sites by position."""

    site_mlps: dict[str, SiteMLP]
    input_names: tuple[str, ...] = eqx.field(static=True)
    output_names: tuple[str, ...] = eqx.field(static=True)
    has_position_axis: bool = eqx.field(static=True)
    roles: tuple[CIRole, ...] = eqx.field(static=True)

    @property
    def capture_keys(self) -> CaptureKeys:
        return frozenset(self.input_names)

    def shardings(self, mesh: Mesh) -> "LayerwiseMLPCIFn":
        return eqx.tree_at(
            lambda f: f.site_mlps,
            self,
            {name: mlp.shardings(mesh) for name, mlp in self.site_mlps.items()},
        )

    def site_preactivations(self, taps: dict[str, Array]) -> dict[str, Array]:
        return self.role_site_preactivations(taps)[0]

    def role_site_preactivations(self, taps: dict[str, Array]) -> tuple[dict[str, Array], ...]:
        """One site-dict per role, in `roles` order. Each site's trunk runs once."""
        assert set(taps) == set(self.input_names), (
            f"tap keys {sorted(taps)} != CI fn inputs {sorted(self.input_names)}"
        )
        per_site = {
            output_name: self.site_mlps[output_name].role_preactivations(taps[input_name])
            for input_name, output_name in zip(self.input_names, self.output_names, strict=True)
        }
        return tuple(
            {name: roles[role_idx] for name, roles in per_site.items()}
            for role_idx in range(len(self.roles))
        )

    def __call__(self, taps: dict[str, Array], *, remat: bool) -> AnyCI:
        del remat  # single-shot (no scan to bound) -> remat is a no-op for the MLP CI fns
        return _bundle(self.roles, self.role_site_preactivations(taps))


def _init_mlp_stack(dims: tuple[int, ...], key: PRNGKeyArray, *, dual: bool = False) -> SiteMLP:
    """One `Linear+GELU` stack `dims[0] -> ... -> dims[-1]`: Kaiming `relu`-gain (`√2`) on
    the hidden layers, linear gain (`1`) on the final head, zero biases.

    A dual stack adds a SECOND final head drawn off a fold of the final head's own key, so the
    trunk and the output head stay bit-identical to a single-role stack at the same seed (the
    split count is unchanged) — same rationale as `_init_chunk_transformer`."""
    relu_gain = 2.0**0.5
    layer_keys = jax.random.split(key, len(dims) - 1)
    weights: list[Array] = []
    biases: list[Array] = []
    for layer_idx, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        gain = relu_gain if layer_idx < len(dims) - 2 else 1.0
        weights.append(jax.random.normal(layer_keys[layer_idx], (d_in, d_out)) * (gain / d_in**0.5))
        biases.append(jnp.zeros((d_out,)))
    hidden_head = None
    if dual:
        d_in, d_out = dims[-2], dims[-1]
        hidden_w = (
            jax.random.normal(jax.random.fold_in(layer_keys[-1], 1), (d_in, d_out)) / d_in**0.5
        )
        hidden_head = (hidden_w, jnp.zeros((d_out,)))
    return SiteMLP(weights=weights, biases=biases, hidden_head=hidden_head)


def init_layerwise_mlp_ci_fn(
    arch: LayerwiseMLPCIArch,
    sites: tuple[SiteSpec, ...],
    key: PRNGKeyArray,
    *,
    dual: bool = False,
) -> LayerwiseMLPCIFn:
    """Per-site MLP init: each site's MLP maps `d_in -> hidden_dims... -> C`."""
    assert arch.hidden_dims, "MLP CI fn needs at least one hidden layer"
    site_mlps = {
        spec.name: _init_mlp_stack(
            (spec.d_in, *arch.hidden_dims, spec.C), jax.random.fold_in(key, site_idx), dual=dual
        )
        for site_idx, spec in enumerate(sites)
    }
    output_names = tuple(s.name for s in sites)
    assert len(arch.input_names) == len(output_names), (arch.input_names, output_names)
    assert len(set(arch.input_names)) == len(arch.input_names), arch.input_names
    return LayerwiseMLPCIFn(
        site_mlps=site_mlps,
        input_names=arch.input_names,
        output_names=output_names,
        has_position_axis=arch.has_position_axis,
        roles=roles_for(dual),
    )


@dataclass(frozen=True)
class TapSpec:
    """One input tap: its capture key and feature width. The key is opaque to core (the
    lab authors it, the target resolves it); the width rides alongside so the consumer
    can size and assert its input without deriving it from a site."""

    key: str
    width: int


@dataclass(frozen=True)
class GlobalMLPCIArch:
    """Hidden widths of the single global MLP shared across ALL sites, plus the input
    taps it concatenates. The taps are DECOUPLED from the output sites: several sites may
    read one physical tap (an LM block's q/k/v share its attention input), so the taps
    are unique keys with explicit widths, never a per-site alignment
    (`LayerwiseMLPCIArch` keeps that alignment — there it is real)."""

    hidden_dims: tuple[int, ...]
    has_position_axis: bool
    input_taps: tuple[TapSpec, ...]
    dual: bool = False
    """Build a SECOND readout head on this same trunk (SPEC S36), so the CI fn scores both
    the output and the hidden reconstruction objectives. An ARCH property, not a build-time
    flag: it changes the parameter tree, so it must ride the arch through checkpointing and
    the run bundle rather than being re-supplied at every construction site."""

    @property
    def capture_keys(self) -> CaptureKeys:
        return frozenset(tap.key for tap in self.input_taps)


class GlobalMLPCIFn(eqx.Module):
    """ONE shared MLP over all sites behind the `CIFn` protocol. The taps are
    concatenated in `input_taps` order into `[*leading, Σ width]`, mapped to `[*leading,
    Σ C]`, and split back per output site by `c_sizes` in `output_names` order — so every
    site's preactivations depend on every tap."""

    mlp: SiteMLP
    input_taps: tuple[TapSpec, ...] = eqx.field(static=True)
    output_names: tuple[str, ...] = eqx.field(static=True)
    c_sizes: tuple[int, ...] = eqx.field(static=True)
    has_position_axis: bool = eqx.field(static=True)
    roles: tuple[CIRole, ...] = eqx.field(static=True)

    @property
    def capture_keys(self) -> CaptureKeys:
        return frozenset(tap.key for tap in self.input_taps)

    def shardings(self, mesh: Mesh) -> "GlobalMLPCIFn":
        return eqx.tree_at(lambda f: f.mlp, self, self.mlp.shardings(mesh))

    def site_preactivations(self, taps: dict[str, Array]) -> dict[str, Array]:
        return self.role_site_preactivations(taps)[0]

    def role_site_preactivations(self, taps: dict[str, Array]) -> tuple[dict[str, Array], ...]:
        """One site-dict per role, in `roles` order. The shared MLP trunk runs once."""
        assert set(taps) == {tap.key for tap in self.input_taps}, (
            f"tap keys {sorted(taps)} != CI fn inputs {sorted(t.key for t in self.input_taps)}"
        )
        for tap in self.input_taps:
            assert taps[tap.key].shape[-1] == tap.width, (
                f"tap {tap.key} width {taps[tap.key].shape[-1]} != expected {tap.width}"
            )
        concatenated = jnp.concatenate([taps[tap.key] for tap in self.input_taps], axis=-1)
        offsets = [0]
        for c in self.c_sizes:
            offsets.append(offsets[-1] + c)
        return tuple(
            {
                name: preactivations[..., offsets[i] : offsets[i + 1]]
                for i, name in enumerate(self.output_names)
            }
            for preactivations in self.mlp.role_preactivations(concatenated)
        )

    def __call__(self, taps: dict[str, Array], *, remat: bool) -> AnyCI:
        del remat  # single-shot (no scan to bound) -> remat is a no-op for the MLP CI fns
        return _bundle(self.roles, self.role_site_preactivations(taps))


def init_global_mlp_ci_fn(
    arch: GlobalMLPCIArch,
    sites: tuple[SiteSpec, ...],
    key: PRNGKeyArray,
    *,
    dual: bool = False,
) -> GlobalMLPCIFn:
    """Global MLP init: one stack `Σ tap width -> hidden_dims... -> Σ C`, same Kaiming
    scheme as the per-site MLP."""
    assert arch.hidden_dims, "global MLP CI fn needs at least one hidden layer"
    tap_keys = tuple(tap.key for tap in arch.input_taps)
    assert tap_keys and len(set(tap_keys)) == len(tap_keys), tap_keys
    c_sizes = tuple(s.C for s in sites)
    dims = (sum(tap.width for tap in arch.input_taps), *arch.hidden_dims, sum(c_sizes))
    return GlobalMLPCIFn(
        mlp=_init_mlp_stack(dims, key, dual=dual),
        input_taps=arch.input_taps,
        output_names=tuple(s.name for s in sites),
        c_sizes=c_sizes,
        has_position_axis=arch.has_position_axis,
        roles=roles_for(dual),
    )


# ----------------------------- construction (placement-agnostic) -----------------------------


CIFnArch = ChunkwiseTransformerCIArch | LayerwiseMLPCIArch | GlobalMLPCIArch
"""Every CI-fn architecture. Construction goes through `build_ci_fn`; sharding/placement is
a separate, scale-driven concern (see `init_placed`), never coupled to arch type."""


def build_ci_fn(arch: CIFnArch, sites: tuple[SiteSpec, ...], key: PRNGKeyArray) -> CIFn:
    """Construct the CI fn for `arch`, host-side and unsharded. Placement is applied by the
    caller by SCALE (mesh × C-divisibility), never by which arch this is.

    `arch.dual` builds the second readout head off the SAME trunk (SPEC S36) — every impl
    splits at its own final readout, so a dual run can use any of them. It rides the ARCH
    rather than being a build-time flag because it changes the parameter tree: the run bundle
    and the checkpoint must agree on it without anyone re-supplying it."""
    match arch:
        case ChunkwiseTransformerCIArch():
            return init_chunkwise_transformer_ci_fn(arch, sites, key, dual=arch.dual)
        case LayerwiseMLPCIArch():
            return init_layerwise_mlp_ci_fn(arch, sites, key, dual=arch.dual)
        case GlobalMLPCIArch():
            return init_global_mlp_ci_fn(arch, sites, key, dual=arch.dual)
