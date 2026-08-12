"""The decomposition representation, shared by every target (LM and toy alike).

`SiteC` / `SiteDims` / `SiteSpec` are the per-site shape primitives (configured name+C,
matrix dimensions, and the combined shape-carrying spec); `ComponentStacks` is the trainable
master pytree, persisted as same-shape STACKS (owner-partitioned layout); `init_component_stacks` seeds it; `site_out` is the one
decomposed-linear primitive (SPEC §4.1, `((x@V)*m)@U + (x@Δ)*d`). These are domain-neutral
— they depend only on the site shapes and the V/U/W arrays — so they live here rather than
inside `model.py` (whose `DecomposedModel` Protocol references `ComponentStacks`/`SiteSpec`) or any
one target.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from typing import ClassVar, Generic

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array
from typing_extensions import TypeVar


@dataclass(frozen=True)
class SiteC:
    """A decomposed site as configured: its torch-module-path name and its C.

    The shape-carrying `SiteSpec` is derived from this plus the target's config."""

    name: str
    C: int


@dataclass(frozen=True, kw_only=True)
class SiteDims:
    d_in: int
    d_out: int


@dataclass(frozen=True)
class SiteSpec:
    name: str
    d_in: int
    d_out: int
    C: int


@dataclass(frozen=True)
class SiteComponents:
    """The two rank-one factor matrices for one decomposed site."""

    V: Array
    U: Array


VUShape = tuple[int, int, int]  # (d_in, d_out, C)

# site name -> (shape group, slot on the group's stack axis); static, canonical site order
SiteSlots = tuple[tuple[str, VUShape, int], ...]

# The V/U leaf type: `Array` for the real fp32 masters (the default — so bare `ComponentStacks`
# means `ComponentStacks[Array]` and no call site needs the parameter), or `NamedSharding` for
# the same-structure placement tree `placement.component_stacks_shardings` returns for
# `jax.jit(out_shardings=...)`.
VULeaf = TypeVar("VULeaf", default=Array)


def vu_shape_groups(sites: tuple[SiteSpec, ...]) -> dict[VUShape, tuple[SiteSpec, ...]]:
    """Sites grouped by V/U shape, site order preserved within a group."""
    groups: dict[VUShape, list[SiteSpec]] = {}
    for spec in sites:
        groups.setdefault((spec.d_in, spec.d_out, spec.C), []).append(spec)
    return {shape: tuple(specs) for shape, specs in groups.items()}


def site_slots_for(sites: tuple[SiteSpec, ...]) -> SiteSlots:
    """The canonical site→(shape, slot) mapping: slots follow site order within each shape
    group (`vu_shape_groups` preserves it), entries follow overall site order."""
    by_name: dict[str, tuple[VUShape, int]] = {}
    for shape, specs in vu_shape_groups(sites).items():
        for slot, spec in enumerate(specs):
            by_name[spec.name] = (shape, slot)
    return tuple((spec.name, *by_name[spec.name]) for spec in sites)


@cache
def _slot_index(site_slots: SiteSlots) -> dict[str, tuple[VUShape, int]]:
    return {name: (shape, slot) for name, shape, slot in site_slots}


class ComponentStacks(eqx.Module, Generic[VULeaf]):
    """The trainable V/U masters, persisted as same-shape STACKS — the owner-partitioned
    layout: one `(Vs [g, d_in, C], Us [g, C, d_out])` pair per `(d_in, d_out, C)` shape
    group, with `site_slots` (static) mapping each site name to its slot on the stack axis.

    Why stacks: per-matrix ownership is only expressible under SPMD by sharding a STACK
    axis (a per-site leaf cannot live wholly on one rank), the llama8b scan consumes
    per-kind stacks natively, and the checkpoint/init trees shrink from `2·n_sites` leaves
    to `2·n_shapes`. Per-site access is `site(name)` — a static slice of the stack.

    Leaves are fp32 master Arrays (`ComponentStacks[Array]`) or `NamedSharding`s in the
    same-structure placement tree `placement.component_stacks_shardings` returns
    (`ComponentStacks[NamedSharding]`). This module is placement-FREE: the per-group row
    lookup and its boundary validation live in `placement.py`, above."""

    stacks: dict[VUShape, tuple[VULeaf, VULeaf]]
    site_slots: SiteSlots = eqx.field(static=True)

    def site(self: "ComponentStacks[Array]", name: str) -> SiteComponents:
        shape, slot = _slot_index(self.site_slots)[name]
        Vs, Us = self.stacks[shape]
        return SiteComponents(V=Vs[slot], U=Us[slot])

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(name for name, _, _ in self.site_slots)

    def sites_items(self: "ComponentStacks[Array]") -> Iterator[tuple[str, SiteComponents]]:
        """Named site components in canonical site order."""
        for name, _, _ in self.site_slots:
            yield name, self.site(name)

    V_AXES: "ClassVar[tuple[str, ...]]" = ("stack", "d_in", "C")
    U_AXES: "ClassVar[tuple[str, ...]]" = ("stack", "C", "d_out")

    def group_lengths(self) -> dict[VUShape, int]:
        """Stack length per shape group, from the STATIC index (works on eval-shape
        trees) — what the placement boundary validates the received row assignment
        against (`placement.component_stacks_shardings` / `_audit`)."""
        lengths: dict[VUShape, int] = {}
        for _name, shape, slot in self.site_slots:
            lengths[shape] = max(lengths.get(shape, 0), slot + 1)
        return lengths


def init_stack_arrays(
    sites: tuple[SiteSpec, ...], key: Array
) -> dict[VUShape, tuple[Array, Array]]:
    """Seeded stack init: `{(d_in, d_out, C): (V [g, d_in, C], U [g, C, d_out])}`, vmapped
    over per-site keys drawn in site order — each site's slice is BIT-IDENTICAL to the
    retired per-site init's draw (pinned by `test_sharding`). Under jit the graph has
    2×n_shapes sharded outputs instead of 2×n_sites, which keeps the init compile from
    scaling with site count."""
    keys = jax.random.split(key, 2 * len(sites))
    site_index = {spec.name: idx for idx, spec in enumerate(sites)}
    stacked: dict[VUShape, tuple[Array, Array]] = {}
    for (d_in, d_out, c), specs in vu_shape_groups(sites).items():
        idxs = jnp.array([site_index[spec.name] for spec in specs])
        Vs = jax.vmap(lambda k, s=(d_in, c): jax.random.normal(k, s))(keys[2 * idxs])
        Us = jax.vmap(lambda k, s=(c, d_out): jax.random.normal(k, s))(keys[2 * idxs + 1])
        stacked[(d_in, d_out, c)] = (Vs * d_in**-0.5, Us * c**-0.5)
    return stacked


def component_stacks_from_sites(vu: dict[str, tuple[Array, Array]]) -> ComponentStacks:
    """Build the stacked `ComponentStacks` from a per-site `{name: (V, U)}` dict (site order =
    dict order). The explicit-arrays constructor for toys and tests; the trainer inits
    directly in the stacked layout (`init_component_stacks`)."""
    sites = tuple(
        SiteSpec(name=name, d_in=V.shape[0], d_out=U.shape[1], C=V.shape[1])
        for name, (V, U) in vu.items()
    )
    stacks = {
        shape: (
            jnp.stack([vu[s.name][0] for s in specs]),
            jnp.stack([vu[s.name][1] for s in specs]),
        )
        for shape, specs in vu_shape_groups(sites).items()
    }
    return ComponentStacks(stacks=stacks, site_slots=site_slots_for(sites))


def init_component_stacks(sites: tuple[SiteSpec, ...], key: Array) -> ComponentStacks:
    """Small random fp32 V ~ N(0, d_in^-0.5), U ~ N(0, C^-0.5) per site, built directly in
    the stacked persistence layout; the weight-delta channel carries the faithfulness
    residual at init (before faithfulness warmup)."""
    return ComponentStacks(stacks=init_stack_arrays(sites, key), site_slots=site_slots_for(sites))


def zero_component_stacks(sites: tuple[SiteSpec, ...]) -> ComponentStacks:
    """All-zero V/U in the stacked layout. `weight_deltas` is `W − (V@U)^T`, so passing this
    reads each site's frozen `W` back through the model protocol without widening it."""
    return component_stacks_from_sites(
        {
            spec.name: (
                jnp.zeros((spec.d_in, spec.C), jnp.float32),
                jnp.zeros((spec.C, spec.d_out), jnp.float32),
            )
            for spec in sites
        }
    )


def _coupled_site_vu(W: Array, key: Array, C: int) -> tuple[Array, Array]:
    """One site's coupled V/U from its frozen `W [d_out, d_in]`: a unit-norm Gaussian seed on
    the NARROW side, the wide side its raw `W`-image. No C-dependent rescale — components sit
    at `W`'s natural scale."""
    d_out, d_in = W.shape
    if d_in <= d_out:
        v = jax.random.normal(key, (d_in, C))
        V = v / jnp.linalg.norm(v, axis=0, keepdims=True)
        return V, (W @ V).T
    u = jax.random.normal(key, (C, d_out))
    U = u / jnp.linalg.norm(u, axis=1, keepdims=True)
    return W.T @ U.T, U


def init_component_stacks_coupled(
    sites: tuple[SiteSpec, ...], target_weights: dict[str, Array], key: Array
) -> ComponentStacks:
    """Coupled seeded init: per site, `_coupled_site_vu` against its own frozen `W`.

    Per-SITE, not vmapped over a shape group: vmap would need the group's `W`s stacked into
    one contiguous buffer (`[2, 14336, 4096]` fp32 = 470MB for llama8b's gate+up alone) and
    would keep every matrix in the group simultaneously live. The draw here is one matmul
    and one norm — there is no RNG fan-out to amortize — so the loop costs nothing and lets
    XLA free each `W` before the next. Outputs still stack per shape group, so the graph
    keeps `init_stack_arrays`' 2×n_shapes shape.
    """
    keys = jax.random.split(key, len(sites))
    return component_stacks_from_sites(
        {
            spec.name: _coupled_site_vu(
                target_weights[spec.name].astype(jnp.float32), keys[idx], spec.C
            )
            for idx, spec in enumerate(sites)
        }
    )


def with_silenced_u(components: ComponentStacks) -> ComponentStacks:
    """U zeroed, V untouched — the `zero_u` arm's one post-transform over any seeded init.

    The component sum is then exactly zero and the delta carries all of `W`, while `x @ V`
    still feeds the CI nets a live signal and `U` has a nonzero gradient from step 0 (`V`'s
    is zero until `U` moves off zero)."""
    return ComponentStacks(
        stacks={shape: (Vs, jnp.zeros_like(Us)) for shape, (Vs, Us) in components.stacks.items()},
        site_slots=components.site_slots,
    )


def site_out(
    x: Array,
    V: Array,
    U: Array,
    W: Array,
    mask: Array | None,
    delta_mask: Array | None,
    route: Array | None,
) -> Array:
    """One decomposed linear (SPEC §4.1): `((x@V)*m)@U + (x@Δ)*d`, routed per position
    against the frozen `x @ W.T`. `mask` may be None (fully on); `route` None routes
    everywhere. `delta_mask` None drops the delta path entirely (constant-source entries
    carry no delta, LOSS_PARITY_DESIGN §4b). `delta_mask`/`route` broadcast over batch;
    trailing dim added here."""
    # Pin the decomposed matmuls DATA-PARALLEL over the FULL mesh: the d_in/d_out-space
    # activation `x` stays batch-sharded over `('replicate', 'fsdp')`, feature-replicated, and
    # the component-space activation `x@V` stays batch-sharded, C-REPLICATED (no TP axis).
    # This forces the sharded V/U masters to be GATHERED for compute and their grads
    # reduced back to the master layout (symmetric — the intended layout). WITHOUT pinning
    # `x`, the weight-grad backward is free to instead shard `x`'s feature dim and REPLICATE
    # the batch (the forward gathers V, the backward does not), which GSPMD can't reshard
    # cheaply -> involuntary full rematerialization -> OOM. Pinning the activations (not
    # V/U) keeps the weights as plain matmul args. Guarded so it's a no-op off-mesh (CPU
    # tests / single device); `run.py` sets the global mesh. waist is `[*leading, d]`,
    # leading = (batch, *position): pin batch over the full mesh (positions + feature
    # replicated for `x`; C replicated for `x@V`).
    on_mesh = not jax.sharding.get_abstract_mesh().empty
    batch_axes = ("replicate", "fsdp")
    if on_mesh:
        x = jax.lax.with_sharding_constraint(x, P(batch_axes, *(None,) * (x.ndim - 1)))
    xV = x @ V
    if on_mesh:
        # batch over the data axes; the component C axis (last) on `tp` (Megatron-C) — so the
        # `x@V` weight is ÷tp and the `xV * mask` stays local (mask is C-on-tp too).
        xV = jax.lax.with_sharding_constraint(xV, P(batch_axes, *(None,) * (xV.ndim - 2), "tp"))
    acts = xV * mask if mask is not None else xV
    out = acts @ U
    if on_mesh:
        # `acts @ U` contracts the tp-sharded C → reduce over `tp`; pin the output d-full /
        # batch-sharded so the tp-reduce + fsdp-gather are symmetric and the weight-grad
        # backward reshards the same way (avoids the involuntary-remat OOM —
        # PLACEMENT_DESIGN.md lesson 3).
        out = jax.lax.with_sharding_constraint(out, P(batch_axes, *(None,) * (out.ndim - 1)))
    if delta_mask is not None:
        # `(x @ Δ.T)` for `Δ = W − (V@U).T`, expanded to activation space as
        # `x@W.T − (x@V)@U` so the `[d_out, d_in]` weight delta is NEVER formed. Under
        # FSDP that delta would mix V's sharded d_in with U's sharded d_out (two dims
        # demanding the data axis) and force a replicate-then-repartition reshard; the
        # activation-space form is all activation×weight matmuls that shard cleanly.
        # (Still a bf16-rounding DIVERGENCE vs the fp32 oracle delta — accepted; the
        # faithfulness loss uses the fp32 `weight_deltas`, SPEC N2, not this path.)
        out = out + delta_mask[..., None] * (x @ W.T - xV @ U)
    if route is not None:
        out = jnp.where(route[..., None], out, x @ W.T)
    return out
