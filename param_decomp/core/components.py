"""The decomposition representation, shared by every target (LM and toy alike).

`SiteC` / `SiteDims` / `SiteSpec` are the per-site shape primitives (configured name+C,
matrix dimensions, and the combined shape-carrying spec); `ComponentStacks` is the trainable
master pytree, grouped by target-declared semantic role; `init_component_stacks` seeds it.
These are domain-neutral — they depend only on the site shapes and the V/U arrays — so they
live here rather than inside `model.py` (whose `DecomposedModel` Protocol references
`ComponentStacks`/`SiteSpec`) or any one target. Executing a decomposed site is placement's
business, above: `decomposed_linear.site_forward`.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cache
from typing import ClassVar, Generic, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from typing_extensions import TypeVar

from param_decomp.core.axes import Axes, SemanticAxis
from param_decomp.core.nonlinearity import (
    AttentionHeads,
    Neurons,
    NonlinearityPartition,
)


def activation_axes(ndim: int, feature: SemanticAxis) -> Axes:
    """THE semantic axis names of a waist activation `[batch, *positions, feature]`.
    Placement lookups are exact-name; every consumer derives the tuple here so a
    misspelled feature axis (silent replication) has no second spelling to hide in.
    The waist comes in exactly TWO shapes (`model.py`) — positionless or one position
    axis — so the position vocabulary is the enumeration below, not an open family."""
    match ndim:
        case 2:
            return ("batch", feature)
        case 3:
            return ("batch", "position", feature)
        case _:
            raise AssertionError(ndim)


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
    group: str
    nonlinearity_partition: NonlinearityPartition | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        match self.nonlinearity_partition:
            case AttentionHeads(head_count=head_count):
                assert self.d_out % head_count == 0, self
            case Neurons() | None:
                pass


def nonlinearity_partitions(sites: tuple[SiteSpec, ...]) -> dict[str, NonlinearityPartition]:
    return {s.name: s.nonlinearity_partition for s in sites if s.nonlinearity_partition is not None}


@dataclass(frozen=True)
class SiteComponents:
    """The two rank-one factor matrices for one decomposed site."""

    V: Array
    U: Array


VUShape = tuple[int, int, int]  # (d_in, d_out, C)

# site name -> (target-declared group, slot on the group's stack axis)
SiteSlots = tuple[tuple[str, str, int], ...]

# The V/U leaf type: `Array` for the real fp32 masters (the default — so bare `ComponentStacks`
# means `ComponentStacks[Array]` and no call site needs the parameter), or `NamedSharding` for
# the same-structure placement tree `placement.component_stacks_shardings` returns for
# `jax.jit(out_shardings=...)`.
VULeaf = TypeVar("VULeaf", default=Array)


def vu_groups(sites: tuple[SiteSpec, ...]) -> dict[str, tuple[SiteSpec, ...]]:
    """Sites grouped by the target's semantic persistence group."""
    groups: dict[str, list[SiteSpec]] = {}
    for spec in sites:
        groups.setdefault(spec.group, []).append(spec)
    for group, specs in groups.items():
        shapes = {(spec.d_in, spec.d_out, spec.C) for spec in specs}
        assert len(shapes) == 1, f"component group {group!r} mixes V/U shapes: {sorted(shapes)}"
    return {group: tuple(specs) for group, specs in groups.items()}


def group_shape(specs: tuple[SiteSpec, ...]) -> VUShape:
    assert specs
    return (specs[0].d_in, specs[0].d_out, specs[0].C)


def site_slots_for(sites: tuple[SiteSpec, ...]) -> SiteSlots:
    """The canonical site→(group, slot) mapping in site order."""
    by_name: dict[str, tuple[str, int]] = {}
    for group, specs in vu_groups(sites).items():
        for slot, spec in enumerate(specs):
            by_name[spec.name] = (group, slot)
    return tuple((spec.name, *by_name[spec.name]) for spec in sites)


@cache
def _slot_index(site_slots: SiteSlots) -> dict[str, tuple[str, int]]:
    return {name: (group, slot) for name, group, slot in site_slots}


class ComponentStacks(eqx.Module, Generic[VULeaf]):
    """The trainable V/U masters: one homogeneous stack per target-declared semantic group.

    A group holds `(Vs [g, d_in, C], Us [g, C, d_out])`; `site_slots` maps each site to
    its slot. LM targets declare matrix kind as the group, making each scan input a leaf.
    Toy targets may declare independent per-site groups. Placement is separate: a rule may
    shard the stack axis for ownership or shard matrix dimensions instead.

    Leaves are fp32 master Arrays (`ComponentStacks[Array]`) or `NamedSharding`s in the
    same-structure placement tree `placement.component_stacks_shardings` returns
    (`ComponentStacks[NamedSharding]`). This module is placement-FREE: the per-group row
    lookup and its boundary validation live in `placement.py`, above."""

    stacks: dict[str, tuple[VULeaf, VULeaf]]
    site_slots: SiteSlots = eqx.field(static=True)

    def slot_of(self, name: str) -> tuple[str, int]:
        return _slot_index(self.site_slots)[name]

    def site(self: "ComponentStacks[Array]", name: str) -> SiteComponents:
        group, slot = self.slot_of(name)
        Vs, Us = self.stacks[group]
        return SiteComponents(V=Vs[slot], U=Us[slot])

    @property
    def site_names(self) -> tuple[str, ...]:
        return tuple(name for name, _, _ in self.site_slots)

    def sites_items(self: "ComponentStacks[Array]") -> Iterator[tuple[str, SiteComponents]]:
        """Named site components in canonical site order."""
        for name, _, _ in self.site_slots:
            yield name, self.site(name)

    V_AXES: ClassVar[Axes] = ("stack", "d_in", "C")
    U_AXES: ClassVar[Axes] = ("stack", "C", "d_out")

    def group_lengths(self) -> dict[str, int]:
        """Stack length per semantic group, available from eval-shape trees."""
        lengths: dict[str, int] = {}
        for _name, group, slot in self.site_slots:
            lengths[group] = max(lengths.get(group, 0), slot + 1)
        return lengths


def init_stack_arrays(sites: tuple[SiteSpec, ...], key: Array) -> dict[str, tuple[Array, Array]]:
    """Seed each semantic group's V/U stacks over per-site keys drawn in site order."""
    keys = jax.random.split(key, 2 * len(sites))
    site_index = {spec.name: idx for idx, spec in enumerate(sites)}
    stacked: dict[str, tuple[Array, Array]] = {}
    for group, specs in vu_groups(sites).items():
        d_in, d_out, c = group_shape(specs)
        idxs = jnp.array([site_index[spec.name] for spec in specs])
        Vs = jax.vmap(lambda k, s=(d_in, c): jax.random.normal(k, s))(keys[2 * idxs])
        Us = jax.vmap(lambda k, s=(c, d_out): jax.random.normal(k, s))(keys[2 * idxs + 1])
        stacked[group] = (Vs * d_in**-0.5, Us * c**-0.5)
    return stacked


def component_stacks_from_site_arrays(
    sites: tuple[SiteSpec, ...], vu: dict[str, tuple[Array, Array]]
) -> ComponentStacks:
    assert tuple(vu) == tuple(spec.name for spec in sites), (tuple(vu), sites)
    stacks = {
        group: (
            jnp.stack([vu[spec.name][0] for spec in specs]),
            jnp.stack([vu[spec.name][1] for spec in specs]),
        )
        for group, specs in vu_groups(sites).items()
    }
    return ComponentStacks(stacks=stacks, site_slots=site_slots_for(sites))


def component_stacks_from_sites(vu: dict[str, tuple[Array, Array]]) -> ComponentStacks:
    """Build independently grouped component leaves from explicit per-site arrays."""
    sites = tuple(
        SiteSpec(name=name, d_in=V.shape[0], d_out=U.shape[1], C=V.shape[1], group=name)
        for name, (V, U) in vu.items()
    )
    return component_stacks_from_site_arrays(sites, vu)


def init_component_stacks(sites: tuple[SiteSpec, ...], key: Array) -> ComponentStacks:
    """Small random fp32 V ~ N(0, d_in^-0.5), U ~ N(0, C^-0.5) per site, built directly in
    the stacked persistence layout; the weight-delta channel carries the faithfulness
    residual at init (before faithfulness warmup)."""
    return ComponentStacks(stacks=init_stack_arrays(sites, key), site_slots=site_slots_for(sites))


def zero_component_stacks(sites: tuple[SiteSpec, ...]) -> ComponentStacks:
    """All-zero V/U in the stacked layout. `weight_deltas` is `W − (V@U)^T`, so passing this
    reads each site's frozen `W` back through the model protocol without widening it."""
    return component_stacks_from_site_arrays(
        sites,
        {
            spec.name: (
                jnp.zeros((spec.d_in, spec.C), jnp.float32),
                jnp.zeros((spec.C, spec.d_out), jnp.float32),
            )
            for spec in sites
        },
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
    return component_stacks_from_site_arrays(
        sites,
        {
            spec.name: _coupled_site_vu(
                target_weights[spec.name].astype(jnp.float32), keys[idx], spec.C
            )
            for idx, spec in enumerate(sites)
        },
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


NeuronAxis = Literal["d_out", "d_in"]
"""Which axis of a site's frozen `W [d_out, d_in]` indexes nonlinearity units (neurons):
`d_out` for a hidden WRITER (`x @ W.T` lands on neurons), `d_in` for the READER (`W`
consumes neurons). Core never knows which site kind is which — the target says."""


class SiteNeuronAlignment(eqx.Module):
    """One site's neuron-aligned start (SPEC T13): `component_of_neuron` is `int32[n]`
    over the site's neuron axis — the subcomponent each neuron is assigned to, `-1` for
    unassigned. The exact (`neuron_aligned_targeted`) start assigns the top-C neurons one
    each (slot `i` = the `i`-th ranked neuron); the wrapped (`neuron_aligned_wrap`) start
    assigns EVERY neuron, rank `j` to slot `j mod C`. Traced through the placed init; the
    axis is static."""

    component_of_neuron: Array
    neuron_axis: NeuronAxis = eqx.field(static=True)


NeuronAlignment = dict[str, SiteNeuronAlignment]
"""Site name -> alignment, for the ALIGNED sites only. A site absent here takes the
`zero_u` values (`init_component_stacks_neuron_aligned`)."""


def validate_neuron_alignment(sites: tuple[SiteSpec, ...], alignment: NeuronAlignment) -> None:
    """Host-side checks the traced init cannot make: every aligned site exists, its
    assignment spans exactly the neuron axis, names only slots `-1..C-1`, and leaves no
    subcomponent without a neuron (an unassigned slot would start at exactly zero)."""
    by_name = {spec.name: spec for spec in sites}
    for name, site in alignment.items():
        assert name in by_name, f"neuron alignment names an undecomposed site {name!r}"
        spec = by_name[name]
        assignment = np.asarray(site.component_of_neuron)
        n = spec.d_out if site.neuron_axis == "d_out" else spec.d_in
        assert assignment.shape == (n,) and assignment.dtype.kind == "i", (
            name,
            assignment.shape,
            n,
        )
        assert assignment.min() >= -1 and assignment.max() < spec.C, (
            name,
            assignment.min(),
            assignment.max(),
            spec.C,
        )
        missing = set(range(spec.C)) - set(assignment.tolist())
        assert not missing, f"{name}: subcomponents {sorted(missing)[:8]}… have no neuron"


def _neuron_aligned_site_vu(W: Array, site: SiteNeuronAlignment, C: int) -> tuple[Array, Array]:
    """One site's neuron-aligned V/U from its frozen `W [d_out, d_in]` and its assignment
    (SPEC T13). With `E [C, n]` the assignment matrix (`E[i, s] = 1` iff neuron `s` is
    assigned to subcomponent `i`):

        writer (neuron axis d_out):  V = (E @ W)ᵀ  [d_in, C],   U = E          [C, d_out]
        reader (neuron axis d_in):   V = Eᵀ        [d_in, C],   U = E @ Wᵀ     [C, d_out]

    Exact start (one neuron per slot): `(V@U)ᵀ` is `W` restricted to the assigned rows
    (writer) / columns (reader) and the delta carries the other `n − C` neurons exactly;
    per subcomponent `‖v_i‖·‖u_i‖` is the neuron's own weight norm, `x @ V` its
    pre-activation (writer) or post-nonlinearity activation (reader). Wrapped start (k
    neurons per slot): slot `i` reads the SUM of its neurons' input weights and writes the
    same value to all of them (writer) / reads all of them and writes the sum of their
    output weights (reader) — every neuron is touched from step 0, at ~k× a neuron's norm,
    and the delta carries `W − (V@U)ᵀ`. Nothing is frozen and nothing is perturbed; both
    factors have a dense nonzero gradient from step 0."""
    d_out, d_in = W.shape
    assignment = site.component_of_neuron
    match site.neuron_axis:
        case "d_out":
            assert assignment.shape == (d_out,), (assignment.shape, d_out)
            selection = jax.nn.one_hot(
                assignment, C, dtype=W.dtype
            ).T  # [C, d_out]; -1 -> zero column
            return (selection @ W).T, selection
        case "d_in":
            assert assignment.shape == (d_in,), (assignment.shape, d_in)
            selection = jax.nn.one_hot(assignment, C, dtype=W.dtype).T  # [C, d_in]
            return selection.T, selection @ W.T


def init_component_stacks_neuron_aligned(
    sites: tuple[SiteSpec, ...],
    target_weights: dict[str, Array],
    alignment: NeuronAlignment,
    key: Array,
) -> ComponentStacks:
    """The `neuron_aligned_targeted` init (SPEC T13): every site in `alignment` takes
    `_neuron_aligned_site_vu`; every other site takes the `zero_u` values.

    Key discipline: the per-site keys are split exactly as `init_component_stacks_coupled`
    splits them and indexed by site position, and an aligned site simply does not consume
    its key — so every non-aligned site is bit-identical to a `zero_u` run at the same
    seed (pinned by `test_weight_init`)."""
    assert set(alignment) <= {spec.name for spec in sites}, set(alignment) - {
        spec.name for spec in sites
    }
    keys = jax.random.split(key, len(sites))
    vu: dict[str, tuple[Array, Array]] = {}
    for idx, spec in enumerate(sites):
        W = target_weights[spec.name].astype(jnp.float32)
        if spec.name in alignment:
            vu[spec.name] = _neuron_aligned_site_vu(W, alignment[spec.name], spec.C)
        else:
            V, U = _coupled_site_vu(W, keys[idx], spec.C)
            vu[spec.name] = (V, jnp.zeros_like(U))
    return component_stacks_from_site_arrays(sites, vu)
