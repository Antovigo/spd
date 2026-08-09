"""`DecomposedModel` — the interface a vendored target implements for the generic trainer.

The trainer (`train.py`) is abstract over the target model: it sees an ordered set of
decomposed **sites** (SPEC §1.2) and a handful of methods on the model `eqx.Module`. The
model carries its FROZEN target weights as fields; the TRAINABLE V/U (`vu`) is passed to
the forward methods explicitly (separate lifecycle). Everything at the boundary is keyed
by site name (flat dicts, torch-module-path style); how a target lays its parameters out
internally (e.g. the Llama target's stacked layer axis) is its own business.

The activation WAIST comes in exactly TWO shapes: positionless `[B, d]` (masks/CI
`[B, C]` — the toys) or with one position axis `[B, P, d]` (masks/CI `[B, P, C]` — an
LM, whose position axis is the token sequence). `has_position_axis` declares which;
`Positionless` / `Positioned` carry the run-scoped extents. Those are the waist's shapes;
a mask's leading axes match only in RANK, and are size 1 wherever the adversary's
`source_shape` says so (`SiteMasks`). Batch is ever-present and
semantics-free (the data/shard axis); CI is always independent over every leading axis.
Masking, routing, source scopes, imp-min, and normalization all operate over the opaque
leading prefix. The three EDGES are generic too — the model INPUT consumed by
`clean_forward` / `masked_forward` (tokens for an LM, a dict for a bio target), the model
OUTPUT carried by `ForwardResult.output` (`Any` — logits, a tuple of heads, coords), and
the recon comparison (`recon_loss_fn`, `kl_per_position` for an LM). Activation identity
and capture lowering are target-owned. Core passes immutable canonical names into the
forward and receives a strict one-key-to-one-array capture dictionary back.

The frozen weights ride on the model `eqx.Module` and reach the jitted step as a pytree
ARG (`eqx.filter_jit` traces the array leaves). Never close over the model in a jit: a
frozen 8B target captured as a constant bakes multi-GB weights into the HLO.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol, runtime_checkable

import jax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Bool, Float
from typing_extensions import TypeVar

from param_decomp.core.components import ComponentStacks, SiteSpec
from param_decomp.core.precision import COMPUTE_DT, cast_floating

BATCH_AXES = ("replicate", "fsdp")
"""Mesh axes that jointly shard the always-leading batch dimension."""


@dataclass(frozen=True)
class Positionless:
    """Waist `[B, d]`; masks/CI `[B, C]`. The toys."""


@dataclass(frozen=True)
class Positioned:
    """Waist `[B, P, d]`; masks/CI `[B, P, C]`. An LM: the position axis is the token
    sequence, so `n_positions` is its training seq_len (run-scoped — from the data
    config, not the model)."""

    n_positions: int


PositionAxis = Positionless | Positioned
"""The run's waist geometry — exactly these two cases, matched exhaustively wherever
shapes are built. Must agree with the model's `has_position_axis`."""


SiteMasks = dict[str, Float[Array, "*leading C"]]
"""Per-site component masks. `*leading` always has the WAIST's RANK, but ANY leading axis
may arrive size 1: an adversarial mask is materialized from a source stored per
`source_shape` (`configs.SourceShape`), and every axis that spelling omits is a size-1
broadcast axis — on a positioned target `c` gives `[1, 1, C]`, `bc` `[B, 1, C]`, `sc`
`[1, P, C]`; positionless `c` gives `[1, C]`. Only the stochastic and constant sources
build their masks at the full waist shape (from the CI). A target must therefore BROADCAST the leading axes against its own
waist, never reshape them: a reshape survives every stochastic step and dies on the first
adversarial one, long after the run looks healthy."""

SiteDeltaMasks = dict[str, Float[Array, "*leading"]]
"""The weight-delta counterpart of `SiteMasks` (the source's trailing channel) — same
leading axes, same broadcast rule, no C axis."""

SiteRoutes = dict[str, Bool[Array, "*leading"]] | None
"""Per-site per-position routing; `None` routes every position to the decomposition
(SPEC §1.3). Positions routing False take the frozen `x @ W` path."""


@dataclass(frozen=True, kw_only=True)
class MaterializedMasking:
    """A masked forward driven by concrete per-site mask arrays.

    Adversarial masks use this arm after optimized sources are converted to arrays; their
    provenance does not change target execution. The component-mask keys define the live sites. ``weight_delta_masks=None`` means the
    frozen-weight correction is disabled; a mapping enables it for every live site.
    Routes, when present, must cover the same sites. These constraints make the previous
    contradictory ``zero delta masks + has_delta=False`` state unrepresentable.
    """

    component_masks: SiteMasks
    weight_delta_masks: SiteDeltaMasks | None = None
    routes: SiteRoutes = None

    def __post_init__(self) -> None:
        live_sites = set(self.component_masks)
        if self.weight_delta_masks is not None:
            assert set(self.weight_delta_masks) == live_sites, (
                self.weight_delta_masks.keys(),
                self.component_masks.keys(),
            )
        if self.routes is not None:
            assert set(self.routes) == live_sites, (self.routes.keys(), self.component_masks.keys())

    @property
    def live_sites(self) -> tuple[str, ...]:
        return tuple(self.component_masks)


@dataclass(frozen=True, kw_only=True)
class StochasticMasking:
    """A recipe for sampling component and weight-delta masks inside checkpointed blocks.

    Scan targets consume the shared CI activations and key directly, discard each layer's
    masks after its forward block, and deterministically redraw them during backward
    recomputation instead of storing a full layer-by-layer mask stack.
    """

    ci_stacked: Any
    draw_key: Array
    live_sites: tuple[str, ...]
    routes: SiteRoutes

    def __post_init__(self) -> None:
        assert self.live_sites, "stochastic masking requires at least one live site"
        assert len(set(self.live_sites)) == len(self.live_sites), self.live_sites
        if self.routes is not None:
            assert set(self.routes) == set(self.live_sites), (self.routes.keys(), self.live_sites)


Masking = MaterializedMasking | StochasticMasking
"""The two complete, non-contradictory descriptions of a masked forward."""


type CaptureKeys = frozenset[str]
"""An orderless, immutable request for named activations from a forward."""

EMPTY_CAPTURE_KEYS: CaptureKeys = frozenset()


def select_captures(captures: dict[str, Array], capture_keys: CaptureKeys) -> dict[str, Array]:
    """Project a capture result onto one deterministic requested view."""
    return {key: captures[key] for key in sorted(capture_keys)}


PreparedT = TypeVar("PreparedT", default=Any)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("output", "captures"),
    meta_fields=(),
)
@dataclass(frozen=True)
class ForwardResult:
    """A target output and its captured activations, keyed one-to-one."""

    output: Any
    captures: dict[str, Array]

    @classmethod
    def from_producer(
        cls,
        *,
        output: Any,
        capture_keys: tuple[str, ...],
        capture_values: tuple[Array, ...],
    ) -> "ForwardResult":
        """Label a target's private capture slots and pin their shared device layout.

        A target resolves public activation names into a private slot layout while tracing,
        then produces arrays in that layout's order. This constructor is the single boundary
        that checks the canonical names and produced arrays agree, labels the arrays, and
        fixes their device layout before any consumer uses them.

        Captures always lead with the batch axis. Without the constraint below, GSPMD may
        independently feature-shard captures in different compiled consumers; cuDNN
        attention then rejects derived Q/K/V tensors whose layouts disagree. The runtime
        installs the HSDP mesh before tracing, while off-mesh targets and empty plans need
        no constraint.

        This must be an explicit producer constructor rather than ``__post_init__``. JAX
        pytree transformations reconstruct this registered dataclass with abstract or
        non-array leaves; reconstruction must not relabel values or apply device placement
        as a side effect.
        """
        assert len(capture_values) == len(capture_keys), (
            len(capture_values),
            capture_keys,
        )
        captures = dict(zip(capture_keys, capture_values, strict=True))
        if captures and not jax.sharding.get_abstract_mesh().empty:
            captures = {
                key: jax.lax.with_sharding_constraint(
                    value,
                    P(BATCH_AXES, *((None,) * (value.ndim - 1))),
                )
                for key, value in captures.items()
            }
        return cls(output, captures)


@runtime_checkable
class DecomposedModel(Protocol[PreparedT]):
    """The target interface consumed by the generic trainer.

    Core passes an immutable set of canonical activation names into each forward. The target
    validates, orders, and lowers those names into its private capture layout when JAX first
    traces that forward; no plan representation crosses this protocol. An empty set must take
    the target's untouched no-capture computation.
    """

    sites: tuple[SiteSpec, ...]
    has_position_axis: bool

    @property
    def site_names(self) -> tuple[str, ...]: ...

    def shardings(self, mesh: Mesh) -> "DecomposedModel[PreparedT]": ...

    def recon_loss_fn(self, masked_output: Any, clean_output: Any) -> Float[Array, ""]: ...

    def site_output_keys(self, sites: tuple[str, ...]) -> tuple[str, ...]:
        """Return each site's canonical linear-output key in request order."""
        ...

    def assert_hidden_acts_reconstruction_points(self, keys: tuple[str, ...]) -> None:
        """Refuse capture points that masking can never change."""
        ...

    def clean_forward(
        self, inputs: Any, /, capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS
    ) -> ForwardResult:
        """All-frozen forward plus exactly `capture_keys`. The same key has the same meaning
        here and in `masked_forward`."""
        ...

    def prepare_compute_weights(self, vu: ComponentStacks) -> PreparedT:
        """Relayout compute-dtype components into the target-private per-step view."""
        ...

    def component_activation_forward(
        self,
        prepared_weights: PreparedT,
        inputs: Any,
        /,
        *,
        capture_keys: CaptureKeys,
    ) -> tuple[ForwardResult, dict[str, Array]]:
        """Run the frozen target once, returning requested captures and each site's ``x @ V``.

        Targets that do not support offline component-activation harvest must raise
        ``NotImplementedError`` explicitly.
        """
        ...

    def stack_ci(self, ci_lower: dict[str, Array]) -> Any:
        """Build the target-private CI form shared by stochastic masked forwards."""
        ...

    def masked_forward(
        self,
        prepared_weights: PreparedT,
        inputs: Any,
        /,
        *,
        masking: Masking,
        capture_keys: CaptureKeys = EMPTY_CAPTURE_KEYS,
        remat: bool,
    ) -> ForwardResult:
        """Masked decomposed forward plus exactly `capture_keys`.

        `masking` carries the complete masking policy: explicit masks, or shared CI plus
        a draw key for rebuilding stochastic masks inside checkpointed blocks. The target
        validates unsupported capture keys fail-closed when this method is first traced.
        """
        ...

    def weight_deltas(self, vu: ComponentStacks) -> dict[str, Float[Array, "d_out d_in"]]: ...


def prepare_compute_weights[PreparedT](
    model: DecomposedModel[PreparedT], components: ComponentStacks
) -> PreparedT:
    """Cast fp32 master components once, then build the target-private compute layout."""
    return model.prepare_compute_weights(cast_floating(components, COMPUTE_DT))


def chunk_sites(site_names: tuple[str, ...], sites_per_chunk: int) -> tuple[tuple[str, ...], ...]:
    """Sequential `sites_per_chunk`-groups in the canonical site order (SPEC S10)."""
    assert len(site_names) % sites_per_chunk == 0, (
        f"{len(site_names)} sites not divisible by sites_per_chunk={sites_per_chunk}"
    )
    return tuple(
        tuple(site_names[i : i + sites_per_chunk])
        for i in range(0, len(site_names), sites_per_chunk)
    )
