"""CI values, named input taps, and the common callable interface."""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Literal, Protocol, Self, runtime_checkable

import jax
from jax.sharding import Mesh, NamedSharding
from jaxtyping import Array

from param_decomp.core.ci_fn.squashing import (
    DEFAULT_UPPER_LEAK,
    lower_leaky_hard_sigmoid,
    upper_leaky_hard_sigmoid,
)
from param_decomp.core.components import SiteCI, map_site_ci
from param_decomp.core.model import CaptureKeys, ComponentActivations
from param_decomp.core.muon_stacked import StageInPlace
from param_decomp.core.pytree import ShardingTree
from param_decomp.sequence import SequenceLayout

SiteDict = dict[str, SiteCI]
"""Per-output-site CI value keyed by OUTPUT site name: full `[*leading, C]` arrays for
dense sites, `SelectedCI` bundles for block-factored, selected-emitting sites
(`components.SiteCI` / `components.SelectedCI`)."""


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
    def from_preactivations(
        preactivations: "Mapping[str, SiteCI]", upper_leak: float = DEFAULT_UPPER_LEAK
    ) -> "CI":
        upper = partial(upper_leaky_hard_sigmoid, alpha=upper_leak)
        return CI(
            preactivations=dict(preactivations),
            lower={k: map_site_ci(lower_leaky_hard_sigmoid, v) for k, v in preactivations.items()},
            upper={k: map_site_ci(upper, v) for k, v in preactivations.items()},
        )


CIRole = Literal["output", "hidden"]
"""Which reconstruction objective a CI value scores for. `output` is the model-output
reconstruction every VPD run has always had; `hidden` scores the SAME pool of subcomponents
for reconstructing named internal activations instead. A single-role run has only `output`
and never mentions the vocabulary."""

DUAL_CI_ROLES: tuple[CIRole, ...] = ("output", "hidden")
"""Role order — the head order in a dual CI fn and the pass order in a dual objective."""

SINGLE_CI_ROLES: tuple[CIRole, ...] = ("output",)


def roles_for(dual: bool) -> tuple[CIRole, ...]:
    """The readout vocabulary a `dual` flag implies — the ONE place that correspondence
    lives, so the arches, the inits and the eval fan-out cannot drift apart on it."""
    return DUAL_CI_ROLES if dual else SINGLE_CI_ROLES


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DualCI:
    """Two `CI` bundles read off ONE trunk: `output` feeds the output-reconstruction
    objective, `hidden` the hidden-acts one. Both score the same subcomponent pool over the
    same inputs — only the readout head differs, so a component can be important for one
    objective and not the other.

    A pytree whose leaves are the two bundles', so `eqx.filter_vjp` over the CI fn takes a
    `DualCI` cotangent and the trunk's parameter gradient is the SUM of the two heads'
    contributions in ONE pullback — the trunk is never forward- or backward-run twice."""

    output: CI
    hidden: CI


AnyCI = CI | DualCI
"""What a CI fn returns: single-role runs keep the bare `CI` they always had."""


def bundle_ci(
    roles: tuple[CIRole, ...],
    per_role: tuple["Mapping[str, SiteCI]", ...],
    upper_leak: float = DEFAULT_UPPER_LEAK,
) -> AnyCI:
    """Squash each role's preactivations into its bundle, returning the shape `roles` implies —
    a bare `CI` for a single-role fn, so nothing downstream of a plain run ever sees `DualCI`."""
    assert len(roles) == len(per_role), (roles, len(per_role))
    match roles:
        case ("output",):
            return CI.from_preactivations(per_role[0], upper_leak)
        case ("output", "hidden"):
            return DualCI(
                output=CI.from_preactivations(per_role[0], upper_leak),
                hidden=CI.from_preactivations(per_role[1], upper_leak),
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


def output_ci(ci: AnyCI) -> CI:
    """The output-role bundle of whatever a CI fn returned. The spelling exists so a
    single-role consumer SAYS which role it means instead of reaching through a union — a
    dual fn's `.lower` is otherwise an easy silent mistake."""
    return ci_for_role(ci, "output")


class CIFnMetadata(Protocol):
    """The named physical captures a CI fn reads and the decomposition sites it scores."""

    @property
    def capture_keys(self) -> CaptureKeys: ...

    @property
    def output_names(self) -> tuple[str, ...]: ...

    @property
    def has_position_axis(self) -> bool: ...

    @property
    def roles(self) -> tuple[CIRole, ...]:
        """The readout vocabulary: `("output",)` for a single-head fn — which returns a bare
        `CI` — and `DUAL_CI_ROLES` for a shared-trunk dual fn, which returns `DualCI`. The
        role count is STATIC (it selects the return type), so a step traced against a dual fn
        never branches on it."""
        ...


def is_dual(ci_fn: CIFnMetadata) -> bool:
    """Whether this CI fn reads two roles off one trunk — a STATIC property of the built fn."""
    return len(ci_fn.roles) > 1


@dataclass(frozen=True)
class CIFnMatrix:
    """An independently updated matrix leaf, preceded by any independent matrix axes, and
    where stacked Muon stages it: a sharding that splits the independent matrices over its
    leading entry with each matrix whole per device, or in place.

    The value refers to the existing parameter leaf; this description owns no state.
    Biases and normalization scales are not matrices, regardless of their array rank.
    """

    value: Array
    staging: NamedSharding | StageInPlace


@runtime_checkable
class CIFn[Conditioning](CIFnMetadata, Protocol):
    """CI parameters, bound at construction to the placement they rest at, and CI
    evaluation over them.

    Parameters remain runtime pytree arguments; placement is static implementation
    state. `prepare` casts the parameters to the compute dtype and relayouts them into
    their compute layout; the train step evaluates only prepared parameters and pulls
    the CI gradient back through the preparation.

    Evaluation casts its taps to the compute dtype, runs the implementation's forward,
    and squashes the preactivations into `CI`. It also receives the target's prepared
    components, which a CI conditioned on component activations reads and every other
    CI ignores. Physical capture requests are exposed by ``capture_keys``; input
    preparation validates the physical capture request before invoking this operation.
    Arrays remain ordinary pytree leaves: autodiff also uses this structure for
    cotangents, whose dtypes need not equal the forward compute dtype.
    """

    def matrix_parameters(self) -> tuple[CIFnMatrix, ...]:
        """Enumerate independent matrices; other parameter leaves are elementwise."""
        ...

    def shardings(self, mesh: Mesh) -> ShardingTree: ...

    def placement_for_log(self) -> str:
        """The resolved placement as the startup audit prints it."""
        ...

    def prepare(self) -> Self: ...

    def __call__(
        self,
        taps: dict[str, Array],
        conditioning: Conditioning,
        components: ComponentActivations,
        *,
        sequence: SequenceLayout | None,
        remat: bool,
    ) -> AnyCI: ...


@dataclass(frozen=True)
class TapSpec:
    """One input tap: its capture key and feature width. The key is opaque to core (the
    lab authors it, the target resolves it); the width rides alongside so the consumer
    can size and assert its input without deriving it from a site."""

    key: str
    width: int
