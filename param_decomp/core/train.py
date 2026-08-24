"""The generic single-pool VPD training step over a `DecomposedModel` (SPEC §4).

One `jax.jit` step: clean target forward → CI envelope → per-persistent-term supplemental
ascents + per-fresh-term sign-PGD ascents (`adversary.py`) → faith + imp-min +
the recon loss TERMS (`recon.py`; each term = plan × mask-source strategy, SPEC
S10') + optional nonlinearity-locality term (S36) → one fused backward over
(components, ci_fn, all persistent sources) → optimizer updates → each persistent
term's final ascent. The default `e2e` adversary retakes only its output-reconstruction
source gradient when the outer term also includes hidden-activation reconstruction; an
explicit `term` adversary reuses the fused graph and ascends the complete term (SPEC
S13'/S14'/S23). All trainable state is fp32 masters (SPEC N1); forwards run in bf16
via explicit casts.

Schedules (imp-min p anneal, source-LR warmup, every scheduled loss coefficient) are
computed inside the step from `state.step`, so the jit signature is stable across the
whole run (SPEC S9, S13); each coefficient resolves ONCE at the top of the step and only
values flow into the loss math.
Per-term RNG: term i draws from `fold_in(step_key, offset + i)` in config-list order
(SPEC R1) — offset 1 for the main grid reproduces the pre-unification production key
derivation exactly.

The factories bind explicit ``ForwardSubstrate`` and ``ReconGrid`` values while keeping
the plain and targeted step bodies separate and readable.
"""

import functools
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from beartype import beartype
from jax import random
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from param_decomp.core.adversary import PersistentAdversary, Sources, init_fresh_pgd_sources
from param_decomp.core.ci_fn import (
    CI,
    AnyCI,
    CIFn,
    CIRole,
    DualCI,
    PlacedCIFn,
    ci_for_role,
    evaluate_compute_ci,
    is_dual,
    materialize_ci_compute_weights,
    output_ci,
)
from param_decomp.core.components import ComponentStacks
from param_decomp.core.configs import LossCoeff
from param_decomp.core.decomposed_linear import constrain_component_activation
from param_decomp.core.faithfulness import FaithfulnessLossFn
from param_decomp.core.jit_util import filter_jit
from param_decomp.core.losses import (
    BatchFrequency,
    BatchFrequencyTerm,
    EmaFrequency,
    EmaFrequencyTerm,
    FrequencyTerm,
    ReconstructionLoss,
    annealed_imp_min_param,
    coeff_at,
    imp_min_terms,
    lp_term,
    mean_reconstruction_losses,
    per_component_frequencies,
    reconstruction_loss,
    reconstruction_loss_metrics,
    reconstruction_spec_at,
    resolve_frequency,
    scheduled_value_at,
    train_frac_at,
)
from param_decomp.core.masking import (
    constant_delta_pinned_masks,
    masks_from_sources,
    mixed_persistent_stochastic_delta_pinned_masks,
    mixed_persistent_stochastic_masks,
    persistent_delta_pinned_masks,
    stochastic_delta_pinned_masks,
    unmasked_no_delta_masks,
)
from param_decomp.core.model import (
    CaptureKeys,
    Masking,
    MaterializedMasking,
    PlacedModel,
    StochasticMasking,
    faithfulness_weight_deltas,
    prepare_compute_weights,
    select_captures,
)
from param_decomp.core.objective import (
    ImportanceMinimalityTerm,
    LossSurface,
    ResolvedNonlinearity,
    TargetedObjective,
)
from param_decomp.core.placement import CIFnPlacement, PlacementRules
from param_decomp.core.recon import (
    AnyReconLossTerm,
    ConstantSources,
    ForwardObservations,
    FreshPGDSources,
    HiddenActsOnlyReconstruction,
    MaskSourceStrategy,
    MixedPersistentStochasticSources,
    OutputOnlyReconstruction,
    PersistentSources,
    ReconLossTerm,
    ReconstructionSpec,
    Routes,
    StochasticSources,
    UnmaskedNoDeltaSources,
    hidden_acts_capture_keys,
    index_persistent_terms,
    reconstruction_observations,
)
from param_decomp.core.schedule import ScheduleConfig
from param_decomp.core.sharding import batch_shard_leading


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Decomposition:
    """The trained PRODUCT: V/U components + the CI fn (fp32 masters). Checkpointed as
    its own orbax item so consumers such as clustering restore it with
    zero knowledge of the training process (optimizer states, adversaries, step)."""

    components: ComponentStacks  # the universal trainable V/U pytree, fp32 masters
    ci_fn: CIFn  # fp32 masters


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrainingItem:
    """The trainer-only trajectory tail: both optimizer states, the persistent adversaries,
    the frequency-EMA buffers, the step counter. Checkpointed as its own orbax item — no
    consumer restores it."""

    components_opt_state: optax.OptState
    ci_fn_opt_state: optax.OptState
    adversaries: dict[str, PersistentAdversary]
    """Persistent-PGD adversaries, `state_key -> adversary` (each owns its sources + Adam
    state + static config). One state_key per persistent loss term (SPEC S23); empty when
    no persistent term."""
    freq_ema: dict[str, Array] | None
    """Per-site `(C,)` fp32 EMA of the per-component firing frequencies `f_c`, feeding the
    smoothed frequency penalty (SPEC S8''); present iff the run's resolved frequency mode
    is `EmaFrequency`, so configs without the EMA keep their checkpoint tree byte-identical."""
    step: Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrainState:
    """The full training pytree, composed of the two checkpoint items so there is ONE
    representation: `decomposition` is the trained product, `training` the trajectory tail.
    Save/restore maps directly onto these two fields — no regrouping."""

    decomposition: Decomposition
    training: TrainingItem


def _grad_norm_metrics(
    components_grad: ComponentStacks, ci_fn_grad: Any, mesh: Mesh | None
) -> dict[str, Array]:
    """Pre-clip gradient L2 norms, matching the torch `component_grad_norms` families.

    Components norms are per SITE per factor — `grad_norms/components.vu['<site>'][0|1]`
    (0=V, 1=U), e.g. `grad_norms/components.vu['layers.18.mlp.gate_proj'][0]`. Sites are
    the semantic unit; the grouped stacks they're stored in are not. The key spells
    the retired per-site pytree path so wandb histories overlay across the stacking
    refactor. Ci-fn norms are per LEAF of whatever pytree the CI fn is
    (`grad_norms/ci_fns<path>`), plus the overlay-critical
    `grad_norms/summary/{components,ci_fns,total}`."""
    out: dict[str, Array] = {}

    def per_slice_sq(stack: Float[Array, "g a b"]) -> Float[Array, " g"]:
        sq = jnp.sum(stack.astype(jnp.float32) ** 2, axis=(1, 2))
        # Replicate the [g] vector ONCE; the per-site scalar reads below are then local
        # slices instead of one tiny cross-mesh broadcast per site per factor (2·n_sites
        # collectives per step under a stack-sharded persist layout).
        if mesh is not None:
            sq = jax.sharding.reshard(sq, NamedSharding(mesh, P()))
        return sq

    factor_sq = {
        shape: (per_slice_sq(Vs), per_slice_sq(Us))
        for shape, (Vs, Us) in components_grad.stacks.items()
    }
    for name, shape, slot in components_grad.site_slots:
        v_sq, u_sq = factor_sq[shape]
        out[f"grad_norms/components.vu['{name}'][0]"] = jnp.sqrt(v_sq[slot])
        out[f"grad_norms/components.vu['{name}'][1]"] = jnp.sqrt(u_sq[slot])
    components_sq = jnp.zeros((), jnp.float32)
    for v_sq, u_sq in factor_sq.values():
        components_sq = components_sq + jnp.sum(v_sq) + jnp.sum(u_sq)
    out["grad_norms/summary/components"] = jnp.sqrt(components_sq)

    ci_fn_sq = jnp.zeros((), jnp.float32)
    for path, leaf in jax.tree_util.tree_flatten_with_path(ci_fn_grad)[0]:
        leaf_sq = jnp.sum(leaf.astype(jnp.float32) ** 2)
        out[f"grad_norms/ci_fns{jax.tree_util.keystr(path)}"] = jnp.sqrt(leaf_sq)
        ci_fn_sq = ci_fn_sq + leaf_sq
    out["grad_norms/summary/ci_fns"] = jnp.sqrt(ci_fn_sq)

    out["grad_norms/summary/total"] = jnp.sqrt(components_sq + ci_fn_sq)
    return out


@eqx.filter_jit
def uv_norm_ratio_metrics(components: ComponentStacks) -> dict[str, Array]:
    """Return each site's Frobenius-norm ratio ``||U|| / ||V||`` and summaries."""

    def per_slice_sq(stack: Float[Array, "g a b"]) -> Float[Array, " g"]:
        sq = jnp.sum(stack.astype(jnp.float32) ** 2, axis=(1, 2))
        # Replicate the tiny [g] vector ONCE before the per-site reads: under a
        # stack-owned persist layout (`sharding: owner`) the stack axis is sharded, and
        # a static per-slot slice of a sharded dim is unimplemented. Ambient-mesh guard
        # (the `site_forward` pattern): a no-op off-mesh (toys / CPU tests).
        if not jax.sharding.get_abstract_mesh().empty:
            sq = jax.sharding.reshard(sq, P())
        return sq

    factor_sq = {
        shape: (per_slice_sq(Vs), per_slice_sq(Us)) for shape, (Vs, Us) in components.stacks.items()
    }
    metrics: dict[str, Array] = {}
    ratios = []
    for name, shape, slot in components.site_slots:
        v_sq, u_sq = factor_sq[shape]
        ratio = jnp.sqrt(u_sq[slot] / v_sq[slot])
        metrics[f"uv_norm_ratio['{name}']"] = ratio
        ratios.append(ratio)

    stacked = jnp.stack(ratios)
    metrics["uv_norm_ratio_mean"] = jnp.mean(stacked)
    metrics["uv_norm_ratio_max"] = jnp.max(stacked)
    return metrics


def _scheduled_coeff_metrics(train_frac: Array, coeffs: dict[str, LossCoeff]) -> dict[str, Array]:
    """Only scheduled coefficients: constants would add log noise."""
    return {
        f"schedules/coeff/{name}": scheduled_value_at(train_frac, coeff)
        for name, coeff in coeffs.items()
        if isinstance(coeff, ScheduleConfig)
    }


@jax.custom_vjp
def _cotangent_scaled(x: Array, by: Array) -> Array:
    del by  # forward-inert: consumed only by the vjp
    return x


def _cotangent_scaled_fwd(x: Array, by: Array) -> tuple[Array, Array]:
    return x, by


def _cotangent_scaled_bwd(by: Array, g: Array) -> tuple[Array, Array]:
    # An UNREDUCED cotangent (the chained-reduced weights') may only multiply a scalar
    # typed `reduced` over the same axes: (Σᵢ aᵢ)·c = Σᵢ(aᵢ·c), a pure retag.
    scale = by.astype(g.dtype)
    unreduced = frozenset(jax.typeof(g).sharding.spec.unreduced)
    if unreduced:
        scale = jax.sharding.reshard(scale, P(reduced=unreduced))
    return g * scale, jnp.zeros_like(by)


_cotangent_scaled.defvjp(_cotangent_scaled_fwd, _cotangent_scaled_bwd)


def model_cotangents_scaled[T](tree: T, by: Array | float) -> T:
    """`tree`, bit-identical in the forward, with every backward cotangent scaled `by`
    the term's per-step coeff. This is WHERE a persistent term's coeff applies (SPEC
    S14'): the term's loss enters the differentiated total UNSCALED so the source path
    carries `dL/ds` directly, and the model-side inputs (prepared weights, CI envelope)
    are wrapped here so the components/CI fn still receive `coeff·dL/dθ` — an exact 0
    while an activation gate holds the coeff at 0, with no division anywhere."""
    by_arr = jnp.asarray(by, jnp.float32)
    return jax.tree.map(lambda leaf: _cotangent_scaled(leaf, by_arr), tree)


type CoeffApplication = Literal["scales_loss", "scales_model_cotangents"]


def coeff_application(term: AnyReconLossTerm) -> CoeffApplication:
    """WHERE this term's coeff applies — static structure, decided at trace time.

    A term that trains its sources FROM the shared backward (a persistent bundle as its
    sources) must keep the source path unscaled so the backward hands the adversary `dL/ds`
    (SPEC S14'): its coeff rides the model-side cotangents (`model_cotangents_scaled`)
    and the term enters the differentiated total at weight 1. Every other term's coeff
    scales its loss scalar in the total."""
    trains_sources_from_backward = isinstance(
        term.sources, PersistentSources | MixedPersistentStochasticSources
    )
    return "scales_model_cotangents" if trains_sources_from_backward else "scales_loss"


# ───────────────────────────── the step vocabulary ─────────────────────────────


@dataclass(frozen=True)
class StreamInputs:
    """One data stream's per-step trace inputs: the sharded opaque batch, the detached
    clean-forward observations it is scored against, the CI-fn input activations, and
    the waist `leading` shape masks/sources/routes live in."""

    batch: Any
    clean: ForwardObservations
    taps: dict[str, Array]
    leading: tuple[int, ...]


@dataclass(frozen=True)
class AscendedAdversaries:
    """The ascent phase's outputs: warmed persistent adversaries (SPEC S24), each
    fresh-PGD term's ascended sources, and the per-term routing draws those ascents
    fixed for the main grid to reuse (SPEC S24, torch parity)."""

    warmed: dict[str, PersistentAdversary]
    fresh_sources: dict[int, Sources]
    fixed_routes: dict[int, tuple[Routes, ...]]


type DrawLoss[S: MaskSourceStrategy] = Callable[
    [int, ReconLossTerm[S], PRNGKeyArray, Routes], ReconstructionLoss
]
"""`(term_idx, term, draw_key, routes) -> the draw's scored recon` — one grid's
per-draw dispatcher, built by the factory that owns the grid's trainables. `S` is the
grid's source-strategy width: the non-target grid's dispatcher takes only the enumerated
non-target strategies, so its match is exhaustive over those arms (SPEC T5)."""

type TermDraws = list[tuple[PRNGKeyArray, Routes]]
"""One term's flat `(draw_key, routes)` forwards."""


def constant_source_masks(
    strategy: ConstantSources, ci_lower: dict[str, Array]
) -> dict[str, Array]:
    """Build constant component masks; no weight-delta path exists for this source."""
    return {site: ci + (1.0 - ci) * strategy.value for site, ci in ci_lower.items()}


@dataclass(frozen=True)
class ReconGrid[S: MaskSourceStrategy]:
    """One reconstruction grid and its first reserved per-term RNG index (SPEC R1)."""

    terms: tuple[ReconLossTerm[S], ...]
    key_offset: int

    def __post_init__(self) -> None:
        assert self.terms, "a reconstruction grid must be non-empty"
        assert self.key_offset >= 1, self.key_offset
        assert len(self.capture_keys_by_term) == len(self.terms), (
            "duplicate reconstruction term names"
        )
        self._persistent_by_key()

    @classmethod
    def of(cls, terms: tuple[ReconLossTerm[S], ...], *, key_offset: int) -> "ReconGrid[S]":
        return cls(terms, key_offset)

    @property
    def capture_keys_by_term(self) -> dict[str, CaptureKeys]:
        return {term.name: term.hidden_acts_capture_keys for term in self.terms}

    def _persistent_by_key(self) -> dict[str, ReconLossTerm[S]]:
        persistent: dict[str, ReconLossTerm[S]] = {}
        for term in self.terms:
            match term.sources:
                case (
                    PersistentSources(state_key=state_key)
                    | MixedPersistentStochasticSources(state_key=state_key)
                ):
                    assert state_key not in persistent, (
                        f"persistent source {state_key!r} feeds multiple terms"
                    )
                    persistent[state_key] = term
                case (
                    StochasticSources()
                    | ConstantSources()
                    | UnmaskedNoDeltaSources()
                    | FreshPGDSources()
                ):
                    pass
        return persistent

    @property
    def persistent_by_key(self) -> dict[str, ReconLossTerm[S]]:
        return self._persistent_by_key()

    @property
    def capture_keys(self) -> CaptureKeys:
        return frozenset(
            key for term_keys in self.capture_keys_by_term.values() for key in term_keys
        )

    def reconstruction_specs_at(self, train_frac: Array) -> dict[str, ReconstructionSpec]:
        return {
            term.name: reconstruction_spec_at(term.hidden_acts_reconstruction, train_frac)
            for term in self.terms
        }

    def adversary_reconstruction_specs(
        self, reconstruction_specs: dict[str, ReconstructionSpec]
    ) -> dict[str, ReconstructionSpec]:
        """Choose each adversary's source-ascent objective independently of the outer loss."""
        return {
            term.name: (
                OutputOnlyReconstruction()
                if isinstance(term.sources, PersistentSources | MixedPersistentStochasticSources)
                and term.sources.cfg.adversary_objective == "e2e"
                else reconstruction_specs[term.name]
            )
            for term in self.terms
        }

    @property
    def e2e_terms_requiring_source_grad_retake_by_key(
        self,
    ) -> dict[str, ReconLossTerm[S]]:
        """Persistent e2e terms whose outer loss includes hidden-activation reconstruction."""
        return {
            state_key: term
            for state_key, term in self.persistent_by_key.items()
            if isinstance(term.sources, PersistentSources | MixedPersistentStochasticSources)
            and term.sources.cfg.adversary_objective == "e2e"
            and term.hidden_acts_reconstruction is not None
        }

    def coeffs_at(self, train_frac: Array) -> tuple[Float[Array, ""] | float, ...]:
        return tuple(coeff_at(train_frac, term.coeff) for term in self.terms)

    def draws(
        self,
        key: PRNGKeyArray,
        fixed_routes: dict[int, tuple[Routes, ...]],
        leading: tuple[int, ...],
    ) -> list[TermDraws]:
        """Materialize every term/draw key chain (SPEC R1)."""
        draws_per_term: list[TermDraws] = []
        for term_idx, term in enumerate(self.terms):
            draw_key, routing_key = random.split(random.fold_in(key, self.key_offset + term_idx))
            match term.sources:
                case FreshPGDSources():
                    routes_per_draw = fixed_routes[term_idx]
                case _:
                    routes_per_draw = term.sample_routing(routing_key, leading)
            assert routes_per_draw, f"term {term.name!r} produced no forwards"
            draws_per_term.append(
                [
                    (random.fold_in(draw_key, draw_idx), routes)
                    for draw_idx, routes in enumerate(routes_per_draw)
                ]
            )
        return draws_per_term

    def losses(
        self,
        draws_per_term: list[TermDraws],
        draw_loss: DrawLoss[S],
    ) -> tuple[ReconstructionLoss, ...]:
        """Mean reconstruction over each term's draws (SPEC S10')."""
        return tuple(
            mean_reconstruction_losses(
                tuple(draw_loss(term_idx, term, draw_key, routes) for draw_key, routes in draws)
            )
            for term_idx, (term, draws) in enumerate(zip(self.terms, draws_per_term, strict=True))
        )


@dataclass(frozen=True)
class ForwardSubstrate[PreparedT]:
    """Array-free run statics owning forward preparation and VJP scaffolding.

    The model is never stored: every method keeps it as a traced argument, preserving
    the HLO-baking rule. `placement_rules` is the model bundle's own rules, pulled off
    it at `of` — the CI/batch constraints below share the model's placement by
    construction.
    """

    remat_recon_forwards: bool
    remat_ci_fn: bool
    ascend_replicate: bool
    """Replicate the detached compute weights before the ascent phase. The ascents are
    detached and read-only, so replicating once beats re-gathering the FSDP shards inside
    every ascent step; costs one full copy of the compute weights for the phase's lifetime."""
    placement_rules: PlacementRules | None
    ci_placement: CIFnPlacement | None
    """The run's CI-fn placement, resolved at assembly (`resolve_ci_placement`) — never
    re-derived from `placement_rules` here. `placed` pairs it with the live masters."""
    ci_capture_keys: CaptureKeys
    recon_loss_fn: Callable[[Any, Any], Array]

    @property
    def mesh(self) -> Mesh | None:
        """The rules' own mesh — a substrate never carries a second copy to desync. A
        substrate that executes forwards needs a concrete mesh, so the abstract
        (spec-check) arm of `PlacementRules.mesh` is refused here."""
        if self.placement_rules is None:
            return None
        mesh = self.placement_rules.mesh
        assert isinstance(mesh, Mesh), type(mesh)
        return mesh

    @classmethod
    def of(
        cls,
        model_static: PlacedModel[PreparedT],
        *,
        remat_recon_forwards: bool,
        remat_ci_fn: bool,
        ci_capture_keys: CaptureKeys,
        ci_placement: CIFnPlacement | None,
        ascend_replicate: bool = False,
    ) -> "ForwardSubstrate[PreparedT]":
        return cls(
            remat_recon_forwards=remat_recon_forwards,
            remat_ci_fn=remat_ci_fn,
            ascend_replicate=ascend_replicate,
            placement_rules=model_static.placement,
            ci_placement=ci_placement,
            ci_capture_keys=ci_capture_keys,
            recon_loss_fn=model_static.recon_loss_fn,
        )

    def shard_batch_tree[T](self, x: T) -> T:
        """Pin the leading (batch) axis of every array in the pytree. The batch and the
        model output are opaque protocol edges (`Any` — tokens for an LM, a dict or tuple
        for another target), so this maps over leaves rather than assuming one array."""
        return jax.tree.map(lambda leaf: batch_shard_leading(leaf, self.mesh), x)

    def _shard_ci_array(self, x: Array) -> Array:
        return constrain_component_activation(x, self.placement_rules)

    def shard_ci[T: AnyCI](self, ci: T) -> T:
        """Keep CI squashings aligned with `site_out`'s batch × component layout.

        A dual bundle pins BOTH heads: they are the same shape and feed the same mask
        multiply, one per objective (SPEC S37)."""

        def pin(bundle: CI) -> CI:
            return CI(
                preactivations={
                    site: self._shard_ci_array(v) for site, v in bundle.preactivations.items()
                },
                lower={site: self._shard_ci_array(v) for site, v in bundle.lower.items()},
                upper={site: self._shard_ci_array(v) for site, v in bundle.upper.items()},
            )

        match ci:
            case DualCI():
                return cast("T", DualCI(output=pin(ci.output), hidden=pin(ci.hidden)))
            case CI():
                return cast("T", pin(ci))

    def replicate_for_ascend(self, prepared_weights: PreparedT) -> PreparedT:
        """The ascent phase's copy of the detached compute weights (`ascend_replicate`).

        No-op when the knob is off or there is no mesh."""
        if not self.ascend_replicate or self.mesh is None:
            return prepared_weights
        replicated = NamedSharding(self.mesh, P())
        return jax.tree.map(
            lambda leaf: (
                jax.lax.with_sharding_constraint(leaf, replicated)
                if hasattr(leaf, "shape")
                else leaf
            ),
            prepared_weights,
        )

    def prep_stream(
        self, model: PlacedModel[PreparedT], batch: Any, hidden_acts_keys: CaptureKeys
    ) -> StreamInputs:
        """Shard one stream's batch, run its detached clean forward, and pull the CI taps +
        recon observations. `hidden_acts_keys` is the stream's own union — a stream whose
        grid carries no hidden-acts reconstruction captures none."""
        batch = self.shard_batch_tree(batch)
        with jax.named_scope("pd_clean_fwd_and_taps"):
            clean_forward_result = jax.tree.map(
                jax.lax.stop_gradient,
                model.clean_forward(batch, self.ci_capture_keys | hidden_acts_keys),
            )
            taps = select_captures(clean_forward_result.captures, self.ci_capture_keys)
            clean = reconstruction_observations(
                clean_forward_result,
                hidden_acts_capture_keys=hidden_acts_keys,
                mesh=self.mesh,
            )
        # `leading` (batch, *positions) — the shape masks/sources/routes live in. Sourced
        # from a tap (always `[*leading, d_tap]`), not the opaque batch, so the engine never
        # assumes the batch's rank/feature dim.
        leading = next(iter(taps.values())).shape[:-1]
        return StreamInputs(batch=batch, clean=clean, taps=taps, leading=leading)

    def component_weights_vjp(
        self, model: PlacedModel[PreparedT], components: ComponentStacks
    ) -> tuple[PreparedT, Callable[[PreparedT], tuple[ComponentStacks]]]:
        """The compute-weights value + vjp — the recon gradient's pullback onto V/U."""
        return jax.vjp(lambda c: prepare_compute_weights(model, c), components)

    def placed(self, ci_fn: CIFn) -> PlacedCIFn:
        """The live masters paired with the run's already-resolved placement."""
        return PlacedCIFn(fn=ci_fn, placement=self.ci_placement)

    def ci_weights_vjp(self, ci_fn: CIFn) -> tuple[PlacedCIFn, Callable[[PlacedCIFn], tuple[Any]]]:
        """The resident BF16 CI weights and their pullback onto the FP32 masters."""
        return eqx.filter_vjp(lambda cf: materialize_ci_compute_weights(self.placed(cf)), ci_fn)

    def ci_forward_vjp(
        self, compute_ci_fn: PlacedCIFn, taps: dict[str, Array]
    ) -> tuple[AnyCI, Callable[[AnyCI], tuple[Any]]]:
        """The CI envelope's value + vjp. The CI envelope is a pure fn of the taps, so it is
        forward-evaluated ONCE per stream — the ascents use the stop_gradient'd value; the
        loss takes the live value and its ci-fn grad is pulled back through the vjp."""
        with jax.named_scope("pd_ci_fn_fwd"):
            return eqx.filter_vjp(
                lambda cf: self.shard_ci(evaluate_compute_ci(cf, taps, remat=self.remat_ci_fn)),
                compute_ci_fn,
            )

    # ONE masked-forward remat policy for recon AND the adversary ascents.
    # `remat_recon_forwards` picks the checkpoint policy of the target's per-block scan:
    # True = `nothing_saveable` — the backward re-forwards one block at a time instead of
    # holding its activations (deep targets need this to fit); False = `dots_saveable` — the
    # backward reads stored batch-scaled activation dots and re-forwards nothing (faster when
    # memory allows; gathered weight operands are never residuals either way — the scanned
    # linear re-derives them in its transpose). This is load-bearing for the ASCENTS too:
    # though they backprop only to the SOURCES (params + CI detached), the source gradient
    # still flows through the per-layer activations (the masks MULTIPLY them), so an
    # un-rematted ascent forward stores `[n_layer, *leading, d_ff]`-scale intermediates.
    @jaxtyped(typechecker=beartype)
    def masked_recon(
        self,
        model: PlacedModel[PreparedT],
        *,
        prepared_weights: PreparedT,
        batch: Any,
        masking: Masking,
        capture_keys: CaptureKeys,
        reconstruction: ReconstructionSpec,
        clean: ForwardObservations,
    ) -> ReconstructionLoss:
        """Run one masked forward and score its complete recon objective — output and
        hidden-acts points alike."""
        masked_forward_result = model.masked_forward(
            prepared_weights,
            batch,
            masking=masking,
            capture_keys=capture_keys,
            remat=self.remat_recon_forwards,
        )
        masked = reconstruction_observations(
            masked_forward_result,
            hidden_acts_capture_keys=capture_keys,
            mesh=self.mesh,
        )
        return reconstruction_loss(
            self.recon_loss_fn,
            masked=masked,
            clean=clean,
            reconstruction=reconstruction,
        )


def ascend_adversaries[PreparedT](
    substrate: ForwardSubstrate[PreparedT],
    grid: ReconGrid[MaskSourceStrategy],
    model: PlacedModel[PreparedT],
    stream: StreamInputs,
    detached_prepared_weights: PreparedT,
    ci_lower_detached: dict[str, Array],
    adversaries: dict[str, PersistentAdversary],
    key: PRNGKeyArray,
    train_frac: Array,
    reconstruction_specs: dict[str, ReconstructionSpec],
    delta_pinned: bool = False,
) -> AscendedAdversaries:
    """Detached adversary ascents for one pass's grid.

    `grid` / `adversaries` scope the phase to ONE pass: a tPD run with a hidden pass (T12)
    ascends each pass's adversaries against ITS OWN CI head and ITS OWN objective, so the
    two never share a source bundle. The grid's `key_offset` (T9/R1) keeps fresh-PGD draws
    disjoint across passes.

    `delta_pinned` is the tPD NON-TARGET pass's polarity (T4; T5/T7 amended 2026-08-19):
    warmup scoring forwards compose the bundle's component channels with every weight-delta
    mask pinned to 1.0, and `stream` is then the BROAD stream. Fresh-PGD has no delta-pinned
    form — it stays unrepresentable off-target, asserted below."""
    if delta_pinned:
        assert not any(isinstance(t.sources, FreshPGDSources) for t in grid.terms), (
            "fresh-PGD is unrepresentable on the non-target pass (SPEC T5)"
        )

    def ascent_masks(
        sources: Sources,
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        if delta_pinned:
            return persistent_delta_pinned_masks(ci_lower_detached, sources)
        return masks_from_sources(ci_lower_detached, sources)

    def warmup_scoring_loss(term: AnyReconLossTerm) -> Callable[[Sources], Array]:
        def objective(sources: Sources) -> Array:
            masks, delta_masks = ascent_masks(sources)
            return substrate.masked_recon(
                model,
                prepared_weights=detached_prepared_weights,
                batch=stream.batch,
                masking=MaterializedMasking(
                    component_masks=masks, weight_delta_masks=delta_masks, routes=None
                ),
                capture_keys=hidden_acts_capture_keys(reconstruction_specs[term.name]),
                reconstruction=reconstruction_specs[term.name],
                clean=stream.clean,
            ).total

        return objective

    with jax.named_scope("pd_pgd_warmup_ascend"):
        warmed = {
            state_key: adv.warmup_ascend(
                warmup_scoring_loss(grid.persistent_by_key[state_key]), train_frac
            )
            for state_key, adv in adversaries.items()
        }

    fresh_sources: dict[int, Sources] = {}
    fixed_routes: dict[int, tuple[Routes, ...]] = {}
    for term_idx, term in enumerate(grid.terms):
        if not isinstance(term.sources, FreshPGDSources):
            continue
        fresh_cfg = term.sources
        routing_key, init_key = random.split(random.fold_in(key, grid.key_offset + term_idx))
        routes_per_draw = term.sample_routing(routing_key, stream.leading)
        fixed_routes[term_idx] = routes_per_draw
        init = init_fresh_pgd_sources(
            sites=model.sites,
            init=fresh_cfg.init,
            source_shape=fresh_cfg.source_shape,
            leading=stream.leading,
            key=init_key,
        )

        def ascent_loss(
            sources: Sources,
            term: AnyReconLossTerm = term,
            routes: tuple[Routes, ...] = routes_per_draw,
        ) -> Array:
            masks, delta_masks = ascent_masks(sources)
            total = jnp.zeros((), jnp.float32)
            for routes_for_draw in routes:
                breakdown = substrate.masked_recon(
                    model,
                    prepared_weights=detached_prepared_weights,
                    batch=stream.batch,
                    masking=MaterializedMasking(
                        component_masks=masks,
                        weight_delta_masks=delta_masks,
                        routes=routes_for_draw,
                    ),
                    capture_keys=hidden_acts_capture_keys(reconstruction_specs[term.name]),
                    reconstruction=reconstruction_specs[term.name],
                    clean=stream.clean,
                )
                total = total + breakdown.total
            return total / len(routes)

        def sign_ascend_body(
            sources: Sources,
            _: None,
            ascent_loss: Callable[[Sources], Array] = ascent_loss,
            step_size: float = fresh_cfg.step_size,
        ) -> tuple[Sources, None]:
            sources_grad = jax.grad(ascent_loss)(sources)
            return jax.tree.map(
                lambda source, gradient: jnp.clip(
                    source + step_size * jnp.sign(gradient), 0.0, 1.0
                ),
                sources,
                sources_grad,
            ), None

        with jax.named_scope("pd_fresh_pgd_ascend"):
            ascended, _ = jax.lax.scan(sign_ascend_body, init, None, length=fresh_cfg.n_steps)
        fresh_sources[term_idx] = jax.lax.stop_gradient(ascended)

    return AscendedAdversaries(
        warmed=warmed, fresh_sources=fresh_sources, fixed_routes=fixed_routes
    )


def main_draw_loss[PreparedT](
    substrate: ForwardSubstrate[PreparedT],
    model: PlacedModel[PreparedT],
    *,
    prepared_weights: PreparedT,
    ci: CI,
    ci_stacked: Any,
    persistent_sources: dict[str, Sources],
    ascended: AscendedAdversaries,
    stream: StreamInputs,
    train_frac: Array,
    reconstruction_specs: dict[str, ReconstructionSpec],
    term_coeffs: dict[str, Array | float],
) -> DrawLoss[MaskSourceStrategy]:
    """The main grid's per-draw dispatcher over the trainables: match the term's
    mask-source strategy, run the masked forward, score against the stream's clean
    observations. Built INSIDE the loss fn — it closes over the live trainables.

    Persistent(-carrying) draws take the coeff on their MODEL-SIDE inputs
    (`model_cotangents_scaled`) and enter the total at weight 1, so the fused
    backward hands each adversary `dL/ds` unscaled (SPEC S14')."""

    def draw_loss(
        term_idx: int,
        term: ReconLossTerm[MaskSourceStrategy],
        draw_key: PRNGKeyArray,
        routes: Routes,
    ) -> ReconstructionLoss:
        match coeff_application(term):
            case "scales_model_cotangents":
                draw_prepared = model_cotangents_scaled(prepared_weights, term_coeffs[term.name])
                draw_ci_lower = model_cotangents_scaled(ci.lower, term_coeffs[term.name])
            case "scales_loss":
                draw_prepared, draw_ci_lower = prepared_weights, ci.lower
        with jax.named_scope("pd_recon_masked_fwd"):
            match term.sources:
                case StochasticSources():
                    # Stochastic recon passes `StochasticMasking` to `masked_forward`, so a scan
                    # target rebuilds masks from shared `ci_stacked` inside each checkpointed
                    # block (the full mask stack is never held). Explicit strategies pass
                    # `MaterializedMasking`; the engine holds no per-forward mask stacks.
                    assert ci_stacked is not None
                    masking: Masking = StochasticMasking(
                        ci_stacked=ci_stacked, draw_key=draw_key, routes=routes
                    )
                case ConstantSources() as strategy:
                    masking = MaterializedMasking(
                        component_masks=constant_source_masks(strategy, ci.lower),
                        weight_delta_masks=None,
                        routes=routes,
                    )
                case UnmaskedNoDeltaSources():
                    raise AssertionError(
                        "UnmaskedNoDeltaSources is non-target-pass vocabulary "
                        "(SPEC T4/T5); the main grid never carries it"
                    )
                case FreshPGDSources():
                    component_masks, weight_delta_masks = masks_from_sources(
                        ci.lower, ascended.fresh_sources[term_idx]
                    )
                    masking = MaterializedMasking(
                        component_masks=component_masks,
                        weight_delta_masks=weight_delta_masks,
                        routes=routes,
                    )
                case PersistentSources(state_key=state_key):
                    component_masks, weight_delta_masks = masks_from_sources(
                        draw_ci_lower, persistent_sources[state_key]
                    )
                    masking = MaterializedMasking(
                        component_masks=component_masks,
                        weight_delta_masks=weight_delta_masks,
                        routes=routes,
                    )
                case MixedPersistentStochasticSources(state_key=state_key):
                    adv_fraction = scheduled_value_at(train_frac, term.sources.cfg.adv_fraction)
                    component_masks, weight_delta_masks, routes = mixed_persistent_stochastic_masks(
                        key=draw_key,
                        ci_lower=model_cotangents_scaled(ci.lower, term_coeffs[term.name]),
                        persistent_sources=persistent_sources[state_key],
                        leading=stream.leading,
                        adv_fraction=adv_fraction,
                        stochastic_routes=routes,
                    )
                    masking = MaterializedMasking(
                        component_masks=component_masks,
                        weight_delta_masks=weight_delta_masks,
                        routes=routes,
                    )
            return substrate.masked_recon(
                model,
                prepared_weights=draw_prepared,
                batch=stream.batch,
                masking=masking,
                capture_keys=hidden_acts_capture_keys(reconstruction_specs[term.name]),
                reconstruction=reconstruction_specs[term.name],
                clean=stream.clean,
            )

    return draw_loss


def retake_e2e_source_grads[PreparedT](
    substrate: ForwardSubstrate[PreparedT],
    grid: ReconGrid[MaskSourceStrategy],
    model: PlacedModel[PreparedT],
    *,
    prepared_weights: PreparedT,
    ci: CI,
    ascended: AscendedAdversaries,
    stream: StreamInputs,
    draws_per_term: list[TermDraws],
    train_frac: Array,
    warmed_sources: dict[str, Sources],
    term_coeffs: dict[str, Array | float],
) -> dict[str, Sources]:
    """Recompute final persistent-source gradients using output reconstruction only."""
    e2e_terms = grid.e2e_terms_requiring_source_grad_retake_by_key
    if not e2e_terms:
        return {}

    detached_ci = jax.lax.stop_gradient(ci)
    term_indices = {term.name: idx for idx, term in enumerate(grid.terms)}
    grads: dict[str, Sources] = {}
    for state_key, term in e2e_terms.items():
        term_idx = term_indices[term.name]

        def e2e_loss(
            sources: Sources,
            state_key: str = state_key,
            term: ReconLossTerm[MaskSourceStrategy] = term,
            term_idx: int = term_idx,
        ) -> Array:
            draw_loss = main_draw_loss(
                substrate,
                model,
                prepared_weights=prepared_weights,
                ci=detached_ci,
                ci_stacked=None,
                persistent_sources=warmed_sources | {state_key: sources},
                ascended=ascended,
                stream=stream,
                train_frac=train_frac,
                reconstruction_specs={term.name: OutputOnlyReconstruction()},
                term_coeffs=term_coeffs,
            )
            return mean_reconstruction_losses(
                tuple(
                    draw_loss(term_idx, term, draw_key, routes)
                    for draw_key, routes in draws_per_term[term_idx]
                )
            ).total

        with jax.named_scope("pd_pgd_e2e_final_grad"):
            grads[state_key] = jax.grad(e2e_loss)(warmed_sources[state_key])
    return grads


def apply_gradients(
    components_optimizer: optax.GradientTransformation,
    ci_fn_optimizer: optax.GradientTransformation,
    decomposition: Decomposition,
    training: TrainingItem,
    warmed_advs: dict[str, PersistentAdversary],
    components_grad: Any,
    ci_fn_grad: Any,
    persistent_source_grads: dict[str, Sources],
    train_frac: Array,
    freq_ema: dict[str, Array] | None,
    mesh: Mesh | None,
) -> tuple[TrainState, dict[str, Array]]:
    """The optimizer tail: grad-norm metrics, each adversary's final ascent from the
    fused graph (SPEC S13'/S14': the source path is never coeff-scaled, so the
    backward's grad IS dL_term/d(sources) — exact since one source bundle feeds one
    term, S23), then both optimizer updates into the next `TrainState`."""
    grad_norm_metrics = _grad_norm_metrics(components_grad, ci_fn_grad, mesh)

    new_adversaries = {
        state_key: warmed_advs[state_key].final_ascend(
            persistent_source_grads[state_key], train_frac
        )
        for state_key in warmed_advs
    }

    components_updates, new_components_opt_state = components_optimizer.update(
        components_grad,
        training.components_opt_state,
        eqx.filter(decomposition.components, eqx.is_array),
    )
    ci_fn_updates, new_ci_fn_opt_state = ci_fn_optimizer.update(
        ci_fn_grad,
        training.ci_fn_opt_state,
        eqx.filter(decomposition.ci_fn, eqx.is_array),
    )
    new_components = eqx.apply_updates(decomposition.components, components_updates)
    new_ci_fn = eqx.apply_updates(decomposition.ci_fn, ci_fn_updates)

    new_state = TrainState(
        decomposition=Decomposition(components=new_components, ci_fn=new_ci_fn),
        training=TrainingItem(
            components_opt_state=new_components_opt_state,
            ci_fn_opt_state=new_ci_fn_opt_state,
            adversaries=new_adversaries,
            freq_ema=freq_ema,
            step=training.step + 1,
        ),
    )
    return new_state, grad_norm_metrics


def shared_step_metrics(
    terms: tuple[ReconLossTerm[MaskSourceStrategy], ...],
    imp: ImportanceMinimalityTerm,
    *,
    total_loss: Array,
    imp_lp: Array,
    imp_freq: Array,
    freq_batch: Array | None,
    imp_min_param: Array,
    term_breakdowns: tuple[ReconstructionLoss, ...],
    grad_norm_metrics: dict[str, Array],
    adversaries: dict[str, PersistentAdversary],
    train_frac: Array,
) -> dict[str, Array]:
    """Metrics shared by plain and targeted steps; each caller adds its own pass metrics."""
    term_losses = tuple(breakdown.total for breakdown in term_breakdowns)
    metrics = {
        "total": total_loss,
        imp.imp_loss_key: imp_lp,
        "freq": imp_freq,
        **({"freq_batch": freq_batch} if freq_batch is not None else {}),
        imp.imp_min_param_key: imp_min_param,
        **{f"loss/{t.name}": v for t, v in zip(terms, term_losses, strict=True)},
        **grad_norm_metrics,
    }
    for term, breakdown in zip(terms, term_breakdowns, strict=True):
        prefix = f"loss/{term.name}"
        metrics |= {
            f"{prefix}/{suffix}": value
            for suffix, value in reconstruction_loss_metrics(breakdown).items()
        }
    source_lrs = {k: adv.source_lr(train_frac) for k, adv in adversaries.items()}
    if len(source_lrs) == 1:
        metrics["src_lr"] = next(iter(source_lrs.values()))
    else:
        metrics |= {f"schedules/lr/src/{k}": v for k, v in source_lrs.items()}
    return metrics


# ───────────────────────────── the step factory ─────────────────────────────


class MainLossAux(NamedTuple):
    """The plain step's `has_aux` payload: `reported_total` is the objective Σ coeff·L
    (the differentiated total differs only in persistent-source plumbing, SPEC S14')."""

    reported_total: Array
    faith_loss: Array
    imp_lp: Array
    freq: FrequencyTerm | None
    nonlinearity_metrics: dict[str, Array]
    term_breakdowns: tuple[ReconstructionLoss, ...]


def make_train_step[PreparedT](
    model_static: PlacedModel[PreparedT],
    *,
    substrate: ForwardSubstrate[PreparedT],
    objective: LossSurface,
    components_optimizer: optax.GradientTransformation,
    ci_fn_optimizer: optax.GradientTransformation,
    total_steps: int,
    faithfulness: FaithfulnessLossFn,
    compiler_options: dict[str, bool | int | str] | None = None,
):
    """Build the plain VPD step from its forward substrate and objective."""
    grid = ReconGrid.of(objective.recon, key_offset=1)
    imp = objective.imp
    nonlinearity = ResolvedNonlinearity.resolve(objective.nonlinearity, model_static.sites)
    assert total_steps > 0, total_steps
    model_static.assert_hidden_acts_reconstruction_points(tuple(sorted(grid.capture_keys)))
    freq_role = resolve_frequency(imp.cfg.frequency)
    coeff_schedules: dict[str, LossCoeff] = {
        imp.name: imp.coeff,
        **{term.name: term.coeff for term in grid.terms},
        **{
            f"{term.name}/hidden_acts_reconstruction": term.hidden_acts_reconstruction.coeff
            for term in grid.terms
            if term.hidden_acts_reconstruction is not None
        },
    }
    coeff_schedules[objective.faith.name] = objective.faith.coeff
    if freq_role is not None:
        coeff_schedules[f"{imp.name}/frequency"] = freq_role.coeff
    if nonlinearity is not None:
        coeff_schedules[nonlinearity.term.name] = nonlinearity.term.coeff

    @jaxtyped(typechecker=beartype)
    def step(
        model: PlacedModel[PreparedT],
        state: TrainState,
        batch: Any,
        key: PRNGKeyArray,
    ) -> tuple[TrainState, dict[str, Array]]:
        decomposition = state.decomposition
        training = state.training
        train_frac = train_frac_at(training.step, total_steps)
        imp_min_param = annealed_imp_min_param(train_frac, imp.cfg)
        # Every coefficient's per-step value, resolved once at the top of the step: the
        # loss math below sees only scalars, never schedule objects.
        imp_coeff = coeff_at(train_frac, imp.coeff)
        freq_coeff = 0.0 if freq_role is None else coeff_at(train_frac, freq_role.coeff)
        recon_coeffs = tuple(grid.coeffs_at(train_frac))
        term_coeffs: dict[str, Array | float] = {
            term.name: coeff for term, coeff in zip(grid.terms, recon_coeffs, strict=True)
        }
        reconstruction_specs = grid.reconstruction_specs_at(train_frac)

        stream = substrate.prep_stream(model, batch, grid.capture_keys)

        # ── adversary ascents: params + CI detached (SPEC §4.5) ──
        prepared_weights, recon_vjp = substrate.component_weights_vjp(
            model, decomposition.components
        )
        detached_prepared_weights = jax.lax.stop_gradient(prepared_weights)
        # The CI envelope is a pure fn of the batch, so compute it ONCE per step — the value +
        # its vjp, mirroring `prepared_weights`/`recon_vjp`. The ascend uses the stop_gradient'd
        # value; `loss_fn` takes the live value and its gradient crosses the forward and
        # resident-weight pullbacks. So the (≈10x-the-target) CI fn is forward-evaluated ONCE,
        # not once detached for the ascend + once inside the main backward.
        compute_ci_fn, ci_weights_vjp = substrate.ci_weights_vjp(decomposition.ci_fn)
        any_ci, ci_vjp = substrate.ci_forward_vjp(compute_ci_fn, stream.taps)
        # The plain objective has one readout role by construction (S37 is tPD-only).
        ci = output_ci(any_ci)
        ci_lower_detached = jax.lax.stop_gradient(ci).lower

        ascended = ascend_adversaries(
            substrate,
            grid,
            model,
            stream,
            detached_prepared_weights,
            ci_lower_detached,
            training.adversaries,
            key,
            train_frac,
            grid.adversary_reconstruction_specs(reconstruction_specs),
        )

        # ── main losses: live components/ci; the PERSISTENT sources participate in
        # the graph so their gradient comes from the SAME backward (SPEC S14'); they
        # are NOT detached here, but components/ci grads through them are what torch
        # gets too (sources are leaves). ──
        warmed_sources = {k: a.sources for k, a in ascended.warmed.items()}
        draws_per_term = grid.draws(key, ascended.fixed_routes, stream.leading)

        def loss_fn(
            trainable: tuple[PreparedT, ComponentStacks, CI, dict[str, Sources]],
        ) -> tuple[Array, MainLossAux]:
            prepared_weights, components, ci, persistent_sources = trainable
            ci_stacked = model.stack_ci(ci.lower)
            faith_loss = faithfulness(faithfulness_weight_deltas(model, components))
            faith_term = coeff_at(train_frac, objective.faith.coeff) * faith_loss
            frequencies = per_component_frequencies(ci.upper, imp.cfg, imp_min_param)
            imp_lp = lp_term(frequencies)

            draw_loss = main_draw_loss(
                substrate,
                model,
                prepared_weights=prepared_weights,
                ci=ci,
                ci_stacked=ci_stacked,
                persistent_sources=persistent_sources,
                ascended=ascended,
                stream=stream,
                train_frac=train_frac,
                reconstruction_specs=reconstruction_specs,
                term_coeffs=term_coeffs,
            )
            term_breakdowns = grid.losses(draws_per_term, draw_loss)
            term_losses = tuple(breakdown.total for breakdown in term_breakdowns)
            match freq_role:
                case None:
                    assert training.freq_ema is None, (
                        "freq_ema state without a frequency config (S8'')"
                    )
                    freq = None
                case BatchFrequency():
                    assert training.freq_ema is None, "freq_ema state without the EMA mode (S8'')"
                    freq = freq_role.resolved(ci.upper).term(frequencies)
                case EmaFrequency():
                    freq = freq_role.resolved(ci.upper).term(
                        frequencies, training.freq_ema, jnp.asarray(training.step, jnp.float32)
                    )
            base = faith_term + imp_coeff * imp_lp
            if freq is not None:
                base = base + freq_coeff * freq.freq
            nonlinearity_metrics: dict[str, Array] = {}
            if nonlinearity is not None:
                weighted, nonlinearity_metrics = nonlinearity.weighted_loss_and_metrics(
                    train_frac, components
                )
                base = base + weighted
            # The differentiated total: persistent-carrying terms enter at weight 1 —
            # their coeff already rides their model-side cotangents — so the backward
            # hands each adversary dL/ds unscaled (SPEC S14'). The OBJECTIVE (the
            # reported `total`, Σ coeff·L) has the same gradients up to that plumbing
            # and the identical value for every non-persistent term.
            total_loss = base
            reported_total = base
            for term, coeff, term_loss in zip(grid.terms, recon_coeffs, term_losses, strict=True):
                match coeff_application(term):
                    case "scales_loss":
                        total_loss = total_loss + coeff * term_loss
                    case "scales_model_cotangents":
                        total_loss = total_loss + term_loss
                reported_total = reported_total + coeff * term_loss
            return total_loss, MainLossAux(
                reported_total=reported_total,
                faith_loss=faith_loss,
                imp_lp=imp_lp,
                freq=freq,
                nonlinearity_metrics=nonlinearity_metrics,
                term_breakdowns=term_breakdowns,
            )

        with jax.named_scope("pd_value_and_grad"):
            (_, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                (prepared_weights, decomposition.components, ci, warmed_sources)
            )
        prepared_grad, components_grad_direct, ci_grad, persistent_source_grads = grads
        components_grad_recon = recon_vjp(prepared_grad)[0]
        # faith and nonlinearity read the components DIRECTLY (weight-space terms, not
        # through the prepared-weights vjp), so the direct grads join the recon-path grads.
        components_grad = jax.tree.map(
            lambda recon_g, direct_g: recon_g + direct_g,
            components_grad_recon,
            components_grad_direct,
        )
        ci_fn_grad = ci_weights_vjp(ci_vjp(ci_grad)[0])[0]
        persistent_source_grads = persistent_source_grads | retake_e2e_source_grads(
            substrate,
            grid,
            model,
            prepared_weights=detached_prepared_weights,
            ci=ci,
            ascended=ascended,
            stream=stream,
            draws_per_term=draws_per_term,
            train_frac=train_frac,
            warmed_sources=warmed_sources,
            term_coeffs=term_coeffs,
        )

        match aux.freq:
            case None:
                imp_freq, freq_batch, new_freq_ema = jnp.zeros((), jnp.float32), None, None
            case BatchFrequencyTerm(freq_value):
                imp_freq, freq_batch, new_freq_ema = freq_value, None, None
            case EmaFrequencyTerm(freq_value, freq_batch_value, new_ema):
                imp_freq, freq_batch, new_freq_ema = freq_value, freq_batch_value, new_ema

        new_state, grad_norm_metrics = apply_gradients(
            components_optimizer,
            ci_fn_optimizer,
            decomposition,
            training,
            ascended.warmed,
            components_grad,
            ci_fn_grad,
            persistent_source_grads,
            train_frac,
            freq_ema=new_freq_ema,
            mesh=substrate.mesh,
        )
        metrics = (
            shared_step_metrics(
                grid.terms,
                imp,
                total_loss=aux.reported_total,
                imp_lp=aux.imp_lp,
                imp_freq=imp_freq,
                freq_batch=freq_batch,
                imp_min_param=imp_min_param,
                term_breakdowns=aux.term_breakdowns,
                grad_norm_metrics=grad_norm_metrics,
                adversaries=training.adversaries,
                train_frac=train_frac,
            )
            | aux.nonlinearity_metrics
            | _scheduled_coeff_metrics(train_frac, coeff_schedules)
        )
        metrics["faith"] = aux.faith_loss
        return new_state, metrics

    return filter_jit(step, donate="all-except-first", compiler_options=compiler_options)


# ───────────────────────────── the targeted (tPD) step factory ─────────────────────────────


@dataclass(frozen=True)
class CIScaledWeightDecay:
    """The tPD CI-scaled weight decay (SPEC T11): its coefficient joined with the
    components optimizer's LR schedule, applied after the optimizer update."""

    coeff: float
    components_lr: ScheduleConfig

    def apply(
        self,
        state: TrainState,
        ci_bundles: Sequence[CI],
        train_frac: Array,
        site_names: tuple[str, ...],
    ) -> tuple[TrainState, dict[str, Array]]:
        """Apply T11 after the optimizer update, using this step's pre-update CIs.

        `ci_bundles` is EVERY pass's bundle — both streams, and both CI heads when the run
        carries a hidden pass. The decay reads their per-component max, so a component that
        matters on either stream OR either head is not dead and is never decayed away."""
        assert ci_bundles, "CI-scaled weight decay needs at least one CI bundle"
        per_bundle = [_per_component_batch_max(ci.lower) for ci in ci_bundles]
        rate = scheduled_value_at(train_frac, self.components_lr) * self.coeff
        decay = {
            site: rate
            * (1.0 - functools.reduce(jnp.maximum, (bundle[site] for bundle in per_bundle)))
            for site in site_names
        }
        decayed = _scale_subcomponents(
            state.decomposition.components, {site: 1.0 - value for site, value in decay.items()}
        )
        new_state = TrainState(
            decomposition=Decomposition(components=decayed, ci_fn=state.decomposition.ci_fn),
            training=state.training,
        )
        decay_all = jnp.concatenate(list(decay.values()))
        return new_state, {
            "ci_scaled_weight_decay/mean": jnp.mean(decay_all),
            "ci_scaled_weight_decay/max": jnp.max(decay_all),
        }


def _per_component_batch_max(ci_lower: dict[str, Array]) -> dict[str, Array]:
    """Each site's per-subcomponent max CI over every leading (batch AND position) axis,
    fp32. Reads `lower` deliberately: `lower ≡ clip(upper, 0, 1)` pointwise (S6), so the
    two squashings agree on this statistic and no clamp is needed."""
    return {
        site: jnp.max(v.astype(jnp.float32), axis=tuple(range(v.ndim - 1)))
        for site, v in ci_lower.items()
    }


def _scale_subcomponents(
    components: ComponentStacks, scale: dict[str, Float[Array, " C"]]
) -> ComponentStacks:
    """Scale each site's V columns and U rows by that site's per-subcomponent factor,
    stacked per semantic group so the multiply stays in the declared layout."""
    rows_by_group: dict[str, list[Array]] = {}
    for name, group, slot in components.site_slots:
        rows = rows_by_group.setdefault(group, [])
        assert slot == len(rows), (name, group, slot)
        rows.append(scale[name])

    def matching(x: Array, keep: Array, c_axis: int) -> Array:
        """`keep [g, C]` re-typed to `x`'s own `(g, C)` assignment before the multiply.

        The factor is derived from CI, whose C axis carries the ACTIVATION layout (`tp`),
        while the component stacks carry the PERSISTENCE layout (C over `('tp',
        'replicate')` under owner partitioning). Broadcasting an elementwise factor across
        that difference is a `ShardingTypeError` under explicit axes, and the two layouts
        need not agree for `V [g, d_in, C]` and `U [g, C, d_out]` either — so the re-type is
        per operand, off that operand's own spec. Pure typing: the value is unchanged."""
        if jax.sharding.get_abstract_mesh().empty:
            return keep
        spec = jax.typeof(x).sharding.spec
        return jax.sharding.reshard(keep, P(spec[0], spec[c_axis]))

    stacks = {}
    for group, (vs, us) in components.stacks.items():
        keep = jnp.stack(rows_by_group[group])  # [g, C]
        stacks[group] = (
            vs * matching(vs, keep, 2)[:, None, :],
            us * matching(us, keep, 1)[:, :, None],
        )
    return ComponentStacks(stacks=stacks, site_slots=components.site_slots)


type _NontargetSources = (
    StochasticSources
    | ConstantSources
    | UnmaskedNoDeltaSources
    | PersistentSources
    | MixedPersistentStochasticSources
)
"""The mask sources a non-target pass admits (SPEC T5), named so the grid's narrowing casts
say which vocabulary they are narrowing to."""

type _Trainable[PreparedT] = tuple[PreparedT, AnyCI, AnyCI, dict[str, Sources]]
"""What the tPD passes differentiate through: the compute weights, the target and non-target
CI bundles (dual when the run has a hidden pass), and the persistent adversaries' warmed
sources. EVERY pass sees the whole tuple; a pass that does not read a leaf simply yields a
zero cotangent for it, which is what makes per-pass gradients sum to the fused gradient."""


class _PassAux(NamedTuple):
    """One pass's reportable results, carried out of its `value_and_grad` as aux."""

    reported: Array
    imp_lp: Array
    imp_freq: Array
    breakdowns: tuple[ReconstructionLoss, ...]


@dataclass(frozen=True)
class _PassPlan:
    """One of the up-to-four tPD passes, as static description (SPEC T1/T12).

    The cartesian product of {target, non-target} stream x {output, hidden} CI role. A pass is
    fully described by which stream it reads, which CI head masks it, its recon grid, its
    importance-minimality coefficient, and what its recon compares against — `points` empty
    means the model output, non-empty means exactly those activations and nothing else."""

    label: str
    """Short identity, used as a dict key and in the named scope: `target`, `hidden`,
    `nontarget`, `nontarget_hidden`."""
    role: CIRole
    on_target_stream: bool
    grid: ReconGrid[MaskSourceStrategy]
    """This pass's recon terms and its reserved per-term RNG offset (SPEC T9/R1), disjoint
    from every other pass's."""
    impmin_coeff: LossCoeff
    points: tuple[str, ...]
    adversary_keys: tuple[str, ...]
    """The persistent bundles this pass's terms carry."""

    @property
    def metric_prefix(self) -> str:
        """The pass's metric namespace, DERIVED from its two axes so the two cannot
        disagree: the STREAM prefix (`nontarget_data/`) composed with the CI-ROLE prefix
        (`hidden_ci/`). The target-OUTPUT pass therefore keeps the EMPTY prefix — its keys
        are the unprefixed ones every tPD run has always logged, so a run's curves stay
        comparable whether or not it carries a hidden pass."""
        stream = "" if self.on_target_stream else "nontarget_data/"
        role = "" if self.role == "output" else "hidden_ci/"
        return f"{stream}{role}"


def make_targeted_train_step[PreparedT](
    model_static: PlacedModel[PreparedT],
    *,
    substrate: ForwardSubstrate[PreparedT],
    objective: TargetedObjective,
    ci_scaled_weight_decay: CIScaledWeightDecay | None,
    components_optimizer: optax.GradientTransformation,
    ci_fn_optimizer: optax.GradientTransformation,
    total_steps: int,
    sequential_passes: bool = False,
    compiler_options: dict[str, bool | int | str] | None = None,
):
    """Build the tPD `step(model, state, batch, nontarget_batch, key)` (SPEC §11 / T12).

    Up to FOUR passes over one shared substrate — {target, non-target} streams x {output,
    hidden} CI roles — whose gradients SUM into one optimizer step (T1). The TARGET-OUTPUT
    pass is the full decomposition objective (recon grid + adversary ascents) on the narrow
    stream; the NON-TARGET pass runs its delta-pinned grid + importance-minimality on the
    broad stream; the two HIDDEN passes (present only when the objective carries them)
    repeat that structure against the CI fn's SECOND readout head, scored at internal
    activations instead of the model output. Each stream runs at its own natural geometry
    (T8). There is no faithfulness role anywhere in it.

    `sequential_passes` chooses HOW the passes reach one gradient, not WHAT that gradient is:

    - fused (default): one `value_and_grad` over the sum of every pass.
    - sequential: one `value_and_grad` PER PASS, gradients added, with an
      `optimization_barrier` threading the trainables through the accumulator between passes
      so XLA cannot hoist the next pass's forwards ahead of the previous pass's backward.

    The two are identical in exact arithmetic — summing per-pass gradients is precisely what
    the fused backward does internally, and each pass's `value_and_grad` sees the same tuple
    (a pass that does not read a leaf contributes a zero cotangent for it). They differ only in
    buffer lifetime: the sequential form holds ONE pass's masked-forward residuals at a time
    instead of every pass's at once, which is what keeps peak memory flat as passes are added.

    The batch args are the streams: `batch` the target stream (whose global batch is
    `pd.batch_size` — persistent sources size from it), `nontarget_batch` the broad stream."""
    assert total_steps > 0, total_steps
    imp = objective.target.imp
    imp_name = imp.name

    # The library boundary behind the non-target passes' narrow types, for objectives built
    # outside them (SPEC T5, amended 2026-08-19/20: the OUTPUT pass admits both persistent
    # adversarial strategies; the HIDDEN pass admits the merged one only — the standalone
    # bundle would be an extra forward, which the amendment deliberately excludes).
    for term in objective.nontarget.recon:
        assert isinstance(
            term.sources,
            StochasticSources
            | ConstantSources
            | UnmaskedNoDeltaSources
            | PersistentSources
            | MixedPersistentStochasticSources,
        ), term.name
    if objective.nontarget_hidden is not None:
        for term in objective.nontarget_hidden.recon:
            assert isinstance(
                term.sources,
                StochasticSources
                | ConstantSources
                | UnmaskedNoDeltaSources
                | MixedPersistentStochasticSources,
            ), term.name

    # Pass ORDER is load-bearing for RNG only (T9): each pass's per-term keys derive from its
    # grid's `key_offset`, and the offsets run in this order. Target-output then non-target
    # keeps a run WITHOUT hidden passes byte-identical to the pre-T12 step; the hidden passes
    # offset past both, so adding them does not move an existing run's draws.
    plans: list[_PassPlan] = []
    persistent_term_by_key: dict[str, AnyReconLossTerm] = {}
    key_offset = 1

    def add_pass(
        label: str,
        role: CIRole,
        on_target_stream: bool,
        terms: tuple[AnyReconLossTerm, ...],
        impmin_coeff: LossCoeff,
        points: tuple[str, ...],
    ) -> None:
        """Resolve one pass and append it. Adversary state keys are a GLOBAL namespace, so
        each pass's bundles are indexed into ONE table — `index_persistent_terms` is what
        catches a key that would feed two passes."""
        nonlocal key_offset
        grid: ReconGrid[MaskSourceStrategy] = ReconGrid.of(terms, key_offset=key_offset)
        index_persistent_terms(terms, into=persistent_term_by_key)
        plans.append(
            _PassPlan(
                label=label,
                role=role,
                on_target_stream=on_target_stream,
                grid=grid,
                impmin_coeff=impmin_coeff,
                points=points,
                adversary_keys=tuple(grid.persistent_by_key),
            )
        )
        key_offset += len(terms)

    add_pass("target", "output", True, objective.target.recon, imp.coeff, ())
    add_pass(
        "nontarget",
        "output",
        False,
        cast("tuple[AnyReconLossTerm, ...]", objective.nontarget.recon),
        objective.nontarget.impmin_coeff,
        (),
    )
    if objective.hidden is not None:
        add_pass(
            "hidden",
            "hidden",
            True,
            objective.hidden.recon,
            objective.hidden.impmin_coeff,
            objective.hidden.points,
        )
    if objective.nontarget_hidden is not None:
        add_pass(
            "nontarget_hidden",
            "hidden",
            False,
            cast("tuple[AnyReconLossTerm, ...]", objective.nontarget_hidden.recon),
            objective.nontarget_hidden.impmin_coeff,
            objective.nontarget_hidden.points,
        )
    passes = tuple(plans)
    target_plan = passes[0]

    def stream_capture_keys(on_target_stream: bool) -> CaptureKeys:
        """What one stream's clean forward must capture: every pass on it contributes its
        terms' S35 rider points and, for a hidden pass, the points it scores at."""
        return frozenset(
            key
            for plan in passes
            if plan.on_target_stream == on_target_stream
            for key in (plan.grid.capture_keys | frozenset(plan.points))
        )

    target_stream_capture_keys = stream_capture_keys(True)
    nontarget_stream_capture_keys = stream_capture_keys(False)
    # The same refusal the S35 rider's points get: a point no masking can reach would
    # contribute guaranteed zeros to the pass's point mean and silently weaken it.
    model_static.assert_hidden_acts_reconstruction_points(
        tuple(sorted(target_stream_capture_keys | nontarget_stream_capture_keys))
    )

    freq_role = resolve_frequency(imp.cfg.frequency)
    nonlinearity = ResolvedNonlinearity.resolve(objective.nonlinearity, model_static.sites)
    coeff_schedules: dict[str, LossCoeff] = {
        imp_name: imp.coeff,
        **{term.name: term.coeff for term in target_plan.grid.terms},
        **{
            f"{term.name}/hidden_acts_reconstruction": term.hidden_acts_reconstruction.coeff
            for term in target_plan.grid.terms
            if term.hidden_acts_reconstruction is not None
        },
    }
    if freq_role is not None:
        coeff_schedules[f"{imp_name}/frequency"] = freq_role.coeff
    if nonlinearity is not None:
        coeff_schedules[nonlinearity.term.name] = nonlinearity.term.coeff
    # Each non-target-output pass's schedules live under its OWN namespace, the same
    # mapping the non-target pass has always used.
    other_pass_schedules: list[tuple[str, dict[str, LossCoeff]]] = [
        (
            plan.metric_prefix,
            {imp_name: plan.impmin_coeff, **{t.name: t.coeff for t in plan.grid.terms}},
        )
        for plan in passes[1:]
    ]

    def nontarget_draw_loss(
        model: PlacedModel[PreparedT],
        prepared_weights: PreparedT,
        nt_ci: CI,
        nt_stream: StreamInputs,
        nt_reconstruction_specs: dict[str, ReconstructionSpec],
        persistent_sources: dict[str, Sources],
        term_coeffs: dict[str, Array | float],
        train_frac: Array,
    ) -> DrawLoss[_NontargetSources]:
        """The non-target grid's per-draw dispatcher: every delta mask pinned to 1.0 —
        except the unmasked-no-delta arm, which pins it to 0.0 (SPEC T4) — scored against the
        broad stream's frozen output, or (non-target HIDDEN pass) its frozen activations.
        The persistent/mixed arms (T5/T7 amended 2026-08-19) compose the bundle's component
        channels delta-pinned, take the term's coeff on their model-side cotangents, and
        enter the total at weight 1 so the backward hands the adversary an unscaled `dL/ds`
        (S14')."""

        def draw_loss(
            term_idx: int,
            term: ReconLossTerm[_NontargetSources],
            draw_key: PRNGKeyArray,
            routes: Routes,
        ) -> ReconstructionLoss:
            del term_idx
            match coeff_application(cast("AnyReconLossTerm", term)):
                case "scales_model_cotangents":
                    draw_prepared = model_cotangents_scaled(
                        prepared_weights, term_coeffs[term.name]
                    )
                    draw_ci_lower = model_cotangents_scaled(nt_ci.lower, term_coeffs[term.name])
                case "scales_loss":
                    draw_prepared, draw_ci_lower = prepared_weights, nt_ci.lower
            with jax.named_scope("pd_nontarget_masked_fwd"):
                match term.sources:
                    case StochasticSources():
                        component_masks, delta_masks = stochastic_delta_pinned_masks(
                            nt_ci.lower, draw_key
                        )
                    case ConstantSources(value=value):
                        component_masks, delta_masks = constant_delta_pinned_masks(
                            value, nt_ci.lower
                        )
                    case UnmaskedNoDeltaSources():
                        component_masks, delta_masks = unmasked_no_delta_masks(nt_ci.lower)
                    case PersistentSources(state_key=state_key):
                        component_masks, delta_masks = persistent_delta_pinned_masks(
                            draw_ci_lower, persistent_sources[state_key]
                        )
                    case MixedPersistentStochasticSources(state_key=state_key):
                        adv_fraction = scheduled_value_at(train_frac, term.sources.cfg.adv_fraction)
                        component_masks, delta_masks, routes = (
                            mixed_persistent_stochastic_delta_pinned_masks(
                                key=draw_key,
                                ci_lower=draw_ci_lower,
                                persistent_sources=persistent_sources[state_key],
                                leading=nt_stream.leading,
                                adv_fraction=adv_fraction,
                                stochastic_routes=routes,
                            )
                        )
                return substrate.masked_recon(
                    model,
                    prepared_weights=draw_prepared,
                    batch=nt_stream.batch,
                    masking=MaterializedMasking(
                        component_masks=component_masks,
                        weight_delta_masks=delta_masks,
                        routes=routes,
                    ),
                    capture_keys=hidden_acts_capture_keys(nt_reconstruction_specs[term.name]),
                    reconstruction=nt_reconstruction_specs[term.name],
                    clean=nt_stream.clean,
                )

        return draw_loss

    @jaxtyped(typechecker=beartype)
    def targeted_step(
        model: PlacedModel[PreparedT],
        state: TrainState,
        batch: Any,
        nontarget_batch: Any,
        key: PRNGKeyArray,
    ) -> tuple[TrainState, dict[str, Array]]:
        decomposition = state.decomposition
        training = state.training
        train_frac = train_frac_at(training.step, total_steps)
        assert is_dual(decomposition.ci_fn) == any(plan.role == "hidden" for plan in passes), (
            "the CI fn's head count and the objective's pass roles disagree: a hidden pass "
            "needs `decomposition.ci.dual`, and a dual CI fn needs a hidden pass — otherwise "
            "its second head would train against nothing"
        )
        imp_min_param = annealed_imp_min_param(train_frac, imp.cfg)
        # Every coefficient's per-step value, resolved once at the top of the step: the
        # loss math below sees only scalars, never schedule objects.
        freq_coeff = 0.0 if freq_role is None else coeff_at(train_frac, freq_role.coeff)
        pass_impmin_coeff = {plan.label: coeff_at(train_frac, plan.impmin_coeff) for plan in passes}
        pass_term_coeffs = {
            plan.label: {
                term.name: coeff
                for term, coeff in zip(
                    plan.grid.terms, plan.grid.coeffs_at(train_frac), strict=True
                )
            }
            for plan in passes
        }

        def specs_for(plan: _PassPlan) -> dict[str, ReconstructionSpec]:
            """What a pass's recon compares against, keyed on its ROLE — the fact that decides
            it — rather than on whether `points` happens to be non-empty. A hidden pass compares
            ONLY its points, with no e2e term at all (T12); an output pass keeps each term's own
            S35 rider resolution."""
            match plan.role:
                case "hidden":
                    assert plan.points, f"hidden pass {plan.label!r} has no points to score"
                    return {
                        term.name: HiddenActsOnlyReconstruction(plan.points)
                        for term in plan.grid.terms
                    }
                case "output":
                    return plan.grid.reconstruction_specs_at(train_frac)

        pass_specs = {plan.label: specs_for(plan) for plan in passes}

        stream = substrate.prep_stream(model, batch, target_stream_capture_keys)
        nt_stream = substrate.prep_stream(model, nontarget_batch, nontarget_stream_capture_keys)

        prepared_weights, recon_vjp = substrate.component_weights_vjp(
            model, decomposition.components
        )
        detached_prepared_weights = jax.lax.stop_gradient(prepared_weights)
        ascend_prepared_weights = substrate.replicate_for_ascend(detached_prepared_weights)
        compute_ci_fn, ci_weights_vjp = substrate.ci_weights_vjp(decomposition.ci_fn)
        # ONE CI forward per stream whatever the head count: the trunk is shared, so both roles
        # come out of a single evaluation behind a single saved pullback (S37).
        ci_any, ci_vjp = substrate.ci_forward_vjp(compute_ci_fn, stream.taps)
        nt_ci_any, nt_ci_vjp = substrate.ci_forward_vjp(compute_ci_fn, nt_stream.taps)

        def bundle_for(plan: _PassPlan, target: AnyCI, nontarget: AnyCI) -> CI:
            return ci_for_role(target if plan.on_target_stream else nontarget, plan.role)

        def stream_for(plan: _PassPlan) -> StreamInputs:
            return stream if plan.on_target_stream else nt_stream

        # ── adversary ascents: each pass against ITS OWN stream, head, and objective ──
        # (§4.5; T7 amended 2026-08-19). Target-stream passes always take the phase (their
        # fresh-PGD entries need it even with no persistent bundle); an off-target pass
        # takes it only when it carries adversaries — then on the BROAD stream, warmup
        # scoring delta-pinned per T4.
        ascended_by_label: dict[str, AscendedAdversaries] = {}
        for plan in passes:
            if not plan.on_target_stream and not plan.adversary_keys:
                continue
            specs = pass_specs[plan.label]
            ascended_by_label[plan.label] = ascend_adversaries(
                substrate,
                plan.grid,
                model,
                stream_for(plan),
                ascend_prepared_weights,
                jax.lax.stop_gradient(bundle_for(plan, ci_any, nt_ci_any)).lower,
                {k: training.adversaries[k] for k in plan.adversary_keys},
                key,
                train_frac,
                # The e2e substitution is an OUTPUT-pass notion: a hidden pass has no
                # end-to-end term to fall back to, so it ascends against its own points.
                plan.grid.adversary_reconstruction_specs(specs) if plan.role == "output" else specs,
                delta_pinned=not plan.on_target_stream,
            )

        warmed_advs = {
            state_key: adv
            for ascended in ascended_by_label.values()
            for state_key, adv in ascended.warmed.items()
        }
        warmed_sources = {state_key: adv.sources for state_key, adv in warmed_advs.items()}
        draws_by_label = {
            plan.label: plan.grid.draws(
                key,
                ascended_by_label[plan.label].fixed_routes
                if plan.label in ascended_by_label
                else {},
                stream_for(plan).leading,
            )
            for plan in passes
        }

        def pass_loss(plan: _PassPlan, trainable: _Trainable[PreparedT]) -> tuple[Array, _PassAux]:
            """ONE pass's scalar loss: importance-minimality on ITS head plus ITS recon grid.

            The differentiated total and the reported one differ only for persistent-carrying
            terms, whose coeff rides their model-side cotangents so the adversary receives an
            unscaled `dL/ds` (SPEC S14')."""
            prepared, ci_target, ci_nontarget, persistent_sources = trainable
            ci = bundle_for(plan, ci_target, ci_nontarget)
            term_coeffs = pass_term_coeffs[plan.label]
            imp_lp, imp_freq = imp_min_terms(ci.upper, imp.cfg, imp_min_param)
            base = pass_impmin_coeff[plan.label] * imp_lp + freq_coeff * imp_freq

            if plan.on_target_stream:
                draw_loss = main_draw_loss(
                    substrate,
                    model,
                    prepared_weights=prepared,
                    ci=ci,
                    ci_stacked=model.stack_ci(ci.lower),
                    persistent_sources=persistent_sources,
                    ascended=ascended_by_label[plan.label],
                    stream=stream,
                    train_frac=train_frac,
                    reconstruction_specs=pass_specs[plan.label],
                    term_coeffs=term_coeffs,
                )
                breakdowns = plan.grid.losses(draws_by_label[plan.label], draw_loss)
            else:
                nt_draw_loss = nontarget_draw_loss(
                    model,
                    prepared,
                    ci,
                    nt_stream,
                    pass_specs[plan.label],
                    persistent_sources,
                    term_coeffs,
                    train_frac,
                )
                breakdowns = cast("ReconGrid[_NontargetSources]", plan.grid).losses(
                    draws_by_label[plan.label], nt_draw_loss
                )

            total = base
            reported = base
            for term, breakdown in zip(plan.grid.terms, breakdowns, strict=True):
                coeff = term_coeffs[term.name]
                match coeff_application(term):
                    case "scales_loss":
                        total = total + coeff * breakdown.total
                    case "scales_model_cotangents":
                        total = total + breakdown.total
                reported = reported + coeff * breakdown.total
            return total, _PassAux(
                reported=reported, imp_lp=imp_lp, imp_freq=imp_freq, breakdowns=breakdowns
            )

        trainable: _Trainable[PreparedT] = (
            prepared_weights,
            ci_any,
            nt_ci_any,
            warmed_sources,
        )

        if sequential_passes:
            grads = None
            aux_by_label: dict[str, _PassAux] = {}
            for plan in passes:
                with jax.named_scope(f"pd_value_and_grad_{plan.label}"):
                    (_, aux), pass_grads = eqx.filter_value_and_grad(
                        lambda t, plan=plan: pass_loss(plan, t), has_aux=True
                    )(trainable)
                grads = (
                    pass_grads
                    if grads is None
                    else jax.tree.map(lambda a, b: a + b, grads, pass_grads)
                )
                # Thread the TRAINABLES through the barrier alongside the accumulator: the next
                # pass's inputs then carry a data dependency on this pass's gradient, so XLA
                # cannot start its forwards before this pass's backward has released its
                # activations. A barrier on the accumulator alone would NOT order them — the
                # next pass's forwards never read it.
                arrays, static = eqx.partition((trainable, grads), eqx.is_array)
                trainable, grads = eqx.combine(jax.lax.optimization_barrier(arrays), static)
                aux_by_label[plan.label] = aux
            assert grads is not None, "a targeted objective always has at least two passes"
        else:

            def fused_loss(t: _Trainable[PreparedT]) -> tuple[Array, dict[str, _PassAux]]:
                total = jnp.zeros((), jnp.float32)
                aux: dict[str, _PassAux] = {}
                for plan in passes:
                    pass_total, pass_aux = pass_loss(plan, t)
                    total = total + pass_total
                    aux[plan.label] = pass_aux
                return total, aux

            with jax.named_scope("pd_value_and_grad"):
                (_, aux_by_label), grads = eqx.filter_value_and_grad(fused_loss, has_aux=True)(
                    trainable
                )

        prepared_grad, ci_grad, nt_ci_grad, persistent_source_grads = grads
        # No faithfulness role ⇒ the components' whole gradient arrives through the
        # compute-weights pullback.
        components_grad = recon_vjp(prepared_grad)[0]
        # The CI fn saw both streams; its total gradient is the sum of the two pullbacks. Under
        # a shared trunk each pullback already carries BOTH heads' cotangents, so the trunk is
        # backward-run once per stream however many passes read it (S37).
        compute_ci_fn_grad = jax.tree.map(
            lambda target_g, nt_g: target_g + nt_g,
            ci_vjp(ci_grad)[0],
            nt_ci_vjp(nt_ci_grad)[0],
        )
        ci_fn_grad = ci_weights_vjp(compute_ci_fn_grad)[0]

        # The nonlinearity prior belongs to no pass: it reads `U` alone, so it is scored
        # ONCE per step against the components and its gradient joins the recon-path
        # component gradient directly — never through the prepared-weights pullback, and
        # never once per pass (which would multiply its coefficient by the pass count).
        nonlinearity_metrics: dict[str, Array] = {}
        nonlinearity_weighted = jnp.zeros((), jnp.float32)
        if nonlinearity is not None:
            resolved_nonlinearity = nonlinearity

            def nonlinearity_value(components: ComponentStacks) -> tuple[Array, dict[str, Array]]:
                return resolved_nonlinearity.weighted_loss_and_metrics(train_frac, components)

            with jax.named_scope("pd_nonlinearity"):
                (nonlinearity_weighted, nonlinearity_metrics), nonlinearity_grad = (
                    eqx.filter_value_and_grad(nonlinearity_value, has_aux=True)(
                        decomposition.components
                    )
                )
            components_grad = jax.tree.map(
                lambda recon_g, direct_g: recon_g + direct_g, components_grad, nonlinearity_grad
            )

        # The e2e retake is an OUTPUT-pass notion (it needs an end-to-end term to re-score
        # against); a hidden pass's grid never qualifies, so its table is empty by
        # construction and the loop simply contributes nothing for it.
        for plan in passes:
            if plan.label not in ascended_by_label or plan.role != "output":
                continue
            persistent_source_grads = persistent_source_grads | retake_e2e_source_grads(
                substrate,
                plan.grid,
                model,
                prepared_weights=detached_prepared_weights,
                ci=bundle_for(plan, ci_any, nt_ci_any),
                ascended=ascended_by_label[plan.label],
                stream=stream_for(plan),
                draws_per_term=draws_by_label[plan.label],
                train_frac=train_frac,
                warmed_sources=warmed_sources,
                term_coeffs=pass_term_coeffs[plan.label],
            )

        assert training.freq_ema is None, "the targeted objective refuses the EMA (S8'')"
        new_state, grad_norm_metrics = apply_gradients(
            components_optimizer,
            ci_fn_optimizer,
            decomposition,
            training,
            warmed_advs,
            components_grad,
            ci_fn_grad,
            persistent_source_grads,
            train_frac,
            freq_ema=None,
            mesh=substrate.mesh,
        )
        wd_metrics: dict[str, Array] = {}
        if ci_scaled_weight_decay is not None:
            # T11: an update rule on the post-step component masters, not a loss term —
            # nothing differentiates through it. Off the step's own pre-update forward CIs,
            # maxed over every pass.
            new_state, wd_metrics = ci_scaled_weight_decay.apply(
                new_state,
                [bundle_for(plan, ci_any, nt_ci_any) for plan in passes],
                train_frac,
                model.site_names,
            )

        # The TARGET-OUTPUT pass keeps `shared_step_metrics`' unprefixed keys, so a run's
        # curves stay comparable whether or not a hidden pass exists; every other pass logs
        # under `<prefix>loss/…`, the shape the non-target pass has always used.
        target_aux = aux_by_label[target_plan.label]
        reported_total = sum(
            (aux.reported for aux in aux_by_label.values()), start=jnp.zeros((), jnp.float32)
        )
        reported_total = reported_total + nonlinearity_weighted
        extra_metrics: dict[str, Array] = {}
        for plan in passes[1:]:
            aux = aux_by_label[plan.label]
            prefix = plan.metric_prefix
            extra_metrics[f"{prefix}loss/{imp.imp_loss_key}"] = aux.imp_lp
            extra_metrics[f"{prefix}loss/FrequencyMinimalityLoss"] = aux.imp_freq
            extra_metrics[f"{prefix}loss/total"] = aux.reported
            for term, breakdown in zip(plan.grid.terms, aux.breakdowns, strict=True):
                extra_metrics[f"{prefix}loss/{term.name}"] = breakdown.total
                for suffix, value in reconstruction_loss_metrics(breakdown).items():
                    extra_metrics[f"{prefix}loss/{term.name}/{suffix}"] = value

        metrics = (
            shared_step_metrics(
                target_plan.grid.terms,
                imp,
                total_loss=reported_total,
                imp_lp=target_aux.imp_lp,
                imp_freq=target_aux.imp_freq,
                freq_batch=None,
                imp_min_param=imp_min_param,
                term_breakdowns=target_aux.breakdowns,
                grad_norm_metrics=grad_norm_metrics,
                adversaries=training.adversaries,
                train_frac=train_frac,
            )
            | extra_metrics
            | nonlinearity_metrics
            | wd_metrics
            | _scheduled_coeff_metrics(train_frac, coeff_schedules)
            | {
                f"{prefix}{key}": value
                for prefix, schedules in other_pass_schedules
                for key, value in _scheduled_coeff_metrics(train_frac, schedules).items()
            }
        )
        return new_state, metrics

    return filter_jit(targeted_step, donate="all-except-first", compiler_options=compiler_options)


# ───────────────────────────── faithfulness warmup (SPEC S21) ─────────────────────────────


def make_faith_warmup_step(
    opt: optax.GradientTransformation,
    faithfulness: FaithfulnessLossFn,
    compiler_options: dict[str, bool | int | str] | None = None,
) -> Callable[
    [PlacedModel, ComponentStacks, optax.OptState],
    tuple[ComponentStacks, optax.OptState, Array],
]:
    """`model` is the jit ARG (frozen weights traced, not baked) — `weight_deltas` reads its
    per-site W slices, so closing over the model would bake them into the HLO."""

    def warmup_step(
        model: PlacedModel, components: ComponentStacks, opt_state: optax.OptState
    ) -> tuple[ComponentStacks, optax.OptState, Array]:
        def loss_fn(components_: ComponentStacks) -> Array:
            return faithfulness(faithfulness_weight_deltas(model, components_))

        loss, grad = eqx.filter_value_and_grad(loss_fn)(components)
        updates, opt_state = opt.update(grad, opt_state, eqx.filter(components, eqx.is_array))
        return eqx.apply_updates(components, updates), opt_state, loss

    return filter_jit(warmup_step, compiler_options=compiler_options)
