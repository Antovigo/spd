"""Open a finished/live JAX single-pool run for offline consumption (harvest, and the
consumers that follow it: clustering, autointerp, slow-eval, app).

This is the reusable "load a JAX run" pattern. It reads a run dir
(`runs/<p-id>/{launch_config.yaml, ckpts/}`), rebuilds the frozen
target + `DecomposedModel` from the pinned config, restores the checkpoint's
`decomposition` item (the trained V/U + ci_fn — optimizer/adversary state is training's
business and is never touched), and exposes the pure forward a consumer needs:

    run = open_jax_run(run_dir, data_root=data_root)   # latest checkpoint
    fwd = run.forward(token_ids)                # one frozen, forward-only pass
    fwd.lower_leaky_ci_by_site[site]            # (B, T, C) leaky CI per site
    fwd.component_activations_by_site[site]     # (B, T, C) ‖U_c‖ · (x @ V) per site
    fwd.output_probabilities                    # (B, T, vocab) softmax of clean logits

No torch, no safetensors bridge: the V/U + CI fn come straight from the
orbax checkpoint and the target is built from its own config. CPU-friendly (jax falls
back to CPU); a single device is enough for a small harvest.

`forward` mirrors the forward-only subset of `eval.make_eval_step`: clean logits +
the CI fn's residual taps + lower-leaky CI, plus per-component acts (the harvest extra,
from the target-owned component-activation forward). That target method combines these
needs into one frozen pass, capturing only the requested CI taps plus the distinct matrix
inputs needed for the decomposed sites. bf16 compute on the components / CI fn (training's
`COMPUTE_DT`) so consumed CI matches the trained model's; output probs are fp32 from the
fp32-upcast frozen forward.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from param_decomp.core import placement
from param_decomp.core.checkpoint import (
    make_read_only_checkpoint_manager,
    restore_decomposition_to_host,
)
from param_decomp.core.ci_fn import (
    ChunkwiseTransformerCIFn,
    GlobalMLPCIFn,
    evaluate_ci,
)
from param_decomp.core.components import ComponentStacks
from param_decomp.core.model import DecomposedModel, prepare_compute_weights
from param_decomp.core.run_state import init_decomposition
from param_decomp.core.sharding import hsdp_mesh, place_target, place_via_shardings
from param_decomp.core.train import Decomposition
from param_decomp.experiments.lm.config import LMCIFnArch, hf_model_family
from param_decomp.experiments.lm.deliverable import ResolvedDeliverable, load_deliverable
from param_decomp.experiments.lm.resolved import (
    AnyLMTargetConfig,
    LlamaSimpleMLPTargetConfig,
    TargetConfig,
    weights_jnp_dtype,
)
from param_decomp.infra import pretrain_cache
from param_decomp.targets import llama_simple_mlp
from param_decomp.targets.glu_transformer import GLUDecomposedModel, glu_site_specs

type TransformerDecomposedModel = GLUDecomposedModel | llama_simple_mlp.SimpleMLPDecomposedModel

LMCIFn = ChunkwiseTransformerCIFn | GlobalMLPCIFn
"""The CI fns an LM deliverable resolves to (`LMCIFnArch`, built). A plain union (not a
`type` alias) so the narrowing isinstance in `LoadedJaxRun.forward` can consume it."""


@partial(
    jax.tree_util.register_dataclass,
    data_fields=(
        "lower_leaky_ci_by_site",
        "component_activations_by_site",
        "output_probabilities",
    ),
    meta_fields=(),
)
@dataclass(frozen=True)
class HarvestForward:
    """One frozen forward-only pass over a token batch, the raw material every harvest
    fn turns into per-component statistics. Site-name-keyed; `(B, T, C_site)`."""

    lower_leaky_ci_by_site: dict[str, Float[Array, "B T C"]]
    component_activations_by_site: dict[str, Float[Array, "B T C"]]
    output_probabilities: Float[Array, "B T vocab"]


def build_target(
    target: AnyLMTargetConfig, mesh: jax.sharding.Mesh, data_root: Path
) -> tuple[TransformerDecomposedModel, int]:
    """`(model, vocab_size)` for one target config — the SECTION, not a whole run config,
    so every run shape (plain, targeted) and every stored-run consumer shares this one
    loader. The `model` (an `eqx.Module`) IS the frozen target — it carries the full model
    weights (embedding included) as fields and embeds its token input internally.
    SimpleMLP reads its pretrain cache under `data_root` (no network); the HF families
    read the HF snapshot. Both cast their weights to the config's `weights_dtype` on
    read — this is the ONLY place that dtype is applied, so train and consume load the
    same target.

    LM-only by type: the toy targets satisfy only the core `TargetSites` protocol and
    cannot reach this loader — they validate via their in-loop target-CI metric."""
    match target:
        case LlamaSimpleMLPTargetConfig():
            cache_dir = pretrain_cache.resolved_cache_dir(data_root, target.pretrain_run_path)
            simple_cfg = llama_simple_mlp.load_model_config(cache_dir)
            sites = llama_simple_mlp.site_specs(simple_cfg, target.sites)
            loaded_model = llama_simple_mlp.load_decomposed_lm_from_pretrain_cache(
                cache_dir, simple_cfg, sites, weights_jnp_dtype(target.weights_dtype)
            )
            model = place_via_shardings(loaded_model, loaded_model.shardings(mesh))
            return model, simple_cfg.vocab_size
        case TargetConfig():
            family = hf_model_family(target.model_name)
            arch_cfg = family.arch_config()
            sites = glu_site_specs(arch_cfg, target.sites)
            loaded_model = family.load(
                target.model_name,
                arch_cfg,
                sites,
                weights_jnp_dtype(target.weights_dtype),
            )
            return place_target(loaded_model, mesh), arch_cfg.vocab_size


def _u_norms(
    components: ComponentStacks, site_names: tuple[str, ...]
) -> dict[str, Float[Array, " C"]]:
    """Per-component output-direction magnitude ‖U_c‖ — the harvest `component_activation`
    scale (torch `harvest_fn/param_decomp.core.py`: `component.U.norm(dim=1)`)."""
    return {
        site: jnp.linalg.norm(components.site(site).U.astype(jnp.float32), axis=1)
        for site in site_names
    }


@dataclass(frozen=True)
class LoadedJaxRun:
    """A JAX run opened for consumption: restored trajectory + frozen target + the pure
    forward consumers need. `layer_activation_sizes` / `vocab_size` mirror the torch
    `PDAdapter` fields the harvest pipeline keys on."""

    run_id: str
    step: int
    model: TransformerDecomposedModel
    deliverable: ResolvedDeliverable
    vocab_size: int
    _decomposition: Decomposition
    _forward: Callable[
        [DecomposedModel, ComponentStacks, LMCIFn, Int[Array, "B T"]],
        HarvestForward,
    ]

    @property
    def site_names(self) -> tuple[str, ...]:
        return self.model.site_names

    @property
    def layer_activation_sizes(self) -> list[tuple[str, int]]:
        """`(site_name, C)` per decomposed site, in canonical order — the harvest
        accumulator's `layers` argument."""
        return [(s.name, s.C) for s in self.model.sites]

    def forward(self, token_ids: Int[Array, "B T"]) -> HarvestForward:
        ci_fn = self._decomposition.ci_fn
        assert isinstance(ci_fn, LMCIFn), "harvest is the LM path only"
        return self._forward(self.model, self._decomposition.components, ci_fn, token_ids)


def _restore_decomposition(
    ci_fn: LMCIFnArch,
    model: TransformerDecomposedModel,
    mesh: jax.sharding.Mesh,
    run_dir: Path,
    step: int | None,
) -> tuple[Decomposition, int]:
    """Restore ONLY the trained decomposition from the run's checkpoint.

    Consumers never need the optimizer moments or persistent-PGD adversary sources
    training also checkpoints — and on a single device those dominate: an 8B run's
    sources + Adam state materialize ~60GB at init and again at restore staging, which
    wedges/OOMs one GPU. `jax.eval_shape` over `init_decomposition` yields the saved
    decomposition's structure with ZERO allocation (and zero knowledge of training's
    optimizers); leaves restore as host numpy, then `device_put` onto the consumer's
    single default device.

    The reference is placement- and key-invariant under `eval_shape` (treedef, shapes,
    dtypes derive from sites + CI arch alone), so nothing from the run's process record
    enters: `ddp` and a constant key stand in for the launch values a consumer never
    needed."""
    rules = placement.from_config_for_consumer("ddp", mesh, model.sites)
    abstract = jax.eval_shape(
        lambda: init_decomposition(model, ci_fn, jax.random.PRNGKey(0), mesh, rules)
    )

    manager = make_read_only_checkpoint_manager(run_dir / "ckpts")
    resolved_step = manager.latest_step() if step is None else step
    assert resolved_step is not None, f"no checkpoints under {run_dir / 'ckpts'}"
    decomposition = jax.device_put(restore_decomposition_to_host(manager, resolved_step, abstract))
    return decomposition, resolved_step


def open_jax_run(run_dir: Path, step: int | None = None, *, data_root: Path) -> LoadedJaxRun:
    """Open the run at `run_dir`; restore checkpoint `step` (latest if None). Restores
    only the trained decomposition (see `_restore_decomposition`). `data_root` resolves a
    `kind: pretrained` target's cache (`<data_root>/pretrain_cache/...`)."""
    deliverable = load_deliverable(run_dir, data_root)
    mesh = hsdp_mesh()
    target_model, vocab_size = build_target(deliverable.target, mesh, data_root)
    decomposition, resolved_step = _restore_decomposition(
        deliverable.ci_fn, target_model, mesh, run_dir, step
    )
    assert isinstance(decomposition.components, ComponentStacks)

    site_names = target_model.site_names
    u_norms = _u_norms(decomposition.components, site_names)

    # `model` is the filter_jit ARG (frozen weights traced, not baked; distinct from the
    # outer `target_model` so an accidental closure capture fails loudly). It embeds the
    # token ids internally — the harvest forward feeds tokens straight in.
    @eqx.filter_jit
    def forward(
        model: DecomposedModel,
        components: ComponentStacks,
        ci_fn: LMCIFn,
        token_ids: Int[Array, "B T"],
    ) -> HarvestForward:
        prepared_weights = prepare_compute_weights(model, components)
        clean_forward_result, raw_component_activations_by_site = (
            model.component_activation_forward(
                prepared_weights,
                token_ids,
                capture_keys=ci_fn.capture_keys,
            )
        )
        ci_input_activations_by_key = clean_forward_result.captures

        ci = evaluate_ci(ci_fn, ci_input_activations_by_key, remat=False)
        lower_leaky_ci_by_site = {site: ci.lower[site].astype(jnp.float32) for site in site_names}
        component_activations_by_site = {
            site: raw_component_activations_by_site[site].astype(jnp.float32) * u_norms[site]
            for site in site_names
        }
        output_probabilities = jax.nn.softmax(
            clean_forward_result.output.astype(jnp.float32), axis=-1
        )

        return HarvestForward(
            lower_leaky_ci_by_site=lower_leaky_ci_by_site,
            component_activations_by_site=component_activations_by_site,
            output_probabilities=output_probabilities,
        )

    return LoadedJaxRun(
        run_id=run_dir.name,
        step=resolved_step,
        model=target_model,
        deliverable=deliverable,
        vocab_size=vocab_size,
        _decomposition=decomposition,
        _forward=forward,
    )


@dataclass(frozen=True)
class RunMetadata:
    """A JAX run's target topology, read from config + cache WITHOUT restoring a
    checkpoint — the metadata the autointerp/clustering consumers need (`n_blocks`,
    `vocab_size`, per-site `(name, C)`). `model_type` selects the canonical-path schema
    consumers use to render human-readable layer descriptions."""

    model_type: str
    n_blocks: int
    vocab_size: int
    layer_activation_sizes: list[tuple[str, int]]


def run_metadata(run_dir: Path, *, data_root: Path) -> RunMetadata:
    """Target topology for `run_dir`, derived from the run's deliverable (+ the SimpleMLP
    pretrain cache's `model_config.yaml` for `n_layer`/`vocab_size`). No orbax restore."""
    target = load_deliverable(run_dir, data_root).target
    match target:
        case LlamaSimpleMLPTargetConfig():
            cache_dir = pretrain_cache.resolved_cache_dir(data_root, target.pretrain_run_path)
            simple_cfg = llama_simple_mlp.load_model_config(cache_dir)
            return RunMetadata(
                model_type="LlamaSimpleMLP",
                n_blocks=simple_cfg.n_layer,
                vocab_size=simple_cfg.vocab_size,
                layer_activation_sizes=[(s.name, s.C) for s in target.sites],
            )
        case TargetConfig():
            family = hf_model_family(target.model_name)
            arch_cfg = family.arch_config()
            return RunMetadata(
                model_type=family.model_type,
                n_blocks=arch_cfg.n_layer,
                vocab_size=arch_cfg.vocab_size,
                layer_activation_sizes=[(s.name, s.C) for s in target.sites],
            )
