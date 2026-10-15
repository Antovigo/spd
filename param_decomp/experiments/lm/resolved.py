"""Runtime objects resolved from an authored LM config."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import jax.numpy as jnp
from jax.typing import DTypeLike

from param_decomp.attention import AttentionImplementation
from param_decomp.core.built_run import BuiltRun
from param_decomp.core.ci_fn.implementations.block_selected.arch import (
    BlockSelectedChunkwiseTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.chunkwise.arch import ChunkwiseTransformerCIFnArch
from param_decomp.core.ci_fn.implementations.global_mlp import GlobalMLPCIFnArch
from param_decomp.core.ci_fn.implementations.global_transformer.arch import (
    GlobalTransformerCIFnArch,
)
from param_decomp.core.ci_fn.implementations.transformer.component_conditioning import (
    ConditionedCIFnArch,
)
from param_decomp.core.components import SiteC
from param_decomp.core.configs import NeuronRanksRef, PDConfig, TargetedPDConfig
from param_decomp.routed.experts import ExpertImplementation
from param_decomp.targets.llama_simple_mlp import SIMPLE_MLP_ANATOMY
from param_decomp.targets.lm_output import OutputEdge
from param_decomp.targets.qwen36_moe import Qwen36MoeConfig
from param_decomp.targets.transformer import GLU_ANATOMY, Anatomy

WeightsDtype = Literal["float32", "bfloat16"]
ComponentInitialization = Literal[
    "random",
    "zero_u",
    "nonlinearity_aligned",
    "neuron_aligned_targeted",
    "neuron_aligned_zero_u",
]
"""How the subcomponent V/U masters are seeded. `random`: target-blind small random.
`zero_u`: the target-coupled seed's `V` with `U` zeroed, so the component sum is exactly
zero at init and the delta carries all of `W` — a subcomponent acquires norm only as the
reconstruction losses demand it. `nonlinearity_aligned`: every site along its own
architectural coordinates, chosen without data. `neuron_aligned_targeted` (tPD only): the
top-C coordinates by activity on the target prompt pool, from the harvested artifact
`neuron_ranks` names. `neuron_aligned_zero_u` (tPD only): the same targeted coordinates,
but with `U` zeroed — `neuron_aligned_targeted`'s `V` under `zero_u`'s discipline."""

ALIGNED_INITIALIZATIONS: frozenset[ComponentInitialization] = frozenset(
    {"neuron_aligned_targeted", "neuron_aligned_zero_u"}
)
"""The inits that consume the harvested `neuron_ranks` artifact, and the only ones for which
`neuron_ranks` may be authored. Both read the SAME ranking; they differ only in whether `U`
carries the selected coordinate's weights or is zeroed."""


def weights_jnp_dtype(dtype: WeightsDtype) -> DTypeLike:
    """The authored frozen-target dtype as the array dtype the target loaders cast to."""
    match dtype:
        case "float32":
            return jnp.float32
        case "bfloat16":
            return jnp.bfloat16


@dataclass(frozen=True)
class ResolvedLMData:
    """Pre-tokenized parquet shard directories: `dir` trains; `eval_dir` is the held-out
    split the eval pass reads."""

    dir: Path
    eval_dir: Path


@dataclass(frozen=True)
class TargetConfig:
    """An HF GLU-transformer target (`model_name` must be in `HF_MODEL_VARIANTS` —
    Llama-3.1-8B or a registered Qwen3 checkpoint)."""

    model_name: str
    sites: tuple[SiteC, ...]
    """Decomposed sites with per-site C, in canonical order (`canonical_site_cs`)."""
    weights_dtype: WeightsDtype
    """The authored `target.weights_dtype`, carried to the composition root's target load."""
    attention_implementation: AttentionImplementation
    output_edge: OutputEdge
    component_initialization: ComponentInitialization
    neuron_ranks: NeuronRanksRef | None = None
    """The ranking artifact the aligned targeted inits read; `None` for every other init."""

    supported_weights_dtypes: ClassVar[frozenset[WeightsDtype]] = frozenset({"bfloat16", "float32"})
    """Frozen-target weight dtypes the loader supports. `HFWeights` casts every tensor on
    read, so the family loaders honour whichever of the two the config names. A config
    requesting a dtype outside this set is refused at convert time — no silent downgrade
    without silently changing the authored dtype."""


@dataclass(frozen=True)
class LlamaSimpleMLPTargetConfig:
    """The locally pretrained `LlamaSimpleMLP` target (`param_decomp.targets.llama_simple_mlp`);
    weights from the store entry `pretrain_run_path` resolves to
    (`infra.pretrain_cache.resolved_cache_dir`)."""

    pretrain_run_path: str
    sites: tuple[SiteC, ...]
    """Decomposed sites with per-site C, in canonical order
    (`llama_simple_mlp.canonical_site_cs`)."""
    weights_dtype: WeightsDtype
    """The authored `target.weights_dtype`, carried to the composition root's target load."""
    attention_implementation: AttentionImplementation
    output_edge: OutputEdge
    component_initialization: ComponentInitialization
    neuron_ranks: NeuronRanksRef | None = None
    """The ranking artifact the aligned targeted inits read; `None` for every other init."""

    supported_weights_dtypes: ClassVar[frozenset[WeightsDtype]] = frozenset({"bfloat16", "float32"})
    """Frozen-target weight dtypes the loader supports — `_checkpoint_weight_getter` casts
    every safetensor on read. See `TargetConfig.supported_weights_dtypes`."""


@dataclass(frozen=True)
class HFSnapshotWeights:
    """The local HF hub snapshot of `model_name` (`transformer.hf_snapshot_dir`)."""

    model_name: str


@dataclass(frozen=True)
class PretrainCacheWeights:
    """The pretrain-cache entry `pretrain_run_path` resolves to
    (`infra.pretrain_cache.resolved_cache_dir`)."""

    pretrain_run_path: str


Qwen36MoeWeights = HFSnapshotWeights | PretrainCacheWeights
"""Where a qwen36_moe target's frozen weights come from; its `arch` is read from the
same place."""


@dataclass(frozen=True)
class Qwen36MoeTargetConfig:
    """A target on the qwen36_moe engine (`param_decomp.targets.qwen36_moe`): the HF
    Qwen3.6-35B-A3B checkpoint or a lab-pretrained `Qwen35Moe` toy, its `arch` resolved
    once from where `weights` come from. Expert and mixer projections support random
    or nonlinearity-aligned decomposition."""

    arch: Qwen36MoeConfig
    weights: Qwen36MoeWeights
    sites: tuple[SiteC, ...]
    """Decomposed sites with per-site C, in canonical order (whole-grid per kind)."""
    weights_dtype: WeightsDtype
    attention_implementation: AttentionImplementation
    """The full-attention SDPA lowering; unsupported flash execution fails explicitly."""
    output_edge: OutputEdge
    """The model-output edge (`targets.lm_output.OutputEdge`): materialized logits, or
    the factored streamed package whose comparisons chunk the 248k vocab axis."""
    expert_implementation: ExpertImplementation
    """The target's dense masked execution or admitted routed kernel. CI expert
    computation is configured independently."""
    component_initialization: ComponentInitialization

    supported_weights_dtypes: ClassVar[frozenset[WeightsDtype]] = frozenset({"bfloat16", "float32"})
    """See `TargetConfig.supported_weights_dtypes`; `HFWeights` casts every tensor on read."""


DenseTransformerTargetConfig = TargetConfig | LlamaSimpleMLPTargetConfig


def dense_transformer_anatomy(target: DenseTransformerTargetConfig) -> Anatomy:
    """The dense transformer family's site anatomy: what each site reads and writes."""
    match target:
        case TargetConfig():
            return GLU_ANATOMY
        case LlamaSimpleMLPTargetConfig():
            return SIMPLE_MLP_ANATOMY


AnyLMTargetConfig = TargetConfig | LlamaSimpleMLPTargetConfig | Qwen36MoeTargetConfig
"""The closed set of LM target configs — what every LM `BuiltRun` carries and every LM
consumer (`build_target`, `run_metadata`, the targeted tokenizer route) dispatches on.
Non-LM targets (the toys) satisfy only the core `TargetSites` protocol and never enter
the LM aliases below."""


LMCIFnArch = (
    ChunkwiseTransformerCIFnArch
    | GlobalTransformerCIFnArch
    | ConditionedCIFnArch[GlobalTransformerCIFnArch]
    | BlockSelectedChunkwiseTransformerCIFnArch
    | GlobalMLPCIFnArch
)
"""What `LMCIFnConfig` resolves to — the arches an LM run (and its stored-run consumers)
can carry."""

UnroutedLMCIFnArch = (
    ChunkwiseTransformerCIFnArch
    | GlobalTransformerCIFnArch
    | ConditionedCIFnArch[GlobalTransformerCIFnArch]
    | GlobalMLPCIFnArch
)
"""The LM arches whose CI fns read no target routing."""


def require_unrouted_ci_fn_arch(arch: LMCIFnArch) -> UnroutedLMCIFnArch:
    """A document-aware target carries no routing, so it pairs only with an unrouted CI."""
    match arch:
        case (
            ChunkwiseTransformerCIFnArch()
            | GlobalTransformerCIFnArch()
            | ConditionedCIFnArch()
            | GlobalMLPCIFnArch()
        ):
            return arch
        case BlockSelectedChunkwiseTransformerCIFnArch():
            raise ValueError("the block-selected CI reads target routing; this target has none")


LMRun = BuiltRun[ResolvedLMData, AnyLMTargetConfig, PDConfig, LMCIFnArch]
LMTargetedRun = BuiltRun[ResolvedLMData, AnyLMTargetConfig, TargetedPDConfig, LMCIFnArch]
LMAnyRun = LMRun | LMTargetedRun
"""The stored-run consumers' view: the closed union of run shapes — consumers read only
the sections the shapes share."""
