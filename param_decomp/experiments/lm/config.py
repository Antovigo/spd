"""LM experiment config schema (target spec, data settings, tiled site specs, full YAML
tree) PLUS the LM YAML→`BuiltRun` conversion.

This module reads the canonical `LMExperimentConfig` schema directly and builds the engine's
`BuiltRun` bundle (`param_decomp.core.built_run`) — the pydantic `pd` / `cadence`
verbatim plus the resolved target / data / CI-fn arch / eval — asserting loudly on anything
the JAX trainer doesn't implement. The composition entry (`run.py`) calls `load_config` /
`build_from_schema`; stored-run consumers rebuild the same canonical schema.

The authored `decomposition.sites` c-specs (`GluTransformerCSpec` / `SimpleMlpCSpec`, keys
typed by each target family's own matrix vocabulary) resolve here into the block-structured
`SiteTree` (`resolve_site_tree`) — the layer index carried as DATA, never parsed back out
of a site name — which the chunkwise CI resolver consumes directly.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self

import yaml
from pydantic import (
    Discriminator,
    Field,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)

from param_decomp.core import placement
from param_decomp.core.base_config import BaseConfig
from param_decomp.core.built_run import BuiltRun
from param_decomp.core.ci_fn import (
    Chunk,
    ChunkwiseTransformerCIArch,
    GQACIAttention,
    MHACIAttention,
)
from param_decomp.core.components import SiteC, SiteSpec
from param_decomp.core.configs import ResumeProvenance
from param_decomp.core.family import ArchFamily
from param_decomp.core.objective import build_objective
from param_decomp.core.sharding import hsdp_abstract_mesh
from param_decomp.experiments.config import (
    ExperimentConfig,
    run_instance,
)
from param_decomp.experiments.eval_config import EvalConfig
from param_decomp.experiments.lm.resolved import (
    LMRun,
    ResolvedLMData,
    WeightsDtype,
)
from param_decomp.experiments.lm.runtime import RuntimeConfig
from param_decomp.infra import pretrain_cache
from param_decomp.infra.dataset_store import DatasetRef, resolve_dataset_ref
from param_decomp.migrations.schedule_knots import migrate_raw as migrate_schedule_knots
from param_decomp.targets import glu_transformer, llama8b, llama_simple_mlp, qwen3_8b
from param_decomp.targets.glu_transformer import GluMatrix
from param_decomp.targets.llama_simple_mlp import SimpleMlpMatrix
from param_decomp.targets.transformer_taps import TransformerTapGrammar, resid_tap_key


class HFTarget(BaseConfig):
    """Load a HuggingFace model via `<model_class>.from_pretrained(<model_name>)`."""

    kind: Literal["hf"] = "hf"
    model_class: str
    model_name: str


class PretrainedTarget(BaseConfig):
    """Load an in-repo lab-pretrained model (`param_decomp.pretrain.train`'s output)."""

    kind: Literal["pretrained"] = "pretrained"
    model_class: str
    run_path: str
    """`entity/project[/runs]/run_id` — the W&B pretrain run whose checkpoint is the
    target's weights. A name, never a location: it resolves to the local store entry
    `<data_root>/pretrain_cache/<project>-<run_id>`, fetched from W&B on first use if
    not already there (`infra.pretrain_cache`), read from disk ever after."""


class HFWeightsInVendored(BaseConfig):
    """Load HF pretrained weights into the vendored `VendoredLlama` architecture.

    Llama-3.1-8B only — `_resolve_decomposition` asserts the class and model name;
    other HF families go through `kind: hf`.
    """

    kind: Literal["hf_weights_in_vendored"] = "hf_weights_in_vendored"
    model_class: str  # must be `VendoredLlama`
    model_name: str  # HF hub id


LMTargetSpec = Annotated[
    HFTarget | PretrainedTarget | HFWeightsInVendored,
    Discriminator("kind"),
]


class LMTargetConfig(BaseConfig):
    """Config for the LM target model."""

    spec: LMTargetSpec
    weights_dtype: WeightsDtype
    """dtype for the FROZEN target weights. Only the frozen target is cast; trained V/U
    components keep their fp32 AdamW master.

    `bfloat16` halves the target's resident footprint on every pool — the dominant resident
    term for an 8B target — and for a natively-bf16 checkpoint costs nothing beyond
    residual/norm accumulation precision.

    `float32` is a genuine option but not a free one: the masked forward promotes where
    fp32 frozen weights meet the bf16 compute V/U, so the whole recon forward runs fp32.
    That is ~2x the activation memory, and it drops off cuDNN flash attention
    (`vendored_jax.llama.attn_implementation` selects it only for the half precisions), so
    the [B, H, T, T] scores materialize. A reference-run dtype, not a production one.

    Deliberately has no default: it is the largest single memory decision in the config,
    and a stored config that omitted it could not say how its run was trained."""


class LMDataConfig(BaseConfig):
    """The run's data: `train` feeds the trainer; `eval` is the held-out split the eval
    pass reads. Each ref's facts (seq_len, tokenizer) ride with its shards as `meta.json`
    (`param_decomp.infra.dataset_store`), read at load."""

    train: DatasetRef
    eval: DatasetRef


class AllLayers(BaseConfig):
    kind: Literal["all"] = "all"


class LayerRange(BaseConfig):
    """Half-open `[start, end)` — matches `range()` / slice semantics."""

    kind: Literal["range"] = "range"
    start: NonNegativeInt
    end: PositiveInt

    @model_validator(mode="after")
    def _nonempty(self) -> Self:
        assert self.start < self.end, (self.start, self.end)
        return self


class LayerList(BaseConfig):
    kind: Literal["list"] = "list"
    indices: list[NonNegativeInt] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _unique_sorted(self) -> Self:
        assert self.indices == sorted(set(self.indices)), self.indices
        return self


LayerSelection = Annotated[AllLayers | LayerRange | LayerList, Field(discriminator="kind")]


class GluTransformerCSpec(BaseConfig):
    """Per-matrix-type C tiled across the selected layers (GLU family, e.g. llama8b). Every
    selected layer is decomposed at the same `cs` matrices and C; a matrix absent from `cs`
    is not decomposed on any layer. Tiled ⇒ every block is structurally identical, so the
    chunkwise CI fn's chunks are homogeneous by construction."""

    kind: Literal["glu_transformer"] = "glu_transformer"
    layers: LayerSelection
    cs: dict[GluMatrix, PositiveInt] = Field(..., min_length=1)


class SimpleMlpCSpec(BaseConfig):
    """Per-matrix-type C tiled across the selected layers (plain-GELU family, LlamaSimpleMLP)."""

    kind: Literal["simple_mlp"] = "simple_mlp"
    layers: LayerSelection
    cs: dict[SimpleMlpMatrix, PositiveInt] = Field(..., min_length=1)


@dataclass(frozen=True)
class BlockSites:
    """One transformer block's decomposed matrices, in canonical within-block (family) order.
    `layer_idx` is a field — the whole point is that structure is never thrown into a string."""

    layer_idx: int
    slots: tuple[tuple[str, int], ...]  # (matrix_type, C)


@dataclass(frozen=True)
class SiteTree:
    """A decomposition as blocks, strictly layer-ascending — the layer index is carried as
    DATA, never parsed back out of a site name. The tiled site specs resolve INTO it, and
    the chunkwise CI resolver consumes it directly (a chunk = a slice of consecutive
    `BlockSites`), so nothing downstream recovers block structure by regex-ing site-name
    strings. The flat `SiteC` view is DERIVED via the family's name grammar —
    construction only, no inverse parse."""

    blocks: tuple[BlockSites, ...]

    def site_cs(self, name_of: Callable[[int, str], str]) -> tuple[SiteC, ...]:
        return tuple(
            SiteC(name_of(b.layer_idx, kind), c) for b in self.blocks for kind, c in b.slots
        )


def _select_layers(sel: LayerSelection, n_layer: int) -> tuple[int, ...]:
    match sel:
        case AllLayers():
            return tuple(range(n_layer))
        case LayerRange(start=start, end=end):
            assert end <= n_layer, f"layer range end {end} exceeds n_layer {n_layer}"
            return tuple(range(start, end))
        case LayerList(indices=indices):
            assert indices[-1] < n_layer, f"layer {indices[-1]} exceeds n_layer {n_layer}"
            return tuple(indices)


def resolve_site_tree(
    sites: GluTransformerCSpec | SimpleMlpCSpec, family: ArchFamily, n_layer: int
) -> SiteTree:
    """Tile the per-matrix-type `cs` across the selected layers into a `SiteTree`. Every block
    shares ONE `slots` tuple (canonical family order, only the requested matrices), so the tree
    is homogeneous by construction — which is exactly what makes the chunkwise CI fn's chunks
    homogeneous. Asserts the spec's declared family matches the target's."""
    assert sites.kind == family.key, f"c-spec family {sites.kind!r} != target family {family.key!r}"
    # cs keys are Literal-typed by the family vocabulary `matrices` derives from, so every
    # key is a family matrix by construction — ordering is the only work left.
    rank = {matrix: i for i, matrix in enumerate(family.matrices)}
    slots = tuple(sorted(sites.cs.items(), key=lambda slot: rank[slot[0]]))
    layers = _select_layers(sites.layers, n_layer)
    return SiteTree(tuple(BlockSites(layer, slots) for layer in layers))


ChunkInputTap = Literal["first_block_resid", "all_block_resids", "all_site_inputs"]
"""Which activations each chunkwise-CI chunk reads. Extend here + add a match arm in
`_chunk_input_taps` below; the concrete tap keys and their widths are the family tap
grammar's (`param_decomp.targets.transformer_taps`) — opaque strings everywhere generic."""


class MHACiAttentionConfig(BaseConfig):
    """Every query head carries its own K/V head."""

    kind: Literal["mha"] = "mha"
    n_heads: PositiveInt


class GQACiAttentionConfig(BaseConfig):
    """Grouped-query attention: `n_heads // n_kv_heads` query heads share each K/V head, so
    `wk`/`wv` narrow to `n_kv_heads * head_dim`. head_dim, the RoPE tables, `wq`/`wo` and
    every sharding are identical to MHA — only the K/V projections change."""

    kind: Literal["gqa"] = "gqa"
    n_heads: PositiveInt
    n_kv_heads: PositiveInt

    @model_validator(mode="after")
    def validate_grouping(self) -> Self:
        assert self.n_heads % self.n_kv_heads == 0, (
            "n_heads must be divisible by n_kv_heads (each K/V head serves an equal group "
            f"of query heads): {self.n_heads} % {self.n_kv_heads}"
        )
        assert self.n_kv_heads < self.n_heads, (
            f"n_kv_heads == n_heads ({self.n_heads}) is MHA — use `kind: mha` rather than "
            "spelling it as a degenerate gqa"
        )
        return self


class GeluCiFfnConfig(BaseConfig):
    """`Linear+b -> GELU -> Linear+b` — two matrices."""

    kind: Literal["gelu"] = "gelu"
    hidden: PositiveInt


class SwigluCiFfnConfig(BaseConfig):
    """`silu(h@w_gate + b_gate) * (h@w1 + b1) -> Linear+b` — THREE matrices, so at a given
    `hidden` this is ~1.5x the GELU FFN's params. Iso-param is `hidden` at 2/3 (Shazeer's GLU
    variants: "decrease d_ff by a factor of 2/3"); nothing here rescales it, because a width
    that silently differs from the one you wrote is worse than doing the arithmetic."""

    kind: Literal["swiglu"] = "swiglu"
    hidden: PositiveInt


CiFfnConfig = Annotated[GeluCiFfnConfig | SwigluCiFfnConfig, Field(discriminator="kind")]
"""The CI transformer's feed-forward sublayer. Named FFN, not MLP: `swiglu` is a gated
linear unit, not a multi-layer perceptron — FFN is the name that stays honest across both
arms. `hidden` belongs to the FFN, not to the transformer around it."""


CiAttentionConfig = Annotated[
    MHACiAttentionConfig | GQACiAttentionConfig, Field(discriminator="kind")
]
"""The CI transformer's attention, keyed by CLASS rather than an optional `n_kv_heads` —
so a K/V head count cannot exist without meaning, and the grouping invariant lives on the
arm that has both fields instead of being a runtime check on a shape that shouldn't parse.
Mirrors how the target's attention variants are keyed (see the Qwen3 family split)."""


class ChunkwiseTransformerCiConfig(BaseConfig):
    """Chunkwise-transformer CI fn (LMs). Each chunk is `blocks_per_chunk` consecutive
    transformer blocks; `input_tap` names which activations the chunk reads and its output
    is CI for every matrix site in those blocks. `d_model`/`n_blocks`/`attention`/`ffn`
    size the per-chunk CI transformer (`d_model % n_heads == 0`; head_dim even for RoPE)."""

    type: Literal["chunkwise_transformer"] = "chunkwise_transformer"
    blocks_per_chunk: PositiveInt
    input_tap: ChunkInputTap = "first_block_resid"
    """`first_block_resid`: the residual stream entering the chunk's first block — one tap.
    `all_block_resids`: the residual entering EVERY block the chunk runs over, RMS-normed
    per tap and concatenated (`ci_fn.Chunk.input_taps` is generic over tap count) —
    `blocks_per_chunk`x the per-chunk CI transformer's input width.
    `all_site_inputs`: the input activation of every decomposed site in the chunk (the
    paper's D = Σd_l input set) — widths are per-site d_in, NOT d_resid (down/o
    projections read intermediates), summed into `input_dim`."""
    d_model: PositiveInt
    n_blocks: PositiveInt
    attention: CiAttentionConfig
    ffn: CiFfnConfig
    learned_norm_scale: bool = Field(
        default=False,
        description="Learned per-channel scale on the block RMSNorms (the per-tap input norms "
        "stay weightless). Inits to ones, so step 0 is identical to weightless.",
    )

    @model_validator(mode="after")
    def validate_head_dim(self) -> Self:
        n_heads = self.attention.n_heads
        assert self.d_model % n_heads == 0, (self.d_model, n_heads)
        assert (self.d_model // n_heads) % 2 == 0, "head_dim must be even for RoPE"
        return self


class LMDecompositionConfig(BaseConfig):
    """The LM decomposition apparatus: a tiled site-spec (per-matrix-type C over a layer
    selection) + the chunkwise-transformer CI-fn arch. Tiled-only ⇒ every block is
    structurally identical ⇒ chunkwise chunks are homogeneous by construction (no `explicit`
    variant here, so a non-compiling heterogeneous decomposition is unrepresentable). The
    `sites.kind` family (glu vs simple-MLP) is checked against the target family at resolve."""

    sites: Annotated[GluTransformerCSpec | SimpleMlpCSpec, Discriminator("kind")]
    ci: ChunkwiseTransformerCiConfig


class LMExperimentConfig(ExperimentConfig):
    runtime: RuntimeConfig
    """The LM's compute substrate. Declared HERE, not on the shared base: an LM run is the
    only domain that spans devices, nodes and a launcher, so it is the only one with a
    world size, a placement policy, remat trades and an XLA-flag surface to author."""

    eval: EvalConfig | None = None
    resume_provenance: ResumeProvenance | None = None
    target: LMTargetConfig
    decomposition: LMDecompositionConfig
    data: LMDataConfig


@dataclass(frozen=True)
class HFModelFamily:
    """One vendored HF model family the LM composition can target: its arch config, its
    HF loader (the family file's `load_decomposed_*_from_hf`), and the path-schema model
    type consumers key on. The families live in `param_decomp/targets/{llama8b,qwen3_8b}.py`
    over the shared `glu_transformer` machinery; this registry is the ONLY place a model
    name selects a family."""

    arch_config: Callable[[], glu_transformer.GLUArch]
    load: Callable[..., glu_transformer.GLUDecomposedModel]
    """`(model_name, cfg, sites, weights_dtype)` — cfg is the family's own arch-config
    type, so the common signature is erased here."""
    model_type: str
    model_class: str
    """The `target.spec.model_class` this family answers to (a stable identifier, never
    imported — see experiments/CLAUDE.md)."""


HF_MODEL_FAMILIES: dict[str, HFModelFamily] = {
    "meta-llama/Llama-3.1-8B": HFModelFamily(
        llama8b.llama31_8b_config,
        llama8b.load_decomposed_llama_from_hf,
        "Llama",
        "transformers.LlamaForCausalLM",
    ),
    "Qwen/Qwen3-8B-Base": HFModelFamily(
        qwen3_8b.qwen3_8b_config,
        qwen3_8b.load_decomposed_qwen3_from_hf,
        "Qwen3",
        "transformers.Qwen3ForCausalLM",
    ),
}
"""The HF model names the LM composition implements. Anything else refuses loudly at
convert time — a new model gets an explicit family entry (config checked against its HF
config.json), never a silent guess."""


def hf_model_family(model_name: str) -> HFModelFamily:
    assert model_name in HF_MODEL_FAMILIES, (
        f"no vendored model family for {model_name!r}; supported: {sorted(HF_MODEL_FAMILIES)}"
    )
    return HF_MODEL_FAMILIES[model_name]


@dataclass(frozen=True)
class TargetConfig:
    """An HF GLU-transformer target (`model_name` must be in `HF_MODEL_FAMILIES` —
    Llama-3.1-8B or Qwen3-8B-Base)."""

    model_name: str
    sites: tuple[SiteC, ...]
    """Decomposed sites with per-site C, in canonical order (`canonical_site_cs`)."""
    weights_dtype: WeightsDtype
    """The authored `target.weights_dtype`, carried to the composition root's target load."""

    supported_weights_dtypes: ClassVar[frozenset[WeightsDtype]] = frozenset({"bfloat16", "float32"})
    """Frozen-target weight dtypes the loader supports. `HFWeights` casts every tensor on
    read, so the family loaders honour whichever of the two the config names. A config
    requesting a dtype outside this set is refused at convert time — no silent downgrade
    (issue #727)."""


@dataclass(frozen=True)
class LlamaSimpleMLPTargetConfig:
    """The `LlamaSimpleMLP` lab-pretrained target (`param_decomp.core.llama_simple_mlp`);
    weights from the store entry `pretrain_run_path` resolves to
    (`infra.pretrain_cache.resolved_cache_dir`)."""

    pretrain_run_path: str
    sites: tuple[SiteC, ...]
    """Decomposed sites with per-site C, in canonical order
    (`llama_simple_mlp.canonical_site_cs`)."""
    weights_dtype: WeightsDtype
    """The authored `target.weights_dtype`, carried to the composition root's target load."""

    supported_weights_dtypes: ClassVar[frozenset[WeightsDtype]] = frozenset({"bfloat16", "float32"})
    """Frozen-target weight dtypes the loader supports — `_checkpoint_weight_getter` casts
    every safetensor on read. See `TargetConfig.supported_weights_dtypes`."""


AnyLMTargetConfig = TargetConfig | LlamaSimpleMLPTargetConfig
"""The LM target configs the LM composition builds. Non-LM targets (the toys) live in the
lab and satisfy `param_decomp.core.built_run.TargetSites` — the core `BuiltRun.target` is typed by
that protocol, never by a closed union, so it accepts a lab target config too."""


@dataclass(frozen=True)
class _ResolvedDecomposition:
    """Target config + its block-structured `SiteTree` + arch family, resolved once and shared
    by the target's flat `.sites`, the chunkwise chunk generator, and validation. `grammar`
    is the family tap grammar bound to this target's shape (block range, residual width,
    per-site d_in) — what chunk tap keys and widths resolve against. `site_specs` are the
    shape-carrying specs (built by the same per-family builder the composition root's target
    load uses), the placement gate's input."""

    target: AnyLMTargetConfig
    tree: SiteTree
    grammar: TransformerTapGrammar
    site_specs: tuple[SiteSpec, ...]


def _bound_grammar(
    family: ArchFamily, n_layer: int, d_resid: int, dims_of: Callable[[str], tuple[int, int]]
) -> TransformerTapGrammar:
    """The family tap grammar bound to a resolved target's shape; `dims_of(kind)` is the
    target's per-matrix `(d_in, d_out)` shape table, closed over its arch config."""
    return TransformerTapGrammar(
        family, n_layer, d_resid, d_in_of=lambda name: dims_of(family.parse(name)[1])[0]
    )


def _resolve_decomposition(
    target_config: LMTargetConfig, decomposition: LMDecompositionConfig, data_root: Path
) -> _ResolvedDecomposition:
    """Target spec + tiled `decomposition.sites` -> target config + `SiteTree`.

    HF specs resolve their family from `HF_MODEL_FAMILIES` (all GLU-transformer targets); `kind:
    pretrained` LlamaSimpleMLP specs map to the pretrain-cache loader (plain-MLP family). The
    tree is tiled from the per-matrix-type `cs` over the selected layers; `resolve_site_tree`
    asserts the c-spec's declared family matches the target's."""
    spec = target_config.spec
    sites = decomposition.sites
    match spec:
        case HFWeightsInVendored() | HFTarget():
            match spec:
                case HFWeightsInVendored():
                    assert spec.model_class.rsplit(".", 1)[-1] == "VendoredLlama", spec.model_class
                    assert "Llama-3.1-8B" in spec.model_name, spec.model_name
                case HFTarget():
                    known_classes = {f.model_class for f in HF_MODEL_FAMILIES.values()}
                    assert spec.model_class in known_classes, spec.model_class
                    assert spec.model_class == hf_model_family(spec.model_name).model_class, (
                        f"{spec.model_class!r} is not {spec.model_name!r}'s family"
                    )
            hf_family = hf_model_family(spec.model_name)  # refuses unknown model names
            arch = hf_family.arch_config()
            tree = resolve_site_tree(sites, glu_transformer.FAMILY, arch.n_layer)
            target = TargetConfig(
                model_name=spec.model_name,
                sites=tree.site_cs(glu_transformer.FAMILY.name_of),
                weights_dtype=target_config.weights_dtype,
            )
            grammar = _bound_grammar(
                glu_transformer.FAMILY,
                arch.n_layer,
                arch.n_embd,
                lambda kind: glu_transformer.site_dims(arch, kind),
            )
            site_specs = glu_transformer.glu_site_specs(arch, target.sites)
            return _ResolvedDecomposition(target, tree, grammar, site_specs)
        case PretrainedTarget():
            assert spec.model_class.rsplit(".", 1)[-1] == "LlamaSimpleMLP", spec.model_class
            cache_dir = pretrain_cache.resolved_cache_dir(data_root, spec.run_path)
            arch = llama_simple_mlp.load_model_config(cache_dir)
            tree = resolve_site_tree(sites, llama_simple_mlp.FAMILY, arch.n_layer)
            target = LlamaSimpleMLPTargetConfig(
                pretrain_run_path=spec.run_path,
                sites=tree.site_cs(llama_simple_mlp.FAMILY.name_of),
                weights_dtype=target_config.weights_dtype,
            )
            grammar = _bound_grammar(
                llama_simple_mlp.FAMILY,
                arch.n_layer,
                arch.n_embd,
                lambda kind: llama_simple_mlp.site_dims(arch, kind),
            )
            site_specs = llama_simple_mlp.site_specs(arch, target.sites)
            return _ResolvedDecomposition(target, tree, grammar, site_specs)


def _chunk_input_taps(
    input_tap: ChunkInputTap, blocks: tuple[BlockSites, ...], family: ArchFamily
) -> tuple[str, ...]:
    """The tap keys a chunk reads, from its config source + the blocks it spans."""
    match input_tap:
        case "first_block_resid":
            return (resid_tap_key(blocks[0].layer_idx),)
        case "all_block_resids":
            return tuple(resid_tap_key(b.layer_idx) for b in blocks)
        case "all_site_inputs":
            return tuple(family.name_of(b.layer_idx, kind) for b in blocks for kind, _ in b.slots)


def _resolved_chunks(
    tree: SiteTree, blocks_per_chunk: int, input_tap: ChunkInputTap, family: ArchFamily
) -> tuple[Chunk, ...]:
    """Partition the site tree's blocks into consecutive `blocks_per_chunk`-block chunks. The
    tree IS the block grouping (layer-ascending, already grouped), so there is no name parsing
    and no groupby: a chunk reads the taps `input_tap` selects and emits CI for every slot in
    its blocks, in tree order."""
    blocks = tree.blocks
    assert len(blocks) % blocks_per_chunk == 0, (
        f"{len(blocks)} decomposed blocks not divisible by blocks_per_chunk={blocks_per_chunk}"
    )
    chunks = []
    for start in range(0, len(blocks), blocks_per_chunk):
        group = blocks[start : start + blocks_per_chunk]
        output_sites = tuple(
            family.name_of(block.layer_idx, kind) for block in group for kind, _ in block.slots
        )
        chunks.append(
            Chunk(
                input_taps=_chunk_input_taps(input_tap, group, family),
                output_sites=output_sites,
            )
        )
    return tuple(chunks)


def _resolve_chunkwise_ci_arch(
    tree: SiteTree,
    ci: ChunkwiseTransformerCiConfig,
    grammar: TransformerTapGrammar,
) -> ChunkwiseTransformerCIArch:
    """Resolve the chunkwise-transformer arch from the site tree: the chunk generator
    (`_resolved_chunks`) + the per-chunk input width (the sum of the chunk's tap widths —
    `grammar.width_of`). The tree is homogeneous by construction, so the first chunk's
    width is every chunk's. The `attention` union collapses here to two concrete head
    counts — MHA is `n_kv_heads == n_heads` — so nothing downstream re-derives the
    grouping, and the fine-tune `parent.ci_fn == built.ci_fn` compare sees concrete
    values on both sides."""
    family = grammar.family
    first_chunk_taps = _chunk_input_taps(ci.input_tap, tree.blocks[: ci.blocks_per_chunk], family)
    input_dim = sum(grammar.width_of(key) for key in first_chunk_taps)
    match ci.attention:
        case MHACiAttentionConfig():
            attention = MHACIAttention(n_heads=ci.attention.n_heads)
        case GQACiAttentionConfig():
            attention = GQACIAttention(
                n_heads=ci.attention.n_heads, n_kv_heads=ci.attention.n_kv_heads
            )
    return ChunkwiseTransformerCIArch(
        chunks=_resolved_chunks(tree, ci.blocks_per_chunk, ci.input_tap, family),
        input_dim=input_dim,
        d_model=ci.d_model,
        n_blocks=ci.n_blocks,
        attention=attention,
        ffn_hidden=ci.ffn.hidden,
        ffn_kind=ci.ffn.kind,
        learned_norm_scale=ci.learned_norm_scale,
    )


def _assert_losses_supported(cfg: LMExperimentConfig, site_names: tuple[str, ...]) -> None:
    """Run the schema's loss configs through `build_objective` so unsupported metrics
    refuse at convert time rather than on the GPUs. The engine reads `pd.loss_metrics`
    verbatim (yaml order is RNG-load-bearing), so nothing is returned."""
    build_objective(cfg.pd.loss_metrics, site_names)


def _data(cfg: LMExperimentConfig, data_root: Path) -> ResolvedLMData:
    shard_dir = resolve_dataset_ref(cfg.data.train, data_root)
    eval_dir = resolve_dataset_ref(cfg.data.eval, data_root)
    assert eval_dir != shard_dir, (
        f"data.eval resolves to the training shard dir ({shard_dir}) — not a holdout"
    )
    return ResolvedLMData(dir=shard_dir, eval_dir=eval_dir)


def _assert_supported_weights_dtype(target: AnyLMTargetConfig) -> None:
    """Refuse a frozen-target weights_dtype the target's loader can't honour (issue #727:
    no silent downgrade). Every build route passes through here, train and consume alike —
    the loaders read `weights_dtype` on both, so a dtype accepted at submit and ignored at
    reload would be exactly the divergence #727 is about."""
    assert target.weights_dtype in target.supported_weights_dtypes, (
        f"target {type(target).__name__} supports frozen-target weights_dtype "
        f"{sorted(target.supported_weights_dtypes)}, config asks for "
        f"{target.weights_dtype!r}. No silent downgrade (issue #727): declare a "
        f"supported dtype in the yaml."
    )


def _assert_placement_claims(resolved: _ResolvedDecomposition, cfg: LMExperimentConfig) -> None:
    """The config-build placement gate (SPEC D4 amendment 2026-07-21): construct the
    run's `PlacementRules` at the mesh shape `runtime.{dp,tp}` implies, firing the
    per-shape-group persist-vs-zero1 assignment and the sharding spec's bidirectional
    claim where the resolved site set and the declared topology first coexist — at
    pre-submit validation and at every in-job /
    consumer config build. The composition root's `placement.from_config` at the
    concrete mesh is the same construction; nothing decides later or deeper."""
    placement.from_config(
        cfg.runtime.sharding,
        hsdp_abstract_mesh(cfg.runtime.dp, cfg.runtime.tp, cfg.runtime.gpus_per_node),
        resolved.site_specs,
    )


def assert_placement_claims(cfg: LMExperimentConfig, data_root: Path) -> None:
    """Standalone spelling of the placement gate for submitters' pre-submit validation and
    the repo-config parse gate; `build_experiment_config` runs it on every build."""
    _assert_placement_claims(_resolve_decomposition(cfg.target, cfg.decomposition, data_root), cfg)


def build_experiment_config(cfg: LMExperimentConfig, run_id: str, data_root: Path) -> LMRun:
    resolved = _resolve_decomposition(cfg.target, cfg.decomposition, data_root)
    target = resolved.target
    _assert_losses_supported(cfg, tuple(sc.name for sc in target.sites))
    _assert_supported_weights_dtype(target)
    _assert_placement_claims(resolved, cfg)
    data = _data(cfg, data_root)
    ci_fn = _resolve_chunkwise_ci_arch(resolved.tree, cfg.decomposition.ci, resolved.grammar)

    return BuiltRun(
        pd=cfg.pd,
        cadence=cfg.cadence,
        run=run_instance(cfg, run_id, data_root, cfg.resume_provenance),
        target=target,
        data=data,
        ci_fn=ci_fn,
    )


def build_from_schema(
    schema_raw: dict[str, Any],
    run_id: str,
    data_root: Path,
) -> tuple[LMRun, LMExperimentConfig]:
    """Validate a single self-contained LM run config (the canonical `LMExperimentConfig`
    schema) and convert it to the engine's `BuiltRun` bundle. `run_id` is the minted run
    identity (the launcher's CLI arg, or the run-dir name when reloading a finished run).

    The authored config comes back alongside the bundle: `runtime` lives there, and the
    composition root threads it into the engine explicitly (the bundle is core's, and core
    reads no substrate).

    The LM composition entry (`run.py`) is LM-only. The toy domains (TMS, ResidMLP) build
    their `BuiltRun` in their own `run.py` via the public shared helpers
    (`run_instance`, `ci_arch`)."""
    cfg = LMExperimentConfig.model_validate(schema_raw)
    return build_experiment_config(cfg, run_id, data_root), cfg


def load_config(
    config_path: Path, run_id: str, data_root: Path
) -> tuple[LMRun, LMExperimentConfig]:
    """Parse one pinned LM run YAML into its built run and authored config.

    The stored-config boundary converts the retired warmup/decay schedule shape in
    memory before canonical validation. The pin stays byte-immutable, while pre-knot
    runs remain loadable, resumable, and usable as fine-tune parents.
    """
    raw = yaml.safe_load(config_path.read_text())
    return build_from_schema(migrate_schedule_knots(raw), run_id, data_root)
