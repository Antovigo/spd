"""`FsdpRuntimeConfig`: the core `RuntimeConfig` plus FSDP2 + torch.compile knobs."""

from pydantic import Field

from param_decomp.configs import RuntimeConfig


class FsdpRuntimeConfig(RuntimeConfig):
    """Compute substrate for the single-pool FSDP2 LM path: core runtime + FSDP/compile knobs."""

    compile_model: bool = Field(
        default=True,
        description="torch.compile the vendored target model (masked forward).",
    )
    compile_ci_fn: bool = Field(
        default=True,
        description="torch.compile the causal-importance function.",
    )
    checkpoint_blocks: bool = Field(
        default=True,
        description="Per-block activation checkpointing on the target model.",
    )
    shard_frozen_target: bool = Field(
        default=True,
        description="Convert frozen target buffers to no-grad params so FSDP2 shards the target "
        "(else they are replicated on every rank).",
    )
