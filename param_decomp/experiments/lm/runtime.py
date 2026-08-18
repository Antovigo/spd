"""The LM run's compute substrate: `RuntimeConfig` (the `runtime:` section of
`LMExperimentConfig`) and the pre-process env surface nested inside it (`launch_env`).

Substrate, not algorithm — every value here perturbs numerics or memory without changing
what is computed, and none of it reaches the engine as an object: the composition root
(`training.py`) unpacks it into the engine's primitives (device counts, a placement spec,
remat flags, compiler options). It is the LM's alone; the single-device toys have no
substrate to author, so their schemas carry no `runtime:` section at all.

Deliberately free of jax and of the rest of the LM schema: `run.py` validates the
`runtime:` block and exports `launch_env` BEFORE importing JAX, so this module must stay
cheap to import.
"""

from typing import Literal, Self

from pydantic import Field, PositiveFloat, PositiveInt, model_validator

from param_decomp.core.base_config import BaseConfig
from param_decomp.core.configs import PlacementTableConfig


class LaunchEnv(BaseConfig):
    """The process-environment surface a rank runs with — the XLA *client* knobs (mem
    fraction / allocator / host-memory limit), NCCL/glibc tuning, and a free-form env escape
    hatch — lifted into the run config so a run's `launch_config.yaml` fully captures its
    environment (tracking + repro), and A/B-ing a knob is a config edit, not a launcher edit.

    XLA *compiler* flags are NOT here — they go through `RuntimeConfig.compiler_options`
    (passed natively to each jit, no env round-trip; see that field). This class is only the
    env that must exist before the process starts (read at backend/NCCL init).

    The pre-JAX bootstrap (`run.py`) exports it before importing JAX; whoever spawns the
    ranks renders the same map into their environment. `LD_LIBRARY_PATH` is NOT here (it is
    machine-specific — resolved against the local CUDA install by whoever starts the
    process — not a tracked decision). These defaults are the single source of truth: a
    submitter renders them, it does not carry its own set.
    """

    xla_python_client_mem_fraction: PositiveFloat = 0.92
    """`XLA_PYTHON_CLIENT_MEM_FRACTION` — the BFC pool cap as a fraction of HBM."""
    xla_python_client_allocator: str | None = None
    """`XLA_PYTHON_CLIENT_ALLOCATOR` — e.g. `platform` for the on-demand cudaMalloc allocator
    (avoids BFC fragmentation OOMs near the HBM cap, at some per-alloc cost). `None` leaves
    the XLA default (BFC)."""
    xla_pjrt_gpu_host_memory_limit_gb: PositiveInt = 1024
    """`XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB` — cap on XLA's pinned host-staging pool
    (allocated on demand)."""
    nccl_debug: str = "WARN"
    """`NCCL_DEBUG` — overrides the INFO + SUBSYS=ALL default some clusters set, which logs
    every collective and bloats a run's logs to tens of GB."""
    malloc_arena_max: PositiveInt = 2
    """`MALLOC_ARENA_MAX` — caps glibc malloc arenas to bound host RSS under many threads."""
    env: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Arbitrary extra exports merged into the rank env LAST (after the typed knobs), "
            "so it can override any of them. The escape hatch for a one-off var without a "
            "schema field."
        ),
    )

    def as_env(self) -> dict[str, str]:
        """Render the ordered `{VAR: value}` map a rank's environment must carry (sans
        `LD_LIBRARY_PATH`, which is machine-specific). Only the env that must exist before
        backend/NCCL init — XLA *compiler* flags are passed natively via
        `RuntimeConfig.compiler_options`, not here. Later keys override earlier, so the
        free-form `env` block wins last."""
        rendered: dict[str, str] = {
            "NCCL_DEBUG": self.nccl_debug,
            "MALLOC_ARENA_MAX": str(self.malloc_arena_max),
            "XLA_PYTHON_CLIENT_MEM_FRACTION": str(self.xla_python_client_mem_fraction),
            "XLA_PJRT_GPU_HOST_MEMORY_LIMIT_GB": str(self.xla_pjrt_gpu_host_memory_limit_gb),
        }
        if self.xla_python_client_allocator is not None:
            rendered["XLA_PYTHON_CLIENT_ALLOCATOR"] = self.xla_python_client_allocator
        rendered |= self.env
        return rendered


class RuntimeConfig(BaseConfig):
    """Compute substrate: world size, placement, rematerialization, XLA compiler flags, and
    the pre-process env surface (`launch_env`).

    Perturbs numerics but doesn't change the algorithm.
    """

    dp: PositiveInt = Field(
        description=(
            "World size — the total device count, THE single source of truth for topology, "
            "NEVER inferred from ambient env (`SLURM_PROCID` is present in every process on "
            "a SLURM box). Process bring-up DERIVES from it: `dp <= gpus_per_node` → ONE "
            "process over exactly `dp` local devices, asserted at startup "
            "(`sharding.assert_inline_topology`) — `dp: 1` is the single-device smoke, "
            "`dp: 8` a run inside an external scheduler's own whole-node job. "
            "`dp > gpus_per_node` → one process per node, brought up via `jax.distributed`'s "
            "own cluster auto-detection (`init_distributed` — the jax ecosystem's contract; "
            "SLURM/MPI/TPU), asserted against the realized `jax.device_count()`. Multiple "
            "processes on one node is deliberately unrepresentable. The batch shards "
            "data-parallel across all `dp` devices."
        ),
    )
    gpus_per_node: PositiveInt = Field(
        default=8,
        description=(
            "GPUs per node — the size of the intra-node NVLink group the mesh's `fsdp*tp` "
            "plane is carved from, and the launcher's node math (`nodes = dp / gpus_per_node`). "
            "A property of the cluster, carried in the config so the pinned launch_config "
            "fully determines the topology. Default 8 (H100/H200/B200 nodes)."
        ),
    )
    fsdp: PositiveInt | None = Field(
        default=None,
        description=(
            "Width of the parameter-sharding plane, in devices. `None` (default) derives it "
            "from `gpus_per_node` — the layout every NVLink node wants, and the only "
            "behaviour any existing run has seen. Set it to pin the plane narrower and put "
            "the remaining devices on `replicate` instead. `fsdp: 1` degenerates the axis "
            "entirely, so every `P(..., 'fsdp', ...)` spec replicates — most consequentially "
            "the frozen target's layer stack, which otherwise gathers one layer's shard per "
            "block per forward. That gather is nearly free on NVLink and dominates the step "
            "without it: `l40-worker` reports CNS (no peer-to-peer) between every GPU pair, "
            "so NCCL routes it over shared memory, and replicating there measured ~4.5x. "
            "Costs the full frozen target resident per device, so it is a capacity decision "
            "as much as a speed one. Same math under either value — layouts differ only by "
            "float reassociation (SPEC D4)."
        ),
    )
    tp: int = Field(
        default=1,
        ge=1,
        le=8,
        description=(
            "Tensor-parallel (Megatron) degree, carved from the intra-node GPUs so "
            "`fsdp * tp = GPUS_PER_NODE` — both stay on NVLink. Shards the component C axis "
            "(V/U, CI-fn output heads) and the CI-fn MLP hidden, halving the per-layer weight "
            "all-gather. `tp = 1` (default) is the pure-HSDP layout (degenerate tp axis, "
            "behaviour-preserving). Must divide both the device count and GPUS_PER_NODE."
        ),
    )
    sequential_passes: bool = Field(
        default=False,
        description=(
            "Score the tPD passes ONE AT A TIME and add their gradients, instead of fusing "
            "them into a single backward (SPEC T1). Identical arithmetic — summing per-pass "
            "gradients is what the fused backward does internally — so this is a memory/speed "
            "trade and nothing else: peak activation memory holds ONE pass's masked forwards "
            "rather than every pass's, at the cost of a params-sized gradient accumulator and "
            "the overlap XLA gives up at each `optimization_barrier`. Worth it when a hidden "
            "pass (T12) doubles the forwards; pointless without one, since a two-pass run "
            "already fits wherever it fit before. At bf16 compute the two paths differ by "
            "backward rounding (~1e-3 on gradient norms, none on the losses)."
        ),
    )
    sharding: Literal["owner", "owner+zero1", "zero1", "ddp"] | PlacementTableConfig = Field(
        description=(
            "Placement policy for the trainable state (placement.py). REQUIRED, no "
            "default — a layout this consequential is written down per config. Presets: "
            "`zero1` = intra-matrix ZeRO-1 over the full data mesh "
            "(~equivalent comms to `owner` under "
            "elementwise optimizers); `owner` = whole-matrix ownership (stack ÷replicate, "
            "d ÷fsdp, C ÷tp) — the muon-motivated layout (Newton-Schulz stays "
            "node-local); STRICT — a shape group whose stack does "
            "not tile ÷replicate is an error; `owner+zero1` = `owner` plus the "
            "`params.zero1` opt-in row, ZeRO-1-ing exactly those non-tiling groups "
            "intra-matrix; `ddp` = fully replicated. Each value is a BIDIRECTIONAL claim "
            "checked at config build (placement.from_config, pre-submission for a submitted run): "
            "`owner` claims every group tiles; `owner+zero1` claims at least one does "
            "not — all-tiling under it is equally an error. Or an explicit "
            "`PlacementTableConfig` table (nested `params: {persist, zero1?, forward}` + "
            "`activations`, each row a semantic-axis -> mesh-axes rule; list order is "
            "semantics). Same math under every value — layouts differ only by float "
            "reassociation (SPEC D4)."
        ),
    )
    remat_recon_forwards: bool = Field(
        default=False,
        description=(
            "JAX trainer memory/compute trade: rematerialize the recon-loss masked "
            "forwards under the full model (deep targets need it to fit). Compute "
            "substrate knob, no algorithm effect."
        ),
    )
    remat_ci_fn: bool = Field(
        default=False,
        description=(
            "JAX trainer memory/compute trade: rematerialize the CI-fn forward "
            "(recompute it in the backward instead of storing its activations). The "
            "CI-fn activations scale with batch, so this is the main lever for larger "
            "batch on big targets. Compute substrate knob, no algorithm effect."
        ),
    )
    ascend_replicate: bool = Field(
        default=False,
        description=(
            "Replicate the ÷fsdp compute weights once before the adversary ascents so the "
            "n_warmup ascend forwards skip the per-layer ÷fsdp→full gather (mask-independent "
            "and detached, so the re-gather is pure redundancy). Numerics-identical. Trades "
            "the full V/U resident during the ascend phase for the eliminated re-gathers."
        ),
    )
    compiler_options: dict[str, bool | int | str] = Field(
        default_factory=lambda: {
            "xla_gpu_enable_latency_hiding_scheduler": True,
            "xla_gpu_enable_triton_gemm": False,
            "xla_gpu_enable_command_buffer": "",
            "xla_gpu_enable_highest_priority_async_stream": True,
            "xla_gpu_all_reduce_combine_threshold_bytes": 1073741824,
            "xla_gpu_all_gather_combine_threshold_bytes": 1073741824,
            "xla_gpu_reduce_scatter_combine_threshold_bytes": 134217728,
            "xla_gpu_enable_pipelined_all_gather": True,
            "xla_gpu_enable_pipelined_reduce_scatter": True,
            "xla_gpu_enable_pipelined_all_reduce": True,
            "xla_gpu_enable_while_loop_double_buffering": True,
            "xla_gpu_enable_all_gather_combine_by_dim": False,
            "xla_gpu_enable_reduce_scatter_combine_by_dim": False,
        },
        description=(
            "XLA compiler flags passed NATIVELY to every jit's `compiler_options` — no "
            "`XLA_FLAGS` env round-trip, and (unlike env) they ARE in the compile-cache key, "
            "so changing one actually recompiles. Full `xla_*` flag names, typed values "
            "(True/int/str, not 'true'). Default = the tuned MaxText set (latency-hiding "
            "scheduler + 1 GiB combine thresholds + pipelined collectives + double-buffering; "
            "`command_buffer:''` disables CUDA-graph capture, a correctness guard). Add "
            "`xla_disable_hlo_passes: rematerialization` to opt into the disable-XLA-remat win "
            "(validate save/resume first). On CPU (toys/tests) the GPU flags are ignored."
        ),
    )
    launch_env: LaunchEnv = Field(default_factory=LaunchEnv)
    """The pre-process env each rank runs with (XLA *client* / NCCL / glibc knobs — the env
    that must exist before backend init; NOT compiler flags, which go via
    `compiler_options`). Applied by the bootstrap in the process it starts, and rendered into
    the rank environment by whoever spawns the ranks; everything else about that environment
    is inherited from the caller."""

    @property
    def distributed(self) -> bool:
        """Derived, never authored: a world larger than one node is multi-process (one per
        node); anything else is one process over `dp` local devices."""
        return self.dp > self.gpus_per_node

    @model_validator(mode="after")
    def validate_topology(self) -> Self:
        if self.dp > self.gpus_per_node:
            assert self.dp % self.gpus_per_node == 0, (
                f"a multi-node world allocates whole {self.gpus_per_node}-GPU nodes — "
                f"dp={self.dp} must be a multiple of gpus_per_node={self.gpus_per_node} "
                f"(a sub-node world runs as one process inside an existing allocation)"
            )
        if self.fsdp is not None:
            plane = self.fsdp * self.tp
            assert self.dp % plane == 0, (
                f"dp={self.dp} must be a multiple of the pinned fsdp × tp plane "
                f"({self.fsdp} × {self.tp} = {plane})"
            )
        return self
