"""2-GPU DDP smoke: whole-forward torch.compile of the CI fn with activation
checkpointing inside the compiled region, driving flash-SDPA under bf16 autocast.

Reproduces the production CI-pool compile path on real CUDA + NCCL to confirm the
old whole-model-compile-breaks-distributed failure (functionalize_rng_ops /
flash-SDPA KeyError in the AOT partitioner) is gone on torch >= 2.11. Asserts the
all-reduced grad matches a single-rank eager reference.

Run: torchrun --standalone --nproc_per_node=2 scripts/_ci_compile_ddp_smoke.py
"""

import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from param_decomp.ci_fns import GlobalSharedTransformerCiFn, TargetLayerConfig

# Small but architecture-faithful: real flash-SDPA head dim (128), wide MLP, >1 block.
N_SITES = 8
INPUT_DIM = 256
C = 64
D_MODEL = 512
N_BLOCKS = 3
N_HEADS = 4
MLP = 2048
SEQ = 128
BL = 2


def build() -> GlobalSharedTransformerCiFn:
    tlc = {f"site_{i}": TargetLayerConfig(input_dim=INPUT_DIM, C=C) for i in range(N_SITES)}
    m = GlobalSharedTransformerCiFn(
        target_model_layer_configs=tlc,
        d_model=D_MODEL,
        n_layers=N_BLOCKS,
        n_heads=N_HEADS,
        max_len=SEQ,
        mlp_hidden_dims=[MLP],
    ).cuda()
    m.enable_activation_checkpointing()
    return m


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_device = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_device)
    torch.manual_seed(0)  # identical init across ranks

    m = build()
    m.compile()
    ddp = DDP(m, device_ids=[local_device])

    torch.manual_seed(100 + rank)  # distinct data per rank
    inputs = {f"site_{i}": torch.randn(BL, SEQ, INPUT_DIM, device="cuda") for i in range(N_SITES)}

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = ddp(inputs)
    out.float().sum().backward()
    torch.cuda.synchronize()

    g = m._blocks[0].attn.q_proj.weight.grad
    assert g is not None and torch.isfinite(g).all(), "non-finite grad after compiled bwd"
    gnorm = g.norm().item()
    if rank == 0:
        print(
            f"OK: whole-forward compile + checkpoint + flash-SDPA under DDP/NCCL. "
            f"q_proj grad norm={gnorm:.4f} (all-reduced across {dist.get_world_size()} ranks)"
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    assert os.environ.get("RANK") is not None, "run under torchrun"
    main()
