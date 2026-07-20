"""Toy model with a trained-in cross-block redundancy problem, for testing recovery methods.

Two pre-RMSNorm residual MLP blocks are trained on the ResidMLP-style task
`y_i = x_i + relu(x_i)` (sparse features, fixed random embedding, tied readout) with
per-sample **block dropout**: each sample keeps both blocks or drops exactly one
(never both). A model that trains to low loss in all three modes must implement the
task redundantly — block 1 always computes it, block 2 detects (through its RMSNorm'd
residual input) whether block 1 fired and fills in whatever is missing.

This is the redundancy problem in miniature: ablating either block alone costs ~0
(the other covers) while ablating both costs the full task loss — so a per-block
sparse decomposition prunes mechanisms that the combined model needs. Ground truth is
clean: the task is analytic, the embedding is fixed, and `verify` certifies the
redundancy (per-mode losses) and reports per-feature neuron alignment for both blocks.

Usage:
    python -m param_decomp_lab.toy_models.redundant_resid_mlp train [--out-dir=PATH]
    python -m param_decomp_lab.toy_models.redundant_resid_mlp verify <run_dir>

Outputs in the run dir: `model.pth`, `config.json`, `verification.json`,
`alignment.npz` (per-block feature<->neuron cosine alignment).
"""

import json
from pathlib import Path
from typing import override

import einops
import fire
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float
from pydantic import PositiveFloat, PositiveInt
from torch import Tensor, nn

from param_decomp.base_config import BaseConfig
from param_decomp.log import logger
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.seed import set_seed


class RedundantResidMLPConfig(BaseConfig):
    n_features: PositiveInt = 32
    d_embed: PositiveInt = 64
    d_mlp: PositiveInt = 48
    n_blocks: PositiveInt = 2
    feature_prob: PositiveFloat = 0.05
    seed: int = 0
    steps: PositiveInt = 30_000
    batch_size: PositiveInt = 2048
    lr: PositiveFloat = 1e-3
    # P(keep both) = 1 - 2 * drop_prob; each block alone is dropped with drop_prob.
    drop_prob: PositiveFloat = 0.3


class RMSNorm(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    @override
    def forward(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * self.weight


class Block(nn.Module):
    """Pre-RMSNorm MLP block: `resid + mlp_out(relu(mlp_in(norm(resid))))`."""

    def __init__(self, d_embed: int, d_mlp: int):
        super().__init__()
        self.norm = RMSNorm(d_embed)
        self.mlp_in = nn.Linear(d_embed, d_mlp, bias=False)
        self.mlp_out = nn.Linear(d_mlp, d_embed, bias=False)

    @override
    def forward(self, resid: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        return self.mlp_out(F.relu(self.mlp_in(self.norm(resid))))


class RedundantResidMLP(nn.Module):
    """Fixed unit-norm embedding + tied readout; only the blocks train."""

    def __init__(self, config: RedundantResidMLPConfig):
        super().__init__()
        self.config = config
        generator = torch.Generator().manual_seed(config.seed)
        w_e = torch.randn(config.n_features, config.d_embed, generator=generator)
        self.register_buffer("W_E", w_e / w_e.norm(dim=-1, keepdim=True))
        self.W_E: Tensor
        self.blocks = nn.ModuleList(
            [Block(config.d_embed, config.d_mlp) for _ in range(config.n_blocks)]
        )

    @override
    def forward(
        self,
        x: Float[Tensor, "batch n_features"],
        block_mask: Float[Tensor, "batch n_blocks"] | None = None,
    ) -> Float[Tensor, "batch n_features"]:
        resid = einops.einsum(x, self.W_E, "b f, f d -> b d")
        for i, block in enumerate(self.blocks):
            out = block(resid)
            if block_mask is not None:
                out = out * block_mask[:, i : i + 1]
            resid = resid + out
        return einops.einsum(resid, self.W_E, "b d, f d -> b f")

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> "RedundantResidMLP":
        config = RedundantResidMLPConfig.model_validate(
            json.loads((run_dir / "config.json").read_text())
        )
        model = cls(config)
        model.load_state_dict(torch.load(run_dir / "model.pth", weights_only=True))
        return model


def sample_batch(
    config: RedundantResidMLPConfig, batch_size: int, generator: torch.Generator
) -> tuple[Float[Tensor, "batch n_features"], Float[Tensor, "batch n_features"]]:
    """Sparse features in U(-1, 1); labels `y = x + relu(x)`."""
    active = torch.rand(batch_size, config.n_features, generator=generator) < config.feature_prob
    x = (torch.rand(batch_size, config.n_features, generator=generator) * 2 - 1) * active
    return x, x + F.relu(x)


def sample_block_mask(
    config: RedundantResidMLPConfig, batch_size: int, generator: torch.Generator
) -> Float[Tensor, "batch n_blocks"]:
    """Per sample: both blocks kept, or exactly one dropped — never both."""
    assert config.n_blocks == 2, "block-dropout scheme is defined for 2 blocks"
    mode = torch.multinomial(
        torch.tensor([1 - 2 * config.drop_prob, config.drop_prob, config.drop_prob]),
        num_samples=batch_size,
        replacement=True,
        generator=generator,
    )
    masks = torch.tensor([[1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    return masks[mode]


def _mode_losses(
    model: RedundantResidMLP, config: RedundantResidMLPConfig, n_samples: int, seed: int
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    x, y = sample_batch(config, n_samples, generator)
    masks = {
        "none_dropped": [1.0, 1.0],
        "block1_dropped": [0.0, 1.0],
        "block2_dropped": [1.0, 0.0],
        "both_dropped": [0.0, 0.0],
    }
    losses = {}
    with torch.no_grad():
        for name, mask in masks.items():
            block_mask = torch.tensor([mask]).expand(n_samples, -1)
            losses[name] = float(F.mse_loss(model(x, block_mask), y))
    return losses


def train(
    out_dir: str | None = None,
    steps: int = 30_000,
    seed: int = 0,
) -> Path:
    """Train the toy and write model + config + verification artifacts."""
    config = RedundantResidMLPConfig(steps=steps, seed=seed)
    set_seed(config.seed)
    run_dir = (
        Path(out_dir).expanduser()
        if out_dir is not None
        else PARAM_DECOMP_OUT_DIR / "runs" / "toy-redundant-mlp-01"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    model = RedundantResidMLP(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    generator = torch.Generator().manual_seed(config.seed + 1)

    for step in range(config.steps + 1):
        x, y = sample_batch(config, config.batch_size, generator)
        block_mask = sample_block_mask(config, config.batch_size, generator)
        loss = F.mse_loss(model(x, block_mask), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 2000 == 0:
            logger.info(f"step {step}: train mse {float(loss):.3e}")

    torch.save(model.state_dict(), run_dir / "model.pth")
    (run_dir / "config.json").write_text(config.model_dump_json(indent=2))
    verify(str(run_dir))
    return run_dir


def verify(run_dir: str, n_samples: int = 65_536, seed: int = 123) -> dict[str, float]:
    """Certify the redundancy structure; assert it actually emerged.

    Writes `verification.json` (per-mode MSE + block-2 conditional output norms) and
    `alignment.npz` (per-block `[n_features, d_mlp]` cosine alignment between feature
    embeddings and mlp_in read directions — the per-neuron ground-truth map).
    """
    path = Path(run_dir).expanduser()
    model = RedundantResidMLP.from_run_dir(path)
    config = model.config
    losses = _mode_losses(model, config, n_samples, seed)

    # Repair evidence: block 2's output norm, with block 1 present vs dropped.
    generator = torch.Generator().manual_seed(seed + 1)
    x, _ = sample_batch(config, n_samples, generator)
    blocks = [block for block in model.blocks if isinstance(block, Block)]
    assert len(blocks) == 2
    with torch.no_grad():
        resid0 = einops.einsum(x, model.W_E, "b f, f d -> b d")
        with_block1 = float(blocks[1](resid0 + blocks[0](resid0)).norm(dim=-1).mean())
        without_block1 = float(blocks[1](resid0).norm(dim=-1).mean())

    def feature_neuron_alignment(block: Block) -> np.ndarray:
        return einops.einsum(
            model.W_E, F.normalize(block.mlp_in.weight, dim=-1), "f d, m d -> f m"
        ).numpy(force=True)

    np.savez(
        path / "alignment.npz",
        block0=feature_neuron_alignment(blocks[0]),
        block1=feature_neuron_alignment(blocks[1]),
    )

    report = losses | {
        "block2_out_norm_with_block1": with_block1,
        "block2_out_norm_without_block1": without_block1,
    }
    (path / "verification.json").write_text(json.dumps(report, indent=2))
    logger.info(f"verification: {json.dumps(report, indent=2)}")

    assert losses["block1_dropped"] < 5 * losses["none_dropped"], (
        f"block 2 does not cover for block 1: {losses}"
    )
    assert losses["block2_dropped"] < 5 * losses["none_dropped"], (
        f"block 1 does not cover for block 2: {losses}"
    )
    assert losses["both_dropped"] > 10 * losses["none_dropped"], (
        f"dropping both blocks barely hurts — no mechanism to be redundant about: {losses}"
    )
    return report


def cli() -> None:
    fire.Fire({"train": train, "verify": verify})


if __name__ == "__main__":
    cli()
