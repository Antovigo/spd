"""Attention copy toy: block-dropout-certified redundancy on the copy task.

Sequences `[x, Q]`, pre-RMSNorm single-head causal attention blocks, final RMSNorm,
tied unembedding, CE on the last position; the target is the position-0 token itself —
each block must copy `e_x` into the last position. Trained with per-sequence block
dropout (subsets uniform over the `2^n - 1` non-empty ones), so every block alone must
implement the copy.

The empty model predicts a constant from `e_Q` (~1/vocab accuracy, not the
derangement's exact 0), so the empty-subset certificate is `< 0.2` rather than binary.

Usage:
    python -m param_decomp_lab.toy_models.toy_model_redundancy_copy train [--out-dir=PATH] [--n-blocks=3]
    python -m param_decomp_lab.toy_models.toy_model_redundancy_copy verify <run_dir>

Outputs: `model.pth`, `config.json`, `verification.json` (per-block-subset accuracy at
the last position), `per_token_accuracy.npz` (`accuracy` = block-alone map, `marginal`
= drop-one-block map, both `[n_blocks, vocab]`).
"""

import itertools
import json
import math
from pathlib import Path
from typing import Literal, override

import einops
import fire
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from pydantic import PositiveFloat, PositiveInt
from torch import Tensor, nn

from param_decomp.base_config import BaseConfig
from param_decomp.log import logger
from param_decomp_lab.infra.settings import PARAM_DECOMP_OUT_DIR
from param_decomp_lab.seed import set_seed


class RMSNorm(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    @override
    def forward(self, x: Float[Tensor, "... d"]) -> Float[Tensor, "... d"]:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * self.weight


class AttnBlock(nn.Module):
    """Pre-RMSNorm single-head causal attention: `resid + o(attn(q, k, v))`."""

    def __init__(self, d_embed: int):
        super().__init__()
        self.norm = RMSNorm(d_embed)
        self.q = nn.Linear(d_embed, d_embed, bias=False)
        self.k = nn.Linear(d_embed, d_embed, bias=False)
        self.v = nn.Linear(d_embed, d_embed, bias=False)
        self.o = nn.Linear(d_embed, d_embed, bias=False)

    @override
    def forward(self, resid: Float[Tensor, "batch seq d"]) -> Float[Tensor, "batch seq d"]:
        h = self.norm(resid)
        scores = einops.einsum(self.q(h), self.k(h), "b sq d, b sk d -> b sq sk")
        scores = scores / math.sqrt(h.shape[-1])
        causal = torch.ones(h.shape[1], h.shape[1], dtype=torch.bool, device=h.device).tril()
        attn = scores.masked_fill(~causal, float("-inf")).softmax(dim=-1)
        return self.o(einops.einsum(attn, self.v(h), "b sq sk, b sk d -> b sq d"))


class ToyModelRedundancyCopyConfig(BaseConfig):
    kind: Literal["copy_attn"] = "copy_attn"
    """Discriminator so the experiment's `build_target` can tell the toys apart."""
    vocab_size: PositiveInt = 32
    d_embed: PositiveInt = 64
    n_blocks: PositiveInt = 3
    seed: int = 0
    steps: PositiveInt = 5_000
    batch_size: PositiveInt = 512
    lr: PositiveFloat = 1e-3
    redundant_tokens: PositiveInt = 32
    """Tokens `< redundant_tokens` get block dropout (n-fold redundancy); the rest
    always see the full network."""


class ToyModelRedundancyCopyTransformer(nn.Module):
    """Fixed unit-norm embedding (vocab + query token), tied unembedding over the vocab."""

    def __init__(self, config: ToyModelRedundancyCopyConfig):
        super().__init__()
        self.config = config
        generator = torch.Generator().manual_seed(config.seed)
        w_e = torch.randn(config.vocab_size + 1, config.d_embed, generator=generator)
        self.register_buffer("W_E", w_e / w_e.norm(dim=-1, keepdim=True))
        self.W_E: Tensor
        self.blocks = nn.ModuleList([AttnBlock(config.d_embed) for _ in range(config.n_blocks)])
        self.final_norm = RMSNorm(config.d_embed)

    @property
    def query_token(self) -> int:
        return self.config.vocab_size

    def enumerate_inputs(self) -> Int[Tensor, "vocab 2"]:
        """One `[x, Q]` sequence per vocab token — the canonical CI-probe batch."""
        x = torch.arange(self.config.vocab_size)
        return torch.stack([x, torch.full_like(x, self.query_token)], dim=1)

    def sample_inputs(self, batch_size: int) -> Int[Tensor, "batch 2"]:
        x = torch.randint(0, self.config.vocab_size, (batch_size,))
        return torch.stack([x, torch.full_like(x, self.query_token)], dim=1)

    @override
    def forward(
        self,
        tokens: Int[Tensor, "batch seq"],
        block_mask: Float[Tensor, "batch n_blocks"] | None = None,
    ) -> Float[Tensor, "batch seq vocab"]:
        resid = self.W_E[tokens]
        for i, block in enumerate(self.blocks):
            out = block(resid)
            if block_mask is not None:
                out = out * block_mask[:, i, None, None]
            resid = resid + out
        unembed = self.W_E[: self.config.vocab_size]
        return einops.einsum(self.final_norm(resid), unembed, "b s d, v d -> b s v")

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> "ToyModelRedundancyCopyTransformer":
        config = ToyModelRedundancyCopyConfig.model_validate(
            json.loads((run_dir / "config.json").read_text())
        )
        model = cls(config)
        model.load_state_dict(torch.load(run_dir / "model.pth", weights_only=True))
        return model


def _sample_block_mask(
    config: ToyModelRedundancyCopyConfig, x: Int[Tensor, " batch"], generator: torch.Generator
) -> Float[Tensor, "batch n_blocks"]:
    """Non-empty block subset per sequence for `x < redundant_tokens`, all-ones else."""
    subset = torch.randint(1, 2**config.n_blocks, (x.shape[0],), generator=generator)
    bits = torch.arange(config.n_blocks)
    mask = ((subset[:, None] >> bits) & 1).float()
    return torch.where((x < config.redundant_tokens)[:, None], mask, torch.ones_like(mask))


def _make_batch(
    config: ToyModelRedundancyCopyConfig, batch_size: int, generator: torch.Generator
) -> Int[Tensor, "batch 2"]:
    x = torch.randint(0, config.vocab_size, (batch_size,), generator=generator)
    return torch.stack([x, torch.full_like(x, config.vocab_size)], dim=1)


def _loss_and_acc(
    model: ToyModelRedundancyCopyTransformer,
    tokens: Int[Tensor, "batch 2"],
    block_mask: Float[Tensor, "batch n_blocks"] | None,
) -> tuple[Tensor, Tensor]:
    logits = model(tokens, block_mask)[:, -1]
    targets = tokens[:, 0]
    loss = F.cross_entropy(logits, targets)
    acc = (logits.argmax(dim=-1) == targets).float().mean()
    return loss, acc


def train(
    out_dir: str | None = None,
    steps: int = 5_000,
    n_blocks: int = 3,
    d_embed: int = 64,
    vocab_size: int = 32,
    redundant_tokens: int = 32,
    seed: int = 0,
) -> Path:
    """Train the copy toy (block dropout on tokens `< redundant_tokens`) and write artifacts."""
    config = ToyModelRedundancyCopyConfig(
        steps=steps,
        n_blocks=n_blocks,
        d_embed=d_embed,
        vocab_size=vocab_size,
        redundant_tokens=redundant_tokens,
        seed=seed,
    )
    set_seed(config.seed)
    run_dir = (
        Path(out_dir).expanduser()
        if out_dir is not None
        else PARAM_DECOMP_OUT_DIR / "runs" / "toy_model_redundancy" / "copy_training"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    model = ToyModelRedundancyCopyTransformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    generator = torch.Generator().manual_seed(config.seed + 1)

    for step in range(config.steps + 1):
        tokens = _make_batch(config, config.batch_size, generator)
        mask = _sample_block_mask(config, tokens[:, 0], generator)
        loss, acc = _loss_and_acc(model, tokens, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 2000 == 0:
            logger.info(f"step {step}: train ce {float(loss):.3e} acc {float(acc):.3f}")

    torch.save(model.state_dict(), run_dir / "model.pth")
    (run_dir / "config.json").write_text(config.model_dump_json(indent=2))
    verify(str(run_dir))
    return run_dir


def verify(run_dir: str, n_samples: int = 8192, seed: int = 123) -> dict[str, dict[str, float]]:
    """Certify the n-fold copy redundancy; assert it actually emerged."""
    path = Path(run_dir).expanduser()
    model = ToyModelRedundancyCopyTransformer.from_run_dir(path)
    config = model.config
    generator = torch.Generator().manual_seed(seed)
    tokens = _make_batch(config, n_samples, generator)

    report: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for keep in itertools.product((0.0, 1.0), repeat=config.n_blocks):
            mask = torch.tensor([keep]).expand(n_samples, -1)
            loss, acc = _loss_and_acc(model, tokens, mask)
            name = "".join(str(int(k)) for k in keep)
            report[name] = {"ce": float(loss), "accuracy": float(acc)}

        def per_token_map(masks: list[Tensor]) -> Tensor:
            out = torch.zeros(config.n_blocks, config.vocab_size)
            for b in range(config.n_blocks):
                pred = model(tokens, masks[b].expand(n_samples, -1)).argmax(dim=-1)[:, -1]
                correct = (pred == tokens[:, 0]).float()
                for v in range(config.vocab_size):
                    out[b, v] = correct[tokens[:, 0] == v].mean()
            return out

        eye = torch.eye(config.n_blocks)
        alone = per_token_map([eye[b][None] for b in range(config.n_blocks)])
        marginal = per_token_map([(1.0 - eye[b])[None] for b in range(config.n_blocks)])

    np.savez(
        path / "per_token_accuracy.npz",
        accuracy=alone.numpy(force=True),
        marginal=marginal.numpy(force=True),
    )
    (path / "verification.json").write_text(json.dumps(report, indent=2))
    logger.info(f"verification: {json.dumps(report, indent=2)}")
    logger.info(f"block-alone acc per block: {alone.mean(dim=1).tolist()}")

    full = "1" * config.n_blocks
    assert report[full]["accuracy"] > 0.99, f"full model should be ~perfect: {report[full]}"
    empty = "0" * config.n_blocks
    assert report[empty]["accuracy"] < 0.2, f"empty model should be ~chance: {report[empty]}"
    redundant = alone[:, : config.redundant_tokens]
    assert bool((redundant > 0.9).all()), (
        f"some redundant (block, token) mechanism is missing: min {float(redundant.min()):.3f}"
    )
    return report


def cli() -> None:
    fire.Fire({"train": train, "verify": verify})


if __name__ == "__main__":
    cli()
