"""Toy transformer with trained-in n-fold cross-block redundancy, for recovery-method tests.

An MLP-only decoder: token-ID sequences enter a fixed random unit-norm embedding, pass
through `n_blocks` pre-RMSNorm residual MLP blocks (the only trained parameters, plus
norm gains), and exit through a final RMSNorm and the tied unembedding as a
distribution over tokens (per-position cross-entropy). No attention and no positional
embedding: the task is per-position, so positions are independent batched samples
through shared weights.

Task: predict `pi(x_t)` at every position, where `pi` is a fixed random *derangement*
of the vocab (no fixed points), sampled once from the seed. With embed/unembed frozen,
each block must rotate `e_x` toward `e_(pi(x))` in the residual stream — one mechanism
per token.

Redundancy: each training sequence keeps a block subset drawn uniformly from all
`2^n - 1` non-empty subsets. Every singleton is trained, so EVERY block alone must
implement the full cipher, and blocks tolerate any combination of live partners
(reading the RMSNorm'd residual tells a block what upstream already contributed). The
derangement makes the certificate binary: the empty subset argmaxes to `x` itself and
scores exactly 0 accuracy, any working subset ~1.

Usage:
    python -m param_decomp_lab.toy_models.toy_model_redundancy train [--out-dir=PATH]
    python -m param_decomp_lab.toy_models.toy_model_redundancy verify <run_dir>

Outputs in the run dir: `model.pth`, `config.json`, `verification.json` (per-subset CE
+ accuracy), `alignment.npz` (per-block read/write alignment to `e_x` / `e_(pi(x))`),
`per_token_accuracy.npz` (`[n_blocks, vocab]` accuracy of each block alone per input
token — the per-token redundancy map).
"""

import itertools
import json
from pathlib import Path
from typing import override

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


class ToyModelRedundancyConfig(BaseConfig):
    vocab_size: PositiveInt = 32
    d_embed: PositiveInt = 64
    d_mlp: PositiveInt = 48
    n_blocks: PositiveInt = 3
    seq_len: PositiveInt = 8
    seed: int = 0
    steps: PositiveInt = 5_000
    batch_size: PositiveInt = 512
    lr: PositiveFloat = 1e-3


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


def _sample_derangement(vocab_size: int, generator: torch.Generator) -> Tensor:
    while True:
        pi = torch.randperm(vocab_size, generator=generator)
        if not (pi == torch.arange(vocab_size)).any():
            return pi


class ToyModelRedundancyTransformer(nn.Module):
    """Fixed unit-norm embedding, tied unembedding, final RMSNorm; only blocks train."""

    def __init__(self, config: ToyModelRedundancyConfig):
        super().__init__()
        self.config = config
        generator = torch.Generator().manual_seed(config.seed)
        w_e = torch.randn(config.vocab_size, config.d_embed, generator=generator)
        self.register_buffer("W_E", w_e / w_e.norm(dim=-1, keepdim=True))
        self.W_E: Tensor
        self.register_buffer("pi", _sample_derangement(config.vocab_size, generator))
        self.pi: Tensor
        self.blocks = nn.ModuleList(
            [Block(config.d_embed, config.d_mlp) for _ in range(config.n_blocks)]
        )
        self.final_norm = RMSNorm(config.d_embed)

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
        return einops.einsum(self.final_norm(resid), self.W_E, "b s d, v d -> b s v")

    @classmethod
    def from_run_dir(cls, run_dir: Path) -> "ToyModelRedundancyTransformer":
        config = ToyModelRedundancyConfig.model_validate(
            json.loads((run_dir / "config.json").read_text())
        )
        model = cls(config)
        model.load_state_dict(torch.load(run_dir / "model.pth", weights_only=True))
        return model


def sample_block_mask(
    config: ToyModelRedundancyConfig, batch_size: int, generator: torch.Generator
) -> Float[Tensor, "batch n_blocks"]:
    """Per sequence: a block subset uniform over all `2^n - 1` non-empty subsets."""
    subset = torch.randint(1, 2**config.n_blocks, (batch_size,), generator=generator)
    bits = torch.arange(config.n_blocks)
    return ((subset[:, None] >> bits) & 1).float()


def _loss_and_acc(
    model: ToyModelRedundancyTransformer,
    tokens: Int[Tensor, "batch seq"],
    block_mask: Float[Tensor, "batch n_blocks"] | None,
) -> tuple[Tensor, Tensor]:
    logits = model(tokens, block_mask)
    targets = model.pi[tokens]
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    acc = (logits.argmax(dim=-1) == targets).float().mean()
    return loss, acc


def train(
    out_dir: str | None = None,
    steps: int = 5_000,
    n_blocks: int = 3,
    seed: int = 0,
) -> Path:
    """Train the toy and write model + config + verification artifacts."""
    config = ToyModelRedundancyConfig(steps=steps, n_blocks=n_blocks, seed=seed)
    set_seed(config.seed)
    run_dir = (
        Path(out_dir).expanduser()
        if out_dir is not None
        else PARAM_DECOMP_OUT_DIR / "runs" / "toy_model_redundancy" / "training"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    model = ToyModelRedundancyTransformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    generator = torch.Generator().manual_seed(config.seed + 1)

    for step in range(config.steps + 1):
        tokens = torch.randint(
            0, config.vocab_size, (config.batch_size, config.seq_len), generator=generator
        )
        block_mask = sample_block_mask(config, config.batch_size, generator)
        loss, acc = _loss_and_acc(model, tokens, block_mask)
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
    """Certify the n-fold redundancy; assert it actually emerged.

    Writes `verification.json` (CE + accuracy for every block subset, keyed by the
    keep-mask bit string, e.g. '101'), `alignment.npz` (per block: `[vocab, d_mlp]`
    cosine alignment of `e_x` with mlp_in read directions and of `e_(pi(x))` with
    mlp_out write directions) and `per_token_accuracy.npz` (`[n_blocks, vocab]`
    accuracy of each block alone, per input token).
    """
    path = Path(run_dir).expanduser()
    model = ToyModelRedundancyTransformer.from_run_dir(path)
    config = model.config
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, config.vocab_size, (n_samples, config.seq_len), generator=generator)

    report: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for keep in itertools.product((0.0, 1.0), repeat=config.n_blocks):
            mask = torch.tensor([keep]).expand(n_samples, -1)
            loss, acc = _loss_and_acc(model, tokens, mask)
            name = "".join(str(int(k)) for k in keep)
            report[name] = {"ce": float(loss), "accuracy": float(acc)}

        # Per-token redundancy map: which blocks host which token's mechanism.
        per_token = torch.zeros(config.n_blocks, config.vocab_size)
        for b in range(config.n_blocks):
            mask = F.one_hot(torch.tensor(b), config.n_blocks).float().expand(n_samples, -1)
            pred = model(tokens, mask).argmax(dim=-1)
            correct = (pred == model.pi[tokens]).float().flatten()
            flat_tokens = tokens.flatten()
            for v in range(config.vocab_size):
                per_token[b, v] = correct[flat_tokens == v].mean()

        blocks = [block for block in model.blocks if isinstance(block, Block)]
        e_pi = model.W_E[model.pi]
        alignment = {}
        for i, block in enumerate(blocks):
            alignment[f"block{i}_read"] = einops.einsum(
                model.W_E, F.normalize(block.mlp_in.weight, dim=-1), "v d, m d -> v m"
            ).numpy(force=True)
            alignment[f"block{i}_write"] = einops.einsum(
                e_pi, F.normalize(block.mlp_out.weight, dim=0), "v d, d m -> v m"
            ).numpy(force=True)
        np.savez(path / "alignment.npz", **alignment)
    np.savez(path / "per_token_accuracy.npz", accuracy=per_token.numpy(force=True))
    (path / "verification.json").write_text(json.dumps(report, indent=2))
    logger.info(f"verification: {json.dumps(report, indent=2)}")

    empty = "0" * config.n_blocks
    assert report[empty]["accuracy"] < 0.05, (
        f"empty model should score ~0 (derangement): {report[empty]}"
    )
    for b in range(config.n_blocks):
        singleton = "".join("1" if i == b else "0" for i in range(config.n_blocks))
        assert report[singleton]["accuracy"] > 0.9, (
            f"block {b} alone does not implement the cipher: {report[singleton]}"
        )
    full = "1" * config.n_blocks
    assert report[full]["accuracy"] > 0.99, f"full model should be ~perfect: {report[full]}"
    return report


def cli() -> None:
    fire.Fire({"train": train, "verify": verify})


if __name__ == "__main__":
    cli()
