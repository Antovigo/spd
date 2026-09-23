"""Reading Llama-3.1-8B tensors straight from the HF safetensors shards, as float32 numpy."""

import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

SNAPSHOT_GLOB = str(
    Path.home() / ".cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/*"
)


def snapshot() -> Path:
    return Path(sorted(glob.glob(SNAPSHOT_GLOB))[-1])


class Weights:
    def __init__(self) -> None:
        self.dir = snapshot()
        self.index = json.loads((self.dir / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        self.config = json.loads((self.dir / "config.json").read_text())

    def get(self, name: str) -> np.ndarray:
        with safe_open(str(self.dir / self.index[name]), framework="flax") as f:
            return np.asarray(f.get_tensor(name), np.float32)

    def get_rows(self, name: str, rows: np.ndarray) -> np.ndarray:
        with safe_open(str(self.dir / self.index[name]), framework="flax") as f:
            return np.asarray(f.get_slice(name)[:][rows], np.float32)


def number_token_ids(max_n: int = 200) -> tuple[np.ndarray, int]:
    """Token ids of "0".."max_n" (single tokens in the Llama-3 vocabulary) and of "-"."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(snapshot() / "tokenizer.json"))
    ids = []
    for n in range(max_n + 1):
        enc = tok.encode(str(n), add_special_tokens=False).ids
        assert len(enc) == 1, (n, enc)
        ids.append(enc[0])
    minus = tok.encode("-", add_special_tokens=False).ids
    assert len(minus) == 1
    return np.asarray(ids), minus[0]


def llama3_inv_freq(cfg: dict[str, Any]) -> np.ndarray:
    """`param_decomp.vendored_jax.llama.llama3_inv_freq` in numpy."""
    dim = cfg["hidden_size"] // cfg["num_attention_heads"]
    inv_freq = 1.0 / (cfg["rope_theta"] ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    rs = cfg["rope_scaling"]
    factor, low, high = rs["factor"], rs["low_freq_factor"], rs["high_freq_factor"]
    old_ctx = rs["original_max_position_embeddings"]
    wavelen = 2 * np.pi / inv_freq
    inv_llama = np.where(wavelen > old_ctx / low, inv_freq / factor, inv_freq)
    smooth = (old_ctx / wavelen - low) / (high - low)
    smoothed = (1 - smooth) * inv_llama / factor + smooth * inv_llama
    is_medium = ~(wavelen < old_ctx / high) & ~(wavelen > old_ctx / low)
    return np.where(is_medium, smoothed, inv_llama).astype(np.float32)
