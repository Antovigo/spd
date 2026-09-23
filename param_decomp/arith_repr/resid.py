"""Where the residual stream is stored and how a read point's input is recovered from it.

The harvest (`scripts/harvest.py`) stores the RAW residual stream of the frozen model — a
property of the model, not of any decomposition — as bf16-as-uint16 `.npy` memmaps of shape
`(n_prompts, T, d_model)`:

    resid.<l>      the residual entering block l (l = 0..n_layer; n_layer = exiting the last)
    post_attn.<l>  the residual after block l's attention add, before its MLP norm

plus `norms.npz` (`ln1`, `ln2`: (n_layer, d) RMSNorm weights; `final`; `eps`) and `pool.json`.
The analysis works on what the projections read, so a read point `attn_in.l` / `mlp_in.l` is
recovered here as `rms_norm(raw, weight)` with the model's own arithmetic (float32 statistics,
bf16 output, bf16 weight). A directory without `norms.npz` (the synthetic test) is taken to
hold read-point inputs directly under the read-point names."""

from pathlib import Path

import ml_dtypes
import numpy as np


def raw_key(read_point: str) -> str:
    """The raw-stream file feeding a read point: `attn_in.l` <- `resid.l`, `mlp_in.l` <- `post_attn.l`."""
    stream, layer = read_point.split(".")
    return {"attn_in": f"resid.{layer}", "mlp_in": f"post_attn.{layer}"}[stream]


def load_raw(resid_dir: Path, key: str, position: int) -> np.ndarray:
    """One position of one stored array as bf16 values (float32 array)."""
    mm = np.load(resid_dir / f"{key}.npy", mmap_mode="r")
    return np.asarray(mm[:, position, :]).view(ml_dtypes.bfloat16).astype(np.float32)


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """`param_decomp.vendored_jax.llama.rms_norm` on bf16 inputs, in numpy."""
    x32 = x.astype(np.float32)
    x32 = x32 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    normed = x32.astype(ml_dtypes.bfloat16) * weight.astype(ml_dtypes.bfloat16)
    return normed.astype(np.float32)


def load_read_input(resid_dir: Path, read_point: str, position: int) -> np.ndarray:
    """What the read point's projections see at `position`: the post-norm residual, (n, d)."""
    norms_path = resid_dir / "norms.npz"
    if not norms_path.exists():
        return load_raw(resid_dir, read_point, position)
    stream, layer = read_point.split(".")
    norms = np.load(norms_path)
    weight = norms["ln1" if stream == "attn_in" else "ln2"][int(layer)]
    return rms_norm(load_raw(resid_dir, raw_key(read_point), position), weight, float(norms["eps"]))
