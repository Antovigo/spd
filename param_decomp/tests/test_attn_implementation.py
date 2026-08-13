"""Pin the SDPA backend dispatch: the XLA composite, unconditionally.

cuDNN is off on this branch because its graph API cannot run against a pre-CUDA-12.8
driver — `CuDnnThunk::Initialize` fails inside `run_auxiliary_kernels`' `cudaMemcpyAsync`
with an unchecked `cudaErrorInvalidValue`, which then resurfaces at an unrelated
`cuModuleGetFunction` and names an innocent kernel. It blocked every dp>=1 targeted run.

The cases below are the dispatch that WOULD have selected cuDNN (GPU, half precision,
seq_len a multiple of 64) alongside the ones that never did — fp32, CPU, and the tPD
target stream's natural prompt lengths (found live: `Unsupported sequence length Q 5,
KV 5` from `check_is_flash_attention` on run p-5de41a1c). They are kept as a record of
the capability check, so restoring it on a newer driver is a one-line edit here and one
in `attn_implementation`."""

import jax.numpy as jnp
import pytest

from param_decomp.vendored_jax.llama import attn_implementation


@pytest.mark.parametrize(
    ("backend", "dtype", "seq_len", "expected"),
    [
        ("gpu", jnp.bfloat16, 2048, "xla"),  # would have been cudnn
        ("gpu", jnp.float16, 512, "xla"),  # would have been cudnn
        ("gpu", jnp.float32, 2048, "xla"),
        ("cpu", jnp.bfloat16, 2048, "xla"),
        ("cpu", jnp.float32, 2048, "xla"),
        ("gpu", jnp.bfloat16, 5, "xla"),  # tPD target stream: natural prompt length
        ("gpu", jnp.bfloat16, 96, "xla"),  # not a multiple of 64
        ("gpu", jnp.bfloat16, 64, "xla"),  # would have been cudnn — the non-target stream
    ],
)
def test_attn_implementation_dispatch(
    backend: str, dtype: jnp.dtype, seq_len: int, expected: str
) -> None:
    assert attn_implementation(backend, jnp.dtype(dtype), seq_len) == expected
