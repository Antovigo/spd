"""Pin the SDPA backend dispatch: cuDNN only where cuDNN flash attention can run
(GPU, half precision) — its SDPA rejects fp32, which must fall back to the XLA composite."""

import jax.numpy as jnp
import pytest

from param_decomp.vendored_jax.llama import attn_implementation


@pytest.mark.parametrize(
    ("backend", "dtype", "expected"),
    [
        ("gpu", jnp.bfloat16, "cudnn"),
        ("gpu", jnp.float16, "cudnn"),
        ("gpu", jnp.float32, "xla"),
        ("cpu", jnp.bfloat16, "xla"),
        ("cpu", jnp.float32, "xla"),
    ],
)
def test_attn_implementation_dispatch(backend: str, dtype: jnp.dtype, expected: str) -> None:
    assert attn_implementation(backend, jnp.dtype(dtype)) == expected
