"""Target-owned output-space reconstruction metrics."""

import math

import jax
import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped


@jaxtyped(typechecker=beartype)
def kl_per_position(
    masked_output: Float[Array, "*leading vocab"], clean_output: Float[Array, "*leading vocab"]
) -> Float[Array, ""]:
    """Mean `KL(softmax(clean) ‖ softmax(masked))` over every leading position."""
    masked_output = masked_output.astype(jnp.float32)
    clean_output = clean_output.astype(jnp.float32)
    log_q = jax.nn.log_softmax(masked_output, axis=-1)
    log_p = jax.nn.log_softmax(clean_output, axis=-1)
    p = jnp.exp(log_p)
    return jnp.sum(p * (log_p - log_q)) / math.prod(masked_output.shape[:-1])
