"""Fixed and per-prompt mask ablations of one CI fn over the pool, weight delta off.

Scores, for every prompt, the KL at the last position and averaged over every position (the
decomposition's own `UnmaskedReconLoss` / `ce_kl/kl_unmasked` reduction) under:

- `all_on`: every component on (= `UnmaskedReconLoss`: masks 1, delta off);
- `alive_union`: the components alive anywhere on the pool, on for every prompt;
- `dead_only`: the complement, on for every prompt;
- `rounded_ci`: this prompt's own components, `CI > threshold` per position;
- `rounded_ci_plus_dead`: this prompt's components plus every dead one.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from param_decomp.ci_filter.objective import kl_rows
from param_decomp.ci_filter.step import Ceilings, Prepared, gather_rows, output_ci
from param_decomp.core.ci_fn import PlacedCIFn
from param_decomp.core.model import MaterializedMasking, PlacedModel
from param_decomp.core.precision import COMPUTE_DT

MASKINGS = ("all_on", "alive_union", "dead_only", "rounded_ci", "rounded_ci_plus_dead")


@eqx.filter_jit
def ablation_rows(
    placed: PlacedModel,
    prepared: Prepared,
    ci_fn: PlacedCIFn,
    ceilings: Ceilings,
    alive: dict[str, Float[Array, " C"]],
    tokens_all: Int[Array, "N T"],
    idx: Int[Array, " B"],
    threshold: float,
) -> dict[str, dict[str, Array]]:
    """`{masking: {"last": (B,), "all_positions": (B,)}}` KLs, replicated."""
    tokens = gather_rows(tokens_all, idx)
    clean = placed.clean_forward(tokens, capture_keys=ci_fn.capture_keys)
    lower = output_ci(placed, ci_fn, ceilings, clean.captures, remat=False).lower
    rounded = {k: (v > threshold).astype(COMPUTE_DT) for k, v in lower.items()}
    fixed = {k: v[None, None, :].astype(COMPUTE_DT) for k, v in alive.items()}
    maskings = {
        "all_on": {k: jnp.ones_like(v) for k, v in fixed.items()},
        "alive_union": fixed,
        "dead_only": {k: 1.0 - v for k, v in fixed.items()},
        "rounded_ci": rounded,
        "rounded_ci_plus_dead": {k: jnp.maximum(v, 1.0 - fixed[k]) for k, v in rounded.items()},
    }
    out: dict[str, dict[str, Array]] = {}
    for name in MASKINGS:
        logits = placed.masked_forward(
            prepared,
            tokens,
            masking=MaterializedMasking(component_masks=maskings[name]),
            remat=False,
        ).output
        per_position = kl_rows(logits, clean.output)
        out[name] = {
            "last": jax.sharding.reshard(per_position[:, -1], P()),
            "all_positions": jax.sharding.reshard(jnp.mean(per_position, axis=1), P()),
        }
    return out
