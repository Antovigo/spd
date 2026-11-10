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
from param_decomp.ci_filter.step import (
    CIConstraints,
    FilterCIFn,
    Prepared,
    all_logits,
    gather_batch,
    masked_output,
    output_ci,
)
from param_decomp.core.model import MaterializedMasking
from param_decomp.core.precision import COMPUTE_DT
from param_decomp.experiments.lm.load_run import PlacedTransformer

MASKINGS = ("all_on", "alive_union", "dead_only", "rounded_ci", "rounded_ci_plus_dead")


@eqx.filter_jit
def ablation_rows(
    placed: PlacedTransformer,
    prepared: Prepared,
    ci_fn: FilterCIFn,
    constraints: CIConstraints,
    alive: dict[str, Float[Array, " C"]],
    tokens_all: Int[Array, "N T"],
    idx: Int[Array, " B"],
    threshold: float,
) -> dict[str, dict[str, Array]]:
    """`{masking: {"last": (B,), "all_positions": (B,)}}` KLs, replicated."""
    clean = placed.clean_forward(gather_batch(tokens_all, idx), ci_fn.capture_keys)
    clean_logits = all_logits(clean.output)
    lower = output_ci(placed, prepared, ci_fn, constraints, clean, remat=False).lower
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
        logits = all_logits(
            masked_output(
                placed,
                prepared,
                clean.conditioning,
                MaterializedMasking(component_masks=maskings[name]),
                remat=False,
            )
        )
        per_position = kl_rows(logits, clean_logits)
        out[name] = {
            "last": jax.sharding.reshard(per_position[:, -1], P()),
            "all_positions": jax.sharding.reshard(jnp.mean(per_position, axis=1), P()),
        }
    return out
