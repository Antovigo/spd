"""What a component does OUTSIDE arithmetic: subtract it from the model and watch general text.

The ablation is the frozen model MINUS one component. With every component mask at 1 and the
weight-delta mask at 1, `site_forward` returns `x @ W` exactly (the delta carries `W - UV`), so
zeroing one component's mask leaves `x @ W - (x . V_c) U_c`. The baseline is therefore the target
model itself, not a reconstruction of it — this run dropped the faithfulness term, so "all
components on, delta off" is NOT the model on general text (its KL there is ~6.3).

Per position we record the KL of that subtraction against the model and the argmax token of both
forwards. A component that matters only on a narrow, rarely-co-occurring task shows near-zero KL
on almost every position with a thin tail of large ones; a component that matters broadly shifts
the bulk. That contrast is the point of the probe."""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int

from param_decomp.ci_filter.objective import kl_rows
from param_decomp.ci_filter.step import Prepared
from param_decomp.core.model import MaterializedMasking, PlacedModel
from param_decomp.core.precision import COMPUTE_DT


class ProbeBaseline(eqx.Module):
    """One text batch's reference forward: the frozen target model."""

    clean_logits: Float[Array, "B T V"]
    clean_top1: Int[Array, "B T"]
    subtract_nothing_kl: Float[Array, "B T"]
    """KL of the masked forward that subtracts NOTHING against the clean one: a numerical
    identity check (bf16 noise), not a model property."""


def _per_position_kl(
    p_logits: Float[Array, "B T V"], q_logits: Float[Array, "B T V"]
) -> Float[Array, "B T"]:
    """`KL(p || q)` at every position, fp32."""
    flat = kl_rows(
        q_logits.reshape(-1, q_logits.shape[-1]), p_logits.reshape(-1, p_logits.shape[-1])
    )
    return flat.reshape(p_logits.shape[:-1])


def _subtracting(
    placed: PlacedModel, prepared: Prepared, tokens: Int[Array, "B T"], keep: dict[str, Array]
) -> Float[Array, "B T V"]:
    """`x @ W` minus the components whose `keep` entry is 0, at every site."""
    masks = {
        site.name: jnp.broadcast_to(keep[site.name].astype(COMPUTE_DT), (*tokens.shape, site.C))
        for site in placed.sites
    }
    delta = {site.name: jnp.ones(tokens.shape, COMPUTE_DT) for site in placed.sites}
    return placed.masked_forward(
        prepared,
        tokens,
        masking=MaterializedMasking(component_masks=masks, weight_delta_masks=delta),
        remat=False,
    ).output


def make_probe_baseline() -> Callable[..., ProbeBaseline]:
    @eqx.filter_jit
    def probe_baseline(
        placed: PlacedModel, prepared: Prepared, tokens: Int[Array, "B T"], keep: dict[str, Array]
    ) -> ProbeBaseline:
        clean_logits = placed.clean_forward(tokens).output
        identity = _subtracting(placed, prepared, tokens, keep)
        return ProbeBaseline(
            clean_logits=clean_logits,
            clean_top1=jax.sharding.reshard(jnp.argmax(clean_logits, axis=-1), P()),
            subtract_nothing_kl=jax.sharding.reshard(_per_position_kl(clean_logits, identity), P()),
        )

    return probe_baseline


def make_probe_component() -> Callable[..., dict[str, Array]]:
    """`(placed, prepared, tokens, baseline, keep) -> {kl, top1}` per position, where `keep` is a
    per-site `(C,)` vector of ones with zeros at the components to subtract. Traced, so scanning
    components costs one compile."""

    @eqx.filter_jit
    def probe_component(
        placed: PlacedModel,
        prepared: Prepared,
        tokens: Int[Array, "B T"],
        baseline: ProbeBaseline,
        keep: dict[str, Array],
    ) -> dict[str, Array]:
        logits = _subtracting(placed, prepared, tokens, keep)
        out = {
            "kl": _per_position_kl(baseline.clean_logits, logits),
            "top1": jnp.argmax(logits, axis=-1),
        }
        return {k: jax.sharding.reshard(v, P()) for k, v in out.items()}

    return probe_component
