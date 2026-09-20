"""What a component does OUTSIDE arithmetic: ablate it and watch general text, token by token.

The baseline is the decomposition with every component ON (weight delta off) — the closest thing
to the target model that still has components to remove — so ablating one component isolates that
component's contribution. Per position we record the KL against that baseline, the KL against the
frozen model itself (for scale), and the argmax tokens of all three forwards.

A component that matters only on a narrow, rarely-co-occurring task shows near-zero KL on almost
every position with a thin tail of large ones; a component that matters broadly shows a shifted
bulk. That contrast is the point of the probe."""

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
    """One text batch's reference forwards, reused by every component."""

    clean_logits: Float[Array, "B T V"]
    """The frozen target model."""
    base_logits: Float[Array, "B T V"]
    """Every component on, weight delta off."""
    clean_top1: Int[Array, "B T"]
    base_top1: Int[Array, "B T"]
    base_kl: Float[Array, "B T"]
    """`KL(clean || base)`: the decomposition's own reconstruction error on this text."""


def _per_position_kl(
    p_logits: Float[Array, "B T V"], q_logits: Float[Array, "B T V"]
) -> Float[Array, "B T"]:
    """`KL(p || q)` at every position, fp32."""
    flat = kl_rows(
        q_logits.reshape(-1, q_logits.shape[-1]), p_logits.reshape(-1, p_logits.shape[-1])
    )
    return flat.reshape(p_logits.shape[:-1])


def make_probe_baseline() -> Callable[[PlacedModel, Prepared, Int[Array, "B T"]], ProbeBaseline]:
    @eqx.filter_jit
    def probe_baseline(
        placed: PlacedModel, prepared: Prepared, tokens: Int[Array, "B T"]
    ) -> ProbeBaseline:
        clean_logits = placed.clean_forward(tokens).output
        ones = {site.name: jnp.ones((*tokens.shape, site.C), COMPUTE_DT) for site in placed.sites}
        base_logits = placed.masked_forward(
            prepared, tokens, masking=MaterializedMasking(component_masks=ones), remat=False
        ).output
        return ProbeBaseline(
            clean_logits=clean_logits,
            base_logits=base_logits,
            clean_top1=jax.sharding.reshard(jnp.argmax(clean_logits, axis=-1), P()),
            base_top1=jax.sharding.reshard(jnp.argmax(base_logits, axis=-1), P()),
            base_kl=jax.sharding.reshard(_per_position_kl(clean_logits, base_logits), P()),
        )

    return probe_baseline


def make_probe_component() -> Callable[..., dict[str, Array]]:
    """`(placed, prepared, tokens, baseline, keep) -> {kl_vs_base, kl_vs_clean, top1}` per
    position. `keep` is a per-site `(C,)` vector of ones with zeros at the ablated components,
    traced so scanning components costs one compile."""

    @eqx.filter_jit
    def probe_component(
        placed: PlacedModel,
        prepared: Prepared,
        tokens: Int[Array, "B T"],
        baseline: ProbeBaseline,
        keep: dict[str, Array],
    ) -> dict[str, Array]:
        masks = {
            site.name: jnp.broadcast_to(keep[site.name].astype(COMPUTE_DT), (*tokens.shape, site.C))
            for site in placed.sites
        }
        logits = placed.masked_forward(
            prepared, tokens, masking=MaterializedMasking(component_masks=masks), remat=False
        ).output
        out = {
            "kl_vs_base": _per_position_kl(baseline.base_logits, logits),
            "kl_vs_clean": _per_position_kl(baseline.clean_logits, logits),
            "top1": jnp.argmax(logits, axis=-1),
        }
        return {k: jax.sharding.reshard(v, P()) for k, v in out.items()}

    return probe_component
