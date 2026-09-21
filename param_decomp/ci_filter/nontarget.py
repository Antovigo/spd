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
from param_decomp.ci_filter.step import UNCONSTRAINED, Prepared, output_ci
from param_decomp.core.ci_fn import PlacedCIFn
from param_decomp.core.model import MaterializedMasking, PlacedModel
from param_decomp.core.precision import COMPUTE_DT


class ProbeBaseline(eqx.Module):
    """One text batch's reference: the SUBTRACT-NOTHING masked forward, which is the model up to
    bf16 kernel differences. Scoring an ablation against this rather than against
    `clean_forward` cancels those differences, and they are not small — ~8e-4 KL, larger than a
    single component's typical effect on general text."""

    logits: Float[Array, "B T V"]
    top1: Int[Array, "B T"]
    clean_kl: Float[Array, "B T"]
    """KL against `clean_forward`: the noise floor this probe cannot see below."""
    clean_top1: Int[Array, "B T"]


def _per_position_kl(
    p_logits: Float[Array, "B T V"], q_logits: Float[Array, "B T V"]
) -> Float[Array, "B T"]:
    """`KL(p || q)` at every position, fp32."""
    flat = kl_rows(
        q_logits.reshape(-1, q_logits.shape[-1]), p_logits.reshape(-1, p_logits.shape[-1])
    )
    return flat.reshape(p_logits.shape[:-1])


def subtracting(
    placed: PlacedModel, prepared: Prepared, tokens: Int[Array, "B T"], keep: dict[str, Array]
) -> Float[Array, "B T V"]:
    """`x @ W` minus the components whose `keep` entry is 0, at every site: with every mask at 1
    and the delta mask at 1 the masked forward IS the frozen model, so a zeroed mask subtracts
    exactly that component's contribution."""
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


def make_probe_ci(selection: tuple[tuple[str, int], ...]) -> Callable[..., Float[Array, "B T K"]]:
    """The decomposition's own output CI of the probed components, in selection order: how active
    each one is at that position, independent of what removing it does. Its own jit — run beside
    the logits it would not fit on one 45 GB card."""

    @eqx.filter_jit
    def probe_ci(
        placed: PlacedModel, ci_fn: PlacedCIFn, tokens: Int[Array, "B T"]
    ) -> Float[Array, "B T K"]:
        captures = placed.clean_forward(tokens, capture_keys=ci_fn.capture_keys).captures
        lower = output_ci(placed, ci_fn, UNCONSTRAINED, captures, remat=False).lower
        ci = jnp.stack(
            [lower[site][:, :, component].astype(jnp.float32) for site, component in selection],
            axis=-1,
        )
        return jax.sharding.reshard(ci, P())

    return probe_ci


def make_probe_baseline() -> Callable[..., ProbeBaseline]:
    @eqx.filter_jit
    def probe_baseline(
        placed: PlacedModel, prepared: Prepared, tokens: Int[Array, "B T"], keep: dict[str, Array]
    ) -> ProbeBaseline:
        clean_logits = placed.clean_forward(tokens).output
        identity = subtracting(placed, prepared, tokens, keep)
        return ProbeBaseline(
            logits=identity,
            top1=jax.sharding.reshard(jnp.argmax(identity, axis=-1), P()),
            clean_kl=jax.sharding.reshard(_per_position_kl(clean_logits, identity), P()),
            clean_top1=jax.sharding.reshard(jnp.argmax(clean_logits, axis=-1), P()),
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
        logits = subtracting(placed, prepared, tokens, keep)
        out = {
            "kl": _per_position_kl(baseline.logits, logits),
            "top1": jnp.argmax(logits, axis=-1),
        }
        return {k: jax.sharding.reshard(v, P()) for k, v in out.items()}

    return probe_component
