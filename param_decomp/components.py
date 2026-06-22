"""The decomposition representation, shared by every target (LM and toy alike).

`DecompVU` is the trainable per-site V/U master pytree; `init_decomp_vu` seeds it;
`site_out` is the one decomposed-linear primitive (SPEC §4.1, `((x@V)*m)@U + (x@Δ)*d`).
These are domain-neutral — they depend only on `SiteSpec` and the V/U/W arrays — so they
live here rather than inside any one target (previously they lived in `llama8b.py`, which
made even the positionless toys import from `llama8b`).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from param_decomp.lm import SiteSpec


class DecompVU(eqx.Module):
    """fp32 master V `(d_in, C_s)` / U `(C_s, d_out)` per decomposed site, keyed by
    site name."""

    vu: dict[str, tuple[Float[Array, "d_in C"], Float[Array, "C d_out"]]]

    def site(self, name: str) -> tuple[Array, Array]:
        return self.vu[name]


def init_decomp_vu(sites: tuple[SiteSpec, ...], key: Array) -> DecompVU:
    """Small random fp32 V ~ N(0, d_in^-0.5), U ~ N(0, C^-0.5) per site; the
    weight-delta channel carries the faithfulness residual at init (before
    faithfulness warmup)."""
    keys = jax.random.split(key, 2 * len(sites))
    vu: dict[str, tuple[Array, Array]] = {}
    for site_idx, spec in enumerate(sites):
        V = jax.random.normal(keys[2 * site_idx], (spec.d_in, spec.C)) * spec.d_in**-0.5
        U = jax.random.normal(keys[2 * site_idx + 1], (spec.C, spec.d_out)) * spec.C**-0.5
        vu[spec.name] = (V, U)
    return DecompVU(vu=vu)


def site_out(
    x: Array,
    V: Array,
    U: Array,
    W: Array,
    mask: Array | None,
    delta_mask: Array | None,
    route: Array | None,
) -> Array:
    """One decomposed linear (SPEC §4.1): `((x@V)*m)@U + (x@Δ)*d`, routed per position
    against the frozen `x @ W.T`. `mask` may be None (fully on); `route` None routes
    everywhere. `delta_mask` None drops the delta path entirely (constant-source entries
    carry no delta, LOSS_PARITY_DESIGN §4b). `delta_mask`/`route` broadcast over batch;
    trailing dim added here."""
    acts = x @ V
    if mask is not None:
        acts = acts * mask
    out = acts @ U
    if delta_mask is not None:
        # DIVERGENCE vs the autocast oracle (documented, accepted): this delta is
        # computed in bf16 from the cast components; the oracle computes W − V@U in fp32 then
        # casts at the einsum. bf16-rounding-level difference on the delta PATH only —
        # the faithfulness loss uses the fp32 `weight_deltas` (SPEC N2), not this.
        delta = W - (V @ U).T  # (d_out, d_in)
        out = out + delta_mask[..., None] * (x @ delta.T)
    if route is not None:
        out = jnp.where(route[..., None], out, x @ W.T)
    return out
