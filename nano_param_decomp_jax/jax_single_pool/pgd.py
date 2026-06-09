"""Persistent-PGD adversary — functional, carried as state across steps.

Mirrors `param_decomp/metrics/persistent_pgd_state.py` but functional (no
in-place mutation): the adversarial `sources` and the PGD-Adam optimizer moments
live in `PGDState`, carried in the trainer's `TrainState` and threaded through
the jit'd step. Source updates per training step = `n_warmup + 1`:

  * `n_warmup` supplemental source-only ascent iters (`pgd_warmup`, a lax.scan),
  * one more ascent inside the fused outer grad (the final fwd+bwd), persisted.

The adversary MAXimizes layerwise recon over `sources`; params MINimize it. The
effective source is `sigmoid(raw)` (sigmoid parameterization) so it stays in
[0,1] unbounded — matching the production `use_sigmoid_parameterization` path.

Under SPMD data-parallelism a shared (single/broadcast/repeat) source is
replicated; its grad is the mean over the sharded batch, which `jax.jit`
all-reduces automatically. No `replica_sync_group` broadcast/AVG-reduce.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from jax_single_pool.losses import Decomposition, interpolate_mask, layerwise_recon_loss
from jax_single_pool.scopes import SourceScope, expand_source_to_batch, source_leading_dims


class PGDAdamState(NamedTuple):
    m: Float[Array, "S ... source_c"]
    v: Float[Array, "S ... source_c"]
    step_count: Float[Array, ""]


class PGDState(NamedTuple):
    sources: Float[Array, "S ... source_c"]  # raw (pre-sigmoid) adversarial sources
    adam: PGDAdamState


class PGDConfig(NamedTuple):
    lr: float
    beta1: float
    beta2: float
    eps: float
    n_warmup: int
    use_delta_component: bool


def init_pgd_state(
    key: PRNGKeyArray,
    scope: SourceScope,
    n_sites: int,
    source_c: int,
    batch_dims: tuple[int, ...],
) -> PGDState:
    leading = source_leading_dims(scope, batch_dims)
    shape = (n_sites, *leading, source_c)
    sources = jax.random.normal(key, shape)  # sigmoid-parameterized: start near 0.5
    zeros = jnp.zeros(shape)
    return PGDState(sources=sources, adam=PGDAdamState(m=zeros, v=zeros, step_count=jnp.array(0.0)))


def adversarial_mask(
    ci: Float[Array, "S ... C"],
    raw_source: Float[Array, "S ... source_c"],
    batch_dims: tuple[int, ...],
    use_delta_component: bool,
) -> Float[Array, "S ... source_c"]:
    """mask = ci + (1-ci)*sigmoid(source), with the scoped source expanded to the batch."""
    eff = jax.nn.sigmoid(raw_source)
    eff_full = jax.vmap(lambda s: expand_source_to_batch(s, batch_dims))(eff)
    return interpolate_mask(ci, eff_full, use_delta_component)


def adversarial_recon(
    decomp: Decomposition,
    x: Float[Array, "S ... d_in"],
    ci: Float[Array, "S ... C"],
    raw_source: Float[Array, "S ... source_c"],
    batch_dims: tuple[int, ...],
    use_delta_component: bool,
) -> Float[Array, ""]:
    mask = adversarial_mask(ci, raw_source, batch_dims, use_delta_component)
    return layerwise_recon_loss(decomp, x, mask, use_delta_component)


def _adam_ascend(
    raw_source: Float[Array, "..."],
    grad: Float[Array, "..."],
    adam: PGDAdamState,
    cfg: PGDConfig,
) -> tuple[Float[Array, "..."], PGDAdamState]:
    """One Adam ascent step on the source (maximizing recon → +grad direction)."""
    step_count = adam.step_count + 1.0
    m = cfg.beta1 * adam.m + (1.0 - cfg.beta1) * grad
    v = cfg.beta2 * adam.v + (1.0 - cfg.beta2) * grad * grad
    m_hat = m / (1.0 - cfg.beta1**step_count)
    v_hat = v / (1.0 - cfg.beta2**step_count)
    new_source = raw_source + cfg.lr * m_hat / (jnp.sqrt(v_hat) + cfg.eps)
    return new_source, PGDAdamState(m=m, v=v, step_count=step_count)


def pgd_warmup(
    decomp: Decomposition,
    x: Float[Array, "S ... d_in"],
    ci: Float[Array, "S ... C"],
    pgd: PGDState,
    batch_dims: tuple[int, ...],
    cfg: PGDConfig,
) -> PGDState:
    """`n_warmup` supplemental source-only ascent iters via lax.scan.

    Params + ci are detached (the adversary only moves sources). Bit-exact to a
    python loop (cf. `jax_spike/stage6_pgd.py` check (a)).
    """
    decomp_det = jax.tree.map(jax.lax.stop_gradient, decomp)
    ci_det = jax.lax.stop_gradient(ci)
    x_det = jax.lax.stop_gradient(x)

    def body(carry: tuple[Array, PGDAdamState], _: None) -> tuple[tuple[Array, PGDAdamState], None]:
        source, adam = carry
        grad = jax.grad(
            lambda s: adversarial_recon(
                decomp_det, x_det, ci_det, s, batch_dims, cfg.use_delta_component
            )
        )(source)
        new_source, new_adam = _adam_ascend(source, grad, adam, cfg)
        return (new_source, new_adam), None

    (final_source, final_adam), _ = jax.lax.scan(
        body, (pgd.sources, pgd.adam), None, length=cfg.n_warmup
    )
    return PGDState(sources=final_source, adam=final_adam)


def pgd_final_ascend(
    decomp: Decomposition,
    x: Float[Array, "S ... d_in"],
    ci: Float[Array, "S ... C"],
    pgd: PGDState,
    batch_dims: tuple[int, ...],
    cfg: PGDConfig,
) -> PGDState:
    """The (n_warmup+1)-th source ascent, applied after the params update so the
    persisted sources are warm-started against the fresh params (matches the
    torch `after_backward` step ordering)."""
    decomp_det = jax.tree.map(jax.lax.stop_gradient, decomp)
    ci_det = jax.lax.stop_gradient(ci)
    x_det = jax.lax.stop_gradient(x)
    grad = jax.grad(
        lambda s: adversarial_recon(
            decomp_det, x_det, ci_det, s, batch_dims, cfg.use_delta_component
        )
    )(pgd.sources)
    new_source, new_adam = _adam_ascend(pgd.sources, grad, pgd.adam, cfg)
    return PGDState(sources=new_source, adam=new_adam)
