"""Single-pool VPD on a real model: decompose the (square) q/k/v sites of a
tiny transformer (CPU-runnable).

This connects the model-agnostic single-pool step to an actual Equinox model
(`nano_pd_jax.TinyTransformer`) instead of synthetic einsums. The step's
stacked-site layout needs *homogeneous* site shapes, so we decompose the
`attn.{q,k,v}` matrices (all `d_model -> d_model` square) across every layer.

Pipeline per step:
  1. Frozen target forward over the batch -> per-site pre-weight acts
     `acts[site]: [B, seq, d_model]` (the input to each q/k/v matmul).
  2. Stack acts + the sites' `(V, U, W_target)` along the site axis S.
  3. Run the single-pool VPD+PGD step (layerwise recon per site).

The target's attention/MLP never re-runs masked — layerwise recon is site-local,
so the step only needs the pre-weight acts + the per-site weights. This is the
same factoring the production LM workload uses (decompose a fixed weight set;
reconstruct each site's output).
"""

import jax
import jax.numpy as jnp
import optax
from nano_pd_jax.decomposed import get_by_path
from nano_pd_jax.linear import Linear
from nano_pd_jax.transformer import TinyTransformer

from jax_single_pool.losses import CIParams, Decomposition
from jax_single_pool.pgd import PGDConfig, init_pgd_state
from jax_single_pool.scopes import BroadcastAcrossBatchScope
from jax_single_pool.step import LossCoeffs, init_train_state, make_step

VOCAB = 256
SEQ = 16
BATCH = 16
D_MODEL = 32
N_HEADS = 4
D_HEAD = 8
D_FF = 64
N_LAYERS = 2
C = 8
USE_DELTA = True

N_STEPS = 300
LOG_EVERY = 50

COEFFS = LossCoeffs(faith=1.0, imp=1e-2, stoch=1.0, ppgd=1.0, p_imp=0.9)
PGD_CFG = PGDConfig(
    lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, n_warmup=3, use_delta_component=USE_DELTA
)
LR_MAIN = 3e-3
LR_CI = 3e-3


def qkv_site_paths(n_layers: int) -> list[str]:
    return [f"blocks.{i}.attn.{name}" for i in range(n_layers) for name in ("q", "k", "v")]


def build_target() -> TinyTransformer:
    return TinyTransformer(
        vocab=VOCAB,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        key=jax.random.PRNGKey(0),
    )


def collect_qkv_weights(model: TinyTransformer, site_paths: list[str]) -> Decomposition:
    """Stack each q/k/v matrix `W_target = inner.weight.T` (`[d_model, d_model]`)
    along the site axis, initialising V/U from scratch."""
    Vs, Us, Ws = [], [], []
    key = jax.random.PRNGKey(1)
    for path in site_paths:
        layer = get_by_path(model, path)
        assert isinstance(layer, Linear)
        W = layer.inner.weight.T  # [d_in, d_out]
        d_in, d_out = W.shape
        kV, kU, key = jax.random.split(key, 3)
        Vs.append(jax.random.normal(kV, (d_in, C)) / jnp.sqrt(d_in))
        Us.append(jax.random.normal(kU, (C, d_out)) / jnp.sqrt(C))
        Ws.append(W)
    return Decomposition(V=jnp.stack(Vs), U=jnp.stack(Us), W_target=jnp.stack(Ws))


def init_ci(site_paths: list[str]) -> CIParams:
    """One linear CI head per site (d_model -> C). Stacked along S."""
    ws, bs = [], []
    key = jax.random.PRNGKey(2)
    for _ in site_paths:
        kw, key = jax.random.split(key)
        ws.append(jax.random.normal(kw, (D_MODEL, C)) * 0.1)
        bs.append(jnp.zeros((C,)))
    return CIParams(w=jnp.stack(ws), b=jnp.stack(bs))


def stacked_acts(model: TinyTransformer, tokens: jax.Array, site_paths: list[str]) -> jax.Array:
    """Run the frozen target over the batch; stack per-site pre-weight acts.

    Returns `x: [S, B, seq, d_model]` (frozen)."""
    _, acts = jax.vmap(lambda t: model(t, None))(tokens)  # acts[site]: [B, seq, d_in]
    return jax.lax.stop_gradient(jnp.stack([acts[p] for p in site_paths]))


def main() -> None:
    model = build_target()
    site_paths = qkv_site_paths(N_LAYERS)
    decomp = collect_qkv_weights(model, site_paths)
    ci = init_ci(site_paths)
    source_c = C + 1 if USE_DELTA else C
    n_sites = len(site_paths)

    key = jax.random.PRNGKey(10)
    key, k_tok = jax.random.split(key)
    tokens = jax.random.randint(k_tok, (BATCH, SEQ), 0, VOCAB)
    x = stacked_acts(model, tokens, site_paths)  # [S, B, seq, d_model]
    batch_dims = x.shape[1:-1]  # (B, seq)

    key, k_pgd = jax.random.split(key)
    pgd = init_pgd_state(k_pgd, BroadcastAcrossBatchScope(), n_sites, source_c, batch_dims)

    opt_main = optax.adam(LR_MAIN)
    opt_ci = optax.adam(LR_CI)
    state = init_train_state(decomp, ci, pgd, opt_main, opt_ci)
    step = make_step(
        COEFFS, PGD_CFG, opt_main, opt_ci, source_c=source_c, use_delta_component=USE_DELTA
    )

    print(f"sites ({n_sites}): {site_paths}")
    print(f"\n{'step':>6} {'total':>10} {'faith':>10} {'imp':>10} {'stoch':>10} {'ppgd':>10}")
    state, m = step(state, x, jax.random.fold_in(key, 0))
    first = {k: float(v) for k, v in m.items()}
    for i in range(N_STEPS):
        key, k_tok, k_step = jax.random.split(key, 3)
        tokens = jax.random.randint(k_tok, (BATCH, SEQ), 0, VOCAB)
        x = stacked_acts(model, tokens, site_paths)
        state, m = step(state, x, k_step)
        if i % LOG_EVERY == 0 or i == N_STEPS - 1:
            print(
                f"{i:>6} {float(m['total']):>10.5f} {float(m['faith']):>10.5f} "
                f"{float(m['imp']):>10.5f} {float(m['stoch']):>10.5f} {float(m['ppgd']):>10.5f}"
            )
    last = {k: float(v) for k, v in m.items()}
    assert last["faith"] < first["faith"], (first["faith"], last["faith"])
    print(
        f"\nPASS: faith {first['faith']:.4e} -> {last['faith']:.4e}, "
        f"stoch {first['stoch']:.4e} -> {last['stoch']:.4e}, "
        f"ppgd {first['ppgd']:.4e} -> {last['ppgd']:.4e}"
    )


if __name__ == "__main__":
    main()
