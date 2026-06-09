"""Single-pool VPD smoke on a synthetic stacked-site target (CPU-runnable).

A bank of `S` random linear "sites" (homogeneous d_in=d_out=d), each a frozen
target `W_target`. We decompose every site into rank-C components and train all
four VPD losses + the persistent-PGD adversary with the single-pool step.

Not a semantic task — the point is to exercise the full step end to end and show
the losses (esp. faithfulness and the recon terms) trend down while the
adversary pushes the worst-case recon up relative to the stochastic one.
"""

import jax
import jax.numpy as jnp
import optax

from jax_single_pool.losses import CIParams, Decomposition
from jax_single_pool.pgd import PGDConfig, PGDState, init_pgd_state
from jax_single_pool.scopes import BroadcastAcrossBatchScope
from jax_single_pool.step import LossCoeffs, init_train_state, make_step

S = 6
D = 32
C = 8
B = 64
N_STEPS = 400
LOG_EVERY = 50
USE_DELTA = True

COEFFS = LossCoeffs(faith=1.0, imp=1e-2, stoch=1.0, ppgd=1.0, p_imp=0.9)
PGD_CFG = PGDConfig(
    lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8, n_warmup=4, use_delta_component=USE_DELTA
)
LR_MAIN = 3e-3
LR_CI = 3e-3


def init_everything(
    key: jax.Array,
) -> tuple[Decomposition, CIParams, PGDState, int]:
    kW, kV, kU, kciw, kcib, kpgd = jax.random.split(key, 6)
    sc = 0.3
    W_target = jax.random.normal(kW, (S, D, D)) * sc
    V = jax.random.normal(kV, (S, D, C)) / jnp.sqrt(D)
    U = jax.random.normal(kU, (S, C, D)) / jnp.sqrt(C)
    decomp = Decomposition(V=V, U=U, W_target=W_target)
    ci = CIParams(
        w=jax.random.normal(kciw, (S, D, C)) * 0.1,
        b=jax.random.normal(kcib, (S, C)) * 0.1,
    )
    source_c = C + 1 if USE_DELTA else C
    pgd = init_pgd_state(
        kpgd, BroadcastAcrossBatchScope(), n_sites=S, source_c=source_c, batch_dims=(B,)
    )
    return decomp, ci, pgd, source_c


def main():
    key = jax.random.PRNGKey(0)
    key, k_init = jax.random.split(key)
    decomp, ci, pgd, source_c = init_everything(k_init)

    opt_main = optax.adam(LR_MAIN)
    opt_ci = optax.adam(LR_CI)
    state = init_train_state(decomp, ci, pgd, opt_main, opt_ci)

    step = make_step(
        COEFFS, PGD_CFG, opt_main, opt_ci, source_c=source_c, use_delta_component=USE_DELTA
    )

    # frozen per-site pre-weight acts: stacked [S, B, D]
    key, kx = jax.random.split(key)
    x = jax.random.normal(kx, (S, B, D))

    print(f"sites={S} d={D} C={C} B={B} delta={USE_DELTA}")
    print(f"\n{'step':>6} {'total':>10} {'faith':>10} {'imp':>10} {'stoch':>10} {'ppgd':>10}")
    state, m = step(state, x, jax.random.fold_in(key, 0))
    first = {k: float(v) for k, v in m.items()}
    for i in range(N_STEPS):
        key, ks = jax.random.split(key)
        state, m = step(state, x, ks)
        if i % LOG_EVERY == 0 or i == N_STEPS - 1:
            print(
                f"{i:>6} {float(m['total']):>10.5f} {float(m['faith']):>10.5f} "
                f"{float(m['imp']):>10.5f} {float(m['stoch']):>10.5f} {float(m['ppgd']):>10.5f}"
            )
    last = {k: float(v) for k, v in m.items()}
    assert last["faith"] < first["faith"], (first["faith"], last["faith"])
    assert last["stoch"] < first["stoch"], (first["stoch"], last["stoch"])
    print(
        f"\nPASS: faith {first['faith']:.4e} -> {last['faith']:.4e}, "
        f"stoch {first['stoch']:.4e} -> {last['stoch']:.4e}, "
        f"ppgd {first['ppgd']:.4e} -> {last['ppgd']:.4e}"
    )


if __name__ == "__main__":
    main()
