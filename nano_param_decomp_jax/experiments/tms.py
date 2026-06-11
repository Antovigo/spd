"""TMS 5 → 2 → 5: sparse binary inputs through a 2-dim bottleneck.

Two stages:
1. Pretrain TMSModel on sparse binary inputs (recovers feature embeddings).
2. Decompose pretrained model, train decomposition for N_STEPS.

Decomposes both Linear sites (layer1: 5→2, layer2: 2→5) with C=5 each.
Expected: faith → ~1e-5 range, stoch → ~1e-4 range, 2-3 components align
with single features (TMS bottleneck artifact).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, PRNGKeyArray
from nano_pd_jax.ci_fn import CIFn
from nano_pd_jax.decomposed import collect_site_paths, substitute_decomposed
from nano_pd_jax.linear import Linear
from nano_pd_jax.trainer import current_model, init_state, make_step_fn

N_FEATURES = 5
HIDDEN = 2
P_ACTIVE = 0.1
BATCH_SIZE = 1024
N_STEPS = 5000
LOG_EVERY = 250
PRETRAIN_STEPS = 5000
PRETRAIN_LOG_EVERY = 1000
C = 5

COEFF_FAITH = 1.0
COEFF_IMP = 1e-3
COEFF_STOCH = 1.0
P_VALUE = 0.9
LR = 1e-3
CI_HIDDEN = 32


class TMSModel(eqx.Module):
    layer1: Linear
    layer2: Linear

    def __init__(self, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        self.layer1 = Linear(N_FEATURES, HIDDEN, key=k1)
        self.layer2 = Linear(HIDDEN, N_FEATURES, key=k2)

    def __call__(
        self,
        x: Float[Array, " 5"],
        masks: dict[str, Float[Array, " C"]] | None = None,
    ) -> tuple[Float[Array, " 5"], dict[str, Float[Array, "..."]]]:
        m1 = masks.get("layer1") if masks is not None else None
        m2 = masks.get("layer2") if masks is not None else None
        h, a1 = self.layer1(x, m1)
        h = jax.nn.relu(h)
        out, a2 = self.layer2(h, m2)
        return out, {"layer1": a1, "layer2": a2}


def sample_batch(key: PRNGKeyArray, batch_size: int) -> Float[Array, "B 5"]:
    return (jax.random.uniform(key, (batch_size, N_FEATURES)) < P_ACTIVE).astype(jnp.float32)


def pretrain_tms(key: PRNGKeyArray, n_steps: int = PRETRAIN_STEPS) -> TMSModel:
    """Standard TMS pretraining: reconstruct sparse input through 2D bottleneck."""
    key_init, key = jax.random.split(key)
    model = TMSModel(key=key_init)
    opt = optax.adam(1e-2)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step(model, opt_state, x):
        def loss_fn(m):
            pred, _ = jax.vmap(m)(x)
            return jnp.mean((pred - x) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    print(f"{'pretrain step':>14} {'recon':>10}")
    for i in range(n_steps):
        key, sub = jax.random.split(key)
        x = sample_batch(sub, BATCH_SIZE)
        model, opt_state, loss = step(model, opt_state, x)
        if i % PRETRAIN_LOG_EVERY == 0 or i == n_steps - 1:
            print(f"{i:>14} {float(loss):>10.4e}")
    return model


def main() -> None:
    key = jax.random.PRNGKey(0)
    key_pre, key_decomp, key_ci, key = jax.random.split(key, 4)

    print("=== Stage 1: pretrain TMS target ===")
    target = pretrain_tms(key_pre)

    print("\n=== Stage 2: decompose ===")
    decomposed = substitute_decomposed(target, {"layer1": C, "layer2": C}, key=key_decomp)
    site_paths = collect_site_paths(decomposed)
    assert site_paths == ["layer1", "layer2"], site_paths

    ci_fn = CIFn(
        d_in_per_site={"layer1": N_FEATURES, "layer2": HIDDEN},
        C_per_site={"layer1": C, "layer2": C},
        hidden=CI_HIDDEN,
        key=key_ci,
    )

    opt_main = optax.adam(LR)
    opt_ci = optax.adam(LR)
    state = init_state(decomposed, ci_fn, opt_main, opt_ci)

    step_fn = make_step_fn(
        site_paths=site_paths,
        coeff_faith=COEFF_FAITH,
        coeff_imp=COEFF_IMP,
        coeff_stoch=COEFF_STOCH,
        p_value=P_VALUE,
        opt_main=opt_main,
        opt_ci=opt_ci,
    )

    print(f"\n{'step':>6} {'total':>10} {'faith':>10} {'imp':>10} {'stoch':>10}")
    for step_i in range(N_STEPS):
        key, sub_data, sub_step = jax.random.split(key, 3)
        x = sample_batch(sub_data, BATCH_SIZE)
        state, losses = step_fn(state, x, sub_step)
        if step_i % LOG_EVERY == 0 or step_i == N_STEPS - 1:
            print(
                f"{step_i:>6} "
                f"{float(losses['total']):>10.5f} "
                f"{float(losses['faith']):>10.5f} "
                f"{float(losses['imp']):>10.5f} "
                f"{float(losses['stoch']):>10.5f}"
            )

    final = current_model(state)
    print(f"\nfinal V (layer1) shape: {final.layer1.V.shape}")
    print(f"final V (layer1):\n{final.layer1.V}")


if __name__ == "__main__":
    main()
