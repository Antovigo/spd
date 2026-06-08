"""TMS (Toy Model of Superposition): 5 -> 2 -> 5, decompose W1 and W2 with C=5 each.

The target reconstructs sparse binary features through a 2-dim bottleneck. The point
of decomposition: discover that 5 components ≈ the 5 underlying features.

The user-facing API is the `model_forward` below — written ONCE, factored through
`linop`. The trainer dispatches target vs decomposed mode purely by leaf type.
"""

import jax
import jax.numpy as jnp
from decomposed import linop
from jaxtyping import Array, Float, PRNGKeyArray
from trainer import TrainConfig, train

D_FEATURES = 5
D_BOTTLENECK = 2
P_ACTIVE = 0.1
BATCH_SIZE = 256
N_STEPS = 5000


def model_forward(
    params: dict, x: Float[Array, "B d"], masks: dict[str, Float[Array, "B C"]] | None = None
) -> tuple[Float[Array, "B d"], dict[str, Float[Array, "B d_in"]]]:
    """TMS forward written ONCE. Same code paths for target & decomposed via linop."""
    pre_acts: dict[str, Float[Array, "B d_in"]] = {}

    pre_acts["W1"] = x
    m1 = masks.get("W1") if masks is not None else None
    h = jax.nn.relu(linop(params["W1"], x, m1) + params["b1"])

    pre_acts["W2"] = h
    m2 = masks.get("W2") if masks is not None else None
    y = linop(params["W2"], h, m2) + params["b2"]

    return y, pre_acts


def init_target_params(key: PRNGKeyArray) -> dict[str, Array]:
    """A pre-trained-looking TMS solution: random init, then it's the trainer's job to
    reconstruct *this* — we don't actually pretrain a TMS, we just decompose a random
    target. (The reference TMS pretraining is orthogonal to the decomposition method
    being demoed.)"""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    # Use small Gaussian inits — same as what a freshly init'd nn.Linear would give.
    W1 = jax.random.normal(k1, (D_FEATURES, D_BOTTLENECK)) / jnp.sqrt(D_FEATURES)
    b1 = jnp.zeros((D_BOTTLENECK,))
    W2 = jax.random.normal(k2, (D_BOTTLENECK, D_FEATURES)) / jnp.sqrt(D_BOTTLENECK)
    b2 = jnp.zeros((D_FEATURES,))
    # Bias the target slightly: tie W2 = W1^T (TMS standard) and use the standard init.
    W2 = W1.T
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def make_data_fn(p_active: float = P_ACTIVE, batch_size: int = BATCH_SIZE):
    def data_fn(key: PRNGKeyArray) -> Float[Array, "B 5"]:
        return (jax.random.uniform(key, (batch_size, D_FEATURES)) < p_active).astype(jnp.float32)
    return data_fn


def main() -> None:
    key = jax.random.PRNGKey(0)
    k_init, k_train = jax.random.split(key)
    target_params = init_target_params(k_init)
    c_per_site = {"W1": D_FEATURES, "W2": D_FEATURES}  # C = 5 for both
    cfg = TrainConfig(
        n_steps=N_STEPS,
        main_lr=1e-3,
        ci_lr=1e-3,
        coeff_faith=1.0,
        coeff_imp=1e-2,
        coeff_stoch=1.0,
        log_every=200,
    )
    print(f"Training TMS decomposition: {D_FEATURES} features -> {D_BOTTLENECK} bottleneck, C={D_FEATURES} per site")
    print(f"Sites: {list(c_per_site)}, steps: {cfg.n_steps}")
    print("-" * 80)
    _, _, history = train(
        key=k_train,
        target_params=target_params,
        forward_fn=model_forward,
        data_fn=make_data_fn(),
        c_per_site=c_per_site,
        cfg=cfg,
    )
    print("-" * 80)
    print(f"Final losses: {history[-1]}")


if __name__ == "__main__":
    main()
