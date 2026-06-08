"""Toy 2-layer MLP: 4 decomposable sites (layer1.up, layer1.down, layer2.up, layer2.down).

Random teacher MLP, gaussian inputs, decompose all 4 sites with C=16 each. Demonstrates
multi-site composition through the *same* polymorphic `linop`.
"""

import jax
import jax.numpy as jnp
from decomposed import linop
from jaxtyping import Array, Float, PRNGKeyArray
from trainer import TrainConfig, train

D_MODEL = 64
D_FF = 128
D_OUT = 32
BATCH_SIZE = 256
N_STEPS = 5000
C_PER_SITE = 16

SITES = ["layer1_up", "layer1_down", "layer2_up", "layer2_down"]


def model_forward(
    params: dict,
    x: Float[Array, "B d_model"],
    masks: dict[str, Float[Array, "B C"]] | None = None,
) -> tuple[Float[Array, "B d_out"], dict[str, Float[Array, "B d_in"]]]:
    """2-layer MLP written ONCE. Same code paths for target & decomposed via linop."""
    pre_acts: dict[str, Float[Array, "B d_in"]] = {}

    pre_acts["layer1_up"] = x
    m = masks.get("layer1_up") if masks is not None else None
    h1 = jax.nn.gelu(linop(params["layer1_up"], x, m) + params["b1_up"])

    pre_acts["layer1_down"] = h1
    m = masks.get("layer1_down") if masks is not None else None
    z1 = linop(params["layer1_down"], h1, m) + params["b1_down"]

    pre_acts["layer2_up"] = z1
    m = masks.get("layer2_up") if masks is not None else None
    h2 = jax.nn.gelu(linop(params["layer2_up"], z1, m) + params["b2_up"])

    pre_acts["layer2_down"] = h2
    m = masks.get("layer2_down") if masks is not None else None
    y = linop(params["layer2_down"], h2, m) + params["b2_down"]

    return y, pre_acts


def init_teacher_params(key: PRNGKeyArray) -> dict[str, Array]:
    keys = jax.random.split(key, 8)
    return {
        "layer1_up":    jax.random.normal(keys[0], (D_MODEL, D_FF))  / jnp.sqrt(D_MODEL),
        "b1_up":        jnp.zeros((D_FF,)),
        "layer1_down":  jax.random.normal(keys[1], (D_FF, D_MODEL))  / jnp.sqrt(D_FF),
        "b1_down":      jnp.zeros((D_MODEL,)),
        "layer2_up":    jax.random.normal(keys[2], (D_MODEL, D_FF))  / jnp.sqrt(D_MODEL),
        "b2_up":        jnp.zeros((D_FF,)),
        "layer2_down":  jax.random.normal(keys[3], (D_FF, D_OUT))    / jnp.sqrt(D_FF),
        "b2_down":      jnp.zeros((D_OUT,)),
    }


def make_data_fn(batch_size: int = BATCH_SIZE):
    def data_fn(key: PRNGKeyArray) -> Float[Array, "B d_model"]:
        return jax.random.normal(key, (batch_size, D_MODEL))
    return data_fn


def main() -> None:
    key = jax.random.PRNGKey(1)
    k_init, k_train = jax.random.split(key)
    target_params = init_teacher_params(k_init)
    c_per_site = {name: C_PER_SITE for name in SITES}
    cfg = TrainConfig(
        n_steps=N_STEPS,
        main_lr=1e-3,
        ci_lr=1e-3,
        coeff_faith=1.0,
        coeff_imp=1e-3,
        coeff_stoch=1.0,
        log_every=200,
    )
    print(f"Training toy MLP decomposition: d_model={D_MODEL} d_ff={D_FF} d_out={D_OUT}")
    print(f"Sites: {SITES}, C={C_PER_SITE} per site, steps: {cfg.n_steps}")
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
