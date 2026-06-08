"""TMS 5->2->5 with sparse binary inputs. Decompose W1 (5x2) and W2 (2x5), C=5 each.

After training, print the top-K-activating feature per component, expected to
align one-feature-per-component (the disentanglement check).
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from ci_fn import SiteSpec, apply_ci_fn, make_ci_fn
from jaxtyping import Array, Float, PRNGKeyArray
from trainer import train

N_FEATURES = 5
HIDDEN = 2
C = 5
N_STEPS = 5000
BATCH = 1024
P_ACTIVE = 0.1


def make_tms_params(key: PRNGKeyArray) -> dict[str, Array]:
    """Train a TMS to convergence so it has interesting structure to decompose.

    Standard TMS: encoder maps R^5 -> R^2, decoder is encoder.T, relu on the output.
    Importance is uniform across features here for simplicity.
    """
    k1, k2 = jax.random.split(key)
    W1 = jax.random.normal(k1, (N_FEATURES, HIDDEN)) * 0.5
    b1 = jnp.zeros((HIDDEN,))
    W2 = W1.T  # tied
    b2 = jnp.zeros((N_FEATURES,))
    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    import optax
    opt = optax.adam(1e-3)
    state = opt.init(params)

    def loss_fn(p: dict[str, Array], x: Float[Array, "b d"]) -> Float[Array, ""]:
        h = jax.nn.relu(x @ p["W1"] + p["b1"])
        y = jax.nn.relu(h @ p["W2"] + p["b2"])
        return jnp.mean((y - x) ** 2)

    @jax.jit
    def pretrain_step(p, state, x):
        loss, grads = jax.value_and_grad(loss_fn)(p, x)
        updates, state = opt.update(grads, state, p)
        p = optax.apply_updates(p, updates)
        return p, state, loss

    key2 = jax.random.PRNGKey(123)
    for step in range(3000):
        key2, k = jax.random.split(key2)
        x = (jax.random.uniform(k, (BATCH, N_FEATURES)) < P_ACTIVE).astype(jnp.float32) * \
            jax.random.uniform(jax.random.fold_in(k, 1), (BATCH, N_FEATURES))
        params, state, loss = pretrain_step(params, state, x)
    print(f"TMS pretrain final reconstruction loss: {float(loss):.4e}")
    return params


def target_forward(
    params: dict[str, Array], x: Float[Array, "b d_in"]
) -> tuple[Float[Array, "b d_out"], dict[str, Float[Array, "b d_in"]]]:
    pre_acts: dict[str, Float[Array, "b *"]] = {}
    pre_acts["W1"] = x
    h_pre = x @ params["W1"] + params["b1"]
    h = jax.nn.relu(h_pre)
    pre_acts["W2"] = h
    y_pre = h @ params["W2"] + params["b2"]
    y = jax.nn.relu(y_pre)
    return y, pre_acts


def decomposed_forward(
    params: dict[str, Array],
    components: dict[str, dict[str, Array]],
    masks: dict[str, Float[Array, "b C"]],
    x: Float[Array, "b d_in"],
) -> tuple[Float[Array, "b d_out"], dict[str, Float[Array, "b d_in"]]]:
    pre_acts: dict[str, Float[Array, "b *"]] = {}
    pre_acts["W1"] = x
    c1 = components["W1"]
    h_pre = ((x @ c1["V"]) * masks["W1"]) @ c1["U"] + x @ c1["W_delta"] + params["b1"]
    h = jax.nn.relu(h_pre)
    pre_acts["W2"] = h
    c2 = components["W2"]
    y_pre = ((h @ c2["V"]) * masks["W2"]) @ c2["U"] + h @ c2["W_delta"] + params["b2"]
    y = jax.nn.relu(y_pre)
    return y, pre_acts


def make_sample_batch(p_active: float) -> Callable[[PRNGKeyArray], Float[Array, "b d"]]:
    def sample(key: PRNGKeyArray) -> Float[Array, "b d"]:
        k_active, k_mag = jax.random.split(key)
        active = (jax.random.uniform(k_active, (BATCH, N_FEATURES)) < p_active).astype(jnp.float32)
        mag = jax.random.uniform(k_mag, (BATCH, N_FEATURES))
        return active * mag
    return sample


def main() -> None:
    key = jax.random.PRNGKey(0)
    params_key, ci_key, train_key, eval_key = jax.random.split(key, 4)

    params = make_tms_params(params_key)

    sites = {
        "W1": SiteSpec(d_in=N_FEATURES, d_out=HIDDEN, C=C),
        "W2": SiteSpec(d_in=HIDDEN, d_out=N_FEATURES, C=C),
    }
    target_weights = {"W1": params["W1"], "W2": params["W2"]}
    ci_fn = make_ci_fn(sites, hidden_size=32, key=ci_key)

    components, ci_fn, _history = train(
        target_forward=target_forward,
        decomposed_forward=decomposed_forward,
        params=params,
        target_weights=target_weights,
        ci_fn=ci_fn,
        sample_batch=make_sample_batch(P_ACTIVE),
        n_steps=N_STEPS,
        key=train_key,
        c_per_site={"W1": C, "W2": C},
        lr_vu=1e-3,
        lr_ci=1e-3,
        coeff_faith=1.0,
        coeff_imp=3e-3,
        coeff_stoch=1.0,
        imp_p=0.9,
        log_every=200,
    )

    # Disentanglement check: for each one-hot feature input, see which components fire.
    one_hots = jnp.eye(N_FEATURES)
    _, pre_acts = target_forward(params, one_hots)
    cis = apply_ci_fn(ci_fn, pre_acts)
    print("\nCI[W1] per feature (rows=features, cols=components):")
    print(np.round(np.asarray(cis["W1"]), 2))
    print("\nCI[W2] per feature (rows=features, cols=components):")
    print(np.round(np.asarray(cis["W2"]), 2))
    print("\nTop-activating feature per component (W1):")
    top_per_comp = jnp.argmax(cis["W1"], axis=0)
    print(np.asarray(top_per_comp))


if __name__ == "__main__":
    main()
