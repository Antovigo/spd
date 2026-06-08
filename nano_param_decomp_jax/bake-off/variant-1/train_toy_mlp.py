"""Random teacher 2-layer MLP: d=64, d_ff=128, output_dim=32.
Student decomposes all 4 sites (l1_up, l1_down, l2_up, l2_down) with C=16.

Verifies multi-site composition. We just check recon loss goes down.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from ci_fn import SiteSpec, make_ci_fn
from jaxtyping import Array, Float, PRNGKeyArray
from trainer import train

D_MODEL = 64
D_FF = 128
D_OUT = 32
C = 16
N_STEPS = 5000
BATCH = 256


def make_teacher_params(key: PRNGKeyArray) -> dict[str, Array]:
    k = jax.random.split(key, 8)
    return {
        "l1_up": jax.random.normal(k[0], (D_MODEL, D_FF)) / jnp.sqrt(D_MODEL),
        "l1_up_b": jnp.zeros((D_FF,)),
        "l1_down": jax.random.normal(k[1], (D_FF, D_MODEL)) / jnp.sqrt(D_FF),
        "l1_down_b": jnp.zeros((D_MODEL,)),
        "l2_up": jax.random.normal(k[2], (D_MODEL, D_FF)) / jnp.sqrt(D_MODEL),
        "l2_up_b": jnp.zeros((D_FF,)),
        "l2_down": jax.random.normal(k[3], (D_FF, D_OUT)) / jnp.sqrt(D_FF),
        "l2_down_b": jnp.zeros((D_OUT,)),
    }


def target_forward(
    params: dict[str, Array], x: Float[Array, "b d"]
) -> tuple[Float[Array, "b d_out"], dict[str, Float[Array, "b *"]]]:
    pre_acts: dict[str, Float[Array, "b *"]] = {}
    pre_acts["l1_up"] = x
    h1 = jax.nn.gelu(x @ params["l1_up"] + params["l1_up_b"])
    pre_acts["l1_down"] = h1
    r1 = h1 @ params["l1_down"] + params["l1_down_b"]
    pre_acts["l2_up"] = r1
    h2 = jax.nn.gelu(r1 @ params["l2_up"] + params["l2_up_b"])
    pre_acts["l2_down"] = h2
    out = h2 @ params["l2_down"] + params["l2_down_b"]
    return out, pre_acts


def _site_out(c: dict[str, Array], m: Array, x: Array) -> Array:
    return ((x @ c["V"]) * m) @ c["U"] + x @ c["W_delta"]


def decomposed_forward(
    params: dict[str, Array],
    components: dict[str, dict[str, Array]],
    masks: dict[str, Float[Array, "b C"]],
    x: Float[Array, "b d"],
) -> tuple[Float[Array, "b d_out"], dict[str, Float[Array, "b *"]]]:
    pre_acts: dict[str, Float[Array, "b *"]] = {}
    pre_acts["l1_up"] = x
    h1 = jax.nn.gelu(_site_out(components["l1_up"], masks["l1_up"], x) + params["l1_up_b"])
    pre_acts["l1_down"] = h1
    r1 = _site_out(components["l1_down"], masks["l1_down"], h1) + params["l1_down_b"]
    pre_acts["l2_up"] = r1
    h2 = jax.nn.gelu(_site_out(components["l2_up"], masks["l2_up"], r1) + params["l2_up_b"])
    pre_acts["l2_down"] = h2
    out = _site_out(components["l2_down"], masks["l2_down"], h2) + params["l2_down_b"]
    return out, pre_acts


def make_sample_batch() -> Callable[[PRNGKeyArray], Float[Array, "b d"]]:
    def sample(key: PRNGKeyArray) -> Float[Array, "b d"]:
        return jax.random.normal(key, (BATCH, D_MODEL))
    return sample


def main() -> None:
    key = jax.random.PRNGKey(0)
    teacher_key, ci_key, train_key = jax.random.split(key, 3)
    params = make_teacher_params(teacher_key)

    target_weights = {
        "l1_up": params["l1_up"],
        "l1_down": params["l1_down"],
        "l2_up": params["l2_up"],
        "l2_down": params["l2_down"],
    }
    sites = {
        "l1_up": SiteSpec(d_in=D_MODEL, d_out=D_FF, C=C),
        "l1_down": SiteSpec(d_in=D_FF, d_out=D_MODEL, C=C),
        "l2_up": SiteSpec(d_in=D_MODEL, d_out=D_FF, C=C),
        "l2_down": SiteSpec(d_in=D_FF, d_out=D_OUT, C=C),
    }
    ci_fn = make_ci_fn(sites, hidden_size=64, key=ci_key)

    components, ci_fn, _history = train(
        target_forward=target_forward,
        decomposed_forward=decomposed_forward,
        params=params,
        target_weights=target_weights,
        ci_fn=ci_fn,
        sample_batch=make_sample_batch(),
        n_steps=N_STEPS,
        key=train_key,
        c_per_site={n: C for n in sites},
        lr_vu=1e-3,
        lr_ci=1e-3,
        coeff_faith=1.0,
        coeff_imp=1e-3,
        coeff_stoch=1.0,
        imp_p=0.9,
        log_every=200,
    )


if __name__ == "__main__":
    main()
