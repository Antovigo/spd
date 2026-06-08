"""Target A: TMS 5 -> 2 -> 5. Sparse binary inputs through a 2-dim bottleneck.

Decompose both Linear sites (layer1: 5->2, layer2: 2->5) with C=5 each.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from ci_fn import CIFn
from decomposed_linear import DecomposedLinear, substitute_decomposed
from jaxtyping import Array, Float, PRNGKeyArray
from trainer import collect_site_paths, current_model, init_state, make_step_fn

N_FEATURES = 5
HIDDEN = 2
P_ACTIVE = 0.1
BATCH_SIZE = 1024
N_STEPS = 5000
LOG_EVERY = 100
C = 5

COEFF_FAITH = 1.0
COEFF_IMP = 1e-3
COEFF_STOCH = 1.0
P_VALUE = 0.9
LR = 1e-3
CI_HIDDEN = 32


class TMSModel(eqx.Module):
    layer1: eqx.nn.Linear  # 5 -> 2
    layer2: eqx.nn.Linear  # 2 -> 5

    def __init__(self, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        self.layer1 = eqx.nn.Linear(N_FEATURES, HIDDEN, use_bias=True, key=k1)
        self.layer2 = eqx.nn.Linear(HIDDEN, N_FEATURES, use_bias=True, key=k2)

    def __call__(
        self,
        x: Float[Array, " 5"],
        masks: dict[str, Float[Array, " C"]] | None = None,
    ) -> Float[Array, " 5"]:
        m1 = masks["layer1"] if masks is not None else None
        m2 = masks["layer2"] if masks is not None else None
        h = _apply(self.layer1, x, m1)
        h = jax.nn.relu(h)
        return _apply(self.layer2, h, m2)

    def forward_with_acts(
        self, x: Float[Array, " 5"]
    ) -> tuple[Float[Array, " 5"], dict[str, Float[Array, "..."]]]:
        """Target-forward returning (out, pre-weight acts at each decomposed site)."""
        acts: dict[str, Float[Array, ...]] = {"layer1": x}
        h = _apply(self.layer1, x, None)
        h = jax.nn.relu(h)
        acts["layer2"] = h
        return _apply(self.layer2, h, None), acts


def _apply(layer, x, mask):
    """Call layer with mask if it's a DecomposedLinear, else without."""
    if isinstance(layer, DecomposedLinear):
        return layer(x, mask)
    return layer(x)


def sample_batch(key: PRNGKeyArray, batch_size: int) -> Float[Array, "B 5"]:
    return (jax.random.uniform(key, (batch_size, N_FEATURES)) < P_ACTIVE).astype(jnp.float32)


def main() -> None:
    key = jax.random.PRNGKey(0)
    key_model, key_decomp, key_ci, key = jax.random.split(key, 4)

    target = TMSModel(key=key_model)
    decomposed = substitute_decomposed(
        target, {"layer1": C, "layer2": C}, key=key_decomp
    )
    site_paths = collect_site_paths(decomposed)
    assert site_paths == ["layer1", "layer2"], site_paths

    d_in_per_site = {"layer1": N_FEATURES, "layer2": HIDDEN}
    C_per_site = {"layer1": C, "layer2": C}
    ci_fn = CIFn(d_in_per_site, C_per_site, CI_HIDDEN, key=key_ci)

    opt_main = optax.adam(LR)
    opt_ci = optax.adam(LR)
    state = init_state(decomposed, ci_fn, opt_main, opt_ci)

    step_fn = make_step_fn(
        target_model=target,
        site_paths=site_paths,
        coeff_faith=COEFF_FAITH,
        coeff_imp=COEFF_IMP,
        coeff_stoch=COEFF_STOCH,
        p_value=P_VALUE,
        opt_main=opt_main,
        opt_ci=opt_ci,
    )

    print(f"variant-3 TMS — {N_STEPS} steps")
    print(f"{'step':>6} {'total':>10} {'faith':>10} {'imp':>10} {'stoch':>10}")
    for step in range(N_STEPS):
        key, sub_data, sub_step = jax.random.split(key, 3)
        x = sample_batch(sub_data, BATCH_SIZE)
        state, losses = step_fn(state, x, sub_step)
        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            print(
                f"{step:>6} "
                f"{float(losses['total']):>10.5f} "
                f"{float(losses['faith']):>10.5f} "
                f"{float(losses['imp']):>10.5f} "
                f"{float(losses['stoch']):>10.5f}"
            )

    final = current_model(state)
    print("\nfinal V (layer1):", final.layer1.V)
    print("final U (layer1):", final.layer1.U)


if __name__ == "__main__":
    main()
