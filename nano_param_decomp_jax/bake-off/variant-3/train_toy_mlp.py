"""Target B: 2-layer MLP (d=64, d_ff=128, output=32) — 4 decomposition sites.

Sites: layer1.up, layer1.down, layer2.up, layer2.down — each with C=16.
Teacher is a randomly-initialized MLP_2layer; student decomposes it.
"""

import equinox as eqx
import jax
import optax
from ci_fn import CIFn
from decomposed_linear import DecomposedLinear, substitute_decomposed
from jaxtyping import Array, Float, PRNGKeyArray
from trainer import collect_site_paths, init_state, make_step_fn

D_MODEL = 64
D_FF = 128
D_OUT = 32
BATCH_SIZE = 256
N_STEPS = 5000
LOG_EVERY = 100
C = 16

COEFF_FAITH = 1.0
COEFF_IMP = 1e-3
COEFF_STOCH = 1.0
P_VALUE = 0.9
LR = 1e-3
CI_HIDDEN = 64


class MLPLayer(eqx.Module):
    up: eqx.nn.Linear  # d_in -> d_ff
    down: eqx.nn.Linear  # d_ff -> d_out

    def __init__(self, d_in: int, d_ff: int, d_out: int, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        self.up = eqx.nn.Linear(d_in, d_ff, key=k1)
        self.down = eqx.nn.Linear(d_ff, d_out, key=k2)

    def __call__(
        self,
        x: Float[Array, " d_in"],
        m_up: Float[Array, " C"] | None,
        m_down: Float[Array, " C"] | None,
    ) -> Float[Array, " d_out"]:
        h = _apply(self.up, x, m_up)
        h = jax.nn.gelu(h)
        return _apply(self.down, h, m_down)

    def forward_with_acts(
        self, x: Float[Array, " d_in"], prefix: str
    ) -> tuple[Float[Array, " d_out"], dict[str, Float[Array, "..."]]]:
        acts: dict[str, Float[Array, ...]] = {f"{prefix}.up": x}
        h = _apply(self.up, x, None)
        h = jax.nn.gelu(h)
        acts[f"{prefix}.down"] = h
        return _apply(self.down, h, None), acts


def _apply(layer, x, mask):
    """Call layer with mask if it's a DecomposedLinear, else without."""
    if isinstance(layer, DecomposedLinear):
        return layer(x, mask)
    return layer(x)


class MLP2(eqx.Module):
    layer1: MLPLayer  # d_model -> d_ff -> d_model
    layer2: MLPLayer  # d_model -> d_ff -> d_out

    def __init__(self, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        self.layer1 = MLPLayer(D_MODEL, D_FF, D_MODEL, key=k1)
        self.layer2 = MLPLayer(D_MODEL, D_FF, D_OUT, key=k2)

    def __call__(
        self,
        x: Float[Array, " d_model"],
        masks: dict[str, Float[Array, " C"]] | None = None,
    ) -> Float[Array, " d_out"]:
        m = masks if masks is not None else {}
        h = self.layer1(x, m.get("layer1.up"), m.get("layer1.down"))
        return self.layer2(h, m.get("layer2.up"), m.get("layer2.down"))

    def forward_with_acts(
        self, x: Float[Array, " d_model"]
    ) -> tuple[Float[Array, " d_out"], dict[str, Float[Array, "..."]]]:
        h, acts1 = self.layer1.forward_with_acts(x, "layer1")
        out, acts2 = self.layer2.forward_with_acts(h, "layer2")
        return out, {**acts1, **acts2}


def main() -> None:
    key = jax.random.PRNGKey(0)
    key_model, key_decomp, key_ci, key = jax.random.split(key, 4)

    target = MLP2(key=key_model)
    site_paths_to_C = {
        "layer1.up": C,
        "layer1.down": C,
        "layer2.up": C,
        "layer2.down": C,
    }
    decomposed = substitute_decomposed(target, site_paths_to_C, key=key_decomp)
    site_paths = collect_site_paths(decomposed)
    assert sorted(site_paths) == sorted(site_paths_to_C), site_paths

    d_in_per_site = {
        "layer1.up": D_MODEL,
        "layer1.down": D_FF,
        "layer2.up": D_MODEL,
        "layer2.down": D_FF,
    }
    ci_fn = CIFn(d_in_per_site, site_paths_to_C, CI_HIDDEN, key=key_ci)

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

    print(f"variant-3 toy MLP — {N_STEPS} steps")
    print(f"{'step':>6} {'total':>10} {'faith':>10} {'imp':>10} {'stoch':>10}")
    for step in range(N_STEPS):
        key, sub_data, sub_step = jax.random.split(key, 3)
        x = jax.random.normal(sub_data, (BATCH_SIZE, D_MODEL))
        state, losses = step_fn(state, x, sub_step)
        if step % LOG_EVERY == 0 or step == N_STEPS - 1:
            print(
                f"{step:>6} "
                f"{float(losses['total']):>10.5f} "
                f"{float(losses['faith']):>10.5f} "
                f"{float(losses['imp']):>10.5f} "
                f"{float(losses['stoch']):>10.5f}"
            )


if __name__ == "__main__":
    main()
