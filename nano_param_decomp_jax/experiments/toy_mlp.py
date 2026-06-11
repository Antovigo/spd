"""Toy 2-layer MLP: d_model=64, d_ff=128, d_out=32, 4 decomposable Linear sites.

Random teacher MLP, gaussian inputs. Demonstrates multi-site composition via
the mask-tolerant Linear shim. Expected: faith floor set by rank-16 approx of
rank-64 random teacher matrices (~1e-2); stoch → 1e-6 range over 5000 steps.

User-model LOC target: < variant-3 bake-off's 150 LOC. The mask-tolerant
Linear means no isinstance dispatch in __call__; the (out, acts) return means
no separate forward_with_acts.
"""

import equinox as eqx
import jax
import optax
from jaxtyping import Array, Float, PRNGKeyArray
from nano_pd_jax.ci_fn import CIFn
from nano_pd_jax.decomposed import collect_site_paths, substitute_decomposed
from nano_pd_jax.linear import Linear
from nano_pd_jax.trainer import init_state, make_step_fn

D_MODEL = 64
D_FF = 128
D_OUT = 32
BATCH_SIZE = 256
N_STEPS = 5000
LOG_EVERY = 250
C = 16

COEFF_FAITH = 1.0
COEFF_IMP = 1e-3
COEFF_STOCH = 1.0
P_VALUE = 0.9
LR = 1e-3
CI_HIDDEN = 64

SITES: dict[str, int] = {
    "layer1.up": C,
    "layer1.down": C,
    "layer2.up": C,
    "layer2.down": C,
}


class MLPLayer(eqx.Module):
    up: Linear
    down: Linear

    def __init__(self, d_in: int, d_ff: int, d_out: int, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        self.up = Linear(d_in, d_ff, key=k1)
        self.down = Linear(d_ff, d_out, key=k2)

    def __call__(
        self,
        x: Float[Array, " d_in"],
        m_up: Float[Array, " C"] | None,
        m_down: Float[Array, " C"] | None,
    ) -> tuple[Float[Array, " d_out"], Float[Array, " d_in"], Float[Array, " d_ff"]]:
        h, a_up = self.up(x, m_up)
        h = jax.nn.gelu(h)
        out, a_down = self.down(h, m_down)
        return out, a_up, a_down


class MLP2(eqx.Module):
    layer1: MLPLayer
    layer2: MLPLayer

    def __init__(self, *, key: PRNGKeyArray):
        k1, k2 = jax.random.split(key)
        self.layer1 = MLPLayer(D_MODEL, D_FF, D_MODEL, key=k1)
        self.layer2 = MLPLayer(D_MODEL, D_FF, D_OUT, key=k2)

    def __call__(
        self,
        x: Float[Array, " d_model"],
        masks: dict[str, Float[Array, " C"]] | None = None,
    ) -> tuple[Float[Array, " d_out"], dict[str, Float[Array, "..."]]]:
        m = masks if masks is not None else {}
        h, a1u, a1d = self.layer1(x, m.get("layer1.up"), m.get("layer1.down"))
        out, a2u, a2d = self.layer2(h, m.get("layer2.up"), m.get("layer2.down"))
        return out, {
            "layer1.up": a1u,
            "layer1.down": a1d,
            "layer2.up": a2u,
            "layer2.down": a2d,
        }


def main() -> None:
    key = jax.random.PRNGKey(0)
    key_model, key_decomp, key_ci, key = jax.random.split(key, 4)

    target = MLP2(key=key_model)
    decomposed = substitute_decomposed(target, SITES, key=key_decomp)
    site_paths = collect_site_paths(decomposed)
    assert sorted(site_paths) == sorted(SITES), site_paths

    d_in_per_site = {
        "layer1.up": D_MODEL,
        "layer1.down": D_FF,
        "layer2.up": D_MODEL,
        "layer2.down": D_FF,
    }
    ci_fn = CIFn(d_in_per_site, SITES, CI_HIDDEN, key=key_ci)

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

    print(f"{'step':>6} {'total':>10} {'faith':>10} {'imp':>10} {'stoch':>10}")
    for step_i in range(N_STEPS):
        key, sub_data, sub_step = jax.random.split(key, 3)
        x = jax.random.normal(sub_data, (BATCH_SIZE, D_MODEL))
        state, losses = step_fn(state, x, sub_step)
        if step_i % LOG_EVERY == 0 or step_i == N_STEPS - 1:
            print(
                f"{step_i:>6} "
                f"{float(losses['total']):>10.5f} "
                f"{float(losses['faith']):>10.5f} "
                f"{float(losses['imp']):>10.5f} "
                f"{float(losses['stoch']):>10.5f}"
            )


if __name__ == "__main__":
    main()
