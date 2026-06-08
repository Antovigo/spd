"""Tiny 2-layer transformer with multi-site decomposition.

vocab=256, seq=32, d_model=64, n_heads=4, d_head=16, d_ff=256, n_layers=2.
13 decomposed sites: 6 per block (q,k,v,o,up,down) + 1 unembed. Random teacher,
random int token inputs — point is to exercise the architecture, not learn
anything semantic. Expected: faith drops meaningfully (floor > 0 due to rank-8
vs full-rank random teacher), stoch loss drops 10-100x.
"""

import jax
import optax
from nano_pd_jax.ci_fn import CIFn
from nano_pd_jax.decomposed import collect_site_paths, substitute_decomposed
from nano_pd_jax.trainer import init_state, make_step_fn
from nano_pd_jax.transformer import TinyTransformer

VOCAB = 256
SEQ = 32
BATCH = 32
D_MODEL = 64
N_HEADS = 4
D_HEAD = 16
D_FF = 256
N_LAYERS = 2
C = 8

N_STEPS = 2000
LOG_EVERY = 100

COEFF_FAITH = 1.0
COEFF_IMP = 1e-3
COEFF_STOCH = 1.0
P_VALUE = 0.9
LR = 1e-3
CI_HIDDEN = 64


def build_site_config(n_layers: int) -> tuple[dict[str, int], dict[str, int]]:
    sites_C: dict[str, int] = {}
    d_in_per_site: dict[str, int] = {}
    for i in range(n_layers):
        for name in ("q", "k", "v"):
            sites_C[f"blocks.{i}.attn.{name}"] = C
            d_in_per_site[f"blocks.{i}.attn.{name}"] = D_MODEL
        sites_C[f"blocks.{i}.attn.o"] = C
        d_in_per_site[f"blocks.{i}.attn.o"] = N_HEADS * D_HEAD
        sites_C[f"blocks.{i}.mlp.up"] = C
        d_in_per_site[f"blocks.{i}.mlp.up"] = D_MODEL
        sites_C[f"blocks.{i}.mlp.down"] = C
        d_in_per_site[f"blocks.{i}.mlp.down"] = D_FF
    sites_C["unembed"] = C
    d_in_per_site["unembed"] = D_MODEL
    return sites_C, d_in_per_site


def main() -> None:
    key = jax.random.PRNGKey(0)
    key_model, key_decomp, key_ci, key = jax.random.split(key, 4)

    target = TinyTransformer(
        vocab=VOCAB,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        key=key_model,
    )

    sites_C, d_in_per_site = build_site_config(N_LAYERS)
    decomposed = substitute_decomposed(target, sites_C, key=key_decomp)
    site_paths = collect_site_paths(decomposed)
    assert sorted(site_paths) == sorted(sites_C), (site_paths, sorted(sites_C))
    assert len(site_paths) == 6 * N_LAYERS + 1, len(site_paths)

    ci_fn = CIFn(d_in_per_site, sites_C, CI_HIDDEN, key=key_ci)

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

    print(f"sites ({len(site_paths)}): {site_paths}")
    print(f"\n{'step':>6} {'total':>10} {'faith':>10} {'imp':>10} {'stoch':>10}")
    for step_i in range(N_STEPS):
        key, sub_data, sub_step = jax.random.split(key, 3)
        x = jax.random.randint(sub_data, (BATCH, SEQ), 0, VOCAB)
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
