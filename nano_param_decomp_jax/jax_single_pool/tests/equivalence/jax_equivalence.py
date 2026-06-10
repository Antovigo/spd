"""JAX equivalence check: the JAX single-pool PD loss terms vs the torch reference.

Run in the JAX (`.venv-cuda`) env AFTER `torch_reference.py`. Loads the SAME fixtures,
builds a JAX `Target` + `DecompVU` with the identical (zeroed-attn) suffix weights, and
computes each loss term through the JAX step's OWN helpers, feeding the FIXED masks /
sources / routing from the fixtures (no RNG). Compares to `torch_reference.json` at fp32
tolerance.

Term wiring (all from `jax_single_pool.llama8b_step`):
  * faith  — `_faith_loss(vu, decomp_layers)`
  * imp    — `_imp_min(ci_upper, p, beta, eps)`
  * stoch  — per chunk: build `mask = ci+(1-ci)*u`, fixed delta mask, fixed route over the
             chunk's 3 sites; `suffix_logits(..., decompose_layer={i})`; `_kl_per_position`
             vs clean. Mean over chunks. (Same path as `_stochastic_recon`/`_stoch_one_chunk`
             but with the fixtures' masks substituted for the RNG draws.)
  * ppgd   — `_ppgd_recon(...)` with the fixed sources (it derives masks via
             `_ppgd_masks_and_deltas`, the torch `get_ppgd_mask_infos` analog).

This is the numeric cross-framework check the task calls for. Bit-identical is impossible
across RNG/FP backends; we assert each term within `RTOL`/`ATOL` of the torch value.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", False)

from jax_single_pool.llama8b import (  # noqa: E402
    KINDS,
    DecompLayerFrozen,
    DecompVU,
    FrozenAttn,
    FrozenBlock,
    FrozenMLP,
    Target,
    suffix_logits,
)
from jax_single_pool.llama8b_step import (  # noqa: E402
    _faith_loss,
    _imp_min,
    _kl_per_position,
    _layerfirst,
    _layerfirst_delta,
    _ppgd_recon,
)

HERE = Path(__file__).resolve().parent
RTOL = 2e-4
ATOL = 1e-5


# fp32 throughout the harness so the cross-framework comparison is fp-tight (the torch
# reference is fp32). The production step runs `DT=bfloat16`; here we override to isolate
# the loss MATH from bf16 rounding (a bf16 forward agrees with torch only to ~1e-3).
FP = jnp.float32


def _zero_attn(d: int, di: int) -> FrozenAttn:
    """Attn with zeroed projections (contributes 0); head dims are arbitrary since the
    output is 0. RoPE never affects a zero output."""
    n_head, n_kv_head, head_dim = 2, 1, d // 2
    qd = n_head * head_dim
    kvd = n_kv_head * head_dim
    z = lambda r, c: jnp.zeros((r, c), FP)  # noqa: E731
    return FrozenAttn(
        wq=z(qd, d), wk=z(kvd, d), wv=z(kvd, d), wo=z(d, qd),
        n_head=n_head, n_kv_head=n_kv_head, head_dim=head_dim, n_rep=n_head // n_kv_head,
    )  # fmt: skip


def _build(f: dict[str, np.ndarray]):
    a = lambda key: jnp.asarray(f[key], dtype=FP)  # noqa: E731
    d = int(f["_scalar_N_EMBD"])
    di = int(f["_scalar_N_INTERMEDIATE"])
    n_layers = int(f["_scalar_N_DECOMP_LAYERS"])
    n_tail = int(f["_scalar_N_TAIL"])
    eps = float(f["_scalar_EPS"])

    decomp_layers = [
        DecompLayerFrozen(
            ln1=a(f"ln1_{i}"),
            ln2=a(f"ln2_{i}"),
            attn=_zero_attn(d, di),
            Wg=a(f"Wg_{i}"),
            Wu=a(f"Wu_{i}"),
            Wd=a(f"Wd_{i}"),
        )  # fmt: skip
        for i in range(n_layers)
    ]
    tail = [
        FrozenBlock(
            ln1=a(f"tail_ln1_{j}"),
            ln2=a(f"tail_ln2_{j}"),
            attn=_zero_attn(d, di),
            mlp=FrozenMLP(wg=a(f"tail_Wg_{j}"), wu=a(f"tail_Wu_{j}"), wd=a(f"tail_Wd_{j}")),
            eps=eps,
        )  # fmt: skip
        for j in range(n_tail)
    ]
    # inv_freq unused (attn zeroed); a dummy valid-shaped array.
    inv_freq = jnp.ones((d // 4,), jnp.float32)
    tgt = Target(
        decomp_layers=decomp_layers, tail=tail, norm=a("norm"), lm_head=a("lm_head"),
        inv_freq=inv_freq, eps=eps,
    )  # fmt: skip
    vu = DecompVU(
        Vg=jnp.stack([a(f"Vg_{i}") for i in range(n_layers)]),
        Ug=jnp.stack([a(f"Ug_{i}") for i in range(n_layers)]),
        Vu=jnp.stack([a(f"Vu_{i}") for i in range(n_layers)]),
        Uu=jnp.stack([a(f"Uu_{i}") for i in range(n_layers)]),
        Vd=jnp.stack([a(f"Vd_{i}") for i in range(n_layers)]),
        Ud=jnp.stack([a(f"Ud_{i}") for i in range(n_layers)]),
    )
    return tgt, vu, n_layers


def compute_jax_terms(f: dict[str, np.ndarray]) -> dict[str, float]:
    """The four JAX loss-term values on the fixtures `f` (fp32). Shared by `main` and the
    pytest so there is one term-computation path."""
    tgt, vu, n_layers = _build(f)
    resid = jnp.asarray(f["resid"], dtype=FP)
    B, T = int(f["_scalar_B"]), int(f["_scalar_T"])

    nomask = {k: None for k in KINDS}
    dm_ones = {k: jnp.ones((n_layers, 1, 1), FP) for k in KINDS}
    no_routes = {k: None for k in KINDS}
    clean = jax.lax.stop_gradient(suffix_logits(tgt, vu, resid, nomask, dm_ones, no_routes))

    ci_lower = {k: jnp.asarray(f[f"ci_lower_{k}"], dtype=FP) for k in KINDS}  # (B,T,L,C)
    ci_upper = {k: jnp.asarray(f[f"ci_upper_{k}"], dtype=FP) for k in KINDS}

    # ---- faith ----
    faith = float(_faith_loss(vu, tgt.decomp_layers))

    # ---- imp ----
    imp = float(
        _imp_min(ci_upper, float(f["_scalar_IMP_P"]), float(f["_scalar_IMP_BETA"]),
                 float(f["_scalar_IMP_EPS"]))
    )  # fmt: skip

    # ---- stoch (per-chunk, FIXED masks) ----
    stoch_total = 0.0
    for i in range(n_layers):
        masks: dict = {}
        delta_masks: dict = {}
        routes: dict = {}
        for k in KINDS:
            ci_k = ci_lower[k]  # (B,T,L,C)
            u = jnp.asarray(f[f"stoch_u_{k}"], dtype=FP)  # (B,T,L,C)
            masks[k] = ci_k + (1.0 - ci_k) * u
            dmv = jnp.asarray(f[f"stoch_delta_{k}"], dtype=FP)[..., None]  # (B,T,L,1)
            delta_masks[k] = dmv
            route_site = jnp.asarray(f[f"route_chunk{i}_{k}"])  # (B,T) bool
            route_full = (
                jnp.zeros((B, T, n_layers, 1), bool).at[:, :, i, :].set(route_site[:, :, None])
            )
            routes[k] = route_full
        decompose = tuple(j == i for j in range(n_layers))
        pred = suffix_logits(
            tgt, vu, resid,
            _layerfirst(masks), _layerfirst_delta(delta_masks), _layerfirst(routes),
            decompose,
        )  # fmt: skip
        stoch_total += float(_kl_per_position(pred, clean))
    stoch = stoch_total / n_layers

    # ---- ppgd (FIXED sources) ----
    source = {k: jnp.asarray(f[f"ppgd_source_{k}"], dtype=FP) for k in KINDS}  # (1,T,L,C+1)
    ppgd = float(_ppgd_recon(tgt, vu, resid, clean, ci_lower, source, suffix_logits))

    return {"faith": faith, "imp": imp, "stoch": stoch, "ppgd": ppgd}


def main() -> None:
    f = dict(np.load(HERE / "fixtures.npz"))
    ref = json.loads((HERE / "torch_reference.json").read_text())
    jaxv = compute_jax_terms(f)
    print(f"{'term':6} {'jax':>16} {'torch':>16} {'rel_err':>12}  ok")
    all_ok = True
    for term in ("faith", "imp", "stoch", "ppgd"):
        jv, tv = jaxv[term], ref[term]
        rel = abs(jv - tv) / (abs(tv) + 1e-30)
        ok = abs(jv - tv) <= ATOL + RTOL * abs(tv)
        all_ok = all_ok and ok
        print(f"{term:6} {jv:16.8e} {tv:16.8e} {rel:12.3e}  {'PASS' if ok else 'FAIL'}")
    assert all_ok, "JAX term(s) diverge from torch reference beyond tolerance"
    print("\nALL TERMS NUMERICALLY EQUIVALENT (fp32) to the torch 2-pool reference.")


if __name__ == "__main__":
    main()
