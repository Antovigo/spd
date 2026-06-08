"""Stage 3: the remaining friction -- genuinely heterogeneous pools.

Stage 2 worked because both pools ran the SAME code over a sharded array. The real
2-pool is heterogeneous: pool A holds a ~20B CI fn; pool B holds the V/U replica.
Different shapes, different work, different memory. This probes how badly that fights
JAX's SPMD model.

Probe 1: can shard_map host different-shaped params per pool directly? (expected: no
         -- shard_map shards ONE array uniformly along the mapped axis.)
Probe 2: the idiom that does work -- pad/separate-array per pool + branch on
         axis_index, with ppermute carrying only the cross-pool tensor. Show grads
         still correct.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

jax.config.update("jax_enable_x64", True)
assert jax.device_count() >= 2
mesh = Mesh(np.array(jax.devices()[:2]), axis_names=("pool",))

D = 8
DA = 16  # pool A hidden width (CI fn) -- bigger
DB = 8  # pool B width (V/U)


# ---------- Probe 1: try heterogeneous shapes directly ----------
def probe1():
    print("\n=== Probe 1: heterogeneous-shaped params in one sharded array ===")
    try:
        # try to stack (16,16) and (8,8) along a pool axis -- impossible, ragged.
        _ = jnp.stack([jnp.zeros((DA, DA)), jnp.zeros((DB, DB))])
        print("  unexpectedly stacked ragged shapes")
    except (ValueError, TypeError) as e:
        print(f"  cannot stack different shapes into one array (expected): {type(e).__name__}")
        print("  -> heterogeneous pools cannot be a single sharded array.")


# ---------- Probe 2: the working idiom ----------
# Reference: A maps x (B,D) -> mask (B,D) via W_a (D,DA)->(DA)->(D); B uses mask with W_b.
def reference(W_a1, W_a2, W_b, x):
    h = jnp.tanh(x @ W_a1)  # (B, DA)
    mask = jax.nn.sigmoid(h @ W_a2)  # (B, D)
    out = (x * mask) @ W_b  # (B, D)
    return jnp.sum(out**2)


def probe2():
    print("\n=== Probe 2: working idiom (per-pool pytrees + ppermute) ===")
    key = random.PRNGKey(0)
    k1, k2, k3, kx = random.split(key, 4)
    W_a1 = random.normal(k1, (D, DA)) * 0.2
    W_a2 = random.normal(k2, (DA, D)) * 0.2
    W_b = random.normal(k3, (D, D)) * 0.2
    x = random.normal(kx, (4, D))

    g_ref = jax.grad(reference, argnums=(0, 1, 2))(W_a1, W_a2, W_b, x)

    # Each pool gets its OWN pytree (NOT a shared sharded array). shard_map maps over
    # a dummy pool-axis array just to place execution; real params are closed over and
    # only the relevant branch uses them. The cross-pool tensor (mask) goes via ppermute.
    def step(dummy, x_b, W_a1_, W_a2_, W_b_):
        axis_idx = jax.lax.axis_index("pool")
        # POOL A branch produces a mask; POOL B produces zeros of the same shape.
        h = jnp.tanh(x_b @ W_a1_)
        mask_a = jax.nn.sigmoid(h @ W_a2_)
        mask_local = jnp.where(axis_idx == 0, mask_a, jnp.zeros_like(mask_a))
        # ship A's mask -> B; cotangents return automatically.
        mask_on_b = jax.lax.ppermute(mask_local, "pool", perm=[(0, 1)])
        out_b = (x_b * mask_on_b) @ W_b_
        loss_b = jnp.sum(out_b**2)
        loss = jnp.where(axis_idx == 1, loss_b, 0.0)
        return loss[None]

    dummy = jnp.zeros((2,))
    x_pair = jnp.broadcast_to(x, (2, *x.shape))

    def total(W_a1_, W_a2_, W_b_):
        sm = shard_map(
            lambda d, xb: step(d, xb, W_a1_, W_a2_, W_b_),
            mesh=mesh,
            in_specs=(P("pool"), P("pool")),
            out_specs=P("pool"),
            check_rep=False,
        )
        return jnp.sum(sm(dummy, x_pair))

    g_pool = jax.grad(total, argnums=(0, 1, 2))(W_a1, W_a2, W_b)

    def rel(a, b):
        return float(jnp.max(jnp.abs(a - b) / (jnp.maximum(jnp.abs(a), jnp.abs(b)) + 1e-12)))

    names = ["W_a1", "W_a2", "W_b"]
    worst = 0.0
    for n, gr, gp in zip(names, g_ref, g_pool, strict=False):
        e = rel(gr, gp)
        worst = max(worst, e)
        print(f"  {n} grad rel err: {e:.3e}")
    ok = worst < 1e-9
    print("  Probe 2:", "PASS" if ok else "FAIL")
    print("  NOTE: A's params (W_a1/W_a2) and B's (W_b) are separate pytrees of")
    print("        DIFFERENT shapes -- but BOTH branches are traced on BOTH devices.")
    return ok


if __name__ == "__main__":
    probe1()
    ok = probe2()
    print("\nSTAGE 3:", "PASS" if ok else "FAIL")
