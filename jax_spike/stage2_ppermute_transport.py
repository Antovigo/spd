"""Stage 2: cross-pool transport via collectives, and does it compose with autograd?

The hard infra question for a JAX 2-pool: there is no native send/recv between two
*heterogeneous* JAX programs. The JAX-native alternative is a single mesh + collectives,
using `jax.lax.ppermute` as the point-to-point primitive. The load-bearing question:

    Does ppermute compose with autograd so that shipping masks A->B inside a
    differentiated region makes the cotangents flow back B->A *automatically*
    (ppermute's transpose is the reverse ppermute)?

If yes, the manual "ship g_CI back over the wire and seed a held graph" seam
collapses to nothing — autograd handles it. That would materially change the
feasibility picture. This tests it on CPU with forced multi-device.

Run with: XLA_FLAGS="--xla_force_host_platform_device_count=2" python stage2_...
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

jax.config.update("jax_enable_x64", True)

D = 8
print("devices:", jax.devices())
assert jax.device_count() >= 2, "need >=2 devices: set XLA_FLAGS host device count"

mesh = Mesh(np.array(jax.devices()[:2]), axis_names=("pool",))


def reference(theta_a, theta_b, x):
    """Monolithic: A makes a mask from x+theta_a, B uses it with theta_b."""
    mask = jax.nn.sigmoid(x @ theta_a)
    out = (x * mask) @ theta_b
    return jnp.sum(out**2)


def make_pooled(theta_a, theta_b, x):
    """theta stacked along the pool axis: index 0 = A's params, 1 = B's params.

    A computes a mask and ppermutes it to B; B computes the loss. We differentiate
    the whole thing and check the gradient that lands back on A's params matches the
    monolithic reference -- i.e. the cotangent came back across the mesh for free.
    """

    def step(theta_pair, x_b):
        # theta_pair: (2, D, D) sharded -> each device sees its own (1, D, D)
        my = theta_pair[0]  # this device's params slice
        axis_idx = jax.lax.axis_index("pool")

        # POOL A (axis 0): make mask. POOL B (axis 1): placeholder.
        mask_local = jax.nn.sigmoid(x_b @ my)

        # ship A's mask -> B  (device 0 -> device 1). ppermute is differentiable;
        # its transpose sends cotangents 1 -> 0 automatically.
        mask_on_b = jax.lax.ppermute(mask_local, "pool", perm=[(0, 1)])

        # POOL B computes the coupled loss using the received mask + its own params.
        out_b = (x_b * mask_on_b) @ my
        loss_b = jnp.sum(out_b**2)

        # only B's loss is real; A contributes 0 to the value but receives cotangents
        # through ppermute's transpose.
        loss = jnp.where(axis_idx == 1, loss_b, 0.0)
        return loss[None]  # singleton pool axis so out_specs P("pool") concatenates

    theta_pair = jnp.stack([theta_a, theta_b])  # (2, D, D)
    x_pair = jnp.stack([x, x])  # same batch on both

    sm = shard_map(
        step,
        mesh=mesh,
        in_specs=(P("pool"), P("pool")),
        out_specs=P("pool"),
        check_rep=False,
    )

    def total(theta_pair_):
        per_dev = sm(theta_pair_, x_pair)  # (2,) one scalar per device
        return jnp.sum(per_dev)

    return total, theta_pair


def main():
    key = random.PRNGKey(0)
    ka, kb, kx = random.split(key, 3)
    theta_a = random.normal(ka, (D, D)) * 0.3
    theta_b = random.normal(kb, (D, D)) * 0.3
    x = random.normal(kx, (4, D))

    # reference grads
    g_ref_a, g_ref_b = jax.grad(reference, argnums=(0, 1))(theta_a, theta_b, x)

    # pooled grads via shard_map + ppermute
    total, theta_pair = make_pooled(theta_a, theta_b, x)
    g_pair = jax.grad(total)(theta_pair)  # (2, D, D)
    g_pooled_a = g_pair[0]  # cotangent that landed back on pool A
    g_pooled_b = g_pair[1]

    def rel(a, b):
        return float(jnp.max(jnp.abs(a - b) / (jnp.maximum(jnp.abs(a), jnp.abs(b)) + 1e-12)))

    err_a = rel(g_ref_a, g_pooled_a)
    err_b = rel(g_ref_b, g_pooled_b)
    print(f"pool A grad rel err (cotangent came back over ppermute): {err_a:.3e}")
    print(f"pool B grad rel err:                                     {err_b:.3e}")
    ok = err_a < 1e-9 and err_b < 1e-9
    print("STAGE 2:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    main()
