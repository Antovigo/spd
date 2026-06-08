"""Stage 4 worker: multi-process JAX 2-pool probe (one process per pool).

Launched twice (process_id 0 and 1) by stage4_launch.py. Each process:
  * joins a 2-process jax.distributed world (1 CPU device each),
  * materializes ONLY its own local shard of the params (the memory-scaling story:
    no process ever holds the full global array),
  * runs ONE global jit spanning both processes: pool 0 makes a mask, ppermutes it
    to pool 1, pool 1 computes the coupled loss; autograd returns cotangents across
    the PROCESS boundary automatically,
  * grad-checks its local shard against a single-process reference (each process can
    rebuild the full reference from the shared seed, since this is a toy).

The question: does the Stage-2 differentiable-ppermute win survive a real
cross-process boundary, with each process holding only its own shard?
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

D = 8


def build_global_from_local(global_shape, sharding, local_shard):
    """Assemble a global jax.Array from this process's single local shard."""
    return jax.make_array_from_single_device_arrays(
        global_shape,
        sharding,
        [jax.device_put(local_shard, d) for d in sharding.addressable_devices],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--process_id", type=int, required=True)
    ap.add_argument("--num_processes", type=int, default=2)
    ap.add_argument("--coordinator", default="localhost:12399")
    args = ap.parse_args()

    jax.distributed.initialize(
        coordinator_address=args.coordinator,
        num_processes=args.num_processes,
        process_id=args.process_id,
    )
    pid = args.process_id
    jax.config.update("jax_enable_x64", True)

    if pid == 0:
        print(f"[p{pid}] global devices: {jax.device_count()}  local: {jax.local_device_count()}")

    mesh = Mesh(np.array(jax.devices()), axis_names=("pool",))
    pool_sharding = NamedSharding(mesh, P("pool"))

    # --- each process builds ONLY its own (D,D) params, deterministic from seed ---
    key = random.PRNGKey(0)
    ka, kb, kx = random.split(key, 3)
    theta_a_full = random.normal(ka, (D, D)) * 0.3  # pool 0's params
    theta_b_full = random.normal(kb, (D, D)) * 0.3  # pool 1's params
    x = random.normal(kx, (4, D))

    my_theta = theta_a_full if pid == 0 else theta_b_full  # this process's shard only

    # global (2,D,D) array; process p materializes ONLY slice p.
    theta_global = build_global_from_local((2, D, D), pool_sharding, my_theta[None])
    # prove no process holds the whole thing:
    addr = theta_global.addressable_shards
    if pid == 0:
        print(
            f"[p{pid}] theta_global shape {theta_global.shape}, "
            f"this process materializes {len(addr)} shard(s) of shape "
            f"{addr[0].data.shape}  (NOT the full (2,{D},{D}))"
        )

    x_global = build_global_from_local((2, *x.shape), pool_sharding, x[None])

    def step(theta_pair, x_pair):
        my = theta_pair[0]
        xb = x_pair[0]
        axis_idx = jax.lax.axis_index("pool")
        mask_local = jax.nn.sigmoid(xb @ my)
        mask_on_b = jax.lax.ppermute(mask_local, "pool", perm=[(0, 1)])  # 0 -> 1
        out_b = (xb * mask_on_b) @ my
        loss_b = jnp.sum(out_b**2)
        loss = jnp.where(axis_idx == 1, loss_b, 0.0)
        return loss[None]

    @jax.jit
    def total(theta_g, x_g):
        sm = shard_map(
            step,
            mesh=mesh,
            in_specs=(P("pool"), P("pool")),
            out_specs=P("pool"),
            check_rep=False,
        )
        return jnp.sum(sm(theta_g, x_g))

    g_global = jax.grad(total, argnums=0)(theta_global, x_global)  # sharded grad

    # --- single-process reference (toy: rebuild full thing locally) ---
    def reference(theta_a, theta_b):
        mask = jax.nn.sigmoid(x @ theta_a)
        out = (x * mask) @ theta_b
        return jnp.sum(out**2)

    g_ref_a, g_ref_b = jax.grad(reference, argnums=(0, 1))(theta_a_full, theta_b_full)
    g_ref_mine = g_ref_a if pid == 0 else g_ref_b

    my_grad_shard = g_global.addressable_shards[0].data[0]  # this process's grad slice
    rel = float(
        jnp.max(
            jnp.abs(my_grad_shard - g_ref_mine)
            / (jnp.maximum(jnp.abs(my_grad_shard), jnp.abs(g_ref_mine)) + 1e-12)
        )
    )
    pool = "A (mask producer)" if pid == 0 else "B (loss/recon)"
    print(
        f"[p{pid}] pool {pool}: local grad-shard rel err vs reference = {rel:.3e}  "
        f"-> {'PASS' if rel < 1e-9 else 'FAIL'}"
    )

    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
