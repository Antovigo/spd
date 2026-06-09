"""Probe: per-step COMM cost of each strategy at MULTI-NODE scale (the cost the NVLink probes hid).

Real runs are ~100 GPU on much slower (cross-node) interconnect, so recurring communication, not
resident memory, decides throughput. This times the collectives that define each candidate:

  ZeRO-3 per step  = all_gather(params) + reduce_scatter(grads)   <- the heaviest
  ZeRO-1 / DDP     = all_reduce(grads)                            <- params replicated, cheaper
  sub-mesh hand-off= reshard a tensor between two node-disjoint sub-meshes (cross-node p2p)

Launch multi-node: `sbatch --nodes=N --ntasks-per-node=8 remote/cw_jax.sbatch probe_comm.py`
(uses init_distributed: 1 process/GPU, jax auto-detects the multi-node coordinator).
"""

import statistics
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from distributed_util import init_distributed
from jax import shard_map
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

init_distributed()
devs = jax.devices()
W = len(devs)
is0 = jax.process_index() == 0
mesh = Mesh(np.array(devs), ("dp",))

# Full V/U scale: ~390M params -> 780 MB bf16. Sharded as (W, K) so each device holds 1/W.
TOTAL = 390_000_000
K = TOTAL // W
gb = TOTAL * 2 / 1e9  # bf16


def med_ms(fn, x, n=50):
    y = fn(x)
    jax.block_until_ready(y)  # compile/warm
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        y = fn(x)
        jax.block_until_ready(y)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


@jax.jit
@partial(shard_map, mesh=mesh, in_specs=P("dp"), out_specs=P(), check_vma=False)
def all_gather(x):  # ZeRO-3: gather sharded params -> full replica
    return jax.lax.all_gather(x, "dp", tiled=True)


@jax.jit
@partial(shard_map, mesh=mesh, in_specs=P(), out_specs=P(), check_vma=False)
def all_reduce(x):  # ZeRO-1 / DDP: sum grads across all ranks
    return jax.lax.psum(x, "dp")


@jax.jit
@partial(shard_map, mesh=mesh, in_specs=P(), out_specs=P("dp"), check_vma=False)
def reduce_scatter(x):  # ZeRO-3: sum grads, leave each rank its 1/W shard
    return jax.lax.psum_scatter(x, "dp", tiled=True)


x_sharded = jax.device_put(jnp.ones((W, K), jnp.bfloat16), NamedSharding(mesh, P("dp")))
x_repl = jax.device_put(jnp.ones((W * K,), jnp.bfloat16), NamedSharding(mesh, P()))

ag = med_ms(all_gather, x_sharded)
ar = med_ms(all_reduce, x_repl)
rs = med_ms(reduce_scatter, x_repl)

# sub-mesh hand-off: reshard between two NODE-disjoint halves (cross-node p2p at >=2 nodes)
reshard_ms = None
if W >= 16:
    half = W // 2
    ma = Mesh(np.array(devs[:half]), ("dp",), axis_types=(AxisType.Explicit,))
    mb = Mesh(np.array(devs[half:]), ("dp",), axis_types=(AxisType.Explicit,))
    h = jax.device_put(
        jnp.ones((half, 8192 * 8192 // half), jnp.bfloat16), NamedSharding(ma, P("dp"))
    )
    move = jax.jit(lambda t: jax.reshard(t, NamedSharding(mb, P("dp"))))
    reshard_ms = med_ms(move, h)

if is0:
    nodes = W // 8
    print(f"[comm] {W} GPU (~{nodes} node(s)), tensor = {gb:.2f} GB bf16")
    print(
        f"[comm] all_gather     {ag:8.2f} ms   {gb / (ag / 1e3):7.0f} GB/s   (ZeRO-3 param gather)"
    )
    print(
        f"[comm] reduce_scatter {rs:8.2f} ms   {gb / (rs / 1e3):7.0f} GB/s   (ZeRO-3 grad scatter)"
    )
    print(
        f"[comm] all_reduce     {ar:8.2f} ms   {gb / (ar / 1e3):7.0f} GB/s   (ZeRO-1 / DDP grads)"
    )
    if reshard_ms is not None:
        hgb = 8192 * 8192 * 2 / 1e9
        print(
            f"[comm] reshard A->B   {reshard_ms:8.2f} ms   {hgb / (reshard_ms / 1e3):7.0f} GB/s   (sub-mesh hand-off, {hgb:.2f}GB)"
        )
    print(f"[comm] ZeRO-3 per-step comm ~= all_gather + reduce_scatter = {ag + rs:.2f} ms")
    print(f"[comm] ZeRO-1 per-step comm ~= all_reduce                  = {ar:.2f} ms")

jax.distributed.shutdown()
