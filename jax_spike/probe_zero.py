"""Probe: does full-mesh ZeRO sharding of params + fp32 Adam state actually free resident memory,
and does a gather-on-use forward still work? (note-2: single-mesh, shard all resident memory.)

Measures per-device resident bytes for a V/U-Adam-sized blob, replicated vs sharded across the
whole mesh, then runs a sharded matmul (XLA gathers on use) to confirm compute composes.
"""

import gc

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

devs = jax.devices()
W = len(devs)
mesh = Mesh(np.array(devs), ("fsdp",))
# Two shardings for the SAME array shape: P() keeps a full copy on every device (what we do today);
# P("fsdp", None, None) splits the leading axis across all GPUs so each holds 1/W (ZeRO/FSDP).
repl = NamedSharding(mesh, P())
shard = NamedSharding(mesh, P("fsdp", None, None))

# ~ V/U + Adam shaped: B blocks of (8192,8192); bf16 param + fp32 m,v. B divisible by W.
B = W
shp = (B, 8192, 8192)
param_gb = B * 8192 * 8192 * (2 + 4 + 4) / 1e9  # bf16 + fp32 m + fp32 v


def live_gb():
    return max(d.memory_stats()["bytes_in_use"] for d in jax.local_devices()) / 1e9


def make(sharding):
    p = jax.device_put(jnp.ones(shp, jnp.bfloat16), sharding)
    m = jax.device_put(jnp.zeros(shp, jnp.float32), sharding)
    v = jax.device_put(jnp.zeros(shp, jnp.float32), sharding)
    jax.block_until_ready((p, m, v))
    return p, m, v


print(f"[zero] W={W} GPUs, blob = {param_gb:.1f} GB total (param+m+v)")

base = live_gb()
p, m, v = make(repl)
print(f"[zero] REPLICATED resident: {live_gb() - base:6.2f} GB/device")
del p, m, v
gc.collect()

base = live_gb()
p, m, v = make(shard)
print(
    f"[zero] SHARDED    resident: {live_gb() - base:6.2f} GB/device  (expect ~1/{W} of replicated)"
)


# confirm a gather-on-use forward composes with sharded params
@jax.jit
def fwd(p, x):
    # x: (B, 8192) per block; contract over the sharded param -> XLA gathers as needed
    return jnp.einsum("bk,bkn->bn", x, p.astype(jnp.float32))


x = jax.device_put(jnp.ones((B, 8192), jnp.float32), NamedSharding(mesh, P("fsdp", None)))
y = fwd(p, x)
jax.block_until_ready(y)
print(f"[zero] sharded gather-forward OK, out shape {y.shape}")
