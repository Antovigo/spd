"""Probe: cost of cross-sub-mesh `jax.reshard` inside a jit (sub-mesh viability).

Two sub-meshes over disjoint device halves; reshard a CI-values-sized tensor A->B inside a
jitted step. Measures median latency + asserts the transfer is device-to-device (host transfer
would raise under the guard). Tells us whether the sub-mesh route's cross-pool hand-offs are cheap.
"""

import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

devs = jax.devices()
assert len(devs) >= 8, f"need >=8 GPUs, got {len(devs)}"
half = len(devs) // 2
# A Mesh is a named array of devices. Two Meshes over DISJOINT device halves = two sub-meshes
# (jax's version of torch's separate rank groups, e.g. "main pool" vs "chunk pool").
# axis_types=Explicit opts into jax's explicit-sharding system, which `reshard` requires (the
# default Auto axes let XLA infer shardings; reshard needs you to name them explicitly).
mesh_a = Mesh(np.array(devs[:half]), ("dp",), axis_types=(AxisType.Explicit,))
mesh_b = Mesh(np.array(devs[half : 2 * half]), ("dp",), axis_types=(AxisType.Explicit,))
# A sharding pairs (which mesh) with (how an array's axes map to mesh axes): P("dp") splits the
# array's leading axis across the "dp" axis; P() would replicate the whole array on every device.
sh_a = NamedSharding(mesh_a, P("dp"))
sh_b = NamedSharding(mesh_b, P("dp"))

# A few realistic hand-off sizes (bf16): ci-values (batch,seq,C), g_VU (C,ffn), full V/U-ish.
shapes = {
    "ci_values (16,1024,8192)": (16, 1024, 8192),
    "g_vu (8192,6400)": (8192, 6400),
    "vu_block (6400,8192)": (6400, 8192),
}

for label, shape in shapes.items():
    x = jax.device_put(jnp.ones(shape, jnp.bfloat16), sh_a)

    @jax.jit
    def move(t):
        # reshard = move data onto a DIFFERENT mesh's sharding (mesh_a -> mesh_b). This is the
        # cross-sub-mesh hop we're pricing; a same-mesh collective (psum/ppermute) can't cross meshes.
        return jax.reshard(t, sh_b)

    y = move(x)
    jax.block_until_ready(y)  # jax dispatch is async — block, else you'd time dispatch not the move

    # d2d proof: if reshard secretly routed through host RAM, this guard raises — so a clean run
    # confirms the move stayed GPU->GPU (what makes the sub-mesh route actually viable).
    with jax.transfer_guard_device_to_host("disallow"):
        ts = []
        for _ in range(30):
            t0 = time.perf_counter()
            y = move(x)
            jax.block_until_ready(y)
            ts.append(time.perf_counter() - t0)
    mb = x.nbytes / 1e6
    print(f"[reshard] {label:28s} {mb:7.1f} MB  A->B  median {1e3 * statistics.median(ts):7.3f} ms")

print("[reshard] done (all transfers were device-to-device; host transfer would have raised)")
