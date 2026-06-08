"""Stage 7: multi-node bring-up. jax.distributed under SLURM + a cross-node collective.

Run under srun (1 task per GPU). Verifies:
  * jax.distributed.initialize() auto-detects the SLURM world,
  * every process sees 1 local GPU and the full global device count,
  * a psum across the WHOLE mesh (spanning nodes) returns the right answer
    (sum of all global device ids) -- i.e. cross-node NCCL works.

Process 0 prints the verdict.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

jax.distributed.initialize()  # SLURM auto-detect

pid = jax.process_index()
nproc = jax.process_count()
ndev = jax.device_count()
local = jax.local_device_count()

if pid == 0:
    print(f"[p0] processes={nproc} global_devices={ndev} local_per_proc={local}")
    print(f"[p0] device kinds: {sorted(set(d.device_kind for d in jax.devices()))}")

mesh = Mesh(np.array(jax.devices()), axis_names=("d",))


@jax.jit
def all_reduce_ids():
    # each device emits its own global id, psum across the full (multi-node) mesh
    def f(_):
        idx = jax.lax.axis_index("d")
        return jax.lax.psum(idx, "d")[None]

    dummy = jnp.zeros((ndev,))
    sm = shard_map(f, mesh=mesh, in_specs=P("d"), out_specs=P("d"), check_rep=False)
    return sm(dummy)


result = all_reduce_ids()
got = int(result.addressable_shards[0].data[0])  # every device holds the same sum
expected = ndev * (ndev - 1) // 2

if pid == 0:
    ok = got == expected and ndev == nproc * local
    print(f"[p0] psum(device_ids) across {ndev} devices = {got}, expected {expected}")
    print(f"[p0] STAGE 7 ({nproc} procs, {ndev} GPUs):", "PASS" if ok else "FAIL")

jax.distributed.shutdown()
