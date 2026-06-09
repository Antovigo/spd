"""Probe: heterogeneous (uneven) rank-groups on ONE flat mesh via `axis_index_groups` (note-2).

The single-mesh route to torch-style per-pool DDP needs `psum` to reduce within *uneven*
subgroups (e.g. main=4 ranks, chunk0=2, chunk1=2). XLA's `replica_groups` historically required
EQUAL-size groups — so this probe checks whether even AND uneven groups work. If uneven fails,
the "single mesh + heterogeneous batch slicing" idea needs padding or sub-meshes instead.
"""

import traceback
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import shard_map  # jax.experimental.shard_map is deprecated since v0.8
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

devs = jax.devices()
W = len(devs)
mesh = Mesh(np.array(devs), ("world",))


def test(groups, label):
    # shard_map runs f once per device-shard; in/out_specs say each device owns one slice of the
    # "world" axis. Inside f, collectives communicate over the named mesh axes.
    @partial(shard_map, mesh=mesh, in_specs=P("world"), out_specs=P("world"))
    def f(x):
        # axis_index_groups makes psum reduce WITHIN each listed subgroup independently — jax's way
        # to express torch-style per-pool rank groups on a single mesh (the question: uneven sizes?).
        return jax.lax.psum(x, "world", axis_index_groups=groups)

    x = jnp.arange(W, dtype=jnp.float32)  # device i holds value i
    try:
        y = np.array(jax.jit(f)(x))
        print(f"[hetero] {label:14s} groups={groups} -> per-device sums {y.tolist()}")
    except Exception as e:  # noqa: BLE001 - probe wants to see the failure mode
        print(f"[hetero] {label:14s} FAILED: {type(e).__name__}: {str(e)[:180]}")
        traceback.print_exc()


# even groups: expect [6,6,6,6, 22,22,22,22]
test([list(range(W // 2)), list(range(W // 2, W))], "even 4+4")
# uneven groups: expect [6,6,6,6, 9,9, 13,13] IF uneven is allowed
test([[0, 1, 2, 3], [4, 5], [6, 7]], "uneven 4+2+2")
print("[hetero] done")
