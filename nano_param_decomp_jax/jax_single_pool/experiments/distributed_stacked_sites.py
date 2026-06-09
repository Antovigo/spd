"""Single-pool VPD, GSPMD-sharded across a (multi-node) device mesh.

The SPMD-collapse runner: the same `jax.jit`'d step as the single-device smoke,
but with the batch sharded `P('dp')` and params + PGD sources replicated. No
manual collectives — XLA inserts the grad all-reduce because the mean-losses
reduce over the sharded batch (axis 1 of the stacked-site `x`). Mirrors
`jax_spike/stage8_train_distributed.py` over the Equinox-typed step.

CORRECTNESS SIGNAL: pure data-parallelism is GPU-count-invariant. For a FIXED
global batch + seed (and a *replicated* source scope — broadcast_across_batch),
the loss trajectory must match at 1 / N GPUs. Run at several scales and diff.

Launch (via jax_spike/remote/gpu.sh or srun):
  NODES=1 GPN=1 ... python -m jax_single_pool.experiments.distributed_stacked_sites
  NODES=1 GPN=8 ...  (same args)   -> trajectory must match
  NODES=2 GPN=8 ...  (same args)   -> trajectory must match
"""

import argparse
import time

import jax
import jax.experimental.multihost_utils
import optax

from jax_single_pool.experiments.toy_stacked_sites import COEFFS, PGD_CFG, init_everything
from jax_single_pool.pgd import init_pgd_state
from jax_single_pool.scopes import BroadcastAcrossBatchScope
from jax_single_pool.sharding import dp_mesh, init_distributed, replicate, shard_batch
from jax_single_pool.step import init_train_state, make_step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--global_batch", type=int, default=256)
    ap.add_argument("--lr_main", type=float, default=3e-3)
    ap.add_argument("--lr_ci", type=float, default=3e-3)
    args = ap.parse_args()

    under_slurm = init_distributed()
    mesh = dp_mesh()
    ndev = mesh.devices.size
    is0 = jax.process_index() == 0

    key = jax.random.PRNGKey(0)
    k_init, k_pgd = jax.random.split(key)
    # init_everything builds V/U/CI + a B=64 PGD state; rebuild the source for the
    # global batch under the replicated broadcast scope.
    decomp, ci, _, source_c = init_everything(k_init)
    n_sites, d_in = decomp.V.shape[0], decomp.V.shape[1]
    pgd = init_pgd_state(
        k_pgd, BroadcastAcrossBatchScope(), n_sites, source_c, (args.global_batch,)
    )

    opt_main = optax.adam(args.lr_main)
    opt_ci = optax.adam(args.lr_ci)
    state = init_train_state(decomp, ci, pgd, opt_main, opt_ci)
    # params, optimizer moments, and the (leading-dim-1 broadcast) sources all
    # replicate; only the batch is sharded.
    state = jax.tree.map(lambda a: replicate(a, mesh), state)

    step = make_step(COEFFS, PGD_CFG, opt_main, opt_ci, source_c=source_c, use_delta_component=True)

    x_full = jax.random.normal(jax.random.PRNGKey(42), (n_sites, args.global_batch, d_in))
    x = shard_batch(x_full, mesh, batch_axis=1)

    if is0:
        print(
            f"[p0] mesh={ndev} dev | global_batch={args.global_batch} S={n_sites} "
            f"d={d_in} C={source_c - 1} n_warmup={PGD_CFG.n_warmup}"
        )

    losses: list[float] = []
    t0 = None
    for s in range(args.steps):
        state, m = step(state, x, jax.random.PRNGKey(1000 + s))
        if s == 0:
            jax.block_until_ready(m["total"])
            t0 = time.time()
        losses.append(float(m["total"]))
        if is0 and (s < 5 or s % 10 == 0):
            print(
                f"[p0] step {s:3d} | total {float(m['total']):.5f} | "
                f"faith {float(m['faith']):.3e} stoch {float(m['stoch']):.3e} "
                f"ppgd {float(m['ppgd']):.3e}"
            )

    jax.block_until_ready(state.pgd.sources)
    assert t0 is not None
    dt = (time.time() - t0) / (args.steps - 1)
    if is0:
        print(f"[p0] TRAJECTORY[:6] = {[round(v, 5) for v in losses[:6]]}")
        print(f"[p0] final {losses[-1]:.5f} (start {losses[0]:.5f})")
        print(
            f"[p0] {dt * 1e3:.2f} ms/step | {args.global_batch / dt:,.0f} samples/s on {ndev} GPU"
        )
        print(f"[p0] DIST ({ndev} GPU): {'PASS' if losses[-1] < losses[0] else 'FAIL'}")

    if under_slurm:
        jax.experimental.multihost_utils.sync_global_devices("dist_done")
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
