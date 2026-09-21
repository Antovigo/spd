"""The read vectors of a CI filter's alive components, kinds q/k/v/gate/up, from the orbax
checkpoint on CPU.

    python -m param_decomp.arith_repr.scripts.extract_v --ckpt <run>/ckpts/<step>/decomposition
        --alive <filter>/alive/alive.json --out V_alive.npz

Writes `{site: V (4096, n_alive) fp32, site + ".ids": alive component ids}` for every site
whose kind reads the residual stream."""

import argparse
import json
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
import orbax.checkpoint as ocp

READ_KINDS = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "gate_proj": "gate", "up_proj": "up"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--alive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    jax.config.update("jax_platforms", "cpu")
    sharding = jax.sharding.SingleDeviceSharding(jax.devices("cpu")[0])

    alive = json.loads(args.alive.read_text())["components"]
    ck = ocp.PyTreeCheckpointer()
    meta = cast(Any, ck.metadata(args.ckpt).item_metadata)
    stacks = meta["components"]["stacks"]
    f32 = np.dtype(np.float32)
    # Restore only the V leaf of the read kinds; every other leaf is skipped.
    wanted = set(READ_KINDS.values())
    restore_args = {
        "components": {
            "stacks": {
                kind: [
                    ocp.ArrayRestoreArgs(sharding=sharding, dtype=f32)
                    if kind in wanted and i == 0
                    else None
                    for i in range(2)
                ]
                for kind in stacks
            }
        }
    }
    item = {
        "components": {
            "stacks": {kind: [stacks[kind][0] if kind in wanted else None, None] for kind in stacks}
        }
    }
    try:
        restored: Any = ck.restore(
            args.ckpt,
            args=ocp.args.PyTreeRestore(item=item, restore_args=restore_args, transforms={}),
        )
    except Exception as exc:  # noqa: BLE001 — partial restore is a convenience, not a contract
        print(f"partial restore failed ({exc!r}); restoring every stack", flush=True)
        full_args = {
            "components": {
                "stacks": {
                    kind: [ocp.ArrayRestoreArgs(sharding=sharding, dtype=f32)] * 2
                    for kind in stacks
                }
            }
        }
        full_item = {"components": {"stacks": {k: list(v) for k, v in stacks.items()}}}
        restored = ck.restore(
            args.ckpt,
            args=ocp.args.PyTreeRestore(item=full_item, restore_args=full_args, transforms={}),
        )
    out: dict[str, np.ndarray] = {}
    for site, ids in alive.items():
        kind_name = site.split(".")[-1]
        if kind_name not in READ_KINDS:
            continue
        layer = int(site.split(".")[1])
        V = np.asarray(restored["components"]["stacks"][READ_KINDS[kind_name]][0][layer])
        ids_arr = np.asarray(ids, np.int32)
        out[site] = V[:, ids_arr].astype(np.float32)
        out[site + ".ids"] = ids_arr
        print(site, out[site].shape, flush=True)
    np.savez(args.out, **cast(dict[str, Any], out))
    print("saved", args.out, sum(v.shape[1] for k, v in out.items() if not k.endswith(".ids")))


if __name__ == "__main__":
    main()
