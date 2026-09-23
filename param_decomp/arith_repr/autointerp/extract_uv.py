"""V and U of every alive component, every site kind, from the orbax checkpoint on CPU.

    python -m param_decomp.arith_repr.autointerp.extract_uv --ckpt <run>/ckpts/<step>/decomposition
        --alive <filter>/alive/alive.json --out uv_alive.npz

Writes `{site + ".V"}` (d_in, n_alive), `{site + ".U"}` (n_alive, d_out), `{site + ".ids"}`, all fp32,
for the 224 sites of `alive.json`."""

import argparse
import json
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
import orbax.checkpoint as ocp

KINDS = {
    "q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o",
    "gate_proj": "gate", "up_proj": "up", "down_proj": "down",
}  # fmt: skip


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
    stacks = cast(Any, ck.metadata(args.ckpt).item_metadata)["components"]["stacks"]
    f32 = np.dtype(np.float32)
    restore_args = {
        "components": {
            "stacks": {k: [ocp.ArrayRestoreArgs(sharding=sharding, dtype=f32)] * 2 for k in stacks}
        }
    }
    item = {"components": {"stacks": {k: list(v) for k, v in stacks.items()}}}
    restored: Any = ck.restore(
        args.ckpt, args=ocp.args.PyTreeRestore(item=item, restore_args=restore_args, transforms={})
    )
    out: dict[str, np.ndarray] = {}
    for site, ids in alive.items():
        kind = KINDS[site.split(".")[-1]]
        layer = int(site.split(".")[1])
        st = restored["components"]["stacks"][kind]
        ids_arr = np.asarray(ids, np.int32)
        out[site + ".V"] = np.asarray(st[0][layer])[:, ids_arr].astype(np.float32)
        out[site + ".U"] = np.asarray(st[1][layer])[ids_arr].astype(np.float32)
        out[site + ".ids"] = ids_arr
        print(site, out[site + ".V"].shape, out[site + ".U"].shape, flush=True)
    np.savez(args.out, **cast(dict[str, Any], out))
    print("saved", args.out)


if __name__ == "__main__":
    main()
