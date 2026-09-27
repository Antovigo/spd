#!/usr/bin/env python
"""Graft a hidden-only checkpoint into a DUAL one, so the run continues under -05's objective.

    python graft_dual_from_hidden_only.py <src ckpts dir> <dst ckpts dir> [--step N] [--dry-run]

WHY THIS IS A GRAFT AND NOT A RESUME OR A FINE-TUNE. The objective is built from the INVOKED
config, never from the checkpoint (`run.py` `_init_or_restore_state` calls `init_train_state(pd,
...)` and only then overwrites leaves from disk), and `pin_launch_config` byte-compares against
the config you invoke — so a FRESH run dir holding the dual config and this grafted checkpoint
resumes at the parent's step under the dual objective. What the stock paths cannot do:
`resume` refuses a changed config on the same id, and `resume_provenance` (SPEC S33) keeps the
decomposition ONLY -- fresh optimizer, fresh adversaries, step 0, schedules re-annealed.

WHAT IS COPIED (hidden -> output), so the output role starts as an exact copy of the head the
hidden-only run actually trained and the two diverge from the first step:
  decomposition  ci_fn.chunks.out_ws[i] <- hidden_out_ws[i],  out_bs[i] <- hidden_out_bs[i]
  training       ci_fn_opt_state[0].{mu,nu}.chunks.{out_ws,out_bs}[i] <- hidden counterparts
                 adversaries["MergedStochasticSubsetPPGDReconLoss"]
                     <- adversaries["HiddenMergedStochasticPPGDRecon"]
                 adversaries["nontarget/MergedStochasticSubsetPPGDReconLoss"]
                     <- adversaries["nontarget_hidden/NontargetHiddenMergedStochasticPPGDRecon"]
The adversary subtrees are copied whole (`sources` + `opt_state`), so the delta channel and the
Adam moments come along. Shapes already agree: both streams' merged terms are `bsc` at the same
batch and sequence length, target (128, 5, C) and non-target (128, 64, C).

`step` is NOT touched: the graft keeps the parent's step so every schedule continues mid-anneal.
The trunk (`in_proj_*`, `blocks`) is shared by both heads and needs no copy.

RUN IT ON THE POD, with the GPUs visible: the checkpoint carries a 4-device topology and orbax
refuses to restore it onto a single CPU device.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import orbax.checkpoint as ocp

HEAD_PAIRS = (("out_ws", "hidden_out_ws"), ("out_bs", "hidden_out_bs"))
ADVERSARY_PAIRS = (
    ("MergedStochasticSubsetPPGDReconLoss", "HiddenMergedStochasticPPGDRecon"),
    (
        "nontarget/MergedStochasticSubsetPPGDReconLoss",
        "nontarget_hidden/NontargetHiddenMergedStochasticPPGDRecon",
    ),
)


def _copy_heads(chunks: dict, label: str) -> None:
    for dst_key, src_key in HEAD_PAIRS:
        assert dst_key in chunks and src_key in chunks, (label, dst_key, src_key, list(chunks))
        dst, src = chunks[dst_key], chunks[src_key]
        assert len(dst) == len(src), (label, len(dst), len(src))
        for i, (d, s) in enumerate(zip(dst, src, strict=True)):
            assert d.shape == s.shape and d.dtype == s.dtype, (label, i, d.shape, s.shape)
        chunks[dst_key] = type(dst)(copy.deepcopy(s) for s in src)
        print(f"  {label}.{dst_key} <- {src_key}  ({len(src)} slots)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_mgr = ocp.CheckpointManager(
        args.src.resolve(), options=ocp.CheckpointManagerOptions(read_only=True)
    )
    step = args.step if args.step is not None else src_mgr.latest_step()
    assert step in src_mgr.all_steps(), (step, src_mgr.all_steps())
    print(f"restoring {args.src} step {step}")
    restored = src_mgr.restore(
        step,
        args=ocp.args.Composite(
            decomposition=ocp.args.StandardRestore(), training=ocp.args.StandardRestore()
        ),
    )
    dec, tr = restored["decomposition"], restored["training"]
    assert int(tr["step"]) == step, (int(tr["step"]), step)

    print("decomposition:")
    _copy_heads(dec["ci_fn"]["chunks"], "ci_fn")
    print("ci_fn optimizer moments:")
    for moment in ("mu", "nu"):
        _copy_heads(tr["ci_fn_opt_state"][0][moment]["chunks"], f"ci_fn_opt_state[0].{moment}")

    print("adversaries:")
    adv = tr["adversaries"]
    for dst_key, src_key in ADVERSARY_PAIRS:
        assert src_key in adv, (src_key, list(adv))
        assert dst_key not in adv, f"{dst_key} already present — refusing to overwrite"
        adv[dst_key] = copy.deepcopy(adv[src_key])
        n = len(adv[dst_key]["sources"])
        print(f"  {dst_key} <- {src_key}  ({n} sites)")
    print(f"  adversaries now: {sorted(adv)}")

    # The heads must now agree exactly, and `step` must be untouched.
    for dst_key, src_key in HEAD_PAIRS:
        for d, s in zip(
            dec["ci_fn"]["chunks"][dst_key], dec["ci_fn"]["chunks"][src_key], strict=True
        ):
            assert np.array_equal(np.asarray(d), np.asarray(s)), dst_key
    assert int(tr["step"]) == step
    print(f"verified: output head == hidden head, step still {step}")

    if args.dry_run:
        print("dry run — nothing written")
        return
    args.dst.mkdir(parents=True, exist_ok=True)
    dst_mgr = ocp.CheckpointManager(
        args.dst.resolve(), options=ocp.CheckpointManagerOptions(enable_async_checkpointing=False)
    )
    dst_mgr.save(
        step,
        args=ocp.args.Composite(
            decomposition=ocp.args.StandardSave(dec), training=ocp.args.StandardSave(tr)
        ),
    )
    dst_mgr.wait_until_finished()
    print(f"wrote {args.dst}/{step}")


if __name__ == "__main__":
    main()
