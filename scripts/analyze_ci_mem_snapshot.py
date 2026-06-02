"""Replay a torch.cuda.memory._snapshot device trace to the peak-allocated moment,
then bucket the live allocations by their Python stack frames.

The snapshot's `segments` are a quiescent post-step view (active ~0). The dynamic
peak (activations + caches + CI-value tensors + imp-min temps) only shows up by
replaying `device_traces[0]` alloc/free events forward and recording the live set
at the moment cumulative live size is maximal.
"""

import pickle
import sys
from collections import defaultdict


def pick_frame(frames: list[dict]) -> str:
    """Most informative project frame for bucketing an allocation."""
    interesting = []
    for fr in frames:
        fn = fr.get("filename", "??")
        name = fr.get("name", "")
        if fn in ("??", "") or "memory_snapshot" in fn:
            continue
        short = fn.split("/")[-1]
        # Prefer project frames; skip pure torch internals for the headline label.
        is_project = "param_decomp" in fn or "three_pool" in fn
        interesting.append((is_project, f"{short}:{name}"))
    project = [x for x in interesting if x[0]]
    if project:
        return project[0][1]
    if interesting:
        return interesting[0][1]
    return "<no-frame>"


def pick_label(frames: list[dict]) -> str:
    """Coarse semantic bucket from the frame stack."""
    joined = " | ".join(
        f"{fr.get('filename', '').split('/')[-1]}:{fr.get('name', '')}" for fr in frames
    )
    rules = [
        ("imp_min", ("importance_minimality", "per_component_lp_sums", "annealed_pnorm")),
        (
            "ci_value_tensors",
            ("calc_causal_importances", "lower_leaky", "upper_leaky", "_split_and_sigmoid"),
        ),
        ("input_projector", ("_input_projector", "input_projector")),
        ("output_head", ("_output_head", "output_head")),
        ("transformer_block", ("TransformerBlock", "_blocks", "attn", "mlp", "rms_norm")),
        (
            "target_fwd_Hcache",
            ("forward_with_pre_weight_acts", "_target_fwd_and_cache", "_prefetch_next_h"),
        ),
        ("adam_optimizer", ("adam", "_single_tensor", "optimizer")),
        ("backward", ("backward", "Backward", "autograd")),
        ("grad_reduce", ("all_reduce_ci_fn_grads", "cross_pool_clip", "reduce")),
        ("portal_send_recv", ("portal", "_send_ci", "_recv_g_ci", "send", "recv")),
    ]
    for label, keys in rules:
        if any(k in joined for k in keys):
            return label
    return "other"


def analyze(path: str):
    with open(path, "rb") as f:
        snap = pickle.load(f)

    segs = snap["segments"]
    reserved = sum(s["total_size"] for s in segs)

    tr = snap["device_traces"][0]
    live: dict[int, dict] = {}
    cur = 0
    peak = 0
    peak_live: dict[int, dict] | None = None
    for e in tr:
        a = e["action"]
        if a == "alloc":
            live[e["addr"]] = e
            cur += e["size"]
            if cur > peak:
                peak = cur
                peak_live = dict(live)
        elif a == "free_completed":
            old = live.pop(e["addr"], None)
            if old is not None:
                cur -= old["size"]

    assert peak_live is not None
    by_label = defaultdict(int)
    by_frame = defaultdict(int)
    for e in peak_live.values():
        fr = e.get("frames", [])
        by_label[pick_label(fr)] += e["size"]
        by_frame[pick_frame(fr)] += e["size"]

    print(f"=== {path.split('/')[-1]} ===")
    print(f"reserved (segments)        = {reserved / 1e9:8.2f} GB")
    print(f"trace peak LIVE (replayed) = {peak / 1e9:8.2f} GB  ({len(peak_live)} live blocks)")
    print(
        f"window covered: {len(tr)} events (trace cap = 200000 → static allocs may be out of window)\n"
    )

    print("--- by semantic bucket (peak live) ---")
    for label, sz in sorted(by_label.items(), key=lambda x: -x[1]):
        print(f"  {label:24s} {sz / 1e9:8.2f} GB  ({100 * sz / peak:5.1f}%)")

    print("\n--- top 18 frames (peak live) ---")
    for fr, sz in sorted(by_frame.items(), key=lambda x: -x[1])[:18]:
        print(f"  {sz / 1e9:7.2f} GB  {fr}")
    print()


if __name__ == "__main__":
    paths = sys.argv[1:] or [
        "/mnt/data/artifacts/mechanisms/param-decomp/mem_profile/chunkwise-smoke-fixed/mem_rank48.pickle"
    ]
    for p in paths:
        analyze(p)
