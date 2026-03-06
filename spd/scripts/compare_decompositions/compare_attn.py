"""Compare attention components across two SPD decompositions.

Loads two runs, computes CI on a prompt, identifies active components per head,
and produces cosine similarity heatmaps + summary statistics.

Usage:
    uv run python -m spd.scripts.compare_decompositions.compare_attn \
        "wandb:entity/project/runs/run_id_1" \
        "wandb:entity/project/runs/run_id_2" \
        --prompt "The cat sat on the mat" \
        --ci_threshold 0.1 \
        --output_dir /tmp/compare_decompositions
"""

from pathlib import Path

import fire
import torch

from spd.scripts.compare_decompositions.utils import (
    compare_attention_heads,
    compute_ci,
    get_active_component_indices,
    get_attn_info,
    get_tokenizer,
    load_decomposition,
    parse_attn_module_path,
    plot_comparison_summary,
    plot_head_comparisons,
)


@torch.no_grad()
def main(
    wandb_path_a: str,
    wandb_path_b: str,
    *,
    label_a: str | None = None,
    label_b: str | None = None,
    prompt: str,
    ci_threshold: float = 0.1,
    device: str = "cuda",
    output_dir: str = "/tmp/compare_decompositions",
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load models
    decomp_a = load_decomposition(wandb_path_a, label_a, device)
    decomp_b = load_decomposition(wandb_path_b, label_b, device)
    print(f"Run A: {decomp_a.label}")
    print(f"Run B: {decomp_b.label}")

    tokenizer = get_tokenizer(decomp_a)
    n_heads, n_kv_heads, d_head = get_attn_info(decomp_a.model)
    print(f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, d_head={d_head}")

    # Compute CI & find active components
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    print(f"Prompt: {prompt!r}  (tokens: {tokens.shape[1]})")

    ci_a = compute_ci(decomp_a.model, tokens, decomp_a.run_info.config.sampling)
    ci_b = compute_ci(decomp_b.model, tokens, decomp_b.run_info.config.sampling)

    active_a = get_active_component_indices(ci_a, ci_threshold)
    active_b = get_active_component_indices(ci_b, ci_threshold)

    print(f"\nActive attention components (CI > {ci_threshold}):")
    for path in sorted(active_a):
        parsed = parse_attn_module_path(path)
        if parsed is None:
            continue
        layer_idx, proj_type = parsed
        n_a = len(active_a[path])
        n_b = len(active_b.get(path, []))
        print(f"  L{layer_idx}.{proj_type}: {n_a} ({decomp_a.label}), {n_b} ({decomp_b.label})")

    # Compare
    results = compare_attention_heads(decomp_a, decomp_b, active_a, active_b)
    print(f"\n{len(results)} projection comparisons")

    # Save heatmaps per (layer, proj)
    for r in results:
        fig = plot_head_comparisons(r)
        path = out / f"L{r.layer_idx}_{r.proj_type}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")

    # Summary
    if results:
        fig = plot_comparison_summary(results)
        path = out / "summary.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")

        print(f"\n{'Layer.Proj':<25} {'Max |cos|':>10} {'Mean best':>10}")
        print("-" * 47)
        for r in results:
            per_pair_max = r.sim_tensor.abs().amax(dim=(-2, -1))  # (Na, Nb)
            max_cos = per_pair_max.max().item()
            best_a = per_pair_max.max(dim=1).values.mean().item()
            best_b = per_pair_max.max(dim=0).values.mean().item()
            mean_best = (best_a + best_b) / 2
            label = f"L{r.layer_idx}.{r.proj_type}"
            print(f"{label:<25} {max_cos:>10.4f} {mean_best:>10.4f}")


if __name__ == "__main__":
    fire.Fire(main)
