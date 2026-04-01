"""Per-matrix reconstruction error: replace one matrix at a time with its rounded-mask decomposition.

For each decomposed matrix, runs a forward pass where ONLY that matrix uses rounded CI masks
(all other matrices keep original target weights). Collects per-example L0 and KL divergence.

Produces a grid figure: columns = matrix types (e.g. q_proj, k_proj, down_proj, ...),
rows = layers. Each subplot is a scatter plot of per-example KL (y) vs L0 (x).

Usage:
    python -m spd.scripts.decomposition_stress_test.per_matrix_recon <model_path> \
        --n-batches 5 --ci-thr 0.01
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from spd.configs import LMTaskConfig
from spd.data import train_loader_and_tokenizer
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import make_mask_infos


def parse_module_name(name: str) -> tuple[int, str]:
    """Extract (layer_index, matrix_type) from a module path.

    E.g. "model.layers.3.self_attn.q_proj" -> (3, "self_attn.q_proj")
         "transformer.h.0.mlp.c_fc"        -> (0, "mlp.c_fc")
         "layers.1.mlp_in"                  -> (1, "mlp_in")
    """
    match = re.search(r"\.(\d+)\.(.*)", name)
    assert match is not None, f"Cannot parse layer index from module name: {name}"
    return int(match.group(1)), match.group(2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-matrix reconstruction error with rounded CI masks"
    )
    parser.add_argument("model_path", help="SPD model path (wandb or local)")
    parser.add_argument("--n-batches", type=int, default=5, help="Number of batches to process")
    parser.add_argument(
        "--batch-size", type=int, help="Batch size (default: config eval_batch_size)"
    )
    parser.add_argument("--ci-thr", type=float, default=0.01, help="CI rounding threshold")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out-dir", type=Path, default=None, help="Output directory for plot (default: model dir)"
    )
    args = parser.parse_args()

    model_path = (
        str(Path(args.model_path).expanduser()) if ":" not in args.model_path else args.model_path
    )
    run_info = SPDRunInfo.from_path(model_path)
    config = run_info.config
    assert isinstance(config.task_config, LMTaskConfig), "Only LM experiments are supported"

    device = torch.device(args.device)
    model = ComponentModel.from_run_info(run_info).to(device)
    model.eval()

    batch_size = args.batch_size or config.eval_batch_size
    loader, _tokenizer = train_loader_and_tokenizer(config, batch_size)

    module_names = model.target_module_paths

    # Per-module accumulators: module_name -> list of (l0_per_example, kl_per_example)
    per_module_l0: dict[str, list[torch.Tensor]] = {name: [] for name in module_names}
    per_module_kl: dict[str, list[torch.Tensor]] = {name: [] for name in module_names}

    for i, batch in enumerate(loader):
        if i >= args.n_batches:
            break
        input_ids = batch["input_ids"].to(device)
        print(f"Batch {i + 1}/{args.n_batches}")

        with torch.no_grad():
            # Get target logits and CI values
            cached = model(input_ids, cache_type="input")
            target_logits = cached.output
            pre_weight_acts = cached.cache

            ci = model.calc_causal_importances(pre_weight_acts, sampling="continuous")
            ci_lower = ci.lower_leaky

            p = torch.softmax(target_logits, dim=-1)
            p_safe = p.clamp(min=1e-12)
            log_p = p_safe.log()

            # For each module, replace only that one with rounded masks
            for module_name in module_names:
                rounded_ci = (ci_lower[module_name] > args.ci_thr).float()
                mask_infos = make_mask_infos({module_name: rounded_ci})
                masked_logits = model(input_ids, mask_infos=mask_infos)

                # Per-position KL, then mean over positions -> per-example
                log_q = F.log_softmax(masked_logits, dim=-1)
                per_pos_kl = (p * (log_p - log_q)).sum(dim=-1)  # (B, S)
                per_example_kl = per_pos_kl.mean(dim=-1)  # (B,)

                # Per-position L0 for this module, then mean over positions -> per-example
                per_pos_l0 = rounded_ci.sum(-1)  # (B, S)
                per_example_l0 = per_pos_l0.mean(dim=-1)  # (B,)

                per_module_l0[module_name].append(per_example_l0.cpu())
                per_module_kl[module_name].append(per_example_kl.cpu())

    # Concatenate across batches
    l0_arrays = {name: torch.cat(vals).numpy() for name, vals in per_module_l0.items()}
    kl_arrays = {name: torch.cat(vals).numpy() for name, vals in per_module_kl.items()}

    # Parse module names into (layer, type) for grid layout
    parsed = {name: parse_module_name(name) for name in module_names}
    all_layers = sorted({layer for layer, _ in parsed.values()})
    all_types = sorted({mtype for _, mtype in parsed.values()})
    layer_to_row = {layer: i for i, layer in enumerate(all_layers)}
    type_to_col = {mtype: j for j, mtype in enumerate(all_types)}

    n_rows = len(all_layers)
    n_cols = len(all_types)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False)

    # Hide all axes first, then show only the ones with data
    for ax_row in axes:
        for ax in ax_row:
            ax.set_visible(False)

    for module_name in module_names:
        layer, mtype = parsed[module_name]
        row = layer_to_row[layer]
        col = type_to_col[mtype]
        ax = axes[row][col]
        ax.set_visible(True)

        l0 = l0_arrays[module_name]
        kl = kl_arrays[module_name]
        ax.scatter(l0, kl, alpha=0.5, s=12, edgecolors="none")
        ax.set_xlabel("L0")
        ax.set_ylabel("KL divergence")
        ax.set_title(f"layer {layer} / {mtype}", fontsize=9)

    fig.suptitle(
        f"Per-matrix reconstruction (rounded threshold={args.ci_thr})",
        fontsize=12,
    )
    fig.tight_layout()

    out_dir = args.out_dir or run_info.checkpoint_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "per_matrix_recon.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
