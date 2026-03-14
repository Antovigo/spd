"""Validate a targeted decomposition by individually ablating each alive component.

Measures the effect on both target (prompts) and nontarget data. Outputs per-component,
per-example KL divergence and next-token predictions to TSV files in the model's run directory.

Usage:
    uv run python -m spd.scripts.decomposition_stress_test.validate_ablations \
        <wandb_path> --prompts path/to/prompts.txt --n-batches 5
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.configs import LMTaskConfig
from spd.experiments.lm.lm_decomposition import _create_lm_loaders
from spd.experiments.lm.prompts_dataset import load_prompts_dataset
from spd.models.component_model import ComponentModel, OutputWithCache, SPDRunInfo
from spd.models.components import WeightDeltaAndMask, make_mask_infos
from spd.utils.general_utils import calc_kl_divergence_lm


def find_alive_components(
    model: ComponentModel,
    dataset: Dataset,
    ci_thr: float,
    device: torch.device,
) -> list[tuple[str, int]]:
    """Find components with max CI > threshold across all prompts and positions."""
    max_ci: dict[str, Tensor] = {}
    for module_name in model.target_module_paths:
        n_c = model.module_to_c[module_name]
        max_ci[module_name] = torch.zeros(n_c, device=device)

    for i in range(len(dataset)):
        input_ids = dataset[i]["input_ids"].unsqueeze(0).to(device)
        output_with_cache: OutputWithCache = model(input_ids, cache_type="input")
        ci = model.calc_causal_importances(
            output_with_cache.cache, sampling="continuous"
        ).upper_leaky
        for module_name, ci_vals in ci.items():
            # ci_vals: (1, seq_len, C) — take max over batch and seq
            module_max = ci_vals.amax(dim=tuple(range(ci_vals.ndim - 1)))
            max_ci[module_name] = torch.maximum(max_ci[module_name], module_max)

    alive = []
    for module_name in model.target_module_paths:
        for c_idx in range(model.module_to_c[module_name]):
            if max_ci[module_name][c_idx] > ci_thr:
                alive.append((module_name, c_idx))

    return alive


def extract_layer_index(module_path: str) -> str:
    """Extract numeric layer index from module path, or return the full path."""
    nums = re.findall(r"\d+", module_path)
    return nums[0] if nums else module_path


def ablate_and_measure(
    model: ComponentModel,
    input_ids: Tensor,
    original_logits: Tensor,
    ablate_module: str,
    ablate_idx: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Ablate one component and return (kl_per_example, ablated_logits).

    kl_per_example: (batch,) mean KL over sequence positions.
    """
    weight_deltas = model.calc_weight_deltas()
    batch_seq_shape = input_ids.shape  # (batch, seq)

    component_masks: dict[str, Tensor] = {}
    wdam: dict[str, WeightDeltaAndMask] = {}
    for module_name in model.target_module_paths:
        n_c = model.module_to_c[module_name]
        mask = torch.ones(*batch_seq_shape, n_c, device=device)
        if module_name == ablate_module:
            mask[..., ablate_idx] = 0.0
        component_masks[module_name] = mask
        wdam[module_name] = (
            weight_deltas[module_name],
            torch.ones(*batch_seq_shape, device=device),
        )

    mask_infos = make_mask_infos(component_masks, weight_deltas_and_masks=wdam)
    ablated_logits: Tensor = model(input_ids, mask_infos=mask_infos)

    # KL per (batch, seq), then mean over seq → (batch,)
    kl = calc_kl_divergence_lm(pred=ablated_logits, target=original_logits, reduce=False)
    kl_per_example = kl.mean(dim=-1)

    return kl_per_example, ablated_logits


def sanitize_for_tsv(text: str) -> str:
    """Replace characters that would break TSV column/row structure."""
    return text.replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")


def next_token_info(logits: Tensor, pos: int) -> tuple[int, float]:
    """Return (argmax_token_id, softmax_prob) at the given position."""
    probs = torch.softmax(logits[pos], dim=-1)
    token_id = int(probs.argmax())
    return token_id, probs[token_id].item()


def write_ablation_rows(
    writer: Any,
    model: ComponentModel,
    input_ids: Tensor,
    original_logits: Tensor,
    alive_components: list[tuple[str, int]],
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
) -> None:
    """Ablate each alive component and write rows for every example in the batch."""
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]

    for ablate_module, ablate_idx in alive_components:
        layer = extract_layer_index(ablate_module)
        kl_per_example, ablated_logits = ablate_and_measure(
            model, input_ids, original_logits, ablate_module, ablate_idx, device
        )

        for b in range(batch_size):
            # Find last non-pad position
            pad_id = tokenizer.pad_token_id
            if pad_id is not None:
                non_pad = (input_ids[b] != pad_id).nonzero(as_tuple=True)[0]  # pyright: ignore[reportAttributeAccessIssue]
                last_pos = int(non_pad[-1]) if len(non_pad) > 0 else seq_len - 1
            else:
                last_pos = seq_len - 1

            orig_tok, orig_prob = next_token_info(original_logits[b], last_pos)
            abl_tok, abl_prob = next_token_info(ablated_logits[b], last_pos)

            input_text = sanitize_for_tsv(tokenizer.decode(input_ids[b], skip_special_tokens=True))

            writer.writerow(
                [
                    layer,
                    f"{ablate_module}:{ablate_idx}",
                    input_text,
                    f"{kl_per_example[b].item():.6f}",
                    sanitize_for_tsv(tokenizer.decode([orig_tok])),
                    f"{orig_prob:.4f}",
                    sanitize_for_tsv(tokenizer.decode([abl_tok])),
                    f"{abl_prob:.4f}",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate targeted decomposition via ablations")
    parser.add_argument("model_path", help="Path to decomposed model (wandb or local)")
    parser.add_argument("--prompts", required=True, help="Path to prompts file (one per line)")
    parser.add_argument(
        "--ci-thr", type=float, default=0.01, help="CI threshold for alive components"
    )
    parser.add_argument("--n-batches", type=int, default=10, help="Number of nontarget batches")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Load model
    model_path = str(Path(args.model_path).expanduser())
    print(f"Loading model from {model_path}...")
    run_info = SPDRunInfo.from_path(model_path)
    config = run_info.config
    model = ComponentModel.from_run_info(run_info).to(device).eval()
    output_dir = run_info.checkpoint_path.parent

    # 2. Load tokenizer
    assert config.tokenizer_name is not None
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 3. Load target prompts
    assert isinstance(config.task_config, LMTaskConfig)
    prompts_dataset = load_prompts_dataset(
        Path(args.prompts).expanduser(), tokenizer, config.task_config.max_seq_len
    )
    print(f"Loaded {len(prompts_dataset)} prompts")

    # 4. Find alive components
    print("Computing causal importances to find alive components...")
    with torch.no_grad():
        alive_components = find_alive_components(model, prompts_dataset, args.ci_thr, device)
    print(f"Found {len(alive_components)} alive components (CI > {args.ci_thr})")

    if not alive_components:
        print("No alive components found. Exiting.")
        return

    for module, idx in alive_components:
        print(f"  {module}:{idx}")

    header = [
        "layer",
        "component",
        "input",
        "loss",
        "orig_next_token",
        "orig_prob",
        "ablated_next_token",
        "ablated_prob",
    ]

    # 5-6. Ablate on target data
    target_path = output_dir / "target_ablation_results.tsv"
    print(f"\nAblating on target prompts → {target_path}")
    with open(target_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)

        with torch.no_grad():
            # Process all prompts as individual examples
            for i in tqdm(range(len(prompts_dataset)), desc="Target prompts"):
                input_ids = prompts_dataset[i]["input_ids"].unsqueeze(0).to(device)
                original_logits: Tensor = model(input_ids)
                write_ablation_rows(
                    writer, model, input_ids, original_logits, alive_components, tokenizer, device
                )

    print(f"Wrote {target_path}")

    # 7-8. Ablate on nontarget data
    assert config.nontarget_task_config is not None, "No nontarget_task_config in config"
    assert isinstance(config.nontarget_task_config, LMTaskConfig)
    assert config.nontarget_batch_size is not None
    assert config.nontarget_eval_batch_size is not None

    _, eval_loader = _create_lm_loaders(
        task_config=config.nontarget_task_config,
        tokenizer_name=config.tokenizer_name,
        train_batch_size=config.nontarget_batch_size,
        eval_batch_size=config.nontarget_eval_batch_size,
        seed=0,
        dist_state=None,
    )

    nontarget_path = output_dir / "nontarget_ablation_results.tsv"
    print(f"\nAblating on nontarget data ({args.n_batches} batches) → {nontarget_path}")
    with open(nontarget_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(eval_loader, total=args.n_batches, desc="Nontarget batches")
            ):
                if batch_idx >= args.n_batches:
                    break
                input_ids = batch["input_ids"].to(device)
                original_logits = model(input_ids)
                write_ablation_rows(
                    writer, model, input_ids, original_logits, alive_components, tokenizer, device
                )

    print(f"Wrote {nontarget_path}")
    print("Done.")


if __name__ == "__main__":
    main()
