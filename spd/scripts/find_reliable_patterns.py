"""Find sequences where the pretrained model reliably predicts specific target tokens.

Given a decomposed SPD model path, loads the original pretrained model and identifies
positions where the next token is one of the specified target tokens AND the model assigns
probability above the threshold.

Usage:
    uv run python -m spd.scripts.find_reliable_patterns <model_path> --n-batches 5 --thr 0.3
    uv run python -m spd.scripts.find_reliable_patterns <model_path> --non-target --tokens he she his her
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedTokenizer

from spd.configs import LMTaskConfig
from spd.data import DatasetConfig, create_data_loader
from spd.experiments.lm.prompts_dataset import create_prompts_data_loader
from spd.models.component_model import SPDRunInfo
from spd.pretrain.run_info import PretrainRunInfo
from spd.utils.general_utils import resolve_class

DEFAULT_TOKENS = ["he", "she", "his", "her"]


def escape_special(s: str) -> str:
    """Escape whitespace/special characters so the string is safe in a single TSV cell."""
    s = s.replace("\\", "\\\\")
    s = s.replace("\n", "\\n")
    s = s.replace("\t", "\\t")
    s = s.replace("\r", "\\r")
    return s


def build_target_token_ids(
    token_strings: list[str], tokenizer: PreTrainedTokenizer
) -> dict[int, str]:
    """Build a mapping from token ID to display string for each target token.

    Tries each token with and without a leading space to catch both variants.
    Only includes tokens that encode to a single token ID.
    """
    token_id_to_str: dict[int, str] = {}
    for tok_str in token_strings:
        for variant in [tok_str, f" {tok_str}"]:
            ids = tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                token_id_to_str[ids[0]] = variant
    return token_id_to_str


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find sequences where target tokens are reliably predicted"
    )
    parser.add_argument("model_path", type=str, help="Decomposed model path (wandb or local)")
    parser.add_argument(
        "--tokens",
        nargs="+",
        default=DEFAULT_TOKENS,
        help="Target tokens to search for (default: he she his her)",
    )
    parser.add_argument("--thr", type=float, default=0.5, help="Probability threshold")
    parser.add_argument(
        "--non-target",
        action="store_true",
        help="Use nontarget_task_config instead of task_config",
    )
    parser.add_argument("--n-batches", type=int, default=100, help="Number of batches to process")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--output", type=Path, default=Path("reliable_patterns.tsv"), help="Output TSV path"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto)")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Load SPD run config
    print(f"Loading SPD run info from {args.model_path}...")
    run_info = SPDRunInfo.from_path(args.model_path)
    config = run_info.config

    # Select task config
    if args.non_target:
        assert config.nontarget_task_config is not None, (
            "No nontarget_task_config found in this run's config"
        )
        task_config = config.nontarget_task_config
    else:
        task_config = config.task_config
    assert isinstance(task_config, LMTaskConfig), (
        f"Expected LMTaskConfig, got {type(task_config).__name__}"
    )

    # Load pretrained model (same pattern as ComponentModel.from_run_info)
    model_class = resolve_class(config.pretrained_model_class)
    if config.pretrained_model_name is not None:
        if config.pretrained_model_class.startswith("spd.pretrain.models."):
            pretrain_run_info = PretrainRunInfo.from_path(config.pretrained_model_name)
            if "model_type" not in pretrain_run_info.model_config_dict:
                pretrain_run_info.model_config_dict["model_type"] = (
                    config.pretrained_model_class.split(".")[-1]
                )
            model = model_class.from_run_info(pretrain_run_info)  # pyright: ignore[reportAttributeAccessIssue]
        else:
            model = model_class.from_pretrained(config.pretrained_model_name)  # pyright: ignore[reportAttributeAccessIssue]
    else:
        assert config.pretrained_model_path is not None
        model = model_class.from_pretrained(config.pretrained_model_path)  # pyright: ignore[reportAttributeAccessIssue]
    model.eval()
    model.to(device)

    # Load tokenizer and build target token set
    assert config.tokenizer_name is not None
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    decode = tokenizer.decode  # pyright: ignore[reportAttributeAccessIssue]

    target_ids = build_target_token_ids(args.tokens, tokenizer)
    assert len(target_ids) > 0, f"None of the tokens {args.tokens} encode to single token IDs"
    print(f"Target token IDs: { {tid: s for tid, s in target_ids.items()} }")

    target_id_tensor = torch.tensor(list(target_ids.keys()), device=device)

    # Create data loader
    print("Loading dataset...")
    if task_config.prompts_file is not None:
        loader, _ = create_prompts_data_loader(
            prompts_file=Path(task_config.prompts_file),
            tokenizer_name=config.tokenizer_name,
            max_seq_len=task_config.max_seq_len,
            batch_size=args.batch_size,
            seed=0,
        )
        column_name = "input_ids"
    else:
        assert task_config.dataset_name is not None
        data_config = DatasetConfig(
            name=task_config.dataset_name,
            hf_tokenizer_path=config.tokenizer_name,
            split=task_config.train_data_split,
            n_ctx=task_config.max_seq_len,
            is_tokenized=task_config.is_tokenized,
            streaming=task_config.streaming,
            column_name=task_config.column_name,
            shuffle_each_epoch=False,
        )
        loader, _ = create_data_loader(
            dataset_config=data_config,
            batch_size=args.batch_size,
            buffer_size=task_config.buffer_size,
            global_seed=0,
        )
        column_name = task_config.column_name

    # Process batches
    rows: list[tuple[str, str, str, str]] = []
    n_total_positions = 0

    print(f"Processing {args.n_batches} batches (batch_size={args.batch_size})...")
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.n_batches:
            break

        input_ids = batch[column_name].to(device)  # (B, seq_len)

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = model(input_ids)  # (B, seq_len, vocab)

        # Probability of actual next token at each position
        probs = F.softmax(logits[:, :-1, :].float(), dim=-1)  # (B, seq_len-1, vocab)
        next_tokens = input_ids[:, 1:]  # (B, seq_len-1)
        next_token_probs = probs.gather(2, next_tokens.unsqueeze(2)).squeeze(2)  # (B, seq_len-1)

        n_total_positions += next_token_probs.numel()

        # Filter: next token must be in target set AND prob > threshold
        is_target = (next_tokens.unsqueeze(-1) == target_id_tensor).any(-1)  # (B, seq_len-1)
        hits = is_target & (next_token_probs > args.thr)
        hit_batch, hit_pos = torch.where(hits)

        for b, pos in zip(hit_batch.tolist(), hit_pos.tolist(), strict=True):
            context_ids = input_ids[b, : pos + 1].tolist()
            next_tok_id = input_ids[b, pos + 1].item()
            prob = next_token_probs[b, pos].item()

            context_str = escape_special(decode(context_ids))
            short_tokens = [escape_special(decode([tid])) for tid in context_ids[-5:]]
            context_short_str = ";".join(short_tokens)
            next_tok_str = escape_special(decode([next_tok_id]))

            rows.append((context_str, context_short_str, next_tok_str, f"{prob:.4f}"))

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{args.n_batches}: {len(rows)} hits so far")

    # Write TSV
    print(f"Writing {len(rows)} rows to {args.output}")
    print(f"  ({len(rows)}/{n_total_positions} positions = {len(rows)/n_total_positions:.1%})")
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["context", "context_short", "next_token", "prob"])
        writer.writerows(rows)

    print("Done.")


if __name__ == "__main__":
    main()
