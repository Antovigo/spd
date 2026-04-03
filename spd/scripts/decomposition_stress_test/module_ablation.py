"""Interactive module ablation tool for residual-stream modules.

Loads a decomposed model's target network and provides a curses-based interface to toggle
residual-stream modules (attention, MLP per layer) on/off. Shows top-5 predicted tokens
with probabilities for both the original and ablated model.

Usage:
    python -m spd.scripts.decomposition_stress_test.module_ablation <model_path> [--device cuda]
"""

import argparse
import curses
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo


def extract_output(raw_output: Any, attr: str | None) -> Tensor:
    """Extract tensor output from model, mirroring ComponentModel._extract_output."""
    if attr is None:
        out = raw_output
    elif attr.startswith("idx_"):
        idx_val = int(attr.split("_")[1])
        assert isinstance(raw_output, Sequence)
        out = raw_output[idx_val]
    else:
        out = getattr(raw_output, attr)
    assert isinstance(out, Tensor)
    return out


def discover_residual_modules(target_model: nn.Module, decomposed_paths: list[str]) -> list[str]:
    """Discover residual-stream modules (attn/mlp) from decomposed module paths.

    Decomposed paths look like "h.3.mlp.c_fc" — we take the parent ("h.3.mlp") and keep
    unique entries that are attn or mlp modules.
    """
    parents: set[str] = set()
    for path in decomposed_paths:
        parts = path.split(".")
        # Walk up until we find an attn or mlp parent
        for i in range(len(parts) - 1, 0, -1):
            if parts[i] in ("attn", "mlp"):
                parents.add(".".join(parts[: i + 1]))
                break

    # Verify each parent actually exists in the model
    verified = []
    for p in parents:
        try:
            target_model.get_submodule(p)
            verified.append(p)
        except AttributeError:
            pass

    # Sort: by layer number, then attn before mlp
    def sort_key(path: str) -> tuple[int, int]:
        parts = path.split(".")
        layer_idx = 0
        for part in parts:
            if part.isdigit():
                layer_idx = int(part)
                break
        module_type = 0 if "attn" in path else 1
        return (layer_idx, module_type)

    return sorted(verified, key=sort_key)


def _zero_hook(_module: nn.Module, _input: Any, output: Any) -> Tensor:
    assert isinstance(output, Tensor)
    return torch.zeros_like(output)


def run_inference(
    target_model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    disabled_modules: list[str],
    output_attr: str | None,
    device: torch.device,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Run original and ablated inference, return top-5 (token, prob) for each."""
    input_ids: Tensor = tokenizer.encode(prompt, return_tensors="pt").to(device)  # pyright: ignore[reportAttributeAccessIssue]

    with torch.no_grad():
        orig_logits = extract_output(target_model(input_ids), output_attr)

        hooks = []
        for path in disabled_modules:
            module = target_model.get_submodule(path)
            hooks.append(module.register_forward_hook(_zero_hook))
        ablated_logits = extract_output(target_model(input_ids), output_attr)
        for h in hooks:
            h.remove()

    def top5(logits: Tensor) -> list[tuple[str, float]]:
        probs = torch.softmax(logits[0, -1], dim=-1)
        top_probs, top_ids = probs.topk(5)
        return [
            (tokenizer.decode([int(tid.item())]), top_probs[i].item())
            for i, tid in enumerate(top_ids)
        ]

    return top5(orig_logits), top5(ablated_logits)


def curses_main(
    stdscr: curses.window,
    target_model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    modules: list[str],
    output_attr: str | None,
    device: torch.device,
) -> None:
    curses.curs_set(0)
    curses.use_default_colors()
    stdscr.nodelay(False)
    stdscr.keypad(True)

    prompt = "The cat sat on the"
    enabled = [True] * len(modules)
    cursor = 0  # Index into modules list
    focus = "modules"  # "prompt" or "modules"
    orig_top5: list[tuple[str, float]] = []
    ablated_top5: list[tuple[str, float]] = []
    status = "Press ENTER to run inference"

    while True:
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()

        row = 0
        # Prompt line
        prefix = "Prompt: "
        prompt_display = prompt
        if focus == "prompt":
            prompt_display += "_"
        stdscr.addnstr(row, 0, f"{prefix}{prompt_display}", max_x - 1)
        row += 2

        # Instructions
        header = "Modules  [SPACE=toggle  ENTER=run  TAB=switch focus  q=quit  a=all on  n=all off]"
        stdscr.addnstr(row, 0, header, max_x - 1)
        row += 1

        # Module list
        for i, mod in enumerate(modules):
            if row >= max_y - 1:
                break
            check = "x" if enabled[i] else " "
            marker = ">" if (focus == "modules" and i == cursor) else " "
            line = f" {marker} [{check}] {mod}"
            stdscr.addnstr(row, 0, line, max_x - 1)
            row += 1

        row += 1

        # Results
        if orig_top5 and row < max_y - 1:
            stdscr.addnstr(row, 0, "--- Results (last token) ---", max_x - 1)
            row += 1

            col_w = max(max_x // 2, 30)
            if row < max_y - 1:
                stdscr.addnstr(row, 0, "Original", max_x - 1)
                stdscr.addnstr(row, col_w, "Ablated", max_x - col_w - 1)
                row += 1

            for j in range(5):
                if row >= max_y - 1:
                    break
                if j < len(orig_top5):
                    tok, prob = orig_top5[j]
                    orig_str = f"  {j + 1}. {tok!r:>12s}  ({prob:.4f})"
                    stdscr.addnstr(row, 0, orig_str, max_x - 1)
                if j < len(ablated_top5):
                    tok, prob = ablated_top5[j]
                    abl_str = f"  {j + 1}. {tok!r:>12s}  ({prob:.4f})"
                    stdscr.addnstr(row, col_w, abl_str, max_x - col_w - 1)
                row += 1

        # Status bar
        if row < max_y:
            stdscr.addnstr(max_y - 1, 0, status, max_x - 1)

        stdscr.refresh()

        key = stdscr.getch()

        if key == ord("q"):
            break
        elif key == ord("\t"):
            focus = "prompt" if focus == "modules" else "modules"
        elif focus == "prompt":
            if key in (curses.KEY_BACKSPACE, 127, 8):
                prompt = prompt[:-1]
            elif 32 <= key <= 126:
                prompt += chr(key)
        elif focus == "modules":
            if key == curses.KEY_UP:
                cursor = max(0, cursor - 1)
            elif key == curses.KEY_DOWN:
                cursor = min(len(modules) - 1, cursor + 1)
            elif key == ord(" "):
                enabled[cursor] = not enabled[cursor]
            elif key == ord("a"):
                enabled = [True] * len(modules)
            elif key == ord("n"):
                enabled = [False] * len(modules)

        # ENTER from either focus runs inference
        if key == ord("\n"):
            if not prompt.strip():
                status = "Enter a non-empty prompt first"
                continue
            status = "Running inference..."
            stdscr.addnstr(max_y - 1, 0, status + " " * (max_x - len(status) - 1), max_x - 1)
            stdscr.refresh()

            disabled = [m for m, en in zip(modules, enabled, strict=True) if not en]
            orig_top5, ablated_top5 = run_inference(
                target_model, tokenizer, prompt, disabled, output_attr, device
            )
            n_disabled = len(disabled)
            status = f"Done — {n_disabled} module(s) ablated"


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive module ablation tool")
    parser.add_argument("model_path", help="SPD model path (wandb or local)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_path = (
        str(Path(args.model_path).expanduser()) if ":" not in args.model_path else args.model_path
    )
    run_info = SPDRunInfo.from_path(model_path)
    config = run_info.config
    assert isinstance(config.task_config, LMTaskConfig), "Only LM experiments are supported"

    device = torch.device(args.device)
    comp_model = ComponentModel.from_run_info(run_info).to(device)
    comp_model.eval()

    target_model = comp_model.target_model
    output_attr = comp_model.pretrained_model_output_attr

    modules = discover_residual_modules(target_model, comp_model.target_module_paths)
    assert modules, "No residual-stream modules found"

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    assert isinstance(tokenizer, PreTrainedTokenizerBase)

    print(f"Loaded model with {len(modules)} residual-stream modules: {modules}")
    print("Launching interactive interface...")

    curses.wrapper(
        partial(
            curses_main,
            target_model=target_model,
            tokenizer=tokenizer,
            modules=modules,
            output_attr=output_attr,
            device=device,
        )
    )


if __name__ == "__main__":
    main()
