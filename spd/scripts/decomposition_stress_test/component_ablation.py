"""Interactive component ablation tool for decomposed SPD models.

Loads a decomposed model and provides a curses-based interface to toggle three parts
per matrix: active components (CI > threshold), inactive components (CI <= threshold),
and the delta component (target weight minus sum of component weights). Shows top-5
predicted tokens with probabilities for both original and ablated models.

Usage:
    python -m spd.scripts.decomposition_stress_test.component_ablation <model_path> [--ci-thr 0.01]
"""

import argparse
import curses
from functools import partial
from pathlib import Path

import torch
from torch import Tensor
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from spd.configs import LMTaskConfig
from spd.models.component_model import ComponentModel, SPDRunInfo
from spd.models.components import WeightDeltaAndMask, make_mask_infos
from spd.utils.general_utils import calc_kl_divergence_lm


def top5(
    logits: Tensor, tokenizer: PreTrainedTokenizerBase
) -> list[tuple[str, float]]:
    probs = torch.softmax(logits[0, -1], dim=-1)
    top_probs, top_ids = probs.topk(5)
    return [
        (tokenizer.decode([tid.item()]), top_probs[i].item())
        for i, tid in enumerate(top_ids)
    ]


def run_inference(
    model: ComponentModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    toggles: dict[str, tuple[bool, bool, bool]],
    ci_thr: float,
    device: torch.device,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]], float]:
    """Run original and ablated inference, return top-5 (token, prob) for each + KL divergence."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    assert isinstance(input_ids, Tensor)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        # Original output (no masks)
        original_logits = model(input_ids)
        assert isinstance(original_logits, Tensor)

        # Cached forward for CI computation
        cached = model(input_ids, cache_type="input")
        ci = model.calc_causal_importances(
            cached.cache, sampling="continuous"
        ).lower_leaky

        weight_deltas = model.calc_weight_deltas()

        component_masks: dict[str, Tensor] = {}
        wdam: dict[str, WeightDeltaAndMask] = {}

        for module_name in model.target_module_paths:
            ci_vals = ci[module_name]
            is_active = (ci_vals > ci_thr).float()
            is_inactive = 1.0 - is_active

            active_on, inactive_on, delta_on = toggles[module_name]
            component_masks[module_name] = (
                float(active_on) * is_active + float(inactive_on) * is_inactive
            )

            delta_mask_val = 1.0 if delta_on else 0.0
            delta_mask = torch.full(
                ci_vals.shape[:-1], delta_mask_val, device=device
            )
            wdam[module_name] = (weight_deltas[module_name], delta_mask)

        ablated_logits = model(
            input_ids,
            mask_infos=make_mask_infos(
                component_masks, weight_deltas_and_masks=wdam
            ),
        )
        assert isinstance(ablated_logits, Tensor)

        kl = calc_kl_divergence_lm(pred=ablated_logits, target=original_logits).item()

    return top5(original_logits, tokenizer), top5(ablated_logits, tokenizer), kl


TOGGLE_LABELS = ("act", "inact", "delta")


def curses_main(
    stdscr: "curses.window",
    model: ComponentModel,
    tokenizer: PreTrainedTokenizerBase,
    modules: list[str],
    ci_thr: float,
    device: torch.device,
) -> None:
    curses.curs_set(0)
    curses.use_default_colors()
    stdscr.nodelay(False)
    stdscr.keypad(True)

    prompt = "The cat sat on the"
    # Each module gets three toggles: (active, inactive, delta), all start True
    toggles: dict[str, list[bool]] = {m: [True, True, True] for m in modules}
    cursor_row = 0
    cursor_col = 0  # 0=active, 1=inactive, 2=delta
    focus = "modules"  # "prompt" or "modules"
    scroll_offset = 0
    orig_top5: list[tuple[str, float]] = []
    ablated_top5: list[tuple[str, float]] = []
    kl_div: float | None = None
    status = "Press ENTER to run inference"

    while True:
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()

        row = 0

        # Prompt line
        prompt_display = prompt + ("_" if focus == "prompt" else "")
        stdscr.addnstr(row, 0, f"Prompt: {prompt_display}", max_x - 1)
        row += 2

        # Column headers
        header = (
            "     act inact delta  Module"
            "  [SPACE=toggle  ENTER=run  TAB=focus  q=quit  a=all on  n=all off]"
        )
        stdscr.addnstr(row, 0, header, max_x - 1)
        row += 1

        # How many module rows fit before the results section (reserve ~10 lines)
        max_module_rows = max_y - row - 12
        if max_module_rows < 1:
            max_module_rows = 1

        # Adjust scroll offset so cursor is visible
        if cursor_row < scroll_offset:
            scroll_offset = cursor_row
        elif cursor_row >= scroll_offset + max_module_rows:
            scroll_offset = cursor_row - max_module_rows + 1

        # Module list
        for i in range(scroll_offset, min(len(modules), scroll_offset + max_module_rows)):
            if row >= max_y - 1:
                break
            mod = modules[i]
            t = toggles[mod]
            marker = ">" if (focus == "modules" and i == cursor_row) else " "

            # Build line with toggle indicators
            parts: list[str] = [f" {marker} "]
            for col_idx in range(3):
                check = "x" if t[col_idx] else " "
                token = f"[{check}]"
                if focus == "modules" and i == cursor_row and col_idx == cursor_col:
                    # Will draw this token with A_REVERSE below
                    parts.append(f"  {token}")
                else:
                    parts.append(f"  {token}")
            parts.append(f"  {mod}")
            line = "".join(parts)
            stdscr.addnstr(row, 0, line, max_x - 1)

            # Highlight focused toggle with reverse video
            if focus == "modules" and i == cursor_row:
                # Calculate column position of the focused toggle
                # Layout: " > " (3) + "  [x]" (5) per toggle
                col_start = 3 + cursor_col * 5 + 2  # +2 for the leading spaces
                check = "x" if t[cursor_col] else " "
                token = f"[{check}]"
                stdscr.addnstr(row, col_start, token, max_x - col_start - 1, curses.A_REVERSE)

            row += 1

        if len(modules) > max_module_rows and row < max_y - 1:
                stdscr.addnstr(
                    row, 0,
                    f"  ... ({len(modules)} matrices, showing {scroll_offset + 1}-"
                    f"{min(len(modules), scroll_offset + max_module_rows)})",
                    max_x - 1,
                )
                row += 1

        row += 1

        # Results
        if orig_top5 and row < max_y - 1:
            kl_str = f"  KL divergence: {kl_div:.6f}" if kl_div is not None else ""
            stdscr.addnstr(row, 0, f"--- Results (last token) ---{kl_str}", max_x - 1)
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
                    stdscr.addnstr(
                        row, 0,
                        f"  {j + 1}. {tok!r:>12s}  ({prob:.4f})",
                        max_x - 1,
                    )
                if j < len(ablated_top5):
                    tok, prob = ablated_top5[j]
                    stdscr.addnstr(
                        row, col_w,
                        f"  {j + 1}. {tok!r:>12s}  ({prob:.4f})",
                        max_x - col_w - 1,
                    )
                row += 1

        # Status bar
        if max_y > 0:
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
                cursor_row = max(0, cursor_row - 1)
            elif key == curses.KEY_DOWN:
                cursor_row = min(len(modules) - 1, cursor_row + 1)
            elif key == curses.KEY_LEFT:
                cursor_col = max(0, cursor_col - 1)
            elif key == curses.KEY_RIGHT:
                cursor_col = min(2, cursor_col + 1)
            elif key == ord(" "):
                mod = modules[cursor_row]
                toggles[mod][cursor_col] = not toggles[mod][cursor_col]
            elif key == ord("a"):
                for m in modules:
                    toggles[m] = [True, True, True]
            elif key == ord("n"):
                for m in modules:
                    toggles[m] = [False, False, False]

        # ENTER from either focus runs inference
        if key == ord("\n"):
            if not prompt.strip():
                status = "Enter a non-empty prompt first"
                continue
            status = "Running inference..."
            stdscr.addnstr(
                max_y - 1, 0,
                status + " " * (max_x - len(status) - 1),
                max_x - 1,
            )
            stdscr.refresh()

            toggle_tuples = {
                m: (t[0], t[1], t[2]) for m, t in toggles.items()
            }
            orig_top5, ablated_top5, kl_div = run_inference(
                model, tokenizer, prompt, toggle_tuples, ci_thr, device
            )
            n_off = sum(
                sum(1 for v in t if not v) for t in toggles.values()
            )
            status = f"Done -- {n_off} toggle(s) disabled"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive component ablation tool for decomposed SPD models"
    )
    parser.add_argument("model_path", help="SPD model path (wandb or local)")
    parser.add_argument(
        "--ci-thr", type=float, default=0.01,
        help="CI threshold for active components",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    model_path = (
        str(Path(args.model_path).expanduser())
        if ":" not in args.model_path
        else args.model_path
    )
    run_info = SPDRunInfo.from_path(model_path)
    config = run_info.config
    assert isinstance(config.task_config, LMTaskConfig), "Only LM experiments are supported"

    device = torch.device(args.device)
    model = ComponentModel.from_run_info(run_info).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    modules = model.target_module_paths
    print(f"Loaded model with {len(modules)} decomposed matrices")
    print("Launching interactive interface...")

    curses.wrapper(
        partial(
            curses_main,
            model=model,
            tokenizer=tokenizer,
            modules=modules,
            ci_thr=args.ci_thr,
            device=device,
        )
    )


if __name__ == "__main__":
    main()
