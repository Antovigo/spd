"""Generate a markdown summary report from a set of WandB sweep runs.

Usage:
    python spd/scripts/sweep_summary/sweep_summary.py s-e8bde534 s-73d2385c ...
    python spd/scripts/sweep_summary/sweep_summary.py s-e8bde534 ... --output report.md
"""

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import wandb

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

CE_KL_KEYS = [
    "eval/ce_kl/kl_unmasked",
    "eval/ce_kl/kl_stoch_masked",
    "eval/ce_kl/kl_ci_masked",
    "eval/ce_kl/kl_rounded_masked",
    "eval/ce_kl/kl_random_masked",
    "eval/ce_kl/kl_zero_masked",
    "eval/ce_kl/ce_difference_unmasked",
    "eval/ce_kl/ce_difference_stoch_masked",
    "eval/ce_kl/ce_difference_ci_masked",
    "eval/ce_kl/ce_difference_rounded_masked",
    "eval/ce_kl/ce_difference_random_masked",
    "eval/ce_kl/ce_unrecovered_unmasked",
    "eval/ce_kl/ce_unrecovered_stoch_masked",
    "eval/ce_kl/ce_unrecovered_ci_masked",
    "eval/ce_kl/ce_unrecovered_rounded_masked",
    "eval/ce_kl/ce_unrecovered_random_masked",
]

EVAL_LOSS_KEYS = [
    "eval/loss/StochasticReconSubsetLoss",
    "eval/loss/PGDReconLoss",
    "eval/loss/StochasticHiddenActsReconLoss",
    "eval/loss/CIHiddenActsReconLoss",
    "eval/loss/FaithfulnessLoss",
    "eval/loss/ImportanceMinimalityLoss",
]

L0_LAYER_KEYS = [
    "eval/l0/0.0_layer_0",
    "eval/l0/0.0_layer_1",
    "eval/l0/0.0_layer_2",
    "eval/l0/0.0_layer_3",
    "eval/l0/0.0_total",
]

TRAIN_LOSS_KEYS = [
    "train/loss/total",
    "train/loss/FaithfulnessLoss",
    "train/loss/ImportanceMinimalityLoss",
    "train/loss/StochasticReconSubsetLoss",
    "train/loss/PersistentPGDReconLoss",
]

MODULES = [
    "h.0.attn.q_proj",
    "h.0.attn.k_proj",
    "h.0.attn.v_proj",
    "h.0.attn.o_proj",
    "h.0.mlp.c_fc",
    "h.0.mlp.down_proj",
    "h.1.attn.q_proj",
    "h.1.attn.k_proj",
    "h.1.attn.v_proj",
    "h.1.attn.o_proj",
    "h.1.mlp.c_fc",
    "h.1.mlp.down_proj",
    "h.2.attn.q_proj",
    "h.2.attn.k_proj",
    "h.2.attn.v_proj",
    "h.2.attn.o_proj",
    "h.2.mlp.c_fc",
    "h.2.mlp.down_proj",
    "h.3.attn.q_proj",
    "h.3.attn.k_proj",
    "h.3.attn.v_proj",
    "h.3.attn.o_proj",
    "h.3.mlp.c_fc",
    "h.3.mlp.down_proj",
]

LAYER_PATTERN = re.compile(r"h\.(\d+)\.")


def _layer_of(module: str) -> int:
    m = LAYER_PATTERN.search(module)
    assert m, f"Cannot extract layer from {module}"
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _fmt(v: float | np.floating) -> str:
    if v == 0:
        return "0"
    abs_v = abs(v)
    if abs_v < 1e-10:
        return f"{v:.14f}"
    if abs_v < 1e-6:
        return f"{v:.10f}"
    if abs_v < 0.0001:
        return f"{v:.8f}"
    if abs_v < 0.01:
        return f"{v:.6f}"
    if abs_v < 1:
        return f"{v:.4f}"
    if abs_v < 100:
        return f"{v:.2f}"
    return f"{v:.1f}"


def _short(key: str) -> str:
    """Extract a short display name for table column headers."""
    for prefix in [
        "eval/ce_kl/",
        "eval/loss/",
        "eval/l0/0.0_",
        "train/loss/",
        "train/l0/",
    ]:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _summary_name(key: str) -> str:
    """Extract a descriptive name for the plain-text summary list."""
    if key.startswith("eval/l0/0.0_"):
        return "L0 " + key[len("eval/l0/0.0_") :]
    if key.startswith("train/loss/"):
        return "train " + key[len("train/loss/") :]
    for prefix in ["eval/ce_kl/", "eval/loss/"]:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _raw_table(
    seeds: list[int],
    keys: list[str],
    data: dict[int, dict[str, float]],
) -> str:
    headers = ["seed"] + [_short(k) for k in keys]
    rows = []
    for s in seeds:
        row = [str(s)]
        for k in keys:
            v = data[s].get(k)
            row.append(_fmt(v) if v is not None else "—")
        rows.append(row)
    return _md_table(headers, rows)


def _summary_table(
    seeds: list[int],
    keys: list[str],
    data: dict[int, dict[str, float]],
) -> str:
    headers = ["stat"] + [_short(k) for k in keys]
    mean_row = ["mean"]
    std_row = ["std"]
    for k in keys:
        vals = [data[s][k] for s in seeds if data[s].get(k) is not None]
        if vals:
            mean_row.append(_fmt(np.mean(vals)))
            std_row.append(_fmt(np.std(vals)))
        else:
            mean_row.append("—")
            std_row.append("—")
    return _md_table(headers, [mean_row, std_row])


def _cross_layer_summary(
    seeds: list[int],
    module_keys: list[str],
    data: dict[int, dict[str, float]],
) -> str:
    """Mean across modules within each layer, then mean/std across seeds."""
    # Extract module name from each key for layer grouping
    key_to_module = {}
    for k in module_keys:
        for m in MODULES:
            if k.endswith(m):
                key_to_module[k] = m
                break

    layers = sorted(set(_layer_of(key_to_module[k]) for k in module_keys if k in key_to_module))
    layer_keys: dict[int, list[str]] = defaultdict(list)
    for k in module_keys:
        if k in key_to_module:
            layer_keys[_layer_of(key_to_module[k])].append(k)

    headers = ["stat"] + [f"layer {ly}" for ly in layers] + ["all"]
    per_seed_layer: dict[int, dict[int, float]] = {}
    per_seed_all: dict[int, float] = {}
    for s in seeds:
        per_seed_layer[s] = {}
        all_vals = []
        for ly in layers:
            vals = [data[s][k] for k in layer_keys[ly] if data[s].get(k) is not None]
            per_seed_layer[s][ly] = float(np.mean(vals)) if vals else float("nan")
            all_vals.extend(vals)
        per_seed_all[s] = float(np.mean(all_vals)) if all_vals else float("nan")

    mean_row = ["mean"]
    std_row = ["std"]
    for ly in layers:
        vals = [per_seed_layer[s][ly] for s in seeds]
        mean_row.append(_fmt(np.mean(vals)))
        std_row.append(_fmt(np.std(vals)))
    all_vals = [per_seed_all[s] for s in seeds]
    mean_row.append(_fmt(np.mean(all_vals)))
    std_row.append(_fmt(np.std(all_vals)))

    return _md_table(headers, [mean_row, std_row])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@dataclass
class TargetModelInfo:
    wandb_path: str
    model_class: str
    architecture: dict[str, object]
    tokenizer: str
    train_loss: float
    val_loss: float
    train_dataset: str
    train_steps: int
    val_loss_curve: list[tuple[int, float]]  # (step, val_loss) sorted by step


def _fetch_target_model_info(
    pretrained_model_name: str,
    pretrained_model_class: str,
    tokenizer_name: str,
    project: str,
) -> TargetModelInfo:
    api = wandb.Api()
    run = api.run(f"{project}/runs/{pretrained_model_name.split('/')[-1]}")
    history = list(run.scan_history(keys=["val_loss", "_step"], page_size=10000))
    val_loss_curve = sorted(
        {(int(h["_step"]), float(h["val_loss"])) for h in history if h.get("val_loss") is not None}
    )
    return TargetModelInfo(
        wandb_path=pretrained_model_name,
        model_class=pretrained_model_class,
        architecture=run.config.get("model", {}),
        tokenizer=tokenizer_name,
        train_loss=float(run.summary.get("train_loss", float("nan"))),
        val_loss=float(run.summary.get("val_loss", float("nan"))),
        train_dataset=run.config.get("train_dataset_config", {}).get("name", "unknown"),
        train_steps=int(run.summary.get("_step", 0)),
        val_loss_curve=val_loss_curve,
    )


def _compute_recovered_pct(target_info: TargetModelInfo, spd_ce: float) -> float | None:
    """Find what % through target training had the same val loss as spd_ce.

    Linearly interpolates between val_loss_curve points. Returns None if the SPD CE
    is worse than the target model's first logged val loss (i.e. no compute recovered).
    """
    curve = target_info.val_loss_curve
    assert len(curve) >= 2
    total_steps = target_info.train_steps

    # If SPD is better than final target, return 100%
    if spd_ce <= curve[-1][1]:
        return 100.0

    # If SPD is worse than the very first checkpoint, return None
    if spd_ce >= curve[0][1]:
        return None

    # Find the interval where val_loss crosses spd_ce (curve is decreasing)
    for i in range(len(curve) - 1):
        step_a, loss_a = curve[i]
        step_b, loss_b = curve[i + 1]
        if loss_a >= spd_ce >= loss_b:
            # Linear interpolation
            frac = (loss_a - spd_ce) / (loss_a - loss_b)
            step = step_a + frac * (step_b - step_a)
            return step / total_steps * 100.0

    return None


def fetch_runs(
    run_ids: list[str], project: str
) -> tuple[list[int], dict[int, dict[str, float]], TargetModelInfo]:
    api = wandb.Api()
    seeds: list[int] = []
    data: dict[int, dict[str, float]] = {}
    first_config: dict[str, object] | None = None
    for rid in run_ids:
        run = api.run(f"{project}/runs/{rid}")
        if first_config is None:
            first_config = run.config
        seed = run.config["seed"]
        seeds.append(seed)
        summary = {}
        for k, v in run.summary.items():
            if isinstance(v, (int, float)):
                summary[k] = float(v)
        data[seed] = summary
    seeds.sort()
    assert first_config is not None
    target_info = _fetch_target_model_info(
        pretrained_model_name=str(first_config["pretrained_model_name"]),
        pretrained_model_class=str(first_config["pretrained_model_class"]),
        tokenizer_name=str(first_config["tokenizer_name"]),
        project=project,
    )
    return seeds, data, target_info


def _per_module_keys(prefix: str) -> list[str]:
    return [f"{prefix}/{m}" for m in MODULES]


def _render_target_model_section(info: TargetModelInfo) -> str:
    arch = info.architecture
    lines = [
        "## Target Model",
        "",
        f"WandB: `{info.wandb_path}`",
        f"Class: `{info.model_class}`",
        f"Tokenizer: `{info.tokenizer}`",
        f"Training dataset: `{info.train_dataset}`",
        f"Training steps: {info.train_steps:,}",
        f"Train CE loss: {_fmt(info.train_loss)}",
        f"Val CE loss: {_fmt(info.val_loss)}",
        "",
        "Architecture:",
        f"Layers: {arch.get('n_layer')}",
        f"Hidden dim: {arch.get('n_embd')}",
        f"Heads: {arch.get('n_head')}",
        f"MLP intermediate: {arch.get('n_intermediate')}",
        f"Context length: {arch.get('n_ctx')}",
        f"Vocab size: {arch.get('vocab_size')}",
    ]
    return "\n\n".join(lines)


def generate_report(
    seeds: list[int], data: dict[int, dict[str, float]], target_info: TargetModelInfo
) -> str:
    sections: list[str] = []

    sections.append(f"# Sweep Summary Report\n\n**Seeds**: {seeds}\n")
    sections.append(_render_target_model_section(target_info))

    # 1. CE/KL
    sections.append("## Output Quality (CE/KL)\n")
    sections.append("### Raw values\n")
    sections.append(_raw_table(seeds, CE_KL_KEYS, data))
    sections.append("\n### Summary\n")
    sections.append(_summary_table(seeds, CE_KL_KEYS, data))

    # 2. Eval losses (aggregate)
    sections.append("\n## Eval Reconstruction Losses (aggregate)\n")
    sections.append("### Raw values\n")
    sections.append(_raw_table(seeds, EVAL_LOSS_KEYS, data))
    sections.append("\n### Summary\n")
    sections.append(_summary_table(seeds, EVAL_LOSS_KEYS, data))

    # 3. Hidden acts recon per module
    for loss_name, prefix in [
        ("StochasticHiddenActsReconLoss", "eval/loss/StochasticHiddenActsReconLoss"),
        ("CIHiddenActsReconLoss", "eval/loss/CIHiddenActsReconLoss"),
    ]:
        mod_keys = _per_module_keys(prefix)
        sections.append(f"\n## {loss_name} (per module)\n")
        sections.append("### Raw values\n")
        sections.append(_raw_table(seeds, mod_keys, data))
        sections.append("\n### Summary across seeds\n")
        sections.append(_summary_table(seeds, mod_keys, data))
        sections.append("\n### Summary across modules per layer\n")
        sections.append(_cross_layer_summary(seeds, mod_keys, data))

    # 4. Sparsity
    sections.append("\n## Sparsity (CI-L0)\n")
    sections.append("### Per-layer (raw)\n")
    sections.append(_raw_table(seeds, L0_LAYER_KEYS, data))
    sections.append("\n### Per-layer (summary)\n")
    sections.append(_summary_table(seeds, L0_LAYER_KEYS, data))

    l0_mod_keys = [f"eval/l0/0.0_{m}" for m in MODULES]
    sections.append("\n### Per-module (raw)\n")
    sections.append(_raw_table(seeds, l0_mod_keys, data))
    sections.append("\n### Per-module (summary across seeds)\n")
    sections.append(_summary_table(seeds, l0_mod_keys, data))
    sections.append("\n### Mean L0 per layer across seeds\n")
    sections.append(_cross_layer_summary(seeds, l0_mod_keys, data))

    # 5. Training losses
    sections.append("\n## Training Losses (final step)\n")
    sections.append("### Raw values\n")
    sections.append(_raw_table(seeds, TRAIN_LOSS_KEYS, data))
    sections.append("\n### Summary\n")
    sections.append(_summary_table(seeds, TRAIN_LOSS_KEYS, data))

    # 6. Training compute recovered
    MASKING_MODES = ["unmasked", "stoch_masked", "ci_masked", "rounded_masked"]
    sections.append("\n## Training Compute Recovered\n")
    sections.append(
        "Percentage through target model training where target val loss equals SPD model CE.\n"
    )
    recovered_headers = ["seed"] + MASKING_MODES
    recovered_rows: list[list[str]] = []
    per_mode_pcts: dict[str, list[float]] = {m: [] for m in MASKING_MODES}
    for s in seeds:
        row = [str(s)]
        for mode in MASKING_MODES:
            ce_diff = data[s].get(f"eval/ce_kl/ce_difference_{mode}")
            if ce_diff is not None:
                spd_ce = target_info.val_loss + ce_diff
                pct = _compute_recovered_pct(target_info, spd_ce)
                if pct is not None:
                    row.append(f"{pct:.1f}%")
                    per_mode_pcts[mode].append(pct)
                else:
                    row.append("< 0%")
            else:
                row.append("—")
        recovered_rows.append(row)
    sections.append(_md_table(recovered_headers, recovered_rows))

    summary_headers = ["stat"] + MASKING_MODES
    mean_row = ["mean"]
    std_row = ["std"]
    for mode in MASKING_MODES:
        vals = per_mode_pcts[mode]
        if vals:
            mean_row.append(f"{np.mean(vals):.1f}%")
            std_row.append(f"{np.std(vals):.1f}%")
        else:
            mean_row.append("—")
            std_row.append("—")
    sections.append("\n### Summary\n")
    sections.append(_md_table(summary_headers, [mean_row, std_row]))

    # 7. Plain-text summary list
    summary_groups: list[tuple[str, list[str]]] = [
        ("Output Quality (CE/KL)", CE_KL_KEYS),
        ("Eval Reconstruction Losses", EVAL_LOSS_KEYS),
        ("Sparsity (CI-L0)", L0_LAYER_KEYS),
        ("Training Losses", TRAIN_LOSS_KEYS),
    ]
    sections.append("\n## All Summary Statistics\n")
    for group_name, keys in summary_groups:
        lines = [f"**{group_name}**"]
        for k in keys:
            vals = [data[s][k] for s in seeds if data[s].get(k) is not None]
            if vals:
                lines.append(
                    f"{_summary_name(k)}: {_fmt(np.mean(vals))} (std: {_fmt(np.std(vals))})"
                )
        sections.append("\n\n" + "\n\n".join(lines))

    return "\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sweep summary report")
    parser.add_argument("run_ids", nargs="+", help="WandB run IDs")
    parser.add_argument("--project", default="goodfire/spd")
    parser.add_argument("--output", default=None, help="Output file (default: stdout)")
    args = parser.parse_args()

    seeds, data, target_info = fetch_runs(args.run_ids, args.project)
    report = generate_report(seeds, data, target_info)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
