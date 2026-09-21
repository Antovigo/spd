"""Figures for `notes/ci_filter/report.md`, from the pulled filter outputs.

    python notes/ci_filter/make_figures.py [--root <run_dir>/analysis]

Reads each filter's `eval/pool_evals.jsonl` and `eval/pgd_recon.json` (the retired ones from
`ci_filter/step_40000/Trash/`) and the attribution under `ablations/step_40000/<filter-id>/`,
and writes `notes/ci_filter/figures/*.png`.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_ROOT = Path("~/out/pod-backup/p-ba5a0c05/analysis").expanduser()
FILTERS = "ci_filter/step_40000"
ABLATIONS = "ablations/step_40000"
FIGURES = Path(__file__).parent / "figures"

RUNS = {  # path under the filters folder -> (label, colour); retired runs live in Trash/
    "Trash/addsub-05-filter-last-pos": ("unconstrained", "tab:red"),
    "Trash/addsub-05-filter-last-pos-alive": ("prune_dead", "tab:orange"),
    "addsub-05-filter-last-pos-ceiling": ("ci_ceiling", "tab:blue"),
    "addsub-05-filter-integers": ("integer KL (ceiling+prune)", "tab:green"),
    "addsub-05-filter-answer-ce": ("answer CE (ceiling+prune)", "tab:purple"),
}
LLAMA_ACCURACY = 0.6671500205993652  # attribution summaries, clean model on the 20k pool


def evals(root: Path, run: str) -> list[dict[str, Any]]:
    path = root / FILTERS / run / "eval" / "pool_evals.jsonl"
    rows: list[dict[str, Any]] = [json.loads(line) for line in path.read_text().splitlines()]
    seen: dict[int, dict[str, Any]] = {}
    for row in rows:  # a pruning run logs step 0 twice: keep the post-removal one
        seen[row["step"]] = row
    return [seen[step] for step in sorted(seen)]


def line(
    ax: Any,
    rows: list[dict[str, Any]],
    key: Callable[[dict[str, Any]], float],
    label: str,
    color: str,
) -> None:
    steps = [r["step"] for r in rows]
    ax.plot(steps, [key(r) for r in rows], marker="o", ms=3, label=label, color=color)


def sparsity_figure(root: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for run, (label, color) in RUNS.items():
        rows = evals(root, run)
        line(axes[0], rows, lambda r: r["n_alive"], label, color)
        line(axes[1], rows, lambda r: r["l0_per_token_last"], label, color)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("alive components (CI > 0.01 anywhere)")
    axes[1].set_ylabel("components per token, last position")
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0].axhline(12553, ls="--", c="gray", lw=1)
    axes[0].text(2500, 13200, "decomposition's alive set", color="gray", fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle("Unconstrained filtering densifies; both constraints stop it")
    fig.tight_layout()
    fig.savefig(FIGURES / "sparsity.png", dpi=150)


def faithfulness_figure(root: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for run, (label, color) in RUNS.items():
        rows = evals(root, run)
        line(axes[0], rows, lambda r: r["rounded"]["kl"], label, color)
        if rows[0]["rounded"].get("accuracy") is not None:
            line(axes[1], rows, lambda r: r["rounded"]["accuracy"], label, color)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("KL to Llama (rounded CI masks)")
    axes[1].set_ylabel("accuracy on the true result")
    axes[1].axhline(LLAMA_ACCURACY, ls="--", c="k", lw=1)
    axes[1].text(200, LLAMA_ACCURACY + 0.015, "Llama-3.1-8B itself", fontsize=8)
    # The integer-KL run predates the accuracy column in the evaluations; its attribution
    # pass measured the same quantity on the same pool.
    integer: dict[str, float] = json.loads(
        (root / ABLATIONS / "addsub-05-filter-integers" / "summary.json").read_text()
    )
    axes[1].axhline(integer["accuracy"], ls=":", c="tab:green", lw=1.5)
    axes[1].text(
        200, integer["accuracy"] - 0.04, "integer-KL filter (final)", fontsize=8, color="tab:green"
    )
    axes[1].set_ylim(0.5, 1.0)
    for ax in axes:
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("The CE filter trades faithfulness for arithmetic accuracy")
    fig.tight_layout()
    fig.savefig(FIGURES / "faithfulness.png", dpi=150)


def attribution_figure(root: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for run, (label, color) in RUNS.items():
        path = root / ABLATIONS / Path(run).name
        if not path.exists():
            continue
        verified: list[dict[str, Any]] = json.loads((path / "verified.json").read_text())
        axes[0].scatter(
            [r["first_order"] for r in verified],
            [r["ablated_delta_score"] for r in verified],
            s=12,
            alpha=0.6,
            label=label,
            color=color,
        )
        with np.load(path / "components.npz") as saved:
            effects = np.concatenate(
                [
                    saved[k]
                    for k in saved.files
                    if k.endswith("|all") and not k.startswith("active:")
                ]
            )
        effects = np.sort(effects[effects != 0.0])
        axes[1].plot(np.arange(effects.size), effects, label=label, color=color)
    limit = 1.1
    axes[0].plot([-limit, limit], [-limit, limit], ls="--", c="gray", lw=1)
    axes[0].set_xlabel("first-order screen")
    axes[0].set_ylabel("measured ablation effect")
    axes[0].set_title("The screen predicts the causal effect")
    axes[1].set_xlabel("components, sorted")
    axes[1].set_ylabel("mean effect of ablating (log-prob)")
    axes[1].set_yscale("symlog", linthresh=1e-3)
    axes[1].set_title("Few components matter; both signs occur")
    for ax in axes:
        ax.axhline(0, c="k", lw=0.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Helpers (negative) and interferers (positive) on the true answer")
    fig.tight_layout()
    fig.savefig(FIGURES / "attribution.png", dpi=150)


def pgd_figure(root: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    names: list[str] = []
    initials: list[float] = []
    finals: list[float] = []
    colors: list[str] = []
    for run, (label, color) in RUNS.items():
        pgd: dict[str, float] = json.loads(
            (root / FILTERS / run / "eval" / "pgd_recon.json").read_text()
        )
        if "answer-ce" in run:  # a cross-entropy, not a KL: not on this axis
            continue
        names.append(label)
        initials.append(pgd["initial"])
        finals.append(pgd["final"])
        colors.append(color)
    x = np.arange(len(names))
    ax.bar(x - 0.2, initials, 0.4, label="start", color="lightgray", edgecolor="k")
    ax.bar(x + 0.2, finals, 0.4, label="after filtering", color=colors, edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("adversarial (PGD) recon KL, last position")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.suptitle("Removing the dead components removes the adversary's lever")
    fig.tight_layout()
    fig.savefig(FIGURES / "pgd.png", dpi=150)


KINDS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
CHAIN = (  # (label, source, colour): the decomposition, then each objective in the chain
    ("decomposition", None, "#2a78d6"),
    ("last-pos KL (ceiling)", "addsub-05-filter-last-pos-ceiling", "#eb6834"),
    ("integer KL", "addsub-05-filter-integers", "#1baf7a"),
    ("answer CE", "addsub-05-filter-answer-ce", "#eda100"),
)
SURFACE = "#fcfcfb"


def _alive_by_site(root: Path, source: str | None) -> dict[str, np.ndarray]:
    """`{site: (C,) bool}`: the decomposition's own alive set, or a filter's at its end."""
    if source is None:
        with np.load(root / ABLATIONS / "decomposition_alive.npz") as saved:
            return {site: saved[site].astype(bool) for site in saved.files}
    with np.load(root / FILTERS / source / "alive" / "max_ci.npz") as saved:
        return {site: saved[site] > 0.01 for site in saved.files}


def pruning_figure(root: Path) -> None:
    """Alive components per layer, one panel per matrix kind, one bar per stage of the chain —
    where each narrower objective removes components. Writes the counts beside the figure."""
    counts = {label: _alive_by_site(root, source) for label, source, _ in CHAIN}
    layers = sorted({int(site.split(".")[1]) for site in counts["decomposition"]})
    width = 0.8 / len(CHAIN)
    fig, axes = plt.subplots(len(KINDS), 1, figsize=(15, 2.3 * len(KINDS)), sharex=True)
    rows = ["\t".join(["kind", "layer", *[label for label, _, _ in CHAIN]])]
    for ax, kind in zip(axes, KINDS, strict=True):
        per_stage = {
            label: [int(alive[f"layers.{layer}.{kind}"].sum()) for layer in layers]
            for label, alive in counts.items()
        }
        for offset, (label, _, colour) in enumerate(CHAIN):
            ax.bar(
                np.array(layers) + (offset - (len(CHAIN) - 1) / 2) * width,
                per_stage[label],
                width,
                color=colour,
                edgecolor=SURFACE,
                linewidth=0.6,
                label=label,
            )
        for index, layer in enumerate(layers):
            rows.append(
                "\t".join(
                    [kind, str(layer), *[str(per_stage[stage][index]) for stage, _, _ in CHAIN]]
                )
            )
        totals = "  ·  ".join(f"{stage} {sum(per_stage[stage]):,}" for stage, _, _ in CHAIN)
        ax.set_title(f"{kind}   ({totals})", fontsize=9, loc="left", color="#0b0b0b")
        ax.set_ylabel("alive", fontsize=8, color="#52514e")
        ax.grid(axis="y", color="#e6e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=8, colors="#52514e")
    axes[-1].set_xticks(layers)
    axes[-1].set_xlabel("layer", fontsize=9, color="#52514e")
    axes[0].legend(
        ncol=len(CHAIN), fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.25), frameon=False
    )
    fig.suptitle(
        "Alive components per layer: the decomposition, then each filtering objective",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "pruning_by_layer.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    table = root / FILTERS / "pruning_by_layer.tsv"
    table.write_text("\n".join(rows) + "\n")
    print(f"-> {FIGURES / 'pruning_by_layer.png'} and {table}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()
    FIGURES.mkdir(exist_ok=True)
    sparsity_figure(args.root)
    faithfulness_figure(args.root)
    attribution_figure(args.root)
    pgd_figure(args.root)
    pruning_figure(args.root)
    print(f"-> {FIGURES}")


if __name__ == "__main__":
    main()
