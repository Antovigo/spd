"""Analyze whether SPD components have absolute-sequence-position roles.

For each component, accumulates a histogram of firing positions across batches,
then computes summary statistics (entropy, early/late firing fractions).

Outputs:
  - .pt file with raw results
  - HTML page with position distribution visualizations
"""

import base64
import io
from itertools import islice

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from spd.adapters.spd import SPDAdapter
from spd.settings import SPD_OUT_DIR
from spd.utils.general_utils import bf16_autocast, extract_batch_data

RUN_ID = "s-55ea3f9b"
N_BATCHES = 500
BATCH_SIZE = 32
CI_THRESHOLD = 0.1
SEQ_LEN = 512


def main():
    adapter = SPDAdapter(RUN_ID)
    model = adapter.component_model.to("cuda").eval()
    dataloader = adapter.dataloader(batch_size=BATCH_SIZE)

    # Accumulate position histograms per layer per component
    # layer_name -> Tensor[n_components, seq_len]
    position_counts: dict[str, torch.Tensor] = {}

    with torch.no_grad(), bf16_autocast():
        for batch_item in islice(dataloader, N_BATCHES):
            batch = extract_batch_data(batch_item).to("cuda")
            actual_seq_len = batch.shape[1]
            out = model(batch, cache_type="input")
            ci_dict = model.calc_causal_importances(
                pre_weight_acts=out.cache,
                detach_inputs=True,
                sampling="continuous",
            ).lower_leaky

            for layer_name, ci in ci_dict.items():
                # ci: [batch, seq_len, n_components]
                firings = (ci > CI_THRESHOLD).float()

                if layer_name not in position_counts:
                    n_components = ci.shape[2]
                    position_counts[layer_name] = torch.zeros(
                        n_components, actual_seq_len, device="cpu"
                    )

                # Sum over batch dim -> [seq_len, n_components], then transpose
                pos_hist = firings.sum(dim=0).cpu()  # [seq_len, n_components]
                position_counts[layer_name] += pos_hist.T  # [n_components, seq_len]

    # Compute summary statistics
    results: dict[str, dict[str, torch.Tensor]] = {}
    all_components: list[tuple[str, int, float, float, float, float]] = []

    for layer_name, counts in position_counts.items():
        n_components, _ = counts.shape
        total_firings = counts.sum(dim=1)  # [n_components]

        # Normalized distribution
        probs = counts / (total_firings.unsqueeze(1) + 1e-10)

        # Entropy
        log_probs = torch.log2(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=1)  # [n_components]

        # Fraction in first/last 10 positions
        first_10_frac = counts[:, :10].sum(dim=1) / (total_firings + 1e-10)
        last_10_frac = counts[:, -10:].sum(dim=1) / (total_firings + 1e-10)

        results[layer_name] = {
            "counts": counts,
            "entropy": entropy,
            "first_10_frac": first_10_frac,
            "last_10_frac": last_10_frac,
            "total_firings": total_firings,
        }

        for comp_idx in range(n_components):
            if total_firings[comp_idx] > 0:
                all_components.append(
                    (
                        layer_name,
                        comp_idx,
                        entropy[comp_idx].item(),
                        first_10_frac[comp_idx].item(),
                        last_10_frac[comp_idx].item(),
                        total_firings[comp_idx].item(),
                    )
                )

    # Save .pt file
    out_dir = SPD_OUT_DIR / "www" / "autointerp" / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(results, out_dir / "position_distributions.pt")
    print(f"Saved results to {out_dir / 'position_distributions.pt'}")

    # Sort by entropy
    all_components.sort(key=lambda x: x[2])
    max_entropy = np.log2(SEQ_LEN)

    # Generate HTML
    html = generate_html(all_components, results, max_entropy)
    html_path = out_dir / "position_distributions.html"
    html_path.write_text(html)
    print(f"Saved HTML to {html_path}")


def make_histogram_b64(counts: torch.Tensor, title: str) -> str:
    """Create a small histogram plot and return as base64 PNG."""
    fig, ax = plt.subplots(figsize=(5, 2))
    values = counts.numpy()
    n_positions = len(values)

    # For concentrated distributions, bin into fewer bars so spikes are visible.
    # Otherwise a single-position spike is 1px wide out of 500px.
    nonzero_positions = int((values > 0).sum())
    if nonzero_positions < n_positions * 0.1:
        # Very concentrated: use scatter-style stems
        nonzero_idx = np.where(values > 0)[0]
        ax.vlines(nonzero_idx, 0, values[nonzero_idx], colors="#0173B2", linewidth=2)
        ax.scatter(nonzero_idx, values[nonzero_idx], color="#0173B2", s=10, zorder=3)
    else:
        ax.bar(np.arange(n_positions), values, width=1.0, color="#0173B2", edgecolor="none")

        # For near-uniform distributions, zoom the y-axis to show variation.
        val_min, val_max = values.min(), values.max()
        if val_min > 0 and val_max > 0 and val_min / val_max > 0.8:
            margin = (val_max - val_min) * 0.2
            ax.set_ylim(val_min - margin, val_max + margin)

    ax.set_xlabel("Position")
    ax.set_ylabel("Firings")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, n_positions)

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def generate_html(
    all_components: list[tuple[str, int, float, float, float, float]],
    results: dict[str, dict[str, torch.Tensor]],
    max_entropy: float,
) -> str:
    lowest_entropy = all_components[:20]
    highest_entropy = all_components[-20:][::-1]

    # Filter out position-0/1 dominated components and noise to find mid-sequence biases
    MIN_FIRINGS = 10  # need enough firings for a meaningful distribution
    mid_sequence_biased = [
        c
        for c in all_components
        if c[3] < 0.3  # less than 30% in first 10 positions
        and c[4] < 0.3  # less than 30% in last 10 positions
        and c[5] >= MIN_FIRINGS  # enough total firings
    ][:20]

    total_components = len(all_components)
    avg_entropy = np.mean([c[2] for c in all_components])

    def make_table(
        components: list[tuple[str, int, float, float, float, float]], label: str
    ) -> str:
        rows = []
        for layer, idx, ent, first10, last10, total in components:
            counts = results[layer]["counts"][idx]
            title = f"{layer}:{idx}"
            img_b64 = make_histogram_b64(counts, title)
            rows.append(f"""
            <tr>
                <td><code>{layer}:{idx}</code></td>
                <td>{ent:.2f}</td>
                <td>{first10:.3f}</td>
                <td>{last10:.3f}</td>
                <td>{int(total)}</td>
                <td><img src="data:image/png;base64,{img_b64}" style="max-width:400px;"></td>
            </tr>""")

        return f"""
        <h2>{label}</h2>
        <table>
            <thead>
                <tr>
                    <th>Component</th>
                    <th>Entropy (bits)</th>
                    <th>First 10 frac</th>
                    <th>Last 10 frac</th>
                    <th>Total firings</th>
                    <th>Position histogram</th>
                </tr>
            </thead>
            <tbody>{"".join(rows)}</tbody>
        </table>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Position Distribution Analysis - {RUN_ID}</title>
    <style>
        body {{ font-family: sans-serif; margin: 20px; max-width: 1400px; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; }}
        th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        tr:hover {{ background: #f9f9f9; }}
        code {{ background: #eee; padding: 2px 4px; border-radius: 3px; font-size: 12px; }}
        .summary {{ background: #f0f7ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>Position Distribution Analysis</h1>
    <div class="summary">
        <strong>Run:</strong> {RUN_ID} |
        <strong>Batches:</strong> {N_BATCHES} |
        <strong>Batch size:</strong> {BATCH_SIZE} |
        <strong>CI threshold:</strong> {CI_THRESHOLD}<br>
        <strong>Total components (with firings):</strong> {total_components} |
        <strong>Mean entropy:</strong> {avg_entropy:.2f} bits |
        <strong>Max possible entropy:</strong> {max_entropy:.2f} bits
    </div>

    {make_table(lowest_entropy, "Top 20 Most Position-Biased Components (Lowest Entropy)")}
    {make_table(mid_sequence_biased, "Top 20 Most Position-Biased (Excluding Start/End-of-Sequence)")}
    {make_table(highest_entropy, "Top 20 Most Uniform Components (Highest Entropy)")}
</body>
</html>"""
    return html


if __name__ == "__main__":
    main()
