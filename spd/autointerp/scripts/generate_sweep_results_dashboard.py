# pyright: reportGeneralTypeIssues=false, reportMissingTypeArgument=false, reportUnknownParameterType=false, reportArgumentType=false
"""Generate a static results dashboard for an autointerp sweep."""

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any

import matplotlib
import yaml
from matplotlib import pyplot as plt

from spd.autointerp.db import DONE_MARKER, InterpDB
from spd.autointerp.schemas import AUTOINTERP_DATA_DIR
from spd.settings import SPD_OUT_DIR

matplotlib.use("Agg")

SWEEP_ROOT = Path("scratch/autointerp_sweeps/jose_threshold_matrix_v2_200")
MANIFEST_PATH = SWEEP_ROOT / "manifest.json"
OUT_HTML = SPD_OUT_DIR / "www" / "autointerp" / "jose_threshold_matrix_v2_200_results.html"
OUT_JSON = SPD_OUT_DIR / "www" / "autointerp" / "jose_threshold_matrix_v2_200_results.json"
ASSETS_DIR = SPD_OUT_DIR / "www" / "autointerp" / "jose_threshold_matrix_v2_200_assets"


@dataclass(frozen=True)
class CandidateRun:
    subrun_id: str
    subrun_dir: Path
    harvest_subrun_id: str
    config: dict[str, Any]
    interpretation_count: int
    detection_scores: dict[str, float]
    fuzzing_scores: dict[str, float]
    interpret_done: bool


def _load_manifest() -> list[dict[str, Any]]:
    return json.loads(MANIFEST_PATH.read_text())


def _scan_candidates(decomposition_id: str) -> list[CandidateRun]:
    base = AUTOINTERP_DATA_DIR / decomposition_id
    if not base.exists():
        return []

    candidates: list[CandidateRun] = []
    for subrun_dir in sorted(base.iterdir()):
        if not subrun_dir.is_dir() or not subrun_dir.name.startswith("a-"):
            continue
        config_path = subrun_dir / "config.yaml"
        db_path = subrun_dir / "interp.db"
        if not config_path.exists() or not db_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text())
        db = InterpDB(db_path, readonly=True)
        try:
            try:
                harvest_subrun_id = db.get_config_value("harvest_subrun_id")
            except Exception:
                continue
            if not isinstance(harvest_subrun_id, str):
                continue
            try:
                interpretation_count = db.get_interpretation_count()
                detection_scores = db.get_scores("detection")
                fuzzing_scores = db.get_scores("fuzzing")
            except Exception:
                continue
            candidates.append(
                CandidateRun(
                    subrun_id=subrun_dir.name,
                    subrun_dir=subrun_dir,
                    harvest_subrun_id=harvest_subrun_id,
                    config=config,
                    interpretation_count=interpretation_count,
                    detection_scores=detection_scores,
                    fuzzing_scores=fuzzing_scores,
                    interpret_done=(subrun_dir / DONE_MARKER).exists(),
                )
            )
        finally:
            db.close()
    return candidates


def _mean_or_none(scores: dict[str, float]) -> float | None:
    return None if not scores else fmean(scores.values())


def _match_candidate(row: dict[str, Any], candidates: list[CandidateRun]) -> CandidateRun | None:
    expected_config = yaml.safe_load(Path(row["config_path"]).read_text())["config"]
    matches = [
        candidate
        for candidate in candidates
        if candidate.harvest_subrun_id == row["harvest_subrun_id"]
        and candidate.config == expected_config
    ]
    if not matches:
        return None
    return max(matches, key=lambda c: c.subrun_id)


def _status_label(candidate: CandidateRun | None) -> str:
    if candidate is None:
        return "waiting"
    if candidate.interpretation_count == 0:
        return "queued"
    if candidate.detection_scores and candidate.fuzzing_scores:
        return "scoring"
    if candidate.interpret_done:
        return "interpreted"
    return "running"


def _coverage_ratio(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total


def _fmt_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _card(row: dict[str, Any], candidate: CandidateRun | None, subset_size: int) -> str:
    import html

    detection_scores = {} if candidate is None else candidate.detection_scores
    fuzzing_scores = {} if candidate is None else candidate.fuzzing_scores
    detection_mean = _mean_or_none(detection_scores)
    fuzzing_mean = _mean_or_none(fuzzing_scores)
    interpretation_count = 0 if candidate is None else candidate.interpretation_count
    detection_count = len(detection_scores)
    fuzzing_count = len(fuzzing_scores)
    status = _status_label(candidate)
    return f"""
    <article class="card">
      <div class="card-head">
        <div>
          <p class="eyebrow">{html.escape(row["threshold_slug"])}</p>
          <h3>{html.escape(row["variant_slug"])}</h3>
        </div>
        <span class="status status-{status}">{status}</span>
      </div>
      <p class="meta">{html.escape(row["description"])}</p>
      <dl class="metrics">
        <div><dt>Subrun</dt><dd>{html.escape(candidate.subrun_id if candidate else "—")}</dd></div>
        <div><dt>Interpret</dt><dd>{interpretation_count}/{subset_size}</dd></div>
        <div><dt>Detection</dt><dd>{detection_count}/{subset_size} · {_fmt_score(detection_mean)}</dd></div>
        <div><dt>Fuzzing</dt><dd>{fuzzing_count}/{subset_size} · {_fmt_score(fuzzing_mean)}</dd></div>
      </dl>
      <div class="bars">
        <div class="bar-row"><span>interp</span><div class="bar"><div style="width:{_fmt_pct(_coverage_ratio(interpretation_count, subset_size))}"></div></div></div>
        <div class="bar-row"><span>det</span><div class="bar"><div style="width:{_fmt_pct(_coverage_ratio(detection_count, subset_size))}"></div></div></div>
        <div class="bar-row"><span>fuzz</span><div class="bar"><div style="width:{_fmt_pct(_coverage_ratio(fuzzing_count, subset_size))}"></div></div></div>
      </div>
      <p class="paths"><code>{html.escape(row["harvest_subrun_id"])}</code></p>
    </article>
    """


def _write_fuzzing_violin(
    threshold_slug: str,
    items: list[tuple[dict[str, Any], CandidateRun | None]],
) -> str | None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    positions: list[int] = []
    series: list[list[float]] = []
    labels = [row["variant_slug"] for row, _ in items]
    for idx, (_, candidate) in enumerate(items, start=1):
        if candidate is None or not candidate.fuzzing_scores:
            continue
        positions.append(idx)
        series.append(list(candidate.fuzzing_scores.values()))

    if not series:
        return None

    fig, ax = plt.subplots(figsize=(12.5, 4.6), constrained_layout=True)
    violin = ax.violinplot(
        series,
        positions=positions,
        widths=0.84,
        showmeans=True,
        showmedians=False,
        showextrema=False,
    )
    for body in list(violin["bodies"]):
        body.set_facecolor("#c66a2b")
        body.set_edgecolor("#8e3500")
        body.set_alpha(0.7)
        body.set_linewidth(1.0)
    violin["cmeans"].set_color("#17130f")
    violin["cmeans"].set_linewidth(1.2)

    ax.set_title(f"{threshold_slug} fuzzing score distribution", fontsize=14, pad=10)
    ax.set_ylabel("fuzzing score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.4, len(items) + 0.6)
    ax.set_xticks(range(1, len(items) + 1))
    ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
    ax.grid(axis="y", color="#d8d0c5", linewidth=0.8, alpha=0.9)
    ax.set_facecolor("#fffdf8")
    fig.patch.set_facecolor("#fffaf1")

    out_path = ASSETS_DIR / f"{threshold_slug}_fuzzing_violin.svg"
    fig.savefig(out_path, format="svg", dpi=160)
    plt.close(fig)
    return out_path.name


def main(
    manifest_path: str | None = None,
    out_html: str | None = None,
    out_json: str | None = None,
) -> None:
    manifest = (
        json.loads(Path(manifest_path).read_text())
        if manifest_path is not None
        else _load_manifest()
    )
    assert manifest, "manifest is empty"
    decomposition_id = "s-55ea3f9b"
    subset_path = Path(manifest[0]["subset_path"])
    subset_size = sum(1 for _ in subset_path.open())
    candidates = _scan_candidates(decomposition_id)

    rows_with_results = []
    grouped: dict[str, list[tuple[dict[str, Any], CandidateRun | None]]] = defaultdict(list)
    for row in manifest:
        candidate = _match_candidate(row, candidates)
        grouped[row["threshold_slug"]].append((row, candidate))
        rows_with_results.append(
            {
                **row,
                "subrun_id": None if candidate is None else candidate.subrun_id,
                "interpretation_count": 0 if candidate is None else candidate.interpretation_count,
                "detection_count": 0 if candidate is None else len(candidate.detection_scores),
                "detection_mean": None
                if candidate is None
                else _mean_or_none(candidate.detection_scores),
                "fuzzing_count": 0 if candidate is None else len(candidate.fuzzing_scores),
                "fuzzing_mean": None
                if candidate is None
                else _mean_or_none(candidate.fuzzing_scores),
                "status": _status_label(candidate),
            }
        )

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "subset_path": subset_path.as_posix(),
        "subset_size": subset_size,
        "rows": rows_with_results,
    }

    html_sections = []
    for threshold_slug in sorted(grouped):
        violin_name = _write_fuzzing_violin(threshold_slug, grouped[threshold_slug])
        cards = "".join(
            _card(row, candidate, subset_size) for row, candidate in grouped[threshold_slug]
        )
        violin_html = (
            ""
            if violin_name is None
            else (
                '<div class="plot-card">'
                f'<img src="./jose_threshold_matrix_v2_200_assets/{violin_name}" '
                f'alt="{threshold_slug} fuzzing violin plot" />'
                "</div>"
            )
        )
        html_sections.append(
            f"""
            <section class="threshold">
              <div class="threshold-head">
                <h2>{threshold_slug}</h2>
                <p>{grouped[threshold_slug][0][0]["harvest_subrun_id"]}</p>
              </div>
              {violin_html}
              <div class="grid">{cards}</div>
            </section>
            """
        )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta http-equiv="refresh" content="60" />
  <title>Jose Threshold Sweep Results</title>
  <style>
    :root {{
      --bg: #f3efe6;
      --panel: #fffdf8;
      --ink: #17130f;
      --muted: #6a6056;
      --line: #d8d0c5;
      --accent: #8e3500;
      --shadow: 0 10px 28px rgba(23, 19, 15, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(142,53,0,0.08), transparent 25%),
        linear-gradient(180deg, #f8f3ea 0%, var(--bg) 100%);
    }}
    main {{
      width: min(1500px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,255,255,0.84), rgba(255,248,238,0.96));
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      padding: 24px 28px;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(2rem, 4vw, 3.2rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
    }}
    .threshold {{
      margin-top: 22px;
    }}
    .threshold-head {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 10px;
    }}
    .threshold-head h2 {{
      margin: 0;
      font-size: 1.5rem;
    }}
    .threshold-head p {{
      margin: 0;
      color: var(--muted);
      font-family: "IBM Plex Mono", monospace;
      font-size: 0.8rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 14px;
    }}
    .plot-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      padding: 12px;
      margin-bottom: 14px;
    }}
    .plot-card img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      padding: 16px;
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
      margin-bottom: 8px;
    }}
    .eyebrow {{
      margin: 0 0 4px;
      color: var(--accent);
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }}
    h3 {{
      margin: 0;
      font-size: 1.15rem;
      letter-spacing: -0.02em;
    }}
    .status {{
      display: inline-flex;
      padding: 5px 9px;
      border-radius: 999px;
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-family: "IBM Plex Mono", monospace;
      border: 1px solid rgba(0,0,0,0.08);
      background: #eee7da;
    }}
    .status-scoring, .status-running {{ background: #f4e4b5; }}
    .status-interpreted {{ background: #e4e7c2; }}
    .status-waiting, .status-queued {{ background: #ece6dc; }}
    .meta {{
      margin: 0 0 10px;
      color: var(--muted);
      min-height: 2.8em;
      font-size: 0.94rem;
    }}
    .metrics {{
      margin: 0;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px 12px;
    }}
    .metrics div {{
      background: rgba(255,255,255,0.55);
      border: 1px solid rgba(0,0,0,0.05);
      padding: 8px 10px;
    }}
    dt {{
      margin: 0 0 4px;
      color: var(--muted);
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    dd {{
      margin: 0;
      font-family: "IBM Plex Mono", monospace;
      font-size: 0.82rem;
    }}
    .bars {{
      margin-top: 12px;
      display: grid;
      gap: 8px;
    }}
    .bar-row {{
      display: grid;
      grid-template-columns: 44px 1fr;
      gap: 10px;
      align-items: center;
      font-family: "IBM Plex Mono", monospace;
      font-size: 0.75rem;
    }}
    .bar {{
      height: 10px;
      background: rgba(0,0,0,0.08);
      border-radius: 999px;
      overflow: hidden;
    }}
    .bar > div {{
      height: 100%;
      background: linear-gradient(90deg, #c66a2b, #8e3500);
    }}
    .paths {{
      margin: 12px 0 0;
      color: var(--muted);
      font-size: 0.75rem;
    }}
    code {{
      font-family: "IBM Plex Mono", monospace;
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Jose Threshold Sweep Results</h1>
      <p>
        Generated from live autointerp subruns matched against the sweep manifest.
        Auto-refreshes every 60 seconds. Shared subset: <code>{subset_path.as_posix()}</code>.
      </p>
    </section>
    {"".join(html_sections)}
  </main>
</body>
</html>
"""

    html_target = Path(out_html) if out_html is not None else OUT_HTML
    json_target = Path(out_json) if out_json is not None else OUT_JSON
    html_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    html_target.write_text(html_doc)
    json_target.write_text(
        json.dumps(
            payload,
            indent=2,
            default=lambda x: None if isinstance(x, float) and math.isnan(x) else x,
        )
    )
    print(html_target)
    print(json_target)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
