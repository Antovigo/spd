"""Where the pipeline writes, under `<run_dir>/integer_only/` (`Layout.root`):

    README.md               index of everything below (rewritten after every stage)
    config.yaml             the pinned `IntegerOnlyConfig`
    status/<stage>.done     a finished stage's facts (JSON); `<stage>.failed` holds the error
    pool.npz                stage 1: the tokenized prompt pool (`pool.PromptPool`)
    max_ci.npz              stage 1: `{site: (T, C)}` max original output CI over the pool
    candidates.json         stage 1: the candidates (max CI > threshold), `integer-only.json`'s schema
    check_set.npy           stage 2: `Q`, rows of the pool in draw order
    reference_log_q.npy     stage 2: `(N, 1000)` float32 target `log q` per pool row
    sweep/lam_<λ>/          stage 3: learn_config.json, log.jsonl, state.npz; stage 4: final.json
    sweep/summary.tsv       stage 4: every mask (and the target, the candidates) scored on `Q`
    integer-only.json       stage 4: the chosen mask
    datasets/               stage 5: inner activations and binarized CI (its own README.md)
    ab_grids.html           stage 6: the applet; its data in `ab_grids/`
    logs/                   the launcher's stage logs

`<stage>` is one of `STAGES`; the learn stage's marker is per `λ`: `learn_lam_<λ>`."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

STAGES = ("candidates", "problem", "learn", "pick", "collect", "applet")


class StatusFolder(Protocol):
    """A pipeline folder with stage markers: `status/<marker>.done` and `.failed`."""

    @property
    def status(self) -> Path: ...

    def done(self, marker: str) -> Path: ...

    def failed(self, marker: str) -> Path: ...


def lam_name(lam: float) -> str:
    return f"lam_{lam:g}"


@dataclass(frozen=True)
class Layout:
    root: Path

    @staticmethod
    def of_run(run_dir: Path) -> "Layout":
        return Layout(run_dir / "integer_only")

    @property
    def readme(self) -> Path:
        return self.root / "README.md"

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def status(self) -> Path:
        return self.root / "status"

    def done(self, marker: str) -> Path:
        return self.status / f"{marker}.done"

    def failed(self, marker: str) -> Path:
        return self.status / f"{marker}.failed"

    @property
    def pool(self) -> Path:
        return self.root / "pool.npz"

    @property
    def max_ci(self) -> Path:
        return self.root / "max_ci.npz"

    @property
    def candidates(self) -> Path:
        return self.root / "candidates.json"

    @property
    def check_set(self) -> Path:
        return self.root / "check_set.npy"

    @property
    def reference(self) -> Path:
        return self.root / "reference_log_q.npy"

    @property
    def sweep(self) -> Path:
        return self.root / "sweep"

    def lam_dir(self, lam: float) -> Path:
        return self.sweep / lam_name(lam)

    @property
    def summary(self) -> Path:
        return self.sweep / "summary.tsv"

    @property
    def integer_only(self) -> Path:
        return self.root / "integer-only.json"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def applet_page(self) -> Path:
        return self.root / "ab_grids.html"

    @property
    def applet_data(self) -> Path:
        return self.root / "ab_grids"

    def facts(self, marker: str) -> dict[str, object] | None:
        path = self.done(marker)
        return json.loads(path.read_text()) if path.exists() else None


README = """\
# Integer-only global mask

Run `{run_dir}`, checkpoint step {step}. Method, settings and stages: `notes/integer_only/plan.md`
in the repository (commit `{commit}`). A component is a site name (a module such as
`layers.26.mlp.down_proj`) plus its index within that site's decomposition, everywhere below.

Notation: `t` (integer, 0 <= t < T = {seq_len}) is a token position, labelled {labels}; `N` =
{n_prompts} prompts, row `i` = line `i` of the prompts file `{prompts}`, counting from 0. A
component is ON at `t` on a prompt when its causal importance (CI), the output head's
lower-leaky CI of the run's own CI function on the original model's activations, exceeds the
threshold {threshold}. `q` is the last-position next-token distribution renormalized over the
1000 tokens of the integers 0..999, and `KL_int` the KL divergence between two such `q` (nats).

## Files

- `config.yaml`: every setting of this pipeline.
- `status/`: one `<stage>.done` (JSON facts: counts, checks, seconds) per finished stage, or
  `<stage>.failed` with the error.
- `pool.npz`: the tokenized prompts: `tokens` (N, T); `op` (N,), the operation as an index into
  `ops`; `a`, `b` (N,), the two integer operands; `r` (N,), the true answer (integer, `a + b` or
  `a - b`); `position_labels` (T,).
- `max_ci.npz`: `{{site: (T, C) float32}}`, the maximum CI over all N prompts per position and
  component (`C` = the site's number of components).
- `candidates.json`: the {n_candidates} components ON at some position of some prompt. Only they
  exist in the learned masks; every other component is permanently off.
- `check_set.npy`: `Q`, {check_size} pool rows of `P` (the prompts with answer >= 0), in the order
  they were drawn; the learner never trains on them and probes on the first {eval_size}.
- `reference_log_q.npy`: `(N, 1000)` float32, the original model's `log q` per pool row.
- `sweep/lam_<λ>/`: one learned mask per sparsity weight `λ` (nats per kept component):
  `learn_config.json` (pinned settings), `log.jsonl` (probe every {eval_every} steps), `state.npz`
  (`logit` per candidate in `candidates.json` order, Adam moments, step; the mask keeps
  `logit > 0`), `final.json` (the mask scored on all of `Q`).
- `sweep/summary.tsv`: every mask, the original model and the all-candidates model on `Q`.
- `integer-only.json`: the chosen mask, the highest `λ` whose mean `KL_int` to the original model
  on `Q` is below {reference_kl_int} ({reference_source}). {chosen}
- `datasets/`: per integer-only component and per position where it is ever ON, its inner
  activation in the integer-only model and its binarized original CI over all N prompts; see
  `datasets/README.md`.
- `ab_grids.html` (data in `ab_grids/`): the (a, b)-grid applet of every integer-only component,
  per operation and position, toggling between the binarized original CI and the integer-only
  model's inner activation.

## Stages

{stages}
"""
