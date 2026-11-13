"""Where the integer-only-adv pipeline writes, under `<run_dir>/integer_only_adv/` (`AdvLayout`);
it reads the integer-only pipeline's pool, check set and score settings from
`<run_dir>/integer_only/` (`integer_only.layout.Layout`):

    README.md               index of everything below (rewritten after every stage)
    config.yaml             the pinned `IntegerOnlyAdvConfig`
    status/<stage>.done     a finished stage's facts (JSON); `<stage>.failed` holds the error
    coefficients.json       train: the run's final-step coefficients (`FinalCoefficients`)
    training/filter_config.yaml  train: the resolved `CIFilterConfig` the steps ran
    training/metrics.jsonl  train: one row per logged step
    training/state_<k>/     train, while running: the resumable state after step k
    ci_fn/                  train: the fine-tuned CI fn (orbax, leaves of the run's CI fn)
    kept.npz                train, with `prune`: `{site: (C,) bool}` the components kept (the rest removed)
    max_ci.npz              alive: `{site: (T, C)}` max fine-tuned CI over the pool
    integer-only-adv.json   alive: the integer-only-adv components
    final.json              score: the integer-only-adv model on `Q`
    comparison.tsv          score: the global-mask sweep summary plus the integer-only-adv row
    datasets/               collect: index.npz, causal_importances/ (binarized fine-tuned CI) and
                            inner_activations/, each with its README.md
    ab_grids.html           applet: the applet; its data in `ab_grids/`
    logs/                   the pod driver's stage logs

`T` (integer) is the prompt length, `C` (integer) a site's number of components."""

import json
from dataclasses import dataclass
from pathlib import Path

STAGES = ("train", "alive", "score", "collect", "applet")


@dataclass(frozen=True)
class AdvLayout:
    root: Path

    @staticmethod
    def of_run(run_dir: Path) -> "AdvLayout":
        return AdvLayout(run_dir / "integer_only_adv")

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
    def coefficients(self) -> Path:
        return self.root / "coefficients.json"

    @property
    def training(self) -> Path:
        return self.root / "training"

    @property
    def filter_config(self) -> Path:
        return self.training / "filter_config.yaml"

    @property
    def metrics(self) -> Path:
        return self.training / "metrics.jsonl"

    @property
    def latest_state(self) -> Path:
        """`{"step": k}` naming the newest complete `state_<k>/`."""
        return self.training / "latest_state.json"

    def state(self, step: int) -> Path:
        return self.training / f"state_{step}"

    @property
    def ci_fn(self) -> Path:
        return self.root / "ci_fn"

    @property
    def kept(self) -> Path:
        return self.root / "kept.npz"

    @property
    def max_ci(self) -> Path:
        return self.root / "max_ci.npz"

    @property
    def alive(self) -> Path:
        return self.root / "integer-only-adv.json"

    @property
    def final(self) -> Path:
        return self.root / "final.json"

    @property
    def comparison(self) -> Path:
        return self.root / "comparison.tsv"

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


ADV_MODEL = "integer_only_adv"
"""The integer-only-adv model's row name in `comparison.tsv`."""

VIEW_LABELS = {
    "ci": "integer-only-adv CI > {threshold}",
    "inner": "inner act (integer-only-adv model)",
}

README = """\
# Integer-only-adv CI filter

Run `{run_dir}`, checkpoint step {step}. Method, settings and stages:
`notes/integer_only/plan_adv.md` in the repository (commit `{commit}`). A component is a site
name (a module such as `layers.26.mlp.down_proj`) plus its index within that site's
decomposition, everywhere below.

Notation: `t` (integer, 0 <= t < T = {seq_len}) is a token position; `N` = {n_prompts} pool
prompts, row `i` = row `i` of `../integer_only/pool.npz`. The integer-only-adv CI is the run's
CI function after the fine-tune below, its output head's lower-leaky CI capped entrywise by the
run's own (the ceiling), evaluated on the original model's activations. A component is ON at
`t` on a prompt when its integer-only-adv CI there exceeds {threshold}. `q` is the
last-position next-token distribution renormalized over the 1000 tokens of the integers 0..999,
and `KL_int` the KL divergence of the original model's `q` to another model's (nats). `Q` is
the integer-only pipeline's check set (`../integer_only/check_set.npy`).

The fine-tune: {steps} Adam steps of batch {batch} on the pool rows outside `Q`, the components
frozen, against `KL_int` at the last position under the run's merged stochastic-subset
persistent-PGD term (the weight delta adversarially ablated, as in the run's target
stream), plus the run's importance minimality; every
coefficient is the run's at its final training step (`coefficients.json`).

## Files

- `config.yaml`: the settings of this pipeline the run does not fix.
- `status/`: one `<stage>.done` (JSON facts) per finished stage, or `<stage>.failed`.
- `coefficients.json`: the run's final-step loss coefficients, adversary optimizer settings and
  CI-optimizer betas, and the run terms not carried (`dropped`, with reasons).
- `training/filter_config.yaml`: the resolved training configuration; `training/metrics.jsonl`:
  loss, reconstruction (`recon`, mean `KL_int`), imp-min terms, L0 and seconds per logged step.
- `ci_fn/`: the fine-tuned CI function (orbax; the run's CI fn's array leaves in order).
- `max_ci.npz`: `{{site: (T, C) float32}}`, the max integer-only-adv CI over all N prompts
  (`C` = the site's number of components).
- `integer-only-adv.json`: the {n_alive} integer-only-adv components (ON at some position of some
  pool prompt), the schema of `../integer_only/integer-only.json`.
- `final.json`: the integer-only-adv model (exactly those components on at every position, every
  other component and the weight delta off) scored on `Q` against the original model.
- `comparison.tsv`: `../integer_only/sweep/summary.tsv`'s rows plus the `{adv_model}` row.
- `datasets/`: per integer-only-adv component and position where it is ever ON, its inner
  activation in the integer-only-adv model and its binarized integer-only-adv CI over all N
  prompts; see `datasets/README.md`, which indexes `causal_importances/` and
  `inner_activations/` (one README each).
- `ab_grids.html` (data in `ab_grids/`): the (a, b)-grid applet of every integer-only-adv
  component, per operation and position, toggling between the binarized integer-only-adv CI
  and the integer-only-adv model's inner activation.

## Stages

{stages}
"""
