# `param_decomp_lab/eval_metrics/`

Batteries-included eval `Metric` set for the in-repo experiments, plus the YAML
dispatch wiring (`AnyEvalMetricConfig` + `EVAL_METRIC_CLASSES`).

## Why this lives in the lab (and not in core)

Eval metrics are **user-extensible** by design. We expect users to add their own eval
metrics for their own decomposition runs, so the metric set isn't part of the public
core API — anyone can instantiate a `Metric` subclass and pass it to
`EvalLoop(metrics=...)`.

This is the deliberate split from **loss metrics**, which are canonical and curated:
loss metrics live in `param_decomp/metrics/` and adding one is a core change. See
[`../../param_decomp/metrics/CLAUDE.md`](../../param_decomp/metrics/CLAUDE.md).

This dir is just the set of eval metrics *we* ship for the in-repo experiments.

## YAML dispatch

The in-repo experiments validate the YAML `eval.metrics` list via the
`AnyEvalMetricConfig` discriminated union (on `EvalConfig`, see
[`../experiments/CLAUDE.md`](../experiments/CLAUDE.md)), then instantiate each entry
with `EVAL_METRIC_CLASSES`:

```python
from param_decomp_lab.eval_metrics import EVAL_METRIC_CLASSES
metrics = [EVAL_METRIC_CLASSES[m.type](m) for m in cfg.eval.metrics]
```

Both pieces live in `__init__.py`.

## Adding a lab eval metric

1. Define `<Name>(Metric[<Name>Config])` + its `<Name>Config(BaseConfig)` in
   `<name>.py`. The config must carry a unique `type: Literal["<Name>"]` discriminator.
2. Append the config to `AnyEvalMetricConfig` in `__init__.py`.
3. Append the class to `EVAL_METRIC_CLASSES` in `__init__.py`.

The class extends `Metric` from `param_decomp.metrics.base`. Lifecycle is the same as
any other metric: `__init__(cfg)` → `bind(model, device)` → `update(ctx)` →
`compute()`.

## External / one-off eval metrics

If you're writing your own caller (not using the in-repo experiment runners), skip the
dispatch table entirely — instantiate your `Metric` subclasses directly and pass them
in `EvalLoop(metrics=...)`. Nothing in the core cares whether they came from a YAML
union or were constructed by hand.

## Targeted (tPD) eval metrics

`TargetReconLoss`, `NontargetReconLoss`, `NontargetCIMeanPerComponent`,
`TargetedCIHeatmap`, and `WeightMagnitude` support targeted decomposition runs. The
three nontarget-data metrics are partitioned out of `EvalLoop.metrics` into
`EvalLoop.nontarget` by `param_decomp_lab/targeted.py::split_eval_metrics` and are fed
by the trainer's mirror nontarget eval loop under `delta_override(1.0)`; the rest stay
in the normal target eval pass.

## `ABGridDataset` — (a,b)-grid snapshots + applet

Slow eval metric (`ab_grid_dataset.py`) for `a<op>b=` prompt pools: at each slow eval
it forwards the whole pool (one cached forward per batch, sharded across DDP ranks)
and writes `<run>/ab_grids/step_<n>.js` — per-subcomponent CI (u8) and normalized
inner-activation `(x·V_c)/‖V_c‖` (f16) grids over (op, a, b) at the configured token
positions — plus `index.html`, a self-contained `file://`-openable applet
(`ab_grids_app.html`, shipped next to the metric) with step/op/position selectors, a
log-scale mean-CI threshold slider, and hover readouts. Full grids are stored only for
subcomponents whose mean CI reaches `mean_ci_floor` (the mean-CI vector is stored for
all, so the cut is visible); disk is ~1 MB per 10 saved subcomponents per snapshot.
On a dual-CI run it records **both** nets from the same cached forward (one CI-fn call
each, no second pass): the payload gains `ci_roles` plus per-module `mean_ci_hidden` /
`ci_hidden` arrays, a subcomponent's grids are saved when *either* net's mean CI reaches
`mean_ci_floor` (so hidden-only components survive the cut), and the applet gains a
green/magenta merge view — subtractive on white, so white = neither, green = hidden-only
(expected), **magenta = output-important but hidden-unimportant, the anomaly the sanity
check looks for**, black = both. The hidden keys are optional and the applet falls back to
the original single-colour rendering for control runs and pre-change snapshots.

Delivery uses `param_decomp_lab/run_artifacts.py::RunDirArtifact` — a `MetricResult`
value the local sink writes verbatim under the run dir (regenerating `manifest.js`)
and wandb/`metrics.jsonl` skip. It lives outside `run_sink` so metrics can import it
without pulling the sink's wandb/infra chain into `eval_metrics` (import cycle).

## Dual-CI (hidden-activation) probes

On a `pd.dual_hidden_ci` run, `CIHiddenActsReconLoss` and `PGDHiddenActsReconLoss` (the
latter from core, registered here as eval-only) both take a `ci_role`. List
`CIHiddenActsReconLoss` **twice**, once per role with distinct `name`s, to read off how much
hidden-activation error each net's CI assignment leaves — that pairing is the direct
measurement of the interference hypothesis, and what makes a dual run comparable to a
single-CI control. See `param_decomp/metrics/CLAUDE.md` for the shared relative-error
definition.

`ABGridDataset` needs no `ci_role`: it records every role the model has (see below).

## Note on `PGDReconLoss` + `StochasticHiddenActsReconLoss`

Both appear in `EVAL_METRIC_CLASSES` even though they're *loss* classes from core.
That's intentional: they're listed here so they can be added to YAML `eval.metrics`
purely for evaluation (without showing up as a training-loss coefficient). When used
as eval-only, their `coeff` is ignored.
