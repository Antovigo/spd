# param_decomp.ci_filter — CI fine-tuning filter

Starts from a FINISHED decomposition, freezes its components, and fine-tunes the output head's
CI fn (the whole fn: trunk and both heads' parameters are trainable; only the output head gets
gradient) against a NARROWER output objective, so components that objective does not need
switch off per prompt. Target stream only: no non-target pass, no hidden-activation pass.

- `config.py` — `CIFilterConfig` (run + step, `init` from the run or from an earlier filter,
  the arithmetic pool, the objective, schedules, batching) and the `PoolEval` summary schema.
- `pool.py` — the `operations x a x b` prompt pool (`"<a><op><b>="`, operation blocks in
  config order, each row-major) and the answer-token set (all-digit tokens + `-`).
- `objective.py` — pure last-position scores: full-vocabulary KL, answer-restricted KL
  (restricted logits, renormalized), top-1 agreement of both.
- `step.py` — the jitted microbatch gradient, gradient accumulation, the Adam update, the
  pool evaluation (continuous CI, rounded CI, alive-set and all-on global masks).
- `grid.py` + `grids_app.html` — the `(a, b)`-grid applet over every operation x position,
  one lazily loaded file per slice (payload format of `experiments/lm/ab_grid_dataset.py`).
- `checkpoint.py` — orbax save/restore of the fine-tuned CI fn (chains objective 2 onto 1).
- `paths.py` — `CIFilterOutputs`: `<run_dir>/analysis/ci_filter/step_<step>/<cf-id>/` (layout
  in the module docstring).
- `scripts/run_ci_filter.py` — entry point.
- `configs/` — template seats for objective 1 (`last_position_kl.yaml`) and objective 2
  (`last_position_integer_kl.yaml`, `init` from objective 1's id); parse-gated by
  `tests/ci_filter/test_ci_filter.py`. Hardware-specific copies (batch split via
  `microbatch_size`, run referenced by `{kind: dir}`) are launch-local, not committed.

Semantics fixed by design: deterministic masks = the lower-leaky CI (no stochastic sources,
no adversary), weight delta OFF, objective scored at the LAST position only while CI (and the
imp-min) covers every position. Imp-min is the decomposition's Geman-McClure term on the
upper-leaky CI with its frequency penalty at a FIXED `a'` (microbatch-invariant). A component
is alive iff its output CI exceeds `alive_threshold` at any position of any pool prompt; the
same cut defines the rounded masks.

Run: `python -m param_decomp.ci_filter.scripts.run_ci_filter --config <yaml> --data_root <root>
[--filter_id cf-xxxxxxxx]`. The mesh is fsdp over every visible device (`restore_jax_run`).
