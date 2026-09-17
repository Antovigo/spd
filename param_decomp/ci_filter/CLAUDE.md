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
  (restricted logits, renormalized), top-1 agreement of both, and the cross-entropy of the TRUE
  result (`last_position_answer_ce`: accuracy, not faithfulness — the filter may beat the model).
  Every jit that scores an objective takes the pool's per-prompt target index vector.
- `step.py` — the jitted microbatch gradient (deterministic CI-masked recon), gradient
  accumulation (on device or host), the Adam update, the pool evaluation (continuous CI,
  rounded CI, alive-set and all-on global masks), the end-of-run fresh-PGD recon eval.
- `ppgd.py` — the alternative recon, the decomposition's own merged stochastic-subset
  persistent-PGD term (SPEC S34, weight delta ON) scored by the filter objective: the
  persistent adversary, its warmup ascents, the main microbatch gradient and the final ascent.
- `attribution.py` + `scripts/run_attribution.py` — helpers vs interferers on the TRUE answer:
  the first-order screen (`-sum_t m * d log p(correct) / d m`, every component in one backward)
  and the causal single-component ablation of the extremes. POSITIVE = ablating it raises the
  correct answer's probability (interferes).
- `ablation.py` + `scripts/mask_ablations.py` — KL (last position and all positions) of a run's
  or a filter's CI fn under all-on / alive-union / dead-only / per-prompt / per-prompt+dead
  masks, weight delta off.
- `grid.py` + `grids_app.html` — the `(a, b)`-grid applet over every operation x position,
  one lazily loaded file per slice (payload format of `experiments/lm/ab_grid_dataset.py`).
- `checkpoint.py` — orbax save/restore of the fine-tuned CI fn (chains objective 2 onto 1), and
  `starting_ci` / `trained_ci`: the CI fn a filter starts from and the constraints capping it.
- `paths.py` — `CIFilterOutputs`: `<run_dir>/analysis/ci_filter/step_<step>/<cf-id>/` (layout
  in the module docstring).
- `scripts/run_ci_filter.py` — entry point.
- `configs/` — template seats for objective 1 (`last_position_kl.yaml`) and objective 2
  (`last_position_integer_kl.yaml`, `init` from objective 1's id); parse-gated by
  `tests/ci_filter/test_ci_filter.py`. Hardware-specific copies (batch split via
  `microbatch_size`, run referenced by `{kind: dir}`) are launch-local, not committed.

Semantics: `recon` is either `ci_masked` (deterministic masks = the lower-leaky CI, weight delta
OFF) or `merged_stochastic_ppgd` (the decomposition's own adversarial term, weight delta ON; the
H100 path — a 1x L40 only fits `ci_masked` at microbatch 256 with the host-held gradient sum).
The objective is scored at the LAST position only while CI (and the imp-min) covers every
position; the evaluations always use deterministic masks with the delta off. Imp-min is the decomposition's Geman-McClure term on the
upper-leaky CI with its frequency penalty at a FIXED `a'` (microbatch-invariant). A component
is alive iff its output CI exceeds `alive_threshold` at any position of any pool prompt; the
same cut defines the rounded masks. Constraints (`step.CIConstraints`, threaded explicitly after
`ci_fn` in every jit, `UNCONSTRAINED` = none) bound every CI read (`step.output_ci`): with
`ci_ceiling` the CI is the entrywise minimum of the trained fn's and each frozen ceiling fn's (the
starting point's CI, chained through `init: ci_filter`); with `prune_dead` the components dead on
the pool at the start get CI 0 and zeroed U/V (`remove_components`, saved as `alive/kept.npz` and
inherited by later filters).

Run: `python -m param_decomp.ci_filter.scripts.run_ci_filter --config <yaml> --data_root <root>
[--filter_id cf-xxxxxxxx]`. The mesh is fsdp over every visible device (`restore_jax_run`).
