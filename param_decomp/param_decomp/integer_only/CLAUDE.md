# param_decomp.integer_only — learned global mask over a finished decomposition

Starts from a FINISHED all-blocks decomposition of a Llama target, keeps its components frozen,
and learns ONE global mask (the same components on at every prompt and position, weight delta
off) that preserves the target's last-position distribution among the integer tokens 0..999.
Method, settings, stages and output layout: `notes/integer_only/plan.md`.

- `config.py` — `IntegerOnlyConfig` (checkpoint step, prompts file + per-operation templates,
  the ON threshold, the λ sweep and learner settings, the reference KL of the λ rule) and
  `LEARN_FIELDS`, pinned per λ.
- `pool.py` — the prompts-file pool (`PromptPool`: tokens, operation, operands, answer, position
  labels), `P` (answers >= 0), the check set `Q`, the integer token table and the answer-token
  check.
- `model.py` — the factored forward `y = ((x V) ⊙ m) U` over a stored component set (`Slots`,
  canonical order: sites sorted by name, component index ascending), the dense target through
  the same block equations, and the inner activations `x V` of the masked model.
- `objective.py` — `q` (renormalized over the integer tokens), `KL_int`, the per-prompt scores
  and the jitted passes (target `log q`, probe, mask gradient).
- `ci_pass.py` — a CI fn over the pool (output head, lower-leaky, under given constraints:
  `UNCONSTRAINED` for the run's own CI): per-batch max and column gathers.
- `learn.py` — the sampled straight-through learner for one λ (resumable `state.npz`).
- `pick.py` — scoring on `Q`, the λ rule, the `integer-only.json` schema.
- `collect.py` — the dataset (ever-on columns per position, fp16 inner, bit-packed CI) and its
  README text; `collect_dataset` writes it for any component set, CI fn and constraints.
- `applet.py` — `(a, b)` slices for `ci_filter/grids_app.html` from the dataset
  (`write_applet`, view labels per pipeline).
- `layout.py` — the output layout (module docstring) and the top-level README text.
- `run.py` — the CLI: one subcommand per stage, done/failed markers in `status/`.
- `adv/` — the integer-only-adv CI filter (`notes/integer_only/plan_adv.md`): the run's CI fn
  fine-tuned (components frozen, ceiling on, prune off) on this pool against `KL_int` under the
  run's merged persistent-PGD term (weight delta ablated as in the run's target stream), every coefficient the run's at its
  final step. `config.py` (`IntegerOnlyAdvConfig`), `coefficients.py` (final-step resolution,
  the `CIFilterConfig` it trains under), `adversary.py` (the run's target-stream adversary from
  its training checkpoint), `train.py` (the resumable loop), `layout.py` (`$RUN/integer_only_adv/`
  and its README texts), `run.py` (stages `train`, `alive`, `score`, `collect`, `applet`; reads
  `$RUN/integer_only/`'s pool, `Q` and score settings).

The pod driver is `notes/integer_only/pod/run.sh` (stage 3 runs one process per λ, one GPU each);
it then calls `run_adv.sh`, the adv stages on all GPUs.
