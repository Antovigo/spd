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
- `ci_pass.py` — the run's own CI fn over the pool (output head, lower-leaky, unconstrained):
  per-batch max and column gathers.
- `learn.py` — the sampled straight-through learner for one λ (resumable `state.npz`).
- `pick.py` — scoring on `Q`, the λ rule, the `integer-only.json` schema.
- `collect.py` — the dataset (ever-on columns per position, fp16 inner, bit-packed CI) and its
  README text.
- `applet.py` — `(a, b)` slices for `ci_filter/grids_app.html` from the dataset.
- `layout.py` — the output layout (module docstring) and the top-level README text.
- `run.py` — the CLI: one subcommand per stage, done/failed markers in `status/`.

The pod driver is `notes/integer_only/pod/run.sh` (stage 3 runs one process per λ, one GPU each).
