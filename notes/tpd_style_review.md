# tPD style / cleanliness review

Living doc for the rolling STYLE/CLEANLINESS janitor over the **targeted PD (tPD)** delta on
`feature/targeted-jax`. Scope = the tPD delta only (`git diff feature/jax..feature/targeted-jax`
+ uncommitted). Rule sources: repo `CLAUDE.md` "Coding guidelines"; `param_decomp/CLAUDE.md`
("the one rule", HLO-baking); `param_decomp_lab/experiments/CLAUDE.md`. Behavior-preserving
fixes only; anything numerics/semantics-changing goes under "Deferred", never into code.

**Status (pass 4):** since pass 3 the branch REBUILT the LM targeted composition root — the
`lm_targeted/` tree that pass 3 recorded as deleted (B1) is back and functional, and the tPD
two-pass now runs end-to-end. Pass 4 reviewed the fresh LM code
(`experiments/lm_targeted/{config,data,run,launch,test_lm_targeted}.py`, the
`numpy_pandas_4L_targeted.yaml` config + `prompts/`), the `recon_positions` seam in
`train.py` / `run.py`, SPEC §11 S38, `experiments/config.py`
(`EXCLUDED_NONTARGET_LOSS_CONFIGS` now drops both PGD variants), and `adapters/pd.py`. The
toy delta is unchanged from pass 3 (still conforms). Fixed 3 behavior-preserving items this
pass (P1–P3 below); `make check` clean (0/0); required suite green (55 passed, 2 skipped).

**Status (pass 3, @ HEAD `14f0bd33d`):** ALL prior findings (B1–B3, M1–M4, m1–m7, N1–N7 =
24) were RESOLVED in commit `14f0bd33d` ("refactor(tPD): clean up per the style review").
(Note: pass-3 finding B1 recorded the `lm_targeted/` tree as deleted "until the LM path is
implemented" — it has since been rebuilt, so B1's deletion no longer holds; the new tree is
reviewed fresh in pass 4.)

---

## Resolved (all fixed in `14f0bd33d`, re-verified pass 3)

Grouped by the original finding ids. Each verified against current code.

**Blockers**
- **B1** — the `lm_targeted/` stub tree (7 `NotImplementedError` bodies) + the `pd-lm-targeted`
  console script + the dead `numpy_pandas_4L_targeted.yaml` were DELETED. Verified: no
  `lm_targeted` dir, no `lm-targeted`/`lm_targeted` entry in `param_decomp_lab/pyproject.toml`,
  no importers. (Per brief: do NOT recreate this tree — LM path is rebuilt when implemented.)
- **B2** — the forked `NontargetConfig` / `EXCLUDED_NONTARGET_LOSS_CONFIGS` /
  `build_nontarget_loss_metrics` went with `lm_targeted/`. The single shared copies live in
  `experiments/config.py` (generic `NontargetConfig[D]`, `NonNegativeFloat` ratio) and are
  imported by both toys.
- **B3** — no re-implemented faith validator; the toys call the shared
  `assert_targeted_faithfulness_off(self.pd)` (`tms/config.py`, `resid_mlp/config.py`).

**Major**
- **M1** — target + non-target recon loops are ONE nested helper `recon_grid(grid_terms, ci,
  batch, clean_output, leading, key_offset, force_delta_on)` (`train.py:420`), called twice.
  The non-target pass is a call, not a copy. Control flow verified byte-equivalent to the old
  target loop (`key_offset=1`, same match arms, same PPGD start-frac gate).
- **M2** — the four `Any = None` sentinels are gone; non-target batch prep is a typed
  `_NontargetInputs` record (`train.py:78`) built only when `nontarget_loss_surface is not
  None`, consumed inside `loss_fn`.
- **M3** — the dead `else imp_min` fallback is gone; nt imp-min reads `nontarget_loss_surface.imp`.
- **M4** — SPEC §11 rewritten to the normative register: real `S34-S37` ids, two-pass
  pseudocode, no `> STATUS:` block, no `S-tPD-*` scheme. Code cites `S34-S37` throughout.

**Minor / nits**
- **m1** — no "today's …" / "byte-identical" change-narration left in tPD docstrings/comments
  (verified by grep; remaining "today's" hits are all in out-of-scope `LOSS_PARITY_DESIGN.md`).
- **m2** — `NontargetPass` docstring trimmed to the one gotcha (delta forced on, one step).
- **m3** — `assert_targeted_faithfulness_off` messages trimmed to the actionable half.
- **m4** — the "bakes into the jit" wording on the toy `probe` now says WHY the bake is fine
  (tiny host array, toy CPU jit), distinct from the HLO-baking ANTI-pattern.
- **m5 / N5** — the swept-hyperparameter / "Mirrors the torch convergence run" lab-notes are
  out of the YAMLs; the resid config points at `notes/targeted_jax_plan.md` for the sweep.
- **m6** — `impmin_coeff_ratio` is `NonNegativeFloat` in the single shared generic.
- **m7** — SPEC S37 now explains the toys' `impmin_coeff_ratio: 20` vs paper ~2 (tuned for
  the small toys) instead of silently disagreeing.
- **N1** — `param_decomp_lab/experiments/tms/test_targeted.py` covers the tPD seams:
  `active_indices` column masking + out-of-range assert, `build_nontarget_loss_metrics`
  (drops excluded, scales imp-min, leaves originals untouched), and a 3-step two-pass step
  smoke exercising `force_delta_on` + the `nontarget/*` metrics.
- **N2** — `delta_override: float | None` → `force_delta_on: bool` (`train.stochastic_entry_masks`);
  the dead `delta_override` param on `adversary.source_masks` was removed entirely (no caller
  passed it — `adversary.py` now carries NO tPD delta vs `feature/jax`).
- **N3** — `nontarget_losses` → `nontarget_loss_surface` (the built `LossSurface`), distinct
  from the `nontarget: NontargetPass` config and the `NontargetConfig` schema.
- **N4** — TMS config header + data comment now name `[2, 13, 26]` (matching the live
  `active_indices`); the stale feature-24 anecdote is gone.
- **N6** — noted-not-fixed originally (cross-domain toy parallelism the repo tolerates); the
  shared bits (`build_nontarget_loss_metrics`, `NontargetPass(...)` tail) are the ONE shared
  helper + config, the per-domain sampling differs by embed. Acceptable as-is.
- **N7** — the non-target data key salt carries `# distinct salt (23) from the target
  stream's (17)` in both toy `run.py`s.

---

## Resolved this pass (pass 4)

- **P1** — SPEC §11 S38 (`recon_positions`) carried a trailing sentence about the paper's ~2
  vs the toys' ~20 `impmin_coeff_ratio`, which is an S37 (imp-min coeff) topic, not an S38
  (recon-position slicing) one. Moved that sentence up to S37 where it belongs.
- **P2** — `lm_targeted/run.py::_nontarget_sample_batch` returned
  `(sample_batch, nontarget_global_batch)` but the sole caller discarded the second element
  (`nontarget_sample_batch, _ = …`). Dropped the dead return value; return the callable only
  (style guide: delete unused code).
- **P3** — `_nontarget_sample_batch` resolved the parquet dir inline as
  `Path(cfg.nontarget.data.data_files).parent` with a `# pyright: ignore[reportArgumentType]`,
  duplicating (minus its asserts) `config.nontarget_parquet_dir`, which `_targeted_data`
  already uses. Reused `nontarget_parquet_dir` (dropping the pyright-ignore) and promoted it
  from `_nontarget_parquet_dir` to a public name since it is now shared across the two modules.

## Open findings (behavior-preserving), ranked

None. The tPD delta (toy + LM) is clean against the guide after P1–P3.

---

## Deferred (behavior-changing / needs a human)

None outstanding. For the record, the following were considered and are NOT bugs — leave as-is
(the brief pins them as intentional design):

- `force_delta_on: bool` (SPEC S35), the shared `recon_grid(...)` helper, `_NontargetInputs`,
  the `nontarget_loss_surface` name, SPEC §11 `S34-S37`, and the shared
  `NontargetConfig`/`build_nontarget_loss_metrics`/`assert_targeted_faithfulness_off` in
  `experiments/config.py` — all correct, keep.
- The rebuilt `lm_targeted/` tree: reviewed fresh in pass 4. Its `run.py::train` mirrors
  `lm/run.py::train` (mesh asserts, eval-cadence assert block, `run_key` split) — the
  composition-root parallelism the repo tolerates, not copy-paste to collapse. Kept as-is.
- The persistent-PGD-forbidden guard in `LMTargetedExperimentConfig._assert_targeted_invariants`
  + the two `assert not force_delta_on` arms in `recon_grid` + `EXCLUDED_NONTARGET_LOSS_CONFIGS`
  dropping both PGD variants: intentional design (SPEC S35), leave.
- `impmin_coeff_ratio: 20.0` in the two toy YAMLs vs the paper's ~2: a TUNED config value, not
  a style defect — SPEC S37 documents the gap. Changing it would change training numerics, so
  it is out of janitor scope regardless.

---

## Incidental correctness / positive notes (not style, carried forward)

- `single_feature_ci` in both toy `run.py`s now passes `remat=False` to the CI fn — a latent
  fix: `CIFn.__call__` / `LayerwiseMLPCIFn.__call__` declare `remat` as a REQUIRED keyword-only
  arg, and pre-tPD `feature/jax` called `ci_fn(...)` with no `remat`, surviving only because
  `ci_fn: Any` dodged pyright. Correct now.
- The `force_delta_on` seam is a static scalar bool threaded through exactly the mask builder
  SPEC S35 names — HLO-baking-safe, no ContextVar/global.
- The two `assert not force_delta_on` arms in `recon_grid` (fresh-PGD / PPGD) are correct
  fail-fast (the non-target set must exclude adversarial sources, SPEC S35).
- `sample_sparse_features(..., active_indices=...)` asserts `all(0 <= i < n_features …)` and
  zeroes non-target columns cleanly in both toys. Well-asserted, jaxtyping'd.
- The non-target imp-min shares the target's annealed `imp_min_param` and scales only the
  coeff (`nontarget_loss_surface.imp.coeff`) — matches SPEC S37 (scale the coeff, not the
  anneal schedule).

## Categories with nothing to report
- **`Union`/`Optional`/`List`/`Dict` / `from __future__`**: none — all PEP-604 / lowercase.
- **`try/except` / defensive fallbacks / sentinel defaults**: none (M2/M3 removed the last).
- **Sphinx/RST / double-backtick in docstrings**: none.
- **Change-narrating comments**: none left in the tPD delta.
- **Core task-agnosticism**: no tPD/toy concept leaks into a generic core file — `NontargetPass`
  lives in the engine and takes an already-filtered `list[AnyLossMetricConfig]`; the
  `active_indices` / target-feature vocabulary stays lab-side.
