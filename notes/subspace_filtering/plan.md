# Subspace restriction — implementation plan (Proposal A)

Implement Proposal A of `suggestions.md`: parameterize subcomponents in the SVD
coordinates of the frozen target matrix, so read vectors lie in `row(W)` and write
vectors in `col(W)` by construction. Scope: transformer (LM) decompositions only,
targeted and full-data. The evaluation criterion is the battery's **raw** flavor
(no centering). Branch `feature/subspace_restriction`, worktree
`Code/param-decomp/subspace_restriction`.

Two questions, answered in order:

- **Q1 — is effective-rank selection critical?** Does training still work with all
  ranks (τ = 0), and what happens as τ rises (r shrinks)?
- **Q2 — does it improve the two battery criteria?**
  (a) project activations onto the subcomponent span, run **original** weights
  (`orig_in_span` / `orig_out_span`, raw);
  (b) project onto the original weight's row/col space, run the **circuit**
  (`circuit_in_row` / `circuit_out_col`, raw).
  (b) should pass by construction; (a) is the open empirical question.

## Parameterization

For each decomposed `nn.Linear` with `W [d_out, d_in]`:

- Economy SVD `W = Q_out Σ Q_inᵀ` once at init, fp32, from the frozen target weight.
  Keep `r = #{σ_i > τ·σ_max}`; `τ = 0` keeps `r = min(d_in, d_out)`.
- Frozen **persistent** buffers `Q_in [d_in, r]`, `Q_out [d_out, r]` (stored in the
  checkpoint — avoids relying on SVD determinism across machines/torch versions).
- Learned `A [r, C]`, `B [C, r]`. Effective `V = Q_in A`, `U = B Q_outᵀ`, exposed as
  **properties** so every existing consumer (battery, harvest, editing, eval metrics)
  works unchanged.
- Forward: `component_acts = (x @ Q_in) @ A`; output `(acts @ B) @ Q_outᵀ`.
  `weight` property: `Q_out (Bᵀ Aᵀ) Q_inᵀ`.
- Init matches the dense init in effective space (Gaussians are rotation-invariant):
  `A ~ init_param_(fan_val=d_in)`, `B ~ init_param_(fan_val=C)`.
- Truncated tail `W − W_r` flows into the on-the-fly delta (`calc_weight_deltas` is
  `target_weight − components.weight`); the addsub recipe has no `FaithfulnessLoss`,
  so truncation adds no loss floor.
- Llama-3.1-8B L18 shapes: `gate/up [14336, 4096]` (constrains `U`),
  `down [4096, 14336]` (constrains `V`), `k/v [1024, 4096]` (GQA — constrains `V`),
  `q/o [4096, 4096]` (at τ = 0 a pure rotation; meaningful only once τ truncates).

## Code changes

1. `param_decomp/components.py`
   - Move `V`/`U` parameter creation from `Components.__init__` into
     `LinearComponents` / `EmbeddingComponents` (base keeps `C` and the abstract
     surface).
   - New `SVDLinearComponents(Components)`: buffers `Q_in`/`Q_out`, params `A`/`B`,
     `V`/`U` properties, overridden `get_component_acts` / `forward` / `weight`.
     Same bias handling as `LinearComponents`.
   - New `Components.scale_subcomponents_(keep)` (in-place per-subcomponent scaling):
     dense scales `V`/`U` columns/rows, SVD scales `A`/`B` — used by CI-scaled weight
     decay (the operation commutes with the coordinate form).
   - `make_components(..., svd_rank_threshold: float | None)`: when set, every
     `nn.Linear` target becomes `SVDLinearComponents`; assert no `Embedding` /
     `Conv1D` / `Identity` targets.
2. `param_decomp/configs.py` — `PDConfig.svd_rank_threshold: float | None = None`.
3. `param_decomp/component_model.py` + `param_decomp/optimize.py` — plumb the field
   `Trainer → ComponentModel → make_components`; CI-scaled decay calls
   `scale_subcomponents_`; `tie_component_weights` asserts dense components.
4. `param_decomp_lab/component_model_io.py` — tied-transpose path asserts dense; load
   path needs no change (model is rebuilt from config, state dict carries `A`/`B` and
   the `Q` buffers).
5. CLAUDE.md updates where interfaces changed.

Non-goals: no per-target thresholds, no offset component (Proposal B), no projection
losses (Proposal C), no toy-model support.

## Tests (`param_decomp/tests/test_svd_components.py`, CPU, fast)

- Legality: `‖(I − Q_in Q_inᵀ) V‖ ≈ 0`, `‖U (I − Q_out Q_outᵀ)‖ ≈ 0`.
- Forward equivalence at τ = 0 on a random full-rank `W`: `SVDLinearComponents` with
  `A = Q_inᵀ V₀`, `B = U₀ Q_out` matches dense forward with `V₀`, `U₀` (fp32 tolerance).
- `get_component_acts(x) == x @ V` (property path vs fused path).
- Rank truncation: `components.weight` stays in the rank-r space; the delta picks up
  the tail exactly.
- `scale_subcomponents_` equals dense column/row scaling in effective space.
- Grads reach `A`/`B` only; `Q` buffers unchanged after an optimizer step.
- `Trainer.snapshot` / `from_snapshot` round-trip with SVD components.

## Experiments

Reference: `~/out/runs/addsub-L18-04-hidden` (L18 MLP C=456×3 + attn q/k C=72,
v/o C=128; 24 000 steps; dp=2; battery results already on disk). Battery =
`collect_filtered_kl.py`, blocks `mlp` + `attn`, subset `active`, ops add+sub —
identical settings to the reference so numbers are directly comparable; **raw** is the
criterion, span-rank stats co-reported (watch rank inflation).

- **E0 — spectrum scan** (1 GPU, minutes). SVD spectra of the 7 L18 matrices →
  numerical ranks and the τ grid for E3 (aim: r/r_full ≈ 0.5 and ≈ 0.1, adjusted to
  where the spectra knee). Script
  `param_decomp_lab/scripts/validation/subspace_restriction/scan_spectra.py`;
  artifacts in `~/pd_scratch/subspace_restriction/spectra/`.
- **E1 — retrofit, no training** (1 GPU, ~2 h). Copy the reference run dir →
  `addsub-L18-04-hidden-Aretro`; project the checkpoint in place
  (`V ← Q_in Q_inᵀ V` where `row(W)` is deficient, `U ← U Q_out Q_outᵀ` where
  `col(W)` is; delta absorbs the residue automatically); re-run the battery.
  Isolates the purely structural effect: (b) should collapse to baseline; whatever
  happens to (a) is the no-retraining floor for Q2.
- **E2 — main A/B** (2×L40, ~22 h). Exact reference recipe + `svd_rank_threshold: 0.0`,
  same seed/steps. Label `addsub-L18-05-svd-tau0`. Memory-probe first (steps=3, no
  wandb). Evals: training parity (TargetReconLoss, NontargetReconLoss, NAlive, CI
  heatmaps) + full battery vs reference and vs E1.
- **E3 — threshold sweep** (2 runs, after E2 trains cleanly). τ values from E0;
  labels `addsub-L18-05-svd-tau<τ>`. Same evals. Answers Q1: where recon and the
  battery degrade (or improve) as ranks are cut.

Compute: ≤6 GPUs at once (job 330 may hold one) → E2 alone first, then the E3 pair
concurrently; battery jobs are single-GPU. Run configs and sbatch files in
`~/pd_scratch/subspace_restriction/`.

## Success criteria

- Sanity (b): `circuit_in_row` / `circuit_out_col` raw ≈ `circuit_baseline`
  (wiring-check level) for E1, E2, E3.
- Primary (a): `orig_in_span` / `orig_out_span` raw KL vs reference, at comparable
  TargetReconLoss / NontargetReconLoss / n_alive and without span-rank inflation
  (ranks within ~1.5× reference).
- Q1: recon-loss curves and battery numbers vs τ; "all ranks works" = E2 matches
  reference parity metrics within noise.

## Risks / notes

- Checkpoint grows ~1 GB (persistent fp32 `Q` buffers); acceptable,
  `keep_last_n_checkpoints: 2`.
- Removed cancellation slack may slow optimization — compare the first ~2k steps of
  `metrics.jsonl` against the reference before letting E2 run out.
- Q buffers live on GPU (~1 GB/rank for the MLP trio) — the steps=3 memory probe
  gates the launch.
