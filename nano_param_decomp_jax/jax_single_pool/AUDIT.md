# Audit: `jax_single_pool` llama8b path vs SPEC.md

Working document for the 2026-06-10 refinement pass. Each row cites the violated
invariant. Status: ☐ open · ☑ fixed · ✗ won't-fix (justified).

**2026-06-10 update: A1–A15, A17, A18 fixed by the generic-trainer restructure**
(`lm.py` + `train.py` + the llama8b `DecomposedLM` adapter; scaffold deleted). The
fixes are pinned by tests: equivalence harness (numeric, per-term), `test_llama8b.py`
(S2/S3/S9/S13/S15/N1 assertions), `test_checkpoint.py` (S22), `invariance_check.py`
(D4). A16 (a real data pipeline: fresh batches + per-batch prefix harvest, wandb/sink,
SLURM entry) is the remaining gap between "bench" and "trainer".

## Violations

| # | spec | severity | where | finding | status |
|---|---|---|---|---|---|
| A1 | S13 | **high** | `llama8b_step.py` warmup/final ascent | Source updates are plain ascent `src += lr·g` — no Adam(β₁.5, β₂.99), no persistent moments, no per-training-step LR schedule (const 0.01 w/ 2.5% warmup). | ☑ |
| A2 | S13/S15 | **high** | `llama8b_step.py:328-332` | No projection between warmup iterates: the `lax.scan` clamps only AFTER all `n_warmup` ascents; raw sources can leave `[0,1]` mid-walk (and the in-use `clip` zeroes their grads — drifted entries die). Torch projects after EVERY step. | ☑ |
| A3 | S15 | **high** | `llama8b_real.py:198,204` | Sources init to **zeros**; spec/torch init `U[0,1]` i.i.d. (clamp parameterization). Zeros = adversary starts fully-off and must walk up. | ☑ |
| A4 | S14 | **high** | `llama8b_step.py:360-368` | Final ascent gradient computed via an EXTRA forward at the **post-update** params (`new_vu_det`); spec/torch take it from the same graph as the main backward (pre-update θ, live ci), no extra forward. | ☑ |
| A5 | S3 | **high** | `llama8b_step.py:318` | Clean target computed through the decomposed identity (`masks=None, delta=1` ⇒ `V@U + (W−V@U)`), not the frozen `x @ W` path — bf16 rounding noise in every recon target and V/U needlessly in the (stopped) graph. `_clean_mlp_out`/`decompose_layer` exists but is not used here. | ☑ |
| A6 | S9 | **high** | `llama8b_real.py:216` | `p_imp` static at 0.4 (the FINAL annealed value); spec anneals 2.0 → 0.4 linearly over training. Early-training sparsity pressure is materially different. | ☑ |
| A7 | S20 | **high** | `llama8b_real.py:188-189` | `optax.adamw(1.5e-4)` — optax's default `weight_decay=1e-4` silently applied where torch uses **0.0**; no cosine-to-0.1× LR schedule on either optimizer. | ☑ |
| A8 | S19 | **high** | `llama8b_real.py` | No grad clip. Torch clips V/U global-norm at **0.01** (and leaves CI fn unclipped) — at these LRs that clip is load-bearing. | ☑ |
| A9 | S21 | medium | runner | No faithfulness warmup (400 × AdamW lr 1e-3 on `L_faith` alone before step 0). | ☑ |
| A10 | S22 | **high** | (missing) | No checkpoint/resume for `Llama8BState`. `checkpoint.py` is typed against the scaffold's `TrainState` only. | ☑ |
| A11 | §4.6 | medium | `ci_fn.py:115` | Extra `relu` after the input projection — torch has NO nonlinearity there. | ☑ |
| A12 | §4.6 | medium | `ci_fn.py` | Missing biases: torch's in-proj / out-head / block-MLP linears carry zero-init biases; JAX has none. | ☑ |
| A13 | §4.6 | low | `ci_fn.py:69-70` | Block norms `ln1/ln2` are LEARNABLE arrays; torch block norms are weightless `F.rms_norm`. (Site-input norms are correctly weightless.) | ☑ |
| A14 | N1/N2 | **high** | `llama8b.py` (`DT`), runner | V/U + CI params and Adam moments are bf16; spec: fp32 masters + fp32 moments, bf16 compute casts. Also makes faith deltas bf16-computed (N2). | ☑ |
| A15 | S8/D2 | medium | `llama8b_step.py::make_llama8b_step_shmap` | The `--shmap` variant `pmean`s the per-shard imp-min AFTER `log2` — Jensen-biased vs the global sum. (`--shard` GSPMD variant is correct: global-batch reduction inside jit.) | ☑ |
| A16 | S18 | medium | runner | Bench loops one fixed batch; no data pipeline. Fine as a bench; a training run needs fresh batches + per-batch prefix harvest. | ☐ |
| A17 | S5 docstring | low | `pgd.py:11-13` (scaffold) | Scaffold docs claim sigmoid-parameterization "matches production" — production is `use_sigmoid_parameterization: false` (clamp). Dies with the scaffold. | ☑ |
| A18 | D4 | medium | (missing) | GPU-count-invariance harness exists only for the scaffold (`distributed_stacked_sites`); needs re-homing onto the llama8b step before scaffold deletion. | ☑ |

## Compliant (verified, no action)

- S1 mask recipe incl. raw delta channel (`_ppgd_masks_and_deltas`, `_stoch_one_chunk`).
- S2 non-live sites run frozen path in stoch chunks (`decompose_layer` + `_clean_mlp_out`).
- S4 CI inputs are clean site inputs (`all_site_inputs` threads the W path).
- S5/S6 two squashings of one logits tensor; `lower_leaky_hard` custom VJP matches torch's
  grad-sign-gated leak; `upper_leaky_hard` plain autodiff of the torch expression.
- S7 imp-min per-(layer,kind) grouping (`_imp_min` keeps `(L, C)` sums; log2 per site).
- S8 in `--shard` mode (global-batch sums inside jit; see A15 for shmap).
- S10 stoch normalization `Σ/(n_chunks·n_samples)`, chunk = one layer's 3 sites.
- S11 uniform-k routing — argsort-permutation construction is distributionally identical
  to torch's double-argsort ranks.
- S12 PPGD: all sites live, route-everywhere, source detached in the param loss.
- S16 shared source under GSPMD: replication + global-mean grad ≡ torch's broadcast +
  AVG-reduce.
- S17 faith = Σ‖Δ‖²/Σnumel over all sites, fp32 accumulation.
- N3 KL in fp32; imp-min reduction in fp32.
- R1/R3 `fold_in`-derived independent draws (bf16 uniform draws match torch-under-autocast).

## Notes (2026-06-11 site-generality restructure)

- The Llama target now accepts ARBITRARY per-layer matrix sites (q/k/v/o/gate/up/down,
  per-site C) instead of contiguous-range MLP-only: per-site `DecompVU` dict replaces
  the six stacked `(L,·,·)` arrays; `Target` is a uniform `SuffixLayer` list (layers
  with no sites run the plain frozen block, SPEC S2). Parity with the stacked
  implementation is pinned by `tests/stacked_parity/` (fixtures generated on
  `feature/jax-single-pool-pd`): clean/masked/site-input forwards bit-identical, the
  2-step train trajectory within rel ~2.5e-6 (clip-global-norm leaf-order
  reassociation is the only divergence source).
- Checkpoints are NOT cross-compatible across the restructure: the `components` pytree
  layout and the V/U init RNG derivation changed (per-site keys instead of 6 stacked
  draws). Old stacked checkpoints would need a one-off destack migration to resume.
- `verify_export_torch.py`'s "production numerics" CI-fn pass is now measure-only
  (never asserted): the documented GELU/eps divergence is amplified on the tiny
  attention fixture (leaky-hard outputs near the clamp boundary → max_rel ~0.18),
  while the jax-matched-numerics pass — the actual mapping proof — stays asserted at
  fp32 tolerance and passes for all three cases (incl. `l18_attn`, heterogeneous C).

## Notes

- Site ordering: torch concatenates CI inputs in sorted-module-path order (per layer:
  down, gate, up); JAX uses (gate, up, down). Equivalent for fresh init (a fixed
  permutation of `in_proj` rows / head columns); matters only for cross-framework weight
  round-trips — **handled in the exporter** (`export.py::ci_fn_state` permutes the
  in-proj row blocks and out-head column blocks to `sorted(site_names)`; proven by the
  `tools/gen_export_fixture.py` → `tools/verify_export_torch.py` round-trip on both a
  single-layer and a two-layer shape). Documented, not a violation.
- CI-fn numerics vs torch (surfaced by the export round-trip, harmless for training but
  visible when the torch stack evaluates an exported checkpoint): JAX `jax.nn.gelu`
  defaults to the TANH approximation where torch's `TransformerBlock` uses exact-erf
  `nn.GELU()` (max pointwise gap ~4.7e-4), and JAX's weightless rms-norms use eps `1e-5`
  where torch's `F.rms_norm` defaults to `finfo(fp32).eps` ≈ 1.19e-7. With both choices
  matched the round-trip agrees to ≤4e-5 max rel; with production torch numerics the
  tiny-fixture CI outputs drift up to ~3e-2 rel (production-width activations sit far
  from the eps floor, so the real-model drift is much smaller). Documented, not fixed —
  changing `ci_fn.py` would perturb live-run trajectories.
- Init distributions: torch `init_param_` vs JAX `normal·fan⁻¹ᐟ²` differ in family;
  spec pins fan-in scaling only. Check `init_param_` when touching A11–A13.
- The equivalence harness (`tests/equivalence`) is fixture-driven (no RNG, zeroed attn,
  fp32) — semantic fixes above should keep it green; A5 tightens it.
