# CI-network saturation — lab notebook

Working question: why does the transformer CI network (`global_shared_transformer`)
produce almost exclusively {0, 1} CI values, while the MLP CI networks leave
intermediate values — and can we fix the transformer so gamma annealing can prune
strays the same way?

Testbed: `copy_training_v8d32_partial` (attention copy toy, vocab 8, d_embed 32,
4 redundant tokens). Reference comparison from the completeness work:

| run | CI net | notes |
|---|---|---|
| `copy_v8d32_partial_joint_norm` | layerwise `mlp`, hidden [16] | the good one |
| `copy_v8d32_partial_joint_globalci` | `global_shared_mlp` [256, 256] | |
| `copy_v8d32_partial_joint_txci` | `global_shared_transformer` d64, 2 blocks, 4 heads | many strays |
| `copy_v8d32_partial_joint_txci_beta1` | same, beta 1.0 | worse (L0 12.9) |

All use `sigmoid_type: leaky_hard` (verified in both experiment_config.yaml — forward
is exactly `clamp(x, 0, 1)`; impmin sees the upper-leaky variant, alpha 0.01 above 1).

## 2026-07-23 — diagnosis

Final-step eval metrics (`metrics.jsonl`, step 10000):

| run | total CI-L0 | smoothL0 (no beta) | stoch recon KL |
|---|---|---|---|
| joint_norm | 3.50 | 3.52 | 1.1e-5 |
| joint_globalci | 4.49 | 4.49 | 3.5e-5 |
| joint_txci | 5.25 | 5.27 | 1.7e-5 |
| joint_txci_beta1 | 12.92 | 12.92 | 6.5e-5 |

CI histograms at step 5000 (before the gamma anneal starts at 7500): `_norm` has
substantial intermediate mass (clusters at 0.05–0.6 in most matrices); `_txci` is
already near-binary. The anneal then sweeps `_norm`'s intermediates to 0; `_txci`'s
strays sit at exactly 1 and survive. Final `active_subcomponents.png`: `_norm` textbook
(1 q, 1 k, clean diagonals), `_txci` has extra q/k routing, duplicate v components,
stray o components in all blocks.

**Smoking gun — pre-sigmoid logit histograms at step 10000:**

- `_norm` (layerwise MLP): logits in ≈ [-1.5, +2.6], hugging the hard sigmoid's
  [0, 1] linear window. A component at CI=1 sits at logit ≈ 1.0–1.3.
- `_txci` (transformer): logits span ≈ **[-80, +150]**. Off-components at -20…-60,
  on-components at +10…+150.

Mechanism: the transformer's output head reads an *unnormalized* residual stream
(no final norm — norm grows across blocks and freely during training), CI-fn weight
decay is 0, and nothing anchors logit scale. Deep-saturated components see only the
0.01-leaky impmin gradient over a ~100-unit round trip back to the linear window →
strays at CI=1 are effectively frozen; the gamma anneal can't reach them. Intermediate
CIs would require logits landing in [0, 1] — ~1% of the network's dynamic range —
so almost none exist. The layerwise MLP (scalar input, hidden 16) can't blow up its
logit scale, which is why its strays linger at intermediate CI where the anneal
kills them.

Secondary hypothesis (not load-bearing given the above): a more expressive network
also *wants* to binarize — confident CIs reduce stochastic-masking loss variance.
The fix below doesn't fight that; it only keeps the saturation shallow enough that
sparsity pressure still works.

## Fix (param_decomp/ci_fns.py)

Three standard-transformer changes to `GlobalSharedTransformerCiFn`:

1. **Final RMS norm** before the output head (`final_rms_norm: true` in
   `GlobalSharedTransformerCiConfig`) — decouples logit scale from residual growth.
2. **Readout zero-init, bias 0.5** (unconditional) — every logit starts at exactly
   0.5, mid-window, in gradient reach of both losses from step 0.
3. **Logit soft-cap** (`logit_softcap: 2.0`) — Gemma-style
   `0.5 + cap·tanh((x-0.5)/cap)`, bounds logits to (-1.5, 2.5) so they can never
   leave gradient reach.

## Experiment: three variants on v8d32_partial

Configs in `~/pd_scratch/combine_layers/configs/`, launched via
`txci_variant.sbatch` (jobs 5431–5433), run ids
`toy_model_redundancy/copy_v8d32_partial_joint_txci_{newinit,fnorm,fnorm_cap2}`:

- `txci_newinit` — baseline config, new code → isolates the head-init change.
- `txci_fnorm` — + `final_rms_norm: true`.
- `txci_fnorm_cap2` — + `logit_softcap: 2.0`.

Success criterion: closer to `_norm` on final CI-L0 / alive counts / stray stats
(`txci_stats.py`, extended with per-module + total alive counts) without recon
regression.

## 2026-07-23 — results

All three finished (~20 min each). Final metrics + cleanliness stats (job 5435):

| run | CI-L0 | recon KL | v/o dupes | shared subs | skipped mechs |
|---|---|---|---|---|---|
| `joint_norm` (reference) | 3.50 | 1.1e-5 | 1 | 0 | 10/19 |
| `joint_txci` (baseline) | 5.25 | 1.7e-5 | 7 | 9 | 4/19 |
| `txci_newinit` (init only) | 5.56 | 5.8e-5 | 11 | 10 | 5/19 |
| `txci_fnorm` | **3.43** | 2.1e-5 | 3 | 1 | 10/19 |
| `txci_fnorm_cap2` | **3.24** | 1.9e-5 | 2 | 0 | 10/19 |

- **The final RMS norm is the load-bearing change.** Init alone does nothing
  (arguably worse). fnorm alone recovers the `_norm`-level result; the soft-cap
  shaves a bit more (block-0 v and o both perfectly clean: 1.00/input, 0 dupes,
  0 shared — matching `_norm` exactly).
- Mechanism confirmed end-to-end: fnorm's final pre-sigmoid logits live in
  ≈ [-5, +3.5] (vs [-80, +150] baseline); its step-5000 CI histograms show
  abundant intermediate mass which the gamma anneal then prunes — same dynamics
  as the MLP run.
- Both fnorm variants skip 10/19 ground-truth mechanisms, exactly like `_norm` —
  the baseline txci's 4/19 "coverage" was strays incidentally covering redundant
  mechanisms, not real completeness.
- Residual dirt vs `_norm`: `fnorm_cap2` keeps 3 alive in blocks.0.k (vs 1) and
  one dupe in each of blocks.1.{v,o}. Small; possibly seed noise (n=1 per variant).

Recommendation: `final_rms_norm: true` + `logit_softcap: 2.0` for all future
`global_shared_transformer` runs. Old txci checkpoints must be evaluated with
`final_rms_norm: false` (their saved configs omit the field, defaulting to the
old behavior, so reloads are unaffected).

## 2026-07-23 — larger sizes: {v12d64, v16d32} × {fnorm, cap2, fnorm_cap2}

Array job 5456 (`txci_large_array.sbatch`, %3 throttle to respect the 6-GPU cap).

### v12d64 (final eval, step 10000)

| run | CI-L0 | recon KL |
|---|---|---|
| layerwise MLP (`joint_norm`) | 4.93 | 2.1e-5 |
| global MLP (`joint_globalci`) | 5.68 | 2.9e-5 |
| transformer baseline (`joint_txci`) | 7.13 | 2.1e-5 |
| `txci_fnorm` | **4.01** | 3.0e-5 |
| `txci_fnorm_cap2` | 4.27 | 4.4e-5 |
| `txci_cap2` (cap only) | 9.19 @9500 | 3.7e-5, then **crashed** |

- Both norm variants beat the layerwise reference again. fnorm alone edges out
  fnorm+cap here (reverse of v8; within seed noise presumably).
- **Cap-only is unstable.** The run trained to 10000 but died in the final eval:
  `np.histogram` hit non-finite pre-sigmoid values. Mechanism: with every logit
  saturating the tanh, CI-fn grad norms collapse to ~1e-8, but Adam's normalized
  updates keep drifting the (unnormed) trunk at ~lr per step until activations
  overflow — the cap alone removes both the gradient pressure *and* the implicit
  constraint on trunk scale. SmoothL0 at the end: 46 (vs ~4 for the norm variants) —
  the CI net froze mid-training and sparsity never happened. No final checkpoint
  (crash preceded the save), so no plot_ci analysis; metrics.jsonl survives.
- Conclusion so far: the final RMS norm is necessary; the cap is only safe on top
  of it.

### v16d32 (final eval, step 10000) — dense collapse repaired

| run | CI-L0 | recon KL | skipped |
|---|---|---|---|
| layerwise MLP | 6.53 | 4.5e-5 | 18/31 |
| global MLP | 6.43 | 2.1e-4 | 9/31 |
| transformer baseline | 19.17 | 7.0e-4 | 3/31 |
| `txci_fnorm` | 5.52 | 2.1e-4 | 17/31 |
| `txci_cap2` | 13.15 | 7.9e-5 | 8/31 |
| `txci_fnorm_cap2` | **5.37** | **7.0e-5** | 16/31 |

- The completeness report's "at v16 the transformer CI collapses to dense" is a
  *logit-scale* pathology, not an expressivity limit: fnorm+cap2 is
  simultaneously the sparsest run at this size AND 10× more faithful than the
  baseline transformer, beating both MLP references on L0 at comparable-or-better
  recon.
- At v16 the cap earns its keep on top of the norm: 3× recon improvement over
  fnorm alone (7.0e-5 vs 2.1e-4) at slightly better L0.
- Cap-only stayed numerically stable at this size (unlike v12) but is still bad
  (L0 13.15) — consistent with "bounds saturation depth but leaves the trunk
  unanchored".
- Its low skip count (8/31, like globalci's 9) comes with L0 13 — dense coverage
  again, not clean recovery.

## 2026-07-23 — WD-only control (v8d32) + 3-seed rigorous sweep

- `txci_wd1e-2` / `txci_wd1e-1` (array 5462): baseline transformer + CI-fn Adam
  weight decay only — does generic logit-scale control suffice, or does the
  norm+cap mechanism matter?
- Seed sweep (array 5464, 25 runs): {txci, fnorm, cap2, fnorm_cap2} ×
  {v8d32, v12d64, v16d32} × seeds {0, 1, 2}, hyperparameters unchanged from
  the exploratory runs (they produced good decompositions at every size —
  no tuning needed).

### Seed-sweep results (final eval; mean [min–max] over 3 seeds)

| variant | v8d32 L0 | v12d64 L0 | v16d32 L0 |
|---|---|---|---|
| layerwise MLP (s0 ref) | 3.50 | 4.93 | 6.53 |
| txci baseline | 4.90 [4.62–5.25] | 9.52 [7.10–14.34] | 17.44 [13.48–19.66] |
| txci_fnorm | **3.29 [3.19–3.43]** | **4.22 [4.01–4.39]** | **5.52 [5.37–5.66]** |
| txci_cap2 | 7.08 [5.55–8.56], 1 crash | 9.33, 2 crashes | 13.76 [13.15–14.79] |
| txci_fnorm_cap2 | 3.66 [3.24–3.87] | 4.41 [4.27–4.66] | 5.73 [5.37–6.04] |

- **fnorm is the robust winner**: tightest ranges, beats the layerwise reference
  and the baseline at every size on every seed. Recon: v8 2.2e-5 [1.6–2.8],
  v12 2.7e-5 [2.4–3.0], v16 1.2e-4 [5.9e-5–2.1e-4].
- **The cap does not reliably add anything on top of the norm** across seeds
  (slightly worse mean L0 at v8/v12); the seed-0 fnorm_cap2 wins at v8/v16 were
  seed luck.
- **cap2-only is confirmed unstable**: 3 crashes in 9 seeds (v8 s0, v12 s0+s1 —
  all with the histogram-on-nonfinite signature), bad L0 when it survives.
- **WD control (v8)**: baseline + CI-fn Adam WD 0.01 → L0 5.32 (≈ baseline's
  4.90-mean); WD 0.1 → 6.83 (worse). Generic weight-space shrinkage does not fix
  the saturation — the final-norm mechanism specifically does.

### Cleanliness at v12/v16 (job 5491) → tuning grid

Per-input multiplicity (dupes) and cross-input sharing for the fnorm winner:

- v12 fnorm: b0.v/o ≈ 1.17/input (1–2 dupes each, 1 shared) — mild coeff issue.
  (Layerwise ref is *worse* here: b0.v 2.25/input, 12 dupes.)
- v16 fnorm: b0.v 2.25/input, 12 dupes, 11 shared; b0.o 1.62/input — both
  pathologies (dupes → impmin coeff too low; shared subs → beta too low).

Tuning grid on fnorm (array 5492): coeff 2e-4 at v8; {coeff 2e-4, beta 1.0,
both} at v12 and v16.

## 2026-07-23 — readout-init audit

Which runs used which output-head init (the zero-init is unconditional in the new
code, so it rode along with every new run):

- **old fan-in random init (bias 0)**: all baseline txci *seed-0* runs (predate the
  code change) and every historical txci run.
- **new zero-init (bias 0.5)**: `txci_newinit`, all fnorm / cap2 / fnorm_cap2 runs,
  the tuning grid, the WD controls, **and the baseline txci s1/s2 seed-sweep runs**
  — i.e. the baseline 3-seed row mixes inits (s0 old, s1/s2 new).

Evidence so far that the init is not the driver: init-alone at v8 ≈ baseline
(5.56 vs 5.25), and the mixed-init baseline seeds show no init-correlated split
(old-init s0 is the *worst* v8 baseline seed). But `fnorm × old-init` was never
run — the measured "fnorm effect" is strictly "fnorm + zero-init".

Closing the gap: `zero_init_readout: bool = True` config field added
(`false` restores fan-in init, bias 0); array 5546 runs
`fnorm_randinit` × 3 sizes × 3 seeds after the tuning grid drains.

## 2026-07-23 — tuning round 1 results (array 5492)

Heuristics: dupes → impmin coeff up; shared subs → beta up. All on the fnorm base:

| run | L0 | recon | b0.v (dupes, shared) | skips |
|---|---|---|---|---|
| v8 fnorm (base) | 3.43 | 2.1e-5 | 1.00 (0, 0) | 10/19 |
| v8 fnorm_i2e-4 | **3.06** | 2.9e-5 | 0.88 (0, 0), 1 zero | 10/19 |
| v12 fnorm (base) | 4.01 | 3.0e-5 | 1.17 (1, 1) | 13/28 |
| v12 fnorm_i2e-4 | **3.88** | 3.8e-5 | 1.17 (1, 2) | 15/28 |
| v12 fnorm_b1 | 3.88 | 3.5e-5 | 1.17 (1, 1) | 14/28 |
| v12 fnorm_i2e-4_b1 | 3.34 | 8.2e-5 | 1.08 (1, 1) | 10/28 |
| v16 fnorm (base) | 5.52 | 2.1e-4 | 2.25 (12, 11) | 17/31 |
| v16 fnorm_i2e-4 | **4.47** | **8.6e-5** | 1.75 (10, 4) | 18/31 |
| v16 fnorm_b1 | 5.18 | 1.5e-4 | 1.75 (9, 4) | 16/31 |
| v16 fnorm_i2e-4_b1 | 4.10 | 1.9e-4 | 1.69 (8, 5) | 18/31 |

- **coeff 2e-4 is the new operating point at every size** — L0 down, cleanliness
  up, recon within band; at v16 it *improves* recon 2.5× while sparsifying (the
  base run was spending capacity keeping strays half-reconstructed).
- **beta 1.0 not adopted**: alone it's dominated by i2e-4; combined it trades
  recon for L0 at v12 (8.2e-5) without cleaning b0 further at v16. Notably the
  v16 shared-subs problem (11 → 4) was fixed by the coeff, not beta.
- v8 i2e-4 shows the first over-pruning hint: one b0.v/o token column zeroed
  (covered by another block; skips unchanged).

Round 2 (array 5553, after randinit): coeff 4e-4 probes at v12/v16 (v16 b0.v
still has 10 dupes) + 3-seed confirmation of fnorm_i2e-4 at all sizes.

Also probing (array 5555, after round 2): CI-net LR halved to 1e-3 (components
stay 2e-3) on fnorm_i2e-4 at all sizes — LR was an untouched axis; the targeted
recipe found CI-LR below components-LR gives cleaner components. All runs so far
used 2e-3 for both.

## 2026-07-23 — v16d32 retired, v8d64 added

Decision: drop v16d32 entirely — all `copy_v16d32_*` runs and the
`copy_training_v16d32_partial` target deleted from disk, its pending probe jobs
cancelled, and the report rewritten without it. (Notebook entries above are kept
as history; their v16 numbers refer to now-deleted runs.) Seed confirmations of
fnorm_i2e-4 also cancelled — goal narrowed to "find something that works".

Replacement: **v8d64** (vocab 8, d_embed 64, 4 redundant tokens) — the
wide-embedding version of the canonical testbed. Train the target, then apply the
current best formulas (fnorm_i2e-4, ± CI-LR/2 pending the probe results).

## 2026-07-23 — cleanup: failed/bad runs, obsolete toys, soft-cap and WD removed

Disk: deleted all non-copy toy runs (cipher `attn_blockdrop_*`, the original MLP
redundancy toy's `training*` / `*ppgd_lateanneal*` / non-copy `*ppgdla*` families),
all failed runs, and every clearly-bad/dead-end run: cap-only and fnorm_cap2
families (soft-cap inconclusive → removed entirely), `txci_beta1`, `txci_newinit`,
the WD controls, and the smoke test. Notebook entries above referring to these
describe now-deleted data.

Code: `logit_softcap` removed from `param_decomp/ci_fns.py` (saved run configs on
disk migrated by stripping the key); obsolete toy modules
`toy_models/toy_model_redundancy.py` and `toy_model_redundancy_attn.py` deleted —
`RMSNorm`/`AttnBlock` now live in `toy_model_redundancy_copy.py`; `run.py` and
`ToyRedundancyCIPlot` are copy-only; the in-repo example YAML now runs the current
best recipe (fnorm, coeff 2e-4) against `copy_training_v8d32_partial`. Obsolete
scratch scripts/configs (cipher/attn/cap/WD/newinit, `toy_redundant.sbatch`)
deleted. `make check` + full lab tests pass; pending arrays (cilr, v8d64) validated
against the new code.

## 2026-07-24 — readout-init attribution result (array 5546, v8/v12 only)

fnorm × old random readout init, 3 seeds, L0 mean [min–max]:

| | zero-init (bias 0.5) | random init (bias 0) |
|---|---|---|
| v8d32 | 3.29 [3.19–3.43] | **4.79 [4.12–6.07]**, recon 3.6–7.3e-5 |
| v12d64 | 4.22 [4.01–4.39] | 4.42 [4.14–4.60] |

**The zero-init is load-bearing given the norm** (v8: random init loses nearly the
whole fnorm advantage — 4.79 vs baseline 4.90). Earlier "init alone does nothing"
(newinit ≈ baseline) still holds; the two changes are synergistic: norm anchors the
logit scale, zero-init starts every logit inside the sigmoid window so components
begin life prunable. Neither suffices alone at v8. Recipe = both.

## 2026-07-23 — cleanup round 2: complete runs + fully-redundant family

- Completeness runs: kept only `copy_v8d32_complete_06-imp2e-4` (best: 0/19
  skipped, cleanest map); deleted complete_{01,02,03,04,05} and the smoke run.
  `notes/completeness/report.md` updated — robustness table condensed to prose,
  run 06 is the single kept exemplar.
- Fully-redundant v8d32 family: kept only `copy_v8d32_imp1e-4_beta0.5` (textbook:
  full clean v/o diagonals in the first block, single q/k routing columns, zero
  strays); deleted the rest of the imp sweep, `copy_v8d32_ppgdla_10k`, and the
  non-partial perblock_b{0,1,2} runs.

## 2026-07-24 — sweep closed: v8d64 + round-2 + CI-LR results

- v8d64 (new target, trained job 5558): formula transfers untouched.
  i2e-4 → L0 3.12 @ 8.5e-6, zero dupes anywhere, textbook. Coeff bracket
  {1e-4: 3.74 @ 6.1e-6, 1.5e-4: 3.25, 2e-4: 3.12} — smooth, no cliff.
- v12 i4e-4 over the ceiling (3.25 @ 8.7e-5, b0 zeros) — 2e-4 stands.
- CI-LR/2: v8d32 win (3.00 @ 1.65e-5, b0 perfect incl. coverage — repairs
  i2e-4's over-pruning); v8d64 3.31 (neutral-negative); v12 3.89 @ 5.6e-5
  dirtier (negative). Per-size knob only.
- Final picks recorded in report.md; next: completeness treatment on the three
  picks (impmin 2e-4 to match, otherwise complete.py defaults).

## 2026-07-24 — completeness treatment on the final picks (jobs 5602–5604)

`complete.py` (two-CI resurrection, impmin 2e-4 to match, otherwise defaults) on
each size's final decomposition. Skipped mechanisms, normal → complete net:

- v8d32: 13/19 → **0/19**
- v8d64: 13/21 → **0/21**
- v12d64: 16/28 → **0/28**

Zero skips at every size, and the complete maps are *structured*: block-0 keeps
full diagonals, blocks 1–2 regrow dedicated backup diagonals for the redundant
tokens plus their own q/k routing — not the dense coverage the old transformer
bought skips with. First time the resurrection protocol is demonstrated beyond
v8 and on transformer-CI decompositions.
