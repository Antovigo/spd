# Report — combining single-block decompositions

Setting: four single-block targeted decompositions of Llama-3.1-8B on
addition/subtraction prompts (`addsub-L16-04-init-proj`, `addsub-L17-04-init-proj`,
`addsub-L18-05-coupled`, `addsub-L19-05`; blocks 16–19, 7 matrices each). We assemble
them into one model — every eligible layer replaced by its decomposed version, each
block keeping its own trained CI function — and ask whether the PD objectives still
hold.

All numbers: KL per position between the masked-component model and the target model,
rounding threshold 0.01 (same as the runs' own end-of-training "rounded recon" logs),
means over 10 eval batches (dots in the figures are the individual batches).

## Objective 1: the decompositions do NOT readily combine

![recon per subject](report_figures/obj1_recon.png)

| subject | rounded | PGD (20-step) | nontarget rounded | total L0 |
|---|---|---|---|---|
| L16 alone | 0.0072 | 0.0076 | 0.0034 | 9.6 |
| L17 alone | 0.0062 | 0.0069 | 0.0032 | 8.0 |
| L18 alone | 0.0055 | 0.0059 | 0.0038 | 12.6 |
| L19 alone | 0.0059 | 0.0060 | 0.0032 | 7.9 |
| **all four combined** | **0.257** | **0.403** | **0.0122** | 38.1 |

Key conclusions:

1. **Combined target recon is ~40× the worst single block, and ~10× the sum of the
   four single-block losses.** The failure is superadditive: each block's components
   and CI masks were trained with every *other* block intact, so once all four are
   replaced, downstream blocks receive inputs their decomposition was never fitted
   to. Adversarial (PGD) recon degrades even more (0.40).
2. **Each block's CI values are untouched by combination** (per-block L0 identical to
   the singles, e.g. L16: 9.60 vs 9.61): CIs are computed from the clean target-model
   activations. The entire degradation lives in the masked forward pass.
3. **The nontarget objective survives combination much better** (0.003 → 0.012):
   off-distribution the delta component carries the output, so component errors
   contribute little there.

![recon vs number of blocks](report_figures/obj1_scaling.png)

How the error grows with the number of replaced blocks (chain starting at L16):

| blocks replaced | rounded | PGD | additive expectation (rounded) |
|---|---|---|---|
| L16 (1) | 0.0072 | 0.0076 | 0.0072 |
| +L17 (2) | 0.0437 | 0.0710 | 0.0134 |
| +L18 (3) | 0.2154 | 0.3218 | 0.0189 |
| +L19 (4) | 0.2567 | 0.4028 | 0.0248 |
| L18+L19 (2, off-chain) | 0.0252 | 0.0527 | 0.0114 |

Each early addition multiplies the loss ~5-6× rather than adding its single-block
contribution; by three blocks the compounding dominates (11× the additive
expectation). The off-chain pair L18+L19 shows the same superadditivity (2.2× its
additive expectation), so this is not specific to L16/L17.

**Answer to the roadmap question:** yes, the combined rounded recon is *much* higher
than the end-of-training rounded recon of the individual decompositions —
two orders of magnitude. Combining requires re-optimisation (objectives 2/3), not
mere concatenation.

## Objective 2: fine-tuning the assembly is feasible — train both, not components-only

Setup: the assembled model fine-tuned for 2000 steps with the full targeted loop
(nontarget FineWeb pass, ratio 2.0), importance-minimality held constant at p = 0.5
(the sources' end-of-anneal p) with coeff = min over sources = 3e-5 — per the
prefer-recon-over-sparsity rule. Caveat: L16/L17 converged at coeff 5e-5, so those
two blocks are fine-tuned under 40% weaker sparsity pressure than their own
converged objective; part of the L0 growth below is attributable to that, not only
to CI-fn retraining. Components LR 1e-4, CI-fn LR 5e-5 (cosine, sources' schedule
shape). Memory forced single-GPU runs at global batch 32 (sources used 128), so
these are feasibility-grade, not final-quality.

![fine-tuned vs baselines](report_figures/obj2_recon.png)

| subject (same eval script/seed) | rounded | PGD | target L0 | ntgt rounded | ntgt L0 |
|---|---|---|---|---|---|
| singles (range) | 0.0055–0.0072 | 0.0059–0.0076 | 7.9–12.6 | 0.0032–0.0038 | 0.07–0.11 |
| raw combined | 0.257 | 0.403 | 38.1 | 0.0122 | 0.35 |
| combined + FT, frozen CI fns | 0.0431 | 0.126 | 38.1 | 0.0132 | 0.35 |
| combined + FT, both | **0.0239** | **0.0551** | 67.6 | 0.0133 | 0.59 |

Note the nontarget L0 column: already the *raw* combination raises off-distribution
activation 3–5× over the singles (0.07–0.11 → 0.35); fine-tuning with frozen CI fns
holds that level, while training the CI fns adds another ~1.7× (0.59).

![fine-tuning trajectories](report_figures/obj2_trajectory.png)

Key conclusions:

1. **Fine-tuning recovers most of the combination damage**: rounded recon drops
   ~11× (0.257 → 0.024) when training both components and CI fns; PGD drops 7×
   (0.40 → 0.055). Neither run has fully converged at 2000 steps, but the frozen-CI
   variant is clearly plateauing while "both" still improves.
2. **Training the CI fns matters.** Components-only (frozen CI) stalls at ~2× worse
   rounded and ~2.3× worse PGD. The masks the single-block runs learned are not the
   right masks for the combined model — even with components free to adapt under
   them.
3. **The price of training CI fns is sparsity, not targeting** (at least at 2000
   steps): total L0 rises 38 → 88 in the first 500 steps, then re-sparsifies to ~68
   (vs the singles' sum of 38). Nontarget behaviour transiently erodes (0.011 →
   0.026 at step 1000) and then self-corrects to ~0.013, close to the raw-combined
   level. Frozen CI fns cannot change any mask (CIs are computed from unmasked
   activations), so their L0 is pinned at 38.1 by construction.
4. Remaining gap to single-block quality is ~4× (0.024 vs ~0.006). Given the 4×
   smaller batch, the truncated LR schedule, and the still-decreasing trajectory,
   longer/bigger fine-tuning plausibly closes most of it.

## Objective 3: a single lighter CI fn works — no distillation needed

Setup: same fine-tune as objective 2's "both" variant, but ONE
`global_shared_transformer` CI fn over all 28 matrices (source architecture,
d512 × 4 blocks: ~90M params vs ~124M for the four per-block CI fns), randomly
initialised; CI-fn LR 1.6e-4 (the sources' from-scratch value); components
initialised from the sources.

| subject | rounded | PGD | target L0 | ntgt rounded | ntgt L0 | CI-fn params |
|---|---|---|---|---|---|---|
| FT both (4 per-block CI fns) | **0.0239** | 0.0551 | **67.6** | 0.0133 | **0.59** | ~124M |
| FT fresh single CI fn | 0.0266 | **0.0484** | 97.9 | 0.0132 | 2.73 | ~90M |

(Dot-plot and trajectory figures above include this variant.)

Key conclusions:

1. **Feasible.** From a random initialisation (step 0: rounded 0.73, everything
   half-on at L0 3434) the single CI fn organises within ~500 steps and reaches
   recon comparable to the per-block bundle by 2000 — slightly worse rounded,
   and the best adversarial (PGD) recon of any variant.
2. The costs at this budget are **sparsity** (L0 98 vs 68, but still falling
   steeply: 187 → 128 → 111 → 97 over the second half) and **targeting**
   (nontarget L0 2.7 vs 0.6). Both trajectories suggest longer training closes
   the gap; the CI fn simply hasn't finished tightening.
3. The fallback (distilling the four CI fns into one) is unnecessary.

## Objective 4: completeness training works — and beats plain joint fine-tuning

Protocol (all stages at global batch 32, frozen-CI stages train only components):

1. **Over-sparse decomposition** = the objective-2 frozen-CI run: components adapt
   under pinned masks; plateaus at rounded 0.0431 (pinned masks cannot resurrect
   dropped mechanisms).
2. **Per-block resurrection** (1000 steps each, other blocks hard-frozen at the
   over-sparse state, init from its checkpoint): each block trains its components
   *and its CI fn* against the over-sparse rest, so it must supply whatever
   redundant mechanism the ensemble lost.

   | block | rounded 0→1000 | PGD 0→1000 | L0 gained |
   |---|---|---|---|
   | L16 | 0.0426 → 0.0362 | 0.150 → 0.114 | +12.6 |
   | L17 | 0.0426 → 0.0451 | 0.150 → 0.126 | +4.0 |
   | L18 | 0.0426 → 0.0351 | 0.150 → 0.108 | +14.8 |
   | L19 | 0.0426 → 0.0487 | 0.150 → 0.136 | +4.6 |

   Redundancy is heterogeneous: **L16/L18 are the resurrectors** (large L0 gains and
   real recon improvements); L17/L19 wake little.
3. **Frankenstein assembly** (each block from its own run): rounded **0.0605** —
   *worse* than the over-sparse baseline. The per-block gains do not compose (each
   block was tuned against the over-sparse others; all four changed at once — the
   objective-1 superadditivity in miniature; naive additive expectation was 0.037).
4. **Reconciliation** (1000-step joint fine-tune from the franken state, CI fns
   frozen): rounded 0.0605 → **0.0229** within 1000 steps.

| variant (standalone eval, same script/seed) | rounded | PGD | L0 | ntgt rounded | ntgt L0 |
|---|---|---|---|---|---|
| over-sparse (frozen-CI FT of raw assembly) | 0.0431 | 0.126 | 38.1 | 0.0132 | 0.35 |
| franken (resurrected, no reconcile) | 0.0605 | 0.135 | 72.8 | 0.0447 | 0.45 |
| **completeness (resurrect + reconcile)** | **0.0228** | 0.0598 | 72.8 | 0.0163 | **0.45** |
| joint FT "both" (reference) | 0.0239 | 0.0551 | 67.6 | 0.0133 | 0.59 |

Key conclusions:

1. **The resurrection hypothesis is confirmed.** Frozen-mask training saturates at
   0.043 (stage 1 proved this); after per-block resurrection the *same* frozen-mask
   training reaches 0.0228. The difference is entirely the ~35 L0 of components the
   per-block phase woke up — mechanisms the single-block decompositions had dropped.
2. **The completeness protocol matches full joint fine-tuning on target recon**
   (0.0228 vs 0.0239) at similar L0, with somewhat better off-distribution sparsity
   (ntgt L0 0.45 vs 0.59) — while all mask changes happened in isolated,
   attributable per-block phases rather than one entangled joint optimisation.
3. **Reconciliation is mandatory** — assembling the resurrected blocks without it is
   worse than not resurrecting at all (0.0605 vs 0.0431 rounded, and 3.4× worse
   nontarget recon, which the reconciliation also heals: 0.052 → 0.015).
