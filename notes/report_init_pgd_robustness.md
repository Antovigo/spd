# Does the weight init change adversarial robustness?

**Question.** Two L18 decompositions differing only in how their components were
initialised — does one survive a long PGD attack better than the other?

**Runs.** Both 20 000 steps.

| run | init |
|---|---|
| `p-88665048` (`addsub-L18-22-zerou`) | `zero_u` |
| `p-6540dfdd` (`addsub-L18-23-neuronaligned`) | `neuron_aligned_targeted` |

**Probe.** Fresh PGD on the output CI head, end-to-end output KL, 4 fixed batches,
10→100 steps, 2 adversary start points, source shape `c` (one mask per site, shared across
batch and position). Non-target arm is delta-pinned (SPEC T4).

![PGD vs adversarial steps by init](plots/init_pgd/01_init_pgd_vs_steps.png)

## What it shows

**General text — the aligned init is ~22% easier to attack.** At 100 steps 0.0127 against
0.0104, and the gap opens from 20 steps onward. This one is solid: the between-init gap is
more than twice either run's own spread across adversary start points (2% aligned, 9%
`zero_u`).

**Task distribution — the aligned init is ~10% *harder* to attack.** 0.0047 against 0.0052
at 100 steps. Smaller than the off-distribution effect and only marginally outside the
noise (`zero_u`'s own spread is 12% at that budget), so treat it as a hint, not a result.

So the alignment appears to buy a little on-distribution robustness and pay more for it
off-distribution.

**Neither init runs away.** Both saturate by ~40 steps. This matters as a control for the
nonlinearity-penalty work: there the penalised runs kept climbing to 2.5–3× while their
control flattened. Here both arms flatten, so "keeps climbing under a longer attack" is a
property of that penalty, not something every L18 decomposition does.

## Numbers (mean over 2 adversary start points)

| steps | 10 | 20 | 40 | 60 | 80 | 100 |
|---|---|---|---|---|---|---|
| task, `zero_u` | 0.0043 | 0.0048 | 0.0050 | 0.0051 | 0.0051 | 0.0052 |
| task, aligned | 0.0040 | 0.0043 | 0.0045 | 0.0045 | 0.0046 | 0.0047 |
| general, `zero_u` | 0.0086 | 0.0098 | 0.0102 | 0.0103 | 0.0104 | 0.0104 |
| general, aligned | 0.0092 | 0.0113 | 0.0126 | 0.0126 | 0.0127 | 0.0127 |

## Caveats

- One training seed per init; two adversary start points. Enough for the 22%
  off-distribution gap, not for the 10% on-distribution one.
- A 20-step probe reads 0.0098 vs 0.0113 — it sees about half the eventual gap. Quote the
  step budget with any of these numbers.
- **The two runs sit on opposite sides of the #1001 merge** and no single build can open
  both: `zero_u` uses the retired `pd.weight_init`, the aligned run uses
  `decomposition.sites.initialization`. They were therefore measured by two builds, which
  is only sound because the probe's computation is unchanged between them —
  `core/{recon_eval,adversary,masking,recon}.py` are byte-identical, `reconstruction_loss`
  and `_row_masked_kl` identical, `prepare_lm_batch` identical apart from a rename, and
  `make_fresh_pgd_step` a pure extraction into `make_fresh_pgd_scorer`. Probe inputs match
  too (`pd.seed` 0, eval batch 128, step size 0.1, 4 batches, same prompts and eval shard),
  so both runs see the same fixed batches. Re-verify this if either side moves again.
- Post-#1001 the aligned init covers **every site kind including attention**; the earlier
  `addsub-L18-18-neuronaligned-bosincl` run did not, and referenced a `neuron_ranks`
  artifact the new schema drops. This report uses the current variant, so its numbers are
  not interchangeable with earlier ones for that run.
- No coupled baseline: `p-5b7fa697`'s pin predates #1000 and no current build can open it,
  and pins are immutable (CONFIGS.md rule 4). A same-code coupled twin at 20k is the only
  way to add one.
