# Magenta components in addsub-L18-23-neuronaligned

*Investigation of components whose output-head CI is high while their hidden-head CI is low
("magenta"), on the finished run p-6540dfdd (step 20000). Scripts: `~/pd_scratch/magenta/`.
Grid data: the run's `ab_grids/step_*.js` (a, b in 1..100, prompt `a+b=`, answer position).*

## 1. How much magenta is there, and where

Thresholds: magenta = output CI >= 0.5 and hidden CI <= 0.1; the reverse ("cyan", hidden-only)
= hidden >= 0.5 and output <= 0.1. Over the 198 saved components x 10000 cells at step 20000:

| pattern | (component, cell) pairs | share |
|---|---|---|
| magenta (output-only) | 1 097 | 0.06 % |
| cyan (hidden-only) | 286 779 | 14.5 % |
| both on | 330 256 | 16.7 % |

- Magenta is rare and concentrated: gate_proj[30] (474 cells) and down_proj[30] (461 cells) carry
  85 % of it; down_proj[34] is a distant third (62 cells). No other component exceeds 22 cells.
- gate[30] and down[30] are the same MLP neuron by construction (neuron-aligned init nests the
  gate/up/down rankings): neuron **3389**, the neuron studied earlier on addsub-L18-04. At step
  20000 the components have drifted from one-hot: 24 % (gate) / 30 % (down) of their energy is
  still on coordinate 3389, the rest is spread over thousands of neurons. up_proj[30] is below the
  mean-CI floor (dead on this grid).
- Neuron 30's magenta cells sit where one operand is a single digit (b in 1..10 for any a, and the
  a in 1..10 row), plus a sparse band around a in 40-90 with b in 50-60. Its inner activation there is
  about half its strength on both-on cells (0.26 vs 0.54 for gate[30]; 0.13 vs 0.32 for down[30]), so
  the neuron fires weakly on magenta cells: the output head keeps it on, the hidden head drops it.
- No substitute: in neuron 30's magenta cells, the hidden-only rate of every other component is
  the same as its grid-wide rate. The hidden head is not swapping in a replacement; it simply
  judges the component unnecessary for the MLP-output targets.
- The dominant disagreement is the reverse one: at the answer position the hidden head keeps
  about 66 of the 198 saved components on, the output head about 33, and there is no cell where
  the output head is the denser of the two.

Across snapshots the two effects grow together: magenta pairs 296 -> 198 -> 578 -> 640 -> 1097
(steps 4k, 8k, 12k, 16k, 20k), neuron 30's magenta cells 118 -> 78 -> 221 -> 292 -> 461, and the
cyan share 9.1 % -> 9.7 % -> 9.2 % -> 11.4 % -> 14.5 %. The output head gets sparser relative to
the hidden head as training proceeds.

## 2. Ablation / direct-path experiments (GPU, `analysis_magenta.py`, job 11091)

All 10 000 prompts, answer position, delta channel on. Cell groups are defined by gate[30]'s
CI (magenta 481 cells, both-on 2555, hidden-only 132, both-off 5715). The clean model gets
94.5 % of the sums right.

**Both masks are faithful; the output mask is internally less faithful.**

| forward | KL to clean | top-1 agree | hidden rel. err @answer (mean of 14 points) | final-residual rel. err |
|---|---|---|---|---|
| all components + delta | 0 | 1.000 | 0 | 0 |
| output-CI mask | 0.0058 | 0.990 | 0.036 | 4.1e-3 |
| hidden-CI mask | 0.0029 | 0.990 | 0.0074 | 1.2e-3 |

The output pattern reproduces the logits with MLP outputs five times further from clean than
the hidden pattern's. Part of this is a plain coefficient asymmetry in the recipe: the CI-shaping
recon weight is 1.5 on the output pass (stochastic 1.0 + PPGD 0.5; the unmasked term does not
touch CI) but 3.0 on the hidden pass (2.0 + 1.0), against the same imp-min coefficient, so the
hidden head's effective sparsity pressure is half the output head's. That alone predicts the
2x L0 gap and most of the "cyan" mass.

**Neuron 30 (3389) on its magenta cells: a real but small output effect, an invisible hidden
effect, no direct path, no compensation.**

| ablate gate/up/down[30] in the full model | magenta cells | both-on cells | both-off cells |
|---|---|---|---|
| KL at answer | 0.0019 (68 % of cells > 1e-3) | 0.0103 | 0.0006 (16 % > 1e-3) |
| top-1 flips | 1.2 % | | |
| its layer-18 write, relative to the MLP-18 output norm | 2.7 % | 6.1 % | 1.5 % |
| hidden rel. err @answer, mean over 14 points (per-position normalized) | 3.8e-4 | 1.6e-3 | 1.9e-4 |
| same, as the training loss sees it (whole prompt) | 1.7e-4 | 5.0e-4 | 1.2e-4 |
| final-residual rel. err @answer | 1.8e-4 | 5.7e-4 | 1.2e-4 |
| KL through the direct path only (its MLP-18 write added to the clean final residual) | 0.0001 | 0.0004 | 0.0001 |
| cosine(direct logit delta, total logit delta) | 0.31 | 0.21 | 0.34 |

- The hidden head is *right* by its own yardstick: adding neuron 30 back into the hidden mask
  leaves the hidden error unchanged (0.0083 -> 0.0082), and removing it from the full model
  moves every later MLP output by <= 0.08 % (relative, squared) at the answer position. Per-position
  normalization does not change this: the answer position already carries 21-31 % of the clean
  norm at layers 18-30 (BOS dominates only layers 29-31: 20 %, 34 %, 90 %), so the earlier
  "massive-activation dilution" hypothesis is not what is happening here (prompts are 5 tokens).
- The output head is also right by its yardstick: KL 0.0019 is ~10x the imp-min coefficient, and
  the drop is context-independent (ablating it inside the output-masked model costs 0.0014, in the
  full model 0.0019; on both-on cells 0.0077 vs 0.0103). It is not a patch that only matters once
  other components are masked, so the "output head cheats" reading is not supported for this
  component.
- The direct path (MLP-18 write -> final norm -> unembed) carries ~5 % of the KL. The logit effect
  is indirect, but it does not run through later MLP outputs (they barely move). By elimination it
  runs through later attention outputs, which are not hidden points. Job 11092 measures this
  directly (section 3).
- Magenta is a threshold effect of one metric mismatch: a write that is 2.7 % of one layer's MLP
  output (7e-4 in squared relative terms, then averaged over 14 points) is below the hidden pass's
  imp-min threshold, while the same write moves the answer logits by KL 2e-3, above the output
  pass's. On both-on cells the neuron fires twice as hard (write 6 %, hidden error 1.6e-3) and clears
  both thresholds. The hidden-only control o_proj[33] is the mirror image: hidden error 1.4e-3 with
  KL 0.0010.

## 3. Delta dependence and where the effect travels (`analysis_delta.py`, job 11092)

**Neither pattern leans on the weight delta.** KL to clean, all cells, answer position:

| forward | delta on | delta 0.5 | delta off |
|---|---|---|---|
| all components | 0 | | 0.0020 |
| output-CI mask | 0.0058 | 0.0057 | 0.0068 |
| hidden-CI mask | 0.0029 | 0.0031 | 0.0040 |

Turning the delta off costs both masks about +0.001, the same as it costs the full component sum.
The output pattern is not hiding task behaviour in the delta, on magenta cells or anywhere else.

**Neuron 30's effect travels through every later MLP output, thinly.** Ablating it in the full
model on magenta cells changes the answer-position activations by (absolute L2, summed over
layers): MLP outputs 1.31, attention outputs 0.32, final residual 1.39 (its norm is 113). So the
perturbation is carried by the MLP outputs the hidden objective measures, not by attention and not
by the direct path. It is simply small everywhere: 1-2 % of each layer's MLP-output norm, which is
3e-4 in squared relative terms, while the same 1.2 % change of the final residual moves the logits
by KL 2e-3 and flips 1.2 % of top-1 answers. Relative squared error is quadratic in a small
perturbation; the logits are not.

The mirror check on the masks themselves: the output pattern's answer-position deviation is
MLP 7.5 / attention 1.8 / final 7.4 (KL 0.0058); the hidden pattern's is 3.8 / 0.9 / 3.8 (KL
0.0029). Twice the internal error buys twice the KL: no sign of a pattern that fixes the logits
while the internals are wrong.

## 3b. Is neuron 30 covered for by hidden-only components? (`analysis_compensation.py`, job 11093)

The worry: the hidden pattern tolerates losing neuron 30 only because it switches on other
components (hidden-only ones) that stand in for it. Test on the 474 magenta cells (controls: 500
both-on, 500 both-off): H = hidden mask, S = the cell's hidden-only set (26 components per cell on
average), plus the 12 most common hidden-only components individually.

| hidden rel. err @answer, magenta cells | with S on | with S off | full model |
|---|---|---|---|
| cost of lacking neuron 30 | +1.0e-4 | +0.7e-4 | +3.8e-4 |

Removing every hidden-only component does not make neuron 30 matter more for the hidden
activations; it matters slightly less. Per component, the interaction (cost of removing c from H,
minus cost of removing c from H+n30) is at most 1e-4 for every candidate, against removal costs
of 5e-4 to 2.5e-2. Nothing in S is a stand-in for neuron 30. The earlier context test agrees:
removing neuron 30 inside the output-masked model (S off) costs 1.2e-4, inside the full model
3.8e-4.

Side finding: S contains mutually cancelling members. Removing all of S from H costs KL 0.0026
(0.0039 -> 0.0065), but removing gate[4] alone costs KL 0.086 and up[18] alone 0.031 (0.14 each on
both-on cells). These components distort the logits massively on their own and not at all as a
group, while the group does carry ~0.02 of hidden error. The hidden head keeps them on as a
group because they matter for the MLP outputs; the output head drops the group because it nets to
nothing at the logits. That is consistent behaviour, not compensation for neuron 30.

## 4. Conclusions

1. **Magenta is a one-neuron, borderline phenomenon, not a cheat.** It is 0.06 % of
   (component, cell) pairs and 85 % of it is one component (neuron 3389, gate/down[30]) on the
   ~5 % of the grid where that neuron fires at half strength. On those cells its marginal effect
   is small under both objectives (KL 0.002; hidden relative error 4e-4), and hidden-only cells of
   the same neuron have the same effect sizes (KL 0.002; 4.8e-4). The two heads are placing their
   decision boundaries differently on a component that sits near both thresholds. The effect is
   context-independent (same cost inside the masked model as in the full one), not delta-carried,
   not a direct path, not an attention path, and not covered for by hidden-only components (3b).
2. **The hypothesis "the output head cheats and the hidden head has to use a different
   pattern" is not supported.** The output pattern is a near-subset of the hidden pattern (no cell
   where it is denser), its logit fidelity scales with its internal fidelity, and it does not depend
   on the delta. What is true is that the output pattern's internals are 5x less faithful, because
   the logits genuinely do not need the extra ~33 components the hidden head keeps.
3. **The large disagreement is the other direction, and it has a recipe cause.** Hidden-only
   pairs are 14.5 % (240x magenta) and the hidden head keeps 2x the components. The CI-shaping
   recon weight is 3.0 on the hidden pass vs 1.5 on the output pass against one imp-min coefficient,
   so the hidden head runs at half the sparsity pressure. If the two heads are meant to be
   comparable, equalize that (hidden `impmin_coeff` 4e-4, or hidden recon coefficients 1.0 + 0.5)
   before reading any head disagreement as mechanism.
4. **The hidden objective cannot see what the logits see, by construction.** Adding the final
   residual as a hidden point would not help (neuron 30 changes it by 1.8e-4 in squared relative
   terms; its norm is ~113). Only a logit-gain-weighted comparison would register this neuron on
   these cells. Given (1), that is not worth building for this run.
5. **The magenta penalty attacks the wrong thing.** At 2e-5 and 1e-4 it fought output recon on a
   component that is legitimately (if marginally) useful, with no channel to raise hidden CI. The
   cancelled 1e-4 runs (p-b264f5bd, p-a8364218) are kept but there is no reason to resume them. If a
   hard `output <= hidden` guarantee is still wanted, parameterize the output head as
   `hidden * gate` rather than penalizing.

Reproduce: `~/pd_scratch/magenta/{load_grids,magenta_stats,magenta_cells}.py` on
`ab_grids/step_*.js`; `sbatch analysis_magenta.sbatch` / `analysis_delta.sbatch` (1 L40, ~10 min
each) then `summarize.py` / `summarize_delta.py` on `results/p-6540dfdd/*.npz`.
