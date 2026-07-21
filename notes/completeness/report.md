# Report — redundancy toy testbed (cipher transformer)

Setting: the combine-layers experiments showed that when several blocks implement the
same computation, a decomposition trained with a sparsity penalty credits one block
and prunes the backups (the ε < τ branch of the threshold account in
`notes/combine_layers/report.md`). On the 8B model we can only infer this indirectly.
This testbed makes the failure exactly measurable: a toy transformer with *known,
trained-in* n-fold redundancy, so any decomposition can be scored by how many
ground-truth mechanisms it skips.

Code: `param_decomp_lab/toy_models/redundant_cipher_transformer.py` (toy) and
`param_decomp_lab/experiments/cipher/` (PD experiment + `plot_ci.py` diagnostic).
Runs: `~/out/runs/toy_model_redundancy_training` (toy),
`~/out/runs/toy_model_redundancy_decomposition` (baseline decomposition).

## The toy model

An MLP-only pre-norm decoder, as transformer-like as the task allows:

- **Inputs** are token-ID sequences (equivalently one-hot); a *fixed* random
  unit-norm embedding `W_E [32, 64]` maps them into the residual stream.
- **Blocks**: 3 residual MLP blocks `resid + mlp_out(relu(mlp_in(RMSNorm(resid))))`,
  `d_mlp = 48`, no biases. Blocks (plus norm gains) are the only trained parameters.
- **Output**: final RMSNorm, then the *tied* unembedding `W_E` produces logits — a
  distribution over tokens per position, per-position cross-entropy.
- No attention and no positional embedding: the task is per-position, so positions
  are just batched samples through shared weights.

**Task**: predict `pi(x_t)` at every position, where `pi` is a fixed random
*derangement* of the vocab (a permutation with no fixed points), sampled once from
the seed. With embed/unembed frozen, each block must rotate `e_x` toward
`e_pi(x)` in the residual stream — one mechanism per input token, 32 per block,
**96 ground-truth mechanisms** total.

**Why redundancy emerges**: every training sequence keeps a random block subset,
drawn uniformly from all `2^3 − 1 = 7` non-empty subsets. Each singleton subset
appears in training, so *every block alone must implement the full cipher*; each
block also tolerates any combination of live partners, because it reads the
RMSNorm'd residual and can see what upstream already contributed. This is the same
mechanism we hypothesize in the 8B model — later layers carry backup copies of
computations earlier layers already do — but here it is exact and certified.

**Why a derangement**: the empty subset (all blocks off) leaves `resid = e_x`, whose
nearest unembedding row is `x` itself; since `pi(x) ≠ x` everywhere, the empty model
scores *exactly* 0 accuracy. The certificate is binary — no partial credit.

Training: 5000 steps, batch 512, Adam lr 1e-3, ~2 min on CPU.

## Ground-truth certificate

`verify` evaluates every block subset (8192 sequences; keep-mask bit string is
`block2 block1 block0`):

| blocks enabled | CE | accuracy |
|---|---|---|
| none (`000`) | 12.6 | **0.0000** |
| block 0 only (`001`) | 2.5e-4 | 1.0000 |
| block 1 only (`010`) | 2.9e-4 | 1.0000 |
| block 2 only (`100`) | 3.4e-4 | 1.0000 |
| blocks 0+1 (`011`) | 2.3e-4 | 1.0000 |
| blocks 0+2 (`101`) | 2.3e-4 | 1.0000 |
| blocks 1+2 (`110`) | 2.6e-4 | 1.0000 |
| all (`111`) | 2.5e-4 | 1.0000 |

Any single block implements the cipher perfectly; any partial combination works
equally well; removing everything fails completely. The per-token map
(`per_token_accuracy.npz`, `[n_blocks, vocab]` accuracy of each block alone per
input token) is all-ones: **every one of the 96 (block, token) mechanisms exists**,
with per-token accuracy 1.0 — the ground truth the decomposition is scored against.

## Baseline decomposition

Standard joint decomposition of all 6 matrices (`blocks.*.mlp_in`, `blocks.*.mlp_out`),
KL reconstruction against the toy's output distribution, uniform random token
sequences as data. Config: `param_decomp_lab/experiments/cipher/cipher1_config.yaml`.

Hyperparameters that matter for this experiment:

| | value | why it matters |
|---|---|---|
| importance minimality coeff λ | **1e-5** | sets the prune threshold: a mechanism is kept only if masking it costs more than ~λ/w in recon |
| impmin pnorm | **2.0, constant** (no anneal, beta 0) | p = 2 gives an *interior* CI optimum `u* ≈ w·ΔKL/2λ` — CIs are graded, and a backup whose marginal ΔKL is small settles below the alive threshold |
| mask-seeing recon losses (w) | StochasticReconLayerwise 1.0 + StochasticRecon 1.0 | the "weight" side of the threshold; both see sampled masks |
| C per matrix | 64 | 2× the 32 mechanisms per matrix — capacity is not the binding constraint |
| CI fn | layerwise MLP, hidden [16] | per-matrix CI, no cross-block sharing |
| sigmoid / sampling | leaky_hard / 1 mask sample | |
| delta component | on | |
| steps × batch | 10 000 × 512, lr 2e-3 constant | |
| faithfulness warmup | 200 steps | |
| alive threshold | max CI over tokens > 0.1 | defines "skipped" in the diagnostic |

The key point: because each mechanism is redundant, masking a block-1 or block-2
subcomponent while its partners are unmasked costs almost nothing in KL — the
residual-stream backup covers for it. Its equilibrium CI therefore sits near the
λ-driven floor, below 0.1, and the sparsity penalty prunes it even though the
*mechanism provably exists* in the weights.

### Results

Final eval: stochastic recon KL ≈ 5e-6 (layerwise and global) — reconstruction is
essentially perfect; the failures below are purely about *attribution*, not fit.

Per-matrix active subcomponents (max CI > 0.1 on any token):

| | mlp_in | mlp_out | skipped (block, token) mechanisms |
|---|---|---|---|
| block 0 | 32 | 32 | 0 / 32 |
| block 1 | 18 | 11 | 12 / 32 |
| block 2 | 15 | 13 | 17 / 32 |
| **total** | | | **29 / 96** |

![active subcomponents](report_figures/active_subcomponents.png)

Block 0 gets a clean one-subcomponent-per-token decomposition in both matrices.
Blocks 1 and 2 keep only partial sets — the decomposition credits the first block
that computes each token and prunes the backups, exactly the combine-layers
pathology. Note the asymmetry within a block (e.g. block 1: 18 mlp_in vs 11 mlp_out
active): a backup can stay half-alive through one matrix while its other half is
pruned.

![coverage vs truth](report_figures/coverage_vs_truth.png)

The SKIPPED panel (ground-truth accuracy > 0.9 but max CI < 0.1 across the block's
subcomponents) is the recovery-method score: a method that recovers redundant
mechanisms should clear these cells. Baseline: 29 skipped, all in blocks 1–2.

Reproduce:

```bash
python -m param_decomp_lab.experiments.cipher.run <config.yaml> --run_id=cipher-<details>-<NN>
python -m param_decomp_lab.experiments.cipher.plot_ci ~/out/runs/<run_id>
```

## Experiment: lowering the impmin coeff

If pruning is threshold-driven, lowering λ should recover mechanisms whose
partner-masked marginal ΔKL sits between the old and new thresholds. Sweep at
λ ∈ {3e-6, 1e-6, 3e-7} (baseline 1e-5), everything else identical; runs
`toy_model_redundancy_impmin_{3e-6,1e-6,3e-7}`, ~20 min each on CPU.

Results (active counts as mlp_in+mlp_out per block; ideal is 32+32 everywhere):

| impmin coeff λ | skipped / 96 | b0 active | b1 active | b2 active | recon KL |
|---|---|---|---|---|---|
| 1e-5 (baseline) | 29 | 32+32 | 18+11 | 15+13 | 5.1e-6 |
| 3e-6 | 14 | 41+40 | 25+20 | 22+18 | 2.5e-6 |
| 1e-6 | **0** | 53+59 | 34+34 | 30+27 | 1.5e-6 |
| 3e-7 | **0** | 57+64 | 49+38 | 45+29 | 8.0e-7 |

Lowering λ monotonically recovers the skipped mechanisms — full coverage from
λ = 1e-6 — confirming the pruning is threshold-driven. But coverage is bought with
fragmentation, and the *cleanliness inverts across blocks*:

- **λ = 3e-6** ([figure](report_figures/active_subcomponents_impmin3e-6.png)):
  still near-diagonal everywhere, but 14 backups remain skipped and block 0
  already over-allocates (41+40).
- **λ = 1e-6** ([figure](report_figures/active_subcomponents_impmin1e-6.png)):
  blocks 1–2 now have clean, near-complete diagonals (27–34 active), while
  block 0 shatters — 53+59 active with heavy off-diagonal mixing, i.e. mechanisms
  split and duplicated across subcomponents. The baseline's picture (block 0
  clean, backups pruned) is exactly inverted.
- **λ = 3e-7** adds further duplication everywhere (up to 64/64 in b0 mlp_out)
  with no coverage gain.

So on this testbed there is no λ that is simultaneously complete and clean:
by the time the threshold is low enough to keep the backups, it is too low to
force one-subcomponent-per-token in the redundant regime. Sparsity alone cannot
do both jobs — which is the motivation for completeness-style protocols
(train sparse, then recover/repair) rather than a single global penalty.
