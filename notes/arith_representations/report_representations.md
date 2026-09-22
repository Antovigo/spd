# Integer representations in the residual stream — findings

Companion to [`plan.md`](plan.md), which fixes the method before any data was read. Decomposition
`p-ba5a0c05` step 40000, read basis = alive set of `addsub-05-filter-last-pos-ceiling` (6,022
residual-reading components: q/k/v/gate/up over 32 layers), prompts `a+b=` (10,000) and `a-b=`
with `a >= b` (5,050). Data: `~/out/arith_repr/` (`V_alive.npz`, `resid/`, `analysis/`); code
`param_decomp/arith_repr/`; branch `experiment/arith_representations`. Applet:
[`app/index.html`](app/index.html) (open over `file://`; presence map + per-cluster centroid
clouds); its synthetic-test twin is [`synth_app/index.html`](synth_app/index.html).

Vocabulary. A **read point** is `attn_in.l` (the post-norm residual layer `l`'s q/k/v read) or
`mlp_in.l` (what its gate/up read); `k` is the dimension of the read basis there. A hypothesis
`Q:tau` is the *pure period-`tau`* code of quantity `Q` (functions of `Q mod tau` orthogonal to
every coarser period and to the linear part), `Q:lin` its linear part, `Q:direct` the
non-periodic remainder. `res` is `a+b` on addition, `a-b` on subtraction; every result hypothesis
is interaction-only (orthogonal to all functions of `a` alone and of `b` alone). The **presence
score** of a hypothesis is its *generalising energy*: the fraction of the read-basis energy at
that (read point, position) carried by the fitted directions that predict prompts whose `a` or
`b` value was held out, above the permutation null. `dim S` is the number of such directions.

## Setup log

- 2026-09-21: `V_alive.npz` extracted (job 12076); residual harvest on one L40 (job 12079, 4 min),
  64 read points x 5 positions x 20,000 prompts, bf16, 49 GB.
- Synthetic check of the pipeline (jobs 12078-12102; `~/pd_scratch/dual_obj_jax/arith_repr/synth_test.py`,
  `synth_harvest.py`): planted `a mod 10`, `b mod 10`, `res mod 10` circles, a linear `a`, a
  subtraction-only `b mod 5`, and an `a == 37` lookup feature in 40-D noise. It caught, before any
  real activation was analysed, a wrong projection on non-product prompt measures and a too-loose
  null; the first two real layers then motivated the linear hypothesis and the interaction-only
  result hypotheses (plan section 6 lists every change). Final synthetic outcome: the three circles
  come back as exactly 2-D subspaces (held-out R² 0.84), `a:lin` as 1-D (R² 0.77), the lookup has
  zero generalising energy, and the clusters on the addition prompts are exactly `{a:lin} {a:10}
  {b:10} {res:10}`; on the pooled set `b@add:10` and `b@sub:10` share a cluster while the planted
  subtraction-only `b:5` is its own.
- Analysis sweep (job 12103, 16 tasks; 12120 for three read points that timed out under CPU
  oversubscription): 64 read points x positions 1-4 x {both, add, sub}, 5 value-held-out folds,
  5 null replicates. ~20 min per read point on 8 cores.

## 1. What is where (presence map)

Numbers are generalising energy, i.e. fraction of the read-basis energy at that position
(`kept` = fraction explained by all kept subspaces together; the remainder is mostly
token-identity lookup, which cannot generalise by construction).

### Position 1 (`a`): magnitude first, then a family of periods

| read point | k | a:lin | a:2 | a:5 | a:10 | a:20 | a:25 | a:50 | kept |
|---|---|---|---|---|---|---|---|---|---|
| attn_in.0 (embedding) | 26 | 0.16 | – | 0.16 | 0.03 | – | – | – | 0.69 |
| mlp_in.0 | 164 | 0.08 | 0.03 | 0.09 | 0.08 | 0.07 | 0.15 | 0.17 | 0.72 |
| mlp_in.6 | 105 | 0.33 | 0.01 | 0.07 | 0.03 | – | 0.03 | 0.12 | 0.72 |
| mlp_in.13 | 123 | 0.18 | 0.04 | 0.12 | 0.12 | 0.06 | 0.06 | 0.12 | 0.75 |
| mlp_in.18 | 204 | 0.26 | 0.02 | 0.12 | 0.07 | 0.04 | 0.06 | 0.10 | 0.77 |
| mlp_in.24 | 189 | 0.33 | 0.02 | 0.10 | 0.06 | 0.03 | 0.02 | 0.08 | 0.75 |
| mlp_in.31 | 200 | 0.36 | 0.03 | 0.12 | 0.06 | – | – | 0.02 | 0.73 |
| attn_in.27 | 10 | 0.69 | – | – | – | – | – | – | 0.97 |

The token embedding of `a`, as the layer-0 reads see it, already has a 1-D linear (magnitude)
direction and a 1-D `mod 5` direction (16 % each) plus a weak `mod 10` one. From layer 0's MLP
input onward the full family is there: linear, `mod 2`, `mod 5`, `mod 10`, `mod 20`, `mod 25`,
`mod 50`, and it stays remarkably stable through the whole network at MLP inputs (`a:lin`
0.2-0.36, `a:5` ~0.1, `a:10` ~0.06, `a:50` ~0.1 fading after L28). The late attention inputs
(tiny read sets, k ≤ 26) read almost only the magnitude (`attn_in.26/27/28/30`: 0.5-0.69). About
25-30 % of the read energy at this position is token identity (`a:100`, never generalises).

### Position 2 (op): the operator, plus `a`'s magnitude

`op:2` is 0.53-1.0 of the read energy everywhere; `a:lin` is 0.07-0.14 from layer 0 on (the
operator position already carries the magnitude of `a`), the periodic codes of `a` at most 0.025.

### Position 3 (`b`): `b`'s own family, and `a`'s magnitude moves in

| read point | k | a:lin | b:lin | b:2 | b:5 | b:10 | b:20 | b:50 | kept |
|---|---|---|---|---|---|---|---|---|---|
| attn_in.0 | 26 | – | 0.19 | – | 0.13 | 0.01 | – | – | 0.69 |
| mlp_in.2 | 102 | 0.08 | 0.17 | 0.02 | 0.06 | 0.04 | 0.03 | 0.09 | 0.82 |
| mlp_in.8 | 84 | 0.17 | 0.19 | 0.01 | 0.02 | 0.01 | 0.01 | – | 0.89 |
| attn_in.13 | 22 | 0.21 | 0.24 | – | – | – | – | – | – |
| mlp_in.18 | 204 | 0.04 | 0.16 | 0.02 | 0.07 | 0.05 | 0.02 | 0.07 | 0.84 |
| attn_in.20 | 33 | 0.01 | 0.07 | 0.07 | 0.19 | 0.11 | 0.03 | 0.09 | 0.95 |
| mlp_in.24 | 189 | 0.04 | 0.19 | 0.02 | 0.04 | 0.03 | 0.01 | 0.02 | 0.83 |

(addition prompts.) `b` gets the same family as `a`. `a:lin` is copied onto the `b` position by
the early-middle layers (0.05 at L2 → 0.17-0.21 at L8-13) and then recedes at MLP inputs
(0.04 from L18) while the attention inputs of L16-23 read `b`'s periodic codes strongly
(`attn_in.18`: `b:5` 0.14, `b:50` 0.13; `attn_in.20`: `b:5` 0.19). **No result code at
position 3**: every `res:*` is < 0.01 except `res:direct` at 0.03-0.06 (MLP inputs L16-31, low
held-out R² ~0.2) — a weak non-additive interaction of `a` and `b`, not a result representation.

### Position 4 (`=`): the result appears at layer 19 and takes over

Addition prompts, MLP inputs (attention inputs show the same codes at lower energy and are
listed in the applet):

| read point | k | a:lin | b:lin | res:2 | res:5 | res:10 | res:20 | res:25 | res:50 | res:100 | res:direct | Σ res |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlp_in.2 | 102 | 0.13 | 0.16 | – | – | – | – | – | – | – | 0.01 | 0.01 |
| mlp_in.13 | 123 | 0.27 | 0.22 | – | – | – | – | – | – | – | 0.01 | 0.01 |
| mlp_in.17 | 177 | 0.08 | 0.12 | – | – | – | – | – | – | 0.02 | 0.01 | 0.03 |
| mlp_in.18 | 204 | 0.07 | 0.09 | – | 0.01 | 0.01 | – | – | – | 0.04 | 0.01 | 0.07 |
| **mlp_in.19** | 138 | 0.07 | 0.07 | 0.05 | 0.05 | 0.05 | 0.01 | – | 0.09 | 0.08 | 0.01 | **0.32** |
| mlp_in.20 | 136 | 0.07 | 0.06 | 0.03 | 0.05 | 0.06 | 0.02 | – | 0.17 | 0.14 | 0.01 | 0.48 |
| mlp_in.21 | 201 | 0.05 | 0.04 | 0.04 | 0.10 | 0.09 | 0.09 | – | 0.16 | 0.14 | 0.01 | 0.64 |
| mlp_in.22 | 136 | 0.04 | 0.04 | 0.07 | 0.09 | 0.14 | 0.11 | – | 0.16 | 0.17 | 0.01 | 0.74 |
| mlp_in.24 | 189 | 0.04 | 0.04 | 0.04 | 0.12 | 0.15 | 0.09 | 0.02 | 0.14 | 0.19 | 0.01 | 0.76 |
| mlp_in.27 | 213 | 0.04 | 0.04 | 0.04 | 0.12 | 0.15 | 0.07 | 0.05 | 0.13 | 0.21 | 0.02 | 0.78 |
| mlp_in.30 | 245 | 0.05 | 0.04 | 0.04 | 0.10 | 0.11 | 0.06 | 0.08 | 0.14 | 0.21 | 0.04 | 0.78 |

Before layer 19 the `=` position holds the two operand magnitudes (`a:lin`, `b:lin` 0.1-0.27
each) and `a`'s periodic codes (`a:5` up to 0.12 at `mlp_in.2/4`), and no result. Between
`mlp_in.18` (0.07) and `mlp_in.19` (0.32) the result codes switch on — every period at once —
and by layer 22 they are three quarters of everything the alive reads see at `=`, staying there
to the end. The operand codes shrink to ~0.04 each. Subtraction shows the same onset and the same
periods (Σ res 0.21 at `mlp_in.19`, 0.45 at L21, 0.52-0.57 from L22), with `res:100` weaker
(0.05-0.08 vs 0.14-0.23) and `res:direct` absent by construction (range 0..99).

**`a` and `b` are represented identically on the two operations, the result too — except for the
magnitudes.** Top principal cosine between the subspace fitted on addition and on subtraction at
the same read point (mean over read points, position 4): `res:5` 0.97, `res:10` 0.97, `res:2`
0.91, `res:20` 0.89, `res:50` 0.87, `res:100` 0.70; `b:2` 0.78, `b:5` 0.70, `b:10` 0.58; but
`a:lin` 0.44 and `b:lin` 0.44 (position 3: `b:lin` 0.39, `a:lin` 0.27, while `b:2/4/5/10` are
0.82-0.89). The periodic codes are shared; the linear magnitude directions differ by operation
(not by a sign flip — the cosine is unsigned).

## 2. Geometry: Fourier circles first, then something closer to a lookup

![residue centroids](figures/result_residue_centroids.png)

Residue centroids of the result codes at `=` (addition), projected on the top-2 principal axes
of each recovered subspace `S_H` (hsv colour = residue, line joins consecutive residues).

- **At onset (`mlp_in.19`) every result code is exactly 2-D and a circle in residue order**:
  `res:10` (10 points in order 0→9), `res:5` (a pentagon), `res:50` (50 points on a clean
  circle), `res:100` (100 points on a circle). The two directions of each subspace have equal
  energy and held-out R² 0.74-0.83. Nothing in the method assumes circles: the hypothesis is
  "functions of `res mod tau`", the shape is what the fitted centroids do.
- **By `mlp_in.22-24` the short periods fill their hypothesis space**: `res:5` and `res:10` become
  4-D (all their Fourier modes; `dim S = 4/4`, R² 0.82-0.92). For `res:10` at `mlp_in.24` the
  centroid curve has 72 % of its power in harmonic `k = 1` and 25 % in `k = 3` (below); the first
  plane is still a circle in order, the second plane is the `k = 3` winding.

![res:10 harmonics](figures/res10_harmonics_mlp24.png)

- **From layer 26 on the long periods stop being circles**: `res:25`, `res:50`, `res:100`,
  `res:direct` grow from 2-4 dimensions to 10-30 (`mlp_in.30`: `res:100` 28-D, `res:50` 17-D),
  their centroid clouds become jagged (bottom row), and the top two principal axes hold only
  35-57 % of the centroid variance. The representation of the result is moving from a few
  Fourier planes towards per-value directions — what the unembedding needs. Total result energy
  is flat (0.76-0.78) while this happens: the same energy is re-expressed in more dimensions.
  `res:5` and `res:10` stay 4-D to the end.

![result-value centroids](figures/result_value_centroids.png)

The centroids of the 199 result values in the union of all result subspaces (`mlp_in.24` 30-D,
`mlp_in.30` 100-D) wind around with the value; the top-2 axes hold only ~20 % of the variance,
so this projection is only a hint of the mod-100 wrap.

## 3. Separability and clusters (the model's own reads)

Per read point and position, two recovered subspaces are *separable* iff some alive read lands in
one at the 99.9th percentile of the random-direction null while sitting at that null on the other,
and vice versa (plan section 4). Clusters are connected components of "not separable".

**At the `=` position of the MLP inputs (addition), the reads sort the result codes into period
families, and the families dissolve into single periods with depth:**

| read point | clusters among the strong result codes (all kinds pooled) |
|---|---|
| mlp_in.19 | {a:2, b:2, res:2}  {a:5, res:5}  {a:10, b:10, b:50, res:10, res:50, res:direct}  {res:100} |
| mlp_in.21 | {a:lin, b:lin, b:50, res:20, res:100, res:direct}  {res:2}  {res:5, res:10}  {res:50} |
| mlp_in.23 | {res:2, res:5, res:10, res:20}  {res:50, res:100} |
| mlp_in.24 | {a:lin, b:lin, res:direct}  {res:2, res:5, res:10, res:20}  {res:25, res:100}  {res:50} |
| mlp_in.26 | {res:5, res:10} and every other period alone |
| mlp_in.28, 29 | every period alone ({a:lin, b:lin, res:direct} at 29) |
| mlp_in.30, 31 | re-merge: {a:lin, b:lin, res:2, res:5, res:10, res:direct} ; {res:100} ; {res:25, res:50} (30); one large cluster at 31 |

(`res:2` and `cross:2` are the same function — the parity of `a+b` equals that of `a-b` — and
always co-cluster; `cross:2` is dropped from the table.) Three readings:

1. At onset and through L24 there is a **"digit" family** (`mod 2/5/10/20`, i.e. the last digit
   and its parity/fives structure), a **"long-period" family** (`mod 25/50/100`), and a
   **"magnitude" family** (`a:lin`, `b:lin`, `res:direct`) that the reads do not separate.
2. Layers 26-29 are where the reads become **selective**: nearly every period is read by some
   component that ignores the others. This is the same depth at which the long periods stop
   being circles (section 2).
3. On subtraction the reads separate much less: one large cluster containing `b:lin` and every
   result period through L28, splitting only at L29-30. Subtraction's result codes are as
   present as addition's, but the alive reads that consume them also read `b`'s magnitude.

Per kind, `gate_proj` and `up_proj` alone almost never separate anything (each kind's reads
alone rarely include the clean single-family reads); the pooled kinds do. The attention read
points are **not informative for separability**: with k = 7-26 reads, the null of "a random
direction in the read span" is nearly degenerate (a 4-D subspace of a 16-D span needs overlap
> 0.9 to count as read), so most hypotheses come out *unread* there. That is a property of the
null the plan fixed, not evidence that attention ignores the result; the presence scores at
those read points are unaffected.

## 4. Answers to the questions of the brief

- **How are the representations arranged?** Each integer quantity is a *sum* of codes, one per
  period, in (nearly) mutually orthogonal planes: linear magnitude (1-D), and Fourier pairs for
  periods 2, 5, 10, 20, 25, 50 (operands) or 2, 5, 10, 20, 25, 50, 100 (result). At onset each
  period is one 2-D circle in residue order; the short periods later use both harmonics (4-D);
  the long periods of the result fan out to 10-30 dimensions in the last six layers.
- **Overlap.** Between periods of the same quantity: the hypotheses are orthogonal by
  construction, and the fitted planes are separable by the reads from L26 on; before that the
  reads mix the digit family and the long-period family. Between quantities: `a` and `b` codes
  are separable from each other where both are present (positions 3-4 before L19); the result
  codes co-cluster with `a:10`/`b:10`/`b:50` only at onset (L19-21). Adjacent residues are
  adjacent on the circle (the line through consecutive residues is a loop in every 2-D panel of
  the figure); the same value at different periods lives in different planes.
- **Linear separability.** Confirmed as the operational definition in plan section 4: for a
  bias-free read `W`, "`W` can point at `A` without being influenced by `B`" means a row of `W`
  with full overlap on `S_A` and chance overlap on `S_B`; the sweep measures that with the
  alive reads themselves. Which representations get a non-zero dot product with a vector aimed
  at `X mod Y`: at MLP inputs from L26 on, essentially only `X mod Y`; at L19-25, the other
  members of its family (digit or long-period); at attention inputs the question cannot be
  answered with this null.
- **Clusters.** Given above; the clusters are the period families, then single periods.
- **Low-dimensional space per cluster.** Recorded per cluster in the applet (`dim` and the
  centroid spectrum): 2 per period at onset, 4 for `mod 5/10` from L22, 10-30 for `mod 25/50/100`
  from L26; the digit family at `mlp_in.24` is 19-D in total, the long-period family 10-D.

## 5. Caveats

- The read basis (6,022 alive components of one CI filter) covers 20-60 % of the residual
  energy (`outside_basis_energy` 0.3-0.5 at MLP inputs, 0.6-0.8 at attention inputs). Everything
  here is about what the *alive reads* see, by assumption 1 of the brief.
- `a:100`/`b:100` (token identity) and a *smooth non-linear* magnitude code are the two things
  the value-held-out test cannot credit; the first by construction (a lookup cannot predict an
  unseen value), the second because only the linear function is in the lattice. A code like
  `log a` would show up spread over `a:50`, `a:25` and `a:lin` — the long operand periods should
  be read with that in mind (their residue circles were not inspected; the result's were).
- A linear *result* code is the same function as `L_a + L_b` and is credited to the operands
  (plan section 2). `a:lin` and `b:lin` at `=` after L19 are 0.04 each and co-cluster with
  `res:direct`; a shared magnitude direction for the sum is consistent with that.
- Subtraction has a third of the prompts and a non-product measure (`lattice_overlap` 0.25
  there vs 0.07-0.10 elsewhere); its long-period codes carry more leakage between incomparable
  divisors (25 vs 10, 20 vs 25). The unique-energy column in the applet is the conservative
  number.
- The applet's presence map is generalising energy under `both`/`add`/`sub`; `both` uses the
  op-conditional `b@add`/`b@sub` and the pooled `res`.
