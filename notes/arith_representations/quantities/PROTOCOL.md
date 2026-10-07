# Protocol (draft): finding the quantities read at a residual-stream site

Instructions for an agent that identifies the quantities represented at one site of the -05
addsub decomposition (run p-ba5a0c05, alive-only model) and tracks them from site to site.
Status: draft, built from the V1 / V2 experiments in `README.md`; steps marked (to build) have no
code yet.

## 1. Definitions

- Prompt: (op, a, b), op in {+, -}, a and b integers 1..100; i indexes the 20000 prompts.
- Token position t in 0..4 (`<BOS>`, a, op, b, =). With causal attention, activations at t = 1
  depend on a only, at t = 2 on (a, op), at t = 3 and 4 on (op, a, b). The domain of a site is
  the set of distinct values of these variables (100 values at t = 1).
- Stream position l in 0..64: 0 after the embedding, 2c+1 after block c's attention, 2c+2 after
  its MLP (c = 0..31).
- Site (c, read point, t): the readers of block c that read stream position l = 2c (q, k, v) or
  l = 2c+1 (gate, up), restricted to the readers active at t (original-model CI > 0.01 somewhere
  on the grid at t).
- x(i) in R^4096: raw residual stream at the site for prompt i (alive-only model, before the
  norm); X: the matrix of x over the domain.
- V~_r = gamma_l * V^_r: reader r's gain-folded unit read direction (gamma_l: the gain of the
  RMSNorm reading l, * element-wise). S = span of the n readers' V~_r, k = dim S, Q an
  orthonormal basis of S (4096 x k).
- Z = X Q: the stream in reader-span coordinates. W = Q^T [V~_1 ... V~_n] (k x n). Raw reads:
  Y = Z W (the reader's dot product with the raw stream; its activation is Y / rho, rho the RMS
  of x). Always fit raw reads; report rho separately (it is a function of the prompt too).
- Quantity q: a feature map phi_q from the domain to R^{d_q} (its encoding), stacked into
  Phi_q (domain x d_q), with directions D_q (k x d_q). Model: Z = 1 mu^T + sum_q Phi_q D_q^T + E.
- Envelope: a subspace of functions on the domain used to measure how much variance a family
  of quantities can carry; never accepted as a quantity itself.
- Reader metric: variance measured on Y (= Z W). Stream metric: on Z. Use the reader metric
  for acceptance; report both.
- Chance level of fitting d features to a residual R with df remaining degrees of freedom
  (df = domain size - 1 - dimensions already accepted): (d / df) * ||R||^2. Excess = increment
  - chance.

## 2. Inputs and tools

- Data: `virtual_weights/alive_only/resid.npy` (raw stream), `virtual_weights/vectors.npz`
  (V~), `virtual_weights/ci_positions.npz` (activity by position), dataset `index.npz`.
- Code (`notes/arith_representations/quantities/`): `v2.py` (site data, stagewise fit,
  chance levels, cross-validation over values, residual SVD), `v2_layers.py` (all sites),
  `v2_feature_test.py` (permutation test of one candidate on a residual),
  `v2_period20_test.py` (both-orders test).
- Run everything through sbatch (the host is a login node).

## 3. The loop at one site

### Step 0: data and sanity checks
1. Build Z, W for the site; check Y = Z W against the stored inner activations times rho
   (relative error < 1e-3).
2. Record n, k, the domain size, and rho over the domain.
3. If k < 10, flag the site as low-power: accept only quantities that are also accepted at an
   adjacent site.

### Step 1: envelope table (order-free)
Decompose the function space of the domain into orthogonal envelopes and report the reader
variance in each, with its chance level (to build).
- t = 1 (functions of a): Fourier over a = 1..100 (frequency k_a = 1..50, k_a periods over the
  range):
  - E_units: k_a in {10, 20, 30, 40, 50} (functions of a mod 10);
  - E_20: k_a in {5, 15, 25, 35, 45} (period 20, f(a + 10) = -f(a));
  - E_25: k_a in {4, 8, 12, ...} not already listed (functions of a mod 25), and similarly
    E_50, E_100 for the remaining k_a;
  - E_low: k_a <= 4 (smooth trends; overlaps E_25 at k_a = 4: assign it to one envelope and say
    which).
  Non-periodic quantities (a threshold such as [a <= 9], a single value) spread over many k_a;
  they are searched as candidates in step 2 across all envelopes.
- t >= 2: first the ANOVA split over the variables present (op, a, b and their interactions,
  orthogonal under the uniform grid), then Fourier envelopes inside each term (for a x b:
  diagonal k_a = k_b holds functions of a + b, anti-diagonal k_a = -k_b functions of a - b).
Envelopes are orthogonal, so this table does not depend on any order. It says where the
variance is; it names nothing.

### Step 2: candidates, cheapest first
Candidates are specific, low-dimensional encodings, in this search order:
1. Inherited: quantities accepted at the previous site (same t), with their feature maps.
2. Derived: functions of inherited quantities (a threshold or a sub-range of a magnitude, half
   of a circle, a coarser partition of a categorical).
3. Combined: functions of two inherited quantities (at t >= 3: e.g. a circle of a + b from
   circles of a and b).
4. New: proposed from the residual (step 4).
Encoding types: one direction (a scalar such as a, log a, a smooth curve of fixed shape, a
square wave, an indicator); a circle (cos, sin, 2 dims); classes of a partition (classes - 1
dims); a one-hot over listed values (one dim per value); a free smooth curve with r dims
(reduced-rank fit on a low-frequency basis); any of these restricted to a support set.
Never accept an envelope-sized family (e.g. "tens digit, 9 classes" or a 24-dim spline) when a
lower-dimensional candidate explains the same variance: broad families are envelopes.

### Step 3: forward selection with pruning
1. On the current residual, fit every candidate; score = excess / d (or the drop in description
   length, see `README.md`, once the grammar code is built).
2. Accept the best candidate if its excess is positive and its permutation p-value (feature
   permuted over the domain, 500 permutations, projected as in `v2_feature_test.py`) is < 0.01.
3. After each acceptance, recompute every accepted quantity's drop-one loss; remove any whose
   loss is below its chance level.
4. Both-orders test: for every pair of accepted or competing quantities whose feature spaces
   overlap (any principal angle < 60 degrees), fit A then B and B then A:
   - B adds nothing (excess <= 0) after A, A adds something after B: keep A, drop B;
   - both add: keep both;
   - neither adds after the other: keep the one with fewer dims (or fewer bits).
   Record the four increments.
5. Within each envelope, compare the accepted quantities' total with the envelope's variance
   (step 1). A gap above the envelope's chance level means a quantity of that family is missing.

### Step 4: residual analysis and new proposals
1. Residual before the one-hot remainder: top singular values vs a permutation null (each
   reader's residual permuted over the domain); top singular functions over the domain; power
   per frequency; values where several readers' residuals are > 3 robust SDs.
2. Read the top singular function as a table of values (largest positive, largest negative)
   and look for an arithmetic description (divisibility, digit patterns, value sets). Example:
   positive on every multiple of 8, negative on 2 mod 4, about 0 on odd numbers -> the 2-adic
   valuation.
3. A frequency peak is not a hypothesis by itself: residue classes mod 4, 6, 7, 8, 9 matched
   the residual's peaks and were rejected. Propose a specific function; then test it.
4. Selection bias: when a candidate is read off a residual, test it on data it was not read
   from (the other half of the values, or other sites) before accepting it.

### Step 5: generalisation and stopping
- Cross-validate the accepted set over values (10 folds over the domain's values; at t >= 3,
  hold out (a, b) pairs): report held-out R^2 per reader and pooled. One-hot quantities cannot
  generalise; structured ones should.
- Stop when (a) no candidate passes step 3, (b) the residual's top singular value is within the
  95% permutation null, and (c) every envelope's unexplained variance is within its chance level.
  Whatever remains is reported as the one-hot remainder (token-specific variance).

### Step 6: report for the site
- The accepted quantities in acceptance order: encoding, dims, increment, chance, excess,
  p-value, number of readers with >= 10% of their variance from it, and their directions D_q.
- The envelope table with explained / unexplained variance per envelope.
- The both-orders tables.
- Held-out R^2; the one-hot remainder; residual diagnostics.
- Geometry of each accepted quantity: for a circle, norm ratio and angle of its two directions;
  for classes, the singular values of the class means (simplex vs low-dimensional).

## 4. Across sites
- Process sites in stream order (l = 0, 1, 2, ...), per token position.
- Fit adjacent sites jointly when they share readers' subspaces: shared feature maps Phi_q,
  site-specific directions D_q,l (to build).
- Track per quantity: the sites where it is accepted, its excess, and the overlap of its
  directions between consecutive sites (projected into the later reader span): passed on
  (same directions), moved (new directions, same feature map), erased (an MLP / attention write
  anti-aligned with D_q), derived or combined (a new feature map that is a function of earlier
  ones).

## 5. Known pitfalls (from V1 / V2)
- Chance levels: with 100 values, a 10-dim feature set explains ~10% of anything. Report excess,
  not raw increments.
- Order dependence: stagewise increments depend on order when feature spaces overlap (the
  24-dim spline place code absorbed the tens digit and the period-20 circle at L0). Use
  envelopes and the both-orders test.
- Post-norm activations hide magnitude (rho grows like log a at L0). Fit raw reads.
- Small reader spans (k < 10) give noisy fits and unstable held-out R^2.
- Directions estimated from 90 values in 4096 dims are mostly noise: do not cross-validate in
  the full stream without a ridge penalty.
- Per-reader selection (V1) inflates fits: every reader reading its own items is a per-reader
  one-hot model. Quantities must be shared directions.
