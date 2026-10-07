# Protocol (draft): finding the quantities read at a residual-stream site

Instructions for an agent that identifies the quantities represented at one site of a
decomposed model and tracks them from site to site. The protocol is task-agnostic: it assumes
only a set of prompts described by task variables, and a set of reader components at the site.
It is a hypothesise-fit-inspect loop with backtracking: the agent guesses a quantity, fits it,
reads what it leaves in the residual, and either keeps it and moves on to the rest of the
residual, or replaces it with a better guess.

Objective: the residual must not be causally important. A reader's activation only has to be
explained on the prompts where that reader is causally important (CI above a threshold); what it
does on the other prompts does not reach the output and is not fit. Those activations are still
shown to the agent as hints: a pattern that is only important on a thin slice of the domain is
often easier to recognise over the whole domain.

Status: draft, built from the V1 / V2 experiments on the addsub task (`README.md`).

## 1. Definitions

- Prompt set: the prompts the site is analysed on, indexed by i = 1..N, each described by task
  variables v_1(i), ..., v_m(i) (for addsub: op, a, b). A token position t is fixed per analysis.
- Domain: the distinct values the site's activations can depend on at t (with causal attention,
  only variables of tokens at or before t). Duplicate prompts with the same domain value are
  averaged.
- Site: a set of n reader components reading the same residual-stream point (for example the
  q/k/v of one block, or its gate/up), restricted to the readers active at t.
- x(i) in R^d_model: the raw residual stream at the site for prompt i (before the norm); X the
  N x d_model matrix.
- V~_r: reader r's read direction with the norm's gain folded in, unit norm. S = span of the n
  V~_r, k = dim S, Q an orthonormal basis of S (d_model x k).
- Z = X Q (N x k): the stream in reader-span coordinates; W = Q^T [V~_1 ... V~_n] (k x n).
  Raw reads Y = Z W (the readers' dot products with the raw stream). The readers' activations
  are Y / rho, rho(i) the RMS of x(i); fit raw reads and report rho separately.
- Quantity q: a feature map phi_q from the domain to R^{d_q} (its encoding: a scalar, a circle
  (cos, sin), the classes of a partition, a one-hot over listed values, a smooth curve of an
  input, ... restricted to a support set if needed), stacked into Phi_q (N x d_q), with
  directions D_q (k x d_q). Model: Z = 1 mu^T + sum_q Phi_q D_q^T + E, mu the mean, E the
  residual. A quantity is shared by all readers: no per-reader selection.
- Causal importance: CI_r(i) in [0, 1], reader r's causal importance on prompt i at position t
  (the decomposition's CI function; for addsub, the filter's output CI on the original model,
  `dataset/original/ci.npy`). Active mask: M(i, r) = 1 if CI_r(i) > 0.01, else 0. N_r = number
  of distinct domain values on which r is active.
- Fitting objective: minimise sum over (i, r) of M(i, r) (Y(i, r) - Yhat(i, r))^2, with
  Yhat(i, r) = c_r + sum_q Phi_q(i) D_q^T w_r (w_r: column r of W, c_r: reader r's intercept
  on its active prompts). Only active entries enter the fit, the residual and every score
  below. When the reader directions are linearly independent (k = n), any reader coefficients
  B = D^T W are reachable, so this is one masked least-squares regression per reader on the
  shared features Phi; when k < n the readers' coefficients are tied through D (fit the D
  directly).
- Reader metric: variance of Y = Z W over active entries (used for decisions). Stream metric:
  variance of Z (all prompts; context only).
- Increment of q given a set A of quantities: the active-entry variance explained by adding q to
  the joint masked fit of A, as a fraction of the total active-entry variance (each reader's
  values centred on its active prompts).
- Chance level: summed over readers, d_q / df_r times reader r's residual variance before adding
  q, df_r = N_r - 1 - (dims already in A). Readers with N_r < 3 (dims of A + q) carry too few
  active values for these features and are left out of that comparison (reported).
  Excess = increment - chance. Low-dimensional features fit to few values explain a lot by
  chance (10 dims on 100 values: about 10%).
- Permutation p-value of q: the fraction of 500 random permutations of phi_q over the domain
  whose increment is at least the real one (same mask).
- Held-out R^2: the fit made on part of the domain values, active entries of the held-out
  values predicted from their features (10 folds over domain values).

## 2. What the agent keeps

- The accepted list A = (q_1, q_2, ...), in acceptance order, each with its rationale.
- The log of every hypothesis tried, with its numbers and the decision taken (accepted,
  revised into ..., rejected, removed because ...). Nothing is deleted from the log.
- The current residual: E = Z - fit of A (all accepted quantities refit jointly after every
  change).

## 3. The loop

### Step 0: data and checks
1. Build Z, W; check Y = Z W against the stored inner activations times rho (relative
   error < 1e-3).
2. Record N, the domain size, n, k, rho over the domain, and for each reader N_r and the
   fraction of its activation variance that falls on active entries.
3. Drop readers with no active entry at t. Readers active on very few values (N_r < 5) are kept
   but flagged: they constrain little and are easy to fit by chance.
4. If k < 10, mark the site as low-power: a quantity is accepted there only if it is also
   accepted at an adjacent site.

### Step 1: look at the residual
With A empty, the residual is each reader's read centred on its active prompts. The residual
is defined on active entries only; for the matrix diagnostics, inactive entries are set to 0.
Produce, in the reader metric:
1. The singular values of the residual matrix (domain values x readers) against a permutation
   null (each reader's active residual values permuted among its active domain values), and the
   top singular functions u_1, u_2, ... (functions over the domain).
2. Each top singular function plotted against every task variable and against every accepted
   quantity's features, and as a table of its largest positive and negative values with their
   domain values.
3. Per task variable (and pair of variables), the residual variance explained by the variable
   as a categorical, minus its chance level: which inputs the residual still depends on.
4. Per reader, over the whole domain: the raw read, with its active entries marked (and its
   CI), the current fit and the residual on active entries (for a sample of readers and for the
   readers with the largest active residuals). The inactive part of the read is a hint, not a
   target: a reader whose read is a clean sine of a over the whole range but active only on a
   slice suggests the sine; the hypothesis is then fit on the slice.

### Step 2: make a hypothesis
Write one hypothesis: a specific quantity (its feature map, its encoding, its support) and the
evidence for it (which singular function, which readers, which pattern). Prefer, in order:
1. quantities accepted at the previous site (passed on);
2. functions of accepted quantities (a sub-range, a threshold, half of a circle, a coarser
   partition);
3. combinations of two accepted quantities;
4. new quantities read from the residual.
Prefer the lowest-dimensional encoding that could produce the pattern (one direction before a
circle, a circle before a categorical, a categorical before a one-hot). A frequency peak or a
correlation is not a hypothesis by itself: name the function.

### Step 3: fit it
Refit A + q jointly with the masked objective (active entries only). Record: increment, chance, excess, p-value, held-out R^2 of A + q, the
number of readers with at least 10% of their variance from q, and the new residual.

### Step 4: judge it from what it leaves
Inspect the new residual on active entries (step 1 again), looking specifically at what is
tied to q:
- the residual plotted against q's features and against the variables q is a function of;
- the drop-one loss of every earlier quantity in A + q.
Then decide:
- **Reject** if excess <= 0 or p >= 0.01. Log it; return to step 1.
- **Revise** if q helps but the residual still has structure tied to q's variables. Examples:
  left-over harmonics of q's period (shape wrong: a square wave fit by a sinusoid, or the
  reverse); structure on part of q's range (support wrong); a residual that is a smooth function
  of q's value (encoding wrong: a non-linear function of the same variable). Write the revised
  hypothesis q', fit A + q' in place of A + q, and keep whichever leaves less structure tied to
  the variables at a higher excess per dimension. If two encodings are both plausible, run the
  both-orders test (fit q then q', and q' then q): keep the one after which the other adds
  nothing; if both add, keep both; if neither adds after the other, keep the one with fewer
  dims.
- **Backtrack** if adding q makes an earlier quantity redundant (its drop-one loss falls below
  its chance level): remove it from A, refit, log the removal and the reason. Also backtrack when
  a revision shows that an earlier accepted quantity was a poor proxy for the new one (the
  earlier one adds nothing after the new one, both orders).
- **Accept** otherwise: q joins A; return to step 1 on the new residual.

### Step 5: check generalisation
After every acceptance, the held-out R^2 of A (active entries of held-out values) must not
decrease. A quantity that raises the
in-sample fit and lowers the held-out fit is fitting individual domain values: revise it
(coarser encoding) or reject it.
Selection bias: a quantity read off the residual of some data is tested on data it was not read
from (the other half of the domain values, or another site) before it is accepted.

### Step 6: stop
Stop when (a) the last few hypotheses were all rejected, (b) the active residual's top singular
value is within the 95% permutation null, and (c) no task variable or pair explains active
residual variance above chance (step 1.3). Structure left on inactive entries does not count. What remains is reported as unexplained; if it is concentrated on
individual domain values, it is reported as value-specific (one-hot) variance.

### Step 7: report the site
- A, in acceptance order: encoding, dims, increment, chance, excess, p-value, readers using it,
  directions D_q, and its geometry (for a circle the norm ratio and angle of its two directions;
  for classes the singular values of the class means).
- The hypothesis log with every revision and backtrack, and why.
- Held-out R^2, the unexplained residual and its diagnostics.
- Coverage: per reader N_r, its active fraction, and its R^2 on active entries; readers left
  out of the chance comparison (too few active values).
- For context, the same fit's R^2 on all entries (active and inactive): a gap shows how much of
  the readers' activity the model is free to ignore.

## 4. Across sites
- Process sites in stream order, per token position, starting each site from the previous
  site's A as the first hypotheses (step 2.1).
- Track each quantity: the sites where it is accepted, its excess, and the overlap of its
  directions between consecutive sites (projected into the later reader span): passed on (same
  directions), moved (same feature map, new directions), erased (a write anti-aligned with D_q),
  derived or combined (a new feature map that is a function of earlier ones).
- A quantity accepted at a low-power site (k < 10) needs support from an adjacent site.

## 5. Pitfalls seen so far (addsub)
- The CI threshold (0.01) decides what must be explained; report how the accepted list changes
  at a stricter threshold (e.g. 0.1) for the main quantities. Weighting the squared errors by CI
  instead of thresholding is an alternative objective; do not mix the two within one analysis.
- A reader active on a few values can be fit by almost anything: its chance level is high and it
  should not drive a hypothesis on its own.
- Mean over active prompts: each reader's intercept is fit on its active prompts, so a reader
  that is active only where its read is high has nothing left to explain but the variation
  within that set.
- Raw increments mislead: report excess over chance.
- Broad feature sets (a 9-class partition, a 24-dim smooth basis) explain almost anything that
  overlaps them and make the order of fitting decide the attribution; prefer specific
  hypotheses and compare competing ones in both orders.
- Post-norm activations can hide a quantity carried by the norm (at L0 the stream's RMS grows
  like log a): fit raw reads.
- Small reader spans (k < 10) give noisy fits.
- Directions fit from about 100 domain values in d_model = 4096 dims are mostly noise: do not
  cross-validate in the full stream without a ridge penalty.
- Per-reader item selection (V1) is a per-reader one-hot model and inflates the fit; quantities
  must be shared directions.
- A residual frequency peak suggested residues mod 4, 6, 7, 8, 9 at token a; all were rejected.
  The singular function read as a table of values gave the right hypothesis (2-adic valuation).

## 6. Tools (addsub, `notes/arith_representations/quantities/`)
`v2.py` (site data, stagewise and joint fits, chance levels, cross-validation over values,
residual SVD), `v2_layers.py` (all sites), `v2_feature_test.py` (permutation test of a
candidate on a residual), `v2_period20_test.py` (both-orders test). These fit all entries; the
masked (CI-restricted) objective is not implemented yet (to build). Run through sbatch.
