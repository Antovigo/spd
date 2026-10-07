# Protocol (draft): finding the quantities read at a residual-stream site

Instructions for an agent that identifies the quantities represented at one site of a
decomposed model and tracks them from site to site. The protocol is task-agnostic: it assumes
only a set of prompts described by task variables, a set of reader components at the site, and
the decomposition's causal importances. It is a hypothesise-fit-inspect loop with backtracking:
the agent guesses a quantity, fits it, reads what it leaves in the residual, and either keeps it
and moves on to the rest of the residual, or replaces it with a better guess. A site is
finished when what is left does not matter causally, and is validated by patching the
reconstructed reads into the model.

Status: draft, built from the V1 / V2 experiments on the addsub task (`README.md`).

## 1. Definitions

- Prompt set: the prompts the site is analysed on, indexed by i = 1..N, each described by task
  variables v_1(i), ..., v_m(i) (for addsub: op, a, b). A token position t is fixed per analysis.
- Domain: the distinct values the site's activations can depend on at t (with causal attention,
  only variables of tokens at or before t). Duplicate prompts with the same domain value are
  averaged.
- Site: a set of n reader components reading the same residual-stream point (for example the
  q/k/v of one block, or its gate/up), restricted to the readers that are causally important
  somewhere at t.
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
  residual. A quantity is shared by all readers: no per-reader selection. The fit uses every
  prompt and every reader (no masking).
- Reader metric: variance of Y = Z W (used for decisions). Stream metric: variance of Z.
- Joint fit: least squares of the centred Z on all accepted quantities' features together, with a
  small ridge penalty (1e-3 of the mean feature variance). Without it, overlapping class
  partitions (units digit, tens digit, residues) are collinear on some subsets of the domain and
  the held-out predictions explode.
- Increment of q given a set A of quantities: the reader-metric variance explained by adding q
  to the joint fit of A, as a fraction of the total. Drop-one loss of q in A: the
  variance lost when q is removed from the joint fit of A.
- Chance level: d_q / df times the residual variance before adding q, df = (number of distinct
  domain values - 1 - dims already in A). Excess = increment - chance. Low-dimensional features
  fit to a small domain explain a lot by chance (10 dims on 100 values: about 10%).
- Permutation p-value of q: the fraction of 500 random permutations of phi_q over the domain
  whose increment is at least the real one.
- Held-out R^2: directions fit on part of the domain values, reads predicted on the held-out
  values from their features (10 folds over domain values).
- Causal importance: CI_r(i) in [0, 1], reader r's causal importance on prompt i at position t
  (the decomposition's CI function; for addsub, the filter's output CI on the original model,
  `dataset/original/ci.npy`). Important entry: (i, r) with CI_r(i) > 0.01.
- Important residual: the residual restricted to important entries. Its share = (residual
  variance on important entries) / (variance of Y on important entries). It is used to decide
  when to stop (step 6) and to find the readers to look at (steps 1 and 7), never in the fit.
  Compute it from held-out predictions for stopping: with 25-30 dims on 100 domain values the
  in-sample residual is small by chance.
- CI-weighted residual of reader r: sum over i of CI_r(i) (Y - Yhat)(i, r)^2; ranks the readers
  whose errors can matter.
- Patched run: the model run with the site's reader activations at t replaced by their
  reconstruction Yhat / rho (rho taken from the patched run itself), everything else unchanged.

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
2. Record N, the domain size, n, k, rho over the domain, and per reader the number of domain
   values on which it is important.
3. If k < 10, mark the site as low-power: a quantity is accepted there only if it is also
   accepted at an adjacent site.

### Step 1: look at the residual
With A empty, the residual is the centred stream. Produce, in the reader metric:
1. The singular values of E against a permutation null (each reader's residual permuted over
   the domain), and the top singular functions u_1, u_2, ... (functions over the domain).
2. Each top singular function plotted against every task variable and against every accepted
   quantity's features, and as a table of its largest positive and negative values with their
   domain values.
3. Per task variable (and pair of variables), the residual variance explained by the variable
   as a categorical, minus its chance level: which inputs the residual still depends on.
4. Per reader, the raw read, the current fit and the residual, with the important entries
   marked (and the CI), for a sample of readers and for the readers with the largest residual
   on important entries.
5. The important residual: its share, the same singular-function analysis restricted to
   important entries (other entries set to 0), and which readers carry it.

### Step 2: make a hypothesis
Write one hypothesis: a specific quantity (its feature map, its encoding, its support) and the
evidence for it (which singular function, which readers, which pattern). Prefer, in order:
1. quantities accepted at the previous site (passed on);
2. functions of accepted quantities (a sub-range, a threshold, half of a circle, a coarser
   partition);
3. combinations of two accepted quantities;
4. new quantities read from the residual.
Prefer the lowest-dimensional encoding that could produce the pattern (one direction before a
circle, a circle before a categorical, a categorical before a one-hot). Do not propose broad
bases (a 24-function spline over the whole range, all residue classes at once): accepted with a
small gain per dimension, they absorb the specific quantities they span and are then inherited
by every later site. A place code is a specific hypothesis when its spacing and width are stated
(bumps of a every 5 values, width 2.5). A frequency peak or a
correlation is not a hypothesis by itself: name the function. Patterns on important entries
come first: a pattern only on unimportant entries is low priority.

### Step 3: fit it
Refit A + q jointly. Record: increment, chance, excess, p-value, held-out R^2 of A + q, the
number of readers with at least 10% of their variance from q, the important residual's share,
and the new residual.

### Step 4: judge it from what it leaves
Inspect the new residual (step 1 again), looking specifically at what is tied to q:
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
  its chance level): remove it from A, refit, log the removal and the reason. When several are
  redundant at once, remove the one with the most dimensions first. Also backtrack when
  a revision shows that an earlier accepted quantity was a poor proxy for the new one (the
  earlier one adds nothing after the new one, both orders).
- **Accept** otherwise: q joins A; return to step 1 on the new residual.

### Step 5: check generalisation
After every acceptance, the held-out R^2 of A must not decrease. A quantity that raises the
in-sample fit and lowers the held-out fit is fitting individual domain values: revise it
(coarser encoding) or reject it.
Selection bias: a quantity read off the residual of some data is tested on data it was not read
from (the other half of the domain values, or another site) before it is accepted.

### Step 6: stop
Stop when either:
- the held-out important residual is small (its share below a target, e.g. 5%) or without structure (its
  top singular value within the permutation null, no task variable explaining it above chance):
  what is left is on unimportant entries or is noise, and does not matter for the output; or
- the last few hypotheses were all rejected and the residual (all entries) is within its
  permutation null.
Report what remains: its share on important and on unimportant entries; if it is concentrated
on individual domain values, as value-specific (one-hot) variance.

### Step 7: validate by patching
Patch each site's readers with the reconstruction, then all sites of the token position together.
Also patch the reconstruction fit without the patched values (held-out), the stricter test.
Run the model with the site's reader activations at t replaced by the reconstruction from A
(patched run), on the prompt set or a random subset, and compare its outputs with the
unpatched model. The test must be closed: the stream at t may reach the rest of the network only
through the reconstruction. Readers that are dead at t (CI <= 0.01 at t on every prompt) but
alive elsewhere still read the true stream, so they are switched off at t in every run, patched
or not (report the cost of that switch alone):
- the output divergence (for addsub: KL at the last position, and answer accuracy);
- two references: patching each reader's activation with its mean over the domain (the cost of
  losing everything the readers carry at t), and patching with the exact reads (should give 0;
  a check of the patching code).
The fraction of the mean-patch divergence that the reconstruction removes measures how much of
what the readers do the quantities capture. If it is low while the important residual looked
small, the CI threshold or the patching is wrong; if the important residual was large, return to
step 1. The mean patch also shows which sites matter for the output at all: at most sites a single
site's mean patch barely moves the output, and the test is informative only where it does.
Worst cases: rank sites by the reconstruction's divergence and by the fraction it leaves; at each,
rank readers by CI-weighted residual, plot their reads with the CI, and refine (back to step 2).
Patterns met so far in the fits (readers with the largest CI-weighted residual): readers
important on one or two values (single-value detectors: a one-hot over those values fixes them
but does not generalise and is reported separately), and windows of the variable whose height a
smooth curve under-fits (a place code with stated spacing).
The original model has no reader components and its dense weights read the whole stream, so a
closed test there must also replace the stream outside the readers' span (not implemented).

### Step 8: report the site
- A, in acceptance order: encoding, dims, increment, chance, excess, p-value, readers using it,
  directions D_q, and its geometry (for a circle the norm ratio and angle of its two directions;
  for classes the singular values of the class means).
- The hypothesis log with every revision and backtrack, and why.
- Held-out R^2; the residual and the important residual with their diagnostics.
- The patching results with both references.

## 4. Across sites
- Process sites in stream order, per token position, starting each site from the previous
  site's A as the first hypotheses (step 2.1).
- Track each quantity: the sites where it is accepted, its excess, and the overlap of its
  directions between consecutive sites (projected into the later reader span): passed on (same
  directions), moved (same feature map, new directions), erased (a write anti-aligned with D_q),
  derived or combined (a new feature map that is a function of earlier ones).
- A quantity accepted at a low-power site (k < 10) needs support from an adjacent site.
- Patch sites one at a time, and also all sites of a token position together: the joint patch
  tells whether errors accumulate.

## 5. Pitfalls seen so far (addsub)
- Inherited quantities must be re-tested at each site (drop-one excess and drop-one permutation
  p), or the accepted list only grows.
- A fixed hypothesis pool that contains quantities found earlier on the same sites does not make
  their acceptance an independent result.
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
- Many readers are important on only a few prompts; their residual elsewhere is irrelevant.
  This is why the CI is a stopping criterion and not a fitting mask.

## 6. Tools (addsub, `notes/arith_representations/quantities/`)
`v2.py` (site data, fits, chance levels, cross-validation over values, residual SVD),
`v2_layers.py` (all sites), `v2_feature_test.py` (permutation test of a candidate on a
residual), `v2_period20_test.py` (both-orders test), `v3.py` / `v3_run.py` (joint fits with
drop-one losses, the important residual, the protocol run at token a),
`v3_inspect.py` (worst sites: CI-weighted residual per reader), `v3_refine.py` (refinements at
given sites). The loop over 63 sites takes about a minute on a 256-core node. Patching:
`qagent/qpatch.py` (GPU). Run through sbatch, or on a pod with the compact token-a
files (the code reads `*_token_a.npy` when present).
