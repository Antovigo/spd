# Finding the quantities read at residual-stream sites: instructions for the agent

You identify the quantities represented at each read site of a decomposed model, one token
position at a time, by guessing them, fitting them, reading what they leave, and revising. A site
is finished when what is left does not matter causally; the result is validated by patching the
reconstruction into the model. The method is task-agnostic; the tools below are written for the
addsub decomposition (run p-ba5a0c05).

## 1. Definitions

- Prompt: indexed by i; described by task variables (addsub: op in {+, -}, operands a and b,
  integers 1..100; derived: s = a + b, d = a - b, the result r = s or d by op, units digits
  u_a = a mod 10, u_b = b mod 10).
- Token position t (addsub: 0 <BOS>, 1 a, 2 op, 3 b, 4 =). Domain of t: the distinct inputs
  activations at t can depend on (causal attention: variables of tokens at or before t). Addsub:
  t = 1: a (100 values); t = 2: (op, a) (200); t = 3, 4: (op, a, b) (20000). D = domain size.
- Site: the readers of one residual-stream read point (attention input of block c: q, k, v;
  MLP input: gate, up) that are causally important somewhere at t. n readers.
- x(i) in R^4096: the raw residual stream at the site (alive-only model: all alive components on,
  dead ones and the weight delta off). V~_r: reader r's unit read direction with the norm's gain
  folded in. Q (4096 x k): orthonormal basis of the readers' span. Z = X Q (D x k), W = Q^T [V~_r]
  (k x n). Raw reads Y = Z W; the readers' activations are Y / rho, rho the stream's RMS. Fit raw
  reads; rho is a function of the prompt too and is reported, not fit.
- Quantity q: a feature map phi_q from the domain to R^{d_q} (its encoding: one direction for a
  scalar, cos and sin for a circle, class indicators for a partition, one direction per value for
  a one-hot, possibly restricted to a support or gated by another variable), with directions D_q
  (k x d_q). Model: Z = 1 mu^T + sum_q Phi_q D_q^T + E. Every reader reads the same directions.
- Joint fit: ridge least squares of the centred Z on all accepted features together (ridge 1e-3 of
  the mean feature variance; without it overlapping partitions are collinear on some training
  folds and held-out predictions explode). Scores use the reader metric (variance of Y = Z W).
- Increment of q given the accepted set A: the variance it adds to the joint fit, as a fraction of
  the total. Chance level: d_q / df of the residual before adding q (df = D - 1 - dims of A).
  Excess = increment - chance. Drop-one loss: what q's removal from A costs.
- Permutation p of q: fraction of 100 permutations of phi_q over the domain whose increment
  reaches the real one.
- Held-out R^2: the fit made without a fold of domain rows, scored on that fold. The folds depend
  on the position (section 4).
- CI: the decomposition's causal importance of reader r on prompt i at t (addsub: the CI filter's
  output CI on the original model). Important entry: CI > 0.01. Important residual share: the
  residual variance on important entries over the read variance there, from held-out
  predictions. CI-weighted residual of reader r: sum over the domain of CI x residual^2. The CI
  never enters the fit: many readers are important on a handful of prompts only.

## 2. What you keep

- The accepted list A per site, in acceptance order, with the reason for each quantity.
- A log of every hypothesis tried and the decision (accepted, revised into ..., rejected, removed
  because ...). Never delete from the log.
- The current residual (A refit jointly after every change).

## 3. The loop at one site

0. **Check.** Y = Z W reproduces the stored activations x rho (relative error < 1e-3). Note n, k,
   D, rho over the domain, and how many domain rows each reader is important on. k < 10: a
   low-power site; accept a quantity there only if an adjacent site accepts it too.
1. **Look at the residual** (`qinspect.py`): the top singular functions of the residual over the
   domain against a permutation null, read as tables of values (largest positive and negative,
   with their (op, a, b)) and as plots against each variable; which variables still explain
   residual variance above chance; per reader, read, fit, residual and CI. Look at the readers
   with the largest CI-weighted residual first. Activations on unimportant prompts are hints, not
   targets: a reader that traces a clean sine of a over the whole domain but is important on a
   slice suggests the sine, which is then fit on the whole domain.
2. **Guess one quantity**, specific, with its evidence. Order of preference: quantities accepted
   at the previous site (passed on); functions of accepted quantities (a sub-range, a threshold,
   half of a circle, a coarser partition, a gated version); combinations of two accepted
   quantities; new quantities read from the residual. Prefer the lowest-dimensional encoding (one
   direction, then a circle, then classes, then a one-hot). Do not propose broad bases (a
   24-function spline over the whole range, all residue classes at once): accepted with a small
   gain per dimension, they absorb the specific quantities they span and are inherited
   everywhere. A place code is specific when its spacing and width are stated. A frequency peak
   or a correlation is not a hypothesis: name the function.
3. **Fit it** jointly with A. Record increment, chance, excess, p, held-out R^2, readers with at
   least 10% of their variance from it, important residual share.
4. **Judge it by what it leaves.**
   - Reject: excess <= 0, increment < 0.2% of the reads' variance, or p >= 0.01.
   - Revise: structure tied to q's variables remains (left-over harmonics: wrong shape; structure
     on part of q's range: wrong support; a smooth function of q's value: wrong encoding). Fit the
     variant in place of q; keep the one leaving less structure at a higher excess per dimension.
     Two plausible encodings: fit each first and see what the other adds (both-orders test); keep
     the one after which the other adds nothing; both add: keep both; neither: fewer dimensions.
   - Backtrack: q makes an earlier quantity redundant (drop-one loss at or below chance; several
     at once: the most dimensions first): remove it and log why. Inherited quantities are
     re-tested at every site (drop-one excess and drop-one permutation p).
   - Accept otherwise, and go back to 1.
5. **Generalisation.** Held-out R^2 of A must not fall when q is added. A quantity read off the
   residual of some data is tested on data it was not read from before acceptance.
6. **Stop** when the held-out important residual share is below 5% or without structure (top
   singular value within the permutation null, no variable explaining it above chance), or when
   the last guesses were all rejected. Report what remains (important and unimportant parts;
   value-specific variance as such).
7. **Patch** (`qpatch.py`), per site and all sites of the position together, in the alive-only
   model (reader activations replaced). The test must be closed: the stream at t may reach the
   rest of the network only through the reconstruction. The readers dead at t (alive elsewhere,
   CI <= 0.01 at t on every prompt) still read the true stream, so they are switched off at t in
   every run, patched or not; report the KL of that switch alone, and the patched KL against
   both references (all alive readers on; dead-at-t readers off). The original model is not
   patched: its dense weights read the whole stream, and a closed test there would also have to
   replace the stream outside the readers' span. Compare with
   the mean patch (the cost of losing what the site carries at t; at many sites it is near 0, and
   the test is informative only where it is not) and the exact patch (a check, KL ~ 0); use the
   held-out reconstruction for the strict version. Worst sites: rank by reconstruction KL and by
   the fraction of the mean-patch KL it leaves; inspect their readers by CI-weighted residual and
   refine (back to 2). Patterns met in the fits at t = 1 (readers with the largest CI-weighted
   residual): single-value detectors (readers important on one or two values: a one-hot over
   those values fixes them and is reported as value-specific) and windows under-fit by a smooth
   curve (a place code with stated spacing).
8. **Report** per site: A with encodings, dims, increments, chance, excess, p, readers using each,
   geometry (a circle's norm ratio and angle; class means' singular values); the log; held-out
   R^2 under each scheme; the important residual; the patching results.

## 4. Positions and held-out data

Held-out rows must be ones whose values the quantities should generalise to, and rows that share
a domain value must be held out together.

| t | domain | primary folds | diagnostics |
|---|---|---|---|
| 1 (a) | a | values of a | - |
| 2 (op) | (op, a) | values of a (both ops together) | - |
| 3 (b), 4 (=) | (op, a, b) | (a, b) pairs (both ops of a pair together) | values of a; values of b |

- t = 1, 2: a held-out value of a is new to the fit; structured quantities of a must predict it,
  value-specific ones cannot. op takes two values and is never held out.
- t >= 3: the primary scheme holds out (a, b) pairs, so each held-out pair's a and b values were
  seen in other pairs: quantities of a, of b and of their combinations (a + b, carry, ...) are
  tested on new combinations. The diagnostics hold out all pairs with given values of a (or b):
  a quantity of a must then generalise to unseen values of a. A quantity that holds under the
  pair scheme but fails under a-values is value-specific in a.
- Permutation tests permute the features over the domain rows.
- A site whose reads are constant over the domain (L0's attention input at a token that is the
  same in every prompt, e.g. "=") has nothing to fit; `qloop.py` records it as constant.
- Process sites in stream order within a position. Quantities accepted at earlier positions
  (e.g. a's quantities at t = 1) are candidates at later positions.

## 5. Tools (`notes/arith_representations/quantities/qagent/`; data under
`~/out/pod-backup/p-ba5a0c05/analysis/quantities/qagent/`)

- `qprep.py <t>`: per-site data of position t from the full activation files (cluster job).
- `qdata.py`: `load(t)` -> Position (domain variables, sites with Z, W, CI, cols, Q);
  `dom_index`.
- `qfeat.py`: `pool(pos)`, the standard candidates (operands, op, results, carry, borrow, gated
  versions), and `Cand(name, var, Phi)` for your own.
- `qfit.py`: `Fitter(site, cands, folds)` with `increment`, `drop_one`, `perm_p`, `heldout_r2`,
  `reconstruct`, `important_share`, `ci_weighted_residual`, `add` (register a new candidate);
  `folds(pos, scheme)`, `primary_scheme(t)`.
- `qloop.py <t> [--sites ...] [--extra mod.fn] [--tag x]`: the automatic part of steps 2-6 over
  the sites of a position (writes `qloop_t<t>_<tag>.json`, `recon_t<t>_<tag>.npz`).
- `qinspect.py <t> <site> [--tag x]`: step 1 / step 7 inspection (tables and a figure).
- `qpatch.py <t> [--tag x]`: step 7, the closed test in the alive-only model (GPU).
Fits run on CPU (a 256-core node runs a position's 63 sites in minutes at t = 1); patching needs
a GPU. Run through sbatch, or on a pod holding the per-position data files and the model weights.

## 6. Pitfalls seen so far

- Report excess over chance, not raw increments; with D = 20000 almost everything is
  significant, so the minimum increment (0.2%) carries the decision.
- Broad bases and order dependence: see step 2 and the both-orders test.
- Post-norm activations hide what the norm carries (at L0 the stream's RMS grows like log a).
- Small reader spans (k < 10) give noisy fits; directions fit from few domain values in 4096
  dims are noise (no cross-validation in the full stream without a ridge).
- Per-reader item selection is a per-reader one-hot model and inflates fits: quantities must be
  shared directions.
- A residual frequency peak suggested residues mod 4, 6, 7, 8, 9 at t = 1; all were rejected.
  The singular function read as a table of values gave the right hypothesis (2-adic valuation).
- In-sample important residuals are small by chance with 25-30 dims on 100 values: use held-out.
- A fixed pool containing quantities found earlier on the same sites does not make their
  acceptance an independent result.
