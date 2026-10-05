# Quantities at a residual-stream site: prototype of the hypothesis loop (L0, token "a")

Run p-ba5a0c05 (-05 addsub decomposition of Llama-3.1-8B), step 40000, alive-only model (every
alive component of `addsub-05-filter-last-pos-ceiling` on at 1, dead components and the weight
delta off). Outputs: `~/out/pod-backup/p-ba5a0c05/analysis/quantities/L0/<site>_<hypothesis set>/`.

## Notation
- a: the first operand, an integer 1..100; the token position is t = 1 (the "a" token). With
  causal attention every activation at t = 1 is a function of a alone, so a site is a matrix
  over the 100 values of a.
- Site: the alive readers of one read point that are active at t = 1 (output CI on the original
  model > 0.01 somewhere on the (a, b) grid at t = 1; b the second operand, 1..100).
  Site 1 = L0 attention input (stream position l = 0, right after the embedding): 8 readers
  (1 q, 4 k, 3 v). Site 2 = L0 MLP input (l = 1, after L0's attention): 115 readers (53 gate,
  62 up).
- A_r(a): reader r's inner activation in the ||V|| = 1 gauge. rho(a): RMS of the residual
  stream at the site (a positive scalar per value of a). Raw read: rho(a) A_r(a) / mean(rho), the
  dot product of the reader's gain-folded direction with the raw stream, which is exactly linear
  in what was written to the stream. The fits target the raw read.
- Quantity: a hypothesised function of a with an encoding, given as a feature matrix
  Phi_q (100 x d_q). Fixed quantity: a reader reads all d_q columns together (a smooth curve, a
  circle). Menu quantity: a reader picks individual columns (one residue class, one interval,
  one token value).
- [P]: 1 if P holds, 0 otherwise. "a mod m = r": the residue class of a modulo m.

## Method (`hyp_fit.py`, `hypotheses.py`, `iterate.py`)
Each reader x (100 values) is fit greedily: start from a constant; each move adds a whole
fixed quantity or one column of a menu quantity; a move is kept if it lowers the description
length

  DL = (N/2) log2(RSS / N) + sum of move costs,

with N = 100, RSS the residual sum of squares, a move costing the quantity's bits the first
time a reader uses it (4 bits less if another reader of the site already uses it: quantities are
declared once and read many times), log2(menu size) for a menu item, and (1/2) log2 N per
weight. Diagnostics drive the next hypothesis set:
- shared residual structure: top singular values of the (100 x readers) residual matrix vs a
  null permuting each reader's residual over a (`residual_svd.png` shows the patterns);
- per reader: strongest periodic component (DFT), best single threshold, best smooth trend,
  each against its permutation null (flagged at the 95% level);
- outlier tokens: values of a with residual > 3 robust SDs in several readers.

## What the loop found
**rho is itself a quantity.** rho(a) rises like log a from 0.0072 (a = 2) to about 0.0105
(`figures/rho.png`): the embedding norm encodes magnitude. Dividing by it hides or flips the
magnitude signal in post-norm activations (`figures/site1_raw_vs_post.png`: L0.k.c0's raw read
falls steadily with a, its post-norm activation is flat), hence the raw-read target.

**Site 1, L0 attention input** (final set `s1_v3`; raw-read variance explained 68%, median
reader R^2 0.90, `figures/site1_fit.png`):

| quantity | readers | note |
|---|---|---|
| magnitude: smooth curve in log a (6-dim spline) | 5 (k.c0, q.c21, v.c1, v.c14, v.c28) | v1 used log a and a; v.c1 decays to 0 at a ~ 35, which needed a free smooth curve |
| digit count: [a <= 9], [a = 100] | 5 (same) | |
| round numbers: [a mod 10 = 0], [a mod 5 = 0] | 4 | k.c25, k.c115: down-spikes at 55, 65, 90 |
| token identity, a few values (18, 55, 64, 65, 90) | 5 | idiosyncratic tokens shared across readers |

The unexplained 32% sits almost entirely in three k readers (k.c25, k.c43, k.c115) whose signal
at this position is mostly per-token jitter; the remaining shared pattern (a slow rise above
a ~ 70) is at the edge of the permutation null.

**Site 2, L0 MLP input** (`s2_v1` -> `s2_v2`; explained 98.6%, median R^2 0.991, residual top
singular value at the null, 19/115 readers flagged by the pattern tests, about the 17 expected
by chance):

| quantity | readers | encoding / support |
|---|---|---|
| token identity (single values) | 96 | many readers are single-value detectors, and residue / interval readers use a few token items to correct per-token amplitudes |
| place code of the magnitude: intervals and soft bumps | 48 + 42 | centres spread over the whole range 1..100 (counts per decade 8, 24, 22, 17, 15, 18, 22, 13, 16, 15) |
| residue classes | 41 | units digit (a mod 10 = r): every r = 0..9 read by 3-7 readers, often two digits with opposite signs in one reader (gate.c44: +7 / -4; up.c52: +3 / -6); multiples of 5 |
| divisibility by 3 (a mod 3 = 0, refined by mod 6, 9, 12) | 10 | found from the residual: gate.c109 had R^2 = 0 under v1 and a dominant period-3 component (DFT frequency 33) |
| residue class x linear ramp in a | 20 | small unique share (0.9%): the residue spikes' amplitudes drift with a |
| digit count | 13 | large unique share where used (44%) |
| magnitude: smooth curve | 8 | |
| soft thresholds | 9 | e.g. a > 10.5 (the one-/two-digit boundary) |

**Where site 2's structure comes from: the embedding, not L0's attention.** At t = 1 the stream
at site 2 is the token embedding plus the writes of L0's alive o components, and each reader's
raw read splits exactly into the two parts (virtual weights; the split reproduces the raw read
to 5e-5 relative error). For every one of the 115 readers the embedding part carries
essentially all the variance over a (median share 1.00) and the attention part essentially none
(median 0.00). So the single values, residue classes, divisibility by 3 and the place code are
already in the token embeddings; site 1's 8 attention readers simply read few of them, while the
115 MLP readers read many. The quantities are therefore properties of the embedding geometry,
and the same hypothesis set should explain the embedding rows E_a directly (a = 1..100).

## Principled summary of the quantities
1. **A quantity is a partition plus a geometry.** On a finite domain, a categorical quantity is
   a partition of the values of a (residue classes, intervals, digit count); its encoding is
   where the class representatives sit in the stream. Summarise it by the class-mean vectors
   m_c (mean of the raw stream over the values in class c) and their Gram matrix G_cc' =
   m_c . m_c' (c, c' classes): a simplex (one-hot), a circle, a line or a half-circle can be
   read off G, for instance by multidimensional scaling. Hypotheses that span the same function
   space are the same quantity: the units-digit indicators and the Fourier modes k = 10, 20,
   30, 40, 50 of a encode the same partition; only G distinguishes a circle from a simplex.
2. **Support.** Record which classes are actually represented (read with a non-zero weight, or
   with ||m_c|| above the within-class scatter). This captures partial representations (the
   half-circle case).
3. **Within-class scatter vs noise.** Per-token amplitude differences inside a class (the
   residue readers' spikes) are either noise or real token identity. Treat them as noise
   unless they are shared across readers beyond chance (the outlier-token and SVD tests), and
   then name the shared tokens explicitly.
4. **One currency for fidelity and compression.** Report, per site: bits of the summary
   (quantity definitions + per-reader items) against bits of the residual, alongside two
   baselines: "token identity only" (every reader is 100 free weights) and "constant". Two
   hypothesis sets compare on total DL; a new quantity is accepted only if it lowers DL beyond
   what revising an existing one achieves. That is the "add or revise" decision of the loop.
5. **Stopping rule.** Stop when the residual's top singular value is within the permutation
   null and the count of pattern-flagged readers is at the false-positive rate (both hold at
   site 2).
6. **Downstream check (not done yet).** Rebuild the stream from the summary (each quantity's
   class means) and check the readers' activations and the final KL through the rest of the
   model. A summary that keeps R^2 but loses behaviour is missing a quantity the readers'
   residual hides.

## Limits of this first pass
- The greedy search with per-reader menus can over-use token items (96 readers at site 2). A
  joint fit that ties the readers of one quantity to a shared class geometry (point 1) would
  replace many per-reader token corrections with one per-quantity representation.
- Hypotheses came from me looking at plots; the library (`hypotheses.py`) records each
  iteration with its rationale, which is the trace an automated agent should also produce.
