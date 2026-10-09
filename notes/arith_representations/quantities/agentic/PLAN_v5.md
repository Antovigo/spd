# V5: per-reader sparse quantities — plan

## Goal
In run 1 (V4 scoring, `qa.py`) every quantity of a site's list explains every reader of the site: the
stream in the readers' span is fit as Z ≈ mean + Σ_q Φ_q D_q (Φ_q: the N × d_q feature matrix of
quantity q, d_q its number of features; D_q: d_q × k_eff directions), so every reader r reads every
quantity unless its read direction happens to be orthogonal to D_q. V5 asks instead that **each reader
respond to as few quantities as possible**, and makes the score pay for every (reader, quantity) link.

## Notation
- N: training rows of the site (12,722 at t = 4); i: a training row.
- Q = [(name, formula)]: the agent's list, n_Q quantities; Φ_q: features of quantity q, standardised on
  the training rows; d_q: rank of Φ_q.
- r: a reader of the site (a component reading the stream there); y_r = column r of Y = Z W, its raw
  read (N values); CI_{i,r} ∈ [0, 1]: its causal importance on row i.
- ω_{i,r} = 1[CI_{i,r} > 0.01]: entry (i, r) is important; N_r = Σ_i ω_{i,r} over the training rows.
- Scored readers: N_r ≥ N_min = 20. Readers with fewer important training rows (some L18.mlp readers
  have one) get the intercept-only model, are not scored, and are listed in the report. Within a
  scored reader, a fold whose training part has fewer than N_min important entries, or whose test
  part has none, is skipped for that reader (in E_r and T_r alike, for every support).
- S_r ⊆ Q: the support of reader r (the quantities it reads).
- B: bits per percentage point of held-out unexplained variance (B = 3000 in run 1).

## Model
Per reader, a weighted ridge regression on its own support, on important entries only:
  y_r ≈ μ_r + Σ_{q ∈ S_r} Φ_q c_{q,r},   minimising Σ_i ω_{i,r} (residual)² (+ the usual small ridge),
with μ_r ∈ ℝ the intercept and c_{q,r} ∈ ℝ^{d_q}.

## Score (lower is better)
  J = Σ_{q used by ≥ 1 reader} formula_bits(q)
    + Σ_r [ log2(n_Q + 1) + log2 C(n_Q, |S_r|) + ½ log2(max(N_r, N_min)) · (1 + Σ_{q ∈ S_r} d_q) ]
    + B · 100 · (1 − R²_ho)
- formula_bits: unchanged from `qa.py` (log2 V per expression node, Elias-δ numbers);
- log2(n_Q + 1) + log2 C(n_Q, |S_r|): which quantities reader r reads (its support size, then which
  subset);
- ½ log2(N_r) per coefficient, now per reader: a quantity of d_q features read by m readers costs
  m · d_q coefficients (run 1: d_q · k_eff, shared by all readers);
- R²_ho = 1 − Σ_r E_r / Σ_r T_r on the inner folds (same value hold-out as run 1), counting important
  entries only: E_r = Σ_{held-out i} ω_{i,r} (y_r − ŷ_r)², T_r = Σ_{held-out i} ω_{i,r} (y_r − ȳ_r)², ȳ_r
  the mean of y_r on the fold's important training entries.

## Support selection (done by the tool, not by the agent)
T = Σ_r T_r does not depend on the supports, so given Q the score separates over readers except for
the formula bits. Per reader, minimise c_r(S) = support bits + coefficient bits + B · 100 · E_r(S) / T exactly
when 2^{n_Q} ≤ 256 (exhaustive over subsets), otherwise by stepwise search (a heuristic): forward (add
the quantity that lowers c_r most), then backward (drop the one whose removal lowers c_r most),
alternating until neither lowers c_r by more than 1e-9 bits; ties broken by the lower quantity index.
It terminates because c_r strictly decreases. Then an outer greedy deletion over whole quantities:
remove the q whose removal (re-selecting the affected readers) lowers J most, until no removal lowers
J (a local optimum; catches quantities that pay full formula bits for few readers).
Validation: on run 1's lists at the two test sites (n_Q = 6, 7) both searches are run and the gap in J
and the share of readers with different supports are reported. Implementation: per reader and fold, Grams of the full feature matrix on
that reader's important rows (Σ_q d_q ≲ 150 columns), so each step is a small solve.

Selection uses the same inner folds that score it, so R²_ho is optimistic; the final test split
(never shown to the agent) stays honest.

## Agent tool (`qa5.py`, same commands as `qa.py`: `sites`, `show`, `fit`)
`fit` reports, besides J, model bits and held-out R² (important entries, and on all entries for
reference):
- per quantity: formula bits, number of readers reading it, share of the CI-weighted explained
  variance, drop-one ΔJ (with re-selection);
- per reader: its support, the mean and CI-weighted mean |S_r|, a histogram of |S_r|;
- figures: a reader × quantity incidence matrix, plus the run-1 read/residual/residual-SVD panels.
Attribution (non-additive, labelled so): cell (r, q) = (SSE_r(S_r \ q) − SSE_r(S_r)) / (SST_r −
SSE_r(S_r)), SSE_r the in-sample important-entry squared error refit without q, SST_r the important
entries' squared deviation from their mean. Per-quantity aggregates weight reader r by its CI mass
m_r = Σ_i ω_{i,r} CI_{i,r}; CI-weighted mean |S_r| = Σ_r m_r |S_r| / Σ_r m_r.
`INSTRUCTIONS_v5.md`: run 1's instructions with the score section rewritten, and one new aim: prefer
quantities that a reader can use alone (a reader needing three quantities to be explained suggests a
missing quantity that combines them).

## Calibration of B
`calibrate.py` with the V5 J at t = 4 L18.attn and L18.mlp (greedy forward over the generic family,
B ∈ {1000, 3000, 10000}). Reference: run 1's final lists at the same sites with every reader reading
every quantity (S_r = Q, the dense model), scored with exactly the V5 metric (important entries, same
fold means, same reader exclusions). Keep B = 3000 unless the V5 held-out R² falls more than 0.02 below
that dense reference; otherwise pick the smallest B within 0.02 of it. All-entries R² reported
separately.

## Test: t = 4, L18.attn and L18.mlp
1. Baseline: run 1's final lists at these two sites, dense (S_r = Q) and rescored by V5 (supports selected automatically):
   J, test R² (important entries and all entries), mean |S_r|, readers per quantity.
2. One blind Sonnet agent, both sites in stream order, ≤ 4 fits per site, V5 instructions; outputs in
   `agentic/run5/t4/<site>/`.
3. `evaluate5.py`: on the test rows, R² of the reads (important entries; all entries), mean and
   CI-weighted mean |S_r|, number of quantities; vs run 1 and vs the baseline.
The closed patching KL is not part of this test (needs the model on a GPU); a reader fit on its
important entries only may predict badly on its unimportant ones, which the alive-only patch also
writes — the all-entries R² flags that.

## Compute
The local SLURM queue does not start even 2-CPU jobs right now (priority queue behind GPU jobs), so
fits run on a RunPod pod as in run 1 (`qa_remote.sh` pattern); pod type and price confirmed with the
user before booking.

## Alternatives not taken
- Shared directions with per-reader zero constraints (Z ≈ Σ_q Φ_q D_q with D_q w_r = 0 for q ∉ S_r):
  cheaper in coefficients when readers outnumber k_eff, but a constrained joint fit; revisit if
  per-reader coefficients make the score dominated by coefficient bits.
- Group lasso path per reader instead of stepwise: smoother selection, same score; stepwise is
  simpler at n_Q ≲ 15.
