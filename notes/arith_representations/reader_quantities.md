# What the readers read: a label-free atlas of quantities

Code: `param_decomp/arith_repr/isa/` (`atlas.py`, `writers.py`). Outputs: `<run>/analysis/arith_repr/atlas/` (report in
`site/`, https://claude.ai/artifact/FHwWcPdX6cjPSech1iGR3Z).

## The problem

For every place where components read the residual stream (the q, k and v readers of an attention
layer, the gate and up readers of an MLP, at positions a, op, b and `=`), find a short list of
*quantities* that together explain the readers' inner activations: unlabeled variables with a value
on every prompt (a point on a circle, a sign, a corner of a simplex). For each quantity: where it is
written in the stream, which readers read it, and whether the same quantity is read elsewhere.

**Inputs.** The readers' inner activations on the 20,000 prompts (a, b in 1..100, both operations)
and the stream at their read point. Labels (a, b, op) are used only to name results afterwards.

**What the method relies on.**
- *Readers are linear in the stream.* Multiplying a reader's inner activation by the stream's RMS
  gives `h_c = x · (g ⊙ V_c)`, a fixed linear read of the raw stream.
- *Each reader reads very few quantities.* On the grids, almost every reader responds to one period,
  sometimes in both operands (plaids such as L18 up.c61) or as a product (checkerboards).
- *Most quantities are spheres.* A binary variable is ±1, a circle has a constant radius, a simplex
  puts its values at the same distance from the centre: inside its own subspace, every prompt has
  the same norm. After whitening, uncorrelated quantities (such as two periods) are orthogonal.
- *The components-only model is cleaner.* It runs only the CI-active components, without the weight
  delta (last-position KL 0.05). The weak "messy" readers of the original model read the same codes
  once the delta and the switched-off components are gone.

## Method (`atlas.py`)

- **A. Quantities at one read point.** Readers times the stream RMS; drop constant readers; rank by
  parallel analysis on the reader correlation; covariance-weighted whitening. Sphere pursuit
  (`pursuit.py`): starting from the strongest unexplained reader, fit for k = 1..4 the frame with
  the most constant squared norm, `J = Var(|z|^2) / (2k)` (0 on a sphere, 1 for noise). Frames are
  fitted on half of the distinct prompts and accepted (J < 0.5) and scored on the other half: at
  positions a and op the 20,000 prompts collapse onto 100 or 200 distinct streams, and a random
  lookup seen at 100 points otherwise yields near-perfect "spheres". Frames holding two independent
  spheres (squared norms not anti-correlated) are split. Each quantity: its coordinates, its stream
  pattern `E[(x - x̄) z]`, and each reader's share of loading energy in it.
- **B. Relations.** `R²(i | j)`: how much quantity i is a function of quantity j, by nearest-
  neighbour regression (`variables.cond_r2`). At L18, `R²(a mod 5 | a mod 10) = 0.84`, 0.23 the
  other way; derived codes are functions of pairs (a + b parity of the two units digits, 0.73).
- **C. Codes across read points.** Two quantities of equal dimension are the same code when all
  their canonical correlations are >= 0.8; a smaller quantity inside a larger one is recorded as
  contained.
- **D. Names, after the fact.** Variance explained by op, a, b, a + b, a - b and the result, and the
  dominant harmonic.

Synthetic check (`synth.py`): circles for a and b mod 50 and mod 10 and (a + b) mod 10 recovered as
k = 2 at subspace overlap 0.98-1.00, b's square wave, lin(a) and the checkerboard as k = 1 at 0.99;
a mod-5 simplex splits into corners. Built twice with different embeddings and readers, every circle
and the square wave link to their twin in stage C.

## Results

Components-only model (original in brackets): 832 (709) quantities at 256 read points, 616 (500)
codes, 20 (12) read at two points or more, mean share of reader energy explained 45 % (32 %). The
largest codes: the op flag (173 read points: positions op, b and =, L0-L31); a mod 100 at positions a
and op (L1-L15); b mod 100 and b mod 50 at position b (L2-L18); b's mod-20 square wave (L15-L20);
result mod 10 and result mod 2 at `=` (L19-L28). By layer at `=`: b's codes from L15 (after L15H13
copies b), a's from L16 (L16H21), the operands' periodic codes at L17-L18, a + b codes from L19, and
multi-harmonic a + b codes from about L24.

Log cosh ISA (`pipeline.py`, whiten + FastICA + dependence grouping) also recovers clean codes on the
components-only model (L18 at `=`, addition: 17 of 26 outputs explained by one label at η² >= 0.8,
circle-like kurtosis), but tiles the long-period codes into sparse bumps on the original model.
`directions.py` gives the stream directions of its outputs (patterns and filters).

## How the quantities are built (`writers.py`)

In the components-only model the stream at a position is exactly the embedding plus the masked writes
of the o and down components, `x = e + sum_c h_c m_c U_c` (checked to within 1 %, the fp16 storage).
Each quantity is a linear read `z = (x - mean) . F`, `F = (g * V) W` from its readers, so each writer's
share of its variance is exact: `sum_j Cov(h_c m_c, z_j) (U_c . F_j) / Var(z)`; the shares of the
embedding and of all earlier writers add up to 1 (0.999-1.001 over the 832 quantities).

Provenance of every writer with >= 15 % of a quantity (J < 0.35): held-out R2 of its masked activation
given the quantities it can read, in the form the component computes: linear for attention (values are
linear in the source streams, over every source position), degree 2 for MLPs (neurons are
silu(g) * u), on single quantities and the best pair. A flexible regression is uninformative: at
positions a and op any code that identifies a predicts everything.

- Op flag: at position b written by L0 head 10 (L0.o.c32, c0, 98 %), linear in the flag at op
  (R2 0.999); at `=` rewritten layer by layer by MLP components that read it (L14.down.c1, ...).
- b's codes: at position b from the embedding and the L0 MLP (b mod 100 at L2: 37 % embedding, 48 %
  L0 MLP); copied to `=` by L15 head 13 (81 % of b mod 50, 99 % of b's mod-20 square wave at the L15
  MLP input), linear in b's codes at position b.
- a's codes: copied to `=` by L16 head 21 (50-97 % of a's codes at the L16-L18 MLP inputs), linear in
  a's codes at positions a and op.
- Result: L19.down.c6 writes 52 % of the result's mod-20 code at the L20 input; its activation is a
  product of b's and a's mod-20 square waves (degree-2 R2 0.83 for the pair, 0.32 and 0.08 alone).
  L17.down.c67 computes the result's parity from a's and b's parities (0.89). L18.down.c4 writes 61 %
  of the result's parity, but the quantities at L18's input predict it poorly (0.16): the operands'
  parities are not among them. Later MLPs refresh the result codes from themselves (L21.down.c1, 0.90).

## What did not work, and why

- *ISA on the original model:* log cosh favours sparse directions, and the original model's
  background makes bumps sparser than circle axes.
- *Varimax plus grouping by anti-correlated squares:* nested periods of one operand (the square of a
  period-T coordinate has period T/2) chain into large groups.
- *Mutual information for relations:* on near-deterministic data a 3 % shared impurity gives 0.3
  nats; `R²` measures the size of the dependence instead.
- *Looser linking across read points* (min(k, k') correlations, or all but one dimension): chains
  unrelated codes through axes many frames share (a's magnitude, the op flag).
- *Masked activations* (inner activation times the CI mask): gated, prompt-selective signals that are
  no longer linear reads of the stream; ISA outputs become heavy-tailed mixtures.

## Open problems

- Magnitudes are not spheres; they are found only inside frames with other quantities.
- Plaid and product readers are split among several quantities.
- "A function of" misses fast-oscillating relations when noise blurs neighbouring values.
- Codes transformed nonlinearly between read points, or read together with another quantity at one of
  them, show up as separate codes.
