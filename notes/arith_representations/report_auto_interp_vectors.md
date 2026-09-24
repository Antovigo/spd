# How Llama-3.1-8B does `a ± b =`, read off the components' U and V

Decomposition `p-ba5a0c05` step 40000, filter `addsub-05-filter-last-pos-ceiling` (11,604 alive
rank-1 components). Descriptive only: no ablations or patching. Every claim below comes from
two measurements:

- **V**: what a component reads. This is its inner activation `h_c = x_norm · V_c` on the
  original model's activations.
- **U**: what it writes. This is `U_c` projected on the codes present in the residual stream
  right after its add.

CI values were not used; everything is computed from V, U and the original model's activations.
The activation-level auto-interp (`report_auto_interp.md`) is the companion to this report. Where
the two overlap they agree, and this report adds the geometry.

Code: `param_decomp/arith_repr/vectors/`. Data and full-size figures:
`<run>/analysis/arith_repr/vectors/`. Working notes: `scratchpad_auto_interp_vectors.md`.

![schematic](figures_auto_interp_vectors/schematic.png)

## TL;DR

1. **Storage.** Each operand is stored at its own token as a stack of periodic codes plus a
   magnitude direction:
   - codes: `a mod 2, 4, 5, 10, 20, 25, 50, 100`, plus `lin(a)`;
   - every code is a circle whose residues sit in order;
   - the token embedding already carries `lin(a)` and `a mod 5`;
   - the L0 MLP builds the other codes with residue-detector components. Each detector's U is
     the spoke of its residue on the circle (for example, each L0 units-digit detector's U
     points at 36° × its digit on the `a mod 10` circle).
   - After that the codes stay in place (step-to-step |cos| ≥ 0.9).
   - `a` and `b` are written in **largely the same directions**. The attention v components read
     `a` at position a and `b` at position b with the same phase.
2. **Moving.** Attention copies the operands to `=`:
   - `b` via L15H13, `a` via L16H21;
   - OV is an identity on the code: the median phase shift from source to `=` is 0.05 of a
     period;
   - routing is set by one (q, k) component pair per head. The k components read slot features
     ("second operand", "first operand") that MLPs at L6-L14 build at the operand positions.
3. **Computing.** The result code is made at `=` by the L16-L18 MLPs:
   - gate and up read `a` and `b` codes at the **same harmonic k**;
   - `silu(gate)·up` contains the cross term `cos k(a) · cos k(b)`, which is a code of
     `a + b` (and of `a − b`);
   - the a×b product predicts the newly created result code with cosine 0.99-1.00 (add) and
     0.82-0.91 (sub);
   - U writes it into a result circle, in phase for 40 / 40 of the (unit, harmonic) pairs.
   - From L19 on, the MLPs receive the result code (40-80 % of their input) and re-write and
     refine it.
   - Period by period (exact bookkeeping over all neurons, section 3.1):
     - mod 100, 5, 10, 50 and 20 are made at L16-L19 from a-code × b-code across gate and up;
     - parity is made inside silu(gate) by a single neuron (L18 unit c4);
     - mod 25 and mod 4 never come from the operands: later MLPs make them by multiplying result
       codes (mod 50 × mod 50 → mod 25; mod 5 × mod 20 → mod 4);
     - from L21 every period is also re-derived that way.
4. **Output.** Every result-code harmonic is phase-aligned with the unembedding's own Fourier
   plane of the number tokens:
   - each plane votes for the tokens `n ≡ res (mod 100/k)`, and the votes add up only at
     `n = res`;
   - the alive writers' direct votes sum to 3.7 logits (add), from hundreds of small
     contributions spread over L17-L29;
   - the hundreds digit comes from a separate magnitude path: gate reads of `lin(a) + lin(b)`
     go into neurons that switch at `a + b ≈ 100`, and their U points along
     `W_U[100..199] − W_U[0..99]`.
5. **Operation.** The op token reaches `=` through L0H23 / L1H6 / L2H2. There it is a large
   flag direction: 45-70 % of |x|, re-written layer by layer by one-bit add-only / sub-only
   MLP components whose U lies along the flag. The flag gates the L15 MLP:
   - for each harmonic, 2-3 neurons (7 in total) write a new "sine axis" of b, reading b at
     the quarter-period phase, with a sign set by the op;
   - the op reaches these neurons either through the gate (one neuron switched on by add,
     another by sub) or through up (a signed op read multiplying silu(gate(b))): SwiGLU's
     gate × up product is what multiplies b's code by ±1;
   - flipping only the sine axis while the cosine axis stays gives exactly `b → −b`
     (reflection centre 0 at every harmonic), so from L16 on the readers see `b` **mirrored**
     on subtraction while `a` is unchanged;
   - the same L16-L18 "adder" units then compute `a + (−b)`.

## Method (one paragraph per tool)

**Line codes.** Prompts form the full 100 × 100 `(a, b)` grid per op (sub includes `a < b`). For
any vector-valued activation `x(a, b)` and each quantity `q ∈ {a, b, a+b, a−b}` I take the line
Fourier coefficient `F(q, k) = mean x · e^{−2πi k q/100}`, `k = 1..50`. This is the
(k,0) / (0,k) / (k,k) / (k,−k) entry of the 2-D DFT of the grid.

- Harmonic `k` has period `100/gcd(k,100)`: k = 10 is the units digit, k = 20 is mod 5, k = 2
  is mod 50, k = 1 is mod 100, k = 50 is parity.
- For a code in the raw stream, `F` is a pair of directions (a plane). Where they have equal
  norm and are orthogonal, the class means lie on a circle in residue order.
- The linear parts of `a` and `b` are split off first; their 1/k Fourier tail would otherwise
  fill k = 1-3.

**Reading V.** `R_c(q, k)` is the same DFT of the scalar `h_c`. Its argument gives the residue the
component prefers: `q* = −arg R · 100/(2πk)`. Its power gives what the component reads.

**Reading U.** `U_c · F_t(q, k)` projects U on the code of the stream point `t` right after the
add. Its argument gives the residue U "looks like" in that code.

The component's own contribution to the code is `R_c U_c`. Its alignment with the stream code
says whether it reinforces the code, erases it, or turns it (`transfer.npz`). For each
sublayer, the sum over components can be compared with the actual change of the code.

**MLP units.** A down component's V is neuron-sparse (participation ratio ≈ 5.6 neurons), while
gate/up U are dense. So a down component is a handful of neurons, and the gate/up components are
shared read directions. For the top 8 neurons of every down component I took the line spectra of
the gate pre-activation `g`, `up` (both linear reads of the stream), `silu(g)` and
`act = silu(g)·up`.

**Attention.** OV coupling of a v component (source) and an o component (`=`) through head h is
`U_v[kv(h)] · V_o[h]`. QK: for query head h at `=` and source s,
`mean h_q · mean h_k · ⟨RoPE₄ U_q[h], RoPE_s U_k[kv(h)]⟩ / √128`.

## 1. How values are stored, and how the representations are built

![storage](figures_auto_interp_vectors/storage_map.png)

Share of the raw residual variance carried by each code, at every stream point (log scale).
"mod 100 (+ identity)" includes all token-identity variance, so it is large by construction.

- **Position a / position b.** The token embedding already carries `lin(a)` and every period.
  The L0 MLP step makes the largest change to the operand codes of the whole network (figure
  below). After it the codes are flat to L31, with modest re-writes by MLPs (L11-L15 refresh
  mod 5 / mod 10).
- **Position `=`.**
  - `b` arrives at L0 (the previous token; all L0/L1 components at `=` read `b` only);
  - `a` arrives by L1-L2;
  - both stay at 1-10 % of the variance until L15-L16, when the dedicated copies arrive;
  - the result codes appear at 16m-18m and dominate from L19;
  - on **both** ops a low-k `a−b` code (period 50-100) builds up at `=` from L2. This is a
    magnitude comparator (sign / size of `a − b`), not `a − b mod 10` (that code only appears
    at 17m-19m on sub).

![code steps](figures_auto_interp_vectors/code_steps.png)

Which sublayer writes each code. Blue bars are attention steps, orange bars MLP steps. Numbers
are the share of the change explained by the alive components' own `R_c U_c`: 0.8 for the L0 MLP
at the operands, 0.78-0.93 for the result code at L16-L25.

**Residue detectors write the spokes of their residue.** The L0 down components at position a
are one-residue detectors: their `h_c` peaks at one residue of `a mod 10`, `mod 5` or `mod 50`.
Their U, projected on a's code plane at the next stream point, points at **that residue's
position on the circle**. A units-digit detector for digit d points at 36° × d. In the figure, the 10 L0 down
components with > 15 % of their variance on `a mod 10` cover digits 0-5 and 7; the detectors of 6, 8
and 9 have less than 15 % of their variance on this harmonic and fall under the cut. So adding
`h_c · U_c` moves the representation toward the class mean of the residue that switched the
component on.

![spokes](figures_auto_interp_vectors/units_spokes.png)

The same holds across the network: for each writer and each harmonic its inner carries (> 15 %
of its variance), I compared the phase of its write `R_c U_c` with the stream code.

| writers | (comp, k) pairs | in phase (< 60°) | opposite (> 120°) |
|---|---|---|---|
| down, position a, L0-4 | 255 | 0.93 | 0.02 |
| down, position a, L5-14 | 479 | 0.88 | 0.06 |
| down, position a, L15-31 | 964 | 0.80 | 0.09 |
| down, position b, L0-4 / L5-31 | 234 / 749 | 0.94 / 0.91 | 0.02 / 0.03 |
| down, `=`, a+b (add), L16-18 / L19-24 / L25-31 | 40 / 335 / 330 | **1.00** / 0.93 / 0.94 | 0.00 / 0.03 / 0.01 |
| down, `=`, a−b (sub), L16-18 / L19-31 | 27 / 255 | 0.85 / 0.87 | 0.11 / 0.05 |
| o, L15 (b at `=`) / L16 (a at `=`) | 67 / 142 | 0.99 / 0.89 | 0.01 / 0.03 |

Almost every writer is a "spoke": it writes the code of the residue that its V selects, in the
coordinates the stream already uses. There is very little erasing (< 10 %).

**Shared directions for the two operands.**

- The v components read the same phase at both operand positions. Example: L15 v c17 reads
  `a mod 5` peaking at 19.4 at position a, and `b mod 5` peaking at 19.4 at position b.
- The operand code planes at the two positions have |cos| ≈ 1 at the embedding and 0.35-0.6
  at L5-L10.
- They climb back to **0.6-0.85 at L12-L17**, right where L15H13 and L16H21 read them (left
  panel below). This looks like a shared "operand format" that the copying heads can read with
  one V.

![geometry](figures_auto_interp_vectors/code_geometry.png)

## 2. How attention moves information between positions

**What is moved: OV is an identity on the code.** For the 100 (L15H13) and 130 (L16H21)
strongest (v, o) pairs, I compared two phases at the dominant harmonic k:

- the phase of `b` (or `a`) that the v component reads at the source;
- the phase that the o component reads at `=`.

The median shift is 0.05 of a period (71 % / 78 % of pairs within 0.1). The heads transport each
Fourier component unchanged: `b mod 5` peaking at 19.4 becomes `b mod 5` peaking at 19.7 at `=`.

![ov](figures_auto_interp_vectors/ov_phase.png)

**Where it is moved from: QK from the q and k vectors.** Each head's alive attention logit is
dominated by one q × k pair:

| head | pair | key feature (k component's mean inner by position BOS / a / op / b / =) | effect |
|---|---|---|---|
| L0H23 | q c17 × k c6 | 0 / 0 / 15.2 / 0 / 5.1: "is the op token" | `=` reads op (logit +2.1) |
| L15H13 | q c138 × k c48 | 0 / 0 / 0.4 / −21 / 0.5: "second operand" | `=` reads b (+4.7 add, +4.0 sub) |
| L16H21 | q c136 × k c5 | 0 / 51.6 / 0.5 / 22.8 / 0.1: "operand, first > second" | `=` reads a (+12.3 vs +4.9 on b) |
| L18H30 | q c104 × k c95 | 0 / 45.5 / 0 / 43.5 / 0: "operand" | both operands |
| L17H9 | q c127 × k c9 | 0 / −43.5 / −45.7 / −43.3 / −13.2 | avoids a / op, BOS sink |

![qk](figures_auto_interp_vectors/qk_routing.png)

The slot features are built earlier by MLPs at the operand positions (writer U · reader V ×
mean inner):

- **"Second operand" (L15 k c48) at position b** comes from L14 down c30, L13 c14, L9 c22 and
  L7 c2, among others. It is exactly zero at position a.
- **L16 k c5** comes from L6-L15 downs at both positions, and is larger at a.

On subtraction the a-copy is weaker (L16H21: a 3.6 vs b 2.5). This matches the activation-level
finding that copies split when `a < b`.

## 3. How the result is computed from the operands

![mlp creation](figures_auto_interp_vectors/mlp_creation.png)

Result-line share of what the MLP neurons at `=` are handed (g and up, both linear in the stream)
versus what they hand to the down components (`silu(g)·up`), per layer:

| layer | result code in g, up | result code in silu(g)·up | share of it newly created | a×b fit (add / sub) |
|---|---|---|---|---|
| L16 | 0.00 | 0.14 | 0.98 | 1.00 / 0.84 |
| L17 | 0.03 | 0.16 | 0.64 | 0.99 / 0.82 |
| L18 | 0.04 | 0.30 | 0.91 | 0.99 / 0.91 |
| L19 | 0.40 | 0.57 | 0.20 | 0.77 / 0.66 |
| L20-L30 | 0.5-0.8 | 0.77-0.91 | re-written | < 0.45 |

**Mechanism (L16-L18).**

- Each unit's gate and up read an `a` code and a `b` code **at the same harmonic k** (see the
  grids below):
  - gate = `α cos k(a − φa) + β cos k(b − φb)`, and up likewise;
  - the product `silu(g)·up` contains `cos k(a−φa) · cos k(b−φb)`, which is
    `½ [cos k(a+b−φa−φb) + cos k(a−b−φa+φb)]`.
- The line-product prediction (`s_a u_b + s_b u_a` plus the cross term inside silu) matches the
  new result-line coefficient in phase and amplitude: cosine 0.99-1.00 on add.
- The phases add: median |peak_out − (peak_a + peak_b)| is 0.04 of a period. Examples:

  | unit | a phase | b phase | predicted a+b phase | measured a+b phase |
  |---|---|---|---|---|
  | L16 down c10 (k = 1) | 81 | 32 | 13 | 12 |
  | L18 c23 (k = 1) | 56 | 57 | 14 | 8 |
  | L18 c12 (k = 2, mod 50) | 48 | 23 | 21.4 | 23.1 |
  | L18 c18 (k = 5, mod 20) | 11.0 | 19.2 | 10.2 | 10.6 |
  | L18 c32 (k = 10, mod 10, sub) | 9.4 | 4.95 | 4.4 (a − b) | 4.4 |

- L18 down c4 is the parity unit (k = 50: a product of `(−1)^a` and `(−1)^b`).
- The U of these units writes the new code in phase with the result circle (40 / 40 pairs,
  section 1 table). Their outputs form the first result code: 0.78-0.92 of the result-code
  change at 16m-18m is their own `R_c U_c`.
- The top writers are L16 down c10 / c29 / c25; L17 c11 / c22 / c67; L18 c4 / c12 / c23 / c21;
  then L19 c0 / c24 / c11 / c32 and L20 c3 / c6 / c2.

![units](figures_auto_interp_vectors/result_units.png)

(a, b)-grids of the top neuron of four units (rows):

- **L16-L18 units.** Gate and up are sums of horizontal (a) and vertical (b) stripes; the
  product is a lattice or blob, i.e. a conjunction of a-window and b-window.
- **L19 c0.** Gate and up already carry diagonal stripes (the result code arrives as input);
  the product stays diagonal (pass-through / sharpening).
- **Sub** (4th column): the same neuron's output follows `a − b` (diagonal).

The product also makes `a − b` harmonics on add (the second term of the identity). These are
smaller (diff share 0.04-0.24 per unit) and do not accumulate across units.

**L19-L30** keep re-writing the result: 75-90 % of their inner variance is on the result line.
The newly created part is no longer an a×b product; it mixes result harmonics (e.g. k = 1 → k =
2 and 3). This matches the earlier finding that the long periods fan out from Fourier circles
into higher-dimensional codes from L26.

### 3.1 Period by period: which MLP makes which period, and how

**Exact bookkeeping.** On the full grid the neuron output `act = silu(g)·u` has, as its 2-D DFT,
the circular convolution of the DFTs of `s = silu(g)` and `u`. So every neuron's coefficient on
the result line (`(k, k)` for a+b, `(k, −k)` for a−b) splits exactly into:

| term | what it is |
|---|---|
| **X** | `s(k,0)·u(0,±k) + s(0,±k)·u(k,0)`: an a-code times a b-code at the same harmonic, one from the gate and one from up |
| **M** | `Σ_p s(res p)·u(res k−p)`: two result codes multiplied (harmonic mixing) |
| **P** | `s̄·u(res k)`, and the part of `ū·s(res k)` that is linear in the gate: a result code the MLP already receives, passed through |
| **Sx / M (inside silu)** | the rest of `ū·s(res k)`, from a quadratic fit of silu. Its a×b part is **Sx**; its result × result part is counted in M |
| **O** | everything else |

The layer's result write `T(k) = W_down · act(res k)` is summed over all 14,336 neurons (exact).
Each term is credited by its projection on `T(k)`; the shares add to 1.

![period map](figures_auto_interp_vectors/period_map.png)

Each dot is one MLP (x) and one result period (y). Area is the size of the write (share of the
stream variance at `=`); colour is the dominant term. Code:
`param_decomp/arith_repr/vectors/mlp_periods.py` (bookkeeping) and `periods_summary.py`
(table, neurons, partners).

**Addition, period by period.** Units are named by their down component. Gate, up and down share
an index from the neuron-aligned init. Shares are of the layer's write at the period's main
harmonic.

| period | made at | how | main units (share of the write) | what the unit reads |
|---|---|---|---|---|
| mod 100 (k1) | L16 (X 0.82); L17-18 (X ≈ 0.4, the rest passed through) | gate × up | L16 c10 0.39, c29 0.23, c43 0.14, c106 0.12; L18 c23 0.81 | c23: gate reads a @53, b @54; up reads a @50, b @51 (all k1) |
| mod 5 (k20) | L16 (X 0.95), L18 (X 0.61) | gate × up | L16 c76 0.25, c94 0.24, c4 0.15, c127 0.14; L18 c26 0.44, c24 0.27, c80 0.18 | c26: gate a @3.5, b @3.3; up a @1.5, b @1.5 |
| mod 10 (k10) | L17 (X 0.91), L18 (X 0.75) | gate × up | L17 c52 0.25, c39 0.19, c49 0.17, c86 0.16; L18 c32 0.37, c88 0.16, c65 0.15, c62 0.14 | c52: gate a @2.4, b @2.4; up (c118) a @3.0, b @2.8 |
| mod 50 (k2) | L18 (X 0.97; the largest creating write of the network, 3.9 % of the variance) | gate × up | c12 0.27, c21 0.17, c22 0.14, c27 0.14 | c12: gate reads a @2.5; up reads b @21.6 |
| mod 20 (k5) | L18 (X 0.98), L19 (X 0.88) | gate × up | L18 c18 0.54, c16 0.24, c59 0.18; L19 c6 0.97 (one neuron) | c6: gate a @19.1, b @19.4; up a @19.5, b @19.7 |
| mod 2 (k50) | L17 (Sx 0.52), L18 (Sx 0.93) | inside silu(gate) | L17 c22 1.0; L18 c4 1.0 (neuron 1712 alone) | gate (c299) reads a's and b's parity; up is ~constant |
| mod 25 (k4) | never from the operands; from L19, growing to L30 (M 0.60-0.93) | result × result | L22 c5 0.29, c6 0.23, c28 0.13, c27 0.12 | partners: k2 + k2 0.51 (mod 50 squared), −k1 + k5, k3 + k1 |
| mod 4 (k25) | L22-L30 (M 0.61-0.98) | result × result | L22 c17 0.88 (neuron 9758) | k20 + k5 0.60, k30 − k5 0.31 (mod 5 × mod 20) |

- **The adder unit (X).** Almost every creating unit reads **both** operands in **both** of
  its inputs, at the same harmonic:
  - gate ≈ `α cos k(a−φ) + β cos k(b−φ′)`, and up likewise;
  - the product's cross terms `cos k(a−φ)·cos k(b−φ′)` are `½[cos k(a+b−φ−φ′) + cos k(a−b−φ+φ′)]`;
  - the phases add (median error 0.04 period).
  - Silu's own curvature adds a little (Sx mostly < 0.1). The exceptions are parity and L16
    mod 10 (Sx 0.37 next to X 0.39).
- **Parity (Sx).** One neuron's gate reads `(−1)^a` and `(−1)^b` together. Silu's curvature
  squares their sum, which makes `(−1)^(a+b)`: an XOR computed inside the gate
  nonlinearity. Up only scales it.
- **Built from components.** Neurons read by alive down components carry 97-100 % of every
  creating write (82-97 % for the mixing ones). The inputs come from the unit's own gate/up
  components (same index) plus a few shared read directions (e.g. gate c88, c18, c299).
- **Later layers re-derive the periods from each other (M, L21-L30).**
  - mod 5 ← k10 + k10 (0.62 at L21: the units-digit code squared);
  - mod 10 ← k20 − k10 (0.80 at L21);
  - mod 100 ← k2 − k1, k3 − k2 (0.69 / 0.20 at L22);
  - mod 50 ← k1 + k1, k5 − k3 (L23);
  - mod 20 ← k3 + k2, k10 − k5 (L24).
  - From L22 roughly half of each late MLP's result write is this re-synthesis; the rest
    passes the code through. This fits the fan-out of the long periods into higher-dimensional
    codes from L26 (report_representations.md).
- **L20, mod 20.** The one write dominated by "other" (O 0.60). Two neurons (units c3, c6)
  mostly pass the mod-20 code through. They also build it from the `a−b` code times `b` or `a`
  codes, e.g. `(a−b)·k5 × b·k10 → (a+b)·k5` (= (a−b) + 2b), which is not a line term.

![period tree](figures_auto_interp_vectors/period_tree.png)

**Subtraction** uses the **same units at the same layers**:

- mod 5: L16 c127 / c94 / c76, then L18 c80 / c26 / c24 / c142;
- mod 10: L17 c39 / c86 / c49 / c34, then L18 c32 / c62 / c88;
- mod 50: L18 c12 / c21 / c27 / c22;
- mod 20: L18 c18 / c16 / c59, then L19 c6;
- parity: L18 c4.

The mirrored b turns each a×b product into `a − b`. The difference is at the long periods: an
`a−b` code of period 100, 50 and 25 is already present at `=` before L16 (the early comparator).
So on sub these periods are mostly passed through (P) rather than created. mod 25 and mod 4 again
come only from mixing (L22-L29).

## 4. How the result representation becomes the output token

The unembedding rows of the number tokens 0..199 (final-norm weight folded in) are themselves
structured. Their variance splits into:

- 7 % hundreds step;
- 3 % linear in `n mod 100`;
- **53 % periodic in `n mod 100`**: T100 19 %, T50 10 %, T25 9 %, T20 / T10 / T5 4 % each;
- 38 % token identity.

A result code `F(k)` in the last residual therefore adds `2 Re(F·Ḡ(k) e^{2πik(res−n)/100})/rms`
to the logit of token n. Here `G(k)` is the unembedding's own Fourier plane.

![votes](figures_auto_interp_vectors/logit_votes.png)

- **Every family's vote is positive at `n = res`, from the moment the code appears (17m).**
  The model writes each result circle in the phase the unembedding reads. Nothing needs to
  "decode" the circle.
- The votes grow to 3-3.5 logits (mod 100 family) at L28-L30 and sum to **3.85** logits (add)
  and 2.24 (sub) at the last point. After the last layer, the profile over `n − res` (right
  panel) peaks sharply at 0. Side peaks at ±10, ±20, ±40 come from the units-digit codes and
  are cancelled by the long periods.
- **Direct votes of the writers** (`R_c(res, k) · U_c·Ḡ(k)`, summed over k) add up to 3.73
  logits (add) and 2.08 (sub). This is almost exactly the stream's total, so the result logit is
  a committee:
  - the largest single component gives 0.07;
  - every layer from L18 to L29 contributes 0.2-0.5;
  - the top voters are units-digit writers (L24 down c4 / c5, L25 c11 / c12, L23 c4: mod 5 and
    mod 10 codes) and long-period writers (L20 c2, L24 c8: k = 1-3).
- **Hundreds (add).** The periodic codes cannot tell `n` from `n + 100`. The hundreds come from a
  magnitude path:
  - gate components read `lin(a) + lin(b)` with equal weights (corr with a + b: 0.75-0.87);
  - their neurons switch on around `a + b ≈ 100`. For example, L27 down c6's inner is 0 below
    80, 2.1 at 95-99, 6.2 at 100-104 and 2.9 above 120;
  - U points along `W_U[100..199] − W_U[0..99]`;
  - the main writers are L27 c6 and L26 c7 (on above 100), plus L22 c29 and L25 c33 (inner
    anti-correlated with a + b ≥ 100; they switch on for the large sums and their U has the
    matching sign).
- **Last layer.** The L31 MLP step shrinks the per-family votes (right edge of the left panels)
  and is only 14 % accounted for by alive `R_c U_c`. It is not captured by this line analysis
  and is probably a norm / confidence adjustment.

## 5. The operation token

![op flag](figures_auto_interp_vectors/op_flag.png)

- **Transport.**
  - L0H23's alive attention at `=` is driven by a key component that fires only on the op token
    (k c6);
  - the flag's size at `=` jumps at each of the first three attention steps: 0.17 of |x| after
    L0 attention, 0.41 after L1 attention, 0.69 after L2 attention;
  - the L1H6 / L2H2 o components are one-op-only: L1 o c370 / c47 on add, L2 o c2 on add,
    o c6 / c14 on sub.
  - At the op token itself, the v components read the op as a binary switch (L1 v c17 on sub
    only, v c8 on add only, …).
- **A large, drifting flag.**
  - At `=`, the add − sub mean difference is **45-70 % of the residual norm** from L2 to L15,
    then decays to 27 %.
  - Its direction is not fixed: cos is 0.04 between L2 and L15, and 0.26 between L8 and L15
    (right panel).
  - The flag is carried by one-bit relay components, e.g. L2 down c266 / c3, L3 c28, L4 c16,
    L6 c11, L7 c23, L8 c928, L9 c50, …, L14 c1, L15 c10, L16 c45, L17 c5. Each is ~0 on one op
    and ±2-5 on the other.
  - Every relay's U is aligned with the flag at its write point (cos +0.2 to +0.54, signed by
    its op). Each relay re-writes the flag into the direction the next layers read.
- **Gating.** Readers at `=` with the largest op dependence are pure switches:
  - L1 gate c27 (d′ 51);
  - L15 gate c72: mean 0.02 on add, 15.6 on sub;
  - L15 gate c9: 18.1 on add, 0.07 on sub;
  - L15 up c117.
- **What the op changes: b is mirrored before the adder.**
  - Before the L15 MLP, the gate/up readers at `=` see `b`'s code identically on both ops
    (same/reflect at mlp_in.15: k10 0.99 / 0.53, k20 0.98 / 0.53).
  - From mlp_in.16 on they see it **reflected**: k2 0.15 / 0.78, k10 0.18 / 0.80, k20
    0.10 / 0.95. `a` stays the same on both ops (figure below).
  - The reflection is exactly `b → −b`. Fitting `R_sub = λ_k · conj(R_add)` over the
    L16-L19 readers puts the reflection centre `c` (`b → c − b`) at 0 mod the period for
    every harmonic: within 1.6 for k = 1, 2, 3, 4; −0.3 for k10; 0.0 for k20; −0.7 for k5.
- **Effect on the adder.** Because the readers see `b → −b`, the same fixed L16-L18 units
  multiply `a` by `−b` and write `a − b` into the result planes. The result codes are shared
  across ops (top principal cosine 0.9-0.97 in the earlier sweep). The op never needs its own
  adder; it only decides the sign of b's sine axis (next subsection).

![mirror](figures_auto_interp_vectors/mirror_readers.png)

### How the L15 MLP builds the mirror

Split b's code at `=`, as the L16 gate/up readers see it, into an op-even part
`E = (F_add + F_sub)/2` and an op-odd part `O = (F_add − F_sub)/2` (per harmonic k).

**Geometric rule.** `F_sub` is the mirror image of `F_add` exactly when two things hold:

- E lies on one axis `e` and carries `cos(2πkb/100)`;
- O lies on one axis `w ⟂ e` and carries `sin(2πkb/100)`.

Then add = `e·cos + w·sin` and sub = `e·cos − w·sin`: the same circle (or ellipse) traversed
backwards, `b → −b`. The ratio |O| / |E| does not matter.

The readers see exactly this:

| L16 readers | k = 1 | k = 2 | k = 3 | k = 10 | k = 20 |
|---|---|---|---|---|---|
| share of E on one axis | 0.94 | 0.68 | 0.72 | 0.87 | 0.92 |
| share of O on one axis | 0.96 | 0.98 | 0.97 | 0.97 | 0.99 |
| \|cos(e, w)\| | 0.00 | 0.13 | 0.18 | 0.06 | 0.05 |
| phase(O) − phase(E) | −90° | +79° | −72° | −80° | −93° |
| reflect / same (P + L15 MLP only) | 0.93 / 0.60 | 0.84 / 0.12 | 0.83 / 0.26 | 0.83 / 0.05 | 0.97 / 0.09 |

(L17 and L18 readers give the same picture.)

- **Where the two parts come from.**
  - The odd part is written by the L15 MLP: 81-103 % of O at mlp_in.16, against ≤ 5 % from
    L16's attention.
  - The even part is mostly the L15H13 copy (40-75 % of E). In the readers' view the copy
    already sits close to one axis (0.6-0.96).
  - The MLP's op-even write removes most of what is left off that axis: it cancels 56-160 % of
    the copy's off-axis part (k20 over-cancels).
- **Why my first attempt looked harmonic-dependent.** Comparing the MLP write `D` with the
  copy `P` mixes the two parts: D contains both the odd write and the even cancellation. Split
  into E and O, the geometry is the same at every harmonic.

![mirror mechanism](figures_auto_interp_vectors/mirror_mechanism.png)

**Which neurons, and how SwiGLU does it.** The MLP output is exactly `W_down · act_b`
(cosine 1.000 with the measured stream change). Splitting the odd write over all 14,336
neurons of L15 gives two findings:

- 2-3 neurons per harmonic carry 85-95 % of it (7 neurons in total).
- `act = silu(g) · u` makes b × (±1) in one of two ways:
  - **A, the op on the gate.** The gate is an op switch fed by gate components c72 / c9. Up
    reads b's code. Two neurons, one opened by add and one by sub, have opposite-sign writes.
  - **B, the op on up.** Up is a signed op read: up components c117 and c34 give u ≈ −2 on
    add and +1 on sub. The gate reads b's code near its silu knee. `act = silu(g(b)) · u(op)`
    is b's code times the op sign, in a single neuron.

| harmonic | neuron | unit (down comp) | op enters via | b read by (peak, T/4 = sine phase) | share of odd write |
|---|---|---|---|---|---|
| k2 (mod 50) | 12769 | c35 | up: u −2.11 add / +0.75 sub (up c117, c34) | gate c35, c19 (13.7; T/4 = 12.5) | 0.45 |
| k2 | 6456 | c19 | up: u +2.07 / −0.65 | gate c35 (37.7; 3T/4 = 37.5) | 0.40 |
| k10 (mod 10) | 9205 | c21 | up: u −1.89 / +1.05 | gate c21 (2.3; T/4 = 2.5) | 0.66 |
| k10 | 9057 | c72 | gate: g −1.42 / +1.49, sub-on (gate c72, c9) | up c72, c67 (7.0; 3T/4 = 7.5) | 0.16 |
| k10 | 13193 | c67 | gate: g +1.01 / −0.21, add-on | up (2.1; T/4 = 2.5) | 0.06 |
| k20 (mod 5) | 7446 | c16 | gate: g +2.73 / +0.34, add-on | up c16, c64 (1.3; T/4 = 1.25) | 0.55 |
| k20 | 11305 | c64 | gate: g −1.78 / +1.69, sub-on | up c64, c16 (1.3) | 0.37 |

Mode shares of the odd write: k2 B 0.94; k10 B 0.75 / A 0.25; k20 A 0.96.

- **Why the centre is 0.** Every odd neuron's b read peaks at a quarter period (T/4 or 3T/4),
  i.e. it reads `sin(2πkb/100)`. So the V vectors of gate c35 / c21 and up c16 / c64 / c72
  select exactly the component of b that must change sign under `b → −b`, and nothing else.
- **What does the switching.** The components behind the whole mechanism are two op switches
  on the gate side (c72 sub-on, c9 add-on) and a signed op read on the up side (c117, c34).
  Each unit is identified by one index across gate/up/down, from the neuron-aligned init.
- **Correction to the earlier version.** The "add-gated c21 vs sub-gated c72" example was
  mislabelled. c21 (mode B) is active on both ops with opposite signs; c72 (mode A) is the
  sub-on partner of c67.

## What the vectors do not settle (for the validation pass)

- **The mirror is read off, not tested.** Settled geometrically in section 5; the causal test
  is to flip the op input of the 7 mirror neurons (n12769, 6456, 9205, 9057, 13193, 7446,
  11305), or of the op components c72 / c9 / c117 / c34, on addition prompts. If this story
  is right, a+b should turn into a−b in the result code.
- **The line analysis only sees additive and line structure.** Conjunctive "window" units
  (blobs in the grids) are captured through their Fourier lines only. Token-identity (lookup)
  variance, which is 25-40 % at every position, is not interpreted.
- **Sub includes `a < b`** (where the model says "?\n"). The line codes are exact on the full
  grid, but the late sub picture mixes the two regimes.
- **Last layer.** L31's MLP and the late fan-out of the long periods (L26+) are only partly
  captured.
- **Suggested first ablations:**
  - the L16-L18 result units (L16 c10, L18 c4 / c12 / c23);
  - the L15 mirror units (down c35, c19, c21, c72, c67, c16, c64);
  - the routing pairs (L15 q c138 × k c48, L16 q c136 × k c5);
  - the hundreds units (L27 c6, L22 c29).

## Files

- `param_decomp/arith_repr/vectors/`:
  - `spectra.py` (frames of the raw stream, read / write spectra);
  - `storage.py`;
  - `transfer.py`;
  - `qk.py`;
  - `mlp_units.py` + `mlp_analysis.py`;
  - `mirror_neurons.py` + `mirror.py` (the b-mirror of section 5);
  - `mlp_periods.py` + `periods_summary.py` (section 3.1);
  - `output.py`;
  - `figures.py`;
  - `common.py` / `load.py`.
- Outputs in `<run>/analysis/arith_repr/vectors/`:
  - `frames_{re,im}.npy` (65 points × 4 positions × 2 ops × 4 lines × 50 k × 4096, fp16);
  - `read_spec.npz`, `write_spec.npz`, `storage.npz`, `transfer.npz`, `qk.npz`, `output.npz`;
  - `mlp/L*.npz`, `mlp_units.parquet`;
  - `mirror/L14-16_neurons.npz` (all 14,336 neurons at `=`);
  - `periods/L14-31.npz`, `periods/summary.npz`;
  - `figs/`.
- sbatch files: `~/pd_scratch/dual_obj_jax/arith_repr/vectors/`.
