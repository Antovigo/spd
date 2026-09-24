# From components to mechanisms: residue arrangements and the components that move them

Decomposition `addsub-all-layers-4xh100-05` (run p-ba5a0c05, step 40000), filter
`addsub-05-filter-last-pos-ceiling`, prompts `a op b =` with a, b in 1..100. This report continues
[report_auto_interp.md](report_auto_interp.md). Like that report, it only describes activations
of the unmodified model. The one exception is §2.4, which runs the model with its MLPs replaced by
linear versions.

It answers two questions.

1. Several components can do one elementary job together: ten components that each handle one
   residue of a mod 10, or the same representation written again at several layers. When is
   grouping them into one **mechanism** justified, and what are the mechanisms?
2. For a group of values of a (or b, or the result), such as the ten residues mod 10, how are
   their representations arranged relative to each other at each layer, and how do the weight
   matrices change that arrangement?

The second question is answered first (§2), because the answer determines which grouping
criteria can work (§1).

<details><summary>Legend</summary>

* **Position**: `a`, `op`, `b`, `=` are the token positions 1–4. **add / sub** is the operation.
  **res** is a+b (add) or a−b (sub).
* **Variable**: a function of the prompt that a component's write can depend on. The candidates
  are `q%τ` (the residue of q mod τ, for q in a, b, res and τ in 2, 4, 5, 10, 20, 25, 50, 100),
  `q//10` (tens digit), `res` (the full value), `units(a,b)` (the pair of units digits, 100
  classes), `tens(a,b)` (the pair of tens digits, 121 classes) and the flags carry/borrow,
  cmp(a,b), res≥100 / res<0. Each writer gets the variable with the fewest classes that explains
  ≥ 90 % of the best adjusted R² of its gated write, provided that best R² is ≥ 0.5.
* **Writer**: an o_proj or down_proj component. Its write at a prompt is w·U, with
  w = inner·[CI > 0.01].
* **Arrangement** of a variable at a residual point: the class means of the raw residual over
  its classes (e.g. the ten mean vectors of the prompts with a ≡ 0, …, 9 mod 10), centred.
  Points are `embed`, then `Ll attn` (after layer l's attention adds its output) and `Ll MLP`
  (after its MLP adds its output).
* **Support** of a component: the classes where its mean write differs from its typical
  (median) class by at least half its largest deviation. These are the classes it moves.
* **k** in spectra: a frequency on the mod-100 circle. Its period is 100/gcd(k, 100), so k = 10
  is period 10, k = 20 period 5, k = 50 period 2 and k = 1 period 100.

</details>

## Summary

* **The shape of the operand code never changes; its position does.** At `a` (and likewise
  at `b`) the ten residues of a mod 10 sit almost like the corners of a simplex, a one-hot-like
  code: every residue is about equally far from every other, and the pattern looks the same
  from each residue (it is shift-symmetric). This already holds in the token embedding. After
  every layer the new arrangement has the same shape as the old one (CKA ≥ 0.97 from L1 on),
  but it gradually moves to new residual directions. Attention steps leave it almost in place.
  Each MLP step moves 10–50 % of it into new directions. Compared with the embedding, the
  mod-10 arrangement at `a` has cos 0.25 after the L15 MLP and 0.07 after L31 (§2).
* **The MLPs that move the operand code are mostly linear, and their gates need not switch.**
  On the model's own neurons, one fixed linear map reproduces 82–93 % of each MLP's write to
  the a mod 10 arrangement at L1–L11 (median 0.88 at L15–L31). The rest is gate × up products,
  not gates switching. Many gates do cross zero across the values of a, mostly inside a
  units-digit class, but only near their knee. With silu taken as a line in all 32 MLPs at `a`,
  the answer moves by KL 0.019 (removing their write: 1.23). The exceptions are L0, where the
  product dominates, and L12–L14, where a few units-digit detector neurons switch between
  classes (KL 0.010). Chaining the linear maps is not enough: linearising all 32 MLPs at `a`
  costs KL 0.47, most of it set off by L0 (§2.4).
* **Writers of the same variable write in almost orthogonal directions.** Take two writers of
  the same variable at the same position. The cosine between their contributions (write
  pattern times U direction) has a median of 0.000–0.002 and a 99th percentile of 0.03–0.07,
  against 0.02–0.03 when U is replaced by a random writer's. Fewer than 0.05 % of pairs exceed
  0.3. Even writers with the same pattern (|corr| ≥ 0.9) have a median |cos U| of only
  0.02–0.06. Groups of such writers have κ ≈ 1 (joint energy = sum of the members' energies)
  in 94 % of the multi-component mechanisms. So "moving the same representation over several
  layers" does not mean pushing along one fixed axis. It means writing the same shape into new
  directions, layer after layer (§1.1, §2).
* **Criteria used.** Writers are grouped only if they have the same position, operation and
  variable, and in addition one of these holds:
  * they sit in one step: one block, or two adjacent layers whose supports tile the classes;
  * across layers, they write the same arrangement shape (CKA ≥ 0.7). For ≥ 12 classes, 12 % of
    real code pairs pass this, against 0.01 % after shuffling one code's classes.

  This turns {{n_writers}} labelled writers into {{n_codes}} codes and {{n_mechs}} mechanisms.
  The result is then checked rather than filtered:
  * 86 % of the multi-component mechanisms tell apart more classes than their best member;
  * 48 % have a downstream reader that draws on ≥ 2 members at once;
  * members' writes add orthogonally (κ in [0.8, 1.25]) in 94 %.

  See §1.2–§1.4 and the [catalogues](#4-complete-catalogues).
* **Tilings exist but are the exception.** A tiling is a set of members that each handle
  their own residues. The units digit of a is written by three tilings at `a`: ten L0 detectors
  (one per residue, no overlap), eight L13 detectors and seven L14 detectors. The units digit
  of b has the same structure at `b` (L0, L11, L13). Among all codes with ≥ 3 members, only
  {{tiling_share}} overlap less than random sets of writers of the same variable. Most codes
  are blocks of overlapping detectors (§3.1, §3.2).
* **The copies into `=` keep the shape.** At L15, H13 writes b mod 10 at `=` with ten o
  components (CKA 0.84 with the arrangement at `b`). At L16, H21 writes a mod 10 with sixteen o
  components (CKA 0.80 with `a`). Each component covers an overlapping subset of residues. Only
  together do they tell all ten apart (§3.3).
* **The result is assembled from pair detectors and then changes shape.** L17's down
  components are units-digit-pair detectors: 13 components that together tell apart 90 of the
  100 (a%10, b%10) pairs. So are L18–L19's (22 components, 88 pairs). The result's units digit
  first appears at L19–L20 as a period-10 circle. From L21 on it is a simplex (8 components
  tell apart all 10 residues) and is re-written by a separate mechanism at almost every later
  layer. Before L16, res mod 100 at `=` is an ordered circle of period 100. On add it is weak
  (≤ 2 % of the variance). On sub it holds 13–16 % of the variance and survives removing the
  separate a and b codes, so it comes from an interaction of a and b. What it encodes is not
  resolved here. From L19 on it spreads over many frequencies and reaches dimension ≈ 30 at the
  output (§2.3, §3.4, §3.5).

## 1. When is a group of components one mechanism?

### 1.1 What a component does to an arrangement, and why grouping cannot rely on directions

A writer c changes the arrangement of a variable by C_c = m_c ⊗ U_c. Here m_c(v) is the mean of
its write coefficient over the prompts of class v (centred), and U_c is its output direction.
This works because the residual is additive: the arrangement after a layer equals the one
before plus the C_c of every writer of that layer, plus whatever the decomposition does not
capture. Summing C_c over a set of writers therefore gives exactly how much they change the
arrangement together.

The simplest idea for "the same representation moved over several layers" would be writers
whose C_c point the same way: same class pattern and overlapping U. The data rule this out:

{{orthogonality}}

<details><summary>Why this rules out direction-based grouping</summary>

The Frobenius cosine of two contributions is corr(m_c, m_c′)·cos(U_c, U_c′). The table shows
that the small values come from the U factor. Pairs whose write patterns are the same
(|corr| ≥ 0.9) still have a median |cos U| of 0.02–0.06, only a few times that of random writer
pairs (0.012). Only their top 1 % reach 0.23–0.47, a tail that random pairs do not have
(0.05–0.09). Aligned re-writes along the same axis therefore exist, but they are rare. κ (next section) says the same about whole
groups: the joint between-class energy equals the sum of the members' energies (median 1.00),
so the members neither add up along one axis (κ > 1) nor cancel (κ < 1). Section 2 shows the
same thing at the level of whole arrangements: the shape is kept and the location drifts.

</details>

### 1.2 Conditions for a valid group, and the check behind each

A set of writers is counted as **one mechanism acting on one variable** when all of the
following hold.

<details><summary>(i) Same variable, position and operation</summary>

Every member's gated write must be a function of that variable: its variable label, R² ≥ 0.5 of
w on the variable's classes. Without this, the sum of the members' contributions is not an
arrangement of that variable. The coarsest explaining variable is used, so that a detector of
a%10 = 3 counts as `a%10`, not `a%100`. The price is that nested variables stay separate: a
parity writer (`a%2`) and a units-digit tiling (`a%10`) are never grouped, even when together
they would give a mod-10 code by the Chinese remainder theorem. Operations are clustered
separately and then matched (`twin`: the mechanism of the other operation with the largest
member overlap).

</details>

<details><summary>(ii) One step: a code = one block, or adjacent layers that tile</summary>

A **code** is all writers of the variable in one block (a layer's MLP, or one attention head,
the head of an o component being the one its V reads most). Blocks at the same or adjacent
layers are merged when their supports tile the classes, i.e. the two blocks' support unions
overlap by ≤ 20 % of the smaller one. This is the "layer N handles residues 0–4, layer N+1
handles 5–9" case. Merging stops at a span of two layers so that tilings cannot chain across
the network.

Checks: `coverage` (classes in some member's support); `groups` (classes the joint write tells
apart, i.e. class means further apart than 5 % of the mean distance) against the best member;
`loo_needed` (share of members whose removal merges two classes); `overlap` = Σ|support| /
|union of supports| (1 = perfect tiling) against 200 random sets of as many writers of the same
variable at that position (`overlap_p` = share of random sets that overlap as little or less).

</details>

<details><summary>(iii) Across layers: the same shape written again</summary>

Codes of the same variable at different layers are joined into one mechanism when their
arrangements have the same shape: linear CKA of their class-mean Grams (class-count weighted)
≥ 0.7, with average linkage. CKA ignores directions, which is exactly what §1.1 requires.
This is the "same representation shifted successively" case. The threshold is compared with
the CKA of the same code pairs after permuting one code's classes:

{{pairs}}

For 2–3-class variables any two codes have the same shape (CKA 1), even after permutation. All
codes of a flag or a parity at one position therefore end up in one mechanism, which is the
intended reading ("the flag, written at these layers"). For 4–11 classes, 1.9 % of permuted
pairs pass 0.7, against 33 % of real pairs. For ≥ 12 classes the figures are 0.01 % against
12 %.

</details>

<details><summary>(iv) Checks reported for every group (not used to form the groups)</summary>

Per mechanism, on every prompt:

* `purity`: the share of the joint write's energy that is between-class.
* `acc`: balanced nearest-centroid accuracy of the variable from the joint write, against the
  best code and the best member.
* `kappa`.
* `consumers`: downstream readers (q/k/v/gate/up components reading the variable with
  R² ≥ 0.3) that draw ≥ 20 % of their class-mean profile of the variable from the joint write.
  The draw goes through their V·(norm weight)·U_c/rms connection. `joint_consumers` are the
  consumers for which ≥ 2 members each supply ≥ 10 %.

</details>

<details><summary>Pitfalls and what was done about them</summary>

* **On-set is not write.** A component can be on for a class and still write nothing that
  differs between classes. Labels and supports therefore use the gated write w, not the CI.
* **The CI function is not causal.** At `a`, `op` and `b` a gate can depend on later tokens
  (report_auto_interp.md §0). The labels are computed per operation, so a gate that switches
  with the operation shows up as a difference between an add mechanism and its sub twin.
* **Result class means mix in the operands.** Class means over res also pick up the a and b
  codes wherever the grid makes a, b and res dependent: res = 199 forces a = 100, b = 99, and on
  sub the sign of res is the comparison a < b. `res_int` subtracts E[E[x|a] + E[x|b] | res],
  which leaves only what an interaction of a and b can write.
* **Deterministic positions.** At `a` and `op` the residual is an exact function of a within
  one operation, so any set of distinct writers decodes a perfectly. There, per-prompt accuracy
  carries no information, and the class-level checks (`groups`, `overlap`, CKA) are the ones to
  read.
* **Greedy order.** Blocks are merged in layer order, first fit. Only the adjacent-layer tiling
  step depends on the order, and it rarely fires (the codes table below counts multi-block
  codes).

</details>

### 1.3 Outcome of the checks

![checks](figures_auto_interp/mech_checks.png)

{{checks}}

How to read it:

* 86 % of the multi-component mechanisms tell apart more classes than their best member (the
  points above the diagonal in the second panel), so grouping usually adds something. The
  exceptions are mostly redundant copies: several members with the same support.
* κ is concentrated at 1 (fourth panel), so members add orthogonal pieces.
* Only a minority of codes are tilings better than random (third panel, red): the units-digit
  lookups at `a` and `b`, the large value lookups of a and b (`a%100` / `b%100` blocks at L0,
  L2, L14–L17 and L20), and a few `tens(a,b)`, `units(a,b)` and result blocks at `=`. In the other codes, overlapping detectors share one block.

### 1.4 How many mechanisms

{{counts}}

At `a` nearly every add mechanism has a sub twin with member overlap ≥ 0.5: there the gates
barely depend on the operator. At `op`, `b` and `=` only 14–29 % do. Their add and sub
mechanisms share a variable but not most of their components. At these positions the residual
itself depends on the operator (the op token, and attention to it), and the gates can also see
it through the non-causal CI, so different components end up active on add and on sub.

![map](figures_auto_interp/mech_map.png)

The map shows, for each position (add), every code as a dot at its layer (size grows with the
number of components). Codes joined into one mechanism share a colour and a line. Most
mechanisms are one code (one block at one layer). The long lines are variables whose shape is
written again at many layers: `a%100`, `a//10` and `a%50` at `a`, `b%100` and `tens(a,b)` at
`b` and `=`, and `res//10` at `=`.

## 2. How the residues of a group are arranged, layer by layer

![code power](figures_auto_interp/mech_code_power.png)

![geometry](figures_auto_interp/mech_geometry.png)

![snapshots](figures_auto_interp/mech_snapshots.png)

![drift](figures_auto_interp/mech_drift.png)

### 2.1 Operands at their own token: a fixed simplex that drifts

* **Shape.** The mod-10 arrangement of a at `a` is simplex-like from the embedding on:
  participation ratio 8.1 of a possible 9, shift-symmetric part 0.95, flat spectrum over
  k = 10, 20, 30, 40, 50. The residues are not ordered (the correlation of distance with cyclic
  distance is 0.15), so this is not a circle. Mod 100, the code is much richer (it is the
  strongest code at `a`, top row of the first figure).
* **What the layers do.** From L1 on, every step keeps the shape (`cka_prev` ≥ 0.97, fifth
  panel of the geometry figure). What changes is where it sits. An attention step keeps
  essentially all of the arrangement in its old span; an MLP step keeps 50–90 % (the zig-zag in
  the last panel). The drift figure adds these up: compared with the embedding, the mod-10
  arrangement at `a` has cos 0.70 after L0, 0.47 after L3, 0.29 after L7, 0.25 after L14 and
  0.07 after L31.
  b at `b` behaves the same way.
* **Which matrices.** The MLPs move the arrangement, and they do so by writing the same shape
  into new directions (§1.1). A units-digit MLP code at `a` has CKA up to 0.77 with the
  arrangement it reads (`cka_in`; 0.77 for the L0 tiling, 0.55–0.70 for L11–L14), but only
  5–13 % of its write lies in that arrangement's span (`in_span`) and its cos with it is
  0.1–0.2 (§3.1). Only L0's MLP adds code strength (share of stream variance
  0.15 → 0.18). Later layers mostly re-write it: the code share stays at 0.16–0.26 up to L31.
  An MLP can reproduce a shape in new directions because the arrangement is (nearly) full rank
  (9 independent class means), so a linear map can send it anywhere. For the same reason,
  "same shape" says nothing about whether the MLP *is* a linear map: any function of the class
  is some linear map of a full-rank arrangement. §2.4 tests that directly. Short answer: most
  of each MLP's write is one linear map, but not all of it. The rest comes from gate × up
  products, not from gates switching. At the neuron level, gate switching matters only at
  L12–L14, where units-digit detector neurons turn on for exactly one units digit.

<details><summary>Arrangement tables (every group, both ops, selected points)</summary>

{{arrangements}}

</details>

### 2.2 Operands at `=`: copied early, faded, copied again with their shape

The `=` token's own embedding is the same in every prompt, so every code at `=` is written
there by attention.

* **b arrives first.** After L0's attention, b mod 100 already accounts for 94 % of the variance
  at `=`. Its shape is irregular: a value code, not a circle. After L1's attention, b mod 10 is
  simplex-like (shift-symmetric part 0.95, code share 0.25).
* **a arrives next.** After L2's attention, a mod 100 accounts for 52 % of the variance. Its
  mod-10 part is not shift-symmetric (≈ 0.33): residues 0 and 5 stand apart from the others
  (snapshots row 3), so what arrives is a's value, not a units-digit code.
* **The early copies fade.** By L13 the b mod 10 code share is down to 0.03.
* **L14–L16 copy both operands again, this time in the operand's own shape** (§3.3). After
  L15's attention, b mod 10 is back to shift-symmetric part 0.94 with share 0.14. After L16's
  attention, a mod 10 goes from 0.45 to 0.94 with share 0.10. Both codes then fade again
  (≈ 0.02 at the output).

### 2.3 The result at `=`: a circle of magnitude, then a simplex of digits

* **res mod 100 before L16.** It is an ordered circle of period 100: k = 1 carries 0.5–0.7 of
  the power and the correlation with cyclic distance is 0.8 after L3.
  * On add, `res_int` ≈ res, but the code share is tiny (≤ 0.016).
  * On sub, `res_int` (the result's class means minus what the separate a and b codes give)
    holds a period-100 circle with share 0.13–0.16 from L7 on. So an interaction of a and b is
    already written at `=` before the copy heads.

  This circle has period 100 and no units-digit part (k = 10, 20, 50 are never among its three
  strongest frequencies before L16),
  so it is not the result's digits. A magnitude-like quantity of a − b would look like this; the
  flag a < b alone would not, since on sub res mod 100 mixes negative and positive results.
* **res mod 10.** It appears after L16–L19. It is a mix of circles (k = 50 parity first, then
  k = 10, 20) until L20. From L21 on it is simplex-like (participation ratio ≈ 8, shift-symmetric
  part 0.95–0.98). This is the shape of the operand codes, so the model ends up with the
  result's units digit in the same one-hot-like shape as the operands' units digits.
* **res mod 100 after L19.** The code share grows from 0.07 (L17) to 0.6 (L25). The spectrum
  spreads over k = 1, 2, 10 and many more, and the dimension grows from 5 to 30 at the output.
  Within the period-100 group the arrangement becomes more and more one-hot-like: separate
  directions per result value, which is what an unembedding over number tokens needs.

### 2.4 Is an MLP a linear map on the operand codes? Do its gates switch?

§2.1 says the MLPs re-write the operand arrangement in its own shape. Two readings of that
would make the operand pathway simple:

1. each MLP acts on the operand as one fixed linear map;
2. the SwiGLU gates never switch on or off across the operand's values in a way that
   matters.

Both are tested here on the frozen model's own neurons (no components). Each MLP is fed the
recorded residual at `a`, `b` and `=`. The population is every prompt of the pool: at `a`,
that is the 100 distinct prefixes.

**Exact split of each MLP's write.** Take one neuron, with gate pre-activation g, up u,
s = silu(g) and output h = s·u. Over the population, fit silu with its least-squares line on
the neuron's own range of g: s ≈ α + βg, with residual r. Then, exactly,

    h − mean(h) =  mean(s)·δu          up        linear: the up read through a frozen gate
                 + mean(u)·β·δg        gate read linear: the gate's read, silu taken as a line
                 + mean(u)·r           silu bend the gate switching on/off (0 if it never crosses its knee)
                 + δs·δu − mean        gate × up the product of two reads (quadratic in the input)

Here δ is the deviation from the population mean. `up` + `gate read` is one affine map of the
MLP's (normed) input, the same for every prompt: this is the MLP "as a linear map". Map each
term through W_down and take its class means. That splits the MLP's write to any arrangement
(§1.1) into four parts. Each part is credited by its projection on the whole write, so the
four shares sum to 1. The R² of a subset is how much of the write that subset reproduces.

![MLP linearity](figures_auto_interp/mech_mlp_linear.png)

* **Most of the operand write is one linear map, except at L0 and L12–L14.** At `a`, one
  linear map reproduces 82–93 % of each MLP's write to the a mod 10 arrangement at L1–L11, and
  69–98 % at L15–L31 (median 0.88). For the full value (a mod 100) it is 62–97 % outside L0 and
  L12–L14; for b mod 10 at `b` it is 54–98 %. The linear part is shared about evenly between
  the up read and the gate read. The gates move a lot, but mean(u) ≠ 0, so a moving gate acts
  like a second linear read.
* **The rest is mostly the gate × up product, not switching.** Outside L12–L14, the silu bend
  is at most 6 % of the write, and usually 0–3 %. The product carries up to 28 % at `a`. It
  carries 13–51 % of the per-prompt write at `b` (variable "prompt" in the tables), where the
  stream also carries a and op.
* **L0 is not linear.** At L0 the product carries 71 % of the a mod 10 write (57 % of a mod
  100), and one linear map reproduces only 35 %. L0 is also the only MLP that writes more code
  than the stream already holds (write / stream energy 2.4). Every later MLP is at 0.07–0.48,
  and L31 at 0.74.
* **L12–L14 switch, but between classes.** Here a handful of neurons carry the silu bend: 4
  neurons hold 90 % of it at L12 and L13, 46 at L14. Its share of the a mod 10 write is 0.21,
  0.38 and 0.10. These are units-digit detectors. Each gate is on for exactly the ten values
  with one units digit and off for all others. For example, at L13 n1653 is on for
  a ∈ {7, 17, …, 97}, n10984 for {3, 13, …, 93}, n9765 for {5, …, 95} and n8597 for
  {8, …, 98}; at L12 n6778, n5408 and n4839 do the same for 9, 0 and 1.
* **Elsewhere many gates cross zero, mostly inside a class, and it does not matter.** Across
  the 100 values of a, 6,000–13,000 of the 14,336 gates cross zero (2–98 % of the values have
  g > 0). Outside L11–L14, typically 10–30 % of the variance of those on/off patterns lies
  between mod-10 classes (up to 38 % at L27; weighted by each neuron's silu bend). Between
  tens-digit classes it is 30–55 %. So most switching happens inside a units-digit class,
  tracking the tens digit or magnitude. These crossings happen near the knee, where silu is
  almost a line, so they add little to the write. At L12–L13 the between-class share is
  0.69–0.79. At `b` it is lower still, 0.03–0.16 outside L12–L14, because the gates there also
  follow a and op.
* **Contrast: the result at `=`.** At L16–L18, one linear map reproduces only 8–15 % of the
  res mod 10 write. The product carries 50–90 % there: this is the a × b multiplication of the
  vector report. From L19 on the result's re-writes are 50–95 % linear.

**Causal test: the model with MLPs replaced by their linear versions.** For each MLP at a
position, three replacements, all fitted on the same population and each reproducing the
MLP's mean output on it exactly:

* `one linear map`: h = mean(h) + up + gate read (nothing switches, nothing multiplies);
* `no switching`: silu replaced by each neuron's line, h = (α + βg)·u + c (the gate × up
  product is kept);
* `write removed`: h = mean(h), for scale.

The model is run in float32 from the embedding on a random half-add, half-sub sample of the
pool, with the replacement at every layer in a range. The table reports the KL of the next-token
distribution at `=` to the clean model's, and the accuracy of the answer's first token. Clean
accuracy is 0.95 on add. On sub it is 0.27–0.29, because a negative answer has to start with
"-". KL is the main measure. Bars at the left edge of the figure are below 1e-4.

![MLP patch](figures_auto_interp/mech_mlp_patch.png)

* **The gates never need to switch.** Taking silu as a line in all 32 MLPs at `a` costs KL
  0.019 (add accuracy 0.95 → 0.93), against 1.23 for removing their write. Of that, 0.010
  comes from L12–L14 alone (the units-digit detectors); L0–L11 and L15–L31 cost 0.001 each.
  At `b` it is 0.045 in total (0.016 from L12–L14, 0.022 from L0–L7). So switching inside a
  class is causally negligible. The one causally visible switch is between classes, and
  small.
* **But the operand pathway is not a chain of linear maps.** One linear map in all 32 MLPs at
  `a` costs KL 0.47 (add accuracy 0.95 → 0.39). That recovers only 62 % of the gap to removing
  their write (1.23). At `b` it is 0.37, and at both positions together 0.89 (add accuracy
  0.11).
  * L0 is the largest single contributor: linearising only L0 at `a` costs 0.057, against
    0.006 for L1–L7 and 0.073 for L1–L31.
  * The effects compound. L0 and L1–L31, each linearised alone, cost 0.057 and 0.073, but
    together 0.47, and L0–L7 together 0.147. Each linear map is fitted on the clean inputs.
    Once an earlier MLP has been linearised, the later ones see different inputs, and there
    the dropped product terms matter.
  * At `b` the maps are fitted over both ops, so the op-gated mirror of b (L15, vector
    report) is part of what a linear map cannot do. Linearising b's L1–L7 changes the top
    token on 22 % of prompts but *raises* accuracy (add 0.94 → 0.99, sub 0.29 → 0.45). This is
    not explained here.
* **At `=` the MLPs are not linear at all.** One linear map in all 32 MLPs at `=` costs KL 1.16,
  almost as much as removing their write (1.53). No switching costs 0.12.

So, for the operand codes, each MLP's write is mostly one linear map of its input, and its
gates never have to switch. What a linear map misses is the gate × up product. It is small in
most single MLPs, but decisive in L0 and once the MLPs are chained. At the neuron level,
detector-like switching (a gate on for one residue class and off for the rest) matters only
at L12–L14. There, units-digit neurons switch between classes, never inside one.

<details><summary>Per-layer split: a mod 10 at `a`</summary>

{{mlp_lin_a}}

</details>

<details><summary>Per-layer split: a mod 100 (every value) at `a`</summary>

{{mlp_lin_a100}}

</details>

<details><summary>Per-layer split: b mod 10 at `b` (add)</summary>

{{mlp_lin_b}}

</details>

<details><summary>Per-layer split: every prompt at `b` (add)</summary>

{{mlp_lin_bp}}

</details>

<details><summary>Per-layer split: res mod 10 at `=` (add)</summary>

{{mlp_lin_res}}

</details>

<details><summary>Every linearised forward</summary>

{{mlp_patch_table}}

</details>

Code: `param_decomp/arith_repr/autointerp/mlp_linearity.py` (split, one job per layer),
`mlp_patch.py` (forwards), `mlp_lin_report.py` (switching structure, tables, figures). Data in
`<run>/analysis/arith_repr/autointerp/mech/mlp_lin/`, `mlp_patch/`, `mlp_*.parquet`.

## 3. The mechanisms

Each mechanism below is a collapsible block. Its summary line gives the mechanism id, the
variable, its kind, the number of components, the layers, how many classes it tells apart and
its twin on the other operation. Inside are the checks, how it transforms the arrangement, and
the codes with every component: the classes it moves (support) and the sign of its write.

### 3.1 Units digit of a, at `a` (add)

Three tilings: L0 (ten detectors, one per residue, overlap 1.00 against 1.67 for random sets),
L13 (eight detectors, overlap 1.00 against 1.50) and L14 (seven detectors, 1.00 against 1.40).
L0 is joined with an L3 block and an L12 block (mean CKA 0.74). L13 and L14 stay separate, even
though each is also a partial one-hot of a mod 10: each covers a different subset, so their
CKA with the L0 tiling is 0.70 and 0.73 but only 0.38 with each other. Under the ≥ 0.7
average-linkage rule they are not merged. A criterion that accepted partial codes consistent
with one full code would join all of L0 / L3 / L11–L14 into one "re-write a's units digit"
mechanism. The code-by-code tables are kept so that either reading is possible. The L11–L14
re-tilings sit just before L15H13 / L16H21, which read `a` and `b` at `=`.

{{hl_a10}}

### 3.2 Units digit of b, at `b` (add)

The same structure as a: tilings at L0 (ten detectors), L11 (eight) and L13 (eight), joined by
CKA into one mechanism.

{{hl_b10}}

### 3.3 Copies into `=`

Every mechanism at `=` (add) whose joint write has CKA ≥ 0.5 with the arrangement of the same
variable at an earlier position (`src_pos`), read at the layer's input:

{{hl_copies}}

### 3.4 Digit-pair lookups at `=` (up to L20)

{{hl_pairs}}

### 3.5 Result codes at `=` (add), one row per mechanism

{{hl_result}}

## 4. Complete catalogues

Every mechanism at every position and operation, grouped by variable, with checks,
transformation and all components:

* `a`: {{app_1}}
* `op`: {{app_2}}
* `b`: {{app_3}}
* `=`: {{app_4}}

<details><summary>How to regenerate</summary>

```
python -m param_decomp.arith_repr.autointerp.class_means  --dataset <filter>/dataset --out <ai>/mech
python -m param_decomp.arith_repr.autointerp.mech_prep    --dataset <filter>/dataset --uv <ai>/uv_alive.npz --wiring <ai>/wiring.npz --out <ai>/mech
python -m param_decomp.arith_repr.autointerp.arrangements --run <run>
python -m param_decomp.arith_repr.autointerp.mechanisms   --run <run>
python -m param_decomp.arith_repr.autointerp.mech_figures --run <run>
python -m param_decomp.arith_repr.autointerp.mech_report  --run <run> --notes notes/arith_representations
```

`<ai>` = `<run>/analysis/arith_repr/autointerp`. Edit `mech_report_template.md`, not this file.

</details>
