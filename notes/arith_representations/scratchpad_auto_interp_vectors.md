# Auto-interp from the component vectors — scratchpad

Working notes; findings that survive go to `report_auto_interp_vectors.md`. Companion to the
activation-level auto-interp (`report_auto_interp.md`, other session): here each claim should rest
on U and V — what V reads from the stream (the inner `h_c = x_norm . V_c`, measured on the original
model = V applied to the measured activations) and what U writes into it (U projected on the code
frames of the stream right after the add). CI is used only to pick what to look at.

## Data / code

- Decomposition p-ba5a0c05 step 40000, filter `addsub-05-filter-last-pos-ceiling`, 11,604 alive comps.
- `U`, `V`: `<run>/analysis/arith_repr/autointerp/uv_alive.npz` (other session's extract_uv).
- Activations: `<filter>/dataset/original/{resid,inner,mlp_gate,mlp_up}.npy` (full 20k grid, 5 pos).
- Code: `param_decomp/arith_repr/vectors/` (this branch, worktree
  `~/pd_scratch/worktrees/arith_repr_vectors` — detached on origin/experiment/arith_representations,
  because `arith_repr_edit`, which holds the branch, is in live use by another session and
  `dual_obj_jax` is on feature/dual_obj_jax). Outputs: `<run>/analysis/arith_repr/vectors/`.
  sbatch/q.sh: `~/pd_scratch/dual_obj_jax/arith_repr/vectors/`.

## Frame

Line DFT over the 100 x 100 grid (per op): for q in {a, b, sum=a+b, diff=a-b} and k = 1..50,
`F_t(q,k) = mean_i x_i e^{-2 pi i k q_i/100}` (vector, raw stream point t). Harmonic k has period
100/gcd(k,100): k=10 -> units digit circle, k=20 -> mod 5, k=2 -> mod 50, k=1 -> mod 100, k=50 ->
parity, k=5 -> mod 20, k=4 -> mod 25.
- Read side of c: R_c(q,k) = same DFT of the scalar inner h_c. Peak value q* = -arg R * 100/(2 pi k).
- Write side of c (o/down): `U_c . F_t(q,k)` at the point after the add; the component's own code
  contribution is `R_c(q,k) U_c`; its alignment with the stream code is
  `rho = R_c (U_c . conj F_t) / |F_t|^2` (Re > 0 reinforces, arg = residue shift).
- MLP: down V is neuron-sparse (median participation ratio 5.6 neurons), gate/up U dense in the
  neuron basis (PR ~1700-2400) -> a down comp ~ a handful of neurons; its inputs are the gate/up
  comps whose U loads on those neurons. o V is sparse in the head-concat basis (PR 3.8 -> one head).
- Attention: v (source read) -> o (dest write) via `U_v[kv(h)] . V_o[h]`; q/k pairs scored with RoPE
  at the relative offset.

## Log

### 2026-09-23 night
- Jobs 12512 frames, 12513 reads, 12514 writes (after frames).
- Neuron alignment: down V top-neuron share median 0.41-0.57, PR 5.6; gate/up U top-neuron share
  0.01-0.02 (dense), top neuron = the down comp's with the same index in 38-70 % of cases (they
  started as the same neuron, neuron-aligned init). So gate/up comps are shared READ directions,
  the down comps are ~neuron groups. Same-index (gate c, up c, down c) is usually the top feeder.
- Read spectra at `=` (inner variance share on lines, add, var-weighted over gate/up/down/o):
  L0-1 b only (0.97/0.82: L0 attention brings the previous token b); L2-14 a 0.45-0.7, b 0.3-0.4,
  sum 0; L15 b 0.83 (L15H13 copy); L16-18 a ~0.4, b ~0.5; L19 sum 0.35; L20 0.55; L21-30 0.75-0.81.
  Sub: diff line 0.14-0.21 already at L6-14 (comparator), 0.24 L19, 0.37 L20, 0.5+ from L21.
- v comps read the SAME code, same phase, at the a and the b position (e.g. L15 v c17: a k5 @19.4
  at pos 1, b k5 @19.4 at pos 3) -> operands are stored in shared, position-agnostic directions;
  the head's routing alone decides which operand arrives.
- OV copy keeps the phase: L15 o c81 (H13) reads b k5 @19.7 at `=` from v c17 (b k5 @19.4, cpl
  +0.15); L16 o c121 (H21) a k10 @8.1 from v c78 a k10 @8.5 (cpl +0.12). Copy = identity on the code.
- QK (qk.npz, mean inners x RoPE'd U dot products): L15H13: q c138 x k c48, where k c48's mean inner
  is -21 at b and ~0 elsewhere -> +4.7 logit to b (add), 4.0 total on sub. L16H21: q c136 x k c5,
  k c5 mean 51.6 at a / 22.8 at b -> +12.3 to a vs +4.9 to b (sub: a 3.6 vs b 2.5 total, the weak
  routing on sub). L0H23: k c6 on only at op (15.2) -> attends op. L18H30: k c95 on at a and b
  (45.5/43.5), q c104 -> both operands. L13H7: k c52 on at b (-22.5).
- Op readers at `=` (mean add - mean sub of inner): L1 gate c27 (d' 51), up c51, L2 gate c2/c5/c4,
  L3 q c109 / gate c58, ..., L15 gate c72 (-15.6, sub-on), up c117 (+14.7) — the op flag is
  readable at `=` from L1 on.
- Unembedding over number tokens 0..199 (final-norm weight folded, centred): hundreds step 7 %,
  linear-in-(n mod 100) 3 %, periodic (functions of n mod 100) 53 % (T100 19 %, T50 10 %, T25 9 %,
  T20/T10/T5 4 % each, T2 1.4 %), token identity 38 %.
- Jobs: 12520/12523 qk (first failed on a transpose), 12521 storage (after frames), 12527 mlp_units
  array (neuron spectra of silu(g), u, act for the top-8 neurons of every down comp).

### 2026-09-23 late — results from frames / writes / transfer / mlp_units / output
- frames job slow (~1 min/point while the mlp array read the 91 GB gate/up files); everything else < 1 min.
- Storage (raw stream, linear parts split off): a's codes at pos a set by the embedding + L0 MLP
  (L0 MLP step = largest a-code change, 0.82 of it = alive R_c U_c), then constant; b the same at
  pos b. At `=`: b arrives at L0 (previous-token attention), a at L1-2; result codes from 16m-18m.
  Diff-line codes at `=` from L2 on BOTH ops at low k (comparator-like), a-b mod 10/20 only from 17m.
- Operand code directions at pos a vs pos b: |cos| ~1 at the embedding, 0.35-0.6 at L5-10, back to
  0.6-0.85 at L12-17 (when L15H13/L16H21 copy them), then decaying. Step-to-step |cos| >= 0.9.
- L0 down comps are residue detectors whose U is the spoke of their residue on a's circle (k10, k20,
  k2 checked, figure units_spokes).
- OV: 100 (L15H13) / 130 (L16H21) strongest (v,o) pairs: median phase shift 0.05 period, 71-78 %
  within 0.1 -> copy is identity on the code.
- mlp_units: L16-L18 at `=`: g/u carry 0.00-0.04 result-line share, act 0.14-0.30; created share
  0.64-0.98; the line-product prediction (s_a u_b + s_b u_a + silu-internal cross) fits the new
  code with cos 0.99-1.00 (add), 0.82-0.91 (sub). L19+: inputs already carry the result (0.40 ->
  0.78), created part no longer a x b (cross fit < 0.3): re-writing / harmonic mixing.
  Phase additivity (add, L16-18 units with res share > 0.2): median |peak_out - (peak_a + peak_b)|
  = 0.04 period (sign of the product ignored -> half-period flips count as errors).
  Examples: L16 down c10 a k1@81 + b k1@32 -> sum k1 @12 (pred 13); L18 c23 a k1@56 + b k1@57 ->
  sum k1 @8 (pred 14); L18 c12 k2 48+23 -> 21 (pred 21.4); L18 c4 parity.
- transfer: result-code steps at `=` L16-L25 are 0.78-0.93 accounted for by the alive R_c U_c;
  L30 0.55, L31 0.14 (the last MLP is not captured).
- Mirror: readers at mlp_in.16-18 see b mirrored on sub (k2 .15/.78, k10 .18/.80, k20 .10/.95;
  a stays same). In the raw stream the full code is NOT mirrored (mirror fraction per MLP step ~0.1).
  As seen by the L16 readers: pre-L15-MLP b code identical add/sub (0.99); the L15 MLP write D flips
  sign between ops (Re cos -0.72..-0.89 at k2/k10/k20); its angle to the copy is harmonic-dependent
  (k2 ~90°, k20 ~140°, k10 ~170°). Vector example: L15 down c21 (add-gated, b%10=7) U decodes to 7
  in the add frame; L15 down c72 (sub-gated, b%10=7) U decodes to 2 (antipode) — same residue,
  opposite sign. c16 (add, b%5=1.3) vs c64 (sub, 1.3): U decodes 1.3 vs 3.8 (antipode).
- Op flag: |mean_add - mean_sub| / |x| at `=` ~0.45-0.7 L2-L15, decaying to 0.27 at the end; its
  direction drifts (cos L2 vs L15 0.04; L8 vs L15 0.26); every one-bit relay comp's U is aligned
  with the flag at its write point (cos 0.2-0.54, signed by its op).
- Routing keys: L15 k c48 ('second operand', -21 at b only) is fed by L7-L14 MLP downs at pos b
  (L14 c30, L13 c14, L9 c22, L7 c2); L16 k c5 (51.6 at a, 22.8 at b) by L6-L15 downs.
- Output: each result-code family votes for n = res with the right phase from 17m; peak ~3.5 logits
  (mod 100 family) at L28-30, all families sum 3.85 (add) / 2.24 (sub) at the last point; the alive
  writers' direct votes sum to 3.73 / 2.08 — a committee (max single comp 0.07; L24 c4/c5, L25
  c11/c12 units-digit writers on top).
- Hundreds (add): L27 down c6 / L22 c29 / L25 c33 / L26 c7: gate comps read lin(a)+lin(b) with equal
  weights (corr with a+b 0.75-0.87), the neuron switches at a+b ~ 100 (c6 peaks at a+b 100-104),
  U along W_U[100..199] - W_U[0..99].
- Writer/stream phase agreement (every writer, every harmonic with > 15 % of its inner variance):
  0.80-0.94 of (comp, k) pairs write in phase with the stream code (< 60°), < 10 % opposite;
  L16-18 a+b writers 40/40 in phase. PITFALL: `Reads.energy` was a plain property recomputed over
  all 11.6k comps per access -> a per-comp loop hung for 12 min; now a cached_property.
- Report written: report_auto_interp_vectors.md (figures copied to figures_auto_interp_vectors/).
