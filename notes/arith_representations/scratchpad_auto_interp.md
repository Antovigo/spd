# Auto-interp of the ceiling-filter components — scratchpad

Working notes, newest at the bottom of each section. Findings that survive go to
`report_auto_interp.md`. No interventions in this pass: everything is read off the
decomposition (V, U, CI, inner activations) and the original model's activations.

## Data

- Decomposition p-ba5a0c05 step 40000, filter `addsub-05-filter-last-pos-ceiling`: 11,604 alive
  components (down 4753, gate 2779, up 2299, o 829, v 666, k 168, q 110).
- `U`, `V` of every alive component: `<run>/analysis/arith_repr/autointerp/uv_alive.npz`
  (job 12407, `param_decomp/arith_repr/autointerp/extract_uv.py`).
- Activations: `<run>/analysis/ci_filter/step_40000/addsub-05-filter-last-pos-ceiling/dataset/`
  — `original/ci.npy` (N,T,A) and `original/inner.npy` (N,T,A) for all 20k prompts (FULL grid,
  sub includes a<b), 5 positions, all alive comps; resid (65,N,T,d) fp16; mlp gate/up.
  The column order of ci/inner is `index.npz: comp_site/comp_index` (sites sorted by name).
- Earlier representation sweep (`report_representations.md`): operand codes lin + mod
  2/5/10/20/25/50 from L0; result codes switch on at mlp_in.19 at `=`.

## Plan

1. Component catalogue (`autointerp/tuning.py`): per component x position x op, CI stats and
   "tuning" of the gated write coefficient `inner * [CI > 0.01]` by group-mean R^2 against a
   fixed feature list (a, b, a+b, a-b, each mod tau; linear; carry/borrow; sign...), plus the
   2-D DFT over the (a, b) torus (sum-diagonal / diff-diagonal / a-only / b-only energy).
2. Write side: residual writers (o, down) -> direct logit effect via final norm + W_U on
   number tokens; periodicity of that logit profile.
3. Wiring: virtual weights between writer U and later reader V (norm-folded), up/gate -> down
   through the MLP neuron basis, v -> o through the head basis.
4. Attention: patterns of the original model's heads from the stored residuals + weights;
   which heads the alive q/k/v/o components live in; what moves from where to `=`.
5. Operator: op-dependence of each component at `=`; how the op token reaches `=`.

## Log

### 2026-09-23 evening — first pass

Jobs: 12408 tuning.npz, 12409 attn_patterns.npy (L,N,H,T,T f16), 12410 wiring.npz, 12416
profiles.npz; catalogue.parquet (catalogue.py). NB: a CPU job that imports jax grabs a GPU on the
L40 node unless `JAX_PLATFORMS=cpu` (12410 did, OOM warnings, then fell back). Fixed in sbatch/q.sh.

- Components are SPARSE: mean on-rate 3-6 % per position for MLP comps. So a component is best
  described by its ON-SET (which prompts turn it on), then by the value of `inner` inside it.
- Main position counts (argmax on-rate): pos1 2565, pos2 2511, pos3 3114, pos4 2969. Pos-4 comps
  concentrate from L15 (102) and grow to L31 (308).
- Pos 1 on-sets over `a` (onsets.py; coarsest period explaining the on-rate profile, R^2>=0.8):
  single values ~500, windows (1 contiguous run on 1..100, width 2-17, start often at a decade
  boundary: x1 187, x0 124) ~780, scattered sets ~700, periodic ~330 (mod 10: 170, mod 50: 110,
  mod 5: 32, mod 20: 11). Same mix at every depth. => `a` is stored by value-bin detectors on
  the number line + residue-class detectors (units digit, mod 50, mod 5).
- Pos 4 labels (group-mean R^2 of the gated write): L10-17 tens(a,b)/units(a,b)/a/b; L18-24
  (a+b)%100, tens(a,b), units(a,b), (a+b)%10, (a+b)//10; L25-31 (a+b)%100 (667) and a+b (525).
  Late comps are result-WINDOW detectors, e.g. L30 gate c399 on for a+b in 101..109 and for a-b in
  2..9 (res mod 100 in 1..9, both ops); L29 up c549 res in 20..28 / 120..125 (tens digit 2);
  L29 gate c140 res mod 100 in 80s; L20 up c13 res mod 10 in {5,6,7} (arc of the units circle);
  L19 gate c55 res mod 10 in {0,1}.
- Attention at `=` (o-comp activity per head, patterns from the original model):
  L15H13 -> b (0.91 add / 0.63 sub), o comps tuned b%10, b//10, b; v comps at pos 3 b%10/b//10.
  L16H21 -> a (0.91 add / 0.58 sub + 0.19 b), o comps a, a%10, a//10; v comps at pos 1 a%10.
  L18H30 (85 o comps) BOS .52/b .25 on add, b .49 on sub; o tuned lin(a), tens(a,b), b, a+b.
  L13H7 add: b .67; sub: `=` .43, b .30. L20H2 add a .14 b .34; sub a .35 b .33.
  Into pos 3: L1H24 and L5H22 copy a (tens(a,b) labels at pos 3), L17H9 / L24H10 read the op.
  Many heads' patterns differ between add and sub (first handle on the op mechanism).
- DLA consistency (tuning over res vs the component's direct logit on the result token):
  median corr 0.47 for `=`-down comps L20-31 (0.2 at L15-19, ~0 before). Top ones corr 0.9+.

### Result stage / output stage / op (same evening)

- (a,b)-grid pictures of `=` comps (figs/grid_examples_*.png): L15-16 bands in a or b only (the
  operand copies), L16-18 2-D blobs/lattices (conjunctions of a- and b-digit classes), L19-22
  anti-diagonal stripes on add (res mod 50/20/10/100) which become DIAGONAL stripes (a-b) on sub
  for the SAME component; L25-31 single thin anti-diagonal lines (res = v mod 100).
- DFT-line energy of the gated `=` writes, weighted by write variance (add): a-only+b-only 0.75
  at L10-14 -> b-only 0.78 at L15 (L15H13 copy) -> a 0.33 / b 0.52 at L16 -> sum-diagonal 0.39 at
  L19 (periods 50 & 100 first: 0.18, 0.10), 0.56 at L20, 0.72-0.78 L21-30. Periods 10/20/5/2 on
  the diagonal come in at L20-21. Sub: diff-diagonal 0.24 L19 -> 0.4-0.45 L21-29 (weaker: half the
  sub grid has a<b where the model says "?\n"). Sub has diff-diag k=1 energy 0.1-0.2 already at
  L10-14 = sign(a-b) (triangle on-sets), a comparator computed early.
- Inner (ungated) of `=` gate/up comps: additive f(a)+g(b) share 0.94-0.96 at L14-18, result
  share 0.54 at L19-20 and 0.86-0.91 from L21 (add). So the RESIDUAL at `=` changes from
  "a-code + b-code" to "result code" across L18-20 MLPs.
- Op phase test (op_phase.py, job 12425): Fourier coefficient z_op(k) of the class means over b
  at `=`; same = |<z_sub,z_add>|, reflect = |<z_sub, conj z_add>|. b@pos3: same (k10 .88), never
  reflected. b@`=`: mlp_in.15 same .88/.08 (L15H13 copies b identically), then attn_in.16 (after
  L15's MLP) same .14 / reflect .57 at k10, k20 .11/.55, k1 .16/.30, k2 .25/.36 -> on sub the b
  code at `=` is MIRRORED (b -> -b) by L15's MLP. a@`=` is never reflected (k10 .88/.02 at
  mlp_in.16). k5 ambiguous (both high).
- L15 MLP at `=` has sub-only comps: gate c72 (on 99 % sub, 0 % add), up c117 (99.8 % sub), down
  c10 (93 % sub), and sub-only b-tuned downs c64 (b%5), c72 (b%10), c80 (b). = the reflector.
- Op flag chain at `=`: heads L0H23, L1H6, L2H2 attend `=` -> op token (0.38-0.56); then ~1-4
  binary MLP comps per layer L1-L19 on for exactly one op (add-flags A / sub-flags S), e.g.
  L6down11S, L7down23S, L8down928S, L9down50S, L12down56S, L13down127S, L14up158S; these are the
  top upstream writers of L15 gate c72 / up c117 (vw * op difference of mean write).
- Output stage (delta_profiles.npz: direct logit on token res+Delta from each late writer, add):
  res%10 family (49 comps) boosts every token with the right units digit (Delta = 0, +-10, +-20,
  +-50, +-100 all ~+0.9, Delta=+-1 slightly negative); res%100 family (280) boosts Delta=0 (+2.0)
  and +-100 (+1.4), neighbours +-1 (+0.85), suppresses +-10; res (205) boosts Delta 0..+-2 (a
  magnitude neighbourhood); res//10 (72) boosts a flat +-10 window (tens-level magnitude);
  tens(a,b) similar coarse window. Sum of all late writers: Delta0 +6.8, +-1 +3.2, -100 +4.2 (!),
  +100 +0.8: hundreds are weakly resolved by direct effects; model's add errors on 110-199 are
  4 % "res-100".
- Model behaviour (original): add acc 94.5 %; sub a>=b 53.9 % (44 % top-1 is "?\n"); sub a<b
  never outputs "-" (71 % "?\n").

### Code attribution, result mechanism, write-up (same evening)

- code_attrib.py (job 12436): alive writers + embedding explain 0.8-1.0 of the operand codes and
  0.96-1.02 of the a+b / a-b codes at mlp_in.19-26 (0.87 / 0.68 at mlp_in.31). a code: embedding
  1.05 at mlp_in.0, L0 MLP 0.33-0.62 at mlp_in.1, L0-L4 MLPs by mlp_in.5, L11 MLP re-writes mod 10
  (0.31 at mlp_in.12), L11-L15 MLPs at attn_in.16. At `=`: L15H13 0.75/0.55/0.51/0.32 of b mod
  2/5/10/20; L16H21 0.85/0.63/0.60/0.29 of a; L15 MLP 0.2-0.44 of b mod 5/10/20 at mlp_in.16.
  First result code (mlp_in.19): L18 MLP (mod 2 .61, mod 20 .49, mod 5 .39, mod 10 .30) + L17
  MLP (mod 10 .26, mod 2 .30); then each layer's MLP re-writes it (L19 .48 mod 10 at mlp_in.20,
  L20 .60 mod 20 at mlp_in.21, L25 .33 mod 100 at mlp_in.26 ...).
- Units-digit plane of the top L17-18 writers: L17 down c67 = a odd AND b odd, c22 = a odd AND
  b even, L18 down c4 = a+b even (61 % of the mod-2 code alone): parity as XOR of ANDs.
  L17 down c52 rectangle (a%10 1-4 AND b%10 0-4); L18 down c88/c32, L17 c39 digit-sum bands.
  Dense L18 c12/c21/c22/c25 (on 65-80 %) = sinusoids of a+b with period ~50.
- q/k comps of L15H13, L16H21 are on for every prompt -> routing by position.
- PITFALL: in quick pandas prints the index is sometimes a row number of a derived table, not
  the dataset column; component ids must come from `df.cidx[col]`. Caught and fixed for the
  pos-1 examples and two DLA examples (L27 down c6, L29 down c132) before they reached the report.
- delta_profiles: first version subtracted a per-writer mean over prompts, which leaves a
  prompt-selection ramp; now dla is centred over tokens 0..200 per writer (pos4_tables.py). The
  all-writers sum still has a "smaller numbers up" tilt not tied to the answer.
- Report: report_auto_interp.md; figures in <run>/analysis/arith_repr/autointerp/figs/.
