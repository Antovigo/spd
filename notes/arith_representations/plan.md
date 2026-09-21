# Integer representations in the residual stream: analysis plan

Decomposition `p-ba5a0c05` (`addsub-all-layers-4xh100-05`, step 40000), read basis from the CI
filter `addsub-05-filter-last-pos-ceiling` (11,604 components alive for the last-position
output distribution). Prompts `<a><op><b>=`, `a, b` in 1..100; subtraction restricted to
`a >= b`. Positions: 0 `<BOS>`, 1 `a`, 2 op, 3 `b`, 4 `=`.

The plan is written before looking at any activation. Every threshold below is either derived
from a null distribution or fixed here; nothing is tuned after the fact. Code:
`param_decomp/arith_repr/`; findings: `report_representations.md`.

## 1. Objects

**Read point.** `r = (layer l, stream s)`, `s in {attn, mlp}`: the post-RMSNorm residual that
layer `l`'s q/k/v projections read (`attn_in.l`) or that its gate/up projections read
(`mlp_in.l`). 64 read points x 5 positions. `x_{r,p}(pi) in R^4096` for prompt `pi`.

**Read basis.** `C_r` = the filter's alive components of the kinds reading at `r` (q, k, v at
attn; gate, up at mlp). `V_r in R^{4096 x n_r}` stacks their input vectors; `B_r` is an
orthonormal basis of `col(V_r)` (SVD; rank `k_r`). Assumption 1 of the brief: everything
causally relevant at `r` lives in `col(V_r)`.

**Coordinates.** `y_{r,p}(pi) = B_r^T x_{r,p}(pi) in R^{k_r}`. `B_r` is orthonormal, so norms
and angles inside the read subspace are preserved: magnitude is kept, not normalised away.

**Prompt quantities.** `a`, `b`, `op`, `res = a op b` (range 2..200 on add, 0..99 on sub); plus
the cross quantities `a-b` on add prompts and `a+b` on sub prompts, cheap to include and a
useful negative control. A quantity that depends on token `n` is only sought at positions
`>= n` (`a` at 1..4, `op` at 2..4, `b` and results at 3..4).

## 2. Hypothesis space (finite, preregistered)

Work in `L^2(P_op)`, the space of real functions on one operation's prompts with the empirical
inner product (mean over prompts). For a quantity `Q` and a period `tau`, let
`M_{Q,tau}` = functions of `Q mod tau` (a `tau`-dimensional space of indicator functions).
The "pure period-`tau`" part is the orthogonal complement inside `M_{Q,tau}` of every coarser
period:

    P_{Q,tau} = M_{Q,tau} (-) sum_{d | tau, d < tau} M_{Q,d}

`tau` ranges over the divisors of 100: `{1, 2, 4, 5, 10, 20, 25, 50, 100}` (`tau = 1` is the
constant). This is an orthogonal decomposition of every function of `Q mod 100` and assumes
nothing about geometry: `P_{Q,10}` is "whatever distinguishes residues mod 10 beyond what
residues mod 2 and mod 5 already distinguish". On `Z/tau` these spaces are spanned by the
Fourier modes with `gcd(k, tau) = 1`, but the *subspace* is basis-free; a circle would be a
finding, not an input. For quantities whose range exceeds 100 (`a + b`), the remainder
`D_Q = functions of Q (-) sum_tau M_{Q,tau}` is the "direct / non-periodic" hypothesis; for
`a`, `b` (range exactly 100) the direct code is indistinguishable from `P_{·,100}` and this
limitation is reported rather than hidden. A nested 1-D "linear in `Q`" direction is tested
inside whatever spaces it projects onto, so a number-line code appears as a special case.

`op` is a 1-D hypothesis (`op` indicator, centred) and is tested only when both operations
are pooled. The second operand is identified **separately per operation** (Antoine,
2026-09-21: `b` may be represented differently on subtraction prompts): on the pooled set the
`b` hypotheses are the op-conditional `b@add` and `b@sub` — functions of `b mod tau` supported
on one operation's prompts, orthogonal to the op indicator — so the same reads decide whether
the two are separable and their principal angles are measured. Every hypothesis is also
fitted on each operation alone and compared across operations (section 4).

Non-orthogonality between quantities is a fact of the design, not a nuisance to hide:
`res = a + b` has additive parts (`f(a) + g(b)`) that no analysis can attribute to a "result"
representation. Fourier modes of `a + b` with `k != 0` factor as `e(ka) e(kb)` and lie entirely
in the `a x b` interaction space, so the *periodic* result hypotheses are clean; only `D_res`
overlaps the `a`/`b` spaces. Each hypothesis therefore gets two scores: **marginal** explained
energy and **unique** explained energy (after partialling out every other hypothesis). Where
the empirical measure is not a product (the subtraction triangle), the pure parts are built by
Gram-Schmidt in divisor order, and the tiny order dependence between incomparable divisors
(4 vs 5, 20 vs 25) is measured and reported.

## 3. Recovering a representation

For hypothesis `H` with an orthonormal basis `Phi_H` (`N x m_H`, centred), at read point `r`
and position `p`:

- fitted means `M_H = Phi_H^T Y` (`m_H x k_r`), explained energy `E_H = ||Phi_H M_H||^2 / ||Y - mean||^2`;
- representation subspace `S_H = row-space(M_H)`, truncated to the directions that survive the
  **value-held-out** test: five folds, each holding out 20 % of the `a` values and 20 % of the
  `b` values (every prompt touching a held-out value is held out). A direction is kept if its
  held-out explained energy exceeds the 99.9th percentile of a label-permutation null (five
  replicates, pooled over hypotheses and directions at that read point and position). This is
  what separates "a code for `a mod 10`" (predicts unseen `a`) from a lookup table over `a`
  (`P_{a,100}` cannot generalise across values by construction).
- `dim(S_H)` and the spectrum of `M_H` are recorded per `(r, p, H)`. The applet's presence map
  is the unique held-out energy of each `H` at each `(r, p)`.

Sanity check of assumption 1 and of the hypothesis set: the energy of `Y` outside
`sum_H S_H` (what no hypothesis explains) and, for a random 1/5 of prompts, the energy of the
raw `x` outside `col(V_r)`.

## 4. Separability and clustering (the model's reads decide)

Fix a read point `r` and a kind `kappa` (q, k, v, gate or up). For an alive component `c` of
kind `kappa` and a recovered subspace `S_A`:

    s_A(c) = ||P_{S_A} V_c|| / ||V_c||

Null: a uniformly random direction in `col(V_r)` (dimension `k_r`), for which
`s_A^2 ~ Beta(dim S_A / 2, (k_r - dim S_A) / 2)`. A component **reads** `A` if `s_A(c)` is
above the null's 99.9th percentile and **ignores** `B` if `s_B(c)` is below the null's 99th.
`A` and `B` are **separable at `(r, kappa)`** iff some alive read reads `A` and ignores `B`
*and* some alive read reads `B` and ignores `A`. This is the operational meaning of "the matrix
can point at `A` without being influenced by `B`" for a bias-free linear read: a weight row
whose projection onto `S_B` is at chance. It is symmetric by construction and has no angle
threshold — only null quantiles.

Two hypotheses that are not separable are joined; **clusters** are the connected components
of that graph, computed per `(r, kappa)` and also with the kinds pooled. A hypothesis that no
alive component reads at all is reported as *unread* and kept out of the graph: nothing points
at it, so it can neither separate from nor merge with anything (letting it in would glue every
cluster together through it). The expected picture
if the model used circles in orthogonal planes would be one cluster per period; the plan does
not assume it.

For each cluster: `S_cluster = sum of its S_H`, its dimension, the PCA spectrum of the pooled
class centroids, and how many dimensions capture 95 % / 99 % of the centroid energy. The
applet shows the centroids (one per residue or value) projected onto the cluster's top
principal axes, norms intact.

Cross-operation comparison: principal angles between `S_{a,tau}` fitted on add and on sub
prompts at the same `(r, p)`; likewise `res` across operations. Shared subspaces mean one
representation of `a` regardless of operation.

## 5. Pipeline

1. **Extract** the alive read vectors `V_c` (q, k, v, gate, up; 32 layers) from the orbax
   checkpoint on CPU -> `V_alive.npz` (`scripts/extract_v.py`).
2. **Harvest** the frozen model's post-norm residuals at every read point and position for
   the 15,050 prompts on one L40 (`scripts/harvest.py`): bf16, one file per read point.
   Only the frozen target is loaded (no components, no CI function).
3. **Analyse** on CPU (`scripts/analyse.py`): bases, hypothesis fits, held-out tests,
   separability, clusters -> one JSON per read point.
4. **Applet** (`app.html`): presence map (read point x position x hypothesis) and per-cluster
   centroid clouds.

## 6. Researcher degrees of freedom, listed

Changes made after the synthetic check (2026-09-21, before any real activation was analysed):
the kept-direction null moved from the 99th to the 99.9th percentile (with ~1,700 direction
tests per read point the 99th let noise directions through), and unread hypotheses were taken
out of the clustering graph (see section 4). Also fixed: the lower spans are orthonormalised
before projecting, so the pure parts are exact on non-product prompt measures.

Fixed by this document: the hypothesis lattice (divisors of 100), the read basis (the ceiling
filter's alive set), post-norm residuals, the value-held-out split (20 %, 5 folds, seed 0), the
null quantiles (99.9 / 99 / 99), the subtraction restriction `a >= b`. Everything else is a
measurement. If any of these has to change, the change and its reason go in the report.
