# Integer representations in the residual stream — findings

Companion to [`plan.md`](plan.md), which fixes the method before any data was read. This file
collects what the data said, in the order it was found. Decomposition `p-ba5a0c05` step 40000,
read basis = alive set of `addsub-05-filter-last-pos-ceiling` (6,022 residual-reading
components: q/k/v/gate/up over 32 layers), prompts `a+b=` (10,000) and `a-b=` with `a >= b`
(5,050). Data: `~/out/arith_repr/` (`V_alive.npz`, `resid/`, `analysis/`); code
`param_decomp/arith_repr/`; branch `experiment/arith_representations`.

## Setup log

- 2026-09-21: `V_alive.npz` extracted (job 12076); residual harvest on one L40 (job 12077),
  64 read points x 5 positions x 20,000 prompts, bf16.
- Synthetic check of the pipeline (jobs 12078-12082): planted `a mod 10`, `b mod 10`,
  `res mod 10` circles, a linear-`a` direction and an `a == 37` lookup feature in 40-D noise
  (`~/pd_scratch/dual_obj_jax/arith_repr/synth_test.py`). It caught two defects before any
  real data was read (plan section 6 records the changes): the lower spans must be
  orthonormalised before projecting (the pooled add+sub measure is not a product, and
  without it `a:50` "found" the mod-10 circle), and the 99th-percentile null let noise
  directions through. After the fixes: `a:10`, `b:10`, `res:10` come back as 2-D subspaces
  with held-out R² 0.84, `a:100` (the lookup) has zero generalising energy, and the clusters
  are `{a:10}`, `{b:10}`, `{res:10}` plus the linear-`a` group (`res:direct`, `cross:direct`,
  `op`), which is the documented additive ambiguity: a linear `a` is also a linear part of
  `a+b` and of `a-b`.
- Synthetic applet: [`synth_app/index.html`](synth_app/index.html) (open over `file://`),
  one fake read point `mlp_in.0`, with a subtraction-only `b mod 5` code added to the plant.
  On the pooled set `b@add:10` and `b@sub:10` share one cluster (the same planted circle) and
  `b@sub:5` is its own; add-vs-sub cosines are 1.0 for the shared codes. Small leaks
  (`res:25` at 0.013 generalising energy on subtraction) come from the incomparable-divisor
  overlap on the triangle measure (`lattice_overlap` 0.25 there, 0.07 pooled) — the unique
  energy (0.001) is the honest number for those.
- Analysis array (job 12093): 64 read points x positions 1-4 x {both, add, sub}.

## Findings

(pending)
