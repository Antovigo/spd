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
- Synthetic check of the pipeline (job 12078): planted `a mod 10`, `b mod 10`, `res mod 10`
  circles, a linear-`a` direction and an `a == 37` lookup feature in 40-D noise.

## Findings

(pending)
