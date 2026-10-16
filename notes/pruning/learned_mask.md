# Learned global mask vs blockwise pruning

**Question.** Does a mask trained by gradient descent (in the spirit of the circuit pruning of
Gao et al. 2025, *Weight-sparse transformers have interpretable circuits*,
arXiv:2511.13653) find a smaller or better arithmetic subset of `S₀` than the blockwise
screen-and-prune of `plan.md`?

**Setting** (unchanged from `plan.md`): run `p-ba8b0c08`, step 40000, ceiling filter
`addsub-08-last-pos`. `S₀` is the filter's alive set of 15,768 components. Components are
zero-ablated with the weight delta off, and one mask is shared by every prompt. `P` is the set
of 239,800 prompts `"<a><op><b>="` with operands `a`, `b` integers in 1..400, operation `op`
in {+, −} and true answer `r ≥ 1` (an integer). `Q ⊂ P` is the fixed 32,768-prompt check set of
the blockwise run (`strata.npz`). `ce_int` is the cross-entropy in nats of the answer token
after renormalizing the next-token distribution over the integer tokens 0..999.

**Baseline.** The blockwise set `S_32` (11,006 components) reaches `ce_int` 0.019 over all of
`P`.

## Method (`param_decomp/pruning/learn.py`)

- Each component `c` of `S₀` (an index over the 15,768) has a real logit `ℓ_c`, initialized to
  `learn_init` = 1, so the run starts from `S₀` with every component on.
- **Forward.** The mask is hard: `m_c = 1` if `ℓ_c > 0`, else 0. The mask the run trains is
  therefore exactly the mask it is scored with, and no binarization step is needed.
- **Backward.** The gradient passes straight through the hard mask, using the surrogate
  `∂m_c/∂ℓ_c = σ'(ℓ_c/T)/T`. Here `σ` is the logistic function and `T` = 1 is the surrogate
  temperature.
- **Loss.** `mean over B of ce_int + λ · Σ_c σ(ℓ_c/T)`.
  - `B` is a fresh minibatch of 512 prompts from `P \ Q` at every step.
  - `λ > 0` is the sparsity weight, in nats per component.
  - Both terms share the surrogate factor, so `ℓ_c` drifts down whenever the linearized cost
    of dropping `c` (`−∂ ce_int/∂m_c`) is below `λ`. A component that was switched off still
    receives the gradient of adding it back, so every decision is reversible.
- **Optimizer.** Adam with learning rate 0.02 and moment decays 0.9 / 0.999, for 600 steps.
- **Precision.** The gradient pass uses the same bf16 matmuls (`full_precision`) as the
  blockwise run's exact checks.
- **Sweep.** `λ ∈ {1e-6, 3e-6, 1e-5, 3e-5, 1e-4}`. The blockwise removal threshold `ε` was 1e-5.

## Comparison

- `learn/baseline.json` scores `S₀` and `S_32` on all of `Q`. Each `learn/lam_<λ>/final.json`
  scores that λ's final mask on `Q` with the same function: size, `ce`, `ce_int`, accuracy,
  `P(integer)` and KL to Llama, over all prompts, additions and subtractions.
- **Verdict.** Both criteria use `S_32`'s `ce_int` on `Q` from `baseline.json` (not its all-`P`
  0.019). The learned mask wins if the λ whose mask keeps about 11,006 components reaches a
  lower `ce_int` on `Q` than `S_32`, or if some λ keeps clearly fewer components at no higher
  `ce_int` on `Q`.
- `learn_config.json` in each λ directory pins the settings that shape it; resuming under
  other settings fails.
- `log.jsonl` records the size and the `ce_int` on the first 4,096 prompts of `Q` every 50
  steps.

## Run

The run uses the A100 pod tooling in `pod/`. `push.sh` with `PRUNE=addsub-08-prune` ships the
blockwise run's small outputs, so the `learn` stage reuses its `Q` and its sets. Launch on the pod
with `NAME=addsub-08-prune ./run.sh --stages learn`.
