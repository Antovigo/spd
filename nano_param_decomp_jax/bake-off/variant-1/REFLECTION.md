# Variant 1 — Reflection

## Final loss values

**TMS** (5000 steps, ~30s on a B200):

- `faith`: 7.4e-01 → 1.7e-07 (>1e6× reduction)
- `stoch`: 3.1e-03 → 2.5e-04 (>12× reduction)
- `imp` (mean ci^0.9): hovers around 0.05 — most components shut off

Disentanglement is partial: out of C=5 components on W1 and W2, two clearly
align with single features and the rest collapse to ~0. This is a TMS-specific
artefact (the 2D bottleneck forces feature mixing in the target itself) rather
than a method issue.

**Toy MLP** (5000 steps, ~30s):

- `faith`: 2.6e-02 → 6.1e-03 (~4×, floor set by rank-16 approx of rank-64
  random teacher matrices — the floor is the singular-value tail squared)
- `stoch`: 3.8e-01 → 1.1e-07 (~3e6× reduction)
- `imp`: saturates at 1.0 — every component is needed (random teacher → no
  sparse structure)

Both runs hit their structural floors smoothly and quickly.

## What worked elegantly

- **The "user writes the forward" contract is dead simple.** The full API for
  adding a target is two pure functions returning `(out, pre_acts)`. No
  abstract base class, no module decoration, no surgical `tree_at`. You read
  one train script and you know the entire user-facing surface.
- **JAX is happy here.** The whole loss flows as one pytree of dict→dict→array,
  `eqx.filter_value_and_grad` slices V/U/CI cleanly, optax handles the dicts
  without complaint. `jit` worked first try (modulo passing optax objects by
  closure rather than as arguments).
- **The two-optimizer pattern is one tuple unpack:** grads come back as
  `(grad_vu, grad_ci)`, and each optimizer eats its slice. Adding separate LRs
  or schedules is one line per optimizer.
- **W_delta as a function of V/U** (recomputed each step, not stored) made
  the trainer state simpler — `vu` is the only thing the optimizer sees, and
  `materialize_components(vu, target_weights)` is a 3-line helper.

## What was friction

- **The user has to write the forward twice.** For TMS this is ~10 extra lines;
  for a transformer block this would be 50+ lines duplicating attention logic.
  In real research you'd want a single forward that takes an optional masking
  callback. This duplication is the main cost of the pattern.
- **`eqx.filter_value_and_grad` only differentiates the first argument.** You
  have to pack `(vu, ci_fn)` as a tuple and unpack the grads. Mild but
  surprising; minor docstring trap.
- **`pre_acts` is a side channel.** The user has to remember to populate it
  with the right keys in both forwards. A typo (e.g. forgetting `pre_acts["W2"]
  = h` in only the decomposed forward) wouldn't crash at JIT — it would fail
  at CI-application time with a missing key. Trivial to debug, slightly
  loose contract.

## Would I bet on this for the full library port?

**No — but it's a clean ceiling against which to judge the others.** For TMS
and toy MLPs the duplication is fine; for transformers with 30+ blocks and
nested attention/MLP, asking users to maintain a second forward in lockstep
with the first is a maintenance hazard. The right answer is probably variant 2
(typed pytree leaf with a `linop` dispatch) — same transparency, no
duplication — but I expect that to introduce some pytree gymnastics this
variant doesn't have. I'd want to read variant 2 before committing.

The trainer code in this variant is genuinely small (~180 LOC) and easy to
modify (adding e.g. layerwise recon is just another loss term on `cis`/`acts`
inside `loss_fn`). If the duplication concern can be addressed (e.g. via a
masking-callback convention that the user puts in their forward), this
factoring would scale fine.
