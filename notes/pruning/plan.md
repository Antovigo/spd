# Arithmetic component pruning: plan

Goal: from the components that survive the ceiling CI filter on addsub-08, extract ONE fixed subset
whose model does `a + b` / `a − b` as well as possible, so that we can study that submodel first and
the remaining components later. The mask is global (one on/off bit per component, the same for
every prompt and position), so the CI network does no work at all: whatever the submodel does, the
components themselves do. Priorities, in order: (1) do not lose a mechanism that matters for
arithmetic; (2) remove as many arithmetic-unrelated components as possible. The exact optimal
subset is not required.

## 1. Definitions

- **Run.** `p-ba8b0c08` (`addsub-all-layers-08`), checkpoint step 40000. It is a decomposition of
  every linear site of Llama-3.1-8B: 32 blocks × 7 sites (`q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `gate_proj`, `up_proj`, `down_proj`) = 224 sites, 124,928 components in all. Component `c` of a
  site with input width `d_in` and output width `d_out` has an input vector `V_c` (length `d_in`) and
  an output vector `U_c` (length `d_out`). When it is on, it adds `(x · V_c) U_c` to the site output
  for the site input `x` (a `d_in` vector at one position).
- **Operands and prompts.** `a`, `b` are integers in 1..400; `op` is `+` or `−`; `r` is the true
  answer, `a + b` or `a − b`, an integer. `P` is the set of prompts `"<a><op><b>="` (BOS + 4 tokens
  = 5 positions) with `r ≥ 1`: 160,000 additions and 79,800 subtractions, `N = |P| = 239,800`.
- **Answer token.** `y_p` is the token of `r` for prompt `p`. Llama-3 encodes every integer 0..999 as
  one token, so `y_p` is `r` itself (asserted for every `p`).
- **Integer tokens.** `V_int` is the 1,000 tokens `"0"`, `"1"`, ..., `"999"` (no leading zeros);
  `x` denotes an integer in 0..999 and its token.
- **Starting set.** `S₀` is the 15,768 components whose output CI under the filter
  `addsub-08-filter-last-pos-ceiling` exceeds 0.01 at some position of some pool prompt (its
  `alive/alive.json`). `S` is any subset of `S₀`; `S_j` is the set after pruning round `j`
  (integer, `j ≥ 0`).
- **Submodel.** `M_S` is Llama-3.1-8B with every linear site replaced by the sum of the components
  of `S` at that site, `y = Σ_{c ∈ S} (x · V_c) U_c` (the weight delta is OFF; embedding, norms and
  unembedding are Llama's). `Llama` is the original model.
- **Last-position distributions.** For a model `M` and prompt `p`, `p_M(· | p)` is the softmax over
  the full vocabulary at the last position (the `=`), in fp32.
- **Losses.** `CE_p(S) = −log p_{M_S}(y_p | p)` (nats). `CE(S)` is its mean over `P`.
  `CEint_p(S) = −log [p_{M_S}(y_p | p) / Σ_{x ∈ V_int} p_{M_S}(x | p)]`, the CE among integer tokens.
- **Ablation effect.** `δ_p(c; S) = CE_p(S ∖ {c}) − CE_p(S)` and
  `ΔCE_c(S) = CE(S ∖ {c}) − CE(S)`, its mean over `P`. Positive = `c` helps. Its estimate is
  `ΔĈE_c` with standard error `SE_c` (§3).
- **Global mask.** `m_c` ∈ [0, 1] scales component `c` everywhere (`m_c = 1` for `c ∈ S`, 0
  otherwise); `g_c = ∂CE/∂m_c` at the current `S`, over all of `P`. Removing an on component
  changes `CE` by about `−g_c`; adding an off one by about `+g_c`.

## 2. Stage 0: references and checks

1. **Factored forward.** Own JAX forward of `M_S` in factored form: at every site,
   `y = ((x V_S) ⊙ m_S) U_S`, where `V_S` stacks the `V_c` of the site's `S₀` components as columns,
   `U_S` stacks the `U_c` as rows, and `m_S` is their mask vector (per row of the batch, so
   different rows can ablate different components). The residual stream, norms, attention and
   softmax are fp32; matmul precision is a config knob (default fp32 for the screen; the full-`P`
   passes of §4 use bf16 inputs with fp32 accumulation, whose noise is far below the 0.01-nat
   round tolerance). This fp32 evaluation is the DEFINITION of `M_S` here: the same weights as the
   decomposition, which trained in bf16, without its rounding noise. Its agreement with the native
   bf16 masked forward is measured (item 3), and `S_final` is also scored with the native forward
   (§5). Factored, the 15,768 components cost ~0.4 GFLOP per token, against 16 for
   dense Llama; the unembedding at the last position (1 GFLOP per prompt) dominates.
2. **Partial forward.** The same forward started at block `l` (integer, 0..31) from a cached
   block-input residual, so ablating a component of block `l` only reruns blocks `l..31`.
3. **Validation on the pod, before anything else:**
   - our `M_{S₀}` against the decomposition's own masked forward with the `alive_union` global mask,
     delta off (`ci_filter/ablation.py`): last-position KL ≈ 0 up to bf16 noise;
   - our dense Llama against `placed.clean_forward`: same;
   - partial forward from cached residuals = full forward, at every restart block `l`, for the
     per-component masks and every precision path (exact where the arithmetic order is the same,
     otherwise a measured tolerance well below the `SE_c` floor);
   - **bf16 / TF32 noise floor.** For 100 random components of `S₀`, compute `δ_p` on `R` (§3) with
     the fp32 path and with the fast path; the fast path is allowed for the screen only if
     `|ΔCE_fast − ΔCE_fp32| < 0.1 · SE_c` for ≥ 99% of them.
4. **Baselines on all of `P`:** `CE`, `CEint`, accuracy (`argmax over V_int == y_p`), `P(V_int)` and
   `KL(p_Llama ‖ p_M)` for `Llama` and `M_{S₀}`, split by `op` and by `r` in bins of 100.

## 3. Stage 1: screening `S₀`

**Prompt sets.**
- `R`: one uniform random sample of 4,096 prompts of `P` (fixed seed), shared by every component.
- `H_c`: for each `c ∈ S₀`, the 256 prompts of `P` with the highest max-over-positions output CI of
  `c` under the filter's CI fn (lower leaky, ceiling applied). Computed in one streaming pass over
  `P` with a running top-256 per component.

**Order.** Phase A (`R`) first: one forward of `M_{S₀}` over `R` caches the 32 block-input residuals
(fp32, 32 × 4096 × 5 × 4096 × 4 B = 10.7 GB) plus the baseline and Llama last-position log-probs on
`R` (2 × 2.1 GB). Each component then costs one partial forward from its block on `R`. Phase B
(`H_c`) second: per component, full forwards of `M_{S₀}`, `M_{S₀ ∖ {c}}` and Llama on its 256
prompts (no shared cache). Phase A alone already gives a usable screen.

**Estimator** (stratified: `H_c` exactly, the rest of `P` from `R ∖ H_c`). For any per-prompt
quantity `f_p` with mean `F` over `P`:

    F̂_c = (1/N) [ Σ_{p ∈ H_c} f_p + w_c Σ_{p ∈ R ∖ H_c} f_p ],   w_c = (N − 256) / |R ∖ H_c|
    SE(F̂_c) = ((N − 256)/N) · sd_{R ∖ H_c}(f) / √|R ∖ H_c|

`ΔĈE_c` and `SE_c` are this with `f_p = δ_p(c; S₀)`. Weighted second and third moments use the same
weights (`1` on `H_c`, `w_c` on `R ∖ H_c`). Sums over `R ∖ H_c` are obtained as the sum over `R`
minus the sum over `R ∩ H_c` (phase B knows which of its prompts are in `R`), so phase A never needs
`H_c`.

**Recorded per component** (all weighted as above; "Δ" = ablated minus `M_{S₀}`):
- `ΔĈE_c`, `SE_c`; mean, sd and skew of `CE_p(S₀ ∖ {c})` and of `δ_p`; the 1%, 50%, 99% quantiles
  of `δ_p`; the fraction of prompts with `δ_p > 0.1` and with `δ_p < −0.1`.
- Δ`CEint`, Δaccuracy, Δ`P(V_int)`, Δ entropy over the full vocabulary and over `V_int`.
- `KL(p_Llama ‖ p_ablated)` and `KL(p_{M_{S₀}} ‖ p_ablated)` (the latter is "KL to the alive-only
  model").
- **Integer shift.** With `q(x) = p(x) / Σ_{x' ∈ V_int} p(x')` the integer-conditional
  distribution, `Δq(x)` = weighted mean over prompts of `q_ablated(x) − q_{M_{S₀}}(x)`, for every
  `x` in 0..999 (1,000 floats).
- **Answer-relative shift.** `Δq_rel(d)` = weighted mean over prompts of
  `Σ_x [x − r = d] (q_ablated(x) − q_{M_{S₀}}(x))`, for every integer `d` in −999..999
  (1,999 floats).
- Derived offline, for each period `τ` in {2, 5, 10, 20, 50, 100} and residue `k` in 0..τ−1:
  `Σ_{x mod τ = k} Δq(x)` and `Σ_{d mod τ = k} Δq_rel(d)`. Every `τ` divides 1,000, so the classes
  of `x` have equal size. Example: a component whose ablation raises `q(x)` for `x mod 20 = 5`
  shows a positive entry at `(τ, k) = (20, 5)`.
- Per-prompt `δ_p` on `R` and on `H_c` (fp16, ~140 MB in all), for later analysis.

**Estimator check.** For 30 components spread over the `ΔĈE_c / SE_c` range, measure `ΔCE_c(S₀)`
exactly over all of `P` (one factored forward over `P` each) and report `(ΔCE − ΔĈE)/SE`; its spread
should be close to 1.

## 4. Stage 2: pruning rounds

Round `j` (from `S_j`):

1. **Removal candidates:** components of `S_j` with `ΔĈE_c + z · SE_c ≤ ε`, most negative `ΔĈE_c`
   first, at most 10% of `|S_j|`. `z = 1` (dimensionless); `ε` is a per-component negligible effect
   in nats, set after stage 1 from the distribution of `SE_c` (recorded in the run config). Noisy
   components with mean near 0 are protected; quiet null components still go.
2. **Remove and check.** `S' = S_j ∖ batch`; compute `CE(S')` exactly over all of `P`. The batch is
   accepted if `CE(S') − CE(S_j) ≤ 0.01` nats (per round) AND `CE(S') − CE(S₀) ≤ 0.03` nats (global
   budget, so the per-round allowance cannot accumulate). Otherwise put back the less-negative half
   of the batch and recheck (repeat until it passes or the batch is empty). `S_{j+1}` = the accepted
   `S'`.
3. **Gradient pass** on `S_{j+1}`: `g_c` for every `c ∈ S₀` (on and off), in one forward and
   backward over all of `P` (microbatched, accumulated). With the factored forward this is `jax.grad`
   with respect to the mask vectors.
4. **Disagreement re-ablations.** For each `c ∈ S_{j+1}` with
   `|−g_c − ΔĈE_c| > max(3 · SE_c, ε)`, re-measure `ΔĈE_c` and `SE_c` by direct ablation on
   `S_{j+1}` (stage-1 estimator, `R` cache rebuilt on `M_{S_{j+1}}`) and update its rank. At most
   2,000 per round, largest disagreements first.
5. **Re-addition.** Each pruned `c` with `g_c < −θ` (`θ = 10⁻³` nats) is a candidate; it is re-added
   if direct ablation-in (the stage-1 estimator with `S_{j+1} ∪ {c}` as the base and `c` removed)
   gives `ΔĈE_c − z · SE_c > θ`. Removal needs the upper bound below `ε` and re-addition the lower
   bound above `θ > ε`, so the same estimate cannot both remove and re-add a component; a component
   re-added in round `j` is not a removal candidate in round `j`. There is no permanent exclusion.

**Stopping.** When step 1 yields no candidate and step 5 re-adds nothing, run a fresh full screen of
the current `S`: the stage-1 ablation of every survivor AND the ablation-in of every excluded
component (not only those with an adverse gradient). Continue the rounds from it if it yields
removals or re-additions. Stop when a fresh screen yields neither. Hard caps: 30 rounds, 3 fresh
screens.

**Logged per round** (JSONL): `|S_j|`, the batch, rollbacks, `CE`, `CEint`, accuracy and KL to Llama
on all of `P`, the number of disagreements and re-additions.

## 5. Final report

For `S_final`, `S₀` and Llama on all of `P`: `CE`, `CEint`, accuracy, `P(V_int)`, KL to Llama, split
by `op` and by `r` in bins of 100, in the fp32 evaluation and in the decomposition's native bf16
masked forward; the final per-component screen of `S_final`; the composition of
`S_final` by layer and site kind; the removal round of every pruned component.

## 6. Code and outputs

- Package `param_decomp/pruning/` (reuses `ci_filter.pool`, `ci_filter.attribution.answer_targets`,
  `ci_filter.checkpoint.trained_ci`, `experiments.lm.load_run.restore_jax_run`):
  - `model.py`: factored and dense Llama forward, partial forward from a cached residual, the
    per-row mask, the mask gradient;
  - `screen.py`: `H_c` selection, phases A and B, the estimator, the recorded statistics;
  - `prune.py`: the rounds of §4;
  - `run.py`: entry point, `--stage {references, select, screen, prune, report}`, every stage
    resumable from its outputs;
  - `config.py`: a pydantic config (run, step, filter id, sizes, seeds, `z`, `ε`, `θ`, caps,
    precisions).
- Tests (`param_decomp/tests/pruning/`, tiny random Llama on CPU): factored forward with every
  component = the decomposition's masked forward with all masks 1, delta off; partial = full
  forward; per-row mask ablation = refolding without `c`; mask gradient = finite difference; the
  estimator is unbiased on synthetic data with a known mean; the `Δq_rel` scatter on a hand case.
- Outputs: `<run_dir>/analysis/pruning/step_40000/<prune-id>/` — `config.yaml`, `references.json`,
  `hc.npz`, `screen/` (per-component table, vectors, per-prompt `δ_p`), `rounds.jsonl`,
  `sets/S_<j>.npz` (`{site: (C,) bool}`), `report.json`, `logs/`.

## 7. Running

- One A100 80 GB (RunPod). Resident: Llama bf16 16 GB, the `S₀` factors in fp32 ~0.8 GB, the R cache
  ~15 GB; the CI fn and the full decomposition only during the `select` and `references` stages.
- Data pushed to the pod: `launch_config.yaml` and `ckpts/40000/decomposition` (10 GB), the filter's
  `config.yaml`, `alive/` and `training/ci_fn` (3.4 GB); Llama weights from HF. Code is rsynced from
  the worktree (no push needed). Tooling in `notes/pruning/pod/`, adapted from
  `notes/ci_filter/pod/`.
- Time estimates (computed, not measured): `select` ~10 min; screen ~1–2 h (the unembedding at fp32
  dominates); a round ~20–40 min (exact `CE` over `P` ~4 min per check, the gradient pass ~15 min,
  re-ablations ≤ 10 min). Removing ~10% per round, going from 15,768 to a few thousand components
  takes ~15 rounds, so ~6–10 h; ~10–14 h in all.
