# Single-pool VPD — semantics spec

Pins the **meaning** of the single-pool VPD training step. An implementation (torch,
JAX, anything) is correct iff it satisfies this document. Ground truth: the stable
torch impl in `goodfire-ai/param-decomp` @ `feature/fsdp-lm-trainer`; production
constants from `param_decomp_lab/experiments/lm/_llama8b/llama8b_l18_b512_2pool_lr_mid.yaml`
(extended 1 → N decomposed layers). Torch file pointers live in §9 (non-normative).

**How to read.** Normative content is: the pseudocode (§4), the invariants (§5–§8),
and the tables (§2, §3, §6). Prose between them is orientation only. Notation:

- `sg[x]` — stop-gradient. `x ~ D` — a fresh independent draw from distribution `D`.
- Shapes in brackets: `[B,T,C]`. `B,T` are *global* batch and sequence length.
- `dp_avg(·)` / exact-global-sum — cross-replica reductions; identity on one device.
- `UPPER_SNAKE` names are **variation points**: pluggable functions with the valid
  instantiation set in §6. ★ marks the production choice.
- Invariants are numbered (`S_`, `N_`, `R_`, `D_`) for citation by audits and tests.
- Bit-exactness with torch is NOT required; identical math / distributions /
  detachment-structure / ordering is.

---

## 1. Glossary — terms that bite

| term | means | NOT to be confused with |
|---|---|---|
| **site** | one decomposed weight matrix, id `(layer, kind)`, `kind ∈ {gate,up,down}` | a transformer layer (a layer owns 3 sites) |
| **component** | one rank-1 slice `V[:,c] ⊗ U[c,:]` of a site's decomposition | a site; a CI-fn unit |
| **chunk** | a sequential group of `sites_per_chunk` sites; the unit one stochastic recon forward masks | a data chunk / micro-batch |
| **"layerwise"/"chunkwise" recon** | masks ONE chunk's sites per forward; the loss is ALWAYS on final logits | ⚠ recon evaluated *at* that layer's output. Site-local recon is NOT this method |
| **clean forward** | suffix forward with every site on its frozen `x @ W` path | the decomposed forward with all masks = 1 (≈ equal only in exact arithmetic) |
| **source** | a `[0,1]` value per channel that a mask is built *from* (noise or adversary) | the mask itself |
| **mask** | `ci + (1−ci)·source`, what the forward consumes | the source; the CI value |
| **delta component** | the `(C+1)`-th maskable channel carrying `x @ (W − V@U)` | the faithfulness loss (same residual, different role) |
| **CI** (causal importance) | the CI fn's per-component prediction in `[0,1]`; `ci=1` ⇒ mask pinned 1 (protected) | a probability; an attribution score |
| **lower / upper leaky** | the two squashings of the SAME CI logits; lower → masks, upper → imp-min | two CI functions (there is one) |
| **faithfulness warmup** | pre-loop phase: V/U trained on `L_faith` alone | the other two "warmups" → |
| **PPGD warmup (`n_warmup`)** | supplemental source-ascent iterations inside each step | LR-schedule warmup; faithfulness warmup |
| **LR warmup (`warmup_pct`)** | linear ramp 0 → start of a schedule's value | the above two |
| **persistent** (PGD) | sources + their optimizer moments survive across training steps | re-initialized-per-step PGD (the eval-only `PGDReconLoss`) |
| **residual-start** | suffix forward from a harvested residual; prefix runs once per batch, never in the graph | KV-caching; activation checkpointing |

---

## 2. Production constants

| | |
|---|---|
| target | Llama-3.1-8B, frozen, bf16 storage |
| decomposed | `layers[18..18].mlp.{gate,up,down}_proj` (bench: `20..31`), right-mult `W: [d_in, d_out]` |
| C | 24576 (bench: 8192); `use_delta_component: true` → source channel dim `C+1` |
| data | fineweb, seq `T=2048`, global batch `B=512`, fresh batch per step |
| coeffs | faith `1e5` · imp `5e-6` · stoch `0.5` · ppgd `0.5` |
| imp-min | `beta 0.2`, `eps 1e-12`, `p: 2.0 → 0.4` linear over `[0, 1]`-frac of training |
| stoch | plan `subset`, routing `uniform_k_subset`, `n_samples 1`, `sites_per_chunk 3` |
| PPGD | scope `broadcast_across_batch`, `n_warmup 2`, clamp-parameterization, Adam(β₁ .5, β₂ .99, ε 1e-8), lr const `0.01` w/ 2.5% LR-warmup |
| V/U opt | AdamW(.9, .999, ε 1e-8, wd 0), lr `1.5e-4` cosine → `0.1×`, **grad-clip 0.01** |
| CI opt | AdamW(.9, .999, ε 1e-8, wd 0), lr `5e-5` cosine → `0.1×`, no clip |
| faith warmup | 400 steps, AdamW lr `1e-3`, wd 0 |
| CI fn | shared transformer: `d_model 4096`, 4 blocks, 64 heads, mlp `[16384]`, RoPE base 10000, bidirectional; `sigmoid_type leaky_hard` |
| steps | 100k; checkpoint cadence per config |

## 3. State

| symbol | shape / type | trains via | persists in ckpt |
|---|---|---|---|
| `W_s` | `[d_in, d_out]` per site | frozen | no (rebuilt from HF) |
| `θ = {V_s [d_in,C], U_s [C,d_out]}` | fp32 master | AdamW (V/U opt) | yes (+ moments) |
| `φ` (CI fn params) | fp32 master | AdamW (CI opt) | yes (+ moments) |
| `src_s` | `[scope-dims, C+1]` per site (§6 SCOPE; ★ `[1, T, C+1]`) | SRC_STEP ascent | yes (+ SRC_STEP moments) |
| `t` (step), schedules `p(t)`, `lr(t)` | scalar | — | yes |

---

## 4. Normative pseudocode

### 4.1 Forward semantics

```
def site_out(x[.., d_in], s, m[.., C]|ONES, d[..]|ONES, route[..]|ALL) -> [.., d_out]:
    Δ_s   = W_s − V_s @ U_s
    y_dec = ((x @ V_s) * m) @ U_s  +  (x @ Δ_s) * d
    return where(route, y_dec, x @ W_s)                      # route=ALL ⇒ y_dec everywhere

def forward(resid[B,T,d], live: set[Site], m, d, route) -> logits[B,T,vocab]:
    suffix forward (layers first..n−1, final norm, LM head);
    each site s ∈ live computes site_out(x, s, m_s, d_s, route_s);
    each site s ∉ live computes x @ W_s                      # frozen path, NOT y_dec(m=1)   (S2)

def clean(resid)   = forward(resid, live=∅)                                                  (S3)
def site_inputs(resid) = the activation entering each site's weight inside clean(resid)
                         # gate_in = up_in = post-ln2 residual; down_in = silu(gate)·up      (S4)
```

### 4.2 CI

```
def ci(φ, site_inputs) -> (lo, up):        # each {site: [B,T,C]}
    logits = CI_TRANSFORMER(φ, site_inputs)          # architecture pinned in §4.6
    return lower_leaky_hard(logits), upper_leaky_hard(logits)                                (S5)

lower_leaky_hard: fwd clamp(x,0,1); CUSTOM bwd: pass g on 0<x≤1;
                  at x≤0 pass α·g ONLY where g<0, else 0; at x>1 zero.  α=0.01               (S6)
upper_leaky_hard: fwd x>1 ? 1+α(x−1) : clamp(x,0,1); ordinary autodiff of that expr.        (S6)
```

### 4.3 Masks and losses

```
def mask(ci_s[B,T,C], src_s[B,T,C+1]) -> (m, d):
    m = ci_s + (1 − ci_s) * src_s[..., :C]
    d = src_s[..., C]                                # delta channel raw: NO ci interpolation (S1)

def recon_div(pred, cln) = Σ_{b,t} KL(softmax(cln[b,t]) ‖ softmax(pred[b,t])) / (B·T)   # fp32 (N3)

def L_faith(θ) = ( Σ_s ‖W_s − V_s@U_s‖_F² ) / ( Σ_s numel(W_s) )                            (S17)

def L_imp(up, t):                                    # per-site grouping                     (S7)
    for s: sum_s[c] = exact_global_sum_{b,t} (up_s[b,t,c] + eps) ** p(t)                 (S8,S9)
    return Σ_s Σ_c (sum_s[c]/(B·T)) · (1 + beta · log2(1 + sum_s[c]))

def L_stoch(θ, lo, resid, cln):
    tot = 0
    for chunk in CHUNKS(sites, sites_per_chunk):     # sequential groups, fixed site order
        repeat n_samples:                            # every draw fresh & independent       (R1)
            m_s = mask(lo_s, u_s ~ U[0,1]^[B,T,C+1])         ∀ s ∈ chunk
            route = ROUTING(chunk, [B,T])                                                   (S11)
            tot += recon_div(forward(resid, chunk, m, d, route), cln)
    return tot / (n_chunks · n_samples)                                                     (S10)

def L_ppgd(θ, lo, src, resid, cln):                  # all sites, route everywhere          (S12)
    m_s, d_s = mask(lo_s, expand(SCOPE, src_s))      ∀ s ∈ sites
    return recon_div(forward(resid, sites, m, d, ALL), cln)
```

### 4.4 The adversary

```
def src_ascent_grad(θ', lo', src, resid, cln):
    return dp_avg( ∂/∂src  L_ppgd(θ', lo', EFFECTIVE(src), resid, cln) )                    (S16)

def src_update(src, g, opt_state):
    src, opt_state = SRC_STEP(src, +g, opt_state)    # ASCENT: maximize L_ppgd
    return PROJ(src), opt_state                                                             (S15)
```

### 4.5 One training step

```
def train_step(state, batch, t):
    resid = sg[ prefix_forward(batch) ]              # per fresh batch                       (S18)
    cln   = sg[ clean(resid) ]                                                               (S3)
    lo, up = ci(φ, site_inputs(resid))               # ONE conceptual CI eval per step;
                                                     # recompute allowed (deterministic)
    # -- supplemental adversary ascents (params & CI detached) --
    set SRC_STEP lr = sched_src(t)                   # stepped once per TRAINING step        (S13)
    repeat n_warmup:
        g = src_ascent_grad(sg[θ], sg[lo], src, resid, cln)
        src, src_opt = src_update(src, g, src_opt)

    # -- main losses: live θ, φ; source detached --
    L = 1e5·L_faith(θ) + 5e-6·L_imp(up, t) + 0.5·L_stoch(θ, lo, resid, cln)
        + 0.5·L_ppgd(θ, lo, sg[EFFECTIVE(src)], resid, cln)

    g_src = dp_avg(∂/∂src of that same L_ppgd term)  # PRE-update θ, live lo — the SAME
                                                     # graph as the main backward           (S14)
    gθ, gφ = dp_avg(∂L/∂θ), dp_avg(∂L/∂φ)            # imp-min global-sum: D2

    src, src_opt = src_update(src, g_src, src_opt)   # the (n_warmup+1)-th ascent           (S13)
    θ = adamw_θ(θ, clip_global_norm(gθ, 0.01), lr_θ(t))                                     (S19)
    φ = adamw_φ(φ, gφ, lr_φ(t))                                                             (S20)
    return state'

before the loop:                                                                             (S21)
    repeat 400: θ = adamw_warm(θ, ∂L_faith/∂θ, lr=1e-3)        # faithfulness warmup
```

### 4.6 CI transformer architecture (pinned)

```
in:   {site: x_s [B,T,d_in_s]}  (clean site inputs, fixed site order)
1.    h_s = rms_norm(x_s)                      # weightless (no learnable scale)
2.    h   = concat_s(h_s) @ W_in + b_in        # → [B,T,d_model]; NO nonlinearity here
3.    × n_blocks (pre-norm):
        h += attn(rms_norm_weightless(h))      # bidirectional MHA, rotate-half RoPE
                                               # base 10000; q/k/v/out bias-FREE
        h += mlp(rms_norm_weightless(h))       # Linear(d→16384)+b → GELU → Linear(→d)+b
4.    logits = h @ W_out + b_out               # → [B,T, Σ_s C_s], split per site in order
init: biases zero; weights fan-in scaled (torch init_param_)
```

---

## 5. Semantic invariants

| id | invariant |
|---|---|
| S1 | `mask = ci + (1−ci)·source` per component channel; the delta channel is the raw source value, never ci-interpolated; CI has no delta output. |
| S2 | A site not live in a forward runs the frozen `x @ W_s` path — zero V/U gradient and zero decomposition rounding from that site. |
| S3 | The recon target `cln` is the frozen-path forward, stop-gradient. Never the `m=1, d=1` decomposed identity (differs in bf16 and pollutes the graph). |
| S4 | CI inputs are the clean site inputs from the frozen path of the same batch. |
| S5 | `lo` and `up` are two squashings of the SAME logits. `lo` feeds every mask; `up` feeds imp-min only; no other crossing. |
| S6 | The squashings' forward/backward are exactly §4.2 — including `lower_leaky_hard`'s grad-sign-gated lower leak (a custom VJP, not autodiff of the forward). |
| S7 | Imp-min groups per site: the `log2(1+sum_s[c])` consumes one site's per-component sum. Merging sites/layers into one group is incorrect (convexity). |
| S8 | `sum_s[c]` is the exact **global-batch** sum, reduced before the `log2` (autograd-aware under DP). Averaging per-rank results after the log is incorrect (Jensen). |
| S9 | `p(t)` anneals linearly `2.0 → 0.4` over the configured frac window; `eps` sits inside the power. |
| S10 | `L_stoch = (Σ_forwards recon_div) / (n_chunks · n_samples)`; chunks are sequential `sites_per_chunk`-groups in the fixed site order. |
| S11 | `uniform_k_subset` routing, per position: `k ~ U{1..|chunk|}` then a uniform `k`-subset of the chunk routes True; non-chunk sites are not live at all. |
| S12 | `L_ppgd` masks ALL sites simultaneously, routes everywhere, and detaches the source; gradient flows to θ and (through `lo`) to φ. |
| S13 | Source updates per training step = `n_warmup + 1`, all through the same persistent SRC_STEP optimizer state; the source LR schedule advances once per training step. |
| S14 | The final ascent's gradient comes from the SAME graph as the main backward: pre-update `θ`, live `lo`. It is applied after backward; it must not use post-update params. |
| S15 | Every source update ends with `PROJ` (★ clamp to `[0,1]`). Init: ★ `src ~ U[0,1]` i.i.d. |
| S16 | Shared-scope sources stay in lockstep under DP: identical init on every replica, `dp_avg`'d grads, identical updates. `per_batch_per_position` shards instead, no sync. |
| S17 | `L_faith` is the global mean of squared delta entries over all sites' parameters (Σ‖Δ‖² / Σ numel), recomputed from live V/U each step. |
| S18 | Each training step consumes a fresh data batch; the prefix harvest runs per batch, outside every gradient graph. |
| S19 | V/U gradients are global-norm-clipped at `0.01` after DP reduction, before the optimizer step. CI fn is unclipped (production). |
| S20 | Both main optimizers are AdamW, `wd=0`, betas `(0.9, 0.999)`, eps `1e-8`; LR cosine to `0.1×` start, no warmup, stepped per training step. |
| S21 | Faithfulness warmup (400 × AdamW lr `1e-3` on `L_faith` alone) precedes step 0; its optimizer is discarded. |
| S22 | Checkpoints round-trip ALL trajectory state of §3 — including adversary sources + SRC_STEP moments + step/schedule counters — such that a resumed run continues the same trajectory (modulo RNG streams). |

## 6. Variation points

| point | valid instantiations | production |
|---|---|---|
| `SRC_STEP` | `adam(β₁,β₂,ε)` with bias correction; `sign` (`src += lr·sign(g)`) | ★ adam(.5, .99, 1e-8) |
| `PROJ` / `EFFECTIVE` | **clamp**: PROJ = clamp[0,1], EFFECTIVE = identity, init U[0,1] · **sigmoid**: PROJ = identity (unbounded raw), EFFECTIVE = sigmoid, init N(0,1) | ★ clamp |
| `SCOPE` | `single (1,1)` · `broadcast (1,T)` · `repeat(n) (n,T), n|B` · `per_batch_per_position (B,T)` | ★ broadcast |
| `ROUTING` | `uniform_k_subset` · `static_probability(p)` · `all` | ★ uniform_k_subset |
| recon plan | `subset(n_samples)` (forwards mask whole chunk) · `per_site` (one forward per site, route=ALL) | ★ subset(1) |
| stoch sampling | `continuous U[0,1]` · `binomial {0,1}` | ★ continuous |
| delta component | on (`C+1` channels) · off (no delta path, no delta mask) | ★ on |
| CI squashing | `leaky_hard` pair (§4.2); other registry sigmoids exist in torch but are out of spec scope | ★ leaky_hard |

A variant choice must hold every invariant not explicitly parameterized by it.

## 7. Numerics (N) and randomness (R)

| id | rule |
|---|---|
| N1 | θ and φ master params fp32; both AdamW moment sets fp32; SRC_STEP moments fp32. Forward compute may be bf16; the frozen target may be stored bf16. |
| N2 | Faithfulness deltas `W − V@U` are computed in fp32 outside any autocast; the sum-of-squares is fp32. |
| N3 | `recon_div` (softmaxes + KL sum) and the imp-min reduction are fp32. Loss scalars and gradient accumulation fp32. |
| R1 | Every stochastic draw (`u`, routing, source init) is independent across sites, positions, forwards, steps — distributions as stated. |
| R2 | RNG stream order/bits need not match torch. |
| R3 | Draws over a sharded batch are independent across ranks (distinct streams). |

## 8. Data-parallel contract (D)

| id | rule |
|---|---|
| D1 | Every loss is defined on the global batch. Per-rank means + averaged grads must compose to the global-batch value for faith/stoch/ppgd (uniform shards). |
| D2 | Imp-min requires the exact global per-component sums *inside* the `log2` (S8) — the one term where mean-of-rank-results ≠ global result. |
| D3 | Shared PPGD sources follow S16. |
| D4 | Validation property: with global batch + seed fixed, the metric trajectory is invariant to device count up to floating-point reassociation (cross-shard reduction order; observed rel ≤ ~1e-5 on the tiny-target harness, `experiments/invariance_check.py`). JAX's counter-based RNG makes even the stochastic draws identical across layouts. |

---

## 9. Non-normative: torch ground-truth pointers & rationale

| spec | torch source |
|---|---|
| §4.1 site/forward, routing | `param_decomp/components.py` (`LinearComponents.forward`), `param_decomp/masks.py` |
| §4.2 squashings | `param_decomp/ci_sigmoids.py` (`LowerLeakyHardSigmoidFunction`, `upper_leaky_hard_sigmoid`) |
| §4.3 faith / imp / stoch / KL | `metrics/faithfulness.py:17` · `metrics/importance_minimality.py` · `param_decomp_lab/metrics/chunkwise_subset_recon.py:73` + `three_pool/step_chunkwise.py::recon_masked_forward` + `three_pool/recon_plan.py` · `param_decomp_lab/batch_and_loss_fns.py::recon_loss_kl` |
| §4.4–4.5 adversary, ordering | `metrics/persistent_pgd_state.py` (init/warmup/step/scopes/`reduce_source_grads`), `metrics/persistent_pgd_recon.py` (`before_backward`/`after_backward`), `param_decomp/train_step.py::run_loss_step` (hook order), `param_decomp/optimize.py:490` (clip → step) |
| §4.5 warmup, schedules | `param_decomp/faithfulness_warmup.py`, `param_decomp/schedule.py::get_scheduled_value` |
| §4.6 CI arch | `param_decomp/ci_fns.py::GlobalSharedTransformerCiFn` (`:289`), `param_decomp/ci_nn_blocks.py` |

Rationale worth keeping: the two squashings give each consumer gradient only in its
permitted direction (masks may push CI up out of saturation; the sparsity penalty may
not push it below 0). The adversary is *persistent* because re-finding the worst-case
ablation from scratch each step under-trains the adversary at any affordable inner-step
count. The `log2` term approximates a description-length / frequency penalty (`L_freq`
in the VPD paper) — its convexity is why S8 demands the true global sum. Fused-linear-KL
and LM-head-bypass are memory/throughput optimizations and must be semantically
invisible (cf. `recon_loss_kl` equivalence).
