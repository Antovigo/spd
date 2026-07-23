# CI-network saturation — lab notebook

Working question: why does the transformer CI network (`global_shared_transformer`)
produce almost exclusively {0, 1} CI values, while the MLP CI networks leave
intermediate values — and can we fix the transformer so gamma annealing can prune
strays the same way?

Testbed: `copy_training_v8d32_partial` (attention copy toy, vocab 8, d_embed 32,
4 redundant tokens). Reference comparison from the completeness work:

| run | CI net | notes |
|---|---|---|
| `copy_v8d32_partial_joint_norm` | layerwise `mlp`, hidden [16] | the good one |
| `copy_v8d32_partial_joint_globalci` | `global_shared_mlp` [256, 256] | |
| `copy_v8d32_partial_joint_txci` | `global_shared_transformer` d64, 2 blocks, 4 heads | many strays |
| `copy_v8d32_partial_joint_txci_beta1` | same, beta 1.0 | worse (L0 12.9) |

All use `sigmoid_type: leaky_hard` (verified in both experiment_config.yaml — forward
is exactly `clamp(x, 0, 1)`; impmin sees the upper-leaky variant, alpha 0.01 above 1).

## 2026-07-23 — diagnosis

Final-step eval metrics (`metrics.jsonl`, step 10000):

| run | total CI-L0 | smoothL0 (no beta) | stoch recon KL |
|---|---|---|---|
| joint_norm | 3.50 | 3.52 | 1.1e-5 |
| joint_globalci | 4.49 | 4.49 | 3.5e-5 |
| joint_txci | 5.25 | 5.27 | 1.7e-5 |
| joint_txci_beta1 | 12.92 | 12.92 | 6.5e-5 |

CI histograms at step 5000 (before the gamma anneal starts at 7500): `_norm` has
substantial intermediate mass (clusters at 0.05–0.6 in most matrices); `_txci` is
already near-binary. The anneal then sweeps `_norm`'s intermediates to 0; `_txci`'s
strays sit at exactly 1 and survive. Final `active_subcomponents.png`: `_norm` textbook
(1 q, 1 k, clean diagonals), `_txci` has extra q/k routing, duplicate v components,
stray o components in all blocks.

**Smoking gun — pre-sigmoid logit histograms at step 10000:**

- `_norm` (layerwise MLP): logits in ≈ [-1.5, +2.6], hugging the hard sigmoid's
  [0, 1] linear window. A component at CI=1 sits at logit ≈ 1.0–1.3.
- `_txci` (transformer): logits span ≈ **[-80, +150]**. Off-components at -20…-60,
  on-components at +10…+150.

Mechanism: the transformer's output head reads an *unnormalized* residual stream
(no final norm — norm grows across blocks and freely during training), CI-fn weight
decay is 0, and nothing anchors logit scale. Deep-saturated components see only the
0.01-leaky impmin gradient over a ~100-unit round trip back to the linear window →
strays at CI=1 are effectively frozen; the gamma anneal can't reach them. Intermediate
CIs would require logits landing in [0, 1] — ~1% of the network's dynamic range —
so almost none exist. The layerwise MLP (scalar input, hidden 16) can't blow up its
logit scale, which is why its strays linger at intermediate CI where the anneal
kills them.

Secondary hypothesis (not load-bearing given the above): a more expressive network
also *wants* to binarize — confident CIs reduce stochastic-masking loss variance.
The fix below doesn't fight that; it only keeps the saturation shallow enough that
sparsity pressure still works.

## Fix (param_decomp/ci_fns.py)

Three standard-transformer changes to `GlobalSharedTransformerCiFn`:

1. **Final RMS norm** before the output head (`final_rms_norm: true` in
   `GlobalSharedTransformerCiConfig`) — decouples logit scale from residual growth.
2. **Readout zero-init, bias 0.5** (unconditional) — every logit starts at exactly
   0.5, mid-window, in gradient reach of both losses from step 0.
3. **Logit soft-cap** (`logit_softcap: 2.0`) — Gemma-style
   `0.5 + cap·tanh((x-0.5)/cap)`, bounds logits to (-1.5, 2.5) so they can never
   leave gradient reach.

## Experiment: three variants on v8d32_partial

Configs in `~/pd_scratch/combine_layers/configs/`, launched via
`txci_variant.sbatch` (jobs 5431–5433), run ids
`toy_model_redundancy/copy_v8d32_partial_joint_txci_{newinit,fnorm,fnorm_cap2}`:

- `txci_newinit` — baseline config, new code → isolates the head-init change.
- `txci_fnorm` — + `final_rms_norm: true`.
- `txci_fnorm_cap2` — + `logit_softcap: 2.0`.

Success criterion: closer to `_norm` on final CI-L0 / alive counts / stray stats
(`txci_stats.py`, extended with per-module + total alive counts) without recon
regression.
