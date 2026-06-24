# Full-network roadmap — progress log

## Objective 1 — Does releasing the impmin coefficient help? **YES (adopted).**

Three runs (ablation on `model.layers.18.mlp.down_proj` only, 20000 steps):

| run (final) | impmin schedule | recon | PGD | L0 | n_alive |
|---|---|---|---|---|---|
| **sched-release** | anneal 5e-4→5e-5 over [5%,100%] | 0.0029 | 0.0028 | 2.76 | 57 |
| sched-const-hi | constant 5e-4 | 0.0036 | 0.0035 | 1.96 | 53 |
| sched-const-lo | constant 5e-5 | 0.0015 | 0.0015 | 6.28 | 103 |

Evidence scheduling works:
1. **Same final coeff (5e-5), ~half the components.** release = 57 comps / L0 2.76 vs
   const-lo = 103 / 6.28. The high-early pressure locks in sparsity that the constant-low
   run never reaches. The *schedule*, not just the endpoint coeff, drives the win.
2. **Pareto-better than interpolating the two constants.** At release's sparsity (L0 2.76,
   near hi), interpolating hi↔lo predicts recon ≈ 0.0032; release achieves 0.0029.
3. **No over-release failure.** Through the full anneal, n_alive stayed flat ~55–60 and
   recon improved monotonically — no subcomponent flood-back. Optimal checkpoint = the
   final one, so releasing all the way to 5e-5 is free.

**Adopted recipe (all remaining objectives):** impmin scheduling — base coeff 5e-5,
`coeff_peak_multiplier` 10 (peak 5e-4), `coeff_warmup_frac` 0.05, anneal
[`start` 0.05, `end` 1.0] → final 5e-5. Plus the prior treatment recipe: nontarget
`impmin_coeff_ratio` 2.0, CI-fn LR 3e-4 (< components LR 1e-3),
`ci_scaled_component_weight_decay` 0.2.

## Objective 2 — Decompose as many layers as possible, then a serious run.

C derived from best layer-18 addition decompositions:
- best MLP (full gate/up/down) = `llama8b-add-refine-treat-01`: **n_alive 114** → **C_mlp = 228**
- best attention (q/k/v/o) = `llama8b-add-attn-01`: **n_alive 58** → **C_attn = 116**

Per layer = 3 MLP matrices (C=228) + 4 attn matrices (C=116).

Parallelism: `dp` is data-parallel replication — every GPU holds the full 16GB target model
+ all components; dp only splits the batch (lowers activation memory). "Fits on 4 GPUs" =
per-GPU < ~45GB with batch/4.

### Memory probe (4 GPUs, `expandable_segments:True`, peak of 46068 MiB; 8 steps + step-0 slow eval)

The OOM is always in `PersistentPGDReconLoss.warmup` — backprop through the full decomposed
stack to optimize the adversarial masks. Cost is dominated by a large **batch-independent**
floor (~44GB already at 6 layers: 16GB model + warmup graph reconstructed-weights + 128K-vocab
KL logits) plus ~2GB/layer. Batch reduction barely helps. Allocation is heterogeneous
(GPUs at 44.3 vs 47.3 GiB) so the serious run must fit the **44 GiB worst case**.

| layers | batch | result | peak MiB |
|---|---|---|---|
| 32 | 96 | ✗ | crash ~44 |
| 32 | 48 | ✗ | 48318 |
| 24 | 96 | ✗ | 41932 |
| 16 | 96 | ✗ | crash |
| 16 | 48 | ✗ | crash |
| 12 | 96 | ✗ | 47022 |
| 12 | 48 | ✗ (44G GPUs) | 46050 |
| 10 | 48 | ✗ | 48060 |
| 8 | 96 | ✗ | 44546 |
| 8 | 48 | ✗ | 48410 |
| **6** | **48** | **✓** | **44326** |

**Max that reliably fits 4 GPUs = 6 layers @ batch 48** (peak 44.3GB, ~1GB margin on the
44GiB GPUs; probe exercised the lifetime-peak ops incl. step-0 slow PGD eval). Serious run:
layers **15–20** (contiguous block centred on 18), MLP C=228 + attn C=116, 20000 steps,
impmin scheduling, batch 48. QOS caps walltime (48h rejected w/ QOSMaxWallDuration), so
run at `--time=12:00:00`; checkpoints every 5000 for resume safety.

### Serious run — job 558 (`llama8b-fullnet-add-6L`)

Launched, stable past the step-0 PGD warmup/eval (the memory bottleneck): **1.55 s/it**,
tqdm ETA ~8.5h for 20000 steps → finishes inside the 12h window, single shot. Layers 15–20,
42 matrices, total C=6888.
