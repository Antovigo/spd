# Which readers drive each MLP writer? First results of the exact variance split (addsub -08)

*Short report on the run of [`plan_variance.md`](plan_variance.md). Block 31 at position `=` was
still running when this was written; every other (block, position) pair is included.*

## What was computed

The model is Llama 3.1 8B decomposed by the run *addsub -08* (`p-ba8b0c08`, step 40000), on the
320,000 prompts `a + b =` and `a - b =`, where the operands $a$ and $b$ are integers from 1 to
400. In each MLP block, the **readers** are the components of `up_proj` and `gate_proj` (they read
the residual stream and write to the 14,336 neurons) and the **writers** are the components of
`down_proj` (they read the neurons and write to the residual stream).

We analyse the **alive-only model**: every component of the CI filter `addsub-08-last-pos`'s alive
set is on at 100 % on every prompt, everything else (dead components, weight delta) is off. Its
mean KL divergence from the original model at the last position, over the whole pool, is 0.134
nats.

For every writer $w$ of every block and every token position $\tau \in \{1, 2, 3, 4\}$ (the tokens
$a$, operator, $b$, `=`), the writer's activation $t_{d,w}$ is split exactly into one term per up
reader and one per gate reader (plan §6), and its variance over the prompts into one **share** per
term, the term's covariance with $t_{d,w}$ divided by $\mathrm{Var}(t_{d,w})$ (plan §7; shares sum
to 1 and can be negative). The **weight** $a$ of a term is its share times the variance the writer
puts into the residual stream, $\mathrm{Var}(t_{d,w}) \lVert U_{d,w} \rVert^2$, where $U_{d,w}$ is
the writer's write vector (plan §8). Up readers' terms are further split by which gate reader
routes them (plan §6, the $J$ terms), on a random subsample of 8,000 prompts.

All the checks of plan §10 passed on all 127 completed (block, position) pairs: the writer
activation recomputed from the readers matches the model's within $3 \times 10^{-3}$ of the bf16
rounding scale, both identities hold to float32 rounding, and shares sum to 1 within $10^{-4}$.
None of the 22,936 (writer, position) pairs was excluded for zero variance; 780 are flagged as
carrying less than 1 % of their block's median writer variance.

Component names below are AB-grid ids, `layers.<l>.mlp.<kind>_proj:<id>`.

## Results

Medians over the blocks of each range, per position. "Top-5" is the fraction of the block's total
writer variance (in the residual stream) carried by its 5 largest writers. "Up" is the fraction of
that variance credited to up readers (the rest goes to gate readers). "Readers for 90 %" is, per
writer, the number of readers whose $\lvert a \rvert$ make up 90 % of the writer's total
$\lvert a \rvert$, then the median over writers weighted by writer variance. "Routed" is the fraction
of the up readers' weight that sits in the $J$ terms, i.e. is passed or blocked by individual gate
readers moving, rather than going through the mean gate.

| position | blocks | top-5 | up | readers for 90 % | routed |
|---|---|---|---|---|---|
| $a$ | 0–11 | 0.61 | 0.48 | 5 | 0.13 |
| $a$ | 12–23 | 0.38 | 0.42 | 8 | 0.26 |
| $a$ | 24–31 | 0.59 | 0.38 | 8 | 0.05 |
| operator | 0–11 | 0.80 | 0.38 | 5 | −0.03 |
| operator | 12–23 | 0.78 | 0.36 | 5 | 0.00 |
| operator | 24–31 | 0.70 | 0.34 | 7 | 0.01 |
| $b$ | 0–11 | 0.56 | 0.52 | 8 | 0.20 |
| $b$ | 12–23 | 0.56 | 0.39 | 8 | 0.13 |
| $b$ | 24–31 | 0.63 | 0.39 | 8 | 0.09 |
| `=` | 0–11 | 1.00 | 0.52 | 3 | −0.09 |
| `=` | 12–23 | 0.42 | 0.60 | 13 | 0.22 |
| `=` | 24–30 | 0.27 | 0.52 | 35 | 0.57 |

1. **The large writers are sparse.** Weighted by variance, a writer's variance is 90 % accounted
   for by 1 to 28 readers (medians 3 to 13 in the table) in every pair except `=` in blocks 24–30,
   out of 69 to 716 readers in the block. At `=` in blocks 24–30 it takes 25 to 91 readers
   (block 29: 91 of 696). Unweighted medians are larger (8 to 196): the many small writers draw
   on many readers.
2. **Cancellation is small.** Negative weights make up 3 % to 16 % of the total $\lvert a \rvert$ in
   every (block, position) pair.
3. **Up and gate readers both matter.** The up readers' fraction ranges from 0.05 (block 0, `=`) to
   0.99 (block 2, `=`), with medians between 0.34 and 0.60.
4. **Gating by individual gate readers matters mostly at `=` in late blocks.** There, 47 % to 73 % of
   the up readers' weight is routed by specific gate readers (blocks 23–30), against medians of
   −0.09 to 0.26 elsewhere. A negative routed fraction (16 of the 32 operator-position blocks, and 11 of
   the 12 `=` pairs in blocks 0–11) means the gate readers' movements reduce the covariance that the up readers would
   carry through the mean gate.
5. **The operator position is dominated by a few writers** (median top-5 fraction 0.70 to 0.80), and at
   `=` in blocks 0–11 by one: the `=` token is the same on every prompt, so the MLP input varies
   only through attention there.
6. **A writer's main reader often has the writer's own id.** Among the 3 largest writers of every
   (block, position) pair (381 writers), 172 have a reader with the same id among their 4 largest
   readers. The run was initialized with `neuron_aligned_zero_u`.

Examples (the two largest writers of a few pairs; numbers are shares of the writer's variance):

| block, position | writer | fraction of block variance | largest readers (share) |
|---|---|---|---|
| 0, `=` | down_proj:92 | 0.95 | gate_proj:170 (0.94) |
| 2, `=` | down_proj:6 | 0.85 | up_proj:401 (0.90), up_proj:5 (0.10) |
| 15, `=` | down_proj:1 | 0.14 | gate_proj:11 (0.41), up_proj:28 (0.37), up_proj:22 (0.27) |
| 16, `=` | down_proj:83 | 0.20 | up_proj:25 (0.45), gate_proj:46 (0.37), gate_proj:36 (0.14) |
| 18, $b$ | down_proj:1 | 0.39 | gate_proj:1 (0.78), up_proj:1 (0.19) |
| 18, `=` | down_proj:34 | 0.08 | up_proj:6 (0.43), up_proj:177 (0.25), up_proj:3 (0.14) |
| 19, `=` | down_proj:3 | 0.11 | up_proj:3 (0.57), up_proj:326 (0.17), up_proj:647 (0.09) |
| 24, `=` | down_proj:2 | 0.09 | gate_proj:2 (0.35), gate_proj:11 (0.12), gate_proj:171 (0.09) |
| 30, `=` | down_proj:50 | 0.03 | gate_proj:909 (0.47), gate_proj:357 (0.16), gate_proj:39 (0.10) |
| 31, operator | down_proj:8 | 0.75 | gate_proj:8 (0.68), gate_proj:69 (0.09), gate_proj:127 (0.06) |

The grouping of plan §9 (clustering the weights into reader–writer groups) is not part of this
report.

## Files

* Results: `<run>/analysis/reader_writer/variance/results/L<l>/{variance,J}_pos<tau>.npz`, indexed
  by column of `alive_only/L<l>/comps.npz` (`variance/ids.py` maps columns to AB-grid ids);
  the table above is `results/summary.json`, written by `variance/summarize.py`.
* Activations of the alive-only model: `<run>/analysis/reader_writer/variance/alive_only/`.
* Code: `notes/reader_writer/variance/` on branch `experiment/arith_representations`.
