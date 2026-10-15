# Plan: splitting a writer's variance over the readers that feed it, through SwiGLU

*Plan for an exact variance decomposition of every MLP writer of the addsub -08 decomposition
into contributions from its up and gate readers, in the alive-only model. It complements the
mean-ablation ranking of [`report_method.md`](report_method.md), which scores readers one at a
time; this plan gives a decomposition whose pieces add up exactly to the writer's variance, and
which resolves pairs of readers.*

## 1. The question

The model is Llama 3.1 8B, decomposed by the run *addsub -08* (`p-ba8b0c08`, step 40000) on
prompts `a + b =` and `a - b =`, where the operands $a$ and $b$ are integers from 1 to 400. Each
prompt has five token positions (beginning-of-sequence, $a$, the operator, $b$, `=`).

Every weight matrix is written as a sum of rank-one **components**. In an MLP block there are
three kinds:

* **up readers** and **gate readers**, the components of `up_proj` and `gate_proj`. Each reads
  one direction of the normalised residual stream and writes a pattern over the MLP neurons;
* **writers**, the components of `down_proj`. Each reads a pattern over the neurons and writes
  one direction into the residual stream.

Between the readers and the writers sits the SwiGLU nonlinearity. We want to find groups of
readers and writers that pass information to each other. Concretely: for each writer, how much of
its variation over prompts comes from each reader, and from each (up reader, gate reader) pair,
with the nonlinearity accounted for exactly.

## 2. The model we analyse: alive-only

We analyse the **alive-only model** of the -08 CI filter `addsub-08-last-pos`:

* the **alive** components of a site are those whose CI (computed on the original model) exceeds
  0.01 on at least one prompt and position. They are listed per block in
  `<run>/analysis/reader_writer/L<l>/comps.npz`;
* every alive component is on at 100 % on every prompt and position;
* every dead component and the weight delta are off.

So there are no per-prompt masks: every term below is present on every prompt. The equations of
this plan are exact for this model, not for the original Llama.

Alive counts per block, from `comps.npz` (up, gate, writers): block 18 has 171, 149 and 194;
block 31 has the most, with 259, 310 and 572.

## 3. Notation

Everything is for one block $l \in \{0, \dots, 31\}$ and one token position
$\tau \in \{0, \dots, 4\}$, held fixed. The MLP is applied token by token, so nothing at position
$\tau$ depends on another position's MLP.

* $x$: a prompt, from the pool $\mathcal X$ of the $2 \times 400 \times 400 = 320{,}000$ prompts.
  $\mathbb E_x$, $\mathrm{Var}_x$ and $\mathrm{Cov}_x$ are the mean, variance and covariance
  over $x \in \mathcal X$ (or over a random subset of it, where stated).
* $\mathbf z(x) \in \mathbb R^{4096}$: the MLP's input at position $\tau$, i.e. the residual stream
  after RMSNorm and its gain, in the alive-only model.
* $N = 14336$: the number of MLP neurons; $n \in \{1, \dots, N\}$ indexes them.
* $\mathcal U$, $\mathcal G$, $\mathcal D$: the sets of alive up readers, alive gate readers and
  alive writers of the block. $c \in \mathcal U$ is an up reader, $c' \in \mathcal G$ a gate
  reader, $w \in \mathcal D$ a writer. $\lvert \mathcal U \rvert$ is the number of elements of
  $\mathcal U$, and likewise for the others.
* Each up reader $c$ has a read vector $V_{u,c} \in \mathbb R^{4096}$ and a write vector
  $U_{u,c} \in \mathbb R^{N}$, whose entry for neuron $n$ is the real number $U_{u,c,n}$. Gate
  readers have $V_{g,c'}$ and $U_{g,c'}$ in the same spaces.
* Each writer $w$ has a read vector $V_{d,w} \in \mathbb R^{N}$ (entries $V_{d,w,n}$) and a write
  vector $U_{d,w} \in \mathbb R^{4096}$.
* **Component activations** (real numbers, one per prompt):
  $t_{u,c}(x) = V_{u,c} \cdot \mathbf z(x)$ for an up reader,
  $t_{g,c'}(x) = V_{g,c'} \cdot \mathbf z(x)$ for a gate reader, and $t_{d,w}(x)$ for a writer
  (defined in section 4).
* **What a reader writes to a neuron** (a real number per prompt and neuron):

$$
y_{u,c,n}(x) = t_{u,c}(x)\, U_{u,c,n}, \qquad y_{g,c',n}(x) = t_{g,c'}(x)\, U_{g,c',n} .
$$

* **Neuron pre-activations**, the sums over all alive readers of the same kind:

$$
h_{u,n}(x) = \sum_{c \in \mathcal U} y_{u,c,n}(x), \qquad h_{g,n}(x) = \sum_{c' \in \mathcal G} y_{g,c',n}(x) .
$$

* $\sigma(\xi) = 1 / (1 + e^{-\xi})$, the logistic sigmoid of a real number $\xi$, and
  $\mathrm{SiLU}(\xi) = \xi\, \sigma(\xi)$.
* $o_n(x) = \mathrm{SiLU}(h_{g,n}(x))\, h_{u,n}(x)$: the output of neuron $n$.

Two remarks on the notation.

* **Gauge.** The decomposition does not fix the scale of $V$ against $U$: multiplying $V_{u,c}$ by
  a nonzero real number $\lambda$ and dividing $U_{u,c}$ by $\lambda$ gives the same component.
  $t_{u,c}$ changes under this, but $y_{u,c,n}$ does not. Everything below is built from the
  $y$'s, so the reader side is gauge-free. The writer side is handled in section 8.
* **Means.** A bar denotes the mean over prompts, e.g.
  $\bar y_{u,c,n} = \mathbb E_x[y_{u,c,n}(x)] = \bar t_{u,c}\, U_{u,c,n}$. Because $h_{u,n}$ and
  $h_{g,n}$ are linear in the $y$'s, $\bar h_{u,n} = \sum_c \bar y_{u,c,n}$ and
  $\bar h_{g,n} = \sum_{c'} \bar y_{g,c',n}$.

## 4. From a sum over neurons to a sum over readers, with zero as the reference

The writer reads the neurons through its read vector:

$$
t_{d,w}(x) = \sum_{n=1}^{N} V_{d,w,n}\, o_n(x) = \sum_{n=1}^{N} V_{d,w,n}\, \mathrm{SiLU}\big(h_{g,n}(x)\big)\, h_{u,n}(x) . \tag{1}
$$

Drop the argument $x$ from here on; every quantity except $U$, $V$ and the means depends on it.

**Step 1: factor SiLU.** By definition $\mathrm{SiLU}(h_{g,n}) = \sigma(h_{g,n})\, h_{g,n}$. Write
$\gamma_n = \sigma(h_{g,n}) \in (0, 1)$, the neuron's **gate factor**. Then

$$
o_n = \gamma_n\, h_{g,n}\, h_{u,n} .
$$

**Step 2: expand the two pre-activations.** Substitute the sums that define $h_{g,n}$ and
$h_{u,n}$, and multiply out the product of two finite sums (distributivity):

$$
h_{g,n}\, h_{u,n} = \Big( \sum_{c' \in \mathcal G} y_{g,c',n} \Big) \Big( \sum_{c \in \mathcal U} y_{u,c,n} \Big) = \sum_{c \in \mathcal U} \sum_{c' \in \mathcal G} y_{u,c,n}\, y_{g,c',n} .
$$

So $o_n = \sum_{c, c'} \gamma_n\, y_{u,c,n}\, y_{g,c',n}$.

**Step 3: put this into (1) and swap the order of the finite sums:**

$$
t_{d,w} = \sum_{n} V_{d,w,n} \sum_{c, c'} \gamma_n\, y_{u,c,n}\, y_{g,c',n} = \sum_{c \in \mathcal U} \sum_{c' \in \mathcal G} \underbrace{\sum_{n} V_{d,w,n}\, \gamma_n\, y_{u,c,n}\, y_{g,c',n}}_{P_{w,c,c'}} .
$$

**Step 4: pull the activations out.** Since $y_{u,c,n} = t_{u,c} U_{u,c,n}$ and
$y_{g,c',n} = t_{g,c'} U_{g,c',n}$, and $t_{u,c}$, $t_{g,c'}$ do not depend on $n$:

$$
P_{w,c,c'} = t_{u,c}\, t_{g,c'}\, B_{w,c,c'}, \qquad B_{w,c,c'} = \sum_n V_{d,w,n}\, U_{u,c,n}\, U_{g,c',n}\, \gamma_n . \tag{2}
$$

This is a **bilinear** form in the readers' activations. The coupling tensor $B$ would be a fixed
weight tensor if SiLU were replaced by the identity; with SiLU, each neuron's term is weighted by
its gate factor $\gamma_n$, which depends on the prompt. When a neuron's gate is wide open,
$\gamma_n \approx 1$; when it is closed, $\gamma_n \approx 0$ and every path through that neuron
vanishes.

**The only choice made.** Steps 2 to 4 are pure algebra. Step 1 is where a choice enters: it
shares $\mathrm{SiLU}(h_{g,n})$ among the gate readers in proportion to what each writes,
$y_{g,c',n}$. That is the integrated-gradients (Aumann–Shapley) attribution of
$\mathrm{SiLU}(h_{g,n})$ along the straight line from the reference $h_{g,n} = 0$. With a path
parameter $\alpha \in [0, 1]$, gate reader $c'$ receives

$$
y_{g,c',n} \int_0^1 \mathrm{SiLU}'(\alpha\, h_{g,n})\, d\alpha = y_{g,c',n}\, \frac{\mathrm{SiLU}(h_{g,n}) - \mathrm{SiLU}(0)}{h_{g,n}} = y_{g,c',n}\, \sigma(h_{g,n}),
$$

where $\mathrm{SiLU}'$ is the derivative of SiLU. The shares add up exactly to
$\mathrm{SiLU}(h_{g,n})$, a reader that writes nothing to neuron $n$ gets nothing, and each
reader's share is proportional to what it writes. No choice is made on the up side: $o_n$ is
linear in $h_{u,n}$.

## 5. Why the reference must be the mean, not zero

The reference $h_{g,n} = 0$ ("every gate reader off") answers "how much of the writer's *value*
does each path carry". We want the writer's *variation*, and here the zero reference credits the
wrong reader.

**Example.** One neuron, $V_{d,w,1} = 1$. Gate reader 1 writes the constant
$y_{g,1,1} = 5$ on every prompt. Gate reader 2 writes $y_{g,2,1} \in \{-10, 0\}$, each on half the
prompts. On the prompts where reader 2 writes $-10$, $h_{g,1} = -5$ and $\gamma_1 \approx 0.007$,
so the neuron is essentially off. Where it writes $0$, $h_{g,1} = 5$, $\gamma_1 \approx 0.993$, and
the whole output, $\gamma_1 \cdot 5 \cdot h_{u,1}$, is credited by (2) to reader 1, because reader 2
writes nothing on those prompts. The variance of the writer comes from reader 2 switching the
neuron on and off, yet the zero-reference decomposition gives it to the constant reader 1.

The fix is to expand around the mean instead of zero. A reader that never varies then has a
deviation of zero on every prompt, and gets no credit for variance.

## 6. The decomposition, expanded around the mean

Write each reader's contribution as its mean plus a **deviation**:

$$
\delta_{u,c,n} = y_{u,c,n} - \bar y_{u,c,n}, \qquad \delta_{g,c',n} = y_{g,c',n} - \bar y_{g,c',n} .
$$

The pre-activations then split as $h_{u,n} = \bar h_{u,n} + \sum_c \delta_{u,c,n}$ and
$h_{g,n} = \bar h_{g,n} + \sum_{c'} \delta_{g,c',n}$.

**Step 1': the secant of SiLU around the mean.** Define the **secant slope** of neuron $n$:

$$
\rho_n = \frac{\mathrm{SiLU}(h_{g,n}) - \mathrm{SiLU}(\bar h_{g,n})}{h_{g,n} - \bar h_{g,n}} \quad \text{(and } \rho_n = \mathrm{SiLU}'(\bar h_{g,n}) \text{ when } h_{g,n} = \bar h_{g,n}\text{)} .
$$

It is the slope of the chord of SiLU between the reference pre-activation and this prompt's. By
the same argument as in section 4, this is the straight-line (Aumann–Shapley) attribution from the
reference $\bar h_{g,n}$. By construction,

$$
\mathrm{SiLU}(h_{g,n}) = \mathrm{SiLU}(\bar h_{g,n}) + \rho_n \sum_{c'} \delta_{g,c',n} .
$$

**Step 2': multiply by the up pre-activation and expand.** With
$h_{u,n} = \bar h_{u,n} + \sum_c \delta_{u,c,n}$:

$$
o_n = \underbrace{\mathrm{SiLU}(\bar h_{g,n})\, \bar h_{u,n}}_{\text{reference}} + \sum_{c} \mathrm{SiLU}(\bar h_{g,n})\, \delta_{u,c,n} + \sum_{c'} \rho_n\, \bar h_{u,n}\, \delta_{g,c',n} + \sum_{c, c'} \rho_n\, \delta_{u,c,n}\, \delta_{g,c',n} .
$$

**Step 3': read out through the writer and swap sums**, as in section 4:

$$
t_{d,w} = t^{\mathrm{ref}}_{d,w} + \sum_{c \in \mathcal U} M^{u}_{w,c} + \sum_{c' \in \mathcal G} M^{g}_{w,c'} + \sum_{c \in \mathcal U} \sum_{c' \in \mathcal G} I_{w,c,c'} , \tag{3}
$$

with

$$
\begin{aligned}
t^{\mathrm{ref}}_{d,w} &= \sum_n V_{d,w,n}\, \mathrm{SiLU}(\bar h_{g,n})\, \bar h_{u,n} && \text{(a constant)},\\
M^{u}_{w,c} &= \sum_n V_{d,w,n}\, \mathrm{SiLU}(\bar h_{g,n})\, \delta_{u,c,n} = (t_{u,c} - \bar t_{u,c}) \sum_n V_{d,w,n}\, U_{u,c,n}\, \mathrm{SiLU}(\bar h_{g,n}) && \text{(up main effect)},\\
M^{g}_{w,c'} &= \sum_n V_{d,w,n}\, \rho_n\, \bar h_{u,n}\, \delta_{g,c',n} && \text{(gate main effect)},\\
I_{w,c,c'} &= \sum_n V_{d,w,n}\, \rho_n\, \delta_{u,c,n}\, \delta_{g,c',n} && \text{(up × gate interaction)}.
\end{aligned}
$$

What each term means:

* $M^{u}_{w,c}$: up reader $c$ moving with the gates held at their mean. It is the reader's
  centred activation times a **fixed** weight, the virtual weight from up reader $c$ to writer
  $w$ through the mean-gated neurons.
* $M^{g}_{w,c'}$: gate reader $c'$ opening or closing neurons that carry the mean up signal.
* $I_{w,c,c'}$: the part that exists only because up reader $c$ and gate reader $c'$ move
  together: what $c$ writes, passed or blocked by what $c'$ does to the same neurons.

The example of section 5 now comes out right: reader 1 has $\delta_{g,1,1} = 0$ on every prompt,
so it appears in no term. Among the gate readers, everything is credited to reader 2 (through
$M^{g}_{w,2}$ and the $I_{w,c,2}$).

Equation (3) is indexed like a functional ANOVA (main effects plus pairwise interactions, no
higher-order terms), and it needs no independence between the readers. It is not a functional
ANOVA: $\rho_n$ depends on every gate reader that writes to neuron $n$, so $M^{g}_{w,c'}$ and
$I_{w,c,c'}$ are not functions of their own readers alone. Only $M^{u}_{w,c}$ is (it is linear in
$t_{u,c}$).

## 7. From the exact sum to a variance decomposition

Write $K_w$ for the list of all terms of (3) except the constant: $\lvert \mathcal U \rvert$ up
main effects, $\lvert \mathcal G \rvert$ gate main effects and
$\lvert \mathcal U \rvert \lvert \mathcal G \rvert$ interactions. Index them by $k \in K_w$, and
write $Q_k(x)$ for term $k$ on prompt $x$, so that $t_{d,w} = t^{\mathrm{ref}}_{d,w} + \sum_k Q_k$.
Because the variance of $t_{d,w}$ equals its covariance with itself, and a constant does not
change a covariance:

$$
\mathrm{Var}_x(t_{d,w}) = \mathrm{Cov}_x\Big(t_{d,w}, \sum_{k} Q_k\Big) = \sum_{k \in K_w} \mathrm{Cov}_x(Q_k, t_{d,w}) . \tag{4}
$$

Define the **share** of term $k$ as $\phi_{w,k} = \mathrm{Cov}_x(Q_k, t_{d,w}) / \mathrm{Var}_x(t_{d,w})$.
The shares of each writer add up to exactly 1. A share can be negative: that term moves against
the rest and cancels part of them.

**Why this is the right split: it is the Shapley value of the variance.** Treat the terms as
players in a cooperative game. The worth of a subset $S \subseteq K_w$ is
$v(S) = \mathrm{Var}_x\big(\sum_{k \in S} Q_k\big) = \sum_{k, k' \in S} C_{k k'}$, with
$C_{k k'} = \mathrm{Cov}_x(Q_k, Q_{k'})$. Adding term $k$ to a subset $S$ that does not contain it
raises the worth by $C_{kk} + 2 \sum_{k' \in S} C_{k k'}$. The Shapley value averages this over
all orderings of the players. In a uniformly random ordering, each other term $k'$ comes before
$k$ with probability $1/2$, so the average is

$$
C_{kk} + 2 \cdot \tfrac12 \sum_{k' \neq k} C_{kk'} = \sum_{k'} C_{kk'} = \mathrm{Cov}_x\Big(Q_k, \sum_{k'} Q_{k'}\Big) = \mathrm{Cov}_x(Q_k, t_{d,w}) .
$$

So (4) is the unique split of the variance among the terms that adds up exactly, treats
interchangeable terms the same, gives nothing to a term that never varies, and is additive across
games. It handles correlated readers without any independence assumption, which a Sobol
(variance-based sensitivity) decomposition cannot.

**Per-reader totals.** The share of up reader $c$ in writer $w$ is
$\phi^{u}_{w,c} = \phi_{w, M^{u}_{w,c}} + \tfrac12 \sum_{c'} \phi_{w, I_{w,c,c'}}$, and that of gate
reader $c'$ is $\phi^{g}_{w,c'} = \phi_{w, M^{g}_{w,c'}} + \tfrac12 \sum_{c} \phi_{w, I_{w,c,c'}}$.
Each interaction is split equally between its two readers. That is a convention, and it makes
$\sum_c \phi^{u}_{w,c} + \sum_{c'} \phi^{g}_{w,c'} = 1$. For finding groups the interaction is kept
as its own pair edge (section 9), so the convention only affects the per-reader summaries.

## 8. The writer side

Shares are ratios, so they do not depend on the writer's gauge. To compare or merge writers we
use the variance each writer puts into the residual stream: writer $w$ adds $t_{d,w} U_{d,w}$,
whose total variance over prompts is $\mathrm{Var}_x(t_{d,w})\, \lVert U_{d,w} \rVert^2$, where
$\lVert \cdot \rVert$ is the Euclidean norm. This is gauge-free. The **absolute** weight of term
$k$ is $\phi_{w,k}\, \mathrm{Var}_x(t_{d,w})\, \lVert U_{d,w} \rVert^2$, in units of squared
residual-stream norm.

## 9. From shares to groups

For each block and position, build a weighted graph:

* nodes: the alive up readers, gate readers and writers;
* a writer–up edge for each $(w, c)$ and a writer–gate edge for each $(w, c')$, weighted by the
  absolute weight of the main effect (section 8);
* a hyperedge for each triple $(w, c, c')$, weighted by the absolute weight of the interaction
  $I_{w,c,c'}$.

Groups are then a clustering of this weighted structure. Two concrete routes:

* **Pairwise graph.** Fold each interaction into half-weight writer–up and writer–gate edges plus
  an up–gate edge, keep the positive weights, and run community detection (Leiden with a scanned
  resolution) on the result. Negative weights are reported separately as cancellations.
* **Tensor route.** Collect the absolute interaction weights into a tensor indexed by
  $(w, c, c')$, keep its positive part, and fit a nonnegative CP (PARAFAC) decomposition. Each
  rank-one factor is a candidate group: a set of writers, up readers and gate readers with
  weights. The main effects enter as an extra "no partner" slice on each reader axis.

A group is supported when its members carry a large fraction of the members' writers' variance.
The number reported per group is the sum of its shares over its writers.

## 10. What to compute

**Data.** No alive-only extraction exists for -08. The existing
`<run>/analysis/reader_writer/L<l>/inner.npy` is the rounded-CI model's, whose upstream masks
change every reader's input. Required: a forward of the alive-only model (all alive components on
at every site of every block, dead components and delta off) over the 320,000-prompt pool, saving
for every block and position the up, gate and writer activations $t_{u,c}$, $t_{g,c'}$, $t_{d,w}$
(at block 18: $171 + 149 + 194 = 514$ numbers per prompt and position, 1.6 GB in float16 for all
five positions). Also save KL(original ‖ alive-only) at the last position, to document how far the
alive-only model is from Llama.

**Main effects and per-reader totals (whole pool).** Per prompt, the up main effects of all
writers form the matrix $(V_{d,w,n}\, \mathrm{SiLU}(\bar h_{g,n}))_{w,n}$ times
$(\delta_{u,c,n})_{n,c}$, i.e. $2 \lvert \mathcal D \rvert \lvert \mathcal U \rvert N$
floating-point operations; at block 18 that is $9.5 \times 10^{8}$ per prompt, $3.0 \times 10^{14}$
for the pool at one position. The gate main effects cost the same with $\lvert \mathcal G \rvert$.
The interaction summed over gates, $\sum_{c'} I_{w,c,c'} = \sum_n V_{d,w,n}\, \rho_n\, \delta_{u,c,n} \sum_{c'} \delta_{g,c',n}$,
has the same cost, and likewise the sum over up readers. So all per-reader totals of section 7
cost three or four such products.

**Full interaction tensor (subsample).** $I_{w,c,c'}$ for all triples costs
$2 \lvert \mathcal D \rvert \lvert \mathcal U \rvert \lvert \mathcal G \rvert N$ operations per
prompt: $1.4 \times 10^{11}$ at block 18 ($4.9 \times 10^{6}$ triples), $1.3 \times 10^{12}$ at
block 31 ($4.6 \times 10^{7}$ triples). On a random 20,000-prompt subsample that is
$2.8 \times 10^{15}$ and $2.6 \times 10^{16}$ per position. Only the running sums
$\sum_x I_{w,c,c'}\, t_{d,w}$ and $\sum_x I_{w,c,c'}$ are kept (one array of the tensor's size each),
from which $\mathrm{Cov}_x(I_{w,c,c'}, t_{d,w})$ follows. Contract over $n$ as a batched matrix
product, one matrix product per prompt: reshape $(V_{d,w,n}\, \rho_n\, \delta_{u,c,n})$ to
$(\lvert \mathcal D \rvert \lvert \mathcal U \rvert) \times N$ and multiply by
$(\delta_{g,c',n})_{n,c'}$.

**Checks, in order.**

1. The forward: $t_{d,w}$ recomputed from the saved $t_{u,c}$, $t_{g,c'}$ and the alive $U$, $V$
   by equation (1) matches the saved $t_{d,w}$.
2. The identity (3): reference plus all terms equals $t_{d,w}$ on every prompt, to float32
   precision. The case $h_{g,n} \approx \bar h_{g,n}$ uses $\mathrm{SiLU}'(\bar h_{g,n})$ to avoid
   0/0.
3. The shares of each writer add up to 1 (equation 4), on the subsample and on the pool.
4. A synthetic test with the example of section 5: the constant gate reader gets share 0.
5. Comparison with the mean-ablation scores of [`report_method.md`](report_method.md) at block 18,
   position `=`. The two are computed on different models (rounded-CI there, alive-only here), so
   the comparison is of rankings, not values.
