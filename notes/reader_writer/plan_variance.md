# Plan: splitting a writer's variance over the readers that feed it, through SwiGLU

*Plan for an exact variance decomposition of every MLP writer of the addsub -08 decomposition
into contributions from its up and gate readers, in the alive-only model. It complements the
mean-ablation ranking of [`report_method.md`](report_method.md), which scores readers one at a
time; this plan gives a decomposition whose pieces add up exactly to the writer's variance, with one
term per reader, and which resolves which gate readers route each up reader's effect.*

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
y_{u,c,n}(x) = t_{u,c}(x) U_{u,c,n}, \qquad y_{g,c',n}(x) = t_{g,c'}(x) U_{g,c',n} .
$$

* **Neuron pre-activations**, the sums over all alive readers of the same kind:

$$
h_{u,n}(x) = \sum_{c \in \mathcal U} y_{u,c,n}(x), \qquad h_{g,n}(x) = \sum_{c' \in \mathcal G} y_{g,c',n}(x) .
$$

* $\sigma(\xi) = 1 / (1 + e^{-\xi})$, the logistic sigmoid of a real number $\xi$, and
  $\mathrm{SiLU}(\xi) = \xi \sigma(\xi)$.
* $o_n(x) = \mathrm{SiLU}(h_{g,n}(x)) h_{u,n}(x)$: the output of neuron $n$.

Two remarks on the notation.

* **Gauge.** The decomposition does not fix the scale of $V$ against $U$: multiplying $V_{u,c}$ by
  a nonzero real number $\lambda$ and dividing $U_{u,c}$ by $\lambda$ gives the same component.
  $t_{u,c}$ changes under this, but $y_{u,c,n}$ does not. Everything below is built from the
  $y$'s, so the reader side is gauge-free. The writer side is handled in section 8.
* **Means.** A bar denotes the mean over prompts, e.g.
  $\bar y_{u,c,n} = \mathbb E_x[y_{u,c,n}(x)] = \bar t_{u,c} U_{u,c,n}$. Because $h_{u,n}$ and
  $h_{g,n}$ are linear in the $y$'s, $\bar h_{u,n} = \sum_c \bar y_{u,c,n}$ and
  $\bar h_{g,n} = \sum_{c'} \bar y_{g,c',n}$.

## 4. From a sum over neurons to a sum over readers, with zero as the reference

The writer reads the neurons through its read vector:

$$
t_{d,w}(x) = \sum_{n=1}^{N} V_{d,w,n} o_n(x) = \sum_{n=1}^{N} V_{d,w,n} \mathrm{SiLU}\big(h_{g,n}(x)\big) h_{u,n}(x) . \tag{1}
$$

Drop the argument $x$ from here on; every quantity except $U$, $V$ and the means depends on it.

**Step 1: factor SiLU.** By definition $\mathrm{SiLU}(h_{g,n}) = \sigma(h_{g,n}) h_{g,n}$. Write
$\gamma_n = \sigma(h_{g,n}) \in (0, 1)$, the neuron's **gate factor**. Then

$$
o_n = \gamma_n h_{g,n} h_{u,n} .
$$

**Step 2: expand the two pre-activations.** Substitute the sums that define $h_{g,n}$ and
$h_{u,n}$, and multiply out the product of two finite sums (distributivity):

$$
h_{g,n} h_{u,n} = \Big( \sum_{c' \in \mathcal G} y_{g,c',n} \Big) \Big( \sum_{c \in \mathcal U} y_{u,c,n} \Big) = \sum_{c \in \mathcal U} \sum_{c' \in \mathcal G} y_{u,c,n} y_{g,c',n} .
$$

So $o_n = \sum_{c, c'} \gamma_n y_{u,c,n} y_{g,c',n}$.

**Step 3: put this into (1) and swap the order of the finite sums:**

$$
t_{d,w} = \sum_{n} V_{d,w,n} \sum_{c, c'} \gamma_n y_{u,c,n} y_{g,c',n} = \sum_{c \in \mathcal U} \sum_{c' \in \mathcal G} \underbrace{\sum_{n} V_{d,w,n} \gamma_n y_{u,c,n} y_{g,c',n}}_{P_{w,c,c'}} .
$$

**Step 4: pull the activations out.** Since $y_{u,c,n} = t_{u,c} U_{u,c,n}$ and
$y_{g,c',n} = t_{g,c'} U_{g,c',n}$, and $t_{u,c}$, $t_{g,c'}$ do not depend on $n$:

$$
P_{w,c,c'} = t_{u,c} t_{g,c'} B_{w,c,c'}, \qquad B_{w,c,c'} = \sum_n V_{d,w,n} U_{u,c,n} U_{g,c',n} \gamma_n . \tag{2}
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
y_{g,c',n} \int_0^1 \mathrm{SiLU}'(\alpha h_{g,n}) d\alpha = y_{g,c',n} \frac{\mathrm{SiLU}(h_{g,n}) - \mathrm{SiLU}(0)}{h_{g,n}} = y_{g,c',n} \sigma(h_{g,n}),
$$

where $\mathrm{SiLU}'$ is the derivative of SiLU. The shares add up exactly to
$\mathrm{SiLU}(h_{g,n})$, a reader that writes nothing to neuron $n$ gets nothing, and each
reader's share is proportional to what it writes. No choice is made on the up side: $o_n$ is
linear in $h_{u,n}$.

## 5. Choosing the reference: what the decomposition should do

Section 4 expands around the reference "every reader writes 0". Any other reference also gives an
exact sum: subtract a reference value from what each reader writes, expand, and collect the
leftover constant. The reference does not change the identity; it changes which reader is
credited with which part of the writer's variation. We want four properties, stated for a single
neuron $n$ read by the writer with $V_{d,w,n} = 1$:

1. If the gate is always off at a low value, an up reader gets no credit, however much it varies.
2. If the gate is always off at a low value, a gate reader gets no credit even if it varies:
   the variation has no consequence.
3. If the gate is on for some prompts and off for others, and an up reader varies only on the off
   prompts and is constant on the on prompts, the up reader gets no credit: none of its variation
   reaches the neuron's output.
4. If the gate pre-activation is variable and negative on some prompts (gate off) and constant
   and positive on the others (gate on), the gate gets credit. Its jump from negative to positive
   is what opens the gate, so its variation does reach the writer.

"Off" has to mean below about $-5$: SiLU is not flat between $-5$ and $0$ (its minimum is
$-0.278$ at $-1.28$). Below $-5$, SiLU is under $0.034$ in absolute value and its slope under
$0.03$. Properties 1 and 2 then hold up to this small leak, and the credit given equals the small
variance that really gets through.

**The zero reference fails on the gate side.** One neuron, $V_{d,w,1} = 1$. Gate reader 1 writes
the constant $y_{g,1,1} = 5$ on every prompt. Gate reader 2 writes $y_{g,2,1} \in \{-10, 0\}$,
each on half the prompts. On the prompts where reader 2 writes $-10$, $h_{g,1} = -5$ and
$\gamma_1 \approx 0.007$, so the neuron is essentially off. Where it writes $0$, $h_{g,1} = 5$,
$\gamma_1 \approx 0.993$, and the whole output, $\gamma_1 \cdot 5 \cdot h_{u,1}$, is credited by
(2) to reader 1, because reader 2 writes nothing on those prompts. The writer's variance comes
from reader 2 switching the neuron on and off, yet the zero reference gives it to the constant
reader 1. The fix on the gate side is to use the mean over prompts as the reference: a gate reader
that never varies then has a deviation of zero on every prompt.

**The plain mean fails on the up side (property 3).** One neuron, one up reader, gate
pre-activation $h_{g,1}$ either $-10$ (off) or a constant positive value (on). Let
$q \in (0, 1)$ be the fraction of prompts where the gate is on, $A$ the up reader's constant value
of $h_{u,1}$ on those prompts, and $B$ its mean over the off prompts. Its plain mean is
$\bar h_{u,1} = q A + (1 - q) B$. Measured from that mean, the up reader's contribution on the on
prompts is $\mathrm{SiLU}(h_{g,1}) (A - \bar h_{u,1}) = \mathrm{SiLU}(h_{g,1}) (1 - q)(A - B)$,
a constant that switches with the gate; on the off prompts it is about 0. Its share of the
writer's variance comes out as $(1 - q)(A - B) / A$. That depends on $B$, the up value on prompts
where the gate blocks it, which has no effect on the output; the share is 0 only if $B = A$. (The
mean-ablation score of [`report_method.md`](report_method.md) has the same dependence.)

The plain mean fails here because it averages in prompts where the gate blocks the up reader. The
reference for the up side should weight each prompt by how much the gate passes.

## 6. The decomposition with a gate-weighted up reference

**Gate side: the plain mean.** Each gate reader's deviation from its mean is

$$
\delta_{g,c',n} = y_{g,c',n} - \bar y_{g,c',n}, \qquad \text{so} \qquad h_{g,n} = \bar h_{g,n} + \sum_{c'} \delta_{g,c',n} .
$$

The **secant slope** of neuron $n$ is

$$
\rho_n = \frac{\mathrm{SiLU}(h_{g,n}) - \mathrm{SiLU}(\bar h_{g,n})}{h_{g,n} - \bar h_{g,n}} \quad \text{(and } \rho_n = \mathrm{SiLU}'(\bar h_{g,n}) \text{ when } h_{g,n} = \bar h_{g,n}\text{)},
$$

the slope of the chord of SiLU between the reference pre-activation and this prompt's. By
construction

$$
\mathrm{SiLU}(h_{g,n}) = \mathrm{SiLU}(\bar h_{g,n}) + \rho_n \sum_{c'} \delta_{g,c',n} . \tag{3}
$$

Sharing the change of SiLU among the gate readers in proportion to their deviations is the
straight-line (Aumann–Shapley) attribution from the reference $\bar h_{g,n}$, by the same
argument as in section 4.

**Up side: the gate-weighted mean.** Give prompt $x$ the weight
$\omega_n(x) = \mathrm{SiLU}(h_{g,n}(x))^2 \geq 0$, the square of how much neuron $n$'s gate passes
on that prompt. The up reference of reader $c$ on neuron $n$ is its $\omega_n$-weighted mean,

$$
r_{c,n} = \frac{\mathbb E_x[\omega_n\, y_{u,c,n}]}{\mathbb E_x[\omega_n]} = \tilde t_{u,c,n}\, U_{u,c,n}, \qquad \tilde t_{u,c,n} = \frac{\mathbb E_x[\omega_n\, t_{u,c}]}{\mathbb E_x[\omega_n]},
$$

where $\tilde t_{u,c,n}$ is the reader's activation averaged with neuron $n$'s weights (a real
number; one per reader and neuron). The neuron's total up reference is
$r_n = \sum_c r_{c,n}$, which is the same weighted mean of $h_{u,n}$. Each up reader's deviation
from its reference is

$$
\varepsilon_{u,c,n} = y_{u,c,n} - r_{c,n}, \qquad \text{so} \qquad h_{u,n} = r_n + \sum_c \varepsilon_{u,c,n} .
$$

Like $y$, the reference $r_{c,n}$ and the deviation $\varepsilon_{u,c,n}$ are gauge-free.

*Why these weights.* $r_n$ is the constant that best explains the neuron's output from its gate
alone: it minimises the mean squared error
$\mathbb E_x\big[(o_n - \mathrm{SiLU}(h_{g,n})\, r)^2\big] = \mathbb E_x\big[\omega_n (h_{u,n} - r)^2\big]$
over real numbers $r$. Prompts where the gate is closed have $\omega_n \approx 0$ and do not pull
the reference. If $\mathbb E_x[\omega_n] = 0$, then $\mathrm{SiLU}(h_{g,n}) = 0$ on every prompt,
the neuron outputs 0 throughout, and $r_{c,n} = 0$ is set by convention.

**Expanding the neuron's output.** Write $o_n = \mathrm{SiLU}(h_{g,n})\, h_{u,n}$ with
$h_{u,n} = r_n + \sum_c \varepsilon_{u,c,n}$, and expand the first factor with (3) in the term
that multiplies $r_n$ only:

$$
o_n = \underbrace{\mathrm{SiLU}(\bar h_{g,n})\, r_n}_{\text{reference}} + \underbrace{\sum_{c'} \rho_n\, r_n\, \delta_{g,c',n}}_{\text{gate part}} + \underbrace{\sum_{c} \mathrm{SiLU}(h_{g,n})\, \varepsilon_{u,c,n}}_{\text{up part}} .
$$

The gate part is what the gate readers do to the neuron when the up side sits at its reference;
the up part is what the up readers add on top of their reference, passed by the actual gate on
this prompt.

**Reading out through the writer** and swapping the finite sums, as in section 4:

$$
t_{d,w} = t^{\mathrm{ref}}_{d,w} + \sum_{c \in \mathcal U} M^{u}_{w,c} + \sum_{c' \in \mathcal G} M^{g}_{w,c'} , \tag{4}
$$

with

$$
\begin{aligned}
t^{\mathrm{ref}}_{d,w} &= \sum_n V_{d,w,n}\, \mathrm{SiLU}(\bar h_{g,n})\, r_n && \text{(a constant)},\\
M^{u}_{w,c} &= \sum_n V_{d,w,n}\, \mathrm{SiLU}(h_{g,n})\, \varepsilon_{u,c,n} && \text{(up reader } c\text{)},\\
M^{g}_{w,c'} &= \sum_n V_{d,w,n}\, \rho_n\, r_n\, \delta_{g,c',n} && \text{(gate reader } c'\text{)}.
\end{aligned}
$$

**Which gate reader routes an up reader's effect.** For finding groups we also want to know
through which gate readers an up reader's effect passes. Expanding $\mathrm{SiLU}(h_{g,n})$ in
$M^{u}_{w,c}$ with (3) splits it exactly into

$$
M^{u}_{w,c} = M^{u,0}_{w,c} + \sum_{c' \in \mathcal G} J_{w,c,c'}, \qquad M^{u,0}_{w,c} = \sum_n V_{d,w,n}\, \mathrm{SiLU}(\bar h_{g,n})\, \varepsilon_{u,c,n}, \qquad J_{w,c,c'} = \sum_n V_{d,w,n}\, \rho_n\, \varepsilon_{u,c,n}\, \delta_{g,c',n} . \tag{5}
$$

$M^{u,0}_{w,c}$ is the up reader's effect through the mean gate; $J_{w,c,c'}$ is the part of
it that is passed or blocked by gate reader $c'$ moving away from its mean. When a gate is
closed at its mean pre-activation (for instance off on 90 % of prompts and slightly on for the
rest), $\mathrm{SiLU}(\bar h_{g,n}) \approx 0$ and the up reader's whole effect sits in the
$J$ terms. This split is a refinement of $M^{u}_{w,c}$ only: the $J$'s are credited to the up
reader, and the gate readers' own credit stays $M^{g}_{w,c'}$.

**The four properties.**

1. Gate always off at $h_{g,n} = -10$: $\mathrm{SiLU}(-10) \approx -4.5 \times 10^{-4}$ on every
   prompt, so the up reader's term carries $\mathrm{SiLU}(-10)^2$ times its variance, the variance
   that actually leaks through. Holds up to that leak.
2. Gate always off, gate reader varying between $-12$ and $-8$: $\rho_n$ is the slope of a chord
   of SiLU in that range, below $2.5 \times 10^{-3}$ in absolute value, so $M^{g}$ is equally small. Holds up to the
   leak.
3. With $q$, $A$ and $B$ as in section 5: the off prompts have
   $\omega_n = \mathrm{SiLU}(-10)^2 \approx 2 \times 10^{-7}$ against
   $\mathrm{SiLU}(h_{g,n})^2$ on the on prompts, so $r_n \approx A$. The up reader's term is then
   about $\mathrm{SiLU}(h_{g,n}) (A - A) = 0$ on the on prompts and about 0 on the off prompts.
   Holds, and no longer depends on $B$.
4. Gate pre-activation variable and negative, or constant and positive; up constant at $A$: every
   $\varepsilon_{u,c,n} = 0$, so $M^{u} = 0$ and the gate terms carry the whole variance. Holds.

The zero-reference example of section 5 also comes out right: gate reader 1 has
$\delta_{g,1,1} = 0$ on every prompt, so $M^{g}_{w,1} = 0$ and every $J_{w,c,1} = 0$.

**What is not guaranteed.** The mirror image of property 3 — a gate reader that varies only on
prompts where $h_{u,n} = 0$ — can still get credit, because the gate part uses the constant
$r_n$ rather than the prompt's actual $h_{u,n}$. The up part then carries an opposite, cancelling
term. The scheme is asymmetric by construction: the up side is measured as passed by the actual
gate, the gate side against the reference up value.

## 7. From the exact sum to a variance decomposition

Write $K_w$ for the list of the non-constant terms of (4): $\lvert \mathcal U \rvert$ up terms and
$\lvert \mathcal G \rvert$ gate terms. Index them by $k \in K_w$, and write $Q_k(x)$ for term $k$
on prompt $x$, so that $t_{d,w} = t^{\mathrm{ref}}_{d,w} + \sum_k Q_k$. Because the variance of
$t_{d,w}$ equals its covariance with itself, and a constant does not change a covariance:

$$
\mathrm{Var}_x(t_{d,w}) = \mathrm{Cov}_x\Big(t_{d,w}, \sum_{k} Q_k\Big) = \sum_{k \in K_w} \mathrm{Cov}_x(Q_k, t_{d,w}) . \tag{6}
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

So (6) is the unique split of the variance among the terms that adds up exactly, treats
interchangeable terms the same, gives nothing to a term that never varies, and is additive across
games. It handles correlated readers without any independence assumption, which a Sobol
(variance-based sensitivity) decomposition cannot.

**Per-reader shares and the routing refinement.** The share of up reader $c$ in writer $w$ is
$\phi^{u}_{w,c} = \mathrm{Cov}_x(M^{u}_{w,c}, t_{d,w}) / \mathrm{Var}_x(t_{d,w})$, and that of gate
reader $c'$ is $\phi^{g}_{w,c'} = \mathrm{Cov}_x(M^{g}_{w,c'}, t_{d,w}) / \mathrm{Var}_x(t_{d,w})$;
$\sum_c \phi^{u}_{w,c} + \sum_{c'} \phi^{g}_{w,c'} = 1$. No convention is needed to split pair
terms between readers: every term of (4) belongs to one reader. Because a share is linear in its
term, the refinement (5) splits the up share exactly:
$\phi^{u}_{w,c} = \phi^{u,0}_{w,c} + \sum_{c'} \phi^{J}_{w,c,c'}$, with
$\phi^{u,0}_{w,c}$ and $\phi^{J}_{w,c,c'}$ the shares of $M^{u,0}_{w,c}$ and $J_{w,c,c'}$. The
Shapley values at the coarse level (one term per reader) and at the fine level (the terms of (5))
are therefore consistent.

## 8. The writer side

Shares are ratios, so they do not depend on the writer's gauge. To compare or merge writers we
use the variance each writer puts into the residual stream: writer $w$ adds $t_{d,w} U_{d,w}$,
whose total variance over prompts is $\mathrm{Var}_x(t_{d,w}) \lVert U_{d,w} \rVert^2$, where
$\lVert \cdot \rVert$ is the Euclidean norm. This is gauge-free. The **absolute** weight of term
$k$ is $\phi_{w,k} \mathrm{Var}_x(t_{d,w}) \lVert U_{d,w} \rVert^2$, in units of squared
residual-stream norm.

## 9. From shares to groups

For each block and position, build a weighted structure:

* nodes: the alive up readers, gate readers and writers;
* a writer–up edge for each $(w, c)$, weighted by the absolute weight of $M^{u}_{w,c}$, and a
  writer–gate edge for each $(w, c')$, weighted by the absolute weight of $M^{g}_{w,c'}$
  (section 8);
* a hyperedge for each triple $(w, c, c')$, weighted by the absolute weight of $J_{w,c,c'}$: the
  part of up reader $c$'s effect on writer $w$ that gate reader $c'$ passes or blocks.

The hyperedge weights are part of the writer–up weights (equation 5), not added to them. Groups
are then a clustering of this structure. Two concrete routes:

* **Pairwise graph.** Keep the writer–up and writer–gate edges, add an up–gate edge for each
  $(c, c')$ weighted by $\sum_w$ (absolute weight of $J_{w,c,c'}$), keep the positive weights, and
  run community detection (Leiden with a scanned resolution). Negative weights are reported
  separately as cancellations.
* **Tensor route.** Collect the positive parts of the absolute weights of $J_{w,c,c'}$ into a
  tensor indexed by $(w, c, c')$, with one extra "mean gate" slice on the gate axis holding
  $M^{u,0}_{w,c}$, and fit a nonnegative CP (PARAFAC) decomposition. Each rank-one factor is a
  candidate group: a set of writers, up readers and gate readers with weights. The gate terms
  $M^{g}_{w,c'}$ enter as an extra "reference up" slice on the up axis.

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

**Two passes per block and position.** The references need statistics over prompts before any
term can be formed.

* *Pass 1 (references).* Accumulate $\bar t_{g,c'}$ (giving $\bar h_{g,n}$), and for every neuron
  $n$ the sums $\sum_x \omega_n$ and $\sum_x \omega_n t_{u,c}$ for all up readers (giving
  $\tilde t_{u,c,n}$ and $r_{c,n}$). $\omega_n$ needs $h_{g,n}$, computed on the fly from the
  $t_{g,c'}$. The second sum is a $\lvert \mathcal U \rvert \times N$ array ($2.5 \times 10^{6}$
  entries at block 18), at $\lvert \mathcal U \rvert N$ operations per prompt.
* *Pass 2 (terms and covariances).* Form the terms and accumulate $\sum_x Q_k t_{d,w}$ and
  $\sum_x Q_k$ for every term, from which $\mathrm{Cov}_x(Q_k, t_{d,w})$ follows. The terms are
  never stored.

**Per-reader terms (whole pool).** Per prompt, the up terms of all writers are the matrix
$(V_{d,w,n} \mathrm{SiLU}(h_{g,n}))_{w,n}$ times $(\varepsilon_{u,c,n})_{n,c}$, i.e.
$2 \lvert \mathcal D \rvert \lvert \mathcal U \rvert N$ floating-point operations; at block 18
that is $9.5 \times 10^{8}$ per prompt, $3.0 \times 10^{14}$ for the pool at one position. The
gate terms are $(V_{d,w,n} \rho_n r_n)_{w,n}$ times $(\delta_{g,c',n})_{n,c'}$, the same cost with
$\lvert \mathcal G \rvert$. The mean-gate parts $M^{u,0}_{w,c}$ cost the same as the up terms.

**Routing tensor (subsample).** $J_{w,c,c'}$ for all triples costs
$2 \lvert \mathcal D \rvert \lvert \mathcal U \rvert \lvert \mathcal G \rvert N$ operations per
prompt: $1.4 \times 10^{11}$ at block 18 ($4.9 \times 10^{6}$ triples), $1.3 \times 10^{12}$ at
block 31 ($4.6 \times 10^{7}$ triples). On a random 20,000-prompt subsample that is
$2.8 \times 10^{15}$ and $2.6 \times 10^{16}$ per position. Only the running sums
$\sum_x J_{w,c,c'} t_{d,w}$ and $\sum_x J_{w,c,c'}$ are kept (one array of the tensor's size each).
Contract over $n$ as a batched matrix product, one matrix product per prompt: reshape
$(V_{d,w,n} \rho_n \varepsilon_{u,c,n})$ to $(\lvert \mathcal D \rvert \lvert \mathcal U \rvert)
\times N$ and multiply by $(\delta_{g,c',n})_{n,c'}$. The references stay those of pass 1 on the
whole pool.

**Checks, in order.**

1. The forward: $t_{d,w}$ recomputed from the saved $t_{u,c}$, $t_{g,c'}$ and the alive $U$, $V$
   by equation (1) matches the saved $t_{d,w}$.
2. The identities (4) and (5): reference plus all terms equals $t_{d,w}$ on every prompt, and
   $M^{u,0}_{w,c} + \sum_{c'} J_{w,c,c'} = M^{u}_{w,c}$, to float32 precision. The case
   $h_{g,n} \approx \bar h_{g,n}$ uses $\mathrm{SiLU}'(\bar h_{g,n})$ to avoid 0/0.
3. The shares of each writer add up to 1 (equation 6), on the subsample and on the pool.
4. Synthetic one-neuron tests of the four properties of section 5 and of the zero-reference
   example: the up reader of property 3 and the constant gate reader get shares below the leak
   bound, the gate of property 4 gets share 1.
5. Comparison with the mean-ablation scores of [`report_method.md`](report_method.md) at block 18,
   position `=`. The two are computed on different models (rounded-CI there, alive-only here), so
   the comparison is of rankings, not values.
