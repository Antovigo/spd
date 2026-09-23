# Appendix B — every attention head with an alive o component

[← back to the report](../report_auto_interp.md)

**How to read the component tables.** `component` = layer, site kind, component index
(o/q components: the head holding most of their weight; k/v: the kv head). `on` = fraction of
the operation's 10,000 prompts on which the component's CI exceeds 0.01 at that position.
`on-set` = the coarsest function of (a, b) visible at the position that explains the on/off
pattern (adjusted R² within 90 % of the best function in a fixed list; `always` = on for > 95 %
of prompts, `unexplained` = no listed function reaches R² 0.5), then the on classes spelled out
completely (`res mod 100 in {1..9, 75}` = on when the result mod 100 is 1-9 or 75; `(tens) res in
{20..29}`; for digit-pair labels, the cells of the units- or tens-digit plane that are on, rows
with the same cells merged). `[coarser: …]` = a coarser period that already explains ≥ 80 % of
the on-rate profile along that quantity. `res` = a+b on addition, a−b on subtraction.
`writes` (o/down only) = the share of each Fourier code of a, b or res that the component's
write supplies, measured at the first read point after it (its contribution's projection on
that code over the code's squared norm, power-weighted over the harmonics of each period; codes
carrying < 0.1 % of the read input's variance and shares below 2 % are omitted; negative = the
component opposes the code). `(reads)` = a q/k/v/gate/up component (it writes into a head or
the MLP hidden layer, not the residual).

<details><summary>L0H0 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.99 / 0.99 | 0.01 / 0.01 |  |  |  |
| op | 0.96 / 0.96 | 0.02 / 0.02 | 0.02 / 0.01 |  |  |
| b | 0.85 / 0.86 | 0.09 / 0.09 | 0.05 / 0.03 | 0.01 / 0.01 |  |
| = | 0.56 / 0.61 | 0.11 / 0.12 | 0.24 / 0.19 | 0.03 / 0.03 | 0.05 / 0.05 |

<details><summary>1 o components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L0 o c7 (H0) | 100% / 100% | always | same | - | same |

</details>

<details><summary>5 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L0 v c0 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L0 v c3 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L0 v c4 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L0 v c5 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L0 v c6 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>2 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L0 v c1 (kv0) | 24% / 24% | **a** (R2 1.00): a in {1..24} | same |
| L0 v c14 (kv0) | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same |

</details>

<details><summary>1 q components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L0 q c17 (H0) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L0H1 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.89 / 0.89 | 0.11 / 0.11 |  |  |  |
| op | 0.71 / 0.23 | 0.17 / 0.63 | 0.13 / 0.14 |  |  |
| b | 0.05 / 0.15 | 0.01 / 0.05 | 0.89 / 0.67 | 0.04 / 0.14 |  |
| = | 0.51 / 0.51 | 0.01 / 0.01 | 0.03 / 0.02 | 0.20 / 0.20 | 0.25 / 0.25 |

<details><summary>1 o components, main position `&lt;BOS&gt;`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L0 o c26 (H1) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same | - | same |

</details>

<details><summary>5 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L0 v c0 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L0 v c3 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L0 v c4 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L0 v c5 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L0 v c6 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>2 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L0 v c1 (kv0) | 24% / 24% | **a** (R2 1.00): a in {1..24} | same |
| L0 v c14 (kv0) | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same |

</details>

<details><summary>1 q components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L0 q c0 (H1) | 100% / 0 | always | off (on 0) |

</details>

</details>

<details><summary>L0H2 (3 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.88 / 0.88 | 0.12 / 0.12 |  |  |  |
| op | 0.22 / 0.19 | 0.65 / 0.68 | 0.12 / 0.13 |  |  |
| b | 0.06 / 0.08 | 0.48 / 0.67 | 0.39 / 0.15 | 0.08 / 0.11 |  |
| = | 0.02 / 0.02 | 0.01 / 0.01 | 0.20 / 0.19 | 0.53 / 0.54 | 0.25 / 0.25 |

<details><summary>1 o components, main position `&lt;BOS&gt;`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L0 o c5 (H2) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same | - | same |

</details>

<details><summary>2 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L0 o c1 (H2) | 100% / 100% | always | same | a: mod100 +7%, mod50 +8%, mod25 +5%, mod20 +7% | a: mod100 +12%, mod50 +14%, mod25 +10%, mod20 +12% |
| L0 o c2 (H2) | 100% / 100% | always | same | - | a: mod100 +2%, mod50 +2%, mod20 +2% |

</details>

<details><summary>5 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L0 v c0 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L0 v c3 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L0 v c4 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L0 v c5 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L0 v c6 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>2 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L0 v c1 (kv0) | 24% / 24% | **a** (R2 1.00): a in {1..24} | same |
| L0 v c14 (kv0) | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same |

</details>

<details><summary>1 q components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L0 q c21 (H2) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L0H10 (2 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.77 / 0.77 | 0.23 / 0.23 |  |  |  |
| op | 0.78 / 0.75 | 0.09 / 0.07 | 0.13 / 0.18 |  |  |
| b | 0.08 / 0.04 | 0.01 / 0.01 | 0.88 / 0.94 | 0.03 / 0.01 |  |
| = | 0.59 / 0.58 | 0.04 / 0.04 | 0.08 / 0.08 | 0.07 / 0.07 | 0.23 / 0.22 |

<details><summary>2 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L0 o c0 (H10) | 100% / 100% | always | same | - | same |
| L0 o c32 (H10) | 100% / 0 | always | off (on 0) | - | same |

</details>

<details><summary>2 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L0 v c2 (kv2) | 0 / 100% | off (on 0) | always |
| L0 v c23 (kv2) | 100% / 100% | always | same |

</details>

<details><summary>1 v components, main position `=` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L0 v c34 (kv2) | 0 / 100% | off (on 0) | always |

</details>

<details><summary>4 k components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L0 k c75 (kv2) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L0 k c79 (kv2) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L0 k c88 (kv2) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L0 k c108 (kv2) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L0 k c6 (kv2) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L0H23 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.84 / 0.84 | 0.16 / 0.16 |  |  |  |
| op | 0.45 / 0.40 | 0.07 / 0.07 | 0.48 / 0.53 |  |  |
| b | 0.31 / 0.27 | 0.06 / 0.05 | 0.57 / 0.62 | 0.06 / 0.06 |  |
| = | 0.46 / 0.38 | 0.06 / 0.05 | 0.32 / 0.43 | 0.06 / 0.05 | 0.11 / 0.09 |

<details><summary>1 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L0 o c6 (H23) | 0 / 100% | off (on 0) | always | - | same |

</details>

</details>

<details><summary>L0H24 (2 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.92 / 0.92 | 0.08 / 0.08 |  |  |  |
| op | 0.63 / 0.68 | 0.08 / 0.05 | 0.29 / 0.27 |  |  |
| b | 0.62 / 0.67 | 0.10 / 0.11 | 0.22 / 0.17 | 0.06 / 0.06 |  |
| = | 0.48 / 0.47 | 0.00 / 0.00 | 0.14 / 0.15 | 0.03 / 0.03 | 0.34 / 0.34 |

<details><summary>2 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L0 o c30 (H24) | 5% / 0 | **a** (R2 1.00): a in {18, 50, 55, 65, 90} | off (on 0) | - | same |
| L0 o c268 (H24) | 8% / 0 | **a** (R2 1.00): a in {18, 45, 50, 55, 57, 60, 65, 90} | off (on 0) | a: mod5 +2%, mod4 +2% | - |

</details>

<details><summary>3 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L0 k c25 (kv6) | 6% / 0 | **a** (R2 1.00): a in {18, 50, 55, 65, 90, 94} | off (on 0) |
| L0 k c43 (kv6) | 5% / 0 | **a** (R2 1.00): a in {18, 50, 55, 90, 94} | off (on 0) |
| L0 k c115 (kv6) | 1% / 0 | **a** (R2 1.00): a in {90} | off (on 0) |

</details>

</details>

<details><summary>L1H5 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.96 / 0.96 | 0.04 / 0.04 |  |  |  |
| op | 0.98 / 0.97 | 0.01 / 0.02 | 0.01 / 0.01 |  |  |
| b | 0.43 / 0.56 | 0.02 / 0.06 | 0.53 / 0.36 | 0.01 / 0.01 |  |
| = | 0.96 / 0.95 | 0.00 / 0.00 | 0.01 / 0.01 | 0.01 / 0.00 | 0.02 / 0.03 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L1 o c491 (H5) | 100% / 100% | always | same | a: mod100 +12%, mod50 +5% | a: mod100 +12%, mod50 +4%; res: mod100 +13% |

</details>

<details><summary>1 q components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L1 q c15 (H5) | 100% / 100% | always | same |

</details>

<details><summary>1 q components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L1 q c140 (H5) | 9% / 3% | **b** (R2 0.79): b in {1..7} | unexplained (best R2 0.50) |

</details>

<details><summary>1 q components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L1 q c66 (H5) | 100% / 100% | always | same |

</details>

<details><summary>2 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L1 k c0 (kv1) | 100% / 100% | always | same |
| L1 k c1 (kv1) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L1H6 (2 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.96 / 0.96 | 0.04 / 0.04 |  |  |  |
| op | 0.92 / 0.94 | 0.03 / 0.05 | 0.05 / 0.01 |  |  |
| b | 0.44 / 0.81 | 0.05 / 0.12 | 0.48 / 0.07 | 0.03 / 0.00 |  |
| = | 0.35 / 0.73 | 0.01 / 0.04 | 0.60 / 0.17 | 0.00 / 0.00 | 0.03 / 0.07 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L1 o c47 (H6) | 100% / 0 | always | off (on 0) | - | same |

</details>

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L1 o c370 (H6) | 100% / 0 | always | off (on 0) | a: mod20 +2% | - |

</details>

<details><summary>1 q components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L1 q c2 (H6) | 100% / 100% | always | same |

</details>

<details><summary>2 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L1 k c0 (kv1) | 100% / 100% | always | same |
| L1 k c1 (kv1) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L1H11 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.94 / 0.94 | 0.06 / 0.06 |  |  |  |
| op | 0.83 / 0.80 | 0.13 / 0.18 | 0.04 / 0.02 |  |  |
| b | 0.78 / 0.79 | 0.07 / 0.12 | 0.12 / 0.08 | 0.03 / 0.02 |  |
| = | 0.35 / 0.42 | 0.11 / 0.17 | 0.35 / 0.25 | 0.15 / 0.12 | 0.04 / 0.05 |

<details><summary>1 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L1 o c10 (H11) | 100% / 100% | always | same | - | same |

</details>

<details><summary>2 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L1 v c4 (kv2) | 100% / 100% | always | same |
| L1 v c75 (kv2) | 89% / 89% | **a//10** (R2 1.00): (tens) a in {1..89} | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L1 v c17 (kv2) | 0 / 100% | off (on 0) | always |

</details>

</details>

<details><summary>L1H18 (3 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.96 / 0.96 | 0.04 / 0.04 |  |  |  |
| op | 0.67 / 0.94 | 0.29 / 0.04 | 0.03 / 0.01 |  |  |
| b | 0.66 / 0.74 | 0.20 / 0.20 | 0.14 / 0.06 | 0.00 / 0.00 |  |
| = | 0.79 / 0.78 | 0.00 / 0.00 | 0.14 / 0.16 | 0.05 / 0.04 | 0.02 / 0.02 |

<details><summary>3 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L1 o c5 (H18) | 1% / 0 | **a** (R2 1.00): a in {90} | off (on 0) | - | same |
| L1 o c359 (H18) | 100% / 84% | always | **a** (R2 1.00): a in {17..100} | a: mod100 +11%, mod50 +6%, mod25 +5%, mod20 +5% | a: mod100 +7%, mod50 +4% |
| L1 o c416 (H18) | 15% / 11% | **a** (R2 0.99): a in {1..5, 11, 21, 31, 41, 51, 61, 71, 81, 91, 100} [coarser: a mod 50 in {1..2, 11, 21, 31, 41}, R2 0.81] | **a** (R2 0.99): a in {1..2, 21, 31, 41, 51, 61, 71, 81, 91, 100} [coarser: a mod 50 in {1, 21, 31, 41}, R2 0.85] | a: mod10 +7%, mod5 +3%, mod2 +3% | a: mod10 +3%, mod5 +2% |

</details>

<details><summary>1 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L1 v c230 (kv4) | 20% / 20% | **a%5** (R2 1.00): a mod 5 in {0} | same |

</details>

<details><summary>2 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L1 k c10 (kv4) | 100% / 100% | always | same |
| L1 k c71 (kv4) | 100% / 100% | always | same |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L1 k c12 (kv4) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L1H20 (3 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.95 / 0.95 | 0.05 / 0.05 |  |  |  |
| op | 0.78 / 0.66 | 0.19 / 0.32 | 0.03 / 0.02 |  |  |
| b | 0.62 / 0.70 | 0.02 / 0.06 | 0.34 / 0.22 | 0.02 / 0.02 |  |
| = | 0.74 / 0.75 | 0.00 / 0.00 | 0.02 / 0.02 | 0.21 / 0.20 | 0.03 / 0.03 |

<details><summary>3 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L1 o c4 (H20) | 0 / 1% | off (on 0) | unexplained (best R2 0.03) | - | same |
| L1 o c112 (H20) | 0 / 10% | off (on 0) | **a** (R2 0.90): a in {69..77} | - | same |
| L1 o c510 (H20) | 0 / 100% | off (on 0) | always | - | a: mod100 +2% |

</details>

<details><summary>1 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L1 v c1 (kv5) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>20 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L1 v c21 (kv5) | 11% / 13% | **a//10** (R2 0.89): (tens) a in {49..59} | **a** (R2 0.99): a in {47..59} |
| L1 v c28 (kv5) | 16% / 16% | **a** (R2 0.98): a in {7, 66..80} | **a** (R2 0.97): a in {7, 66..80} |
| L1 v c43 (kv5) | 11% / 12% | **a%10** (R2 0.89): a mod 10 in {5} | **a%10** (R2 0.86): a mod 10 in {5} |
| L1 v c50 (kv5) | 4% / 4% | **a** (R2 0.83): a in {37..39} | **a** (R2 0.82): a in {37..39} |
| L1 v c56 (kv5) | 4% / 3% | **a** (R2 0.87): a in {35, 37..38} | **a** (R2 0.90): a in {35, 37..38} |
| L1 v c60 (kv5) | 5% / 5% | **a** (R2 0.98): a in {5, 15, 25, 65, 75} [coarser: a mod 50 in {5, 15, 25}, R2 0.90] | **a** (R2 0.94): a in {5, 15, 25, 65, 75} [coarser: a mod 50 in {5, 15, 25}, R2 0.90] |
| L1 v c61 (kv5) | 6% / 6% | **a** (R2 0.86): a in {7..10, 98} | **a** (R2 0.82): a in {7..10, 98} |
| L1 v c64 (kv5) | 8% / 7% | **a** (R2 0.99): a in {86, 88..94} | **a** (R2 0.95): a in {86, 89..94} |
| L1 v c65 (kv5) | 6% / 6% | **a** (R2 0.94): a in {7, 17, 27, 37, 87, 97} | **a** (R2 0.96): a in {7, 17, 27, 37, 87, 97} |
| L1 v c78 (kv5) | 17% / 14% | **a** (R2 0.96): a in {1..4, 21, 31..32, 41, 51, 61..62, 71..72, 81, 91..92, 98} | **a** (R2 0.88): a in {1..4, 21, 31..32, 41, 51, 61..62, 81, 91..92} |
| L1 v c82 (kv5) | 14% / 16% | **a** (R2 0.98): a in {76..89} | **a** (R2 0.98): a in {8, 75..89} |
| L1 v c94 (kv5) | 11% / 7% | **a** (R2 0.86): a in {6, 13, 16, 23, 43, 53, 56, 66, 73, 93, 96} [coarser: a mod 50 in {6, 13, 16, 23, 43, 46}, R2 0.82] | **a** (R2 0.67): a in {6, 13, 16, 23, 43, 53, 56} |
| L1 v c116 (kv5) | 8% / 11% | **a%50** (R2 0.84): a mod 50 in {1, 11, 21, 31, 41} [coarser: a mod 10 in {1}, R2 0.85] | **a%50** (R2 0.85): a mod 50 in {1, 9, 11, 21, 31, 41} [coarser: a mod 10 in {1}, R2 0.88] |
| L1 v c117 (kv5) | 8% / 7% | **a** (R2 0.95): a in {2, 12, 32, 52, 62, 72, 82, 92} [coarser: a mod 20 in {2, 12}, R2 0.81] | **a%50** (R2 0.88): a mod 50 in {2, 12, 32} |
| L1 v c118 (kv5) | 9% / 9% | **a** (R2 0.98): a in {9, 19, 29, 39, 49, 79, 89, 98..99} [coarser: a mod 50 in {9, 29, 39, 49}, R2 0.82] | **a** (R2 0.99): a in {9, 19, 29, 39, 49, 79, 89, 98..99} [coarser: a mod 50 in {9, 29, 39, 49}, R2 0.82] |
| L1 v c146 (kv5) | 5% / 5% | **a** (R2 0.99): a in {45..49} | **a** (R2 1.00): a in {45..49} |
| L1 v c154 (kv5) | 10% / 9% | **a%10** (R2 0.90): a mod 10 in {4} | **a%10** (R2 0.82): a mod 10 in {4} |
| L1 v c157 (kv5) | 14% / 16% | **a** (R2 0.94): a in {6, 56, 60..69, 86} | **a** (R2 0.89): a in {6, 56, 59..69, 86} |
| L1 v c215 (kv5) | 8% / 8% | **a** (R2 0.99): a in {57..64} | **a** (R2 1.00): a in {57..64} |
| L1 v c228 (kv5) | 4% / 4% | **a** (R2 1.00): a in {97..100} | same |

</details>

<details><summary>2 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L1 v c8 (kv5) | 100% / 0 | always | off (on 0) |
| L1 v c15 (kv5) | 100% / 0 | always | off (on 0) |

</details>

<details><summary>4 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L1 k c74 (kv5) | 0 / 0 | off (on 0) | same |
| L1 k c119 (kv5) | 0 / 0 | off (on 0) | same |
| L1 k c139 (kv5) | 1% / 1% | unexplained (best R2 0.03) | same |
| L1 k c142 (kv5) | 2% / 2% | unexplained (best R2 0.03) | unexplained (best R2 0.04) |

</details>

</details>

<details><summary>L1H23 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.93 / 0.93 | 0.07 / 0.07 |  |  |  |
| op | 0.96 / 0.98 | 0.00 / 0.00 | 0.04 / 0.02 |  |  |
| b | 0.94 / 0.91 | 0.01 / 0.02 | 0.00 / 0.01 | 0.05 / 0.06 |  |
| = | 0.90 / 0.91 | 0.00 / 0.00 | 0.01 / 0.00 | 0.00 / 0.00 | 0.08 / 0.08 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L1 o c492 (H23) | 3% / 4% | unexplained (best R2 0.32) | **res%100** (R2 0.53): res mod 100 in {0} | - | res: mod100 +4% |

</details>

<details><summary>1 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L1 v c1 (kv5) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>20 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L1 v c21 (kv5) | 11% / 13% | **a//10** (R2 0.89): (tens) a in {49..59} | **a** (R2 0.99): a in {47..59} |
| L1 v c28 (kv5) | 16% / 16% | **a** (R2 0.98): a in {7, 66..80} | **a** (R2 0.97): a in {7, 66..80} |
| L1 v c43 (kv5) | 11% / 12% | **a%10** (R2 0.89): a mod 10 in {5} | **a%10** (R2 0.86): a mod 10 in {5} |
| L1 v c50 (kv5) | 4% / 4% | **a** (R2 0.83): a in {37..39} | **a** (R2 0.82): a in {37..39} |
| L1 v c56 (kv5) | 4% / 3% | **a** (R2 0.87): a in {35, 37..38} | **a** (R2 0.90): a in {35, 37..38} |
| L1 v c60 (kv5) | 5% / 5% | **a** (R2 0.98): a in {5, 15, 25, 65, 75} [coarser: a mod 50 in {5, 15, 25}, R2 0.90] | **a** (R2 0.94): a in {5, 15, 25, 65, 75} [coarser: a mod 50 in {5, 15, 25}, R2 0.90] |
| L1 v c61 (kv5) | 6% / 6% | **a** (R2 0.86): a in {7..10, 98} | **a** (R2 0.82): a in {7..10, 98} |
| L1 v c64 (kv5) | 8% / 7% | **a** (R2 0.99): a in {86, 88..94} | **a** (R2 0.95): a in {86, 89..94} |
| L1 v c65 (kv5) | 6% / 6% | **a** (R2 0.94): a in {7, 17, 27, 37, 87, 97} | **a** (R2 0.96): a in {7, 17, 27, 37, 87, 97} |
| L1 v c78 (kv5) | 17% / 14% | **a** (R2 0.96): a in {1..4, 21, 31..32, 41, 51, 61..62, 71..72, 81, 91..92, 98} | **a** (R2 0.88): a in {1..4, 21, 31..32, 41, 51, 61..62, 81, 91..92} |
| L1 v c82 (kv5) | 14% / 16% | **a** (R2 0.98): a in {76..89} | **a** (R2 0.98): a in {8, 75..89} |
| L1 v c94 (kv5) | 11% / 7% | **a** (R2 0.86): a in {6, 13, 16, 23, 43, 53, 56, 66, 73, 93, 96} [coarser: a mod 50 in {6, 13, 16, 23, 43, 46}, R2 0.82] | **a** (R2 0.67): a in {6, 13, 16, 23, 43, 53, 56} |
| L1 v c116 (kv5) | 8% / 11% | **a%50** (R2 0.84): a mod 50 in {1, 11, 21, 31, 41} [coarser: a mod 10 in {1}, R2 0.85] | **a%50** (R2 0.85): a mod 50 in {1, 9, 11, 21, 31, 41} [coarser: a mod 10 in {1}, R2 0.88] |
| L1 v c117 (kv5) | 8% / 7% | **a** (R2 0.95): a in {2, 12, 32, 52, 62, 72, 82, 92} [coarser: a mod 20 in {2, 12}, R2 0.81] | **a%50** (R2 0.88): a mod 50 in {2, 12, 32} |
| L1 v c118 (kv5) | 9% / 9% | **a** (R2 0.98): a in {9, 19, 29, 39, 49, 79, 89, 98..99} [coarser: a mod 50 in {9, 29, 39, 49}, R2 0.82] | **a** (R2 0.99): a in {9, 19, 29, 39, 49, 79, 89, 98..99} [coarser: a mod 50 in {9, 29, 39, 49}, R2 0.82] |
| L1 v c146 (kv5) | 5% / 5% | **a** (R2 0.99): a in {45..49} | **a** (R2 1.00): a in {45..49} |
| L1 v c154 (kv5) | 10% / 9% | **a%10** (R2 0.90): a mod 10 in {4} | **a%10** (R2 0.82): a mod 10 in {4} |
| L1 v c157 (kv5) | 14% / 16% | **a** (R2 0.94): a in {6, 56, 60..69, 86} | **a** (R2 0.89): a in {6, 56, 59..69, 86} |
| L1 v c215 (kv5) | 8% / 8% | **a** (R2 0.99): a in {57..64} | **a** (R2 1.00): a in {57..64} |
| L1 v c228 (kv5) | 4% / 4% | **a** (R2 1.00): a in {97..100} | same |

</details>

<details><summary>2 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L1 v c8 (kv5) | 100% / 0 | always | off (on 0) |
| L1 v c15 (kv5) | 100% / 0 | always | off (on 0) |

</details>

<details><summary>1 q components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L1 q c0 (H23) | 100% / 100% | always | same |

</details>

<details><summary>2 q components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L1 q c102 (H23) | 1% / 1% | unexplained (best R2 0.19) | unexplained (best R2 0.18) |
| L1 q c135 (H23) | 3% / 3% | **tens(a,b)** (R2 0.56): a//10 in {7} -> b//10 in {7}; a//10 in {8} -> b//10 in {8}; a//10 in {9} -> b//10 in {9}; a//10 in {10} -> b//10 in {9,10} | **tens(a,b)** (R2 0.59): a//10 in {7} -> b//10 in {7}; a//10 in {9,10} -> b//10 in {9,10} |

</details>

<details><summary>4 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L1 k c74 (kv5) | 0 / 0 | off (on 0) | same |
| L1 k c119 (kv5) | 0 / 0 | off (on 0) | same |
| L1 k c139 (kv5) | 1% / 1% | unexplained (best R2 0.03) | same |
| L1 k c142 (kv5) | 2% / 2% | unexplained (best R2 0.03) | unexplained (best R2 0.04) |

</details>

</details>

<details><summary>L1H24 (27 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.94 / 0.94 | 0.06 / 0.06 |  |  |  |
| op | 0.92 / 0.90 | 0.07 / 0.10 | 0.01 / 0.00 |  |  |
| b | 0.42 / 0.27 | 0.54 / 0.72 | 0.03 / 0.01 | 0.01 / 0.00 |  |
| = | 0.92 / 0.93 | 0.02 / 0.04 | 0.04 / 0.02 | 0.01 / 0.01 | 0.01 / 0.01 |

<details><summary>27 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L1 o c9 (H24) | 2% / 75% | unexplained (best R2 0.32) | **tens(a,b)** (R2 0.76): a//10 in {2,3} -> b//10 in {1,2,3,4,5,6,7,8,9}; a//10 in {4,5,6,7,8,9,10} -> b//10 in {1,2,3,4,5,6,7,8,9,10} | - | a: mod100 +4%; res: mod100 +6% |
| L1 o c24 (H24) | 2% / 15% | unexplained (best R2 0.39) | **tens(a,b)** (R2 0.70): a//10 in {1} -> b//10 in {1,2,3}; a//10 in {2} -> b//10 in {0,1,2,3,4,5}; a//10 in {3} -> b//10 in {2,3,4,5}; a//10 in {4} -> b//10 in {4} | - | a: mod50 +2%; res: mod100 +5% |
| L1 o c40 (H24) | 0 / 0 | off (on 0) | same | - | same |
| L1 o c49 (H24) | 0 / 1% | off (on 0) | unexplained (best R2 0.42) | - | same |
| L1 o c57 (H24) | 0 / 0 | off (on 0) | same | - | same |
| L1 o c67 (H24) | 1% / 2% | unexplained (best R2 0.20) | unexplained (best R2 0.29) | - | same |
| L1 o c73 (H24) | 1% / 5% | unexplained (best R2 0.38) | **tens(a,b)** (R2 0.66): a//10 in {3} -> b//10 in {3,4,5}; a//10 in {4} -> b//10 in {4,5}; a//10 in {5} -> b//10 in {5} | - | same |
| L1 o c91 (H24) | 0 / 1% | off (on 0) | **a** (R2 0.83): a in {1} | - | same |
| L1 o c107 (H24) | 0 / 0 | off (on 0) | same | - | same |
| L1 o c135 (H24) | 0 / 0 | off (on 0) | same | - | same |
| L1 o c141 (H24) | 0 / 0 | off (on 0) | same | - | same |
| L1 o c142 (H24) | 1% / 1% | unexplained (best R2 0.31) | unexplained (best R2 0.36) | - | same |
| L1 o c150 (H24) | 0 / 0 | off (on 0) | same | - | same |
| L1 o c170 (H24) | 1% / 5% | unexplained (best R2 0.19) | **a** (R2 0.73): a in {6..9} | - | same |
| L1 o c175 (H24) | 2% / 4% | **a** (R2 0.53): a in {10..12} | **a** (R2 0.83): a in {9..12} | - | same |
| L1 o c180 (H24) | 1% / 5% | unexplained (best R2 0.23) | **a** (R2 0.60): a in {5..9} | - | same |
| L1 o c186 (H24) | 1% / 2% | unexplained (best R2 0.28) | **tens(a,b)** (R2 0.58): a//10 in {3} -> b//10 in {3,4}; a//10 in {4} -> b//10 in {4} | - | same |
| L1 o c223 (H24) | 11% / 19% | **tens(a,b)** (R2 0.80): a//10 in {7} -> b//10 in {6}; a//10 in {8,10} -> b//10 in {6,7}; a//10 in {9} -> b//10 in {4,5,6,7,8,9} | **tens(a,b)** (R2 0.85): a//10 in {7} -> b//10 in {5,6,7}; a//10 in {8} -> b//10 in {3,4,5,6,7,8}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {4,5,6,7,8,9,10} | - | a: mod100 +2%, mod50 +3%, mod20 +2% |
| L1 o c261 (H24) | 0 / 0 | off (on 0) | same | - | same |
| L1 o c270 (H24) | 1% / 2% | unexplained (best R2 0.25) | unexplained (best R2 0.44) | - | same |
| L1 o c285 (H24) | 4% / 6% | **a** (R2 0.78): a in {1..4} | **a** (R2 0.84): a in {1..5} | - | same |
| L1 o c303 (H24) | 1% / 11% | unexplained (best R2 0.24) | **a** (R2 0.76): a in {10..23} | - | same |
| L1 o c371 (H24) | 0 / 0 | off (on 0) | same | - | same |
| L1 o c378 (H24) | 0 / 0 | off (on 0) | same | - | same |
| L1 o c463 (H24) | 49% / 51% | **a//10** (R2 0.86): (tens) a in {1..49} | **a//10** (R2 0.83): (tens) a in {1..49, 54} | a: mod100 +12%, mod50 +4% | a: mod100 +11%, mod50 +3%; res: mod100 +3% |
| L1 o c465 (H24) | 62% / 68% | **a//10** (R2 0.91): (tens) a in {39, 41..100} | **a//10** (R2 0.89): (tens) a in {33..100} | a: mod100 +22%, mod50 +11%, mod25 +7%, mod20 +5% | a: mod100 +22%, mod50 +12%, mod25 +7%, mod20 +6%; res: mod100 +5% |
| L1 o c466 (H24) | 1% / 2% | unexplained (best R2 0.23) | unexplained (best R2 0.41) | - | same |

</details>

<details><summary>23 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L1 v c22 (kv6) | 5% / 5% | **a** (R2 1.00): a in {13..17} | same |
| L1 v c45 (kv6) | 1% / 1% | **a** (R2 1.00): a in {24} | same |
| L1 v c52 (kv6) | 1% / 1% | **a** (R2 0.93): a in {8} | **a** (R2 0.79): a in {8} |
| L1 v c53 (kv6) | 91% / 91% | **a** (R2 0.98): a in {1..23, 33..100} | **a** (R2 1.00): a in {1..23, 33..100} |
| L1 v c71 (kv6) | 15% / 17% | **a** (R2 0.96): a in {1..12, 18, 21} | **a** (R2 0.99): a in {1..12, 18..22} |
| L1 v c72 (kv6) | 2% / 2% | **a** (R2 1.00): a in {11..12} | same |
| L1 v c85 (kv6) | 1% / 2% | **a** (R2 0.57): a in {24} | **a** (R2 0.91): a in {24..25} |
| L1 v c86 (kv6) | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same |
| L1 v c89 (kv6) | 2% / 2% | **a** (R2 1.00): a in {32..33} | same |
| L1 v c90 (kv6) | 1% / 1% | **a** (R2 1.00): a in {1} | same |
| L1 v c93 (kv6) | 1% / 1% | **a** (R2 0.90): a in {80} | **a** (R2 0.88): a in {80} |
| L1 v c131 (kv6) | 13% / 13% | **a** (R2 1.00): a in {19..31} | **a** (R2 0.98): a in {19..31} |
| L1 v c138 (kv6) | 11% / 14% | **a** (R2 0.96): a in {26..36} | **a** (R2 0.97): a in {24..37} |
| L1 v c147 (kv6) | 0 / 1% | off (on 0) | **a** (R2 0.62): a in {28} |
| L1 v c156 (kv6) | 4% / 4% | **a** (R2 1.00): a in {20..23} | **a** (R2 0.97): a in {20..23} |
| L1 v c171 (kv6) | 25% / 27% | **a** (R2 0.98): a in {4..28} | **a** (R2 1.00): a in {4..30} |
| L1 v c183 (kv6) | 4% / 5% | **a** (R2 0.99): a in {10..11, 50, 100} | **a** (R2 1.00): a in {10..12, 50, 100} |
| L1 v c213 (kv6) | 1% / 2% | **a** (R2 1.00): a in {1} | **a%50** (R2 0.80): a mod 50 in {1} |
| L1 v c217 (kv6) | 6% / 6% | **a** (R2 1.00): a in {8..13} | same |
| L1 v c221 (kv6) | 5% / 5% | **a** (R2 1.00): a in {40..44} | same |
| L1 v c233 (kv6) | 2% / 3% | **a** (R2 0.95): a in {49..50} | **a** (R2 0.76): a in {48..50} |
| L1 v c241 (kv6) | 2% / 2% | **a** (R2 1.00): a in {84, 86} | **a** (R2 0.99): a in {84, 86} |
| L1 v c255 (kv6) | 1% / 1% | **a** (R2 1.00): a in {99} | same |

</details>

</details>

<details><summary>L1H26 (26 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.81 / 0.81 | 0.19 / 0.19 |  |  |  |
| op | 0.58 / 0.56 | 0.41 / 0.42 | 0.01 / 0.01 |  |  |
| b | 0.76 / 0.67 | 0.13 / 0.28 | 0.08 / 0.03 | 0.04 / 0.02 |  |
| = | 0.84 / 0.87 | 0.01 / 0.02 | 0.04 / 0.02 | 0.04 / 0.04 | 0.06 / 0.06 |

<details><summary>2 o components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L1 o c70 (H26) | 78% / 80% | **a** (R2 1.00): a in {8..12, 21..32, 40..100} | **a** (R2 0.98): a in {7..12, 20..32, 40..100} | - | same |
| L1 o c231 (H26) | 100% / 100% | always | same | - | same |

</details>

<details><summary>23 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L1 o c58 (H26) | 0 / 12% | off (on 0) | **a** (R2 0.99): a in {18..29} | - | same |
| L1 o c66 (H26) | 9% / 16% | **a** (R2 0.92): a in {50..58} | **a** (R2 0.90): a in {48..62} | - | same |
| L1 o c106 (H26) | 15% / 17% | **a** (R2 0.98): a in {34, 36..49} | **a** (R2 0.97): a in {4, 34, 36..49} | - | a: mod50 +3% |
| L1 o c108 (H26) | 19% / 28% | **a** (R2 0.98): a in {2, 12, 18, 20..24, 26..34, 36, 42} | **a** (R2 0.96): a in {2, 12, 18..42} | a: mod2 +2% | a: mod100 +3%, mod50 +2% |
| L1 o c110 (H26) | 17% / 10% | **a%50** (R2 0.96): a mod 50 in {0, 10, 15, 20, 25, 30, 35, 40} [coarser: a mod 5 in {0}, R2 0.82] | **a%10** (R2 1.00): a mod 10 in {0} | a: mod10 +9%, mod5 +9%, mod2 +7% | a: mod10 +5%, mod5 +5%, mod2 +4% |
| L1 o c159 (H26) | 0 / 1% | off (on 0) | **a//10** (R2 1.00): (tens) a in {100} | - | same |
| L1 o c179 (H26) | 1% / 2% | unexplained (best R2 0.03) | unexplained (best R2 0.14) | - | same |
| L1 o c188 (H26) | 8% / 9% | **a** (R2 1.00): a in {1..8} | **a//10** (R2 1.00): (tens) a in {1..9} | - | same |
| L1 o c191 (H26) | 0 / 1% | off (on 0) | unexplained (best R2 0.04) | - | same |
| L1 o c201 (H26) | 19% / 10% | **a** (R2 0.98): a in {3..14, 17, 19, 59, 69, 79, 89, 99} | **a** (R2 0.99): a in {3..12} | - | same |
| L1 o c243 (H26) | 1% / 0 | **a** (R2 1.00): a in {90} | off (on 0) | - | same |
| L1 o c244 (H26) | 1% / 0 | **a** (R2 0.99): a in {11} | off (on 0) | - | same |
| L1 o c315 (H26) | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | - | same |
| L1 o c320 (H26) | 3% / 3% | **a** (R2 1.00): a in {10..12} | same | - | same |
| L1 o c330 (H26) | 16% / 20% | **a** (R2 0.99): a in {9, 86..100} | **a** (R2 1.00): a in {9..12, 85..100} | a: mod100 +3%, mod50 +5%, mod25 +4%, mod20 +5% | a: mod100 +3%, mod50 +5%, mod25 +4%, mod20 +4% |
| L1 o c353 (H26) | 0 / 1% | off (on 0) | unexplained (best R2 0.10) | - | same |
| L1 o c377 (H26) | 18% / 18% | **a//10** (R2 0.94): (tens) a in {1..17, 19} | **a//10** (R2 0.94): (tens) a in {1..18} | a: mod100 +7%, mod50 +10%, mod25 +7%, mod20 +7% | a: mod100 +5%, mod50 +7%, mod25 +5%, mod20 +5% |
| L1 o c388 (H26) | 18% / 17% | **a%50** (R2 0.93): a mod 50 in {0, 10, 15, 20, 25, 30, 35, 40} [coarser: a mod 5 in {0}, R2 0.88] | **a%50** (R2 0.88): a mod 50 in {0, 5, 10, 15, 20, 25, 30, 35, 40} [coarser: a mod 5 in {0}, R2 0.85] | a: mod5 +21% | a: mod5 +10% |
| L1 o c398 (H26) | 22% / 25% | **a** (R2 0.95): a in {75, 77..89, 93..100} | **a** (R2 0.96): a in {8..9, 75..89, 93..100} | - | same |
| L1 o c403 (H26) | 19% / 19% | **a** (R2 1.00): a in {5..23} | **a** (R2 0.99): a in {6..23, 99} | a: mod100 +3%, mod50 +4%, mod25 +2% | a: mod100 +3%, mod50 +3%, mod25 +2% |
| L1 o c413 (H26) | 37% / 38% | **a//10** (R2 0.92): (tens) a in {2..38} | **a//10** (R2 0.96): (tens) a in {2..39} | a: mod100 +9%, mod50 +7% | a: mod100 +7%, mod50 +5% |
| L1 o c469 (H26) | 8% / 2% | **a** (R2 0.98): a in {20, 30, 40, 50, 60, 70, 80, 90} [coarser: a mod 10 in {0}, R2 0.82] | **a** (R2 0.95): a in {80, 90} | - | same |
| L1 o c488 (H26) | 4% / 2% | **a** (R2 0.92): a in {1, 11, 21, 51} [coarser: a mod 50 in {1, 11, 21}, R2 0.82] | **a%50** (R2 0.93): a mod 50 in {1} | - | same |

</details>

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L1 o c251 (H26) | 1% / 0 | **a** (R2 0.60): a in {90} | off (on 0) | - | same |

</details>

<details><summary>23 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L1 v c22 (kv6) | 5% / 5% | **a** (R2 1.00): a in {13..17} | same |
| L1 v c45 (kv6) | 1% / 1% | **a** (R2 1.00): a in {24} | same |
| L1 v c52 (kv6) | 1% / 1% | **a** (R2 0.93): a in {8} | **a** (R2 0.79): a in {8} |
| L1 v c53 (kv6) | 91% / 91% | **a** (R2 0.98): a in {1..23, 33..100} | **a** (R2 1.00): a in {1..23, 33..100} |
| L1 v c71 (kv6) | 15% / 17% | **a** (R2 0.96): a in {1..12, 18, 21} | **a** (R2 0.99): a in {1..12, 18..22} |
| L1 v c72 (kv6) | 2% / 2% | **a** (R2 1.00): a in {11..12} | same |
| L1 v c85 (kv6) | 1% / 2% | **a** (R2 0.57): a in {24} | **a** (R2 0.91): a in {24..25} |
| L1 v c86 (kv6) | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same |
| L1 v c89 (kv6) | 2% / 2% | **a** (R2 1.00): a in {32..33} | same |
| L1 v c90 (kv6) | 1% / 1% | **a** (R2 1.00): a in {1} | same |
| L1 v c93 (kv6) | 1% / 1% | **a** (R2 0.90): a in {80} | **a** (R2 0.88): a in {80} |
| L1 v c131 (kv6) | 13% / 13% | **a** (R2 1.00): a in {19..31} | **a** (R2 0.98): a in {19..31} |
| L1 v c138 (kv6) | 11% / 14% | **a** (R2 0.96): a in {26..36} | **a** (R2 0.97): a in {24..37} |
| L1 v c147 (kv6) | 0 / 1% | off (on 0) | **a** (R2 0.62): a in {28} |
| L1 v c156 (kv6) | 4% / 4% | **a** (R2 1.00): a in {20..23} | **a** (R2 0.97): a in {20..23} |
| L1 v c171 (kv6) | 25% / 27% | **a** (R2 0.98): a in {4..28} | **a** (R2 1.00): a in {4..30} |
| L1 v c183 (kv6) | 4% / 5% | **a** (R2 0.99): a in {10..11, 50, 100} | **a** (R2 1.00): a in {10..12, 50, 100} |
| L1 v c213 (kv6) | 1% / 2% | **a** (R2 1.00): a in {1} | **a%50** (R2 0.80): a mod 50 in {1} |
| L1 v c217 (kv6) | 6% / 6% | **a** (R2 1.00): a in {8..13} | same |
| L1 v c221 (kv6) | 5% / 5% | **a** (R2 1.00): a in {40..44} | same |
| L1 v c233 (kv6) | 2% / 3% | **a** (R2 0.95): a in {49..50} | **a** (R2 0.76): a in {48..50} |
| L1 v c241 (kv6) | 2% / 2% | **a** (R2 1.00): a in {84, 86} | **a** (R2 0.99): a in {84, 86} |
| L1 v c255 (kv6) | 1% / 1% | **a** (R2 1.00): a in {99} | same |

</details>

</details>

<details><summary>L2H2 (5 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 1.00 / 1.00 | 0.00 / 0.00 |  |  |  |
| op | 0.97 / 1.00 | 0.00 / 0.00 | 0.03 / 0.00 |  |  |
| b | 0.96 / 0.96 | 0.01 / 0.01 | 0.02 / 0.02 | 0.00 / 0.01 |  |
| = | 0.34 / 0.49 | 0.00 / 0.00 | 0.64 / 0.48 | 0.01 / 0.03 | 0.01 / 0.01 |

<details><summary>5 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L2 o c2 (H2) | 100% / 0 | always | off (on 0) | a: mod100 +3%, mod50 +3%, mod25 +4%, mod20 +9%, mod10 +12%, mod5 +24%, mod4 +6%, mod2 +14%; b: mod4 +3% | - |
| L2 o c3 (H2) | 100% / 100% | always | same | a: mod100 +7%, mod50 +4%, mod25 +4%, mod20 +4%, mod10 +7%, mod5 +11%, mod4 +6%, mod2 +10% | a: mod25 +4%, mod10 +8%, mod4 +3% |
| L2 o c6 (H2) | 0 / 100% | off (on 0) | always | - | a: mod100 +12%, mod50 +15%, mod25 +5%, mod20 +17%, mod10 +8%, mod4 +8% |
| L2 o c14 (H2) | 0 / 100% | off (on 0) | always | - | a: mod100 +4%, mod25 +3%, mod10 +8%, mod4 +3%; b: mod100 +3%, mod50 +3%, mod20 +2% |
| L2 o c217 (H2) | 15% / 0 | **a** (R2 0.87): a in {1..14} | off (on 0) | a: mod100 +13%, mod50 +15%, mod25 +11%, mod20 +11% | - |

</details>

<details><summary>3 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L2 v c9 (kv0) | 16% / 17% | **a** (R2 0.97): a in {6..21} | **a** (R2 0.86): a in {8..24} |
| L2 v c22 (kv0) | 0 / 100% | off (on 0) | always |
| L2 v c27 (kv0) | 0 / 100% | off (on 0) | always |

</details>

</details>

<details><summary>L2H8 (2 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.98 / 0.98 | 0.02 / 0.02 |  |  |  |
| op | 0.88 / 0.94 | 0.12 / 0.06 | 0.00 / 0.00 |  |  |
| b | 0.92 / 0.88 | 0.07 / 0.11 | 0.01 / 0.01 | 0.00 / 0.00 |  |
| = | 0.90 / 0.91 | 0.02 / 0.03 | 0.04 / 0.02 | 0.03 / 0.03 | 0.00 / 0.01 |

<details><summary>2 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L2 o c26 (H8) | 10% / 1% | **a** (R2 1.00): a in {11, 13..14, 16..19, 21, 86, 91} | **a** (R2 0.92): a in {14} | - | same |
| L2 o c506 (H8) | 5% / 2% | **a** (R2 1.00): a in {11, 16, 18, 21, 55} | **a** (R2 0.96): a in {18, 70} | - | same |

</details>

<details><summary>8 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L2 v c76 (kv2) | 1% / 1% | **a** (R2 1.00): a in {11} | same |
| L2 v c92 (kv2) | 1% / 1% | **a** (R2 1.00): a in {18} | **a** (R2 0.51): a in {18} |
| L2 v c160 (kv2) | 10% / 10% | **a//10** (R2 1.00): (tens) a in {50..59} | **a//10** (R2 0.94): (tens) a in {50, 52..59} |
| L2 v c173 (kv2) | 2% / 1% | **a** (R2 0.99): a in {18, 21} | **a** (R2 0.85): a in {18} |
| L2 v c177 (kv2) | 3% / 4% | **a** (R2 0.91): a in {70, 74, 80} | **a** (R2 0.83): a in {70, 74, 80} |
| L2 v c198 (kv2) | 1% / 0 | **a** (R2 1.00): a in {18} | off (on 0) |
| L2 v c247 (kv2) | 2% / 1% | **a** (R2 0.99): a in {21, 70} | **a** (R2 0.82): a in {70} |
| L2 v c251 (kv2) | 3% / 3% | **a** (R2 1.00): a in {83..84, 86} | **a** (R2 0.92): a in {83..84, 86} |

</details>

<details><summary>1 q components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L2 q c112 (H8) | 28% / 0 | **a** (R2 0.99): a in {11, 13..14, 16..19, 21, 51, 55, 57, 81..83, 86..99} | off (on 0) |

</details>

<details><summary>1 q components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L2 q c131 (H8) | 1% / 4% | unexplained (best R2 0.30) | unexplained (best R2 0.16) |

</details>

<details><summary>5 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L2 k c8 (kv2) | 100% / 100% | always | same |
| L2 k c62 (kv2) | 10% / 2% | **a** (R2 1.00): a in {11, 16..18, 21, 55, 86, 90..92} | **a** (R2 0.65): a in {18, 70} |
| L2 k c80 (kv2) | 0 / 0 | off (on 0) | same |
| L2 k c114 (kv2) | 3% / 0 | **a** (R2 1.00): a in {11, 18, 21} | off (on 0) |
| L2 k c125 (kv2) | 2% / 0 | **a** (R2 1.00): a in {11, 18} | off (on 0) |

</details>

</details>

<details><summary>L2H13 (4 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.99 / 0.99 | 0.01 / 0.01 |  |  |  |
| op | 0.99 / 1.00 | 0.00 / 0.00 | 0.01 / 0.00 |  |  |
| b | 0.81 / 0.86 | 0.00 / 0.01 | 0.18 / 0.12 | 0.01 / 0.01 |  |
| = | 0.82 / 0.94 | 0.00 / 0.00 | 0.02 / 0.01 | 0.14 / 0.02 | 0.03 / 0.03 |

<details><summary>4 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L2 o c5 (H13) | 30% / 15% | **tens(a,b)** (R2 0.71): a//10 in {0} -> b//10 in {0}; a//10 in {1} -> b//10 in {1,2}; a//10 in {2} -> b//10 in {2,3}; a//10 in {3} -> b//10 in {3,4}; a//10 in {4} -> b//10 in {4,5,6}; a//10 in {5,6} -> b//10 in {5,6,7,8,9}; a//10 in {7,8} -> b//10 in {6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {9,10} | **res%100** (R2 0.69): res mod 100 in {0} | res: mod100 +5% | res: mod100 +5%, mod50 +9%, mod25 +8%, mod20 +7%, mod10 +3% |
| L2 o c196 (H13) | 2% / 15% | unexplained (best R2 0.15) | **res%100** (R2 0.71): res mod 100 in {0, 89} | - | res: mod100 +7%, mod50 +15%, mod25 +16%, mod20 +15%, mod10 +7%, mod5 +4% |
| L2 o c246 (H13) | 2% / 0 | unexplained (best R2 0.15) | off (on 0) | - | same |
| L2 o c410 (H13) | 0 / 50% | off (on 0) | **tens(a,b)** (R2 0.76): a//10 in {0} -> b//10 in {1}; a//10 in {1} -> b//10 in {1,2,3,4}; a//10 in {2} -> b//10 in {2,3,4,5,6,7,8}; a//10 in {3} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {4} -> b//10 in {4,5,6,7,8,9}; a//10 in {5,6} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {5,6,7,8,9,10}; a//10 in {8} -> b//10 in {6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {6,7,8,9} | - | res: mod100 +18%, mod50 +16%, mod25 +10%, mod20 +9%, mod10 +4% |

</details>

<details><summary>4 q components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L2 q c45 (H13) | 1% / 3% | unexplained (best R2 0.33) | **tens(a,b)** (R2 0.58): a//10 in {5} -> b//10 in {5,6}; a//10 in {6} -> b//10 in {6} |
| L2 q c62 (H13) | 0 / 2% | off (on 0) | unexplained (best R2 0.45) |
| L2 q c117 (H13) | 11% / 23% | **tens(a,b)** (R2 0.60): a//10 in {1} -> b//10 in {1}; a//10 in {2} -> b//10 in {2}; a//10 in {3} -> b//10 in {3,4}; a//10 in {4} -> b//10 in {4,5}; a//10 in {5} -> b//10 in {5,6}; a//10 in {6} -> b//10 in {6}; a//10 in {7} -> b//10 in {7} | **tens(a,b)** (R2 0.72): a//10 in {0} -> b//10 in {0}; a//10 in {1} -> b//10 in {1,2}; a//10 in {2} -> b//10 in {2,3,4,5}; a//10 in {3} -> b//10 in {3,4,5,6}; a//10 in {4} -> b//10 in {4,5,6}; a//10 in {5} -> b//10 in {4,5,6,7}; a//10 in {6} -> b//10 in {5,6,7}; a//10 in {7} -> b//10 in {7} |
| L2 q c142 (H13) | 2% / 4% | **tens(a,b)** (R2 0.72): a//10 in {8} -> b//10 in {9}; a//10 in {9} -> b//10 in {9,10} | **tens(a,b)** (R2 0.73): a//10 in {7,8,9} -> b//10 in {9,10} |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L2 k c123 (kv3) | 100% / 100% | always | same |

</details>

<details><summary>12 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L2 k c37 (kv3) | 0 / 13% | off (on 0) | unexplained (best R2 0.08) |
| L2 k c38 (kv3) | 1% / 8% | unexplained (best R2 0.03) | unexplained (best R2 0.14) |
| L2 k c69 (kv3) | 3% / 10% | unexplained (best R2 0.06) | unexplained (best R2 0.12) |
| L2 k c73 (kv3) | 1% / 7% | unexplained (best R2 0.02) | unexplained (best R2 0.05) |
| L2 k c82 (kv3) | 0 / 2% | off (on 0) | unexplained (best R2 0.04) |
| L2 k c98 (kv3) | 3% / 6% | unexplained (best R2 0.09) | unexplained (best R2 0.13) |
| L2 k c111 (kv3) | 1% / 8% | unexplained (best R2 0.06) | unexplained (best R2 0.17) |
| L2 k c112 (kv3) | 3% / 9% | unexplained (best R2 0.08) | unexplained (best R2 0.15) |
| L2 k c117 (kv3) | 0 / 18% | off (on 0) | unexplained (best R2 0.08) |
| L2 k c120 (kv3) | 0 / 3% | off (on 0) | unexplained (best R2 0.09) |
| L2 k c133 (kv3) | 0 / 5% | off (on 0) | unexplained (best R2 0.10) |
| L2 k c139 (kv3) | 2% / 7% | unexplained (best R2 0.05) | unexplained (best R2 0.13) |

</details>

</details>

<details><summary>L2H17 (9 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 1.00 / 1.00 | 0.00 / 0.00 |  |  |  |
| op | 0.97 / 0.96 | 0.01 / 0.01 | 0.02 / 0.03 |  |  |
| b | 0.48 / 0.73 | 0.02 / 0.05 | 0.46 / 0.19 | 0.04 / 0.03 |  |
| = | 0.75 / 0.84 | 0.01 / 0.01 | 0.13 / 0.05 | 0.07 / 0.04 | 0.04 / 0.06 |

<details><summary>9 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L2 o c15 (H17) | 99% / 99% | always | same | - | a: mod5 +3%, mod2 +2%; b: mod100 +3%; res: mod100 +4%, mod50 +6%, mod25 +7%, mod20 +7%, mod10 +7%, mod5 +6% |
| L2 o c17 (H17) | 100% / 0 | always | off (on 0) | a: mod5 +2%; res: mod100 +4% | - |
| L2 o c21 (H17) | 100% / 0 | always | off (on 0) | a: mod10 +3%, mod5 +2% | - |
| L2 o c57 (H17) | 0 / 5% | off (on 0) | unexplained (best R2 0.35) | - | same |
| L2 o c114 (H17) | 1% / 0 | unexplained (best R2 0.32) | off (on 0) | - | same |
| L2 o c136 (H17) | 0 / 10% | off (on 0) | **tens(a,b)** (R2 0.60): a//10 in {0} -> b//10 in {5,6,7,8,9,10}; a//10 in {1} -> b//10 in {7,8,9} | - | same |
| L2 o c137 (H17) | 1% / 0 | unexplained (best R2 0.34) | off (on 0) | - | same |
| L2 o c184 (H17) | 21% / 1% | **a//10** (R2 0.79): (tens) a in {1..19, 24} | unexplained (best R2 0.08) | a: mod100 +5%, mod50 +6%, mod25 +4%, mod20 +5% | - |
| L2 o c330 (H17) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>5 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L2 v c18 (kv4) | 100% / 0 | always | off (on 0) |
| L2 v c23 (kv4) | 100% / 0 | always | off (on 0) |
| L2 v c61 (kv4) | 100% / 0 | always | off (on 0) |
| L2 v c97 (kv4) | 89% / 0 | **a** (R2 1.00): a in {1..19, 21..29, 31..39, 41..49, 51..54, 56..59, 61..64, 66..69, 71..79, 81..89, 91..99} [coarser: a mod 50 in {1..4, 6..9, 11..14, 16..19, 21..29, 31..39, 41..49}, R2 0.85] | off (on 0) |
| L2 v c120 (kv4) | 11% / 12% | **a** (R2 1.00): a in {1..10, 12} | **a** (R2 0.98): a in {1..12} |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L2 k c11 (kv4) | 100% / 0 | always | off (on 0) |

</details>

</details>

<details><summary>L2H23 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.76 / 0.76 | 0.24 / 0.24 |  |  |  |
| op | 0.86 / 0.91 | 0.13 / 0.09 | 0.00 / 0.00 |  |  |
| b | 0.95 / 0.88 | 0.04 / 0.11 | 0.01 / 0.00 | 0.00 / 0.00 |  |
| = | 0.97 / 0.96 | 0.02 / 0.03 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |

<details><summary>1 o components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L2 o c89 (H23) | 100% / 100% | always | same | - | same |

</details>

</details>

<details><summary>L2H24 (9 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.98 / 0.98 | 0.02 / 0.02 |  |  |  |
| op | 0.80 / 0.81 | 0.18 / 0.18 | 0.02 / 0.01 |  |  |
| b | 0.90 / 0.71 | 0.02 / 0.10 | 0.07 / 0.18 | 0.01 / 0.02 |  |
| = | 0.89 / 0.86 | 0.02 / 0.01 | 0.02 / 0.02 | 0.03 / 0.05 | 0.03 / 0.05 |

<details><summary>9 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L2 o c20 (H24) | 20% / 20% | **a** (R2 1.00): a in {11, 81..84, 86..100} | **a** (R2 0.98): a in {79, 81..99} | a: mod100 +2%, mod50 +2% | a: mod100 +2%, mod50 +3% |
| L2 o c48 (H24) | 55% / 58% | **a//10** (R2 0.90): (tens) a in {44, 46, 48..100} | **a//10** (R2 0.91): (tens) a in {43..100} | a: mod100 +10%, mod50 +2% | a: mod100 +14%, mod50 +6%, mod25 +2% |
| L2 o c77 (H24) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {70} | - | same |
| L2 o c93 (H24) | 33% / 2% | **a** (R2 0.97): a in {22..39, 41..49, 52..54, 56, 58..59} | **a** (R2 0.87): a in {46..47} | - | same |
| L2 o c228 (H24) | 3% / 0 | **a** (R2 1.00): a in {11, 18, 21} | off (on 0) | - | same |
| L2 o c278 (H24) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {70} | - | same |
| L2 o c337 (H24) | 4% / 0 | **a** (R2 1.00): a in {95, 97..99} | off (on 0) | - | same |
| L2 o c389 (H24) | 1% / 1% | **a** (R2 1.00): a in {70} | same | - | same |
| L2 o c396 (H24) | 1% / 0 | **a** (R2 1.00): a in {21} | off (on 0) | - | same |

</details>

<details><summary>5 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L2 v c69 (kv6) | 18% / 21% | **a//10** (R2 0.89): (tens) a in {81..84, 86..99} | **a** (R2 1.00): a in {78..79, 81..99} |
| L2 v c157 (kv6) | 2% / 2% | **a** (R2 1.00): a in {98..99} | same |
| L2 v c172 (kv6) | 13% / 13% | **a** (R2 1.00): a in {3..14, 16} | **a** (R2 0.98): a in {3..14, 16} |
| L2 v c194 (kv6) | 30% / 31% | **a//10** (R2 1.00): (tens) a in {20..49} | **a//10** (R2 0.95): (tens) a in {20..50} |
| L2 v c207 (kv6) | 16% / 16% | **a** (R2 1.00): a in {8, 10..24} | **a** (R2 0.99): a in {8, 10..24} |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L2 v c185 (kv6) | 100% / 100% | always | same |

</details>

<details><summary>1 q components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L2 q c124 (H24) | 73% / 79% | **tens(a,b)** (R2 0.83): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1,2} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {3,4} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {5,6,7} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10}; a//10 in {10} -> b//10 in {6,7,8,9} | **tens(a,b)** (R2 0.58): a//10 in {0,1} -> b//10 in {1,2,3,4,5,6,7,8,9}; a//10 in {2} -> b//10 in {2,3,4,5,6,7,8,9}; a//10 in {3,4} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {5} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {0,1,2,4,5,6,7,8,9}; a//10 in {7} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,6,7,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,7,8,9,10}; a//10 in {10} -> b//10 in {1,2,6,7,8,9} |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L2 k c93 (kv6) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L2H27 (10 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 1.00 / 1.00 | 0.00 / 0.00 |  |  |  |
| op | 0.98 / 0.97 | 0.01 / 0.02 | 0.01 / 0.01 |  |  |
| b | 0.92 / 0.93 | 0.00 / 0.01 | 0.08 / 0.06 | 0.00 / 0.01 |  |
| = | 0.82 / 0.92 | 0.01 / 0.01 | 0.02 / 0.01 | 0.11 / 0.03 | 0.04 / 0.03 |

<details><summary>10 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L2 o c36 (H27) | 0 / 0 | off (on 0) | same | - | same |
| L2 o c303 (H27) | 1% / 1% | **cmp(a,b)** (R2 0.93): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 1.00, cmp(a,b)=1: 0.00 | **cmp(a,b)** (R2 0.97): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.98, cmp(a,b)=1: 0.00 | - | same |
| L2 o c318 (H27) | 1% / 1% | **cmp(a,b)** (R2 0.94): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 1.00, cmp(a,b)=1: 0.00 | **cmp(a,b)** (R2 0.92): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 1.00, cmp(a,b)=1: 0.00 | - | res: mod5 +2% |
| L2 o c332 (H27) | 1% / 0 | **cmp(a,b)** (R2 0.76): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.92, cmp(a,b)=1: 0.00 | off (on 0) | - | same |
| L2 o c381 (H27) | 0 / 1% | off (on 0) | **cmp(a,b)** (R2 0.70): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.73, cmp(a,b)=1: 0.00 | - | same |
| L2 o c450 (H27) | 1% / 1% | **cmp(a,b)** (R2 0.68): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 1.00, cmp(a,b)=1: 0.01 | **cmp(a,b)** (R2 0.75): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.98, cmp(a,b)=1: 0.01 | - | same |
| L2 o c464 (H27) | 9% / 21% | unexplained (best R2 0.50) | **res%100** (R2 0.57): res mod 100 in {0, 5, 10, 89..90, 94, 98..99} | - | res: mod100 +5%, mod50 +14%, mod25 +23%, mod20 +23%, mod10 +20%, mod5 +16% |
| L2 o c479 (H27) | 1% / 1% | **cmp(a,b)** (R2 0.94): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 1.00, cmp(a,b)=1: 0.00 | **cmp(a,b)** (R2 0.95): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.95, cmp(a,b)=1: 0.00 | - | same |
| L2 o c507 (H27) | 0 / 0 | off (on 0) | same | - | same |
| L2 o c509 (H27) | 1% / 1% | **cmp(a,b)** (R2 0.98): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 1.00, cmp(a,b)=1: 0.00 | **cmp(a,b)** (R2 0.51): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.51, cmp(a,b)=1: 0.00 | - | same |

</details>

<details><summary>5 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L2 v c69 (kv6) | 18% / 21% | **a//10** (R2 0.89): (tens) a in {81..84, 86..99} | **a** (R2 1.00): a in {78..79, 81..99} |
| L2 v c157 (kv6) | 2% / 2% | **a** (R2 1.00): a in {98..99} | same |
| L2 v c172 (kv6) | 13% / 13% | **a** (R2 1.00): a in {3..14, 16} | **a** (R2 0.98): a in {3..14, 16} |
| L2 v c194 (kv6) | 30% / 31% | **a//10** (R2 1.00): (tens) a in {20..49} | **a//10** (R2 0.95): (tens) a in {20..50} |
| L2 v c207 (kv6) | 16% / 16% | **a** (R2 1.00): a in {8, 10..24} | **a** (R2 0.99): a in {8, 10..24} |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L2 v c185 (kv6) | 100% / 100% | always | same |

</details>

<details><summary>6 q components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L2 q c35 (H27) | 1% / 6% | unexplained (best R2 0.28) | **tens(a,b)** (R2 0.55): a//10 in {5} -> b//10 in {5}; a//10 in {6} -> b//10 in {5,6}; a//10 in {7} -> b//10 in {6,7}; a//10 in {10} -> b//10 in {10} |
| L2 q c53 (H27) | 1% / 1% | **cmp(a,b)** (R2 0.72): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.75, cmp(a,b)=1: 0.00 | **cmp(a,b)** (R2 0.62): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.84, cmp(a,b)=1: 0.00 |
| L2 q c61 (H27) | 3% / 5% | unexplained (best R2 0.33) | unexplained (best R2 0.49) |
| L2 q c89 (H27) | 3% / 6% | unexplained (best R2 0.28) | unexplained (best R2 0.34) |
| L2 q c115 (H27) | 7% / 15% | **tens(a,b)** (R2 0.51): a//10 in {6} -> b//10 in {6}; a//10 in {7} -> b//10 in {7}; a//10 in {8} -> b//10 in {8}; a//10 in {9} -> b//10 in {9}; a//10 in {10} -> b//10 in {10} | **res%100** (R2 0.64): res mod 100 in {0} |
| L2 q c119 (H27) | 29% / 38% | **tens(a,b)** (R2 0.71): a//10 in {0} -> b//10 in {0}; a//10 in {1} -> b//10 in {1}; a//10 in {2} -> b//10 in {2,3}; a//10 in {3} -> b//10 in {3,4}; a//10 in {4} -> b//10 in {4,5,6}; a//10 in {5} -> b//10 in {4,5,6,7,8}; a//10 in {6} -> b//10 in {5,6,7,8,9}; a//10 in {7,8} -> b//10 in {6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {8,9,10} | **tens(a,b)** (R2 0.65): a//10 in {0} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {1,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {2,3,6,7,8,9,10}; a//10 in {3} -> b//10 in {3,4}; a//10 in {4} -> b//10 in {4,5}; a//10 in {5} -> b//10 in {5,6}; a//10 in {6} -> b//10 in {6,7,8,9}; a//10 in {7} -> b//10 in {6,7,8,9,10}; a//10 in {8,10} -> b//10 in {7,8,9,10}; a//10 in {9} -> b//10 in {8,9,10} |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L2 k c93 (kv6) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L3H4 (2 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.99 / 0.99 | 0.01 / 0.01 |  |  |  |
| op | 0.95 / 0.98 | 0.04 / 0.02 | 0.01 / 0.00 |  |  |
| b | 0.97 / 0.97 | 0.02 / 0.01 | 0.01 / 0.01 | 0.00 / 0.01 |  |
| = | 0.96 / 0.96 | 0.00 / 0.00 | 0.01 / 0.00 | 0.02 / 0.03 | 0.01 / 0.01 |

<details><summary>2 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L3 o c0 (H4) | 1% / 0 | **a** (R2 1.00): a in {55} | off (on 0) | - | same |
| L3 o c380 (H4) | 12% / 0 | **a** (R2 0.99): a in {50..52, 54..57, 60, 62, 64..66} | off (on 0) | - | same |

</details>

<details><summary>34 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L3 v c12 (kv1) | 68% / 68% | **a//10** (R2 0.93): (tens) a in {32..99} | same |
| L3 v c13 (kv1) | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {1, 11} [coarser: a mod 10 in {1}, R2 0.89] | **a%10** (R2 0.87): a mod 10 in {1} |
| L3 v c26 (kv1) | 20% / 20% | **a//10** (R2 0.94): (tens) a in {1..20} | same |
| L3 v c28 (kv1) | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same |
| L3 v c48 (kv1) | 21% / 23% | **a//10** (R2 1.00): (tens) a in {80..100} | **a//10** (R2 0.91): (tens) a in {78..100} |
| L3 v c54 (kv1) | 18% / 18% | **a** (R2 1.00): a in {50..58, 60..68} | same |
| L3 v c59 (kv1) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {70} |
| L3 v c70 (kv1) | 100% / 100% | always | same |
| L3 v c75 (kv1) | 1% / 2% | **a** (R2 1.00): a in {55} | **a** (R2 0.99): a in {55, 77} |
| L3 v c90 (kv1) | 11% / 11% | **a%50** (R2 0.95): a mod 50 in {0, 10, 20, 25, 30, 40} | same |
| L3 v c101 (kv1) | 8% / 9% | **a** (R2 0.98): a in {13..19, 21} | **a** (R2 1.00): a in {13..21} |
| L3 v c112 (kv1) | 6% / 10% | **a** (R2 1.00): a in {1..4, 11, 13} | **a** (R2 0.98): a in {1..4, 10..15} |
| L3 v c131 (kv1) | 1% / 7% | **a** (R2 1.00): a in {62} | **a** (R2 0.96): a in {58..63, 71} |
| L3 v c133 (kv1) | 1% / 3% | **a** (R2 1.00): a in {33} | **a** (R2 0.98): a in {32..34} |
| L3 v c149 (kv1) | 1% / 3% | **a** (R2 0.97): a in {74} | **a** (R2 0.99): a in {70, 73..74} |
| L3 v c151 (kv1) | 1% / 0 | **a** (R2 1.00): a in {55} | off (on 0) |
| L3 v c159 (kv1) | 2% / 4% | **a** (R2 0.97): a in {52..53} | **a** (R2 0.97): a in {51..54} |
| L3 v c160 (kv1) | 4% / 5% | **a** (R2 0.99): a in {41..44} | **a** (R2 1.00): a in {41..45} |
| L3 v c176 (kv1) | 1% / 0 | **a** (R2 1.00): a in {55} | off (on 0) |
| L3 v c179 (kv1) | 1% / 2% | **a** (R2 0.98): a in {51} | **a** (R2 1.00): a in {51, 53} |
| L3 v c180 (kv1) | 3% / 12% | **a** (R2 0.99): a in {11, 51, 53} | **a** (R2 0.99): a in {11, 31, 41, 43, 46..47, 49, 51..53, 61, 81} |
| L3 v c187 (kv1) | 1% / 2% | **a** (R2 1.00): a in {24} | **a** (R2 0.75): a in {24, 48} |
| L3 v c191 (kv1) | 2% / 2% | **a** (R2 1.00): a in {86, 90} | same |
| L3 v c195 (kv1) | 13% / 17% | **a** (R2 0.98): a in {86, 88..99} | **a** (R2 0.95): a in {10..13, 86..99} |
| L3 v c205 (kv1) | 1% / 2% | **a** (R2 0.99): a in {55} | **a** (R2 0.99): a in {43..44} |
| L3 v c207 (kv1) | 6% / 12% | **a** (R2 0.99): a in {95..100} | **a** (R2 0.90): a in {10..13, 15, 93..100} |
| L3 v c211 (kv1) | 11% / 17% | **a** (R2 0.99): a in {16..24, 26..27} | **a** (R2 0.99): a in {13..29} |
| L3 v c222 (kv1) | 1% / 1% | **a** (R2 1.00): a in {52} | same |
| L3 v c225 (kv1) | 2% / 6% | **a** (R2 0.99): a in {48, 50} | **a** (R2 1.00): a in {45..50} |
| L3 v c228 (kv1) | 17% / 17% | **a** (R2 0.98): a in {15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100} [coarser: a mod 5 in {0}, R2 0.81] | **a** (R2 0.93): a in {20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100} [coarser: a mod 5 in {0}, R2 0.81] |
| L3 v c230 (kv1) | 0 / 12% | off (on 0) | **a** (R2 1.00): a in {63, 67..74, 77..79} |
| L3 v c232 (kv1) | 1% / 1% | **a** (R2 1.00): a in {75} | same |
| L3 v c237 (kv1) | 8% / 8% | **a** (R2 1.00): a in {1..8} | same |
| L3 v c247 (kv1) | 48% / 48% | **a//10** (R2 0.92): (tens) a in {51, 54..100} | same |

</details>

<details><summary>1 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L3 v c126 (kv1) | 0 / 48% | off (on 0) | **cmp(a,b)** (R2 0.95): cmp(a,b)=-1: 0.97, cmp(a,b)=0: 0.01, cmp(a,b)=1: 0.00 |

</details>

<details><summary>1 q components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L3 q c27 (H4) | 100% / 0 | always | off (on 0) |

</details>

<details><summary>4 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L3 k c80 (kv1) | 0 / 8% | off (on 0) | **a** (R2 0.90): a in {20, 41..42, 61..62, 70, 81} |
| L3 k c114 (kv1) | 1% / 0 | **a** (R2 1.00): a in {55} | off (on 0) |
| L3 k c133 (kv1) | 2% / 0 | **a** (R2 1.00): a in {50, 55} | off (on 0) |
| L3 k c135 (kv1) | 9% / 1% | **a** (R2 1.00): a in {21, 50..52, 55..57, 62, 65} | **a** (R2 0.96): a in {55} |

</details>

</details>

<details><summary>L3H5 (3 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.95 / 0.95 | 0.05 / 0.05 |  |  |  |
| op | 0.90 / 0.91 | 0.07 / 0.07 | 0.03 / 0.01 |  |  |
| b | 0.92 / 0.94 | 0.03 / 0.02 | 0.02 / 0.01 | 0.04 / 0.02 |  |
| = | 0.88 / 0.86 | 0.03 / 0.03 | 0.03 / 0.02 | 0.06 / 0.07 | 0.01 / 0.02 |

<details><summary>2 o components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L3 o c128 (H5) | 100% / 100% | always | same | - | same |
| L3 o c354 (H5) | 100% / 100% | always | same | - | same |

</details>

<details><summary>1 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L3 o c221 (H5) | 8% / 9% | **a** (R2 1.00): a in {1..8} | **a//10** (R2 1.00): (tens) a in {1..9} | - | same |

</details>

<details><summary>34 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L3 v c12 (kv1) | 68% / 68% | **a//10** (R2 0.93): (tens) a in {32..99} | same |
| L3 v c13 (kv1) | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {1, 11} [coarser: a mod 10 in {1}, R2 0.89] | **a%10** (R2 0.87): a mod 10 in {1} |
| L3 v c26 (kv1) | 20% / 20% | **a//10** (R2 0.94): (tens) a in {1..20} | same |
| L3 v c28 (kv1) | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same |
| L3 v c48 (kv1) | 21% / 23% | **a//10** (R2 1.00): (tens) a in {80..100} | **a//10** (R2 0.91): (tens) a in {78..100} |
| L3 v c54 (kv1) | 18% / 18% | **a** (R2 1.00): a in {50..58, 60..68} | same |
| L3 v c59 (kv1) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {70} |
| L3 v c70 (kv1) | 100% / 100% | always | same |
| L3 v c75 (kv1) | 1% / 2% | **a** (R2 1.00): a in {55} | **a** (R2 0.99): a in {55, 77} |
| L3 v c90 (kv1) | 11% / 11% | **a%50** (R2 0.95): a mod 50 in {0, 10, 20, 25, 30, 40} | same |
| L3 v c101 (kv1) | 8% / 9% | **a** (R2 0.98): a in {13..19, 21} | **a** (R2 1.00): a in {13..21} |
| L3 v c112 (kv1) | 6% / 10% | **a** (R2 1.00): a in {1..4, 11, 13} | **a** (R2 0.98): a in {1..4, 10..15} |
| L3 v c131 (kv1) | 1% / 7% | **a** (R2 1.00): a in {62} | **a** (R2 0.96): a in {58..63, 71} |
| L3 v c133 (kv1) | 1% / 3% | **a** (R2 1.00): a in {33} | **a** (R2 0.98): a in {32..34} |
| L3 v c149 (kv1) | 1% / 3% | **a** (R2 0.97): a in {74} | **a** (R2 0.99): a in {70, 73..74} |
| L3 v c151 (kv1) | 1% / 0 | **a** (R2 1.00): a in {55} | off (on 0) |
| L3 v c159 (kv1) | 2% / 4% | **a** (R2 0.97): a in {52..53} | **a** (R2 0.97): a in {51..54} |
| L3 v c160 (kv1) | 4% / 5% | **a** (R2 0.99): a in {41..44} | **a** (R2 1.00): a in {41..45} |
| L3 v c176 (kv1) | 1% / 0 | **a** (R2 1.00): a in {55} | off (on 0) |
| L3 v c179 (kv1) | 1% / 2% | **a** (R2 0.98): a in {51} | **a** (R2 1.00): a in {51, 53} |
| L3 v c180 (kv1) | 3% / 12% | **a** (R2 0.99): a in {11, 51, 53} | **a** (R2 0.99): a in {11, 31, 41, 43, 46..47, 49, 51..53, 61, 81} |
| L3 v c187 (kv1) | 1% / 2% | **a** (R2 1.00): a in {24} | **a** (R2 0.75): a in {24, 48} |
| L3 v c191 (kv1) | 2% / 2% | **a** (R2 1.00): a in {86, 90} | same |
| L3 v c195 (kv1) | 13% / 17% | **a** (R2 0.98): a in {86, 88..99} | **a** (R2 0.95): a in {10..13, 86..99} |
| L3 v c205 (kv1) | 1% / 2% | **a** (R2 0.99): a in {55} | **a** (R2 0.99): a in {43..44} |
| L3 v c207 (kv1) | 6% / 12% | **a** (R2 0.99): a in {95..100} | **a** (R2 0.90): a in {10..13, 15, 93..100} |
| L3 v c211 (kv1) | 11% / 17% | **a** (R2 0.99): a in {16..24, 26..27} | **a** (R2 0.99): a in {13..29} |
| L3 v c222 (kv1) | 1% / 1% | **a** (R2 1.00): a in {52} | same |
| L3 v c225 (kv1) | 2% / 6% | **a** (R2 0.99): a in {48, 50} | **a** (R2 1.00): a in {45..50} |
| L3 v c228 (kv1) | 17% / 17% | **a** (R2 0.98): a in {15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100} [coarser: a mod 5 in {0}, R2 0.81] | **a** (R2 0.93): a in {20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100} [coarser: a mod 5 in {0}, R2 0.81] |
| L3 v c230 (kv1) | 0 / 12% | off (on 0) | **a** (R2 1.00): a in {63, 67..74, 77..79} |
| L3 v c232 (kv1) | 1% / 1% | **a** (R2 1.00): a in {75} | same |
| L3 v c237 (kv1) | 8% / 8% | **a** (R2 1.00): a in {1..8} | same |
| L3 v c247 (kv1) | 48% / 48% | **a//10** (R2 0.92): (tens) a in {51, 54..100} | same |

</details>

<details><summary>1 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L3 v c126 (kv1) | 0 / 48% | off (on 0) | **cmp(a,b)** (R2 0.95): cmp(a,b)=-1: 0.97, cmp(a,b)=0: 0.01, cmp(a,b)=1: 0.00 |

</details>

<details><summary>1 q components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L3 q c109 (H5) | 0 / 63% | off (on 0) | **tens(a,b)** (R2 0.85): a//10 in {1,5} -> b//10 in {5,6,7,8,9,10}; a//10 in {2,3,4} -> b//10 in {6,7,8,9,10}; a//10 in {6} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {7,8,9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} |

</details>

<details><summary>4 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L3 k c80 (kv1) | 0 / 8% | off (on 0) | **a** (R2 0.90): a in {20, 41..42, 61..62, 70, 81} |
| L3 k c114 (kv1) | 1% / 0 | **a** (R2 1.00): a in {55} | off (on 0) |
| L3 k c133 (kv1) | 2% / 0 | **a** (R2 1.00): a in {50, 55} | off (on 0) |
| L3 k c135 (kv1) | 9% / 1% | **a** (R2 1.00): a in {21, 50..52, 55..57, 62, 65} | **a** (R2 0.96): a in {55} |

</details>

</details>

<details><summary>L3H7 (16 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.99 / 0.99 | 0.01 / 0.01 |  |  |  |
| op | 0.94 / 0.81 | 0.04 / 0.18 | 0.02 / 0.01 |  |  |
| b | 0.85 / 0.84 | 0.10 / 0.11 | 0.04 / 0.04 | 0.01 / 0.01 |  |
| = | 0.91 / 0.86 | 0.04 / 0.05 | 0.02 / 0.04 | 0.02 / 0.02 | 0.01 / 0.02 |

<details><summary>12 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L3 o c11 (H7) | 0 / 16% | off (on 0) | **a** (R2 1.00): a in {41..49, 51..57} | - | same |
| L3 o c27 (H7) | 1% / 10% | **a** (R2 1.00): a in {1} | **a** (R2 0.97): a in {1, 31, 41, 47, 49, 51, 53, 61, 81, 91} | - | same |
| L3 o c34 (H7) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {35, 41..45} | - | same |
| L3 o c35 (H7) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {41..45} | - | same |
| L3 o c49 (H7) | 65% / 74% | **a** (R2 1.00): a in {21..79, 81, 83..84, 87..89} | **a** (R2 1.00): a in {21..94} | - | a: mod100 +7%, mod10 +2% |
| L3 o c75 (H7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {70} | - | same |
| L3 o c77 (H7) | 0 / 1% | off (on 0) | **a//10** (R2 1.00): (tens) a in {100} | - | same |
| L3 o c104 (H7) | 4% / 2% | **a** (R2 0.99): a in {50, 98..100} | **a%50** (R2 0.98): a mod 50 in {0} | - | same |
| L3 o c174 (H7) | 0 / 1% | off (on 0) | **a//10** (R2 1.00): (tens) a in {100} | - | same |
| L3 o c190 (H7) | 0 / 16% | off (on 0) | **a** (R2 0.98): a in {41, 43, 47, 49, 51..55, 57..63} | - | same |
| L3 o c257 (H7) | 0 / 1% | off (on 0) | **a//10** (R2 1.00): (tens) a in {100} | - | same |
| L3 o c469 (H7) | 0 / 2% | off (on 0) | **a** (R2 0.98): a in {24, 100} | - | same |

</details>

<details><summary>4 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L3 o c43 (H7) | 0 / 0 | off (on 0) | same | - | same |
| L3 o c51 (H7) | 0 / 3% | off (on 0) | unexplained (best R2 0.21) | - | same |
| L3 o c98 (H7) | 0 / 0 | off (on 0) | same | - | same |
| L3 o c157 (H7) | 0 / 1% | off (on 0) | unexplained (best R2 0.10) | - | same |

</details>

<details><summary>34 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L3 v c12 (kv1) | 68% / 68% | **a//10** (R2 0.93): (tens) a in {32..99} | same |
| L3 v c13 (kv1) | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {1, 11} [coarser: a mod 10 in {1}, R2 0.89] | **a%10** (R2 0.87): a mod 10 in {1} |
| L3 v c26 (kv1) | 20% / 20% | **a//10** (R2 0.94): (tens) a in {1..20} | same |
| L3 v c28 (kv1) | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same |
| L3 v c48 (kv1) | 21% / 23% | **a//10** (R2 1.00): (tens) a in {80..100} | **a//10** (R2 0.91): (tens) a in {78..100} |
| L3 v c54 (kv1) | 18% / 18% | **a** (R2 1.00): a in {50..58, 60..68} | same |
| L3 v c59 (kv1) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {70} |
| L3 v c70 (kv1) | 100% / 100% | always | same |
| L3 v c75 (kv1) | 1% / 2% | **a** (R2 1.00): a in {55} | **a** (R2 0.99): a in {55, 77} |
| L3 v c90 (kv1) | 11% / 11% | **a%50** (R2 0.95): a mod 50 in {0, 10, 20, 25, 30, 40} | same |
| L3 v c101 (kv1) | 8% / 9% | **a** (R2 0.98): a in {13..19, 21} | **a** (R2 1.00): a in {13..21} |
| L3 v c112 (kv1) | 6% / 10% | **a** (R2 1.00): a in {1..4, 11, 13} | **a** (R2 0.98): a in {1..4, 10..15} |
| L3 v c131 (kv1) | 1% / 7% | **a** (R2 1.00): a in {62} | **a** (R2 0.96): a in {58..63, 71} |
| L3 v c133 (kv1) | 1% / 3% | **a** (R2 1.00): a in {33} | **a** (R2 0.98): a in {32..34} |
| L3 v c149 (kv1) | 1% / 3% | **a** (R2 0.97): a in {74} | **a** (R2 0.99): a in {70, 73..74} |
| L3 v c151 (kv1) | 1% / 0 | **a** (R2 1.00): a in {55} | off (on 0) |
| L3 v c159 (kv1) | 2% / 4% | **a** (R2 0.97): a in {52..53} | **a** (R2 0.97): a in {51..54} |
| L3 v c160 (kv1) | 4% / 5% | **a** (R2 0.99): a in {41..44} | **a** (R2 1.00): a in {41..45} |
| L3 v c176 (kv1) | 1% / 0 | **a** (R2 1.00): a in {55} | off (on 0) |
| L3 v c179 (kv1) | 1% / 2% | **a** (R2 0.98): a in {51} | **a** (R2 1.00): a in {51, 53} |
| L3 v c180 (kv1) | 3% / 12% | **a** (R2 0.99): a in {11, 51, 53} | **a** (R2 0.99): a in {11, 31, 41, 43, 46..47, 49, 51..53, 61, 81} |
| L3 v c187 (kv1) | 1% / 2% | **a** (R2 1.00): a in {24} | **a** (R2 0.75): a in {24, 48} |
| L3 v c191 (kv1) | 2% / 2% | **a** (R2 1.00): a in {86, 90} | same |
| L3 v c195 (kv1) | 13% / 17% | **a** (R2 0.98): a in {86, 88..99} | **a** (R2 0.95): a in {10..13, 86..99} |
| L3 v c205 (kv1) | 1% / 2% | **a** (R2 0.99): a in {55} | **a** (R2 0.99): a in {43..44} |
| L3 v c207 (kv1) | 6% / 12% | **a** (R2 0.99): a in {95..100} | **a** (R2 0.90): a in {10..13, 15, 93..100} |
| L3 v c211 (kv1) | 11% / 17% | **a** (R2 0.99): a in {16..24, 26..27} | **a** (R2 0.99): a in {13..29} |
| L3 v c222 (kv1) | 1% / 1% | **a** (R2 1.00): a in {52} | same |
| L3 v c225 (kv1) | 2% / 6% | **a** (R2 0.99): a in {48, 50} | **a** (R2 1.00): a in {45..50} |
| L3 v c228 (kv1) | 17% / 17% | **a** (R2 0.98): a in {15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100} [coarser: a mod 5 in {0}, R2 0.81] | **a** (R2 0.93): a in {20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100} [coarser: a mod 5 in {0}, R2 0.81] |
| L3 v c230 (kv1) | 0 / 12% | off (on 0) | **a** (R2 1.00): a in {63, 67..74, 77..79} |
| L3 v c232 (kv1) | 1% / 1% | **a** (R2 1.00): a in {75} | same |
| L3 v c237 (kv1) | 8% / 8% | **a** (R2 1.00): a in {1..8} | same |
| L3 v c247 (kv1) | 48% / 48% | **a//10** (R2 0.92): (tens) a in {51, 54..100} | same |

</details>

<details><summary>1 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L3 v c126 (kv1) | 0 / 48% | off (on 0) | **cmp(a,b)** (R2 0.95): cmp(a,b)=-1: 0.97, cmp(a,b)=0: 0.01, cmp(a,b)=1: 0.00 |

</details>

<details><summary>4 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L3 k c80 (kv1) | 0 / 8% | off (on 0) | **a** (R2 0.90): a in {20, 41..42, 61..62, 70, 81} |
| L3 k c114 (kv1) | 1% / 0 | **a** (R2 1.00): a in {55} | off (on 0) |
| L3 k c133 (kv1) | 2% / 0 | **a** (R2 1.00): a in {50, 55} | off (on 0) |
| L3 k c135 (kv1) | 9% / 1% | **a** (R2 1.00): a in {21, 50..52, 55..57, 62, 65} | **a** (R2 0.96): a in {55} |

</details>

</details>

<details><summary>L3H15 (34 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.98 / 0.98 | 0.02 / 0.02 |  |  |  |
| op | 0.69 / 0.63 | 0.26 / 0.35 | 0.05 / 0.02 |  |  |
| b | 0.81 / 0.79 | 0.13 / 0.17 | 0.04 / 0.02 | 0.03 / 0.02 |  |
| = | 0.77 / 0.82 | 0.02 / 0.04 | 0.04 / 0.03 | 0.08 / 0.07 | 0.09 / 0.04 |

<details><summary>32 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L3 o c74 (H15) | 12% / 15% | **a** (R2 1.00): a in {68..79} | **a** (R2 1.00): a in {63, 66..79} | - | same |
| L3 o c93 (H15) | 0 / 7% | off (on 0) | **a** (R2 1.00): a in {41, 43, 46..47, 49, 51, 53} | - | same |
| L3 o c101 (H15) | 28% / 11% | **a** (R2 1.00): a in {20, 25, 30, 35, 40, 50, 60, 65, 70..85, 87..88, 90, 100} | **a** (R2 1.00): a in {60, 70, 75, 80..85, 90, 100} | - | same |
| L3 o c109 (H15) | 12% / 0 | **a** (R2 0.99): a in {51..58, 61..62, 64, 66} | off (on 0) | - | same |
| L3 o c119 (H15) | 1% / 0 | **a** (R2 1.00): a in {55} | off (on 0) | - | same |
| L3 o c151 (H15) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {47, 49, 51, 53} | - | same |
| L3 o c161 (H15) | 0 / 13% | off (on 0) | **a** (R2 1.00): a in {54, 56, 62..64, 66..70, 72, 74, 77} | - | same |
| L3 o c170 (H15) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {61..62, 74} | - | same |
| L3 o c183 (H15) | 2% / 0 | **a** (R2 1.00): a in {21, 91} | off (on 0) | - | same |
| L3 o c194 (H15) | 4% / 0 | **a** (R2 1.00): a in {95, 97..99} | off (on 0) | - | same |
| L3 o c205 (H15) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {70, 74, 77} | - | same |
| L3 o c235 (H15) | 2% / 0 | **a** (R2 1.00): a in {98..99} | off (on 0) | - | same |
| L3 o c243 (H15) | 21% / 23% | **a//10** (R2 1.00): (tens) a in {80..100} | **a//10** (R2 0.91): (tens) a in {78..100} | - | a: mod50 +3% |
| L3 o c246 (H15) | 3% / 0 | **a** (R2 1.00): a in {18..19, 21} | off (on 0) | - | same |
| L3 o c247 (H15) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {65, 75} | - | same |
| L3 o c261 (H15) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {70} | - | same |
| L3 o c307 (H15) | 1% / 0 | **a** (R2 1.00): a in {86} | off (on 0) | - | same |
| L3 o c324 (H15) | 2% / 15% | **a** (R2 1.00): a in {11, 51} | **a** (R2 1.00): a in {7..13, 31, 41, 49, 51, 53, 61, 81, 91} | - | same |
| L3 o c327 (H15) | 4% / 3% | **a** (R2 1.00): a in {18, 55, 57, 65} | **a** (R2 1.00): a in {55, 65, 75} | - | same |
| L3 o c350 (H15) | 1% / 0 | **a** (R2 1.00): a in {21} | off (on 0) | - | same |
| L3 o c368 (H15) | 6% / 18% | **a** (R2 1.00): a in {14, 16..19, 21} | **a** (R2 1.00): a in {13..29, 31} | - | same |
| L3 o c384 (H15) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {51, 53} | - | same |
| L3 o c393 (H15) | 4% / 8% | **a** (R2 1.00): a in {17..19, 21} | **a** (R2 0.93): a in {17..23} | - | same |
| L3 o c399 (H15) | 3% / 0 | **a** (R2 1.00): a in {18, 21, 55} | off (on 0) | - | same |
| L3 o c422 (H15) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {70} | - | same |
| L3 o c427 (H15) | 28% / 23% | **a** (R2 1.00): a in {15, 20, 25, 30, 35, 40, 45, 50..52, 54..58, 60..66, 70, 75, 80, 85, 90, 100} | **a** (R2 1.00): a in {20, 25, 30, 35, 40, 45, 50, 52, 54..57, 60, 63..66, 70, 75, 80, 85, 90, 100} | a: mod5 +3% | a: mod5 +10% |
| L3 o c431 (H15) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {70} | - | same |
| L3 o c432 (H15) | 7% / 15% | **a** (R2 0.99): a in {93..99} | **a** (R2 0.98): a in {10..16, 92..99} | - | same |
| L3 o c441 (H15) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {67, 70} | - | same |
| L3 o c453 (H15) | 8% / 1% | **a** (R2 1.00): a in {21..24, 26..29} | **a** (R2 1.00): a in {24} | - | same |
| L3 o c464 (H15) | 37% / 43% | **a** (R2 1.00): a in {51, 54..69, 71..89, 91} | **a** (R2 1.00): a in {54..69, 71..97} | - | a: mod100 +4% |
| L3 o c489 (H15) | 2% / 0 | **a** (R2 1.00): a in {98..99} | off (on 0) | - | same |

</details>

<details><summary>2 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L3 o c153 (H15) | 4% / 0 | unexplained (best R2 0.29) | off (on 0) | - | same |
| L3 o c383 (H15) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>4 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L3 v c27 (kv3) | 1% / 1% | **a** (R2 1.00): a in {21} | same |
| L3 v c63 (kv3) | 1% / 1% | **a** (R2 1.00): a in {77} | same |
| L3 v c83 (kv3) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {70} |
| L3 v c198 (kv3) | 1% / 1% | **a** (R2 1.00): a in {67} | same |

</details>

<details><summary>1 q components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L3 q c0 (H15) | 100% / 100% | always | same |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L3 k c89 (kv3) | 3% / 0 | **a** (R2 1.00): a in {11, 21, 55} | off (on 0) |

</details>

</details>

<details><summary>L3H18 (3 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.97 / 0.97 | 0.03 / 0.03 |  |  |  |
| op | 0.91 / 0.88 | 0.08 / 0.11 | 0.01 / 0.02 |  |  |
| b | 0.77 / 0.74 | 0.03 / 0.06 | 0.18 / 0.17 | 0.02 / 0.03 |  |
| = | 0.87 / 0.81 | 0.00 / 0.00 | 0.02 / 0.04 | 0.08 / 0.11 | 0.03 / 0.04 |

<details><summary>3 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L3 o c66 (H18) | 0 / 99% | off (on 0) | always | - | same |
| L3 o c105 (H18) | 48% / 1% | unexplained (best R2 0.49) | unexplained (best R2 0.30) | - | same |
| L3 o c353 (H18) | 12% / 0 | unexplained (best R2 0.38) | off (on 0) | - | same |

</details>

<details><summary>1 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L3 v c210 (kv4) | 0 / 51% | off (on 0) | **cmp(a,b)** (R2 0.99): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 1.00, cmp(a,b)=1: 1.00 |

</details>

</details>

<details><summary>L3H19 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.92 / 0.92 | 0.08 / 0.08 |  |  |  |
| op | 0.92 / 0.84 | 0.01 / 0.05 | 0.07 / 0.12 |  |  |
| b | 0.93 / 0.87 | 0.01 / 0.04 | 0.04 / 0.04 | 0.03 / 0.05 |  |
| = | 0.91 / 0.85 | 0.00 / 0.01 | 0.02 / 0.04 | 0.02 / 0.03 | 0.05 / 0.07 |

<details><summary>1 o components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L3 o c85 (H19) | 2% / 2% | **a** (R2 0.99): a in {52, 74} | **a** (R2 0.96): a in {52, 74} | - | same |

</details>

<details><summary>1 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L3 v c210 (kv4) | 0 / 51% | off (on 0) | **cmp(a,b)** (R2 0.99): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 1.00, cmp(a,b)=1: 1.00 |

</details>

<details><summary>1 q components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L3 q c1 (H19) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L3H27 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.97 / 0.97 | 0.03 / 0.03 |  |  |  |
| op | 0.98 / 0.99 | 0.01 / 0.00 | 0.01 / 0.01 |  |  |
| b | 0.87 / 0.91 | 0.03 / 0.03 | 0.02 / 0.00 | 0.08 / 0.06 |  |
| = | 0.85 / 0.93 | 0.01 / 0.01 | 0.08 / 0.02 | 0.02 / 0.01 | 0.03 / 0.03 |

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L3 o c154 (H27) | 100% / 0 | always | off (on 0) | - | same |

</details>

</details>

<details><summary>L4H3 (5 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.98 / 0.98 | 0.02 / 0.02 |  |  |  |
| op | 0.92 / 0.94 | 0.04 / 0.02 | 0.04 / 0.04 |  |  |
| b | 0.64 / 0.88 | 0.05 / 0.04 | 0.26 / 0.06 | 0.04 / 0.02 |  |
| = | 0.88 / 0.89 | 0.02 / 0.02 | 0.04 / 0.04 | 0.04 / 0.02 | 0.02 / 0.03 |

<details><summary>5 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L4 o c34 (H3) | 6% / 0 | **tens(a,b)** (R2 0.52): a//10 in {3,4,7,8,9,10} -> b//10 in {0} | off (on 0) | - | same |
| L4 o c37 (H3) | 10% / 0 | **tens(a,b)** (R2 0.58): a//10 in {9} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {10} -> b//10 in {4,6} | off (on 0) | - | same |
| L4 o c59 (H3) | 10% / 0 | **a** (R2 0.58): a in {89..98} | off (on 0) | - | same |
| L4 o c77 (H3) | 93% / 0 | unexplained (best R2 0.34) | off (on 0) | - | same |
| L4 o c233 (H3) | 1% / 0 | unexplained (best R2 0.49) | off (on 0) | - | same |

</details>

<details><summary>2 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L4 v c219 (kv0) | 0 / 100% | off (on 0) | always |
| L4 v c243 (kv0) | 100% / 0 | always | off (on 0) |

</details>

<details><summary>2 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L4 v c9 (kv0) | 14% / 0 | **b** (R2 0.89): b in {1..13} | off (on 0) |
| L4 v c127 (kv0) | 100% / 100% | always | same |

</details>

<details><summary>1 v components, main position `=` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L4 v c69 (kv0) | 100% / 100% | always | same |

</details>

<details><summary>1 q components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L4 q c3 (H3) | 100% / 100% | always | same |

</details>

<details><summary>3 k components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L4 k c20 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L4 k c50 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L4 k c72 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L4 k c68 (kv0) | 100% / 100% | always | same |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L4 k c100 (kv0) | 0 / 100% | off (on 0) | always |

</details>

</details>

<details><summary>L4H16 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.91 / 0.91 | 0.09 / 0.09 |  |  |  |
| op | 0.96 / 0.88 | 0.00 / 0.01 | 0.04 / 0.11 |  |  |
| b | 0.92 / 0.90 | 0.02 / 0.02 | 0.00 / 0.00 | 0.05 / 0.08 |  |
| = | 0.95 / 0.91 | 0.00 / 0.00 | 0.01 / 0.00 | 0.00 / 0.00 | 0.03 / 0.09 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L4 o c174 (H16) | 2% / 2% | **cmp(a,b)** (R2 0.53): cmp(a,b)=-1: 0.01, cmp(a,b)=0: 1.00, cmp(a,b)=1: 0.01 | **cmp(a,b)** (R2 0.62): cmp(a,b)=-1: 0.01, cmp(a,b)=0: 0.98, cmp(a,b)=1: 0.00 | - | same |

</details>

<details><summary>1 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L4 v c0 (kv4) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

</details>

<details><summary>L4H18 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 1.00 / 1.00 | 0.00 / 0.00 |  |  |  |
| op | 0.99 / 1.00 | 0.00 / 0.00 | 0.01 / 0.00 |  |  |
| b | 0.99 / 0.99 | 0.00 / 0.00 | 0.01 / 0.00 | 0.00 / 0.01 |  |
| = | 0.86 / 0.93 | 0.01 / 0.01 | 0.04 / 0.03 | 0.04 / 0.02 | 0.05 / 0.02 |

<details><summary>1 o components, main position `&lt;BOS&gt;`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L4 o c7 (H18) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same | - | same |

</details>

<details><summary>1 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L4 v c0 (kv4) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>1 q components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L4 q c1 (H18) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L4H24 (2 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.97 / 0.97 | 0.03 / 0.03 |  |  |  |
| op | 0.85 / 0.91 | 0.10 / 0.08 | 0.05 / 0.01 |  |  |
| b | 0.83 / 0.80 | 0.02 / 0.06 | 0.10 / 0.13 | 0.05 / 0.02 |  |
| = | 0.70 / 0.84 | 0.02 / 0.01 | 0.06 / 0.03 | 0.12 / 0.05 | 0.09 / 0.06 |

<details><summary>1 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L4 o c113 (H24) | 100% / 0 | always | off (on 0) | - | same |

</details>

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L4 o c14 (H24) | 100% / 0 | always | off (on 0) | - | same |

</details>

<details><summary>1 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L4 v c6 (kv6) | 100% / 100% | always | same |

</details>

<details><summary>1 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L4 v c22 (kv6) | 0 / 62% | off (on 0) | **res//10** (R2 0.81): (tens) res in {-99..12} |

</details>

</details>

<details><summary>L4H25 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.92 / 0.92 | 0.08 / 0.08 |  |  |  |
| op | 0.96 / 0.95 | 0.03 / 0.03 | 0.01 / 0.02 |  |  |
| b | 0.92 / 0.93 | 0.02 / 0.03 | 0.04 / 0.02 | 0.02 / 0.02 |  |
| = | 0.61 / 0.75 | 0.03 / 0.03 | 0.12 / 0.07 | 0.18 / 0.07 | 0.06 / 0.08 |

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L4 o c23 (H25) | 100% / 100% | always | same | - | same |

</details>

<details><summary>1 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L4 v c6 (kv6) | 100% / 100% | always | same |

</details>

<details><summary>1 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L4 v c22 (kv6) | 0 / 62% | off (on 0) | **res//10** (R2 0.81): (tens) res in {-99..12} |

</details>

</details>

<details><summary>L4H26 (2 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.85 / 0.85 | 0.15 / 0.15 |  |  |  |
| op | 0.90 / 0.80 | 0.02 / 0.02 | 0.08 / 0.18 |  |  |
| b | 0.88 / 0.85 | 0.01 / 0.01 | 0.03 / 0.02 | 0.08 / 0.13 |  |
| = | 0.91 / 0.87 | 0.01 / 0.00 | 0.01 / 0.01 | 0.01 / 0.02 | 0.05 / 0.10 |

<details><summary>1 o components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L4 o c13 (H26) | 31% / 31% | **a** (R2 1.00): a in {21, 31, 48..49, 52, 69, 71..93, 96, 99} | **a** (R2 0.99): a in {21, 31, 48..49, 52, 69, 71..93, 96, 99} | - | same |

</details>

<details><summary>1 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L4 o c239 (H26) | 0 / 98% | off (on 0) | always | - | same |

</details>

<details><summary>1 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L4 v c6 (kv6) | 100% / 100% | always | same |

</details>

<details><summary>1 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L4 v c22 (kv6) | 0 / 62% | off (on 0) | **res//10** (R2 0.81): (tens) res in {-99..12} |

</details>

</details>

<details><summary>L5H14 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.98 / 0.98 | 0.02 / 0.02 |  |  |  |
| op | 0.85 / 0.94 | 0.04 / 0.04 | 0.11 / 0.02 |  |  |
| b | 0.58 / 0.76 | 0.02 / 0.04 | 0.32 / 0.15 | 0.08 / 0.05 |  |
| = | 0.61 / 0.69 | 0.02 / 0.01 | 0.12 / 0.05 | 0.20 / 0.16 | 0.05 / 0.08 |

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L5 o c423 (H14) | 100% / 0 | always | off (on 0) | - | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L5 v c131 (kv3) | 0 / 100% | off (on 0) | always |

</details>

<details><summary>4 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L5 v c3 (kv3) | 100% / 1% | always | **cmp(a,b)** (R2 0.73): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.87, cmp(a,b)=1: 0.00 |
| L5 v c73 (kv3) | 0 / 50% | off (on 0) | **cmp(a,b)** (R2 0.99): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.91, cmp(a,b)=1: 1.00 |
| L5 v c88 (kv3) | 5% / 42% | unexplained (best R2 0.39) | **res//10** (R2 0.69): (tens) res in {-98..-7} |
| L5 v c180 (kv3) | 0 / 36% | off (on 0) | **tens(a,b)** (R2 0.77): a//10 in {0} -> b//10 in {1}; a//10 in {1} -> b//10 in {2,3}; a//10 in {2} -> b//10 in {2,3,4,5,6,7,8,9}; a//10 in {3} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {4} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {5,6,7,8,9,10}; a//10 in {6} -> b//10 in {6,7,8,9,10}; a//10 in {7} -> b//10 in {8,9,10}; a//10 in {8,9} -> b//10 in {9,10}; a//10 in {10} -> b//10 in {9} |

</details>

</details>

<details><summary>L5H22 (30 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.96 / 0.96 | 0.04 / 0.04 |  |  |  |
| op | 0.96 / 0.91 | 0.01 / 0.01 | 0.03 / 0.08 |  |  |
| b | 0.44 / 0.52 | 0.41 / 0.34 | 0.14 / 0.10 | 0.02 / 0.04 |  |
| = | 0.68 / 0.81 | 0.05 / 0.05 | 0.21 / 0.07 | 0.04 / 0.01 | 0.02 / 0.06 |

<details><summary>30 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L5 o c16 (H22) | 99% / 36% | always | **tens(a,b)** (R2 0.76): a//10 in {4} -> b//10 in {4,5,6,7}; a//10 in {5,7,10} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {8} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10} | a: mod100 +6% | a: mod100 +3% |
| L5 o c17 (H22) | 24% / 14% | **a//10** (R2 0.81): (tens) a in {76, 78..100} | **tens(a,b)** (R2 0.73): a//10 in {7} -> b//10 in {6,7,8,9}; a//10 in {8} -> b//10 in {5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10}; a//10 in {10} -> b//10 in {0,3,4,5,6,7,8,9,10} | a: mod100 +4%, mod50 +6%, mod20 +3% | - |
| L5 o c27 (H22) | 70% / 37% | **a//10** (R2 0.78): (tens) a in {1..71} | **tens(a,b)** (R2 0.74): a//10 in {1} -> b//10 in {2,3}; a//10 in {2} -> b//10 in {2,3,4,5,6,7,8,9}; a//10 in {3,4} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {7,8,9,10} | a: mod100 +5% | - |
| L5 o c31 (H22) | 19% / 17% | **a** (R2 0.70): a in {22..40, 42, 44..45} | **tens(a,b)** (R2 0.73): a//10 in {2} -> b//10 in {1,2,3,4,5,6}; a//10 in {3} -> b//10 in {1,2,3,4,5,6,7,8}; a//10 in {4} -> b//10 in {3,4,5,6} | a: mod50 +4% | a: mod50 +3% |
| L5 o c40 (H22) | 0 / 3% | off (on 0) | unexplained (best R2 0.18) | - | same |
| L5 o c42 (H22) | 9% / 23% | unexplained (best R2 0.43) | **tens(a,b)** (R2 0.74): a//10 in {3} -> b//10 in {3}; a//10 in {4} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {5} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {4,5,6,7,8,9}; a//10 in {7} -> b//10 in {7,8,9} | - | same |
| L5 o c45 (H22) | 0 / 9% | off (on 0) | **tens(a,b)** (R2 0.62): a//10 in {2} -> b//10 in {3}; a//10 in {6} -> b//10 in {7,8,9}; a//10 in {7} -> b//10 in {8,9,10}; a//10 in {8} -> b//10 in {9,10}; a//10 in {9} -> b//10 in {10} | - | same |
| L5 o c49 (H22) | 4% / 8% | unexplained (best R2 0.19) | unexplained (best R2 0.23) | - | a: mod10 +2% |
| L5 o c53 (H22) | 14% / 18% | **a** (R2 0.48): a in {39, 41..45, 47..49, 51..54, 56} | **tens(a,b)** (R2 0.67): a//10 in {3} -> b//10 in {3,4}; a//10 in {4} -> b//10 in {2,3,4,5,6,7,8,9}; a//10 in {5} -> b//10 in {3,4,5,6,7,8,9,10} | - | same |
| L5 o c59 (H22) | 27% / 10% | **a//10** (R2 0.88): (tens) a in {1..28} | **tens(a,b)** (R2 0.51): a//10 in {0} -> b//10 in {1}; a//10 in {1} -> b//10 in {1,2,3}; a//10 in {2} -> b//10 in {2,3,4,5,6} | a: mod100 +3%, mod50 +3% | - |
| L5 o c62 (H22) | 4% / 9% | unexplained (best R2 0.46) | unexplained (best R2 0.47) | - | same |
| L5 o c78 (H22) | 13% / 2% | **a//10** (R2 0.68): (tens) a in {1..13, 15} | unexplained (best R2 0.45) | - | same |
| L5 o c85 (H22) | 3% / 10% | unexplained (best R2 0.11) | unexplained (best R2 0.24) | - | a: mod10 +3% |
| L5 o c102 (H22) | 0 / 1% | off (on 0) | unexplained (best R2 0.20) | - | same |
| L5 o c111 (H22) | 4% / 2% | **tens(a,b)** (R2 0.62): a//10 in {9,10} -> b//10 in {7,8,9} | **tens(a,b)** (R2 0.63): a//10 in {9} -> b//10 in {9,10}; a//10 in {10} -> b//10 in {7,8,9,10} | - | same |
| L5 o c130 (H22) | 3% / 0 | unexplained (best R2 0.36) | off (on 0) | - | same |
| L5 o c131 (H22) | 3% / 9% | unexplained (best R2 0.25) | unexplained (best R2 0.29) | - | same |
| L5 o c136 (H22) | 1% / 5% | unexplained (best R2 0.09) | unexplained (best R2 0.30) | - | same |
| L5 o c158 (H22) | 34% / 22% | **a** (R2 0.59): a in {12, 16, 22, 24, 26, 28, 32, 34, 36, 38, 40, 42, 44, 46, 48, 52, 54, 56, 58, 60, 62, 64, 66, 68, 72, 74, 76, 78, 80, 82, 84, 86, 88, 92, 94, 96} [coarser: a mod 10 in {2, 4, 6, 8}, R2 0.80] | unexplained (best R2 0.37) | a: mod4 +7%, mod2 +26% | a: mod4 +3%, mod2 +18% |
| L5 o c159 (H22) | 0 / 1% | off (on 0) | unexplained (best R2 0.12) | - | same |
| L5 o c166 (H22) | 0 / 1% | off (on 0) | unexplained (best R2 0.14) | - | same |
| L5 o c179 (H22) | 0 / 3% | off (on 0) | unexplained (best R2 0.23) | - | same |
| L5 o c204 (H22) | 0 / 0 | off (on 0) | same | - | same |
| L5 o c288 (H22) | 1% / 2% | unexplained (best R2 0.05) | unexplained (best R2 0.17) | - | same |
| L5 o c295 (H22) | 0 / 8% | off (on 0) | unexplained (best R2 0.18) | - | same |
| L5 o c303 (H22) | 16% / 10% | **a** (R2 0.76): a in {10, 18, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 90, 100} [coarser: a mod 10 in {0}, R2 0.84] | unexplained (best R2 0.29) | a: mod10 +5%, mod5 +10%, mod2 +4% | - |
| L5 o c317 (H22) | 2% / 8% | unexplained (best R2 0.13) | unexplained (best R2 0.24) | - | same |
| L5 o c363 (H22) | 0 / 0 | off (on 0) | same | - | same |
| L5 o c372 (H22) | 11% / 0 | **a** (R2 0.80): a in {20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 90, 100} [coarser: a mod 50 in {0, 20, 30, 40}, R2 0.82] | off (on 0) | a: mod5 +3% | - |
| L5 o c387 (H22) | 4% / 21% | unexplained (best R2 0.15) | unexplained (best R2 0.35) | - | a: mod2 +6% |

</details>

<details><summary>17 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L5 v c11 (kv5) | 25% / 17% | **a//10** (R2 0.80): (tens) a in {53..54, 56..79} | unexplained (best R2 0.48) |
| L5 v c13 (kv5) | 27% / 16% | **a//10** (R2 0.87): (tens) a in {74..100} | unexplained (best R2 0.48) |
| L5 v c23 (kv5) | 24% / 22% | **a** (R2 0.88): a in {24..48} | **a//10** (R2 0.64): (tens) a in {24..49} |
| L5 v c27 (kv5) | 4% / 2% | unexplained (best R2 0.42) | unexplained (best R2 0.17) |
| L5 v c33 (kv5) | 14% / 9% | **a%10** (R2 0.77): a mod 10 in {0} | **a** (R2 0.62): a in {20, 25, 30, 40, 50, 60, 70, 80, 100} [coarser: a mod 10 in {0}, R2 0.82] |
| L5 v c39 (kv5) | 3% / 5% | unexplained (best R2 0.19) | unexplained (best R2 0.30) |
| L5 v c46 (kv5) | 18% / 18% | **a** (R2 0.77): a in {41..54, 56..59} | **a** (R2 0.56): a in {37, 39, 41..59} |
| L5 v c48 (kv5) | 24% / 15% | **a** (R2 0.91): a in {66..89} | unexplained (best R2 0.46) |
| L5 v c55 (kv5) | 18% / 12% | **a** (R2 0.90): a in {12..28} | **a//10** (R2 0.53): (tens) a in {18, 20..29} |
| L5 v c75 (kv5) | 7% / 10% | unexplained (best R2 0.40) | unexplained (best R2 0.29) |
| L5 v c109 (kv5) | 1% / 4% | unexplained (best R2 0.14) | unexplained (best R2 0.37) |
| L5 v c121 (kv5) | 0 / 0 | off (on 0) | same |
| L5 v c152 (kv5) | 12% / 15% | unexplained (best R2 0.47) | unexplained (best R2 0.20) |
| L5 v c165 (kv5) | 3% / 8% | unexplained (best R2 0.19) | unexplained (best R2 0.30) |
| L5 v c190 (kv5) | 14% / 4% | **a** (R2 0.88): a in {4..18} | unexplained (best R2 0.21) |
| L5 v c201 (kv5) | 39% / 28% | **a** (R2 0.78): a in {12, 16, 22, 24, 26, 28, 32, 34, 36, 38, 40, 42, 44, 46, 48, 52, 54, 56, 58, 60, 62, 64, 66, 68, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98} [coarser: a mod 10 in {2, 4, 6, 8}, R2 0.82] | unexplained (best R2 0.48) |
| L5 v c255 (kv5) | 1% / 1% | unexplained (best R2 0.09) | unexplained (best R2 0.11) |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L5 v c20 (kv5) | 12% / 0 | **a** (R2 0.95): a in {1..10, 12} | off (on 0) |

</details>

<details><summary>3 k components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L5 k c15 (kv5) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L5 k c25 (kv5) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L5 k c51 (kv5) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L5 k c56 (kv5) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L5H26 (2 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.92 / 0.92 | 0.08 / 0.08 |  |  |  |
| op | 0.90 / 0.93 | 0.06 / 0.05 | 0.04 / 0.02 |  |  |
| b | 0.91 / 0.91 | 0.04 / 0.05 | 0.04 / 0.02 | 0.02 / 0.02 |  |
| = | 0.87 / 0.90 | 0.05 / 0.04 | 0.04 / 0.02 | 0.02 / 0.02 | 0.02 / 0.03 |

<details><summary>1 o components, main position `&lt;BOS&gt;`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L5 o c2 (H26) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same | - | same |

</details>

<details><summary>1 o components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L5 o c6 (H26) | 100% / 100% | always | same | - | same |

</details>

<details><summary>1 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L5 v c118 (kv6) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L5 v c2 (kv6) | 100% / 0 | always | off (on 0) |

</details>

</details>

<details><summary>L5H30 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.99 / 0.99 | 0.01 / 0.01 |  |  |  |
| op | 0.99 / 0.99 | 0.01 / 0.01 | 0.00 / 0.01 |  |  |
| b | 0.85 / 0.95 | 0.06 / 0.02 | 0.03 / 0.02 | 0.06 / 0.02 |  |
| = | 0.33 / 0.40 | 0.04 / 0.06 | 0.16 / 0.20 | 0.44 / 0.26 | 0.03 / 0.07 |

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L5 o c141 (H30) | 98% / 1% | always | unexplained (best R2 0.16) | b: mod100 +2% | - |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L5 v c186 (kv7) | 0 / 21% | off (on 0) | **a//10** (R2 0.91): (tens) a in {1..21} |

</details>

<details><summary>1 k components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L5 k c34 (kv7) | 100% / 60% | always | **tens(a,b)** (R2 0.77): a//10 in {0} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,1,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,2,9,10}; a//10 in {4} -> b//10 in {0,1,2,3,10}; a//10 in {5,6,7} -> b//10 in {0,1,2,3,4}; a//10 in {8} -> b//10 in {0,1,2,3,4,5}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8,10} |

</details>

</details>

<details><summary>L5H31 (4 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.99 / 0.99 | 0.01 / 0.01 |  |  |  |
| op | 0.97 / 0.98 | 0.02 / 0.01 | 0.01 / 0.01 |  |  |
| b | 0.84 / 0.93 | 0.04 / 0.01 | 0.08 / 0.03 | 0.05 / 0.03 |  |
| = | 0.33 / 0.32 | 0.05 / 0.06 | 0.22 / 0.23 | 0.29 / 0.31 | 0.11 / 0.08 |

<details><summary>4 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L5 o c22 (H31) | 0 / 99% | off (on 0) | always | - | a: mod100 +3%, mod50 +3%, mod4 +2%; b: mod100 +2%, mod50 +3%; res: mod100 +3% |
| L5 o c75 (H31) | 0 / 49% | off (on 0) | **cmp(a,b)** (R2 0.94): cmp(a,b)=-1: 0.98, cmp(a,b)=0: 0.00, cmp(a,b)=1: 0.01 | - | a: mod100 +9%, mod50 +7%, mod25 +3%, mod20 +4%; b: mod100 +5%, mod50 +3%; res: mod100 +6%, mod50 +5%, mod25 +3%, mod20 +3% |
| L5 o c310 (H31) | 0 / 33% | off (on 0) | **res//10** (R2 0.78): (tens) res in {21..99} | - | a: mod100 +3%; b: mod100 +8%, mod50 +5%, mod25 +4%, mod20 +5%; res: mod100 +8% |
| L5 o c450 (H31) | 0 / 5% | off (on 0) | **tens(a,b)** (R2 0.62): a//10 in {0} -> b//10 in {0,1,2}; a//10 in {1} -> b//10 in {2} | - | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L5 v c186 (kv7) | 0 / 21% | off (on 0) | **a//10** (R2 0.91): (tens) a in {1..21} |

</details>

<details><summary>1 q components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L5 q c126 (H31) | 100% / 95% | always | same |

</details>

<details><summary>1 k components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L5 k c34 (kv7) | 100% / 60% | always | **tens(a,b)** (R2 0.77): a//10 in {0} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,1,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,2,9,10}; a//10 in {4} -> b//10 in {0,1,2,3,10}; a//10 in {5,6,7} -> b//10 in {0,1,2,3,4}; a//10 in {8} -> b//10 in {0,1,2,3,4,5}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8,10} |

</details>

</details>

<details><summary>L6H7 (2 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.97 / 0.97 | 0.03 / 0.03 |  |  |  |
| op | 0.92 / 0.94 | 0.06 / 0.03 | 0.02 / 0.03 |  |  |
| b | 0.77 / 0.83 | 0.06 / 0.03 | 0.08 / 0.08 | 0.08 / 0.06 |  |
| = | 0.56 / 0.52 | 0.03 / 0.03 | 0.18 / 0.10 | 0.17 / 0.23 | 0.06 / 0.13 |

<details><summary>2 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L6 o c290 (H7) | 0 / 51% | off (on 0) | **cmp(a,b)** (R2 0.98): cmp(a,b)=-1: 0.01, cmp(a,b)=0: 1.00, cmp(a,b)=1: 1.00 | - | a: mod100 +4%, mod50 +3%, mod25 +2%; b: mod100 +4%, mod50 +2%; res: mod100 +6%, mod50 +5%, mod25 +4%, mod20 +4%, mod10 +3% |
| L6 o c310 (H7) | 0 / 35% | off (on 0) | **tens(a,b)** (R2 0.79): a//10 in {0} -> b//10 in {1}; a//10 in {1} -> b//10 in {2,3}; a//10 in {2} -> b//10 in {3,4,5,6,7}; a//10 in {3} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {5,6,7,8,9,10}; a//10 in {5,6} -> b//10 in {6,7,8,9,10}; a//10 in {7} -> b//10 in {7,8,9,10}; a//10 in {8,9} -> b//10 in {9,10} | - | b: mod100 +4%, mod50 +2%; res: mod100 +10%, mod50 +9%, mod25 +6%, mod20 +6%, mod10 +4%, mod5 +3% |

</details>

<details><summary>2 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L6 v c212 (kv1) | 21% / 0 | **a//10** (R2 0.84): (tens) a in {70, 80, 82..100} | off (on 0) |
| L6 v c226 (kv1) | 0 / 15% | off (on 0) | **a** (R2 0.92): a in {1..16} |

</details>

<details><summary>2 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L6 v c173 (kv1) | 0 / 45% | off (on 0) | **cmp(a,b)** (R2 0.78): cmp(a,b)=-1: 0.01, cmp(a,b)=0: 0.95, cmp(a,b)=1: 0.88 |
| L6 v c190 (kv1) | 0 / 21% | off (on 0) | **tens(a,b)** (R2 0.86): a//10 in {0,1} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {5,6,7,8,9,10}; a//10 in {3} -> b//10 in {8,9,10}; a//10 in {4,5} -> b//10 in {10} |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L6 k c2 (kv1) | 100% / 100% | always | same |

</details>

<details><summary>1 k components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L6 k c42 (kv1) | 0 / 74% | off (on 0) | **tens(a,b)** (R2 0.79): a//10 in {0,1,2,3} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {5,6,7} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10}; a//10 in {10} -> b//10 in {3,4,5,6,7,8,9} |

</details>

</details>

<details><summary>L6H8 (2 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.98 / 0.98 | 0.02 / 0.02 |  |  |  |
| op | 0.88 / 0.81 | 0.06 / 0.14 | 0.05 / 0.06 |  |  |
| b | 0.44 / 0.59 | 0.10 / 0.10 | 0.42 / 0.29 | 0.03 / 0.02 |  |
| = | 0.74 / 0.82 | 0.04 / 0.02 | 0.15 / 0.05 | 0.06 / 0.08 | 0.02 / 0.03 |

<details><summary>2 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L6 o c23 (H8) | 100% / 0 | always | off (on 0) | - | same |
| L6 o c192 (H8) | 17% / 0 | **a** (R2 0.78): a in {55, 80, 86, 88..100} | off (on 0) | a: mod50 +2% | - |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L6 v c189 (kv2) | 1% / 0 | **a** (R2 1.00): a in {90} | off (on 0) |

</details>

</details>

<details><summary>L6H16 (4 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.98 / 0.98 | 0.02 / 0.02 |  |  |  |
| op | 0.96 / 0.89 | 0.01 / 0.05 | 0.03 / 0.06 |  |  |
| b | 0.34 / 0.49 | 0.02 / 0.02 | 0.56 / 0.46 | 0.08 / 0.04 |  |
| = | 0.62 / 0.77 | 0.01 / 0.01 | 0.12 / 0.04 | 0.07 / 0.04 | 0.19 / 0.14 |

<details><summary>4 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L6 o c24 (H16) | 0 / 97% | off (on 0) | always | - | a: mod5 +2% |
| L6 o c52 (H16) | 0 / 0 | off (on 0) | same | - | same |
| L6 o c161 (H16) | 1% / 0 | **a** (R2 0.53): a in {55} | off (on 0) | - | same |
| L6 o c436 (H16) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>3 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L6 v c132 (kv4) | 4% / 0 | **a** (R2 1.00): a in {50, 55, 60, 65} | off (on 0) |
| L6 v c169 (kv4) | 1% / 0 | **a** (R2 1.00): a in {18} | off (on 0) |
| L6 v c231 (kv4) | 4% / 0 | **a** (R2 0.93): a in {70, 80, 90, 100} | off (on 0) |

</details>

<details><summary>1 q components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L6 q c125 (H16) | 100% / 96% | always | same |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L6 k c137 (kv4) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L6H22 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.90 / 0.90 | 0.10 / 0.10 |  |  |  |
| op | 0.91 / 0.91 | 0.03 / 0.05 | 0.05 / 0.03 |  |  |
| b | 0.91 / 0.87 | 0.04 / 0.05 | 0.03 / 0.04 | 0.03 / 0.05 |  |
| = | 0.76 / 0.83 | 0.04 / 0.03 | 0.04 / 0.01 | 0.08 / 0.03 | 0.08 / 0.10 |

<details><summary>1 o components, main position `&lt;BOS&gt;`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L6 o c4 (H22) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same | - | same |

</details>

<details><summary>1 k components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L6 k c136 (kv5) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

</details>

<details><summary>L6H24 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.94 / 0.94 | 0.06 / 0.06 |  |  |  |
| op | 0.92 / 0.96 | 0.03 / 0.01 | 0.05 / 0.03 |  |  |
| b | 0.92 / 0.91 | 0.03 / 0.03 | 0.05 / 0.04 | 0.01 / 0.03 |  |
| = | 0.88 / 0.93 | 0.03 / 0.01 | 0.06 / 0.03 | 0.01 / 0.02 | 0.02 / 0.01 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L6 o c85 (H24) | 100% / 100% | always | same | - | same |

</details>

<details><summary>1 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L6 v c1 (kv6) | 100% / 100% | always | same |

</details>

<details><summary>2 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L6 v c45 (kv6) | 0 / 97% | off (on 0) | always |
| L6 v c228 (kv6) | 100% / 0 | always | off (on 0) |

</details>

<details><summary>1 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L6 v c221 (kv6) | 0 / 0 | off (on 0) | same |

</details>

<details><summary>1 v components, main position `=` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L6 v c150 (kv6) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L7H0 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.81 / 0.81 | 0.19 / 0.19 |  |  |  |
| op | 0.64 / 0.73 | 0.06 / 0.03 | 0.30 / 0.24 |  |  |
| b | 0.86 / 0.78 | 0.03 / 0.03 | 0.05 / 0.05 | 0.06 / 0.14 |  |
| = | 0.80 / 0.72 | 0.01 / 0.01 | 0.02 / 0.02 | 0.04 / 0.03 | 0.13 / 0.22 |

<details><summary>1 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L7 o c455 (H0) | 97% / 0 | always | off (on 0) | - | same |

</details>

<details><summary>2 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L7 v c30 (kv0) | 4% / 4% | **a** (R2 1.00): a in {1..4} | same |
| L7 v c230 (kv0) | 100% / 100% | always | same |

</details>

<details><summary>1 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L7 v c215 (kv0) | 0 / 100% | off (on 0) | always |

</details>

<details><summary>1 k components, main position `=` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L7 k c140 (kv0) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L7H5 (2 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.93 / 0.93 | 0.07 / 0.07 |  |  |  |
| op | 0.71 / 0.73 | 0.18 / 0.19 | 0.11 / 0.08 |  |  |
| b | 0.56 / 0.59 | 0.03 / 0.04 | 0.32 / 0.26 | 0.09 / 0.11 |  |
| = | 0.65 / 0.73 | 0.00 / 0.00 | 0.03 / 0.02 | 0.20 / 0.14 | 0.12 / 0.12 |

<details><summary>1 o components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L7 o c3 (H5) | 100% / 100% | always | same | - | same |

</details>

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L7 o c96 (H5) | 0 / 19% | off (on 0) | **tens(a,b)** (R2 0.70): a//10 in {1} -> b//10 in {4,5,6,7,8,9}; a//10 in {8} -> b//10 in {2,3,4,5}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,8} | - | a: mod25 +2%, mod20 +4% |

</details>

<details><summary>5 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L7 v c0 (kv1) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L7 v c1 (kv1) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L7 v c3 (kv1) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L7 v c5 (kv1) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L7 v c11 (kv1) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>5 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L7 v c78 (kv1) | 0 / 100% | off (on 0) | always |
| L7 v c95 (kv1) | 0 / 23% | off (on 0) | **a** (R2 0.71): a in {10..19, 81, 84..85, 87..99} |
| L7 v c111 (kv1) | 100% / 0 | always | off (on 0) |
| L7 v c179 (kv1) | 1% / 0 | **a** (R2 0.95): a in {11} | off (on 0) |
| L7 v c194 (kv1) | 1% / 0 | **a** (R2 1.00): a in {11} | off (on 0) |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L7 k c98 (kv1) | 0 / 78% | off (on 0) | unexplained (best R2 0.40) |

</details>

</details>

<details><summary>L7H7 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.97 / 0.97 | 0.03 / 0.03 |  |  |  |
| op | 0.92 / 0.88 | 0.02 / 0.02 | 0.05 / 0.10 |  |  |
| b | 0.91 / 0.88 | 0.04 / 0.04 | 0.02 / 0.03 | 0.02 / 0.05 |  |
| = | 0.91 / 0.92 | 0.03 / 0.04 | 0.03 / 0.01 | 0.01 / 0.01 | 0.02 / 0.02 |

<details><summary>1 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L7 o c276 (H7) | 0 / 22% | off (on 0) | **a** (R2 0.95): a in {73, 78..79, 81..99} | - | same |

</details>

<details><summary>5 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L7 v c0 (kv1) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L7 v c1 (kv1) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L7 v c3 (kv1) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L7 v c5 (kv1) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L7 v c11 (kv1) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>5 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L7 v c78 (kv1) | 0 / 100% | off (on 0) | always |
| L7 v c95 (kv1) | 0 / 23% | off (on 0) | **a** (R2 0.71): a in {10..19, 81, 84..85, 87..99} |
| L7 v c111 (kv1) | 100% / 0 | always | off (on 0) |
| L7 v c179 (kv1) | 1% / 0 | **a** (R2 0.95): a in {11} | off (on 0) |
| L7 v c194 (kv1) | 1% / 0 | **a** (R2 1.00): a in {11} | off (on 0) |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L7 k c98 (kv1) | 0 / 78% | off (on 0) | unexplained (best R2 0.40) |

</details>

</details>

<details><summary>L7H20 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.96 / 0.96 | 0.04 / 0.04 |  |  |  |
| op | 0.86 / 0.92 | 0.03 / 0.02 | 0.11 / 0.05 |  |  |
| b | 0.84 / 0.91 | 0.02 / 0.02 | 0.04 / 0.02 | 0.11 / 0.05 |  |
| = | 0.76 / 0.80 | 0.01 / 0.01 | 0.01 / 0.01 | 0.04 / 0.02 | 0.18 / 0.17 |

<details><summary>1 o components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L7 o c14 (H20) | 100% / 100% | always | same | - | same |

</details>

<details><summary>2 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L7 v c19 (kv5) | 0 / 0 | off (on 0) | same |
| L7 v c200 (kv5) | 99% / 0 | always | off (on 0) |

</details>

</details>

<details><summary>L7H23 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.94 / 0.94 | 0.06 / 0.06 |  |  |  |
| op | 0.81 / 0.86 | 0.04 / 0.04 | 0.15 / 0.10 |  |  |
| b | 0.91 / 0.87 | 0.04 / 0.03 | 0.01 / 0.02 | 0.04 / 0.08 |  |
| = | 0.93 / 0.93 | 0.01 / 0.01 | 0.01 / 0.01 | 0.02 / 0.01 | 0.04 / 0.04 |

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L7 o c11 (H23) | 99% / 89% | always | unexplained (best R2 0.34) | - | same |

</details>

<details><summary>2 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L7 v c19 (kv5) | 0 / 0 | off (on 0) | same |
| L7 v c200 (kv5) | 99% / 0 | always | off (on 0) |

</details>

</details>

<details><summary>L7H27 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.96 / 0.96 | 0.04 / 0.04 |  |  |  |
| op | 0.92 / 0.93 | 0.04 / 0.01 | 0.04 / 0.05 |  |  |
| b | 0.37 / 0.69 | 0.53 / 0.21 | 0.08 / 0.06 | 0.02 / 0.05 |  |
| = | 0.81 / 0.84 | 0.10 / 0.07 | 0.05 / 0.07 | 0.01 / 0.01 | 0.03 / 0.02 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L7 o c40 (H27) | 99% / 13% | always | **tens(a,b)** (R2 0.51): a//10 in {3} -> b//10 in {4}; a//10 in {4} -> b//10 in {5,6}; a//10 in {5} -> b//10 in {6,7}; a//10 in {6} -> b//10 in {7,8}; a//10 in {7} -> b//10 in {8,9,10}; a//10 in {8} -> b//10 in {9,10}; a//10 in {10} -> b//10 in {7,8,9,10} | - | same |

</details>

<details><summary>1 v components, main position `=` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L7 v c10 (kv6) | 100% / 100% | always | same |

</details>

<details><summary>1 q components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L7 q c80 (H27) | 99% / 20% | always | **tens(a,b)** (R2 0.65): a//10 in {2} -> b//10 in {3}; a//10 in {3} -> b//10 in {4,5,6}; a//10 in {4} -> b//10 in {5,6,7}; a//10 in {5} -> b//10 in {6,7,8}; a//10 in {6} -> b//10 in {6,7,8,9}; a//10 in {7} -> b//10 in {7,8,9,10}; a//10 in {8} -> b//10 in {9,10}; a//10 in {10} -> b//10 in {8,9,10} |

</details>

<details><summary>1 k components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L7 k c26 (kv6) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L7 k c113 (kv6) | 93% / 0 | **a** (R2 0.87): a in {4..17, 19..54, 56..85, 87..89, 91..100} | off (on 0) |

</details>

</details>

<details><summary>L7H28 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.98 / 0.98 | 0.02 / 0.02 |  |  |  |
| op | 0.93 / 0.89 | 0.03 / 0.05 | 0.04 / 0.06 |  |  |
| b | 0.75 / 0.66 | 0.13 / 0.07 | 0.09 / 0.23 | 0.04 / 0.04 |  |
| = | 0.67 / 0.80 | 0.03 / 0.03 | 0.05 / 0.03 | 0.13 / 0.07 | 0.12 / 0.07 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L7 o c33 (H28) | 0 / 92% | off (on 0) | unexplained (best R2 0.29) | - | a: mod25 +3%, mod20 +2%, mod5 +2%, mod4 +3% |

</details>

</details>

<details><summary>L8H16 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.81 / 0.81 | 0.19 / 0.19 |  |  |  |
| op | 0.74 / 0.79 | 0.07 / 0.08 | 0.19 / 0.14 |  |  |
| b | 0.68 / 0.66 | 0.05 / 0.09 | 0.09 / 0.07 | 0.19 / 0.17 |  |
| = | 0.48 / 0.60 | 0.02 / 0.02 | 0.06 / 0.03 | 0.16 / 0.06 | 0.27 / 0.28 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L8 o c289 (H16) | 100% / 0 | always | off (on 0) | - | same |

</details>

<details><summary>4 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L8 v c37 (kv4) | 0 / 1% | off (on 0) | **a** (R2 0.76): a in {42} |
| L8 v c47 (kv4) | 0 / 17% | off (on 0) | **a//10** (R2 0.47): (tens) a in {2..19, 21..23} |
| L8 v c89 (kv4) | 0 / 7% | off (on 0) | unexplained (best R2 0.49) |
| L8 v c149 (kv4) | 0 / 22% | off (on 0) | **a//10** (R2 0.64): (tens) a in {10..17, 19, 81, 83..99} |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L8 k c72 (kv4) | 1% / 23% | unexplained (best R2 0.04) | unexplained (best R2 0.37) |

</details>

</details>

<details><summary>L8H19 (6 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.93 / 0.93 | 0.07 / 0.07 |  |  |  |
| op | 0.81 / 0.77 | 0.06 / 0.15 | 0.13 / 0.08 |  |  |
| b | 0.56 / 0.53 | 0.22 / 0.11 | 0.15 / 0.23 | 0.07 / 0.12 |  |
| = | 0.53 / 0.69 | 0.06 / 0.03 | 0.08 / 0.01 | 0.16 / 0.04 | 0.17 / 0.23 |

<details><summary>6 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L8 o c113 (H19) | 0 / 20% | off (on 0) | **tens(a,b)** (R2 0.69): a//10 in {1} -> b//10 in {4,5,6,7,8,9}; a//10 in {8} -> b//10 in {1,2,3,4,5,6}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,8} | - | a: mod100 +2%, mod50 +3%, mod25 +3%, mod20 +4%, mod4 +3% |
| L8 o c265 (H19) | 0 / 8% | off (on 0) | **tens(a,b)** (R2 0.52): a//10 in {1} -> b//10 in {5,6,7,8,9,10}; a//10 in {2} -> b//10 in {7,8,9,10} | - | same |
| L8 o c299 (H19) | 0 / 20% | off (on 0) | **tens(a,b)** (R2 0.59): a//10 in {0} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {1} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {8,9,10}; a//10 in {3} -> b//10 in {9}; a//10 in {9} -> b//10 in {2,3,4,5,6} | - | same |
| L8 o c320 (H19) | 0 / 3% | off (on 0) | **a** (R2 0.53): a in {10..14} | - | same |
| L8 o c422 (H19) | 0 / 5% | off (on 0) | **tens(a,b)** (R2 0.73): a//10 in {0} -> b//10 in {3,4,5,6,7,8,9} | - | same |
| L8 o c423 (H19) | 0 / 5% | off (on 0) | **tens(a,b)** (R2 0.71): a//10 in {1} -> b//10 in {4,5,6,7,8,9} | - | same |

</details>

<details><summary>4 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L8 v c37 (kv4) | 0 / 1% | off (on 0) | **a** (R2 0.76): a in {42} |
| L8 v c47 (kv4) | 0 / 17% | off (on 0) | **a//10** (R2 0.47): (tens) a in {2..19, 21..23} |
| L8 v c89 (kv4) | 0 / 7% | off (on 0) | unexplained (best R2 0.49) |
| L8 v c149 (kv4) | 0 / 22% | off (on 0) | **a//10** (R2 0.64): (tens) a in {10..17, 19, 81, 83..99} |

</details>

<details><summary>1 q components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L8 q c131 (H19) | 0 / 59% | off (on 0) | unexplained (best R2 0.47) |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L8 k c72 (kv4) | 1% / 23% | unexplained (best R2 0.04) | unexplained (best R2 0.37) |

</details>

</details>

<details><summary>L8H22 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.96 / 0.96 | 0.04 / 0.04 |  |  |  |
| op | 0.90 / 0.78 | 0.07 / 0.17 | 0.03 / 0.05 |  |  |
| b | 0.67 / 0.59 | 0.07 / 0.09 | 0.11 / 0.28 | 0.15 / 0.04 |  |
| = | 0.74 / 0.69 | 0.05 / 0.06 | 0.04 / 0.04 | 0.06 / 0.04 | 0.11 / 0.15 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L8 o c19 (H22) | 0 / 31% | off (on 0) | unexplained (best R2 0.47) | - | same |

</details>

<details><summary>1 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L8 v c204 (kv5) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L8H23 (3 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.69 / 0.69 | 0.31 / 0.31 |  |  |  |
| op | 0.72 / 0.76 | 0.05 / 0.04 | 0.23 / 0.20 |  |  |
| b | 0.75 / 0.72 | 0.04 / 0.03 | 0.04 / 0.03 | 0.18 / 0.23 |  |
| = | 0.64 / 0.76 | 0.04 / 0.02 | 0.04 / 0.01 | 0.07 / 0.02 | 0.21 / 0.19 |

<details><summary>1 o components, main position `&lt;BOS&gt;`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L8 o c7 (H23) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same | - | same |

</details>

<details><summary>1 o components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L8 o c3 (H23) | 93% / 93% | **a** (R2 0.98): a in {8..100} | **a** (R2 0.97): a in {8..100} | - | same |

</details>

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L8 o c16 (H23) | 0 / 87% | off (on 0) | unexplained (best R2 0.43) | - | res: mod100 +3% |

</details>

<details><summary>1 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L8 v c204 (kv5) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L9H2 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.97 / 0.97 | 0.03 / 0.03 |  |  |  |
| op | 0.92 / 0.91 | 0.02 / 0.02 | 0.06 / 0.07 |  |  |
| b | 0.78 / 0.70 | 0.08 / 0.06 | 0.04 / 0.19 | 0.10 / 0.05 |  |
| = | 0.72 / 0.86 | 0.03 / 0.02 | 0.02 / 0.00 | 0.07 / 0.01 | 0.17 / 0.11 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L9 o c246 (H2) | 0 / 25% | off (on 0) | unexplained (best R2 0.46) | - | a: mod20 +2% |

</details>

<details><summary>1 q components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L9 q c113 (H2) | 0 / 28% | off (on 0) | **tens(a,b)** (R2 0.57): a//10 in {0} -> b//10 in {4,5,6,7,8,9}; a//10 in {1} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {5,6,7,8,9,10}; a//10 in {3} -> b//10 in {7,8,9,10}; a//10 in {4} -> b//10 in {8,9,10}; a//10 in {6} -> b//10 in {10}; a//10 in {9} -> b//10 in {2,3,4,5,6,7,8} |

</details>

<details><summary>2 k components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L9 k c11 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L9 k c23 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L9 k c13 (kv0) | 0 / 13% | off (on 0) | **a** (R2 0.53): a in {10..17, 19, 93..99} |

</details>

</details>

<details><summary>L9H11 (3 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.86 / 0.86 | 0.14 / 0.14 |  |  |  |
| op | 0.66 / 0.73 | 0.22 / 0.23 | 0.13 / 0.04 |  |  |
| b | 0.39 / 0.45 | 0.06 / 0.13 | 0.49 / 0.37 | 0.06 / 0.06 |  |
| = | 0.51 / 0.63 | 0.01 / 0.01 | 0.10 / 0.03 | 0.25 / 0.18 | 0.13 / 0.15 |

<details><summary>2 o components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L9 o c9 (H11) | 100% / 100% | always | same | - | same |
| L9 o c183 (H11) | 100% / 100% | always | same | - | same |

</details>

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L9 o c231 (H11) | 79% / 0 | unexplained (best R2 0.31) | off (on 0) | - | same |

</details>

<details><summary>2 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L9 v c0 (kv2) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L9 v c6 (kv2) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L9 k c47 (kv2) | 96% / 0 | always | off (on 0) |

</details>

</details>

<details><summary>L9H22 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 1.00 / 1.00 | 0.00 / 0.00 |  |  |  |
| op | 0.98 / 0.99 | 0.01 / 0.01 | 0.01 / 0.01 |  |  |
| b | 0.84 / 0.89 | 0.02 / 0.02 | 0.04 / 0.04 | 0.10 / 0.05 |  |
| = | 0.65 / 0.74 | 0.01 / 0.01 | 0.02 / 0.01 | 0.14 / 0.03 | 0.19 / 0.21 |

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L9 o c409 (H22) | 97% / 0 | always | off (on 0) | a: mod100 +2%; b: mod100 +3%, mod50 +3%, mod25 +2% | - |

</details>

<details><summary>1 k components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L9 k c34 (kv5) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

</details>

<details><summary>L10H15 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.97 / 0.97 | 0.03 / 0.03 |  |  |  |
| op | 0.94 / 0.96 | 0.02 / 0.03 | 0.03 / 0.01 |  |  |
| b | 0.35 / 0.80 | 0.03 / 0.03 | 0.33 / 0.08 | 0.29 / 0.08 |  |
| = | 0.36 / 0.31 | 0.01 / 0.02 | 0.08 / 0.03 | 0.28 / 0.21 | 0.27 / 0.43 |

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L10 o c54 (H15) | 100% / 100% | always | same | - | same |

</details>

<details><summary>1 k components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L10 k c20 (kv3) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

</details>

<details><summary>L10H30 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.95 / 0.95 | 0.05 / 0.05 |  |  |  |
| op | 0.89 / 0.91 | 0.06 / 0.03 | 0.05 / 0.06 |  |  |
| b | 0.72 / 0.77 | 0.08 / 0.08 | 0.06 / 0.06 | 0.15 / 0.09 |  |
| = | 0.53 / 0.71 | 0.03 / 0.04 | 0.03 / 0.01 | 0.10 / 0.04 | 0.30 / 0.20 |

<details><summary>1 o components, main position `&lt;BOS&gt;`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L10 o c1 (H30) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same | - | same |

</details>

<details><summary>2 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L10 v c0 (kv7) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L10 v c1 (kv7) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>1 k components, main position `=` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L10 k c58 (kv7) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L11H4 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.99 / 0.99 | 0.01 / 0.01 |  |  |  |
| op | 0.97 / 0.98 | 0.03 / 0.01 | 0.01 / 0.01 |  |  |
| b | 0.88 / 0.93 | 0.07 / 0.03 | 0.02 / 0.02 | 0.03 / 0.02 |  |
| = | 0.46 / 0.50 | 0.11 / 0.08 | 0.06 / 0.09 | 0.32 / 0.14 | 0.05 / 0.19 |

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L11 o c428 (H4) | 0 / 0 | off (on 0) | same | - | same |

</details>

</details>

<details><summary>L11H16 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.96 / 0.96 | 0.04 / 0.04 |  |  |  |
| op | 0.85 / 0.84 | 0.08 / 0.08 | 0.07 / 0.09 |  |  |
| b | 0.51 / 0.49 | 0.04 / 0.08 | 0.13 / 0.37 | 0.32 / 0.06 |  |
| = | 0.62 / 0.74 | 0.02 / 0.03 | 0.04 / 0.04 | 0.10 / 0.06 | 0.22 / 0.13 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L11 o c37 (H16) | 0 / 32% | off (on 0) | **tens(a,b)** (R2 0.62): a//10 in {1} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {6,7,8,9,10}; a//10 in {3,4} -> b//10 in {7,8,9,10}; a//10 in {5} -> b//10 in {9,10}; a//10 in {6} -> b//10 in {10}; a//10 in {7} -> b//10 in {3,4}; a//10 in {8} -> b//10 in {4,5}; a//10 in {9} -> b//10 in {1,2,3,4,5,6,7,8,9} | - | a: mod25 +2%, mod20 +3% |

</details>

<details><summary>2 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L11 v c0 (kv4) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L11 v c1 (kv4) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L11 v c171 (kv4) | 0 / 8% | off (on 0) | **a** (R2 0.96): a in {92..99} |

</details>

<details><summary>1 q components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L11 q c84 (H16) | 0 / 32% | off (on 0) | **tens(a,b)** (R2 0.61): a//10 in {0,6} -> b//10 in {10}; a//10 in {1} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {5,6,7,8,9,10}; a//10 in {3,4} -> b//10 in {7,8,9,10}; a//10 in {5} -> b//10 in {9,10}; a//10 in {7} -> b//10 in {3,4}; a//10 in {8} -> b//10 in {4,5}; a//10 in {9} -> b//10 in {1,2,3,4,5,6,7,8} |

</details>

<details><summary>1 k components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L11 k c30 (kv4) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

</details>

<details><summary>L11H20 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.99 / 0.99 | 0.01 / 0.01 |  |  |  |
| op | 0.95 / 0.97 | 0.03 / 0.02 | 0.02 / 0.01 |  |  |
| b | 0.91 / 0.92 | 0.04 / 0.04 | 0.04 / 0.03 | 0.01 / 0.01 |  |
| = | 0.89 / 0.90 | 0.02 / 0.02 | 0.03 / 0.02 | 0.05 / 0.02 | 0.01 / 0.03 |

<details><summary>1 o components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L11 o c4 (H20) | 100% / 100% | always | same | - | same |

</details>

<details><summary>1 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L11 v c16 (kv5) | 100% / 100% | always | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L11 v c189 (kv5) | 81% / 0 | **a** (R2 1.00): a in {4, 6..10, 12..17, 19, 21..24, 26..29, 31..39, 41..49, 51..54, 56..59, 61..64, 66..69, 71..74, 76..79, 81..89, 91..99} | off (on 0) |

</details>

<details><summary>1 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L11 v c230 (kv5) | 100% / 0 | always | off (on 0) |

</details>

</details>

<details><summary>L11H26 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.97 / 0.97 | 0.03 / 0.03 |  |  |  |
| op | 0.93 / 0.93 | 0.04 / 0.03 | 0.03 / 0.04 |  |  |
| b | 0.80 / 0.78 | 0.09 / 0.06 | 0.08 / 0.12 | 0.04 / 0.04 |  |
| = | 0.63 / 0.76 | 0.05 / 0.08 | 0.08 / 0.03 | 0.13 / 0.04 | 0.10 / 0.09 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L11 o c15 (H26) | 0 / 5% | off (on 0) | **a** (R2 0.64): a in {93..99} | - | same |

</details>

<details><summary>2 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L11 v c4 (kv6) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L11 v c9 (kv6) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L11 v c76 (kv6) | 0 / 8% | off (on 0) | **a** (R2 0.95): a in {92..99} |

</details>

</details>

<details><summary>L12H3 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.85 / 0.85 | 0.15 / 0.15 |  |  |  |
| op | 0.81 / 0.89 | 0.08 / 0.07 | 0.12 / 0.04 |  |  |
| b | 0.60 / 0.68 | 0.20 / 0.17 | 0.13 / 0.07 | 0.07 / 0.08 |  |
| = | 0.75 / 0.82 | 0.06 / 0.07 | 0.08 / 0.04 | 0.06 / 0.04 | 0.05 / 0.04 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L12 o c30 (H3) | 68% / 0 | unexplained (best R2 0.46) | off (on 0) | - | same |

</details>

<details><summary>1 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L12 v c1 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>1 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L12 v c200 (kv0) | 100% / 100% | always | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L12 v c52 (kv0) | 100% / 0 | always | off (on 0) |

</details>

<details><summary>1 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L12 v c87 (kv0) | 78% / 0 | unexplained (best R2 0.28) | off (on 0) |

</details>

<details><summary>1 v components, main position `=` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L12 v c145 (kv0) | 0 / 0 | off (on 0) | same |

</details>

<details><summary>1 q components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L12 q c40 (H3) | 0 / 0 | off (on 0) | same |

</details>

</details>

<details><summary>L12H19 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.97 / 0.97 | 0.03 / 0.03 |  |  |  |
| op | 0.91 / 0.90 | 0.06 / 0.07 | 0.03 / 0.02 |  |  |
| b | 0.62 / 0.50 | 0.03 / 0.05 | 0.12 / 0.39 | 0.23 / 0.05 |  |
| = | 0.77 / 0.77 | 0.00 / 0.01 | 0.03 / 0.04 | 0.08 / 0.04 | 0.12 / 0.14 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L12 o c74 (H19) | 0 / 90% | off (on 0) | **tens(a,b)** (R2 0.61): a//10 in {0,1,2} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {3,4,5,6,7} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {8,9,10} -> b//10 in {2,3,4,5,6,7,8,9,10} | - | same |

</details>

</details>

<details><summary>L12H20 (4 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.66 / 0.66 | 0.34 / 0.34 |  |  |  |
| op | 0.75 / 0.69 | 0.05 / 0.06 | 0.20 / 0.25 |  |  |
| b | 0.77 / 0.71 | 0.03 / 0.05 | 0.06 / 0.06 | 0.14 / 0.18 |  |
| = | 0.73 / 0.81 | 0.02 / 0.02 | 0.04 / 0.03 | 0.09 / 0.04 | 0.13 / 0.11 |

<details><summary>2 o components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L12 o c41 (H20) | 100% / 100% | always | same | - | same |
| L12 o c209 (H20) | 100% / 100% | always | same | - | same |

</details>

<details><summary>2 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L12 o c319 (H20) | 96% / 0 | always | off (on 0) | - | same |
| L12 o c459 (H20) | 0 / 98% | off (on 0) | always | - | same |

</details>

<details><summary>2 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L12 v c116 (kv5) | 0 / 98% | off (on 0) | always |
| L12 v c138 (kv5) | 0 / 9% | off (on 0) | **a** (R2 1.00): a in {91..99} |

</details>

<details><summary>1 q components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L12 q c142 (H20) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L13H7 (9 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.98 / 0.98 | 0.02 / 0.02 |  |  |  |
| op | 0.96 / 0.93 | 0.02 / 0.04 | 0.02 / 0.02 |  |  |
| b | 0.80 / 0.87 | 0.04 / 0.04 | 0.04 / 0.03 | 0.12 / 0.06 |  |
| = | 0.19 / 0.25 | 0.01 / 0.02 | 0.03 / 0.01 | 0.67 / 0.30 | 0.11 / 0.43 |

<details><summary>9 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L13 o c1 (H7) | 100% / 100% | always | same | a: mod50 +2%, mod25 +2%; res: mod50 +3%, mod2 +2% | - |
| L13 o c72 (H7) | 5% / 0 | **a** (R2 0.59): a in {1..4} | off (on 0) | a: mod50 +2%, mod25 +2% | - |
| L13 o c145 (H7) | 0 / 100% | off (on 0) | always | - | a: mod100 +2%, mod50 +4%, mod2 +2%; b: mod100 +4%, mod50 +2%, mod20 +2%, mod5 +4%; res: mod100 +2%, mod25 +3%, mod20 +3%, mod10 +3%, mod5 +2% |
| L13 o c186 (H7) | 0 / 1% | off (on 0) | unexplained (best R2 0.36) | - | same |
| L13 o c213 (H7) | 0 / 60% | off (on 0) | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,9,10}; a//10 in {6} -> b//10 in {0,1,2,3,10}; a//10 in {7,8} -> b//10 in {0,1,2,3,4}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7}; a//10 in {10} -> b//10 in {0,1,2,3} | - | a: mod25 +2%, mod4 +5%; b: mod100 +3%; res: mod100 +9%, mod5 +3% |
| L13 o c231 (H7) | 10% / 6% | unexplained (best R2 0.41) | **res%100** (R2 0.65): res mod 100 in {0} | - | same |
| L13 o c304 (H7) | 8% / 21% | unexplained (best R2 0.33) | **res//10** (R2 0.55): (tens) res in {-1..20, 99} | - | res: mod100 +3%, mod50 +5%, mod25 +5%, mod20 +5%, mod10 +3% |
| L13 o c379 (H7) | 1% / 1% | **cmp(a,b)** (R2 0.92): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.96, cmp(a,b)=1: 0.00 | **cmp(a,b)** (R2 0.97): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 1.00, cmp(a,b)=1: 0.00 | - | same |
| L13 o c414 (H7) | 26% / 0 | **tens(a,b)** (R2 0.78): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {0,10}; a//10 in {2} -> b//10 in {0,1,10}; a//10 in {3,4} -> b//10 in {0,1}; a//10 in {5,6,7} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1,2,3}; a//10 in {9} -> b//10 in {0,1,2,3,4}; a//10 in {10} -> b//10 in {0,1,2,3,4,5} | off (on 0) | b: mod100 +11%, mod50 +7%, mod25 +3%, mod4 +2%; res: mod100 +4% | - |

</details>

<details><summary>1 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L13 v c116 (kv1) | 0 / 0 | at &lt;BOS&gt; (constant), on 0 | same |

</details>

<details><summary>6 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L13 v c65 (kv1) | 1% / 1% | **cmp(a,b)** (R2 0.94): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.99, cmp(a,b)=1: 0.00 | **cmp(a,b)** (R2 0.97): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.99, cmp(a,b)=1: 0.00 |
| L13 v c69 (kv1) | 16% / 2% | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {3,4,5,6,8,9,10}; a//10 in {2} -> b//10 in {9,10}; a//10 in {3,4} -> b//10 in {10} | unexplained (best R2 0.39) |
| L13 v c70 (kv1) | 3% / 4% | unexplained (best R2 0.35) | **res%100** (R2 0.69): res mod 100 in {0} |
| L13 v c172 (kv1) | 0 / 39% | off (on 0) | **res//10** (R2 0.74): (tens) res in {-99..-95, -65, -61..-1} |
| L13 v c187 (kv1) | 29% / 0 | **tens(a,b)** (R2 0.81): a//10 in {1} -> b//10 in {0,10}; a//10 in {2,3} -> b//10 in {0,1}; a//10 in {4,5,6} -> b//10 in {0,1,2}; a//10 in {7,8} -> b//10 in {0,1,2,3}; a//10 in {9} -> b//10 in {0,1,2,3,4,5}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7} | off (on 0) |
| L13 v c238 (kv1) | 17% / 9% | unexplained (best R2 0.46) | **res%100** (R2 0.55): res mod 100 in {0} |

</details>

<details><summary>1 v components, main position `=` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L13 v c31 (kv1) | 0 / 100% | off (on 0) | always |

</details>

<details><summary>1 q components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L13 q c121 (H7) | 100% / 100% | always | same |

</details>

<details><summary>1 k components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L13 k c52 (kv1) | 99% / 66% | always | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {0,1}; a//10 in {1} -> b//10 in {0,1,2,3}; a//10 in {2} -> b//10 in {0,1,2,3,4,5,6,7,8,9}; a//10 in {3,4} -> b//10 in {1,2,3,4,5,6,7,8,9}; a//10 in {5} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {7,8} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} |

</details>

</details>

<details><summary>L13H16 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.95 / 0.95 | 0.05 / 0.05 |  |  |  |
| op | 0.93 / 0.84 | 0.03 / 0.06 | 0.04 / 0.10 |  |  |
| b | 0.76 / 0.84 | 0.07 / 0.05 | 0.06 / 0.04 | 0.10 / 0.06 |  |
| = | 0.49 / 0.47 | 0.09 / 0.07 | 0.07 / 0.03 | 0.13 / 0.06 | 0.21 / 0.36 |

<details><summary>1 o components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L13 o c0 (H16) | 100% / 100% | always | same | - | same |

</details>

<details><summary>2 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L13 v c6 (kv4) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L13 v c8 (kv4) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>1 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L13 v c193 (kv4) | 0 / 67% | off (on 0) | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {0,1,2}; a//10 in {1} -> b//10 in {0,1,2,3}; a//10 in {2} -> b//10 in {0,1,2,3,4,5,6,7,8,9}; a//10 in {3,4} -> b//10 in {1,2,3,4,5,6,7,8,9}; a//10 in {5} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {6,7,8} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} |

</details>

<details><summary>1 v components, main position `=` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L13 v c1 (kv4) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L14H18 (6 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.96 / 0.96 | 0.04 / 0.04 |  |  |  |
| op | 0.93 / 0.59 | 0.05 / 0.28 | 0.02 / 0.13 |  |  |
| b | 0.92 / 0.91 | 0.03 / 0.05 | 0.03 / 0.02 | 0.02 / 0.02 |  |
| = | 0.36 / 0.37 | 0.28 / 0.30 | 0.04 / 0.04 | 0.28 / 0.26 | 0.03 / 0.02 |

<details><summary>5 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L14 o c92 (H18) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {80, 90} | - | same |
| L14 o c129 (H18) | 0 / 12% | off (on 0) | **a** (R2 1.00): a in {11, 21, 31, 41, 47, 49, 51, 53, 61, 71, 81, 91} [coarser: a mod 50 in {11, 21, 31, 41}, R2 0.81] | - | a: mod2 +2% |
| L14 o c248 (H18) | 0 / 19% | off (on 0) | **a** (R2 1.00): a in {4..12, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100} | - | a: mod5 +3% |
| L14 o c296 (H18) | 0 / 11% | off (on 0) | **a//10** (R2 0.90): (tens) a in {78, 80..89} | - | same |
| L14 o c392 (H18) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {35, 45} | - | same |

</details>

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L14 o c101 (H18) | 0 / 11% | off (on 0) | **a//10** (R2 0.83): (tens) a in {90..100} | - | a: mod100 +2%, mod50 +3%, mod25 +3%, mod20 +5% |

</details>

<details><summary>9 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L14 v c46 (kv4) | 5% / 13% | **a** (R2 0.96): a in {78, 82, 84, 86, 88} | **a** (R2 0.99): a in {78, 80..90, 92} |
| L14 v c54 (kv4) | 8% / 8% | **a** (R2 1.00): a in {93..100} | same |
| L14 v c67 (kv4) | 1% / 2% | **a** (R2 0.94): a in {28} | **a** (R2 1.00): a in {28..29} |
| L14 v c81 (kv4) | 4% / 4% | **a** (R2 0.99): a in {7, 17, 27, 97} | **a** (R2 1.00): a in {7, 17, 27, 97} |
| L14 v c109 (kv4) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {24} |
| L14 v c125 (kv4) | 18% / 24% | **a** (R2 0.87): a in {1..14, 20, 100} | **a** (R2 0.98): a in {1..16, 20, 30, 40, 50, 60, 80, 90, 100} |
| L14 v c148 (kv4) | 2% / 3% | **a** (R2 0.99): a in {85, 87} | **a** (R2 1.00): a in {24, 85, 87} |
| L14 v c173 (kv4) | 3% / 10% | **a** (R2 0.92): a in {70, 80, 90} | **a%10** (R2 1.00): a mod 10 in {0} |
| L14 v c221 (kv4) | 7% / 10% | **a** (R2 0.98): a in {11, 41, 51, 61, 71, 81, 91} | **a** (R2 1.00): a in {11, 21, 31, 41, 51, 53, 61, 71, 81, 91} [coarser: a mod 10 in {1}, R2 0.80] |

</details>

<details><summary>1 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L14 v c53 (kv4) | 0 / 50% | off (on 0) | **tens(a,b)** (R2 0.53): a//10 in {0} -> b//10 in {0,1}; a//10 in {1} -> b//10 in {1,2,3}; a//10 in {2} -> b//10 in {1,2,3,4,5,6}; a//10 in {3} -> b//10 in {2,3,4,5,6,7,8}; a//10 in {4,5} -> b//10 in {3,4,5,6,7,8}; a//10 in {6} -> b//10 in {4,5,6,7,8,9}; a//10 in {7,8} -> b//10 in {5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10}; a//10 in {10} -> b//10 in {3,4,5,6,7,8,9} |

</details>

<details><summary>1 k components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L14 k c7 (kv4) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L14 k c39 (kv4) | 0 / 60% | off (on 0) | **a** (R2 0.86): a in {10..64, 80, 90} |

</details>

</details>

<details><summary>L14H19 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.94 / 0.94 | 0.06 / 0.06 |  |  |  |
| op | 0.92 / 0.65 | 0.01 / 0.29 | 0.07 / 0.06 |  |  |
| b | 0.87 / 0.92 | 0.01 / 0.00 | 0.00 / 0.00 | 0.11 / 0.07 |  |
| = | 0.70 / 0.70 | 0.00 / 0.00 | 0.02 / 0.02 | 0.00 / 0.01 | 0.27 / 0.26 |

<details><summary>1 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L14 o c404 (H19) | 0 / 8% | off (on 0) | **a** (R2 0.99): a in {93..100} | - | same |

</details>

<details><summary>9 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L14 v c46 (kv4) | 5% / 13% | **a** (R2 0.96): a in {78, 82, 84, 86, 88} | **a** (R2 0.99): a in {78, 80..90, 92} |
| L14 v c54 (kv4) | 8% / 8% | **a** (R2 1.00): a in {93..100} | same |
| L14 v c67 (kv4) | 1% / 2% | **a** (R2 0.94): a in {28} | **a** (R2 1.00): a in {28..29} |
| L14 v c81 (kv4) | 4% / 4% | **a** (R2 0.99): a in {7, 17, 27, 97} | **a** (R2 1.00): a in {7, 17, 27, 97} |
| L14 v c109 (kv4) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {24} |
| L14 v c125 (kv4) | 18% / 24% | **a** (R2 0.87): a in {1..14, 20, 100} | **a** (R2 0.98): a in {1..16, 20, 30, 40, 50, 60, 80, 90, 100} |
| L14 v c148 (kv4) | 2% / 3% | **a** (R2 0.99): a in {85, 87} | **a** (R2 1.00): a in {24, 85, 87} |
| L14 v c173 (kv4) | 3% / 10% | **a** (R2 0.92): a in {70, 80, 90} | **a%10** (R2 1.00): a mod 10 in {0} |
| L14 v c221 (kv4) | 7% / 10% | **a** (R2 0.98): a in {11, 41, 51, 61, 71, 81, 91} | **a** (R2 1.00): a in {11, 21, 31, 41, 51, 53, 61, 71, 81, 91} [coarser: a mod 10 in {1}, R2 0.80] |

</details>

<details><summary>1 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L14 v c53 (kv4) | 0 / 50% | off (on 0) | **tens(a,b)** (R2 0.53): a//10 in {0} -> b//10 in {0,1}; a//10 in {1} -> b//10 in {1,2,3}; a//10 in {2} -> b//10 in {1,2,3,4,5,6}; a//10 in {3} -> b//10 in {2,3,4,5,6,7,8}; a//10 in {4,5} -> b//10 in {3,4,5,6,7,8}; a//10 in {6} -> b//10 in {4,5,6,7,8,9}; a//10 in {7,8} -> b//10 in {5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10}; a//10 in {10} -> b//10 in {3,4,5,6,7,8,9} |

</details>

<details><summary>1 q components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L14 q c78 (H19) | 0 / 98% | off (on 0) | always |

</details>

<details><summary>1 k components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L14 k c7 (kv4) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L14 k c39 (kv4) | 0 / 60% | off (on 0) | **a** (R2 0.86): a in {10..64, 80, 90} |

</details>

</details>

<details><summary>L14H26 (4 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.92 / 0.92 | 0.08 / 0.08 |  |  |  |
| op | 0.48 / 0.62 | 0.45 / 0.33 | 0.06 / 0.05 |  |  |
| b | 0.29 / 0.35 | 0.09 / 0.19 | 0.57 / 0.42 | 0.05 / 0.04 |  |
| = | 0.38 / 0.43 | 0.00 / 0.00 | 0.03 / 0.03 | 0.47 / 0.43 | 0.12 / 0.12 |

<details><summary>1 o components, main position `&lt;BOS&gt;`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L14 o c59 (H26) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same | - | same |

</details>

<details><summary>3 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L14 o c37 (H26) | 100% / 0 | always | off (on 0) | - | same |
| L14 o c49 (H26) | 0 / 3% | off (on 0) | unexplained (best R2 0.17) | - | same |
| L14 o c158 (H26) | 0 / 100% | off (on 0) | always | - | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L14 v c163 (kv6) | 0 / 100% | off (on 0) | always |

</details>

</details>

<details><summary>L14H27 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.95 / 0.95 | 0.05 / 0.05 |  |  |  |
| op | 0.92 / 0.80 | 0.03 / 0.03 | 0.05 / 0.18 |  |  |
| b | 0.85 / 0.81 | 0.03 / 0.04 | 0.04 / 0.08 | 0.07 / 0.07 |  |
| = | 0.82 / 0.85 | 0.03 / 0.03 | 0.03 / 0.04 | 0.07 / 0.03 | 0.05 / 0.05 |

<details><summary>1 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L14 o c495 (H27) | 0 / 98% | off (on 0) | always | - | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L14 v c163 (kv6) | 0 / 100% | off (on 0) | always |

</details>

</details>

<details><summary>L14H31 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.99 / 0.99 | 0.01 / 0.01 |  |  |  |
| op | 0.97 / 0.98 | 0.02 / 0.01 | 0.01 / 0.01 |  |  |
| b | 0.87 / 0.95 | 0.03 / 0.01 | 0.01 / 0.01 | 0.09 / 0.02 |  |
| = | 0.29 / 0.71 | 0.03 / 0.04 | 0.06 / 0.03 | 0.55 / 0.13 | 0.06 / 0.10 |

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L14 o c84 (H31) | 100% / 100% | always | same | a: mod100 +2%, mod50 +4%, mod25 +3%, mod20 +3%, mod4 +4%; b: mod50 +3%, mod25 +3%; res: mod50 +3%, mod2 +5% | - |

</details>

<details><summary>2 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L14 v c9 (kv7) | 100% / 100% | always | same |
| L14 v c131 (kv7) | 29% / 47% | unexplained (best R2 0.49) | **a** (R2 0.88): a in {4, 31..49, 51..55, 61..63, 65, 74, 81..94, 96} |

</details>

<details><summary>1 v components, main position `=` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L14 v c129 (kv7) | 100% / 100% | always | same |

</details>

<details><summary>1 q components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L14 q c3 (H31) | 100% / 100% | always | same |

</details>

<details><summary>1 q components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L14 q c123 (H31) | 87% / 50% | unexplained (best R2 0.24) | **tens(a,b)** (R2 0.57): a//10 in {2} -> b//10 in {3,4,5}; a//10 in {3} -> b//10 in {4,5,6,7}; a//10 in {4} -> b//10 in {5,6,7,8}; a//10 in {5} -> b//10 in {4,5,6,7,8,9}; a//10 in {6,7,8} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {4,5,6,7,8,9,10} |

</details>

</details>

<details><summary>L15H3 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.99 / 0.99 | 0.01 / 0.01 |  |  |  |
| op | 0.98 / 0.99 | 0.01 / 0.01 | 0.01 / 0.01 |  |  |
| b | 0.86 / 0.96 | 0.01 / 0.01 | 0.02 / 0.02 | 0.10 / 0.01 |  |
| = | 0.46 / 0.65 | 0.05 / 0.02 | 0.04 / 0.03 | 0.38 / 0.28 | 0.07 / 0.02 |

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L15 o c270 (H3) | 0 / 3% | off (on 0) | unexplained (best R2 0.34) | - | same |

</details>

<details><summary>1 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L15 v c138 (kv0) | 0 / 29% | off (on 0) | **res//10** (R2 0.70): (tens) res in {-99, -39, -35..-1} |

</details>

</details>

<details><summary>L15H5 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.98 / 0.98 | 0.02 / 0.02 |  |  |  |
| op | 0.93 / 0.94 | 0.05 / 0.04 | 0.03 / 0.02 |  |  |
| b | 0.61 / 0.84 | 0.26 / 0.09 | 0.05 / 0.03 | 0.08 / 0.03 |  |
| = | 0.50 / 0.66 | 0.08 / 0.09 | 0.08 / 0.02 | 0.19 / 0.06 | 0.15 / 0.17 |

<details><summary>1 o components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L15 o c3 (H5) | 100% / 100% | always | same | - | same |

</details>

<details><summary>2 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L15 v c1 (kv1) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L15 v c3 (kv1) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

</details>

<details><summary>L15H6 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.53 / 0.53 | 0.47 / 0.47 |  |  |  |
| op | 0.48 / 0.45 | 0.15 / 0.13 | 0.37 / 0.42 |  |  |
| b | 0.22 / 0.36 | 0.21 / 0.16 | 0.35 / 0.19 | 0.23 / 0.30 |  |
| = | 0.45 / 0.49 | 0.03 / 0.06 | 0.06 / 0.06 | 0.12 / 0.12 | 0.34 / 0.27 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L15 o c36 (H6) | 0 / 7% | off (on 0) | **a** (R2 0.69): a in {93..99} | - | same |

</details>

<details><summary>2 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L15 v c1 (kv1) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L15 v c3 (kv1) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

</details>

<details><summary>L15H7 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.78 / 0.78 | 0.22 / 0.22 |  |  |  |
| op | 0.62 / 0.72 | 0.30 / 0.22 | 0.08 / 0.06 |  |  |
| b | 0.56 / 0.39 | 0.08 / 0.13 | 0.13 / 0.36 | 0.23 / 0.11 |  |
| = | 0.60 / 0.62 | 0.02 / 0.02 | 0.04 / 0.05 | 0.23 / 0.13 | 0.11 / 0.19 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L15 o c499 (H7) | 0 / 99% | off (on 0) | always | - | same |

</details>

<details><summary>2 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L15 v c1 (kv1) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L15 v c3 (kv1) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

</details>

<details><summary>L15H13 (39 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 1.00 / 1.00 | 0.00 / 0.00 |  |  |  |
| op | 0.99 / 0.98 | 0.01 / 0.00 | 0.01 / 0.01 |  |  |
| b | 0.86 / 0.96 | 0.03 / 0.00 | 0.10 / 0.03 | 0.01 / 0.00 |  |
| = | 0.03 / 0.14 | 0.05 / 0.04 | 0.01 / 0.09 | 0.91 / 0.63 | 0.01 / 0.10 |

<details><summary>39 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L15 o c17 (H13) | 59% / 30% | **b** (R2 0.74): b in {12..19, 21..25, 27..29, 33, 43, 51..89, 93..95, 99} | **tens(a,b)** (R2 0.57): a//10 in {2} -> b//10 in {1}; a//10 in {3} -> b//10 in {1,2,6,7}; a//10 in {4} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {4,6,7,8}; a//10 in {6} -> b//10 in {5,6,7,8}; a//10 in {7,8,10} -> b//10 in {5,6,7,8,9}; a//10 in {9} -> b//10 in {6,7,8,9} | b: mod100 +3%, mod50 +3% | - |
| L15 o c19 (H13) | 34% / 17% | **b** (R2 0.94): b in {3, 33, 35, 37..65, 99..100} | **tens(a,b)** (R2 0.65): a//10 in {4} -> b//10 in {4}; a//10 in {5} -> b//10 in {4,5,10}; a//10 in {6,7,8,9,10} -> b//10 in {4,5,6} | - | same |
| L15 o c25 (H13) | 8% / 4% | unexplained (best R2 0.27) | unexplained (best R2 0.30) | - | same |
| L15 o c30 (H13) | 47% / 46% | **b//10** (R2 0.88): (tens) b in {2..47} | **tens(a,b)** (R2 0.69): a//10 in {0} -> b//10 in {1}; a//10 in {1} -> b//10 in {0,1}; a//10 in {2,3,4} -> b//10 in {0,1,2,3}; a//10 in {5} -> b//10 in {0,1,2,3,4}; a//10 in {6} -> b//10 in {0,1,2,3,4,5}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,6}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,7}; a//10 in {9} -> b//10 in {0,1,2,3,8}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8,9} | b: mod100 +6%, mod50 +2% | b: mod100 +3% |
| L15 o c33 (H13) | 31% / 23% | **b%10** (R2 0.95): b mod 10 in {0..1, 9} | **b%10** (R2 0.62): b mod 10 in {0..1, 9} | b: mod10 +6%, mod5 +4% | b: mod10 +6%, mod5 +3% |
| L15 o c51 (H13) | 0 / 3% | off (on 0) | unexplained (best R2 0.40) | - | same |
| L15 o c53 (H13) | 40% / 33% | **b%10** (R2 0.98): b mod 10 in {3, 7..9} | **b%10** (R2 0.72): b mod 10 in {3, 7..9} | b: mod10 +8%, mod5 +9%, mod2 +2% | b: mod10 +7%, mod5 +7% |
| L15 o c54 (H13) | 0 / 0 | off (on 0) | same | - | same |
| L15 o c58 (H13) | 40% / 24% | **b** (R2 0.93): b in {20..25, 54..86} | **tens(a,b)** (R2 0.69): a//10 in {2,3} -> b//10 in {6,7}; a//10 in {4,5,6,8,9} -> b//10 in {6,7,8}; a//10 in {7} -> b//10 in {6,7,8,10}; a//10 in {10} -> b//10 in {2,5,6,7,8} | b: mod100 +3%, mod50 +5% | - |
| L15 o c59 (H13) | 11% / 11% | **b** (R2 0.88): b in {24..33} | **b** (R2 0.78): b in {25..33} | - | same |
| L15 o c60 (H13) | 50% / 51% | **b%2** (R2 0.99): b mod 2 in {1} | **b%2** (R2 0.92): b mod 2 in {1} | b: mod2 +31% | b: mod2 +30% |
| L15 o c69 (H13) | 24% / 15% | **b** (R2 0.56): b in {31..32, 34..38, 40..42, 71..72, 75..85, 87, 89..90, 96} | unexplained (best R2 0.45) | b: mod50 +2% | - |
| L15 o c73 (H13) | 30% / 28% | **b%10** (R2 0.99): b mod 10 in {5..7} | **b%10** (R2 0.87): b mod 10 in {5..7} | b: mod10 +16%, mod5 +6%, mod2 +3% | b: mod10 +12%, mod5 +5% |
| L15 o c75 (H13) | 34% / 41% | **b** (R2 0.70): b in {1..23, 61, 63, 97, 99..100} | **tens(a,b)** (R2 0.72): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1,2,7} -> b//10 in {0,1,2}; a//10 in {3} -> b//10 in {0,1,3,4}; a//10 in {4} -> b//10 in {0,1,2,4,5}; a//10 in {5} -> b//10 in {0,1,5,6,10}; a//10 in {6} -> b//10 in {0,1,2,6,10}; a//10 in {8} -> b//10 in {0,1,10}; a//10 in {9,10} -> b//10 in {0,1,9,10} | b: mod100 +5%, mod50 +5% | b: mod100 +4%, mod50 +5%, mod25 +2% |
| L15 o c77 (H13) | 35% / 19% | **b//10** (R2 0.87): (tens) b in {20..54} | **tens(a,b)** (R2 0.66): a//10 in {1} -> b//10 in {3}; a//10 in {2,3,4,5,10} -> b//10 in {2,3,4}; a//10 in {6,7,8,9} -> b//10 in {3,4} | b: mod100 +4%, mod50 +4% | b: mod50 +2% |
| L15 o c78 (H13) | 21% / 11% | **b** (R2 0.67): b in {4, 12..16, 18, 24, 26, 34, 36, 44, 46, 54, 56, 64, 66, 74, 84, 94, 96} [coarser: b mod 50 in {4, 6, 12, 14, 16, 18, 24, 34, 44, 46}, R2 0.84] | unexplained (best R2 0.25) | - | same |
| L15 o c81 (H13) | 69% / 59% | **b** (R2 0.96): b in {5..24, 33..34, 43..59, 65..94} | **b** (R2 0.53): b in {5..12, 15..24, 31..34, 44..55, 58, 65..95} | b: mod50 +4%, mod20 +28%, mod4 +10% | b: mod20 +18%, mod4 +4% |
| L15 o c82 (H13) | 36% / 30% | **b//10** (R2 0.84): (tens) b in {7, 9, 32, 70..100} | **b//10** (R2 0.83): (tens) b in {71..100} | b: mod100 +9%, mod50 +9% | b: mod100 +5%, mod50 +6% |
| L15 o c86 (H13) | 0 / 1% | off (on 0) | unexplained (best R2 0.40) | - | same |
| L15 o c93 (H13) | 11% / 10% | **b%10** (R2 0.91): b mod 10 in {5} | **b%10** (R2 0.77): b mod 10 in {5} | b: mod5 +6%, mod2 +3% | b: mod5 +3%, mod2 +2% |
| L15 o c105 (H13) | 12% / 9% | **b** (R2 0.83): b in {53..54, 56..64, 66} | **tens(a,b)** (R2 0.66): a//10 in {5} -> b//10 in {5}; a//10 in {6,7,8,10} -> b//10 in {5,6}; a//10 in {9} -> b//10 in {6} | - | same |
| L15 o c107 (H13) | 50% / 20% | **b** (R2 0.89): b in {2, 4, 12..14, 22..34, 42..44, 52..54, 62..64, 66..74, 81..94} | unexplained (best R2 0.41) | b: mod20 +5%, mod10 +4%, mod5 +2%, mod2 +2% | b: mod20 +2%, mod10 +2% |
| L15 o c111 (H13) | 59% / 38% | **b%10** (R2 0.96): b mod 10 in {0..2, 6..8} | unexplained (best R2 0.49) | b: mod5 +11%, mod2 +9% | b: mod5 +5%, mod2 +4% |
| L15 o c119 (H13) | 22% / 47% | **b** (R2 0.58): b in {65..71, 73..80, 84..86, 88, 93..95} | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {0}; a//10 in {1,2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {1,2,3,4,6,7}; a//10 in {4} -> b//10 in {3,4,6,7,8,9}; a//10 in {5} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {6} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {7,8,10} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10} | - | res: mod100 +5%, mod50 +3% |
| L15 o c120 (H13) | 35% / 17% | **b%20** (R2 0.78): b mod 20 in {0..1, 5, 9..11, 15, 19} [coarser: b mod 10 in {0..1, 5, 9}, R2 0.86] | unexplained (best R2 0.50) | b: mod5 +3%, mod2 +3% | - |
| L15 o c122 (H13) | 49% / 45% | **b** (R2 0.98): b in {1..4, 32..64, 82..84, 92..100} | **b** (R2 0.70): b in {1..5, 32..64, 82..84, 92..100} | b: mod50 +4%, mod20 +7%, mod4 +3% | b: mod50 +3%, mod20 +5% |
| L15 o c129 (H13) | 30% / 18% | **b%10** (R2 0.98): b mod 10 in {4..6} | **b%10** (R2 0.48): b mod 10 in {4..6} | b: mod10 +5%, mod5 +4% | b: mod10 +3%, mod5 +2% |
| L15 o c132 (H13) | 30% / 24% | **b%10** (R2 0.93): b mod 10 in {0, 8..9} | **b%10** (R2 0.58): b mod 10 in {0, 8..9} | b: mod10 +6%, mod5 +3%, mod2 +3% | b: mod10 +4%, mod5 +2%, mod2 +2% |
| L15 o c136 (H13) | 3% / 3% | **b** (R2 0.84): b in {36, 56, 64, 96} | **b** (R2 0.60): b in {16, 36, 56, 96} [coarser: b mod 20 in {16}, R2 0.81] | - | same |
| L15 o c139 (H13) | 14% / 13% | **b%10** (R2 0.69): b mod 10 in {0} | unexplained (best R2 0.46) | b: mod5 +3% | - |
| L15 o c141 (H13) | 0 / 10% | off (on 0) | unexplained (best R2 0.34) | - | same |
| L15 o c151 (H13) | 11% / 7% | **b** (R2 0.86): b in {44..46, 48..54} | **tens(a,b)** (R2 0.69): a//10 in {5} -> b//10 in {4}; a//10 in {6,7,8,10} -> b//10 in {4,5} | - | same |
| L15 o c153 (H13) | 12% / 8% | **b//10** (R2 0.94): (tens) b in {90..100} | **b** (R2 0.68): b in {91..100} | - | same |
| L15 o c159 (H13) | 11% / 6% | **b%50** (R2 0.63): b mod 50 in {8, 18..19, 28, 38, 48} [coarser: b mod 10 in {8}, R2 0.85] | unexplained (best R2 0.31) | - | same |
| L15 o c167 (H13) | 22% / 1% | **b** (R2 0.62): b in {11, 21, 31..32, 34, 36, 41..42, 56, 61..62, 64, 71..72, 81..84, 87, 89..90, 92, 96} | unexplained (best R2 0.28) | - | same |
| L15 o c168 (H13) | 30% / 22% | **b%10** (R2 0.96): b mod 10 in {3..5} | **b%10** (R2 0.58): b mod 10 in {3..5} | b: mod10 +10%, mod5 +5% | b: mod10 +8%, mod5 +4% |
| L15 o c178 (H13) | 50% / 46% | **b%10** (R2 0.96): b mod 10 in {1..3, 7, 9} | **b%10** (R2 0.77): b mod 10 in {1..3, 7, 9} | b: mod10 +6%, mod5 +5%, mod2 +12% | b: mod10 +5%, mod5 +5%, mod2 +11% |
| L15 o c187 (H13) | 0 / 10% | off (on 0) | **b//10** (R2 0.75): (tens) b in {1..9} | - | b: mod25 +2% |
| L15 o c191 (H13) | 31% / 15% | **b//10** (R2 0.89): (tens) b in {1..30, 100} | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {0,1}; a//10 in {1,2} -> b//10 in {0,1,2}; a//10 in {3,4,5} -> b//10 in {1,2}; a//10 in {8} -> b//10 in {10}; a//10 in {9} -> b//10 in {1}; a//10 in {10} -> b//10 in {1,2,10} | b: mod100 +3%, mod50 +4% | - |

</details>

<details><summary>1 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L15 v c0 (kv3) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>28 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L15 v c17 (kv3) | 50% / 41% | **b%20** (R2 0.99): b mod 20 in {5..14} | **b%20** (R2 0.71): b mod 20 in {5..14} |
| L15 v c35 (kv3) | 35% / 28% | **b** (R2 0.94): b in {20..25, 59..86} | **tens(a,b)** (R2 0.70): a//10 in {2} -> b//10 in {2,6,7}; a//10 in {3} -> b//10 in {2,6,7,8}; a//10 in {4,5,6} -> b//10 in {6,7,8}; a//10 in {7} -> b//10 in {6,7,8,9,10}; a//10 in {8} -> b//10 in {6,7,8,9}; a//10 in {9} -> b//10 in {6,7,8,10}; a//10 in {10} -> b//10 in {2,6,7,8,10} |
| L15 v c38 (kv3) | 23% / 11% | **b** (R2 0.96): b in {19..41} | **tens(a,b)** (R2 0.58): a//10 in {2} -> b//10 in {2}; a//10 in {3,4,5,10} -> b//10 in {2,3}; a//10 in {6,7,8} -> b//10 in {3} |
| L15 v c43 (kv3) | 38% / 26% | **b** (R2 0.95): b in {7, 10, 13, 17, 20, 23, 25..33, 37, 40, 47, 50, 53, 57, 60, 63, 65, 67, 70, 73, 77, 80, 83, 85, 87, 90, 96..100} | **b** (R2 0.56): b in {7, 10, 13, 17, 23, 25..33, 37, 47, 50, 60, 67, 70, 73, 77, 80, 87, 90, 99..100} |
| L15 v c47 (kv3) | 30% / 24% | **b%10** (R2 0.99): b mod 10 in {4..6} | **b%10** (R2 0.66): b mod 10 in {4..6} |
| L15 v c57 (kv3) | 50% / 45% | **b%2** (R2 0.99): b mod 2 in {1} | **b%2** (R2 0.80): b mod 2 in {1} |
| L15 v c59 (kv3) | 37% / 28% | **b//10** (R2 0.87): (tens) b in {20..56} | **tens(a,b)** (R2 0.79): a//10 in {1} -> b//10 in {3,4}; a//10 in {2,3,4,5} -> b//10 in {2,3,4}; a//10 in {6,7,8,9,10} -> b//10 in {2,3,4,5} |
| L15 v c68 (kv3) | 31% / 27% | **b//10** (R2 0.98): (tens) b in {70..100} | **b//10** (R2 0.83): (tens) b in {71..100} |
| L15 v c74 (kv3) | 11% / 10% | **b%10** (R2 0.94): b mod 10 in {5} | **b%10** (R2 0.88): b mod 10 in {5} |
| L15 v c89 (kv3) | 30% / 24% | **b%10** (R2 0.99): b mod 10 in {0..2} | **b%10** (R2 0.72): b mod 10 in {0..2} |
| L15 v c94 (kv3) | 25% / 19% | **b** (R2 0.95): b in {40..65} | **tens(a,b)** (R2 0.66): a//10 in {5,6} -> b//10 in {4,5}; a//10 in {7} -> b//10 in {4,5,6}; a//10 in {8} -> b//10 in {4,5,6,7}; a//10 in {9} -> b//10 in {4,5,6,7,8}; a//10 in {10} -> b//10 in {4,5,6,7,8,9} |
| L15 v c97 (kv3) | 38% / 31% | **b%10** (R2 0.92): b mod 10 in {2..4, 8} | **b%10** (R2 0.75): b mod 10 in {2..4} |
| L15 v c102 (kv3) | 30% / 19% | **b%10** (R2 0.97): b mod 10 in {3, 6, 9} | **b%10** (R2 0.63): b mod 10 in {6, 9} |
| L15 v c107 (kv3) | 33% / 18% | **b** (R2 0.96): b in {43..74} | **tens(a,b)** (R2 0.62): a//10 in {3,4} -> b//10 in {6}; a//10 in {5,6} -> b//10 in {4,5,6}; a//10 in {7} -> b//10 in {5,6}; a//10 in {8,9} -> b//10 in {5,6,7}; a//10 in {10} -> b//10 in {4,5,6,7} |
| L15 v c111 (kv3) | 30% / 27% | **b%10** (R2 0.98): b mod 10 in {0, 8..9} | **b%10** (R2 0.83): b mod 10 in {0, 8..9} |
| L15 v c112 (kv3) | 10% / 10% | **b%10** (R2 1.00): b mod 10 in {2} | **b%10** (R2 0.96): b mod 10 in {2} |
| L15 v c113 (kv3) | 10% / 6% | **b//10** (R2 0.87): (tens) b in {91..100} | **b** (R2 0.70): b in {95..100} |
| L15 v c115 (kv3) | 9% / 5% | **b%10** (R2 0.86): b mod 10 in {9} | unexplained (best R2 0.45) |
| L15 v c127 (kv3) | 28% / 12% | **b%10** (R2 0.91): b mod 10 in {5..7} | unexplained (best R2 0.42) |
| L15 v c131 (kv3) | 44% / 38% | **b** (R2 0.98): b in {1..4, 15..24, 35..44, 55..64, 75..84} [coarser: b mod 20 in {0..4, 15..19}, R2 0.81] | **b** (R2 0.75): b in {1..4, 15..24, 35..44, 55..64, 76..84} [coarser: b mod 20 in {0..4, 16..19}, R2 0.81] |
| L15 v c132 (kv3) | 10% / 7% | **b%10** (R2 0.97): b mod 10 in {5} | **b%10** (R2 0.64): b mod 10 in {5} |
| L15 v c136 (kv3) | 10% / 8% | **b%10** (R2 0.95): b mod 10 in {0} | **b%10** (R2 0.66): b mod 10 in {0} |
| L15 v c141 (kv3) | 26% / 15% | **b** (R2 0.88): b in {7, 15..19, 27, 35..39, 47, 55..59, 67, 77, 87, 95..100} [coarser: b mod 20 in {7, 15..19}, R2 0.86] | **b** (R2 0.50): b in {7, 16..18, 36..39, 47, 57, 77, 96..99} [coarser: b mod 20 in {17..18}, R2 0.86] |
| L15 v c147 (kv3) | 37% / 32% | **b//10** (R2 0.92): (tens) b in {1..37} | **b//10** (R2 0.75): (tens) b in {4..38} |
| L15 v c154 (kv3) | 10% / 9% | **b%10** (R2 0.96): b mod 10 in {8} | **b%10** (R2 0.82): b mod 10 in {8} |
| L15 v c160 (kv3) | 25% / 7% | **b//10** (R2 0.83): (tens) b in {5..29, 100} | **tens(a,b)** (R2 0.51): a//10 in {1,2} -> b//10 in {1}; a//10 in {3,10} -> b//10 in {1,2} |
| L15 v c173 (kv3) | 30% / 21% | **b%10** (R2 0.80): b mod 10 in {1, 3, 7} | unexplained (best R2 0.48) |
| L15 v c192 (kv3) | 20% / 10% | **b//10** (R2 0.91): (tens) b in {10..28, 100} | **tens(a,b)** (R2 0.59): a//10 in {1,9} -> b//10 in {1}; a//10 in {2,3,4,5,10} -> b//10 in {1,2} |

</details>

<details><summary>1 q components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L15 q c138 (H13) | 100% / 100% | always | same |

</details>

<details><summary>1 k components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L15 k c8 (kv3) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L15 k c13 (kv3) | 100% / 100% | always | same |

</details>

<details><summary>1 k components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L15 k c48 (kv3) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L15H18 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.96 / 0.96 | 0.04 / 0.04 |  |  |  |
| op | 0.97 / 0.96 | 0.02 / 0.03 | 0.01 / 0.02 |  |  |
| b | 0.97 / 0.96 | 0.02 / 0.02 | 0.01 / 0.01 | 0.01 / 0.01 |  |
| = | 0.82 / 0.90 | 0.03 / 0.02 | 0.03 / 0.01 | 0.06 / 0.02 | 0.06 / 0.04 |

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L15 o c446 (H18) | 100% / 100% | always | same | - | b: mod100 +3% |

</details>

<details><summary>2 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L15 v c199 (kv4) | 100% / 0 | always | off (on 0) |
| L15 v c238 (kv4) | 0 / 100% | off (on 0) | always |

</details>

<details><summary>13 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L15 v c29 (kv4) | 0 / 0 | off (on 0) | same |
| L15 v c46 (kv4) | 65% / 1% | unexplained (best R2 0.40) | unexplained (best R2 0.16) |
| L15 v c63 (kv4) | 0 / 19% | off (on 0) | **tens(a,b)** (R2 0.75): a//10 in {3} -> b//10 in {1}; a//10 in {4,5,9} -> b//10 in {1,2}; a//10 in {6,7,8,10} -> b//10 in {1,2,3} |
| L15 v c65 (kv4) | 19% / 17% | **b//10** (R2 0.97): (tens) b in {1..19} | **b//10** (R2 0.84): (tens) b in {1..18} |
| L15 v c70 (kv4) | 16% / 10% | **b%20** (R2 0.71): b mod 20 in {6..8, 17} | unexplained (best R2 0.37) |
| L15 v c73 (kv4) | 25% / 12% | **b** (R2 0.87): b in {64..89} | **tens(a,b)** (R2 0.52): a//10 in {4,5} -> b//10 in {7}; a//10 in {6,7,9} -> b//10 in {7,8}; a//10 in {8,10} -> b//10 in {6,7,8} |
| L15 v c78 (kv4) | 37% / 25% | **b** (R2 0.83): b in {1..14, 44..53, 88, 90..100} | **b** (R2 0.50): b in {10..11, 45..51, 89..100} |
| L15 v c90 (kv4) | 10% / 30% | **b%10** (R2 0.77): b mod 10 in {4} | **res//10** (R2 0.54): (tens) res in {-1..30, 99} |
| L15 v c109 (kv4) | 20% / 13% | **b%10** (R2 0.67): b mod 10 in {0, 2} | unexplained (best R2 0.45) |
| L15 v c126 (kv4) | 39% / 41% | **b%10** (R2 0.92): b mod 10 in {2, 6..8} | unexplained (best R2 0.42) |
| L15 v c134 (kv4) | 3% / 1% | **b** (R2 0.71): b in {25..26, 50} | unexplained (best R2 0.13) |
| L15 v c153 (kv4) | 4% / 3% | **b** (R2 0.71): b in {1..4} | unexplained (best R2 0.39) |
| L15 v c224 (kv4) | 2% / 13% | unexplained (best R2 0.29) | unexplained (best R2 0.48) |

</details>

<details><summary>1 v components, main position `=` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L15 v c189 (kv4) | 92% / 99% | unexplained (best R2 0.41) | always |

</details>

</details>

<details><summary>L15H28 (3 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 1.00 / 1.00 | 0.00 / 0.00 |  |  |  |
| op | 0.99 / 0.99 | 0.00 / 0.00 | 0.01 / 0.00 |  |  |
| b | 0.54 / 0.92 | 0.00 / 0.00 | 0.39 / 0.07 | 0.07 / 0.01 |  |
| = | 0.85 / 0.95 | 0.02 / 0.01 | 0.00 / 0.00 | 0.12 / 0.04 | 0.01 / 0.01 |

<details><summary>3 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L15 o c46 (H28) | 32% / 0 | unexplained (best R2 0.21) | off (on 0) | - | same |
| L15 o c131 (H28) | 19% / 1% | **a//10** (R2 0.69): (tens) a in {80, 85..99} | **tens(a,b)** (R2 0.53): a//10 in {9} -> b//10 in {9,10} | a: mod100 +2%, mod50 +3% | - |
| L15 o c228 (H28) | 99% / 0 | always | off (on 0) | - | same |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L15 k c81 (kv7) | 10% / 0 | **a** (R2 0.60): a in {86..88, 92..93, 96..98} | off (on 0) |

</details>

</details>

<details><summary>L16H12 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.97 / 0.97 | 0.03 / 0.03 |  |  |  |
| op | 0.95 / 0.97 | 0.03 / 0.03 | 0.02 / 0.01 |  |  |
| b | 0.58 / 0.69 | 0.02 / 0.04 | 0.08 / 0.16 | 0.33 / 0.11 |  |
| = | 0.80 / 0.87 | 0.02 / 0.01 | 0.02 / 0.02 | 0.08 / 0.05 | 0.08 / 0.05 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 o c362 (H12) | 0 / 81% | off (on 0) | unexplained (best R2 0.38) | - | same |

</details>

<details><summary>1 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L16 v c2 (kv3) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L16H14 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.97 / 0.97 | 0.03 / 0.03 |  |  |  |
| op | 0.97 / 0.96 | 0.02 / 0.02 | 0.01 / 0.02 |  |  |
| b | 0.62 / 0.67 | 0.07 / 0.07 | 0.08 / 0.18 | 0.24 / 0.07 |  |
| = | 0.84 / 0.92 | 0.01 / 0.01 | 0.01 / 0.01 | 0.04 / 0.02 | 0.09 / 0.04 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 o c334 (H14) | 0 / 5% | off (on 0) | **tens(a,b)** (R2 0.55): a//10 in {9} -> b//10 in {6,7,8} | - | same |

</details>

<details><summary>1 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L16 v c2 (kv3) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L16H21 (52 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 1.00 / 1.00 | 0.00 / 0.00 |  |  |  |
| op | 0.97 / 0.76 | 0.01 / 0.10 | 0.02 / 0.14 |  |  |
| b | 0.98 / 0.97 | 0.00 / 0.01 | 0.02 / 0.01 | 0.00 / 0.01 |  |
| = | 0.03 / 0.19 | 0.91 / 0.58 | 0.03 / 0.04 | 0.03 / 0.19 | 0.00 / 0.00 |

<details><summary>1 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 o c165 (H21) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {47, 49, 51, 53} | - | same |

</details>

<details><summary>51 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 o c16 (H21) | 30% / 30% | **a** (R2 0.93): a in {1..4, 74..100} | **a** (R2 0.77): a in {75..100} | a: mod100 +4%, mod50 +6% | a: mod100 +3%, mod50 +4% |
| L16 o c24 (H21) | 13% / 1% | **a** (R2 0.78): a in {85..99} | unexplained (best R2 0.44) | - | same |
| L16 o c25 (H21) | 20% / 12% | **a%10** (R2 0.91): a mod 10 in {5, 7} | **a** (R2 0.50): a in {35, 55, 65, 67, 75, 77, 85, 87, 95, 97} | a: mod2 +3% | - |
| L16 o c26 (H21) | 20% / 16% | **a%10** (R2 0.99): a mod 10 in {2, 5} | **a%50** (R2 0.74): a mod 50 in {2, 5, 12, 15, 22, 25, 32, 35, 42, 45} [coarser: a mod 10 in {2, 5}, R2 0.88] | a: mod10 +2%, mod5 +4%, mod2 +6% | a: mod5 +3%, mod2 +4% |
| L16 o c28 (H21) | 32% / 28% | **a** (R2 0.83): a in {1..5, 13, 21, 23, 31, 41, 43, 51..65, 71, 81..83, 91} | **a** (R2 0.66): a in {1, 31, 41, 43, 51..65, 71, 73, 81..83, 91, 93, 100} | - | same |
| L16 o c31 (H21) | 24% / 22% | **a** (R2 0.94): a in {69..92} | **a** (R2 0.65): a in {70..92, 100} | a: mod50 +2% | - |
| L16 o c34 (H21) | 29% / 24% | **a%10** (R2 0.97): a mod 10 in {0..2} | **a%10** (R2 0.68): a mod 10 in {0..2} | a: mod10 +7%, mod5 +4% | a: mod10 +6%, mod5 +4% |
| L16 o c35 (H21) | 6% / 6% | **a** (R2 0.87): a in {95..100} | **a** (R2 0.67): a in {97..100} | - | same |
| L16 o c36 (H21) | 5% / 0 | **a** (R2 0.70): a in {96..99} | off (on 0) | - | same |
| L16 o c37 (H21) | 34% / 27% | **a** (R2 0.82): a in {4..5, 9..11, 13..17, 19..20, 24..25, 30, 34..35, 40, 44..45, 50, 54..55, 60, 64..65, 70, 74..75, 80, 84..85, 90, 94..95, 100} [coarser: a mod 10 in {0, 4..5}, R2 0.80] | **a%50** (R2 0.67): a mod 50 in {0, 5, 10, 14..15, 20, 24..25, 30, 35, 40, 44..45} [coarser: a mod 5 in {0}, R2 0.81] | a: mod5 +5% | a: mod5 +6% |
| L16 o c41 (H21) | 28% / 23% | **a** (R2 0.96): a in {23..50} | **a** (R2 0.73): a in {25..47, 50} | a: mod50 +4% | a: mod50 +2% |
| L16 o c44 (H21) | 4% / 2% | **a** (R2 0.51): a in {6, 36, 46} | unexplained (best R2 0.38) | - | same |
| L16 o c45 (H21) | 39% / 35% | **a%10** (R2 0.96): a mod 10 in {1..3, 7} | **a%10** (R2 0.79): a mod 10 in {1..3, 7} | a: mod5 +9%, mod2 +5% | a: mod5 +7%, mod2 +3% |
| L16 o c48 (H21) | 47% / 31% | **a%50** (R2 0.74): a mod 50 in {3, 5..19, 21, 23, 27, 29, 37, 39, 47, 49} | **a%50** (R2 0.47): a mod 50 in {7, 9, 11, 13..15, 17, 19, 21, 27, 29, 37, 39, 47, 49} | a: mod2 +4% | a: mod2 +3% |
| L16 o c52 (H21) | 31% / 30% | **a//10** (R2 0.85): (tens) a in {61..89, 91..92} | **a//10** (R2 0.71): (tens) a in {61..89, 91..92} | a: mod100 +3%, mod50 +3% | a: mod100 +2%, mod50 +3% |
| L16 o c53 (H21) | 41% / 38% | **a%10** (R2 0.97): a mod 10 in {1, 3, 6, 9} | **a%10** (R2 0.79): a mod 10 in {1, 3, 6, 9} | a: mod10 +5%, mod5 +2%, mod2 +21% | a: mod10 +4%, mod5 +2%, mod2 +18% |
| L16 o c72 (H21) | 8% / 7% | **a** (R2 0.81): a in {65..69, 85..89} | **a** (R2 0.63): a in {65..69, 85..89} | - | same |
| L16 o c82 (H21) | 9% / 8% | **a** (R2 0.83): a in {65..74} | **a** (R2 0.65): a in {65..74} | - | same |
| L16 o c86 (H21) | 0 / 0 | off (on 0) | same | - | same |
| L16 o c87 (H21) | 45% / 43% | **a//10** (R2 0.89): (tens) a in {1..45} | **a//10** (R2 0.83): (tens) a in {2..45} | a: mod100 +8%, mod50 +5% | a: mod100 +3%, mod50 +3% |
| L16 o c89 (H21) | 0 / 2% | off (on 0) | **a** (R2 0.92): a in {99..100} | - | same |
| L16 o c90 (H21) | 35% / 23% | **a** (R2 0.93): a in {2..4, 16..24, 35..44, 61..64, 76..84} | **a** (R2 0.69): a in {21..24, 35..44, 61, 76..84} | a: mod20 +8% | a: mod20 +5% |
| L16 o c94 (H21) | 17% / 15% | **a** (R2 0.85): a in {55..71} | **a** (R2 0.73): a in {55..69, 71} | - | same |
| L16 o c97 (H21) | 31% / 29% | **a%10** (R2 0.97): a mod 10 in {5..7} | **a%10** (R2 0.78): a mod 10 in {5..7} | a: mod10 +9%, mod5 +6% | a: mod10 +8%, mod5 +5% |
| L16 o c98 (H21) | 44% / 34% | **a** (R2 0.88): a in {1..14, 65..79, 85..100} | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {0,1,9,10}; a//10 in {1} -> b//10 in {0,9,10}; a//10 in {2} -> b//10 in {10}; a//10 in {3,4,5} -> b//10 in {9}; a//10 in {6} -> b//10 in {5}; a//10 in {7} -> b//10 in {0,3,4,5,6,7,9,10}; a//10 in {8} -> b//10 in {0,1,4,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | a: mod100 +3%, mod50 +7% | a: mod100 +3%, mod50 +5%, mod25 +3%, mod20 +2% |
| L16 o c101 (H21) | 10% / 4% | **a%10** (R2 0.95): a mod 10 in {8} | unexplained (best R2 0.46) | - | same |
| L16 o c116 (H21) | 22% / 19% | **a%10** (R2 0.73): a mod 10 in {4, 6} | **a** (R2 0.61): a in {16, 24, 26, 34, 36, 44, 46, 54, 56, 64, 66, 74..76, 84, 86, 94..96, 100} [coarser: a mod 20 in {4, 6, 14, 16}, R2 0.82] | a: mod2 +2% | - |
| L16 o c117 (H21) | 30% / 26% | **a%10** (R2 0.99): a mod 10 in {3..5} | **a%10** (R2 0.76): a mod 10 in {3..5} | a: mod10 +8%, mod5 +5%, mod2 +3% | a: mod10 +6%, mod5 +4%, mod2 +2% |
| L16 o c120 (H21) | 17% / 16% | **a** (R2 0.84): a in {47..64} | **a** (R2 0.73): a in {49..61, 64, 100} | - | same |
| L16 o c121 (H21) | 39% / 35% | **a%10** (R2 0.97): a mod 10 in {2..4, 8} | **a%10** (R2 0.78): a mod 10 in {2..4} | a: mod10 +9%, mod5 +5%, mod2 +8% | a: mod10 +8%, mod5 +4%, mod2 +7% |
| L16 o c129 (H21) | 61% / 58% | **a** (R2 0.85): a in {2, 4, 6, 8..12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 38, 40, 42, 44, 46, 48..76, 78, 80, 82, 86, 88, 90, 92, 94, 96, 98, 100} | **a** (R2 0.65): a in {2, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 38, 40, 42, 44, 46, 48..76, 78, 80, 82, 86, 88, 90, 92, 98, 100} | a: mod50 +2%, mod2 +15% | a: mod2 +14% |
| L16 o c131 (H21) | 2% / 1% | **a** (R2 0.54): a in {25} | unexplained (best R2 0.22) | - | same |
| L16 o c132 (H21) | 1% / 1% | unexplained (best R2 0.17) | **a//10** (R2 0.64): (tens) a in {100} | - | same |
| L16 o c137 (H21) | 11% / 10% | **a%10** (R2 0.91): a mod 10 in {4} | **a%20** (R2 0.56): a mod 20 in {4, 14} [coarser: a mod 10 in {4}, R2 0.86] | a: mod5 +2% | - |
| L16 o c138 (H21) | 37% / 46% | **a** (R2 0.94): a in {27, 29..64} | **a** (R2 0.72): a in {4, 26..64} | a: mod100 +6%, mod50 +7% | a: mod100 +3%, mod50 +4% |
| L16 o c142 (H21) | 29% / 19% | **a%10** (R2 0.96): a mod 10 in {6..8} | **a** (R2 0.65): a in {16, 26, 28, 36, 38, 46, 56..58, 66..68, 76..78, 86..88, 96..98} [coarser: a mod 10 in {6..8}, R2 0.81] | a: mod10 +3%, mod5 +3% | - |
| L16 o c144 (H21) | 50% / 44% | **a%20** (R2 1.00): a mod 20 in {5..14} | **a** (R2 0.83): a in {14, 25..34, 45..54, 65..74, 85..94} [coarser: a mod 20 in {5..14}, R2 0.83] | a: mod20 +22%, mod4 +5% | a: mod20 +14%, mod4 +4% |
| L16 o c154 (H21) | 10% / 7% | **a%10** (R2 0.99): a mod 10 in {9} | **a%50** (R2 0.60): a mod 50 in {9, 19, 29, 39, 49} [coarser: a mod 10 in {9}, R2 0.89] | a: mod5 +3% | - |
| L16 o c155 (H21) | 12% / 11% | **a** (R2 0.77): a in {85..96} | **a** (R2 0.69): a in {83..95} | - | same |
| L16 o c156 (H21) | 9% / 3% | **a%10** (R2 0.94): a mod 10 in {4} | unexplained (best R2 0.36) | - | same |
| L16 o c163 (H21) | 9% / 6% | **a%10** (R2 0.92): a mod 10 in {2} | **a** (R2 0.66): a in {22, 32, 52, 62, 72, 82, 92} [coarser: a mod 20 in {12}, R2 0.82] | - | same |
| L16 o c164 (H21) | 16% / 10% | **a** (R2 0.88): a in {38..53} | **a** (R2 0.59): a in {41, 43..44, 46..47, 49..52} | - | same |
| L16 o c180 (H21) | 30% / 29% | **a%10** (R2 1.00): a mod 10 in {7..9} | **a%10** (R2 0.82): a mod 10 in {7..9} | a: mod10 +9%, mod5 +3%, mod2 +3% | a: mod10 +7%, mod5 +3%, mod2 +2% |
| L16 o c195 (H21) | 45% / 36% | **a%20** (R2 0.89): a mod 20 in {0, 8..10, 15..19} | **a** (R2 0.68): a in {10, 18..20, 28..30, 38..40, 48..50, 55..60, 68..70, 75..80, 88..90, 95..100} [coarser: a mod 20 in {0, 8..10, 15..19}, R2 0.89] | a: mod20 +5%, mod10 +7%, mod5 +2%, mod4 +2% | a: mod20 +4%, mod10 +7%, mod5 +2%, mod4 +2% |
| L16 o c219 (H21) | 23% / 22% | **a** (R2 0.94): a in {1..23} | **a//10** (R2 0.84): (tens) a in {1..21} | a: mod100 +3%, mod50 +3% | - |
| L16 o c233 (H21) | 0 / 0 | off (on 0) | same | - | same |
| L16 o c244 (H21) | 1% / 0 | unexplained (best R2 0.18) | off (on 0) | - | same |
| L16 o c251 (H21) | 10% / 14% | **a%10** (R2 0.95): a mod 10 in {8} | **a%50** (R2 0.65): a mod 50 in {8, 18, 28, 38, 48..49} [coarser: a mod 10 in {8}, R2 0.89] | a: mod5 +3%, mod2 +2% | a: mod5 +2% |
| L16 o c257 (H21) | 9% / 10% | **a%10** (R2 0.91): a mod 10 in {1} | **a%10** (R2 0.77): a mod 10 in {1} | a: mod5 +3% | a: mod5 +3% |
| L16 o c284 (H21) | 11% / 4% | **a** (R2 0.90): a in {1..11} | unexplained (best R2 0.46) | - | same |
| L16 o c299 (H21) | 16% / 2% | **a** (R2 0.70): a in {10..26, 28} | unexplained (best R2 0.38) | - | same |

</details>

<details><summary>20 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L16 v c25 (kv5) | 40% / 37% | **a** (R2 0.99): a in {4, 24..52, 65..67, 85..91} | **a** (R2 0.94): a in {24..52, 85..91} |
| L16 v c28 (kv5) | 4% / 3% | **a** (R2 0.73): a in {10, 50, 98, 100} | **a** (R2 0.54): a in {50, 100} [coarser: a mod 50 in {0}, R2 0.84] |
| L16 v c31 (kv5) | 14% / 15% | **a** (R2 0.99): a in {65..78} | **a** (R2 1.00): a in {65..79} |
| L16 v c40 (kv5) | 22% / 16% | **a%10** (R2 0.85): a mod 10 in {6, 8} | **a%20** (R2 0.74): a mod 20 in {6, 8, 16} [coarser: a mod 10 in {6, 8}, R2 0.90] |
| L16 v c43 (kv5) | 26% / 23% | **a%20** (R2 0.95): a mod 20 in {15..19} | **a** (R2 0.93): a in {15, 35..39, 55..59, 75..79, 95..100} [coarser: a mod 20 in {15..19}, R2 0.86] |
| L16 v c61 (kv5) | 30% / 30% | **a%10** (R2 1.00): a mod 10 in {3, 6, 9} | same |
| L16 v c66 (kv5) | 8% / 8% | **a** (R2 0.94): a in {92, 94..100} | **a** (R2 0.94): a in {94..100} |
| L16 v c68 (kv5) | 19% / 15% | **a** (R2 0.95): a in {1..4, 55..65, 80..84} | **a** (R2 0.85): a in {1, 55..65, 80..84} |
| L16 v c70 (kv5) | 39% / 39% | **a** (R2 1.00): a in {15..44, 76..84} | same |
| L16 v c78 (kv5) | 20% / 20% | **a%10** (R2 1.00): a mod 10 in {3..4} | same |
| L16 v c80 (kv5) | 28% / 25% | **a%10** (R2 0.94): a mod 10 in {0, 3, 7} | **a%10** (R2 0.86): a mod 10 in {0, 3, 7} |
| L16 v c103 (kv5) | 27% / 21% | **a** (R2 0.94): a in {7..14, 30..34, 50..54, 70..74, 90..94} [coarser: a mod 20 in {10..14}, R2 0.89] | **a%20** (R2 0.78): a mod 20 in {10..14} |
| L16 v c106 (kv5) | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {4} | **a%10** (R2 0.98): a mod 10 in {4} |
| L16 v c119 (kv5) | 20% / 19% | **a%10** (R2 1.00): a mod 10 in {2, 9} | **a%10** (R2 0.97): a mod 10 in {2, 9} |
| L16 v c121 (kv5) | 30% / 27% | **a%10** (R2 0.99): a mod 10 in {7..9} | **a%10** (R2 0.87): a mod 10 in {7..9} |
| L16 v c176 (kv5) | 1% / 1% | **a** (R2 0.98): a in {1} | **a** (R2 1.00): a in {1} |
| L16 v c185 (kv5) | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {8} | same |
| L16 v c200 (kv5) | 5% / 5% | **a%20** (R2 1.00): a mod 20 in {4} | **a%20** (R2 0.99): a mod 20 in {4} |
| L16 v c229 (kv5) | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {1} | **a%10** (R2 0.96): a mod 10 in {1} |
| L16 v c245 (kv5) | 4% / 4% | **a** (R2 0.99): a in {34..37} | **a** (R2 0.95): a in {34..37} |

</details>

<details><summary>1 q components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L16 q c136 (H21) | 100% / 100% | always | same |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L16 k c5 (kv5) | 100% / 100% | always | same |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L16 k c122 (kv5) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L16H22 (33 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.83 / 0.83 | 0.17 / 0.17 |  |  |  |
| op | 0.65 / 0.56 | 0.31 / 0.42 | 0.03 / 0.02 |  |  |
| b | 0.28 / 0.53 | 0.20 / 0.34 | 0.41 / 0.06 | 0.11 / 0.07 |  |
| = | 0.61 / 0.40 | 0.15 / 0.12 | 0.13 / 0.10 | 0.04 / 0.32 | 0.06 / 0.06 |

<details><summary>26 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 o c58 (H22) | 3% / 5% | **a** (R2 0.90): a in {89..91} | **a** (R2 1.00): a in {70, 80..81, 90..91} | - | same |
| L16 o c67 (H22) | 1% / 0 | **a** (R2 1.00): a in {11} | off (on 0) | - | same |
| L16 o c71 (H22) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {75} | - | same |
| L16 o c73 (H22) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {46..47} | - | same |
| L16 o c113 (H22) | 0 / 7% | off (on 0) | **a** (R2 0.99): a in {9, 29, 39, 49, 59, 69, 79} | - | same |
| L16 o c119 (H22) | 1% / 12% | **a** (R2 1.00): a in {7} | **a** (R2 0.99): a in {7, 47, 57, 63, 66..69, 71..73, 77} | - | same |
| L16 o c170 (H22) | 3% / 7% | **a** (R2 0.98): a in {6, 16, 96} | **a** (R2 1.00): a in {6, 36, 46, 56, 66, 76, 96} | - | same |
| L16 o c181 (H22) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {50} | - | same |
| L16 o c188 (H22) | 2% / 4% | **a** (R2 1.00): a in {10, 90} | **a** (R2 1.00): a in {70, 73..75} | - | same |
| L16 o c197 (H22) | 6% / 28% | **a** (R2 0.99): a in {50, 55, 60, 90, 95, 100} | **a** (R2 1.00): a in {5, 10, 20, 25, 30, 35, 40, 45..53, 55..57, 60..61, 65, 70, 75, 80, 85, 90, 100} | - | a: mod5 +3% |
| L16 o c237 (H22) | 3% / 13% | **a** (R2 0.94): a in {60..62} | **a** (R2 0.99): a in {55, 57..65, 67..69} | - | same |
| L16 o c243 (H22) | 9% / 12% | **a** (R2 1.00): a in {81..89} | **a** (R2 1.00): a in {78..89} | - | same |
| L16 o c248 (H22) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {51} | - | same |
| L16 o c249 (H22) | 6% / 1% | **a** (R2 0.99): a in {90, 92, 96, 98..100} | **a//10** (R2 1.00): (tens) a in {100} | - | same |
| L16 o c261 (H22) | 1% / 7% | **a** (R2 1.00): a in {55} | **a** (R2 0.99): a in {43, 53..57, 63} | - | same |
| L16 o c268 (H22) | 3% / 8% | **a** (R2 0.99): a in {1, 21, 91} | **a** (R2 1.00): a in {21, 31, 41..42, 51, 61, 71, 81} | - | same |
| L16 o c314 (H22) | 1% / 3% | **a** (R2 1.00): a in {97} | **a** (R2 1.00): a in {7, 47, 77} | - | same |
| L16 o c321 (H22) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {74..75} | - | same |
| L16 o c329 (H22) | 3% / 7% | **a** (R2 1.00): a in {3, 13, 93} | **a** (R2 1.00): a in {33, 43, 53, 63, 73, 83, 93} | - | same |
| L16 o c337 (H22) | 11% / 10% | **a%10** (R2 0.91): a mod 10 in {2} | **a%10** (R2 1.00): a mod 10 in {2} | - | same |
| L16 o c355 (H22) | 1% / 3% | **a** (R2 1.00): a in {18} | **a** (R2 1.00): a in {8, 38, 58} [coarser: a mod 50 in {8}, R2 0.83] | - | same |
| L16 o c378 (H22) | 0 / 1% | off (on 0) | **a** (R2 0.99): a in {8} | - | same |
| L16 o c387 (H22) | 0 / 5% | off (on 0) | **a** (R2 0.98): a in {34, 49, 53..54, 74} | - | same |
| L16 o c392 (H22) | 1% / 1% | **a** (R2 1.00): a in {11} | **a** (R2 1.00): a in {51} | - | same |
| L16 o c463 (H22) | 1% / 0 | **a** (R2 1.00): a in {1} | off (on 0) | - | same |
| L16 o c492 (H22) | 8% / 9% | **a** (R2 0.98): a in {4, 14, 24, 44, 64, 74, 84, 94} [coarser: a mod 20 in {4, 14}, R2 0.85] | **a%20** (R2 0.90): a mod 20 in {4, 14} [coarser: a mod 10 in {4}, R2 0.89] | - | same |

</details>

<details><summary>6 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 o c56 (H22) | 0 / 21% | off (on 0) | **a** (R2 0.71): a in {63..69, 71..85, 87} | - | same |
| L16 o c61 (H22) | 15% / 11% | **a//10** (R2 0.71): (tens) a in {1..14, 16} | **a//10** (R2 0.82): (tens) a in {1..10} | - | same |
| L16 o c91 (H22) | 1% / 37% | unexplained (best R2 0.10) | **a** (R2 0.66): a in {22..24, 26..59, 61..63} | - | same |
| L16 o c194 (H22) | 0 / 0 | off (on 0) | same | - | same |
| L16 o c267 (H22) | 0 / 8% | off (on 0) | **a** (R2 0.78): a in {91..99} | - | same |
| L16 o c322 (H22) | 1% / 2% | unexplained (best R2 0.36) | unexplained (best R2 0.18) | - | same |

</details>

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 o c160 (H22) | 51% / 74% | **a//10** (R2 0.91): (tens) a in {51..100} | **tens(a,b)** (R2 0.79): a//10 in {0,1,2,3} -> b//10 in {5,6,7,8,9,10}; a//10 in {4} -> b//10 in {2,5,6,7,8,9,10}; a//10 in {5,6,7,8,9} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,10} | a: mod100 +3% | a: mod100 +2% |

</details>

<details><summary>20 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L16 v c25 (kv5) | 40% / 37% | **a** (R2 0.99): a in {4, 24..52, 65..67, 85..91} | **a** (R2 0.94): a in {24..52, 85..91} |
| L16 v c28 (kv5) | 4% / 3% | **a** (R2 0.73): a in {10, 50, 98, 100} | **a** (R2 0.54): a in {50, 100} [coarser: a mod 50 in {0}, R2 0.84] |
| L16 v c31 (kv5) | 14% / 15% | **a** (R2 0.99): a in {65..78} | **a** (R2 1.00): a in {65..79} |
| L16 v c40 (kv5) | 22% / 16% | **a%10** (R2 0.85): a mod 10 in {6, 8} | **a%20** (R2 0.74): a mod 20 in {6, 8, 16} [coarser: a mod 10 in {6, 8}, R2 0.90] |
| L16 v c43 (kv5) | 26% / 23% | **a%20** (R2 0.95): a mod 20 in {15..19} | **a** (R2 0.93): a in {15, 35..39, 55..59, 75..79, 95..100} [coarser: a mod 20 in {15..19}, R2 0.86] |
| L16 v c61 (kv5) | 30% / 30% | **a%10** (R2 1.00): a mod 10 in {3, 6, 9} | same |
| L16 v c66 (kv5) | 8% / 8% | **a** (R2 0.94): a in {92, 94..100} | **a** (R2 0.94): a in {94..100} |
| L16 v c68 (kv5) | 19% / 15% | **a** (R2 0.95): a in {1..4, 55..65, 80..84} | **a** (R2 0.85): a in {1, 55..65, 80..84} |
| L16 v c70 (kv5) | 39% / 39% | **a** (R2 1.00): a in {15..44, 76..84} | same |
| L16 v c78 (kv5) | 20% / 20% | **a%10** (R2 1.00): a mod 10 in {3..4} | same |
| L16 v c80 (kv5) | 28% / 25% | **a%10** (R2 0.94): a mod 10 in {0, 3, 7} | **a%10** (R2 0.86): a mod 10 in {0, 3, 7} |
| L16 v c103 (kv5) | 27% / 21% | **a** (R2 0.94): a in {7..14, 30..34, 50..54, 70..74, 90..94} [coarser: a mod 20 in {10..14}, R2 0.89] | **a%20** (R2 0.78): a mod 20 in {10..14} |
| L16 v c106 (kv5) | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {4} | **a%10** (R2 0.98): a mod 10 in {4} |
| L16 v c119 (kv5) | 20% / 19% | **a%10** (R2 1.00): a mod 10 in {2, 9} | **a%10** (R2 0.97): a mod 10 in {2, 9} |
| L16 v c121 (kv5) | 30% / 27% | **a%10** (R2 0.99): a mod 10 in {7..9} | **a%10** (R2 0.87): a mod 10 in {7..9} |
| L16 v c176 (kv5) | 1% / 1% | **a** (R2 0.98): a in {1} | **a** (R2 1.00): a in {1} |
| L16 v c185 (kv5) | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {8} | same |
| L16 v c200 (kv5) | 5% / 5% | **a%20** (R2 1.00): a mod 20 in {4} | **a%20** (R2 0.99): a mod 20 in {4} |
| L16 v c229 (kv5) | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {1} | **a%10** (R2 0.96): a mod 10 in {1} |
| L16 v c245 (kv5) | 4% / 4% | **a** (R2 0.99): a in {34..37} | **a** (R2 0.95): a in {34..37} |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L16 k c5 (kv5) | 100% / 100% | always | same |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L16 k c122 (kv5) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L16H27 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.96 / 0.96 | 0.04 / 0.04 |  |  |  |
| op | 0.95 / 0.93 | 0.02 / 0.05 | 0.03 / 0.03 |  |  |
| b | 0.45 / 0.62 | 0.09 / 0.09 | 0.13 / 0.17 | 0.32 / 0.12 |  |
| = | 0.82 / 0.90 | 0.04 / 0.02 | 0.03 / 0.01 | 0.03 / 0.02 | 0.07 / 0.05 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 o c296 (H27) | 99% / 3% | always | unexplained (best R2 0.06) | a: mod5 +3% | - |

</details>

<details><summary>1 k components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L16 k c31 (kv6) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

</details>

<details><summary>L16H30 (3 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.53 / 0.53 | 0.47 / 0.47 |  |  |  |
| op | 0.59 / 0.68 | 0.04 / 0.03 | 0.37 / 0.29 |  |  |
| b | 0.38 / 0.48 | 0.01 / 0.01 | 0.08 / 0.08 | 0.53 / 0.43 |  |
| = | 0.32 / 0.38 | 0.01 / 0.01 | 0.01 / 0.01 | 0.03 / 0.03 | 0.63 / 0.57 |

<details><summary>1 o components, main position `&lt;BOS&gt;`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 o c393 (H30) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same | - | same |

</details>

<details><summary>2 o components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 o c17 (H30) | 24% / 25% | **a** (R2 0.98): a in {14, 28, 30..31, 35, 40, 42, 49..52, 54, 56..58, 60, 70, 74..75, 90, 92..93, 99..100} | **a** (R2 0.97): a in {14, 28, 30..31, 35, 40, 42, 49..52, 54, 56..58, 60, 70, 74..75, 90, 92..93, 99..100} | - | same |
| L16 o c207 (H30) | 100% / 100% | always | same | - | same |

</details>

<details><summary>1 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L16 v c1 (kv7) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

</details>

<details><summary>L17H9 (14 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.98 / 0.98 | 0.02 / 0.02 |  |  |  |
| op | 0.91 / 0.98 | 0.00 / 0.01 | 0.09 / 0.01 |  |  |
| b | 0.16 / 0.28 | 0.03 / 0.04 | 0.41 / 0.56 | 0.41 / 0.12 |  |
| = | 0.92 / 0.91 | 0.00 / 0.00 | 0.01 / 0.01 | 0.03 / 0.01 | 0.03 / 0.06 |

<details><summary>1 o components, main position `&lt;BOS&gt;`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L17 o c17 (H9) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same | - | same |

</details>

<details><summary>13 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L17 o c13 (H9) | 0 / 87% | off (on 0) | unexplained (best R2 0.18) | - | same |
| L17 o c33 (H9) | 14% / 1% | **a** (R2 0.83): a in {86, 88..99} | **tens(a,b)** (R2 0.54): a//10 in {9} -> b//10 in {9} | a: mod100 +3%, mod50 +4%, mod25 +3%, mod20 +5% | - |
| L17 o c37 (H9) | 0 / 10% | off (on 0) | **a//10** (R2 0.74): (tens) a in {89..99} | - | a: mod100 +2%, mod50 +3%, mod25 +3%, mod20 +4% |
| L17 o c46 (H9) | 100% / 0 | always | off (on 0) | - | same |
| L17 o c79 (H9) | 0 / 3% | off (on 0) | unexplained (best R2 0.46) | - | same |
| L17 o c84 (H9) | 8% / 0 | unexplained (best R2 0.49) | off (on 0) | - | same |
| L17 o c126 (H9) | 0 / 0 | off (on 0) | same | - | same |
| L17 o c214 (H9) | 0 / 2% | off (on 0) | **a** (R2 0.70): a in {98..99} | - | same |
| L17 o c236 (H9) | 0 / 0 | off (on 0) | same | - | same |
| L17 o c291 (H9) | 0 / 9% | off (on 0) | **a//10** (R2 0.93): (tens) a in {1..9} | - | same |
| L17 o c312 (H9) | 0 / 9% | off (on 0) | **a//10** (R2 0.92): (tens) a in {1..9} | - | same |
| L17 o c322 (H9) | 2% / 0 | **a** (R2 0.58): a in {90} | off (on 0) | - | same |
| L17 o c412 (H9) | 99% / 100% | always | same | - | res: mod50 +2%, mod25 +3%, mod20 +4% |

</details>

<details><summary>2 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L17 v c11 (kv2) | 0 / 99% | off (on 0) | always |
| L17 v c143 (kv2) | 0 / 9% | off (on 0) | **a//10** (R2 1.00): (tens) a in {1..9} |

</details>

<details><summary>1 q components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L17 q c106 (H9) | 99% / 99% | always | same |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L17 k c9 (kv2) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L18H1 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.85 / 0.85 | 0.15 / 0.15 |  |  |  |
| op | 0.89 / 0.90 | 0.02 / 0.02 | 0.09 / 0.08 |  |  |
| b | 0.83 / 0.88 | 0.03 / 0.02 | 0.06 / 0.02 | 0.08 / 0.08 |  |
| = | 0.94 / 0.97 | 0.01 / 0.00 | 0.01 / 0.01 | 0.02 / 0.01 | 0.02 / 0.01 |

<details><summary>1 o components, main position `&lt;BOS&gt;`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L18 o c442 (H1) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same | - | same |

</details>

</details>

<details><summary>L18H7 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 1.00 / 1.00 | 0.00 / 0.00 |  |  |  |
| op | 0.89 / 0.99 | 0.00 / 0.01 | 0.11 / 0.01 |  |  |
| b | 0.18 / 0.95 | 0.00 / 0.01 | 0.81 / 0.04 | 0.01 / 0.00 |  |
| = | 0.88 / 0.84 | 0.00 / 0.01 | 0.02 / 0.03 | 0.01 / 0.02 | 0.09 / 0.11 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L18 o c3 (H7) | 100% / 0 | always | off (on 0) | - | same |

</details>

<details><summary>5 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L18 v c35 (kv1) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {47, 49..51, 53} |
| L18 v c97 (kv1) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {51, 53} |
| L18 v c146 (kv1) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {65, 70..71, 73..75} |
| L18 v c183 (kv1) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {70} |
| L18 v c211 (kv1) | 0 / 8% | off (on 0) | **a** (R2 1.00): a in {49, 51..57} |

</details>

<details><summary>1 q components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L18 q c92 (H7) | 100% / 100% | always | same |

</details>

<details><summary>1 q components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L18 q c104 (H7) | 100% / 100% | always | same |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L18 k c95 (kv1) | 100% / 100% | always | same |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L18 k c10 (kv1) | 100% / 82% | always | **a** (R2 0.97): a in {1..11, 25..26, 29..95, 99..100} |

</details>

</details>

<details><summary>L18H13 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.88 / 0.88 | 0.12 / 0.12 |  |  |  |
| op | 0.94 / 0.86 | 0.04 / 0.03 | 0.02 / 0.11 |  |  |
| b | 0.90 / 0.67 | 0.06 / 0.21 | 0.02 / 0.09 | 0.03 / 0.04 |  |
| = | 0.98 / 0.99 | 0.01 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.01 / 0.01 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L18 o c81 (H13) | 0 / 7% | off (on 0) | unexplained (best R2 0.46) | - | same |

</details>

</details>

<details><summary>L18H16 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 1.00 / 1.00 | 0.00 / 0.00 |  |  |  |
| op | 0.99 / 0.98 | 0.01 / 0.01 | 0.01 / 0.01 |  |  |
| b | 0.97 / 0.97 | 0.00 / 0.00 | 0.02 / 0.02 | 0.01 / 0.00 |  |
| = | 0.70 / 0.54 | 0.12 / 0.14 | 0.02 / 0.03 | 0.11 / 0.24 | 0.04 / 0.05 |

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L18 o c496 (H16) | 22% / 92% | unexplained (best R2 0.27) | unexplained (best R2 0.44) | - | same |

</details>

<details><summary>2 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L18 v c43 (kv4) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {74} |
| L18 v c196 (kv4) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {70} |

</details>

</details>

<details><summary>L18H17 (3 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.96 / 0.96 | 0.04 / 0.04 |  |  |  |
| op | 0.96 / 0.76 | 0.03 / 0.12 | 0.01 / 0.12 |  |  |
| b | 0.98 / 0.93 | 0.01 / 0.04 | 0.01 / 0.02 | 0.00 / 0.01 |  |
| = | 0.91 / 0.78 | 0.02 / 0.04 | 0.02 / 0.02 | 0.04 / 0.12 | 0.02 / 0.03 |

<details><summary>3 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L18 o c299 (H17) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {51} | - | same |
| L18 o c436 (H17) | 0 / 7% | off (on 0) | **a** (R2 1.00): a in {51..57} | - | same |
| L18 o c495 (H17) | 1% / 3% | **a** (R2 1.00): a in {90} | **a** (R2 1.00): a in {70, 74..75} | - | same |

</details>

<details><summary>2 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L18 v c43 (kv4) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {74} |
| L18 v c196 (kv4) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {70} |

</details>

<details><summary>2 q components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L18 q c124 (H17) | 3% / 0 | **a** (R2 1.00): a in {50, 90, 100} [coarser: a mod 50 in {0}, R2 0.83] | off (on 0) |
| L18 q c130 (H17) | 1% / 24% | **a** (R2 1.00): a in {90} | **a** (R2 0.99): a in {41..43, 46..47, 49, 51..55, 57, 59..62, 65, 70..71, 73..75, 77, 80} |

</details>

</details>

<details><summary>L18H18 (10 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 1.00 / 1.00 | 0.00 / 0.00 |  |  |  |
| op | 0.99 / 1.00 | 0.00 / 0.00 | 0.00 / 0.00 |  |  |
| b | 0.99 / 0.99 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |  |
| = | 0.18 / 0.37 | 0.26 / 0.19 | 0.04 / 0.02 | 0.51 / 0.40 | 0.02 / 0.02 |

<details><summary>10 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L18 o c166 (H18) | 18% / 19% | **tens(a,b)** (R2 0.73): a//10 in {0} -> b//10 in {4,5}; a//10 in {1,2,3,6,9,10} -> b//10 in {5}; a//10 in {4} -> b//10 in {0,4,5}; a//10 in {5} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.74): a//10 in {0,1,2,3,4,6} -> b//10 in {5}; a//10 in {5} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {4,5} | - | same |
| L18 o c210 (H18) | 4% / 7% | unexplained (best R2 0.23) | unexplained (best R2 0.22) | - | same |
| L18 o c230 (H18) | 1% / 2% | unexplained (best R2 0.22) | **a** (R2 0.50): a in {70} | - | same |
| L18 o c285 (H18) | 1% / 0 | unexplained (best R2 0.15) | off (on 0) | - | same |
| L18 o c293 (H18) | 0 / 0 | off (on 0) | same | - | same |
| L18 o c324 (H18) | 21% / 18% | **tens(a,b)** (R2 0.57): a//10 in {0,1,2,3,4,5,6,8} -> b//10 in {9,10}; a//10 in {7} -> b//10 in {10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.55): a//10 in {0,4,5,6,8} -> b//10 in {9,10}; a//10 in {1,2,3} -> b//10 in {10}; a//10 in {7} -> b//10 in {9}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | - | same |
| L18 o c327 (H18) | 0 / 0 | off (on 0) | same | - | same |
| L18 o c336 (H18) | 0 / 2% | off (on 0) | **a** (R2 0.64): a in {99} | - | same |
| L18 o c387 (H18) | 0 / 0 | off (on 0) | same | - | same |
| L18 o c416 (H18) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>2 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L18 v c43 (kv4) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {74} |
| L18 v c196 (kv4) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {70} |

</details>

</details>

<details><summary>L18H30 (85 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 1.00 / 1.00 | 0.00 / 0.00 |  |  |  |
| op | 0.93 / 0.48 | 0.03 / 0.28 | 0.04 / 0.25 |  |  |
| b | 0.88 / 0.98 | 0.00 / 0.00 | 0.10 / 0.01 | 0.02 / 0.00 |  |
| = | 0.52 / 0.33 | 0.10 / 0.07 | 0.06 / 0.05 | 0.25 / 0.49 | 0.07 / 0.07 |

<details><summary>70 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L18 o c14 (H30) | 0 / 20% | off (on 0) | **a** (R2 1.00): a in {11, 21, 26, 30..31, 35..37, 39..41, 45..46, 51, 60..61, 65, 71, 81, 91} | - | same |
| L18 o c37 (H30) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {34..35, 45} | - | same |
| L18 o c42 (H30) | 0 / 19% | off (on 0) | **a** (R2 1.00): a in {46..47, 49..63, 66..67} | - | same |
| L18 o c49 (H30) | 0 / 40% | off (on 0) | **a** (R2 1.00): a in {12, 16, 20..28, 30..38, 40..50, 52, 54..57, 64, 66, 72, 76} | - | a: mod2 +4% |
| L18 o c53 (H30) | 0 / 28% | off (on 0) | **a** (R2 1.00): a in {5, 9..10, 13..21, 25, 29..30, 35, 39, 45, 49..50, 55, 59..60, 65, 69, 75, 79, 90} | - | same |
| L18 o c56 (H30) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {44, 70, 74..75, 77, 80} | - | same |
| L18 o c62 (H30) | 0 / 8% | off (on 0) | **a** (R2 1.00): a in {46, 48, 50, 52, 54, 56..58} | - | same |
| L18 o c74 (H30) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {68..71} | - | same |
| L18 o c80 (H30) | 0 / 12% | off (on 0) | **a** (R2 1.00): a in {18..29} | - | same |
| L18 o c88 (H30) | 0 / 22% | off (on 0) | **a** (R2 1.00): a in {40..41, 61, 64..65, 71..73, 75..76, 78..89} | - | same |
| L18 o c96 (H30) | 0 / 15% | off (on 0) | **a** (R2 1.00): a in {7, 17, 27, 37, 41..45, 47, 57, 67, 77, 87, 97} [coarser: a mod 50 in {7, 17, 27, 37, 47}, R2 0.80] | - | same |
| L18 o c97 (H30) | 0 / 19% | off (on 0) | **a** (R2 1.00): a in {12, 21..23, 32, 41..43, 51..53, 62..63, 71..73, 82..83, 92} | - | a: mod10 +2% |
| L18 o c106 (H30) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {51} | - | same |
| L18 o c114 (H30) | 0 / 7% | off (on 0) | **a** (R2 1.00): a in {25..31} | - | same |
| L18 o c134 (H30) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {49, 69} | - | same |
| L18 o c138 (H30) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {19} | - | same |
| L18 o c142 (H30) | 0 / 20% | off (on 0) | **a** (R2 1.00): a in {8, 13, 16..19, 27..28, 36..38, 47..48, 57..58, 67..68, 78, 87..88} | - | same |
| L18 o c144 (H30) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {58..63} | - | same |
| L18 o c146 (H30) | 0 / 9% | off (on 0) | **a** (R2 1.00): a in {32, 48, 51..54, 56, 64, 72} | - | same |
| L18 o c147 (H30) | 0 / 13% | off (on 0) | **a** (R2 1.00): a in {12, 20..28, 32, 72, 82} | - | same |
| L18 o c151 (H30) | 0 / 7% | off (on 0) | **a** (R2 1.00): a in {46..47, 49, 52..54, 60} | - | same |
| L18 o c152 (H30) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {77..78} | - | same |
| L18 o c154 (H30) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {50, 80} | - | same |
| L18 o c158 (H30) | 0 / 19% | off (on 0) | **a//10** (R2 0.94): (tens) a in {60..77, 79} | - | same |
| L18 o c159 (H30) | 0 / 13% | off (on 0) | **a** (R2 1.00): a in {6, 16, 26, 36, 56, 63..68, 86, 96} | - | same |
| L18 o c160 (H30) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {15, 35, 55, 85} | - | same |
| L18 o c164 (H30) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {21, 51, 53, 61} | - | same |
| L18 o c171 (H30) | 0 / 7% | off (on 0) | **a** (R2 1.00): a in {70, 75..77, 79..81} | - | same |
| L18 o c173 (H30) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {6, 15..16, 24..26} | - | same |
| L18 o c187 (H30) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {54} | - | same |
| L18 o c188 (H30) | 0 / 9% | off (on 0) | **a%20** (R2 0.90): a mod 20 in {9, 19} [coarser: a mod 10 in {9}, R2 0.89] | - | same |
| L18 o c190 (H30) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {29, 31, 49, 51, 53} | - | same |
| L18 o c196 (H30) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {13..14, 24, 64, 74} [coarser: a mod 50 in {14, 24}, R2 0.89] | - | same |
| L18 o c208 (H30) | 0 / 12% | off (on 0) | **a** (R2 1.00): a in {4, 14, 24, 34, 54, 73..76, 79, 84, 94} | - | same |
| L18 o c212 (H30) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {77} | - | same |
| L18 o c218 (H30) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {18, 24, 28, 48} | - | same |
| L18 o c221 (H30) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {41..42} | - | same |
| L18 o c227 (H30) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {16} | - | same |
| L18 o c229 (H30) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {38, 58} | - | same |
| L18 o c233 (H30) | 0 / 8% | off (on 0) | **a** (R2 1.00): a in {15, 23, 25, 33..35, 43, 85} | - | same |
| L18 o c236 (H30) | 0 / 7% | off (on 0) | **a** (R2 1.00): a in {38..44} | - | same |
| L18 o c239 (H30) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {31} | - | same |
| L18 o c266 (H30) | 0 / 10% | off (on 0) | **a** (R2 1.00): a in {35, 45, 54..59, 65, 75} | - | same |
| L18 o c271 (H30) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {34, 94} | - | same |
| L18 o c272 (H30) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {61..63} | - | same |
| L18 o c290 (H30) | 0 / 25% | off (on 0) | **a** (R2 0.99): a in {9, 19, 21, 29, 38..39, 41..44, 46..49, 51, 53..54, 58..59, 69, 79, 81, 97..99} | - | same |
| L18 o c292 (H30) | 0 / 7% | off (on 0) | **a** (R2 1.00): a in {24, 43..44, 48, 54, 63..64} | - | same |
| L18 o c304 (H30) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {49, 51, 53} | - | same |
| L18 o c306 (H30) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {20, 40} | - | same |
| L18 o c308 (H30) | 0 / 7% | off (on 0) | **a** (R2 1.00): a in {17, 19, 23, 27, 29, 79, 89} | - | same |
| L18 o c310 (H30) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {41} | - | same |
| L18 o c325 (H30) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {58, 78, 88} | - | same |
| L18 o c346 (H30) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {24, 74, 84} [coarser: a mod 50 in {24}, R2 0.83] | - | same |
| L18 o c356 (H30) | 0 / 9% | off (on 0) | **a** (R2 1.00): a in {21..23, 31, 41, 61, 71, 81, 91} [coarser: a mod 50 in {21, 31, 41}, R2 0.82] | - | same |
| L18 o c361 (H30) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {45} | - | same |
| L18 o c362 (H30) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {14, 16..18} | - | same |
| L18 o c382 (H30) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {13..16} | - | same |
| L18 o c407 (H30) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {25, 30} | - | same |
| L18 o c412 (H30) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {25, 55, 65, 75, 85} | - | same |
| L18 o c413 (H30) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {58..60, 62} | - | same |
| L18 o c418 (H30) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {34, 54, 74, 78} | - | same |
| L18 o c422 (H30) | 0 / 9% | off (on 0) | **a** (R2 1.00): a in {37, 43, 45..49, 57, 67} | - | same |
| L18 o c433 (H30) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {75} | - | same |
| L18 o c444 (H30) | 0 / 10% | off (on 0) | **a** (R2 1.00): a in {6..7, 16..17, 26..27, 47, 76..77, 87} | - | same |
| L18 o c446 (H30) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {30, 100} | - | same |
| L18 o c453 (H30) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {14, 16} | - | same |
| L18 o c454 (H30) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {60..61} | - | same |
| L18 o c455 (H30) | 0 / 2% | off (on 0) | **a** (R2 0.99): a in {56..57} | - | same |
| L18 o c462 (H30) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {60..62, 65..67} | - | same |
| L18 o c487 (H30) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {54, 64} | - | same |

</details>

<details><summary>15 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L18 o c66 (H30) | 30% / 36% | **tens(a,b)** (R2 0.77): a//10 in {0} -> b//10 in {0,1,2,3,4}; a//10 in {1} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,1,2,3,4,10}; a//10 in {3,4} -> b//10 in {0,1,2}; a//10 in {5} -> b//10 in {1,2}; a//10 in {10} -> b//10 in {1} | **tens(a,b)** (R2 0.82): a//10 in {0,3} -> b//10 in {0,1,2,3,4}; a//10 in {1} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,1,2,3,4,5,6,7,10}; a//10 in {4} -> b//10 in {0,1,2}; a//10 in {5} -> b//10 in {1,2}; a//10 in {6,7,8} -> b//10 in {1} | - | same |
| L18 o c73 (H30) | 34% / 33% | **tens(a,b)** (R2 0.66): a//10 in {0,1,2} -> b//10 in {5,6,7}; a//10 in {3,4,8,9,10} -> b//10 in {6}; a//10 in {5} -> b//10 in {0,5,6,7}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,6,7} | **tens(a,b)** (R2 0.66): a//10 in {0,1,5} -> b//10 in {5,6,7}; a//10 in {2,3,4} -> b//10 in {6,7}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,2,3,6,7}; a//10 in {9,10} -> b//10 in {6} | - | same |
| L18 o c82 (H30) | 18% / 27% | **tens(a,b)** (R2 0.71): a//10 in {0,1,2,5,9,10} -> b//10 in {3}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,10}; a//10 in {4} -> b//10 in {0,1,3} | **tens(a,b)** (R2 0.73): a//10 in {0,1,2} -> b//10 in {3,4}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,3,4,5,9}; a//10 in {5,6,7} -> b//10 in {3} | - | same |
| L18 o c107 (H30) | 44% / 42% | **tens(a,b)** (R2 0.83): a//10 in {0,1,7} -> b//10 in {7,8,9,10}; a//10 in {2,3,4,5,6} -> b//10 in {8,9,10}; a//10 in {8,9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.74): a//10 in {0,1,2,3} -> b//10 in {7,8,9,10}; a//10 in {4,5,6,7} -> b//10 in {8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,7,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,10} | - | b: mod100 +3%, mod50 +2% |
| L18 o c120 (H30) | 2% / 3% | **tens(a,b)** (R2 0.62): a//10 in {0,1,2,3,4,5,6,7,8,9} -> b//10 in {10}; a//10 in {10} -> b//10 in {0,10} | unexplained (best R2 0.49) | - | same |
| L18 o c135 (H30) | 9% / 11% | unexplained (best R2 0.39) | unexplained (best R2 0.34) | - | same |
| L18 o c137 (H30) | 13% / 11% | **tens(a,b)** (R2 0.63): a//10 in {0,3} -> b//10 in {3,4}; a//10 in {1,2} -> b//10 in {4}; a//10 in {4} -> b//10 in {0,1,2,3,4,5} | **tens(a,b)** (R2 0.57): a//10 in {0,1,2,3,8} -> b//10 in {4}; a//10 in {4} -> b//10 in {0,1,2,3,4,7,8,9,10} | - | same |
| L18 o c157 (H30) | 0 / 1% | off (on 0) | **tens(a,b)** (R2 0.88): a//10 in {0,1,2,3,4,5,6,10} -> b//10 in {10} | - | same |
| L18 o c195 (H30) | 6% / 10% | **units(a,b)** (R2 0.54): a%10 in {2} -> b%10 in {4}; a%10 in {4} -> b%10 in {1,2,4,6} | **units(a,b)** (R2 0.62): a%10 in {0,5,6,8,9} -> b%10 in {4}; a%10 in {4} -> b%10 in {3,4,8,9} | - | same |
| L18 o c200 (H30) | 12% / 12% | **units(a,b)** (R2 0.74): a%10 in {0,1,2,3,6,7} -> b%10 in {5}; a%10 in {5} -> b%10 in {1,3,5,6,7} | **units(a,b)** (R2 0.66): a%10 in {0,1,2,3,4,6,7,9} -> b%10 in {5}; a%10 in {5} -> b%10 in {3,4,5,7,9} | - | same |
| L18 o c313 (H30) | 22% / 21% | **units(a,b)** (R2 0.89): a%10 in {0} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {1,3,4,6,7,8,9} -> b%10 in {0}; a%10 in {2,5} -> b%10 in {0,5} | **units(a,b)** (R2 0.83): a%10 in {0} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {1,2,3,4,6,7,8,9} -> b%10 in {0}; a%10 in {5} -> b%10 in {0,5} | b: mod5 +3% | a: mod5 +4%; b: mod5 +6% |
| L18 o c355 (H30) | 4% / 13% | **tens(a,b)** (R2 0.59): a//10 in {0} -> b//10 in {7}; a//10 in {7} -> b//10 in {0,1,7} | **tens(a,b)** (R2 0.70): a//10 in {0,1,2,3,4,5} -> b//10 in {7}; a//10 in {7} -> b//10 in {0,1,2,3,4,7,10} | - | same |
| L18 o c402 (H30) | 11% / 15% | **tens(a,b)** (R2 0.52): a//10 in {0,1,2,3,5} -> b//10 in {4}; a//10 in {4} -> b//10 in {0,1,2,4} | unexplained (best R2 0.49) | - | same |
| L18 o c427 (H30) | 5% / 6% | **tens(a,b)** (R2 0.51): a//10 in {0,1} -> b//10 in {2}; a//10 in {2} -> b//10 in {0,1,2} | unexplained (best R2 0.37) | - | same |
| L18 o c445 (H30) | 7% / 10% | **units(a,b)** (R2 0.51): a%10 in {1,2,5} -> b%10 in {3}; a%10 in {3} -> b%10 in {1,2,3} | **units(a,b)** (R2 0.54): a%10 in {1,2,7,8,9} -> b%10 in {3}; a%10 in {3} -> b%10 in {3,7,8} | - | same |

</details>

<details><summary>55 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L18 v c7 (kv7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {61} |
| L18 v c21 (kv7) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {15, 35, 55, 65, 75} |
| L18 v c23 (kv7) | 2% / 8% | **a** (R2 0.67): a in {70} | **a** (R2 1.00): a in {67..74} |
| L18 v c24 (kv7) | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {5} | same |
| L18 v c25 (kv7) | 5% / 11% | **a** (R2 0.78): a in {64..67} | **a//10** (R2 0.91): (tens) a in {56, 60..69} |
| L18 v c28 (kv7) | 10% / 13% | **a** (R2 0.88): a in {79..86, 89} | **a** (R2 1.00): a in {76, 78..89} |
| L18 v c38 (kv7) | 26% / 28% | **a//10** (R2 0.86): (tens) a in {2..26} | **a//10** (R2 0.96): (tens) a in {2..29} |
| L18 v c39 (kv7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {18} |
| L18 v c49 (kv7) | 2% / 11% | **a** (R2 0.54): a in {62, 82} | **a** (R2 0.99): a in {12, 22, 32, 42, 52, 61..62, 72, 82..83, 92} [coarser: a mod 50 in {12, 22, 32, 42}, R2 0.85] |
| L18 v c50 (kv7) | 0 / 2% | off (on 0) | **a** (R2 0.99): a in {52, 72} |
| L18 v c54 (kv7) | 9% / 9% | **a%10** (R2 0.88): a mod 10 in {1} | **a%20** (R2 0.90): a mod 20 in {1, 11} [coarser: a mod 10 in {1}, R2 0.89] |
| L18 v c60 (kv7) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {24, 48, 64} |
| L18 v c61 (kv7) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {42..45} |
| L18 v c68 (kv7) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {41, 61} |
| L18 v c78 (kv7) | 1% / 9% | unexplained (best R2 0.13) | **a** (R2 0.98): a in {16, 24, 32, 36, 48, 52, 56, 64, 72} |
| L18 v c80 (kv7) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {60..62} |
| L18 v c86 (kv7) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {30..33} |
| L18 v c89 (kv7) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {20, 70, 80} [coarser: a mod 50 in {20}, R2 0.83] |
| L18 v c90 (kv7) | 12% / 16% | **a//10** (R2 0.87): (tens) a in {39..49} | **a** (R2 1.00): a in {34, 36..50} |
| L18 v c91 (kv7) | 0 / 11% | off (on 0) | **a** (R2 0.99): a in {68..77, 79} |
| L18 v c98 (kv7) | 9% / 22% | **a%10** (R2 0.88): a mod 10 in {6} | **a** (R2 0.99): a in {6, 16..18, 24, 26..28, 36..37, 45..48, 56..57, 66..67, 76, 86..87, 96} |
| L18 v c107 (kv7) | 1% / 15% | unexplained (best R2 0.39) | **a** (R2 1.00): a in {12, 20..25, 32, 42..44, 52, 62, 72, 82} |
| L18 v c121 (kv7) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {13..17} |
| L18 v c124 (kv7) | 0 / 17% | off (on 0) | **a** (R2 1.00): a in {9, 19, 21, 29, 31, 39, 41, 47, 49, 51, 59, 61, 69, 79, 81, 89, 99} [coarser: a mod 20 in {1, 9, 19}, R2 0.80] |
| L18 v c126 (kv7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {52} |
| L18 v c128 (kv7) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {15..19, 52} |
| L18 v c130 (kv7) | 10% / 12% | **a%10** (R2 1.00): a mod 10 in {9} | **a%50** (R2 0.90): a mod 50 in {9, 19, 29, 39, 49} [coarser: a mod 10 in {9}, R2 0.83] |
| L18 v c139 (kv7) | 8% / 14% | **a** (R2 0.93): a in {72..79} | **a** (R2 1.00): a in {69..82} |
| L18 v c147 (kv7) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {58..63} |
| L18 v c148 (kv7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {55} |
| L18 v c157 (kv7) | 11% / 15% | **a** (R2 0.80): a in {23..34} | **a** (R2 0.98): a in {22..35, 37} |
| L18 v c158 (kv7) | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {7} | same |
| L18 v c159 (kv7) | 23% / 24% | **a** (R2 0.99): a in {41..63} | **a** (R2 1.00): a in {41..63, 66} |
| L18 v c165 (kv7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {62} |
| L18 v c175 (kv7) | 10% / 10% | **a%10** (R2 0.97): a mod 10 in {4} | **a%10** (R2 1.00): a mod 10 in {4} |
| L18 v c176 (kv7) | 8% / 13% | **a** (R2 0.83): a in {36..42} | **a** (R2 1.00): a in {32..44} |
| L18 v c180 (kv7) | 0 / 2% | off (on 0) | **a%50** (R2 1.00): a mod 50 in {16} |
| L18 v c188 (kv7) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {24, 47, 49, 51, 53} |
| L18 v c202 (kv7) | 1% / 8% | unexplained (best R2 0.10) | **a** (R2 1.00): a in {33..40} |
| L18 v c207 (kv7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {24} |
| L18 v c212 (kv7) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {20..23} |
| L18 v c213 (kv7) | 10% / 11% | **a%10** (R2 1.00): a mod 10 in {8} | **a%10** (R2 0.91): a mod 10 in {8} |
| L18 v c226 (kv7) | 1% / 5% | unexplained (best R2 0.16) | **a** (R2 1.00): a in {25..29} |
| L18 v c227 (kv7) | 9% / 13% | **a** (R2 0.80): a in {30..38} | **a** (R2 1.00): a in {28..40} |
| L18 v c230 (kv7) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {19..20, 78..81} |
| L18 v c231 (kv7) | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {3} | same |
| L18 v c233 (kv7) | 2% / 12% | unexplained (best R2 0.32) | **a** (R2 1.00): a in {18..29} |
| L18 v c234 (kv7) | 2% / 9% | **a** (R2 0.65): a in {50} | **a** (R2 1.00): a in {25, 46..53} |
| L18 v c237 (kv7) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {30..31} |
| L18 v c239 (kv7) | 0 / 7% | off (on 0) | **a%50** (R2 0.92): a mod 50 in {23..25} |
| L18 v c242 (kv7) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {18, 78} |
| L18 v c243 (kv7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {80} |
| L18 v c245 (kv7) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {60..62} |
| L18 v c246 (kv7) | 15% / 19% | **a%10** (R2 0.81): a mod 10 in {0, 5} [coarser: a mod 5 in {0}, R2 0.84] | **a%5** (R2 0.92): a mod 5 in {0} |
| L18 v c251 (kv7) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {25, 74..77} |

</details>

<details><summary>2 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L18 v c37 (kv7) | 32% / 27% | **b//10** (R2 0.92): (tens) b in {68, 70..100} | **b//10** (R2 0.80): (tens) b in {73..100} |
| L18 v c229 (kv7) | 24% / 18% | **b** (R2 0.85): b in {55..77, 79} | **b//10** (R2 0.66): (tens) b in {58..75, 77} |

</details>

<details><summary>1 q components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L18 q c135 (H30) | 0 / 98% | off (on 0) | always |

</details>

</details>

<details><summary>L18H31 (6 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.99 / 0.99 | 0.01 / 0.01 |  |  |  |
| op | 0.94 / 0.96 | 0.02 / 0.02 | 0.04 / 0.03 |  |  |
| b | 0.89 / 0.93 | 0.01 / 0.01 | 0.09 / 0.03 | 0.02 / 0.03 |  |
| = | 0.43 / 0.36 | 0.13 / 0.16 | 0.04 / 0.14 | 0.35 / 0.26 | 0.05 / 0.08 |

<details><summary>6 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L18 o c94 (H31) | 0 / 0 | off (on 0) | same | - | same |
| L18 o c130 (H31) | 1% / 1% | unexplained (best R2 0.11) | unexplained (best R2 0.18) | - | same |
| L18 o c234 (H31) | 0 / 0 | off (on 0) | same | - | same |
| L18 o c320 (H31) | 0 / 0 | off (on 0) | same | - | same |
| L18 o c423 (H31) | 1% / 11% | unexplained (best R2 0.25) | **tens(a,b)** (R2 0.64): a//10 in {0,1,2,3} -> b//10 in {7}; a//10 in {7} -> b//10 in {0,1,2,3,4} | - | same |
| L18 o c486 (H31) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>55 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L18 v c7 (kv7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {61} |
| L18 v c21 (kv7) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {15, 35, 55, 65, 75} |
| L18 v c23 (kv7) | 2% / 8% | **a** (R2 0.67): a in {70} | **a** (R2 1.00): a in {67..74} |
| L18 v c24 (kv7) | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {5} | same |
| L18 v c25 (kv7) | 5% / 11% | **a** (R2 0.78): a in {64..67} | **a//10** (R2 0.91): (tens) a in {56, 60..69} |
| L18 v c28 (kv7) | 10% / 13% | **a** (R2 0.88): a in {79..86, 89} | **a** (R2 1.00): a in {76, 78..89} |
| L18 v c38 (kv7) | 26% / 28% | **a//10** (R2 0.86): (tens) a in {2..26} | **a//10** (R2 0.96): (tens) a in {2..29} |
| L18 v c39 (kv7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {18} |
| L18 v c49 (kv7) | 2% / 11% | **a** (R2 0.54): a in {62, 82} | **a** (R2 0.99): a in {12, 22, 32, 42, 52, 61..62, 72, 82..83, 92} [coarser: a mod 50 in {12, 22, 32, 42}, R2 0.85] |
| L18 v c50 (kv7) | 0 / 2% | off (on 0) | **a** (R2 0.99): a in {52, 72} |
| L18 v c54 (kv7) | 9% / 9% | **a%10** (R2 0.88): a mod 10 in {1} | **a%20** (R2 0.90): a mod 20 in {1, 11} [coarser: a mod 10 in {1}, R2 0.89] |
| L18 v c60 (kv7) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {24, 48, 64} |
| L18 v c61 (kv7) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {42..45} |
| L18 v c68 (kv7) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {41, 61} |
| L18 v c78 (kv7) | 1% / 9% | unexplained (best R2 0.13) | **a** (R2 0.98): a in {16, 24, 32, 36, 48, 52, 56, 64, 72} |
| L18 v c80 (kv7) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {60..62} |
| L18 v c86 (kv7) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {30..33} |
| L18 v c89 (kv7) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {20, 70, 80} [coarser: a mod 50 in {20}, R2 0.83] |
| L18 v c90 (kv7) | 12% / 16% | **a//10** (R2 0.87): (tens) a in {39..49} | **a** (R2 1.00): a in {34, 36..50} |
| L18 v c91 (kv7) | 0 / 11% | off (on 0) | **a** (R2 0.99): a in {68..77, 79} |
| L18 v c98 (kv7) | 9% / 22% | **a%10** (R2 0.88): a mod 10 in {6} | **a** (R2 0.99): a in {6, 16..18, 24, 26..28, 36..37, 45..48, 56..57, 66..67, 76, 86..87, 96} |
| L18 v c107 (kv7) | 1% / 15% | unexplained (best R2 0.39) | **a** (R2 1.00): a in {12, 20..25, 32, 42..44, 52, 62, 72, 82} |
| L18 v c121 (kv7) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {13..17} |
| L18 v c124 (kv7) | 0 / 17% | off (on 0) | **a** (R2 1.00): a in {9, 19, 21, 29, 31, 39, 41, 47, 49, 51, 59, 61, 69, 79, 81, 89, 99} [coarser: a mod 20 in {1, 9, 19}, R2 0.80] |
| L18 v c126 (kv7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {52} |
| L18 v c128 (kv7) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {15..19, 52} |
| L18 v c130 (kv7) | 10% / 12% | **a%10** (R2 1.00): a mod 10 in {9} | **a%50** (R2 0.90): a mod 50 in {9, 19, 29, 39, 49} [coarser: a mod 10 in {9}, R2 0.83] |
| L18 v c139 (kv7) | 8% / 14% | **a** (R2 0.93): a in {72..79} | **a** (R2 1.00): a in {69..82} |
| L18 v c147 (kv7) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {58..63} |
| L18 v c148 (kv7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {55} |
| L18 v c157 (kv7) | 11% / 15% | **a** (R2 0.80): a in {23..34} | **a** (R2 0.98): a in {22..35, 37} |
| L18 v c158 (kv7) | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {7} | same |
| L18 v c159 (kv7) | 23% / 24% | **a** (R2 0.99): a in {41..63} | **a** (R2 1.00): a in {41..63, 66} |
| L18 v c165 (kv7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {62} |
| L18 v c175 (kv7) | 10% / 10% | **a%10** (R2 0.97): a mod 10 in {4} | **a%10** (R2 1.00): a mod 10 in {4} |
| L18 v c176 (kv7) | 8% / 13% | **a** (R2 0.83): a in {36..42} | **a** (R2 1.00): a in {32..44} |
| L18 v c180 (kv7) | 0 / 2% | off (on 0) | **a%50** (R2 1.00): a mod 50 in {16} |
| L18 v c188 (kv7) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {24, 47, 49, 51, 53} |
| L18 v c202 (kv7) | 1% / 8% | unexplained (best R2 0.10) | **a** (R2 1.00): a in {33..40} |
| L18 v c207 (kv7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {24} |
| L18 v c212 (kv7) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {20..23} |
| L18 v c213 (kv7) | 10% / 11% | **a%10** (R2 1.00): a mod 10 in {8} | **a%10** (R2 0.91): a mod 10 in {8} |
| L18 v c226 (kv7) | 1% / 5% | unexplained (best R2 0.16) | **a** (R2 1.00): a in {25..29} |
| L18 v c227 (kv7) | 9% / 13% | **a** (R2 0.80): a in {30..38} | **a** (R2 1.00): a in {28..40} |
| L18 v c230 (kv7) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {19..20, 78..81} |
| L18 v c231 (kv7) | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {3} | same |
| L18 v c233 (kv7) | 2% / 12% | unexplained (best R2 0.32) | **a** (R2 1.00): a in {18..29} |
| L18 v c234 (kv7) | 2% / 9% | **a** (R2 0.65): a in {50} | **a** (R2 1.00): a in {25, 46..53} |
| L18 v c237 (kv7) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {30..31} |
| L18 v c239 (kv7) | 0 / 7% | off (on 0) | **a%50** (R2 0.92): a mod 50 in {23..25} |
| L18 v c242 (kv7) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {18, 78} |
| L18 v c243 (kv7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {80} |
| L18 v c245 (kv7) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {60..62} |
| L18 v c246 (kv7) | 15% / 19% | **a%10** (R2 0.81): a mod 10 in {0, 5} [coarser: a mod 5 in {0}, R2 0.84] | **a%5** (R2 0.92): a mod 5 in {0} |
| L18 v c251 (kv7) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {25, 74..77} |

</details>

<details><summary>2 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L18 v c37 (kv7) | 32% / 27% | **b//10** (R2 0.92): (tens) b in {68, 70..100} | **b//10** (R2 0.80): (tens) b in {73..100} |
| L18 v c229 (kv7) | 24% / 18% | **b** (R2 0.85): b in {55..77, 79} | **b//10** (R2 0.66): (tens) b in {58..75, 77} |

</details>

</details>

<details><summary>L19H7 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.97 / 0.97 | 0.03 / 0.03 |  |  |  |
| op | 0.97 / 0.98 | 0.01 / 0.01 | 0.02 / 0.01 |  |  |
| b | 0.86 / 0.91 | 0.04 / 0.03 | 0.07 / 0.02 | 0.04 / 0.03 |  |
| = | 0.90 / 0.88 | 0.00 / 0.01 | 0.02 / 0.01 | 0.02 / 0.02 | 0.06 / 0.09 |

<details><summary>1 o components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L19 o c23 (H7) | 100% / 100% | always | same | - | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L19 v c208 (kv1) | 93% / 0 | **a** (R2 0.86): a in {1..39, 41..49, 51..54, 56..59, 61..69, 71..79, 81..99} | off (on 0) |

</details>

</details>

<details><summary>L19H8 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.89 / 0.89 | 0.11 / 0.11 |  |  |  |
| op | 0.76 / 0.67 | 0.13 / 0.24 | 0.11 / 0.08 |  |  |
| b | 0.51 / 0.66 | 0.09 / 0.07 | 0.10 / 0.12 | 0.29 / 0.15 |  |
| = | 0.56 / 0.57 | 0.01 / 0.01 | 0.02 / 0.02 | 0.13 / 0.13 | 0.29 / 0.27 |

<details><summary>1 o components, main position `&lt;BOS&gt;`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L19 o c3 (H8) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same | - | same |

</details>

<details><summary>1 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L19 v c0 (kv2) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

</details>

<details><summary>L20H2 (25 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 1.00 / 1.00 | 0.00 / 0.00 |  |  |  |
| op | 0.98 / 0.88 | 0.01 / 0.04 | 0.01 / 0.08 |  |  |
| b | 0.93 / 0.98 | 0.02 / 0.00 | 0.03 / 0.01 | 0.03 / 0.01 |  |
| = | 0.46 / 0.29 | 0.14 / 0.35 | 0.04 / 0.02 | 0.34 / 0.33 | 0.01 / 0.01 |

<details><summary>25 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L20 o c135 (H2) | 1% / 67% | unexplained (best R2 0.06) | unexplained (best R2 0.34) | - | a: mod4 +7%; b: mod4 +2% |
| L20 o c168 (H2) | 0 / 0 | off (on 0) | same | - | same |
| L20 o c179 (H2) | 0 / 3% | off (on 0) | **a** (R2 0.79): a in {97..99} | - | same |
| L20 o c192 (H2) | 6% / 12% | **units(a,b)** (R2 0.54): a%10 in {0,5} -> b%10 in {0,5} | **units(a,b)** (R2 0.56): a%10 in {0} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {5} -> b%10 in {0,5} | - | same |
| L20 o c196 (H2) | 0 / 0 | off (on 0) | same | - | same |
| L20 o c207 (H2) | 0 / 0 | off (on 0) | same | - | same |
| L20 o c243 (H2) | 0 / 2% | off (on 0) | **a** (R2 0.57): a in {45} | - | same |
| L20 o c244 (H2) | 0 / 0 | off (on 0) | same | - | same |
| L20 o c252 (H2) | 4% / 12% | unexplained (best R2 0.15) | unexplained (best R2 0.45) | - | same |
| L20 o c259 (H2) | 0 / 0 | off (on 0) | same | - | same |
| L20 o c273 (H2) | 0 / 0 | off (on 0) | same | - | same |
| L20 o c277 (H2) | 0 / 0 | off (on 0) | same | - | same |
| L20 o c288 (H2) | 4% / 13% | **units(a,b)** (R2 0.78): a%10 in {0} -> b%10 in {5}; a%10 in {5} -> b%10 in {0,5} | **units(a,b)** (R2 0.71): a%10 in {0,7,8,9} -> b%10 in {5}; a%10 in {5} -> b%10 in {0,1,2,3,4,5,6,7,8,9} | - | same |
| L20 o c292 (H2) | 3% / 15% | unexplained (best R2 0.16) | unexplained (best R2 0.37) | - | a: mod4 +3%; b: mod4 +2% |
| L20 o c295 (H2) | 0 / 0 | off (on 0) | same | - | same |
| L20 o c312 (H2) | 0 / 0 | off (on 0) | same | - | same |
| L20 o c316 (H2) | 0 / 0 | off (on 0) | same | - | same |
| L20 o c317 (H2) | 4% / 8% | unexplained (best R2 0.15) | unexplained (best R2 0.35) | - | same |
| L20 o c328 (H2) | 5% / 19% | unexplained (best R2 0.19) | unexplained (best R2 0.40) | a: mod4 +3%; b: mod4 +3% | a: mod4 +10%; b: mod4 +6% |
| L20 o c356 (H2) | 0 / 0 | off (on 0) | same | - | same |
| L20 o c396 (H2) | 2% / 11% | unexplained (best R2 0.48) | unexplained (best R2 0.49) | - | same |
| L20 o c401 (H2) | 1% / 6% | unexplained (best R2 0.10) | unexplained (best R2 0.36) | - | same |
| L20 o c435 (H2) | 0 / 0 | off (on 0) | same | - | same |
| L20 o c479 (H2) | 6% / 15% | **units(a,b)** (R2 0.66): a%10 in {0,5} -> b%10 in {0,5} | **units(a,b)** (R2 0.61): a%10 in {0} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {5} -> b%10 in {0,5} | - | same |
| L20 o c498 (H2) | 1% / 2% | unexplained (best R2 0.07) | unexplained (best R2 0.17) | - | same |

</details>

<details><summary>1 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L20 v c11 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>19 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L20 v c73 (kv0) | 1% / 8% | **a** (R2 0.81): a in {31} | **a** (R2 0.96): a in {21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 50 in {21, 31, 41}, R2 0.83] |
| L20 v c81 (kv0) | 0 / 0 | off (on 0) | same |
| L20 v c85 (kv0) | 1% / 6% | **a** (R2 0.74): a in {52} | **a** (R2 0.82): a in {32, 48, 52..53, 92} |
| L20 v c108 (kv0) | 7% / 12% | **a** (R2 0.54): a in {30, 60, 90} | **a%50** (R2 0.75): a mod 50 in {10, 20, 30, 40, 45} [coarser: a mod 10 in {0}, R2 0.86] |
| L20 v c123 (kv0) | 0 / 2% | off (on 0) | **a** (R2 0.92): a in {97..98} |
| L20 v c129 (kv0) | 0 / 1% | off (on 0) | **a** (R2 0.98): a in {45} |
| L20 v c158 (kv0) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {42} |
| L20 v c164 (kv0) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {54} |
| L20 v c172 (kv0) | 4% / 10% | unexplained (best R2 0.44) | **a%10** (R2 0.92): a mod 10 in {5} |
| L20 v c173 (kv0) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {98} |
| L20 v c174 (kv0) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {70} |
| L20 v c181 (kv0) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {98} |
| L20 v c206 (kv0) | 4% / 11% | **a** (R2 0.56): a in {32, 64} | **a** (R2 0.69): a in {8, 16, 20, 32, 40, 64, 80, 88, 96} |
| L20 v c221 (kv0) | 2% / 6% | **a** (R2 0.77): a in {9, 99} | **a** (R2 0.88): a in {9, 49, 69, 97..99} |
| L20 v c232 (kv0) | 0 / 1% | off (on 0) | **a** (R2 0.86): a in {54} |
| L20 v c237 (kv0) | 0 / 1% | off (on 0) | **a** (R2 0.99): a in {45} |
| L20 v c240 (kv0) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {29..31} |
| L20 v c244 (kv0) | 7% / 13% | **a** (R2 0.56): a in {12, 24, 48, 96} | **a** (R2 0.76): a in {6, 8..9, 12, 16, 24, 32, 36, 48, 64, 72, 96} |
| L20 v c253 (kv0) | 0 / 0 | off (on 0) | same |

</details>

<details><summary>1 q components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L20 q c131 (H2) | 8% / 79% | unexplained (best R2 0.37) | unexplained (best R2 0.49) |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L20 k c58 (kv0) | 3% / 25% | unexplained (best R2 0.12) | **a** (R2 0.69): a in {10, 12, 24, 30..32, 36, 40, 45, 48, 50, 52, 54, 60, 64, 72, 80, 90, 99..100} |

</details>

</details>

<details><summary>L20H3 (6 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.76 / 0.76 | 0.24 / 0.24 |  |  |  |
| op | 0.89 / 0.70 | 0.07 / 0.25 | 0.04 / 0.06 |  |  |
| b | 0.71 / 0.79 | 0.03 / 0.02 | 0.03 / 0.02 | 0.22 / 0.17 |  |
| = | 0.79 / 0.70 | 0.01 / 0.01 | 0.03 / 0.02 | 0.05 / 0.12 | 0.12 / 0.15 |

<details><summary>6 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L20 o c270 (H3) | 1% / 5% | **a** (R2 1.00): a in {52} | **a** (R2 1.00): a in {32, 48, 52..54} | - | same |
| L20 o c290 (H3) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {21, 29..31, 51} | - | same |
| L20 o c349 (H3) | 1% / 5% | **a** (R2 1.00): a in {90} | **a** (R2 1.00): a in {30, 35, 45, 60, 90} | - | same |
| L20 o c386 (H3) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {54} | - | same |
| L20 o c417 (H3) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {31} | - | same |
| L20 o c492 (H3) | 0 / 9% | off (on 0) | **a** (R2 1.00): a in {12, 16, 24, 32, 36, 48, 64, 72, 96} | - | same |

</details>

<details><summary>1 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L20 v c11 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>19 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L20 v c73 (kv0) | 1% / 8% | **a** (R2 0.81): a in {31} | **a** (R2 0.96): a in {21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 50 in {21, 31, 41}, R2 0.83] |
| L20 v c81 (kv0) | 0 / 0 | off (on 0) | same |
| L20 v c85 (kv0) | 1% / 6% | **a** (R2 0.74): a in {52} | **a** (R2 0.82): a in {32, 48, 52..53, 92} |
| L20 v c108 (kv0) | 7% / 12% | **a** (R2 0.54): a in {30, 60, 90} | **a%50** (R2 0.75): a mod 50 in {10, 20, 30, 40, 45} [coarser: a mod 10 in {0}, R2 0.86] |
| L20 v c123 (kv0) | 0 / 2% | off (on 0) | **a** (R2 0.92): a in {97..98} |
| L20 v c129 (kv0) | 0 / 1% | off (on 0) | **a** (R2 0.98): a in {45} |
| L20 v c158 (kv0) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {42} |
| L20 v c164 (kv0) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {54} |
| L20 v c172 (kv0) | 4% / 10% | unexplained (best R2 0.44) | **a%10** (R2 0.92): a mod 10 in {5} |
| L20 v c173 (kv0) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {98} |
| L20 v c174 (kv0) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {70} |
| L20 v c181 (kv0) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {98} |
| L20 v c206 (kv0) | 4% / 11% | **a** (R2 0.56): a in {32, 64} | **a** (R2 0.69): a in {8, 16, 20, 32, 40, 64, 80, 88, 96} |
| L20 v c221 (kv0) | 2% / 6% | **a** (R2 0.77): a in {9, 99} | **a** (R2 0.88): a in {9, 49, 69, 97..99} |
| L20 v c232 (kv0) | 0 / 1% | off (on 0) | **a** (R2 0.86): a in {54} |
| L20 v c237 (kv0) | 0 / 1% | off (on 0) | **a** (R2 0.99): a in {45} |
| L20 v c240 (kv0) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {29..31} |
| L20 v c244 (kv0) | 7% / 13% | **a** (R2 0.56): a in {12, 24, 48, 96} | **a** (R2 0.76): a in {6, 8..9, 12, 16, 24, 32, 36, 48, 64, 72, 96} |
| L20 v c253 (kv0) | 0 / 0 | off (on 0) | same |

</details>

<details><summary>1 q components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L20 q c83 (H3) | 0 / 21% | off (on 0) | **a** (R2 0.99): a in {1..2, 21, 24, 26..32, 36, 45..46, 48, 51..52, 54, 60, 64, 90} |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L20 k c58 (kv0) | 3% / 25% | unexplained (best R2 0.12) | **a** (R2 0.69): a in {10, 12, 24, 30..32, 36, 40, 45, 48, 50, 52, 54, 60, 64, 72, 80, 90, 99..100} |

</details>

</details>

<details><summary>L20H17 (2 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.74 / 0.74 | 0.26 / 0.26 |  |  |  |
| op | 0.80 / 0.81 | 0.03 / 0.04 | 0.17 / 0.15 |  |  |
| b | 0.74 / 0.64 | 0.02 / 0.04 | 0.08 / 0.06 | 0.16 / 0.26 |  |
| = | 0.92 / 0.95 | 0.01 / 0.01 | 0.01 / 0.00 | 0.02 / 0.00 | 0.03 / 0.04 |

<details><summary>2 o components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L20 o c0 (H17) | 89% / 90% | **a** (R2 1.00): a in {1..4, 6, 9, 11..14, 16..19, 21..29, 31..39, 41..49, 51..79, 81..99} | **a** (R2 0.96): a in {1..4, 6, 9, 11..14, 16..29, 31..39, 41..49, 51..79, 81..99} | - | same |
| L20 o c8 (H17) | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | - | same |

</details>

<details><summary>2 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L20 v c25 (kv4) | 5% / 5% | **a** (R2 1.00): a in {1..4, 6} | **a** (R2 0.99): a in {1..4, 6} |
| L20 v c223 (kv4) | 100% / 100% | always | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L20 v c168 (kv4) | 0 / 86% | off (on 0) | **a** (R2 0.96): a in {1..23, 25..29, 31..44, 46..69, 71, 73, 75..77, 79, 81..87, 91..95, 97} |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L20 k c106 (kv4) | 2% / 2% | **a** (R2 1.00): a in {1..2} | same |

</details>

</details>

<details><summary>L20H18 (4 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.96 / 0.96 | 0.04 / 0.04 |  |  |  |
| op | 0.89 / 0.85 | 0.03 / 0.04 | 0.08 / 0.10 |  |  |
| b | 0.83 / 0.89 | 0.03 / 0.02 | 0.09 / 0.04 | 0.05 / 0.05 |  |
| = | 0.96 / 0.98 | 0.00 / 0.00 | 0.01 / 0.00 | 0.01 / 0.00 | 0.01 / 0.01 |

<details><summary>4 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L20 o c74 (H18) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {1..6} | - | same |
| L20 o c81 (H18) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {1} | - | same |
| L20 o c124 (H18) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {1..6} | - | same |
| L20 o c176 (H18) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {1} | - | same |

</details>

<details><summary>2 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L20 v c25 (kv4) | 5% / 5% | **a** (R2 1.00): a in {1..4, 6} | **a** (R2 0.99): a in {1..4, 6} |
| L20 v c223 (kv4) | 100% / 100% | always | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L20 v c168 (kv4) | 0 / 86% | off (on 0) | **a** (R2 0.96): a in {1..23, 25..29, 31..44, 46..69, 71, 73, 75..77, 79, 81..87, 91..95, 97} |

</details>

<details><summary>1 q components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L20 q c95 (H18) | 0 / 9% | off (on 0) | **a//10** (R2 1.00): (tens) a in {1..9} |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L20 k c106 (kv4) | 2% / 2% | **a** (R2 1.00): a in {1..2} | same |

</details>

</details>

<details><summary>L20H19 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.97 / 0.97 | 0.03 / 0.03 |  |  |  |
| op | 0.83 / 0.86 | 0.05 / 0.05 | 0.12 / 0.10 |  |  |
| b | 0.87 / 0.91 | 0.02 / 0.02 | 0.03 / 0.02 | 0.08 / 0.05 |  |
| = | 0.96 / 0.96 | 0.00 / 0.01 | 0.00 / 0.00 | 0.01 / 0.00 | 0.03 / 0.02 |

<details><summary>1 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L20 o c60 (H19) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {26..29} | - | same |

</details>

<details><summary>2 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L20 v c25 (kv4) | 5% / 5% | **a** (R2 1.00): a in {1..4, 6} | **a** (R2 0.99): a in {1..4, 6} |
| L20 v c223 (kv4) | 100% / 100% | always | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L20 v c168 (kv4) | 0 / 86% | off (on 0) | **a** (R2 0.96): a in {1..23, 25..29, 31..44, 46..69, 71, 73, 75..77, 79, 81..87, 91..95, 97} |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L20 k c106 (kv4) | 2% / 2% | **a** (R2 1.00): a in {1..2} | same |

</details>

</details>

<details><summary>L20H21 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.99 / 0.99 | 0.01 / 0.01 |  |  |  |
| op | 0.90 / 0.96 | 0.02 / 0.02 | 0.08 / 0.02 |  |  |
| b | 0.28 / 0.85 | 0.02 / 0.02 | 0.63 / 0.10 | 0.07 / 0.03 |  |
| = | 0.91 / 0.89 | 0.02 / 0.01 | 0.03 / 0.02 | 0.02 / 0.03 | 0.02 / 0.05 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L20 o c2 (H21) | 99% / 0 | always | off (on 0) | - | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L20 v c13 (kv5) | 99% / 0 | always | off (on 0) |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L20 k c26 (kv5) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L21H18 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.87 / 0.87 | 0.13 / 0.13 |  |  |  |
| op | 0.77 / 0.87 | 0.06 / 0.05 | 0.17 / 0.09 |  |  |
| b | 0.57 / 0.78 | 0.04 / 0.04 | 0.19 / 0.07 | 0.21 / 0.11 |  |
| = | 0.63 / 0.56 | 0.02 / 0.03 | 0.04 / 0.06 | 0.08 / 0.04 | 0.23 / 0.32 |

<details><summary>1 o components, main position `&lt;BOS&gt;`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L21 o c222 (H18) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same | - | same |

</details>

<details><summary>1 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L21 v c249 (kv4) | 100% / 100% | always | same |

</details>

<details><summary>1 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L21 v c246 (kv4) | 0 / 97% | off (on 0) | always |

</details>

</details>

<details><summary>L21H21 (3 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.99 / 0.99 | 0.01 / 0.01 |  |  |  |
| op | 0.94 / 0.99 | 0.00 / 0.00 | 0.06 / 0.01 |  |  |
| b | 0.48 / 0.95 | 0.01 / 0.00 | 0.37 / 0.03 | 0.13 / 0.01 |  |
| = | 0.95 / 0.95 | 0.00 / 0.00 | 0.01 / 0.00 | 0.00 / 0.00 | 0.03 / 0.04 |

<details><summary>3 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L21 o c10 (H21) | 13% / 0 | **a** (R2 0.65): a in {1..11, 13} | off (on 0) | - | same |
| L21 o c14 (H21) | 100% / 0 | always | off (on 0) | - | same |
| L21 o c20 (H21) | 3% / 0 | **a** (R2 0.57): a in {55, 65} | off (on 0) | - | same |

</details>

<details><summary>3 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L21 v c3 (kv5) | 100% / 0 | always | off (on 0) |
| L21 v c7 (kv5) | 6% / 0 | **a** (R2 1.00): a in {18, 50, 55, 60, 65, 90} | off (on 0) |
| L21 v c18 (kv5) | 0 / 97% | off (on 0) | always |

</details>

<details><summary>1 v components, main position `=` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L21 v c105 (kv5) | 1% / 54% | unexplained (best R2 0.05) | **tens(a,b)** (R2 0.52): a//10 in {0,1} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {5,6,10} -> b//10 in {1,2}; a//10 in {7} -> b//10 in {0,1,2,3}; a//10 in {8} -> b//10 in {1,2,3}; a//10 in {9} -> b//10 in {1,2,3,4,5} |

</details>

<details><summary>2 q components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L21 q c110 (H21) | 38% / 0 | **tens(a,b)** (R2 0.53): a//10 in {0} -> b//10 in {0,1,2,3,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {1,2,3,6,7,8}; a//10 in {2} -> b//10 in {2,7}; a//10 in {5} -> b//10 in {5,6,7}; a//10 in {6} -> b//10 in {6,7,8,9}; a//10 in {7,8,10} -> b//10 in {6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10} | off (on 0) |
| L21 q c116 (H21) | 0 / 0 | off (on 0) | same |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L21 k c29 (kv5) | 100% / 0 | always | off (on 0) |

</details>

</details>

<details><summary>L21H23 (3 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.99 / 0.99 | 0.01 / 0.01 |  |  |  |
| op | 0.59 / 0.97 | 0.01 / 0.01 | 0.40 / 0.03 |  |  |
| b | 0.59 / 0.87 | 0.02 / 0.01 | 0.28 / 0.10 | 0.12 / 0.03 |  |
| = | 0.93 / 0.89 | 0.01 / 0.01 | 0.02 / 0.01 | 0.01 / 0.01 | 0.03 / 0.08 |

<details><summary>3 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L21 o c1 (H23) | 100% / 0 | always | off (on 0) | - | same |
| L21 o c3 (H23) | 2% / 0 | **a** (R2 1.00): a in {55, 99} | off (on 0) | - | same |
| L21 o c7 (H23) | 1% / 0 | **a** (R2 1.00): a in {99} | off (on 0) | - | same |

</details>

<details><summary>3 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L21 v c3 (kv5) | 100% / 0 | always | off (on 0) |
| L21 v c7 (kv5) | 6% / 0 | **a** (R2 1.00): a in {18, 50, 55, 60, 65, 90} | off (on 0) |
| L21 v c18 (kv5) | 0 / 97% | off (on 0) | always |

</details>

<details><summary>1 v components, main position `=` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L21 v c105 (kv5) | 1% / 54% | unexplained (best R2 0.05) | **tens(a,b)** (R2 0.52): a//10 in {0,1} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {5,6,10} -> b//10 in {1,2}; a//10 in {7} -> b//10 in {0,1,2,3}; a//10 in {8} -> b//10 in {1,2,3}; a//10 in {9} -> b//10 in {1,2,3,4,5} |

</details>

<details><summary>1 q components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L21 q c111 (H23) | 37% / 0 | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {9}; a//10 in {1,2} -> b//10 in {0,9}; a//10 in {3} -> b//10 in {0,1,9}; a//10 in {4,5} -> b//10 in {0,1,2,9}; a//10 in {6,7,8} -> b//10 in {0,1,2}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5} | off (on 0) |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L21 k c29 (kv5) | 100% / 0 | always | off (on 0) |

</details>

</details>

<details><summary>L22H3 (2 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 1.00 / 1.00 | 0.00 / 0.00 |  |  |  |
| op | 0.97 / 0.94 | 0.01 / 0.04 | 0.02 / 0.02 |  |  |
| b | 0.96 / 0.99 | 0.00 / 0.00 | 0.03 / 0.00 | 0.00 / 0.00 |  |
| = | 0.88 / 0.79 | 0.03 / 0.07 | 0.02 / 0.01 | 0.06 / 0.11 | 0.01 / 0.03 |

<details><summary>2 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L22 o c246 (H3) | 3% / 6% | unexplained (best R2 0.49) | **a** (R2 0.69): a in {37, 47, 57, 67, 97} | - | same |
| L22 o c352 (H3) | 1% / 2% | **a** (R2 0.54): a in {97} | **a** (R2 0.64): a in {97} | - | same |

</details>

<details><summary>4 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L22 v c18 (kv0) | 0 / 8% | off (on 0) | **a** (R2 1.00): a in {22..23, 25..29, 33} |
| L22 v c56 (kv0) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {56..57} |
| L22 v c82 (kv0) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {56..57} |
| L22 v c220 (kv0) | 1% / 6% | unexplained (best R2 0.10) | **a** (R2 1.00): a in {72, 76..77, 79..81} |

</details>

<details><summary>1 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L22 v c96 (kv0) | 0 / 0 | off (on 0) | same |

</details>

<details><summary>1 q components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L22 q c136 (H3) | 75% / 33% | unexplained (best R2 0.29) | unexplained (best R2 0.40) |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L22 k c92 (kv0) | 6% / 9% | **a** (R2 0.93): a in {47, 57, 67, 77, 87, 97} | **a%20** (R2 0.90): a mod 20 in {7, 17} [coarser: a mod 10 in {7}, R2 0.89] |

</details>

</details>

<details><summary>L22H14 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.99 / 0.99 | 0.01 / 0.01 |  |  |  |
| op | 0.95 / 0.90 | 0.01 / 0.05 | 0.04 / 0.05 |  |  |
| b | 0.94 / 0.98 | 0.01 / 0.00 | 0.02 / 0.00 | 0.03 / 0.01 |  |
| = | 0.66 / 0.66 | 0.07 / 0.11 | 0.04 / 0.04 | 0.12 / 0.16 | 0.11 / 0.04 |

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L22 o c115 (H14) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>35 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L22 v c15 (kv3) | 1% / 10% | **a** (R2 0.70): a in {6} | **a%10** (R2 1.00): a mod 10 in {6} |
| L22 v c34 (kv3) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {15} |
| L22 v c37 (kv3) | 7% / 10% | **a** (R2 0.91): a in {82..88} | **a//10** (R2 1.00): (tens) a in {80..89} |
| L22 v c38 (kv3) | 1% / 5% | unexplained (best R2 0.15) | **a** (R2 1.00): a in {15..19} |
| L22 v c41 (kv3) | 13% / 14% | **a** (R2 0.92): a in {4, 6..17} | **a** (R2 0.97): a in {4, 6..18} |
| L22 v c52 (kv3) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {31} |
| L22 v c59 (kv3) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {21, 24} |
| L22 v c63 (kv3) | 0 / 1% | off (on 0) | **a//10** (R2 1.00): (tens) a in {100} |
| L22 v c75 (kv3) | 1% / 0 | **a** (R2 1.00): a in {4} | off (on 0) |
| L22 v c78 (kv3) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {10, 20, 40, 50} |
| L22 v c84 (kv3) | 0 / 7% | off (on 0) | **a** (R2 1.00): a in {90..96} |
| L22 v c89 (kv3) | 5% / 9% | **a** (R2 0.96): a in {4, 24, 48, 72, 96} | **a** (R2 0.97): a in {12, 16, 23..24, 26, 36, 48, 72, 96} |
| L22 v c101 (kv3) | 4% / 10% | **a** (R2 0.63): a in {52, 55} | **a** (R2 1.00): a in {45, 50..57, 60} |
| L22 v c109 (kv3) | 13% / 15% | **a** (R2 0.92): a in {86..98} | **a** (R2 0.97): a in {80, 85..97} |
| L22 v c110 (kv3) | 1% / 5% | **a** (R2 0.82): a in {3} | **a** (R2 0.99): a in {13, 23, 33, 83, 93} |
| L22 v c113 (kv3) | 3% / 12% | unexplained (best R2 0.42) | **a** (R2 1.00): a in {40..51} |
| L22 v c114 (kv3) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {21, 41..42} |
| L22 v c122 (kv3) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {24} |
| L22 v c130 (kv3) | 10% / 16% | **a** (R2 0.79): a in {15, 25, 30, 35, 40, 45, 55, 65, 75, 85} | **a** (R2 0.95): a in {15, 25, 30, 32..38, 40, 45, 50, 55, 85} |
| L22 v c138 (kv3) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {22..23} |
| L22 v c140 (kv3) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {22..24} |
| L22 v c165 (kv3) | 9% / 11% | **a** (R2 0.92): a in {92..100} | **a** (R2 0.99): a in {25, 50, 75, 77, 94..100} |
| L22 v c167 (kv3) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {17, 19..23} |
| L22 v c172 (kv3) | 10% / 17% | **a** (R2 0.85): a in {24..31} | **a** (R2 1.00): a in {21..31, 33..37, 45} |
| L22 v c177 (kv3) | 1% / 0 | **a** (R2 1.00): a in {4} | off (on 0) |
| L22 v c191 (kv3) | 1% / 6% | unexplained (best R2 0.09) | **a** (R2 1.00): a in {79..83, 85} |
| L22 v c192 (kv3) | 6% / 14% | **a** (R2 0.56): a in {34, 36..38} | **a** (R2 1.00): a in {30..42, 45} |
| L22 v c194 (kv3) | 0 / 7% | off (on 0) | **a** (R2 1.00): a in {26..29, 87..89} |
| L22 v c196 (kv3) | 6% / 11% | **a** (R2 0.84): a in {16, 18, 20..22} | **a** (R2 0.94): a in {14..23} |
| L22 v c199 (kv3) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {17..18, 88} |
| L22 v c203 (kv3) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {80} |
| L22 v c212 (kv3) | 0 / 8% | off (on 0) | **a** (R2 1.00): a in {20..27} |
| L22 v c213 (kv3) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {90} |
| L22 v c231 (kv3) | 5% / 6% | **a** (R2 0.90): a in {31, 41, 51, 91} | **a** (R2 0.99): a in {21, 31, 41, 51, 81, 91} [coarser: a mod 50 in {31, 41}, R2 0.82] |
| L22 v c239 (kv3) | 1% / 11% | **a** (R2 0.50): a in {31} | **a** (R2 0.98): a in {28..33, 52, 55..57, 85} |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L22 k c75 (kv3) | 8% / 0 | **a** (R2 0.98): a in {1..8} | off (on 0) |

</details>

</details>

<details><summary>L22H15 (58 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.99 / 0.99 | 0.01 / 0.01 |  |  |  |
| op | 0.96 / 0.64 | 0.02 / 0.23 | 0.02 / 0.13 |  |  |
| b | 0.96 / 0.99 | 0.00 / 0.00 | 0.02 / 0.01 | 0.02 / 0.00 |  |
| = | 0.40 / 0.79 | 0.13 / 0.04 | 0.07 / 0.01 | 0.31 / 0.09 | 0.08 / 0.08 |

<details><summary>56 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L22 o c22 (H15) | 0 / 15% | off (on 0) | **a** (R2 1.00): a in {6, 16, 25..27, 36, 46, 50, 52, 55..56, 66, 76, 86, 96} [coarser: a mod 50 in {6, 16, 26, 36, 46}, R2 0.80] | - | same |
| L22 o c31 (H15) | 0 / 10% | off (on 0) | **a** (R2 0.99): a in {13..22} | - | same |
| L22 o c32 (H15) | 0 / 10% | off (on 0) | **a//10** (R2 1.00): (tens) a in {80..89} | - | same |
| L22 o c33 (H15) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {21, 31, 81, 91} | - | same |
| L22 o c36 (H15) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {76..77, 79..81, 100} | - | same |
| L22 o c41 (H15) | 0 / 22% | off (on 0) | **a** (R2 1.00): a in {21..39, 45, 47, 57} | - | same |
| L22 o c48 (H15) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {96, 98} | - | same |
| L22 o c55 (H15) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {55..57} | - | same |
| L22 o c56 (H15) | 0 / 58% | off (on 0) | **a** (R2 1.00): a in {16..22, 37..61, 63..73, 75..89} | - | same |
| L22 o c62 (H15) | 0 / 1% | off (on 0) | **a//10** (R2 1.00): (tens) a in {100} | - | same |
| L22 o c68 (H15) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {24, 36, 94, 96} | - | same |
| L22 o c71 (H15) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {21, 31, 51} | - | same |
| L22 o c82 (H15) | 0 / 7% | off (on 0) | **a** (R2 1.00): a in {26..28, 87..89, 97} | - | same |
| L22 o c88 (H15) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {20..23} | - | same |
| L22 o c95 (H15) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {30} | - | same |
| L22 o c109 (H15) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {24} | - | same |
| L22 o c111 (H15) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {45} | - | same |
| L22 o c116 (H15) | 0 / 9% | off (on 0) | **a** (R2 0.99): a in {45, 50..53, 55..57, 60} | - | same |
| L22 o c119 (H15) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {93} | - | same |
| L22 o c123 (H15) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {18, 28} | - | same |
| L22 o c135 (H15) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {50, 99..100} [coarser: a mod 50 in {0}, R2 0.83] | - | same |
| L22 o c145 (H15) | 0 / 21% | off (on 0) | **a** (R2 1.00): a in {15..16, 25, 30..42, 45..46, 50, 55, 85} | - | same |
| L22 o c154 (H15) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {19..21} | - | same |
| L22 o c158 (H15) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {20} | - | same |
| L22 o c161 (H15) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {55} | - | same |
| L22 o c181 (H15) | 0 / 10% | off (on 0) | **a** (R2 1.00): a in {15..16, 18..23, 25..26} | - | same |
| L22 o c187 (H15) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {23, 25, 29} | - | same |
| L22 o c188 (H15) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {21} | - | same |
| L22 o c193 (H15) | 0 / 7% | off (on 0) | **a** (R2 0.97): a in {9, 19, 29, 39, 79, 89, 99} [coarser: a mod 50 in {9, 29, 39}, R2 0.80] | - | same |
| L22 o c196 (H15) | 0 / 11% | off (on 0) | **a** (R2 1.00): a in {86..96} | - | same |
| L22 o c205 (H15) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {30} | - | same |
| L22 o c224 (H15) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {22} | - | same |
| L22 o c226 (H15) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {12..14, 93} | - | same |
| L22 o c230 (H15) | 0 / 7% | off (on 0) | **a** (R2 1.00): a in {13..15, 25, 35, 45, 55} | - | same |
| L22 o c238 (H15) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {17, 27} | - | same |
| L22 o c253 (H15) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {27..29} | - | same |
| L22 o c257 (H15) | 0 / 7% | off (on 0) | **a** (R2 1.00): a in {94..100} | - | same |
| L22 o c259 (H15) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {30..31, 91} | - | same |
| L22 o c263 (H15) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {28, 30..32} | - | same |
| L22 o c266 (H15) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {18, 28, 88} | - | same |
| L22 o c281 (H15) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {42} | - | same |
| L22 o c301 (H15) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {48} | - | same |
| L22 o c302 (H15) | 0 / 1% | off (on 0) | **a//10** (R2 1.00): (tens) a in {100} | - | same |
| L22 o c306 (H15) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {40} | - | same |
| L22 o c309 (H15) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {30} | - | same |
| L22 o c320 (H15) | 0 / 13% | off (on 0) | **a** (R2 1.00): a in {12, 16, 22..28, 36, 48, 96, 100} | - | same |
| L22 o c322 (H15) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {12..14, 16} | - | same |
| L22 o c372 (H15) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {12..13, 33} | - | same |
| L22 o c376 (H15) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {21, 41..42, 91} | - | same |
| L22 o c397 (H15) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {90} | - | same |
| L22 o c434 (H15) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {98} | - | same |
| L22 o c436 (H15) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {90} | - | same |
| L22 o c444 (H15) | 0 / 8% | off (on 0) | **a** (R2 1.00): a in {19..25, 50} | - | same |
| L22 o c445 (H15) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {29} | - | same |
| L22 o c498 (H15) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {17, 22, 37, 77, 97} | - | same |
| L22 o c499 (H15) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {76} | - | same |

</details>

<details><summary>2 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L22 o c10 (H15) | 29% / 19% | **tens(a,b)** (R2 0.78): a//10 in {0,1,2} -> b//10 in {8,9,10}; a//10 in {3,4,5,6,7} -> b//10 in {9,10}; a//10 in {8} -> b//10 in {0,2,3,7,8,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.82): a//10 in {0,1,2,3,4} -> b//10 in {9,10}; a//10 in {5,6,8} -> b//10 in {10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | - | same |
| L22 o c200 (H15) | 13% / 1% | **tens(a,b)** (R2 0.59): a//10 in {0,1,2,3,4,8} -> b//10 in {9,10}; a//10 in {5,6,7} -> b//10 in {10}; a//10 in {9} -> b//10 in {3,4,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.62): a//10 in {0,1,2,3,4,5,6} -> b//10 in {10}; a//10 in {10} -> b//10 in {1,10} | - | same |

</details>

<details><summary>35 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L22 v c15 (kv3) | 1% / 10% | **a** (R2 0.70): a in {6} | **a%10** (R2 1.00): a mod 10 in {6} |
| L22 v c34 (kv3) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {15} |
| L22 v c37 (kv3) | 7% / 10% | **a** (R2 0.91): a in {82..88} | **a//10** (R2 1.00): (tens) a in {80..89} |
| L22 v c38 (kv3) | 1% / 5% | unexplained (best R2 0.15) | **a** (R2 1.00): a in {15..19} |
| L22 v c41 (kv3) | 13% / 14% | **a** (R2 0.92): a in {4, 6..17} | **a** (R2 0.97): a in {4, 6..18} |
| L22 v c52 (kv3) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {31} |
| L22 v c59 (kv3) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {21, 24} |
| L22 v c63 (kv3) | 0 / 1% | off (on 0) | **a//10** (R2 1.00): (tens) a in {100} |
| L22 v c75 (kv3) | 1% / 0 | **a** (R2 1.00): a in {4} | off (on 0) |
| L22 v c78 (kv3) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {10, 20, 40, 50} |
| L22 v c84 (kv3) | 0 / 7% | off (on 0) | **a** (R2 1.00): a in {90..96} |
| L22 v c89 (kv3) | 5% / 9% | **a** (R2 0.96): a in {4, 24, 48, 72, 96} | **a** (R2 0.97): a in {12, 16, 23..24, 26, 36, 48, 72, 96} |
| L22 v c101 (kv3) | 4% / 10% | **a** (R2 0.63): a in {52, 55} | **a** (R2 1.00): a in {45, 50..57, 60} |
| L22 v c109 (kv3) | 13% / 15% | **a** (R2 0.92): a in {86..98} | **a** (R2 0.97): a in {80, 85..97} |
| L22 v c110 (kv3) | 1% / 5% | **a** (R2 0.82): a in {3} | **a** (R2 0.99): a in {13, 23, 33, 83, 93} |
| L22 v c113 (kv3) | 3% / 12% | unexplained (best R2 0.42) | **a** (R2 1.00): a in {40..51} |
| L22 v c114 (kv3) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {21, 41..42} |
| L22 v c122 (kv3) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {24} |
| L22 v c130 (kv3) | 10% / 16% | **a** (R2 0.79): a in {15, 25, 30, 35, 40, 45, 55, 65, 75, 85} | **a** (R2 0.95): a in {15, 25, 30, 32..38, 40, 45, 50, 55, 85} |
| L22 v c138 (kv3) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {22..23} |
| L22 v c140 (kv3) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {22..24} |
| L22 v c165 (kv3) | 9% / 11% | **a** (R2 0.92): a in {92..100} | **a** (R2 0.99): a in {25, 50, 75, 77, 94..100} |
| L22 v c167 (kv3) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {17, 19..23} |
| L22 v c172 (kv3) | 10% / 17% | **a** (R2 0.85): a in {24..31} | **a** (R2 1.00): a in {21..31, 33..37, 45} |
| L22 v c177 (kv3) | 1% / 0 | **a** (R2 1.00): a in {4} | off (on 0) |
| L22 v c191 (kv3) | 1% / 6% | unexplained (best R2 0.09) | **a** (R2 1.00): a in {79..83, 85} |
| L22 v c192 (kv3) | 6% / 14% | **a** (R2 0.56): a in {34, 36..38} | **a** (R2 1.00): a in {30..42, 45} |
| L22 v c194 (kv3) | 0 / 7% | off (on 0) | **a** (R2 1.00): a in {26..29, 87..89} |
| L22 v c196 (kv3) | 6% / 11% | **a** (R2 0.84): a in {16, 18, 20..22} | **a** (R2 0.94): a in {14..23} |
| L22 v c199 (kv3) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {17..18, 88} |
| L22 v c203 (kv3) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {80} |
| L22 v c212 (kv3) | 0 / 8% | off (on 0) | **a** (R2 1.00): a in {20..27} |
| L22 v c213 (kv3) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {90} |
| L22 v c231 (kv3) | 5% / 6% | **a** (R2 0.90): a in {31, 41, 51, 91} | **a** (R2 0.99): a in {21, 31, 41, 51, 81, 91} [coarser: a mod 50 in {31, 41}, R2 0.82] |
| L22 v c239 (kv3) | 1% / 11% | **a** (R2 0.50): a in {31} | **a** (R2 0.98): a in {28..33, 52, 55..57, 85} |

</details>

<details><summary>1 q components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L22 q c109 (H15) | 0 / 85% | off (on 0) | **a** (R2 0.98): a in {6, 9..10, 12..57, 60..61, 63, 65..69, 71..73, 76..100} |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L22 k c75 (kv3) | 8% / 0 | **a** (R2 0.98): a in {1..8} | off (on 0) |

</details>

</details>

<details><summary>L22H24 (25 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.81 / 0.81 | 0.19 / 0.19 |  |  |  |
| op | 0.60 / 0.73 | 0.20 / 0.22 | 0.21 / 0.05 |  |  |
| b | 0.39 / 0.36 | 0.21 / 0.12 | 0.27 / 0.42 | 0.13 / 0.11 |  |
| = | 0.77 / 0.74 | 0.03 / 0.03 | 0.03 / 0.02 | 0.11 / 0.07 | 0.07 / 0.14 |

<details><summary>25 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L22 o c0 (H24) | 26% / 8% | **a** (R2 0.62): a in {18, 20..21, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100} [coarser: a mod 50 in {0, 10, 15, 18, 20, 25, 30, 35, 40, 45}, R2 0.82] | unexplained (best R2 0.15) | a: mod5 +4% | - |
| L22 o c1 (H24) | 0 / 0 | off (on 0) | same | - | same |
| L22 o c2 (H24) | 87% / 32% | **a//10** (R2 0.58): (tens) a in {1..89} | unexplained (best R2 0.40) | - | same |
| L22 o c16 (H24) | 19% / 0 | **a** (R2 0.69): a in {3..10, 12..17, 19} | off (on 0) | - | same |
| L22 o c39 (H24) | 0 / 0 | off (on 0) | same | - | same |
| L22 o c46 (H24) | 0 / 0 | off (on 0) | same | - | same |
| L22 o c52 (H24) | 0 / 0 | off (on 0) | same | - | same |
| L22 o c63 (H24) | 0 / 1% | off (on 0) | unexplained (best R2 0.14) | - | same |
| L22 o c65 (H24) | 2% / 32% | unexplained (best R2 0.23) | unexplained (best R2 0.41) | - | same |
| L22 o c69 (H24) | 0 / 0 | off (on 0) | same | - | same |
| L22 o c76 (H24) | 0 / 30% | off (on 0) | **a** (R2 0.60): a in {1, 4..10, 12..32} | - | same |
| L22 o c89 (H24) | 4% / 1% | **a** (R2 0.54): a in {90..91} | **tens(a,b)** (R2 0.52): a//10 in {9} -> b//10 in {9,10} | - | same |
| L22 o c102 (H24) | 5% / 1% | unexplained (best R2 0.49) | unexplained (best R2 0.41) | - | same |
| L22 o c120 (H24) | 9% / 1% | **a//10** (R2 0.84): (tens) a in {1..9} | unexplained (best R2 0.14) | a: mod50 +2%, mod25 +3%, mod20 +3% | - |
| L22 o c124 (H24) | 0 / 0 | off (on 0) | same | - | same |
| L22 o c125 (H24) | 0 / 0 | off (on 0) | same | - | same |
| L22 o c134 (H24) | 0 / 0 | off (on 0) | same | - | same |
| L22 o c164 (H24) | 0 / 0 | off (on 0) | same | - | same |
| L22 o c244 (H24) | 0 / 2% | off (on 0) | unexplained (best R2 0.09) | - | same |
| L22 o c283 (H24) | 0 / 0 | off (on 0) | same | - | same |
| L22 o c298 (H24) | 0 / 0 | off (on 0) | same | - | same |
| L22 o c310 (H24) | 0 / 0 | off (on 0) | same | - | same |
| L22 o c343 (H24) | 0 / 0 | off (on 0) | same | - | same |
| L22 o c410 (H24) | 0 / 0 | off (on 0) | same | - | same |
| L22 o c418 (H24) | 1% / 0 | **a** (R2 0.76): a in {90} | off (on 0) | - | same |

</details>

<details><summary>1 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L22 v c143 (kv6) | 2% / 2% | **a** (R2 0.94): a in {60, 90} | **a** (R2 0.91): a in {60, 90} |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L22 v c190 (kv6) | 6% / 0 | **a** (R2 0.98): a in {1..6} | off (on 0) |

</details>

</details>

<details><summary>L22H25 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.79 / 0.79 | 0.21 / 0.21 |  |  |  |
| op | 0.82 / 0.95 | 0.05 / 0.03 | 0.12 / 0.03 |  |  |
| b | 0.73 / 0.64 | 0.05 / 0.04 | 0.12 / 0.21 | 0.10 / 0.11 |  |
| = | 0.91 / 0.90 | 0.02 / 0.03 | 0.04 / 0.02 | 0.02 / 0.02 | 0.01 / 0.02 |

<details><summary>1 o components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L22 o c6 (H25) | 9% / 9% | **a** (R2 1.00): a in {16..19, 21..24, 31} | same | - | same |

</details>

<details><summary>1 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L22 v c143 (kv6) | 2% / 2% | **a** (R2 0.94): a in {60, 90} | **a** (R2 0.91): a in {60, 90} |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L22 v c190 (kv6) | 6% / 0 | **a** (R2 0.98): a in {1..6} | off (on 0) |

</details>

</details>

<details><summary>L22H28 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.99 / 0.99 | 0.01 / 0.01 |  |  |  |
| op | 0.98 / 0.90 | 0.01 / 0.03 | 0.02 / 0.06 |  |  |
| b | 0.98 / 0.99 | 0.00 / 0.00 | 0.01 / 0.01 | 0.00 / 0.00 |  |
| = | 0.81 / 0.66 | 0.06 / 0.07 | 0.02 / 0.03 | 0.08 / 0.20 | 0.03 / 0.03 |

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L22 o c147 (H28) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>2 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L22 k c1 (kv7) | 100% / 100% | always | same |
| L22 k c27 (kv7) | 1% / 1% | **a** (R2 1.00): a in {93} | same |

</details>

</details>

<details><summary>L23H2 (2 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.85 / 0.85 | 0.15 / 0.15 |  |  |  |
| op | 0.69 / 0.63 | 0.03 / 0.14 | 0.28 / 0.23 |  |  |
| b | 0.75 / 0.44 | 0.01 / 0.04 | 0.16 / 0.28 | 0.07 / 0.23 |  |
| = | 0.95 / 0.95 | 0.01 / 0.01 | 0.02 / 0.01 | 0.01 / 0.01 | 0.01 / 0.02 |

<details><summary>2 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L23 o c3 (H2) | 0 / 95% | off (on 0) | always | - | same |
| L23 o c9 (H2) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>1 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L23 v c166 (kv0) | 99% / 99% | always | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L23 v c5 (kv0) | 0 / 93% | off (on 0) | **a** (R2 0.84): a in {1..29, 31..44, 46..50, 52, 54..59, 61, 63..69, 71..79, 81..99} |

</details>

</details>

<details><summary>L23H7 (11 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.97 / 0.97 | 0.03 / 0.03 |  |  |  |
| op | 0.93 / 0.90 | 0.01 / 0.07 | 0.06 / 0.03 |  |  |
| b | 0.82 / 0.82 | 0.01 / 0.00 | 0.16 / 0.16 | 0.01 / 0.01 |  |
| = | 0.87 / 0.83 | 0.04 / 0.05 | 0.02 / 0.02 | 0.04 / 0.08 | 0.02 / 0.02 |

<details><summary>9 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L23 o c94 (H7) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {68, 78} | - | same |
| L23 o c204 (H7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {88} | - | same |
| L23 o c315 (H7) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {88..89} | - | same |
| L23 o c318 (H7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {98} | - | same |
| L23 o c320 (H7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {58} | - | same |
| L23 o c333 (H7) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {58, 78, 98} | - | same |
| L23 o c353 (H7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {88} | - | same |
| L23 o c470 (H7) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {58, 68} | - | same |
| L23 o c474 (H7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {58} | - | same |

</details>

<details><summary>2 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L23 o c187 (H7) | 1% / 4% | **a** (R2 0.70): a in {98} | **a** (R2 0.64): a in {97..99} | - | same |
| L23 o c280 (H7) | 1% / 3% | unexplained (best R2 0.41) | **a** (R2 0.56): a in {8, 88} | - | same |

</details>

<details><summary>1 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L23 v c133 (kv1) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>13 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L23 v c61 (kv1) | 1% / 1% | **a** (R2 1.00): a in {98} | **a** (R2 0.95): a in {98} |
| L23 v c91 (kv1) | 0 / 2% | off (on 0) | **a** (R2 0.99): a in {68, 78} |
| L23 v c96 (kv1) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {88} |
| L23 v c151 (kv1) | 1% / 1% | **a** (R2 1.00): a in {88} | **a** (R2 0.99): a in {88} |
| L23 v c154 (kv1) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {88} |
| L23 v c164 (kv1) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {98} |
| L23 v c175 (kv1) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {58} |
| L23 v c201 (kv1) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {55, 58} |
| L23 v c221 (kv1) | 1% / 1% | **a** (R2 1.00): a in {88} | same |
| L23 v c230 (kv1) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {58} |
| L23 v c231 (kv1) | 1% / 1% | **a** (R2 1.00): a in {88} | same |
| L23 v c236 (kv1) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {58} |
| L23 v c246 (kv1) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {55, 58} |

</details>

<details><summary>1 q components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L23 q c83 (H7) | 0 / 15% | off (on 0) | **a** (R2 1.00): a in {44..45, 55, 58, 64..65, 68, 75, 78, 84..85, 88..89, 94, 98} |

</details>

<details><summary>1 q components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L23 q c74 (H7) | 0 / 0 | off (on 0) | same |

</details>

<details><summary>4 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L23 k c2 (kv1) | 99% / 99% | always | same |
| L23 k c90 (kv1) | 0 / 2% | off (on 0) | **a** (R2 0.99): a in {88, 98} |
| L23 k c129 (kv1) | 2% / 6% | **a** (R2 0.92): a in {88, 98} | **a** (R2 1.00): a in {28, 58, 68, 78, 88, 98} |
| L23 k c142 (kv1) | 1% / 4% | **a** (R2 0.91): a in {98} | **a** (R2 1.00): a in {58, 68, 88, 98} |

</details>

</details>

<details><summary>L23H14 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.77 / 0.77 | 0.23 / 0.23 |  |  |  |
| op | 0.78 / 0.70 | 0.12 / 0.11 | 0.09 / 0.19 |  |  |
| b | 0.81 / 0.86 | 0.02 / 0.03 | 0.08 / 0.03 | 0.10 / 0.08 |  |
| = | 0.82 / 0.65 | 0.05 / 0.11 | 0.03 / 0.04 | 0.08 / 0.17 | 0.02 / 0.03 |

<details><summary>1 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L23 o c36 (H14) | 8% / 5% | **a** (R2 0.98): a in {61..64, 66..69} | **a** (R2 1.00): a in {63..67} | - | same |

</details>

<details><summary>7 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L23 v c29 (kv3) | 6% / 8% | **a** (R2 0.90): a in {20, 25, 50, 75, 100} [coarser: a mod 25 in {0}, R2 0.82] | **a** (R2 0.93): a in {20, 25, 40, 50, 75, 80, 100} [coarser: a mod 50 in {0, 20, 25, 30, 40}, R2 0.80] |
| L23 v c33 (kv3) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {88} |
| L23 v c51 (kv3) | 7% / 8% | **a** (R2 0.98): a in {62..68} | **a** (R2 0.95): a in {62..68} |
| L23 v c52 (kv3) | 3% / 4% | unexplained (best R2 0.35) | unexplained (best R2 0.46) |
| L23 v c55 (kv3) | 5% / 9% | **a** (R2 0.93): a in {88..92} | **a** (R2 0.94): a in {85..92} |
| L23 v c131 (kv3) | 9% / 9% | **a** (R2 1.00): a in {91..99} | same |
| L23 v c216 (kv3) | 1% / 2% | **a** (R2 0.86): a in {50} | **a** (R2 0.95): a in {25, 50} |

</details>

<details><summary>6 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L23 v c53 (kv3) | 0 / 0 | off (on 0) | same |
| L23 v c79 (kv3) | 0 / 0 | off (on 0) | same |
| L23 v c122 (kv3) | 0 / 0 | off (on 0) | same |
| L23 v c147 (kv3) | 0 / 0 | off (on 0) | same |
| L23 v c193 (kv3) | 0 / 0 | off (on 0) | same |
| L23 v c198 (kv3) | 0 / 0 | off (on 0) | same |

</details>

</details>

<details><summary>L23H15 (2 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.98 / 0.98 | 0.02 / 0.02 |  |  |  |
| op | 0.94 / 0.87 | 0.03 / 0.09 | 0.02 / 0.04 |  |  |
| b | 0.96 / 0.99 | 0.01 / 0.00 | 0.01 / 0.00 | 0.02 / 0.01 |  |
| = | 0.69 / 0.68 | 0.11 / 0.08 | 0.02 / 0.01 | 0.17 / 0.21 | 0.02 / 0.02 |

<details><summary>2 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L23 o c308 (H15) | 96% / 91% | always | unexplained (best R2 0.41) | - | same |
| L23 o c311 (H15) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>7 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L23 v c29 (kv3) | 6% / 8% | **a** (R2 0.90): a in {20, 25, 50, 75, 100} [coarser: a mod 25 in {0}, R2 0.82] | **a** (R2 0.93): a in {20, 25, 40, 50, 75, 80, 100} [coarser: a mod 50 in {0, 20, 25, 30, 40}, R2 0.80] |
| L23 v c33 (kv3) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {88} |
| L23 v c51 (kv3) | 7% / 8% | **a** (R2 0.98): a in {62..68} | **a** (R2 0.95): a in {62..68} |
| L23 v c52 (kv3) | 3% / 4% | unexplained (best R2 0.35) | unexplained (best R2 0.46) |
| L23 v c55 (kv3) | 5% / 9% | **a** (R2 0.93): a in {88..92} | **a** (R2 0.94): a in {85..92} |
| L23 v c131 (kv3) | 9% / 9% | **a** (R2 1.00): a in {91..99} | same |
| L23 v c216 (kv3) | 1% / 2% | **a** (R2 0.86): a in {50} | **a** (R2 0.95): a in {25, 50} |

</details>

<details><summary>6 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L23 v c53 (kv3) | 0 / 0 | off (on 0) | same |
| L23 v c79 (kv3) | 0 / 0 | off (on 0) | same |
| L23 v c122 (kv3) | 0 / 0 | off (on 0) | same |
| L23 v c147 (kv3) | 0 / 0 | off (on 0) | same |
| L23 v c193 (kv3) | 0 / 0 | off (on 0) | same |
| L23 v c198 (kv3) | 0 / 0 | off (on 0) | same |

</details>

<details><summary>1 q components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L23 q c121 (H15) | 0 / 0 | off (on 0) | same |

</details>

</details>

<details><summary>L23H21 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.99 / 0.99 | 0.01 / 0.01 |  |  |  |
| op | 0.91 / 0.93 | 0.00 / 0.04 | 0.08 / 0.03 |  |  |
| b | 0.96 / 0.99 | 0.00 / 0.00 | 0.03 / 0.00 | 0.01 / 0.00 |  |
| = | 0.89 / 0.85 | 0.02 / 0.03 | 0.03 / 0.01 | 0.03 / 0.07 | 0.03 / 0.04 |

<details><summary>1 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L23 o c415 (H21) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {85} | - | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L23 v c17 (kv5) | 80% / 0 | **a** (R2 0.99): a in {2..10, 12..54, 56..59, 61..69, 71..74, 76..79, 81..84, 86..87, 99} | off (on 0) |

</details>

</details>

<details><summary>L23H22 (11 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 1.00 / 1.00 | 0.00 / 0.00 |  |  |  |
| op | 0.94 / 0.94 | 0.00 / 0.04 | 0.05 / 0.02 |  |  |
| b | 0.91 / 0.99 | 0.00 / 0.00 | 0.08 / 0.00 | 0.01 / 0.01 |  |
| = | 0.78 / 0.78 | 0.05 / 0.06 | 0.09 / 0.01 | 0.06 / 0.14 | 0.02 / 0.02 |

<details><summary>2 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L23 o c323 (H22) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {84} | - | same |
| L23 o c410 (H22) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {94} | - | same |

</details>

<details><summary>9 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L23 o c152 (H22) | 4% / 9% | unexplained (best R2 0.50) | unexplained (best R2 0.46) | - | same |
| L23 o c234 (H22) | 2% / 5% | unexplained (best R2 0.38) | same | - | same |
| L23 o c299 (H22) | 0 / 0 | off (on 0) | same | - | same |
| L23 o c370 (H22) | 1% / 1% | **a** (R2 0.82): a in {55} | **a** (R2 0.67): a in {55} | - | same |
| L23 o c394 (H22) | 6% / 9% | unexplained (best R2 0.35) | unexplained (best R2 0.34) | - | same |
| L23 o c402 (H22) | 0 / 0 | off (on 0) | same | - | same |
| L23 o c420 (H22) | 0 / 0 | off (on 0) | same | - | same |
| L23 o c435 (H22) | 1% / 2% | **a** (R2 0.54): a in {65} | unexplained (best R2 0.48) | - | same |
| L23 o c456 (H22) | 1% / 3% | unexplained (best R2 0.21) | unexplained (best R2 0.39) | - | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L23 v c17 (kv5) | 80% / 0 | **a** (R2 0.99): a in {2..10, 12..54, 56..59, 61..69, 71..74, 76..79, 81..84, 86..87, 99} | off (on 0) |

</details>

</details>

<details><summary>L24H8 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.95 / 0.95 | 0.05 / 0.05 |  |  |  |
| op | 0.93 / 0.96 | 0.03 / 0.02 | 0.04 / 0.02 |  |  |
| b | 0.92 / 0.88 | 0.01 / 0.01 | 0.03 / 0.08 | 0.03 / 0.02 |  |
| = | 0.96 / 0.95 | 0.00 / 0.01 | 0.01 / 0.01 | 0.01 / 0.01 | 0.02 / 0.03 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L24 o c232 (H8) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>4 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L24 v c48 (kv2) | 9% / 0 | **a//10** (R2 1.00): (tens) a in {1..9} | off (on 0) |
| L24 v c69 (kv2) | 0 / 94% | off (on 0) | unexplained (best R2 0.26) |
| L24 v c169 (kv2) | 0 / 9% | off (on 0) | **a//10** (R2 1.00): (tens) a in {1..9} |
| L24 v c199 (kv2) | 9% / 0 | **a** (R2 0.97): a in {90..98} | off (on 0) |

</details>

</details>

<details><summary>L24H10 (14 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.63 / 0.63 | 0.37 / 0.37 |  |  |  |
| op | 0.45 / 0.90 | 0.08 / 0.08 | 0.47 / 0.03 |  |  |
| b | 0.27 / 0.39 | 0.03 / 0.04 | 0.51 / 0.46 | 0.19 / 0.12 |  |
| = | 0.96 / 0.92 | 0.00 / 0.01 | 0.01 / 0.01 | 0.01 / 0.02 | 0.02 / 0.04 |

<details><summary>1 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L24 o c26 (H10) | 1% / 0 | **a//10** (R2 1.00): (tens) a in {100} | off (on 0) | - | same |

</details>

<details><summary>13 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L24 o c0 (H10) | 24% / 6% | **a** (R2 0.46): a in {1..10, 12..17, 88} | **a** (R2 0.48): a in {2, 4..9} | - | same |
| L24 o c2 (H10) | 98% / 0 | always | off (on 0) | a: mod5 +3% | - |
| L24 o c10 (H10) | 0 / 0 | off (on 0) | same | - | same |
| L24 o c13 (H10) | 27% / 0 | **tens(a,b)** (R2 0.59): a//10 in {0} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {8,9,10}; a//10 in {4} -> b//10 in {9,10}; a//10 in {5,6,7,8} -> b//10 in {10} | off (on 0) | a: mod100 +2%, mod50 +2% | - |
| L24 o c18 (H10) | 0 / 94% | off (on 0) | unexplained (best R2 0.19) | - | a: mod10 +2%, mod5 +2% |
| L24 o c21 (H10) | 0 / 0 | off (on 0) | same | - | same |
| L24 o c40 (H10) | 0 / 1% | off (on 0) | unexplained (best R2 0.05) | - | same |
| L24 o c49 (H10) | 0 / 0 | off (on 0) | same | - | same |
| L24 o c53 (H10) | 0 / 1% | off (on 0) | unexplained (best R2 0.04) | - | same |
| L24 o c54 (H10) | 0 / 0 | off (on 0) | same | - | same |
| L24 o c72 (H10) | 0 / 0 | off (on 0) | same | - | same |
| L24 o c79 (H10) | 4% / 0 | **a** (R2 0.78): a in {1..3} | off (on 0) | - | same |
| L24 o c88 (H10) | 0 / 6% | off (on 0) | unexplained (best R2 0.41) | - | same |

</details>

<details><summary>4 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L24 v c48 (kv2) | 9% / 0 | **a//10** (R2 1.00): (tens) a in {1..9} | off (on 0) |
| L24 v c69 (kv2) | 0 / 94% | off (on 0) | unexplained (best R2 0.26) |
| L24 v c169 (kv2) | 0 / 9% | off (on 0) | **a//10** (R2 1.00): (tens) a in {1..9} |
| L24 v c199 (kv2) | 9% / 0 | **a** (R2 0.97): a in {90..98} | off (on 0) |

</details>

<details><summary>1 q components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L24 q c121 (H10) | 6% / 95% | **a** (R2 0.99): a in {1..6} | always |

</details>

</details>

<details><summary>L24H13 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.91 / 0.91 | 0.09 / 0.09 |  |  |  |
| op | 0.86 / 0.85 | 0.06 / 0.09 | 0.08 / 0.06 |  |  |
| b | 0.88 / 0.83 | 0.03 / 0.04 | 0.04 / 0.04 | 0.05 / 0.08 |  |
| = | 0.52 / 0.45 | 0.04 / 0.09 | 0.10 / 0.06 | 0.14 / 0.17 | 0.20 / 0.23 |

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L24 o c23 (H13) | 68% / 83% | **tens(a,b)** (R2 0.56): a//10 in {0,1} -> b//10 in {6,7,8,9}; a//10 in {2,3} -> b//10 in {5,6,7,8,9}; a//10 in {4} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {1,2,3,4,5,7,8,9,10}; a//10 in {7,8,9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | unexplained (best R2 0.40) | - | same |

</details>

</details>

<details><summary>L24H14 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.53 / 0.53 | 0.47 / 0.47 |  |  |  |
| op | 0.69 / 0.91 | 0.07 / 0.03 | 0.24 / 0.07 |  |  |
| b | 0.83 / 0.72 | 0.04 / 0.04 | 0.07 / 0.04 | 0.06 / 0.20 |  |
| = | 0.85 / 0.81 | 0.01 / 0.02 | 0.02 / 0.03 | 0.02 / 0.02 | 0.10 / 0.12 |

<details><summary>1 o components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L24 o c39 (H14) | 100% / 100% | always | same | - | same |

</details>

</details>

<details><summary>L24H17 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 1.00 / 1.00 | 0.00 / 0.00 |  |  |  |
| op | 0.99 / 0.98 | 0.00 / 0.01 | 0.00 / 0.01 |  |  |
| b | 0.99 / 0.99 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.01 |  |
| = | 0.93 / 0.89 | 0.04 / 0.04 | 0.01 / 0.01 | 0.03 / 0.06 | 0.01 / 0.00 |

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L24 o c503 (H17) | 1% / 1% | **a** (R2 0.89): a in {90} | **a** (R2 0.82): a in {90} | - | same |

</details>

<details><summary>1 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L24 v c219 (kv4) | 1% / 1% | **a** (R2 1.00): a in {90} | same |

</details>

</details>

<details><summary>L24H19 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 1.00 / 1.00 | 0.00 / 0.00 |  |  |  |
| op | 0.98 / 0.92 | 0.00 / 0.06 | 0.02 / 0.02 |  |  |
| b | 0.99 / 0.99 | 0.00 / 0.00 | 0.01 / 0.00 | 0.00 / 0.00 |  |
| = | 0.93 / 0.87 | 0.02 / 0.02 | 0.01 / 0.01 | 0.03 / 0.07 | 0.02 / 0.03 |

<details><summary>1 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L24 o c469 (H19) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {90} | - | same |

</details>

<details><summary>1 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L24 v c219 (kv4) | 1% / 1% | **a** (R2 1.00): a in {90} | same |

</details>

</details>

<details><summary>L24H21 (12 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 1.00 / 1.00 | 0.00 / 0.00 |  |  |  |
| op | 0.97 / 0.89 | 0.01 / 0.10 | 0.02 / 0.01 |  |  |
| b | 0.98 / 1.00 | 0.00 / 0.00 | 0.01 / 0.00 | 0.01 / 0.00 |  |
| = | 0.86 / 0.82 | 0.04 / 0.05 | 0.01 / 0.01 | 0.09 / 0.11 | 0.00 / 0.01 |

<details><summary>12 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L24 o c241 (H21) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {63, 73} | - | same |
| L24 o c249 (H21) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {24, 61..62} | - | same |
| L24 o c302 (H21) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {72} | - | same |
| L24 o c318 (H21) | 0 / 17% | off (on 0) | **a** (R2 1.00): a in {23..24, 58..69, 71..73} | - | same |
| L24 o c321 (H21) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {63} | - | same |
| L24 o c338 (H21) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {61} | - | same |
| L24 o c364 (H21) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {61} | - | same |
| L24 o c376 (H21) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {63..64} | - | same |
| L24 o c394 (H21) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {23} | - | same |
| L24 o c401 (H21) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {63..66, 68} | - | same |
| L24 o c439 (H21) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {68} | - | same |
| L24 o c480 (H21) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {24, 64, 68..69, 72} | - | same |

</details>

<details><summary>6 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L24 v c155 (kv5) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {24} |
| L24 v c185 (kv5) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {24, 64} |
| L24 v c203 (kv5) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {63} |
| L24 v c225 (kv5) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {24} |
| L24 v c232 (kv5) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {24, 61..63} |
| L24 v c233 (kv5) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {31, 61} |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L24 k c6 (kv5) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L24H22 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.99 / 0.99 | 0.01 / 0.01 |  |  |  |
| op | 0.99 / 0.98 | 0.00 / 0.02 | 0.00 / 0.01 |  |  |
| b | 0.98 / 0.98 | 0.00 / 0.00 | 0.01 / 0.00 | 0.01 / 0.01 |  |
| = | 0.88 / 0.86 | 0.05 / 0.05 | 0.01 / 0.01 | 0.06 / 0.08 | 0.01 / 0.01 |

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L24 o c501 (H22) | 0 / 2% | off (on 0) | **a** (R2 0.63): a in {27..29} | - | same |

</details>

<details><summary>6 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L24 v c155 (kv5) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {24} |
| L24 v c185 (kv5) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {24, 64} |
| L24 v c203 (kv5) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {63} |
| L24 v c225 (kv5) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {24} |
| L24 v c232 (kv5) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {24, 61..63} |
| L24 v c233 (kv5) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {31, 61} |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L24 k c6 (kv5) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L25H9 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.73 / 0.73 | 0.27 / 0.27 |  |  |  |
| op | 0.62 / 0.90 | 0.07 / 0.04 | 0.31 / 0.06 |  |  |
| b | 0.63 / 0.48 | 0.05 / 0.06 | 0.12 / 0.08 | 0.20 / 0.38 |  |
| = | 0.84 / 0.87 | 0.03 / 0.03 | 0.07 / 0.01 | 0.03 / 0.05 | 0.03 / 0.04 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L25 o c220 (H9) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L25 v c178 (kv2) | 99% / 0 | always | off (on 0) |

</details>

<details><summary>1 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L25 v c24 (kv2) | 0 / 24% | off (on 0) | unexplained (best R2 0.32) |

</details>

</details>

<details><summary>L25H23 (2 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.89 / 0.89 | 0.11 / 0.11 |  |  |  |
| op | 0.67 / 0.36 | 0.24 / 0.50 | 0.09 / 0.14 |  |  |
| b | 0.70 / 0.58 | 0.08 / 0.11 | 0.12 / 0.17 | 0.09 / 0.13 |  |
| = | 0.64 / 0.70 | 0.00 / 0.00 | 0.01 / 0.01 | 0.08 / 0.11 | 0.27 / 0.17 |

<details><summary>1 o components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L25 o c394 (H23) | 100% / 100% | always | same | - | same |

</details>

<details><summary>1 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L25 o c1 (H23) | 0 / 83% | off (on 0) | **a** (R2 1.00): a in {3..40, 42..54, 56..58, 60, 63..70, 72, 76..78, 80, 82..94, 96, 100} | - | same |

</details>

<details><summary>1 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L25 v c0 (kv5) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>1 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L25 v c18 (kv5) | 100% / 100% | always | same |

</details>

<details><summary>1 v components, main position `=` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L25 v c122 (kv5) | 0 / 0 | off (on 0) | same |

</details>

<details><summary>1 q components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L25 q c119 (H23) | 0 / 86% | off (on 0) | **a** (R2 0.99): a in {3..40, 42..58, 60, 63..70, 72, 76..94, 96, 100} |

</details>

</details>

<details><summary>L25H29 (2 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.92 / 0.92 | 0.08 / 0.08 |  |  |  |
| op | 0.95 / 0.94 | 0.03 / 0.05 | 0.02 / 0.01 |  |  |
| b | 0.86 / 0.76 | 0.02 / 0.02 | 0.08 / 0.19 | 0.04 / 0.03 |  |
| = | 0.94 / 0.92 | 0.01 / 0.01 | 0.02 / 0.02 | 0.02 / 0.03 | 0.02 / 0.03 |

<details><summary>2 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L25 o c52 (H29) | 0 / 2% | off (on 0) | unexplained (best R2 0.46) | - | same |
| L25 o c240 (H29) | 0 / 11% | off (on 0) | unexplained (best R2 0.41) | - | same |

</details>

<details><summary>4 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L25 v c2 (kv7) | 0 / 97% | off (on 0) | always |
| L25 v c22 (kv7) | 0 / 16% | off (on 0) | **a** (R2 0.72): a in {77..80, 87..91, 94..100} |
| L25 v c92 (kv7) | 0 / 6% | off (on 0) | **a** (R2 0.59): a in {47..50, 100} |
| L25 v c165 (kv7) | 0 / 28% | off (on 0) | **a** (R2 0.77): a in {58..69, 71..82, 84..85, 100} |

</details>

<details><summary>1 q components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L25 q c64 (H29) | 0 / 2% | off (on 0) | **tens(a,b)** (R2 0.54): a//10 in {6,7,8} -> b//10 in {0} |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L25 k c59 (kv7) | 0 / 17% | off (on 0) | **a** (R2 0.97): a in {82..98} |

</details>

</details>

<details><summary>L25H31 (5 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.90 / 0.90 | 0.10 / 0.10 |  |  |  |
| op | 0.92 / 0.89 | 0.03 / 0.07 | 0.05 / 0.04 |  |  |
| b | 0.73 / 0.47 | 0.02 / 0.01 | 0.16 / 0.45 | 0.10 / 0.07 |  |
| = | 0.93 / 0.90 | 0.00 / 0.00 | 0.01 / 0.01 | 0.03 / 0.02 | 0.03 / 0.07 |

<details><summary>5 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L25 o c6 (H31) | 0 / 38% | off (on 0) | unexplained (best R2 0.30) | - | same |
| L25 o c10 (H31) | 0 / 28% | off (on 0) | **a** (R2 0.65): a in {58..69, 71..82, 84, 87, 100} | - | same |
| L25 o c15 (H31) | 0 / 92% | off (on 0) | unexplained (best R2 0.17) | - | a: mod100 +4% |
| L25 o c35 (H31) | 0 / 1% | off (on 0) | unexplained (best R2 0.44) | - | same |
| L25 o c86 (H31) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>4 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L25 v c2 (kv7) | 0 / 97% | off (on 0) | always |
| L25 v c22 (kv7) | 0 / 16% | off (on 0) | **a** (R2 0.72): a in {77..80, 87..91, 94..100} |
| L25 v c92 (kv7) | 0 / 6% | off (on 0) | **a** (R2 0.59): a in {47..50, 100} |
| L25 v c165 (kv7) | 0 / 28% | off (on 0) | **a** (R2 0.77): a in {58..69, 71..82, 84..85, 100} |

</details>

<details><summary>1 q components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L25 q c116 (H31) | 1% / 89% | unexplained (best R2 0.18) | unexplained (best R2 0.16) |

</details>

<details><summary>1 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L25 k c59 (kv7) | 0 / 17% | off (on 0) | **a** (R2 0.97): a in {82..98} |

</details>

</details>

<details><summary>L26H10 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.96 / 0.96 | 0.04 / 0.04 |  |  |  |
| op | 0.96 / 0.96 | 0.01 / 0.02 | 0.03 / 0.01 |  |  |
| b | 0.96 / 0.95 | 0.01 / 0.01 | 0.01 / 0.01 | 0.02 / 0.03 |  |
| = | 0.92 / 0.81 | 0.01 / 0.03 | 0.02 / 0.03 | 0.02 / 0.03 | 0.03 / 0.10 |

<details><summary>1 o components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L26 o c19 (H10) | 100% / 100% | always | same | - | same |

</details>

<details><summary>1 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L26 v c32 (kv2) | 100% / 100% | always | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L26 v c91 (kv2) | 98% / 0 | always | off (on 0) |

</details>

</details>

<details><summary>L26H14 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.99 / 0.99 | 0.01 / 0.01 |  |  |  |
| op | 0.99 / 0.98 | 0.01 / 0.02 | 0.00 / 0.00 |  |  |
| b | 0.98 / 0.98 | 0.00 / 0.00 | 0.01 / 0.00 | 0.01 / 0.01 |  |
| = | 0.90 / 0.87 | 0.05 / 0.06 | 0.01 / 0.01 | 0.04 / 0.05 | 0.00 / 0.00 |

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L26 o c379 (H14) | 3% / 4% | **a** (R2 0.61): a in {92..94} | **a** (R2 0.81): a in {92..94} | - | same |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L26 k c1 (kv3) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L27H1 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 1.00 / 1.00 | 0.00 / 0.00 |  |  |  |
| op | 0.99 / 0.83 | 0.00 / 0.14 | 0.01 / 0.03 |  |  |
| b | 0.99 / 0.98 | 0.00 / 0.00 | 0.00 / 0.01 | 0.00 / 0.01 |  |
| = | 0.88 / 0.79 | 0.06 / 0.06 | 0.01 / 0.02 | 0.05 / 0.10 | 0.01 / 0.02 |

<details><summary>1 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L27 o c269 (H1) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {16, 36, 48, 56, 66, 96} | - | same |

</details>

<details><summary>1 q components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L27 q c130 (H1) | 0 / 100% | off (on 0) | always |

</details>

<details><summary>1 k components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L27 k c13 (kv0) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L27 k c5 (kv0) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L27H14 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.85 / 0.85 | 0.15 / 0.15 |  |  |  |
| op | 0.71 / 0.59 | 0.15 / 0.29 | 0.14 / 0.12 |  |  |
| b | 0.81 / 0.70 | 0.07 / 0.14 | 0.06 / 0.06 | 0.06 / 0.10 |  |
| = | 0.70 / 0.57 | 0.02 / 0.05 | 0.06 / 0.06 | 0.10 / 0.13 | 0.11 / 0.19 |

<details><summary>1 o components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L27 o c5 (H14) | 100% / 100% | always | same | - | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L27 v c85 (kv3) | 0 / 97% | off (on 0) | always |

</details>

<details><summary>1 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L27 v c238 (kv3) | 70% / 0 | unexplained (best R2 0.19) | off (on 0) |

</details>

<details><summary>1 v components, main position `=` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L27 v c239 (kv3) | 2% / 60% | unexplained (best R2 0.08) | **tens(a,b)** (R2 0.51): a//10 in {0} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,6,7,8,9}; a//10 in {6,10} -> b//10 in {1,2,8}; a//10 in {7} -> b//10 in {0,1,2,8,9}; a//10 in {8} -> b//10 in {1,2}; a//10 in {9} -> b//10 in {1,2,3,4,5} |

</details>

</details>

<details><summary>L27H31 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.88 / 0.88 | 0.12 / 0.12 |  |  |  |
| op | 0.67 / 0.19 | 0.17 / 0.64 | 0.16 / 0.17 |  |  |
| b | 0.85 / 0.91 | 0.04 / 0.03 | 0.06 / 0.02 | 0.05 / 0.04 |  |
| = | 0.84 / 0.76 | 0.01 / 0.04 | 0.02 / 0.03 | 0.08 / 0.09 | 0.04 / 0.07 |

<details><summary>1 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L27 o c29 (H31) | 0 / 99% | off (on 0) | always | - | same |

</details>

<details><summary>1 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L27 v c56 (kv7) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

</details>

<details><summary>L28H4 (8 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.65 / 0.65 | 0.35 / 0.35 |  |  |  |
| op | 0.73 / 0.97 | 0.15 / 0.03 | 0.11 / 0.01 |  |  |
| b | 0.22 / 0.35 | 0.10 / 0.16 | 0.47 / 0.31 | 0.21 / 0.18 |  |
| = | 0.95 / 0.95 | 0.00 / 0.00 | 0.01 / 0.00 | 0.02 / 0.01 | 0.03 / 0.03 |

<details><summary>8 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L28 o c9 (H4) | 0 / 1% | off (on 0) | unexplained (best R2 0.04) | - | same |
| L28 o c25 (H4) | 0 / 0 | off (on 0) | same | - | same |
| L28 o c27 (H4) | 0 / 0 | off (on 0) | same | - | same |
| L28 o c38 (H4) | 0 / 0 | off (on 0) | same | - | same |
| L28 o c95 (H4) | 66% / 0 | **a** (R2 0.55): a in {3..4, 7, 9..10, 12..17, 19, 21..24, 26..29, 31..49, 51..54, 56..59, 61..64, 66..69, 71..74, 76..79, 82..83, 86..87, 90} | off (on 0) | a: mod5 +3% | - |
| L28 o c150 (H4) | 0 / 1% | off (on 0) | unexplained (best R2 0.04) | - | same |
| L28 o c170 (H4) | 0 / 0 | off (on 0) | same | - | same |
| L28 o c458 (H4) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L28 v c173 (kv1) | 0 / 100% | off (on 0) | always |

</details>

<details><summary>1 q components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L28 q c132 (H4) | 0 / 100% | off (on 0) | always |

</details>

</details>

<details><summary>L28H7 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.98 / 0.98 | 0.02 / 0.02 |  |  |  |
| op | 0.87 / 0.57 | 0.10 / 0.40 | 0.03 / 0.04 |  |  |
| b | 0.79 / 0.67 | 0.02 / 0.03 | 0.17 / 0.27 | 0.02 / 0.02 |  |
| = | 0.85 / 0.79 | 0.00 / 0.00 | 0.02 / 0.01 | 0.09 / 0.16 | 0.04 / 0.04 |

<details><summary>1 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L28 o c63 (H7) | 0 / 100% | off (on 0) | always | - | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L28 v c173 (kv1) | 0 / 100% | off (on 0) | always |

</details>

</details>

<details><summary>L28H10 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.94 / 0.94 | 0.06 / 0.06 |  |  |  |
| op | 0.63 / 0.54 | 0.22 / 0.32 | 0.15 / 0.15 |  |  |
| b | 0.78 / 0.73 | 0.08 / 0.11 | 0.05 / 0.03 | 0.09 / 0.13 |  |
| = | 0.29 / 0.34 | 0.04 / 0.07 | 0.12 / 0.08 | 0.30 / 0.20 | 0.26 / 0.30 |

<details><summary>1 o components, main position `&lt;BOS&gt;`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L28 o c122 (H10) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same | - | same |

</details>

</details>

<details><summary>L29H19 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.92 / 0.92 | 0.08 / 0.08 |  |  |  |
| op | 0.97 / 0.98 | 0.01 / 0.01 | 0.01 / 0.01 |  |  |
| b | 0.97 / 0.96 | 0.01 / 0.01 | 0.01 / 0.01 | 0.01 / 0.01 |  |
| = | 0.91 / 0.94 | 0.01 / 0.01 | 0.02 / 0.01 | 0.03 / 0.01 | 0.02 / 0.03 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L29 o c370 (H19) | 0 / 0 | off (on 0) | same | - | same |

</details>

</details>

<details><summary>L29H21 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.99 / 0.99 | 0.01 / 0.01 |  |  |  |
| op | 0.87 / 0.96 | 0.01 / 0.02 | 0.12 / 0.02 |  |  |
| b | 0.62 / 0.92 | 0.01 / 0.01 | 0.26 / 0.05 | 0.11 / 0.02 |  |
| = | 0.86 / 0.85 | 0.01 / 0.01 | 0.02 / 0.00 | 0.05 / 0.02 | 0.06 / 0.11 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L29 o c20 (H21) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>1 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L29 v c8 (kv5) | 100% / 100% | always | same |

</details>

<details><summary>2 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L29 v c4 (kv5) | 0 / 56% | off (on 0) | unexplained (best R2 0.43) |
| L29 v c212 (kv5) | 31% / 0 | unexplained (best R2 0.34) | off (on 0) |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L29 k c1 (kv5) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L29H25 (5 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.88 / 0.88 | 0.12 / 0.12 |  |  |  |
| op | 0.80 / 0.82 | 0.02 / 0.03 | 0.19 / 0.14 |  |  |
| b | 0.88 / 0.86 | 0.00 / 0.00 | 0.02 / 0.01 | 0.10 / 0.13 |  |
| = | 0.57 / 0.75 | 0.00 / 0.00 | 0.00 / 0.00 | 0.01 / 0.03 | 0.41 / 0.22 |

<details><summary>1 o components, main position `&lt;BOS&gt;`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L29 o c1 (H25) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same | - | same |

</details>

<details><summary>4 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L29 o c129 (H25) | 0 / 0 | off (on 0) | same | - | same |
| L29 o c324 (H25) | 0 / 0 | off (on 0) | same | - | same |
| L29 o c325 (H25) | 0 / 1% | off (on 0) | unexplained (best R2 0.09) | - | same |
| L29 o c352 (H25) | 0 / 1% | off (on 0) | unexplained (best R2 0.11) | - | same |

</details>

<details><summary>3 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L29 v c3 (kv6) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L29 v c86 (kv6) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L29 v c100 (kv6) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>3 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L29 v c31 (kv6) | 0 / 0 | off (on 0) | same |
| L29 v c118 (kv6) | 0 / 0 | off (on 0) | same |
| L29 v c249 (kv6) | 0 / 0 | off (on 0) | same |

</details>

<details><summary>1 v components, main position `=` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L29 v c214 (kv6) | 100% / 100% | always | same |

</details>

</details>

<details><summary>L29H27 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.98 / 0.98 | 0.02 / 0.02 |  |  |  |
| op | 0.86 / 0.87 | 0.11 / 0.11 | 0.03 / 0.01 |  |  |
| b | 0.88 / 0.79 | 0.01 / 0.01 | 0.10 / 0.19 | 0.01 / 0.01 |  |
| = | 0.81 / 0.80 | 0.00 / 0.00 | 0.01 / 0.01 | 0.16 / 0.15 | 0.02 / 0.04 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L29 o c0 (H27) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>3 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L29 v c3 (kv6) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L29 v c86 (kv6) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |
| L29 v c100 (kv6) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>3 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L29 v c31 (kv6) | 0 / 0 | off (on 0) | same |
| L29 v c118 (kv6) | 0 / 0 | off (on 0) | same |
| L29 v c249 (kv6) | 0 / 0 | off (on 0) | same |

</details>

<details><summary>1 v components, main position `=` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L29 v c214 (kv6) | 100% / 100% | always | same |

</details>

<details><summary>1 q components, main position `a`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L29 q c56 (H27) | 0 / 0 | off (on 0) | same |

</details>

</details>

<details><summary>L29H30 (2 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.79 / 0.79 | 0.21 / 0.21 |  |  |  |
| op | 0.59 / 0.42 | 0.09 / 0.25 | 0.31 / 0.33 |  |  |
| b | 0.35 / 0.36 | 0.05 / 0.04 | 0.17 / 0.08 | 0.44 / 0.52 |  |
| = | 0.13 / 0.09 | 0.04 / 0.06 | 0.14 / 0.08 | 0.41 / 0.52 | 0.27 / 0.25 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L29 o c7 (H30) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L29 o c73 (H30) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L29 v c43 (kv7) | 1% / 0 | **a** (R2 1.00): a in {55} | off (on 0) |

</details>

</details>

<details><summary>L30H9 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.97 / 0.97 | 0.03 / 0.03 |  |  |  |
| op | 0.70 / 0.91 | 0.17 / 0.07 | 0.13 / 0.02 |  |  |
| b | 0.66 / 0.87 | 0.13 / 0.08 | 0.17 / 0.02 | 0.04 / 0.03 |  |
| = | 0.65 / 0.60 | 0.03 / 0.07 | 0.08 / 0.06 | 0.07 / 0.12 | 0.16 / 0.15 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L30 o c283 (H9) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L30 v c252 (kv2) | 0 / 21% | off (on 0) | **a//10** (R2 0.90): (tens) a in {1..21} |

</details>

</details>

<details><summary>L30H10 (2 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.91 / 0.91 | 0.09 / 0.09 |  |  |  |
| op | 0.66 / 0.33 | 0.17 / 0.53 | 0.17 / 0.14 |  |  |
| b | 0.66 / 0.51 | 0.06 / 0.10 | 0.10 / 0.17 | 0.19 / 0.23 |  |
| = | 0.49 / 0.55 | 0.03 / 0.03 | 0.07 / 0.02 | 0.17 / 0.12 | 0.24 / 0.28 |

<details><summary>1 o components, main position `op`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L30 o c83 (H10) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {80} | - | same |

</details>

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L30 o c229 (H10) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>1 v components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L30 v c252 (kv2) | 0 / 21% | off (on 0) | **a//10** (R2 0.90): (tens) a in {1..21} |

</details>

</details>

<details><summary>L30H18 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.88 / 0.88 | 0.12 / 0.12 |  |  |  |
| op | 0.55 / 0.70 | 0.10 / 0.18 | 0.35 / 0.12 |  |  |
| b | 0.75 / 0.69 | 0.06 / 0.17 | 0.08 / 0.02 | 0.11 / 0.13 |  |
| = | 0.23 / 0.44 | 0.03 / 0.06 | 0.30 / 0.06 | 0.27 / 0.12 | 0.17 / 0.32 |

<details><summary>1 o components, main position `&lt;BOS&gt;`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L30 o c446 (H18) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same | - | same |

</details>

</details>

<details><summary>L30H20 (29 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.78 / 0.78 | 0.22 / 0.22 |  |  |  |
| op | 0.52 / 0.48 | 0.08 / 0.10 | 0.40 / 0.42 |  |  |
| b | 0.63 / 0.59 | 0.05 / 0.05 | 0.23 / 0.27 | 0.09 / 0.09 |  |
| = | 0.66 / 0.70 | 0.04 / 0.03 | 0.09 / 0.08 | 0.09 / 0.05 | 0.12 / 0.14 |

<details><summary>29 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L30 o c46 (H20) | 0 / 5% | off (on 0) | **a** (R2 0.87): a in {1..5} | - | same |
| L30 o c61 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c69 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c84 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c87 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c137 (H20) | 0 / 21% | off (on 0) | **a//10** (R2 0.85): (tens) a in {1..19} | - | a: mod100 +2%, mod50 +3%, mod25 +2% |
| L30 o c180 (H20) | 0 / 1% | off (on 0) | **a** (R2 0.70): a in {1} | - | same |
| L30 o c181 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c183 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c200 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c215 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c291 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c312 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c313 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c334 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c343 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c344 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c361 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c362 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c372 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c413 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c420 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c428 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c436 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c458 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c462 (H20) | 0 / 2% | off (on 0) | **a** (R2 0.74): a in {1} | - | same |
| L30 o c477 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c493 (H20) | 0 / 0 | off (on 0) | same | - | same |
| L30 o c507 (H20) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>2 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L30 k c22 (kv5) | 0 / 8% | off (on 0) | **a** (R2 1.00): a in {1..8} |
| L30 k c26 (kv5) | 0 / 19% | off (on 0) | **a//10** (R2 0.98): (tens) a in {1..19} |

</details>

</details>

<details><summary>L30H22 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.91 / 0.91 | 0.09 / 0.09 |  |  |  |
| op | 0.65 / 0.47 | 0.16 / 0.41 | 0.19 / 0.12 |  |  |
| b | 0.39 / 0.60 | 0.10 / 0.12 | 0.37 / 0.15 | 0.13 / 0.13 |  |
| = | 0.54 / 0.37 | 0.04 / 0.08 | 0.10 / 0.05 | 0.19 / 0.28 | 0.12 / 0.22 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L30 o c404 (H22) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>2 k components, main position `op` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L30 k c22 (kv5) | 0 / 8% | off (on 0) | **a** (R2 1.00): a in {1..8} |
| L30 k c26 (kv5) | 0 / 19% | off (on 0) | **a//10** (R2 0.98): (tens) a in {1..19} |

</details>

</details>

<details><summary>L30H25 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.65 / 0.65 | 0.35 / 0.35 |  |  |  |
| op | 0.30 / 0.37 | 0.05 / 0.03 | 0.65 / 0.60 |  |  |
| b | 0.30 / 0.33 | 0.01 / 0.02 | 0.03 / 0.03 | 0.65 / 0.62 |  |
| = | 0.36 / 0.35 | 0.00 / 0.00 | 0.00 / 0.01 | 0.01 / 0.02 | 0.63 / 0.62 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L30 o c18 (H25) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>1 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L30 v c3 (kv6) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>2 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L30 v c114 (kv6) | 0 / 0 | off (on 0) | same |
| L30 v c135 (kv6) | 0 / 0 | off (on 0) | same |

</details>

</details>

<details><summary>L31H5 (2 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.70 / 0.70 | 0.30 / 0.30 |  |  |  |
| op | 0.83 / 0.42 | 0.01 / 0.00 | 0.16 / 0.58 |  |  |
| b | 0.70 / 0.80 | 0.00 / 0.00 | 0.02 / 0.01 | 0.28 / 0.18 |  |
| = | 0.51 / 0.64 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.49 / 0.35 |

<details><summary>2 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L31 o c56 (H5) | 0 / 0 | off (on 0) | same | - | same |
| L31 o c168 (H5) | 0 / 0 | off (on 0) | same | - | same |

</details>

</details>

<details><summary>L31H12 (2 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.97 / 0.97 | 0.03 / 0.03 |  |  |  |
| op | 0.82 / 0.18 | 0.04 / 0.19 | 0.15 / 0.63 |  |  |
| b | 0.93 / 0.94 | 0.01 / 0.02 | 0.02 / 0.01 | 0.03 / 0.04 |  |
| = | 0.34 / 0.37 | 0.01 / 0.01 | 0.03 / 0.02 | 0.09 / 0.07 | 0.53 / 0.53 |

<details><summary>2 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L31 o c77 (H12) | 0 / 0 | off (on 0) | same | - | same |
| L31 o c135 (H12) | 100% / 100% | always | same | - | same |

</details>

<details><summary>1 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L31 v c250 (kv3) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>2 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L31 v c28 (kv3) | 100% / 100% | always | same |
| L31 v c43 (kv3) | 100% / 100% | always | same |

</details>

<details><summary>2 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L31 v c123 (kv3) | 1% / 99% | unexplained (best R2 0.14) | always |
| L31 v c168 (kv3) | 99% / 0 | always | off (on 0) |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L31 k c4 (kv3) | 100% / 100% | always | same |

</details>

<details><summary>1 k components, main position `=` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L31 k c1 (kv3) | 89% / 3% | unexplained (best R2 0.18) | unexplained (best R2 0.32) |

</details>

</details>

<details><summary>L31H14 (4 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.00 / 0.00 | 1.00 / 1.00 |  |  |  |
| op | 0.00 / 0.00 | 0.01 / 0.11 | 0.99 / 0.89 |  |  |
| b | 0.00 / 0.00 | 0.00 / 0.00 | 0.01 / 0.00 | 0.99 / 1.00 |  |
| = | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.02 / 0.01 | 0.97 / 0.99 |

<details><summary>1 o components, main position `&lt;BOS&gt;`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L31 o c74 (H14) | 0 / 0 | at &lt;BOS&gt; (constant), on 0 | same | - | same |

</details>

<details><summary>3 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L31 o c7 (H14) | 100% / 100% | always | same | - | same |
| L31 o c18 (H14) | 0 / 0 | off (on 0) | same | - | same |
| L31 o c58 (H14) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>1 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L31 v c250 (kv3) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>2 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L31 v c28 (kv3) | 100% / 100% | always | same |
| L31 v c43 (kv3) | 100% / 100% | always | same |

</details>

<details><summary>2 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L31 v c123 (kv3) | 1% / 99% | unexplained (best R2 0.14) | always |
| L31 v c168 (kv3) | 99% / 0 | always | off (on 0) |

</details>

<details><summary>1 q components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L31 q c1 (H14) | 100% / 100% | always | same |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L31 k c4 (kv3) | 100% / 100% | always | same |

</details>

<details><summary>1 k components, main position `=` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L31 k c1 (kv3) | 89% / 3% | unexplained (best R2 0.18) | unexplained (best R2 0.32) |

</details>

</details>

<details><summary>L31H15 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.86 / 0.86 | 0.14 / 0.14 |  |  |  |
| op | 0.64 / 0.83 | 0.17 / 0.12 | 0.19 / 0.05 |  |  |
| b | 0.58 / 0.66 | 0.07 / 0.06 | 0.14 / 0.11 | 0.21 / 0.17 |  |
| = | 0.64 / 0.67 | 0.01 / 0.01 | 0.04 / 0.01 | 0.06 / 0.08 | 0.25 / 0.23 |

<details><summary>1 o components, main position `=`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L31 o c165 (H15) | 98% / 33% | always | **tens(a,b)** (R2 0.52): a//10 in {0,1} -> b//10 in {0,10}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3,4} -> b//10 in {0}; a//10 in {5} -> b//10 in {0,4,5}; a//10 in {6} -> b//10 in {5,6}; a//10 in {7} -> b//10 in {0,6}; a//10 in {8} -> b//10 in {0,1,4,6,7,8}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | - | same |

</details>

<details><summary>1 v components, main position `&lt;BOS&gt;` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L31 v c250 (kv3) | 100% / 100% | at &lt;BOS&gt; (constant), on 100% | same |

</details>

<details><summary>2 v components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L31 v c28 (kv3) | 100% / 100% | always | same |
| L31 v c43 (kv3) | 100% / 100% | always | same |

</details>

<details><summary>2 v components, main position `b` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L31 v c123 (kv3) | 1% / 99% | unexplained (best R2 0.14) | always |
| L31 v c168 (kv3) | 99% / 0 | always | off (on 0) |

</details>

<details><summary>1 q components, main position `&lt;BOS&gt;`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L31 q c51 (H15) | 0 / 0 | at &lt;BOS&gt; (constant), on 0 | same |

</details>

<details><summary>1 k components, main position `a` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L31 k c4 (kv3) | 100% / 100% | always | same |

</details>

<details><summary>1 k components, main position `=` (kv head shared by 4 query heads)</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L31 k c1 (kv3) | 89% / 3% | unexplained (best R2 0.18) | unexplained (best R2 0.32) |

</details>

</details>

<details><summary>L31H27 (1 alive o components)</summary>

| query \ key (add / sub) | &lt;BOS&gt; | a | op | b | = |
|---|---|---|---|---|---|
| a | 0.59 / 0.59 | 0.41 / 0.41 |  |  |  |
| op | 0.71 / 0.81 | 0.16 / 0.10 | 0.13 / 0.09 |  |  |
| b | 0.72 / 0.73 | 0.07 / 0.10 | 0.06 / 0.03 | 0.15 / 0.15 |  |
| = | 0.72 / 0.72 | 0.01 / 0.02 | 0.02 / 0.01 | 0.04 / 0.03 | 0.21 / 0.22 |

<details><summary>1 o components, main position `b`</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L31 o c458 (H27) | 0 / 0 | off (on 0) | same | - | same |

</details>

</details>

