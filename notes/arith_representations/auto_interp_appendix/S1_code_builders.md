# §1 — every writer of the operand codes

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

## The a code at the `a` token

<details><summary>`mlp_in.0` at `a` (addition prompts): who writes the a code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 1.05 | embedding +1.05 |
| mod 50 | 1.05 | embedding +1.05 |
| mod 25 | 1.05 | embedding +1.05 |
| mod 20 | 1.05 | embedding +1.05 |
| mod 10 | 1.05 | embedding +1.05 |
| mod 5 | 1.05 | embedding +1.05 |
| mod 4 | 1.06 | embedding +1.06 |
| mod 2 | 1.05 | embedding +1.05 |

</details>

<details><summary>`mlp_in.1` at `a` (addition prompts): who writes the a code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.90 | L0 MLP +0.53, embedding +0.37 |
| mod 50 | 0.89 | L0 MLP +0.50, embedding +0.39 |
| mod 25 | 0.86 | L0 MLP +0.43, embedding +0.43 |
| mod 20 | 0.90 | L0 MLP +0.48, embedding +0.41 |
| mod 10 | 0.96 | L0 MLP +0.59, embedding +0.37 |
| mod 5 | 0.97 | L0 MLP +0.58, embedding +0.38 |
| mod 4 | 0.82 | embedding +0.51, L0 MLP +0.30 |
| mod 2 | 1.02 | L0 MLP +0.62, embedding +0.40 |

<details><summary>period 100: 3 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c6 | +0.091 | 50% | **a//10** (R2 0.96): (tens) a in {51..100} |
| L0 down c5 | +0.076 | 84% | **a** (R2 1.00): a in {13..96} |
| L0 down c23 | +0.047 | 39% | **a//10** (R2 1.00): (tens) a in {1..39} |

</details>

<details><summary>period 50: 3 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c6 | +0.044 | 50% | **a//10** (R2 0.96): (tens) a in {51..100} |
| L0 down c23 | +0.030 | 39% | **a//10** (R2 1.00): (tens) a in {1..39} |
| L0 down c22 | +0.021 | 19% | **a//10** (R2 0.94): (tens) a in {61..79} |

</details>

<details><summary>period 25: 1 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c252 | +0.021 | 9% | **a//10** (R2 1.00): (tens) a in {1..9} |

</details>

<details><summary>period 20: 1 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c252 | +0.026 | 9% | **a//10** (R2 1.00): (tens) a in {1..9} |

</details>

<details><summary>period 10: 10 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c36 | +0.071 | 10% | **a%10** (R2 1.00): a mod 10 in {1} |
| L0 down c38 | +0.057 | 10% | **a%10** (R2 1.00): a mod 10 in {9} |
| L0 down c145 | +0.052 | 10% | **a%10** (R2 1.00): a mod 10 in {0} |
| L0 down c45 | +0.045 | 10% | **a%10** (R2 1.00): a mod 10 in {5} |
| L0 down c52 | +0.042 | 10% | **a%10** (R2 1.00): a mod 10 in {3} |
| L0 down c62 | +0.040 | 10% | **a%10** (R2 1.00): a mod 10 in {4} |
| L0 down c44 | +0.040 | 10% | **a%10** (R2 1.00): a mod 10 in {7} |
| L0 down c50 | +0.039 | 10% | **a%10** (R2 1.00): a mod 10 in {8} |
| L0 down c55 | +0.038 | 10% | **a%10** (R2 1.00): a mod 10 in {6} |
| L0 down c81 | +0.036 | 10% | **a%10** (R2 1.00): a mod 10 in {2} |

</details>

<details><summary>period 5: 11 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c162 | +0.102 | 19% | **a%5** (R2 0.94): a mod 5 in {0} |
| L0 down c36 | +0.066 | 10% | **a%10** (R2 1.00): a mod 10 in {1} |
| L0 down c38 | +0.055 | 10% | **a%10** (R2 1.00): a mod 10 in {9} |
| L0 down c45 | +0.046 | 10% | **a%10** (R2 1.00): a mod 10 in {5} |
| L0 down c145 | +0.043 | 10% | **a%10** (R2 1.00): a mod 10 in {0} |
| L0 down c62 | +0.042 | 10% | **a%10** (R2 1.00): a mod 10 in {4} |
| L0 down c55 | +0.040 | 10% | **a%10** (R2 1.00): a mod 10 in {6} |
| L0 down c50 | +0.039 | 10% | **a%10** (R2 1.00): a mod 10 in {8} |
| L0 down c44 | +0.038 | 10% | **a%10** (R2 1.00): a mod 10 in {7} |
| L0 down c52 | +0.037 | 10% | **a%10** (R2 1.00): a mod 10 in {3} |
| L0 down c81 | +0.032 | 10% | **a%10** (R2 1.00): a mod 10 in {2} |

</details>

<details><summary>period 4: 1 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c127 | +0.074 | 8% | **a** (R2 1.00): a in {16, 24, 32, 36, 48, 64, 72, 96} |

</details>

<details><summary>period 2: 10 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c36 | +0.070 | 10% | **a%10** (R2 1.00): a mod 10 in {1} |
| L0 down c38 | +0.069 | 10% | **a%10** (R2 1.00): a mod 10 in {9} |
| L0 down c52 | +0.060 | 10% | **a%10** (R2 1.00): a mod 10 in {3} |
| L0 down c44 | +0.059 | 10% | **a%10** (R2 1.00): a mod 10 in {7} |
| L0 down c45 | +0.056 | 10% | **a%10** (R2 1.00): a mod 10 in {5} |
| L0 down c50 | +0.056 | 10% | **a%10** (R2 1.00): a mod 10 in {8} |
| L0 down c55 | +0.051 | 10% | **a%10** (R2 1.00): a mod 10 in {6} |
| L0 down c62 | +0.051 | 10% | **a%10** (R2 1.00): a mod 10 in {4} |
| L0 down c145 | +0.046 | 10% | **a%10** (R2 1.00): a mod 10 in {0} |
| L0 down c81 | +0.045 | 10% | **a%10** (R2 1.00): a mod 10 in {2} |

</details>

</details>

<details><summary>`mlp_in.2` at `a` (addition prompts): who writes the a code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.81 | L0 MLP +0.39, embedding +0.26, L1 MLP +0.15 |
| mod 50 | 0.80 | L0 MLP +0.38, embedding +0.28, L1 MLP +0.13 |
| mod 25 | 0.74 | L0 MLP +0.34, embedding +0.31, L1 MLP +0.09 |
| mod 20 | 0.78 | L0 MLP +0.36, embedding +0.30, L1 MLP +0.12 |
| mod 10 | 0.84 | L0 MLP +0.45, embedding +0.29, L1 MLP +0.11 |
| mod 5 | 0.83 | L0 MLP +0.45, embedding +0.30, L1 MLP +0.08 |
| mod 4 | 0.66 | embedding +0.39, L0 MLP +0.25, L1 MLP +0.03 |
| mod 2 | 0.87 | L0 MLP +0.50, embedding +0.32, L1 MLP +0.05 |

<details><summary>period 100: 5 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c6 | +0.069 | 50% | **a//10** (R2 0.96): (tens) a in {51..100} |
| L0 down c5 | +0.058 | 84% | **a** (R2 1.00): a in {13..96} |
| L1 down c96 | +0.044 | 27% | **a//10** (R2 0.92): (tens) a in {1..27} |
| L0 down c23 | +0.035 | 39% | **a//10** (R2 1.00): (tens) a in {1..39} |
| L1 down c17 | +0.033 | 100% | always |

</details>

<details><summary>period 50: 3 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L1 down c96 | +0.036 | 27% | **a//10** (R2 0.92): (tens) a in {1..27} |
| L0 down c6 | +0.034 | 50% | **a//10** (R2 0.96): (tens) a in {51..100} |
| L0 down c23 | +0.023 | 39% | **a//10** (R2 1.00): (tens) a in {1..39} |

</details>

<details><summary>period 20: 1 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c252 | +0.021 | 9% | **a//10** (R2 1.00): (tens) a in {1..9} |

</details>

<details><summary>period 10: 10 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c36 | +0.056 | 10% | **a%10** (R2 1.00): a mod 10 in {1} |
| L0 down c38 | +0.042 | 10% | **a%10** (R2 1.00): a mod 10 in {9} |
| L0 down c145 | +0.040 | 10% | **a%10** (R2 1.00): a mod 10 in {0} |
| L0 down c45 | +0.034 | 10% | **a%10** (R2 1.00): a mod 10 in {5} |
| L0 down c62 | +0.031 | 10% | **a%10** (R2 1.00): a mod 10 in {4} |
| L0 down c50 | +0.031 | 10% | **a%10** (R2 1.00): a mod 10 in {8} |
| L0 down c44 | +0.031 | 10% | **a%10** (R2 1.00): a mod 10 in {7} |
| L0 down c52 | +0.030 | 10% | **a%10** (R2 1.00): a mod 10 in {3} |
| L0 down c55 | +0.029 | 10% | **a%10** (R2 1.00): a mod 10 in {6} |
| L0 down c81 | +0.028 | 10% | **a%10** (R2 1.00): a mod 10 in {2} |

</details>

<details><summary>period 5: 11 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c162 | +0.082 | 19% | **a%5** (R2 0.94): a mod 5 in {0} |
| L0 down c36 | +0.054 | 10% | **a%10** (R2 1.00): a mod 10 in {1} |
| L0 down c38 | +0.042 | 10% | **a%10** (R2 1.00): a mod 10 in {9} |
| L0 down c45 | +0.034 | 10% | **a%10** (R2 1.00): a mod 10 in {5} |
| L0 down c145 | +0.033 | 10% | **a%10** (R2 1.00): a mod 10 in {0} |
| L0 down c62 | +0.033 | 10% | **a%10** (R2 1.00): a mod 10 in {4} |
| L0 down c55 | +0.032 | 10% | **a%10** (R2 1.00): a mod 10 in {6} |
| L0 down c50 | +0.031 | 10% | **a%10** (R2 1.00): a mod 10 in {8} |
| L0 down c44 | +0.029 | 10% | **a%10** (R2 1.00): a mod 10 in {7} |
| L0 down c52 | +0.028 | 10% | **a%10** (R2 1.00): a mod 10 in {3} |
| L0 down c81 | +0.025 | 10% | **a%10** (R2 1.00): a mod 10 in {2} |

</details>

<details><summary>period 4: 1 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c127 | +0.060 | 8% | **a** (R2 1.00): a in {16, 24, 32, 36, 48, 64, 72, 96} |

</details>

<details><summary>period 2: 10 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c36 | +0.057 | 10% | **a%10** (R2 1.00): a mod 10 in {1} |
| L0 down c38 | +0.056 | 10% | **a%10** (R2 1.00): a mod 10 in {9} |
| L0 down c52 | +0.049 | 10% | **a%10** (R2 1.00): a mod 10 in {3} |
| L0 down c44 | +0.049 | 10% | **a%10** (R2 1.00): a mod 10 in {7} |
| L0 down c45 | +0.044 | 10% | **a%10** (R2 1.00): a mod 10 in {5} |
| L0 down c50 | +0.043 | 10% | **a%10** (R2 1.00): a mod 10 in {8} |
| L0 down c62 | +0.042 | 10% | **a%10** (R2 1.00): a mod 10 in {4} |
| L0 down c55 | +0.041 | 10% | **a%10** (R2 1.00): a mod 10 in {6} |
| L0 down c145 | +0.037 | 10% | **a%10** (R2 1.00): a mod 10 in {0} |
| L0 down c81 | +0.036 | 10% | **a%10** (R2 1.00): a mod 10 in {2} |

</details>

</details>

<details><summary>`mlp_in.3` at `a` (addition prompts): who writes the a code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.74 | L2 MLP +0.25, L0 MLP +0.24, embedding +0.15, L1 MLP +0.09 |
| mod 50 | 0.72 | L0 MLP +0.24, L2 MLP +0.23, embedding +0.17, L1 MLP +0.08 |
| mod 25 | 0.65 | L0 MLP +0.22, L2 MLP +0.19, embedding +0.18, L1 MLP +0.06 |
| mod 20 | 0.68 | L0 MLP +0.25, embedding +0.18, L2 MLP +0.18, L1 MLP +0.08 |
| mod 10 | 0.74 | L0 MLP +0.31, embedding +0.18, L2 MLP +0.16, L1 MLP +0.07 |
| mod 5 | 0.76 | L0 MLP +0.29, L2 MLP +0.24, embedding +0.18, L1 MLP +0.05 |
| mod 4 | 0.56 | embedding +0.23, L0 MLP +0.17, L2 MLP +0.14 |
| mod 2 | 0.76 | L0 MLP +0.36, embedding +0.21, L2 MLP +0.15, L1 MLP +0.04 |

<details><summary>period 100: 8 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c6 | +0.047 | 50% | **a//10** (R2 0.96): (tens) a in {51..100} |
| L0 down c5 | +0.039 | 84% | **a** (R2 1.00): a in {13..96} |
| L2 down c92 | +0.037 | 31% | **a//10** (R2 1.00): (tens) a in {70..100} |
| L2 down c11 | +0.033 | 23% | **a** (R2 1.00): a in {27..49} |
| L2 down c90 | +0.032 | 22% | **a** (R2 1.00): a in {11..32} |
| L2 down c79 | +0.025 | 14% | **a** (R2 1.00): a in {1..14} |
| L1 down c96 | +0.025 | 27% | **a//10** (R2 0.92): (tens) a in {1..27} |
| L0 down c23 | +0.023 | 39% | **a//10** (R2 1.00): (tens) a in {1..39} |

</details>

<details><summary>period 50: 6 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L2 down c92 | +0.031 | 31% | **a//10** (R2 1.00): (tens) a in {70..100} |
| L2 down c79 | +0.028 | 14% | **a** (R2 1.00): a in {1..14} |
| L2 down c90 | +0.027 | 22% | **a** (R2 1.00): a in {11..32} |
| L1 down c96 | +0.023 | 27% | **a//10** (R2 0.92): (tens) a in {1..27} |
| L0 down c6 | +0.023 | 50% | **a//10** (R2 0.96): (tens) a in {51..100} |
| L2 down c639 | +0.022 | 14% | **a** (R2 1.00): a in {2..15} |

</details>

<details><summary>period 25: 1 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L2 down c79 | +0.024 | 14% | **a** (R2 1.00): a in {1..14} |

</details>

<details><summary>period 20: 1 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L2 down c79 | +0.027 | 14% | **a** (R2 1.00): a in {1..14} |

</details>

<details><summary>period 10: 12 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L2 down c9 | +0.042 | 10% | **a** (R2 1.00): a in {1..2, 21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 10 in {1}, R2 0.80] |
| L0 down c36 | +0.040 | 10% | **a%10** (R2 1.00): a mod 10 in {1} |
| L0 down c38 | +0.029 | 10% | **a%10** (R2 1.00): a mod 10 in {9} |
| L2 down c147 | +0.028 | 12% | **a%50** (R2 0.90): a mod 50 in {0, 10, 20, 30, 40} [coarser: a mod 10 in {0}, R2 0.83] |
| L0 down c145 | +0.028 | 10% | **a%10** (R2 1.00): a mod 10 in {0} |
| L0 down c45 | +0.024 | 10% | **a%10** (R2 1.00): a mod 10 in {5} |
| L0 down c50 | +0.022 | 10% | **a%10** (R2 1.00): a mod 10 in {8} |
| L0 down c62 | +0.022 | 10% | **a%10** (R2 1.00): a mod 10 in {4} |
| L0 down c52 | +0.022 | 10% | **a%10** (R2 1.00): a mod 10 in {3} |
| L0 down c55 | +0.021 | 10% | **a%10** (R2 1.00): a mod 10 in {6} |
| L0 down c44 | +0.021 | 10% | **a%10** (R2 1.00): a mod 10 in {7} |
| L0 down c81 | +0.020 | 10% | **a%10** (R2 1.00): a mod 10 in {2} |

</details>

<details><summary>period 5: 8 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L2 down c473 | +0.121 | 20% | **a%5** (R2 1.00): a mod 5 in {0} |
| L0 down c162 | +0.058 | 19% | **a%5** (R2 0.94): a mod 5 in {0} |
| L2 down c9 | +0.038 | 10% | **a** (R2 1.00): a in {1..2, 21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 10 in {1}, R2 0.80] |
| L0 down c36 | +0.033 | 10% | **a%10** (R2 1.00): a mod 10 in {1} |
| L0 down c38 | +0.025 | 10% | **a%10** (R2 1.00): a mod 10 in {9} |
| L2 down c147 | +0.023 | 12% | **a%50** (R2 0.90): a mod 50 in {0, 10, 20, 30, 40} [coarser: a mod 10 in {0}, R2 0.83] |
| L0 down c145 | +0.023 | 10% | **a%10** (R2 1.00): a mod 10 in {0} |
| L0 down c45 | +0.022 | 10% | **a%10** (R2 1.00): a mod 10 in {5} |

</details>

<details><summary>period 4: 2 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L2 down c118 | +0.057 | 19% | **a** (R2 1.00): a in {16, 24, 32, 36, 44, 46, 48..49, 52, 54, 56, 64, 72, 74, 76, 84, 86, 88, 96} |
| L0 down c127 | +0.041 | 8% | **a** (R2 1.00): a in {16, 24, 32, 36, 48, 64, 72, 96} |

</details>

<details><summary>period 2: 13 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c36 | +0.041 | 10% | **a%10** (R2 1.00): a mod 10 in {1} |
| L0 down c38 | +0.040 | 10% | **a%10** (R2 1.00): a mod 10 in {9} |
| L2 down c118 | +0.037 | 19% | **a** (R2 1.00): a in {16, 24, 32, 36, 44, 46, 48..49, 52, 54, 56, 64, 72, 74, 76, 84, 86, 88, 96} |
| L0 down c52 | +0.037 | 10% | **a%10** (R2 1.00): a mod 10 in {3} |
| L0 down c44 | +0.035 | 10% | **a%10** (R2 1.00): a mod 10 in {7} |
| L0 down c50 | +0.032 | 10% | **a%10** (R2 1.00): a mod 10 in {8} |
| L0 down c45 | +0.030 | 10% | **a%10** (R2 1.00): a mod 10 in {5} |
| L2 down c9 | +0.030 | 10% | **a** (R2 1.00): a in {1..2, 21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 10 in {1}, R2 0.80] |
| L0 down c62 | +0.030 | 10% | **a%10** (R2 1.00): a mod 10 in {4} |
| L0 down c55 | +0.029 | 10% | **a%10** (R2 1.00): a mod 10 in {6} |
| L0 down c145 | +0.028 | 10% | **a%10** (R2 1.00): a mod 10 in {0} |
| L0 down c81 | +0.025 | 10% | **a%10** (R2 1.00): a mod 10 in {2} |
| L2 down c147 | +0.024 | 12% | **a%50** (R2 0.90): a mod 50 in {0, 10, 20, 30, 40} [coarser: a mod 10 in {0}, R2 0.83] |

</details>

</details>

<details><summary>`mlp_in.5` at `a` (addition prompts): who writes the a code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.70 | L4 MLP +0.17, L3 MLP +0.16, L2 MLP +0.13, L0 MLP +0.12, embedding +0.07, L1 MLP +0.05 |
| mod 50 | 0.70 | L4 MLP +0.17, L3 MLP +0.17, L0 MLP +0.13, L2 MLP +0.11, embedding +0.08, L1 MLP +0.04 |
| mod 25 | 0.59 | L4 MLP +0.14, L3 MLP +0.12, L0 MLP +0.12, L2 MLP +0.10, embedding +0.08, L1 MLP +0.03 |
| mod 20 | 0.63 | L3 MLP +0.16, L0 MLP +0.13, L4 MLP +0.13, L2 MLP +0.09, embedding +0.08, L1 MLP +0.04 |
| mod 10 | 0.73 | L3 MLP +0.20, L0 MLP +0.17, L4 MLP +0.15, embedding +0.08, L2 MLP +0.08, L1 MLP +0.04 |
| mod 5 | 0.73 | L3 MLP +0.20, L0 MLP +0.16, L4 MLP +0.13, L2 MLP +0.12, embedding +0.09, L1 MLP +0.03 |
| mod 4 | 0.46 | embedding +0.11, L0 MLP +0.09, L4 MLP +0.09, L2 MLP +0.08, L3 MLP +0.08 |
| mod 2 | 0.83 | L4 MLP +0.25, L0 MLP +0.20, L3 MLP +0.19, embedding +0.10, L2 MLP +0.07 |

<details><summary>period 100: 5 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L4 down c132 | +0.033 | 23% | **a** (R2 1.00): a in {2..23, 25} |
| L3 down c11 | +0.027 | 53% | **a** (R2 1.00): a in {47..49, 51..100} |
| L3 down c44 | +0.026 | 23% | **a** (R2 1.00): a in {2..23, 25} |
| L0 down c6 | +0.025 | 50% | **a//10** (R2 0.96): (tens) a in {51..100} |
| L0 down c5 | +0.023 | 84% | **a** (R2 1.00): a in {13..96} |

</details>

<details><summary>period 50: 3 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L4 down c132 | +0.032 | 23% | **a** (R2 1.00): a in {2..23, 25} |
| L3 down c44 | +0.031 | 23% | **a** (R2 1.00): a in {2..23, 25} |
| L4 down c158 | +0.020 | 17% | **a** (R2 1.00): a in {83..99} |

</details>

<details><summary>period 20: 1 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L4 down c127 | +0.022 | 9% | **a//10** (R2 1.00): (tens) a in {1..9} |

</details>

<details><summary>period 10: 6 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L4 down c66 | +0.095 | 39% | **a%10** (R2 0.96): a mod 10 in {6..9} |
| L3 down c63 | +0.058 | 30% | **a%10** (R2 1.00): a mod 10 in {5..7} |
| L3 down c136 | +0.056 | 37% | **a%10** (R2 0.91): a mod 10 in {2, 7..9} |
| L3 down c54 | +0.033 | 20% | **a%10** (R2 1.00): a mod 10 in {1, 9} |
| L0 down c36 | +0.023 | 10% | **a%10** (R2 1.00): a mod 10 in {1} |
| L2 down c9 | +0.022 | 10% | **a** (R2 1.00): a in {1..2, 21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 10 in {1}, R2 0.80] |

</details>

<details><summary>period 5: 8 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L4 down c43 | +0.095 | 21% | **a%5** (R2 0.94): a mod 5 in {0} |
| L3 down c615 | +0.067 | 19% | **a%5** (R2 0.94): a mod 5 in {0} |
| L2 down c473 | +0.064 | 20% | **a%5** (R2 1.00): a mod 5 in {0} |
| L3 down c68 | +0.047 | 38% | **a%50** (R2 0.91): a mod 50 in {4, 9, 13..14, 17, 19, 23..24, 27, 29, 33..34, 37, 39, 43..44, 49} [coarser: a mod 10 in {3..4, 7, 9}, R2 0.86] |
| L0 down c162 | +0.033 | 19% | **a%5** (R2 0.94): a mod 5 in {0} |
| L3 down c54 | +0.030 | 20% | **a%10** (R2 1.00): a mod 10 in {1, 9} |
| L2 down c9 | +0.020 | 10% | **a** (R2 1.00): a in {1..2, 21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 10 in {1}, R2 0.80] |
| L3 down c63 | +0.020 | 30% | **a%10** (R2 1.00): a mod 10 in {5..7} |

</details>

<details><summary>period 4: 4 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L3 down c153 | +0.038 | 6% | **a** (R2 1.00): a in {12, 24, 36, 48, 72, 96} |
| L2 down c118 | +0.035 | 19% | **a** (R2 1.00): a in {16, 24, 32, 36, 44, 46, 48..49, 52, 54, 56, 64, 72, 74, 76, 84, 86, 88, 96} |
| L4 down c135 | +0.026 | 46% | **a%10** (R2 0.90): a mod 10 in {0, 2, 4, 6, 8} [coarser: a mod 2 in {0}, R2 0.85] |
| L0 down c127 | +0.023 | 8% | **a** (R2 1.00): a in {16, 24, 32, 36, 48, 64, 72, 96} |

</details>

<details><summary>period 2: 7 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L4 down c135 | +0.220 | 46% | **a%10** (R2 0.90): a mod 10 in {0, 2, 4, 6, 8} [coarser: a mod 2 in {0}, R2 0.85] |
| L3 down c54 | +0.082 | 20% | **a%10** (R2 1.00): a mod 10 in {1, 9} |
| L3 down c63 | +0.028 | 30% | **a%10** (R2 1.00): a mod 10 in {5..7} |
| L0 down c36 | +0.025 | 10% | **a%10** (R2 1.00): a mod 10 in {1} |
| L0 down c38 | +0.022 | 10% | **a%10** (R2 1.00): a mod 10 in {9} |
| L0 down c44 | +0.020 | 10% | **a%10** (R2 1.00): a mod 10 in {7} |
| L0 down c52 | +0.020 | 10% | **a%10** (R2 1.00): a mod 10 in {3} |

</details>

</details>

<details><summary>`mlp_in.8` at `a` (addition prompts): who writes the a code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.67 | L7 MLP +0.10, L5 MLP +0.10, L6 MLP +0.09, L4 MLP +0.09, L3 MLP +0.08, L0 MLP +0.07, L2 MLP +0.07, embedding +0.03, L1 MLP +0.03 |
| mod 50 | 0.66 | L4 MLP +0.10, L5 MLP +0.09, L7 MLP +0.09, L3 MLP +0.09, L6 MLP +0.08, L0 MLP +0.08, L2 MLP +0.07, embedding +0.04, L1 MLP +0.03 |
| mod 25 | 0.55 | L4 MLP +0.08, L0 MLP +0.07, L3 MLP +0.07, L6 MLP +0.07, L5 MLP +0.07, L7 MLP +0.07, L2 MLP +0.06, embedding +0.04, L1 MLP +0.02 |
| mod 20 | 0.58 | L3 MLP +0.10, L0 MLP +0.08, L4 MLP +0.08, L7 MLP +0.06, L5 MLP +0.06, L6 MLP +0.06, L2 MLP +0.05, embedding +0.04, L1 MLP +0.03 |
| mod 10 | 0.60 | L3 MLP +0.14, L0 MLP +0.12, L4 MLP +0.10, L2 MLP +0.05, L6 MLP +0.05, embedding +0.04, L5 MLP +0.04, L7 MLP +0.03, L1 MLP +0.03 |
| mod 5 | 0.57 | L3 MLP +0.12, L0 MLP +0.10, L4 MLP +0.09, L2 MLP +0.07, L5 MLP +0.07, embedding +0.05, L6 MLP +0.04 |
| mod 4 | 0.40 | L4 MLP +0.06, L0 MLP +0.06, L2 MLP +0.05, embedding +0.05, L3 MLP +0.05, L5 MLP +0.04, L6 MLP +0.04, L7 MLP +0.04 |
| mod 2 | 0.61 | L4 MLP +0.18, L0 MLP +0.13, L3 MLP +0.12, embedding +0.05, L2 MLP +0.05, L5 MLP +0.03, L6 MLP +0.02 |

<details><summary>period 100: 3 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L7 down c5 | +0.046 | 28% | **a** (R2 1.00): a in {3..23, 25..31} |
| L6 down c25 | +0.039 | 30% | **a** (R2 1.00): a in {2..31} |
| L5 down c27 | +0.032 | 30% | **a** (R2 1.00): a in {2..31} |

</details>

<details><summary>period 50: 4 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L5 down c27 | +0.033 | 30% | **a** (R2 1.00): a in {2..31} |
| L7 down c5 | +0.029 | 28% | **a** (R2 1.00): a in {3..23, 25..31} |
| L6 down c25 | +0.025 | 30% | **a** (R2 1.00): a in {2..31} |
| L5 down c124 | +0.020 | 12% | **a** (R2 1.00): a in {1..12} |

</details>

<details><summary>period 20: 1 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L5 down c124 | +0.020 | 12% | **a** (R2 1.00): a in {1..12} |

</details>

<details><summary>period 10: 4 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L4 down c66 | +0.067 | 39% | **a%10** (R2 0.96): a mod 10 in {6..9} |
| L3 down c63 | +0.043 | 30% | **a%10** (R2 1.00): a mod 10 in {5..7} |
| L3 down c136 | +0.039 | 37% | **a%10** (R2 0.91): a mod 10 in {2, 7..9} |
| L3 down c54 | +0.023 | 20% | **a%10** (R2 1.00): a mod 10 in {1, 9} |

</details>

<details><summary>period 5: 7 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L4 down c43 | +0.069 | 21% | **a%5** (R2 0.94): a mod 5 in {0} |
| L5 down c28 | +0.059 | 18% | **a** (R2 0.99): a in {10, 15, 20, 25, 30, 35, 40, 45, 50, 52, 55, 60, 70, 75, 80, 90, 99..100} [coarser: a mod 50 in {0, 5, 10, 20, 25, 30, 35, 40}, R2 0.80] |
| L2 down c473 | +0.041 | 20% | **a%5** (R2 1.00): a mod 5 in {0} |
| L3 down c615 | +0.039 | 19% | **a%5** (R2 0.94): a mod 5 in {0} |
| L3 down c68 | +0.031 | 38% | **a%50** (R2 0.91): a mod 50 in {4, 9, 13..14, 17, 19, 23..24, 27, 29, 33..34, 37, 39, 43..44, 49} [coarser: a mod 10 in {3..4, 7, 9}, R2 0.86] |
| L6 down c244 | +0.025 | 4% | **a** (R2 1.00): a in {40, 50, 60, 100} |
| L0 down c162 | +0.024 | 19% | **a%5** (R2 0.94): a mod 5 in {0} |

</details>

<details><summary>period 4: 3 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L3 down c153 | +0.023 | 6% | **a** (R2 1.00): a in {12, 24, 36, 48, 72, 96} |
| L2 down c118 | +0.021 | 19% | **a** (R2 1.00): a in {16, 24, 32, 36, 44, 46, 48..49, 52, 54, 56, 64, 72, 74, 76, 84, 86, 88, 96} |
| L5 down c332 | +0.021 | 4% | **a** (R2 1.00): a in {24, 36, 48, 72} |

</details>

<details><summary>period 2: 2 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L4 down c135 | +0.167 | 46% | **a%10** (R2 0.90): a mod 10 in {0, 2, 4, 6, 8} [coarser: a mod 2 in {0}, R2 0.85] |
| L3 down c54 | +0.052 | 20% | **a%10** (R2 1.00): a mod 10 in {1, 9} |

</details>

</details>

<details><summary>`mlp_in.12` at `a` (addition prompts): who writes the a code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.65 | L11 MLP +0.14, L4 MLP +0.07, L0 MLP +0.06, L3 MLP +0.06, L5 MLP +0.06, L7 MLP +0.05, L6 MLP +0.05, L2 MLP +0.05, embedding +0.02, L1 MLP +0.02 |
| mod 50 | 0.66 | L11 MLP +0.16, L4 MLP +0.08, L0 MLP +0.07, L3 MLP +0.07, L5 MLP +0.05, L2 MLP +0.05, L7 MLP +0.05, L6 MLP +0.04, embedding +0.02, L10 MLP +0.02, L1 MLP +0.02 |
| mod 25 | 0.59 | L11 MLP +0.12, L0 MLP +0.07, L4 MLP +0.07, L3 MLP +0.06, L2 MLP +0.04, L5 MLP +0.04, L6 MLP +0.04, L7 MLP +0.04, L10 MLP +0.03, embedding +0.03 |
| mod 20 | 0.62 | L11 MLP +0.17, L0 MLP +0.08, L3 MLP +0.08, L4 MLP +0.06, L2 MLP +0.04, L5 MLP +0.03, L7 MLP +0.03, L1 MLP +0.03, L6 MLP +0.03, embedding +0.02 |
| mod 10 | 0.71 | L11 MLP +0.28, L3 MLP +0.10, L0 MLP +0.10, L4 MLP +0.08, L2 MLP +0.03, embedding +0.03, L1 MLP +0.02 |
| mod 5 | 0.63 | L11 MLP +0.20, L3 MLP +0.09, L0 MLP +0.09, L4 MLP +0.06, L2 MLP +0.05, L5 MLP +0.04, embedding +0.03 |
| mod 4 | 0.36 | L4 MLP +0.06, L0 MLP +0.05, L2 MLP +0.04, L3 MLP +0.04, L5 MLP +0.04, L7 MLP +0.03, embedding +0.03, L6 MLP +0.03, L10 MLP +0.02 |
| mod 2 | 0.67 | L4 MLP +0.15, L11 MLP +0.14, L9 MLP +0.11, L0 MLP +0.09, L3 MLP +0.08, embedding +0.03, L2 MLP +0.02 |

<details><summary>period 100: 3 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L7 down c5 | +0.025 | 28% | **a** (R2 1.00): a in {3..23, 25..31} |
| L6 down c25 | +0.022 | 30% | **a** (R2 1.00): a in {2..31} |
| L11 down c10 | +0.022 | 14% | **a** (R2 1.00): a in {58..71} |

</details>

<details><summary>period 50: 2 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L11 down c31 | +0.022 | 16% | **a** (R2 1.00): a in {8..23} |
| L11 down c10 | +0.020 | 14% | **a** (R2 1.00): a in {58..71} |

</details>

<details><summary>period 20: 3 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L11 down c18 | +0.030 | 13% | **a** (R2 1.00): a in {50..62} |
| L11 down c22 | +0.025 | 14% | **a** (R2 1.00): a in {87..100} |
| L11 down c30 | +0.023 | 12% | **a** (R2 1.00): a in {68..79} |

</details>

<details><summary>period 10: 10 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L11 down c20 | +0.081 | 27% | **a%50** (R2 0.92): a mod 50 in {2, 11..12, 21..23, 31..33, 41..43} [coarser: a mod 10 in {1..3}, R2 0.87] |
| L4 down c66 | +0.059 | 39% | **a%10** (R2 0.96): a mod 10 in {6..9} |
| L11 down c27 | +0.048 | 11% | **a%10** (R2 0.91): a mod 10 in {5} |
| L3 down c63 | +0.037 | 30% | **a%10** (R2 1.00): a mod 10 in {5..7} |
| L11 down c26 | +0.036 | 13% | **a** (R2 1.00): a in {4, 14, 23..24, 34, 42..44, 54, 64, 74, 84, 94} [coarser: a mod 20 in {4, 14}, R2 0.82] |
| L3 down c136 | +0.032 | 37% | **a%10** (R2 0.91): a mod 10 in {2, 7..9} |
| L11 down c42 | +0.029 | 10% | **a%10** (R2 1.00): a mod 10 in {0} |
| L11 down c36 | +0.024 | 10% | **a%10** (R2 1.00): a mod 10 in {7} |
| L11 down c40 | +0.022 | 9% | **a%20** (R2 0.90): a mod 20 in {8, 18} [coarser: a mod 10 in {8}, R2 0.89] |
| L11 down c63 | +0.022 | 10% | **a%10** (R2 1.00): a mod 10 in {6} |

</details>

<details><summary>period 5: 9 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L4 down c43 | +0.050 | 21% | **a%5** (R2 0.94): a mod 5 in {0} |
| L11 down c27 | +0.040 | 11% | **a%10** (R2 0.91): a mod 10 in {5} |
| L11 down c20 | +0.039 | 27% | **a%50** (R2 0.92): a mod 50 in {2, 11..12, 21..23, 31..33, 41..43} [coarser: a mod 10 in {1..3}, R2 0.87] |
| L11 down c42 | +0.037 | 10% | **a%10** (R2 1.00): a mod 10 in {0} |
| L5 down c28 | +0.033 | 18% | **a** (R2 0.99): a in {10, 15, 20, 25, 30, 35, 40, 45, 50, 52, 55, 60, 70, 75, 80, 90, 99..100} [coarser: a mod 50 in {0, 5, 10, 20, 25, 30, 35, 40}, R2 0.80] |
| L2 down c473 | +0.029 | 20% | **a%5** (R2 1.00): a mod 5 in {0} |
| L3 down c68 | +0.028 | 38% | **a%50** (R2 0.91): a mod 50 in {4, 9, 13..14, 17, 19, 23..24, 27, 29, 33..34, 37, 39, 43..44, 49} [coarser: a mod 10 in {3..4, 7, 9}, R2 0.86] |
| L3 down c615 | +0.027 | 19% | **a%5** (R2 0.94): a mod 5 in {0} |
| L11 down c36 | +0.025 | 10% | **a%10** (R2 1.00): a mod 10 in {7} |

</details>

<details><summary>period 2: 6 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L4 down c135 | +0.140 | 46% | **a%10** (R2 0.90): a mod 10 in {0, 2, 4, 6, 8} [coarser: a mod 2 in {0}, R2 0.85] |
| L9 down c64 | +0.099 | 36% | **a%50** (R2 0.91): a mod 50 in {9, 11, 13, 17, 19, 21, 23, 29, 31, 33, 37, 39, 41, 43, 47, 49} [coarser: a mod 10 in {1, 3, 7, 9}, R2 0.85] |
| L3 down c54 | +0.035 | 20% | **a%10** (R2 1.00): a mod 10 in {1, 9} |
| L11 down c26 | +0.030 | 13% | **a** (R2 1.00): a in {4, 14, 23..24, 34, 42..44, 54, 64, 74, 84, 94} [coarser: a mod 20 in {4, 14}, R2 0.82] |
| L11 down c42 | +0.028 | 10% | **a%10** (R2 1.00): a mod 10 in {0} |
| L11 down c63 | +0.025 | 10% | **a%10** (R2 1.00): a mod 10 in {6} |

</details>

</details>

<details><summary>`attn_in.16` at `a` (addition prompts): who writes the a code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.80 | L14 MLP +0.15, L15 MLP +0.14, L13 MLP +0.11, L12 MLP +0.07, L11 MLP +0.06, L0 MLP +0.05, L4 MLP +0.05, L3 MLP +0.04, L2 MLP +0.04, L5 MLP +0.03, L7 MLP +0.02, L6 MLP +0.02 |
| mod 50 | 0.81 | L15 MLP +0.13, L13 MLP +0.13, L14 MLP +0.12, L12 MLP +0.07, L11 MLP +0.07, L0 MLP +0.06, L4 MLP +0.05, L3 MLP +0.04, L2 MLP +0.03, L5 MLP +0.02 |
| mod 25 | 0.77 | L15 MLP +0.17, L14 MLP +0.13, L13 MLP +0.10, L12 MLP +0.06, L11 MLP +0.05, L0 MLP +0.05, L4 MLP +0.04, L3 MLP +0.04, L2 MLP +0.03 |
| mod 20 | 0.83 | L15 MLP +0.15, L12 MLP +0.14, L14 MLP +0.13, L13 MLP +0.12, L11 MLP +0.07, L0 MLP +0.06, L3 MLP +0.05, L4 MLP +0.03, L2 MLP +0.02 |
| mod 10 | 0.92 | L13 MLP +0.21, L14 MLP +0.16, L12 MLP +0.15, L11 MLP +0.09, L0 MLP +0.06, L3 MLP +0.06, L15 MLP +0.06, L4 MLP +0.05 |
| mod 5 | 0.89 | L13 MLP +0.20, L14 MLP +0.20, L12 MLP +0.10, L15 MLP +0.10, L11 MLP +0.07, L0 MLP +0.06, L3 MLP +0.05, L4 MLP +0.03, L2 MLP +0.02 |
| mod 4 | 0.66 | L15 MLP +0.26, L14 MLP +0.22, L12 MLP +0.03, L0 MLP +0.03, L13 MLP +0.02, L4 MLP +0.02 |
| mod 2 | 0.84 | L14 MLP +0.18, L13 MLP +0.13, L4 MLP +0.10, L0 MLP +0.07, L12 MLP +0.07, L9 MLP +0.06, L15 MLP +0.06, L11 MLP +0.06, L3 MLP +0.05, embedding +0.02 |

<details><summary>period 100: 2 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L14 down c8 | +0.050 | 46% | **a//10** (R2 0.92): (tens) a in {1..40, 42, 44..48} |
| L13 down c10 | +0.022 | 24% | **a** (R2 1.00): a in {75..98} |

</details>

<details><summary>period 50: 1 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L13 down c10 | +0.025 | 24% | **a** (R2 1.00): a in {75..98} |

</details>

<details><summary>period 20: 2 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L12 down c9 | +0.063 | 25% | **a%20** (R2 1.00): a mod 20 in {10..14} |
| L12 down c73 | +0.028 | 21% | **a** (R2 1.00): a in {15..17, 35..37, 39, 55..59, 75..77, 79, 95..99} [coarser: a mod 20 in {15..17, 19}, R2 0.88] |

</details>

<details><summary>period 10: 17 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L13 down c11 | +0.039 | 10% | **a%10** (R2 1.00): a mod 10 in {8} |
| L13 down c9 | +0.038 | 10% | **a%10** (R2 1.00): a mod 10 in {7} |
| L4 down c66 | +0.036 | 39% | **a%10** (R2 0.96): a mod 10 in {6..9} |
| L12 down c31 | +0.033 | 30% | **a%10** (R2 0.98): a mod 10 in {5..7} |
| L14 down c14 | +0.032 | 10% | **a%10** (R2 1.00): a mod 10 in {4} |
| L13 down c38 | +0.031 | 20% | **a%10** (R2 1.00): a mod 10 in {0, 9} |
| L12 down c7 | +0.031 | 10% | **a%10** (R2 1.00): a mod 10 in {9} |
| L11 down c20 | +0.031 | 27% | **a%50** (R2 0.92): a mod 50 in {2, 11..12, 21..23, 31..33, 41..43} [coarser: a mod 10 in {1..3}, R2 0.87] |
| L14 down c15 | +0.030 | 10% | **a%10** (R2 1.00): a mod 10 in {2} |
| L13 down c13 | +0.029 | 10% | **a%10** (R2 1.00): a mod 10 in {5} |
| L13 down c15 | +0.027 | 10% | **a%10** (R2 1.00): a mod 10 in {3} |
| L14 down c21 | +0.025 | 10% | **a%10** (R2 1.00): a mod 10 in {9} |
| L12 down c19 | +0.023 | 10% | **a%10** (R2 1.00): a mod 10 in {0} |
| L3 down c63 | +0.022 | 30% | **a%10** (R2 1.00): a mod 10 in {5..7} |
| L12 down c32 | +0.021 | 23% | **a** (R2 1.00): a in {2..4, 13, 22..24, 33, 42..44, 53, 62..64, 72..73, 82..84, 92..94} [coarser: a mod 20 in {2..4, 13}, R2 0.89] |
| L12 down c15 | +0.021 | 10% | **a%10** (R2 1.00): a mod 10 in {1} |
| L3 down c136 | +0.021 | 37% | **a%10** (R2 0.91): a mod 10 in {2, 7..9} |

</details>

<details><summary>period 5: 14 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L15 down c70 | +0.061 | 20% | **a%5** (R2 1.00): a mod 5 in {0} |
| L14 down c18 | +0.046 | 20% | **a%5** (R2 1.00): a mod 5 in {0} |
| L13 down c13 | +0.042 | 10% | **a%10** (R2 1.00): a mod 10 in {5} |
| L13 down c9 | +0.033 | 10% | **a%10** (R2 1.00): a mod 10 in {7} |
| L14 down c14 | +0.033 | 10% | **a%10** (R2 1.00): a mod 10 in {4} |
| L13 down c15 | +0.033 | 10% | **a%10** (R2 1.00): a mod 10 in {3} |
| L13 down c11 | +0.032 | 10% | **a%10** (R2 1.00): a mod 10 in {8} |
| L12 down c7 | +0.028 | 10% | **a%10** (R2 1.00): a mod 10 in {9} |
| L14 down c15 | +0.027 | 10% | **a%10** (R2 1.00): a mod 10 in {2} |
| L12 down c19 | +0.027 | 10% | **a%10** (R2 1.00): a mod 10 in {0} |
| L14 down c21 | +0.027 | 10% | **a%10** (R2 1.00): a mod 10 in {9} |
| L12 down c15 | +0.024 | 10% | **a%10** (R2 1.00): a mod 10 in {1} |
| L4 down c43 | +0.023 | 21% | **a%5** (R2 0.94): a mod 5 in {0} |
| L14 down c33 | +0.020 | 10% | **a%10** (R2 1.00): a mod 10 in {6} |

</details>

<details><summary>period 4: 2 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L15 down c42 | +0.203 | 15% | **a** (R2 1.00): a in {12, 16, 24, 32, 36, 40, 44, 48, 56, 64, 72, 76, 80, 84, 96} |
| L14 down c42 | +0.140 | 15% | **a** (R2 0.99): a in {12, 18, 22, 32, 38, 42..43, 48, 58, 62, 72, 78, 82..83, 98} |

</details>

<details><summary>period 2: 14 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L4 down c135 | +0.102 | 46% | **a%10** (R2 0.90): a mod 10 in {0, 2, 4, 6, 8} [coarser: a mod 2 in {0}, R2 0.85] |
| L9 down c64 | +0.062 | 36% | **a%50** (R2 0.91): a mod 50 in {9, 11, 13, 17, 19, 21, 23, 29, 31, 33, 37, 39, 41, 43, 47, 49} [coarser: a mod 10 in {1, 3, 7, 9}, R2 0.85] |
| L14 down c14 | +0.039 | 10% | **a%10** (R2 1.00): a mod 10 in {4} |
| L14 down c15 | +0.035 | 10% | **a%10** (R2 1.00): a mod 10 in {2} |
| L15 down c42 | +0.034 | 15% | **a** (R2 1.00): a in {12, 16, 24, 32, 36, 40, 44, 48, 56, 64, 72, 76, 80, 84, 96} |
| L14 down c21 | +0.031 | 10% | **a%10** (R2 1.00): a mod 10 in {9} |
| L13 down c15 | +0.031 | 10% | **a%10** (R2 1.00): a mod 10 in {3} |
| L14 down c33 | +0.028 | 10% | **a%10** (R2 1.00): a mod 10 in {6} |
| L13 down c9 | +0.026 | 10% | **a%10** (R2 1.00): a mod 10 in {7} |
| L3 down c54 | +0.022 | 20% | **a%10** (R2 1.00): a mod 10 in {1, 9} |
| L13 down c13 | +0.022 | 10% | **a%10** (R2 1.00): a mod 10 in {5} |
| L12 down c19 | +0.021 | 10% | **a%10** (R2 1.00): a mod 10 in {0} |
| L13 down c11 | +0.020 | 10% | **a%10** (R2 1.00): a mod 10 in {8} |
| L12 down c7 | +0.020 | 10% | **a%10** (R2 1.00): a mod 10 in {9} |

</details>

</details>

<details><summary>`attn_in.18` at `a` (addition prompts): who writes the a code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.84 | L16 MLP +0.18, L17 MLP +0.16, L14 MLP +0.09, L15 MLP +0.08, L13 MLP +0.07, L12 MLP +0.04, L11 MLP +0.04, L0 MLP +0.03, L4 MLP +0.03, L3 MLP +0.03, L2 MLP +0.02 |
| mod 50 | 0.83 | L17 MLP +0.16, L16 MLP +0.13, L15 MLP +0.08, L14 MLP +0.08, L13 MLP +0.08, L12 MLP +0.05, L11 MLP +0.05, L0 MLP +0.04, L4 MLP +0.03, L3 MLP +0.03, L2 MLP +0.02 |
| mod 25 | 0.80 | L17 MLP +0.16, L16 MLP +0.14, L15 MLP +0.11, L14 MLP +0.09, L13 MLP +0.07, L12 MLP +0.04, L11 MLP +0.04, L0 MLP +0.03, L4 MLP +0.03, L3 MLP +0.03 |
| mod 20 | 0.82 | L17 MLP +0.14, L16 MLP +0.11, L15 MLP +0.10, L12 MLP +0.09, L14 MLP +0.09, L13 MLP +0.08, L11 MLP +0.05, L0 MLP +0.04, L3 MLP +0.03, L4 MLP +0.02 |
| mod 10 | 0.90 | L13 MLP +0.17, L17 MLP +0.12, L14 MLP +0.12, L12 MLP +0.11, L16 MLP +0.09, L11 MLP +0.07, L0 MLP +0.05, L3 MLP +0.04, L4 MLP +0.03, L15 MLP +0.03 |
| mod 5 | 0.90 | L17 MLP +0.19, L13 MLP +0.14, L14 MLP +0.13, L16 MLP +0.11, L12 MLP +0.07, L15 MLP +0.06, L11 MLP +0.05, L0 MLP +0.04, L3 MLP +0.03 |
| mod 4 | 0.72 | L15 MLP +0.17, L14 MLP +0.15, L17 MLP +0.14, L16 MLP +0.14, L12 MLP +0.02, L0 MLP +0.02 |
| mod 2 | 0.85 | L14 MLP +0.13, L17 MLP +0.10, L13 MLP +0.10, L16 MLP +0.10, L4 MLP +0.08, L0 MLP +0.05, L12 MLP +0.05, L9 MLP +0.05, L15 MLP +0.04, L11 MLP +0.04, L3 MLP +0.04 |

<details><summary>period 100: 5 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L14 down c8 | +0.034 | 46% | **a//10** (R2 0.92): (tens) a in {1..40, 42, 44..48} |
| L16 down c5 | +0.028 | 69% | **a//10** (R2 0.93): (tens) a in {32..100} |
| L16 down c9 | +0.027 | 20% | **a//10** (R2 0.94): (tens) a in {80, 82..100} |
| L17 down c58 | +0.025 | 21% | **a//10** (R2 0.90): (tens) a in {1..21} |
| L16 down c3 | +0.022 | 27% | **a** (R2 1.00): a in {32..58} |

</details>

<details><summary>period 50: 2 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L17 down c58 | +0.022 | 21% | **a//10** (R2 0.90): (tens) a in {1..21} |
| L16 down c9 | +0.021 | 20% | **a//10** (R2 0.94): (tens) a in {80, 82..100} |

</details>

<details><summary>period 20: 1 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L12 down c9 | +0.042 | 25% | **a%20** (R2 1.00): a mod 20 in {10..14} |

</details>

<details><summary>period 10: 12 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L13 down c11 | +0.034 | 10% | **a%10** (R2 1.00): a mod 10 in {8} |
| L13 down c9 | +0.034 | 10% | **a%10** (R2 1.00): a mod 10 in {7} |
| L16 down c34 | +0.027 | 10% | **a%10** (R2 1.00): a mod 10 in {8} |
| L4 down c66 | +0.027 | 39% | **a%10** (R2 0.96): a mod 10 in {6..9} |
| L12 down c31 | +0.025 | 30% | **a%10** (R2 0.98): a mod 10 in {5..7} |
| L14 down c14 | +0.024 | 10% | **a%10** (R2 1.00): a mod 10 in {4} |
| L12 down c7 | +0.023 | 10% | **a%10** (R2 1.00): a mod 10 in {9} |
| L13 down c38 | +0.023 | 20% | **a%10** (R2 1.00): a mod 10 in {0, 9} |
| L11 down c20 | +0.022 | 27% | **a%50** (R2 0.92): a mod 50 in {2, 11..12, 21..23, 31..33, 41..43} [coarser: a mod 10 in {1..3}, R2 0.87] |
| L13 down c13 | +0.021 | 10% | **a%10** (R2 1.00): a mod 10 in {5} |
| L14 down c15 | +0.021 | 10% | **a%10** (R2 1.00): a mod 10 in {2} |
| L16 down c74 | +0.020 | 10% | **a%10** (R2 1.00): a mod 10 in {7} |

</details>

<details><summary>period 5: 10 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L17 down c8 | +0.119 | 20% | **a%5** (R2 1.00): a mod 5 in {0} |
| L15 down c70 | +0.039 | 20% | **a%5** (R2 1.00): a mod 5 in {0} |
| L16 down c50 | +0.036 | 13% | **a** (R2 0.99): a in {16, 21, 26, 31, 41, 46, 51, 61, 71, 76, 81, 86, 91} |
| L14 down c18 | +0.032 | 20% | **a%5** (R2 1.00): a mod 5 in {0} |
| L13 down c13 | +0.029 | 10% | **a%10** (R2 1.00): a mod 10 in {5} |
| L13 down c9 | +0.025 | 10% | **a%10** (R2 1.00): a mod 10 in {7} |
| L13 down c11 | +0.022 | 10% | **a%10** (R2 1.00): a mod 10 in {8} |
| L16 down c34 | +0.022 | 10% | **a%10** (R2 1.00): a mod 10 in {8} |
| L14 down c14 | +0.020 | 10% | **a%10** (R2 1.00): a mod 10 in {4} |
| L16 down c120 | +0.020 | 12% | **a%50** (R2 0.98): a mod 50 in {0, 10, 20, 25, 30, 40} [coarser: a mod 10 in {0}, R2 0.84] |

</details>

<details><summary>period 4: 4 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L15 down c42 | +0.131 | 15% | **a** (R2 1.00): a in {12, 16, 24, 32, 36, 40, 44, 48, 56, 64, 72, 76, 80, 84, 96} |
| L14 down c42 | +0.098 | 15% | **a** (R2 0.99): a in {12, 18, 22, 32, 38, 42..43, 48, 58, 62, 72, 78, 82..83, 98} |
| L16 down c30 | +0.072 | 7% | **a** (R2 0.99): a in {24, 32, 36, 48, 64, 72, 96} |
| L17 down c66 | +0.024 | 6% | **a** (R2 1.00): a in {12, 24, 36, 42, 48, 72} |

</details>

<details><summary>period 2: 11 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L4 down c135 | +0.080 | 46% | **a%10** (R2 0.90): a mod 10 in {0, 2, 4, 6, 8} [coarser: a mod 2 in {0}, R2 0.85] |
| L9 down c64 | +0.047 | 36% | **a%50** (R2 0.91): a mod 50 in {9, 11, 13, 17, 19, 21, 23, 29, 31, 33, 37, 39, 41, 43, 47, 49} [coarser: a mod 10 in {1, 3, 7, 9}, R2 0.85] |
| L16 down c34 | +0.032 | 10% | **a%10** (R2 1.00): a mod 10 in {8} |
| L14 down c14 | +0.027 | 10% | **a%10** (R2 1.00): a mod 10 in {4} |
| L15 down c42 | +0.025 | 15% | **a** (R2 1.00): a in {12, 16, 24, 32, 36, 40, 44, 48, 56, 64, 72, 76, 80, 84, 96} |
| L14 down c21 | +0.025 | 10% | **a%10** (R2 1.00): a mod 10 in {9} |
| L14 down c15 | +0.024 | 10% | **a%10** (R2 1.00): a mod 10 in {2} |
| L17 down c54 | +0.023 | 10% | **a** (R2 1.00): a in {6, 26, 36, 46, 56, 60, 66, 76, 86, 96} [coarser: a mod 10 in {6}, R2 0.80] |
| L13 down c9 | +0.023 | 10% | **a%10** (R2 1.00): a mod 10 in {7} |
| L13 down c11 | +0.022 | 10% | **a%10** (R2 1.00): a mod 10 in {8} |
| L14 down c33 | +0.022 | 10% | **a%10** (R2 1.00): a mod 10 in {6} |

</details>

</details>

<details><summary>`attn_in.20` at `a` (addition prompts): who writes the a code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.85 | L16 MLP +0.13, L19 MLP +0.12, L17 MLP +0.11, L18 MLP +0.09, L14 MLP +0.07, L15 MLP +0.06, L13 MLP +0.05, L12 MLP +0.03, L0 MLP +0.03, L11 MLP +0.03, L4 MLP +0.03, L3 MLP +0.02, L2 MLP +0.02 |
| mod 50 | 0.82 | L17 MLP +0.11, L16 MLP +0.10, L19 MLP +0.09, L18 MLP +0.08, L15 MLP +0.07, L14 MLP +0.06, L13 MLP +0.06, L12 MLP +0.04, L11 MLP +0.04, L0 MLP +0.03, L4 MLP +0.03, L3 MLP +0.03 |
| mod 25 | 0.80 | L17 MLP +0.12, L16 MLP +0.10, L15 MLP +0.09, L19 MLP +0.08, L18 MLP +0.08, L14 MLP +0.07, L13 MLP +0.05, L12 MLP +0.03, L0 MLP +0.03, L11 MLP +0.03, L3 MLP +0.02, L4 MLP +0.02 |
| mod 20 | 0.81 | L17 MLP +0.10, L15 MLP +0.08, L12 MLP +0.08, L16 MLP +0.08, L14 MLP +0.07, L19 MLP +0.07, L13 MLP +0.06, L18 MLP +0.06, L11 MLP +0.04, L0 MLP +0.04, L3 MLP +0.03, L4 MLP +0.02 |
| mod 10 | 0.89 | L13 MLP +0.15, L12 MLP +0.10, L14 MLP +0.10, L17 MLP +0.10, L16 MLP +0.07, L19 MLP +0.07, L11 MLP +0.06, L18 MLP +0.05, L0 MLP +0.04, L3 MLP +0.04, L4 MLP +0.03, L15 MLP +0.03 |
| mod 5 | 0.88 | L17 MLP +0.14, L13 MLP +0.12, L14 MLP +0.11, L16 MLP +0.09, L19 MLP +0.08, L12 MLP +0.06, L18 MLP +0.06, L15 MLP +0.04, L11 MLP +0.04, L0 MLP +0.04, L3 MLP +0.03 |
| mod 4 | 0.70 | L15 MLP +0.13, L14 MLP +0.12, L17 MLP +0.11, L16 MLP +0.10, L19 MLP +0.08, L18 MLP +0.04 |
| mod 2 | 0.80 | L14 MLP +0.11, L13 MLP +0.08, L16 MLP +0.08, L17 MLP +0.08, L19 MLP +0.07, L4 MLP +0.07, L0 MLP +0.05, L12 MLP +0.05, L9 MLP +0.04, L18 MLP +0.04, L3 MLP +0.04, L11 MLP +0.03, L15 MLP +0.03 |

<details><summary>period 100: 5 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L19 down c10 | +0.031 | 37% | **a** (R2 1.00): a in {64..100} |
| L19 down c13 | +0.030 | 67% | **a** (R2 1.00): a in {32..49, 51..99} |
| L18 down c13 | +0.028 | 56% | **a** (R2 1.00): a in {2..52, 55, 60, 75, 90, 100} |
| L14 down c8 | +0.027 | 46% | **a//10** (R2 0.92): (tens) a in {1..40, 42, 44..48} |
| L16 down c5 | +0.022 | 69% | **a//10** (R2 0.93): (tens) a in {32..100} |

</details>

<details><summary>period 20: 1 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L12 down c9 | +0.035 | 25% | **a%20** (R2 1.00): a mod 20 in {10..14} |

</details>

<details><summary>period 10: 7 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L13 down c9 | +0.030 | 10% | **a%10** (R2 1.00): a mod 10 in {7} |
| L13 down c11 | +0.028 | 10% | **a%10** (R2 1.00): a mod 10 in {8} |
| L19 down c52 | +0.027 | 10% | **a** (R2 1.00): a in {1..2, 21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 10 in {1}, R2 0.80] |
| L4 down c66 | +0.024 | 39% | **a%10** (R2 0.96): a mod 10 in {6..9} |
| L12 down c31 | +0.023 | 30% | **a%10** (R2 0.98): a mod 10 in {5..7} |
| L12 down c7 | +0.021 | 10% | **a%10** (R2 1.00): a mod 10 in {9} |
| L16 down c34 | +0.021 | 10% | **a%10** (R2 1.00): a mod 10 in {8} |

</details>

<details><summary>period 5: 6 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L17 down c8 | +0.085 | 20% | **a%5** (R2 1.00): a mod 5 in {0} |
| L19 down c47 | +0.041 | 19% | **a%50** (R2 0.90): a mod 50 in {0, 15, 20, 25, 30, 35, 40, 45} [coarser: a mod 5 in {0}, R2 0.82] |
| L15 down c70 | +0.030 | 20% | **a%5** (R2 1.00): a mod 5 in {0} |
| L16 down c50 | +0.028 | 13% | **a** (R2 0.99): a in {16, 21, 26, 31, 41, 46, 51, 61, 71, 76, 81, 86, 91} |
| L14 down c18 | +0.026 | 20% | **a%5** (R2 1.00): a mod 5 in {0} |
| L13 down c13 | +0.024 | 10% | **a%10** (R2 1.00): a mod 10 in {5} |

</details>

<details><summary>period 4: 4 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L15 down c42 | +0.103 | 15% | **a** (R2 1.00): a in {12, 16, 24, 32, 36, 40, 44, 48, 56, 64, 72, 76, 80, 84, 96} |
| L14 down c42 | +0.078 | 15% | **a** (R2 0.99): a in {12, 18, 22, 32, 38, 42..43, 48, 58, 62, 72, 78, 82..83, 98} |
| L16 down c30 | +0.051 | 7% | **a** (R2 0.99): a in {24, 32, 36, 48, 64, 72, 96} |
| L19 down c206 | +0.040 | 6% | **a** (R2 1.00): a in {8, 16, 32, 40, 64, 80} |

</details>

<details><summary>period 2: 7 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L4 down c135 | +0.069 | 46% | **a%10** (R2 0.90): a mod 10 in {0, 2, 4, 6, 8} [coarser: a mod 2 in {0}, R2 0.85] |
| L9 down c64 | +0.040 | 36% | **a%50** (R2 0.91): a mod 50 in {9, 11, 13, 17, 19, 21, 23, 29, 31, 33, 37, 39, 41, 43, 47, 49} [coarser: a mod 10 in {1, 3, 7, 9}, R2 0.85] |
| L16 down c34 | +0.025 | 10% | **a%10** (R2 1.00): a mod 10 in {8} |
| L14 down c21 | +0.022 | 10% | **a%10** (R2 1.00): a mod 10 in {9} |
| L14 down c14 | +0.022 | 10% | **a%10** (R2 1.00): a mod 10 in {4} |
| L14 down c15 | +0.020 | 10% | **a%10** (R2 1.00): a mod 10 in {2} |
| L13 down c9 | +0.020 | 10% | **a%10** (R2 1.00): a mod 10 in {7} |

</details>

</details>

## The b code at the `b` token

<details><summary>`attn_in.2` at `b` (addition prompts): who writes the b code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.81 | L0 MLP +0.38, embedding +0.24, L1 MLP +0.19 |
| mod 50 | 0.80 | L0 MLP +0.39, embedding +0.24, L1 MLP +0.17 |
| mod 25 | 0.73 | L0 MLP +0.33, embedding +0.28, L1 MLP +0.12 |
| mod 20 | 0.81 | L0 MLP +0.40, embedding +0.25, L1 MLP +0.16 |
| mod 10 | 0.88 | L0 MLP +0.47, embedding +0.24, L1 MLP +0.17 |
| mod 5 | 0.86 | L0 MLP +0.46, embedding +0.25, L1 MLP +0.15 |
| mod 4 | 0.60 | embedding +0.35, L0 MLP +0.22, L1 MLP +0.03 |
| mod 2 | 0.82 | L0 MLP +0.46, embedding +0.26, L1 MLP +0.10 |

<details><summary>period 100: 4 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c6 | +0.088 | 47% | **b//10** (R2 0.90): (tens) b in {54..100} |
| L0 down c5 | +0.056 | 76% | **b** (R2 0.99): b in {16..89, 91..92} |
| L1 down c12 | +0.055 | 70% | **tens(a,b)** (R2 0.79): a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,1,4,5,6,7,8,9}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {4,5} -> b//10 in {0,1,2,4,5,6,7,8,9,10}; a//10 in {6,7,8} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,6,7,8,9,10} |
| L0 down c23 | +0.036 | 39% | **b//10** (R2 1.00): (tens) b in {1..39} |

</details>

<details><summary>period 50: 4 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c6 | +0.039 | 47% | **b//10** (R2 0.90): (tens) b in {54..100} |
| L1 down c12 | +0.027 | 70% | **tens(a,b)** (R2 0.79): a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,1,4,5,6,7,8,9}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {4,5} -> b//10 in {0,1,2,4,5,6,7,8,9,10}; a//10 in {6,7,8} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,6,7,8,9,10} |
| L0 down c23 | +0.024 | 39% | **b//10** (R2 1.00): (tens) b in {1..39} |
| L0 down c22 | +0.021 | 19% | **b//10** (R2 0.94): (tens) b in {61..79} |

</details>

<details><summary>period 20: 2 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L1 down c62 | +0.026 | 10% | **b//10** (R2 0.92): (tens) b in {1..9} |
| L1 down c271 | +0.021 | 11% | **b//10** (R2 0.90): (tens) b in {55, 80..89} |

</details>

<details><summary>period 10: 14 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L1 down c140 | +0.048 | 20% | **b** (R2 0.96): b in {5, 15..16, 24..26, 35, 45, 55, 64..66, 75..76, 84..86, 95..96} [coarser: b mod 50 in {5, 15..16, 24..26, 35..36, 45..46}, R2 0.87] |
| L0 down c38 | +0.046 | 10% | **b%10** (R2 1.00): b mod 10 in {9} |
| L0 down c44 | +0.041 | 10% | **b%10** (R2 1.00): b mod 10 in {7} |
| L0 down c52 | +0.040 | 10% | **b%10** (R2 1.00): b mod 10 in {3} |
| L0 down c45 | +0.039 | 10% | **b%10** (R2 1.00): b mod 10 in {5} |
| L0 down c55 | +0.039 | 10% | **b%10** (R2 1.00): b mod 10 in {6} |
| L0 down c81 | +0.038 | 10% | **b%10** (R2 1.00): b mod 10 in {2} |
| L0 down c62 | +0.038 | 10% | **b%10** (R2 1.00): b mod 10 in {4} |
| L0 down c36 | +0.037 | 10% | **b%10** (R2 1.00): b mod 10 in {1} |
| L0 down c50 | +0.037 | 10% | **b%10** (R2 1.00): b mod 10 in {8} |
| L0 down c145 | +0.034 | 10% | **b%10** (R2 1.00): b mod 10 in {0} |
| L1 down c356 | +0.031 | 10% | **b%10** (R2 1.00): b mod 10 in {9} |
| L1 down c329 | +0.030 | 10% | **b%10** (R2 1.00): b mod 10 in {0} |
| L1 down c401 | +0.029 | 10% | **b%10** (R2 0.98): b mod 10 in {7} |

</details>

<details><summary>period 5: 15 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c162 | +0.051 | 20% | **b%5** (R2 1.00): b mod 5 in {0} |
| L0 down c38 | +0.047 | 10% | **b%10** (R2 1.00): b mod 10 in {9} |
| L0 down c55 | +0.042 | 10% | **b%10** (R2 1.00): b mod 10 in {6} |
| L0 down c62 | +0.042 | 10% | **b%10** (R2 1.00): b mod 10 in {4} |
| L0 down c44 | +0.041 | 10% | **b%10** (R2 1.00): b mod 10 in {7} |
| L0 down c45 | +0.039 | 10% | **b%10** (R2 1.00): b mod 10 in {5} |
| L0 down c52 | +0.038 | 10% | **b%10** (R2 1.00): b mod 10 in {3} |
| L0 down c50 | +0.037 | 10% | **b%10** (R2 1.00): b mod 10 in {8} |
| L0 down c36 | +0.037 | 10% | **b%10** (R2 1.00): b mod 10 in {1} |
| L1 down c329 | +0.036 | 10% | **b%10** (R2 1.00): b mod 10 in {0} |
| L1 down c356 | +0.035 | 10% | **b%10** (R2 1.00): b mod 10 in {9} |
| L0 down c81 | +0.035 | 10% | **b%10** (R2 1.00): b mod 10 in {2} |
| L1 down c140 | +0.032 | 20% | **b** (R2 0.96): b in {5, 15..16, 24..26, 35, 45, 55, 64..66, 75..76, 84..86, 95..96} [coarser: b mod 50 in {5, 15..16, 24..26, 35..36, 45..46}, R2 0.87] |
| L0 down c145 | +0.025 | 10% | **b%10** (R2 1.00): b mod 10 in {0} |
| L1 down c401 | +0.022 | 10% | **b%10** (R2 0.98): b mod 10 in {7} |

</details>

<details><summary>period 4: 1 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c127 | +0.096 | 9% | **b** (R2 1.00): b in {16, 24, 32, 36, 48, 56, 64, 72, 96} |

</details>

<details><summary>period 2: 13 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c52 | +0.055 | 10% | **b%10** (R2 1.00): b mod 10 in {3} |
| L0 down c44 | +0.051 | 10% | **b%10** (R2 1.00): b mod 10 in {7} |
| L0 down c38 | +0.050 | 10% | **b%10** (R2 1.00): b mod 10 in {9} |
| L0 down c50 | +0.047 | 10% | **b%10** (R2 1.00): b mod 10 in {8} |
| L0 down c55 | +0.045 | 10% | **b%10** (R2 1.00): b mod 10 in {6} |
| L0 down c45 | +0.041 | 10% | **b%10** (R2 1.00): b mod 10 in {5} |
| L0 down c62 | +0.037 | 10% | **b%10** (R2 1.00): b mod 10 in {4} |
| L0 down c36 | +0.037 | 10% | **b%10** (R2 1.00): b mod 10 in {1} |
| L1 down c401 | +0.035 | 10% | **b%10** (R2 0.98): b mod 10 in {7} |
| L0 down c81 | +0.033 | 10% | **b%10** (R2 1.00): b mod 10 in {2} |
| L1 down c356 | +0.033 | 10% | **b%10** (R2 1.00): b mod 10 in {9} |
| L0 down c145 | +0.030 | 10% | **b%10** (R2 1.00): b mod 10 in {0} |
| L1 down c329 | +0.022 | 10% | **b%10** (R2 1.00): b mod 10 in {0} |

</details>

</details>

<details><summary>`attn_in.15` at `b` (addition prompts): who writes the b code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.75 | L14 MLP +0.18, L13 MLP +0.13, L12 MLP +0.10, L11 MLP +0.05, L0 MLP +0.04, L4 MLP +0.03, L9 MLP +0.03, L10 MLP +0.03, L7 MLP +0.03, L3 MLP +0.02, L8 MLP +0.02, L5 MLP +0.02 |
| mod 50 | 0.71 | L14 MLP +0.17, L13 MLP +0.16, L12 MLP +0.10, L11 MLP +0.05, L0 MLP +0.04, L4 MLP +0.03, L3 MLP +0.03 |
| mod 25 | 0.61 | L14 MLP +0.16, L13 MLP +0.12, L12 MLP +0.08, L11 MLP +0.04, L0 MLP +0.03, L4 MLP +0.03 |
| mod 20 | 0.68 | L12 MLP +0.19, L14 MLP +0.13, L13 MLP +0.13, L0 MLP +0.04, L11 MLP +0.04, L3 MLP +0.02, L4 MLP +0.02 |
| mod 10 | 0.85 | L13 MLP +0.24, L14 MLP +0.17, L12 MLP +0.16, L11 MLP +0.08, L0 MLP +0.05, L3 MLP +0.04, L4 MLP +0.04, embedding +0.02 |
| mod 5 | 0.80 | L13 MLP +0.23, L14 MLP +0.20, L12 MLP +0.11, L11 MLP +0.06, L0 MLP +0.05, L3 MLP +0.04, embedding +0.03, L4 MLP +0.02 |
| mod 4 | 0.43 | L14 MLP +0.17, L12 MLP +0.07, L13 MLP +0.03, L4 MLP +0.03, L0 MLP +0.02, embedding +0.02 |
| mod 2 | 0.70 | L14 MLP +0.18, L13 MLP +0.12, L4 MLP +0.12, L12 MLP +0.06, L0 MLP +0.06, L11 MLP +0.04, L3 MLP +0.04, embedding +0.03 |

<details><summary>period 100: 5 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L14 down c8 | +0.030 | 37% | **b//10** (R2 0.91): (tens) b in {1..39} |
| L14 down c0 | +0.025 | 99% | always |
| L13 down c10 | +0.025 | 25% | **b** (R2 0.96): b in {75..99} |
| L14 down c34 | +0.021 | 21% | **b//10** (R2 0.91): (tens) b in {80, 82..100} |
| L12 down c68 | +0.020 | 43% | **tens(a,b)** (R2 0.82): a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {0,1,2}; a//10 in {4} -> b//10 in {0,1,2,3}; a//10 in {5} -> b//10 in {0,1,2,3,4}; a//10 in {6,7} -> b//10 in {0,1,2,3,4,5}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8} |

</details>

<details><summary>period 50: 2 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L14 down c34 | +0.035 | 21% | **b//10** (R2 0.91): (tens) b in {80, 82..100} |
| L13 down c10 | +0.027 | 25% | **b** (R2 0.96): b in {75..99} |

</details>

<details><summary>period 20: 2 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L12 down c9 | +0.089 | 25% | **b%20** (R2 1.00): b mod 20 in {10..14} |
| L12 down c73 | +0.049 | 25% | **b%20** (R2 0.94): b mod 20 in {15..19} |

</details>

<details><summary>period 10: 16 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L13 down c38 | +0.044 | 20% | **b%10** (R2 1.00): b mod 10 in {0, 9} |
| L13 down c13 | +0.042 | 10% | **b%10** (R2 0.98): b mod 10 in {5} |
| L12 down c31 | +0.037 | 30% | **b%10** (R2 0.98): b mod 10 in {5..7} |
| L13 down c9 | +0.037 | 10% | **b%10** (R2 1.00): b mod 10 in {7} |
| L14 down c15 | +0.035 | 10% | **b%10** (R2 0.99): b mod 10 in {2} |
| L13 down c11 | +0.034 | 10% | **b%10** (R2 1.00): b mod 10 in {8} |
| L14 down c14 | +0.032 | 10% | **b%10** (R2 0.99): b mod 10 in {4} |
| L4 down c66 | +0.032 | 38% | **b%10** (R2 0.94): b mod 10 in {6..9} |
| L12 down c32 | +0.031 | 33% | **b%10** (R2 0.84): b mod 10 in {2..4} |
| L13 down c15 | +0.029 | 10% | **b%10** (R2 1.00): b mod 10 in {3} |
| L11 down c20 | +0.027 | 30% | **b%10** (R2 0.98): b mod 10 in {1..3} |
| L14 down c71 | +0.026 | 10% | **b%10** (R2 1.00): b mod 10 in {1} |
| L14 down c21 | +0.023 | 10% | **b%10** (R2 1.00): b mod 10 in {9} |
| L12 down c7 | +0.023 | 10% | **b%10** (R2 1.00): b mod 10 in {9} |
| L12 down c15 | +0.021 | 10% | **b%10** (R2 0.99): b mod 10 in {1} |
| L12 down c9 | +0.020 | 25% | **b%20** (R2 1.00): b mod 20 in {10..14} |

</details>

<details><summary>period 5: 12 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L13 down c13 | +0.055 | 10% | **b%10** (R2 0.98): b mod 10 in {5} |
| L13 down c9 | +0.038 | 10% | **b%10** (R2 1.00): b mod 10 in {7} |
| L14 down c15 | +0.035 | 10% | **b%10** (R2 0.99): b mod 10 in {2} |
| L14 down c14 | +0.035 | 10% | **b%10** (R2 0.99): b mod 10 in {4} |
| L13 down c15 | +0.034 | 10% | **b%10** (R2 1.00): b mod 10 in {3} |
| L13 down c11 | +0.033 | 10% | **b%10** (R2 1.00): b mod 10 in {8} |
| L14 down c18 | +0.032 | 15% | **b%10** (R2 0.75): b mod 10 in {0} [coarser: b mod 5 in {0}, R2 0.82] |
| L14 down c21 | +0.029 | 10% | **b%10** (R2 1.00): b mod 10 in {9} |
| L13 down c38 | +0.029 | 20% | **b%10** (R2 1.00): b mod 10 in {0, 9} |
| L12 down c15 | +0.027 | 10% | **b%10** (R2 0.99): b mod 10 in {1} |
| L14 down c71 | +0.025 | 10% | **b%10** (R2 1.00): b mod 10 in {1} |
| L12 down c7 | +0.024 | 10% | **b%10** (R2 1.00): b mod 10 in {9} |

</details>

<details><summary>period 4: 2 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L14 down c42 | +0.096 | 8% | **b** (R2 0.74): b in {38, 58, 62, 78, 82..83, 98} |
| L12 down c9 | +0.034 | 25% | **b%20** (R2 1.00): b mod 20 in {10..14} |

</details>

<details><summary>period 2: 9 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L4 down c135 | +0.117 | 48% | **b%2** (R2 0.92): b mod 2 in {0} |
| L14 down c15 | +0.038 | 10% | **b%10** (R2 0.99): b mod 10 in {2} |
| L14 down c14 | +0.037 | 10% | **b%10** (R2 0.99): b mod 10 in {4} |
| L13 down c13 | +0.029 | 10% | **b%10** (R2 0.98): b mod 10 in {5} |
| L14 down c21 | +0.028 | 10% | **b%10** (R2 1.00): b mod 10 in {9} |
| L13 down c15 | +0.027 | 10% | **b%10** (R2 1.00): b mod 10 in {3} |
| L14 down c33 | +0.025 | 10% | **b%10** (R2 1.00): b mod 10 in {6} |
| L13 down c9 | +0.025 | 10% | **b%10** (R2 1.00): b mod 10 in {7} |
| L14 down c71 | +0.022 | 10% | **b%10** (R2 1.00): b mod 10 in {1} |

</details>

</details>

<details><summary>`mlp_in.0` at `b` (addition prompts): who writes the b code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 1.02 | embedding +1.02 |
| mod 50 | 1.02 | embedding +1.02 |
| mod 25 | 1.02 | embedding +1.02 |
| mod 20 | 1.02 | embedding +1.02 |
| mod 10 | 1.02 | embedding +1.02 |
| mod 5 | 1.02 | embedding +1.02 |
| mod 4 | 1.02 | embedding +1.02 |
| mod 2 | 1.02 | embedding +1.02 |

</details>

<details><summary>`mlp_in.1` at `b` (addition prompts): who writes the b code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.93 | embedding +0.52, L0 MLP +0.41 |
| mod 50 | 0.92 | embedding +0.54, L0 MLP +0.38 |
| mod 25 | 0.91 | embedding +0.60, L0 MLP +0.30 |
| mod 20 | 0.93 | embedding +0.57, L0 MLP +0.35 |
| mod 10 | 0.99 | embedding +0.53, L0 MLP +0.46 |
| mod 5 | 1.00 | embedding +0.53, L0 MLP +0.46 |
| mod 4 | 0.86 | embedding +0.64, L0 MLP +0.22 |
| mod 2 | 1.00 | embedding +0.50, L0 MLP +0.50 |

<details><summary>period 100: 3 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c6 | +0.092 | 47% | **b//10** (R2 0.90): (tens) b in {54..100} |
| L0 down c5 | +0.059 | 76% | **b** (R2 0.99): b in {16..89, 91..92} |
| L0 down c23 | +0.041 | 39% | **b//10** (R2 1.00): (tens) b in {1..39} |

</details>

<details><summary>period 50: 2 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c6 | +0.041 | 47% | **b//10** (R2 0.90): (tens) b in {54..100} |
| L0 down c23 | +0.025 | 39% | **b//10** (R2 1.00): (tens) b in {1..39} |

</details>

<details><summary>period 10: 10 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c38 | +0.048 | 10% | **b%10** (R2 1.00): b mod 10 in {9} |
| L0 down c145 | +0.040 | 10% | **b%10** (R2 1.00): b mod 10 in {0} |
| L0 down c36 | +0.040 | 10% | **b%10** (R2 1.00): b mod 10 in {1} |
| L0 down c44 | +0.037 | 10% | **b%10** (R2 1.00): b mod 10 in {7} |
| L0 down c50 | +0.036 | 10% | **b%10** (R2 1.00): b mod 10 in {8} |
| L0 down c45 | +0.036 | 10% | **b%10** (R2 1.00): b mod 10 in {5} |
| L0 down c52 | +0.035 | 10% | **b%10** (R2 1.00): b mod 10 in {3} |
| L0 down c62 | +0.034 | 10% | **b%10** (R2 1.00): b mod 10 in {4} |
| L0 down c55 | +0.032 | 10% | **b%10** (R2 1.00): b mod 10 in {6} |
| L0 down c81 | +0.032 | 10% | **b%10** (R2 1.00): b mod 10 in {2} |

</details>

<details><summary>period 5: 11 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c162 | +0.069 | 20% | **b%5** (R2 1.00): b mod 5 in {0} |
| L0 down c38 | +0.049 | 10% | **b%10** (R2 1.00): b mod 10 in {9} |
| L0 down c36 | +0.039 | 10% | **b%10** (R2 1.00): b mod 10 in {1} |
| L0 down c45 | +0.038 | 10% | **b%10** (R2 1.00): b mod 10 in {5} |
| L0 down c50 | +0.035 | 10% | **b%10** (R2 1.00): b mod 10 in {8} |
| L0 down c44 | +0.035 | 10% | **b%10** (R2 1.00): b mod 10 in {7} |
| L0 down c62 | +0.035 | 10% | **b%10** (R2 1.00): b mod 10 in {4} |
| L0 down c55 | +0.035 | 10% | **b%10** (R2 1.00): b mod 10 in {6} |
| L0 down c145 | +0.034 | 10% | **b%10** (R2 1.00): b mod 10 in {0} |
| L0 down c52 | +0.032 | 10% | **b%10** (R2 1.00): b mod 10 in {3} |
| L0 down c81 | +0.028 | 10% | **b%10** (R2 1.00): b mod 10 in {2} |

</details>

<details><summary>period 4: 1 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c127 | +0.095 | 9% | **b** (R2 1.00): b in {16, 24, 32, 36, 48, 56, 64, 72, 96} |

</details>

<details><summary>period 2: 11 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c38 | +0.058 | 10% | **b%10** (R2 1.00): b mod 10 in {9} |
| L0 down c44 | +0.054 | 10% | **b%10** (R2 1.00): b mod 10 in {7} |
| L0 down c52 | +0.051 | 10% | **b%10** (R2 1.00): b mod 10 in {3} |
| L0 down c50 | +0.046 | 10% | **b%10** (R2 1.00): b mod 10 in {8} |
| L0 down c45 | +0.045 | 10% | **b%10** (R2 1.00): b mod 10 in {5} |
| L0 down c55 | +0.043 | 10% | **b%10** (R2 1.00): b mod 10 in {6} |
| L0 down c62 | +0.041 | 10% | **b%10** (R2 1.00): b mod 10 in {4} |
| L0 down c36 | +0.041 | 10% | **b%10** (R2 1.00): b mod 10 in {1} |
| L0 down c81 | +0.037 | 10% | **b%10** (R2 1.00): b mod 10 in {2} |
| L0 down c145 | +0.035 | 10% | **b%10** (R2 1.00): b mod 10 in {0} |
| L0 down c127 | +0.021 | 9% | **b** (R2 1.00): b in {16, 24, 32, 36, 48, 56, 64, 72, 96} |

</details>

</details>

<details><summary>`mlp_in.2` at `b` (addition prompts): who writes the b code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.81 | embedding +0.32, L0 MLP +0.27, L1 MLP +0.20, L2 attn +0.02 |
| mod 50 | 0.79 | embedding +0.34, L0 MLP +0.25, L1 MLP +0.18 |
| mod 25 | 0.73 | embedding +0.39, L0 MLP +0.21, L1 MLP +0.10 |
| mod 20 | 0.77 | embedding +0.38, L0 MLP +0.24, L1 MLP +0.13 |
| mod 10 | 0.86 | embedding +0.37, L0 MLP +0.33, L1 MLP +0.15 |
| mod 5 | 0.86 | embedding +0.36, L0 MLP +0.33, L1 MLP +0.14, L2 attn +0.03 |
| mod 4 | 0.65 | embedding +0.44, L0 MLP +0.17, L1 MLP +0.03 |
| mod 2 | 0.84 | L0 MLP +0.38, embedding +0.37, L1 MLP +0.08 |

<details><summary>period 100: 4 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L1 down c12 | +0.077 | 70% | **tens(a,b)** (R2 0.79): a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,1,4,5,6,7,8,9}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {4,5} -> b//10 in {0,1,2,4,5,6,7,8,9,10}; a//10 in {6,7,8} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,6,7,8,9,10} |
| L0 down c6 | +0.062 | 47% | **b//10** (R2 0.90): (tens) b in {54..100} |
| L0 down c5 | +0.037 | 76% | **b** (R2 0.99): b in {16..89, 91..92} |
| L0 down c23 | +0.026 | 39% | **b//10** (R2 1.00): (tens) b in {1..39} |

</details>

<details><summary>period 50: 3 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L1 down c12 | +0.041 | 70% | **tens(a,b)** (R2 0.79): a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,1,4,5,6,7,8,9}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {4,5} -> b//10 in {0,1,2,4,5,6,7,8,9,10}; a//10 in {6,7,8} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,6,7,8,9,10} |
| L0 down c6 | +0.028 | 47% | **b//10** (R2 0.90): (tens) b in {54..100} |
| L1 down c62 | +0.028 | 10% | **b//10** (R2 0.92): (tens) b in {1..9} |

</details>

<details><summary>period 25: 1 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L1 down c62 | +0.026 | 10% | **b//10** (R2 0.92): (tens) b in {1..9} |

</details>

<details><summary>period 20: 1 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L1 down c62 | +0.034 | 10% | **b//10** (R2 0.92): (tens) b in {1..9} |

</details>

<details><summary>period 10: 14 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c38 | +0.036 | 10% | **b%10** (R2 1.00): b mod 10 in {9} |
| L1 down c329 | +0.032 | 10% | **b%10** (R2 1.00): b mod 10 in {0} |
| L1 down c140 | +0.030 | 20% | **b** (R2 0.96): b in {5, 15..16, 24..26, 35, 45, 55, 64..66, 75..76, 84..86, 95..96} [coarser: b mod 50 in {5, 15..16, 24..26, 35..36, 45..46}, R2 0.87] |
| L0 down c36 | +0.029 | 10% | **b%10** (R2 1.00): b mod 10 in {1} |
| L0 down c145 | +0.028 | 10% | **b%10** (R2 1.00): b mod 10 in {0} |
| L1 down c356 | +0.027 | 10% | **b%10** (R2 1.00): b mod 10 in {9} |
| L0 down c44 | +0.027 | 10% | **b%10** (R2 1.00): b mod 10 in {7} |
| L0 down c50 | +0.026 | 10% | **b%10** (R2 1.00): b mod 10 in {8} |
| L0 down c62 | +0.026 | 10% | **b%10** (R2 1.00): b mod 10 in {4} |
| L0 down c55 | +0.025 | 10% | **b%10** (R2 1.00): b mod 10 in {6} |
| L0 down c45 | +0.025 | 10% | **b%10** (R2 1.00): b mod 10 in {5} |
| L0 down c52 | +0.024 | 10% | **b%10** (R2 1.00): b mod 10 in {3} |
| L0 down c81 | +0.022 | 10% | **b%10** (R2 1.00): b mod 10 in {2} |
| L1 down c401 | +0.020 | 10% | **b%10** (R2 0.98): b mod 10 in {7} |

</details>

<details><summary>period 5: 13 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c162 | +0.047 | 20% | **b%5** (R2 1.00): b mod 5 in {0} |
| L1 down c329 | +0.046 | 10% | **b%10** (R2 1.00): b mod 10 in {0} |
| L0 down c38 | +0.035 | 10% | **b%10** (R2 1.00): b mod 10 in {9} |
| L0 down c36 | +0.029 | 10% | **b%10** (R2 1.00): b mod 10 in {1} |
| L0 down c45 | +0.027 | 10% | **b%10** (R2 1.00): b mod 10 in {5} |
| L0 down c55 | +0.027 | 10% | **b%10** (R2 1.00): b mod 10 in {6} |
| L1 down c356 | +0.026 | 10% | **b%10** (R2 1.00): b mod 10 in {9} |
| L0 down c62 | +0.026 | 10% | **b%10** (R2 1.00): b mod 10 in {4} |
| L0 down c145 | +0.026 | 10% | **b%10** (R2 1.00): b mod 10 in {0} |
| L0 down c50 | +0.025 | 10% | **b%10** (R2 1.00): b mod 10 in {8} |
| L0 down c44 | +0.024 | 10% | **b%10** (R2 1.00): b mod 10 in {7} |
| L0 down c52 | +0.022 | 10% | **b%10** (R2 1.00): b mod 10 in {3} |
| L0 down c81 | +0.020 | 10% | **b%10** (R2 1.00): b mod 10 in {2} |

</details>

<details><summary>period 4: 1 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c127 | +0.074 | 9% | **b** (R2 1.00): b in {16, 24, 32, 36, 48, 56, 64, 72, 96} |

</details>

<details><summary>period 2: 12 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L0 down c38 | +0.046 | 10% | **b%10** (R2 1.00): b mod 10 in {9} |
| L0 down c44 | +0.042 | 10% | **b%10** (R2 1.00): b mod 10 in {7} |
| L0 down c52 | +0.042 | 10% | **b%10** (R2 1.00): b mod 10 in {3} |
| L0 down c55 | +0.035 | 10% | **b%10** (R2 1.00): b mod 10 in {6} |
| L0 down c50 | +0.034 | 10% | **b%10** (R2 1.00): b mod 10 in {8} |
| L0 down c45 | +0.033 | 10% | **b%10** (R2 1.00): b mod 10 in {5} |
| L0 down c62 | +0.032 | 10% | **b%10** (R2 1.00): b mod 10 in {4} |
| L0 down c36 | +0.031 | 10% | **b%10** (R2 1.00): b mod 10 in {1} |
| L0 down c81 | +0.028 | 10% | **b%10** (R2 1.00): b mod 10 in {2} |
| L1 down c356 | +0.027 | 10% | **b%10** (R2 1.00): b mod 10 in {9} |
| L0 down c145 | +0.026 | 10% | **b%10** (R2 1.00): b mod 10 in {0} |
| L1 down c401 | +0.025 | 10% | **b%10** (R2 0.98): b mod 10 in {7} |

</details>

</details>

<details><summary>`mlp_in.5` at `b` (addition prompts): who writes the b code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.58 | L4 MLP +0.17, L3 MLP +0.12, L0 MLP +0.08, embedding +0.07, L2 MLP +0.07, L1 MLP +0.05 |
| mod 50 | 0.51 | L4 MLP +0.13, L3 MLP +0.10, L0 MLP +0.08, embedding +0.07, L2 MLP +0.05, L1 MLP +0.04 |
| mod 25 | 0.38 | L4 MLP +0.08, embedding +0.08, L0 MLP +0.07, L3 MLP +0.06, L2 MLP +0.04, L1 MLP +0.03 |
| mod 20 | 0.42 | L4 MLP +0.09, L0 MLP +0.08, L3 MLP +0.08, embedding +0.07, L2 MLP +0.04, L1 MLP +0.04 |
| mod 10 | 0.54 | L0 MLP +0.13, L3 MLP +0.12, L4 MLP +0.11, embedding +0.10, L1 MLP +0.05, L2 MLP +0.02 |
| mod 5 | 0.55 | L3 MLP +0.12, L0 MLP +0.11, embedding +0.10, L4 MLP +0.09, L2 MLP +0.05, L1 MLP +0.04 |
| mod 4 | 0.30 | embedding +0.10, L4 MLP +0.08, L0 MLP +0.06, L3 MLP +0.02, L2 MLP +0.02 |
| mod 2 | 0.64 | L4 MLP +0.25, L0 MLP +0.15, embedding +0.11, L3 MLP +0.10, L1 MLP +0.03 |

<details><summary>period 100: 4 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L4 down c70 | +0.043 | 49% | **cmp(a,b)** (R2 0.87): cmp(a,b)=-1: 0.03, cmp(a,b)=0: 0.00, cmp(a,b)=1: 0.96 |
| L3 down c10 | +0.036 | 74% | **b//10** (R2 0.73): (tens) b in {1..70, 73, 75, 100} |
| L4 down c21 | +0.028 | 31% | **b//10** (R2 0.92): (tens) b in {68, 70..85, 87..100} |
| L0 down c6 | +0.021 | 47% | **b//10** (R2 0.90): (tens) b in {54..100} |

</details>

<details><summary>period 50: 1 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L4 down c327 | +0.025 | 13% | **b** (R2 0.94): b in {1..13} |

</details>

<details><summary>period 20: 1 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L4 down c327 | +0.029 | 13% | **b** (R2 0.94): b in {1..13} |

</details>

<details><summary>period 10: 4 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L4 down c66 | +0.076 | 38% | **b%10** (R2 0.94): b mod 10 in {6..9} |
| L3 down c63 | +0.039 | 30% | **b%10** (R2 0.99): b mod 10 in {5..7} |
| L3 down c136 | +0.032 | 31% | **b%10** (R2 0.94): b mod 10 in {7..9} |
| L3 down c54 | +0.021 | 20% | **b%10** (R2 1.00): b mod 10 in {1, 9} |

</details>

<details><summary>period 5: 4 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L4 down c43 | +0.057 | 24% | **b%50** (R2 0.85): b mod 50 in {0, 5, 10, 15, 20, 25, 30, 35, 40, 45} [coarser: b mod 5 in {0}, R2 0.86] |
| L2 down c473 | +0.042 | 20% | **b%5** (R2 1.00): b mod 5 in {0} |
| L3 down c615 | +0.037 | 19% | **b%5** (R2 0.94): b mod 5 in {0} |
| L3 down c68 | +0.030 | 30% | **b%10** (R2 0.97): b mod 10 in {3..4, 9} |

</details>

<details><summary>period 4: 2 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L4 down c135 | +0.044 | 48% | **b%2** (R2 0.92): b mod 2 in {0} |
| L0 down c127 | +0.027 | 9% | **b** (R2 1.00): b in {16, 24, 32, 36, 48, 56, 64, 72, 96} |

</details>

<details><summary>period 2: 2 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L4 down c135 | +0.244 | 48% | **b%2** (R2 0.92): b mod 2 in {0} |
| L3 down c54 | +0.048 | 20% | **b%10** (R2 1.00): b mod 10 in {1, 9} |

</details>

</details>

