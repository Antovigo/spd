[← back to the report](../report_mechanisms.md)

# Every mechanism at `b`, add

<details><summary>`b%100`: 23 mechanisms, 418 components</summary>

<details><summary><b>3a-296</b> `b%100` @ `b` (add) — block code, 3 codes of the same shape; 14 comps, L0 L1 L11 L26; tells apart 15/100 classes (best member 7); on sub: 3s-387 (member overlap 0.29)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.47 |
| classes told apart: joint / best code / best member | 15 / 15 / 7 (of 100) |
| members whose removal merges classes | 0.50 |
| support overlap (1 = tiling) / random sets / p | 1.66 / 1.28 / 0.97 |
| mean CKA between its codes | 0.73 |
| purity of the joint write (per prompt) | 0.83 |
| decoding acc. joint / best code / best member (chance) | 0.37 / 0.30 / 0.13 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.05 |
| consumers / read jointly | 2 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | embed |
| CKA(arrangement before, joint write) | 0.37 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.64 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (1:0.10 2:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.21 2:0.19 3:0.16) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.20 / 2.98 |
| best source position (CKA) | `a` (0.38) |

<details><summary>codes and components</summary>

**code 3a-296.0 (L0 L1): 4 comps, tells apart 8/100, coverage 0.16, overlap 1.25 (random 1.04, p 0.90)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 o c32 | H10 | b%100 in {0, 3, 6..8, 78} (+) | 0.98 | 1.000 |
| L0 o c0 | H10 | b%100 in {0, 56, 78} (+) | 0.66 | 1.000 |
| L0 o c2 | H2 | b%100 in {8..9, 14..15, 64, 86} (-) | 0.79 | 1.000 |
| L1 o c359 | H18 | b%100 in {1..5} (-) | 0.83 | 0.075 |

**code 3a-296.1 (L11): 8 comps, tells apart 15/100, coverage 0.43, overlap 1.14 (random 1.14, p 0.48)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L11 down c712 | MLP | b%100 in {1..5, 7} (-) | 0.82 | 0.087 |
| L11 down c31 | MLP | b%100 in {12..21} (-) | 0.98 | 0.151 |
| L11 down c133 | MLP | b%100 in {2..8} (+) | 0.62 | 0.068 |
| L11 down c61 | MLP | b%100 in {23} (-) | 0.51 | 0.018 |
| L11 down c89 | MLP | b%100 in {32, 34, 36} (+) | 0.63 | 0.024 |
| L11 down c18 | MLP | b%100 in {51..59} (+) | 0.98 | 0.121 |
| L11 down c52 | MLP | b%100 in {80..85} (+) | 0.92 | 0.081 |
| L11 down c50 | MLP | b%100 in {85..91} (-) | 0.93 | 0.089 |

**code 3a-296.2 (L26): 2 comps, tells apart 4/100, coverage 0.09, overlap 1.00 (random 1.00, p 0.86)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 down c43 | MLP | b%100 in {1} (+) | 0.54 | 0.008 |
| L26 down c1005 | MLP | b%100 in {2..9} (+) | 0.82 | 0.084 |

</details>

</details>

<details><summary><b>3a-297</b> `b%100` @ `b` (add) — tiling; 82 comps, L0; tells apart 88/100 classes (best member 9); on sub: 3s-380 (member overlap 0.78)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 88 /  / 9 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p | 2.53 / 3.60 / 0.00 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 1.00 /  / 0.24 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.25 |
| consumers / read jointly | 101 / 94 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 attn |
| CKA(arrangement before, joint write) | 0.68 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.12 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.41 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (1:0.10 2:0.06) |
| joint write: shape (spectrum k:share) | symmetric mix of circles (2:0.16 3:0.11 1:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.75 / 19.65 |

<details><summary>codes and components</summary>

**code 3a-297.0 (L0): 82 comps, tells apart 88/100, coverage 1.00, overlap 2.53 (random 3.61, p 0.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 down c34 | MLP | b%100 in {0, 3, 10, 21, 31, 33, 39, 41, 44, 46, 49, 61..62, 81..82, 86, 90..91} (+) | 0.99 | 1.000 |
| L0 down c252 | MLP | b%100 in {1..5} (+) | 0.99 | 0.080 |
| L0 down c47 | MLP | b%100 in {1..6} (-) | 0.99 | 0.060 |
| L0 down c133 | MLP | b%100 in {10} (-) | 1.00 | 0.010 |
| L0 down c169 | MLP | b%100 in {11} (+) | 1.00 | 0.010 |
| L0 down c121 | MLP | b%100 in {12} (-) | 1.00 | 0.022 |
| L0 down c53 | MLP | b%100 in {13..16} (+) | 1.00 | 0.050 |
| L0 down c127 | MLP | b%100 in {16, 32, 48, 64, 96} (-) | 1.00 | 0.090 |
| L0 down c101 | MLP | b%100 in {17..18} (-) | 1.00 | 0.040 |
| L0 down c59 | MLP | b%100 in {19..23} (+) | 1.00 | 0.120 |
| L0 down c182 | MLP | b%100 in {1} (-) | 1.00 | 0.010 |
| L0 down c423 | MLP | b%100 in {1} (-) | 1.00 | 0.010 |
| L0 down c21 | MLP | b%100 in {2..16, 18, 20, 91} (-) | 0.99 | 1.000 |
| L0 down c948 | MLP | b%100 in {20} (-) | 1.00 | 0.010 |
| L0 down c313 | MLP | b%100 in {24..25} (-) | 1.00 | 0.030 |
| L0 down c287 | MLP | b%100 in {25..27} (-) | 1.00 | 0.040 |
| L0 down c30 | MLP | b%100 in {25..30} (+) | 1.00 | 0.090 |
| L0 down c171 | MLP | b%100 in {26} (-) | 1.00 | 0.010 |
| L0 down c352 | MLP | b%100 in {2} (+) | 1.00 | 0.010 |
| L0 down c164 | MLP | b%100 in {30..33} (+) | 1.00 | 0.060 |
| L0 down c220 | MLP | b%100 in {30} (-) | 1.00 | 0.010 |
| L0 down c237 | MLP | b%100 in {31} (+) | 1.00 | 0.010 |
| L0 down c163 | MLP | b%100 in {32, 64} (+) | 1.00 | 0.020 |
| L0 down c141 | MLP | b%100 in {32..34} (+) | 1.00 | 0.040 |
| L0 down c984 | MLP | b%100 in {33} (-) | 1.00 | 0.010 |
| L0 down c469 | MLP | b%100 in {34} (-) | 1.00 | 0.010 |
| L0 down c213 | MLP | b%100 in {35..36} (-) | 1.00 | 0.030 |
| L0 down c109 | MLP | b%100 in {36, 45, 48, 54, 72} (+) | 0.96 | 0.058 |
| L0 down c63 | MLP | b%100 in {36..39} (+) | 1.00 | 0.080 |
| L0 down c165 | MLP | b%100 in {37..38} (+) | 1.00 | 0.040 |
| L0 down c119 | MLP | b%100 in {39} (+) | 1.00 | 0.010 |
| L0 down c189 | MLP | b%100 in {40..42} (+) | 1.00 | 0.030 |
| L0 down c317 | MLP | b%100 in {40} (-) | 1.00 | 0.010 |
| L0 down c170 | MLP | b%100 in {41..45} (+) | 1.00 | 0.072 |
| L0 down c572 | MLP | b%100 in {42} (-) | 1.00 | 0.010 |
| L0 down c679 | MLP | b%100 in {44} (+) | 1.00 | 0.010 |
| L0 down c524 | MLP | b%100 in {44} (-) | 1.00 | 0.010 |
| L0 down c60 | MLP | b%100 in {45..48} (-) | 1.00 | 0.050 |
| L0 down c110 | MLP | b%100 in {46..49} (+) | 1.00 | 0.050 |
| L0 down c387 | MLP | b%100 in {46} (+) | 1.00 | 0.010 |
| L0 down c406 | MLP | b%100 in {46} (+) | 1.00 | 0.010 |
| L0 down c143 | MLP | b%100 in {49..54} (+) | 1.00 | 0.070 |
| L0 down c348 | MLP | b%100 in {4} (-) | 0.99 | 0.010 |
| L0 down c43 | MLP | b%100 in {5..9} (+) | 1.00 | 0.080 |
| L0 down c135 | MLP | b%100 in {51..52} (+) | 1.00 | 0.040 |
| L0 down c321 | MLP | b%100 in {52} (+) | 1.00 | 0.010 |
| L0 down c48 | MLP | b%100 in {53..58} (-) | 1.00 | 0.080 |
| L0 down c184 | MLP | b%100 in {55} (+) | 1.00 | 0.010 |
| L0 down c167 | MLP | b%100 in {56} (-) | 1.00 | 0.010 |
| L0 down c69 | MLP | b%100 in {57..59} (-) | 1.00 | 0.070 |
| L0 down c207 | MLP | b%100 in {5} (+) | 1.00 | 0.010 |
| L0 down c54 | MLP | b%100 in {60, 80} (-) | 1.00 | 0.020 |
| L0 down c90 | MLP | b%100 in {60..65} (-) | 1.00 | 0.100 |
| L0 down c85 | MLP | b%100 in {61..62} (+) | 1.00 | 0.040 |
| L0 down c22 | MLP | b%100 in {63..76} (+) | 1.00 | 0.190 |
| L0 down c105 | MLP | b%100 in {65..69} (-) | 1.00 | 0.060 |
| L0 down c467 | MLP | b%100 in {66, 77, 88} (+) | 1.00 | 0.030 |
| L0 down c349 | MLP | b%100 in {68} (-) | 1.00 | 0.010 |
| L0 down c253 | MLP | b%100 in {69} (-) | 1.00 | 0.020 |
| L0 down c376 | MLP | b%100 in {7, 77} (+) | 0.98 | 0.020 |
| L0 down c108 | MLP | b%100 in {70..72} (+) | 1.00 | 0.050 |
| L0 down c94 | MLP | b%100 in {70..75} (-) | 1.00 | 0.080 |
| L0 down c515 | MLP | b%100 in {70} (+) | 1.00 | 0.010 |
| L0 down c75 | MLP | b%100 in {74, 76..79} (-) | 1.00 | 0.070 |
| L0 down c72 | MLP | b%100 in {75..80} (+) | 1.00 | 0.070 |
| L0 down c88 | MLP | b%100 in {79..84} (-) | 1.00 | 0.081 |
| L0 down c77 | MLP | b%100 in {80, 82..85} (-) | 1.00 | 0.070 |
| L0 down c687 | MLP | b%100 in {83} (+) | 1.00 | 0.010 |
| L0 down c97 | MLP | b%100 in {85..89} (-) | 1.00 | 0.060 |
| L0 down c738 | MLP | b%100 in {86} (-) | 1.00 | 0.010 |
| L0 down c61 | MLP | b%100 in {87} (+) | 1.00 | 0.010 |
| L0 down c318 | MLP | b%100 in {88} (+) | 1.00 | 0.020 |
| L0 down c330 | MLP | b%100 in {89} (+) | 1.00 | 0.010 |
| L0 down c351 | MLP | b%100 in {8} (-) | 0.98 | 0.010 |
| L0 down c100 | MLP | b%100 in {90..91} (-) | 1.00 | 0.030 |
| L0 down c1020 | MLP | b%100 in {91} (-) | 1.00 | 0.010 |
| L0 down c87 | MLP | b%100 in {92..95} (+) | 1.00 | 0.070 |
| L0 down c73 | MLP | b%100 in {95..98} (-) | 1.00 | 0.070 |
| L0 down c225 | MLP | b%100 in {96} (-) | 1.00 | 0.010 |
| L0 down c58 | MLP | b%100 in {98..99} (-) | 1.00 | 0.060 |
| L0 down c586 | MLP | b%100 in {99} (+) | 1.00 | 0.010 |
| L0 down c273 | MLP | b%100 in {9} (+) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>3a-298</b> `b%100` @ `b` (add) — block code; 3 comps, L1 L2; tells apart 16/100 classes (best member 5); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.17 |
| classes told apart: joint / best code / best member | 16 /  / 5 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.12 / 1.00 / 0.89 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.60 |
| decoding acc. joint / best code / best member (chance) | 0.13 /  / 0.07 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.02 |
| consumers / read jointly | 6 / 4 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.13 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.19 2:0.10 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (19:0.11 40:0.10 1:0.06) |
| frequencies new in the write | 19 |
| write: shift-symmetric part / dimension (PR) | 0.04 / 1.66 |
| best source position (CKA) | `op` (0.05) |

<details><summary>codes and components</summary>

**code 3a-298.0 (L1 L2): 3 comps, tells apart 16/100, coverage 0.17, overlap 1.12 (random 1.00, p 0.88)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c47 | H6 | b%100 in {0..1, 84, 86, 91} (+) | 0.81 | 1.000 |
| L1 o c370 | H6 | b%100 in {1..2} (+) | 0.86 | 0.027 |
| L2 o c21 | H17 | b%100 in {0, 34, 38, 42, 53, 59, 61, 64..65, 75, 87, 90} (-) | 0.53 | 1.000 |

</details>

</details>

<details><summary><b>3a-299</b> `b%100` @ `b` (add) — block code; 21 comps, L1; tells apart 56/100 classes (best member 7); on sub: 3s-381 (member overlap 0.68)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.69 |
| classes told apart: joint / best code / best member | 56 /  / 7 (of 100) |
| members whose removal merges classes | 0.76 |
| support overlap (1 = tiling) / random sets / p | 1.28 / 1.55 / 0.06 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.94 |
| decoding acc. joint / best code / best member (chance) | 0.62 /  / 0.08 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L1 attn |
| CKA(arrangement before, joint write) | 0.53 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.08 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.05 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.19 2:0.10 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.17 2:0.16 3:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.32 / 7.72 |

<details><summary>codes and components</summary>

**code 3a-299.0 (L1): 21 comps, tells apart 56/100, coverage 0.69, overlap 1.28 (random 1.48, p 0.07)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 down c481 | MLP | b%100 in {0, 99} (+) | 1.00 | 0.020 |
| L1 down c159 | MLP | b%100 in {1, 44, 61} (-) | 0.92 | 0.287 |
| L1 down c126 | MLP | b%100 in {1} (-) | 0.99 | 0.010 |
| L1 down c264 | MLP | b%100 in {25..31} (+) | 0.96 | 0.072 |
| L1 down c964 | MLP | b%100 in {33..39} (-) | 1.00 | 0.081 |
| L1 down c525 | MLP | b%100 in {40..47} (-) | 0.99 | 0.104 |
| L1 down c365 | MLP | b%100 in {47..53} (+) | 1.00 | 0.092 |
| L1 down c546 | MLP | b%100 in {48} (+) | 0.99 | 0.010 |
| L1 down c140 | MLP | b%100 in {5, 15, 25, 35, 45, 55, 65, 75, 85, 95..96} (-) | 0.97 | 0.195 |
| L1 down c80 | MLP | b%100 in {52} (-) | 1.00 | 0.010 |
| L1 down c314 | MLP | b%100 in {58} (+) | 0.82 | 0.012 |
| L1 down c631 | MLP | b%100 in {61} (-) | 0.90 | 0.020 |
| L1 down c212 | MLP | b%100 in {65..76} (+) | 0.99 | 0.173 |
| L1 down c885 | MLP | b%100 in {75..78} (-) | 1.00 | 0.041 |
| L1 down c251 | MLP | b%100 in {8..9} (+) | 0.71 | 0.023 |
| L1 down c917 | MLP | b%100 in {81..82} (-) | 1.00 | 0.020 |
| L1 down c118 | MLP | b%100 in {82..83} (-) | 0.96 | 0.019 |
| L1 down c64 | MLP | b%100 in {86..87, 91..98} (+) | 0.81 | 0.168 |
| L1 down c210 | MLP | b%100 in {87..89} (+) | 1.00 | 0.030 |
| L1 down c335 | MLP | b%100 in {90} (-) | 0.88 | 0.025 |
| L1 down c454 | MLP | b%100 in {91..92} (-) | 1.00 | 0.030 |

</details>

</details>

<details><summary><b>3a-300</b> `b%100` @ `b` (add) — block code; 28 comps, L2; tells apart 37/100 classes (best member 8); on sub: 3s-382 (member overlap 0.54)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.63 |
| classes told apart: joint / best code / best member | 37 /  / 8 (of 100) |
| members whose removal merges classes | 0.71 |
| support overlap (1 = tiling) / random sets / p | 1.51 / 1.74 / 0.14 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.88 |
| decoding acc. joint / best code / best member (chance) | 0.50 /  / 0.06 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.02 |
| consumers / read jointly | 9 / 5 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 attn |
| CKA(arrangement before, joint write) | 0.75 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.07 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.08 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.21 2:0.10 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.40 2:0.14 3:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.28 / 2.80 |

<details><summary>codes and components</summary>

**code 3a-300.0 (L2): 28 comps, tells apart 37/100, coverage 0.63, overlap 1.51 (random 1.69, p 0.15)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 down c147 | MLP | b%100 in {0, 99} (+) | 0.95 | 0.021 |
| L2 down c1 | MLP | b%100 in {1..5, 7..8, 44, 59, 78, 87} (-) | 0.60 | 1.000 |
| L2 down c885 | MLP | b%100 in {1..6} (+) | 0.92 | 0.077 |
| L2 down c8 | MLP | b%100 in {1..8, 13} (+) | 0.83 | 1.000 |
| L2 down c111 | MLP | b%100 in {12..14} (-) | 0.97 | 0.080 |
| L2 down c90 | MLP | b%100 in {17..25} (+) | 0.95 | 0.200 |
| L2 down c162 | MLP | b%100 in {18..19} (-) | 0.96 | 0.027 |
| L2 down c369 | MLP | b%100 in {18} (+) | 0.98 | 0.011 |
| L2 down c967 | MLP | b%100 in {1} (+) | 0.95 | 0.049 |
| L2 down c118 | MLP | b%100 in {24, 48, 96} (-) | 0.89 | 0.039 |
| L2 down c149 | MLP | b%100 in {24} (+) | 0.91 | 0.010 |
| L2 down c428 | MLP | b%100 in {26..28} (-) | 0.93 | 0.036 |
| L2 down c777 | MLP | b%100 in {31..32} (+) | 0.85 | 0.016 |
| L2 down c380 | MLP | b%100 in {33, 44, 66} (-) | 0.97 | 0.033 |
| L2 down c102 | MLP | b%100 in {36, 72} (+) | 0.71 | 0.025 |
| L2 down c275 | MLP | b%100 in {3} (+) | 0.80 | 0.010 |
| L2 down c934 | MLP | b%100 in {44} (+) | 0.95 | 0.011 |
| L2 down c363 | MLP | b%100 in {51..52} (-) | 0.94 | 0.029 |
| L2 down c271 | MLP | b%100 in {55} (-) | 0.93 | 0.014 |
| L2 down c617 | MLP | b%100 in {60..62} (-) | 0.98 | 0.031 |
| L2 down c1003 | MLP | b%100 in {61} (+) | 0.98 | 0.010 |
| L2 down c58 | MLP | b%100 in {76, 78..87} (+) | 0.97 | 0.210 |
| L2 down c245 | MLP | b%100 in {86} (-) | 0.99 | 0.011 |
| L2 down c247 | MLP | b%100 in {9..12} (-) | 0.96 | 0.048 |
| L2 down c92 | MLP | b%100 in {90..98} (-) | 0.98 | 0.152 |
| L2 down c448 | MLP | b%100 in {90} (-) | 0.88 | 0.014 |
| L2 down c584 | MLP | b%100 in {91} (-) | 0.95 | 0.110 |
| L2 down c864 | MLP | b%100 in {99} (-) | 0.99 | 0.011 |

</details>

</details>

<details><summary><b>3a-301</b> `b%100` @ `b` (add) — block code; 23 comps, L3; tells apart 46/100 classes (best member 8); on sub: 3s-383 (member overlap 0.36)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.69 |
| classes told apart: joint / best code / best member | 46 /  / 8 (of 100) |
| members whose removal merges classes | 0.70 |
| support overlap (1 = tiling) / random sets / p | 1.38 / 1.57 / 0.14 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.92 |
| decoding acc. joint / best code / best member (chance) | 0.54 /  / 0.06 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.97 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L3 attn |
| CKA(arrangement before, joint write) | 0.54 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.06 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.04 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.21 2:0.10 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.16 2:0.12 3:0.11) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.35 / 9.56 |

<details><summary>codes and components</summary>

**code 3a-301.0 (L3): 23 comps, tells apart 46/100, coverage 0.69, overlap 1.38 (random 1.59, p 0.10)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 down c3 | MLP | b%100 in {0..2, 44, 86} (+) | 0.73 | 0.184 |
| L3 down c849 | MLP | b%100 in {1..2} (-) | 0.85 | 0.059 |
| L3 down c325 | MLP | b%100 in {11..12} (+) | 0.94 | 0.028 |
| L3 down c176 | MLP | b%100 in {27..31} (-) | 0.71 | 0.045 |
| L3 down c61 | MLP | b%100 in {30..34, 36..37} (+) | 0.97 | 0.129 |
| L3 down c466 | MLP | b%100 in {33, 44, 55, 66, 88} (-) | 0.97 | 0.074 |
| L3 down c79 | MLP | b%100 in {38..42, 44} (-) | 0.96 | 0.067 |
| L3 down c156 | MLP | b%100 in {41..47, 63} (+) | 0.96 | 0.116 |
| L3 down c153 | MLP | b%100 in {48, 72, 96} (+) | 0.80 | 0.046 |
| L3 down c41 | MLP | b%100 in {49} (+) | 0.68 | 0.023 |
| L3 down c210 | MLP | b%100 in {57..63} (-) | 0.97 | 0.071 |
| L3 down c29 | MLP | b%100 in {64..69} (-) | 0.75 | 0.076 |
| L3 down c77 | MLP | b%100 in {73..80} (+) | 0.95 | 0.129 |
| L3 down c665 | MLP | b%100 in {76, 86} (-) | 0.98 | 0.044 |
| L3 down c121 | MLP | b%100 in {81..89} (+) | 0.97 | 0.126 |
| L3 down c269 | MLP | b%100 in {81} (+) | 0.90 | 0.018 |
| L3 down c577 | MLP | b%100 in {83} (+) | 1.00 | 0.010 |
| L3 down c298 | MLP | b%100 in {86} (-) | 0.76 | 0.008 |
| L3 down c145 | MLP | b%100 in {91..95} (+) | 0.97 | 0.067 |
| L3 down c667 | MLP | b%100 in {91} (+) | 0.95 | 0.014 |
| L3 down c476 | MLP | b%100 in {95..98} (-) | 0.83 | 0.040 |
| L3 down c122 | MLP | b%100 in {95..99} (-) | 0.99 | 0.093 |
| L3 down c225 | MLP | b%100 in {9} (-) | 0.83 | 0.010 |

</details>

</details>

<details><summary><b>3a-302</b> `b%100` @ `b` (add) — single component; 1 comps, L4; tells apart 3/100 classes (best member 3); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.09 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.52 |
| decoding acc. joint / best code / best member (chance) | 0.03 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L3 MLP |
| CKA(arrangement before, joint write) | 0.28 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.21 2:0.10 3:0.06) |
| joint write: shape (spectrum k:share) | line (3:0.11 2:0.11 5:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.03 / 1.00 |
| best source position (CKA) | `op` (0.13) |

<details><summary>codes and components</summary>

**code 3a-302.0 (L4): 1 comps, tells apart 3/100, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 o c14 | H24 | b%100 in {1..5, 7, 44, 49, 71} (+) | 0.52 | 0.998 |

</details>

</details>

<details><summary><b>3a-303</b> `b%100` @ `b` (add) — block code; 19 comps, L4; tells apart 66/100 classes (best member 9); on sub: 3s-384 (member overlap 0.29)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.89 |
| classes told apart: joint / best code / best member | 66 /  / 9 (of 100) |
| members whose removal merges classes | 0.74 |
| support overlap (1 = tiling) / random sets / p | 1.61 / 1.46 / 0.76 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.88 |
| decoding acc. joint / best code / best member (chance) | 0.77 /  / 0.06 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.96 |
| consumers / read jointly | 19 / 6 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L4 attn |
| CKA(arrangement before, joint write) | 0.54 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.08 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.05 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.21 2:0.10 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.18 50:0.15 2:0.14) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.48 / 8.18 |

<details><summary>codes and components</summary>

**code 3a-303.0 (L4): 19 comps, tells apart 66/100, coverage 0.89, overlap 1.61 (random 1.46, p 0.74)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 down c204 | MLP | b%100 in {0, 96..99} (+) | 0.97 | 0.079 |
| L4 down c1 | MLP | b%100 in {0..10, 16, 20, 25, 28, 30, 39, 50, 62, 69, 74, 76..79, 93} (+) | 0.69 | 1.000 |
| L4 down c103 | MLP | b%100 in {11..18} (-) | 0.96 | 0.095 |
| L4 down c135 | MLP | b%100 in {16, 24, 32, 36, 48, 52, 56, 64, 72, 80, 84, 96} (+) | 0.94 | 0.478 |
| L4 down c62 | MLP | b%100 in {16..31} (-) | 0.88 | 0.215 |
| L4 down c237 | MLP | b%100 in {17..21} (-) | 0.92 | 0.071 |
| L4 down c14 | MLP | b%100 in {1} (+) | 0.89 | 0.010 |
| L4 down c181 | MLP | b%100 in {31..40} (-) | 0.87 | 0.103 |
| L4 down c109 | MLP | b%100 in {32, 64} (-) | 0.77 | 0.175 |
| L4 down c625 | MLP | b%100 in {33, 44, 66, 77, 88} (-) | 0.94 | 0.077 |
| L4 down c97 | MLP | b%100 in {36, 45, 48, 54, 60, 72, 90} (+) | 0.82 | 0.078 |
| L4 down c80 | MLP | b%100 in {43..54} (+) | 0.97 | 0.158 |
| L4 down c162 | MLP | b%100 in {44, 61, 86, 91} (+) | 0.68 | 0.111 |
| L4 down c57 | MLP | b%100 in {52} (+) | 0.85 | 0.022 |
| L4 down c211 | MLP | b%100 in {55..69} (+) | 0.99 | 0.175 |
| L4 down c442 | MLP | b%100 in {55} (+) | 0.62 | 0.097 |
| L4 down c743 | MLP | b%100 in {86} (+) | 0.98 | 0.010 |
| L4 down c158 | MLP | b%100 in {88..98} (+) | 0.98 | 0.153 |
| L4 down c590 | MLP | b%100 in {99} (+) | 0.81 | 0.031 |

</details>

</details>

<details><summary><b>3a-304</b> `b%100` @ `b` (add) — block code; 4 comps, L5; tells apart 9/100 classes (best member 7); on sub: 3s-385 (member overlap 0.20)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.05 |
| classes told apart: joint / best code / best member | 9 /  / 7 (of 100) |
| members whose removal merges classes | 0.50 |
| support overlap (1 = tiling) / random sets / p | 1.60 / 1.00 / 0.98 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.57 |
| decoding acc. joint / best code / best member (chance) | 0.07 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.98 |
| consumers / read jointly | 2 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L5 attn |
| CKA(arrangement before, joint write) | 0.41 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.02 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.22 2:0.10 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.26 2:0.14 3:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.06 / 1.22 |

<details><summary>codes and components</summary>

**code 3a-304.0 (L5): 4 comps, tells apart 9/100, coverage 0.05, overlap 1.60 (random 1.03, p 0.99)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 down c230 | MLP | b%100 in {1..2} (+) | 0.91 | 0.065 |
| L5 down c0 | MLP | b%100 in {1..3} (-) | 0.53 | 0.989 |
| L5 down c43 | MLP | b%100 in {1} (-) | 0.89 | 0.010 |
| L5 down c105 | MLP | b%100 in {86, 91} (-) | 0.74 | 0.026 |

</details>

</details>

<details><summary><b>3a-305</b> `b%100` @ `b` (add) — single component; 1 comps, L6; tells apart 4/100 classes (best member 4); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.03 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.52 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L5 MLP |
| CKA(arrangement before, joint write) | 0.19 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.22 2:0.10 3:0.06) |
| joint write: shape (spectrum k:share) | line (20:0.09 40:0.09 5:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.02 / 1.00 |
| best source position (CKA) | `op` (0.03) |

<details><summary>codes and components</summary>

**code 3a-305.0 (L6): 1 comps, tells apart 4/100, coverage 0.03, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L6 o c4 | H22 | b%100 in {0..2} (-) | 0.52 | 1.000 |

</details>

</details>

<details><summary><b>3a-306</b> `b%100` @ `b` (add) — single component; 1 comps, L6; tells apart 4/100 classes (best member 4); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.07 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.51 |
| decoding acc. joint / best code / best member (chance) | 0.03 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L5 MLP |
| CKA(arrangement before, joint write) | 0.42 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.22 2:0.10 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.19 2:0.13 3:0.12) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.04 / 1.00 |
| best source position (CKA) | `a` (0.25) |

<details><summary>codes and components</summary>

**code 3a-306.0 (L6): 1 comps, tells apart 4/100, coverage 0.07, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L6 o c23 | H8 | b%100 in {0..5, 7} (+) | 0.51 | 0.995 |

</details>

</details>

<details><summary><b>3a-307</b> `b%100` @ `b` (add) — block code, 2 codes of the same shape; 7 comps, L6 L9; tells apart 16/100 classes (best member 9); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.27 |
| classes told apart: joint / best code / best member | 16 / 11 / 9 (of 100) |
| members whose removal merges classes | 0.57 |
| support overlap (1 = tiling) / random sets / p | 1.78 / 1.11 / 1.00 |
| mean CKA between its codes | 0.94 |
| purity of the joint write (per prompt) | 0.68 |
| decoding acc. joint / best code / best member (chance) | 0.10 / 0.07 / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.03 |
| consumers / read jointly | 3 / 3 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L6 attn |
| CKA(arrangement before, joint write) | 0.64 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.03 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.06 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.22 2:0.10 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.35 2:0.14 3:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.12 / 1.44 |

<details><summary>codes and components</summary>

**code 3a-307.0 (L6): 4 comps, tells apart 11/100, coverage 0.12, overlap 1.08 (random 1.00, p 0.70)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L6 down c283 | MLP | b%100 in {0} (+) | 0.68 | 0.033 |
| L6 down c94 | MLP | b%100 in {1..8, 10, 20} (+) | 0.74 | 0.427 |
| L6 down c592 | MLP | b%100 in {1} (-) | 0.92 | 0.011 |
| L6 down c381 | MLP | b%100 in {91} (+) | 0.69 | 0.055 |

**code 3a-307.1 (L9): 3 comps, tells apart 10/100, coverage 0.26, overlap 1.35 (random 1.00, p 0.96)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L9 down c44 | MLP | b%100 in {0..13, 15, 20} (-) | 0.59 | 0.482 |
| L9 down c64 | MLP | b%100 in {1, 11, 13, 19, 29, 31, 39, 41, 49, 51, 59, 61, 69} (-) | 0.58 | 0.156 |
| L9 down c171 | MLP | b%100 in {1..5, 7} (+) | 0.81 | 0.082 |

</details>

</details>

<details><summary><b>3a-308</b> `b%100` @ `b` (add) — block code, 4 codes of the same shape; 10 comps, L7 L8 L10 L30; tells apart 11/100 classes (best member 7); on sub: 3s-386 (member overlap 0.07)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.05 |
| classes told apart: joint / best code / best member | 11 / 9 / 7 (of 100) |
| members whose removal merges classes | 0.20 |
| support overlap (1 = tiling) / random sets / p | 2.60 / 1.20 / 1.00 |
| mean CKA between its codes | 0.81 |
| purity of the joint write (per prompt) | 0.59 |
| decoding acc. joint / best code / best member (chance) | 0.06 / 0.05 / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.04 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L7 attn |
| CKA(arrangement before, joint write) | 0.30 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.03 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.22 2:0.10 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.09 2:0.09 3:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.03 / 1.32 |

<details><summary>codes and components</summary>

**code 3a-308.0 (L7): 5 comps, tells apart 6/100, coverage 0.03, overlap 1.67 (random 1.07, p 1.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L7 down c12 | MLP | b%100 in {1} (+) | 0.85 | 0.010 |
| L7 down c597 | MLP | b%100 in {1} (-) | 0.88 | 0.026 |
| L7 down c1003 | MLP | b%100 in {86} (+) | 0.93 | 0.010 |
| L7 down c53 | MLP | b%100 in {86} (-) | 0.71 | 0.007 |
| L7 down c38 | MLP | b%100 in {99} (-) | 0.94 | 0.011 |

**code 3a-308.1 (L8): 2 comps, tells apart 7/100, coverage 0.02, overlap 1.00 (random 1.00, p 0.86)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L8 down c627 | MLP | b%100 in {1} (-) | 0.83 | 0.072 |
| L8 down c43 | MLP | b%100 in {86} (-) | 0.66 | 0.008 |

**code 3a-308.2 (L10): 1 comps, tells apart 6/100, coverage 0.03, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L10 down c527 | MLP | b%100 in {1..3} (-) | 0.51 | 0.030 |

**code 3a-308.3 (L30): 2 comps, tells apart 9/100, coverage 0.03, overlap 1.00 (random 1.00, p 0.87)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 down c18 | MLP | b%100 in {1} (+) | 0.53 | 0.057 |
| L30 down c37 | MLP | b%100 in {2..3} (+) | 0.55 | 0.033 |

</details>

</details>

<details><summary><b>3a-309</b> `b%100` @ `b` (add) — block code; 10 comps, L12; tells apart 19/100 classes (best member 7); on sub: 3s-388 (member overlap 0.40)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.49 |
| classes told apart: joint / best code / best member | 19 /  / 7 (of 100) |
| members whose removal merges classes | 0.90 |
| support overlap (1 = tiling) / random sets / p | 1.08 / 1.19 / 0.14 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.95 |
| decoding acc. joint / best code / best member (chance) | 0.33 /  / 0.08 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 12 / 2 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L12 attn |
| CKA(arrangement before, joint write) | 0.46 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.08 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.04 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.23 2:0.11 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.27 2:0.24 3:0.14) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.32 / 3.80 |

<details><summary>codes and components</summary>

**code 3a-309.0 (L12): 10 comps, tells apart 19/100, coverage 0.49, overlap 1.08 (random 1.20, p 0.15)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L12 down c303 | MLP | b%100 in {1..2} (-) | 0.76 | 0.071 |
| L12 down c22 | MLP | b%100 in {12..22} (-) | 0.99 | 0.151 |
| L12 down c861 | MLP | b%100 in {1} (+) | 0.86 | 0.009 |
| L12 down c633 | MLP | b%100 in {2..9} (+) | 0.89 | 0.080 |
| L12 down c18 | MLP | b%100 in {23..29} (+) | 0.99 | 0.142 |
| L12 down c76 | MLP | b%100 in {33, 44, 55, 66, 77, 88, 99} (+) | 0.94 | 0.073 |
| L12 down c550 | MLP | b%100 in {56..57} (+) | 0.70 | 0.024 |
| L12 down c17 | MLP | b%100 in {71..81} (+) | 0.99 | 0.173 |
| L12 down c341 | MLP | b%100 in {86} (-) | 0.80 | 0.009 |
| L12 down c224 | MLP | b%100 in {97..99} (-) | 0.82 | 0.052 |

</details>

</details>

<details><summary><b>3a-310</b> `b%100` @ `b` (add) — block code; 21 comps, L13; tells apart 30/100 classes (best member 8); on sub: 3s-389 (member overlap 0.59)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.75 |
| classes told apart: joint / best code / best member | 30 /  / 8 (of 100) |
| members whose removal merges classes | 0.38 |
| support overlap (1 = tiling) / random sets / p | 1.44 / 1.48 / 0.41 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.95 |
| decoding acc. joint / best code / best member (chance) | 0.56 /  / 0.07 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.97 |
| consumers / read jointly | 56 / 25 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L13 attn |
| CKA(arrangement before, joint write) | 0.59 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.08 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.09 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.23 2:0.12 10:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.28 2:0.23 3:0.14) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.59 / 7.11 |

<details><summary>codes and components</summary>

**code 3a-310.0 (L13): 21 comps, tells apart 30/100, coverage 0.75, overlap 1.44 (random 1.51, p 0.35)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 down c3 | MLP | b%100 in {1..2} (-) | 0.84 | 0.031 |
| L13 down c76 | MLP | b%100 in {1..4} (-) | 0.83 | 0.042 |
| L13 down c149 | MLP | b%100 in {1..5, 7} (-) | 0.84 | 0.079 |
| L13 down c774 | MLP | b%100 in {15..16} (+) | 0.92 | 0.024 |
| L13 down c195 | MLP | b%100 in {20..23} (-) | 0.97 | 0.092 |
| L13 down c54 | MLP | b%100 in {31..32, 61..63} (-) | 0.73 | 0.052 |
| L13 down c46 | MLP | b%100 in {32..34, 52..54} (-) | 0.94 | 0.071 |
| L13 down c545 | MLP | b%100 in {32} (+) | 0.95 | 0.011 |
| L13 down c187 | MLP | b%100 in {33, 66} (-) | 0.98 | 0.020 |
| L13 down c23 | MLP | b%100 in {35..42} (-) | 0.97 | 0.148 |
| L13 down c24 | MLP | b%100 in {4..12} (-) | 0.98 | 0.138 |
| L13 down c21 | MLP | b%100 in {44..53} (+) | 0.98 | 0.169 |
| L13 down c20 | MLP | b%100 in {52..65} (-) | 0.98 | 0.177 |
| L13 down c133 | MLP | b%100 in {64..66} (-) | 0.89 | 0.036 |
| L13 down c27 | MLP | b%100 in {65..74} (+) | 0.98 | 0.177 |
| L13 down c433 | MLP | b%100 in {75} (+) | 0.96 | 0.011 |
| L13 down c999 | MLP | b%100 in {76..77} (+) | 0.84 | 0.029 |
| L13 down c96 | MLP | b%100 in {86} (-) | 0.79 | 0.066 |
| L13 down c48 | MLP | b%100 in {90..97} (+) | 0.96 | 0.098 |
| L13 down c16 | MLP | b%100 in {93..99} (-) | 0.94 | 0.142 |
| L13 down c155 | MLP | b%100 in {97..99} (+) | 0.63 | 0.067 |

</details>

</details>

<details><summary><b>3a-311</b> `b%100` @ `b` (add) — tiling; 42 comps, L14; tells apart 69/100 classes (best member 7); on sub: 3s-390 (member overlap 0.57)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.86 |
| classes told apart: joint / best code / best member | 69 /  / 7 (of 100) |
| members whose removal merges classes | 0.79 |
| support overlap (1 = tiling) / random sets / p | 1.72 / 2.16 / 0.01 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.85 |
| decoding acc. joint / best code / best member (chance) | 0.63 /  / 0.07 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.96 |
| consumers / read jointly | 20 / 12 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L14 attn |
| CKA(arrangement before, joint write) | 0.67 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.07 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.22 2:0.12 10:0.08) |
| joint write: shape (spectrum k:share) | irregular (1:0.21 2:0.17 3:0.12) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.45 / 8.77 |

<details><summary>codes and components</summary>

**code 3a-311.0 (L14): 42 comps, tells apart 69/100, coverage 0.86, overlap 1.72 (random 2.16, p 0.01)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L14 down c400 | MLP | b%100 in {0, 10} (-) | 0.97 | 0.021 |
| L14 down c588 | MLP | b%100 in {0, 97..99} (-) | 0.91 | 0.046 |
| L14 down c81 | MLP | b%100 in {0..6, 10} (+) | 0.74 | 0.274 |
| L14 down c170 | MLP | b%100 in {1..4} (-) | 0.94 | 0.072 |
| L14 down c637 | MLP | b%100 in {10} (-) | 0.96 | 0.010 |
| L14 down c542 | MLP | b%100 in {11..12} (-) | 0.93 | 0.041 |
| L14 down c131 | MLP | b%100 in {13..14} (+) | 0.98 | 0.020 |
| L14 down c851 | MLP | b%100 in {14..26} (+) | 0.86 | 0.181 |
| L14 down c615 | MLP | b%100 in {15} (-) | 0.93 | 0.018 |
| L14 down c96 | MLP | b%100 in {16..17} (+) | 0.96 | 0.028 |
| L14 down c666 | MLP | b%100 in {19, 59, 61} (+) | 0.80 | 0.041 |
| L14 down c24 | MLP | b%100 in {1} (+) | 0.53 | 0.006 |
| L14 down c88 | MLP | b%100 in {2..11} (+) | 0.78 | 0.125 |
| L14 down c248 | MLP | b%100 in {23..24} (-) | 0.98 | 0.021 |
| L14 down c215 | MLP | b%100 in {25, 50} (-) | 0.95 | 0.024 |
| L14 down c45 | MLP | b%100 in {25..34} (+) | 0.98 | 0.141 |
| L14 down c48 | MLP | b%100 in {26..30} (-) | 0.90 | 0.058 |
| L14 down c151 | MLP | b%100 in {3..9} (+) | 0.95 | 0.081 |
| L14 down c547 | MLP | b%100 in {34..36} (-) | 0.76 | 0.024 |
| L14 down c41 | MLP | b%100 in {35, 37..42} (-) | 0.90 | 0.104 |
| L14 down c176 | MLP | b%100 in {36, 56..57} (+) | 0.67 | 0.029 |
| L14 down c42 | MLP | b%100 in {38, 58, 62, 78, 82..83, 98} (+) | 0.83 | 0.079 |
| L14 down c129 | MLP | b%100 in {44, 55} (+) | 0.93 | 0.019 |
| L14 down c65 | MLP | b%100 in {47..50} (-) | 0.74 | 0.061 |
| L14 down c625 | MLP | b%100 in {48, 96} (-) | 0.79 | 0.025 |
| L14 down c441 | MLP | b%100 in {52} (+) | 0.95 | 0.010 |
| L14 down c671 | MLP | b%100 in {53..54} (-) | 0.87 | 0.023 |
| L14 down c70 | MLP | b%100 in {55, 57..58, 61} (+) | 0.61 | 0.062 |
| L14 down c787 | MLP | b%100 in {5} (+) | 0.94 | 0.010 |
| L14 down c701 | MLP | b%100 in {65..66} (+) | 0.65 | 0.021 |
| L14 down c833 | MLP | b%100 in {66..68} (+) | 0.77 | 0.031 |
| L14 down c201 | MLP | b%100 in {7..8} (+) | 0.94 | 0.020 |
| L14 down c213 | MLP | b%100 in {70..71} (+) | 0.93 | 0.020 |
| L14 down c320 | MLP | b%100 in {72..74} (-) | 0.94 | 0.032 |
| L14 down c182 | MLP | b%100 in {76..79} (+) | 0.93 | 0.062 |
| L14 down c799 | MLP | b%100 in {80} (+) | 0.94 | 0.010 |
| L14 down c644 | MLP | b%100 in {82..83} (+) | 0.70 | 0.024 |
| L14 down c835 | MLP | b%100 in {84..91} (-) | 0.82 | 0.170 |
| L14 down c104 | MLP | b%100 in {88..89} (-) | 0.89 | 0.038 |
| L14 down c290 | MLP | b%100 in {89..90} (+) | 0.78 | 0.114 |
| L14 down c229 | MLP | b%100 in {91} (+) | 0.56 | 0.015 |
| L14 down c148 | MLP | b%100 in {99} (-) | 0.97 | 0.010 |

</details>

</details>

<details><summary><b>3a-312</b> `b%100` @ `b` (add) — tiling; 28 comps, L15; tells apart 52/100 classes (best member 9); on sub: 3s-391 (member overlap 0.35)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.71 |
| classes told apart: joint / best code / best member | 52 /  / 9 (of 100) |
| members whose removal merges classes | 0.89 |
| support overlap (1 = tiling) / random sets / p | 1.13 / 1.74 / 0.00 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.85 |
| decoding acc. joint / best code / best member (chance) | 0.51 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.99 |
| consumers / read jointly | 7 / 1 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 attn |
| CKA(arrangement before, joint write) | 0.61 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.04 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.22 2:0.13 10:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.14 2:0.12 3:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.35 / 10.77 |

<details><summary>codes and components</summary>

**code 3a-312.0 (L15): 28 comps, tells apart 52/100, coverage 0.71, overlap 1.13 (random 1.69, p 0.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c170 | MLP | b%100 in {1..8} (+) | 0.81 | 0.140 |
| L15 down c45 | MLP | b%100 in {10..15} (+) | 0.96 | 0.114 |
| L15 down c96 | MLP | b%100 in {11} (-) | 0.97 | 0.011 |
| L15 down c42 | MLP | b%100 in {16, 32, 64, 96} (-) | 0.78 | 0.093 |
| L15 down c451 | MLP | b%100 in {20..21} (+) | 0.90 | 0.019 |
| L15 down c584 | MLP | b%100 in {21} (-) | 0.97 | 0.010 |
| L15 down c98 | MLP | b%100 in {22..23} (-) | 0.95 | 0.022 |
| L15 down c236 | MLP | b%100 in {24} (+) | 0.90 | 0.014 |
| L15 down c188 | MLP | b%100 in {28} (+) | 0.87 | 0.015 |
| L15 down c426 | MLP | b%100 in {32, 52, 72} (-) | 0.82 | 0.032 |
| L15 down c772 | MLP | b%100 in {32..33} (-) | 0.95 | 0.021 |
| L15 down c888 | MLP | b%100 in {34} (-) | 0.96 | 0.012 |
| L15 down c743 | MLP | b%100 in {36..37} (+) | 0.80 | 0.019 |
| L15 down c104 | MLP | b%100 in {40, 60, 80} (+) | 0.66 | 0.040 |
| L15 down c470 | MLP | b%100 in {44} (+) | 0.84 | 0.023 |
| L15 down c184 | MLP | b%100 in {45, 90} (-) | 0.97 | 0.021 |
| L15 down c110 | MLP | b%100 in {48..51} (+) | 0.88 | 0.069 |
| L15 down c73 | MLP | b%100 in {53..55, 57} (+) | 0.88 | 0.096 |
| L15 down c144 | MLP | b%100 in {58..61} (-) | 0.82 | 0.052 |
| L15 down c894 | MLP | b%100 in {63} (-) | 0.68 | 0.015 |
| L15 down c904 | MLP | b%100 in {65..67} (-) | 0.81 | 0.039 |
| L15 down c366 | MLP | b%100 in {69..71, 73..74} (+) | 0.60 | 0.046 |
| L15 down c233 | MLP | b%100 in {76..79} (-) | 0.82 | 0.069 |
| L15 down c854 | MLP | b%100 in {85..86} (-) | 0.54 | 0.023 |
| L15 down c53 | MLP | b%100 in {90..95} (+) | 0.88 | 0.079 |
| L15 down c364 | MLP | b%100 in {95..99} (-) | 0.87 | 0.053 |
| L15 down c762 | MLP | b%100 in {99} (+) | 0.83 | 0.012 |
| L15 down c310 | MLP | b%100 in {9} (+) | 0.89 | 0.011 |

</details>

</details>

<details><summary><b>3a-313</b> `b%100` @ `b` (add) — block code; 21 comps, L16; tells apart 43/100 classes (best member 7); on sub: 3s-392 (member overlap 0.08)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.42 |
| classes told apart: joint / best code / best member | 43 /  / 7 (of 100) |
| members whose removal merges classes | 0.81 |
| support overlap (1 = tiling) / random sets / p | 1.31 / 1.52 / 0.10 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.75 |
| decoding acc. joint / best code / best member (chance) | 0.31 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.06 |
| consumers / read jointly | 3 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L16 attn |
| CKA(arrangement before, joint write) | 0.53 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.23 2:0.13 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.16 2:0.11 3:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.22 / 7.02 |

<details><summary>codes and components</summary>

**code 3a-313.0 (L16): 21 comps, tells apart 43/100, coverage 0.42, overlap 1.31 (random 1.55, p 0.09)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c407 | MLP | b%100 in {0, 20, 30, 50} (+) | 0.61 | 0.129 |
| L16 down c170 | MLP | b%100 in {1..3} (+) | 0.72 | 0.040 |
| L16 down c6 | MLP | b%100 in {1..9} (+) | 0.82 | 0.108 |
| L16 down c557 | MLP | b%100 in {16, 96} (+) | 0.84 | 0.022 |
| L16 down c209 | MLP | b%100 in {18} (+) | 0.82 | 0.017 |
| L16 down c30 | MLP | b%100 in {24, 48, 96} (+) | 0.70 | 0.046 |
| L16 down c44 | MLP | b%100 in {30..31} (-) | 0.86 | 0.049 |
| L16 down c554 | MLP | b%100 in {30} (-) | 0.65 | 0.007 |
| L16 down c3 | MLP | b%100 in {38..44} (+) | 0.55 | 0.120 |
| L16 down c272 | MLP | b%100 in {3} (-) | 0.62 | 0.010 |
| L16 down c370 | MLP | b%100 in {40} (-) | 0.78 | 0.018 |
| L16 down c151 | MLP | b%100 in {42, 52} (-) | 0.76 | 0.022 |
| L16 down c929 | MLP | b%100 in {52} (+) | 0.95 | 0.011 |
| L16 down c461 | MLP | b%100 in {55} (+) | 0.87 | 0.013 |
| L16 down c354 | MLP | b%100 in {64..65} (+) | 0.66 | 0.018 |
| L16 down c438 | MLP | b%100 in {69..70} (+) | 0.66 | 0.029 |
| L16 down c74 | MLP | b%100 in {7, 67, 77} (+) | 0.77 | 0.051 |
| L16 down c57 | MLP | b%100 in {79..82, 84} (-) | 0.80 | 0.100 |
| L16 down c34 | MLP | b%100 in {8, 68, 88} (-) | 0.65 | 0.060 |
| L16 down c248 | MLP | b%100 in {86} (-) | 0.89 | 0.012 |
| L16 down c692 | MLP | b%100 in {88} (+) | 0.91 | 0.012 |

</details>

</details>

<details><summary><b>3a-314</b> `b%100` @ `b` (add) — tiling; 24 comps, L17; tells apart 34/100 classes (best member 7); on sub: 3s-393 (member overlap 0.13)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.36 |
| classes told apart: joint / best code / best member | 34 /  / 7 (of 100) |
| members whose removal merges classes | 0.75 |
| support overlap (1 = tiling) / random sets / p | 1.11 / 1.62 / 0.00 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.76 |
| decoding acc. joint / best code / best member (chance) | 0.24 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.99 |
| consumers / read jointly | 2 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L17 attn |
| CKA(arrangement before, joint write) | 0.36 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.07 |
| share of the write inside the old arrangement's span | 0.09 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.24 2:0.12 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.13 2:0.09 3:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.18 / 8.71 |

<details><summary>codes and components</summary>

**code 3a-314.0 (L17): 24 comps, tells apart 34/100, coverage 0.36, overlap 1.11 (random 1.59, p 0.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 down c56 | MLP | b%100 in {10..11} (-) | 0.81 | 0.044 |
| L17 down c66 | MLP | b%100 in {12} (+) | 0.77 | 0.021 |
| L17 down c186 | MLP | b%100 in {15} (+) | 0.82 | 0.010 |
| L17 down c48 | MLP | b%100 in {18..29} (-) | 0.70 | 0.129 |
| L17 down c0 | MLP | b%100 in {1} (-) | 0.86 | 0.027 |
| L17 down c265 | MLP | b%100 in {32} (-) | 0.74 | 0.020 |
| L17 down c165 | MLP | b%100 in {33} (-) | 0.79 | 0.016 |
| L17 down c779 | MLP | b%100 in {33} (-) | 0.98 | 0.010 |
| L17 down c172 | MLP | b%100 in {37..38} (+) | 0.73 | 0.024 |
| L17 down c229 | MLP | b%100 in {40} (-) | 0.51 | 0.013 |
| L17 down c383 | MLP | b%100 in {40} (-) | 0.58 | 0.011 |
| L17 down c106 | MLP | b%100 in {42} (-) | 0.63 | 0.007 |
| L17 down c465 | MLP | b%100 in {44} (+) | 0.69 | 0.014 |
| L17 down c303 | MLP | b%100 in {48, 96} (-) | 0.65 | 0.020 |
| L17 down c144 | MLP | b%100 in {50} (+) | 0.88 | 0.018 |
| L17 down c458 | MLP | b%100 in {52} (-) | 0.80 | 0.009 |
| L17 down c249 | MLP | b%100 in {54} (-) | 0.71 | 0.009 |
| L17 down c161 | MLP | b%100 in {5} (-) | 0.57 | 0.014 |
| L17 down c54 | MLP | b%100 in {6, 76} (+) | 0.73 | 0.048 |
| L17 down c212 | MLP | b%100 in {61..62} (-) | 0.87 | 0.043 |
| L17 down c778 | MLP | b%100 in {86} (+) | 0.94 | 0.011 |
| L17 down c138 | MLP | b%100 in {86} (-) | 0.79 | 0.016 |
| L17 down c195 | MLP | b%100 in {91} (+) | 0.54 | 0.007 |
| L17 down c314 | MLP | b%100 in {91} (-) | 0.70 | 0.013 |

</details>

</details>

<details><summary><b>3a-315</b> `b%100` @ `b` (add) — block code, 6 codes of the same shape; 35 comps, L18 L19 L20 L22 L23 L24 L27; tells apart 27/100 classes (best member 8); on sub: 3s-394 (member overlap 0.05)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.73 |
| classes told apart: joint / best code / best member | 27 / 19 / 8 (of 100) |
| members whose removal merges classes | 0.43 |
| support overlap (1 = tiling) / random sets / p | 3.96 / 1.95 / 1.00 |
| mean CKA between its codes | 0.92 |
| purity of the joint write (per prompt) | 0.57 |
| decoding acc. joint / best code / best member (chance) | 0.19 / 0.14 / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.10 |
| consumers / read jointly | 26 / 26 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L18 attn |
| CKA(arrangement before, joint write) | 0.72 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.30 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.24 2:0.12 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.43 2:0.09 3:0.05) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.15 / 1.47 |

<details><summary>codes and components</summary>

**code 3a-315.0 (L18 L19): 10 comps, tells apart 12/100, coverage 0.34, overlap 1.26 (random 1.17, p 0.75)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 down c9 | MLP | b%100 in {1} (+) | 0.66 | 0.007 |
| L18 down c243 | MLP | b%100 in {1} (-) | 0.65 | 0.024 |
| L18 down c693 | MLP | b%100 in {95..96, 98} (+) | 0.60 | 0.039 |
| L18 down c40 | MLP | b%100 in {99} (-) | 0.85 | 0.013 |
| L19 down c101 | MLP | b%100 in {1..4} (-) | 0.59 | 0.063 |
| L19 down c225 | MLP | b%100 in {1..8, 10..11, 13, 32, 36, 56..57, 60, 63..66, 71..72, 75, 81..84, 87, 89} (-) | 0.52 | 0.673 |
| L19 down c52 | MLP | b%100 in {1} (-) | 0.90 | 0.082 |
| L19 down c93 | MLP | b%100 in {3} (-) | 0.80 | 0.029 |
| L19 down c78 | MLP | b%100 in {7} (-) | 0.70 | 0.026 |
| L19 down c51 | MLP | b%100 in {9} (-) | 0.87 | 0.010 |

**code 3a-315.1 (L20): 11 comps, tells apart 17/100, coverage 0.47, overlap 1.15 (random 1.23, p 0.26)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 down c17 | MLP | b%100 in {0..14, 18..19, 32, 36, 56, 60, 62..66, 71..72, 75, 81..84, 89} (+) | 0.56 | 0.800 |
| L20 down c248 | MLP | b%100 in {0} (+) | 0.96 | 0.035 |
| L20 down c1 | MLP | b%100 in {1} (+) | 0.77 | 0.046 |
| L20 down c108 | MLP | b%100 in {1} (-) | 0.71 | 0.015 |
| L20 down c81 | MLP | b%100 in {22..23} (+) | 0.67 | 0.036 |
| L20 down c22 | MLP | b%100 in {31} (-) | 0.81 | 0.042 |
| L20 down c46 | MLP | b%100 in {44, 46..49} (+) | 0.77 | 0.086 |
| L20 down c66 | MLP | b%100 in {64..68} (-) | 0.81 | 0.052 |
| L20 down c77 | MLP | b%100 in {76..77} (+) | 0.77 | 0.029 |
| L20 down c348 | MLP | b%100 in {7} (-) | 0.79 | 0.025 |
| L20 down c932 | MLP | b%100 in {99} (-) | 0.96 | 0.011 |

**code 3a-315.2 (L22): 6 comps, tells apart 19/100, coverage 0.38, overlap 1.18 (random 1.09, p 0.77)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 down c165 | MLP | b%100 in {0, 20, 30, 90} (-) | 0.52 | 0.242 |
| L22 down c37 | MLP | b%100 in {0..1} (+) | 0.69 | 0.038 |
| L22 down c71 | MLP | b%100 in {0..2} (-) | 0.71 | 0.038 |
| L22 down c96 | MLP | b%100 in {0} (+) | 0.75 | 0.017 |
| L22 down c66 | MLP | b%100 in {1..6, 8..9, 11, 13, 32, 36, 56, 63..65, 72, 75, 81..84, 89} (-) | 0.56 | 0.462 |
| L22 down c91 | MLP | b%100 in {86..89, 91..97, 99} (-) | 0.65 | 0.149 |

**code 3a-315.3 (L23): 2 comps, tells apart 3/100, coverage 0.27, overlap 1.04 (random 1.00, p 0.85)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 down c62 | MLP | b%100 in {0..4, 32, 36, 45, 53, 56..57, 60, 62..66, 71..74, 81..84, 87, 89} (-) | 0.54 | 0.584 |
| L23 down c101 | MLP | b%100 in {0} (+) | 0.65 | 0.044 |

**code 3a-315.4 (L24): 5 comps, tells apart 15/100, coverage 0.58, overlap 1.40 (random 1.06, p 0.96)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 down c86 | MLP | b%100 in {0, 10, 15, 20, 25, 30, 40} (+) | 0.63 | 0.235 |
| L24 down c93 | MLP | b%100 in {0, 76..80, 85..86, 88, 90..99} (-) | 0.72 | 0.300 |
| L24 down c201 | MLP | b%100 in {0..11, 13, 18..19, 32, 36, 56..57, 60, 62..65, 68, 71..74, 81..84, 87, 89} (+) | 0.55 | 0.667 |
| L24 down c264 | MLP | b%100 in {1..11} (+) | 0.81 | 0.109 |
| L24 down c0 | MLP | b%100 in {1..9} (-) | 0.56 | 1.000 |

**code 3a-315.5 (L27): 1 comps, tells apart 1/100, coverage 0.38, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 down c201 | MLP | b%100 in {0..11, 13..15, 18..20, 32, 36, 53, 56..57, 62..66, 71..74, 81..84, 87, 89} (+) | 0.53 | 0.641 |

</details>

</details>

<details><summary><b>3a-316</b> `b%100` @ `b` (add) — block code; 15 comps, L21; tells apart 29/100 classes (best member 8); on sub: 3s-395 (member overlap 0.12)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.65 |
| classes told apart: joint / best code / best member | 29 /  / 8 (of 100) |
| members whose removal merges classes | 0.40 |
| support overlap (1 = tiling) / random sets / p | 1.15 / 1.35 / 0.09 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.81 |
| decoding acc. joint / best code / best member (chance) | 0.27 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 attn |
| CKA(arrangement before, joint write) | 0.49 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.10 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.24 2:0.12 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.35 2:0.23 3:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.40 / 4.14 |

<details><summary>codes and components</summary>

**code 3a-316.0 (L21): 15 comps, tells apart 29/100, coverage 0.65, overlap 1.15 (random 1.33, p 0.07)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c47 | MLP | b%100 in {1, 3, 5, 7} (+) | 0.58 | 0.042 |
| L21 down c56 | MLP | b%100 in {12, 14..26, 28, 30} (-) | 0.87 | 0.224 |
| L21 down c326 | MLP | b%100 in {13} (-) | 0.85 | 0.016 |
| L21 down c302 | MLP | b%100 in {16, 96} (+) | 0.78 | 0.025 |
| L21 down c426 | MLP | b%100 in {27} (+) | 0.92 | 0.010 |
| L21 down c18 | MLP | b%100 in {30..31, 33..42} (-) | 0.82 | 0.158 |
| L21 down c219 | MLP | b%100 in {3} (+) | 0.78 | 0.015 |
| L21 down c34 | MLP | b%100 in {46..53, 55} (-) | 0.71 | 0.097 |
| L21 down c104 | MLP | b%100 in {47, 57, 67, 77, 97} (-) | 0.60 | 0.046 |
| L21 down c33 | MLP | b%100 in {48, 64, 96} (+) | 0.70 | 0.077 |
| L21 down c364 | MLP | b%100 in {4} (-) | 0.81 | 0.038 |
| L21 down c154 | MLP | b%100 in {79, 82, 84..85} (-) | 0.57 | 0.055 |
| L21 down c97 | MLP | b%100 in {85, 87..99} (-) | 0.82 | 0.175 |
| L21 down c98 | MLP | b%100 in {8} (-) | 0.94 | 0.010 |
| L21 down c539 | MLP | b%100 in {91} (+) | 0.84 | 0.012 |

</details>

</details>

<details><summary><b>3a-317</b> `b%100` @ `b` (add) — block code; 3 comps, L25; tells apart 8/100 classes (best member 6); on sub: 3s-397 (member overlap 0.33)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.13 |
| classes told apart: joint / best code / best member | 8 /  / 6 (of 100) |
| members whose removal merges classes | 0.67 |
| support overlap (1 = tiling) / random sets / p | 1.23 / 1.00 / 0.94 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.73 |
| decoding acc. joint / best code / best member (chance) | 0.06 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L25 attn |
| CKA(arrangement before, joint write) | 0.31 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.23 2:0.12 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (10:0.14 1:0.10 3:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.07 / 2.00 |

<details><summary>codes and components</summary>

**code 3a-317.0 (L25): 3 comps, tells apart 8/100, coverage 0.13, overlap 1.23 (random 1.00, p 0.95)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L25 down c5 | MLP | b%100 in {1..3, 21, 23, 31, 51, 61, 73, 83, 93} (+) | 0.63 | 0.132 |
| L25 down c1 | MLP | b%100 in {1} (-) | 0.77 | 0.015 |
| L25 down c378 | MLP | b%100 in {2..4, 6} (+) | 0.82 | 0.074 |

</details>

</details>

<details><summary><b>3a-318</b> `b%100` @ `b` (add) — block code; 5 comps, L28 L29; tells apart 20/100 classes (best member 6); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.14 |
| classes told apart: joint / best code / best member | 20 /  / 6 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.07 / 1.05 / 0.60 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.61 |
| decoding acc. joint / best code / best member (chance) | 0.13 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.94 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L28 attn |
| CKA(arrangement before, joint write) | 0.33 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.23 2:0.12 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (3:0.09 1:0.09 40:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.07 / 2.92 |

<details><summary>codes and components</summary>

**code 3a-318.0 (L28 L29): 5 comps, tells apart 20/100, coverage 0.14, overlap 1.07 (random 1.07, p 0.54)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 down c409 | MLP | b%100 in {2..4, 6} (+) | 0.62 | 0.046 |
| L29 down c2 | MLP | b%100 in {0} (+) | 0.59 | 0.994 |
| L29 down c9 | MLP | b%100 in {0} (-) | 0.54 | 1.000 |
| L29 down c107 | MLP | b%100 in {1, 21, 31, 41, 61, 71, 91} (-) | 0.80 | 0.075 |
| L29 down c212 | MLP | b%100 in {32, 64} (+) | 0.69 | 0.042 |

</details>

</details>

</details>

<details><summary>`tens(a,b)`: 15 mechanisms, 219 components</summary>

<details><summary><b>3a-335</b> `tens(a,b)` @ `b` (add) — block code, 2 codes of the same shape; 4 comps, L0 L1 L4; tells apart 17/121 classes (best member 7); on sub: 3s-413 (member overlap 0.20)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.25 |
| classes told apart: joint / best code / best member | 17 / 10 / 7 (of 121) |
| members whose removal merges classes | 0.25 |
| support overlap (1 = tiling) / random sets / p | 1.50 / 1.18 / 0.93 |
| mean CKA between its codes | 0.80 |
| purity of the joint write (per prompt) | 0.70 |
| decoding acc. joint / best code / best member (chance) | 0.10 / 0.10 / 0.08 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.35 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | embed |
| CKA(arrangement before, joint write) | 0.04 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.00 |
| write energy / code energy before | 0.05 |
| best source position (CKA) | `a` (0.32) |

<details><summary>codes and components</summary>

**code 3a-335.0 (L0 L1): 2 comps, tells apart 9/121, coverage 0.23, overlap 1.07 (random 1.00, p 0.76)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 o c7 | H0 | a//10 {0} x b//10 {0..5, 7}; a//10 {1} x b//10 {0..2}; a//10 {2} x b//10 {0..1}; a//10 {3..4} x b//10 {0}; a//10 {10} x b//10 {5..10} (-) | 0.87 | 1.000 |
| L1 o c223 | H24 | a//10 {8, 10} x b//10 {6..7}; a//10 {9} x b//10 {4..9} (+) | 0.83 | 0.107 |

**code 3a-335.1 (L4): 2 comps, tells apart 10/121, coverage 0.07, overlap 1.67 (random 1.00, p 0.98)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 o c37 | H3 | a//10 {9} x b//10 {3..8}; a//10 {10} x b//10 {4, 6} (+) | 0.61 | 0.101 |
| L4 o c59 | H3 | a//10 {9} x b//10 {3..9} (+) | 0.68 | 0.100 |

</details>

</details>

<details><summary><b>3a-336</b> `tens(a,b)` @ `b` (add) — block code; 2 comps, L1 L2; tells apart 2/121 classes (best member 3); on sub: 3s-415 (member overlap 0.14)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.18 |
| classes told apart: joint / best code / best member | 2 /  / 3 (of 121) |
| members whose removal merges classes | 0.50 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.65 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.74 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 5 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.08 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.06 |
| best source position (CKA) | `a` (0.05) |

<details><summary>codes and components</summary>

**code 3a-336.0 (L1 L2): 2 comps, tells apart 2/121, coverage 0.18, overlap 1.00 (random 1.00, p 0.66)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c403 | H26 | a//10 {1} x b//10 {0} (-) | 0.56 | 0.014 |
| L2 o c5 | H13 | a//10 {1} x b//10 {1}; a//10 {2} x b//10 {2..3}; a//10 {3} x b//10 {3..4}; a//10 {4} x b//10 {4..5}; a//10 {5} x b//10 {5..6}; a//10 {6} x b//10 {6..8}; a//10 {7} x b//10 {7..9}; a//10 {8} x b//10 {8..10}; a//10 {9} x b//10 {9..10}; a//10 {10} x b//10 {10} (-) | 0.74 | 0.302 |

</details>

</details>

<details><summary><b>3a-337</b> `tens(a,b)` @ `b` (add) — tiling, 15 codes of the same shape; 127 comps, L1 L3 L4 L5 L6 L7 L8 L9 L10 L11 L12 L13 L14 L15 L17; tells apart 67/121 classes (best member 9); on sub: 3s-417 (member overlap 0.06)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 67 / 67 / 9 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p | 14.61 / 13.10 / 0.99 |
| mean CKA between its codes | 0.79 |
| purity of the joint write (per prompt) | 0.73 |
| decoding acc. joint / best code / best member (chance) | 0.50 / 0.56 / 0.07 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.80 |
| consumers / read jointly | 438 / 302 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L1 attn |
| CKA(arrangement before, joint write) | 0.62 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.02 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 21.07 |
| best source position (CKA) | `op` (0.39) |

<details><summary>codes and components</summary>

**code 3a-337.0 (L1): 15 comps, tells apart 34/121, coverage 0.97, overlap 2.03 (random 2.12, p 0.41)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 down c53 | MLP | a//10 {0..3} x b//10 {7..10}; a//10 {4} x b//10 {8..10}; a//10 {5} x b//10 {9..10} (-) | 0.93 | 0.336 |
| L1 down c257 | MLP | a//10 {0} x b//10 {0..1}; a//10 {1} x b//10 {1} (-) | 0.58 | 0.035 |
| L1 down c535 | MLP | a//10 {0} x b//10 {0..2}; a//10 {1} x b//10 {1} (-) | 0.91 | 0.057 |
| L1 down c752 | MLP | a//10 {0} x b//10 {0} (+) | 0.71 | 0.015 |
| L1 down c91 | MLP | a//10 {0} x b//10 {1..2} (+) | 0.68 | 0.033 |
| L1 down c26 | MLP | a//10 {0} x b//10 {2..10}; a//10 {1} x b//10 {0, 9..10}; a//10 {2} x b//10 {0..3, 10}; a//10 {3} x b//10 {1..6}; a//10 {4..5} x b//10 {1..8}; a//10 {6..7} x b//10 {6..9}; a//10 {8} x b//10 {0, 8}; a//10 {9} x b//10 {0, 2..4}; a//10 {10} x b//10 {0..5} (-) | 0.86 | 0.768 |
| L1 down c42 | MLP | a//10 {0} x b//10 {3..5}; a//10 {1} x b//10 {2..6}; a//10 {2} x b//10 {4..5} (+) | 0.82 | 0.192 |
| L1 down c13 | MLP | a//10 {0} x b//10 {4..8}; a//10 {1} x b//10 {0..1}; a//10 {2, 5} x b//10 {0, 2}; a//10 {3} x b//10 {0, 2..3}; a//10 {9} x b//10 {5..6}; a//10 {10} x b//10 {1..8} (+) | 0.55 | 1.000 |
| L1 down c51 | MLP | a//10 {0} x b//10 {4..9}; a//10 {1..2} x b//10 {8..9}; a//10 {3..10} x b//10 {1} (+) | 0.78 | 1.000 |
| L1 down c100 | MLP | a//10 {1} x b//10 {0, 5..6}; a//10 {8} x b//10 {5..6}; a//10 {9} x b//10 {5..7}; a//10 {10} x b//10 {6} (+) | 0.67 | 0.152 |
| L1 down c265 | MLP | a//10 {1} x b//10 {0}; a//10 {2..3} x b//10 {0..1} (-) | 0.64 | 0.063 |
| L1 down c12 | MLP | a//10 {2} x b//10 {0}; a//10 {3} x b//10 {0..1, 6, 8}; a//10 {4..6} x b//10 {0..1, 6..8, 10}; a//10 {7} x b//10 {0..2, 7..10}; a//10 {8} x b//10 {0..2, 8..10}; a//10 {9} x b//10 {0..2, 9..10}; a//10 {10} x b//10 {0..1, 10} (+) | 0.96 | 0.703 |
| L1 down c57 | MLP | a//10 {2} x b//10 {1..2} (-) | 0.67 | 0.044 |
| L1 down c28 | MLP | a//10 {4} x b//10 {2}; a//10 {5} x b//10 {1..3}; a//10 {6} x b//10 {1..4}; a//10 {7} x b//10 {1..5}; a//10 {8} x b//10 {1..6}; a//10 {9} x b//10 {1..7}; a//10 {10} x b//10 {2..6} (+) | 0.93 | 0.406 |
| L1 down c113 | MLP | a//10 {9..10} x b//10 {9..10} (+) | 0.59 | 0.031 |

**code 3a-337.1 (L3): 10 comps, tells apart 23/121, coverage 0.64, overlap 1.58 (random 1.66, p 0.34)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 down c108 | MLP | a//10 {0..1, 5} x b//10 {0..1}; a//10 {2..4} x b//10 {0..2} (+) | 0.91 | 0.264 |
| L3 down c7 | MLP | a//10 {0} x b//10 {0..6, 8..10}; a//10 {1} x b//10 {0, 10}; a//10 {2, 7, 9} x b//10 {0}; a//10 {3..4, 8} x b//10 {0, 7}; a//10 {5..6} x b//10 {7}; a//10 {10} x b//10 {0..1} (-) | 0.70 | 1.000 |
| L3 down c259 | MLP | a//10 {0} x b//10 {0} (-) | 0.63 | 0.009 |
| L3 down c85 | MLP | a//10 {0} x b//10 {1..10}; a//10 {1} x b//10 {3..10}; a//10 {2} x b//10 {4..10}; a//10 {3} x b//10 {7..10}; a//10 {4} x b//10 {9} (-) | 0.96 | 0.444 |
| L3 down c35 | MLP | a//10 {0} x b//10 {3..9} (-) | 0.67 | 0.056 |
| L3 down c78 | MLP | a//10 {1..6, 10} x b//10 {0}; a//10 {7..9} x b//10 {0..1} (-) | 0.95 | 0.288 |
| L3 down c539 | MLP | a//10 {3..10} x b//10 {0} (-) | 0.83 | 0.092 |
| L3 down c24 | MLP | a//10 {5, 9} x b//10 {3}; a//10 {6} x b//10 {2..3}; a//10 {7..8} x b//10 {2..5}; a//10 {10} x b//10 {3, 6} (+) | 0.69 | 0.233 |
| L3 down c39 | MLP | a//10 {8..9} x b//10 {9} (-) | 0.56 | 0.043 |
| L3 down c819 | MLP | a//10 {8} x b//10 {4}; a//10 {9} x b//10 {3..6}; a//10 {10} x b//10 {4, 6} (-) | 0.72 | 0.116 |

**code 3a-337.2 (L4 L5): 8 comps, tells apart 28/121, coverage 0.44, overlap 1.51 (random 1.53, p 0.47)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 down c13 | MLP | a//10 {0} x b//10 {0, 8..10}; a//10 {1..5, 7..9} x b//10 {0}; a//10 {10} x b//10 {0..1, 10} (-) | 0.67 | 0.998 |
| L4 down c121 | MLP | a//10 {0} x b//10 {0..2}; a//10 {1} x b//10 {1..2} (+) | 0.81 | 0.080 |
| L4 down c646 | MLP | a//10 {1, 4..7} x b//10 {1}; a//10 {2} x b//10 {0..1}; a//10 {3} x b//10 {0..2} (-) | 0.83 | 0.194 |
| L4 down c34 | MLP | a//10 {3..10} x b//10 {0} (+) | 0.87 | 0.083 |
| L4 down c70 | MLP | a//10 {3} x b//10 {0..1}; a//10 {4..5} x b//10 {0..2}; a//10 {6..9} x b//10 {0..3}; a//10 {10} x b//10 {1..3} (-) | 0.95 | 0.489 |
| L4 down c73 | MLP | a//10 {6} x b//10 {6..7}; a//10 {7} x b//10 {7..8}; a//10 {8} x b//10 {8..9}; a//10 {9} x b//10 {9} (+) | 0.52 | 0.074 |
| L4 down c140 | MLP | a//10 {8..9} x b//10 {9} (+) | 0.57 | 0.042 |
| L5 o c111 | H22 | a//10 {9..10} x b//10 {7..9} (-) | 0.64 | 0.035 |

**code 3a-337.3 (L5): 23 comps, tells apart 67/121, coverage 0.98, overlap 2.03 (random 2.75, p 0.01)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 down c440 | MLP | a//10 {0..1} x b//10 {1..2}; a//10 {2} x b//10 {2} (-) | 0.54 | 0.042 |
| L5 down c5 | MLP | a//10 {0} x b//10 {0..2, 10}; a//10 {1} x b//10 {1, 10}; a//10 {2, 5..10} x b//10 {10}; a//10 {3} x b//10 {3, 10} (+) | 0.52 | 0.277 |
| L5 down c42 | MLP | a//10 {0} x b//10 {2..10}; a//10 {1} x b//10 {3..10}; a//10 {2} x b//10 {5..9} (+) | 0.92 | 0.377 |
| L5 down c252 | MLP | a//10 {1, 10} x b//10 {2..5} (+) | 0.73 | 0.099 |
| L5 down c165 | MLP | a//10 {1, 5..8} x b//10 {0}; a//10 {2..4} x b//10 {0..1} (+) | 0.91 | 0.185 |
| L5 down c530 | MLP | a//10 {1..5} x b//10 {0} (-) | 0.69 | 0.053 |
| L5 down c127 | MLP | a//10 {2, 7} x b//10 {9}; a//10 {3, 5..6} x b//10 {8..9}; a//10 {4} x b//10 {7..9} (-) | 0.89 | 0.196 |
| L5 down c81 | MLP | a//10 {2..3} x b//10 {5..8}; a//10 {4} x b//10 {6..8} (-) | 0.79 | 0.238 |
| L5 down c97 | MLP | a//10 {2} x b//10 {4..6}; a//10 {3} x b//10 {4..7}; a//10 {4} x b//10 {5..6} (+) | 0.69 | 0.159 |
| L5 down c70 | MLP | a//10 {3..10} x b//10 {0} (-) | 0.92 | 0.081 |
| L5 down c96 | MLP | a//10 {3..4} x b//10 {1}; a//10 {5, 9..10} x b//10 {1..2}; a//10 {6..8} x b//10 {0..3} (-) | 0.95 | 0.492 |
| L5 down c99 | MLP | a//10 {3} x b//10 {2..3}; a//10 {4} x b//10 {2..4}; a//10 {5} x b//10 {3..4} (+) | 0.72 | 0.101 |
| L5 down c50 | MLP | a//10 {4..5} x b//10 {6..8} (+) | 0.70 | 0.075 |
| L5 down c106 | MLP | a//10 {4} x b//10 {2}; a//10 {5, 8} x b//10 {1..2}; a//10 {6..7} x b//10 {1..3} (+) | 0.76 | 0.217 |
| L5 down c40 | MLP | a//10 {5} x b//10 {4}; a//10 {6, 9} x b//10 {4..8}; a//10 {7..8} x b//10 {3..8}; a//10 {10} x b//10 {3..9} (+) | 0.87 | 0.353 |
| L5 down c163 | MLP | a//10 {5} x b//10 {5}; a//10 {6} x b//10 {6..7}; a//10 {7} x b//10 {7..8}; a//10 {8} x b//10 {7..9}; a//10 {9} x b//10 {8..9}; a//10 {10} x b//10 {10} (-) | 0.74 | 0.190 |
| L5 down c121 | MLP | a//10 {6} x b//10 {4..5}; a//10 {7} x b//10 {4..6}; a//10 {8} x b//10 {4..7}; a//10 {10} x b//10 {5, 7..8} (-) | 0.73 | 0.188 |
| L5 down c23 | MLP | a//10 {6} x b//10 {7}; a//10 {7} x b//10 {7..8}; a//10 {8} x b//10 {8..9}; a//10 {9..10} x b//10 {9} (-) | 0.66 | 0.133 |
| L5 down c89 | MLP | a//10 {7} x b//10 {8..9}; a//10 {8} x b//10 {9} (-) | 0.64 | 0.047 |
| L5 down c53 | MLP | a//10 {8, 10} x b//10 {3..6}; a//10 {9} x b//10 {2..7} (-) | 0.89 | 0.156 |
| L5 down c368 | MLP | a//10 {8, 10} x b//10 {4}; a//10 {9} x b//10 {3..5} (+) | 0.50 | 0.112 |
| L5 down c109 | MLP | a//10 {8..9} x b//10 {9} (+) | 0.58 | 0.036 |
| L5 down c95 | MLP | a//10 {8} x b//10 {7}; a//10 {9} x b//10 {6..9}; a//10 {10} x b//10 {7..9} (+) | 0.78 | 0.076 |

**code 3a-337.4 (L6): 8 comps, tells apart 36/121, coverage 0.66, overlap 1.29 (random 1.49, p 0.12)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L6 down c49 | MLP | a//10 {0} x b//10 {0, 4, 8..10}; a//10 {1..4, 9} x b//10 {0}; a//10 {10} x b//10 {0..1, 10} (+) | 0.61 | 0.989 |
| L6 down c162 | MLP | a//10 {0} x b//10 {1..10}; a//10 {1} x b//10 {3..10}; a//10 {2} x b//10 {6..10} (-) | 0.91 | 0.305 |
| L6 down c85 | MLP | a//10 {1} x b//10 {1..2}; a//10 {2} x b//10 {2} (+) | 0.58 | 0.035 |
| L6 down c300 | MLP | a//10 {2..10} x b//10 {0} (-) | 0.89 | 0.127 |
| L6 down c127 | MLP | a//10 {3} x b//10 {1}; a//10 {4} x b//10 {1..2}; a//10 {5} x b//10 {0..2}; a//10 {6..8} x b//10 {0..3}; a//10 {9..10} x b//10 {1..3} (-) | 0.94 | 0.409 |
| L6 down c153 | MLP | a//10 {6} x b//10 {6}; a//10 {7} x b//10 {7}; a//10 {8..9} x b//10 {8..9} (-) | 0.66 | 0.109 |
| L6 down c123 | MLP | a//10 {7} x b//10 {4..5}; a//10 {8} x b//10 {4..7}; a//10 {9} x b//10 {2..9}; a//10 {10} x b//10 {4..9} (-) | 0.87 | 0.237 |
| L6 down c57 | MLP | a//10 {8, 10} x b//10 {4}; a//10 {9} x b//10 {3..5} (-) | 0.62 | 0.082 |

**code 3a-337.5 (L7): 8 comps, tells apart 32/121, coverage 0.61, overlap 1.41 (random 1.50, p 0.32)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L7 down c309 | MLP | a//10 {0, 2} x b//10 {0}; a//10 {1} x b//10 {0..1}; a//10 {10} x b//10 {10} (+) | 0.61 | 0.059 |
| L7 down c31 | MLP | a//10 {0} x b//10 {0, 4, 10}; a//10 {1..3, 5..8} x b//10 {0}; a//10 {4} x b//10 {0, 6..7}; a//10 {9} x b//10 {0, 4}; a//10 {10} x b//10 {0, 10} (+) | 0.64 | 0.973 |
| L7 down c397 | MLP | a//10 {0} x b//10 {1..9}; a//10 {1} x b//10 {2..10}; a//10 {2} x b//10 {5..10} (-) | 0.77 | 0.209 |
| L7 down c187 | MLP | a//10 {0} x b//10 {2..9}; a//10 {1} x b//10 {5..6} (+) | 0.76 | 0.217 |
| L7 down c48 | MLP | a//10 {1} x b//10 {1..2}; a//10 {2} x b//10 {2} (+) | 0.61 | 0.095 |
| L7 down c29 | MLP | a//10 {3} x b//10 {1}; a//10 {4..5} x b//10 {0..2}; a//10 {6..10} x b//10 {0..3} (+) | 0.92 | 0.484 |
| L7 down c195 | MLP | a//10 {7} x b//10 {1}; a//10 {8} x b//10 {0..1}; a//10 {9} x b//10 {0..2} (-) | 0.56 | 0.110 |
| L7 down c179 | MLP | a//10 {7} x b//10 {6}; a//10 {8} x b//10 {6..8}; a//10 {9} x b//10 {5..9}; a//10 {10} x b//10 {7..9} (+) | 0.78 | 0.165 |

**code 3a-337.6 (L8): 8 comps, tells apart 40/121, coverage 0.58, overlap 1.66 (random 1.52, p 0.78)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L8 down c0 | MLP | a//10 {0} x b//10 {0, 10}; a//10 {1..2, 4, 8} x b//10 {0}; a//10 {3} x b//10 {0, 6..7}; a//10 {9} x b//10 {0, 9}; a//10 {10} x b//10 {0, 4, 10} (-) | 0.56 | 1.000 |
| L8 down c69 | MLP | a//10 {0} x b//10 {0, 3..4, 9..10}; a//10 {1} x b//10 {0, 10}; a//10 {2..9} x b//10 {0}; a//10 {10} x b//10 {0..2, 10} (+) | 0.73 | 0.940 |
| L8 down c248 | MLP | a//10 {1..2} x b//10 {0}; a//10 {3..4} x b//10 {0..1}; a//10 {5..9} x b//10 {0..2}; a//10 {10} x b//10 {0..3} (+) | 0.91 | 0.357 |
| L8 down c119 | MLP | a//10 {3..6, 8..9} x b//10 {0} (+) | 0.61 | 0.093 |
| L8 down c138 | MLP | a//10 {4} x b//10 {3}; a//10 {5} x b//10 {4}; a//10 {6} x b//10 {3..5}; a//10 {7} x b//10 {3..6}; a//10 {8} x b//10 {3..7}; a//10 {9} x b//10 {4..8}; a//10 {10} x b//10 {3..9} (-) | 0.84 | 0.398 |
| L8 down c363 | MLP | a//10 {6} x b//10 {6..8}; a//10 {7..8} x b//10 {7..9}; a//10 {9} x b//10 {7..10}; a//10 {10} x b//10 {8..9} (-) | 0.77 | 0.240 |
| L8 down c871 | MLP | a//10 {9} x b//10 {1..7}; a//10 {10} x b//10 {4, 6} (-) | 0.62 | 0.182 |
| L8 down c65 | MLP | a//10 {9} x b//10 {8..9} (+) | 0.57 | 0.025 |

**code 3a-337.7 (L9): 6 comps, tells apart 29/121, coverage 0.82, overlap 1.31 (random 1.37, p 0.39)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L9 down c3 | MLP | a//10 {0} x b//10 {0, 10}; a//10 {3..4} x b//10 {1..2}; a//10 {5} x b//10 {1..3}; a//10 {6..7} x b//10 {1..5}; a//10 {8} x b//10 {1..7}; a//10 {9} x b//10 {2, 5, 9}; a//10 {10} x b//10 {1..5, 7, 10} (+) | 0.58 | 0.965 |
| L9 down c22 | MLP | a//10 {0} x b//10 {0, 4, 10}; a//10 {1, 3, 5..6, 8} x b//10 {0}; a//10 {9} x b//10 {0, 4}; a//10 {10} x b//10 {0, 3..4} (+) | 0.55 | 0.999 |
| L9 down c208 | MLP | a//10 {0} x b//10 {1..10}; a//10 {1} x b//10 {2..10}; a//10 {2} x b//10 {4..9}; a//10 {3} x b//10 {8..9}; a//10 {4} x b//10 {9} (+) | 0.88 | 0.477 |
| L9 down c340 | MLP | a//10 {2..3} x b//10 {0}; a//10 {4..9} x b//10 {0..1}; a//10 {10} x b//10 {0..2} (+) | 0.89 | 0.240 |
| L9 down c57 | MLP | a//10 {2} x b//10 {5..8}; a//10 {3..4} x b//10 {5..9}; a//10 {5} x b//10 {6..9}; a//10 {6..7} x b//10 {8..9} (-) | 0.65 | 0.582 |
| L9 down c778 | MLP | a//10 {6} x b//10 {6..8}; a//10 {7..8} x b//10 {7..9}; a//10 {9} x b//10 {7..10}; a//10 {10} x b//10 {9} (+) | 0.71 | 0.178 |

**code 3a-337.8 (L10): 6 comps, tells apart 22/121, coverage 0.73, overlap 1.41 (random 1.35, p 0.67)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L10 down c99 | MLP | a//10 {0, 5} x b//10 {0..1, 10}; a//10 {1..4} x b//10 {0, 10}; a//10 {6..8} x b//10 {0..1}; a//10 {10} x b//10 {0..2, 10} (-) | 0.72 | 0.487 |
| L10 down c13 | MLP | a//10 {0} x b//10 {0, 4, 10}; a//10 {1..2, 8} x b//10 {0}; a//10 {3} x b//10 {0, 6..7, 9}; a//10 {4} x b//10 {6..7, 9}; a//10 {9} x b//10 {0, 9}; a//10 {10} x b//10 {0, 10} (-) | 0.55 | 0.999 |
| L10 down c71 | MLP | a//10 {0} x b//10 {2..9}; a//10 {1} x b//10 {4..9}; a//10 {2} x b//10 {7..8} (+) | 0.66 | 0.145 |
| L10 down c295 | MLP | a//10 {3, 9} x b//10 {1}; a//10 {4..5} x b//10 {0..1}; a//10 {6..8} x b//10 {0..2}; a//10 {10} x b//10 {0..3} (-) | 0.85 | 0.239 |
| L10 down c257 | MLP | a//10 {5} x b//10 {2}; a//10 {6} x b//10 {1..4}; a//10 {7} x b//10 {1..5}; a//10 {8} x b//10 {1..7}; a//10 {9} x b//10 {1..3, 5..8}; a//10 {10} x b//10 {1..3, 5..9} (-) | 0.79 | 0.417 |
| L10 down c80 | MLP | a//10 {5} x b//10 {5..6}; a//10 {6} x b//10 {6..8}; a//10 {7..8} x b//10 {7..9}; a//10 {9} x b//10 {7..10}; a//10 {10} x b//10 {9} (-) | 0.74 | 0.207 |

**code 3a-337.9 (L11): 5 comps, tells apart 12/121, coverage 0.55, overlap 1.03 (random 1.26, p 0.03)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L11 down c168 | MLP | a//10 {0..2} x b//10 {0}; a//10 {10} x b//10 {10} (-) | 0.55 | 0.037 |
| L11 down c146 | MLP | a//10 {0} x b//10 {6..8}; a//10 {1} x b//10 {5..9}; a//10 {2} x b//10 {4..9}; a//10 {3..4} x b//10 {6..9}; a//10 {5} x b//10 {8} (-) | 0.69 | 0.283 |
| L11 down c297 | MLP | a//10 {2} x b//10 {0}; a//10 {3} x b//10 {1}; a//10 {4..7} x b//10 {0..1}; a//10 {8..9} x b//10 {0..2}; a//10 {10} x b//10 {0..3, 5} (+) | 0.84 | 0.263 |
| L11 down c799 | MLP | a//10 {6} x b//10 {6..7}; a//10 {7} x b//10 {6..9}; a//10 {8..10} x b//10 {7..9} (-) | 0.77 | 0.203 |
| L11 down c67 | MLP | a//10 {9} x b//10 {2, 4..6}; a//10 {10} x b//10 {4} (-) | 0.50 | 0.053 |

**code 3a-337.10 (L12): 6 comps, tells apart 10/121, coverage 0.53, overlap 1.14 (random 1.38, p 0.10)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L12 down c41 | MLP | a//10 {0} x b//10 {2..9}; a//10 {1} x b//10 {4..9}; a//10 {2} x b//10 {7, 9} (+) | 0.70 | 0.380 |
| L12 down c68 | MLP | a//10 {3} x b//10 {0..1}; a//10 {4..5} x b//10 {0..2}; a//10 {6, 9} x b//10 {0..3}; a//10 {7..8} x b//10 {0..4}; a//10 {10} x b//10 {0..5} (+) | 0.90 | 0.431 |
| L12 down c10 | MLP | a//10 {6..8} x b//10 {0..1}; a//10 {10} x b//10 {1} (-) | 0.76 | 0.113 |
| L12 down c627 | MLP | a//10 {6} x b//10 {7..8}; a//10 {7..8} x b//10 {8..9}; a//10 {9} x b//10 {9} (-) | 0.52 | 0.107 |
| L12 down c493 | MLP | a//10 {8} x b//10 {7}; a//10 {9} x b//10 {5..9}; a//10 {10} x b//10 {6..9} (-) | 0.71 | 0.126 |
| L12 down c242 | MLP | a//10 {9} x b//10 {9} (+) | 0.53 | 0.013 |

**code 3a-337.11 (L13): 8 comps, tells apart 31/121, coverage 0.78, overlap 1.53 (random 1.51, p 0.55)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 down c12 | MLP | a//10 {0} x b//10 {0, 3..4, 9..10}; a//10 {1} x b//10 {0, 10}; a//10 {2..3} x b//10 {0, 6..8, 10}; a//10 {4..5} x b//10 {0, 6..8}; a//10 {6} x b//10 {0, 5, 8}; a//10 {7} x b//10 {0, 5..6}; a//10 {8} x b//10 {0, 6}; a//10 {9} x b//10 {0, 9..10}; a//10 {10} x b//10 {0..5, 10} (+) | 0.60 | 0.875 |
| L13 down c95 | MLP | a//10 {0} x b//10 {1..10}; a//10 {1} x b//10 {2..10}; a//10 {2} x b//10 {4..9} (+) | 0.86 | 0.300 |
| L13 down c39 | MLP | a//10 {1} x b//10 {0, 10}; a//10 {2} x b//10 {0}; a//10 {3..8} x b//10 {0..1}; a//10 {10} x b//10 {0..3} (+) | 0.71 | 0.216 |
| L13 down c61 | MLP | a//10 {2, 4} x b//10 {6..9}; a//10 {3, 5} x b//10 {7..9} (-) | 0.63 | 0.172 |
| L13 down c635 | MLP | a//10 {2} x b//10 {2}; a//10 {3} x b//10 {3}; a//10 {4} x b//10 {4}; a//10 {5} x b//10 {5}; a//10 {6} x b//10 {6..7}; a//10 {7} x b//10 {7..8}; a//10 {8} x b//10 {8..9}; a//10 {9} x b//10 {9..10}; a//10 {10} x b//10 {10} (-) | 0.51 | 0.151 |
| L13 down c475 | MLP | a//10 {4} x b//10 {1}; a//10 {5..6, 9..10} x b//10 {1..2}; a//10 {7} x b//10 {0..2}; a//10 {8} x b//10 {0..3} (+) | 0.86 | 0.249 |
| L13 down c571 | MLP | a//10 {6..8} x b//10 {0} (+) | 0.62 | 0.042 |
| L13 down c264 | MLP | a//10 {6} x b//10 {5}; a//10 {7} x b//10 {6}; a//10 {8} x b//10 {6..7}; a//10 {9} x b//10 {6..8}; a//10 {10} x b//10 {6..9} (-) | 0.63 | 0.139 |

**code 3a-337.12 (L14): 3 comps, tells apart 3/121, coverage 0.42, overlap 1.00 (random 1.11, p 0.24)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L14 down c47 | MLP | a//10 {0} x b//10 {2..9}; a//10 {1} x b//10 {4..9}; a//10 {2} x b//10 {6..7, 9} (+) | 0.79 | 0.205 |
| L14 down c954 | MLP | a//10 {3} x b//10 {1}; a//10 {4, 9} x b//10 {1..2}; a//10 {5..6} x b//10 {0..2}; a//10 {7..8, 10} x b//10 {0..3} (+) | 0.87 | 0.362 |
| L14 down c73 | MLP | a//10 {7} x b//10 {8..9}; a//10 {8..10} x b//10 {7..9} (+) | 0.73 | 0.092 |

**code 3a-337.13 (L15): 6 comps, tells apart 27/121, coverage 0.56, overlap 1.54 (random 1.37, p 0.77)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c103 | MLP | a//10 {0..1, 5} x b//10 {8..9}; a//10 {2} x b//10 {5, 7..9}; a//10 {3..4} x b//10 {7..9}; a//10 {6..7} x b//10 {9} (-) | 0.73 | 0.294 |
| L15 down c24 | MLP | a//10 {0} x b//10 {0, 3..5, 9..10}; a//10 {1..6, 8..9} x b//10 {0}; a//10 {7} x b//10 {0, 6}; a//10 {10} x b//10 {0..1, 4} (+) | 0.61 | 0.995 |
| L15 down c71 | MLP | a//10 {0} x b//10 {2..9}; a//10 {1} x b//10 {6}; a//10 {4..5} x b//10 {0} (+) | 0.63 | 0.151 |
| L15 down c57 | MLP | a//10 {2} x b//10 {0, 7..8}; a//10 {3} x b//10 {0..1, 7..8}; a//10 {4..5, 7} x b//10 {0..1, 6..8}; a//10 {6} x b//10 {0..2, 6..8}; a//10 {8} x b//10 {0..2, 8}; a//10 {9} x b//10 {0..2}; a//10 {10} x b//10 {0..5} (-) | 0.77 | 0.715 |
| L15 down c111 | MLP | a//10 {6..8} x b//10 {0..1}; a//10 {10} x b//10 {0..2} (+) | 0.64 | 0.075 |
| L15 down c851 | MLP | a//10 {9} x b//10 {3..7}; a//10 {10} x b//10 {4, 6} (-) | 0.57 | 0.548 |

**code 3a-337.14 (L17): 7 comps, tells apart 18/121, coverage 0.50, overlap 1.17 (random 1.42, p 0.07)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 down c58 | MLP | a//10 {0..1} x b//10 {0} (-) | 0.60 | 0.019 |
| L17 down c42 | MLP | a//10 {0..1} x b//10 {8..9}; a//10 {2} x b//10 {5..9}; a//10 {3} x b//10 {5, 7..9}; a//10 {4..5} x b//10 {7..9}; a//10 {6..8} x b//10 {9} (-) | 0.71 | 0.328 |
| L17 down c176 | MLP | a//10 {0..2} x b//10 {1..2} (+) | 0.70 | 0.098 |
| L17 down c145 | MLP | a//10 {0} x b//10 {2..9}; a//10 {1} x b//10 {5..9} (+) | 0.62 | 0.205 |
| L17 down c81 | MLP | a//10 {3, 5} x b//10 {1}; a//10 {4} x b//10 {0..1}; a//10 {6..10} x b//10 {0..2} (+) | 0.80 | 0.262 |
| L17 down c100 | MLP | a//10 {4..9} x b//10 {0} (+) | 0.68 | 0.050 |
| L17 down c242 | MLP | a//10 {9} x b//10 {8..9} (+) | 0.76 | 0.190 |

</details>

</details>

<details><summary><b>3a-338</b> `tens(a,b)` @ `b` (add) — block code, copy from `op`; 8 comps, L2 L3; tells apart 22/121 classes (best member 7); on sub: 3s-417 (member overlap 0.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.74 |
| classes told apart: joint / best code / best member | 22 /  / 7 (of 121) |
| members whose removal merges classes | 0.88 |
| support overlap (1 = tiling) / random sets / p | 1.25 / 1.49 / 0.10 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.84 |
| decoding acc. joint / best code / best member (chance) | 0.23 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 3 / 3 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 attn |
| CKA(arrangement before, joint write) | 0.59 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.05 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.12 |
| best source position (CKA) | `op` (0.60) |

<details><summary>codes and components</summary>

**code 3a-338.0 (L2 L3): 8 comps, tells apart 22/121, coverage 0.74, overlap 1.25 (random 1.53, p 0.07)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 down c697 | MLP | a//10 {0} x b//10 {0} (+) | 0.73 | 0.008 |
| L2 down c40 | MLP | a//10 {0} x b//10 {1..10}; a//10 {1} x b//10 {2..10}; a//10 {2} x b//10 {4..10}; a//10 {3} x b//10 {7..10}; a//10 {4} x b//10 {9..10} (+) | 0.95 | 0.487 |
| L2 down c52 | MLP | a//10 {0} x b//10 {2..9} (-) | 0.88 | 0.084 |
| L2 down c105 | MLP | a//10 {1..2} x b//10 {2} (-) | 0.57 | 0.045 |
| L2 down c50 | MLP | a//10 {1} x b//10 {1..2}; a//10 {2} x b//10 {2..3}; a//10 {3} x b//10 {3..4, 6, 10}; a//10 {4} x b//10 {4..10}; a//10 {5} x b//10 {5..10}; a//10 {6} x b//10 {6..10}; a//10 {7} x b//10 {7..10}; a//10 {8} x b//10 {8..10}; a//10 {9} x b//10 {9..10}; a//10 {10} x b//10 {10} (+) | 0.81 | 0.394 |
| L2 down c666 | MLP | a//10 {3} x b//10 {3..4}; a//10 {4} x b//10 {4}; a//10 {5} x b//10 {5}; a//10 {10} x b//10 {10} (+) | 0.51 | 0.110 |
| L2 down c23 | MLP | a//10 {8..9} x b//10 {9} (-) | 0.64 | 0.023 |
| L3 o c105 | H18 | a//10 {2} x b//10 {0}; a//10 {3, 5..6, 8} x b//10 {0..2}; a//10 {4} x b//10 {0..2, 7}; a//10 {7} x b//10 {0..1}; a//10 {9} x b//10 {1..2}; a//10 {10} x b//10 {0..3} (+) | 0.56 | 0.477 |

</details>

</details>

<details><summary><b>3a-340</b> `tens(a,b)` @ `b` (add) — single component; 1 comps, L3; tells apart 5/121 classes (best member 5); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.11 |
| classes told apart: joint / best code / best member | 5 /  / 5 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.51 |
| decoding acc. joint / best code / best member (chance) | 0.02 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 MLP |
| CKA(arrangement before, joint write) | 0.37 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.00 |
| best source position (CKA) | `op` (0.08) |

<details><summary>codes and components</summary>

**code 3a-340.0 (L3): 1 comps, tells apart 5/121, coverage 0.11, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 o c154 | H27 | a//10 {2..8} x b//10 {0}; a//10 {9} x b//10 {0, 4..6}; a//10 {10} x b//10 {0, 4} (+) | 0.50 | 0.110 |

</details>

</details>

<details><summary><b>3a-339</b> `tens(a,b)` @ `b` (add) — block code, copy from `op`; 2 comps, L3; tells apart 14/121 classes (best member 7); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.29 |
| classes told apart: joint / best code / best member | 14 /  / 7 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.63 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.68 |
| decoding acc. joint / best code / best member (chance) | 0.07 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 MLP |
| CKA(arrangement before, joint write) | 0.25 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.01 |
| best source position (CKA) | `op` (0.63) |

<details><summary>codes and components</summary>

**code 3a-339.0 (L3): 2 comps, tells apart 14/121, coverage 0.29, overlap 1.00 (random 1.00, p 0.62)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 o c128 | H5 | a//10 {0} x b//10 {0..5, 8..10}; a//10 {1} x b//10 {0..1, 4}; a//10 {3..4} x b//10 {4}; a//10 {6} x b//10 {0..2}; a//10 {7} x b//10 {0..3}; a//10 {8} x b//10 {1..2, 6..7}; a//10 {10} x b//10 {1..2} (+) | 0.64 | 1.000 |
| L3 o c243 | H15 | a//10 {9} x b//10 {3..10} (-) | 0.76 | 0.111 |

</details>

</details>

<details><summary><b>3a-341</b> `tens(a,b)` @ `b` (add) — single component; 1 comps, L7; tells apart 2/121 classes (best member 2); on sub: 3s-424 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.25 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.62 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L6 MLP |
| CKA(arrangement before, joint write) | 0.49 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.00 |
| best source position (CKA) | `op` (0.25) |

<details><summary>codes and components</summary>

**code 3a-341.0 (L7): 1 comps, tells apart 2/121, coverage 0.25, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L7 o c3 | H5 | a//10 {0..1} x b//10 {10}; a//10 {2} x b//10 {8, 10}; a//10 {3} x b//10 {5..6, 8}; a//10 {4} x b//10 {5..9}; a//10 {5} x b//10 {1, 8, 10}; a//10 {6} x b//10 {0..2, 10}; a//10 {7} x b//10 {1..3}; a//10 {8..9} x b//10 {1..3, 10} (+) | 0.62 | 1.000 |

</details>

</details>

<details><summary><b>3a-342</b> `tens(a,b)` @ `b` (add) — single component; 1 comps, L12; tells apart 1/121 classes (best member 1); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.27 |
| classes told apart: joint / best code / best member | 1 /  / 1 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.51 |
| decoding acc. joint / best code / best member (chance) | 0.03 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L11 MLP |
| CKA(arrangement before, joint write) | 0.51 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.01 |
| best source position (CKA) | `a` (0.30) |

<details><summary>codes and components</summary>

**code 3a-342.0 (L12): 1 comps, tells apart 1/121, coverage 0.27, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L12 o c30 | H3 | a//10 {0} x b//10 {2, 5..10}; a//10 {1, 3..4} x b//10 {5..9}; a//10 {2} x b//10 {4..9}; a//10 {5} x b//10 {7..9}; a//10 {6} x b//10 {9}; a//10 {10} x b//10 {0} (+) | 0.50 | 0.685 |

</details>

</details>

<details><summary><b>3a-343</b> `tens(a,b)` @ `b` (add) — single component; 1 comps, L13; tells apart 2/121 classes (best member 2); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.28 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.54 |
| decoding acc. joint / best code / best member (chance) | 0.03 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L12 MLP |
| CKA(arrangement before, joint write) | 0.39 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.00 |
| best source position (CKA) | `a` (0.36) |

<details><summary>codes and components</summary>

**code 3a-343.0 (L13): 1 comps, tells apart 2/121, coverage 0.28, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 o c1 | H7 | a//10 {0} x b//10 {0..10}; a//10 {3} x b//10 {0, 2}; a//10 {4..5} x b//10 {0, 2..3}; a//10 {6} x b//10 {0, 3}; a//10 {7} x b//10 {2..3}; a//10 {8} x b//10 {0}; a//10 {9} x b//10 {0, 4..5}; a//10 {10} x b//10 {0..5, 10} (+) | 0.54 | 0.988 |

</details>

</details>

<details><summary><b>3a-345</b> `tens(a,b)` @ `b` (add) — single component; 1 comps, L15; tells apart 3/121 classes (best member 3); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.25 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.51 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L14 MLP |
| CKA(arrangement before, joint write) | 0.25 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.01 |
| best source position (CKA) | `a` (0.19) |

<details><summary>codes and components</summary>

**code 3a-345.0 (L15): 1 comps, tells apart 3/121, coverage 0.25, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 o c228 | H28 | a//10 {1..2} x b//10 {0, 5}; a//10 {3..4} x b//10 {0, 3}; a//10 {6..8} x b//10 {7..9}; a//10 {9} x b//10 {0..2, 4..5}; a//10 {10} x b//10 {0..6, 10} (+) | 0.51 | 0.991 |

</details>

</details>

<details><summary><b>3a-344</b> `tens(a,b)` @ `b` (add) — block code; 2 comps, L15 L16; tells apart 13/121 classes (best member 3); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.21 |
| classes told apart: joint / best code / best member | 13 /  / 3 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.60 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.70 |
| decoding acc. joint / best code / best member (chance) | 0.08 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L14 MLP |
| CKA(arrangement before, joint write) | 0.41 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.01 |
| best source position (CKA) | `op` (0.50) |

<details><summary>codes and components</summary>

**code 3a-344.0 (L15 L16): 2 comps, tells apart 13/121, coverage 0.21, overlap 1.00 (random 1.00, p 0.59)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 o c446 | H18 | a//10 {0} x b//10 {0, 4, 10}; a//10 {1..3} x b//10 {0}; a//10 {9} x b//10 {0, 4}; a//10 {10} x b//10 {0..6, 10} (+) | 0.55 | 0.995 |
| L16 o c160 | H22 | a//10 {7} x b//10 {7}; a//10 {8} x b//10 {8..9}; a//10 {9} x b//10 {5..10}; a//10 {10} x b//10 {9} (-) | 0.85 | 0.256 |

</details>

</details>

<details><summary><b>3a-346</b> `tens(a,b)` @ `b` (add) — block code, 13 codes of the same shape; 63 comps, L16 L17 L18 L19 L20 L21 L22 L23 L25 L26 L27 L28 L29 L30; tells apart 35/121 classes (best member 9); on sub: 3s-417 (member overlap 0.03)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.81 |
| classes told apart: joint / best code / best member | 35 / 25 / 9 (of 121) |
| members whose removal merges classes | 0.32 |
| support overlap (1 = tiling) / random sets / p | 5.17 / 6.52 / 0.00 |
| mean CKA between its codes | 0.81 |
| purity of the joint write (per prompt) | 0.74 |
| decoding acc. joint / best code / best member (chance) | 0.29 / 0.12 / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.42 |
| consumers / read jointly | 75 / 74 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L16 attn |
| CKA(arrangement before, joint write) | 0.63 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 1.65 |
| best source position (CKA) | `a` (0.10) |

<details><summary>codes and components</summary>

**code 3a-346.0 (L16): 4 comps, tells apart 8/121, coverage 0.24, overlap 1.03 (random 1.17, p 0.12)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c20 | MLP | a//10 {0..3} x b//10 {1} (-) | 0.74 | 0.067 |
| L16 down c22 | MLP | a//10 {1..2} x b//10 {2} (-) | 0.50 | 0.029 |
| L16 down c78 | MLP | a//10 {1} x b//10 {0} (-) | 0.52 | 0.019 |
| L16 down c75 | MLP | a//10 {3..5} x b//10 {0..1}; a//10 {6..9} x b//10 {0..2}; a//10 {10} x b//10 {0..3, 5} (-) | 0.82 | 0.513 |

**code 3a-346.1 (L17): 1 comps, tells apart 4/121, coverage 0.07, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 o c84 | H9 | a//10 {6..8, 10} x b//10 {0..1} (-) | 0.55 | 0.080 |

**code 3a-346.2 (L18): 3 comps, tells apart 7/121, coverage 0.25, overlap 1.20 (random 1.09, p 0.70)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 down c376 | MLP | a//10 {0..2} x b//10 {0..2} (+) | 0.70 | 0.093 |
| L18 down c55 | MLP | a//10 {0..6} x b//10 {0} (+) | 0.75 | 0.059 |
| L18 down c10 | MLP | a//10 {3} x b//10 {1}; a//10 {4..5, 7, 9} x b//10 {0..1}; a//10 {6, 8} x b//10 {0..2}; a//10 {10} x b//10 {0..3, 5} (+) | 0.72 | 0.571 |

**code 3a-346.3 (L19 L20): 6 comps, tells apart 13/121, coverage 0.24, overlap 1.17 (random 1.33, p 0.23)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L19 down c162 | MLP | a//10 {3} x b//10 {1}; a//10 {4..10} x b//10 {0..1} (-) | 0.80 | 0.169 |
| L19 down c80 | MLP | a//10 {8..9} x b//10 {9} (+) | 0.66 | 0.165 |
| L19 down c10 | MLP | a//10 {8} x b//10 {8..9}; a//10 {9} x b//10 {8..10}; a//10 {10} x b//10 {9} (+) | 0.61 | 0.165 |
| L19 down c70 | MLP | a//10 {9} x b//10 {9} (+) | 0.51 | 0.015 |
| L20 down c55 | MLP | a//10 {0..1} x b//10 {0..2}; a//10 {2} x b//10 {0, 2} (+) | 0.76 | 0.086 |
| L20 down c82 | MLP | a//10 {0..1} x b//10 {0} (-) | 0.62 | 0.029 |

**code 3a-346.4 (L21): 8 comps, tells apart 22/121, coverage 0.22, overlap 1.33 (random 1.47, p 0.28)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c58 | MLP | a//10 {0..2, 10} x b//10 {0} (+) | 0.69 | 0.046 |
| L21 down c116 | MLP | a//10 {0..2} x b//10 {1} (+) | 0.64 | 0.043 |
| L21 down c150 | MLP | a//10 {0..5} x b//10 {0} (-) | 0.77 | 0.094 |
| L21 down c22 | MLP | a//10 {0} x b//10 {0} (+) | 0.50 | 0.009 |
| L21 down c146 | MLP | a//10 {3} x b//10 {1}; a//10 {4..8, 10} x b//10 {0..1}; a//10 {9} x b//10 {0} (-) | 0.70 | 0.341 |
| L21 down c141 | MLP | a//10 {8..9} x b//10 {9} (-) | 0.55 | 0.017 |
| L21 down c159 | MLP | a//10 {9} x b//10 {0..3} (-) | 0.50 | 0.059 |
| L21 down c93 | MLP | a//10 {9} x b//10 {8..9} (+) | 0.64 | 0.076 |

**code 3a-346.5 (L22): 5 comps, tells apart 22/121, coverage 0.33, overlap 1.15 (random 1.26, p 0.28)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 down c34 | MLP | a//10 {0} x b//10 {0..5}; a//10 {1} x b//10 {0..2}; a//10 {2} x b//10 {0..1} (+) | 0.85 | 0.264 |
| L22 down c31 | MLP | a//10 {0} x b//10 {1..2, 7}; a//10 {1} x b//10 {1..2}; a//10 {6} x b//10 {6..7}; a//10 {7} x b//10 {7..8}; a//10 {8..9} x b//10 {7..10}; a//10 {10} x b//10 {9..10} (+) | 0.60 | 0.343 |
| L22 down c58 | MLP | a//10 {3} x b//10 {1}; a//10 {4..7, 10} x b//10 {0..1}; a//10 {8} x b//10 {0..2} (+) | 0.70 | 0.210 |
| L22 down c67 | MLP | a//10 {9} x b//10 {9} (+) | 0.54 | 0.020 |
| L22 down c213 | MLP | a//10 {9} x b//10 {9} (-) | 0.65 | 0.044 |

**code 3a-346.6 (L23): 6 comps, tells apart 13/121, coverage 0.32, overlap 1.13 (random 1.34, p 0.08)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 down c150 | MLP | a//10 {0..1} x b//10 {0} (+) | 0.54 | 0.012 |
| L23 down c567 | MLP | a//10 {0..5} x b//10 {0} (-) | 0.73 | 0.070 |
| L23 down c250 | MLP | a//10 {0} x b//10 {3..9}; a//10 {1} x b//10 {4..5, 9} (-) | 0.51 | 0.111 |
| L23 down c53 | MLP | a//10 {4..5} x b//10 {0}; a//10 {6..8, 10} x b//10 {0..1} (+) | 0.75 | 0.191 |
| L23 down c364 | MLP | a//10 {7..8, 10} x b//10 {7..9}; a//10 {9} x b//10 {5..10} (+) | 0.79 | 0.278 |
| L23 down c294 | MLP | a//10 {9} x b//10 {9} (+) | 0.57 | 0.022 |

**code 3a-346.7 (L25): 3 comps, tells apart 10/121, coverage 0.32, overlap 1.00 (random 1.10, p 0.24)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L25 down c9 | MLP | a//10 {0..1} x b//10 {6..9}; a//10 {2..6} x b//10 {9} (-) | 0.66 | 0.234 |
| L25 down c49 | MLP | a//10 {0} x b//10 {0..3, 10}; a//10 {1} x b//10 {0..2, 10}; a//10 {2..3} x b//10 {0} (-) | 0.69 | 0.114 |
| L25 down c362 | MLP | a//10 {3} x b//10 {1}; a//10 {4..10} x b//10 {0..1} (+) | 0.76 | 0.282 |

**code 3a-346.8 (L26): 6 comps, tells apart 16/121, coverage 0.26, overlap 1.12 (random 1.32, p 0.14)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 down c26 | MLP | a//10 {0..1} x b//10 {0..2}; a//10 {2} x b//10 {1..2} (-) | 0.81 | 0.099 |
| L26 down c5 | MLP | a//10 {0..1} x b//10 {0} (+) | 0.66 | 0.030 |
| L26 down c76 | MLP | a//10 {3, 6..8} x b//10 {2}; a//10 {9} x b//10 {1..3, 6..7} (-) | 0.60 | 0.131 |
| L26 down c100 | MLP | a//10 {3} x b//10 {1}; a//10 {4..8, 10} x b//10 {0..1} (+) | 0.77 | 0.343 |
| L26 down c143 | MLP | a//10 {9} x b//10 {7..9} (+) | 0.72 | 0.039 |
| L26 down c171 | MLP | a//10 {9} x b//10 {9} (+) | 0.56 | 0.014 |

**code 3a-346.9 (L27): 6 comps, tells apart 25/121, coverage 0.37, overlap 1.18 (random 1.36, p 0.16)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 down c383 | MLP | a//10 {1} x b//10 {1..2}; a//10 {2} x b//10 {2} (-) | 0.54 | 0.043 |
| L27 down c18 | MLP | a//10 {3..4} x b//10 {2}; a//10 {6} x b//10 {2..3}; a//10 {7} x b//10 {2..6}; a//10 {8} x b//10 {2..7}; a//10 {9} x b//10 {1..8} (+) | 0.71 | 0.418 |
| L27 down c774 | MLP | a//10 {3..8} x b//10 {0} (-) | 0.70 | 0.053 |
| L27 down c50 | MLP | a//10 {4} x b//10 {1}; a//10 {5} x b//10 {0}; a//10 {6..8, 10} x b//10 {0..1} (+) | 0.71 | 0.220 |
| L27 down c192 | MLP | a//10 {7} x b//10 {7..8}; a//10 {8} x b//10 {7..9}; a//10 {9} x b//10 {7..10}; a//10 {10} x b//10 {9} (+) | 0.72 | 0.109 |
| L27 down c776 | MLP | a//10 {9} x b//10 {9} (-) | 0.57 | 0.018 |

**code 3a-346.10 (L28): 4 comps, tells apart 11/121, coverage 0.30, overlap 1.17 (random 1.17, p 0.49)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 down c213 | MLP | a//10 {0..3, 5..8} x b//10 {10}; a//10 {9} x b//10 {0..10}; a//10 {10} x b//10 {9..10} (+) | 0.66 | 0.152 |
| L28 down c29 | MLP | a//10 {3, 9} x b//10 {1}; a//10 {4..7} x b//10 {0..1}; a//10 {8, 10} x b//10 {0..2} (+) | 0.77 | 0.488 |
| L28 down c631 | MLP | a//10 {6..8, 10} x b//10 {0} (-) | 0.68 | 0.056 |
| L28 down c4 | MLP | a//10 {9} x b//10 {9} (-) | 0.56 | 0.044 |

**code 3a-346.11 (L29): 5 comps, tells apart 15/121, coverage 0.12, overlap 1.87 (random 1.26, p 0.97)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L29 down c114 | MLP | a//10 {0..2, 10} x b//10 {0} (-) | 0.61 | 0.038 |
| L29 down c593 | MLP | a//10 {3..7, 10} x b//10 {0} (+) | 0.75 | 0.064 |
| L29 down c561 | MLP | a//10 {4..10} x b//10 {0} (-) | 0.81 | 0.072 |
| L29 down c687 | MLP | a//10 {6..8, 10} x b//10 {0..1} (-) | 0.70 | 0.139 |
| L29 down c164 | MLP | a//10 {6..8} x b//10 {0} (+) | 0.62 | 0.033 |

**code 3a-346.12 (L30): 6 comps, tells apart 15/121, coverage 0.40, overlap 1.56 (random 1.36, p 0.81)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 down c63 | MLP | a//10 {0..2} x b//10 {1..2} (+) | 0.69 | 0.098 |
| L30 down c3 | MLP | a//10 {0} x b//10 {3..6}; a//10 {2..3} x b//10 {8}; a//10 {4} x b//10 {0, 7..9}; a//10 {5} x b//10 {0, 6..8}; a//10 {6} x b//10 {0..1, 6..10}; a//10 {7} x b//10 {0, 6..10}; a//10 {8} x b//10 {0, 8..10}; a//10 {9} x b//10 {8..10}; a//10 {10} x b//10 {0..1, 4, 10} (+) | 0.63 | 0.610 |
| L30 down c313 | MLP | a//10 {3} x b//10 {1}; a//10 {4, 6..8, 10} x b//10 {0..1}; a//10 {5} x b//10 {0} (-) | 0.79 | 0.293 |
| L30 down c100 | MLP | a//10 {3} x b//10 {1}; a//10 {4, 6..8} x b//10 {0..1}; a//10 {5} x b//10 {0} (-) | 0.64 | 0.255 |
| L30 down c134 | MLP | a//10 {6..8, 10} x b//10 {0..1} (+) | 0.63 | 0.158 |
| L30 down c59 | MLP | a//10 {9} x b//10 {9} (+) | 0.56 | 0.042 |

</details>

</details>

<details><summary><b>3a-347</b> `tens(a,b)` @ `b` (add) — block code; 2 comps, L21; tells apart 5/121 classes (best member 1); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.31 |
| classes told apart: joint / best code / best member | 5 /  / 1 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.03 / 1.00 / 0.69 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.57 |
| decoding acc. joint / best code / best member (chance) | 0.07 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.84 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L20 MLP |
| CKA(arrangement before, joint write) | 0.47 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.02 |
| best source position (CKA) | `a` (0.17) |

<details><summary>codes and components</summary>

**code 3a-347.0 (L21): 2 comps, tells apart 5/121, coverage 0.31, overlap 1.03 (random 1.00, p 0.68)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 o c14 | H21 | a//10 {0} x b//10 {1..3, 5..9}; a//10 {1} x b//10 {1..3, 5, 7..8}; a//10 {7} x b//10 {8}; a//10 {8} x b//10 {8..10}; a//10 {9} x b//10 {9..10}; a//10 {10} x b//10 {0} (-) | 0.58 | 0.997 |
| L21 o c1 | H23 | a//10 {2} x b//10 {0}; a//10 {3..10} x b//10 {0..1} (+) | 0.61 | 0.662 |

</details>

</details>

<details><summary><b>3a-348</b> `tens(a,b)` @ `b` (add) — block code; 3 comps, L24; tells apart 10/121 classes (best member 4); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.15 |
| classes told apart: joint / best code / best member | 10 /  / 4 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.07 / 0.33 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.62 |
| decoding acc. joint / best code / best member (chance) | 0.06 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L24 attn |
| CKA(arrangement before, joint write) | 0.31 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.00 |

<details><summary>codes and components</summary>

**code 3a-348.0 (L24): 3 comps, tells apart 10/121, coverage 0.15, overlap 1.00 (random 1.10, p 0.24)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 down c48 | MLP | a//10 {0} x b//10 {0..4}; a//10 {1} x b//10 {0..2}; a//10 {2..3} x b//10 {0} (-) | 0.63 | 0.162 |
| L24 down c174 | MLP | a//10 {1..7} x b//10 {10} (+) | 0.59 | 0.014 |
| L24 down c105 | MLP | a//10 {9} x b//10 {9} (+) | 0.51 | 0.013 |

</details>

</details>

<details><summary><b>3a-349</b> `tens(a,b)` @ `b` (add) — single component; 1 comps, L26; tells apart 3/121 classes (best member 3); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.05 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.51 |
| decoding acc. joint / best code / best member (chance) | 0.03 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L25 MLP |
| CKA(arrangement before, joint write) | 0.18 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.04 |
| share of the write inside the old arrangement's span | 0.29 |
| write energy / code energy before | 0.00 |
| best source position (CKA) | `a` (0.31) |

<details><summary>codes and components</summary>

**code 3a-349.0 (L26): 1 comps, tells apart 3/121, coverage 0.05, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 o c19 | H10 | a//10 {0} x b//10 {0, 10}; a//10 {1} x b//10 {2, 10}; a//10 {9..10} x b//10 {10} (-) | 0.50 | 1.000 |

</details>

</details>

</details>

<details><summary>`b//10`: 8 mechanisms, 71 components</summary>

<details><summary><b>3a-328</b> `b//10` @ `b` (add) — block code, 3 codes of the same shape; 15 comps, L0 L3 L14; tells apart 9/11 classes (best member 5); on sub: 3s-406 (member overlap 0.62)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 9 / 10 / 5 (of 11) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 3.64 / 2.71 / 0.96 |
| mean CKA between its codes | 0.83 |
| purity of the joint write (per prompt) | 0.90 |
| decoding acc. joint / best code / best member (chance) | 0.82 / 0.90 / 0.50 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 0.90 |
| consumers / read jointly | 165 / 144 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 attn |
| CKA(arrangement before, joint write) | 0.78 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.11 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 12.89 |

<details><summary>codes and components</summary>

**code 3a-328.0 (L0): 7 comps, tells apart 10/11, coverage 1.00, overlap 1.64 (random 1.80, p 0.29)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 down c5 | MLP | b//10 in {0..1, 3..6, 9..10} (-) | 0.94 | 0.762 |
| L0 down c23 | MLP | b//10 in {0..2} (+) | 0.96 | 0.390 |
| L0 down c19 | MLP | b//10 in {0} (+) | 0.91 | 0.910 |
| L0 down c41 | MLP | b//10 in {0} (+) | 0.90 | 0.091 |
| L0 down c405 | MLP | b//10 in {10} (+) | 1.00 | 0.010 |
| L0 down c123 | MLP | b//10 in {10} (-) | 1.00 | 0.010 |
| L0 down c6 | MLP | b//10 in {7..9} (-) | 0.97 | 0.472 |

**code 3a-328.1 (L3): 4 comps, tells apart 7/11, coverage 0.91, overlap 1.30 (random 1.40, p 0.24)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 down c10 | MLP | b//10 in {0..4, 7..9} (-) | 0.85 | 0.738 |
| L3 down c212 | MLP | b//10 in {1..2} (+) | 0.86 | 0.215 |
| L3 down c113 | MLP | b//10 in {5} (-) | 0.89 | 0.162 |
| L3 down c93 | MLP | b//10 in {6..7} (-) | 0.89 | 0.269 |

**code 3a-328.2 (L14): 4 comps, tells apart 8/11, coverage 0.45, overlap 1.80 (random 1.40, p 0.72)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L14 down c0 | MLP | b//10 in {0..2, 10} (+) | 0.90 | 0.989 |
| L14 down c8 | MLP | b//10 in {0..2} (-) | 0.94 | 0.372 |
| L14 down c255 | MLP | b//10 in {10} (+) | 0.87 | 0.129 |
| L14 down c34 | MLP | b//10 in {9} (-) | 0.90 | 0.206 |

</details>

</details>

<details><summary><b>3a-327</b> `b//10` @ `b` (add) — block code, 5 codes of the same shape; 15 comps, L1 L2 L5 L6 L7 L17; tells apart 5/11 classes (best member 4); on sub: 3s-407 (member overlap 0.16)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.55 |
| classes told apart: joint / best code / best member | 5 / 5 / 4 (of 11) |
| members whose removal merges classes | 0.07 |
| support overlap (1 = tiling) / random sets / p | 3.00 / 2.68 / 0.79 |
| mean CKA between its codes | 0.84 |
| purity of the joint write (per prompt) | 0.85 |
| decoding acc. joint / best code / best member (chance) | 0.64 / 0.49 / 0.33 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 26 / 17 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L1 attn |
| CKA(arrangement before, joint write) | 0.50 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 1.18 |

<details><summary>codes and components</summary>

**code 3a-327.0 (L1): 4 comps, tells apart 5/11, coverage 0.36, overlap 1.25 (random 1.38, p 0.28)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 down c27 | MLP | b//10 in {0..1} (-) | 0.92 | 0.202 |
| L1 down c62 | MLP | b//10 in {0} (-) | 0.90 | 0.098 |
| L1 down c271 | MLP | b//10 in {8} (+) | 0.93 | 0.111 |
| L1 down c337 | MLP | b//10 in {9} (+) | 0.87 | 0.128 |

**code 3a-327.1 (L2): 4 comps, tells apart 5/11, coverage 0.36, overlap 1.25 (random 1.40, p 0.25)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 down c67 | MLP | b//10 in {0} (+) | 0.87 | 0.146 |
| L2 down c109 | MLP | b//10 in {0} (-) | 0.79 | 0.916 |
| L2 down c107 | MLP | b//10 in {1..2} (+) | 0.87 | 0.246 |
| L2 down c729 | MLP | b//10 in {10} (+) | 0.99 | 0.010 |

**code 3a-327.2 (L5): 2 comps, tells apart 4/11, coverage 0.27, overlap 1.00 (random 1.00, p 0.67)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 down c27 | MLP | b//10 in {0..1} (+) | 0.86 | 0.201 |
| L5 down c936 | MLP | b//10 in {10} (-) | 0.96 | 0.010 |

**code 3a-327.3 (L6 L7): 3 comps, tells apart 4/11, coverage 0.18, overlap 1.50 (random 1.33, p 0.89)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L6 down c133 | MLP | b//10 in {10} (+) | 0.98 | 0.012 |
| L6 down c848 | MLP | b//10 in {10} (+) | 0.92 | 0.010 |
| L7 down c80 | MLP | b//10 in {0} (+) | 0.84 | 0.126 |

**code 3a-327.4 (L17): 2 comps, tells apart 4/11, coverage 0.18, overlap 1.00 (random 1.00, p 0.62)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 down c27 | MLP | b//10 in {0} (+) | 0.82 | 0.148 |
| L17 down c110 | MLP | b//10 in {10} (+) | 0.98 | 0.016 |

</details>

</details>

<details><summary><b>3a-329</b> `b//10` @ `b` (add) — block code; 4 comps, L4; tells apart 5/11 classes (best member 3); on sub: 3s-410 (member overlap 0.22)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.45 |
| classes told apart: joint / best code / best member | 5 /  / 3 (of 11) |
| members whose removal merges classes | 0.50 |
| support overlap (1 = tiling) / random sets / p | 1.20 / 1.42 / 0.18 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.90 |
| decoding acc. joint / best code / best member (chance) | 0.48 /  / 0.25 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 0.99 |
| consumers / read jointly | 4 / 1 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L4 attn |
| CKA(arrangement before, joint write) | 0.74 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.07 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.10 |

<details><summary>codes and components</summary>

**code 3a-329.0 (L4): 4 comps, tells apart 5/11, coverage 0.45, overlap 1.20 (random 1.40, p 0.18)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 down c327 | MLP | b//10 in {0} (-) | 0.87 | 0.135 |
| L4 down c313 | MLP | b//10 in {10} (+) | 0.99 | 0.010 |
| L4 down c21 | MLP | b//10 in {7..9} (-) | 0.93 | 0.306 |
| L4 down c390 | MLP | b//10 in {8} (-) | 0.90 | 0.112 |

</details>

</details>

<details><summary><b>3a-330</b> `b//10` @ `b` (add) — block code; 5 comps, L11; tells apart 6/11 classes (best member 3); on sub: 3s-409 (member overlap 0.80)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.45 |
| classes told apart: joint / best code / best member | 6 /  / 3 (of 11) |
| members whose removal merges classes | 0.80 |
| support overlap (1 = tiling) / random sets / p | 1.20 / 1.50 / 0.17 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.91 |
| decoding acc. joint / best code / best member (chance) | 0.60 /  / 0.27 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 0.99 |
| consumers / read jointly | 5 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L11 attn |
| CKA(arrangement before, joint write) | 0.42 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.08 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.04 |

<details><summary>codes and components</summary>

**code 3a-330.0 (L11): 5 comps, tells apart 6/11, coverage 0.45, overlap 1.20 (random 1.67, p 0.08)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L11 down c465 | MLP | b//10 in {10} (-) | 0.95 | 0.010 |
| L11 down c47 | MLP | b//10 in {2} (+) | 0.91 | 0.110 |
| L11 down c10 | MLP | b//10 in {6} (-) | 0.91 | 0.156 |
| L11 down c30 | MLP | b//10 in {7} (-) | 0.88 | 0.122 |
| L11 down c22 | MLP | b//10 in {9..10} (-) | 0.93 | 0.139 |

</details>

</details>

<details><summary><b>3a-331</b> `b//10` @ `b` (add) — tiling; 5 comps, L12; tells apart 7/11 classes (best member 3); on sub: 3s-410 (member overlap 0.33)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.55 |
| classes told apart: joint / best code / best member | 7 /  / 3 (of 11) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.17 / 1.50 / 0.04 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.90 |
| decoding acc. joint / best code / best member (chance) | 0.66 /  / 0.27 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 0.98 |
| consumers / read jointly | 12 / 6 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L12 attn |
| CKA(arrangement before, joint write) | 0.72 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.07 |

<details><summary>codes and components</summary>

**code 3a-331.0 (L12): 5 comps, tells apart 7/11, coverage 0.55, overlap 1.17 (random 1.50, p 0.04)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L12 down c28 | MLP | b//10 in {0..1, 10} (-) | 0.89 | 0.200 |
| L12 down c261 | MLP | b//10 in {10} (+) | 0.99 | 0.010 |
| L12 down c26 | MLP | b//10 in {3} (-) | 0.91 | 0.171 |
| L12 down c13 | MLP | b//10 in {4} (+) | 0.90 | 0.183 |
| L12 down c59 | MLP | b//10 in {6} (-) | 0.93 | 0.126 |

</details>

</details>

<details><summary><b>3a-332</b> `b//10` @ `b` (add) — block code, 2 codes of the same shape; 7 comps, L13 L21; tells apart 3/11 classes (best member 4); on sub: 3s-410 (member overlap 0.08)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.18 |
| classes told apart: joint / best code / best member | 3 / 5 / 4 (of 11) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 3.50 / 1.80 / 1.00 |
| mean CKA between its codes | 0.87 |
| purity of the joint write (per prompt) | 0.89 |
| decoding acc. joint / best code / best member (chance) | 0.34 / 0.35 / 0.26 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.18 |
| consumers / read jointly | 19 / 5 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L13 attn |
| CKA(arrangement before, joint write) | 0.31 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.08 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.10 |

<details><summary>codes and components</summary>

**code 3a-332.0 (L13): 3 comps, tells apart 5/11, coverage 0.18, overlap 1.50 (random 1.33, p 0.83)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 down c161 | MLP | b//10 in {10} (-) | 0.89 | 0.021 |
| L13 down c479 | MLP | b//10 in {10} (-) | 0.99 | 0.010 |
| L13 down c10 | MLP | b//10 in {8} (+) | 0.92 | 0.254 |

**code 3a-332.1 (L21): 4 comps, tells apart 3/11, coverage 0.18, overlap 2.00 (random 1.42, p 0.90)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c149 | MLP | b//10 in {10} (+) | 0.97 | 0.011 |
| L21 down c188 | MLP | b//10 in {10} (+) | 0.99 | 0.011 |
| L21 down c55 | MLP | b//10 in {10} (-) | 0.86 | 0.019 |
| L21 down c84 | MLP | b//10 in {8} (-) | 0.81 | 0.102 |

</details>

</details>

<details><summary><b>3a-333</b> `b//10` @ `b` (add) — block code, 2 codes of the same shape; 6 comps, L15 L16; tells apart 5/11 classes (best member 4); on sub: 3s-411 (member overlap 0.33)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.27 |
| classes told apart: joint / best code / best member | 5 / 5 / 4 (of 11) |
| members whose removal merges classes | 0.17 |
| support overlap (1 = tiling) / random sets / p | 2.33 / 1.69 / 0.90 |
| mean CKA between its codes | 0.90 |
| purity of the joint write (per prompt) | 0.87 |
| decoding acc. joint / best code / best member (chance) | 0.39 / 0.36 / 0.27 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.03 |
| consumers / read jointly | 13 / 13 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 attn |
| CKA(arrangement before, joint write) | 0.42 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 0.04 |

<details><summary>codes and components</summary>

**code 3a-333.0 (L15): 4 comps, tells apart 5/11, coverage 0.27, overlap 1.67 (random 1.38, p 0.73)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c48 | MLP | b//10 in {0} (+) | 0.84 | 0.082 |
| L15 down c398 | MLP | b//10 in {10} (+) | 0.96 | 0.010 |
| L15 down c193 | MLP | b//10 in {10} (-) | 0.79 | 0.008 |
| L15 down c31 | MLP | b//10 in {9..10} (-) | 0.81 | 0.135 |

**code 3a-333.1 (L16): 2 comps, tells apart 4/11, coverage 0.18, overlap 1.00 (random 1.00, p 0.58)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c38 | MLP | b//10 in {10} (-) | 0.87 | 0.019 |
| L16 down c9 | MLP | b//10 in {9} (-) | 0.89 | 0.177 |

</details>

</details>

<details><summary><b>3a-334</b> `b//10` @ `b` (add) — block code, 10 codes of the same shape; 14 comps, L18 L19 L22 L23 L24 L25 L26 L27 L29 L30; tells apart 2/11 classes (best member 2); on sub: 3s-412 (member overlap 0.33)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.09 |
| classes told apart: joint / best code / best member | 2 / 2 / 2 (of 11) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 14.00 / 2.63 / 1.00 |
| mean CKA between its codes | 1.00 |
| purity of the joint write (per prompt) | 0.95 |
| decoding acc. joint / best code / best member (chance) | 0.20 / 0.19 / 0.19 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.63 |
| consumers / read jointly | 11 / 10 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L18 attn |
| CKA(arrangement before, joint write) | 0.14 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.04 |

<details><summary>codes and components</summary>

**code 3a-334.0 (L18): 2 comps, tells apart 2/11, coverage 0.09, overlap 2.00 (random 1.00, p 1.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 down c51 | MLP | b//10 in {10} (+) | 0.91 | 0.022 |
| L18 down c460 | MLP | b//10 in {10} (+) | 0.98 | 0.011 |

**code 3a-334.1 (L19): 1 comps, tells apart 2/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L19 down c700 | MLP | b//10 in {10} (-) | 0.96 | 0.010 |

**code 3a-334.2 (L22): 2 comps, tells apart 2/11, coverage 0.09, overlap 2.00 (random 1.00, p 1.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 down c7 | MLP | b//10 in {10} (-) | 1.00 | 0.010 |
| L22 down c757 | MLP | b//10 in {10} (-) | 0.85 | 0.012 |

**code 3a-334.3 (L23): 1 comps, tells apart 2/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 down c126 | MLP | b//10 in {10} (-) | 0.96 | 0.011 |

**code 3a-334.4 (L24): 1 comps, tells apart 2/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 down c21 | MLP | b//10 in {10} (-) | 0.99 | 0.012 |

**code 3a-334.5 (L25): 1 comps, tells apart 2/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L25 down c87 | MLP | b//10 in {10} (-) | 0.98 | 0.014 |

**code 3a-334.6 (L26): 1 comps, tells apart 2/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 down c737 | MLP | b//10 in {10} (+) | 0.94 | 0.013 |

**code 3a-334.7 (L27): 1 comps, tells apart 2/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 down c36 | MLP | b//10 in {10} (-) | 0.97 | 0.013 |

**code 3a-334.8 (L29): 3 comps, tells apart 2/11, coverage 0.09, overlap 3.00 (random 1.33, p 1.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L29 down c139 | MLP | b//10 in {10} (+) | 0.48 | 0.018 |
| L29 down c465 | MLP | b//10 in {10} (+) | 0.99 | 0.011 |
| L29 down c133 | MLP | b//10 in {10} (-) | 0.90 | 0.022 |

**code 3a-334.9 (L30): 1 comps, tells apart 2/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 down c15 | MLP | b//10 in {10} (+) | 1.00 | 0.011 |

</details>

</details>

</details>

<details><summary>`a%100`: 19 mechanisms, 61 components</summary>

<details><summary><b>3a-264</b> `a%100` @ `b` (add) — block code, 3 codes of the same shape; 4 comps, L1 L3; tells apart 7/100 classes (best member 7); on sub: 3s-355 (member overlap 0.22)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.07 |
| classes told apart: joint / best code / best member | 7 / 9 / 7 (of 100) |
| members whose removal merges classes | 0.25 |
| support overlap (1 = tiling) / random sets / p | 2.00 / 1.25 / 0.99 |
| mean CKA between its codes | 0.91 |
| purity of the joint write (per prompt) | 0.78 |
| decoding acc. joint / best code / best member (chance) | 0.04 / 0.03 / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.44 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.93 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.26 2:0.10 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.09 2:0.08 3:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.03 / 1.10 |
| best source position (CKA) | `op` (0.35) |

<details><summary>codes and components</summary>

**code 3a-264.0 (L1): 2 comps, tells apart 9/100, coverage 0.07, overlap 1.00 (random 1.00, p 0.56)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c285 | H24 | a%100 in {1..4} (-) | 0.81 | 0.038 |
| L1 o c175 | H24 | a%100 in {10..12} (-) | 0.57 | 0.022 |

**code 3a-264.1 (L3): 1 comps, tells apart 6/100, coverage 0.03, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 down c717 | MLP | a%100 in {1..3} (+) | 0.77 | 0.033 |

**code 3a-264.2 (L3): 1 comps, tells apart 7/100, coverage 0.04, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 o c221 | H5 | a%100 in {1..4} (-) | 0.81 | 0.040 |

</details>

</details>

<details><summary><b>3a-263</b> `a%100` @ `b` (add) — block code; 5 comps, L1; tells apart 17/100 classes (best member 6); on sub: 3s-354 (member overlap 0.20)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.14 |
| classes told apart: joint / best code / best member | 17 /  / 6 (of 100) |
| members whose removal merges classes | 0.80 |
| support overlap (1 = tiling) / random sets / p | 1.07 / 1.30 / 0.07 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.74 |
| decoding acc. joint / best code / best member (chance) | 0.10 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 6 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.25 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.80 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.26 2:0.10 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (20:0.18 40:0.17 10:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.14 / 3.13 |
| best source position (CKA) | `a` (0.31) |

<details><summary>codes and components</summary>

**code 3a-263.0 (L1): 5 comps, tells apart 17/100, coverage 0.14, overlap 1.07 (random 1.30, p 0.07)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c388 | H26 | a%100 in {0, 50, 60, 80, 90} (+) | 0.63 | 0.108 |
| L1 o c320 | H26 | a%100 in {10..12} (-) | 0.78 | 0.027 |
| L1 o c416 | H18 | a%100 in {1} (-) | 0.82 | 0.078 |
| L1 o c251 | H26 | a%100 in {90} (+) | 0.77 | 0.013 |
| L1 o c330 | H26 | a%100 in {95..99} (-) | 0.84 | 0.098 |

</details>

</details>

<details><summary><b>3a-265</b> `a%100` @ `b` (add) — block code, 4 codes of the same shape; 6 comps, L1 L9 L10 L11; tells apart 10/100 classes (best member 9); on sub: 3s-361 (member overlap 0.08)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.13 |
| classes told apart: joint / best code / best member | 10 / 9 / 9 (of 100) |
| members whose removal merges classes | 0.17 |
| support overlap (1 = tiling) / random sets / p | 2.62 / 1.38 / 1.00 |
| mean CKA between its codes | 0.90 |
| purity of the joint write (per prompt) | 0.80 |
| decoding acc. joint / best code / best member (chance) | 0.05 / 0.06 / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.99 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L1 attn |
| CKA(arrangement before, joint write) | 0.44 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 1.43 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.34 2:0.14 20:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.21 2:0.18 3:0.15) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.07 / 1.11 |

<details><summary>codes and components</summary>

**code 3a-265.0 (L1): 3 comps, tells apart 9/100, coverage 0.13, overlap 1.08 (random 1.11, p 0.41)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 down c691 | MLP | a%100 in {0, 40, 50, 80, 90} (+) | 0.74 | 0.077 |
| L1 down c98 | MLP | a%100 in {1..8} (+) | 0.87 | 0.096 |
| L1 down c258 | MLP | a%100 in {1} (-) | 0.93 | 0.010 |

**code 3a-265.1 (L9): 1 comps, tells apart 9/100, coverage 0.06, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L9 down c865 | MLP | a%100 in {1..6} (+) | 0.84 | 0.061 |

**code 3a-265.2 (L10): 1 comps, tells apart 7/100, coverage 0.06, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L10 down c384 | MLP | a%100 in {1..6} (-) | 0.83 | 0.079 |

**code 3a-265.3 (L11): 1 comps, tells apart 6/100, coverage 0.08, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L11 down c663 | MLP | a%100 in {1..8} (+) | 0.76 | 0.110 |

</details>

</details>

<details><summary><b>3a-266</b> `a%100` @ `b` (add) — block code; 4 comps, L2 L3; tells apart 10/100 classes (best member 6); on sub: 3s-357 (member overlap 0.29)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.21 |
| classes told apart: joint / best code / best member | 10 /  / 6 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.05 / 1.24 / 0.10 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.58 |
| decoding acc. joint / best code / best member (chance) | 0.06 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 2 / 1 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 attn |
| CKA(arrangement before, joint write) | 0.24 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.35 2:0.13 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.32 2:0.14 40:0.12) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.16 / 2.09 |
| best source position (CKA) | `a` (0.31) |

<details><summary>codes and components</summary>

**code 3a-266.0 (L2 L3): 4 comps, tells apart 10/100, coverage 0.21, overlap 1.05 (random 1.20, p 0.11)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 down c561 | MLP | a%100 in {1} (-) | 0.79 | 0.015 |
| L3 o c427 | H15 | a%100 in {0, 40, 50, 55, 60, 65, 70, 80, 90} (+) | 0.71 | 0.128 |
| L3 o c49 | H7 | a%100 in {33, 37, 39, 41..43, 46..47, 49, 51..52} (-) | 0.51 | 0.195 |
| L3 o c153 | H15 | a%100 in {90} (-) | 0.73 | 0.043 |

</details>

</details>

<details><summary><b>3a-267</b> `a%100` @ `b` (add) — block code, copy from `a`; 7 comps, L5 L6; tells apart 27/100 classes (best member 7); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.69 |
| classes told apart: joint / best code / best member | 27 /  / 7 (of 100) |
| members whose removal merges classes | 0.71 |
| support overlap (1 = tiling) / random sets / p | 1.30 / 1.48 / 0.18 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.78 |
| decoding acc. joint / best code / best member (chance) | 0.24 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.03 |
| consumers / read jointly | 11 / 3 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L4 MLP |
| CKA(arrangement before, joint write) | 0.60 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.09 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.07 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.35 2:0.13 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.38 2:0.25 50:0.11) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.40 / 3.37 |
| best source position (CKA) | `a` (0.61) |

<details><summary>codes and components</summary>

**code 3a-267.0 (L5 L6): 7 comps, tells apart 27/100, coverage 0.69, overlap 1.30 (random 1.53, p 0.15)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 o c372 | H22 | a%100 in {0, 40, 50, 60, 70, 80} (-) | 0.85 | 0.111 |
| L5 o c158 | H22 | a%100 in {12, 24, 32, 34, 36, 42, 44, 48, 52, 54, 60, 62, 64, 68, 72, 74, 76, 78, 80, 82, 84, 86, 92, 96} (-) | 0.69 | 0.338 |
| L5 o c31 | H22 | a%100 in {24..39} (+) | 0.77 | 0.194 |
| L5 o c78 | H22 | a%100 in {3..12} (-) | 0.75 | 0.126 |
| L5 o c53 | H22 | a%100 in {39, 41..54} (-) | 0.54 | 0.139 |
| L5 o c59 | H22 | a%100 in {7..8, 10..25} (-) | 0.90 | 0.269 |
| L6 o c161 | H16 | a%100 in {55} (+) | 0.56 | 0.006 |

</details>

</details>

<details><summary><b>3a-268</b> `a%100` @ `b` (add) — block code, 4 codes of the same shape; 5 comps, L6 L7 L8 L10 L12; tells apart 6/100 classes (best member 5); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.15 |
| classes told apart: joint / best code / best member | 6 / 6 / 5 (of 100) |
| members whose removal merges classes | 0.80 |
| support overlap (1 = tiling) / random sets / p | 2.73 / 1.33 / 1.00 |
| mean CKA between its codes | 0.80 |
| purity of the joint write (per prompt) | 0.61 |
| decoding acc. joint / best code / best member (chance) | 0.09 / 0.05 / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.41 |
| consumers / read jointly | 5 / 5 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L6 attn |
| CKA(arrangement before, joint write) | 0.53 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.11 |
| write energy / code energy before | 0.10 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.35 2:0.13 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.46 2:0.16 20:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.13 / 1.05 |
| best source position (CKA) | `a` (0.49) |

<details><summary>codes and components</summary>

**code 3a-268.0 (L6 L7): 2 comps, tells apart 6/100, coverage 0.11, overlap 1.00 (random 1.00, p 0.60)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L6 down c53 | MLP | a%100 in {0} (+) | 0.52 | 0.017 |
| L7 o c11 | H23 | a%100 in {1..6, 10..11, 18, 90} (+) | 0.52 | 0.970 |

**code 3a-268.1 (L8): 1 comps, tells apart 3/100, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L8 o c289 | H16 | a%100 in {1..6, 18, 55, 90} (+) | 0.71 | 0.995 |

**code 3a-268.2 (L10): 1 comps, tells apart 3/100, coverage 0.08, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L10 o c54 | H15 | a%100 in {1..6, 10, 90} (+) | 0.48 | 0.913 |

**code 3a-268.3 (L12): 1 comps, tells apart 3/100, coverage 0.13, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L12 o c209 | H20 | a%100 in {0..11, 90} (-) | 0.55 | 1.000 |

</details>

</details>

<details><summary><b>3a-269</b> `a%100` @ `b` (add) — copy from `a`; 1 comps, L8; tells apart 2/100 classes (best member 2); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.22 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.56 |
| decoding acc. joint / best code / best member (chance) | 0.03 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L7 MLP |
| CKA(arrangement before, joint write) | 0.54 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.35 2:0.13 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.43 2:0.31) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.14 / 1.00 |
| best source position (CKA) | `a` (0.51) |

<details><summary>codes and components</summary>

**code 3a-269.0 (L8): 1 comps, tells apart 2/100, coverage 0.22, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L8 o c7 | H23 | a%100 in {0..1, 4, 6..9, 12..19, 21, 23..24, 86, 90..91, 99} (-) | 0.55 | 1.000 |

</details>

</details>

<details><summary><b>3a-270</b> `a%100` @ `b` (add) — single component; 1 comps, L8; tells apart 1/100 classes (best member 1); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.17 |
| classes told apart: joint / best code / best member | 1 /  / 1 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.74 |
| decoding acc. joint / best code / best member (chance) | 0.03 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L8 attn |
| CKA(arrangement before, joint write) | 0.75 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.08 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.04 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.35 2:0.13 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.56 2:0.21 3:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.19 / 1.00 |

<details><summary>codes and components</summary>

**code 3a-270.0 (L8): 1 comps, tells apart 1/100, coverage 0.17, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L8 down c31 | MLP | a%100 in {2..16, 19..20} (+) | 0.73 | 0.581 |

</details>

</details>

<details><summary><b>3a-271</b> `a%100` @ `b` (add) — block code, copy from `op`; 2 comps, L14; tells apart 5/100 classes (best member 3); on sub: 3s-364 (member overlap 0.50)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.22 |
| classes told apart: joint / best code / best member | 5 /  / 3 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.14 / 1.00 / 0.80 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.72 |
| decoding acc. joint / best code / best member (chance) | 0.10 /  / 0.06 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.99 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L13 MLP |
| CKA(arrangement before, joint write) | 0.36 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.35 2:0.13 3:0.05) |
| joint write: shape (spectrum k:share) | line (1:0.47 2:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.14 / 1.17 |
| best source position (CKA) | `op` (0.61) |

<details><summary>codes and components</summary>

**code 3a-271.0 (L14): 2 comps, tells apart 5/100, coverage 0.22, overlap 1.14 (random 1.00, p 0.79)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L14 o c59 | H26 | a%100 in {0..1, 5, 7, 15, 30, 40, 50, 55, 60, 65, 70, 91, 97..99} (-) | 0.58 | 1.000 |
| L14 o c37 | H26 | a%100 in {1..2, 11, 55, 90..92, 94, 96} (+) | 0.74 | 0.997 |

</details>

</details>

<details><summary><b>3a-274</b> `a%100` @ `b` (add) — block code, 3 codes of the same shape, copy from `op`; 5 comps, L15 L17 L30; tells apart 8/100 classes (best member 5); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.34 |
| classes told apart: joint / best code / best member | 8 / 7 / 5 (of 100) |
| members whose removal merges classes | 0.40 |
| support overlap (1 = tiling) / random sets / p | 1.59 / 1.34 / 0.84 |
| mean CKA between its codes | 0.82 |
| purity of the joint write (per prompt) | 0.84 |
| decoding acc. joint / best code / best member (chance) | 0.10 / 0.06 / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.27 |
| consumers / read jointly | 21 / 18 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L14 MLP |
| CKA(arrangement before, joint write) | 0.42 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.14 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.35 2:0.12 3:0.05) |
| joint write: shape (spectrum k:share) | line (1:0.30 2:0.21 3:0.15) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.12 / 1.39 |
| best source position (CKA) | `op` (0.58) |

<details><summary>codes and components</summary>

**code 3a-274.0 (L15): 1 comps, tells apart 3/100, coverage 0.12, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 o c131 | H28 | a%100 in {86, 88..98} (-) | 0.81 | 0.186 |

**code 3a-274.1 (L17): 2 comps, tells apart 7/100, coverage 0.10, overlap 1.10 (random 1.00, p 0.73)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 o c33 | H9 | a%100 in {89..98} (-) | 0.90 | 0.137 |
| L17 o c322 | H9 | a%100 in {90} (+) | 0.78 | 0.016 |

**code 3a-274.2 (L30): 2 comps, tells apart 5/100, coverage 0.31, overlap 1.00 (random 1.00, p 0.56)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 o c446 | H18 | a%100 in {1, 23..24, 29, 33..34, 37..39, 41..44, 46..49, 60, 64, 70, 80, 85} (+) | 0.75 | 1.000 |
| L30 down c219 | MLP | a%100 in {90..98} (-) | 0.77 | 0.122 |

</details>

</details>

<details><summary><b>3a-273</b> `a%100` @ `b` (add) — single component; 1 comps, L16; tells apart 1/100 classes (best member 1); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.16 |
| classes told apart: joint / best code / best member | 1 /  / 1 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.52 |
| decoding acc. joint / best code / best member (chance) | 0.03 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 MLP |
| CKA(arrangement before, joint write) | 0.44 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.34 2:0.12 3:0.05) |
| joint write: shape (spectrum k:share) | line (1:0.34 4:0.19 2:0.15) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.09 / 1.00 |
| best source position (CKA) | `op` (0.49) |

<details><summary>codes and components</summary>

**code 3a-273.0 (L16): 1 comps, tells apart 1/100, coverage 0.16, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c207 | H30 | a%100 in {11, 25, 27..28, 30..31, 89..98} (-) | 0.51 | 1.000 |

</details>

</details>

<details><summary><b>3a-272</b> `a%100` @ `b` (add) — block code; 2 comps, L16; tells apart 4/100 classes (best member 3); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.02 |
| classes told apart: joint / best code / best member | 4 /  / 3 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.60 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.51 |
| decoding acc. joint / best code / best member (chance) | 0.02 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 MLP |
| CKA(arrangement before, joint write) | 0.15 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.34 2:0.12 3:0.05) |
| joint write: shape (spectrum k:share) | irregular () |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.02 / 1.55 |
| best source position (CKA) | `op` (0.11) |

<details><summary>codes and components</summary>

**code 3a-272.0 (L16): 2 comps, tells apart 4/100, coverage 0.02, overlap 1.00 (random 1.00, p 0.51)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c322 | H22 | a%100 in {18} (-) | 0.50 | 0.012 |
| L16 o c58 | H22 | a%100 in {90} (-) | 0.50 | 0.011 |

</details>

</details>

<details><summary><b>3a-276</b> `a%100` @ `b` (add) — single component; 1 comps, L21; tells apart 3/100 classes (best member 3); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.01 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.86 |
| decoding acc. joint / best code / best member (chance) | 0.02 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L20 MLP |
| CKA(arrangement before, joint write) | 0.08 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.32 2:0.11 3:0.05) |
| joint write: shape (spectrum k:share) | line () |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.01 / 1.00 |
| best source position (CKA) | `op` (0.12) |

<details><summary>codes and components</summary>

**code 3a-276.0 (L21): 1 comps, tells apart 3/100, coverage 0.01, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 o c3 | H23 | a%100 in {55} (-) | 0.85 | 0.016 |

</details>

</details>

<details><summary><b>3a-275</b> `a%100` @ `b` (add) — block code, copy from `op`; 3 comps, L21 L22; tells apart 5/100 classes (best member 6); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.20 |
| classes told apart: joint / best code / best member | 5 /  / 6 (of 100) |
| members whose removal merges classes | 0.33 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.11 / 0.24 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.72 |
| decoding acc. joint / best code / best member (chance) | 0.05 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L20 MLP |
| CKA(arrangement before, joint write) | 0.58 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.09 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.32 2:0.11 3:0.05) |
| joint write: shape (spectrum k:share) | irregular (1:0.25 2:0.19 3:0.15) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.17 / 2.43 |
| best source position (CKA) | `op` (0.62) |

<details><summary>codes and components</summary>

**code 3a-275.0 (L21 L22): 3 comps, tells apart 5/100, coverage 0.20, overlap 1.00 (random 1.12, p 0.25)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 o c10 | H21 | a%100 in {1..11} (+) | 0.72 | 0.125 |
| L21 o c20 | H21 | a%100 in {55} (+) | 0.85 | 0.029 |
| L22 down c64 | MLP | a%100 in {91..98} (-) | 0.68 | 0.072 |

</details>

</details>

<details><summary><b>3a-277</b> `a%100` @ `b` (add) — block code, copy from `op`; 7 comps, L22; tells apart 17/100 classes (best member 8); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.43 |
| classes told apart: joint / best code / best member | 17 /  / 8 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.21 / 1.47 / 0.07 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.73 |
| decoding acc. joint / best code / best member (chance) | 0.15 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 MLP |
| CKA(arrangement before, joint write) | 0.70 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.16 |
| share of the write inside the old arrangement's span | 0.15 |
| write energy / code energy before | 0.03 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.31 2:0.11 3:0.05) |
| joint write: shape (spectrum k:share) | irregular (1:0.28 2:0.19 3:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.17 / 2.47 |
| best source position (CKA) | `op` (0.75) |

<details><summary>codes and components</summary>

**code 3a-277.0 (L22): 7 comps, tells apart 17/100, coverage 0.43, overlap 1.21 (random 1.46, p 0.14)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 o c2 | H24 | a%100 in {0, 11, 80, 90..99} (-) | 0.65 | 0.875 |
| L22 o c0 | H24 | a%100 in {0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90} (-) | 0.78 | 0.258 |
| L22 o c120 | H24 | a%100 in {1..4} (+) | 0.95 | 0.090 |
| L22 o c16 | H24 | a%100 in {3..10, 12..17, 19} (-) | 0.77 | 0.187 |
| L22 o c89 | H24 | a%100 in {90..91} (-) | 0.68 | 0.044 |
| L22 o c102 | H24 | a%100 in {90} (+) | 0.63 | 0.045 |
| L22 o c418 | H24 | a%100 in {90} (-) | 0.86 | 0.013 |

</details>

</details>

<details><summary><b>3a-278</b> `a%100` @ `b` (add) — block code, copy from `op`; 4 comps, L24; tells apart 16/100 classes (best member 5); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.34 |
| classes told apart: joint / best code / best member | 16 /  / 5 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.09 / 1.21 / 0.25 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.60 |
| decoding acc. joint / best code / best member (chance) | 0.11 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L23 MLP |
| CKA(arrangement before, joint write) | 0.69 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.09 |
| share of the write inside the old arrangement's span | 0.09 |
| write energy / code energy before | 0.03 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.30 2:0.11 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.25 2:0.14 20:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.15 / 2.77 |
| best source position (CKA) | `op` (0.71) |

<details><summary>codes and components</summary>

**code 3a-278.0 (L24): 4 comps, tells apart 16/100, coverage 0.34, overlap 1.09 (random 1.21, p 0.22)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 o c2 | H10 | a%100 in {0..2, 11, 55, 60, 65, 70, 75, 80} (-) | 0.60 | 0.984 |
| L24 o c79 | H10 | a%100 in {1..3} (-) | 0.90 | 0.036 |
| L24 o c13 | H10 | a%100 in {3..10, 12..17, 19} (+) | 0.64 | 0.268 |
| L24 down c135 | MLP | a%100 in {90..98} (-) | 0.50 | 0.282 |

</details>

</details>

<details><summary><b>3a-279</b> `a%100` @ `b` (add) — single component; 1 comps, L25; tells apart 3/100 classes (best member 3); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.01 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.59 |
| decoding acc. joint / best code / best member (chance) | 0.02 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L25 attn |
| CKA(arrangement before, joint write) | 0.16 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.30 2:0.11 3:0.06) |
| joint write: shape (spectrum k:share) | line () |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.01 / 1.00 |

<details><summary>codes and components</summary>

**code 3a-279.0 (L25): 1 comps, tells apart 3/100, coverage 0.01, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L25 down c193 | MLP | a%100 in {1} (+) | 0.59 | 0.011 |

</details>

</details>

<details><summary><b>3a-280</b> `a%100` @ `b` (add) — single component; 1 comps, L28; tells apart 1/100 classes (best member 1); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.22 |
| classes told apart: joint / best code / best member | 1 /  / 1 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.69 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L27 MLP |
| CKA(arrangement before, joint write) | 0.34 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.21 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.30 2:0.11 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.49 3:0.11 20:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.13 / 1.00 |
| best source position (CKA) | `op` (0.45) |

<details><summary>codes and components</summary>

**code 3a-280.0 (L28): 1 comps, tells apart 1/100, coverage 0.22, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 o c122 | H10 | a%100 in {9, 31, 33..34, 36..39, 41..44, 46..49, 53, 65, 86, 95..97} (-) | 0.69 | 1.000 |

</details>

</details>

<details><summary><b>3a-281</b> `a%100` @ `b` (add) — copy from `op`; 1 comps, L28; tells apart 1/100 classes (best member 1); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.50 |
| classes told apart: joint / best code / best member | 1 /  / 1 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.69 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L27 MLP |
| CKA(arrangement before, joint write) | 0.49 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.30 2:0.11 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.53 20:0.13 40:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.15 / 1.00 |
| best source position (CKA) | `op` (0.59) |

<details><summary>codes and components</summary>

**code 3a-281.0 (L28): 1 comps, tells apart 1/100, coverage 0.50, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 o c95 | H4 | a%100 in {0..2, 5..6, 11, 18, 20, 25, 29, 32..34, 36..39, 41..44, 46..50, 52..53, 55, 59..60, 65, 69..70, 75, 80..81, 85, 88..99} (+) | 0.69 | 0.663 |

</details>

</details>

</details>

<details><summary>`b%10`: 6 mechanisms, 50 components</summary>

<details><summary><b>3a-290</b> `b%10` @ `b` (add) — tiling, 3 codes of the same shape; 26 comps, L0 L11 L13; tells apart 10/10 classes (best member 4); on sub: 3s-371 (member overlap 0.32)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 10 / 10 / 4 (of 10) |
| members whose removal merges classes | 0.04 |
| support overlap (1 = tiling) / random sets / p | 2.80 / 3.50 / 0.01 |
| mean CKA between its codes | 0.73 |
| purity of the joint write (per prompt) | 0.97 |
| decoding acc. joint / best code / best member (chance) | 1.00 / 1.00 / 0.33 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.10 |
| consumers / read jointly | 196 / 159 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 attn |
| CKA(arrangement before, joint write) | 0.79 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.18 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 24.21 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (20:0.27 10:0.26 40:0.19) |
| joint write: shape (spectrum k:share) | simplex (one-hot like) (10:0.27 20:0.24 30:0.21) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.72 / 6.28 |

<details><summary>codes and components</summary>

**code 3a-290.0 (L0): 10 comps, tells apart 10/10, coverage 1.00, overlap 1.00 (random 1.68, p 0.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 down c145 | MLP | b%10 in {0} (+) | 0.95 | 0.100 |
| L0 down c36 | MLP | b%10 in {1} (+) | 0.94 | 0.100 |
| L0 down c81 | MLP | b%10 in {2} (-) | 0.95 | 0.100 |
| L0 down c52 | MLP | b%10 in {3} (+) | 0.97 | 0.100 |
| L0 down c62 | MLP | b%10 in {4} (+) | 0.97 | 0.100 |
| L0 down c45 | MLP | b%10 in {5} (+) | 0.95 | 0.100 |
| L0 down c55 | MLP | b%10 in {6} (-) | 0.98 | 0.100 |
| L0 down c44 | MLP | b%10 in {7} (-) | 0.99 | 0.100 |
| L0 down c50 | MLP | b%10 in {8} (-) | 0.98 | 0.100 |
| L0 down c38 | MLP | b%10 in {9} (-) | 0.98 | 0.100 |

**code 3a-290.1 (L11): 8 comps, tells apart 9/10, coverage 0.90, overlap 1.00 (random 1.50, p 0.01)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L11 down c42 | MLP | b%10 in {0} (+) | 0.92 | 0.100 |
| L11 down c20 | MLP | b%10 in {1..2} (-) | 0.96 | 0.297 |
| L11 down c38 | MLP | b%10 in {3} (+) | 0.95 | 0.100 |
| L11 down c26 | MLP | b%10 in {4} (-) | 0.95 | 0.100 |
| L11 down c27 | MLP | b%10 in {5} (-) | 0.95 | 0.142 |
| L11 down c63 | MLP | b%10 in {6} (-) | 0.97 | 0.100 |
| L11 down c36 | MLP | b%10 in {7} (-) | 0.97 | 0.100 |
| L11 down c40 | MLP | b%10 in {8} (+) | 0.93 | 0.097 |

**code 3a-290.2 (L13): 8 comps, tells apart 9/10, coverage 0.90, overlap 1.00 (random 1.57, p 0.01)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 down c38 | MLP | b%10 in {0, 9} (+) | 0.98 | 0.200 |
| L13 down c105 | MLP | b%10 in {1} (+) | 0.97 | 0.099 |
| L13 down c92 | MLP | b%10 in {2} (-) | 0.97 | 0.100 |
| L13 down c15 | MLP | b%10 in {3} (+) | 0.97 | 0.100 |
| L13 down c51 | MLP | b%10 in {4} (-) | 0.94 | 0.100 |
| L13 down c13 | MLP | b%10 in {5} (+) | 0.98 | 0.101 |
| L13 down c9 | MLP | b%10 in {7} (-) | 0.99 | 0.100 |
| L13 down c11 | MLP | b%10 in {8} (+) | 0.98 | 0.100 |

</details>

</details>

<details><summary><b>3a-291</b> `b%10` @ `b` (add) — block code; 3 comps, L1; tells apart 4/10 classes (best member 2); on sub: 3s-372 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.30 |
| classes told apart: joint / best code / best member | 4 /  / 2 (of 10) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.62 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.94 |
| decoding acc. joint / best code / best member (chance) | 0.40 /  / 0.20 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 0.99 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L1 attn |
| CKA(arrangement before, joint write) | 0.63 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.11 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.09 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (20:0.30 10:0.30 50:0.15) |
| joint write: shape (spectrum k:share) | irregular (10:0.25 20:0.22 40:0.22) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.31 / 2.75 |

<details><summary>codes and components</summary>

**code 3a-291.0 (L1): 3 comps, tells apart 4/10, coverage 0.30, overlap 1.00 (random 1.00, p 0.58)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 down c329 | MLP | b%10 in {0} (-) | 0.94 | 0.100 |
| L1 down c401 | MLP | b%10 in {7} (+) | 0.92 | 0.102 |
| L1 down c356 | MLP | b%10 in {9} (+) | 0.95 | 0.100 |

</details>

</details>

<details><summary><b>3a-292</b> `b%10` @ `b` (add) — block code, 2 codes of the same shape; 9 comps, L3 L12; tells apart 7/10 classes (best member 4); on sub: 3s-376 (member overlap 0.44)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.90 |
| classes told apart: joint / best code / best member | 7 / 10 / 4 (of 10) |
| members whose removal merges classes | 0.22 |
| support overlap (1 = tiling) / random sets / p | 2.00 / 1.62 / 0.95 |
| mean CKA between its codes | 0.79 |
| purity of the joint write (per prompt) | 0.96 |
| decoding acc. joint / best code / best member (chance) | 0.95 / 0.95 / 0.31 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.02 |
| consumers / read jointly | 76 / 60 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L3 attn |
| CKA(arrangement before, joint write) | 0.69 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.24 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 1.11 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (20:0.30 10:0.28 40:0.18) |
| joint write: shape (spectrum k:share) | irregular (10:0.34 20:0.28 40:0.15) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.50 / 3.96 |

<details><summary>codes and components</summary>

**code 3a-292.0 (L3): 5 comps, tells apart 10/10, coverage 0.90, overlap 1.33 (random 1.25, p 0.64)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 down c615 | MLP | b%10 in {0} (+) | 0.94 | 0.191 |
| L3 down c54 | MLP | b%10 in {1, 9} (+) | 0.94 | 0.200 |
| L3 down c68 | MLP | b%10 in {3..4, 9} (-) | 0.94 | 0.298 |
| L3 down c63 | MLP | b%10 in {5..7} (+) | 0.94 | 0.297 |
| L3 down c136 | MLP | b%10 in {7..9} (-) | 0.90 | 0.314 |

**code 3a-292.1 (L12): 4 comps, tells apart 6/10, coverage 0.60, overlap 1.00 (random 1.20, p 0.38)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L12 down c19 | MLP | b%10 in {0} (+) | 0.95 | 0.100 |
| L12 down c15 | MLP | b%10 in {1} (+) | 0.97 | 0.100 |
| L12 down c31 | MLP | b%10 in {5..7} (-) | 0.96 | 0.298 |
| L12 down c7 | MLP | b%10 in {9} (-) | 0.98 | 0.100 |

</details>

</details>

<details><summary><b>3a-293</b> `b%10` @ `b` (add) — block code; 2 comps, L4; tells apart 6/10 classes (best member 4); on sub: 3s-374 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.60 |
| classes told apart: joint / best code / best member | 6 /  / 4 (of 10) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.82 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.91 |
| decoding acc. joint / best code / best member (chance) | 0.45 /  / 0.30 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 0.90 |
| consumers / read jointly | 7 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L4 attn |
| CKA(arrangement before, joint write) | 0.66 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.15 |
| share of the write inside the old arrangement's span | 0.09 |
| write energy / code energy before | 0.06 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (20:0.31 10:0.29 40:0.17) |
| joint write: shape (spectrum k:share) | irregular (10:0.49 20:0.28 40:0.21) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.35 / 1.92 |

<details><summary>codes and components</summary>

**code 3a-293.0 (L4): 2 comps, tells apart 6/10, coverage 0.60, overlap 1.00 (random 1.00, p 0.90)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 down c43 | MLP | b%10 in {0, 5} (-) | 0.89 | 0.238 |
| L4 down c66 | MLP | b%10 in {6..9} (-) | 0.94 | 0.385 |

</details>

</details>

<details><summary><b>3a-294</b> `b%10` @ `b` (add) — block code; 6 comps, L14; tells apart 7/10 classes (best member 2); on sub: 3s-378 (member overlap 0.83)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.60 |
| classes told apart: joint / best code / best member | 7 /  / 2 (of 10) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.33 / 0.11 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.96 |
| decoding acc. joint / best code / best member (chance) | 0.70 /  / 0.20 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 0.95 |
| consumers / read jointly | 59 / 12 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L14 attn |
| CKA(arrangement before, joint write) | 0.63 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.16 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.13 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (10:0.32 20:0.28 50:0.15) |
| joint write: shape (spectrum k:share) | irregular (10:0.24 40:0.23 30:0.22) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.59 / 5.29 |

<details><summary>codes and components</summary>

**code 3a-294.0 (L14): 6 comps, tells apart 7/10, coverage 0.60, overlap 1.00 (random 1.33, p 0.10)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L14 down c71 | MLP | b%10 in {1} (-) | 0.95 | 0.100 |
| L14 down c15 | MLP | b%10 in {2} (-) | 0.97 | 0.099 |
| L14 down c14 | MLP | b%10 in {4} (-) | 0.96 | 0.099 |
| L14 down c226 | MLP | b%10 in {5} (+) | 0.97 | 0.100 |
| L14 down c33 | MLP | b%10 in {6} (-) | 0.96 | 0.100 |
| L14 down c21 | MLP | b%10 in {9} (+) | 0.98 | 0.100 |

</details>

</details>

<details><summary><b>3a-295</b> `b%10` @ `b` (add) — block code, 2 codes of the same shape; 4 comps, L17 L21; tells apart 4/10 classes (best member 3); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.30 |
| classes told apart: joint / best code / best member | 4 / 4 / 3 (of 10) |
| members whose removal merges classes | 0.25 |
| support overlap (1 = tiling) / random sets / p | 1.67 / 1.20 / 0.96 |
| mean CKA between its codes | 0.83 |
| purity of the joint write (per prompt) | 0.90 |
| decoding acc. joint / best code / best member (chance) | 0.38 / 0.39 / 0.24 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.11 |
| consumers / read jointly | 32 / 24 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L17 attn |
| CKA(arrangement before, joint write) | 0.60 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.21 |
| share of the write inside the old arrangement's span | 0.13 |
| write energy / code energy before | 0.11 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (20:0.28 10:0.26 40:0.20) |
| joint write: shape (spectrum k:share) | irregular (20:0.36 40:0.35 10:0.12) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.22 / 1.57 |

<details><summary>codes and components</summary>

**code 3a-295.0 (L17): 1 comps, tells apart 3/10, coverage 0.20, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 down c8 | MLP | b%10 in {0, 5} (+) | 0.86 | 0.187 |

**code 3a-295.1 (L21): 3 comps, tells apart 4/10, coverage 0.30, overlap 1.00 (random 1.00, p 0.67)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c19 | MLP | b%10 in {0} (-) | 0.94 | 0.108 |
| L21 down c81 | MLP | b%10 in {3} (-) | 0.88 | 0.092 |
| L21 down c194 | MLP | b%10 in {5} (-) | 0.90 | 0.103 |

</details>

</details>

</details>

<details><summary>`a//10`: 5 mechanisms, 21 components</summary>

<details><summary><b>3a-287</b> `a//10` @ `b` (add) — single component; 1 comps, L1; tells apart 3/11 classes (best member 3); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.45 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 11) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.80 |
| decoding acc. joint / best code / best member (chance) | 0.38 /  / 0.38 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 6 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.47 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 3.87 |
| best source position (CKA) | `a` (0.48) |

<details><summary>codes and components</summary>

**code 3a-287.0 (L1): 1 comps, tells apart 3/11, coverage 0.45, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c491 | H5 | a//10 in {0..1, 4, 9..10} (+) | 0.80 | 1.000 |

</details>

</details>

<details><summary><b>3a-288</b> `a//10` @ `b` (add) — 3 codes of the same shape; 3 comps, L1 L4 L12; tells apart 4/11 classes (best member 4); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.27 |
| classes told apart: joint / best code / best member | 4 / 4 / 4 (of 11) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 2.33 / 1.44 / 0.98 |
| mean CKA between its codes | 0.89 |
| purity of the joint write (per prompt) | 0.72 |
| decoding acc. joint / best code / best member (chance) | 0.26 / 0.27 / 0.27 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 0.97 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L1 attn |
| CKA(arrangement before, joint write) | 0.77 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.04 |
| share of the write inside the old arrangement's span | 0.00 |
| write energy / code energy before | 2.50 |

<details><summary>codes and components</summary>

**code 3a-288.0 (L1): 1 comps, tells apart 3/11, coverage 0.27, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 down c86 | MLP | a//10 in {0..2} (+) | 0.90 | 0.339 |

**code 3a-288.1 (L4): 1 comps, tells apart 4/11, coverage 0.18, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 down c90 | MLP | a//10 in {0..1} (+) | 0.91 | 0.280 |

**code 3a-288.2 (L12): 1 comps, tells apart 4/11, coverage 0.18, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L12 down c669 | MLP | a//10 in {0..1} (-) | 0.58 | 0.277 |

</details>

</details>

<details><summary><b>3a-285</b> `a//10` @ `b` (add) — block code, 3 codes of the same shape, copy from `a`; 6 comps, L1 L3 L5; tells apart 9/11 classes (best member 6); on sub: 3s-368 (member overlap 0.27)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 9 / 8 / 6 (of 11) |
| members whose removal merges classes | 0.67 |
| support overlap (1 = tiling) / random sets / p | 3.09 / 2.00 / 1.00 |
| mean CKA between its codes | 0.87 |
| purity of the joint write (per prompt) | 0.81 |
| decoding acc. joint / best code / best member (chance) | 0.74 / 0.65 / 0.45 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 0.89 |
| consumers / read jointly | 38 / 32 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.57 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 114.20 |
| best source position (CKA) | `a` (0.82) |

<details><summary>codes and components</summary>

**code 3a-285.0 (L1): 2 comps, tells apart 6/11, coverage 0.64, overlap 1.57 (random 1.14, p 0.86)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c465 | H24 | a//10 in {0..3, 7..9} (+) | 0.94 | 0.619 |
| L1 o c463 | H24 | a//10 in {0..3} (+) | 0.93 | 0.485 |

**code 3a-285.1 (L3): 1 comps, tells apart 5/11, coverage 0.36, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 o c354 | H5 | a//10 in {0..3} (-) | 0.91 | 0.620 |

**code 3a-285.2 (L5): 3 comps, tells apart 8/11, coverage 1.00, overlap 1.73 (random 1.44, p 0.81)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 o c16 | H22 | a//10 in {0..4, 6..8} (-) | 0.81 | 0.993 |
| L5 o c27 | H22 | a//10 in {1..5, 8..10} (+) | 0.83 | 0.704 |
| L5 o c17 | H22 | a//10 in {8..10} (+) | 0.85 | 0.242 |

</details>

</details>

<details><summary><b>3a-286</b> `a//10` @ `b` (add) — block code, 6 codes of the same shape, copy from `op`; 9 comps, L1 L2 L3 L6 L7 L16 L24; tells apart 5/11 classes (best member 5); on sub: 3s-369 (member overlap 0.12)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.45 |
| classes told apart: joint / best code / best member | 5 / 6 / 5 (of 11) |
| members whose removal merges classes | 0.22 |
| support overlap (1 = tiling) / random sets / p | 2.20 / 2.85 / 0.04 |
| mean CKA between its codes | 0.95 |
| purity of the joint write (per prompt) | 0.81 |
| decoding acc. joint / best code / best member (chance) | 0.50 / 0.38 / 0.26 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.13 |
| consumers / read jointly | 13 / 10 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.85 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.08 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 199.27 |
| best source position (CKA) | `op` (0.81) |

<details><summary>codes and components</summary>

**code 3a-286.0 (L1): 2 comps, tells apart 4/11, coverage 0.27, overlap 1.00 (random 1.25, p 0.34)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c377 | H26 | a//10 in {0} (-) | 0.92 | 0.139 |
| L1 o c413 | H26 | a//10 in {1..2} (+) | 0.78 | 0.277 |

**code 3a-286.1 (L2 L3): 2 comps, tells apart 4/11, coverage 0.18, overlap 1.00 (random 1.25, p 0.34)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 o c184 | H17 | a//10 in {0} (+) | 0.90 | 0.213 |
| L3 o c104 | H7 | a//10 in {10} (-) | 0.86 | 0.013 |

**code 3a-286.2 (L3): 1 comps, tells apart 2/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 down c300 | MLP | a//10 in {0} (-) | 0.83 | 0.106 |

**code 3a-286.3 (L6 L7): 2 comps, tells apart 6/11, coverage 0.27, overlap 1.00 (random 1.25, p 0.36)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L6 o c192 | H8 | a//10 in {9..10} (-) | 0.80 | 0.173 |
| L7 down c551 | MLP | a//10 in {0} (+) | 0.78 | 0.149 |

**code 3a-286.4 (L16): 1 comps, tells apart 3/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c61 | H22 | a//10 in {0} (+) | 0.88 | 0.152 |

**code 3a-286.5 (L24): 1 comps, tells apart 3/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 o c0 | H10 | a//10 in {0} (-) | 0.76 | 0.238 |

</details>

</details>

<details><summary><b>3a-289</b> `a//10` @ `b` (add) — 2 codes of the same shape, copy from `op`; 2 comps, L9 L22; tells apart 4/11 classes (best member 3); on sub: 3s-369 (member overlap 0.09)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.18 |
| classes told apart: joint / best code / best member | 4 / 3 / 3 (of 11) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.50 / 1.22 / 0.86 |
| mean CKA between its codes | 0.93 |
| purity of the joint write (per prompt) | 0.82 |
| decoding acc. joint / best code / best member (chance) | 0.26 / 0.25 / 0.25 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 0.99 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L9 attn |
| CKA(arrangement before, joint write) | 0.45 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.06 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.06 |
| best source position (CKA) | `op` (0.59) |

<details><summary>codes and components</summary>

**code 3a-289.0 (L9): 1 comps, tells apart 3/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L9 down c118 | MLP | a//10 in {9} (+) | 0.76 | 0.130 |

**code 3a-289.1 (L22): 1 comps, tells apart 3/11, coverage 0.18, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 o c10 | H15 | a//10 in {9..10} (-) | 0.91 | 0.124 |

</details>

</details>

</details>

<details><summary>`units(a,b)`: 3 mechanisms, 11 components</summary>

<details><summary><b>3a-350</b> `units(a,b)` @ `b` (add) — 8 codes of the same shape; 8 comps, L1 L2 L6 L19 L25 L26 L28 L29; tells apart 7/100 classes (best member 8); on sub: 3s-435 (member overlap 0.20)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.03 |
| classes told apart: joint / best code / best member | 7 / 8 / 8 (of 100) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 7.00 / 1.64 / 1.00 |
| mean CKA between its codes | 0.89 |
| purity of the joint write (per prompt) | 0.60 |
| decoding acc. joint / best code / best member (chance) | 0.03 / 0.03 / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.19 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L1 attn |
| CKA(arrangement before, joint write) | 0.26 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.98 |

<details><summary>codes and components</summary>

**code 3a-350.0 (L1): 1 comps, tells apart 4/100, coverage 0.03, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 down c368 | MLP | a%10 {0} x b%10 {0, 5}; a%10 {5} x b%10 {0} (+) | 0.76 | 0.047 |

**code 3a-350.1 (L2): 1 comps, tells apart 5/100, coverage 0.02, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 down c35 | MLP | a%10 {0, 5} x b%10 {0} (-) | 0.67 | 0.067 |

**code 3a-350.2 (L6): 1 comps, tells apart 6/100, coverage 0.02, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L6 down c244 | MLP | a%10 {0, 5} x b%10 {0} (-) | 0.59 | 0.046 |

**code 3a-350.3 (L19): 1 comps, tells apart 7/100, coverage 0.03, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L19 down c47 | MLP | a%10 {0} x b%10 {0, 5}; a%10 {5} x b%10 {0} (-) | 0.61 | 0.057 |

**code 3a-350.4 (L25): 1 comps, tells apart 6/100, coverage 0.03, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L25 down c119 | MLP | a%10 {0} x b%10 {0, 5}; a%10 {5} x b%10 {0} (-) | 0.55 | 0.027 |

**code 3a-350.5 (L26): 1 comps, tells apart 8/100, coverage 0.03, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 down c109 | MLP | a%10 {0} x b%10 {0, 5}; a%10 {5} x b%10 {0} (-) | 0.60 | 0.075 |

**code 3a-350.6 (L28): 1 comps, tells apart 6/100, coverage 0.03, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 down c17 | MLP | a%10 {0} x b%10 {0, 5}; a%10 {5} x b%10 {0} (-) | 0.54 | 0.030 |

**code 3a-350.7 (L29): 1 comps, tells apart 4/100, coverage 0.02, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L29 down c877 | MLP | a%10 {0, 5} x b%10 {0} (-) | 0.51 | 0.024 |

</details>

</details>

<details><summary><b>3a-351</b> `units(a,b)` @ `b` (add) — single component; 1 comps, L3; tells apart 7/100 classes (best member 7); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.04 |
| classes told apart: joint / best code / best member | 7 /  / 7 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.57 |
| decoding acc. joint / best code / best member (chance) | 0.03 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L3 attn |
| CKA(arrangement before, joint write) | 0.08 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.00 |

<details><summary>codes and components</summary>

**code 3a-351.0 (L3): 1 comps, tells apart 7/100, coverage 0.04, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 down c111 | MLP | a%10 {0, 5} x b%10 {0, 5} (+) | 0.56 | 0.069 |

</details>

</details>

<details><summary><b>3a-352</b> `units(a,b)` @ `b` (add) — block code; 2 comps, L5; tells apart 5/100 classes (best member 5); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.28 |
| classes told apart: joint / best code / best member | 5 /  / 5 (of 100) |
| members whose removal merges classes | 0.50 |
| support overlap (1 = tiling) / random sets / p | 1.04 / 1.67 / 0.10 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.79 |
| decoding acc. joint / best code / best member (chance) | 0.05 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 8 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L5 attn |
| CKA(arrangement before, joint write) | 0.21 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.06 |
| share of the write inside the old arrangement's span | 0.29 |
| write energy / code energy before | 0.03 |

<details><summary>codes and components</summary>

**code 3a-352.0 (L5): 2 comps, tells apart 5/100, coverage 0.28, overlap 1.04 (random 1.67, p 0.07)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 down c91 | MLP | a%10 {0, 2, 4, 6, 8} x b%10 {1, 3, 5, 7, 9} (-) | 0.79 | 0.295 |
| L5 down c62 | MLP | a%10 {0, 5} x b%10 {0, 5} (+) | 0.85 | 0.041 |

</details>

</details>

</details>

<details><summary>`b%50`: 5 mechanisms, 9 components</summary>

<details><summary><b>3a-322</b> `b%50` @ `b` (add) — single component; 1 comps, L2; tells apart 3/50 classes (best member 3); on sub: 3s-401 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.02 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 50) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.96 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.04 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 attn |
| CKA(arrangement before, joint write) | 0.10 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.20 20:0.11 10:0.11) |
| joint write: shape (spectrum k:share) | line () |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.02 / 1.00 |

<details><summary>codes and components</summary>

**code 3a-322.0 (L2): 1 comps, tells apart 3/50, coverage 0.02, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 down c98 | MLP | b%50 in {25} (-) | 0.96 | 0.025 |

</details>

</details>

<details><summary><b>3a-323</b> `b%50` @ `b` (add) — block code, 3 codes of the same shape; 4 comps, L5 L14 L30; tells apart 8/50 classes (best member 7); on sub: 3s-403 (member overlap 0.29)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.12 |
| classes told apart: joint / best code / best member | 8 / 7 / 7 (of 50) |
| members whose removal merges classes | 0.75 |
| support overlap (1 = tiling) / random sets / p | 1.83 / 1.17 / 0.99 |
| mean CKA between its codes | 0.86 |
| purity of the joint write (per prompt) | 0.71 |
| decoding acc. joint / best code / best member (chance) | 0.10 / 0.10 / 0.06 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.02 |
| consumers / read jointly | 3 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L5 attn |
| CKA(arrangement before, joint write) | 0.31 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.11 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.21 20:0.11 10:0.10) |
| joint write: shape (spectrum k:share) | line (20:0.25 40:0.25 10:0.12) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.10 / 1.25 |

<details><summary>codes and components</summary>

**code 3a-323.0 (L5): 1 comps, tells apart 5/50, coverage 0.06, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 down c28 | MLP | b%50 in {0, 30, 40} (+) | 0.67 | 0.155 |

**code 3a-323.1 (L14): 2 comps, tells apart 7/50, coverage 0.12, overlap 1.00 (random 1.00, p 0.77)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L14 down c18 | MLP | b%50 in {0, 10, 20, 30, 40} (+) | 0.87 | 0.148 |
| L14 down c448 | MLP | b%50 in {25} (-) | 0.86 | 0.062 |

**code 3a-323.2 (L30): 1 comps, tells apart 7/50, coverage 0.04, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 down c994 | MLP | b%50 in {0, 30} (+) | 0.63 | 0.091 |

</details>

</details>

<details><summary><b>3a-324</b> `b%50` @ `b` (add) — block code; 2 comps, L16; tells apart 8/50 classes (best member 7); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.12 |
| classes told apart: joint / best code / best member | 8 /  / 7 (of 50) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.77 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.71 |
| decoding acc. joint / best code / best member (chance) | 0.06 /  / 0.05 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 2 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L16 attn |
| CKA(arrangement before, joint write) | 0.19 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.24 10:0.12 20:0.12) |
| joint write: shape (spectrum k:share) | line (40:0.33 20:0.32 30:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.13 / 1.16 |

<details><summary>codes and components</summary>

**code 3a-324.0 (L16): 2 comps, tells apart 8/50, coverage 0.12, overlap 1.00 (random 1.00, p 0.74)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c120 | MLP | b%50 in {0} (-) | 0.49 | 0.028 |
| L16 down c50 | MLP | b%50 in {1, 11, 21, 31, 41} (+) | 0.74 | 0.122 |

</details>

</details>

<details><summary><b>3a-325</b> `b%50` @ `b` (add) — single component; 1 comps, L20; tells apart 5/50 classes (best member 5); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.08 |
| classes told apart: joint / best code / best member | 5 /  / 5 (of 50) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.87 |
| decoding acc. joint / best code / best member (chance) | 0.06 /  / 0.06 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L20 attn |
| CKA(arrangement before, joint write) | 0.13 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.07 |
| share of the write inside the old arrangement's span | 0.20 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.24 20:0.12 10:0.10) |
| joint write: shape (spectrum k:share) | line (40:0.21 10:0.21 20:0.21) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.10 / 1.00 |

<details><summary>codes and components</summary>

**code 3a-325.0 (L20): 1 comps, tells apart 5/50, coverage 0.08, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 down c41 | MLP | b%50 in {2, 22, 32, 42} (-) | 0.87 | 0.086 |

</details>

</details>

<details><summary><b>3a-326</b> `b%50` @ `b` (add) — single component; 1 comps, L27; tells apart 5/50 classes (best member 5); on sub: 3s-405 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.16 |
| classes told apart: joint / best code / best member | 5 /  / 5 (of 50) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.78 |
| decoding acc. joint / best code / best member (chance) | 0.05 /  / 0.05 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L27 attn |
| CKA(arrangement before, joint write) | 0.15 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.23 20:0.11 10:0.10) |
| joint write: shape (spectrum k:share) | line (10:0.39 20:0.29 30:0.15) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.13 / 1.00 |

<details><summary>codes and components</summary>

**code 3a-326.0 (L27): 1 comps, tells apart 5/50, coverage 0.16, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 down c5 | MLP | b%50 in {2..3, 22..23, 32..33, 42..43} (-) | 0.78 | 0.155 |

</details>

</details>

</details>

<details><summary>`a%50`: 2 mechanisms, 3 components</summary>

<details><summary><b>3a-283</b> `a%50` @ `b` (add) — single component; 1 comps, L3; tells apart 4/50 classes (best member 4); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.02 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 50) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.57 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.04 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 MLP |
| CKA(arrangement before, joint write) | 0.17 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (2:0.31 4:0.12 20:0.11) |
| joint write: shape (spectrum k:share) | line (10:0.12 40:0.12 20:0.12) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.04 / 1.00 |
| best source position (CKA) | `a` (0.21) |

<details><summary>codes and components</summary>

**code 3a-283.0 (L3): 1 comps, tells apart 4/50, coverage 0.02, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 o c27 | H7 | a%50 in {1} (+) | 0.57 | 0.038 |

</details>

</details>

<details><summary><b>3a-284</b> `a%50` @ `b` (add) — 2 codes of the same shape, copy from `op`; 2 comps, L5 L16; tells apart 7/50 classes (best member 6); on sub: 3s-367 (member overlap 0.33)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.10 |
| classes told apart: joint / best code / best member | 7 / 6 / 6 (of 50) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.60 / 1.00 / 1.00 |
| mean CKA between its codes | 0.84 |
| purity of the joint write (per prompt) | 0.72 |
| decoding acc. joint / best code / best member (chance) | 0.07 / 0.05 / 0.05 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.04 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L5 attn |
| CKA(arrangement before, joint write) | 0.41 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.03 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.03 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.32 20:0.11 10:0.11) |
| joint write: shape (spectrum k:share) | line (20:0.32 40:0.31 10:0.12) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.12 / 1.06 |
| best source position (CKA) | `op` (0.60) |

<details><summary>codes and components</summary>

**code 3a-284.0 (L5): 1 comps, tells apart 5/50, coverage 0.08, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 down c36 | MLP | a%50 in {0, 20, 30, 40} (-) | 0.72 | 0.088 |

**code 3a-284.1 (L16): 1 comps, tells apart 6/50, coverage 0.08, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c197 | H22 | a%50 in {0, 10, 30, 40} (+) | 0.71 | 0.174 |

</details>

</details>

</details>

<details><summary>`b%20`: 1 mechanisms, 3 components</summary>

<details><summary><b>3a-319</b> `b%20` @ `b` (add) — block code; 3 comps, L12; tells apart 10/20 classes (best member 5); on sub: 3s-398 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.55 |
| classes told apart: joint / best code / best member | 10 /  / 5 (of 20) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.09 /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.96 |
| decoding acc. joint / best code / best member (chance) | 0.46 /  / 0.19 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 14 / 12 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L12 attn |
| CKA(arrangement before, joint write) | 0.36 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.11 |
| share of the write inside the old arrangement's span | 0.09 |
| write energy / code energy before | 0.11 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (10:0.23 20:0.22 5:0.15) |
| joint write: shape (spectrum k:share) | irregular (5:0.40 10:0.34 15:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.29 / 1.98 |

<details><summary>codes and components</summary>

**code 3a-319.0 (L12): 3 comps, tells apart 10/20, coverage 0.55, overlap 1.09 (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L12 down c9 | MLP | b%20 in {10..14} (-) | 0.97 | 0.250 |
| L12 down c73 | MLP | b%20 in {15..17} (-) | 0.94 | 0.246 |
| L12 down c32 | MLP | b%20 in {2..4, 13} (+) | 0.94 | 0.332 |

</details>

</details>

</details>

<details><summary>`b%25`: 1 mechanisms, 3 components</summary>

<details><summary><b>3a-320</b> `b%25` @ `b` (add) — 3 codes of the same shape; 3 comps, L0 L15 L20; tells apart 4/25 classes (best member 4); on sub: 3s-400 (member overlap 0.33)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.16 |
| classes told apart: joint / best code / best member | 4 / 4 / 4 (of 25) |
| members whose removal merges classes | 0.33 |
| support overlap (1 = tiling) / random sets / p | 1.50 /  /  |
| mean CKA between its codes | 0.93 |
| purity of the joint write (per prompt) | 0.66 |
| decoding acc. joint / best code / best member (chance) | 0.10 / 0.12 / 0.12 (0.04) |
| kappa (1 orthogonal, >1 constructive) | 1.08 |
| consumers / read jointly | 3 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 attn |
| CKA(arrangement before, joint write) | 0.42 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 1.08 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (20:0.17 4:0.16 40:0.12) |
| joint write: shape (spectrum k:share) | line (20:0.34 40:0.34) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.13 / 1.02 |

<details><summary>codes and components</summary>

**code 3a-320.0 (L0): 1 comps, tells apart 3/25, coverage 0.16, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 down c162 | MLP | b%25 in {0, 5, 10, 15} (-) | 0.95 | 0.200 |

**code 3a-320.1 (L15): 1 comps, tells apart 3/25, coverage 0.04, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c70 | MLP | b%25 in {0} (+) | 0.69 | 0.129 |

**code 3a-320.2 (L20): 1 comps, tells apart 4/25, coverage 0.04, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 down c85 | MLP | b%25 in {0} (+) | 0.59 | 0.092 |

</details>

</details>

</details>

<details><summary>`a%10`: 1 mechanisms, 1 components</summary>

<details><summary><b>3a-262</b> `a%10` @ `b` (add) — copy from `op`; 1 comps, L1; tells apart 2/10 classes (best member 2); on sub: 3s-353 (member overlap 0.50)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.10 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 10) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.88 |
| decoding acc. joint / best code / best member (chance) | 0.20 /  / 0.20 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 2 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.83 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.16 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 1.83 |
| arrangement before: shape (spectrum k:share) | irregular (20:0.33 40:0.28 10:0.16) |
| joint write: shape (spectrum k:share) | line (40:0.23 20:0.23 30:0.22) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.11 / 1.00 |
| best source position (CKA) | `op` (0.71) |

<details><summary>codes and components</summary>

**code 3a-262.0 (L1): 1 comps, tells apart 2/10, coverage 0.10, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c110 | H26 | a%10 in {0} (-) | 0.88 | 0.101 |

</details>

</details>

</details>

<details><summary>`a%20`: 1 mechanisms, 1 components</summary>

<details><summary><b>3a-282</b> `a%20` @ `b` (add) — copy from `op`; 1 comps, L5; tells apart 4/20 classes (best member 4); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.10 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 20) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.85 |
| decoding acc. joint / best code / best member (chance) | 0.13 /  / 0.13 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 3 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L4 MLP |
| CKA(arrangement before, joint write) | 0.71 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.15 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.07 |
| arrangement before: shape (spectrum k:share) | irregular (20:0.24 40:0.19 10:0.18) |
| joint write: shape (spectrum k:share) | line (40:0.32 20:0.30 10:0.15) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.12 / 1.00 |
| best source position (CKA) | `op` (0.82) |

<details><summary>codes and components</summary>

**code 3a-282.0 (L5): 1 comps, tells apart 4/20, coverage 0.10, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 o c303 | H22 | a%20 in {0, 10} (+) | 0.85 | 0.161 |

</details>

</details>

</details>

<details><summary>`b%5`: 1 mechanisms, 1 components</summary>

<details><summary><b>3a-321</b> `b%5` @ `b` (add) — single component; 1 comps, L2; tells apart 2/5 classes (best member 2); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.20 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 5) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.94 |
| decoding acc. joint / best code / best member (chance) | 0.40 /  / 0.40 (0.20) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 4 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 attn |
| CKA(arrangement before, joint write) | 0.73 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.19 |
| share of the write inside the old arrangement's span | 0.09 |
| write energy / code energy before | 0.08 |
| arrangement before: shape (spectrum k:share) | circle period 5 (20:0.67 40:0.33) |
| joint write: shape (spectrum k:share) | line (20:0.50 40:0.50) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.25 / 1.00 |

<details><summary>codes and components</summary>

**code 3a-321.0 (L2): 1 comps, tells apart 2/5, coverage 0.20, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 down c473 | MLP | b%5 in {0} (+) | 0.94 | 0.200 |

</details>

</details>

</details>
