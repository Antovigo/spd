[← back to the report](../report_mechanisms.md)

# Every mechanism at `op`, add

<details><summary>`a%100`: 40 mechanisms, 687 components</summary>

<details><summary><b>2a-151</b> `a%100` @ `op` (add) — block code; 3 comps, L0; tells apart 11/100 classes (best member 6); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.13 |
| classes told apart: joint / best code / best member | 11 /  / 6 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.31 / 1.07 / 0.82 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.27 /  / 0.21 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.13 |
| consumers / read jointly | 7 / 7 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | embed |
| CKA(arrangement before, joint write) | 0.00 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.00 |
| write energy / code energy before | 117101445164280431532572672.00 |
| arrangement before: shape (spectrum k:share) | line () |
| joint write: shape (spectrum k:share) | irregular (1:0.26 3:0.11 2:0.09) |
| frequencies new in the write | 1 3 |
| write: shift-symmetric part / dimension (PR) | 0.08 / 1.67 |
| best source position (CKA) | `a` (0.28) |

<details><summary>codes and components</summary>

**code 2a-151.0 (L0): 3 comps, tells apart 11/100, coverage 0.13, overlap 1.31 (random 1.06, p 0.87)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 o c2 | H2 | a%100 in {1..9} (-) | 1.00 | 1.000 |
| L0 o c30 | H24 | a%100 in {18, 55, 65, 90} (+) | 1.00 | 0.050 |
| L0 o c268 | H24 | a%100 in {18, 55, 65, 90} (+) | 1.00 | 0.080 |

</details>

</details>

<details><summary><b>2a-152</b> `a%100` @ `op` (add) — block code; 4 comps, L0; tells apart 18/100 classes (best member 8); on sub: 2s-200 (member overlap 0.17)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.20 |
| classes told apart: joint / best code / best member | 18 /  / 8 (of 100) |
| members whose removal merges classes | 0.75 |
| support overlap (1 = tiling) / random sets / p | 1.45 / 1.14 / 0.90 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 1.00 /  / 0.27 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.84 |
| consumers / read jointly | 9 / 9 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 attn |
| CKA(arrangement before, joint write) | 0.41 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.14 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.18 2:0.08 40:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.12 2:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.03 / 1.41 |

<details><summary>codes and components</summary>

**code 2a-152.0 (L0): 4 comps, tells apart 18/100, coverage 0.20, overlap 1.45 (random 1.14, p 0.86)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 down c16 | MLP | a%100 in {1..8, 55, 65, 90} (-) | 1.00 | 1.000 |
| L0 down c46 | MLP | a%100 in {18, 55, 65, 90} (-) | 1.00 | 1.000 |
| L0 down c11 | MLP | a%100 in {3, 5, 10, 18, 20, 30, 40, 50, 60, 70, 80} (+) | 1.00 | 1.000 |
| L0 down c21 | MLP | a%100 in {5, 55, 90} (+) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2a-154</b> `a%100` @ `op` (add) — block code; 4 comps, L1 L2; tells apart 13/100 classes (best member 4); on sub: 2s-202 (member overlap 0.25)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.04 |
| classes told apart: joint / best code / best member | 13 /  / 4 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.25 / 1.10 / 0.76 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.25 /  / 0.15 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.16 |
| consumers / read jointly | 5 / 1 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.18 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.27 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.19 2:0.09 40:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.07 40:0.07 10:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.04 / 2.03 |
| best source position (CKA) | `a` (0.19) |

<details><summary>codes and components</summary>

**code 2a-154.0 (L1 L2): 4 comps, tells apart 13/100, coverage 0.04, overlap 1.25 (random 1.11, p 0.74)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c416 | H18 | a%100 in {1} (-) | 1.00 | 0.151 |
| L1 o c5 | H18 | a%100 in {90} (-) | 1.00 | 0.010 |
| L2 o c26 | H8 | a%100 in {11, 18} (-) | 1.00 | 0.100 |
| L2 o c506 | H8 | a%100 in {18} (-) | 1.00 | 0.050 |

</details>

</details>

<details><summary><b>2a-153</b> `a%100` @ `op` (add) — tiling; 6 comps, L1 L2; tells apart 8/100 classes (best member 5); on sub: 2s-201 (member overlap 0.30)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.50 |
| classes told apart: joint / best code / best member | 8 /  / 5 (of 100) |
| members whose removal merges classes | 0.67 |
| support overlap (1 = tiling) / random sets / p | 1.02 / 1.27 / 0.02 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.98 |
| decoding acc. joint / best code / best member (chance) | 0.87 /  / 0.70 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.11 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.44 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.19 2:0.09 40:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.56 2:0.11) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.23 / 1.42 |
| best source position (CKA) | `a` (0.35) |

<details><summary>codes and components</summary>

**code 2a-153.0 (L1 L2): 6 comps, tells apart 8/100, coverage 0.50, overlap 1.02 (random 1.27, p 0.01)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c10 | H11 | a%100 in {0..5, 7..10, 15, 64, 90, 99} (+) | 1.00 | 1.000 |
| L2 o c228 | H24 | a%100 in {18} (+) | 1.00 | 0.030 |
| L2 o c396 | H24 | a%100 in {21} (+) | 1.00 | 0.010 |
| L2 o c93 | H24 | a%100 in {22..39, 41..49, 52..54, 56, 58..59} (+) | 0.98 | 0.328 |
| L2 o c389 | H24 | a%100 in {70} (+) | 1.00 | 0.010 |
| L2 o c337 | H24 | a%100 in {99} (-) | 1.00 | 0.040 |

</details>

</details>

<details><summary><b>2a-156</b> `a%100` @ `op` (add) — block code; 11 comps, L1; tells apart 29/100 classes (best member 8); on sub: 2s-252 (member overlap 0.21)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.51 |
| classes told apart: joint / best code / best member | 29 /  / 8 (of 100) |
| members whose removal merges classes | 0.45 |
| support overlap (1 = tiling) / random sets / p | 2.06 / 1.55 / 0.90 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.99 /  / 0.79 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.97 |
| consumers / read jointly | 29 / 25 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L1 attn |
| CKA(arrangement before, joint write) | 0.75 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.54 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.26 2:0.12 20:0.08) |
| joint write: shape (spectrum k:share) | irregular (1:0.31 2:0.17 3:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.22 / 2.95 |

<details><summary>codes and components</summary>

**code 2a-156.0 (L1): 11 comps, tells apart 29/100, coverage 0.51, overlap 2.06 (random 1.59, p 0.91)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 down c27 | MLP | a%100 in {0, 80, 90..93, 95..96} (+) | 1.00 | 1.000 |
| L1 down c25 | MLP | a%100 in {0, 9, 20, 25, 30, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90} (-) | 1.00 | 1.000 |
| L1 down c15 | MLP | a%100 in {0..4, 6..9, 11, 30, 40, 50, 60, 65, 70, 75, 80, 90} (-) | 1.00 | 0.970 |
| L1 down c13 | MLP | a%100 in {0..4, 7, 50, 60, 80, 88, 90..92, 94..98} (+) | 1.00 | 1.000 |
| L1 down c150 | MLP | a%100 in {1..4} (-) | 1.00 | 0.080 |
| L1 down c93 | MLP | a%100 in {1..7} (+) | 1.00 | 0.110 |
| L1 down c87 | MLP | a%100 in {10, 14..30, 35} (+) | 1.00 | 0.270 |
| L1 down c258 | MLP | a%100 in {1} (-) | 0.99 | 0.116 |
| L1 down c548 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L1 down c181 | MLP | a%100 in {9, 89..99} (+) | 1.00 | 0.241 |
| L1 down c247 | MLP | a%100 in {99} (-) | 0.93 | 0.009 |

</details>

</details>

<details><summary><b>2a-155</b> `a%100` @ `op` (add) — block code, copy from `a`; 12 comps, L1; tells apart 31/100 classes (best member 7); on sub: 2s-204 (member overlap 0.75)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.59 |
| classes told apart: joint / best code / best member | 31 /  / 7 (of 100) |
| members whose removal merges classes | 0.75 |
| support overlap (1 = tiling) / random sets / p | 1.34 / 1.69 / 0.14 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.99 |
| decoding acc. joint / best code / best member (chance) | 0.76 /  / 0.21 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 5 / 4 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.30 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.72 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.19 2:0.09 40:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.28 2:0.20 3:0.14) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.33 / 4.32 |
| best source position (CKA) | `a` (0.58) |

<details><summary>codes and components</summary>

**code 2a-155.0 (L1): 12 comps, tells apart 31/100, coverage 0.59, overlap 1.34 (random 1.64, p 0.12)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c398 | H26 | a%100 in {0, 98..99} (-) | 0.99 | 0.225 |
| L1 o c188 | H26 | a%100 in {1..6} (+) | 1.00 | 0.080 |
| L1 o c320 | H26 | a%100 in {10..12} (-) | 1.00 | 0.030 |
| L1 o c403 | H26 | a%100 in {10..22} (-) | 1.00 | 0.190 |
| L1 o c244 | H26 | a%100 in {11} (-) | 1.00 | 0.010 |
| L1 o c108 | H26 | a%100 in {2, 12, 18, 20..24, 26..34, 42} (-) | 0.99 | 0.193 |
| L1 o c469 | H26 | a%100 in {20, 30, 40, 50, 60, 70, 80, 90} (-) | 0.99 | 0.082 |
| L1 o c106 | H26 | a%100 in {38..48} (+) | 0.99 | 0.154 |
| L1 o c66 | H26 | a%100 in {50..57} (-) | 0.96 | 0.089 |
| L1 o c201 | H26 | a%100 in {6..11} (+) | 1.00 | 0.192 |
| L1 o c243 | H26 | a%100 in {90} (+) | 1.00 | 0.010 |
| L1 o c251 | H26 | a%100 in {90} (+) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2a-158</b> `a%100` @ `op` (add) — block code; 9 comps, L2; tells apart 13/100 classes (best member 6); on sub: 2s-206 (member overlap 0.06)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.53 |
| classes told apart: joint / best code / best member | 13 /  / 6 (of 100) |
| members whose removal merges classes | 0.33 |
| support overlap (1 = tiling) / random sets / p | 1.19 / 1.45 / 0.12 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.94 /  / 0.60 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.98 |
| consumers / read jointly | 13 / 1 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 attn |
| CKA(arrangement before, joint write) | 0.47 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.14 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.29 2:0.12 20:0.08) |
| joint write: shape (spectrum k:share) | irregular (1:0.41 2:0.14 20:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.30 / 2.94 |

<details><summary>codes and components</summary>

**code 2a-158.0 (L2): 9 comps, tells apart 13/100, coverage 0.53, overlap 1.19 (random 1.43, p 0.12)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 down c214 | MLP | a%100 in {0, 90, 95, 98..99} (-) | 1.00 | 0.070 |
| L2 down c0 | MLP | a%100 in {0..2, 60, 70, 75, 80, 90, 95, 99} (-) | 1.00 | 1.000 |
| L2 down c42 | MLP | a%100 in {18} (+) | 1.00 | 0.010 |
| L2 down c114 | MLP | a%100 in {18} (+) | 1.00 | 0.070 |
| L2 down c11 | MLP | a%100 in {25, 28..45} (-) | 1.00 | 0.330 |
| L2 down c39 | MLP | a%100 in {50, 55, 65} (+) | 1.00 | 0.030 |
| L2 down c18 | MLP | a%100 in {76..79, 81..89, 91..99} (+) | 1.00 | 0.220 |
| L2 down c448 | MLP | a%100 in {90} (-) | 1.00 | 0.010 |
| L2 down c573 | MLP | a%100 in {90} (-) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2a-157</b> `a%100` @ `op` (add) — block code; 22 comps, L2 L3; tells apart 44/100 classes (best member 8); on sub: 2s-207 (member overlap 0.17)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.53 |
| classes told apart: joint / best code / best member | 44 /  / 8 (of 100) |
| members whose removal merges classes | 0.41 |
| support overlap (1 = tiling) / random sets / p | 1.75 / 2.28 / 0.05 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.95 /  / 0.50 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.05 |
| consumers / read jointly | 3 / 2 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L1 MLP |
| CKA(arrangement before, joint write) | 0.45 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.07 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.07 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.26 2:0.12 20:0.09) |
| joint write: shape (spectrum k:share) | irregular (1:0.26 2:0.10 3:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.27 / 5.78 |
| best source position (CKA) | `a` (0.50) |

<details><summary>codes and components</summary>

**code 2a-157.0 (L2 L3): 22 comps, tells apart 44/100, coverage 0.53, overlap 1.75 (random 2.18, p 0.07)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 o c89 | H23 | a%100 in {1..3, 15, 20, 25, 30, 49..50, 75, 94} (+) | 1.00 | 1.000 |
| L3 o c101 | H15 | a%100 in {0, 60, 70, 75, 80..81, 83..85, 90} (-) | 1.00 | 0.280 |
| L3 o c324 | H15 | a%100 in {11, 51} (-) | 1.00 | 0.020 |
| L3 o c368 | H15 | a%100 in {17..19, 21} (-) | 1.00 | 0.060 |
| L3 o c393 | H15 | a%100 in {18..19, 21} (+) | 1.00 | 0.040 |
| L3 o c246 | H15 | a%100 in {18..19, 21} (-) | 1.00 | 0.030 |
| L3 o c183 | H15 | a%100 in {21, 91} (-) | 1.00 | 0.020 |
| L3 o c453 | H15 | a%100 in {21..22} (+) | 1.00 | 0.080 |
| L3 o c350 | H15 | a%100 in {21} (-) | 1.00 | 0.010 |
| L3 o c109 | H15 | a%100 in {51..52, 55, 57} (-) | 0.99 | 0.122 |
| L3 o c464 | H15 | a%100 in {55, 57, 61, 63, 65, 67, 69, 71, 73..74, 76..79, 81..88} (+) | 1.00 | 0.370 |
| L3 o c427 | H15 | a%100 in {55, 65} (+) | 1.00 | 0.280 |
| L3 o c327 | H15 | a%100 in {55} (+) | 1.00 | 0.040 |
| L3 o c399 | H15 | a%100 in {55} (+) | 1.00 | 0.030 |
| L3 o c119 | H15 | a%100 in {55} (-) | 1.00 | 0.010 |
| L3 o c74 | H15 | a%100 in {68..79} (-) | 1.00 | 0.120 |
| L3 o c307 | H15 | a%100 in {86} (+) | 1.00 | 0.010 |
| L3 o c153 | H15 | a%100 in {90} (-) | 1.00 | 0.010 |
| L3 o c432 | H15 | a%100 in {96..99} (+) | 1.00 | 0.069 |
| L3 o c194 | H15 | a%100 in {98..99} (+) | 1.00 | 0.040 |
| L3 o c235 | H15 | a%100 in {98..99} (-) | 1.00 | 0.020 |
| L3 o c489 | H15 | a%100 in {98..99} (-) | 1.00 | 0.020 |

</details>

</details>

<details><summary><b>2a-159</b> `a%100` @ `op` (add) — block code, 2 codes of the same shape; 5 comps, L3 L4 L31; tells apart 3/100 classes (best member 3); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.06 |
| classes told apart: joint / best code / best member | 3 / 6 / 3 (of 100) |
| members whose removal merges classes | 0.20 |
| support overlap (1 = tiling) / random sets / p | 1.50 / 1.17 / 0.88 |
| mean CKA between its codes | 0.83 |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.59 / 0.59 / 0.51 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 5 / 2 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 MLP |
| CKA(arrangement before, joint write) | 0.06 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 1.78 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.32 2:0.11 20:0.07) |
| joint write: shape (spectrum k:share) | line () |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.01 / 1.22 |
| best source position (CKA) | `a` (0.05) |

<details><summary>codes and components</summary>

**code 2a-159.0 (L3 L4): 3 comps, tells apart 6/100, coverage 0.06, overlap 1.17 (random 1.05, p 0.71)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 o c380 | H4 | a%100 in {55} (+) | 1.00 | 0.122 |
| L3 o c0 | H4 | a%100 in {55} (-) | 1.00 | 0.010 |
| L4 o c113 | H24 | a%100 in {1..2, 11, 16, 18} (-) | 1.00 | 1.000 |

**code 2a-159.1 (L31): 2 comps, tells apart 3/100, coverage 0.02, overlap 1.00 (random 1.00, p 0.70)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L31 down c61 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L31 down c180 | MLP | a%100 in {55} (+) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2a-160</b> `a%100` @ `op` (add) — block code, copy from `a`; 5 comps, L3; tells apart 16/100 classes (best member 7); on sub: 2s-208 (member overlap 0.17)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.39 |
| classes told apart: joint / best code / best member | 16 /  / 7 (of 100) |
| members whose removal merges classes | 0.80 |
| support overlap (1 = tiling) / random sets / p | 1.15 / 1.19 / 0.45 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.98 /  / 0.70 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 MLP |
| CKA(arrangement before, joint write) | 0.62 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.09 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.32 2:0.11 20:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.68) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.51 / 2.17 |
| best source position (CKA) | `a` (0.66) |

<details><summary>codes and components</summary>

**code 2a-160.0 (L3): 5 comps, tells apart 16/100, coverage 0.39, overlap 1.15 (random 1.18, p 0.42)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 o c104 | H7 | a%100 in {0, 99} (-) | 1.00 | 0.040 |
| L3 o c128 | H5 | a%100 in {1..12, 14, 16, 24, 55, 57, 60, 65, 71, 73, 75, 77, 85} (+) | 1.00 | 1.000 |
| L3 o c221 | H5 | a%100 in {1..4} (-) | 1.00 | 0.080 |
| L3 o c27 | H7 | a%100 in {1} (+) | 1.00 | 0.010 |
| L3 o c49 | H7 | a%100 in {29, 31, 37, 41..43, 45..47, 51..55} (-) | 1.00 | 0.650 |

</details>

</details>

<details><summary><b>2a-161</b> `a%100` @ `op` (add) — tiling; 17 comps, L3; tells apart 24/100 classes (best member 8); on sub: 2s-209 (member overlap 0.06)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.47 |
| classes told apart: joint / best code / best member | 24 /  / 8 (of 100) |
| members whose removal merges classes | 0.24 |
| support overlap (1 = tiling) / random sets / p | 1.38 / 1.93 / 0.02 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.92 /  / 0.68 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.04 |
| consumers / read jointly | 30 / 15 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L3 attn |
| CKA(arrangement before, joint write) | 0.63 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.16 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.34 2:0.11 20:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.38 2:0.14 3:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.34 / 3.80 |

<details><summary>codes and components</summary>

**code 2a-161.0 (L3): 17 comps, tells apart 24/100, coverage 0.47, overlap 1.38 (random 1.97, p 0.03)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 down c18 | MLP | a%100 in {0..2, 55, 60, 70, 80, 90, 95} (+) | 1.00 | 1.000 |
| L3 down c259 | MLP | a%100 in {1..5} (-) | 1.00 | 0.080 |
| L3 down c88 | MLP | a%100 in {11, 13..14, 16, 18} (-) | 1.00 | 0.160 |
| L3 down c236 | MLP | a%100 in {14..29} (-) | 1.00 | 0.190 |
| L3 down c184 | MLP | a%100 in {18, 21} (-) | 1.00 | 0.020 |
| L3 down c322 | MLP | a%100 in {18} (+) | 1.00 | 0.010 |
| L3 down c593 | MLP | a%100 in {18} (+) | 1.00 | 0.050 |
| L3 down c632 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L3 down c683 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L3 down c431 | MLP | a%100 in {50, 55} (+) | 1.00 | 0.170 |
| L3 down c41 | MLP | a%100 in {50} (+) | 1.00 | 0.010 |
| L3 down c635 | MLP | a%100 in {55} (+) | 1.00 | 0.010 |
| L3 down c603 | MLP | a%100 in {55} (-) | 1.00 | 0.010 |
| L3 down c125 | MLP | a%100 in {68..80, 82, 85} (-) | 1.00 | 0.190 |
| L3 down c488 | MLP | a%100 in {90} (+) | 1.00 | 0.010 |
| L3 down c6 | MLP | a%100 in {98..99} (-) | 1.00 | 0.020 |
| L3 down c225 | MLP | a%100 in {9} (-) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2a-162</b> `a%100` @ `op` (add) — single component; 1 comps, L4; tells apart 4/100 classes (best member 4); on sub: 2s-210 (member overlap 0.50)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.11 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.38 /  / 0.38 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L3 MLP |
| CKA(arrangement before, joint write) | 0.36 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.34 2:0.12 20:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.53) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.15 / 1.00 |
| best source position (CKA) | `a` (0.35) |

<details><summary>codes and components</summary>

**code 2a-162.0 (L4): 1 comps, tells apart 4/100, coverage 0.11, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 o c7 | H18 | a%100 in {2, 9..11, 13..14, 19, 55, 60, 70, 80} (+) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2a-163</b> `a%100` @ `op` (add) — block code; 17 comps, L4; tells apart 29/100 classes (best member 8); on sub: 2s-211 (member overlap 0.09)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.40 |
| classes told apart: joint / best code / best member | 29 /  / 8 (of 100) |
| members whose removal merges classes | 0.47 |
| support overlap (1 = tiling) / random sets / p | 2.17 / 1.95 / 0.72 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.94 /  / 0.72 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.02 |
| consumers / read jointly | 33 / 26 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L4 attn |
| CKA(arrangement before, joint write) | 0.75 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.02 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.22 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.33 2:0.12 20:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.34 2:0.16 40:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.31 / 3.75 |

<details><summary>codes and components</summary>

**code 2a-163.0 (L4): 17 comps, tells apart 29/100, coverage 0.40, overlap 2.17 (random 1.94, p 0.73)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 down c129 | MLP | a%100 in {0, 50, 60, 80} (-) | 1.00 | 0.040 |
| L4 down c204 | MLP | a%100 in {0, 99} (+) | 1.00 | 0.030 |
| L4 down c30 | MLP | a%100 in {0..5, 10, 15, 18, 20..21, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 79..80, 89..91} (-) | 1.00 | 0.870 |
| L4 down c33 | MLP | a%100 in {1..10, 12} (-) | 1.00 | 0.440 |
| L4 down c106 | MLP | a%100 in {1..2} (-) | 1.00 | 0.030 |
| L4 down c842 | MLP | a%100 in {1..4} (-) | 1.00 | 0.070 |
| L4 down c89 | MLP | a%100 in {11..19, 21..22} (-) | 1.00 | 0.290 |
| L4 down c52 | MLP | a%100 in {11} (+) | 1.00 | 0.010 |
| L4 down c159 | MLP | a%100 in {11} (-) | 1.00 | 0.010 |
| L4 down c103 | MLP | a%100 in {13..14, 16} (-) | 1.00 | 0.040 |
| L4 down c237 | MLP | a%100 in {18} (-) | 1.00 | 0.010 |
| L4 down c487 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L4 down c303 | MLP | a%100 in {2..13} (+) | 1.00 | 0.210 |
| L4 down c61 | MLP | a%100 in {50, 55, 60, 65} (-) | 1.00 | 0.210 |
| L4 down c442 | MLP | a%100 in {55} (+) | 1.00 | 0.010 |
| L4 down c726 | MLP | a%100 in {90} (-) | 1.00 | 0.010 |
| L4 down c145 | MLP | a%100 in {99} (+) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2a-164</b> `a%100` @ `op` (add) — block code; 3 comps, L5 L6; tells apart 17/100 classes (best member 5); on sub: 2s-212 (member overlap 0.67)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.20 |
| classes told apart: joint / best code / best member | 17 /  / 5 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.15 / 1.06 / 0.67 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.99 /  / 0.48 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.15 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L4 MLP |
| CKA(arrangement before, joint write) | 0.39 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.34 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.40 2:0.10 40:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.16 / 1.78 |
| best source position (CKA) | `a` (0.36) |

<details><summary>codes and components</summary>

**code 2a-164.0 (L5 L6): 3 comps, tells apart 17/100, coverage 0.20, overlap 1.15 (random 1.04, p 0.76)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 o c2 | H26 | a%100 in {1, 3..6, 9..10, 13..15, 65, 70, 72, 80, 92..93} (+) | 1.00 | 1.000 |
| L5 o c6 | H26 | a%100 in {2, 55, 65, 70, 72, 89} (-) | 1.00 | 1.000 |
| L6 o c4 | H22 | a%100 in {11} (+) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2a-165</b> `a%100` @ `op` (add) — tiling, 26 codes of the same shape; 505 comps, L5 L6 L7 L8 L9 L10 L11 L12 L13 L14 L15 L16 L17 L18 L19 L20 L21 L22 L23 L24 L25 L26 L27 L28 L29 L30; tells apart 31/100 classes (best member 10); on sub: 2s-227 (member overlap 0.02)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 31 / 39 / 10 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p | 37.45 / 36.16 / 0.90 |
| mean CKA between its codes | 0.88 |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 1.00 / 1.00 / 0.77 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.30 |
| consumers / read jointly | 805 / 334 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L5 attn |
| CKA(arrangement before, joint write) | 0.71 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.02 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 90.63 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.34 2:0.12 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.32 2:0.14 3:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.33 / 4.74 |

<details><summary>codes and components</summary>

**code 2a-165.0 (L5): 18 comps, tells apart 24/100, coverage 0.70, overlap 2.03 (random 2.00, p 0.53)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 down c15 | MLP | a%100 in {0..2, 11, 22..24, 26..29, 31..34, 36..39, 41..44, 46..50, 55, 60, 65, 70, 75, 80, 82, 84..99} (-) | 1.00 | 0.760 |
| L5 down c395 | MLP | a%100 in {0} (-) | 1.00 | 0.020 |
| L5 down c43 | MLP | a%100 in {1..2} (-) | 1.00 | 0.020 |
| L5 down c13 | MLP | a%100 in {1..3, 11, 18, 55, 76, 85, 90} (-) | 1.00 | 1.000 |
| L5 down c227 | MLP | a%100 in {1..6} (-) | 1.00 | 0.110 |
| L5 down c251 | MLP | a%100 in {11} (+) | 1.00 | 0.061 |
| L5 down c134 | MLP | a%100 in {18, 21} (-) | 1.00 | 0.100 |
| L5 down c203 | MLP | a%100 in {1} (+) | 1.00 | 0.020 |
| L5 down c1 | MLP | a%100 in {2..3, 8..11, 13, 21, 39, 60, 62, 65..66, 68, 70, 75..76, 80, 85, 90} (-) | 1.00 | 1.000 |
| L5 down c123 | MLP | a%100 in {2..3} (-) | 1.00 | 0.020 |
| L5 down c20 | MLP | a%100 in {34, 37..39, 41..44, 46..49, 51, 53, 59} (+) | 0.98 | 0.171 |
| L5 down c304 | MLP | a%100 in {40, 45, 50, 55, 60, 65} (-) | 1.00 | 0.100 |
| L5 down c191 | MLP | a%100 in {55} (+) | 1.00 | 0.010 |
| L5 down c321 | MLP | a%100 in {55} (-) | 1.00 | 0.155 |
| L5 down c669 | MLP | a%100 in {86, 88..98} (+) | 1.00 | 0.140 |
| L5 down c605 | MLP | a%100 in {89, 91..99} (+) | 1.00 | 0.116 |
| L5 down c77 | MLP | a%100 in {90} (-) | 1.00 | 0.010 |
| L5 down c941 | MLP | a%100 in {90} (-) | 1.00 | 0.010 |

**code 2a-165.1 (L6): 13 comps, tells apart 33/100, coverage 0.46, overlap 2.13 (random 1.71, p 0.86)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L6 down c35 | MLP | a%100 in {0, 10, 15, 20, 25, 30, 40, 60, 70, 75, 80} (+) | 1.00 | 0.141 |
| L6 down c3 | MLP | a%100 in {1..2, 55, 90..92, 96} (+) | 1.00 | 0.990 |
| L6 down c189 | MLP | a%100 in {1..3} (+) | 1.00 | 0.040 |
| L6 down c163 | MLP | a%100 in {1..6} (+) | 1.00 | 0.110 |
| L6 down c90 | MLP | a%100 in {11, 14, 16..19, 21, 55} (-) | 1.00 | 0.220 |
| L6 down c706 | MLP | a%100 in {11} (+) | 1.00 | 0.040 |
| L6 down c83 | MLP | a%100 in {18} (-) | 1.00 | 0.010 |
| L6 down c0 | MLP | a%100 in {2..8, 10, 12..17, 59, 79, 89..94, 97, 99} (+) | 1.00 | 1.000 |
| L6 down c27 | MLP | a%100 in {3..17, 19} (-) | 1.00 | 0.290 |
| L6 down c279 | MLP | a%100 in {50, 55, 60, 65, 70, 75, 80, 85} (+) | 1.00 | 0.470 |
| L6 down c240 | MLP | a%100 in {55} (+) | 1.00 | 0.050 |
| L6 down c424 | MLP | a%100 in {89, 91..97} (+) | 1.00 | 0.090 |
| L6 down c449 | MLP | a%100 in {95, 97..99} (+) | 1.00 | 0.090 |

**code 2a-165.2 (L7): 14 comps, tells apart 22/100, coverage 0.71, overlap 1.82 (random 1.80, p 0.53)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L7 down c10 | MLP | a%100 in {0..5, 11, 16, 18, 22..24, 26..29, 31..39, 41..50, 54..55, 60, 65, 70, 75, 80..83, 85..98} (+) | 1.00 | 0.670 |
| L7 down c12 | MLP | a%100 in {1..2, 11, 18, 55, 90..94, 96} (-) | 1.00 | 1.000 |
| L7 down c316 | MLP | a%100 in {1..2} (+) | 1.00 | 0.040 |
| L7 down c2 | MLP | a%100 in {1..3, 9, 18..19, 60, 65, 70, 80, 85, 90} (-) | 1.00 | 1.000 |
| L7 down c11 | MLP | a%100 in {1..6} (+) | 1.00 | 0.100 |
| L7 down c575 | MLP | a%100 in {11, 55} (+) | 1.00 | 0.110 |
| L7 down c27 | MLP | a%100 in {11} (+) | 1.00 | 0.010 |
| L7 down c537 | MLP | a%100 in {18, 21} (-) | 1.00 | 0.020 |
| L7 down c112 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L7 down c411 | MLP | a%100 in {4..17, 19} (-) | 1.00 | 0.210 |
| L7 down c1016 | MLP | a%100 in {55, 65} (-) | 1.00 | 0.020 |
| L7 down c457 | MLP | a%100 in {89, 91..98} (+) | 1.00 | 0.130 |
| L7 down c552 | MLP | a%100 in {90} (+) | 1.00 | 0.010 |
| L7 down c191 | MLP | a%100 in {91..94, 96..97} (-) | 1.00 | 0.080 |

**code 2a-165.3 (L8): 12 comps, tells apart 27/100, coverage 0.67, overlap 1.63 (random 1.66, p 0.45)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L8 down c99 | MLP | a%100 in {0, 20, 22..31, 33..49, 90} (+) | 1.00 | 0.300 |
| L8 down c58 | MLP | a%100 in {0, 40, 50, 60, 65, 70, 75..76, 80, 82..85} (+) | 1.00 | 0.490 |
| L8 down c4 | MLP | a%100 in {0..3, 11, 18, 50, 55, 80, 90..94, 96..97} (-) | 1.00 | 1.000 |
| L8 down c648 | MLP | a%100 in {1..2} (+) | 1.00 | 0.030 |
| L8 down c741 | MLP | a%100 in {1..6} (-) | 1.00 | 0.100 |
| L8 down c1019 | MLP | a%100 in {11, 14, 16..18, 21} (-) | 1.00 | 0.080 |
| L8 down c134 | MLP | a%100 in {11} (+) | 1.00 | 0.010 |
| L8 down c388 | MLP | a%100 in {15, 18..21} (+) | 1.00 | 0.070 |
| L8 down c187 | MLP | a%100 in {40, 50, 55, 60, 65, 70, 75} (-) | 1.00 | 0.210 |
| L8 down c14 | MLP | a%100 in {55, 86, 89..98} (+) | 1.00 | 0.220 |
| L8 down c261 | MLP | a%100 in {55} (-) | 1.00 | 0.010 |
| L8 down c212 | MLP | a%100 in {89, 91..99} (-) | 1.00 | 0.130 |

**code 2a-165.4 (L9): 21 comps, tells apart 28/100, coverage 0.81, overlap 1.80 (random 2.18, p 0.16)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L9 down c67 | MLP | a%100 in {0, 60, 70, 75, 80, 85} (+) | 1.00 | 0.060 |
| L9 down c184 | MLP | a%100 in {1..2} (+) | 1.00 | 0.040 |
| L9 down c36 | MLP | a%100 in {1..3, 11, 18, 55, 89..98} (-) | 1.00 | 1.000 |
| L9 down c68 | MLP | a%100 in {1..5} (-) | 1.00 | 0.060 |
| L9 down c134 | MLP | a%100 in {10} (+) | 1.00 | 0.010 |
| L9 down c930 | MLP | a%100 in {11, 13} (+) | 1.00 | 0.060 |
| L9 down c554 | MLP | a%100 in {11, 18, 55} (+) | 1.00 | 0.040 |
| L9 down c14 | MLP | a%100 in {11, 55, 90} (-) | 1.00 | 0.030 |
| L9 down c317 | MLP | a%100 in {11} (+) | 1.00 | 0.010 |
| L9 down c47 | MLP | a%100 in {15, 20, 25, 30, 35} (-) | 1.00 | 0.050 |
| L9 down c140 | MLP | a%100 in {18} (+) | 1.00 | 0.020 |
| L9 down c97 | MLP | a%100 in {22..24, 26..29, 31..39, 41..49} (-) | 1.00 | 0.383 |
| L9 down c768 | MLP | a%100 in {3..10, 12..23} (-) | 1.00 | 0.250 |
| L9 down c481 | MLP | a%100 in {3..10} (-) | 1.00 | 0.100 |
| L9 down c294 | MLP | a%100 in {50, 55, 60, 65} (+) | 1.00 | 0.040 |
| L9 down c198 | MLP | a%100 in {55} (+) | 1.00 | 0.010 |
| L9 down c122 | MLP | a%100 in {70, 72..85} (-) | 0.99 | 0.153 |
| L9 down c837 | MLP | a%100 in {8..9} (-) | 0.98 | 0.021 |
| L9 down c107 | MLP | a%100 in {86, 88..99} (-) | 1.00 | 0.180 |
| L9 down c664 | MLP | a%100 in {91..94, 96..97} (+) | 1.00 | 0.080 |
| L9 down c131 | MLP | a%100 in {91..97} (-) | 1.00 | 0.090 |

**code 2a-165.5 (L10): 25 comps, tells apart 39/100, coverage 0.95, overlap 2.78 (random 2.42, p 0.76)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L10 down c319 | MLP | a%100 in {0, 30, 40, 50, 60, 65, 70, 75, 80, 85} (-) | 1.00 | 0.230 |
| L10 down c38 | MLP | a%100 in {0, 60, 70, 75, 80, 85, 90} (-) | 1.00 | 0.090 |
| L10 down c18 | MLP | a%100 in {0..16, 18, 20, 24, 26..34, 36..44, 46..56, 58..60, 63, 65, 67..70, 73, 75, 80, 85..86, 89..99} (-) | 1.00 | 0.590 |
| L10 down c2 | MLP | a%100 in {1, 9, 11, 38..39, 41, 43, 55, 66, 74..76, 82..85, 87..88} (+) | 1.00 | 1.000 |
| L10 down c3 | MLP | a%100 in {1..2, 11, 55, 90, 92} (-) | 1.00 | 1.000 |
| L10 down c699 | MLP | a%100 in {1..2} (-) | 1.00 | 0.020 |
| L10 down c238 | MLP | a%100 in {1..4} (+) | 1.00 | 0.060 |
| L10 down c155 | MLP | a%100 in {10, 15, 20, 25, 30, 35, 40, 50} (-) | 1.00 | 0.080 |
| L10 down c277 | MLP | a%100 in {11} (+) | 1.00 | 0.010 |
| L10 down c74 | MLP | a%100 in {11} (-) | 1.00 | 0.020 |
| L10 down c107 | MLP | a%100 in {11} (-) | 1.00 | 0.010 |
| L10 down c144 | MLP | a%100 in {12..14, 16} (-) | 1.00 | 0.040 |
| L10 down c628 | MLP | a%100 in {18, 55, 65} (-) | 1.00 | 0.100 |
| L10 down c459 | MLP | a%100 in {18} (-) | 1.00 | 0.010 |
| L10 down c4 | MLP | a%100 in {19, 22..24, 26..29, 31..34, 36..39, 41..44, 46..49, 53} (+) | 1.00 | 0.400 |
| L10 down c15 | MLP | a%100 in {22..29, 31..39, 41..49} (+) | 1.00 | 0.331 |
| L10 down c309 | MLP | a%100 in {3..10, 12} (+) | 1.00 | 0.160 |
| L10 down c601 | MLP | a%100 in {55} (-) | 1.00 | 0.010 |
| L10 down c67 | MLP | a%100 in {6..7} (-) | 1.00 | 0.020 |
| L10 down c104 | MLP | a%100 in {7..23} (-) | 1.00 | 0.180 |
| L10 down c130 | MLP | a%100 in {72..74, 76..79, 81..89, 98..99} (-) | 1.00 | 0.221 |
| L10 down c115 | MLP | a%100 in {86, 88..98} (-) | 1.00 | 0.130 |
| L10 down c472 | MLP | a%100 in {91..94, 97} (-) | 0.99 | 0.065 |
| L10 down c101 | MLP | a%100 in {91..98} (-) | 1.00 | 0.100 |
| L10 down c182 | MLP | a%100 in {99} (-) | 1.00 | 0.010 |

**code 2a-165.6 (L11): 17 comps, tells apart 26/100, coverage 0.66, overlap 1.71 (random 1.93, p 0.27)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L11 down c173 | MLP | a%100 in {0, 30, 40, 50, 55, 60, 65, 70, 75, 80} (-) | 1.00 | 0.131 |
| L11 down c104 | MLP | a%100 in {0, 70, 75, 80} (+) | 0.99 | 0.042 |
| L11 down c9 | MLP | a%100 in {0..2, 11, 18, 55, 89..98} (-) | 1.00 | 1.000 |
| L11 down c1011 | MLP | a%100 in {1..3} (-) | 1.00 | 0.060 |
| L11 down c116 | MLP | a%100 in {11, 13..14, 16, 18} (-) | 1.00 | 0.129 |
| L11 down c0 | MLP | a%100 in {11, 16..18, 21} (-) | 1.00 | 1.000 |
| L11 down c985 | MLP | a%100 in {11} (+) | 1.00 | 0.010 |
| L11 down c402 | MLP | a%100 in {18, 21} (+) | 1.00 | 0.020 |
| L11 down c604 | MLP | a%100 in {22..24, 26..29, 31..34, 36..39, 41..44, 46..49, 53} (-) | 1.00 | 0.410 |
| L11 down c202 | MLP | a%100 in {3..10, 12..17, 19..20} (+) | 1.00 | 0.230 |
| L11 down c88 | MLP | a%100 in {3..9} (+) | 1.00 | 0.090 |
| L11 down c41 | MLP | a%100 in {5..10} (-) | 1.00 | 0.060 |
| L11 down c509 | MLP | a%100 in {55} (-) | 1.00 | 0.010 |
| L11 down c65 | MLP | a%100 in {86, 89} (+) | 1.00 | 0.020 |
| L11 down c24 | MLP | a%100 in {89, 91..98} (+) | 1.00 | 0.131 |
| L11 down c42 | MLP | a%100 in {90} (+) | 1.00 | 0.010 |
| L11 down c231 | MLP | a%100 in {90} (-) | 1.00 | 0.020 |

**code 2a-165.7 (L12): 18 comps, tells apart 34/100, coverage 0.88, overlap 1.99 (random 2.04, p 0.46)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L12 down c1000 | MLP | a%100 in {0, 60, 65, 70, 75, 80, 85} (+) | 1.00 | 0.070 |
| L12 down c5 | MLP | a%100 in {1..2, 11, 18, 55, 89..98} (+) | 1.00 | 1.000 |
| L12 down c104 | MLP | a%100 in {1..2} (-) | 1.00 | 0.020 |
| L12 down c39 | MLP | a%100 in {1..6} (-) | 1.00 | 0.110 |
| L12 down c60 | MLP | a%100 in {11, 18, 55} (+) | 1.00 | 0.090 |
| L12 down c622 | MLP | a%100 in {11, 55} (+) | 1.00 | 0.020 |
| L12 down c38 | MLP | a%100 in {14..17, 19..39, 41..45} (+) | 1.00 | 0.320 |
| L12 down c0 | MLP | a%100 in {2..3, 11, 18, 39, 55, 89} (+) | 1.00 | 1.000 |
| L12 down c598 | MLP | a%100 in {22..24, 26..29, 31..34, 36..39, 41..44, 46..49, 53..54, 59} (-) | 1.00 | 0.385 |
| L12 down c113 | MLP | a%100 in {3, 5..10} (+) | 1.00 | 0.070 |
| L12 down c789 | MLP | a%100 in {4..10, 12..17, 19} (-) | 1.00 | 0.202 |
| L12 down c53 | MLP | a%100 in {5, 10, 15, 18, 20, 25, 30, 35, 40} (+) | 1.00 | 0.160 |
| L12 down c172 | MLP | a%100 in {55} (-) | 1.00 | 0.010 |
| L12 down c64 | MLP | a%100 in {63, 68..69, 71..74, 76..79} (-) | 0.99 | 0.117 |
| L12 down c90 | MLP | a%100 in {81..82, 84, 86..99} (+) | 1.00 | 0.230 |
| L12 down c242 | MLP | a%100 in {89..98} (+) | 1.00 | 0.120 |
| L12 down c478 | MLP | a%100 in {91..94, 96..97} (+) | 1.00 | 0.060 |
| L12 down c252 | MLP | a%100 in {98..99} (+) | 1.00 | 0.050 |

**code 2a-165.8 (L13): 17 comps, tells apart 25/100, coverage 0.92, overlap 1.87 (random 1.95, p 0.40)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 down c182 | MLP | a%100 in {0, 20, 25, 30, 40, 50, 60, 65, 70, 75, 80, 85} (-) | 1.00 | 0.150 |
| L13 down c26 | MLP | a%100 in {0..11, 18, 22..24, 26..29, 31..34, 36..39, 41..44, 46..50, 53..55, 59..60, 65, 70, 75, 80, 82, 85..86, 88..99} (-) | 1.00 | 0.650 |
| L13 down c698 | MLP | a%100 in {1..3} (+) | 1.00 | 0.040 |
| L13 down c0 | MLP | a%100 in {1..4, 6, 11, 18, 90..93, 99} (-) | 1.00 | 1.000 |
| L13 down c286 | MLP | a%100 in {1..4} (-) | 1.00 | 0.050 |
| L13 down c3 | MLP | a%100 in {11, 55, 90..93} (+) | 1.00 | 1.000 |
| L13 down c566 | MLP | a%100 in {11, 55} (+) | 1.00 | 0.020 |
| L13 down c906 | MLP | a%100 in {11} (-) | 1.00 | 0.010 |
| L13 down c550 | MLP | a%100 in {18, 21} (+) | 1.00 | 0.020 |
| L13 down c24 | MLP | a%100 in {3..10} (-) | 0.99 | 0.082 |
| L13 down c781 | MLP | a%100 in {3..9} (+) | 1.00 | 0.070 |
| L13 down c497 | MLP | a%100 in {4..10, 12..20} (+) | 1.00 | 0.180 |
| L13 down c25 | MLP | a%100 in {55} (-) | 1.00 | 0.010 |
| L13 down c897 | MLP | a%100 in {55} (-) | 1.00 | 0.010 |
| L13 down c268 | MLP | a%100 in {58, 61, 63, 66..69, 71..79, 81..89, 99} (-) | 1.00 | 0.320 |
| L13 down c491 | MLP | a%100 in {89, 91..98} (+) | 1.00 | 0.120 |
| L13 down c353 | MLP | a%100 in {99} (-) | 1.00 | 0.010 |

**code 2a-165.9 (L14): 23 comps, tells apart 34/100, coverage 0.78, overlap 1.83 (random 2.33, p 0.10)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L14 down c91 | MLP | a%100 in {0, 20, 25, 30, 35, 40, 50, 60, 70, 75, 80, 85} (-) | 1.00 | 0.150 |
| L14 down c18 | MLP | a%100 in {0, 20, 25, 30, 40, 50, 70, 75, 80} (+) | 1.00 | 0.160 |
| L14 down c24 | MLP | a%100 in {1, 11, 55} (-) | 1.00 | 0.990 |
| L14 down c32 | MLP | a%100 in {1..2, 11} (+) | 1.00 | 0.040 |
| L14 down c170 | MLP | a%100 in {1..3} (-) | 1.00 | 0.061 |
| L14 down c106 | MLP | a%100 in {1..5} (-) | 1.00 | 0.130 |
| L14 down c842 | MLP | a%100 in {11, 18} (-) | 1.00 | 0.020 |
| L14 down c919 | MLP | a%100 in {11} (+) | 1.00 | 0.040 |
| L14 down c280 | MLP | a%100 in {11} (-) | 1.00 | 0.010 |
| L14 down c78 | MLP | a%100 in {18, 21, 50, 55, 60, 65} (-) | 1.00 | 0.200 |
| L14 down c229 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L14 down c218 | MLP | a%100 in {22..24, 26..29, 31..34, 36..39, 41..44, 46..49} (+) | 1.00 | 0.310 |
| L14 down c133 | MLP | a%100 in {3, 5} (-) | 1.00 | 0.020 |
| L14 down c359 | MLP | a%100 in {3..10, 12..17, 19..20} (-) | 1.00 | 0.300 |
| L14 down c164 | MLP | a%100 in {37..39, 41..44, 46..49} (+) | 1.00 | 0.110 |
| L14 down c298 | MLP | a%100 in {4..10} (-) | 1.00 | 0.071 |
| L14 down c736 | MLP | a%100 in {40, 60, 70, 75, 80, 85} (-) | 1.00 | 0.060 |
| L14 down c973 | MLP | a%100 in {55} (-) | 1.00 | 0.010 |
| L14 down c274 | MLP | a%100 in {60, 65, 70, 75, 80, 85} (+) | 1.00 | 0.060 |
| L14 down c555 | MLP | a%100 in {74, 76, 78..79, 81..85, 87..89, 99} (+) | 1.00 | 0.230 |
| L14 down c199 | MLP | a%100 in {89..98} (+) | 1.00 | 0.151 |
| L14 down c143 | MLP | a%100 in {90} (+) | 1.00 | 0.010 |
| L14 down c34 | MLP | a%100 in {90} (-) | 1.00 | 0.010 |

**code 2a-165.10 (L15): 18 comps, tells apart 20/100, coverage 0.54, overlap 1.70 (random 2.02, p 0.17)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c60 | MLP | a%100 in {0..3, 11, 18, 55, 90..94, 96..97} (-) | 1.00 | 1.000 |
| L15 down c71 | MLP | a%100 in {1..2} (+) | 1.00 | 0.050 |
| L15 down c489 | MLP | a%100 in {1..2} (+) | 1.00 | 0.020 |
| L15 down c196 | MLP | a%100 in {1..4, 86} (-) | 1.00 | 0.130 |
| L15 down c58 | MLP | a%100 in {11, 13..14, 16..19, 21} (+) | 1.00 | 0.110 |
| L15 down c99 | MLP | a%100 in {11, 18} (+) | 1.00 | 0.020 |
| L15 down c758 | MLP | a%100 in {11} (+) | 1.00 | 0.010 |
| L15 down c243 | MLP | a%100 in {11} (-) | 1.00 | 0.011 |
| L15 down c506 | MLP | a%100 in {18, 55} (-) | 1.00 | 0.030 |
| L15 down c359 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L15 down c229 | MLP | a%100 in {22..24, 26..29, 31..39, 41..49, 53, 59} (+) | 1.00 | 0.491 |
| L15 down c157 | MLP | a%100 in {55, 65} (-) | 1.00 | 0.050 |
| L15 down c73 | MLP | a%100 in {55} (+) | 1.00 | 0.010 |
| L15 down c331 | MLP | a%100 in {89, 98..99} (+) | 1.00 | 0.090 |
| L15 down c31 | MLP | a%100 in {90, 92, 95} (-) | 1.00 | 0.080 |
| L15 down c1 | MLP | a%100 in {90..94, 96..98} (-) | 1.00 | 0.091 |
| L15 down c39 | MLP | a%100 in {90..98} (+) | 1.00 | 0.120 |
| L15 down c104 | MLP | a%100 in {90} (+) | 1.00 | 0.010 |

**code 2a-165.11 (L16): 20 comps, tells apart 29/100, coverage 0.69, overlap 1.81 (random 2.12, p 0.14)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c55 | MLP | a%100 in {0, 20, 25, 30, 35, 40, 50, 55, 60, 65, 70, 75, 80, 85} (+) | 1.00 | 0.170 |
| L16 down c222 | MLP | a%100 in {0, 40, 60, 65, 70, 75, 80, 85} (-) | 1.00 | 0.080 |
| L16 down c4 | MLP | a%100 in {0, 90, 99} (-) | 1.00 | 1.000 |
| L16 down c93 | MLP | a%100 in {1..10, 12..17} (+) | 1.00 | 0.250 |
| L16 down c14 | MLP | a%100 in {1..2, 11, 18, 55, 89..98} (+) | 1.00 | 1.000 |
| L16 down c509 | MLP | a%100 in {1..2} (+) | 1.00 | 0.020 |
| L16 down c113 | MLP | a%100 in {1..4} (-) | 1.00 | 0.060 |
| L16 down c58 | MLP | a%100 in {1..6} (+) | 1.00 | 0.060 |
| L16 down c444 | MLP | a%100 in {11, 13} (+) | 1.00 | 0.070 |
| L16 down c99 | MLP | a%100 in {11, 55, 90} (+) | 1.00 | 0.030 |
| L16 down c643 | MLP | a%100 in {18, 21, 55} (+) | 1.00 | 0.094 |
| L16 down c143 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L16 down c634 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L16 down c282 | MLP | a%100 in {2..3} (-) | 1.00 | 0.020 |
| L16 down c475 | MLP | a%100 in {22..24, 26..29, 31..34, 36..39, 41..44, 46..49} (-) | 1.00 | 0.345 |
| L16 down c336 | MLP | a%100 in {2} (+) | 1.00 | 0.010 |
| L16 down c908 | MLP | a%100 in {55} (-) | 1.00 | 0.010 |
| L16 down c13 | MLP | a%100 in {86, 88..98} (-) | 1.00 | 0.441 |
| L16 down c9 | MLP | a%100 in {90} (-) | 1.00 | 0.011 |
| L16 down c117 | MLP | a%100 in {91..97} (+) | 1.00 | 0.080 |

**code 2a-165.12 (L17): 18 comps, tells apart 29/100, coverage 0.83, overlap 1.59 (random 2.04, p 0.07)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 down c62 | MLP | a%100 in {0, 20, 25, 30, 40, 50, 60, 65, 70, 75, 80, 85} (-) | 1.00 | 0.160 |
| L17 down c46 | MLP | a%100 in {0, 72..85, 88, 99} (+) | 1.00 | 0.314 |
| L17 down c24 | MLP | a%100 in {1..2, 11, 18, 55, 90..97} (-) | 1.00 | 0.183 |
| L17 down c304 | MLP | a%100 in {1..2} (-) | 1.00 | 0.040 |
| L17 down c281 | MLP | a%100 in {1..3} (-) | 1.00 | 0.040 |
| L17 down c218 | MLP | a%100 in {1..6} (+) | 1.00 | 0.080 |
| L17 down c56 | MLP | a%100 in {10..11} (-) | 1.00 | 0.020 |
| L17 down c53 | MLP | a%100 in {11, 18, 21} (-) | 1.00 | 0.080 |
| L17 down c0 | MLP | a%100 in {11, 55} (+) | 1.00 | 1.000 |
| L17 down c60 | MLP | a%100 in {22..24, 26..29, 31..34, 36..39, 41..44, 46..49, 53, 59} (+) | 1.00 | 0.557 |
| L17 down c69 | MLP | a%100 in {3..10, 12..17, 19..20} (-) | 1.00 | 0.296 |
| L17 down c159 | MLP | a%100 in {55} (-) | 1.00 | 0.060 |
| L17 down c161 | MLP | a%100 in {5} (-) | 1.00 | 0.010 |
| L17 down c91 | MLP | a%100 in {84, 86..89, 94..98} (+) | 1.00 | 0.111 |
| L17 down c242 | MLP | a%100 in {90..98} (+) | 1.00 | 0.100 |
| L17 down c8 | MLP | a%100 in {90} (+) | 1.00 | 0.010 |
| L17 down c85 | MLP | a%100 in {90} (-) | 1.00 | 0.010 |
| L17 down c181 | MLP | a%100 in {91..98} (-) | 1.00 | 0.110 |

**code 2a-165.13 (L18): 15 comps, tells apart 28/100, coverage 0.71, overlap 1.68 (random 1.79, p 0.28)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 down c113 | MLP | a%100 in {0, 20, 25, 30, 40, 50, 60, 65, 70, 75, 80, 85} (+) | 1.00 | 0.150 |
| L18 down c494 | MLP | a%100 in {0, 79, 83..84, 88, 99} (-) | 0.99 | 0.196 |
| L18 down c9 | MLP | a%100 in {0..3, 11, 18, 55, 90..97} (-) | 1.00 | 1.000 |
| L18 down c223 | MLP | a%100 in {1..3} (-) | 1.00 | 0.061 |
| L18 down c13 | MLP | a%100 in {10, 15, 20, 25, 30, 40, 70, 75, 80, 85} (-) | 1.00 | 0.150 |
| L18 down c849 | MLP | a%100 in {18, 55} (+) | 0.97 | 0.036 |
| L18 down c191 | MLP | a%100 in {22..24, 26..29, 31..34, 36..39, 41..44, 46..49} (+) | 1.00 | 0.320 |
| L18 down c64 | MLP | a%100 in {3..10, 12..17, 19} (-) | 1.00 | 0.270 |
| L18 down c639 | MLP | a%100 in {3..5} (+) | 0.99 | 0.030 |
| L18 down c283 | MLP | a%100 in {55} (-) | 1.00 | 0.010 |
| L18 down c117 | MLP | a%100 in {65} (-) | 0.99 | 0.010 |
| L18 down c279 | MLP | a%100 in {86, 88..98} (-) | 1.00 | 0.120 |
| L18 down c1 | MLP | a%100 in {90..97} (+) | 1.00 | 1.000 |
| L18 down c93 | MLP | a%100 in {91..97} (+) | 1.00 | 0.090 |
| L18 down c730 | MLP | a%100 in {99} (+) | 1.00 | 0.010 |

**code 2a-165.14 (L19): 21 comps, tells apart 29/100, coverage 0.75, overlap 2.31 (random 2.20, p 0.59)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L19 down c12 | MLP | a%100 in {0, 2..8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 65, 80} (-) | 1.00 | 0.530 |
| L19 down c48 | MLP | a%100 in {0, 20, 25, 30, 40, 50, 60, 65, 70, 75, 80, 85} (-) | 1.00 | 0.140 |
| L19 down c84 | MLP | a%100 in {0, 40, 51, 56..58, 60..88, 99} (-) | 1.00 | 0.490 |
| L19 down c836 | MLP | a%100 in {0, 99} (+) | 1.00 | 0.070 |
| L19 down c42 | MLP | a%100 in {0..4, 11, 18, 50, 55, 90..97} (+) | 1.00 | 0.890 |
| L19 down c139 | MLP | a%100 in {1..2, 11} (+) | 1.00 | 0.040 |
| L19 down c45 | MLP | a%100 in {1..2, 4, 11, 33..34, 39, 41, 43, 47, 60, 64..65, 86, 93, 95..97} (-) | 1.00 | 1.000 |
| L19 down c30 | MLP | a%100 in {1..5} (+) | 1.00 | 0.090 |
| L19 down c536 | MLP | a%100 in {18, 21} (+) | 1.00 | 0.090 |
| L19 down c132 | MLP | a%100 in {18, 21} (-) | 1.00 | 0.020 |
| L19 down c326 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L19 down c52 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L19 down c25 | MLP | a%100 in {2..3} (+) | 1.00 | 0.030 |
| L19 down c308 | MLP | a%100 in {3..17, 19} (-) | 1.00 | 0.160 |
| L19 down c180 | MLP | a%100 in {5..9} (-) | 1.00 | 0.050 |
| L19 down c194 | MLP | a%100 in {55} (+) | 1.00 | 0.070 |
| L19 down c431 | MLP | a%100 in {6..10, 12..17, 19..20} (-) | 1.00 | 0.180 |
| L19 down c10 | MLP | a%100 in {90..97} (+) | 1.00 | 0.100 |
| L19 down c5 | MLP | a%100 in {91..97} (-) | 1.00 | 0.070 |
| L19 down c80 | MLP | a%100 in {92..93} (+) | 1.00 | 0.030 |
| L19 down c254 | MLP | a%100 in {99} (+) | 1.00 | 0.010 |

**code 2a-165.15 (L20): 25 comps, tells apart 30/100, coverage 0.81, overlap 2.01 (random 2.45, p 0.10)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 down c244 | MLP | a%100 in {0, 20, 25, 30, 35, 40, 45, 50, 60, 65, 70, 75, 80, 85} (-) | 1.00 | 0.150 |
| L20 down c24 | MLP | a%100 in {0..3, 11, 18, 50, 55, 60, 80, 90..98} (+) | 1.00 | 0.880 |
| L20 down c5 | MLP | a%100 in {1..2} (+) | 1.00 | 0.021 |
| L20 down c45 | MLP | a%100 in {1..2} (-) | 1.00 | 0.020 |
| L20 down c64 | MLP | a%100 in {1..4} (+) | 1.00 | 0.060 |
| L20 down c50 | MLP | a%100 in {10} (+) | 1.00 | 0.010 |
| L20 down c154 | MLP | a%100 in {11} (-) | 1.00 | 0.010 |
| L20 down c87 | MLP | a%100 in {18} (+) | 1.00 | 0.020 |
| L20 down c193 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L20 down c1 | MLP | a%100 in {2..4, 39, 41, 43, 86, 89, 91, 95..97} (+) | 1.00 | 1.000 |
| L20 down c545 | MLP | a%100 in {22..24, 26..29, 31..34, 36..39, 41..44, 46..49} (-) | 1.00 | 0.271 |
| L20 down c41 | MLP | a%100 in {2} (-) | 1.00 | 0.010 |
| L20 down c37 | MLP | a%100 in {3, 11} (+) | 1.00 | 0.020 |
| L20 down c353 | MLP | a%100 in {3..10, 12..17, 19..20} (-) | 1.00 | 0.300 |
| L20 down c577 | MLP | a%100 in {3..10} (-) | 1.00 | 0.090 |
| L20 down c177 | MLP | a%100 in {5..9} (+) | 1.00 | 0.050 |
| L20 down c0 | MLP | a%100 in {50, 55, 60, 65, 70, 75, 80, 85, 90..97} (+) | 1.00 | 0.580 |
| L20 down c904 | MLP | a%100 in {55} (+) | 1.00 | 0.050 |
| L20 down c34 | MLP | a%100 in {73..74, 76..79, 81..85, 87..88, 98..99} (+) | 0.99 | 0.218 |
| L20 down c25 | MLP | a%100 in {90} (+) | 1.00 | 0.010 |
| L20 down c72 | MLP | a%100 in {90} (+) | 1.00 | 0.010 |
| L20 down c331 | MLP | a%100 in {90} (-) | 1.00 | 0.010 |
| L20 down c525 | MLP | a%100 in {91..97} (+) | 1.00 | 0.090 |
| L20 down c538 | MLP | a%100 in {91..98} (-) | 0.98 | 0.085 |
| L20 down c579 | MLP | a%100 in {99} (-) | 1.00 | 0.010 |

**code 2a-165.16 (L21): 24 comps, tells apart 27/100, coverage 0.80, overlap 1.96 (random 2.39, p 0.15)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c55 | MLP | a%100 in {0, 30, 40, 70, 75, 80} (-) | 1.00 | 0.111 |
| L21 down c83 | MLP | a%100 in {1..2, 11, 62, 66, 74, 76..79, 81..84, 86..99} (-) | 1.00 | 0.490 |
| L21 down c77 | MLP | a%100 in {1..3, 11, 18, 55, 89..98} (-) | 1.00 | 0.884 |
| L21 down c11 | MLP | a%100 in {1..5, 7..8, 55, 60, 80} (+) | 1.00 | 1.000 |
| L21 down c103 | MLP | a%100 in {1..5} (-) | 1.00 | 0.090 |
| L21 down c58 | MLP | a%100 in {1..7, 11} (+) | 1.00 | 0.100 |
| L21 down c307 | MLP | a%100 in {11} (+) | 1.00 | 0.030 |
| L21 down c150 | MLP | a%100 in {11} (-) | 1.00 | 0.010 |
| L21 down c769 | MLP | a%100 in {11} (-) | 1.00 | 0.010 |
| L21 down c122 | MLP | a%100 in {18, 21} (-) | 1.00 | 0.020 |
| L21 down c287 | MLP | a%100 in {18, 21} (-) | 1.00 | 0.030 |
| L21 down c289 | MLP | a%100 in {18} (-) | 1.00 | 0.010 |
| L21 down c313 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L21 down c130 | MLP | a%100 in {22..24, 26..29, 31..34, 36..39, 41..44, 46..49, 53} (-) | 1.00 | 0.340 |
| L21 down c80 | MLP | a%100 in {3..10, 12..17, 19} (-) | 1.00 | 0.250 |
| L21 down c288 | MLP | a%100 in {30, 40, 60, 70, 75, 80, 85} (+) | 1.00 | 0.070 |
| L21 down c216 | MLP | a%100 in {40, 50, 55, 60, 65, 70, 75, 80} (-) | 1.00 | 0.190 |
| L21 down c423 | MLP | a%100 in {55} (+) | 1.00 | 0.010 |
| L21 down c121 | MLP | a%100 in {89..98} (-) | 1.00 | 0.120 |
| L21 down c66 | MLP | a%100 in {90} (+) | 1.00 | 0.010 |
| L21 down c19 | MLP | a%100 in {90} (-) | 1.00 | 0.010 |
| L21 down c141 | MLP | a%100 in {90} (-) | 1.00 | 0.010 |
| L21 down c93 | MLP | a%100 in {91..94, 96..97} (+) | 1.00 | 0.080 |
| L21 down c94 | MLP | a%100 in {99} (-) | 1.00 | 0.010 |

**code 2a-165.17 (L22): 18 comps, tells apart 26/100, coverage 0.77, overlap 1.77 (random 2.04, p 0.17)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 down c96 | MLP | a%100 in {0, 20, 30, 40, 60, 70, 75, 80} (+) | 1.00 | 0.131 |
| L22 down c170 | MLP | a%100 in {0, 40, 50, 55, 60, 65, 70, 75, 80, 85..86} (-) | 1.00 | 0.373 |
| L22 down c1 | MLP | a%100 in {1..2, 11, 55, 90..98} (+) | 1.00 | 0.899 |
| L22 down c192 | MLP | a%100 in {1..2} (-) | 1.00 | 0.020 |
| L22 down c0 | MLP | a%100 in {1..4, 8, 11, 13, 18, 55, 99} (-) | 1.00 | 1.000 |
| L22 down c85 | MLP | a%100 in {1..6} (+) | 1.00 | 0.120 |
| L22 down c326 | MLP | a%100 in {11} (+) | 1.00 | 0.010 |
| L22 down c829 | MLP | a%100 in {18, 55} (+) | 1.00 | 0.050 |
| L22 down c51 | MLP | a%100 in {20, 25, 30, 35, 40, 45, 60, 65, 70, 75, 80, 85} (-) | 1.00 | 0.120 |
| L22 down c318 | MLP | a%100 in {22..24, 26..29, 31..34, 36..39, 41..44, 46..49, 53} (+) | 1.00 | 0.240 |
| L22 down c218 | MLP | a%100 in {4..10, 12..17, 19} (+) | 1.00 | 0.300 |
| L22 down c373 | MLP | a%100 in {5..9} (-) | 1.00 | 0.050 |
| L22 down c402 | MLP | a%100 in {55} (-) | 1.00 | 0.010 |
| L22 down c408 | MLP | a%100 in {55} (-) | 1.00 | 0.020 |
| L22 down c167 | MLP | a%100 in {78..79, 81..84, 88..89, 98..99} (-) | 1.00 | 0.202 |
| L22 down c67 | MLP | a%100 in {90..98} (+) | 1.00 | 0.100 |
| L22 down c26 | MLP | a%100 in {90} (+) | 1.00 | 0.010 |
| L22 down c138 | MLP | a%100 in {91..94, 96..97} (+) | 1.00 | 0.060 |

**code 2a-165.18 (L23): 20 comps, tells apart 24/100, coverage 0.86, overlap 1.79 (random 2.18, p 0.12)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 down c66 | MLP | a%100 in {0, 20, 25, 30, 35, 40, 45, 50, 57, 60, 62, 64..66, 68, 70..88} (+) | 1.00 | 0.461 |
| L23 down c54 | MLP | a%100 in {0..3, 11, 18, 55, 90..98} (+) | 1.00 | 0.870 |
| L23 down c6 | MLP | a%100 in {1, 11, 55, 92, 94} (+) | 1.00 | 1.000 |
| L23 down c22 | MLP | a%100 in {1..3, 55, 90..98} (-) | 1.00 | 0.900 |
| L23 down c247 | MLP | a%100 in {1..4} (-) | 1.00 | 0.060 |
| L23 down c163 | MLP | a%100 in {11} (+) | 1.00 | 0.010 |
| L23 down c295 | MLP | a%100 in {18} (+) | 1.00 | 0.010 |
| L23 down c124 | MLP | a%100 in {18} (-) | 1.00 | 0.020 |
| L23 down c154 | MLP | a%100 in {18} (-) | 0.92 | 0.023 |
| L23 down c116 | MLP | a%100 in {20, 25, 30, 35, 40, 50, 60, 65, 70, 75, 80, 85} (-) | 1.00 | 0.140 |
| L23 down c462 | MLP | a%100 in {2} (+) | 1.00 | 0.010 |
| L23 down c165 | MLP | a%100 in {3..10, 12..17, 19} (+) | 1.00 | 0.310 |
| L23 down c252 | MLP | a%100 in {3..9} (-) | 1.00 | 0.080 |
| L23 down c271 | MLP | a%100 in {37..39, 41..44, 46..49} (-) | 1.00 | 0.110 |
| L23 down c23 | MLP | a%100 in {51..54, 56..59, 61..64, 66..69, 77} (-) | 0.99 | 0.186 |
| L23 down c175 | MLP | a%100 in {55} (+) | 1.00 | 0.020 |
| L23 down c18 | MLP | a%100 in {91..94, 96} (+) | 1.00 | 0.060 |
| L23 down c969 | MLP | a%100 in {91..97} (+) | 1.00 | 0.080 |
| L23 down c139 | MLP | a%100 in {99} (+) | 1.00 | 0.010 |
| L23 down c223 | MLP | a%100 in {99} (-) | 1.00 | 0.010 |

**code 2a-165.19 (L24): 20 comps, tells apart 28/100, coverage 0.72, overlap 1.79 (random 2.16, p 0.14)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 down c480 | MLP | a%100 in {0, 20, 25, 30, 40, 50, 60, 65, 70, 75, 80, 85} (-) | 1.00 | 0.150 |
| L24 down c107 | MLP | a%100 in {0, 50, 90} (+) | 1.00 | 0.030 |
| L24 down c14 | MLP | a%100 in {1..2, 11, 55, 90..97} (-) | 1.00 | 0.930 |
| L24 down c30 | MLP | a%100 in {1..3} (-) | 1.00 | 0.030 |
| L24 down c246 | MLP | a%100 in {1..6} (-) | 1.00 | 0.090 |
| L24 down c0 | MLP | a%100 in {11, 91..98} (+) | 1.00 | 1.000 |
| L24 down c200 | MLP | a%100 in {11} (+) | 1.00 | 0.010 |
| L24 down c161 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L24 down c146 | MLP | a%100 in {22..24, 26..29, 31..34, 36..39, 41..44, 46..49} (+) | 1.00 | 0.380 |
| L24 down c85 | MLP | a%100 in {3..10, 12..17, 19} (-) | 1.00 | 0.290 |
| L24 down c56 | MLP | a%100 in {3..4} (+) | 1.00 | 0.020 |
| L24 down c214 | MLP | a%100 in {4..10} (-) | 1.00 | 0.070 |
| L24 down c386 | MLP | a%100 in {55} (-) | 1.00 | 0.040 |
| L24 down c286 | MLP | a%100 in {7..10, 12..14, 16..19, 21} (+) | 1.00 | 0.140 |
| L24 down c60 | MLP | a%100 in {82, 84, 86..89, 98..99} (+) | 1.00 | 0.121 |
| L24 down c49 | MLP | a%100 in {90} (+) | 1.00 | 0.010 |
| L24 down c325 | MLP | a%100 in {90} (+) | 1.00 | 0.010 |
| L24 down c105 | MLP | a%100 in {91..98} (+) | 1.00 | 0.101 |
| L24 down c418 | MLP | a%100 in {92..94} (-) | 1.00 | 0.030 |
| L24 down c258 | MLP | a%100 in {99} (-) | 1.00 | 0.010 |

**code 2a-165.20 (L25): 20 comps, tells apart 27/100, coverage 0.70, overlap 1.73 (random 2.13, p 0.11)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L25 down c68 | MLP | a%100 in {0, 20, 25, 30, 35, 40, 50, 60, 65, 70, 75, 80, 85} (-) | 1.00 | 0.160 |
| L25 down c1 | MLP | a%100 in {1..2, 11, 55, 90..94, 96..97} (+) | 1.00 | 1.000 |
| L25 down c83 | MLP | a%100 in {1..3} (+) | 1.00 | 0.040 |
| L25 down c304 | MLP | a%100 in {11, 18} (-) | 1.00 | 0.020 |
| L25 down c563 | MLP | a%100 in {11} (+) | 1.00 | 0.010 |
| L25 down c740 | MLP | a%100 in {12..14, 16..17} (+) | 1.00 | 0.050 |
| L25 down c908 | MLP | a%100 in {18, 55} (+) | 1.00 | 0.030 |
| L25 down c193 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L25 down c360 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L25 down c476 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L25 down c80 | MLP | a%100 in {2..10, 12} (+) | 1.00 | 0.180 |
| L25 down c97 | MLP | a%100 in {20, 25, 30, 35, 40, 70} (+) | 1.00 | 0.060 |
| L25 down c229 | MLP | a%100 in {22..24, 26..29, 31..34, 36..39, 41..44, 46..49, 52..54, 59} (+) | 1.00 | 0.450 |
| L25 down c378 | MLP | a%100 in {2} (+) | 1.00 | 0.020 |
| L25 down c41 | MLP | a%100 in {55} (+) | 1.00 | 0.010 |
| L25 down c192 | MLP | a%100 in {7..10, 12..17, 19} (+) | 1.00 | 0.180 |
| L25 down c100 | MLP | a%100 in {89..98} (-) | 1.00 | 0.160 |
| L25 down c553 | MLP | a%100 in {90, 92..97} (+) | 1.00 | 0.070 |
| L25 down c84 | MLP | a%100 in {90} (-) | 1.00 | 0.010 |
| L25 down c19 | MLP | a%100 in {91..97} (-) | 1.00 | 0.070 |

**code 2a-165.21 (L26): 21 comps, tells apart 25/100, coverage 0.83, overlap 2.36 (random 2.22, p 0.63)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 down c174 | MLP | a%100 in {0, 20, 24..25, 28..30, 33..50, 70, 75, 80} (+) | 1.00 | 0.280 |
| L26 down c84 | MLP | a%100 in {0..21, 25..26, 29..50, 52..55, 58..60, 65, 67, 69..70, 73, 75, 80, 85..86, 88..99} (-) | 1.00 | 0.520 |
| L26 down c43 | MLP | a%100 in {1, 55, 91..94, 96} (-) | 1.00 | 0.990 |
| L26 down c525 | MLP | a%100 in {1..2} (+) | 1.00 | 0.030 |
| L26 down c291 | MLP | a%100 in {1..3} (+) | 1.00 | 0.040 |
| L26 down c655 | MLP | a%100 in {11} (+) | 1.00 | 0.010 |
| L26 down c56 | MLP | a%100 in {18} (+) | 1.00 | 0.010 |
| L26 down c760 | MLP | a%100 in {18} (-) | 1.00 | 0.010 |
| L26 down c77 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L26 down c90 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L26 down c47 | MLP | a%100 in {2, 55, 60, 64..65, 70, 75, 80, 82, 84..86, 88..98} (-) | 1.00 | 0.350 |
| L26 down c130 | MLP | a%100 in {2..8} (-) | 1.00 | 0.110 |
| L26 down c371 | MLP | a%100 in {5..10, 12..17, 19, 22..24, 26..29, 33..34, 37..38} (+) | 1.00 | 0.450 |
| L26 down c243 | MLP | a%100 in {50} (+) | 1.00 | 0.010 |
| L26 down c425 | MLP | a%100 in {50} (+) | 1.00 | 0.010 |
| L26 down c250 | MLP | a%100 in {55} (-) | 1.00 | 0.010 |
| L26 down c818 | MLP | a%100 in {82, 84, 86..89, 98..99} (+) | 1.00 | 0.101 |
| L26 down c171 | MLP | a%100 in {90..98} (+) | 1.00 | 0.100 |
| L26 down c68 | MLP | a%100 in {90} (+) | 1.00 | 0.020 |
| L26 down c432 | MLP | a%100 in {90} (+) | 1.00 | 0.010 |
| L26 down c399 | MLP | a%100 in {92} (-) | 1.00 | 0.010 |

**code 2a-165.22 (L27): 26 comps, tells apart 24/100, coverage 0.69, overlap 1.71 (random 2.52, p 0.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 down c218 | MLP | a%100 in {0, 20, 25, 30, 35, 40, 45, 50, 60, 65, 70, 75, 80, 85} (+) | 1.00 | 0.142 |
| L27 down c40 | MLP | a%100 in {0, 50, 90} (+) | 1.00 | 0.030 |
| L27 down c52 | MLP | a%100 in {1..2} (-) | 1.00 | 0.020 |
| L27 down c26 | MLP | a%100 in {1..3, 11, 18, 55, 92} (+) | 1.00 | 1.000 |
| L27 down c4 | MLP | a%100 in {1..3} (-) | 1.00 | 0.030 |
| L27 down c284 | MLP | a%100 in {1..5} (-) | 1.00 | 0.080 |
| L27 down c2 | MLP | a%100 in {11, 18, 55, 90..92, 94, 96, 99} (+) | 1.00 | 1.000 |
| L27 down c220 | MLP | a%100 in {11} (+) | 1.00 | 0.010 |
| L27 down c53 | MLP | a%100 in {11} (-) | 1.00 | 0.010 |
| L27 down c757 | MLP | a%100 in {11} (-) | 1.00 | 0.010 |
| L27 down c89 | MLP | a%100 in {18} (+) | 1.00 | 0.020 |
| L27 down c169 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L27 down c273 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L27 down c57 | MLP | a%100 in {22..24, 26..29, 31..34, 36..39, 41..44, 46..49, 53, 59} (+) | 1.00 | 0.395 |
| L27 down c303 | MLP | a%100 in {2} (-) | 1.00 | 0.010 |
| L27 down c423 | MLP | a%100 in {2} (-) | 1.00 | 0.010 |
| L27 down c489 | MLP | a%100 in {3..10, 12..17, 19} (+) | 1.00 | 0.210 |
| L27 down c20 | MLP | a%100 in {50} (+) | 1.00 | 0.010 |
| L27 down c541 | MLP | a%100 in {55} (-) | 1.00 | 0.041 |
| L27 down c396 | MLP | a%100 in {90..98} (+) | 1.00 | 0.130 |
| L27 down c776 | MLP | a%100 in {91..97} (-) | 1.00 | 0.090 |
| L27 down c154 | MLP | a%100 in {92..94, 96..97} (-) | 1.00 | 0.050 |
| L27 down c594 | MLP | a%100 in {99} (+) | 1.00 | 0.010 |
| L27 down c34 | MLP | a%100 in {99} (-) | 1.00 | 0.010 |
| L27 down c111 | MLP | a%100 in {99} (-) | 1.00 | 0.010 |
| L27 down c390 | MLP | a%100 in {99} (-) | 1.00 | 0.010 |

**code 2a-165.23 (L28): 23 comps, tells apart 24/100, coverage 0.75, overlap 2.39 (random 2.29, p 0.57)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 down c158 | MLP | a%100 in {0, 5..17, 20, 25, 30, 40, 55, 60, 65, 70, 75, 80} (-) | 1.00 | 1.000 |
| L28 down c809 | MLP | a%100 in {0, 50, 60, 70, 75, 80, 85} (-) | 1.00 | 0.374 |
| L28 down c153 | MLP | a%100 in {1, 9, 12..17, 19, 90..98} (-) | 1.00 | 0.920 |
| L28 down c156 | MLP | a%100 in {1..21, 55, 69, 73, 90..97} (+) | 1.00 | 0.720 |
| L28 down c215 | MLP | a%100 in {1..2} (-) | 1.00 | 0.020 |
| L28 down c0 | MLP | a%100 in {1..3, 8..9, 11, 18, 55, 66, 82, 84..88, 99} (+) | 1.00 | 1.000 |
| L28 down c8 | MLP | a%100 in {1..7} (-) | 1.00 | 0.110 |
| L28 down c536 | MLP | a%100 in {11} (+) | 1.00 | 0.010 |
| L28 down c690 | MLP | a%100 in {11} (+) | 1.00 | 0.010 |
| L28 down c151 | MLP | a%100 in {15, 20, 25, 30} (+) | 0.99 | 0.041 |
| L28 down c260 | MLP | a%100 in {18} (-) | 1.00 | 0.010 |
| L28 down c252 | MLP | a%100 in {22..24, 26..29, 31..34, 36..39, 41..44, 46..49, 53} (+) | 1.00 | 0.241 |
| L28 down c357 | MLP | a%100 in {2} (-) | 1.00 | 0.010 |
| L28 down c140 | MLP | a%100 in {4} (-) | 1.00 | 0.010 |
| L28 down c306 | MLP | a%100 in {5..7} (+) | 1.00 | 0.030 |
| L28 down c159 | MLP | a%100 in {50, 55} (-) | 1.00 | 0.020 |
| L28 down c421 | MLP | a%100 in {55} (-) | 1.00 | 0.010 |
| L28 down c181 | MLP | a%100 in {7..10, 12..17, 19} (+) | 1.00 | 0.130 |
| L28 down c829 | MLP | a%100 in {90, 95} (+) | 1.00 | 0.040 |
| L28 down c147 | MLP | a%100 in {90..91, 94..95} (-) | 0.99 | 0.041 |
| L28 down c4 | MLP | a%100 in {90..98} (-) | 1.00 | 0.137 |
| L28 down c431 | MLP | a%100 in {90} (-) | 1.00 | 0.010 |
| L28 down c385 | MLP | a%100 in {91..97} (+) | 1.00 | 0.070 |

**code 2a-165.24 (L29): 23 comps, tells apart 23/100, coverage 0.76, overlap 1.95 (random 2.32, p 0.09)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L29 down c344 | MLP | a%100 in {0} (+) | 1.00 | 0.120 |
| L29 down c9 | MLP | a%100 in {1, 11, 18, 55, 90, 99} (+) | 1.00 | 1.000 |
| L29 down c34 | MLP | a%100 in {1..2, 11, 18, 55, 90..97} (-) | 1.00 | 1.000 |
| L29 down c3 | MLP | a%100 in {1..3, 11, 18, 55} (-) | 1.00 | 1.000 |
| L29 down c135 | MLP | a%100 in {1..4} (+) | 1.00 | 0.060 |
| L29 down c841 | MLP | a%100 in {11, 93} (+) | 0.52 | 0.015 |
| L29 down c616 | MLP | a%100 in {11} (+) | 1.00 | 0.010 |
| L29 down c299 | MLP | a%100 in {15, 20, 25, 30, 35, 40, 45, 60, 65, 70, 75, 80, 85} (+) | 1.00 | 0.131 |
| L29 down c26 | MLP | a%100 in {18} (-) | 1.00 | 0.010 |
| L29 down c114 | MLP | a%100 in {2..6} (-) | 1.00 | 0.060 |
| L29 down c17 | MLP | a%100 in {22..24, 26..29, 31..34, 36..39, 41..44, 46..49, 53, 59} (+) | 1.00 | 0.440 |
| L29 down c373 | MLP | a%100 in {2} (+) | 1.00 | 0.010 |
| L29 down c158 | MLP | a%100 in {3..10, 12..17, 19} (-) | 1.00 | 0.250 |
| L29 down c31 | MLP | a%100 in {55} (+) | 1.00 | 0.030 |
| L29 down c271 | MLP | a%100 in {55} (+) | 1.00 | 0.010 |
| L29 down c303 | MLP | a%100 in {55} (-) | 1.00 | 0.010 |
| L29 down c4 | MLP | a%100 in {8..21, 25, 30, 40, 55, 67..68, 74, 78, 82, 84, 89} (+) | 1.00 | 1.000 |
| L29 down c967 | MLP | a%100 in {9, 12..17, 19} (-) | 1.00 | 0.090 |
| L29 down c393 | MLP | a%100 in {90..98} (+) | 1.00 | 0.130 |
| L29 down c490 | MLP | a%100 in {90} (-) | 1.00 | 0.010 |
| L29 down c124 | MLP | a%100 in {91..97} (+) | 1.00 | 0.070 |
| L29 down c534 | MLP | a%100 in {99} (+) | 1.00 | 0.010 |
| L29 down c161 | MLP | a%100 in {99} (-) | 1.00 | 0.108 |

**code 2a-165.25 (L30): 15 comps, tells apart 19/100, coverage 0.60, overlap 1.87 (random 1.85, p 0.52)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 down c1 | MLP | a%100 in {0, 7..9, 11, 13, 55} (+) | 1.00 | 1.000 |
| L30 down c92 | MLP | a%100 in {0..3, 11, 50, 55, 90..98} (+) | 1.00 | 0.890 |
| L30 down c18 | MLP | a%100 in {1, 11, 13, 18, 29, 31, 33..34, 39, 41, 43..44, 47, 49, 55, 60, 65, 75, 82, 85..87, 92..93} (-) | 1.00 | 1.000 |
| L30 down c32 | MLP | a%100 in {1, 11, 55, 90..94} (-) | 1.00 | 1.000 |
| L30 down c377 | MLP | a%100 in {1, 55} (-) | 0.89 | 0.045 |
| L30 down c216 | MLP | a%100 in {1..5} (+) | 1.00 | 0.080 |
| L30 down c9 | MLP | a%100 in {11} (-) | 0.98 | 0.011 |
| L30 down c548 | MLP | a%100 in {24, 26, 28..29, 31..34, 36..39, 41..44, 46..49, 53} (-) | 1.00 | 0.240 |
| L30 down c55 | MLP | a%100 in {3} (-) | 0.81 | 0.012 |
| L30 down c88 | MLP | a%100 in {4..10, 12..17, 19} (-) | 1.00 | 0.270 |
| L30 down c668 | MLP | a%100 in {55} (+) | 1.00 | 0.080 |
| L30 down c791 | MLP | a%100 in {55} (-) | 1.00 | 0.010 |
| L30 down c59 | MLP | a%100 in {90..98} (+) | 1.00 | 0.130 |
| L30 down c314 | MLP | a%100 in {99} (+) | 1.00 | 0.010 |
| L30 down c747 | MLP | a%100 in {99} (-) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2a-166</b> `a%100` @ `op` (add) — single component; 1 comps, L7; tells apart 4/100 classes (best member 4); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.04 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.72 /  / 0.72 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L6 MLP |
| CKA(arrangement before, joint write) | 0.28 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.02 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.33 2:0.11 40:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.24 10:0.09 2:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.04 / 1.00 |
| best source position (CKA) | `a` (0.19) |

<details><summary>codes and components</summary>

**code 2a-166.0 (L7): 1 comps, tells apart 4/100, coverage 0.04, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L7 o c455 | H0 | a%100 in {1..3, 18} (+) | 1.00 | 0.970 |

</details>

</details>

<details><summary><b>2a-167</b> `a%100` @ `op` (add) — single component; 1 comps, L7; tells apart 5/100 classes (best member 5); on sub: 2s-216 (member overlap 0.33)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.11 |
| classes told apart: joint / best code / best member | 5 /  / 5 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.61 /  / 0.61 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L6 MLP |
| CKA(arrangement before, joint write) | 0.37 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.33 2:0.11 40:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.40 20:0.06 3:0.05) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.09 / 1.00 |
| best source position (CKA) | `a` (0.33) |

<details><summary>codes and components</summary>

**code 2a-167.0 (L7): 1 comps, tells apart 5/100, coverage 0.11, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L7 o c14 | H20 | a%100 in {1, 5, 7, 10, 15, 20, 25, 35, 45, 55, 96} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2a-168</b> `a%100` @ `op` (add) — copy from `a`; 1 comps, L7; tells apart 1/100 classes (best member 1); on sub: 2s-214 (member overlap 0.33)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.26 |
| classes told apart: joint / best code / best member | 1 /  / 1 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.37 /  / 0.37 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L6 MLP |
| CKA(arrangement before, joint write) | 0.51 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.33 2:0.11 40:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.67) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.23 / 1.00 |
| best source position (CKA) | `a` (0.56) |

<details><summary>codes and components</summary>

**code 2a-168.0 (L7): 1 comps, tells apart 1/100, coverage 0.26, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L7 o c3 | H5 | a%100 in {4..11, 15, 18..20, 26, 29, 55, 60, 62, 65..66, 70, 79..81, 86..88} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2a-169</b> `a%100` @ `op` (add) — single component; 1 comps, L8; tells apart 4/100 classes (best member 4); on sub: 2s-218 (member overlap 0.50)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.03 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.31 /  / 0.31 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L7 MLP |
| CKA(arrangement before, joint write) | 0.28 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.33 2:0.11 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.18 2:0.13 41:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.03 / 1.00 |
| best source position (CKA) | `a` (0.12) |

<details><summary>codes and components</summary>

**code 2a-169.0 (L8): 1 comps, tells apart 4/100, coverage 0.03, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L8 o c7 | H23 | a%100 in {11, 18, 55} (+) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2a-170</b> `a%100` @ `op` (add) — block code, copy from `a`; 2 comps, L9; tells apart 12/100 classes (best member 3); on sub: 2s-219 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.49 |
| classes told apart: joint / best code / best member | 12 /  / 3 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.18 / 1.00 / 0.85 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.90 /  / 0.35 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.03 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L8 MLP |
| CKA(arrangement before, joint write) | 0.41 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.33 2:0.11 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.40 40:0.07 3:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.14 / 1.58 |
| best source position (CKA) | `a` (0.53) |

<details><summary>codes and components</summary>

**code 2a-170.0 (L9): 2 comps, tells apart 12/100, coverage 0.49, overlap 1.18 (random 1.00, p 0.89)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L9 o c183 | H11 | a%100 in {0..3, 11, 18, 55..59, 61..64, 66, 68, 71, 74..77, 79, 81..82, 85..89, 91..92, 94..95, 98} (-) | 1.00 | 1.000 |
| L9 o c9 | H11 | a%100 in {2..4, 11, 19, 25, 30, 35, 38..39, 42, 48..50, 55, 60, 75, 80, 86, 91..92, 94, 96} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2a-171</b> `a%100` @ `op` (add) — single component; 1 comps, L10; tells apart 2/100 classes (best member 2); on sub: 2s-220 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.12 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.18 /  / 0.18 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L9 MLP |
| CKA(arrangement before, joint write) | 0.16 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.33 2:0.11 3:0.06) |
| joint write: shape (spectrum k:share) | line (2:0.19 40:0.10 3:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.03 / 1.00 |
| best source position (CKA) | `a` (0.11) |

<details><summary>codes and components</summary>

**code 2a-171.0 (L10): 1 comps, tells apart 2/100, coverage 0.12, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L10 o c1 | H30 | a%100 in {0..2, 11, 18, 50, 55, 60, 86, 93..94, 96} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2a-172</b> `a%100` @ `op` (add) — single component; 1 comps, L11; tells apart 4/100 classes (best member 4); on sub: 2s-221 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.05 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.18 /  / 0.18 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L10 MLP |
| CKA(arrangement before, joint write) | 0.34 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.34 2:0.11 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.22 11:0.08 5:0.05) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.04 / 1.00 |
| best source position (CKA) | `a` (0.15) |

<details><summary>codes and components</summary>

**code 2a-172.0 (L11): 1 comps, tells apart 4/100, coverage 0.05, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L11 o c4 | H20 | a%100 in {11, 18, 55, 92..93} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2a-173</b> `a%100` @ `op` (add) — block code; 2 comps, L12; tells apart 7/100 classes (best member 2); on sub: 2s-222 (member overlap 0.25)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.16 |
| classes told apart: joint / best code / best member | 7 /  / 2 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.06 / 1.00 / 0.81 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.98 /  / 0.56 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L11 MLP |
| CKA(arrangement before, joint write) | 0.48 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.34 2:0.10 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.23 2:0.09 20:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.05 / 1.18 |
| best source position (CKA) | `a` (0.18) |

<details><summary>codes and components</summary>

**code 2a-173.0 (L12): 2 comps, tells apart 7/100, coverage 0.16, overlap 1.06 (random 1.00, p 0.77)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L12 o c319 | H20 | a%100 in {1..2, 11, 55} (-) | 1.00 | 0.960 |
| L12 o c209 | H20 | a%100 in {11, 18, 62, 64, 86, 88..89, 91, 94..98} (+) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2a-174</b> `a%100` @ `op` (add) — block code; 2 comps, L13 L14; tells apart 10/100 classes (best member 3); on sub: 2s-224 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.03 |
| classes told apart: joint / best code / best member | 10 /  / 3 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.65 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.89 /  / 0.30 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L12 MLP |
| CKA(arrangement before, joint write) | 0.40 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.32 2:0.10 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.23 2:0.09 20:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.08 / 1.91 |
| best source position (CKA) | `a` (0.29) |

<details><summary>codes and components</summary>

**code 2a-174.0 (L13 L14): 2 comps, tells apart 10/100, coverage 0.03, overlap 1.00 (random 1.00, p 0.73)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 o c0 | H16 | a%100 in {55, 93} (+) | 1.00 | 1.000 |
| L14 o c59 | H26 | a%100 in {1} (+) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2a-176</b> `a%100` @ `op` (add) — single component; 1 comps, L15; tells apart 3/100 classes (best member 3); on sub: 2s-226 (member overlap 0.50)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.14 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.37 /  / 0.37 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L14 MLP |
| CKA(arrangement before, joint write) | 0.53 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.02 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.31 2:0.10 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.43 2:0.20) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.12 / 1.00 |
| best source position (CKA) | `a` (0.24) |

<details><summary>codes and components</summary>

**code 2a-176.0 (L15): 1 comps, tells apart 3/100, coverage 0.14, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 o c3 | H5 | a%100 in {1..5, 10..11, 55, 90..94, 96} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2a-175</b> `a%100` @ `op` (add) — block code; 2 comps, L15 L16; tells apart 6/100 classes (best member 3); on sub: 2s-230 (member overlap 0.50)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.32 |
| classes told apart: joint / best code / best member | 6 /  / 3 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.06 / 1.00 / 0.76 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.62 /  / 0.53 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L14 MLP |
| CKA(arrangement before, joint write) | 0.52 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.31 2:0.10 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (2:0.22 1:0.16 4:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.10 / 1.82 |
| best source position (CKA) | `a` (0.37) |

<details><summary>codes and components</summary>

**code 2a-175.0 (L15 L16): 2 comps, tells apart 6/100, coverage 0.32, overlap 1.06 (random 1.00, p 0.79)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 o c131 | H28 | a%100 in {89..98} (-) | 1.00 | 0.100 |
| L16 o c207 | H30 | a%100 in {2, 11..12, 16..18, 21, 30, 34, 40, 45..47, 49..50, 60, 65, 70, 78..79, 84, 92..93, 99} (+) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2a-177</b> `a%100` @ `op` (add) — tiling; 22 comps, L16; tells apart 36/100 classes (best member 7); on sub: 2s-229 (member overlap 0.47)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.38 |
| classes told apart: joint / best code / best member | 36 /  / 7 (of 100) |
| members whose removal merges classes | 0.55 |
| support overlap (1 = tiling) / random sets / p | 1.63 / 2.24 / 0.01 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.51 /  / 0.13 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.97 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 MLP |
| CKA(arrangement before, joint write) | 0.57 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.10 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.31 2:0.11 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.11 2:0.11 3:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.15 / 5.85 |
| best source position (CKA) | `a` (0.46) |

<details><summary>codes and components</summary>

**code 2a-177.0 (L16): 22 comps, tells apart 36/100, coverage 0.38, overlap 1.63 (random 2.33, p 0.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c197 | H22 | a%100 in {0, 50, 55, 90} (+) | 1.00 | 0.060 |
| L16 o c249 | H22 | a%100 in {0, 90, 92, 96, 98..99} (-) | 1.00 | 0.060 |
| L16 o c267 | H22 | a%100 in {0, 99} (-) | 1.00 | 0.020 |
| L16 o c268 | H22 | a%100 in {1, 21, 91} (+) | 1.00 | 0.030 |
| L16 o c61 | H22 | a%100 in {1..4} (+) | 1.00 | 0.110 |
| L16 o c188 | H22 | a%100 in {10, 90} (-) | 1.00 | 0.020 |
| L16 o c67 | H22 | a%100 in {11} (+) | 1.00 | 0.010 |
| L16 o c392 | H22 | a%100 in {11} (+) | 1.00 | 0.010 |
| L16 o c355 | H22 | a%100 in {18} (+) | 1.00 | 0.010 |
| L16 o c322 | H22 | a%100 in {18} (-) | 1.00 | 0.010 |
| L16 o c463 | H22 | a%100 in {1} (+) | 1.00 | 0.010 |
| L16 o c337 | H22 | a%100 in {2, 92} (-) | 1.00 | 0.110 |
| L16 o c329 | H22 | a%100 in {3, 13, 93} (+) | 1.00 | 0.030 |
| L16 o c492 | H22 | a%100 in {4, 14, 94} (+) | 1.00 | 0.082 |
| L16 o c261 | H22 | a%100 in {55} (+) | 1.00 | 0.010 |
| L16 o c170 | H22 | a%100 in {6, 16, 96} (-) | 1.00 | 0.030 |
| L16 o c237 | H22 | a%100 in {60..62} (-) | 0.99 | 0.032 |
| L16 o c119 | H22 | a%100 in {7} (-) | 1.00 | 0.010 |
| L16 o c243 | H22 | a%100 in {81..89} (-) | 1.00 | 0.090 |
| L16 o c58 | H22 | a%100 in {90..91} (-) | 0.97 | 0.026 |
| L16 o c160 | H22 | a%100 in {90..97} (-) | 1.00 | 0.120 |
| L16 o c314 | H22 | a%100 in {97} (-) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2a-178</b> `a%100` @ `op` (add) — block code; 2 comps, L17; tells apart 6/100 classes (best member 4); on sub: 2s-231 (member overlap 0.33)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.13 |
| classes told apart: joint / best code / best member | 6 /  / 4 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.23 / 1.00 / 0.92 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.26 /  / 0.23 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L16 MLP |
| CKA(arrangement before, joint write) | 0.54 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.31 2:0.10 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.19 2:0.17 3:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.08 / 1.72 |
| best source position (CKA) | `a` (0.26) |

<details><summary>codes and components</summary>

**code 2a-178.0 (L17): 2 comps, tells apart 6/100, coverage 0.13, overlap 1.23 (random 1.00, p 0.90)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 o c17 | H9 | a%100 in {1..3, 11, 55, 90..91, 93..94} (-) | 1.00 | 1.000 |
| L17 o c33 | H9 | a%100 in {91..97} (-) | 1.00 | 0.070 |

</details>

</details>

<details><summary><b>2a-180</b> `a%100` @ `op` (add) — single component; 1 comps, L18; tells apart 2/100 classes (best member 2); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.01 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.02 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L17 MLP |
| CKA(arrangement before, joint write) | 0.06 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.30 2:0.11 3:0.06) |
| joint write: shape (spectrum k:share) | line () |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.01 / 1.00 |
| best source position (CKA) | `a` (0.05) |

<details><summary>codes and components</summary>

**code 2a-180.0 (L18): 1 comps, tells apart 2/100, coverage 0.01, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 o c313 | H30 | a%100 in {90} (+) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2a-179</b> `a%100` @ `op` (add) — block code; 2 comps, L18; tells apart 6/100 classes (best member 5); on sub: 2s-231 (member overlap 0.33)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.07 |
| classes told apart: joint / best code / best member | 6 /  / 5 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.74 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.58 /  / 0.58 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L17 MLP |
| CKA(arrangement before, joint write) | 0.30 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.30 2:0.11 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.27 20:0.12 40:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.07 / 1.37 |
| best source position (CKA) | `a` (0.23) |

<details><summary>codes and components</summary>

**code 2a-179.0 (L18): 2 comps, tells apart 6/100, coverage 0.07, overlap 1.00 (random 1.00, p 0.69)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 o c442 | H1 | a%100 in {1..2, 50, 60, 65, 99} (+) | 1.00 | 1.000 |
| L18 o c495 | H17 | a%100 in {90} (+) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2a-181</b> `a%100` @ `op` (add) — single component; 1 comps, L19; tells apart 2/100 classes (best member 2); on sub: 2s-232 (member overlap 0.25)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.13 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.53 /  / 0.53 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L18 MLP |
| CKA(arrangement before, joint write) | 0.29 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.30 2:0.11 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.35 2:0.07 5:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.07 / 1.00 |
| best source position (CKA) | `a` (0.24) |

<details><summary>codes and components</summary>

**code 2a-181.0 (L19): 1 comps, tells apart 2/100, coverage 0.13, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L19 o c23 | H7 | a%100 in {1, 4, 18, 78, 88, 90, 92..98} (+) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2a-182</b> `a%100` @ `op` (add) — block code; 2 comps, L20 L21; tells apart 4/100 classes (best member 3); on sub: 2s-234 (member overlap 0.17)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.20 |
| classes told apart: joint / best code / best member | 4 /  / 3 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.62 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.99 |
| decoding acc. joint / best code / best member (chance) | 0.55 /  / 0.54 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L19 MLP |
| CKA(arrangement before, joint write) | 0.23 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.29 2:0.11 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.23 40:0.16 20:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.05 / 1.02 |
| best source position (CKA) | `a` (0.35) |

<details><summary>codes and components</summary>

**code 2a-182.0 (L20 L21): 2 comps, tells apart 4/100, coverage 0.20, overlap 1.00 (random 1.00, p 0.65)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 o c0 | H17 | a%100 in {1..3, 5..6, 8, 10..11, 15, 18, 20, 25, 30, 40, 60, 70, 75, 80, 90} (-) | 0.99 | 0.816 |
| L21 o c20 | H21 | a%100 in {55} (+) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2a-183</b> `a%100` @ `op` (add) — tiling; 5 comps, L20 L21; tells apart 5/100 classes (best member 3); on sub: 2s-236 (member overlap 0.22)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.15 |
| classes told apart: joint / best code / best member | 5 /  / 3 (of 100) |
| members whose removal merges classes | 0.60 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.20 / 0.04 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.77 /  / 0.76 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L19 MLP |
| CKA(arrangement before, joint write) | 0.48 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.29 2:0.11 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.26 20:0.09 41:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.05 / 1.04 |
| best source position (CKA) | `a` (0.15) |

<details><summary>codes and components</summary>

**code 2a-183.0 (L20 L21): 5 comps, tells apart 5/100, coverage 0.15, overlap 1.00 (random 1.20, p 0.04)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 o c270 | H3 | a%100 in {52} (+) | 1.00 | 0.010 |
| L20 o c349 | H3 | a%100 in {90} (+) | 1.00 | 0.010 |
| L21 o c1 | H23 | a%100 in {1..2, 11, 60, 70, 91..94, 96..97} (-) | 1.00 | 1.000 |
| L21 o c3 | H23 | a%100 in {55} (-) | 1.00 | 0.020 |
| L21 o c7 | H23 | a%100 in {99} (+) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2a-184</b> `a%100` @ `op` (add) — single component; 1 comps, L21; tells apart 1/100 classes (best member 1); on sub: 2s-235 (member overlap 0.33)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.39 |
| classes told apart: joint / best code / best member | 1 /  / 1 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.57 /  / 0.57 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L20 MLP |
| CKA(arrangement before, joint write) | 0.26 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.03 |
| share of the write inside the old arrangement's span | 0.20 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.30 2:0.11 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.62 4:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.20 / 1.00 |
| best source position (CKA) | `a` (0.31) |

<details><summary>codes and components</summary>

**code 2a-184.0 (L21): 1 comps, tells apart 1/100, coverage 0.39, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 o c222 | H18 | a%100 in {1, 12..14, 17..19, 21, 34, 36, 38..39, 41..44, 46..48, 55, 63, 66..69, 71..79, 81, 83..85, 90} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2a-185</b> `a%100` @ `op` (add) — block code; 3 comps, L22 L23; tells apart 7/100 classes (best member 6); on sub: 2s-237 (member overlap 0.02)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.38 |
| classes told apart: joint / best code / best member | 7 /  / 6 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.05 / 1.05 / 0.51 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.99 |
| decoding acc. joint / best code / best member (chance) | 0.62 /  / 0.56 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 MLP |
| CKA(arrangement before, joint write) | 0.39 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.29 2:0.11 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.39 2:0.13) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.11 / 1.17 |
| best source position (CKA) | `a` (0.27) |

<details><summary>codes and components</summary>

**code 2a-185.0 (L22 L23): 3 comps, tells apart 7/100, coverage 0.38, overlap 1.05 (random 1.08, p 0.47)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 o c2 | H24 | a%100 in {0..1, 8..9, 14..17, 19, 22, 47, 50, 55, 67, 80, 82..84, 87..88, 90..99} (-) | 0.99 | 0.744 |
| L22 o c120 | H24 | a%100 in {1..3} (+) | 1.00 | 0.030 |
| L23 o c36 | H14 | a%100 in {61..64, 66..68} (-) | 1.00 | 0.082 |

</details>

</details>

<details><summary><b>2a-186</b> `a%100` @ `op` (add) — block code; 3 comps, L23 L24; tells apart 5/100 classes (best member 3); on sub: 2s-239 (member overlap 0.12)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.07 |
| classes told apart: joint / best code / best member | 5 /  / 3 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.07 / 0.40 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.72 /  / 0.71 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L22 MLP |
| CKA(arrangement before, joint write) | 0.29 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.29 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.17 3:0.10 2:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.03 / 1.02 |
| best source position (CKA) | `a` (0.27) |

<details><summary>codes and components</summary>

**code 2a-186.0 (L23 L24): 3 comps, tells apart 5/100, coverage 0.07, overlap 1.00 (random 1.04, p 0.46)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 o c280 | H7 | a%100 in {88} (-) | 1.00 | 0.010 |
| L23 o c187 | H7 | a%100 in {98} (-) | 1.00 | 0.010 |
| L24 o c2 | H10 | a%100 in {1..3, 11, 55} (-) | 1.00 | 0.970 |

</details>

</details>

<details><summary><b>2a-187</b> `a%100` @ `op` (add) — single component; 1 comps, L25; tells apart 3/100 classes (best member 3); on sub: 2s-242 (member overlap 0.50)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.12 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.54 /  / 0.54 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L24 MLP |
| CKA(arrangement before, joint write) | 0.62 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.30 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.29 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.55 2:0.11) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.16 / 1.00 |
| best source position (CKA) | `a` (0.23) |

<details><summary>codes and components</summary>

**code 2a-187.0 (L25): 1 comps, tells apart 3/100, coverage 0.12, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L25 o c394 | H23 | a%100 in {0..2, 55, 90..94, 96..98} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2a-188</b> `a%100` @ `op` (add) — block code; 2 comps, L26 L27; tells apart 6/100 classes (best member 2); on sub: 2s-246 (member overlap 0.50)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.18 |
| classes told apart: joint / best code / best member | 6 /  / 2 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.71 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.99 |
| decoding acc. joint / best code / best member (chance) | 0.98 /  / 0.51 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.02 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L25 MLP |
| CKA(arrangement before, joint write) | 0.13 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.25 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.29 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.09 40:0.07 2:0.05) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.02 / 1.22 |
| best source position (CKA) | `a` (0.10) |

<details><summary>codes and components</summary>

**code 2a-188.0 (L26 L27): 2 comps, tells apart 6/100, coverage 0.18, overlap 1.00 (random 1.00, p 0.71)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 o c19 | H10 | a%100 in {0..1, 10..14, 16..19, 52, 55, 57, 60, 62, 66} (-) | 1.00 | 1.000 |
| L27 o c5 | H14 | a%100 in {51} (+) | 0.99 | 0.990 |

</details>

</details>

<details><summary><b>2a-189</b> `a%100` @ `op` (add) — single component; 1 comps, L28; tells apart 2/100 classes (best member 2); on sub: 2s-247 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.19 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.45 /  / 0.45 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L27 MLP |
| CKA(arrangement before, joint write) | 0.13 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.25 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.29 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (3:0.27 2:0.12 6:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.06 / 1.00 |
| best source position (CKA) | `a` (0.15) |

<details><summary>codes and components</summary>

**code 2a-189.0 (L28): 1 comps, tells apart 2/100, coverage 0.19, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 o c122 | H10 | a%100 in {0..1, 3, 8, 26, 43, 47, 49..50, 62, 64, 90..92, 94..97, 99} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2a-190</b> `a%100` @ `op` (add) — block code; 2 comps, L29 L30; tells apart 10/100 classes (best member 3); on sub: 2s-251 (member overlap 0.50)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.40 |
| classes told apart: joint / best code / best member | 10 /  / 3 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.02 / 1.00 / 0.68 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.93 /  / 0.37 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.10 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L28 MLP |
| CKA(arrangement before, joint write) | 0.34 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.16 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.29 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.24 2:0.11 40:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.07 / 1.46 |
| best source position (CKA) | `a` (0.17) |

<details><summary>codes and components</summary>

**code 2a-190.0 (L29 L30): 2 comps, tells apart 10/100, coverage 0.40, overlap 1.02 (random 1.00, p 0.70)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L29 o c1 | H25 | a%100 in {0, 55, 90..94, 96..97} (+) | 1.00 | 1.000 |
| L30 o c446 | H18 | a%100 in {5, 13, 37, 39, 41, 43, 47, 50, 52, 55..58, 61..62, 64, 66..69, 71..72, 74, 76..79, 81..82, 84, 87..88} (-) | 1.00 | 1.000 |

</details>

</details>

</details>

<details><summary>`a//10`: 5 mechanisms, 54 components</summary>

<details><summary><b>2a-193</b> `a//10` @ `op` (add) — block code, 5 codes of the same shape; 9 comps, L0 L1 L3 L19 L21 L22 L23; tells apart 5/11 classes (best member 4); on sub: 2s-259 (member overlap 0.27)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.27 |
| classes told apart: joint / best code / best member | 5 / 5 / 4 (of 11) |
| members whose removal merges classes | 0.11 |
| support overlap (1 = tiling) / random sets / p | 3.33 / 2.32 / 0.98 |
| mean CKA between its codes | 0.90 |
| purity of the joint write (per prompt) | 0.92 |
| decoding acc. joint / best code / best member (chance) | 0.69 / 0.74 / 0.65 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.06 |
| consumers / read jointly | 10 / 4 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | embed |
| CKA(arrangement before, joint write) | 0.00 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.00 |
| write energy / code energy before | 3179995793402195022605640007680.00 |
| best source position (CKA) | `a` (0.48) |

<details><summary>codes and components</summary>

**code 2a-193.0 (L0 L1): 2 comps, tells apart 5/11, coverage 0.27, overlap 1.00 (random 1.00, p 0.55)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 o c7 | H0 | a//10 in {0, 10} (+) | 0.97 | 1.000 |
| L1 o c465 | H24 | a//10 in {9} (+) | 0.90 | 0.131 |

**code 2a-193.1 (L3): 1 comps, tells apart 4/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 o c243 | H15 | a//10 in {9} (-) | 0.96 | 0.210 |

**code 2a-193.2 (L19): 2 comps, tells apart 4/11, coverage 0.18, overlap 1.00 (random 1.00, p 0.51)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L19 down c69 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L19 down c70 | MLP | a//10 in {9} (+) | 0.91 | 0.200 |

**code 2a-193.3 (L21 L22): 2 comps, tells apart 3/11, coverage 0.18, overlap 1.00 (random 1.05, p 0.50)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c310 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L22 o c10 | H15 | a//10 in {9} (-) | 0.98 | 0.110 |

**code 2a-193.4 (L23): 2 comps, tells apart 5/11, coverage 0.18, overlap 1.00 (random 1.00, p 0.59)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 down c322 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L23 down c294 | MLP | a//10 in {9} (+) | 0.93 | 0.394 |

</details>

</details>

<details><summary><b>2a-194</b> `a//10` @ `op` (add) — block code, 7 codes of the same shape, copy from `a`; 15 comps, L0 L1 L2 L7 L8 L15 L16; tells apart 8/11 classes (best member 6); on sub: 2s-258 (member overlap 0.03)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.82 |
| classes told apart: joint / best code / best member | 8 / 10 / 6 (of 11) |
| members whose removal merges classes | 0.13 |
| support overlap (1 = tiling) / random sets / p | 3.11 / 3.52 / 0.20 |
| mean CKA between its codes | 0.84 |
| purity of the joint write (per prompt) | 0.91 |
| decoding acc. joint / best code / best member (chance) | 0.82 / 0.75 / 0.57 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 0.98 |
| consumers / read jointly | 99 / 72 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | embed |
| CKA(arrangement before, joint write) | 0.00 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.00 |
| write energy / code energy before | 1847925602793693479732087095296.00 |
| best source position (CKA) | `a` (0.63) |

<details><summary>codes and components</summary>

**code 2a-194.0 (L0): 1 comps, tells apart 4/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 o c1 | H2 | a//10 in {0} (+) | 0.96 | 1.000 |

**code 2a-194.1 (L1): 1 comps, tells apart 5/11, coverage 0.18, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c359 | H18 | a//10 in {0..1} (+) | 0.93 | 1.000 |

**code 2a-194.2 (L1): 2 comps, tells apart 5/11, coverage 0.27, overlap 1.00 (random 1.00, p 0.55)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 down c73 | MLP | a//10 in {0..1} (+) | 0.96 | 0.333 |
| L1 down c302 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |

**code 2a-194.3 (L2): 4 comps, tells apart 8/11, coverage 0.45, overlap 1.40 (random 1.47, p 0.45)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 down c158 | MLP | a//10 in {0..2} (-) | 0.94 | 0.561 |
| L2 down c697 | MLP | a//10 in {0} (+) | 0.94 | 0.180 |
| L2 down c147 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L2 down c44 | MLP | a//10 in {9..10} (+) | 0.94 | 0.180 |

**code 2a-194.4 (L7 L8): 3 comps, tells apart 10/11, coverage 0.64, overlap 1.00 (random 1.33, p 0.13)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L7 down c61 | MLP | a//10 in {5..8} (-) | 0.92 | 0.380 |
| L7 down c703 | MLP | a//10 in {9} (-) | 0.92 | 0.270 |
| L8 down c17 | MLP | a//10 in {0..1} (+) | 0.90 | 0.270 |

**code 2a-194.5 (L15): 2 comps, tells apart 5/11, coverage 0.27, overlap 1.00 (random 1.00, p 0.56)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c49 | MLP | a//10 in {0..1} (+) | 0.90 | 0.281 |
| L15 down c398 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |

**code 2a-194.6 (L16): 2 comps, tells apart 5/11, coverage 0.45, overlap 1.00 (random 1.00, p 0.54)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c5 | MLP | a//10 in {0..2, 7} (-) | 0.92 | 0.730 |
| L16 down c38 | MLP | a//10 in {10} (-) | 0.98 | 0.012 |

</details>

</details>

<details><summary><b>2a-195</b> `a//10` @ `op` (add) — block code, 7 codes of the same shape, copy from `a`; 19 comps, L1 L2 L3 L4 L5 L6 L11; tells apart 9/11 classes (best member 6); on sub: 2s-258 (member overlap 0.06)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.91 |
| classes told apart: joint / best code / best member | 9 / 10 / 6 (of 11) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 6.70 / 4.20 / 1.00 |
| mean CKA between its codes | 0.79 |
| purity of the joint write (per prompt) | 0.94 |
| decoding acc. joint / best code / best member (chance) | 0.94 / 0.94 / 0.59 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 0.94 |
| consumers / read jointly | 152 / 141 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.61 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.00 |
| write energy / code energy before | 266.36 |
| best source position (CKA) | `a` (0.82) |

<details><summary>codes and components</summary>

**code 2a-195.0 (L1): 6 comps, tells apart 10/11, coverage 0.73, overlap 2.00 (random 1.86, p 0.72)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c413 | H26 | a//10 in {0..2} (+) | 0.92 | 0.370 |
| L1 o c377 | H26 | a//10 in {0} (-) | 0.94 | 0.180 |
| L1 o c315 | H26 | a//10 in {10} (+) | 1.00 | 0.010 |
| L1 o c231 | H26 | a//10 in {2..3, 8..10} (+) | 0.93 | 0.920 |
| L1 o c70 | H26 | a//10 in {7..10} (+) | 0.97 | 0.490 |
| L1 o c330 | H26 | a//10 in {9..10} (-) | 0.92 | 0.162 |

**code 2a-195.1 (L2): 2 comps, tells apart 7/11, coverage 0.82, overlap 1.11 (random 1.10, p 0.54)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 o c48 | H24 | a//10 in {0..4, 6..9} (-) | 0.93 | 0.550 |
| L2 o c20 | H24 | a//10 in {9} (+) | 0.91 | 0.200 |

**code 2a-195.2 (L3): 3 comps, tells apart 7/11, coverage 0.55, overlap 1.33 (random 1.33, p 0.55)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 down c43 | MLP | a//10 in {0..1} (+) | 0.93 | 0.240 |
| L3 down c3 | MLP | a//10 in {0..3} (-) | 0.91 | 1.000 |
| L3 down c39 | MLP | a//10 in {8..9} (-) | 0.97 | 0.420 |

**code 2a-195.3 (L3): 1 comps, tells apart 3/11, coverage 0.91, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 o c354 | H5 | a//10 in {0..4, 6..10} (+) | 0.97 | 0.580 |

**code 2a-195.4 (L4): 3 comps, tells apart 8/11, coverage 0.82, overlap 1.44 (random 1.33, p 0.65)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 down c14 | MLP | a//10 in {0..2, 7..10} (+) | 0.92 | 1.000 |
| L4 down c8 | MLP | a//10 in {2..4, 8..9} (-) | 0.92 | 0.910 |
| L4 down c140 | MLP | a//10 in {9} (+) | 0.91 | 0.160 |

**code 2a-195.5 (L5 L6): 3 comps, tells apart 9/11, coverage 0.55, overlap 1.00 (random 1.33, p 0.14)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 down c29 | MLP | a//10 in {0..2} (-) | 0.95 | 0.380 |
| L5 down c936 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L6 down c93 | MLP | a//10 in {8..9} (+) | 0.96 | 0.500 |

**code 2a-195.6 (L11): 1 comps, tells apart 4/11, coverage 0.36, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L11 down c221 | MLP | a//10 in {7..10} (+) | 0.94 | 0.330 |

</details>

</details>

<details><summary><b>2a-196</b> `a//10` @ `op` (add) — block code, 8 codes of the same shape; 9 comps, L13 L14 L17 L20 L22 L24 L26 L29; tells apart 2/11 classes (best member 2); on sub: 2s-258 (member overlap 0.02)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.09 |
| classes told apart: joint / best code / best member | 2 / 2 / 2 (of 11) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 9.00 / 2.38 / 1.00 |
| mean CKA between its codes | 1.00 |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.18 / 0.18 / 0.18 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.29 |
| consumers / read jointly | 3 / 3 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L13 attn |
| CKA(arrangement before, joint write) | 0.10 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.00 |
| write energy / code energy before | 0.01 |
| best source position (CKA) | `a` (0.12) |

<details><summary>codes and components</summary>

**code 2a-196.0 (L13): 1 comps, tells apart 2/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 down c161 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |

**code 2a-196.1 (L14): 2 comps, tells apart 2/11, coverage 0.09, overlap 2.00 (random 1.00, p 1.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L14 down c8 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L14 down c301 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |

**code 2a-196.2 (L17): 1 comps, tells apart 2/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 down c110 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |

**code 2a-196.3 (L20): 1 comps, tells apart 2/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 down c248 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |

**code 2a-196.4 (L22): 1 comps, tells apart 2/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 down c7 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |

**code 2a-196.5 (L24): 1 comps, tells apart 2/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 o c26 | H10 | a//10 in {10} (+) | 1.00 | 0.010 |

**code 2a-196.6 (L26): 1 comps, tells apart 2/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 down c737 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |

**code 2a-196.7 (L29): 1 comps, tells apart 2/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L29 down c564 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2a-197</b> `a//10` @ `op` (add) — block code; 2 comps, L18; tells apart 3/11 classes (best member 2); on sub: 2s-258 (member overlap 0.01)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.18 |
| classes told apart: joint / best code / best member | 3 /  / 2 (of 11) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.56 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.91 |
| decoding acc. joint / best code / best member (chance) | 0.27 /  / 0.18 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L18 attn |
| CKA(arrangement before, joint write) | 0.37 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.00 |

<details><summary>codes and components</summary>

**code 2a-197.0 (L18): 2 comps, tells apart 3/11, coverage 0.18, overlap 1.00 (random 1.00, p 0.56)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 down c55 | MLP | a//10 in {0} (+) | 0.91 | 0.100 |
| L18 down c51 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |

</details>

</details>

</details>

<details><summary>`a%50`: 2 mechanisms, 12 components</summary>

<details><summary><b>2a-191</b> `a%50` @ `op` (add) — block code, 9 codes of the same shape; 11 comps, L1 L2 L4 L6 L7 L9 L12 L15; tells apart 6/50 classes (best member 6); on sub: 2s-253 (member overlap 0.16)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.20 |
| classes told apart: joint / best code / best member | 6 / 6 / 6 (of 50) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 6.60 / 6.10 / 0.94 |
| mean CKA between its codes | 0.91 |
| purity of the joint write (per prompt) | 0.93 |
| decoding acc. joint / best code / best member (chance) | 0.17 / 0.11 / 0.11 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 0.96 |
| consumers / read jointly | 76 / 72 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.42 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 184.27 |
| arrangement before: shape (spectrum k:share) | irregular (2:0.18 40:0.13 20:0.11) |
| joint write: shape (spectrum k:share) | line (40:0.40 20:0.40) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.17 / 1.04 |
| best source position (CKA) | `a` (0.31) |

<details><summary>codes and components</summary>

**code 2a-191.0 (L1): 2 comps, tells apart 6/50, coverage 0.18, overlap 1.00 (random 1.62, p 0.17)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c388 | H26 | a%50 in {0, 10, 15, 20, 25, 30, 35, 40} (+) | 0.93 | 0.180 |
| L1 o c488 | H26 | a%50 in {1} (+) | 0.85 | 0.045 |

**code 2a-191.1 (L1): 1 comps, tells apart 4/50, coverage 0.10, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 down c43 | MLP | a%50 in {0, 10, 20, 30, 40} (+) | 0.92 | 0.210 |

**code 2a-191.2 (L2): 1 comps, tells apart 6/50, coverage 0.14, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 down c21 | MLP | a%50 in {0, 10, 20, 25, 30, 35, 40} (+) | 0.92 | 1.000 |

**code 2a-191.3 (L4): 2 comps, tells apart 6/50, coverage 0.12, overlap 1.17 (random 1.62, p 0.40)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 down c43 | MLP | a%50 in {0, 10, 20, 25, 30, 40} (-) | 0.97 | 0.170 |
| L4 down c638 | MLP | a%50 in {0} (-) | 0.94 | 0.020 |

**code 2a-191.4 (L6): 1 comps, tells apart 4/50, coverage 0.16, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L6 down c21 | MLP | a%50 in {0, 10, 20, 25, 30, 35, 40, 45} (-) | 0.92 | 0.260 |

**code 2a-191.5 (L7): 1 comps, tells apart 5/50, coverage 0.16, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L7 down c148 | MLP | a%50 in {0, 10, 15, 20, 25, 30, 35, 40} (-) | 0.95 | 0.190 |

**code 2a-191.6 (L9): 1 comps, tells apart 5/50, coverage 0.16, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L9 down c66 | MLP | a%50 in {0, 10, 15, 20, 25, 30, 35, 40} (-) | 0.95 | 0.179 |

**code 2a-191.7 (L12): 1 comps, tells apart 5/50, coverage 0.12, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L12 down c428 | MLP | a%50 in {0, 20, 25, 30, 35, 40} (-) | 0.90 | 0.160 |

**code 2a-191.8 (L15): 1 comps, tells apart 4/50, coverage 0.16, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c22 | MLP | a%50 in {0, 10, 15, 20, 25, 30, 35, 40} (+) | 0.91 | 0.170 |

</details>

</details>

<details><summary><b>2a-192</b> `a%50` @ `op` (add) — single component; 1 comps, L8; tells apart 2/50 classes (best member 2); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.02 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 50) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.04 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L8 attn |
| CKA(arrangement before, joint write) | 0.18 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (2:0.26 40:0.14 20:0.14) |
| joint write: shape (spectrum k:share) | line () |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.02 / 1.00 |

<details><summary>codes and components</summary>

**code 2a-192.0 (L8): 1 comps, tells apart 2/50, coverage 0.02, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L8 down c760 | MLP | a%50 in {0} (-) | 1.00 | 0.020 |

</details>

</details>

</details>

<details><summary>`a%10`: 1 mechanisms, 5 components</summary>

<details><summary><b>2a-150</b> `a%10` @ `op` (add) — 5 codes of the same shape, copy from `a`; 5 comps, L1 L2 L3 L5; tells apart 3/10 classes (best member 3); on sub: 2s-198 (member overlap 0.12)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.20 |
| classes told apart: joint / best code / best member | 3 / 3 / 3 (of 10) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 3.50 /  /  |
| mean CKA between its codes | 0.95 |
| purity of the joint write (per prompt) | 0.92 |
| decoding acc. joint / best code / best member (chance) | 0.28 / 0.29 / 0.29 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 0.96 |
| consumers / read jointly | 57 / 48 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.95 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.00 |
| write energy / code energy before | 104.94 |
| arrangement before: shape (spectrum k:share) | irregular (40:0.39 20:0.35 10:0.11) |
| joint write: shape (spectrum k:share) | line (20:0.45 40:0.44) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.20 / 1.01 |
| best source position (CKA) | `a` (0.53) |

<details><summary>codes and components</summary>

**code 2a-150.0 (L1): 1 comps, tells apart 3/10, coverage 0.10, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c110 | H26 | a%10 in {0} (-) | 0.93 | 0.170 |

**code 2a-150.1 (L1): 1 comps, tells apart 3/10, coverage 0.10, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 down c368 | MLP | a%10 in {0} (+) | 0.92 | 0.180 |

**code 2a-150.2 (L2): 1 comps, tells apart 3/10, coverage 0.20, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 down c26 | MLP | a%10 in {0, 5} (+) | 0.96 | 0.200 |

**code 2a-150.3 (L3): 1 comps, tells apart 3/10, coverage 0.10, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 down c22 | MLP | a%10 in {0} (+) | 0.91 | 0.180 |

**code 2a-150.4 (L5): 1 comps, tells apart 3/10, coverage 0.20, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 down c62 | MLP | a%10 in {0, 5} (+) | 0.92 | 0.210 |

</details>

</details>

</details>
