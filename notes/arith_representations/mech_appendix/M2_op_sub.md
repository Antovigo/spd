[← back to the report](../report_mechanisms.md)

# Every mechanism at `op`, sub

<details><summary>`a%100`: 53 mechanisms, 1059 components</summary>

<details><summary><b>2s-200</b> `a%100` @ `op` (sub) — block code; 3 comps, L0; tells apart 10/100 classes (best member 3); on add: 2a-152 (member overlap 0.17)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.10 |
| classes told apart: joint / best code / best member | 10 /  / 3 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.68 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.98 /  / 0.34 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.13 |
| consumers / read jointly | 6 / 4 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | embed |
| CKA(arrangement before, joint write) | 0.00 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.00 |
| write energy / code energy before | 161025411276796165030412288.00 |
| arrangement before: shape (spectrum k:share) | line () |
| joint write: shape (spectrum k:share) | line (1:0.51 2:0.12 20:0.05) |
| frequencies new in the write | 1 2 |
| write: shift-symmetric part / dimension (PR) | 0.19 / 1.35 |
| best source position (CKA) | `a` (0.34) |

<details><summary>codes and components</summary>

**code 2s-200.0 (L0): 3 comps, tells apart 10/100, coverage 0.10, overlap 1.00 (random 1.00, p 0.61)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 o c6 | H23 | a%100 in {0, 77, 99} (+) | 1.00 | 1.000 |
| L0 down c10 | MLP | a%100 in {1..2, 15, 25, 30} (-) | 1.00 | 1.000 |
| L0 down c46 | MLP | a%100 in {70, 84} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2s-202</b> `a%100` @ `op` (sub) — single component; 1 comps, L1; tells apart 5/100 classes (best member 5); on add: 2a-154 (member overlap 0.25)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.01 |
| classes told apart: joint / best code / best member | 5 /  / 5 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.09 /  / 0.09 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 4 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.10 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.04 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.33 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (10:0.15 20:0.13 30:0.11) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.04 / 1.00 |
| best source position (CKA) | `a` (0.12) |

<details><summary>codes and components</summary>

**code 2s-202.0 (L1): 1 comps, tells apart 5/100, coverage 0.01, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c416 | H18 | a%100 in {1} (-) | 1.00 | 0.111 |

</details>

</details>

<details><summary><b>2s-203</b> `a%100` @ `op` (sub) — block code, copy from `a`; 2 comps, L1; tells apart 7/100 classes (best member 6); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.18 |
| classes told apart: joint / best code / best member | 7 /  / 6 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.81 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.56 /  / 0.50 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.74 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.09 |
| write energy / code energy before | 0.07 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.33 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.51 2:0.12 3:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.19 / 1.31 |
| best source position (CKA) | `a` (0.53) |

<details><summary>codes and components</summary>

**code 2s-203.0 (L1): 2 comps, tells apart 7/100, coverage 0.18, overlap 1.00 (random 1.00, p 0.86)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c510 | H20 | a%100 in {0..8, 12} (+) | 1.00 | 1.000 |
| L1 o c112 | H20 | a%100 in {69..76} (-) | 0.97 | 0.100 |

</details>

</details>

<details><summary><b>2s-252</b> `a%100` @ `op` (sub) — block code, 2 codes of the same shape; 6 comps, L1 L30; tells apart 7/100 classes (best member 7); on add: 2a-156 (member overlap 0.21)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.13 |
| classes told apart: joint / best code / best member | 7 / 8 / 7 (of 100) |
| members whose removal merges classes | 0.33 |
| support overlap (1 = tiling) / random sets / p | 1.38 / 1.12 / 0.93 |
| mean CKA between its codes | 0.78 |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.46 / 0.46 / 0.34 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 12 / 9 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L1 attn |
| CKA(arrangement before, joint write) | 0.22 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 1.21 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.29 2:0.14 20:0.08) |
| joint write: shape (spectrum k:share) | line (1:0.07 2:0.07 3:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.02 / 1.04 |
| best source position (CKA) | `a` (0.19) |

<details><summary>codes and components</summary>

**code 2s-252.0 (L1): 5 comps, tells apart 8/100, coverage 0.13, overlap 1.23 (random 1.09, p 0.86)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 down c302 | MLP | a%100 in {0} (+) | 1.00 | 0.080 |
| L1 down c13 | MLP | a%100 in {1, 74, 90..91, 95..96, 98} (+) | 1.00 | 1.000 |
| L1 down c150 | MLP | a%100 in {1..6} (-) | 1.00 | 0.090 |
| L1 down c258 | MLP | a%100 in {1} (-) | 1.00 | 0.030 |
| L1 down c788 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |

**code 2s-252.1 (L30): 1 comps, tells apart 5/100, coverage 0.02, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 o c46 | H20 | a%100 in {1..2} (+) | 1.00 | 0.040 |

</details>

</details>

<details><summary><b>2s-201</b> `a%100` @ `op` (sub) — block code; 7 comps, L1 L2; tells apart 6/100 classes (best member 3); on add: 2a-153 (member overlap 0.30)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.26 |
| classes told apart: joint / best code / best member | 6 /  / 3 (of 100) |
| members whose removal merges classes | 0.43 |
| support overlap (1 = tiling) / random sets / p | 1.08 / 1.17 / 0.20 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.95 |
| decoding acc. joint / best code / best member (chance) | 0.63 /  / 0.61 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.04 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.28 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.05 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.33 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.30) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.14 / 2.80 |
| best source position (CKA) | `a` (0.32) |

<details><summary>codes and components</summary>

**code 2s-201.0 (L1 L2): 7 comps, tells apart 6/100, coverage 0.26, overlap 1.08 (random 1.16, p 0.24)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c10 | H11 | a%100 in {1..5, 7..9, 11, 34, 48, 54, 56, 58, 60, 64, 66, 76, 86, 93, 99} (+) | 1.00 | 1.000 |
| L2 o c26 | H8 | a%100 in {14} (-) | 0.89 | 0.011 |
| L2 o c506 | H8 | a%100 in {18} (-) | 0.99 | 0.021 |
| L2 o c93 | H24 | a%100 in {46..47} (+) | 0.86 | 0.017 |
| L2 o c77 | H24 | a%100 in {70} (+) | 1.00 | 0.010 |
| L2 o c278 | H24 | a%100 in {70} (+) | 1.00 | 0.010 |
| L2 o c389 | H24 | a%100 in {70} (+) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2s-204</b> `a%100` @ `op` (sub) — block code, copy from `a`; 9 comps, L1; tells apart 25/100 classes (best member 8); on add: 2a-155 (member overlap 0.75)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.63 |
| classes told apart: joint / best code / best member | 25 /  / 8 (of 100) |
| members whose removal merges classes | 0.78 |
| support overlap (1 = tiling) / random sets / p | 1.24 / 1.19 / 0.63 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.99 |
| decoding acc. joint / best code / best member (chance) | 0.74 /  / 0.26 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 3 / 3 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.34 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.65 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.33 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.34 2:0.22 3:0.14) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.42 / 4.46 |
| best source position (CKA) | `a` (0.60) |

<details><summary>codes and components</summary>

**code 2s-204.0 (L1): 9 comps, tells apart 25/100, coverage 0.63, overlap 1.24 (random 1.23, p 0.52)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c398 | H26 | a%100 in {0, 98..99} (-) | 0.99 | 0.255 |
| L1 o c188 | H26 | a%100 in {1..6} (+) | 1.00 | 0.090 |
| L1 o c320 | H26 | a%100 in {10..12} (-) | 1.00 | 0.030 |
| L1 o c403 | H26 | a%100 in {10..22} (-) | 1.00 | 0.191 |
| L1 o c108 | H26 | a%100 in {2, 12, 20..34, 36, 42} (-) | 0.99 | 0.281 |
| L1 o c106 | H26 | a%100 in {4, 34, 38..48} (+) | 0.99 | 0.166 |
| L1 o c66 | H26 | a%100 in {48..60} (-) | 0.97 | 0.156 |
| L1 o c201 | H26 | a%100 in {6..11} (+) | 1.00 | 0.101 |
| L1 o c469 | H26 | a%100 in {80, 90} (-) | 0.97 | 0.021 |

</details>

</details>

<details><summary><b>2s-205</b> `a%100` @ `op` (sub) — single component; 1 comps, L2; tells apart 2/100 classes (best member 2); on add: 2a-157 (member overlap 0.05)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.18 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.50 /  / 0.50 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L1 MLP |
| CKA(arrangement before, joint write) | 0.54 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.11 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.33 2:0.15 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.33 2:0.20 20:0.05) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.08 / 1.00 |
| best source position (CKA) | `a` (0.47) |

<details><summary>codes and components</summary>

**code 2s-205.0 (L2): 1 comps, tells apart 2/100, coverage 0.18, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 o c89 | H23 | a%100 in {0..6, 9, 69, 73, 78..79, 83..84, 86, 89, 96, 99} (+) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2s-206</b> `a%100` @ `op` (sub) — block code; 9 comps, L2; tells apart 15/100 classes (best member 5); on add: 2a-158 (member overlap 0.06)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.26 |
| classes told apart: joint / best code / best member | 15 /  / 5 (of 100) |
| members whose removal merges classes | 0.56 |
| support overlap (1 = tiling) / random sets / p | 1.23 / 1.21 / 0.57 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.69 /  / 0.54 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.06 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 attn |
| CKA(arrangement before, joint write) | 0.32 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.03 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.37 2:0.15 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.19 3:0.11 2:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.11 / 2.68 |

<details><summary>codes and components</summary>

**code 2s-206.0 (L2): 9 comps, tells apart 15/100, coverage 0.26, overlap 1.23 (random 1.19, p 0.57)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 down c17 | MLP | a%100 in {1..3, 11..12, 96..99} (+) | 1.00 | 1.000 |
| L2 down c111 | MLP | a%100 in {10..15} (-) | 1.00 | 0.062 |
| L2 down c90 | MLP | a%100 in {17..27} (+) | 1.00 | 0.221 |
| L2 down c237 | MLP | a%100 in {1} (+) | 1.00 | 0.020 |
| L2 down c51 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L2 down c561 | MLP | a%100 in {1} (-) | 1.00 | 0.020 |
| L2 down c275 | MLP | a%100 in {3} (+) | 1.00 | 0.010 |
| L2 down c11 | MLP | a%100 in {46} (-) | 0.84 | 0.010 |
| L2 down c6 | MLP | a%100 in {70} (-) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2s-208</b> `a%100` @ `op` (sub) — block code; 9 comps, L3; tells apart 19/100 classes (best member 8); on add: 2a-160 (member overlap 0.17)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.49 |
| classes told apart: joint / best code / best member | 19 /  / 8 (of 100) |
| members whose removal merges classes | 0.56 |
| support overlap (1 = tiling) / random sets / p | 1.53 / 1.22 / 0.95 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.75 /  / 0.66 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.11 |
| consumers / read jointly | 6 / 4 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 MLP |
| CKA(arrangement before, joint write) | 0.31 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.09 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.07 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.41 2:0.14 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.74 2:0.05) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.36 / 1.32 |
| best source position (CKA) | `a` (0.41) |

<details><summary>codes and components</summary>

**code 2s-208.0 (L3): 9 comps, tells apart 19/100, coverage 0.49, overlap 1.53 (random 1.21, p 0.94)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 o c49 | H7 | a%100 in {0..20, 33..34, 41..45, 47, 51..55, 62, 93, 95..99} (+) | 1.00 | 0.740 |
| L3 o c469 | H7 | a%100 in {0} (+) | 1.00 | 0.021 |
| L3 o c27 | H7 | a%100 in {1, 31, 41, 51, 61, 81, 91} (+) | 1.00 | 0.105 |
| L3 o c157 | H7 | a%100 in {31} (-) | 1.00 | 0.010 |
| L3 o c34 | H7 | a%100 in {41..42, 44..45} (+) | 1.00 | 0.060 |
| L3 o c35 | H7 | a%100 in {41..42} (+) | 1.00 | 0.050 |
| L3 o c11 | H7 | a%100 in {42, 44..45, 51..55} (+) | 1.00 | 0.160 |
| L3 o c190 | H7 | a%100 in {51..55, 57, 59, 61..63} (-) | 1.00 | 0.164 |
| L3 o c75 | H7 | a%100 in {70} (-) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2s-209</b> `a%100` @ `op` (sub) — block code; 19 comps, L3; tells apart 29/100 classes (best member 10); on add: 2a-161 (member overlap 0.06)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.54 |
| classes told apart: joint / best code / best member | 29 /  / 10 (of 100) |
| members whose removal merges classes | 0.74 |
| support overlap (1 = tiling) / random sets / p | 1.54 / 1.57 / 0.43 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.98 /  / 0.60 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 27 / 14 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L3 attn |
| CKA(arrangement before, joint write) | 0.52 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.14 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.45 2:0.13 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.33 2:0.14 3:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.32 / 4.23 |

<details><summary>codes and components</summary>

**code 2s-209.0 (L3): 19 comps, tells apart 29/100, coverage 0.54, overlap 1.54 (random 1.59, p 0.41)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 down c70 | MLP | a%100 in {0, 50, 60, 70, 80} (-) | 1.00 | 0.110 |
| L3 down c12 | MLP | a%100 in {0..3} (-) | 1.00 | 1.000 |
| L3 down c54 | MLP | a%100 in {1, 11, 51} (+) | 0.99 | 0.030 |
| L3 down c802 | MLP | a%100 in {1..3} (+) | 1.00 | 0.040 |
| L3 down c50 | MLP | a%100 in {1..5} (+) | 1.00 | 0.090 |
| L3 down c325 | MLP | a%100 in {10..12} (+) | 1.00 | 0.030 |
| L3 down c87 | MLP | a%100 in {10..23} (+) | 1.00 | 0.290 |
| L3 down c613 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L3 down c279 | MLP | a%100 in {30, 45} (-) | 1.00 | 0.030 |
| L3 down c189 | MLP | a%100 in {31, 49, 51, 53} (+) | 1.00 | 0.110 |
| L3 down c176 | MLP | a%100 in {31} (-) | 0.51 | 0.005 |
| L3 down c41 | MLP | a%100 in {47..56} (+) | 0.99 | 0.196 |
| L3 down c68 | MLP | a%100 in {4} (-) | 1.00 | 0.010 |
| L3 down c63 | MLP | a%100 in {5..7, 15, 35, 45} (+) | 0.98 | 0.064 |
| L3 down c29 | MLP | a%100 in {58..70, 74} (-) | 1.00 | 0.161 |
| L3 down c360 | MLP | a%100 in {6..7} (-) | 1.00 | 0.020 |
| L3 down c111 | MLP | a%100 in {60, 75} (+) | 1.00 | 0.020 |
| L3 down c225 | MLP | a%100 in {8..9} (-) | 1.00 | 0.040 |
| L3 down c136 | MLP | a%100 in {9} (-) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2s-207</b> `a%100` @ `op` (sub) — block code, 3 codes of the same shape, copy from `a`; 48 comps, L3 L10 L15; tells apart 36/100 classes (best member 8); on add: 2a-157 (member overlap 0.17)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.86 |
| classes told apart: joint / best code / best member | 36 / 33 / 8 (of 100) |
| members whose removal merges classes | 0.31 |
| support overlap (1 = tiling) / random sets / p | 3.50 / 2.78 / 0.95 |
| mean CKA between its codes | 0.77 |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.99 / 0.98 / 0.80 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.03 |
| consumers / read jointly | 15 / 14 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 MLP |
| CKA(arrangement before, joint write) | 0.53 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 2.95 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.41 2:0.14 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.39 2:0.16 3:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.40 / 4.21 |
| best source position (CKA) | `a` (0.58) |

<details><summary>codes and components</summary>

**code 2s-207.0 (L3): 22 comps, tells apart 33/100, coverage 0.72, overlap 1.82 (random 1.71, p 0.71)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 o c427 | H15 | a%100 in {0, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90} (+) | 1.00 | 0.231 |
| L3 o c101 | H15 | a%100 in {0, 60, 70, 75, 80, 82..85, 90} (-) | 1.00 | 0.110 |
| L3 o c221 | H5 | a%100 in {1..6} (-) | 1.00 | 0.090 |
| L3 o c324 | H15 | a%100 in {10..11, 31, 41, 51, 61, 81, 91} (-) | 1.00 | 0.150 |
| L3 o c368 | H15 | a%100 in {16..29} (-) | 1.00 | 0.180 |
| L3 o c393 | H15 | a%100 in {17..23} (+) | 0.99 | 0.078 |
| L3 o c453 | H15 | a%100 in {24} (+) | 1.00 | 0.010 |
| L3 o c93 | H15 | a%100 in {43, 47, 49, 51} (-) | 1.00 | 0.070 |
| L3 o c151 | H15 | a%100 in {47, 49, 51, 53} (+) | 1.00 | 0.040 |
| L3 o c384 | H15 | a%100 in {51, 53} (-) | 1.00 | 0.020 |
| L3 o c161 | H15 | a%100 in {54, 62..64, 66..70, 74} (+) | 1.00 | 0.130 |
| L3 o c327 | H15 | a%100 in {55, 65} (+) | 1.00 | 0.030 |
| L3 o c464 | H15 | a%100 in {61..69, 71..79, 81..89, 91} (+) | 1.00 | 0.430 |
| L3 o c170 | H15 | a%100 in {62, 74} (+) | 1.00 | 0.030 |
| L3 o c247 | H15 | a%100 in {65, 75} (-) | 1.00 | 0.020 |
| L3 o c441 | H15 | a%100 in {67, 70} (+) | 1.00 | 0.020 |
| L3 o c205 | H15 | a%100 in {70, 74, 77} (-) | 1.00 | 0.030 |
| L3 o c74 | H15 | a%100 in {70..74, 77..78} (-) | 1.00 | 0.150 |
| L3 o c261 | H15 | a%100 in {70} (+) | 1.00 | 0.010 |
| L3 o c431 | H15 | a%100 in {70} (+) | 1.00 | 0.010 |
| L3 o c422 | H15 | a%100 in {70} (-) | 1.00 | 0.010 |
| L3 o c432 | H15 | a%100 in {95..99} (+) | 1.00 | 0.145 |

**code 2s-207.1 (L10): 13 comps, tells apart 24/100, coverage 0.65, overlap 1.49 (random 1.33, p 0.77)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L10 down c375 | MLP | a%100 in {0} (+) | 1.00 | 0.030 |
| L10 down c277 | MLP | a%100 in {1..6, 42, 51, 70, 90, 92..96, 98} (-) | 1.00 | 1.000 |
| L10 down c196 | MLP | a%100 in {10..12} (+) | 1.00 | 0.060 |
| L10 down c171 | MLP | a%100 in {13..21} (-) | 1.00 | 0.090 |
| L10 down c391 | MLP | a%100 in {1} (+) | 1.00 | 0.040 |
| L10 down c2 | MLP | a%100 in {21, 25, 42, 53, 62, 70, 91} (-) | 1.00 | 1.000 |
| L10 down c161 | MLP | a%100 in {41..44} (+) | 1.00 | 0.040 |
| L10 down c41 | MLP | a%100 in {47, 51, 53} (+) | 0.99 | 0.075 |
| L10 down c74 | MLP | a%100 in {62, 70} (-) | 0.99 | 0.122 |
| L10 down c4 | MLP | a%100 in {63..69, 71..86} (+) | 1.00 | 0.260 |
| L10 down c5 | MLP | a%100 in {70} (-) | 1.00 | 0.010 |
| L10 down c298 | MLP | a%100 in {71..73, 75..91} (-) | 1.00 | 0.215 |
| L10 down c49 | MLP | a%100 in {93..99} (-) | 1.00 | 0.071 |

**code 2s-207.2 (L15): 13 comps, tells apart 19/100, coverage 0.53, overlap 1.38 (random 1.35, p 0.55)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c70 | MLP | a%100 in {0, 5, 10, 20, 30, 50, 70} (+) | 1.00 | 0.120 |
| L15 down c96 | MLP | a%100 in {11} (-) | 1.00 | 0.030 |
| L15 down c45 | MLP | a%100 in {12} (+) | 1.00 | 0.010 |
| L15 down c359 | MLP | a%100 in {1} (+) | 1.00 | 0.040 |
| L15 down c12 | MLP | a%100 in {27, 70} (-) | 1.00 | 0.990 |
| L15 down c79 | MLP | a%100 in {51..53, 58..67, 70, 74} (+) | 1.00 | 0.260 |
| L15 down c297 | MLP | a%100 in {63, 65..69, 71..86} (+) | 1.00 | 0.320 |
| L15 down c243 | MLP | a%100 in {70} (-) | 1.00 | 0.020 |
| L15 down c720 | MLP | a%100 in {82..89} (-) | 1.00 | 0.080 |
| L15 down c13 | MLP | a%100 in {93, 95..99} (+) | 1.00 | 0.060 |
| L15 down c268 | MLP | a%100 in {93..99} (-) | 0.99 | 0.072 |
| L15 down c134 | MLP | a%100 in {99} (+) | 0.95 | 0.011 |
| L15 down c310 | MLP | a%100 in {9} (+) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2s-210</b> `a%100` @ `op` (sub) — block code; 2 comps, L4; tells apart 7/100 classes (best member 3); on add: 2a-162 (member overlap 0.50)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.24 |
| classes told apart: joint / best code / best member | 7 /  / 3 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.82 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.97 /  / 0.61 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L3 MLP |
| CKA(arrangement before, joint write) | 0.27 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.48 2:0.14 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.38 3:0.09 6:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.13 / 1.64 |
| best source position (CKA) | `a` (0.28) |

<details><summary>codes and components</summary>

**code 2s-210.0 (L4): 2 comps, tells apart 7/100, coverage 0.24, overlap 1.00 (random 1.00, p 0.90)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 o c239 | H26 | a%100 in {0, 70} (+) | 1.00 | 0.980 |
| L4 o c7 | H18 | a%100 in {1..2, 10, 12, 20, 36, 46, 51..54, 57, 59..60, 71, 91, 93..95, 97..99} (+) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2s-211</b> `a%100` @ `op` (sub) — block code, 2 codes of the same shape; 33 comps, L4 L8; tells apart 23/100 classes (best member 7); on add: 2a-163 (member overlap 0.09)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.93 |
| classes told apart: joint / best code / best member | 23 / 22 / 7 (of 100) |
| members whose removal merges classes | 0.27 |
| support overlap (1 = tiling) / random sets / p | 2.62 / 2.14 / 0.94 |
| mean CKA between its codes | 0.84 |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 1.00 / 0.98 / 0.70 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.03 |
| consumers / read jointly | 41 / 27 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L4 attn |
| CKA(arrangement before, joint write) | 0.73 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.03 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.60 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.46 2:0.14 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.51 2:0.16 3:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.54 / 3.72 |

<details><summary>codes and components</summary>

**code 2s-211.0 (L4): 17 comps, tells apart 22/100, coverage 0.62, overlap 1.76 (random 1.50, p 0.92)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 down c6 | MLP | a%100 in {0..1, 70..71, 73..75, 77..79, 81..87, 89, 91..99} (-) | 1.00 | 1.000 |
| L4 down c103 | MLP | a%100 in {10..17} (-) | 1.00 | 0.140 |
| L4 down c237 | MLP | a%100 in {17..21} (-) | 1.00 | 0.081 |
| L4 down c222 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L4 down c487 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L4 down c135 | MLP | a%100 in {2, 4, 6, 8} (+) | 1.00 | 0.040 |
| L4 down c263 | MLP | a%100 in {3..5} (+) | 1.00 | 0.030 |
| L4 down c472 | MLP | a%100 in {4..9} (-) | 1.00 | 0.060 |
| L4 down c148 | MLP | a%100 in {41, 43..44, 46..47, 49, 51..53} (-) | 1.00 | 0.284 |
| L4 down c379 | MLP | a%100 in {41..42, 44} (+) | 1.00 | 0.060 |
| L4 down c910 | MLP | a%100 in {42} (+) | 1.00 | 0.010 |
| L4 down c983 | MLP | a%100 in {42} (+) | 1.00 | 0.010 |
| L4 down c66 | MLP | a%100 in {6..9} (-) | 1.00 | 0.040 |
| L4 down c52 | MLP | a%100 in {70} (+) | 1.00 | 0.060 |
| L4 down c312 | MLP | a%100 in {71..94} (+) | 1.00 | 0.360 |
| L4 down c740 | MLP | a%100 in {9..12} (+) | 1.00 | 0.080 |
| L4 down c443 | MLP | a%100 in {93..99} (-) | 1.00 | 0.080 |

**code 2s-211.1 (L8): 16 comps, tells apart 20/100, coverage 0.91, overlap 1.48 (random 1.46, p 0.58)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L8 down c13 | MLP | a%100 in {0..9, 54, 58..59, 70, 97..99} (-) | 1.00 | 1.000 |
| L8 down c278 | MLP | a%100 in {1..2} (-) | 1.00 | 0.020 |
| L8 down c257 | MLP | a%100 in {10..15} (+) | 1.00 | 0.110 |
| L8 down c140 | MLP | a%100 in {13..19} (-) | 1.00 | 0.070 |
| L8 down c855 | MLP | a%100 in {1} (+) | 1.00 | 0.020 |
| L8 down c581 | MLP | a%100 in {23..38} (-) | 1.00 | 0.170 |
| L8 down c64 | MLP | a%100 in {28..29, 31..41, 43..49, 51..53} (+) | 1.00 | 0.280 |
| L8 down c297 | MLP | a%100 in {42} (+) | 1.00 | 0.010 |
| L8 down c267 | MLP | a%100 in {47, 51, 53} (+) | 1.00 | 0.062 |
| L8 down c276 | MLP | a%100 in {61..63, 65..69, 73..76, 79} (-) | 0.99 | 0.133 |
| L8 down c680 | MLP | a%100 in {66..69, 71..97} (-) | 1.00 | 0.364 |
| L8 down c134 | MLP | a%100 in {70} (+) | 1.00 | 0.010 |
| L8 down c223 | MLP | a%100 in {70} (+) | 1.00 | 0.010 |
| L8 down c32 | MLP | a%100 in {70} (-) | 1.00 | 0.010 |
| L8 down c580 | MLP | a%100 in {93..99} (-) | 1.00 | 0.070 |
| L8 down c378 | MLP | a%100 in {95..99} (-) | 1.00 | 0.070 |

</details>

</details>

<details><summary><b>2s-212</b> `a%100` @ `op` (sub) — block code; 2 comps, L5; tells apart 6/100 classes (best member 4); on add: 2a-164 (member overlap 0.67)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.28 |
| classes told apart: joint / best code / best member | 6 /  / 4 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.11 / 1.00 / 0.90 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.99 /  / 0.52 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.02 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L4 MLP |
| CKA(arrangement before, joint write) | 0.33 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.45 2:0.14 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.42 2:0.14 3:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.14 / 1.35 |
| best source position (CKA) | `a` (0.32) |

<details><summary>codes and components</summary>

**code 2s-212.0 (L5): 2 comps, tells apart 6/100, coverage 0.28, overlap 1.11 (random 1.00, p 0.94)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 o c6 | H26 | a%100 in {0..2, 10, 18, 24, 36, 70, 89..91, 94, 98..99} (+) | 1.00 | 1.000 |
| L5 o c2 | H26 | a%100 in {13, 16, 30, 42, 48..50, 70, 79, 83..84, 92..96, 99} (+) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2s-213</b> `a%100` @ `op` (sub) — block code; 10 comps, L5; tells apart 26/100 classes (best member 7); on add: 2a-165 (member overlap 0.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.48 |
| classes told apart: joint / best code / best member | 26 /  / 7 (of 100) |
| members whose removal merges classes | 0.80 |
| support overlap (1 = tiling) / random sets / p | 1.15 / 1.24 / 0.25 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.95 /  / 0.49 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.99 |
| consumers / read jointly | 15 / 3 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L5 attn |
| CKA(arrangement before, joint write) | 0.49 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.03 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.09 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.45 2:0.14 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.22 2:0.18 3:0.12) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.29 / 5.49 |

<details><summary>codes and components</summary>

**code 2s-213.0 (L5): 10 comps, tells apart 26/100, coverage 0.48, overlap 1.15 (random 1.24, p 0.22)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 down c86 | MLP | a%100 in {0, 50, 60, 80} (-) | 1.00 | 0.092 |
| L5 down c1 | MLP | a%100 in {1, 18, 24, 26, 36, 50, 70, 99} (-) | 1.00 | 1.000 |
| L5 down c43 | MLP | a%100 in {1..2} (-) | 1.00 | 0.038 |
| L5 down c383 | MLP | a%100 in {10..20} (+) | 1.00 | 0.170 |
| L5 down c987 | MLP | a%100 in {2..9} (-) | 1.00 | 0.080 |
| L5 down c424 | MLP | a%100 in {41..42} (+) | 1.00 | 0.068 |
| L5 down c458 | MLP | a%100 in {42} (-) | 1.00 | 0.010 |
| L5 down c115 | MLP | a%100 in {47, 49, 51..53} (-) | 1.00 | 0.232 |
| L5 down c137 | MLP | a%100 in {57..69} (+) | 1.00 | 0.250 |
| L5 down c375 | MLP | a%100 in {70} (+) | 1.00 | 0.060 |

</details>

</details>

<details><summary><b>2s-214</b> `a%100` @ `op` (sub) — block code; 3 comps, L6 L7; tells apart 12/100 classes (best member 6); on add: 2a-168 (member overlap 0.33)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.13 |
| classes told apart: joint / best code / best member | 12 /  / 6 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.64 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.99 |
| decoding acc. joint / best code / best member (chance) | 0.82 /  / 0.25 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.96 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L5 MLP |
| CKA(arrangement before, joint write) | 0.27 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.45 2:0.14 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.14 3:0.07 4:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.06 / 2.40 |
| best source position (CKA) | `a` (0.26) |

<details><summary>codes and components</summary>

**code 2s-214.0 (L6 L7): 3 comps, tells apart 12/100, coverage 0.13, overlap 1.00 (random 1.00, p 0.58)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L6 o c4 | H22 | a%100 in {70} (+) | 1.00 | 1.000 |
| L7 o c3 | H5 | a%100 in {0..1, 10..12, 42, 50, 60} (-) | 1.00 | 1.000 |
| L7 o c33 | H28 | a%100 in {95, 97..99} (+) | 0.99 | 0.041 |

</details>

</details>

<details><summary><b>2s-215</b> `a%100` @ `op` (sub) — block code; 10 comps, L6; tells apart 26/100 classes (best member 8); on add: 2a-165 (member overlap 0.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.61 |
| classes told apart: joint / best code / best member | 26 /  / 8 (of 100) |
| members whose removal merges classes | 0.90 |
| support overlap (1 = tiling) / random sets / p | 1.61 / 1.22 / 0.95 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.88 /  / 0.58 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.98 |
| consumers / read jointly | 9 / 6 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L6 attn |
| CKA(arrangement before, joint write) | 0.68 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.03 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.11 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.45 2:0.14 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.38 2:0.17 3:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.35 / 3.70 |

<details><summary>codes and components</summary>

**code 2s-215.0 (L6): 10 comps, tells apart 26/100, coverage 0.61, overlap 1.61 (random 1.25, p 0.96)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L6 down c185 | MLP | a%100 in {0, 40, 50, 60, 75, 80, 90} (+) | 1.00 | 0.090 |
| L6 down c146 | MLP | a%100 in {1..12} (-) | 1.00 | 0.160 |
| L6 down c77 | MLP | a%100 in {1..4} (+) | 1.00 | 0.070 |
| L6 down c74 | MLP | a%100 in {17..38} (+) | 1.00 | 0.360 |
| L6 down c187 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L6 down c631 | MLP | a%100 in {21..29, 31} (+) | 1.00 | 0.125 |
| L6 down c0 | MLP | a%100 in {3..9, 13..14, 16..18, 20..26, 28, 30, 35, 51, 70, 74..75, 81, 91, 94..99} (+) | 1.00 | 1.000 |
| L6 down c104 | MLP | a%100 in {41..44} (+) | 1.00 | 0.161 |
| L6 down c399 | MLP | a%100 in {49, 51, 53} (+) | 1.00 | 0.104 |
| L6 down c892 | MLP | a%100 in {70} (+) | 1.00 | 0.050 |

</details>

</details>

<details><summary><b>2s-216</b> `a%100` @ `op` (sub) — block code; 3 comps, L7 L8; tells apart 9/100 classes (best member 5); on add: 2a-167 (member overlap 0.33)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.22 |
| classes told apart: joint / best code / best member | 9 /  / 5 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.65 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.98 |
| decoding acc. joint / best code / best member (chance) | 0.55 /  / 0.36 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L6 MLP |
| CKA(arrangement before, joint write) | 0.48 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.44 2:0.13 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.32 2:0.17 3:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.14 / 1.86 |
| best source position (CKA) | `a` (0.42) |

<details><summary>codes and components</summary>

**code 2s-216.0 (L7 L8): 3 comps, tells apart 9/100, coverage 0.22, overlap 1.00 (random 1.00, p 0.59)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L7 o c14 | H20 | a%100 in {0, 11..12, 42, 70} (+) | 1.00 | 1.000 |
| L7 o c276 | H7 | a%100 in {91..99} (-) | 0.99 | 0.224 |
| L8 o c19 | H22 | a%100 in {65, 67, 69, 73..76, 79} (+) | 0.95 | 0.078 |

</details>

</details>

<details><summary><b>2s-217</b> `a%100` @ `op` (sub) — block code, 6 codes of the same shape; 120 comps, L7 L9 L11 L13 L18 L20; tells apart 80/100 classes (best member 9); on add: 2a-165 (member overlap 0.01)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 80 / 85 / 9 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p | 8.43 / 6.46 / 1.00 |
| mean CKA between its codes | 0.77 |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 1.00 / 1.00 / 0.79 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.97 |
| consumers / read jointly | 180 / 168 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L7 attn |
| CKA(arrangement before, joint write) | 0.75 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.05 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 1.54 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.44 2:0.13 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.36 2:0.16 3:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.65 / 7.65 |

<details><summary>codes and components</summary>

**code 2s-217.0 (L7): 11 comps, tells apart 17/100, coverage 0.94, overlap 1.66 (random 1.27, p 0.97)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L7 down c2 | MLP | a%100 in {0, 5, 12, 20..22, 24..26, 28, 30, 42..46, 50, 55, 73..75, 77, 79, 81} (-) | 1.00 | 1.000 |
| L7 down c7 | MLP | a%100 in {0..10, 12, 31..34, 36..41, 43..44, 46..49, 51..59, 61, 63..64, 70, 93..99} (-) | 1.00 | 1.000 |
| L7 down c383 | MLP | a%100 in {1..3} (-) | 1.00 | 0.070 |
| L7 down c832 | MLP | a%100 in {14..20} (+) | 0.98 | 0.086 |
| L7 down c259 | MLP | a%100 in {41..44, 47, 52..53} (-) | 1.00 | 0.162 |
| L7 down c394 | MLP | a%100 in {51..53, 58..59, 61..62} (+) | 0.99 | 0.072 |
| L7 down c22 | MLP | a%100 in {57..69, 71..84} (+) | 1.00 | 0.360 |
| L7 down c27 | MLP | a%100 in {70} (+) | 1.00 | 0.010 |
| L7 down c44 | MLP | a%100 in {70} (-) | 1.00 | 0.020 |
| L7 down c89 | MLP | a%100 in {71, 73..94} (+) | 1.00 | 0.280 |
| L7 down c567 | MLP | a%100 in {92..99} (+) | 1.00 | 0.090 |

**code 2s-217.1 (L9): 17 comps, tells apart 27/100, coverage 0.83, overlap 1.35 (random 1.49, p 0.17)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L9 down c19 | MLP | a%100 in {0, 50} (+) | 0.98 | 0.042 |
| L9 down c21 | MLP | a%100 in {0..12, 39, 70, 95, 97..99} (-) | 1.00 | 1.000 |
| L9 down c847 | MLP | a%100 in {1..3} (-) | 1.00 | 0.060 |
| L9 down c929 | MLP | a%100 in {10..13} (+) | 1.00 | 0.090 |
| L9 down c75 | MLP | a%100 in {13..23} (-) | 1.00 | 0.110 |
| L9 down c985 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L9 down c369 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L9 down c89 | MLP | a%100 in {23..31, 35} (-) | 1.00 | 0.100 |
| L9 down c14 | MLP | a%100 in {3..6} (-) | 1.00 | 0.040 |
| L9 down c195 | MLP | a%100 in {42} (-) | 1.00 | 0.010 |
| L9 down c398 | MLP | a%100 in {49, 51, 53} (+) | 1.00 | 0.051 |
| L9 down c137 | MLP | a%100 in {55, 57..69, 73..74} (-) | 1.00 | 0.300 |
| L9 down c32 | MLP | a%100 in {65..69, 71..91} (-) | 1.00 | 0.352 |
| L9 down c242 | MLP | a%100 in {70} (+) | 1.00 | 0.010 |
| L9 down c30 | MLP | a%100 in {70} (-) | 1.00 | 0.010 |
| L9 down c79 | MLP | a%100 in {92..99} (-) | 1.00 | 0.080 |
| L9 down c622 | MLP | a%100 in {99} (-) | 1.00 | 0.010 |

**code 2s-217.2 (L11): 11 comps, tells apart 16/100, coverage 0.74, overlap 1.99 (random 1.30, p 1.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L11 down c0 | MLP | a%100 in {0, 3, 9..10, 30, 42, 62..63, 65..67, 70, 73..75, 90, 95..96, 98} (-) | 1.00 | 1.000 |
| L11 down c42 | MLP | a%100 in {0, 60, 70, 80, 90} (+) | 1.00 | 0.050 |
| L11 down c44 | MLP | a%100 in {0..9, 37, 70, 95..99} (-) | 1.00 | 1.000 |
| L11 down c921 | MLP | a%100 in {0} (+) | 1.00 | 0.041 |
| L11 down c73 | MLP | a%100 in {1, 4..16} (-) | 1.00 | 0.180 |
| L11 down c19 | MLP | a%100 in {1..10, 14..18, 21, 24..28, 30..32, 41..42, 55..59, 63, 65..69, 71..72, 76..79, 86..89, 91..94} (-) | 0.99 | 0.741 |
| L11 down c1 | MLP | a%100 in {1..7, 9, 15..16, 27} (-) | 1.00 | 0.512 |
| L11 down c204 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L11 down c92 | MLP | a%100 in {2..12} (+) | 1.00 | 0.110 |
| L11 down c618 | MLP | a%100 in {70} (-) | 1.00 | 0.010 |
| L11 down c69 | MLP | a%100 in {76..91} (+) | 1.00 | 0.220 |

**code 2s-217.3 (L13): 13 comps, tells apart 21/100, coverage 0.74, overlap 1.32 (random 1.36, p 0.41)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 down c0 | MLP | a%100 in {0, 14..18, 20..22, 24, 27, 92..93, 95..99} (-) | 1.00 | 1.000 |
| L13 down c1 | MLP | a%100 in {1..9, 35..36, 38, 70, 94..95} (-) | 1.00 | 1.000 |
| L13 down c105 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L13 down c169 | MLP | a%100 in {1} (-) | 1.00 | 0.020 |
| L13 down c190 | MLP | a%100 in {2..4} (+) | 1.00 | 0.030 |
| L13 down c277 | MLP | a%100 in {49, 51, 53, 61..62, 70} (-) | 1.00 | 0.090 |
| L13 down c13 | MLP | a%100 in {5, 45} (+) | 1.00 | 0.020 |
| L13 down c800 | MLP | a%100 in {57..69, 71..76} (-) | 1.00 | 0.250 |
| L13 down c24 | MLP | a%100 in {6..13} (-) | 1.00 | 0.101 |
| L13 down c734 | MLP | a%100 in {69, 71..91} (-) | 1.00 | 0.279 |
| L13 down c6 | MLP | a%100 in {70} (+) | 1.00 | 0.010 |
| L13 down c9 | MLP | a%100 in {7} (-) | 1.00 | 0.010 |
| L13 down c11 | MLP | a%100 in {8} (+) | 1.00 | 0.010 |

**code 2s-217.4 (L18): 39 comps, tells apart 85/100, coverage 0.94, overlap 2.33 (random 2.30, p 0.54)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 down c4 | MLP | a%100 in {0, 16, 22, 24, 26, 28, 30, 32, 36, 40, 42, 46, 48, 52, 54, 56, 60, 64, 66, 68, 72, 76, 78, 80, 82, 84, 86, 88} (-) | 1.00 | 0.450 |
| L18 down c256 | MLP | a%100 in {0, 30, 40, 50, 60, 80, 90} (-) | 1.00 | 0.080 |
| L18 down c51 | MLP | a%100 in {0} (+) | 1.00 | 0.050 |
| L18 down c373 | MLP | a%100 in {1..3} (+) | 1.00 | 0.040 |
| L18 down c15 | MLP | a%100 in {1..4, 6, 9, 46, 48, 70, 99} (+) | 1.00 | 1.000 |
| L18 down c440 | MLP | a%100 in {10..12} (+) | 1.00 | 0.080 |
| L18 down c88 | MLP | a%100 in {13..14, 23..24, 33, 72..73, 82..84} (+) | 1.00 | 0.110 |
| L18 down c181 | MLP | a%100 in {13..23} (-) | 1.00 | 0.150 |
| L18 down c24 | MLP | a%100 in {15, 19..20} (+) | 1.00 | 0.030 |
| L18 down c26 | MLP | a%100 in {16..17, 21} (-) | 1.00 | 0.030 |
| L18 down c142 | MLP | a%100 in {17..18, 22, 27} (+) | 1.00 | 0.060 |
| L18 down c8 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L18 down c169 | MLP | a%100 in {1} (-) | 1.00 | 0.020 |
| L18 down c474 | MLP | a%100 in {21..22, 31, 41, 71, 81} (-) | 1.00 | 0.070 |
| L18 down c702 | MLP | a%100 in {22, 32, 42, 52, 72, 82} (-) | 1.00 | 0.140 |
| L18 down c381 | MLP | a%100 in {31..33, 51..54, 71..73} (-) | 1.00 | 0.220 |
| L18 down c59 | MLP | a%100 in {35..38, 56..57, 76, 78} (+) | 1.00 | 0.080 |
| L18 down c18 | MLP | a%100 in {38..40, 43..44, 58..63, 79} (-) | 1.00 | 0.141 |
| L18 down c479 | MLP | a%100 in {39..44, 47, 49} (-) | 1.00 | 0.160 |
| L18 down c937 | MLP | a%100 in {40, 60, 75, 80} (-) | 1.00 | 0.040 |
| L18 down c70 | MLP | a%100 in {40..45} (-) | 1.00 | 0.090 |
| L18 down c303 | MLP | a%100 in {49, 51, 70} (-) | 1.00 | 0.110 |
| L18 down c839 | MLP | a%100 in {4} (-) | 1.00 | 0.010 |
| L18 down c482 | MLP | a%100 in {51..53, 55} (+) | 1.00 | 0.200 |
| L18 down c333 | MLP | a%100 in {59..65, 67} (+) | 1.00 | 0.110 |
| L18 down c275 | MLP | a%100 in {5} (+) | 1.00 | 0.010 |
| L18 down c41 | MLP | a%100 in {6..9} (-) | 1.00 | 0.050 |
| L18 down c11 | MLP | a%100 in {65, 67..81} (+) | 1.00 | 0.310 |
| L18 down c348 | MLP | a%100 in {70} (+) | 1.00 | 0.010 |
| L18 down c935 | MLP | a%100 in {70} (+) | 1.00 | 0.010 |
| L18 down c345 | MLP | a%100 in {70} (-) | 1.00 | 0.010 |
| L18 down c188 | MLP | a%100 in {74} (+) | 1.00 | 0.010 |
| L18 down c361 | MLP | a%100 in {75} (-) | 1.00 | 0.010 |
| L18 down c305 | MLP | a%100 in {78..79, 87..89} (+) | 1.00 | 0.060 |
| L18 down c143 | MLP | a%100 in {80..87} (+) | 1.00 | 0.260 |
| L18 down c98 | MLP | a%100 in {81} (+) | 1.00 | 0.010 |
| L18 down c407 | MLP | a%100 in {83..89} (-) | 1.00 | 0.070 |
| L18 down c424 | MLP | a%100 in {86..91} (-) | 1.00 | 0.060 |
| L18 down c123 | MLP | a%100 in {95..99} (+) | 1.00 | 0.050 |

**code 2s-217.5 (L20): 29 comps, tells apart 48/100, coverage 0.65, overlap 1.71 (random 1.93, p 0.15)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 down c145 | MLP | a%100 in {0..2, 70} (-) | 0.99 | 0.982 |
| L20 down c140 | MLP | a%100 in {1, 21, 31, 41, 51, 61, 71, 81} (-) | 1.00 | 0.110 |
| L20 down c584 | MLP | a%100 in {10..14} (+) | 1.00 | 0.100 |
| L20 down c432 | MLP | a%100 in {10} (-) | 1.00 | 0.010 |
| L20 down c158 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L20 down c839 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L20 down c41 | MLP | a%100 in {2, 22, 42, 52, 62} (-) | 0.99 | 0.061 |
| L20 down c35 | MLP | a%100 in {24} (+) | 0.98 | 0.011 |
| L20 down c81 | MLP | a%100 in {24} (+) | 1.00 | 0.010 |
| L20 down c97 | MLP | a%100 in {3, 7..8} (+) | 1.00 | 0.030 |
| L20 down c536 | MLP | a%100 in {3..5} (-) | 1.00 | 0.030 |
| L20 down c399 | MLP | a%100 in {30..31} (+) | 1.00 | 0.040 |
| L20 down c201 | MLP | a%100 in {31..32, 81..84, 87..88, 91..94} (-) | 1.00 | 0.161 |
| L20 down c22 | MLP | a%100 in {31} (-) | 1.00 | 0.060 |
| L20 down c107 | MLP | a%100 in {37, 39..47} (+) | 1.00 | 0.180 |
| L20 down c429 | MLP | a%100 in {4..9} (+) | 1.00 | 0.070 |
| L20 down c923 | MLP | a%100 in {51..53} (+) | 1.00 | 0.100 |
| L20 down c820 | MLP | a%100 in {51} (-) | 1.00 | 0.010 |
| L20 down c488 | MLP | a%100 in {59..63, 65} (-) | 1.00 | 0.070 |
| L20 down c131 | MLP | a%100 in {5} (+) | 1.00 | 0.010 |
| L20 down c734 | MLP | a%100 in {61..63} (-) | 1.00 | 0.030 |
| L20 down c94 | MLP | a%100 in {69, 71..89} (+) | 1.00 | 0.310 |
| L20 down c320 | MLP | a%100 in {6} (-) | 1.00 | 0.010 |
| L20 down c169 | MLP | a%100 in {70, 73..75} (+) | 1.00 | 0.133 |
| L20 down c738 | MLP | a%100 in {70} (+) | 1.00 | 0.010 |
| L20 down c311 | MLP | a%100 in {70} (-) | 1.00 | 0.010 |
| L20 down c50 | MLP | a%100 in {8..10} (+) | 1.00 | 0.030 |
| L20 down c33 | MLP | a%100 in {88..89} (-) | 1.00 | 0.050 |
| L20 down c870 | MLP | a%100 in {9} (+) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2s-218</b> `a%100` @ `op` (sub) — block code; 2 comps, L8; tells apart 8/100 classes (best member 8); on add: 2a-169 (member overlap 0.50)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.16 |
| classes told apart: joint / best code / best member | 8 /  / 8 (of 100) |
| members whose removal merges classes | 0.50 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.84 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.38 /  / 0.29 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L7 MLP |
| CKA(arrangement before, joint write) | 0.33 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.07 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.43 2:0.13 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.19 2:0.16 3:0.13) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.07 / 1.27 |
| best source position (CKA) | `a` (0.27) |

<details><summary>codes and components</summary>

**code 2s-218.0 (L8): 2 comps, tells apart 8/100, coverage 0.16, overlap 1.00 (random 1.00, p 0.80)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L8 o c7 | H23 | a%100 in {1, 11..13, 18..19, 42, 70} (+) | 1.00 | 1.000 |
| L8 o c113 | H19 | a%100 in {92..99} (+) | 1.00 | 0.090 |

</details>

</details>

<details><summary><b>2s-219</b> `a%100` @ `op` (sub) — block code, copy from `a`; 2 comps, L9; tells apart 13/100 classes (best member 4); on add: 2a-170 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.24 |
| classes told apart: joint / best code / best member | 13 /  / 4 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.08 / 1.00 / 0.92 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.84 /  / 0.34 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.99 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L8 MLP |
| CKA(arrangement before, joint write) | 0.50 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.42 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.25 4:0.12 2:0.12) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.11 / 1.90 |
| best source position (CKA) | `a` (0.54) |

<details><summary>codes and components</summary>

**code 2s-219.0 (L9): 2 comps, tells apart 13/100, coverage 0.24, overlap 1.08 (random 1.00, p 0.89)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L9 o c183 | H11 | a%100 in {0..4, 12, 26, 33..35, 37, 42, 46, 92..93} (+) | 1.00 | 1.000 |
| L9 o c9 | H11 | a%100 in {4..5, 7, 70, 93..99} (+) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2s-220</b> `a%100` @ `op` (sub) — single component; 1 comps, L10; tells apart 3/100 classes (best member 3); on add: 2a-171 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.02 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.22 /  / 0.22 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L9 MLP |
| CKA(arrangement before, joint write) | 0.19 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.41 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (4:0.15 10:0.08 5:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.03 / 1.00 |
| best source position (CKA) | `a` (0.09) |

<details><summary>codes and components</summary>

**code 2s-220.0 (L10): 1 comps, tells apart 3/100, coverage 0.02, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L10 o c1 | H30 | a%100 in {70, 99} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2s-221</b> `a%100` @ `op` (sub) — single component; 1 comps, L11; tells apart 5/100 classes (best member 5); on add: 2a-172 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.05 |
| classes told apart: joint / best code / best member | 5 /  / 5 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.22 /  / 0.22 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L10 MLP |
| CKA(arrangement before, joint write) | 0.29 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.40 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.35 4:0.13 10:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.07 / 1.00 |
| best source position (CKA) | `a` (0.22) |

<details><summary>codes and components</summary>

**code 2s-221.0 (L11): 1 comps, tells apart 5/100, coverage 0.05, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L11 o c4 | H20 | a%100 in {0, 42, 70, 74, 93} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2s-222</b> `a%100` @ `op` (sub) — block code; 3 comps, L12; tells apart 5/100 classes (best member 4); on add: 2a-173 (member overlap 0.25)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.03 |
| classes told apart: joint / best code / best member | 5 /  / 4 (of 100) |
| members whose removal merges classes | 0.67 |
| support overlap (1 = tiling) / random sets / p | 1.33 / 1.00 / 0.95 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.99 |
| decoding acc. joint / best code / best member (chance) | 0.97 /  / 0.62 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.94 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L11 MLP |
| CKA(arrangement before, joint write) | 0.16 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.03 |
| share of the write inside the old arrangement's span | 0.12 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.38 2:0.13 3:0.07) |
| joint write: shape (spectrum k:share) | line (6:0.08 4:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.02 / 1.47 |
| best source position (CKA) | `a` (0.10) |

<details><summary>codes and components</summary>

**code 2s-222.0 (L12): 3 comps, tells apart 5/100, coverage 0.03, overlap 1.33 (random 1.00, p 0.96)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L12 o c459 | H20 | a%100 in {0, 70} (+) | 1.00 | 0.980 |
| L12 o c41 | H20 | a%100 in {24} (-) | 0.91 | 0.011 |
| L12 o c209 | H20 | a%100 in {70} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2s-223</b> `a%100` @ `op` (sub) — block code; 14 comps, L12; tells apart 16/100 classes (best member 6); on add: 2a-165 (member overlap 0.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.61 |
| classes told apart: joint / best code / best member | 16 /  / 6 (of 100) |
| members whose removal merges classes | 0.64 |
| support overlap (1 = tiling) / random sets / p | 1.25 / 1.39 / 0.19 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.86 /  / 0.40 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.98 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L12 attn |
| CKA(arrangement before, joint write) | 0.54 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.07 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.06 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.38 2:0.13 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.33 2:0.19 3:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.28 / 3.52 |

<details><summary>codes and components</summary>

**code 2s-223.0 (L12): 14 comps, tells apart 16/100, coverage 0.61, overlap 1.25 (random 1.40, p 0.18)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L12 down c15 | MLP | a%100 in {1, 11} (+) | 1.00 | 0.020 |
| L12 down c0 | MLP | a%100 in {1, 70} (+) | 1.00 | 1.000 |
| L12 down c544 | MLP | a%100 in {1..7} (-) | 1.00 | 0.080 |
| L12 down c9 | MLP | a%100 in {11} (-) | 0.99 | 0.010 |
| L12 down c169 | MLP | a%100 in {1} (-) | 1.00 | 0.020 |
| L12 down c32 | MLP | a%100 in {2..4} (+) | 1.00 | 0.030 |
| L12 down c1004 | MLP | a%100 in {2..4} (+) | 1.00 | 0.030 |
| L12 down c144 | MLP | a%100 in {26..29, 31..41, 43..59} (-) | 1.00 | 0.353 |
| L12 down c661 | MLP | a%100 in {51, 53} (-) | 1.00 | 0.020 |
| L12 down c228 | MLP | a%100 in {7..9} (-) | 0.98 | 0.030 |
| L12 down c74 | MLP | a%100 in {70} (+) | 1.00 | 0.010 |
| L12 down c506 | MLP | a%100 in {78, 81..89} (-) | 1.00 | 0.100 |
| L12 down c183 | MLP | a%100 in {92..99} (-) | 1.00 | 0.090 |
| L12 down c7 | MLP | a%100 in {9} (-) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2s-224</b> `a%100` @ `op` (sub) — block code; 2 comps, L13 L14; tells apart 12/100 classes (best member 4); on add: 2a-174 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.06 |
| classes told apart: joint / best code / best member | 12 /  / 4 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.84 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.84 /  / 0.29 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.06 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L12 MLP |
| CKA(arrangement before, joint write) | 0.30 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.37 2:0.13 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.22 4:0.10 10:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.06 / 1.59 |
| best source position (CKA) | `a` (0.29) |

<details><summary>codes and components</summary>

**code 2s-224.0 (L13 L14): 2 comps, tells apart 12/100, coverage 0.06, overlap 1.00 (random 1.00, p 0.86)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 o c0 | H16 | a%100 in {0, 70} (+) | 1.00 | 1.000 |
| L14 o c59 | H26 | a%100 in {1, 5, 15, 99} (+) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2s-226</b> `a%100` @ `op` (sub) — block code; 2 comps, L14 L15; tells apart 6/100 classes (best member 4); on add: 2a-176 (member overlap 0.50)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.05 |
| classes told apart: joint / best code / best member | 6 /  / 4 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.87 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.99 /  / 0.68 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L13 MLP |
| CKA(arrangement before, joint write) | 0.27 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.37 2:0.13 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.34 2:0.14 5:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.08 / 1.09 |
| best source position (CKA) | `a` (0.21) |

<details><summary>codes and components</summary>

**code 2s-226.0 (L14 L15): 2 comps, tells apart 6/100, coverage 0.05, overlap 1.00 (random 1.00, p 0.81)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L14 o c495 | H27 | a%100 in {1..2, 29..30} (+) | 1.00 | 0.980 |
| L15 o c3 | H5 | a%100 in {70} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2s-225</b> `a%100` @ `op` (sub) — block code; 5 comps, L14; tells apart 15/100 classes (best member 9); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.20 |
| classes told apart: joint / best code / best member | 15 /  / 9 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.10 / 1.08 / 0.56 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.40 /  / 0.19 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L13 MLP |
| CKA(arrangement before, joint write) | 0.26 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.06 |
| share of the write inside the old arrangement's span | 0.11 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.37 2:0.13 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (20:0.11 40:0.11 10:0.11) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.11 / 2.94 |
| best source position (CKA) | `a` (0.34) |

<details><summary>codes and components</summary>

**code 2s-225.0 (L14): 5 comps, tells apart 15/100, coverage 0.20, overlap 1.10 (random 1.07, p 0.58)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L14 o c248 | H18 | a%100 in {0, 10..12, 30, 70} (+) | 1.00 | 0.190 |
| L14 o c404 | H19 | a%100 in {0, 94..99} (-) | 1.00 | 0.081 |
| L14 o c129 | H18 | a%100 in {11, 31, 51, 81, 91} (-) | 1.00 | 0.120 |
| L14 o c392 | H18 | a%100 in {35, 45} (-) | 1.00 | 0.020 |
| L14 o c92 | H18 | a%100 in {80, 90} (-) | 1.00 | 0.020 |

</details>

</details>

<details><summary><b>2s-227</b> `a%100` @ `op` (sub) — tiling, 6 codes of the same shape, copy from `a`; 179 comps, L14 L16 L17 L19 L20 L22 L24; tells apart 82/100 classes (best member 10); on add: 2a-165 (member overlap 0.02)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 82 / 69 / 10 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p | 8.65 / 9.51 / 0.14 |
| mean CKA between its codes | 0.75 |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.99 / 1.00 / 0.81 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 343 / 306 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L14 attn |
| CKA(arrangement before, joint write) | 0.83 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.05 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 1.91 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.36 2:0.13 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.33 2:0.14 3:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.65 / 9.16 |
| best source position (CKA) | `a` (0.69) |

<details><summary>codes and components</summary>

**code 2s-227.0 (L14): 24 comps, tells apart 28/100, coverage 0.66, overlap 1.80 (random 1.73, p 0.60)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L14 down c400 | MLP | a%100 in {0, 10} (-) | 1.00 | 0.020 |
| L14 down c18 | MLP | a%100 in {0, 80} (+) | 1.00 | 0.100 |
| L14 down c4 | MLP | a%100 in {1..4, 6, 98..99} (+) | 1.00 | 1.000 |
| L14 down c170 | MLP | a%100 in {1..4} (-) | 1.00 | 0.090 |
| L14 down c31 | MLP | a%100 in {1..6, 70, 95, 97, 99} (-) | 1.00 | 1.000 |
| L14 down c526 | MLP | a%100 in {10..22} (+) | 1.00 | 0.140 |
| L14 down c108 | MLP | a%100 in {11..12} (-) | 1.00 | 0.020 |
| L14 down c542 | MLP | a%100 in {11..12} (-) | 1.00 | 0.020 |
| L14 down c71 | MLP | a%100 in {11} (-) | 1.00 | 0.010 |
| L14 down c53 | MLP | a%100 in {14..29} (-) | 1.00 | 0.180 |
| L14 down c150 | MLP | a%100 in {1} (+) | 1.00 | 0.021 |
| L14 down c15 | MLP | a%100 in {2} (-) | 1.00 | 0.010 |
| L14 down c14 | MLP | a%100 in {4} (-) | 1.00 | 0.010 |
| L14 down c110 | MLP | a%100 in {51, 53, 70} (+) | 1.00 | 0.130 |
| L14 down c343 | MLP | a%100 in {61..69, 71..86} (+) | 1.00 | 0.341 |
| L14 down c33 | MLP | a%100 in {6} (-) | 1.00 | 0.010 |
| L14 down c201 | MLP | a%100 in {7..8} (+) | 1.00 | 0.020 |
| L14 down c141 | MLP | a%100 in {70} (-) | 1.00 | 0.030 |
| L14 down c156 | MLP | a%100 in {81..84, 86} (-) | 1.00 | 0.050 |
| L14 down c67 | MLP | a%100 in {92..99} (+) | 1.00 | 0.090 |
| L14 down c866 | MLP | a%100 in {93..99} (+) | 0.99 | 0.079 |
| L14 down c122 | MLP | a%100 in {97..99} (+) | 1.00 | 0.030 |
| L14 down c21 | MLP | a%100 in {9} (+) | 1.00 | 0.010 |
| L14 down c142 | MLP | a%100 in {9} (-) | 1.00 | 0.010 |

**code 2s-227.1 (L16): 23 comps, tells apart 32/100, coverage 0.74, overlap 1.73 (random 1.72, p 0.53)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c893 | MLP | a%100 in {0, 69, 71, 73..74, 78..80, 90} (+) | 0.99 | 0.154 |
| L16 down c36 | MLP | a%100 in {0..3, 70} (-) | 1.00 | 0.980 |
| L16 down c143 | MLP | a%100 in {1..3} (+) | 1.00 | 0.090 |
| L16 down c4 | MLP | a%100 in {1..4, 14, 16..18, 21..22, 24, 26..28, 30, 34, 39, 41..42, 49, 51, 70, 84, 99} (+) | 1.00 | 1.000 |
| L16 down c1020 | MLP | a%100 in {10..12} (-) | 1.00 | 0.030 |
| L16 down c20 | MLP | a%100 in {10..22} (-) | 1.00 | 0.160 |
| L16 down c59 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L16 down c336 | MLP | a%100 in {2, 62} (+) | 1.00 | 0.030 |
| L16 down c282 | MLP | a%100 in {2..6} (-) | 1.00 | 0.050 |
| L16 down c30 | MLP | a%100 in {24} (+) | 1.00 | 0.020 |
| L16 down c44 | MLP | a%100 in {30..31} (-) | 1.00 | 0.050 |
| L16 down c50 | MLP | a%100 in {31, 51} (+) | 1.00 | 0.020 |
| L16 down c272 | MLP | a%100 in {3} (-) | 1.00 | 0.010 |
| L16 down c3 | MLP | a%100 in {40..47, 49..53} (+) | 1.00 | 0.220 |
| L16 down c587 | MLP | a%100 in {51, 53} (+) | 1.00 | 0.050 |
| L16 down c7 | MLP | a%100 in {68..69, 71..73, 75..90} (-) | 1.00 | 0.258 |
| L16 down c74 | MLP | a%100 in {7, 77} (+) | 1.00 | 0.020 |
| L16 down c136 | MLP | a%100 in {70} (+) | 1.00 | 0.070 |
| L16 down c280 | MLP | a%100 in {74..76} (-) | 1.00 | 0.030 |
| L16 down c111 | MLP | a%100 in {81..86} (-) | 1.00 | 0.060 |
| L16 down c79 | MLP | a%100 in {81} (+) | 1.00 | 0.010 |
| L16 down c34 | MLP | a%100 in {8} (-) | 1.00 | 0.010 |
| L16 down c139 | MLP | a%100 in {93..99} (-) | 1.00 | 0.072 |

**code 2s-227.2 (L17): 23 comps, tells apart 34/100, coverage 0.65, overlap 1.82 (random 1.66, p 0.72)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 down c8 | MLP | a%100 in {0, 20, 30, 40, 50, 70, 80, 90} (+) | 1.00 | 0.120 |
| L17 down c273 | MLP | a%100 in {0} (+) | 1.00 | 0.040 |
| L17 down c2 | MLP | a%100 in {1, 70} (+) | 1.00 | 0.990 |
| L17 down c6 | MLP | a%100 in {1..8} (-) | 1.00 | 0.080 |
| L17 down c56 | MLP | a%100 in {10..11} (-) | 1.00 | 0.040 |
| L17 down c32 | MLP | a%100 in {12..20, 59..69, 71..79, 82..84} (+) | 1.00 | 0.460 |
| L17 down c440 | MLP | a%100 in {12..22, 81..88} (-) | 1.00 | 0.200 |
| L17 down c66 | MLP | a%100 in {12} (+) | 1.00 | 0.020 |
| L17 down c245 | MLP | a%100 in {13..14} (-) | 1.00 | 0.020 |
| L17 down c48 | MLP | a%100 in {14..30} (-) | 1.00 | 0.180 |
| L17 down c186 | MLP | a%100 in {15} (+) | 1.00 | 0.010 |
| L17 down c304 | MLP | a%100 in {1} (-) | 1.00 | 0.030 |
| L17 down c143 | MLP | a%100 in {2..5} (+) | 1.00 | 0.040 |
| L17 down c58 | MLP | a%100 in {2..8} (-) | 1.00 | 0.080 |
| L17 down c331 | MLP | a%100 in {24} (+) | 1.00 | 0.010 |
| L17 down c167 | MLP | a%100 in {4} (-) | 1.00 | 0.010 |
| L17 down c491 | MLP | a%100 in {51, 70} (+) | 1.00 | 0.050 |
| L17 down c161 | MLP | a%100 in {5} (-) | 1.00 | 0.030 |
| L17 down c681 | MLP | a%100 in {61..63, 67} (-) | 0.99 | 0.071 |
| L17 down c54 | MLP | a%100 in {6} (+) | 0.99 | 0.032 |
| L17 down c609 | MLP | a%100 in {70} (+) | 1.00 | 0.010 |
| L17 down c1012 | MLP | a%100 in {70} (+) | 1.00 | 0.010 |
| L17 down c120 | MLP | a%100 in {9} (-) | 1.00 | 0.010 |

**code 2s-227.3 (L19 L20): 27 comps, tells apart 50/100, coverage 0.70, overlap 1.67 (random 1.86, p 0.20)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L19 down c860 | MLP | a%100 in {0, 40, 80} (+) | 1.00 | 0.100 |
| L19 down c47 | MLP | a%100 in {0, 70, 80} (-) | 1.00 | 0.101 |
| L19 down c45 | MLP | a%100 in {0..1, 9, 42, 60, 93, 95, 97, 99} (-) | 1.00 | 1.000 |
| L19 down c541 | MLP | a%100 in {1..2} (-) | 1.00 | 0.040 |
| L19 down c21 | MLP | a%100 in {1..3, 70} (+) | 1.00 | 0.990 |
| L19 down c746 | MLP | a%100 in {13..23} (+) | 1.00 | 0.200 |
| L19 down c52 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L19 down c893 | MLP | a%100 in {21, 41, 51, 61, 71, 81} (+) | 1.00 | 0.090 |
| L19 down c677 | MLP | a%100 in {24} (-) | 1.00 | 0.010 |
| L19 down c93 | MLP | a%100 in {3, 53} (-) | 1.00 | 0.050 |
| L19 down c27 | MLP | a%100 in {41, 49, 51, 53, 57, 59, 61..63} (+) | 1.00 | 0.290 |
| L19 down c228 | MLP | a%100 in {41..44} (-) | 1.00 | 0.040 |
| L19 down c168 | MLP | a%100 in {42..43} (+) | 1.00 | 0.020 |
| L19 down c135 | MLP | a%100 in {42} (+) | 1.00 | 0.010 |
| L19 down c430 | MLP | a%100 in {47, 49, 51, 53} (+) | 1.00 | 0.080 |
| L19 down c643 | MLP | a%100 in {51..57} (-) | 1.00 | 0.110 |
| L19 down c662 | MLP | a%100 in {51} (-) | 1.00 | 0.020 |
| L19 down c319 | MLP | a%100 in {58..68} (-) | 1.00 | 0.140 |
| L19 down c78 | MLP | a%100 in {6..9} (-) | 1.00 | 0.050 |
| L19 down c1011 | MLP | a%100 in {62} (+) | 1.00 | 0.040 |
| L19 down c195 | MLP | a%100 in {65, 67..69, 71..79} (-) | 1.00 | 0.350 |
| L19 down c242 | MLP | a%100 in {70, 74} (-) | 0.99 | 0.035 |
| L19 down c707 | MLP | a%100 in {70} (+) | 1.00 | 0.040 |
| L19 down c520 | MLP | a%100 in {74..75} (+) | 1.00 | 0.080 |
| L19 down c309 | MLP | a%100 in {92..99} (-) | 1.00 | 0.100 |
| L19 down c51 | MLP | a%100 in {9} (-) | 1.00 | 0.010 |
| L20 o c60 | H19 | a%100 in {26..29} (+) | 1.00 | 0.040 |

**code 2s-227.4 (L22): 35 comps, tells apart 65/100, coverage 0.88, overlap 1.82 (random 2.20, p 0.07)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 down c183 | MLP | a%100 in {0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 80, 90} (+) | 1.00 | 0.160 |
| L22 down c0 | MLP | a%100 in {0, 70} (+) | 1.00 | 0.061 |
| L22 down c638 | MLP | a%100 in {0, 80} (-) | 1.00 | 0.020 |
| L22 down c352 | MLP | a%100 in {0} (+) | 1.00 | 0.020 |
| L22 down c187 | MLP | a%100 in {1, 41} (-) | 1.00 | 0.070 |
| L22 down c10 | MLP | a%100 in {1..8, 70} (+) | 1.00 | 0.990 |
| L22 down c552 | MLP | a%100 in {10..13} (-) | 1.00 | 0.050 |
| L22 down c151 | MLP | a%100 in {11..23} (+) | 1.00 | 0.420 |
| L22 down c881 | MLP | a%100 in {15} (-) | 1.00 | 0.010 |
| L22 down c8 | MLP | a%100 in {19} (+) | 1.00 | 0.010 |
| L22 down c93 | MLP | a%100 in {1} (+) | 1.00 | 0.020 |
| L22 down c521 | MLP | a%100 in {2..4} (-) | 1.00 | 0.040 |
| L22 down c43 | MLP | a%100 in {2..5, 41..44} (-) | 1.00 | 0.090 |
| L22 down c26 | MLP | a%100 in {24, 52} (-) | 1.00 | 0.071 |
| L22 down c27 | MLP | a%100 in {25..31, 33} (+) | 1.00 | 0.170 |
| L22 down c666 | MLP | a%100 in {32..40, 46} (-) | 1.00 | 0.120 |
| L22 down c5 | MLP | a%100 in {38..44} (-) | 1.00 | 0.100 |
| L22 down c135 | MLP | a%100 in {41..55} (-) | 1.00 | 0.221 |
| L22 down c488 | MLP | a%100 in {42} (-) | 1.00 | 0.010 |
| L22 down c609 | MLP | a%100 in {49} (-) | 0.99 | 0.072 |
| L22 down c706 | MLP | a%100 in {51, 53} (+) | 1.00 | 0.090 |
| L22 down c35 | MLP | a%100 in {51} (-) | 1.00 | 0.010 |
| L22 down c98 | MLP | a%100 in {52..56} (-) | 1.00 | 0.050 |
| L22 down c121 | MLP | a%100 in {56, 65..68, 86, 88} (+) | 1.00 | 0.090 |
| L22 down c571 | MLP | a%100 in {6..8} (-) | 1.00 | 0.030 |
| L22 down c88 | MLP | a%100 in {69, 71..87} (-) | 1.00 | 0.210 |
| L22 down c1014 | MLP | a%100 in {7..11} (-) | 1.00 | 0.060 |
| L22 down c23 | MLP | a%100 in {70} (+) | 1.00 | 0.050 |
| L22 down c326 | MLP | a%100 in {70} (+) | 1.00 | 0.010 |
| L22 down c885 | MLP | a%100 in {70} (-) | 1.00 | 0.010 |
| L22 down c28 | MLP | a%100 in {79} (+) | 1.00 | 0.010 |
| L22 down c59 | MLP | a%100 in {80, 84} (+) | 1.00 | 0.090 |
| L22 down c374 | MLP | a%100 in {81} (+) | 1.00 | 0.010 |
| L22 down c324 | MLP | a%100 in {84, 96..98} (-) | 1.00 | 0.090 |
| L22 down c752 | MLP | a%100 in {95, 97..99} (-) | 1.00 | 0.040 |

**code 2s-227.5 (L24): 47 comps, tells apart 69/100, coverage 1.00, overlap 2.23 (random 2.73, p 0.04)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 down c181 | MLP | a%100 in {0, 50} (+) | 1.00 | 0.030 |
| L24 down c0 | MLP | a%100 in {0, 70} (+) | 1.00 | 0.040 |
| L24 down c6 | MLP | a%100 in {0..10, 33..34, 36..39, 43, 46..50, 58..59, 70, 94..99} (-) | 1.00 | 1.000 |
| L24 down c470 | MLP | a%100 in {1..6} (-) | 1.00 | 0.080 |
| L24 down c1 | MLP | a%100 in {1..9, 31, 33, 37, 39} (+) | 1.00 | 0.331 |
| L24 down c875 | MLP | a%100 in {10..11} (-) | 1.00 | 0.020 |
| L24 down c29 | MLP | a%100 in {10..23} (-) | 1.00 | 0.230 |
| L24 down c171 | MLP | a%100 in {13..15} (+) | 1.00 | 0.030 |
| L24 down c22 | MLP | a%100 in {15} (+) | 1.00 | 0.010 |
| L24 down c4 | MLP | a%100 in {17, 27} (+) | 1.00 | 0.060 |
| L24 down c212 | MLP | a%100 in {19} (-) | 1.00 | 0.010 |
| L24 down c606 | MLP | a%100 in {1} (-) | 1.00 | 0.021 |
| L24 down c110 | MLP | a%100 in {20..31} (-) | 1.00 | 0.140 |
| L24 down c5 | MLP | a%100 in {22, 82, 92} (-) | 1.00 | 0.030 |
| L24 down c16 | MLP | a%100 in {22..23, 72} (+) | 1.00 | 0.083 |
| L24 down c9 | MLP | a%100 in {24, 48} (-) | 1.00 | 0.040 |
| L24 down c50 | MLP | a%100 in {24..25} (-) | 1.00 | 0.020 |
| L24 down c32 | MLP | a%100 in {25..29, 31} (+) | 1.00 | 0.100 |
| L24 down c234 | MLP | a%100 in {25..31, 34..37} (+) | 1.00 | 0.120 |
| L24 down c52 | MLP | a%100 in {32..33} (+) | 1.00 | 0.020 |
| L24 down c621 | MLP | a%100 in {33, 53} (-) | 1.00 | 0.030 |
| L24 down c548 | MLP | a%100 in {40, 45, 50} (+) | 1.00 | 0.080 |
| L24 down c134 | MLP | a%100 in {41..42, 44} (-) | 1.00 | 0.030 |
| L24 down c397 | MLP | a%100 in {45} (-) | 1.00 | 0.010 |
| L24 down c524 | MLP | a%100 in {47, 49} (+) | 1.00 | 0.030 |
| L24 down c17 | MLP | a%100 in {47..48} (-) | 0.99 | 0.051 |
| L24 down c28 | MLP | a%100 in {49, 51..53} (+) | 1.00 | 0.063 |
| L24 down c206 | MLP | a%100 in {51..55} (+) | 1.00 | 0.111 |
| L24 down c189 | MLP | a%100 in {51} (+) | 1.00 | 0.060 |
| L24 down c167 | MLP | a%100 in {52, 54, 56} (-) | 1.00 | 0.030 |
| L24 down c13 | MLP | a%100 in {55..57} (+) | 1.00 | 0.030 |
| L24 down c8 | MLP | a%100 in {56..57, 66, 68..69} (-) | 1.00 | 0.060 |
| L24 down c432 | MLP | a%100 in {59, 65, 67, 71, 73..77} (-) | 1.00 | 0.187 |
| L24 down c147 | MLP | a%100 in {60..62} (+) | 1.00 | 0.092 |
| L24 down c11 | MLP | a%100 in {60..63} (+) | 1.00 | 0.040 |
| L24 down c866 | MLP | a%100 in {62} (-) | 1.00 | 0.010 |
| L24 down c226 | MLP | a%100 in {63..69, 71..72, 74} (+) | 1.00 | 0.188 |
| L24 down c36 | MLP | a%100 in {69, 71..73} (-) | 1.00 | 0.040 |
| L24 down c260 | MLP | a%100 in {70} (+) | 1.00 | 0.010 |
| L24 down c18 | MLP | a%100 in {70} (-) | 1.00 | 0.010 |
| L24 down c15 | MLP | a%100 in {78..86, 88} (+) | 1.00 | 0.210 |
| L24 down c39 | MLP | a%100 in {82..96} (+) | 0.99 | 0.186 |
| L24 down c37 | MLP | a%100 in {84..85} (+) | 1.00 | 0.050 |
| L24 down c88 | MLP | a%100 in {91..94} (+) | 1.00 | 0.040 |
| L24 down c12 | MLP | a%100 in {94} (-) | 1.00 | 0.010 |
| L24 down c91 | MLP | a%100 in {94} (-) | 0.93 | 0.011 |
| L24 down c68 | MLP | a%100 in {96..98} (-) | 1.00 | 0.030 |

</details>

</details>

<details><summary><b>2s-230</b> `a%100` @ `op` (sub) — single component; 1 comps, L16; tells apart 2/100 classes (best member 2); on add: 2a-175 (member overlap 0.50)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.30 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 100) |
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
| input point | L15 MLP |
| CKA(arrangement before, joint write) | 0.45 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.34 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.74) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.27 / 1.00 |
| best source position (CKA) | `a` (0.31) |

<details><summary>codes and components</summary>

**code 2s-230.0 (L16): 1 comps, tells apart 2/100, coverage 0.30, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c207 | H30 | a%100 in {1, 31, 40..42, 44, 47..52, 62, 75, 83..91, 93..99} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2s-228</b> `a%100` @ `op` (sub) — block code; 22 comps, L16; tells apart 74/100 classes (best member 8); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.73 |
| classes told apart: joint / best code / best member | 74 /  / 8 (of 100) |
| members whose removal merges classes | 0.73 |
| support overlap (1 = tiling) / random sets / p | 2.32 / 1.67 / 1.00 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.86 /  / 0.29 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.14 |
| consumers / read jointly | 1 / 1 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 MLP |
| CKA(arrangement before, joint write) | 0.47 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.08 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.34 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.21 2:0.20 3:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.35 / 6.47 |
| best source position (CKA) | `a` (0.49) |

<details><summary>codes and components</summary>

**code 2s-228.0 (L16): 22 comps, tells apart 74/100, coverage 0.73, overlap 2.32 (random 1.66, p 0.98)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c129 | H21 | a%100 in {0, 66, 68..70, 80, 90} (-) | 1.00 | 0.232 |
| L16 o c98 | H21 | a%100 in {0, 89..94} (-) | 1.00 | 0.130 |
| L16 o c37 | H21 | a%100 in {14..15} (+) | 1.00 | 0.030 |
| L16 o c26 | H21 | a%100 in {15, 25, 55, 65, 75, 85} (-) | 1.00 | 0.060 |
| L16 o c97 | H21 | a%100 in {16, 26, 35..36, 45, 55..56, 66, 85..86} (+) | 1.00 | 0.121 |
| L16 o c87 | H21 | a%100 in {21..33, 35..36} (-) | 1.00 | 0.203 |
| L16 o c41 | H21 | a%100 in {27, 29..37} (+) | 1.00 | 0.180 |
| L16 o c53 | H21 | a%100 in {31, 39, 49, 51, 59, 69, 71, 79, 81, 89, 91} (+) | 1.00 | 0.252 |
| L16 o c34 | H21 | a%100 in {31, 41, 61, 81, 91} (+) | 1.00 | 0.061 |
| L16 o c257 | H21 | a%100 in {31, 61, 71, 81, 91} (+) | 1.00 | 0.050 |
| L16 o c138 | H21 | a%100 in {35..45, 47..57} (-) | 1.00 | 0.290 |
| L16 o c195 | H21 | a%100 in {39, 58..59, 78..80} (+) | 1.00 | 0.091 |
| L16 o c165 | H21 | a%100 in {47, 49, 51, 53} (-) | 1.00 | 0.040 |
| L16 o c121 | H21 | a%100 in {54, 64, 82, 84, 92, 94} (-) | 1.00 | 0.171 |
| L16 o c142 | H21 | a%100 in {56, 66, 76, 86} (-) | 1.00 | 0.040 |
| L16 o c28 | H21 | a%100 in {56..61, 63} (+) | 1.00 | 0.080 |
| L16 o c94 | H21 | a%100 in {59..62} (-) | 1.00 | 0.041 |
| L16 o c52 | H21 | a%100 in {71, 77..78, 81, 83..87} (+) | 1.00 | 0.121 |
| L16 o c31 | H21 | a%100 in {77, 79..91} (-) | 1.00 | 0.180 |
| L16 o c16 | H21 | a%100 in {80..92} (+) | 1.00 | 0.180 |
| L16 o c180 | H21 | a%100 in {87} (-) | 1.00 | 0.050 |
| L16 o c72 | H21 | a%100 in {88} (-) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2s-229</b> `a%100` @ `op` (sub) — block code; 25 comps, L16; tells apart 58/100 classes (best member 7); on add: 2a-177 (member overlap 0.47)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.57 |
| classes told apart: joint / best code / best member | 58 /  / 7 (of 100) |
| members whose removal merges classes | 0.60 |
| support overlap (1 = tiling) / random sets / p | 1.93 / 1.78 / 0.69 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.84 /  / 0.28 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.04 |
| consumers / read jointly | 2 / 2 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 MLP |
| CKA(arrangement before, joint write) | 0.45 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.12 |
| share of the write inside the old arrangement's span | 0.11 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.34 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.27 2:0.11 3:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.34 / 6.38 |
| best source position (CKA) | `a` (0.43) |

<details><summary>codes and components</summary>

**code 2s-229.0 (L16): 25 comps, tells apart 58/100, coverage 0.57, overlap 1.93 (random 1.77, p 0.76)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c197 | H22 | a%100 in {0, 50..52, 55, 60, 70} (+) | 1.00 | 0.280 |
| L16 o c492 | H22 | a%100 in {24, 44, 54, 64, 74, 84} (+) | 1.00 | 0.090 |
| L16 o c113 | H22 | a%100 in {29, 49, 59, 69, 79} (-) | 1.00 | 0.071 |
| L16 o c329 | H22 | a%100 in {33, 43, 53, 63, 73, 83} (+) | 1.00 | 0.070 |
| L16 o c91 | H22 | a%100 in {40..45, 47, 51..52} (+) | 1.00 | 0.220 |
| L16 o c268 | H22 | a%100 in {41, 51, 61, 71, 81} (+) | 1.00 | 0.080 |
| L16 o c337 | H22 | a%100 in {42, 52, 62, 72} (-) | 1.00 | 0.100 |
| L16 o c261 | H22 | a%100 in {43, 53..57, 63} (+) | 1.00 | 0.070 |
| L16 o c73 | H22 | a%100 in {46..47} (-) | 1.00 | 0.020 |
| L16 o c181 | H22 | a%100 in {50} (-) | 1.00 | 0.010 |
| L16 o c248 | H22 | a%100 in {51} (+) | 1.00 | 0.010 |
| L16 o c392 | H22 | a%100 in {51} (+) | 1.00 | 0.010 |
| L16 o c387 | H22 | a%100 in {53..54, 74} (-) | 0.99 | 0.051 |
| L16 o c119 | H22 | a%100 in {57, 66..69, 71..73, 77} (-) | 1.00 | 0.121 |
| L16 o c237 | H22 | a%100 in {57..63} (-) | 1.00 | 0.132 |
| L16 o c170 | H22 | a%100 in {6, 36, 46, 56, 66, 76, 96} (-) | 1.00 | 0.070 |
| L16 o c56 | H22 | a%100 in {69, 71, 73..77, 79} (+) | 1.00 | 0.130 |
| L16 o c314 | H22 | a%100 in {7, 47, 77} (-) | 1.00 | 0.030 |
| L16 o c188 | H22 | a%100 in {70, 74} (-) | 1.00 | 0.040 |
| L16 o c58 | H22 | a%100 in {70, 80..81, 90..91} (-) | 1.00 | 0.050 |
| L16 o c321 | H22 | a%100 in {74..75} (-) | 1.00 | 0.020 |
| L16 o c71 | H22 | a%100 in {75} (+) | 1.00 | 0.010 |
| L16 o c355 | H22 | a%100 in {8, 38, 58} (+) | 1.00 | 0.030 |
| L16 o c378 | H22 | a%100 in {8} (+) | 0.99 | 0.010 |
| L16 o c267 | H22 | a%100 in {95..99} (-) | 1.00 | 0.050 |

</details>

</details>

<details><summary><b>2s-231</b> `a%100` @ `op` (sub) — block code; 2 comps, L17 L18; tells apart 3/100 classes (best member 4); on add: 2a-178 (member overlap 0.33)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.33 |
| classes told apart: joint / best code / best member | 3 /  / 4 (of 100) |
| members whose removal merges classes | 0.50 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.83 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.91 /  / 0.56 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L16 MLP |
| CKA(arrangement before, joint write) | 0.31 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.35 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (2:0.42 1:0.34 3:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.16 / 1.08 |
| best source position (CKA) | `a` (0.38) |

<details><summary>codes and components</summary>

**code 2s-231.0 (L17 L18): 2 comps, tells apart 3/100, coverage 0.33, overlap 1.00 (random 1.00, p 0.81)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 o c17 | H9 | a%100 in {0, 51, 53, 62, 70, 89..90, 92..99} (+) | 1.00 | 1.000 |
| L18 o c442 | H1 | a%100 in {12..29} (-) | 1.00 | 0.820 |

</details>

</details>

<details><summary><b>2s-232</b> `a%100` @ `op` (sub) — block code; 4 comps, L18 L19; tells apart 12/100 classes (best member 7); on add: 2a-181 (member overlap 0.25)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.09 |
| classes told apart: joint / best code / best member | 12 /  / 7 (of 100) |
| members whose removal merges classes | 0.75 |
| support overlap (1 = tiling) / random sets / p | 1.11 / 1.07 / 0.66 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.62 /  / 0.56 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.99 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L17 MLP |
| CKA(arrangement before, joint write) | 0.26 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.12 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.35 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (2:0.11 1:0.10 3:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.06 / 2.81 |
| best source position (CKA) | `a` (0.19) |

<details><summary>codes and components</summary>

**code 2s-232.0 (L18 L19): 4 comps, tells apart 12/100, coverage 0.09, overlap 1.11 (random 1.05, p 0.73)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 o c436 | H17 | a%100 in {51..53, 55} (-) | 1.00 | 0.070 |
| L18 o c299 | H17 | a%100 in {51} (-) | 1.00 | 0.010 |
| L18 o c495 | H17 | a%100 in {70} (+) | 1.00 | 0.030 |
| L19 o c23 | H7 | a%100 in {1..4} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2s-233</b> `a%100` @ `op` (sub) — block code, copy from `a`; 79 comps, L18; tells apart 89/100 classes (best member 7); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.79 |
| classes told apart: joint / best code / best member | 89 /  / 7 (of 100) |
| members whose removal merges classes | 0.15 |
| support overlap (1 = tiling) / random sets / p | 4.58 / 4.35 / 0.68 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.99 /  / 0.38 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.06 |
| consumers / read jointly | 41 / 33 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L17 MLP |
| CKA(arrangement before, joint write) | 0.60 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.22 |
| share of the write inside the old arrangement's span | 0.17 |
| write energy / code energy before | 0.07 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.35 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.21 2:0.11 3:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.63 / 15.33 |
| best source position (CKA) | `a` (0.65) |

<details><summary>codes and components</summary>

**code 2s-233.0 (L18): 79 comps, tells apart 89/100, coverage 0.79, overlap 4.58 (random 4.29, p 0.67)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 o c446 | H30 | a%100 in {0, 30} (-) | 1.00 | 0.020 |
| L18 o c107 | H30 | a%100 in {0, 78, 80, 84..89, 96} (-) | 1.00 | 0.320 |
| L18 o c445 | H30 | a%100 in {13, 23, 33, 43, 73} (+) | 1.00 | 0.170 |
| L18 o c382 | H30 | a%100 in {13..16} (+) | 1.00 | 0.040 |
| L18 o c66 | H30 | a%100 in {13..24} (-) | 1.00 | 0.200 |
| L18 o c362 | H30 | a%100 in {14, 16..18} (+) | 1.00 | 0.040 |
| L18 o c453 | H30 | a%100 in {14, 16} (+) | 1.00 | 0.020 |
| L18 o c195 | H30 | a%100 in {14, 24, 34, 44, 54, 64, 74, 84, 94} (+) | 1.00 | 0.130 |
| L18 o c208 | H30 | a%100 in {14, 24, 34, 54, 74, 84} (+) | 1.00 | 0.120 |
| L18 o c53 | H30 | a%100 in {15, 17, 19..20, 50} (+) | 1.00 | 0.280 |
| L18 o c233 | H30 | a%100 in {15, 23, 25, 33..35, 43, 85} (-) | 1.00 | 0.080 |
| L18 o c160 | H30 | a%100 in {15, 35, 55, 85} (-) | 1.00 | 0.040 |
| L18 o c159 | H30 | a%100 in {16, 26, 56, 64, 66, 68} (+) | 1.00 | 0.130 |
| L18 o c444 | H30 | a%100 in {16..17, 26..27, 47, 87} (-) | 1.00 | 0.100 |
| L18 o c227 | H30 | a%100 in {16} (-) | 1.00 | 0.010 |
| L18 o c96 | H30 | a%100 in {17, 27, 37, 42, 47, 57, 67, 87} (+) | 1.00 | 0.150 |
| L18 o c142 | H30 | a%100 in {17..18, 28, 38, 48, 58, 68, 78, 88} (+) | 1.00 | 0.200 |
| L18 o c218 | H30 | a%100 in {18, 24, 28, 48} (+) | 1.00 | 0.040 |
| L18 o c308 | H30 | a%100 in {19, 23, 29} (-) | 1.00 | 0.070 |
| L18 o c290 | H30 | a%100 in {19, 39, 49, 51} (+) | 1.00 | 0.252 |
| L18 o c80 | H30 | a%100 in {19..26} (-) | 1.00 | 0.120 |
| L18 o c138 | H30 | a%100 in {19} (-) | 1.00 | 0.010 |
| L18 o c306 | H30 | a%100 in {20, 40} (+) | 1.00 | 0.020 |
| L18 o c147 | H30 | a%100 in {20..26} (+) | 1.00 | 0.130 |
| L18 o c14 | H30 | a%100 in {21, 31, 41, 51, 61} (-) | 1.00 | 0.200 |
| L18 o c164 | H30 | a%100 in {21, 51, 53} (-) | 1.00 | 0.040 |
| L18 o c356 | H30 | a%100 in {21..22, 31, 41} (+) | 1.00 | 0.090 |
| L18 o c97 | H30 | a%100 in {22..23, 42, 52, 62, 72, 82} (+) | 1.00 | 0.190 |
| L18 o c49 | H30 | a%100 in {22..28, 32, 36, 40, 42, 44..46, 48, 52, 56, 72} (-) | 1.00 | 0.400 |
| L18 o c292 | H30 | a%100 in {24, 43..44, 48, 64} (+) | 1.00 | 0.070 |
| L18 o c346 | H30 | a%100 in {24, 84} (+) | 1.00 | 0.030 |
| L18 o c407 | H30 | a%100 in {25, 30} (+) | 1.00 | 0.020 |
| L18 o c412 | H30 | a%100 in {25, 55, 65, 75, 85} (-) | 1.00 | 0.050 |
| L18 o c427 | H30 | a%100 in {25..29} (+) | 1.00 | 0.050 |
| L18 o c114 | H30 | a%100 in {26..31} (+) | 1.00 | 0.070 |
| L18 o c135 | H30 | a%100 in {29..33} (-) | 1.00 | 0.070 |
| L18 o c82 | H30 | a%100 in {30..40} (+) | 1.00 | 0.130 |
| L18 o c239 | H30 | a%100 in {31} (+) | 1.00 | 0.010 |
| L18 o c418 | H30 | a%100 in {34, 54, 74, 78} (+) | 1.00 | 0.040 |
| L18 o c271 | H30 | a%100 in {34, 94} (+) | 1.00 | 0.020 |
| L18 o c37 | H30 | a%100 in {34..35, 45} (-) | 1.00 | 0.030 |
| L18 o c266 | H30 | a%100 in {35, 45, 54..56, 65} (-) | 1.00 | 0.100 |
| L18 o c229 | H30 | a%100 in {38, 58} (+) | 1.00 | 0.020 |
| L18 o c236 | H30 | a%100 in {38..44} (-) | 1.00 | 0.070 |
| L18 o c137 | H30 | a%100 in {40..45} (+) | 1.00 | 0.100 |
| L18 o c221 | H30 | a%100 in {41..42} (-) | 1.00 | 0.020 |
| L18 o c310 | H30 | a%100 in {41} (+) | 1.00 | 0.010 |
| L18 o c422 | H30 | a%100 in {45..49} (+) | 1.00 | 0.090 |
| L18 o c361 | H30 | a%100 in {45} (-) | 1.00 | 0.010 |
| L18 o c62 | H30 | a%100 in {46, 48, 50, 52, 54, 56..58} (-) | 1.00 | 0.080 |
| L18 o c402 | H30 | a%100 in {46..51} (+) | 1.00 | 0.110 |
| L18 o c151 | H30 | a%100 in {47, 49, 52..54} (+) | 1.00 | 0.070 |
| L18 o c190 | H30 | a%100 in {49, 51, 53} (-) | 1.00 | 0.050 |
| L18 o c304 | H30 | a%100 in {49, 51, 53} (-) | 1.00 | 0.030 |
| L18 o c134 | H30 | a%100 in {49, 69} (-) | 1.00 | 0.020 |
| L18 o c42 | H30 | a%100 in {49..57, 60..62} (-) | 1.00 | 0.190 |
| L18 o c154 | H30 | a%100 in {50, 80} (-) | 1.00 | 0.020 |
| L18 o c146 | H30 | a%100 in {51..54} (+) | 1.00 | 0.090 |
| L18 o c106 | H30 | a%100 in {51} (-) | 1.00 | 0.010 |
| L18 o c487 | H30 | a%100 in {54, 64} (-) | 1.00 | 0.020 |
| L18 o c187 | H30 | a%100 in {54} (-) | 1.00 | 0.010 |
| L18 o c455 | H30 | a%100 in {56..57} (+) | 1.00 | 0.020 |
| L18 o c325 | H30 | a%100 in {58, 78, 88} (+) | 1.00 | 0.030 |
| L18 o c413 | H30 | a%100 in {58..60, 62} (-) | 1.00 | 0.040 |
| L18 o c144 | H30 | a%100 in {58..63} (+) | 1.00 | 0.060 |
| L18 o c173 | H30 | a%100 in {6, 15..16, 24..26} (-) | 1.00 | 0.060 |
| L18 o c73 | H30 | a%100 in {60, 62..68} (-) | 0.99 | 0.166 |
| L18 o c454 | H30 | a%100 in {60..61} (-) | 1.00 | 0.020 |
| L18 o c462 | H30 | a%100 in {60..62, 65..67} (-) | 1.00 | 0.060 |
| L18 o c272 | H30 | a%100 in {61..62} (-) | 1.00 | 0.030 |
| L18 o c355 | H30 | a%100 in {70, 72, 74} (+) | 1.00 | 0.140 |
| L18 o c56 | H30 | a%100 in {70, 74..75} (-) | 1.00 | 0.060 |
| L18 o c171 | H30 | a%100 in {70, 75..77, 79..81} (+) | 1.00 | 0.070 |
| L18 o c74 | H30 | a%100 in {70} (+) | 1.00 | 0.040 |
| L18 o c158 | H30 | a%100 in {70} (+) | 1.00 | 0.190 |
| L18 o c433 | H30 | a%100 in {75} (+) | 1.00 | 0.010 |
| L18 o c152 | H30 | a%100 in {77..78} (-) | 1.00 | 0.020 |
| L18 o c212 | H30 | a%100 in {77} (+) | 1.00 | 0.010 |
| L18 o c88 | H30 | a%100 in {79..82, 84..86} (-) | 1.00 | 0.220 |

</details>

</details>

<details><summary><b>2s-235</b> `a%100` @ `op` (sub) — block code; 3 comps, L20 L21; tells apart 7/100 classes (best member 5); on add: 2a-184 (member overlap 0.33)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.06 |
| classes told apart: joint / best code / best member | 7 /  / 5 (of 100) |
| members whose removal merges classes | 0.67 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.64 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.35 /  / 0.31 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.07 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L19 MLP |
| CKA(arrangement before, joint write) | 0.12 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.32 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.08 10:0.08 3:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.03 / 1.78 |
| best source position (CKA) | `a` (0.11) |

<details><summary>codes and components</summary>

**code 2s-235.0 (L20 L21): 3 comps, tells apart 7/100, coverage 0.06, overlap 1.00 (random 1.00, p 0.66)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 o c192 | H2 | a%100 in {30, 90} (+) | 1.00 | 0.040 |
| L20 o c479 | H2 | a%100 in {40} (+) | 1.00 | 0.010 |
| L21 o c222 | H18 | a%100 in {0, 18, 70} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2s-234</b> `a%100` @ `op` (sub) — block code; 5 comps, L20; tells apart 6/100 classes (best member 5); on add: 2a-182 (member overlap 0.17)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.23 |
| classes told apart: joint / best code / best member | 6 /  / 5 (of 100) |
| members whose removal merges classes | 0.20 |
| support overlap (1 = tiling) / random sets / p | 1.17 / 1.09 / 0.76 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.66 /  / 0.63 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.18 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L19 MLP |
| CKA(arrangement before, joint write) | 0.22 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.32 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (40:0.08 3:0.06 1:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.04 / 1.91 |
| best source position (CKA) | `a` (0.21) |

<details><summary>codes and components</summary>

**code 2s-234.0 (L20): 5 comps, tells apart 6/100, coverage 0.23, overlap 1.17 (random 1.08, p 0.79)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 o c0 | H17 | a%100 in {0, 10..12, 24, 28, 30, 45, 56, 58, 60, 62, 70, 72, 74, 77..78, 80, 90, 98} (-) | 1.00 | 0.820 |
| L20 o c74 | H18 | a%100 in {1..2, 6} (+) | 1.00 | 0.060 |
| L20 o c124 | H18 | a%100 in {1..2} (-) | 1.00 | 0.060 |
| L20 o c81 | H18 | a%100 in {1} (+) | 1.00 | 0.010 |
| L20 o c176 | H18 | a%100 in {1} (-) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2s-236</b> `a%100` @ `op` (sub) — block code; 6 comps, L20; tells apart 16/100 classes (best member 6); on add: 2a-183 (member overlap 0.22)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.07 |
| classes told apart: joint / best code / best member | 16 /  / 6 (of 100) |
| members whose removal merges classes | 0.67 |
| support overlap (1 = tiling) / random sets / p | 1.14 / 1.12 / 0.61 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.22 /  / 0.10 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.03 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L19 MLP |
| CKA(arrangement before, joint write) | 0.10 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.06 |
| share of the write inside the old arrangement's span | 0.16 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.32 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular () |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.05 / 3.99 |
| best source position (CKA) | `a` (0.10) |

<details><summary>codes and components</summary>

**code 2s-236.0 (L20): 6 comps, tells apart 16/100, coverage 0.07, overlap 1.14 (random 1.11, p 0.60)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 o c492 | H3 | a%100 in {24} (+) | 1.00 | 0.090 |
| L20 o c349 | H3 | a%100 in {30, 45, 90} (+) | 1.00 | 0.050 |
| L20 o c290 | H3 | a%100 in {31} (+) | 1.00 | 0.050 |
| L20 o c417 | H3 | a%100 in {31} (-) | 1.00 | 0.010 |
| L20 o c270 | H3 | a%100 in {52} (+) | 1.00 | 0.050 |
| L20 o c386 | H3 | a%100 in {54} (-) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2s-240</b> `a%100` @ `op` (sub) — tiling, 2 codes of the same shape; 75 comps, L21 L23; tells apart 77/100 classes (best member 8); on add: 2a-165 (member overlap 0.01)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.98 |
| classes told apart: joint / best code / best member | 77 / 68 / 8 (of 100) |
| members whose removal merges classes | 0.27 |
| support overlap (1 = tiling) / random sets / p | 3.04 / 4.00 / 0.01 |
| mean CKA between its codes | 0.74 |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 1.00 / 1.00 / 0.70 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.02 |
| consumers / read jointly | 20 / 19 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 attn |
| CKA(arrangement before, joint write) | 0.80 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.22 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.30 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.25 2:0.16 3:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.45 / 8.45 |

<details><summary>codes and components</summary>

**code 2s-240.0 (L21): 37 comps, tells apart 68/100, coverage 0.76, overlap 1.97 (random 2.33, p 0.11)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c157 | MLP | a%100 in {0, 25, 30, 35, 40, 45, 50, 80} (+) | 1.00 | 0.080 |
| L21 down c138 | MLP | a%100 in {0, 60, 80} (+) | 1.00 | 0.090 |
| L21 down c41 | MLP | a%100 in {1..12, 70} (-) | 1.00 | 0.910 |
| L21 down c241 | MLP | a%100 in {10, 12} (-) | 1.00 | 0.020 |
| L21 down c236 | MLP | a%100 in {10} (+) | 1.00 | 0.010 |
| L21 down c116 | MLP | a%100 in {11..12} (+) | 1.00 | 0.020 |
| L21 down c128 | MLP | a%100 in {11} (+) | 1.00 | 0.010 |
| L21 down c326 | MLP | a%100 in {13..14} (-) | 1.00 | 0.020 |
| L21 down c222 | MLP | a%100 in {13..23} (+) | 1.00 | 0.130 |
| L21 down c873 | MLP | a%100 in {15, 25, 35, 45, 55, 65, 75, 85} (-) | 1.00 | 0.100 |
| L21 down c247 | MLP | a%100 in {17, 27, 37, 47, 57, 67, 77, 87} (+) | 1.00 | 0.090 |
| L21 down c313 | MLP | a%100 in {1} (+) | 1.00 | 0.020 |
| L21 down c39 | MLP | a%100 in {20, 22..26} (-) | 1.00 | 0.140 |
| L21 down c33 | MLP | a%100 in {24, 32, 48, 52, 64} (+) | 1.00 | 0.080 |
| L21 down c358 | MLP | a%100 in {24} (+) | 1.00 | 0.050 |
| L21 down c47 | MLP | a%100 in {3, 5, 7} (+) | 1.00 | 0.030 |
| L21 down c453 | MLP | a%100 in {3..5} (+) | 1.00 | 0.030 |
| L21 down c58 | MLP | a%100 in {3..9} (+) | 1.00 | 0.080 |
| L21 down c144 | MLP | a%100 in {4, 14, 54, 84} (+) | 1.00 | 0.040 |
| L21 down c631 | MLP | a%100 in {4..9} (-) | 1.00 | 0.080 |
| L21 down c674 | MLP | a%100 in {41..44} (-) | 1.00 | 0.090 |
| L21 down c87 | MLP | a%100 in {41} (-) | 1.00 | 0.010 |
| L21 down c334 | MLP | a%100 in {47, 49..53} (+) | 1.00 | 0.228 |
| L21 down c581 | MLP | a%100 in {49} (+) | 1.00 | 0.010 |
| L21 down c747 | MLP | a%100 in {49} (+) | 1.00 | 0.060 |
| L21 down c580 | MLP | a%100 in {51} (-) | 1.00 | 0.080 |
| L21 down c765 | MLP | a%100 in {6} (-) | 1.00 | 0.010 |
| L21 down c769 | MLP | a%100 in {70} (-) | 1.00 | 0.020 |
| L21 down c175 | MLP | a%100 in {71, 75..87, 89} (+) | 1.00 | 0.190 |
| L21 down c15 | MLP | a%100 in {71} (-) | 1.00 | 0.010 |
| L21 down c714 | MLP | a%100 in {73..75} (+) | 1.00 | 0.080 |
| L21 down c481 | MLP | a%100 in {74} (+) | 1.00 | 0.010 |
| L21 down c306 | MLP | a%100 in {74} (-) | 1.00 | 0.010 |
| L21 down c98 | MLP | a%100 in {8} (-) | 1.00 | 0.010 |
| L21 down c162 | MLP | a%100 in {9..17, 19} (-) | 1.00 | 0.150 |
| L21 down c906 | MLP | a%100 in {93, 95..99} (-) | 1.00 | 0.128 |
| L21 down c457 | MLP | a%100 in {9} (+) | 1.00 | 0.010 |

**code 2s-240.1 (L23): 38 comps, tells apart 60/100, coverage 0.84, overlap 1.76 (random 2.32, p 0.01)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 down c939 | MLP | a%100 in {0, 20, 30, 40, 45, 50, 60, 70, 75, 80, 90} (+) | 1.00 | 0.140 |
| L23 down c117 | MLP | a%100 in {0, 24} (-) | 1.00 | 0.110 |
| L23 down c19 | MLP | a%100 in {0..9, 70} (+) | 1.00 | 1.000 |
| L23 down c101 | MLP | a%100 in {0} (+) | 1.00 | 0.070 |
| L23 down c30 | MLP | a%100 in {1..3} (-) | 1.00 | 0.030 |
| L23 down c47 | MLP | a%100 in {1..3} (-) | 1.00 | 0.030 |
| L23 down c82 | MLP | a%100 in {10..21} (-) | 1.00 | 0.160 |
| L23 down c237 | MLP | a%100 in {11} (+) | 1.00 | 0.010 |
| L23 down c14 | MLP | a%100 in {13, 23, 93} (+) | 0.92 | 0.027 |
| L23 down c9 | MLP | a%100 in {15} (-) | 1.00 | 0.010 |
| L23 down c24 | MLP | a%100 in {16, 26, 56, 85..86, 96} (-) | 1.00 | 0.070 |
| L23 down c960 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L23 down c77 | MLP | a%100 in {20} (+) | 1.00 | 0.010 |
| L23 down c4 | MLP | a%100 in {21} (-) | 0.95 | 0.014 |
| L23 down c268 | MLP | a%100 in {2} (+) | 1.00 | 0.010 |
| L23 down c905 | MLP | a%100 in {32..39} (-) | 1.00 | 0.080 |
| L23 down c5 | MLP | a%100 in {33..38} (-) | 1.00 | 0.090 |
| L23 down c15 | MLP | a%100 in {35..38} (-) | 1.00 | 0.050 |
| L23 down c603 | MLP | a%100 in {41, 51, 61} (+) | 1.00 | 0.050 |
| L23 down c593 | MLP | a%100 in {49, 51..53} (-) | 1.00 | 0.080 |
| L23 down c722 | MLP | a%100 in {5..7} (-) | 1.00 | 0.030 |
| L23 down c672 | MLP | a%100 in {51, 53} (-) | 1.00 | 0.020 |
| L23 down c627 | MLP | a%100 in {51} (-) | 1.00 | 0.010 |
| L23 down c90 | MLP | a%100 in {52, 54} (+) | 1.00 | 0.020 |
| L23 down c347 | MLP | a%100 in {55, 57..63, 65} (-) | 1.00 | 0.171 |
| L23 down c893 | MLP | a%100 in {62..63} (-) | 1.00 | 0.030 |
| L23 down c17 | MLP | a%100 in {65..69, 71} (-) | 1.00 | 0.140 |
| L23 down c16 | MLP | a%100 in {68..73} (-) | 1.00 | 0.120 |
| L23 down c193 | MLP | a%100 in {70} (+) | 1.00 | 0.010 |
| L23 down c468 | MLP | a%100 in {70} (+) | 1.00 | 0.010 |
| L23 down c157 | MLP | a%100 in {74..75} (-) | 1.00 | 0.111 |
| L23 down c273 | MLP | a%100 in {74} (+) | 1.00 | 0.040 |
| L23 down c1003 | MLP | a%100 in {80..92} (+) | 1.00 | 0.150 |
| L23 down c18 | MLP | a%100 in {81, 85, 87..88, 91} (+) | 0.99 | 0.051 |
| L23 down c940 | MLP | a%100 in {87..89, 94} (-) | 1.00 | 0.040 |
| L23 down c733 | MLP | a%100 in {9, 29, 31} (-) | 1.00 | 0.030 |
| L23 down c147 | MLP | a%100 in {97..99} (-) | 1.00 | 0.030 |
| L23 down c394 | MLP | a%100 in {98} (-) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2s-238</b> `a%100` @ `op` (sub) — block code; 4 comps, L22 L23; tells apart 8/100 classes (best member 5); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.36 |
| classes told apart: joint / best code / best member | 8 /  / 5 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.05 / 0.42 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.71 /  / 0.68 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 MLP |
| CKA(arrangement before, joint write) | 0.24 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.29 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (4:0.20 3:0.17 2:0.15) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.06 / 1.18 |
| best source position (CKA) | `a` (0.18) |

<details><summary>codes and components</summary>

**code 2s-238.0 (L22 L23): 4 comps, tells apart 8/100, coverage 0.36, overlap 1.00 (random 1.04, p 0.41)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 o c246 | H3 | a%100 in {47, 57} (+) | 1.00 | 0.060 |
| L22 o c352 | H3 | a%100 in {97} (-) | 1.00 | 0.010 |
| L23 o c3 | H2 | a%100 in {0..9, 17..23, 28, 30, 40, 45, 50..51, 60, 80, 83, 90..94, 98} (+) | 1.00 | 0.810 |
| L23 o c415 | H21 | a%100 in {85} (+) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2s-237</b> `a%100` @ `op` (sub) — tiling; 55 comps, L22 L23; tells apart 65/100 classes (best member 8); on add: 2a-185 (member overlap 0.02)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.83 |
| classes told apart: joint / best code / best member | 65 /  / 8 (of 100) |
| members whose removal merges classes | 0.35 |
| support overlap (1 = tiling) / random sets / p | 2.43 / 3.14 / 0.03 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.87 /  / 0.47 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.12 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 MLP |
| CKA(arrangement before, joint write) | 0.51 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.19 |
| share of the write inside the old arrangement's span | 0.19 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.29 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.19 2:0.10 3:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.40 / 11.41 |
| best source position (CKA) | `a` (0.48) |

<details><summary>codes and components</summary>

**code 2s-237.0 (L22 L23): 55 comps, tells apart 65/100, coverage 0.83, overlap 2.43 (random 3.08, p 0.01)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 o c36 | H15 | a%100 in {0, 80..81} (+) | 1.00 | 0.060 |
| L22 o c257 | H15 | a%100 in {0, 96..97} (-) | 1.00 | 0.070 |
| L22 o c56 | H15 | a%100 in {0..15, 23..36, 41..42, 46..48, 50..52, 56, 62, 71, 74, 76, 78..81, 87..88, 90..99} (+) | 1.00 | 0.581 |
| L22 o c135 | H15 | a%100 in {0} (-) | 1.00 | 0.030 |
| L22 o c200 | H15 | a%100 in {0} (-) | 1.00 | 0.090 |
| L22 o c372 | H15 | a%100 in {12..13} (+) | 1.00 | 0.030 |
| L22 o c226 | H15 | a%100 in {12..14, 93} (+) | 1.00 | 0.040 |
| L22 o c322 | H15 | a%100 in {12..14} (-) | 1.00 | 0.040 |
| L22 o c230 | H15 | a%100 in {13..15, 25, 35, 45, 55} (+) | 1.00 | 0.070 |
| L22 o c31 | H15 | a%100 in {13..20} (-) | 1.00 | 0.101 |
| L22 o c238 | H15 | a%100 in {17, 27} (-) | 1.00 | 0.020 |
| L22 o c498 | H15 | a%100 in {17, 37, 77, 97} (-) | 1.00 | 0.050 |
| L22 o c266 | H15 | a%100 in {18, 28, 88} (+) | 1.00 | 0.030 |
| L22 o c123 | H15 | a%100 in {18, 28} (+) | 1.00 | 0.020 |
| L22 o c181 | H15 | a%100 in {20, 22} (+) | 1.00 | 0.100 |
| L22 o c444 | H15 | a%100 in {20..24} (+) | 1.00 | 0.080 |
| L22 o c158 | H15 | a%100 in {20} (+) | 1.00 | 0.010 |
| L22 o c154 | H15 | a%100 in {20} (-) | 1.00 | 0.030 |
| L22 o c71 | H15 | a%100 in {21, 31, 51} (+) | 1.00 | 0.030 |
| L22 o c33 | H15 | a%100 in {21, 31, 81, 91} (-) | 1.00 | 0.040 |
| L22 o c376 | H15 | a%100 in {21, 41..42, 91} (-) | 1.00 | 0.040 |
| L22 o c88 | H15 | a%100 in {21..23} (-) | 1.00 | 0.040 |
| L22 o c188 | H15 | a%100 in {21} (+) | 1.00 | 0.010 |
| L22 o c224 | H15 | a%100 in {22} (-) | 1.00 | 0.010 |
| L22 o c187 | H15 | a%100 in {23, 25, 29} (+) | 1.00 | 0.030 |
| L22 o c320 | H15 | a%100 in {23..24, 26} (+) | 1.00 | 0.130 |
| L22 o c68 | H15 | a%100 in {24, 36, 96} (-) | 1.00 | 0.040 |
| L22 o c41 | H15 | a%100 in {24..30} (+) | 1.00 | 0.221 |
| L22 o c109 | H15 | a%100 in {24} (+) | 1.00 | 0.010 |
| L22 o c82 | H15 | a%100 in {27, 87} (-) | 1.00 | 0.070 |
| L22 o c253 | H15 | a%100 in {27..28} (-) | 1.00 | 0.030 |
| L22 o c445 | H15 | a%100 in {29} (-) | 1.00 | 0.010 |
| L22 o c263 | H15 | a%100 in {30..32} (+) | 1.00 | 0.040 |
| L22 o c205 | H15 | a%100 in {30} (+) | 1.00 | 0.010 |
| L22 o c95 | H15 | a%100 in {30} (-) | 1.00 | 0.010 |
| L22 o c309 | H15 | a%100 in {30} (-) | 1.00 | 0.010 |
| L22 o c259 | H15 | a%100 in {31, 91} (-) | 1.00 | 0.030 |
| L22 o c145 | H15 | a%100 in {35..37} (+) | 1.00 | 0.210 |
| L22 o c306 | H15 | a%100 in {40} (-) | 1.00 | 0.010 |
| L22 o c281 | H15 | a%100 in {42} (-) | 1.00 | 0.010 |
| L22 o c111 | H15 | a%100 in {45} (-) | 1.00 | 0.010 |
| L22 o c301 | H15 | a%100 in {48} (-) | 1.00 | 0.010 |
| L22 o c116 | H15 | a%100 in {50..53, 55..57} (-) | 1.00 | 0.090 |
| L22 o c55 | H15 | a%100 in {55..57} (+) | 1.00 | 0.030 |
| L22 o c161 | H15 | a%100 in {55} (-) | 1.00 | 0.010 |
| L22 o c499 | H15 | a%100 in {76} (+) | 1.00 | 0.010 |
| L22 o c32 | H15 | a%100 in {82..83, 85..88} (+) | 1.00 | 0.100 |
| L22 o c196 | H15 | a%100 in {89..93} (-) | 1.00 | 0.110 |
| L22 o c193 | H15 | a%100 in {9, 19, 29, 39, 89, 99} (-) | 1.00 | 0.072 |
| L22 o c397 | H15 | a%100 in {90} (-) | 1.00 | 0.010 |
| L22 o c436 | H15 | a%100 in {90} (-) | 1.00 | 0.010 |
| L22 o c119 | H15 | a%100 in {93} (-) | 1.00 | 0.010 |
| L22 o c48 | H15 | a%100 in {96} (-) | 1.00 | 0.020 |
| L22 o c434 | H15 | a%100 in {98} (+) | 1.00 | 0.010 |
| L23 o c36 | H14 | a%100 in {63..66} (-) | 1.00 | 0.050 |

</details>

</details>

<details><summary><b>2s-239</b> `a%100` @ `op` (sub) — block code; 16 comps, L23 L24; tells apart 11/100 classes (best member 4); on add: 2a-186 (member overlap 0.12)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.11 |
| classes told apart: joint / best code / best member | 11 /  / 4 (of 100) |
| members whose removal merges classes | 0.25 |
| support overlap (1 = tiling) / random sets / p | 1.64 / 1.44 / 0.82 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.14 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.78 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L22 MLP |
| CKA(arrangement before, joint write) | 0.05 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.21 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.29 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular () |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.02 / 2.10 |
| best source position (CKA) | `a` (0.07) |

<details><summary>codes and components</summary>

**code 2s-239.0 (L23 L24): 16 comps, tells apart 11/100, coverage 0.11, overlap 1.64 (random 1.43, p 0.82)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 o c394 | H22 | a%100 in {0, 25, 50} (-) | 1.00 | 0.030 |
| L23 o c320 | H7 | a%100 in {58} (+) | 1.00 | 0.010 |
| L23 o c474 | H7 | a%100 in {58} (+) | 1.00 | 0.010 |
| L23 o c435 | H22 | a%100 in {64} (-) | 1.00 | 0.010 |
| L23 o c94 | H7 | a%100 in {68} (+) | 1.00 | 0.020 |
| L23 o c470 | H7 | a%100 in {68} (-) | 1.00 | 0.020 |
| L23 o c323 | H22 | a%100 in {84} (-) | 1.00 | 0.010 |
| L23 o c315 | H7 | a%100 in {88} (+) | 1.00 | 0.020 |
| L23 o c204 | H7 | a%100 in {88} (-) | 1.00 | 0.010 |
| L23 o c280 | H7 | a%100 in {88} (-) | 1.00 | 0.030 |
| L23 o c353 | H7 | a%100 in {88} (-) | 1.00 | 0.010 |
| L23 o c410 | H22 | a%100 in {94} (+) | 1.00 | 0.010 |
| L23 o c187 | H7 | a%100 in {98} (-) | 1.00 | 0.020 |
| L23 o c318 | H7 | a%100 in {98} (-) | 1.00 | 0.010 |
| L23 o c333 | H7 | a%100 in {98} (-) | 1.00 | 0.030 |
| L24 o c469 | H19 | a%100 in {90} (+) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2s-241</b> `a%100` @ `op` (sub) — block code; 12 comps, L24; tells apart 16/100 classes (best member 9); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.12 |
| classes told apart: joint / best code / best member | 16 /  / 9 (of 100) |
| members whose removal merges classes | 0.33 |
| support overlap (1 = tiling) / random sets / p | 2.00 / 1.31 / 1.00 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.18 /  / 0.17 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.38 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L23 MLP |
| CKA(arrangement before, joint write) | 0.15 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.06 |
| share of the write inside the old arrangement's span | 0.16 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.28 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (2:0.14 1:0.13 3:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.06 / 2.10 |
| best source position (CKA) | `a` (0.11) |

<details><summary>codes and components</summary>

**code 2s-241.0 (L24): 12 comps, tells apart 16/100, coverage 0.12, overlap 2.00 (random 1.31, p 1.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 o c394 | H21 | a%100 in {23} (-) | 1.00 | 0.010 |
| L24 o c249 | H21 | a%100 in {24, 61..62} (-) | 1.00 | 0.030 |
| L24 o c480 | H21 | a%100 in {24, 64} (-) | 1.00 | 0.050 |
| L24 o c318 | H21 | a%100 in {61..63, 67..69} (+) | 1.00 | 0.170 |
| L24 o c338 | H21 | a%100 in {61} (+) | 1.00 | 0.010 |
| L24 o c364 | H21 | a%100 in {61} (+) | 1.00 | 0.010 |
| L24 o c241 | H21 | a%100 in {63, 73} (+) | 1.00 | 0.020 |
| L24 o c376 | H21 | a%100 in {63..64} (+) | 1.00 | 0.020 |
| L24 o c401 | H21 | a%100 in {63..65} (-) | 1.00 | 0.050 |
| L24 o c321 | H21 | a%100 in {63} (-) | 1.00 | 0.010 |
| L24 o c439 | H21 | a%100 in {68} (-) | 1.00 | 0.010 |
| L24 o c302 | H21 | a%100 in {72} (-) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2s-242</b> `a%100` @ `op` (sub) — block code; 2 comps, L25; tells apart 4/100 classes (best member 6); on add: 2a-187 (member overlap 0.50)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.20 |
| classes told apart: joint / best code / best member | 4 /  / 6 (of 100) |
| members whose removal merges classes | 0.50 |
| support overlap (1 = tiling) / random sets / p | 1.15 / 1.00 / 0.94 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.97 /  / 0.60 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.96 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L24 MLP |
| CKA(arrangement before, joint write) | 0.21 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.28 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.20 5:0.11 50:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.05 / 1.13 |
| best source position (CKA) | `a` (0.20) |

<details><summary>codes and components</summary>

**code 2s-242.0 (L25): 2 comps, tells apart 4/100, coverage 0.20, overlap 1.15 (random 1.00, p 0.95)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L25 o c1 | H23 | a%100 in {1..2, 24, 41, 55, 59, 61..62, 71, 73..75, 79, 81, 95, 97..99} (-) | 1.00 | 0.830 |
| L25 o c394 | H23 | a%100 in {1..3, 51, 98} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2s-243</b> `a%100` @ `op` (sub) — tiling, 4 codes of the same shape; 155 comps, L25 L27 L29 L30; tells apart 88/100 classes (best member 8); on add: 2a-165 (member overlap 0.01)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 88 / 79 / 8 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p | 8.37 / 8.24 / 0.59 |
| mean CKA between its codes | 0.72 |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 1.00 / 1.00 / 0.93 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.20 |
| consumers / read jointly | 38 / 32 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L25 attn |
| CKA(arrangement before, joint write) | 0.85 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.99 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.28 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.33 2:0.12 3:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.51 / 6.98 |

<details><summary>codes and components</summary>

**code 2s-243.0 (L25): 33 comps, tells apart 49/100, coverage 0.85, overlap 1.76 (random 2.14, p 0.07)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L25 down c69 | MLP | a%100 in {0, 20, 30} (-) | 1.00 | 0.040 |
| L25 down c97 | MLP | a%100 in {0, 20, 40, 50} (+) | 1.00 | 0.050 |
| L25 down c24 | MLP | a%100 in {0, 88} (+) | 1.00 | 0.030 |
| L25 down c56 | MLP | a%100 in {1, 70, 98} (+) | 1.00 | 0.990 |
| L25 down c34 | MLP | a%100 in {14, 16..18, 22, 24, 26..28, 30..39} (-) | 1.00 | 0.210 |
| L25 down c46 | MLP | a%100 in {17..18} (+) | 0.97 | 0.026 |
| L25 down c112 | MLP | a%100 in {17..24} (+) | 1.00 | 0.110 |
| L25 down c632 | MLP | a%100 in {1} (+) | 1.00 | 0.030 |
| L25 down c12 | MLP | a%100 in {20} (+) | 0.98 | 0.010 |
| L25 down c403 | MLP | a%100 in {22..31} (+) | 1.00 | 0.110 |
| L25 down c50 | MLP | a%100 in {22} (+) | 1.00 | 0.010 |
| L25 down c98 | MLP | a%100 in {23} (-) | 1.00 | 0.010 |
| L25 down c149 | MLP | a%100 in {24, 54} (+) | 0.99 | 0.021 |
| L25 down c39 | MLP | a%100 in {24} (+) | 1.00 | 0.010 |
| L25 down c125 | MLP | a%100 in {27..29, 31} (+) | 1.00 | 0.040 |
| L25 down c396 | MLP | a%100 in {39, 43, 47, 49, 51} (+) | 1.00 | 0.220 |
| L25 down c71 | MLP | a%100 in {49..50, 53, 55, 75} (-) | 1.00 | 0.050 |
| L25 down c20 | MLP | a%100 in {5, 7..12} (+) | 1.00 | 0.110 |
| L25 down c765 | MLP | a%100 in {51..53, 61} (-) | 1.00 | 0.070 |
| L25 down c514 | MLP | a%100 in {51} (+) | 1.00 | 0.030 |
| L25 down c16 | MLP | a%100 in {59..63, 65..69, 71..77, 79} (+) | 1.00 | 0.220 |
| L25 down c332 | MLP | a%100 in {62} (+) | 1.00 | 0.010 |
| L25 down c290 | MLP | a%100 in {64} (+) | 1.00 | 0.010 |
| L25 down c126 | MLP | a%100 in {70} (+) | 1.00 | 0.010 |
| L25 down c165 | MLP | a%100 in {70} (-) | 1.00 | 0.010 |
| L25 down c939 | MLP | a%100 in {74..75} (+) | 1.00 | 0.030 |
| L25 down c954 | MLP | a%100 in {74} (-) | 1.00 | 0.010 |
| L25 down c249 | MLP | a%100 in {76, 78..87} (-) | 1.00 | 0.150 |
| L25 down c66 | MLP | a%100 in {81} (+) | 1.00 | 0.040 |
| L25 down c343 | MLP | a%100 in {82..97, 99} (+) | 1.00 | 0.220 |
| L25 down c10 | MLP | a%100 in {89..92} (-) | 1.00 | 0.051 |
| L25 down c231 | MLP | a%100 in {93..99} (-) | 1.00 | 0.070 |
| L25 down c11 | MLP | a%100 in {96} (-) | 1.00 | 0.010 |

**code 2s-243.1 (L27): 36 comps, tells apart 59/100, coverage 0.96, overlap 2.41 (random 2.26, p 0.69)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 down c2 | MLP | a%100 in {0, 14..15, 18..20, 22, 24, 61, 71..73, 77, 81..97} (-) | 1.00 | 1.000 |
| L27 down c61 | MLP | a%100 in {0, 30, 40, 45, 50, 60} (-) | 1.00 | 0.070 |
| L27 down c394 | MLP | a%100 in {0} (-) | 1.00 | 0.070 |
| L27 down c3 | MLP | a%100 in {1..3, 6, 51, 70} (+) | 1.00 | 1.000 |
| L27 down c455 | MLP | a%100 in {1..3} (-) | 1.00 | 0.040 |
| L27 down c213 | MLP | a%100 in {12..29, 31..32} (-) | 1.00 | 0.310 |
| L27 down c706 | MLP | a%100 in {13..18} (+) | 1.00 | 0.060 |
| L27 down c64 | MLP | a%100 in {18..22} (+) | 0.98 | 0.076 |
| L27 down c167 | MLP | a%100 in {19} (+) | 1.00 | 0.010 |
| L27 down c306 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L27 down c99 | MLP | a%100 in {1} (-) | 1.00 | 0.020 |
| L27 down c694 | MLP | a%100 in {2..8} (-) | 1.00 | 0.070 |
| L27 down c119 | MLP | a%100 in {24, 34, 36, 44, 46, 48, 54, 56, 64, 66..68, 84} (+) | 1.00 | 0.140 |
| L27 down c425 | MLP | a%100 in {24} (-) | 1.00 | 0.010 |
| L27 down c689 | MLP | a%100 in {32..41, 43..44, 46} (+) | 1.00 | 0.200 |
| L27 down c254 | MLP | a%100 in {38} (+) | 1.00 | 0.010 |
| L27 down c168 | MLP | a%100 in {40, 42..44} (-) | 1.00 | 0.050 |
| L27 down c114 | MLP | a%100 in {43, 46..49, 51..55, 57..59, 61, 63, 67} (-) | 1.00 | 0.400 |
| L27 down c237 | MLP | a%100 in {46..49, 52} (+) | 1.00 | 0.050 |
| L27 down c149 | MLP | a%100 in {48, 50} (-) | 1.00 | 0.020 |
| L27 down c37 | MLP | a%100 in {51, 70, 90..95, 97, 99} (+) | 1.00 | 0.100 |
| L27 down c358 | MLP | a%100 in {51..53} (+) | 1.00 | 0.060 |
| L27 down c177 | MLP | a%100 in {51} (-) | 1.00 | 0.010 |
| L27 down c873 | MLP | a%100 in {52} (+) | 1.00 | 0.010 |
| L27 down c4 | MLP | a%100 in {60..61, 74..75} (-) | 1.00 | 0.040 |
| L27 down c30 | MLP | a%100 in {61, 74..77} (+) | 1.00 | 0.090 |
| L27 down c31 | MLP | a%100 in {65, 67..69, 71..87} (-) | 1.00 | 0.260 |
| L27 down c44 | MLP | a%100 in {70} (+) | 1.00 | 0.060 |
| L27 down c13 | MLP | a%100 in {74} (-) | 1.00 | 0.010 |
| L27 down c117 | MLP | a%100 in {76, 78..79} (-) | 1.00 | 0.030 |
| L27 down c10 | MLP | a%100 in {76..79, 81..82} (+) | 1.00 | 0.070 |
| L27 down c75 | MLP | a%100 in {80..81, 83..91} (+) | 1.00 | 0.110 |
| L27 down c33 | MLP | a%100 in {90..97} (-) | 1.00 | 0.168 |
| L27 down c604 | MLP | a%100 in {92..94, 96..98} (-) | 1.00 | 0.060 |
| L27 down c407 | MLP | a%100 in {93..99} (+) | 1.00 | 0.070 |
| L27 down c274 | MLP | a%100 in {99} (-) | 1.00 | 0.010 |

**code 2s-243.2 (L29): 54 comps, tells apart 79/100, coverage 0.99, overlap 2.46 (random 3.09, p 0.03)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L29 down c258 | MLP | a%100 in {0, 20, 25, 30, 40, 45, 50, 60, 80, 90} (+) | 1.00 | 0.114 |
| L29 down c113 | MLP | a%100 in {0, 70} (+) | 1.00 | 0.020 |
| L29 down c13 | MLP | a%100 in {1..2} (-) | 1.00 | 0.050 |
| L29 down c4 | MLP | a%100 in {1..3} (+) | 1.00 | 0.970 |
| L29 down c841 | MLP | a%100 in {1..8} (+) | 1.00 | 0.080 |
| L29 down c87 | MLP | a%100 in {10..21} (+) | 1.00 | 0.120 |
| L29 down c467 | MLP | a%100 in {13..14} (+) | 1.00 | 0.020 |
| L29 down c739 | MLP | a%100 in {13} (+) | 1.00 | 0.010 |
| L29 down c280 | MLP | a%100 in {15} (-) | 1.00 | 0.030 |
| L29 down c175 | MLP | a%100 in {16, 24..28, 30, 32, 34, 36, 51, 61, 70..71, 73..77, 79, 81..82, 85, 95, 97..99} (+) | 1.00 | 0.890 |
| L29 down c343 | MLP | a%100 in {16} (-) | 0.68 | 0.007 |
| L29 down c279 | MLP | a%100 in {19..20} (-) | 1.00 | 0.050 |
| L29 down c38 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L29 down c9 | MLP | a%100 in {2..5, 7, 15, 29, 31, 33, 37, 39, 42, 51, 70, 76, 91, 95, 97, 99} (+) | 1.00 | 1.000 |
| L29 down c316 | MLP | a%100 in {23..29} (-) | 1.00 | 0.081 |
| L29 down c94 | MLP | a%100 in {24} (-) | 1.00 | 0.010 |
| L29 down c214 | MLP | a%100 in {24} (-) | 1.00 | 0.010 |
| L29 down c178 | MLP | a%100 in {26..27} (-) | 1.00 | 0.020 |
| L29 down c123 | MLP | a%100 in {26..29, 31} (+) | 1.00 | 0.060 |
| L29 down c189 | MLP | a%100 in {29..31, 33} (+) | 1.00 | 0.091 |
| L29 down c853 | MLP | a%100 in {34..38} (+) | 1.00 | 0.050 |
| L29 down c566 | MLP | a%100 in {34} (+) | 0.72 | 0.011 |
| L29 down c805 | MLP | a%100 in {4..8, 24} (-) | 1.00 | 0.070 |
| L29 down c727 | MLP | a%100 in {41..44, 46..49} (-) | 1.00 | 0.080 |
| L29 down c308 | MLP | a%100 in {46..47, 74} (-) | 1.00 | 0.040 |
| L29 down c244 | MLP | a%100 in {47, 49, 51, 53} (+) | 1.00 | 0.120 |
| L29 down c964 | MLP | a%100 in {48} (-) | 1.00 | 0.010 |
| L29 down c267 | MLP | a%100 in {49, 51..52} (+) | 1.00 | 0.090 |
| L29 down c433 | MLP | a%100 in {52} (-) | 1.00 | 0.030 |
| L29 down c153 | MLP | a%100 in {54..57} (+) | 1.00 | 0.080 |
| L29 down c437 | MLP | a%100 in {58..59} (-) | 1.00 | 0.030 |
| L29 down c435 | MLP | a%100 in {59..61, 63..67} (+) | 1.00 | 0.080 |
| L29 down c461 | MLP | a%100 in {6..7} (+) | 1.00 | 0.030 |
| L29 down c331 | MLP | a%100 in {60..63} (-) | 1.00 | 0.040 |
| L29 down c166 | MLP | a%100 in {61..68} (+) | 1.00 | 0.110 |
| L29 down c90 | MLP | a%100 in {66} (+) | 1.00 | 0.010 |
| L29 down c11 | MLP | a%100 in {67, 69, 71, 73..79} (+) | 1.00 | 0.230 |
| L29 down c144 | MLP | a%100 in {68..69, 71..74} (+) | 1.00 | 0.090 |
| L29 down c134 | MLP | a%100 in {6} (+) | 1.00 | 0.010 |
| L29 down c832 | MLP | a%100 in {70} (-) | 1.00 | 0.010 |
| L29 down c146 | MLP | a%100 in {73..74} (-) | 1.00 | 0.040 |
| L29 down c225 | MLP | a%100 in {74} (+) | 1.00 | 0.030 |
| L29 down c150 | MLP | a%100 in {79..84} (+) | 1.00 | 0.100 |
| L29 down c759 | MLP | a%100 in {8..12} (-) | 1.00 | 0.060 |
| L29 down c450 | MLP | a%100 in {80..82, 90} (+) | 1.00 | 0.102 |
| L29 down c54 | MLP | a%100 in {81..84, 86} (+) | 1.00 | 0.060 |
| L29 down c19 | MLP | a%100 in {83..84} (+) | 1.00 | 0.020 |
| L29 down c149 | MLP | a%100 in {83..89} (+) | 1.00 | 0.150 |
| L29 down c172 | MLP | a%100 in {85} (-) | 1.00 | 0.010 |
| L29 down c725 | MLP | a%100 in {88..94} (+) | 1.00 | 0.110 |
| L29 down c213 | MLP | a%100 in {90..92} (-) | 1.00 | 0.060 |
| L29 down c226 | MLP | a%100 in {93..99} (-) | 1.00 | 0.200 |
| L29 down c71 | MLP | a%100 in {93} (+) | 1.00 | 0.030 |
| L29 down c208 | MLP | a%100 in {97..99} (+) | 1.00 | 0.030 |

**code 2s-243.3 (L30): 32 comps, tells apart 63/100, coverage 0.91, overlap 2.33 (random 2.06, p 0.82)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 down c666 | MLP | a%100 in {0, 70, 80, 90} (-) | 1.00 | 0.070 |
| L30 down c11 | MLP | a%100 in {1..2} (-) | 0.98 | 0.986 |
| L30 down c264 | MLP | a%100 in {1..2} (-) | 1.00 | 0.030 |
| L30 down c1 | MLP | a%100 in {1..3, 5, 7, 15, 17, 24, 66, 68..70, 73, 76, 78..79, 82..87, 89, 91..92, 94..95, 97..99} (-) | 1.00 | 1.000 |
| L30 down c60 | MLP | a%100 in {1..7, 9} (-) | 1.00 | 0.081 |
| L30 down c56 | MLP | a%100 in {1..8} (-) | 1.00 | 0.130 |
| L30 down c294 | MLP | a%100 in {17..18, 51, 53} (-) | 1.00 | 0.090 |
| L30 down c642 | MLP | a%100 in {17} (-) | 1.00 | 0.010 |
| L30 down c739 | MLP | a%100 in {18} (-) | 1.00 | 0.010 |
| L30 down c1004 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L30 down c266 | MLP | a%100 in {2..4, 7..8} (-) | 1.00 | 0.060 |
| L30 down c283 | MLP | a%100 in {20} (-) | 1.00 | 0.010 |
| L30 down c706 | MLP | a%100 in {21..26} (+) | 1.00 | 0.080 |
| L30 down c997 | MLP | a%100 in {21} (-) | 1.00 | 0.010 |
| L30 down c62 | MLP | a%100 in {22..31} (-) | 1.00 | 0.320 |
| L30 down c34 | MLP | a%100 in {3, 5..10} (-) | 1.00 | 0.090 |
| L30 down c145 | MLP | a%100 in {33..34, 36..39, 41, 43..44, 46..49, 54} (-) | 1.00 | 0.250 |
| L30 down c223 | MLP | a%100 in {38..39} (-) | 1.00 | 0.030 |
| L30 down c406 | MLP | a%100 in {41..44, 46, 64} (-) | 1.00 | 0.141 |
| L30 down c868 | MLP | a%100 in {51, 55, 57..61, 65..67} (-) | 1.00 | 0.100 |
| L30 down c1003 | MLP | a%100 in {61} (-) | 1.00 | 0.010 |
| L30 down c322 | MLP | a%100 in {68..69, 71, 74} (-) | 1.00 | 0.040 |
| L30 down c38 | MLP | a%100 in {69, 71, 73..77} (+) | 1.00 | 0.180 |
| L30 down c252 | MLP | a%100 in {7..19} (-) | 1.00 | 0.190 |
| L30 down c377 | MLP | a%100 in {70..71, 74..75, 99} (+) | 1.00 | 1.000 |
| L30 down c23 | MLP | a%100 in {71..73, 75..79, 81..89, 91..95, 97..99} (+) | 1.00 | 0.345 |
| L30 down c299 | MLP | a%100 in {74, 91..97} (+) | 0.97 | 0.083 |
| L30 down c21 | MLP | a%100 in {74..75, 97..99} (-) | 1.00 | 0.081 |
| L30 down c162 | MLP | a%100 in {78..79, 83, 85..92} (+) | 1.00 | 0.150 |
| L30 down c31 | MLP | a%100 in {80..81} (+) | 0.99 | 0.063 |
| L30 down c357 | MLP | a%100 in {84} (-) | 0.54 | 0.006 |
| L30 down c336 | MLP | a%100 in {93..99} (-) | 1.00 | 0.123 |

</details>

</details>

<details><summary><b>2s-244</b> `a%100` @ `op` (sub) — block code; 3 comps, L26 L27; tells apart 11/100 classes (best member 6); on add: 2a-188 (member overlap 0.25)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.08 |
| classes told apart: joint / best code / best member | 11 /  / 6 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.63 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.98 /  / 0.63 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L25 MLP |
| CKA(arrangement before, joint write) | 0.10 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.27 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (3:0.09 5:0.07 20:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.04 / 1.99 |
| best source position (CKA) | `a` (0.08) |

<details><summary>codes and components</summary>

**code 2s-244.0 (L26 L27): 3 comps, tells apart 11/100, coverage 0.08, overlap 1.00 (random 1.00, p 0.72)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 o c19 | H10 | a%100 in {0, 2, 10, 70} (-) | 1.00 | 1.000 |
| L27 o c269 | H1 | a%100 in {36, 56, 96} (+) | 1.00 | 0.060 |
| L27 o c29 | H31 | a%100 in {77} (+) | 1.00 | 0.990 |

</details>

</details>

<details><summary><b>2s-245</b> `a%100` @ `op` (sub) — block code; 32 comps, L26; tells apart 72/100 classes (best member 9); on add: 2a-165 (member overlap 0.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.79 |
| classes told apart: joint / best code / best member | 72 /  / 9 (of 100) |
| members whose removal merges classes | 0.50 |
| support overlap (1 = tiling) / random sets / p | 2.04 / 2.06 / 0.44 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 1.00 /  / 0.63 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.97 |
| consumers / read jointly | 7 / 3 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L26 attn |
| CKA(arrangement before, joint write) | 0.68 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.06 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.27 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.32 2:0.13 3:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.48 / 7.27 |

<details><summary>codes and components</summary>

**code 2s-245.0 (L26): 32 comps, tells apart 72/100, coverage 0.79, overlap 2.04 (random 2.05, p 0.47)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 down c109 | MLP | a%100 in {0, 50} (-) | 1.00 | 0.060 |
| L26 down c93 | MLP | a%100 in {1..2} (-) | 1.00 | 0.050 |
| L26 down c0 | MLP | a%100 in {1..3, 70, 99} (-) | 1.00 | 1.000 |
| L26 down c33 | MLP | a%100 in {14..16} (+) | 1.00 | 0.050 |
| L26 down c37 | MLP | a%100 in {16, 26, 36, 96} (+) | 1.00 | 0.040 |
| L26 down c902 | MLP | a%100 in {17..19} (-) | 1.00 | 0.030 |
| L26 down c8 | MLP | a%100 in {19..20} (-) | 1.00 | 0.040 |
| L26 down c609 | MLP | a%100 in {2..3, 6} (-) | 1.00 | 0.030 |
| L26 down c270 | MLP | a%100 in {22..29, 31} (+) | 1.00 | 0.140 |
| L26 down c65 | MLP | a%100 in {24} (+) | 1.00 | 0.020 |
| L26 down c16 | MLP | a%100 in {26, 28..30} (+) | 1.00 | 0.060 |
| L26 down c20 | MLP | a%100 in {28..31} (-) | 1.00 | 0.060 |
| L26 down c120 | MLP | a%100 in {30, 45, 60, 80, 90} (-) | 1.00 | 0.080 |
| L26 down c522 | MLP | a%100 in {39, 47, 49, 51, 53} (-) | 1.00 | 0.100 |
| L26 down c121 | MLP | a%100 in {41..44} (-) | 1.00 | 0.040 |
| L26 down c113 | MLP | a%100 in {42..45, 47..48} (+) | 1.00 | 0.060 |
| L26 down c268 | MLP | a%100 in {49, 51..55} (-) | 1.00 | 0.100 |
| L26 down c382 | MLP | a%100 in {49, 73..75, 77} (+) | 1.00 | 0.130 |
| L26 down c44 | MLP | a%100 in {51} (-) | 1.00 | 0.010 |
| L26 down c124 | MLP | a%100 in {55..59, 61..63, 65..67} (+) | 1.00 | 0.130 |
| L26 down c12 | MLP | a%100 in {56..57, 59} (-) | 1.00 | 0.040 |
| L26 down c94 | MLP | a%100 in {56} (+) | 1.00 | 0.010 |
| L26 down c83 | MLP | a%100 in {60..61, 65, 70, 75} (+) | 1.00 | 0.320 |
| L26 down c67 | MLP | a%100 in {61, 63, 65..69, 71..91} (-) | 1.00 | 0.372 |
| L26 down c89 | MLP | a%100 in {71..79} (-) | 1.00 | 0.090 |
| L26 down c103 | MLP | a%100 in {79..81} (+) | 1.00 | 0.030 |
| L26 down c161 | MLP | a%100 in {81..85} (-) | 1.00 | 0.110 |
| L26 down c90 | MLP | a%100 in {83..84} (-) | 1.00 | 0.020 |
| L26 down c46 | MLP | a%100 in {83..89} (-) | 1.00 | 0.140 |
| L26 down c19 | MLP | a%100 in {88..89, 92, 94} (-) | 1.00 | 0.040 |
| L26 down c320 | MLP | a%100 in {89..96} (+) | 1.00 | 0.080 |
| L26 down c367 | MLP | a%100 in {99} (-) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2s-246</b> `a%100` @ `op` (sub) — single component; 1 comps, L27; tells apart 2/100 classes (best member 2); on add: 2a-188 (member overlap 0.50)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.14 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.59 /  / 0.59 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L26 MLP |
| CKA(arrangement before, joint write) | 0.12 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.17 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.27 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (5:0.13 50:0.11 25:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.03 / 1.00 |
| best source position (CKA) | `a` (0.08) |

<details><summary>codes and components</summary>

**code 2s-246.0 (L27): 1 comps, tells apart 2/100, coverage 0.14, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 o c5 | H14 | a%100 in {5, 22, 24..26, 40, 47, 50..51, 53, 70, 74..75, 82} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2s-247</b> `a%100` @ `op` (sub) — single component; 1 comps, L28; tells apart 3/100 classes (best member 3); on add: 2a-189 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.24 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.51 /  / 0.51 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L27 MLP |
| CKA(arrangement before, joint write) | 0.16 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.17 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.27 2:0.11 3:0.07) |
| joint write: shape (spectrum k:share) | line (6:0.20 1:0.17 3:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.05 / 1.00 |
| best source position (CKA) | `a` (0.13) |

<details><summary>codes and components</summary>

**code 2s-247.0 (L28): 1 comps, tells apart 3/100, coverage 0.24, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 o c122 | H10 | a%100 in {0..2, 11, 29, 35..37, 46..51, 64, 74, 79, 91..92, 94..97, 99} (+) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2s-248</b> `a%100` @ `op` (sub) — single component; 1 comps, L28; tells apart 4/100 classes (best member 4); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.06 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.65 /  / 0.65 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L27 MLP |
| CKA(arrangement before, joint write) | 0.25 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.27 2:0.11 3:0.07) |
| joint write: shape (spectrum k:share) | line (2:0.24 3:0.14 6:0.12) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.05 / 1.00 |
| best source position (CKA) | `a` (0.15) |

<details><summary>codes and components</summary>

**code 2s-248.0 (L28): 1 comps, tells apart 4/100, coverage 0.06, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 o c63 | H7 | a%100 in {1..2, 23, 25..26, 70} (+) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>2s-249</b> `a%100` @ `op` (sub) — block code; 44 comps, L28; tells apart 73/100 classes (best member 8); on add: 2a-165 (member overlap 0.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.92 |
| classes told apart: joint / best code / best member | 73 /  / 8 (of 100) |
| members whose removal merges classes | 0.39 |
| support overlap (1 = tiling) / random sets / p | 2.24 / 2.65 / 0.08 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 1.00 /  / 0.69 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.97 |
| consumers / read jointly | 2 / 2 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L28 attn |
| CKA(arrangement before, joint write) | 0.75 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.07 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.27 2:0.11 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.19 2:0.15 3:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.37 / 8.50 |

<details><summary>codes and components</summary>

**code 2s-249.0 (L28): 44 comps, tells apart 73/100, coverage 0.92, overlap 2.24 (random 2.55, p 0.17)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 down c97 | MLP | a%100 in {0, 30, 40, 50, 60, 71, 73..77, 79, 98..99} (+) | 1.00 | 0.180 |
| L28 down c158 | MLP | a%100 in {1..2, 15, 24, 29..31, 49, 52, 75, 80..97} (+) | 1.00 | 1.000 |
| L28 down c847 | MLP | a%100 in {1..3} (+) | 1.00 | 0.090 |
| L28 down c5 | MLP | a%100 in {1..4, 51, 70, 74} (+) | 1.00 | 1.000 |
| L28 down c1 | MLP | a%100 in {1..9} (-) | 1.00 | 0.950 |
| L28 down c94 | MLP | a%100 in {10..12, 99} (-) | 1.00 | 0.040 |
| L28 down c792 | MLP | a%100 in {13..14, 21} (+) | 0.99 | 0.031 |
| L28 down c327 | MLP | a%100 in {15, 18..20} (+) | 1.00 | 0.090 |
| L28 down c51 | MLP | a%100 in {17} (-) | 1.00 | 0.010 |
| L28 down c236 | MLP | a%100 in {19..22} (+) | 1.00 | 0.110 |
| L28 down c360 | MLP | a%100 in {21, 29, 31} (-) | 1.00 | 0.040 |
| L28 down c271 | MLP | a%100 in {21} (-) | 0.90 | 0.009 |
| L28 down c79 | MLP | a%100 in {24} (-) | 1.00 | 0.010 |
| L28 down c98 | MLP | a%100 in {25..31, 33} (-) | 1.00 | 0.160 |
| L28 down c786 | MLP | a%100 in {26..29} (+) | 1.00 | 0.040 |
| L28 down c165 | MLP | a%100 in {27} (-) | 1.00 | 0.010 |
| L28 down c358 | MLP | a%100 in {30} (-) | 1.00 | 0.010 |
| L28 down c555 | MLP | a%100 in {32..33, 63} (-) | 1.00 | 0.030 |
| L28 down c858 | MLP | a%100 in {32..35, 39} (-) | 1.00 | 0.050 |
| L28 down c304 | MLP | a%100 in {32} (+) | 1.00 | 0.010 |
| L28 down c70 | MLP | a%100 in {35..36, 38, 40..42, 45} (-) | 1.00 | 0.110 |
| L28 down c844 | MLP | a%100 in {4, 6..9} (-) | 1.00 | 0.080 |
| L28 down c434 | MLP | a%100 in {42..48} (-) | 1.00 | 0.070 |
| L28 down c277 | MLP | a%100 in {48} (-) | 1.00 | 0.010 |
| L28 down c138 | MLP | a%100 in {49, 51, 53} (+) | 1.00 | 0.030 |
| L28 down c268 | MLP | a%100 in {50..51} (+) | 1.00 | 0.020 |
| L28 down c129 | MLP | a%100 in {51, 88} (-) | 1.00 | 0.030 |
| L28 down c350 | MLP | a%100 in {52} (-) | 1.00 | 0.020 |
| L28 down c170 | MLP | a%100 in {53, 55, 57..63, 65..69, 71, 73..75} (+) | 1.00 | 0.221 |
| L28 down c161 | MLP | a%100 in {6, 51, 53} (-) | 1.00 | 0.030 |
| L28 down c80 | MLP | a%100 in {66..69, 74} (+) | 1.00 | 0.060 |
| L28 down c38 | MLP | a%100 in {6} (-) | 1.00 | 0.010 |
| L28 down c92 | MLP | a%100 in {70} (+) | 1.00 | 0.040 |
| L28 down c74 | MLP | a%100 in {71, 73, 75, 79, 83, 85} (-) | 1.00 | 0.110 |
| L28 down c247 | MLP | a%100 in {74..75} (-) | 1.00 | 0.020 |
| L28 down c81 | MLP | a%100 in {81..87} (-) | 1.00 | 0.110 |
| L28 down c20 | MLP | a%100 in {88..89, 92..94} (+) | 1.00 | 0.050 |
| L28 down c41 | MLP | a%100 in {9, 11} (+) | 1.00 | 0.020 |
| L28 down c320 | MLP | a%100 in {90..92} (+) | 1.00 | 0.030 |
| L28 down c433 | MLP | a%100 in {91..94} (+) | 1.00 | 0.040 |
| L28 down c386 | MLP | a%100 in {92..99} (+) | 1.00 | 0.150 |
| L28 down c22 | MLP | a%100 in {93..96} (+) | 1.00 | 0.050 |
| L28 down c126 | MLP | a%100 in {96..98} (+) | 1.00 | 0.030 |
| L28 down c468 | MLP | a%100 in {9} (+) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2s-250</b> `a%100` @ `op` (sub) — block code; 2 comps, L29 L30; tells apart 5/100 classes (best member 7); on add: 2a-190 (member overlap 0.33)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.12 |
| classes told apart: joint / best code / best member | 5 /  / 7 (of 100) |
| members whose removal merges classes | 0.50 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.83 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.25 /  / 0.24 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L28 MLP |
| CKA(arrangement before, joint write) | 0.10 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.26 2:0.11 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (3:0.06 50:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.03 / 1.97 |
| best source position (CKA) | `a` (0.08) |

<details><summary>codes and components</summary>

**code 2s-250.0 (L29 L30): 2 comps, tells apart 5/100, coverage 0.12, overlap 1.00 (random 1.00, p 0.88)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L29 o c1 | H25 | a%100 in {0..1, 8, 24, 30, 50, 60, 70, 81, 91, 97} (+) | 1.00 | 1.000 |
| L30 o c83 | H10 | a%100 in {80} (-) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>2s-251</b> `a%100` @ `op` (sub) — single component; 1 comps, L30; tells apart 3/100 classes (best member 3); on add: 2a-190 (member overlap 0.50)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.20 |
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
| input point | L29 MLP |
| CKA(arrangement before, joint write) | 0.16 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.26 2:0.11 3:0.07) |
| joint write: shape (spectrum k:share) | line (4:0.21 50:0.11 1:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.05 / 1.00 |
| best source position (CKA) | `a` (0.17) |

<details><summary>codes and components</summary>

**code 2s-251.0 (L30): 1 comps, tells apart 3/100, coverage 0.20, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 o c446 | H18 | a%100 in {1, 4, 8, 11, 14, 19, 28, 32, 43, 45, 47, 49, 58, 61, 68, 72..73, 75, 91, 99} (-) | 1.00 | 1.000 |

</details>

</details>

</details>

<details><summary>`a//10`: 5 mechanisms, 208 components</summary>

<details><summary><b>2s-259</b> `a//10` @ `op` (sub) — block code, 3 codes of the same shape; 10 comps, L0 L1 L3 L22; tells apart 4/11 classes (best member 4); on add: 2a-193 (member overlap 0.27)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.45 |
| classes told apart: joint / best code / best member | 4 / 5 / 4 (of 11) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 3.00 / 2.29 / 0.95 |
| mean CKA between its codes | 0.79 |
| purity of the joint write (per prompt) | 0.91 |
| decoding acc. joint / best code / best member (chance) | 0.74 / 0.81 / 0.71 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.02 |
| consumers / read jointly | 7 / 4 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | embed |
| CKA(arrangement before, joint write) | 0.00 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.00 |
| write energy / code energy before | 300935423597693463980301877248.00 |
| best source position (CKA) | `a` (0.40) |

<details><summary>codes and components</summary>

**code 2s-259.0 (L0 L1): 2 comps, tells apart 5/11, coverage 0.45, overlap 1.00 (random 1.00, p 0.60)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 o c7 | H0 | a//10 in {0..1, 10} (-) | 0.97 | 1.000 |
| L1 o c465 | H24 | a//10 in {8..9} (+) | 0.93 | 0.201 |

**code 2s-259.1 (L3): 5 comps, tells apart 4/11, coverage 0.27, overlap 2.00 (random 1.50, p 0.93)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 o c77 | H7 | a//10 in {10} (+) | 1.00 | 0.010 |
| L3 o c257 | H7 | a//10 in {10} (+) | 1.00 | 0.010 |
| L3 o c104 | H7 | a//10 in {10} (-) | 0.97 | 0.021 |
| L3 o c174 | H7 | a//10 in {10} (-) | 1.00 | 0.010 |
| L3 o c243 | H15 | a//10 in {8..9} (-) | 0.94 | 0.230 |

**code 2s-259.2 (L22): 3 comps, tells apart 4/11, coverage 0.18, overlap 2.00 (random 1.33, p 0.96)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 o c62 | H15 | a//10 in {10} (+) | 1.00 | 0.010 |
| L22 o c302 | H15 | a//10 in {10} (+) | 1.00 | 0.010 |
| L22 o c10 | H15 | a//10 in {9..10} (-) | 0.90 | 0.200 |

</details>

</details>

<details><summary><b>2s-257</b> `a//10` @ `op` (sub) — block code, 7 codes of the same shape, copy from `a`; 20 comps, L0 L24 L25 L27 L29 L30; tells apart 7/11 classes (best member 6); on add: 2a-194 (member overlap 0.03)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.64 |
| classes told apart: joint / best code / best member | 7 / 5 / 6 (of 11) |
| members whose removal merges classes | 0.10 |
| support overlap (1 = tiling) / random sets / p | 4.14 / 3.78 / 0.77 |
| mean CKA between its codes | 0.87 |
| purity of the joint write (per prompt) | 0.95 |
| decoding acc. joint / best code / best member (chance) | 0.80 / 0.66 / 0.58 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.08 |
| consumers / read jointly | 20 / 15 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | embed |
| CKA(arrangement before, joint write) | 0.00 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.00 |
| write energy / code energy before | 26428712425231930312040581693440.00 |
| best source position (CKA) | `a` (0.56) |

<details><summary>codes and components</summary>

**code 2s-257.0 (L0): 2 comps, tells apart 3/11, coverage 0.09, overlap 2.00 (random 1.00, p 1.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 o c1 | H2 | a//10 in {0} (+) | 0.96 | 1.000 |
| L0 o c2 | H2 | a//10 in {0} (-) | 0.93 | 1.000 |

**code 2s-257.1 (L0): 3 comps, tells apart 5/11, coverage 0.27, overlap 1.67 (random 1.33, p 0.91)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 down c21 | MLP | a//10 in {0..1, 10} (-) | 0.95 | 1.000 |
| L0 down c325 | MLP | a//10 in {0} (+) | 0.98 | 0.090 |
| L0 down c12 | MLP | a//10 in {0} (-) | 0.90 | 1.000 |

**code 2s-257.2 (L24): 5 comps, tells apart 5/11, coverage 0.36, overlap 1.50 (random 1.50, p 0.53)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 down c266 | MLP | a//10 in {0..1} (-) | 0.93 | 0.250 |
| L24 down c65 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L24 down c21 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L24 down c76 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L24 down c144 | MLP | a//10 in {9} (+) | 0.90 | 0.110 |

**code 2s-257.3 (L25): 4 comps, tells apart 5/11, coverage 0.27, overlap 1.67 (random 1.33, p 0.81)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L25 down c416 | MLP | a//10 in {0..1} (-) | 0.95 | 0.325 |
| L25 down c244 | MLP | a//10 in {0} (+) | 0.95 | 0.090 |
| L25 down c346 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L25 down c468 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |

**code 2s-257.4 (L27): 1 comps, tells apart 4/11, coverage 0.18, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 down c15 | MLP | a//10 in {0..1} (-) | 0.96 | 0.300 |

**code 2s-257.5 (L29): 4 comps, tells apart 5/11, coverage 0.55, overlap 1.33 (random 1.40, p 0.45)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L29 down c43 | MLP | a//10 in {0..3, 7} (+) | 0.94 | 0.800 |
| L29 down c109 | MLP | a//10 in {0} (-) | 0.96 | 0.170 |
| L29 down c135 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L29 down c465 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |

**code 2s-257.6 (L30): 1 comps, tells apart 3/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 o c137 | H20 | a//10 in {0} (+) | 0.95 | 0.190 |

</details>

</details>

<details><summary><b>2s-258</b> `a//10` @ `op` (sub) — tiling, 30 codes of the same shape, copy from `a`; 172 comps, L1 L2 L3 L4 L5 L6 L7 L8 L9 L10 L11 L12 L13 L14 L15 L16 L17 L18 L19 L20 L21 L22 L23 L26 L28 L30; tells apart 10/11 classes (best member 6); on add: 2a-195 (member overlap 0.06)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 10 / 11 / 6 (of 11) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p | 29.64 / 28.27 / 0.99 |
| mean CKA between its codes | 0.79 |
| purity of the joint write (per prompt) | 0.94 |
| decoding acc. joint / best code / best member (chance) | 0.94 / 0.97 / 0.66 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 0.90 |
| consumers / read jointly | 754 / 534 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.91 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.00 |
| write energy / code energy before | 6183.42 |
| best source position (CKA) | `a` (0.90) |

<details><summary>codes and components</summary>

**code 2s-258.0 (L1 L2): 3 comps, tells apart 9/11, coverage 0.55, overlap 1.33 (random 1.25, p 0.66)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c359 | H18 | a//10 in {0..2} (+) | 0.97 | 0.840 |
| L2 o c48 | H24 | a//10 in {7..9} (+) | 0.94 | 0.580 |
| L2 o c20 | H24 | a//10 in {8..9} (+) | 0.93 | 0.201 |

**code 2s-258.1 (L1): 8 comps, tells apart 11/11, coverage 0.82, overlap 2.00 (random 2.00, p 0.61)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c413 | H26 | a//10 in {0..2} (+) | 0.94 | 0.380 |
| L1 o c377 | H26 | a//10 in {0} (-) | 0.94 | 0.180 |
| L1 o c315 | H26 | a//10 in {10} (+) | 1.00 | 0.010 |
| L1 o c159 | H26 | a//10 in {10} (-) | 1.00 | 0.010 |
| L1 o c231 | H26 | a//10 in {2..4, 7..10} (+) | 0.97 | 0.790 |
| L1 o c58 | H26 | a//10 in {2} (+) | 0.91 | 0.120 |
| L1 o c70 | H26 | a//10 in {7..9} (+) | 0.97 | 0.550 |
| L1 o c330 | H26 | a//10 in {9} (-) | 0.91 | 0.200 |

**code 2s-258.2 (L1): 4 comps, tells apart 6/11, coverage 0.64, overlap 1.43 (random 1.39, p 0.57)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 down c24 | MLP | a//10 in {0, 9} (+) | 0.94 | 1.000 |
| L1 down c85 | MLP | a//10 in {0..2} (-) | 0.95 | 0.373 |
| L1 down c49 | MLP | a//10 in {0} (-) | 0.97 | 0.190 |
| L1 down c94 | MLP | a//10 in {7..10} (+) | 0.97 | 0.890 |

**code 2s-258.3 (L2): 6 comps, tells apart 9/11, coverage 0.64, overlap 1.86 (random 1.75, p 0.65)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 down c73 | MLP | a//10 in {0..1, 10} (+) | 0.94 | 1.000 |
| L2 down c278 | MLP | a//10 in {0..1} (-) | 0.98 | 0.691 |
| L2 down c210 | MLP | a//10 in {0} (+) | 0.91 | 0.090 |
| L2 down c128 | MLP | a//10 in {1..3} (+) | 0.96 | 0.400 |
| L2 down c147 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L2 down c44 | MLP | a//10 in {8..10} (+) | 0.97 | 0.354 |

**code 2s-258.4 (L3): 8 comps, tells apart 11/11, coverage 1.00, overlap 1.55 (random 2.00, p 0.03)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 down c910 | MLP | a//10 in {0..1} (-) | 0.97 | 0.310 |
| L3 down c74 | MLP | a//10 in {1..3} (+) | 0.95 | 0.430 |
| L3 down c373 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L3 down c84 | MLP | a//10 in {4..6} (+) | 0.90 | 0.370 |
| L3 down c6 | MLP | a//10 in {6..9} (+) | 0.96 | 0.630 |
| L3 down c125 | MLP | a//10 in {7} (-) | 0.93 | 0.110 |
| L3 down c39 | MLP | a//10 in {8..9} (-) | 0.95 | 0.220 |
| L3 down c2 | MLP | a//10 in {9} (-) | 0.94 | 0.291 |

**code 2s-258.5 (L3): 2 comps, tells apart 8/11, coverage 0.82, overlap 1.67 (random 1.00, p 0.86)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 o c128 | H5 | a//10 in {0..2, 5..8} (-) | 0.91 | 1.000 |
| L3 o c354 | H5 | a//10 in {0..4, 6..8} (-) | 0.95 | 0.871 |

**code 2s-258.6 (L4): 11 comps, tells apart 9/11, coverage 0.73, overlap 2.38 (random 2.43, p 0.48)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 down c11 | MLP | a//10 in {0..1, 6} (+) | 0.97 | 0.910 |
| L4 down c62 | MLP | a//10 in {0..2} (-) | 0.96 | 0.380 |
| L4 down c264 | MLP | a//10 in {0} (+) | 0.96 | 0.166 |
| L4 down c597 | MLP | a//10 in {1..2} (-) | 0.93 | 0.340 |
| L4 down c164 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L4 down c204 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L4 down c300 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L4 down c77 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L4 down c253 | MLP | a//10 in {2..4} (-) | 0.90 | 0.310 |
| L4 down c128 | MLP | a//10 in {9..10} (-) | 0.96 | 0.200 |
| L4 down c145 | MLP | a//10 in {9} (+) | 0.96 | 0.230 |

**code 2s-258.7 (L5): 10 comps, tells apart 11/11, coverage 1.00, overlap 1.82 (random 2.37, p 0.04)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 down c11 | MLP | a//10 in {0, 3..5, 9..10} (+) | 0.94 | 1.000 |
| L5 down c425 | MLP | a//10 in {0} (-) | 0.96 | 0.241 |
| L5 down c120 | MLP | a//10 in {1..2} (-) | 0.93 | 0.320 |
| L5 down c28 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L5 down c614 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L5 down c798 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L5 down c936 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L5 down c76 | MLP | a//10 in {6..9} (-) | 0.95 | 0.480 |
| L5 down c59 | MLP | a//10 in {8..9} (-) | 0.95 | 0.268 |
| L5 down c339 | MLP | a//10 in {9} (+) | 0.92 | 0.159 |

**code 2s-258.8 (L6): 8 comps, tells apart 10/11, coverage 0.91, overlap 1.60 (random 2.00, p 0.07)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L6 down c24 | MLP | a//10 in {0, 3..6, 10} (+) | 0.90 | 1.000 |
| L6 down c217 | MLP | a//10 in {0} (+) | 0.98 | 0.091 |
| L6 down c642 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L6 down c428 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L6 down c6 | MLP | a//10 in {1} (-) | 0.91 | 0.340 |
| L6 down c79 | MLP | a//10 in {6..7} (-) | 0.93 | 0.370 |
| L6 down c117 | MLP | a//10 in {7..9} (-) | 0.93 | 0.348 |
| L6 down c110 | MLP | a//10 in {9} (+) | 0.91 | 0.131 |

**code 2s-258.9 (L7): 5 comps, tells apart 8/11, coverage 0.45, overlap 1.20 (random 1.60, p 0.09)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L7 down c230 | MLP | a//10 in {0} (+) | 0.95 | 0.160 |
| L7 down c183 | MLP | a//10 in {1..2} (-) | 0.93 | 0.383 |
| L7 down c109 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L7 down c880 | MLP | a//10 in {1} (-) | 0.92 | 0.170 |
| L7 down c267 | MLP | a//10 in {9} (+) | 0.95 | 0.334 |

**code 2s-258.10 (L8): 6 comps, tells apart 7/11, coverage 0.55, overlap 1.33 (random 1.74, p 0.07)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L8 down c280 | MLP | a//10 in {0} (+) | 0.97 | 0.174 |
| L8 down c225 | MLP | a//10 in {1..2} (-) | 0.94 | 0.421 |
| L8 down c648 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L8 down c708 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L8 down c226 | MLP | a//10 in {8..9} (-) | 0.94 | 0.218 |
| L8 down c215 | MLP | a//10 in {9} (-) | 0.91 | 0.150 |

**code 2s-258.11 (L9): 6 comps, tells apart 7/11, coverage 0.45, overlap 1.80 (random 1.71, p 0.64)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L9 down c38 | MLP | a//10 in {0, 10} (-) | 0.99 | 0.100 |
| L9 down c13 | MLP | a//10 in {0..2} (-) | 0.96 | 0.310 |
| L9 down c903 | MLP | a//10 in {0} (+) | 0.99 | 0.090 |
| L9 down c609 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L9 down c813 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L9 down c710 | MLP | a//10 in {9} (-) | 0.95 | 0.200 |

**code 2s-258.12 (L10): 5 comps, tells apart 9/11, coverage 0.91, overlap 1.50 (random 1.50, p 0.51)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L10 down c14 | MLP | a//10 in {0..1, 3..6, 9..10} (+) | 0.91 | 0.800 |
| L10 down c279 | MLP | a//10 in {0..2} (+) | 0.98 | 0.400 |
| L10 down c97 | MLP | a//10 in {0} (-) | 0.98 | 0.090 |
| L10 down c392 | MLP | a//10 in {1} (+) | 0.92 | 0.100 |
| L10 down c28 | MLP | a//10 in {8..9} (-) | 0.95 | 0.190 |

**code 2s-258.13 (L11): 4 comps, tells apart 7/11, coverage 0.64, overlap 1.00 (random 1.40, p 0.04)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L11 down c25 | MLP | a//10 in {0} (-) | 0.97 | 0.090 |
| L11 down c611 | MLP | a//10 in {1..2} (-) | 0.92 | 0.220 |
| L11 down c45 | MLP | a//10 in {6..8} (+) | 0.90 | 0.360 |
| L11 down c83 | MLP | a//10 in {9} (+) | 0.94 | 0.210 |

**code 2s-258.14 (L12): 7 comps, tells apart 9/11, coverage 0.82, overlap 1.67 (random 2.00, p 0.21)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L12 down c286 | MLP | a//10 in {0..1, 5..6, 9} (-) | 0.90 | 0.730 |
| L12 down c703 | MLP | a//10 in {0..3} (-) | 0.96 | 0.503 |
| L12 down c625 | MLP | a//10 in {0} (-) | 0.96 | 0.100 |
| L12 down c123 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L12 down c258 | MLP | a//10 in {1} (+) | 0.99 | 0.100 |
| L12 down c163 | MLP | a//10 in {8..9} (-) | 0.93 | 0.290 |
| L12 down c136 | MLP | a//10 in {9} (-) | 0.95 | 0.190 |

**code 2s-258.15 (L13): 4 comps, tells apart 7/11, coverage 0.45, overlap 1.20 (random 1.40, p 0.25)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 down c18 | MLP | a//10 in {0..2} (-) | 0.97 | 0.310 |
| L13 down c338 | MLP | a//10 in {0} (-) | 0.96 | 0.170 |
| L13 down c161 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L13 down c865 | MLP | a//10 in {9} (-) | 0.94 | 0.210 |

**code 2s-258.16 (L14): 5 comps, tells apart 8/11, coverage 0.55, overlap 1.50 (random 1.53, p 0.50)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L14 down c80 | MLP | a//10 in {0..1} (-) | 0.97 | 0.310 |
| L14 down c8 | MLP | a//10 in {0..2} (-) | 0.94 | 0.333 |
| L14 down c151 | MLP | a//10 in {0} (+) | 0.93 | 0.090 |
| L14 down c86 | MLP | a//10 in {10} (+) | 0.91 | 0.020 |
| L14 down c411 | MLP | a//10 in {8..9} (-) | 0.95 | 0.264 |

**code 2s-258.17 (L15): 7 comps, tells apart 8/11, coverage 0.55, overlap 1.67 (random 1.88, p 0.23)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c7 | MLP | a//10 in {0..1} (-) | 0.92 | 1.000 |
| L15 down c48 | MLP | a//10 in {0} (+) | 0.97 | 0.090 |
| L15 down c418 | MLP | a//10 in {0} (-) | 0.96 | 0.160 |
| L15 down c27 | MLP | a//10 in {1..2} (+) | 0.95 | 0.200 |
| L15 down c398 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L15 down c31 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L15 down c727 | MLP | a//10 in {8..9} (-) | 0.96 | 0.290 |

**code 2s-258.18 (L16): 4 comps, tells apart 7/11, coverage 0.45, overlap 1.40 (random 1.41, p 0.50)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c61 | H22 | a//10 in {0} (+) | 0.95 | 0.110 |
| L16 o c249 | H22 | a//10 in {10} (-) | 1.00 | 0.010 |
| L16 o c160 | H22 | a//10 in {7..10} (-) | 0.91 | 0.520 |
| L16 o c243 | H22 | a//10 in {8} (-) | 0.91 | 0.120 |

**code 2s-258.19 (L16): 7 comps, tells apart 9/11, coverage 0.91, overlap 1.70 (random 1.83, p 0.32)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c58 | MLP | a//10 in {0..2} (+) | 0.97 | 0.320 |
| L16 down c5 | MLP | a//10 in {0..3, 5..7} (-) | 0.96 | 0.680 |
| L16 down c54 | MLP | a//10 in {0} (-) | 0.95 | 0.140 |
| L16 down c631 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L16 down c38 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L16 down c160 | MLP | a//10 in {8..9} (-) | 0.93 | 0.213 |
| L16 down c9 | MLP | a//10 in {9..10} (-) | 0.93 | 0.210 |

**code 2s-258.20 (L17): 4 comps, tells apart 8/11, coverage 0.36, overlap 1.25 (random 1.50, p 0.22)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 down c367 | MLP | a//10 in {0} (-) | 0.96 | 0.300 |
| L17 down c110 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L17 down c71 | MLP | a//10 in {8..9} (+) | 0.93 | 0.460 |
| L17 down c537 | MLP | a//10 in {9} (-) | 0.91 | 0.109 |

**code 2s-258.21 (L18): 7 comps, tells apart 8/11, coverage 0.55, overlap 1.67 (random 1.86, p 0.27)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 down c122 | MLP | a//10 in {0..1} (-) | 0.96 | 0.330 |
| L18 down c55 | MLP | a//10 in {0} (+) | 0.95 | 0.170 |
| L18 down c1 | MLP | a//10 in {0} (-) | 0.93 | 0.910 |
| L18 down c38 | MLP | a//10 in {1..3} (+) | 0.93 | 0.300 |
| L18 down c66 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L18 down c507 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L18 down c114 | MLP | a//10 in {9} (-) | 0.92 | 0.180 |

**code 2s-258.22 (L19): 6 comps, tells apart 7/11, coverage 0.45, overlap 1.80 (random 1.73, p 0.61)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L19 down c116 | MLP | a//10 in {0..1} (-) | 0.91 | 0.241 |
| L19 down c25 | MLP | a//10 in {0} (+) | 0.95 | 0.100 |
| L19 down c699 | MLP | a//10 in {0} (+) | 0.98 | 0.090 |
| L19 down c69 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L19 down c14 | MLP | a//10 in {8..9} (-) | 0.95 | 0.256 |
| L19 down c10 | MLP | a//10 in {9..10} (+) | 0.94 | 0.210 |

**code 2s-258.23 (L20): 5 comps, tells apart 8/11, coverage 0.45, overlap 1.40 (random 1.55, p 0.30)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 down c0 | MLP | a//10 in {0..1} (-) | 0.92 | 0.870 |
| L20 down c156 | MLP | a//10 in {0} (+) | 0.97 | 0.180 |
| L20 down c16 | MLP | a//10 in {1..2} (+) | 0.94 | 0.340 |
| L20 down c248 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L20 down c67 | MLP | a//10 in {9} (-) | 0.91 | 0.244 |

**code 2s-258.24 (L21): 6 comps, tells apart 8/11, coverage 0.73, overlap 1.50 (random 1.67, p 0.27)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c11 | MLP | a//10 in {0..1} (+) | 0.92 | 0.800 |
| L21 down c23 | MLP | a//10 in {0..4} (+) | 0.96 | 0.571 |
| L21 down c271 | MLP | a//10 in {0} (+) | 0.96 | 0.100 |
| L21 down c188 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L21 down c55 | MLP | a//10 in {10} (-) | 0.97 | 0.020 |
| L21 down c88 | MLP | a//10 in {8..9} (+) | 0.96 | 0.230 |

**code 2s-258.25 (L22): 4 comps, tells apart 8/11, coverage 0.45, overlap 1.20 (random 1.40, p 0.17)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 down c937 | MLP | a//10 in {0..1} (-) | 0.97 | 0.450 |
| L22 down c96 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L22 down c132 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L22 down c311 | MLP | a//10 in {8..9} (-) | 0.97 | 0.220 |

**code 2s-258.26 (L23): 7 comps, tells apart 8/11, coverage 0.73, overlap 1.38 (random 1.86, p 0.03)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 down c50 | MLP | a//10 in {0..1} (-) | 0.94 | 0.420 |
| L23 down c39 | MLP | a//10 in {0} (+) | 0.95 | 0.110 |
| L23 down c72 | MLP | a//10 in {1..2} (+) | 0.92 | 0.220 |
| L23 down c6 | MLP | a//10 in {10} (-) | 0.93 | 0.020 |
| L23 down c26 | MLP | a//10 in {3..5} (-) | 0.91 | 0.381 |
| L23 down c3 | MLP | a//10 in {9} (+) | 0.94 | 0.140 |
| L23 down c400 | MLP | a//10 in {9} (+) | 0.93 | 0.111 |

**code 2s-258.27 (L26): 5 comps, tells apart 6/11, coverage 0.45, overlap 1.20 (random 1.60, p 0.06)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 down c30 | MLP | a//10 in {0} (-) | 0.93 | 0.180 |
| L26 down c39 | MLP | a//10 in {1..2} (-) | 0.92 | 0.220 |
| L26 down c139 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L26 down c330 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L26 down c15 | MLP | a//10 in {9} (+) | 0.92 | 0.131 |

**code 2s-258.28 (L28): 6 comps, tells apart 7/11, coverage 0.64, overlap 1.43 (random 1.71, p 0.12)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 down c445 | MLP | a//10 in {0} (+) | 0.94 | 0.180 |
| L28 down c37 | MLP | a//10 in {1..3} (-) | 0.95 | 0.330 |
| L28 down c450 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L28 down c0 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L28 down c915 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L28 down c28 | MLP | a//10 in {3..5} (-) | 0.93 | 0.320 |

**code 2s-258.29 (L30): 2 comps, tells apart 6/11, coverage 0.45, overlap 1.00 (random 1.00, p 0.68)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 down c7 | MLP | a//10 in {0..3} (-) | 0.93 | 0.570 |
| L30 down c32 | MLP | a//10 in {10} (+) | 0.96 | 0.010 |

</details>

</details>

<details><summary><b>2s-260</b> `a//10` @ `op` (sub) — block code; 2 comps, L14; tells apart 3/11 classes (best member 2); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.18 |
| classes told apart: joint / best code / best member | 3 /  / 2 (of 11) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.59 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.97 |
| decoding acc. joint / best code / best member (chance) | 0.28 /  / 0.19 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 0.99 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L13 MLP |
| CKA(arrangement before, joint write) | 0.24 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.00 |
| best source position (CKA) | `a` (0.30) |

<details><summary>codes and components</summary>

**code 2s-260.0 (L14): 2 comps, tells apart 3/11, coverage 0.18, overlap 1.00 (random 1.00, p 0.59)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L14 o c101 | H18 | a//10 in {10} (-) | 1.00 | 0.010 |
| L14 o c296 | H18 | a//10 in {8} (-) | 0.97 | 0.111 |

</details>

</details>

<details><summary><b>2s-261</b> `a//10` @ `op` (sub) — block code, 2 codes of the same shape; 4 comps, L16 L18; tells apart 2/11 classes (best member 2); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.09 |
| classes told apart: joint / best code / best member | 2 / 2 / 2 (of 11) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 4.00 / 1.40 / 1.00 |
| mean CKA between its codes | 1.00 |
| purity of the joint write (per prompt) | 0.98 |
| decoding acc. joint / best code / best member (chance) | 0.19 / 0.19 / 0.19 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.57 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 MLP |
| CKA(arrangement before, joint write) | 0.13 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 0.00 |
| best source position (CKA) | `a` (0.13) |

<details><summary>codes and components</summary>

**code 2s-261.0 (L16): 2 comps, tells apart 2/11, coverage 0.09, overlap 2.00 (random 1.00, p 1.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c35 | H21 | a//10 in {10} (+) | 1.00 | 0.010 |
| L16 o c89 | H21 | a//10 in {10} (-) | 1.00 | 0.010 |

**code 2s-261.1 (L18): 2 comps, tells apart 2/11, coverage 0.09, overlap 2.00 (random 1.00, p 1.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 o c120 | H30 | a//10 in {10} (+) | 0.94 | 0.020 |
| L18 o c157 | H30 | a//10 in {10} (+) | 1.00 | 0.010 |

</details>

</details>

</details>

<details><summary>`a%50`: 4 mechanisms, 17 components</summary>

<details><summary><b>2s-253</b> `a%50` @ `op` (sub) — block code, 7 codes of the same shape; 11 comps, L1 L4 L18 L20 L24 L25; tells apart 9/50 classes (best member 7); on add: 2a-191 (member overlap 0.16)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.28 |
| classes told apart: joint / best code / best member | 9 / 10 / 7 (of 50) |
| members whose removal merges classes | 0.18 |
| support overlap (1 = tiling) / random sets / p | 3.07 / 2.25 / 0.91 |
| mean CKA between its codes | 0.87 |
| purity of the joint write (per prompt) | 0.93 |
| decoding acc. joint / best code / best member (chance) | 0.21 / 0.18 / 0.10 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.17 |
| consumers / read jointly | 17 / 12 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.38 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 198.61 |
| arrangement before: shape (spectrum k:share) | irregular (2:0.28 4:0.11 20:0.09) |
| joint write: shape (spectrum k:share) | line (20:0.30 40:0.29 30:0.12) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.12 / 1.11 |
| best source position (CKA) | `a` (0.31) |

<details><summary>codes and components</summary>

**code 2s-253.0 (L1): 2 comps, tells apart 5/50, coverage 0.16, overlap 1.00 (random 1.00, p 0.59)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c388 | H26 | a%50 in {0, 5, 10, 20, 25, 30, 40} (+) | 0.89 | 0.170 |
| L1 o c488 | H26 | a%50 in {1} (+) | 0.96 | 0.021 |

**code 2s-253.1 (L1): 1 comps, tells apart 4/50, coverage 0.12, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 down c368 | MLP | a%50 in {0, 10, 20, 25, 30, 40} (+) | 0.96 | 0.140 |

**code 2s-253.2 (L4): 1 comps, tells apart 5/50, coverage 0.08, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 down c43 | MLP | a%50 in {0, 20, 30, 40} (-) | 0.91 | 0.131 |

**code 2s-253.3 (L18): 3 comps, tells apart 10/50, coverage 0.22, overlap 1.00 (random 1.30, p 0.28)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 o c313 | H30 | a%50 in {0, 10, 20, 30, 40} (+) | 0.95 | 0.181 |
| L18 o c196 | H30 | a%50 in {14, 24} (+) | 0.91 | 0.050 |
| L18 o c188 | H30 | a%50 in {9, 19, 29, 39} (+) | 0.92 | 0.090 |

**code 2s-253.4 (L20): 1 comps, tells apart 5/50, coverage 0.08, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 down c25 | MLP | a%50 in {0, 20, 30, 40} (+) | 0.93 | 0.170 |

**code 2s-253.5 (L24): 2 comps, tells apart 6/50, coverage 0.10, overlap 1.20 (random 1.00, p 0.60)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 down c117 | MLP | a%50 in {0, 10, 20, 30, 40} (+) | 0.92 | 0.130 |
| L24 down c174 | MLP | a%50 in {0} (+) | 0.97 | 0.020 |

**code 2s-253.6 (L25): 1 comps, tells apart 7/50, coverage 0.08, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L25 down c8 | MLP | a%50 in {0, 10, 30, 40} (-) | 0.90 | 0.121 |

</details>

</details>

<details><summary><b>2s-256</b> `a%50` @ `op` (sub) — block code, 2 codes of the same shape; 3 comps, L16 L28 L29; tells apart 5/50 classes (best member 4); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.08 |
| classes told apart: joint / best code / best member | 5 / 5 / 4 (of 50) |
| members whose removal merges classes | 0.33 |
| support overlap (1 = tiling) / random sets / p | 1.50 / 1.33 / 0.66 |
| mean CKA between its codes | 0.85 |
| purity of the joint write (per prompt) | 0.94 |
| decoding acc. joint / best code / best member (chance) | 0.08 / 0.07 / 0.06 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L16 attn |
| CKA(arrangement before, joint write) | 0.19 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 0.03 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.30 4:0.14 10:0.09) |
| joint write: shape (spectrum k:share) | irregular (20:0.11 40:0.11 10:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.05 / 1.64 |

<details><summary>codes and components</summary>

**code 2s-256.0 (L16): 1 comps, tells apart 4/50, coverage 0.04, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c120 | MLP | a%50 in {0, 30} (-) | 0.93 | 0.080 |

**code 2s-256.1 (L28 L29): 2 comps, tells apart 5/50, coverage 0.08, overlap 1.00 (random 1.00, p 0.62)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 down c807 | MLP | a%50 in {0, 30, 40} (-) | 0.93 | 0.060 |
| L29 down c72 | MLP | a%50 in {21} (-) | 0.99 | 0.020 |

</details>

</details>

<details><summary><b>2s-254</b> `a%50` @ `op` (sub) — single component; 1 comps, L18; tells apart 7/50 classes (best member 7); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.08 |
| classes told apart: joint / best code / best member | 7 /  / 7 (of 50) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.92 |
| decoding acc. joint / best code / best member (chance) | 0.08 /  / 0.08 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L18 attn |
| CKA(arrangement before, joint write) | 0.11 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.27 4:0.13 10:0.09) |
| joint write: shape (spectrum k:share) | line (20:0.21 30:0.21 40:0.20) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.10 / 1.00 |

<details><summary>codes and components</summary>

**code 2s-254.0 (L18): 1 comps, tells apart 7/50, coverage 0.08, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 down c982 | MLP | a%50 in {14, 24, 34, 44} (+) | 0.92 | 0.100 |

</details>

</details>

<details><summary><b>2s-255</b> `a%50` @ `op` (sub) — block code; 2 comps, L22; tells apart 7/50 classes (best member 6); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.12 |
| classes told apart: joint / best code / best member | 7 /  / 6 (of 50) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.61 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.91 |
| decoding acc. joint / best code / best member (chance) | 0.07 /  / 0.05 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 MLP |
| CKA(arrangement before, joint write) | 0.11 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.06 |
| share of the write inside the old arrangement's span | 0.19 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.27 4:0.13 10:0.09) |
| joint write: shape (spectrum k:share) | line (20:0.20 10:0.18 30:0.18) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.09 / 1.39 |
| best source position (CKA) | `a` (0.15) |

<details><summary>codes and components</summary>

**code 2s-255.0 (L22): 2 comps, tells apart 7/50, coverage 0.12, overlap 1.00 (random 1.00, p 0.59)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 down c116 | MLP | a%50 in {0} (-) | 0.96 | 0.020 |
| L22 o c22 | H15 | a%50 in {6, 16, 26, 36, 46} (-) | 0.90 | 0.150 |

</details>

</details>

</details>

<details><summary>`a%10`: 2 mechanisms, 5 components</summary>

<details><summary><b>2s-198</b> `a%10` @ `op` (sub) — 4 codes of the same shape; 4 comps, L1 L2 L12 L21; tells apart 2/10 classes (best member 2); on add: 2a-150 (member overlap 0.12)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.10 |
| classes told apart: joint / best code / best member | 2 / 2 / 2 (of 10) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 4.00 / 2.00 / 1.00 |
| mean CKA between its codes | 1.00 |
| purity of the joint write (per prompt) | 0.98 |
| decoding acc. joint / best code / best member (chance) | 0.21 / 0.21 / 0.21 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.24 |
| consumers / read jointly | 14 / 11 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.86 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.07 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 106.46 |
| arrangement before: shape (spectrum k:share) | irregular (20:0.31 40:0.29 10:0.16) |
| joint write: shape (spectrum k:share) | line (20:0.23 40:0.23 10:0.22) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.11 / 1.00 |
| best source position (CKA) | `a` (0.48) |

<details><summary>codes and components</summary>

**code 2s-198.0 (L1): 1 comps, tells apart 2/10, coverage 0.10, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c110 | H26 | a%10 in {0} (-) | 0.95 | 0.100 |

**code 2s-198.1 (L2): 1 comps, tells apart 2/10, coverage 0.10, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 down c473 | MLP | a%10 in {0} (+) | 0.96 | 0.110 |

**code 2s-198.2 (L12): 1 comps, tells apart 2/10, coverage 0.10, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L12 down c19 | MLP | a%10 in {0} (+) | 0.98 | 0.100 |

**code 2s-198.3 (L21): 1 comps, tells apart 2/10, coverage 0.10, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c19 | MLP | a%10 in {0} (-) | 0.97 | 0.110 |

</details>

</details>

<details><summary><b>2s-199</b> `a%10` @ `op` (sub) — single component; 1 comps, L18; tells apart 2/10 classes (best member 2); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.10 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 10) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.93 |
| decoding acc. joint / best code / best member (chance) | 0.21 /  / 0.21 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L17 MLP |
| CKA(arrangement before, joint write) | 0.25 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.12 |
| share of the write inside the old arrangement's span | 0.15 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (10:0.29 20:0.28 40:0.17) |
| joint write: shape (spectrum k:share) | line (10:0.23 40:0.23 20:0.21) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.11 / 1.00 |
| best source position (CKA) | `a` (0.34) |

<details><summary>codes and components</summary>

**code 2s-199.0 (L18): 1 comps, tells apart 2/10, coverage 0.10, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 o c200 | H30 | a%10 in {5} (+) | 0.93 | 0.110 |

</details>

</details>

</details>
