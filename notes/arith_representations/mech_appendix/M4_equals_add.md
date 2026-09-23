[← back to the report](../report_mechanisms.md)

# Every mechanism at `=`, add

<details><summary>`res%100`: 12 mechanisms, 288 components</summary>

<details><summary><b>4a-502</b> `res%100` @ `=` (add) — block code; 3 comps, L20; tells apart 16/100 classes (best member 2); on sub: 4s-626 (member overlap 0.05)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.59 |
| classes told apart: joint / best code / best member | 16 /  / 2 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.19 / 1.00 / 0.88 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.75 |
| decoding acc. joint / best code / best member (chance) | 0.15 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.02 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L20 attn |
| CKA(arrangement before, joint write) | 0.53 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.10 |
| share of the write inside the old arrangement's span | 0.13 |
| write energy / code energy before | 0.08 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.34 1:0.27 10:0.12) |
| joint write: shape (spectrum k:share) | irregular (2:0.47 1:0.20 20:0.14) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.37 / 2.59 |

<details><summary>codes and components</summary>

**code 4a-502.0 (L20): 3 comps, tells apart 16/100, coverage 0.59, overlap 1.19 (random 1.00, p 0.90)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 down c31 | MLP | res%100 in {0..12, 96..99} (+) | 0.76 | 0.286 |
| L20 down c29 | MLP | res%100 in {12..23, 61..80} (-) | 0.75 | 0.557 |
| L20 down c100 | MLP | res%100 in {3..4, 13..14, 19, 23, 26..27, 30..31, 36, 43, 53..54, 59, 63..64, 73, 83, 93, 99} (+) | 0.71 | 0.707 |

</details>

</details>

<details><summary><b>4a-503</b> `res%100` @ `=` (add) — block code; 5 comps, L21; tells apart 17/100 classes (best member 7); on sub: 4s-625 (member overlap 0.09)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.40 |
| classes told apart: joint / best code / best member | 17 /  / 7 (of 100) |
| members whose removal merges classes | 0.80 |
| support overlap (1 = tiling) / random sets / p | 1.35 / 1.07 / 0.95 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.79 |
| decoding acc. joint / best code / best member (chance) | 0.14 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.95 |
| consumers / read jointly | 4 / 2 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 attn |
| CKA(arrangement before, joint write) | 0.38 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.07 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.04 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.25 1:0.23 20:0.16) |
| joint write: shape (spectrum k:share) | irregular (2:0.46 1:0.18 4:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.25 / 1.91 |

<details><summary>codes and components</summary>

**code 4a-503.0 (L21): 5 comps, tells apart 17/100, coverage 0.40, overlap 1.35 (random 1.06, p 0.95)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c43 | MLP | res%100 in {0..5, 86..99} (+) | 0.87 | 0.280 |
| L21 down c7 | MLP | res%100 in {0..7, 96..99} (-) | 0.84 | 0.420 |
| L21 down c96 | MLP | res%100 in {18..20, 76..80} (-) | 0.57 | 0.118 |
| L21 down c63 | MLP | res%100 in {23..29} (-) | 0.70 | 0.077 |
| L21 down c916 | MLP | res%100 in {7..8, 28, 48, 68, 87..88} (+) | 0.73 | 0.084 |

</details>

</details>

<details><summary><b>4a-504</b> `res%100` @ `=` (add) — block code; 9 comps, L22; tells apart 35/100 classes (best member 7); on sub: 4s-626 (member overlap 0.04)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.50 |
| classes told apart: joint / best code / best member | 35 /  / 7 (of 100) |
| members whose removal merges classes | 0.78 |
| support overlap (1 = tiling) / random sets / p | 1.26 / 1.15 / 0.76 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.82 |
| decoding acc. joint / best code / best member (chance) | 0.24 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 16 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L22 attn |
| CKA(arrangement before, joint write) | 0.36 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.06 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.21 2:0.21 10:0.17) |
| joint write: shape (spectrum k:share) | irregular (1:0.20 2:0.19 3:0.15) |
| frequencies new in the write | 4 |
| write: shift-symmetric part / dimension (PR) | 0.24 / 3.66 |

<details><summary>codes and components</summary>

**code 4a-504.0 (L22): 9 comps, tells apart 35/100, coverage 0.50, overlap 1.26 (random 1.15, p 0.83)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 down c123 | MLP | res%100 in {0..4, 98} (-) | 0.78 | 0.073 |
| L22 down c22 | MLP | res%100 in {1, 7, 11, 15, 17, 21, 25, 27, 31, 41, 51, 61, 67, 71, 77, 81, 91} (-) | 0.60 | 0.223 |
| L22 down c48 | MLP | res%100 in {28, 68, 88} (-) | 0.80 | 0.114 |
| L22 down c27 | MLP | res%100 in {29..36} (+) | 0.86 | 0.118 |
| L22 down c6 | MLP | res%100 in {3..10} (-) | 0.85 | 0.164 |
| L22 down c30 | MLP | res%100 in {31..36} (+) | 0.74 | 0.066 |
| L22 down c5 | MLP | res%100 in {42..49} (-) | 0.88 | 0.161 |
| L22 down c257 | MLP | res%100 in {45..46} (-) | 0.62 | 0.018 |
| L22 down c98 | MLP | res%100 in {55..59} (-) | 0.77 | 0.084 |

</details>

</details>

<details><summary><b>4a-505</b> `res%100` @ `=` (add) — block code; 11 comps, L23; tells apart 40/100 classes (best member 10); on sub: 4s-627 (member overlap 0.08)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.60 |
| classes told apart: joint / best code / best member | 40 /  / 10 (of 100) |
| members whose removal merges classes | 0.91 |
| support overlap (1 = tiling) / random sets / p | 1.30 / 1.21 / 0.77 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.88 |
| decoding acc. joint / best code / best member (chance) | 0.35 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 29 / 6 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L23 attn |
| CKA(arrangement before, joint write) | 0.37 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.09 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.20 2:0.19 10:0.15) |
| joint write: shape (spectrum k:share) | irregular (1:0.20 2:0.18 3:0.17) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.32 / 4.71 |

<details><summary>codes and components</summary>

**code 4a-505.0 (L23): 11 comps, tells apart 40/100, coverage 0.60, overlap 1.30 (random 1.21, p 0.76)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 down c3 | MLP | res%100 in {0, 94..99} (+) | 0.91 | 0.150 |
| L23 down c202 | MLP | res%100 in {14..20} (-) | 0.89 | 0.089 |
| L23 down c459 | MLP | res%100 in {15, 25, 35, 55, 65, 75, 85, 95} (+) | 0.76 | 0.072 |
| L23 down c77 | MLP | res%100 in {21, 61, 81} (+) | 0.82 | 0.056 |
| L23 down c141 | MLP | res%100 in {28, 88} (+) | 0.65 | 0.023 |
| L23 down c105 | MLP | res%100 in {3, 13, 23, 43, 53, 63, 73, 83, 93} (-) | 0.74 | 0.079 |
| L23 down c15 | MLP | res%100 in {31..35, 40} (+) | 0.86 | 0.118 |
| L23 down c5 | MLP | res%100 in {35..41} (-) | 0.88 | 0.132 |
| L23 down c216 | MLP | res%100 in {63..70, 85..86} (-) | 0.76 | 0.131 |
| L23 down c17 | MLP | res%100 in {67..74} (-) | 0.88 | 0.151 |
| L23 down c16 | MLP | res%100 in {73..83} (-) | 0.92 | 0.177 |

</details>

</details>

<details><summary><b>4a-506</b> `res%100` @ `=` (add) — block code; 24 comps, L24; tells apart 65/100 classes (best member 12); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.83 |
| classes told apart: joint / best code / best member | 65 /  / 12 (of 100) |
| members whose removal merges classes | 0.88 |
| support overlap (1 = tiling) / random sets / p | 1.83 / 1.54 / 0.89 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.87 |
| decoding acc. joint / best code / best member (chance) | 0.67 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 40 / 30 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L24 attn |
| CKA(arrangement before, joint write) | 0.51 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.13 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.20 2:0.17 10:0.14) |
| joint write: shape (spectrum k:share) | irregular (2:0.15 1:0.13 3:0.13) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.51 / 11.33 |

<details><summary>codes and components</summary>

**code 4a-506.0 (L24): 24 comps, tells apart 65/100, coverage 0.83, overlap 1.83 (random 1.54, p 0.94)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 down c377 | MLP | res%100 in {0..3, 94..99} (-) | 0.83 | 0.104 |
| L24 down c593 | MLP | res%100 in {10, 50, 70} (+) | 0.64 | 0.041 |
| L24 down c22 | MLP | res%100 in {14..17, 64, 66..67, 74} (+) | 0.82 | 0.142 |
| L24 down c43 | MLP | res%100 in {15..17, 37, 56..57, 75..79, 95..99} (-) | 0.79 | 0.220 |
| L24 down c91 | MLP | res%100 in {16, 56, 96} (-) | 0.83 | 0.080 |
| L24 down c101 | MLP | res%100 in {16..24} (+) | 0.93 | 0.100 |
| L24 down c212 | MLP | res%100 in {20, 40, 80} (-) | 0.62 | 0.035 |
| L24 down c82 | MLP | res%100 in {21, 31..32, 41..42, 81} (-) | 0.82 | 0.085 |
| L24 down c16 | MLP | res%100 in {21..22, 24..26, 71..72, 74..76, 82} (+) | 0.82 | 0.183 |
| L24 down c11 | MLP | res%100 in {24, 62..67} (+) | 0.89 | 0.172 |
| L24 down c50 | MLP | res%100 in {26} (-) | 0.85 | 0.061 |
| L24 down c52 | MLP | res%100 in {34} (+) | 0.85 | 0.092 |
| L24 down c263 | MLP | res%100 in {4..8} (+) | 0.67 | 0.050 |
| L24 down c17 | MLP | res%100 in {49..53} (-) | 0.90 | 0.121 |
| L24 down c28 | MLP | res%100 in {53..55} (+) | 0.84 | 0.161 |
| L24 down c13 | MLP | res%100 in {54..62} (+) | 0.89 | 0.143 |
| L24 down c36 | MLP | res%100 in {71..76} (-) | 0.92 | 0.093 |
| L24 down c191 | MLP | res%100 in {77..83} (-) | 0.89 | 0.082 |
| L24 down c31 | MLP | res%100 in {8..15} (-) | 0.87 | 0.188 |
| L24 down c87 | MLP | res%100 in {81..86} (-) | 0.85 | 0.076 |
| L24 down c39 | MLP | res%100 in {82..85, 89..92} (+) | 0.88 | 0.122 |
| L24 down c15 | MLP | res%100 in {83..87} (+) | 0.91 | 0.108 |
| L24 down c41 | MLP | res%100 in {85..94} (-) | 0.91 | 0.154 |
| L24 down c37 | MLP | res%100 in {86..87} (+) | 0.86 | 0.092 |

</details>

</details>

<details><summary><b>4a-507</b> `res%100` @ `=` (add) — block code; 24 comps, L25; tells apart 59/100 classes (best member 11); on sub: 4s-629 (member overlap 0.15)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.70 |
| classes told apart: joint / best code / best member | 59 /  / 11 (of 100) |
| members whose removal merges classes | 0.71 |
| support overlap (1 = tiling) / random sets / p | 1.54 / 1.56 / 0.47 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.88 |
| decoding acc. joint / best code / best member (chance) | 0.64 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.04 |
| consumers / read jointly | 30 / 19 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L25 attn |
| CKA(arrangement before, joint write) | 0.42 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.10 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.17 2:0.15 10:0.13) |
| joint write: shape (spectrum k:share) | irregular (1:0.11 2:0.10 5:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.26 / 8.34 |

<details><summary>codes and components</summary>

**code 4a-507.0 (L25): 24 comps, tells apart 59/100, coverage 0.70, overlap 1.54 (random 1.56, p 0.44)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L25 down c40 | MLP | res%100 in {0..1} (+) | 0.89 | 0.094 |
| L25 down c15 | MLP | res%100 in {1..5} (-) | 0.90 | 0.085 |
| L25 down c216 | MLP | res%100 in {10, 20, 40, 60, 70, 80} (-) | 0.81 | 0.062 |
| L25 down c42 | MLP | res%100 in {10..13} (+) | 0.88 | 0.090 |
| L25 down c207 | MLP | res%100 in {12, 52, 72, 92} (-) | 0.83 | 0.058 |
| L25 down c586 | MLP | res%100 in {13, 33, 73} (-) | 0.75 | 0.033 |
| L25 down c76 | MLP | res%100 in {18, 38, 58..59, 78, 98} (+) | 0.81 | 0.146 |
| L25 down c46 | MLP | res%100 in {18..19} (+) | 0.90 | 0.051 |
| L25 down c50 | MLP | res%100 in {23, 63} (+) | 0.89 | 0.113 |
| L25 down c98 | MLP | res%100 in {24, 64} (-) | 0.82 | 0.035 |
| L25 down c151 | MLP | res%100 in {27..29} (+) | 0.85 | 0.077 |
| L25 down c63 | MLP | res%100 in {40, 50, 90} (+) | 0.83 | 0.098 |
| L25 down c66 | MLP | res%100 in {42..45, 82..83} (+) | 0.92 | 0.132 |
| L25 down c31 | MLP | res%100 in {5..13} (+) | 0.84 | 0.155 |
| L25 down c335 | MLP | res%100 in {53} (-) | 0.85 | 0.090 |
| L25 down c130 | MLP | res%100 in {56} (-) | 0.68 | 0.015 |
| L25 down c30 | MLP | res%100 in {59..63, 66, 69..73} (+) | 0.87 | 0.189 |
| L25 down c248 | MLP | res%100 in {75..79} (-) | 0.89 | 0.058 |
| L25 down c48 | MLP | res%100 in {8..10, 13, 89..90, 99} (+) | 0.86 | 0.163 |
| L25 down c297 | MLP | res%100 in {84..89} (+) | 0.90 | 0.083 |
| L25 down c114 | MLP | res%100 in {86, 94..99} (+) | 0.84 | 0.075 |
| L25 down c10 | MLP | res%100 in {90..95} (-) | 0.91 | 0.128 |
| L25 down c139 | MLP | res%100 in {95..99} (-) | 0.86 | 0.120 |
| L25 down c331 | MLP | res%100 in {98..99} (-) | 0.81 | 0.025 |

</details>

</details>

<details><summary><b>4a-508</b> `res%100` @ `=` (add) — block code; 36 comps, L26; tells apart 85/100 classes (best member 11); on sub: 4s-626 (member overlap 0.08)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.86 |
| classes told apart: joint / best code / best member | 85 /  / 11 (of 100) |
| members whose removal merges classes | 0.64 |
| support overlap (1 = tiling) / random sets / p | 1.90 / 1.95 / 0.38 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.84 |
| decoding acc. joint / best code / best member (chance) | 0.75 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.05 |
| consumers / read jointly | 6 / 2 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L26 attn |
| CKA(arrangement before, joint write) | 0.61 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.07 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.16 2:0.14 10:0.12) |
| joint write: shape (spectrum k:share) | irregular (1:0.12 2:0.11 3:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.49 / 15.55 |

<details><summary>codes and components</summary>

**code 4a-508.0 (L26): 36 comps, tells apart 85/100, coverage 0.86, overlap 1.90 (random 1.93, p 0.41)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 down c139 | MLP | res%100 in {0..3, 99} (+) | 0.88 | 0.089 |
| L26 down c86 | MLP | res%100 in {1..6} (+) | 0.80 | 0.077 |
| L26 down c180 | MLP | res%100 in {10} (+) | 0.66 | 0.007 |
| L26 down c205 | MLP | res%100 in {11..13} (+) | 0.83 | 0.051 |
| L26 down c363 | MLP | res%100 in {12, 32, 52, 72, 92} (+) | 0.82 | 0.063 |
| L26 down c33 | MLP | res%100 in {12..13, 16..19} (+) | 0.84 | 0.080 |
| L26 down c258 | MLP | res%100 in {13, 73, 93} (+) | 0.84 | 0.080 |
| L26 down c813 | MLP | res%100 in {15..16} (+) | 0.82 | 0.035 |
| L26 down c23 | MLP | res%100 in {2..6} (+) | 0.75 | 0.045 |
| L26 down c8 | MLP | res%100 in {20..22} (-) | 0.90 | 0.067 |
| L26 down c348 | MLP | res%100 in {25, 34..35, 74..75, 94} (-) | 0.72 | 0.069 |
| L26 down c20 | MLP | res%100 in {30..33} (-) | 0.90 | 0.073 |
| L26 down c24 | MLP | res%100 in {33, 35, 55, 83, 85, 88} (+) | 0.77 | 0.163 |
| L26 down c529 | MLP | res%100 in {4, 9..13} (+) | 0.69 | 0.086 |
| L26 down c157 | MLP | res%100 in {45, 94..95} (+) | 0.56 | 0.053 |
| L26 down c113 | MLP | res%100 in {45..49} (+) | 0.91 | 0.084 |
| L26 down c88 | MLP | res%100 in {48, 88} (+) | 0.83 | 0.051 |
| L26 down c237 | MLP | res%100 in {49..54} (-) | 0.86 | 0.074 |
| L26 down c840 | MLP | res%100 in {53, 93} (+) | 0.78 | 0.018 |
| L26 down c142 | MLP | res%100 in {55, 95} (+) | 0.63 | 0.061 |
| L26 down c94 | MLP | res%100 in {57..59} (+) | 0.89 | 0.059 |
| L26 down c194 | MLP | res%100 in {58, 78} (-) | 0.87 | 0.109 |
| L26 down c12 | MLP | res%100 in {60..65} (-) | 0.91 | 0.094 |
| L26 down c219 | MLP | res%100 in {61} (+) | 0.75 | 0.009 |
| L26 down c703 | MLP | res%100 in {65, 67..69} (-) | 0.84 | 0.044 |
| L26 down c2 | MLP | res%100 in {66..73, 76..77} (+) | 0.75 | 0.109 |
| L26 down c37 | MLP | res%100 in {7..8, 17..18, 27, 37..38, 57, 97} (+) | 0.84 | 0.125 |
| L26 down c207 | MLP | res%100 in {7..9} (-) | 0.81 | 0.051 |
| L26 down c118 | MLP | res%100 in {72..77, 79..85} (+) | 0.90 | 0.176 |
| L26 down c299 | MLP | res%100 in {75..76, 78} (-) | 0.86 | 0.077 |
| L26 down c89 | MLP | res%100 in {77..78} (-) | 0.93 | 0.074 |
| L26 down c176 | MLP | res%100 in {83..89} (+) | 0.66 | 0.070 |
| L26 down c46 | MLP | res%100 in {87..90} (-) | 0.84 | 0.088 |
| L26 down c155 | MLP | res%100 in {89..91} (+) | 0.78 | 0.107 |
| L26 down c570 | MLP | res%100 in {90..94} (-) | 0.86 | 0.050 |
| L26 down c341 | MLP | res%100 in {92..98} (+) | 0.80 | 0.071 |

</details>

</details>

<details><summary><b>4a-509</b> `res%100` @ `=` (add) — block code; 27 comps, L27; tells apart 52/100 classes (best member 10); on sub: 4s-626 (member overlap 0.05)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.71 |
| classes told apart: joint / best code / best member | 52 /  / 10 (of 100) |
| members whose removal merges classes | 0.59 |
| support overlap (1 = tiling) / random sets / p | 1.49 / 1.67 / 0.12 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.86 |
| decoding acc. joint / best code / best member (chance) | 0.61 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.09 |
| consumers / read jointly | 11 / 1 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L27 attn |
| CKA(arrangement before, joint write) | 0.51 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.07 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.15 2:0.13 10:0.11) |
| joint write: shape (spectrum k:share) | irregular (2:0.10 1:0.10 5:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.32 / 10.93 |

<details><summary>codes and components</summary>

**code 4a-509.0 (L27): 27 comps, tells apart 52/100, coverage 0.71, overlap 1.49 (random 1.64, p 0.16)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 down c157 | MLP | res%100 in {0, 2..5} (-) | 0.76 | 0.051 |
| L27 down c122 | MLP | res%100 in {0, 8..10, 12..13} (-) | 0.69 | 0.063 |
| L27 down c823 | MLP | res%100 in {0} (-) | 0.62 | 0.008 |
| L27 down c269 | MLP | res%100 in {1, 41, 51, 61, 71, 81, 91} (-) | 0.81 | 0.074 |
| L27 down c139 | MLP | res%100 in {1} (-) | 0.82 | 0.102 |
| L27 down c586 | MLP | res%100 in {20..21} (-) | 0.83 | 0.019 |
| L27 down c204 | MLP | res%100 in {21, 31, 61, 71, 81, 91} (-) | 0.83 | 0.086 |
| L27 down c64 | MLP | res%100 in {22..25, 52..53} (+) | 0.84 | 0.097 |
| L27 down c203 | MLP | res%100 in {27, 67, 87} (-) | 0.88 | 0.046 |
| L27 down c24 | MLP | res%100 in {35..37} (+) | 0.86 | 0.073 |
| L27 down c168 | MLP | res%100 in {41..44} (-) | 0.83 | 0.068 |
| L27 down c112 | MLP | res%100 in {44, 84} (+) | 0.88 | 0.083 |
| L27 down c125 | MLP | res%100 in {48..49, 98..99} (+) | 0.78 | 0.066 |
| L27 down c200 | MLP | res%100 in {48..50, 89} (+) | 0.70 | 0.070 |
| L27 down c62 | MLP | res%100 in {5..6} (-) | 0.79 | 0.112 |
| L27 down c82 | MLP | res%100 in {52..55, 57..63} (+) | 0.81 | 0.156 |
| L27 down c107 | MLP | res%100 in {52..58} (+) | 0.86 | 0.105 |
| L27 down c461 | MLP | res%100 in {56} (-) | 0.73 | 0.024 |
| L27 down c42 | MLP | res%100 in {69} (-) | 0.89 | 0.142 |
| L27 down c32 | MLP | res%100 in {7..10} (+) | 0.78 | 0.084 |
| L27 down c140 | MLP | res%100 in {71..78} (+) | 0.93 | 0.110 |
| L27 down c110 | MLP | res%100 in {78..79} (+) | 0.91 | 0.105 |
| L27 down c17 | MLP | res%100 in {80..83} (+) | 0.94 | 0.075 |
| L27 down c480 | MLP | res%100 in {88..89} (-) | 0.87 | 0.044 |
| L27 down c80 | MLP | res%100 in {9, 49, 89} (+) | 0.87 | 0.108 |
| L27 down c33 | MLP | res%100 in {93..98} (-) | 0.85 | 0.141 |
| L27 down c38 | MLP | res%100 in {99} (-) | 0.89 | 0.103 |

</details>

</details>

<details><summary><b>4a-510</b> `res%100` @ `=` (add) — tiling; 37 comps, L28; tells apart 63/100 classes (best member 9); on sub: 4s-630 (member overlap 0.06)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.76 |
| classes told apart: joint / best code / best member | 63 /  / 9 (of 100) |
| members whose removal merges classes | 0.81 |
| support overlap (1 = tiling) / random sets / p | 1.57 / 1.93 / 0.01 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.86 |
| decoding acc. joint / best code / best member (chance) | 0.72 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.05 |
| consumers / read jointly | 2 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L28 attn |
| CKA(arrangement before, joint write) | 0.54 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.08 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.17 2:0.13 10:0.09) |
| joint write: shape (spectrum k:share) | irregular (2:0.08 1:0.08 5:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.31 / 12.76 |

<details><summary>codes and components</summary>

**code 4a-510.0 (L28): 37 comps, tells apart 63/100, coverage 0.76, overlap 1.57 (random 1.94, p 0.01)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 down c40 | MLP | res%100 in {1, 99} (+) | 0.82 | 0.032 |
| L28 down c191 | MLP | res%100 in {1..4, 51} (+) | 0.85 | 0.096 |
| L28 down c90 | MLP | res%100 in {11..12, 52, 71..72, 90..92} (+) | 0.63 | 0.138 |
| L28 down c10 | MLP | res%100 in {13..15} (+) | 0.87 | 0.100 |
| L28 down c24 | MLP | res%100 in {16, 76} (-) | 0.91 | 0.122 |
| L28 down c522 | MLP | res%100 in {17, 47} (+) | 0.72 | 0.016 |
| L28 down c51 | MLP | res%100 in {18} (-) | 0.89 | 0.054 |
| L28 down c903 | MLP | res%100 in {2, 72} (+) | 0.69 | 0.017 |
| L28 down c296 | MLP | res%100 in {20, 40, 60, 80} (+) | 0.73 | 0.052 |
| L28 down c794 | MLP | res%100 in {21..24} (-) | 0.68 | 0.039 |
| L28 down c165 | MLP | res%100 in {26..27, 68} (+) | 0.83 | 0.062 |
| L28 down c67 | MLP | res%100 in {27, 47, 77} (+) | 0.88 | 0.131 |
| L28 down c99 | MLP | res%100 in {28, 38, 48, 58, 68, 88, 98} (-) | 0.90 | 0.092 |
| L28 down c121 | MLP | res%100 in {29} (-) | 0.91 | 0.103 |
| L28 down c25 | MLP | res%100 in {3, 7..8} (+) | 0.68 | 0.029 |
| L28 down c46 | MLP | res%100 in {30} (-) | 0.91 | 0.081 |
| L28 down c719 | MLP | res%100 in {32, 52, 72, 92} (-) | 0.81 | 0.041 |
| L28 down c140 | MLP | res%100 in {4, 64, 74, 84, 94} (-) | 0.74 | 0.050 |
| L28 down c275 | MLP | res%100 in {41, 61, 81} (-) | 0.88 | 0.047 |
| L28 down c595 | MLP | res%100 in {42..46} (-) | 0.75 | 0.053 |
| L28 down c82 | MLP | res%100 in {43} (-) | 0.84 | 0.157 |
| L28 down c71 | MLP | res%100 in {46, 65..66} (-) | 0.91 | 0.145 |
| L28 down c124 | MLP | res%100 in {53, 93} (-) | 0.81 | 0.027 |
| L28 down c291 | MLP | res%100 in {57, 67, 77, 87, 97} (+) | 0.63 | 0.057 |
| L28 down c80 | MLP | res%100 in {70..73} (+) | 0.92 | 0.084 |
| L28 down c965 | MLP | res%100 in {71, 91} (+) | 0.56 | 0.018 |
| L28 down c83 | MLP | res%100 in {78} (+) | 0.92 | 0.123 |
| L28 down c62 | MLP | res%100 in {80, 86..87, 90..93, 95..96} (-) | 0.54 | 0.126 |
| L28 down c115 | MLP | res%100 in {81..84} (-) | 0.89 | 0.117 |
| L28 down c543 | MLP | res%100 in {86..87} (-) | 0.78 | 0.073 |
| L28 down c298 | MLP | res%100 in {89..95} (-) | 0.84 | 0.107 |
| L28 down c435 | MLP | res%100 in {8} (+) | 0.85 | 0.009 |
| L28 down c952 | MLP | res%100 in {9, 69} (-) | 0.82 | 0.019 |
| L28 down c433 | MLP | res%100 in {93, 95..96} (+) | 0.64 | 0.046 |
| L28 down c22 | MLP | res%100 in {96..98} (+) | 0.89 | 0.172 |
| L28 down c545 | MLP | res%100 in {99} (+) | 0.66 | 0.012 |
| L28 down c416 | MLP | res%100 in {9} (-) | 0.80 | 0.010 |

</details>

</details>

<details><summary><b>4a-511</b> `res%100` @ `=` (add) — tiling; 57 comps, L29; tells apart 71/100 classes (best member 9); on sub: 4s-631 (member overlap 0.03)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.74 |
| classes told apart: joint / best code / best member | 71 /  / 9 (of 100) |
| members whose removal merges classes | 0.51 |
| support overlap (1 = tiling) / random sets / p | 1.66 / 2.62 / 0.00 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.85 |
| decoding acc. joint / best code / best member (chance) | 0.77 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.15 |
| consumers / read jointly | 2 / 2 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L29 attn |
| CKA(arrangement before, joint write) | 0.56 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.09 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.15 2:0.12 10:0.09) |
| joint write: shape (spectrum k:share) | irregular (2:0.09 1:0.08 5:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.34 / 15.56 |

<details><summary>codes and components</summary>

**code 4a-511.0 (L29): 57 comps, tells apart 71/100, coverage 0.74, overlap 1.66 (random 2.59, p 0.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L29 down c358 | MLP | res%100 in {0, 99} (-) | 0.86 | 0.090 |
| L29 down c174 | MLP | res%100 in {0..5, 99} (+) | 0.85 | 0.141 |
| L29 down c497 | MLP | res%100 in {0} (+) | 0.71 | 0.014 |
| L29 down c794 | MLP | res%100 in {1..3} (+) | 0.85 | 0.057 |
| L29 down c48 | MLP | res%100 in {10..11} (+) | 0.84 | 0.125 |
| L29 down c604 | MLP | res%100 in {11, 51, 71} (+) | 0.88 | 0.102 |
| L29 down c445 | MLP | res%100 in {11} (+) | 0.83 | 0.012 |
| L29 down c209 | MLP | res%100 in {12, 62} (-) | 0.82 | 0.121 |
| L29 down c230 | MLP | res%100 in {12..13, 72..73, 92..93} (+) | 0.85 | 0.145 |
| L29 down c318 | MLP | res%100 in {12} (+) | 0.85 | 0.013 |
| L29 down c263 | MLP | res%100 in {13} (-) | 0.88 | 0.016 |
| L29 down c739 | MLP | res%100 in {14} (+) | 0.73 | 0.019 |
| L29 down c467 | MLP | res%100 in {15} (+) | 0.86 | 0.011 |
| L29 down c477 | MLP | res%100 in {16, 96} (+) | 0.69 | 0.021 |
| L29 down c280 | MLP | res%100 in {16} (-) | 0.90 | 0.025 |
| L29 down c343 | MLP | res%100 in {17} (-) | 0.92 | 0.016 |
| L29 down c577 | MLP | res%100 in {19, 29, 49, 79, 89} (-) | 0.73 | 0.073 |
| L29 down c842 | MLP | res%100 in {21} (+) | 0.75 | 0.055 |
| L29 down c448 | MLP | res%100 in {25} (+) | 0.85 | 0.033 |
| L29 down c376 | MLP | res%100 in {26} (-) | 0.92 | 0.017 |
| L29 down c178 | MLP | res%100 in {28} (-) | 0.93 | 0.037 |
| L29 down c128 | MLP | res%100 in {3..8} (-) | 0.84 | 0.085 |
| L29 down c415 | MLP | res%100 in {31} (-) | 0.80 | 0.033 |
| L29 down c187 | MLP | res%100 in {33} (-) | 0.82 | 0.062 |
| L29 down c183 | MLP | res%100 in {34} (-) | 0.89 | 0.018 |
| L29 down c566 | MLP | res%100 in {35} (+) | 0.79 | 0.014 |
| L29 down c145 | MLP | res%100 in {36, 56, 86} (+) | 0.86 | 0.080 |
| L29 down c130 | MLP | res%100 in {4, 64} (-) | 0.89 | 0.122 |
| L29 down c136 | MLP | res%100 in {41} (+) | 0.90 | 0.130 |
| L29 down c282 | MLP | res%100 in {42} (+) | 0.92 | 0.104 |
| L29 down c172 | MLP | res%100 in {45, 85} (+) | 0.78 | 0.066 |
| L29 down c964 | MLP | res%100 in {49..50} (-) | 0.74 | 0.025 |
| L29 down c434 | MLP | res%100 in {51, 81} (-) | 0.85 | 0.028 |
| L29 down c433 | MLP | res%100 in {52..53} (-) | 0.72 | 0.028 |
| L29 down c153 | MLP | res%100 in {57..58} (+) | 0.75 | 0.037 |
| L29 down c193 | MLP | res%100 in {6, 8, 46, 48, 66, 68, 86, 88} (-) | 0.80 | 0.090 |
| L29 down c68 | MLP | res%100 in {6..7} (-) | 0.82 | 0.100 |
| L29 down c420 | MLP | res%100 in {60, 90} (-) | 0.58 | 0.013 |
| L29 down c438 | MLP | res%100 in {62} (-) | 0.65 | 0.007 |
| L29 down c435 | MLP | res%100 in {65..66} (-) | 0.52 | 0.102 |
| L29 down c321 | MLP | res%100 in {65} (+) | 0.86 | 0.085 |
| L29 down c90 | MLP | res%100 in {67..68} (+) | 0.88 | 0.078 |
| L29 down c146 | MLP | res%100 in {74..76} (-) | 0.89 | 0.078 |
| L29 down c238 | MLP | res%100 in {79} (+) | 0.87 | 0.031 |
| L29 down c53 | MLP | res%100 in {83} (-) | 0.62 | 0.020 |
| L29 down c595 | MLP | res%100 in {87..89} (-) | 0.86 | 0.047 |
| L29 down c140 | MLP | res%100 in {88} (-) | 0.86 | 0.060 |
| L29 down c149 | MLP | res%100 in {89..90} (+) | 0.84 | 0.054 |
| L29 down c930 | MLP | res%100 in {9, 69, 99} (-) | 0.77 | 0.068 |
| L29 down c1009 | MLP | res%100 in {90..91, 93, 95} (-) | 0.62 | 0.029 |
| L29 down c871 | MLP | res%100 in {90} (-) | 0.53 | 0.007 |
| L29 down c444 | MLP | res%100 in {91..92, 95..98} (+) | 0.73 | 0.080 |
| L29 down c213 | MLP | res%100 in {91..94} (-) | 0.70 | 0.033 |
| L29 down c426 | MLP | res%100 in {92} (+) | 0.73 | 0.015 |
| L29 down c71 | MLP | res%100 in {94..95} (+) | 0.90 | 0.095 |
| L29 down c414 | MLP | res%100 in {94} (-) | 0.75 | 0.009 |
| L29 down c736 | MLP | res%100 in {99} (+) | 0.79 | 0.020 |

</details>

</details>

<details><summary><b>4a-512</b> `res%100` @ `=` (add) — tiling; 36 comps, L30; tells apart 58/100 classes (best member 10); on sub: 4s-632 (member overlap 0.05)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.51 |
| classes told apart: joint / best code / best member | 58 /  / 10 (of 100) |
| members whose removal merges classes | 0.75 |
| support overlap (1 = tiling) / random sets / p | 1.49 / 1.89 / 0.01 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.85 |
| decoding acc. joint / best code / best member (chance) | 0.56 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.17 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L30 attn |
| CKA(arrangement before, joint write) | 0.50 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.03 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.05 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.13 2:0.11 10:0.08) |
| joint write: shape (spectrum k:share) | irregular (2:0.07 1:0.06 10:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.25 / 13.86 |

<details><summary>codes and components</summary>

**code 4a-512.0 (L30): 36 comps, tells apart 58/100, coverage 0.51, overlap 1.49 (random 1.90, p 0.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 down c519 | MLP | res%100 in {14} (-) | 0.91 | 0.013 |
| L30 down c214 | MLP | res%100 in {17, 77} (+) | 0.87 | 0.073 |
| L30 down c294 | MLP | res%100 in {19} (-) | 0.82 | 0.011 |
| L30 down c739 | MLP | res%100 in {19} (-) | 0.83 | 0.009 |
| L30 down c796 | MLP | res%100 in {20, 60} (-) | 0.83 | 0.035 |
| L30 down c283 | MLP | res%100 in {21} (-) | 0.79 | 0.020 |
| L30 down c997 | MLP | res%100 in {22} (-) | 0.81 | 0.014 |
| L30 down c628 | MLP | res%100 in {23} (+) | 0.86 | 0.018 |
| L30 down c551 | MLP | res%100 in {24} (+) | 0.76 | 0.018 |
| L30 down c359 | MLP | res%100 in {24} (-) | 0.91 | 0.075 |
| L30 down c810 | MLP | res%100 in {2} (+) | 0.81 | 0.026 |
| L30 down c594 | MLP | res%100 in {3, 5..8} (-) | 0.67 | 0.049 |
| L30 down c399 | MLP | res%100 in {3..8} (-) | 0.79 | 0.106 |
| L30 down c183 | MLP | res%100 in {32, 82} (+) | 0.83 | 0.058 |
| L30 down c580 | MLP | res%100 in {32} (-) | 0.73 | 0.025 |
| L30 down c209 | MLP | res%100 in {34, 74} (+) | 0.87 | 0.097 |
| L30 down c187 | MLP | res%100 in {37, 87} (+) | 0.87 | 0.097 |
| L30 down c317 | MLP | res%100 in {3} (+) | 0.81 | 0.029 |
| L30 down c309 | MLP | res%100 in {4..6} (-) | 0.81 | 0.029 |
| L30 down c223 | MLP | res%100 in {40} (-) | 0.82 | 0.024 |
| L30 down c961 | MLP | res%100 in {51..52} (-) | 0.83 | 0.055 |
| L30 down c177 | MLP | res%100 in {56..57, 96..97} (-) | 0.81 | 0.100 |
| L30 down c475 | MLP | res%100 in {58, 68} (+) | 0.66 | 0.015 |
| L30 down c249 | MLP | res%100 in {63} (+) | 0.84 | 0.083 |
| L30 down c808 | MLP | res%100 in {67..69} (-) | 0.87 | 0.046 |
| L30 down c603 | MLP | res%100 in {70} (-) | 0.82 | 0.025 |
| L30 down c452 | MLP | res%100 in {71} (+) | 0.89 | 0.012 |
| L30 down c379 | MLP | res%100 in {73..74} (-) | 0.83 | 0.043 |
| L30 down c232 | MLP | res%100 in {80..84, 88} (-) | 0.79 | 0.086 |
| L30 down c486 | MLP | res%100 in {80} (-) | 0.92 | 0.028 |
| L30 down c23 | MLP | res%100 in {84..88} (+) | 0.55 | 0.040 |
| L30 down c364 | MLP | res%100 in {84} (-) | 0.92 | 0.081 |
| L30 down c357 | MLP | res%100 in {85..87} (-) | 0.82 | 0.039 |
| L30 down c501 | MLP | res%100 in {91..92} (+) | 0.82 | 0.048 |
| L30 down c370 | MLP | res%100 in {92..94} (-) | 0.83 | 0.078 |
| L30 down c535 | MLP | res%100 in {97..99} (-) | 0.87 | 0.046 |

</details>

</details>

<details><summary><b>4a-513</b> `res%100` @ `=` (add) — block code; 19 comps, L31; tells apart 31/100 classes (best member 7); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.28 |
| classes told apart: joint / best code / best member | 31 /  / 7 (of 100) |
| members whose removal merges classes | 0.79 |
| support overlap (1 = tiling) / random sets / p | 1.71 / 1.42 / 0.97 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.83 |
| decoding acc. joint / best code / best member (chance) | 0.21 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.26 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L31 attn |
| CKA(arrangement before, joint write) | 0.35 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.12 2:0.09 10:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.15 2:0.08 5:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.13 / 4.55 |

<details><summary>codes and components</summary>

**code 4a-513.0 (L31): 19 comps, tells apart 31/100, coverage 0.28, overlap 1.71 (random 1.43, p 0.94)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L31 down c56 | MLP | res%100 in {11} (-) | 0.63 | 0.008 |
| L31 down c128 | MLP | res%100 in {11} (-) | 0.63 | 0.007 |
| L31 down c196 | MLP | res%100 in {1} (+) | 0.73 | 0.009 |
| L31 down c239 | MLP | res%100 in {21, 61, 81, 91} (+) | 0.79 | 0.039 |
| L31 down c211 | MLP | res%100 in {34} (+) | 0.84 | 0.011 |
| L31 down c813 | MLP | res%100 in {4..9} (+) | 0.78 | 0.094 |
| L31 down c998 | MLP | res%100 in {5..7} (+) | 0.79 | 0.028 |
| L31 down c522 | MLP | res%100 in {71, 81} (+) | 0.81 | 0.019 |
| L31 down c495 | MLP | res%100 in {76..78} (-) | 0.93 | 0.032 |
| L31 down c623 | MLP | res%100 in {77..79, 81} (+) | 0.89 | 0.044 |
| L31 down c296 | MLP | res%100 in {81..83} (+) | 0.91 | 0.035 |
| L31 down c219 | MLP | res%100 in {82..86} (-) | 0.84 | 0.055 |
| L31 down c126 | MLP | res%100 in {83} (-) | 0.78 | 0.010 |
| L31 down c247 | MLP | res%100 in {84..85, 89} (-) | 0.75 | 0.029 |
| L31 down c355 | MLP | res%100 in {86..89} (+) | 0.77 | 0.065 |
| L31 down c65 | MLP | res%100 in {97..98} (+) | 0.67 | 0.039 |
| L31 down c527 | MLP | res%100 in {97..98} (-) | 0.81 | 0.017 |
| L31 down c53 | MLP | res%100 in {97} (+) | 0.57 | 0.011 |
| L31 down c484 | MLP | res%100 in {98} (+) | 0.70 | 0.012 |

</details>

</details>

</details>

<details><summary>`res`: 14 mechanisms, 238 components</summary>

<details><summary><b>4a-480</b> `res` @ `=` (add) — single component; 1 comps, L18; tells apart 5/199 classes (best member 5); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.04 |
| classes told apart: joint / best code / best member | 5 /  / 5 (of 199) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.69 |
| decoding acc. joint / best code / best member (chance) | 0.02 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L18 attn |
| CKA(arrangement before, joint write) | 0.18 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.00 |

<details><summary>codes and components</summary>

**code 4a-480.0 (L18): 1 comps, tells apart 5/199, coverage 0.04, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 down c34 | MLP | res in {2..7, 9, 13} (-) | 0.68 | 0.006 |

</details>

</details>

<details><summary><b>4a-481</b> `res` @ `=` (add) — single component; 1 comps, L20; tells apart 7/199 classes (best member 7); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.03 |
| classes told apart: joint / best code / best member | 7 /  / 7 (of 199) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.71 |
| decoding acc. joint / best code / best member (chance) | 0.01 /  / 0.01 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L20 attn |
| CKA(arrangement before, joint write) | 0.12 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.11 |
| write energy / code energy before | 0.00 |

<details><summary>codes and components</summary>

**code 4a-481.0 (L20): 1 comps, tells apart 7/199, coverage 0.03, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 down c223 | MLP | res in {113..118} (+) | 0.71 | 0.048 |

</details>

</details>

<details><summary><b>4a-482</b> `res` @ `=` (add) — block code; 9 comps, L21; tells apart 46/199 classes (best member 8); on sub: 4s-613 (member overlap 0.06)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.49 |
| classes told apart: joint / best code / best member | 46 /  / 8 (of 199) |
| members whose removal merges classes | 0.78 |
| support overlap (1 = tiling) / random sets / p | 1.29 / 1.19 / 0.80 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.81 |
| decoding acc. joint / best code / best member (chance) | 0.24 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.04 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 attn |
| CKA(arrangement before, joint write) | 0.43 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.02 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.04 |

<details><summary>codes and components</summary>

**code 4a-482.0 (L21): 9 comps, tells apart 46/199, coverage 0.49, overlap 1.29 (random 1.20, p 0.75)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c57 | MLP | res in {106..119} (+) | 0.77 | 0.177 |
| L21 down c92 | MLP | res in {131..142} (-) | 0.93 | 0.095 |
| L21 down c279 | MLP | res in {133..137} (+) | 0.70 | 0.067 |
| L21 down c125 | MLP | res in {138..142, 146..150} (+) | 0.80 | 0.128 |
| L21 down c133 | MLP | res in {165..176} (+) | 0.81 | 0.151 |
| L21 down c38 | MLP | res in {2..3, 200} (-) | 0.60 | 1.000 |
| L21 down c17 | MLP | res in {2..7, 9..13, 15, 17, 19, 21..23, 31, 41, 51..53, 55, 57, 59, 61..62, 71, 73, 75, 81, 91..93, 95, 97, 99, 101..103, 111..113, 115, 117, 121} (-) | 0.67 | 0.775 |
| L21 down c390 | MLP | res in {42, 62, 102, 122, 132, 142, 152, 162, 172, 182, 192} (+) | 0.68 | 0.057 |
| L21 down c35 | MLP | res in {62..65, 160..168} (-) | 0.87 | 0.141 |

</details>

</details>

<details><summary><b>4a-483</b> `res` @ `=` (add) — block code; 4 comps, L22; tells apart 12/199 classes (best member 7); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.28 |
| classes told apart: joint / best code / best member | 12 /  / 7 (of 199) |
| members whose removal merges classes | 0.75 |
| support overlap (1 = tiling) / random sets / p | 1.05 / 1.04 / 0.54 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.78 |
| decoding acc. joint / best code / best member (chance) | 0.05 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.04 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L22 attn |
| CKA(arrangement before, joint write) | 0.39 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.09 |
| share of the write inside the old arrangement's span | 0.13 |
| write energy / code energy before | 0.01 |

<details><summary>codes and components</summary>

**code 4a-483.0 (L22): 4 comps, tells apart 12/199, coverage 0.28, overlap 1.05 (random 1.04, p 0.58)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 down c101 | MLP | res in {120..121, 123..127} (+) | 0.84 | 0.078 |
| L22 down c56 | MLP | res in {125..133, 135..137} (+) | 0.63 | 0.099 |
| L22 down c376 | MLP | res in {2..15} (+) | 0.86 | 0.012 |
| L22 down c44 | MLP | res in {66, 68..74, 76, 160..176} (-) | 0.80 | 0.180 |

</details>

</details>

<details><summary><b>4a-484</b> `res` @ `=` (add) — block code; 8 comps, L23; tells apart 23/199 classes (best member 9); on sub: 4s-615 (member overlap 0.05)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.27 |
| classes told apart: joint / best code / best member | 23 /  / 9 (of 199) |
| members whose removal merges classes | 0.88 |
| support overlap (1 = tiling) / random sets / p | 1.11 / 1.16 / 0.33 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.88 |
| decoding acc. joint / best code / best member (chance) | 0.11 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 2 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L23 attn |
| CKA(arrangement before, joint write) | 0.30 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.02 |

<details><summary>codes and components</summary>

**code 4a-484.0 (L23): 8 comps, tells apart 23/199, coverage 0.27, overlap 1.11 (random 1.15, p 0.35)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 down c269 | MLP | res in {112..116} (-) | 0.81 | 0.056 |
| L23 down c12 | MLP | res in {114, 124, 134} (-) | 0.56 | 0.019 |
| L23 down c60 | MLP | res in {120, 130, 140, 150, 160, 170, 180, 190} (-) | 0.88 | 0.038 |
| L23 down c20 | MLP | res in {124..135} (-) | 0.95 | 0.180 |
| L23 down c931 | MLP | res in {152..158, 161..162} (-) | 0.85 | 0.049 |
| L23 down c310 | MLP | res in {83, 93, 123, 143, 153, 163, 183, 193} (-) | 0.59 | 0.041 |
| L23 down c21 | MLP | res in {85..88, 95, 105, 107, 145} (+) | 0.58 | 0.123 |
| L23 down c123 | MLP | res in {92, 101..102, 151..152, 171} (+) | 0.52 | 0.043 |

</details>

</details>

<details><summary><b>4a-485</b> `res` @ `=` (add) — block code; 9 comps, L24; tells apart 34/199 classes (best member 9); on sub: 4s-616 (member overlap 0.08)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.49 |
| classes told apart: joint / best code / best member | 34 /  / 9 (of 199) |
| members whose removal merges classes | 0.89 |
| support overlap (1 = tiling) / random sets / p | 1.16 / 1.18 / 0.47 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.75 |
| decoding acc. joint / best code / best member (chance) | 0.12 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L24 attn |
| CKA(arrangement before, joint write) | 0.32 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.01 |

<details><summary>codes and components</summary>

**code 4a-485.0 (L24): 9 comps, tells apart 34/199, coverage 0.49, overlap 1.16 (random 1.18, p 0.46)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 down c158 | MLP | res in {10..14} (-) | 0.84 | 0.014 |
| L24 down c76 | MLP | res in {116..120, 197..200} (-) | 0.59 | 0.051 |
| L24 down c99 | MLP | res in {128..133} (-) | 0.92 | 0.115 |
| L24 down c294 | MLP | res in {144..149} (+) | 0.86 | 0.041 |
| L24 down c553 | MLP | res in {163..172} (-) | 0.82 | 0.048 |
| L24 down c1 | MLP | res in {19..20, 26..27, 86..89, 91..92, 94..99, 137, 142, 146..149, 151..153, 155..159, 161..162, 167..169, 171..172, 176..179, 181..198} (-) | 0.60 | 0.752 |
| L24 down c110 | MLP | res in {22..29, 32} (-) | 0.73 | 0.024 |
| L24 down c234 | MLP | res in {27..30} (+) | 0.72 | 0.009 |
| L24 down c152 | MLP | res in {75..79} (-) | 0.78 | 0.037 |

</details>

</details>

<details><summary><b>4a-486</b> `res` @ `=` (add) — block code; 5 comps, L25; tells apart 24/199 classes (best member 6); on sub: 4s-617 (member overlap 0.05)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.28 |
| classes told apart: joint / best code / best member | 24 /  / 6 (of 199) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.09 / 1.06 / 0.62 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.83 |
| decoding acc. joint / best code / best member (chance) | 0.08 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.02 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L25 attn |
| CKA(arrangement before, joint write) | 0.41 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.02 |
| share of the write inside the old arrangement's span | 0.09 |
| write energy / code energy before | 0.01 |

<details><summary>codes and components</summary>

**code 4a-486.0 (L25): 5 comps, tells apart 24/199, coverage 0.28, overlap 1.09 (random 1.06, p 0.63)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L25 down c92 | MLP | res in {113..126} (-) | 0.92 | 0.171 |
| L25 down c110 | MLP | res in {116..118} (+) | 0.89 | 0.035 |
| L25 down c51 | MLP | res in {158, 161..169} (-) | 0.85 | 0.111 |
| L25 down c55 | MLP | res in {46..50, 52..60, 62..68, 87..88} (-) | 0.83 | 0.198 |
| L25 down c74 | MLP | res in {64..65, 69..75, 80} (+) | 0.74 | 0.215 |

</details>

</details>

<details><summary><b>4a-487</b> `res` @ `=` (add) — block code; 9 comps, L26; tells apart 35/199 classes (best member 11); on sub: 4s-618 (member overlap 0.11)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.32 |
| classes told apart: joint / best code / best member | 35 /  / 11 (of 199) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.13 / 1.19 / 0.29 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.90 |
| decoding acc. joint / best code / best member (chance) | 0.16 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.02 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L26 attn |
| CKA(arrangement before, joint write) | 0.19 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.02 |

<details><summary>codes and components</summary>

**code 4a-487.0 (L26): 9 comps, tells apart 35/199, coverage 0.32, overlap 1.13 (random 1.19, p 0.30)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 down c4 | MLP | res in {122..128, 135..138} (+) | 0.94 | 0.186 |
| L26 down c95 | MLP | res in {129..132, 135..137, 139..141, 180} (+) | 0.88 | 0.170 |
| L26 down c393 | MLP | res in {134, 137..139} (-) | 0.80 | 0.035 |
| L26 down c360 | MLP | res in {141..145} (-) | 0.85 | 0.046 |
| L26 down c226 | MLP | res in {146..148} (-) | 0.87 | 0.043 |
| L26 down c789 | MLP | res in {179} (+) | 0.55 | 0.007 |
| L26 down c320 | MLP | res in {194..199} (-) | 0.68 | 0.011 |
| L26 down c47 | MLP | res in {2, 16, 20, 36, 56, 60..68, 76, 96..97, 104, 106, 116..117, 156, 196} (-) | 0.61 | 0.225 |
| L26 down c182 | MLP | res in {24..30} (-) | 0.86 | 0.029 |

</details>

</details>

<details><summary><b>4a-488</b> `res` @ `=` (add) — block code; 23 comps, L27; tells apart 108/199 classes (best member 10); on sub: 4s-619 (member overlap 0.09)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.71 |
| classes told apart: joint / best code / best member | 108 /  / 10 (of 199) |
| members whose removal merges classes | 0.96 |
| support overlap (1 = tiling) / random sets / p | 1.63 / 1.58 / 0.60 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.81 |
| decoding acc. joint / best code / best member (chance) | 0.47 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.05 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L27 attn |
| CKA(arrangement before, joint write) | 0.47 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.02 |

<details><summary>codes and components</summary>

**code 4a-488.0 (L27): 23 comps, tells apart 108/199, coverage 0.71, overlap 1.63 (random 1.58, p 0.63)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 down c30 | MLP | res in {108, 116..119, 126..129, 137..139, 147..148, 156..159, 167..169, 177..179, 187..188} (-) | 0.82 | 0.173 |
| L27 down c417 | MLP | res in {111..112} (+) | 0.80 | 0.018 |
| L27 down c312 | MLP | res in {118, 138, 158, 168} (-) | 0.73 | 0.054 |
| L27 down c599 | MLP | res in {119..121, 125} (+) | 0.73 | 0.028 |
| L27 down c835 | MLP | res in {129..133} (+) | 0.92 | 0.055 |
| L27 down c193 | MLP | res in {13..17} (+) | 0.86 | 0.019 |
| L27 down c222 | MLP | res in {131..135} (+) | 0.89 | 0.038 |
| L27 down c51 | MLP | res in {140..144} (-) | 0.95 | 0.090 |
| L27 down c365 | MLP | res in {157} (+) | 0.76 | 0.034 |
| L27 down c818 | MLP | res in {161..162} (+) | 0.78 | 0.016 |
| L27 down c147 | MLP | res in {161..169} (-) | 0.90 | 0.136 |
| L27 down c723 | MLP | res in {180..188} (-) | 0.82 | 0.036 |
| L27 down c2 | MLP | res in {2, 52, 56, 79, 85, 95, 98..100, 105, 108, 115, 121, 136, 142, 152, 154, 156, 162, 172, 181..184} (+) | 0.64 | 1.000 |
| L27 down c167 | MLP | res in {20..26} (+) | 0.78 | 0.022 |
| L27 down c73 | MLP | res in {3, 13..14, 22..24, 33..34, 42..43, 53, 63, 73, 83, 93, 103} (-) | 0.82 | 0.172 |
| L27 down c0 | MLP | res in {3..15, 17, 22..23, 25..27, 31..33, 47, 51..58, 61..64, 87, 98..99, 119..120, 124, 138..140, 143, 145, 149, 152, 154, 169, 175, 178..180, 189, 195} (+) | 0.51 | 1.000 |
| L27 down c254 | MLP | res in {38..43} (+) | 0.85 | 0.047 |
| L27 down c142 | MLP | res in {49, 149} (-) | 0.79 | 0.049 |
| L27 down c133 | MLP | res in {49..67} (+) | 0.91 | 0.168 |
| L27 down c149 | MLP | res in {50..52, 150..151} (-) | 0.81 | 0.028 |
| L27 down c849 | MLP | res in {61..68} (-) | 0.82 | 0.054 |
| L27 down c246 | MLP | res in {73, 133, 173} (-) | 0.82 | 0.057 |
| L27 down c117 | MLP | res in {80..86} (-) | 0.87 | 0.066 |

</details>

</details>

<details><summary><b>4a-489</b> `res` @ `=` (add) — tiling; 31 comps, L28; tells apart 77/199 classes (best member 10); on sub: 4s-620 (member overlap 0.11)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.53 |
| classes told apart: joint / best code / best member | 77 /  / 10 (of 199) |
| members whose removal merges classes | 0.68 |
| support overlap (1 = tiling) / random sets / p | 1.45 / 1.87 / 0.00 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.87 |
| decoding acc. joint / best code / best member (chance) | 0.43 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.13 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L28 attn |
| CKA(arrangement before, joint write) | 0.45 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.03 |

<details><summary>codes and components</summary>

**code 4a-489.0 (L28): 31 comps, tells apart 77/199, coverage 0.53, overlap 1.45 (random 1.86, p 0.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 down c12 | MLP | res in {107..118} (-) | 0.90 | 0.274 |
| L28 down c73 | MLP | res in {10} (+) | 0.82 | 0.027 |
| L28 down c294 | MLP | res in {112..114} (+) | 0.81 | 0.035 |
| L28 down c288 | MLP | res in {113..116} (+) | 0.92 | 0.039 |
| L28 down c218 | MLP | res in {116..121} (-) | 0.94 | 0.075 |
| L28 down c1010 | MLP | res in {13..19} (+) | 0.90 | 0.017 |
| L28 down c377 | MLP | res in {133..135} (+) | 0.88 | 0.076 |
| L28 down c301 | MLP | res in {134..136} (+) | 0.84 | 0.029 |
| L28 down c259 | MLP | res in {137..138, 140..142} (+) | 0.75 | 0.042 |
| L28 down c176 | MLP | res in {138..140} (-) | 0.89 | 0.064 |
| L28 down c309 | MLP | res in {142, 151..152} (+) | 0.50 | 0.015 |
| L28 down c979 | MLP | res in {145..150} (-) | 0.88 | 0.052 |
| L28 down c760 | MLP | res in {150, 170} (+) | 0.76 | 0.012 |
| L28 down c666 | MLP | res in {153..154, 156..161, 163} (+) | 0.87 | 0.055 |
| L28 down c136 | MLP | res in {154..155} (+) | 0.84 | 0.060 |
| L28 down c316 | MLP | res in {154} (-) | 0.71 | 0.007 |
| L28 down c310 | MLP | res in {157, 160..164, 167} (+) | 0.75 | 0.060 |
| L28 down c150 | MLP | res in {159} (-) | 0.80 | 0.018 |
| L28 down c211 | MLP | res in {2, 9..11} (+) | 0.69 | 0.048 |
| L28 down c158 | MLP | res in {2..3, 78, 96..97} (+) | 0.50 | 1.000 |
| L28 down c327 | MLP | res in {20..21} (+) | 0.88 | 0.006 |
| L28 down c236 | MLP | res in {22..24} (+) | 0.90 | 0.019 |
| L28 down c990 | MLP | res in {25..28} (-) | 0.76 | 0.011 |
| L28 down c98 | MLP | res in {28..39} (-) | 0.89 | 0.059 |
| L28 down c304 | MLP | res in {34..38, 135..136} (-) | 0.80 | 0.064 |
| L28 down c593 | MLP | res in {40..42} (+) | 0.80 | 0.020 |
| L28 down c588 | MLP | res in {5} (-) | 0.72 | 0.007 |
| L28 down c752 | MLP | res in {62} (+) | 0.79 | 0.016 |
| L28 down c182 | MLP | res in {76..79} (+) | 0.83 | 0.049 |
| L28 down c247 | MLP | res in {7} (-) | 0.73 | 0.011 |
| L28 down c58 | MLP | res in {99, 104..105, 107..111, 114..115, 119..125, 127, 134..135, 149..150, 179, 184..185, 189..190, 194..195} (+) | 0.77 | 0.357 |

</details>

</details>

<details><summary><b>4a-490</b> `res` @ `=` (add) — tiling; 41 comps, L29; tells apart 103/199 classes (best member 11); on sub: 4s-621 (member overlap 0.08)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.76 |
| classes told apart: joint / best code / best member | 103 /  / 11 (of 199) |
| members whose removal merges classes | 0.59 |
| support overlap (1 = tiling) / random sets / p | 1.81 / 2.21 / 0.01 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.87 |
| decoding acc. joint / best code / best member (chance) | 0.67 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.09 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L29 attn |
| CKA(arrangement before, joint write) | 0.49 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.05 |

<details><summary>codes and components</summary>

**code 4a-490.0 (L29): 41 comps, tells apart 103/199, coverage 0.76, overlap 1.81 (random 2.25, p 0.04)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L29 down c28 | MLP | res in {106..107, 117, 146..147, 156..157, 160, 166..167, 176..177, 187, 196} (-) | 0.73 | 0.057 |
| L29 down c662 | MLP | res in {112..113, 122..123, 132..133, 142..143, 152..153, 173, 182..183, 192..193} (-) | 0.84 | 0.092 |
| L29 down c549 | MLP | res in {112..118} (+) | 0.90 | 0.068 |
| L29 down c232 | MLP | res in {114..116, 124..126, 134..136, 144..146, 154..156, 164..166, 174..176, 184..185} (+) | 0.86 | 0.208 |
| L29 down c880 | MLP | res in {114..120} (+) | 0.88 | 0.061 |
| L29 down c326 | MLP | res in {122, 132} (-) | 0.65 | 0.014 |
| L29 down c42 | MLP | res in {127..137} (-) | 0.93 | 0.165 |
| L29 down c618 | MLP | res in {127} (+) | 0.82 | 0.009 |
| L29 down c160 | MLP | res in {129, 136} (+) | 0.78 | 0.021 |
| L29 down c14 | MLP | res in {130, 140, 142, 146..148, 160, 170, 176..178, 180, 182} (+) | 0.67 | 0.054 |
| L29 down c824 | MLP | res in {133} (+) | 0.91 | 0.024 |
| L29 down c64 | MLP | res in {138} (+) | 0.92 | 0.098 |
| L29 down c217 | MLP | res in {139} (-) | 0.88 | 0.067 |
| L29 down c219 | MLP | res in {149..152, 154} (-) | 0.82 | 0.075 |
| L29 down c762 | MLP | res in {15, 35, 65} (+) | 0.57 | 0.012 |
| L29 down c686 | MLP | res in {151..153} (-) | 0.77 | 0.027 |
| L29 down c92 | MLP | res in {153..154} (-) | 0.90 | 0.113 |
| L29 down c449 | MLP | res in {156..169} (+) | 0.83 | 0.070 |
| L29 down c904 | MLP | res in {187..190, 197..198} (+) | 0.81 | 0.043 |
| L29 down c35 | MLP | res in {2, 40, 60..61, 70, 90} (+) | 0.56 | 0.044 |
| L29 down c4 | MLP | res in {2..4, 89, 117..119, 130, 154, 160, 172..173, 189, 191, 194} (+) | 0.52 | 1.000 |
| L29 down c9 | MLP | res in {2..5} (+) | 0.50 | 1.000 |
| L29 down c279 | MLP | res in {20} (-) | 0.88 | 0.030 |
| L29 down c853 | MLP | res in {37..39} (+) | 0.55 | 0.013 |
| L29 down c893 | MLP | res in {40..53} (-) | 0.90 | 0.115 |
| L29 down c502 | MLP | res in {42..45} (-) | 0.88 | 0.050 |
| L29 down c308 | MLP | res in {47..49, 148} (-) | 0.83 | 0.036 |
| L29 down c171 | MLP | res in {57, 67, 87, 97, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 133, 137, 139, 141, 143, 147, 149, 151, 153, 157, 161, 163, 167, 169, 177, 183, 187} (-) | 0.82 | 0.335 |
| L29 down c300 | MLP | res in {59, 159} (-) | 0.75 | 0.011 |
| L29 down c437 | MLP | res in {60..61} (-) | 0.88 | 0.043 |
| L29 down c480 | MLP | res in {61, 71, 91, 101, 111, 121, 131, 141, 161, 171, 191} (-) | 0.89 | 0.073 |
| L29 down c331 | MLP | res in {62..65} (-) | 0.78 | 0.047 |
| L29 down c166 | MLP | res in {65..69} (+) | 0.80 | 0.044 |
| L29 down c122 | MLP | res in {6} (+) | 0.81 | 0.010 |
| L29 down c144 | MLP | res in {70..75} (+) | 0.90 | 0.076 |
| L29 down c653 | MLP | res in {71, 170..172} (+) | 0.69 | 0.028 |
| L29 down c94 | MLP | res in {7} (-) | 0.55 | 0.007 |
| L29 down c150 | MLP | res in {80..86} (+) | 0.90 | 0.076 |
| L29 down c134 | MLP | res in {8} (+) | 0.53 | 0.007 |
| L29 down c927 | MLP | res in {96, 98, 189..199} (-) | 0.60 | 0.038 |
| L29 down c635 | MLP | res in {9} (-) | 0.64 | 0.022 |

</details>

</details>

<details><summary><b>4a-491</b> `res` @ `=` (add) — tiling; 30 comps, L30 L31; tells apart 78/199 classes (best member 10); on sub: 4s-622 (member overlap 0.13)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.52 |
| classes told apart: joint / best code / best member | 78 /  / 10 (of 199) |
| members whose removal merges classes | 0.70 |
| support overlap (1 = tiling) / random sets / p | 1.55 / 1.80 / 0.04 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.84 |
| decoding acc. joint / best code / best member (chance) | 0.45 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.13 |
| consumers / read jointly | 7 / 2 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L30 attn |
| CKA(arrangement before, joint write) | 0.50 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.02 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.05 |
| best source position (CKA) | `a` (0.19) |

<details><summary>codes and components</summary>

**code 4a-491.0 (L30 L31): 30 comps, tells apart 78/199, coverage 0.52, overlap 1.55 (random 1.81, p 0.04)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 down c293 | MLP | res in {109, 159} (-) | 0.85 | 0.057 |
| L30 down c320 | MLP | res in {112, 114, 116, 118, 122, 124, 126, 128, 132, 134, 136, 138, 144, 148, 152, 154, 156, 158, 162, 164, 166, 168, 172, 174, 176, 178, 188, 194} (+) | 0.74 | 0.215 |
| L30 down c17 | MLP | res in {112..114, 131, 171} (+) | 0.77 | 0.040 |
| L30 down c181 | MLP | res in {116..120} (+) | 0.71 | 0.080 |
| L30 down c879 | MLP | res in {116} (-) | 0.62 | 0.007 |
| L30 down c642 | MLP | res in {118} (-) | 0.57 | 0.006 |
| L30 down c159 | MLP | res in {125..126} (+) | 0.93 | 0.093 |
| L30 down c660 | MLP | res in {125} (-) | 0.84 | 0.043 |
| L30 down c407 | MLP | res in {126..129} (+) | 0.90 | 0.057 |
| L30 down c207 | MLP | res in {128..131} (-) | 0.91 | 0.079 |
| L30 down c121 | MLP | res in {131, 133..134, 136..138} (-) | 0.83 | 0.085 |
| L30 down c465 | MLP | res in {134..136} (-) | 0.81 | 0.070 |
| L30 down c8 | MLP | res in {136..139} (+) | 0.79 | 0.054 |
| L30 down c414 | MLP | res in {142} (-) | 0.89 | 0.018 |
| L30 down c205 | MLP | res in {146..148} (-) | 0.92 | 0.106 |
| L30 down c661 | MLP | res in {147..148} (+) | 0.74 | 0.019 |
| L30 down c700 | MLP | res in {152} (-) | 0.81 | 0.007 |
| L30 down c831 | MLP | res in {158} (-) | 0.92 | 0.024 |
| L30 down c941 | MLP | res in {164..165} (-) | 0.86 | 0.032 |
| L30 down c706 | MLP | res in {25, 27} (+) | 0.59 | 0.017 |
| L30 down c882 | MLP | res in {27, 127} (-) | 0.89 | 0.021 |
| L30 down c192 | MLP | res in {30} (-) | 0.79 | 0.014 |
| L30 down c199 | MLP | res in {44, 144} (-) | 0.91 | 0.084 |
| L30 down c326 | MLP | res in {56, 156..157} (-) | 0.87 | 0.026 |
| L30 down c377 | MLP | res in {6, 14..16, 18..23, 25..26, 36, 43, 47, 50, 66, 74, 84, 98, 101, 104, 117, 124, 127, 132, 134, 142, 144, 148, 152, 156..159, 163..164, 167..169, 174, 179, 182..184, 192} (+) | 0.56 | 0.999 |
| L30 down c394 | MLP | res in {67, 87, 127, 167, 187} (-) | 0.91 | 0.029 |
| L30 down c322 | MLP | res in {70..76} (-) | 0.93 | 0.087 |
| L30 down c1012 | MLP | res in {79, 179} (+) | 0.88 | 0.018 |
| L30 down c251 | MLP | res in {81, 111..112, 121..122, 141, 161, 181..182} (-) | 0.86 | 0.086 |
| L31 o c135 | H12 | res in {2..5, 7, 117} (+) | 0.53 | 1.000 |

</details>

</details>

<details><summary><b>4a-492</b> `res` @ `=` (add) — single component; 1 comps, L31; tells apart 2/199 classes (best member 2); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.07 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 199) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.58 |
| decoding acc. joint / best code / best member (chance) | 0.03 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L30 MLP |
| CKA(arrangement before, joint write) | 0.21 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.00 |
| best source position (CKA) | `op` (0.34) |

<details><summary>codes and components</summary>

**code 4a-492.0 (L31): 1 comps, tells apart 2/199, coverage 0.07, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L31 o c7 | H14 | res in {2..4, 117, 149, 168..169, 179, 186..189, 191, 193} (+) | 0.57 | 1.000 |

</details>

</details>

<details><summary><b>4a-493</b> `res` @ `=` (add) — block code; 66 comps, L31; tells apart 125/199 classes (best member 10); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.96 |
| classes told apart: joint / best code / best member | 125 /  / 10 (of 199) |
| members whose removal merges classes | 0.30 |
| support overlap (1 = tiling) / random sets / p | 4.07 / 3.14 / 0.99 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.74 |
| decoding acc. joint / best code / best member (chance) | 0.53 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.35 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L31 attn |
| CKA(arrangement before, joint write) | 0.60 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.13 |
| write energy / code energy before | 0.02 |

<details><summary>codes and components</summary>

**code 4a-493.0 (L31): 66 comps, tells apart 125/199, coverage 0.96, overlap 4.07 (random 3.16, p 0.99)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L31 down c652 | MLP | res in {101, 111, 121, 131, 141, 151, 161, 171, 181, 191} (+) | 0.91 | 0.064 |
| L31 down c387 | MLP | res in {105, 135, 155, 160, 165, 185} (+) | 0.76 | 0.040 |
| L31 down c899 | MLP | res in {105..130} (+) | 0.92 | 0.250 |
| L31 down c640 | MLP | res in {107, 111..115} (+) | 0.80 | 0.227 |
| L31 down c587 | MLP | res in {109, 113, 131, 133, 142..143, 153..154, 163..164, 167, 169, 171..172, 182..184, 186..189, 191..194, 197} (+) | 0.59 | 0.347 |
| L31 down c137 | MLP | res in {109, 189, 194..199} (+) | 0.65 | 0.018 |
| L31 down c500 | MLP | res in {111, 131, 151} (+) | 0.80 | 0.022 |
| L31 down c675 | MLP | res in {112..114} (+) | 0.88 | 0.027 |
| L31 down c927 | MLP | res in {113, 115, 117} (-) | 0.86 | 0.030 |
| L31 down c314 | MLP | res in {113..114, 134, 152..154} (+) | 0.80 | 0.045 |
| L31 down c992 | MLP | res in {115..117} (-) | 0.87 | 0.037 |
| L31 down c466 | MLP | res in {115..119} (+) | 0.91 | 0.041 |
| L31 down c426 | MLP | res in {116..118, 124..139, 156, 166, 176..177} (+) | 0.87 | 0.173 |
| L31 down c319 | MLP | res in {118} (-) | 0.83 | 0.009 |
| L31 down c531 | MLP | res in {121..124} (-) | 0.94 | 0.032 |
| L31 down c580 | MLP | res in {123, 128, 132..133, 138, 142..143} (+) | 0.85 | 0.055 |
| L31 down c194 | MLP | res in {124} (+) | 0.64 | 0.006 |
| L31 down c293 | MLP | res in {131..134, 136..139} (+) | 0.65 | 0.063 |
| L31 down c867 | MLP | res in {137..138, 147..148, 173..174} (+) | 0.81 | 0.040 |
| L31 down c926 | MLP | res in {139..146, 149} (-) | 0.91 | 0.060 |
| L31 down c108 | MLP | res in {140..145} (-) | 0.86 | 0.034 |
| L31 down c455 | MLP | res in {141..142} (-) | 0.66 | 0.011 |
| L31 down c859 | MLP | res in {141} (-) | 0.79 | 0.006 |
| L31 down c301 | MLP | res in {144..149} (-) | 0.77 | 0.029 |
| L31 down c188 | MLP | res in {146, 148..149, 195..196, 198..199} (-) | 0.62 | 0.016 |
| L31 down c165 | MLP | res in {148, 150..158} (+) | 0.79 | 0.058 |
| L31 down c27 | MLP | res in {148, 151, 157..158, 161, 168, 171, 173..174, 177..179, 184, 188, 197..199} (+) | 0.67 | 0.128 |
| L31 down c132 | MLP | res in {149..159, 161, 171, 191} (-) | 0.89 | 0.091 |
| L31 down c85 | MLP | res in {150, 155, 160, 170, 175} (+) | 0.53 | 0.015 |
| L31 down c125 | MLP | res in {151..153} (+) | 0.51 | 0.010 |
| L31 down c48 | MLP | res in {154, 156, 164} (-) | 0.50 | 0.018 |
| L31 down c271 | MLP | res in {154..156} (+) | 0.62 | 0.011 |
| L31 down c847 | MLP | res in {162..164, 166..169} (-) | 0.85 | 0.048 |
| L31 down c612 | MLP | res in {163, 168..169} (-) | 0.68 | 0.030 |
| L31 down c79 | MLP | res in {163..167, 169} (-) | 0.72 | 0.019 |
| L31 down c423 | MLP | res in {169} (-) | 0.81 | 0.008 |
| L31 down c244 | MLP | res in {174..200} (-) | 0.91 | 0.036 |
| L31 down c147 | MLP | res in {175..176, 178} (-) | 0.67 | 0.025 |
| L31 down c756 | MLP | res in {2, 12, 16, 18..22, 26, 30, 35..36, 38..43, 49, 51..62, 66, 76, 82, 126, 148, 168..169, 171, 174, 179..181, 183..185, 187, 189, 191..192} (+) | 0.76 | 0.987 |
| L31 down c754 | MLP | res in {2, 8, 10..31, 33..40, 44, 47..48, 50, 77, 80, 197..200} (-) | 0.58 | 0.264 |
| L31 down c259 | MLP | res in {2, 9, 11..12, 15, 18..32, 34..35, 39..51, 56, 59..61, 63..70, 80, 83, 89, 109, 148, 171..172, 178, 197, 200} (-) | 0.58 | 0.473 |
| L31 down c23 | MLP | res in {2..9, 171} (+) | 0.56 | 1.000 |
| L31 down c213 | MLP | res in {3..5, 8, 14, 17..18, 26..29, 42, 44..49, 51, 54..61, 63..68} (-) | 0.58 | 0.184 |
| L31 down c174 | MLP | res in {32, 52, 54..56, 58..69, 72, 74..76, 79, 82..86, 90..92, 96} (+) | 0.62 | 0.429 |
| L31 down c6 | MLP | res in {4..11, 14..15, 18, 20..22, 24, 34, 87, 116, 134, 152..153, 156, 159, 161..164, 167..173} (-) | 0.68 | 1.000 |
| L31 down c250 | MLP | res in {46..49} (+) | 0.78 | 0.017 |
| L31 down c176 | MLP | res in {59, 79, 99, 119, 139, 149, 159, 179, 199} (-) | 0.87 | 0.062 |
| L31 down c748 | MLP | res in {59..60, 79..80, 119..120, 159..160, 170, 179..180} (+) | 0.80 | 0.104 |
| L31 down c902 | MLP | res in {60..65} (-) | 0.82 | 0.044 |
| L31 down c184 | MLP | res in {66..68} (-) | 0.80 | 0.028 |
| L31 down c680 | MLP | res in {7..9, 11..31, 33..35, 37..38, 40..44, 46, 48, 51, 54, 56, 58, 61..64, 66..68, 71, 74, 76..78, 81..88} (+) | 0.76 | 0.309 |
| L31 down c51 | MLP | res in {71, 138, 143, 171} (+) | 0.70 | 0.072 |
| L31 down c180 | MLP | res in {71} (-) | 0.82 | 0.009 |
| L31 down c595 | MLP | res in {75, 115, 125, 135, 155, 175} (+) | 0.67 | 0.045 |
| L31 down c54 | MLP | res in {76, 86, 106, 116, 126, 134..136, 146, 154, 156, 164..166, 176, 186} (+) | 0.82 | 0.123 |
| L31 down c444 | MLP | res in {81, 111, 121, 131, 141, 151, 161, 171, 181} (+) | 0.83 | 0.049 |
| L31 down c321 | MLP | res in {81, 83..85} (+) | 0.55 | 0.040 |
| L31 down c546 | MLP | res in {85, 115, 135, 145, 155, 165, 185} (-) | 0.85 | 0.060 |
| L31 down c700 | MLP | res in {85, 135, 175, 185} (+) | 0.68 | 0.024 |
| L31 down c73 | MLP | res in {86} (-) | 0.57 | 0.010 |
| L31 down c160 | MLP | res in {87..88, 107..108, 117..118, 127..128, 133, 137..138, 147..148, 157..158, 167..168, 173..174, 177..178, 183, 187..188} (-) | 0.83 | 0.161 |
| L31 down c242 | MLP | res in {89, 177..179, 187..190, 192, 194..200} (+) | 0.84 | 0.052 |
| L31 down c1023 | MLP | res in {90, 170..171, 180..181, 183..199} (+) | 0.83 | 0.076 |
| L31 down c315 | MLP | res in {97..99, 193..199} (+) | 0.84 | 0.053 |
| L31 down c921 | MLP | res in {98, 117..120, 122} (+) | 0.78 | 0.079 |
| L31 down c44 | MLP | res in {99, 118..120, 122, 124..129} (+) | 0.85 | 0.102 |

</details>

</details>

</details>

<details><summary>`tens(a,b)`: 31 mechanisms, 138 components</summary>

<details><summary><b>4a-535</b> `tens(a,b)` @ `=` (add) — single component; 1 comps, L1; tells apart 3/121 classes (best member 3); on sub: 4s-639 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.01 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.87 |
| decoding acc. joint / best code / best member (chance) | 0.02 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.09 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.06 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.02 |
| best source position (CKA) | `op` (0.06) |

<details><summary>codes and components</summary>

**code 4a-535.0 (L1): 1 comps, tells apart 3/121, coverage 0.01, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c377 | H26 | a//10 {0} x b//10 {0} (-) | 0.87 | 0.009 |

</details>

</details>

<details><summary><b>4a-541</b> `tens(a,b)` @ `=` (add) — 2 codes of the same shape; 2 comps, L1 L9; tells apart 2/121 classes (best member 3); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.15 |
| classes told apart: joint / best code / best member | 2 / 3 / 3 (of 121) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 1.50 / 1.00 / 0.98 |
| mean CKA between its codes | 0.75 |
| purity of the joint write (per prompt) | 0.72 |
| decoding acc. joint / best code / best member (chance) | 0.06 / 0.05 / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.99 |
| consumers / read jointly | 3 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.49 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 14.88 |
| best source position (CKA) | `b` (0.29) |

<details><summary>codes and components</summary>

**code 4a-541.0 (L1): 1 comps, tells apart 3/121, coverage 0.10, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c370 | H6 | a//10 {0} x b//10 {0..4}; a//10 {1..4, 6..7, 10} x b//10 {0} (+) | 0.75 | 1.000 |

**code 4a-541.1 (L9): 1 comps, tells apart 2/121, coverage 0.12, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L9 o c409 | H22 | a//10 {0} x b//10 {0..5, 9..10}; a//10 {1..3, 9..10} x b//10 {0}; a//10 {4..5} x b//10 {8} (+) | 0.72 | 0.970 |

</details>

</details>

<details><summary><b>4a-536</b> `tens(a,b)` @ `=` (add) — block code, copy from `b`; 2 comps, L3 L4; tells apart 7/121 classes (best member 6); on sub: 4s-640 (member overlap 0.14)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.10 |
| classes told apart: joint / best code / best member | 7 /  / 6 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.02 / 0.48 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.64 |
| decoding acc. joint / best code / best member (chance) | 0.06 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 MLP |
| CKA(arrangement before, joint write) | 0.42 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.03 |
| best source position (CKA) | `b` (0.51) |

<details><summary>codes and components</summary>

**code 4a-536.0 (L3 L4): 2 comps, tells apart 7/121, coverage 0.10, overlap 1.00 (random 1.00, p 0.52)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 o c354 | H5 | a//10 {0, 2..3} x b//10 {0}; a//10 {1} x b//10 {0..1} (-) | 0.67 | 0.066 |
| L4 o c23 | H25 | a//10 {6..7, 9} x b//10 {0}; a//10 {10} x b//10 {0..1, 3..4} (+) | 0.61 | 1.000 |

</details>

</details>

<details><summary><b>4a-537</b> `tens(a,b)` @ `=` (add) — single component; 1 comps, L4; tells apart 7/121 classes (best member 7); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.08 |
| classes told apart: joint / best code / best member | 7 /  / 7 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.77 |
| decoding acc. joint / best code / best member (chance) | 0.02 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L4 attn |
| CKA(arrangement before, joint write) | 0.48 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.05 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.06 |

<details><summary>codes and components</summary>

**code 4a-537.0 (L4): 1 comps, tells apart 7/121, coverage 0.08, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 down c4 | MLP | a//10 {0} x b//10 {0..5, 9..10}; a//10 {1..2} x b//10 {0} (+) | 0.77 | 0.115 |

</details>

</details>

<details><summary><b>4a-538</b> `tens(a,b)` @ `=` (add) — single component; 1 comps, L5; tells apart 2/121 classes (best member 2); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.21 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.61 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L4 MLP |
| CKA(arrangement before, joint write) | 0.48 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.08 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.02 |
| best source position (CKA) | `b` (0.43) |

<details><summary>codes and components</summary>

**code 4a-538.0 (L5): 1 comps, tells apart 2/121, coverage 0.21, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 o c141 | H30 | a//10 {0} x b//10 {0..1, 10}; a//10 {1} x b//10 {1, 10}; a//10 {2} x b//10 {1..2, 10}; a//10 {3} x b//10 {2..3, 10}; a//10 {4} x b//10 {2}; a//10 {6, 9} x b//10 {7}; a//10 {7} x b//10 {8..9}; a//10 {8} x b//10 {7, 9}; a//10 {10} x b//10 {3..10} (-) | 0.61 | 0.977 |

</details>

</details>

<details><summary><b>4a-539</b> `tens(a,b)` @ `=` (add) — block code, 2 codes of the same shape; 5 comps, L6 L31; tells apart 3/121 classes (best member 3); on sub: 4s-646 (member overlap 0.11)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.47 |
| classes told apart: joint / best code / best member | 3 / 3 / 3 (of 121) |
| members whose removal merges classes | 0.20 |
| support overlap (1 = tiling) / random sets / p | 1.61 / 1.28 / 0.94 |
| mean CKA between its codes | 0.73 |
| purity of the joint write (per prompt) | 0.67 |
| decoding acc. joint / best code / best member (chance) | 0.09 / 0.09 / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.56 |
| consumers / read jointly | 2 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L6 attn |
| CKA(arrangement before, joint write) | 0.63 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 26.93 |

<details><summary>codes and components</summary>

**code 4a-539.0 (L6): 1 comps, tells apart 2/121, coverage 0.12, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L6 down c0 | MLP | a//10 {0} x b//10 {0..1, 3..6}; a//10 {1} x b//10 {0..4}; a//10 {2} x b//10 {0, 2..4} (+) | 0.73 | 1.000 |

**code 4a-539.1 (L31): 4 comps, tells apart 3/121, coverage 0.42, overlap 1.51 (random 1.21, p 0.90)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L31 down c9 | MLP | a//10 {0..1} x b//10 {0..2}; a//10 {2} x b//10 {0..1, 5}; a//10 {6..7} x b//10 {9}; a//10 {8} x b//10 {2..4, 6..9}; a//10 {9} x b//10 {0..10}; a//10 {10} x b//10 {3..7, 9} (+) | 0.62 | 1.000 |
| L31 down c47 | MLP | a//10 {0} x b//10 {0..2}; a//10 {1} x b//10 {0..1}; a//10 {2} x b//10 {0} (-) | 0.54 | 1.000 |
| L31 down c29 | MLP | a//10 {0} x b//10 {0..4, 10}; a//10 {1..2} x b//10 {0..2}; a//10 {3..5} x b//10 {0}; a//10 {7} x b//10 {9}; a//10 {9} x b//10 {7}; a//10 {10} x b//10 {6, 10} (-) | 0.74 | 0.994 |
| L31 down c134 | MLP | a//10 {0} x b//10 {8..9}; a//10 {2} x b//10 {6}; a//10 {5} x b//10 {3}; a//10 {6} x b//10 {2..3}; a//10 {8} x b//10 {0..1, 8..9}; a//10 {9} x b//10 {0..1, 4..5, 7..9} (+) | 0.56 | 0.315 |

</details>

</details>

<details><summary><b>4a-540</b> `tens(a,b)` @ `=` (add) — block code, 10 codes of the same shape; 15 comps, L7 L8 L9 L10 L11 L12 L13 L14; tells apart 23/121 classes (best member 4); on sub: 4s-642 (member overlap 0.07)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.57 |
| classes told apart: joint / best code / best member | 23 / 5 / 4 (of 121) |
| members whose removal merges classes | 0.73 |
| support overlap (1 = tiling) / random sets / p | 4.75 / 2.22 / 1.00 |
| mean CKA between its codes | 0.86 |
| purity of the joint write (per prompt) | 0.69 |
| decoding acc. joint / best code / best member (chance) | 0.20 / 0.09 / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.73 |
| consumers / read jointly | 53 / 52 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L6 MLP |
| CKA(arrangement before, joint write) | 0.68 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.02 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 2.56 |
| best source position (CKA) | `b` (0.46) |

<details><summary>codes and components</summary>

**code 4a-540.0 (L7): 2 comps, tells apart 4/121, coverage 0.15, overlap 1.17 (random 1.00, p 0.85)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L7 down c1 | MLP | a//10 {0} x b//10 {0, 10}; a//10 {1} x b//10 {0}; a//10 {9} x b//10 {0, 9}; a//10 {10} x b//10 {10} (+) | 0.52 | 1.000 |
| L7 down c28 | MLP | a//10 {0} x b//10 {0..6, 9..10}; a//10 {1} x b//10 {0..1}; a//10 {2} x b//10 {0..2}; a//10 {3} x b//10 {0} (-) | 0.82 | 0.502 |

**code 4a-540.1 (L7): 1 comps, tells apart 3/121, coverage 0.02, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L7 o c11 | H23 | a//10 {0..1} x b//10 {0} (+) | 0.74 | 0.990 |

**code 4a-540.2 (L8): 1 comps, tells apart 2/121, coverage 0.20, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L8 down c84 | MLP | a//10 {0} x b//10 {0..6, 8..10}; a//10 {1..2} x b//10 {0..3, 10}; a//10 {3..4, 9} x b//10 {0}; a//10 {10} x b//10 {10} (+) | 0.74 | 0.354 |

**code 4a-540.3 (L8): 1 comps, tells apart 2/121, coverage 0.07, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L8 o c289 | H16 | a//10 {0} x b//10 {0..1, 4, 10}; a//10 {1} x b//10 {0..1}; a//10 {2} x b//10 {0}; a//10 {10} x b//10 {10} (+) | 0.66 | 0.970 |

**code 4a-540.4 (L9): 1 comps, tells apart 1/121, coverage 0.45, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L9 down c7 | MLP | a//10 {0} x b//10 {0..10}; a//10 {1} x b//10 {0..5, 10}; a//10 {2} x b//10 {0..2, 10}; a//10 {3} x b//10 {0}; a//10 {4} x b//10 {0, 7..8}; a//10 {5..6} x b//10 {0, 7..9}; a//10 {7} x b//10 {0, 3..9}; a//10 {8} x b//10 {0, 5..9}; a//10 {9} x b//10 {0, 6..8}; a//10 {10} x b//10 {0, 10} (+) | 0.77 | 0.809 |

**code 4a-540.5 (L10): 1 comps, tells apart 1/121, coverage 0.21, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L10 down c255 | MLP | a//10 {0} x b//10 {0..1, 3..4, 9..10}; a//10 {1..2} x b//10 {0..1}; a//10 {3..4, 9} x b//10 {0}; a//10 {5} x b//10 {0, 8}; a//10 {6} x b//10 {7..9}; a//10 {7} x b//10 {6, 8..9}; a//10 {8} x b//10 {6..7}; a//10 {10} x b//10 {0, 10} (+) | 0.68 | 0.884 |

**code 4a-540.6 (L11): 2 comps, tells apart 5/121, coverage 0.26, overlap 1.23 (random 1.03, p 0.93)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L11 down c11 | MLP | a//10 {0..5, 9} x b//10 {0}; a//10 {10} x b//10 {0..1} (+) | 0.58 | 0.989 |
| L11 down c437 | MLP | a//10 {0} x b//10 {0..6, 8..10}; a//10 {1..2} x b//10 {0..3, 10}; a//10 {3} x b//10 {0, 3, 10}; a//10 {4} x b//10 {0, 4}; a//10 {5..6, 9} x b//10 {0}; a//10 {10} x b//10 {10} (-) | 0.69 | 0.346 |

**code 4a-540.7 (L12): 2 comps, tells apart 1/121, coverage 0.49, overlap 1.15 (random 1.04, p 0.81)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L12 down c4 | MLP | a//10 {0} x b//10 {0..10}; a//10 {1} x b//10 {0..4, 10}; a//10 {2} x b//10 {0..2, 10}; a//10 {3} x b//10 {0, 3, 10}; a//10 {4} x b//10 {0, 4, 7..8}; a//10 {5} x b//10 {0, 7..8}; a//10 {6} x b//10 {0, 7..9}; a//10 {7} x b//10 {0, 3..6, 8..9}; a//10 {8} x b//10 {0, 5..7}; a//10 {9} x b//10 {0}; a//10 {10} x b//10 {0, 10} (+) | 0.73 | 0.856 |
| L12 down c14 | MLP | a//10 {4} x b//10 {1}; a//10 {5..6} x b//10 {7..9}; a//10 {7} x b//10 {8..9}; a//10 {8} x b//10 {7, 9}; a//10 {9} x b//10 {9}; a//10 {10} x b//10 {0, 4..9} (-) | 0.53 | 1.000 |

**code 4a-540.8 (L13): 2 comps, tells apart 3/121, coverage 0.24, overlap 1.62 (random 1.03, p 1.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 down c4 | MLP | a//10 {0} x b//10 {0..1, 3..6, 8..10}; a//10 {1} x b//10 {0, 10}; a//10 {2..6, 9} x b//10 {0}; a//10 {10} x b//10 {0..4, 10} (-) | 0.58 | 0.999 |
| L13 down c5 | MLP | a//10 {0} x b//10 {0..6, 8..10}; a//10 {1} x b//10 {0..1, 10}; a//10 {2} x b//10 {0..2, 10}; a//10 {3} x b//10 {0, 10}; a//10 {4..6, 9} x b//10 {0}; a//10 {10} x b//10 {10} (-) | 0.72 | 0.427 |

**code 4a-540.9 (L14): 2 comps, tells apart 4/121, coverage 0.23, overlap 1.46 (random 1.03, p 0.97)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L14 down c1 | MLP | a//10 {0} x b//10 {0, 4, 8..10}; a//10 {1} x b//10 {0, 10}; a//10 {2..5, 9..10} x b//10 {0} (-) | 0.56 | 1.000 |
| L14 down c23 | MLP | a//10 {0} x b//10 {0..10}; a//10 {1} x b//10 {0..1, 10}; a//10 {2} x b//10 {0..2, 10}; a//10 {3, 10} x b//10 {0, 10}; a//10 {4..9} x b//10 {0} (+) | 0.75 | 0.377 |

</details>

</details>

<details><summary><b>4a-542</b> `tens(a,b)` @ `=` (add) — single component; 1 comps, L12; tells apart 2/121 classes (best member 2); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.31 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.65 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L11 MLP |
| CKA(arrangement before, joint write) | 0.22 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.00 |
| best source position (CKA) | `b` (0.41) |

<details><summary>codes and components</summary>

**code 4a-542.0 (L12): 1 comps, tells apart 2/121, coverage 0.31, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L12 o c209 | H20 | a//10 {0} x b//10 {0..9}; a//10 {2..3} x b//10 {1, 10}; a//10 {4..5} x b//10 {0..1, 7..8}; a//10 {6} x b//10 {0, 7..8}; a//10 {7} x b//10 {0..1, 8..9}; a//10 {8} x b//10 {0}; a//10 {9} x b//10 {0..3, 9..10}; a//10 {10} x b//10 {4, 6} (-) | 0.65 | 1.000 |

</details>

</details>

<details><summary><b>4a-543</b> `tens(a,b)` @ `=` (add) — single component; 1 comps, L13; tells apart 1/121 classes (best member 1); on sub: 4s-651 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.22 |
| classes told apart: joint / best code / best member | 1 /  / 1 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.64 |
| decoding acc. joint / best code / best member (chance) | 0.05 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L12 MLP |
| CKA(arrangement before, joint write) | 0.70 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.09 |
| write energy / code energy before | 0.00 |
| best source position (CKA) | `b` (0.44) |

<details><summary>codes and components</summary>

**code 4a-543.0 (L13): 1 comps, tells apart 1/121, coverage 0.22, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 o c0 | H16 | a//10 {0} x b//10 {0, 10}; a//10 {1} x b//10 {0..1, 9..10}; a//10 {2..3} x b//10 {0..1}; a//10 {4..5, 10} x b//10 {0}; a//10 {6} x b//10 {0, 6, 8}; a//10 {7} x b//10 {0, 8}; a//10 {8} x b//10 {0, 7..10}; a//10 {9} x b//10 {6..8, 10} (-) | 0.63 | 1.000 |

</details>

</details>

<details><summary><b>4a-544</b> `tens(a,b)` @ `=` (add) — block code, copy from `b`; 3 comps, L13 L14; tells apart 6/121 classes (best member 4); on sub: 4s-653 (member overlap 0.33)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.42 |
| classes told apart: joint / best code / best member | 6 /  / 4 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.22 / 1.12 / 0.70 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.75 |
| decoding acc. joint / best code / best member (chance) | 0.10 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 3 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L12 MLP |
| CKA(arrangement before, joint write) | 0.54 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.14 |
| share of the write inside the old arrangement's span | 0.20 |
| write energy / code energy before | 0.09 |
| best source position (CKA) | `b` (0.74) |

<details><summary>codes and components</summary>

**code 4a-544.0 (L13 L14): 3 comps, tells apart 6/121, coverage 0.42, overlap 1.22 (random 1.16, p 0.65)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 o c1 | H7 | a//10 {0} x b//10 {0..10}; a//10 {1} x b//10 {0, 9}; a//10 {2..9} x b//10 {0}; a//10 {10} x b//10 {0, 10} (+) | 0.62 | 1.000 |
| L13 o c414 | H7 | a//10 {1} x b//10 {0, 10}; a//10 {2} x b//10 {0}; a//10 {3..5} x b//10 {0..1}; a//10 {6..8} x b//10 {0..2}; a//10 {9} x b//10 {0..3}; a//10 {10} x b//10 {0..5} (+) | 0.84 | 0.263 |
| L14 o c59 | H26 | a//10 {0} x b//10 {5}; a//10 {6} x b//10 {6, 8}; a//10 {7} x b//10 {8}; a//10 {8} x b//10 {6..9}; a//10 {9} x b//10 {6, 8..9} (+) | 0.52 | 1.000 |

</details>

</details>

<details><summary><b>4a-545</b> `tens(a,b)` @ `=` (add) — block code; 2 comps, L14 L15; tells apart 10/121 classes (best member 4); on sub: 4s-654 (member overlap 0.50)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.26 |
| classes told apart: joint / best code / best member | 10 /  / 4 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.03 / 1.00 / 0.55 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.64 |
| decoding acc. joint / best code / best member (chance) | 0.06 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 2 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L13 MLP |
| CKA(arrangement before, joint write) | 0.65 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.09 |
| write energy / code energy before | 0.05 |
| best source position (CKA) | `b` (0.40) |

<details><summary>codes and components</summary>

**code 4a-545.0 (L14 L15): 2 comps, tells apart 10/121, coverage 0.26, overlap 1.03 (random 1.03, p 0.51)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L14 o c84 | H31 | a//10 {0} x b//10 {0..10}; a//10 {1..5} x b//10 {0}; a//10 {9} x b//10 {0, 6..7}; a//10 {10} x b//10 {0, 10} (+) | 0.64 | 1.000 |
| L15 o c119 | H13 | a//10 {5, 7} x b//10 {7}; a//10 {6} x b//10 {6..7}; a//10 {8..9} x b//10 {7..9}; a//10 {10} x b//10 {9} (+) | 0.62 | 0.217 |

</details>

</details>

<details><summary><b>4a-546</b> `tens(a,b)` @ `=` (add) — block code; 4 comps, L15; tells apart 29/121 classes (best member 4); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.47 |
| classes told apart: joint / best code / best member | 29 /  / 4 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.12 / 1.21 / 0.28 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.63 |
| decoding acc. joint / best code / best member (chance) | 0.09 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.05 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 attn |
| CKA(arrangement before, joint write) | 0.62 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.13 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.09 |

<details><summary>codes and components</summary>

**code 4a-546.0 (L15): 4 comps, tells apart 29/121, coverage 0.47, overlap 1.12 (random 1.20, p 0.27)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c18 | MLP | a//10 {0} x b//10 {0..1, 10}; a//10 {1} x b//10 {0..1}; a//10 {2..4, 9} x b//10 {0}; a//10 {10} x b//10 {0, 10} (-) | 0.52 | 0.224 |
| L15 down c9 | MLP | a//10 {0} x b//10 {2..10}; a//10 {1} x b//10 {3..10}; a//10 {10} x b//10 {0} (-) | 0.71 | 0.195 |
| L15 down c15 | MLP | a//10 {2..3} x b//10 {4}; a//10 {4} x b//10 {3, 5}; a//10 {5} x b//10 {3..4}; a//10 {6, 9..10} x b//10 {3..5}; a//10 {7..8} x b//10 {2..5} (+) | 0.67 | 0.271 |
| L15 down c11 | MLP | a//10 {3..8} x b//10 {0}; a//10 {9} x b//10 {0..1}; a//10 {10} x b//10 {0..3} (-) | 0.66 | 0.141 |

</details>

</details>

<details><summary><b>4a-547</b> `tens(a,b)` @ `=` (add) — block code; 8 comps, L16; tells apart 63/121 classes (best member 4); on sub: 4s-657 (member overlap 0.06)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.81 |
| classes told apart: joint / best code / best member | 63 /  / 4 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.52 / 1.59 / 0.41 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.92 |
| decoding acc. joint / best code / best member (chance) | 0.65 /  / 0.07 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.07 |
| consumers / read jointly | 17 / 12 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L16 attn |
| CKA(arrangement before, joint write) | 0.61 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.13 |
| share of the write inside the old arrangement's span | 0.14 |
| write energy / code energy before | 0.21 |

<details><summary>codes and components</summary>

**code 4a-547.0 (L16): 8 comps, tells apart 63/121, coverage 0.81, overlap 1.52 (random 1.53, p 0.47)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c61 | MLP | a//10 {0..2} x b//10 {0..4}; a//10 {3} x b//10 {0..3}; a//10 {4} x b//10 {1..2} (+) | 0.94 | 0.399 |
| L16 down c18 | MLP | a//10 {0} x b//10 {0..4, 10}; a//10 {1} x b//10 {0..3, 7..8}; a//10 {2..3} x b//10 {0..3, 7..9}; a//10 {4} x b//10 {0, 4, 6..9}; a//10 {5..6} x b//10 {0, 7..9}; a//10 {7} x b//10 {0, 8..9}; a//10 {9} x b//10 {0}; a//10 {10} x b//10 {0, 10} (-) | 0.82 | 0.856 |
| L16 down c130 | MLP | a//10 {0} x b//10 {6..7}; a//10 {9..10} x b//10 {5..8} (+) | 0.89 | 0.245 |
| L16 down c43 | MLP | a//10 {1..2} x b//10 {4..7}; a//10 {3} x b//10 {5..6}; a//10 {5} x b//10 {8..9}; a//10 {10} x b//10 {6} (+) | 0.89 | 0.518 |
| L16 down c25 | MLP | a//10 {4, 6..7} x b//10 {5..7}; a//10 {5} x b//10 {5..8} (-) | 0.93 | 0.337 |
| L16 down c106 | MLP | a//10 {5, 8} x b//10 {1..3}; a//10 {6..7} x b//10 {1..4}; a//10 {9} x b//10 {1} (-) | 0.92 | 0.350 |
| L16 down c29 | MLP | a//10 {6} x b//10 {5..6}; a//10 {7} x b//10 {4..6}; a//10 {8..9} x b//10 {3..6}; a//10 {10} x b//10 {4..5} (+) | 0.92 | 0.501 |
| L16 down c10 | MLP | a//10 {6} x b//10 {7..9}; a//10 {7..8} x b//10 {3, 6..9}; a//10 {9} x b//10 {6..8} (-) | 0.92 | 0.481 |

</details>

</details>

<details><summary><b>4a-548</b> `tens(a,b)` @ `=` (add) — block code; 9 comps, L17; tells apart 80/121 classes (best member 4); on sub: 4s-657 (member overlap 0.06)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.79 |
| classes told apart: joint / best code / best member | 80 /  / 4 (of 121) |
| members whose removal merges classes | 0.89 |
| support overlap (1 = tiling) / random sets / p | 1.61 / 1.65 / 0.43 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.86 |
| decoding acc. joint / best code / best member (chance) | 0.55 /  / 0.07 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.07 |
| consumers / read jointly | 11 / 6 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L17 attn |
| CKA(arrangement before, joint write) | 0.68 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.12 |
| share of the write inside the old arrangement's span | 0.13 |
| write energy / code energy before | 0.22 |

<details><summary>codes and components</summary>

**code 4a-548.0 (L17): 9 comps, tells apart 80/121, coverage 0.79, overlap 1.61 (random 1.67, p 0.40)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 down c119 | MLP | a//10 {0} x b//10 {0..4, 10}; a//10 {1} x b//10 {0..1}; a//10 {2} x b//10 {0, 2}; a//10 {3} x b//10 {0}; a//10 {10} x b//10 {10} (+) | 0.69 | 0.279 |
| L17 down c235 | MLP | a//10 {0} x b//10 {7..9}; a//10 {1} x b//10 {6..10}; a//10 {2} x b//10 {6..9}; a//10 {3} x b//10 {7} (-) | 0.87 | 0.168 |
| L17 down c107 | MLP | a//10 {1, 7} x b//10 {4..5}; a//10 {2, 6} x b//10 {4}; a//10 {8..10} x b//10 {3..5} (+) | 0.84 | 0.242 |
| L17 down c40 | MLP | a//10 {1..2} x b//10 {4..6}; a//10 {3} x b//10 {3..6}; a//10 {4} x b//10 {2..5}; a//10 {5} x b//10 {2..4}; a//10 {6} x b//10 {3}; a//10 {9} x b//10 {1..3}; a//10 {10} x b//10 {1..2} (-) | 0.89 | 0.472 |
| L17 down c23 | MLP | a//10 {1} x b//10 {4, 8..9}; a//10 {2} x b//10 {1, 3..4, 8..10}; a//10 {3} x b//10 {1, 4, 8..9}; a//10 {6} x b//10 {1, 3..4, 8..9}; a//10 {7} x b//10 {1, 3..4, 6, 8..10}; a//10 {8} x b//10 {1, 4, 6, 9} (-) | 0.85 | 0.653 |
| L17 down c3 | MLP | a//10 {1} x b//10 {5, 9}; a//10 {3} x b//10 {1, 4..5, 10}; a//10 {4} x b//10 {5}; a//10 {5, 7} x b//10 {7..8}; a//10 {6} x b//10 {6..8}; a//10 {8..9} x b//10 {7} (+) | 0.59 | 1.000 |
| L17 down c122 | MLP | a//10 {2..7} x b//10 {6..7}; a//10 {8} x b//10 {6} (-) | 0.79 | 0.192 |
| L17 down c11 | MLP | a//10 {5} x b//10 {8..9}; a//10 {6..7} x b//10 {7..10}; a//10 {8} x b//10 {8..10}; a//10 {9} x b//10 {8} (+) | 0.94 | 0.461 |
| L17 down c15 | MLP | a//10 {6, 9} x b//10 {3}; a//10 {7} x b//10 {1..6}; a//10 {8} x b//10 {2..4, 6}; a//10 {10} x b//10 {2..7} (+) | 0.82 | 0.444 |

</details>

</details>

<details><summary><b>4a-549</b> `tens(a,b)` @ `=` (add) — block code; 3 comps, L18; tells apart 15/121 classes (best member 3); on sub: 4s-659 (member overlap 0.20)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.25 |
| classes told apart: joint / best code / best member | 15 /  / 3 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.12 / 0.14 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.79 |
| decoding acc. joint / best code / best member (chance) | 0.12 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.99 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L17 MLP |
| CKA(arrangement before, joint write) | 0.43 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.01 |
| best source position (CKA) | `b` (0.35) |

<details><summary>codes and components</summary>

**code 4a-549.0 (L18): 3 comps, tells apart 15/121, coverage 0.25, overlap 1.00 (random 1.12, p 0.14)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 o c166 | H18 | a//10 {0, 3..4, 10} x b//10 {5}; a//10 {5} x b//10 {2..5, 10} (-) | 0.80 | 0.177 |
| L18 o c324 | H18 | a//10 {0..1} x b//10 {9..10}; a//10 {3} x b//10 {10}; a//10 {8} x b//10 {9}; a//10 {9..10} x b//10 {0, 9..10} (+) | 0.70 | 0.208 |
| L18 o c442 | H1 | a//10 {0..2} x b//10 {0..2} (-) | 0.83 | 0.954 |

</details>

</details>

<details><summary><b>4a-550</b> `tens(a,b)` @ `=` (add) — tiling; 10 comps, L18; tells apart 33/121 classes (best member 6); on sub: 4s-661 (member overlap 0.67)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.48 |
| classes told apart: joint / best code / best member | 33 /  / 6 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.22 / 1.72 / 0.01 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.83 |
| decoding acc. joint / best code / best member (chance) | 0.29 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.95 |
| consumers / read jointly | 1 / 1 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L17 MLP |
| CKA(arrangement before, joint write) | 0.47 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.11 |
| share of the write inside the old arrangement's span | 0.15 |
| write energy / code energy before | 0.03 |
| best source position (CKA) | `b` (0.34) |

<details><summary>codes and components</summary>

**code 4a-550.0 (L18): 10 comps, tells apart 33/121, coverage 0.48, overlap 1.22 (random 1.74, p 0.01)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 o c427 | H30 | a//10 {0..1} x b//10 {2}; a//10 {2} x b//10 {0, 2} (+) | 0.52 | 0.046 |
| L18 o c107 | H30 | a//10 {0..1} x b//10 {8..10}; a//10 {2} x b//10 {9}; a//10 {8} x b//10 {0..1, 8..10}; a//10 {9} x b//10 {0..1, 7..10}; a//10 {10} x b//10 {0, 8..10} (-) | 0.87 | 0.438 |
| L18 o c120 | H30 | a//10 {0..2, 10} x b//10 {10} (+) | 0.75 | 0.016 |
| L18 o c82 | H30 | a//10 {0..2, 4} x b//10 {3}; a//10 {3} x b//10 {0, 3..4} (+) | 0.81 | 0.177 |
| L18 o c402 | H30 | a//10 {0..2, 4} x b//10 {4} (+) | 0.51 | 0.107 |
| L18 o c137 | H30 | a//10 {0..3} x b//10 {4}; a//10 {4} x b//10 {0, 3..4} (+) | 0.72 | 0.134 |
| L18 o c73 | H30 | a//10 {0..5} x b//10 {6}; a//10 {6} x b//10 {0..3, 6..7} (-) | 0.82 | 0.342 |
| L18 o c66 | H30 | a//10 {0} x b//10 {1..2}; a//10 {1..2} x b//10 {0..2} (-) | 0.89 | 0.302 |
| L18 o c42 | H30 | a//10 {0} x b//10 {5} (-) | 0.51 | 0.040 |
| L18 o c355 | H30 | a//10 {0} x b//10 {7}; a//10 {7} x b//10 {0} (+) | 0.61 | 0.039 |

</details>

</details>

<details><summary><b>4a-551</b> `tens(a,b)` @ `=` (add) — block code; 20 comps, L18; tells apart 116/121 classes (best member 10); on sub: 4s-662 (member overlap 0.58)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.95 |
| classes told apart: joint / best code / best member | 116 /  / 10 (of 121) |
| members whose removal merges classes | 0.50 |
| support overlap (1 = tiling) / random sets / p | 2.44 / 2.76 / 0.23 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.82 |
| decoding acc. joint / best code / best member (chance) | 0.75 /  / 0.06 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.08 |
| consumers / read jointly | 93 / 93 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L18 attn |
| CKA(arrangement before, joint write) | 0.53 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.18 |
| share of the write inside the old arrangement's span | 0.25 |
| write energy / code energy before | 0.32 |

<details><summary>codes and components</summary>

**code 4a-551.0 (L18): 20 comps, tells apart 116/121, coverage 0.95, overlap 2.44 (random 2.72, p 0.18)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 down c66 | MLP | a//10 {0..1, 8} x b//10 {9..10}; a//10 {9..10} x b//10 {0..2, 7..10} (+) | 0.89 | 0.316 |
| L18 down c54 | MLP | a//10 {0..9} x b//10 {10}; a//10 {10} x b//10 {5, 10} (+) | 0.72 | 0.018 |
| L18 down c1 | MLP | a//10 {0} x b//10 {0..1, 3..4, 10}; a//10 {1..2, 4} x b//10 {0}; a//10 {3} x b//10 {0, 3}; a//10 {5, 9} x b//10 {0, 10}; a//10 {6} x b//10 {0, 9..10}; a//10 {7} x b//10 {10}; a//10 {8} x b//10 {6, 10}; a//10 {10} x b//10 {0, 2, 10} (-) | 0.53 | 1.000 |
| L18 down c61 | MLP | a//10 {0} x b//10 {1, 5..6}; a//10 {1, 10} x b//10 {5..6}; a//10 {5} x b//10 {0..1, 5..6, 10}; a//10 {6} x b//10 {0..1, 5} (+) | 0.78 | 0.325 |
| L18 down c0 | MLP | a//10 {0} x b//10 {1..4}; a//10 {1} x b//10 {0, 2..4}; a//10 {2} x b//10 {0..4}; a//10 {3} x b//10 {0, 2..3}; a//10 {4..5} x b//10 {0, 6..9}; a//10 {6} x b//10 {0, 5..10}; a//10 {7, 9..10} x b//10 {6..10}; a//10 {8} x b//10 {5..10} (-) | 0.82 | 1.000 |
| L18 down c21 | MLP | a//10 {0} x b//10 {2}; a//10 {1} x b//10 {1..2, 7}; a//10 {5..6} x b//10 {2..3, 7} (+) | 0.80 | 0.697 |
| L18 down c25 | MLP | a//10 {1, 7} x b//10 {3, 8}; a//10 {2} x b//10 {8}; a//10 {5} x b//10 {9}; a//10 {6} x b//10 {3..4, 8..9} (+) | 0.80 | 0.780 |
| L18 down c196 | MLP | a//10 {1} x b//10 {7..8}; a//10 {2} x b//10 {6..8}; a//10 {3} x b//10 {7} (-) | 0.77 | 0.091 |
| L18 down c79 | MLP | a//10 {2..10} x b//10 {0} (+) | 0.84 | 0.073 |
| L18 down c91 | MLP | a//10 {2} x b//10 {3..4}; a//10 {3} x b//10 {2..5}; a//10 {4} x b//10 {2..4} (-) | 0.89 | 0.274 |
| L18 down c30 | MLP | a//10 {3, 10} x b//10 {5}; a//10 {7..8} x b//10 {3..5}; a//10 {9} x b//10 {4..5} (-) | 0.75 | 0.263 |
| L18 down c36 | MLP | a//10 {3, 8} x b//10 {1..2, 6..7}; a//10 {4} x b//10 {1}; a//10 {9} x b//10 {1, 6..7} (+) | 0.79 | 0.581 |
| L18 down c27 | MLP | a//10 {3, 8} x b//10 {3..4, 8..9}; a//10 {4, 9} x b//10 {3, 8} (+) | 0.81 | 0.333 |
| L18 down c111 | MLP | a//10 {3..4} x b//10 {0..1}; a//10 {5} x b//10 {0} (+) | 0.73 | 0.126 |
| L18 down c12 | MLP | a//10 {4} x b//10 {2, 7, 9..10}; a//10 {5} x b//10 {1..2, 4, 6..7, 9..10}; a//10 {6} x b//10 {9}; a//10 {9} x b//10 {2, 7}; a//10 {10} x b//10 {2, 4, 7, 9..10} (+) | 0.81 | 0.784 |
| L18 down c22 | MLP | a//10 {4} x b//10 {3..4, 8..10}; a//10 {5, 9..10} x b//10 {3..4, 8..9} (+) | 0.82 | 0.652 |
| L18 down c23 | MLP | a//10 {4} x b//10 {5..6}; a//10 {5} x b//10 {4..6}; a//10 {6} x b//10 {4..5} (+) | 0.88 | 0.519 |
| L18 down c132 | MLP | a//10 {5, 9..10} x b//10 {7..8}; a//10 {6} x b//10 {7..9}; a//10 {7..8} x b//10 {6..10} (+) | 0.95 | 0.335 |
| L18 down c625 | MLP | a//10 {6..8} x b//10 {10}; a//10 {9..10} x b//10 {9..10} (+) | 0.73 | 0.145 |
| L18 down c118 | MLP | a//10 {9} x b//10 {0..1, 7..9} (-) | 0.54 | 0.060 |

</details>

</details>

<details><summary><b>4a-552</b> `tens(a,b)` @ `=` (add) — single component; 1 comps, L19; tells apart 1/121 classes (best member 1); on sub: 4s-660 (member overlap 0.33)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.21 |
| classes told apart: joint / best code / best member | 1 /  / 1 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.57 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L18 MLP |
| CKA(arrangement before, joint write) | 0.53 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.03 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.00 |
| best source position (CKA) | `a` (0.45) |

<details><summary>codes and components</summary>

**code 4a-552.0 (L19): 1 comps, tells apart 1/121, coverage 0.21, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L19 o c23 | H7 | a//10 {0} x b//10 {0..2, 5..6}; a//10 {1} x b//10 {1}; a//10 {2} x b//10 {0..2}; a//10 {4..5} x b//10 {7}; a//10 {6} x b//10 {7, 10}; a//10 {7, 9} x b//10 {7..10}; a//10 {8} x b//10 {6..10} (+) | 0.56 | 1.000 |

</details>

</details>

<details><summary><b>4a-553</b> `tens(a,b)` @ `=` (add) — tiling; 9 comps, L19; tells apart 59/121 classes (best member 7); on sub: 4s-657 (member overlap 0.05)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.64 |
| classes told apart: joint / best code / best member | 59 /  / 7 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.23 / 1.64 / 0.01 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.80 |
| decoding acc. joint / best code / best member (chance) | 0.40 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.02 |
| consumers / read jointly | 7 / 5 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L19 attn |
| CKA(arrangement before, joint write) | 0.42 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.02 |
| share of the write inside the old arrangement's span | 0.21 |
| write energy / code energy before | 0.09 |

<details><summary>codes and components</summary>

**code 4a-553.0 (L19): 9 comps, tells apart 59/121, coverage 0.64, overlap 1.23 (random 1.69, p 0.01)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L19 down c94 | MLP | a//10 {0} x b//10 {0, 8..10}; a//10 {5..9} x b//10 {0}; a//10 {10} x b//10 {0, 10} (+) | 0.65 | 0.134 |
| L19 down c1 | MLP | a//10 {0} x b//10 {0..1, 10}; a//10 {1} x b//10 {0..1}; a//10 {2..3} x b//10 {0}; a//10 {10} x b//10 {10} (+) | 0.65 | 1.000 |
| L19 down c45 | MLP | a//10 {0} x b//10 {0..4, 9..10}; a//10 {1} x b//10 {0..2, 10}; a//10 {2} x b//10 {0..1} (+) | 0.79 | 0.541 |
| L19 down c33 | MLP | a//10 {1} x b//10 {7..8}; a//10 {2} x b//10 {6..7}; a//10 {3} x b//10 {6}; a//10 {4} x b//10 {5}; a//10 {5} x b//10 {2..4}; a//10 {6} x b//10 {1..3}; a//10 {7} x b//10 {1..2}; a//10 {8} x b//10 {1}; a//10 {10} x b//10 {8} (+) | 0.75 | 0.330 |
| L19 down c147 | MLP | a//10 {1} x b//10 {9..10}; a//10 {2} x b//10 {7..10}; a//10 {9} x b//10 {10} (-) | 0.72 | 0.089 |
| L19 down c423 | MLP | a//10 {2..3} x b//10 {1..3} (+) | 0.83 | 0.111 |
| L19 down c11 | MLP | a//10 {2} x b//10 {3, 7..8}; a//10 {3} x b//10 {2, 7..8}; a//10 {7} x b//10 {3}; a//10 {8} x b//10 {2..3, 7}; a//10 {9} x b//10 {2} (-) | 0.79 | 0.464 |
| L19 down c151 | MLP | a//10 {3} x b//10 {9}; a//10 {4, 8} x b//10 {9..10}; a//10 {9} x b//10 {3..5, 9..10}; a//10 {10} x b//10 {4, 9} (-) | 0.74 | 0.091 |
| L19 down c28 | MLP | a//10 {4} x b//10 {6..8}; a//10 {5..6} x b//10 {5..7}; a//10 {7} x b//10 {4..6} (-) | 0.90 | 0.568 |

</details>

</details>

<details><summary><b>4a-554</b> `tens(a,b)` @ `=` (add) — tiling; 8 comps, L20; tells apart 38/121 classes (best member 9); on sub: 4s-657 (member overlap 0.02)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.33 |
| classes told apart: joint / best code / best member | 38 /  / 9 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.25 / 1.57 / 0.04 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.73 |
| decoding acc. joint / best code / best member (chance) | 0.20 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.03 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L20 attn |
| CKA(arrangement before, joint write) | 0.36 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.04 |

<details><summary>codes and components</summary>

**code 4a-554.0 (L20): 8 comps, tells apart 38/121, coverage 0.33, overlap 1.25 (random 1.57, p 0.04)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 down c80 | MLP | a//10 {0} x b//10 {9}; a//10 {8} x b//10 {9..10}; a//10 {9} x b//10 {0, 8..10}; a//10 {10} x b//10 {8..10} (-) | 0.81 | 0.096 |
| L20 down c14 | MLP | a//10 {1} x b//10 {7}; a//10 {3} x b//10 {5}; a//10 {4} x b//10 {4, 9}; a//10 {5} x b//10 {3}; a//10 {6} x b//10 {2}; a//10 {8} x b//10 {9..10}; a//10 {9} x b//10 {4, 9..10}; a//10 {10} x b//10 {8..9} (+) | 0.72 | 0.649 |
| L20 down c266 | MLP | a//10 {3} x b//10 {3} (+) | 0.56 | 0.034 |
| L20 down c180 | MLP | a//10 {3} x b//10 {9}; a//10 {4} x b//10 {8}; a//10 {5} x b//10 {7}; a//10 {8} x b//10 {4} (-) | 0.63 | 0.109 |
| L20 down c106 | MLP | a//10 {4} x b//10 {9}; a//10 {5} x b//10 {8..9} (+) | 0.50 | 0.034 |
| L20 down c136 | MLP | a//10 {5} x b//10 {7}; a//10 {6..7} x b//10 {6..7} (-) | 0.71 | 0.075 |
| L20 down c101 | MLP | a//10 {7} x b//10 {7..10}; a//10 {8} x b//10 {7, 10}; a//10 {10} x b//10 {7} (+) | 0.85 | 0.142 |
| L20 down c172 | MLP | a//10 {9} x b//10 {1..3, 6}; a//10 {10} x b//10 {1..3} (+) | 0.73 | 0.049 |

</details>

</details>

<details><summary><b>4a-555</b> `tens(a,b)` @ `=` (add) — copy from `a`; 1 comps, L21; tells apart 1/121 classes (best member 1); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.27 |
| classes told apart: joint / best code / best member | 1 /  / 1 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.60 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L20 MLP |
| CKA(arrangement before, joint write) | 0.50 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.06 |
| share of the write inside the old arrangement's span | 0.16 |
| write energy / code energy before | 0.00 |
| best source position (CKA) | `a` (0.53) |

<details><summary>codes and components</summary>

**code 4a-555.0 (L21): 1 comps, tells apart 1/121, coverage 0.27, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 o c222 | H18 | a//10 {0} x b//10 {0..5, 7..10}; a//10 {1} x b//10 {0..1, 9..10}; a//10 {2} x b//10 {0, 10}; a//10 {5} x b//10 {8}; a//10 {6} x b//10 {8..9}; a//10 {7} x b//10 {6, 8..9}; a//10 {8} x b//10 {2, 6..7}; a//10 {9} x b//10 {6..7, 10}; a//10 {10} x b//10 {3, 5..8} (-) | 0.59 | 1.000 |

</details>

</details>

<details><summary><b>4a-562</b> `tens(a,b)` @ `=` (add) — block code, 2 codes of the same shape; 9 comps, L21 L26 L27; tells apart 29/121 classes (best member 7); on sub: 4s-668 (member overlap 0.09)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.46 |
| classes told apart: joint / best code / best member | 29 / 18 / 7 (of 121) |
| members whose removal merges classes | 0.78 |
| support overlap (1 = tiling) / random sets / p | 1.41 / 1.63 / 0.15 |
| mean CKA between its codes | 0.71 |
| purity of the joint write (per prompt) | 0.73 |
| decoding acc. joint / best code / best member (chance) | 0.17 / 0.10 / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.06 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 attn |
| CKA(arrangement before, joint write) | 0.51 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.03 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.07 |

<details><summary>codes and components</summary>

**code 4a-562.0 (L21): 4 comps, tells apart 18/121, coverage 0.18, overlap 1.23 (random 1.20, p 0.55)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c206 | MLP | a//10 {0..1} x b//10 {1..2}; a//10 {2} x b//10 {1} (-) | 0.76 | 0.060 |
| L21 down c85 | MLP | a//10 {0} x b//10 {0..3, 10}; a//10 {1} x b//10 {0..1}; a//10 {2..3} x b//10 {0} (+) | 0.67 | 0.160 |
| L21 down c49 | MLP | a//10 {4} x b//10 {9..10}; a//10 {5} x b//10 {8..9}; a//10 {6} x b//10 {7..8}; a//10 {7} x b//10 {6}; a//10 {8} x b//10 {5}; a//10 {9} x b//10 {4} (-) | 0.79 | 0.163 |
| L21 down c99 | MLP | a//10 {5} x b//10 {9..10}; a//10 {6} x b//10 {8..9} (+) | 0.67 | 0.206 |

**code 4a-562.1 (L26 L27): 5 comps, tells apart 10/121, coverage 0.39, overlap 1.11 (random 1.29, p 0.08)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 down c115 | MLP | a//10 {0} x b//10 {2..3}; a//10 {1} x b//10 {2}; a//10 {3} x b//10 {0} (-) | 0.58 | 0.033 |
| L26 down c14 | MLP | a//10 {1} x b//10 {3}; a//10 {4} x b//10 {7, 10}; a//10 {6} x b//10 {3, 7}; a//10 {7, 10} x b//10 {8, 10}; a//10 {8} x b//10 {8..10}; a//10 {9} x b//10 {8..9} (-) | 0.55 | 1.000 |
| L26 down c641 | MLP | a//10 {8..9} x b//10 {10}; a//10 {10} x b//10 {8..10} (+) | 0.62 | 0.012 |
| L27 down c68 | MLP | a//10 {1} x b//10 {4..7}; a//10 {2} x b//10 {4, 6}; a//10 {3} x b//10 {2, 5}; a//10 {4, 6} x b//10 {1..2}; a//10 {5} x b//10 {1..3}; a//10 {7} x b//10 {1} (+) | 0.56 | 0.256 |
| L27 down c25 | MLP | a//10 {4..5} x b//10 {9..10}; a//10 {6} x b//10 {7..8}; a//10 {7} x b//10 {6..7}; a//10 {8} x b//10 {5..6}; a//10 {9} x b//10 {4}; a//10 {10} x b//10 {4..5} (-) | 0.79 | 0.169 |

</details>

</details>

<details><summary><b>4a-557</b> `tens(a,b)` @ `=` (add) — single component; 1 comps, L22; tells apart 2/121 classes (best member 2); on sub: 4s-657 (member overlap 0.01)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.15 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.53 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L22 attn |
| CKA(arrangement before, joint write) | 0.25 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.13 |
| write energy / code energy before | 0.00 |

<details><summary>codes and components</summary>

**code 4a-557.0 (L22): 1 comps, tells apart 2/121, coverage 0.15, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 down c0 | MLP | a//10 {0} x b//10 {0, 4..5}; a//10 {1, 4..5, 9} x b//10 {10}; a//10 {2, 10} x b//10 {5}; a//10 {6} x b//10 {3, 8}; a//10 {7} x b//10 {2..3, 6}; a//10 {8} x b//10 {1..3, 10} (+) | 0.52 | 1.000 |

</details>

</details>

<details><summary><b>4a-556</b> `tens(a,b)` @ `=` (add) — block code; 8 comps, L22; tells apart 23/121 classes (best member 7); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.36 |
| classes told apart: joint / best code / best member | 23 /  / 7 (of 121) |
| members whose removal merges classes | 0.62 |
| support overlap (1 = tiling) / random sets / p | 1.37 / 1.55 / 0.21 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.80 |
| decoding acc. joint / best code / best member (chance) | 0.16 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.93 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 MLP |
| CKA(arrangement before, joint write) | 0.16 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.14 |
| write energy / code energy before | 0.01 |
| best source position (CKA) | `op` (0.38) |

<details><summary>codes and components</summary>

**code 4a-556.0 (L22): 8 comps, tells apart 23/121, coverage 0.36, overlap 1.37 (random 1.58, p 0.17)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 o c31 | H15 | a//10 {0..1} x b//10 {1} (-) | 0.53 | 0.037 |
| L22 o c10 | H15 | a//10 {0..2, 8} x b//10 {9..10}; a//10 {3..4, 7} x b//10 {10}; a//10 {9} x b//10 {0..10}; a//10 {10} x b//10 {0..1, 9..10} (-) | 0.90 | 0.295 |
| L22 o c257 | H15 | a//10 {0..3, 9..10} x b//10 {10} (-) | 0.58 | 0.037 |
| L22 o c41 | H15 | a//10 {0..3} x b//10 {2..3} (+) | 0.65 | 0.155 |
| L22 o c145 | H15 | a//10 {0..3} x b//10 {3} (+) | 0.65 | 0.082 |
| L22 o c135 | H15 | a//10 {0..5, 10} x b//10 {10} (-) | 0.76 | 0.012 |
| L22 o c116 | H15 | a//10 {0} x b//10 {5}; a//10 {5} x b//10 {0, 5} (-) | 0.62 | 0.063 |
| L22 o c32 | H15 | a//10 {0} x b//10 {8}; a//10 {8} x b//10 {0, 8} (+) | 0.73 | 0.041 |

</details>

</details>

<details><summary><b>4a-558</b> `tens(a,b)` @ `=` (add) — block code; 4 comps, L23; tells apart 15/121 classes (best member 7); on sub: 4s-665 (member overlap 0.08)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.29 |
| classes told apart: joint / best code / best member | 15 /  / 7 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.31 / 1.21 / 0.73 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.71 |
| decoding acc. joint / best code / best member (chance) | 0.11 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L23 attn |
| CKA(arrangement before, joint write) | 0.51 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.00 |

<details><summary>codes and components</summary>

**code 4a-558.0 (L23): 4 comps, tells apart 15/121, coverage 0.29, overlap 1.31 (random 1.21, p 0.73)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 down c41 | MLP | a//10 {0, 3} x b//10 {1..2}; a//10 {1} x b//10 {0..3}; a//10 {2} x b//10 {0..2} (-) | 0.85 | 0.149 |
| L23 down c13 | MLP | a//10 {0..1, 5} x b//10 {10}; a//10 {8} x b//10 {9..10}; a//10 {9..10} x b//10 {8..10} (+) | 0.69 | 0.069 |
| L23 down c465 | MLP | a//10 {0} x b//10 {0, 10}; a//10 {10} x b//10 {10} (-) | 0.58 | 0.026 |
| L23 down c28 | MLP | a//10 {0} x b//10 {0..3, 6, 10}; a//10 {1} x b//10 {0..1}; a//10 {2} x b//10 {0, 2}; a//10 {3..5} x b//10 {0}; a//10 {6} x b//10 {3..4}; a//10 {7} x b//10 {2..3}; a//10 {8} x b//10 {1..2}; a//10 {9} x b//10 {1}; a//10 {10} x b//10 {10} (+) | 0.66 | 1.000 |

</details>

</details>

<details><summary><b>4a-559</b> `tens(a,b)` @ `=` (add) — single component; 1 comps, L24; tells apart 1/121 classes (best member 1); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.26 |
| classes told apart: joint / best code / best member | 1 /  / 1 (of 121) |
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
| input point | L23 MLP |
| CKA(arrangement before, joint write) | 0.46 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.05 |
| share of the write inside the old arrangement's span | 0.15 |
| write energy / code energy before | 0.00 |
| best source position (CKA) | `b` (0.31) |

<details><summary>codes and components</summary>

**code 4a-559.0 (L24): 1 comps, tells apart 1/121, coverage 0.26, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 o c23 | H13 | a//10 {0..1} x b//10 {0..5, 10}; a//10 {2} x b//10 {0..4, 10}; a//10 {3} x b//10 {0..3, 10}; a//10 {4..5} x b//10 {0..1}; a//10 {6} x b//10 {0, 6}; a//10 {7} x b//10 {8} (-) | 0.56 | 0.679 |

</details>

</details>

<details><summary><b>4a-560</b> `tens(a,b)` @ `=` (add) — block code; 2 comps, L24; tells apart 10/121 classes (best member 8); on sub: 4s-666 (member overlap 0.17)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.10 |
| classes told apart: joint / best code / best member | 10 /  / 8 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.53 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.70 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L24 attn |
| CKA(arrangement before, joint write) | 0.25 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.00 |

<details><summary>codes and components</summary>

**code 4a-560.0 (L24): 2 comps, tells apart 10/121, coverage 0.10, overlap 1.00 (random 1.03, p 0.48)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 down c207 | MLP | a//10 {0, 10} x b//10 {9}; a//10 {9} x b//10 {0, 9..10} (+) | 0.63 | 0.064 |
| L24 down c70 | MLP | a//10 {0} x b//10 {3..4}; a//10 {1} x b//10 {2..3}; a//10 {3} x b//10 {0..1}; a//10 {4} x b//10 {0} (+) | 0.72 | 0.093 |

</details>

</details>

<details><summary><b>4a-561</b> `tens(a,b)` @ `=` (add) — block code; 2 comps, L25; tells apart 14/121 classes (best member 4); on sub: 4s-665 (member overlap 0.09)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.39 |
| classes told apart: joint / best code / best member | 14 /  / 4 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.06 / 1.00 / 0.68 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.57 |
| decoding acc. joint / best code / best member (chance) | 0.05 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.98 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L25 attn |
| CKA(arrangement before, joint write) | 0.51 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.09 |
| write energy / code energy before | 0.00 |

<details><summary>codes and components</summary>

**code 4a-561.0 (L25): 2 comps, tells apart 14/121, coverage 0.39, overlap 1.06 (random 1.00, p 0.64)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L25 down c6 | MLP | a//10 {0} x b//10 {0..5}; a//10 {1} x b//10 {0, 3..4}; a//10 {2} x b//10 {0..1, 3..4}; a//10 {3} x b//10 {0..2, 7..8}; a//10 {4} x b//10 {0..1, 6..8}; a//10 {5} x b//10 {0..1, 5..7}; a//10 {6} x b//10 {6..7}; a//10 {7} x b//10 {7, 10}; a//10 {8} x b//10 {2, 7, 9}; a//10 {9} x b//10 {7}; a//10 {10} x b//10 {0..1, 3..4, 7, 10} (+) | 0.54 | 1.000 |
| L25 down c79 | MLP | a//10 {2} x b//10 {8}; a//10 {3} x b//10 {7}; a//10 {7} x b//10 {3..4}; a//10 {8} x b//10 {2..3}; a//10 {10} x b//10 {1..2} (-) | 0.61 | 0.198 |

</details>

</details>

<details><summary><b>4a-563</b> `tens(a,b)` @ `=` (add) — single component; 1 comps, L27; tells apart 2/121 classes (best member 2); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.22 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.53 |
| decoding acc. joint / best code / best member (chance) | 0.03 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L26 MLP |
| CKA(arrangement before, joint write) | 0.21 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.03 |
| share of the write inside the old arrangement's span | 0.18 |
| write energy / code energy before | 0.00 |
| best source position (CKA) | `b` (0.31) |

<details><summary>codes and components</summary>

**code 4a-563.0 (L27): 1 comps, tells apart 2/121, coverage 0.22, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 o c5 | H14 | a//10 {0} x b//10 {0..1}; a//10 {1} x b//10 {1, 7..9}; a//10 {2} x b//10 {7}; a//10 {3} x b//10 {6..7}; a//10 {5} x b//10 {3..4, 7..8}; a//10 {6} x b//10 {0, 3, 10}; a//10 {7} x b//10 {0, 2..3}; a//10 {8} x b//10 {8..9}; a//10 {9} x b//10 {0, 8..9}; a//10 {10} x b//10 {0..1, 10} (+) | 0.53 | 1.000 |

</details>

</details>

<details><summary><b>4a-564</b> `tens(a,b)` @ `=` (add) — single component; 1 comps, L28; tells apart 7/121 classes (best member 7); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.05 |
| classes told apart: joint / best code / best member | 7 /  / 7 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.78 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L28 attn |
| CKA(arrangement before, joint write) | 0.20 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.01 |

<details><summary>codes and components</summary>

**code 4a-564.0 (L28): 1 comps, tells apart 7/121, coverage 0.05, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 down c91 | MLP | a//10 {7} x b//10 {9..10}; a//10 {8} x b//10 {8..10}; a//10 {10} x b//10 {7} (+) | 0.77 | 0.112 |

</details>

</details>

<details><summary><b>4a-565</b> `tens(a,b)` @ `=` (add) — block code; 2 comps, L29 L30; tells apart 3/121 classes (best member 3); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.35 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 121) |
| members whose removal merges classes | 0.50 |
| support overlap (1 = tiling) / random sets / p | 1.07 / 1.00 / 0.68 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.62 |
| decoding acc. joint / best code / best member (chance) | 0.07 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L29 attn |
| CKA(arrangement before, joint write) | 0.47 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.01 |
| best source position (CKA) | `a` (0.45) |

<details><summary>codes and components</summary>

**code 4a-565.0 (L29 L30): 2 comps, tells apart 3/121, coverage 0.35, overlap 1.07 (random 1.00, p 0.70)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L29 down c11 | MLP | a//10 {0} x b//10 {0..3}; a//10 {1..3} x b//10 {0}; a//10 {6} x b//10 {7, 9..10}; a//10 {7..8} x b//10 {6..7, 9..10}; a//10 {9} x b//10 {5..8}; a//10 {10} x b//10 {5..8, 10} (+) | 0.63 | 0.736 |
| L30 o c446 | H18 | a//10 {0} x b//10 {4, 10}; a//10 {1, 4} x b//10 {4}; a//10 {5} x b//10 {1, 7..8}; a//10 {6} x b//10 {3, 6}; a//10 {7} x b//10 {2, 5..6}; a//10 {8} x b//10 {1, 5}; a//10 {9} x b//10 {7..10} (-) | 0.50 | 1.000 |

</details>

</details>

</details>

<details><summary>`res//10`: 10 mechanisms, 61 components</summary>

<details><summary><b>4a-525</b> `res//10` @ `=` (add) — block code, 2 codes of the same shape; 5 comps, L18 L31; tells apart 8/21 classes (best member 5); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.52 |
| classes told apart: joint / best code / best member | 8 / 8 / 5 (of 21) |
| members whose removal merges classes | 0.60 |
| support overlap (1 = tiling) / random sets / p | 1.82 / 1.43 / 0.84 |
| mean CKA between its codes | 0.89 |
| purity of the joint write (per prompt) | 0.60 |
| decoding acc. joint / best code / best member (chance) | 0.30 / 0.30 / 0.17 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L18 attn |
| CKA(arrangement before, joint write) | 0.59 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.02 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 0.45 |

<details><summary>codes and components</summary>

**code 4a-525.0 (L18): 1 comps, tells apart 5/21, coverage 0.24, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 down c147 | MLP | res//10 in {0..4} (-) | 0.82 | 0.155 |

**code 4a-525.1 (L31): 4 comps, tells apart 8/21, coverage 0.48, overlap 1.50 (random 1.30, p 0.81)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L31 down c622 | MLP | res//10 in {0..2, 20} (+) | 0.49 | 0.114 |
| L31 down c307 | MLP | res//10 in {0..3} (-) | 0.69 | 0.108 |
| L31 down c201 | MLP | res//10 in {15..18} (+) | 0.83 | 0.127 |
| L31 down c760 | MLP | res//10 in {18..20} (-) | 0.86 | 0.022 |

</details>

</details>

<details><summary><b>4a-526</b> `res//10` @ `=` (add) — block code, 2 codes of the same shape; 11 comps, L19 L23; tells apart 14/21 classes (best member 5); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.86 |
| classes told apart: joint / best code / best member | 14 / 13 / 5 (of 21) |
| members whose removal merges classes | 0.45 |
| support overlap (1 = tiling) / random sets / p | 2.50 / 2.21 / 0.71 |
| mean CKA between its codes | 0.71 |
| purity of the joint write (per prompt) | 0.85 |
| decoding acc. joint / best code / best member (chance) | 0.64 / 0.55 / 0.22 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.12 |
| consumers / read jointly | 7 / 7 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L19 attn |
| CKA(arrangement before, joint write) | 0.83 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.36 |

<details><summary>codes and components</summary>

**code 4a-526.0 (L19): 5 comps, tells apart 13/21, coverage 0.52, overlap 1.82 (random 1.40, p 0.89)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L19 down c3 | MLP | res//10 in {0..3} (+) | 0.69 | 0.267 |
| L19 down c34 | MLP | res//10 in {0..4} (+) | 0.85 | 0.144 |
| L19 down c454 | MLP | res//10 in {0} (-) | 0.73 | 0.006 |
| L19 down c86 | MLP | res//10 in {14..17} (+) | 0.88 | 0.246 |
| L19 down c24 | MLP | res//10 in {4..6, 14..16} (-) | 0.78 | 0.436 |

**code 4a-526.1 (L23): 6 comps, tells apart 10/21, coverage 0.86, overlap 1.39 (random 1.55, p 0.27)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 down c58 | MLP | res//10 in {0..4} (-) | 0.89 | 0.150 |
| L23 down c10 | MLP | res//10 in {0..5} (-) | 0.62 | 0.188 |
| L23 down c29 | MLP | res//10 in {10..12} (+) | 0.87 | 0.579 |
| L23 down c130 | MLP | res//10 in {14..18} (+) | 0.91 | 0.278 |
| L23 down c7 | MLP | res//10 in {4..7} (-) | 0.87 | 0.387 |
| L23 down c18 | MLP | res//10 in {8..9} (+) | 0.78 | 0.266 |

</details>

</details>

<details><summary><b>4a-527</b> `res//10` @ `=` (add) — block code, 3 codes of the same shape; 15 comps, L20 L24 L25; tells apart 17/21 classes (best member 6); on sub: 4s-636 (member overlap 0.06)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.90 |
| classes told apart: joint / best code / best member | 17 / 14 / 6 (of 21) |
| members whose removal merges classes | 0.20 |
| support overlap (1 = tiling) / random sets / p | 3.32 / 2.85 / 0.84 |
| mean CKA between its codes | 0.76 |
| purity of the joint write (per prompt) | 0.85 |
| decoding acc. joint / best code / best member (chance) | 0.73 / 0.61 / 0.26 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.11 |
| consumers / read jointly | 42 / 36 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L20 attn |
| CKA(arrangement before, joint write) | 0.66 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.08 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.42 |

<details><summary>codes and components</summary>

**code 4a-527.0 (L20): 4 comps, tells apart 14/21, coverage 0.86, overlap 1.72 (random 1.29, p 0.93)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 down c5 | MLP | res//10 in {0..1, 11..17} (-) | 0.82 | 0.714 |
| L20 down c20 | MLP | res//10 in {0..4, 6..9, 12..14, 16..19} (-) | 0.80 | 0.544 |
| L20 down c2 | MLP | res//10 in {13..14} (-) | 0.85 | 1.000 |
| L20 down c153 | MLP | res//10 in {14..17} (+) | 0.87 | 0.193 |

**code 4a-527.1 (L24): 5 comps, tells apart 11/21, coverage 0.62, overlap 1.23 (random 1.44, p 0.22)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 down c350 | MLP | res//10 in {0..2, 20} (+) | 0.78 | 0.093 |
| L24 down c112 | MLP | res//10 in {12..13} (-) | 0.86 | 0.164 |
| L24 down c51 | MLP | res//10 in {12..14} (-) | 0.85 | 0.284 |
| L24 down c34 | MLP | res//10 in {14..18} (-) | 0.86 | 0.277 |
| L24 down c8 | MLP | res//10 in {6..7} (-) | 0.79 | 0.248 |

**code 4a-527.2 (L25): 6 comps, tells apart 12/21, coverage 0.62, overlap 1.23 (random 1.57, p 0.10)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L25 down c220 | MLP | res//10 in {0..2} (+) | 0.73 | 0.045 |
| L25 down c43 | MLP | res//10 in {0..3} (-) | 0.65 | 0.068 |
| L25 down c33 | MLP | res//10 in {12..13} (-) | 0.90 | 0.417 |
| L25 down c200 | MLP | res//10 in {14..16} (+) | 0.86 | 0.171 |
| L25 down c256 | MLP | res//10 in {17..18} (+) | 0.86 | 0.072 |
| L25 down c73 | MLP | res//10 in {8..9} (+) | 0.70 | 0.266 |

</details>

</details>

<details><summary><b>4a-528</b> `res//10` @ `=` (add) — block code; 4 comps, L21; tells apart 14/21 classes (best member 6); on sub: 4s-634 (member overlap 0.14)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.48 |
| classes told apart: joint / best code / best member | 14 /  / 6 (of 21) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.10 / 1.27 / 0.26 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.82 |
| decoding acc. joint / best code / best member (chance) | 0.45 /  / 0.17 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 0.99 |
| consumers / read jointly | 15 / 1 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 attn |
| CKA(arrangement before, joint write) | 0.44 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.10 |

<details><summary>codes and components</summary>

**code 4a-528.0 (L21): 4 comps, tells apart 14/21, coverage 0.48, overlap 1.10 (random 1.30, p 0.21)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c105 | MLP | res//10 in {0..1} (-) | 0.80 | 0.032 |
| L21 down c0 | MLP | res//10 in {11..13} (-) | 0.82 | 0.332 |
| L21 down c267 | MLP | res//10 in {18..19} (+) | 0.77 | 0.074 |
| L21 down c36 | MLP | res//10 in {7, 16..18} (-) | 0.83 | 0.351 |

</details>

</details>

<details><summary><b>4a-529</b> `res//10` @ `=` (add) — tiling; 8 comps, L22; tells apart 15/21 classes (best member 6); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.86 |
| classes told apart: joint / best code / best member | 15 /  / 6 (of 21) |
| members whose removal merges classes | 0.62 |
| support overlap (1 = tiling) / random sets / p | 1.28 / 1.80 / 0.04 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.83 |
| decoding acc. joint / best code / best member (chance) | 0.68 /  / 0.20 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 0.97 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L22 attn |
| CKA(arrangement before, joint write) | 0.72 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.08 |

<details><summary>codes and components</summary>

**code 4a-529.0 (L22): 8 comps, tells apart 15/21, coverage 0.86, overlap 1.28 (random 1.83, p 0.03)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 down c267 | MLP | res//10 in {0..1} (+) | 0.50 | 0.060 |
| L22 down c107 | MLP | res//10 in {0..4} (-) | 0.84 | 0.134 |
| L22 down c134 | MLP | res//10 in {11} (-) | 0.79 | 0.203 |
| L22 down c29 | MLP | res//10 in {12..18} (-) | 0.92 | 0.447 |
| L22 down c404 | MLP | res//10 in {19..20} (-) | 0.54 | 0.035 |
| L22 down c35 | MLP | res//10 in {5, 15} (-) | 0.81 | 0.148 |
| L22 down c28 | MLP | res//10 in {8, 18} (+) | 0.82 | 0.144 |
| L22 down c33 | MLP | res//10 in {9, 19} (-) | 0.77 | 0.217 |

</details>

</details>

<details><summary><b>4a-530</b> `res//10` @ `=` (add) — block code; 3 comps, L26; tells apart 9/21 classes (best member 6); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.62 |
| classes told apart: joint / best code / best member | 9 /  / 6 (of 21) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.12 / 0.40 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.88 |
| decoding acc. joint / best code / best member (chance) | 0.45 /  / 0.18 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.03 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L26 attn |
| CKA(arrangement before, joint write) | 0.67 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.06 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.03 |

<details><summary>codes and components</summary>

**code 4a-530.0 (L26): 3 comps, tells apart 9/21, coverage 0.62, overlap 1.00 (random 1.17, p 0.38)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 down c85 | MLP | res//10 in {0..2} (-) | 0.81 | 0.102 |
| L26 down c7 | MLP | res//10 in {11..15} (+) | 0.88 | 0.519 |
| L26 down c605 | MLP | res//10 in {16..20} (-) | 0.92 | 0.150 |

</details>

</details>

<details><summary><b>4a-531</b> `res//10` @ `=` (add) — block code; 3 comps, L27; tells apart 9/21 classes (best member 6); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.29 |
| classes told apart: joint / best code / best member | 9 /  / 6 (of 21) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.17 / 1.14 / 0.56 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.88 |
| decoding acc. joint / best code / best member (chance) | 0.34 /  / 0.21 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.05 |
| consumers / read jointly | 12 / 6 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L27 attn |
| CKA(arrangement before, joint write) | 0.57 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.10 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.06 |

<details><summary>codes and components</summary>

**code 4a-531.0 (L27): 3 comps, tells apart 9/21, coverage 0.29, overlap 1.17 (random 1.17, p 0.52)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 down c6 | MLP | res//10 in {10..12} (+) | 0.89 | 0.611 |
| L27 down c23 | MLP | res//10 in {11} (-) | 0.87 | 0.127 |
| L27 down c93 | MLP | res//10 in {16..18} (+) | 0.82 | 0.123 |

</details>

</details>

<details><summary><b>4a-532</b> `res//10` @ `=` (add) — tiling; 7 comps, L28; tells apart 10/21 classes (best member 5); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.52 |
| classes told apart: joint / best code / best member | 10 /  / 5 (of 21) |
| members whose removal merges classes | 0.71 |
| support overlap (1 = tiling) / random sets / p | 1.18 / 1.72 / 0.01 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.80 |
| decoding acc. joint / best code / best member (chance) | 0.40 /  / 0.17 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.02 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L28 attn |
| CKA(arrangement before, joint write) | 0.60 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.02 |

<details><summary>codes and components</summary>

**code 4a-532.0 (L28): 7 comps, tells apart 10/21, coverage 0.52, overlap 1.18 (random 1.67, p 0.02)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 down c338 | MLP | res//10 in {10} (-) | 0.81 | 0.097 |
| L28 down c145 | MLP | res//10 in {12..16} (-) | 0.76 | 0.277 |
| L28 down c638 | MLP | res//10 in {19..20} (+) | 0.92 | 0.007 |
| L28 down c462 | MLP | res//10 in {19..20} (-) | 0.76 | 0.012 |
| L28 down c70 | MLP | res//10 in {4} (-) | 0.86 | 0.110 |
| L28 down c268 | MLP | res//10 in {5} (+) | 0.79 | 0.083 |
| L28 down c312 | MLP | res//10 in {7} (+) | 0.80 | 0.082 |

</details>

</details>

<details><summary><b>4a-533</b> `res//10` @ `=` (add) — block code; 3 comps, L29; tells apart 7/21 classes (best member 5); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.24 |
| classes told apart: joint / best code / best member | 7 /  / 5 (of 21) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.60 / 1.17 / 0.93 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.80 |
| decoding acc. joint / best code / best member (chance) | 0.26 /  / 0.19 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.02 |
| consumers / read jointly | 2 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L29 attn |
| CKA(arrangement before, joint write) | 0.38 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.02 |

<details><summary>codes and components</summary>

**code 4a-533.0 (L29): 3 comps, tells apart 7/21, coverage 0.24, overlap 1.60 (random 1.17, p 0.94)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L29 down c584 | MLP | res//10 in {0..2, 20} (+) | 0.57 | 0.111 |
| L29 down c67 | MLP | res//10 in {0..2} (+) | 0.76 | 0.060 |
| L29 down c51 | MLP | res//10 in {12} (+) | 0.85 | 0.137 |

</details>

</details>

<details><summary><b>4a-534</b> `res//10` @ `=` (add) — block code; 2 comps, L30; tells apart 6/21 classes (best member 4); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.43 |
| classes told apart: joint / best code / best member | 6 /  / 4 (of 21) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.22 / 1.00 / 0.77 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.61 |
| decoding acc. joint / best code / best member (chance) | 0.28 /  / 0.24 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.03 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L30 attn |
| CKA(arrangement before, joint write) | 0.58 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.01 |

<details><summary>codes and components</summary>

**code 4a-534.0 (L30): 2 comps, tells apart 6/21, coverage 0.43, overlap 1.22 (random 1.00, p 0.81)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 down c21 | MLP | res//10 in {0..3, 15..19} (-) | 0.60 | 0.880 |
| L30 down c809 | MLP | res//10 in {15..16} (+) | 0.64 | 0.116 |

</details>

</details>

</details>

<details><summary>`units(a,b)`: 7 mechanisms, 52 components</summary>

<details><summary><b>4a-566</b> `units(a,b)` @ `=` (add) — block code; 6 comps, L16; tells apart 31/100 classes (best member 5); on sub: 4s-674 (member overlap 0.83)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.60 |
| classes told apart: joint / best code / best member | 31 /  / 5 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.37 / 1.38 / 0.46 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.93 |
| decoding acc. joint / best code / best member (chance) | 0.36 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.97 |
| consumers / read jointly | 2 / 2 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L16 attn |
| CKA(arrangement before, joint write) | 0.35 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.09 |
| share of the write inside the old arrangement's span | 0.12 |
| write energy / code energy before | 0.10 |

<details><summary>codes and components</summary>

**code 4a-566.0 (L16): 6 comps, tells apart 31/100, coverage 0.60, overlap 1.37 (random 1.36, p 0.53)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c79 | MLP | a%10 {0, 5} x b%10 {0..1, 5..6}; a%10 {1, 6} x b%10 {0, 5} (+) | 0.95 | 0.334 |
| L16 down c94 | MLP | a%10 {1, 6} x b%10 {4}; a%10 {2, 7} x b%10 {0, 4..5, 9} (+) | 0.90 | 0.213 |
| L16 down c127 | MLP | a%10 {1..2, 6..7} x b%10 {3..4, 8..9} (-) | 0.89 | 0.215 |
| L16 down c76 | MLP | a%10 {2, 7} x b%10 {0, 5}; a%10 {3, 8} x b%10 {0, 5..6} (+) | 0.95 | 0.261 |
| L16 down c107 | MLP | a%10 {3..6} x b%10 {3..5}; a%10 {7..8} x b%10 {3..4} (+) | 0.94 | 0.218 |
| L16 down c275 | MLP | a%10 {3} x b%10 {0..2, 5..7}; a%10 {4, 9} x b%10 {0..1, 5..6}; a%10 {8} x b%10 {1..2, 6..7} (-) | 0.89 | 0.213 |

</details>

</details>

<details><summary><b>4a-567</b> `units(a,b)` @ `=` (add) — block code; 13 comps, L17; tells apart 90/100 classes (best member 8); on sub: 4s-675 (member overlap 0.54)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.98 |
| classes told apart: joint / best code / best member | 90 /  / 8 (of 100) |
| members whose removal merges classes | 0.92 |
| support overlap (1 = tiling) / random sets / p | 2.70 / 2.09 / 0.99 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.93 |
| decoding acc. joint / best code / best member (chance) | 0.91 /  / 0.07 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.07 |
| consumers / read jointly | 32 / 26 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L17 attn |
| CKA(arrangement before, joint write) | 0.61 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.11 |
| share of the write inside the old arrangement's span | 0.19 |
| write energy / code energy before | 0.39 |

<details><summary>codes and components</summary>

**code 4a-567.0 (L17): 13 comps, tells apart 90/100, coverage 0.98, overlap 2.70 (random 2.07, p 0.99)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 down c47 | MLP | a%10 {0, 4..5, 9} x b%10 {0, 3..5, 8..9}; a%10 {3, 8} x b%10 {0, 4..5, 9} (+) | 0.89 | 0.328 |
| L17 down c61 | MLP | a%10 {0, 4..5} x b%10 {3..4, 8..9}; a%10 {3} x b%10 {0, 3..5, 8..9}; a%10 {6} x b%10 {3, 8}; a%10 {8} x b%10 {0, 3..6, 8..9}; a%10 {9} x b%10 {3..5, 8..9} (+) | 0.92 | 0.395 |
| L17 down c37 | MLP | a%10 {0} x b%10 {0..3, 8..9}; a%10 {1} x b%10 {0..1, 8..9}; a%10 {2..3} x b%10 {0, 8..9}; a%10 {4} x b%10 {8..9}; a%10 {8..9} x b%10 {0, 3, 8..9} (+) | 0.91 | 0.426 |
| L17 down c39 | MLP | a%10 {0} x b%10 {5..8}; a%10 {3} x b%10 {6..9}; a%10 {4, 9} x b%10 {5..9}; a%10 {8} x b%10 {6..7} (-) | 0.92 | 0.387 |
| L17 down c22 | MLP | a%10 {1, 3, 5, 7, 9} x b%10 {0, 2, 4, 6, 8} (+) | 0.95 | 0.253 |
| L17 down c67 | MLP | a%10 {1, 3, 5, 7, 9} x b%10 {1, 3, 5, 7, 9} (-) | 0.95 | 0.247 |
| L17 down c116 | MLP | a%10 {1..2, 6..7} x b%10 {1..2, 6..7} (-) | 0.92 | 0.252 |
| L17 down c52 | MLP | a%10 {1} x b%10 {3..4}; a%10 {2} x b%10 {2..4}; a%10 {3..4} x b%10 {1..4} (+) | 0.93 | 0.258 |
| L17 down c86 | MLP | a%10 {1} x b%10 {6..8}; a%10 {2} x b%10 {5..8}; a%10 {3} x b%10 {1..2}; a%10 {4} x b%10 {0..2, 8..9}; a%10 {5} x b%10 {0..1, 8..9}; a%10 {6} x b%10 {9} (+) | 0.88 | 0.306 |
| L17 down c70 | MLP | a%10 {2, 7} x b%10 {1..4, 7, 9}; a%10 {4, 9} x b%10 {2} (-) | 0.84 | 0.183 |
| L17 down c34 | MLP | a%10 {5..6} x b%10 {0, 5..9}; a%10 {7} x b%10 {5..9}; a%10 {8} x b%10 {5..7} (+) | 0.95 | 0.346 |
| L17 down c49 | MLP | a%10 {5} x b%10 {0..2}; a%10 {6..7} x b%10 {0..2, 5, 9}; a%10 {8} x b%10 {1, 5} (-) | 0.92 | 0.380 |
| L17 down c118 | MLP | a%10 {7..8} x b%10 {6..8}; a%10 {9} x b%10 {6..7} (-) | 0.80 | 0.104 |

</details>

</details>

<details><summary><b>4a-568</b> `units(a,b)` @ `=` (add) — block code; 2 comps, L18; tells apart 8/100 classes (best member 7); on sub: 4s-676 (member overlap 0.25)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.13 |
| classes told apart: joint / best code / best member | 8 /  / 7 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.04 / 0.42 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.84 |
| decoding acc. joint / best code / best member (chance) | 0.06 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L17 MLP |
| CKA(arrangement before, joint write) | 0.23 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.08 |
| share of the write inside the old arrangement's span | 0.15 |
| write energy / code energy before | 0.01 |
| best source position (CKA) | `b` (0.50) |

<details><summary>codes and components</summary>

**code 4a-568.0 (L18): 2 comps, tells apart 8/100, coverage 0.13, overlap 1.00 (random 1.04, p 0.45)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 o c313 | H30 | a%10 {0} x b%10 {0, 5}; a%10 {1..7} x b%10 {0} (+) | 0.86 | 0.223 |
| L18 o c200 | H30 | a%10 {1, 3, 5..6} x b%10 {5} (+) | 0.56 | 0.118 |

</details>

</details>

<details><summary><b>4a-569</b> `units(a,b)` @ `=` (add) — tiling, 2 codes of the same shape; 22 comps, L18 L19; tells apart 88/100 classes (best member 7); on sub: 4s-677 (member overlap 0.28)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.96 |
| classes told apart: joint / best code / best member | 88 / 84 / 7 (of 100) |
| members whose removal merges classes | 0.45 |
| support overlap (1 = tiling) / random sets / p | 2.80 / 3.20 / 0.08 |
| mean CKA between its codes | 0.78 |
| purity of the joint write (per prompt) | 0.92 |
| decoding acc. joint / best code / best member (chance) | 0.93 / 0.94 / 0.08 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.08 |
| consumers / read jointly | 135 / 119 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L18 attn |
| CKA(arrangement before, joint write) | 0.40 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.17 |
| share of the write inside the old arrangement's span | 0.39 |
| write energy / code energy before | 0.56 |

<details><summary>codes and components</summary>

**code 4a-569.0 (L18): 16 comps, tells apart 84/100, coverage 0.93, overlap 1.99 (random 2.45, p 0.03)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 down c4 | MLP | a%10 {0, 2, 4, 6, 8} x b%10 {0, 2, 4, 6, 8} (-) | 0.98 | 0.413 |
| L18 down c375 | MLP | a%10 {0, 5} x b%10 {2..4, 7..9} (-) | 0.88 | 0.367 |
| L18 down c62 | MLP | a%10 {0} x b%10 {0..1, 9}; a%10 {1} x b%10 {0..1, 8..9}; a%10 {8} x b%10 {1..2}; a%10 {9} x b%10 {0..2, 9} (-) | 0.96 | 0.248 |
| L18 down c24 | MLP | a%10 {0} x b%10 {2}; a%10 {3} x b%10 {3..4, 7..9}; a%10 {4, 9} x b%10 {2..3, 7..8}; a%10 {5} x b%10 {2, 7}; a%10 {8} x b%10 {3..4, 8} (+) | 0.92 | 0.572 |
| L18 down c474 | MLP | a%10 {0} x b%10 {3..4}; a%10 {1} x b%10 {1..4}; a%10 {2} x b%10 {1..3}; a%10 {3} x b%10 {1..2}; a%10 {4} x b%10 {1} (-) | 0.91 | 0.177 |
| L18 down c98 | MLP | a%10 {0} x b%10 {5..7}; a%10 {1} x b%10 {4..6}; a%10 {2} x b%10 {5}; a%10 {3} x b%10 {0}; a%10 {9} x b%10 {5..6} (+) | 0.88 | 0.335 |
| L18 down c65 | MLP | a%10 {1, 5} x b%10 {7}; a%10 {2} x b%10 {5..7}; a%10 {6} x b%10 {6..8}; a%10 {7} x b%10 {5..8} (+) | 0.89 | 0.356 |
| L18 down c142 | MLP | a%10 {2, 7} x b%10 {2, 7} (+) | 0.86 | 0.539 |
| L18 down c103 | MLP | a%10 {2, 7} x b%10 {2, 7} (-) | 0.76 | 0.045 |
| L18 down c101 | MLP | a%10 {2} x b%10 {3, 7..9}; a%10 {3, 8} x b%10 {2..3, 7..8}; a%10 {7} x b%10 {3..4, 7..9}; a%10 {9} x b%10 {7} (-) | 0.90 | 0.423 |
| L18 down c88 | MLP | a%10 {2} x b%10 {3..4}; a%10 {3} x b%10 {2..5}; a%10 {4} x b%10 {1..4}; a%10 {5} x b%10 {3} (+) | 0.93 | 0.303 |
| L18 down c80 | MLP | a%10 {3, 8} x b%10 {3, 8} (+) | 0.86 | 0.514 |
| L18 down c26 | MLP | a%10 {3, 8} x b%10 {4, 9}; a%10 {4, 9} x b%10 {3..4, 8..9}; a%10 {5} x b%10 {3, 8} (-) | 0.92 | 0.695 |
| L18 down c32 | MLP | a%10 {3} x b%10 {4..6}; a%10 {4} x b%10 {3..7}; a%10 {5} x b%10 {3..6}; a%10 {6} x b%10 {3..5} (-) | 0.94 | 0.457 |
| L18 down c261 | MLP | a%10 {5} x b%10 {5..7}; a%10 {6..7} x b%10 {5} (-) | 0.88 | 0.060 |
| L18 down c120 | MLP | a%10 {6} x b%10 {4}; a%10 {7..9} x b%10 {3..4} (+) | 0.94 | 0.190 |

**code 4a-569.1 (L19): 6 comps, tells apart 25/100, coverage 0.64, overlap 1.31 (random 1.37, p 0.30)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L19 down c55 | MLP | a%10 {0} x b%10 {0..1}; a%10 {1} x b%10 {0, 9}; a%10 {2} x b%10 {8..9}; a%10 {3} x b%10 {7..8}; a%10 {4} x b%10 {6..7}; a%10 {6} x b%10 {4}; a%10 {7} x b%10 {3..4}; a%10 {8} x b%10 {2..3}; a%10 {9} x b%10 {1..2} (+) | 0.87 | 0.233 |
| L19 down c32 | MLP | a%10 {0} x b%10 {2, 4, 6, 8}; a%10 {2, 4, 6, 8} x b%10 {0, 2, 4, 6, 8} (+) | 0.87 | 0.257 |
| L19 down c41 | MLP | a%10 {0} x b%10 {6..9}; a%10 {1} x b%10 {6..8}; a%10 {7} x b%10 {0..1, 9}; a%10 {8} x b%10 {0, 8..9}; a%10 {9} x b%10 {0, 7..9} (+) | 0.90 | 0.401 |
| L19 down c143 | MLP | a%10 {1, 6} x b%10 {3, 8}; a%10 {2, 7} x b%10 {7}; a%10 {8} x b%10 {6} (-) | 0.70 | 0.223 |
| L19 down c106 | MLP | a%10 {1} x b%10 {3..4}; a%10 {5} x b%10 {9}; a%10 {6} x b%10 {8..9}; a%10 {7} x b%10 {7}; a%10 {8} x b%10 {6}; a%10 {9} x b%10 {5} (+) | 0.59 | 0.101 |
| L19 down c74 | MLP | a%10 {1} x b%10 {7}; a%10 {2} x b%10 {6..7}; a%10 {3} x b%10 {5..6}; a%10 {4} x b%10 {5}; a%10 {5} x b%10 {3..4}; a%10 {6} x b%10 {2..3}; a%10 {7} x b%10 {2} (-) | 0.78 | 0.196 |

</details>

</details>

<details><summary><b>4a-570</b> `units(a,b)` @ `=` (add) — block code; 2 comps, L20; tells apart 9/100 classes (best member 2); on sub: 4s-679 (member overlap 0.25)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.31 |
| classes told apart: joint / best code / best member | 9 /  / 2 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.10 / 1.05 / 0.67 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.89 |
| decoding acc. joint / best code / best member (chance) | 0.08 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L20 attn |
| CKA(arrangement before, joint write) | 0.22 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.29 |
| write energy / code energy before | 0.03 |

<details><summary>codes and components</summary>

**code 4a-570.0 (L20): 2 comps, tells apart 9/100, coverage 0.31, overlap 1.10 (random 1.06, p 0.70)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 down c47 | MLP | a%10 {1, 3, 5, 7, 9} x b%10 {1, 3, 5, 7, 9} (-) | 0.91 | 0.431 |
| L20 down c378 | MLP | a%10 {5} x b%10 {8..9}; a%10 {6} x b%10 {7..8}; a%10 {7} x b%10 {6..7}; a%10 {8} x b%10 {5..6}; a%10 {9} x b%10 {5} (-) | 0.78 | 0.180 |

</details>

</details>

<details><summary><b>4a-571</b> `units(a,b)` @ `=` (add) — block code, 3 codes of the same shape; 5 comps, L20 L23 L31; tells apart 6/100 classes (best member 5); on sub: 4s-678 (member overlap 0.20)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.04 |
| classes told apart: joint / best code / best member | 6 / 5 / 5 (of 100) |
| members whose removal merges classes | 0.20 |
| support overlap (1 = tiling) / random sets / p | 4.00 / 1.29 / 1.00 |
| mean CKA between its codes | 0.77 |
| purity of the joint write (per prompt) | 0.78 |
| decoding acc. joint / best code / best member (chance) | 0.05 / 0.04 / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.28 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L19 MLP |
| CKA(arrangement before, joint write) | 0.13 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.01 |
| best source position (CKA) | `b` (0.29) |

<details><summary>codes and components</summary>

**code 4a-571.0 (L20): 3 comps, tells apart 5/100, coverage 0.04, overlap 2.00 (random 1.13, p 1.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 o c192 | H2 | a%10 {0, 5} x b%10 {0} (+) | 0.59 | 0.061 |
| L20 o c479 | H2 | a%10 {0} x b%10 {0, 5}; a%10 {5} x b%10 {0} (+) | 0.75 | 0.064 |
| L20 o c288 | H2 | a%10 {0} x b%10 {5}; a%10 {5} x b%10 {0, 5} (-) | 0.74 | 0.035 |

**code 4a-571.1 (L23): 1 comps, tells apart 4/100, coverage 0.04, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 down c387 | MLP | a%10 {0, 5} x b%10 {0, 5} (-) | 0.95 | 0.040 |

**code 4a-571.2 (L31): 1 comps, tells apart 4/100, coverage 0.04, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L31 down c187 | MLP | a%10 {0, 5} x b%10 {0, 5} (-) | 0.72 | 0.134 |

</details>

</details>

<details><summary><b>4a-572</b> `units(a,b)` @ `=` (add) — 2 codes of the same shape; 2 comps, L26 L28; tells apart 2/100 classes (best member 2); on sub: 4s-677 (member overlap 0.13)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.27 |
| classes told apart: joint / best code / best member | 2 / 2 / 2 (of 100) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 1.89 / 1.04 / 0.99 |
| mean CKA between its codes | 0.94 |
| purity of the joint write (per prompt) | 0.80 |
| decoding acc. joint / best code / best member (chance) | 0.05 / 0.03 / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.05 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L26 attn |
| CKA(arrangement before, joint write) | 0.30 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.07 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.02 |

<details><summary>codes and components</summary>

**code 4a-572.0 (L26): 1 comps, tells apart 2/100, coverage 0.27, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 down c36 | MLP | a%10 {0} x b%10 {2, 4, 6, 8}; a%10 {2, 4, 6, 8} x b%10 {0, 2, 4, 6, 8}; a%10 {3} x b%10 {5}; a%10 {9} x b%10 {7, 9} (+) | 0.78 | 0.405 |

**code 4a-572.1 (L28): 1 comps, tells apart 2/100, coverage 0.24, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 down c23 | MLP | a%10 {0} x b%10 {2, 4, 6, 8}; a%10 {2, 4, 6, 8} x b%10 {0, 2, 4, 6, 8} (+) | 0.80 | 0.411 |

</details>

</details>

</details>

<details><summary>`res%10`: 8 mechanisms, 44 components</summary>

<details><summary><b>4a-494</b> `res%10` @ `=` (add) — block code; 3 comps, L19 L20; tells apart 8/10 classes (best member 3); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.70 |
| classes told apart: joint / best code / best member | 8 /  / 3 (of 10) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.68 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.87 |
| decoding acc. joint / best code / best member (chance) | 0.68 /  / 0.29 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.03 |
| consumers / read jointly | 9 / 8 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L19 attn |
| CKA(arrangement before, joint write) | 0.38 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.24 |
| share of the write inside the old arrangement's span | 0.25 |
| write energy / code energy before | 0.30 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (50:0.39 20:0.33 10:0.28) |
| joint write: shape (spectrum k:share) | circle period 10 (10:0.79 20:0.19) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.72 / 2.18 |

<details><summary>codes and components</summary>

**code 4a-494.0 (L19 L20): 3 comps, tells apart 8/10, coverage 0.70, overlap 1.00 (random 1.00, p 0.72)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L19 down c149 | MLP | res%10 in {2..3} (+) | 0.80 | 0.274 |
| L20 down c13 | MLP | res%10 in {5..7} (-) | 0.84 | 0.421 |
| L20 down c21 | MLP | res%10 in {8..9} (-) | 0.91 | 0.417 |

</details>

</details>

<details><summary><b>4a-495</b> `res%10` @ `=` (add) — block code; 8 comps, L21; tells apart 10/10 classes (best member 4); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.90 |
| classes told apart: joint / best code / best member | 10 /  / 4 (of 10) |
| members whose removal merges classes | 0.88 |
| support overlap (1 = tiling) / random sets / p | 1.22 / 1.38 / 0.17 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.89 |
| decoding acc. joint / best code / best member (chance) | 0.93 /  / 0.32 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 0.93 |
| consumers / read jointly | 82 / 38 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 attn |
| CKA(arrangement before, joint write) | 0.66 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.10 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.37 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (20:0.44 10:0.37 50:0.19) |
| joint write: shape (spectrum k:share) | simplex (one-hot like) (10:0.40 20:0.21 30:0.17) |
| frequencies new in the write | 30 40 |
| write: shift-symmetric part / dimension (PR) | 0.70 / 5.28 |

<details><summary>codes and components</summary>

**code 4a-495.0 (L21): 8 comps, tells apart 10/10, coverage 0.90, overlap 1.22 (random 1.43, p 0.15)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c68 | MLP | res%10 in {0, 8} (+) | 0.82 | 0.196 |
| L21 down c42 | MLP | res%10 in {0} (-) | 0.85 | 0.101 |
| L21 down c4 | MLP | res%10 in {1..3} (+) | 0.89 | 0.410 |
| L21 down c10 | MLP | res%10 in {1} (-) | 0.91 | 0.106 |
| L21 down c13 | MLP | res%10 in {4} (+) | 0.89 | 0.242 |
| L21 down c14 | MLP | res%10 in {5} (+) | 0.91 | 0.110 |
| L21 down c16 | MLP | res%10 in {6} (+) | 0.92 | 0.107 |
| L21 down c53 | MLP | res%10 in {9} (-) | 0.88 | 0.103 |

</details>

</details>

<details><summary><b>4a-496</b> `res%10` @ `=` (add) — block code; 5 comps, L22; tells apart 6/10 classes (best member 2); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.50 |
| classes told apart: joint / best code / best member | 6 /  / 2 (of 10) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.25 / 0.30 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.89 |
| decoding acc. joint / best code / best member (chance) | 0.56 /  / 0.20 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 0.98 |
| consumers / read jointly | 58 / 11 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L22 attn |
| CKA(arrangement before, joint write) | 0.41 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.14 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (10:0.41 20:0.29 50:0.22) |
| joint write: shape (spectrum k:share) | irregular (30:0.23 20:0.23 10:0.22) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.31 / 2.76 |

<details><summary>codes and components</summary>

**code 4a-496.0 (L22): 5 comps, tells apart 6/10, coverage 0.50, overlap 1.00 (random 1.20, p 0.29)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 down c8 | MLP | res%10 in {0} (+) | 0.87 | 0.101 |
| L22 down c212 | MLP | res%10 in {3} (-) | 0.90 | 0.101 |
| L22 down c333 | MLP | res%10 in {4} (-) | 0.87 | 0.098 |
| L22 down c129 | MLP | res%10 in {6} (+) | 0.81 | 0.097 |
| L22 down c9 | MLP | res%10 in {9} (-) | 0.92 | 0.104 |

</details>

</details>

<details><summary><b>4a-500</b> `res%10` @ `=` (add) — block code, 2 codes of the same shape; 12 comps, L23 L28; tells apart 8/10 classes (best member 3); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.80 |
| classes told apart: joint / best code / best member | 8 / 7 / 3 (of 10) |
| members whose removal merges classes | 0.17 |
| support overlap (1 = tiling) / random sets / p | 1.50 / 1.71 / 0.18 |
| mean CKA between its codes | 0.80 |
| purity of the joint write (per prompt) | 0.92 |
| decoding acc. joint / best code / best member (chance) | 0.89 / 0.71 / 0.23 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.23 |
| consumers / read jointly | 92 / 55 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L23 attn |
| CKA(arrangement before, joint write) | 0.46 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.17 |
| share of the write inside the old arrangement's span | 0.11 |
| write energy / code energy before | 0.34 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (10:0.38 20:0.29 50:0.19) |
| joint write: shape (spectrum k:share) | irregular (40:0.24 30:0.24 20:0.23) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.42 / 3.78 |

<details><summary>codes and components</summary>

**code 4a-500.0 (L23): 6 comps, tells apart 7/10, coverage 0.60, overlap 1.00 (random 1.29, p 0.15)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 down c4 | MLP | res%10 in {2} (-) | 0.92 | 0.108 |
| L23 down c14 | MLP | res%10 in {4} (+) | 0.94 | 0.106 |
| L23 down c9 | MLP | res%10 in {6} (-) | 0.95 | 0.104 |
| L23 down c24 | MLP | res%10 in {7} (-) | 0.91 | 0.101 |
| L23 down c205 | MLP | res%10 in {8} (-) | 0.87 | 0.102 |
| L23 down c106 | MLP | res%10 in {9} (+) | 0.82 | 0.088 |

**code 4a-500.1 (L28): 6 comps, tells apart 7/10, coverage 0.60, overlap 1.00 (random 1.20, p 0.13)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 down c36 | MLP | res%10 in {2} (-) | 0.90 | 0.130 |
| L28 down c271 | MLP | res%10 in {3} (-) | 0.83 | 0.100 |
| L28 down c300 | MLP | res%10 in {4} (-) | 0.92 | 0.099 |
| L28 down c93 | MLP | res%10 in {5} (-) | 0.88 | 0.139 |
| L28 down c69 | MLP | res%10 in {6} (-) | 0.89 | 0.104 |
| L28 down c160 | MLP | res%10 in {7} (-) | 0.80 | 0.103 |

</details>

</details>

<details><summary><b>4a-497</b> `res%10` @ `=` (add) — block code; 6 comps, L24; tells apart 7/10 classes (best member 2); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.60 |
| classes told apart: joint / best code / best member | 7 /  / 2 (of 10) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.20 / 0.17 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.93 |
| decoding acc. joint / best code / best member (chance) | 0.70 /  / 0.21 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 90 / 13 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L24 attn |
| CKA(arrangement before, joint write) | 0.43 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.12 |
| share of the write inside the old arrangement's span | 0.09 |
| write energy / code energy before | 0.22 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (10:0.34 20:0.29 50:0.16) |
| joint write: shape (spectrum k:share) | irregular (20:0.23 30:0.23 10:0.22) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.36 / 3.20 |

<details><summary>codes and components</summary>

**code 4a-497.0 (L24): 6 comps, tells apart 7/10, coverage 0.60, overlap 1.00 (random 1.20, p 0.17)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 down c46 | MLP | res%10 in {0} (+) | 0.93 | 0.101 |
| L24 down c5 | MLP | res%10 in {3} (-) | 0.94 | 0.108 |
| L24 down c12 | MLP | res%10 in {5} (-) | 0.93 | 0.104 |
| L24 down c108 | MLP | res%10 in {6} (+) | 0.92 | 0.099 |
| L24 down c4 | MLP | res%10 in {8} (+) | 0.94 | 0.114 |
| L24 down c83 | MLP | res%10 in {9} (+) | 0.93 | 0.100 |

</details>

</details>

<details><summary><b>4a-498</b> `res%10` @ `=` (add) — block code; 5 comps, L25; tells apart 6/10 classes (best member 3); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.50 |
| classes told apart: joint / best code / best member | 6 /  / 3 (of 10) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.25 / 0.30 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.90 |
| decoding acc. joint / best code / best member (chance) | 0.61 /  / 0.22 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 71 / 7 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L25 attn |
| CKA(arrangement before, joint write) | 0.48 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.14 |
| share of the write inside the old arrangement's span | 0.12 |
| write energy / code energy before | 0.12 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (10:0.31 20:0.28 30:0.16) |
| joint write: shape (spectrum k:share) | irregular (20:0.24 30:0.22 10:0.22) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.40 / 3.55 |

<details><summary>codes and components</summary>

**code 4a-498.0 (L25): 5 comps, tells apart 6/10, coverage 0.50, overlap 1.00 (random 1.25, p 0.24)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L25 down c12 | MLP | res%10 in {1} (+) | 0.93 | 0.114 |
| L25 down c108 | MLP | res%10 in {2} (+) | 0.85 | 0.137 |
| L25 down c53 | MLP | res%10 in {5} (+) | 0.85 | 0.102 |
| L25 down c11 | MLP | res%10 in {7} (-) | 0.92 | 0.106 |
| L25 down c24 | MLP | res%10 in {9} (+) | 0.89 | 0.114 |

</details>

</details>

<details><summary><b>4a-499</b> `res%10` @ `=` (add) — block code; 2 comps, L26 L27; tells apart 3/10 classes (best member 2); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.20 |
| classes told apart: joint / best code / best member | 3 /  / 2 (of 10) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.84 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.90 |
| decoding acc. joint / best code / best member (chance) | 0.29 /  / 0.20 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L26 attn |
| CKA(arrangement before, joint write) | 0.43 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (10:0.29 20:0.28 30:0.17) |
| joint write: shape (spectrum k:share) | irregular (20:0.23 40:0.23 10:0.22) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.20 / 1.83 |

<details><summary>codes and components</summary>

**code 4a-499.0 (L26 L27): 2 comps, tells apart 3/10, coverage 0.20, overlap 1.00 (random 1.00, p 0.85)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 down c175 | MLP | res%10 in {1} (-) | 0.92 | 0.099 |
| L27 down c165 | MLP | res%10 in {6} (-) | 0.87 | 0.099 |

</details>

</details>

<details><summary><b>4a-501</b> `res%10` @ `=` (add) — block code; 3 comps, L29 L30; tells apart 4/10 classes (best member 2); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.30 |
| classes told apart: joint / best code / best member | 4 /  / 2 (of 10) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.66 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.85 |
| decoding acc. joint / best code / best member (chance) | 0.39 /  / 0.20 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 4 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L29 attn |
| CKA(arrangement before, joint write) | 0.45 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.16 |
| share of the write inside the old arrangement's span | 0.11 |
| write energy / code energy before | 0.03 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (10:0.28 20:0.25 30:0.18) |
| joint write: shape (spectrum k:share) | irregular (40:0.24 20:0.23 10:0.23) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.28 / 2.47 |

<details><summary>codes and components</summary>

**code 4a-501.0 (L29 L30): 3 comps, tells apart 4/10, coverage 0.30, overlap 1.00 (random 1.00, p 0.69)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L29 down c293 | MLP | res%10 in {0} (-) | 0.82 | 0.122 |
| L29 down c132 | MLP | res%10 in {4} (-) | 0.87 | 0.106 |
| L30 down c230 | MLP | res%10 in {8} (+) | 0.83 | 0.101 |

</details>

</details>

</details>

<details><summary>`a%100`: 15 mechanisms, 44 components</summary>

<details><summary><b>4a-439</b> `a%100` @ `=` (add) — single component; 1 comps, L2; tells apart 3/100 classes (best member 3); on sub: 4s-575 (member overlap 0.50)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.12 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 100) |
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
| input point | L1 MLP |
| CKA(arrangement before, joint write) | 0.33 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.15 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.21 2:0.11 20:0.09) |
| joint write: shape (spectrum k:share) | line (1:0.51 2:0.07 5:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.14 / 1.00 |
| best source position (CKA) | `b` (0.40) |

<details><summary>codes and components</summary>

**code 4a-439.0 (L2): 1 comps, tells apart 3/100, coverage 0.12, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 o c89 | H23 | a%100 in {2..4, 6..8, 10, 39, 41, 51, 55, 90} (+) | 0.69 | 1.000 |

</details>

</details>

<details><summary><b>4a-438</b> `a%100` @ `=` (add) — block code, 2 codes of the same shape, copy from `op`; 4 comps, L2; tells apart 23/100 classes (best member 4); on sub: 4s-574 (member overlap 0.17)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.34 |
| classes told apart: joint / best code / best member | 23 / 14 / 4 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 2.15 / 1.14 / 1.00 |
| mean CKA between its codes | 0.85 |
| purity of the joint write (per prompt) | 0.93 |
| decoding acc. joint / best code / best member (chance) | 0.36 / 0.30 / 0.09 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.98 |
| consumers / read jointly | 21 / 18 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L1 MLP |
| CKA(arrangement before, joint write) | 0.44 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 1.40 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.21 2:0.11 20:0.09) |
| joint write: shape (spectrum k:share) | irregular (20:0.29 40:0.18 1:0.15) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.12 / 1.57 |
| best source position (CKA) | `op` (0.56) |

<details><summary>codes and components</summary>

**code 4a-438.0 (L2): 2 comps, tells apart 14/100, coverage 0.26, overlap 1.27 (random 1.00, p 0.96)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 o c2 | H2 | a%100 in {0..1, 10, 18, 20, 30, 40, 50, 60, 70, 80, 90} (-) | 0.93 | 1.000 |
| L2 o c3 | H2 | a%100 in {0..9, 11, 13..14, 17, 22..23, 50, 60, 70, 80, 90} (-) | 0.97 | 1.000 |

**code 4a-438.1 (L2): 2 comps, tells apart 6/100, coverage 0.34, overlap 1.18 (random 1.00, p 0.86)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 down c3 | MLP | a%100 in {0..1, 10, 15, 18, 20, 25, 30, 40, 50, 55, 60, 65, 70, 80, 90} (+) | 0.90 | 1.000 |
| L2 down c4 | MLP | a%100 in {1..17, 19, 21..23, 70, 80, 90} (+) | 0.83 | 1.000 |

</details>

</details>

<details><summary><b>4a-441</b> `a%100` @ `=` (add) — copy from `b`; 1 comps, L3; tells apart 1/100 classes (best member 1); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.23 |
| classes told apart: joint / best code / best member | 1 /  / 1 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.79 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 3 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 MLP |
| CKA(arrangement before, joint write) | 0.68 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.26 2:0.11 20:0.10) |
| joint write: shape (spectrum k:share) | line (1:0.59 2:0.10 5:0.05) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.18 / 1.00 |
| best source position (CKA) | `b` (0.75) |

<details><summary>codes and components</summary>

**code 4a-441.0 (L3): 1 comps, tells apart 1/100, coverage 0.23, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 o c128 | H5 | a%100 in {1..10, 12, 24, 31, 75, 77, 80, 82, 84..85, 87, 89, 94..95} (+) | 0.78 | 1.000 |

</details>

</details>

<details><summary><b>4a-440</b> `a%100` @ `=` (add) — block code; 2 comps, L3 L4; tells apart 9/100 classes (best member 4); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.13 |
| classes told apart: joint / best code / best member | 9 /  / 4 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.65 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.57 |
| decoding acc. joint / best code / best member (chance) | 0.08 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.98 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 MLP |
| CKA(arrangement before, joint write) | 0.39 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.26 2:0.11 20:0.10) |
| joint write: shape (spectrum k:share) | irregular (1:0.28 3:0.14 2:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.12 / 1.98 |
| best source position (CKA) | `a` (0.44) |

<details><summary>codes and components</summary>

**code 4a-440.0 (L3 L4): 2 comps, tells apart 9/100, coverage 0.13, overlap 1.00 (random 1.00, p 0.65)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 o c154 | H27 | a%100 in {1..4, 50} (-) | 0.67 | 1.000 |
| L4 o c14 | H24 | a%100 in {15, 20..21, 24..26, 77..78} (-) | 0.52 | 1.000 |

</details>

</details>

<details><summary><b>4a-442</b> `a%100` @ `=` (add) — block code; 2 comps, L5 L6; tells apart 14/100 classes (best member 4); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.13 |
| classes told apart: joint / best code / best member | 14 /  / 4 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.08 / 1.00 / 0.78 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.56 |
| decoding acc. joint / best code / best member (chance) | 0.08 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.98 |
| consumers / read jointly | 4 / 3 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L4 MLP |
| CKA(arrangement before, joint write) | 0.44 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.04 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.30 2:0.11 20:0.08) |
| joint write: shape (spectrum k:share) | irregular (1:0.25 20:0.16 2:0.11) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.11 / 1.79 |
| best source position (CKA) | `b` (0.38) |

<details><summary>codes and components</summary>

**code 4a-442.0 (L5 L6): 2 comps, tells apart 14/100, coverage 0.13, overlap 1.08 (random 1.00, p 0.81)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 o c423 | H14 | a%100 in {1..3, 7..9, 11} (+) | 0.61 | 1.000 |
| L6 down c17 | MLP | a%100 in {0..1, 10, 50, 55, 80, 90} (-) | 0.52 | 1.000 |

</details>

</details>

<details><summary><b>4a-443</b> `a%100` @ `=` (add) — block code; 2 comps, L5; tells apart 11/100 classes (best member 5); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.10 |
| classes told apart: joint / best code / best member | 11 /  / 5 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.65 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.53 |
| decoding acc. joint / best code / best member (chance) | 0.08 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.05 |
| consumers / read jointly | 4 / 1 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L5 attn |
| CKA(arrangement before, joint write) | 0.43 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.03 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.30 2:0.11 20:0.07) |
| joint write: shape (spectrum k:share) | irregular (10:0.20 1:0.18 20:0.13) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.08 / 1.56 |

<details><summary>codes and components</summary>

**code 4a-443.0 (L5): 2 comps, tells apart 11/100, coverage 0.10, overlap 1.00 (random 1.00, p 0.68)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 down c3 | MLP | a%100 in {0, 80, 90} (+) | 0.54 | 1.000 |
| L5 down c9 | MLP | a%100 in {1..5, 24..25} (+) | 0.51 | 1.000 |

</details>

</details>

<details><summary><b>4a-444</b> `a%100` @ `=` (add) — block code; 2 comps, L7; tells apart 14/100 classes (best member 3); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.23 |
| classes told apart: joint / best code / best member | 14 /  / 3 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.57 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.57 |
| decoding acc. joint / best code / best member (chance) | 0.07 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L6 MLP |
| CKA(arrangement before, joint write) | 0.42 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.31 2:0.10 20:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.33 2:0.14 20:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.14 / 1.90 |
| best source position (CKA) | `b` (0.44) |

<details><summary>codes and components</summary>

**code 4a-444.0 (L7): 2 comps, tells apart 14/100, coverage 0.23, overlap 1.00 (random 1.00, p 0.59)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L7 down c2 | MLP | a%100 in {1..2, 18, 40, 50, 55, 60} (+) | 0.58 | 1.000 |
| L7 o c3 | H5 | a%100 in {4, 6, 8..9, 12..13, 90..99} (+) | 0.54 | 1.000 |

</details>

</details>

<details><summary><b>4a-445</b> `a%100` @ `=` (add) — copy from `b`; 1 comps, L8; tells apart 1/100 classes (best member 1); on sub: 4s-578 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.27 |
| classes told apart: joint / best code / best member | 1 /  / 1 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.75 |
| decoding acc. joint / best code / best member (chance) | 0.03 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L7 MLP |
| CKA(arrangement before, joint write) | 0.65 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.32 2:0.10 20:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.68 2:0.19) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.25 / 1.00 |
| best source position (CKA) | `b` (0.66) |

<details><summary>codes and components</summary>

**code 4a-445.0 (L8): 1 comps, tells apart 1/100, coverage 0.27, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L8 o c7 | H23 | a%100 in {5..10, 12..24, 26..27, 55, 78, 81, 86..88} (-) | 0.74 | 1.000 |

</details>

</details>

<details><summary><b>4a-446</b> `a%100` @ `=` (add) — single component; 1 comps, L10; tells apart 5/100 classes (best member 5); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.05 |
| classes told apart: joint / best code / best member | 5 /  / 5 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.56 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L9 MLP |
| CKA(arrangement before, joint write) | 0.46 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.14 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.36 2:0.10 40:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.34 20:0.07 40:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.07 / 1.00 |
| best source position (CKA) | `b` (0.50) |

<details><summary>codes and components</summary>

**code 4a-446.0 (L10): 1 comps, tells apart 5/100, coverage 0.05, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L10 o c54 | H15 | a%100 in {1..4, 90} (+) | 0.56 | 1.000 |

</details>

</details>

<details><summary><b>4a-447</b> `a%100` @ `=` (add) — single component; 1 comps, L13; tells apart 7/100 classes (best member 7); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.04 |
| classes told apart: joint / best code / best member | 7 /  / 7 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.64 |
| decoding acc. joint / best code / best member (chance) | 0.02 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L12 MLP |
| CKA(arrangement before, joint write) | 0.39 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.10 |
| share of the write inside the old arrangement's span | 0.12 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.39 2:0.10 40:0.05) |
| joint write: shape (spectrum k:share) | line (1:0.14 2:0.11 3:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.04 / 1.00 |
| best source position (CKA) | `b` (0.40) |

<details><summary>codes and components</summary>

**code 4a-447.0 (L13): 1 comps, tells apart 7/100, coverage 0.04, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 o c72 | H7 | a%100 in {1..4} (-) | 0.64 | 0.053 |

</details>

</details>

<details><summary><b>4a-449</b> `a%100` @ `=` (add) — block code; 3 comps, L16; tells apart 11/100 classes (best member 8); on sub: 4s-582 (member overlap 0.33)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.04 |
| classes told apart: joint / best code / best member | 11 /  / 8 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.06 / 0.29 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.72 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 MLP |
| CKA(arrangement before, joint write) | 0.07 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.37 2:0.13) |
| joint write: shape (spectrum k:share) | irregular (5:0.08 10:0.08 15:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.03 / 1.62 |
| best source position (CKA) | `a` (0.10) |

<details><summary>codes and components</summary>

**code 4a-449.0 (L16): 3 comps, tells apart 11/100, coverage 0.04, overlap 1.00 (random 1.07, p 0.31)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c470 | MLP | a%100 in {0} (-) | 0.51 | 0.015 |
| L16 o c61 | H22 | a%100 in {1} (+) | 0.81 | 0.012 |
| L16 down c214 | MLP | a%100 in {64, 84} (-) | 0.70 | 0.039 |

</details>

</details>

<details><summary><b>4a-448</b> `a%100` @ `=` (add) — block code, copy from `op`; 18 comps, L16; tells apart 66/100 classes (best member 7); on sub: 4s-581 (member overlap 0.39)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.88 |
| classes told apart: joint / best code / best member | 66 /  / 7 (of 100) |
| members whose removal merges classes | 0.39 |
| support overlap (1 = tiling) / random sets / p | 2.07 / 2.05 / 0.53 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.96 |
| decoding acc. joint / best code / best member (chance) | 0.90 /  / 0.13 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.07 |
| consumers / read jointly | 29 / 24 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 MLP |
| CKA(arrangement before, joint write) | 0.29 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.07 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.38 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.37 2:0.13) |
| joint write: shape (spectrum k:share) | irregular (1:0.32 2:0.26 3:0.11) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.52 / 5.29 |
| best source position (CKA) | `op` (0.53) |

<details><summary>codes and components</summary>

**code 4a-448.0 (L16): 18 comps, tells apart 66/100, coverage 0.88, overlap 2.07 (random 2.07, p 0.51)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c98 | H21 | a%100 in {0, 88..99} (-) | 0.99 | 0.440 |
| L16 o c35 | H21 | a%100 in {0, 96..99} (+) | 0.98 | 0.063 |
| L16 o c28 | H21 | a%100 in {1, 51, 53, 55..64} (+) | 0.95 | 0.323 |
| L16 o c299 | H21 | a%100 in {14..24} (+) | 0.83 | 0.157 |
| L16 o c90 | H21 | a%100 in {21..22, 24, 36..44, 81..84} (+) | 0.98 | 0.351 |
| L16 o c41 | H21 | a%100 in {25..41} (+) | 0.99 | 0.279 |
| L16 o c131 | H21 | a%100 in {25} (-) | 0.59 | 0.018 |
| L16 o c44 | H21 | a%100 in {36, 46} (+) | 0.59 | 0.038 |
| L16 o c164 | H21 | a%100 in {40..52} (-) | 0.94 | 0.156 |
| L16 o c120 | H21 | a%100 in {49..59} (-) | 0.96 | 0.172 |
| L16 o c129 | H21 | a%100 in {50, 60, 62, 64, 66, 68..70, 72} (-) | 0.96 | 0.611 |
| L16 o c94 | H21 | a%100 in {58..65, 67} (-) | 0.95 | 0.167 |
| L16 o c82 | H21 | a%100 in {65, 67..74} (-) | 0.86 | 0.093 |
| L16 o c72 | H21 | a%100 in {65..69, 85..89} (-) | 0.82 | 0.083 |
| L16 o c31 | H21 | a%100 in {70..72, 75..91} (-) | 0.98 | 0.244 |
| L16 o c155 | H21 | a%100 in {85..95} (-) | 0.85 | 0.116 |
| L16 o c24 | H21 | a%100 in {91..98} (-) | 0.86 | 0.128 |
| L16 o c36 | H21 | a%100 in {96..99} (-) | 0.79 | 0.047 |

</details>

</details>

<details><summary><b>4a-450</b> `a%100` @ `=` (add) — block code; 4 comps, L22 L23; tells apart 6/100 classes (best member 4); on sub: 4s-585 (member overlap 0.75)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.05 |
| classes told apart: joint / best code / best member | 6 /  / 4 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.13 / 0.12 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.66 |
| decoding acc. joint / best code / best member (chance) | 0.05 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 MLP |
| CKA(arrangement before, joint write) | 0.05 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.34 2:0.15 10:0.07) |
| joint write: shape (spectrum k:share) | irregular () |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.03 / 2.82 |
| best source position (CKA) | `a` (0.06) |

<details><summary>codes and components</summary>

**code 4a-450.0 (L22 L23): 4 comps, tells apart 6/100, coverage 0.05, overlap 1.00 (random 1.14, p 0.12)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 o c246 | H3 | a%100 in {47, 57} (+) | 0.59 | 0.029 |
| L22 o c352 | H3 | a%100 in {97} (-) | 0.69 | 0.015 |
| L23 o c370 | H22 | a%100 in {55} (-) | 0.79 | 0.011 |
| L23 o c187 | H7 | a%100 in {98} (-) | 0.77 | 0.014 |

</details>

</details>

<details><summary><b>4a-451</b> `a%100` @ `=` (add) — single component; 1 comps, L24; tells apart 2/100 classes (best member 2); on sub: 4s-577 (member overlap 0.06)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.01 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.89 |
| decoding acc. joint / best code / best member (chance) | 0.02 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L23 MLP |
| CKA(arrangement before, joint write) | 0.03 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.35 2:0.14 3:0.06) |
| joint write: shape (spectrum k:share) | line () |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.01 / 1.00 |
| best source position (CKA) | `b` (0.14) |

<details><summary>codes and components</summary>

**code 4a-451.0 (L24): 1 comps, tells apart 2/100, coverage 0.01, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 o c503 | H17 | a%100 in {90} (+) | 0.88 | 0.010 |

</details>

</details>

<details><summary><b>4a-452</b> `a%100` @ `=` (add) — single component; 1 comps, L26; tells apart 4/100 classes (best member 4); on sub: 4s-577 (member overlap 0.06)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.03 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.66 |
| decoding acc. joint / best code / best member (chance) | 0.02 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L25 MLP |
| CKA(arrangement before, joint write) | 0.07 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.35 2:0.14 3:0.05) |
| joint write: shape (spectrum k:share) | line (1:0.08 2:0.07 3:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.02 / 1.00 |
| best source position (CKA) | `op` (0.30) |

<details><summary>codes and components</summary>

**code 4a-452.0 (L26): 1 comps, tells apart 4/100, coverage 0.03, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 o c379 | H14 | a%100 in {92..94} (+) | 0.66 | 0.027 |

</details>

</details>

</details>

<details><summary>`b%100`: 10 mechanisms, 30 components</summary>

<details><summary><b>4a-461</b> `b%100` @ `=` (add) — block code; 3 comps, L0; tells apart 12/100 classes (best member 6); on sub: 4s-597 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.13 |
| classes told apart: joint / best code / best member | 12 /  / 6 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.23 / 1.11 / 0.82 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.84 |
| decoding acc. joint / best code / best member (chance) | 0.20 /  / 0.07 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 6 / 2 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 attn |
| CKA(arrangement before, joint write) | 0.50 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.03 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.10 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.25 2:0.11 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (2:0.16 1:0.11 3:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.05 / 1.58 |

<details><summary>codes and components</summary>

**code 4a-461.0 (L0): 3 comps, tells apart 12/100, coverage 0.13, overlap 1.23 (random 1.09, p 0.80)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 down c37 | MLP | b%100 in {0, 4..5, 7..8, 92, 98..99} (-) | 0.77 | 1.000 |
| L0 down c46 | MLP | b%100 in {0, 40, 50, 90} (-) | 0.88 | 1.000 |
| L0 down c4 | MLP | b%100 in {0..2, 4} (-) | 0.85 | 1.000 |

</details>

</details>

<details><summary><b>4a-462</b> `b%100` @ `=` (add) — single component; 1 comps, L1; tells apart 5/100 classes (best member 5); on sub: 4s-598 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.04 |
| classes told apart: joint / best code / best member | 5 /  / 5 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.76 |
| decoding acc. joint / best code / best member (chance) | 0.05 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.56 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.09 |
| write energy / code energy before | 0.03 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.26 2:0.11 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.31 4:0.14 2:0.14) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.07 / 1.00 |
| best source position (CKA) | `b` (0.30) |

<details><summary>codes and components</summary>

**code 4a-462.0 (L1): 1 comps, tells apart 5/100, coverage 0.04, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c10 | H11 | b%100 in {1..3, 5} (+) | 0.76 | 1.000 |

</details>

</details>

<details><summary><b>4a-463</b> `b%100` @ `=` (add) — single component; 1 comps, L1; tells apart 2/100 classes (best member 2); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.19 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.53 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.57 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.26 2:0.11 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.34 2:0.15 40:0.12) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.08 / 1.00 |
| best source position (CKA) | `op` (0.44) |

<details><summary>codes and components</summary>

**code 4a-463.0 (L1): 1 comps, tells apart 2/100, coverage 0.19, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c47 | H6 | b%100 in {1..10, 12, 18, 20, 40, 45, 51, 60, 71, 82} (+) | 0.52 | 1.000 |

</details>

</details>

<details><summary><b>4a-464</b> `b%100` @ `=` (add) — block code; 2 comps, L1; tells apart 9/100 classes (best member 5); on sub: 4s-599 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.12 |
| classes told apart: joint / best code / best member | 9 /  / 5 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.50 / 1.03 / 0.98 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.72 |
| decoding acc. joint / best code / best member (chance) | 0.10 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.24 |
| consumers / read jointly | 6 / 6 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L1 attn |
| CKA(arrangement before, joint write) | 0.43 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.23 2:0.12 20:0.11) |
| joint write: shape (spectrum k:share) | line (1:0.31 2:0.17 4:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.08 / 1.20 |

<details><summary>codes and components</summary>

**code 4a-464.0 (L1): 2 comps, tells apart 9/100, coverage 0.12, overlap 1.50 (random 1.00, p 0.96)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 down c20 | MLP | b%100 in {1..5, 8..9, 43} (+) | 0.69 | 1.000 |
| L1 down c9 | MLP | b%100 in {1..6, 9, 41, 55, 91} (-) | 0.61 | 1.000 |

</details>

</details>

<details><summary><b>4a-465</b> `b%100` @ `=` (add) — single component; 1 comps, L5; tells apart 1/100 classes (best member 1); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.24 |
| classes told apart: joint / best code / best member | 1 /  / 1 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.47 |
| decoding acc. joint / best code / best member (chance) | 0.03 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 2 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L4 MLP |
| CKA(arrangement before, joint write) | 0.44 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.22 2:0.11 20:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.51 2:0.12) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.14 / 1.00 |
| best source position (CKA) | `b` (0.47) |

<details><summary>codes and components</summary>

**code 4a-465.0 (L5): 1 comps, tells apart 1/100, coverage 0.24, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 o c6 | H26 | b%100 in {2, 5..6, 8, 10, 12..18, 20..26, 30, 48, 67, 79, 92} (-) | 0.47 | 1.000 |

</details>

</details>

<details><summary><b>4a-467</b> `b%100` @ `=` (add) — single component; 1 comps, L15; tells apart 3/100 classes (best member 3); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.10 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.84 |
| decoding acc. joint / best code / best member (chance) | 0.08 /  / 0.08 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L14 MLP |
| CKA(arrangement before, joint write) | 0.43 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.11 |
| write energy / code energy before | 0.03 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.31 2:0.13 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.74 50:0.05) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.28 / 1.00 |
| best source position (CKA) | `b` (0.41) |

<details><summary>codes and components</summary>

**code 4a-467.0 (L15): 1 comps, tells apart 3/100, coverage 0.10, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 o c446 | H18 | b%100 in {1..5, 7, 11, 54, 58, 60} (+) | 0.84 | 1.000 |

</details>

</details>

<details><summary><b>4a-466</b> `b%100` @ `=` (add) — block code, copy from `b`; 15 comps, L15; tells apart 52/100 classes (best member 6); on sub: 4s-600 (member overlap 0.28)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.89 |
| classes told apart: joint / best code / best member | 52 /  / 6 (of 100) |
| members whose removal merges classes | 0.53 |
| support overlap (1 = tiling) / random sets / p | 2.33 / 2.05 / 0.91 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.93 |
| decoding acc. joint / best code / best member (chance) | 0.75 /  / 0.11 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.24 |
| consumers / read jointly | 35 / 34 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L14 MLP |
| CKA(arrangement before, joint write) | 0.32 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.06 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.36 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.31 2:0.13 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (2:0.31 1:0.28 5:0.19) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.46 / 4.38 |
| best source position (CKA) | `b` (0.50) |

<details><summary>codes and components</summary>

**code 4a-466.0 (L15): 15 comps, tells apart 52/100, coverage 0.89, overlap 2.33 (random 2.06, p 0.89)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 o c107 | H13 | b%100 in {12, 14, 22, 24, 28..29, 31..34, 52..54, 72..74, 82..84, 92..94} (-) | 0.95 | 0.502 |
| L15 o c159 | H13 | b%100 in {18, 28, 38, 48, 58, 68, 78} (+) | 0.79 | 0.114 |
| L15 o c3 | H5 | b%100 in {1} (+) | 0.65 | 0.018 |
| L15 o c59 | H13 | b%100 in {25..31} (+) | 0.96 | 0.106 |
| L15 o c167 | H13 | b%100 in {31..32, 41, 61, 64, 71..72, 81..82, 84} (+) | 0.73 | 0.221 |
| L15 o c69 | H13 | b%100 in {36, 41, 79..85, 89} (+) | 0.68 | 0.241 |
| L15 o c136 | H13 | b%100 in {36, 56, 96} (-) | 0.88 | 0.035 |
| L15 o c78 | H13 | b%100 in {4, 14, 16, 18, 24, 54, 56, 64, 84} (+) | 0.77 | 0.213 |
| L15 o c122 | H13 | b%100 in {4, 35..44, 50, 52, 54..59, 62, 64, 95..96, 98..99} (-) | 0.96 | 0.492 |
| L15 o c19 | H13 | b%100 in {41..45, 47..63} (+) | 0.96 | 0.342 |
| L15 o c151 | H13 | b%100 in {45..46, 48..54} (+) | 0.89 | 0.111 |
| L15 o c81 | H13 | b%100 in {5..10, 18..24, 45..54, 65..66, 75..82, 90..91, 93} (-) | 0.97 | 0.691 |
| L15 o c105 | H13 | b%100 in {56..64} (+) | 0.88 | 0.117 |
| L15 o c17 | H13 | b%100 in {59, 63..75, 79} (-) | 0.95 | 0.591 |
| L15 o c58 | H13 | b%100 in {60..81} (-) | 0.97 | 0.396 |

</details>

</details>

<details><summary><b>4a-468</b> `b%100` @ `=` (add) — block code; 3 comps, L16; tells apart 9/100 classes (best member 7); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.21 |
| classes told apart: joint / best code / best member | 9 /  / 7 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.43 / 1.11 / 0.95 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.75 |
| decoding acc. joint / best code / best member (chance) | 0.05 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.12 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L16 attn |
| CKA(arrangement before, joint write) | 0.48 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.08 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.29 2:0.18 20:0.10) |
| joint write: shape (spectrum k:share) | irregular (1:0.29 2:0.24 3:0.16) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.18 / 2.00 |

<details><summary>codes and components</summary>

**code 4a-468.0 (L16): 3 comps, tells apart 9/100, coverage 0.21, overlap 1.43 (random 1.09, p 0.96)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c33 | MLP | b%100 in {1..11} (-) | 0.74 | 0.144 |
| L16 down c161 | MLP | b%100 in {1..7} (+) | 0.90 | 0.075 |
| L16 down c49 | MLP | b%100 in {10..21} (+) | 0.68 | 0.150 |

</details>

</details>

<details><summary><b>4a-469</b> `b%100` @ `=` (add) — block code; 2 comps, L17 L18; tells apart 7/100 classes (best member 6); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.07 |
| classes told apart: joint / best code / best member | 7 /  / 6 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.54 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.79 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L17 attn |
| CKA(arrangement before, joint write) | 0.20 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.02 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.27 2:0.18 5:0.10) |
| joint write: shape (spectrum k:share) | line (5:0.16 1:0.14 4:0.11) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.05 / 1.09 |
| best source position (CKA) | `b` (0.22) |

<details><summary>codes and components</summary>

**code 4a-469.0 (L17 L18): 2 comps, tells apart 7/100, coverage 0.07, overlap 1.00 (random 1.00, p 0.55)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 down c170 | MLP | b%100 in {5..10} (-) | 0.80 | 0.093 |
| L18 o c356 | H30 | b%100 in {1} (+) | 0.63 | 0.031 |

</details>

</details>

<details><summary><b>4a-470</b> `b%100` @ `=` (add) — single component; 1 comps, L22; tells apart 6/100 classes (best member 6); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.03 |
| classes told apart: joint / best code / best member | 6 /  / 6 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.62 |
| decoding acc. joint / best code / best member (chance) | 0.03 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 MLP |
| CKA(arrangement before, joint write) | 0.12 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.15 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.28 2:0.18 5:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.14 2:0.12 3:0.11) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.04 / 1.00 |
| best source position (CKA) | `b` (0.16) |

<details><summary>codes and components</summary>

**code 4a-470.0 (L22): 1 comps, tells apart 6/100, coverage 0.03, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 o c200 | H15 | b%100 in {0, 98..99} (-) | 0.62 | 0.129 |

</details>

</details>

</details>

<details><summary>`a%10`: 1 mechanisms, 16 components</summary>

<details><summary><b>4a-437</b> `a%10` @ `=` (add) — block code, copy from `a`; 16 comps, L16; tells apart 10/10 classes (best member 4); on sub: 4s-573 (member overlap 0.25)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 10 /  / 4 (of 10) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 3.20 /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.97 |
| decoding acc. joint / best code / best member (chance) | 0.99 /  / 0.41 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.27 |
| consumers / read jointly | 68 / 67 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 MLP |
| CKA(arrangement before, joint write) | 0.49 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.18 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 3.82 |
| arrangement before: shape (spectrum k:share) | irregular (20:0.30 10:0.27 40:0.25) |
| joint write: shape (spectrum k:share) | simplex (one-hot like) (10:0.39 20:0.35 50:0.13) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.87 / 5.38 |
| best source position (CKA) | `a` (0.80) |

<details><summary>codes and components</summary>

**code 4a-437.0 (L16): 16 comps, tells apart 10/10, coverage 1.00, overlap 3.20 (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c34 | H21 | a%10 in {0..2} (+) | 0.93 | 0.295 |
| L16 o c53 | H21 | a%10 in {1, 3, 9} (+) | 0.97 | 0.406 |
| L16 o c257 | H21 | a%10 in {1} (+) | 0.90 | 0.092 |
| L16 o c26 | H21 | a%10 in {2, 5} (-) | 0.98 | 0.198 |
| L16 o c45 | H21 | a%10 in {2..3, 7} (-) | 0.95 | 0.391 |
| L16 o c121 | H21 | a%10 in {2..4} (-) | 0.97 | 0.395 |
| L16 o c163 | H21 | a%10 in {2} (-) | 0.90 | 0.093 |
| L16 o c117 | H21 | a%10 in {3..4} (-) | 0.98 | 0.299 |
| L16 o c137 | H21 | a%10 in {4} (-) | 0.92 | 0.107 |
| L16 o c156 | H21 | a%10 in {4} (-) | 0.92 | 0.095 |
| L16 o c97 | H21 | a%10 in {5..7} (+) | 0.98 | 0.306 |
| L16 o c142 | H21 | a%10 in {6..8} (-) | 0.95 | 0.293 |
| L16 o c180 | H21 | a%10 in {7..9} (-) | 0.97 | 0.299 |
| L16 o c101 | H21 | a%10 in {8} (+) | 0.94 | 0.099 |
| L16 o c251 | H21 | a%10 in {8} (-) | 0.97 | 0.104 |
| L16 o c154 | H21 | a%10 in {9} (-) | 0.98 | 0.100 |

</details>

</details>

</details>

<details><summary>`b%10`: 3 mechanisms, 15 components</summary>

<details><summary><b>4a-459</b> `b%10` @ `=` (add) — block code; 4 comps, L15; tells apart 7/10 classes (best member 3); on sub: 4s-593 (member overlap 0.60)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.90 |
| classes told apart: joint / best code / best member | 7 /  / 3 (of 10) |
| members whose removal merges classes | 0.75 |
| support overlap (1 = tiling) / random sets / p | 1.33 / 1.50 / 0.27 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.92 |
| decoding acc. joint / best code / best member (chance) | 0.71 /  / 0.33 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.04 |
| consumers / read jointly | 23 / 11 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 attn |
| CKA(arrangement before, joint write) | 0.52 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.06 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.33 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (20:0.31 10:0.30 50:0.21) |
| joint write: shape (spectrum k:share) | irregular (10:0.83 20:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.56 / 1.58 |

<details><summary>codes and components</summary>

**code 4a-459.0 (L15): 4 comps, tells apart 7/10, coverage 0.90, overlap 1.33 (random 1.50, p 0.30)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c172 | MLP | b%10 in {0} (-) | 0.95 | 0.102 |
| L15 down c21 | MLP | b%10 in {1..4} (-) | 0.92 | 0.507 |
| L15 down c129 | MLP | b%10 in {4..6} (+) | 0.95 | 0.298 |
| L15 down c67 | MLP | b%10 in {5..8} (-) | 0.83 | 0.372 |

</details>

</details>

<details><summary><b>4a-458</b> `b%10` @ `=` (add) — block code, copy from `b`; 10 comps, L15; tells apart 10/10 classes (best member 6); on sub: 4s-592 (member overlap 0.80)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 10 /  / 6 (of 10) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 2.80 / 2.80 / 0.59 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.95 |
| decoding acc. joint / best code / best member (chance) | 0.99 /  / 0.41 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.21 |
| consumers / read jointly | 39 / 39 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L14 MLP |
| CKA(arrangement before, joint write) | 0.75 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.12 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 2.88 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (20:0.29 10:0.27 50:0.20) |
| joint write: shape (spectrum k:share) | symmetric mix of circles (10:0.42 20:0.40 50:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.84 / 4.64 |
| best source position (CKA) | `b` (0.84) |

<details><summary>codes and components</summary>

**code 4a-458.0 (L15): 10 comps, tells apart 10/10, coverage 1.00, overlap 2.80 (random 2.80, p 0.53)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 o c132 | H13 | b%10 in {0, 8..9} (+) | 0.92 | 0.296 |
| L15 o c33 | H13 | b%10 in {0..1, 9} (-) | 0.92 | 0.305 |
| L15 o c139 | H13 | b%10 in {0} (+) | 0.87 | 0.138 |
| L15 o c178 | H13 | b%10 in {1..3} (+) | 0.93 | 0.503 |
| L15 o c111 | H13 | b%10 in {2..5, 9} (+) | 0.92 | 0.593 |
| L15 o c168 | H13 | b%10 in {3..5} (+) | 0.96 | 0.303 |
| L15 o c129 | H13 | b%10 in {4..6} (+) | 0.96 | 0.298 |
| L15 o c73 | H13 | b%10 in {5..7} (-) | 0.96 | 0.301 |
| L15 o c93 | H13 | b%10 in {5} (+) | 0.96 | 0.109 |
| L15 o c53 | H13 | b%10 in {7..9} (+) | 0.95 | 0.395 |

</details>

</details>

<details><summary><b>4a-460</b> `b%10` @ `=` (add) — single component; 1 comps, L16; tells apart 4/10 classes (best member 4); on sub: 4s-594 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.20 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 10) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.83 |
| decoding acc. joint / best code / best member (chance) | 0.24 /  / 0.24 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L16 attn |
| CKA(arrangement before, joint write) | 0.39 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (20:0.34 10:0.32 50:0.20) |
| joint write: shape (spectrum k:share) | line (10:0.50 20:0.31 30:0.12) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.18 / 1.00 |

<details><summary>codes and components</summary>

**code 4a-460.0 (L16): 1 comps, tells apart 4/10, coverage 0.20, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c295 | MLP | b%10 in {6..7} (+) | 0.83 | 0.230 |

</details>

</details>

</details>

<details><summary>`res%50`: 5 mechanisms, 12 components</summary>

<details><summary><b>4a-520</b> `res%50` @ `=` (add) — block code; 2 comps, L19; tells apart 1/50 classes (best member 1); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.92 |
| classes told apart: joint / best code / best member | 1 /  / 1 (of 50) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 1.17 / 1.00 / 0.92 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.79 |
| decoding acc. joint / best code / best member (chance) | 0.11 /  / 0.06 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.02 |
| consumers / read jointly | 35 / 16 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L19 attn |
| CKA(arrangement before, joint write) | 0.47 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.18 |
| share of the write inside the old arrangement's span | 0.14 |
| write energy / code energy before | 0.23 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.34 50:0.25 20:0.22) |
| joint write: shape (spectrum k:share) | line (2:0.93 4:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.60 / 1.39 |

<details><summary>codes and components</summary>

**code 4a-520.0 (L19): 2 comps, tells apart 1/50, coverage 0.92, overlap 1.17 (random 1.00, p 0.95)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L19 down c20 | MLP | res%50 in {0..2, 7..24, 33..49} (+) | 0.73 | 0.596 |
| L19 down c0 | MLP | res%50 in {20..35} (-) | 0.80 | 0.515 |

</details>

</details>

<details><summary><b>4a-521</b> `res%50` @ `=` (add) — block code; 2 comps, L27; tells apart 9/50 classes (best member 6); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.24 |
| classes told apart: joint / best code / best member | 9 /  / 6 (of 50) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.66 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.79 |
| decoding acc. joint / best code / best member (chance) | 0.09 /  / 0.06 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L27 attn |
| CKA(arrangement before, joint write) | 0.38 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.22 10:0.18 20:0.16) |
| joint write: shape (spectrum k:share) | irregular (20:0.30 10:0.29 40:0.17) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.21 / 1.88 |

<details><summary>codes and components</summary>

**code 4a-521.0 (L27): 2 comps, tells apart 9/50, coverage 0.24, overlap 1.00 (random 1.00, p 0.68)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 down c224 | MLP | res%50 in {0, 10, 20, 30, 40} (-) | 0.79 | 0.155 |
| L27 down c69 | MLP | res%50 in {7..8, 18, 28, 38, 47..48} (+) | 0.79 | 0.158 |

</details>

</details>

<details><summary><b>4a-522</b> `res%50` @ `=` (add) — block code; 3 comps, L28; tells apart 10/50 classes (best member 5); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.14 |
| classes told apart: joint / best code / best member | 10 /  / 5 (of 50) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.29 / 1.09 / 0.96 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.83 |
| decoding acc. joint / best code / best member (chance) | 0.11 /  / 0.05 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.08 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L28 attn |
| CKA(arrangement before, joint write) | 0.31 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.06 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.23 10:0.16 20:0.13) |
| joint write: shape (spectrum k:share) | irregular (20:0.17 40:0.17 10:0.16) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.14 / 2.28 |

<details><summary>codes and components</summary>

**code 4a-522.0 (L28): 3 comps, tells apart 10/50, coverage 0.14, overlap 1.29 (random 1.08, p 0.98)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 down c359 | MLP | res%50 in {0, 20, 40} (+) | 0.80 | 0.111 |
| L28 down c277 | MLP | res%50 in {0, 49} (-) | 0.78 | 0.044 |
| L28 down c129 | MLP | res%50 in {9, 19, 39, 49} (-) | 0.87 | 0.089 |

</details>

</details>

<details><summary><b>4a-523</b> `res%50` @ `=` (add) — block code; 4 comps, L29 L30; tells apart 12/50 classes (best member 8); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.12 |
| classes told apart: joint / best code / best member | 12 /  / 8 (of 50) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.14 / 0.15 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.83 |
| decoding acc. joint / best code / best member (chance) | 0.16 /  / 0.09 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L29 attn |
| CKA(arrangement before, joint write) | 0.19 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.02 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.20 10:0.15 20:0.13) |
| joint write: shape (spectrum k:share) | irregular (20:0.14 10:0.10 30:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.07 / 1.89 |

<details><summary>codes and components</summary>

**code 4a-523.0 (L29 L30): 4 comps, tells apart 12/50, coverage 0.12, overlap 1.00 (random 1.15, p 0.17)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L29 down c865 | MLP | res%50 in {13} (-) | 0.80 | 0.022 |
| L29 down c72 | MLP | res%50 in {20, 22} (-) | 0.81 | 0.129 |
| L30 down c305 | MLP | res%50 in {1} (+) | 0.79 | 0.024 |
| L30 down c133 | MLP | res%50 in {8, 48} (+) | 0.85 | 0.086 |

</details>

</details>

<details><summary><b>4a-524</b> `res%50` @ `=` (add) — single component; 1 comps, L31; tells apart 3/50 classes (best member 3); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.02 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 50) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.79 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.04 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L31 attn |
| CKA(arrangement before, joint write) | 0.11 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.16 10:0.13 20:0.12) |
| joint write: shape (spectrum k:share) | line (10:0.07 2:0.07 4:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.02 / 1.00 |

<details><summary>codes and components</summary>

**code 4a-524.0 (L31): 1 comps, tells apart 3/50, coverage 0.02, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L31 down c819 | MLP | res%50 in {36} (-) | 0.79 | 0.035 |

</details>

</details>

</details>

<details><summary>`b//10`: 2 mechanisms, 11 components</summary>

<details><summary><b>4a-477</b> `b//10` @ `=` (add) — 2 codes of the same shape, copy from `b`; 2 comps, L0; tells apart 4/11 classes (best member 4); on sub: 4s-607 (member overlap 0.50)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.18 |
| classes told apart: joint / best code / best member | 4 / 4 / 4 (of 11) |
| members whose removal merges classes | 0.50 |
| support overlap (1 = tiling) / random sets / p | 1.50 / 1.00 / 0.91 |
| mean CKA between its codes | 0.74 |
| purity of the joint write (per prompt) | 0.85 |
| decoding acc. joint / best code / best member (chance) | 0.57 / 0.59 / 0.59 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 0.49 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | embed |
| CKA(arrangement before, joint write) | 0.00 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.00 |
| write energy / code energy before | 45144240223180538989510656.00 |
| best source position (CKA) | `b` (0.50) |

<details><summary>codes and components</summary>

**code 4a-477.0 (L0): 1 comps, tells apart 4/11, coverage 0.18, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 o c7 | H0 | b//10 in {0, 10} (+) | 0.96 | 1.000 |

**code 4a-477.1 (L0): 1 comps, tells apart 3/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 o c2 | H2 | b//10 in {0} (-) | 0.87 | 1.000 |

</details>

</details>

<details><summary><b>4a-478</b> `b//10` @ `=` (add) — block code, 2 codes of the same shape, copy from `b`; 9 comps, L15; tells apart 11/11 classes (best member 5); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.91 |
| classes told apart: joint / best code / best member | 11 / 10 / 5 (of 11) |
| members whose removal merges classes | 0.22 |
| support overlap (1 = tiling) / random sets / p | 2.80 / 2.70 / 0.88 |
| mean CKA between its codes | 0.79 |
| purity of the joint write (per prompt) | 0.91 |
| decoding acc. joint / best code / best member (chance) | 0.88 / 0.84 / 0.34 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.10 |
| consumers / read jointly | 68 / 64 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L14 MLP |
| CKA(arrangement before, joint write) | 0.73 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.06 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 2.08 |
| best source position (CKA) | `b` (0.84) |

<details><summary>codes and components</summary>

**code 4a-478.0 (L15): 6 comps, tells apart 10/11, coverage 0.73, overlap 1.88 (random 2.00, p 0.25)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 o c75 | H13 | b//10 in {0..1} (-) | 0.86 | 0.337 |
| L15 o c191 | H13 | b//10 in {1..2} (-) | 0.90 | 0.307 |
| L15 o c30 | H13 | b//10 in {1..3} (+) | 0.93 | 0.465 |
| L15 o c77 | H13 | b//10 in {2..4} (-) | 0.88 | 0.352 |
| L15 o c82 | H13 | b//10 in {8..10} (+) | 0.93 | 0.355 |
| L15 o c153 | H13 | b//10 in {9..10} (-) | 0.87 | 0.116 |

**code 4a-478.1 (L15): 3 comps, tells apart 9/11, coverage 0.82, overlap 1.44 (random 1.33, p 0.65)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c4 | MLP | b//10 in {0..3, 6..8} (-) | 0.92 | 0.602 |
| L15 down c54 | MLP | b//10 in {1..3} (-) | 0.93 | 0.431 |
| L15 down c29 | MLP | b//10 in {8..10} (-) | 0.87 | 0.407 |

</details>

</details>

</details>

<details><summary>`res%20`: 4 mechanisms, 10 components</summary>

<details><summary><b>4a-515</b> `res%20` @ `=` (add) — block code; 2 comps, L20; tells apart 8/20 classes (best member 4); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.80 |
| classes told apart: joint / best code / best member | 8 /  / 4 (of 20) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.19 / 1.03 / 0.71 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.68 |
| decoding acc. joint / best code / best member (chance) | 0.26 /  / 0.13 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.04 |
| consumers / read jointly | 42 / 31 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L20 attn |
| CKA(arrangement before, joint write) | 0.14 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.13 |
| share of the write inside the old arrangement's span | 0.41 |
| write energy / code energy before | 0.37 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (10:0.32 50:0.28 20:0.26) |
| joint write: shape (spectrum k:share) | circle period 20 (5:0.96) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.77 / 1.65 |

<details><summary>codes and components</summary>

**code 4a-515.0 (L20): 2 comps, tells apart 8/20, coverage 0.80, overlap 1.19 (random 1.03, p 0.67)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 down c3 | MLP | res%20 in {10..15} (+) | 0.68 | 0.607 |
| L20 down c6 | MLP | res%20 in {4..10, 14..19} (-) | 0.65 | 0.604 |

</details>

</details>

<details><summary><b>4a-516</b> `res%20` @ `=` (add) — block code; 5 comps, L21 L22; tells apart 9/20 classes (best member 6); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.50 |
| classes told apart: joint / best code / best member | 9 /  / 6 (of 20) |
| members whose removal merges classes | 0.80 |
| support overlap (1 = tiling) / random sets / p | 1.30 / 1.43 / 0.28 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.80 |
| decoding acc. joint / best code / best member (chance) | 0.37 /  / 0.15 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.11 |
| consumers / read jointly | 22 / 19 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 attn |
| CKA(arrangement before, joint write) | 0.42 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.13 |
| share of the write inside the old arrangement's span | 0.23 |
| write energy / code energy before | 0.13 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (20:0.32 10:0.27 5:0.27) |
| joint write: shape (spectrum k:share) | irregular (5:0.60 10:0.21 15:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.42 / 2.07 |

<details><summary>codes and components</summary>

**code 4a-516.0 (L21 L22): 5 comps, tells apart 9/20, coverage 0.50, overlap 1.30 (random 1.44, p 0.29)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c2 | MLP | res%20 in {11..15} (-) | 0.71 | 0.363 |
| L21 down c231 | MLP | res%20 in {11} (+) | 0.65 | 0.063 |
| L22 down c11 | MLP | res%20 in {0..3, 19} (-) | 0.86 | 0.325 |
| L22 down c427 | MLP | res%20 in {12} (+) | 0.83 | 0.065 |
| L22 down c17 | MLP | res%20 in {2} (-) | 0.83 | 0.105 |

</details>

</details>

<details><summary><b>4a-517</b> `res%20` @ `=` (add) — block code; 2 comps, L23 L24; tells apart 5/20 classes (best member 3); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.15 |
| classes told apart: joint / best code / best member | 5 /  / 3 (of 20) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.17 / 0.42 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.78 |
| decoding acc. joint / best code / best member (chance) | 0.15 /  / 0.10 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L23 attn |
| CKA(arrangement before, joint write) | 0.27 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.02 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (10:0.28 5:0.26 20:0.21) |
| joint write: shape (spectrum k:share) | irregular (10:0.18 20:0.18 30:0.18) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.14 / 1.88 |

<details><summary>codes and components</summary>

**code 4a-517.0 (L23 L24): 2 comps, tells apart 5/20, coverage 0.15, overlap 1.00 (random 1.00, p 0.51)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 down c125 | MLP | res%20 in {11} (+) | 0.86 | 0.061 |
| L24 down c210 | MLP | res%20 in {2, 12} (-) | 0.74 | 0.074 |

</details>

</details>

<details><summary><b>4a-518</b> `res%20` @ `=` (add) — single component; 1 comps, L30; tells apart 4/20 classes (best member 4); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.20 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 20) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.81 |
| decoding acc. joint / best code / best member (chance) | 0.16 /  / 0.16 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L30 attn |
| CKA(arrangement before, joint write) | 0.22 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (10:0.20 20:0.18 5:0.18) |
| joint write: shape (spectrum k:share) | line (40:0.35 30:0.26 50:0.19) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.14 / 1.00 |

<details><summary>codes and components</summary>

**code 4a-518.0 (L30): 1 comps, tells apart 4/20, coverage 0.20, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 down c239 | MLP | res%20 in {2..3, 12..13} (+) | 0.81 | 0.185 |

</details>

</details>

</details>

<details><summary>`a//10`: 2 mechanisms, 9 components</summary>

<details><summary><b>4a-456</b> `a//10` @ `=` (add) — copy from `op`; 1 comps, L2; tells apart 3/11 classes (best member 3); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.09 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 11) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.84 |
| decoding acc. joint / best code / best member (chance) | 0.19 /  / 0.19 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 4 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L1 MLP |
| CKA(arrangement before, joint write) | 0.81 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.11 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 1.49 |
| best source position (CKA) | `op` (0.75) |

<details><summary>codes and components</summary>

**code 4a-456.0 (L2): 1 comps, tells apart 3/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 o c217 | H2 | a//10 in {0} (+) | 0.84 | 0.146 |

</details>

</details>

<details><summary><b>4a-457</b> `a//10` @ `=` (add) — block code, 2 codes of the same shape, copy from `a`; 8 comps, L16; tells apart 11/11 classes (best member 5); on sub: 4s-589 (member overlap 0.20)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.82 |
| classes told apart: joint / best code / best member | 11 / 10 / 5 (of 11) |
| members whose removal merges classes | 0.25 |
| support overlap (1 = tiling) / random sets / p | 2.22 / 2.11 / 0.93 |
| mean CKA between its codes | 0.74 |
| purity of the joint write (per prompt) | 0.93 |
| decoding acc. joint / best code / best member (chance) | 0.94 / 0.92 / 0.40 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 0.80 |
| consumers / read jointly | 55 / 53 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 MLP |
| CKA(arrangement before, joint write) | 0.75 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.17 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.74 |
| best source position (CKA) | `a` (0.85) |

<details><summary>codes and components</summary>

**code 4a-457.0 (L16): 6 comps, tells apart 10/11, coverage 0.82, overlap 1.44 (random 1.71, p 0.07)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c219 | H21 | a//10 in {0..1} (+) | 0.94 | 0.234 |
| L16 o c284 | H21 | a//10 in {0} (-) | 0.93 | 0.109 |
| L16 o c87 | H21 | a//10 in {1..3} (-) | 0.94 | 0.447 |
| L16 o c138 | H21 | a//10 in {3..5} (-) | 0.93 | 0.370 |
| L16 o c52 | H21 | a//10 in {7..8} (+) | 0.92 | 0.314 |
| L16 o c16 | H21 | a//10 in {8..9} (+) | 0.93 | 0.303 |

**code 4a-457.1 (L16): 2 comps, tells apart 5/11, coverage 0.64, overlap 1.00 (random 1.00, p 0.65)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c207 | H30 | a//10 in {0..3} (+) | 0.92 | 0.558 |
| L16 o c160 | H22 | a//10 in {7..9} (-) | 0.96 | 0.510 |

</details>

</details>

</details>

<details><summary>`a%20`: 1 mechanisms, 5 components</summary>

<details><summary><b>4a-453</b> `a%20` @ `=` (add) — block code, 2 codes of the same shape; 5 comps, L16; tells apart 11/20 classes (best member 5); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 11 / 11 / 5 (of 20) |
| members whose removal merges classes | 0.60 |
| support overlap (1 = tiling) / random sets / p | 1.55 /  /  |
| mean CKA between its codes | 0.80 |
| purity of the joint write (per prompt) | 0.94 |
| decoding acc. joint / best code / best member (chance) | 0.65 / 0.63 / 0.24 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.28 |
| consumers / read jointly | 10 / 9 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 MLP |
| CKA(arrangement before, joint write) | 0.30 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.07 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 1.07 |
| arrangement before: shape (spectrum k:share) | irregular (20:0.22 10:0.19 5:0.19) |
| joint write: shape (spectrum k:share) | irregular (5:0.67 10:0.16) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.38 / 1.61 |
| best source position (CKA) | `a` (0.38) |

<details><summary>codes and components</summary>

**code 4a-453.0 (L16): 1 comps, tells apart 3/20, coverage 0.45, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c73 | MLP | a%20 in {0..3, 15..19} (+) | 0.90 | 0.488 |

**code 4a-453.1 (L16): 4 comps, tells apart 11/20, coverage 0.85, overlap 1.29 (random 1.35, p 0.13)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c195 | H21 | a%20 in {0, 9..10, 18..19} (+) | 0.95 | 0.449 |
| L16 o c116 | H21 | a%20 in {4, 6, 14, 16} (-) | 0.82 | 0.215 |
| L16 o c25 | H21 | a%20 in {5, 7, 15, 17} (+) | 0.91 | 0.202 |
| L16 o c144 | H21 | a%20 in {5..13} (+) | 0.95 | 0.500 |

</details>

</details>

</details>

<details><summary>`b%50`: 2 mechanisms, 5 components</summary>

<details><summary><b>4a-475</b> `b%50` @ `=` (add) — block code, 2 codes of the same shape; 4 comps, L15 L16; tells apart 3/50 classes (best member 2); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.96 |
| classes told apart: joint / best code / best member | 3 / 4 / 2 (of 50) |
| members whose removal merges classes | 0.25 |
| support overlap (1 = tiling) / random sets / p | 1.67 / 1.94 / 0.41 |
| mean CKA between its codes | 0.71 |
| purity of the joint write (per prompt) | 0.90 |
| decoding acc. joint / best code / best member (chance) | 0.25 / 0.26 / 0.09 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.15 |
| consumers / read jointly | 21 / 19 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 attn |
| CKA(arrangement before, joint write) | 0.57 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.11 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.45 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.28 20:0.15 10:0.15) |
| joint write: shape (spectrum k:share) | circle period 50 (2:0.92) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.64 / 1.51 |

<details><summary>codes and components</summary>

**code 4a-475.0 (L15): 3 comps, tells apart 4/50, coverage 0.82, overlap 1.56 (random 1.46, p 0.66)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c41 | MLP | b%50 in {19..29} (-) | 0.90 | 0.527 |
| L15 down c19 | MLP | b%50 in {2..24, 29..42} (-) | 0.92 | 0.619 |
| L15 down c35 | MLP | b%50 in {6..7, 9..22} (-) | 0.94 | 0.555 |

**code 4a-475.1 (L16): 1 comps, tells apart 1/50, coverage 0.32, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c42 | MLP | b%50 in {34..49} (-) | 0.80 | 0.479 |

</details>

</details>

<details><summary><b>4a-476</b> `b%50` @ `=` (add) — single component; 1 comps, L17; tells apart 1/50 classes (best member 1); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.66 |
| classes told apart: joint / best code / best member | 1 /  / 1 (of 50) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.90 |
| decoding acc. joint / best code / best member (chance) | 0.07 /  / 0.07 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 7 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L17 attn |
| CKA(arrangement before, joint write) | 0.38 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.07 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.35 10:0.17 20:0.15) |
| joint write: shape (spectrum k:share) | line (2:0.98) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.48 / 1.00 |

<details><summary>codes and components</summary>

**code 4a-476.0 (L17): 1 comps, tells apart 1/50, coverage 0.66, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 down c10 | MLP | b%50 in {0..11, 20..31, 33..34, 43..49} (-) | 0.90 | 0.650 |

</details>

</details>

</details>

<details><summary>`a%50`: 2 mechanisms, 3 components</summary>

<details><summary><b>4a-454</b> `a%50` @ `=` (add) — block code; 2 comps, L16; tells apart 6/50 classes (best member 5); on sub: 4s-588 (member overlap 0.25)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.38 |
| classes told apart: joint / best code / best member | 6 /  / 5 (of 50) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.09 / 0.31 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.85 |
| decoding acc. joint / best code / best member (chance) | 0.18 /  / 0.10 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 0.96 |
| consumers / read jointly | 2 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 MLP |
| CKA(arrangement before, joint write) | 0.23 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.09 |
| write energy / code energy before | 0.09 |
| arrangement before: shape (spectrum k:share) | irregular (2:0.31 4:0.10 20:0.10) |
| joint write: shape (spectrum k:share) | irregular (20:0.30 2:0.22 40:0.15) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.19 / 1.95 |
| best source position (CKA) | `a` (0.43) |

<details><summary>codes and components</summary>

**code 4a-454.0 (L16): 2 comps, tells apart 6/50, coverage 0.38, overlap 1.00 (random 1.09, p 0.33)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c37 | H21 | a%50 in {0, 5, 10, 14..15, 20, 25, 30, 35, 40, 45} (+) | 0.83 | 0.342 |
| L16 o c48 | H21 | a%50 in {7, 9, 13, 17, 19, 29, 39, 49} (-) | 0.89 | 0.475 |

</details>

</details>

<details><summary><b>4a-455</b> `a%50` @ `=` (add) — single component; 1 comps, L17; tells apart 1/50 classes (best member 1); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.28 |
| classes told apart: joint / best code / best member | 1 /  / 1 (of 50) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.89 |
| decoding acc. joint / best code / best member (chance) | 0.08 /  / 0.08 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 2 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L17 attn |
| CKA(arrangement before, joint write) | 0.39 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.05 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.25 10:0.20 20:0.18) |
| joint write: shape (spectrum k:share) | line (2:0.93 4:0.05) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.44 / 1.00 |

<details><summary>codes and components</summary>

**code 4a-455.0 (L17): 1 comps, tells apart 1/50, coverage 0.28, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 down c33 | MLP | a%50 in {26..39} (-) | 0.89 | 0.602 |

</details>

</details>

</details>

<details><summary>`b%20`: 2 mechanisms, 3 components</summary>

<details><summary><b>4a-472</b> `b%20` @ `=` (add) — single component; 1 comps, L15; tells apart 5/20 classes (best member 5); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.15 |
| classes told apart: joint / best code / best member | 5 /  / 5 (of 20) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.87 |
| decoding acc. joint / best code / best member (chance) | 0.18 /  / 0.18 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L14 MLP |
| CKA(arrangement before, joint write) | 0.27 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.06 |
| arrangement before: shape (spectrum k:share) | irregular (20:0.20 10:0.18 5:0.17) |
| joint write: shape (spectrum k:share) | line (20:0.32 10:0.28 50:0.22) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.14 / 1.00 |
| best source position (CKA) | `b` (0.32) |

<details><summary>codes and components</summary>

**code 4a-472.0 (L15): 1 comps, tells apart 5/20, coverage 0.15, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 o c120 | H13 | b%20 in {9, 11, 19} (-) | 0.87 | 0.348 |

</details>

</details>

<details><summary><b>4a-473</b> `b%20` @ `=` (add) — 2 codes of the same shape; 2 comps, L15 L16; tells apart 2/20 classes (best member 2); on sub: 4s-603 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 2 / 2 / 2 (of 20) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 2.00 / 1.15 / 1.00 |
| mean CKA between its codes | 0.99 |
| purity of the joint write (per prompt) | 0.93 |
| decoding acc. joint / best code / best member (chance) | 0.12 / 0.12 / 0.12 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.15 |
| consumers / read jointly | 16 / 13 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 attn |
| CKA(arrangement before, joint write) | 0.35 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 0.42 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (20:0.24 10:0.23 50:0.17) |
| joint write: shape (spectrum k:share) | line (5:0.82 15:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.34 / 1.00 |

<details><summary>codes and components</summary>

**code 4a-473.0 (L15): 1 comps, tells apart 2/20, coverage 1.00, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c20 | MLP | b%20 in {0..19} (+) | 0.91 | 0.495 |

**code 4a-473.1 (L16): 1 comps, tells apart 2/20, coverage 1.00, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c24 | MLP | b%20 in {0..19} (+) | 0.94 | 0.500 |

</details>

</details>

</details>

<details><summary>`b%2`: 1 mechanisms, 2 components</summary>

<details><summary><b>4a-471</b> `b%2` @ `=` (add) — 2 codes of the same shape, copy from `b`; 2 comps, L15; tells apart 2/2 classes (best member 2); on sub: 4s-602 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 2 / 2 / 2 (of 2) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 2.00 /  /  |
| mean CKA between its codes | 1.00 |
| purity of the joint write (per prompt) | 0.94 |
| decoding acc. joint / best code / best member (chance) | 1.00 / 1.00 / 1.00 (0.50) |
| kappa (1 orthogonal, >1 constructive) | 1.18 |
| consumers / read jointly | 6 / 5 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L14 MLP |
| CKA(arrangement before, joint write) | 1.00 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.08 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 4.77 |
| arrangement before: shape (spectrum k:share) | line (50:1.00) |
| joint write: shape (spectrum k:share) | line (50:1.00) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 1.00 / 1.00 |
| best source position (CKA) | `b` (1.00) |

<details><summary>codes and components</summary>

**code 4a-471.0 (L15): 1 comps, tells apart 2/2, coverage 1.00, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 o c60 | H13 | b%2 in {0..1} (+) | 0.89 | 0.502 |

**code 4a-471.1 (L15): 1 comps, tells apart 2/2, coverage 1.00, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c40 | MLP | b%2 in {0..1} (+) | 0.95 | 0.501 |

</details>

</details>

</details>

<details><summary>`res%5`: 1 mechanisms, 2 components</summary>

<details><summary><b>4a-519</b> `res%5` @ `=` (add) — block code; 2 comps, L20; tells apart 5/5 classes (best member 4); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 5 /  / 4 (of 5) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.40 /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.88 |
| decoding acc. joint / best code / best member (chance) | 0.94 /  / 0.62 (0.20) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 49 / 31 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L20 attn |
| CKA(arrangement before, joint write) | 0.99 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.49 |
| share of the write inside the old arrangement's span | 0.26 |
| write energy / code energy before | 0.89 |
| arrangement before: shape (spectrum k:share) | circle period 5 (20:0.96) |
| joint write: shape (spectrum k:share) | circle period 5 (20:0.99) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.93 / 1.89 |

<details><summary>codes and components</summary>

**code 4a-519.0 (L20): 2 comps, tells apart 5/5, coverage 1.00, overlap 1.40 (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 down c9 | MLP | res%5 in {0, 2, 4} (-) | 0.90 | 0.603 |
| L20 down c15 | MLP | res%5 in {0..1, 3..4} (+) | 0.84 | 0.612 |

</details>

</details>

</details>

<details><summary>`cmp(a,b)`: 1 mechanisms, 2 components</summary>

<details><summary><b>4a-479</b> `cmp(a,b)` @ `=` (add) — 2 codes of the same shape, copy from `b`; 2 comps, L13 L15; tells apart 2/3 classes (best member 2); on sub: 4s-609 (member overlap 0.11)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.33 |
| classes told apart: joint / best code / best member | 2 / 2 / 2 (of 3) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 2.00 /  /  |
| mean CKA between its codes | 1.00 |
| purity of the joint write (per prompt) | 0.87 |
| decoding acc. joint / best code / best member (chance) | 0.65 / 0.66 / 0.66 (0.33) |
| kappa (1 orthogonal, >1 constructive) | 1.14 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L12 MLP |
| CKA(arrangement before, joint write) | 0.94 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.00 |
| write energy / code energy before | 0.01 |
| best source position (CKA) | `b` (0.73) |

<details><summary>codes and components</summary>

**code 4a-479.0 (L13): 1 comps, tells apart 2/3, coverage 0.33, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 o c379 | H7 | cmp(a,b) in {0} (-) | 0.93 | 0.010 |

**code 4a-479.1 (L15): 1 comps, tells apart 2/3, coverage 0.33, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c329 | MLP | cmp(a,b) in {0} (+) | 0.84 | 0.013 |

</details>

</details>

</details>

<details><summary>`b%5`: 1 mechanisms, 1 components</summary>

<details><summary><b>4a-474</b> `b%5` @ `=` (add) — single component; 1 comps, L15; tells apart 3/5 classes (best member 3); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.80 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 5) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.94 |
| decoding acc. joint / best code / best member (chance) | 0.61 /  / 0.61 (0.20) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 14 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 attn |
| CKA(arrangement before, joint write) | 0.64 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.36 |
| share of the write inside the old arrangement's span | 0.38 |
| write energy / code energy before | 1.13 |
| arrangement before: shape (spectrum k:share) | circle period 5 (20:0.70 40:0.30) |
| joint write: shape (spectrum k:share) | line (20:0.93 40:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.44 / 1.00 |

<details><summary>codes and components</summary>

**code 4a-474.0 (L15): 1 comps, tells apart 3/5, coverage 0.80, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c16 | MLP | b%5 in {1..4} (-) | 0.94 | 0.606 |

</details>

</details>

</details>

<details><summary>`res%2`: 1 mechanisms, 1 components</summary>

<details><summary><b>4a-514</b> `res%2` @ `=` (add) — single component; 1 comps, L21; tells apart 2/2 classes (best member 2); on sub: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 2) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.93 |
| decoding acc. joint / best code / best member (chance) | 1.00 /  / 1.00 (0.50) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 13 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 attn |
| CKA(arrangement before, joint write) | 1.00 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.46 |
| share of the write inside the old arrangement's span | 0.21 |
| write energy / code energy before | 0.39 |
| arrangement before: shape (spectrum k:share) | line (50:1.00) |
| joint write: shape (spectrum k:share) | line (50:1.00) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 1.00 / 1.00 |

<details><summary>codes and components</summary>

**code 4a-514.0 (L21): 1 comps, tells apart 2/2, coverage 1.00, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c1 | MLP | res%2 in {0..1} (+) | 0.93 | 0.500 |

</details>

</details>

</details>
