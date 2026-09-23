[← back to the report](../report_mechanisms.md)

# Every mechanism at `a`, add

<details><summary>`a%100`: 42 mechanisms, 1009 components</summary>

<details><summary><b>1a-10</b> `a%100` @ `a` (add) — single component; 1 comps, L0; tells apart 4/100 classes (best member 4); on sub: 1s-85 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.14 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.25 /  / 0.25 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | embed |
| CKA(arrangement before, joint write) | 0.31 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (1:0.10 2:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.45 2:0.13) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.11 / 1.00 |

<details><summary>codes and components</summary>

**code 1a-10.0 (L0): 1 comps, tells apart 4/100, coverage 0.14, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 o c5 | H2 | a%100 in {1..10, 12, 74, 77, 83} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>1a-9</b> `a%100` @ `a` (add) — block code; 2 comps, L0; tells apart 16/100 classes (best member 14); on sub: 1s-84 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.17 |
| classes told apart: joint / best code / best member | 16 /  / 14 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.85 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.65 /  / 0.17 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | embed |
| CKA(arrangement before, joint write) | 0.20 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (1:0.10 2:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.10 8:0.09 20:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.04 / 1.76 |

<details><summary>codes and components</summary>

**code 1a-9.0 (L0): 2 comps, tells apart 16/100, coverage 0.17, overlap 1.00 (random 1.00, p 0.88)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 o c7 | H0 | a%100 in {0, 99} (+) | 1.00 | 1.000 |
| L0 o c26 | H1 | a%100 in {2..9, 12, 34, 46, 64, 74, 83, 95} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>1a-31</b> `a%100` @ `a` (add) — tiling, 2 codes of the same shape; 149 comps, L0 L14; tells apart 93/100 classes (best member 9); on sub: 1s-86 (member overlap 0.99)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 93 / 96 / 9 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p | 4.86 / 6.77 / 0.00 |
| mean CKA between its codes | 0.75 |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 1.00 / 1.00 / 0.69 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.05 |
| consumers / read jointly | 217 / 186 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 attn |
| CKA(arrangement before, joint write) | 0.71 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.06 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 10.40 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (1:0.10 2:0.06) |
| joint write: shape (spectrum k:share) | irregular (2:0.14 1:0.14 3:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.67 / 19.93 |

<details><summary>codes and components</summary>

**code 1a-31.0 (L0): 96 comps, tells apart 96/100, coverage 0.99, overlap 2.92 (random 4.45, p 0.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 down c19 | MLP | a%100 in {1..10} (+) | 1.00 | 0.900 |
| L0 down c3 | MLP | a%100 in {1..16, 83, 86} (-) | 1.00 | 1.000 |
| L0 down c9 | MLP | a%100 in {1..5} (+) | 1.00 | 1.000 |
| L0 down c252 | MLP | a%100 in {1..5} (+) | 1.00 | 0.090 |
| L0 down c31 | MLP | a%100 in {1..8} (-) | 1.00 | 0.100 |
| L0 down c133 | MLP | a%100 in {10} (-) | 1.00 | 0.010 |
| L0 down c169 | MLP | a%100 in {11} (+) | 1.00 | 0.010 |
| L0 down c121 | MLP | a%100 in {12} (-) | 1.00 | 0.020 |
| L0 down c53 | MLP | a%100 in {13..16} (+) | 1.00 | 0.050 |
| L0 down c127 | MLP | a%100 in {16, 32, 48, 64, 96} (-) | 1.00 | 0.080 |
| L0 down c101 | MLP | a%100 in {17..18} (-) | 1.00 | 0.040 |
| L0 down c109 | MLP | a%100 in {18, 24, 36, 42, 45, 48, 54, 72} (+) | 1.00 | 0.100 |
| L0 down c59 | MLP | a%100 in {19..22} (+) | 1.00 | 0.130 |
| L0 down c112 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L0 down c182 | MLP | a%100 in {1} (-) | 1.00 | 0.030 |
| L0 down c309 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L0 down c948 | MLP | a%100 in {20} (-) | 1.00 | 0.010 |
| L0 down c539 | MLP | a%100 in {21} (+) | 1.00 | 0.010 |
| L0 down c148 | MLP | a%100 in {22..23} (-) | 1.00 | 0.040 |
| L0 down c313 | MLP | a%100 in {24..25} (-) | 1.00 | 0.030 |
| L0 down c287 | MLP | a%100 in {25..27} (-) | 1.00 | 0.040 |
| L0 down c30 | MLP | a%100 in {26..30} (+) | 1.00 | 0.090 |
| L0 down c171 | MLP | a%100 in {26} (-) | 1.00 | 0.010 |
| L0 down c697 | MLP | a%100 in {29} (-) | 1.00 | 0.010 |
| L0 down c352 | MLP | a%100 in {2} (+) | 1.00 | 0.010 |
| L0 down c164 | MLP | a%100 in {30..33} (+) | 1.00 | 0.070 |
| L0 down c220 | MLP | a%100 in {30} (-) | 1.00 | 0.020 |
| L0 down c237 | MLP | a%100 in {31} (+) | 1.00 | 0.010 |
| L0 down c163 | MLP | a%100 in {32, 64} (+) | 1.00 | 0.020 |
| L0 down c141 | MLP | a%100 in {32..34} (+) | 1.00 | 0.040 |
| L0 down c984 | MLP | a%100 in {33} (-) | 1.00 | 0.010 |
| L0 down c469 | MLP | a%100 in {34} (-) | 1.00 | 0.010 |
| L0 down c213 | MLP | a%100 in {35} (-) | 1.00 | 0.030 |
| L0 down c63 | MLP | a%100 in {36..39} (+) | 1.00 | 0.080 |
| L0 down c165 | MLP | a%100 in {37..38} (+) | 1.00 | 0.040 |
| L0 down c426 | MLP | a%100 in {38} (-) | 1.00 | 0.010 |
| L0 down c119 | MLP | a%100 in {39} (+) | 1.00 | 0.010 |
| L0 down c131 | MLP | a%100 in {3} (-) | 1.00 | 0.010 |
| L0 down c189 | MLP | a%100 in {40..42} (+) | 1.00 | 0.030 |
| L0 down c317 | MLP | a%100 in {40} (-) | 1.00 | 0.020 |
| L0 down c170 | MLP | a%100 in {41..45} (+) | 1.00 | 0.080 |
| L0 down c572 | MLP | a%100 in {42} (-) | 1.00 | 0.010 |
| L0 down c679 | MLP | a%100 in {44} (+) | 1.00 | 0.010 |
| L0 down c60 | MLP | a%100 in {45..48} (-) | 1.00 | 0.050 |
| L0 down c110 | MLP | a%100 in {46..49} (+) | 1.00 | 0.060 |
| L0 down c387 | MLP | a%100 in {46} (+) | 1.00 | 0.010 |
| L0 down c406 | MLP | a%100 in {46} (+) | 1.00 | 0.010 |
| L0 down c143 | MLP | a%100 in {49..54} (+) | 1.00 | 0.070 |
| L0 down c348 | MLP | a%100 in {4} (-) | 1.00 | 0.010 |
| L0 down c43 | MLP | a%100 in {5..9} (+) | 1.00 | 0.080 |
| L0 down c245 | MLP | a%100 in {50} (-) | 1.00 | 0.010 |
| L0 down c135 | MLP | a%100 in {51..52} (+) | 1.00 | 0.040 |
| L0 down c321 | MLP | a%100 in {52} (+) | 1.00 | 0.020 |
| L0 down c48 | MLP | a%100 in {53..57} (-) | 1.00 | 0.080 |
| L0 down c184 | MLP | a%100 in {55} (+) | 1.00 | 0.010 |
| L0 down c167 | MLP | a%100 in {56} (-) | 1.00 | 0.010 |
| L0 down c69 | MLP | a%100 in {57..59} (-) | 1.00 | 0.070 |
| L0 down c207 | MLP | a%100 in {5} (+) | 1.00 | 0.010 |
| L0 down c21 | MLP | a%100 in {6, 10..14, 17, 21..27, 33, 35, 66, 70, 74..75, 87, 94, 98} (+) | 1.00 | 1.000 |
| L0 down c54 | MLP | a%100 in {60, 80} (-) | 1.00 | 0.030 |
| L0 down c85 | MLP | a%100 in {60..62} (+) | 1.00 | 0.040 |
| L0 down c90 | MLP | a%100 in {60..65} (-) | 1.00 | 0.100 |
| L0 down c67 | MLP | a%100 in {62} (+) | 1.00 | 0.010 |
| L0 down c22 | MLP | a%100 in {63..76} (+) | 1.00 | 0.190 |
| L0 down c105 | MLP | a%100 in {65..69} (-) | 1.00 | 0.060 |
| L0 down c467 | MLP | a%100 in {66, 77, 88} (+) | 1.00 | 0.030 |
| L0 down c349 | MLP | a%100 in {68} (-) | 1.00 | 0.010 |
| L0 down c795 | MLP | a%100 in {68} (-) | 1.00 | 0.010 |
| L0 down c253 | MLP | a%100 in {69} (-) | 1.00 | 0.020 |
| L0 down c266 | MLP | a%100 in {6} (+) | 1.00 | 0.010 |
| L0 down c108 | MLP | a%100 in {70..72} (+) | 1.00 | 0.050 |
| L0 down c94 | MLP | a%100 in {70..75} (-) | 1.00 | 0.080 |
| L0 down c515 | MLP | a%100 in {70} (+) | 1.00 | 0.010 |
| L0 down c75 | MLP | a%100 in {74, 76..79} (-) | 1.00 | 0.070 |
| L0 down c244 | MLP | a%100 in {74..76} (+) | 1.00 | 0.030 |
| L0 down c17 | MLP | a%100 in {74} (+) | 1.00 | 0.010 |
| L0 down c72 | MLP | a%100 in {75..80} (+) | 1.00 | 0.070 |
| L0 down c88 | MLP | a%100 in {79..83} (-) | 1.00 | 0.090 |
| L0 down c376 | MLP | a%100 in {7} (+) | 1.00 | 0.020 |
| L0 down c77 | MLP | a%100 in {80, 82..85} (-) | 1.00 | 0.070 |
| L0 down c946 | MLP | a%100 in {81} (-) | 1.00 | 0.010 |
| L0 down c687 | MLP | a%100 in {83} (+) | 1.00 | 0.010 |
| L0 down c97 | MLP | a%100 in {85..89} (-) | 1.00 | 0.060 |
| L0 down c61 | MLP | a%100 in {86..87} (+) | 1.00 | 0.020 |
| L0 down c738 | MLP | a%100 in {86} (-) | 1.00 | 0.010 |
| L0 down c318 | MLP | a%100 in {88} (+) | 1.00 | 0.020 |
| L0 down c330 | MLP | a%100 in {89} (+) | 1.00 | 0.010 |
| L0 down c351 | MLP | a%100 in {8} (-) | 1.00 | 0.010 |
| L0 down c100 | MLP | a%100 in {90} (-) | 1.00 | 0.040 |
| L0 down c49 | MLP | a%100 in {91, 93} (+) | 1.00 | 0.020 |
| L0 down c1020 | MLP | a%100 in {91} (-) | 1.00 | 0.010 |
| L0 down c87 | MLP | a%100 in {92..95} (+) | 1.00 | 0.080 |
| L0 down c73 | MLP | a%100 in {95..98} (-) | 1.00 | 0.070 |
| L0 down c58 | MLP | a%100 in {98..99} (-) | 1.00 | 0.060 |
| L0 down c586 | MLP | a%100 in {99} (+) | 1.00 | 0.010 |
| L0 down c273 | MLP | a%100 in {9} (+) | 1.00 | 0.020 |

**code 1a-31.1 (L14): 53 comps, tells apart 91/100, coverage 1.00, overlap 1.97 (random 2.65, p 0.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L14 down c400 | MLP | a%100 in {0, 10} (-) | 1.00 | 0.020 |
| L14 down c588 | MLP | a%100 in {0, 95..99} (-) | 1.00 | 0.060 |
| L14 down c170 | MLP | a%100 in {1..3} (-) | 1.00 | 0.040 |
| L14 down c2 | MLP | a%100 in {1..5} (+) | 1.00 | 0.050 |
| L14 down c637 | MLP | a%100 in {10} (-) | 1.00 | 0.010 |
| L14 down c542 | MLP | a%100 in {11..12} (-) | 1.00 | 0.020 |
| L14 down c1002 | MLP | a%100 in {12} (+) | 1.00 | 0.010 |
| L14 down c131 | MLP | a%100 in {13..14} (+) | 1.00 | 0.020 |
| L14 down c615 | MLP | a%100 in {14..15} (-) | 1.00 | 0.030 |
| L14 down c148 | MLP | a%100 in {15, 75, 99} (-) | 1.00 | 0.030 |
| L14 down c96 | MLP | a%100 in {16..17} (+) | 1.00 | 0.034 |
| L14 down c42 | MLP | a%100 in {18, 22, 38, 42, 58, 62, 78, 82..83, 98} (+) | 1.00 | 0.151 |
| L14 down c150 | MLP | a%100 in {1} (+) | 1.00 | 0.020 |
| L14 down c55 | MLP | a%100 in {21, 31, 41, 51, 61, 71, 81, 91} (+) | 1.00 | 0.080 |
| L14 down c393 | MLP | a%100 in {21, 41..42} (-) | 0.66 | 0.020 |
| L14 down c121 | MLP | a%100 in {23, 43, 46, 53, 73} (-) | 1.00 | 0.070 |
| L14 down c248 | MLP | a%100 in {23..24} (-) | 1.00 | 0.020 |
| L14 down c215 | MLP | a%100 in {25, 50..51} (-) | 1.00 | 0.050 |
| L14 down c48 | MLP | a%100 in {26..30} (-) | 1.00 | 0.070 |
| L14 down c137 | MLP | a%100 in {27, 84..87} (+) | 1.00 | 0.050 |
| L14 down c45 | MLP | a%100 in {27..33} (+) | 1.00 | 0.110 |
| L14 down c151 | MLP | a%100 in {3..9} (+) | 1.00 | 0.090 |
| L14 down c154 | MLP | a%100 in {30, 40, 60, 70, 80, 90} (+) | 1.00 | 0.060 |
| L14 down c476 | MLP | a%100 in {32} (+) | 1.00 | 0.010 |
| L14 down c547 | MLP | a%100 in {34..36} (-) | 1.00 | 0.040 |
| L14 down c398 | MLP | a%100 in {35, 45} (-) | 1.00 | 0.020 |
| L14 down c41 | MLP | a%100 in {35..43} (-) | 1.00 | 0.130 |
| L14 down c176 | MLP | a%100 in {36, 56..58} (+) | 1.00 | 0.050 |
| L14 down c165 | MLP | a%100 in {37, 74} (-) | 1.00 | 0.020 |
| L14 down c310 | MLP | a%100 in {3} (+) | 1.00 | 0.010 |
| L14 down c129 | MLP | a%100 in {44, 55} (+) | 0.99 | 0.031 |
| L14 down c238 | MLP | a%100 in {45..46} (-) | 1.00 | 0.020 |
| L14 down c65 | MLP | a%100 in {45..52} (-) | 1.00 | 0.120 |
| L14 down c105 | MLP | a%100 in {49} (+) | 1.00 | 0.010 |
| L14 down c671 | MLP | a%100 in {52..54} (-) | 1.00 | 0.030 |
| L14 down c441 | MLP | a%100 in {52} (+) | 1.00 | 0.010 |
| L14 down c551 | MLP | a%100 in {52} (+) | 1.00 | 0.010 |
| L14 down c70 | MLP | a%100 in {53..63} (+) | 1.00 | 0.131 |
| L14 down c666 | MLP | a%100 in {59, 61} (+) | 1.00 | 0.020 |
| L14 down c787 | MLP | a%100 in {5} (+) | 0.98 | 0.011 |
| L14 down c1021 | MLP | a%100 in {63..64} (-) | 1.00 | 0.020 |
| L14 down c701 | MLP | a%100 in {64..66} (+) | 1.00 | 0.050 |
| L14 down c833 | MLP | a%100 in {66..69} (+) | 1.00 | 0.040 |
| L14 down c213 | MLP | a%100 in {69..71} (+) | 1.00 | 0.050 |
| L14 down c201 | MLP | a%100 in {7..8, 17..18} (+) | 1.00 | 0.040 |
| L14 down c320 | MLP | a%100 in {72..75} (-) | 1.00 | 0.050 |
| L14 down c448 | MLP | a%100 in {75..76} (-) | 1.00 | 0.020 |
| L14 down c182 | MLP | a%100 in {76..79} (+) | 1.00 | 0.070 |
| L14 down c644 | MLP | a%100 in {81..84} (+) | 1.00 | 0.040 |
| L14 down c104 | MLP | a%100 in {88..89} (-) | 1.00 | 0.080 |
| L14 down c34 | MLP | a%100 in {88..98} (-) | 1.00 | 0.160 |
| L14 down c290 | MLP | a%100 in {89..90} (+) | 1.00 | 0.020 |
| L14 down c142 | MLP | a%100 in {9, 19..20, 39..40, 79..80} (+) | 1.00 | 0.070 |

</details>

</details>

<details><summary><b>1a-11</b> `a%100` @ `a` (add) — block code; 4 comps, L1; tells apart 6/100 classes (best member 2); on sub: 1s-87 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.50 |
| classes told apart: joint / best code / best member | 6 /  / 2 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.30 / 1.05 / 0.93 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.96 /  / 0.65 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.94 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.53 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.20 2:0.10 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.51 6:0.09 3:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.19 / 1.36 |

<details><summary>codes and components</summary>

**code 1a-11.0 (L1): 4 comps, tells apart 6/100, coverage 0.50, overlap 1.30 (random 1.01, p 0.94)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c231 | H26 | a%100 in {0..2, 22, 33..36, 76..78, 80..84, 86, 88..93, 95..99} (+) | 1.00 | 1.000 |
| L1 o c70 | H26 | a%100 in {1..7, 13..20, 33..39, 73..74, 80, 85, 87, 89..90, 92, 94..98} (-) | 1.00 | 0.780 |
| L1 o c330 | H26 | a%100 in {99} (-) | 1.00 | 0.010 |
| L1 o c201 | H26 | a%100 in {9} (+) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>1a-12</b> `a%100` @ `a` (add) — block code; 28 comps, L1; tells apart 61/100 classes (best member 8); on sub: 1s-88 (member overlap 0.96)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.83 |
| classes told apart: joint / best code / best member | 61 /  / 8 (of 100) |
| members whose removal merges classes | 0.71 |
| support overlap (1 = tiling) / random sets / p | 1.84 / 1.79 / 0.61 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 1.00 /  / 0.71 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.99 |
| consumers / read jointly | 10 / 10 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L1 attn |
| CKA(arrangement before, joint write) | 0.72 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.12 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.08 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.20 2:0.10 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.20 2:0.12 3:0.11) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.36 / 8.22 |

<details><summary>codes and components</summary>

**code 1a-12.0 (L1): 28 comps, tells apart 61/100, coverage 0.83, overlap 1.84 (random 1.82, p 0.56)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 down c481 | MLP | a%100 in {0, 99} (+) | 1.00 | 0.110 |
| L1 down c13 | MLP | a%100 in {0..12, 15, 30, 51, 62, 65, 69, 73..74} (-) | 1.00 | 1.000 |
| L1 down c39 | MLP | a%100 in {0..5, 10, 15, 17..21, 24..25, 30, 56, 58, 61..63, 67, 73, 76, 79, 82..83} (+) | 1.00 | 1.000 |
| L1 down c617 | MLP | a%100 in {1..3, 61} (-) | 1.00 | 0.110 |
| L1 down c166 | MLP | a%100 in {1..6} (-) | 1.00 | 0.100 |
| L1 down c140 | MLP | a%100 in {15, 25, 35, 55, 64..65, 75, 85..86, 95..96} (-) | 1.00 | 0.180 |
| L1 down c356 | MLP | a%100 in {19, 29, 39, 49, 59, 69, 79, 89} (+) | 1.00 | 0.090 |
| L1 down c163 | MLP | a%100 in {1} (+) | 1.00 | 0.040 |
| L1 down c264 | MLP | a%100 in {23..31} (+) | 1.00 | 0.090 |
| L1 down c964 | MLP | a%100 in {33..39} (-) | 1.00 | 0.080 |
| L1 down c80 | MLP | a%100 in {42, 52} (-) | 1.00 | 0.020 |
| L1 down c365 | MLP | a%100 in {47..53} (+) | 1.00 | 0.090 |
| L1 down c546 | MLP | a%100 in {48} (+) | 1.00 | 0.010 |
| L1 down c468 | MLP | a%100 in {55} (+) | 1.00 | 0.010 |
| L1 down c631 | MLP | a%100 in {61..62} (-) | 1.00 | 0.040 |
| L1 down c212 | MLP | a%100 in {65..77} (+) | 1.00 | 0.190 |
| L1 down c331 | MLP | a%100 in {70} (-) | 1.00 | 0.010 |
| L1 down c885 | MLP | a%100 in {76..78} (-) | 1.00 | 0.040 |
| L1 down c37 | MLP | a%100 in {8..10} (-) | 1.00 | 0.030 |
| L1 down c251 | MLP | a%100 in {8..9} (+) | 1.00 | 0.020 |
| L1 down c118 | MLP | a%100 in {82..83} (-) | 1.00 | 0.020 |
| L1 down c917 | MLP | a%100 in {82} (-) | 1.00 | 0.010 |
| L1 down c210 | MLP | a%100 in {87..88} (+) | 1.00 | 0.030 |
| L1 down c335 | MLP | a%100 in {90..91, 93..94} (-) | 1.00 | 0.150 |
| L1 down c454 | MLP | a%100 in {91..92} (-) | 1.00 | 0.030 |
| L1 down c337 | MLP | a%100 in {91..99} (+) | 1.00 | 0.152 |
| L1 down c128 | MLP | a%100 in {91} (+) | 1.00 | 0.010 |
| L1 down c663 | MLP | a%100 in {99} (+) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>1a-13</b> `a%100` @ `a` (add) — block code; 2 comps, L2 L3; tells apart 4/100 classes (best member 5); on sub: 1s-89 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.18 |
| classes told apart: joint / best code / best member | 4 /  / 5 (of 100) |
| members whose removal merges classes | 0.50 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.85 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.67 /  / 0.65 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L1 MLP |
| CKA(arrangement before, joint write) | 0.18 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.21 2:0.10 3:0.06) |
| joint write: shape (spectrum k:share) | line (2:0.13 10:0.12 1:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.03 / 1.03 |

<details><summary>codes and components</summary>

**code 1a-13.0 (L2 L3): 2 comps, tells apart 4/100, coverage 0.18, overlap 1.00 (random 1.00, p 0.85)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 o c89 | H23 | a%100 in {0..2, 41, 51..52, 59, 66, 68, 73, 78..79, 91, 96..98} (+) | 1.00 | 1.000 |
| L3 o c243 | H15 | a%100 in {92..93} (-) | 0.87 | 0.023 |

</details>

</details>

<details><summary><b>1a-14</b> `a%100` @ `a` (add) — tiling, 10 codes of the same shape; 300 comps, L2 L3 L4 L5 L7 L13 L18 L19 L21 L24; tells apart 93/100 classes (best member 8); on sub: 1s-90 (member overlap 0.98)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 93 / 83 / 8 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p | 14.29 / 13.51 / 0.83 |
| mean CKA between its codes | 0.78 |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 1.00 / 1.00 / 0.68 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.12 |
| consumers / read jointly | 751 / 608 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 attn |
| CKA(arrangement before, joint write) | 0.86 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.13 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 20.24 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.21 2:0.10 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.34 2:0.14 3:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.54 / 7.10 |

<details><summary>codes and components</summary>

**code 1a-14.0 (L2): 43 comps, tells apart 66/100, coverage 0.78, overlap 1.68 (random 2.31, p 0.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 down c44 | MLP | a%100 in {0, 95..99} (+) | 1.00 | 0.110 |
| L2 down c147 | MLP | a%100 in {0} (+) | 1.00 | 0.120 |
| L2 down c79 | MLP | a%100 in {1..4} (-) | 1.00 | 0.140 |
| L2 down c254 | MLP | a%100 in {10, 20} (+) | 1.00 | 0.020 |
| L2 down c385 | MLP | a%100 in {10..11} (-) | 1.00 | 0.020 |
| L2 down c102 | MLP | a%100 in {12, 18, 24, 36, 45, 60, 72} (+) | 1.00 | 0.110 |
| L2 down c111 | MLP | a%100 in {12..14} (-) | 1.00 | 0.100 |
| L2 down c61 | MLP | a%100 in {14, 21, 28, 42} (-) | 1.00 | 0.040 |
| L2 down c351 | MLP | a%100 in {15..17} (+) | 1.00 | 0.030 |
| L2 down c90 | MLP | a%100 in {15..28} (+) | 1.00 | 0.220 |
| L2 down c643 | MLP | a%100 in {15} (-) | 1.00 | 0.010 |
| L2 down c313 | MLP | a%100 in {16} (+) | 1.00 | 0.060 |
| L2 down c474 | MLP | a%100 in {17} (+) | 1.00 | 0.010 |
| L2 down c162 | MLP | a%100 in {18..19} (-) | 1.00 | 0.060 |
| L2 down c369 | MLP | a%100 in {18} (+) | 1.00 | 0.020 |
| L2 down c9 | MLP | a%100 in {1} (+) | 1.00 | 0.100 |
| L2 down c967 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L2 down c612 | MLP | a%100 in {22} (-) | 1.00 | 0.010 |
| L2 down c118 | MLP | a%100 in {24, 48} (-) | 1.00 | 0.190 |
| L2 down c149 | MLP | a%100 in {24} (+) | 1.00 | 0.030 |
| L2 down c428 | MLP | a%100 in {25..30} (-) | 1.00 | 0.070 |
| L2 down c544 | MLP | a%100 in {2} (-) | 1.00 | 0.010 |
| L2 down c639 | MLP | a%100 in {3..10} (+) | 1.00 | 0.140 |
| L2 down c777 | MLP | a%100 in {31..32} (+) | 1.00 | 0.040 |
| L2 down c380 | MLP | a%100 in {33, 66} (-) | 1.00 | 0.020 |
| L2 down c275 | MLP | a%100 in {3} (+) | 1.00 | 0.030 |
| L2 down c934 | MLP | a%100 in {42} (+) | 1.00 | 0.010 |
| L2 down c213 | MLP | a%100 in {44..45, 56} (-) | 1.00 | 0.030 |
| L2 down c363 | MLP | a%100 in {51..52} (-) | 1.00 | 0.030 |
| L2 down c282 | MLP | a%100 in {5} (-) | 1.00 | 0.010 |
| L2 down c617 | MLP | a%100 in {60..62} (-) | 1.00 | 0.060 |
| L2 down c6 | MLP | a%100 in {62..77} (-) | 1.00 | 0.180 |
| L2 down c39 | MLP | a%100 in {65} (+) | 1.00 | 0.020 |
| L2 down c543 | MLP | a%100 in {74} (+) | 1.00 | 0.010 |
| L2 down c58 | MLP | a%100 in {76..88} (+) | 1.00 | 0.220 |
| L2 down c199 | MLP | a%100 in {7} (+) | 1.00 | 0.030 |
| L2 down c140 | MLP | a%100 in {83} (-) | 1.00 | 0.010 |
| L2 down c74 | MLP | a%100 in {88} (-) | 1.00 | 0.010 |
| L2 down c247 | MLP | a%100 in {9..12, 99} (-) | 1.00 | 0.050 |
| L2 down c448 | MLP | a%100 in {90} (-) | 1.00 | 0.070 |
| L2 down c717 | MLP | a%100 in {92} (+) | 1.00 | 0.010 |
| L2 down c1 | MLP | a%100 in {99} (-) | 1.00 | 0.990 |
| L2 down c864 | MLP | a%100 in {99} (-) | 1.00 | 0.040 |

**code 1a-14.1 (L3): 29 comps, tells apart 58/100, coverage 0.94, overlap 1.76 (random 1.84, p 0.32)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 down c0 | MLP | a%100 in {1, 11..12, 14..16, 18..23, 25..28, 30, 32, 34, 74, 81, 86, 91..99} (-) | 1.00 | 1.000 |
| L3 down c76 | MLP | a%100 in {1..2} (+) | 1.00 | 0.030 |
| L3 down c325 | MLP | a%100 in {11..12} (+) | 1.00 | 0.020 |
| L3 down c153 | MLP | a%100 in {12, 24, 48, 72, 96} (+) | 1.00 | 0.060 |
| L3 down c17 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L3 down c257 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L3 down c44 | MLP | a%100 in {2..17} (-) | 1.00 | 0.230 |
| L3 down c177 | MLP | a%100 in {23..24} (+) | 1.00 | 0.100 |
| L3 down c271 | MLP | a%100 in {24} (+) | 1.00 | 0.010 |
| L3 down c176 | MLP | a%100 in {28..31} (-) | 1.00 | 0.060 |
| L3 down c61 | MLP | a%100 in {30..37} (+) | 1.00 | 0.111 |
| L3 down c466 | MLP | a%100 in {33, 44, 55, 66, 77, 88} (-) | 1.00 | 0.070 |
| L3 down c79 | MLP | a%100 in {38..45} (-) | 1.00 | 0.100 |
| L3 down c156 | MLP | a%100 in {41..47} (+) | 1.00 | 0.230 |
| L3 down c175 | MLP | a%100 in {42} (-) | 1.00 | 0.020 |
| L3 down c665 | MLP | a%100 in {46, 76, 86} (-) | 1.00 | 0.080 |
| L3 down c41 | MLP | a%100 in {48..52} (+) | 1.00 | 0.060 |
| L3 down c113 | MLP | a%100 in {50..59} (-) | 1.00 | 0.170 |
| L3 down c210 | MLP | a%100 in {57..63} (-) | 1.00 | 0.090 |
| L3 down c360 | MLP | a%100 in {6, 46} (-) | 1.00 | 0.020 |
| L3 down c93 | MLP | a%100 in {62..76} (-) | 1.00 | 0.220 |
| L3 down c361 | MLP | a%100 in {64, 83} (-) | 1.00 | 0.020 |
| L3 down c510 | MLP | a%100 in {65} (+) | 1.00 | 0.010 |
| L3 down c77 | MLP | a%100 in {73..81} (+) | 1.00 | 0.140 |
| L3 down c225 | MLP | a%100 in {8..9} (-) | 1.00 | 0.020 |
| L3 down c577 | MLP | a%100 in {83} (+) | 1.00 | 0.010 |
| L3 down c145 | MLP | a%100 in {90..95} (+) | 1.00 | 0.070 |
| L3 down c39 | MLP | a%100 in {92..93} (-) | 1.00 | 0.020 |
| L3 down c122 | MLP | a%100 in {95..99} (-) | 1.00 | 0.111 |

**code 1a-14.2 (L4): 32 comps, tells apart 48/100, coverage 0.95, overlap 1.76 (random 1.97, p 0.25)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 down c204 | MLP | a%100 in {0, 96..99} (+) | 1.00 | 0.070 |
| L4 down c1 | MLP | a%100 in {0..2} (+) | 1.00 | 1.000 |
| L4 down c32 | MLP | a%100 in {0..2} (-) | 1.00 | 1.000 |
| L4 down c127 | MLP | a%100 in {1..5} (-) | 1.00 | 0.090 |
| L4 down c103 | MLP | a%100 in {12..17} (-) | 1.00 | 0.080 |
| L4 down c62 | MLP | a%100 in {13..31} (-) | 1.00 | 0.230 |
| L4 down c237 | MLP | a%100 in {17..21} (-) | 1.00 | 0.060 |
| L4 down c97 | MLP | a%100 in {18, 24, 36, 45, 48, 54, 60, 72, 90} (+) | 1.00 | 0.100 |
| L4 down c561 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L4 down c487 | MLP | a%100 in {1} (-) | 1.00 | 0.020 |
| L4 down c317 | MLP | a%100 in {24} (+) | 1.00 | 0.010 |
| L4 down c132 | MLP | a%100 in {3..17} (-) | 1.00 | 0.230 |
| L4 down c181 | MLP | a%100 in {30..43} (-) | 1.00 | 0.140 |
| L4 down c735 | MLP | a%100 in {31} (+) | 1.00 | 0.010 |
| L4 down c408 | MLP | a%100 in {42, 83} (-) | 1.00 | 0.020 |
| L4 down c80 | MLP | a%100 in {44..53} (+) | 1.00 | 0.170 |
| L4 down c361 | MLP | a%100 in {48, 72} (-) | 1.00 | 0.040 |
| L4 down c57 | MLP | a%100 in {52} (+) | 1.00 | 0.050 |
| L4 down c530 | MLP | a%100 in {53..59} (+) | 1.00 | 0.070 |
| L4 down c211 | MLP | a%100 in {55..69} (+) | 1.00 | 0.180 |
| L4 down c110 | MLP | a%100 in {61} (+) | 1.00 | 0.010 |
| L4 down c109 | MLP | a%100 in {64} (-) | 1.00 | 0.010 |
| L4 down c622 | MLP | a%100 in {74} (+) | 1.00 | 0.040 |
| L4 down c864 | MLP | a%100 in {74} (-) | 1.00 | 0.010 |
| L4 down c12 | MLP | a%100 in {76, 91..93} (+) | 1.00 | 0.040 |
| L4 down c625 | MLP | a%100 in {77, 88, 99} (-) | 1.00 | 0.050 |
| L4 down c390 | MLP | a%100 in {79..89} (-) | 1.00 | 0.140 |
| L4 down c768 | MLP | a%100 in {83} (+) | 1.00 | 0.010 |
| L4 down c158 | MLP | a%100 in {87..99} (+) | 1.00 | 0.170 |
| L4 down c140 | MLP | a%100 in {92..93} (+) | 1.00 | 0.020 |
| L4 down c145 | MLP | a%100 in {93, 95, 99} (+) | 0.97 | 0.032 |
| L4 down c590 | MLP | a%100 in {99} (+) | 1.00 | 0.010 |

**code 1a-14.3 (L5): 14 comps, tells apart 20/100, coverage 0.56, overlap 1.41 (random 1.34, p 0.62)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 down c45 | MLP | a%100 in {1..2} (+) | 1.00 | 0.040 |
| L5 down c43 | MLP | a%100 in {1..2} (-) | 1.00 | 0.020 |
| L5 down c7 | MLP | a%100 in {1..4} (+) | 1.00 | 1.000 |
| L5 down c16 | MLP | a%100 in {15..49} (+) | 1.00 | 0.350 |
| L5 down c838 | MLP | a%100 in {2..4} (+) | 1.00 | 0.030 |
| L5 down c332 | MLP | a%100 in {24, 36, 48, 72} (-) | 1.00 | 0.040 |
| L5 down c27 | MLP | a%100 in {3..21} (+) | 1.00 | 0.300 |
| L5 down c14 | MLP | a%100 in {3} (+) | 1.00 | 0.010 |
| L5 down c83 | MLP | a%100 in {42, 52} (-) | 1.00 | 0.020 |
| L5 down c568 | MLP | a%100 in {52} (+) | 1.00 | 0.010 |
| L5 down c198 | MLP | a%100 in {61, 91} (+) | 1.00 | 0.020 |
| L5 down c139 | MLP | a%100 in {74} (+) | 1.00 | 0.020 |
| L5 down c177 | MLP | a%100 in {82..83} (-) | 1.00 | 0.020 |
| L5 down c207 | MLP | a%100 in {83} (+) | 1.00 | 0.010 |

**code 1a-14.4 (L7): 17 comps, tells apart 20/100, coverage 0.55, overlap 1.58 (random 1.46, p 0.73)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L7 down c0 | MLP | a%100 in {0..3, 10, 26, 28, 52, 61, 99} (+) | 1.00 | 1.000 |
| L7 down c2 | MLP | a%100 in {1, 17, 29, 65, 77, 83, 88} (-) | 1.00 | 1.000 |
| L7 down c15 | MLP | a%100 in {1..7} (-) | 1.00 | 0.120 |
| L7 down c150 | MLP | a%100 in {19, 22..23} (-) | 0.98 | 0.030 |
| L7 down c112 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L7 down c578 | MLP | a%100 in {24} (-) | 1.00 | 0.010 |
| L7 down c698 | MLP | a%100 in {2} (-) | 1.00 | 0.010 |
| L7 down c5 | MLP | a%100 in {3..23, 25..27, 30} (+) | 1.00 | 0.280 |
| L7 down c287 | MLP | a%100 in {5..7} (-) | 1.00 | 0.030 |
| L7 down c199 | MLP | a%100 in {52} (-) | 1.00 | 0.010 |
| L7 down c507 | MLP | a%100 in {52} (-) | 1.00 | 0.010 |
| L7 down c13 | MLP | a%100 in {53, 56..59, 62} (-) | 1.00 | 0.060 |
| L7 down c455 | MLP | a%100 in {7..9} (-) | 1.00 | 0.030 |
| L7 down c450 | MLP | a%100 in {83} (-) | 1.00 | 0.010 |
| L7 down c141 | MLP | a%100 in {87..99} (+) | 1.00 | 0.130 |
| L7 down c38 | MLP | a%100 in {88, 99} (-) | 1.00 | 0.020 |
| L7 down c478 | MLP | a%100 in {92..93} (-) | 0.97 | 0.021 |

**code 1a-14.5 (L13): 33 comps, tells apart 62/100, coverage 0.96, overlap 1.81 (random 1.96, p 0.27)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 down c0 | MLP | a%100 in {1, 3..10} (+) | 1.00 | 1.000 |
| L13 down c2 | MLP | a%100 in {1..3, 5} (+) | 1.00 | 1.000 |
| L13 down c76 | MLP | a%100 in {1..3} (-) | 1.00 | 0.042 |
| L13 down c7 | MLP | a%100 in {11..31} (-) | 1.00 | 0.210 |
| L13 down c774 | MLP | a%100 in {14..17} (+) | 1.00 | 0.040 |
| L13 down c59 | MLP | a%100 in {18, 24, 36, 42, 45, 48, 60, 72, 90} (+) | 0.95 | 0.097 |
| L13 down c332 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L13 down c291 | MLP | a%100 in {2..9} (+) | 1.00 | 0.080 |
| L13 down c195 | MLP | a%100 in {20..22} (-) | 1.00 | 0.080 |
| L13 down c526 | MLP | a%100 in {23..24} (-) | 1.00 | 0.020 |
| L13 down c54 | MLP | a%100 in {31..32, 60..63} (-) | 1.00 | 0.080 |
| L13 down c46 | MLP | a%100 in {32..34, 52..54, 74} (-) | 1.00 | 0.070 |
| L13 down c187 | MLP | a%100 in {33, 66} (-) | 1.00 | 0.020 |
| L13 down c23 | MLP | a%100 in {35..42} (-) | 1.00 | 0.150 |
| L13 down c795 | MLP | a%100 in {36, 56, 76, 96} (-) | 1.00 | 0.040 |
| L13 down c24 | MLP | a%100 in {4..12} (-) | 1.00 | 0.140 |
| L13 down c361 | MLP | a%100 in {44, 55} (+) | 1.00 | 0.020 |
| L13 down c21 | MLP | a%100 in {45..51} (+) | 1.00 | 0.150 |
| L13 down c560 | MLP | a%100 in {49} (-) | 1.00 | 0.010 |
| L13 down c429 | MLP | a%100 in {51} (+) | 1.00 | 0.010 |
| L13 down c545 | MLP | a%100 in {52} (+) | 0.94 | 0.015 |
| L13 down c20 | MLP | a%100 in {53..63} (-) | 1.00 | 0.170 |
| L13 down c208 | MLP | a%100 in {59} (+) | 1.00 | 0.010 |
| L13 down c133 | MLP | a%100 in {63..67} (-) | 1.00 | 0.060 |
| L13 down c27 | MLP | a%100 in {65..74} (+) | 1.00 | 0.139 |
| L13 down c96 | MLP | a%100 in {68..69, 86..89} (-) | 1.00 | 0.060 |
| L13 down c999 | MLP | a%100 in {73..77} (+) | 1.00 | 0.051 |
| L13 down c433 | MLP | a%100 in {75} (+) | 1.00 | 0.010 |
| L13 down c16 | MLP | a%100 in {77, 94..99} (-) | 1.00 | 0.130 |
| L13 down c957 | MLP | a%100 in {78..79} (-) | 1.00 | 0.020 |
| L13 down c185 | MLP | a%100 in {80, 90..91} (-) | 1.00 | 0.030 |
| L13 down c108 | MLP | a%100 in {83..85} (-) | 1.00 | 0.030 |
| L13 down c48 | MLP | a%100 in {90..97} (+) | 1.00 | 0.090 |

**code 1a-14.6 (L18): 28 comps, tells apart 59/100, coverage 0.75, overlap 1.97 (random 1.77, p 0.78)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 down c243 | MLP | a%100 in {0, 99} (-) | 1.00 | 0.020 |
| L18 down c3 | MLP | a%100 in {0..10, 12, 15, 25, 30, 50, 90, 99} (-) | 1.00 | 1.000 |
| L18 down c1 | MLP | a%100 in {1..10} (-) | 1.00 | 1.000 |
| L18 down c58 | MLP | a%100 in {1..2} (-) | 1.00 | 0.040 |
| L18 down c440 | MLP | a%100 in {11} (+) | 1.00 | 0.010 |
| L18 down c43 | MLP | a%100 in {13..31} (-) | 1.00 | 0.190 |
| L18 down c8 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L18 down c31 | MLP | a%100 in {21, 31, 41, 51, 61, 71, 81, 91} (-) | 1.00 | 0.080 |
| L18 down c188 | MLP | a%100 in {24, 74} (+) | 1.00 | 0.070 |
| L18 down c373 | MLP | a%100 in {2} (+) | 1.00 | 0.010 |
| L18 down c417 | MLP | a%100 in {30..31} (-) | 1.00 | 0.030 |
| L18 down c293 | MLP | a%100 in {32, 40, 64} (-) | 1.00 | 0.040 |
| L18 down c281 | MLP | a%100 in {33} (+) | 1.00 | 0.040 |
| L18 down c345 | MLP | a%100 in {35, 65} (-) | 1.00 | 0.090 |
| L18 down c81 | MLP | a%100 in {38} (+) | 1.00 | 0.010 |
| L18 down c1011 | MLP | a%100 in {38} (+) | 1.00 | 0.010 |
| L18 down c340 | MLP | a%100 in {39..42} (+) | 1.00 | 0.040 |
| L18 down c28 | MLP | a%100 in {5, 10} (+) | 1.00 | 0.020 |
| L18 down c13 | MLP | a%100 in {5..15, 20, 25, 30} (-) | 1.00 | 0.560 |
| L18 down c419 | MLP | a%100 in {51..53} (-) | 1.00 | 0.060 |
| L18 down c274 | MLP | a%100 in {52, 72} (-) | 1.00 | 0.030 |
| L18 down c41 | MLP | a%100 in {6..15} (-) | 1.00 | 0.150 |
| L18 down c420 | MLP | a%100 in {60} (+) | 1.00 | 0.010 |
| L18 down c354 | MLP | a%100 in {68..75} (-) | 1.00 | 0.090 |
| L18 down c365 | MLP | a%100 in {80, 82..83, 85..99} (-) | 1.00 | 0.180 |
| L18 down c811 | MLP | a%100 in {90..92} (+) | 1.00 | 0.040 |
| L18 down c693 | MLP | a%100 in {92..99} (+) | 1.00 | 0.080 |
| L18 down c40 | MLP | a%100 in {99} (-) | 0.99 | 0.092 |

**code 1a-14.7 (L19): 31 comps, tells apart 59/100, coverage 0.72, overlap 1.86 (random 1.90, p 0.45)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L19 down c69 | MLP | a%100 in {0, 50, 52, 60} (-) | 1.00 | 0.040 |
| L19 down c13 | MLP | a%100 in {0..31, 50, 67, 73} (-) | 1.00 | 0.670 |
| L19 down c31 | MLP | a%100 in {1, 5..12, 15} (-) | 1.00 | 0.120 |
| L19 down c45 | MLP | a%100 in {1..3} (-) | 1.00 | 1.000 |
| L19 down c25 | MLP | a%100 in {1..6} (+) | 1.00 | 0.110 |
| L19 down c291 | MLP | a%100 in {12..13} (-) | 1.00 | 0.030 |
| L19 down c584 | MLP | a%100 in {13, 17} (+) | 1.00 | 0.020 |
| L19 down c166 | MLP | a%100 in {14, 21} (+) | 1.00 | 0.020 |
| L19 down c206 | MLP | a%100 in {16, 32, 64, 80} (+) | 1.00 | 0.060 |
| L19 down c64 | MLP | a%100 in {17..19} (+) | 1.00 | 0.030 |
| L19 down c52 | MLP | a%100 in {1} (-) | 1.00 | 0.100 |
| L19 down c241 | MLP | a%100 in {23..29} (-) | 1.00 | 0.090 |
| L19 down c677 | MLP | a%100 in {24} (-) | 1.00 | 0.020 |
| L19 down c627 | MLP | a%100 in {28, 38, 78} (+) | 1.00 | 0.040 |
| L19 down c93 | MLP | a%100 in {3, 93} (-) | 1.00 | 0.090 |
| L19 down c44 | MLP | a%100 in {31..34} (-) | 1.00 | 0.050 |
| L19 down c219 | MLP | a%100 in {34..35} (-) | 1.00 | 0.020 |
| L19 down c619 | MLP | a%100 in {35..37} (+) | 1.00 | 0.030 |
| L19 down c72 | MLP | a%100 in {47, 49, 55} (-) | 1.00 | 0.030 |
| L19 down c103 | MLP | a%100 in {47..52} (+) | 1.00 | 0.070 |
| L19 down c654 | MLP | a%100 in {48, 72} (-) | 1.00 | 0.020 |
| L19 down c360 | MLP | a%100 in {50} (-) | 1.00 | 0.010 |
| L19 down c820 | MLP | a%100 in {58..63} (+) | 1.00 | 0.060 |
| L19 down c78 | MLP | a%100 in {6..8} (-) | 1.00 | 0.050 |
| L19 down c572 | MLP | a%100 in {75, 95} (+) | 1.00 | 0.020 |
| L19 down c131 | MLP | a%100 in {75} (-) | 1.00 | 0.010 |
| L19 down c54 | MLP | a%100 in {77..83} (+) | 1.00 | 0.070 |
| L19 down c23 | MLP | a%100 in {90, 92..94} (+) | 1.00 | 0.040 |
| L19 down c189 | MLP | a%100 in {97..99} (+) | 1.00 | 0.060 |
| L19 down c455 | MLP | a%100 in {99} (+) | 1.00 | 0.020 |
| L19 down c51 | MLP | a%100 in {9} (-) | 1.00 | 0.020 |

**code 1a-14.8 (L21): 54 comps, tells apart 83/100, coverage 0.91, overlap 2.48 (random 2.74, p 0.23)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c419 | MLP | a%100 in {0, 10, 20, 30, 40, 50} (+) | 1.00 | 0.080 |
| L21 down c11 | MLP | a%100 in {0..10} (+) | 1.00 | 1.000 |
| L21 down c313 | MLP | a%100 in {1..2} (+) | 1.00 | 0.020 |
| L21 down c128 | MLP | a%100 in {11, 23} (+) | 1.00 | 0.020 |
| L21 down c116 | MLP | a%100 in {11..12, 14, 17..18} (+) | 1.00 | 0.070 |
| L21 down c326 | MLP | a%100 in {13} (-) | 1.00 | 0.020 |
| L21 down c156 | MLP | a%100 in {14..15, 21} (-) | 1.00 | 0.030 |
| L21 down c302 | MLP | a%100 in {16, 32} (+) | 1.00 | 0.050 |
| L21 down c106 | MLP | a%100 in {18..20, 26, 76} (+) | 1.00 | 0.050 |
| L21 down c298 | MLP | a%100 in {18} (+) | 1.00 | 0.010 |
| L21 down c56 | MLP | a%100 in {18} (-) | 1.00 | 0.010 |
| L21 down c58 | MLP | a%100 in {2..12} (+) | 1.00 | 0.210 |
| L21 down c219 | MLP | a%100 in {2..3} (+) | 1.00 | 0.020 |
| L21 down c243 | MLP | a%100 in {20, 38} (+) | 1.00 | 0.030 |
| L21 down c110 | MLP | a%100 in {21..31} (+) | 1.00 | 0.110 |
| L21 down c33 | MLP | a%100 in {24, 32, 48, 64, 72, 96} (+) | 1.00 | 0.140 |
| L21 down c39 | MLP | a%100 in {24} (-) | 1.00 | 0.030 |
| L21 down c426 | MLP | a%100 in {27} (+) | 1.00 | 0.010 |
| L21 down c160 | MLP | a%100 in {28..30} (+) | 1.00 | 0.060 |
| L21 down c432 | MLP | a%100 in {28} (+) | 1.00 | 0.010 |
| L21 down c47 | MLP | a%100 in {3, 5, 7, 15, 21} (+) | 1.00 | 0.060 |
| L21 down c8 | MLP | a%100 in {3..34} (+) | 1.00 | 0.390 |
| L21 down c66 | MLP | a%100 in {30, 60, 90} (+) | 1.00 | 0.030 |
| L21 down c422 | MLP | a%100 in {30..31} (-) | 1.00 | 0.020 |
| L21 down c18 | MLP | a%100 in {30..42} (-) | 1.00 | 0.190 |
| L21 down c87 | MLP | a%100 in {31, 41, 51, 61, 71, 81, 91} (-) | 1.00 | 0.080 |
| L21 down c312 | MLP | a%100 in {31} (-) | 1.00 | 0.010 |
| L21 down c203 | MLP | a%100 in {33, 73, 83, 93} (+) | 1.00 | 0.040 |
| L21 down c154 | MLP | a%100 in {35, 55, 79..85} (-) | 1.00 | 0.090 |
| L21 down c44 | MLP | a%100 in {36} (+) | 1.00 | 0.010 |
| L21 down c72 | MLP | a%100 in {37..39} (-) | 1.00 | 0.040 |
| L21 down c272 | MLP | a%100 in {39..41, 43} (-) | 1.00 | 0.070 |
| L21 down c524 | MLP | a%100 in {45, 75, 95} (+) | 1.00 | 0.030 |
| L21 down c152 | MLP | a%100 in {45..47} (-) | 1.00 | 0.030 |
| L21 down c34 | MLP | a%100 in {45..56} (-) | 1.00 | 0.140 |
| L21 down c144 | MLP | a%100 in {4} (+) | 1.00 | 0.080 |
| L21 down c364 | MLP | a%100 in {4} (-) | 1.00 | 0.010 |
| L21 down c37 | MLP | a%100 in {5..7} (+) | 1.00 | 0.030 |
| L21 down c86 | MLP | a%100 in {52} (+) | 1.00 | 0.010 |
| L21 down c166 | MLP | a%100 in {52} (-) | 1.00 | 0.010 |
| L21 down c216 | MLP | a%100 in {55} (-) | 1.00 | 0.010 |
| L21 down c177 | MLP | a%100 in {59..62, 64..65} (-) | 1.00 | 0.070 |
| L21 down c229 | MLP | a%100 in {67..72} (-) | 1.00 | 0.060 |
| L21 down c346 | MLP | a%100 in {70} (-) | 1.00 | 0.010 |
| L21 down c132 | MLP | a%100 in {73..77} (-) | 1.00 | 0.050 |
| L21 down c515 | MLP | a%100 in {74} (-) | 1.00 | 0.010 |
| L21 down c84 | MLP | a%100 in {80..86, 88..89} (-) | 1.00 | 0.090 |
| L21 down c109 | MLP | a%100 in {85, 87..89} (+) | 1.00 | 0.040 |
| L21 down c430 | MLP | a%100 in {88} (-) | 1.00 | 0.010 |
| L21 down c98 | MLP | a%100 in {8} (-) | 1.00 | 0.030 |
| L21 down c539 | MLP | a%100 in {91..92} (+) | 1.00 | 0.020 |
| L21 down c213 | MLP | a%100 in {99} (-) | 1.00 | 0.020 |
| L21 down c573 | MLP | a%100 in {99} (-) | 1.00 | 0.050 |
| L21 down c457 | MLP | a%100 in {9} (+) | 1.00 | 0.020 |

**code 1a-14.9 (L24): 19 comps, tells apart 41/100, coverage 0.76, overlap 1.55 (random 1.49, p 0.62)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 down c0 | MLP | a%100 in {0..4, 8..9, 24, 29, 55, 65, 70, 85} (-) | 1.00 | 1.000 |
| L24 down c3 | MLP | a%100 in {0..7, 10, 12, 50, 99} (+) | 1.00 | 1.000 |
| L24 down c35 | MLP | a%100 in {1..3} (+) | 1.00 | 0.040 |
| L24 down c45 | MLP | a%100 in {1} (+) | 1.00 | 0.060 |
| L24 down c9 | MLP | a%100 in {24, 32, 36, 42, 48, 64, 72} (-) | 1.00 | 0.150 |
| L24 down c310 | MLP | a%100 in {26..31} (-) | 1.00 | 0.070 |
| L24 down c33 | MLP | a%100 in {32..49} (-) | 1.00 | 0.190 |
| L24 down c95 | MLP | a%100 in {38..39} (+) | 1.00 | 0.050 |
| L24 down c170 | MLP | a%100 in {49..54} (+) | 1.00 | 0.060 |
| L24 down c213 | MLP | a%100 in {4} (-) | 1.00 | 0.030 |
| L24 down c32 | MLP | a%100 in {5..31} (+) | 1.00 | 0.290 |
| L24 down c208 | MLP | a%100 in {52} (-) | 1.00 | 0.010 |
| L24 down c141 | MLP | a%100 in {54..57} (+) | 1.00 | 0.070 |
| L24 down c49 | MLP | a%100 in {60, 80, 90} (+) | 1.00 | 0.040 |
| L24 down c61 | MLP | a%100 in {74, 76, 78, 80, 82..84} (+) | 1.00 | 0.070 |
| L24 down c227 | MLP | a%100 in {90..93} (+) | 1.00 | 0.040 |
| L24 down c292 | MLP | a%100 in {90} (-) | 1.00 | 0.010 |
| L24 down c109 | MLP | a%100 in {99} (+) | 1.00 | 0.010 |
| L24 down c334 | MLP | a%100 in {99} (+) | 1.00 | 0.040 |

</details>

</details>

<details><summary><b>1a-15</b> `a%100` @ `a` (add) — block code; 4 comps, L3; tells apart 10/100 classes (best member 4); on sub: 1s-91 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.42 |
| classes told apart: joint / best code / best member | 10 /  / 4 (of 100) |
| members whose removal merges classes | 0.75 |
| support overlap (1 = tiling) / random sets / p | 1.07 / 1.06 / 0.57 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.99 /  / 0.69 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.94 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 MLP |
| CKA(arrangement before, joint write) | 0.69 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.25 2:0.10 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.41 2:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.23 / 2.46 |

<details><summary>codes and components</summary>

**code 1a-15.0 (L3): 4 comps, tells apart 10/100, coverage 0.42, overlap 1.07 (random 1.00, p 0.67)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 o c221 | H5 | a%100 in {2..4} (-) | 1.00 | 0.030 |
| L3 o c128 | H5 | a%100 in {3..4, 79, 97} (-) | 1.00 | 1.000 |
| L3 o c354 | H5 | a%100 in {4..17, 19..21, 23, 25, 27, 29, 37, 72, 75..76, 78, 80, 82, 85..86, 89..91, 93, 95, 99} (-) | 1.00 | 1.000 |
| L3 o c85 | H19 | a%100 in {52, 74} (-) | 0.98 | 0.020 |

</details>

</details>

<details><summary><b>1a-16</b> `a%100` @ `a` (add) — block code; 2 comps, L4; tells apart 4/100 classes (best member 3); on sub: 1s-92 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.47 |
| classes told apart: joint / best code / best member | 4 /  / 3 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.09 / 1.00 / 0.88 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.62 /  / 0.36 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.98 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L3 MLP |
| CKA(arrangement before, joint write) | 0.30 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.24 2:0.11 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.34 2:0.19 3:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.09 / 1.05 |

<details><summary>codes and components</summary>

**code 1a-16.0 (L4): 2 comps, tells apart 4/100, coverage 0.47, overlap 1.09 (random 1.00, p 0.92)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 o c7 | H18 | a%100 in {1..6, 9..13, 24, 36, 40, 52, 56, 59, 75, 88..89, 98} (+) | 1.00 | 1.000 |
| L4 o c13 | H26 | a%100 in {21, 31, 48..49, 52, 69, 71..93, 96} (-) | 1.00 | 0.311 |

</details>

</details>

<details><summary><b>1a-17</b> `a%100` @ `a` (add) — block code; 2 comps, L5; tells apart 8/100 classes (best member 3); on sub: 1s-93 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.17 |
| classes told apart: joint / best code / best member | 8 /  / 3 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.12 / 1.00 / 0.95 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.89 /  / 0.45 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.07 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L4 MLP |
| CKA(arrangement before, joint write) | 0.21 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.25 2:0.12 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.14 2:0.13 40:0.11) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.04 / 1.22 |

<details><summary>codes and components</summary>

**code 1a-17.0 (L5): 2 comps, tells apart 8/100, coverage 0.17, overlap 1.12 (random 1.00, p 0.94)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 o c2 | H26 | a%100 in {1..2, 73, 75} (+) | 1.00 | 1.000 |
| L5 o c6 | H26 | a%100 in {1..4, 11, 33, 38, 45, 61, 77, 79, 83, 91..93} (+) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>1a-18</b> `a%100` @ `a` (add) — single component; 1 comps, L6; tells apart 12/100 classes (best member 12); on sub: 1s-94 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.25 |
| classes told apart: joint / best code / best member | 12 /  / 12 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.12 /  / 0.12 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L5 MLP |
| CKA(arrangement before, joint write) | 0.10 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.26 2:0.12 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.12 2:0.10 7:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.02 / 1.00 |

<details><summary>codes and components</summary>

**code 1a-18.0 (L6): 1 comps, tells apart 12/100, coverage 0.25, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L6 o c4 | H22 | a%100 in {0..2, 9..13, 24, 27, 39, 42..43, 52, 60..62, 64, 83, 91, 95..99} (+) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>1a-41</b> `a%100` @ `a` (add) — tiling, 8 codes of the same shape; 195 comps, L6 L20 L22 L25 L26 L27 L28 L30; tells apart 69/100 classes (best member 7); on sub: 1s-95 (member overlap 0.99)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.98 |
| classes told apart: joint / best code / best member | 69 / 61 / 7 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p | 8.65 / 8.79 / 0.41 |
| mean CKA between its codes | 0.77 |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 1.00 / 1.00 / 0.71 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.20 |
| consumers / read jointly | 107 / 101 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L6 attn |
| CKA(arrangement before, joint write) | 0.74 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 10.17 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.26 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.21 2:0.15 3:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.32 / 7.24 |

<details><summary>codes and components</summary>

**code 1a-41.0 (L6): 14 comps, tells apart 21/100, coverage 0.37, overlap 1.76 (random 1.35, p 0.95)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L6 down c244 | MLP | a%100 in {0, 40, 50, 60} (-) | 1.00 | 0.040 |
| L6 down c15 | MLP | a%100 in {0..5, 7, 10} (+) | 1.00 | 1.000 |
| L6 down c848 | MLP | a%100 in {0} (+) | 1.00 | 0.020 |
| L6 down c191 | MLP | a%100 in {1..3} (-) | 1.00 | 0.040 |
| L6 down c0 | MLP | a%100 in {1..8, 10, 12} (-) | 1.00 | 1.000 |
| L6 down c187 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L6 down c220 | MLP | a%100 in {2..10} (-) | 1.00 | 0.100 |
| L6 down c96 | MLP | a%100 in {24, 42, 83} (-) | 1.00 | 0.030 |
| L6 down c997 | MLP | a%100 in {52} (-) | 1.00 | 0.010 |
| L6 down c62 | MLP | a%100 in {61, 91} (-) | 1.00 | 0.020 |
| L6 down c892 | MLP | a%100 in {74} (+) | 1.00 | 0.010 |
| L6 down c28 | MLP | a%100 in {83..99} (-) | 1.00 | 0.170 |
| L6 down c522 | MLP | a%100 in {91} (+) | 1.00 | 0.010 |
| L6 down c423 | MLP | a%100 in {95, 97..99} (+) | 1.00 | 0.060 |

**code 1a-41.1 (L20): 37 comps, tells apart 61/100, coverage 0.79, overlap 1.59 (random 2.07, p 0.01)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 down c607 | MLP | a%100 in {0, 99} (+) | 1.00 | 0.020 |
| L20 down c440 | MLP | a%100 in {1..2} (+) | 1.00 | 0.030 |
| L20 down c0 | MLP | a%100 in {1..9} (-) | 1.00 | 1.000 |
| L20 down c605 | MLP | a%100 in {10} (+) | 1.00 | 0.010 |
| L20 down c35 | MLP | a%100 in {12, 24, 36, 48, 72, 96} (+) | 1.00 | 0.090 |
| L20 down c69 | MLP | a%100 in {14..18} (-) | 1.00 | 0.070 |
| L20 down c986 | MLP | a%100 in {14} (+) | 1.00 | 0.010 |
| L20 down c575 | MLP | a%100 in {16, 96} (-) | 1.00 | 0.030 |
| L20 down c520 | MLP | a%100 in {18} (+) | 1.00 | 0.010 |
| L20 down c839 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L20 down c349 | MLP | a%100 in {2..4} (-) | 1.00 | 0.030 |
| L20 down c131 | MLP | a%100 in {2..8} (+) | 1.00 | 0.120 |
| L20 down c81 | MLP | a%100 in {21..24} (+) | 1.00 | 0.110 |
| L20 down c38 | MLP | a%100 in {25..31} (-) | 1.00 | 0.070 |
| L20 down c247 | MLP | a%100 in {27} (+) | 1.00 | 0.010 |
| L20 down c399 | MLP | a%100 in {30..33} (+) | 1.00 | 0.040 |
| L20 down c22 | MLP | a%100 in {31} (-) | 1.00 | 0.080 |
| L20 down c630 | MLP | a%100 in {38..43} (-) | 1.00 | 0.060 |
| L20 down c331 | MLP | a%100 in {40, 50, 60, 70, 80, 90} (-) | 0.99 | 0.077 |
| L20 down c710 | MLP | a%100 in {40} (+) | 1.00 | 0.010 |
| L20 down c46 | MLP | a%100 in {43..49} (+) | 1.00 | 0.090 |
| L20 down c98 | MLP | a%100 in {49..52} (-) | 1.00 | 0.050 |
| L20 down c75 | MLP | a%100 in {53..60} (+) | 1.00 | 0.090 |
| L20 down c66 | MLP | a%100 in {60, 62..69} (-) | 1.00 | 0.090 |
| L20 down c71 | MLP | a%100 in {65, 67} (-) | 1.00 | 0.020 |
| L20 down c110 | MLP | a%100 in {65..66} (+) | 1.00 | 0.020 |
| L20 down c320 | MLP | a%100 in {6} (-) | 1.00 | 0.020 |
| L20 down c50 | MLP | a%100 in {7..12, 15} (+) | 1.00 | 0.100 |
| L20 down c77 | MLP | a%100 in {74..77} (+) | 1.00 | 0.040 |
| L20 down c322 | MLP | a%100 in {7} (-) | 1.00 | 0.010 |
| L20 down c348 | MLP | a%100 in {7} (-) | 1.00 | 0.010 |
| L20 down c643 | MLP | a%100 in {86, 88} (+) | 1.00 | 0.020 |
| L20 down c78 | MLP | a%100 in {90..93} (+) | 1.00 | 0.060 |
| L20 down c149 | MLP | a%100 in {98} (-) | 1.00 | 0.010 |
| L20 down c932 | MLP | a%100 in {99} (-) | 1.00 | 0.020 |
| L20 down c870 | MLP | a%100 in {9} (+) | 1.00 | 0.010 |
| L20 down c915 | MLP | a%100 in {9} (+) | 1.00 | 0.010 |

**code 1a-41.2 (L22): 26 comps, tells apart 54/100, coverage 0.75, overlap 1.72 (random 1.71, p 0.51)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 down c113 | MLP | a%100 in {0, 10, 20, 25, 30, 40, 50} (+) | 1.00 | 0.110 |
| L22 down c2 | MLP | a%100 in {0..5} (-) | 1.00 | 0.990 |
| L22 down c49 | MLP | a%100 in {1..2} (+) | 1.00 | 0.020 |
| L22 down c93 | MLP | a%100 in {1..2} (+) | 1.00 | 0.040 |
| L22 down c0 | MLP | a%100 in {1..9} (-) | 1.00 | 1.000 |
| L22 down c176 | MLP | a%100 in {14..23} (-) | 1.00 | 0.110 |
| L22 down c197 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L22 down c74 | MLP | a%100 in {2..11} (+) | 1.00 | 0.150 |
| L22 down c85 | MLP | a%100 in {2..5} (+) | 1.00 | 0.040 |
| L22 down c242 | MLP | a%100 in {2..5} (+) | 1.00 | 0.040 |
| L22 down c39 | MLP | a%100 in {21, 31, 41, 51, 61, 71, 81, 91} (+) | 1.00 | 0.080 |
| L22 down c26 | MLP | a%100 in {24, 32, 36, 48, 52, 64, 72, 96} (-) | 0.97 | 0.103 |
| L22 down c162 | MLP | a%100 in {2} (-) | 1.00 | 0.010 |
| L22 down c308 | MLP | a%100 in {30, 40, 50, 60, 70, 75, 80, 90} (+) | 1.00 | 0.120 |
| L22 down c70 | MLP | a%100 in {32, 64} (-) | 1.00 | 0.020 |
| L22 down c186 | MLP | a%100 in {33, 38..39, 43, 88, 93} (+) | 1.00 | 0.060 |
| L22 down c54 | MLP | a%100 in {35} (+) | 1.00 | 0.010 |
| L22 down c21 | MLP | a%100 in {38, 40..48} (+) | 1.00 | 0.130 |
| L22 down c148 | MLP | a%100 in {54..57} (-) | 1.00 | 0.040 |
| L22 down c495 | MLP | a%100 in {60} (+) | 1.00 | 0.010 |
| L22 down c120 | MLP | a%100 in {72..76} (+) | 1.00 | 0.050 |
| L22 down c264 | MLP | a%100 in {80} (-) | 1.00 | 0.010 |
| L22 down c52 | MLP | a%100 in {82, 86..93} (-) | 1.00 | 0.090 |
| L22 down c24 | MLP | a%100 in {91..98} (-) | 1.00 | 0.081 |
| L22 down c723 | MLP | a%100 in {99} (+) | 1.00 | 0.020 |
| L22 down c640 | MLP | a%100 in {9} (+) | 1.00 | 0.020 |

**code 1a-41.3 (L25): 26 comps, tells apart 51/100, coverage 0.59, overlap 1.64 (random 1.72, p 0.40)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L25 down c97 | MLP | a%100 in {0, 10, 20, 25, 30, 40, 50} (+) | 1.00 | 0.110 |
| L25 down c119 | MLP | a%100 in {0, 30, 40, 45, 50, 60, 65, 70, 75, 80, 85, 90} (-) | 1.00 | 0.170 |
| L25 down c346 | MLP | a%100 in {0, 75} (+) | 1.00 | 0.020 |
| L25 down c0 | MLP | a%100 in {0..1, 99} (+) | 1.00 | 1.000 |
| L25 down c120 | MLP | a%100 in {1..2} (-) | 1.00 | 0.020 |
| L25 down c378 | MLP | a%100 in {1..3} (+) | 1.00 | 0.040 |
| L25 down c47 | MLP | a%100 in {12, 32, 36, 42, 48, 52, 72} (+) | 1.00 | 0.110 |
| L25 down c360 | MLP | a%100 in {1} (+) | 1.00 | 0.060 |
| L25 down c363 | MLP | a%100 in {2..5} (-) | 1.00 | 0.080 |
| L25 down c39 | MLP | a%100 in {24, 48} (+) | 1.00 | 0.040 |
| L25 down c59 | MLP | a%100 in {3..12} (+) | 1.00 | 0.160 |
| L25 down c369 | MLP | a%100 in {31} (-) | 1.00 | 0.010 |
| L25 down c36 | MLP | a%100 in {32, 64} (+) | 1.00 | 0.060 |
| L25 down c70 | MLP | a%100 in {32..43} (-) | 1.00 | 0.120 |
| L25 down c607 | MLP | a%100 in {47..51} (+) | 1.00 | 0.050 |
| L25 down c24 | MLP | a%100 in {49} (+) | 1.00 | 0.010 |
| L25 down c458 | MLP | a%100 in {50} (+) | 1.00 | 0.010 |
| L25 down c101 | MLP | a%100 in {52} (-) | 1.00 | 0.010 |
| L25 down c293 | MLP | a%100 in {57..60} (-) | 1.00 | 0.080 |
| L25 down c84 | MLP | a%100 in {60, 90} (-) | 1.00 | 0.020 |
| L25 down c121 | MLP | a%100 in {72..74} (-) | 1.00 | 0.030 |
| L25 down c20 | MLP | a%100 in {9, 11..12} (+) | 1.00 | 0.040 |
| L25 down c65 | MLP | a%100 in {90} (-) | 1.00 | 0.050 |
| L25 down c28 | MLP | a%100 in {92..94, 96..98} (+) | 1.00 | 0.060 |
| L25 down c592 | MLP | a%100 in {95} (+) | 1.00 | 0.010 |
| L25 down c176 | MLP | a%100 in {99} (-) | 1.00 | 0.060 |

**code 1a-41.4 (L26): 27 comps, tells apart 49/100, coverage 0.72, overlap 1.71 (random 1.77, p 0.40)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 down c54 | MLP | a%100 in {0, 60, 80, 90} (+) | 1.00 | 0.060 |
| L26 down c18 | MLP | a%100 in {0..8, 10} (+) | 1.00 | 1.000 |
| L26 down c737 | MLP | a%100 in {0} (+) | 1.00 | 0.020 |
| L26 down c47 | MLP | a%100 in {1..10} (+) | 1.00 | 1.000 |
| L26 down c59 | MLP | a%100 in {1..2} (+) | 1.00 | 0.040 |
| L26 down c173 | MLP | a%100 in {15, 20, 30, 40, 60, 90} (+) | 1.00 | 0.060 |
| L26 down c114 | MLP | a%100 in {18} (-) | 1.00 | 0.060 |
| L26 down c82 | MLP | a%100 in {21, 31, 41, 51, 61, 71, 81, 91} (+) | 1.00 | 0.080 |
| L26 down c65 | MLP | a%100 in {24, 48, 72} (+) | 1.00 | 0.040 |
| L26 down c5 | MLP | a%100 in {3..15, 18} (+) | 1.00 | 0.210 |
| L26 down c252 | MLP | a%100 in {3..6} (-) | 1.00 | 0.050 |
| L26 down c574 | MLP | a%100 in {31} (-) | 1.00 | 0.010 |
| L26 down c586 | MLP | a%100 in {32..40} (+) | 1.00 | 0.090 |
| L26 down c587 | MLP | a%100 in {40, 50, 70, 80} (-) | 1.00 | 0.040 |
| L26 down c200 | MLP | a%100 in {41, 51, 61, 71, 81, 91} (+) | 1.00 | 0.060 |
| L26 down c107 | MLP | a%100 in {42..43, 46..47, 49} (-) | 1.00 | 0.050 |
| L26 down c38 | MLP | a%100 in {49, 69, 76} (-) | 1.00 | 0.060 |
| L26 down c506 | MLP | a%100 in {5..8, 10} (-) | 1.00 | 0.050 |
| L26 down c210 | MLP | a%100 in {50} (+) | 1.00 | 0.010 |
| L26 down c117 | MLP | a%100 in {52} (-) | 1.00 | 0.010 |
| L26 down c249 | MLP | a%100 in {55} (-) | 1.00 | 0.010 |
| L26 down c369 | MLP | a%100 in {60, 62..68} (+) | 1.00 | 0.090 |
| L26 down c120 | MLP | a%100 in {60, 90} (-) | 1.00 | 0.030 |
| L26 down c582 | MLP | a%100 in {74} (+) | 1.00 | 0.010 |
| L26 down c407 | MLP | a%100 in {80..84} (+) | 1.00 | 0.050 |
| L26 down c204 | MLP | a%100 in {92..98} (+) | 1.00 | 0.070 |
| L26 down c453 | MLP | a%100 in {99} (+) | 1.00 | 0.050 |

**code 1a-41.5 (L27): 25 comps, tells apart 51/100, coverage 0.66, overlap 1.68 (random 1.71, p 0.43)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 down c690 | MLP | a%100 in {0, 20, 30, 40, 45, 50, 60, 70, 75, 80, 90} (+) | 1.00 | 0.160 |
| L27 down c1 | MLP | a%100 in {0..8, 10} (-) | 1.00 | 1.000 |
| L27 down c36 | MLP | a%100 in {0} (-) | 1.00 | 0.050 |
| L27 down c303 | MLP | a%100 in {1..3} (-) | 1.00 | 0.040 |
| L27 down c2 | MLP | a%100 in {10, 15, 20, 26, 29, 32, 34..35, 37..38, 47, 49..50, 74} (+) | 1.00 | 1.000 |
| L27 down c63 | MLP | a%100 in {12, 16, 32, 36, 42, 48, 72, 96} (-) | 1.00 | 0.080 |
| L27 down c273 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L27 down c327 | MLP | a%100 in {2..8} (-) | 1.00 | 0.110 |
| L27 down c141 | MLP | a%100 in {21, 31, 41, 51, 61, 71, 81, 91} (+) | 1.00 | 0.080 |
| L27 down c144 | MLP | a%100 in {21..23, 28..29} (-) | 1.00 | 0.050 |
| L27 down c103 | MLP | a%100 in {24, 48, 64} (+) | 1.00 | 0.070 |
| L27 down c852 | MLP | a%100 in {2} (-) | 1.00 | 0.010 |
| L27 down c43 | MLP | a%100 in {33, 38..39, 41..44, 46..49} (-) | 1.00 | 0.210 |
| L27 down c109 | MLP | a%100 in {41, 51} (-) | 1.00 | 0.020 |
| L27 down c131 | MLP | a%100 in {42} (-) | 1.00 | 0.010 |
| L27 down c779 | MLP | a%100 in {4} (+) | 1.00 | 0.010 |
| L27 down c472 | MLP | a%100 in {5, 10} (+) | 1.00 | 0.060 |
| L27 down c149 | MLP | a%100 in {50..52} (-) | 1.00 | 0.040 |
| L27 down c608 | MLP | a%100 in {60, 90} (+) | 1.00 | 0.030 |
| L27 down c277 | MLP | a%100 in {7..12} (-) | 1.00 | 0.070 |
| L27 down c559 | MLP | a%100 in {74} (+) | 1.00 | 0.010 |
| L27 down c199 | MLP | a%100 in {80} (+) | 1.00 | 0.010 |
| L27 down c197 | MLP | a%100 in {85..89} (+) | 1.00 | 0.070 |
| L27 down c552 | MLP | a%100 in {88, 90, 92} (-) | 1.00 | 0.050 |
| L27 down c923 | MLP | a%100 in {99} (+) | 1.00 | 0.010 |

**code 1a-41.6 (L28): 22 comps, tells apart 46/100, coverage 0.65, overlap 1.62 (random 1.57, p 0.59)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 down c17 | MLP | a%100 in {0, 20, 25, 30, 35, 40, 45, 50, 60, 70, 75, 80, 85, 90, 95} (-) | 1.00 | 0.170 |
| L28 down c915 | MLP | a%100 in {0, 30, 40, 50, 60, 80, 90} (-) | 1.00 | 0.070 |
| L28 down c18 | MLP | a%100 in {0, 80} (-) | 1.00 | 0.020 |
| L28 down c0 | MLP | a%100 in {0..5, 7, 99} (+) | 1.00 | 1.000 |
| L28 down c96 | MLP | a%100 in {1..2} (-) | 1.00 | 0.020 |
| L28 down c741 | MLP | a%100 in {12, 48} (+) | 1.00 | 0.020 |
| L28 down c6 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L28 down c497 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L28 down c477 | MLP | a%100 in {2..3} (-) | 1.00 | 0.020 |
| L28 down c627 | MLP | a%100 in {2..8} (-) | 1.00 | 0.100 |
| L28 down c424 | MLP | a%100 in {22..23, 25..31} (-) | 1.00 | 0.100 |
| L28 down c79 | MLP | a%100 in {24} (-) | 1.00 | 0.051 |
| L28 down c135 | MLP | a%100 in {32..45} (-) | 1.00 | 0.160 |
| L28 down c77 | MLP | a%100 in {49..50} (+) | 1.00 | 0.030 |
| L28 down c63 | MLP | a%100 in {5, 7, 9} (+) | 1.00 | 0.030 |
| L28 down c9 | MLP | a%100 in {5..12, 15} (+) | 1.00 | 0.140 |
| L28 down c350 | MLP | a%100 in {52} (-) | 1.00 | 0.050 |
| L28 down c501 | MLP | a%100 in {54..55} (+) | 1.00 | 0.070 |
| L28 down c8 | MLP | a%100 in {75} (-) | 1.00 | 0.010 |
| L28 down c639 | MLP | a%100 in {83, 86..98} (-) | 1.00 | 0.170 |
| L28 down c4 | MLP | a%100 in {93} (-) | 1.00 | 0.090 |
| L28 down c405 | MLP | a%100 in {99} (+) | 1.00 | 0.080 |

**code 1a-41.7 (L30): 18 comps, tells apart 33/100, coverage 0.61, overlap 1.51 (random 1.47, p 0.58)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 down c480 | MLP | a%100 in {0, 5, 10, 40, 50, 99} (-) | 1.00 | 1.000 |
| L30 down c32 | MLP | a%100 in {1..2, 99} (-) | 1.00 | 1.000 |
| L30 down c142 | MLP | a%100 in {1..2} (+) | 1.00 | 0.020 |
| L30 down c1 | MLP | a%100 in {1..3, 31, 37..38, 51, 61, 66, 71, 81, 88, 91..93} (+) | 1.00 | 1.000 |
| L30 down c259 | MLP | a%100 in {14, 16, 18..20, 22..30} (+) | 1.00 | 0.141 |
| L30 down c65 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L30 down c114 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L30 down c55 | MLP | a%100 in {2..10} (-) | 1.00 | 0.168 |
| L30 down c759 | MLP | a%100 in {21, 41, 51, 61, 71, 81, 91} (+) | 1.00 | 0.070 |
| L30 down c768 | MLP | a%100 in {29..31} (+) | 1.00 | 0.061 |
| L30 down c260 | MLP | a%100 in {2} (-) | 1.00 | 0.010 |
| L30 down c155 | MLP | a%100 in {31} (-) | 1.00 | 0.070 |
| L30 down c476 | MLP | a%100 in {39, 42..52} (+) | 1.00 | 0.280 |
| L30 down c948 | MLP | a%100 in {3} (+) | 1.00 | 0.020 |
| L30 down c189 | MLP | a%100 in {70, 80, 90} (+) | 1.00 | 0.040 |
| L30 down c968 | MLP | a%100 in {74} (-) | 1.00 | 0.010 |
| L30 down c991 | MLP | a%100 in {88, 90..99} (+) | 1.00 | 0.144 |
| L30 down c274 | MLP | a%100 in {99} (-) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>1a-19</b> `a%100` @ `a` (add) — single component; 1 comps, L7; tells apart 5/100 classes (best member 5); on sub: 1s-96 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.14 |
| classes told apart: joint / best code / best member | 5 /  / 5 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.40 /  / 0.40 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L6 MLP |
| CKA(arrangement before, joint write) | 0.41 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.27 2:0.12 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.21 2:0.11 5:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.04 / 1.00 |

<details><summary>codes and components</summary>

**code 1a-19.0 (L7): 1 comps, tells apart 5/100, coverage 0.14, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L7 o c14 | H20 | a%100 in {1..2, 8, 20, 23..24, 59, 83, 88, 95..99} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>1a-20</b> `a%100` @ `a` (add) — single component; 1 comps, L7; tells apart 11/100 classes (best member 11); on sub: 1s-97 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.02 |
| classes told apart: joint / best code / best member | 11 /  / 11 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.11 /  / 0.11 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L6 MLP |
| CKA(arrangement before, joint write) | 0.17 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.27 2:0.12 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.12 4:0.07 44:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.02 / 1.00 |

<details><summary>codes and components</summary>

**code 1a-20.0 (L7): 1 comps, tells apart 11/100, coverage 0.02, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L7 o c3 | H5 | a%100 in {1, 83} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>1a-21</b> `a%100` @ `a` (add) — block code; 2 comps, L8; tells apart 3/100 classes (best member 6); on sub: 1s-98 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.07 |
| classes told apart: joint / best code / best member | 3 /  / 6 (of 100) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 1.43 / 1.00 / 1.00 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.99 |
| decoding acc. joint / best code / best member (chance) | 0.94 /  / 0.48 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.98 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L7 MLP |
| CKA(arrangement before, joint write) | 0.46 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.26 2:0.12 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.26 2:0.17 3:0.11) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.07 / 1.01 |

<details><summary>codes and components</summary>

**code 1a-21.0 (L8): 2 comps, tells apart 3/100, coverage 0.07, overlap 1.43 (random 1.00, p 0.99)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L8 o c7 | H23 | a%100 in {1..3} (+) | 1.00 | 1.000 |
| L8 o c3 | H23 | a%100 in {1..7} (-) | 0.99 | 0.932 |

</details>

</details>

<details><summary><b>1a-22</b> `a%100` @ `a` (add) — block code, 3 codes of the same shape; 29 comps, L8 L9 L10; tells apart 24/100 classes (best member 5); on sub: 1s-99 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.39 |
| classes told apart: joint / best code / best member | 24 / 16 / 5 (of 100) |
| members whose removal merges classes | 0.31 |
| support overlap (1 = tiling) / random sets / p | 2.56 / 1.87 / 0.98 |
| mean CKA between its codes | 0.81 |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 1.00 / 0.90 / 0.36 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.91 |
| consumers / read jointly | 4 / 4 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L8 attn |
| CKA(arrangement before, joint write) | 0.54 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.09 |
| share of the write inside the old arrangement's span | 0.09 |
| write energy / code energy before | 0.10 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.26 2:0.12 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.15 2:0.12 3:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.13 / 3.73 |

<details><summary>codes and components</summary>

**code 1a-22.0 (L8): 8 comps, tells apart 10/100, coverage 0.19, overlap 1.84 (random 1.14, p 0.98)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L8 down c69 | MLP | a%100 in {1..2} (-) | 1.00 | 1.000 |
| L8 down c0 | MLP | a%100 in {1..3, 5, 24} (-) | 1.00 | 1.000 |
| L8 down c23 | MLP | a%100 in {1..3} (+) | 1.00 | 0.040 |
| L8 down c612 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L8 down c12 | MLP | a%100 in {2..10} (+) | 1.00 | 0.090 |
| L8 down c48 | MLP | a%100 in {3..9} (+) | 1.00 | 0.080 |
| L8 down c370 | MLP | a%100 in {52} (+) | 1.00 | 0.010 |
| L8 down c100 | MLP | a%100 in {92..95, 97..99} (-) | 1.00 | 0.070 |

**code 1a-22.1 (L9): 11 comps, tells apart 16/100, coverage 0.25, overlap 1.44 (random 1.25, p 0.83)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L9 down c529 | MLP | a%100 in {0, 50, 52} (+) | 1.00 | 0.030 |
| L9 down c38 | MLP | a%100 in {1..2} (+) | 1.00 | 1.000 |
| L9 down c369 | MLP | a%100 in {1..2} (-) | 1.00 | 0.030 |
| L9 down c158 | MLP | a%100 in {2..4} (+) | 1.00 | 0.030 |
| L9 down c289 | MLP | a%100 in {3..10} (+) | 1.00 | 0.080 |
| L9 down c64 | MLP | a%100 in {31, 41, 49, 51, 53, 59, 61, 81, 91} (-) | 1.00 | 0.360 |
| L9 down c165 | MLP | a%100 in {52} (+) | 1.00 | 0.010 |
| L9 down c841 | MLP | a%100 in {52} (-) | 1.00 | 0.010 |
| L9 down c18 | MLP | a%100 in {91, 93} (+) | 1.00 | 0.020 |
| L9 down c62 | MLP | a%100 in {91..93, 99} (-) | 1.00 | 0.040 |
| L9 down c738 | MLP | a%100 in {99} (-) | 1.00 | 0.010 |

**code 1a-22.2 (L10): 10 comps, tells apart 10/100, coverage 0.24, overlap 1.21 (random 1.21, p 0.49)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L10 down c2 | MLP | a%100 in {1..2} (-) | 1.00 | 1.000 |
| L10 down c493 | MLP | a%100 in {1..2} (-) | 1.00 | 0.020 |
| L10 down c391 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L10 down c138 | MLP | a%100 in {2} (+) | 1.00 | 0.010 |
| L10 down c203 | MLP | a%100 in {3..10} (+) | 1.00 | 0.080 |
| L10 down c75 | MLP | a%100 in {42..43} (+) | 1.00 | 0.020 |
| L10 down c54 | MLP | a%100 in {49, 51..59} (-) | 1.00 | 0.100 |
| L10 down c86 | MLP | a%100 in {52} (+) | 1.00 | 0.010 |
| L10 down c321 | MLP | a%100 in {74} (-) | 1.00 | 0.010 |
| L10 down c423 | MLP | a%100 in {83} (-) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>1a-23</b> `a%100` @ `a` (add) — block code; 2 comps, L9; tells apart 19/100 classes (best member 15); on sub: 1s-100 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.26 |
| classes told apart: joint / best code / best member | 19 /  / 15 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.27 / 1.00 / 0.96 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.65 /  / 0.19 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.99 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L8 MLP |
| CKA(arrangement before, joint write) | 0.65 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.26 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.31 2:0.13 3:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.13 / 1.93 |

<details><summary>codes and components</summary>

**code 1a-23.0 (L9): 2 comps, tells apart 19/100, coverage 0.26, overlap 1.27 (random 1.00, p 0.94)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L9 o c9 | H11 | a%100 in {0, 3..15, 24, 83, 86, 92, 95, 97..99} (-) | 1.00 | 1.000 |
| L9 o c183 | H11 | a%100 in {0..5, 10, 24, 50, 56, 99} (+) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>1a-24</b> `a%100` @ `a` (add) — single component; 1 comps, L10; tells apart 14/100 classes (best member 14); on sub: 1s-101 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.07 |
| classes told apart: joint / best code / best member | 14 /  / 14 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.14 /  / 0.14 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L9 MLP |
| CKA(arrangement before, joint write) | 0.24 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.25 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (4:0.21 2:0.10 40:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.04 / 1.00 |

<details><summary>codes and components</summary>

**code 1a-24.0 (L10): 1 comps, tells apart 14/100, coverage 0.07, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L10 o c1 | H30 | a%100 in {0..3, 5, 50, 52} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>1a-25</b> `a%100` @ `a` (add) — single component; 1 comps, L11; tells apart 8/100 classes (best member 8); on sub: 1s-102 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.09 |
| classes told apart: joint / best code / best member | 8 /  / 8 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.08 /  / 0.08 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L10 MLP |
| CKA(arrangement before, joint write) | 0.38 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.25 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.39 50:0.13 40:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.10 / 1.00 |

<details><summary>codes and components</summary>

**code 1a-25.0 (L11): 1 comps, tells apart 8/100, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L11 o c4 | H20 | a%100 in {0, 10, 12, 24, 47, 52..53, 67, 71} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>1a-26</b> `a%100` @ `a` (add) — block code; 17 comps, L11; tells apart 33/100 classes (best member 7); on sub: 1s-103 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.71 |
| classes told apart: joint / best code / best member | 33 /  / 7 (of 100) |
| members whose removal merges classes | 0.71 |
| support overlap (1 = tiling) / random sets / p | 1.63 / 1.44 / 0.81 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.99 /  / 0.47 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.03 |
| consumers / read jointly | 8 / 2 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L11 attn |
| CKA(arrangement before, joint write) | 0.72 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.08 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.25 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.27 2:0.18 3:0.13) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.46 / 6.58 |

<details><summary>codes and components</summary>

**code 1a-26.0 (L11): 17 comps, tells apart 33/100, coverage 0.71, overlap 1.63 (random 1.43, p 0.83)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L11 down c465 | MLP | a%100 in {0, 99} (-) | 1.00 | 0.020 |
| L11 down c1 | MLP | a%100 in {0..2, 5, 10, 50, 52, 99} (+) | 1.00 | 1.000 |
| L11 down c0 | MLP | a%100 in {1..3, 10..23, 30, 48, 50..52, 81, 83..88, 96} (-) | 1.00 | 1.000 |
| L11 down c294 | MLP | a%100 in {1..6} (+) | 1.00 | 0.060 |
| L11 down c31 | MLP | a%100 in {11..21} (-) | 1.00 | 0.160 |
| L11 down c43 | MLP | a%100 in {1} (+) | 1.00 | 0.010 |
| L11 down c47 | MLP | a%100 in {21..29} (+) | 1.00 | 0.130 |
| L11 down c113 | MLP | a%100 in {31..33} (-) | 1.00 | 0.030 |
| L11 down c89 | MLP | a%100 in {32..38} (+) | 1.00 | 0.080 |
| L11 down c74 | MLP | a%100 in {42} (-) | 1.00 | 0.010 |
| L11 down c117 | MLP | a%100 in {48..51} (+) | 1.00 | 0.040 |
| L11 down c13 | MLP | a%100 in {5..9} (+) | 1.00 | 0.050 |
| L11 down c18 | MLP | a%100 in {51..59} (+) | 1.00 | 0.130 |
| L11 down c223 | MLP | a%100 in {74, 76} (+) | 0.98 | 0.020 |
| L11 down c52 | MLP | a%100 in {79..87} (+) | 1.00 | 0.110 |
| L11 down c50 | MLP | a%100 in {85..92} (-) | 1.00 | 0.100 |
| L11 down c35 | MLP | a%100 in {93} (-) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>1a-27</b> `a%100` @ `a` (add) — block code; 2 comps, L12; tells apart 11/100 classes (best member 4); on sub: 1s-104 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.11 |
| classes told apart: joint / best code / best member | 11 /  / 4 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.45 / 1.00 / 0.98 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.84 /  / 0.46 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.91 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L11 MLP |
| CKA(arrangement before, joint write) | 0.19 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.25 2:0.13 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.19 40:0.11 4:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.04 / 1.13 |

<details><summary>codes and components</summary>

**code 1a-27.0 (L12): 2 comps, tells apart 11/100, coverage 0.11, overlap 1.45 (random 1.00, p 0.98)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L12 o c41 | H20 | a%100 in {0..1, 10, 50, 52, 60, 75, 99} (+) | 1.00 | 1.000 |
| L12 o c209 | H20 | a%100 in {0..2, 5, 7, 10, 50, 52} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>1a-28</b> `a%100` @ `a` (add) — block code; 18 comps, L12; tells apart 36/100 classes (best member 7); on sub: 1s-105 (member overlap 0.95)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.68 |
| classes told apart: joint / best code / best member | 36 /  / 7 (of 100) |
| members whose removal merges classes | 0.83 |
| support overlap (1 = tiling) / random sets / p | 1.54 / 1.47 / 0.64 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.99 |
| decoding acc. joint / best code / best member (chance) | 0.97 /  / 0.56 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.99 |
| consumers / read jointly | 6 / 1 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L12 attn |
| CKA(arrangement before, joint write) | 0.54 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.09 |
| write energy / code energy before | 0.04 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.25 2:0.13 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.23 2:0.20 3:0.12) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.36 / 5.92 |

<details><summary>codes and components</summary>

**code 1a-28.0 (L12): 18 comps, tells apart 36/100, coverage 0.68, overlap 1.54 (random 1.48, p 0.64)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L12 down c0 | MLP | a%100 in {0..1, 10, 20..21, 24..25, 30..31, 39, 50, 53, 55, 57..58, 62, 82..83, 85..89, 92..98} (-) | 1.00 | 1.000 |
| L12 down c8 | MLP | a%100 in {0..2, 52} (-) | 1.00 | 1.000 |
| L12 down c168 | MLP | a%100 in {1..2} (-) | 0.91 | 0.017 |
| L12 down c22 | MLP | a%100 in {11..23} (-) | 1.00 | 0.140 |
| L12 down c932 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L12 down c18 | MLP | a%100 in {23..29} (+) | 1.00 | 0.140 |
| L12 down c214 | MLP | a%100 in {24, 42, 48, 72} (-) | 1.00 | 0.050 |
| L12 down c76 | MLP | a%100 in {33, 44, 55, 66, 88, 99} (+) | 0.97 | 0.065 |
| L12 down c69 | MLP | a%100 in {34, 36} (-) | 1.00 | 0.020 |
| L12 down c327 | MLP | a%100 in {52} (+) | 1.00 | 0.010 |
| L12 down c550 | MLP | a%100 in {57..58} (+) | 0.94 | 0.022 |
| L12 down c870 | MLP | a%100 in {62} (-) | 1.00 | 0.010 |
| L12 down c364 | MLP | a%100 in {70..71, 73} (-) | 1.00 | 0.030 |
| L12 down c17 | MLP | a%100 in {71..81} (+) | 1.00 | 0.170 |
| L12 down c47 | MLP | a%100 in {76, 84, 86} (+) | 1.00 | 0.030 |
| L12 down c110 | MLP | a%100 in {78..82} (-) | 1.00 | 0.050 |
| L12 down c138 | MLP | a%100 in {92..93} (-) | 1.00 | 0.020 |
| L12 down c224 | MLP | a%100 in {92..99} (-) | 1.00 | 0.090 |

</details>

</details>

<details><summary><b>1a-29</b> `a%100` @ `a` (add) — single component; 1 comps, L13; tells apart 3/100 classes (best member 3); on sub: 1s-106 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.09 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.21 /  / 0.21 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L12 MLP |
| CKA(arrangement before, joint write) | 0.22 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.24 2:0.13 10:0.08) |
| joint write: shape (spectrum k:share) | line (1:0.30 2:0.28 4:0.05) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.09 / 1.00 |

<details><summary>codes and components</summary>

**code 1a-29.0 (L13): 1 comps, tells apart 3/100, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 o c0 | H16 | a%100 in {0, 2..5, 10, 33, 42, 52} (+) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>1a-30</b> `a%100` @ `a` (add) — single component; 1 comps, L14; tells apart 13/100 classes (best member 13); on sub: 1s-107 (member overlap 0.50)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.02 |
| classes told apart: joint / best code / best member | 13 /  / 13 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.13 /  / 0.13 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L13 MLP |
| CKA(arrangement before, joint write) | 0.25 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.23 2:0.13 10:0.08) |
| joint write: shape (spectrum k:share) | line (1:0.28 4:0.09 3:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.06 / 1.00 |

<details><summary>codes and components</summary>

**code 1a-30.0 (L14): 1 comps, tells apart 13/100, coverage 0.02, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L14 o c59 | H26 | a%100 in {0..1} (+) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>1a-32</b> `a%100` @ `a` (add) — block code; 3 comps, L15 L16; tells apart 5/100 classes (best member 4); on sub: 1s-108 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.08 |
| classes told apart: joint / best code / best member | 5 /  / 4 (of 100) |
| members whose removal merges classes | 0.67 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.69 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.25 /  / 0.23 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L14 MLP |
| CKA(arrangement before, joint write) | 0.16 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.23 2:0.12 10:0.08) |
| joint write: shape (spectrum k:share) | irregular (2:0.09 1:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.04 / 2.62 |

<details><summary>codes and components</summary>

**code 1a-32.0 (L15 L16): 3 comps, tells apart 5/100, coverage 0.08, overlap 1.00 (random 1.00, p 0.71)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 o c3 | H5 | a%100 in {0..1, 13, 43, 83, 99} (-) | 1.00 | 1.000 |
| L16 o c492 | H22 | a%100 in {24} (+) | 1.00 | 0.010 |
| L16 o c197 | H22 | a%100 in {50} (+) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>1a-33</b> `a%100` @ `a` (add) — block code; 44 comps, L15; tells apart 76/100 classes (best member 9); on sub: 1s-109 (member overlap 0.98)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.89 |
| classes told apart: joint / best code / best member | 76 /  / 9 (of 100) |
| members whose removal merges classes | 0.43 |
| support overlap (1 = tiling) / random sets / p | 1.94 / 2.39 / 0.06 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.98 /  / 0.46 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 10 / 3 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 attn |
| CKA(arrangement before, joint write) | 0.69 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.11 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.23 2:0.12 10:0.08) |
| joint write: shape (spectrum k:share) | irregular (1:0.21 2:0.08 3:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.44 / 12.23 |

<details><summary>codes and components</summary>

**code 1a-33.0 (L15): 44 comps, tells apart 76/100, coverage 0.89, overlap 1.94 (random 2.37, p 0.06)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c1 | MLP | a%100 in {0..1, 93} (+) | 1.00 | 1.000 |
| L15 down c17 | MLP | a%100 in {1..2} (+) | 1.00 | 0.020 |
| L15 down c96 | MLP | a%100 in {11} (-) | 1.00 | 0.020 |
| L15 down c45 | MLP | a%100 in {12..15} (+) | 1.00 | 0.090 |
| L15 down c775 | MLP | a%100 in {13, 26} (+) | 1.00 | 0.020 |
| L15 down c188 | MLP | a%100 in {14, 28} (+) | 1.00 | 0.040 |
| L15 down c42 | MLP | a%100 in {16, 32, 36, 56, 64, 96} (-) | 1.00 | 0.150 |
| L15 down c590 | MLP | a%100 in {17..19} (+) | 1.00 | 0.030 |
| L15 down c927 | MLP | a%100 in {18} (+) | 1.00 | 0.010 |
| L15 down c483 | MLP | a%100 in {19..20} (-) | 1.00 | 0.040 |
| L15 down c359 | MLP | a%100 in {1} (+) | 1.00 | 0.020 |
| L15 down c451 | MLP | a%100 in {20..21} (+) | 1.00 | 0.020 |
| L15 down c584 | MLP | a%100 in {21} (-) | 1.00 | 0.010 |
| L15 down c98 | MLP | a%100 in {22..23} (-) | 1.00 | 0.030 |
| L15 down c236 | MLP | a%100 in {24..25} (+) | 1.00 | 0.030 |
| L15 down c772 | MLP | a%100 in {32..33} (-) | 1.00 | 0.020 |
| L15 down c888 | MLP | a%100 in {34} (-) | 1.00 | 0.010 |
| L15 down c743 | MLP | a%100 in {35..39} (+) | 1.00 | 0.060 |
| L15 down c5 | MLP | a%100 in {38, 43, 46..49, 53..54, 56..59, 61..69, 71..74, 76..79, 81..89, 91..98} (+) | 1.00 | 0.459 |
| L15 down c104 | MLP | a%100 in {40, 60, 70, 80} (+) | 1.00 | 0.060 |
| L15 down c426 | MLP | a%100 in {42, 52} (-) | 0.99 | 0.031 |
| L15 down c853 | MLP | a%100 in {42} (+) | 1.00 | 0.010 |
| L15 down c470 | MLP | a%100 in {43..44} (+) | 1.00 | 0.060 |
| L15 down c184 | MLP | a%100 in {45, 90} (-) | 1.00 | 0.030 |
| L15 down c110 | MLP | a%100 in {47..51} (+) | 1.00 | 0.070 |
| L15 down c477 | MLP | a%100 in {50, 52} (+) | 1.00 | 0.030 |
| L15 down c73 | MLP | a%100 in {52..57} (+) | 1.00 | 0.090 |
| L15 down c144 | MLP | a%100 in {57..61} (-) | 1.00 | 0.060 |
| L15 down c894 | MLP | a%100 in {62..64} (-) | 1.00 | 0.030 |
| L15 down c904 | MLP | a%100 in {64..69} (-) | 1.00 | 0.060 |
| L15 down c366 | MLP | a%100 in {69..74} (+) | 1.00 | 0.080 |
| L15 down c814 | MLP | a%100 in {74..76} (+) | 1.00 | 0.040 |
| L15 down c878 | MLP | a%100 in {75} (+) | 1.00 | 0.010 |
| L15 down c233 | MLP | a%100 in {76..80} (-) | 1.00 | 0.100 |
| L15 down c599 | MLP | a%100 in {77..78} (-) | 1.00 | 0.020 |
| L15 down c946 | MLP | a%100 in {80..82} (+) | 1.00 | 0.030 |
| L15 down c854 | MLP | a%100 in {83..86} (-) | 1.00 | 0.060 |
| L15 down c597 | MLP | a%100 in {87..89} (-) | 1.00 | 0.030 |
| L15 down c425 | MLP | a%100 in {88} (-) | 0.56 | 0.006 |
| L15 down c14 | MLP | a%100 in {9, 16..18, 27} (+) | 1.00 | 0.050 |
| L15 down c53 | MLP | a%100 in {90..95} (+) | 1.00 | 0.070 |
| L15 down c364 | MLP | a%100 in {95..99} (-) | 1.00 | 0.050 |
| L15 down c762 | MLP | a%100 in {98..99} (+) | 1.00 | 0.020 |
| L15 down c310 | MLP | a%100 in {9} (+) | 1.00 | 0.020 |

</details>

</details>

<details><summary><b>1a-34</b> `a%100` @ `a` (add) — block code; 2 comps, L16; tells apart 3/100 classes (best member 4); on sub: 1s-110 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.25 |
| classes told apart: joint / best code / best member | 3 /  / 4 (of 100) |
| members whose removal merges classes | 0.50 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.88 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.98 |
| decoding acc. joint / best code / best member (chance) | 0.48 /  / 0.30 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 MLP |
| CKA(arrangement before, joint write) | 0.10 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.23 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (5:0.09 12:0.06 28:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.02 / 1.06 |

<details><summary>codes and components</summary>

**code 1a-34.0 (L16): 2 comps, tells apart 3/100, coverage 0.25, overlap 1.00 (random 1.00, p 0.87)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c17 | H30 | a%100 in {0, 14, 28, 30..31, 35, 40, 42, 49..52, 54, 56..58, 60, 70, 74..75, 90, 92..93, 99} (+) | 0.98 | 0.244 |
| L16 o c207 | H30 | a%100 in {1} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>1a-35</b> `a%100` @ `a` (add) — tiling; 55 comps, L16; tells apart 87/100 classes (best member 6); on sub: 1s-111 (member overlap 0.96)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.95 |
| classes told apart: joint / best code / best member | 87 /  / 6 (of 100) |
| members whose removal merges classes | 0.40 |
| support overlap (1 = tiling) / random sets / p | 2.15 / 2.76 / 0.01 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 1.00 /  / 0.63 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.03 |
| consumers / read jointly | 38 / 16 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L16 attn |
| CKA(arrangement before, joint write) | 0.71 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.06 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.12 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.23 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.25 2:0.11 3:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.60 / 13.22 |

<details><summary>codes and components</summary>

**code 1a-35.0 (L16): 55 comps, tells apart 87/100, coverage 0.95, overlap 2.15 (random 2.77, p 0.03)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c4 | MLP | a%100 in {0..2, 74, 80, 90..94, 97..99} (-) | 1.00 | 1.000 |
| L16 down c38 | MLP | a%100 in {0} (-) | 1.00 | 0.020 |
| L16 down c31 | MLP | a%100 in {1..13, 15..16, 20} (+) | 1.00 | 1.000 |
| L16 down c59 | MLP | a%100 in {1..2} (-) | 1.00 | 0.040 |
| L16 down c1020 | MLP | a%100 in {10} (-) | 1.00 | 0.010 |
| L16 down c20 | MLP | a%100 in {11..21} (-) | 1.00 | 0.120 |
| L16 down c271 | MLP | a%100 in {13, 26..27} (-) | 1.00 | 0.030 |
| L16 down c50 | MLP | a%100 in {16, 21, 31, 41, 46, 51, 61, 71, 76, 81, 86, 91} (+) | 1.00 | 0.131 |
| L16 down c557 | MLP | a%100 in {16, 96} (+) | 1.00 | 0.020 |
| L16 down c449 | MLP | a%100 in {16..18} (-) | 1.00 | 0.030 |
| L16 down c22 | MLP | a%100 in {17, 19..31} (-) | 1.00 | 0.140 |
| L16 down c209 | MLP | a%100 in {18} (+) | 1.00 | 0.040 |
| L16 down c290 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L16 down c282 | MLP | a%100 in {2..3} (-) | 1.00 | 0.020 |
| L16 down c151 | MLP | a%100 in {22, 42, 52} (+) | 1.00 | 0.050 |
| L16 down c524 | MLP | a%100 in {23, 43, 53, 73, 83, 93} (+) | 1.00 | 0.070 |
| L16 down c610 | MLP | a%100 in {23..24} (+) | 1.00 | 0.020 |
| L16 down c30 | MLP | a%100 in {24, 36, 48, 72, 96} (+) | 1.00 | 0.070 |
| L16 down c807 | MLP | a%100 in {25..26} (+) | 1.00 | 0.020 |
| L16 down c814 | MLP | a%100 in {26, 46, 76} (+) | 1.00 | 0.030 |
| L16 down c256 | MLP | a%100 in {27..28} (-) | 1.00 | 0.020 |
| L16 down c336 | MLP | a%100 in {2} (+) | 1.00 | 0.010 |
| L16 down c44 | MLP | a%100 in {30..32} (-) | 1.00 | 0.050 |
| L16 down c554 | MLP | a%100 in {30} (-) | 1.00 | 0.010 |
| L16 down c513 | MLP | a%100 in {32, 92} (-) | 0.99 | 0.035 |
| L16 down c3 | MLP | a%100 in {32..55} (+) | 1.00 | 0.270 |
| L16 down c443 | MLP | a%100 in {32} (+) | 1.00 | 0.010 |
| L16 down c137 | MLP | a%100 in {34..37} (-) | 1.00 | 0.090 |
| L16 down c598 | MLP | a%100 in {35} (+) | 1.00 | 0.010 |
| L16 down c370 | MLP | a%100 in {39..41} (-) | 1.00 | 0.030 |
| L16 down c272 | MLP | a%100 in {3} (-) | 1.00 | 0.030 |
| L16 down c484 | MLP | a%100 in {41} (+) | 1.00 | 0.010 |
| L16 down c570 | MLP | a%100 in {44} (-) | 1.00 | 0.020 |
| L16 down c114 | MLP | a%100 in {5, 10} (+) | 1.00 | 0.020 |
| L16 down c238 | MLP | a%100 in {51} (+) | 1.00 | 0.010 |
| L16 down c929 | MLP | a%100 in {52} (+) | 1.00 | 0.030 |
| L16 down c461 | MLP | a%100 in {55, 59} (+) | 1.00 | 0.050 |
| L16 down c260 | MLP | a%100 in {56..63} (+) | 1.00 | 0.090 |
| L16 down c327 | MLP | a%100 in {60, 80} (-) | 1.00 | 0.040 |
| L16 down c250 | MLP | a%100 in {60} (+) | 1.00 | 0.010 |
| L16 down c665 | MLP | a%100 in {61..62} (+) | 1.00 | 0.020 |
| L16 down c762 | MLP | a%100 in {64, 84} (-) | 1.00 | 0.020 |
| L16 down c354 | MLP | a%100 in {64..65} (+) | 1.00 | 0.040 |
| L16 down c438 | MLP | a%100 in {68..71} (+) | 1.00 | 0.050 |
| L16 down c74 | MLP | a%100 in {7, 47} (+) | 1.00 | 0.100 |
| L16 down c864 | MLP | a%100 in {70..75} (+) | 1.00 | 0.060 |
| L16 down c720 | MLP | a%100 in {74} (-) | 1.00 | 0.010 |
| L16 down c820 | MLP | a%100 in {75} (+) | 1.00 | 0.010 |
| L16 down c211 | MLP | a%100 in {78, 82} (+) | 1.00 | 0.020 |
| L16 down c57 | MLP | a%100 in {78..83} (-) | 1.00 | 0.100 |
| L16 down c406 | MLP | a%100 in {7} (-) | 1.00 | 0.010 |
| L16 down c248 | MLP | a%100 in {86..88, 90} (-) | 1.00 | 0.040 |
| L16 down c692 | MLP | a%100 in {88..89} (+) | 1.00 | 0.020 |
| L16 down c298 | MLP | a%100 in {97..99} (+) | 1.00 | 0.060 |
| L16 down c259 | MLP | a%100 in {99} (+) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>1a-36</b> `a%100` @ `a` (add) — single component; 1 comps, L17; tells apart 4/100 classes (best member 4); on sub: 1s-112 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.10 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.27 /  / 0.27 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L16 MLP |
| CKA(arrangement before, joint write) | 0.16 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.26 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (2:0.24 1:0.09 5:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.05 / 1.00 |

<details><summary>codes and components</summary>

**code 1a-36.0 (L17): 1 comps, tells apart 4/100, coverage 0.10, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 o c17 | H9 | a%100 in {0..1, 5, 7, 10, 44, 50, 74, 88, 99} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>1a-37</b> `a%100` @ `a` (add) — tiling; 66 comps, L17; tells apart 90/100 classes (best member 7); on sub: 1s-113 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.94 |
| classes told apart: joint / best code / best member | 90 /  / 7 (of 100) |
| members whose removal merges classes | 0.41 |
| support overlap (1 = tiling) / random sets / p | 1.76 / 3.23 / 0.00 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 1.00 /  / 0.47 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.04 |
| consumers / read jointly | 12 / 5 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L17 attn |
| CKA(arrangement before, joint write) | 0.65 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.09 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.10 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.26 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.18 2:0.10 3:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.56 / 17.66 |

<details><summary>codes and components</summary>

**code 1a-37.0 (L17): 66 comps, tells apart 90/100, coverage 0.94, overlap 1.76 (random 3.26, p 0.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 down c4 | MLP | a%100 in {0..2, 90, 99} (+) | 1.00 | 1.000 |
| L17 down c75 | MLP | a%100 in {1..2} (-) | 1.00 | 0.030 |
| L17 down c66 | MLP | a%100 in {12} (+) | 1.00 | 0.060 |
| L17 down c165 | MLP | a%100 in {13, 33} (-) | 1.00 | 0.030 |
| L17 down c245 | MLP | a%100 in {14} (-) | 1.00 | 0.020 |
| L17 down c186 | MLP | a%100 in {15} (+) | 1.00 | 0.030 |
| L17 down c497 | MLP | a%100 in {16, 46} (+) | 1.00 | 0.020 |
| L17 down c357 | MLP | a%100 in {18, 38, 78, 88, 98} (+) | 1.00 | 0.060 |
| L17 down c48 | MLP | a%100 in {18..30} (-) | 1.00 | 0.190 |
| L17 down c679 | MLP | a%100 in {18} (-) | 1.00 | 0.030 |
| L17 down c304 | MLP | a%100 in {1} (-) | 1.00 | 0.020 |
| L17 down c411 | MLP | a%100 in {20} (+) | 1.00 | 0.010 |
| L17 down c998 | MLP | a%100 in {20} (-) | 1.00 | 0.010 |
| L17 down c864 | MLP | a%100 in {21} (+) | 1.00 | 0.010 |
| L17 down c565 | MLP | a%100 in {22} (+) | 1.00 | 0.020 |
| L17 down c331 | MLP | a%100 in {24} (+) | 1.00 | 0.010 |
| L17 down c278 | MLP | a%100 in {25..27} (-) | 1.00 | 0.030 |
| L17 down c418 | MLP | a%100 in {28..30} (+) | 1.00 | 0.050 |
| L17 down c664 | MLP | a%100 in {30..31} (-) | 1.00 | 0.020 |
| L17 down c687 | MLP | a%100 in {31..32} (-) | 1.00 | 0.020 |
| L17 down c458 | MLP | a%100 in {32, 52} (-) | 1.00 | 0.040 |
| L17 down c265 | MLP | a%100 in {32..34} (-) | 1.00 | 0.080 |
| L17 down c779 | MLP | a%100 in {33} (-) | 1.00 | 0.010 |
| L17 down c720 | MLP | a%100 in {34} (-) | 1.00 | 0.010 |
| L17 down c474 | MLP | a%100 in {35} (-) | 1.00 | 0.040 |
| L17 down c12 | MLP | a%100 in {36} (-) | 1.00 | 0.010 |
| L17 down c172 | MLP | a%100 in {37..39} (+) | 1.00 | 0.070 |
| L17 down c429 | MLP | a%100 in {38..39} (-) | 1.00 | 0.030 |
| L17 down c383 | MLP | a%100 in {39..40} (-) | 1.00 | 0.020 |
| L17 down c229 | MLP | a%100 in {40..41} (-) | 1.00 | 0.070 |
| L17 down c475 | MLP | a%100 in {41} (+) | 1.00 | 0.010 |
| L17 down c106 | MLP | a%100 in {42..43} (-) | 1.00 | 0.020 |
| L17 down c465 | MLP | a%100 in {44, 55} (+) | 1.00 | 0.020 |
| L17 down c849 | MLP | a%100 in {44..45} (-) | 1.00 | 0.020 |
| L17 down c321 | MLP | a%100 in {45..49} (+) | 1.00 | 0.050 |
| L17 down c823 | MLP | a%100 in {45} (-) | 1.00 | 0.020 |
| L17 down c299 | MLP | a%100 in {46, 65..66} (+) | 0.99 | 0.030 |
| L17 down c834 | MLP | a%100 in {47..48} (+) | 1.00 | 0.020 |
| L17 down c303 | MLP | a%100 in {48, 96} (-) | 1.00 | 0.050 |
| L17 down c356 | MLP | a%100 in {49, 51..52} (-) | 1.00 | 0.030 |
| L17 down c731 | MLP | a%100 in {49} (+) | 1.00 | 0.010 |
| L17 down c167 | MLP | a%100 in {4} (-) | 1.00 | 0.010 |
| L17 down c144 | MLP | a%100 in {50} (+) | 1.00 | 0.050 |
| L17 down c837 | MLP | a%100 in {51, 91} (-) | 1.00 | 0.020 |
| L17 down c270 | MLP | a%100 in {51} (-) | 1.00 | 0.010 |
| L17 down c750 | MLP | a%100 in {52..53} (-) | 1.00 | 0.020 |
| L17 down c249 | MLP | a%100 in {54} (-) | 1.00 | 0.010 |
| L17 down c72 | MLP | a%100 in {55..60, 62..64, 67} (+) | 1.00 | 0.120 |
| L17 down c159 | MLP | a%100 in {55} (-) | 1.00 | 0.010 |
| L17 down c442 | MLP | a%100 in {56..58} (-) | 1.00 | 0.030 |
| L17 down c161 | MLP | a%100 in {5} (-) | 1.00 | 0.030 |
| L17 down c542 | MLP | a%100 in {60, 80} (+) | 1.00 | 0.020 |
| L17 down c212 | MLP | a%100 in {61..62, 82} (-) | 1.00 | 0.040 |
| L17 down c171 | MLP | a%100 in {64..75} (+) | 1.00 | 0.130 |
| L17 down c54 | MLP | a%100 in {6} (+) | 1.00 | 0.100 |
| L17 down c609 | MLP | a%100 in {70..72} (+) | 1.00 | 0.030 |
| L17 down c857 | MLP | a%100 in {74} (+) | 1.00 | 0.010 |
| L17 down c359 | MLP | a%100 in {75..79} (+) | 1.00 | 0.050 |
| L17 down c778 | MLP | a%100 in {76, 86} (+) | 1.00 | 0.020 |
| L17 down c490 | MLP | a%100 in {80} (-) | 1.00 | 0.010 |
| L17 down c138 | MLP | a%100 in {83..88} (-) | 1.00 | 0.070 |
| L17 down c56 | MLP | a%100 in {9..11} (-) | 1.00 | 0.090 |
| L17 down c195 | MLP | a%100 in {91..93} (+) | 1.00 | 0.030 |
| L17 down c415 | MLP | a%100 in {92..95} (+) | 1.00 | 0.040 |
| L17 down c43 | MLP | a%100 in {96..98} (-) | 1.00 | 0.030 |
| L17 down c865 | MLP | a%100 in {98..99} (-) | 1.00 | 0.030 |

</details>

</details>

<details><summary><b>1a-38</b> `a%100` @ `a` (add) — single component; 1 comps, L18; tells apart 2/100 classes (best member 2); on sub: 1s-114 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.21 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.43 /  / 0.43 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L17 MLP |
| CKA(arrangement before, joint write) | 0.21 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (1:0.25 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (2:0.18 50:0.16 1:0.11) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.06 / 1.00 |

<details><summary>codes and components</summary>

**code 1a-38.0 (L18): 1 comps, tells apart 2/100, coverage 0.21, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 o c442 | H1 | a%100 in {0, 2, 4..9, 24, 31, 43, 61, 69, 71, 73, 79, 81, 88, 91, 97, 99} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>1a-39</b> `a%100` @ `a` (add) — block code; 5 comps, L19 L20; tells apart 5/100 classes (best member 2); on sub: 1s-115 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.20 |
| classes told apart: joint / best code / best member | 5 /  / 2 (of 100) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.06 / 0.28 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.57 /  / 0.53 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L18 MLP |
| CKA(arrangement before, joint write) | 0.23 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.25 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.10 20:0.06 40:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.06 / 3.77 |

<details><summary>codes and components</summary>

**code 1a-39.0 (L19 L20): 5 comps, tells apart 5/100, coverage 0.20, overlap 1.00 (random 1.07, p 0.33)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L19 o c23 | H7 | a%100 in {0, 2..3, 5, 7, 30, 40, 46, 50, 53, 56, 61, 63, 66, 90, 99} (-) | 1.00 | 1.000 |
| L20 o c492 | H3 | a%100 in {24} (+) | 1.00 | 0.010 |
| L20 o c290 | H3 | a%100 in {31} (+) | 1.00 | 0.010 |
| L20 o c349 | H3 | a%100 in {45} (+) | 1.00 | 0.010 |
| L20 o c270 | H3 | a%100 in {52} (+) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>1a-40</b> `a%100` @ `a` (add) — block code; 2 comps, L20; tells apart 4/100 classes (best member 4); on sub: 1s-116 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.15 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 100) |
| members whose removal merges classes | 0.50 |
| support overlap (1 = tiling) / random sets / p | 1.13 / 1.00 / 0.95 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.73 /  / 0.73 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L19 MLP |
| CKA(arrangement before, joint write) | 0.30 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.26 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (40:0.19 1:0.15 20:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.05 / 1.09 |

<details><summary>codes and components</summary>

**code 1a-40.0 (L20): 2 comps, tells apart 4/100, coverage 0.15, overlap 1.13 (random 1.00, p 0.96)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 o c0 | H17 | a%100 in {0..2, 5, 7..8, 10, 15, 20, 30, 40, 50, 80, 89, 93} (-) | 1.00 | 0.890 |
| L20 o c8 | H17 | a%100 in {1..2} (-) | 1.00 | 0.020 |

</details>

</details>

<details><summary><b>1a-42</b> `a%100` @ `a` (add) — block code; 2 comps, L21 L22; tells apart 7/100 classes (best member 6); on sub: 1s-117 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.18 |
| classes told apart: joint / best code / best member | 7 /  / 6 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.88 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.43 /  / 0.34 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L20 MLP |
| CKA(arrangement before, joint write) | 0.15 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.27 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.17 2:0.13 3:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.04 / 1.04 |

<details><summary>codes and components</summary>

**code 1a-42.0 (L21 L22): 2 comps, tells apart 7/100, coverage 0.18, overlap 1.00 (random 1.00, p 0.88)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 o c222 | H18 | a%100 in {0, 8..10, 47, 57..58, 64, 90, 99} (-) | 1.00 | 1.000 |
| L22 o c6 | H25 | a%100 in {16..19, 22..24, 31} (+) | 1.00 | 0.090 |

</details>

</details>

<details><summary><b>1a-43</b> `a%100` @ `a` (add) — block code; 3 comps, L22 L23; tells apart 6/100 classes (best member 4); on sub: 1s-118 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.07 |
| classes told apart: joint / best code / best member | 6 /  / 4 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.69 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.08 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 MLP |
| CKA(arrangement before, joint write) | 0.12 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.26 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.07 5:0.06 6:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.02 / 1.34 |

<details><summary>codes and components</summary>

**code 1a-43.0 (L22 L23): 3 comps, tells apart 6/100, coverage 0.07, overlap 1.00 (random 1.00, p 0.68)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 o c10 | H15 | a%100 in {0, 99} (-) | 1.00 | 0.020 |
| L23 o c3 | H2 | a%100 in {74, 90, 92..93} (-) | 1.00 | 0.040 |
| L23 o c280 | H7 | a%100 in {88} (-) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>1a-44</b> `a%100` @ `a` (add) — block code; 25 comps, L23; tells apart 46/100 classes (best member 5); on sub: 1s-119 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.77 |
| classes told apart: joint / best code / best member | 46 /  / 5 (of 100) |
| members whose removal merges classes | 0.72 |
| support overlap (1 = tiling) / random sets / p | 1.48 / 1.68 / 0.13 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.96 /  / 0.60 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.97 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L23 attn |
| CKA(arrangement before, joint write) | 0.54 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.26 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (2:0.11 1:0.10 3:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.30 / 12.16 |

<details><summary>codes and components</summary>

**code 1a-44.0 (L23): 25 comps, tells apart 46/100, coverage 0.77, overlap 1.48 (random 1.66, p 0.16)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 down c67 | MLP | a%100 in {0, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90} (+) | 1.00 | 0.140 |
| L23 down c37 | MLP | a%100 in {1, 31, 41, 51, 61, 81, 91} (-) | 1.00 | 0.080 |
| L23 down c47 | MLP | a%100 in {1..2} (-) | 1.00 | 0.030 |
| L23 down c237 | MLP | a%100 in {11..12} (+) | 1.00 | 0.020 |
| L23 down c6 | MLP | a%100 in {12, 39, 41..43, 46..49, 51..52} (+) | 1.00 | 1.000 |
| L23 down c114 | MLP | a%100 in {13..14, 16..19, 21} (+) | 1.00 | 0.070 |
| L23 down c283 | MLP | a%100 in {16, 32, 64} (+) | 1.00 | 0.030 |
| L23 down c239 | MLP | a%100 in {18..19} (-) | 1.00 | 0.020 |
| L23 down c79 | MLP | a%100 in {20..21} (+) | 1.00 | 0.020 |
| L23 down c148 | MLP | a%100 in {24} (-) | 1.00 | 0.040 |
| L23 down c284 | MLP | a%100 in {26..31} (+) | 1.00 | 0.060 |
| L23 down c975 | MLP | a%100 in {2} (-) | 1.00 | 0.010 |
| L23 down c932 | MLP | a%100 in {30} (-) | 1.00 | 0.010 |
| L23 down c49 | MLP | a%100 in {32..38} (-) | 1.00 | 0.070 |
| L23 down c412 | MLP | a%100 in {36, 48, 52, 72} (-) | 1.00 | 0.040 |
| L23 down c44 | MLP | a%100 in {38..40, 42..44, 46..49, 99} (+) | 1.00 | 0.110 |
| L23 down c493 | MLP | a%100 in {5..12, 15} (+) | 1.00 | 0.090 |
| L23 down c741 | MLP | a%100 in {50, 52} (-) | 1.00 | 0.040 |
| L23 down c40 | MLP | a%100 in {58..63, 65..67} (+) | 1.00 | 0.140 |
| L23 down c81 | MLP | a%100 in {5} (+) | 1.00 | 0.010 |
| L23 down c122 | MLP | a%100 in {88} (+) | 1.00 | 0.020 |
| L23 down c96 | MLP | a%100 in {89} (+) | 0.97 | 0.025 |
| L23 down c292 | MLP | a%100 in {90} (-) | 1.00 | 0.010 |
| L23 down c31 | MLP | a%100 in {92..99} (+) | 1.00 | 0.080 |
| L23 down c554 | MLP | a%100 in {99} (+) | 1.00 | 0.040 |

</details>

</details>

<details><summary><b>1a-45</b> `a%100` @ `a` (add) — block code; 2 comps, L24 L25; tells apart 6/100 classes (best member 2); on sub: 1s-120 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.08 |
| classes told apart: joint / best code / best member | 6 /  / 2 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.85 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 1.00 /  / 0.54 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L23 MLP |
| CKA(arrangement before, joint write) | 0.30 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.27 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.34 20:0.11 40:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.10 / 1.35 |

<details><summary>codes and components</summary>

**code 1a-45.0 (L24 L25): 2 comps, tells apart 6/100, coverage 0.08, overlap 1.00 (random 1.00, p 0.86)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 o c39 | H14 | a%100 in {0, 5, 9..10, 53, 56, 99} (-) | 1.00 | 1.000 |
| L25 o c394 | H23 | a%100 in {1} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>1a-46</b> `a%100` @ `a` (add) — single component; 1 comps, L26; tells apart 1/100 classes (best member 1); on sub: 1s-121 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.16 |
| classes told apart: joint / best code / best member | 1 /  / 1 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.58 /  / 0.58 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L25 MLP |
| CKA(arrangement before, joint write) | 0.46 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.21 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.27 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.36 40:0.19 2:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.09 / 1.00 |

<details><summary>codes and components</summary>

**code 1a-46.0 (L26): 1 comps, tells apart 1/100, coverage 0.16, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 o c19 | H10 | a%100 in {0, 2..8, 10, 12, 15, 20, 41, 50, 54, 81} (-) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>1a-47</b> `a%100` @ `a` (add) — single component; 1 comps, L27; tells apart 3/100 classes (best member 3); on sub: 1s-122 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.02 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.46 /  / 0.46 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L26 MLP |
| CKA(arrangement before, joint write) | 0.07 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.18 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.27 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (10:0.13 25:0.08 12:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.03 / 1.00 |

<details><summary>codes and components</summary>

**code 1a-47.0 (L27): 1 comps, tells apart 3/100, coverage 0.02, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 o c5 | H14 | a%100 in {0, 99} (+) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>1a-48</b> `a%100` @ `a` (add) — block code; 2 comps, L28 L29; tells apart 14/100 classes (best member 4); on sub: 1s-123 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.24 |
| classes told apart: joint / best code / best member | 14 /  / 4 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.08 / 1.00 / 0.92 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.95 /  / 0.44 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.05 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L27 MLP |
| CKA(arrangement before, joint write) | 0.36 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.18 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.27 2:0.12 3:0.06) |
| joint write: shape (spectrum k:share) | line (2:0.21 1:0.17 40:0.15) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.07 / 1.34 |

<details><summary>codes and components</summary>

**code 1a-48.0 (L28 L29): 2 comps, tells apart 14/100, coverage 0.24, overlap 1.08 (random 1.00, p 0.91)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 o c122 | H10 | a%100 in {0, 4, 9, 11..12, 14, 17, 19, 22..23, 26, 28..29, 50, 89..90} (+) | 1.00 | 1.000 |
| L29 o c1 | H25 | a%100 in {0, 2, 50, 52, 60, 70, 74, 76, 78, 86} (+) | 1.00 | 1.000 |

</details>

</details>

<details><summary><b>1a-49</b> `a%100` @ `a` (add) — tiling; 27 comps, L29; tells apart 42/100 classes (best member 7); on sub: 1s-124 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.78 |
| classes told apart: joint / best code / best member | 42 /  / 7 (of 100) |
| members whose removal merges classes | 0.48 |
| support overlap (1 = tiling) / random sets / p | 1.41 / 1.75 / 0.03 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 1.00 /  / 0.52 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 3 / 3 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L29 attn |
| CKA(arrangement before, joint write) | 0.78 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.05 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.27 2:0.12 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.37 2:0.12 3:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.44 / 5.48 |

<details><summary>codes and components</summary>

**code 1a-49.0 (L29): 27 comps, tells apart 42/100, coverage 0.78, overlap 1.41 (random 1.72, p 0.04)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L29 down c3 | MLP | a%100 in {0..3, 5, 10} (-) | 1.00 | 1.000 |
| L29 down c9 | MLP | a%100 in {0..3} (+) | 1.00 | 1.000 |
| L29 down c133 | MLP | a%100 in {0} (-) | 1.00 | 0.020 |
| L29 down c143 | MLP | a%100 in {1..2} (+) | 1.00 | 0.030 |
| L29 down c111 | MLP | a%100 in {10, 12} (-) | 1.00 | 0.020 |
| L29 down c123 | MLP | a%100 in {11..31} (+) | 1.00 | 0.230 |
| L29 down c574 | MLP | a%100 in {12, 36, 48} (+) | 1.00 | 0.030 |
| L29 down c30 | MLP | a%100 in {14, 18} (-) | 1.00 | 0.020 |
| L29 down c2 | MLP | a%100 in {1} (-) | 1.00 | 0.010 |
| L29 down c228 | MLP | a%100 in {2..4} (-) | 1.00 | 0.040 |
| L29 down c203 | MLP | a%100 in {24, 48} (-) | 1.00 | 0.040 |
| L29 down c212 | MLP | a%100 in {32, 64} (+) | 1.00 | 0.100 |
| L29 down c63 | MLP | a%100 in {32} (-) | 1.00 | 0.040 |
| L29 down c421 | MLP | a%100 in {34..39} (+) | 1.00 | 0.060 |
| L29 down c214 | MLP | a%100 in {36, 48, 72, 96} (-) | 1.00 | 0.040 |
| L29 down c1005 | MLP | a%100 in {40} (-) | 1.00 | 0.040 |
| L29 down c576 | MLP | a%100 in {42..43, 46..47, 49} (+) | 1.00 | 0.120 |
| L29 down c920 | MLP | a%100 in {45} (+) | 1.00 | 0.010 |
| L29 down c829 | MLP | a%100 in {5..10} (+) | 1.00 | 0.070 |
| L29 down c22 | MLP | a%100 in {60, 80} (+) | 1.00 | 0.020 |
| L29 down c81 | MLP | a%100 in {62, 67, 73..74, 76..79, 81..89, 91..98} (-) | 1.00 | 0.440 |
| L29 down c108 | MLP | a%100 in {74} (-) | 1.00 | 0.010 |
| L29 down c838 | MLP | a%100 in {8..9, 11} (-) | 1.00 | 0.040 |
| L29 down c457 | MLP | a%100 in {88, 99} (+) | 1.00 | 0.040 |
| L29 down c603 | MLP | a%100 in {90} (-) | 1.00 | 0.010 |
| L29 down c242 | MLP | a%100 in {92..93} (-) | 1.00 | 0.100 |
| L29 down c991 | MLP | a%100 in {99} (-) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>1a-50</b> `a%100` @ `a` (add) — single component; 1 comps, L30; tells apart 1/100 classes (best member 1); on sub: 1s-125 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.17 |
| classes told apart: joint / best code / best member | 1 /  / 1 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.34 /  / 0.34 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L29 MLP |
| CKA(arrangement before, joint write) | 0.15 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.27 2:0.12 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.47 20:0.12 4:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.12 / 1.00 |

<details><summary>codes and components</summary>

**code 1a-50.0 (L30): 1 comps, tells apart 1/100, coverage 0.17, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 o c446 | H18 | a%100 in {0..3, 7, 12, 23, 31, 55..56, 58, 61, 88, 93, 95, 97, 99} (+) | 1.00 | 1.000 |

</details>

</details>

</details>

<details><summary>`a//10`: 10 mechanisms, 100 components</summary>

<details><summary><b>1a-65</b> `a//10` @ `a` (add) — block code, 10 codes of the same shape; 40 comps, L0 L6 L14 L20 L22 L23 L25 L27 L28 L30; tells apart 6/11 classes (best member 6); on sub: 1s-140 (member overlap 0.82)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 6 / 10 / 6 (of 11) |
| members whose removal merges classes | 0.03 |
| support overlap (1 = tiling) / random sets / p | 7.82 / 6.82 / 0.92 |
| mean CKA between its codes | 0.88 |
| purity of the joint write (per prompt) | 0.94 |
| decoding acc. joint / best code / best member (chance) | 0.87 / 0.93 / 0.67 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.21 |
| consumers / read jointly | 195 / 170 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 attn |
| CKA(arrangement before, joint write) | 0.67 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.00 |
| write energy / code energy before | 729.66 |

<details><summary>codes and components</summary>

**code 1a-65.0 (L0): 6 comps, tells apart 10/11, coverage 0.82, overlap 1.89 (random 1.75, p 0.64)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 down c5 | MLP | a//10 in {0..1, 3..5, 9..10} (+) | 0.94 | 0.840 |
| L0 down c23 | MLP | a//10 in {0..2} (+) | 0.96 | 0.390 |
| L0 down c28 | MLP | a//10 in {0..2} (-) | 0.95 | 0.320 |
| L0 down c405 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L0 down c123 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L0 down c6 | MLP | a//10 in {8..9} (-) | 0.97 | 0.500 |

**code 1a-65.1 (L6): 2 comps, tells apart 4/11, coverage 0.36, overlap 1.00 (random 1.00, p 0.63)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L6 down c25 | MLP | a//10 in {0..2} (-) | 0.91 | 0.300 |
| L6 down c133 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |

**code 1a-65.2 (L14): 3 comps, tells apart 6/11, coverage 0.55, overlap 1.67 (random 1.25, p 0.90)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L14 down c0 | MLP | a//10 in {0..2, 6..7, 10} (+) | 0.92 | 1.000 |
| L14 down c8 | MLP | a//10 in {0..2} (-) | 0.97 | 0.461 |
| L14 down c255 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |

**code 1a-65.3 (L20): 4 comps, tells apart 8/11, coverage 0.73, overlap 1.50 (random 1.40, p 0.65)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 down c7 | MLP | a//10 in {0..2, 7..8, 10} (-) | 0.94 | 0.860 |
| L20 down c23 | MLP | a//10 in {0..3} (-) | 0.97 | 0.500 |
| L20 down c248 | MLP | a//10 in {10} (+) | 0.96 | 0.020 |
| L20 down c33 | MLP | a//10 in {9} (-) | 0.95 | 0.222 |

**code 1a-65.4 (L22): 8 comps, tells apart 7/11, coverage 0.64, overlap 1.71 (random 2.00, p 0.17)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 down c14 | MLP | a//10 in {0..2} (-) | 0.92 | 0.320 |
| L22 down c37 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L22 down c96 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L22 down c268 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L22 down c7 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L22 down c71 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L22 down c757 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L22 down c416 | MLP | a//10 in {7..9} (-) | 0.92 | 0.300 |

**code 1a-65.5 (L23): 6 comps, tells apart 7/11, coverage 0.55, overlap 1.67 (random 1.75, p 0.41)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 down c1 | MLP | a//10 in {0..2, 10} (+) | 0.91 | 0.690 |
| L23 down c150 | MLP | a//10 in {0} (+) | 0.91 | 0.140 |
| L23 down c22 | MLP | a//10 in {10} (-) | 0.91 | 0.990 |
| L23 down c126 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L23 down c322 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L23 down c533 | MLP | a//10 in {8..9} (+) | 0.93 | 0.190 |

**code 1a-65.6 (L25): 4 comps, tells apart 6/11, coverage 0.55, overlap 1.33 (random 1.50, p 0.41)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L25 down c338 | MLP | a//10 in {0..2} (-) | 0.95 | 0.330 |
| L25 down c638 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L25 down c87 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L25 down c35 | MLP | a//10 in {8..10} (-) | 0.99 | 0.310 |

**code 1a-65.7 (L27): 2 comps, tells apart 5/11, coverage 0.36, overlap 1.00 (random 1.00, p 0.62)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 down c22 | MLP | a//10 in {0..2} (-) | 0.91 | 0.340 |
| L27 down c336 | MLP | a//10 in {9} (-) | 0.97 | 0.100 |

**code 1a-65.8 (L28): 1 comps, tells apart 4/11, coverage 0.27, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 down c589 | MLP | a//10 in {0..2} (+) | 0.93 | 0.380 |

**code 1a-65.9 (L30): 4 comps, tells apart 5/11, coverage 0.36, overlap 1.50 (random 1.41, p 0.62)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 down c63 | MLP | a//10 in {0..2} (+) | 0.95 | 0.520 |
| L30 down c15 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L30 down c405 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L30 down c925 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>1a-66</b> `a//10` @ `a` (add) — block code, 8 codes of the same shape; 28 comps, L1 L5 L15 L16 L17 L18 L24 L29; tells apart 7/11 classes (best member 4); on sub: 1s-144 (member overlap 0.71)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.82 |
| classes told apart: joint / best code / best member | 7 / 8 / 4 (of 11) |
| members whose removal merges classes | 0.11 |
| support overlap (1 = tiling) / random sets / p | 4.67 / 5.00 / 0.34 |
| mean CKA between its codes | 0.78 |
| purity of the joint write (per prompt) | 0.94 |
| decoding acc. joint / best code / best member (chance) | 0.76 / 0.75 / 0.55 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.10 |
| consumers / read jointly | 131 / 123 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L1 attn |
| CKA(arrangement before, joint write) | 0.70 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 29.38 |

<details><summary>codes and components</summary>

**code 1a-66.0 (L1): 4 comps, tells apart 7/11, coverage 0.55, overlap 1.17 (random 1.40, p 0.15)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 down c96 | MLP | a//10 in {0..1} (-) | 0.96 | 0.270 |
| L1 down c17 | MLP | a//10 in {1..3} (-) | 0.92 | 1.000 |
| L1 down c525 | MLP | a//10 in {4} (-) | 0.91 | 0.100 |
| L1 down c271 | MLP | a//10 in {8} (+) | 0.92 | 0.123 |

**code 1a-66.1 (L5): 4 comps, tells apart 5/11, coverage 0.55, overlap 1.17 (random 1.50, p 0.14)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 down c1 | MLP | a//10 in {0..1, 7..8} (+) | 0.91 | 1.000 |
| L5 down c124 | MLP | a//10 in {0} (+) | 0.91 | 0.120 |
| L5 down c936 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L5 down c387 | MLP | a//10 in {9} (+) | 0.98 | 0.100 |

**code 1a-66.2 (L15): 5 comps, tells apart 5/11, coverage 0.27, overlap 2.00 (random 1.60, p 0.88)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c36 | MLP | a//10 in {0} (+) | 0.92 | 0.090 |
| L15 down c48 | MLP | a//10 in {0} (+) | 0.97 | 0.130 |
| L15 down c398 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L15 down c193 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L15 down c31 | MLP | a//10 in {9..10} (-) | 0.91 | 0.160 |

**code 1a-66.3 (L16): 4 comps, tells apart 8/11, coverage 0.36, overlap 2.00 (random 1.43, p 0.93)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c0 | MLP | a//10 in {0..2} (-) | 0.95 | 1.000 |
| L16 down c5 | MLP | a//10 in {0..2} (-) | 0.94 | 0.690 |
| L16 down c78 | MLP | a//10 in {0} (-) | 0.94 | 0.150 |
| L16 down c9 | MLP | a//10 in {9} (-) | 0.95 | 0.200 |

**code 1a-66.4 (L17): 3 comps, tells apart 5/11, coverage 0.45, overlap 1.00 (random 1.33, p 0.21)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 down c58 | MLP | a//10 in {0..1} (-) | 0.97 | 0.210 |
| L17 down c110 | MLP | a//10 in {10} (+) | 0.95 | 0.030 |
| L17 down c85 | MLP | a//10 in {8..9} (-) | 0.90 | 0.240 |

**code 1a-66.5 (L18): 2 comps, tells apart 4/11, coverage 0.18, overlap 1.00 (random 1.00, p 0.61)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 down c55 | MLP | a//10 in {0} (+) | 0.94 | 0.160 |
| L18 down c460 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |

**code 1a-66.6 (L24): 3 comps, tells apart 6/11, coverage 0.27, overlap 1.00 (random 1.33, p 0.23)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 down c222 | MLP | a//10 in {0} (+) | 0.90 | 0.170 |
| L24 down c21 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L24 down c115 | MLP | a//10 in {9} (+) | 0.90 | 0.170 |

**code 1a-66.7 (L29): 3 comps, tells apart 5/11, coverage 0.27, overlap 1.33 (random 1.33, p 0.64)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L29 down c114 | MLP | a//10 in {0..1} (-) | 0.93 | 0.240 |
| L29 down c332 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L29 down c465 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>1a-67</b> `a//10` @ `a` (add) — tiling; 4 comps, L2; tells apart 7/11 classes (best member 4); on sub: 1s-141 (member overlap 0.75)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.55 |
| classes told apart: joint / best code / best member | 7 /  / 4 (of 11) |
| members whose removal merges classes | 0.75 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.50 / 0.07 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.94 |
| decoding acc. joint / best code / best member (chance) | 0.66 /  / 0.40 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 0.99 |
| consumers / read jointly | 8 / 1 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 attn |
| CKA(arrangement before, joint write) | 0.62 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.16 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.12 |

<details><summary>codes and components</summary>

**code 1a-67.0 (L2): 4 comps, tells apart 7/11, coverage 0.55, overlap 1.00 (random 1.50, p 0.04)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 down c729 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L2 down c11 | MLP | a//10 in {3..4} (-) | 0.92 | 0.230 |
| L2 down c271 | MLP | a//10 in {5} (-) | 0.91 | 0.120 |
| L2 down c92 | MLP | a//10 in {8..9} (-) | 0.97 | 0.310 |

</details>

</details>

<details><summary><b>1a-68</b> `a//10` @ `a` (add) — 2 codes of the same shape; 2 comps, L3 L4; tells apart 2/11 classes (best member 2); on sub: 1s-142 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.09 |
| classes told apart: joint / best code / best member | 2 / 2 / 2 (of 11) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 2.00 / 1.00 / 1.00 |
| mean CKA between its codes | 1.00 |
| purity of the joint write (per prompt) | 1.00 |
| decoding acc. joint / best code / best member (chance) | 0.18 / 0.18 / 0.18 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 MLP |
| CKA(arrangement before, joint write) | 0.14 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 0.00 |

<details><summary>codes and components</summary>

**code 1a-68.0 (L3): 1 comps, tells apart 2/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 o c104 | H7 | a//10 in {10} (-) | 1.00 | 0.010 |

**code 1a-68.1 (L4): 1 comps, tells apart 2/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 down c808 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |

</details>

</details>

<details><summary><b>1a-69</b> `a//10` @ `a` (add) — block code, 2 codes of the same shape; 8 comps, L3 L19; tells apart 9/11 classes (best member 5); on sub: 1s-143 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.82 |
| classes told apart: joint / best code / best member | 9 / 7 / 5 (of 11) |
| members whose removal merges classes | 0.38 |
| support overlap (1 = tiling) / random sets / p | 2.22 / 2.00 / 0.67 |
| mean CKA between its codes | 0.78 |
| purity of the joint write (per prompt) | 0.96 |
| decoding acc. joint / best code / best member (chance) | 0.87 / 0.74 / 0.58 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.07 |
| consumers / read jointly | 13 / 5 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L3 attn |
| CKA(arrangement before, joint write) | 0.78 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 1.48 |

<details><summary>codes and components</summary>

**code 1a-69.0 (L3): 5 comps, tells apart 7/11, coverage 0.82, overlap 1.22 (random 1.67, p 0.07)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 down c11 | MLP | a//10 in {0..3, 7..9} (-) | 0.96 | 0.530 |
| L3 down c100 | MLP | a//10 in {0} (+) | 0.91 | 0.090 |
| L3 down c126 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L3 down c29 | MLP | a//10 in {6} (-) | 0.90 | 0.121 |
| L3 down c121 | MLP | a//10 in {8} (+) | 0.92 | 0.150 |

**code 1a-69.1 (L19): 3 comps, tells apart 7/11, coverage 0.73, overlap 1.12 (random 1.29, p 0.28)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L19 down c9 | MLP | a//10 in {0..3} (-) | 0.95 | 1.000 |
| L19 down c700 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L19 down c10 | MLP | a//10 in {7..10} (+) | 0.97 | 0.370 |

</details>

</details>

<details><summary><b>1a-70</b> `a//10` @ `a` (add) — block code; 3 comps, L11; tells apart 5/11 classes (best member 3); on sub: 1s-145 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.36 |
| classes told apart: joint / best code / best member | 5 /  / 3 (of 11) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.27 / 0.22 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.93 |
| decoding acc. joint / best code / best member (chance) | 0.47 /  / 0.28 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 0.98 |
| consumers / read jointly | 9 / 1 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L11 attn |
| CKA(arrangement before, joint write) | 0.45 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.11 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.06 |

<details><summary>codes and components</summary>

**code 1a-70.0 (L11): 3 comps, tells apart 5/11, coverage 0.36, overlap 1.00 (random 1.29, p 0.24)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L11 down c10 | MLP | a//10 in {6} (-) | 0.92 | 0.140 |
| L11 down c30 | MLP | a//10 in {7} (-) | 0.92 | 0.120 |
| L11 down c22 | MLP | a//10 in {9..10} (-) | 0.95 | 0.140 |

</details>

</details>

<details><summary><b>1a-71</b> `a//10` @ `a` (add) — tiling; 5 comps, L12; tells apart 7/11 classes (best member 3); on sub: 1s-146 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.45 |
| classes told apart: joint / best code / best member | 7 /  / 3 (of 11) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.60 / 0.01 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.93 |
| decoding acc. joint / best code / best member (chance) | 0.63 /  / 0.23 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 0.98 |
| consumers / read jointly | 13 / 7 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L12 attn |
| CKA(arrangement before, joint write) | 0.61 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.06 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.07 |

<details><summary>codes and components</summary>

**code 1a-71.0 (L12): 5 comps, tells apart 7/11, coverage 0.45, overlap 1.00 (random 1.67, p 0.01)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L12 down c633 | MLP | a//10 in {0} (+) | 0.91 | 0.150 |
| L12 down c154 | MLP | a//10 in {10} (-) | 1.00 | 0.010 |
| L12 down c26 | MLP | a//10 in {3} (-) | 0.94 | 0.130 |
| L12 down c13 | MLP | a//10 in {4} (+) | 0.93 | 0.150 |
| L12 down c59 | MLP | a//10 in {6} (-) | 0.96 | 0.110 |

</details>

</details>

<details><summary><b>1a-72</b> `a//10` @ `a` (add) — block code; 3 comps, L13; tells apart 5/11 classes (best member 4); on sub: 1s-147 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.18 |
| classes told apart: joint / best code / best member | 5 /  / 4 (of 11) |
| members whose removal merges classes | 0.33 |
| support overlap (1 = tiling) / random sets / p | 1.50 / 1.33 / 0.84 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.93 |
| decoding acc. joint / best code / best member (chance) | 0.35 /  / 0.25 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 16 / 1 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L13 attn |
| CKA(arrangement before, joint write) | 0.34 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.07 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.04 |

<details><summary>codes and components</summary>

**code 1a-72.0 (L13): 3 comps, tells apart 5/11, coverage 0.18, overlap 1.50 (random 1.25, p 0.86)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 down c733 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L13 down c161 | MLP | a//10 in {10} (-) | 0.94 | 0.020 |
| L13 down c10 | MLP | a//10 in {8} (+) | 0.92 | 0.240 |

</details>

</details>

<details><summary><b>1a-73</b> `a//10` @ `a` (add) — block code; 2 comps, L21; tells apart 3/11 classes (best member 2); on sub: 1s-148 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.18 |
| classes told apart: joint / best code / best member | 3 /  / 2 (of 11) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.64 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.96 |
| decoding acc. joint / best code / best member (chance) | 0.27 /  / 0.18 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 attn |
| CKA(arrangement before, joint write) | 0.34 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.01 |

<details><summary>codes and components</summary>

**code 1a-73.0 (L21): 2 comps, tells apart 3/11, coverage 0.18, overlap 1.00 (random 1.00, p 0.62)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c188 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L21 down c141 | MLP | a//10 in {9} (-) | 0.96 | 0.110 |

</details>

</details>

<details><summary><b>1a-74</b> `a//10` @ `a` (add) — block code; 5 comps, L26; tells apart 6/11 classes (best member 4); on sub: 1s-149 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.45 |
| classes told apart: joint / best code / best member | 6 /  / 4 (of 11) |
| members whose removal merges classes | 0.40 |
| support overlap (1 = tiling) / random sets / p | 1.40 / 1.60 / 0.28 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.92 |
| decoding acc. joint / best code / best member (chance) | 0.50 /  / 0.27 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.04 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L26 attn |
| CKA(arrangement before, joint write) | 0.65 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 0.04 |

<details><summary>codes and components</summary>

**code 1a-74.0 (L26): 5 comps, tells apart 6/11, coverage 0.45, overlap 1.40 (random 1.57, p 0.34)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 down c16 | MLP | a//10 in {1..2} (+) | 0.91 | 0.240 |
| L26 down c69 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L26 down c286 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L26 down c766 | MLP | a//10 in {10} (+) | 1.00 | 0.010 |
| L26 down c562 | MLP | a//10 in {8..9} (+) | 0.91 | 0.250 |

</details>

</details>

</details>

<details><summary>`a%10`: 9 mechanisms, 48 components</summary>

<details><summary><b>1a-0</b> `a%10` @ `a` (add) — tiling, 3 codes of the same shape; 19 comps, L0 L3 L12; tells apart 10/10 classes (best member 5); on sub: 1s-75 (member overlap 0.58)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 10 / 10 / 5 (of 10) |
| members whose removal merges classes | 0.05 |
| support overlap (1 = tiling) / random sets / p | 2.70 / 2.56 / 0.68 |
| mean CKA between its codes | 0.74 |
| purity of the joint write (per prompt) | 0.98 |
| decoding acc. joint / best code / best member (chance) | 1.00 / 1.00 / 0.40 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.21 |
| consumers / read jointly | 163 / 136 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 attn |
| CKA(arrangement before, joint write) | 0.77 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.19 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 21.76 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (20:0.27 10:0.26 40:0.19) |
| joint write: shape (spectrum k:share) | irregular (10:0.33 20:0.30 40:0.14) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.58 / 4.58 |

<details><summary>codes and components</summary>

**code 1a-0.0 (L0): 10 comps, tells apart 10/10, coverage 1.00, overlap 1.00 (random 1.67, p 0.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 down c145 | MLP | a%10 in {0} (+) | 0.95 | 0.100 |
| L0 down c36 | MLP | a%10 in {1} (+) | 0.97 | 0.100 |
| L0 down c81 | MLP | a%10 in {2} (-) | 0.96 | 0.100 |
| L0 down c52 | MLP | a%10 in {3} (+) | 0.97 | 0.100 |
| L0 down c62 | MLP | a%10 in {4} (+) | 0.98 | 0.100 |
| L0 down c45 | MLP | a%10 in {5} (+) | 0.96 | 0.100 |
| L0 down c55 | MLP | a%10 in {6} (-) | 0.98 | 0.100 |
| L0 down c44 | MLP | a%10 in {7} (-) | 0.98 | 0.100 |
| L0 down c50 | MLP | a%10 in {8} (-) | 0.99 | 0.100 |
| L0 down c38 | MLP | a%10 in {9} (-) | 0.98 | 0.100 |

**code 1a-0.1 (L3): 5 comps, tells apart 10/10, coverage 0.90, overlap 1.33 (random 1.25, p 0.70)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 down c615 | MLP | a%10 in {0} (+) | 0.94 | 0.190 |
| L3 down c54 | MLP | a%10 in {1, 9} (+) | 0.95 | 0.200 |
| L3 down c68 | MLP | a%10 in {3..4, 9} (-) | 0.93 | 0.380 |
| L3 down c63 | MLP | a%10 in {5..7} (+) | 0.98 | 0.300 |
| L3 down c136 | MLP | a%10 in {7..9} (-) | 0.95 | 0.370 |

**code 1a-0.2 (L12): 4 comps, tells apart 6/10, coverage 0.50, overlap 1.00 (random 1.20, p 0.39)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L12 down c19 | MLP | a%10 in {0} (+) | 0.99 | 0.100 |
| L12 down c15 | MLP | a%10 in {1} (+) | 0.97 | 0.100 |
| L12 down c31 | MLP | a%10 in {6..7} (-) | 0.95 | 0.296 |
| L12 down c7 | MLP | a%10 in {9} (-) | 0.99 | 0.100 |

</details>

</details>

<details><summary><b>1a-1</b> `a%10` @ `a` (add) — single component; 1 comps, L1; tells apart 2/10 classes (best member 2); on sub: 1s-76 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.10 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 10) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.95 |
| decoding acc. joint / best code / best member (chance) | 0.20 /  / 0.20 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L1 attn |
| CKA(arrangement before, joint write) | 0.26 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.08 |
| share of the write inside the old arrangement's span | 0.09 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (10:0.30 20:0.30 40:0.16) |
| joint write: shape (spectrum k:share) | line (30:0.22 20:0.22 40:0.22) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.11 / 1.00 |

<details><summary>codes and components</summary>

**code 1a-1.0 (L1): 1 comps, tells apart 2/10, coverage 0.10, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 down c401 | MLP | a%10 in {7} (+) | 0.95 | 0.100 |

</details>

</details>

<details><summary><b>1a-2</b> `a%10` @ `a` (add) — block code; 2 comps, L4; tells apart 5/10 classes (best member 4); on sub: 1s-77 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.60 |
| classes told apart: joint / best code / best member | 5 /  / 4 (of 10) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.88 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.93 |
| decoding acc. joint / best code / best member (chance) | 0.45 /  / 0.30 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 0.91 |
| consumers / read jointly | 4 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L4 attn |
| CKA(arrangement before, joint write) | 0.69 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.18 |
| share of the write inside the old arrangement's span | 0.11 |
| write energy / code energy before | 0.09 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (20:0.30 10:0.30 40:0.16) |
| joint write: shape (spectrum k:share) | irregular (10:0.43 20:0.29 40:0.26) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.33 / 1.94 |

<details><summary>codes and components</summary>

**code 1a-2.0 (L4): 2 comps, tells apart 5/10, coverage 0.60, overlap 1.00 (random 1.00, p 0.82)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 down c43 | MLP | a%10 in {0, 5} (-) | 0.90 | 0.210 |
| L4 down c66 | MLP | a%10 in {6..9} (-) | 0.96 | 0.390 |

</details>

</details>

<details><summary><b>1a-3</b> `a%10` @ `a` (add) — block code; 5 comps, L11; tells apart 6/10 classes (best member 2); on sub: 1s-78 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.50 |
| classes told apart: joint / best code / best member | 6 /  / 2 (of 10) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.25 / 0.17 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.96 |
| decoding acc. joint / best code / best member (chance) | 0.61 /  / 0.22 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 0.94 |
| consumers / read jointly | 3 / 1 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L11 attn |
| CKA(arrangement before, joint write) | 0.69 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.12 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.14 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (10:0.30 20:0.30 50:0.19) |
| joint write: shape (spectrum k:share) | irregular (10:0.27 20:0.24 40:0.21) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.51 / 4.52 |

<details><summary>codes and components</summary>

**code 1a-3.0 (L11): 5 comps, tells apart 6/10, coverage 0.50, overlap 1.00 (random 1.25, p 0.22)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L11 down c42 | MLP | a%10 in {0} (+) | 0.96 | 0.100 |
| L11 down c26 | MLP | a%10 in {4} (-) | 0.93 | 0.130 |
| L11 down c27 | MLP | a%10 in {5} (-) | 0.98 | 0.110 |
| L11 down c63 | MLP | a%10 in {6} (-) | 0.98 | 0.100 |
| L11 down c36 | MLP | a%10 in {7} (-) | 0.98 | 0.100 |

</details>

</details>

<details><summary><b>1a-4</b> `a%10` @ `a` (add) — tiling; 8 comps, L13; tells apart 9/10 classes (best member 3); on sub: 1s-80 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.90 |
| classes told apart: joint / best code / best member | 9 /  / 3 (of 10) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.50 / 0.00 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.99 |
| decoding acc. joint / best code / best member (chance) | 0.97 /  / 0.27 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 0.93 |
| consumers / read jointly | 95 / 15 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L13 attn |
| CKA(arrangement before, joint write) | 0.55 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.20 |
| share of the write inside the old arrangement's span | 0.09 |
| write energy / code energy before | 0.29 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (10:0.34 20:0.28 50:0.16) |
| joint write: shape (spectrum k:share) | irregular (40:0.24 10:0.24 30:0.24) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.63 / 5.60 |

<details><summary>codes and components</summary>

**code 1a-4.0 (L13): 8 comps, tells apart 9/10, coverage 0.90, overlap 1.00 (random 1.50, p 0.01)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 down c38 | MLP | a%10 in {0, 9} (+) | 0.99 | 0.200 |
| L13 down c105 | MLP | a%10 in {1} (+) | 0.96 | 0.100 |
| L13 down c92 | MLP | a%10 in {2} (-) | 0.99 | 0.100 |
| L13 down c15 | MLP | a%10 in {3} (+) | 0.99 | 0.100 |
| L13 down c51 | MLP | a%10 in {4} (-) | 0.96 | 0.100 |
| L13 down c13 | MLP | a%10 in {5} (+) | 0.99 | 0.100 |
| L13 down c9 | MLP | a%10 in {7} (-) | 1.00 | 0.100 |
| L13 down c11 | MLP | a%10 in {8} (+) | 0.99 | 0.100 |

</details>

</details>

<details><summary><b>1a-5</b> `a%10` @ `a` (add) — tiling; 7 comps, L14; tells apart 8/10 classes (best member 3); on sub: 1s-75 (member overlap 0.32)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.70 |
| classes told apart: joint / best code / best member | 8 /  / 3 (of 10) |
| members whose removal merges classes | 0.86 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.43 / 0.02 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.97 |
| decoding acc. joint / best code / best member (chance) | 0.80 /  / 0.29 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 0.96 |
| consumers / read jointly | 67 / 14 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L14 attn |
| CKA(arrangement before, joint write) | 0.70 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.17 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.18 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (10:0.32 20:0.28 40:0.16) |
| joint write: shape (spectrum k:share) | irregular (40:0.27 20:0.23 10:0.20) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.65 / 5.80 |

<details><summary>codes and components</summary>

**code 1a-5.0 (L14): 7 comps, tells apart 8/10, coverage 0.70, overlap 1.00 (random 1.40, p 0.03)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L14 down c18 | MLP | a%10 in {0} (+) | 0.91 | 0.200 |
| L14 down c71 | MLP | a%10 in {1} (-) | 0.94 | 0.100 |
| L14 down c15 | MLP | a%10 in {2} (-) | 0.99 | 0.100 |
| L14 down c14 | MLP | a%10 in {4} (-) | 0.99 | 0.100 |
| L14 down c226 | MLP | a%10 in {5} (+) | 0.98 | 0.100 |
| L14 down c33 | MLP | a%10 in {6} (-) | 0.98 | 0.100 |
| L14 down c21 | MLP | a%10 in {9} (+) | 0.99 | 0.100 |

</details>

</details>

<details><summary><b>1a-6</b> `a%10` @ `a` (add) — block code; 2 comps, L16 L17; tells apart 3/10 classes (best member 2); on sub: 1s-81 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.20 |
| classes told apart: joint / best code / best member | 3 /  / 2 (of 10) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.88 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.92 |
| decoding acc. joint / best code / best member (chance) | 0.30 /  / 0.20 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 11 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L16 attn |
| CKA(arrangement before, joint write) | 0.34 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.12 |
| share of the write inside the old arrangement's span | 0.11 |
| write energy / code energy before | 0.04 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (10:0.28 20:0.28 40:0.20) |
| joint write: shape (spectrum k:share) | irregular (40:0.23 30:0.22 20:0.22) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.20 / 1.78 |

<details><summary>codes and components</summary>

**code 1a-6.0 (L16 L17): 2 comps, tells apart 3/10, coverage 0.20, overlap 1.00 (random 1.00, p 0.87)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c34 | MLP | a%10 in {8} (-) | 0.93 | 0.100 |
| L17 down c120 | MLP | a%10 in {9} (-) | 0.91 | 0.100 |

</details>

</details>

<details><summary><b>1a-7</b> `a%10` @ `a` (add) — block code; 3 comps, L21; tells apart 4/10 classes (best member 2); on sub: 1s-82 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.30 |
| classes told apart: joint / best code / best member | 4 /  / 2 (of 10) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.57 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.96 |
| decoding acc. joint / best code / best member (chance) | 0.42 /  / 0.22 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 2 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 attn |
| CKA(arrangement before, joint write) | 0.65 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.21 |
| share of the write inside the old arrangement's span | 0.13 |
| write energy / code energy before | 0.04 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (20:0.28 40:0.25 10:0.22) |
| joint write: shape (spectrum k:share) | irregular (20:0.25 40:0.22 30:0.21) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.22 / 2.00 |

<details><summary>codes and components</summary>

**code 1a-7.0 (L21): 3 comps, tells apart 4/10, coverage 0.30, overlap 1.00 (random 1.00, p 0.67)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c19 | MLP | a%10 in {0} (-) | 0.95 | 0.120 |
| L21 down c81 | MLP | a%10 in {3} (-) | 1.00 | 0.100 |
| L21 down c104 | MLP | a%10 in {7} (-) | 0.99 | 0.100 |

</details>

</details>

<details><summary><b>1a-8</b> `a%10` @ `a` (add) — single component; 1 comps, L23; tells apart 2/10 classes (best member 2); on sub: 1s-83 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.10 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 10) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.98 |
| decoding acc. joint / best code / best member (chance) | 0.20 /  / 0.20 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L23 attn |
| CKA(arrangement before, joint write) | 0.24 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.08 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (20:0.27 40:0.24 10:0.21) |
| joint write: shape (spectrum k:share) | line (10:0.22 20:0.22 30:0.22) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.11 / 1.00 |

<details><summary>codes and components</summary>

**code 1a-8.0 (L23): 1 comps, tells apart 2/10, coverage 0.10, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 down c46 | MLP | a%10 in {9} (-) | 0.98 | 0.100 |

</details>

</details>

</details>

<details><summary>`a%50`: 9 mechanisms, 27 components</summary>

<details><summary><b>1a-56</b> `a%50` @ `a` (add) — single component; 1 comps, L2; tells apart 4/50 classes (best member 4); on sub: 1s-131 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.02 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 50) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.93 |
| decoding acc. joint / best code / best member (chance) | 0.06 /  / 0.06 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 attn |
| CKA(arrangement before, joint write) | 0.13 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.21 10:0.11 20:0.11) |
| joint write: shape (spectrum k:share) | line (20:0.10 40:0.10 32:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.03 / 1.00 |

<details><summary>codes and components</summary>

**code 1a-56.0 (L2): 1 comps, tells apart 4/50, coverage 0.02, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 down c98 | MLP | a%50 in {25} (-) | 0.93 | 0.060 |

</details>

</details>

<details><summary><b>1a-62</b> `a%50` @ `a` (add) — 2 codes of the same shape; 2 comps, L4 L22; tells apart 3/50 classes (best member 3); on sub: 1s-137 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.04 |
| classes told apart: joint / best code / best member | 3 / 3 / 3 (of 50) |
| members whose removal merges classes | 0.50 |
| support overlap (1 = tiling) / random sets / p | 1.50 / 1.00 / 0.92 |
| mean CKA between its codes | 0.77 |
| purity of the joint write (per prompt) | 0.96 |
| decoding acc. joint / best code / best member (chance) | 0.06 / 0.05 / 0.05 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L4 attn |
| CKA(arrangement before, joint write) | 0.18 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 0.04 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.23 20:0.11 10:0.11) |
| joint write: shape (spectrum k:share) | line (4:0.07 8:0.07 12:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.03 / 1.06 |

<details><summary>codes and components</summary>

**code 1a-62.0 (L4): 1 comps, tells apart 2/50, coverage 0.02, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 down c24 | MLP | a%50 in {0} (+) | 0.96 | 0.020 |

**code 1a-62.1 (L22): 1 comps, tells apart 3/50, coverage 0.04, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 down c116 | MLP | a%50 in {0, 25} (-) | 0.95 | 0.040 |

</details>

</details>

<details><summary><b>1a-57</b> `a%50` @ `a` (add) — block code, 5 codes of the same shape; 9 comps, L5 L15 L16 L17 L18 L23 L30; tells apart 9/50 classes (best member 5); on sub: 1s-132 (member overlap 0.90)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.40 |
| classes told apart: joint / best code / best member | 9 / 10 / 5 (of 50) |
| members whose removal merges classes | 0.33 |
| support overlap (1 = tiling) / random sets / p | 1.55 / 1.70 / 0.28 |
| mean CKA between its codes | 0.82 |
| purity of the joint write (per prompt) | 0.94 |
| decoding acc. joint / best code / best member (chance) | 0.25 / 0.18 / 0.08 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.03 |
| consumers / read jointly | 9 / 6 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L5 attn |
| CKA(arrangement before, joint write) | 0.31 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 1.33 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.24 10:0.12 20:0.11) |
| joint write: shape (spectrum k:share) | line (40:0.25 20:0.25 30:0.14) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.10 / 1.19 |

<details><summary>codes and components</summary>

**code 1a-57.0 (L5): 1 comps, tells apart 5/50, coverage 0.10, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 down c28 | MLP | a%50 in {0, 10, 25, 30, 40} (+) | 0.93 | 0.181 |

**code 1a-57.1 (L15 L16): 4 comps, tells apart 10/50, coverage 0.24, overlap 1.00 (random 1.22, p 0.14)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c77 | MLP | a%50 in {1, 11, 21, 31, 41} (-) | 0.90 | 0.090 |
| L15 down c374 | MLP | a%50 in {19, 29, 39, 49} (-) | 0.92 | 0.090 |
| L15 down c532 | MLP | a%50 in {43} (+) | 0.96 | 0.020 |
| L16 down c120 | MLP | a%50 in {0, 25} (-) | 0.97 | 0.118 |

**code 1a-57.2 (L17 L18): 2 comps, tells apart 4/50, coverage 0.10, overlap 1.00 (random 1.00, p 0.68)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 down c704 | MLP | a%50 in {17, 27, 37, 47} (-) | 0.98 | 0.080 |
| L18 down c51 | MLP | a%50 in {0} (+) | 0.93 | 0.110 |

**code 1a-57.3 (L23): 1 comps, tells apart 4/50, coverage 0.08, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 down c101 | MLP | a%50 in {0, 10, 20, 30} (+) | 0.97 | 0.130 |

**code 1a-57.4 (L30): 1 comps, tells apart 5/50, coverage 0.10, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 down c994 | MLP | a%50 in {0, 20, 25, 30, 40} (+) | 0.93 | 0.112 |

</details>

</details>

<details><summary><b>1a-58</b> `a%50` @ `a` (add) — block code; 3 comps, L10 L11; tells apart 9/50 classes (best member 5); on sub: 1s-133 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.28 |
| classes told apart: joint / best code / best member | 9 /  / 5 (of 50) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.10 / 0.36 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.93 |
| decoding acc. joint / best code / best member (chance) | 0.16 /  / 0.12 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L10 attn |
| CKA(arrangement before, joint write) | 0.28 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.03 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.24 10:0.12 20:0.11) |
| joint write: shape (spectrum k:share) | irregular (10:0.39 20:0.23 30:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.22 / 1.94 |

<details><summary>codes and components</summary>

**code 1a-58.0 (L10 L11): 3 comps, tells apart 9/50, coverage 0.28, overlap 1.00 (random 1.12, p 0.30)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L10 down c937 | MLP | a%50 in {0} (-) | 0.91 | 0.030 |
| L11 down c38 | MLP | a%50 in {13, 23, 33, 43} (+) | 0.91 | 0.090 |
| L11 down c20 | MLP | a%50 in {2, 11..12, 21..22, 31..32, 41..42} (-) | 0.93 | 0.270 |

</details>

</details>

<details><summary><b>1a-59</b> `a%50` @ `a` (add) — block code; 2 comps, L13 L14; tells apart 6/50 classes (best member 5); on sub: 1s-134 (member overlap 0.67)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.12 |
| classes told apart: joint / best code / best member | 6 /  / 5 (of 50) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.67 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.91 |
| decoding acc. joint / best code / best member (chance) | 0.08 /  / 0.05 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L13 attn |
| CKA(arrangement before, joint write) | 0.32 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.02 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.24 10:0.15 20:0.13) |
| joint write: shape (spectrum k:share) | irregular (10:0.17 40:0.17 20:0.16) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.10 / 1.63 |

<details><summary>codes and components</summary>

**code 1a-59.0 (L13 L14): 2 comps, tells apart 6/50, coverage 0.12, overlap 1.00 (random 1.00, p 0.65)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 down c263 | MLP | a%50 in {0, 10, 20, 30, 40} (+) | 0.91 | 0.090 |
| L14 down c625 | MLP | a%50 in {48} (-) | 0.92 | 0.030 |

</details>

</details>

<details><summary><b>1a-60</b> `a%50` @ `a` (add) — block code, 2 codes of the same shape; 5 comps, L19 L20 L28 L29; tells apart 9/50 classes (best member 6); on sub: 1s-135 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.34 |
| classes told apart: joint / best code / best member | 9 / 9 / 6 (of 50) |
| members whose removal merges classes | 0.60 |
| support overlap (1 = tiling) / random sets / p | 1.59 / 1.36 / 0.77 |
| mean CKA between its codes | 0.74 |
| purity of the joint write (per prompt) | 0.92 |
| decoding acc. joint / best code / best member (chance) | 0.19 / 0.14 / 0.08 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 3 / 1 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L19 attn |
| CKA(arrangement before, joint write) | 0.44 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.15 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.23 20:0.12 4:0.10) |
| joint write: shape (spectrum k:share) | irregular (40:0.30 20:0.28 10:0.13) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.28 / 2.61 |

<details><summary>codes and components</summary>

**code 1a-60.0 (L19 L20): 2 comps, tells apart 8/50, coverage 0.26, overlap 1.00 (random 1.00, p 0.67)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L19 down c47 | MLP | a%50 in {0, 10, 15, 20, 25, 30, 35, 40, 45} (-) | 0.92 | 0.190 |
| L20 down c41 | MLP | a%50 in {2, 22, 32, 42} (-) | 0.94 | 0.090 |

**code 1a-60.1 (L28 L29): 3 comps, tells apart 9/50, coverage 0.28, overlap 1.00 (random 1.10, p 0.35)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 down c174 | MLP | a%50 in {2, 22, 32, 42} (-) | 0.94 | 0.090 |
| L29 down c877 | MLP | a%50 in {0, 10, 20, 30, 40, 45} (-) | 0.91 | 0.188 |
| L29 down c107 | MLP | a%50 in {1, 21, 31, 41} (-) | 0.93 | 0.090 |

</details>

</details>

<details><summary><b>1a-61</b> `a%50` @ `a` (add) — block code; 2 comps, L21; tells apart 7/50 classes (best member 4); on sub: 1s-136 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.08 |
| classes told apart: joint / best code / best member | 7 /  / 4 (of 50) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.69 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.94 |
| decoding acc. joint / best code / best member (chance) | 0.11 /  / 0.07 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 attn |
| CKA(arrangement before, joint write) | 0.27 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.06 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.24 20:0.12 4:0.11) |
| joint write: shape (spectrum k:share) | irregular (20:0.16 40:0.15 10:0.12) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.09 / 1.90 |

<details><summary>codes and components</summary>

**code 1a-61.0 (L21): 2 comps, tells apart 7/50, coverage 0.08, overlap 1.00 (random 1.00, p 0.69)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c55 | MLP | a%50 in {0} (-) | 0.97 | 0.050 |
| L21 down c194 | MLP | a%50 in {5, 35, 45} (-) | 0.92 | 0.100 |

</details>

</details>

<details><summary><b>1a-63</b> `a%50` @ `a` (add) — block code; 2 comps, L24 L25; tells apart 7/50 classes (best member 6); on sub: 1s-138 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.36 |
| classes told apart: joint / best code / best member | 7 /  / 6 (of 50) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.68 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.93 |
| decoding acc. joint / best code / best member (chance) | 0.14 /  / 0.12 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L24 attn |
| CKA(arrangement before, joint write) | 0.39 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.25 20:0.11 40:0.10) |
| joint write: shape (spectrum k:share) | irregular (10:0.37 20:0.30 40:0.19) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.24 / 1.78 |

<details><summary>codes and components</summary>

**code 1a-63.0 (L24 L25): 2 comps, tells apart 7/50, coverage 0.36, overlap 1.00 (random 1.00, p 0.70)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 down c174 | MLP | a%50 in {0, 10, 20, 25, 30, 40} (+) | 0.96 | 0.180 |
| L25 down c5 | MLP | a%50 in {1..3, 21..23, 31..33, 41..43} (+) | 0.91 | 0.270 |

</details>

</details>

<details><summary><b>1a-64</b> `a%50` @ `a` (add) — single component; 1 comps, L27; tells apart 4/50 classes (best member 4); on sub: 1s-139 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.16 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 50) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.92 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.04 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 3 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L27 attn |
| CKA(arrangement before, joint write) | 0.15 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | irregular (2:0.25 20:0.11 40:0.11) |
| joint write: shape (spectrum k:share) | line (10:0.42 20:0.31 30:0.16) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.15 / 1.00 |

<details><summary>codes and components</summary>

**code 1a-64.0 (L27): 1 comps, tells apart 4/50, coverage 0.16, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 down c5 | MLP | a%50 in {2..3, 22..23, 32..33, 42..43} (-) | 0.92 | 0.180 |

</details>

</details>

</details>

<details><summary>`a%20`: 3 mechanisms, 6 components</summary>

<details><summary><b>1a-51</b> `a%20` @ `a` (add) — single component; 1 comps, L4; tells apart 4/20 classes (best member 4); on sub: 1s-126 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.40 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 20) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.93 |
| decoding acc. joint / best code / best member (chance) | 0.17 /  / 0.17 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L4 attn |
| CKA(arrangement before, joint write) | 0.32 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.13 |
| share of the write inside the old arrangement's span | 0.21 |
| write energy / code energy before | 0.03 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (20:0.21 10:0.21 5:0.18) |
| joint write: shape (spectrum k:share) | line (50:0.80 40:0.05 10:0.05) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.64 / 1.00 |

<details><summary>codes and components</summary>

**code 1a-51.0 (L4): 1 comps, tells apart 4/20, coverage 0.40, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 down c135 | MLP | a%20 in {2, 4, 6, 8, 12, 14, 16, 18} (+) | 0.93 | 0.460 |

</details>

</details>

<details><summary><b>1a-52</b> `a%20` @ `a` (add) — tiling; 4 comps, L11 L12; tells apart 11/20 classes (best member 5); on sub: 1s-127 (member overlap 0.75)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.65 |
| classes told apart: joint / best code / best member | 11 /  / 5 (of 20) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.08 / 1.31 / 0.05 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.95 |
| decoding acc. joint / best code / best member (chance) | 0.51 /  / 0.20 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 10 / 4 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L11 attn |
| CKA(arrangement before, joint write) | 0.35 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.13 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.10 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (10:0.22 20:0.22 5:0.17) |
| joint write: shape (spectrum k:share) | irregular (5:0.34 10:0.32 15:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.29 / 2.43 |

<details><summary>codes and components</summary>

**code 1a-52.0 (L11 L12): 4 comps, tells apart 11/20, coverage 0.65, overlap 1.08 (random 1.31, p 0.05)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L11 down c40 | MLP | a%20 in {8, 18} (+) | 0.91 | 0.090 |
| L12 down c9 | MLP | a%20 in {10..14} (-) | 0.97 | 0.250 |
| L12 down c73 | MLP | a%20 in {15..17} (-) | 0.91 | 0.210 |
| L12 down c32 | MLP | a%20 in {2..4, 13} (+) | 0.93 | 0.230 |

</details>

</details>

<details><summary><b>1a-53</b> `a%20` @ `a` (add) — single component; 1 comps, L26; tells apart 5/20 classes (best member 5); on sub: 1s-128 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.20 |
| classes told apart: joint / best code / best member | 5 /  / 5 (of 20) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.90 |
| decoding acc. joint / best code / best member (chance) | 0.12 /  / 0.12 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L26 attn |
| CKA(arrangement before, joint write) | 0.70 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (20:0.21 40:0.19 10:0.16) |
| joint write: shape (spectrum k:share) | line (40:0.47 20:0.47) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.22 / 1.00 |

<details><summary>codes and components</summary>

**code 1a-53.0 (L26): 1 comps, tells apart 5/20, coverage 0.20, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 down c109 | MLP | a%20 in {0, 5, 10, 15} (-) | 0.90 | 0.190 |

</details>

</details>

</details>

<details><summary>`a%25`: 1 mechanisms, 3 components</summary>

<details><summary><b>1a-54</b> `a%25` @ `a` (add) — 3 codes of the same shape; 3 comps, L0 L15 L20; tells apart 3/25 classes (best member 4); on sub: 1s-129 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.20 |
| classes told apart: joint / best code / best member | 3 / 4 / 4 (of 25) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 2.40 /  /  |
| mean CKA between its codes | 0.98 |
| purity of the joint write (per prompt) | 0.94 |
| decoding acc. joint / best code / best member (chance) | 0.12 / 0.12 / 0.12 (0.04) |
| kappa (1 orthogonal, >1 constructive) | 1.07 |
| consumers / read jointly | 16 / 4 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 attn |
| CKA(arrangement before, joint write) | 0.47 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 8.86 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (20:0.17 4:0.16 40:0.12) |
| joint write: shape (spectrum k:share) | line (20:0.46 40:0.46) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.21 / 1.01 |

<details><summary>codes and components</summary>

**code 1a-54.0 (L0): 1 comps, tells apart 4/25, coverage 0.16, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 down c162 | MLP | a%25 in {0, 5, 10, 15} (-) | 0.91 | 0.190 |

**code 1a-54.1 (L15): 1 comps, tells apart 3/25, coverage 0.16, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c70 | MLP | a%25 in {0, 5, 10, 20} (+) | 0.93 | 0.200 |

**code 1a-54.2 (L20): 1 comps, tells apart 4/25, coverage 0.16, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 down c85 | MLP | a%25 in {0, 10, 15, 20} (+) | 0.94 | 0.190 |

</details>

</details>

</details>

<details><summary>`a%5`: 1 mechanisms, 2 components</summary>

<details><summary><b>1a-55</b> `a%5` @ `a` (add) — 2 codes of the same shape; 2 comps, L2 L17; tells apart 2/5 classes (best member 2); on sub: 1s-130 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.20 |
| classes told apart: joint / best code / best member | 2 / 2 / 2 (of 5) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 2.00 /  /  |
| mean CKA between its codes | 1.00 |
| purity of the joint write (per prompt) | 0.92 |
| decoding acc. joint / best code / best member (chance) | 0.39 / 0.39 / 0.39 (0.20) |
| kappa (1 orthogonal, >1 constructive) | 1.04 |
| consumers / read jointly | 27 / 1 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 attn |
| CKA(arrangement before, joint write) | 0.73 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.12 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 5.07 |
| arrangement before: shape (spectrum k:share) | circle period 5 (20:0.67 40:0.33) |
| joint write: shape (spectrum k:share) | line (40:0.50 20:0.50) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.25 / 1.00 |

<details><summary>codes and components</summary>

**code 1a-55.0 (L2): 1 comps, tells apart 2/5, coverage 0.20, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 down c473 | MLP | a%5 in {0} (+) | 0.92 | 0.200 |

**code 1a-55.1 (L17): 1 comps, tells apart 2/5, coverage 0.20, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 down c8 | MLP | a%5 in {0} (+) | 0.92 | 0.200 |

</details>

</details>

</details>
