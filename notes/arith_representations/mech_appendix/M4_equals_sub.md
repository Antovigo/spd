[← back to the report](../report_mechanisms.md)

# Every mechanism at `=`, sub

<details><summary>`tens(a,b)`: 35 mechanisms, 255 components</summary>

<details><summary><b>4s-639</b> `tens(a,b)` @ `=` (sub) — single component; 1 comps, L1; tells apart 5/121 classes (best member 5); on add: 4a-535 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.01 |
| classes told apart: joint / best code / best member | 5 /  / 5 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.72 |
| decoding acc. joint / best code / best member (chance) | 0.02 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.15 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.08 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.03 |
| best source position (CKA) | `op` (0.08) |

<details><summary>codes and components</summary>

**code 4s-639.0 (L1): 1 comps, tells apart 5/121, coverage 0.01, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c377 | H26 | a//10 {0} x b//10 {0} (-) | 0.72 | 0.016 |

</details>

</details>

<details><summary><b>4s-640</b> `tens(a,b)` @ `=` (sub) — block code, 3 codes of the same shape; 6 comps, L2 L3 L4; tells apart 27/121 classes (best member 7); on add: 4a-536 (member overlap 0.14)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.43 |
| classes told apart: joint / best code / best member | 27 / 11 / 7 (of 121) |
| members whose removal merges classes | 0.83 |
| support overlap (1 = tiling) / random sets / p | 1.58 / 1.41 / 0.74 |
| mean CKA between its codes | 0.75 |
| purity of the joint write (per prompt) | 0.74 |
| decoding acc. joint / best code / best member (chance) | 0.26 / 0.12 / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.88 |
| consumers / read jointly | 17 / 10 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 attn |
| CKA(arrangement before, joint write) | 0.48 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.28 |
| best source position (CKA) | `b` (0.41) |

<details><summary>codes and components</summary>

**code 4s-640.0 (L2 L3): 2 comps, tells apart 11/121, coverage 0.11, overlap 1.00 (random 1.02, p 0.47)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 down c266 | MLP | a//10 {0} x b//10 {0..3}; a//10 {3} x b//10 {10}; a//10 {6, 9} x b//10 {0}; a//10 {10} x b//10 {0..2} (+) | 0.72 | 1.000 |
| L3 o c243 | H15 | a//10 {8} x b//10 {9}; a//10 {9} x b//10 {9..10} (-) | 0.76 | 0.055 |

**code 4s-640.1 (L3): 2 comps, tells apart 9/121, coverage 0.17, overlap 1.19 (random 1.03, p 0.84)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 down c28 | MLP | a//10 {0, 10} x b//10 {0..2, 10}; a//10 {1, 8..9} x b//10 {0} (-) | 0.74 | 1.000 |
| L3 down c5 | MLP | a//10 {1} x b//10 {1, 5, 10}; a//10 {2..5} x b//10 {10}; a//10 {7} x b//10 {2}; a//10 {8} x b//10 {1..2}; a//10 {10} x b//10 {0..2, 10} (+) | 0.63 | 1.000 |

**code 4s-640.2 (L4): 2 comps, tells apart 6/121, coverage 0.34, overlap 1.07 (random 1.00, p 0.70)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 o c23 | H25 | a//10 {0..2} x b//10 {0}; a//10 {3} x b//10 {0, 8}; a//10 {4} x b//10 {0..1}; a//10 {5} x b//10 {0..1, 10}; a//10 {6} x b//10 {0..1, 7..8}; a//10 {7} x b//10 {0..1, 8}; a//10 {8} x b//10 {0, 8..9}; a//10 {9} x b//10 {0..3}; a//10 {10} x b//10 {8..9} (+) | 0.77 | 1.000 |
| L4 down c16 | MLP | a//10 {0} x b//10 {0..10}; a//10 {8..9} x b//10 {0}; a//10 {10} x b//10 {0..2, 4, 10} (-) | 0.75 | 1.000 |

</details>

</details>

<details><summary><b>4s-641</b> `tens(a,b)` @ `=` (sub) — block code; 3 comps, L3 L4; tells apart 11/121 classes (best member 5); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.35 |
| classes told apart: joint / best code / best member | 11 /  / 5 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.12 / 1.13 / 0.47 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.82 |
| decoding acc. joint / best code / best member (chance) | 0.17 /  / 0.06 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.94 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L2 MLP |
| CKA(arrangement before, joint write) | 0.64 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.06 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.05 |
| best source position (CKA) | `op` (0.42) |

<details><summary>codes and components</summary>

**code 4s-641.0 (L3 L4): 3 comps, tells apart 11/121, coverage 0.35, overlap 1.12 (random 1.12, p 0.49)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L3 o c221 | H5 | a//10 {0} x b//10 {0..10}; a//10 {4..10} x b//10 {0} (-) | 0.88 | 0.167 |
| L3 o c128 | H5 | a//10 {0} x b//10 {0..3, 10}; a//10 {1..3} x b//10 {0..1}; a//10 {5} x b//10 {8}; a//10 {6} x b//10 {6..9}; a//10 {7} x b//10 {6..10}; a//10 {8} x b//10 {7..9}; a//10 {9} x b//10 {9} (-) | 0.85 | 1.000 |
| L4 o c7 | H18 | a//10 {9} x b//10 {1..2}; a//10 {10} x b//10 {9..10} (+) | 0.62 | 1.000 |

</details>

</details>

<details><summary><b>4s-643</b> `tens(a,b)` @ `=` (sub) — block code, 2 codes of the same shape, copy from `b`; 4 comps, L5; tells apart 2/121 classes (best member 2); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.78 |
| classes told apart: joint / best code / best member | 2 / 2 / 2 (of 121) |
| members whose removal merges classes | 0.25 |
| support overlap (1 = tiling) / random sets / p | 1.89 / 1.22 / 1.00 |
| mean CKA between its codes | 0.77 |
| purity of the joint write (per prompt) | 0.86 |
| decoding acc. joint / best code / best member (chance) | 0.21 / 0.17 / 0.06 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.02 |
| consumers / read jointly | 21 / 19 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L4 MLP |
| CKA(arrangement before, joint write) | 0.60 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.02 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.25 |
| best source position (CKA) | `b` (0.57) |

<details><summary>codes and components</summary>

**code 4s-643.0 (L5): 1 comps, tells apart 1/121, coverage 0.30, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 o c6 | H26 | a//10 {1..2} x b//10 {0}; a//10 {3, 10} x b//10 {0, 9..10}; a//10 {4} x b//10 {0..1, 9..10}; a//10 {5} x b//10 {0..1, 6..7, 9..10}; a//10 {6} x b//10 {0..2, 6..7, 9..10}; a//10 {7} x b//10 {6..10}; a//10 {8} x b//10 {7..10}; a//10 {9} x b//10 {9..10} (+) | 0.75 | 1.000 |

**code 4s-643.1 (L5): 3 comps, tells apart 2/121, coverage 0.77, overlap 1.53 (random 1.11, p 0.99)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 down c1 | MLP | a//10 {0} x b//10 {0, 3..8}; a//10 {1} x b//10 {0, 2}; a//10 {2} x b//10 {0, 2..3}; a//10 {3} x b//10 {0, 3..4}; a//10 {4} x b//10 {0, 5, 9..10}; a//10 {5} x b//10 {0, 6, 9}; a//10 {6} x b//10 {0, 6..7, 9..10}; a//10 {7} x b//10 {0, 9..10}; a//10 {8} x b//10 {0, 8..10}; a//10 {9} x b//10 {9..10}; a//10 {10} x b//10 {0..4} (+) | 0.78 | 1.000 |
| L5 down c82 | MLP | a//10 {0} x b//10 {0..10}; a//10 {1} x b//10 {0}; a//10 {2} x b//10 {0..1}; a//10 {3} x b//10 {0..1, 5..10}; a//10 {4} x b//10 {0..2, 5..10}; a//10 {5..7} x b//10 {0..2, 6..10}; a//10 {8} x b//10 {0..2, 7..10}; a//10 {9} x b//10 {0..2, 9..10}; a//10 {10} x b//10 {0..2, 9} (-) | 0.90 | 0.736 |
| L5 down c9 | MLP | a//10 {0} x b//10 {0..3, 6..9}; a//10 {1} x b//10 {1..2, 9..10}; a//10 {4..5} x b//10 {3}; a//10 {6} x b//10 {3, 5}; a//10 {7} x b//10 {1, 3..5}; a//10 {8} x b//10 {5}; a//10 {9} x b//10 {9}; a//10 {10} x b//10 {1..8} (+) | 0.61 | 1.000 |

</details>

</details>

<details><summary><b>4s-642</b> `tens(a,b)` @ `=` (sub) — block code, 8 codes of the same shape; 30 comps, L5 L7 L8 L9 L11 L12 L13 L14; tells apart 25/121 classes (best member 10); on add: 4a-540 (member overlap 0.07)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.82 |
| classes told apart: joint / best code / best member | 25 / 14 / 10 (of 121) |
| members whose removal merges classes | 0.17 |
| support overlap (1 = tiling) / random sets / p | 6.27 / 4.06 / 1.00 |
| mean CKA between its codes | 0.84 |
| purity of the joint write (per prompt) | 0.75 |
| decoding acc. joint / best code / best member (chance) | 0.27 / 0.19 / 0.06 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.76 |
| consumers / read jointly | 76 / 74 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L4 MLP |
| CKA(arrangement before, joint write) | 0.60 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 6.87 |
| best source position (CKA) | `b` (0.49) |

<details><summary>codes and components</summary>

**code 4s-642.0 (L5): 3 comps, tells apart 9/121, coverage 0.19, overlap 1.00 (random 1.11, p 0.15)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 o c450 | H31 | a//10 {0} x b//10 {0..2} (+) | 0.65 | 0.053 |
| L5 o c22 | H31 | a//10 {0} x b//10 {3..9}; a//10 {1} x b//10 {0, 8..9}; a//10 {2..5, 7} x b//10 {0}; a//10 {6} x b//10 {0, 8} (-) | 0.85 | 0.985 |
| L5 o c17 | H22 | a//10 {9} x b//10 {9..10}; a//10 {10} x b//10 {9} (+) | 0.64 | 0.033 |

**code 4s-642.1 (L7): 5 comps, tells apart 13/121, coverage 0.50, overlap 1.90 (random 1.30, p 0.98)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L7 down c1 | MLP | a//10 {0} x b//10 {0, 3..8}; a//10 {4..5} x b//10 {0}; a//10 {6} x b//10 {0, 6..7}; a//10 {7} x b//10 {6..8, 10}; a//10 {8} x b//10 {7..10}; a//10 {9} x b//10 {9..10}; a//10 {10} x b//10 {0..4} (-) | 0.75 | 1.000 |
| L7 down c42 | MLP | a//10 {0} x b//10 {0..10}; a//10 {1} x b//10 {0, 3..10}; a//10 {2} x b//10 {0, 10}; a//10 {3..9} x b//10 {0}; a//10 {10} x b//10 {0..1} (-) | 0.87 | 0.375 |
| L7 down c23 | MLP | a//10 {0} x b//10 {0..2, 10}; a//10 {1} x b//10 {0} (+) | 0.58 | 0.999 |
| L7 down c2 | MLP | a//10 {0} x b//10 {3..10}; a//10 {1} x b//10 {4..10}; a//10 {2} x b//10 {8..10}; a//10 {3} x b//10 {4..6}; a//10 {4} x b//10 {5..7}; a//10 {5} x b//10 {6..8}; a//10 {6..7} x b//10 {6..10}; a//10 {8} x b//10 {7..10}; a//10 {9} x b//10 {9..10} (-) | 0.86 | 1.000 |
| L7 down c186 | MLP | a//10 {5} x b//10 {6..7}; a//10 {6} x b//10 {7}; a//10 {7} x b//10 {8..10}; a//10 {8..9} x b//10 {9..10} (-) | 0.50 | 0.136 |

**code 4s-642.2 (L8): 4 comps, tells apart 14/121, coverage 0.42, overlap 1.27 (random 1.21, p 0.62)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L8 down c928 | MLP | a//10 {0} x b//10 {0..5, 8}; a//10 {1, 10} x b//10 {0} (+) | 0.78 | 1.000 |
| L8 down c8 | MLP | a//10 {0} x b//10 {3..10}; a//10 {1} x b//10 {0, 4..10}; a//10 {2} x b//10 {0, 8..10}; a//10 {4..5} x b//10 {0} (-) | 0.80 | 0.396 |
| L8 o c16 | H23 | a//10 {1} x b//10 {2}; a//10 {5} x b//10 {6..7}; a//10 {6} x b//10 {6..8}; a//10 {7} x b//10 {6..10}; a//10 {8} x b//10 {7..10}; a//10 {9} x b//10 {9..10} (+) | 0.58 | 0.874 |
| L8 down c1 | MLP | a//10 {1} x b//10 {7..10}; a//10 {2} x b//10 {9..10}; a//10 {5..6} x b//10 {0, 2}; a//10 {9} x b//10 {1..4, 9..10}; a//10 {10} x b//10 {10} (+) | 0.60 | 1.000 |

**code 4s-642.3 (L9): 4 comps, tells apart 14/121, coverage 0.36, overlap 1.79 (random 1.22, p 0.99)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L9 down c50 | MLP | a//10 {0} x b//10 {0..10}; a//10 {1} x b//10 {0}; a//10 {9} x b//10 {10}; a//10 {10} x b//10 {0, 10} (+) | 0.77 | 1.000 |
| L9 down c6 | MLP | a//10 {0} x b//10 {2..10}; a//10 {1} x b//10 {4..10}; a//10 {2} x b//10 {8..10} (-) | 0.90 | 0.232 |
| L9 down c10 | MLP | a//10 {0} x b//10 {3..10}; a//10 {1} x b//10 {0, 4..10}; a//10 {2} x b//10 {0, 8, 10}; a//10 {3..6} x b//10 {0}; a//10 {7..8} x b//10 {0..1}; a//10 {9} x b//10 {0..2}; a//10 {10} x b//10 {0..4, 10} (-) | 0.78 | 0.366 |
| L9 down c1 | MLP | a//10 {4..5} x b//10 {0}; a//10 {7} x b//10 {7}; a//10 {9} x b//10 {9..10}; a//10 {10} x b//10 {1..2} (-) | 0.61 | 1.000 |

**code 4s-642.4 (L11): 5 comps, tells apart 8/121, coverage 0.51, overlap 1.53 (random 1.30, p 0.85)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L11 down c0 | MLP | a//10 {0, 9} x b//10 {10} (-) | 0.62 | 1.000 |
| L11 down c34 | MLP | a//10 {0} x b//10 {2..10}; a//10 {1} x b//10 {4..10}; a//10 {2} x b//10 {5, 7..10}; a//10 {3} x b//10 {10} (+) | 0.85 | 0.200 |
| L11 down c75 | MLP | a//10 {0} x b//10 {3..10}; a//10 {1} x b//10 {0, 4..10}; a//10 {2} x b//10 {0, 8..10}; a//10 {3} x b//10 {0, 10}; a//10 {4..6} x b//10 {0}; a//10 {7} x b//10 {0..2}; a//10 {8} x b//10 {0..1}; a//10 {9} x b//10 {0..4}; a//10 {10} x b//10 {0..8, 10} (-) | 0.67 | 0.363 |
| L11 down c176 | MLP | a//10 {0} x b//10 {3..10}; a//10 {1} x b//10 {5, 7..8, 10} (-) | 0.73 | 0.101 |
| L11 down c468 | MLP | a//10 {6} x b//10 {6..8}; a//10 {7..8} x b//10 {6..9}; a//10 {9} x b//10 {9..10}; a//10 {10} x b//10 {9} (+) | 0.65 | 0.138 |

**code 4s-642.5 (L12): 3 comps, tells apart 13/121, coverage 0.26, overlap 1.41 (random 1.10, p 0.96)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L12 down c14 | MLP | a//10 {0..2} x b//10 {10}; a//10 {9} x b//10 {9..10} (-) | 0.73 | 1.000 |
| L12 down c56 | MLP | a//10 {0} x b//10 {0..8, 10}; a//10 {1, 4..7} x b//10 {0}; a//10 {9} x b//10 {9..10}; a//10 {10} x b//10 {0, 10} (-) | 0.67 | 1.000 |
| L12 down c16 | MLP | a//10 {0} x b//10 {3..10}; a//10 {1} x b//10 {4..10}; a//10 {2} x b//10 {8..10}; a//10 {10} x b//10 {0..2} (-) | 0.78 | 0.420 |

**code 4s-642.6 (L13): 3 comps, tells apart 5/121, coverage 0.55, overlap 1.32 (random 1.12, p 0.94)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 down c127 | MLP | a//10 {0} x b//10 {0..10}; a//10 {1, 10} x b//10 {0, 10}; a//10 {9} x b//10 {10} (-) | 0.72 | 1.000 |
| L13 down c32 | MLP | a//10 {0} x b//10 {3..10}; a//10 {1} x b//10 {0, 4..10}; a//10 {2} x b//10 {0, 8..10}; a//10 {3} x b//10 {0, 6}; a//10 {4} x b//10 {0}; a//10 {5..6} x b//10 {0, 6..8}; a//10 {7} x b//10 {0, 6..9}; a//10 {8} x b//10 {0, 7, 9..10}; a//10 {9} x b//10 {0..1, 10}; a//10 {10} x b//10 {0..2, 6..9} (-) | 0.81 | 0.540 |
| L13 down c4 | MLP | a//10 {1} x b//10 {5..10}; a//10 {2} x b//10 {9..10}; a//10 {3} x b//10 {3..4}; a//10 {4..5} x b//10 {4..5}; a//10 {9} x b//10 {2..7, 10} (-) | 0.59 | 0.999 |

**code 4s-642.7 (L14): 3 comps, tells apart 8/121, coverage 0.73, overlap 1.28 (random 1.11, p 0.87)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L14 down c43 | MLP | a//10 {0..1} x b//10 {3..10}; a//10 {2} x b//10 {5..10}; a//10 {3} x b//10 {8..10} (+) | 0.84 | 0.233 |
| L14 down c6 | MLP | a//10 {0} x b//10 {0..8, 10}; a//10 {1, 9..10} x b//10 {10} (-) | 0.67 | 1.000 |
| L14 down c72 | MLP | a//10 {0} x b//10 {3..10}; a//10 {1} x b//10 {0, 4..10}; a//10 {2} x b//10 {0, 2..3, 10}; a//10 {3} x b//10 {0, 2..6}; a//10 {4} x b//10 {0, 5..7}; a//10 {5} x b//10 {0..1, 4..9}; a//10 {6} x b//10 {0..1, 5..9}; a//10 {7..8} x b//10 {0..2, 6..10}; a//10 {9} x b//10 {0..4, 9..10}; a//10 {10} x b//10 {0..3, 6..7, 9} (-) | 0.82 | 0.686 |

</details>

</details>

<details><summary><b>4s-644</b> `tens(a,b)` @ `=` (sub) — block code, copy from `b`; 2 comps, L6; tells apart 2/121 classes (best member 3); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.37 |
| classes told apart: joint / best code / best member | 2 /  / 3 (of 121) |
| members whose removal merges classes | 0.50 |
| support overlap (1 = tiling) / random sets / p | 1.02 / 1.03 / 0.46 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.80 |
| decoding acc. joint / best code / best member (chance) | 0.09 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L5 MLP |
| CKA(arrangement before, joint write) | 0.48 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.14 |
| share of the write inside the old arrangement's span | 0.16 |
| write energy / code energy before | 0.03 |
| best source position (CKA) | `b` (0.67) |

<details><summary>codes and components</summary>

**code 4s-644.0 (L6): 2 comps, tells apart 2/121, coverage 0.37, overlap 1.02 (random 1.03, p 0.47)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L6 o c4 | H22 | a//10 {0} x b//10 {0, 3..5, 9..10}; a//10 {8} x b//10 {7..9}; a//10 {9} x b//10 {3..4, 6..9}; a//10 {10} x b//10 {0} (-) | 0.74 | 1.000 |
| L6 o c310 | H7 | a//10 {1} x b//10 {2}; a//10 {2} x b//10 {3..5}; a//10 {3} x b//10 {4..7, 10}; a//10 {4} x b//10 {5..10}; a//10 {5} x b//10 {6..10}; a//10 {6} x b//10 {7..10}; a//10 {7} x b//10 {8..10}; a//10 {8} x b//10 {9..10}; a//10 {9} x b//10 {10} (+) | 0.81 | 0.345 |

</details>

</details>

<details><summary><b>4s-645</b> `tens(a,b)` @ `=` (sub) — block code; 2 comps, L6 L7; tells apart 3/121 classes (best member 1); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.41 |
| classes told apart: joint / best code / best member | 3 /  / 1 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.08 / 1.03 / 0.70 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.78 |
| decoding acc. joint / best code / best member (chance) | 0.12 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.97 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L5 MLP |
| CKA(arrangement before, joint write) | 0.47 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.01 |
| best source position (CKA) | `op` (0.44) |

<details><summary>codes and components</summary>

**code 4s-645.0 (L6 L7): 2 comps, tells apart 3/121, coverage 0.41, overlap 1.08 (random 1.00, p 0.70)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L6 o c85 | H24 | a//10 {0} x b//10 {3..4, 10}; a//10 {2, 5} x b//10 {4}; a//10 {7} x b//10 {5..7}; a//10 {8} x b//10 {4..10}; a//10 {9} x b//10 {3..10}; a//10 {10} x b//10 {6..10} (-) | 0.81 | 1.000 |
| L7 o c3 | H5 | a//10 {0} x b//10 {3..5}; a//10 {1} x b//10 {4}; a//10 {2} x b//10 {4..6}; a//10 {3..4} x b//10 {0}; a//10 {5} x b//10 {0..1, 10}; a//10 {6..7} x b//10 {0..1}; a//10 {8..9} x b//10 {0..2}; a//10 {10} x b//10 {0..2, 9} (+) | 0.67 | 1.000 |

</details>

</details>

<details><summary><b>4s-646</b> `tens(a,b)` @ `=` (sub) — block code; 5 comps, L6; tells apart 10/121 classes (best member 3); on add: 4a-539 (member overlap 0.11)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.57 |
| classes told apart: joint / best code / best member | 10 /  / 3 (of 121) |
| members whose removal merges classes | 0.80 |
| support overlap (1 = tiling) / random sets / p | 1.78 / 1.31 / 0.97 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.86 |
| decoding acc. joint / best code / best member (chance) | 0.21 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 6 / 6 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L6 attn |
| CKA(arrangement before, joint write) | 0.65 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.09 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.21 |

<details><summary>codes and components</summary>

**code 4s-646.0 (L6): 5 comps, tells apart 10/121, coverage 0.57, overlap 1.78 (random 1.30, p 0.97)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L6 down c17 | MLP | a//10 {0, 5..7} x b//10 {0}; a//10 {10} x b//10 {0..2} (+) | 0.60 | 1.000 |
| L6 down c1 | MLP | a//10 {0} x b//10 {0..10}; a//10 {1} x b//10 {0, 4..10}; a//10 {2} x b//10 {0, 8..10}; a//10 {3..7, 10} x b//10 {0}; a//10 {9} x b//10 {0..1} (-) | 0.86 | 0.417 |
| L6 down c11 | MLP | a//10 {0} x b//10 {0..6, 8, 10}; a//10 {1} x b//10 {0}; a//10 {10} x b//10 {0..2, 10} (-) | 0.75 | 1.000 |
| L6 down c47 | MLP | a//10 {0} x b//10 {1..10}; a//10 {1} x b//10 {2..10}; a//10 {2} x b//10 {4..10} (+) | 0.91 | 0.255 |
| L6 down c0 | MLP | a//10 {0} x b//10 {2..10}; a//10 {1} x b//10 {7}; a//10 {2} x b//10 {3}; a//10 {3} x b//10 {3..5}; a//10 {4} x b//10 {4..6}; a//10 {5} x b//10 {5..8}; a//10 {6} x b//10 {6..9}; a//10 {7} x b//10 {0, 8..10}; a//10 {8} x b//10 {8..10}; a//10 {9} x b//10 {0, 10}; a//10 {10} x b//10 {0..10} (-) | 0.76 | 1.000 |

</details>

</details>

<details><summary><b>4s-647</b> `tens(a,b)` @ `=` (sub) — single component; 1 comps, L7; tells apart 3/121 classes (best member 3); on add: 4a-540 (member overlap 0.07)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.17 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.57 |
| decoding acc. joint / best code / best member (chance) | 0.05 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L6 MLP |
| CKA(arrangement before, joint write) | 0.50 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.01 |
| best source position (CKA) | `b` (0.40) |

<details><summary>codes and components</summary>

**code 4s-647.0 (L7): 1 comps, tells apart 3/121, coverage 0.17, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L7 o c11 | H23 | a//10 {0} x b//10 {4..9}; a//10 {1..2} x b//10 {2}; a//10 {5} x b//10 {6..7}; a//10 {6} x b//10 {6..8}; a//10 {7} x b//10 {7..10}; a//10 {8..9} x b//10 {9..10} (+) | 0.56 | 0.894 |

</details>

</details>

<details><summary><b>4s-648</b> `tens(a,b)` @ `=` (sub) — single component; 1 comps, L9; tells apart 3/121 classes (best member 3); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.12 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 121) |
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
| input point | L8 MLP |
| CKA(arrangement before, joint write) | 0.22 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.09 |
| write energy / code energy before | 0.00 |
| best source position (CKA) | `b` (0.21) |

<details><summary>codes and components</summary>

**code 4s-648.0 (L9): 1 comps, tells apart 3/121, coverage 0.12, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L9 o c183 | H11 | a//10 {3} x b//10 {10}; a//10 {4} x b//10 {0, 10}; a//10 {5..6} x b//10 {0}; a//10 {7} x b//10 {0..1, 6}; a//10 {8} x b//10 {6..7}; a//10 {9} x b//10 {6..7, 9..10}; a//10 {10} x b//10 {9} (+) | 0.51 | 1.000 |

</details>

</details>

<details><summary><b>4s-649</b> `tens(a,b)` @ `=` (sub) — copy from `b`; 1 comps, L10; tells apart 3/121 classes (best member 3); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.02 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.56 |
| decoding acc. joint / best code / best member (chance) | 0.05 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L9 MLP |
| CKA(arrangement before, joint write) | 0.48 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.14 |
| write energy / code energy before | 0.00 |
| best source position (CKA) | `b` (0.51) |

<details><summary>codes and components</summary>

**code 4s-649.0 (L10): 1 comps, tells apart 3/121, coverage 0.02, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L10 o c54 | H15 | a//10 {9} x b//10 {9..10}; a//10 {10} x b//10 {0} (+) | 0.56 | 0.998 |

</details>

</details>

<details><summary><b>4s-650</b> `tens(a,b)` @ `=` (sub) — block code; 4 comps, L10; tells apart 13/121 classes (best member 4); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.81 |
| classes told apart: joint / best code / best member | 13 /  / 4 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.12 / 1.18 / 0.33 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.84 |
| decoding acc. joint / best code / best member (chance) | 0.21 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 6 / 3 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L10 attn |
| CKA(arrangement before, joint write) | 0.88 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.13 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.30 |

<details><summary>codes and components</summary>

**code 4s-650.0 (L10): 4 comps, tells apart 13/121, coverage 0.81, overlap 1.12 (random 1.21, p 0.24)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L10 down c611 | MLP | a//10 {0} x b//10 {0, 3..10}; a//10 {1} x b//10 {0}; a//10 {2} x b//10 {0, 3}; a//10 {3} x b//10 {0, 4..7}; a//10 {4} x b//10 {0, 5..9}; a//10 {5} x b//10 {0, 6..9}; a//10 {6} x b//10 {0, 6..10}; a//10 {7} x b//10 {0..1, 6..10}; a//10 {8} x b//10 {0..1, 8..10}; a//10 {9} x b//10 {0..1, 9..10}; a//10 {10} x b//10 {0..2, 10} (+) | 0.87 | 0.906 |
| L10 down c252 | MLP | a//10 {1} x b//10 {5..10}; a//10 {6} x b//10 {6}; a//10 {7} x b//10 {7}; a//10 {9} x b//10 {9..10} (-) | 0.73 | 0.984 |
| L10 down c385 | MLP | a//10 {2} x b//10 {1}; a//10 {3} x b//10 {1..2}; a//10 {4} x b//10 {1..3}; a//10 {5} x b//10 {1..4}; a//10 {6} x b//10 {1..5}; a//10 {7} x b//10 {1..7}; a//10 {8..9} x b//10 {1..8}; a//10 {10} x b//10 {3..8} (+) | 0.84 | 0.527 |
| L10 down c9 | MLP | a//10 {9} x b//10 {9..10} (+) | 0.66 | 1.000 |

</details>

</details>

<details><summary><b>4s-651</b> `tens(a,b)` @ `=` (sub) — single component; 1 comps, L13; tells apart 4/121 classes (best member 4); on add: 4a-543 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.08 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.68 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L12 MLP |
| CKA(arrangement before, joint write) | 0.45 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.03 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.00 |
| best source position (CKA) | `b` (0.36) |

<details><summary>codes and components</summary>

**code 4s-651.0 (L13): 1 comps, tells apart 4/121, coverage 0.08, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 o c0 | H16 | a//10 {4..6} x b//10 {0}; a//10 {10} x b//10 {4..10} (+) | 0.67 | 1.000 |

</details>

</details>

<details><summary><b>4s-652</b> `tens(a,b)` @ `=` (sub) — block code, copy from `b`; 2 comps, L13; tells apart 7/121 classes (best member 1); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.64 |
| classes told apart: joint / best code / best member | 7 /  / 1 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.17 / 1.03 / 0.84 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.68 |
| decoding acc. joint / best code / best member (chance) | 0.09 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.90 |
| consumers / read jointly | 3 / 1 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L12 MLP |
| CKA(arrangement before, joint write) | 0.67 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.13 |
| share of the write inside the old arrangement's span | 0.18 |
| write energy / code energy before | 0.06 |
| best source position (CKA) | `b` (0.59) |

<details><summary>codes and components</summary>

**code 4s-652.0 (L13): 2 comps, tells apart 7/121, coverage 0.64, overlap 1.17 (random 1.04, p 0.83)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 o c145 | H7 | a//10 {0} x b//10 {0, 2..10}; a//10 {1} x b//10 {0, 10}; a//10 {2} x b//10 {10}; a//10 {3} x b//10 {5}; a//10 {4} x b//10 {5..6}; a//10 {5} x b//10 {0, 6..7}; a//10 {6} x b//10 {7..8}; a//10 {7} x b//10 {8..9}; a//10 {10} x b//10 {9..10} (-) | 0.70 | 0.998 |
| L13 o c213 | H7 | a//10 {0} x b//10 {0}; a//10 {1} x b//10 {5..9}; a//10 {2} x b//10 {2..3}; a//10 {3} x b//10 {2..6}; a//10 {4} x b//10 {0, 3, 5..6}; a//10 {5} x b//10 {0..1, 3..4, 6..8}; a//10 {6} x b//10 {0..2, 5..8}; a//10 {7} x b//10 {0..2, 5..10}; a//10 {8} x b//10 {0..3, 6..10}; a//10 {9} x b//10 {0..5, 9..10}; a//10 {10} x b//10 {1..2, 4..10} (+) | 0.69 | 0.601 |

</details>

</details>

<details><summary><b>4s-653</b> `tens(a,b)` @ `=` (sub) — single component; 1 comps, L14; tells apart 1/121 classes (best member 1); on add: 4a-544 (member overlap 0.33)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.23 |
| classes told apart: joint / best code / best member | 1 /  / 1 (of 121) |
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
| input point | L13 MLP |
| CKA(arrangement before, joint write) | 0.34 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.00 |
| best source position (CKA) | `a` (0.33) |

<details><summary>codes and components</summary>

**code 4s-653.0 (L14): 1 comps, tells apart 1/121, coverage 0.23, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L14 o c59 | H26 | a//10 {2} x b//10 {4..5, 7}; a//10 {3} x b//10 {5, 7..9}; a//10 {4} x b//10 {5}; a//10 {6} x b//10 {5..6}; a//10 {7} x b//10 {6..7}; a//10 {8} x b//10 {0, 6..8}; a//10 {9} x b//10 {2..5, 8..10}; a//10 {10} x b//10 {3..4, 8..10} (+) | 0.62 | 1.000 |

</details>

</details>

<details><summary><b>4s-654</b> `tens(a,b)` @ `=` (sub) — single component; 1 comps, L14; tells apart 1/121 classes (best member 1); on add: 4a-545 (member overlap 0.50)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.19 |
| classes told apart: joint / best code / best member | 1 /  / 1 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.65 |
| decoding acc. joint / best code / best member (chance) | 0.05 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L13 MLP |
| CKA(arrangement before, joint write) | 0.59 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.09 |
| write energy / code energy before | 0.01 |
| best source position (CKA) | `b` (0.33) |

<details><summary>codes and components</summary>

**code 4s-654.0 (L14): 1 comps, tells apart 1/121, coverage 0.19, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L14 o c84 | H31 | a//10 {0} x b//10 {0..5, 8}; a//10 {4} x b//10 {0}; a//10 {5} x b//10 {7..8}; a//10 {6} x b//10 {5, 8}; a//10 {7} x b//10 {5..6, 9..10}; a//10 {8} x b//10 {6..7}; a//10 {10} x b//10 {5..9} (-) | 0.64 | 1.000 |

</details>

</details>

<details><summary><b>4s-656</b> `tens(a,b)` @ `=` (sub) — block code, copy from `a`; 2 comps, L15 L16; tells apart 4/121 classes (best member 1); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.48 |
| classes told apart: joint / best code / best member | 4 /  / 1 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.10 / 1.03 / 0.76 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.87 |
| decoding acc. joint / best code / best member (chance) | 0.16 /  / 0.06 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.04 |
| consumers / read jointly | 2 / 1 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L14 MLP |
| CKA(arrangement before, joint write) | 0.51 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.12 |
| write energy / code energy before | 0.06 |
| best source position (CKA) | `a` (0.59) |

<details><summary>codes and components</summary>

**code 4s-656.0 (L15 L16): 2 comps, tells apart 4/121, coverage 0.48, overlap 1.10 (random 1.05, p 0.69)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 o c446 | H18 | a//10 {0} x b//10 {0..3, 7..8, 10}; a//10 {1..2} x b//10 {0, 10}; a//10 {3, 7} x b//10 {0, 5..6}; a//10 {4, 8} x b//10 {0, 6}; a//10 {5} x b//10 {0, 6..8}; a//10 {6} x b//10 {0, 5}; a//10 {9} x b//10 {0}; a//10 {10} x b//10 {0, 4..7, 10} (+) | 0.73 | 0.999 |
| L16 o c160 | H22 | a//10 {7, 9} x b//10 {0..10}; a//10 {8} x b//10 {0, 4..10} (-) | 0.94 | 0.739 |

</details>

</details>

<details><summary><b>4s-655</b> `tens(a,b)` @ `=` (sub) — block code, copy from `b`; 10 comps, L15; tells apart 41/121 classes (best member 7); on add: 4a-545 (member overlap 0.09)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.44 |
| classes told apart: joint / best code / best member | 41 /  / 7 (of 121) |
| members whose removal merges classes | 0.40 |
| support overlap (1 = tiling) / random sets / p | 1.85 / 1.78 / 0.59 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.79 |
| decoding acc. joint / best code / best member (chance) | 0.29 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.20 |
| consumers / read jointly | 15 / 12 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L14 MLP |
| CKA(arrangement before, joint write) | 0.64 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.12 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.09 |
| best source position (CKA) | `b` (0.58) |

<details><summary>codes and components</summary>

**code 4s-655.0 (L15): 10 comps, tells apart 41/121, coverage 0.44, overlap 1.85 (random 1.80, p 0.56)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 o c191 | H13 | a//10 {0, 4, 10} x b//10 {1}; a//10 {1..3} x b//10 {1..2} (-) | 0.75 | 0.146 |
| L15 o c75 | H13 | a//10 {0..10} x b//10 {0} (-) | 0.88 | 0.413 |
| L15 o c30 | H13 | a//10 {2..3} x b//10 {1..2}; a//10 {4, 10} x b//10 {1..3}; a//10 {5} x b//10 {2..3}; a//10 {6} x b//10 {3} (+) | 0.75 | 0.459 |
| L15 o c77 | H13 | a//10 {3..4} x b//10 {3}; a//10 {5..7, 10} x b//10 {3..4}; a//10 {8} x b//10 {4} (-) | 0.75 | 0.195 |
| L15 o c58 | H13 | a//10 {4..5, 7, 9} x b//10 {7}; a//10 {8, 10} x b//10 {6..7} (-) | 0.80 | 0.241 |
| L15 o c17 | H13 | a//10 {5, 9} x b//10 {7}; a//10 {6} x b//10 {6}; a//10 {7..8, 10} x b//10 {6..7} (-) | 0.72 | 0.300 |
| L15 o c151 | H13 | a//10 {5} x b//10 {4}; a//10 {6..8, 10} x b//10 {4..5} (+) | 0.66 | 0.070 |
| L15 o c19 | H13 | a//10 {5} x b//10 {4}; a//10 {6..8} x b//10 {4..5}; a//10 {10} x b//10 {4..6} (+) | 0.75 | 0.170 |
| L15 o c105 | H13 | a//10 {6} x b//10 {5}; a//10 {7..8, 10} x b//10 {5..6} (+) | 0.72 | 0.090 |
| L15 o c119 | H13 | a//10 {6} x b//10 {6}; a//10 {7} x b//10 {6..7}; a//10 {8} x b//10 {6..9}; a//10 {9} x b//10 {9}; a//10 {10} x b//10 {7..9} (+) | 0.78 | 0.472 |

</details>

</details>

<details><summary><b>4s-657</b> `tens(a,b)` @ `=` (sub) — tiling, 7 codes of the same shape; 92 comps, L15 L16 L17 L19 L20 L22 L31; tells apart 73/121 classes (best member 8); on add: 4a-547 (member overlap 0.06)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 73 / 81 / 8 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p | 11.23 / 11.55 / 0.34 |
| mean CKA between its codes | 0.75 |
| purity of the joint write (per prompt) | 0.74 |
| decoding acc. joint / best code / best member (chance) | 0.44 / 0.49 / 0.06 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.17 |
| consumers / read jointly | 464 / 367 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 attn |
| CKA(arrangement before, joint write) | 0.79 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.06 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 7.58 |

<details><summary>codes and components</summary>

**code 4s-657.0 (L15): 13 comps, tells apart 59/121, coverage 0.74, overlap 2.25 (random 2.10, p 0.66)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c90 | MLP | a//10 {0} x b//10 {0, 2..10}; a//10 {1} x b//10 {0, 7..8, 10}; a//10 {2} x b//10 {0}; a//10 {3} x b//10 {0, 4}; a//10 {4} x b//10 {0..1, 6..7}; a//10 {5..6} x b//10 {0..1, 6..9}; a//10 {7..8} x b//10 {0..1, 6..10}; a//10 {9} x b//10 {0..2, 9..10}; a//10 {10} x b//10 {0..2, 7..9} (-) | 0.84 | 0.627 |
| L15 down c6 | MLP | a//10 {0} x b//10 {0..10}; a//10 {1} x b//10 {1..2, 7..8, 10}; a//10 {2} x b//10 {10}; a//10 {3, 6} x b//10 {5}; a//10 {7} x b//10 {5, 9..10}; a//10 {10} x b//10 {4..10} (-) | 0.55 | 1.000 |
| L15 down c80 | MLP | a//10 {0} x b//10 {1}; a//10 {1..6} x b//10 {1..2} (+) | 0.84 | 0.248 |
| L15 down c535 | MLP | a//10 {0} x b//10 {2..10}; a//10 {1} x b//10 {5, 8, 10} (-) | 0.79 | 0.134 |
| L15 down c76 | MLP | a//10 {0} x b//10 {7..9}; a//10 {1} x b//10 {7..10}; a//10 {2} x b//10 {9} (-) | 0.62 | 0.084 |
| L15 down c28 | MLP | a//10 {3, 10} x b//10 {2..3}; a//10 {4..8} x b//10 {3} (+) | 0.70 | 0.135 |
| L15 down c41 | MLP | a//10 {3} x b//10 {2}; a//10 {4..9} x b//10 {7}; a//10 {10} x b//10 {2, 7..8} (-) | 0.73 | 0.333 |
| L15 down c19 | MLP | a//10 {4} x b//10 {3}; a//10 {5} x b//10 {3..4}; a//10 {10} x b//10 {3..4, 8..9} (-) | 0.67 | 0.248 |
| L15 down c4 | MLP | a//10 {5, 9} x b//10 {7..8}; a//10 {6} x b//10 {6..8}; a//10 {7..8, 10} x b//10 {6..9} (-) | 0.88 | 0.433 |
| L15 down c10 | MLP | a//10 {5} x b//10 {4}; a//10 {6} x b//10 {5..6}; a//10 {7} x b//10 {6..7}; a//10 {8} x b//10 {6..9}; a//10 {9} x b//10 {9..10}; a//10 {10} x b//10 {5..9} (+) | 0.72 | 0.933 |
| L15 down c35 | MLP | a//10 {7..8} x b//10 {6}; a//10 {10} x b//10 {6..7} (+) | 0.63 | 0.166 |
| L15 down c2 | MLP | a//10 {7} x b//10 {6}; a//10 {8} x b//10 {7..8}; a//10 {10} x b//10 {6..9} (+) | 0.54 | 0.188 |
| L15 down c29 | MLP | a//10 {7} x b//10 {9}; a//10 {8..9} x b//10 {9..10}; a//10 {10} x b//10 {8..10} (-) | 0.82 | 0.286 |

**code 4s-657.1 (L16): 16 comps, tells apart 52/121, coverage 0.78, overlap 2.07 (random 2.44, p 0.17)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c161 | MLP | a//10 {0} x b//10 {0, 2..9}; a//10 {1..10} x b//10 {0} (+) | 0.82 | 0.165 |
| L16 down c67 | MLP | a//10 {0} x b//10 {0..2}; a//10 {1} x b//10 {1..2}; a//10 {9} x b//10 {9..10}; a//10 {10} x b//10 {10} (+) | 0.57 | 0.115 |
| L16 down c288 | MLP | a//10 {0} x b//10 {7..9}; a//10 {1} x b//10 {5..9}; a//10 {2} x b//10 {8..9} (-) | 0.73 | 0.109 |
| L16 down c130 | MLP | a//10 {10} x b//10 {2..5} (+) | 0.91 | 0.121 |
| L16 down c61 | MLP | a//10 {10} x b//10 {6..8} (+) | 0.55 | 0.016 |
| L16 down c77 | MLP | a//10 {2..5} x b//10 {0}; a//10 {6..10} x b//10 {0..1} (-) | 0.82 | 0.228 |
| L16 down c17 | MLP | a//10 {2} x b//10 {1..2}; a//10 {3} x b//10 {1..3}; a//10 {4} x b//10 {3} (+) | 0.51 | 0.054 |
| L16 down c16 | MLP | a//10 {4, 8} x b//10 {0..3}; a//10 {5..6, 9} x b//10 {1..3}; a//10 {7} x b//10 {1..4} (+) | 0.71 | 0.277 |
| L16 down c25 | MLP | a//10 {4} x b//10 {3}; a//10 {5} x b//10 {3..4}; a//10 {6..7} x b//10 {3..5} (-) | 0.82 | 0.249 |
| L16 down c71 | MLP | a//10 {4} x b//10 {3}; a//10 {5} x b//10 {3..4}; a//10 {6} x b//10 {4..5, 8}; a//10 {7, 10} x b//10 {4..9}; a//10 {8} x b//10 {5..9}; a//10 {9} x b//10 {7..8} (+) | 0.50 | 0.615 |
| L16 down c152 | MLP | a//10 {4} x b//10 {5..8}; a//10 {5} x b//10 {6..8}; a//10 {6} x b//10 {7..9}; a//10 {7} x b//10 {8..10}; a//10 {8} x b//10 {9..10}; a//10 {10} x b//10 {8} (-) | 0.64 | 0.155 |
| L16 down c53 | MLP | a//10 {5} x b//10 {7}; a//10 {6, 10} x b//10 {6..8}; a//10 {7..8} x b//10 {6..10}; a//10 {9} x b//10 {7..10} (-) | 0.79 | 0.327 |
| L16 down c29 | MLP | a//10 {6} x b//10 {4}; a//10 {7} x b//10 {3..5}; a//10 {8, 10} x b//10 {3..6} (+) | 0.84 | 0.311 |
| L16 down c10 | MLP | a//10 {6} x b//10 {5}; a//10 {7..8} x b//10 {5..7}; a//10 {10} x b//10 {6} (+) | 0.85 | 0.469 |
| L16 down c106 | MLP | a//10 {6} x b//10 {6..8}; a//10 {7} x b//10 {6..9}; a//10 {8} x b//10 {7..8} (-) | 0.76 | 0.163 |
| L16 down c45 | MLP | a//10 {7..8} x b//10 {6..7}; a//10 {9} x b//10 {9..10}; a//10 {10} x b//10 {6..9} (-) | 0.75 | 0.979 |

**code 4s-657.2 (L17): 13 comps, tells apart 69/121, coverage 0.84, overlap 1.92 (random 2.09, p 0.23)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 down c503 | MLP | a//10 {0..1} x b//10 {4..9}; a//10 {10} x b//10 {0} (-) | 0.71 | 0.131 |
| L17 down c3 | MLP | a//10 {0} x b//10 {1..10}; a//10 {1, 9} x b//10 {10}; a//10 {2, 10} x b//10 {4, 10}; a//10 {3} x b//10 {4..5}; a//10 {4} x b//10 {3}; a//10 {5} x b//10 {4}; a//10 {6, 8} x b//10 {2, 6}; a//10 {7} x b//10 {6..7, 10} (+) | 0.57 | 1.000 |
| L17 down c96 | MLP | a//10 {0} x b//10 {1..10}; a//10 {1..2} x b//10 {3..10}; a//10 {3} x b//10 {4, 8} (+) | 0.89 | 0.394 |
| L17 down c29 | MLP | a//10 {0} x b//10 {2..3, 5..7}; a//10 {3} x b//10 {0}; a//10 {4} x b//10 {0..1}; a//10 {5..7} x b//10 {0..2}; a//10 {8..9} x b//10 {0..3} (+) | 0.71 | 0.350 |
| L17 down c40 | MLP | a//10 {10} x b//10 {6..8} (+) | 0.62 | 0.154 |
| L17 down c23 | MLP | a//10 {2} x b//10 {0..1}; a//10 {6} x b//10 {5}; a//10 {7} x b//10 {5..6}; a//10 {8} x b//10 {6} (-) | 0.70 | 0.216 |
| L17 down c10 | MLP | a//10 {2} x b//10 {1..2}; a//10 {3} x b//10 {1..2, 7}; a//10 {4} x b//10 {2, 7}; a//10 {5..6, 9} x b//10 {7}; a//10 {7..8} x b//10 {6..7}; a//10 {10} x b//10 {2, 6..8} (+) | 0.75 | 0.461 |
| L17 down c33 | MLP | a//10 {3} x b//10 {1..3}; a//10 {7} x b//10 {5..6}; a//10 {8} x b//10 {4..8} (-) | 0.71 | 0.264 |
| L17 down c11 | MLP | a//10 {3} x b//10 {2}; a//10 {5..7} x b//10 {0..1, 10}; a//10 {8} x b//10 {0..1}; a//10 {10} x b//10 {1, 5, 10} (+) | 0.73 | 0.269 |
| L17 down c13 | MLP | a//10 {4} x b//10 {6}; a//10 {5} x b//10 {4, 6..8}; a//10 {6} x b//10 {5..9}; a//10 {7} x b//10 {5..10}; a//10 {8} x b//10 {6..10}; a//10 {9} x b//10 {10}; a//10 {10} x b//10 {7..8} (-) | 0.71 | 0.399 |
| L17 down c107 | MLP | a//10 {7..8} x b//10 {5..6}; a//10 {9} x b//10 {6}; a//10 {10} x b//10 {4..7} (+) | 0.80 | 0.148 |
| L17 down c15 | MLP | a//10 {7} x b//10 {0, 2..5}; a//10 {10} x b//10 {2..4} (+) | 0.68 | 0.089 |
| L17 down c26 | MLP | a//10 {9} x b//10 {8..10}; a//10 {10} x b//10 {4..10} (+) | 0.70 | 0.048 |

**code 4s-657.3 (L19): 17 comps, tells apart 77/121, coverage 0.67, overlap 2.17 (random 2.51, p 0.17)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L19 down c94 | MLP | a//10 {0} x b//10 {0, 2}; a//10 {8..9} x b//10 {0} (+) | 0.68 | 0.091 |
| L19 down c82 | MLP | a//10 {0} x b//10 {0} (+) | 0.63 | 0.015 |
| L19 down c454 | MLP | a//10 {0} x b//10 {0} (-) | 0.65 | 0.011 |
| L19 down c1 | MLP | a//10 {0} x b//10 {0}; a//10 {4} x b//10 {5, 9}; a//10 {5} x b//10 {9..10}; a//10 {7..8} x b//10 {6..7}; a//10 {9} x b//10 {0, 4}; a//10 {10} x b//10 {0..5} (-) | 0.58 | 1.000 |
| L19 down c367 | MLP | a//10 {0} x b//10 {1..10}; a//10 {1} x b//10 {4..10} (-) | 0.86 | 0.196 |
| L19 down c8 | MLP | a//10 {0} x b//10 {4..5, 7..8}; a//10 {1} x b//10 {4..9}; a//10 {2} x b//10 {2, 8}; a//10 {3} x b//10 {3}; a//10 {5} x b//10 {4..5}; a//10 {6} x b//10 {4..8}; a//10 {7} x b//10 {1..2, 5..9}; a//10 {8} x b//10 {6..9}; a//10 {9} x b//10 {8..10}; a//10 {10} x b//10 {4..10} (-) | 0.75 | 0.870 |
| L19 down c3 | MLP | a//10 {0} x b//10 {8..10}; a//10 {9} x b//10 {0..1}; a//10 {10} x b//10 {0..6} (-) | 0.62 | 0.985 |
| L19 down c45 | MLP | a//10 {10} x b//10 {2..3, 5..7, 9} (-) | 0.70 | 0.995 |
| L19 down c86 | MLP | a//10 {10} x b//10 {4..5} (+) | 0.79 | 0.006 |
| L19 down c24 | MLP | a//10 {10} x b//10 {4..5} (-) | 0.69 | 0.185 |
| L19 down c11 | MLP | a//10 {2} x b//10 {1}; a//10 {3} x b//10 {2}; a//10 {7} x b//10 {6}; a//10 {8} x b//10 {7} (-) | 0.75 | 0.441 |
| L19 down c20 | MLP | a//10 {4} x b//10 {3}; a//10 {5} x b//10 {3..4}; a//10 {6} x b//10 {4..5}; a//10 {7} x b//10 {5}; a//10 {9} x b//10 {8}; a//10 {10} x b//10 {3..4, 8..9} (-) | 0.65 | 0.255 |
| L19 down c17 | MLP | a//10 {5..7} x b//10 {0}; a//10 {8..9} x b//10 {0..1}; a//10 {10} x b//10 {0..2} (+) | 0.74 | 0.113 |
| L19 down c28 | MLP | a//10 {5} x b//10 {3}; a//10 {6} x b//10 {3..4}; a//10 {7} x b//10 {3..5}; a//10 {8} x b//10 {5..6} (-) | 0.75 | 0.198 |
| L19 down c36 | MLP | a//10 {5} x b//10 {4}; a//10 {6} x b//10 {5..6}; a//10 {7} x b//10 {6}; a//10 {8} x b//10 {7..8}; a//10 {10} x b//10 {6..9} (-) | 0.51 | 0.173 |
| L19 down c278 | MLP | a//10 {5} x b//10 {8..10}; a//10 {6..8, 10} x b//10 {7..10}; a//10 {9} x b//10 {5..10} (-) | 0.87 | 0.539 |
| L19 down c0 | MLP | a//10 {8} x b//10 {5..6}; a//10 {9} x b//10 {7}; a//10 {10} x b//10 {2, 7} (-) | 0.69 | 0.596 |

**code 4s-657.4 (L20): 10 comps, tells apart 81/121, coverage 0.76, overlap 1.59 (random 1.81, p 0.19)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 down c79 | MLP | a//10 {0, 6} x b//10 {3..10}; a//10 {1, 8, 10} x b//10 {4..10}; a//10 {2} x b//10 {8..10}; a//10 {4} x b//10 {0}; a//10 {5} x b//10 {3..4, 7, 10}; a//10 {7} x b//10 {4..7, 9..10}; a//10 {9} x b//10 {0, 9..10} (-) | 0.75 | 0.644 |
| L20 down c266 | MLP | a//10 {1} x b//10 {3}; a//10 {2} x b//10 {3..4}; a//10 {3} x b//10 {0..6, 10}; a//10 {4} x b//10 {0..6} (+) | 0.85 | 0.288 |
| L20 down c29 | MLP | a//10 {3} x b//10 {1, 7}; a//10 {4} x b//10 {2, 7}; a//10 {5} x b//10 {8}; a//10 {8} x b//10 {6..7}; a//10 {9} x b//10 {7..8}; a//10 {10} x b//10 {2..3, 8} (-) | 0.66 | 0.530 |
| L20 down c73 | MLP | a//10 {5..6, 8} x b//10 {0..1}; a//10 {7} x b//10 {0..2}; a//10 {10} x b//10 {1, 3} (+) | 0.67 | 0.122 |
| L20 down c181 | MLP | a//10 {5} x b//10 {7}; a//10 {6} x b//10 {7..8}; a//10 {7} x b//10 {8..10}; a//10 {8} x b//10 {9..10} (+) | 0.74 | 0.130 |
| L20 down c2 | MLP | a//10 {7} x b//10 {3..4}; a//10 {8} x b//10 {4..5}; a//10 {10} x b//10 {5..6} (-) | 0.72 | 0.998 |
| L20 down c32 | MLP | a//10 {7} x b//10 {4..7, 9..10}; a//10 {8} x b//10 {5..9}; a//10 {9} x b//10 {7..8}; a//10 {10} x b//10 {3..9} (-) | 0.68 | 0.351 |
| L20 down c14 | MLP | a//10 {8} x b//10 {0, 5}; a//10 {9} x b//10 {0..1}; a//10 {10} x b//10 {0..1, 6} (+) | 0.60 | 0.121 |
| L20 down c5 | MLP | a//10 {8} x b//10 {5..7}; a//10 {10} x b//10 {4..8} (-) | 0.73 | 1.000 |
| L20 down c305 | MLP | a//10 {9} x b//10 {9..10} (+) | 0.56 | 0.011 |

**code 4s-657.5 (L22): 12 comps, tells apart 52/121, coverage 0.86, overlap 1.61 (random 2.06, p 0.05)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 down c107 | MLP | a//10 {0..2} x b//10 {0..2} (-) | 0.66 | 0.076 |
| L22 down c3 | MLP | a//10 {0} x b//10 {0, 5..10}; a//10 {1} x b//10 {0..1}; a//10 {2} x b//10 {2}; a//10 {6} x b//10 {6}; a//10 {7} x b//10 {7}; a//10 {9} x b//10 {9..10}; a//10 {10} x b//10 {2..8, 10} (-) | 0.59 | 1.000 |
| L22 down c267 | MLP | a//10 {0} x b//10 {0..2} (+) | 0.67 | 0.071 |
| L22 down c0 | MLP | a//10 {0} x b//10 {2}; a//10 {1} x b//10 {2..4}; a//10 {2} x b//10 {3..5}; a//10 {3} x b//10 {4..5}; a//10 {4} x b//10 {5}; a//10 {5} x b//10 {0, 9..10}; a//10 {10} x b//10 {4, 6..7, 10} (+) | 0.56 | 1.000 |
| L22 down c232 | MLP | a//10 {0} x b//10 {3..9}; a//10 {1} x b//10 {3..10}; a//10 {2} x b//10 {4..5, 7..9}; a//10 {3} x b//10 {2}; a//10 {4} x b//10 {3}; a//10 {5} x b//10 {3..4}; a//10 {6} x b//10 {3..6, 10}; a//10 {7} x b//10 {4..7, 10}; a//10 {8} x b//10 {4..10}; a//10 {9} x b//10 {6..10}; a//10 {10} x b//10 {0, 3..10} (+) | 0.76 | 0.763 |
| L22 down c29 | MLP | a//10 {2..5} x b//10 {10} (-) | 0.58 | 0.009 |
| L22 down c160 | MLP | a//10 {2} x b//10 {4}; a//10 {3} x b//10 {4..6}; a//10 {4} x b//10 {6..7}; a//10 {5} x b//10 {6..8}; a//10 {6} x b//10 {7..9}; a//10 {7} x b//10 {8..10}; a//10 {8} x b//10 {9..10} (-) | 0.75 | 0.369 |
| L22 down c38 | MLP | a//10 {4} x b//10 {0} (-) | 0.52 | 0.019 |
| L22 down c55 | MLP | a//10 {4} x b//10 {10}; a//10 {5} x b//10 {2, 10}; a//10 {6} x b//10 {1..3, 10}; a//10 {7} x b//10 {2..4}; a//10 {8} x b//10 {1..5}; a//10 {9} x b//10 {0..7}; a//10 {10} x b//10 {0} (+) | 0.63 | 0.320 |
| L22 down c254 | MLP | a//10 {6} x b//10 {0, 10}; a//10 {7} x b//10 {0} (-) | 0.68 | 0.024 |
| L22 down c297 | MLP | a//10 {6} x b//10 {4}; a//10 {7} x b//10 {5..6}; a//10 {8} x b//10 {6..7}; a//10 {9} x b//10 {7}; a//10 {10} x b//10 {5..8} (-) | 0.69 | 0.193 |
| L22 down c33 | MLP | a//10 {9..10} x b//10 {0} (-) | 0.63 | 0.097 |

**code 4s-657.6 (L31): 11 comps, tells apart 25/121, coverage 0.85, overlap 2.71 (random 1.89, p 0.98)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L31 down c90 | MLP | a//10 {0} x b//10 {0..10}; a//10 {1} x b//10 {3..10}; a//10 {2} x b//10 {7..10}; a//10 {3} x b//10 {10} (+) | 0.71 | 0.532 |
| L31 down c533 | MLP | a//10 {0} x b//10 {0..1}; a//10 {1} x b//10 {1, 4, 6..9}; a//10 {2} x b//10 {0..2, 6, 8}; a//10 {3} x b//10 {2, 9..10}; a//10 {4} x b//10 {3, 9..10}; a//10 {5} x b//10 {0, 3..5}; a//10 {6} x b//10 {4..6}; a//10 {7} x b//10 {5..8}; a//10 {8} x b//10 {0, 6..10}; a//10 {9} x b//10 {8..9}; a//10 {10} x b//10 {2, 4..5, 7..10} (+) | 0.56 | 0.681 |
| L31 down c544 | MLP | a//10 {0} x b//10 {0..2} (+) | 0.54 | 0.129 |
| L31 down c307 | MLP | a//10 {0} x b//10 {0..9}; a//10 {1} x b//10 {0..8}; a//10 {2} x b//10 {0, 3..4}; a//10 {3} x b//10 {0, 4}; a//10 {4} x b//10 {0} (-) | 0.84 | 0.437 |
| L31 down c134 | MLP | a//10 {0} x b//10 {10}; a//10 {9} x b//10 {0, 10}; a//10 {10} x b//10 {0} (+) | 0.56 | 0.102 |
| L31 down c622 | MLP | a//10 {0} x b//10 {3..10}; a//10 {1} x b//10 {4..10}; a//10 {2} x b//10 {0, 7..10}; a//10 {3} x b//10 {2, 10}; a//10 {4} x b//10 {0, 10}; a//10 {5} x b//10 {0, 4}; a//10 {6} x b//10 {4..6}; a//10 {7} x b//10 {0, 5..7}; a//10 {8} x b//10 {6..10}; a//10 {9} x b//10 {8..10}; a//10 {10} x b//10 {1..2} (-) | 0.71 | 0.786 |
| L31 down c595 | MLP | a//10 {0} x b//10 {3..5, 7..10}; a//10 {1} x b//10 {4..10}; a//10 {2} x b//10 {7..10}; a//10 {3} x b//10 {6, 10}; a//10 {5} x b//10 {0, 4, 6}; a//10 {6} x b//10 {1, 6, 8..9}; a//10 {7} x b//10 {0..2, 6..8}; a//10 {8} x b//10 {0..3, 6..9}; a//10 {9} x b//10 {1..4, 10}; a//10 {10} x b//10 {5..6, 8..10} (+) | 0.57 | 0.524 |
| L31 down c45 | MLP | a//10 {0} x b//10 {4..5, 7..9}; a//10 {1} x b//10 {0, 4..9}; a//10 {2} x b//10 {0..1, 4..9}; a//10 {3} x b//10 {0..2}; a//10 {4} x b//10 {0..3}; a//10 {5} x b//10 {3..4}; a//10 {6} x b//10 {0..6}; a//10 {7} x b//10 {5..6}; a//10 {8} x b//10 {4..7}; a//10 {9} x b//10 {9}; a//10 {10} x b//10 {5..9} (+) | 0.67 | 0.696 |
| L31 down c6 | MLP | a//10 {0} x b//10 {4..9}; a//10 {3} x b//10 {0..1, 3}; a//10 {4} x b//10 {0..1}; a//10 {5} x b//10 {1..2, 10}; a//10 {6} x b//10 {1..2, 8}; a//10 {7} x b//10 {1, 8..10}; a//10 {10} x b//10 {3..5} (-) | 0.50 | 1.000 |
| L31 down c474 | MLP | a//10 {0} x b//10 {7}; a//10 {7} x b//10 {0..2}; a//10 {8} x b//10 {0} (+) | 0.60 | 0.127 |
| L31 down c306 | MLP | a//10 {8} x b//10 {6..7}; a//10 {9} x b//10 {9..10}; a//10 {10} x b//10 {7, 9..10} (+) | 0.52 | 0.191 |

</details>

</details>

<details><summary><b>4s-658</b> `tens(a,b)` @ `=` (sub) — block code; 2 comps, L16; tells apart 5/121 classes (best member 5); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.14 |
| classes told apart: joint / best code / best member | 5 /  / 5 (of 121) |
| members whose removal merges classes | 0.50 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.01 / 0.50 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.91 |
| decoding acc. joint / best code / best member (chance) | 0.06 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 5 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 MLP |
| CKA(arrangement before, joint write) | 0.14 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.03 |
| best source position (CKA) | `a` (0.43) |

<details><summary>codes and components</summary>

**code 4s-658.0 (L16): 2 comps, tells apart 5/121, coverage 0.14, overlap 1.00 (random 1.04, p 0.41)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c284 | H21 | a//10 {0} x b//10 {0..1} (-) | 0.71 | 0.042 |
| L16 o c87 | H21 | a//10 {1} x b//10 {0..2}; a//10 {2} x b//10 {0..5, 9..10}; a//10 {3} x b//10 {0..3} (-) | 0.92 | 0.427 |

</details>

</details>

<details><summary><b>4s-659</b> `tens(a,b)` @ `=` (sub) — block code; 3 comps, L17 L18; tells apart 9/121 classes (best member 7); on add: 4a-549 (member overlap 0.20)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.31 |
| classes told apart: joint / best code / best member | 9 /  / 7 (of 121) |
| members whose removal merges classes | 0.67 |
| support overlap (1 = tiling) / random sets / p | 1.05 / 1.13 / 0.28 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.81 |
| decoding acc. joint / best code / best member (chance) | 0.09 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L16 MLP |
| CKA(arrangement before, joint write) | 0.10 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.13 |
| write energy / code energy before | 0.00 |
| best source position (CKA) | `op` (0.15) |

<details><summary>codes and components</summary>

**code 4s-659.0 (L17 L18): 3 comps, tells apart 9/121, coverage 0.31, overlap 1.05 (random 1.12, p 0.26)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 o c17 | H9 | a//10 {0} x b//10 {1..8, 10}; a//10 {1} x b//10 {2, 7, 10}; a//10 {2} x b//10 {3, 10}; a//10 {9} x b//10 {7..8} (-) | 0.70 | 1.000 |
| L18 o c423 | H31 | a//10 {0, 2} x b//10 {7}; a//10 {7} x b//10 {0..3, 7} (+) | 0.61 | 0.107 |
| L18 o c166 | H18 | a//10 {0..4, 10} x b//10 {5}; a//10 {5} x b//10 {0..2, 4..10} (-) | 0.84 | 0.193 |

</details>

</details>

<details><summary><b>4s-660</b> `tens(a,b)` @ `=` (sub) — block code, 2 codes of the same shape; 3 comps, L18 L19; tells apart 9/121 classes (best member 2); on add: 4a-552 (member overlap 0.33)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.29 |
| classes told apart: joint / best code / best member | 9 / 3 / 2 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.29 / 1.13 / 0.82 |
| mean CKA between its codes | 0.71 |
| purity of the joint write (per prompt) | 0.72 |
| decoding acc. joint / best code / best member (chance) | 0.14 / 0.10 / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.95 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L17 MLP |
| CKA(arrangement before, joint write) | 0.30 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.02 |
| best source position (CKA) | `a` (0.34) |

<details><summary>codes and components</summary>

**code 4s-660.0 (L18 L19): 2 comps, tells apart 3/121, coverage 0.24, overlap 1.00 (random 1.03, p 0.48)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 o c442 | H1 | a//10 {0} x b//10 {0..3}; a//10 {1} x b//10 {0..3, 10}; a//10 {2} x b//10 {0..2}; a//10 {3} x b//10 {1} (-) | 0.83 | 0.913 |
| L19 o c23 | H7 | a//10 {0} x b//10 {10}; a//10 {4} x b//10 {0}; a//10 {6} x b//10 {5}; a//10 {8} x b//10 {0..2}; a//10 {9} x b//10 {0..5}; a//10 {10} x b//10 {0, 5..7} (+) | 0.62 | 1.000 |

**code 4s-660.1 (L18): 1 comps, tells apart 1/121, coverage 0.13, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 o c496 | H16 | a//10 {0} x b//10 {0..2}; a//10 {1..2} x b//10 {0..2, 10}; a//10 {4} x b//10 {7..9}; a//10 {5} x b//10 {8}; a//10 {10} x b//10 {9} (-) | 0.69 | 0.920 |

</details>

</details>

<details><summary><b>4s-661</b> `tens(a,b)` @ `=` (sub) — tiling; 10 comps, L18; tells apart 45/121 classes (best member 7); on add: 4a-550 (member overlap 0.67)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.63 |
| classes told apart: joint / best code / best member | 45 /  / 7 (of 121) |
| members whose removal merges classes | 0.90 |
| support overlap (1 = tiling) / random sets / p | 1.21 / 1.79 / 0.00 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.81 |
| decoding acc. joint / best code / best member (chance) | 0.33 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.94 |
| consumers / read jointly | 4 / 3 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L17 MLP |
| CKA(arrangement before, joint write) | 0.30 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.13 |
| share of the write inside the old arrangement's span | 0.17 |
| write energy / code energy before | 0.05 |
| best source position (CKA) | `a` (0.32) |

<details><summary>codes and components</summary>

**code 4s-661.0 (L18): 10 comps, tells apart 45/121, coverage 0.63, overlap 1.21 (random 1.79, p 0.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 o c157 | H30 | a//10 {0, 2..5, 10} x b//10 {10} (+) | 0.86 | 0.008 |
| L18 o c73 | H30 | a//10 {0, 2..5} x b//10 {6}; a//10 {6} x b//10 {0, 6..7, 10} (-) | 0.77 | 0.329 |
| L18 o c120 | H30 | a//10 {0, 2..6, 9} x b//10 {10}; a//10 {10} x b//10 {0, 10} (+) | 0.66 | 0.025 |
| L18 o c107 | H30 | a//10 {0, 2} x b//10 {8..10}; a//10 {1, 3..5} x b//10 {8..9}; a//10 {6..7} x b//10 {9}; a//10 {8} x b//10 {0, 8..10}; a//10 {9} x b//10 {0..10}; a//10 {10} x b//10 {0..1, 10} (-) | 0.82 | 0.423 |
| L18 o c82 | H30 | a//10 {0..2, 4} x b//10 {3}; a//10 {3} x b//10 {3..4} (+) | 0.79 | 0.272 |
| L18 o c42 | H30 | a//10 {0..2, 5} x b//10 {5} (-) | 0.57 | 0.067 |
| L18 o c80 | H30 | a//10 {0..2} x b//10 {2} (-) | 0.63 | 0.070 |
| L18 o c137 | H30 | a//10 {0..4} x b//10 {4} (+) | 0.60 | 0.115 |
| L18 o c355 | H30 | a//10 {0..5, 7} x b//10 {7} (+) | 0.80 | 0.134 |
| L18 o c66 | H30 | a//10 {0} x b//10 {1..2}; a//10 {1..2} x b//10 {0..2}; a//10 {3} x b//10 {1} (-) | 0.89 | 0.357 |

</details>

</details>

<details><summary><b>4s-662</b> `tens(a,b)` @ `=` (sub) — tiling; 21 comps, L18; tells apart 98/121 classes (best member 7); on add: 4a-551 (member overlap 0.58)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.80 |
| classes told apart: joint / best code / best member | 98 /  / 7 (of 121) |
| members whose removal merges classes | 0.33 |
| support overlap (1 = tiling) / random sets / p | 2.10 / 3.02 / 0.01 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.74 |
| decoding acc. joint / best code / best member (chance) | 0.68 /  / 0.06 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.02 |
| consumers / read jointly | 52 / 48 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L18 attn |
| CKA(arrangement before, joint write) | 0.78 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.13 |
| share of the write inside the old arrangement's span | 0.16 |
| write energy / code energy before | 0.18 |

<details><summary>codes and components</summary>

**code 4s-662.0 (L18): 21 comps, tells apart 98/121, coverage 0.80, overlap 2.10 (random 2.94, p 0.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 down c147 | MLP | a//10 {0, 3} x b//10 {0..2}; a//10 {1..2} x b//10 {0..3} (-) | 0.92 | 0.287 |
| L18 down c66 | MLP | a//10 {0..1} x b//10 {10}; a//10 {9} x b//10 {0..2, 8..9}; a//10 {10} x b//10 {0..3, 8..10} (+) | 0.80 | 0.369 |
| L18 down c75 | MLP | a//10 {0} x b//10 {0, 2..10}; a//10 {1} x b//10 {8, 10}; a//10 {2..3} x b//10 {10}; a//10 {4, 9} x b//10 {0}; a//10 {5} x b//10 {7}; a//10 {6} x b//10 {5, 7..8}; a//10 {7} x b//10 {6, 8..10}; a//10 {8} x b//10 {6..8, 10}; a//10 {10} x b//10 {0, 8, 10} (+) | 0.78 | 0.958 |
| L18 down c213 | MLP | a//10 {0} x b//10 {7}; a//10 {6, 8} x b//10 {1..2}; a//10 {7} x b//10 {0..2} (-) | 0.71 | 0.196 |
| L18 down c25 | MLP | a//10 {1..2} x b//10 {1}; a//10 {6} x b//10 {1, 5..6}; a//10 {7} x b//10 {6}; a//10 {10} x b//10 {5..6} (+) | 0.64 | 0.321 |
| L18 down c625 | MLP | a//10 {10} x b//10 {5, 8, 10} (+) | 0.52 | 0.009 |
| L18 down c21 | MLP | a//10 {1} x b//10 {1}; a//10 {5, 10} x b//10 {1..2, 6..7}; a//10 {6} x b//10 {2, 6..7} (+) | 0.70 | 0.357 |
| L18 down c111 | MLP | a//10 {2} x b//10 {4}; a//10 {3..4} x b//10 {0..5, 10}; a//10 {5} x b//10 {0, 2} (+) | 0.80 | 0.359 |
| L18 down c63 | MLP | a//10 {3, 10} x b//10 {3}; a//10 {6} x b//10 {2..3, 6}; a//10 {7} x b//10 {3, 6..7} (-) | 0.59 | 0.200 |
| L18 down c27 | MLP | a//10 {3} x b//10 {1}; a//10 {8} x b//10 {6..7} (+) | 0.72 | 0.228 |
| L18 down c36 | MLP | a//10 {3} x b//10 {2..3, 8}; a//10 {4} x b//10 {3, 8}; a//10 {8} x b//10 {3, 7..8}; a//10 {9} x b//10 {8}; a//10 {10} x b//10 {3..5, 8..9} (+) | 0.62 | 0.234 |
| L18 down c23 | MLP | a//10 {4} x b//10 {3}; a//10 {5} x b//10 {3..4}; a//10 {6} x b//10 {4..5} (+) | 0.77 | 0.355 |
| L18 down c61 | MLP | a//10 {5} x b//10 {3..4}; a//10 {6} x b//10 {4}; a//10 {10} x b//10 {4, 9} (+) | 0.72 | 0.128 |
| L18 down c1 | MLP | a//10 {6} x b//10 {5}; a//10 {7} x b//10 {6}; a//10 {8} x b//10 {7..8}; a//10 {9} x b//10 {9}; a//10 {10} x b//10 {4..9} (+) | 0.52 | 1.000 |
| L18 down c230 | MLP | a//10 {7..8} x b//10 {10}; a//10 {9} x b//10 {7, 9..10}; a//10 {10} x b//10 {0..1, 4..10} (-) | 0.76 | 0.297 |
| L18 down c30 | MLP | a//10 {7} x b//10 {3..5}; a//10 {8} x b//10 {4..5} (-) | 0.70 | 0.085 |
| L18 down c245 | MLP | a//10 {8, 10} x b//10 {6..7}; a//10 {9} x b//10 {7} (-) | 0.76 | 0.068 |
| L18 down c118 | MLP | a//10 {8} x b//10 {0..2}; a//10 {9} x b//10 {0..7}; a//10 {10} x b//10 {0..3, 10} (-) | 0.76 | 0.478 |
| L18 down c0 | MLP | a//10 {9} x b//10 {10} (-) | 0.64 | 0.999 |
| L18 down c22 | MLP | a//10 {9} x b//10 {6}; a//10 {10} x b//10 {1, 5..6} (+) | 0.78 | 0.367 |
| L18 down c12 | MLP | a//10 {9} x b//10 {7}; a//10 {10} x b//10 {2..3, 5, 7..8} (+) | 0.72 | 0.592 |

</details>

</details>

<details><summary><b>4s-663</b> `tens(a,b)` @ `=` (sub) — block code; 8 comps, L21; tells apart 33/121 classes (best member 7); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.53 |
| classes told apart: joint / best code / best member | 33 /  / 7 (of 121) |
| members whose removal merges classes | 0.88 |
| support overlap (1 = tiling) / random sets / p | 1.28 / 1.63 / 0.06 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.72 |
| decoding acc. joint / best code / best member (chance) | 0.28 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.98 |
| consumers / read jointly | 16 / 5 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 attn |
| CKA(arrangement before, joint write) | 0.59 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.07 |

<details><summary>codes and components</summary>

**code 4s-663.0 (L21): 8 comps, tells apart 33/121, coverage 0.53, overlap 1.28 (random 1.63, p 0.06)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c38 | MLP | a//10 {0} x b//10 {0}; a//10 {1} x b//10 {10}; a//10 {3} x b//10 {4..5}; a//10 {9..10} x b//10 {9..10} (-) | 0.54 | 0.999 |
| L21 down c51 | MLP | a//10 {0} x b//10 {1, 3..10}; a//10 {1} x b//10 {3..10}; a//10 {2} x b//10 {4..5, 8..9} (+) | 0.84 | 0.614 |
| L21 down c36 | MLP | a//10 {10} x b//10 {2..4} (-) | 0.66 | 0.073 |
| L21 down c439 | MLP | a//10 {10} x b//10 {3..9} (-) | 0.92 | 0.011 |
| L21 down c124 | MLP | a//10 {3..4} x b//10 {6}; a//10 {5} x b//10 {6..10}; a//10 {6} x b//10 {6..9}; a//10 {7} x b//10 {7..10}; a//10 {8} x b//10 {9..10} (-) | 0.64 | 0.293 |
| L21 down c43 | MLP | a//10 {6} x b//10 {6..8}; a//10 {7} x b//10 {7..9}; a//10 {8} x b//10 {8..10}; a//10 {9} x b//10 {0, 9..10}; a//10 {10} x b//10 {0..1, 9..10} (+) | 0.70 | 0.238 |
| L21 down c212 | MLP | a//10 {7} x b//10 {0..3} (+) | 0.66 | 0.081 |
| L21 down c0 | MLP | a//10 {7} x b//10 {5}; a//10 {8} x b//10 {6..7}; a//10 {9} x b//10 {7}; a//10 {10} x b//10 {7..8} (-) | 0.73 | 0.258 |

</details>

</details>

<details><summary><b>4s-664</b> `tens(a,b)` @ `=` (sub) — single component; 1 comps, L22; tells apart 6/121 classes (best member 6); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.06 |
| classes told apart: joint / best code / best member | 6 /  / 6 (of 121) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.72 |
| decoding acc. joint / best code / best member (chance) | 0.03 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 MLP |
| CKA(arrangement before, joint write) | 0.06 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.12 |
| write energy / code energy before | 0.00 |
| best source position (CKA) | `b` (0.07) |

<details><summary>codes and components</summary>

**code 4s-664.0 (L22): 1 comps, tells apart 6/121, coverage 0.06, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 o c200 | H15 | a//10 {0..5, 10} x b//10 {10} (-) | 0.71 | 0.011 |

</details>

</details>

<details><summary><b>4s-665</b> `tens(a,b)` @ `=` (sub) — tiling, 2 codes of the same shape; 10 comps, L23 L25; tells apart 51/121 classes (best member 6); on add: 4a-561 (member overlap 0.09)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.48 |
| classes told apart: joint / best code / best member | 51 / 42 / 6 (of 121) |
| members whose removal merges classes | 0.90 |
| support overlap (1 = tiling) / random sets / p | 1.34 / 1.78 / 0.02 |
| mean CKA between its codes | 0.75 |
| purity of the joint write (per prompt) | 0.67 |
| decoding acc. joint / best code / best member (chance) | 0.30 / 0.27 / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.03 |
| consumers / read jointly | 10 / 4 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L23 attn |
| CKA(arrangement before, joint write) | 0.80 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.06 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.05 |

<details><summary>codes and components</summary>

**code 4s-665.0 (L23): 8 comps, tells apart 42/121, coverage 0.45, overlap 1.22 (random 1.63, p 0.02)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 down c58 | MLP | a//10 {0} x b//10 {2}; a//10 {1} x b//10 {0, 2}; a//10 {2} x b//10 {0}; a//10 {3} x b//10 {0..1}; a//10 {4} x b//10 {1..2} (-) | 0.78 | 0.252 |
| L23 down c41 | MLP | a//10 {2} x b//10 {1}; a//10 {3} x b//10 {2}; a//10 {5} x b//10 {4}; a//10 {6} x b//10 {5}; a//10 {7} x b//10 {5..6}; a//10 {8} x b//10 {6..7}; a//10 {10} x b//10 {6..9} (-) | 0.71 | 0.298 |
| L23 down c7 | MLP | a//10 {5..6} x b//10 {0, 10}; a//10 {10} x b//10 {3..5} (-) | 0.52 | 0.143 |
| L23 down c110 | MLP | a//10 {5} x b//10 {6..8}; a//10 {6} x b//10 {7..9}; a//10 {7} x b//10 {8..10}; a//10 {8} x b//10 {9..10} (-) | 0.68 | 0.221 |
| L23 down c16 | MLP | a//10 {5} x b//10 {7}; a//10 {10} x b//10 {2} (-) | 0.58 | 0.185 |
| L23 down c8 | MLP | a//10 {6, 8} x b//10 {0..1}; a//10 {7} x b//10 {0..2} (-) | 0.64 | 0.115 |
| L23 down c10 | MLP | a//10 {7} x b//10 {7}; a//10 {8} x b//10 {6..8}; a//10 {9} x b//10 {9..10}; a//10 {10} x b//10 {0, 5..7, 9..10} (+) | 0.69 | 0.971 |
| L23 down c18 | MLP | a//10 {8} x b//10 {0}; a//10 {9} x b//10 {0..2, 10}; a//10 {10} x b//10 {0..2} (+) | 0.75 | 0.175 |

**code 4s-665.1 (L25): 2 comps, tells apart 3/121, coverage 0.09, overlap 1.00 (random 1.03, p 0.46)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L25 down c32 | MLP | a//10 {7..8} x b//10 {0..2}; a//10 {9} x b//10 {1..4} (+) | 0.51 | 0.113 |
| L25 down c6 | MLP | a//10 {9} x b//10 {10} (-) | 0.62 | 0.999 |

</details>

</details>

<details><summary><b>4s-666</b> `tens(a,b)` @ `=` (sub) — block code; 5 comps, L24; tells apart 22/121 classes (best member 7); on add: 4a-560 (member overlap 0.17)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.36 |
| classes told apart: joint / best code / best member | 22 /  / 7 (of 121) |
| members whose removal merges classes | 0.80 |
| support overlap (1 = tiling) / random sets / p | 1.23 / 1.31 / 0.35 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.69 |
| decoding acc. joint / best code / best member (chance) | 0.16 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L24 attn |
| CKA(arrangement before, joint write) | 0.56 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.02 |

<details><summary>codes and components</summary>

**code 4s-666.0 (L24): 5 comps, tells apart 22/121, coverage 0.36, overlap 1.23 (random 1.28, p 0.38)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 down c350 | MLP | a//10 {0} x b//10 {0..2}; a//10 {1} x b//10 {1..2}; a//10 {2} x b//10 {2}; a//10 {10} x b//10 {10} (+) | 0.64 | 0.128 |
| L24 down c0 | MLP | a//10 {0} x b//10 {10}; a//10 {3..4} x b//10 {0}; a//10 {5} x b//10 {0..1}; a//10 {6} x b//10 {6..9}; a//10 {7} x b//10 {7..8}; a//10 {8} x b//10 {9}; a//10 {9} x b//10 {9..10}; a//10 {10} x b//10 {4, 10} (-) | 0.51 | 1.000 |
| L24 down c70 | MLP | a//10 {4} x b//10 {0, 5, 10}; a//10 {5} x b//10 {0..1}; a//10 {10} x b//10 {5} (+) | 0.67 | 0.183 |
| L24 down c313 | MLP | a//10 {5} x b//10 {4}; a//10 {6} x b//10 {5..6}; a//10 {7..8} x b//10 {6..9}; a//10 {9} x b//10 {8..10}; a//10 {10} x b//10 {7..9} (-) | 0.64 | 0.267 |
| L24 down c8 | MLP | a//10 {6} x b//10 {0, 10}; a//10 {7..8} x b//10 {0..1}; a//10 {10} x b//10 {3} (-) | 0.78 | 0.281 |

</details>

</details>

<details><summary><b>4s-667</b> `tens(a,b)` @ `=` (sub) — block code; 4 comps, L26; tells apart 33/121 classes (best member 8); on add: 4a-562 (member overlap 0.08)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.36 |
| classes told apart: joint / best code / best member | 33 /  / 8 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.23 / 1.22 / 0.56 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.67 |
| decoding acc. joint / best code / best member (chance) | 0.15 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L26 attn |
| CKA(arrangement before, joint write) | 0.59 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.02 |

<details><summary>codes and components</summary>

**code 4s-667.0 (L26): 4 comps, tells apart 33/121, coverage 0.36, overlap 1.23 (random 1.22, p 0.53)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 down c85 | MLP | a//10 {0..1} x b//10 {0..2}; a//10 {2} x b//10 {0} (-) | 0.80 | 0.139 |
| L26 down c10 | MLP | a//10 {0} x b//10 {1, 3..5, 8..9}; a//10 {1} x b//10 {2..5}; a//10 {2} x b//10 {3..4}; a//10 {3} x b//10 {4..5}; a//10 {7} x b//10 {8}; a//10 {10} x b//10 {4..9} (+) | 0.64 | 0.955 |
| L26 down c75 | MLP | a//10 {0} x b//10 {3..10}; a//10 {1} x b//10 {5..10}; a//10 {4, 6..8, 10} x b//10 {0}; a//10 {9} x b//10 {0..1} (+) | 0.68 | 0.268 |
| L26 down c115 | MLP | a//10 {3..4} x b//10 {0}; a//10 {7} x b//10 {4}; a//10 {10} x b//10 {6} (-) | 0.61 | 0.163 |

</details>

</details>

<details><summary><b>4s-668</b> `tens(a,b)` @ `=` (sub) — block code; 3 comps, L27; tells apart 12/121 classes (best member 6); on add: 4a-562 (member overlap 0.09)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.22 |
| classes told apart: joint / best code / best member | 12 /  / 6 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.11 / 1.12 / 0.47 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.60 |
| decoding acc. joint / best code / best member (chance) | 0.06 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.99 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L27 attn |
| CKA(arrangement before, joint write) | 0.42 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.00 |

<details><summary>codes and components</summary>

**code 4s-668.0 (L27): 3 comps, tells apart 12/121, coverage 0.22, overlap 1.11 (random 1.11, p 0.50)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 down c2 | MLP | a//10 {0} x b//10 {4..6}; a//10 {1} x b//10 {5}; a//10 {2} x b//10 {0..1}; a//10 {3} x b//10 {1..2}; a//10 {4} x b//10 {1..3}; a//10 {5} x b//10 {3}; a//10 {6} x b//10 {4}; a//10 {7} x b//10 {5..6}; a//10 {8} x b//10 {6..7}; a//10 {9} x b//10 {10}; a//10 {10} x b//10 {2..3, 6..8} (-) | 0.62 | 1.000 |
| L27 down c68 | MLP | a//10 {10} x b//10 {3..7} (+) | 0.66 | 0.010 |
| L27 down c117 | MLP | a//10 {8} x b//10 {0}; a//10 {10} x b//10 {1} (-) | 0.53 | 0.028 |

</details>

</details>

<details><summary><b>4s-669</b> `tens(a,b)` @ `=` (sub) — tiling; 7 comps, L28; tells apart 58/121 classes (best member 6); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.49 |
| classes told apart: joint / best code / best member | 58 /  / 6 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.15 / 1.49 / 0.01 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.64 |
| decoding acc. joint / best code / best member (chance) | 0.23 /  / 0.05 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L28 attn |
| CKA(arrangement before, joint write) | 0.63 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.07 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.02 |

<details><summary>codes and components</summary>

**code 4s-669.0 (L28): 7 comps, tells apart 58/121, coverage 0.49, overlap 1.15 (random 1.50, p 0.04)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 down c98 | MLP | a//10 {0, 6} x b//10 {3}; a//10 {3} x b//10 {0, 4, 6}; a//10 {4} x b//10 {0..1, 6}; a//10 {5} x b//10 {2}; a//10 {7} x b//10 {4}; a//10 {8} x b//10 {5}; a//10 {10} x b//10 {6} (-) | 0.56 | 0.272 |
| L28 down c1 | MLP | a//10 {0} x b//10 {0}; a//10 {9} x b//10 {10} (-) | 0.54 | 0.998 |
| L28 down c11 | MLP | a//10 {0} x b//10 {1, 3..5, 7..9}; a//10 {1} x b//10 {2..9}; a//10 {2} x b//10 {3..6, 8}; a//10 {3} x b//10 {2, 4..5}; a//10 {7} x b//10 {8..9}; a//10 {9} x b//10 {9..10}; a//10 {10} x b//10 {0, 4..9} (+) | 0.69 | 0.983 |
| L28 down c298 | MLP | a//10 {0} x b//10 {9..10}; a//10 {9..10} x b//10 {0} (-) | 0.52 | 0.052 |
| L28 down c70 | MLP | a//10 {4} x b//10 {0, 5, 10}; a//10 {5} x b//10 {0..1}; a//10 {6} x b//10 {1}; a//10 {7} x b//10 {3}; a//10 {8} x b//10 {4}; a//10 {10} x b//10 {5} (-) | 0.62 | 0.239 |
| L28 down c312 | MLP | a//10 {7} x b//10 {0} (+) | 0.56 | 0.010 |
| L28 down c62 | MLP | a//10 {8} x b//10 {0}; a//10 {9} x b//10 {0..3}; a//10 {10} x b//10 {1} (-) | 0.76 | 0.120 |

</details>

</details>

<details><summary><b>4s-670</b> `tens(a,b)` @ `=` (sub) — block code; 4 comps, L29; tells apart 18/121 classes (best member 10); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.22 |
| classes told apart: joint / best code / best member | 18 /  / 10 (of 121) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.04 / 1.25 / 0.04 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.58 |
| decoding acc. joint / best code / best member (chance) | 0.09 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L29 attn |
| CKA(arrangement before, joint write) | 0.55 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.01 |

<details><summary>codes and components</summary>

**code 4s-670.0 (L29): 4 comps, tells apart 18/121, coverage 0.22, overlap 1.04 (random 1.21, p 0.08)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L29 down c584 | MLP | a//10 {0} x b//10 {0..2}; a//10 {1} x b//10 {1} (+) | 0.64 | 0.108 |
| L29 down c12 | MLP | a//10 {5} x b//10 {4}; a//10 {6} x b//10 {5..6}; a//10 {7} x b//10 {6..7}; a//10 {8} x b//10 {7}; a//10 {9} x b//10 {9..10}; a//10 {10} x b//10 {0, 3..10} (-) | 0.56 | 0.920 |
| L29 down c144 | MLP | a//10 {7} x b//10 {0}; a//10 {10} x b//10 {2} (+) | 0.52 | 0.110 |
| L29 down c213 | MLP | a//10 {9} x b//10 {0, 3..5}; a//10 {10} x b//10 {0} (-) | 0.63 | 0.056 |

</details>

</details>

<details><summary><b>4s-671</b> `tens(a,b)` @ `=` (sub) — block code; 3 comps, L30; tells apart 6/121 classes (best member 6); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.31 |
| classes told apart: joint / best code / best member | 6 /  / 6 (of 121) |
| members whose removal merges classes | 0.67 |
| support overlap (1 = tiling) / random sets / p | 1.30 / 1.11 / 0.84 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.61 |
| decoding acc. joint / best code / best member (chance) | 0.09 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.12 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L30 attn |
| CKA(arrangement before, joint write) | 0.57 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.01 |

<details><summary>codes and components</summary>

**code 4s-671.0 (L30): 3 comps, tells apart 6/121, coverage 0.31, overlap 1.30 (random 1.14, p 0.85)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 down c3 | MLP | a//10 {0..1} x b//10 {3..9}; a//10 {2} x b//10 {7..9}; a//10 {7} x b//10 {1}; a//10 {9} x b//10 {1..2} (+) | 0.61 | 0.362 |
| L30 down c201 | MLP | a//10 {0} x b//10 {3..10} (+) | 0.86 | 0.070 |
| L30 down c381 | MLP | a//10 {1} x b//10 {3..4, 7..8}; a//10 {2} x b//10 {3..4}; a//10 {5} x b//10 {4..5, 10}; a//10 {6} x b//10 {6, 10}; a//10 {7} x b//10 {7}; a//10 {9} x b//10 {9..10}; a//10 {10} x b//10 {0, 4..5, 8..10} (-) | 0.56 | 0.986 |

</details>

</details>

<details><summary><b>4s-672</b> `tens(a,b)` @ `=` (sub) — single component; 1 comps, L31; tells apart 2/121 classes (best member 2); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.26 |
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
| input point | L30 MLP |
| CKA(arrangement before, joint write) | 0.44 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.00 |
| best source position (CKA) | `op` (0.42) |

<details><summary>codes and components</summary>

**code 4s-672.0 (L31): 1 comps, tells apart 2/121, coverage 0.26, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L31 o c7 | H14 | a//10 {1} x b//10 {3}; a//10 {2} x b//10 {3..4}; a//10 {3} x b//10 {4, 10}; a//10 {5} x b//10 {4}; a//10 {6} x b//10 {4..6}; a//10 {7} x b//10 {0, 3, 6..7}; a//10 {8} x b//10 {4, 6..8}; a//10 {9} x b//10 {4..10}; a//10 {10} x b//10 {3..9} (+) | 0.65 | 1.000 |

</details>

</details>

<details><summary><b>4s-673</b> `tens(a,b)` @ `=` (sub) — single component; 1 comps, L31; tells apart 3/121 classes (best member 3); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.14 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 121) |
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
| input point | L30 MLP |
| CKA(arrangement before, joint write) | 0.39 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.00 |
| best source position (CKA) | `b` (0.16) |

<details><summary>codes and components</summary>

**code 4s-673.0 (L31): 1 comps, tells apart 3/121, coverage 0.14, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L31 o c165 | H15 | a//10 {0..1} x b//10 {0}; a//10 {2} x b//10 {0..1}; a//10 {5} x b//10 {4}; a//10 {6} x b//10 {5}; a//10 {7} x b//10 {6}; a//10 {8} x b//10 {6..7}; a//10 {9} x b//10 {9..10}; a//10 {10} x b//10 {0, 6..10} (-) | 0.52 | 0.327 |

</details>

</details>

</details>

<details><summary>`res`: 13 mechanisms, 169 components</summary>

<details><summary><b>4s-610</b> `res` @ `=` (sub) — single component; 1 comps, L18; tells apart 1/199 classes (best member 1); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.08 |
| classes told apart: joint / best code / best member | 1 /  / 1 (of 199) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.56 |
| decoding acc. joint / best code / best member (chance) | 0.01 /  / 0.01 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L18 attn |
| CKA(arrangement before, joint write) | 0.27 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.37 |
| write energy / code energy before | 0.01 |

<details><summary>codes and components</summary>

**code 4s-610.0 (L18): 1 comps, tells apart 1/199, coverage 0.08, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 down c26 | MLP | res in {2..3, 7..8, 12..13, 17..18, 22..23, 27..28, 32, 97..98} (-) | 0.56 | 0.424 |

</details>

</details>

<details><summary><b>4s-611</b> `res` @ `=` (sub) — block code; 2 comps, L19; tells apart 10/199 classes (best member 8); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.05 |
| classes told apart: joint / best code / best member | 10 /  / 8 (of 199) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.77 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.65 |
| decoding acc. joint / best code / best member (chance) | 0.02 /  / 0.01 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.99 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L19 attn |
| CKA(arrangement before, joint write) | 0.17 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.01 |

<details><summary>codes and components</summary>

**code 4s-611.0 (L19): 2 comps, tells apart 10/199, coverage 0.05, overlap 1.00 (random 1.00, p 0.84)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L19 down c149 | MLP | res in {2..3, 12..13, 22..23} (+) | 0.71 | 0.108 |
| L19 down c74 | MLP | res in {8..9, 99} (-) | 0.56 | 0.051 |

</details>

</details>

<details><summary><b>4s-612</b> `res` @ `=` (sub) — block code; 5 comps, L20; tells apart 65/199 classes (best member 4); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.29 |
| classes told apart: joint / best code / best member | 65 /  / 4 (of 199) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.47 / 1.11 / 0.97 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.59 |
| decoding acc. joint / best code / best member (chance) | 0.14 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.97 |
| consumers / read jointly | 35 / 32 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L20 attn |
| CKA(arrangement before, joint write) | 0.26 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.11 |
| share of the write inside the old arrangement's span | 0.37 |
| write energy / code energy before | 0.09 |

<details><summary>codes and components</summary>

**code 4s-612.0 (L20): 5 comps, tells apart 65/199, coverage 0.29, overlap 1.47 (random 1.11, p 0.97)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 down c9 | MLP | res in {-23, -18, -13, -8, -3, 2..3, 7, 12..13, 17, 22..23, 27, 32, 37, 42, 47, 52, 57, 62, 72, 77, 82, 87, 92, 97} (+) | 0.67 | 0.485 |
| L20 down c3 | MLP | res in {10..16, 30..34} (+) | 0.54 | 0.526 |
| L20 down c100 | MLP | res in {3..5, 13..15, 23..24, 94, 99} (+) | 0.58 | 0.197 |
| L20 down c6 | MLP | res in {4..10, 25..30} (-) | 0.50 | 0.490 |
| L20 down c13 | MLP | res in {6..8, 16..18, 26..27, 36..37, 46..47, 56..57, 67, 76..77, 86..87, 95..97} (-) | 0.71 | 0.392 |

</details>

</details>

<details><summary><b>4s-613</b> `res` @ `=` (sub) — block code; 9 comps, L21; tells apart 40/199 classes (best member 9); on add: 4a-482 (member overlap 0.06)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.25 |
| classes told apart: joint / best code / best member | 40 /  / 9 (of 199) |
| members whose removal merges classes | 0.78 |
| support overlap (1 = tiling) / random sets / p | 1.51 / 1.27 / 0.91 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.66 |
| decoding acc. joint / best code / best member (chance) | 0.12 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.98 |
| consumers / read jointly | 13 / 6 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 attn |
| CKA(arrangement before, joint write) | 0.37 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.08 |
| share of the write inside the old arrangement's span | 0.19 |
| write energy / code energy before | 0.07 |

<details><summary>codes and components</summary>

**code 4s-613.0 (L21): 9 comps, tells apart 40/199, coverage 0.25, overlap 1.51 (random 1.29, p 0.86)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c4 | MLP | res in {-18, -8, 1..3, 11..13, 21..23, 31..33, 42..43, 52, 62, 72, 82, 92} (+) | 0.65 | 0.396 |
| L21 down c42 | MLP | res in {10, 20, 30, 40, 80, 90} (-) | 0.55 | 0.044 |
| L21 down c231 | MLP | res in {11, 31} (+) | 0.52 | 0.013 |
| L21 down c2 | MLP | res in {11..15, 33..34} (-) | 0.65 | 0.278 |
| L21 down c63 | MLP | res in {21..30} (-) | 0.59 | 0.118 |
| L21 down c13 | MLP | res in {4, 14, 24, 34, 84, 94} (+) | 0.75 | 0.101 |
| L21 down c57 | MLP | res in {6..20} (+) | 0.63 | 0.167 |
| L21 down c916 | MLP | res in {7..8, 88} (+) | 0.60 | 0.024 |
| L21 down c53 | MLP | res in {9, 19, 29, 99} (-) | 0.71 | 0.061 |

</details>

</details>

<details><summary><b>4s-614</b> `res` @ `=` (sub) — block code; 10 comps, L22; tells apart 44/199 classes (best member 11); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.18 |
| classes told apart: joint / best code / best member | 44 /  / 11 (of 199) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.46 / 1.33 / 0.76 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.64 |
| decoding acc. joint / best code / best member (chance) | 0.11 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.99 |
| consumers / read jointly | 17 / 5 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L22 attn |
| CKA(arrangement before, joint write) | 0.27 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.15 |
| write energy / code energy before | 0.03 |

<details><summary>codes and components</summary>

**code 4s-614.0 (L22): 10 comps, tells apart 44/199, coverage 0.18, overlap 1.46 (random 1.31, p 0.78)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 down c11 | MLP | res in {-19, 1..2, 19..23, 40..41, 61, 80..81, 98..99} (-) | 0.54 | 0.246 |
| L22 down c22 | MLP | res in {1, 5, 7, 11, 21, 31} (-) | 0.54 | 0.071 |
| L22 down c8 | MLP | res in {10, 20, 30} (+) | 0.68 | 0.095 |
| L22 down c427 | MLP | res in {12} (+) | 0.56 | 0.015 |
| L22 down c17 | MLP | res in {2, 22..23} (-) | 0.63 | 0.054 |
| L22 down c27 | MLP | res in {29..35} (+) | 0.68 | 0.098 |
| L22 down c212 | MLP | res in {3, 13, 23} (-) | 0.72 | 0.045 |
| L22 down c48 | MLP | res in {8, 28} (-) | 0.59 | 0.035 |
| L22 down c9 | MLP | res in {9, 19, 29, 39, 79, 89, 99} (-) | 0.70 | 0.076 |
| L22 down c121 | MLP | res in {9..10, 19, 29} (+) | 0.56 | 0.043 |

</details>

</details>

<details><summary><b>4s-615</b> `res` @ `=` (sub) — block code; 12 comps, L23; tells apart 43/199 classes (best member 9); on add: 4a-484 (member overlap 0.05)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.23 |
| classes told apart: joint / best code / best member | 43 /  / 9 (of 199) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.33 / 1.41 / 0.32 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.69 |
| decoding acc. joint / best code / best member (chance) | 0.16 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 46 / 10 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L23 attn |
| CKA(arrangement before, joint write) | 0.25 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.13 |
| write energy / code energy before | 0.06 |

<details><summary>codes and components</summary>

**code 4s-615.0 (L23): 12 comps, tells apart 43/199, coverage 0.23, overlap 1.33 (random 1.39, p 0.37)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 down c125 | MLP | res in {11, 31} (+) | 0.61 | 0.032 |
| L23 down c202 | MLP | res in {13..20} (-) | 0.80 | 0.095 |
| L23 down c4 | MLP | res in {2, 12, 22, 32, 42, 82, 92} (-) | 0.75 | 0.164 |
| L23 down c77 | MLP | res in {21} (+) | 0.61 | 0.043 |
| L23 down c20 | MLP | res in {25..32} (-) | 0.67 | 0.119 |
| L23 down c15 | MLP | res in {31..35} (+) | 0.57 | 0.098 |
| L23 down c5 | MLP | res in {34..41} (-) | 0.60 | 0.214 |
| L23 down c14 | MLP | res in {4, 14, 24, 34, 84, 94} (+) | 0.76 | 0.077 |
| L23 down c459 | MLP | res in {5, 15, 25} (+) | 0.66 | 0.028 |
| L23 down c24 | MLP | res in {7, 17, 27, 37, 87, 97} (-) | 0.76 | 0.075 |
| L23 down c205 | MLP | res in {8} (-) | 0.75 | 0.040 |
| L23 down c106 | MLP | res in {9, 19, 29, 89, 99} (+) | 0.67 | 0.024 |

</details>

</details>

<details><summary><b>4s-616</b> `res` @ `=` (sub) — block code; 17 comps, L24; tells apart 54/199 classes (best member 12); on add: 4a-485 (member overlap 0.08)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.29 |
| classes told apart: joint / best code / best member | 54 /  / 12 (of 199) |
| members whose removal merges classes | 0.71 |
| support overlap (1 = tiling) / random sets / p | 1.60 / 1.62 / 0.48 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.71 |
| decoding acc. joint / best code / best member (chance) | 0.20 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 44 / 5 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L24 attn |
| CKA(arrangement before, joint write) | 0.33 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.13 |
| write energy / code energy before | 0.06 |

<details><summary>codes and components</summary>

**code 4s-616.0 (L24): 17 comps, tells apart 54/199, coverage 0.29, overlap 1.60 (random 1.65, p 0.42)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 down c31 | MLP | res in {10..13} (-) | 0.80 | 0.151 |
| L24 down c22 | MLP | res in {13..15} (-) | 0.74 | 0.068 |
| L24 down c101 | MLP | res in {15..24} (+) | 0.77 | 0.111 |
| L24 down c212 | MLP | res in {20, 40} (-) | 0.58 | 0.016 |
| L24 down c82 | MLP | res in {21, 31..32, 41} (-) | 0.53 | 0.053 |
| L24 down c16 | MLP | res in {21..22, 24..26, 72} (-) | 0.60 | 0.120 |
| L24 down c11 | MLP | res in {22..26, 62..66} (+) | 0.53 | 0.092 |
| L24 down c110 | MLP | res in {22..28} (-) | 0.73 | 0.064 |
| L24 down c50 | MLP | res in {26} (-) | 0.57 | 0.026 |
| L24 down c99 | MLP | res in {28..33} (-) | 0.64 | 0.058 |
| L24 down c5 | MLP | res in {3, 13, 23, 33, 43, 83, 93} (-) | 0.74 | 0.125 |
| L24 down c52 | MLP | res in {34} (+) | 0.54 | 0.029 |
| L24 down c91 | MLP | res in {6, 16, 96..97} (-) | 0.63 | 0.037 |
| L24 down c4 | MLP | res in {8, 18, 28, 38, 88, 98} (+) | 0.74 | 0.118 |
| L24 down c87 | MLP | res in {80..85} (-) | 0.60 | 0.026 |
| L24 down c41 | MLP | res in {84..94} (-) | 0.67 | 0.041 |
| L24 down c83 | MLP | res in {9, 19, 29, 89, 99} (+) | 0.71 | 0.035 |

</details>

</details>

<details><summary><b>4s-617</b> `res` @ `=` (sub) — tiling; 15 comps, L25; tells apart 50/199 classes (best member 11); on add: 4a-486 (member overlap 0.05)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.25 |
| classes told apart: joint / best code / best member | 50 /  / 11 (of 199) |
| members whose removal merges classes | 0.73 |
| support overlap (1 = tiling) / random sets / p | 1.30 / 1.55 / 0.04 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.74 |
| decoding acc. joint / best code / best member (chance) | 0.17 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.02 |
| consumers / read jointly | 19 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L25 attn |
| CKA(arrangement before, joint write) | 0.58 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.04 |

<details><summary>codes and components</summary>

**code 4s-617.0 (L25): 15 comps, tells apart 50/199, coverage 0.25, overlap 1.30 (random 1.51, p 0.04)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L25 down c42 | MLP | res in {10..13} (+) | 0.84 | 0.077 |
| L25 down c207 | MLP | res in {12, 14} (-) | 0.74 | 0.021 |
| L25 down c108 | MLP | res in {12..13, 22, 32, 72, 92} (+) | 0.64 | 0.063 |
| L25 down c53 | MLP | res in {15} (+) | 0.70 | 0.048 |
| L25 down c110 | MLP | res in {16..18} (+) | 0.72 | 0.034 |
| L25 down c46 | MLP | res in {18..19} (+) | 0.78 | 0.043 |
| L25 down c216 | MLP | res in {20} (-) | 0.61 | 0.018 |
| L25 down c50 | MLP | res in {23} (+) | 0.66 | 0.043 |
| L25 down c98 | MLP | res in {24} (-) | 0.55 | 0.006 |
| L25 down c151 | MLP | res in {26..29} (+) | 0.61 | 0.033 |
| L25 down c220 | MLP | res in {3..15} (+) | 0.78 | 0.302 |
| L25 down c66 | MLP | res in {42..45, 82..83} (+) | 0.50 | 0.049 |
| L25 down c11 | MLP | res in {7, 17, 27, 37, 47, 57, 67, 77, 87, 97} (-) | 0.74 | 0.106 |
| L25 down c297 | MLP | res in {84..89} (+) | 0.57 | 0.025 |
| L25 down c139 | MLP | res in {94..98} (-) | 0.56 | 0.008 |

</details>

</details>

<details><summary><b>4s-618</b> `res` @ `=` (sub) — block code; 12 comps, L26; tells apart 43/199 classes (best member 10); on add: 4a-487 (member overlap 0.11)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.21 |
| classes told apart: joint / best code / best member | 43 /  / 10 (of 199) |
| members whose removal merges classes | 0.83 |
| support overlap (1 = tiling) / random sets / p | 1.41 / 1.38 / 0.58 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.73 |
| decoding acc. joint / best code / best member (chance) | 0.15 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.07 |
| consumers / read jointly | 5 / 5 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L26 attn |
| CKA(arrangement before, joint write) | 0.33 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.11 |
| write energy / code energy before | 0.03 |

<details><summary>codes and components</summary>

**code 4s-618.0 (L26): 12 comps, tells apart 43/199, coverage 0.21, overlap 1.41 (random 1.41, p 0.53)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 down c175 | MLP | res in {11, 21, 31, 41, 51, 91} (-) | 0.61 | 0.053 |
| L26 down c205 | MLP | res in {11..13} (+) | 0.83 | 0.051 |
| L26 down c258 | MLP | res in {13} (+) | 0.57 | 0.013 |
| L26 down c813 | MLP | res in {14..17} (+) | 0.79 | 0.033 |
| L26 down c33 | MLP | res in {16..20} (+) | 0.78 | 0.085 |
| L26 down c8 | MLP | res in {20..22} (-) | 0.78 | 0.086 |
| L26 down c4 | MLP | res in {23..28} (+) | 0.62 | 0.080 |
| L26 down c182 | MLP | res in {23..30} (-) | 0.65 | 0.129 |
| L26 down c20 | MLP | res in {30..33} (-) | 0.69 | 0.097 |
| L26 down c37 | MLP | res in {7..8, 17..18, 37, 97} (+) | 0.77 | 0.067 |
| L26 down c46 | MLP | res in {86..90} (-) | 0.62 | 0.028 |
| L26 down c341 | MLP | res in {92..98} (+) | 0.51 | 0.022 |

</details>

</details>

<details><summary><b>4s-619</b> `res` @ `=` (sub) — block code; 15 comps, L27; tells apart 44/199 classes (best member 9); on add: 4a-488 (member overlap 0.09)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.21 |
| classes told apart: joint / best code / best member | 44 /  / 9 (of 199) |
| members whose removal merges classes | 0.87 |
| support overlap (1 = tiling) / random sets / p | 1.44 / 1.53 / 0.30 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.67 |
| decoding acc. joint / best code / best member (chance) | 0.12 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.06 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L27 attn |
| CKA(arrangement before, joint write) | 0.41 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.12 |
| write energy / code energy before | 0.02 |

<details><summary>codes and components</summary>

**code 4s-619.0 (L27): 15 comps, tells apart 44/199, coverage 0.21, overlap 1.44 (random 1.52, p 0.32)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 down c269 | MLP | res in {1, 41} (-) | 0.55 | 0.038 |
| L27 down c204 | MLP | res in {11, 21, 31, 71} (-) | 0.57 | 0.038 |
| L27 down c23 | MLP | res in {11..17} (-) | 0.67 | 0.058 |
| L27 down c193 | MLP | res in {13..16} (+) | 0.83 | 0.086 |
| L27 down c586 | MLP | res in {17, 20..21} (-) | 0.64 | 0.026 |
| L27 down c167 | MLP | res in {19..26} (+) | 0.76 | 0.094 |
| L27 down c64 | MLP | res in {22..24} (+) | 0.70 | 0.033 |
| L27 down c203 | MLP | res in {27, 87} (-) | 0.56 | 0.014 |
| L27 down c24 | MLP | res in {35..37} (+) | 0.59 | 0.048 |
| L27 down c168 | MLP | res in {40..45} (-) | 0.52 | 0.044 |
| L27 down c165 | MLP | res in {6, 16, 26, 36, 86, 96} (-) | 0.54 | 0.039 |
| L27 down c69 | MLP | res in {8, 18, 88, 97..98} (+) | 0.54 | 0.032 |
| L27 down c80 | MLP | res in {9, 49, 89} (+) | 0.64 | 0.027 |
| L27 down c142 | MLP | res in {9, 49} (-) | 0.59 | 0.025 |
| L27 down c38 | MLP | res in {99} (-) | 0.52 | 0.035 |

</details>

</details>

<details><summary><b>4s-620</b> `res` @ `=` (sub) — tiling; 20 comps, L28; tells apart 40/199 classes (best member 9); on add: 4a-489 (member overlap 0.11)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.19 |
| classes told apart: joint / best code / best member | 40 /  / 9 (of 199) |
| members whose removal merges classes | 0.35 |
| support overlap (1 = tiling) / random sets / p | 1.43 / 1.76 / 0.04 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.68 |
| decoding acc. joint / best code / best member (chance) | 0.15 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.05 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L28 attn |
| CKA(arrangement before, joint write) | 0.31 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.09 |
| write energy / code energy before | 0.02 |

<details><summary>codes and components</summary>

**code 4s-620.0 (L28): 20 comps, tells apart 40/199, coverage 0.19, overlap 1.43 (random 1.71, p 0.04)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 down c160 | MLP | res in {-97, 7, 37, 57, 67, 77, 87, 97} (-) | 0.59 | 0.063 |
| L28 down c271 | MLP | res in {13, 23, 33, 83, 93} (-) | 0.53 | 0.052 |
| L28 down c10 | MLP | res in {13..15} (+) | 0.80 | 0.067 |
| L28 down c300 | MLP | res in {14, 24} (-) | 0.66 | 0.039 |
| L28 down c1010 | MLP | res in {14..17} (+) | 0.57 | 0.026 |
| L28 down c288 | MLP | res in {15} (+) | 0.58 | 0.009 |
| L28 down c24 | MLP | res in {16} (-) | 0.68 | 0.149 |
| L28 down c91 | MLP | res in {17} (+) | 0.70 | 0.009 |
| L28 down c522 | MLP | res in {17} (+) | 0.58 | 0.005 |
| L28 down c51 | MLP | res in {18} (-) | 0.70 | 0.024 |
| L28 down c327 | MLP | res in {20..21} (+) | 0.68 | 0.024 |
| L28 down c794 | MLP | res in {20..25} (-) | 0.55 | 0.075 |
| L28 down c236 | MLP | res in {23..24} (+) | 0.73 | 0.043 |
| L28 down c165 | MLP | res in {26..27} (+) | 0.59 | 0.026 |
| L28 down c67 | MLP | res in {27} (+) | 0.53 | 0.067 |
| L28 down c121 | MLP | res in {29} (-) | 0.62 | 0.041 |
| L28 down c46 | MLP | res in {30} (-) | 0.62 | 0.052 |
| L28 down c304 | MLP | res in {34..37} (-) | 0.51 | 0.210 |
| L28 down c99 | MLP | res in {8, 28, 38, 88, 98} (-) | 0.56 | 0.040 |
| L28 down c129 | MLP | res in {9, 99} (-) | 0.77 | 0.045 |

</details>

</details>

<details><summary><b>4s-621</b> `res` @ `=` (sub) — tiling; 29 comps, L29; tells apart 44/199 classes (best member 11); on add: 4a-490 (member overlap 0.08)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.27 |
| classes told apart: joint / best code / best member | 44 /  / 11 (of 199) |
| members whose removal merges classes | 0.52 |
| support overlap (1 = tiling) / random sets / p | 1.45 / 2.13 / 0.00 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.71 |
| decoding acc. joint / best code / best member (chance) | 0.18 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.07 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L29 attn |
| CKA(arrangement before, joint write) | 0.41 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.03 |

<details><summary>codes and components</summary>

**code 4s-621.0 (L29): 29 comps, tells apart 44/199, coverage 0.27, overlap 1.45 (random 2.12, p 0.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L29 down c174 | MLP | res in {-99, -97..-96, 95, 97..99} (+) | 0.55 | 0.008 |
| L29 down c445 | MLP | res in {11} (+) | 0.88 | 0.018 |
| L29 down c318 | MLP | res in {12} (+) | 0.83 | 0.024 |
| L29 down c263 | MLP | res in {13} (-) | 0.83 | 0.024 |
| L29 down c865 | MLP | res in {13} (-) | 0.69 | 0.015 |
| L29 down c739 | MLP | res in {14} (+) | 0.66 | 0.006 |
| L29 down c762 | MLP | res in {15, 35} (+) | 0.67 | 0.023 |
| L29 down c467 | MLP | res in {15} (+) | 0.79 | 0.012 |
| L29 down c477 | MLP | res in {16, 96} (+) | 0.66 | 0.010 |
| L29 down c280 | MLP | res in {16} (-) | 0.83 | 0.015 |
| L29 down c343 | MLP | res in {17} (-) | 0.76 | 0.017 |
| L29 down c577 | MLP | res in {19, 99} (-) | 0.55 | 0.032 |
| L29 down c72 | MLP | res in {20, 22..23} (-) | 0.58 | 0.044 |
| L29 down c279 | MLP | res in {20} (-) | 0.78 | 0.060 |
| L29 down c842 | MLP | res in {21} (+) | 0.50 | 0.016 |
| L29 down c132 | MLP | res in {24, 34, 44, 54, 64, 74, 84, 94} (-) | 0.57 | 0.068 |
| L29 down c448 | MLP | res in {25} (+) | 0.70 | 0.042 |
| L29 down c376 | MLP | res in {26} (-) | 0.62 | 0.021 |
| L29 down c618 | MLP | res in {27} (+) | 0.61 | 0.006 |
| L29 down c178 | MLP | res in {28} (-) | 0.67 | 0.028 |
| L29 down c415 | MLP | res in {31} (-) | 0.54 | 0.079 |
| L29 down c187 | MLP | res in {33..34} (-) | 0.56 | 0.070 |
| L29 down c853 | MLP | res in {37..39} (+) | 0.54 | 0.034 |
| L29 down c130 | MLP | res in {4, 64} (-) | 0.55 | 0.043 |
| L29 down c67 | MLP | res in {4, 9, 12..25} (+) | 0.65 | 0.160 |
| L29 down c150 | MLP | res in {80..86} (+) | 0.59 | 0.032 |
| L29 down c595 | MLP | res in {87..89} (-) | 0.56 | 0.011 |
| L29 down c1009 | MLP | res in {90..91, 93, 95} (-) | 0.55 | 0.008 |
| L29 down c736 | MLP | res in {99} (+) | 0.58 | 0.006 |

</details>

</details>

<details><summary><b>4s-622</b> `res` @ `=` (sub) — tiling; 22 comps, L30 L31; tells apart 24/199 classes (best member 8); on add: 4a-491 (member overlap 0.13)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.12 |
| classes told apart: joint / best code / best member | 24 /  / 8 (of 199) |
| members whose removal merges classes | 0.23 |
| support overlap (1 = tiling) / random sets / p | 1.48 / 1.79 / 0.06 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.68 |
| decoding acc. joint / best code / best member (chance) | 0.09 /  / 0.01 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.15 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L30 attn |
| CKA(arrangement before, joint write) | 0.22 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.11 |
| write energy / code energy before | 0.01 |

<details><summary>codes and components</summary>

**code 4s-622.0 (L30 L31): 22 comps, tells apart 24/199, coverage 0.12, overlap 1.48 (random 1.83, p 0.02)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 down c519 | MLP | res in {14} (-) | 0.74 | 0.009 |
| L30 down c181 | MLP | res in {17..19} (+) | 0.51 | 0.024 |
| L30 down c214 | MLP | res in {17} (+) | 0.60 | 0.013 |
| L30 down c642 | MLP | res in {18} (-) | 0.65 | 0.009 |
| L30 down c294 | MLP | res in {19} (-) | 0.71 | 0.017 |
| L30 down c739 | MLP | res in {19} (-) | 0.61 | 0.005 |
| L30 down c31 | MLP | res in {20} (+) | 0.79 | 0.007 |
| L30 down c800 | MLP | res in {20} (+) | 0.78 | 0.009 |
| L30 down c666 | MLP | res in {20} (-) | 0.51 | 0.005 |
| L30 down c796 | MLP | res in {20} (-) | 0.60 | 0.013 |
| L30 down c283 | MLP | res in {21} (-) | 0.63 | 0.049 |
| L30 down c997 | MLP | res in {22} (-) | 0.76 | 0.021 |
| L30 down c628 | MLP | res in {23} (+) | 0.54 | 0.011 |
| L30 down c551 | MLP | res in {24} (+) | 0.71 | 0.022 |
| L30 down c359 | MLP | res in {24} (-) | 0.76 | 0.029 |
| L30 down c159 | MLP | res in {25..26} (+) | 0.67 | 0.040 |
| L30 down c706 | MLP | res in {25..28} (+) | 0.69 | 0.041 |
| L30 down c882 | MLP | res in {27} (-) | 0.68 | 0.028 |
| L30 down c207 | MLP | res in {29..30} (-) | 0.61 | 0.034 |
| L30 down c183 | MLP | res in {32} (+) | 0.65 | 0.013 |
| L30 down c187 | MLP | res in {37, 87} (+) | 0.59 | 0.046 |
| L31 down c219 | MLP | res in {81..85} (-) | 0.57 | 0.014 |

</details>

</details>

</details>

<details><summary>`a%100`: 13 mechanisms, 79 components</summary>

<details><summary><b>4s-575</b> `a%100` @ `=` (sub) — block code; 2 comps, L2 L3; tells apart 4/100 classes (best member 4); on add: 4a-439 (member overlap 0.50)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.15 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 100) |
| members whose removal merges classes | 0.50 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.00 / 0.64 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.79 |
| decoding acc. joint / best code / best member (chance) | 0.05 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L1 MLP |
| CKA(arrangement before, joint write) | 0.10 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.23 2:0.11 20:0.10) |
| joint write: shape (spectrum k:share) | line (1:0.08 5:0.05) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.02 / 1.46 |
| best source position (CKA) | `a` (0.14) |

<details><summary>codes and components</summary>

**code 4s-575.0 (L2 L3): 2 comps, tells apart 4/100, coverage 0.15, overlap 1.00 (random 1.00, p 0.66)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 o c89 | H23 | a%100 in {1..4, 6, 12, 28..29, 39, 50, 86, 88, 91, 96} (+) | 0.59 | 1.000 |
| L3 o c432 | H15 | a%100 in {99} (+) | 0.86 | 0.027 |

</details>

</details>

<details><summary><b>4s-574</b> `a%100` @ `=` (sub) — block code, copy from `op`; 3 comps, L2; tells apart 13/100 classes (best member 4); on add: 4a-438 (member overlap 0.17)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.46 |
| classes told apart: joint / best code / best member | 13 /  / 4 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.37 / 1.06 / 0.92 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.74 |
| decoding acc. joint / best code / best member (chance) | 0.18 /  / 0.08 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.97 |
| consumers / read jointly | 13 / 4 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L1 MLP |
| CKA(arrangement before, joint write) | 0.60 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.07 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.43 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.23 2:0.11 20:0.10) |
| joint write: shape (spectrum k:share) | irregular (1:0.37 2:0.12 10:0.11) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.17 / 1.91 |
| best source position (CKA) | `op` (0.62) |

<details><summary>codes and components</summary>

**code 4s-574.0 (L2): 3 comps, tells apart 13/100, coverage 0.46, overlap 1.37 (random 1.06, p 0.94)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L2 o c3 | H2 | a%100 in {0, 3..9, 16, 35..36, 73, 90, 92} (-) | 0.77 | 1.000 |
| L2 o c14 | H2 | a%100 in {0..1, 5, 7..9, 11, 14..17, 19, 35..36, 55, 60..63, 67, 72..75, 81, 83..84, 86..87, 89..99} (+) | 0.70 | 1.000 |
| L2 o c6 | H2 | a%100 in {1..7, 10, 12} (+) | 0.91 | 1.000 |

</details>

</details>

<details><summary><b>4s-576</b> `a%100` @ `=` (sub) — block code, 3 codes of the same shape; 10 comps, L4 L30 L31; tells apart 9/100 classes (best member 7); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.12 |
| classes told apart: joint / best code / best member | 9 / 10 / 7 (of 100) |
| members whose removal merges classes | 0.20 |
| support overlap (1 = tiling) / random sets / p | 3.08 / 1.46 / 1.00 |
| mean CKA between its codes | 0.87 |
| purity of the joint write (per prompt) | 0.84 |
| decoding acc. joint / best code / best member (chance) | 0.08 / 0.09 / 0.06 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.40 |
| consumers / read jointly | 12 / 3 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L4 attn |
| CKA(arrangement before, joint write) | 0.52 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 14.39 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.40 2:0.13 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.18 2:0.16 3:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.06 / 1.30 |
| best source position (CKA) | `op` (0.42) |

<details><summary>codes and components</summary>

**code 4s-576.0 (L4): 1 comps, tells apart 7/100, coverage 0.07, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 down c4 | MLP | a%100 in {1..7} (+) | 0.69 | 0.085 |

**code 4s-576.1 (L30): 3 comps, tells apart 10/100, coverage 0.08, overlap 1.38 (random 1.05, p 0.94)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 o c46 | H20 | a%100 in {1..3} (+) | 0.94 | 0.050 |
| L30 o c137 | H20 | a%100 in {1..5} (+) | 0.96 | 0.200 |
| L30 down c535 | MLP | a%100 in {97..99} (-) | 0.69 | 0.036 |

**code 4s-576.2 (L31): 6 comps, tells apart 9/100, coverage 0.11, overlap 1.73 (random 1.22, p 0.96)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L31 down c213 | MLP | a%100 in {1..6} (-) | 0.75 | 0.163 |
| L31 down c52 | MLP | a%100 in {2..3} (+) | 0.78 | 0.030 |
| L31 down c494 | MLP | a%100 in {2..5} (-) | 0.90 | 0.070 |
| L31 down c10 | MLP | a%100 in {2} (+) | 0.52 | 0.062 |
| L31 down c629 | MLP | a%100 in {95..99} (+) | 0.71 | 0.054 |
| L31 down c690 | MLP | a%100 in {99} (-) | 0.65 | 0.011 |

</details>

</details>

<details><summary><b>4s-577</b> `a%100` @ `=` (sub) — block code, 5 codes of the same shape, copy from `b`; 17 comps, L5 L16 L18 L24 L25 L26; tells apart 22/100 classes (best member 6); on add: 4a-451 (member overlap 0.06)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.62 |
| classes told apart: joint / best code / best member | 22 / 14 / 6 (of 100) |
| members whose removal merges classes | 0.53 |
| support overlap (1 = tiling) / random sets / p | 2.18 / 1.83 / 0.94 |
| mean CKA between its codes | 0.81 |
| purity of the joint write (per prompt) | 0.72 |
| decoding acc. joint / best code / best member (chance) | 0.20 / 0.09 / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.07 |
| consumers / read jointly | 4 / 1 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L5 attn |
| CKA(arrangement before, joint write) | 0.79 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.03 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 2.48 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.42 2:0.13 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.44 2:0.22 3:0.10) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.20 / 1.59 |
| best source position (CKA) | `b` (0.68) |

<details><summary>codes and components</summary>

**code 4s-577.0 (L5): 1 comps, tells apart 2/100, coverage 0.21, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 down c10 | MLP | a%100 in {1..21} (+) | 0.85 | 0.292 |

**code 4s-577.1 (L16): 4 comps, tells apart 14/100, coverage 0.39, overlap 1.05 (random 1.12, p 0.24)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c73 | MLP | a%100 in {0, 36..41, 55..62, 75..83} (+) | 0.60 | 0.307 |
| L16 down c175 | MLP | a%100 in {0, 90, 99} (-) | 0.67 | 0.188 |
| L16 down c470 | MLP | a%100 in {0} (-) | 0.73 | 0.038 |
| L16 down c164 | MLP | a%100 in {1..12, 14} (+) | 0.76 | 0.154 |

**code 4s-577.2 (L18): 4 comps, tells apart 7/100, coverage 0.24, overlap 1.04 (random 1.12, p 0.23)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 o c324 | H18 | a%100 in {0, 91, 95, 97..99} (+) | 0.59 | 0.181 |
| L18 down c290 | MLP | a%100 in {2..17, 19} (+) | 0.77 | 0.249 |
| L18 o c230 | H18 | a%100 in {70} (+) | 0.52 | 0.018 |
| L18 o c336 | H18 | a%100 in {99} (-) | 0.69 | 0.015 |

**code 4s-577.3 (L24): 4 comps, tells apart 9/100, coverage 0.26, overlap 1.00 (random 1.11, p 0.17)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 down c79 | MLP | a%100 in {2..21} (-) | 0.65 | 0.323 |
| L24 o c501 | H22 | a%100 in {28..29} (-) | 0.69 | 0.023 |
| L24 o c503 | H17 | a%100 in {90} (+) | 0.84 | 0.011 |
| L24 down c207 | MLP | a%100 in {97..99} (+) | 0.66 | 0.069 |

**code 4s-577.4 (L25 L26): 4 comps, tells apart 11/100, coverage 0.22, overlap 1.00 (random 1.12, p 0.11)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L25 down c43 | MLP | a%100 in {6..14} (-) | 0.60 | 0.174 |
| L25 down c73 | MLP | a%100 in {98..99} (+) | 0.52 | 0.036 |
| L26 o c19 | H10 | a%100 in {0, 33, 52..53, 70, 74..75, 85} (-) | 0.53 | 1.000 |
| L26 o c379 | H14 | a%100 in {92..94} (+) | 0.91 | 0.038 |

</details>

</details>

<details><summary><b>4s-578</b> `a%100` @ `=` (sub) — single component; 1 comps, L8; tells apart 3/100 classes (best member 3); on add: 4a-445 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.08 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.59 |
| decoding acc. joint / best code / best member (chance) | 0.03 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L7 MLP |
| CKA(arrangement before, joint write) | 0.37 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.41 2:0.14 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.64 3:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.21 / 1.00 |
| best source position (CKA) | `op` (0.41) |

<details><summary>codes and components</summary>

**code 4s-578.0 (L8): 1 comps, tells apart 3/100, coverage 0.08, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L8 o c7 | H23 | a%100 in {1..2, 13, 19, 68, 73, 75, 99} (+) | 0.59 | 1.000 |

</details>

</details>

<details><summary><b>4s-579</b> `a%100` @ `=` (sub) — single component; 1 comps, L12; tells apart 2/100 classes (best member 2); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.13 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 100) |
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
| CKA(arrangement before, joint write) | 0.55 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.41 2:0.14 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.24 2:0.18 3:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.06 / 1.00 |
| best source position (CKA) | `a` (0.48) |

<details><summary>codes and components</summary>

**code 4s-579.0 (L12): 1 comps, tells apart 2/100, coverage 0.13, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L12 o c209 | H20 | a%100 in {0..10, 30, 70} (-) | 0.65 | 1.000 |

</details>

</details>

<details><summary><b>4s-580</b> `a%100` @ `=` (sub) — block code, 3 codes of the same shape; 8 comps, L13 L14 L22 L23 L27; tells apart 15/100 classes (best member 8); on add: 4a-450 (member overlap 0.09)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.24 |
| classes told apart: joint / best code / best member | 15 / 10 / 8 (of 100) |
| members whose removal merges classes | 0.75 |
| support overlap (1 = tiling) / random sets / p | 1.33 / 1.35 / 0.47 |
| mean CKA between its codes | 0.88 |
| purity of the joint write (per prompt) | 0.72 |
| decoding acc. joint / best code / best member (chance) | 0.10 / 0.10 / 0.06 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.09 |
| consumers / read jointly | 3 / 3 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L13 attn |
| CKA(arrangement before, joint write) | 0.22 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.15 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.42 2:0.14 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.20 2:0.17 3:0.14) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.07 / 1.19 |
| best source position (CKA) | `b` (0.50) |

<details><summary>codes and components</summary>

**code 4s-580.0 (L13 L14): 2 comps, tells apart 8/100, coverage 0.19, overlap 1.00 (random 1.00, p 0.66)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 down c0 | MLP | a%100 in {1..12} (+) | 0.50 | 1.000 |
| L14 o c101 | H18 | a%100 in {0, 94..99} (-) | 0.92 | 0.113 |

**code 4s-580.1 (L22 L23): 5 comps, tells apart 10/100, coverage 0.08, overlap 1.00 (random 1.18, p 0.07)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 o c498 | H15 | a%100 in {17} (-) | 0.65 | 0.010 |
| L22 o c55 | H15 | a%100 in {57} (+) | 0.62 | 0.009 |
| L22 o c10 | H15 | a%100 in {97..99} (-) | 0.72 | 0.189 |
| L23 o c234 | H22 | a%100 in {34..35} (+) | 0.51 | 0.052 |
| L23 o c370 | H22 | a%100 in {55} (-) | 0.57 | 0.011 |

**code 4s-580.2 (L27): 1 comps, tells apart 8/100, coverage 0.05, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 down c33 | MLP | a%100 in {95..99} (-) | 0.65 | 0.127 |

</details>

</details>

<details><summary><b>4s-582</b> `a%100` @ `=` (sub) — single component; 1 comps, L16; tells apart 4/100 classes (best member 4); on add: 4a-449 (member overlap 0.33)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.01 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 100) |
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
| input point | L15 MLP |
| CKA(arrangement before, joint write) | 0.17 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.03 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.42 2:0.14 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.07 2:0.06 3:0.05) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.01 / 1.00 |
| best source position (CKA) | `op` (0.22) |

<details><summary>codes and components</summary>

**code 4s-582.0 (L16): 1 comps, tells apart 4/100, coverage 0.01, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c61 | H22 | a%100 in {1} (+) | 0.58 | 0.028 |

</details>

</details>

<details><summary><b>4s-581</b> `a%100` @ `=` (sub) — block code, copy from `op`; 25 comps, L16; tells apart 85/100 classes (best member 8); on add: 4a-448 (member overlap 0.39)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.96 |
| classes told apart: joint / best code / best member | 85 /  / 8 (of 100) |
| members whose removal merges classes | 0.28 |
| support overlap (1 = tiling) / random sets / p | 2.99 / 2.34 / 1.00 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.77 |
| decoding acc. joint / best code / best member (chance) | 0.66 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.15 |
| consumers / read jointly | 78 / 71 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 MLP |
| CKA(arrangement before, joint write) | 0.41 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.09 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.28 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.42 2:0.14 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (1:0.33 2:0.15 5:0.12) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.55 / 6.52 |
| best source position (CKA) | `op` (0.56) |

<details><summary>codes and components</summary>

**code 4s-581.0 (L16): 25 comps, tells apart 85/100, coverage 0.96, overlap 2.99 (random 2.37, p 1.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c195 | H21 | a%100 in {0, 19..20, 38..40, 58..60, 70, 78..80, 89..90, 98..99} (+) | 0.78 | 0.355 |
| L16 o c34 | H21 | a%100 in {0, 21, 31, 40..42, 51..52, 61..62, 71, 80..82, 91} (+) | 0.77 | 0.238 |
| L16 o c129 | H21 | a%100 in {0, 50, 60, 62, 64, 66, 68, 70, 72, 80, 90} (-) | 0.81 | 0.581 |
| L16 o c35 | H21 | a%100 in {0, 98..99} (+) | 0.91 | 0.062 |
| L16 o c89 | H21 | a%100 in {0, 99} (-) | 0.92 | 0.019 |
| L16 o c219 | H21 | a%100 in {1..19} (+) | 0.82 | 0.224 |
| L16 o c48 | H21 | a%100 in {13, 17, 19, 57, 59, 67, 69, 79, 89, 99} (-) | 0.63 | 0.314 |
| L16 o c144 | H21 | a%100 in {25..33, 46..53, 65..74, 85..93} (+) | 0.84 | 0.444 |
| L16 o c142 | H21 | a%100 in {26, 36, 46, 56, 58, 66, 76..78, 86, 88, 96} (-) | 0.65 | 0.192 |
| L16 o c41 | H21 | a%100 in {26..40} (+) | 0.76 | 0.227 |
| L16 o c154 | H21 | a%100 in {29, 59, 69, 79, 89, 99} (-) | 0.65 | 0.074 |
| L16 o c163 | H21 | a%100 in {32, 52, 62, 72, 82, 92} (-) | 0.70 | 0.060 |
| L16 o c138 | H21 | a%100 in {33..57} (-) | 0.82 | 0.461 |
| L16 o c137 | H21 | a%100 in {34, 44, 54, 74, 84, 94} (-) | 0.75 | 0.097 |
| L16 o c116 | H21 | a%100 in {36, 56, 64, 66, 74, 76, 84, 94..96} (-) | 0.73 | 0.189 |
| L16 o c90 | H21 | a%100 in {38..43, 81..84} (+) | 0.74 | 0.228 |
| L16 o c164 | H21 | a%100 in {40..41, 43..44, 46..47, 49..52} (-) | 0.55 | 0.096 |
| L16 o c120 | H21 | a%100 in {50..59} (-) | 0.78 | 0.165 |
| L16 o c25 | H21 | a%100 in {55, 75, 77, 95} (+) | 0.61 | 0.119 |
| L16 o c28 | H21 | a%100 in {56..63} (+) | 0.78 | 0.275 |
| L16 o c94 | H21 | a%100 in {58..65} (-) | 0.80 | 0.154 |
| L16 o c72 | H21 | a%100 in {65..69, 85..89} (-) | 0.64 | 0.072 |
| L16 o c82 | H21 | a%100 in {65..74} (-) | 0.67 | 0.085 |
| L16 o c31 | H21 | a%100 in {76..82, 85..91} (-) | 0.74 | 0.222 |
| L16 o c155 | H21 | a%100 in {85..94} (-) | 0.76 | 0.114 |

</details>

</details>

<details><summary><b>4s-584</b> `a%100` @ `=` (sub) — single component; 1 comps, L20; tells apart 3/100 classes (best member 3); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.02 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.85 |
| decoding acc. joint / best code / best member (chance) | 0.02 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L20 attn |
| CKA(arrangement before, joint write) | 0.12 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.02 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.29 2:0.15 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.05) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.02 / 1.00 |

<details><summary>codes and components</summary>

**code 4s-584.0 (L20): 1 comps, tells apart 3/100, coverage 0.02, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 down c172 | MLP | a%100 in {0, 99} (+) | 0.85 | 0.025 |

</details>

</details>

<details><summary><b>4s-583</b> `a%100` @ `=` (sub) — block code; 6 comps, L20; tells apart 16/100 classes (best member 8); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.15 |
| classes told apart: joint / best code / best member | 16 /  / 8 (of 100) |
| members whose removal merges classes | 0.67 |
| support overlap (1 = tiling) / random sets / p | 1.60 / 1.22 / 0.95 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.67 |
| decoding acc. joint / best code / best member (chance) | 0.09 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.38 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L19 MLP |
| CKA(arrangement before, joint write) | 0.14 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.07 |
| share of the write inside the old arrangement's span | 0.14 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.33 2:0.16 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (40:0.15 20:0.15 10:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.07 / 2.25 |
| best source position (CKA) | `b` (0.20) |

<details><summary>codes and components</summary>

**code 4s-583.0 (L20): 6 comps, tells apart 16/100, coverage 0.15, overlap 1.60 (random 1.23, p 0.94)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 o c252 | H2 | a%100 in {12, 18, 36, 45, 48, 72, 90} (-) | 0.51 | 0.118 |
| L20 o c479 | H2 | a%100 in {20, 30, 40, 45, 60, 70, 80, 90} (+) | 0.62 | 0.155 |
| L20 o c192 | H2 | a%100 in {30, 45, 60, 70, 90} (+) | 0.60 | 0.120 |
| L20 o c243 | H2 | a%100 in {45} (+) | 0.79 | 0.017 |
| L20 o c179 | H2 | a%100 in {98..99} (-) | 0.87 | 0.033 |
| L20 o c396 | H2 | a%100 in {99} (+) | 0.78 | 0.109 |

</details>

</details>

<details><summary><b>4s-585</b> `a%100` @ `=` (sub) — block code; 3 comps, L22 L23; tells apart 7/100 classes (best member 7); on add: 4a-450 (member overlap 0.75)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.04 |
| classes told apart: joint / best code / best member | 7 /  / 7 (of 100) |
| members whose removal merges classes | 0.67 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.05 / 0.42 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.77 |
| decoding acc. joint / best code / best member (chance) | 0.06 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 MLP |
| CKA(arrangement before, joint write) | 0.06 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.30 2:0.15 3:0.07) |
| joint write: shape (spectrum k:share) | irregular (40:0.06 30:0.06 20:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.03 / 2.21 |
| best source position (CKA) | `b` (0.07) |

<details><summary>codes and components</summary>

**code 4s-585.0 (L22 L23): 3 comps, tells apart 7/100, coverage 0.04, overlap 1.00 (random 1.05, p 0.39)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 o c246 | H3 | a%100 in {47, 57} (+) | 0.78 | 0.059 |
| L22 o c352 | H3 | a%100 in {97} (-) | 0.83 | 0.016 |
| L23 o c187 | H7 | a%100 in {98} (-) | 0.65 | 0.041 |

</details>

</details>

<details><summary><b>4s-586</b> `a%100` @ `=` (sub) — copy from `b`; 1 comps, L27; tells apart 3/100 classes (best member 3); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.07 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.55 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L26 MLP |
| CKA(arrangement before, joint write) | 0.37 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.13 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.27 2:0.13 3:0.06) |
| joint write: shape (spectrum k:share) | line (1:0.45 4:0.14 2:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.12 / 1.00 |
| best source position (CKA) | `b` (0.56) |

<details><summary>codes and components</summary>

**code 4s-586.0 (L27): 1 comps, tells apart 3/100, coverage 0.07, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 o c5 | H14 | a%100 in {93..99} (+) | 0.54 | 1.000 |

</details>

</details>

</details>

<details><summary>`res%100`: 10 mechanisms, 75 components</summary>

<details><summary><b>4s-623</b> `res%100` @ `=` (sub) — block code; 2 comps, L13; tells apart 7/100 classes (best member 7); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.06 |
| classes told apart: joint / best code / best member | 7 /  / 7 (of 100) |
| members whose removal merges classes | 0.50 |
| support overlap (1 = tiling) / random sets / p | 1.33 / 1.00 / 0.91 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.74 |
| decoding acc. joint / best code / best member (chance) | 0.03 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.33 |
| consumers / read jointly | 1 / 1 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L12 MLP |
| CKA(arrangement before, joint write) | 0.46 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.17 |
| share of the write inside the old arrangement's span | 0.18 |
| write energy / code energy before | 0.03 |
| arrangement before: shape (spectrum k:share) | circle period 100 (1:0.67 2:0.12) |
| joint write: shape (spectrum k:share) | line (1:0.33 2:0.19 3:0.11) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.09 / 1.05 |
| best source position (CKA) | `b` (0.38) |

<details><summary>codes and components</summary>

**code 4s-623.0 (L13): 2 comps, tells apart 7/100, coverage 0.06, overlap 1.33 (random 1.00, p 0.92)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 o c231 | H7 | res%100 in {0..1, 99} (+) | 0.69 | 0.059 |
| L13 o c304 | H7 | res%100 in {0..4} (-) | 0.73 | 0.206 |

</details>

</details>

<details><summary><b>4s-624</b> `res%100` @ `=` (sub) — single component; 1 comps, L19; tells apart 4/100 classes (best member 4); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.05 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.52 |
| decoding acc. joint / best code / best member (chance) | 0.02 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L19 attn |
| CKA(arrangement before, joint write) | 0.15 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.04 |
| share of the write inside the old arrangement's span | 0.09 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.46 2:0.12 50:0.10) |
| joint write: shape (spectrum k:share) | line (10:0.37 20:0.21 30:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.10 / 1.00 |

<details><summary>codes and components</summary>

**code 4s-624.0 (L19): 1 comps, tells apart 4/100, coverage 0.05, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L19 down c55 | MLP | res%100 in {1, 10..11, 20..21} (+) | 0.52 | 0.146 |

</details>

</details>

<details><summary><b>4s-626</b> `res%100` @ `=` (sub) — block code, 4 codes of the same shape; 18 comps, L20 L22 L26 L27; tells apart 18/100 classes (best member 7); on add: 4a-508 (member overlap 0.08)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.12 |
| classes told apart: joint / best code / best member | 18 / 13 / 7 (of 100) |
| members whose removal merges classes | 0.28 |
| support overlap (1 = tiling) / random sets / p | 5.92 / 2.54 / 1.00 |
| mean CKA between its codes | 0.85 |
| purity of the joint write (per prompt) | 0.71 |
| decoding acc. joint / best code / best member (chance) | 0.12 / 0.09 / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.14 |
| consumers / read jointly | 30 / 28 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L20 attn |
| CKA(arrangement before, joint write) | 0.62 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.02 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.21 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.40 2:0.18 50:0.10) |
| joint write: shape (spectrum k:share) | irregular (1:0.28 2:0.19 3:0.13) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.11 / 1.55 |

<details><summary>codes and components</summary>

**code 4s-626.0 (L20): 1 comps, tells apart 5/100, coverage 0.11, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 down c31 | MLP | res%100 in {1..11} (+) | 0.66 | 0.254 |

**code 4s-626.1 (L22): 3 comps, tells apart 12/100, coverage 0.11, overlap 1.73 (random 1.14, p 0.97)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L22 down c376 | MLP | res%100 in {1..11} (+) | 0.72 | 0.221 |
| L22 down c6 | MLP | res%100 in {4..10} (-) | 0.75 | 0.135 |
| L22 down c129 | MLP | res%100 in {6} (+) | 0.62 | 0.013 |

**code 4s-626.2 (L26): 9 comps, tells apart 11/100, coverage 0.11, overlap 1.73 (random 1.64, p 0.58)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 down c139 | MLP | res%100 in {0..2} (+) | 0.54 | 0.073 |
| L26 down c779 | MLP | res%100 in {10} (+) | 0.72 | 0.008 |
| L26 down c486 | MLP | res%100 in {2} (+) | 0.77 | 0.009 |
| L26 down c525 | MLP | res%100 in {2} (+) | 0.73 | 0.009 |
| L26 down c552 | MLP | res%100 in {3..4} (-) | 0.74 | 0.144 |
| L26 down c106 | MLP | res%100 in {3} (+) | 0.68 | 0.008 |
| L26 down c23 | MLP | res%100 in {4..8} (+) | 0.72 | 0.050 |
| L26 down c180 | MLP | res%100 in {6, 10} (+) | 0.58 | 0.019 |
| L26 down c207 | MLP | res%100 in {7..9} (-) | 0.78 | 0.026 |

**code 4s-626.3 (L27): 5 comps, tells apart 13/100, coverage 0.11, overlap 2.00 (random 1.33, p 0.94)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 down c227 | MLP | res%100 in {1..11} (-) | 0.67 | 0.149 |
| L27 down c139 | MLP | res%100 in {1} (-) | 0.71 | 0.015 |
| L27 down c137 | MLP | res%100 in {3, 5..9} (-) | 0.57 | 0.043 |
| L27 down c73 | MLP | res%100 in {3..4} (-) | 0.52 | 0.045 |
| L27 down c62 | MLP | res%100 in {5..6} (-) | 0.70 | 0.019 |

</details>

</details>

<details><summary><b>4s-625</b> `res%100` @ `=` (sub) — block code; 7 comps, L21; tells apart 31/100 classes (best member 8); on add: 4a-503 (member overlap 0.09)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.63 |
| classes told apart: joint / best code / best member | 31 /  / 8 (of 100) |
| members whose removal merges classes | 0.86 |
| support overlap (1 = tiling) / random sets / p | 1.51 / 1.53 / 0.49 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.65 |
| decoding acc. joint / best code / best member (chance) | 0.18 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.03 |
| consumers / read jointly | 73 / 56 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L21 attn |
| CKA(arrangement before, joint write) | 0.68 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.12 |
| share of the write inside the old arrangement's span | 0.13 |
| write energy / code energy before | 0.16 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.32 2:0.16 10:0.11) |
| joint write: shape (spectrum k:share) | irregular (50:0.25 1:0.18 2:0.12) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.38 / 3.84 |

<details><summary>codes and components</summary>

**code 4s-625.0 (L21): 7 comps, tells apart 31/100, coverage 0.63, overlap 1.51 (random 1.65, p 0.39)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c1 | MLP | res%100 in {0..25, 27, 29..31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 53, 55, 57, 59, 63, 76, 78, 80, 82, 84, 86..90, 92..97} (-) | 0.66 | 0.565 |
| L21 down c10 | MLP | res%100 in {1, 11, 21, 31, 81, 91} (-) | 0.66 | 0.182 |
| L21 down c9 | MLP | res%100 in {1..3, 5..13} (+) | 0.57 | 0.933 |
| L21 down c7 | MLP | res%100 in {1..8} (-) | 0.67 | 0.325 |
| L21 down c390 | MLP | res%100 in {2} (+) | 0.62 | 0.015 |
| L21 down c14 | MLP | res%100 in {5, 15, 25, 95} (+) | 0.71 | 0.101 |
| L21 down c16 | MLP | res%100 in {6, 16, 26, 96} (+) | 0.57 | 0.099 |

</details>

</details>

<details><summary><b>4s-627</b> `res%100` @ `=` (sub) — block code; 3 comps, L23; tells apart 10/100 classes (best member 6); on add: 4a-505 (member overlap 0.08)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.13 |
| classes told apart: joint / best code / best member | 10 /  / 6 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.15 / 1.14 / 0.53 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.67 |
| decoding acc. joint / best code / best member (chance) | 0.06 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 11 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L23 attn |
| CKA(arrangement before, joint write) | 0.28 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.11 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.29 2:0.16 10:0.13) |
| joint write: shape (spectrum k:share) | irregular (10:0.16 40:0.13 20:0.12) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.09 / 1.70 |

<details><summary>codes and components</summary>

**code 4s-627.0 (L23): 3 comps, tells apart 10/100, coverage 0.13, overlap 1.15 (random 1.14, p 0.53)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 down c196 | MLP | res%100 in {2..8} (-) | 0.74 | 0.079 |
| L23 down c105 | MLP | res%100 in {3, 23} (-) | 0.47 | 0.014 |
| L23 down c9 | MLP | res%100 in {6, 16, 26, 36, 86, 96} (-) | 0.64 | 0.141 |

</details>

</details>

<details><summary><b>4s-628</b> `res%100` @ `=` (sub) — block code; 8 comps, L24; tells apart 15/100 classes (best member 6); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.19 |
| classes told apart: joint / best code / best member | 15 /  / 6 (of 100) |
| members whose removal merges classes | 0.50 |
| support overlap (1 = tiling) / random sets / p | 1.26 / 1.60 / 0.10 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.70 |
| decoding acc. joint / best code / best member (chance) | 0.09 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 16 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L24 attn |
| CKA(arrangement before, joint write) | 0.25 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.07 |
| share of the write inside the old arrangement's span | 0.15 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.28 2:0.16 10:0.12) |
| joint write: shape (spectrum k:share) | irregular (10:0.16 20:0.15 40:0.14) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.12 / 2.30 |

<details><summary>codes and components</summary>

**code 4s-628.0 (L24): 8 comps, tells apart 15/100, coverage 0.19, overlap 1.26 (random 1.60, p 0.12)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 down c678 | MLP | res%100 in {1..3} (+) | 0.72 | 0.059 |
| L24 down c46 | MLP | res%100 in {10, 20, 30, 70, 90} (+) | 0.56 | 0.086 |
| L24 down c240 | MLP | res%100 in {4} (+) | 0.50 | 0.022 |
| L24 down c482 | MLP | res%100 in {4} (-) | 0.74 | 0.008 |
| L24 down c12 | MLP | res%100 in {5, 15, 25, 95} (-) | 0.72 | 0.109 |
| L24 down c501 | MLP | res%100 in {5..7} (-) | 0.77 | 0.025 |
| L24 down c348 | MLP | res%100 in {6..7} (-) | 0.81 | 0.018 |
| L24 down c158 | MLP | res%100 in {9..13} (-) | 0.69 | 0.135 |

</details>

</details>

<details><summary><b>4s-629</b> `res%100` @ `=` (sub) — block code; 6 comps, L25; tells apart 18/100 classes (best member 8); on add: 4a-507 (member overlap 0.15)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.11 |
| classes told apart: joint / best code / best member | 18 /  / 8 (of 100) |
| members whose removal merges classes | 0.83 |
| support overlap (1 = tiling) / random sets / p | 1.36 / 1.44 / 0.43 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.69 |
| decoding acc. joint / best code / best member (chance) | 0.10 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 6 / 4 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L25 attn |
| CKA(arrangement before, joint write) | 0.46 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.26 2:0.15 10:0.12) |
| joint write: shape (spectrum k:share) | irregular (1:0.13 2:0.10 3:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.09 / 2.52 |

<details><summary>codes and components</summary>

**code 4s-629.0 (L25): 6 comps, tells apart 18/100, coverage 0.11, overlap 1.36 (random 1.50, p 0.34)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L25 down c40 | MLP | res%100 in {0..1} (+) | 0.53 | 0.034 |
| L25 down c1005 | MLP | res%100 in {1} (+) | 0.65 | 0.121 |
| L25 down c748 | MLP | res%100 in {1} (-) | 0.77 | 0.013 |
| L25 down c15 | MLP | res%100 in {2..5} (-) | 0.77 | 0.079 |
| L25 down c31 | MLP | res%100 in {6..9} (+) | 0.58 | 0.070 |
| L25 down c48 | MLP | res%100 in {8..10} (+) | 0.65 | 0.118 |

</details>

</details>

<details><summary><b>4s-630</b> `res%100` @ `=` (sub) — tiling; 13 comps, L28; tells apart 14/100 classes (best member 9); on add: 4a-510 (member overlap 0.06)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.11 |
| classes told apart: joint / best code / best member | 14 /  / 9 (of 100) |
| members whose removal merges classes | 0.23 |
| support overlap (1 = tiling) / random sets / p | 1.36 / 2.10 / 0.01 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.73 |
| decoding acc. joint / best code / best member (chance) | 0.13 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.07 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L28 attn |
| CKA(arrangement before, joint write) | 0.47 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.08 |
| share of the write inside the old arrangement's span | 0.09 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.25 2:0.13 10:0.09) |
| joint write: shape (spectrum k:share) | irregular (1:0.07 10:0.06 2:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.09 / 5.68 |

<details><summary>codes and components</summary>

**code 4s-630.0 (L28): 13 comps, tells apart 14/100, coverage 0.11, overlap 1.36 (random 2.00, p 0.01)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 down c73 | MLP | res%100 in {10} (+) | 0.75 | 0.029 |
| L28 down c211 | MLP | res%100 in {11} (+) | 0.50 | 0.060 |
| L28 down c12 | MLP | res%100 in {11} (-) | 0.54 | 0.082 |
| L28 down c660 | MLP | res%100 in {1} (+) | 0.69 | 0.016 |
| L28 down c408 | MLP | res%100 in {1} (-) | 0.55 | 0.014 |
| L28 down c357 | MLP | res%100 in {2} (-) | 0.77 | 0.012 |
| L28 down c430 | MLP | res%100 in {3} (-) | 0.77 | 0.010 |
| L28 down c140 | MLP | res%100 in {4} (-) | 0.74 | 0.029 |
| L28 down c588 | MLP | res%100 in {5} (-) | 0.77 | 0.025 |
| L28 down c69 | MLP | res%100 in {6} (-) | 0.67 | 0.111 |
| L28 down c416 | MLP | res%100 in {7..9} (-) | 0.73 | 0.088 |
| L28 down c247 | MLP | res%100 in {7} (-) | 0.81 | 0.051 |
| L28 down c435 | MLP | res%100 in {8} (+) | 0.81 | 0.016 |

</details>

</details>

<details><summary><b>4s-631</b> `res%100` @ `=` (sub) — block code; 14 comps, L29; tells apart 9/100 classes (best member 5); on add: 4a-511 (member overlap 0.03)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.11 |
| classes told apart: joint / best code / best member | 9 /  / 5 (of 100) |
| members whose removal merges classes | 0.29 |
| support overlap (1 = tiling) / random sets / p | 1.82 / 2.21 / 0.27 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.80 |
| decoding acc. joint / best code / best member (chance) | 0.11 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.06 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L29 attn |
| CKA(arrangement before, joint write) | 0.23 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.11 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.25 2:0.13 10:0.09) |
| joint write: shape (spectrum k:share) | irregular () |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.05 / 4.41 |

<details><summary>codes and components</summary>

**code 4s-631.0 (L29): 14 comps, tells apart 9/100, coverage 0.11, overlap 1.82 (random 2.25, p 0.27)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L29 down c514 | MLP | res%100 in {1..2} (-) | 0.76 | 0.030 |
| L29 down c48 | MLP | res%100 in {10..11} (+) | 0.69 | 0.022 |
| L29 down c390 | MLP | res%100 in {1} (-) | 0.82 | 0.011 |
| L29 down c805 | MLP | res%100 in {3, 7..8} (-) | 0.69 | 0.022 |
| L29 down c177 | MLP | res%100 in {3} (-) | 0.60 | 0.007 |
| L29 down c928 | MLP | res%100 in {3} (-) | 0.77 | 0.009 |
| L29 down c761 | MLP | res%100 in {4..5} (-) | 0.66 | 0.027 |
| L29 down c103 | MLP | res%100 in {5, 9} (+) | 0.70 | 0.024 |
| L29 down c469 | MLP | res%100 in {5} (+) | 0.79 | 0.015 |
| L29 down c122 | MLP | res%100 in {6} (+) | 0.82 | 0.015 |
| L29 down c68 | MLP | res%100 in {6} (-) | 0.63 | 0.018 |
| L29 down c94 | MLP | res%100 in {7} (-) | 0.82 | 0.016 |
| L29 down c134 | MLP | res%100 in {8} (+) | 0.83 | 0.019 |
| L29 down c635 | MLP | res%100 in {9} (-) | 0.80 | 0.021 |

</details>

</details>

<details><summary><b>4s-632</b> `res%100` @ `=` (sub) — block code, 2 codes of the same shape; 3 comps, L30 L31; tells apart 4/100 classes (best member 3); on add: 4a-512 (member overlap 0.05)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.02 |
| classes told apart: joint / best code / best member | 4 / 4 / 3 (of 100) |
| members whose removal merges classes | 0.33 |
| support overlap (1 = tiling) / random sets / p | 1.50 / 1.17 / 0.94 |
| mean CKA between its codes | 0.74 |
| purity of the joint write (per prompt) | 0.63 |
| decoding acc. joint / best code / best member (chance) | 0.03 / 0.03 / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L30 attn |
| CKA(arrangement before, joint write) | 0.18 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.24 2:0.12 10:0.08) |
| joint write: shape (spectrum k:share) | irregular () |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.02 / 1.80 |

<details><summary>codes and components</summary>

**code 4s-632.0 (L30): 2 comps, tells apart 4/100, coverage 0.02, overlap 1.00 (random 1.00, p 0.66)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 down c810 | MLP | res%100 in {2} (+) | 0.63 | 0.009 |
| L30 down c309 | MLP | res%100 in {5} (-) | 0.62 | 0.014 |

**code 4s-632.1 (L31): 1 comps, tells apart 3/100, coverage 0.01, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L31 down c85 | MLP | res%100 in {5} (+) | 0.49 | 0.013 |

</details>

</details>

</details>

<details><summary>`units(a,b)`: 8 mechanisms, 42 components</summary>

<details><summary><b>4s-674</b> `units(a,b)` @ `=` (sub) — block code; 5 comps, L16; tells apart 31/100 classes (best member 6); on add: 4a-566 (member overlap 0.83)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.49 |
| classes told apart: joint / best code / best member | 31 /  / 6 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.22 / 1.25 / 0.40 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.62 |
| decoding acc. joint / best code / best member (chance) | 0.16 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.93 |
| consumers / read jointly | 1 / 1 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L16 attn |
| CKA(arrangement before, joint write) | 0.40 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.08 |
| share of the write inside the old arrangement's span | 0.12 |
| write energy / code energy before | 0.06 |

<details><summary>codes and components</summary>

**code 4s-674.0 (L16): 5 comps, tells apart 31/100, coverage 0.49, overlap 1.22 (random 1.26, p 0.36)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c79 | MLP | a%10 {0} x b%10 {0, 4..6, 9}; a%10 {1, 6} x b%10 {0, 5}; a%10 {5} x b%10 {0, 4..5} (+) | 0.78 | 0.282 |
| L16 down c94 | MLP | a%10 {0} x b%10 {0..1, 5..6}; a%10 {2} x b%10 {0..1, 6}; a%10 {5} x b%10 {0}; a%10 {7} x b%10 {1, 6} (+) | 0.51 | 0.218 |
| L16 down c127 | MLP | a%10 {1, 6} x b%10 {1..2, 6..7}; a%10 {2} x b%10 {1..2, 7}; a%10 {7} x b%10 {1..2} (-) | 0.69 | 0.181 |
| L16 down c107 | MLP | a%10 {3..7} x b%10 {5..7}; a%10 {8} x b%10 {7} (+) | 0.50 | 0.154 |
| L16 down c275 | MLP | a%10 {3} x b%10 {4, 8..9}; a%10 {4, 8..9} x b%10 {4, 9} (-) | 0.56 | 0.143 |

</details>

</details>

<details><summary><b>4s-675</b> `units(a,b)` @ `=` (sub) — block code; 7 comps, L17; tells apart 57/100 classes (best member 7); on add: 4a-567 (member overlap 0.54)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.82 |
| classes told apart: joint / best code / best member | 57 /  / 7 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.48 / 1.42 / 0.64 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.60 |
| decoding acc. joint / best code / best member (chance) | 0.33 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.06 |
| consumers / read jointly | 9 / 7 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L17 attn |
| CKA(arrangement before, joint write) | 0.37 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.10 |
| share of the write inside the old arrangement's span | 0.27 |
| write energy / code energy before | 0.15 |

<details><summary>codes and components</summary>

**code 4s-675.0 (L17): 7 comps, tells apart 57/100, coverage 0.82, overlap 1.48 (random 1.41, p 0.66)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 down c39 | MLP | a%10 {0, 9} x b%10 {1..4}; a%10 {3..4} x b%10 {2..4}; a%10 {8} x b%10 {1, 3..4} (-) | 0.56 | 0.235 |
| L17 down c37 | MLP | a%10 {0} x b%10 {0..2, 8..9}; a%10 {1} x b%10 {0..2, 9}; a%10 {2} x b%10 {0..1}; a%10 {8..9} x b%10 {0} (+) | 0.66 | 0.310 |
| L17 down c86 | MLP | a%10 {0} x b%10 {2..3}; a%10 {1..2} x b%10 {2..4}; a%10 {4} x b%10 {0..1, 9}; a%10 {5} x b%10 {1} (-) | 0.51 | 0.165 |
| L17 down c22 | MLP | a%10 {1, 3, 5, 7, 9} x b%10 {0, 2, 4, 6, 8} (+) | 0.68 | 0.254 |
| L17 down c67 | MLP | a%10 {1, 3, 5, 7, 9} x b%10 {1, 3, 5, 7, 9} (-) | 0.61 | 0.202 |
| L17 down c49 | MLP | a%10 {5, 8} x b%10 {4, 8..9}; a%10 {6} x b%10 {0..1, 4, 7..9}; a%10 {7} x b%10 {0, 4, 8..9} (-) | 0.55 | 0.271 |
| L17 down c34 | MLP | a%10 {5} x b%10 {0..4}; a%10 {6..7} x b%10 {1..4} (+) | 0.52 | 0.214 |

</details>

</details>

<details><summary><b>4s-676</b> `units(a,b)` @ `=` (sub) — block code, copy from `b`; 3 comps, L18 L19; tells apart 12/100 classes (best member 6); on add: 4a-568 (member overlap 0.25)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.21 |
| classes told apart: joint / best code / best member | 12 /  / 6 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 / 1.11 / 0.09 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.74 |
| decoding acc. joint / best code / best member (chance) | 0.09 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 0.99 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L17 MLP |
| CKA(arrangement before, joint write) | 0.44 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.09 |
| share of the write inside the old arrangement's span | 0.17 |
| write energy / code energy before | 0.05 |
| best source position (CKA) | `b` (0.57) |

<details><summary>codes and components</summary>

**code 4s-676.0 (L18 L19): 3 comps, tells apart 12/100, coverage 0.21, overlap 1.00 (random 1.12, p 0.09)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 o c195 | H30 | a%10 {0, 4..6, 9} x b%10 {4} (+) | 0.57 | 0.101 |
| L18 o c313 | H30 | a%10 {0} x b%10 {0, 5}; a%10 {2, 5} x b%10 {0} (+) | 0.84 | 0.213 |
| L19 down c41 | MLP | a%10 {0} x b%10 {1..3}; a%10 {7..8} x b%10 {0..1, 9}; a%10 {9} x b%10 {0..2} (+) | 0.62 | 0.269 |

</details>

</details>

<details><summary><b>4s-677</b> `units(a,b)` @ `=` (sub) — block code, 4 codes of the same shape; 15 comps, L18 L23 L26 L28; tells apart 44/100 classes (best member 8); on add: 4a-569 (member overlap 0.28)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.71 |
| classes told apart: joint / best code / best member | 44 / 49 / 8 (of 100) |
| members whose removal merges classes | 0.60 |
| support overlap (1 = tiling) / random sets / p | 2.70 / 2.21 / 0.95 |
| mean CKA between its codes | 0.81 |
| purity of the joint write (per prompt) | 0.77 |
| decoding acc. joint / best code / best member (chance) | 0.45 / 0.41 / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.21 |
| consumers / read jointly | 101 / 72 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L18 attn |
| CKA(arrangement before, joint write) | 0.37 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.14 |
| share of the write inside the old arrangement's span | 0.31 |
| write energy / code energy before | 0.56 |

<details><summary>codes and components</summary>

**code 4s-677.0 (L18): 8 comps, tells apart 49/100, coverage 0.70, overlap 1.46 (random 1.49, p 0.38)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 down c4 | MLP | a%10 {0, 2, 4, 6, 8} x b%10 {0, 2, 4, 6, 8} (-) | 0.87 | 0.404 |
| L18 down c62 | MLP | a%10 {0..1, 9} x b%10 {0..1, 9} (-) | 0.83 | 0.278 |
| L18 down c65 | MLP | a%10 {0} x b%10 {1..3}; a%10 {1..2} x b%10 {1..4}; a%10 {3} x b%10 {2..3} (-) | 0.69 | 0.291 |
| L18 down c375 | MLP | a%10 {0} x b%10 {2..3, 7..8}; a%10 {5} x b%10 {2, 7..8} (-) | 0.58 | 0.256 |
| L18 down c24 | MLP | a%10 {0} x b%10 {3..4, 8}; a%10 {4} x b%10 {2..4, 8}; a%10 {5} x b%10 {3..4}; a%10 {9} x b%10 {2..4, 7..8} (+) | 0.64 | 0.506 |
| L18 down c98 | MLP | a%10 {0} x b%10 {3..5}; a%10 {1} x b%10 {4..6}; a%10 {2..3} x b%10 {1..2}; a%10 {4} x b%10 {2}; a%10 {9} x b%10 {3..4} (+) | 0.61 | 0.272 |
| L18 down c32 | MLP | a%10 {3, 5} x b%10 {4..6}; a%10 {4} x b%10 {3..6}; a%10 {6} x b%10 {5} (-) | 0.64 | 0.297 |
| L18 down c120 | MLP | a%10 {6} x b%10 {6}; a%10 {7} x b%10 {5..7}; a%10 {8} x b%10 {5..8}; a%10 {9} x b%10 {6..7} (+) | 0.55 | 0.156 |

**code 4s-677.1 (L23): 2 comps, tells apart 6/100, coverage 0.27, overlap 1.04 (random 1.04, p 0.47)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 down c387 | MLP | a%10 {0, 5} x b%10 {0, 5} (-) | 0.88 | 0.042 |
| L23 down c55 | MLP | a%10 {0} x b%10 {0, 2, 6, 8}; a%10 {2, 4, 6, 8} x b%10 {0, 2, 4, 6, 8} (-) | 0.79 | 0.223 |

**code 4s-677.2 (L26): 1 comps, tells apart 3/100, coverage 0.25, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 down c36 | MLP | a%10 {0, 2, 4, 6, 8} x b%10 {0, 2, 4, 6, 8} (+) | 0.71 | 0.299 |

**code 4s-677.3 (L28): 4 comps, tells apart 13/100, coverage 0.31, overlap 1.19 (random 1.18, p 0.55)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 down c359 | MLP | a%10 {0} x b%10 {0, 5}; a%10 {1} x b%10 {1}; a%10 {5} x b%10 {5} (+) | 0.52 | 0.070 |
| L28 down c23 | MLP | a%10 {0} x b%10 {2, 4, 6, 8}; a%10 {2, 4, 6, 8} x b%10 {0, 2, 4, 6, 8} (+) | 0.63 | 0.194 |
| L28 down c93 | MLP | a%10 {0} x b%10 {5}; a%10 {5} x b%10 {0} (-) | 0.64 | 0.084 |
| L28 down c36 | MLP | a%10 {0} x b%10 {8}; a%10 {2} x b%10 {0}; a%10 {4} x b%10 {2}; a%10 {5} x b%10 {3}; a%10 {6} x b%10 {4}; a%10 {7} x b%10 {5}; a%10 {8} x b%10 {6} (-) | 0.55 | 0.130 |

</details>

</details>

<details><summary><b>4s-678</b> `units(a,b)` @ `=` (sub) — single component; 1 comps, L20; tells apart 5/100 classes (best member 5); on add: 4a-571 (member overlap 0.20)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.08 |
| classes told apart: joint / best code / best member | 5 /  / 5 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.72 |
| decoding acc. joint / best code / best member (chance) | 0.03 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L19 MLP |
| CKA(arrangement before, joint write) | 0.19 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.13 |
| write energy / code energy before | 0.01 |
| best source position (CKA) | `a` (0.27) |

<details><summary>codes and components</summary>

**code 4s-678.0 (L20): 1 comps, tells apart 5/100, coverage 0.08, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 o c288 | H2 | a%10 {0} x b%10 {5}; a%10 {5} x b%10 {0..2, 5..8} (-) | 0.72 | 0.134 |

</details>

</details>

<details><summary><b>4s-679</b> `units(a,b)` @ `=` (sub) — block code; 3 comps, L20; tells apart 19/100 classes (best member 2); on add: 4a-570 (member overlap 0.25)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.52 |
| classes told apart: joint / best code / best member | 19 /  / 2 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.15 / 1.12 / 0.67 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.59 |
| decoding acc. joint / best code / best member (chance) | 0.10 /  / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 7 / 6 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L20 attn |
| CKA(arrangement before, joint write) | 0.33 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.11 |
| share of the write inside the old arrangement's span | 0.32 |
| write energy / code energy before | 0.06 |

<details><summary>codes and components</summary>

**code 4s-679.0 (L20): 3 comps, tells apart 19/100, coverage 0.52, overlap 1.15 (random 1.11, p 0.68)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L20 down c21 | MLP | a%10 {0} x b%10 {1..2}; a%10 {1} x b%10 {1..3}; a%10 {2} x b%10 {2..3}; a%10 {4..5} x b%10 {6}; a%10 {6} x b%10 {7..8}; a%10 {7} x b%10 {8..9}; a%10 {8} x b%10 {0..1, 8..9}; a%10 {9} x b%10 {0..2, 9} (-) | 0.58 | 0.374 |
| L20 down c15 | MLP | a%10 {0} x b%10 {9}; a%10 {1} x b%10 {0}; a%10 {4, 9} x b%10 {3..4, 8..9}; a%10 {5} x b%10 {4, 9}; a%10 {6} x b%10 {5, 9} (-) | 0.51 | 0.463 |
| L20 down c47 | MLP | a%10 {1, 3, 5, 7, 9} x b%10 {1, 3, 5, 7, 9} (-) | 0.74 | 0.282 |

</details>

</details>

<details><summary><b>4s-680</b> `units(a,b)` @ `=` (sub) — tiling; 5 comps, L24 L25; tells apart 12/100 classes (best member 8); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.21 |
| classes told apart: joint / best code / best member | 12 /  / 8 (of 100) |
| members whose removal merges classes | 0.60 |
| support overlap (1 = tiling) / random sets / p | 1.05 / 1.26 / 0.01 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.61 |
| decoding acc. joint / best code / best member (chance) | 0.06 /  / 0.02 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 29 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L24 attn |
| CKA(arrangement before, joint write) | 0.41 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.11 |
| share of the write inside the old arrangement's span | 0.15 |
| write energy / code energy before | 0.04 |

<details><summary>codes and components</summary>

**code 4s-680.0 (L24 L25): 5 comps, tells apart 12/100, coverage 0.21, overlap 1.05 (random 1.26, p 0.01)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 down c108 | MLP | a%10 {0} x b%10 {4}; a%10 {1} x b%10 {5}; a%10 {4} x b%10 {8}; a%10 {6} x b%10 {0} (+) | 0.53 | 0.051 |
| L25 down c145 | MLP | a%10 {0} x b%10 {0} (-) | 0.70 | 0.015 |
| L25 down c715 | MLP | a%10 {0} x b%10 {0} (-) | 0.75 | 0.014 |
| L25 down c24 | MLP | a%10 {0} x b%10 {1}; a%10 {1} x b%10 {2}; a%10 {2} x b%10 {3}; a%10 {6} x b%10 {7}; a%10 {7} x b%10 {8}; a%10 {9} x b%10 {0} (+) | 0.58 | 0.088 |
| L25 down c12 | MLP | a%10 {0} x b%10 {9}; a%10 {1} x b%10 {0}; a%10 {2} x b%10 {1}; a%10 {3} x b%10 {2}; a%10 {4} x b%10 {3}; a%10 {5} x b%10 {4}; a%10 {6} x b%10 {5}; a%10 {7} x b%10 {6}; a%10 {8} x b%10 {7}; a%10 {9} x b%10 {8} (+) | 0.62 | 0.278 |

</details>

</details>

<details><summary><b>4s-681</b> `units(a,b)` @ `=` (sub) — block code, 2 codes of the same shape; 3 comps, L27 L31; tells apart 7/100 classes (best member 6); on add: 4a-571 (member overlap 0.14)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.04 |
| classes told apart: joint / best code / best member | 7 / 7 / 6 (of 100) |
| members whose removal merges classes | 0.33 |
| support overlap (1 = tiling) / random sets / p | 3.00 / 1.12 / 1.00 |
| mean CKA between its codes | 0.91 |
| purity of the joint write (per prompt) | 0.74 |
| decoding acc. joint / best code / best member (chance) | 0.03 / 0.03 / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.13 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L27 attn |
| CKA(arrangement before, joint write) | 0.19 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.06 |
| share of the write inside the old arrangement's span | 0.12 |
| write energy / code energy before | 0.00 |

<details><summary>codes and components</summary>

**code 4s-681.0 (L27): 2 comps, tells apart 7/100, coverage 0.04, overlap 2.00 (random 1.04, p 1.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 down c188 | MLP | a%10 {0, 5} x b%10 {0, 5} (+) | 0.88 | 0.040 |
| L27 down c224 | MLP | a%10 {0, 5} x b%10 {0, 5} (-) | 0.74 | 0.083 |

**code 4s-681.1 (L31): 1 comps, tells apart 5/100, coverage 0.04, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L31 down c187 | MLP | a%10 {0, 5} x b%10 {0, 5} (-) | 0.67 | 0.051 |

</details>

</details>

</details>

<details><summary>`b%100`: 6 mechanisms, 20 components</summary>

<details><summary><b>4s-596</b> `b%100` @ `=` (sub) — copy from `a`; 1 comps, L0; tells apart 3/100 classes (best member 3); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.09 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.90 |
| decoding acc. joint / best code / best member (chance) | 0.06 /  / 0.06 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | embed |
| CKA(arrangement before, joint write) | 0.00 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.00 |
| write energy / code energy before | 6382746208517253054332928.00 |
| arrangement before: shape (spectrum k:share) | line () |
| joint write: shape (spectrum k:share) | line (1:0.40 2:0.13 5:0.07) |
| frequencies new in the write | 1 2 |
| write: shift-symmetric part / dimension (PR) | 0.09 / 1.00 |
| best source position (CKA) | `a` (0.57) |

<details><summary>codes and components</summary>

**code 4s-596.0 (L0): 1 comps, tells apart 3/100, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 o c6 | H23 | b%100 in {0, 2..9} (-) | 0.90 | 1.000 |

</details>

</details>

<details><summary><b>4s-597</b> `b%100` @ `=` (sub) — block code; 3 comps, L0; tells apart 11/100 classes (best member 6); on add: 4a-461 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.34 |
| classes told apart: joint / best code / best member | 11 /  / 6 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.12 / 1.09 / 0.59 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.84 |
| decoding acc. joint / best code / best member (chance) | 0.21 /  / 0.08 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.11 |
| consumers / read jointly | 7 / 5 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 attn |
| CKA(arrangement before, joint write) | 0.34 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.02 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.07 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.26 2:0.11 3:0.07) |
| joint write: shape (spectrum k:share) | line (2:0.17 3:0.08 1:0.06) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.04 / 1.26 |

<details><summary>codes and components</summary>

**code 4s-597.0 (L0): 3 comps, tells apart 11/100, coverage 0.34, overlap 1.12 (random 1.11, p 0.53)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 down c46 | MLP | b%100 in {0, 40, 45, 50, 90, 95} (-) | 0.89 | 1.000 |
| L0 down c37 | MLP | b%100 in {0..1, 4..5, 7..9, 12, 15, 20, 25, 28, 34..35, 44, 64, 66..67, 70, 74..75, 77, 80, 92..94, 98..99} (+) | 0.69 | 1.000 |
| L0 down c4 | MLP | b%100 in {0..2, 64} (-) | 0.83 | 1.000 |

</details>

</details>

<details><summary><b>4s-598</b> `b%100` @ `=` (sub) — single component; 1 comps, L1; tells apart 5/100 classes (best member 5); on add: 4a-462 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.06 |
| classes told apart: joint / best code / best member | 5 /  / 5 (of 100) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.87 |
| decoding acc. joint / best code / best member (chance) | 0.07 /  / 0.07 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 7 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L0 MLP |
| CKA(arrangement before, joint write) | 0.73 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.09 |
| write energy / code energy before | 0.06 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.26 2:0.12 3:0.07) |
| joint write: shape (spectrum k:share) | line (1:0.33 2:0.16 4:0.11) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.08 / 1.00 |
| best source position (CKA) | `b` (0.43) |

<details><summary>codes and components</summary>

**code 4s-598.0 (L1): 1 comps, tells apart 5/100, coverage 0.06, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 o c10 | H11 | b%100 in {1..5, 9} (+) | 0.87 | 1.000 |

</details>

</details>

<details><summary><b>4s-599</b> `b%100` @ `=` (sub) — block code; 2 comps, L1; tells apart 8/100 classes (best member 5); on add: 4a-464 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.30 |
| classes told apart: joint / best code / best member | 8 /  / 5 (of 100) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.20 / 1.03 / 0.91 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.73 |
| decoding acc. joint / best code / best member (chance) | 0.09 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.15 |
| consumers / read jointly | 6 / 6 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L1 attn |
| CKA(arrangement before, joint write) | 0.53 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.01 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.02 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.23 2:0.12 20:0.12) |
| joint write: shape (spectrum k:share) | line (1:0.35 2:0.20 4:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.11 / 1.34 |

<details><summary>codes and components</summary>

**code 4s-599.0 (L1): 2 comps, tells apart 8/100, coverage 0.30, overlap 1.20 (random 1.00, p 0.93)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L1 down c20 | MLP | b%100 in {1..5, 9..19, 21..22, 24, 26..29, 50, 70..72} (+) | 0.78 | 1.000 |
| L1 down c9 | MLP | b%100 in {1..9} (-) | 0.66 | 1.000 |

</details>

</details>

<details><summary><b>4s-600</b> `b%100` @ `=` (sub) — block code, copy from `a`; 8 comps, L15; tells apart 27/100 classes (best member 6); on add: 4a-466 (member overlap 0.28)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.81 |
| classes told apart: joint / best code / best member | 27 /  / 6 (of 100) |
| members whose removal merges classes | 0.88 |
| support overlap (1 = tiling) / random sets / p | 1.31 / 1.52 / 0.07 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.74 |
| decoding acc. joint / best code / best member (chance) | 0.31 /  / 0.04 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 18 / 17 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L14 MLP |
| CKA(arrangement before, joint write) | 0.39 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.08 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.14 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.37 2:0.12 3:0.06) |
| joint write: shape (spectrum k:share) | irregular (1:0.31 2:0.28 5:0.21) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.37 / 3.35 |
| best source position (CKA) | `a` (0.57) |

<details><summary>codes and components</summary>

**code 4s-600.0 (L15): 8 comps, tells apart 27/100, coverage 0.81, overlap 1.31 (random 1.49, p 0.09)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 o c82 | H13 | b%100 in {0, 77, 79..99} (+) | 0.84 | 0.303 |
| L15 o c153 | H13 | b%100 in {0, 96..99} (-) | 0.75 | 0.083 |
| L15 o c107 | H13 | b%100 in {14, 32..34, 72..74} (-) | 0.51 | 0.201 |
| L15 o c136 | H13 | b%100 in {16, 36, 56, 96} (-) | 0.59 | 0.032 |
| L15 o c59 | H13 | b%100 in {25..33} (+) | 0.81 | 0.107 |
| L15 o c122 | H13 | b%100 in {35..44, 49, 54..59} (-) | 0.67 | 0.446 |
| L15 o c81 | H13 | b%100 in {6..7, 10, 15..24, 45..54, 65..66, 76..84} (+) | 0.72 | 0.589 |
| L15 o c120 | H13 | b%100 in {9, 29, 39, 49, 69, 89, 99} (-) | 0.59 | 0.171 |

</details>

</details>

<details><summary><b>4s-601</b> `b%100` @ `=` (sub) — block code, 2 codes of the same shape, copy from `b`; 5 comps, L16 L18; tells apart 12/100 classes (best member 6); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.32 |
| classes told apart: joint / best code / best member | 12 / 10 / 6 (of 100) |
| members whose removal merges classes | 0.80 |
| support overlap (1 = tiling) / random sets / p | 1.38 / 1.28 / 0.72 |
| mean CKA between its codes | 0.84 |
| purity of the joint write (per prompt) | 0.73 |
| decoding acc. joint / best code / best member (chance) | 0.10 / 0.08 / 0.03 (0.01) |
| kappa (1 orthogonal, >1 constructive) | 1.06 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L16 attn |
| CKA(arrangement before, joint write) | 0.60 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.04 |
| arrangement before: shape (spectrum k:share) | irregular (1:0.30 2:0.13 20:0.09) |
| joint write: shape (spectrum k:share) | line (1:0.32 2:0.23 3:0.15) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.14 / 1.45 |
| best source position (CKA) | `b` (0.64) |

<details><summary>codes and components</summary>

**code 4s-601.0 (L16): 1 comps, tells apart 5/100, coverage 0.08, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c501 | MLP | b%100 in {3..10} (-) | 0.78 | 0.136 |

**code 4s-601.1 (L18): 4 comps, tells apart 10/100, coverage 0.32, overlap 1.12 (random 1.18, p 0.31)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 down c434 | MLP | b%100 in {1..12} (+) | 0.78 | 0.219 |
| L18 o c402 | H30 | b%100 in {45..50} (+) | 0.50 | 0.155 |
| L18 o c158 | H30 | b%100 in {67, 69..74, 77} (+) | 0.55 | 0.120 |
| L18 o c96 | H30 | b%100 in {7, 17, 27, 37, 47, 57, 67, 77, 87, 97} (+) | 0.53 | 0.075 |

</details>

</details>

</details>

<details><summary>`b%10`: 4 mechanisms, 14 components</summary>

<details><summary><b>4s-593</b> `b%10` @ `=` (sub) — block code; 4 comps, L15; tells apart 5/10 classes (best member 4); on add: 4a-459 (member overlap 0.60)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.90 |
| classes told apart: joint / best code / best member | 5 /  / 4 (of 10) |
| members whose removal merges classes | 0.25 |
| support overlap (1 = tiling) / random sets / p | 1.78 / 1.41 / 0.88 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.64 |
| decoding acc. joint / best code / best member (chance) | 0.46 /  / 0.24 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.12 |
| consumers / read jointly | 17 / 15 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 attn |
| CKA(arrangement before, joint write) | 0.47 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.12 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.26 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (20:0.31 10:0.31 50:0.19) |
| joint write: shape (spectrum k:share) | line (10:0.82 30:0.07 20:0.07) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.46 / 1.34 |

<details><summary>codes and components</summary>

**code 4s-593.0 (L15): 4 comps, tells apart 5/10, coverage 0.90, overlap 1.78 (random 1.40, p 0.92)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c172 | MLP | b%10 in {0} (-) | 0.48 | 0.056 |
| L15 down c21 | MLP | b%10 in {1..4} (+) | 0.65 | 0.425 |
| L15 down c72 | MLP | b%10 in {1..8} (+) | 0.62 | 0.476 |
| L15 down c129 | MLP | b%10 in {4..6} (+) | 0.54 | 0.214 |

</details>

</details>

<details><summary><b>4s-592</b> `b%10` @ `=` (sub) — block code, copy from `b`; 8 comps, L15; tells apart 10/10 classes (best member 5); on add: 4a-458 (member overlap 0.80)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 10 /  / 5 (of 10) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 1.80 / 2.21 / 0.15 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.69 |
| decoding acc. joint / best code / best member (chance) | 0.86 /  / 0.27 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.12 |
| consumers / read jointly | 35 / 34 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L14 MLP |
| CKA(arrangement before, joint write) | 0.72 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.12 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 1.12 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (20:0.31 10:0.25 50:0.20) |
| joint write: shape (spectrum k:share) | symmetric mix of circles (10:0.46 20:0.37 50:0.09) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.84 / 4.60 |
| best source position (CKA) | `b` (0.84) |

<details><summary>codes and components</summary>

**code 4s-592.0 (L15): 8 comps, tells apart 10/10, coverage 1.00, overlap 1.80 (random 2.21, p 0.09)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 o c132 | H13 | b%10 in {0, 8..9} (+) | 0.62 | 0.241 |
| L15 o c33 | H13 | b%10 in {0..1, 9} (-) | 0.61 | 0.234 |
| L15 o c178 | H13 | b%10 in {1..3} (+) | 0.75 | 0.457 |
| L15 o c111 | H13 | b%10 in {2} (-) | 0.53 | 0.383 |
| L15 o c168 | H13 | b%10 in {3..4} (+) | 0.60 | 0.217 |
| L15 o c73 | H13 | b%10 in {5..7} (-) | 0.77 | 0.284 |
| L15 o c93 | H13 | b%10 in {5} (+) | 0.75 | 0.096 |
| L15 o c53 | H13 | b%10 in {8..9} (+) | 0.71 | 0.328 |

</details>

</details>

<details><summary><b>4s-594</b> `b%10` @ `=` (sub) — single component; 1 comps, L16; tells apart 3/10 classes (best member 3); on add: 4a-460 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.20 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 10) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.54 |
| decoding acc. joint / best code / best member (chance) | 0.18 /  / 0.18 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L16 attn |
| CKA(arrangement before, joint write) | 0.38 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.02 |
| share of the write inside the old arrangement's span | 0.02 |
| write energy / code energy before | 0.03 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (20:0.31 10:0.30 50:0.25) |
| joint write: shape (spectrum k:share) | line (10:0.54 20:0.33 30:0.11) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.20 / 1.00 |

<details><summary>codes and components</summary>

**code 4s-594.0 (L16): 1 comps, tells apart 3/10, coverage 0.20, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c295 | MLP | b%10 in {3..4} (+) | 0.54 | 0.195 |

</details>

</details>

<details><summary><b>4s-595</b> `b%10` @ `=` (sub) — single component; 1 comps, L18; tells apart 2/10 classes (best member 2); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.10 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 10) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.50 |
| decoding acc. joint / best code / best member (chance) | 0.17 /  / 0.17 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L17 MLP |
| CKA(arrangement before, joint write) | 0.29 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | simplex (one-hot like) (10:0.42 20:0.27 50:0.18) |
| joint write: shape (spectrum k:share) | line (20:0.23 10:0.23 40:0.21) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.11 / 1.00 |
| best source position (CKA) | `b` (0.25) |

<details><summary>codes and components</summary>

**code 4s-595.0 (L18): 1 comps, tells apart 2/10, coverage 0.10, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 o c445 | H30 | b%10 in {3} (+) | 0.50 | 0.104 |

</details>

</details>

</details>

<details><summary>`res//10`: 5 mechanisms, 14 components</summary>

<details><summary><b>4s-633</b> `res//10` @ `=` (sub) — block code, 4 codes of the same shape, copy from `b`; 5 comps, L4 L5 L8 L25; tells apart 2/20 classes (best member 3); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 2 / 5 / 3 (of 20) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 2.25 / 1.53 / 0.99 |
| mean CKA between its codes | 0.79 |
| purity of the joint write (per prompt) | 0.75 |
| decoding acc. joint / best code / best member (chance) | 0.29 / 0.27 / 0.18 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 21 / 17 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L4 attn |
| CKA(arrangement before, joint write) | 0.83 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.07 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 4.57 |
| best source position (CKA) | `b` (0.89) |

<details><summary>codes and components</summary>

**code 4s-633.0 (L4): 1 comps, tells apart 2/20, coverage 0.75, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L4 down c0 | MLP | res//10 in {-10..-4, 0..7} (-) | 0.66 | 1.000 |

**code 4s-633.1 (L5): 2 comps, tells apart 5/20, coverage 0.80, overlap 1.00 (random 1.00, p 0.53)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L5 o c75 | H31 | res//10 in {-10..-2} (-) | 0.91 | 0.488 |
| L5 o c310 | H31 | res//10 in {3..9} (+) | 0.87 | 0.332 |

**code 4s-633.2 (L8): 1 comps, tells apart 2/20, coverage 0.35, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L8 down c33 | MLP | res//10 in {3..9} (-) | 0.86 | 0.456 |

**code 4s-633.3 (L25): 1 comps, tells apart 2/20, coverage 0.35, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L25 down c106 | MLP | res//10 in {-10..-8, -4..-1} (-) | 0.65 | 0.449 |

</details>

</details>

<details><summary><b>4s-634</b> `res//10` @ `=` (sub) — 4 codes of the same shape; 4 comps, L13 L17 L21 L29; tells apart 6/20 classes (best member 5); on add: 4a-528 (member overlap 0.14)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.15 |
| classes told apart: joint / best code / best member | 6 / 5 / 5 (of 20) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 2.67 / 1.38 / 1.00 |
| mean CKA between its codes | 0.85 |
| purity of the joint write (per prompt) | 0.59 |
| decoding acc. joint / best code / best member (chance) | 0.16 / 0.13 / 0.13 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.03 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L13 attn |
| CKA(arrangement before, joint write) | 0.50 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.05 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 0.43 |

<details><summary>codes and components</summary>

**code 4s-634.0 (L13): 1 comps, tells apart 5/20, coverage 0.15, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 down c22 | MLP | res//10 in {0..2} (+) | 0.56 | 0.192 |

**code 4s-634.1 (L17): 1 comps, tells apart 4/20, coverage 0.10, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L17 down c89 | MLP | res//10 in {0..1} (-) | 0.53 | 0.191 |

**code 4s-634.2 (L21): 1 comps, tells apart 5/20, coverage 0.10, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L21 down c105 | MLP | res//10 in {0..1} (-) | 0.73 | 0.334 |

**code 4s-634.3 (L29): 1 comps, tells apart 4/20, coverage 0.05, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L29 down c115 | MLP | res//10 in {0} (+) | 0.54 | 0.248 |

</details>

</details>

<details><summary><b>4s-635</b> `res//10` @ `=` (sub) — single component; 1 comps, L15; tells apart 4/20 classes (best member 4); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.20 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 20) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.53 |
| decoding acc. joint / best code / best member (chance) | 0.10 /  / 0.10 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 attn |
| CKA(arrangement before, joint write) | 0.58 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.14 |
| share of the write inside the old arrangement's span | 0.10 |
| write energy / code energy before | 0.03 |

<details><summary>codes and components</summary>

**code 4s-635.0 (L15): 1 comps, tells apart 4/20, coverage 0.20, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c43 | MLP | res//10 in {-10..-7} (-) | 0.53 | 0.117 |

</details>

</details>

<details><summary><b>4s-636</b> `res//10` @ `=` (sub) — block code; 3 comps, L19 L20; tells apart 6/20 classes (best member 5); on add: 4a-527 (member overlap 0.06)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.30 |
| classes told apart: joint / best code / best member | 6 /  / 5 (of 20) |
| members whose removal merges classes | 0.67 |
| support overlap (1 = tiling) / random sets / p | 1.17 / 1.20 / 0.43 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.71 |
| decoding acc. joint / best code / best member (chance) | 0.22 /  / 0.13 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.03 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L19 attn |
| CKA(arrangement before, joint write) | 0.61 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.05 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.02 |

<details><summary>codes and components</summary>

**code 4s-636.0 (L19 L20): 3 comps, tells apart 6/20, coverage 0.30, overlap 1.17 (random 1.21, p 0.41)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L19 down c33 | MLP | res//10 in {8..9} (+) | 0.52 | 0.036 |
| L20 down c20 | MLP | res//10 in {0..3} (-) | 0.72 | 0.484 |
| L20 down c223 | MLP | res//10 in {1} (+) | 0.64 | 0.070 |

</details>

</details>

<details><summary><b>4s-637</b> `res//10` @ `=` (sub) — single component; 1 comps, L23; tells apart 3/20 classes (best member 3); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.05 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 20) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.69 |
| decoding acc. joint / best code / best member (chance) | 0.12 /  / 0.12 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L23 attn |
| CKA(arrangement before, joint write) | 0.49 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.06 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.01 |

<details><summary>codes and components</summary>

**code 4s-637.0 (L23): 1 comps, tells apart 3/20, coverage 0.05, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L23 down c465 | MLP | res//10 in {0} (-) | 0.69 | 0.290 |

</details>

</details>

</details>

<details><summary>`cmp(a,b)`: 1 mechanisms, 8 components</summary>

<details><summary><b>4s-609</b> `cmp(a,b)` @ `=` (sub) — block code, 7 codes of the same shape, copy from `b`; 8 comps, L13 L15 L24 L26 L27 L28 L30; tells apart 2/3 classes (best member 2); on add: 4a-479 (member overlap 0.11)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.33 |
| classes told apart: joint / best code / best member | 2 / 2 / 2 (of 3) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 8.00 /  /  |
| mean CKA between its codes | 1.00 |
| purity of the joint write (per prompt) | 0.69 |
| decoding acc. joint / best code / best member (chance) | 0.64 / 0.64 / 0.64 (0.33) |
| kappa (1 orthogonal, >1 constructive) | 1.55 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L12 MLP |
| CKA(arrangement before, joint write) | 0.65 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.00 |
| write energy / code energy before | 0.04 |
| best source position (CKA) | `b` (0.52) |

<details><summary>codes and components</summary>

**code 4s-609.0 (L13): 1 comps, tells apart 2/3, coverage 0.33, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L13 o c379 | H7 | cmp(a,b) in {0} (-) | 0.90 | 0.010 |

**code 4s-609.1 (L15): 1 comps, tells apart 2/3, coverage 0.33, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c46 | MLP | cmp(a,b) in {0} (-) | 0.81 | 0.011 |

**code 4s-609.2 (L24): 1 comps, tells apart 2/3, coverage 0.33, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L24 down c209 | MLP | cmp(a,b) in {0} (+) | 0.46 | 0.043 |

**code 4s-609.3 (L26): 2 comps, tells apart 2/3, coverage 0.33, overlap 2.00 (random 2.00, p 1.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L26 down c271 | MLP | cmp(a,b) in {0} (+) | 0.68 | 0.013 |
| L26 down c161 | MLP | cmp(a,b) in {0} (-) | 0.66 | 0.011 |

**code 4s-609.4 (L27): 1 comps, tells apart 2/3, coverage 0.33, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L27 down c823 | MLP | cmp(a,b) in {0} (-) | 0.60 | 0.017 |

**code 4s-609.5 (L28): 1 comps, tells apart 2/3, coverage 0.33, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L28 down c584 | MLP | cmp(a,b) in {0} (-) | 0.70 | 0.026 |

**code 4s-609.6 (L30): 1 comps, tells apart 2/3, coverage 0.33, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 down c1003 | MLP | cmp(a,b) in {0} (-) | 0.76 | 0.011 |

</details>

</details>

</details>

<details><summary>`a//10`: 3 mechanisms, 6 components</summary>

<details><summary><b>4s-590</b> `a//10` @ `=` (sub) — copy from `a`; 1 comps, L16; tells apart 4/11 classes (best member 4); on add: 4a-457 (member overlap 0.12)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.82 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 11) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.89 |
| decoding acc. joint / best code / best member (chance) | 0.27 /  / 0.27 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 MLP |
| CKA(arrangement before, joint write) | 0.73 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.03 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 0.06 |
| best source position (CKA) | `a` (0.79) |

<details><summary>codes and components</summary>

**code 4s-590.0 (L16): 1 comps, tells apart 4/11, coverage 0.82, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c207 | H30 | a//10 in {0..3, 6..10} (+) | 0.89 | 0.584 |

</details>

</details>

<details><summary><b>4s-589</b> `a//10` @ `=` (sub) — block code, copy from `b`; 4 comps, L16; tells apart 6/11 classes (best member 4); on add: 4a-457 (member overlap 0.20)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.36 |
| classes told apart: joint / best code / best member | 6 /  / 4 (of 11) |
| members whose removal merges classes | 0.50 |
| support overlap (1 = tiling) / random sets / p | 2.00 / 1.67 / 1.00 |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.85 |
| decoding acc. joint / best code / best member (chance) | 0.51 /  / 0.27 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.14 |
| consumers / read jointly | 28 / 26 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 MLP |
| CKA(arrangement before, joint write) | 0.41 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.09 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.17 |
| best source position (CKA) | `b` (0.71) |

<details><summary>codes and components</summary>

**code 4s-589.0 (L16): 4 comps, tells apart 6/11, coverage 0.36, overlap 2.00 (random 1.67, p 1.00)**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c132 | H21 | a//10 in {10} (+) | 0.80 | 0.013 |
| L16 o c52 | H21 | a//10 in {7..8} (+) | 0.78 | 0.296 |
| L16 o c16 | H21 | a//10 in {8..10} (+) | 0.88 | 0.302 |
| L16 o c98 | H21 | a//10 in {9..10} (-) | 0.85 | 0.339 |

</details>

</details>

<details><summary><b>4s-591</b> `a//10` @ `=` (sub) — single component; 1 comps, L30; tells apart 3/11 classes (best member 3); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.18 |
| classes told apart: joint / best code / best member | 3 /  / 3 (of 11) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.79 |
| decoding acc. joint / best code / best member (chance) | 0.23 /  / 0.23 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 6 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L30 attn |
| CKA(arrangement before, joint write) | 0.66 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.11 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.03 |

<details><summary>codes and components</summary>

**code 4s-591.0 (L30): 1 comps, tells apart 3/11, coverage 0.18, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L30 down c7 | MLP | a//10 in {0..1} (-) | 0.79 | 0.219 |

</details>

</details>

</details>

<details><summary>`a%10`: 1 mechanisms, 4 components</summary>

<details><summary><b>4s-573</b> `a%10` @ `=` (sub) — block code, copy from `a`; 4 comps, L16; tells apart 9/10 classes (best member 4); on add: 4a-437 (member overlap 0.25)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.90 |
| classes told apart: joint / best code / best member | 9 /  / 4 (of 10) |
| members whose removal merges classes | 0.75 |
| support overlap (1 = tiling) / random sets / p | 1.22 /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.79 |
| decoding acc. joint / best code / best member (chance) | 0.74 /  / 0.28 (0.10) |
| kappa (1 orthogonal, >1 constructive) | 1.05 |
| consumers / read jointly | 33 / 31 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 MLP |
| CKA(arrangement before, joint write) | 0.48 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.12 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.94 |
| arrangement before: shape (spectrum k:share) | irregular (20:0.31 10:0.27 40:0.20) |
| joint write: shape (spectrum k:share) | irregular (10:0.55 20:0.21 50:0.17) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.57 / 2.81 |
| best source position (CKA) | `a` (0.60) |

<details><summary>codes and components</summary>

**code 4s-573.0 (L16): 4 comps, tells apart 9/10, coverage 0.90, overlap 1.22 (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c53 | H21 | a%10 in {1, 3, 9} (+) | 0.79 | 0.384 |
| L16 o c121 | H21 | a%10 in {2..4} (-) | 0.79 | 0.352 |
| L16 o c97 | H21 | a%10 in {5..6} (+) | 0.77 | 0.288 |
| L16 o c180 | H21 | a%10 in {7..9} (-) | 0.77 | 0.291 |

</details>

</details>

</details>

<details><summary>`b//10`: 2 mechanisms, 4 components</summary>

<details><summary><b>4s-607</b> `b//10` @ `=` (sub) — single component; 1 comps, L0; tells apart 4/11 classes (best member 4); on add: 4a-477 (member overlap 0.50)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.18 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 11) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.96 |
| decoding acc. joint / best code / best member (chance) | 0.58 /  / 0.58 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 3 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | embed |
| CKA(arrangement before, joint write) | 0.00 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.00 |
| write energy / code energy before | 44529886363307014224871424.00 |
| best source position (CKA) | `b` (0.44) |

<details><summary>codes and components</summary>

**code 4s-607.0 (L0): 1 comps, tells apart 4/11, coverage 0.18, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 o c7 | H0 | b//10 in {0, 10} (+) | 0.96 | 1.000 |

</details>

</details>

<details><summary><b>4s-608</b> `b//10` @ `=` (sub) — 3 codes of the same shape; 3 comps, L0 L15; tells apart 3/11 classes (best member 3); on add: 4a-477 (member overlap 0.25)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.09 |
| classes told apart: joint / best code / best member | 3 / 3 / 3 (of 11) |
| members whose removal merges classes | 0.33 |
| support overlap (1 = tiling) / random sets / p | 3.00 / 2.00 / 1.00 |
| mean CKA between its codes | 0.93 |
| purity of the joint write (per prompt) | 0.81 |
| decoding acc. joint / best code / best member (chance) | 0.26 / 0.32 / 0.32 (0.09) |
| kappa (1 orthogonal, >1 constructive) | 1.12 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | embed |
| CKA(arrangement before, joint write) | 0.00 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.00 |
| share of the write inside the old arrangement's span | 0.00 |
| write energy / code energy before | 45272571882233018957615333376.00 |
| best source position (CKA) | `b` (0.33) |

<details><summary>codes and components</summary>

**code 4s-608.0 (L0): 1 comps, tells apart 3/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L0 o c2 | H2 | b//10 in {0} (-) | 0.87 | 1.000 |

**code 4s-608.1 (L15): 1 comps, tells apart 3/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c141 | MLP | b//10 in {0} (+) | 0.78 | 0.119 |

**code 4s-608.2 (L15): 1 comps, tells apart 2/11, coverage 0.09, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 o c187 | H13 | b//10 in {0} (-) | 0.82 | 0.097 |

</details>

</details>

</details>

<details><summary>`a%20`: 1 mechanisms, 3 components</summary>

<details><summary><b>4s-587</b> `a%20` @ `=` (sub) — block code; 3 comps, L16; tells apart 7/20 classes (best member 4); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.30 |
| classes told apart: joint / best code / best member | 7 /  / 4 (of 20) |
| members whose removal merges classes | 1.00 |
| support overlap (1 = tiling) / random sets / p | 1.00 /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.76 |
| decoding acc. joint / best code / best member (chance) | 0.22 /  / 0.13 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.01 |
| consumers / read jointly | 2 / 1 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 MLP |
| CKA(arrangement before, joint write) | 0.39 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.07 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.13 |
| arrangement before: shape (spectrum k:share) | irregular (5:0.26 20:0.19 10:0.16) |
| joint write: shape (spectrum k:share) | irregular (20:0.28 10:0.24 40:0.19) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.32 / 2.79 |
| best source position (CKA) | `a` (0.50) |

<details><summary>codes and components</summary>

**code 4s-587.0 (L16): 3 comps, tells apart 7/20, coverage 0.30, overlap 1.00 (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c257 | H21 | a%20 in {1, 11} (+) | 0.78 | 0.099 |
| L16 o c26 | H21 | a%20 in {5, 15} (-) | 0.76 | 0.158 |
| L16 o c251 | H21 | a%20 in {8, 18} (-) | 0.74 | 0.139 |

</details>

</details>

</details>

<details><summary>`a%50`: 1 mechanisms, 3 components</summary>

<details><summary><b>4s-588</b> `a%50` @ `=` (sub) — block code; 3 comps, L16; tells apart 10/50 classes (best member 4); on add: 4a-454 (member overlap 0.25)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.56 |
| classes told apart: joint / best code / best member | 10 /  / 4 (of 50) |
| members whose removal merges classes | 0.67 |
| support overlap (1 = tiling) / random sets / p | 1.18 /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.74 |
| decoding acc. joint / best code / best member (chance) | 0.15 /  / 0.06 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 15 / 11 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 MLP |
| CKA(arrangement before, joint write) | 0.14 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.06 |
| share of the write inside the old arrangement's span | 0.07 |
| write energy / code energy before | 0.09 |
| arrangement before: shape (spectrum k:share) | irregular (2:0.40 4:0.12 20:0.07) |
| joint write: shape (spectrum k:share) | irregular (20:0.42 10:0.27 40:0.13) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.35 / 2.57 |
| best source position (CKA) | `a` (0.38) |

<details><summary>codes and components</summary>

**code 4s-588.0 (L16): 3 comps, tells apart 10/50, coverage 0.56, overlap 1.18 (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 o c37 | H21 | a%50 in {0, 10, 14..15, 20, 25, 30, 35, 40, 45} (+) | 0.72 | 0.268 |
| L16 o c117 | H21 | a%50 in {13..14, 23..24, 33..34, 43..44} (-) | 0.76 | 0.256 |
| L16 o c45 | H21 | a%50 in {7, 12..13, 17, 21..23, 27, 32..33, 37, 41..43, 47} (-) | 0.75 | 0.346 |

</details>

</details>

</details>

<details><summary>`b%2`: 1 mechanisms, 2 components</summary>

<details><summary><b>4s-602</b> `b%2` @ `=` (sub) — 2 codes of the same shape, copy from `b`; 2 comps, L15; tells apart 2/2 classes (best member 2); on add: 4a-471 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 2 / 2 / 2 (of 2) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 2.00 /  /  |
| mean CKA between its codes | 1.00 |
| purity of the joint write (per prompt) | 0.72 |
| decoding acc. joint / best code / best member (chance) | 0.95 / 0.98 / 0.98 (0.50) |
| kappa (1 orthogonal, >1 constructive) | 1.16 |
| consumers / read jointly | 8 / 6 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L14 MLP |
| CKA(arrangement before, joint write) | 1.00 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.08 |
| share of the write inside the old arrangement's span | 0.01 |
| write energy / code energy before | 4.02 |
| arrangement before: shape (spectrum k:share) | line (50:1.00) |
| joint write: shape (spectrum k:share) | line (50:1.00) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 1.00 / 1.00 |
| best source position (CKA) | `b` (1.00) |

<details><summary>codes and components</summary>

**code 4s-602.0 (L15): 1 comps, tells apart 2/2, coverage 1.00, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 o c60 | H13 | b%2 in {0..1} (+) | 0.80 | 0.513 |

**code 4s-602.1 (L15): 1 comps, tells apart 2/2, coverage 1.00, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c40 | MLP | b%2 in {0..1} (+) | 0.69 | 0.482 |

</details>

</details>

</details>

<details><summary>`b%20`: 1 mechanisms, 2 components</summary>

<details><summary><b>4s-603</b> `b%20` @ `=` (sub) — 2 codes of the same shape; 2 comps, L15 L16; tells apart 2/20 classes (best member 2); on add: 4a-473 (member overlap 1.00)</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 2 / 2 / 2 (of 20) |
| members whose removal merges classes | 0.00 |
| support overlap (1 = tiling) / random sets / p | 2.00 /  /  |
| mean CKA between its codes | 1.00 |
| purity of the joint write (per prompt) | 0.58 |
| decoding acc. joint / best code / best member (chance) | 0.10 / 0.11 / 0.11 (0.05) |
| kappa (1 orthogonal, >1 constructive) | 1.16 |
| consumers / read jointly | 13 / 10 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 attn |
| CKA(arrangement before, joint write) | 0.28 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.00 |
| share of the write inside the old arrangement's span | 0.00 |
| write energy / code energy before | 0.27 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (20:0.25 10:0.24 5:0.16) |
| joint write: shape (spectrum k:share) | line (5:0.86 15:0.08) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.37 / 1.00 |

<details><summary>codes and components</summary>

**code 4s-603.0 (L15): 1 comps, tells apart 2/20, coverage 1.00, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c20 | MLP | b%20 in {0..19} (+) | 0.62 | 0.452 |

**code 4s-603.1 (L16): 1 comps, tells apart 2/20, coverage 1.00, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L16 down c24 | MLP | b%20 in {0..19} (+) | 0.53 | 0.424 |

</details>

</details>

</details>

<details><summary>`b%50`: 2 mechanisms, 2 components</summary>

<details><summary><b>4s-605</b> `b%50` @ `=` (sub) — single component; 1 comps, L15; tells apart 5/50 classes (best member 5); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.10 |
| classes told apart: joint / best code / best member | 5 /  / 5 (of 50) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.56 |
| decoding acc. joint / best code / best member (chance) | 0.04 /  / 0.04 (0.02) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 0 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L14 MLP |
| CKA(arrangement before, joint write) | 0.17 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.01 |
| share of the write inside the old arrangement's span | 0.04 |
| write energy / code energy before | 0.01 |
| arrangement before: shape (spectrum k:share) | irregular (2:0.29 4:0.12 20:0.10) |
| joint write: shape (spectrum k:share) | line (10:0.30 20:0.23 40:0.19) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.11 / 1.00 |
| best source position (CKA) | `b` (0.27) |

<details><summary>codes and components</summary>

**code 4s-605.0 (L15): 1 comps, tells apart 5/50, coverage 0.10, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 o c139 | H13 | b%50 in {0, 10, 20, 30, 40} (+) | 0.56 | 0.125 |

</details>

</details>

<details><summary><b>4s-606</b> `b%50` @ `=` (sub) — single component; 1 comps, L18; tells apart 5/50 classes (best member 5); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.10 |
| classes told apart: joint / best code / best member | 5 /  / 5 (of 50) |
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
| input point | L17 MLP |
| CKA(arrangement before, joint write) | 0.19 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.06 |
| share of the write inside the old arrangement's span | 0.08 |
| write energy / code energy before | 0.00 |
| arrangement before: shape (spectrum k:share) | symmetric mix of circles (2:0.30 10:0.21 20:0.14) |
| joint write: shape (spectrum k:share) | line (10:0.24 20:0.22 40:0.21) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.11 / 1.00 |
| best source position (CKA) | `b` (0.20) |

<details><summary>codes and components</summary>

**code 4s-606.0 (L18): 1 comps, tells apart 5/50, coverage 0.10, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L18 o c200 | H30 | b%50 in {5, 15, 25, 35, 45} (+) | 0.56 | 0.118 |

</details>

</details>

</details>

<details><summary>`b%5`: 1 mechanisms, 1 components</summary>

<details><summary><b>4s-604</b> `b%5` @ `=` (sub) — single component; 1 comps, L15; tells apart 4/5 classes (best member 4); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 0.80 |
| classes told apart: joint / best code / best member | 4 /  / 4 (of 5) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.69 |
| decoding acc. joint / best code / best member (chance) | 0.56 /  / 0.56 (0.20) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 8 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L15 attn |
| CKA(arrangement before, joint write) | 0.59 |
| cos with the arrangement before (>0 amplify, <0 erase) | -0.14 |
| share of the write inside the old arrangement's span | 0.06 |
| write energy / code energy before | 0.43 |
| arrangement before: shape (spectrum k:share) | circle period 5 (20:0.69 40:0.31) |
| joint write: shape (spectrum k:share) | line (20:0.98) |
| frequencies new in the write | none |
| write: shift-symmetric part / dimension (PR) | 0.48 / 1.00 |

<details><summary>codes and components</summary>

**code 4s-604.0 (L15): 1 comps, tells apart 4/5, coverage 0.80, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L15 down c64 | MLP | b%5 in {1..4} (+) | 0.69 | 0.580 |

</details>

</details>

</details>

<details><summary>`res<0`: 1 mechanisms, 1 components</summary>

<details><summary><b>4s-638</b> `res<0` @ `=` (sub) — copy from `b`; 1 comps, L6; tells apart 2/2 classes (best member 2); on add: no counterpart</summary>

Checks

| check | value |
|---|---|
| classes in some member's support | 1.00 |
| classes told apart: joint / best code / best member | 2 /  / 2 (of 2) |
| members whose removal merges classes |  |
| support overlap (1 = tiling) / random sets / p |  /  /  |
| mean CKA between its codes |  |
| purity of the joint write (per prompt) | 0.94 |
| decoding acc. joint / best code / best member (chance) | 1.00 /  / 1.00 (0.50) |
| kappa (1 orthogonal, >1 constructive) | 1.00 |
| consumers / read jointly | 1 / 0 |

How it transforms the arrangement

| arrangement | value |
|---|---|
| input point | L5 MLP |
| CKA(arrangement before, joint write) | 1.00 |
| cos with the arrangement before (>0 amplify, <0 erase) | 0.23 |
| share of the write inside the old arrangement's span | 0.05 |
| write energy / code energy before | 0.35 |
| best source position (CKA) | `b` (1.00) |

<details><summary>codes and components</summary>

**code 4s-638.0 (L6): 1 comps, tells apart 2/2, coverage 1.00, overlap  (random , p )**

| component | block | support (classes it moves; sign of its write) | R^2 | on |
|---|---|---|---|---|
| L6 o c290 | H7 | res<0 in {0..1} (+) | 0.94 | 0.511 |

</details>

</details>

</details>
