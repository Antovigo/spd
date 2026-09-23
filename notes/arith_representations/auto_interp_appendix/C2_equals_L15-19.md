# Appendix C2 — components whose main position is `=`, L15-19

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

<details><summary>layer 15: 102 components with main position `=`</summary>

<details><summary>down: 34</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L15 down c2 | 0 / 19% | off (on 0) | unexplained (best R2 0.45) | - | res: mod25 +2%, mod20 +2% |
| L15 down c4 | 60% / 43% | **b//10** (R2 0.89): (tens) b in {40..100} | **tens(a,b)** (R2 0.77): a//10 in {2} -> b//10 in {5,6,7,8}; a//10 in {3} -> b//10 in {5,6,7,8,9}; a//10 in {4,5,6} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {7,8,9,10} -> b//10 in {5,6,7,8,9,10} | b: mod100 +28% | a: mod100 +2%; b: mod100 +6% |
| L15 down c6 | 100% / 100% | always | same | - | same |
| L15 down c9 | 19% / 0 | **tens(a,b)** (R2 0.62): a//10 in {0} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5} | off (on 0) | a: mod100 +4%, mod50 +4% | - |
| L15 down c10 | 1% / 93% | unexplained (best R2 0.06) | unexplained (best R2 0.34) | - | a: mod100 +3%, mod4 +3%; b: mod100 +3%; res: mod100 +13%, mod50 +5%, mod25 +3%, mod20 +3%, mod5 +2% |
| L15 down c11 | 14% / 0 | **b** (R2 0.52): b in {1..12, 21} | off (on 0) | - | same |
| L15 down c15 | 27% / 0 | **b** (R2 0.60): b in {24..27, 29..32, 34..54, 56, 59} | off (on 0) | res: mod100 +3% | - |
| L15 down c16 | 61% / 0 | **b%5** (R2 0.97): b mod 5 in {0, 3..4} | off (on 0) | a: mod5 +2%; b: mod5 +49% | - |
| L15 down c18 | 22% / 11% | unexplained (best R2 0.47) | unexplained (best R2 0.31) | a: mod100 +3%, mod50 +3%, mod25 +4%, mod20 +2%, mod10 +2%; b: mod25 +2% | res: mod50 +3%, mod25 +5%, mod20 +6%, mod10 +7%, mod5 +3% |
| L15 down c19 | 62% / 25% | **b%50** (R2 0.81): b mod 50 in {0..3, 5, 7, 25..49} | **tens(a,b)** (R2 0.61): a//10 in {3} -> b//10 in {2,3,4}; a//10 in {4} -> b//10 in {3}; a//10 in {5,6} -> b//10 in {3,4,8}; a//10 in {7,10} -> b//10 in {3,4,8,9}; a//10 in {8} -> b//10 in {3,4,7,8,9}; a//10 in {9} -> b//10 in {3,8,9} | b: mod50 +20%; res: mod50 +3% | b: mod50 +2% |
| L15 down c20 | 50% / 45% | **b%20** (R2 0.98): b mod 20 in {0..4, 15..19} | **b%20** (R2 0.76): b mod 20 in {0..4, 15..19} | b: mod20 +41%, mod4 +29% | b: mod20 +37%, mod4 +14% |
| L15 down c21 | 51% / 43% | **b%10** (R2 0.96): b mod 10 in {0..4} | **b%10** (R2 0.69): b mod 10 in {0..4} | b: mod10 +34% | b: mod10 +18% |
| L15 down c28 | 0 / 13% | off (on 0) | **tens(a,b)** (R2 0.63): a//10 in {3,4,5} -> b//10 in {2,3}; a//10 in {6,7,9} -> b//10 in {3}; a//10 in {8} -> b//10 in {3,4}; a//10 in {10} -> b//10 in {2,3,4} | - | b: mod100 +2% |
| L15 down c29 | 41% / 29% | **b** (R2 0.89): b in {1..15, 75..76, 78..100} | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {0,10}; a//10 in {1} -> b//10 in {0,1,10}; a//10 in {2} -> b//10 in {0,1,9,10}; a//10 in {3,10} -> b//10 in {0,1,8,9,10}; a//10 in {4,5,8} -> b//10 in {8,9,10}; a//10 in {6,7} -> b//10 in {7,8,9,10}; a//10 in {9} -> b//10 in {0,8,9,10} | b: mod100 +8% | b: mod100 +4% |
| L15 down c35 | 56% / 17% | **b%50** (R2 0.90): b mod 50 in {0..26, 49} | **tens(a,b)** (R2 0.54): a//10 in {2,3,4,5} -> b//10 in {1}; a//10 in {6,7} -> b//10 in {5,6}; a//10 in {8} -> b//10 in {5,6,7}; a//10 in {9} -> b//10 in {6,7}; a//10 in {10} -> b//10 in {1,5,6,7} | b: mod50 +16%, mod25 +3% | - |
| L15 down c38 | 0 / 0 | off (on 0) | same | - | same |
| L15 down c40 | 50% / 48% | **b%2** (R2 0.99): b mod 2 in {1} | **b%2** (R2 0.85): b mod 2 in {1} | b: mod2 +36% | b: mod2 +41% |
| L15 down c41 | 53% / 33% | **b%50** (R2 0.89): b mod 50 in {12..37} | **tens(a,b)** (R2 0.60): a//10 in {2} -> b//10 in {1,2,7}; a//10 in {3} -> b//10 in {1,2,7,8}; a//10 in {4,10} -> b//10 in {1,2,3,6,7,8}; a//10 in {5} -> b//10 in {2,3,7,8}; a//10 in {6,7} -> b//10 in {3,6,7,8}; a//10 in {8} -> b//10 in {2,3,6,7,8}; a//10 in {9} -> b//10 in {2,6,7,8} | b: mod50 +14%, mod25 +2% | b: mod50 +7% |
| L15 down c43 | 0 / 12% | off (on 0) | **res//10** (R2 0.52): (tens) res in {-97..-61} | - | same |
| L15 down c46 | 4% / 1% | unexplained (best R2 0.33) | **cmp(a,b)** (R2 0.78): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.92, cmp(a,b)=1: 0.00 | - | same |
| L15 down c51 | 8% / 0 | unexplained (best R2 0.42) | off (on 0) | - | same |
| L15 down c54 | 43% / 0 | **b//10** (R2 0.85): (tens) b in {1..42, 45..46} | off (on 0) | b: mod100 +6%, mod50 +2% | - |
| L15 down c64 | 0 / 58% | off (on 0) | **b%5** (R2 0.56): b mod 5 in {0..2} | - | b: mod5 +25% |
| L15 down c67 | 37% / 0 | **b%10** (R2 0.87): b mod 10 in {5..8} | off (on 0) | b: mod10 +5% | - |
| L15 down c72 | 0 / 48% | off (on 0) | **b%10** (R2 0.66): b mod 10 in {5..9} | - | b: mod10 +14% |
| L15 down c76 | 0 / 8% | off (on 0) | **tens(a,b)** (R2 0.60): a//10 in {0} -> b//10 in {7,8,9}; a//10 in {1} -> b//10 in {6,7,8,9,10}; a//10 in {2} -> b//10 in {9} | - | same |
| L15 down c80 | 0 / 25% | off (on 0) | **b** (R2 0.77): b in {9..34} | - | b: mod100 +2%, mod50 +2% |
| L15 down c90 | 0 / 63% | off (on 0) | **tens(a,b)** (R2 0.72): a//10 in {1} -> b//10 in {1,2,3,4}; a//10 in {2} -> b//10 in {1,2,3,4,5,6,7,9}; a//10 in {3} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {4,5,6,7,8,9,10} -> b//10 in {3,4,5,6,7,8,9,10} | - | a: mod100 +5%, mod50 +4%; b: mod100 +6%, mod50 +4%, mod25 +3%; res: mod100 +7% |
| L15 down c129 | 30% / 21% | **b%10** (R2 0.98): b mod 10 in {4..6} | **b%10** (R2 0.62): b mod 10 in {4..6} | b: mod10 +6%, mod5 +3% | b: mod10 +5%, mod5 +2% |
| L15 down c141 | 0 / 12% | off (on 0) | **b//10** (R2 0.70): (tens) b in {1..12, 14} | - | same |
| L15 down c172 | 10% / 6% | **b%10** (R2 0.98): b mod 10 in {0} | **b%10** (R2 0.53): b mod 10 in {0} | - | same |
| L15 down c329 | 1% / 0 | **cmp(a,b)** (R2 0.73): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.98, cmp(a,b)=1: 0.00 | off (on 0) | - | same |
| L15 down c535 | 0 / 13% | off (on 0) | **a** (R2 0.66): a in {1..12, 14} | - | a: mod50 +2%, mod25 +2%, mod20 +2% |
| L15 down c757 | 0 / 1% | off (on 0) | unexplained (best R2 0.40) | - | same |

</details>

<details><summary>gate: 10</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L15 gate c0 | 100% / 100% | always | same | (reads) | same |
| L15 gate c4 | 56% / 42% | **b//10** (R2 0.89): (tens) b in {45..100} | **tens(a,b)** (R2 0.75): a//10 in {2} -> b//10 in {6,7}; a//10 in {3,9,10} -> b//10 in {5,6,7,8,9,10}; a//10 in {4} -> b//10 in {6,7,8,9,10}; a//10 in {5,6,7,8} -> b//10 in {4,5,6,7,8,9,10} | (reads) | same |
| L15 gate c9 | 100% / 0 | always | off (on 0) | (reads) | same |
| L15 gate c19 | 46% / 21% | **b%50** (R2 0.92): b mod 50 in {1..24} | **tens(a,b)** (R2 0.57): a//10 in {1} -> b//10 in {0}; a//10 in {2,3,4,5} -> b//10 in {0,1}; a//10 in {6,7} -> b//10 in {0,1,5,6}; a//10 in {8} -> b//10 in {5,6}; a//10 in {9} -> b//10 in {1,6}; a//10 in {10} -> b//10 in {0,1,5,6,7} | (reads) | same |
| L15 gate c20 | 49% / 45% | **b%20** (R2 0.97): b mod 20 in {0..4, 15..19} | **b%20** (R2 0.76): b mod 20 in {0..4, 15..19} | (reads) | same |
| L15 gate c21 | 51% / 43% | **b%10** (R2 0.96): b mod 10 in {0..4} | **b%10** (R2 0.69): b mod 10 in {0..4} | (reads) | same |
| L15 gate c35 | 60% / 27% | **b** (R2 0.89): b in {24..56, 72, 75..100} [coarser: b mod 50 in {0..2, 24..49}, R2 0.89] | **tens(a,b)** (R2 0.63): a//10 in {2} -> b//10 in {3,4}; a//10 in {3} -> b//10 in {2,3,4,8}; a//10 in {4,5,6} -> b//10 in {3,4,8}; a//10 in {7,8} -> b//10 in {3,4,8,9}; a//10 in {9} -> b//10 in {3,8,9}; a//10 in {10} -> b//10 in {2,3,4,8,9} | (reads) | same |
| L15 gate c54 | 51% / 23% | **b//10** (R2 0.91): (tens) b in {1..50} | **tens(a,b)** (R2 0.62): a//10 in {0} -> b//10 in {0}; a//10 in {1} -> b//10 in {1,2}; a//10 in {2} -> b//10 in {0,1,2}; a//10 in {3,4} -> b//10 in {1,2,3}; a//10 in {5,10} -> b//10 in {1,2,3,4}; a//10 in {6,7,8} -> b//10 in {3,4} | (reads) | same |
| L15 gate c72 | 0 / 99% | off (on 0) | always | (reads) | same |
| L15 gate c101 | 0 / 10% | off (on 0) | **tens(a,b)** (R2 0.70): a//10 in {0} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {5,8,10} | (reads) | same |

</details>

<details><summary>o: 41</summary>

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
| L15 o c270 (H3) | 0 / 3% | off (on 0) | unexplained (best R2 0.34) | - | same |
| L15 o c446 (H18) | 100% / 100% | always | same | - | b: mod100 +3% |

</details>

<details><summary>q: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L15 q c138 (H13) | 100% / 100% | always | same | (reads) | same |

</details>

<details><summary>up: 15</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L15 up c11 | 18% / 10% | unexplained (best R2 0.44) | unexplained (best R2 0.33) | (reads) | same |
| L15 up c16 | 60% / 21% | **b%5** (R2 0.99): b mod 5 in {0, 3..4} | unexplained (best R2 0.30) | (reads) | same |
| L15 up c28 | 0 / 8% | off (on 0) | **tens(a,b)** (R2 0.63): a//10 in {3} -> b//10 in {2,3}; a//10 in {4,5,6,7,8,9} -> b//10 in {3}; a//10 in {10} -> b//10 in {2,3,4} | (reads) | same |
| L15 up c29 | 42% / 28% | **b** (R2 0.90): b in {1..15, 75..100} | **tens(a,b)** (R2 0.70): a//10 in {0} -> b//10 in {0,9,10}; a//10 in {1} -> b//10 in {0,1,10}; a//10 in {2} -> b//10 in {0,1,9,10}; a//10 in {3,4} -> b//10 in {8,9,10}; a//10 in {5,6,7,8} -> b//10 in {7,8,9,10}; a//10 in {9} -> b//10 in {0,8,9,10}; a//10 in {10} -> b//10 in {0,1,8,9,10} | (reads) | same |
| L15 up c40 | 50% / 47% | **b%2** (R2 0.98): b mod 2 in {0} | **b%2** (R2 0.83): b mod 2 in {0} | (reads) | same |
| L15 up c41 | 52% / 33% | **b%50** (R2 0.89): b mod 50 in {12..37} | **tens(a,b)** (R2 0.59): a//10 in {2,3} -> b//10 in {1,2,7}; a//10 in {4,5} -> b//10 in {1,2,3,7,8}; a//10 in {6,7} -> b//10 in {3,6,7,8}; a//10 in {8} -> b//10 in {6,7,8}; a//10 in {9} -> b//10 in {2,6,7,8}; a//10 in {10} -> b//10 in {1,2,3,6,7,8} | (reads) | same |
| L15 up c54 | 0 / 67% | off (on 0) | **tens(a,b)** (R2 0.63): a//10 in {0} -> b//10 in {0,1}; a//10 in {1} -> b//10 in {1,2,3}; a//10 in {2} -> b//10 in {0,1,2,3,4,5,6}; a//10 in {3} -> b//10 in {1,2,3,4,5,6,7,8,9}; a//10 in {4,10} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {6,7,8,9} -> b//10 in {3,4,5,6,7,8,9,10} | (reads) | same |
| L15 up c64 | 40% / 35% | **b%5** (R2 0.97): b mod 5 in {3..4} | **b%5** (R2 0.76): b mod 5 in {3..4} | (reads) | same |
| L15 up c67 | 0 / 23% | off (on 0) | unexplained (best R2 0.37) | (reads) | same |
| L15 up c72 | 10% / 20% | **b%10** (R2 0.95): b mod 10 in {4} | **b%10** (R2 0.58): b mod 10 in {2..4} | (reads) | same |
| L15 up c117 | 1% / 100% | unexplained (best R2 0.07) | always | (reads) | same |
| L15 up c129 | 29% / 17% | **b%10** (R2 0.96): b mod 10 in {0..1, 9} | unexplained (best R2 0.49) | (reads) | same |
| L15 up c232 | 11% / 0 | **b//10** (R2 0.80): (tens) b in {1..11} | off (on 0) | (reads) | same |
| L15 up c261 | 29% / 20% | **b%10** (R2 0.91): b mod 10 in {4..6} | unexplained (best R2 0.50) | (reads) | same |
| L15 up c308 | 13% / 27% | **b** (R2 0.50): b in {12..13, 31, 41..42, 51, 71, 81, 91} | unexplained (best R2 0.46) | (reads) | same |

</details>

<details><summary>v: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L15 v c189 (kv4) | 92% / 99% | unexplained (best R2 0.41) | always | (reads) | same |

</details>

</details>

<details><summary>layer 16: 148 components with main position `=`</summary>

<details><summary>down: 44</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 down c1 | 100% / 100% | always | same | - | same |
| L16 down c10 | 48% / 47% | **tens(a,b)** (R2 0.80): a//10 in {0} -> b//10 in {2,3,5,6,7,8}; a//10 in {1} -> b//10 in {6}; a//10 in {5} -> b//10 in {3,4,5,7,8,9,10}; a//10 in {6,10} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {8,9} -> b//10 in {1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.73): a//10 in {3} -> b//10 in {2,6}; a//10 in {4} -> b//10 in {3,6,7}; a//10 in {5} -> b//10 in {3,4,6,7}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,8,9}; a//10 in {7,8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {1,2,3,4,5,6,7,8,9} | a: mod100 +2%; b: mod100 +6%; res: mod100 +33% | b: mod100 +5%; res: mod100 +4%, mod50 +2% |
| L16 down c16 | 0 / 28% | off (on 0) | **tens(a,b)** (R2 0.61): a//10 in {3} -> b//10 in {0,1}; a//10 in {4,5} -> b//10 in {0,1,2,3}; a//10 in {6} -> b//10 in {1,2,3}; a//10 in {7,8} -> b//10 in {0,1,2,3,4}; a//10 in {9} -> b//10 in {1,2,3,4,5} | - | same |
| L16 down c17 | 0 / 5% | off (on 0) | unexplained (best R2 0.46) | - | same |
| L16 down c18 | 86% / 0 | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {5,6,7,8,9,10}; a//10 in {1} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {3,4,5,9} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {6,7,8,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | off (on 0) | - | same |
| L16 down c24 | 50% / 42% | **b%20** (R2 0.99): b mod 20 in {0..4, 15..19} | **b%20** (R2 0.69): b mod 20 in {0..4, 15..19} | b: mod20 +43%, mod4 +26% | b: mod20 +25%, mod4 +11% |
| L16 down c25 | 34% / 25% | **tens(a,b)** (R2 0.83): a//10 in {3} -> b//10 in {5,6,7}; a//10 in {4} -> b//10 in {4,5,6,7,8,9}; a//10 in {5,6} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {3,4,5,6,7,8}; a//10 in {8} -> b//10 in {4,5,6,7}; a//10 in {9} -> b//10 in {6} | **tens(a,b)** (R2 0.66): a//10 in {3} -> b//10 in {2}; a//10 in {4,5} -> b//10 in {1,2,3,4,5,6}; a//10 in {6} -> b//10 in {2,3,4,5,6}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {3,4,5} | a: mod100 +3%; b: mod100 +3%; res: mod100 +10% | b: mod100 +2%; res: mod100 +2% |
| L16 down c29 | 50% / 31% | **tens(a,b)** (R2 0.80): a//10 in {0,9,10} -> b//10 in {0,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {8}; a//10 in {5} -> b//10 in {5,6,7}; a//10 in {6} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.74): a//10 in {0} -> b//10 in {0,1,10}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {1}; a//10 in {5} -> b//10 in {3}; a//10 in {6} -> b//10 in {2,3,4,5}; a//10 in {7} -> b//10 in {1,2,3,4,5,6}; a//10 in {8} -> b//10 in {2,3,4,5,6,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | b: mod100 +2%; res: mod100 +19% | - |
| L16 down c33 | 14% / 0 | **b** (R2 0.60): b in {1..11} | off (on 0) | - | same |
| L16 down c42 | 48% / 2% | **b** (R2 0.84): b in {25..53, 81..100} [coarser: b mod 50 in {0, 27..49}, R2 0.86] | unexplained (best R2 0.30) | b: mod50 +10%; res: mod50 +8% | - |
| L16 down c43 | 52% / 11% | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {4,5,6,7,8}; a//10 in {1} -> b//10 in {3,4,5,6,7,8}; a//10 in {2} -> b//10 in {2,3,4,5,6,7,8}; a//10 in {3} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {1,2,3,4,5,7,8,9,10}; a//10 in {6} -> b//10 in {1,2,8,9,10}; a//10 in {7} -> b//10 in {1,8}; a//10 in {8} -> b//10 in {6}; a//10 in {9} -> b//10 in {5,6,7}; a//10 in {10} -> b//10 in {5,6} | unexplained (best R2 0.40) | b: mod100 +2%; res: mod100 +9% | - |
| L16 down c45 | 1% / 98% | unexplained (best R2 0.08) | always | - | a: mod100 +2%; res: mod100 +8%, mod50 +3% |
| L16 down c49 | 15% / 0 | **tens(a,b)** (R2 0.70): a//10 in {0,3} -> b//10 in {1,2}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2,4,5,6,7,8} -> b//10 in {1} | off (on 0) | - | same |
| L16 down c53 | 4% / 33% | unexplained (best R2 0.37) | **tens(a,b)** (R2 0.64): a//10 in {3} -> b//10 in {6,7,10}; a//10 in {4} -> b//10 in {4,6,7,8,9,10}; a//10 in {5,8} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {6,7,10} -> b//10 in {5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10} | - | b: mod100 +3% |
| L16 down c61 | 40% / 2% | **res//10** (R2 0.78): (tens) res in {2..90} | unexplained (best R2 0.50) | res: mod100 +2% | - |
| L16 down c67 | 6% / 11% | unexplained (best R2 0.24) | unexplained (best R2 0.47) | - | res: mod50 +3%, mod25 +4%, mod20 +4%, mod10 +3% |
| L16 down c71 | 0 / 62% | off (on 0) | unexplained (best R2 0.42) | - | a: mod100 +3%, mod50 +3%; b: mod10 +11%; res: mod100 +3%, mod50 +3%, mod25 +3%, mod20 +3% |
| L16 down c73 | 49% / 31% | **a%20** (R2 0.95): a mod 20 in {0..4, 15..19} | **a** (R2 0.61): a in {21, 35..41, 43, 55..64, 75..84, 99..100} | a: mod20 +19%, mod4 +8% | a: mod20 +9%, mod4 +4% |
| L16 down c76 | 26% / 15% | **units(a,b)** (R2 0.92): a%10 in {2,7} -> b%10 in {0,1,5,6}; a%10 in {3,8} -> b%10 in {0,1,4,5,6,9}; a%10 in {4} -> b%10 in {0,4,5,9}; a%10 in {9} -> b%10 in {0,4,5} | **units(a,b)** (R2 0.52): a%10 in {2} -> b%10 in {0,4,5}; a%10 in {3} -> b%10 in {0,1,4,5,6}; a%10 in {7,8} -> b%10 in {0,5} | a: mod5 +3%; res: mod5 +20% | res: mod5 +7% |
| L16 down c77 | 0 / 23% | off (on 0) | **tens(a,b)** (R2 0.70): a//10 in {0,1,2,3} -> b//10 in {0}; a//10 in {4,5} -> b//10 in {0,1}; a//10 in {6,7,8,10} -> b//10 in {0,1,2}; a//10 in {9} -> b//10 in {0,1,2,3,4,5} | - | b: mod100 +3%, mod50 +3%, mod25 +3% |
| L16 down c79 | 33% / 28% | **units(a,b)** (R2 0.93): a%10 in {0,1,5,6} -> b%10 in {0,1,4,5,6,9}; a%10 in {4,9} -> b%10 in {0,1,5,6} | **units(a,b)** (R2 0.76): a%10 in {0,1,5,6} -> b%10 in {0,1,4,5,6,9}; a%10 in {4,9} -> b%10 in {0,4,5,9} | a: mod5 +5%; b: mod5 +2%; res: mod5 +11% | a: mod5 +4%; b: mod5 +3%; res: mod5 +8% |
| L16 down c86 | 0 / 0 | off (on 0) | same | - | same |
| L16 down c90 | 1% / 2% | unexplained (best R2 0.32) | unexplained (best R2 0.39) | - | res: mod2 +3% |
| L16 down c94 | 21% / 22% | **units(a,b)** (R2 0.89): a%10 in {1,2,6,7} -> b%10 in {0,4,5,9}; a%10 in {3,8} -> b%10 in {4,9} | unexplained (best R2 0.47) | res: mod5 +16% | res: mod5 +9% |
| L16 down c96 | 0 / 0 | off (on 0) | same | - | same |
| L16 down c106 | 35% / 16% | **tens(a,b)** (R2 0.77): a//10 in {0} -> b//10 in {2,3,4}; a//10 in {1,4} -> b//10 in {1,2,3}; a//10 in {2} -> b//10 in {2,3}; a//10 in {5} -> b//10 in {1,2,3,4}; a//10 in {6,7} -> b//10 in {0,1,2,3,4,10}; a//10 in {8} -> b//10 in {0,1,2,3,10}; a//10 in {9} -> b//10 in {1,2} | **tens(a,b)** (R2 0.70): a//10 in {5} -> b//10 in {7,8}; a//10 in {6} -> b//10 in {5,6,7,8}; a//10 in {7} -> b//10 in {0,6,7,8,9}; a//10 in {8} -> b//10 in {0,7,8,9} | a: mod100 +2%; res: mod100 +9% | - |
| L16 down c107 | 22% / 15% | **units(a,b)** (R2 0.90): a%10 in {3,4,5,6,7,8} -> b%10 in {3,4,5}; a%10 in {9} -> b%10 in {3,4} | unexplained (best R2 0.50) | a: mod10 +2%; b: mod10 +4% | b: mod10 +3%; res: mod10 +3% |
| L16 down c127 | 22% / 18% | **units(a,b)** (R2 0.90): a%10 in {0,5} -> b%10 in {4,9}; a%10 in {1} -> b%10 in {0,3,4,5,8,9}; a%10 in {2,6,7} -> b%10 in {3,4,8,9} | **units(a,b)** (R2 0.69): a%10 in {0,5} -> b%10 in {1,6}; a%10 in {1} -> b%10 in {0,1,2,5,6,7}; a%10 in {2} -> b%10 in {1,2,7}; a%10 in {6,7} -> b%10 in {1,2,6,7} | res: mod5 +17% | res: mod5 +16% |
| L16 down c130 | 25% / 12% | **tens(a,b)** (R2 0.82): a//10 in {0} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {1} -> b//10 in {3,4,5,6,7,8}; a//10 in {2} -> b//10 in {5,6}; a//10 in {8} -> b//10 in {5,6,7,8,9}; a//10 in {9,10} -> b//10 in {4,5,6,7,8,9} | **tens(a,b)** (R2 0.81): a//10 in {8} -> b//10 in {0,1,2,3,4}; a//10 in {9} -> b//10 in {0,1,2,3,4,5}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6} | res: mod100 +6% | - |
| L16 down c152 | 0 / 15% | off (on 0) | **tens(a,b)** (R2 0.61): a//10 in {3} -> b//10 in {4}; a//10 in {4} -> b//10 in {5,6,7,8}; a//10 in {5} -> b//10 in {6,7,8}; a//10 in {6} -> b//10 in {7,8,9}; a//10 in {7} -> b//10 in {8,9,10}; a//10 in {8} -> b//10 in {9,10}; a//10 in {10} -> b//10 in {8,9} | - | same |
| L16 down c161 | 8% / 16% | **b** (R2 0.87): b in {1..7} | **tens(a,b)** (R2 0.86): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1,2,3,4,5,6,7,8,9,10} -> b//10 in {0} | - | b: mod25 +2% |
| L16 down c164 | 0 / 15% | off (on 0) | **a//10** (R2 0.70): (tens) a in {1..16} | - | a: mod100 +3%, mod50 +4%, mod25 +2% |
| L16 down c175 | 1% / 19% | unexplained (best R2 0.30) | unexplained (best R2 0.46) | - | a: mod5 +3% |
| L16 down c176 | 1% / 7% | unexplained (best R2 0.15) | unexplained (best R2 0.21) | - | same |
| L16 down c214 | 4% / 1% | **a** (R2 0.63): a in {44, 64, 84} [coarser: a mod 20 in {4}, R2 0.82] | **a** (R2 0.51): a in {64, 84} | - | same |
| L16 down c275 | 21% / 14% | **units(a,b)** (R2 0.89): a%10 in {3,8} -> b%10 in {0,1,2,5,6,7}; a%10 in {4,9} -> b%10 in {0,1,5,6} | **units(a,b)** (R2 0.58): a%10 in {3} -> b%10 in {0,3,4,8,9}; a%10 in {4,9} -> b%10 in {0,4,5,9}; a%10 in {8} -> b%10 in {3,4,8,9} | res: mod5 +9% | res: mod5 +7% |
| L16 down c284 | 0 / 0 | off (on 0) | same | - | same |
| L16 down c287 | 4% / 0 | unexplained (best R2 0.28) | off (on 0) | - | same |
| L16 down c288 | 0 / 11% | off (on 0) | **tens(a,b)** (R2 0.70): a//10 in {0} -> b//10 in {6,7,8,9}; a//10 in {1} -> b//10 in {5,6,7,8,9}; a//10 in {2} -> b//10 in {7,8,9} | - | same |
| L16 down c295 | 23% / 19% | **b%10** (R2 0.80): b mod 10 in {6..7} | **b%10** (R2 0.54): b mod 10 in {3..4} | b: mod10 +4% | b: mod10 +5%, mod5 +2% |
| L16 down c470 | 2% / 4% | unexplained (best R2 0.39) | unexplained (best R2 0.45) | - | same |
| L16 down c501 | 0 / 14% | off (on 0) | **b** (R2 0.69): b in {1..10, 12, 14, 16, 18} | - | same |
| L16 down c681 | 4% / 2% | unexplained (best R2 0.30) | unexplained (best R2 0.32) | - | same |
| L16 down c757 | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>gate: 26</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 gate c1 | 99% / 14% | always | **tens(a,b)** (R2 0.54): a//10 in {2} -> b//10 in {1}; a//10 in {3} -> b//10 in {2}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {3,4,5}; a//10 in {7} -> b//10 in {4,5,6} | (reads) | same |
| L16 gate c4 | 100% / 100% | always | same | (reads) | same |
| L16 gate c10 | 49% / 40% | **a//10** (R2 0.78): (tens) a in {2..4, 56..100} | **a//10** (R2 0.71): (tens) a in {58..97, 100} | (reads) | same |
| L16 gate c25 | 38% / 27% | **tens(a,b)** (R2 0.77): a//10 in {0,1} -> b//10 in {0,1}; a//10 in {2} -> b//10 in {0,1,2}; a//10 in {3,8} -> b//10 in {5,6,7}; a//10 in {4} -> b//10 in {5,6,7,8}; a//10 in {5} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {6} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {3,4,5,6,7,8}; a//10 in {9} -> b//10 in {6}; a//10 in {10} -> b//10 in {5,6} | **tens(a,b)** (R2 0.67): a//10 in {3} -> b//10 in {2}; a//10 in {4} -> b//10 in {1,2,3,4,5,6}; a//10 in {5} -> b//10 in {1,2,3,4,6}; a//10 in {6} -> b//10 in {1,2,3,4,5,6,7}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {3,4,5} | (reads) | same |
| L16 gate c29 | 40% / 17% | **tens(a,b)** (R2 0.82): a//10 in {0} -> b//10 in {2,3,4,5,6,7,8}; a//10 in {1} -> b//10 in {3,4,5,6,7}; a//10 in {6} -> b//10 in {4,5,6,7}; a//10 in {7} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {8,9,10} -> b//10 in {2,3,4,5,6,7,8,9} | **tens(a,b)** (R2 0.75): a//10 in {6} -> b//10 in {3,4}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {2,3,4,5,6}; a//10 in {9,10} -> b//10 in {1,2,3,4,5,6,7} | (reads) | same |
| L16 gate c37 | 100% / 100% | always | same | (reads) | same |
| L16 gate c42 | 2% / 0 | unexplained (best R2 0.34) | off (on 0) | (reads) | same |
| L16 gate c43 | 22% / 23% | **tens(a,b)** (R2 0.76): a//10 in {2} -> b//10 in {1}; a//10 in {3,9} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {1,2,3}; a//10 in {5,6} -> b//10 in {0,1,2,3,10}; a//10 in {7} -> b//10 in {0,1,2,10}; a//10 in {8} -> b//10 in {0,1,2} | **tens(a,b)** (R2 0.72): a//10 in {3} -> b//10 in {7}; a//10 in {4} -> b//10 in {6,7,8}; a//10 in {5,6,7,8} -> b//10 in {6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {7,8,9} | (reads) | same |
| L16 gate c71 | 0 / 29% | off (on 0) | **b%10** (R2 0.51): b mod 10 in {6..9} | (reads) | same |
| L16 gate c73 | 50% / 31% | **a%20** (R2 0.98): a mod 20 in {5..14} | **a** (R2 0.63): a in {27..32, 45..47, 49..53, 65..74, 85..94} | (reads) | same |
| L16 gate c76 | 46% / 27% | **units(a,b)** (R2 0.85): a%10 in {0,1,5} -> b%10 in {2,3,4,7,8,9}; a%10 in {2,3,7,8} -> b%10 in {2,3,7,8}; a%10 in {4} -> b%10 in {3,8}; a%10 in {6} -> b%10 in {2,3,7,8,9}; a%10 in {9} -> b%10 in {2,3,8} | **units(a,b)** (R2 0.51): a%10 in {0,5} -> b%10 in {1,2,3,6,7,8}; a%10 in {1,6} -> b%10 in {2,3,7,8}; a%10 in {2,3,7,8,9} -> b%10 in {2,7}; a%10 in {4} -> b%10 in {2} | (reads) | same |
| L16 gate c79 | 0 / 4% | off (on 0) | unexplained (best R2 0.25) | (reads) | same |
| L16 gate c90 | 0 / 0 | off (on 0) | same | (reads) | same |
| L16 gate c94 | 50% / 35% | **units(a,b)** (R2 0.89): a%10 in {0,1,5} -> b%10 in {1,2,3,6,7,8}; a%10 in {2,7} -> b%10 in {2,6,7}; a%10 in {3,4,6,8,9} -> b%10 in {1,2,6,7,8} | **units(a,b)** (R2 0.54): a%10 in {0,1,4,5,9} -> b%10 in {2,3,4,8,9}; a%10 in {2} -> b%10 in {3,8}; a%10 in {3,6,8} -> b%10 in {3,4,8,9}; a%10 in {7} -> b%10 in {3} | (reads) | same |
| L16 gate c106 | 47% / 20% | **b//10** (R2 0.86): (tens) b in {1..47, 49} | **tens(a,b)** (R2 0.66): a//10 in {4,9} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {6,7,8}; a//10 in {6} -> b//10 in {5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9,10}; a//10 in {8} -> b//10 in {5,6,7,8,9}; a//10 in {10} -> b//10 in {6,7,8,9} | (reads) | same |
| L16 gate c107 | 8% / 5% | unexplained (best R2 0.19) | unexplained (best R2 0.28) | (reads) | same |
| L16 gate c127 | 32% / 20% | **units(a,b)** (R2 0.85): a%10 in {0,3,5,8} -> b%10 in {4,9}; a%10 in {1,2,6,7} -> b%10 in {0,3,4,5,8,9} | **units(a,b)** (R2 0.54): a%10 in {0,5,7} -> b%10 in {1,6}; a%10 in {1,6} -> b%10 in {0,1,2,5,6,7}; a%10 in {2} -> b%10 in {1,2,6,7} | (reads) | same |
| L16 gate c130 | 45% / 24% | **a//10** (R2 0.80): (tens) a in {1..25, 81..100} | **tens(a,b)** (R2 0.71): a//10 in {0} -> b//10 in {0,1,2,9,10}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {1,2}; a//10 in {8} -> b//10 in {0,3,4,5,6,8,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | (reads) | same |
| L16 gate c175 | 0 / 3% | off (on 0) | unexplained (best R2 0.47) | (reads) | same |
| L16 gate c392 | 30% / 16% | **units(a,b)** (R2 0.88): a%10 in {0,5,9} -> b%10 in {4,9}; a%10 in {1,2,3,6,7,8} -> b%10 in {3,4,8,9}; a%10 in {4} -> b%10 in {9} | unexplained (best R2 0.42) | (reads) | same |
| L16 gate c470 | 11% / 14% | **a%10** (R2 0.49): a mod 10 in {0} | unexplained (best R2 0.41) | (reads) | same |
| L16 gate c493 | 13% / 9% | **tens(a,b)** (R2 0.64): a//10 in {4,5} -> b//10 in {4,5,6,7,8}; a//10 in {6} -> b//10 in {6} | **tens(a,b)** (R2 0.52): a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3,4}; a//10 in {5} -> b//10 in {2,3,4}; a//10 in {10} -> b//10 in {4} | (reads) | same |
| L16 gate c501 | 7% / 23% | **b** (R2 0.64): b in {2..9} | **tens(a,b)** (R2 0.72): a//10 in {0,1,2,3} -> b//10 in {0}; a//10 in {4,5,6,7,8} -> b//10 in {0,1,2}; a//10 in {9} -> b//10 in {0,1,2,3}; a//10 in {10} -> b//10 in {0,1} | (reads) | same |
| L16 gate c527 | 15% / 13% | **units(a,b)** (R2 0.81): a%10 in {4} -> b%10 in {3,4,5,6,7}; a%10 in {5,6} -> b%10 in {3,4,5,6} | unexplained (best R2 0.35) | (reads) | same |
| L16 gate c832 | 38% / 14% | **a//10** (R2 0.82): (tens) a in {11, 13..49} | **tens(a,b)** (R2 0.62): a//10 in {1} -> b//10 in {0,1}; a//10 in {2} -> b//10 in {0,1,2}; a//10 in {3,4} -> b//10 in {0,1,2,3} | (reads) | same |
| L16 gate c872 | 8% / 10% | **tens(a,b)** (R2 0.62): a//10 in {1} -> b//10 in {5,6,7}; a//10 in {9} -> b//10 in {5,6,7,8}; a//10 in {10} -> b//10 in {4,5,6,7,8} | **tens(a,b)** (R2 0.71): a//10 in {9} -> b//10 in {0,1,2,3,4,5,8}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8,9} | (reads) | same |

</details>

<details><summary>k: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 k c67 (kv4) | 100% / 100% | always | same | (reads) | same |

</details>

<details><summary>o: 52</summary>

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
| L16 o c160 (H22) | 51% / 74% | **a//10** (R2 0.91): (tens) a in {51..100} | **tens(a,b)** (R2 0.79): a//10 in {0,1,2,3} -> b//10 in {5,6,7,8,9,10}; a//10 in {4} -> b//10 in {2,5,6,7,8,9,10}; a//10 in {5,6,7,8,9} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,10} | a: mod100 +3% | a: mod100 +2% |
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

<details><summary>q: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 q c136 (H21) | 100% / 100% | always | same | (reads) | same |

</details>

<details><summary>up: 23</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 up c4 | 100% / 100% | always | same | (reads) | same |
| L16 up c10 | 52% / 43% | **b//10** (R2 0.79): (tens) b in {53..100} | **tens(a,b)** (R2 0.74): a//10 in {2} -> b//10 in {6}; a//10 in {3} -> b//10 in {3,5,6,7}; a//10 in {4} -> b//10 in {3,4,6,7,8}; a//10 in {5,6,7} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {8,9,10} -> b//10 in {4,5,6,7,8,9,10} | (reads) | same |
| L16 up c24 | 49% / 29% | **b%20** (R2 0.96): b mod 20 in {0..4, 15..19} | unexplained (best R2 0.42) | (reads) | same |
| L16 up c25 | 35% / 17% | **b//10** (R2 0.75): (tens) b in {51..81, 83..85} | **tens(a,b)** (R2 0.53): a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4,9} -> b//10 in {3}; a//10 in {5,6,7,8} -> b//10 in {3,4}; a//10 in {10} -> b//10 in {1,2,3,4,5,6,7,8} | (reads) | same |
| L16 up c29 | 63% / 31% | **tens(a,b)** (R2 0.67): a//10 in {0} -> b//10 in {0,3,4,5,6}; a//10 in {1,2} -> b//10 in {3,4,5,6}; a//10 in {3} -> b//10 in {3,4,5,6,7}; a//10 in {4} -> b//10 in {2,3,4,5,6,7}; a//10 in {5,10} -> b//10 in {2,3,4,5,6,7,8}; a//10 in {6} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {1,2,3,4,5,6,7,8,9}; a//10 in {8,9} -> b//10 in {1,2,3,4,5,6,7,8} | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {0,1,2,10}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {3,4,5,6}; a//10 in {7} -> b//10 in {3,4,5,6,7,10}; a//10 in {8} -> b//10 in {0,3,4,5,6,7,10}; a//10 in {9} -> b//10 in {0,1,3,4,5,6,7,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,10} | (reads) | same |
| L16 up c42 | 51% / 0 | **b//10** (R2 0.77): (tens) b in {1..49} | off (on 0) | (reads) | same |
| L16 up c43 | 50% / 12% | **tens(a,b)** (R2 0.75): a//10 in {0,9} -> b//10 in {2,3,4,5,6,7,8,9}; a//10 in {1,2} -> b//10 in {2,3,4,5,6,7,8}; a//10 in {3,10} -> b//10 in {2,3,4,5,6,7}; a//10 in {4} -> b//10 in {3,4,5,6}; a//10 in {5,7} -> b//10 in {3,4,5}; a//10 in {8} -> b//10 in {3,4,5,6,8} | **tens(a,b)** (R2 0.55): a//10 in {8} -> b//10 in {4,5,6}; a//10 in {9} -> b//10 in {3,4,5,6,7}; a//10 in {10} -> b//10 in {2,3,4,5,6,7} | (reads) | same |
| L16 up c71 | 0 / 23% | off (on 0) | unexplained (best R2 0.37) | (reads) | same |
| L16 up c76 | 47% / 31% | **units(a,b)** (R2 0.89): a%10 in {1,6} -> b%10 in {5}; a%10 in {2,4,7,9} -> b%10 in {0,1,4,5,6,9}; a%10 in {3,8} -> b%10 in {0,1,2,3,4,5,6,7,8,9} | unexplained (best R2 0.45) | (reads) | same |
| L16 up c77 | 0 / 47% | off (on 0) | **tens(a,b)** (R2 0.68): a//10 in {0,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,9,10}; a//10 in {4,5} -> b//10 in {0,1,10}; a//10 in {6,7,8} -> b//10 in {0,1,2}; a//10 in {9} -> b//10 in {0,1,2,3,4,5} | (reads) | same |
| L16 up c90 | 2% / 2% | unexplained (best R2 0.14) | unexplained (best R2 0.22) | (reads) | same |
| L16 up c94 | 3% / 5% | unexplained (best R2 0.39) | unexplained (best R2 0.33) | (reads) | same |
| L16 up c106 | 41% / 23% | **a//10** (R2 0.78): (tens) a in {1..42, 100} | **a//10** (R2 0.61): (tens) a in {10..11, 13..32, 35, 100} | (reads) | same |
| L16 up c107 | 0 / 1% | off (on 0) | **a//10** (R2 0.81): (tens) a in {100} | (reads) | same |
| L16 up c127 | 28% / 19% | **units(a,b)** (R2 0.87): a%10 in {0,5} -> b%10 in {4,9}; a%10 in {1,6,7} -> b%10 in {0,3,4,5,8,9}; a%10 in {2} -> b%10 in {3,4,5,8,9} | unexplained (best R2 0.44) | (reads) | same |
| L16 up c130 | 28% / 80% | **b** (R2 0.68): b in {49, 53..79, 81} | **tens(a,b)** (R2 0.62): a//10 in {0,1,2,3,4} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,3,4,5,9}; a//10 in {6,7,8,10} -> b//10 in {0,1,2,3,4,5}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7} | (reads) | same |
| L16 up c161 | 8% / 10% | **b** (R2 0.91): b in {1..8} | **b//10** (R2 0.78): (tens) b in {1..9} | (reads) | same |
| L16 up c275 | 23% / 6% | **units(a,b)** (R2 0.69): a%10 in {3,4} -> b%10 in {0,1,5,6}; a%10 in {7} -> b%10 in {6}; a%10 in {8} -> b%10 in {0,1,2,4,5,6,7,9}; a%10 in {9} -> b%10 in {0,1,4,5,6,9} | unexplained (best R2 0.23) | (reads) | same |
| L16 up c389 | 18% / 12% | **units(a,b)** (R2 0.71): a%10 in {0} -> b%10 in {0,5,6,7}; a%10 in {4,6,8,9} -> b%10 in {5}; a%10 in {5} -> b%10 in {0,1,2,3,4,5,6,7,8,9} | **a** (R2 0.49): a in {40, 45, 50, 55, 65, 75, 80, 85, 90, 95, 100} [coarser: a mod 5 in {0}, R2 0.82] | (reads) | same |
| L16 up c392 | 28% / 13% | **units(a,b)** (R2 0.79): a%10 in {0} -> b%10 in {4}; a%10 in {3} -> b%10 in {0,1,4,5,6,9}; a%10 in {4,9} -> b%10 in {0,1,3,4,5,6,9}; a%10 in {8} -> b%10 in {0,1,4,5,6} | unexplained (best R2 0.37) | (reads) | same |
| L16 up c420 | 32% / 1% | **tens(a,b)** (R2 0.73): a//10 in {0} -> b//10 in {2}; a//10 in {1} -> b//10 in {2,3}; a//10 in {2} -> b//10 in {1,3}; a//10 in {3} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {1,2,3}; a//10 in {5,6,9,10} -> b//10 in {1,2,3,4}; a//10 in {7,8} -> b//10 in {0,1,2,3,4} | unexplained (best R2 0.09) | (reads) | same |
| L16 up c757 | 10% / 12% | **tens(a,b)** (R2 0.64): a//10 in {5} -> b//10 in {0}; a//10 in {6} -> b//10 in {0,1,2,6}; a//10 in {7,8} -> b//10 in {0,1} | **tens(a,b)** (R2 0.56): a//10 in {6} -> b//10 in {0,3,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,7,8,10} | (reads) | same |
| L16 up c821 | 49% / 25% | **b%20** (R2 0.95): b mod 20 in {5..14} | unexplained (best R2 0.37) | (reads) | same |

</details>

<details><summary>v: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 v c3 (kv4) | 100% / 100% | always | same | (reads) | same |

</details>

</details>

<details><summary>layer 17: 86 components with main position `=`</summary>

<details><summary>down: 38</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L17 down c3 | 100% / 100% | always | same | - | same |
| L17 down c5 | 100% / 1% | always | unexplained (best R2 0.10) | - | same |
| L17 down c10 | 65% / 46% | **b%50** (R2 0.84): b mod 50 in {11..43} | **tens(a,b)** (R2 0.69): a//10 in {1} -> b//10 in {1,2,3}; a//10 in {2} -> b//10 in {1,2,3,6,7}; a//10 in {3,4,5,6,7,8,9,10} -> b//10 in {1,2,3,6,7,8} | b: mod50 +15% | b: mod50 +11% |
| L17 down c11 | 46% / 27% | **tens(a,b)** (R2 0.79): a//10 in {2} -> b//10 in {6,7}; a//10 in {3} -> b//10 in {6,7,9,10}; a//10 in {4,6} -> b//10 in {0,1,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {0,5,6,7,8,9,10}; a//10 in {9,10} -> b//10 in {4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.61): a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {10}; a//10 in {5} -> b//10 in {0,1,2,8,9,10}; a//10 in {6} -> b//10 in {0,1,2,9,10}; a//10 in {7} -> b//10 in {0,1,2,3,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,10}; a//10 in {9} -> b//10 in {1,3}; a//10 in {10} -> b//10 in {1,2,3,4,5,10} | a: mod100 +9%; b: mod100 +6%; res: mod100 +22%, mod50 +5% | - |
| L17 down c13 | 0 / 40% | off (on 0) | **tens(a,b)** (R2 0.61): a//10 in {2} -> b//10 in {3,4,6}; a//10 in {3} -> b//10 in {3,4,5,6,7}; a//10 in {4} -> b//10 in {4,5,6,7,8}; a//10 in {5} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {6,10} -> b//10 in {4,5,6,7,8,9}; a//10 in {7} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {5,6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10} | - | a: mod100 +3%; b: mod100 +4%; res: mod100 +5%, mod25 +2% |
| L17 down c15 | 44% / 9% | **tens(a,b)** (R2 0.62): a//10 in {3} -> b//10 in {6}; a//10 in {4} -> b//10 in {1,2,3,5,6,7}; a//10 in {5} -> b//10 in {2,3,6}; a//10 in {6,9} -> b//10 in {1,2,3,4,5,6,7}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,7,10}; a//10 in {10} -> b//10 in {1,2,3,4,5,6,7,8} | **tens(a,b)** (R2 0.67): a//10 in {7} -> b//10 in {0,1,2,3,4,5}; a//10 in {8} -> b//10 in {3}; a//10 in {10} -> b//10 in {0,1,2,3,4} | a: mod100 +3% | - |
| L17 down c20 | 1% / 2% | unexplained (best R2 0.08) | unexplained (best R2 0.15) | - | same |
| L17 down c22 | 25% / 25% | **units(a,b)** (R2 0.98): a%10 in {1,3,5,7,9} -> b%10 in {0,2,4,6,8} | **units(a,b)** (R2 0.81): a%10 in {1,3,5,7,9} -> b%10 in {0,2,4,6,8} | a: mod2 +8%; res: mod2 +42% | a: mod2 +13%; res: mod2 +45% |
| L17 down c23 | 65% / 22% | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {1,4,8,9}; a//10 in {1} -> b//10 in {1,3,4,5,6,8,9,10}; a//10 in {2,3,6} -> b//10 in {1,2,3,4,5,6,8,9,10}; a//10 in {4} -> b//10 in {1,3,8,9}; a//10 in {5} -> b//10 in {1,3,4,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,8,9,10}; a//10 in {9} -> b//10 in {1,3,4,6,8}; a//10 in {10} -> b//10 in {3,4,8,9} | **tens(a,b)** (R2 0.58): a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {1,3}; a//10 in {6} -> b//10 in {0,5}; a//10 in {7} -> b//10 in {0,1,3,4,5,6}; a//10 in {8} -> b//10 in {3,4,5,6,8,9} | b: mod50 +13%; res: mod50 +48% | b: mod50 +2% |
| L17 down c25 | 0 / 0 | off (on 0) | same | - | same |
| L17 down c26 | 0 / 5% | off (on 0) | **tens(a,b)** (R2 0.62): a//10 in {8} -> b//10 in {7,8,10}; a//10 in {9} -> b//10 in {8,9,10}; a//10 in {10} -> b//10 in {4,5,6,7,8,9,10} | - | same |
| L17 down c29 | 1% / 35% | unexplained (best R2 0.10) | **tens(a,b)** (R2 0.62): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8}; a//10 in {1} -> b//10 in {0,6,7}; a//10 in {3} -> b//10 in {0}; a//10 in {4,5,6} -> b//10 in {0,1,2}; a//10 in {7,8} -> b//10 in {0,1,2,3}; a//10 in {9} -> b//10 in {0,1,2,3,4,5} | - | b: mod100 +4%, mod50 +3%, mod25 +3%; res: mod100 +3% |
| L17 down c33 | 60% / 26% | **a%50** (R2 0.72): a mod 50 in {19..47} | **tens(a,b)** (R2 0.59): a//10 in {2} -> b//10 in {0,1,2,3}; a//10 in {3} -> b//10 in {0,1,2,3,4}; a//10 in {4} -> b//10 in {3}; a//10 in {7} -> b//10 in {3,4,5,6,7,8}; a//10 in {8} -> b//10 in {0,3,4,5,6,7,8,9}; a//10 in {9} -> b//10 in {8}; a//10 in {10} -> b//10 in {8,9} | a: mod50 +15% | a: mod50 +6% |
| L17 down c34 | 35% / 21% | **units(a,b)** (R2 0.93): a%10 in {0} -> b%10 in {5,6,7}; a%10 in {1} -> b%10 in {5,6}; a%10 in {5} -> b%10 in {0,1,2,5,6,7,8,9}; a%10 in {6} -> b%10 in {0,1,5,6,7,8,9}; a%10 in {7} -> b%10 in {0,5,6,7,8,9}; a%10 in {8} -> b%10 in {5,6,7,8,9}; a%10 in {9} -> b%10 in {5,6,7,8} | **units(a,b)** (R2 0.55): a%10 in {0} -> b%10 in {4}; a%10 in {5,6} -> b%10 in {0,1,2,3,4,9}; a%10 in {7} -> b%10 in {0,1,2,3,4}; a%10 in {8} -> b%10 in {2,3,4}; a%10 in {9} -> b%10 in {3,4} | a: mod10 +8%, mod5 +2%; b: mod10 +6%; res: mod10 +25% | a: mod10 +3%; b: mod10 +4%; res: mod10 +16% |
| L17 down c37 | 43% / 31% | **units(a,b)** (R2 0.92): a%10 in {0,8,9} -> b%10 in {0,1,2,3,4,8,9}; a%10 in {1,2} -> b%10 in {0,1,2,3,8,9}; a%10 in {3} -> b%10 in {0,1,8,9}; a%10 in {4} -> b%10 in {0,8,9}; a%10 in {5} -> b%10 in {8,9} | **units(a,b)** (R2 0.68): a%10 in {0,9} -> b%10 in {0,1,2,7,8,9}; a%10 in {1,2,8} -> b%10 in {0,1,2,8,9}; a%10 in {3} -> b%10 in {0,1,2,9}; a%10 in {4} -> b%10 in {0,1} | a: mod10 +5%; b: mod10 +7%; res: mod10 +6% | a: mod10 +4%; b: mod10 +5%; res: mod10 +4% |
| L17 down c38 | 11% / 0 | unexplained (best R2 0.38) | off (on 0) | - | same |
| L17 down c39 | 39% / 24% | **units(a,b)** (R2 0.87): a%10 in {0,4} -> b%10 in {5,6,7,8,9}; a%10 in {1} -> b%10 in {5,6,7,8}; a%10 in {2} -> b%10 in {7,8}; a%10 in {3} -> b%10 in {0,6,7,8,9}; a%10 in {5} -> b%10 in {6,7,8}; a%10 in {7} -> b%10 in {6,7,8,9}; a%10 in {8,9} -> b%10 in {0,5,6,7,8,9} | **units(a,b)** (R2 0.54): a%10 in {0,3,4,8,9} -> b%10 in {1,2,3,4}; a%10 in {1} -> b%10 in {2,3,4}; a%10 in {5} -> b%10 in {3,4}; a%10 in {7} -> b%10 in {1,2,3} | a: mod10 +6%; res: mod10 +19% | a: mod10 +3%; res: mod10 +14% |
| L17 down c40 | 47% / 15% | **tens(a,b)** (R2 0.78): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7}; a//10 in {1,2} -> b//10 in {2,3,4,5,6,7}; a//10 in {3,4} -> b//10 in {1,2,3,4,5,6}; a//10 in {5} -> b//10 in {1,2,3,4,5}; a//10 in {6} -> b//10 in {1,2,3,4}; a//10 in {8} -> b//10 in {2}; a//10 in {9,10} -> b//10 in {0,1,2,3,4} | unexplained (best R2 0.44) | res: mod100 +10% | - |
| L17 down c47 | 33% / 19% | **units(a,b)** (R2 0.95): a%10 in {0,4,5,9} -> b%10 in {0,3,4,5,8,9}; a%10 in {3,8} -> b%10 in {0,4,5,9} | **units(a,b)** (R2 0.55): a%10 in {0,5} -> b%10 in {0,1,2,5,6,7}; a%10 in {3} -> b%10 in {0}; a%10 in {4} -> b%10 in {0,1,5,6}; a%10 in {8} -> b%10 in {5}; a%10 in {9} -> b%10 in {0,6} | a: mod5 +5%; b: mod5 +5%; res: mod5 +10% | a: mod5 +3%; res: mod5 +3% |
| L17 down c49 | 38% / 27% | **units(a,b)** (R2 0.90): a%10 in {5} -> b%10 in {0,1,2,3,5,6,7,9}; a%10 in {6} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {7} -> b%10 in {0,1,2,3,5,6,7,8,9}; a%10 in {8} -> b%10 in {0,1,2,5,6,8,9}; a%10 in {9} -> b%10 in {0,1,5,9} | **a** (R2 0.50): a in {1, 26..27, 36..37, 46..47, 55..58, 65..68, 75..78, 85..88, 95..97} [coarser: a mod 20 in {6..7, 15..17}, R2 0.80] | b: mod10 +2%; res: mod10 +15% | res: mod10 +12% |
| L17 down c52 | 26% / 17% | **units(a,b)** (R2 0.87): a%10 in {0} -> b%10 in {2,3,4}; a%10 in {1} -> b%10 in {1,2,3,4}; a%10 in {2} -> b%10 in {0,1,2,3,4}; a%10 in {3,4} -> b%10 in {0,1,2,3,4,9} | unexplained (best R2 0.49) | a: mod10 +5%; b: mod10 +4%; res: mod10 +16% | a: mod10 +2%; res: mod10 +8% |
| L17 down c61 | 40% / 15% | **units(a,b)** (R2 0.88): a%10 in {0,5} -> b%10 in {3,4,8,9}; a%10 in {1,6} -> b%10 in {3,8}; a%10 in {3,4,9} -> b%10 in {0,3,4,5,8,9}; a%10 in {8} -> b%10 in {0,1,3,4,5,6,7,8,9} | unexplained (best R2 0.26) | a: mod5 +4%; b: mod5 +4%; res: mod5 +10% | res: mod5 +3% |
| L17 down c67 | 25% / 20% | **units(a,b)** (R2 0.98): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} | **units(a,b)** (R2 0.73): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} | a: mod2 +18%; b: mod2 +15%; res: mod2 +29% | a: mod2 +6%; b: mod2 +6%; res: mod2 +17% |
| L17 down c70 | 18% / 4% | **units(a,b)** (R2 0.79): a%10 in {2} -> b%10 in {1,2,3,4,7,8,9}; a%10 in {4} -> b%10 in {2,7}; a%10 in {7} -> b%10 in {1,2,3,4,6,7,8,9}; a%10 in {9} -> b%10 in {2} | unexplained (best R2 0.22) | a: mod5 +4% | - |
| L17 down c86 | 31% / 16% | **units(a,b)** (R2 0.85): a%10 in {0} -> b%10 in {6,7,8}; a%10 in {1} -> b%10 in {5,6,7,8,9}; a%10 in {2} -> b%10 in {5,6,7,8}; a%10 in {3} -> b%10 in {0,1,2,6,7,8,9}; a%10 in {4} -> b%10 in {0,1,2,8,9}; a%10 in {5} -> b%10 in {0,1,8,9}; a%10 in {6} -> b%10 in {0,8,9} | unexplained (best R2 0.48) | a: mod10 +2%; res: mod10 +11% | res: mod10 +8% |
| L17 down c89 | 0 / 19% | off (on 0) | **res//10** (R2 0.53): (tens) res in {0..22, 99} | - | res: mod100 +3%, mod50 +5%, mod25 +3%, mod20 +3% |
| L17 down c90 | 1% / 0 | unexplained (best R2 0.35) | off (on 0) | - | same |
| L17 down c96 | 0 / 39% | off (on 0) | **tens(a,b)** (R2 0.80): a//10 in {0} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {5,6,7,8,9,10}; a//10 in {5} -> b//10 in {8} | - | a: mod100 +5% |
| L17 down c107 | 24% / 15% | **tens(a,b)** (R2 0.81): a//10 in {0,1} -> b//10 in {2,3,4,5}; a//10 in {2} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {4,5}; a//10 in {7} -> b//10 in {4,5,6}; a//10 in {8,9} -> b//10 in {3,4,5,6}; a//10 in {10} -> b//10 in {2,3,4,5,6} | **tens(a,b)** (R2 0.76): a//10 in {6} -> b//10 in {4,5}; a//10 in {7,8} -> b//10 in {3,4,5,6}; a//10 in {9,10} -> b//10 in {3,4,5,6,7} | res: mod100 +2% | - |
| L17 down c115 | 0 / 0 | off (on 0) | same | - | same |
| L17 down c116 | 25% / 10% | **units(a,b)** (R2 0.90): a%10 in {0} -> b%10 in {1,2,7}; a%10 in {1,6} -> b%10 in {1,2,6,7}; a%10 in {2,7} -> b%10 in {0,1,2,5,6,7}; a%10 in {5} -> b%10 in {2,7} | unexplained (best R2 0.34) | res: mod5 +7% | res: mod5 +3% |
| L17 down c118 | 10% / 4% | **units(a,b)** (R2 0.81): a%10 in {6} -> b%10 in {8}; a%10 in {7} -> b%10 in {6,7,8,9}; a%10 in {8} -> b%10 in {6,7,8}; a%10 in {9} -> b%10 in {6,7} | unexplained (best R2 0.36) | - | same |
| L17 down c119 | 28% / 9% | **tens(a,b)** (R2 0.60): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1,2} -> b//10 in {0,1,2,3,10}; a//10 in {3} -> b//10 in {0,3}; a//10 in {4} -> b//10 in {0,4}; a//10 in {5,6,7,8} -> b//10 in {0}; a//10 in {9} -> b//10 in {0,9}; a//10 in {10} -> b//10 in {0,10} | unexplained (best R2 0.41) | a: mod100 +2%, mod25 +2% | res: mod25 +3%, mod20 +3% |
| L17 down c122 | 19% / 0 | **tens(a,b)** (R2 0.72): a//10 in {2,3,4,7,8} -> b//10 in {6,7}; a//10 in {5,6} -> b//10 in {5,6,7} | off (on 0) | - | same |
| L17 down c136 | 0 / 11% | off (on 0) | unexplained (best R2 0.42) | - | b: mod20 +3% |
| L17 down c170 | 9% / 0 | **b** (R2 0.70): b in {5..11, 86, 88} | off (on 0) | - | same |
| L17 down c235 | 17% / 0 | **tens(a,b)** (R2 0.80): a//10 in {0,1,2} -> b//10 in {6,7,8,9,10}; a//10 in {3} -> b//10 in {6,7}; a//10 in {9} -> b//10 in {9,10}; a//10 in {10} -> b//10 in {9} | off (on 0) | res: mod100 +2% | - |
| L17 down c503 | 0 / 13% | off (on 0) | **tens(a,b)** (R2 0.63): a//10 in {0,1} -> b//10 in {4,5,6,7,8,9}; a//10 in {2} -> b//10 in {9}; a//10 in {10} -> b//10 in {0} | - | same |

</details>

<details><summary>gate: 18</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L17 gate c5 | 99% / 0 | always | off (on 0) | (reads) | same |
| L17 gate c11 | 42% / 12% | **b** (R2 0.76): b in {57..100} | unexplained (best R2 0.44) | (reads) | same |
| L17 gate c13 | 0 / 25% | off (on 0) | **tens(a,b)** (R2 0.62): a//10 in {3,4} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {6,7,8}; a//10 in {6,10} -> b//10 in {6,7,8,9}; a//10 in {7,8} -> b//10 in {6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10} | (reads) | same |
| L17 gate c22 | 26% / 26% | **units(a,b)** (R2 0.95): a%10 in {1,3,5,7,9} -> b%10 in {0,2,4,6,8} | **units(a,b)** (R2 0.79): a%10 in {1,3,5,7,9} -> b%10 in {0,2,4,6,8} | (reads) | same |
| L17 gate c23 | 7% / 5% | **tens(a,b)** (R2 0.57): a//10 in {2,3,8} -> b//10 in {1}; a//10 in {7} -> b//10 in {1,6,9,10} | unexplained (best R2 0.36) | (reads) | same |
| L17 gate c34 | 35% / 24% | **units(a,b)** (R2 0.88): a%10 in {0} -> b%10 in {5,6}; a%10 in {5,6} -> b%10 in {0,1,2,3,5,6,7,8,9}; a%10 in {7} -> b%10 in {0,1,5,6,7,8,9}; a%10 in {8} -> b%10 in {5,6,7,8}; a%10 in {9} -> b%10 in {5,6,7} | unexplained (best R2 0.45) | (reads) | same |
| L17 gate c37 | 29% / 20% | **units(a,b)** (R2 0.83): a%10 in {0,1} -> b%10 in {0,1,2,8,9}; a%10 in {2,3,4,8,9} -> b%10 in {0,1,8,9} | **units(a,b)** (R2 0.52): a%10 in {0,1} -> b%10 in {0,1,2,9}; a%10 in {2} -> b%10 in {0,1,2}; a%10 in {8,9} -> b%10 in {0,1,9} | (reads) | same |
| L17 gate c39 | 38% / 26% | **units(a,b)** (R2 0.92): a%10 in {0,1,2} -> b%10 in {6,7,8}; a%10 in {3,4,5,6,7,8} -> b%10 in {6,7,8,9}; a%10 in {9} -> b%10 in {5,6,7,8,9} | **b** (R2 0.56): b in {1..4, 11..14, 21..24, 31..34, 41..44, 52..54, 62..63, 73, 82..83} [coarser: b mod 10 in {1..4}, R2 0.87] | (reads) | same |
| L17 gate c40 | 21% / 2% | **tens(a,b)** (R2 0.67): a//10 in {0,3,4,5,9,10} -> b//10 in {2,3,4}; a//10 in {1,2} -> b//10 in {3,4}; a//10 in {6,8} -> b//10 in {3} | unexplained (best R2 0.43) | (reads) | same |
| L17 gate c47 | 8% / 7% | unexplained (best R2 0.23) | unexplained (best R2 0.38) | (reads) | same |
| L17 gate c49 | 34% / 18% | **units(a,b)** (R2 0.91): a%10 in {5} -> b%10 in {0,1,2,3,6,7}; a%10 in {6,7} -> b%10 in {0,1,2,3,5,6,7,8,9}; a%10 in {8} -> b%10 in {0,1,2,5,6,7,8,9}; a%10 in {9} -> b%10 in {5,6} | unexplained (best R2 0.47) | (reads) | same |
| L17 gate c52 | 40% / 22% | **units(a,b)** (R2 0.88): a%10 in {0,1,3,7,8,9} -> b%10 in {1,2,3,4}; a%10 in {2} -> b%10 in {0,1,2,3,4}; a%10 in {4} -> b%10 in {0,1,2,3,4,9}; a%10 in {5} -> b%10 in {3,4}; a%10 in {6} -> b%10 in {2,3,4} | unexplained (best R2 0.28) | (reads) | same |
| L17 gate c67 | 25% / 18% | **units(a,b)** (R2 0.99): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} | **units(a,b)** (R2 0.62): a%10 in {1,5,7,9} -> b%10 in {1,3,5,7,9}; a%10 in {3} -> b%10 in {1,3,7,9} | (reads) | same |
| L17 gate c86 | 26% / 10% | **units(a,b)** (R2 0.85): a%10 in {5} -> b%10 in {2,3,4}; a%10 in {6} -> b%10 in {1,2,3,4}; a%10 in {7} -> b%10 in {0,1,2,3,4,9}; a%10 in {8} -> b%10 in {0,1,2,3,4,8,9}; a%10 in {9} -> b%10 in {0,1,2,3,9} | unexplained (best R2 0.37) | (reads) | same |
| L17 gate c89 | 1% / 99% | unexplained (best R2 0.08) | always | (reads) | same |
| L17 gate c107 | 36% / 34% | **b** (R2 0.75): b in {25..60, 64} | unexplained (best R2 0.50) | (reads) | same |
| L17 gate c116 | 25% / 11% | **units(a,b)** (R2 0.91): a%10 in {0} -> b%10 in {1,2,7}; a%10 in {1} -> b%10 in {0,1,2,6,7}; a%10 in {2,7} -> b%10 in {0,1,2,5,6,7}; a%10 in {5} -> b%10 in {2,7}; a%10 in {6} -> b%10 in {1,2,6,7} | unexplained (best R2 0.34) | (reads) | same |
| L17 gate c504 | 0 / 14% | off (on 0) | **tens(a,b)** (R2 0.77): a//10 in {0} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {8,9} | (reads) | same |

</details>

<details><summary>up: 29</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L17 up c5 | 0 / 4% | off (on 0) | unexplained (best R2 0.35) | (reads) | same |
| L17 up c9 | 100% / 100% | always | same | (reads) | same |
| L17 up c10 | 64% / 40% | **b%50** (R2 0.85): b mod 50 in {11..42} | **tens(a,b)** (R2 0.65): a//10 in {1} -> b//10 in {2}; a//10 in {2} -> b//10 in {1,2,3,7}; a//10 in {3,4,5,6,7,8,9,10} -> b//10 in {1,2,3,6,7,8} | (reads) | same |
| L17 up c11 | 57% / 31% | **tens(a,b)** (R2 0.78): a//10 in {0} -> b//10 in {8}; a//10 in {2} -> b//10 in {10}; a//10 in {3} -> b//10 in {1,9,10}; a//10 in {4} -> b//10 in {0,1,2,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,6,7,8,9,10}; a//10 in {6} -> b//10 in {0,1,2,4,5,6,7,8,9,10}; a//10 in {7,8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {9,10} -> b//10 in {1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.69): a//10 in {2,3} -> b//10 in {2}; a//10 in {4} -> b//10 in {0,10}; a//10 in {5} -> b//10 in {0,1,2,8,9,10}; a//10 in {6} -> b//10 in {0,1,2,3,9,10}; a//10 in {7,8} -> b//10 in {0,1,2,3,4,10}; a//10 in {9} -> b//10 in {0,1,2,3,4}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,10} | (reads) | same |
| L17 up c13 | 0 / 55% | off (on 0) | **tens(a,b)** (R2 0.64): a//10 in {1} -> b//10 in {2,3}; a//10 in {2} -> b//10 in {1,2,3,4,5,6,7}; a//10 in {3} -> b//10 in {1,2,3,4,5,6,7,8,9}; a//10 in {4} -> b//10 in {3,4,5,6,7,8}; a//10 in {5,6,7,8} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10}; a//10 in {10} -> b//10 in {1,2,3,4,5,6,7,8,9,10} | (reads) | same |
| L17 up c15 | 2% / 0 | unexplained (best R2 0.46) | off (on 0) | (reads) | same |
| L17 up c23 | 55% / 13% | **b** (R2 0.63): b in {3, 5, 27..58, 75, 77..100} [coarser: b mod 50 in {0..5, 27..49}, R2 0.96] | **tens(a,b)** (R2 0.67): a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {1}; a//10 in {6} -> b//10 in {0,5}; a//10 in {7} -> b//10 in {0,5,6}; a//10 in {8,10} -> b//10 in {5,6} | (reads) | same |
| L17 up c33 | 63% / 24% | **tens(a,b)** (R2 0.69): a//10 in {1,5} -> b//10 in {6,7}; a//10 in {2,3,7,8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {5,6,7}; a//10 in {9} -> b//10 in {1,2,3,4,5,6,7,8,10}; a//10 in {10} -> b//10 in {1,2,5,6,7} | **tens(a,b)** (R2 0.58): a//10 in {2,3} -> b//10 in {1,2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {7} -> b//10 in {3,4,5,6,7,8}; a//10 in {8} -> b//10 in {0,3,4,5,6,7,8,9}; a//10 in {9,10} -> b//10 in {8} | (reads) | same |
| L17 up c34 | 64% / 17% | **units(a,b)** (R2 0.54): a%10 in {0} -> b%10 in {0,1,3,5,6,7,8,9}; a%10 in {1,5,6} -> b%10 in {0,1,5,6,7,8,9}; a%10 in {2} -> b%10 in {0,6,7,8,9}; a%10 in {3} -> b%10 in {0,6,8,9}; a%10 in {4} -> b%10 in {0,8,9}; a%10 in {7} -> b%10 in {0,5,6,7,8,9}; a%10 in {8} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {9} -> b%10 in {0,1,3,4,5,6,7,8,9} | unexplained (best R2 0.26) | (reads) | same |
| L17 up c37 | 35% / 17% | **units(a,b)** (R2 0.83): a%10 in {0,1,2,3,4,9} -> b%10 in {5,6,7}; a%10 in {5} -> b%10 in {0,4,5,6,7}; a%10 in {6,7} -> b%10 in {4,5,6,7}; a%10 in {8} -> b%10 in {5,6} | unexplained (best R2 0.39) | (reads) | same |
| L17 up c39 | 20% / 15% | **units(a,b)** (R2 0.87): a%10 in {1} -> b%10 in {8}; a%10 in {2} -> b%10 in {6,7,8,9}; a%10 in {3} -> b%10 in {0,5,6,7,8,9}; a%10 in {4} -> b%10 in {5,6,7,8,9}; a%10 in {5} -> b%10 in {6,7,8} | **units(a,b)** (R2 0.61): a%10 in {1} -> b%10 in {2}; a%10 in {2} -> b%10 in {1,2,3}; a%10 in {3} -> b%10 in {0,1,2,3,4}; a%10 in {4} -> b%10 in {1,2,3,4}; a%10 in {5} -> b%10 in {2,4} | (reads) | same |
| L17 up c40 | 41% / 17% | **tens(a,b)** (R2 0.74): a//10 in {0} -> b//10 in {0,1,2,3,4,5,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,3,4,9,10}; a//10 in {2} -> b//10 in {0,1,2,3,9,10}; a//10 in {3} -> b//10 in {0,1,2,10}; a//10 in {4} -> b//10 in {0}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,9,10} | **tens(a,b)** (R2 0.67): a//10 in {0} -> b//10 in {0}; a//10 in {7} -> b//10 in {6}; a//10 in {8,9} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,4,5,6,7,8,9,10} | (reads) | same |
| L17 up c47 | 32% / 18% | **units(a,b)** (R2 0.95): a%10 in {0,4,5,9} -> b%10 in {0,3,4,5,8,9}; a%10 in {3,8} -> b%10 in {0,4,5,9} | **units(a,b)** (R2 0.54): a%10 in {0} -> b%10 in {0,1,2,5,6,7}; a%10 in {3} -> b%10 in {0}; a%10 in {4} -> b%10 in {0,1,5,6}; a%10 in {5} -> b%10 in {0,1,2,5,6}; a%10 in {8} -> b%10 in {5}; a%10 in {9} -> b%10 in {0,6} | (reads) | same |
| L17 up c49 | 41% / 26% | **units(a,b)** (R2 0.84): a%10 in {0} -> b%10 in {0,1}; a%10 in {1,3} -> b%10 in {0,1,2}; a%10 in {2,4} -> b%10 in {0,1,2,9}; a%10 in {5,6} -> b%10 in {0,1,2,3,9}; a%10 in {7,8} -> b%10 in {0,1,2,5,9}; a%10 in {9} -> b%10 in {0,1,9} | unexplained (best R2 0.47) | (reads) | same |
| L17 up c52 | 25% / 25% | **units(a,b)** (R2 0.85): a%10 in {5} -> b%10 in {0,1,5,6,7,8,9}; a%10 in {6} -> b%10 in {0,5,6,7,8,9}; a%10 in {7} -> b%10 in {5,6,7,8,9}; a%10 in {8} -> b%10 in {5,6,7,8}; a%10 in {9} -> b%10 in {5,6,7} | unexplained (best R2 0.32) | (reads) | same |
| L17 up c61 | 32% / 11% | **units(a,b)** (R2 0.94): a%10 in {0,5} -> b%10 in {3,4,8,9}; a%10 in {3} -> b%10 in {0,4,5,8,9}; a%10 in {4,8,9} -> b%10 in {0,3,4,5,8,9} | unexplained (best R2 0.29) | (reads) | same |
| L17 up c67 | 25% / 20% | **units(a,b)** (R2 0.99): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} | **units(a,b)** (R2 0.72): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} | (reads) | same |
| L17 up c70 | 5% / 2% | **units(a,b)** (R2 0.72): a%10 in {2,7} -> b%10 in {4,9} | unexplained (best R2 0.23) | (reads) | same |
| L17 up c86 | 50% / 13% | **units(a,b)** (R2 0.64): a%10 in {0} -> b%10 in {2,3,4}; a%10 in {1} -> b%10 in {1,2,3,4}; a%10 in {2} -> b%10 in {0,1,2,3,4}; a%10 in {3} -> b%10 in {0,1,2,3,4,9}; a%10 in {4,5,6,7} -> b%10 in {0,1,2,3,4,8,9}; a%10 in {8} -> b%10 in {0,1,2,3,9}; a%10 in {9} -> b%10 in {0,1,2} | unexplained (best R2 0.20) | (reads) | same |
| L17 up c107 | 0 / 17% | off (on 0) | **tens(a,b)** (R2 0.59): a//10 in {2} -> b//10 in {1}; a//10 in {3} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {1,2,3}; a//10 in {5} -> b//10 in {1,2,3,4,5}; a//10 in {6} -> b//10 in {1,3,4,5,6}; a//10 in {10} -> b//10 in {9} | (reads) | same |
| L17 up c116 | 0 / 14% | off (on 0) | **tens(a,b)** (R2 0.77): a//10 in {0} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {1} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {8,9} | (reads) | same |
| L17 up c118 | 34% / 16% | **units(a,b)** (R2 0.73): a%10 in {0,1} -> b%10 in {2,3,4}; a%10 in {2} -> b%10 in {0,1,2,3,4,9}; a%10 in {3} -> b%10 in {0,1,2,3,4,7,8,9}; a%10 in {4} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {5,9} -> b%10 in {4}; a%10 in {6} -> b%10 in {3,4} | unexplained (best R2 0.30) | (reads) | same |
| L17 up c136 | 16% / 1% | **b** (R2 0.84): b in {5..21} | unexplained (best R2 0.17) | (reads) | same |
| L17 up c170 | 6% / 1% | **b** (R2 0.91): b in {15..20} | unexplained (best R2 0.19) | (reads) | same |
| L17 up c173 | 6% / 0 | unexplained (best R2 0.36) | off (on 0) | (reads) | same |
| L17 up c193 | 32% / 11% | **tens(a,b)** (R2 0.63): a//10 in {1} -> b//10 in {3,4,5,8}; a//10 in {2} -> b//10 in {3,4,5,8,9}; a//10 in {3} -> b//10 in {3,4,8}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6,8} -> b//10 in {2,3,4,5}; a//10 in {7} -> b//10 in {2,3,4,5,9}; a//10 in {9} -> b//10 in {2,3,4}; a//10 in {10} -> b//10 in {3,4,5} | **tens(a,b)** (R2 0.62): a//10 in {2} -> b//10 in {0,1}; a//10 in {6} -> b//10 in {5}; a//10 in {7} -> b//10 in {0,5,6}; a//10 in {8,10} -> b//10 in {5,6,7} | (reads) | same |
| L17 up c454 | 7% / 4% | **units(a,b)** (R2 0.57): a%10 in {1} -> b%10 in {1,6,8,9}; a%10 in {6} -> b%10 in {1,6} | unexplained (best R2 0.32) | (reads) | same |
| L17 up c508 | 27% / 11% | **units(a,b)** (R2 0.89): a%10 in {6} -> b%10 in {3,4,7,8,9}; a%10 in {7} -> b%10 in {2,3,4,6,7,8,9}; a%10 in {8} -> b%10 in {1,2,3,4,6,7,8,9}; a%10 in {9} -> b%10 in {2,3,4,6,7,8} | unexplained (best R2 0.35) | (reads) | same |
| L17 up c636 | 23% / 14% | **units(a,b)** (R2 0.91): a%10 in {3} -> b%10 in {0,1,8,9}; a%10 in {4} -> b%10 in {0,1,6,7,8,9}; a%10 in {5} -> b%10 in {0,1,5,6,7,8,9}; a%10 in {6} -> b%10 in {0,5,6,7,8,9} | unexplained (best R2 0.50) | (reads) | same |

</details>

<details><summary>v: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L17 v c164 (kv4) | 0 / 52% | off (on 0) | unexplained (best R2 0.44) | (reads) | same |

</details>

</details>

<details><summary>layer 18: 204 components with main position `=`</summary>

<details><summary>down: 55</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L18 down c0 | 100% / 100% | always | same | - | same |
| L18 down c4 | 41% / 40% | **units(a,b)** (R2 0.74): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8}; a%10 in {1,5} -> b%10 in {3,5,7}; a%10 in {3,7,9} -> b%10 in {1,3,5,7,9} | **units(a,b)** (R2 0.52): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} | a: mod2 +54%; b: mod2 +48%; res: mod2 +61% | a: mod2 +48%; b: mod2 +48%; res: mod2 +64% |
| L18 down c12 | 78% / 59% | unexplained (best R2 0.46) | **tens(a,b)** (R2 0.51): a//10 in {0} -> b//10 in {0,1,4,5,9,10}; a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,8,9}; a//10 in {3} -> b//10 in {0,1,2,6,7,8}; a//10 in {4} -> b//10 in {0,1,2,3,4,6,7,8,9,10}; a//10 in {5,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {0,3,4,5,6,8,9,10}; a//10 in {7} -> b//10 in {0,4,5,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,5,6,7,8,10}; a//10 in {9} -> b//10 in {0,1,2,3,5,6,7,8,9,10} | b: mod50 +5%; res: mod50 +27% | b: mod50 +3%; res: mod50 +13% |
| L18 down c16 | 37% / 21% | **b%20** (R2 0.59): b mod 20 in {0..4, 15..19} | unexplained (best R2 0.27) | a: mod20 +10%, mod4 +4%; b: mod20 +8%, mod4 +5%; res: mod20 +24% | a: mod20 +4%; res: mod25 +2%, mod20 +12% |
| L18 down c18 | 35% / 28% | **b%20** (R2 0.60): b mod 20 in {0..4, 17..19} | unexplained (best R2 0.38) | a: mod20 +11%, mod4 +5%; res: mod20 +46% | a: mod20 +4%; res: mod20 +25% |
| L18 down c20 | 0 / 0 | off (on 0) | same | - | same |
| L18 down c21 | 70% / 36% | **tens(a,b)** (R2 0.57): a//10 in {0} -> b//10 in {1,2,3,4,6,7,8,9}; a//10 in {1,6} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,1,2,5,6,7,10}; a//10 in {3} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {4} -> b//10 in {2,3,4,8,9,10}; a//10 in {5} -> b//10 in {1,2,3,4,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,4,5,6,7,9,10}; a//10 in {8} -> b//10 in {0,1,4,5,9,10}; a//10 in {9} -> b//10 in {3,4,8,9,10}; a//10 in {10} -> b//10 in {2,3,4,7,8,9} | **tens(a,b)** (R2 0.57): a//10 in {0} -> b//10 in {1}; a//10 in {1} -> b//10 in {1,2,6,7}; a//10 in {2} -> b//10 in {2}; a//10 in {3} -> b//10 in {6}; a//10 in {4,9} -> b//10 in {1,6,7}; a//10 in {5} -> b//10 in {1,2,3,6,7}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,8,10}; a//10 in {7} -> b//10 in {3,4,7,8,10}; a//10 in {8} -> b//10 in {5}; a//10 in {10} -> b//10 in {0,1,2,5,6,7} | res: mod50 +18% | res: mod50 +5% |
| L18 down c22 | 65% / 37% | **res%50** (R2 0.65): res mod 50 in {0, 18..49} | **tens(a,b)** (R2 0.64): a//10 in {2} -> b//10 in {10}; a//10 in {3} -> b//10 in {0,1,5,6,9,10}; a//10 in {4} -> b//10 in {0,1,2,5,6,7,9,10}; a//10 in {5} -> b//10 in {0,1,2,6,7}; a//10 in {6} -> b//10 in {3}; a//10 in {7} -> b//10 in {3,4,5,10}; a//10 in {8} -> b//10 in {0,4,5,6,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,10}; a//10 in {10} -> b//10 in {0,1,2,5,6,7,10} | b: mod50 +2%; res: mod50 +15% | res: mod50 +5% |
| L18 down c23 | 52% / 35% | **res//10** (R2 0.71): (tens) res in {2..29, 31, 83..130, 179..200} [coarser: res mod 100 in {0..31, 81..99}, R2 0.99] | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {0,9,10}; a//10 in {1} -> b//10 in {0,1,10}; a//10 in {2} -> b//10 in {0,1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3,4,6}; a//10 in {5} -> b//10 in {3,4,5,6}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9}; a//10 in {9} -> b//10 in {0,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,8,9,10} | res: mod100 +22% | res: mod100 +6%, mod50 +5% |
| L18 down c24 | 57% / 51% | **units(a,b)** (R2 0.87): a%10 in {0,5} -> b%10 in {1,2,6,7}; a%10 in {1} -> b%10 in {0,1,5,6,9}; a%10 in {2,7} -> b%10 in {0,3,4,5,8,9}; a%10 in {3} -> b%10 in {0,2,3,4,7,8,9}; a%10 in {4,9} -> b%10 in {1,2,3,6,7,8}; a%10 in {6} -> b%10 in {0,1,4,5,6,9}; a%10 in {8} -> b%10 in {2,3,4,7,8,9} | unexplained (best R2 0.49) | res: mod5 +16% | res: mod5 +12% |
| L18 down c25 | 78% / 32% | **tens(a,b)** (R2 0.52): a//10 in {0} -> b//10 in {3,4,5,7,8,9,10}; a//10 in {1} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {2} -> b//10 in {1,2,3,4,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,2,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,4,5,6,8,9,10}; a//10 in {5} -> b//10 in {0,3,4,5,7,8,9,10}; a//10 in {6} -> b//10 in {0,2,3,4,7,8,9,10}; a//10 in {7,8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {0,1,3,4,5,6,8,9,10}; a//10 in {10} -> b//10 in {0,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.54): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {0,1}; a//10 in {2} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {0,5,10}; a//10 in {5} -> b//10 in {0,1,5,6,9,10}; a//10 in {6} -> b//10 in {0,1,2,5,6,7,9,10}; a//10 in {7} -> b//10 in {1,2,3,6,7}; a//10 in {8} -> b//10 in {4}; a//10 in {9} -> b//10 in {0,4,5,9}; a//10 in {10} -> b//10 in {0,1,4,5,6,9,10} | a: mod50 +3%; b: mod50 +2%; res: mod50 +14% | res: mod50 +5% |
| L18 down c26 | 69% / 42% | **units(a,b)** (R2 0.80): a%10 in {0,5} -> b%10 in {1,2,3,4,6,7,8,9}; a%10 in {1,6} -> b%10 in {0,1,2,3,5,6,7,8}; a%10 in {2,7} -> b%10 in {0,1,4,5,6,9}; a%10 in {3,8} -> b%10 in {0,3,4,5,8,9}; a%10 in {4,9} -> b%10 in {0,2,3,4,5,7,8,9} | **res** (R2 0.51): res in {-97, -48, -43..-42, -38..-37, -33, -28..-27, -23..-22, -18..-17, -13..-12, -8..-7, -3..-1, 1..3, 6..8, 11..13, 16..18, 21..23, 26..28, 31..33, 37..38, 42..43, 47..48, 52..53, 57..58, 62..63, 67..68, 72..73, 77..78, 82..83, 87..88, 92..93, 96..98} | res: mod5 +22% | res: mod5 +14% |
| L18 down c27 | 33% / 23% | **tens(a,b)** (R2 0.69): a//10 in {2} -> b//10 in {4,8,9,10}; a//10 in {3} -> b//10 in {3,4,7,8,9,10}; a//10 in {4} -> b//10 in {2,3,7,8,9}; a//10 in {5} -> b//10 in {8}; a//10 in {7} -> b//10 in {3,4,5,8,9,10}; a//10 in {8} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {9} -> b//10 in {2,3,4,7,8}; a//10 in {10} -> b//10 in {2,3,7,8} | **tens(a,b)** (R2 0.62): a//10 in {2} -> b//10 in {0,1,10}; a//10 in {3} -> b//10 in {0,1,2,6,7}; a//10 in {4} -> b//10 in {1,2,7}; a//10 in {7} -> b//10 in {0,5,6,10}; a//10 in {8} -> b//10 in {0,1,2,5,6,7,10}; a//10 in {9,10} -> b//10 in {1,2,6,7} | a: mod50 +3%; b: mod50 +2%; res: mod50 +9% | res: mod50 +4% |
| L18 down c30 | 26% / 9% | **tens(a,b)** (R2 0.51): a//10 in {2,4} -> b//10 in {5}; a//10 in {3} -> b//10 in {0,5,10}; a//10 in {6} -> b//10 in {0,4,5}; a//10 in {7,8} -> b//10 in {0,3,4,5,6,10}; a//10 in {9} -> b//10 in {3,4,5}; a//10 in {10} -> b//10 in {0,3,4,5} | **tens(a,b)** (R2 0.55): a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {4,5} | - | same |
| L18 down c32 | 46% / 30% | **units(a,b)** (R2 0.85): a%10 in {0} -> b%10 in {3,4,5,6}; a%10 in {1,3,9} -> b%10 in {3,4,5,6,7}; a%10 in {2} -> b%10 in {5,6,7}; a%10 in {4,5} -> b%10 in {2,3,4,5,6,7}; a%10 in {6} -> b%10 in {1,2,3,4,5,6}; a%10 in {7} -> b%10 in {1,2,3,4}; a%10 in {8} -> b%10 in {4,5,6} | unexplained (best R2 0.50) | res: mod10 +19% | res: mod10 +13% |
| L18 down c34 | 1% / 4% | **res** (R2 0.72): res in {2..7, 9, 13} | **res%100** (R2 0.50): res mod 100: no class above 0.5 (max 0.45) | - | same |
| L18 down c36 | 58% / 23% | **tens(a,b)** (R2 0.54): a//10 in {0} -> b//10 in {4,5,6,10}; a//10 in {1} -> b//10 in {4,5,10}; a//10 in {2,3} -> b//10 in {1,2,3,6,7}; a//10 in {4,9} -> b//10 in {0,1,2,5,6,7,10}; a//10 in {5} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {6} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {7,8} -> b//10 in {1,2,3,5,6,7,8}; a//10 in {10} -> b//10 in {0,1,2,4,5,6,7,10} | **tens(a,b)** (R2 0.55): a//10 in {3} -> b//10 in {2,3,8}; a//10 in {4} -> b//10 in {3,8}; a//10 in {5} -> b//10 in {0,4,5,8,9,10}; a//10 in {6} -> b//10 in {0,9,10}; a//10 in {8} -> b//10 in {3,4,7,8}; a//10 in {9} -> b//10 in {3,4,7,8,9}; a//10 in {10} -> b//10 in {0,3,4,5,7,8,9,10} | b: mod50 +2%, mod25 +3%; res: mod50 +8% | res: mod50 +5% |
| L18 down c48 | 12% / 4% | unexplained (best R2 0.41) | unexplained (best R2 0.32) | - | same |
| L18 down c53 | 4% / 0 | unexplained (best R2 0.48) | off (on 0) | - | same |
| L18 down c54 | 2% / 0 | **tens(a,b)** (R2 0.71): a//10 in {0,1,2,3,4,5,6,7,8,9} -> b//10 in {10}; a//10 in {10} -> b//10 in {5,6,7,10} | off (on 0) | - | same |
| L18 down c59 | 17% / 8% | **a%20** (R2 0.52): a mod 20 in {15..17} | unexplained (best R2 0.24) | b: mod20 +3%; res: mod20 +14% | res: mod20 +3% |
| L18 down c61 | 33% / 13% | **tens(a,b)** (R2 0.65): a//10 in {0} -> b//10 in {0,1,4,5,6,7,10}; a//10 in {1,6} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {2} -> b//10 in {0,4,5,10}; a//10 in {4} -> b//10 in {0,1,6}; a//10 in {5,10} -> b//10 in {0,1,5,6,10}; a//10 in {7} -> b//10 in {0,9,10}; a//10 in {9} -> b//10 in {1,6} | **tens(a,b)** (R2 0.61): a//10 in {5} -> b//10 in {3,4,5,8,9,10}; a//10 in {6} -> b//10 in {0,4,5,9,10}; a//10 in {10} -> b//10 in {3,4,8,9} | res: mod50 +6% | - |
| L18 down c62 | 25% / 28% | **units(a,b)** (R2 0.89): a%10 in {0} -> b%10 in {0,1,2,8,9}; a%10 in {1} -> b%10 in {0,1,7,8,9}; a%10 in {2} -> b%10 in {0,7,8,9}; a%10 in {7} -> b%10 in {1,2}; a%10 in {8} -> b%10 in {0,1,2}; a%10 in {9} -> b%10 in {0,1,2,9} | **units(a,b)** (R2 0.68): a%10 in {0} -> b%10 in {0,1,2,8,9}; a%10 in {1} -> b%10 in {0,1,2,3,8,9}; a%10 in {2} -> b%10 in {0,1,2,9}; a%10 in {7} -> b%10 in {8,9}; a%10 in {8} -> b%10 in {0,8,9}; a%10 in {9} -> b%10 in {0,1,7,8,9} | res: mod10 +9% | a: mod10 +2%; b: mod10 +2%; res: mod10 +13% |
| L18 down c63 | 11% / 20% | unexplained (best R2 0.38) | unexplained (best R2 0.49) | - | same |
| L18 down c65 | 36% / 29% | **units(a,b)** (R2 0.88): a%10 in {0,5} -> b%10 in {7,8,9}; a%10 in {1,6} -> b%10 in {6,7,8,9}; a%10 in {2,7} -> b%10 in {5,6,7,8}; a%10 in {3} -> b%10 in {4,5,6,7}; a%10 in {4} -> b%10 in {4,5,6}; a%10 in {8} -> b%10 in {5,6,7}; a%10 in {9} -> b%10 in {5,6,9} | unexplained (best R2 0.49) | res: mod10 +7% | res: mod10 +8% |
| L18 down c66 | 32% / 37% | **tens(a,b)** (R2 0.77): a//10 in {0} -> b//10 in {0,1,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,8,9,10}; a//10 in {2} -> b//10 in {9,10}; a//10 in {3,4,5,6} -> b//10 in {10}; a//10 in {7} -> b//10 in {0,9,10}; a//10 in {8} -> b//10 in {0,1,2,7,8,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.71): a//10 in {0,1} -> b//10 in {0,1,2,7,8,9,10}; a//10 in {2} -> b//10 in {8,9,10}; a//10 in {3,4,5} -> b//10 in {9}; a//10 in {6,7} -> b//10 in {8,9}; a//10 in {8} -> b//10 in {0,1,2,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,8,9,10} | - | same |
| L18 down c75 | 0 / 96% | off (on 0) | always | - | same |
| L18 down c79 | 7% / 0 | **tens(a,b)** (R2 0.84): a//10 in {2,3,4,5,6,7,8,9,10} -> b//10 in {0} | off (on 0) | - | same |
| L18 down c80 | 51% / 24% | **units(a,b)** (R2 0.82): a%10 in {0,5} -> b%10 in {0,1,5,6}; a%10 in {1,6} -> b%10 in {0,4,5,9}; a%10 in {2} -> b%10 in {3,4,7,8,9}; a%10 in {3,8} -> b%10 in {2,3,4,7,8,9}; a%10 in {4} -> b%10 in {1,2,6,7}; a%10 in {7} -> b%10 in {0,2,3,4,7,8,9}; a%10 in {9} -> b%10 in {1,2,6,7,8} | unexplained (best R2 0.46) | res: mod5 +7% | res: mod5 +8% |
| L18 down c88 | 30% / 19% | **units(a,b)** (R2 0.89): a%10 in {1} -> b%10 in {4,5,6}; a%10 in {2} -> b%10 in {3,4,5,6}; a%10 in {3} -> b%10 in {2,3,4,5,6}; a%10 in {4} -> b%10 in {1,2,3,4,5}; a%10 in {5} -> b%10 in {1,2,3,4,5,6,7}; a%10 in {6} -> b%10 in {1,5,6,7}; a%10 in {7} -> b%10 in {5,6} | unexplained (best R2 0.46) | res: mod10 +8% | res: mod10 +5% |
| L18 down c91 | 27% / 3% | **tens(a,b)** (R2 0.77): a//10 in {0} -> b//10 in {3,4}; a//10 in {1} -> b//10 in {2,3,4,5}; a//10 in {2} -> b//10 in {1,2,3,4,5,6}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6}; a//10 in {4} -> b//10 in {1,2,3,4,5}; a//10 in {5} -> b//10 in {2,3,4} | unexplained (best R2 0.22) | res: mod100 +2% | - |
| L18 down c98 | 34% / 27% | **units(a,b)** (R2 0.81): a%10 in {0,1} -> b%10 in {4,5,6,7}; a%10 in {2} -> b%10 in {0,4,5,6,8,9}; a%10 in {3} -> b%10 in {0,4,5,8,9}; a%10 in {4} -> b%10 in {7,8}; a%10 in {7} -> b%10 in {8,9}; a%10 in {8} -> b%10 in {6,7,8,9}; a%10 in {9} -> b%10 in {5,6,7,8} | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {2,3,4,5,6}; a%10 in {1} -> b%10 in {1,3,4,5,6}; a%10 in {2} -> b%10 in {0,1,2,4,5,6}; a%10 in {3} -> b%10 in {0,1,2,3}; a%10 in {4} -> b%10 in {1,2,3}; a%10 in {8} -> b%10 in {2,3,4}; a%10 in {9} -> b%10 in {2,3,4,5} | res: mod10 +7% | res: mod10 +8% |
| L18 down c99 | 1% / 0 | unexplained (best R2 0.27) | off (on 0) | - | same |
| L18 down c101 | 42% / 24% | **units(a,b)** (R2 0.85): a%10 in {1,6} -> b%10 in {4,8,9}; a%10 in {2,7} -> b%10 in {2,3,4,7,8,9}; a%10 in {3,8} -> b%10 in {1,2,3,4,6,7,8,9}; a%10 in {4,9} -> b%10 in {1,2,6,7,8} | unexplained (best R2 0.45) | res: mod5 +8% | res: mod5 +6% |
| L18 down c103 | 4% / 0 | **units(a,b)** (R2 0.75): a%10 in {2,7} -> b%10 in {2,7} | off (on 0) | - | same |
| L18 down c111 | 13% / 36% | **tens(a,b)** (R2 0.58): a//10 in {0} -> b//10 in {3}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {0,1,2,3}; a//10 in {4} -> b//10 in {0,1,2}; a//10 in {5} -> b//10 in {0} | **tens(a,b)** (R2 0.70): a//10 in {0} -> b//10 in {2,3,4,5,6}; a//10 in {1} -> b//10 in {3,4,5}; a//10 in {2} -> b//10 in {0,3,4,5}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,7,9,10}; a//10 in {4} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,5,6}; a//10 in {6} -> b//10 in {2} | - | same |
| L18 down c118 | 6% / 48% | **tens(a,b)** (R2 0.56): a//10 in {9} -> b//10 in {0,1,2,7,8,9}; a//10 in {10} -> b//10 in {1,2} | **tens(a,b)** (R2 0.59): a//10 in {0} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {5,6,7,8,9}; a//10 in {2} -> b//10 in {7,8,9,10}; a//10 in {3} -> b//10 in {8,9}; a//10 in {4} -> b//10 in {0,1,2,8,9}; a//10 in {5} -> b//10 in {0,1,2,9}; a//10 in {6} -> b//10 in {2}; a//10 in {7} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1,2,3,4,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,7,10} | - | a: mod50 +2% |
| L18 down c120 | 19% / 16% | **units(a,b)** (R2 0.85): a%10 in {0} -> b%10 in {3}; a%10 in {5} -> b%10 in {5,6,7}; a%10 in {6} -> b%10 in {4,5,6}; a%10 in {7} -> b%10 in {3,4,5,6}; a%10 in {8} -> b%10 in {2,3,4,5}; a%10 in {9} -> b%10 in {2,3,4} | **units(a,b)** (R2 0.51): a%10 in {0} -> b%10 in {7,8}; a%10 in {5} -> b%10 in {4,5}; a%10 in {6} -> b%10 in {4,5,6}; a%10 in {7} -> b%10 in {5,6,7}; a%10 in {8} -> b%10 in {5,6,7,8}; a%10 in {9} -> b%10 in {6,7,8} | res: mod10 +3% | res: mod10 +4% |
| L18 down c130 | 2% / 0 | unexplained (best R2 0.18) | off (on 0) | - | same |
| L18 down c132 | 34% / 0 | **tens(a,b)** (R2 0.89): a//10 in {3} -> b//10 in {9}; a//10 in {4} -> b//10 in {6,7,8,9,10}; a//10 in {5,6} -> b//10 in {5,6,7,8,9,10}; a//10 in {7,8,9,10} -> b//10 in {4,5,6,7,8,9,10} | off (on 0) | a: mod100 +2%; b: mod100 +2%; res: mod100 +3% | - |
| L18 down c142 | 54% / 21% | **units(a,b)** (R2 0.88): a%10 in {0,5} -> b%10 in {0,3,4,5,9}; a%10 in {1,6} -> b%10 in {2,3,4,7,8,9}; a%10 in {2,7} -> b%10 in {1,2,3,6,7,8}; a%10 in {3,8} -> b%10 in {1,2,6,7}; a%10 in {4} -> b%10 in {0,1,5,6}; a%10 in {9} -> b%10 in {0,1,4,5,6} | **res** (R2 0.52): res in {0, 3..5, 8..10, 13..15, 19..20, 23..25, 29..30, 34..35, 39, 44..45, 59, 69, 74..75, 79, 84..85, 89..90, 94..95, 99} | res: mod5 +8% | res: mod5 +4% |
| L18 down c147 | 16% / 29% | **tens(a,b)** (R2 0.80): a//10 in {0} -> b//10 in {0,1,2,3,10}; a//10 in {1} -> b//10 in {0,1,2,3,9,10}; a//10 in {2} -> b//10 in {0,1,2,9,10}; a//10 in {3} -> b//10 in {0,1}; a//10 in {10} -> b//10 in {1,10} | **tens(a,b)** (R2 0.82): a//10 in {0,4} -> b//10 in {0,1,2,3,4}; a//10 in {1} -> b//10 in {0,1,2,3,4,8,9,10}; a//10 in {2} -> b//10 in {0,1,2,3,4,9,10}; a//10 in {3} -> b//10 in {0,1,2,3,4,10}; a//10 in {10} -> b//10 in {7,8,9,10} | - | same |
| L18 down c167 | 0 / 2% | off (on 0) | unexplained (best R2 0.39) | - | same |
| L18 down c196 | 9% / 0 | **tens(a,b)** (R2 0.71): a//10 in {0,1} -> b//10 in {7,8}; a//10 in {2} -> b//10 in {6,7,8}; a//10 in {3} -> b//10 in {6,7} | off (on 0) | - | same |
| L18 down c213 | 1% / 20% | unexplained (best R2 0.21) | **tens(a,b)** (R2 0.58): a//10 in {0} -> b//10 in {6,7}; a//10 in {1} -> b//10 in {7}; a//10 in {2} -> b//10 in {7,8}; a//10 in {6} -> b//10 in {0,1,2}; a//10 in {7,8} -> b//10 in {0,1,2,3} | - | same |
| L18 down c230 | 1% / 30% | unexplained (best R2 0.07) | **tens(a,b)** (R2 0.55): a//10 in {4} -> b//10 in {7}; a//10 in {5} -> b//10 in {7,9,10}; a//10 in {6,7} -> b//10 in {0,6,7,8,9,10}; a//10 in {8} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | - | same |
| L18 down c245 | 0 / 7% | off (on 0) | **tens(a,b)** (R2 0.69): a//10 in {8,10} -> b//10 in {5,6,7,8}; a//10 in {9} -> b//10 in {6,7,8} | - | same |
| L18 down c261 | 6% / 5% | **units(a,b)** (R2 0.93): a%10 in {5} -> b%10 in {5,6,7}; a%10 in {6} -> b%10 in {5,6}; a%10 in {7} -> b%10 in {5} | unexplained (best R2 0.48) | - | same |
| L18 down c290 | 0 / 25% | off (on 0) | **a//10** (R2 0.69): (tens) a in {2..24} | - | a: mod100 +3%, mod50 +3% |
| L18 down c375 | 37% / 26% | **units(a,b)** (R2 0.87): a%10 in {0,5} -> b%10 in {2,3,4,7,8,9}; a%10 in {1,6} -> b%10 in {2,3,7,8}; a%10 in {3,8} -> b%10 in {0,5}; a%10 in {4,9} -> b%10 in {0,3,4,5,8,9} | unexplained (best R2 0.46) | res: mod5 +5% | res: mod5 +6% |
| L18 down c434 | 0 / 22% | off (on 0) | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {0,10}; a//10 in {1,4,6,7,8,9} -> b//10 in {0,1}; a//10 in {2,3} -> b//10 in {0}; a//10 in {5} -> b//10 in {0,1,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,9} | - | b: mod50 +2%, mod25 +2% |
| L18 down c474 | 18% / 9% | **units(a,b)** (R2 0.92): a%10 in {0} -> b%10 in {2,3,4}; a%10 in {1} -> b%10 in {1,2,3,4}; a%10 in {2} -> b%10 in {1,2,3}; a%10 in {3} -> b%10 in {0,1,2}; a%10 in {4} -> b%10 in {0,1,9}; a%10 in {9} -> b%10 in {4,5} | unexplained (best R2 0.38) | res: mod10 +3% | - |
| L18 down c518 | 2% / 0 | unexplained (best R2 0.32) | off (on 0) | - | same |
| L18 down c625 | 15% / 1% | **tens(a,b)** (R2 0.60): a//10 in {0,1,2} -> b//10 in {10}; a//10 in {3,4,5,6,7} -> b//10 in {9,10}; a//10 in {8} -> b//10 in {8,9,10}; a//10 in {9,10} -> b//10 in {4,5,6,7,8,9,10} | unexplained (best R2 0.49) | - | same |
| L18 down c677 | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>gate: 45</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L18 gate c4 | 26% / 40% | **units(a,b)** (R2 0.94): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} | **units(a,b)** (R2 0.58): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} | (reads) | same |
| L18 gate c12 | 56% / 38% | **a%50** (R2 0.85): a mod 50 in {14..41} | **a** (R2 0.63): a in {20..36, 66..89} [coarser: a mod 50 in {17..39}, R2 0.88] | (reads) | same |
| L18 gate c16 | 38% / 19% | **b%20** (R2 0.65): b mod 20 in {0..4, 15..19} | unexplained (best R2 0.27) | (reads) | same |
| L18 gate c18 | 41% / 32% | **b%20** (R2 0.70): b mod 20 in {0..4, 15..19} | unexplained (best R2 0.47) | (reads) | same |
| L18 gate c21 | 57% / 38% | **a%50** (R2 0.76): a mod 50 in {0..25, 46..49} | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {0,1,10}; a//10 in {1} -> b//10 in {0,1,2,5,6}; a//10 in {2} -> b//10 in {2}; a//10 in {3} -> b//10 in {6}; a//10 in {4,7} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {0,1,2,3,4,5,6,7,10}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {7}; a//10 in {10} -> b//10 in {0,1,2,3,5,6,7,8,10} | (reads) | same |
| L18 gate c22 | 55% / 47% | **a%50** (R2 0.66): a mod 50 in {0..11, 34..49} | **a** (R2 0.68): a in {1, 34..61, 81..100} | (reads) | same |
| L18 gate c23 | 57% / 49% | **tens(a,b)** (R2 0.76): a//10 in {0,1} -> b//10 in {3,4,5,6}; a//10 in {2} -> b//10 in {4,5}; a//10 in {3} -> b//10 in {0,1,4,5,6,10}; a//10 in {4,5} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {0,1,3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {4,5,6,7}; a//10 in {9,10} -> b//10 in {3,4,5,6,7} | **tens(a,b)** (R2 0.70): a//10 in {0,1,2} -> b//10 in {4,5}; a//10 in {3} -> b//10 in {0,3,4,5,6,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,3,4,5,6,8,9,10}; a//10 in {6} -> b//10 in {0,1,2,4,5,9,10}; a//10 in {7} -> b//10 in {1,2,3,4,5}; a//10 in {8} -> b//10 in {3,4,5}; a//10 in {9,10} -> b//10 in {3,4,5,6} | (reads) | same |
| L18 gate c24 | 39% / 23% | **units(a,b)** (R2 0.90): a%10 in {1,6} -> b%10 in {1,2,3,6,7,8}; a%10 in {2} -> b%10 in {1,2,3,5,6,7,8}; a%10 in {3,8} -> b%10 in {1,2,6,7}; a%10 in {5} -> b%10 in {2,7}; a%10 in {7} -> b%10 in {0,1,2,3,5,6,7,8} | unexplained (best R2 0.48) | (reads) | same |
| L18 gate c25 | 55% / 19% | **tens(a,b)** (R2 0.65): a//10 in {0} -> b//10 in {2,3,4,7,8,9}; a//10 in {1,6} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2,7} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {8}; a//10 in {4} -> b//10 in {9}; a//10 in {5} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {9} -> b//10 in {10}; a//10 in {10} -> b//10 in {3,4,7,8,9,10} | **tens(a,b)** (R2 0.55): a//10 in {1} -> b//10 in {0,1}; a//10 in {2} -> b//10 in {1,2}; a//10 in {5} -> b//10 in {1,6}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,10}; a//10 in {7} -> b//10 in {5,6,7}; a//10 in {10} -> b//10 in {1,5,6,7} | (reads) | same |
| L18 gate c26 | 61% / 25% | **units(a,b)** (R2 0.78): a%10 in {0,5} -> b%10 in {2,3,7,8,9}; a%10 in {2,7} -> b%10 in {2,3,4,7,8,9}; a%10 in {3,8,9} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {4} -> b%10 in {0,1,2,3,4,6,7,8,9} | unexplained (best R2 0.38) | (reads) | same |
| L18 gate c27 | 44% / 13% | **tens(a,b)** (R2 0.69): a//10 in {0,1} -> b//10 in {8}; a//10 in {2} -> b//10 in {3,4,7,8,9,10}; a//10 in {3} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {4,9} -> b//10 in {2,3,4,7,8,9}; a//10 in {5,6} -> b//10 in {3,8,9}; a//10 in {7} -> b//10 in {3,4,5,7,8,9,10}; a//10 in {8} -> b//10 in {2,3,4,5,7,8,9,10}; a//10 in {10} -> b//10 in {2,3,4,7,8} | **tens(a,b)** (R2 0.60): a//10 in {2,3} -> b//10 in {1}; a//10 in {7} -> b//10 in {6}; a//10 in {8} -> b//10 in {0,1,2,5,6,7}; a//10 in {9} -> b//10 in {6,7} | (reads) | same |
| L18 gate c32 | 57% / 16% | **units(a,b)** (R2 0.85): a%10 in {0,3,9} -> b%10 in {2,3,4,5,6,7}; a%10 in {1,2,8} -> b%10 in {3,4,5,6,7}; a%10 in {4,5,6} -> b%10 in {1,2,3,4,5,6,7}; a%10 in {7} -> b%10 in {3,4,5,6} | unexplained (best R2 0.39) | (reads) | same |
| L18 gate c36 | 45% / 25% | **tens(a,b)** (R2 0.66): a//10 in {0,6,7} -> b//10 in {1,2,6,7}; a//10 in {1} -> b//10 in {6,7}; a//10 in {2} -> b//10 in {1,6,7}; a//10 in {3} -> b//10 in {1,2,6,7,8}; a//10 in {4,9,10} -> b//10 in {1,2,5,6,7,8,10}; a//10 in {5} -> b//10 in {1,2,5,6,7}; a//10 in {8} -> b//10 in {1,2,3,5,6,7,8} | **tens(a,b)** (R2 0.59): a//10 in {3,4} -> b//10 in {2,3,7,8}; a//10 in {5} -> b//10 in {3,4,7,8}; a//10 in {6,7} -> b//10 in {8}; a//10 in {8} -> b//10 in {2,3,7,8,9}; a//10 in {9} -> b//10 in {3,7,8,9,10}; a//10 in {10} -> b//10 in {2,3,4,5,6,7,8,9,10} | (reads) | same |
| L18 gate c59 | 16% / 6% | **a%20** (R2 0.58): a mod 20 in {15..18} | unexplained (best R2 0.25) | (reads) | same |
| L18 gate c61 | 51% / 26% | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1,3} -> b//10 in {0,1,5,6,9,10}; a//10 in {2} -> b//10 in {0,1,10}; a//10 in {4,5} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {6,8} -> b//10 in {0,1,5,6,10}; a//10 in {7} -> b//10 in {0,10}; a//10 in {9,10} -> b//10 in {0,1,3,4,5,6,9,10} | **tens(a,b)** (R2 0.60): a//10 in {0} -> b//10 in {0,9,10}; a//10 in {3} -> b//10 in {0,9}; a//10 in {4,5} -> b//10 in {0,3,4,5,9,10}; a//10 in {6} -> b//10 in {4,9,10}; a//10 in {8} -> b//10 in {0,4,5,9}; a//10 in {9} -> b//10 in {0,4,5,8,9,10}; a//10 in {10} -> b//10 in {0,3,4,5,6,8,9,10} | (reads) | same |
| L18 gate c62 | 32% / 21% | **units(a,b)** (R2 0.88): a%10 in {0,1} -> b%10 in {2,3,4}; a%10 in {2} -> b%10 in {1,2,3,4}; a%10 in {3} -> b%10 in {0,1,2,3,4}; a%10 in {4} -> b%10 in {0,1,2,3,4,5,9}; a%10 in {5} -> b%10 in {0,1,2,3,4,5}; a%10 in {6} -> b%10 in {3,4}; a%10 in {9} -> b%10 in {4} | unexplained (best R2 0.42) | (reads) | same |
| L18 gate c65 | 28% / 16% | **units(a,b)** (R2 0.88): a%10 in {0} -> b%10 in {6,7}; a%10 in {1,2,7} -> b%10 in {5,6,7,8}; a%10 in {3,8} -> b%10 in {5,6,7}; a%10 in {5,6} -> b%10 in {6,7,8}; a%10 in {9} -> b%10 in {5,6} | unexplained (best R2 0.37) | (reads) | same |
| L18 gate c66 | 31% / 35% | **tens(a,b)** (R2 0.80): a//10 in {0} -> b//10 in {0,1,8,9,10}; a//10 in {1} -> b//10 in {0,1,7,8,9,10}; a//10 in {2} -> b//10 in {8,9,10}; a//10 in {3,4,5,6,7} -> b//10 in {9,10}; a//10 in {8} -> b//10 in {0,1,2,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.75): a//10 in {0} -> b//10 in {0,1,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,7,8,9,10}; a//10 in {2,5} -> b//10 in {9,10}; a//10 in {3,4} -> b//10 in {9}; a//10 in {6,7} -> b//10 in {8,9}; a//10 in {8} -> b//10 in {0,1,2,8,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | (reads) | same |
| L18 gate c80 | 26% / 16% | **units(a,b)** (R2 0.75): a%10 in {0} -> b%10 in {0,1,5,6,9}; a%10 in {1,6} -> b%10 in {0,4,5,9}; a%10 in {2} -> b%10 in {9}; a%10 in {5} -> b%10 in {0,1,5,6}; a%10 in {7} -> b%10 in {0,4,9}; a%10 in {9} -> b%10 in {1,6} | **units(a,b)** (R2 0.65): a%10 in {0,5} -> b%10 in {0,4,5,9}; a%10 in {1,6} -> b%10 in {0,1,5,6} | (reads) | same |
| L18 gate c88 | 44% / 37% | **units(a,b)** (R2 0.90): a%10 in {0,1,2,6} -> b%10 in {3,4,5,6}; a%10 in {3} -> b%10 in {2,3,4,5,6,7}; a%10 in {4,5} -> b%10 in {1,2,3,4,5,6,7}; a%10 in {7,8} -> b%10 in {5}; a%10 in {9} -> b%10 in {4,5,6} | **units(a,b)** (R2 0.53): a%10 in {0,1,2,3,6} -> b%10 in {3,4,5,6,7}; a%10 in {4,5} -> b%10 in {3,4,5,6,7,9}; a%10 in {7} -> b%10 in {5,6}; a%10 in {8} -> b%10 in {5}; a%10 in {9} -> b%10 in {4,5,6} | (reads) | same |
| L18 gate c91 | 30% / 13% | **tens(a,b)** (R2 0.70): a//10 in {0,10} -> b//10 in {3}; a//10 in {2} -> b//10 in {0,3,4,9}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,3,4,8,9}; a//10 in {7,8,9} -> b//10 in {3,4} | **tens(a,b)** (R2 0.55): a//10 in {2} -> b//10 in {0}; a//10 in {3} -> b//10 in {0,1,2,3,6,10}; a//10 in {4} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {5,6}; a//10 in {9,10} -> b//10 in {6} | (reads) | same |
| L18 gate c98 | 27% / 23% | **units(a,b)** (R2 0.86): a%10 in {0} -> b%10 in {5,6,7}; a%10 in {1} -> b%10 in {0,4,5,6,7,9}; a%10 in {2} -> b%10 in {0,4,5,6,7,8,9}; a%10 in {3} -> b%10 in {0,5,6,7,8,9}; a%10 in {4} -> b%10 in {6,7,8}; a%10 in {9} -> b%10 in {6,7} | **units(a,b)** (R2 0.56): a%10 in {0} -> b%10 in {3,4,5}; a%10 in {1,2} -> b%10 in {0,1,2,3,4,5,6}; a%10 in {3} -> b%10 in {1,2,3,4}; a%10 in {4} -> b%10 in {2,3,4}; a%10 in {9} -> b%10 in {3,4} | (reads) | same |
| L18 gate c101 | 55% / 30% | **units(a,b)** (R2 0.91): a%10 in {0,5} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {1,6} -> b%10 in {0,1,2,3,5,6,7,8}; a%10 in {2,7} -> b%10 in {0,1,5,6}; a%10 in {3} -> b%10 in {0,1,5}; a%10 in {4,8,9} -> b%10 in {0,5} | unexplained (best R2 0.50) | (reads) | same |
| L18 gate c111 | 17% / 11% | **tens(a,b)** (R2 0.74): a//10 in {0} -> b//10 in {3,4}; a//10 in {2,3,4} -> b//10 in {0,1,2,3,4} | **tens(a,b)** (R2 0.50): a//10 in {0} -> b//10 in {4}; a//10 in {3} -> b//10 in {0,1,3,4,10}; a//10 in {4} -> b//10 in {0,1,2,4,10} | (reads) | same |
| L18 gate c120 | 34% / 32% | **units(a,b)** (R2 0.81): a%10 in {0,1} -> b%10 in {3,4}; a%10 in {3,4} -> b%10 in {3,4,5,6}; a%10 in {5} -> b%10 in {3,4,5}; a%10 in {6,7,8,9} -> b%10 in {2,3,4,5} | **units(a,b)** (R2 0.56): a%10 in {0,3,4,7,8,9} -> b%10 in {5,6,7,8}; a%10 in {1} -> b%10 in {6,7,8}; a%10 in {2} -> b%10 in {7}; a%10 in {5,6} -> b%10 in {4,5,6,7,8} | (reads) | same |
| L18 gate c130 | 8% / 4% | **units(a,b)** (R2 0.83): a%10 in {0} -> b%10 in {1,2}; a%10 in {1} -> b%10 in {0,1,2}; a%10 in {2} -> b%10 in {0,1} | unexplained (best R2 0.34) | (reads) | same |
| L18 gate c142 | 32% / 23% | **units(a,b)** (R2 0.88): a%10 in {1,6} -> b%10 in {2,7}; a%10 in {2,7} -> b%10 in {1,2,3,4,6,7,8,9}; a%10 in {3} -> b%10 in {2,3,4,7,8,9}; a%10 in {8} -> b%10 in {2,3,7,8,9} | unexplained (best R2 0.46) | (reads) | same |
| L18 gate c147 | 19% / 27% | **tens(a,b)** (R2 0.82): a//10 in {0,3} -> b//10 in {0,1,2,3}; a//10 in {1} -> b//10 in {0,1,2,3,10}; a//10 in {2} -> b//10 in {0,1,2,3,9,10}; a//10 in {4} -> b//10 in {0} | **tens(a,b)** (R2 0.84): a//10 in {0,4} -> b//10 in {0,1,2,3,4}; a//10 in {1,2} -> b//10 in {0,1,2,3,4,9,10}; a//10 in {3} -> b//10 in {0,1,2,3,4,10} | (reads) | same |
| L18 gate c153 | 22% / 18% | **a//10** (R2 0.81): (tens) a in {59..79} | **tens(a,b)** (R2 0.76): a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,10}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,6,7} | (reads) | same |
| L18 gate c165 | 0 / 0 | off (on 0) | same | (reads) | same |
| L18 gate c261 | 26% / 16% | **units(a,b)** (R2 0.84): a%10 in {0} -> b%10 in {6}; a%10 in {1} -> b%10 in {5,6}; a%10 in {2} -> b%10 in {4,5}; a%10 in {3} -> b%10 in {4}; a%10 in {4} -> b%10 in {1,2,3,7}; a%10 in {5} -> b%10 in {0,1,2,5,6,7}; a%10 in {6} -> b%10 in {0,1,5,6,7,9}; a%10 in {7} -> b%10 in {0,5,6} | unexplained (best R2 0.40) | (reads) | same |
| L18 gate c267 | 25% / 11% | unexplained (best R2 0.33) | unexplained (best R2 0.21) | (reads) | same |
| L18 gate c299 | 46% / 40% | **res%2** (R2 0.80): res mod 2 in {0} | **units(a,b)** (R2 0.55): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} | (reads) | same |
| L18 gate c306 | 100% / 100% | always | same | (reads) | same |
| L18 gate c375 | 28% / 19% | **units(a,b)** (R2 0.85): a%10 in {0,5} -> b%10 in {0,2,3,4,5,7,8,9}; a%10 in {3,8} -> b%10 in {0}; a%10 in {4,9} -> b%10 in {0,4,5,9} | **units(a,b)** (R2 0.62): a%10 in {0} -> b%10 in {0,1,2,3,5,6,7,8,9}; a%10 in {5} -> b%10 in {0,1,2,5,6,7}; a%10 in {9} -> b%10 in {0} | (reads) | same |
| L18 gate c400 | 3% / 1% | unexplained (best R2 0.31) | unexplained (best R2 0.27) | (reads) | same |
| L18 gate c474 | 29% / 21% | **units(a,b)** (R2 0.92): a%10 in {0,5,6} -> b%10 in {8,9}; a%10 in {1,2,7} -> b%10 in {7,8,9}; a%10 in {3} -> b%10 in {7,8}; a%10 in {8} -> b%10 in {6,7,8,9}; a%10 in {9} -> b%10 in {0,1,5,6,7,8,9} | **units(a,b)** (R2 0.53): a%10 in {0} -> b%10 in {0,1,2}; a%10 in {1,9} -> b%10 in {0,1,2,3}; a%10 in {2,7,8} -> b%10 in {1,2,3}; a%10 in {3} -> b%10 in {2,3}; a%10 in {5,6} -> b%10 in {1} | (reads) | same |
| L18 gate c483 | 2% / 0 | unexplained (best R2 0.43) | off (on 0) | (reads) | same |
| L18 gate c543 | 16% / 6% | **tens(a,b)** (R2 0.65): a//10 in {0,1,2} -> b//10 in {10}; a//10 in {3,6,7} -> b//10 in {9,10}; a//10 in {4,5} -> b//10 in {8,9,10}; a//10 in {8} -> b//10 in {4,9,10}; a//10 in {9,10} -> b//10 in {4,8,9,10} | unexplained (best R2 0.46) | (reads) | same |
| L18 gate c544 | 3% / 2% | unexplained (best R2 0.43) | unexplained (best R2 0.19) | (reads) | same |
| L18 gate c556 | 7% / 3% | **units(a,b)** (R2 0.84): a%10 in {0} -> b%10 in {2}; a%10 in {1} -> b%10 in {1,2,3}; a%10 in {2} -> b%10 in {1,2}; a%10 in {3} -> b%10 in {1} | unexplained (best R2 0.27) | (reads) | same |
| L18 gate c664 | 21% / 0 | **units(a,b)** (R2 0.79): a%10 in {0} -> b%10 in {2,4,6,8}; a%10 in {2,4,6,8} -> b%10 in {0,2,4,6,8} | off (on 0) | (reads) | same |
| L18 gate c677 | 13% / 7% | **tens(a,b)** (R2 0.60): a//10 in {3} -> b//10 in {1,2,3,7,8}; a//10 in {4} -> b//10 in {1,2,3,6,7}; a//10 in {8} -> b//10 in {2,7}; a//10 in {9} -> b//10 in {2} | **tens(a,b)** (R2 0.54): a//10 in {3} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {3}; a//10 in {8} -> b//10 in {6,7}; a//10 in {9} -> b//10 in {7} | (reads) | same |
| L18 gate c702 | 7% / 2% | unexplained (best R2 0.43) | unexplained (best R2 0.15) | (reads) | same |
| L18 gate c801 | 0 / 0 | off (on 0) | same | (reads) | same |

</details>

<details><summary>o: 32</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L18 o c66 (H30) | 30% / 36% | **tens(a,b)** (R2 0.77): a//10 in {0} -> b//10 in {0,1,2,3,4}; a//10 in {1} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,1,2,3,4,10}; a//10 in {3,4} -> b//10 in {0,1,2}; a//10 in {5} -> b//10 in {1,2}; a//10 in {10} -> b//10 in {1} | **tens(a,b)** (R2 0.82): a//10 in {0,3} -> b//10 in {0,1,2,3,4}; a//10 in {1} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,1,2,3,4,5,6,7,10}; a//10 in {4} -> b//10 in {0,1,2}; a//10 in {5} -> b//10 in {1,2}; a//10 in {6,7,8} -> b//10 in {1} | - | same |
| L18 o c73 (H30) | 34% / 33% | **tens(a,b)** (R2 0.66): a//10 in {0,1,2} -> b//10 in {5,6,7}; a//10 in {3,4,8,9,10} -> b//10 in {6}; a//10 in {5} -> b//10 in {0,5,6,7}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,6,7} | **tens(a,b)** (R2 0.66): a//10 in {0,1,5} -> b//10 in {5,6,7}; a//10 in {2,3,4} -> b//10 in {6,7}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,2,3,6,7}; a//10 in {9,10} -> b//10 in {6} | - | same |
| L18 o c82 (H30) | 18% / 27% | **tens(a,b)** (R2 0.71): a//10 in {0,1,2,5,9,10} -> b//10 in {3}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,10}; a//10 in {4} -> b//10 in {0,1,3} | **tens(a,b)** (R2 0.73): a//10 in {0,1,2} -> b//10 in {3,4}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,3,4,5,9}; a//10 in {5,6,7} -> b//10 in {3} | - | same |
| L18 o c94 (H31) | 0 / 0 | off (on 0) | same | - | same |
| L18 o c107 (H30) | 44% / 42% | **tens(a,b)** (R2 0.83): a//10 in {0,1,7} -> b//10 in {7,8,9,10}; a//10 in {2,3,4,5,6} -> b//10 in {8,9,10}; a//10 in {8,9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.74): a//10 in {0,1,2,3} -> b//10 in {7,8,9,10}; a//10 in {4,5,6,7} -> b//10 in {8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,7,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,10} | - | b: mod100 +3%, mod50 +2% |
| L18 o c120 (H30) | 2% / 3% | **tens(a,b)** (R2 0.62): a//10 in {0,1,2,3,4,5,6,7,8,9} -> b//10 in {10}; a//10 in {10} -> b//10 in {0,10} | unexplained (best R2 0.49) | - | same |
| L18 o c130 (H31) | 1% / 1% | unexplained (best R2 0.11) | unexplained (best R2 0.18) | - | same |
| L18 o c135 (H30) | 9% / 11% | unexplained (best R2 0.39) | unexplained (best R2 0.34) | - | same |
| L18 o c137 (H30) | 13% / 11% | **tens(a,b)** (R2 0.63): a//10 in {0,3} -> b//10 in {3,4}; a//10 in {1,2} -> b//10 in {4}; a//10 in {4} -> b//10 in {0,1,2,3,4,5} | **tens(a,b)** (R2 0.57): a//10 in {0,1,2,3,8} -> b//10 in {4}; a//10 in {4} -> b//10 in {0,1,2,3,4,7,8,9,10} | - | same |
| L18 o c157 (H30) | 0 / 1% | off (on 0) | **tens(a,b)** (R2 0.88): a//10 in {0,1,2,3,4,5,6,10} -> b//10 in {10} | - | same |
| L18 o c166 (H18) | 18% / 19% | **tens(a,b)** (R2 0.73): a//10 in {0} -> b//10 in {4,5}; a//10 in {1,2,3,6,9,10} -> b//10 in {5}; a//10 in {4} -> b//10 in {0,4,5}; a//10 in {5} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.74): a//10 in {0,1,2,3,4,6} -> b//10 in {5}; a//10 in {5} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {4,5} | - | same |
| L18 o c195 (H30) | 6% / 10% | **units(a,b)** (R2 0.54): a%10 in {2} -> b%10 in {4}; a%10 in {4} -> b%10 in {1,2,4,6} | **units(a,b)** (R2 0.62): a%10 in {0,5,6,8,9} -> b%10 in {4}; a%10 in {4} -> b%10 in {3,4,8,9} | - | same |
| L18 o c200 (H30) | 12% / 12% | **units(a,b)** (R2 0.74): a%10 in {0,1,2,3,6,7} -> b%10 in {5}; a%10 in {5} -> b%10 in {1,3,5,6,7} | **units(a,b)** (R2 0.66): a%10 in {0,1,2,3,4,6,7,9} -> b%10 in {5}; a%10 in {5} -> b%10 in {3,4,5,7,9} | - | same |
| L18 o c210 (H18) | 4% / 7% | unexplained (best R2 0.23) | unexplained (best R2 0.22) | - | same |
| L18 o c230 (H18) | 1% / 2% | unexplained (best R2 0.22) | **a** (R2 0.50): a in {70} | - | same |
| L18 o c234 (H31) | 0 / 0 | off (on 0) | same | - | same |
| L18 o c285 (H18) | 1% / 0 | unexplained (best R2 0.15) | off (on 0) | - | same |
| L18 o c293 (H18) | 0 / 0 | off (on 0) | same | - | same |
| L18 o c313 (H30) | 22% / 21% | **units(a,b)** (R2 0.89): a%10 in {0} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {1,3,4,6,7,8,9} -> b%10 in {0}; a%10 in {2,5} -> b%10 in {0,5} | **units(a,b)** (R2 0.83): a%10 in {0} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {1,2,3,4,6,7,8,9} -> b%10 in {0}; a%10 in {5} -> b%10 in {0,5} | b: mod5 +3% | a: mod5 +4%; b: mod5 +6% |
| L18 o c320 (H31) | 0 / 0 | off (on 0) | same | - | same |
| L18 o c324 (H18) | 21% / 18% | **tens(a,b)** (R2 0.57): a//10 in {0,1,2,3,4,5,6,8} -> b//10 in {9,10}; a//10 in {7} -> b//10 in {10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.55): a//10 in {0,4,5,6,8} -> b//10 in {9,10}; a//10 in {1,2,3} -> b//10 in {10}; a//10 in {7} -> b//10 in {9}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | - | same |
| L18 o c327 (H18) | 0 / 0 | off (on 0) | same | - | same |
| L18 o c336 (H18) | 0 / 2% | off (on 0) | **a** (R2 0.64): a in {99} | - | same |
| L18 o c355 (H30) | 4% / 13% | **tens(a,b)** (R2 0.59): a//10 in {0} -> b//10 in {7}; a//10 in {7} -> b//10 in {0,1,7} | **tens(a,b)** (R2 0.70): a//10 in {0,1,2,3,4,5} -> b//10 in {7}; a//10 in {7} -> b//10 in {0,1,2,3,4,7,10} | - | same |
| L18 o c387 (H18) | 0 / 0 | off (on 0) | same | - | same |
| L18 o c402 (H30) | 11% / 15% | **tens(a,b)** (R2 0.52): a//10 in {0,1,2,3,5} -> b//10 in {4}; a//10 in {4} -> b//10 in {0,1,2,4} | unexplained (best R2 0.49) | - | same |
| L18 o c416 (H18) | 0 / 0 | off (on 0) | same | - | same |
| L18 o c423 (H31) | 1% / 11% | unexplained (best R2 0.25) | **tens(a,b)** (R2 0.64): a//10 in {0,1,2,3} -> b//10 in {7}; a//10 in {7} -> b//10 in {0,1,2,3,4} | - | same |
| L18 o c427 (H30) | 5% / 6% | **tens(a,b)** (R2 0.51): a//10 in {0,1} -> b//10 in {2}; a//10 in {2} -> b//10 in {0,1,2} | unexplained (best R2 0.37) | - | same |
| L18 o c445 (H30) | 7% / 10% | **units(a,b)** (R2 0.51): a%10 in {1,2,5} -> b%10 in {3}; a%10 in {3} -> b%10 in {1,2,3} | **units(a,b)** (R2 0.54): a%10 in {1,2,7,8,9} -> b%10 in {3}; a%10 in {3} -> b%10 in {3,7,8} | - | same |
| L18 o c486 (H31) | 0 / 0 | off (on 0) | same | - | same |
| L18 o c496 (H16) | 22% / 92% | unexplained (best R2 0.27) | unexplained (best R2 0.44) | - | same |

</details>

<details><summary>q: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L18 q c104 (H7) | 100% / 100% | always | same | (reads) | same |

</details>

<details><summary>up: 71</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L18 up c12 | 55% / 38% | **b** (R2 0.67): b in {10..33, 58..86} [coarser: b mod 50 in {9..36}, R2 0.98] | **tens(a,b)** (R2 0.68): a//10 in {2} -> b//10 in {8}; a//10 in {3,4,8} -> b//10 in {1,2,3,6,7,8}; a//10 in {5} -> b//10 in {2,3,7,8}; a//10 in {6} -> b//10 in {3,7,8}; a//10 in {7} -> b//10 in {3,8}; a//10 in {9} -> b//10 in {1,2,3,6,7,8,10}; a//10 in {10} -> b//10 in {1,2,3,4,6,7,8} | (reads) | same |
| L18 up c16 | 19% / 15% | unexplained (best R2 0.31) | unexplained (best R2 0.34) | (reads) | same |
| L18 up c18 | 25% / 25% | unexplained (best R2 0.33) | unexplained (best R2 0.44) | (reads) | same |
| L18 up c21 | 50% / 34% | **tens(a,b)** (R2 0.71): a//10 in {0,1,5,6,10} -> b//10 in {1,2,3,6,7,8}; a//10 in {2} -> b//10 in {1,2,3,6,7}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4,9} -> b//10 in {2,3,7,8}; a//10 in {7} -> b//10 in {1,2,6,7}; a//10 in {8} -> b//10 in {2} | **tens(a,b)** (R2 0.65): a//10 in {1} -> b//10 in {1,2,3}; a//10 in {2} -> b//10 in {2}; a//10 in {3} -> b//10 in {6,7}; a//10 in {4,9} -> b//10 in {1,2,3,6,7}; a//10 in {5,6,10} -> b//10 in {1,2,3,6,7,8}; a//10 in {7} -> b//10 in {3,7}; a//10 in {8} -> b//10 in {7} | (reads) | same |
| L18 up c22 | 57% / 23% | **b%50** (R2 0.77): b mod 50 in {0..26, 48..49} | **tens(a,b)** (R2 0.59): a//10 in {3} -> b//10 in {3}; a//10 in {4,8} -> b//10 in {3,4,8}; a//10 in {5,6} -> b//10 in {3,4,5,8,9,10}; a//10 in {7} -> b//10 in {3,4,9}; a//10 in {9} -> b//10 in {3,8,9,10}; a//10 in {10} -> b//10 in {3,4,5,7,8,9,10} | (reads) | same |
| L18 up c23 | 37% / 27% | **tens(a,b)** (R2 0.81): a//10 in {2} -> b//10 in {5,6,7}; a//10 in {3} -> b//10 in {4,5,6,7,8,9}; a//10 in {4} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {5} -> b//10 in {3,4,5,6,7,8}; a//10 in {6} -> b//10 in {2,3,4,5,6}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {3,4,5}; a//10 in {9,10} -> b//10 in {3,4} | **tens(a,b)** (R2 0.73): a//10 in {3} -> b//10 in {2,3,4,5,6}; a//10 in {4,5,6} -> b//10 in {3,4,5,6}; a//10 in {7} -> b//10 in {4,5,6}; a//10 in {8} -> b//10 in {4,5,6,7} | (reads) | same |
| L18 up c24 | 56% / 48% | **units(a,b)** (R2 0.84): a%10 in {0,5} -> b%10 in {1,2,6,7}; a%10 in {2} -> b%10 in {0,2,3,4,6,7,8,9}; a%10 in {3,8} -> b%10 in {1,2,3,4,6,7,8,9}; a%10 in {4,9} -> b%10 in {1,2,3,6,7,8}; a%10 in {6} -> b%10 in {4,6}; a%10 in {7} -> b%10 in {0,2,3,4,5,7,8,9} | unexplained (best R2 0.49) | (reads) | same |
| L18 up c25 | 52% / 17% | **tens(a,b)** (R2 0.77): a//10 in {0,1} -> b//10 in {2,3,4,7,8,9}; a//10 in {2} -> b//10 in {2,3,7,8,9}; a//10 in {3} -> b//10 in {8,9,10}; a//10 in {4,5,6,9,10} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {7} -> b//10 in {2,3,7,8,9,10}; a//10 in {8} -> b//10 in {3,4,8,9,10} | **tens(a,b)** (R2 0.51): a//10 in {1,2,4} -> b//10 in {1}; a//10 in {5} -> b//10 in {1,2,6,7}; a//10 in {6,9} -> b//10 in {1,6,7}; a//10 in {10} -> b//10 in {0,1,2,5,6,7} | (reads) | same |
| L18 up c26 | 37% / 43% | **units(a,b)** (R2 0.90): a%10 in {0,3,4,8,9} -> b%10 in {0,3,4,5,8,9}; a%10 in {1} -> b%10 in {3}; a%10 in {5} -> b%10 in {0,3,4,8,9} | **units(a,b)** (R2 0.57): a%10 in {0,5} -> b%10 in {0,1,2,3,5,6,7,8}; a%10 in {2,7} -> b%10 in {0}; a%10 in {3,4,8,9} -> b%10 in {0,1,2,5,6,7} | (reads) | same |
| L18 up c27 | 61% / 50% | **b%50** (R2 0.63): b mod 50 in {0..9, 11, 31..49} | **tens(a,b)** (R2 0.65): a//10 in {0} -> b//10 in {0,1,4,5,8,9,10}; a//10 in {1} -> b//10 in {0,9,10}; a//10 in {2,7} -> b//10 in {0,5,9,10}; a//10 in {3,4} -> b//10 in {0,1,4,5,9,10}; a//10 in {5} -> b//10 in {0,1,5,6,9,10}; a//10 in {6} -> b//10 in {0,1,5,9,10}; a//10 in {8,10} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,8,9,10} | (reads) | same |
| L18 up c32 | 28% / 42% | **units(a,b)** (R2 0.88): a%10 in {2} -> b%10 in {5,6,7}; a%10 in {3} -> b%10 in {3,4,5,6,7}; a%10 in {4} -> b%10 in {2,3,4,5,6,7}; a%10 in {5} -> b%10 in {1,2,3,4,5,6,7}; a%10 in {6} -> b%10 in {1,2,3,4,5,6} | unexplained (best R2 0.47) | (reads) | same |
| L18 up c36 | 35% / 26% | **tens(a,b)** (R2 0.65): a//10 in {0,1,5,6} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {2} -> b//10 in {2,3,7,8}; a//10 in {7} -> b//10 in {2,3,4,8}; a//10 in {10} -> b//10 in {3,4,8} | **tens(a,b)** (R2 0.65): a//10 in {0} -> b//10 in {0,1}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {1,2}; a//10 in {5} -> b//10 in {0,1,2,5,6,7,10}; a//10 in {6} -> b//10 in {0,1,2,3,5,6,7,10}; a//10 in {7} -> b//10 in {1,6,7}; a//10 in {10} -> b//10 in {0,1,5,6,7} | (reads) | same |
| L18 up c59 | 13% / 6% | unexplained (best R2 0.24) | unexplained (best R2 0.18) | (reads) | same |
| L18 up c61 | 49% / 36% | **tens(a,b)** (R2 0.69): a//10 in {0,5} -> b//10 in {3,8}; a//10 in {2} -> b//10 in {3,4,8,9,10}; a//10 in {3,8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {4,9} -> b//10 in {0,2,3,4,5,7,8,9,10}; a//10 in {7} -> b//10 in {3,4,9,10}; a//10 in {10} -> b//10 in {2,3,4,7,8,9} | **tens(a,b)** (R2 0.67): a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {0,1,2,3,6,7,10}; a//10 in {4} -> b//10 in {0,1,2,6,7,8,10}; a//10 in {7} -> b//10 in {0,5,6}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,8,10}; a//10 in {10} -> b//10 in {0,1,2,5,6,7} | (reads) | same |
| L18 up c62 | 27% / 17% | **units(a,b)** (R2 0.83): a%10 in {0} -> b%10 in {6,7}; a%10 in {1} -> b%10 in {5,6}; a%10 in {4} -> b%10 in {7}; a%10 in {5} -> b%10 in {6,7,8}; a%10 in {6,7,8} -> b%10 in {5,6,7,8,9}; a%10 in {9} -> b%10 in {5,6,7,8} | unexplained (best R2 0.41) | (reads) | same |
| L18 up c63 | 0 / 2% | off (on 0) | unexplained (best R2 0.36) | (reads) | same |
| L18 up c65 | 40% / 34% | **units(a,b)** (R2 0.87): a%10 in {0} -> b%10 in {5,6}; a%10 in {4,9} -> b%10 in {5,6,7}; a%10 in {5} -> b%10 in {0,3,4,5,6,7,8,9}; a%10 in {6} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {7} -> b%10 in {0,2,3,4,5,6,7,8,9}; a%10 in {8} -> b%10 in {5,6,7,8} | **units(a,b)** (R2 0.63): a%10 in {4} -> b%10 in {4,5}; a%10 in {5} -> b%10 in {0,1,2,3,4,5,6,7}; a%10 in {6} -> b%10 in {0,1,2,3,4,5,6,7,8}; a%10 in {7} -> b%10 in {1,2,3,4,5,6,7,8}; a%10 in {8} -> b%10 in {2,3,4,5,6}; a%10 in {9} -> b%10 in {3,4} | (reads) | same |
| L18 up c73 | 0 / 0 | off (on 0) | same | (reads) | same |
| L18 up c77 | 0 / 12% | off (on 0) | unexplained (best R2 0.43) | (reads) | same |
| L18 up c80 | 25% / 58% | **units(a,b)** (R2 0.81): a%10 in {0,1} -> b%10 in {0,3,4,5,8,9}; a%10 in {4} -> b%10 in {0,9}; a%10 in {5} -> b%10 in {0,4,5,8,9}; a%10 in {9} -> b%10 in {0,1,4,5,9} | unexplained (best R2 0.28) | (reads) | same |
| L18 up c85 | 0 / 3% | off (on 0) | **tens(a,b)** (R2 0.54): a//10 in {9} -> b//10 in {6}; a//10 in {10} -> b//10 in {0,1,5,6,7} | (reads) | same |
| L18 up c88 | 32% / 17% | **units(a,b)** (R2 0.90): a%10 in {0,4} -> b%10 in {1,2,3,4}; a%10 in {1,2} -> b%10 in {1,2,3,4,5,6,7}; a%10 in {3} -> b%10 in {1,2,3,4,5,6}; a%10 in {5} -> b%10 in {1,2,3} | unexplained (best R2 0.37) | (reads) | same |
| L18 up c91 | 31% / 9% | **tens(a,b)** (R2 0.82): a//10 in {0} -> b//10 in {3,4}; a//10 in {1} -> b//10 in {2,3,4}; a//10 in {2,3,4} -> b//10 in {0,1,2,3,4,5,6}; a//10 in {5} -> b//10 in {2,3,4,5}; a//10 in {6} -> b//10 in {4} | **tens(a,b)** (R2 0.52): a//10 in {3} -> b//10 in {2,3,4}; a//10 in {4} -> b//10 in {3,4}; a//10 in {5} -> b//10 in {4,5} | (reads) | same |
| L18 up c98 | 31% / 24% | **units(a,b)** (R2 0.83): a%10 in {0,8,9} -> b%10 in {4,5,6,7}; a%10 in {1} -> b%10 in {3,4,5,6,7}; a%10 in {2,3,6,7} -> b%10 in {4,5,6}; a%10 in {4,5} -> b%10 in {5,6} | **units(a,b)** (R2 0.55): a%10 in {0,8,9} -> b%10 in {3,4,5,6}; a%10 in {1} -> b%10 in {3,4,5,6,7}; a%10 in {2,7} -> b%10 in {4,5,6}; a%10 in {6} -> b%10 in {5} | (reads) | same |
| L18 up c101 | 0 / 1% | off (on 0) | unexplained (best R2 0.21) | (reads) | same |
| L18 up c109 | 4% / 6% | unexplained (best R2 0.29) | unexplained (best R2 0.20) | (reads) | same |
| L18 up c111 | 1% / 10% | **tens(a,b)** (R2 0.51): a//10 in {9} -> b//10 in {9,10}; a//10 in {10} -> b//10 in {10} | **tens(a,b)** (R2 0.67): a//10 in {7,8,9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {5,6,7,8,9,10} | (reads) | same |
| L18 up c118 | 0 / 61% | off (on 0) | **tens(a,b)** (R2 0.57): a//10 in {0,1,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,1,4,5,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,4,5,8,9,10}; a//10 in {5} -> b//10 in {0,1,2}; a//10 in {6,7} -> b//10 in {0,1,2,3}; a//10 in {8} -> b//10 in {0,1,2,3,4}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,10} | (reads) | same |
| L18 up c120 | 26% / 14% | **units(a,b)** (R2 0.88): a%10 in {2} -> b%10 in {4,5,6,7}; a%10 in {3,4} -> b%10 in {2,3,4,5,6,7}; a%10 in {5} -> b%10 in {2,3,4,5,6}; a%10 in {6} -> b%10 in {3,4,5}; a%10 in {7} -> b%10 in {3} | unexplained (best R2 0.48) | (reads) | same |
| L18 up c123 | 5% / 2% | **units(a,b)** (R2 0.87): a%10 in {5} -> b%10 in {2,3,4}; a%10 in {6} -> b%10 in {2,3} | unexplained (best R2 0.25) | (reads) | same |
| L18 up c130 | 17% / 1% | **tens(a,b)** (R2 0.73): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1,2,3,4,5,6,7,8,9} -> b//10 in {0}; a//10 in {10} -> b//10 in {0,1} | **tens(a,b)** (R2 0.56): a//10 in {0} -> b//10 in {0} | (reads) | same |
| L18 up c142 | 95% / 9% | always | unexplained (best R2 0.26) | (reads) | same |
| L18 up c156 | 26% / 11% | **units(a,b)** (R2 0.93): a%10 in {1} -> b%10 in {1,2,3,7,8,9}; a%10 in {2} -> b%10 in {1,2,3,6,7,8}; a%10 in {3} -> b%10 in {1,2,6,7}; a%10 in {4,9} -> b%10 in {1}; a%10 in {6} -> b%10 in {2}; a%10 in {7} -> b%10 in {1,2,3,7}; a%10 in {8} -> b%10 in {1,2} | unexplained (best R2 0.37) | (reads) | same |
| L18 up c178 | 2% / 0 | **units(a,b)** (R2 0.69): a%10 in {5} -> b%10 in {5} | off (on 0) | (reads) | same |
| L18 up c213 | 9% / 5% | **units(a,b)** (R2 0.76): a%10 in {0,9} -> b%10 in {4,5,6}; a%10 in {8} -> b%10 in {5,6} | unexplained (best R2 0.35) | (reads) | same |
| L18 up c227 | 0 / 1% | off (on 0) | unexplained (best R2 0.28) | (reads) | same |
| L18 up c261 | 0 / 0 | off (on 0) | same | (reads) | same |
| L18 up c297 | 19% / 4% | **b** (R2 0.60): b in {36..38, 40..48, 50..53, 55..56} | unexplained (best R2 0.35) | (reads) | same |
| L18 up c336 | 10% / 9% | **units(a,b)** (R2 0.86): a%10 in {2,7} -> b%10 in {9}; a%10 in {3,4,8,9} -> b%10 in {8,9} | unexplained (best R2 0.45) | (reads) | same |
| L18 up c346 | 13% / 6% | **units(a,b)** (R2 0.85): a%10 in {0} -> b%10 in {3,4,5,6}; a%10 in {1} -> b%10 in {3,4,5}; a%10 in {8} -> b%10 in {4,5}; a%10 in {9} -> b%10 in {3,4,5,6,9} | unexplained (best R2 0.49) | (reads) | same |
| L18 up c359 | 16% / 6% | **units(a,b)** (R2 0.89): a%10 in {1} -> b%10 in {2,7}; a%10 in {2,7} -> b%10 in {1,2,6,7}; a%10 in {3} -> b%10 in {1}; a%10 in {6} -> b%10 in {2,7,8}; a%10 in {8} -> b%10 in {1,6} | unexplained (best R2 0.31) | (reads) | same |
| L18 up c368 | 2% / 1% | **units(a,b)** (R2 0.53): a%10 in {7} -> b%10 in {2} | unexplained (best R2 0.18) | (reads) | same |
| L18 up c375 | 44% / 29% | **units(a,b)** (R2 0.84): a%10 in {0} -> b%10 in {2,3,7,8}; a%10 in {2,7} -> b%10 in {3,4,8,9}; a%10 in {3,8} -> b%10 in {0,2,3,4,5,7,8,9}; a%10 in {4,9} -> b%10 in {2,3,4,7,8,9}; a%10 in {5} -> b%10 in {2,3,4,7,8} | unexplained (best R2 0.45) | (reads) | same |
| L18 up c381 | 3% / 1% | **units(a,b)** (R2 0.54): a%10 in {4,5} -> b%10 in {4} | unexplained (best R2 0.15) | (reads) | same |
| L18 up c390 | 13% / 8% | **tens(a,b)** (R2 0.57): a//10 in {1} -> b//10 in {2}; a//10 in {2} -> b//10 in {1,2,3,7,8}; a//10 in {3} -> b//10 in {2,7}; a//10 in {7} -> b//10 in {2,7,8}; a//10 in {8} -> b//10 in {7} | **tens(a,b)** (R2 0.51): a//10 in {1} -> b//10 in {2}; a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {7,8} -> b//10 in {7} | (reads) | same |
| L18 up c400 | 33% / 17% | **tens(a,b)** (R2 0.73): a//10 in {0} -> b//10 in {0,1,2,6,10}; a//10 in {1,6} -> b//10 in {0,1,10}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {0,1,2}; a//10 in {4,5,7} -> b//10 in {0,1,2,10}; a//10 in {8} -> b//10 in {0,1,2,6,7,10}; a//10 in {9,10} -> b//10 in {0,1,2,5,6,7,10} | **tens(a,b)** (R2 0.67): a//10 in {3,4,6} -> b//10 in {8}; a//10 in {5,7} -> b//10 in {8,9}; a//10 in {8} -> b//10 in {3,4,7,8,9}; a//10 in {9} -> b//10 in {3,7,8,9,10}; a//10 in {10} -> b//10 in {2,3,4,7,8,9,10} | (reads) | same |
| L18 up c402 | 13% / 0 | **tens(a,b)** (R2 0.70): a//10 in {6} -> b//10 in {7,8,9}; a//10 in {7,8} -> b//10 in {6,7,8,9}; a//10 in {9} -> b//10 in {6,7,8} | off (on 0) | (reads) | same |
| L18 up c405 | 2% / 5% | unexplained (best R2 0.34) | unexplained (best R2 0.38) | (reads) | same |
| L18 up c435 | 34% / 13% | **tens(a,b)** (R2 0.65): a//10 in {0,1} -> b//10 in {0,4,5,6,9,10}; a//10 in {2} -> b//10 in {4,5,9,10}; a//10 in {4,9} -> b//10 in {0,5,10}; a//10 in {5,6,10} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {7} -> b//10 in {0,4,5,9,10} | **tens(a,b)** (R2 0.58): a//10 in {2} -> b//10 in {0}; a//10 in {5,6,10} -> b//10 in {0,4,5,9,10}; a//10 in {7} -> b//10 in {0,5} | (reads) | same |
| L18 up c446 | 8% / 8% | **units(a,b)** (R2 0.82): a%10 in {0} -> b%10 in {7,8,9}; a%10 in {1,2} -> b%10 in {6,7,8} | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {1,2,3}; a%10 in {1} -> b%10 in {1,2,3,4}; a%10 in {2} -> b%10 in {2,3} | (reads) | same |
| L18 up c474 | 24% / 18% | **units(a,b)** (R2 0.85): a%10 in {3} -> b%10 in {4,6}; a%10 in {4,5} -> b%10 in {0,3,4,5,6,7,8,9}; a%10 in {6} -> b%10 in {0,3,4,5,6,8,9} | **units(a,b)** (R2 0.53): a%10 in {4} -> b%10 in {0,1,2,3,4,5,6}; a%10 in {5} -> b%10 in {0,1,2,4,5,6}; a%10 in {6} -> b%10 in {4,5,6} | (reads) | same |
| L18 up c505 | 9% / 15% | **tens(a,b)** (R2 0.61): a//10 in {6} -> b//10 in {0,3,4,8}; a//10 in {7} -> b//10 in {2,3,7,8} | **tens(a,b)** (R2 0.67): a//10 in {2} -> b//10 in {7}; a//10 in {6} -> b//10 in {0,1,2,6,7,10}; a//10 in {7} -> b//10 in {0,1,2,3,6,7} | (reads) | same |
| L18 up c511 | 11% / 0 | unexplained (best R2 0.42) | off (on 0) | (reads) | same |
| L18 up c513 | 28% / 15% | **tens(a,b)** (R2 0.62): a//10 in {0,1} -> b//10 in {0,1,3,4,5,8,9,10}; a//10 in {2,10} -> b//10 in {0,4,9,10}; a//10 in {3,4,5,6,9} -> b//10 in {0,9,10}; a//10 in {7,8} -> b//10 in {9,10} | **tens(a,b)** (R2 0.58): a//10 in {0,1} -> b//10 in {0,1,10}; a//10 in {2} -> b//10 in {0,1}; a//10 in {4,7} -> b//10 in {0}; a//10 in {5,6} -> b//10 in {0,10}; a//10 in {9} -> b//10 in {0,9,10}; a//10 in {10} -> b//10 in {0,1,4,5,6,9,10} | (reads) | same |
| L18 up c518 | 19% / 10% | **tens(a,b)** (R2 0.77): a//10 in {3} -> b//10 in {5,6,7}; a//10 in {4,5,6} -> b//10 in {4,5,6,7}; a//10 in {7} -> b//10 in {5,6}; a//10 in {9,10} -> b//10 in {6} | **tens(a,b)** (R2 0.65): a//10 in {3} -> b//10 in {3}; a//10 in {4,5,10} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {3,4,5,6} | (reads) | same |
| L18 up c542 | 0 / 2% | off (on 0) | **tens(a,b)** (R2 0.69): a//10 in {9} -> b//10 in {9,10}; a//10 in {10} -> b//10 in {4,5,6,7,8,9} | (reads) | same |
| L18 up c543 | 27% / 12% | **b** (R2 0.65): b in {22..47} | **tens(a,b)** (R2 0.55): a//10 in {3,7,8} -> b//10 in {6}; a//10 in {4,5,6,9} -> b//10 in {6,7}; a//10 in {10} -> b//10 in {5,6,7} | (reads) | same |
| L18 up c549 | 3% / 1% | unexplained (best R2 0.46) | unexplained (best R2 0.12) | (reads) | same |
| L18 up c573 | 0 / 0 | off (on 0) | same | (reads) | same |
| L18 up c619 | 16% / 6% | **units(a,b)** (R2 0.79): a%10 in {3,8} -> b%10 in {3,4,8,9}; a%10 in {4} -> b%10 in {3,8}; a%10 in {7} -> b%10 in {4}; a%10 in {9} -> b%10 in {2,3,4,7,8,9} | unexplained (best R2 0.33) | (reads) | same |
| L18 up c630 | 1% / 1% | **units(a,b)** (R2 0.62): a%10 in {5} -> b%10 in {3} | unexplained (best R2 0.19) | (reads) | same |
| L18 up c657 | 0 / 0 | off (on 0) | same | (reads) | same |
| L18 up c676 | 0 / 0 | off (on 0) | same | (reads) | same |
| L18 up c695 | 20% / 8% | **units(a,b)** (R2 0.75): a%10 in {0} -> b%10 in {2,7}; a%10 in {1,2,6,7} -> b%10 in {1,2,6,7}; a%10 in {5} -> b%10 in {2,6,7} | unexplained (best R2 0.25) | (reads) | same |
| L18 up c708 | 5% / 2% | unexplained (best R2 0.18) | unexplained (best R2 0.13) | (reads) | same |
| L18 up c753 | 1% / 0 | unexplained (best R2 0.19) | off (on 0) | (reads) | same |
| L18 up c779 | 10% / 4% | **tens(a,b)** (R2 0.62): a//10 in {4} -> b//10 in {4,5,6,10}; a//10 in {5,10} -> b//10 in {4,5,10}; a//10 in {6} -> b//10 in {4,5}; a//10 in {9} -> b//10 in {5,10} | **tens(a,b)** (R2 0.53): a//10 in {4} -> b//10 in {10}; a//10 in {5} -> b//10 in {4,5,10}; a//10 in {6} -> b//10 in {5}; a//10 in {10} -> b//10 in {4,5} | (reads) | same |
| L18 up c819 | 5% / 2% | **units(a,b)** (R2 0.81): a%10 in {3,4} -> b%10 in {3,4} | unexplained (best R2 0.22) | (reads) | same |
| L18 up c841 | 7% / 9% | **units(a,b)** (R2 0.58): a%10 in {0} -> b%10 in {6,7}; a%10 in {1} -> b%10 in {5,6,7}; a%10 in {2} -> b%10 in {5}; a%10 in {9} -> b%10 in {7} | unexplained (best R2 0.39) | (reads) | same |
| L18 up c870 | 24% / 17% | **units(a,b)** (R2 0.83): a%10 in {0,2} -> b%10 in {0,1,9}; a%10 in {1} -> b%10 in {0,1,8,9}; a%10 in {3,7,8} -> b%10 in {1}; a%10 in {4,5,6} -> b%10 in {0,1}; a%10 in {9} -> b%10 in {0,1,2,9} | **units(a,b)** (R2 0.62): a%10 in {0,2} -> b%10 in {0,1,9}; a%10 in {1} -> b%10 in {0,1,2,9}; a%10 in {4,5,8} -> b%10 in {9}; a%10 in {9} -> b%10 in {0,1,8,9} | (reads) | same |
| L18 up c886 | 5% / 2% | **units(a,b)** (R2 0.69): a%10 in {5} -> b%10 in {2,3,8}; a%10 in {6} -> b%10 in {2} | unexplained (best R2 0.23) | (reads) | same |

</details>

</details>

<details><summary>layer 19: 80 components with main position `=`</summary>

<details><summary>down: 33</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L19 down c0 | 51% / 60% | **res%50** (R2 0.84): res mod 50 in {14..39} | **tens(a,b)** (R2 0.54): a//10 in {0,5} -> b//10 in {2,3,7,8}; a//10 in {1} -> b//10 in {3,4,7,8,9}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {1,2,6,7,8}; a//10 in {6} -> b//10 in {2,3,4,7,8}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,10}; a//10 in {9} -> b//10 in {1,2,3,6,7}; a//10 in {10} -> b//10 in {1,2,3,7,8} | a: mod50 +2%; res: mod50 +26%, mod25 +64% | res: mod50 +18%, mod25 +4% |
| L19 down c1 | 100% / 100% | always | same | - | same |
| L19 down c3 | 27% / 98% | **res//10** (R2 0.63): (tens) res in {2..64, 200} | always | - | same |
| L19 down c6 | 66% / 66% | unexplained (best R2 0.26) | unexplained (best R2 0.21) | a: mod20 -3%; b: mod20 +14%, mod4 +8%; res: mod20 +47% | b: mod20 +11%, mod4 +4%; res: mod20 +33% |
| L19 down c8 | 4% / 87% | unexplained (best R2 0.14) | unexplained (best R2 0.46) | - | res: mod100 +7%, mod25 +2% |
| L19 down c11 | 46% / 44% | **tens(a,b)** (R2 0.71): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {4,8,9,10}; a//10 in {2,7} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {3,8} -> b//10 in {1,2,3,6,7,8}; a//10 in {4} -> b//10 in {1,2,5,6,7}; a//10 in {5} -> b//10 in {4,5,6}; a//10 in {6} -> b//10 in {3,4,5,8,9,10}; a//10 in {9} -> b//10 in {1,2,6,7,10}; a//10 in {10} -> b//10 in {0,1,2,5,6,7,10} | **tens(a,b)** (R2 0.61): a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {0,1,2,3,6,7,8,10}; a//10 in {3} -> b//10 in {1,2,3,7,8}; a//10 in {4} -> b//10 in {2,3,7,8}; a//10 in {5} -> b//10 in {3,4,8}; a//10 in {6} -> b//10 in {4,5,6,8,9}; a//10 in {7} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {1,2,3,6,7,8,9}; a//10 in {9} -> b//10 in {7,8,9}; a//10 in {10} -> b//10 in {3,4,8,9} | a: mod50 +4%; res: mod50 +11%, mod25 +18% | b: mod50 +4%; res: mod100 +2%, mod50 +11%, mod25 +5% |
| L19 down c17 | 0 / 11% | off (on 0) | **tens(a,b)** (R2 0.65): a//10 in {4,5,6,7} -> b//10 in {0}; a//10 in {8,9} -> b//10 in {0,1}; a//10 in {10} -> b//10 in {0,1,2} | - | same |
| L19 down c20 | 60% / 25% | **res%50** (R2 0.68): res mod 50 in {4..31} | **tens(a,b)** (R2 0.56): a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {2,3}; a//10 in {5} -> b//10 in {3,4,8}; a//10 in {6} -> b//10 in {0,4,5,9,10}; a//10 in {7} -> b//10 in {0,5,6}; a//10 in {8} -> b//10 in {0,6,7}; a//10 in {9} -> b//10 in {2,7,8}; a//10 in {10} -> b//10 in {2,3,4,8,9} | res: mod50 +9%, mod25 +4% | res: mod50 +4% |
| L19 down c24 | 44% / 19% | **res//10** (R2 0.77): (tens) res in {24, 26, 28..80, 142..174} [coarser: res mod 100 in {34..76}, R2 0.81] | **tens(a,b)** (R2 0.57): a//10 in {2} -> b//10 in {6,7,8}; a//10 in {3} -> b//10 in {7,8}; a//10 in {4} -> b//10 in {7,8,9}; a//10 in {5} -> b//10 in {0,5,8,9,10}; a//10 in {6} -> b//10 in {0,9,10}; a//10 in {8,9} -> b//10 in {3,4}; a//10 in {10} -> b//10 in {3,4,5,6} | res: mod100 +17% | - |
| L19 down c28 | 57% / 20% | **res//10** (R2 0.79): (tens) res in {14..52, 100..162, 164} [coarser: res mod 100 in {1, 10..56}, R2 0.81] | **tens(a,b)** (R2 0.65): a//10 in {2} -> b//10 in {0}; a//10 in {4} -> b//10 in {1}; a//10 in {5} -> b//10 in {1,2,3}; a//10 in {6} -> b//10 in {2,3,4}; a//10 in {7} -> b//10 in {3,4,5}; a//10 in {8} -> b//10 in {4,5,6}; a//10 in {9} -> b//10 in {6,7}; a//10 in {10} -> b//10 in {5,6,7,8} | res: mod100 +12% | - |
| L19 down c32 | 26% / 11% | **units(a,b)** (R2 0.82): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} | unexplained (best R2 0.31) | a: mod2 +18%; b: mod2 +18%; res: mod2 +13% | a: mod2 +4%; b: mod2 +3%; res: mod2 +3% |
| L19 down c33 | 33% / 4% | **res%100** (R2 0.73): res mod 100 in {1, 75..98} | **res** (R2 0.58): res in {78..99} | res: mod100 +2% | - |
| L19 down c34 | 14% / 0 | **res//10** (R2 0.82): (tens) res in {2..53} | off (on 0) | - | same |
| L19 down c36 | 0 / 17% | off (on 0) | unexplained (best R2 0.45) | - | res: mod25 +2% |
| L19 down c41 | 40% / 27% | **units(a,b)** (R2 0.85): a%10 in {0} -> b%10 in {6,7,8,9}; a%10 in {1} -> b%10 in {5,6,7,8}; a%10 in {2} -> b%10 in {5,6,7}; a%10 in {3} -> b%10 in {5,6}; a%10 in {5} -> b%10 in {1,2,3,4}; a%10 in {6} -> b%10 in {0,1,2,3,9}; a%10 in {7} -> b%10 in {0,1,2,8,9}; a%10 in {8} -> b%10 in {0,1,7,8,9}; a%10 in {9} -> b%10 in {0,6,7,8,9} | **units(a,b)** (R2 0.55): a%10 in {0} -> b%10 in {1,2,3,4}; a%10 in {1} -> b%10 in {2,3,4}; a%10 in {6} -> b%10 in {0,8,9}; a%10 in {7} -> b%10 in {0,1,2,8,9}; a%10 in {8} -> b%10 in {0,1,2,9}; a%10 in {9} -> b%10 in {0,1,2,3} | a: mod10 +3%; b: mod10 +3%; res: mod10 +13% | res: mod10 +7% |
| L19 down c53 | 1% / 2% | unexplained (best R2 0.20) | unexplained (best R2 0.23) | - | same |
| L19 down c55 | 23% / 15% | **res%10** (R2 0.81): res mod 10 in {0..1} | **res%100** (R2 0.48): res mod 100 in {0..1, 10..11, 20, 30, 81, 90..91} | res: mod10 +7%, mod5 +3% | res: mod10 +6%, mod5 +3% |
| L19 down c74 | 20% / 5% | **res%10** (R2 0.73): res mod 10 in {8..9} | **res** (R2 0.57): res in {8..9, 18..19, 28..29, 98..99} | res: mod10 +2% | - |
| L19 down c82 | 0 / 2% | off (on 0) | **tens(a,b)** (R2 0.60): a//10 in {0} -> b//10 in {0} | - | same |
| L19 down c85 | 6% / 3% | unexplained (best R2 0.15) | unexplained (best R2 0.08) | - | same |
| L19 down c86 | 25% / 1% | **res//10** (R2 0.92): (tens) res in {132..200} | **tens(a,b)** (R2 0.62): a//10 in {6} -> b//10 in {10}; a//10 in {10} -> b//10 in {3,4,5,6} | a: mod100 +3%; res: mod100 +5% | - |
| L19 down c94 | 13% / 9% | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {0,2,3,7,8,9,10}; a//10 in {1,2,3,4,5,6,7,8,9} -> b//10 in {0}; a//10 in {10} -> b//10 in {0,10} | **tens(a,b)** (R2 0.58): a//10 in {0} -> b//10 in {0,1,2,10}; a//10 in {1,3,7,8,10} -> b//10 in {0}; a//10 in {9} -> b//10 in {0,9,10} | - | same |
| L19 down c106 | 10% / 2% | **units(a,b)** (R2 0.59): a%10 in {0} -> b%10 in {4}; a%10 in {1} -> b%10 in {3,4}; a%10 in {3} -> b%10 in {1}; a%10 in {5} -> b%10 in {9}; a%10 in {6} -> b%10 in {8}; a%10 in {7} -> b%10 in {7}; a%10 in {8} -> b%10 in {6}; a%10 in {9} -> b%10 in {5} | unexplained (best R2 0.49) | - | same |
| L19 down c136 | 0 / 1% | off (on 0) | unexplained (best R2 0.15) | - | same |
| L19 down c143 | 22% / 9% | **units(a,b)** (R2 0.62): a%10 in {1,6} -> b%10 in {3,7,8,9}; a%10 in {2} -> b%10 in {2,7,8}; a%10 in {3,8} -> b%10 in {1,6,7}; a%10 in {7} -> b%10 in {2,3,7,8} | unexplained (best R2 0.24) | res: mod5 +3% | - |
| L19 down c147 | 9% / 0 | **tens(a,b)** (R2 0.64): a//10 in {1,9} -> b//10 in {9,10}; a//10 in {2,3} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {10} | off (on 0) | - | same |
| L19 down c149 | 27% / 11% | **res%100** (R2 0.77): res mod 100 in {2..4, 12..14, 22..24, 32..34, 42..44, 52..53, 62..63, 72..73, 82..83, 92..93} [coarser: res mod 10 in {2..4}, R2 0.97] | **res** (R2 0.65): res in {2..3, 12..13, 22..24, 32..33, 42..43, 82..83, 92..93} | res: mod10 +7%, mod5 +3% | res: mod10 +3% |
| L19 down c151 | 9% / 0 | **tens(a,b)** (R2 0.72): a//10 in {3,4,8} -> b//10 in {9,10}; a//10 in {5} -> b//10 in {9}; a//10 in {9} -> b//10 in {3,4,5,8,9,10}; a//10 in {10} -> b//10 in {4,8,9} | off (on 0) | - | same |
| L19 down c183 | 0 / 0 | off (on 0) | same | - | same |
| L19 down c278 | 2% / 54% | unexplained (best R2 0.25) | **tens(a,b)** (R2 0.67): a//10 in {0} -> b//10 in {7,8,9,10}; a//10 in {2} -> b//10 in {8,9,10}; a//10 in {3,4} -> b//10 in {6,7,8,9,10}; a//10 in {5} -> b//10 in {3,4,6,7,8,9,10}; a//10 in {6} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {8,9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | - | a: mod100 +2%; b: mod100 +3% |
| L19 down c367 | 0 / 20% | off (on 0) | **a//10** (R2 0.77): (tens) a in {1..19} | - | a: mod100 +2%, mod50 +2% |
| L19 down c423 | 11% / 0 | **tens(a,b)** (R2 0.74): a//10 in {1} -> b//10 in {2,3}; a//10 in {2,3} -> b//10 in {1,2,3} | off (on 0) | - | same |
| L19 down c454 | 1% / 1% | **res//10** (R2 0.62): (tens) res in {2..9} | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {0} | - | same |

</details>

<details><summary>gate: 27</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L19 gate c0 | 48% / 30% | **res%50** (R2 0.84): res mod 50 in {15..38} | **tens(a,b)** (R2 0.56): a//10 in {2} -> b//10 in {0,4,5,9,10}; a//10 in {3} -> b//10 in {0,1,5,6,7}; a//10 in {4} -> b//10 in {1,2,6,7}; a//10 in {5} -> b//10 in {2,3,7,8}; a//10 in {6} -> b//10 in {3,4,7,8}; a//10 in {7} -> b//10 in {4,5,9,10}; a//10 in {8} -> b//10 in {0,5,6,10}; a//10 in {9} -> b//10 in {6,7}; a//10 in {10} -> b//10 in {1,2,3,6,7,8} | (reads) | same |
| L19 gate c6 | 64% / 44% | unexplained (best R2 0.14) | unexplained (best R2 0.18) | (reads) | same |
| L19 gate c11 | 34% / 23% | **tens(a,b)** (R2 0.68): a//10 in {1} -> b//10 in {8,9}; a//10 in {2} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {3} -> b//10 in {1,2,3,6,7,8}; a//10 in {4,9} -> b//10 in {1,2,6,7}; a//10 in {6} -> b//10 in {4}; a//10 in {7} -> b//10 in {2,3,4,7,8,9}; a//10 in {8} -> b//10 in {1,2,3,7,8}; a//10 in {10} -> b//10 in {1,2} | **tens(a,b)** (R2 0.56): a//10 in {2} -> b//10 in {0,1,2,7,8}; a//10 in {3} -> b//10 in {1,2,3,7,8}; a//10 in {4} -> b//10 in {2,3,8}; a//10 in {7} -> b//10 in {5,6,7,8}; a//10 in {8} -> b//10 in {2,3,6,7,8}; a//10 in {9} -> b//10 in {7,8}; a//10 in {10} -> b//10 in {3,4,8,9} | (reads) | same |
| L19 gate c20 | 50% / 23% | **res%50** (R2 0.79): res mod 50 in {4..26} | **tens(a,b)** (R2 0.57): a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4,8}; a//10 in {6} -> b//10 in {0,4,5,9,10}; a//10 in {7} -> b//10 in {0,5,6}; a//10 in {8} -> b//10 in {6,7}; a//10 in {9} -> b//10 in {7,8}; a//10 in {10} -> b//10 in {3,4,8,9} | (reads) | same |
| L19 gate c23 | 79% / 28% | **res//10** (R2 0.90): (tens) res in {67..200} | **tens(a,b)** (R2 0.71): a//10 in {0,1,2} -> b//10 in {7,8,9,10}; a//10 in {3} -> b//10 in {9,10}; a//10 in {4} -> b//10 in {10}; a//10 in {7,10} -> b//10 in {0,1,2,3}; a//10 in {8} -> b//10 in {0,1,2,3,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5} | (reads) | same |
| L19 gate c24 | 27% / 8% | **res//10** (R2 0.79): (tens) res in {2..69} | **tens(a,b)** (R2 0.67): a//10 in {0,1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {2} | (reads) | same |
| L19 gate c28 | 43% / 17% | **tens(a,b)** (R2 0.69): a//10 in {0,10} -> b//10 in {6,7,8}; a//10 in {1} -> b//10 in {5,6,7}; a//10 in {2} -> b//10 in {4,5,6}; a//10 in {3} -> b//10 in {3,4,5}; a//10 in {4} -> b//10 in {2,3,4,7}; a//10 in {5} -> b//10 in {1,2,3,6,7,8}; a//10 in {6} -> b//10 in {0,1,2,5,6,7}; a//10 in {7} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {8} -> b//10 in {0,8,9,10}; a//10 in {9} -> b//10 in {7,8,9} | **tens(a,b)** (R2 0.58): a//10 in {3} -> b//10 in {6}; a//10 in {4} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {3,6,7,8}; a//10 in {6} -> b//10 in {3,4,7,8}; a//10 in {7} -> b//10 in {3,4,5,8,9,10}; a//10 in {8} -> b//10 in {10}; a//10 in {10} -> b//10 in {2,3,7} | (reads) | same |
| L19 gate c32 | 25% / 5% | **units(a,b)** (R2 0.79): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} | unexplained (best R2 0.22) | (reads) | same |
| L19 gate c41 | 37% / 25% | **units(a,b)** (R2 0.87): a%10 in {0} -> b%10 in {6,7,8,9}; a%10 in {1} -> b%10 in {5,6,7,8,9}; a%10 in {2} -> b%10 in {5,6,7}; a%10 in {3} -> b%10 in {5,6}; a%10 in {4} -> b%10 in {5}; a%10 in {5} -> b%10 in {1,2,3,4}; a%10 in {6} -> b%10 in {0,1,2,3}; a%10 in {7} -> b%10 in {0,1,2,8,9}; a%10 in {8} -> b%10 in {0,1,8,9}; a%10 in {9} -> b%10 in {0,7,8,9} | **units(a,b)** (R2 0.56): a%10 in {0} -> b%10 in {0,1,2,3,4}; a%10 in {1} -> b%10 in {1,2,3}; a%10 in {6} -> b%10 in {0,8,9}; a%10 in {7} -> b%10 in {0,1,8,9}; a%10 in {8} -> b%10 in {0,1,2,9}; a%10 in {9} -> b%10 in {0,1,2,3} | (reads) | same |
| L19 gate c55 | 24% / 16% | **res%10** (R2 0.82): res mod 10 in {0..1} | **res%100** (R2 0.49): res mod 100 in {0..2, 10..11, 20, 30, 81, 90..91} | (reads) | same |
| L19 gate c85 | 20% / 9% | **units(a,b)** (R2 0.88): a%10 in {0} -> b%10 in {5}; a%10 in {5} -> b%10 in {0,5,6,7,8,9}; a%10 in {6} -> b%10 in {5,6,7,8,9}; a%10 in {7,8} -> b%10 in {5,6,7}; a%10 in {9} -> b%10 in {5,6} | unexplained (best R2 0.32) | (reads) | same |
| L19 gate c106 | 11% / 3% | unexplained (best R2 0.48) | unexplained (best R2 0.33) | (reads) | same |
| L19 gate c143 | 25% / 11% | **units(a,b)** (R2 0.76): a%10 in {1} -> b%10 in {3,4,7,8,9}; a%10 in {2,7} -> b%10 in {2,3,7,8}; a%10 in {3} -> b%10 in {1,2,6,7}; a%10 in {6} -> b%10 in {3,8,9}; a%10 in {8} -> b%10 in {1,6,7} | unexplained (best R2 0.33) | (reads) | same |
| L19 gate c149 | 24% / 10% | **res%10** (R2 0.81): res mod 10 in {2..3} | **res** (R2 0.65): res in {2..3, 12..13, 22..23, 32..33, 42..43, 82..83, 92..93} | (reads) | same |
| L19 gate c159 | 17% / 17% | **units(a,b)** (R2 0.62): a%10 in {0} -> b%10 in {4,8}; a%10 in {2} -> b%10 in {2,4,6,8}; a%10 in {4,6,8} -> b%10 in {0,2,4,6,8} | unexplained (best R2 0.28) | (reads) | same |
| L19 gate c241 | 0 / 0 | off (on 0) | same | (reads) | same |
| L19 gate c255 | 2% / 0 | unexplained (best R2 0.08) | off (on 0) | (reads) | same |
| L19 gate c275 | 24% / 14% | unexplained (best R2 0.32) | unexplained (best R2 0.25) | (reads) | same |
| L19 gate c341 | 0 / 1% | off (on 0) | unexplained (best R2 0.11) | (reads) | same |
| L19 gate c423 | 32% / 3% | **res//10** (R2 0.70): (tens) res in {36, 38, 40..74, 143..172} [coarser: res mod 100 in {42..72}, R2 0.96] | unexplained (best R2 0.33) | (reads) | same |
| L19 gate c426 | 8% / 23% | unexplained (best R2 0.40) | **tens(a,b)** (R2 0.58): a//10 in {0} -> b//10 in {0,10}; a//10 in {4} -> b//10 in {9,10}; a//10 in {5} -> b//10 in {4,5,6,9,10}; a//10 in {6} -> b//10 in {6,7,8,9,10}; a//10 in {7} -> b//10 in {7,8,9,10}; a//10 in {8} -> b//10 in {8,9,10}; a//10 in {9} -> b//10 in {4,5,6,8,9,10}; a//10 in {10} -> b//10 in {0,4,5,6,9,10} | (reads) | same |
| L19 gate c513 | 18% / 3% | unexplained (best R2 0.26) | unexplained (best R2 0.13) | (reads) | same |
| L19 gate c639 | 47% / 14% | **res%50** (R2 0.62): res mod 50 in {0..5, 33..49} | unexplained (best R2 0.41) | (reads) | same |
| L19 gate c703 | 0 / 6% | off (on 0) | unexplained (best R2 0.26) | (reads) | same |
| L19 gate c731 | 0 / 0 | off (on 0) | same | (reads) | same |
| L19 gate c797 | 23% / 14% | unexplained (best R2 0.30) | unexplained (best R2 0.29) | (reads) | same |
| L19 gate c815 | 100% / 100% | always | same | (reads) | same |

</details>

<details><summary>up: 20</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L19 up c0 | 53% / 52% | **res%50** (R2 0.77): res mod 50 in {14..38} | **tens(a,b)** (R2 0.53): a//10 in {0} -> b//10 in {2,7}; a//10 in {1} -> b//10 in {2,3}; a//10 in {2} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {3,8} -> b//10 in {0,1,2,5,6,7,10}; a//10 in {4} -> b//10 in {1,2,3,6,7,8}; a//10 in {5} -> b//10 in {2,3,7,8}; a//10 in {6} -> b//10 in {2,3,4,7,8}; a//10 in {7} -> b//10 in {0,1,2,4,5,6,8,9,10}; a//10 in {9} -> b//10 in {2,6,7}; a//10 in {10} -> b//10 in {1,2,3,7,8} | (reads) | same |
| L19 up c6 | 99% / 63% | always | unexplained (best R2 0.31) | (reads) | same |
| L19 up c11 | 46% / 40% | **tens(a,b)** (R2 0.66): a//10 in {1} -> b//10 in {3,4,8,9}; a//10 in {2} -> b//10 in {1,2,3,4,7,8,9,10}; a//10 in {3,8} -> b//10 in {1,2,3,6,7,8}; a//10 in {4} -> b//10 in {1,2,6,7}; a//10 in {6} -> b//10 in {3,4,5,7,8,9,10}; a//10 in {7} -> b//10 in {1,2,3,4,6,7,8,9,10}; a//10 in {9} -> b//10 in {1,2,6,7,10}; a//10 in {10} -> b//10 in {5,6,7} | **tens(a,b)** (R2 0.60): a//10 in {2} -> b//10 in {0,1,2,3,6,7,8}; a//10 in {3} -> b//10 in {1,2,3,7,8}; a//10 in {4} -> b//10 in {3,7,8}; a//10 in {5} -> b//10 in {4,8}; a//10 in {6} -> b//10 in {5,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {1,2,3,6,7,8,9}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {3,4,5,6,7,8,9,10} | (reads) | same |
| L19 up c20 | 4% / 0 | **tens(a,b)** (R2 0.61): a//10 in {4} -> b//10 in {4}; a//10 in {5} -> b//10 in {3,4} | off (on 0) | (reads) | same |
| L19 up c24 | 13% / 100% | unexplained (best R2 0.45) | always | (reads) | same |
| L19 up c28 | 46% / 18% | **res//10** (R2 0.73): (tens) res in {100..171, 174..176, 178, 196, 198, 200} | **tens(a,b)** (R2 0.67): a//10 in {4} -> b//10 in {1}; a//10 in {5} -> b//10 in {1,2,3,4}; a//10 in {6} -> b//10 in {2,3,4,5}; a//10 in {7} -> b//10 in {3,4,5}; a//10 in {8} -> b//10 in {4,5,6}; a//10 in {9} -> b//10 in {6,7}; a//10 in {10} -> b//10 in {5,6,7,8} | (reads) | same |
| L19 up c41 | 40% / 26% | **units(a,b)** (R2 0.85): a%10 in {0} -> b%10 in {6,7,8,9}; a%10 in {1} -> b%10 in {5,6,7,8,9}; a%10 in {2} -> b%10 in {5,6,7}; a%10 in {3} -> b%10 in {5,6}; a%10 in {5} -> b%10 in {1,2,3,4}; a%10 in {6} -> b%10 in {0,1,2,3,9}; a%10 in {7} -> b%10 in {0,1,2,7,8,9}; a%10 in {8} -> b%10 in {0,1,7,8,9}; a%10 in {9} -> b%10 in {0,6,7,8,9} | **units(a,b)** (R2 0.56): a%10 in {0} -> b%10 in {1,2,3,4}; a%10 in {1} -> b%10 in {2,3,4}; a%10 in {6} -> b%10 in {0,8,9}; a%10 in {7} -> b%10 in {0,1,2,8,9}; a%10 in {8} -> b%10 in {0,1,2,9}; a%10 in {9} -> b%10 in {0,1,2,3} | (reads) | same |
| L19 up c53 | 0 / 1% | off (on 0) | unexplained (best R2 0.31) | (reads) | same |
| L19 up c55 | 23% / 15% | **res%10** (R2 0.81): res mod 10 in {0..1} | **res%100** (R2 0.47): res mod 100 in {0..2, 10..11, 20..21, 90..91} | (reads) | same |
| L19 up c85 | 29% / 33% | unexplained (best R2 0.41) | unexplained (best R2 0.39) | (reads) | same |
| L19 up c94 | 0 / 15% | off (on 0) | **tens(a,b)** (R2 0.68): a//10 in {1,2,3,4,5,6} -> b//10 in {0}; a//10 in {7,8,9} -> b//10 in {0,1}; a//10 in {10} -> b//10 in {0,1,2,3,4,5} | (reads) | same |
| L19 up c106 | 0 / 0 | off (on 0) | same | (reads) | same |
| L19 up c143 | 6% / 1% | unexplained (best R2 0.46) | unexplained (best R2 0.11) | (reads) | same |
| L19 up c346 | 23% / 14% | unexplained (best R2 0.30) | unexplained (best R2 0.19) | (reads) | same |
| L19 up c347 | 10% / 42% | unexplained (best R2 0.39) | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {0,1}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {1,2,3,4,6}; a//10 in {4} -> b//10 in {3,4,6}; a//10 in {5} -> b//10 in {3,4,5,6,7}; a//10 in {6} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {7} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10}; a//10 in {10} -> b//10 in {5,7,8,9,10} | (reads) | same |
| L19 up c423 | 48% / 27% | **res//10** (R2 0.75): (tens) res in {2..28, 81..127, 186..200} [coarser: res mod 100 in {0..27, 83..99}, R2 0.98] | **tens(a,b)** (R2 0.69): a//10 in {0} -> b//10 in {0,1,9,10}; a//10 in {1} -> b//10 in {0,1,2,10}; a//10 in {2} -> b//10 in {0,1,2}; a//10 in {3} -> b//10 in {2}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {4}; a//10 in {6} -> b//10 in {4,5,6,8}; a//10 in {7} -> b//10 in {5,6,7,8}; a//10 in {8} -> b//10 in {6,7,8,9}; a//10 in {9,10} -> b//10 in {0,8,9,10} | (reads) | same |
| L19 up c426 | 53% / 24% | **tens(a,b)** (R2 0.65): a//10 in {0} -> b//10 in {1,2,3,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,6,7,8,10}; a//10 in {2} -> b//10 in {0,1,5,6,10}; a//10 in {3} -> b//10 in {0,3,4,5,6,9,10}; a//10 in {4} -> b//10 in {3,4,5,8,9,10}; a//10 in {5} -> b//10 in {2,3,4}; a//10 in {6} -> b//10 in {0,1,2,3}; a//10 in {7} -> b//10 in {0,1,2,10}; a//10 in {8} -> b//10 in {0,1,3,4,5,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,7,8,9,10}; a//10 in {10} -> b//10 in {2,3,7,8,9,10} | **tens(a,b)** (R2 0.64): a//10 in {5} -> b//10 in {7}; a//10 in {6} -> b//10 in {6,7,8}; a//10 in {7} -> b//10 in {0,7,8,9,10}; a//10 in {8} -> b//10 in {0,1,5,6,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,5,6,7,9,10}; a//10 in {10} -> b//10 in {0,1,2,6,7,10} | (reads) | same |
| L19 up c506 | 6% / 5% | unexplained (best R2 0.18) | unexplained (best R2 0.20) | (reads) | same |
| L19 up c595 | 16% / 24% | unexplained (best R2 0.42) | unexplained (best R2 0.32) | (reads) | same |
| L19 up c804 | 10% / 3% | **units(a,b)** (R2 0.71): a%10 in {1} -> b%10 in {8}; a%10 in {2} -> b%10 in {7}; a%10 in {3} -> b%10 in {6}; a%10 in {4} -> b%10 in {5}; a%10 in {5} -> b%10 in {3,4}; a%10 in {6} -> b%10 in {3}; a%10 in {7} -> b%10 in {2}; a%10 in {8} -> b%10 in {1} | unexplained (best R2 0.44) | (reads) | same |

</details>

</details>

