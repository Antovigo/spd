# Appendix C3 — components whose main position is `=`, L20-24

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

<details><summary>layer 20: 101 components with main position `=`</summary>

<details><summary>down: 34</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L20 down c2 | 100% / 100% | always | same | res: mod100 +11%, mod50 +3% | res: mod100 +3%, mod50 +2% |
| L20 down c3 | 61% / 53% | **res%20** (R2 0.74): res mod 20 in {8..19} | unexplained (best R2 0.43) | a: mod10 +3%; res: mod20 +35% | b: mod20 +4%; res: mod25 +8%, mod20 +24% |
| L20 down c5 | 71% / 100% | **tens(a,b)** (R2 0.74): a//10 in {0,1,2} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,10}; a//10 in {4,5} -> b//10 in {0,1,2,3,4,5,6}; a//10 in {6} -> b//10 in {0,1,2,3,4}; a//10 in {7} -> b//10 in {0,1,2,3}; a//10 in {8} -> b//10 in {0,1,2,9}; a//10 in {9} -> b//10 in {0,1,2,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,6,8,9,10} | always | a: mod100 +3%; b: mod100 +2%; res: mod100 +3% | - |
| L20 down c6 | 60% / 49% | **res%20** (R2 0.60): res mod 20 in {1..11} | unexplained (best R2 0.33) | res: mod20 +31% | b: mod20 +4%; res: mod25 +6%, mod20 +23% |
| L20 down c9 | 60% / 49% | **res%5** (R2 0.95): res mod 5 in {1..3} | **res** (R2 0.62): res in {-97, -93, -63, -58, -53..-52, -48..-47, -43..-42, -38..-37, -33..-32, -28..-27, -23..-22, -19..-17, -13..-12, -9..-7, -4..-2, 1..3, 6..8, 11..13, 16..18, 21..23, 26..28, 31..33, 36..38, 41..43, 46..48, 51..53, 56..58, 61..63, 66..68, 71..73, 76..78, 81..83, 86..88, 91..93, 96..98} | res: mod5 +25% | res: mod5 +22% |
| L20 down c13 | 42% / 39% | **res%10** (R2 0.86): res mod 10 in {5..8} | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {2,3,4,5}; a%10 in {1} -> b%10 in {3,4,5,6}; a%10 in {2} -> b%10 in {4,5,6,7}; a%10 in {3} -> b%10 in {5,6,7}; a%10 in {4} -> b%10 in {6,7,8,9}; a%10 in {5} -> b%10 in {0,8,9}; a%10 in {6} -> b%10 in {0,1,8,9}; a%10 in {7} -> b%10 in {0,1,2,3,9}; a%10 in {8} -> b%10 in {0,1,2,3,4}; a%10 in {9} -> b%10 in {1,2,3,4,5} | res: mod10 +17% | res: mod10 +17% |
| L20 down c14 | 65% / 12% | **res** (R2 0.66): res in {39, 73..104, 114, 118..165, 167..200} | **tens(a,b)** (R2 0.55): a//10 in {8} -> b//10 in {0,4,5,9,10}; a//10 in {9} -> b//10 in {0,1,4,5,6}; a//10 in {10} -> b//10 in {0,1,2,5,6,7,10} | b: mod50 +2%; res: mod50 +6%, mod25 +15% | - |
| L20 down c15 | 61% / 46% | **res%5** (R2 0.83): res mod 5 in {0..2} | unexplained (best R2 0.44) | res: mod5 +18% | a: mod5 +2%; b: mod5 +3%; res: mod5 +14% |
| L20 down c20 | 54% / 48% | **res//10** (R2 0.81): (tens) res in {2..50, 99..150, 198..200} [coarser: res mod 100 in {0..50, 98..99}, R2 0.99] | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {0,1,2,3}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {0,1,2,3,9,10}; a//10 in {3} -> b//10 in {0,1,2,3,4,6,10}; a//10 in {4} -> b//10 in {0,1,2,3,4,6}; a//10 in {5} -> b//10 in {1,2,3,4,5}; a//10 in {6} -> b//10 in {2,3,4,5,6}; a//10 in {7} -> b//10 in {3,4,5,6,9}; a//10 in {8} -> b//10 in {4,5,6,7,8,9}; a//10 in {9} -> b//10 in {6,7,8,9,10}; a//10 in {10} -> b//10 in {0,5,6,7,8,9} | res: mod100 +8% | res: mod100 +4% |
| L20 down c21 | 42% / 37% | **res%10** (R2 0.89): res mod 10 in {0, 7..9} | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {0,1,2,3,9}; a%10 in {1} -> b%10 in {0,1,2,3}; a%10 in {2} -> b%10 in {1,2,3,4}; a%10 in {3} -> b%10 in {3,4,5,6}; a%10 in {4} -> b%10 in {4,5,6}; a%10 in {5} -> b%10 in {6,7}; a%10 in {6} -> b%10 in {6,7,8}; a%10 in {7} -> b%10 in {0,7,8,9}; a%10 in {8} -> b%10 in {0,1,8,9}; a%10 in {9} -> b%10 in {0,1,2,8,9} | res: mod10 +17%, mod5 +3% | a: mod10 +2%; b: mod10 +2%; res: mod10 +15%, mod5 +2% |
| L20 down c29 | 56% / 53% | **res%100** (R2 0.82): res mod 100 in {11..29, 55..86} | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {2,6,7,8}; a//10 in {1} -> b//10 in {7,8,9,10}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {1,2,6,7,8}; a//10 in {4,9} -> b//10 in {1,2,3,6,7,8}; a//10 in {5} -> b//10 in {2,3,4,7,8,9}; a//10 in {6} -> b//10 in {0,4,8,9,10}; a//10 in {7} -> b//10 in {0,1,5,6,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,5,6,7,8,10}; a//10 in {10} -> b//10 in {2,3,4,7,8} | a: mod50 +2%; res: mod50 +10% | b: mod50 +3%; res: mod50 +9% |
| L20 down c31 | 29% / 25% | **res//10** (R2 0.79): (tens) res in {2..13, 89..117, 188..200} [coarser: res mod 100 in {0..15, 89..99}, R2 0.98] | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {0,1,9,10}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {4,5}; a//10 in {6} -> b//10 in {5,6,7,8}; a//10 in {7} -> b//10 in {6,7,8,9}; a//10 in {8} -> b//10 in {7,8,9,10}; a//10 in {9,10} -> b//10 in {0,8,9,10} | res: mod100 +4%, mod25 +6% | res: mod100 +3%, mod50 +3%, mod25 +4% |
| L20 down c32 | 0 / 35% | off (on 0) | **tens(a,b)** (R2 0.59): a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {1,2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {3,4,5,6,7,8}; a//10 in {7,10} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10} | - | same |
| L20 down c44 | 7% / 12% | unexplained (best R2 0.28) | unexplained (best R2 0.32) | a: mod4 +8%; b: mod4 +10%, mod2 +2% | a: mod4 +10%; b: mod4 +9% |
| L20 down c47 | 43% / 28% | **res%2** (R2 0.74): res mod 2 in {0} | **units(a,b)** (R2 0.75): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} | a: mod2 -8%; b: mod2 -5%; res: mod2 +16% | a: mod2 -9%; b: mod2 -5%; res: mod2 +13% |
| L20 down c73 | 0 / 12% | off (on 0) | **tens(a,b)** (R2 0.64): a//10 in {4} -> b//10 in {0}; a//10 in {5,8} -> b//10 in {0,1}; a//10 in {6,7} -> b//10 in {0,1,2}; a//10 in {10} -> b//10 in {0,1,2,3,4} | - | same |
| L20 down c79 | 5% / 64% | unexplained (best R2 0.26) | **tens(a,b)** (R2 0.54): a//10 in {0,1} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,1,3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,2,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,4,5,6,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,6,8,9}; a//10 in {6,7,10} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1,2,3}; a//10 in {9} -> b//10 in {0,1,2,3,4,5} | - | a: mod100 +3% |
| L20 down c80 | 10% / 4% | **tens(a,b)** (R2 0.73): a//10 in {0} -> b//10 in {8,9,10}; a//10 in {7} -> b//10 in {10}; a//10 in {8,10} -> b//10 in {0,8,9,10}; a//10 in {9} -> b//10 in {0,7,8,9,10} | unexplained (best R2 0.40) | - | same |
| L20 down c100 | 71% / 20% | **res%100** (R2 0.54): res mod 100 in {1..5, 8..9, 13..15, 18..25, 29, 33..35, 38..45, 48..49, 52..65, 68..69, 73..75, 78..85, 88..89, 93..95, 97..99} | **res** (R2 0.50): res in {3..5, 9, 13..15, 19, 23..25, 33..34, 43..45, 54..55, 59, 63..65, 74..75, 79, 83..85, 89, 93..95, 98..99} | res: mod10 +3%, mod5 +6% | res: mod10 +2%, mod5 +3% |
| L20 down c101 | 14% / 0 | **res//10** (R2 0.77): (tens) res in {148..188, 190, 192..198, 200} | off (on 0) | - | same |
| L20 down c106 | 3% / 0 | unexplained (best R2 0.48) | off (on 0) | - | same |
| L20 down c128 | 0 / 0 | off (on 0) | same | - | same |
| L20 down c136 | 8% / 0 | **tens(a,b)** (R2 0.67): a//10 in {5} -> b//10 in {7}; a//10 in {6} -> b//10 in {6,7,8}; a//10 in {7} -> b//10 in {6,7} | off (on 0) | - | same |
| L20 down c153 | 19% / 0 | **res//10** (R2 0.91): (tens) res in {140..196, 198} | off (on 0) | - | same |
| L20 down c172 | 5% / 3% | **tens(a,b)** (R2 0.66): a//10 in {9} -> b//10 in {1,2,3,6}; a//10 in {10} -> b//10 in {1,2,3} | **a** (R2 0.79): a in {99..100} | - | same |
| L20 down c180 | 11% / 1% | **res** (R2 0.65): res in {121..137} | unexplained (best R2 0.24) | res: mod25 +4% | - |
| L20 down c181 | 0 / 13% | off (on 0) | **tens(a,b)** (R2 0.64): a//10 in {4} -> b//10 in {6}; a//10 in {5} -> b//10 in {6,7,8}; a//10 in {6} -> b//10 in {7,8,9}; a//10 in {7} -> b//10 in {8,9,10}; a//10 in {8} -> b//10 in {9,10} | - | same |
| L20 down c184 | 4% / 3% | unexplained (best R2 0.22) | unexplained (best R2 0.31) | - | same |
| L20 down c223 | 5% / 7% | **res** (R2 0.73): res in {113..118} | **res//10** (R2 0.69): (tens) res in {10..19} | res: mod25 +2% | - |
| L20 down c266 | 3% / 29% | unexplained (best R2 0.48) | **tens(a,b)** (R2 0.74): a//10 in {0} -> b//10 in {3,4}; a//10 in {1} -> b//10 in {2,3,4}; a//10 in {2} -> b//10 in {2,3,4,5}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,10}; a//10 in {4} -> b//10 in {0,1,2,3,4,5,6}; a//10 in {5,6} -> b//10 in {2,3} | - | same |
| L20 down c305 | 0 / 1% | off (on 0) | **tens(a,b)** (R2 0.55): a//10 in {9} -> b//10 in {9,10} | - | same |
| L20 down c378 | 18% / 6% | **units(a,b)** (R2 0.64): a%10 in {0} -> b%10 in {3}; a%10 in {5} -> b%10 in {7,8,9}; a%10 in {6} -> b%10 in {6,7,8,9}; a%10 in {7,8} -> b%10 in {5,6,7}; a%10 in {9} -> b%10 in {5,6} | unexplained (best R2 0.41) | - | same |
| L20 down c529 | 1% / 0 | unexplained (best R2 0.37) | off (on 0) | - | same |
| L20 down c676 | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>gate: 15</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L20 gate c2 | 43% / 22% | **res//10** (R2 0.76): (tens) res in {24..55, 116..164} [coarser: res mod 100 in {20..58}, R2 0.86] | **res//10** (R2 0.56): (tens) res in {21..50} | (reads) | same |
| L20 gate c3 | 53% / 24% | **units(a,b)** (R2 0.88): a%10 in {0} -> b%10 in {5,6,7,8,9}; a%10 in {1} -> b%10 in {5,6,7,8}; a%10 in {2} -> b%10 in {5,6,7}; a%10 in {3} -> b%10 in {5,6}; a%10 in {4} -> b%10 in {5}; a%10 in {5} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {6} -> b%10 in {0,1,2,3,5,6,7,8,9}; a%10 in {7} -> b%10 in {0,1,2,5,6,7,8,9}; a%10 in {8} -> b%10 in {0,1,5,6,7,8,9}; a%10 in {9} -> b%10 in {0,5,6,7,8,9} | unexplained (best R2 0.41) | (reads) | same |
| L20 gate c5 | 23% / 64% | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {6,7,8,9,10}; a//10 in {1} -> b//10 in {6,7,8}; a//10 in {2} -> b//10 in {6,7}; a//10 in {6} -> b//10 in {1,2}; a//10 in {7} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1,9,10}; a//10 in {9,10} -> b//10 in {0,8,9,10} | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {2,5,6,7,8,9,10}; a//10 in {3,4} -> b//10 in {7,8,9}; a//10 in {5} -> b//10 in {5,6,7,8,9,10}; a//10 in {6} -> b//10 in {0,1,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,7,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,7,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,9,10} | (reads) | same |
| L20 gate c6 | 44% / 23% | **units(a,b)** (R2 0.63): a%10 in {0} -> b%10 in {1,2,3,4}; a%10 in {1} -> b%10 in {0,1,2,3,4}; a%10 in {2} -> b%10 in {0,1,2,3,4,9}; a%10 in {3} -> b%10 in {0,1,2,3,4,8,9}; a%10 in {4} -> b%10 in {0,1,2,3,4,7,8,9}; a%10 in {7} -> b%10 in {4}; a%10 in {8} -> b%10 in {3,4}; a%10 in {9} -> b%10 in {2,3,4} | unexplained (best R2 0.35) | (reads) | same |
| L20 gate c13 | 47% / 21% | **res%10** (R2 0.78): res mod 10 in {0..3, 9} | **res** (R2 0.58): res in {-29, -19, -9, 0..3, 10..13, 20..23, 30..32, 41..42, 81..82, 91..92, 99} | (reads) | same |
| L20 gate c14 | 30% / 4% | **res//10** (R2 0.67): (tens) res in {81..99, 132..147, 174..200} | unexplained (best R2 0.40) | (reads) | same |
| L20 gate c20 | 48% / 31% | **res//10** (R2 0.76): (tens) res in {46..88, 141..188} [coarser: res mod 100 in {45..88}, R2 0.96] | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {7}; a//10 in {2} -> b//10 in {5,6,7,8}; a//10 in {3,4} -> b//10 in {7,8}; a//10 in {5} -> b//10 in {0,7,8,9,10}; a//10 in {6} -> b//10 in {0,1,2,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,3,10}; a//10 in {8} -> b//10 in {0,1,2,3}; a//10 in {9} -> b//10 in {1,2,3,4}; a//10 in {10} -> b//10 in {2,3,4,5} | (reads) | same |
| L20 gate c21 | 0 / 0 | off (on 0) | same | (reads) | same |
| L20 gate c31 | 36% / 31% | **res//10** (R2 0.78): (tens) res in {2..23, 91..125, 191..200} [coarser: res mod 100 in {0..23, 25, 91..99}, R2 0.98] | **tens(a,b)** (R2 0.69): a//10 in {0} -> b//10 in {0,1,10}; a//10 in {1} -> b//10 in {0,1}; a//10 in {2} -> b//10 in {0,1,2}; a//10 in {3} -> b//10 in {1,2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4,5}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {0,7,8,9,10} | (reads) | same |
| L20 gate c44 | 6% / 12% | unexplained (best R2 0.25) | unexplained (best R2 0.32) | (reads) | same |
| L20 gate c47 | 40% / 26% | **units(a,b)** (R2 0.78): a%10 in {0} -> b%10 in {4,6}; a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9}; a%10 in {2} -> b%10 in {4,6,8}; a%10 in {4,6} -> b%10 in {0,2,4,6,8}; a%10 in {8} -> b%10 in {2,4,6,8} | **units(a,b)** (R2 0.71): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} | (reads) | same |
| L20 gate c118 | 100% / 100% | always | same | (reads) | same |
| L20 gate c585 | 0 / 0 | off (on 0) | same | (reads) | same |
| L20 gate c587 | 0 / 20% | off (on 0) | unexplained (best R2 0.40) | (reads) | same |
| L20 gate c910 | 48% / 9% | unexplained (best R2 0.31) | unexplained (best R2 0.19) | (reads) | same |

</details>

<details><summary>o: 25</summary>

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

<details><summary>q: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L20 q c131 (H2) | 8% / 79% | unexplained (best R2 0.37) | unexplained (best R2 0.49) | (reads) | same |

</details>

<details><summary>up: 26</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L20 up c3 | 49% / 60% | unexplained (best R2 0.33) | unexplained (best R2 0.22) | (reads) | same |
| L20 up c6 | 57% / 29% | unexplained (best R2 0.34) | unexplained (best R2 0.29) | (reads) | same |
| L20 up c9 | 60% / 52% | **res%5** (R2 0.97): res mod 5 in {0..1, 4} | **res%100** (R2 0.51): res mod 100 in {0..1, 4..6, 9..11, 14..16, 19..21, 24..26, 29..31, 34..36, 39..41, 44..46, 49..51, 54..56, 59..61, 64..66, 69..71, 74..76, 79..81, 84..86, 89..91, 94..96, 99} [coarser: res mod 10 in {0..1, 4..6, 9}, R2 0.80] | (reads) | same |
| L20 up c13 | 43% / 33% | **res%10** (R2 0.88): res mod 10 in {5..8} | **res** (R2 0.59): res in {-97, -95, -35..-33, -25..-23, -15..-13, -6..-2, 5..8, 15..18, 25..28, 35..38, 45..48, 55..58, 65..68, 74..78, 85..88, 95..98} | (reads) | same |
| L20 up c14 | 79% / 10% | **res//10** (R2 0.76): (tens) res in {26..53, 81..200} | unexplained (best R2 0.49) | (reads) | same |
| L20 up c15 | 59% / 39% | **res%5** (R2 0.86): res mod 5 in {2..4} | unexplained (best R2 0.45) | (reads) | same |
| L20 up c21 | 40% / 40% | **res%10** (R2 0.90): res mod 10 in {0, 7..9} | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {0,1,2,3,9}; a%10 in {1} -> b%10 in {0,1,2,3}; a%10 in {2} -> b%10 in {1,2,3,4}; a%10 in {3} -> b%10 in {3,4,5,6}; a%10 in {4} -> b%10 in {4,5,6}; a%10 in {5} -> b%10 in {6,7}; a%10 in {6} -> b%10 in {6,7,8}; a%10 in {7} -> b%10 in {0,7,8,9}; a%10 in {8} -> b%10 in {0,1,2,7,8,9}; a%10 in {9} -> b%10 in {0,1,2,8,9} | (reads) | same |
| L20 up c29 | 50% / 53% | **res%100** (R2 0.81): res mod 100 in {0..8, 12, 30..57, 86..99} [coarser: res mod 50 in {0..7, 34..49}, R2 0.83] | **tens(a,b)** (R2 0.57): a//10 in {0} -> b//10 in {0,1}; a//10 in {1} -> b//10 in {1,2,3}; a//10 in {2} -> b//10 in {2,3,4}; a//10 in {3} -> b//10 in {0,3,4,5,6,9}; a//10 in {4,9,10} -> b//10 in {0,4,5,6,9,10}; a//10 in {5} -> b//10 in {0,1,2,4,5,6,7,9,10}; a//10 in {6} -> b//10 in {0,1,2,3,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {2,3,4,6,7,8,9}; a//10 in {8} -> b//10 in {3,4,5,8,9,10} | (reads) | same |
| L20 up c58 | 14% / 0 | **res//10** (R2 0.81): (tens) res in {146..181} | off (on 0) | (reads) | same |
| L20 up c100 | 30% / 23% | **res%10** (R2 0.95): res mod 10 in {3..5} | **res** (R2 0.63): res in {-6..-5, 3..5, 13..15, 23..25, 33..35, 43..45, 53..55, 63..65, 73..75, 83..85, 93..96, 99} | (reads) | same |
| L20 up c153 | 10% / 0 | **res//10** (R2 0.71): (tens) res in {145, 149..170, 173..174} | off (on 0) | (reads) | same |
| L20 up c157 | 14% / 5% | unexplained (best R2 0.37) | unexplained (best R2 0.22) | (reads) | same |
| L20 up c180 | 43% / 44% | **res%100** (R2 0.66): res mod 100 in {11..13, 15..28, 30..33, 57, 60..63, 65..82} [coarser: res mod 50 in {10..13, 15..33}, R2 0.86] | **tens(a,b)** (R2 0.53): a//10 in {2} -> b//10 in {0,1,6}; a//10 in {3} -> b//10 in {0,1,2,6,7,8}; a//10 in {4} -> b//10 in {1,2,3,6,7,8}; a//10 in {5} -> b//10 in {2,3,4,7,8}; a//10 in {6} -> b//10 in {3,4,5,8,9}; a//10 in {7} -> b//10 in {0,4,5,6,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,7,8,10}; a//10 in {9} -> b//10 in {1,2,6,7,8}; a//10 in {10} -> b//10 in {2,3,4,6,7,8,9} | (reads) | same |
| L20 up c223 | 10% / 6% | **res//10** (R2 0.77): (tens) res in {110..120} | **res** (R2 0.68): res in {11..18} | (reads) | same |
| L20 up c242 | 0 / 0 | off (on 0) | same | (reads) | same |
| L20 up c305 | 37% / 10% | **res** (R2 0.66): res in {31..32, 36..37, 41..42, 50..52, 56..63, 66..68, 70..102, 171..172, 176..177, 180..182, 186..187, 190..192} | unexplained (best R2 0.33) | (reads) | same |
| L20 up c378 | 27% / 7% | **tens(a,b)** (R2 0.63): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,3,9,10}; a//10 in {2} -> b//10 in {0,1,2,10}; a//10 in {3,9} -> b//10 in {0,1}; a//10 in {4,5,6,7,10} -> b//10 in {0} | unexplained (best R2 0.29) | (reads) | same |
| L20 up c416 | 35% / 11% | **res%100** (R2 0.49): res mod 100 in {1, 46..48, 50..68, 96, 98..99} | **res%100** (R2 0.63): res mod 100 in {6, 8, 10} | (reads) | same |
| L20 up c446 | 0 / 33% | off (on 0) | **tens(a,b)** (R2 0.53): a//10 in {3} -> b//10 in {7}; a//10 in {4} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {4,7,8,9}; a//10 in {6,7,8,10} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10} | (reads) | same |
| L20 up c484 | 0 / 2% | off (on 0) | unexplained (best R2 0.27) | (reads) | same |
| L20 up c485 | 26% / 15% | unexplained (best R2 0.44) | unexplained (best R2 0.42) | (reads) | same |
| L20 up c564 | 12% / 15% | unexplained (best R2 0.43) | unexplained (best R2 0.47) | (reads) | same |
| L20 up c624 | 8% / 3% | unexplained (best R2 0.19) | unexplained (best R2 0.14) | (reads) | same |
| L20 up c641 | 4% / 4% | unexplained (best R2 0.45) | unexplained (best R2 0.25) | (reads) | same |
| L20 up c653 | 32% / 16% | **res%20** (R2 0.54): res mod 20 in {0, 14..19} | unexplained (best R2 0.34) | (reads) | same |
| L20 up c767 | 16% / 9% | unexplained (best R2 0.38) | unexplained (best R2 0.31) | (reads) | same |

</details>

</details>

<details><summary>layer 21: 96 components with main position `=`</summary>

<details><summary>down: 44</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L21 down c0 | 33% / 26% | **res//10** (R2 0.84): (tens) res in {8..40, 108..140} [coarser: res mod 100 in {8..40}, R2 1.00] | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {2}; a//10 in {1} -> b//10 in {0,10}; a//10 in {2} -> b//10 in {0,1,9,10}; a//10 in {3} -> b//10 in {0,1,2,10}; a//10 in {4} -> b//10 in {1,2,3}; a//10 in {5} -> b//10 in {2,3,4}; a//10 in {6} -> b//10 in {3,4,5}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {5,6,7}; a//10 in {9,10} -> b//10 in {6,7,8} | res: mod100 +15%, mod50 +5% | a: mod50 +2%; res: mod100 +7%, mod50 +6% |
| L21 down c1 | 50% / 57% | **res%2** (R2 0.99): res mod 2 in {1} | **res%2** (R2 0.65): res mod 2 in {1} | res: mod2 +33% | a: mod2 +9%; res: mod2 +34% |
| L21 down c2 | 36% / 28% | **res%20** (R2 0.89): res mod 20 in {10..16} | **res** (R2 0.55): res in {-10..-4, 10..17, 30..36, 50..56, 70..76, 90..96} | res: mod20 +17% | res: mod25 +6%, mod20 +14% |
| L21 down c4 | 41% / 40% | **res%10** (R2 0.85): res mod 10 in {0..3} | **units(a,b)** (R2 0.46): a%10 in {0} -> b%10 in {0,7,8,9}; a%10 in {1} -> b%10 in {0,1,8,9}; a%10 in {2} -> b%10 in {0,1,2,8,9}; a%10 in {3} -> b%10 in {0,1,2,3,9}; a%10 in {4} -> b%10 in {1,2,3,4}; a%10 in {5} -> b%10 in {2,3,4,5,7}; a%10 in {6} -> b%10 in {4,5,6}; a%10 in {7} -> b%10 in {4,5,6,7}; a%10 in {8} -> b%10 in {5,6,7,8}; a%10 in {9} -> b%10 in {6,7,8,9} | res: mod10 +13% | res: mod10 +15% |
| L21 down c7 | 42% / 32% | **res%50** (R2 0.81): res mod 50 in {0..12, 42..49} | **res%100** (R2 0.49): res mod 100 in {0..13, 45, 47, 49..53, 55, 90..99} | res: mod100 +2%, mod50 +10%, mod25 +21% | res: mod100 +2%, mod50 +10%, mod25 +4% |
| L21 down c9 | 18% / 93% | unexplained (best R2 0.46) | **res%100** (R2 0.46): res mod 100 in {0..2, 4..98} | - | res: mod100 +3% |
| L21 down c10 | 11% / 18% | **res%10** (R2 0.93): res mod 10 in {1} | unexplained (best R2 0.50) | res: mod10 +5%, mod5 +6% | res: mod10 +8%, mod5 +10%, mod2 +3% |
| L21 down c13 | 24% / 10% | **res%10** (R2 0.73): res mod 10 in {3..5} | **res** (R2 0.55): res in {-16, -6, 4, 14, 24..25, 34, 44, 54, 64, 74, 84, 94} | res: mod10 +6%, mod5 +3% | res: mod10 +3%, mod5 +2% |
| L21 down c14 | 11% / 10% | **res%10** (R2 0.89): res mod 10 in {5} | **res%10** (R2 0.60): res mod 10 in {5} | res: mod10 +5%, mod5 +5% | res: mod10 +3%, mod5 +4% |
| L21 down c15 | 50% / 29% | **res%20** (R2 0.61): res mod 20 in {0, 4, 13..19} | unexplained (best R2 0.44) | res: mod20 +7% | res: mod25 +3%, mod20 +7% |
| L21 down c16 | 11% / 10% | **res%10** (R2 0.93): res mod 10 in {6} | **units(a,b)** (R2 0.55): a%10 in {0} -> b%10 in {4}; a%10 in {1} -> b%10 in {5}; a%10 in {2} -> b%10 in {6}; a%10 in {4} -> b%10 in {8}; a%10 in {5} -> b%10 in {9}; a%10 in {6} -> b%10 in {0}; a%10 in {7} -> b%10 in {1}; a%10 in {8} -> b%10 in {2}; a%10 in {9} -> b%10 in {3} | res: mod10 +3%, mod5 +4% | res: mod5 +3% |
| L21 down c17 | 78% / 2% | **res** (R2 0.60): res in {4, 8, 14, 16, 18, 20, 24..30, 32..40, 42..50, 54, 56, 58, 60, 63..70, 72, 74, 76..80, 82..90, 94, 96, 98, 100, 104..110, 114, 116, 118..120, 122, 124..200} | unexplained (best R2 0.23) | - | same |
| L21 down c35 | 14% / 3% | **res%100** (R2 0.85): res mod 100 in {57..70} | unexplained (best R2 0.33) | res: mod25 +10% | - |
| L21 down c36 | 35% / 7% | **res//10** (R2 0.79): (tens) res in {60..91, 153..195, 197} [coarser: res mod 100 in {56..95}, R2 0.92] | **tens(a,b)** (R2 0.65): a//10 in {8} -> b//10 in {0,1,2,10}; a//10 in {9} -> b//10 in {1,2}; a//10 in {10} -> b//10 in {1,2,3,4} | res: mod100 +3% | - |
| L21 down c38 | 100% / 100% | always | same | - | same |
| L21 down c42 | 10% / 4% | **res%10** (R2 0.91): res mod 10 in {0} | **res** (R2 0.56): res in {10, 20, 30, 40, 50, 70, 80} | - | same |
| L21 down c43 | 28% / 24% | **res%100** (R2 0.88): res mod 100 in {0..5, 79..99} | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {9,10}; a//10 in {5} -> b//10 in {7}; a//10 in {6} -> b//10 in {6,7,8}; a//10 in {7} -> b//10 in {7,8,9,10}; a//10 in {8} -> b//10 in {0,8,9,10}; a//10 in {9} -> b//10 in {0,1,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,9,10} | res: mod100 +2% | - |
| L21 down c48 | 1% / 10% | unexplained (best R2 0.11) | unexplained (best R2 0.36) | - | b: mod4 +2% |
| L21 down c49 | 16% / 2% | **res//10** (R2 0.70): (tens) res in {40, 44, 132..156, 158, 160} | unexplained (best R2 0.27) | res: mod25 +2% | - |
| L21 down c51 | 1% / 61% | unexplained (best R2 0.10) | **tens(a,b)** (R2 0.68): a//10 in {0,1} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {3,4} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {5,6,7,8,9}; a//10 in {6} -> b//10 in {6,7,8,9}; a//10 in {7} -> b//10 in {1,2,7,8,9,10}; a//10 in {8} -> b//10 in {1,2,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,10}; a//10 in {10} -> b//10 in {0,1,2,10} | - | a: mod100 +2% |
| L21 down c53 | 10% / 6% | **res%10** (R2 0.96): res mod 10 in {9} | **res** (R2 0.60): res in {-99, -11, -1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99} | res: mod5 +2% | - |
| L21 down c57 | 18% / 17% | **res** (R2 0.76): res in {65..71, 106..119} | **res** (R2 0.55): res in {6..21} | res: mod25 +3% | res: mod50 +2% |
| L21 down c63 | 8% / 12% | **res%100** (R2 0.70): res mod 100 in {23..30} | unexplained (best R2 0.49) | res: mod25 +2% | - |
| L21 down c68 | 20% / 10% | **res%10** (R2 0.96): res mod 10 in {0, 8} | **res** (R2 0.52): res in {0, 8, 10, 18, 20, 28, 88, 98} | - | same |
| L21 down c85 | 16% / 4% | **tens(a,b)** (R2 0.58): a//10 in {0} -> b//10 in {0,1,2,3,4,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,10}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3,9,10} -> b//10 in {0} | unexplained (best R2 0.40) | - | same |
| L21 down c92 | 10% / 0 | **res** (R2 0.91): res in {130..143} | off (on 0) | res: mod25 +4% | - |
| L21 down c96 | 12% / 10% | **res%100** (R2 0.56): res mod 100 in {18..20, 58, 76..80, 98} | unexplained (best R2 0.29) | - | same |
| L21 down c99 | 21% / 0 | **res** (R2 0.70): res in {50..53, 96, 136..159, 187..200} | off (on 0) | res: mod25 +3% | - |
| L21 down c105 | 3% / 33% | **res//10** (R2 0.68): (tens) res in {2..25} | **tens(a,b)** (R2 0.72): a//10 in {0} -> b//10 in {0,1,2}; a//10 in {1} -> b//10 in {0,1,2,10}; a//10 in {2} -> b//10 in {0,1,2,3}; a//10 in {3} -> b//10 in {1,2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9}; a//10 in {9,10} -> b//10 in {7,8,9,10} | - | res: mod100 +4%, mod50 +2% |
| L21 down c111 | 3% / 0 | unexplained (best R2 0.44) | off (on 0) | - | same |
| L21 down c124 | 0 / 29% | off (on 0) | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {0}; a//10 in {1} -> b//10 in {1}; a//10 in {2} -> b//10 in {6}; a//10 in {3} -> b//10 in {4,5,6,7,8}; a//10 in {4} -> b//10 in {6,7,8,9}; a//10 in {5} -> b//10 in {5,6,7,8,9,10}; a//10 in {6} -> b//10 in {6,7,8,9,10}; a//10 in {7} -> b//10 in {7,8,9,10}; a//10 in {8} -> b//10 in {8,9,10}; a//10 in {10} -> b//10 in {10} | - | same |
| L21 down c125 | 13% / 0 | **res** (R2 0.67): res in {38, 40, 47..49, 137..151, 158, 198} [coarser: res mod 100 in {37..42, 46..50}, R2 0.82] | off (on 0) | - | same |
| L21 down c133 | 15% / 1% | **res** (R2 0.70): res in {65..75, 161..192} | unexplained (best R2 0.16) | - | same |
| L21 down c206 | 6% / 0 | **res//10** (R2 0.75): (tens) res in {17..18, 22..39} | off (on 0) | - | same |
| L21 down c212 | 1% / 8% | unexplained (best R2 0.20) | **tens(a,b)** (R2 0.58): a//10 in {7} -> b//10 in {0,1,2,3}; a//10 in {8} -> b//10 in {3} | - | same |
| L21 down c231 | 6% / 1% | **res%20** (R2 0.68): res mod 20 in {11} | unexplained (best R2 0.50) | - | same |
| L21 down c267 | 7% / 0 | **tens(a,b)** (R2 0.69): a//10 in {4} -> b//10 in {4}; a//10 in {7} -> b//10 in {10}; a//10 in {8} -> b//10 in {9,10}; a//10 in {9} -> b//10 in {8,9,10}; a//10 in {10} -> b//10 in {7,8,9,10} | off (on 0) | - | same |
| L21 down c279 | 7% / 0 | **res** (R2 0.64): res in {133..138, 157, 174..177, 193..196} | off (on 0) | - | same |
| L21 down c321 | 0 / 0 | off (on 0) | same | - | same |
| L21 down c341 | 0 / 1% | off (on 0) | unexplained (best R2 0.08) | - | same |
| L21 down c390 | 6% / 1% | **res** (R2 0.80): res in {42, 62, 102, 122, 132, 142, 152, 162, 172, 182, 192, 198, 200} | **res%100** (R2 0.53): res mod 100: no class above 0.5 (max 0.44) | - | same |
| L21 down c439 | 0 / 1% | off (on 0) | **tens(a,b)** (R2 0.66): a//10 in {10} -> b//10 in {3,4,5,6,7,8,9} | - | same |
| L21 down c819 | 0 / 0 | off (on 0) | same | - | same |
| L21 down c916 | 8% / 2% | **res%100** (R2 0.70): res mod 100 in {47..48, 67..68, 87..88} | **res** (R2 0.53): res in {7..8} | - | same |

</details>

<details><summary>gate: 29</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L21 gate c0 | 32% / 21% | **res//10** (R2 0.84): (tens) res in {10..35, 108..140} [coarser: res mod 100 in {8..40}, R2 0.96] | **res** (R2 0.82): res in {8..32} | (reads) | same |
| L21 gate c1 | 51% / 32% | **res%2** (R2 0.96): res mod 2 in {1} | **res** (R2 0.64): res in {-99, -97, -95, -21, -11, -9, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99} | (reads) | same |
| L21 gate c2 | 40% / 23% | **res%20** (R2 0.86): res mod 20 in {10..17} | **res** (R2 0.53): res in {-9..-4, 10..17, 31..36, 51..56, 71..75, 91..96} | (reads) | same |
| L21 gate c4 | 10% / 4% | **res%10** (R2 0.98): res mod 10 in {1} | **res** (R2 0.65): res in {1, 11, 21, 31} | (reads) | same |
| L21 gate c7 | 1% / 3% | unexplained (best R2 0.46) | unexplained (best R2 0.50) | (reads) | same |
| L21 gate c10 | 12% / 55% | **res%10** (R2 0.81): res mod 10 in {1} | **res%10** (R2 0.51): res mod 10 in {1, 3, 5, 7, 9} [coarser: res mod 2 in {1}, R2 0.91] | (reads) | same |
| L21 gate c13 | 15% / 5% | **res%10** (R2 0.71): res mod 10 in {4} | **res** (R2 0.62): res in {4, 14, 24, 34, 44, 74, 84, 94} | (reads) | same |
| L21 gate c14 | 10% / 9% | **res%10** (R2 0.94): res mod 10 in {5} | unexplained (best R2 0.50) | (reads) | same |
| L21 gate c15 | 38% / 19% | **res%20** (R2 0.83): res mod 20 in {2..8} | **res** (R2 0.55): res in {2..9, 22..28, 44..45, 84..88} | (reads) | same |
| L21 gate c16 | 21% / 16% | **res%10** (R2 0.96): res mod 10 in {5..6} | **units(a,b)** (R2 0.57): a%10 in {0} -> b%10 in {4,5}; a%10 in {1} -> b%10 in {5,6}; a%10 in {2} -> b%10 in {6}; a%10 in {4} -> b%10 in {8,9}; a%10 in {5} -> b%10 in {0,9}; a%10 in {6} -> b%10 in {0,1}; a%10 in {7} -> b%10 in {1}; a%10 in {8} -> b%10 in {2,3}; a%10 in {9} -> b%10 in {3,4} | (reads) | same |
| L21 gate c24 | 100% / 100% | always | same | (reads) | same |
| L21 gate c33 | 0 / 0 | off (on 0) | same | (reads) | same |
| L21 gate c35 | 20% / 3% | **res** (R2 0.85): res in {57..70, 154..182, 184} [coarser: res mod 100 in {56..74, 78..80}, R2 0.81] | unexplained (best R2 0.36) | (reads) | same |
| L21 gate c36 | 8% / 2% | **tens(a,b)** (R2 0.69): a//10 in {6} -> b//10 in {9,10}; a//10 in {7} -> b//10 in {8,9,10}; a//10 in {8} -> b//10 in {7,8,10}; a//10 in {9} -> b//10 in {6,7}; a//10 in {10} -> b//10 in {6,7,10} | unexplained (best R2 0.40) | (reads) | same |
| L21 gate c43 | 20% / 15% | **res%100** (R2 0.73): res mod 100 in {0..2, 81..84, 94..99} | **tens(a,b)** (R2 0.53): a//10 in {5} -> b//10 in {7}; a//10 in {6} -> b//10 in {6,7,8}; a//10 in {7} -> b//10 in {7,8,9}; a//10 in {8} -> b//10 in {8,9,10}; a//10 in {9,10} -> b//10 in {9,10} | (reads) | same |
| L21 gate c53 | 49% / 29% | **res%2** (R2 0.83): res mod 2 in {0} | **res** (R2 0.57): res in {-98, -96, -94, -92, -90, -88, -86, -84, -82, -80, -78, -76, -74, -64, -4, -2, 0, 2, 4, 6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 32, 34, 36, 38, 40, 42, 44, 46, 48, 54, 56, 58, 60, 62, 64, 66, 68, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98} [coarser: res mod 100 in {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 34, 36, 38, 40, 42, 44, 46, 76, 78, 82, 84, 86, 88, 90, 92, 94, 96, 98}, R2 0.83] | (reads) | same |
| L21 gate c57 | 10% / 0 | **res** (R2 0.77): res in {116..127} | off (on 0) | (reads) | same |
| L21 gate c63 | 0 / 7% | off (on 0) | unexplained (best R2 0.35) | (reads) | same |
| L21 gate c68 | 51% / 36% | **res%2** (R2 0.81): res mod 2 in {0} | unexplained (best R2 0.48) | (reads) | same |
| L21 gate c92 | 1% / 0 | unexplained (best R2 0.37) | off (on 0) | (reads) | same |
| L21 gate c96 | 13% / 17% | **res%100** (R2 0.55): res mod 100 in {18..21, 58..60, 77..80} | unexplained (best R2 0.33) | (reads) | same |
| L21 gate c99 | 44% / 20% | **res%100** (R2 0.69): res mod 100 in {0..1, 30..58, 82..83, 89..99} | unexplained (best R2 0.48) | (reads) | same |
| L21 gate c105 | 17% / 0 | **res** (R2 0.80): res in {126, 130, 132, 134, 136, 140, 142, 144, 146, 152, 154, 156, 160, 162, 164..198, 200} | off (on 0) | (reads) | same |
| L21 gate c133 | 31% / 4% | **res//10** (R2 0.84): (tens) res in {61..90, 159..195} [coarser: res mod 100 in {60..92}, R2 0.94] | unexplained (best R2 0.49) | (reads) | same |
| L21 gate c175 | 31% / 19% | **res%10** (R2 0.96): res mod 10 in {7..9} | **res** (R2 0.61): res in {-97, -23, -13, -3..-2, 7..9, 17..19, 27..29, 37..39, 47..49, 57..59, 67..69, 77..79, 87..89, 97..99} | (reads) | same |
| L21 gate c247 | 1% / 2% | unexplained (best R2 0.40) | **res%100** (R2 0.66): res mod 100 in {0} | (reads) | same |
| L21 gate c279 | 37% / 23% | **res%20** (R2 0.78): res mod 20 in {13..19} | **res** (R2 0.53): res in {-97, -23, -5..-3, 13..19, 33..39, 53..59, 74..79, 93..99} | (reads) | same |
| L21 gate c390 | 12% / 7% | **res%10** (R2 0.80): res mod 10 in {2} | **res** (R2 0.64): res in {-18, 2, 12, 22, 32, 42, 52, 62, 72, 82, 92} | (reads) | same |
| L21 gate c653 | 9% / 1% | **res** (R2 0.57): res in {2..8, 10, 12, 18, 20..25, 41..42, 60..65} | unexplained (best R2 0.15) | (reads) | same |

</details>

<details><summary>up: 22</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L21 up c0 | 35% / 24% | **res//10** (R2 0.83): (tens) res in {8..40, 108..140} [coarser: res mod 100 in {8..40}, R2 0.99] | **tens(a,b)** (R2 0.70): a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,1,10}; a//10 in {3} -> b//10 in {0,1,2}; a//10 in {4} -> b//10 in {1,2,3}; a//10 in {5} -> b//10 in {2,3,4}; a//10 in {6} -> b//10 in {3,4,5}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {5,6,7}; a//10 in {9,10} -> b//10 in {6,7,8} | (reads) | same |
| L21 up c2 | 36% / 9% | **res%20** (R2 0.83): res mod 20 in {11..17} | **res** (R2 0.54): res in {11..16, 33, 95} | (reads) | same |
| L21 up c3 | 100% / 100% | always | same | (reads) | same |
| L21 up c4 | 42% / 34% | **res%10** (R2 0.86): res mod 10 in {0..3} | **res** (R2 0.54): res in {-99, -39..-38, -29..-28, -19..-18, -10..-8, 0..3, 10..13, 20..23, 30..33, 40..43, 50..53, 60..63, 70..73, 80..83, 90..93} | (reads) | same |
| L21 up c7 | 58% / 23% | **res%50** (R2 0.82): res mod 50 in {18..45} | unexplained (best R2 0.47) | (reads) | same |
| L21 up c10 | 11% / 16% | **res%10** (R2 0.91): res mod 10 in {1} | unexplained (best R2 0.49) | (reads) | same |
| L21 up c13 | 31% / 15% | **res%10** (R2 0.92): res mod 10 in {2..4} | **res** (R2 0.64): res in {2..4, 12..14, 22..24, 32..34, 43..44, 54, 64, 74, 83..84, 92..94} | (reads) | same |
| L21 up c14 | 10% / 8% | **res%10** (R2 0.96): res mod 10 in {5} | **res%10** (R2 0.65): res mod 10 in {5} | (reads) | same |
| L21 up c15 | 24% / 6% | **res%20** (R2 0.56): res mod 20 in {4..8} | unexplained (best R2 0.39) | (reads) | same |
| L21 up c16 | 31% / 18% | **res%10** (R2 0.89): res mod 10 in {6..8} | **res** (R2 0.62): res in {-23, -14..-13, -4..-2, 6..8, 16..18, 26..28, 36..38, 46..47, 56..58, 66..67, 76..78, 86..88, 96..98} | (reads) | same |
| L21 up c35 | 17% / 2% | **res%100** (R2 0.80): res mod 100 in {57..73} | unexplained (best R2 0.25) | (reads) | same |
| L21 up c36 | 38% / 12% | **res//10** (R2 0.74): (tens) res in {58, 60..91, 148..150, 152..198, 200} [coarser: res mod 100 in {0, 56..95, 97..98}, R2 0.86] | **tens(a,b)** (R2 0.60): a//10 in {5} -> b//10 in {7}; a//10 in {6} -> b//10 in {8}; a//10 in {7} -> b//10 in {0,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,10}; a//10 in {9} -> b//10 in {1,2}; a//10 in {10} -> b//10 in {1,2,3,4} | (reads) | same |
| L21 up c42 | 31% / 22% | **res%10** (R2 0.97): res mod 10 in {4..6} | **res** (R2 0.69): res in {-25, -15, -6..-4, 4..6, 14..16, 24..26, 34..36, 44..46, 54..56, 64..66, 74..76, 84..86, 94..96} | (reads) | same |
| L21 up c43 | 27% / 33% | **res%100** (R2 0.82): res mod 100 in {0..17, 88..99} | **res%100** (R2 0.61): res mod 100 in {0..19, 85..99} | (reads) | same |
| L21 up c53 | 20% / 8% | **res%10** (R2 0.94): res mod 10 in {6, 9} | **res** (R2 0.55): res in {6, 9, 16, 19, 26, 29, 89, 99} | (reads) | same |
| L21 up c63 | 87% / 17% | **res** (R2 0.63): res in {16..20, 26, 28..30, 33..80, 86..121, 126..130, 132..170, 172..180, 186..200} | unexplained (best R2 0.41) | (reads) | same |
| L21 up c68 | 0 / 5% | off (on 0) | unexplained (best R2 0.34) | (reads) | same |
| L21 up c105 | 19% / 8% | **res//10** (R2 0.74): (tens) res in {127, 130..154, 156..157, 187} | **tens(a,b)** (R2 0.58): a//10 in {4} -> b//10 in {10}; a//10 in {6} -> b//10 in {3}; a//10 in {7} -> b//10 in {3,4}; a//10 in {8} -> b//10 in {4,5}; a//10 in {9} -> b//10 in {6}; a//10 in {10} -> b//10 in {4,5,6,7,8} | (reads) | same |
| L21 up c111 | 0 / 32% | off (on 0) | unexplained (best R2 0.46) | (reads) | same |
| L21 up c125 | 17% / 17% | **res** (R2 0.71): res in {2..4, 38..40, 44, 48..50, 54, 134, 138..150, 154..155, 158..160} [coarser: res mod 100 in {38..45, 48..50, 54..55, 58..60}, R2 0.80] | **tens(a,b)** (R2 0.55): a//10 in {4} -> b//10 in {0,10}; a//10 in {5} -> b//10 in {0,1,10}; a//10 in {6} -> b//10 in {0,1,2,10}; a//10 in {7} -> b//10 in {0,1,2,3}; a//10 in {8} -> b//10 in {3,4}; a//10 in {10} -> b//10 in {4,5,6,10} | (reads) | same |
| L21 up c206 | 19% / 1% | **res%100** (R2 0.51): res mod 100 in {1, 47..49, 57, 96..99} | unexplained (best R2 0.23) | (reads) | same |
| L21 up c341 | 1% / 3% | unexplained (best R2 0.09) | unexplained (best R2 0.38) | (reads) | same |

</details>

<details><summary>v: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L21 v c105 (kv5) | 1% / 54% | unexplained (best R2 0.05) | **tens(a,b)** (R2 0.52): a//10 in {0,1} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {5,6,10} -> b//10 in {1,2}; a//10 in {7} -> b//10 in {0,1,2,3}; a//10 in {8} -> b//10 in {1,2,3}; a//10 in {9} -> b//10 in {1,2,3,4,5} | (reads) | same |

</details>

</details>

<details><summary>layer 22: 82 components with main position `=`</summary>

<details><summary>down: 39</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L22 down c3 | 100% / 100% | always | same | - | same |
| L22 down c5 | 16% / 13% | **res%100** (R2 0.76): res mod 100 in {39..51} | unexplained (best R2 0.45) | res: mod100 +5%, mod50 +4%, mod25 +15% | - |
| L22 down c6 | 16% / 14% | **res%100** (R2 0.88): res mod 100 in {0..1, 3..15} | **res%100** (R2 0.74): res mod 100 in {1, 3..14} | res: mod100 +4%, mod50 +2%, mod25 +11% | res: mod100 +4%, mod50 +5%, mod25 +11%, mod20 +3% |
| L22 down c8 | 10% / 9% | **res%10** (R2 0.98): res mod 10 in {0} | **res%10** (R2 0.52): res mod 10 in {0} | res: mod10 +5%, mod5 +8% | res: mod10 +3%, mod5 +6% |
| L22 down c9 | 10% / 8% | **res%10** (R2 0.94): res mod 10 in {9} | **units(a,b)** (R2 0.50): a%10 in {0} -> b%10 in {1}; a%10 in {1} -> b%10 in {2}; a%10 in {2} -> b%10 in {3}; a%10 in {5} -> b%10 in {6}; a%10 in {6} -> b%10 in {7}; a%10 in {7} -> b%10 in {8}; a%10 in {9} -> b%10 in {0} | res: mod10 +6%, mod5 +7%, mod2 +3% | res: mod10 +2%, mod5 +3% |
| L22 down c11 | 33% / 25% | **res%20** (R2 0.89): res mod 20 in {0..3, 18..19} | **res** (R2 0.51): res in {-99..-98, -41..-38, -22..-18, 0..3, 18..24, 38..43, 58..62, 78..82, 97..99} | res: mod20 +11% | res: mod20 +10% |
| L22 down c17 | 10% / 5% | **res%20** (R2 0.84): res mod 20 in {2..3} | **res** (R2 0.58): res in {-18, 2..3, 22..23, 42, 82} | res: mod20 +3%, mod4 +45% | - |
| L22 down c22 | 22% / 7% | **res** (R2 0.65): res in {3, 5, 7, 11, 21, 31, 41, 51, 61, 67, 71, 75, 77, 81, 91, 101, 107, 111, 113, 115, 117, 121, 123, 125, 127, 131, 133, 135, 137, 141, 143, 145, 147, 151, 153, 161, 163, 167, 171, 173, 175, 177, 181, 191} [coarser: res mod 100 in {1, 3, 5, 7, 11, 15, 17, 21, 27, 31, 41, 51, 61, 67, 71, 73, 75, 77, 81, 91}, R2 0.82] | **res** (R2 0.50): res in {1, 3, 5, 7, 11, 21, 31} | - | same |
| L22 down c27 | 12% / 10% | **res%100** (R2 0.81): res mod 100 in {27..38} | **res** (R2 0.57): res in {27..38} | res: mod25 +3% | - |
| L22 down c28 | 14% / 4% | **res%100** (R2 0.87): res mod 100 in {77..92} | **res** (R2 0.55): res in {77..90} | res: mod25 +6% | - |
| L22 down c29 | 45% / 1% | **res//10** (R2 0.89): (tens) res in {108..200} | **tens(a,b)** (R2 0.51): a//10 in {1,2,3,4,5} -> b//10 in {10} | a: mod100 +4%, mod50 +2%; b: mod100 +5%, mod50 +2% | - |
| L22 down c30 | 7% / 2% | **res%100** (R2 0.71): res mod 100 in {31..36} | unexplained (best R2 0.28) | - | same |
| L22 down c33 | 22% / 10% | **res%100** (R2 0.82): res mod 100 in {1, 86..99} | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {9,10}; a//10 in {7} -> b//10 in {8}; a//10 in {8} -> b//10 in {9}; a//10 in {9} -> b//10 in {0,1,3,4,5,9,10}; a//10 in {10} -> b//10 in {0,1} | res: mod100 +2%, mod25 +4% | - |
| L22 down c35 | 15% / 4% | **res%100** (R2 0.85): res mod 100 in {47..61} | unexplained (best R2 0.39) | res: mod25 +4% | - |
| L22 down c38 | 0 / 2% | off (on 0) | **tens(a,b)** (R2 0.50): a//10 in {4} -> b//10 in {0} | - | same |
| L22 down c44 | 18% / 0 | **res%100** (R2 0.67): res mod 100 in {52, 60..76} | off (on 0) | - | same |
| L22 down c48 | 11% / 3% | **res%100** (R2 0.69): res mod 100 in {8, 18, 27..30, 38, 48, 68, 88} | **res** (R2 0.54): res in {8, 27..29} | res: mod4 +3% | - |
| L22 down c55 | 1% / 32% | unexplained (best R2 0.08) | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {10}; a//10 in {2} -> b//10 in {9,10}; a//10 in {3} -> b//10 in {0,9,10}; a//10 in {4} -> b//10 in {1,2,9,10}; a//10 in {5} -> b//10 in {2,9,10}; a//10 in {6} -> b//10 in {1,2,3,10}; a//10 in {7} -> b//10 in {2,3,4}; a//10 in {8} -> b//10 in {1,2,3,4,5}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7}; a//10 in {10} -> b//10 in {0,1,2,3,5,6,7,10} | - | same |
| L22 down c56 | 10% / 5% | **res** (R2 0.61): res in {125..133, 135..137} | **tens(a,b)** (R2 0.61): a//10 in {6,7,8} -> b//10 in {0}; a//10 in {10} -> b//10 in {2,3,4,7} | - | same |
| L22 down c98 | 8% / 1% | **res%100** (R2 0.70): res mod 100 in {54..60} | unexplained (best R2 0.21) | res: mod25 +2% | - |
| L22 down c101 | 8% / 4% | **res%100** (R2 0.74): res mod 100 in {20..21, 23..27} | **res** (R2 0.53): res in {21, 23..26} | - | same |
| L22 down c107 | 13% / 8% | **res//10** (R2 0.76): (tens) res in {2..49, 200} | **tens(a,b)** (R2 0.73): a//10 in {0,1,2} -> b//10 in {0,1,2} | - | same |
| L22 down c122 | 0 / 1% | off (on 0) | unexplained (best R2 0.09) | - | same |
| L22 down c123 | 7% / 0 | **res%100** (R2 0.78): res mod 100 in {0..1, 96..99} | off (on 0) | - | same |
| L22 down c129 | 10% / 1% | **res%10** (R2 0.94): res mod 10 in {6} | **res%100** (R2 0.60): res mod 100: no class above 0.5 (max 0.47) | - | same |
| L22 down c134 | 20% / 1% | **tens(a,b)** (R2 0.70): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {9,10}; a//10 in {2} -> b//10 in {8,9,10}; a//10 in {3} -> b//10 in {7,8,9,10}; a//10 in {4} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {6}; a//10 in {6} -> b//10 in {4,5}; a//10 in {7} -> b//10 in {3,4,5}; a//10 in {8} -> b//10 in {2,3}; a//10 in {9} -> b//10 in {1,2,3,10}; a//10 in {10} -> b//10 in {0,1,2,3} | unexplained (best R2 0.20) | - | same |
| L22 down c160 | 0 / 37% | off (on 0) | **res//10** (R2 0.63): (tens) res in {-99, -60, -52..0} | - | same |
| L22 down c212 | 10% / 5% | **res%10** (R2 0.97): res mod 10 in {3} | **res** (R2 0.65): res in {3, 13, 23, 33, 43, 83, 93} | - | same |
| L22 down c232 | 1% / 76% | unexplained (best R2 0.05) | **tens(a,b)** (R2 0.52): a//10 in {0,1,2} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {0,1,2,7,8,9}; a//10 in {7} -> b//10 in {0,1,2,3,8,9}; a//10 in {8} -> b//10 in {0,1,2,3,9}; a//10 in {9} -> b//10 in {0,1,2,3,4,5}; a//10 in {10} -> b//10 in {1} | - | a: mod100 +3%, mod50 +2% |
| L22 down c254 | 0 / 2% | off (on 0) | **tens(a,b)** (R2 0.65): a//10 in {6} -> b//10 in {0,10}; a//10 in {7} -> b//10 in {0} | - | same |
| L22 down c257 | 2% / 1% | **res%100** (R2 0.63): res mod 100 in {45..46} | unexplained (best R2 0.27) | - | same |
| L22 down c267 | 6% / 7% | unexplained (best R2 0.37) | **tens(a,b)** (R2 0.59): a//10 in {0,1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {2} | - | same |
| L22 down c297 | 1% / 19% | unexplained (best R2 0.18) | **tens(a,b)** (R2 0.61): a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {3,4,5}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {4,5,6,7}; a//10 in {9} -> b//10 in {7,8}; a//10 in {10} -> b//10 in {3,4,5,6,7,8,9} | - | same |
| L22 down c333 | 10% / 0 | **res%10** (R2 0.91): res mod 10 in {4} | off (on 0) | - | same |
| L22 down c376 | 1% / 22% | **res** (R2 0.81): res in {2..15} | **res%100** (R2 0.62): res mod 100 in {6..14} | - | same |
| L22 down c404 | 4% / 4% | **res//10** (R2 0.58): (tens) res in {180..200} | unexplained (best R2 0.39) | - | same |
| L22 down c423 | 1% / 10% | unexplained (best R2 0.11) | unexplained (best R2 0.37) | - | same |
| L22 down c427 | 7% / 2% | **res%20** (R2 0.78): res mod 20 in {12} | unexplained (best R2 0.46) | res: mod4 +10% | - |
| L22 down c969 | 3% / 0 | **tens(a,b)** (R2 0.46): a//10 in {8,9} -> b//10 in {9,10}; a//10 in {10} -> b//10 in {8,10} | off (on 0) | - | same |

</details>

<details><summary>gate: 20</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L22 gate c0 | 100% / 100% | always | same | (reads) | same |
| L22 gate c5 | 18% / 12% | **res%100** (R2 0.77): res mod 100 in {39..54} | unexplained (best R2 0.46) | (reads) | same |
| L22 gate c6 | 17% / 20% | **res%100** (R2 0.87): res mod 100 in {0..15} | **res//10** (R2 0.63): (tens) res in {0..16} | (reads) | same |
| L22 gate c8 | 10% / 11% | **res%10** (R2 0.98): res mod 10 in {0} | **res%100** (R2 0.52): res mod 100 in {0..1, 10, 20, 30, 40, 50, 60, 70, 80, 90} | (reads) | same |
| L22 gate c9 | 10% / 6% | **res%10** (R2 0.96): res mod 10 in {9} | **res** (R2 0.57): res in {9, 19, 29, 39, 49, 59, 69, 79, 89, 99} | (reads) | same |
| L22 gate c11 | 32% / 26% | **res%20** (R2 0.86): res mod 20 in {0..3, 18..19} | unexplained (best R2 0.49) | (reads) | same |
| L22 gate c17 | 18% / 7% | **res%10** (R2 0.81): res mod 10 in {2..3} | **res** (R2 0.60): res in {2..3, 12..13, 22..23, 43} | (reads) | same |
| L22 gate c22 | 36% / 13% | **res%50** (R2 0.71): res mod 50 in {1, 3, 7, 9, 11, 13, 17, 19, 21, 23, 27, 29, 31, 33, 39, 41, 43, 49} [coarser: res mod 10 in {1, 3, 7, 9}, R2 0.82] | **res** (R2 0.55): res in {1, 3, 5, 7, 9, 11, 13, 19, 21, 23, 29, 31, 39, 69, 79, 89, 99} | (reads) | same |
| L22 gate c28 | 15% / 5% | **res%100** (R2 0.87): res mod 100 in {77..92} | **tens(a,b)** (R2 0.53): a//10 in {8} -> b//10 in {0,10}; a//10 in {9} -> b//10 in {0,1}; a//10 in {10} -> b//10 in {1} | (reads) | same |
| L22 gate c29 | 48% / 1% | **res//10** (R2 0.82): (tens) res in {2, 107..200} | unexplained (best R2 0.43) | (reads) | same |
| L22 gate c33 | 20% / 4% | **res%100** (R2 0.80): res mod 100 in {1, 87..98} | unexplained (best R2 0.46) | (reads) | same |
| L22 gate c35 | 24% / 7% | **res** (R2 0.73): res in {18, 20..28, 64..68, 74..78, 80..88, 122..128, 165..166, 183..186} [coarser: res mod 100 in {21..28, 64..68, 81..86, 88}, R2 0.83] | **res** (R2 0.52): res in {21..28} | (reads) | same |
| L22 gate c48 | 15% / 10% | **res%100** (R2 0.74): res mod 100 in {27..34, 36..38, 48, 68, 88} | **res** (R2 0.52): res in {26..36, 38} | (reads) | same |
| L22 gate c56 | 27% / 4% | **res%100** (R2 0.60): res mod 100 in {16, 51..61, 63..66, 71..77} | unexplained (best R2 0.24) | (reads) | same |
| L22 gate c98 | 7% / 0 | **res%100** (R2 0.80): res mod 100 in {83..89} | off (on 0) | (reads) | same |
| L22 gate c129 | 10% / 3% | **res%10** (R2 0.95): res mod 10 in {6} | **res** (R2 0.67): res in {6, 16, 26} | (reads) | same |
| L22 gate c160 | 0 / 49% | off (on 0) | **tens(a,b)** (R2 0.57): a//10 in {1} -> b//10 in {1,2,3}; a//10 in {2} -> b//10 in {2,4,5,6}; a//10 in {3} -> b//10 in {2,3,4,5,6,7,8}; a//10 in {4} -> b//10 in {3,4,5,6,7,8}; a//10 in {5,6,10} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {7,8} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10} | (reads) | same |
| L22 gate c212 | 14% / 4% | **res%20** (R2 0.85): res mod 20 in {0, 8, 10} | **res** (R2 0.52): res in {10, 20, 30} | (reads) | same |
| L22 gate c257 | 10% / 3% | **res%100** (R2 0.84): res mod 100 in {41..49} | unexplained (best R2 0.43) | (reads) | same |
| L22 gate c427 | 10% / 3% | **res%10** (R2 0.90): res mod 10 in {2} | **res** (R2 0.64): res in {2, 12, 22} | (reads) | same |

</details>

<details><summary>o: 6</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L22 o c10 (H15) | 29% / 19% | **tens(a,b)** (R2 0.78): a//10 in {0,1,2} -> b//10 in {8,9,10}; a//10 in {3,4,5,6,7} -> b//10 in {9,10}; a//10 in {8} -> b//10 in {0,2,3,7,8,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.82): a//10 in {0,1,2,3,4} -> b//10 in {9,10}; a//10 in {5,6,8} -> b//10 in {10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | - | same |
| L22 o c115 (H14) | 0 / 0 | off (on 0) | same | - | same |
| L22 o c147 (H28) | 0 / 0 | off (on 0) | same | - | same |
| L22 o c200 (H15) | 13% / 1% | **tens(a,b)** (R2 0.59): a//10 in {0,1,2,3,4,8} -> b//10 in {9,10}; a//10 in {5,6,7} -> b//10 in {10}; a//10 in {9} -> b//10 in {3,4,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.62): a//10 in {0,1,2,3,4,5,6} -> b//10 in {10}; a//10 in {10} -> b//10 in {1,10} | - | same |
| L22 o c246 (H3) | 3% / 6% | unexplained (best R2 0.49) | **a** (R2 0.69): a in {37, 47, 57, 67, 97} | - | same |
| L22 o c352 (H3) | 1% / 2% | **a** (R2 0.54): a in {97} | **a** (R2 0.64): a in {97} | - | same |

</details>

<details><summary>up: 17</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L22 up c5 | 42% / 13% | **res%100** (R2 0.66): res mod 100 in {31..59, 61} | **tens(a,b)** (R2 0.51): a//10 in {4} -> b//10 in {0,4,5,9,10}; a//10 in {5} -> b//10 in {0,1,9,10}; a//10 in {6} -> b//10 in {1}; a//10 in {7} -> b//10 in {3}; a//10 in {8} -> b//10 in {3,4}; a//10 in {9} -> b//10 in {4,5}; a//10 in {10} -> b//10 in {4,5,6} | (reads) | same |
| L22 up c6 | 17% / 14% | **res%100** (R2 0.88): res mod 100 in {0..1, 4, 6..15} | **res%100** (R2 0.72): res mod 100 in {6..15} | (reads) | same |
| L22 up c8 | 10% / 10% | **res%10** (R2 0.98): res mod 10 in {0} | **res%10** (R2 0.51): res mod 10 in {0} | (reads) | same |
| L22 up c9 | 10% / 8% | **res%10** (R2 0.94): res mod 10 in {9} | **units(a,b)** (R2 0.51): a%10 in {0} -> b%10 in {1}; a%10 in {1} -> b%10 in {2}; a%10 in {2} -> b%10 in {3}; a%10 in {5} -> b%10 in {6}; a%10 in {6} -> b%10 in {7}; a%10 in {7} -> b%10 in {8}; a%10 in {9} -> b%10 in {0} | (reads) | same |
| L22 up c11 | 33% / 6% | **res%20** (R2 0.90): res mod 20 in {0..4, 18..19} | unexplained (best R2 0.44) | (reads) | same |
| L22 up c17 | 12% / 8% | **res%20** (R2 0.78): res mod 20 in {2..3} | **res%100** (R2 0.54): res mod 100 in {82} | (reads) | same |
| L22 up c27 | 1% / 0 | **res%100** (R2 0.82): res mod 100 in {22} | off (on 0) | (reads) | same |
| L22 up c28 | 22% / 10% | **res%100** (R2 0.84): res mod 100 in {77..99} | **tens(a,b)** (R2 0.55): a//10 in {5} -> b//10 in {7}; a//10 in {6} -> b//10 in {7,8}; a//10 in {7} -> b//10 in {8}; a//10 in {8} -> b//10 in {0,9,10}; a//10 in {9} -> b//10 in {0,1,9,10}; a//10 in {10} -> b//10 in {0,1,2,10} | (reads) | same |
| L22 up c33 | 35% / 9% | **res%100** (R2 0.68): res mod 100 in {0..1, 12..14, 32, 51..55, 71..74, 87..99} | unexplained (best R2 0.50) | (reads) | same |
| L22 up c35 | 36% / 6% | **res%100** (R2 0.79): res mod 100 in {22..26, 42, 60..88} | unexplained (best R2 0.43) | (reads) | same |
| L22 up c56 | 31% / 20% | **res//10** (R2 0.80): (tens) res in {22..38, 111..145} [coarser: res mod 100 in {14..17, 19..41}, R2 0.81] | **res//10** (R2 0.65): (tens) res in {11..19, 21..38} | (reads) | same |
| L22 up c98 | 25% / 4% | **res%100** (R2 0.72): res mod 100 in {50..77} | unexplained (best R2 0.35) | (reads) | same |
| L22 up c106 | 1% / 95% | unexplained (best R2 0.08) | always | (reads) | same |
| L22 up c121 | 29% / 21% | **res%10** (R2 0.88): res mod 10 in {0..1, 9} | **res** (R2 0.52): res in {-99, -40..-39, -31..-30, -21..-19, -10, -1..2, 9..10, 19..21, 29..30, 39..41, 50, 59..61, 79..80, 99} | (reads) | same |
| L22 up c151 | 96% / 96% | always | same | (reads) | same |
| L22 up c276 | 28% / 11% | **res%10** (R2 0.83): res mod 10 in {7..9} | **res** (R2 0.53): res in {-99, 8..9, 18..19, 27..29, 39, 49, 59, 67..69, 78..79, 87..89, 97..99} | (reads) | same |
| L22 up c446 | 25% / 18% | **res%100** (R2 0.78): res mod 100 in {1, 4, 6..9, 11, 14..16, 94..97, 99} | **res//10** (R2 0.57): (tens) res in {-99, -97, -95, 1..17, 94..99} | (reads) | same |

</details>

</details>

<details><summary>layer 23: 120 components with main position `=`</summary>

<details><summary>down: 47</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L23 down c3 | 15% / 17% | **res%100** (R2 0.88): res mod 100 in {0..1, 90..99} | unexplained (best R2 0.45) | res: mod100 +5%, mod50 +4%, mod25 +12% | - |
| L23 down c4 | 11% / 16% | **res%10** (R2 0.91): res mod 10 in {2} | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {8}; a%10 in {1} -> b%10 in {9}; a%10 in {2} -> b%10 in {0,2,4,6,8}; a%10 in {3} -> b%10 in {1}; a%10 in {4} -> b%10 in {2}; a%10 in {5} -> b%10 in {3,7}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {5}; a%10 in {8} -> b%10 in {6}; a%10 in {9} -> b%10 in {7} | res: mod10 +6%, mod5 +6%, mod2 +3% | res: mod10 +5%, mod5 +8%, mod2 +3% |
| L23 down c5 | 13% / 21% | **res%100** (R2 0.79): res mod 100 in {32..43} | unexplained (best R2 0.46) | res: mod100 +2%, mod25 +6% | res: mod50 +2%, mod25 +3% |
| L23 down c7 | 39% / 14% | **res//10** (R2 0.75): (tens) res in {7..8, 15, 18, 20..88} | unexplained (best R2 0.41) | - | same |
| L23 down c8 | 0 / 11% | off (on 0) | **tens(a,b)** (R2 0.62): a//10 in {5,6} -> b//10 in {0,1}; a//10 in {7} -> b//10 in {0,1,2,3}; a//10 in {8} -> b//10 in {0,1,3}; a//10 in {10} -> b//10 in {2,3} | - | same |
| L23 down c9 | 10% / 14% | **res%10** (R2 0.95): res mod 10 in {6} | **units(a,b)** (R2 0.53): a%10 in {0} -> b%10 in {4}; a%10 in {1} -> b%10 in {5}; a%10 in {2} -> b%10 in {6}; a%10 in {4} -> b%10 in {0,8}; a%10 in {5} -> b%10 in {9}; a%10 in {6} -> b%10 in {0}; a%10 in {7} -> b%10 in {1}; a%10 in {8} -> b%10 in {2,4}; a%10 in {9} -> b%10 in {3} | res: mod10 +3%, mod5 +5%, mod2 +2% | res: mod10 +3%, mod5 +5%, mod2 +2% |
| L23 down c10 | 19% / 97% | **res//10** (R2 0.63): (tens) res in {2..4, 6, 8..59, 61..63} | always | - | same |
| L23 down c12 | 2% / 0 | **res** (R2 0.62): res in {114, 124, 134} | off (on 0) | - | same |
| L23 down c13 | 7% / 2% | **tens(a,b)** (R2 0.64): a//10 in {0,1,2,5,7} -> b//10 in {10}; a//10 in {8} -> b//10 in {9,10}; a//10 in {9} -> b//10 in {8,9,10}; a//10 in {10} -> b//10 in {0,6,7,8,9,10} | unexplained (best R2 0.45) | - | same |
| L23 down c14 | 11% / 8% | **res%10** (R2 0.93): res mod 10 in {4} | **res%100** (R2 0.58): res mod 100 in {4, 14, 24, 34, 44, 54, 64, 74, 84, 94} [coarser: res mod 50 in {4, 14, 24, 34, 44}, R2 0.80] | res: mod10 +4%, mod5 +5%, mod2 +2% | res: mod10 +3%, mod5 +3% |
| L23 down c15 | 12% / 10% | **res%100** (R2 0.78): res mod 100 in {30..35, 38..42} | unexplained (best R2 0.38) | res: mod25 +6% | - |
| L23 down c16 | 18% / 19% | **res%100** (R2 0.91): res mod 100 in {70..86} | **tens(a,b)** (R2 0.55): a//10 in {0,6} -> b//10 in {7,8}; a//10 in {4} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {7}; a//10 in {7} -> b//10 in {0,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,10}; a//10 in {9,10} -> b//10 in {1,2} | res: mod100 +3%, mod50 +3%, mod25 +5% | - |
| L23 down c17 | 15% / 11% | **res%100** (R2 0.86): res mod 100 in {64..76} | unexplained (best R2 0.40) | res: mod100 +2%, mod50 +3%, mod25 +4% | - |
| L23 down c18 | 27% / 18% | **res//10** (R2 0.75): (tens) res in {73..102, 190..191} | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {7,8,9,10}; a//10 in {7} -> b//10 in {8}; a//10 in {8} -> b//10 in {0,1,2,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,9,10}; a//10 in {10} -> b//10 in {0,1,2} | - | same |
| L23 down c20 | 18% / 12% | **res//10** (R2 0.80): (tens) res in {25..36, 121..139} [coarser: res mod 100 in {21..37}, R2 0.91] | **res** (R2 0.56): res in {23..36} | res: mod100 +2%, mod50 +2%, mod25 +3% | - |
| L23 down c21 | 12% / 0 | **res** (R2 0.60): res in {65, 85..90, 95, 105, 107, 115, 125, 145, 185} | off (on 0) | - | same |
| L23 down c24 | 10% / 7% | **res%10** (R2 0.96): res mod 10 in {7} | **res** (R2 0.67): res in {-97, -23, -13, -3, 7, 17, 27, 37, 47, 57, 67, 77, 87, 97} | res: mod10 +3%, mod5 +3% | res: mod5 +3% |
| L23 down c28 | 100% / 100% | always | same | - | same |
| L23 down c29 | 58% / 3% | **res//10** (R2 0.85): (tens) res in {92..199} | unexplained (best R2 0.45) | res: mod100 +2% | - |
| L23 down c32 | 10% / 1% | unexplained (best R2 0.42) | unexplained (best R2 0.16) | - | same |
| L23 down c38 | 6% / 8% | unexplained (best R2 0.16) | unexplained (best R2 0.29) | - | same |
| L23 down c41 | 15% / 30% | **res//10** (R2 0.79): (tens) res in {6..59} | **res//10** (R2 0.53): (tens) res in {2..31} | - | res: mod100 +2% |
| L23 down c55 | 14% / 22% | unexplained (best R2 0.49) | **units(a,b)** (R2 0.81): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} | a: mod2 +3%; b: mod2 +3% | a: mod2 +13%; b: mod2 +13%; res: mod2 +5% |
| L23 down c57 | 0 / 0 | off (on 0) | same | - | same |
| L23 down c58 | 15% / 25% | **res//10** (R2 0.83): (tens) res in {3..56, 200} | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {0,1,2,3}; a//10 in {1,2,3} -> b//10 in {0,1,2,3,4}; a//10 in {4} -> b//10 in {0,1,2}; a//10 in {5} -> b//10 in {2} | - | same |
| L23 down c60 | 4% / 0 | **res** (R2 0.88): res in {120, 130, 140, 150, 160, 170, 180, 190, 200} | off (on 0) | - | same |
| L23 down c65 | 0 / 19% | off (on 0) | unexplained (best R2 0.26) | - | same |
| L23 down c77 | 6% / 4% | **res%100** (R2 0.69): res mod 100 in {1, 21, 41, 61, 81} | unexplained (best R2 0.43) | res: mod4 +4% | - |
| L23 down c105 | 8% / 1% | **res%20** (R2 0.73): res mod 20 in {3, 13} | **res%100** (R2 0.48): res mod 100: no class above 0.5 (max 0.38) | - | same |
| L23 down c106 | 9% / 2% | **res%10** (R2 0.84): res mod 10 in {9} | **res** (R2 0.66): res in {9, 19, 29, 89, 99} | - | same |
| L23 down c110 | 0 / 22% | off (on 0) | **tens(a,b)** (R2 0.56): a//10 in {3,4} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {6,7,8}; a//10 in {6} -> b//10 in {6,7,8,9,10}; a//10 in {7} -> b//10 in {7,8,9,10}; a//10 in {8} -> b//10 in {8,9,10}; a//10 in {9} -> b//10 in {7,9,10} | - | same |
| L23 down c113 | 0 / 0 | off (on 0) | same | - | same |
| L23 down c123 | 4% / 0 | **res** (R2 0.54): res in {92, 101..102, 151..152} | off (on 0) | - | same |
| L23 down c125 | 6% / 3% | **res%20** (R2 0.83): res mod 20 in {11} | **res** (R2 0.52): res in {11, 31, 91} | - | same |
| L23 down c130 | 28% / 0 | **res//10** (R2 0.83): (tens) res in {128..200} | off (on 0) | - | same |
| L23 down c141 | 2% / 1% | **res%100** (R2 0.61): res mod 100 in {28, 88} | unexplained (best R2 0.40) | - | same |
| L23 down c196 | 1% / 8% | unexplained (best R2 0.28) | **res%100** (R2 0.76): res mod 100: no class above 0.5 (max 0.46) | - | same |
| L23 down c202 | 9% / 9% | **res%100** (R2 0.86): res mod 100 in {13..20} | **res** (R2 0.63): res in {13..20} | - | same |
| L23 down c205 | 10% / 4% | **res%10** (R2 0.96): res mod 10 in {8} | **res** (R2 0.61): res in {8, 18, 28, 38, 48, 88, 98} | - | same |
| L23 down c216 | 13% / 1% | **res%100** (R2 0.70): res mod 100 in {60..70, 84..86, 88} | unexplained (best R2 0.25) | res: mod25 +2% | - |
| L23 down c257 | 0 / 0 | off (on 0) | same | - | same |
| L23 down c269 | 6% / 0 | **res** (R2 0.80): res in {112..118} | off (on 0) | - | same |
| L23 down c310 | 4% / 0 | **res** (R2 0.66): res in {83, 93, 123, 143, 153, 163, 183, 193} [coarser: res mod 100 in {83, 93}, R2 0.82] | off (on 0) | - | same |
| L23 down c387 | 4% / 4% | **units(a,b)** (R2 0.98): a%10 in {0,5} -> b%10 in {0,5} | **units(a,b)** (R2 0.88): a%10 in {0,5} -> b%10 in {0,5} | - | same |
| L23 down c459 | 7% / 3% | **res%50** (R2 0.74): res mod 50 in {5, 15, 25, 35} [coarser: res mod 10 in {5}, R2 0.85] | **res** (R2 0.66): res in {5, 15, 25} | - | same |
| L23 down c465 | 3% / 29% | unexplained (best R2 0.43) | **tens(a,b)** (R2 0.60): a//10 in {0} -> b//10 in {0,1,2,3,4,5,9,10}; a//10 in {1} -> b//10 in {0,1,2,3,10}; a//10 in {2} -> b//10 in {1,2,3}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3,4}; a//10 in {5} -> b//10 in {4,5}; a//10 in {6} -> b//10 in {5,6,7}; a//10 in {7} -> b//10 in {6,7,8}; a//10 in {8} -> b//10 in {7,8,9}; a//10 in {9} -> b//10 in {8,9,10}; a//10 in {10} -> b//10 in {9,10} | - | same |
| L23 down c931 | 5% / 0 | **res** (R2 0.81): res in {151..158, 161..162} | off (on 0) | - | same |

</details>

<details><summary>gate: 31</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L23 gate c3 | 19% / 27% | **res%100** (R2 0.81): res mod 100 in {0..7, 89..99} | **tens(a,b)** (R2 0.56): a//10 in {0} -> b//10 in {0,1,8,9,10}; a//10 in {1} -> b//10 in {1,10}; a//10 in {2} -> b//10 in {2}; a//10 in {4} -> b//10 in {4,9,10}; a//10 in {5} -> b//10 in {5,6,10}; a//10 in {6} -> b//10 in {6,7,8,9,10}; a//10 in {7} -> b//10 in {7,8,9,10}; a//10 in {8} -> b//10 in {8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,9,10}; a//10 in {10} -> b//10 in {0,9,10} | (reads) | same |
| L23 gate c4 | 26% / 9% | **res%50** (R2 0.69): res mod 50 in {1..3, 11..13, 21..23, 31..32, 41..42} | **res** (R2 0.58): res in {-38, -28, -8, 2, 12, 22, 32, 42, 52, 62, 72, 82, 92} | (reads) | same |
| L23 gate c5 | 14% / 23% | **res%100** (R2 0.80): res mod 100 in {32..43} | unexplained (best R2 0.48) | (reads) | same |
| L23 gate c8 | 100% / 100% | always | same | (reads) | same |
| L23 gate c9 | 10% / 14% | **res%10** (R2 0.98): res mod 10 in {6} | **units(a,b)** (R2 0.55): a%10 in {0} -> b%10 in {4}; a%10 in {1} -> b%10 in {5}; a%10 in {2} -> b%10 in {6}; a%10 in {4} -> b%10 in {0,4,8}; a%10 in {5} -> b%10 in {9}; a%10 in {6} -> b%10 in {0}; a%10 in {7} -> b%10 in {1}; a%10 in {8} -> b%10 in {2,4}; a%10 in {9} -> b%10 in {3,5} | (reads) | same |
| L23 gate c14 | 11% / 8% | **res%10** (R2 0.93): res mod 10 in {4} | **res%100** (R2 0.57): res mod 100 in {4, 14, 24, 34, 44, 64, 74, 84, 94} | (reads) | same |
| L23 gate c15 | 17% / 12% | **res%100** (R2 0.90): res mod 100 in {26..41} | **res** (R2 0.64): res in {26..41} | (reads) | same |
| L23 gate c16 | 34% / 29% | **res%100** (R2 0.91): res mod 100 in {58..91} | **tens(a,b)** (R2 0.61): a//10 in {0,5} -> b//10 in {7,8}; a//10 in {3,4} -> b//10 in {6,7}; a//10 in {6} -> b//10 in {0,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,7,10}; a//10 in {10} -> b//10 in {1,2,3,4} | (reads) | same |
| L23 gate c17 | 17% / 4% | **res%100** (R2 0.94): res mod 100 in {64..80} | unexplained (best R2 0.36) | (reads) | same |
| L23 gate c18 | 29% / 13% | **res** (R2 0.90): res in {73..104, 191..199} | **tens(a,b)** (R2 0.62): a//10 in {0} -> b//10 in {8,9,10}; a//10 in {8} -> b//10 in {0,1,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,9,10}; a//10 in {10} -> b//10 in {0,1,2} | (reads) | same |
| L23 gate c20 | 28% / 13% | **res%100** (R2 0.78): res mod 100 in {23..41, 83..91} | **res** (R2 0.54): res in {23..35, 83..89} | (reads) | same |
| L23 gate c24 | 19% / 10% | **res%10** (R2 0.93): res mod 10 in {4, 7} | **res** (R2 0.64): res in {-97, -3, 4, 7, 14, 17, 24, 27, 37, 47, 57, 67, 77, 84, 87, 97} | (reads) | same |
| L23 gate c38 | 3% / 5% | unexplained (best R2 0.13) | unexplained (best R2 0.21) | (reads) | same |
| L23 gate c41 | 16% / 50% | **res//10** (R2 0.80): (tens) res in {2..59} | **res//10** (R2 0.59): (tens) res in {-29, -27..31} | (reads) | same |
| L23 gate c55 | 23% / 25% | **units(a,b)** (R2 0.75): a%10 in {0} -> b%10 in {0,2,8}; a%10 in {1} -> b%10 in {1}; a%10 in {2,4,8} -> b%10 in {0,2,4,6,8}; a%10 in {3} -> b%10 in {9}; a%10 in {5} -> b%10 in {7}; a%10 in {6} -> b%10 in {0,2,6,8}; a%10 in {7} -> b%10 in {5}; a%10 in {9} -> b%10 in {3} | **units(a,b)** (R2 0.77): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8}; a%10 in {3} -> b%10 in {1} | (reads) | same |
| L23 gate c58 | 11% / 4% | **res%100** (R2 0.84): res mod 100 in {0..1, 92..99} | unexplained (best R2 0.23) | (reads) | same |
| L23 gate c77 | 13% / 4% | **res%100** (R2 0.69): res mod 100 in {1, 21, 59..63, 77..82} | unexplained (best R2 0.35) | (reads) | same |
| L23 gate c105 | 10% / 17% | **res%10** (R2 0.98): res mod 10 in {2} | **units(a,b)** (R2 0.51): a%10 in {0} -> b%10 in {8}; a%10 in {1} -> b%10 in {9}; a%10 in {2} -> b%10 in {0,4,6,8}; a%10 in {3} -> b%10 in {1}; a%10 in {4} -> b%10 in {2,6,8}; a%10 in {5} -> b%10 in {3,7}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {5}; a%10 in {8} -> b%10 in {6}; a%10 in {9} -> b%10 in {7} | (reads) | same |
| L23 gate c106 | 24% / 9% | **res%10** (R2 0.77): res mod 10 in {0, 9} | **res** (R2 0.57): res in {-99, 0, 9..10, 19..20, 29..30, 89..90, 99} | (reads) | same |
| L23 gate c123 | 10% / 5% | **res%10** (R2 0.95): res mod 10 in {2} | **res** (R2 0.64): res in {2, 12, 22, 32, 72, 82, 92} | (reads) | same |
| L23 gate c141 | 19% / 8% | **res%10** (R2 0.93): res mod 10 in {4, 7} | **res** (R2 0.61): res in {4, 7, 14, 17, 24, 27, 37, 87, 97} | (reads) | same |
| L23 gate c196 | 1% / 11% | **res//10** (R2 0.59): (tens) res in {2..10} | **res//10** (R2 0.66): (tens) res in {1..10} | (reads) | same |
| L23 gate c202 | 11% / 13% | **res%100** (R2 0.84): res mod 100 in {11..21} | **res** (R2 0.71): res in {9..21} | (reads) | same |
| L23 gate c216 | 76% / 8% | **res** (R2 0.67): res in {45..50, 55..61, 63..71, 73..81, 83..91, 98, 100..198, 200} | unexplained (best R2 0.40) | (reads) | same |
| L23 gate c257 | 7% / 2% | **res** (R2 0.77): res in {57, 97, 117, 137, 143, 147, 153, 157, 163, 167, 173, 177, 183, 187, 193, 197} | **res** (R2 0.61): res in {7, 17} | (reads) | same |
| L23 gate c310 | 10% / 4% | **res%10** (R2 0.91): res mod 10 in {2} | **res** (R2 0.71): res in {2, 12, 22, 32, 42, 52, 72, 82, 92} | (reads) | same |
| L23 gate c439 | 5% / 1% | **res%100** (R2 0.67): res mod 100 in {38..39, 58..59} | unexplained (best R2 0.35) | (reads) | same |
| L23 gate c459 | 17% / 5% | **res%10** (R2 0.84): res mod 10 in {4..5} | **res** (R2 0.63): res in {4, 14, 24, 34, 84, 94} | (reads) | same |
| L23 gate c606 | 10% / 4% | **res%10** (R2 0.99): res mod 10 in {6} | **res** (R2 0.67): res in {6, 16, 26, 36, 96} | (reads) | same |
| L23 gate c738 | 10% / 2% | **res%10** (R2 0.97): res mod 10 in {6} | **res** (R2 0.73): res in {6, 16, 26, 96} | (reads) | same |
| L23 gate c784 | 10% / 2% | **res%10** (R2 0.96): res mod 10 in {6} | **res** (R2 0.74): res in {6, 16, 96} | (reads) | same |

</details>

<details><summary>o: 13</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L23 o c152 (H22) | 4% / 9% | unexplained (best R2 0.50) | unexplained (best R2 0.46) | - | same |
| L23 o c187 (H7) | 1% / 4% | **a** (R2 0.70): a in {98} | **a** (R2 0.64): a in {97..99} | - | same |
| L23 o c234 (H22) | 2% / 5% | unexplained (best R2 0.38) | same | - | same |
| L23 o c280 (H7) | 1% / 3% | unexplained (best R2 0.41) | **a** (R2 0.56): a in {8, 88} | - | same |
| L23 o c299 (H22) | 0 / 0 | off (on 0) | same | - | same |
| L23 o c308 (H15) | 96% / 91% | always | unexplained (best R2 0.41) | - | same |
| L23 o c311 (H15) | 0 / 0 | off (on 0) | same | - | same |
| L23 o c370 (H22) | 1% / 1% | **a** (R2 0.82): a in {55} | **a** (R2 0.67): a in {55} | - | same |
| L23 o c394 (H22) | 6% / 9% | unexplained (best R2 0.35) | unexplained (best R2 0.34) | - | same |
| L23 o c402 (H22) | 0 / 0 | off (on 0) | same | - | same |
| L23 o c420 (H22) | 0 / 0 | off (on 0) | same | - | same |
| L23 o c435 (H22) | 1% / 2% | **a** (R2 0.54): a in {65} | unexplained (best R2 0.48) | - | same |
| L23 o c456 (H22) | 1% / 3% | unexplained (best R2 0.21) | unexplained (best R2 0.39) | - | same |

</details>

<details><summary>up: 29</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L23 up c3 | 27% / 8% | **res%100** (R2 0.74): res mod 100 in {33..40, 55..58, 74..79, 93..99} | unexplained (best R2 0.42) | (reads) | same |
| L23 up c4 | 34% / 12% | **res** (R2 0.80): res in {8, 12, 22, 28, 32, 38, 42, 48, 52, 58, 62, 68, 72, 78, 82, 88, 92, 94, 98, 102, 108, 112, 118, 122, 128, 132, 134, 138, 142..144, 146..200} | **units(a,b)** (R2 0.49): a%10 in {0} -> b%10 in {2,8}; a%10 in {2} -> b%10 in {0}; a%10 in {3} -> b%10 in {1}; a%10 in {4} -> b%10 in {2}; a%10 in {5} -> b%10 in {3,7}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {5}; a%10 in {8} -> b%10 in {6}; a%10 in {9} -> b%10 in {7} | (reads) | same |
| L23 up c5 | 33% / 7% | **res//10** (R2 0.76): (tens) res in {43..51, 53..71, 140..190} [coarser: res mod 100 in {41..73}, R2 0.83] | unexplained (best R2 0.34) | (reads) | same |
| L23 up c9 | 22% / 98% | **res** (R2 0.62): res in {2..29, 32, 34..39, 41, 46, 56, 66, 76, 85..86, 95..96, 106, 116, 126, 136, 146, 156, 166, 176, 186, 196} | always | (reads) | same |
| L23 up c14 | 20% / 12% | **res%10** (R2 0.98): res mod 10 in {2, 4} | **res** (R2 0.63): res in {2, 4, 12, 14, 22, 24, 32, 34, 42, 44, 52, 54, 62, 64, 72, 74, 82, 84, 92, 94} | (reads) | same |
| L23 up c15 | 18% / 18% | **res** (R2 0.75): res in {19..21, 35..45, 47..50, 59..61, 79..81, 136..143} | unexplained (best R2 0.36) | (reads) | same |
| L23 up c16 | 33% / 28% | **res%100** (R2 0.69): res mod 100 in {1..11, 44..70} | unexplained (best R2 0.44) | (reads) | same |
| L23 up c17 | 28% / 20% | **res%100** (R2 0.89): res mod 100 in {59..85} | **tens(a,b)** (R2 0.59): a//10 in {0,4} -> b//10 in {7}; a//10 in {3} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {7,8}; a//10 in {6} -> b//10 in {0,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,9,10}; a//10 in {8} -> b//10 in {0,1,2}; a//10 in {9} -> b//10 in {1,2}; a//10 in {10} -> b//10 in {2,3} | (reads) | same |
| L23 up c18 | 8% / 0 | **res//10** (R2 0.77): (tens) res in {160, 163..200} | unexplained (best R2 0.31) | (reads) | same |
| L23 up c20 | 12% / 14% | **res%100** (R2 0.76): res mod 100 in {27..34, 70..71} | unexplained (best R2 0.38) | (reads) | same |
| L23 up c24 | 20% / 12% | **res%10** (R2 0.78): res mod 10 in {7..8} | **res** (R2 0.64): res in {-97, -23, -13, -3, 7..8, 17..18, 27..28, 37..39, 47, 57, 67, 77, 87..88, 97..99} | (reads) | same |
| L23 up c29 | 71% / 9% | **res//10** (R2 0.91): (tens) res in {79..200} | **tens(a,b)** (R2 0.55): a//10 in {0,1,2} -> b//10 in {9,10}; a//10 in {9} -> b//10 in {0,1,9}; a//10 in {10} -> b//10 in {0,1,2,10} | (reads) | same |
| L23 up c35 | 10% / 4% | **res%10** (R2 0.94): res mod 10 in {6} | **res** (R2 0.57): res in {6, 16, 26, 36, 86, 96} | (reads) | same |
| L23 up c38 | 6% / 7% | unexplained (best R2 0.17) | unexplained (best R2 0.27) | (reads) | same |
| L23 up c41 | 7% / 12% | **res%100** (R2 0.61): res mod 100 in {9} | **res%100** (R2 0.65): res mod 100: no class above 0.5 (max 0.49) | (reads) | same |
| L23 up c77 | 12% / 8% | **res%20** (R2 0.70): res mod 20 in {1, 11} [coarser: res mod 10 in {1}, R2 0.81] | **res** (R2 0.54): res in {-19, 1..2, 11, 21..23, 41, 61, 81, 91} | (reads) | same |
| L23 up c125 | 17% / 7% | **res%10** (R2 0.79): res mod 10 in {3..4} | **res** (R2 0.57): res in {3..4, 14, 23..24, 34, 84, 94} | (reads) | same |
| L23 up c205 | 42% / 15% | **res** (R2 0.71): res in {2..35, 37, 39..41, 43..45, 47, 49..51, 53, 55, 57, 59, 63, 65, 67, 73, 75, 77..95, 97..101, 103, 105, 107, 183..191, 193, 197} | **tens(a,b)** (R2 0.47): a//10 in {0} -> b//10 in {0}; a//10 in {1} -> b//10 in {1}; a//10 in {6} -> b//10 in {6,7}; a//10 in {7} -> b//10 in {7,8}; a//10 in {8} -> b//10 in {0,8,9,10}; a//10 in {9} -> b//10 in {0,9,10}; a//10 in {10} -> b//10 in {0,1,10} | (reads) | same |
| L23 up c225 | 10% / 9% | **res%10** (R2 0.98): res mod 10 in {6} | **res%100** (R2 0.49): res mod 100 in {6, 16, 26, 36, 46, 56, 66, 76, 86, 96} [coarser: res mod 10 in {6}, R2 0.80] | (reads) | same |
| L23 up c231 | 15% / 8% | **res%100** (R2 0.69): res mod 100 in {31..37, 51, 53..55, 71, 73..75, 93..95} | unexplained (best R2 0.46) | (reads) | same |
| L23 up c338 | 12% / 5% | **res** (R2 0.89): res in {2..12, 14..16, 18, 20, 22, 26, 36, 46, 56, 66, 76, 86, 96, 106, 116, 126, 136, 146, 156, 166, 176, 186, 196} | **res** (R2 0.60): res in {6, 16, 26, 36, 96} | (reads) | same |
| L23 up c340 | 7% / 2% | **res%100** (R2 0.62): res mod 100 in {31..34, 72} | unexplained (best R2 0.36) | (reads) | same |
| L23 up c387 | 4% / 4% | **units(a,b)** (R2 0.98): a%10 in {0,5} -> b%10 in {0,5} | **units(a,b)** (R2 0.94): a%10 in {0,5} -> b%10 in {0,5} | (reads) | same |
| L23 up c407 | 3% / 36% | **res%100** (R2 0.49): res mod 100 in {22} | **tens(a,b)** (R2 0.61): a//10 in {3} -> b//10 in {2,5,6,7}; a//10 in {4} -> b//10 in {3,6,7}; a//10 in {5} -> b//10 in {3,4,5,6,7,8}; a//10 in {6} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10}; a//10 in {10} -> b//10 in {5,6,7,8,9} | (reads) | same |
| L23 up c465 | 18% / 6% | **res%100** (R2 0.85): res mod 100 in {73..90} | unexplained (best R2 0.41) | (reads) | same |
| L23 up c562 | 15% / 5% | **res** (R2 0.74): res in {38..44, 49, 136..151, 159..161, 179..183, 190, 200} | unexplained (best R2 0.31) | (reads) | same |
| L23 up c638 | 3% / 1% | **res%100** (R2 0.82): res mod 100 in {69..71} | unexplained (best R2 0.20) | (reads) | same |
| L23 up c653 | 38% / 20% | **res%100** (R2 0.77): res mod 100 in {5, 7..39, 69} | **res** (R2 0.72): res in {5, 7..33} | (reads) | same |
| L23 up c878 | 2% / 1% | **res%100** (R2 0.72): res mod 100 in {96} | **res%100** (R2 0.63): res mod 100: no class above 0.5 (max 0.45) | (reads) | same |

</details>

</details>

<details><summary>layer 24: 144 components with main position `=`</summary>

<details><summary>down: 52</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L24 down c1 | 75% / 90% | **res** (R2 0.57): res in {2..16, 18, 21..25, 29..85, 89..90, 93, 100..136, 138, 140, 143..145, 150, 154, 160, 163..166, 170, 173..176, 180, 190, 199..200} | unexplained (best R2 0.17) | - | same |
| L24 down c4 | 11% / 12% | **res%10** (R2 0.86): res mod 10 in {8} | unexplained (best R2 0.50) | res: mod10 +7%, mod5 +8%, mod2 +4% | res: mod10 +5%, mod5 +6% |
| L24 down c5 | 11% / 12% | **res%10** (R2 0.90): res mod 10 in {3} | unexplained (best R2 0.47) | res: mod10 +7%, mod5 +8%, mod2 +4% | res: mod10 +5%, mod5 +6% |
| L24 down c8 | 25% / 28% | **res//10** (R2 0.74): (tens) res in {52..85, 169} | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {6,7,8}; a//10 in {2,3,4} -> b//10 in {7,8}; a//10 in {5} -> b//10 in {8,9}; a//10 in {6} -> b//10 in {0,1,9,10}; a//10 in {7} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1,2,3}; a//10 in {9} -> b//10 in {1,2,3,4}; a//10 in {10} -> b//10 in {2,3,4} | res: mod100 +2% | - |
| L24 down c11 | 17% / 9% | **res%100** (R2 0.86): res mod 100 in {21..28, 61..69} | **res** (R2 0.50): res in {22..27, 62..66} | res: mod25 +3%, mod20 +3% | - |
| L24 down c12 | 10% / 11% | **res%10** (R2 0.96): res mod 10 in {5} | **units(a,b)** (R2 0.64): a%10 in {0} -> b%10 in {5}; a%10 in {1} -> b%10 in {6}; a%10 in {2} -> b%10 in {7}; a%10 in {3} -> b%10 in {8}; a%10 in {4} -> b%10 in {9}; a%10 in {5} -> b%10 in {0,5}; a%10 in {6} -> b%10 in {1}; a%10 in {7} -> b%10 in {2}; a%10 in {8} -> b%10 in {3}; a%10 in {9} -> b%10 in {4} | res: mod10 +3%, mod5 +4% | res: mod10 +4%, mod5 +5% |
| L24 down c13 | 14% / 10% | **res%100** (R2 0.89): res mod 100 in {52..64} | unexplained (best R2 0.40) | res: mod100 +3%, mod50 +2%, mod25 +4% | - |
| L24 down c16 | 18% / 12% | **res%100** (R2 0.76): res mod 100 in {16..17, 21..22, 24..28, 70..72, 74..77, 81..82} [coarser: res mod 50 in {16, 20..22, 24..27}, R2 0.84] | **res** (R2 0.50): res in {15..17, 20..22, 24..27, 71..72} | res: mod25 +5% | res: mod25 +3% |
| L24 down c17 | 12% / 9% | **res%100** (R2 0.86): res mod 100 in {46..56} | unexplained (best R2 0.35) | res: mod25 +3% | - |
| L24 down c22 | 14% / 7% | **res%100** (R2 0.71): res mod 100 in {13..17, 63..68, 70..71, 74} [coarser: res mod 50 in {13..17, 24}, R2 0.83] | **res** (R2 0.55): res in {13..17, 24} | res: mod25 +2% | - |
| L24 down c28 | 16% / 3% | **res%50** (R2 0.70): res mod 50 in {2..5, 13..14, 43..44} | unexplained (best R2 0.25) | res: mod25 +3% | - |
| L24 down c31 | 19% / 15% | **res//10** (R2 0.82): (tens) res in {8..20, 102..119, 200} [coarser: res mod 100 in {0, 7..19}, R2 0.86] | **res** (R2 0.69): res in {4..19} | - | res: mod25 +2% |
| L24 down c34 | 28% / 0 | **res//10** (R2 0.89): (tens) res in {126..200} | off (on 0) | a: mod100 +3%; b: mod100 +3% | - |
| L24 down c36 | 9% / 3% | **res%100** (R2 0.94): res mod 100 in {70..78} | unexplained (best R2 0.26) | res: mod25 +3% | - |
| L24 down c37 | 9% / 2% | **res%100** (R2 0.69): res mod 100 in {47, 77, 85..89} | unexplained (best R2 0.30) | - | same |
| L24 down c41 | 15% / 4% | **res%100** (R2 0.90): res mod 100 in {82..96} | unexplained (best R2 0.43) | res: mod25 +3% | - |
| L24 down c43 | 22% / 21% | **res%100** (R2 0.79): res mod 100 in {15..19, 36..37, 55..59, 75..79, 95..99} [coarser: res mod 20 in {15..19}, R2 0.87] | unexplained (best R2 0.42) | res: mod20 +3% | a: mod20 +2%; b: mod20 +2%; res: mod20 +5% |
| L24 down c46 | 10% / 9% | **res%10** (R2 0.98): res mod 10 in {0} | **res%20** (R2 0.47): res mod 20 in {0, 10} [coarser: res mod 10 in {0}, R2 0.83] | - | same |
| L24 down c50 | 6% / 3% | **res%100** (R2 0.75): res mod 100 in {25..27, 76, 86} | unexplained (best R2 0.42) | res: mod4 +3% | - |
| L24 down c51 | 28% / 0 | **tens(a,b)** (R2 0.80): a//10 in {2} -> b//10 in {9,10}; a//10 in {3} -> b//10 in {8,9,10}; a//10 in {4,5} -> b//10 in {6,7,8,9,10}; a//10 in {6} -> b//10 in {5,6,7,8,9,10}; a//10 in {7} -> b//10 in {4,5,6,7}; a//10 in {8} -> b//10 in {3,4,5,6}; a//10 in {9,10} -> b//10 in {2,3,4,5,10} | off (on 0) | - | same |
| L24 down c52 | 9% / 3% | **res%100** (R2 0.75): res mod 100 in {14, 24, 34..35, 44, 54, 74, 84, 94} [coarser: res mod 50 in {14, 24, 34, 44}, R2 0.81] | unexplained (best R2 0.45) | res: mod4 +5% | - |
| L24 down c65 | 3% / 8% | unexplained (best R2 0.23) | unexplained (best R2 0.25) | - | same |
| L24 down c70 | 9% / 18% | **res** (R2 0.86): res in {35..49, 51..55} | **tens(a,b)** (R2 0.56): a//10 in {0} -> b//10 in {4,5}; a//10 in {3} -> b//10 in {0,4,5}; a//10 in {4} -> b//10 in {0,1,4,5,10}; a//10 in {5} -> b//10 in {0,1,2,6}; a//10 in {6} -> b//10 in {1,2}; a//10 in {7} -> b//10 in {3}; a//10 in {10} -> b//10 in {5,6} | - | same |
| L24 down c76 | 5% / 0 | **res** (R2 0.63): res in {116..117, 119..120, 194..200} | off (on 0) | - | same |
| L24 down c79 | 0 / 32% | off (on 0) | **a** (R2 0.51): a in {2..24, 97} | - | a: mod100 +2% |
| L24 down c82 | 8% / 5% | **res%100** (R2 0.81): res mod 100 in {11, 21, 31..32, 41..42, 81..82} | unexplained (best R2 0.50) | - | same |
| L24 down c83 | 10% / 4% | **res%10** (R2 0.96): res mod 10 in {9} | **res** (R2 0.69): res in {9, 19, 29, 39, 69, 79, 89, 99} | - | same |
| L24 down c87 | 8% / 3% | **res%100** (R2 0.83): res mod 100 in {80..86} | unexplained (best R2 0.49) | - | same |
| L24 down c91 | 8% / 4% | **res%100** (R2 0.73): res mod 100 in {16, 36..37, 56, 95..96} | **res** (R2 0.54): res in {6, 16, 36, 95..97} | res: mod4 +3% | - |
| L24 down c99 | 11% / 6% | **res%100** (R2 0.83): res mod 100 in {26..36} | **res** (R2 0.63): res in {26..34} | - | same |
| L24 down c101 | 10% / 11% | **res%100** (R2 0.91): res mod 100 in {15..24} | **res** (R2 0.63): res in {14..24} | - | same |
| L24 down c108 | 10% / 5% | **res%10** (R2 0.96): res mod 10 in {6} | **res** (R2 0.59): res in {6, 16, 26, 36, 86, 96} | - | same |
| L24 down c112 | 16% / 0 | **res//10** (R2 0.84): (tens) res in {120..142} | off (on 0) | - | same |
| L24 down c152 | 4% / 0 | **res%100** (R2 0.72): res mod 100 in {75..78} | off (on 0) | - | same |
| L24 down c158 | 1% / 14% | **res** (R2 0.66): res in {6..15} | unexplained (best R2 0.44) | - | same |
| L24 down c191 | 8% / 3% | **res%100** (R2 0.91): res mod 100 in {76..83} | unexplained (best R2 0.33) | - | same |
| L24 down c207 | 6% / 7% | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {9}; a//10 in {8} -> b//10 in {10}; a//10 in {9} -> b//10 in {0,8,9,10}; a//10 in {10} -> b//10 in {8,9,10} | **a** (R2 0.61): a in {93..99} | - | same |
| L24 down c209 | 0 / 4% | off (on 0) | unexplained (best R2 0.27) | - | same |
| L24 down c210 | 7% / 1% | **res%20** (R2 0.74): res mod 20 in {12} | unexplained (best R2 0.28) | - | same |
| L24 down c212 | 4% / 2% | **res%100** (R2 0.61): res mod 100 in {20, 40, 80} | **res** (R2 0.55): res in {20} | - | same |
| L24 down c240 | 0 / 2% | off (on 0) | unexplained (best R2 0.46) | - | same |
| L24 down c263 | 5% / 2% | **res%100** (R2 0.70): res mod 100: no class above 0.5 (max 0.44) | unexplained (best R2 0.35) | - | same |
| L24 down c294 | 4% / 0 | **res** (R2 0.83): res in {142..149} | off (on 0) | - | same |
| L24 down c313 | 0 / 27% | off (on 0) | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {0,1}; a//10 in {1} -> b//10 in {1}; a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {4,5,7}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {5,6,7,8,9,10} | - | same |
| L24 down c348 | 0 / 2% | off (on 0) | **res%100** (R2 0.85): res mod 100: no class above 0.5 (max 0.46) | - | same |
| L24 down c350 | 9% / 13% | **res//10** (R2 0.71): (tens) res in {2..43, 200} | unexplained (best R2 0.49) | - | same |
| L24 down c377 | 10% / 0 | **res%100** (R2 0.84): res mod 100 in {0..1, 93..99} | off (on 0) | - | same |
| L24 down c482 | 0 / 1% | off (on 0) | **res%100** (R2 0.78): res mod 100: no class above 0.5 (max 0.41) | - | same |
| L24 down c501 | 0 / 3% | off (on 0) | **res%100** (R2 0.81): res mod 100: no class above 0.5 (max 0.46) | - | same |
| L24 down c553 | 5% / 0 | **res** (R2 0.76): res in {160..174, 184} | off (on 0) | - | same |
| L24 down c593 | 4% / 0 | **res** (R2 0.67): res in {70, 90, 110, 130, 140, 150, 170} | off (on 0) | - | same |
| L24 down c678 | 0 / 6% | off (on 0) | **res%100** (R2 0.51): res mod 100 in {3} | - | same |

</details>

<details><summary>gate: 56</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L24 gate c4 | 12% / 14% | **res%10** (R2 0.81): res mod 10 in {8} | unexplained (best R2 0.48) | (reads) | same |
| L24 gate c5 | 11% / 14% | **res%10** (R2 0.90): res mod 10 in {3} | unexplained (best R2 0.46) | (reads) | same |
| L24 gate c8 | 16% / 4% | **res%100** (R2 0.76): res mod 100 in {56..70} | unexplained (best R2 0.37) | (reads) | same |
| L24 gate c11 | 17% / 8% | **res%100** (R2 0.88): res mod 100 in {21..27, 61..68} | **res** (R2 0.54): res in {22..27, 62..66} | (reads) | same |
| L24 gate c12 | 21% / 17% | **res%10** (R2 0.95): res mod 10 in {5, 8} | **res%100** (R2 0.52): res mod 100 in {5, 8, 15, 18, 25, 35, 45, 55, 65, 75, 78, 85, 88, 95, 98} [coarser: res mod 50 in {5, 15, 25, 35, 38, 45, 48}, R2 0.81] | (reads) | same |
| L24 gate c13 | 17% / 7% | **res%100** (R2 0.94): res mod 100 in {49..64} | unexplained (best R2 0.37) | (reads) | same |
| L24 gate c16 | 23% / 12% | **res%100** (R2 0.77): res mod 100 in {16, 20..22, 24..27, 66..67, 70..77, 81..82} [coarser: res mod 50 in {16, 20..27}, R2 0.84] | unexplained (best R2 0.44) | (reads) | same |
| L24 gate c17 | 18% / 10% | **res%100** (R2 0.86): res mod 100 in {46..62} | unexplained (best R2 0.42) | (reads) | same |
| L24 gate c22 | 19% / 8% | **res%100** (R2 0.80): res mod 100 in {9..17, 64..67, 69..71} | **res** (R2 0.75): res in {9..17, 19} | (reads) | same |
| L24 gate c28 | 11% / 6% | **res%10** (R2 0.91): res mod 10 in {3} | **res** (R2 0.62): res in {3, 13, 23, 33, 43, 53, 73, 83, 93} | (reads) | same |
| L24 gate c31 | 29% / 23% | **res%100** (R2 0.88): res mod 100 in {0..1, 4..24, 96..99} | **res%100** (R2 0.61): res mod 100 in {1..4, 6..22, 99} | (reads) | same |
| L24 gate c34 | 41% / 9% | **res//10** (R2 0.86): (tens) res in {2..15, 17, 114..200} | **res%100** (R2 0.52): res mod 100 in {0, 10..11} | (reads) | same |
| L24 gate c36 | 17% / 9% | **res%100** (R2 0.89): res mod 100 in {68..84} | **tens(a,b)** (R2 0.59): a//10 in {0} -> b//10 in {7}; a//10 in {7,8} -> b//10 in {0,1,2}; a//10 in {9} -> b//10 in {1,2}; a//10 in {10} -> b//10 in {2} | (reads) | same |
| L24 gate c37 | 8% / 1% | **res%100** (R2 0.87): res mod 100 in {84..91} | unexplained (best R2 0.36) | (reads) | same |
| L24 gate c41 | 23% / 5% | **res%100** (R2 0.91): res mod 100 in {0..1, 81..99} | unexplained (best R2 0.48) | (reads) | same |
| L24 gate c46 | 13% / 12% | **res%50** (R2 0.79): res mod 50 in {3, 13, 23, 33, 43} | **res** (R2 0.58): res in {-98, 0, 3, 8, 10, 13, 18, 20, 23, 28, 30, 33, 38, 43, 83, 88, 93, 98..99} | (reads) | same |
| L24 gate c50 | 12% / 8% | **res%100** (R2 0.83): res mod 100 in {21..32} | **res** (R2 0.60): res in {21..29} | (reads) | same |
| L24 gate c51 | 18% / 0 | **res//10** (R2 0.88): (tens) res in {140..190, 192, 194..198, 200} | off (on 0) | (reads) | same |
| L24 gate c52 | 6% / 1% | **res%100** (R2 0.72): res mod 100 in {32, 34, 54, 94} | unexplained (best R2 0.33) | (reads) | same |
| L24 gate c65 | 2% / 6% | unexplained (best R2 0.29) | unexplained (best R2 0.27) | (reads) | same |
| L24 gate c70 | 0 / 0 | off (on 0) | same | (reads) | same |
| L24 gate c71 | 83% / 93% | **res%100** (R2 0.62): res mod 100 in {0, 2..35, 38, 40, 43..85, 90, 93, 95, 99} | unexplained (best R2 0.22) | (reads) | same |
| L24 gate c82 | 20% / 11% | **res%10** (R2 0.89): res mod 10 in {1, 8} | **res** (R2 0.59): res in {1..2, 8, 11, 18, 21, 28, 31, 38, 41, 48, 78, 88, 91, 98} | (reads) | same |
| L24 gate c83 | 10% / 9% | **res%10** (R2 0.96): res mod 10 in {5} | **res%10** (R2 0.57): res mod 10 in {5} | (reads) | same |
| L24 gate c86 | 0 / 1% | off (on 0) | **res%100** (R2 0.59): res mod 100: no class above 0.5 (max 0.39) | (reads) | same |
| L24 gate c87 | 8% / 0 | **res%100** (R2 0.85): res mod 100 in {70..76} | off (on 0) | (reads) | same |
| L24 gate c91 | 17% / 10% | **res%20** (R2 0.87): res mod 20 in {3, 13, 16} [coarser: res mod 10 in {3, 6}, R2 0.84] | **res** (R2 0.59): res in {3, 6, 13, 16, 23, 33, 36, 43, 53, 63, 73, 83, 93, 96} | (reads) | same |
| L24 gate c99 | 7% / 2% | **res%100** (R2 0.81): res mod 100 in {26..27, 29..32, 34} | unexplained (best R2 0.44) | (reads) | same |
| L24 gate c101 | 29% / 18% | **res%100** (R2 0.80): res mod 100 in {26..42, 44..52} | **tens(a,b)** (R2 0.56): a//10 in {3} -> b//10 in {0,3,4,10}; a//10 in {4} -> b//10 in {0,1,4,5,10}; a//10 in {5} -> b//10 in {0,1,2}; a//10 in {6} -> b//10 in {2,3}; a//10 in {7} -> b//10 in {3,4}; a//10 in {8} -> b//10 in {4,5}; a//10 in {10} -> b//10 in {5,6,7} | (reads) | same |
| L24 gate c108 | 8% / 4% | **res%100** (R2 0.72): res mod 100 in {19, 29, 39, 59, 69, 79} | **res** (R2 0.53): res in {9, 19..20, 29, 79, 99} | (reads) | same |
| L24 gate c112 | 9% / 2% | **res%10** (R2 0.92): res mod 10 in {8} | unexplained (best R2 0.43) | (reads) | same |
| L24 gate c152 | 3% / 0 | **res%100** (R2 0.75): res mod 100 in {73..76} | off (on 0) | (reads) | same |
| L24 gate c171 | 1% / 16% | **res** (R2 0.69): res in {2..7, 9, 104, 200} | **res//10** (R2 0.55): (tens) res in {-97, -3, 0..8, 99} | (reads) | same |
| L24 gate c191 | 3% / 0 | **res** (R2 0.72): res in {76..77, 173..177} [coarser: res mod 100 in {74..77}, R2 0.88] | off (on 0) | (reads) | same |
| L24 gate c210 | 9% / 2% | **res%100** (R2 0.80): res mod 100 in {32, 42, 52, 71..74, 92} | unexplained (best R2 0.44) | (reads) | same |
| L24 gate c263 | 7% / 1% | **res%100** (R2 0.92): res mod 100 in {23, 33, 43, 53, 63, 73, 83} | unexplained (best R2 0.50) | (reads) | same |
| L24 gate c285 | 10% / 5% | **res%10** (R2 0.97): res mod 10 in {8} | **res** (R2 0.62): res in {8, 18, 28, 38, 48, 88, 98} | (reads) | same |
| L24 gate c357 | 3% / 0 | **res%100** (R2 0.56): res mod 100 in {55, 75, 95} | off (on 0) | (reads) | same |
| L24 gate c361 | 6% / 1% | **res%100** (R2 0.84): res mod 100 in {55..61} | unexplained (best R2 0.18) | (reads) | same |
| L24 gate c367 | 1% / 3% | unexplained (best R2 0.45) | **res%100** (R2 0.57): res mod 100 in {0, 5, 99} | (reads) | same |
| L24 gate c419 | 5% / 0 | **res%100** (R2 0.93): res mod 100 in {53, 63, 73, 83, 93} | off (on 0) | (reads) | same |
| L24 gate c444 | 2% / 0 | **res%100** (R2 0.73): res mod 100 in {36, 96} | off (on 0) | (reads) | same |
| L24 gate c482 | 9% / 3% | **res%10** (R2 0.88): res mod 10 in {4} | **res** (R2 0.56): res in {4, 24, 34, 84} | (reads) | same |
| L24 gate c501 | 3% / 3% | **res%100** (R2 0.75): res mod 100: no class above 0.5 (max 0.44) | **res%100** (R2 0.72): res mod 100: no class above 0.5 (max 0.45) | (reads) | same |
| L24 gate c553 | 1% / 0 | unexplained (best R2 0.48) | off (on 0) | (reads) | same |
| L24 gate c555 | 10% / 3% | **res%10** (R2 0.96): res mod 10 in {3} | **res** (R2 0.67): res in {3, 13, 23, 93} | (reads) | same |
| L24 gate c560 | 0 / 1% | off (on 0) | unexplained (best R2 0.48) | (reads) | same |
| L24 gate c586 | 100% / 100% | always | same | (reads) | same |
| L24 gate c627 | 19% / 6% | **res%100** (R2 0.79): res mod 100 in {9..10, 26..27, 29..30, 46..51, 66..70} | unexplained (best R2 0.44) | (reads) | same |
| L24 gate c658 | 0 / 0 | **res** (R2 0.59): res in {174} | off (on 0) | (reads) | same |
| L24 gate c694 | 1% / 1% | **res%100** (R2 0.68): res mod 100: no class above 0.5 (max 0.37) | **res%100** (R2 0.66): res mod 100: no class above 0.5 (max 0.37) | (reads) | same |
| L24 gate c706 | 8% / 3% | **res%10** (R2 0.81): res mod 10 in {5} | **res** (R2 0.64): res in {5, 15, 25, 35} | (reads) | same |
| L24 gate c736 | 10% / 3% | **res%10** (R2 0.96): res mod 10 in {5} | **res** (R2 0.64): res in {5, 15, 25, 85, 95} | (reads) | same |
| L24 gate c744 | 8% / 2% | **res%100** (R2 0.90): res mod 100 in {18, 48, 58, 68, 78, 88, 98} | **res** (R2 0.64): res in {8, 18, 88, 98} | (reads) | same |
| L24 gate c755 | 10% / 6% | **res%10** (R2 0.98): res mod 10 in {5} | **res%100** (R2 0.61): res mod 100 in {5, 15, 25, 85, 95} | (reads) | same |
| L24 gate c893 | 9% / 2% | **res%10** (R2 0.86): res mod 10 in {3} | **res** (R2 0.57): res in {13, 23} | (reads) | same |

</details>

<details><summary>o: 3</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L24 o c23 (H13) | 68% / 83% | **tens(a,b)** (R2 0.56): a//10 in {0,1} -> b//10 in {6,7,8,9}; a//10 in {2,3} -> b//10 in {5,6,7,8,9}; a//10 in {4} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {1,2,3,4,5,7,8,9,10}; a//10 in {7,8,9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | unexplained (best R2 0.40) | - | same |
| L24 o c501 (H22) | 0 / 2% | off (on 0) | **a** (R2 0.63): a in {27..29} | - | same |
| L24 o c503 (H17) | 1% / 1% | **a** (R2 0.89): a in {90} | **a** (R2 0.82): a in {90} | - | same |

</details>

<details><summary>up: 33</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L24 up c5 | 10% / 5% | **res%10** (R2 0.98): res mod 10 in {3} | **res** (R2 0.68): res in {3, 13, 23, 33, 43, 73, 83, 93} | (reads) | same |
| L24 up c8 | 26% / 29% | **res//10** (R2 0.72): (tens) res in {52..85, 169} | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {6,7,8}; a//10 in {2,3,4} -> b//10 in {7,8}; a//10 in {5} -> b//10 in {8,9}; a//10 in {6} -> b//10 in {0,1,2,9,10}; a//10 in {7} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1,2,3}; a//10 in {9} -> b//10 in {1,2,3,4}; a//10 in {10} -> b//10 in {2,3,4} | (reads) | same |
| L24 up c11 | 25% / 4% | **res%100** (R2 0.85): res mod 100 in {60..76, 78..84} | unexplained (best R2 0.35) | (reads) | same |
| L24 up c12 | 17% / 12% | **res%10** (R2 0.82): res mod 10 in {5, 8} | **res%100** (R2 0.61): res mod 100 in {5, 8, 15, 25, 35, 45, 55, 65, 75, 85, 95, 98} | (reads) | same |
| L24 up c13 | 19% / 7% | **res%100** (R2 0.85): res mod 100 in {52..68} | unexplained (best R2 0.38) | (reads) | same |
| L24 up c15 | 11% / 4% | **res%100** (R2 0.86): res mod 100 in {81..90} | unexplained (best R2 0.42) | (reads) | same |
| L24 up c16 | 25% / 16% | **res%100** (R2 0.82): res mod 100 in {10..13, 20..23, 31..32, 41..42, 50..52, 70..73, 81..82, 90..92} | **res** (R2 0.56): res in {-19, 1..2, 10..13, 20..23, 31..32, 71..72, 82} | (reads) | same |
| L24 up c17 | 21% / 13% | **res//10** (R2 0.74): (tens) res in {38..58, 140..156} [coarser: res mod 100 in {39..56}, R2 0.97] | unexplained (best R2 0.46) | (reads) | same |
| L24 up c22 | 16% / 9% | **res%100** (R2 0.79): res mod 100 in {13..15, 24..25, 34, 63..65, 73..75, 83..85, 94} [coarser: res mod 50 in {13..15, 23..25, 33..35, 44}, R2 0.82] | **res** (R2 0.52): res in {4, 13..15, 23..25, 34, 74, 84, 94} | (reads) | same |
| L24 up c28 | 13% / 2% | **res%100** (R2 0.84): res mod 100 in {49..60} | unexplained (best R2 0.35) | (reads) | same |
| L24 up c36 | 8% / 1% | **res%100** (R2 0.89): res mod 100 in {70..77} | unexplained (best R2 0.25) | (reads) | same |
| L24 up c37 | 9% / 2% | **res%100** (R2 0.81): res mod 100 in {26..27, 47, 67, 85..88} | unexplained (best R2 0.40) | (reads) | same |
| L24 up c39 | 15% / 6% | **res%100** (R2 0.82): res mod 100 in {0, 50, 88..99} | unexplained (best R2 0.38) | (reads) | same |
| L24 up c43 | 21% / 12% | **res%100** (R2 0.75): res mod 100 in {8..11, 29..30, 47..52, 69..70, 86..92} | unexplained (best R2 0.48) | (reads) | same |
| L24 up c46 | 10% / 9% | **res%10** (R2 0.98): res mod 10 in {0} | **res%100** (R2 0.54): res mod 100 in {0, 10, 20, 30, 50, 70, 80, 90} [coarser: res mod 10 in {0}, R2 0.82] | (reads) | same |
| L24 up c50 | 13% / 4% | **res%50** (R2 0.84): res mod 50 in {6, 16..17, 26, 36, 46} [coarser: res mod 10 in {6}, R2 0.82] | **res** (R2 0.52): res in {16..17, 26, 96} | (reads) | same |
| L24 up c52 | 9% / 4% | **res%100** (R2 0.79): res mod 100 in {14, 24..25, 34, 64..65, 74, 84, 94} [coarser: res mod 50 in {14, 24, 34}, R2 0.88] | **res** (R2 0.54): res in {4, 14, 24..25, 34, 84} | (reads) | same |
| L24 up c65 | 98% / 67% | always | unexplained (best R2 0.24) | (reads) | same |
| L24 up c76 | 5% / 82% | unexplained (best R2 0.44) | unexplained (best R2 0.26) | (reads) | same |
| L24 up c82 | 14% / 9% | **res%100** (R2 0.83): res mod 100 in {21, 31..42} | unexplained (best R2 0.47) | (reads) | same |
| L24 up c91 | 5% / 21% | **res%100** (R2 0.75): res mod 100 in {2..9} | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {0}; a//10 in {1} -> b//10 in {0,1}; a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {4,5}; a//10 in {6} -> b//10 in {5,6,7,8}; a//10 in {7} -> b//10 in {6,7,8,9}; a//10 in {8} -> b//10 in {7,8,9}; a//10 in {9} -> b//10 in {8,9,10}; a//10 in {10} -> b//10 in {5,8,9,10} | (reads) | same |
| L24 up c99 | 13% / 6% | **res%100** (R2 0.80): res mod 100 in {25..36} | **res** (R2 0.59): res in {26..33} | (reads) | same |
| L24 up c101 | 22% / 17% | **res** (R2 0.80): res in {2..7, 12..45, 49, 117..129} | **tens(a,b)** (R2 0.56): a//10 in {0} -> b//10 in {2}; a//10 in {2} -> b//10 in {0,2,10}; a//10 in {3} -> b//10 in {0,1}; a//10 in {4} -> b//10 in {1,2}; a//10 in {5} -> b//10 in {2,3}; a//10 in {6} -> b//10 in {3,4}; a//10 in {7} -> b//10 in {4,5}; a//10 in {8} -> b//10 in {5,6}; a//10 in {9} -> b//10 in {7}; a//10 in {10} -> b//10 in {7,8} | (reads) | same |
| L24 up c108 | 15% / 9% | **res%50** (R2 0.80): res mod 50 in {6, 16, 25..27, 36, 46} | unexplained (best R2 0.45) | (reads) | same |
| L24 up c158 | 5% / 1% | **res%100** (R2 0.76): res mod 100 in {49..54} | unexplained (best R2 0.28) | (reads) | same |
| L24 up c191 | 11% / 6% | **res%100** (R2 0.88): res mod 100 in {75..85} | unexplained (best R2 0.38) | (reads) | same |
| L24 up c210 | 18% / 10% | **res%100** (R2 0.79): res mod 100 in {16..19, 56..59, 75..79, 95..98} | unexplained (best R2 0.47) | (reads) | same |
| L24 up c212 | 15% / 3% | **res** (R2 0.72): res in {19..20, 40, 70, 79..80, 99..100, 110, 119..122, 129..130, 139..140, 149..150, 160, 169..170, 179..180} [coarser: res mod 100 in {19..22, 29..30, 40, 60, 70, 79..80, 99}, R2 0.80] | unexplained (best R2 0.50) | (reads) | same |
| L24 up c234 | 5% / 1% | **res** (R2 0.75): res in {66..67, 116..117, 166..171} | unexplained (best R2 0.45) | (reads) | same |
| L24 up c444 | 6% / 13% | **res%100** (R2 0.68): res mod 100 in {2..11} | **res%100** (R2 0.75): res mod 100 in {10..12} | (reads) | same |
| L24 up c482 | 1% / 0 | **res%100** (R2 0.84): res mod 100 in {34} | off (on 0) | (reads) | same |
| L24 up c553 | 4% / 0 | **res** (R2 0.72): res in {160..172} | off (on 0) | (reads) | same |
| L24 up c593 | 5% / 1% | **res%100** (R2 0.80): res mod 100 in {62, 82..84} | unexplained (best R2 0.33) | (reads) | same |

</details>

</details>

