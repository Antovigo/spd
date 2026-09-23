# Appendix C4 — components whose main position is `=`, L25-27

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

<details><summary>layer 25: 143 components with main position `=`</summary>

<details><summary>down: 50</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L25 down c6 | 100% / 100% | always | same | - | same |
| L25 down c10 | 13% / 9% | **res%100** (R2 0.81): res mod 100 in {86..96, 98} | unexplained (best R2 0.47) | res: mod100 +3%, mod50 +2%, mod25 +6% | - |
| L25 down c11 | 11% / 11% | **res%10** (R2 0.92): res mod 10 in {7} | **units(a,b)** (R2 0.51): a%10 in {0} -> b%10 in {3}; a%10 in {1} -> b%10 in {4}; a%10 in {2} -> b%10 in {5}; a%10 in {3} -> b%10 in {6}; a%10 in {5} -> b%10 in {8}; a%10 in {6} -> b%10 in {9}; a%10 in {7} -> b%10 in {0}; a%10 in {8} -> b%10 in {1}; a%10 in {9} -> b%10 in {2} | res: mod10 +5%, mod5 +5%, mod2 +4% | res: mod10 +3%, mod5 +4% |
| L25 down c12 | 11% / 28% | **res%10** (R2 0.86): res mod 10 in {1} | unexplained (best R2 0.47) | res: mod10 +4%, mod5 +4%, mod2 +3% | res: mod10 +4%, mod5 +5%, mod2 +5% |
| L25 down c15 | 9% / 8% | **res%100** (R2 0.87): res mod 100 in {0..7} | **res%100** (R2 0.68): res mod 100 in {1..7} | res: mod100 +2%, mod25 +4% | res: mod25 +5% |
| L25 down c24 | 11% / 9% | **res%10** (R2 0.86): res mod 10 in {9} | **units(a,b)** (R2 0.53): a%10 in {0} -> b%10 in {1}; a%10 in {1} -> b%10 in {2}; a%10 in {2} -> b%10 in {3}; a%10 in {5} -> b%10 in {6}; a%10 in {6} -> b%10 in {7}; a%10 in {7} -> b%10 in {8}; a%10 in {8} -> b%10 in {9}; a%10 in {9} -> b%10 in {0} | res: mod10 +3%, mod5 +3%, mod2 +2% | res: mod5 +3% |
| L25 down c30 | 19% / 5% | **res%100** (R2 0.78): res mod 100 in {52..57, 59..67, 69..73} | unexplained (best R2 0.30) | - | same |
| L25 down c31 | 16% / 7% | **res%100** (R2 0.86): res mod 100 in {1, 13..14} | **res%100** (R2 0.62): res mod 100 in {6..7} | res: mod50 +2% | - |
| L25 down c32 | 0 / 11% | off (on 0) | unexplained (best R2 0.47) | - | same |
| L25 down c33 | 42% / 0 | **res//10** (R2 0.94): (tens) res in {110..200} | off (on 0) | b: mod100 +2% | - |
| L25 down c40 | 9% / 3% | **res%100** (R2 0.80): res mod 100 in {0..1, 95..99} | **res%100** (R2 0.55): res mod 100 in {0..1, 99} | - | same |
| L25 down c42 | 9% / 8% | **res%100** (R2 0.86): res mod 100 in {7..8, 10..14} | **res%100** (R2 0.73): res mod 100 in {10..12} | - | res: mod25 +4% |
| L25 down c43 | 7% / 17% | **res//10** (R2 0.68): (tens) res in {4..33, 35} | **a** (R2 0.53): a in {3, 5..16} | - | same |
| L25 down c46 | 5% / 4% | **res%100** (R2 0.70): res mod 100 in {17..20, 79} | **res** (R2 0.57): res in {17..20} | - | same |
| L25 down c48 | 16% / 12% | **res%100** (R2 0.86): res mod 100 in {0..1, 8..11, 13..14, 88..91, 93..94, 98..99} | **res%100** (R2 0.49): res mod 100 in {1, 8..11, 13..14, 89..91, 98..99} | - | same |
| L25 down c50 | 11% / 4% | **res%100** (R2 0.79): res mod 100 in {13, 22..25, 43, 53, 63, 73, 83} | **res** (R2 0.60): res in {3, 13, 23..25} | res: mod4 +6% | - |
| L25 down c51 | 11% / 0 | **res** (R2 0.76): res in {62, 122, 152..178, 182..186} | off (on 0) | - | same |
| L25 down c53 | 10% / 5% | **res%10** (R2 0.96): res mod 10 in {5} | **res%100** (R2 0.52): res mod 100 in {15, 25, 85, 95} | - | same |
| L25 down c55 | 20% / 4% | **res** (R2 0.74): res in {42..50, 52..68, 85..89} | unexplained (best R2 0.25) | - | same |
| L25 down c63 | 10% / 2% | **res%100** (R2 0.72): res mod 100 in {30, 40, 50..51, 70, 80, 90..91} | unexplained (best R2 0.31) | - | same |
| L25 down c66 | 13% / 5% | **res%100** (R2 0.89): res mod 100 in {41..47, 81..86} | unexplained (best R2 0.45) | - | same |
| L25 down c67 | 0 / 0 | off (on 0) | same | - | same |
| L25 down c73 | 27% / 4% | **res** (R2 0.67): res in {38, 41..42, 44..51, 56..58, 76..99} | unexplained (best R2 0.48) | - | same |
| L25 down c74 | 22% / 7% | **res** (R2 0.68): res in {20, 22..25, 30, 32..34, 60..85, 165, 173..174} | unexplained (best R2 0.32) | - | same |
| L25 down c76 | 15% / 3% | **res%100** (R2 0.74): res mod 100 in {17..18, 36..39, 57..59, 76..79, 97..98} [coarser: res mod 20 in {17..19}, R2 0.84] | unexplained (best R2 0.38) | res: mod4 +3% | - |
| L25 down c79 | 20% / 0 | **tens(a,b)** (R2 0.52): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {9}; a//10 in {2} -> b//10 in {8,9}; a//10 in {3} -> b//10 in {7,8,9}; a//10 in {4} -> b//10 in {6,8}; a//10 in {6} -> b//10 in {4,5}; a//10 in {7} -> b//10 in {3,4,5}; a//10 in {8} -> b//10 in {2,3,4}; a//10 in {9} -> b//10 in {1,2,3}; a//10 in {10} -> b//10 in {0,1,2,3} | off (on 0) | - | same |
| L25 down c92 | 17% / 0 | **res//10** (R2 0.81): (tens) res in {111..129, 135} | off (on 0) | - | same |
| L25 down c98 | 3% / 1% | **res%100** (R2 0.73): res mod 100 in {24, 64, 84} | **res** (R2 0.53): res in {24} | - | same |
| L25 down c106 | 0 / 45% | off (on 0) | **cmp(a,b)** (R2 0.70): cmp(a,b)=-1: 0.87, cmp(a,b)=0: 0.19, cmp(a,b)=1: 0.04 | - | same |
| L25 down c108 | 14% / 6% | **res%20** (R2 0.85): res mod 20 in {2, 12..13} [coarser: res mod 10 in {2}, R2 0.82] | **res** (R2 0.57): res in {12..13, 22, 32..33, 72, 92} | - | same |
| L25 down c110 | 3% / 3% | **res%100** (R2 0.75): res mod 100 in {16..18} | **res** (R2 0.70): res in {15..18} | - | same |
| L25 down c114 | 7% / 2% | **res%100** (R2 0.84): res mod 100 in {86, 94..98} | unexplained (best R2 0.22) | - | same |
| L25 down c130 | 1% / 0 | **res%100** (R2 0.59): res mod 100 in {56} | off (on 0) | - | same |
| L25 down c139 | 12% / 1% | **res%100** (R2 0.82): res mod 100 in {0..1, 93..99} | unexplained (best R2 0.35) | - | same |
| L25 down c145 | 2% / 2% | unexplained (best R2 0.36) | **units(a,b)** (R2 0.58): a%10 in {0} -> b%10 in {0} | - | same |
| L25 down c151 | 8% / 3% | **res%100** (R2 0.74): res mod 100 in {25..30, 68} | **res** (R2 0.54): res in {25..30} | - | same |
| L25 down c200 | 17% / 0 | **res//10** (R2 0.81): (tens) res in {137..170, 185..188, 195, 198} | off (on 0) | - | same |
| L25 down c207 | 6% / 2% | **res%100** (R2 0.78): res mod 100 in {12, 14, 52, 72, 92} [coarser: res mod 20 in {12}, R2 0.80] | **res** (R2 0.62): res in {12, 14} | res: mod4 +2% | - |
| L25 down c216 | 6% / 2% | **res%100** (R2 0.80): res mod 100 in {20, 30, 40, 60, 70, 80} | **res** (R2 0.54): res in {20} | - | same |
| L25 down c220 | 5% / 30% | **res//10** (R2 0.68): (tens) res in {2, 4..28, 30} | **tens(a,b)** (R2 0.64): a//10 in {0,1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {0,1,2,3}; a//10 in {3} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {7,8,9} | - | res: mod100 +2% |
| L25 down c248 | 6% / 1% | **res%100** (R2 0.86): res mod 100 in {75..79} | unexplained (best R2 0.27) | - | same |
| L25 down c256 | 7% / 0 | **res//10** (R2 0.83): (tens) res in {165..200} | off (on 0) | - | same |
| L25 down c297 | 8% / 3% | **res%100** (R2 0.84): res mod 100 in {82..89} | unexplained (best R2 0.41) | - | same |
| L25 down c331 | 3% / 1% | **res%100** (R2 0.72): res mod 100 in {98} | unexplained (best R2 0.36) | - | same |
| L25 down c335 | 9% / 2% | **res%100** (R2 0.75): res mod 100 in {13, 51..56, 93} | unexplained (best R2 0.27) | - | same |
| L25 down c464 | 0 / 0 | off (on 0) | same | - | same |
| L25 down c586 | 3% / 1% | **res%100** (R2 0.77): res mod 100 in {33, 53, 73} | unexplained (best R2 0.26) | - | same |
| L25 down c715 | 0 / 1% | off (on 0) | **units(a,b)** (R2 0.60): a%10 in {0} -> b%10 in {0} | - | same |
| L25 down c748 | 0 / 1% | off (on 0) | **res%100** (R2 0.63): res mod 100: no class above 0.5 (max 0.45) | - | same |
| L25 down c1005 | 3% / 12% | unexplained (best R2 0.33) | unexplained (best R2 0.48) | - | same |

</details>

<details><summary>gate: 52</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L25 gate c10 | 18% / 13% | **res%100** (R2 0.89): res mod 100 in {1, 85..99} | **tens(a,b)** (R2 0.53): a//10 in {0} -> b//10 in {9,10}; a//10 in {6} -> b//10 in {7}; a//10 in {7} -> b//10 in {8}; a//10 in {8} -> b//10 in {8,9}; a//10 in {9} -> b//10 in {0,1,3,4,5,9,10}; a//10 in {10} -> b//10 in {0,1,10} | (reads) | same |
| L25 gate c11 | 11% / 11% | **res%10** (R2 0.89): res mod 10 in {7} | **units(a,b)** (R2 0.51): a%10 in {0} -> b%10 in {3}; a%10 in {1} -> b%10 in {4}; a%10 in {2} -> b%10 in {5}; a%10 in {3} -> b%10 in {6}; a%10 in {5} -> b%10 in {8}; a%10 in {6} -> b%10 in {9}; a%10 in {7} -> b%10 in {0}; a%10 in {8} -> b%10 in {1}; a%10 in {9} -> b%10 in {2} | (reads) | same |
| L25 gate c12 | 12% / 42% | **res%10** (R2 0.84): res mod 10 in {1} | unexplained (best R2 0.42) | (reads) | same |
| L25 gate c15 | 18% / 17% | **res%100** (R2 0.91): res mod 100 in {0..14, 98..99} | **res%100** (R2 0.72): res mod 100 in {0..2, 4, 6..14, 99} | (reads) | same |
| L25 gate c24 | 12% / 9% | **res%10** (R2 0.81): res mod 10 in {9} | **units(a,b)** (R2 0.53): a%10 in {0} -> b%10 in {1}; a%10 in {1} -> b%10 in {2}; a%10 in {2} -> b%10 in {3}; a%10 in {5} -> b%10 in {6}; a%10 in {6} -> b%10 in {7}; a%10 in {7} -> b%10 in {8}; a%10 in {8} -> b%10 in {9}; a%10 in {9} -> b%10 in {0} | (reads) | same |
| L25 gate c30 | 24% / 7% | **res%100** (R2 0.86): res mod 100 in {51..74} | unexplained (best R2 0.40) | (reads) | same |
| L25 gate c31 | 16% / 12% | **res%100** (R2 0.93): res mod 100 in {0..15} | **res%100** (R2 0.77): res mod 100 in {8..14} | (reads) | same |
| L25 gate c33 | 6% / 0 | **res** (R2 0.77): res in {102..104, 118..120, 123..124} | off (on 0) | (reads) | same |
| L25 gate c36 | 10% / 6% | **res%10** (R2 0.94): res mod 10 in {7} | **res** (R2 0.63): res in {-97, 7, 17, 27, 37, 47, 57, 67, 77, 87, 97} | (reads) | same |
| L25 gate c40 | 12% / 9% | **res%100** (R2 0.87): res mod 100 in {0..6, 95..99} | **res%100** (R2 0.65): res mod 100 in {0..5, 95..96} | (reads) | same |
| L25 gate c41 | 0 / 0 | off (on 0) | same | (reads) | same |
| L25 gate c42 | 34% / 8% | **res** (R2 0.72): res in {2..50, 52, 68, 76, 78..80, 82..90, 96, 126, 128..130, 138..140, 142..144, 146, 183, 186} | unexplained (best R2 0.38) | (reads) | same |
| L25 gate c46 | 11% / 6% | **res%10** (R2 0.86): res mod 10 in {9} | **res** (R2 0.55): res in {-99, 9, 18..19, 29, 39, 79, 89, 99} | (reads) | same |
| L25 gate c48 | 2% / 18% | **res** (R2 0.65): res in {2, 4, 8..12, 109, 189} | **res%100** (R2 0.54): res mod 100 in {7..14} | (reads) | same |
| L25 gate c50 | 17% / 8% | **res%100** (R2 0.84): res mod 100 in {3, 13, 20, 22..28, 33, 43, 53, 63..64, 73, 83, 93} | **res** (R2 0.65): res in {3, 13, 20, 22..28, 33, 43, 83} | (reads) | same |
| L25 gate c53 | 10% / 6% | **res%10** (R2 0.99): res mod 10 in {5} | **res%100** (R2 0.61): res mod 100 in {5, 15, 25, 85, 95} | (reads) | same |
| L25 gate c61 | 2% / 1% | **res** (R2 0.51): res in {124, 142..143, 182..183} | unexplained (best R2 0.10) | (reads) | same |
| L25 gate c63 | 9% / 4% | **res%10** (R2 0.82): res mod 10 in {0} | **res** (R2 0.59): res in {10, 20, 30, 90} | (reads) | same |
| L25 gate c66 | 14% / 5% | **res%100** (R2 0.84): res mod 100 in {41..46, 81..86} | unexplained (best R2 0.44) | (reads) | same |
| L25 gate c74 | 28% / 8% | **res%100** (R2 0.58): res mod 100 in {14..15, 20, 22..25, 60, 62..66, 68..80} | unexplained (best R2 0.32) | (reads) | same |
| L25 gate c76 | 3% / 1% | **res%100** (R2 0.77): res mod 100 in {23, 63} | **res** (R2 0.51): res in {23} | (reads) | same |
| L25 gate c79 | 6% / 0 | **res%100** (R2 0.89): res mod 100 in {1} | off (on 0) | (reads) | same |
| L25 gate c92 | 84% / 6% | **res** (R2 0.84): res in {21, 31, 41, 45, 51, 55, 58, 60..66, 68..76, 78, 80..200} | unexplained (best R2 0.40) | (reads) | same |
| L25 gate c98 | 2% / 10% | **res%100** (R2 0.72): res mod 100 in {91, 93} | unexplained (best R2 0.39) | (reads) | same |
| L25 gate c106 | 0 / 1% | off (on 0) | unexplained (best R2 0.10) | (reads) | same |
| L25 gate c110 | 12% / 6% | **res%100** (R2 0.92): res mod 100 in {7, 16..18, 27, 37, 47, 57, 67, 77, 87, 97} [coarser: res mod 20 in {7, 17}, R2 0.83] | **res** (R2 0.65): res in {7, 16..18, 27, 37, 97} | (reads) | same |
| L25 gate c111 | 100% / 100% | always | same | (reads) | same |
| L25 gate c139 | 10% / 1% | **res%100** (R2 0.87): res mod 100 in {88..95, 98} | unexplained (best R2 0.48) | (reads) | same |
| L25 gate c149 | 5% / 1% | **res%50** (R2 0.55): res mod 50 in {1, 21, 31} | unexplained (best R2 0.40) | (reads) | same |
| L25 gate c151 | 6% / 4% | **res%100** (R2 0.92): res mod 100 in {0..5} | **res%100** (R2 0.73): res mod 100 in {2} | (reads) | same |
| L25 gate c174 | 6% / 2% | **res%100** (R2 0.85): res mod 100 in {18..19, 38, 58, 78, 98} | **res** (R2 0.59): res in {18..19} | (reads) | same |
| L25 gate c190 | 10% / 0 | **res%100** (R2 0.77): res mod 100 in {53, 92..96, 98} | off (on 0) | (reads) | same |
| L25 gate c197 | 0 / 0 | off (on 0) | same | (reads) | same |
| L25 gate c200 | 22% / 8% | **res%100** (R2 0.83): res mod 100 in {38..60} | unexplained (best R2 0.47) | (reads) | same |
| L25 gate c216 | 11% / 11% | **res%10** (R2 0.86): res mod 10 in {1} | **res%100** (R2 0.47): res mod 100 in {1, 11, 21, 31, 41, 51, 61, 71, 81, 91} [coarser: res mod 50 in {1, 11, 21, 31, 41}, R2 0.80] | (reads) | same |
| L25 gate c220 | 4% / 5% | **res%100** (R2 0.83): res mod 100 in {4} | **res%100** (R2 0.63): res mod 100: no class above 0.5 (max 0.45) | (reads) | same |
| L25 gate c243 | 5% / 0 | **res%100** (R2 0.88): res mod 100 in {90..93} | off (on 0) | (reads) | same |
| L25 gate c248 | 1% / 0 | **res%100** (R2 0.69): res mod 100 in {75} | off (on 0) | (reads) | same |
| L25 gate c256 | 11% / 0 | **res//10** (R2 0.81): (tens) res in {152..154, 156..200} | off (on 0) | (reads) | same |
| L25 gate c297 | 2% / 0 | **res** (R2 0.84): res in {90, 186..193} | off (on 0) | (reads) | same |
| L25 gate c314 | 6% / 3% | **res%100** (R2 0.89): res mod 100 in {7, 17, 27, 37, 47, 97} | **res** (R2 0.77): res in {7, 17, 27, 97} | (reads) | same |
| L25 gate c491 | 1% / 1% | **res%100** (R2 0.82): res mod 100: no class above 0.5 (max 0.49) | **cmp(a,b)** (R2 0.74): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.88, cmp(a,b)=1: 0.00 | (reads) | same |
| L25 gate c535 | 9% / 3% | **res%10** (R2 0.88): res mod 10 in {9} | **res** (R2 0.62): res in {9, 19, 29, 39, 79, 99} | (reads) | same |
| L25 gate c538 | 1% / 2% | **res%100** (R2 0.65): res mod 100 in {15} | unexplained (best R2 0.47) | (reads) | same |
| L25 gate c670 | 9% / 4% | **res%10** (R2 0.94): res mod 10 in {7} | **res** (R2 0.64): res in {7, 17, 27, 37, 87, 97} | (reads) | same |
| L25 gate c683 | 6% / 2% | **res%100** (R2 0.87): res mod 100 in {31, 41, 51, 61, 71, 81, 91} | unexplained (best R2 0.38) | (reads) | same |
| L25 gate c748 | 9% / 9% | **res%100** (R2 0.87): res mod 100 in {8, 10..14, 16, 18} | **res** (R2 0.78): res in {8..14, 16..18} | (reads) | same |
| L25 gate c754 | 19% / 6% | **res%100** (R2 0.83): res mod 100 in {13, 15..16, 32..36, 52..56, 73, 75, 93, 95..96} [coarser: res mod 20 in {12..13, 15..16}, R2 0.85] | **res** (R2 0.62): res in {13, 15..16, 32..36, 93, 95..96} | (reads) | same |
| L25 gate c797 | 5% / 1% | **res%20** (R2 0.81): res mod 20 in {13} | unexplained (best R2 0.34) | (reads) | same |
| L25 gate c830 | 10% / 3% | **res%10** (R2 0.96): res mod 10 in {9} | **res** (R2 0.67): res in {9, 19, 29, 39, 79, 89, 99} | (reads) | same |
| L25 gate c898 | 9% / 1% | **res%10** (R2 0.83): res mod 10 in {7} | unexplained (best R2 0.50) | (reads) | same |
| L25 gate c901 | 3% / 0 | **res** (R2 0.88): res in {97, 137, 147, 157, 197} | off (on 0) | (reads) | same |

</details>

<details><summary>up: 40</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L25 up c11 | 21% / 3% | **res%100** (R2 0.78): res mod 100 in {1, 27, 31, 37, 41, 47, 51, 57, 61, 65, 67, 71, 77, 81, 87, 91, 97} | unexplained (best R2 0.27) | (reads) | same |
| L25 up c12 | 21% / 38% | **res%10** (R2 0.96): res mod 10 in {1, 7} | unexplained (best R2 0.49) | (reads) | same |
| L25 up c15 | 10% / 6% | **res%100** (R2 0.81): res mod 100 in {1..4, 23..24, 43, 63, 83} | **res%100** (R2 0.67): res mod 100: no class above 0.5 (max 0.46) | (reads) | same |
| L25 up c30 | 30% / 3% | **res** (R2 0.78): res in {53, 59..64, 69..73, 99..104, 109..113, 142..144, 149..154, 158..165, 167..174, 181..183} | unexplained (best R2 0.15) | (reads) | same |
| L25 up c31 | 8% / 1% | **res%100** (R2 0.69): res mod 100 in {18..19} | **res** (R2 0.67): res in {18..19} | (reads) | same |
| L25 up c33 | 47% / 2% | **res//10** (R2 0.88): (tens) res in {105..198, 200} | unexplained (best R2 0.34) | (reads) | same |
| L25 up c40 | 25% / 13% | **res%50** (R2 0.84): res mod 50 in {0..1, 9..11, 19..20, 29..30, 39..40, 49} [coarser: res mod 10 in {0..1, 9}, R2 0.85] | **res%100** (R2 0.47): res mod 100 in {0..1, 9..11, 19, 89..90, 99} | (reads) | same |
| L25 up c42 | 19% / 8% | **res%100** (R2 0.83): res mod 100 in {10..14, 33, 53, 61..63, 70..73, 90..94} | **res** (R2 0.58): res in {10..14, 72..73, 90..94} | (reads) | same |
| L25 up c46 | 20% / 7% | **res** (R2 0.75): res in {9, 15..24, 91..95, 116..123, 147, 150..157, 165..167, 190..196} [coarser: res mod 100 in {16..23, 91..96}, R2 0.81] | **res** (R2 0.64): res in {16..21, 23, 91..95} | (reads) | same |
| L25 up c48 | 33% / 11% | **res%100** (R2 0.75): res mod 100 in {0, 2..7, 12..16, 25, 43..46, 55..57, 65..67, 75..76, 85..86, 92..97} | **res** (R2 0.61): res in {3..7, 13..17, 93..96} | (reads) | same |
| L25 up c50 | 9% / 4% | **res%10** (R2 0.90): res mod 10 in {3} | **res** (R2 0.63): res in {3, 13, 23, 83, 93} | (reads) | same |
| L25 up c55 | 0 / 26% | off (on 0) | **tens(a,b)** (R2 0.53): a//10 in {1,2} -> b//10 in {1}; a//10 in {3} -> b//10 in {2}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {4,7}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7,8} -> b//10 in {6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {7,8,9} | (reads) | same |
| L25 up c63 | 10% / 2% | **res%100** (R2 0.85): res mod 100 in {10, 40, 50, 90..94} | unexplained (best R2 0.38) | (reads) | same |
| L25 up c66 | 16% / 3% | **res%100** (R2 0.81): res mod 100 in {1, 8..10, 89, 97..99} | **res** (R2 0.58): res in {-99, 8..10, 89, 97..99} | (reads) | same |
| L25 up c73 | 5% / 0 | **res//10** (R2 0.78): (tens) res in {169..200} | off (on 0) | (reads) | same |
| L25 up c74 | 37% / 7% | **res//10** (R2 0.67): (tens) res in {14..16, 18, 20..90, 123} | unexplained (best R2 0.32) | (reads) | same |
| L25 up c76 | 20% / 5% | **res%100** (R2 0.77): res mod 100 in {18, 36..39, 57..59, 76..79, 88, 94..99} | unexplained (best R2 0.36) | (reads) | same |
| L25 up c92 | 0 / 1% | off (on 0) | **res%100** (R2 0.61): res mod 100: no class above 0.5 (max 0.37) | (reads) | same |
| L25 up c98 | 8% / 2% | **res%100** (R2 0.86): res mod 100 in {14, 24, 54, 64, 74, 84, 94} [coarser: res mod 50 in {4, 14, 24, 44}, R2 0.86] | **res** (R2 0.64): res in {14, 24, 84} | (reads) | same |
| L25 up c108 | 19% / 11% | **res%10** (R2 0.86): res mod 10 in {0, 9} | **units(a,b)** (R2 0.48): a%10 in {0} -> b%10 in {0,1}; a%10 in {1} -> b%10 in {1,2}; a%10 in {4} -> b%10 in {4}; a%10 in {7,8} -> b%10 in {8}; a%10 in {9} -> b%10 in {0,9} | (reads) | same |
| L25 up c110 | 9% / 7% | **res%100** (R2 0.84): res mod 100 in {13..19} | **res** (R2 0.79): res in {12..20} | (reads) | same |
| L25 up c149 | 5% / 4% | **res%100** (R2 0.81): res mod 100 in {8..9, 88..89, 98..99} | unexplained (best R2 0.50) | (reads) | same |
| L25 up c151 | 22% / 7% | **res%100** (R2 0.80): res mod 100 in {6..9, 26..29, 46..49, 58, 66..68, 86..89, 96..99} | **res** (R2 0.62): res in {-97, 7..9, 26..29, 86..89, 96..99} | (reads) | same |
| L25 up c200 | 1% / 0 | **res%100** (R2 0.66): res mod 100 in {24, 64} | **res** (R2 0.53): res in {24} | (reads) | same |
| L25 up c206 | 0 / 1% | off (on 0) | **res%100** (R2 0.74): res mod 100: no class above 0.5 (max 0.38) | (reads) | same |
| L25 up c207 | 10% / 5% | **res%10** (R2 0.89): res mod 10 in {2} | **res** (R2 0.57): res in {2, 12..13, 22, 32, 72, 92} | (reads) | same |
| L25 up c218 | 100% / 100% | always | same | (reads) | same |
| L25 up c248 | 6% / 2% | **res%100** (R2 0.75): res mod 100 in {15, 25, 35, 55, 75, 95} [coarser: res mod 20 in {15}, R2 0.83] | **res** (R2 0.52): res in {15, 25, 95} | (reads) | same |
| L25 up c256 | 12% / 3% | **res** (R2 0.77): res in {75..77, 86, 165..200} | **res** (R2 0.60): res in {74..80, 82, 84..89} | (reads) | same |
| L25 up c276 | 3% / 0 | **res%100** (R2 0.66): res mod 100 in {72..74} | off (on 0) | (reads) | same |
| L25 up c297 | 20% / 3% | **res%100** (R2 0.75): res mod 100 in {1, 81, 83..98} | unexplained (best R2 0.40) | (reads) | same |
| L25 up c396 | 13% / 2% | **res%100** (R2 0.73): res mod 100 in {1, 39, 49, 59, 69, 79, 89, 99} | unexplained (best R2 0.46) | (reads) | same |
| L25 up c479 | 0 / 2% | off (on 0) | **res** (R2 0.82): res in {10..11} | (reads) | same |
| L25 up c489 | 1% / 1% | **res%100** (R2 0.86): res mod 100: no class above 0.5 (max 0.44) | **cmp(a,b)** (R2 0.80): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.85, cmp(a,b)=1: 0.00 | (reads) | same |
| L25 up c553 | 2% / 8% | **res%100** (R2 0.68): res mod 100 in {8..9} | **res%100** (R2 0.67): res mod 100 in {8..10} | (reads) | same |
| L25 up c572 | 7% / 11% | **res//10** (R2 0.72): (tens) res in {2..33, 35} | **tens(a,b)** (R2 0.62): a//10 in {0,1,2} -> b//10 in {0,1,2} | (reads) | same |
| L25 up c580 | 1% / 0 | **res%100** (R2 0.77): res mod 100: no class above 0.5 (max 0.40) | off (on 0) | (reads) | same |
| L25 up c735 | 7% / 4% | **res** (R2 0.65): res in {4, 6, 56..60, 100, 156..164} | **res%100** (R2 0.68): res mod 100 in {0, 2, 99} | (reads) | same |
| L25 up c851 | 4% / 0 | **res//10** (R2 0.79): (tens) res in {171..200} | off (on 0) | (reads) | same |
| L25 up c873 | 1% / 0 | **res%100** (R2 0.71): res mod 100: no class above 0.5 (max 0.39) | off (on 0) | (reads) | same |

</details>

<details><summary>v: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L25 v c122 (kv5) | 0 / 0 | off (on 0) | same | (reads) | same |

</details>

</details>

<details><summary>layer 26: 168 components with main position `=`</summary>

<details><summary>down: 66</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L26 down c1 | 0 / 0 | off (on 0) | same | - | same |
| L26 down c2 | 11% / 1% | **res%100** (R2 0.72): res mod 100 in {66..74, 76..77} | unexplained (best R2 0.15) | - | same |
| L26 down c4 | 19% / 8% | **res** (R2 0.90): res in {23..29, 119..129, 131..143} | **res** (R2 0.57): res in {22..29, 34..35} | res: mod25 +3% | - |
| L26 down c7 | 52% / 0 | **res>=100** (R2 0.75): res>=100=0: 0.07, res>=100=1: 0.94 | off (on 0) | a: mod100 +3%; b: mod100 +3% | - |
| L26 down c8 | 7% / 9% | **res%100** (R2 0.84): res mod 100 in {19..24} | unexplained (best R2 0.49) | - | res: mod25 +2% |
| L26 down c10 | 0 / 96% | off (on 0) | always | - | same |
| L26 down c12 | 9% / 4% | **res%100** (R2 0.89): res mod 100 in {57, 59..66} | unexplained (best R2 0.28) | res: mod25 +2% | - |
| L26 down c14 | 100% / 100% | always | same | - | same |
| L26 down c20 | 7% / 10% | **res%100** (R2 0.82): res mod 100 in {29..34} | unexplained (best R2 0.40) | - | same |
| L26 down c23 | 5% / 5% | **res%100** (R2 0.75): res mod 100: no class above 0.5 (max 0.45) | **res%100** (R2 0.68): res mod 100: no class above 0.5 (max 0.44) | - | same |
| L26 down c24 | 16% / 3% | **res%100** (R2 0.73): res mod 100 in {25, 32..36, 38, 53..56, 58, 83..86, 88} | unexplained (best R2 0.30) | - | same |
| L26 down c33 | 8% / 8% | **res%100** (R2 0.81): res mod 100 in {12, 16..20} | **res** (R2 0.60): res in {12..13, 16..21} | - | same |
| L26 down c36 | 41% / 30% | **res%2** (R2 0.68): res mod 2 in {0} | **units(a,b)** (R2 0.67): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} | a: mod2 +5%; b: mod2 +3%; res: mod2 +8% | a: mod2 +8%; b: mod2 +7%; res: mod2 +10% |
| L26 down c37 | 12% / 7% | **res%100** (R2 0.82): res mod 100 in {7..8, 17..18, 27, 37..38, 57..58, 87, 97..98} | **res** (R2 0.66): res in {-97, 7..8, 17..18, 27, 37..38, 87, 97..98} | res: mod4 +3% | res: mod4 +4% |
| L26 down c72 | 1% / 7% | unexplained (best R2 0.11) | unexplained (best R2 0.42) | - | same |
| L26 down c75 | 1% / 27% | unexplained (best R2 0.23) | **tens(a,b)** (R2 0.62): a//10 in {0} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,9}; a//10 in {3,4,5,6,7,8,10} -> b//10 in {0}; a//10 in {9} -> b//10 in {0,1,2,3,4,5} | - | same |
| L26 down c85 | 10% / 14% | **res//10** (R2 0.73): (tens) res in {2..45} | **tens(a,b)** (R2 0.71): a//10 in {0} -> b//10 in {0,1,2,3,4}; a//10 in {1,2} -> b//10 in {0,1,2}; a//10 in {3,4} -> b//10 in {0} | - | same |
| L26 down c86 | 8% / 0 | **res//10** (R2 0.77): (tens) res in {101..109} | off (on 0) | - | same |
| L26 down c88 | 5% / 1% | **res%100** (R2 0.70): res mod 100 in {46, 48, 86, 88} | unexplained (best R2 0.28) | res: mod4 +3% | - |
| L26 down c89 | 7% / 4% | **res%100** (R2 0.90): res mod 100 in {76..82} | unexplained (best R2 0.33) | - | same |
| L26 down c94 | 6% / 1% | **res%100** (R2 0.81): res mod 100 in {53..54, 56..59} | unexplained (best R2 0.19) | - | same |
| L26 down c95 | 17% / 0 | **res** (R2 0.78): res in {120..121, 125..141, 150..151, 160..161, 170..171, 175..176, 179..181} | off (on 0) | - | same |
| L26 down c106 | 0 / 1% | off (on 0) | **res%100** (R2 0.79): res mod 100: no class above 0.5 (max 0.41) | - | same |
| L26 down c113 | 8% / 4% | **res%100** (R2 0.81): res mod 100 in {44..50} | unexplained (best R2 0.32) | - | same |
| L26 down c115 | 3% / 16% | **tens(a,b)** (R2 0.56): a//10 in {0} -> b//10 in {2,3}; a//10 in {1} -> b//10 in {2}; a//10 in {3} -> b//10 in {0} | **tens(a,b)** (R2 0.55): a//10 in {3} -> b//10 in {0,3,4,5,10}; a//10 in {4} -> b//10 in {0,1,5}; a//10 in {5} -> b//10 in {1,2}; a//10 in {6} -> b//10 in {2,3}; a//10 in {7} -> b//10 in {3,4}; a//10 in {8} -> b//10 in {4,5}; a//10 in {10} -> b//10 in {6,7} | - | same |
| L26 down c118 | 18% / 6% | **res%100** (R2 0.93): res mod 100 in {69..86} | unexplained (best R2 0.46) | - | same |
| L26 down c119 | 0 / 0 | off (on 0) | same | - | same |
| L26 down c136 | 0 / 0 | off (on 0) | same | - | same |
| L26 down c137 | 0 / 0 | off (on 0) | same | - | same |
| L26 down c139 | 9% / 7% | **res%100** (R2 0.71): res mod 100 in {0..5, 96..99} | unexplained (best R2 0.44) | - | same |
| L26 down c142 | 6% / 1% | **res%100** (R2 0.57): res mod 100 in {55, 75, 94..99} | **res** (R2 0.50): res in {15, 99} | - | same |
| L26 down c154 | 1% / 37% | unexplained (best R2 0.09) | unexplained (best R2 0.37) | - | a: mod4 +5%; b: mod4 +5%; res: mod4 +5% |
| L26 down c155 | 11% / 2% | **res** (R2 0.74): res in {89..91, 109..110, 129..130, 139, 149..150, 169..170, 180, 189..194, 200} | unexplained (best R2 0.18) | - | same |
| L26 down c157 | 5% / 0 | **res%100** (R2 0.52): res mod 100 in {45, 94..95} | unexplained (best R2 0.23) | - | same |
| L26 down c175 | 10% / 5% | **res%10** (R2 0.97): res mod 10 in {1} | **res** (R2 0.59): res in {11, 21, 31, 41, 51, 61, 71, 91} | - | same |
| L26 down c176 | 7% / 1% | **res%100** (R2 0.60): res mod 100 in {84..88} | unexplained (best R2 0.38) | - | same |
| L26 down c180 | 1% / 2% | **res%100** (R2 0.67): res mod 100: no class above 0.5 (max 0.42) | **res%100** (R2 0.66): res mod 100: no class above 0.5 (max 0.44) | - | same |
| L26 down c182 | 3% / 13% | **res** (R2 0.77): res in {20..30} | unexplained (best R2 0.48) | - | same |
| L26 down c187 | 2% / 4% | unexplained (best R2 0.09) | unexplained (best R2 0.22) | - | same |
| L26 down c189 | 4% / 4% | unexplained (best R2 0.30) | unexplained (best R2 0.33) | - | same |
| L26 down c194 | 11% / 1% | **res%100** (R2 0.77): res mod 100 in {18, 28, 38, 48, 57..59, 68, 78..79, 98} | unexplained (best R2 0.33) | - | same |
| L26 down c205 | 5% / 5% | **res%100** (R2 0.83): res mod 100 in {10..15} | **res** (R2 0.79): res in {10..15} | - | same |
| L26 down c207 | 5% / 3% | **res%100** (R2 0.70): res mod 100 in {9} | **res%100** (R2 0.81): res mod 100: no class above 0.5 (max 0.46) | - | same |
| L26 down c219 | 1% / 0 | **res%100** (R2 0.83): res mod 100 in {61} | off (on 0) | - | same |
| L26 down c226 | 4% / 1% | **res%100** (R2 0.80): res mod 100 in {45..49} | unexplained (best R2 0.22) | - | same |
| L26 down c237 | 7% / 3% | **res%100** (R2 0.84): res mod 100 in {49..55} | unexplained (best R2 0.27) | - | same |
| L26 down c258 | 8% / 1% | **res%100** (R2 0.76): res mod 100 in {13, 33, 53, 63, 73..74, 83, 93} | unexplained (best R2 0.48) | - | same |
| L26 down c271 | 0 / 1% | off (on 0) | **cmp(a,b)** (R2 0.64): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.90, cmp(a,b)=1: 0.00 | - | same |
| L26 down c299 | 8% / 1% | **res%100** (R2 0.82): res mod 100 in {72..78} | unexplained (best R2 0.22) | - | same |
| L26 down c341 | 7% / 2% | **res%100** (R2 0.83): res mod 100 in {91..96} | unexplained (best R2 0.32) | - | same |
| L26 down c348 | 7% / 2% | **res%100** (R2 0.69): res mod 100 in {25, 34..35, 74..75, 94} | unexplained (best R2 0.36) | - | same |
| L26 down c360 | 5% / 1% | **res** (R2 0.76): res in {41..42, 141..146} [coarser: res mod 100 in {41..45}, R2 0.82] | unexplained (best R2 0.25) | - | same |
| L26 down c363 | 6% / 1% | **res%20** (R2 0.77): res mod 20 in {12} | unexplained (best R2 0.36) | - | same |
| L26 down c393 | 3% / 1% | **res** (R2 0.74): res in {134, 136..139} | unexplained (best R2 0.35) | - | same |
| L26 down c486 | 0 / 1% | off (on 0) | **res%100** (R2 0.75): res mod 100: no class above 0.5 (max 0.42) | - | same |
| L26 down c529 | 9% / 0 | **res%100** (R2 0.71): res mod 100 in {1} | off (on 0) | - | same |
| L26 down c552 | 0 / 14% | off (on 0) | **res%100** (R2 0.55): res mod 100: no class above 0.5 (max 0.47) | - | same |
| L26 down c570 | 5% / 0 | **res%100** (R2 0.86): res mod 100 in {90..94} | off (on 0) | - | same |
| L26 down c605 | 15% / 0 | **res//10** (R2 0.83): (tens) res in {147, 149..200} | off (on 0) | - | same |
| L26 down c641 | 1% / 0 | **res//10** (R2 0.51): (tens) res in {181..182, 186, 192, 194..200} | off (on 0) | - | same |
| L26 down c703 | 4% / 1% | **res%100** (R2 0.78): res mod 100 in {65, 67..69} | unexplained (best R2 0.21) | - | same |
| L26 down c779 | 0 / 1% | off (on 0) | **res** (R2 0.84): res in {10} | - | same |
| L26 down c782 | 0 / 0 | off (on 0) | same | - | same |
| L26 down c789 | 1% / 0 | **res** (R2 0.54): res in {179..180} | off (on 0) | - | same |
| L26 down c813 | 4% / 3% | **res%100** (R2 0.77): res mod 100 in {14..17} | **res** (R2 0.74): res in {14..17} | - | same |
| L26 down c840 | 2% / 0 | **res%100** (R2 0.79): res mod 100 in {53, 93} | off (on 0) | - | same |

</details>

<details><summary>gate: 52</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L26 gate c4 | 20% / 12% | **res** (R2 0.92): res in {118..144} | **res//10** (R2 0.66): (tens) res in {21..38} | (reads) | same |
| L26 gate c7 | 6% / 0 | **res** (R2 0.92): res in {120..127} | off (on 0) | (reads) | same |
| L26 gate c8 | 11% / 8% | **res%100** (R2 0.82): res mod 100 in {19..23, 47..48, 86..89} | **res** (R2 0.52): res in {19..23, 86..89} | (reads) | same |
| L26 gate c10 | 99% / 100% | always | same | (reads) | same |
| L26 gate c12 | 11% / 5% | **res%100** (R2 0.92): res mod 100 in {56..66} | unexplained (best R2 0.28) | (reads) | same |
| L26 gate c20 | 13% / 10% | **res%100** (R2 0.78): res mod 100 in {28..34, 60..65} | unexplained (best R2 0.44) | (reads) | same |
| L26 gate c24 | 12% / 3% | **res%100** (R2 0.79): res mod 100 in {31..36, 53..56, 83..86} | **res** (R2 0.53): res in {31..35} | (reads) | same |
| L26 gate c33 | 12% / 11% | **res%100** (R2 0.87): res mod 100 in {10..21} | **res** (R2 0.72): res in {10..22} | (reads) | same |
| L26 gate c37 | 15% / 7% | **res%100** (R2 0.88): res mod 100 in {7..8, 17..18, 27..28, 37..38, 57..58, 67..68, 87, 97..98} [coarser: res mod 20 in {7..8, 17..18}, R2 0.81] | **res** (R2 0.67): res in {-97, 7..8, 17..18, 27..28, 37..38, 97..98} | (reads) | same |
| L26 gate c72 | 1% / 1% | unexplained (best R2 0.06) | unexplained (best R2 0.09) | (reads) | same |
| L26 gate c83 | 3% / 2% | **res%100** (R2 0.72): res mod 100 in {9} | **res%100** (R2 0.81): res mod 100: no class above 0.5 (max 0.48) | (reads) | same |
| L26 gate c88 | 5% / 0 | **res%100** (R2 0.77): res mod 100 in {45..48, 86, 88} | off (on 0) | (reads) | same |
| L26 gate c89 | 13% / 4% | **res%100** (R2 0.87): res mod 100 in {72..84} | unexplained (best R2 0.35) | (reads) | same |
| L26 gate c94 | 16% / 3% | **res%100** (R2 0.88): res mod 100 in {52..66, 68} | unexplained (best R2 0.41) | (reads) | same |
| L26 gate c95 | 7% / 0 | **res** (R2 0.70): res in {120..121, 130..131, 160..161, 170..171, 176..177, 179..182, 190, 200} | off (on 0) | (reads) | same |
| L26 gate c113 | 6% / 2% | **res%100** (R2 0.84): res mod 100 in {44..49} | unexplained (best R2 0.29) | (reads) | same |
| L26 gate c115 | 6% / 4% | **res** (R2 0.84): res in {25..35, 130..133} [coarser: res mod 100 in {29..33}, R2 0.80] | **res** (R2 0.51): res in {27..32} | (reads) | same |
| L26 gate c118 | 19% / 8% | **res%100** (R2 0.93): res mod 100 in {67..85} | **tens(a,b)** (R2 0.52): a//10 in {7} -> b//10 in {0,10}; a//10 in {8} -> b//10 in {0,1}; a//10 in {9} -> b//10 in {1,2}; a//10 in {10} -> b//10 in {2,3} | (reads) | same |
| L26 gate c123 | 2% / 4% | unexplained (best R2 0.09) | unexplained (best R2 0.22) | (reads) | same |
| L26 gate c136 | 4% / 3% | **res%100** (R2 0.86): res mod 100 in {20..22, 95} | **res** (R2 0.67): res in {20..22} | (reads) | same |
| L26 gate c137 | 0 / 2% | off (on 0) | **res%100** (R2 0.74): res mod 100: no class above 0.5 (max 0.41) | (reads) | same |
| L26 gate c139 | 10% / 11% | **res%100** (R2 0.74): res mod 100 in {0..6, 95..99} | **res%100** (R2 0.50): res mod 100 in {0..4, 97..99} | (reads) | same |
| L26 gate c142 | 7% / 1% | **res%100** (R2 0.68): res mod 100 in {15, 55..56, 59, 75, 95..96} | unexplained (best R2 0.46) | (reads) | same |
| L26 gate c152 | 0 / 1% | off (on 0) | **res%100** (R2 0.54): res mod 100: no class above 0.5 (max 0.28) | (reads) | same |
| L26 gate c155 | 8% / 2% | **res%100** (R2 0.76): res mod 100 in {9, 69, 88..92} | unexplained (best R2 0.49) | (reads) | same |
| L26 gate c157 | 7% / 2% | **res%100** (R2 0.65): res mod 100 in {14, 44..47, 64, 74, 94} | **res%100** (R2 0.47): res mod 100: no class above 0.5 (max 0.41) | (reads) | same |
| L26 gate c180 | 0 / 1% | off (on 0) | **res%100** (R2 0.75): res mod 100: no class above 0.5 (max 0.45) | (reads) | same |
| L26 gate c182 | 3% / 2% | **res%100** (R2 0.88): res mod 100 in {31..33} | **res** (R2 0.57): res in {31..33} | (reads) | same |
| L26 gate c187 | 1% / 1% | unexplained (best R2 0.05) | unexplained (best R2 0.08) | (reads) | same |
| L26 gate c194 | 10% / 1% | **res%100** (R2 0.69): res mod 100 in {54, 56..63, 78} | unexplained (best R2 0.19) | (reads) | same |
| L26 gate c205 | 4% / 4% | **res%100** (R2 0.82): res mod 100 in {11..13, 15} | **res** (R2 0.78): res in {11..15} | (reads) | same |
| L26 gate c207 | 5% / 5% | **res%100** (R2 0.77): res mod 100 in {20..23} | **res** (R2 0.60): res in {20..23} | (reads) | same |
| L26 gate c219 | 4% / 1% | **res%100** (R2 0.95): res mod 100 in {60..63} | unexplained (best R2 0.14) | (reads) | same |
| L26 gate c226 | 12% / 3% | **res%100** (R2 0.79): res mod 100 in {40..52} | unexplained (best R2 0.33) | (reads) | same |
| L26 gate c237 | 10% / 4% | **res%100** (R2 0.89): res mod 100 in {45..54} | unexplained (best R2 0.33) | (reads) | same |
| L26 gate c258 | 0 / 0 | off (on 0) | same | (reads) | same |
| L26 gate c260 | 0 / 1% | off (on 0) | **res%100** (R2 0.79): res mod 100: no class above 0.5 (max 0.41) | (reads) | same |
| L26 gate c299 | 15% / 2% | **res%100** (R2 0.93): res mod 100 in {66..80} | unexplained (best R2 0.41) | (reads) | same |
| L26 gate c320 | 10% / 2% | **res%100** (R2 0.78): res mod 100 in {0..1, 91..99} | unexplained (best R2 0.22) | (reads) | same |
| L26 gate c360 | 2% / 0 | unexplained (best R2 0.49) | off (on 0) | (reads) | same |
| L26 gate c363 | 4% / 1% | **res%100** (R2 0.76): res mod 100 in {32, 52, 72} | unexplained (best R2 0.43) | (reads) | same |
| L26 gate c393 | 9% / 4% | **res//10** (R2 0.86): (tens) res in {2..41} | unexplained (best R2 0.38) | (reads) | same |
| L26 gate c486 | 0 / 1% | off (on 0) | **res%100** (R2 0.82): res mod 100: no class above 0.5 (max 0.44) | (reads) | same |
| L26 gate c513 | 6% / 1% | **res%100** (R2 0.69): res mod 100 in {85..89} | unexplained (best R2 0.49) | (reads) | same |
| L26 gate c529 | 6% / 7% | **res%100** (R2 0.71): res mod 100 in {1..3} | **res%100** (R2 0.69): res mod 100: no class above 0.5 (max 0.44) | (reads) | same |
| L26 gate c552 | 5% / 3% | **res%100** (R2 0.79): res mod 100 in {43, 53, 63, 73, 93} | **res%100** (R2 0.58): res mod 100: no class above 0.5 (max 0.45) | (reads) | same |
| L26 gate c570 | 8% / 2% | **res%100** (R2 0.74): res mod 100 in {9, 19, 39, 59, 69, 89..91, 93} [coarser: res mod 50 in {9, 19, 39}, R2 0.81] | **res** (R2 0.69): res in {9, 19, 89} | (reads) | same |
| L26 gate c703 | 3% / 0 | **res** (R2 0.90): res in {159..165} | off (on 0) | (reads) | same |
| L26 gate c707 | 5% / 4% | **res** (R2 0.79): res in {76..79, 176..182, 196..200} | unexplained (best R2 0.22) | (reads) | same |
| L26 gate c712 | 99% / 3% | always | unexplained (best R2 0.31) | (reads) | same |
| L26 gate c782 | 0 / 2% | off (on 0) | **res%100** (R2 0.78): res mod 100: no class above 0.5 (max 0.43) | (reads) | same |
| L26 gate c813 | 1% / 1% | **res%100** (R2 0.68): res mod 100 in {20} | **res** (R2 0.57): res in {20} | (reads) | same |

</details>

<details><summary>o: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L26 o c379 (H14) | 3% / 4% | **a** (R2 0.61): a in {92..94} | **a** (R2 0.81): a in {92..94} | - | same |

</details>

<details><summary>up: 49</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L26 up c4 | 12% / 9% | **res%100** (R2 0.90): res mod 100 in {20..30} | **res//10** (R2 0.56): (tens) res in {19..30} | (reads) | same |
| L26 up c7 | 9% / 1% | **res//10** (R2 0.72): (tens) res in {161, 163..200} | unexplained (best R2 0.19) | (reads) | same |
| L26 up c8 | 15% / 11% | **res%100** (R2 0.90): res mod 100 in {15..28} | **res** (R2 0.63): res in {16..26} | (reads) | same |
| L26 up c12 | 17% / 3% | **res%100** (R2 0.83): res mod 100 in {60..74, 77} | unexplained (best R2 0.38) | (reads) | same |
| L26 up c20 | 8% / 6% | **res%100** (R2 0.78): res mod 100 in {30..32, 60..62} | unexplained (best R2 0.39) | (reads) | same |
| L26 up c23 | 3% / 2% | unexplained (best R2 0.42) | unexplained (best R2 0.40) | (reads) | same |
| L26 up c24 | 11% / 5% | **res%50** (R2 0.69): res mod 50 in {3, 5, 23, 25, 33, 35} | **res** (R2 0.60): res in {3, 5, 13, 23, 25, 33, 35, 85} | (reads) | same |
| L26 up c33 | 7% / 8% | **res%100** (R2 0.76): res mod 100 in {9..15} | **res%100** (R2 0.55): res mod 100 in {10..14} | (reads) | same |
| L26 up c36 | 12% / 22% | **res%100** (R2 0.81): res mod 100 in {1, 3, 7, 9, 11, 21, 29, 31, 41, 51, 61, 71, 81, 91} | unexplained (best R2 0.40) | (reads) | same |
| L26 up c37 | 2% / 3% | **res%100** (R2 0.81): res mod 100 in {7..8} | **res%100** (R2 0.71): res mod 100 in {8} | (reads) | same |
| L26 up c46 | 2% / 0 | **res%100** (R2 0.80): res mod 100 in {77..78} | off (on 0) | (reads) | same |
| L26 up c72 | 99% / 13% | always | **res** (R2 0.51): res in {4..15, 88..89} | (reads) | same |
| L26 up c86 | 8% / 10% | **res//10** (R2 0.79): (tens) res in {2..4, 7..8, 101..109} [coarser: res mod 100 in {1..9}, R2 0.95] | **res//10** (R2 0.78): (tens) res in {0..9} | (reads) | same |
| L26 up c88 | 10% / 6% | **res%10** (R2 0.94): res mod 10 in {8} | unexplained (best R2 0.47) | (reads) | same |
| L26 up c89 | 20% / 6% | **res%100** (R2 0.84): res mod 100 in {16..19, 37, 47..48, 56..59, 66..68, 76..78, 87..88, 97} | **res** (R2 0.53): res in {16..19, 57, 77..78, 87..88, 97} | (reads) | same |
| L26 up c94 | 13% / 5% | **res%100** (R2 0.81): res mod 100 in {9, 17..19, 29, 39, 57..59, 69, 79, 89, 99} | **res** (R2 0.57): res in {9, 17..19, 29, 39, 69, 79, 89, 99} | (reads) | same |
| L26 up c95 | 15% / 7% | **res** (R2 0.77): res in {2..10, 12..14, 17, 110, 120, 129..141, 150, 160..161, 170..171, 180..181} | **res%100** (R2 0.63): res mod 100: no class above 0.5 (max 0.49) | (reads) | same |
| L26 up c113 | 10% / 4% | **res%100** (R2 0.81): res mod 100 in {42..50} | unexplained (best R2 0.33) | (reads) | same |
| L26 up c115 | 4% / 1% | **res//10** (R2 0.53): (tens) res in {30, 32, 34..39, 180..190, 192..200} | unexplained (best R2 0.21) | (reads) | same |
| L26 up c118 | 3% / 0 | **res** (R2 0.66): res in {77..78, 87, 147, 167, 177..178} [coarser: res mod 100 in {47, 77..78}, R2 0.87] | off (on 0) | (reads) | same |
| L26 up c119 | 0 / 1% | off (on 0) | unexplained (best R2 0.06) | (reads) | same |
| L26 up c142 | 7% / 1% | **res%100** (R2 0.68): res mod 100 in {0, 55, 94..99} | unexplained (best R2 0.26) | (reads) | same |
| L26 up c154 | 4% / 8% | unexplained (best R2 0.14) | unexplained (best R2 0.20) | (reads) | same |
| L26 up c155 | 10% / 2% | **res** (R2 0.77): res in {50, 89..92, 100, 109..110, 129..130, 149..150, 160, 169..170, 179..180, 189..192, 200} | unexplained (best R2 0.26) | (reads) | same |
| L26 up c157 | 10% / 4% | **res%10** (R2 0.84): res mod 10 in {5} | **res** (R2 0.51): res in {15, 25, 35, 45, 85, 95} | (reads) | same |
| L26 up c175 | 9% / 3% | **res%10** (R2 0.89): res mod 10 in {1} | unexplained (best R2 0.49) | (reads) | same |
| L26 up c178 | 14% / 4% | **res%100** (R2 0.88): res mod 100 in {27..29, 47..49, 57..59, 77..79, 88..89} | unexplained (best R2 0.46) | (reads) | same |
| L26 up c182 | 15% / 15% | **res%100** (R2 0.91): res mod 100 in {30..44} | **res** (R2 0.52): res in {30..44} | (reads) | same |
| L26 up c189 | 1% / 0 | unexplained (best R2 0.38) | off (on 0) | (reads) | same |
| L26 up c194 | 5% / 1% | **res%100** (R2 0.83): res mod 100 in {48, 57..59, 78} | unexplained (best R2 0.18) | (reads) | same |
| L26 up c205 | 2% / 2% | **res%100** (R2 0.79): res mod 100 in {12} | **res** (R2 0.80): res in {11..13} | (reads) | same |
| L26 up c207 | 5% / 4% | **res%100** (R2 0.74): res mod 100 in {7..9} | **res%100** (R2 0.71): res mod 100 in {8} | (reads) | same |
| L26 up c226 | 6% / 3% | **res%100** (R2 0.74): res mod 100 in {7, 27, 47, 67, 77, 87} | **res** (R2 0.51): res in {7, 27} | (reads) | same |
| L26 up c237 | 1% / 0 | **res%100** (R2 0.52): res mod 100 in {31} | off (on 0) | (reads) | same |
| L26 up c258 | 10% / 3% | **res%100** (R2 0.69): res mod 100 in {13..14, 34, 53..54, 73..75, 84, 93..94} | unexplained (best R2 0.40) | (reads) | same |
| L26 up c284 | 21% / 94% | **res** (R2 0.54): res in {3..33, 57, 59..65, 120..121, 160..163, 200} | unexplained (best R2 0.25) | (reads) | same |
| L26 up c299 | 5% / 0 | **res%100** (R2 0.58): res mod 100 in {78, 83, 85, 88} | off (on 0) | (reads) | same |
| L26 up c318 | 22% / 0 | **res//10** (R2 0.77): (tens) res in {131, 137, 139..200} | off (on 0) | (reads) | same |
| L26 up c320 | 4% / 1% | **res%100** (R2 0.60): res mod 100 in {84..87} | unexplained (best R2 0.23) | (reads) | same |
| L26 up c341 | 8% / 3% | **res%100** (R2 0.90): res mod 100 in {90..95} | unexplained (best R2 0.48) | (reads) | same |
| L26 up c348 | 9% / 4% | **res%10** (R2 0.88): res mod 10 in {6} | **res** (R2 0.55): res in {6, 16, 26, 36, 96} | (reads) | same |
| L26 up c360 | 7% / 2% | **res%100** (R2 0.81): res mod 100 in {52..59} | unexplained (best R2 0.32) | (reads) | same |
| L26 up c486 | 9% / 3% | **res%10** (R2 0.85): res mod 10 in {3} | **res** (R2 0.64): res in {3, 13, 33} | (reads) | same |
| L26 up c525 | 0 / 1% | off (on 0) | **res%100** (R2 0.54): res mod 100: no class above 0.5 (max 0.29) | (reads) | same |
| L26 up c540 | 5% / 0 | **res** (R2 0.82): res in {134..139} | off (on 0) | (reads) | same |
| L26 up c552 | 0 / 19% | off (on 0) | **tens(a,b)** (R2 0.57): a//10 in {5} -> b//10 in {5,6,7}; a//10 in {6} -> b//10 in {6,7,8,9,10}; a//10 in {7} -> b//10 in {7,8,9,10}; a//10 in {8} -> b//10 in {8,9,10}; a//10 in {9,10} -> b//10 in {9,10} | (reads) | same |
| L26 up c568 | 0 / 0 | off (on 0) | same | (reads) | same |
| L26 up c813 | 5% / 2% | **res%100** (R2 0.81): res mod 100 in {2, 32, 52, 72, 92} | **res** (R2 0.58): res in {2, 32} | (reads) | same |
| L26 up c875 | 1% / 1% | **res%100** (R2 0.88): res mod 100 in {31} | **res** (R2 0.51): res in {31} | (reads) | same |

</details>

</details>

<details><summary>layer 27: 180 components with main position `=`</summary>

<details><summary>down: 63</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L27 down c0 | 100% / 100% | always | same | - | same |
| L27 down c6 | 61% / 2% | **res//10** (R2 0.87): (tens) res in {90..199} | unexplained (best R2 0.39) | a: mod100 +2%, mod50 +3%; b: mod100 +3%, mod50 +2%; res: mod100 +3% | - |
| L27 down c17 | 8% / 4% | **res%100** (R2 0.91): res mod 100 in {79..85} | unexplained (best R2 0.33) | res: mod25 +3% | - |
| L27 down c23 | 13% / 6% | **res** (R2 0.91): res in {14..17, 110..124} | **res** (R2 0.71): res in {11..17} | - | same |
| L27 down c24 | 7% / 5% | **res%100** (R2 0.72): res mod 100 in {33..38} | unexplained (best R2 0.42) | - | same |
| L27 down c25 | 17% / 0 | **res//10** (R2 0.83): (tens) res in {130, 132..160} | off (on 0) | - | same |
| L27 down c30 | 17% / 0 | **res** (R2 0.78): res in {107..109, 116..119, 125..129, 136..139, 146..149, 156..159, 166..169, 176..179, 186..189, 196..199} | off (on 0) | - | same |
| L27 down c32 | 8% / 1% | **res%100** (R2 0.79): res mod 100: no class above 0.5 (max 0.49) | unexplained (best R2 0.18) | - | same |
| L27 down c33 | 14% / 13% | **res//10** (R2 0.59): (tens) res in {91..99, 186, 188, 191..197} [coarser: res mod 100 in {91..98}, R2 0.84] | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {9,10}; a//10 in {1,2} -> b//10 in {9}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4} | - | same |
| L27 down c38 | 10% / 3% | **res%100** (R2 0.83): res mod 100 in {19, 29, 39, 49, 58..59, 79, 89, 97..99} | **res** (R2 0.53): res in {-99, 19, 29, 39, 79, 98..99} | res: mod4 +4% | - |
| L27 down c42 | 14% / 4% | **res%100** (R2 0.72): res mod 100 in {9, 19, 29, 39, 48..49, 65..70, 79, 89} | unexplained (best R2 0.48) | res: mod4 +3% | - |
| L27 down c51 | 9% / 3% | **res%100** (R2 0.82): res mod 100 in {38..45} | unexplained (best R2 0.48) | - | same |
| L27 down c62 | 11% / 2% | **res%100** (R2 0.67): res mod 100 in {45, 55, 65} | **res%100** (R2 0.68): res mod 100: no class above 0.5 (max 0.44) | - | same |
| L27 down c64 | 10% / 3% | **res%100** (R2 0.88): res mod 100 in {21..25, 51..55} | **res** (R2 0.63): res in {22..25} | - | same |
| L27 down c68 | 26% / 1% | **tens(a,b)** (R2 0.50): a//10 in {0} -> b//10 in {6}; a//10 in {1} -> b//10 in {4,5,6,7}; a//10 in {2} -> b//10 in {3,4,5,6}; a//10 in {3} -> b//10 in {1,2,3,4,5}; a//10 in {4,5} -> b//10 in {1,2,3}; a//10 in {6} -> b//10 in {1,2}; a//10 in {7} -> b//10 in {1} | unexplained (best R2 0.45) | - | same |
| L27 down c69 | 16% / 3% | **res%10** (R2 0.77): res mod 10 in {7..8} | unexplained (best R2 0.49) | - | same |
| L27 down c71 | 0 / 9% | off (on 0) | unexplained (best R2 0.40) | - | same |
| L27 down c73 | 17% / 5% | **res** (R2 0.73): res in {2..15, 22..24, 32..35, 42..44, 52..54, 62..64, 73, 83, 93..94, 103..104, 113, 123, 133, 163} | unexplained (best R2 0.48) | - | same |
| L27 down c80 | 11% / 3% | **res%10** (R2 0.76): res mod 10 in {9} | **res** (R2 0.57): res in {9, 29, 89} | - | same |
| L27 down c82 | 16% / 5% | **res%100** (R2 0.91): res mod 100 in {50..65} | unexplained (best R2 0.36) | - | same |
| L27 down c93 | 12% / 0 | **res//10** (R2 0.88): (tens) res in {150..198} | off (on 0) | - | same |
| L27 down c107 | 11% / 2% | **res%100** (R2 0.84): res mod 100 in {51..59, 62..63} | unexplained (best R2 0.36) | - | same |
| L27 down c110 | 10% / 4% | **res%100** (R2 0.71): res mod 100 in {19, 29, 39, 58..59, 76..79, 89, 99} | unexplained (best R2 0.38) | - | same |
| L27 down c112 | 8% / 1% | **res%100** (R2 0.71): res mod 100 in {24, 44..45, 48, 84..85} | unexplained (best R2 0.30) | res: mod4 +2% | - |
| L27 down c117 | 7% / 3% | **res** (R2 0.84): res in {79..86} | unexplained (best R2 0.46) | - | same |
| L27 down c122 | 6% / 0 | **res%100** (R2 0.70): res mod 100: no class above 0.5 (max 0.48) | off (on 0) | - | same |
| L27 down c125 | 7% / 3% | **res%100** (R2 0.69): res mod 100 in {48..49, 96..99} [coarser: res mod 50 in {47..49}, R2 0.84] | unexplained (best R2 0.34) | - | same |
| L27 down c133 | 17% / 3% | **res//10** (R2 0.77): (tens) res in {39, 41, 45..71} | unexplained (best R2 0.41) | - | same |
| L27 down c137 | 0 / 4% | off (on 0) | **res%100** (R2 0.59): res mod 100: no class above 0.5 (max 0.40) | - | same |
| L27 down c139 | 10% / 2% | **res%100** (R2 0.76): res mod 100 in {1, 21, 31, 41, 51, 61, 71, 81} | **res%100** (R2 0.59): res mod 100: no class above 0.5 (max 0.46) | - | same |
| L27 down c140 | 11% / 1% | **res//10** (R2 0.83): (tens) res in {70..79, 87, 170..178} [coarser: res mod 100 in {70..79, 87}, R2 0.98] | unexplained (best R2 0.34) | - | same |
| L27 down c142 | 5% / 3% | **res** (R2 0.67): res in {9, 19, 29, 39, 47..49, 59, 69, 79, 149} | **res** (R2 0.53): res in {9, 29} | - | same |
| L27 down c147 | 14% / 1% | **res** (R2 0.78): res in {60..67, 155..176} | unexplained (best R2 0.28) | - | same |
| L27 down c157 | 5% / 0 | **res%100** (R2 0.77): res mod 100: no class above 0.5 (max 0.46) | off (on 0) | - | same |
| L27 down c165 | 10% / 4% | **res%10** (R2 0.99): res mod 10 in {6} | **res** (R2 0.68): res in {6, 16, 26, 86, 96} | - | same |
| L27 down c167 | 2% / 9% | **res** (R2 0.69): res in {19..27} | **res** (R2 0.65): res in {17..27} | - | same |
| L27 down c168 | 7% / 4% | **res%100** (R2 0.76): res mod 100 in {40..47} | unexplained (best R2 0.47) | - | same |
| L27 down c188 | 2% / 4% | unexplained (best R2 0.37) | **units(a,b)** (R2 0.87): a%10 in {0,5} -> b%10 in {0,5} | - | same |
| L27 down c193 | 2% / 9% | **res** (R2 0.62): res in {12..20} | **res//10** (R2 0.66): (tens) res in {10..19} | - | same |
| L27 down c200 | 7% / 0 | **res%100** (R2 0.64): res mod 100 in {48..50, 89, 94} | off (on 0) | - | same |
| L27 down c203 | 5% / 1% | **res%100** (R2 0.79): res mod 100 in {27, 47, 67, 87} | unexplained (best R2 0.42) | - | same |
| L27 down c204 | 9% / 4% | **res%50** (R2 0.81): res mod 50 in {11, 21, 31, 41} [coarser: res mod 10 in {1}, R2 0.84] | **res** (R2 0.57): res in {11, 21, 31, 71, 91} | - | same |
| L27 down c222 | 4% / 0 | **res** (R2 0.89): res in {131..135} | off (on 0) | - | same |
| L27 down c224 | 16% / 8% | **res%100** (R2 0.69): res mod 100 in {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95} [coarser: res mod 10 in {0}, R2 0.96] | **units(a,b)** (R2 0.58): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {1} | - | same |
| L27 down c227 | 0 / 15% | off (on 0) | **res%100** (R2 0.56): res mod 100: no class above 0.5 (max 0.48) | - | same |
| L27 down c246 | 6% / 0 | **res** (R2 0.72): res in {73, 113..114, 133..134, 137, 153, 163, 173..174, 183} | off (on 0) | - | same |
| L27 down c254 | 5% / 3% | **res** (R2 0.76): res in {38..43, 139..141} [coarser: res mod 100 in {38..43}, R2 0.85] | unexplained (best R2 0.44) | - | same |
| L27 down c269 | 7% / 4% | **res%100** (R2 0.83): res mod 100 in {1, 21, 41, 51, 61, 71, 81, 91} [coarser: res mod 10 in {1}, R2 0.88] | **res** (R2 0.57): res in {1, 11, 31, 41} | - | same |
| L27 down c278 | 0 / 0 | off (on 0) | same | - | same |
| L27 down c312 | 5% / 0 | **res** (R2 0.77): res in {58, 68, 78, 88, 118, 128, 138, 148, 158, 168, 178} | off (on 0) | - | same |
| L27 down c365 | 3% / 0 | **res** (R2 0.66): res in {97, 107, 147, 157..158} | off (on 0) | - | same |
| L27 down c381 | 2% / 6% | unexplained (best R2 0.10) | unexplained (best R2 0.29) | - | same |
| L27 down c417 | 2% / 0 | **res** (R2 0.84): res in {111..112} | off (on 0) | - | same |
| L27 down c461 | 2% / 0 | **res%100** (R2 0.75): res mod 100 in {56, 58..59} | off (on 0) | - | same |
| L27 down c480 | 4% / 1% | **res%100** (R2 0.81): res mod 100 in {86..89} | unexplained (best R2 0.37) | - | same |
| L27 down c586 | 2% / 3% | **res%100** (R2 0.85): res mod 100 in {20..21} | **res** (R2 0.61): res in {17..18, 20..21} | - | same |
| L27 down c599 | 3% / 0 | **res** (R2 0.69): res in {119..121, 125} | off (on 0) | - | same |
| L27 down c715 | 0 / 1% | off (on 0) | unexplained (best R2 0.08) | - | same |
| L27 down c723 | 4% / 0 | **res** (R2 0.69): res in {83..84, 179..189} | off (on 0) | - | same |
| L27 down c818 | 2% / 0 | **res%100** (R2 0.64): res mod 100 in {61..62} | off (on 0) | - | same |
| L27 down c823 | 1% / 2% | **res%100** (R2 0.64): res mod 100: no class above 0.5 (max 0.36) | **cmp(a,b)** (R2 0.53): cmp(a,b)=-1: 0.01, cmp(a,b)=0: 0.96, cmp(a,b)=1: 0.00 | - | same |
| L27 down c835 | 5% / 1% | **res** (R2 0.88): res in {31..33, 129..134} [coarser: res mod 100 in {29, 31..34}, R2 0.86] | unexplained (best R2 0.40) | - | same |
| L27 down c849 | 5% / 2% | **res** (R2 0.78): res in {61..68} | unexplained (best R2 0.38) | - | same |

</details>

<details><summary>gate: 58</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L27 gate c0 | 100% / 100% | always | same | (reads) | same |
| L27 gate c6 | 62% / 2% | **res//10** (R2 0.86): (tens) res in {86..87, 89..199} | unexplained (best R2 0.41) | (reads) | same |
| L27 gate c17 | 11% / 4% | **res//10** (R2 0.85): (tens) res in {79..89, 178..189} [coarser: res mod 100 in {79..89}, R2 0.99] | unexplained (best R2 0.37) | (reads) | same |
| L27 gate c23 | 15% / 7% | **res%100** (R2 0.89): res mod 100 in {9..19} | **res** (R2 0.80): res in {10..17} | (reads) | same |
| L27 gate c24 | 8% / 5% | **res%100** (R2 0.80): res mod 100 in {32..39} | unexplained (best R2 0.44) | (reads) | same |
| L27 gate c25 | 13% / 0 | **res** (R2 0.88): res in {135..159} | off (on 0) | (reads) | same |
| L27 gate c32 | 10% / 0 | **res%100** (R2 0.76): res mod 100 in {87..89} | off (on 0) | (reads) | same |
| L27 gate c33 | 14% / 12% | **res//10** (R2 0.58): (tens) res in {91..99, 186..188, 191..197} [coarser: res mod 100 in {91..98}, R2 0.84] | **tens(a,b)** (R2 0.59): a//10 in {0} -> b//10 in {9,10}; a//10 in {1,2,4} -> b//10 in {9}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,9,10}; a//10 in {10} -> b//10 in {0,1,2,4} | (reads) | same |
| L27 gate c38 | 10% / 5% | **res%10** (R2 0.96): res mod 10 in {9} | **res** (R2 0.64): res in {-99, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99} | (reads) | same |
| L27 gate c42 | 11% / 1% | **res%100** (R2 0.70): res mod 100 in {29, 49, 65..72} | unexplained (best R2 0.19) | (reads) | same |
| L27 gate c51 | 11% / 4% | **res//10** (R2 0.81): (tens) res in {40..49, 140..151} [coarser: res mod 100 in {40..50}, R2 0.95] | unexplained (best R2 0.41) | (reads) | same |
| L27 gate c62 | 5% / 2% | **res%100** (R2 0.60): res mod 100 in {55} | **res%100** (R2 0.73): res mod 100: no class above 0.5 (max 0.43) | (reads) | same |
| L27 gate c64 | 10% / 6% | **res%100** (R2 0.79): res mod 100 in {20..25, 49..54} | **res** (R2 0.64): res in {19..25} | (reads) | same |
| L27 gate c73 | 6% / 1% | **res%100** (R2 0.79): res mod 100 in {4, 44, 84} | **res%100** (R2 0.53): res mod 100: no class above 0.5 (max 0.40) | (reads) | same |
| L27 gate c80 | 4% / 1% | **res%100** (R2 0.77): res mod 100 in {9, 49, 89, 91} | **res** (R2 0.70): res in {9, 89} | (reads) | same |
| L27 gate c82 | 17% / 3% | **res%100** (R2 0.94): res mod 100 in {50..65} | unexplained (best R2 0.39) | (reads) | same |
| L27 gate c107 | 4% / 0 | **res%100** (R2 0.70): res mod 100 in {55..59} | off (on 0) | (reads) | same |
| L27 gate c110 | 9% / 2% | **res%100** (R2 0.87): res mod 100 in {76..83} | unexplained (best R2 0.23) | (reads) | same |
| L27 gate c112 | 6% / 1% | **res%100** (R2 0.66): res mod 100 in {44..45, 84..85} | unexplained (best R2 0.30) | (reads) | same |
| L27 gate c117 | 4% / 2% | **res%100** (R2 0.91): res mod 100 in {80..83} | unexplained (best R2 0.39) | (reads) | same |
| L27 gate c122 | 4% / 0 | **res** (R2 0.63): res in {140..145} | off (on 0) | (reads) | same |
| L27 gate c125 | 7% / 1% | **res%100** (R2 0.78): res mod 100 in {48..52, 98..99} | unexplained (best R2 0.26) | (reads) | same |
| L27 gate c139 | 2% / 1% | **res%100** (R2 0.76): res mod 100 in {1, 61} | **res%100** (R2 0.80): res mod 100: no class above 0.5 (max 0.46) | (reads) | same |
| L27 gate c140 | 4% / 0 | **res** (R2 0.69): res in {71, 77, 169, 171..177} | off (on 0) | (reads) | same |
| L27 gate c142 | 1% / 0 | **res%100** (R2 0.76): res mod 100 in {49} | off (on 0) | (reads) | same |
| L27 gate c147 | 26% / 3% | **res//10** (R2 0.82): (tens) res in {55..78, 150..178} [coarser: res mod 100 in {52..78}, R2 0.98] | unexplained (best R2 0.39) | (reads) | same |
| L27 gate c149 | 5% / 0 | **res//10** (R2 0.78): (tens) res in {168..199} | off (on 0) | (reads) | same |
| L27 gate c157 | 2% / 1% | **res%100** (R2 0.76): res mod 100 in {1, 98..99} | **res%100** (R2 0.73): res mod 100 in {1, 99} | (reads) | same |
| L27 gate c165 | 10% / 3% | **res%10** (R2 0.98): res mod 10 in {6} | **res** (R2 0.67): res in {6, 16, 26, 96} | (reads) | same |
| L27 gate c168 | 8% / 7% | **res%100** (R2 0.75): res mod 100 in {35..37, 39..44} | unexplained (best R2 0.50) | (reads) | same |
| L27 gate c195 | 2% / 0 | **res%100** (R2 0.88): res mod 100 in {68..69} | off (on 0) | (reads) | same |
| L27 gate c200 | 4% / 1% | **res%100** (R2 0.68): res mod 100 in {48..51} | unexplained (best R2 0.23) | (reads) | same |
| L27 gate c203 | 3% / 0 | **res%100** (R2 0.68): res mod 100 in {59, 79, 98..99} | off (on 0) | (reads) | same |
| L27 gate c204 | 7% / 2% | **res%100** (R2 0.76): res mod 100 in {1, 21, 31, 41, 61, 71, 81, 91} [coarser: res mod 10 in {1}, R2 0.84] | **res%100** (R2 0.53): res mod 100: no class above 0.5 (max 0.45) | (reads) | same |
| L27 gate c222 | 14% / 4% | **res%100** (R2 0.82): res mod 100 in {34..46} | unexplained (best R2 0.46) | (reads) | same |
| L27 gate c224 | 13% / 5% | **res%100** (R2 0.52): res mod 100 in {0, 15, 20, 25, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 95} [coarser: res mod 10 in {0, 5}, R2 0.80] | **units(a,b)** (R2 0.74): a%10 in {0,5} -> b%10 in {0,5} | (reads) | same |
| L27 gate c246 | 3% / 0 | **res** (R2 0.77): res in {73, 113, 133, 173..174} | off (on 0) | (reads) | same |
| L27 gate c254 | 0 / 0 | off (on 0) | same | (reads) | same |
| L27 gate c269 | 10% / 5% | **res%10** (R2 0.95): res mod 10 in {1} | **res** (R2 0.64): res in {1, 11, 21, 31, 41, 71, 81, 91} | (reads) | same |
| L27 gate c270 | 7% / 0 | **res** (R2 0.79): res in {156..177} | off (on 0) | (reads) | same |
| L27 gate c278 | 1% / 0 | **res//10** (R2 0.58): (tens) res in {180..183, 185..188} | off (on 0) | (reads) | same |
| L27 gate c320 | 12% / 9% | **units(a,b)** (R2 0.91): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {9}; a%10 in {2} -> b%10 in {8}; a%10 in {3} -> b%10 in {7}; a%10 in {4} -> b%10 in {6}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {3}; a%10 in {8} -> b%10 in {2}; a%10 in {9} -> b%10 in {1} | **units(a,b)** (R2 0.61): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {1}; a%10 in {4} -> b%10 in {4}; a%10 in {9} -> b%10 in {9} | (reads) | same |
| L27 gate c339 | 20% / 3% | **res%10** (R2 0.96): res mod 10 in {8..9} | unexplained (best R2 0.41) | (reads) | same |
| L27 gate c365 | 8% / 2% | **res%20** (R2 0.83): res mod 20 in {7, 17} [coarser: res mod 10 in {7}, R2 0.87] | unexplained (best R2 0.41) | (reads) | same |
| L27 gate c421 | 24% / 1% | **res//10** (R2 0.79): (tens) res in {96..119, 199} | unexplained (best R2 0.32) | (reads) | same |
| L27 gate c461 | 1% / 0 | **res%100** (R2 0.81): res mod 100 in {56} | off (on 0) | (reads) | same |
| L27 gate c480 | 1% / 0 | **res** (R2 0.74): res in {89, 186..189} | off (on 0) | (reads) | same |
| L27 gate c533 | 7% / 0 | **res%100** (R2 0.80): res mod 100 in {13, 33, 53, 63, 73, 93} [coarser: res mod 20 in {13}, R2 0.83] | off (on 0) | (reads) | same |
| L27 gate c586 | 7% / 7% | **res%100** (R2 0.87): res mod 100 in {13..20} | **res** (R2 0.75): res in {13..21} | (reads) | same |
| L27 gate c650 | 5% / 0 | **res** (R2 0.62): res in {137, 140..149} | off (on 0) | (reads) | same |
| L27 gate c657 | 13% / 1% | **res%100** (R2 0.85): res mod 100 in {27, 72..83, 87} | unexplained (best R2 0.29) | (reads) | same |
| L27 gate c723 | 7% / 1% | **res%100** (R2 0.87): res mod 100 in {80..85, 89} | unexplained (best R2 0.36) | (reads) | same |
| L27 gate c745 | 9% / 4% | **res%100** (R2 0.85): res mod 100 in {1, 41..43, 62, 81..83} | **res** (R2 0.57): res in {1..2, 42, 81..83} | (reads) | same |
| L27 gate c748 | 0 / 0 | off (on 0) | same | (reads) | same |
| L27 gate c823 | 2% / 1% | **res%100** (R2 0.76): res mod 100 in {97..99} | **cmp(a,b)** (R2 0.57): cmp(a,b)=-1: 0.01, cmp(a,b)=0: 0.89, cmp(a,b)=1: 0.00 | (reads) | same |
| L27 gate c835 | 9% / 3% | **res%100** (R2 0.87): res mod 100 in {29..37} | unexplained (best R2 0.40) | (reads) | same |
| L27 gate c839 | 1% / 0 | **res%100** (R2 0.87): res mod 100 in {69} | off (on 0) | (reads) | same |
| L27 gate c849 | 1% / 0 | **res//10** (R2 0.47): (tens) res in {182..183, 186..188, 198} | off (on 0) | (reads) | same |

</details>

<details><summary>up: 58</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L27 up c17 | 9% / 2% | **res** (R2 0.79): res in {79..83, 101, 160..163, 170..173, 179..184} | unexplained (best R2 0.28) | (reads) | same |
| L27 up c23 | 21% / 14% | **res//10** (R2 0.80): (tens) res in {10..22, 24, 110..134} [coarser: res mod 100 in {10..26, 28, 30..31}, R2 0.90] | **res//10** (R2 0.67): (tens) res in {10..25} | (reads) | same |
| L27 up c24 | 8% / 3% | **res%100** (R2 0.74): res mod 100 in {38..43, 50, 60, 80} | unexplained (best R2 0.41) | (reads) | same |
| L27 up c25 | 28% / 0 | **res//10** (R2 0.86): (tens) res in {126, 128..200} | off (on 0) | (reads) | same |
| L27 up c30 | 21% / 4% | **res** (R2 0.84): res in {31, 41..43, 61..63, 71, 80..83, 101..103, 121..123, 131..133, 141..143, 151..153, 161..164, 171..173, 180..183} [coarser: res mod 100 in {1, 21..23, 31..33, 41..43, 61..63, 71..73, 80..83}, R2 0.83] | unexplained (best R2 0.35) | (reads) | same |
| L27 up c32 | 15% / 9% | **res%100** (R2 0.86): res mod 100 in {1..11, 99} | **res%100** (R2 0.72): res mod 100 in {3, 5..7, 10} | (reads) | same |
| L27 up c38 | 7% / 4% | **res%100** (R2 0.61): res mod 100 in {0..1, 59, 95..99} | **res%100** (R2 0.53): res mod 100 in {0..1, 98..99} | (reads) | same |
| L27 up c42 | 19% / 5% | **res%10** (R2 0.90): res mod 10 in {8..9} | **res** (R2 0.58): res in {-99, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99} | (reads) | same |
| L27 up c51 | 13% / 5% | **res%100** (R2 0.83): res mod 100 in {37..48} | unexplained (best R2 0.48) | (reads) | same |
| L27 up c62 | 16% / 4% | **res%50** (R2 0.69): res mod 50 in {5..7, 35..37, 45..46} | **res** (R2 0.56): res in {5..6, 35..36, 85..86, 95..97} | (reads) | same |
| L27 up c64 | 9% / 4% | **res%100** (R2 0.77): res mod 100 in {22..25, 50, 52..53} | **res** (R2 0.67): res in {20, 22..25} | (reads) | same |
| L27 up c69 | 20% / 2% | **res%100** (R2 0.72): res mod 100 in {8, 18, 27..28, 38, 47..49, 57..58, 67..69, 78, 87..88} | unexplained (best R2 0.37) | (reads) | same |
| L27 up c73 | 12% / 2% | **res** (R2 0.72): res in {2..35, 38..47, 53, 143} | unexplained (best R2 0.23) | (reads) | same |
| L27 up c80 | 11% / 3% | **res%100** (R2 0.71): res mod 100 in {9, 35..37, 49..51, 69, 89..92} | unexplained (best R2 0.49) | (reads) | same |
| L27 up c82 | 13% / 2% | **res%100** (R2 0.86): res mod 100 in {49..61} | unexplained (best R2 0.33) | (reads) | same |
| L27 up c85 | 24% / 5% | **res//10** (R2 0.79): (tens) res in {70..95, 171..185} [coarser: res mod 100 in {70..91, 94}, R2 0.88] | **res//10** (R2 0.66): (tens) res in {70..91, 93..94} | (reads) | same |
| L27 up c93 | 12% / 0 | **res//10** (R2 0.88): (tens) res in {152..199} | off (on 0) | (reads) | same |
| L27 up c107 | 12% / 3% | **res//10** (R2 0.81): (tens) res in {60..70, 160..170} [coarser: res mod 100 in {60..70}, R2 0.99] | unexplained (best R2 0.34) | (reads) | same |
| L27 up c110 | 9% / 1% | **res%100** (R2 0.83): res mod 100 in {71..80} | unexplained (best R2 0.25) | (reads) | same |
| L27 up c112 | 9% / 1% | **res** (R2 0.78): res in {24, 44, 48, 84, 113, 115..120, 144, 184} | unexplained (best R2 0.27) | (reads) | same |
| L27 up c117 | 6% / 2% | **res%100** (R2 0.95): res mod 100 in {79..84} | unexplained (best R2 0.25) | (reads) | same |
| L27 up c125 | 6% / 1% | **res%100** (R2 0.73): res mod 100 in {48..49, 69, 89, 98..99} [coarser: res mod 50 in {48..49}, R2 0.85] | unexplained (best R2 0.24) | (reads) | same |
| L27 up c133 | 2% / 0 | **res//10** (R2 0.81): (tens) res in {180..200} | off (on 0) | (reads) | same |
| L27 up c139 | 9% / 4% | **res%10** (R2 0.90): res mod 10 in {1} | **res** (R2 0.64): res in {1, 11, 21, 31} | (reads) | same |
| L27 up c140 | 1% / 0 | **res** (R2 0.73): res in {178..179} | off (on 0) | (reads) | same |
| L27 up c142 | 3% / 0 | **res%100** (R2 0.89): res mod 100 in {49, 69} | off (on 0) | (reads) | same |
| L27 up c147 | 6% / 0 | **res%100** (R2 0.63): res mod 100 in {63..69} | off (on 0) | (reads) | same |
| L27 up c149 | 6% / 0 | **res** (R2 0.76): res in {44, 84, 140..148, 184} | off (on 0) | (reads) | same |
| L27 up c157 | 4% / 0 | **res%100** (R2 0.65): res mod 100 in {90, 92} | off (on 0) | (reads) | same |
| L27 up c165 | 9% / 3% | **res%10** (R2 0.92): res mod 10 in {6} | **res** (R2 0.67): res in {6, 16, 26, 96} | (reads) | same |
| L27 up c167 | 7% / 0 | **res** (R2 0.75): res in {136..150} | off (on 0) | (reads) | same |
| L27 up c168 | 1% / 0 | **res** (R2 0.66): res in {44, 69, 169} [coarser: res mod 100 in {69}, R2 0.83] | off (on 0) | (reads) | same |
| L27 up c193 | 4% / 0 | **res%100** (R2 0.88): res mod 100 in {80..82} | off (on 0) | (reads) | same |
| L27 up c197 | 5% / 0 | **res** (R2 0.91): res in {114..119} | off (on 0) | (reads) | same |
| L27 up c200 | 2% / 0 | **res%50** (R2 0.78): res mod 50 in {49} | off (on 0) | (reads) | same |
| L27 up c244 | 99% / 99% | always | same | (reads) | same |
| L27 up c246 | 9% / 3% | **res%100** (R2 0.86): res mod 100 in {33..39, 73..74} | **res** (R2 0.53): res in {34..37} | (reads) | same |
| L27 up c251 | 0 / 0 | off (on 0) | same | (reads) | same |
| L27 up c254 | 6% / 3% | **res%100** (R2 0.90): res mod 100 in {38..43} | unexplained (best R2 0.45) | (reads) | same |
| L27 up c269 | 3% / 1% | **res%100** (R2 0.67): res mod 100 in {1, 61, 71} | **res%100** (R2 0.76): res mod 100: no class above 0.5 (max 0.47) | (reads) | same |
| L27 up c278 | 0 / 0 | off (on 0) | same | (reads) | same |
| L27 up c292 | 1% / 0 | **res//10** (R2 0.64): (tens) res in {180..188} | off (on 0) | (reads) | same |
| L27 up c312 | 10% / 3% | **res%10** (R2 0.95): res mod 10 in {9} | **res** (R2 0.59): res in {9, 19, 29, 79, 89} | (reads) | same |
| L27 up c320 | 1% / 0 | **res%100** (R2 0.73): res mod 100 in {84} | off (on 0) | (reads) | same |
| L27 up c336 | 2% / 1% | **res%100** (R2 0.75): res mod 100 in {35..36} | unexplained (best R2 0.32) | (reads) | same |
| L27 up c360 | 3% / 2% | **res%100** (R2 0.92): res mod 100 in {35..37} | **res** (R2 0.53): res in {35..37} | (reads) | same |
| L27 up c365 | 9% / 1% | **res%100** (R2 0.67): res mod 100 in {17, 36..37, 47, 56..57, 77, 87} | unexplained (best R2 0.33) | (reads) | same |
| L27 up c457 | 1% / 0 | unexplained (best R2 0.45) | off (on 0) | (reads) | same |
| L27 up c461 | 3% / 3% | **res%100** (R2 0.80): res mod 100 in {35..36, 56} | unexplained (best R2 0.41) | (reads) | same |
| L27 up c591 | 4% / 0 | **res** (R2 0.79): res in {140..145, 182} | off (on 0) | (reads) | same |
| L27 up c657 | 6% / 0 | **res%100** (R2 0.79): res mod 100: no class above 0.5 (max 0.47) | off (on 0) | (reads) | same |
| L27 up c723 | 8% / 0 | **res** (R2 0.83): res in {44, 84, 144, 164..198} | off (on 0) | (reads) | same |
| L27 up c745 | 1% / 0 | **res%100** (R2 0.77): res mod 100 in {35} | off (on 0) | (reads) | same |
| L27 up c791 | 5% / 10% | **res%100** (R2 0.78): res mod 100 in {4..10} | **res%100** (R2 0.66): res mod 100: no class above 0.5 (max 0.46) | (reads) | same |
| L27 up c823 | 10% / 1% | **res** (R2 0.80): res in {73, 108..117, 133, 173} | unexplained (best R2 0.36) | (reads) | same |
| L27 up c835 | 5% / 23% | **res%100** (R2 0.86): res mod 100 in {80..84} | unexplained (best R2 0.49) | (reads) | same |
| L27 up c843 | 1% / 0 | **res%100** (R2 0.75): res mod 100 in {44, 84} | off (on 0) | (reads) | same |
| L27 up c849 | 0 / 0 | off (on 0) | same | (reads) | same |

</details>

<details><summary>v: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L27 v c239 (kv3) | 2% / 60% | unexplained (best R2 0.08) | **tens(a,b)** (R2 0.51): a//10 in {0} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,6,7,8,9}; a//10 in {6,10} -> b//10 in {1,2,8}; a//10 in {7} -> b//10 in {0,1,2,8,9}; a//10 in {8} -> b//10 in {1,2}; a//10 in {9} -> b//10 in {1,2,3,4,5} | (reads) | same |

</details>

</details>

