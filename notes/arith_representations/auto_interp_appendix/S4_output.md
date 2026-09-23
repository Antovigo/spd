# §4 — every L20-L31 writer at `=`, by on-set family

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

<details><summary>res%100: 278 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L20 down c29 | 56% / 53% | **res%100** (R2 0.82): res mod 100 in {11..29, 55..86} | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {2,6,7,8}; a//10 in {1} -> b//10 in {7,8,9,10}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {1,2,6,7,8}; a//10 in {4,9} -> b//10 in {1,2,3,6,7,8}; a//10 in {5} -> b//10 in {2,3,4,7,8,9}; a//10 in {6} -> b//10 in {0,4,8,9,10}; a//10 in {7} -> b//10 in {0,1,5,6,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,5,6,7,8,10}; a//10 in {10} -> b//10 in {2,3,4,7,8} | a: mod50 +2%; res: mod50 +10% | b: mod50 +3%; res: mod50 +9% | 46, 48, 49, 50, 51 | +0.63 / +0.69 |
| L20 down c100 | 71% / 20% | **res%100** (R2 0.54): res mod 100 in {1..5, 8..9, 13..15, 18..25, 29, 33..35, 38..45, 48..49, 52..65, 68..69, 73..75, 78..85, 88..89, 93..95, 97..99} | **res** (R2 0.50): res in {3..5, 9, 13..15, 19, 23..25, 33..34, 43..45, 54..55, 59, 63..65, 74..75, 79, 83..85, 89, 93..95, 98..99} | res: mod10 +3%, mod5 +6% | res: mod10 +2%, mod5 +3% | 3, 4, 74, 84, 103 | +0.66 / +0.75 |
| L21 down c35 | 14% / 3% | **res%100** (R2 0.85): res mod 100 in {57..70} | unexplained (best R2 0.33) | res: mod25 +10% | - | 42, 52, 55, 82, 83 | +0.48 / +0.47 |
| L21 down c43 | 28% / 24% | **res%100** (R2 0.88): res mod 100 in {0..5, 79..99} | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {9,10}; a//10 in {5} -> b//10 in {7}; a//10 in {6} -> b//10 in {6,7,8}; a//10 in {7} -> b//10 in {7,8,9,10}; a//10 in {8} -> b//10 in {0,8,9,10}; a//10 in {9} -> b//10 in {0,1,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,9,10} | res: mod100 +2% | - | 82, 84, 85, 89, 198 | +0.59 / +0.74 |
| L21 down c63 | 8% / 12% | **res%100** (R2 0.70): res mod 100 in {23..30} | unexplained (best R2 0.49) | res: mod25 +2% | - | 7, 8, 9, 13, 16 | +0.35 / +0.58 |
| L21 down c96 | 12% / 10% | **res%100** (R2 0.56): res mod 100 in {18..20, 58, 76..80, 98} | unexplained (best R2 0.29) | - | same | 32, 92, 93, 129, 132 | +0.42 / +0.17 |
| L21 down c916 | 8% / 2% | **res%100** (R2 0.70): res mod 100 in {47..48, 67..68, 87..88} | **res** (R2 0.53): res in {7..8} | - | same | 133, 135, 144, 148, 160 | +0.15 / +0.07 |
| L22 down c5 | 16% / 13% | **res%100** (R2 0.76): res mod 100 in {39..51} | unexplained (best R2 0.45) | res: mod100 +5%, mod50 +4%, mod25 +15% | - | 29, 122, 125, 127, 182 | +0.73 / +0.81 |
| L22 down c6 | 16% / 14% | **res%100** (R2 0.88): res mod 100 in {0..1, 3..15} | **res%100** (R2 0.74): res mod 100 in {1, 3..14} | res: mod100 +4%, mod50 +2%, mod25 +11% | res: mod100 +4%, mod50 +5%, mod25 +11%, mod20 +3% | 25, 27, 46, 48, 199 | +0.72 / +0.14 |
| L22 down c27 | 12% / 10% | **res%100** (R2 0.81): res mod 100 in {27..38} | **res** (R2 0.57): res in {27..38} | res: mod25 +3% | - | 29, 30, 31, 32, 33 | +0.54 / +0.73 |
| L22 down c28 | 14% / 4% | **res%100** (R2 0.87): res mod 100 in {77..92} | **res** (R2 0.55): res in {77..90} | res: mod25 +6% | - | 81, 83, 84, 181, 184 | +0.74 / +0.82 |
| L22 down c30 | 7% / 2% | **res%100** (R2 0.71): res mod 100 in {31..36} | unexplained (best R2 0.28) | - | same | 39, 106, 107, 134, 135 | +0.33 / +0.23 |
| L22 down c33 | 22% / 10% | **res%100** (R2 0.82): res mod 100 in {1, 86..99} | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {9,10}; a//10 in {7} -> b//10 in {8}; a//10 in {8} -> b//10 in {9}; a//10 in {9} -> b//10 in {0,1,3,4,5,9,10}; a//10 in {10} -> b//10 in {0,1} | res: mod100 +2%, mod25 +4% | - | 31, 32, 34, 37, 46 | +0.44 / +0.68 |
| L22 down c35 | 15% / 4% | **res%100** (R2 0.85): res mod 100 in {47..61} | unexplained (best R2 0.39) | res: mod25 +4% | - | 37, 71, 137, 187, 190 | +0.67 / +0.47 |
| L22 down c44 | 18% / 0 | **res%100** (R2 0.67): res mod 100 in {52, 60..76} | off (on 0) | - | same | 1, 2, 32, 42, 100 | +0.25 / +0.17 |
| L22 down c48 | 11% / 3% | **res%100** (R2 0.69): res mod 100 in {8, 18, 27..30, 38, 48, 68, 88} | **res** (R2 0.54): res in {8, 27..29} | res: mod4 +3% | - | 46, 50, 98, 185, 198 | +0.51 / +0.60 |
| L22 down c98 | 8% / 1% | **res%100** (R2 0.70): res mod 100 in {54..60} | unexplained (best R2 0.21) | res: mod25 +2% | - | 47, 52, 63, 148, 185 | +0.50 / +0.60 |
| L22 down c101 | 8% / 4% | **res%100** (R2 0.74): res mod 100 in {20..21, 23..27} | **res** (R2 0.53): res in {21, 23..26} | - | same | 2, 25, 124, 125, 152 | +0.33 / +0.26 |
| L22 down c123 | 7% / 0 | **res%100** (R2 0.78): res mod 100 in {0..1, 96..99} | off (on 0) | - | same | 1, 120, 138, 139, 143 | +0.45 / +0.14 |
| L22 down c257 | 2% / 1% | **res%100** (R2 0.63): res mod 100 in {45..46} | unexplained (best R2 0.27) | - | same | 38, 149, 162, 185, 186 | +0.49 / +0.62 |
| L23 down c3 | 15% / 17% | **res%100** (R2 0.88): res mod 100 in {0..1, 90..99} | unexplained (best R2 0.45) | res: mod100 +5%, mod50 +4%, mod25 +12% | - | 95, 96, 97, 98, 99 | +0.76 / +0.87 |
| L23 down c5 | 13% / 21% | **res%100** (R2 0.79): res mod 100 in {32..43} | unexplained (best R2 0.46) | res: mod100 +2%, mod25 +6% | res: mod50 +2%, mod25 +3% | 61, 77, 117, 176, 179 | +0.76 / +0.88 |
| L23 down c15 | 12% / 10% | **res%100** (R2 0.78): res mod 100 in {30..35, 38..42} | unexplained (best R2 0.38) | res: mod25 +6% | - | 32, 33, 34, 133, 134 | +0.65 / +0.72 |
| L23 down c16 | 18% / 19% | **res%100** (R2 0.91): res mod 100 in {70..86} | **tens(a,b)** (R2 0.55): a//10 in {0,6} -> b//10 in {7,8}; a//10 in {4} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {7}; a//10 in {7} -> b//10 in {0,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,10}; a//10 in {9,10} -> b//10 in {1,2} | res: mod100 +3%, mod50 +3%, mod25 +5% | - | 61, 63, 66, 67, 196 | +0.74 / +0.72 |
| L23 down c17 | 15% / 11% | **res%100** (R2 0.86): res mod 100 in {64..76} | unexplained (best R2 0.40) | res: mod100 +2%, mod50 +3%, mod25 +4% | - | 119, 125, 188, 189, 198 | +0.77 / +0.85 |
| L23 down c77 | 6% / 4% | **res%100** (R2 0.69): res mod 100 in {1, 21, 41, 61, 81} | unexplained (best R2 0.43) | res: mod4 +4% | - | 21, 121, 161, 168, 181 | +0.42 / +0.47 |
| L23 down c141 | 2% / 1% | **res%100** (R2 0.61): res mod 100 in {28, 88} | unexplained (best R2 0.40) | - | same | 81, 133, 183, 185, 188 | +0.21 / +0.30 |
| L23 down c202 | 9% / 9% | **res%100** (R2 0.86): res mod 100 in {13..20} | **res** (R2 0.63): res in {13..20} | - | same | 37, 42, 57, 75, 77 | +0.52 / +0.84 |
| L23 down c216 | 13% / 1% | **res%100** (R2 0.70): res mod 100 in {60..70, 84..86, 88} | unexplained (best R2 0.25) | res: mod25 +2% | - | 108, 111, 125, 127, 188 | +0.31 / +0.14 |
| L24 down c11 | 17% / 9% | **res%100** (R2 0.86): res mod 100 in {21..28, 61..69} | **res** (R2 0.50): res in {22..27, 62..66} | res: mod25 +3%, mod20 +3% | - | 63, 64, 65, 164, 165 | +0.77 / +0.68 |
| L24 down c13 | 14% / 10% | **res%100** (R2 0.89): res mod 100 in {52..64} | unexplained (best R2 0.40) | res: mod100 +3%, mod50 +2%, mod25 +4% | - | 57, 58, 59, 60, 61 | +0.75 / +0.85 |
| L24 down c16 | 18% / 12% | **res%100** (R2 0.76): res mod 100 in {16..17, 21..22, 24..28, 70..72, 74..77, 81..82} [coarser: res mod 50 in {16, 20..22, 24..27}, R2 0.84] | **res** (R2 0.50): res in {15..17, 20..22, 24..27, 71..72} | res: mod25 +5% | res: mod25 +3% | 24, 25, 74, 124, 125 | +0.72 / +0.69 |
| L24 down c17 | 12% / 9% | **res%100** (R2 0.86): res mod 100 in {46..56} | unexplained (best R2 0.35) | res: mod25 +3% | - | 71, 93, 103, 170, 193 | +0.73 / +0.83 |
| L24 down c22 | 14% / 7% | **res%100** (R2 0.71): res mod 100 in {13..17, 63..68, 70..71, 74} [coarser: res mod 50 in {13..17, 24}, R2 0.83] | **res** (R2 0.55): res in {13..17, 24} | res: mod25 +2% | - | 4, 66, 116, 166, 167 | +0.64 / +0.53 |
| L24 down c36 | 9% / 3% | **res%100** (R2 0.94): res mod 100 in {70..78} | unexplained (best R2 0.26) | res: mod25 +3% | - | 82, 164, 167, 181, 182 | +0.70 / +0.59 |
| L24 down c37 | 9% / 2% | **res%100** (R2 0.69): res mod 100 in {47, 77, 85..89} | unexplained (best R2 0.30) | - | same | 86, 87, 175, 186, 187 | +0.51 / +0.60 |
| L24 down c41 | 15% / 4% | **res%100** (R2 0.90): res mod 100 in {82..96} | unexplained (best R2 0.43) | res: mod25 +3% | - | 23, 25, 27, 126, 162 | +0.71 / +0.67 |
| L24 down c43 | 22% / 21% | **res%100** (R2 0.79): res mod 100 in {15..19, 36..37, 55..59, 75..79, 95..99} [coarser: res mod 20 in {15..19}, R2 0.87] | unexplained (best R2 0.42) | res: mod20 +3% | a: mod20 +2%; b: mod20 +2%; res: mod20 +5% | 28, 45, 47, 146, 147 | +0.61 / +0.71 |
| L24 down c50 | 6% / 3% | **res%100** (R2 0.75): res mod 100 in {25..27, 76, 86} | unexplained (best R2 0.42) | res: mod4 +3% | - | 6, 23, 24, 96, 129 | +0.56 / +0.47 |
| L24 down c52 | 9% / 3% | **res%100** (R2 0.75): res mod 100 in {14, 24, 34..35, 44, 54, 74, 84, 94} [coarser: res mod 50 in {14, 24, 34, 44}, R2 0.81] | unexplained (best R2 0.45) | res: mod4 +5% | - | 34, 116, 134, 169, 170 | +0.40 / +0.45 |
| L24 down c82 | 8% / 5% | **res%100** (R2 0.81): res mod 100 in {11, 21, 31..32, 41..42, 81..82} | unexplained (best R2 0.50) | - | same | 11, 12, 29, 61, 111 | +0.38 / +0.36 |
| L24 down c87 | 8% / 3% | **res%100** (R2 0.83): res mod 100 in {80..86} | unexplained (best R2 0.49) | - | same | 125, 187, 193, 194, 195 | +0.64 / +0.83 |
| L24 down c91 | 8% / 4% | **res%100** (R2 0.73): res mod 100 in {16, 36..37, 56, 95..96} | **res** (R2 0.54): res in {6, 16, 36, 95..97} | res: mod4 +3% | - | 6, 133, 134, 177, 190 | +0.54 / +0.56 |
| L24 down c99 | 11% / 6% | **res%100** (R2 0.83): res mod 100 in {26..36} | **res** (R2 0.63): res in {26..34} | - | same | 24, 34, 49, 74, 144 | +0.18 / +0.28 |
| L24 down c101 | 10% / 11% | **res%100** (R2 0.91): res mod 100 in {15..24} | **res** (R2 0.63): res in {14..24} | - | same | 16, 17, 18, 19, 21 | +0.40 / +0.62 |
| L24 down c152 | 4% / 0 | **res%100** (R2 0.72): res mod 100 in {75..78} | off (on 0) | - | same | 72, 173, 177, 179, 180 | +0.32 / +0.59 |
| L24 down c191 | 8% / 3% | **res%100** (R2 0.91): res mod 100 in {76..83} | unexplained (best R2 0.33) | - | same | 68, 86, 109, 113, 198 | +0.45 / +0.26 |
| L24 down c212 | 4% / 2% | **res%100** (R2 0.61): res mod 100 in {20, 40, 80} | **res** (R2 0.55): res in {20} | - | same | 48, 50, 58, 158, 177 | +0.40 / +0.51 |
| L24 down c263 | 5% / 2% | **res%100** (R2 0.70): res mod 100: no class above 0.5 (max 0.44) | unexplained (best R2 0.35) | - | same | 22, 104, 107, 124, 152 | +0.19 / +0.02 |
| L24 down c377 | 10% / 0 | **res%100** (R2 0.84): res mod 100 in {0..1, 93..99} | off (on 0) | - | same | 87, 91, 97, 110, 167 | +0.14 / -0.03 |
| L25 down c10 | 13% / 9% | **res%100** (R2 0.81): res mod 100 in {86..96, 98} | unexplained (best R2 0.47) | res: mod100 +3%, mod50 +2%, mod25 +6% | - | 88, 89, 122, 188, 198 | +0.69 / +0.68 |
| L25 down c15 | 9% / 8% | **res%100** (R2 0.87): res mod 100 in {0..7} | **res%100** (R2 0.68): res mod 100 in {1..7} | res: mod100 +2%, mod25 +4% | res: mod25 +5% | 111, 113, 124, 198, 199 | +0.68 / +0.47 |
| L25 down c30 | 19% / 5% | **res%100** (R2 0.78): res mod 100 in {52..57, 59..67, 69..73} | unexplained (best R2 0.30) | - | same | 160, 161, 162, 170, 172 | +0.59 / +0.48 |
| L25 down c31 | 16% / 7% | **res%100** (R2 0.86): res mod 100 in {1, 13..14} | **res%100** (R2 0.62): res mod 100 in {6..7} | res: mod50 +2% | - | 105, 106, 107, 108, 109 | +0.60 / -0.22 |
| L25 down c40 | 9% / 3% | **res%100** (R2 0.80): res mod 100 in {0..1, 95..99} | **res%100** (R2 0.55): res mod 100 in {0..1, 99} | - | same | 35, 100, 101, 185, 200 | +0.58 / +0.40 |
| L25 down c42 | 9% / 8% | **res%100** (R2 0.86): res mod 100 in {7..8, 10..14} | **res%100** (R2 0.73): res mod 100 in {10..12} | - | res: mod25 +4% | 11, 12, 111, 112, 113 | +0.61 / +0.64 |
| L25 down c46 | 5% / 4% | **res%100** (R2 0.70): res mod 100 in {17..20, 79} | **res** (R2 0.57): res in {17..20} | - | same | 18, 19, 119, 131, 181 | +0.40 / +0.55 |
| L25 down c48 | 16% / 12% | **res%100** (R2 0.86): res mod 100 in {0..1, 8..11, 13..14, 88..91, 93..94, 98..99} | **res%100** (R2 0.49): res mod 100 in {1, 8..11, 13..14, 89..91, 98..99} | - | same | 8, 9, 10, 90, 109 | +0.60 / +0.73 |
| L25 down c50 | 11% / 4% | **res%100** (R2 0.79): res mod 100 in {13, 22..25, 43, 53, 63, 73, 83} | **res** (R2 0.60): res in {3, 13, 23..25} | res: mod4 +6% | - | 12, 23, 63, 123, 163 | +0.36 / +0.16 |
| L25 down c63 | 10% / 2% | **res%100** (R2 0.72): res mod 100 in {30, 40, 50..51, 70, 80, 90..91} | unexplained (best R2 0.31) | - | same | 90, 115, 140, 150, 190 | +0.43 / +0.34 |
| L25 down c66 | 13% / 5% | **res%100** (R2 0.89): res mod 100 in {41..47, 81..86} | unexplained (best R2 0.45) | - | same | 43, 45, 83, 143, 183 | +0.73 / +0.85 |
| L25 down c76 | 15% / 3% | **res%100** (R2 0.74): res mod 100 in {17..18, 36..39, 57..59, 76..79, 97..98} [coarser: res mod 20 in {17..19}, R2 0.84] | unexplained (best R2 0.38) | res: mod4 +3% | - | 117, 118, 131, 157, 191 | +0.34 / +0.24 |
| L25 down c98 | 3% / 1% | **res%100** (R2 0.73): res mod 100 in {24, 64, 84} | **res** (R2 0.53): res in {24} | - | same | 4, 63, 136, 168, 175 | +0.54 / +0.54 |
| L25 down c110 | 3% / 3% | **res%100** (R2 0.75): res mod 100 in {16..18} | **res** (R2 0.70): res in {15..18} | - | same | 17, 83, 116, 117, 167 | +0.31 / +0.25 |
| L25 down c114 | 7% / 2% | **res%100** (R2 0.84): res mod 100 in {86, 94..98} | unexplained (best R2 0.22) | - | same | 86, 94, 96, 186, 196 | +0.64 / +0.76 |
| L25 down c130 | 1% / 0 | **res%100** (R2 0.59): res mod 100 in {56} | off (on 0) | - | same | 42, 123, 194, 195, 197 | +0.28 / +0.19 |
| L25 down c139 | 12% / 1% | **res%100** (R2 0.82): res mod 100 in {0..1, 93..99} | unexplained (best R2 0.35) | - | same | 115, 118, 122, 129, 133 | +0.56 / +0.38 |
| L25 down c151 | 8% / 3% | **res%100** (R2 0.74): res mod 100 in {25..30, 68} | **res** (R2 0.54): res in {25..30} | - | same | 100, 102, 128, 186, 194 | +0.15 / +0.19 |
| L25 down c207 | 6% / 2% | **res%100** (R2 0.78): res mod 100 in {12, 14, 52, 72, 92} [coarser: res mod 20 in {12}, R2 0.80] | **res** (R2 0.62): res in {12, 14} | res: mod4 +2% | - | 2, 82, 114, 154, 182 | +0.34 / +0.20 |
| L25 down c216 | 6% / 2% | **res%100** (R2 0.80): res mod 100 in {20, 30, 40, 60, 70, 80} | **res** (R2 0.54): res in {20} | - | same | 42, 62, 63, 154, 162 | +0.35 / +0.18 |
| L25 down c248 | 6% / 1% | **res%100** (R2 0.86): res mod 100 in {75..79} | unexplained (best R2 0.27) | - | same | 24, 40, 99, 129, 185 | +0.55 / +0.51 |
| L25 down c297 | 8% / 3% | **res%100** (R2 0.84): res mod 100 in {82..89} | unexplained (best R2 0.41) | - | same | 113, 185, 186, 187, 188 | +0.52 / +0.51 |
| L25 down c331 | 3% / 1% | **res%100** (R2 0.72): res mod 100 in {98} | unexplained (best R2 0.36) | - | same | 113, 118, 119, 122, 197 | +0.17 / +0.29 |
| L25 down c335 | 9% / 2% | **res%100** (R2 0.75): res mod 100 in {13, 51..56, 93} | unexplained (best R2 0.27) | - | same | 73, 139, 149, 166, 167 | +0.52 / +0.55 |
| L25 down c586 | 3% / 1% | **res%100** (R2 0.77): res mod 100 in {33, 53, 73} | unexplained (best R2 0.26) | - | same | 93, 139, 185, 187, 194 | +0.24 / +0.04 |
| L26 down c2 | 11% / 1% | **res%100** (R2 0.72): res mod 100 in {66..74, 76..77} | unexplained (best R2 0.15) | - | same | 67, 170, 171, 172, 177 | +0.55 / +0.23 |
| L26 down c8 | 7% / 9% | **res%100** (R2 0.84): res mod 100 in {19..24} | unexplained (best R2 0.49) | - | res: mod25 +2% | 18, 115, 117, 118, 181 | +0.50 / +0.63 |
| L26 down c12 | 9% / 4% | **res%100** (R2 0.89): res mod 100 in {57, 59..66} | unexplained (best R2 0.28) | res: mod25 +2% | - | 70, 71, 170, 171, 173 | +0.74 / +0.67 |
| L26 down c20 | 7% / 10% | **res%100** (R2 0.82): res mod 100 in {29..34} | unexplained (best R2 0.40) | - | same | 36, 37, 41, 42, 126 | +0.65 / +0.61 |
| L26 down c23 | 5% / 5% | **res%100** (R2 0.75): res mod 100: no class above 0.5 (max 0.45) | **res%100** (R2 0.68): res mod 100: no class above 0.5 (max 0.44) | - | same | 4, 6, 103, 104, 105 | +0.39 / +0.58 |
| L26 down c24 | 16% / 3% | **res%100** (R2 0.73): res mod 100 in {25, 32..36, 38, 53..56, 58, 83..86, 88} | unexplained (best R2 0.30) | - | same | 55, 85, 128, 133, 185 | +0.58 / +0.33 |
| L26 down c33 | 8% / 8% | **res%100** (R2 0.81): res mod 100 in {12, 16..20} | **res** (R2 0.60): res in {12..13, 16..21} | - | same | 17, 18, 19, 118, 119 | +0.65 / +0.80 |
| L26 down c37 | 12% / 7% | **res%100** (R2 0.82): res mod 100 in {7..8, 17..18, 27, 37..38, 57..58, 87, 97..98} | **res** (R2 0.66): res in {-97, 7..8, 17..18, 27, 37..38, 87, 97..98} | res: mod4 +3% | res: mod4 +4% | 5, 6, 37, 117, 137 | +0.61 / +0.65 |
| L26 down c88 | 5% / 1% | **res%100** (R2 0.70): res mod 100 in {46, 48, 86, 88} | unexplained (best R2 0.28) | res: mod4 +3% | - | 48, 56, 88, 148, 156 | +0.69 / +0.62 |
| L26 down c89 | 7% / 4% | **res%100** (R2 0.90): res mod 100 in {76..82} | unexplained (best R2 0.33) | - | same | 28, 62, 90, 91, 184 | +0.73 / +0.80 |
| L26 down c94 | 6% / 1% | **res%100** (R2 0.81): res mod 100 in {53..54, 56..59} | unexplained (best R2 0.19) | - | same | 56, 57, 59, 159, 195 | +0.55 / +0.69 |
| L26 down c113 | 8% / 4% | **res%100** (R2 0.81): res mod 100 in {44..50} | unexplained (best R2 0.32) | - | same | 46, 47, 48, 49, 149 | +0.71 / +0.77 |
| L26 down c118 | 18% / 6% | **res%100** (R2 0.93): res mod 100 in {69..86} | unexplained (best R2 0.46) | - | same | 71, 172, 173, 174, 175 | +0.70 / +0.78 |
| L26 down c139 | 9% / 7% | **res%100** (R2 0.71): res mod 100 in {0..5, 96..99} | unexplained (best R2 0.44) | - | same | 0, 1, 100, 101, 200 | +0.64 / +0.45 |
| L26 down c142 | 6% / 1% | **res%100** (R2 0.57): res mod 100 in {55, 75, 94..99} | **res** (R2 0.50): res in {15, 99} | - | same | 27, 68, 95, 155, 195 | +0.31 / +0.01 |
| L26 down c157 | 5% / 0 | **res%100** (R2 0.52): res mod 100 in {45, 94..95} | unexplained (best R2 0.23) | - | same | 45, 64, 145, 159, 164 | +0.31 / +0.30 |
| L26 down c176 | 7% / 1% | **res%100** (R2 0.60): res mod 100 in {84..88} | unexplained (best R2 0.38) | - | same | 86, 159, 186, 193, 195 | +0.22 / +0.39 |
| L26 down c180 | 1% / 2% | **res%100** (R2 0.67): res mod 100: no class above 0.5 (max 0.42) | **res%100** (R2 0.66): res mod 100: no class above 0.5 (max 0.44) | - | same | 6, 61, 91, 110, 133 | +0.23 / +0.33 |
| L26 down c194 | 11% / 1% | **res%100** (R2 0.77): res mod 100 in {18, 28, 38, 48, 57..59, 68, 78..79, 98} | unexplained (best R2 0.33) | - | same | 54, 56, 76, 156, 176 | +0.42 / +0.24 |
| L26 down c205 | 5% / 5% | **res%100** (R2 0.83): res mod 100 in {10..15} | **res** (R2 0.79): res in {10..15} | - | same | 11, 12, 13, 111, 112 | +0.43 / +0.52 |
| L26 down c207 | 5% / 3% | **res%100** (R2 0.70): res mod 100 in {9} | **res%100** (R2 0.81): res mod 100: no class above 0.5 (max 0.46) | - | same | 6, 78, 82, 87, 191 | +0.42 / +0.00 |
| L26 down c219 | 1% / 0 | **res%100** (R2 0.83): res mod 100 in {61} | off (on 0) | - | same | 61, 83, 133, 161, 167 | +0.45 / +0.44 |
| L26 down c226 | 4% / 1% | **res%100** (R2 0.80): res mod 100 in {45..49} | unexplained (best R2 0.22) | - | same | 7, 27, 56, 77, 143 | +0.54 / +0.58 |
| L26 down c237 | 7% / 3% | **res%100** (R2 0.84): res mod 100 in {49..55} | unexplained (best R2 0.27) | - | same | 94, 96, 126, 128, 193 | +0.68 / +0.71 |
| L26 down c258 | 8% / 1% | **res%100** (R2 0.76): res mod 100 in {13, 33, 53, 63, 73..74, 83, 93} | unexplained (best R2 0.48) | - | same | 41, 73, 173, 183, 193 | +0.46 / +0.26 |
| L26 down c299 | 8% / 1% | **res%100** (R2 0.82): res mod 100 in {72..78} | unexplained (best R2 0.22) | - | same | 80, 85, 86, 115, 179 | +0.13 / +0.18 |
| L26 down c341 | 7% / 2% | **res%100** (R2 0.83): res mod 100 in {91..96} | unexplained (best R2 0.32) | - | same | 133, 134, 135, 143, 153 | -0.09 / -0.02 |
| L26 down c348 | 7% / 2% | **res%100** (R2 0.69): res mod 100 in {25, 34..35, 74..75, 94} | unexplained (best R2 0.36) | - | same | 36, 76, 126, 136, 176 | +0.41 / +0.30 |
| L26 down c529 | 9% / 0 | **res%100** (R2 0.71): res mod 100 in {1} | off (on 0) | - | same | 27, 28, 29, 110, 180 | +0.24 / -0.12 |
| L26 down c570 | 5% / 0 | **res%100** (R2 0.86): res mod 100 in {90..94} | off (on 0) | - | same | 10, 12, 77, 97, 110 | +0.00 / +0.09 |
| L26 down c703 | 4% / 1% | **res%100** (R2 0.78): res mod 100 in {65, 67..69} | unexplained (best R2 0.21) | - | same | 62, 90, 92, 180, 190 | +0.51 / +0.46 |
| L26 down c813 | 4% / 3% | **res%100** (R2 0.77): res mod 100 in {14..17} | **res** (R2 0.74): res in {14..17} | - | same | 14, 15, 16, 29, 116 | +0.30 / +0.52 |
| L26 down c840 | 2% / 0 | **res%100** (R2 0.79): res mod 100 in {53, 93} | off (on 0) | - | same | 53, 81, 93, 153, 193 | +0.49 / +0.35 |
| L27 down c17 | 8% / 4% | **res%100** (R2 0.91): res mod 100 in {79..85} | unexplained (best R2 0.33) | res: mod25 +3% | - | 80, 81, 82, 180, 181 | +0.65 / +0.66 |
| L27 down c24 | 7% / 5% | **res%100** (R2 0.72): res mod 100 in {33..38} | unexplained (best R2 0.42) | - | same | 35, 36, 37, 135, 136 | +0.53 / +0.69 |
| L27 down c32 | 8% / 1% | **res%100** (R2 0.79): res mod 100: no class above 0.5 (max 0.49) | unexplained (best R2 0.18) | - | same | 3, 107, 108, 109, 110 | +0.54 / -0.09 |
| L27 down c38 | 10% / 3% | **res%100** (R2 0.83): res mod 100 in {19, 29, 39, 49, 58..59, 79, 89, 97..99} | **res** (R2 0.53): res in {-99, 19, 29, 39, 79, 98..99} | res: mod4 +4% | - | 9, 95, 96, 97, 190 | +0.58 / +0.40 |
| L27 down c42 | 14% / 4% | **res%100** (R2 0.72): res mod 100 in {9, 19, 29, 39, 48..49, 65..70, 79, 89} | unexplained (best R2 0.48) | res: mod4 +3% | - | 59, 99, 109, 159, 167 | +0.20 / +0.07 |
| L27 down c51 | 9% / 3% | **res%100** (R2 0.82): res mod 100 in {38..45} | unexplained (best R2 0.48) | - | same | 106, 110, 111, 113, 180 | +0.72 / +0.33 |
| L27 down c62 | 11% / 2% | **res%100** (R2 0.67): res mod 100 in {45, 55, 65} | **res%100** (R2 0.68): res mod 100: no class above 0.5 (max 0.44) | - | same | 140, 141, 142, 154, 157 | +0.50 / +0.10 |
| L27 down c64 | 10% / 3% | **res%100** (R2 0.88): res mod 100 in {21..25, 51..55} | **res** (R2 0.63): res in {22..25} | - | same | 52, 152, 153, 154, 155 | +0.55 / +0.40 |
| L27 down c82 | 16% / 5% | **res%100** (R2 0.91): res mod 100 in {50..65} | unexplained (best R2 0.36) | - | same | 76, 86, 89, 160, 161 | -0.13 / -0.75 |
| L27 down c107 | 11% / 2% | **res%100** (R2 0.84): res mod 100 in {51..59, 62..63} | unexplained (best R2 0.36) | - | same | 53, 54, 55, 56, 155 | +0.66 / +0.70 |
| L27 down c110 | 10% / 4% | **res%100** (R2 0.71): res mod 100 in {19, 29, 39, 58..59, 76..79, 89, 99} | unexplained (best R2 0.38) | - | same | 70, 78, 79, 178, 179 | +0.40 / +0.38 |
| L27 down c112 | 8% / 1% | **res%100** (R2 0.71): res mod 100 in {24, 44..45, 48, 84..85} | unexplained (best R2 0.30) | res: mod4 +2% | - | 44, 84, 85, 144, 184 | +0.77 / +0.74 |
| L27 down c122 | 6% / 0 | **res%100** (R2 0.70): res mod 100: no class above 0.5 (max 0.48) | off (on 0) | - | same | 40, 50, 90, 95, 116 | +0.65 / -0.18 |
| L27 down c125 | 7% / 3% | **res%100** (R2 0.69): res mod 100 in {48..49, 96..99} [coarser: res mod 50 in {47..49}, R2 0.84] | unexplained (best R2 0.34) | - | same | 48, 49, 99, 149, 199 | +0.55 / +0.43 |
| L27 down c139 | 10% / 2% | **res%100** (R2 0.76): res mod 100 in {1, 21, 31, 41, 51, 61, 71, 81} | **res%100** (R2 0.59): res mod 100: no class above 0.5 (max 0.46) | - | same | 41, 51, 71, 81, 91 | +0.37 / +0.15 |
| L27 down c157 | 5% / 0 | **res%100** (R2 0.77): res mod 100: no class above 0.5 (max 0.46) | off (on 0) | - | same | 70, 108, 109, 125, 152 | +0.43 / +0.09 |
| L27 down c168 | 7% / 4% | **res%100** (R2 0.76): res mod 100 in {40..47} | unexplained (best R2 0.47) | - | same | 50, 52, 53, 84, 85 | +0.73 / +0.88 |
| L27 down c200 | 7% / 0 | **res%100** (R2 0.64): res mod 100 in {48..50, 89, 94} | off (on 0) | - | same | 49, 146, 147, 148, 149 | +0.64 / +0.56 |
| L27 down c203 | 5% / 1% | **res%100** (R2 0.79): res mod 100 in {27, 47, 67, 87} | unexplained (best R2 0.42) | - | same | 138, 141, 149, 161, 168 | +0.46 / +0.57 |
| L27 down c224 | 16% / 8% | **res%100** (R2 0.69): res mod 100 in {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95} [coarser: res mod 10 in {0}, R2 0.96] | **units(a,b)** (R2 0.58): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {1} | - | same | 89, 124, 129, 189, 192 | +0.82 / +0.72 |
| L27 down c269 | 7% / 4% | **res%100** (R2 0.83): res mod 100 in {1, 21, 41, 51, 61, 71, 81, 91} [coarser: res mod 10 in {1}, R2 0.88] | **res** (R2 0.57): res in {1, 11, 31, 41} | - | same | 21, 111, 121, 131, 181 | +0.13 / +0.28 |
| L27 down c461 | 2% / 0 | **res%100** (R2 0.75): res mod 100 in {56, 58..59} | off (on 0) | - | same | 42, 44, 59, 126, 159 | +0.57 / +0.62 |
| L27 down c480 | 4% / 1% | **res%100** (R2 0.81): res mod 100 in {86..89} | unexplained (best R2 0.37) | - | same | 36, 126, 146, 149, 184 | +0.47 / +0.52 |
| L27 down c586 | 2% / 3% | **res%100** (R2 0.85): res mod 100 in {20..21} | **res** (R2 0.61): res in {17..18, 20..21} | - | same | 23, 24, 25, 119, 125 | +0.02 / +0.03 |
| L27 down c818 | 2% / 0 | **res%100** (R2 0.64): res mod 100 in {61..62} | off (on 0) | - | same | 62, 139, 149, 161, 162 | +0.41 / +0.30 |
| L27 down c823 | 1% / 2% | **res%100** (R2 0.64): res mod 100: no class above 0.5 (max 0.36) | **cmp(a,b)** (R2 0.53): cmp(a,b)=-1: 0.01, cmp(a,b)=0: 0.96, cmp(a,b)=1: 0.00 | - | same | 10, 15, 130, 150, 200 | +0.02 / +0.26 |
| L28 down c22 | 17% / 15% | **res%100** (R2 0.72): res mod 100 in {0..1, 7, 17, 37, 47, 57, 87, 91..99} | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {9,10}; a//10 in {1,2,3,4,8} -> b//10 in {9}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,9,10}; a//10 in {10} -> b//10 in {0} | - | same | 9, 49, 79, 196, 197 | +0.10 / -0.19 |
| L28 down c24 | 12% / 15% | **res%100** (R2 0.81): res mod 100 in {6, 15..17, 26, 36, 46, 56, 66, 75..77, 86, 96} [coarser: res mod 20 in {6, 16}, R2 0.80] | unexplained (best R2 0.47) | res: mod4 +4% | res: mod4 +3% | 6, 74, 86, 106, 174 | +0.26 / -0.02 |
| L28 down c25 | 3% / 0 | **res%100** (R2 0.67): res mod 100: no class above 0.5 (max 0.42) | off (on 0) | - | same | 103, 107, 108, 123, 192 | +0.78 / +nan |
| L28 down c40 | 3% / 1% | **res%100** (R2 0.73): res mod 100 in {1} | unexplained (best R2 0.32) | - | same | 71, 99, 101, 103, 131 | +0.58 / +0.68 |
| L28 down c46 | 8% / 5% | **res%100** (R2 0.73): res mod 100 in {29..33, 70, 80, 90} | unexplained (best R2 0.37) | - | same | 20, 26, 27, 28, 29 | +0.52 / +0.59 |
| L28 down c51 | 5% / 2% | **res%100** (R2 0.75): res mod 100 in {18, 38, 58, 78, 98} [coarser: res mod 20 in {18}, R2 0.87] | unexplained (best R2 0.40) | - | res: mod4 +2% | 20, 116, 117, 119, 148 | +0.50 / +0.68 |
| L28 down c62 | 13% / 12% | **res%100** (R2 0.52): res mod 100 in {80, 86..87, 90..92, 96} | **tens(a,b)** (R2 0.68): a//10 in {7,10} -> b//10 in {1}; a//10 in {8} -> b//10 in {0,1,2}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,10} | - | same | 49, 55, 58, 195, 196 | +0.46 / +0.63 |
| L28 down c67 | 13% / 7% | **res%100** (R2 0.85): res mod 100 in {7, 17, 27, 37, 47, 57, 67, 75..79, 87, 97} [coarser: res mod 50 in {7, 17, 27..28, 37, 47}, R2 0.88] | **res** (R2 0.66): res in {-23, -13, -3, 7, 17, 27, 37, 47, 57, 67, 77, 87, 97} | - | same | 15, 27, 47, 127, 147 | +0.17 / +0.09 |
| L28 down c71 | 14% / 3% | **res%100** (R2 0.83): res mod 100 in {26, 45..46, 60..69, 86} | unexplained (best R2 0.32) | - | same | 48, 68, 86, 106, 126 | +0.35 / +0.44 |
| L28 down c80 | 8% / 4% | **res%100** (R2 0.89): res mod 100 in {68..75} | unexplained (best R2 0.19) | - | same | 71, 169, 170, 171, 172 | +0.51 / +0.47 |
| L28 down c82 | 16% / 8% | **res%100** (R2 0.84): res mod 100 in {2..3, 13, 23, 33, 39..45, 53, 63, 73, 83, 93} | **res** (R2 0.60): res in {3, 13, 23, 33, 40..44, 53, 63, 73, 83, 93} | - | same | 44, 46, 73, 163, 173 | +0.05 / -0.03 |
| L28 down c83 | 12% / 3% | **res%100** (R2 0.85): res mod 100 in {8, 18, 28, 38, 48, 58, 68, 75..79, 98} | unexplained (best R2 0.48) | - | same | 78, 107, 116, 126, 167 | +0.02 / +0.01 |
| L28 down c90 | 14% / 4% | **res%100** (R2 0.58): res mod 100 in {12, 31..32, 52, 71..72, 90..92} | unexplained (best R2 0.29) | - | same | 83, 120, 140, 165, 166 | -0.12 / -0.06 |
| L28 down c99 | 9% / 4% | **res%100** (R2 0.89): res mod 100 in {8, 28, 38, 48, 58, 68, 87..88, 98} [coarser: res mod 10 in {8}, R2 0.81] | **res** (R2 0.54): res in {8, 28, 38, 48, 88, 98} | - | same | 102, 106, 126, 162, 197 | +0.69 / +0.65 |
| L28 down c115 | 12% / 3% | **res%100** (R2 0.94): res mod 100 in {78..88} | unexplained (best R2 0.46) | - | same | 80, 81, 87, 102, 180 | -0.19 / -0.33 |
| L28 down c121 | 10% / 4% | **res%100** (R2 0.85): res mod 100 in {9, 19, 28..29, 39, 49, 59, 69, 79, 89} [coarser: res mod 50 in {9, 19, 28..29, 39, 49}, R2 0.85] | unexplained (best R2 0.42) | - | same | 9, 19, 109, 119, 169 | +0.62 / +0.81 |
| L28 down c124 | 3% / 0 | **res%100** (R2 0.74): res mod 100 in {52..53, 93} | off (on 0) | - | same | 54, 143, 154, 157, 159 | +0.43 / +0.42 |
| L28 down c140 | 5% / 3% | **res%100** (R2 0.82): res mod 100 in {4, 64, 74, 84, 94} | unexplained (best R2 0.49) | - | same | 3, 98, 99, 103, 105 | +0.73 / +0.83 |
| L28 down c150 | 2% / 0 | **res%100** (R2 0.64): res mod 100 in {59} | off (on 0) | - | same | 29, 49, 109, 149, 189 | +0.24 / +0.07 |
| L28 down c165 | 6% / 3% | **res%100** (R2 0.79): res mod 100 in {25..28, 66..68} | unexplained (best R2 0.48) | - | same | 27, 48, 98, 126, 148 | +0.51 / +0.34 |
| L28 down c182 | 5% / 2% | **res%100** (R2 0.69): res mod 100 in {76..79} | unexplained (best R2 0.30) | - | same | 76, 77, 78, 79, 190 | +0.43 / +0.68 |
| L28 down c191 | 10% / 2% | **res%100** (R2 0.84): res mod 100 in {0..1, 51..53} | unexplained (best R2 0.47) | - | same | 51, 101, 102, 151, 153 | +0.57 / +0.39 |
| L28 down c291 | 6% / 2% | **res%100** (R2 0.61): res mod 100 in {17, 57, 67, 77} | unexplained (best R2 0.40) | - | same | 57, 67, 77, 97, 115 | +0.45 / +0.60 |
| L28 down c296 | 5% / 3% | **res%100** (R2 0.67): res mod 100 in {0, 20, 40, 60, 80} [coarser: res mod 20 in {0}, R2 0.88] | unexplained (best R2 0.48) | - | same | 60, 120, 140, 160, 180 | +0.52 / +0.21 |
| L28 down c298 | 11% / 5% | **res%100** (R2 0.71): res mod 100 in {0, 49, 89..99} | unexplained (best R2 0.44) | - | same | 132, 137, 153, 163, 172 | +0.74 / +0.41 |
| L28 down c416 | 1% / 9% | **res%100** (R2 0.77): res mod 100 in {9} | **res%100** (R2 0.63): res mod 100: no class above 0.5 (max 0.47) | - | same | 19, 77, 78, 79, 99 | +0.02 / +0.22 |
| L28 down c433 | 5% / 0 | **res%100** (R2 0.66): res mod 100: no class above 0.5 (max 0.46) | off (on 0) | - | same | 23, 27, 41, 87, 88 | +0.06 / +0.08 |
| L28 down c435 | 1% / 2% | **res%100** (R2 0.89): res mod 100 in {8} | **res%100** (R2 0.51): res mod 100 in {8} | - | same | 8, 108, 116, 117, 157 | +0.41 / +0.62 |
| L28 down c522 | 2% / 1% | **res%100** (R2 0.74): res mod 100 in {17, 47} | **res** (R2 0.61): res in {17} | - | same | 47, 117, 147, 171, 174 | +0.44 / +0.27 |
| L28 down c545 | 1% / 0 | **res%100** (R2 0.53): res mod 100: no class above 0.5 (max 0.40) | off (on 0) | - | same | 33, 47, 77, 99, 147 | +0.35 / +0.47 |
| L28 down c595 | 5% / 1% | **res%100** (R2 0.73): res mod 100 in {42..47} | unexplained (best R2 0.21) | - | same | 48, 49, 85, 149, 150 | +0.45 / +0.36 |
| L28 down c660 | 2% / 2% | **res%100** (R2 0.52): res mod 100 in {82} | **res%100** (R2 0.66): res mod 100 in {0} | - | same | 110, 114, 117, 128, 171 | +0.19 / +0.28 |
| L28 down c719 | 4% / 1% | **res%100** (R2 0.81): res mod 100 in {32, 52, 72, 92} | unexplained (best R2 0.33) | - | same | 71, 102, 112, 131, 171 | +0.60 / +0.39 |
| L28 down c752 | 2% / 0 | **res%100** (R2 0.69): res mod 100 in {61..62} | off (on 0) | - | same | 62, 175, 176, 177, 178 | +0.17 / +0.39 |
| L28 down c794 | 4% / 7% | **res%100** (R2 0.69): res mod 100 in {21..24} | unexplained (best R2 0.46) | - | same | 118, 119, 126, 129, 139 | +0.10 / -0.02 |
| L28 down c903 | 2% / 0 | **res%100** (R2 0.71): res mod 100 in {72} | off (on 0) | - | same | 72, 102, 128, 135, 172 | +0.60 / +0.56 |
| L28 down c952 | 2% / 0 | **res%100** (R2 0.89): res mod 100 in {69} | off (on 0) | - | same | 59, 79, 89, 99, 159 | +0.07 / +0.02 |
| L28 down c965 | 2% / 0 | **res%100** (R2 0.58): res mod 100 in {71, 91} | off (on 0) | - | same | 10, 71, 91, 177, 191 | +0.33 / +0.34 |
| L29 down c48 | 12% / 2% | **res%100** (R2 0.75): res mod 100 in {1, 10..11, 50, 70} | **res%100** (R2 0.68): res mod 100 in {10..11} | - | same | 77, 79, 83, 94, 110 | -0.11 / -0.47 |
| L29 down c64 | 10% / 6% | **res%100** (R2 0.72): res mod 100 in {35..41, 58, 88} | unexplained (best R2 0.47) | - | same | 69, 71, 73, 113, 138 | -0.02 / -0.11 |
| L29 down c68 | 10% / 2% | **res%100** (R2 0.75): res mod 100 in {6..7, 9, 46, 56, 66, 86} | **res%100** (R2 0.66): res mod 100 in {6} | - | same | 104, 105, 126, 136, 147 | +0.25 / -0.05 |
| L29 down c71 | 9% / 2% | **res%100** (R2 0.71): res mod 100 in {14, 34, 44..45, 54, 93..95} | unexplained (best R2 0.22) | - | same | 94, 95, 108, 144, 194 | +0.40 / +0.42 |
| L29 down c90 | 8% / 1% | **res%100** (R2 0.74): res mod 100 in {66..70} | unexplained (best R2 0.17) | - | same | 6, 67, 68, 167, 168 | +0.47 / +0.49 |
| L29 down c92 | 11% / 1% | **res%100** (R2 0.78): res mod 100 in {33, 51..58, 63, 73, 93} | unexplained (best R2 0.23) | - | same | 51, 52, 55, 151, 152 | +0.01 / +0.08 |
| L29 down c122 | 1% / 2% | **res%100** (R2 0.85): res mod 100 in {6} | **res%100** (R2 0.54): res mod 100 in {6} | - | same | 6, 17, 61, 67, 117 | +0.56 / +0.73 |
| L29 down c128 | 9% / 0 | **res%100** (R2 0.85): res mod 100: no class above 0.5 (max 0.50) | off (on 0) | - | same | 101, 114, 116, 117, 147 | +0.57 / -0.15 |
| L29 down c130 | 12% / 4% | **res%100** (R2 0.81): res mod 100 in {4, 14, 24, 34, 44, 54, 63..65, 74, 84, 94} [coarser: res mod 10 in {4}, R2 0.81] | **res** (R2 0.65): res in {4, 14, 24, 34, 44, 64, 84, 94} | - | same | 24, 34, 74, 114, 174 | +0.28 / -0.14 |
| L29 down c136 | 13% / 7% | **res%100** (R2 0.76): res mod 100 in {1, 11, 21, 31, 39..43, 45, 51, 61, 71, 81, 91} [coarser: res mod 50 in {1, 11, 21, 31, 41}, R2 0.82] | unexplained (best R2 0.40) | - | same | 41, 64, 84, 91, 141 | +0.30 / +0.32 |
| L29 down c140 | 6% / 0 | **res%100** (R2 0.81): res mod 100 in {83..89} | off (on 0) | - | same | 108, 183, 184, 185, 186 | +0.29 / +0.18 |
| L29 down c146 | 8% / 2% | **res%100** (R2 0.74): res mod 100 in {73..79} | unexplained (best R2 0.23) | - | same | 95, 172, 173, 177, 185 | +0.32 / +0.46 |
| L29 down c150 | 8% / 3% | **res%100** (R2 0.81): res mod 100 in {81..87} | unexplained (best R2 0.42) | - | same | 80, 82, 83, 84, 85 | +0.70 / +0.75 |
| L29 down c172 | 7% / 1% | **res%100** (R2 0.69): res mod 100 in {45, 47, 49, 83, 85, 87, 89} | unexplained (best R2 0.20) | res: mod4 +3% | - | 45, 85, 97, 145, 185 | +0.60 / +0.60 |
| L29 down c174 | 14% / 1% | **res%100** (R2 0.74): res mod 100 in {1, 97} | unexplained (best R2 0.45) | - | same | 100, 102, 103, 104, 109 | +0.65 / +0.45 |
| L29 down c178 | 4% / 3% | **res%100** (R2 0.61): res mod 100 in {27..29, 68} | unexplained (best R2 0.43) | - | same | 8, 68, 117, 126, 168 | +0.58 / +0.77 |
| L29 down c183 | 2% / 0 | **res%100** (R2 0.89): res mod 100 in {34, 94} | off (on 0) | - | same | 44, 54, 74, 144, 154 | +0.44 / +0.58 |
| L29 down c213 | 3% / 6% | **res%100** (R2 0.64): res mod 100: no class above 0.5 (max 0.50) | **tens(a,b)** (R2 0.62): a//10 in {9} -> b//10 in {0,2,3,4,5}; a//10 in {10} -> b//10 in {0} | - | same | 84, 86, 99, 101, 102 | +0.64 / +0.71 |
| L29 down c217 | 7% / 1% | **res%100** (R2 0.63): res mod 100 in {39, 49, 69, 79, 89} [coarser: res mod 50 in {39}, R2 0.80] | unexplained (best R2 0.36) | - | same | 38, 88, 138, 149, 179 | +0.27 / +0.24 |
| L29 down c238 | 3% / 0 | **res%100** (R2 0.86): res mod 100 in {77..79} | off (on 0) | - | same | 45, 79, 116, 136, 179 | +0.27 / +0.47 |
| L29 down c263 | 2% / 2% | **res%100** (R2 0.66): res mod 100 in {13} | unexplained (best R2 0.39) | - | same | 11, 15, 112, 114, 115 | +0.49 / +0.78 |
| L29 down c279 | 3% / 6% | **res%100** (R2 0.63): res mod 100 in {0, 19..22} | **res** (R2 0.51): res in {-20, 19..23} | - | same | 16, 17, 18, 19, 121 | +0.09 / +0.26 |
| L29 down c280 | 3% / 1% | **res%100** (R2 0.60): res mod 100 in {16} | **res** (R2 0.63): res in {16..17} | - | same | 18, 76, 115, 118, 176 | +0.40 / +0.50 |
| L29 down c282 | 10% / 3% | **res%100** (R2 0.83): res mod 100 in {22, 40..43, 52, 62, 82, 92} | **res** (R2 0.51): res in {22, 41..43} | - | same | 42, 71, 75, 94, 142 | +0.20 / +0.22 |
| L29 down c308 | 4% / 1% | **res%100** (R2 0.72): res mod 100 in {46..49} | unexplained (best R2 0.24) | - | same | 38, 129, 146, 150, 189 | +0.42 / +0.59 |
| L29 down c318 | 1% / 2% | **res%100** (R2 0.77): res mod 100 in {12} | unexplained (best R2 0.50) | - | same | 12, 74, 164, 167, 168 | +0.18 / +0.34 |
| L29 down c321 | 8% / 1% | **res%100** (R2 0.72): res mod 100 in {5, 15, 25, 35, 45, 64..66, 85, 95} | **res** (R2 0.53): res in {5} | - | same | 65, 117, 129, 152, 165 | +0.38 / -0.18 |
| L29 down c343 | 2% / 2% | **res%100** (R2 0.67): res mod 100 in {17} | unexplained (best R2 0.47) | - | same | 13, 15, 116, 118, 165 | +0.69 / +0.87 |
| L29 down c358 | 9% / 3% | **res%100** (R2 0.83): res mod 100 in {1, 95..97, 99} | unexplained (best R2 0.48) | - | same | 103, 109, 110, 119, 120 | +0.09 / -0.15 |
| L29 down c376 | 2% / 2% | **res%100** (R2 0.65): res mod 100 in {26} | unexplained (best R2 0.30) | - | same | 6, 25, 66, 68, 125 | +0.75 / +0.86 |
| L29 down c414 | 1% / 0 | **res%100** (R2 0.76): res mod 100 in {94} | off (on 0) | - | same | 44, 95, 144, 145, 195 | +0.41 / +0.48 |
| L29 down c420 | 1% / 0 | **res%100** (R2 0.61): res mod 100 in {90} | off (on 0) | - | same | 61, 88, 89, 95, 148 | +0.50 / +0.58 |
| L29 down c433 | 3% / 2% | **res%100** (R2 0.70): res mod 100 in {51..53} | unexplained (best R2 0.26) | - | same | 57, 58, 62, 64, 122 | +0.62 / +0.77 |
| L29 down c434 | 3% / 1% | **res%100** (R2 0.90): res mod 100 in {1, 51, 81} | unexplained (best R2 0.25) | - | same | 71, 83, 111, 161, 171 | +0.56 / +0.45 |
| L29 down c438 | 1% / 1% | **res%100** (R2 0.61): res mod 100 in {62} | unexplained (best R2 0.09) | - | same | 64, 71, 75, 76, 77 | +0.33 / +0.67 |
| L29 down c444 | 8% / 1% | **res%100** (R2 0.69): res mod 100 in {90..93, 97..98} | unexplained (best R2 0.25) | - | same | 16, 32, 98, 99, 123 | +0.04 / +0.24 |
| L29 down c445 | 1% / 2% | **res%100** (R2 0.67): res mod 100 in {11} | **res%100** (R2 0.49): res mod 100 in {11} | - | same | 11, 111, 119, 167, 187 | +0.61 / +0.75 |
| L29 down c467 | 1% / 1% | **res%100** (R2 0.74): res mod 100 in {15} | **res** (R2 0.57): res in {15} | - | same | 15, 115, 152, 153, 158 | +0.58 / +0.76 |
| L29 down c477 | 2% / 1% | **res%100** (R2 0.63): res mod 100 in {16, 96} | **res** (R2 0.60): res in {16, 96} | - | same | 16, 96, 116, 161, 196 | +0.65 / +0.66 |
| L29 down c497 | 1% / 0 | **res%100** (R2 0.69): res mod 100 in {1} | off (on 0) | - | same | 89, 193, 195, 196, 197 | +0.11 / -0.05 |
| L29 down c566 | 1% / 0 | **res%100** (R2 0.64): res mod 100 in {35} | off (on 0) | - | same | 34, 35, 68, 135, 163 | +0.63 / +0.79 |
| L29 down c595 | 5% / 1% | **res%100** (R2 0.67): res mod 100 in {87..89} | unexplained (best R2 0.29) | - | same | 39, 139, 179, 191, 193 | +0.60 / +0.70 |
| L29 down c618 | 1% / 1% | **res%100** (R2 0.87): res mod 100 in {27} | **res** (R2 0.76): res in {27} | - | same | 112, 127, 134, 135, 136 | +0.19 / -0.05 |
| L29 down c653 | 3% / 0 | **res%100** (R2 0.77): res mod 100 in {70..71} | off (on 0) | - | same | 3, 8, 83, 85, 172 | +0.04 / -0.22 |
| L29 down c736 | 2% / 1% | **res%100** (R2 0.60): res mod 100: no class above 0.5 (max 0.47) | **res** (R2 0.61): res in {19, 99} | - | same | 72, 77, 87, 99, 165 | +0.36 / +0.45 |
| L29 down c794 | 6% / 0 | **res%100** (R2 0.83): res mod 100 in {1} | off (on 0) | - | same | 70, 78, 89, 90, 190 | -0.34 / -0.17 |
| L29 down c824 | 2% / 0 | **res%100** (R2 0.84): res mod 100 in {33, 83, 93} | off (on 0) | - | same | 62, 93, 133, 183, 193 | +0.49 / +0.38 |
| L29 down c842 | 5% / 2% | **res%100** (R2 0.74): res mod 100 in {21, 41, 51, 61, 81} | unexplained (best R2 0.28) | - | same | 21, 102, 109, 121, 189 | +0.25 / +0.67 |
| L29 down c865 | 2% / 2% | **res%100** (R2 0.75): res mod 100 in {13, 63} [coarser: res mod 50 in {13}, R2 0.85] | **res** (R2 0.62): res in {13, 23} | - | same | 3, 122, 124, 166, 183 | +0.70 / +0.57 |
| L29 down c930 | 7% / 2% | **res%100** (R2 0.66): res mod 100 in {29, 59, 66, 69, 96, 99} | unexplained (best R2 0.24) | - | same | 71, 72, 73, 98, 189 | +0.63 / +0.47 |
| L29 down c964 | 2% / 0 | **res%100** (R2 0.76): res mod 100 in {49..50} | off (on 0) | - | same | 46, 47, 48, 148, 170 | +0.37 / +0.36 |
| L29 down c1009 | 3% / 1% | **res%100** (R2 0.62): res mod 100 in {90} | unexplained (best R2 0.41) | - | same | 89, 108, 110, 145, 189 | +0.42 / +0.39 |
| L30 down c177 | 10% / 1% | **res%100** (R2 0.71): res mod 100 in {26, 56..58, 66, 95..98} | unexplained (best R2 0.25) | - | same | 46, 57, 76, 97, 157 | +0.59 / +0.43 |
| L30 down c181 | 8% / 2% | **res%100** (R2 0.59): res mod 100 in {16..20} | **res** (R2 0.51): res in {17..19} | - | same | 4, 44, 74, 92, 94 | -0.12 / -0.64 |
| L30 down c183 | 6% / 1% | **res%100** (R2 0.62): res mod 100 in {31..33, 82} | unexplained (best R2 0.48) | - | same | 49, 164, 167, 169, 170 | +0.03 / +0.08 |
| L30 down c199 | 8% / 1% | **res%100** (R2 0.76): res mod 100 in {40..47} | unexplained (best R2 0.25) | - | same | 140, 142, 146, 184, 194 | -0.13 / -0.04 |
| L30 down c207 | 8% / 3% | **res%100** (R2 0.80): res mod 100 in {26..33} | **res** (R2 0.65): res in {28..31} | - | same | 28, 29, 126, 127, 128 | -0.30 / -0.59 |
| L30 down c214 | 7% / 1% | **res%100** (R2 0.77): res mod 100 in {17, 27, 37, 57, 77..78, 97} | **res** (R2 0.60): res in {17} | - | same | 48, 68, 77, 144, 177 | +0.26 / -0.03 |
| L30 down c232 | 9% / 1% | **res%100** (R2 0.74): res mod 100 in {77, 81..84, 88} | unexplained (best R2 0.31) | - | same | 86, 180, 182, 184, 186 | +0.20 / +0.48 |
| L30 down c239 | 19% / 3% | **res%100** (R2 0.83): res mod 100 in {33, 42..43, 52..53, 62..63, 72..73, 82..83, 92..93} | unexplained (best R2 0.41) | res: mod2 +2% | - | 113, 133, 143, 153, 173 | +0.89 / +0.77 |
| L30 down c249 | 8% / 1% | **res%100** (R2 0.71): res mod 100 in {33, 60, 62..65, 83} | unexplained (best R2 0.22) | - | same | 63, 83, 94, 140, 163 | +0.22 / +0.12 |
| L30 down c309 | 3% / 1% | **res%100** (R2 0.81): res mod 100 in {4..5} | **res%100** (R2 0.52): res mod 100: no class above 0.5 (max 0.42) | - | same | 102, 103, 107, 108, 109 | +0.14 / +0.06 |
| L30 down c326 | 3% / 0 | **res%100** (R2 0.74): res mod 100 in {56..57} | off (on 0) | - | same | 55, 58, 59, 137, 153 | +0.15 / +0.06 |
| L30 down c357 | 4% / 0 | **res%100** (R2 0.73): res mod 100 in {85..87} | off (on 0) | - | same | 82, 83, 84, 183, 184 | +0.28 / +0.43 |
| L30 down c359 | 8% / 3% | **res%100** (R2 0.82): res mod 100 in {23..26, 44, 64, 74, 84} | **res** (R2 0.71): res in {23..26} | - | same | 14, 22, 23, 26, 126 | +0.08 / +0.12 |
| L30 down c370 | 8% / 1% | **res%100** (R2 0.71): res mod 100 in {90..95} | unexplained (best R2 0.39) | - | same | 52, 89, 90, 91, 191 | +0.15 / -0.10 |
| L30 down c394 | 3% / 0 | **res%100** (R2 0.77): res mod 100 in {67, 87} | off (on 0) | - | same | 66, 68, 69, 77, 169 | +0.58 / +0.30 |
| L30 down c407 | 6% / 2% | **res%100** (R2 0.76): res mod 100 in {25..29} | unexplained (best R2 0.50) | - | same | 16, 34, 72, 127, 128 | +0.22 / -0.37 |
| L30 down c452 | 1% / 0 | **res%100** (R2 0.84): res mod 100 in {71} | off (on 0) | - | same | 71, 92, 160, 188, 192 | +0.63 / +0.86 |
| L30 down c475 | 1% / 0 | **res%100** (R2 0.68): res mod 100 in {58, 68} | off (on 0) | - | same | 58, 68, 97, 158, 168 | +0.67 / +0.73 |
| L30 down c486 | 3% / 1% | **res%100** (R2 0.87): res mod 100 in {79..81} | unexplained (best R2 0.21) | - | same | 82, 83, 88, 182, 183 | +0.34 / +0.33 |
| L30 down c501 | 5% / 1% | **res%100** (R2 0.76): res mod 100 in {51, 61, 90..92} | unexplained (best R2 0.19) | - | same | 51, 61, 91, 138, 191 | +0.36 / +0.31 |
| L30 down c519 | 1% / 1% | **res%100** (R2 0.76): res mod 100 in {14} | **res** (R2 0.62): res in {14} | - | same | 112, 116, 117, 124, 134 | +0.52 / +0.85 |
| L30 down c535 | 5% / 4% | **res%100** (R2 0.76): res mod 100 in {0, 94..99} | **a** (R2 0.69): a in {97..99} | - | same | 93, 118, 143, 169, 191 | +0.41 / -0.17 |
| L30 down c594 | 5% / 0 | **res%100** (R2 0.60): res mod 100: no class above 0.5 (max 0.44) | off (on 0) | - | same | 110, 111, 113, 115, 118 | +0.20 / -0.17 |
| L30 down c603 | 3% / 1% | **res%100** (R2 0.68): res mod 100 in {69..70} | unexplained (best R2 0.14) | - | same | 67, 72, 74, 167, 172 | +0.29 / +0.42 |
| L30 down c628 | 2% / 1% | **res%100** (R2 0.77): res mod 100 in {23..24} | **res** (R2 0.60): res in {23..24} | - | same | 23, 47, 92, 102, 123 | +0.47 / +0.54 |
| L30 down c642 | 1% / 1% | **res%100** (R2 0.59): res mod 100 in {18} | **res** (R2 0.58): res in {18} | - | same | 14, 119, 141, 157, 195 | +0.36 / +0.82 |
| L30 down c739 | 1% / 1% | **res%100** (R2 0.84): res mod 100 in {19} | **res** (R2 0.66): res in {19} | - | same | 9, 59, 116, 117, 118 | +0.06 / +0.25 |
| L30 down c796 | 3% / 1% | **res%100** (R2 0.73): res mod 100 in {20, 40, 60} | **res** (R2 0.54): res in {20} | - | same | 61, 62, 121, 122, 162 | +0.12 / +0.28 |
| L30 down c808 | 5% / 0 | **res%100** (R2 0.83): res mod 100 in {67..70} | off (on 0) | - | same | 66, 67, 71, 79, 166 | -0.33 / -0.37 |
| L30 down c810 | 3% / 1% | **res%100** (R2 0.75): res mod 100 in {52, 82} | **res%100** (R2 0.59): res mod 100 in {2} | - | same | 52, 70, 90, 102, 152 | +0.41 / +0.15 |
| L30 down c879 | 1% / 1% | **res%100** (R2 0.64): res mod 100 in {16} | **res** (R2 0.65): res in {16} | - | same | 69, 173, 179, 187, 189 | +0.35 / +0.94 |
| L30 down c961 | 5% / 1% | **res%100** (R2 0.80): res mod 100 in {50..54} | unexplained (best R2 0.26) | - | same | 53, 54, 55, 154, 157 | +0.17 / -0.00 |
| L30 down c997 | 1% / 2% | **res%100** (R2 0.58): res mod 100 in {22} | unexplained (best R2 0.39) | - | same | 20, 23, 121, 123, 182 | +0.53 / +0.66 |
| L30 down c1012 | 2% / 0 | **res%100** (R2 0.80): res mod 100 in {79..80} | off (on 0) | - | same | 58, 67, 68, 164, 179 | +0.17 / +0.07 |
| L31 down c56 | 1% / 0 | **res%100** (R2 0.61): res mod 100: no class above 0.5 (max 0.39) | off (on 0) | - | same | 79, 101, 171, 190, 191 | +0.80 / +nan |
| L31 down c65 | 4% / 0 | **res%100** (R2 0.56): res mod 100 in {97} | off (on 0) | - | same | 97, 98, 104, 107, 108 | +0.83 / +0.91 |
| L31 down c73 | 1% / 0 | **res%100** (R2 0.57): res mod 100 in {86} | off (on 0) | - | same | 48, 84, 97, 196, 197 | +0.36 / +0.47 |
| L31 down c126 | 1% / 0 | **res%100** (R2 0.77): res mod 100 in {83} | off (on 0) | - | same | 87, 155, 156, 193, 197 | +0.61 / +nan |
| L31 down c128 | 1% / 0 | **res%100** (R2 0.66): res mod 100: no class above 0.5 (max 0.38) | off (on 0) | - | same | 100, 107, 145, 166, 175 | +0.46 / +nan |
| L31 down c187 | 13% / 5% | **res%100** (R2 0.58): res mod 100 in {25, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90} | unexplained (best R2 0.49) | - | same | 74, 76, 117, 178, 179 | +0.87 / +0.84 |
| L31 down c196 | 1% / 0 | **res%100** (R2 0.72): res mod 100 in {1} | off (on 0) | - | same | 78, 94, 101, 135, 172 | +0.79 / +nan |
| L31 down c211 | 1% / 0 | **res%100** (R2 0.80): res mod 100 in {34} | off (on 0) | - | same | 34, 67, 134, 149, 194 | +0.90 / +0.78 |
| L31 down c219 | 5% / 1% | **res%100** (R2 0.80): res mod 100 in {81..86} | **res** (R2 0.52): res in {81..86} | - | same | 71, 87, 89, 178, 186 | +0.79 / +0.91 |
| L31 down c239 | 4% / 2% | **res%100** (R2 0.73): res mod 100 in {61, 81, 91} | unexplained (best R2 0.20) | - | same | 61, 81, 91, 121, 161 | +0.84 / +0.77 |
| L31 down c247 | 3% / 0 | **res%100** (R2 0.71): res mod 100 in {84..85, 89} | off (on 0) | - | same | 88, 109, 137, 138, 197 | +0.84 / +0.93 |
| L31 down c296 | 3% / 1% | **res%100** (R2 0.82): res mod 100 in {81..83} | unexplained (best R2 0.31) | - | same | 81, 82, 181, 182, 183 | +0.92 / +0.90 |
| L31 down c355 | 6% / 3% | **res%100** (R2 0.72): res mod 100 in {66..67, 86..89} | unexplained (best R2 0.32) | - | same | 66, 86, 87, 88, 89 | +0.86 / +0.79 |
| L31 down c484 | 1% / 0 | **res%100** (R2 0.67): res mod 100: no class above 0.5 (max 0.42) | off (on 0) | - | same | 98, 176, 180, 183, 189 | +0.70 / +0.65 |
| L31 down c495 | 3% / 1% | **res%100** (R2 0.90): res mod 100 in {76..78} | unexplained (best R2 0.29) | - | same | 71, 105, 125, 155, 163 | +0.90 / +0.80 |
| L31 down c522 | 2% / 0 | **res%100** (R2 0.74): res mod 100 in {71, 81} | off (on 0) | - | same | 71, 81, 142, 169, 181 | +0.84 / +0.79 |
| L31 down c527 | 2% / 0 | **res%100** (R2 0.84): res mod 100: no class above 0.5 (max 0.46) | off (on 0) | - | same | 95, 96, 119, 148, 199 | +0.70 / +0.80 |
| L31 down c623 | 4% / 1% | **res%100** (R2 0.86): res mod 100 in {77..79, 81} | unexplained (best R2 0.34) | - | same | 78, 177, 178, 179, 181 | +0.91 / +0.81 |
| L31 down c640 | 23% / 0 | **res%100** (R2 0.76): res mod 100 in {1, 96..98} | off (on 0) | - | same | 101, 102, 104, 112, 114 | +0.78 / -0.36 |
| L31 down c998 | 3% / 0 | **res%100** (R2 0.75): res mod 100: no class above 0.5 (max 0.45) | off (on 0) | - | same | 105, 106, 107, 195, 197 | +0.84 / +nan |

</details>

<details><summary>res: 192 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L20 down c14 | 65% / 12% | **res** (R2 0.66): res in {39, 73..104, 114, 118..165, 167..200} | **tens(a,b)** (R2 0.55): a//10 in {8} -> b//10 in {0,4,5,9,10}; a//10 in {9} -> b//10 in {0,1,4,5,6}; a//10 in {10} -> b//10 in {0,1,2,5,6,7,10} | b: mod50 +2%; res: mod50 +6%, mod25 +15% | - | 84, 89, 90, 92, 94 | +0.56 / +0.81 |
| L20 down c180 | 11% / 1% | **res** (R2 0.65): res in {121..137} | unexplained (best R2 0.24) | res: mod25 +4% | - | 19, 26, 91, 97, 191 | -0.06 / -0.07 |
| L20 down c223 | 5% / 7% | **res** (R2 0.73): res in {113..118} | **res//10** (R2 0.69): (tens) res in {10..19} | res: mod25 +2% | - | 46, 64, 151, 161, 191 | -0.02 / +0.28 |
| L21 down c17 | 78% / 2% | **res** (R2 0.60): res in {4, 8, 14, 16, 18, 20, 24..30, 32..40, 42..50, 54, 56, 58, 60, 63..70, 72, 74, 76..80, 82..90, 94, 96, 98, 100, 104..110, 114, 116, 118..120, 122, 124..200} | unexplained (best R2 0.23) | - | same | 134, 136, 138, 144, 146 | +0.58 / +0.42 |
| L21 down c57 | 18% / 17% | **res** (R2 0.76): res in {65..71, 106..119} | **res** (R2 0.55): res in {6..21} | res: mod25 +3% | res: mod50 +2% | 112, 118, 158, 178, 181 | +0.24 / -0.45 |
| L21 down c92 | 10% / 0 | **res** (R2 0.91): res in {130..143} | off (on 0) | res: mod25 +4% | - | 28, 115, 116, 118, 146 | +0.41 / -0.01 |
| L21 down c99 | 21% / 0 | **res** (R2 0.70): res in {50..53, 96, 136..159, 187..200} | off (on 0) | res: mod25 +3% | - | 19, 21, 22, 23, 191 | +0.35 / +0.10 |
| L21 down c125 | 13% / 0 | **res** (R2 0.67): res in {38, 40, 47..49, 137..151, 158, 198} [coarser: res mod 100 in {37..42, 46..50}, R2 0.82] | off (on 0) | - | same | 14, 114, 148, 149, 150 | +0.31 / -0.06 |
| L21 down c133 | 15% / 1% | **res** (R2 0.70): res in {65..75, 161..192} | unexplained (best R2 0.16) | - | same | 97, 139, 169, 175, 200 | +0.35 / -0.08 |
| L21 down c279 | 7% / 0 | **res** (R2 0.64): res in {133..138, 157, 174..177, 193..196} | off (on 0) | - | same | 61, 135, 137, 175, 177 | +0.45 / +0.26 |
| L21 down c390 | 6% / 1% | **res** (R2 0.80): res in {42, 62, 102, 122, 132, 142, 152, 162, 172, 182, 192, 198, 200} | **res%100** (R2 0.53): res mod 100: no class above 0.5 (max 0.44) | - | same | 2, 65, 155, 182, 184 | +0.07 / +0.69 |
| L22 down c22 | 22% / 7% | **res** (R2 0.65): res in {3, 5, 7, 11, 21, 31, 41, 51, 61, 67, 71, 75, 77, 81, 91, 101, 107, 111, 113, 115, 117, 121, 123, 125, 127, 131, 133, 135, 137, 141, 143, 145, 147, 151, 153, 161, 163, 167, 171, 173, 175, 177, 181, 191} [coarser: res mod 100 in {1, 3, 5, 7, 11, 15, 17, 21, 27, 31, 41, 51, 61, 67, 71, 73, 75, 77, 81, 91}, R2 0.82] | **res** (R2 0.50): res in {1, 3, 5, 7, 11, 21, 31} | - | same | 29, 59, 89, 101, 159 | -0.07 / +0.02 |
| L22 down c56 | 10% / 5% | **res** (R2 0.61): res in {125..133, 135..137} | **tens(a,b)** (R2 0.61): a//10 in {6,7,8} -> b//10 in {0}; a//10 in {10} -> b//10 in {2,3,4,7} | - | same | 64, 65, 66, 68, 175 | -0.13 / +0.47 |
| L22 down c376 | 1% / 22% | **res** (R2 0.81): res in {2..15} | **res%100** (R2 0.62): res mod 100 in {6..14} | - | same | 2, 3, 4, 5, 9 | +0.53 / +0.77 |
| L23 down c12 | 2% / 0 | **res** (R2 0.62): res in {114, 124, 134} | off (on 0) | - | same | 1, 75, 108, 127, 197 | +0.28 / +nan |
| L23 down c21 | 12% / 0 | **res** (R2 0.60): res in {65, 85..90, 95, 105, 107, 115, 125, 145, 185} | off (on 0) | - | same | 85, 95, 107, 127, 145 | +0.50 / +0.48 |
| L23 down c60 | 4% / 0 | **res** (R2 0.88): res in {120, 130, 140, 150, 160, 170, 180, 190, 200} | off (on 0) | - | same | 24, 50, 76, 102, 122 | +0.45 / -0.13 |
| L23 down c123 | 4% / 0 | **res** (R2 0.54): res in {92, 101..102, 151..152} | off (on 0) | - | same | 28, 92, 101, 102, 192 | +0.38 / +0.27 |
| L23 down c269 | 6% / 0 | **res** (R2 0.80): res in {112..118} | off (on 0) | - | same | 19, 25, 26, 135, 169 | +0.19 / -0.01 |
| L23 down c310 | 4% / 0 | **res** (R2 0.66): res in {83, 93, 123, 143, 153, 163, 183, 193} [coarser: res mod 100 in {83, 93}, R2 0.82] | off (on 0) | - | same | 21, 75, 130, 152, 169 | +0.31 / +0.20 |
| L23 down c931 | 5% / 0 | **res** (R2 0.81): res in {151..158, 161..162} | off (on 0) | - | same | 44, 48, 51, 117, 169 | +0.32 / -0.18 |
| L24 down c1 | 75% / 90% | **res** (R2 0.57): res in {2..16, 18, 21..25, 29..85, 89..90, 93, 100..136, 138, 140, 143..145, 150, 154, 160, 163..166, 170, 173..176, 180, 190, 199..200} | unexplained (best R2 0.17) | - | same | 103, 104, 107, 133, 144 | +0.30 / +0.49 |
| L24 down c70 | 9% / 18% | **res** (R2 0.86): res in {35..49, 51..55} | **tens(a,b)** (R2 0.56): a//10 in {0} -> b//10 in {4,5}; a//10 in {3} -> b//10 in {0,4,5}; a//10 in {4} -> b//10 in {0,1,4,5,10}; a//10 in {5} -> b//10 in {0,1,2,6}; a//10 in {6} -> b//10 in {1,2}; a//10 in {7} -> b//10 in {3}; a//10 in {10} -> b//10 in {5,6} | - | same | 36, 38, 42, 44, 48 | +0.60 / +0.73 |
| L24 down c76 | 5% / 0 | **res** (R2 0.63): res in {116..117, 119..120, 194..200} | off (on 0) | - | same | 14, 15, 16, 55, 100 | +0.52 / +nan |
| L24 down c158 | 1% / 14% | **res** (R2 0.66): res in {6..15} | unexplained (best R2 0.44) | - | same | 19, 20, 21, 23, 29 | +0.58 / +0.75 |
| L24 down c294 | 4% / 0 | **res** (R2 0.83): res in {142..149} | off (on 0) | - | same | 74, 92, 112, 113, 171 | +0.27 / +nan |
| L24 down c553 | 5% / 0 | **res** (R2 0.76): res in {160..174, 184} | off (on 0) | - | same | 61, 63, 144, 146, 156 | +0.76 / +nan |
| L24 down c593 | 4% / 0 | **res** (R2 0.67): res in {70, 90, 110, 130, 140, 150, 170} | off (on 0) | - | same | 3, 5, 110, 150, 170 | +0.26 / -0.14 |
| L25 down c51 | 11% / 0 | **res** (R2 0.76): res in {62, 122, 152..178, 182..186} | off (on 0) | - | same | 53, 120, 127, 150, 151 | +0.71 / +0.14 |
| L25 down c55 | 20% / 4% | **res** (R2 0.74): res in {42..50, 52..68, 85..89} | unexplained (best R2 0.25) | - | same | 151, 153, 159, 161, 162 | +0.55 / +0.74 |
| L25 down c73 | 27% / 4% | **res** (R2 0.67): res in {38, 41..42, 44..51, 56..58, 76..99} | unexplained (best R2 0.48) | - | same | 46, 47, 49, 51, 63 | +0.39 / -0.03 |
| L25 down c74 | 22% / 7% | **res** (R2 0.68): res in {20, 22..25, 30, 32..34, 60..85, 165, 173..174} | unexplained (best R2 0.32) | - | same | 47, 65, 107, 138, 147 | +0.04 / -0.00 |
| L26 down c4 | 19% / 8% | **res** (R2 0.90): res in {23..29, 119..129, 131..143} | **res** (R2 0.57): res in {22..29, 34..35} | res: mod25 +3% | - | 124, 125, 126, 127, 128 | +0.82 / -0.06 |
| L26 down c95 | 17% / 0 | **res** (R2 0.78): res in {120..121, 125..141, 150..151, 160..161, 170..171, 175..176, 179..181} | off (on 0) | - | same | 63, 83, 130, 180, 181 | +0.38 / -0.29 |
| L26 down c155 | 11% / 2% | **res** (R2 0.74): res in {89..91, 109..110, 129..130, 139, 149..150, 169..170, 180, 189..194, 200} | unexplained (best R2 0.18) | - | same | 110, 117, 129, 189, 190 | +0.47 / +0.26 |
| L26 down c182 | 3% / 13% | **res** (R2 0.77): res in {20..30} | unexplained (best R2 0.48) | - | same | 34, 35, 36, 37, 38 | +0.70 / +0.82 |
| L26 down c360 | 5% / 1% | **res** (R2 0.76): res in {41..42, 141..146} [coarser: res mod 100 in {41..45}, R2 0.82] | unexplained (best R2 0.25) | - | same | 47, 48, 107, 161, 185 | +0.33 / +0.18 |
| L26 down c393 | 3% / 1% | **res** (R2 0.74): res in {134, 136..139} | unexplained (best R2 0.35) | - | same | 32, 52, 94, 97, 140 | +0.46 / +0.17 |
| L26 down c789 | 1% / 0 | **res** (R2 0.54): res in {179..180} | off (on 0) | - | same | 27, 66, 107, 127, 179 | +0.44 / -0.01 |
| L27 down c23 | 13% / 6% | **res** (R2 0.91): res in {14..17, 110..124} | **res** (R2 0.71): res in {11..17} | - | same | 107, 133, 135, 170, 173 | +0.78 / +0.01 |
| L27 down c30 | 17% / 0 | **res** (R2 0.78): res in {107..109, 116..119, 125..129, 136..139, 146..149, 156..159, 166..169, 176..179, 186..189, 196..199} | off (on 0) | - | same | 97, 98, 163, 181, 191 | +0.80 / +nan |
| L27 down c73 | 17% / 5% | **res** (R2 0.73): res in {2..15, 22..24, 32..35, 42..44, 52..54, 62..64, 73, 83, 93..94, 103..104, 113, 123, 133, 163} | unexplained (best R2 0.48) | - | same | 132, 143, 153, 182, 193 | -0.06 / -0.12 |
| L27 down c117 | 7% / 3% | **res** (R2 0.84): res in {79..86} | unexplained (best R2 0.46) | - | same | 179, 180, 181, 182, 183 | +0.74 / +0.85 |
| L27 down c142 | 5% / 3% | **res** (R2 0.67): res in {9, 19, 29, 39, 47..49, 59, 69, 79, 149} | **res** (R2 0.53): res in {9, 29} | - | same | 50, 70, 90, 99, 189 | +0.26 / +0.49 |
| L27 down c147 | 14% / 1% | **res** (R2 0.78): res in {60..67, 155..176} | unexplained (best R2 0.28) | - | same | 13, 54, 127, 178, 188 | +0.83 / +0.20 |
| L27 down c167 | 2% / 9% | **res** (R2 0.69): res in {19..27} | **res** (R2 0.65): res in {17..27} | - | same | 19, 20, 22, 25, 146 | +0.51 / +0.75 |
| L27 down c193 | 2% / 9% | **res** (R2 0.62): res in {12..20} | **res//10** (R2 0.66): (tens) res in {10..19} | - | same | 13, 14, 15, 16, 17 | +0.58 / +0.94 |
| L27 down c222 | 4% / 0 | **res** (R2 0.89): res in {131..135} | off (on 0) | - | same | 131, 132, 133, 134, 135 | +0.58 / +0.02 |
| L27 down c246 | 6% / 0 | **res** (R2 0.72): res in {73, 113..114, 133..134, 137, 153, 163, 173..174, 183} | off (on 0) | - | same | 23, 43, 53, 93, 103 | +0.47 / +nan |
| L27 down c254 | 5% / 3% | **res** (R2 0.76): res in {38..43, 139..141} [coarser: res mod 100 in {38..43}, R2 0.85] | unexplained (best R2 0.44) | - | same | 38, 39, 40, 41, 42 | +0.58 / +0.75 |
| L27 down c312 | 5% / 0 | **res** (R2 0.77): res in {58, 68, 78, 88, 118, 128, 138, 148, 158, 168, 178} | off (on 0) | - | same | 8, 38, 48, 68, 98 | -0.12 / -0.33 |
| L27 down c365 | 3% / 0 | **res** (R2 0.66): res in {97, 107, 147, 157..158} | off (on 0) | - | same | 57, 89, 114, 157, 174 | +0.35 / +0.19 |
| L27 down c417 | 2% / 0 | **res** (R2 0.84): res in {111..112} | off (on 0) | - | same | 111, 112, 152, 153, 183 | +0.38 / -0.26 |
| L27 down c599 | 3% / 0 | **res** (R2 0.69): res in {119..121, 125} | off (on 0) | - | same | 10, 48, 102, 120, 125 | +0.28 / -0.14 |
| L27 down c723 | 4% / 0 | **res** (R2 0.69): res in {83..84, 179..189} | off (on 0) | - | same | 79, 80, 87, 172, 176 | +0.81 / -0.30 |
| L27 down c835 | 5% / 1% | **res** (R2 0.88): res in {31..33, 129..134} [coarser: res mod 100 in {29, 31..34}, R2 0.86] | unexplained (best R2 0.40) | - | same | 130, 132, 133, 166, 184 | +0.44 / +0.09 |
| L27 down c849 | 5% / 2% | **res** (R2 0.78): res in {61..68} | unexplained (best R2 0.38) | - | same | 77, 82, 83, 84, 168 | +0.42 / +0.60 |
| L28 down c58 | 36% / 3% | **res** (R2 0.73): res in {15, 24..25, 35, 45, 65, 74..75, 84..85, 90, 94..95, 99, 104..105, 107..111, 114..115, 117, 119..127, 131, 134..135, 139..141, 144..145, 149..151, 155, 165, 169..171, 174..175, 179..181, 184..191, 194..195} | unexplained (best R2 0.37) | - | same | 95, 105, 115, 125, 135 | +0.74 / +0.62 |
| L28 down c73 | 3% / 3% | **res** (R2 0.75): res in {2, 8..11, 30, 50, 70, 110} | **res%100** (R2 0.52): res mod 100 in {10..11} | - | res: mod4 +2% | 10, 101, 102, 103, 110 | +0.67 / +0.93 |
| L28 down c98 | 6% / 27% | **res** (R2 0.77): res in {26..41} | **tens(a,b)** (R2 0.53): a//10 in {0} -> b//10 in {3}; a//10 in {1,6} -> b//10 in {2,3}; a//10 in {2} -> b//10 in {3,4,5}; a//10 in {3} -> b//10 in {0,3,4,5,6,9,10}; a//10 in {4} -> b//10 in {0,1,2,4,5,6}; a//10 in {5} -> b//10 in {2,6}; a//10 in {7} -> b//10 in {4}; a//10 in {8} -> b//10 in {5}; a//10 in {9,10} -> b//10 in {6} | - | same | 61, 79, 81, 130, 131 | +0.80 / +0.91 |
| L28 down c136 | 6% / 1% | **res** (R2 0.75): res in {54, 56..57, 151..158} [coarser: res mod 100 in {53..57}, R2 0.85] | unexplained (best R2 0.18) | - | same | 54, 55, 56, 77, 155 | +0.50 / +0.64 |
| L28 down c176 | 6% / 1% | **res** (R2 0.71): res in {38..39, 89, 136..141, 179, 188..189} | unexplained (best R2 0.39) | - | same | 41, 127, 129, 160, 169 | +0.47 / +0.17 |
| L28 down c203 | 1% / 4% | **res** (R2 0.68): res in {12..16} | **res** (R2 0.73): res in {13..16} | - | same | 12, 13, 14, 15, 16 | -0.35 / +0.82 |
| L28 down c211 | 5% / 6% | **res** (R2 0.60): res in {2, 6..14, 16..17, 21, 31, 51, 71, 111, 171, 191} | unexplained (best R2 0.26) | - | same | 11, 71, 111, 119, 166 | +0.57 / +0.83 |
| L28 down c218 | 8% / 0 | **res** (R2 0.93): res in {115..123, 179} | off (on 0) | - | same | 125, 126, 128, 129, 141 | +0.61 / -0.11 |
| L28 down c247 | 1% / 5% | **res** (R2 0.69): res in {5..8, 107} | **res%100** (R2 0.65): res mod 100 in {7..8} | - | same | 108, 119, 137, 147, 177 | +0.41 / +0.50 |
| L28 down c259 | 4% / 0 | **res** (R2 0.70): res in {137..144} | off (on 0) | - | same | 71, 85, 140, 141, 142 | +0.44 / +nan |
| L28 down c268 | 8% / 2% | **res** (R2 0.80): res in {49..60, 155..156} | unexplained (best R2 0.40) | - | same | 52, 55, 58, 101, 198 | +0.21 / +0.39 |
| L28 down c288 | 4% / 1% | **res** (R2 0.89): res in {15, 113..116} | **res** (R2 0.60): res in {15} | - | same | 15, 113, 114, 115, 116 | +0.94 / +0.46 |
| L28 down c294 | 3% / 0 | **res** (R2 0.79): res in {111..114} | off (on 0) | - | same | 19, 112, 113, 114, 125 | +0.71 / +nan |
| L28 down c301 | 3% / 0 | **res** (R2 0.81): res in {132, 134..136, 156} | off (on 0) | - | same | 29, 112, 135, 136, 156 | +0.34 / +nan |
| L28 down c304 | 6% / 21% | **res** (R2 0.69): res in {32..39, 133, 135..137} [coarser: res mod 100 in {33..39}, R2 0.90] | unexplained (best R2 0.46) | - | same | 46, 56, 57, 67, 96 | -0.11 / -0.07 |
| L28 down c310 | 6% / 0 | **res** (R2 0.67): res in {107, 117, 127, 147, 157..158, 160..164, 167} | off (on 0) | - | same | 107, 117, 157, 160, 164 | +0.51 / -0.01 |
| L28 down c316 | 1% / 0 | **res** (R2 0.71): res in {154} [coarser: res mod 100 in {54}, R2 0.86] | off (on 0) | - | same | 44, 53, 56, 153, 156 | +0.38 / +0.48 |
| L28 down c377 | 8% / 1% | **res** (R2 0.80): res in {34, 103..104, 132..136, 144, 153..154, 174} | unexplained (best R2 0.45) | - | same | 67, 104, 133, 134, 144 | +0.48 / +0.30 |
| L28 down c462 | 1% / 0 | **res** (R2 0.71): res in {186..188, 190..200} | off (on 0) | - | same | 85, 89, 93, 105, 178 | +0.92 / +nan |
| L28 down c517 | 4% / 6% | **res** (R2 0.51): res in {12, 20, 102, 112, 132, 191..192, 198..200} | unexplained (best R2 0.38) | - | same | 52, 61, 151, 152, 162 | +0.50 / +0.67 |
| L28 down c543 | 7% / 0 | **res** (R2 0.78): res in {56, 85..88, 136..137, 146, 156..157, 185..188} [coarser: res mod 100 in {56, 85..88}, R2 0.83] | off (on 0) | - | same | 6, 66, 144, 190, 192 | +0.60 / +0.34 |
| L28 down c593 | 2% / 0 | **res** (R2 0.81): res in {40..44} | off (on 0) | - | same | 40, 133, 152, 154, 190 | +0.08 / +0.21 |
| L28 down c666 | 5% / 0 | **res** (R2 0.85): res in {151..164} | off (on 0) | - | same | 152, 153, 156, 158, 159 | +0.49 / +nan |
| L28 down c760 | 1% / 0 | **res** (R2 0.70): res in {50, 150, 170, 180} | off (on 0) | - | same | 101, 103, 150, 170, 180 | +0.70 / +0.27 |
| L28 down c979 | 5% / 0 | **res** (R2 0.83): res in {144..152} | off (on 0) | - | same | 102, 127, 128, 159, 164 | +0.37 / -0.08 |
| L28 down c990 | 1% / 3% | **res** (R2 0.72): res in {24..28} | unexplained (best R2 0.49) | - | same | 30, 32, 34, 36, 48 | +0.34 / +0.50 |
| L28 down c1010 | 2% / 3% | **res** (R2 0.85): res in {7..9, 11..20} | **res** (R2 0.51): res in {15..17} | - | same | 8, 155, 172, 173, 198 | +0.01 / +0.26 |
| L29 down c14 | 5% / 0 | **res** (R2 0.65): res in {126, 130, 140, 142, 146..148, 160, 170, 176..178, 180, 182} | off (on 0) | - | same | 146, 147, 148, 177, 178 | +0.73 / +0.20 |
| L29 down c28 | 6% / 0 | **res** (R2 0.74): res in {106..107, 117, 146..147, 155..157, 160, 166..167, 176..177, 186..187, 196} | off (on 0) | - | same | 18, 108, 171, 182, 191 | +0.90 / +0.14 |
| L29 down c35 | 4% / 33% | **res** (R2 0.70): res in {2, 30..31, 40..41, 60..61, 70, 90} | unexplained (best R2 0.31) | - | same | 30, 31, 41, 61, 90 | +0.46 / +0.67 |
| L29 down c42 | 17% / 0 | **res** (R2 0.85): res in {30, 33, 120, 122..142} | off (on 0) | - | same | 121, 149, 153, 154, 156 | +0.54 / -0.46 |
| L29 down c51 | 14% / 2% | **res** (R2 0.88): res in {20..23, 116..130} | **res** (R2 0.62): res in {20..22} | - | same | 122, 123, 124, 125, 126 | +0.62 / -0.58 |
| L29 down c53 | 2% / 0 | **res** (R2 0.61): res in {83, 147, 180, 182..183} | off (on 0) | - | same | 119, 125, 155, 158, 164 | +0.86 / +0.78 |
| L29 down c67 | 6% / 16% | **res** (R2 0.72): res in {4, 6..28, 37, 41..42} | **res** (R2 0.59): res in {1, 4, 9, 12..26} | - | same | 9, 13, 14, 41, 42 | +0.46 / +0.58 |
| L29 down c160 | 2% / 0 | **res** (R2 0.72): res in {129, 135..136} | off (on 0) | - | same | 29, 129, 135, 136, 192 | +0.91 / +0.74 |
| L29 down c171 | 33% / 1% | **res** (R2 0.81): res in {47, 57, 59, 61, 63, 67, 79, 81, 83, 85, 87, 89, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183, 185, 187, 189, 191, 193, 195, 197} | unexplained (best R2 0.15) | res: mod2 +7% | - | 104, 106, 108, 112, 146 | +0.67 / +0.28 |
| L29 down c187 | 6% / 7% | **res** (R2 0.65): res in {30..37, 39, 53, 132..135} [coarser: res mod 100 in {31..35}, R2 0.85] | unexplained (best R2 0.30) | - | same | 113, 130, 131, 135, 136 | +0.39 / +0.70 |
| L29 down c219 | 8% / 0 | **res** (R2 0.74): res in {147..162, 170, 175} | off (on 0) | - | same | 49, 51, 52, 53, 54 | +0.73 / -0.21 |
| L29 down c232 | 21% / 0 | **res** (R2 0.83): res in {104..106, 109, 114..116, 118..119, 123..126, 129, 134..136, 143..146, 149, 154..156, 158..159, 163..166, 169, 174..176, 178..179, 184..186, 189, 194..196} | off (on 0) | - | same | 124, 125, 145, 155, 175 | +0.63 / +nan |
| L29 down c300 | 1% / 0 | **res** (R2 0.73): res in {59, 149, 159} [coarser: res mod 100 in {59}, R2 0.86] | off (on 0) | - | same | 29, 69, 109, 119, 129 | +0.24 / +0.37 |
| L29 down c326 | 1% / 0 | **res** (R2 0.63): res in {122, 131..132} | off (on 0) | - | same | 62, 71, 72, 81, 82 | +0.53 / +nan |
| L29 down c331 | 5% / 5% | **res** (R2 0.74): res in {60..67} | unexplained (best R2 0.37) | - | same | 68, 162, 164, 165, 166 | +0.40 / +0.56 |
| L29 down c415 | 3% / 8% | **res** (R2 0.59): res in {30..32, 41, 91, 131} | unexplained (best R2 0.31) | - | same | 121, 129, 130, 132, 133 | +0.50 / +0.67 |
| L29 down c426 | 1% / 0 | **res** (R2 0.80): res in {92, 152, 192} [coarser: res mod 100 in {52, 92}, R2 0.88] | off (on 0) | - | same | 92, 109, 111, 123, 192 | +0.31 / +0.42 |
| L29 down c437 | 4% / 2% | **res** (R2 0.84): res in {59..64, 160} | unexplained (best R2 0.18) | - | same | 66, 81, 158, 162, 163 | +0.47 / +0.63 |
| L29 down c448 | 3% / 4% | **res** (R2 0.62): res in {24..26, 45, 75, 125} [coarser: res mod 100 in {25..26, 75}, R2 0.81] | unexplained (best R2 0.46) | - | same | 25, 26, 72, 125, 137 | +0.65 / +0.81 |
| L29 down c480 | 7% / 1% | **res** (R2 0.90): res in {61, 71, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191} [coarser: res mod 100 in {1, 21, 31, 41, 61, 71, 91}, R2 0.82] | unexplained (best R2 0.30) | - | same | 102, 103, 130, 162, 163 | +0.69 / +0.30 |
| L29 down c549 | 7% / 0 | **res** (R2 0.88): res in {112..119, 125} | off (on 0) | - | same | 114, 115, 116, 117, 118 | +0.46 / +nan |
| L29 down c635 | 2% / 2% | **res** (R2 0.51): res in {9, 69, 89, 149} | unexplained (best R2 0.47) | - | same | 49, 59, 69, 89, 189 | +0.62 / +0.71 |
| L29 down c662 | 9% / 0 | **res** (R2 0.82): res in {102..103, 112..113, 122..123, 132..133, 142..143, 152..153, 163, 172..173, 182..183, 192..193} | off (on 0) | - | same | 125, 126, 130, 131, 150 | +0.67 / -0.09 |
| L29 down c686 | 3% / 0 | **res** (R2 0.71): res in {150..154, 156..157} | off (on 0) | - | same | 140, 142, 148, 149, 160 | +0.65 / +nan |
| L29 down c739 | 2% / 1% | **res** (R2 0.66): res in {14, 114, 144, 194} | **res** (R2 0.66): res in {14} | - | same | 14, 93, 140, 144, 145 | +0.35 / +0.49 |
| L29 down c762 | 1% / 2% | **res** (R2 0.56): res in {15, 25, 35, 65} | **res** (R2 0.55): res in {15, 25, 35} | - | same | 15, 35, 65, 104, 196 | +0.60 / +0.57 |
| L29 down c880 | 6% / 0 | **res** (R2 0.86): res in {114..120, 140} | off (on 0) | - | same | 41, 118, 119, 120, 150 | +0.49 / +0.05 |
| L29 down c904 | 4% / 0 | **res** (R2 0.80): res in {88, 98, 183..200} | off (on 0) | - | same | 186, 187, 188, 189, 190 | +0.82 / +0.52 |
| L29 down c927 | 4% / 0 | **res** (R2 0.62): res in {96..98, 186..200} | off (on 0) | - | same | 90, 91, 92, 174, 180 | +0.83 / +0.13 |
| L30 down c8 | 5% / 0 | **res** (R2 0.72): res in {118, 131, 135..139, 171, 174, 178..179} | off (on 0) | - | same | 118, 131, 137, 138, 139 | +0.83 / -0.03 |
| L30 down c17 | 4% / 0 | **res** (R2 0.76): res in {112..115, 131, 171, 181} | off (on 0) | - | same | 113, 131, 151, 171, 191 | +0.71 / +0.11 |
| L30 down c121 | 9% / 0 | **res** (R2 0.87): res in {113, 128..138, 188} | off (on 0) | - | same | 114, 127, 148, 175, 176 | +0.88 / -0.02 |
| L30 down c192 | 1% / 1% | **res** (R2 0.61): res in {29..30, 130} | unexplained (best R2 0.25) | - | same | 33, 34, 129, 131, 132 | +0.59 / +0.82 |
| L30 down c205 | 11% / 2% | **res** (R2 0.80): res in {43..49, 140..151} [coarser: res mod 100 in {42..50}, R2 0.86] | unexplained (best R2 0.45) | - | same | 44, 141, 142, 143, 144 | -0.09 / -0.34 |
| L30 down c223 | 2% / 3% | **res** (R2 0.60): res in {39..41, 140} [coarser: res mod 100 in {40}, R2 0.80] | unexplained (best R2 0.27) | - | same | 36, 38, 130, 138, 170 | +0.52 / +0.73 |
| L30 down c251 | 9% / 0 | **res** (R2 0.80): res in {61, 81..82, 111..112, 121..123, 141..142, 161..162, 181..182} | off (on 0) | - | same | 83, 118, 119, 124, 132 | +0.38 / +0.10 |
| L30 down c283 | 2% / 5% | **res** (R2 0.56): res in {21, 121, 171} [coarser: res mod 100 in {21}, R2 0.89] | unexplained (best R2 0.28) | - | same | 19, 66, 72, 74, 122 | +0.62 / +0.83 |
| L30 down c293 | 6% / 0 | **res** (R2 0.74): res in {59, 108..109, 119, 129, 158..160, 169} | off (on 0) | - | same | 57, 113, 149, 156, 157 | +0.13 / -0.03 |
| L30 down c317 | 3% / 0 | **res** (R2 0.67): res in {103..104, 143, 163, 183} | off (on 0) | - | same | 46, 47, 75, 103, 183 | +0.21 / -0.15 |
| L30 down c320 | 22% / 0 | **res** (R2 0.76): res in {102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 182, 184, 186, 188, 192, 194, 196, 198} | off (on 0) | res: mod2 +2% | - | 57, 63, 67, 73, 93 | +0.20 / +nan |
| L30 down c379 | 4% / 1% | **res** (R2 0.76): res in {72..74, 123, 133, 173..174} [coarser: res mod 100 in {73..74}, R2 0.84] | unexplained (best R2 0.20) | - | same | 71, 72, 76, 77, 171 | +0.54 / +0.47 |
| L30 down c551 | 2% / 2% | **res** (R2 0.76): res in {23..25, 124} [coarser: res mod 100 in {23..24}, R2 0.82] | **res** (R2 0.59): res in {23..24} | - | same | 23, 24, 124, 176, 192 | +0.68 / +0.93 |
| L30 down c580 | 2% / 6% | **res** (R2 0.59): res in {32, 52, 132} | unexplained (best R2 0.21) | - | same | 30, 31, 53, 176, 182 | +0.65 / +0.75 |
| L30 down c660 | 4% / 5% | **res** (R2 0.65): res in {24..27, 75, 85, 125} [coarser: res mod 100 in {24..25, 75, 85}, R2 0.84] | unexplained (best R2 0.29) | - | same | 56, 76, 127, 155, 166 | +0.68 / +0.80 |
| L30 down c661 | 2% / 0 | **res** (R2 0.63): res in {47, 147..149} | off (on 0) | - | same | 47, 74, 147, 148, 149 | +0.82 / +0.30 |
| L30 down c700 | 1% / 0 | **res** (R2 0.64): res in {152} | off (on 0) | - | same | 51, 151, 158, 162, 165 | +0.55 / -0.13 |
| L30 down c809 | 12% / 0 | **res** (R2 0.89): res in {146..172, 196, 198} | off (on 0) | - | same | 76, 77, 78, 86, 88 | -0.57 / +nan |
| L30 down c831 | 2% / 0 | **res** (R2 0.84): res in {57, 59, 157..159} [coarser: res mod 100 in {57..59}, R2 0.89] | off (on 0) | - | same | 56, 57, 59, 148, 152 | +0.16 / -0.30 |
| L30 down c882 | 2% / 3% | **res** (R2 0.77): res in {25..29, 127} | **res** (R2 0.58): res in {26..29} | - | same | 21, 25, 126, 129, 187 | +0.42 / +0.54 |
| L30 down c941 | 3% / 0 | **res** (R2 0.73): res in {64, 161..165, 167..169} | off (on 0) | - | same | 66, 67, 83, 166, 183 | +0.26 / -0.02 |
| L31 down c44 | 10% / 0 | **res** (R2 0.84): res in {99, 117..120, 122..129} | off (on 0) | - | same | 123, 124, 125, 127, 129 | +0.79 / +0.65 |
| L31 down c51 | 7% / 0 | **res** (R2 0.68): res in {38..39, 71, 111, 129, 131, 138..139, 143, 159, 161, 163, 171} | off (on 0) | - | same | 71, 138, 139, 143, 171 | +0.88 / +0.83 |
| L31 down c53 | 1% / 0 | **res** (R2 0.59): res in {97, 194..198} | off (on 0) | - | same | 97, 194, 196, 197, 198 | +0.78 / +0.74 |
| L31 down c54 | 12% / 0 | **res** (R2 0.77): res in {76, 84, 86, 106, 114..116, 126, 134..136, 146, 154..156, 164..167, 175..176, 184..186, 196} | off (on 0) | - | same | 146, 154, 156, 166, 176 | +0.90 / +0.57 |
| L31 down c79 | 2% / 0 | **res** (R2 0.68): res in {163..167, 169} | off (on 0) | - | same | 130, 137, 145, 161, 187 | +0.89 / +nan |
| L31 down c85 | 2% / 1% | **res** (R2 0.57): res in {5, 150, 155, 160, 170, 175} | **res%100** (R2 0.63): res mod 100: no class above 0.5 (max 0.47) | - | same | 5, 150, 155, 160, 175 | +0.72 / +0.81 |
| L31 down c108 | 3% / 0 | **res** (R2 0.85): res in {140..145} | off (on 0) | - | same | 123, 154, 155, 159, 166 | +0.82 / +nan |
| L31 down c125 | 1% / 0 | **res** (R2 0.51): res in {151..153} | off (on 0) | - | same | 151, 152, 153, 185, 189 | +0.71 / +nan |
| L31 down c132 | 9% / 0 | **res** (R2 0.80): res in {91, 149..162, 170..171, 175, 181, 191} | off (on 0) | - | same | 98, 133, 179, 183, 196 | +0.93 / +0.03 |
| L31 down c137 | 2% / 0 | **res** (R2 0.57): res in {109, 189, 192..199} | off (on 0) | - | same | 109, 189, 196, 197, 198 | +0.88 / +nan |
| L31 down c147 | 3% / 0 | **res** (R2 0.62): res in {174..179, 182, 184, 187..189, 194..198} | off (on 0) | - | same | 25, 74, 84, 86, 170 | +0.80 / +0.17 |
| L31 down c160 | 16% / 0 | **res** (R2 0.80): res in {74, 83, 87..88, 107..108, 113..114, 117..118, 127..128, 133, 137..138, 147..148, 157..158, 163, 167..168, 173..175, 177..179, 183..184, 187..188} | off (on 0) | - | same | 129, 149, 193, 194, 196 | +0.90 / +0.49 |
| L31 down c165 | 6% / 0 | **res** (R2 0.77): res in {146..159} | off (on 0) | - | same | 152, 154, 156, 157, 158 | +0.93 / +nan |
| L31 down c176 | 6% / 1% | **res** (R2 0.80): res in {39, 59, 79, 99, 119, 139, 149, 159, 179, 189, 192, 194..195, 197..199} | unexplained (best R2 0.32) | - | same | 125, 154, 155, 156, 175 | +0.84 / +0.67 |
| L31 down c184 | 3% / 1% | **res** (R2 0.70): res in {66..68} [coarser: res mod 100 in {66..68}, R2 0.84] | unexplained (best R2 0.19) | - | same | 82, 109, 129, 160, 174 | +0.85 / +0.86 |
| L31 down c188 | 2% / 0 | **res** (R2 0.62): res in {146, 148..149, 192, 194..199} | off (on 0) | - | same | 144, 154, 174, 177, 187 | +0.88 / +0.11 |
| L31 down c194 | 1% / 0 | **res** (R2 0.67): res in {124} | off (on 0) | - | same | 124, 171, 181, 191, 193 | +0.87 / +nan |
| L31 down c242 | 5% / 1% | **res** (R2 0.78): res in {87..89, 176..180, 185..200} | unexplained (best R2 0.34) | - | same | 89, 177, 178, 187, 189 | +0.91 / +0.71 |
| L31 down c244 | 4% / 0 | **res** (R2 0.90): res in {174..200} | off (on 0) | - | same | 136, 146, 149, 159, 163 | +0.96 / -0.06 |
| L31 down c250 | 2% / 0 | **res** (R2 0.77): res in {46..49} | off (on 0) | - | same | 46, 47, 48, 49, 184 | +0.82 / +0.87 |
| L31 down c259 | 47% / 2% | **res** (R2 0.50): res in {2..70, 75, 79..80, 82..83, 89..90, 92..95, 100, 109, 139, 155, 159..161, 163..165, 167..169, 180..183, 185, 189..190, 193..196, 200} | unexplained (best R2 0.15) | - | same | 81, 91, 138, 145, 174 | +0.39 / +0.02 |
| L31 down c271 | 1% / 0 | **res** (R2 0.63): res in {154..156} | off (on 0) | - | same | 142, 147, 154, 155, 156 | +0.83 / +nan |
| L31 down c293 | 6% / 0 | **res** (R2 0.65): res in {131..134, 136..139, 141} | off (on 0) | - | same | 131, 136, 137, 138, 139 | +0.66 / +0.13 |
| L31 down c301 | 3% / 0 | **res** (R2 0.72): res in {144..149} | off (on 0) | - | same | 47, 137, 155, 167, 175 | +0.92 / +0.06 |
| L31 down c314 | 4% / 0 | **res** (R2 0.76): res in {113..114, 133..134, 152..154} | off (on 0) | - | same | 133, 134, 152, 153, 154 | +0.91 / +0.29 |
| L31 down c315 | 5% / 0 | **res** (R2 0.77): res in {97..99, 149..150, 187..199} | off (on 0) | - | same | 99, 194, 196, 197, 198 | +0.92 / +0.78 |
| L31 down c319 | 1% / 0 | **res** (R2 0.82): res in {118} | off (on 0) | - | same | 66, 102, 184, 194, 197 | +0.80 / +nan |
| L31 down c321 | 4% / 1% | **res** (R2 0.66): res in {80..84} | unexplained (best R2 0.34) | - | same | 79, 80, 81, 83, 84 | +0.75 / +0.76 |
| L31 down c328 | 1% / 3% | **res** (R2 0.64): res in {13..15} | **res** (R2 0.70): res in {13..15} | - | same | 117, 121, 160, 162, 175 | -0.48 / +0.69 |
| L31 down c387 | 4% / 0 | **res** (R2 0.73): res in {105, 110, 135, 155, 160, 165, 170, 180, 185, 195} | off (on 0) | - | same | 105, 135, 155, 160, 165 | +0.80 / +nan |
| L31 down c423 | 1% / 0 | **res** (R2 0.72): res in {168..169} | off (on 0) | - | same | 44, 66, 114, 167, 180 | +0.89 / +0.08 |
| L31 down c426 | 17% / 0 | **res** (R2 0.81): res in {34, 36, 116..118, 124..139, 156, 164, 166..167, 174..179} | unexplained (best R2 0.19) | - | same | 126, 127, 128, 133, 134 | +0.89 / +0.57 |
| L31 down c444 | 5% / 0 | **res** (R2 0.82): res in {81, 111, 121, 131, 141, 151, 161, 171, 181} | off (on 0) | - | same | 81, 121, 161, 171, 181 | +0.90 / +0.74 |
| L31 down c455 | 1% / 0 | **res** (R2 0.68): res in {141..142} | off (on 0) | - | same | 42, 61, 69, 139, 169 | +0.71 / +nan |
| L31 down c466 | 4% / 0 | **res** (R2 0.90): res in {115..119} | off (on 0) | - | same | 115, 116, 117, 118, 119 | +0.94 / +nan |
| L31 down c474 | 13% / 13% | **res** (R2 0.76): res in {7, 68..81} | **tens(a,b)** (R2 0.53): a//10 in {0} -> b//10 in {7,8}; a//10 in {7} -> b//10 in {0,1,2,3,10}; a//10 in {8} -> b//10 in {0,1,2}; a//10 in {9,10} -> b//10 in {2} | - | same | 71, 72, 73, 74, 75 | +0.32 / +0.70 |
| L31 down c500 | 2% / 0 | **res** (R2 0.78): res in {111, 131, 151} | off (on 0) | - | same | 111, 131, 151, 163, 166 | +0.85 / -0.02 |
| L31 down c531 | 3% / 0 | **res** (R2 0.91): res in {121..124} | off (on 0) | - | same | 109, 113, 120, 133, 137 | +0.93 / +nan |
| L31 down c546 | 6% / 0 | **res** (R2 0.76): res in {85, 114..115, 135, 145, 155, 165, 175, 185, 195} | off (on 0) | - | same | 136, 138, 153, 171, 176 | +0.92 / +0.78 |
| L31 down c580 | 5% / 0 | **res** (R2 0.81): res in {123, 127..128, 132..133, 138, 142..143} | off (on 0) | - | same | 128, 132, 133, 142, 143 | +0.86 / +nan |
| L31 down c595 | 5% / 52% | **res** (R2 0.63): res in {75, 85, 115, 125, 135, 155, 175} | unexplained (best R2 0.49) | - | res: mod100 +3% | 75, 85, 125, 155, 175 | +0.75 / +0.08 |
| L31 down c612 | 3% / 0 | **res** (R2 0.63): res in {68, 162..164, 167..169, 182..184, 192, 194, 196..198} | off (on 0) | - | same | 15, 24, 115, 133, 190 | +0.89 / +0.48 |
| L31 down c652 | 6% / 1% | **res** (R2 0.87): res in {91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191} | unexplained (best R2 0.18) | - | same | 111, 131, 151, 161, 171 | +0.84 / -0.24 |
| L31 down c675 | 3% / 0 | **res** (R2 0.86): res in {112..114} | off (on 0) | - | same | 112, 113, 114, 196, 197 | +0.88 / +nan |
| L31 down c680 | 31% / 1% | **res** (R2 0.72): res in {6..31, 33..48, 51, 53..56, 58..59, 61..64, 66..68, 71, 73..74, 76..78, 80..88} | unexplained (best R2 0.15) | - | same | 61, 68, 71, 83, 84 | +0.55 / +0.57 |
| L31 down c700 | 2% / 0 | **res** (R2 0.61): res in {85, 135, 155, 175, 185} | off (on 0) | - | same | 85, 135, 170, 175, 185 | +0.86 / +0.85 |
| L31 down c748 | 10% / 1% | **res** (R2 0.72): res in {59..60, 79..80, 110, 119..120, 130, 140, 150, 155, 159..160, 165, 169..170, 175..180, 190, 195} | unexplained (best R2 0.20) | - | same | 159, 160, 170, 175, 179 | +0.87 / +0.77 |
| L31 down c813 | 9% / 0 | **res** (R2 0.79): res in {101..109, 188..189, 192..199} | off (on 0) | - | same | 103, 104, 107, 108, 109 | +0.90 / +0.03 |
| L31 down c847 | 5% / 0 | **res** (R2 0.74): res in {63, 68, 156, 162..164, 166..169, 174, 182..184} | off (on 0) | - | same | 61, 133, 173, 175, 181 | +0.92 / +0.46 |
| L31 down c859 | 1% / 0 | **res** (R2 0.72): res in {141} | off (on 0) | - | same | 55, 155, 177, 196, 197 | +0.81 / +nan |
| L31 down c867 | 4% / 0 | **res** (R2 0.76): res in {74, 137..138, 147..148, 173..174, 177} | off (on 0) | - | same | 137, 138, 147, 148, 174 | +0.91 / +0.39 |
| L31 down c899 | 25% / 0 | **res** (R2 0.88): res in {105..133} | off (on 0) | - | same | 112, 115, 116, 117, 128 | +0.92 / +nan |
| L31 down c902 | 4% / 1% | **res** (R2 0.79): res in {60..65} | unexplained (best R2 0.30) | - | same | 54, 77, 81, 109, 160 | +0.82 / +0.86 |
| L31 down c921 | 8% / 0 | **res** (R2 0.72): res in {97..98, 102, 117..124} | off (on 0) | - | same | 117, 118, 119, 120, 122 | +0.86 / +0.63 |
| L31 down c926 | 6% / 0 | **res** (R2 0.83): res in {139..146, 149} | off (on 0) | - | same | 133, 155, 187, 197, 198 | +0.92 / -0.04 |
| L31 down c927 | 3% / 0 | **res** (R2 0.81): res in {113, 115, 117} | off (on 0) | - | same | 43, 91, 133, 183, 191 | +0.81 / +nan |
| L31 down c992 | 4% / 0 | **res** (R2 0.85): res in {114..117, 155} | off (on 0) | - | same | 16, 95, 101, 119, 120 | +0.32 / +nan |
| L31 down c1023 | 8% / 4% | **res** (R2 0.76): res in {13..15, 71, 90..91, 169..174, 179..200} | unexplained (best R2 0.36) | - | same | 170, 171, 180, 190, 191 | +0.91 / +0.47 |

</details>

<details><summary>rarely on (< 0.5 % of prompts): 80 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L20 down c128 | 0 / 0 | off (on 0) | same | - | same | 83, 100, 108, 148, 180 | +nan / +nan |
| L20 o c168 (H2) | 0 / 0 | off (on 0) | same | - | same | 5, 55, 71, 105, 155 | +nan / -0.02 |
| L20 o c196 (H2) | 0 / 0 | off (on 0) | same | - | same | 46, 77, 86, 106, 153 | -0.04 / +0.07 |
| L20 o c207 (H2) | 0 / 0 | off (on 0) | same | - | same | 20, 25, 30, 99, 123 | +nan / -0.02 |
| L20 o c244 (H2) | 0 / 0 | off (on 0) | same | - | same | 32, 36, 42, 75, 100 | +nan / -0.00 |
| L20 o c259 (H2) | 0 / 0 | off (on 0) | same | - | same | 0, 54, 64, 66, 99 | +0.12 / +0.30 |
| L20 o c273 (H2) | 0 / 0 | off (on 0) | same | - | same | 48, 102, 107, 141, 147 | +nan / +nan |
| L20 o c277 (H2) | 0 / 0 | off (on 0) | same | - | same | 20, 44, 50, 55, 110 | +nan / +nan |
| L20 o c295 (H2) | 0 / 0 | off (on 0) | same | - | same | 18, 90, 120, 142, 180 | +nan / +nan |
| L20 o c312 (H2) | 0 / 0 | off (on 0) | same | - | same | 52, 106, 114, 136, 164 | +nan / +0.07 |
| L20 o c316 (H2) | 0 / 0 | off (on 0) | same | - | same | 32, 54, 64, 81, 183 | +0.08 / +0.13 |
| L20 o c356 (H2) | 0 / 0 | off (on 0) | same | - | same | 45, 50, 55, 60, 185 | +nan / +0.04 |
| L20 o c435 (H2) | 0 / 0 | off (on 0) | same | - | same | 45, 69, 70, 90, 180 | +nan / +0.04 |
| L20 down c676 | 0 / 0 | off (on 0) | same | - | same | 111, 117, 137, 154, 196 | +0.10 / +nan |
| L21 down c321 | 0 / 0 | off (on 0) | same | - | same | 3, 98, 99, 101, 149 | +nan / +0.08 |
| L21 down c819 | 0 / 0 | off (on 0) | same | - | same | 20, 110, 123, 140, 141 | +nan / +nan |
| L22 o c115 (H14) | 0 / 0 | off (on 0) | same | - | same | 102, 106, 113, 170, 182 | +nan / +nan |
| L22 o c147 (H28) | 0 / 0 | off (on 0) | same | - | same | 9, 51, 71, 84, 186 | +0.02 / +0.11 |
| L23 down c57 | 0 / 0 | off (on 0) | same | - | same | 12, 96, 116, 119, 120 | +nan / +nan |
| L23 down c113 | 0 / 0 | off (on 0) | same | - | same | 48, 64, 128, 144, 192 | +nan / +nan |
| L23 down c257 | 0 / 0 | off (on 0) | same | - | same | 95, 106, 119, 175, 182 | +0.11 / +nan |
| L23 o c299 (H22) | 0 / 0 | off (on 0) | same | - | same | 35, 48, 74, 154, 182 | +nan / +nan |
| L23 o c311 (H15) | 0 / 0 | off (on 0) | same | - | same | 33, 34, 37, 122, 133 | +0.10 / +0.05 |
| L23 o c402 (H22) | 0 / 0 | off (on 0) | same | - | same | 44, 45, 46, 47, 49 | -0.01 / -0.03 |
| L23 o c420 (H22) | 0 / 0 | off (on 0) | same | - | same | 64, 85, 124, 142, 172 | +nan / -0.05 |
| L25 down c67 | 0 / 0 | off (on 0) | same | - | same | 10, 80, 88, 150, 155 | +0.07 / +nan |
| L25 down c464 | 0 / 0 | off (on 0) | same | - | same | 4, 5, 149, 152, 155 | +0.21 / +0.24 |
| L26 down c1 | 0 / 0 | off (on 0) | same | - | same | 43, 48, 65, 143, 153 | +nan / -0.04 |
| L26 down c119 | 0 / 0 | off (on 0) | same | - | same | 54, 81, 108, 135, 168 | +0.20 / +0.01 |
| L26 down c136 | 0 / 0 | off (on 0) | same | - | same | 21, 99, 119, 151, 179 | +0.13 / +nan |
| L26 down c137 | 0 / 0 | off (on 0) | same | - | same | 17, 119, 121, 131, 157 | +0.09 / +0.12 |
| L26 down c782 | 0 / 0 | off (on 0) | same | - | same | 71, 92, 101, 109, 141 | +0.28 / +nan |
| L27 down c278 | 0 / 0 | off (on 0) | same | - | same | 109, 137, 171, 172, 177 | +0.22 / +0.31 |
| L28 down c49 | 0 / 0 | off (on 0) | same | - | same | 113, 133, 182, 185, 188 | +nan / +nan |
| L28 down c242 | 0 / 0 | off (on 0) | same | - | same | 95, 146, 163, 166, 168 | +nan / +nan |
| L28 down c744 | 0 / 0 | off (on 0) | same | - | same | 53, 55, 56, 85, 152 | +nan / +nan |
| L29 o c73 (H30) | 0 / 0 | off (on 0) | same | - | same | 66, 190, 191, 195, 196 | +nan / +0.44 |
| L29 down c301 | 0 / 0 | off (on 0) | same | - | same | 103, 106, 149, 154, 163 | -0.14 / +nan |
| L29 down c560 | 0 / 0 | off (on 0) | same | - | same | 21, 101, 110, 155, 200 | +0.31 / +0.45 |
| L29 down c905 | 0 / 0 | off (on 0) | same | - | same | 47, 68, 86, 105, 106 | +0.13 / +0.40 |
| L30 down c270 | 0 / 0 | off (on 0) | same | - | same | 103, 115, 116, 153, 193 | +nan / +nan |
| L30 down c633 | 0 / 0 | off (on 0) | same | - | same | 66, 68, 88, 166, 168 | +0.64 / +nan |
| L30 down c718 | 0 / 0 | off (on 0) | same | - | same | 52, 79, 136, 196, 197 | +nan / +nan |
| L30 down c918 | 0 / 0 | off (on 0) | same | - | same | 110, 111, 131, 153, 181 | +0.32 / +0.53 |
| L30 down c956 | 0 / 0 | off (on 0) | same | - | same | 41, 42, 71, 77, 84 | +nan / +nan |
| L31 down c17 | 0 / 0 | off (on 0) | same | - | same | 122, 125, 129, 130, 200 | +nan / +nan |
| L31 o c18 (H14) | 0 / 0 | off (on 0) | same | - | same | 147, 153, 177, 192, 195 | +0.01 / -0.03 |
| L31 down c33 | 0 / 0 | off (on 0) | same | - | same | 76, 115, 145, 155, 179 | +0.16 / +0.09 |
| L31 down c38 | 0 / 0 | off (on 0) | same | - | same | 86, 87, 88, 118, 178 | +0.25 / -0.05 |
| L31 down c42 | 0 / 0 | off (on 0) | same | - | same | 39, 158, 196, 197, 200 | +0.05 / +nan |
| L31 o c58 (H14) | 0 / 0 | off (on 0) | same | - | same | 56, 60, 62, 66, 96 | +0.14 / +nan |
| L31 down c64 | 0 / 0 | off (on 0) | same | - | same | 78, 110, 148, 168, 179 | +0.01 / -0.02 |
| L31 o c77 (H12) | 0 / 0 | off (on 0) | same | - | same | 113, 127, 162, 163, 171 | +nan / +0.03 |
| L31 down c80 | 0 / 0 | off (on 0) | same | - | same | 62, 189, 190, 191, 194 | -0.01 / +nan |
| L31 down c87 | 0 / 0 | off (on 0) | same | - | same | 36, 48, 57, 156, 157 | +0.00 / +nan |
| L31 down c122 | 0 / 0 | off (on 0) | same | - | same | 160, 161, 162, 191, 197 | +0.17 / +nan |
| L31 down c127 | 0 / 0 | off (on 0) | same | - | same | 139, 162, 177, 184, 188 | +0.89 / +nan |
| L31 down c236 | 0 / 0 | off (on 0) | same | - | same | 37, 56, 69, 97, 153 | -0.14 / +nan |
| L31 down c256 | 0 / 0 | off (on 0) | same | - | same | 49, 98, 99, 101, 102 | -0.01 / +nan |
| L31 down c273 | 0 / 0 | off (on 0) | same | - | same | 0, 80, 90, 180, 187 | +0.09 / +0.13 |
| L31 down c297 | 0 / 0 | off (on 0) | same | - | same | 73, 76, 121, 171, 178 | -0.15 / +0.04 |
| L31 down c375 | 0 / 0 | off (on 0) | same | - | same | 79, 151, 179, 188, 195 | +0.08 / +nan |
| L31 down c386 | 0 / 0 | off (on 0) | same | - | same | 53, 63, 110, 117, 122 | -0.02 / +nan |
| L31 down c406 | 0 / 0 | off (on 0) | same | - | same | 65, 138, 189, 190, 195 | +0.02 / -0.01 |
| L31 down c430 | 0 / 0 | off (on 0) | same | - | same | 99, 100, 174, 197, 199 | +0.14 / -0.08 |
| L31 down c477 | 0 / 0 | off (on 0) | same | - | same | 117, 122, 154, 166, 175 | +nan / +0.13 |
| L31 down c479 | 0 / 0 | off (on 0) | same | - | same | 80, 153, 166, 167, 200 | +0.18 / +nan |
| L31 down c482 | 0 / 0 | off (on 0) | same | - | same | 3, 4, 125, 133, 151 | -0.07 / +nan |
| L31 down c493 | 0 / 0 | off (on 0) | same | - | same | 72, 89, 139, 154, 192 | +nan / +0.11 |
| L31 down c598 | 0 / 0 | off (on 0) | same | - | same | 123, 167, 191, 194, 198 | +nan / +nan |
| L31 down c641 | 0 / 0 | off (on 0) | same | - | same | 44, 48, 49, 90, 120 | +0.42 / +nan |
| L31 down c644 | 0 / 0 | off (on 0) | same | - | same | 53, 62, 132, 136, 181 | +nan / +0.10 |
| L31 down c645 | 0 / 0 | off (on 0) | same | - | same | 26, 99, 109, 159, 169 | +nan / +0.51 |
| L31 down c655 | 0 / 0 | off (on 0) | same | - | same | 117, 177, 188, 196, 199 | +nan / +0.07 |
| L31 down c674 | 0 / 0 | off (on 0) | same | - | same | 22, 44, 69, 162, 187 | +0.05 / +nan |
| L31 down c695 | 0 / 0 | off (on 0) | same | - | same | 45, 151, 158, 160, 168 | -0.01 / +nan |
| L31 down c773 | 0 / 0 | off (on 0) | same | - | same | 106, 107, 163, 171, 175 | +nan / +0.36 |
| L31 down c805 | 0 / 0 | off (on 0) | same | - | same | 48, 89, 97, 183, 187 | +nan / +nan |
| L31 down c844 | 0 / 0 | off (on 0) | same | - | same | 54, 55, 71, 85, 88 | +0.14 / +nan |
| L31 down c895 | 0 / 0 | off (on 0) | same | - | same | 156, 180, 186, 189, 198 | +0.81 / +nan |

</details>

<details><summary>unexplained: 79 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L20 down c44 | 7% / 12% | unexplained (best R2 0.28) | unexplained (best R2 0.32) | a: mod4 +8%; b: mod4 +10%, mod2 +2% | a: mod4 +10%; b: mod4 +9% | 30, 38, 67, 93, 98 | +0.46 / +0.43 |
| L20 down c79 | 5% / 64% | unexplained (best R2 0.26) | **tens(a,b)** (R2 0.54): a//10 in {0,1} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,1,3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,2,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,4,5,6,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,6,8,9}; a//10 in {6,7,10} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1,2,3}; a//10 in {9} -> b//10 in {0,1,2,3,4,5} | - | a: mod100 +3% | 76, 97, 107, 129, 131 | +0.06 / +0.52 |
| L20 down c106 | 3% / 0 | unexplained (best R2 0.48) | off (on 0) | - | same | 51, 58, 151, 183, 184 | -0.28 / +0.12 |
| L20 o c135 (H2) | 1% / 67% | unexplained (best R2 0.06) | unexplained (best R2 0.34) | - | a: mod4 +7%; b: mod4 +2% | 40, 48, 60, 80, 128 | +0.21 / +0.24 |
| L20 down c184 | 4% / 3% | unexplained (best R2 0.22) | unexplained (best R2 0.31) | - | same | 14, 15, 16, 19, 28 | -0.23 / +0.17 |
| L20 o c252 (H2) | 4% / 12% | unexplained (best R2 0.15) | unexplained (best R2 0.45) | - | same | 40, 64, 70, 80, 140 | +0.64 / +0.57 |
| L20 down c266 | 3% / 29% | unexplained (best R2 0.48) | **tens(a,b)** (R2 0.74): a//10 in {0} -> b//10 in {3,4}; a//10 in {1} -> b//10 in {2,3,4}; a//10 in {2} -> b//10 in {2,3,4,5}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,10}; a//10 in {4} -> b//10 in {0,1,2,3,4,5,6}; a//10 in {5,6} -> b//10 in {2,3} | - | same | 31, 32, 33, 34, 39 | +0.14 / +0.62 |
| L20 o c292 (H2) | 3% / 15% | unexplained (best R2 0.16) | unexplained (best R2 0.37) | - | a: mod4 +3%; b: mod4 +2% | 77, 79, 133, 137, 143 | +0.74 / +0.43 |
| L20 o c317 (H2) | 4% / 8% | unexplained (best R2 0.15) | unexplained (best R2 0.35) | - | same | 32, 64, 80, 128, 164 | +0.65 / +0.63 |
| L20 o c328 (H2) | 5% / 19% | unexplained (best R2 0.19) | unexplained (best R2 0.40) | a: mod4 +3%; b: mod4 +3% | a: mod4 +10%; b: mod4 +6% | 18, 38, 135, 138, 150 | +0.57 / +0.40 |
| L20 o c396 (H2) | 2% / 11% | unexplained (best R2 0.48) | unexplained (best R2 0.49) | - | same | 66, 91, 97, 98, 99 | +0.06 / +0.53 |
| L20 o c401 (H2) | 1% / 6% | unexplained (best R2 0.10) | unexplained (best R2 0.36) | - | same | 54, 70, 82, 90, 150 | +0.42 / +0.11 |
| L20 o c498 (H2) | 1% / 2% | unexplained (best R2 0.07) | unexplained (best R2 0.17) | - | same | 12, 36, 60, 72, 120 | +0.43 / +0.47 |
| L20 down c529 | 1% / 0 | unexplained (best R2 0.37) | off (on 0) | - | same | 116, 123, 148, 173, 185 | -0.15 / -0.12 |
| L21 down c9 | 18% / 93% | unexplained (best R2 0.46) | **res%100** (R2 0.46): res mod 100 in {0..2, 4..98} | - | res: mod100 +3% | 95, 112, 115, 132, 155 | +0.31 / +0.30 |
| L21 down c48 | 1% / 10% | unexplained (best R2 0.11) | unexplained (best R2 0.36) | - | b: mod4 +2% | 81, 96, 98, 99, 144 | +0.17 / +0.51 |
| L21 down c51 | 1% / 61% | unexplained (best R2 0.10) | **tens(a,b)** (R2 0.68): a//10 in {0,1} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {3,4} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {5,6,7,8,9}; a//10 in {6} -> b//10 in {6,7,8,9}; a//10 in {7} -> b//10 in {1,2,7,8,9,10}; a//10 in {8} -> b//10 in {1,2,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,10}; a//10 in {10} -> b//10 in {0,1,2,10} | - | a: mod100 +2% | 120, 177, 190, 191, 200 | +0.28 / +0.40 |
| L21 down c111 | 3% / 0 | unexplained (best R2 0.44) | off (on 0) | - | same | 23, 25, 26, 30, 179 | +0.20 / +nan |
| L21 down c212 | 1% / 8% | unexplained (best R2 0.20) | **tens(a,b)** (R2 0.58): a//10 in {7} -> b//10 in {0,1,2,3}; a//10 in {8} -> b//10 in {3} | - | same | 6, 7, 70, 71, 73 | +0.41 / +0.30 |
| L22 down c55 | 1% / 32% | unexplained (best R2 0.08) | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {10}; a//10 in {2} -> b//10 in {9,10}; a//10 in {3} -> b//10 in {0,9,10}; a//10 in {4} -> b//10 in {1,2,9,10}; a//10 in {5} -> b//10 in {2,9,10}; a//10 in {6} -> b//10 in {1,2,3,10}; a//10 in {7} -> b//10 in {2,3,4}; a//10 in {8} -> b//10 in {1,2,3,4,5}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7}; a//10 in {10} -> b//10 in {0,1,2,3,5,6,7,10} | - | same | 21, 25, 31, 41, 100 | +0.22 / +0.50 |
| L22 down c232 | 1% / 76% | unexplained (best R2 0.05) | **tens(a,b)** (R2 0.52): a//10 in {0,1,2} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {0,1,2,7,8,9}; a//10 in {7} -> b//10 in {0,1,2,3,8,9}; a//10 in {8} -> b//10 in {0,1,2,3,9}; a//10 in {9} -> b//10 in {0,1,2,3,4,5}; a//10 in {10} -> b//10 in {1} | - | a: mod100 +3%, mod50 +2% | 59, 100, 110, 120, 126 | -0.14 / -0.47 |
| L22 o c246 (H3) | 3% / 6% | unexplained (best R2 0.49) | **a** (R2 0.69): a in {37, 47, 57, 67, 97} | - | same | 37, 47, 48, 49, 58 | +0.19 / +0.25 |
| L22 down c267 | 6% / 7% | unexplained (best R2 0.37) | **tens(a,b)** (R2 0.59): a//10 in {0,1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {2} | - | same | 99, 127, 168, 178, 185 | -0.20 / +0.11 |
| L22 down c297 | 1% / 19% | unexplained (best R2 0.18) | **tens(a,b)** (R2 0.61): a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {3,4,5}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {4,5,6,7}; a//10 in {9} -> b//10 in {7,8}; a//10 in {10} -> b//10 in {3,4,5,6,7,8,9} | - | same | 100, 106, 120, 126, 136 | -0.08 / +0.30 |
| L22 down c423 | 1% / 10% | unexplained (best R2 0.11) | unexplained (best R2 0.37) | - | same | 1, 6, 102, 105, 106 | -0.11 / +0.00 |
| L23 down c32 | 10% / 1% | unexplained (best R2 0.42) | unexplained (best R2 0.16) | - | same | 125, 152, 162, 173, 182 | +0.13 / +0.15 |
| L23 down c38 | 6% / 8% | unexplained (best R2 0.16) | unexplained (best R2 0.29) | - | same | 36, 72, 120, 144, 180 | +0.76 / +0.71 |
| L23 down c55 | 14% / 22% | unexplained (best R2 0.49) | **units(a,b)** (R2 0.81): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} | a: mod2 +3%; b: mod2 +3% | a: mod2 +13%; b: mod2 +13%; res: mod2 +5% | 57, 63, 87, 89, 97 | +0.71 / +0.89 |
| L23 o c152 (H22) | 4% / 9% | unexplained (best R2 0.50) | unexplained (best R2 0.46) | - | same | 35, 55, 65, 85, 95 | +0.52 / +0.57 |
| L23 down c196 | 1% / 8% | unexplained (best R2 0.28) | **res%100** (R2 0.76): res mod 100: no class above 0.5 (max 0.46) | - | same | 6, 14, 164, 186, 194 | -0.06 / -0.20 |
| L23 o c234 (H22) | 2% / 5% | unexplained (best R2 0.38) | same | - | same | 34, 35, 36, 55, 65 | +0.49 / +0.58 |
| L23 o c280 (H7) | 1% / 3% | unexplained (best R2 0.41) | **a** (R2 0.56): a in {8, 88} | - | same | 150, 151, 153, 193, 195 | +0.03 / +0.31 |
| L23 o c394 (H22) | 6% / 9% | unexplained (best R2 0.35) | unexplained (best R2 0.34) | - | same | 9, 36, 68, 84, 89 | +0.37 / +0.16 |
| L23 o c456 (H22) | 1% / 3% | unexplained (best R2 0.21) | unexplained (best R2 0.39) | - | same | 54, 55, 56, 107, 164 | +0.23 / +0.27 |
| L23 down c465 | 3% / 29% | unexplained (best R2 0.43) | **tens(a,b)** (R2 0.60): a//10 in {0} -> b//10 in {0,1,2,3,4,5,9,10}; a//10 in {1} -> b//10 in {0,1,2,3,10}; a//10 in {2} -> b//10 in {1,2,3}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3,4}; a//10 in {5} -> b//10 in {4,5}; a//10 in {6} -> b//10 in {5,6,7}; a//10 in {7} -> b//10 in {6,7,8}; a//10 in {8} -> b//10 in {7,8,9}; a//10 in {9} -> b//10 in {8,9,10}; a//10 in {10} -> b//10 in {9,10} | - | same | 20, 24, 25, 102, 104 | +0.57 / +0.79 |
| L24 down c65 | 3% / 8% | unexplained (best R2 0.23) | unexplained (best R2 0.25) | - | same | 75, 100, 125, 150, 200 | +0.42 / +0.40 |
| L25 down c145 | 2% / 2% | unexplained (best R2 0.36) | **units(a,b)** (R2 0.58): a%10 in {0} -> b%10 in {0} | - | same | 68, 71, 72, 82, 180 | +0.38 / +0.62 |
| L25 down c1005 | 3% / 12% | unexplained (best R2 0.33) | unexplained (best R2 0.48) | - | same | 0, 1, 2, 128, 144 | +0.12 / +0.81 |
| L26 down c72 | 1% / 7% | unexplained (best R2 0.11) | unexplained (best R2 0.42) | - | same | 23, 39, 48, 64, 199 | +0.57 / +0.59 |
| L26 down c75 | 1% / 27% | unexplained (best R2 0.23) | **tens(a,b)** (R2 0.62): a//10 in {0} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,9}; a//10 in {3,4,5,6,7,8,10} -> b//10 in {0}; a//10 in {9} -> b//10 in {0,1,2,3,4,5} | - | same | 6, 62, 82, 94, 180 | +0.21 / +0.41 |
| L26 down c154 | 1% / 37% | unexplained (best R2 0.09) | unexplained (best R2 0.37) | - | a: mod4 +5%; b: mod4 +5%; res: mod4 +5% | 14, 26, 58, 94, 107 | +0.42 / +0.32 |
| L26 down c187 | 2% / 4% | unexplained (best R2 0.09) | unexplained (best R2 0.22) | - | same | 6, 42, 66, 76, 98 | +0.28 / +0.38 |
| L26 down c189 | 4% / 4% | unexplained (best R2 0.30) | unexplained (best R2 0.33) | - | same | 71, 107, 172, 183, 192 | +0.53 / +0.46 |
| L27 down c188 | 2% / 4% | unexplained (best R2 0.37) | **units(a,b)** (R2 0.87): a%10 in {0,5} -> b%10 in {0,5} | - | same | 20, 25, 30, 35, 40 | +0.57 / +0.77 |
| L27 down c381 | 2% / 6% | unexplained (best R2 0.10) | unexplained (best R2 0.29) | - | same | 16, 40, 56, 64, 70 | +0.50 / +0.65 |
| L28 down c11 | 2% / 98% | unexplained (best R2 0.30) | always | - | same | 127, 130, 155, 175, 180 | +0.08 / +0.25 |
| L28 down c309 | 1% / 0 | unexplained (best R2 0.46) | off (on 0) | - | same | 76, 129, 142, 151, 152 | +0.76 / +nan |
| L28 down c588 | 1% / 3% | unexplained (best R2 0.40) | **res%100** (R2 0.50): res mod 100: no class above 0.5 (max 0.48) | - | same | 30, 75, 81, 103, 104 | +0.44 / +0.75 |
| L29 down c6 | 95% / 98% | unexplained (best R2 0.20) | always | - | same | 72, 88, 90, 96, 172 | +0.46 / +0.28 |
| L29 down c11 | 74% / 2% | unexplained (best R2 0.43) | unexplained (best R2 0.17) | a: mod100 +3% | - | 0, 118, 173, 174, 175 | +0.39 / +0.31 |
| L29 down c12 | 6% / 92% | unexplained (best R2 0.23) | unexplained (best R2 0.29) | - | same | 31, 32, 64, 69, 77 | +0.38 / +0.07 |
| L29 down c94 | 1% / 2% | unexplained (best R2 0.44) | **res%100** (R2 0.55): res mod 100 in {7} | - | same | 3, 9, 17, 109, 117 | +0.67 / +0.79 |
| L29 down c115 | 1% / 25% | unexplained (best R2 0.36) | **tens(a,b)** (R2 0.53): a//10 in {0} -> b//10 in {0}; a//10 in {1} -> b//10 in {1}; a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9,10}; a//10 in {8} -> b//10 in {6,7,8,9,10}; a//10 in {9} -> b//10 in {7,9,10}; a//10 in {10} -> b//10 in {6,7,8,9} | - | res: mod100 +2% | 1, 3, 4, 6, 167 | +0.45 / +0.46 |
| L29 down c134 | 1% / 2% | unexplained (best R2 0.23) | unexplained (best R2 0.49) | - | same | 8, 80, 81, 140, 142 | +0.69 / +0.84 |
| L29 down c159 | 11% / 19% | unexplained (best R2 0.41) | unexplained (best R2 0.27) | a: mod2 +2%; b: mod4 +7%, mod2 +2%; res: mod4 +4% | a: mod4 +10%, mod2 +5%; b: mod4 +11%, mod2 +5%; res: mod4 +5% | 98, 99, 118, 146, 173 | +0.79 / +0.71 |
| L29 down c325 | 5% / 5% | unexplained (best R2 0.27) | unexplained (best R2 0.21) | - | same | 42, 45, 54, 90, 180 | +0.59 / +0.64 |
| L29 down c584 | 11% / 11% | unexplained (best R2 0.50) | **tens(a,b)** (R2 0.54): a//10 in {0} -> b//10 in {0,1,2,10}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {2}; a//10 in {9} -> b//10 in {9,10} | - | same | 1, 42, 44, 111, 135 | +0.34 / +0.29 |
| L29 down c871 | 1% / 0 | unexplained (best R2 0.48) | off (on 0) | - | same | 69, 87, 88, 89, 105 | +0.59 / +0.63 |
| L30 down c21 | 88% / 1% | unexplained (best R2 0.27) | unexplained (best R2 0.37) | a: mod100 +4%, mod50 +3%, mod25 +2%; b: mod100 +3%, mod25 +3% | - | 68, 69, 70, 73, 134 | +0.46 / +0.05 |
| L30 down c47 | 63% / 71% | unexplained (best R2 0.28) | unexplained (best R2 0.35) | - | same | 53, 63, 123, 137, 167 | +0.18 / +0.27 |
| L30 down c107 | 23% / 1% | unexplained (best R2 0.28) | unexplained (best R2 0.05) | - | same | 8, 20, 72, 80, 90 | +0.06 / +0.26 |
| L30 down c381 | 4% / 99% | unexplained (best R2 0.41) | always | - | same | 29, 40, 73, 160, 162 | +0.36 / -0.06 |
| L31 down c15 | 7% / 90% | unexplained (best R2 0.19) | unexplained (best R2 0.27) | - | same | 71, 72, 73, 129, 159 | +0.29 / +0.13 |
| L31 down c45 | 5% / 70% | unexplained (best R2 0.45) | **tens(a,b)** (R2 0.52): a//10 in {0,9} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1,2} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {4} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {7,8,9}; a//10 in {7} -> b//10 in {0,1,2,3,4,7,8,9,10}; a//10 in {8} -> b//10 in {1,2,3,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,10} | - | res: mod100 +2% | 155, 172, 183, 184, 186 | +0.64 / +0.40 |
| L31 down c48 | 2% / 0 | unexplained (best R2 0.42) | off (on 0) | - | same | 124, 125, 160, 167, 169 | +0.67 / +nan |
| L31 down c66 | 1% / 0 | unexplained (best R2 0.36) | off (on 0) | - | same | 88, 96, 97, 99, 129 | +0.28 / +0.09 |
| L31 down c100 | 2% / 10% | unexplained (best R2 0.20) | unexplained (best R2 0.30) | - | same | 196, 197, 198, 199, 200 | -0.01 / -0.17 |
| L31 down c103 | 2% / 3% | unexplained (best R2 0.47) | **res%100** (R2 0.49): res mod 100 in {9} | - | same | 8, 9, 11, 89, 189 | +0.46 / +0.30 |
| L31 down c104 | 48% / 1% | unexplained (best R2 0.44) | unexplained (best R2 0.21) | - | same | 157, 159, 161, 163, 175 | +0.49 / +0.58 |
| L31 down c106 | 1% / 0 | unexplained (best R2 0.43) | off (on 0) | - | same | 117, 121, 122, 131, 137 | +0.63 / +nan |
| L31 down c181 | 87% / 92% | unexplained (best R2 0.13) | unexplained (best R2 0.25) | - | same | 22, 26, 98, 103, 112 | +0.44 / +0.14 |
| L31 down c306 | 6% / 19% | unexplained (best R2 0.33) | unexplained (best R2 0.42) | - | same | 60, 80, 88, 175, 200 | +0.17 / -0.15 |
| L31 down c365 | 12% / 1% | unexplained (best R2 0.26) | unexplained (best R2 0.39) | - | same | 102, 137, 138, 139, 183 | +0.08 / +0.47 |
| L31 down c465 | 2% / 0 | unexplained (best R2 0.19) | off (on 0) | - | same | 43, 44, 74, 183, 190 | +0.27 / +nan |
| L31 down c494 | 2% / 7% | unexplained (best R2 0.47) | **a** (R2 0.66): a in {1..6} | - | same | 11, 34, 120, 134, 141 | +0.33 / +0.08 |
| L31 down c533 | 5% / 68% | unexplained (best R2 0.37) | unexplained (best R2 0.47) | - | same | 107, 112, 142, 147, 175 | +0.53 / +0.45 |
| L31 down c587 | 35% / 1% | unexplained (best R2 0.47) | unexplained (best R2 0.13) | - | same | 189, 191, 192, 196, 197 | +0.58 / +0.15 |
| L31 down c622 | 11% / 79% | unexplained (best R2 0.38) | **tens(a,b)** (R2 0.52): a//10 in {0,1} -> b//10 in {0,1,2,3}; a//10 in {2} -> b//10 in {1,2,3,4,5,6}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,7,8,9}; a//10 in {4} -> b//10 in {1,2,3,4,5,6,7,8,9}; a//10 in {5,6,7,8,9} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,4,5,6,7,8,9,10} | - | a: mod100 +5%, mod50 +5%; b: mod50 +3%, mod25 +3%; res: mod100 +7%, mod50 +3% | 42, 46, 47, 71, 79 | +0.07 / -0.30 |
| L31 down c754 | 26% / 62% | unexplained (best R2 0.46) | unexplained (best R2 0.39) | - | same | 108, 109, 138, 168, 182 | +0.51 / +0.48 |

</details>

<details><summary>res//10: 63 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L20 down c20 | 54% / 48% | **res//10** (R2 0.81): (tens) res in {2..50, 99..150, 198..200} [coarser: res mod 100 in {0..50, 98..99}, R2 0.99] | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {0,1,2,3}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {0,1,2,3,9,10}; a//10 in {3} -> b//10 in {0,1,2,3,4,6,10}; a//10 in {4} -> b//10 in {0,1,2,3,4,6}; a//10 in {5} -> b//10 in {1,2,3,4,5}; a//10 in {6} -> b//10 in {2,3,4,5,6}; a//10 in {7} -> b//10 in {3,4,5,6,9}; a//10 in {8} -> b//10 in {4,5,6,7,8,9}; a//10 in {9} -> b//10 in {6,7,8,9,10}; a//10 in {10} -> b//10 in {0,5,6,7,8,9} | res: mod100 +8% | res: mod100 +4% | 61, 65, 71, 72, 79 | +0.81 / +0.87 |
| L20 down c31 | 29% / 25% | **res//10** (R2 0.79): (tens) res in {2..13, 89..117, 188..200} [coarser: res mod 100 in {0..15, 89..99}, R2 0.98] | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {0,1,9,10}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {4,5}; a//10 in {6} -> b//10 in {5,6,7,8}; a//10 in {7} -> b//10 in {6,7,8,9}; a//10 in {8} -> b//10 in {7,8,9,10}; a//10 in {9,10} -> b//10 in {0,8,9,10} | res: mod100 +4%, mod25 +6% | res: mod100 +3%, mod50 +3%, mod25 +4% | 96, 99, 104, 105, 109 | +0.57 / -0.08 |
| L20 down c101 | 14% / 0 | **res//10** (R2 0.77): (tens) res in {148..188, 190, 192..198, 200} | off (on 0) | - | same | 182, 183, 184, 187, 190 | +0.21 / +nan |
| L20 down c153 | 19% / 0 | **res//10** (R2 0.91): (tens) res in {140..196, 198} | off (on 0) | - | same | 151, 152, 153, 177, 183 | +0.51 / -0.39 |
| L21 down c0 | 33% / 26% | **res//10** (R2 0.84): (tens) res in {8..40, 108..140} [coarser: res mod 100 in {8..40}, R2 1.00] | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {2}; a//10 in {1} -> b//10 in {0,10}; a//10 in {2} -> b//10 in {0,1,9,10}; a//10 in {3} -> b//10 in {0,1,2,10}; a//10 in {4} -> b//10 in {1,2,3}; a//10 in {5} -> b//10 in {2,3,4}; a//10 in {6} -> b//10 in {3,4,5}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {5,6,7}; a//10 in {9,10} -> b//10 in {6,7,8} | res: mod100 +15%, mod50 +5% | a: mod50 +2%; res: mod100 +7%, mod50 +6% | 47, 48, 49, 59, 79 | +0.60 / +0.78 |
| L21 down c36 | 35% / 7% | **res//10** (R2 0.79): (tens) res in {60..91, 153..195, 197} [coarser: res mod 100 in {56..95}, R2 0.92] | **tens(a,b)** (R2 0.65): a//10 in {8} -> b//10 in {0,1,2,10}; a//10 in {9} -> b//10 in {1,2}; a//10 in {10} -> b//10 in {1,2,3,4} | res: mod100 +3% | - | 35, 36, 113, 121, 131 | +0.62 / +0.54 |
| L21 down c49 | 16% / 2% | **res//10** (R2 0.70): (tens) res in {40, 44, 132..156, 158, 160} | unexplained (best R2 0.27) | res: mod25 +2% | - | 28, 29, 30, 31, 35 | +0.07 / -0.22 |
| L21 down c105 | 3% / 33% | **res//10** (R2 0.68): (tens) res in {2..25} | **tens(a,b)** (R2 0.72): a//10 in {0} -> b//10 in {0,1,2}; a//10 in {1} -> b//10 in {0,1,2,10}; a//10 in {2} -> b//10 in {0,1,2,3}; a//10 in {3} -> b//10 in {1,2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9}; a//10 in {9,10} -> b//10 in {7,8,9,10} | - | res: mod100 +4%, mod50 +2% | 35, 39, 40, 45, 48 | +0.61 / +0.93 |
| L21 down c206 | 6% / 0 | **res//10** (R2 0.75): (tens) res in {17..18, 22..39} | off (on 0) | - | same | 111, 114, 123, 124, 131 | +0.55 / +0.57 |
| L22 down c29 | 45% / 1% | **res//10** (R2 0.89): (tens) res in {108..200} | **tens(a,b)** (R2 0.51): a//10 in {1,2,3,4,5} -> b//10 in {10} | a: mod100 +4%, mod50 +2%; b: mod100 +5%, mod50 +2% | - | 36, 37, 42, 43, 57 | +0.79 / -0.33 |
| L22 down c107 | 13% / 8% | **res//10** (R2 0.76): (tens) res in {2..49, 200} | **tens(a,b)** (R2 0.73): a//10 in {0,1,2} -> b//10 in {0,1,2} | - | same | 44, 58, 64, 148, 152 | +0.44 / +0.82 |
| L22 down c404 | 4% / 4% | **res//10** (R2 0.58): (tens) res in {180..200} | unexplained (best R2 0.39) | - | same | 30, 60, 70, 168, 175 | +0.54 / +0.28 |
| L23 down c7 | 39% / 14% | **res//10** (R2 0.75): (tens) res in {7..8, 15, 18, 20..88} | unexplained (best R2 0.41) | - | same | 103, 107, 133, 139, 174 | +0.92 / +0.84 |
| L23 down c10 | 19% / 97% | **res//10** (R2 0.63): (tens) res in {2..4, 6, 8..59, 61..63} | always | - | same | 74, 94, 114, 115, 160 | +0.84 / +0.33 |
| L23 down c18 | 27% / 18% | **res//10** (R2 0.75): (tens) res in {73..102, 190..191} | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {7,8,9,10}; a//10 in {7} -> b//10 in {8}; a//10 in {8} -> b//10 in {0,1,2,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,9,10}; a//10 in {10} -> b//10 in {0,1,2} | - | same | 80, 83, 86, 88, 90 | +0.84 / +0.88 |
| L23 down c20 | 18% / 12% | **res//10** (R2 0.80): (tens) res in {25..36, 121..139} [coarser: res mod 100 in {21..37}, R2 0.91] | **res** (R2 0.56): res in {23..36} | res: mod100 +2%, mod50 +2%, mod25 +3% | - | 107, 114, 117, 118, 119 | +0.75 / +0.68 |
| L23 down c29 | 58% / 3% | **res//10** (R2 0.85): (tens) res in {92..199} | unexplained (best R2 0.45) | res: mod100 +2% | - | 102, 104, 108, 115, 117 | +0.91 / +0.51 |
| L23 down c41 | 15% / 30% | **res//10** (R2 0.79): (tens) res in {6..59} | **res//10** (R2 0.53): (tens) res in {2..31} | - | res: mod100 +2% | 116, 127, 130, 132, 135 | +0.72 / +0.78 |
| L23 down c58 | 15% / 25% | **res//10** (R2 0.83): (tens) res in {3..56, 200} | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {0,1,2,3}; a//10 in {1,2,3} -> b//10 in {0,1,2,3,4}; a//10 in {4} -> b//10 in {0,1,2}; a//10 in {5} -> b//10 in {2} | - | same | 57, 58, 83, 97, 98 | +0.52 / +0.89 |
| L23 down c130 | 28% / 0 | **res//10** (R2 0.83): (tens) res in {128..200} | off (on 0) | - | same | 8, 161, 181, 182, 191 | +0.49 / -0.24 |
| L24 down c8 | 25% / 28% | **res//10** (R2 0.74): (tens) res in {52..85, 169} | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {6,7,8}; a//10 in {2,3,4} -> b//10 in {7,8}; a//10 in {5} -> b//10 in {8,9}; a//10 in {6} -> b//10 in {0,1,9,10}; a//10 in {7} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1,2,3}; a//10 in {9} -> b//10 in {1,2,3,4}; a//10 in {10} -> b//10 in {2,3,4} | res: mod100 +2% | - | 23, 31, 33, 36, 37 | +0.88 / +0.93 |
| L24 down c31 | 19% / 15% | **res//10** (R2 0.82): (tens) res in {8..20, 102..119, 200} [coarser: res mod 100 in {0, 7..19}, R2 0.86] | **res** (R2 0.69): res in {4..19} | - | res: mod25 +2% | 41, 97, 143, 161, 181 | +0.68 / +0.76 |
| L24 down c34 | 28% / 0 | **res//10** (R2 0.89): (tens) res in {126..200} | off (on 0) | a: mod100 +3%; b: mod100 +3% | - | 51, 57, 59, 103, 105 | +0.95 / -0.09 |
| L24 down c112 | 16% / 0 | **res//10** (R2 0.84): (tens) res in {120..142} | off (on 0) | - | same | 23, 25, 26, 28, 29 | +0.67 / -0.21 |
| L24 down c350 | 9% / 13% | **res//10** (R2 0.71): (tens) res in {2..43, 200} | unexplained (best R2 0.49) | - | same | 0, 15, 16, 17, 20 | +0.52 / +0.47 |
| L25 down c33 | 42% / 0 | **res//10** (R2 0.94): (tens) res in {110..200} | off (on 0) | b: mod100 +2% | - | 9, 27, 28, 29, 31 | +0.90 / -0.30 |
| L25 down c43 | 7% / 17% | **res//10** (R2 0.68): (tens) res in {4..33, 35} | **a** (R2 0.53): a in {3, 5..16} | - | same | 54, 74, 114, 148, 166 | +0.40 / +0.75 |
| L25 down c92 | 17% / 0 | **res//10** (R2 0.81): (tens) res in {111..129, 135} | off (on 0) | - | same | 18, 19, 20, 21, 22 | +0.49 / -0.48 |
| L25 down c200 | 17% / 0 | **res//10** (R2 0.81): (tens) res in {137..170, 185..188, 195, 198} | off (on 0) | - | same | 147, 148, 153, 155, 157 | +0.87 / +0.05 |
| L25 down c220 | 5% / 30% | **res//10** (R2 0.68): (tens) res in {2, 4..28, 30} | **tens(a,b)** (R2 0.64): a//10 in {0,1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {0,1,2,3}; a//10 in {3} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {7,8,9} | - | res: mod100 +2% | 4, 5, 6, 7, 8 | +0.64 / +0.81 |
| L25 down c256 | 7% / 0 | **res//10** (R2 0.83): (tens) res in {165..200} | off (on 0) | - | same | 176, 178, 179, 180, 181 | +0.86 / -0.06 |
| L26 down c85 | 10% / 14% | **res//10** (R2 0.73): (tens) res in {2..45} | **tens(a,b)** (R2 0.71): a//10 in {0} -> b//10 in {0,1,2,3,4}; a//10 in {1,2} -> b//10 in {0,1,2}; a//10 in {3,4} -> b//10 in {0} | - | same | 86, 112, 135, 136, 196 | +0.67 / +0.73 |
| L26 down c86 | 8% / 0 | **res//10** (R2 0.77): (tens) res in {101..109} | off (on 0) | - | same | 102, 103, 104, 105, 106 | +0.77 / -0.15 |
| L26 down c605 | 15% / 0 | **res//10** (R2 0.83): (tens) res in {147, 149..200} | off (on 0) | - | same | 53, 61, 63, 71, 77 | +0.92 / +0.09 |
| L26 down c641 | 1% / 0 | **res//10** (R2 0.51): (tens) res in {181..182, 186, 192, 194..200} | off (on 0) | - | same | 181, 195, 196, 197, 198 | +0.73 / +0.10 |
| L27 down c6 | 61% / 2% | **res//10** (R2 0.87): (tens) res in {90..199} | unexplained (best R2 0.39) | a: mod100 +2%, mod50 +3%; b: mod100 +3%, mod50 +2%; res: mod100 +3% | - | 101, 102, 103, 104, 105 | +0.94 / +0.66 |
| L27 down c25 | 17% / 0 | **res//10** (R2 0.83): (tens) res in {130, 132..160} | off (on 0) | - | same | 41, 42, 43, 44, 46 | +0.89 / +nan |
| L27 down c33 | 14% / 13% | **res//10** (R2 0.59): (tens) res in {91..99, 186, 188, 191..197} [coarser: res mod 100 in {91..98}, R2 0.84] | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {9,10}; a//10 in {1,2} -> b//10 in {9}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4} | - | same | 149, 150, 162, 182, 186 | +0.80 / +0.95 |
| L27 down c93 | 12% / 0 | **res//10** (R2 0.88): (tens) res in {150..198} | off (on 0) | - | same | 176, 178, 179, 190, 191 | +0.69 / -0.08 |
| L27 down c133 | 17% / 3% | **res//10** (R2 0.77): (tens) res in {39, 41, 45..71} | unexplained (best R2 0.41) | - | same | 48, 60, 61, 62, 64 | +0.52 / +0.55 |
| L27 down c140 | 11% / 1% | **res//10** (R2 0.83): (tens) res in {70..79, 87, 170..178} [coarser: res mod 100 in {70..79, 87}, R2 0.98] | unexplained (best R2 0.34) | - | same | 77, 78, 87, 174, 178 | +0.55 / +0.52 |
| L28 down c10 | 10% / 7% | **res//10** (R2 0.86): (tens) res in {11..17, 110..119} [coarser: res mod 100 in {10..19}, R2 0.95] | **res** (R2 0.78): res in {11..17} | - | same | 99, 101, 114, 180, 181 | -0.06 / -0.50 |
| L28 down c12 | 27% / 8% | **res//10** (R2 0.86): (tens) res in {9..17, 100..129, 199..200} | **res** (R2 0.81): res in {9..17} | - | same | 124, 125, 127, 128, 139 | +0.34 / +0.08 |
| L28 down c70 | 11% / 24% | **res//10** (R2 0.75): (tens) res in {36..57} | **tens(a,b)** (R2 0.56): a//10 in {0} -> b//10 in {4,5}; a//10 in {1,3,10} -> b//10 in {5}; a//10 in {4} -> b//10 in {0,4,5,9,10}; a//10 in {5} -> b//10 in {0,1,5,6,9,10}; a//10 in {6} -> b//10 in {0,1,2}; a//10 in {7} -> b//10 in {1,2,3}; a//10 in {8} -> b//10 in {3,4}; a//10 in {9} -> b//10 in {4} | - | same | 23, 26, 28, 79, 146 | +0.84 / +0.91 |
| L28 down c91 | 11% / 1% | **res//10** (R2 0.79): (tens) res in {17, 157..200} | **res** (R2 0.63): res in {17} | - | same | 170, 171, 173, 175, 176 | +0.44 / +0.13 |
| L28 down c145 | 28% / 0 | **res//10** (R2 0.76): (tens) res in {121..170, 172..175, 178, 180..186, 194, 198, 200} | off (on 0) | - | same | 102, 104, 105, 107, 110 | +0.02 / +0.14 |
| L28 down c312 | 8% / 1% | **res//10** (R2 0.80): (tens) res in {70..79, 171, 173, 175} | **tens(a,b)** (R2 0.58): a//10 in {7} -> b//10 in {0} | - | same | 5, 66, 77, 167, 189 | +0.16 / +0.23 |
| L28 down c338 | 10% / 0 | **res//10** (R2 0.78): (tens) res in {100..109, 194..200} | off (on 0) | - | same | 60, 140, 151, 152, 162 | +0.33 / +nan |
| L28 down c638 | 1% / 0 | **res//10** (R2 0.89): (tens) res in {190..200} | off (on 0) | - | same | 193, 194, 195, 196, 197 | +0.92 / +nan |
| L29 down c144 | 8% / 11% | **res//10** (R2 0.84): (tens) res in {70..79} | unexplained (best R2 0.49) | - | same | 71, 72, 73, 74, 76 | +0.79 / +0.79 |
| L29 down c435 | 10% / 9% | **res//10** (R2 0.80): (tens) res in {61..69, 161..169} [coarser: res mod 100 in {61..69}, R2 1.00] | unexplained (best R2 0.40) | - | same | 63, 64, 65, 67, 68 | -0.06 / +0.77 |
| L29 down c502 | 5% / 3% | **res//10** (R2 0.78): (tens) res in {40..49} | unexplained (best R2 0.37) | - | same | 48, 57, 58, 141, 142 | -0.03 / +0.10 |
| L29 down c893 | 11% / 5% | **res//10** (R2 0.79): (tens) res in {30..31, 35..58} | unexplained (best R2 0.41) | - | same | 63, 65, 66, 149, 152 | +0.14 / +0.26 |
| L30 down c159 | 9% / 4% | **res//10** (R2 0.82): (tens) res in {23..29, 120..129} [coarser: res mod 100 in {22..29}, R2 0.93] | **res** (R2 0.71): res in {23..29} | - | same | 17, 40, 68, 70, 80 | -0.45 / -0.64 |
| L30 down c322 | 9% / 1% | **res//10** (R2 0.81): (tens) res in {70..79, 170..173} | unexplained (best R2 0.22) | - | same | 170, 171, 172, 173, 174 | -0.21 / -0.22 |
| L30 down c364 | 8% / 1% | **res//10** (R2 0.77): (tens) res in {81..89, 181..186} [coarser: res mod 100 in {81..87}, R2 0.95] | unexplained (best R2 0.41) | - | same | 81, 82, 86, 88, 186 | -0.23 / -0.23 |
| L30 down c399 | 11% / 0 | **res//10** (R2 0.73): (tens) res in {101..109, 198..200} | off (on 0) | - | same | 103, 105, 106, 107, 108 | -0.82 / -0.04 |
| L30 down c465 | 7% / 1% | **res//10** (R2 0.64): (tens) res in {130..140} | unexplained (best R2 0.17) | - | same | 35, 36, 37, 129, 131 | -0.15 / -0.46 |
| L31 down c27 | 13% / 1% | **res//10** (R2 0.60): (tens) res in {142..143, 147..154, 157..164, 167..199} | unexplained (best R2 0.15) | - | same | 122, 143, 148, 173, 182 | +0.67 / -0.20 |
| L31 down c201 | 13% / 0 | **res//10** (R2 0.81): (tens) res in {147, 151..188, 191, 193..196} | off (on 0) | - | same | 162, 163, 173, 175, 178 | +0.90 / +0.01 |
| L31 down c213 | 18% / 16% | **res//10** (R2 0.60): (tens) res in {3..6, 8, 18, 26..29, 41..42, 44..68} | **tens(a,b)** (R2 0.63): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {1,2,4,5,6}; a//10 in {5,6} -> b//10 in {0}; a//10 in {10} -> b//10 in {10} | - | a: mod50 +3%, mod25 +3%, mod20 +2% | 76, 77, 82, 83, 169 | +0.51 / +0.46 |
| L31 down c307 | 11% / 44% | **res//10** (R2 0.64): (tens) res in {2..19, 21..41} | **tens(a,b)** (R2 0.67): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,3,4,5,6,7,8,9}; a//10 in {2} -> b//10 in {0,1,2,3,4,5,6}; a//10 in {3} -> b//10 in {0,3,4,5,6}; a//10 in {4} -> b//10 in {0,4,5,6}; a//10 in {5} -> b//10 in {0,1,6}; a//10 in {6} -> b//10 in {0,7} | - | a: mod100 +9%, mod50 +8%, mod25 +7%, mod20 +6%, mod10 +4%; b: mod50 +2%, mod25 +3%, mod20 +2% | 29, 107, 136, 142, 163 | +0.20 / +0.23 |
| L31 down c760 | 2% / 0 | **res//10** (R2 0.82): (tens) res in {182..200} | off (on 0) | - | same | 41, 89, 137, 149, 178 | +0.98 / -0.03 |

</details>

<details><summary>res%10: 47 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L20 down c13 | 42% / 39% | **res%10** (R2 0.86): res mod 10 in {5..8} | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {2,3,4,5}; a%10 in {1} -> b%10 in {3,4,5,6}; a%10 in {2} -> b%10 in {4,5,6,7}; a%10 in {3} -> b%10 in {5,6,7}; a%10 in {4} -> b%10 in {6,7,8,9}; a%10 in {5} -> b%10 in {0,8,9}; a%10 in {6} -> b%10 in {0,1,8,9}; a%10 in {7} -> b%10 in {0,1,2,3,9}; a%10 in {8} -> b%10 in {0,1,2,3,4}; a%10 in {9} -> b%10 in {1,2,3,4,5} | res: mod10 +17% | res: mod10 +17% | 11, 103, 122, 173, 180 | +0.82 / +0.85 |
| L20 down c21 | 42% / 37% | **res%10** (R2 0.89): res mod 10 in {0, 7..9} | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {0,1,2,3,9}; a%10 in {1} -> b%10 in {0,1,2,3}; a%10 in {2} -> b%10 in {1,2,3,4}; a%10 in {3} -> b%10 in {3,4,5,6}; a%10 in {4} -> b%10 in {4,5,6}; a%10 in {5} -> b%10 in {6,7}; a%10 in {6} -> b%10 in {6,7,8}; a%10 in {7} -> b%10 in {0,7,8,9}; a%10 in {8} -> b%10 in {0,1,8,9}; a%10 in {9} -> b%10 in {0,1,2,8,9} | res: mod10 +17%, mod5 +3% | a: mod10 +2%; b: mod10 +2%; res: mod10 +15%, mod5 +2% | 25, 104, 123, 125, 145 | +0.83 / +0.86 |
| L21 down c4 | 41% / 40% | **res%10** (R2 0.85): res mod 10 in {0..3} | **units(a,b)** (R2 0.46): a%10 in {0} -> b%10 in {0,7,8,9}; a%10 in {1} -> b%10 in {0,1,8,9}; a%10 in {2} -> b%10 in {0,1,2,8,9}; a%10 in {3} -> b%10 in {0,1,2,3,9}; a%10 in {4} -> b%10 in {1,2,3,4}; a%10 in {5} -> b%10 in {2,3,4,5,7}; a%10 in {6} -> b%10 in {4,5,6}; a%10 in {7} -> b%10 in {4,5,6,7}; a%10 in {8} -> b%10 in {5,6,7,8}; a%10 in {9} -> b%10 in {6,7,8,9} | res: mod10 +13% | res: mod10 +15% | 42, 62, 102, 122, 132 | +0.82 / +0.83 |
| L21 down c10 | 11% / 18% | **res%10** (R2 0.93): res mod 10 in {1} | unexplained (best R2 0.50) | res: mod10 +5%, mod5 +6% | res: mod10 +8%, mod5 +10%, mod2 +3% | 2, 3, 53, 63, 113 | +0.74 / +0.74 |
| L21 down c13 | 24% / 10% | **res%10** (R2 0.73): res mod 10 in {3..5} | **res** (R2 0.55): res in {-16, -6, 4, 14, 24..25, 34, 44, 54, 64, 74, 84, 94} | res: mod10 +6%, mod5 +3% | res: mod10 +3%, mod5 +2% | 4, 124, 174, 184, 194 | +0.73 / +0.73 |
| L21 down c14 | 11% / 10% | **res%10** (R2 0.89): res mod 10 in {5} | **res%10** (R2 0.60): res mod 10 in {5} | res: mod10 +5%, mod5 +5% | res: mod10 +3%, mod5 +4% | 5, 35, 105, 115, 125 | +0.74 / +0.74 |
| L21 down c16 | 11% / 10% | **res%10** (R2 0.93): res mod 10 in {6} | **units(a,b)** (R2 0.55): a%10 in {0} -> b%10 in {4}; a%10 in {1} -> b%10 in {5}; a%10 in {2} -> b%10 in {6}; a%10 in {4} -> b%10 in {8}; a%10 in {5} -> b%10 in {9}; a%10 in {6} -> b%10 in {0}; a%10 in {7} -> b%10 in {1}; a%10 in {8} -> b%10 in {2}; a%10 in {9} -> b%10 in {3} | res: mod10 +3%, mod5 +4% | res: mod5 +3% | 6, 106, 116, 176, 186 | +0.70 / +0.73 |
| L21 down c42 | 10% / 4% | **res%10** (R2 0.91): res mod 10 in {0} | **res** (R2 0.56): res in {10, 20, 30, 40, 50, 70, 80} | - | same | 20, 25, 164, 192, 200 | +0.18 / -0.10 |
| L21 down c53 | 10% / 6% | **res%10** (R2 0.96): res mod 10 in {9} | **res** (R2 0.60): res in {-99, -11, -1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99} | res: mod5 +2% | - | 85, 86, 97, 145, 147 | +0.44 / +0.43 |
| L21 down c68 | 20% / 10% | **res%10** (R2 0.96): res mod 10 in {0, 8} | **res** (R2 0.52): res in {0, 8, 10, 18, 20, 28, 88, 98} | - | same | 0, 3, 38, 128, 158 | +0.47 / +0.49 |
| L22 down c8 | 10% / 9% | **res%10** (R2 0.98): res mod 10 in {0} | **res%10** (R2 0.52): res mod 10 in {0} | res: mod10 +5%, mod5 +8% | res: mod10 +3%, mod5 +6% | 10, 20, 30, 60, 70 | +0.77 / +0.79 |
| L22 down c9 | 10% / 8% | **res%10** (R2 0.94): res mod 10 in {9} | **units(a,b)** (R2 0.50): a%10 in {0} -> b%10 in {1}; a%10 in {1} -> b%10 in {2}; a%10 in {2} -> b%10 in {3}; a%10 in {5} -> b%10 in {6}; a%10 in {6} -> b%10 in {7}; a%10 in {7} -> b%10 in {8}; a%10 in {9} -> b%10 in {0} | res: mod10 +6%, mod5 +7%, mod2 +3% | res: mod10 +2%, mod5 +3% | 31, 41, 51, 71, 91 | +0.66 / +0.68 |
| L22 down c129 | 10% / 1% | **res%10** (R2 0.94): res mod 10 in {6} | **res%100** (R2 0.60): res mod 100: no class above 0.5 (max 0.47) | - | same | 3, 13, 67, 103, 193 | -0.10 / +0.09 |
| L22 down c212 | 10% / 5% | **res%10** (R2 0.97): res mod 10 in {3} | **res** (R2 0.65): res in {3, 13, 23, 33, 43, 83, 93} | - | same | 9, 11, 41, 45, 85 | +0.24 / +0.48 |
| L22 down c333 | 10% / 0 | **res%10** (R2 0.91): res mod 10 in {4} | off (on 0) | - | same | 0, 45, 50, 100, 192 | +0.28 / +0.01 |
| L23 down c4 | 11% / 16% | **res%10** (R2 0.91): res mod 10 in {2} | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {8}; a%10 in {1} -> b%10 in {9}; a%10 in {2} -> b%10 in {0,2,4,6,8}; a%10 in {3} -> b%10 in {1}; a%10 in {4} -> b%10 in {2}; a%10 in {5} -> b%10 in {3,7}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {5}; a%10 in {8} -> b%10 in {6}; a%10 in {9} -> b%10 in {7} | res: mod10 +6%, mod5 +6%, mod2 +3% | res: mod10 +5%, mod5 +8%, mod2 +3% | 73, 93, 140, 184, 190 | +0.89 / +0.87 |
| L23 down c9 | 10% / 14% | **res%10** (R2 0.95): res mod 10 in {6} | **units(a,b)** (R2 0.53): a%10 in {0} -> b%10 in {4}; a%10 in {1} -> b%10 in {5}; a%10 in {2} -> b%10 in {6}; a%10 in {4} -> b%10 in {0,8}; a%10 in {5} -> b%10 in {9}; a%10 in {6} -> b%10 in {0}; a%10 in {7} -> b%10 in {1}; a%10 in {8} -> b%10 in {2,4}; a%10 in {9} -> b%10 in {3} | res: mod10 +3%, mod5 +5%, mod2 +2% | res: mod10 +3%, mod5 +5%, mod2 +2% | 87, 98, 148, 184, 185 | +0.87 / +0.89 |
| L23 down c14 | 11% / 8% | **res%10** (R2 0.93): res mod 10 in {4} | **res%100** (R2 0.58): res mod 100 in {4, 14, 24, 34, 44, 54, 64, 74, 84, 94} [coarser: res mod 50 in {4, 14, 24, 34, 44}, R2 0.80] | res: mod10 +4%, mod5 +5%, mod2 +2% | res: mod10 +3%, mod5 +3% | 4, 14, 44, 74, 94 | +0.86 / +0.88 |
| L23 down c24 | 10% / 7% | **res%10** (R2 0.96): res mod 10 in {7} | **res** (R2 0.67): res in {-97, -23, -13, -3, 7, 17, 27, 37, 47, 57, 67, 77, 87, 97} | res: mod10 +3%, mod5 +3% | res: mod5 +3% | 39, 89, 135, 139, 159 | +0.82 / +0.80 |
| L23 down c106 | 9% / 2% | **res%10** (R2 0.84): res mod 10 in {9} | **res** (R2 0.66): res in {9, 19, 29, 89, 99} | - | same | 59, 79, 109, 129, 139 | +0.38 / +0.23 |
| L23 down c205 | 10% / 4% | **res%10** (R2 0.96): res mod 10 in {8} | **res** (R2 0.61): res in {8, 18, 28, 38, 48, 88, 98} | - | same | 36, 46, 71, 86, 106 | +0.75 / +0.75 |
| L24 down c4 | 11% / 12% | **res%10** (R2 0.86): res mod 10 in {8} | unexplained (best R2 0.50) | res: mod10 +7%, mod5 +8%, mod2 +4% | res: mod10 +5%, mod5 +6% | 28, 38, 58, 68, 88 | +0.90 / +0.88 |
| L24 down c5 | 11% / 12% | **res%10** (R2 0.90): res mod 10 in {3} | unexplained (best R2 0.47) | res: mod10 +7%, mod5 +8%, mod2 +4% | res: mod10 +5%, mod5 +6% | 29, 71, 92, 147, 171 | +0.89 / +0.84 |
| L24 down c12 | 10% / 11% | **res%10** (R2 0.96): res mod 10 in {5} | **units(a,b)** (R2 0.64): a%10 in {0} -> b%10 in {5}; a%10 in {1} -> b%10 in {6}; a%10 in {2} -> b%10 in {7}; a%10 in {3} -> b%10 in {8}; a%10 in {4} -> b%10 in {9}; a%10 in {5} -> b%10 in {0,5}; a%10 in {6} -> b%10 in {1}; a%10 in {7} -> b%10 in {2}; a%10 in {8} -> b%10 in {3}; a%10 in {9} -> b%10 in {4} | res: mod10 +3%, mod5 +4% | res: mod10 +4%, mod5 +5% | 4, 7, 94, 103, 107 | +0.80 / +0.89 |
| L24 down c46 | 10% / 9% | **res%10** (R2 0.98): res mod 10 in {0} | **res%20** (R2 0.47): res mod 20 in {0, 10} [coarser: res mod 10 in {0}, R2 0.83] | - | same | 40, 50, 60, 70, 120 | +0.72 / +0.68 |
| L24 down c83 | 10% / 4% | **res%10** (R2 0.96): res mod 10 in {9} | **res** (R2 0.69): res in {9, 19, 29, 39, 69, 79, 89, 99} | - | same | 6, 29, 39, 89, 121 | +0.24 / +0.37 |
| L24 down c108 | 10% / 5% | **res%10** (R2 0.96): res mod 10 in {6} | **res** (R2 0.59): res in {6, 16, 26, 36, 86, 96} | - | same | 46, 106, 119, 146, 166 | +0.54 / +0.46 |
| L25 down c11 | 11% / 11% | **res%10** (R2 0.92): res mod 10 in {7} | **units(a,b)** (R2 0.51): a%10 in {0} -> b%10 in {3}; a%10 in {1} -> b%10 in {4}; a%10 in {2} -> b%10 in {5}; a%10 in {3} -> b%10 in {6}; a%10 in {5} -> b%10 in {8}; a%10 in {6} -> b%10 in {9}; a%10 in {7} -> b%10 in {0}; a%10 in {8} -> b%10 in {1}; a%10 in {9} -> b%10 in {2} | res: mod10 +5%, mod5 +5%, mod2 +4% | res: mod10 +3%, mod5 +4% | 36, 126, 131, 151, 156 | +0.88 / +0.92 |
| L25 down c12 | 11% / 28% | **res%10** (R2 0.86): res mod 10 in {1} | unexplained (best R2 0.47) | res: mod10 +4%, mod5 +4%, mod2 +3% | res: mod10 +4%, mod5 +5%, mod2 +5% | 41, 51, 61, 81, 91 | +0.90 / +0.88 |
| L25 down c24 | 11% / 9% | **res%10** (R2 0.86): res mod 10 in {9} | **units(a,b)** (R2 0.53): a%10 in {0} -> b%10 in {1}; a%10 in {1} -> b%10 in {2}; a%10 in {2} -> b%10 in {3}; a%10 in {5} -> b%10 in {6}; a%10 in {6} -> b%10 in {7}; a%10 in {7} -> b%10 in {8}; a%10 in {8} -> b%10 in {9}; a%10 in {9} -> b%10 in {0} | res: mod10 +3%, mod5 +3%, mod2 +2% | res: mod5 +3% | 39, 59, 69, 79, 149 | +0.90 / +0.89 |
| L25 down c53 | 10% / 5% | **res%10** (R2 0.96): res mod 10 in {5} | **res%100** (R2 0.52): res mod 100 in {15, 25, 85, 95} | - | same | 15, 75, 115, 155, 175 | +0.62 / +0.43 |
| L26 down c175 | 10% / 5% | **res%10** (R2 0.97): res mod 10 in {1} | **res** (R2 0.59): res in {11, 21, 31, 41, 51, 61, 71, 91} | - | same | 77, 99, 111, 130, 170 | +0.14 / +0.10 |
| L27 down c69 | 16% / 3% | **res%10** (R2 0.77): res mod 10 in {7..8} | unexplained (best R2 0.49) | - | same | 10, 11, 32, 92, 131 | -0.25 / -0.08 |
| L27 down c80 | 11% / 3% | **res%10** (R2 0.76): res mod 10 in {9} | **res** (R2 0.57): res in {9, 29, 89} | - | same | 49, 109, 149, 154, 158 | +0.16 / +0.03 |
| L27 down c165 | 10% / 4% | **res%10** (R2 0.99): res mod 10 in {6} | **res** (R2 0.68): res in {6, 16, 26, 86, 96} | - | same | 6, 56, 66, 116, 136 | -0.40 / -0.38 |
| L28 down c36 | 13% / 13% | **res%10** (R2 0.75): res mod 10 in {2} | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {8}; a%10 in {2} -> b%10 in {0,2}; a%10 in {3} -> b%10 in {1}; a%10 in {4} -> b%10 in {2}; a%10 in {5} -> b%10 in {3}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {5}; a%10 in {8} -> b%10 in {6}; a%10 in {9} -> b%10 in {7} | res: mod10 +3%, mod5 +3% | res: mod5 +2% | 84, 106, 154, 164, 184 | +0.92 / +0.88 |
| L28 down c69 | 10% / 11% | **res%10** (R2 0.93): res mod 10 in {6} | **res%100** (R2 0.46): res mod 100 in {6, 16, 26, 36, 46, 56, 66, 76, 86, 96} [coarser: res mod 50 in {6, 16, 26, 36, 46}, R2 0.81] | - | res: mod5 +2% | 88, 108, 148, 158, 184 | +0.89 / +0.74 |
| L28 down c93 | 14% / 8% | **res%10** (R2 0.77): res mod 10 in {5} | **units(a,b)** (R2 0.62): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {6}; a%10 in {6} -> b%10 in {1}; a%10 in {9} -> b%10 in {4} | res: mod5 +3% | - | 67, 72, 73, 74, 79 | +0.86 / +0.77 |
| L28 down c129 | 9% / 5% | **res%10** (R2 0.87): res mod 10 in {9} | **res** (R2 0.57): res in {9, 19, 39, 49, 59, 69, 79, 89, 99} | - | same | 42, 87, 110, 168, 172 | +0.73 / +0.65 |
| L28 down c160 | 10% / 6% | **res%10** (R2 0.93): res mod 10 in {7} | **res** (R2 0.66): res in {-97, -3, 7, 17, 27, 37, 47, 57, 67, 77, 87, 97} | - | same | 106, 131, 139, 151, 163 | +0.90 / +0.89 |
| L28 down c271 | 10% / 5% | **res%10** (R2 0.96): res mod 10 in {3} | **res** (R2 0.67): res in {3, 13, 23, 33, 53, 63, 73, 83, 93} | - | same | 71, 101, 136, 151, 171 | +0.80 / +0.80 |
| L28 down c300 | 10% / 4% | **res%10** (R2 0.97): res mod 10 in {4} | **res** (R2 0.57): res in {14, 24, 34, 84, 94} | - | same | 4, 108, 115, 126, 128 | +0.74 / +0.56 |
| L29 down c132 | 11% / 7% | **res%10** (R2 0.90): res mod 10 in {4} | **res** (R2 0.53): res in {-26, -16, -6, 14, 24, 34, 44, 54, 64, 74, 84, 94} [coarser: res mod 100 in {14, 24, 34, 54, 64, 74, 84, 94}, R2 0.82] | res: mod10 +2%, mod5 +3%, mod2 +2% | - | 95, 105, 106, 166, 182 | +0.94 / +0.93 |
| L29 down c604 | 10% / 3% | **res%10** (R2 0.77): res mod 10 in {1} | **res** (R2 0.50): res in {11, 21, 31, 51, 71} | - | same | 51, 71, 99, 151, 171 | +0.44 / +0.30 |
| L30 down c187 | 10% / 5% | **res%10** (R2 0.82): res mod 10 in {7} | **res** (R2 0.59): res in {17, 27, 37, 47, 57, 77, 87} | - | same | 5, 37, 114, 137, 164 | -0.05 / -0.13 |
| L30 down c209 | 10% / 4% | **res%10** (R2 0.91): res mod 10 in {4} | **res** (R2 0.59): res in {14, 24, 34, 44, 74, 84, 94} | - | same | 47, 62, 65, 67, 174 | -0.01 / -0.30 |
| L30 down c230 | 10% / 6% | **res%10** (R2 0.89): res mod 10 in {8} | unexplained (best R2 0.44) | res: mod5 +2% | - | 78, 128, 148, 168, 178 | +0.89 / +0.88 |

</details>

<details><summary>always: 26 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L20 down c2 | 100% / 100% | always | same | res: mod100 +11%, mod50 +3% | res: mod100 +3%, mod50 +2% | 82, 86, 88, 89, 189 | +0.61 / +0.86 |
| L21 down c38 | 100% / 100% | always | same | - | same | 26, 27, 29, 33, 61 | +0.49 / +0.49 |
| L22 down c3 | 100% / 100% | always | same | - | same | 40, 42, 44, 45, 64 | +0.14 / +0.05 |
| L23 down c28 | 100% / 100% | always | same | - | same | 163, 173, 183, 184, 185 | +0.09 / +0.55 |
| L23 o c308 (H15) | 96% / 91% | always | unexplained (best R2 0.41) | - | same | 20, 80, 176, 177, 184 | +0.24 / -0.12 |
| L25 down c6 | 100% / 100% | always | same | - | same | 83, 85, 88, 99, 104 | +0.39 / -0.09 |
| L26 down c14 | 100% / 100% | always | same | - | same | 45, 55, 73, 78, 93 | +0.47 / +0.52 |
| L27 down c0 | 100% / 100% | always | same | - | same | 34, 35, 94, 129, 184 | +0.17 / +0.05 |
| L28 down c1 | 100% / 100% | always | same | - | same | 108, 109, 131, 136, 168 | +0.47 / +0.36 |
| L29 down c4 | 100% / 100% | always | same | - | same | 1, 3, 5, 7, 8 | +0.08 / +0.25 |
| L30 down c377 | 100% / 100% | always | same | - | same | 0, 1, 4, 10, 100 | -0.39 / -0.04 |
| L31 down c2 | 100% / 100% | always | same | - | same | 106, 195, 196, 197, 200 | +0.24 / -0.53 |
| L31 down c6 | 100% / 100% | always | same | - | same | 146, 166, 167, 168, 179 | -0.61 / +0.11 |
| L31 down c7 | 100% / 100% | always | same | - | same | 151, 164, 167, 187, 189 | +0.14 / -0.02 |
| L31 o c7 (H14) | 100% / 100% | always | same | - | same | 114, 134, 136, 163, 166 | +0.05 / -0.27 |
| L31 down c9 | 100% / 100% | always | same | - | same | 98, 103, 105, 136, 137 | -0.31 / +0.03 |
| L31 down c11 | 97% / 45% | always | unexplained (best R2 0.35) | - | same | 124, 126, 128, 142, 152 | +0.01 / -0.18 |
| L31 down c18 | 100% / 100% | always | same | - | same | 117, 118, 124, 132, 134 | -0.56 / -0.43 |
| L31 down c22 | 100% / 100% | always | same | - | same | 119, 123, 139, 142, 143 | +0.48 / +0.37 |
| L31 down c23 | 100% / 100% | always | same | - | same | 71, 72, 73, 74, 77 | -0.14 / -0.10 |
| L31 down c29 | 99% / 45% | always | unexplained (best R2 0.42) | - | same | 19, 39, 48, 49, 50 | -0.60 / -0.31 |
| L31 down c47 | 100% / 100% | always | same | - | same | 70, 74, 75, 76, 78 | -0.21 / +0.13 |
| L31 down c61 | 100% / 100% | always | same | - | same | 92, 93, 94, 97, 129 | -0.38 / -0.45 |
| L31 o c135 (H12) | 100% / 100% | always | same | - | same | 124, 158, 167, 168, 193 | +0.20 / +0.14 |
| L31 o c165 (H15) | 98% / 33% | always | **tens(a,b)** (R2 0.52): a//10 in {0,1} -> b//10 in {0,10}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3,4} -> b//10 in {0}; a//10 in {5} -> b//10 in {0,4,5}; a//10 in {6} -> b//10 in {5,6}; a//10 in {7} -> b//10 in {0,6}; a//10 in {8} -> b//10 in {0,1,4,6,7,8}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | - | same | 50, 57, 59, 105, 149 | +0.24 / +0.48 |
| L31 down c756 | 99% / 98% | always | same | - | same | 0, 1, 2, 3, 4 | -0.43 / -0.36 |

</details>

<details><summary>tens(a,b): 22 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L20 down c5 | 71% / 100% | **tens(a,b)** (R2 0.74): a//10 in {0,1,2} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,10}; a//10 in {4,5} -> b//10 in {0,1,2,3,4,5,6}; a//10 in {6} -> b//10 in {0,1,2,3,4}; a//10 in {7} -> b//10 in {0,1,2,3}; a//10 in {8} -> b//10 in {0,1,2,9}; a//10 in {9} -> b//10 in {0,1,2,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,6,8,9,10} | always | a: mod100 +3%; b: mod100 +2%; res: mod100 +3% | - | 55, 62, 72, 81, 91 | +0.73 / +0.46 |
| L20 down c80 | 10% / 4% | **tens(a,b)** (R2 0.73): a//10 in {0} -> b//10 in {8,9,10}; a//10 in {7} -> b//10 in {10}; a//10 in {8,10} -> b//10 in {0,8,9,10}; a//10 in {9} -> b//10 in {0,7,8,9,10} | unexplained (best R2 0.40) | - | same | 46, 47, 48, 153, 165 | +0.31 / +0.27 |
| L20 down c136 | 8% / 0 | **tens(a,b)** (R2 0.67): a//10 in {5} -> b//10 in {7}; a//10 in {6} -> b//10 in {6,7,8}; a//10 in {7} -> b//10 in {6,7} | off (on 0) | - | same | 125, 128, 129, 130, 188 | -0.27 / -0.14 |
| L20 down c172 | 5% / 3% | **tens(a,b)** (R2 0.66): a//10 in {9} -> b//10 in {1,2,3,6}; a//10 in {10} -> b//10 in {1,2,3} | **a** (R2 0.79): a in {99..100} | - | same | 96, 97, 98, 99, 106 | -0.02 / +0.60 |
| L21 down c85 | 16% / 4% | **tens(a,b)** (R2 0.58): a//10 in {0} -> b//10 in {0,1,2,3,4,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,10}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3,9,10} -> b//10 in {0} | unexplained (best R2 0.40) | - | same | 79, 93, 101, 147, 148 | +0.11 / -0.18 |
| L21 down c267 | 7% / 0 | **tens(a,b)** (R2 0.69): a//10 in {4} -> b//10 in {4}; a//10 in {7} -> b//10 in {10}; a//10 in {8} -> b//10 in {9,10}; a//10 in {9} -> b//10 in {8,9,10}; a//10 in {10} -> b//10 in {7,8,9,10} | off (on 0) | - | same | 99, 178, 179, 189, 191 | +0.49 / -0.17 |
| L22 o c10 (H15) | 29% / 19% | **tens(a,b)** (R2 0.78): a//10 in {0,1,2} -> b//10 in {8,9,10}; a//10 in {3,4,5,6,7} -> b//10 in {9,10}; a//10 in {8} -> b//10 in {0,2,3,7,8,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.82): a//10 in {0,1,2,3,4} -> b//10 in {9,10}; a//10 in {5,6,8} -> b//10 in {10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | - | same | 7, 27, 28, 194, 195 | -0.11 / +0.94 |
| L22 down c134 | 20% / 1% | **tens(a,b)** (R2 0.70): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {9,10}; a//10 in {2} -> b//10 in {8,9,10}; a//10 in {3} -> b//10 in {7,8,9,10}; a//10 in {4} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {6}; a//10 in {6} -> b//10 in {4,5}; a//10 in {7} -> b//10 in {3,4,5}; a//10 in {8} -> b//10 in {2,3}; a//10 in {9} -> b//10 in {1,2,3,10}; a//10 in {10} -> b//10 in {0,1,2,3} | unexplained (best R2 0.20) | - | same | 36, 52, 55, 56, 179 | +0.66 / +0.22 |
| L22 o c200 (H15) | 13% / 1% | **tens(a,b)** (R2 0.59): a//10 in {0,1,2,3,4,8} -> b//10 in {9,10}; a//10 in {5,6,7} -> b//10 in {10}; a//10 in {9} -> b//10 in {3,4,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.62): a//10 in {0,1,2,3,4,5,6} -> b//10 in {10}; a//10 in {10} -> b//10 in {1,10} | - | same | 86, 93, 94, 95, 96 | +0.47 / -0.20 |
| L22 down c969 | 3% / 0 | **tens(a,b)** (R2 0.46): a//10 in {8,9} -> b//10 in {9,10}; a//10 in {10} -> b//10 in {8,10} | off (on 0) | - | same | 4, 63, 76, 172, 174 | +0.10 / +nan |
| L23 down c13 | 7% / 2% | **tens(a,b)** (R2 0.64): a//10 in {0,1,2,5,7} -> b//10 in {10}; a//10 in {8} -> b//10 in {9,10}; a//10 in {9} -> b//10 in {8,9,10}; a//10 in {10} -> b//10 in {0,6,7,8,9,10} | unexplained (best R2 0.45) | - | same | 181, 182, 185, 188, 198 | +0.74 / -0.18 |
| L24 o c23 (H13) | 68% / 83% | **tens(a,b)** (R2 0.56): a//10 in {0,1} -> b//10 in {6,7,8,9}; a//10 in {2,3} -> b//10 in {5,6,7,8,9}; a//10 in {4} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {1,2,3,4,5,7,8,9,10}; a//10 in {7,8,9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | unexplained (best R2 0.40) | - | same | 89, 127, 135, 183, 198 | +0.41 / +0.11 |
| L24 down c51 | 28% / 0 | **tens(a,b)** (R2 0.80): a//10 in {2} -> b//10 in {9,10}; a//10 in {3} -> b//10 in {8,9,10}; a//10 in {4,5} -> b//10 in {6,7,8,9,10}; a//10 in {6} -> b//10 in {5,6,7,8,9,10}; a//10 in {7} -> b//10 in {4,5,6,7}; a//10 in {8} -> b//10 in {3,4,5,6}; a//10 in {9,10} -> b//10 in {2,3,4,5,10} | off (on 0) | - | same | 27, 29, 30, 35, 183 | +0.80 / +0.07 |
| L24 down c207 | 6% / 7% | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {9}; a//10 in {8} -> b//10 in {10}; a//10 in {9} -> b//10 in {0,8,9,10}; a//10 in {10} -> b//10 in {8,9,10} | **a** (R2 0.61): a in {93..99} | - | same | 100, 101, 108, 110, 188 | +0.35 / +0.26 |
| L25 down c79 | 20% / 0 | **tens(a,b)** (R2 0.52): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {9}; a//10 in {2} -> b//10 in {8,9}; a//10 in {3} -> b//10 in {7,8,9}; a//10 in {4} -> b//10 in {6,8}; a//10 in {6} -> b//10 in {4,5}; a//10 in {7} -> b//10 in {3,4,5}; a//10 in {8} -> b//10 in {2,3,4}; a//10 in {9} -> b//10 in {1,2,3}; a//10 in {10} -> b//10 in {0,1,2,3} | off (on 0) | - | same | 8, 53, 57, 106, 116 | +0.16 / -0.16 |
| L26 down c115 | 3% / 16% | **tens(a,b)** (R2 0.56): a//10 in {0} -> b//10 in {2,3}; a//10 in {1} -> b//10 in {2}; a//10 in {3} -> b//10 in {0} | **tens(a,b)** (R2 0.55): a//10 in {3} -> b//10 in {0,3,4,5,10}; a//10 in {4} -> b//10 in {0,1,5}; a//10 in {5} -> b//10 in {1,2}; a//10 in {6} -> b//10 in {2,3}; a//10 in {7} -> b//10 in {3,4}; a//10 in {8} -> b//10 in {4,5}; a//10 in {10} -> b//10 in {6,7} | - | same | 13, 125, 131, 132, 137 | +0.59 / +0.64 |
| L27 down c68 | 26% / 1% | **tens(a,b)** (R2 0.50): a//10 in {0} -> b//10 in {6}; a//10 in {1} -> b//10 in {4,5,6,7}; a//10 in {2} -> b//10 in {3,4,5,6}; a//10 in {3} -> b//10 in {1,2,3,4,5}; a//10 in {4,5} -> b//10 in {1,2,3}; a//10 in {6} -> b//10 in {1,2}; a//10 in {7} -> b//10 in {1} | unexplained (best R2 0.45) | - | same | 70, 74, 86, 87, 112 | +0.41 / +0.36 |
| L29 down c449 | 7% / 0 | **tens(a,b)** (R2 0.71): a//10 in {5} -> b//10 in {10}; a//10 in {6} -> b//10 in {9,10}; a//10 in {7} -> b//10 in {8,9,10}; a//10 in {8} -> b//10 in {7,8}; a//10 in {9} -> b//10 in {6}; a//10 in {10} -> b//10 in {5,6} | off (on 0) | - | same | 157, 158, 159, 160, 164 | +0.48 / +nan |
| L31 down c41 | 65% / 17% | **tens(a,b)** (R2 0.63): a//10 in {0,1,2} -> b//10 in {6,7,8,9,10}; a//10 in {3} -> b//10 in {5,6,7,8,9,10}; a//10 in {4,6} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {0,2,3,4,5,6,7,8,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | unexplained (best R2 0.36) | - | same | 146, 147, 154, 166, 174 | +0.76 / +0.04 |
| L31 down c134 | 32% / 10% | **tens(a,b)** (R2 0.56): a//10 in {0} -> b//10 in {8,9}; a//10 in {1} -> b//10 in {7,8}; a//10 in {2} -> b//10 in {6,7}; a//10 in {3} -> b//10 in {5,6}; a//10 in {4} -> b//10 in {4,5}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {2,3}; a//10 in {7} -> b//10 in {1,2,9}; a//10 in {8} -> b//10 in {0,1,8,9}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,8,9} | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {9,10}; a//10 in {8} -> b//10 in {0}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,9,10}; a//10 in {10} -> b//10 in {0,1,2} | - | same | 91, 92, 93, 94, 95 | +0.48 / +0.97 |
| L31 down c174 | 43% / 7% | **tens(a,b)** (R2 0.53): a//10 in {0} -> b//10 in {4,5,6,7,8}; a//10 in {1} -> b//10 in {4,5,6,7}; a//10 in {2} -> b//10 in {3,4,5,6}; a//10 in {3} -> b//10 in {1,2,3,4,5}; a//10 in {4} -> b//10 in {0,1,2,3,4}; a//10 in {5} -> b//10 in {0,1,2,3}; a//10 in {6,7} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1}; a//10 in {9} -> b//10 in {7} | unexplained (best R2 0.45) | - | same | 73, 74, 76, 77, 80 | +0.80 / +0.77 |
| L31 down c544 | 6% / 13% | **tens(a,b)** (R2 0.54): a//10 in {0} -> b//10 in {1}; a//10 in {1,2} -> b//10 in {0,1} | unexplained (best R2 0.41) | - | same | 23, 25, 26, 27, 29 | +0.39 / +0.29 |

</details>

<details><summary>sub only: res%100: 22 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L24 down c348 | 0 / 2% | off (on 0) | **res%100** (R2 0.85): res mod 100: no class above 0.5 (max 0.46) | - | same | 97, 151, 188, 196, 197 | +0.51 / +0.71 |
| L24 down c482 | 0 / 1% | off (on 0) | **res%100** (R2 0.78): res mod 100: no class above 0.5 (max 0.41) | - | same | 64, 144, 154, 164, 183 | +nan / +0.75 |
| L24 down c501 | 0 / 3% | off (on 0) | **res%100** (R2 0.81): res mod 100: no class above 0.5 (max 0.46) | - | same | 15, 16, 17, 81, 151 | +0.40 / +0.60 |
| L24 down c678 | 0 / 6% | off (on 0) | **res%100** (R2 0.51): res mod 100 in {3} | - | same | 1, 2, 3, 28, 39 | +0.32 / +0.80 |
| L25 down c748 | 0 / 1% | off (on 0) | **res%100** (R2 0.63): res mod 100: no class above 0.5 (max 0.45) | - | same | 2, 11, 21, 100, 200 | -0.03 / +0.51 |
| L26 down c106 | 0 / 1% | off (on 0) | **res%100** (R2 0.79): res mod 100: no class above 0.5 (max 0.41) | - | same | 3, 72, 77, 99, 137 | +0.46 / +0.84 |
| L26 down c486 | 0 / 1% | off (on 0) | **res%100** (R2 0.75): res mod 100: no class above 0.5 (max 0.42) | - | same | 2, 96, 98, 99, 104 | +0.77 / +0.86 |
| L26 down c552 | 0 / 14% | off (on 0) | **res%100** (R2 0.55): res mod 100: no class above 0.5 (max 0.47) | - | same | 102, 103, 104, 105, 106 | +0.58 / +0.69 |
| L27 down c137 | 0 / 4% | off (on 0) | **res%100** (R2 0.59): res mod 100: no class above 0.5 (max 0.40) | - | same | 1, 38, 107, 108, 109 | +nan / +0.50 |
| L27 down c227 | 0 / 15% | off (on 0) | **res%100** (R2 0.56): res mod 100: no class above 0.5 (max 0.48) | - | same | 32, 33, 34, 43, 66 | +0.03 / +0.29 |
| L28 down c357 | 0 / 1% | off (on 0) | **res%100** (R2 0.68): res mod 100 in {2} | - | same | 3, 5, 45, 103, 123 | -0.03 / +0.78 |
| L28 down c408 | 0 / 1% | off (on 0) | **res%100** (R2 0.66): res mod 100: no class above 0.5 (max 0.48) | - | same | 0, 2, 3, 4, 23 | -0.40 / +0.73 |
| L28 down c430 | 0 / 1% | off (on 0) | **res%100** (R2 0.77): res mod 100: no class above 0.5 (max 0.44) | - | same | 1, 4, 59, 64, 194 | +0.44 / +0.63 |
| L29 down c103 | 0 / 2% | off (on 0) | **res%100** (R2 0.70): res mod 100: no class above 0.5 (max 0.46) | - | same | 5, 9, 15, 192, 194 | +0.55 / +0.87 |
| L29 down c177 | 0 / 1% | off (on 0) | **res%100** (R2 0.65): res mod 100: no class above 0.5 (max 0.34) | - | same | 73, 135, 144, 169, 170 | +nan / +0.32 |
| L29 down c390 | 0 / 1% | off (on 0) | **res%100** (R2 0.80): res mod 100: no class above 0.5 (max 0.47) | - | same | 4, 23, 155, 173, 177 | +nan / +0.60 |
| L29 down c469 | 0 / 2% | off (on 0) | **res%100** (R2 0.61): res mod 100: no class above 0.5 (max 0.48) | - | same | 5, 52, 53, 58, 128 | +0.46 / +0.62 |
| L29 down c514 | 0 / 3% | off (on 0) | **res%100** (R2 0.59): res mod 100: no class above 0.5 (max 0.49) | - | same | 0, 6, 92, 182, 200 | +0.03 / +0.72 |
| L29 down c761 | 0 / 3% | off (on 0) | **res%100** (R2 0.59): res mod 100: no class above 0.5 (max 0.45) | - | same | 1, 2, 3, 6, 9 | +0.32 / +0.46 |
| L29 down c928 | 0 / 1% | off (on 0) | **res%100** (R2 0.79): res mod 100: no class above 0.5 (max 0.43) | - | same | 1, 2, 4, 5, 83 | +0.02 / +0.53 |
| L31 down c234 | 0 / 1% | off (on 0) | **res%100** (R2 0.73): res mod 100: no class above 0.5 (max 0.40) | - | same | 31, 107, 155, 156, 178 | +nan / +0.83 |
| L31 down c776 | 0 / 3% | off (on 0) | **res%100** (R2 0.53): res mod 100: no class above 0.5 (max 0.45) | - | same | 8, 9, 40, 80, 84 | +0.05 / +0.81 |

</details>

<details><summary>res%20: 18 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L20 down c3 | 61% / 53% | **res%20** (R2 0.74): res mod 20 in {8..19} | unexplained (best R2 0.43) | a: mod10 +3%; res: mod20 +35% | b: mod20 +4%; res: mod25 +8%, mod20 +24% | 13, 14, 55, 71, 94 | +0.79 / +0.67 |
| L20 down c6 | 60% / 49% | **res%20** (R2 0.60): res mod 20 in {1..11} | unexplained (best R2 0.33) | res: mod20 +31% | b: mod20 +4%; res: mod25 +6%, mod20 +23% | 96, 97, 116, 117, 119 | +0.73 / +0.76 |
| L21 down c2 | 36% / 28% | **res%20** (R2 0.89): res mod 20 in {10..16} | **res** (R2 0.55): res in {-10..-4, 10..17, 30..36, 50..56, 70..76, 90..96} | res: mod20 +17% | res: mod25 +6%, mod20 +14% | 22, 24, 44, 83, 162 | +0.69 / +0.67 |
| L21 down c15 | 50% / 29% | **res%20** (R2 0.61): res mod 20 in {0, 4, 13..19} | unexplained (best R2 0.44) | res: mod20 +7% | res: mod25 +3%, mod20 +7% | 29, 109, 148, 149, 168 | +0.85 / +0.80 |
| L21 down c231 | 6% / 1% | **res%20** (R2 0.68): res mod 20 in {11} | unexplained (best R2 0.50) | - | same | 71, 111, 131, 159, 191 | +0.47 / +0.24 |
| L22 down c11 | 33% / 25% | **res%20** (R2 0.89): res mod 20 in {0..3, 18..19} | **res** (R2 0.51): res in {-99..-98, -41..-38, -22..-18, 0..3, 18..24, 38..43, 58..62, 78..82, 97..99} | res: mod20 +11% | res: mod20 +10% | 11, 33, 50, 54, 89 | +0.71 / +0.72 |
| L22 down c17 | 10% / 5% | **res%20** (R2 0.84): res mod 20 in {2..3} | **res** (R2 0.58): res in {-18, 2..3, 22..23, 42, 82} | res: mod20 +3%, mod4 +45% | - | 21, 52, 72, 92, 121 | +0.47 / +0.41 |
| L22 down c427 | 7% / 2% | **res%20** (R2 0.78): res mod 20 in {12} | unexplained (best R2 0.46) | res: mod4 +10% | - | 72, 132, 152, 172, 192 | +0.40 / +0.36 |
| L23 down c105 | 8% / 1% | **res%20** (R2 0.73): res mod 20 in {3, 13} | **res%100** (R2 0.48): res mod 100: no class above 0.5 (max 0.38) | - | same | 129, 130, 149, 159, 169 | +0.25 / +0.23 |
| L23 down c125 | 6% / 3% | **res%20** (R2 0.83): res mod 20 in {11} | **res** (R2 0.52): res in {11, 31, 91} | - | same | 11, 26, 31, 51, 91 | +0.43 / +0.50 |
| L24 down c210 | 7% / 1% | **res%20** (R2 0.74): res mod 20 in {12} | unexplained (best R2 0.28) | - | same | 86, 134, 151, 154, 186 | +0.07 / -0.01 |
| L25 down c108 | 14% / 6% | **res%20** (R2 0.85): res mod 20 in {2, 12..13} [coarser: res mod 10 in {2}, R2 0.82] | **res** (R2 0.57): res in {12..13, 22, 32..33, 72, 92} | - | same | 13, 32, 92, 113, 173 | +0.41 / +0.64 |
| L26 down c363 | 6% / 1% | **res%20** (R2 0.77): res mod 20 in {12} | unexplained (best R2 0.36) | - | same | 92, 132, 152, 153, 172 | +0.38 / +0.33 |
| L28 down c275 | 5% / 4% | **res%20** (R2 0.80): res mod 20 in {1} | unexplained (best R2 0.42) | - | same | 21, 31, 91, 131, 171 | +0.58 / +0.44 |
| L29 down c145 | 8% / 2% | **res%20** (R2 0.78): res mod 20 in {6, 16} [coarser: res mod 10 in {6}, R2 0.82] | unexplained (best R2 0.29) | - | same | 5, 17, 36, 56, 136 | +0.27 / +0.31 |
| L29 down c193 | 9% / 1% | **res%20** (R2 0.83): res mod 20 in {6, 8} | unexplained (best R2 0.19) | - | same | 76, 98, 107, 165, 176 | +0.53 / +0.39 |
| L29 down c209 | 12% / 3% | **res%20** (R2 0.82): res mod 20 in {2, 12} [coarser: res mod 10 in {2}, R2 0.84] | **res** (R2 0.65): res in {12, 22, 32} | - | same | 42, 82, 142, 172, 182 | +0.15 / -0.02 |
| L29 down c230 | 15% / 4% | **res%20** (R2 0.69): res mod 20 in {12..13} | **res** (R2 0.59): res in {12..13, 33, 92..93} | res: mod4 +2% | - | 60, 81, 84, 93, 173 | +0.13 / -0.00 |

</details>

<details><summary>sub only: unexplained: 15 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L21 down c341 | 0 / 1% | off (on 0) | unexplained (best R2 0.08) | - | same | 39, 44, 54, 65, 178 | +0.28 / +0.38 |
| L22 down c122 | 0 / 1% | off (on 0) | unexplained (best R2 0.09) | - | same | 4, 22, 86, 180, 181 | +0.10 / +0.16 |
| L23 down c65 | 0 / 19% | off (on 0) | unexplained (best R2 0.26) | - | same | 13, 34, 41, 141, 156 | +0.21 / +0.45 |
| L24 down c209 | 0 / 4% | off (on 0) | unexplained (best R2 0.27) | - | same | 0, 42, 43, 84, 86 | -0.10 / +0.73 |
| L24 down c240 | 0 / 2% | off (on 0) | unexplained (best R2 0.46) | - | same | 4, 41, 42, 51, 55 | +nan / +0.50 |
| L25 down c32 | 0 / 11% | off (on 0) | unexplained (best R2 0.47) | - | same | 77, 127, 162, 173, 175 | +0.23 / +0.38 |
| L27 down c71 | 0 / 9% | off (on 0) | unexplained (best R2 0.40) | - | same | 33, 123, 128, 143, 159 | +nan / +0.40 |
| L27 down c715 | 0 / 1% | off (on 0) | unexplained (best R2 0.08) | - | same | 24, 73, 85, 142, 146 | +nan / +nan |
| L28 down c584 | 0 / 3% | off (on 0) | unexplained (best R2 0.38) | - | same | 3, 4, 5, 45, 191 | +0.11 / +0.52 |
| L31 down c13 | 0 / 1% | off (on 0) | unexplained (best R2 0.38) | - | same | 0, 99, 118, 158, 176 | -0.02 / +0.48 |
| L31 down c24 | 0 / 3% | off (on 0) | unexplained (best R2 0.42) | - | same | 71, 119, 121, 122, 137 | +nan / -0.01 |
| L31 down c72 | 0 / 1% | off (on 0) | unexplained (best R2 0.29) | - | same | 107, 159, 178, 179, 182 | +0.00 / +0.03 |
| L31 down c90 | 0 / 53% | off (on 0) | unexplained (best R2 0.47) | - | same | 85, 86, 87, 88, 89 | -0.08 / -0.10 |
| L31 down c351 | 0 / 1% | off (on 0) | unexplained (best R2 0.47) | - | same | 47, 148, 186, 193, 194 | +nan / +0.06 |
| L31 down c983 | 0 / 7% | off (on 0) | unexplained (best R2 0.38) | - | same | 27, 48, 135, 175, 183 | -0.00 / -0.34 |

</details>

<details><summary>sub only: tens(a,b): 12 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L20 down c32 | 0 / 35% | off (on 0) | **tens(a,b)** (R2 0.59): a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {1,2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {3,4,5,6,7,8}; a//10 in {7,10} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10} | - | same | 116, 122, 130, 143, 149 | -0.04 / -0.12 |
| L20 down c73 | 0 / 12% | off (on 0) | **tens(a,b)** (R2 0.64): a//10 in {4} -> b//10 in {0}; a//10 in {5,8} -> b//10 in {0,1}; a//10 in {6,7} -> b//10 in {0,1,2}; a//10 in {10} -> b//10 in {0,1,2,3,4} | - | same | 102, 114, 117, 161, 163 | +0.15 / +0.25 |
| L20 down c181 | 0 / 13% | off (on 0) | **tens(a,b)** (R2 0.64): a//10 in {4} -> b//10 in {6}; a//10 in {5} -> b//10 in {6,7,8}; a//10 in {6} -> b//10 in {7,8,9}; a//10 in {7} -> b//10 in {8,9,10}; a//10 in {8} -> b//10 in {9,10} | - | same | 9, 11, 155, 190, 192 | -0.16 / +0.08 |
| L20 down c305 | 0 / 1% | off (on 0) | **tens(a,b)** (R2 0.55): a//10 in {9} -> b//10 in {9,10} | - | same | 59, 63, 64, 88, 100 | +nan / -0.30 |
| L21 down c124 | 0 / 29% | off (on 0) | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {0}; a//10 in {1} -> b//10 in {1}; a//10 in {2} -> b//10 in {6}; a//10 in {3} -> b//10 in {4,5,6,7,8}; a//10 in {4} -> b//10 in {6,7,8,9}; a//10 in {5} -> b//10 in {5,6,7,8,9,10}; a//10 in {6} -> b//10 in {6,7,8,9,10}; a//10 in {7} -> b//10 in {7,8,9,10}; a//10 in {8} -> b//10 in {8,9,10}; a//10 in {10} -> b//10 in {10} | - | same | 30, 90, 91, 92, 96 | -0.12 / +0.25 |
| L21 down c439 | 0 / 1% | off (on 0) | **tens(a,b)** (R2 0.66): a//10 in {10} -> b//10 in {3,4,5,6,7,8,9} | - | same | 25, 69, 125, 127, 129 | +0.03 / +0.16 |
| L22 down c38 | 0 / 2% | off (on 0) | **tens(a,b)** (R2 0.50): a//10 in {4} -> b//10 in {0} | - | same | 64, 65, 75, 104, 185 | +0.12 / +0.30 |
| L22 down c254 | 0 / 2% | off (on 0) | **tens(a,b)** (R2 0.65): a//10 in {6} -> b//10 in {0,10}; a//10 in {7} -> b//10 in {0} | - | same | 76, 81, 86, 103, 152 | +nan / +0.27 |
| L23 down c8 | 0 / 11% | off (on 0) | **tens(a,b)** (R2 0.62): a//10 in {5,6} -> b//10 in {0,1}; a//10 in {7} -> b//10 in {0,1,2,3}; a//10 in {8} -> b//10 in {0,1,3}; a//10 in {10} -> b//10 in {2,3} | - | same | 135, 165, 167, 175, 179 | +0.13 / +0.60 |
| L23 down c110 | 0 / 22% | off (on 0) | **tens(a,b)** (R2 0.56): a//10 in {3,4} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {6,7,8}; a//10 in {6} -> b//10 in {6,7,8,9,10}; a//10 in {7} -> b//10 in {7,8,9,10}; a//10 in {8} -> b//10 in {8,9,10}; a//10 in {9} -> b//10 in {7,9,10} | - | same | 82, 89, 90, 97, 98 | +0.11 / +0.31 |
| L24 down c313 | 0 / 27% | off (on 0) | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {0,1}; a//10 in {1} -> b//10 in {1}; a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {4,5,7}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {5,6,7,8,9,10} | - | same | 83, 101, 132, 139, 168 | +0.21 / +0.12 |
| L30 down c201 | 0 / 7% | off (on 0) | **tens(a,b)** (R2 0.85): a//10 in {0} -> b//10 in {2,3,4,5,6,7,8,9,10} | - | same | 8, 24, 133, 135, 139 | +nan / +0.02 |

</details>

<details><summary>res%50: 11 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L21 down c7 | 42% / 32% | **res%50** (R2 0.81): res mod 50 in {0..12, 42..49} | **res%100** (R2 0.49): res mod 100 in {0..13, 45, 47, 49..53, 55, 90..99} | res: mod100 +2%, mod50 +10%, mod25 +21% | res: mod100 +2%, mod50 +10%, mod25 +4% | 82, 83, 85, 172, 198 | +0.54 / +0.32 |
| L23 down c459 | 7% / 3% | **res%50** (R2 0.74): res mod 50 in {5, 15, 25, 35} [coarser: res mod 10 in {5}, R2 0.85] | **res** (R2 0.66): res in {5, 15, 25} | - | same | 2, 8, 48, 122, 141 | +0.08 / +0.13 |
| L24 down c28 | 16% / 3% | **res%50** (R2 0.70): res mod 50 in {2..5, 13..14, 43..44} | unexplained (best R2 0.25) | res: mod25 +3% | - | 54, 55, 104, 154, 155 | +0.58 / +0.57 |
| L27 down c204 | 9% / 4% | **res%50** (R2 0.81): res mod 50 in {11, 21, 31, 41} [coarser: res mod 10 in {1}, R2 0.84] | **res** (R2 0.57): res in {11, 21, 31, 71, 91} | - | same | 1, 30, 51, 129, 151 | +0.50 / +0.11 |
| L28 down c277 | 4% / 2% | **res%50** (R2 0.69): res mod 50 in {0, 49} | unexplained (best R2 0.20) | - | same | 45, 55, 170, 180, 190 | +0.55 / +0.32 |
| L29 down c72 | 13% / 4% | **res%50** (R2 0.74): res mod 50 in {20, 22..23, 27, 30, 32} | unexplained (best R2 0.47) | - | same | 52, 70, 92, 120, 170 | +0.55 / +0.26 |
| L29 down c577 | 7% / 3% | **res%50** (R2 0.72): res mod 50 in {19, 29, 39, 49} [coarser: res mod 10 in {9}, R2 0.83] | **res** (R2 0.52): res in {19, 29, 39, 79, 99} | - | same | 9, 47, 117, 127, 147 | +0.84 / +0.54 |
| L30 down c133 | 9% / 5% | **res%50** (R2 0.88): res mod 50 in {8, 18, 28, 48} | **res** (R2 0.61): res in {-2, 8, 18, 28, 38, 48, 98} | - | same | 36, 39, 89, 148, 198 | +0.15 / -0.05 |
| L30 down c305 | 2% / 0 | **res%50** (R2 0.72): res mod 50 in {1} | off (on 0) | - | same | 51, 101, 151, 185, 191 | +0.88 / +0.81 |
| L30 down c414 | 2% / 0 | **res%50** (R2 0.84): res mod 50 in {42} | off (on 0) | - | same | 43, 91, 93, 140, 143 | +0.27 / +0.23 |
| L31 down c819 | 3% / 1% | **res%50** (R2 0.71): res mod 50 in {36} | unexplained (best R2 0.35) | - | same | 89, 129, 149, 155, 178 | +0.86 / +0.91 |

</details>

<details><summary>sub only: a: 8 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L20 o c179 (H2) | 0 / 3% | off (on 0) | **a** (R2 0.79): a in {97..99} | - | same | 25, 40, 91, 100, 200 | +nan / -0.02 |
| L20 o c243 (H2) | 0 / 2% | off (on 0) | **a** (R2 0.57): a in {45} | - | same | 32, 45, 123, 135, 180 | -0.01 / +0.03 |
| L24 down c79 | 0 / 32% | off (on 0) | **a** (R2 0.51): a in {2..24, 97} | - | a: mod100 +2% | 102, 106, 192, 193, 198 | +0.02 / -0.03 |
| L24 o c501 (H22) | 0 / 2% | off (on 0) | **a** (R2 0.63): a in {27..29} | - | same | 10, 32, 33, 34, 37 | +0.47 / +0.18 |
| L31 down c10 | 0 / 6% | off (on 0) | **a** (R2 0.53): a in {1..2, 6..9} | - | same | 1, 2, 139, 159, 200 | +0.03 / +0.08 |
| L31 down c52 | 0 / 3% | off (on 0) | **a** (R2 0.68): a in {1..3} | - | same | 14, 69, 78, 86, 91 | +0.19 / -0.01 |
| L31 down c629 | 0 / 5% | off (on 0) | **a** (R2 0.65): a in {94..99} | - | same | 41, 61, 77, 120, 199 | +0.01 / +0.07 |
| L31 down c690 | 0 / 1% | off (on 0) | **a** (R2 0.63): a in {99} | - | same | 59, 86, 101, 102, 190 | +nan / +0.00 |

</details>

<details><summary>units(a,b): 7 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L20 o c192 (H2) | 6% / 12% | **units(a,b)** (R2 0.54): a%10 in {0,5} -> b%10 in {0,5} | **units(a,b)** (R2 0.56): a%10 in {0} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {5} -> b%10 in {0,5} | - | same | 30, 60, 70, 90, 180 | +0.41 / +0.54 |
| L20 o c288 (H2) | 4% / 13% | **units(a,b)** (R2 0.78): a%10 in {0} -> b%10 in {5}; a%10 in {5} -> b%10 in {0,5} | **units(a,b)** (R2 0.71): a%10 in {0,7,8,9} -> b%10 in {5}; a%10 in {5} -> b%10 in {0,1,2,3,4,5,6,7,8,9} | - | same | 69, 70, 71, 116, 166 | +0.62 / +0.26 |
| L20 down c378 | 18% / 6% | **units(a,b)** (R2 0.64): a%10 in {0} -> b%10 in {3}; a%10 in {5} -> b%10 in {7,8,9}; a%10 in {6} -> b%10 in {6,7,8,9}; a%10 in {7,8} -> b%10 in {5,6,7}; a%10 in {9} -> b%10 in {5,6} | unexplained (best R2 0.41) | - | same | 147, 161, 167, 187, 189 | +0.16 / +0.40 |
| L20 o c479 (H2) | 6% / 15% | **units(a,b)** (R2 0.66): a%10 in {0,5} -> b%10 in {0,5} | **units(a,b)** (R2 0.61): a%10 in {0} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {5} -> b%10 in {0,5} | - | same | 20, 30, 40, 60, 70 | +0.52 / +0.49 |
| L23 down c387 | 4% / 4% | **units(a,b)** (R2 0.98): a%10 in {0,5} -> b%10 in {0,5} | **units(a,b)** (R2 0.88): a%10 in {0,5} -> b%10 in {0,5} | - | same | 0, 62, 101, 174, 199 | +0.64 / +0.80 |
| L28 down c359 | 11% / 7% | **units(a,b)** (R2 0.88): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {9}; a%10 in {2} -> b%10 in {8}; a%10 in {3} -> b%10 in {7}; a%10 in {4} -> b%10 in {6}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {3}; a%10 in {8} -> b%10 in {2}; a%10 in {9} -> b%10 in {1} | **units(a,b)** (R2 0.56): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {1} | - | same | 50, 70, 100, 120, 200 | +0.67 / +0.62 |
| L29 down c293 | 12% / 11% | **units(a,b)** (R2 0.95): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {9}; a%10 in {2} -> b%10 in {8}; a%10 in {3} -> b%10 in {7}; a%10 in {4} -> b%10 in {6}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {3}; a%10 in {8} -> b%10 in {2}; a%10 in {9} -> b%10 in {1} | **units(a,b)** (R2 0.51): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {1}; a%10 in {2} -> b%10 in {2}; a%10 in {4} -> b%10 in {4}; a%10 in {8} -> b%10 in {8}; a%10 in {9} -> b%10 in {9} | - | same | 101, 111, 129, 142, 161 | +0.90 / +0.90 |

</details>

<details><summary>a: 6 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L22 o c352 (H3) | 1% / 2% | **a** (R2 0.54): a in {97} | **a** (R2 0.64): a in {97} | - | same | 82, 90, 101, 104, 110 | +0.25 / -0.22 |
| L23 o c187 (H7) | 1% / 4% | **a** (R2 0.70): a in {98} | **a** (R2 0.64): a in {97..99} | - | same | 40, 41, 59, 60, 159 | +0.12 / +0.23 |
| L23 o c370 (H22) | 1% / 1% | **a** (R2 0.82): a in {55} | **a** (R2 0.67): a in {55} | - | same | 15, 25, 30, 120, 150 | +0.04 / +0.07 |
| L23 o c435 (H22) | 1% / 2% | **a** (R2 0.54): a in {65} | unexplained (best R2 0.48) | - | same | 27, 36, 55, 68, 175 | +0.35 / -0.09 |
| L24 o c503 (H17) | 1% / 1% | **a** (R2 0.89): a in {90} | **a** (R2 0.82): a in {90} | - | same | 89, 90, 91, 180, 190 | -0.21 / -0.04 |
| L26 o c379 (H14) | 3% / 4% | **a** (R2 0.61): a in {92..94} | **a** (R2 0.81): a in {92..94} | - | same | 91, 92, 93, 94, 193 | +0.10 / +0.28 |

</details>

<details><summary>res%2: 4 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L20 down c47 | 43% / 28% | **res%2** (R2 0.74): res mod 2 in {0} | **units(a,b)** (R2 0.75): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} | a: mod2 -8%; b: mod2 -5%; res: mod2 +16% | a: mod2 -9%; b: mod2 -5%; res: mod2 +13% | 25, 39, 107, 137, 167 | +0.87 / +0.75 |
| L21 down c1 | 50% / 57% | **res%2** (R2 0.99): res mod 2 in {1} | **res%2** (R2 0.65): res mod 2 in {1} | res: mod2 +33% | a: mod2 +9%; res: mod2 +34% | 26, 52, 56, 62, 182 | +0.93 / +0.95 |
| L26 down c36 | 41% / 30% | **res%2** (R2 0.68): res mod 2 in {0} | **units(a,b)** (R2 0.67): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} | a: mod2 +5%; b: mod2 +3%; res: mod2 +8% | a: mod2 +8%; b: mod2 +7%; res: mod2 +10% | 16, 38, 98, 128, 176 | +0.83 / +0.87 |
| L28 down c23 | 41% / 19% | **res%2** (R2 0.70): res mod 2 in {0} | **units(a,b)** (R2 0.65): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} | a: mod2 +15%; b: mod2 +11%; res: mod2 +11% | a: mod2 +7%; b: mod2 +7%; res: mod2 +4% | 116, 124, 140, 156, 196 | +0.82 / +0.71 |

</details>

<details><summary>sub only: cmp(a,b): 3 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L25 down c106 | 0 / 45% | off (on 0) | **cmp(a,b)** (R2 0.70): cmp(a,b)=-1: 0.87, cmp(a,b)=0: 0.19, cmp(a,b)=1: 0.04 | - | same | 89, 90, 93, 100, 159 | +nan / +0.01 |
| L26 down c271 | 0 / 1% | off (on 0) | **cmp(a,b)** (R2 0.64): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.90, cmp(a,b)=1: 0.00 | - | same | 0, 103, 105, 111, 183 | +0.03 / +0.84 |
| L30 down c1003 | 0 / 1% | off (on 0) | **cmp(a,b)** (R2 0.74): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.92, cmp(a,b)=1: 0.00 | - | same | 1, 4, 10, 20, 200 | +0.15 / +0.41 |

</details>

<details><summary>sub only: res: 3 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L26 down c779 | 0 / 1% | off (on 0) | **res** (R2 0.84): res in {10} | - | same | 10, 115, 135, 155, 175 | +0.15 / +0.28 |
| L30 down c800 | 0 / 1% | off (on 0) | **res** (R2 0.73): res in {20} | - | same | 20, 55, 145, 146, 193 | +0.65 / +0.81 |
| L31 down c83 | 0 / 1% | off (on 0) | **res** (R2 0.52): res in {20} | - | same | 75, 99, 119, 126, 141 | +0.26 / +0.57 |

</details>

<details><summary>res%5: 2 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L20 down c9 | 60% / 49% | **res%5** (R2 0.95): res mod 5 in {1..3} | **res** (R2 0.62): res in {-97, -93, -63, -58, -53..-52, -48..-47, -43..-42, -38..-37, -33..-32, -28..-27, -23..-22, -19..-17, -13..-12, -9..-7, -4..-2, 1..3, 6..8, 11..13, 16..18, 21..23, 26..28, 31..33, 36..38, 41..43, 46..48, 51..53, 56..58, 61..63, 66..68, 71..73, 76..78, 81..83, 86..88, 91..93, 96..98} | res: mod5 +25% | res: mod5 +22% | 7, 37, 77, 82, 92 | +0.86 / +0.86 |
| L20 down c15 | 61% / 46% | **res%5** (R2 0.83): res mod 5 in {0..2} | unexplained (best R2 0.44) | res: mod5 +18% | a: mod5 +2%; b: mod5 +3%; res: mod5 +14% | 13, 33, 38, 68, 168 | +0.82 / +0.81 |

</details>

<details><summary>sub only: res//10: 1 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L22 down c160 | 0 / 37% | off (on 0) | **res//10** (R2 0.63): (tens) res in {-99, -60, -52..0} | - | same | 82, 88, 90, 93, 99 | -0.20 / +0.02 |

</details>

<details><summary>sub only: units(a,b): 1 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L25 down c715 | 0 / 1% | off (on 0) | **units(a,b)** (R2 0.60): a%10 in {0} -> b%10 in {0} | - | same | 174, 181, 183, 184, 188 | +0.24 / +0.24 |

</details>

<details><summary>res>=100: 1 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L26 down c7 | 52% / 0 | **res>=100** (R2 0.75): res>=100=0: 0.07, res>=100=1: 0.94 | off (on 0) | a: mod100 +3%; b: mod100 +3% | - | 104, 106, 107, 122, 123 | +0.76 / +nan |

</details>

<details><summary>sub only: always: 1 writers</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | top-5 tokens of its direct logit effect | corr(tuning over res, logit of res) add / sub |
|---|---|---|---|---|---|---|---|
| L26 down c10 | 0 / 96% | off (on 0) | always | - | same | 0, 64, 114, 182, 185 | -0.05 / +0.11 |

</details>

