# Appendix C5 — components whose main position is `=`, L28-31

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

<details><summary>layer 28: 242 components with main position `=`</summary>

<details><summary>down: 95</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L28 down c1 | 100% / 100% | always | same | - | same |
| L28 down c10 | 10% / 7% | **res//10** (R2 0.86): (tens) res in {11..17, 110..119} [coarser: res mod 100 in {10..19}, R2 0.95] | **res** (R2 0.78): res in {11..17} | - | same |
| L28 down c11 | 2% / 98% | unexplained (best R2 0.30) | always | - | same |
| L28 down c12 | 27% / 8% | **res//10** (R2 0.86): (tens) res in {9..17, 100..129, 199..200} | **res** (R2 0.81): res in {9..17} | - | same |
| L28 down c22 | 17% / 15% | **res%100** (R2 0.72): res mod 100 in {0..1, 7, 17, 37, 47, 57, 87, 91..99} | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {9,10}; a//10 in {1,2,3,4,8} -> b//10 in {9}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,9,10}; a//10 in {10} -> b//10 in {0} | - | same |
| L28 down c23 | 41% / 19% | **res%2** (R2 0.70): res mod 2 in {0} | **units(a,b)** (R2 0.65): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} | a: mod2 +15%; b: mod2 +11%; res: mod2 +11% | a: mod2 +7%; b: mod2 +7%; res: mod2 +4% |
| L28 down c24 | 12% / 15% | **res%100** (R2 0.81): res mod 100 in {6, 15..17, 26, 36, 46, 56, 66, 75..77, 86, 96} [coarser: res mod 20 in {6, 16}, R2 0.80] | unexplained (best R2 0.47) | res: mod4 +4% | res: mod4 +3% |
| L28 down c25 | 3% / 0 | **res%100** (R2 0.67): res mod 100: no class above 0.5 (max 0.42) | off (on 0) | - | same |
| L28 down c36 | 13% / 13% | **res%10** (R2 0.75): res mod 10 in {2} | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {8}; a%10 in {2} -> b%10 in {0,2}; a%10 in {3} -> b%10 in {1}; a%10 in {4} -> b%10 in {2}; a%10 in {5} -> b%10 in {3}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {5}; a%10 in {8} -> b%10 in {6}; a%10 in {9} -> b%10 in {7} | res: mod10 +3%, mod5 +3% | res: mod5 +2% |
| L28 down c40 | 3% / 1% | **res%100** (R2 0.73): res mod 100 in {1} | unexplained (best R2 0.32) | - | same |
| L28 down c46 | 8% / 5% | **res%100** (R2 0.73): res mod 100 in {29..33, 70, 80, 90} | unexplained (best R2 0.37) | - | same |
| L28 down c49 | 0 / 0 | off (on 0) | same | - | same |
| L28 down c51 | 5% / 2% | **res%100** (R2 0.75): res mod 100 in {18, 38, 58, 78, 98} [coarser: res mod 20 in {18}, R2 0.87] | unexplained (best R2 0.40) | - | res: mod4 +2% |
| L28 down c58 | 36% / 3% | **res** (R2 0.73): res in {15, 24..25, 35, 45, 65, 74..75, 84..85, 90, 94..95, 99, 104..105, 107..111, 114..115, 117, 119..127, 131, 134..135, 139..141, 144..145, 149..151, 155, 165, 169..171, 174..175, 179..181, 184..191, 194..195} | unexplained (best R2 0.37) | - | same |
| L28 down c62 | 13% / 12% | **res%100** (R2 0.52): res mod 100 in {80, 86..87, 90..92, 96} | **tens(a,b)** (R2 0.68): a//10 in {7,10} -> b//10 in {1}; a//10 in {8} -> b//10 in {0,1,2}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,10} | - | same |
| L28 down c67 | 13% / 7% | **res%100** (R2 0.85): res mod 100 in {7, 17, 27, 37, 47, 57, 67, 75..79, 87, 97} [coarser: res mod 50 in {7, 17, 27..28, 37, 47}, R2 0.88] | **res** (R2 0.66): res in {-23, -13, -3, 7, 17, 27, 37, 47, 57, 67, 77, 87, 97} | - | same |
| L28 down c69 | 10% / 11% | **res%10** (R2 0.93): res mod 10 in {6} | **res%100** (R2 0.46): res mod 100 in {6, 16, 26, 36, 46, 56, 66, 76, 86, 96} [coarser: res mod 50 in {6, 16, 26, 36, 46}, R2 0.81] | - | res: mod5 +2% |
| L28 down c70 | 11% / 24% | **res//10** (R2 0.75): (tens) res in {36..57} | **tens(a,b)** (R2 0.56): a//10 in {0} -> b//10 in {4,5}; a//10 in {1,3,10} -> b//10 in {5}; a//10 in {4} -> b//10 in {0,4,5,9,10}; a//10 in {5} -> b//10 in {0,1,5,6,9,10}; a//10 in {6} -> b//10 in {0,1,2}; a//10 in {7} -> b//10 in {1,2,3}; a//10 in {8} -> b//10 in {3,4}; a//10 in {9} -> b//10 in {4} | - | same |
| L28 down c71 | 14% / 3% | **res%100** (R2 0.83): res mod 100 in {26, 45..46, 60..69, 86} | unexplained (best R2 0.32) | - | same |
| L28 down c73 | 3% / 3% | **res** (R2 0.75): res in {2, 8..11, 30, 50, 70, 110} | **res%100** (R2 0.52): res mod 100 in {10..11} | - | res: mod4 +2% |
| L28 down c80 | 8% / 4% | **res%100** (R2 0.89): res mod 100 in {68..75} | unexplained (best R2 0.19) | - | same |
| L28 down c82 | 16% / 8% | **res%100** (R2 0.84): res mod 100 in {2..3, 13, 23, 33, 39..45, 53, 63, 73, 83, 93} | **res** (R2 0.60): res in {3, 13, 23, 33, 40..44, 53, 63, 73, 83, 93} | - | same |
| L28 down c83 | 12% / 3% | **res%100** (R2 0.85): res mod 100 in {8, 18, 28, 38, 48, 58, 68, 75..79, 98} | unexplained (best R2 0.48) | - | same |
| L28 down c90 | 14% / 4% | **res%100** (R2 0.58): res mod 100 in {12, 31..32, 52, 71..72, 90..92} | unexplained (best R2 0.29) | - | same |
| L28 down c91 | 11% / 1% | **res//10** (R2 0.79): (tens) res in {17, 157..200} | **res** (R2 0.63): res in {17} | - | same |
| L28 down c93 | 14% / 8% | **res%10** (R2 0.77): res mod 10 in {5} | **units(a,b)** (R2 0.62): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {6}; a%10 in {6} -> b%10 in {1}; a%10 in {9} -> b%10 in {4} | res: mod5 +3% | - |
| L28 down c98 | 6% / 27% | **res** (R2 0.77): res in {26..41} | **tens(a,b)** (R2 0.53): a//10 in {0} -> b//10 in {3}; a//10 in {1,6} -> b//10 in {2,3}; a//10 in {2} -> b//10 in {3,4,5}; a//10 in {3} -> b//10 in {0,3,4,5,6,9,10}; a//10 in {4} -> b//10 in {0,1,2,4,5,6}; a//10 in {5} -> b//10 in {2,6}; a//10 in {7} -> b//10 in {4}; a//10 in {8} -> b//10 in {5}; a//10 in {9,10} -> b//10 in {6} | - | same |
| L28 down c99 | 9% / 4% | **res%100** (R2 0.89): res mod 100 in {8, 28, 38, 48, 58, 68, 87..88, 98} [coarser: res mod 10 in {8}, R2 0.81] | **res** (R2 0.54): res in {8, 28, 38, 48, 88, 98} | - | same |
| L28 down c115 | 12% / 3% | **res%100** (R2 0.94): res mod 100 in {78..88} | unexplained (best R2 0.46) | - | same |
| L28 down c121 | 10% / 4% | **res%100** (R2 0.85): res mod 100 in {9, 19, 28..29, 39, 49, 59, 69, 79, 89} [coarser: res mod 50 in {9, 19, 28..29, 39, 49}, R2 0.85] | unexplained (best R2 0.42) | - | same |
| L28 down c124 | 3% / 0 | **res%100** (R2 0.74): res mod 100 in {52..53, 93} | off (on 0) | - | same |
| L28 down c129 | 9% / 5% | **res%10** (R2 0.87): res mod 10 in {9} | **res** (R2 0.57): res in {9, 19, 39, 49, 59, 69, 79, 89, 99} | - | same |
| L28 down c136 | 6% / 1% | **res** (R2 0.75): res in {54, 56..57, 151..158} [coarser: res mod 100 in {53..57}, R2 0.85] | unexplained (best R2 0.18) | - | same |
| L28 down c140 | 5% / 3% | **res%100** (R2 0.82): res mod 100 in {4, 64, 74, 84, 94} | unexplained (best R2 0.49) | - | same |
| L28 down c145 | 28% / 0 | **res//10** (R2 0.76): (tens) res in {121..170, 172..175, 178, 180..186, 194, 198, 200} | off (on 0) | - | same |
| L28 down c150 | 2% / 0 | **res%100** (R2 0.64): res mod 100 in {59} | off (on 0) | - | same |
| L28 down c160 | 10% / 6% | **res%10** (R2 0.93): res mod 10 in {7} | **res** (R2 0.66): res in {-97, -3, 7, 17, 27, 37, 47, 57, 67, 77, 87, 97} | - | same |
| L28 down c165 | 6% / 3% | **res%100** (R2 0.79): res mod 100 in {25..28, 66..68} | unexplained (best R2 0.48) | - | same |
| L28 down c176 | 6% / 1% | **res** (R2 0.71): res in {38..39, 89, 136..141, 179, 188..189} | unexplained (best R2 0.39) | - | same |
| L28 down c182 | 5% / 2% | **res%100** (R2 0.69): res mod 100 in {76..79} | unexplained (best R2 0.30) | - | same |
| L28 down c191 | 10% / 2% | **res%100** (R2 0.84): res mod 100 in {0..1, 51..53} | unexplained (best R2 0.47) | - | same |
| L28 down c203 | 1% / 4% | **res** (R2 0.68): res in {12..16} | **res** (R2 0.73): res in {13..16} | - | same |
| L28 down c211 | 5% / 6% | **res** (R2 0.60): res in {2, 6..14, 16..17, 21, 31, 51, 71, 111, 171, 191} | unexplained (best R2 0.26) | - | same |
| L28 down c218 | 8% / 0 | **res** (R2 0.93): res in {115..123, 179} | off (on 0) | - | same |
| L28 down c242 | 0 / 0 | off (on 0) | same | - | same |
| L28 down c247 | 1% / 5% | **res** (R2 0.69): res in {5..8, 107} | **res%100** (R2 0.65): res mod 100 in {7..8} | - | same |
| L28 down c259 | 4% / 0 | **res** (R2 0.70): res in {137..144} | off (on 0) | - | same |
| L28 down c268 | 8% / 2% | **res** (R2 0.80): res in {49..60, 155..156} | unexplained (best R2 0.40) | - | same |
| L28 down c271 | 10% / 5% | **res%10** (R2 0.96): res mod 10 in {3} | **res** (R2 0.67): res in {3, 13, 23, 33, 53, 63, 73, 83, 93} | - | same |
| L28 down c275 | 5% / 4% | **res%20** (R2 0.80): res mod 20 in {1} | unexplained (best R2 0.42) | - | same |
| L28 down c277 | 4% / 2% | **res%50** (R2 0.69): res mod 50 in {0, 49} | unexplained (best R2 0.20) | - | same |
| L28 down c288 | 4% / 1% | **res** (R2 0.89): res in {15, 113..116} | **res** (R2 0.60): res in {15} | - | same |
| L28 down c291 | 6% / 2% | **res%100** (R2 0.61): res mod 100 in {17, 57, 67, 77} | unexplained (best R2 0.40) | - | same |
| L28 down c294 | 3% / 0 | **res** (R2 0.79): res in {111..114} | off (on 0) | - | same |
| L28 down c296 | 5% / 3% | **res%100** (R2 0.67): res mod 100 in {0, 20, 40, 60, 80} [coarser: res mod 20 in {0}, R2 0.88] | unexplained (best R2 0.48) | - | same |
| L28 down c298 | 11% / 5% | **res%100** (R2 0.71): res mod 100 in {0, 49, 89..99} | unexplained (best R2 0.44) | - | same |
| L28 down c300 | 10% / 4% | **res%10** (R2 0.97): res mod 10 in {4} | **res** (R2 0.57): res in {14, 24, 34, 84, 94} | - | same |
| L28 down c301 | 3% / 0 | **res** (R2 0.81): res in {132, 134..136, 156} | off (on 0) | - | same |
| L28 down c304 | 6% / 21% | **res** (R2 0.69): res in {32..39, 133, 135..137} [coarser: res mod 100 in {33..39}, R2 0.90] | unexplained (best R2 0.46) | - | same |
| L28 down c309 | 1% / 0 | unexplained (best R2 0.46) | off (on 0) | - | same |
| L28 down c310 | 6% / 0 | **res** (R2 0.67): res in {107, 117, 127, 147, 157..158, 160..164, 167} | off (on 0) | - | same |
| L28 down c312 | 8% / 1% | **res//10** (R2 0.80): (tens) res in {70..79, 171, 173, 175} | **tens(a,b)** (R2 0.58): a//10 in {7} -> b//10 in {0} | - | same |
| L28 down c316 | 1% / 0 | **res** (R2 0.71): res in {154} [coarser: res mod 100 in {54}, R2 0.86] | off (on 0) | - | same |
| L28 down c338 | 10% / 0 | **res//10** (R2 0.78): (tens) res in {100..109, 194..200} | off (on 0) | - | same |
| L28 down c357 | 0 / 1% | off (on 0) | **res%100** (R2 0.68): res mod 100 in {2} | - | same |
| L28 down c359 | 11% / 7% | **units(a,b)** (R2 0.88): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {9}; a%10 in {2} -> b%10 in {8}; a%10 in {3} -> b%10 in {7}; a%10 in {4} -> b%10 in {6}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {3}; a%10 in {8} -> b%10 in {2}; a%10 in {9} -> b%10 in {1} | **units(a,b)** (R2 0.56): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {1} | - | same |
| L28 down c377 | 8% / 1% | **res** (R2 0.80): res in {34, 103..104, 132..136, 144, 153..154, 174} | unexplained (best R2 0.45) | - | same |
| L28 down c408 | 0 / 1% | off (on 0) | **res%100** (R2 0.66): res mod 100: no class above 0.5 (max 0.48) | - | same |
| L28 down c416 | 1% / 9% | **res%100** (R2 0.77): res mod 100 in {9} | **res%100** (R2 0.63): res mod 100: no class above 0.5 (max 0.47) | - | same |
| L28 down c430 | 0 / 1% | off (on 0) | **res%100** (R2 0.77): res mod 100: no class above 0.5 (max 0.44) | - | same |
| L28 down c433 | 5% / 0 | **res%100** (R2 0.66): res mod 100: no class above 0.5 (max 0.46) | off (on 0) | - | same |
| L28 down c435 | 1% / 2% | **res%100** (R2 0.89): res mod 100 in {8} | **res%100** (R2 0.51): res mod 100 in {8} | - | same |
| L28 down c462 | 1% / 0 | **res** (R2 0.71): res in {186..188, 190..200} | off (on 0) | - | same |
| L28 down c517 | 4% / 6% | **res** (R2 0.51): res in {12, 20, 102, 112, 132, 191..192, 198..200} | unexplained (best R2 0.38) | - | same |
| L28 down c522 | 2% / 1% | **res%100** (R2 0.74): res mod 100 in {17, 47} | **res** (R2 0.61): res in {17} | - | same |
| L28 down c543 | 7% / 0 | **res** (R2 0.78): res in {56, 85..88, 136..137, 146, 156..157, 185..188} [coarser: res mod 100 in {56, 85..88}, R2 0.83] | off (on 0) | - | same |
| L28 down c545 | 1% / 0 | **res%100** (R2 0.53): res mod 100: no class above 0.5 (max 0.40) | off (on 0) | - | same |
| L28 down c584 | 0 / 3% | off (on 0) | unexplained (best R2 0.38) | - | same |
| L28 down c588 | 1% / 3% | unexplained (best R2 0.40) | **res%100** (R2 0.50): res mod 100: no class above 0.5 (max 0.48) | - | same |
| L28 down c593 | 2% / 0 | **res** (R2 0.81): res in {40..44} | off (on 0) | - | same |
| L28 down c595 | 5% / 1% | **res%100** (R2 0.73): res mod 100 in {42..47} | unexplained (best R2 0.21) | - | same |
| L28 down c638 | 1% / 0 | **res//10** (R2 0.89): (tens) res in {190..200} | off (on 0) | - | same |
| L28 down c660 | 2% / 2% | **res%100** (R2 0.52): res mod 100 in {82} | **res%100** (R2 0.66): res mod 100 in {0} | - | same |
| L28 down c666 | 5% / 0 | **res** (R2 0.85): res in {151..164} | off (on 0) | - | same |
| L28 down c719 | 4% / 1% | **res%100** (R2 0.81): res mod 100 in {32, 52, 72, 92} | unexplained (best R2 0.33) | - | same |
| L28 down c744 | 0 / 0 | off (on 0) | same | - | same |
| L28 down c752 | 2% / 0 | **res%100** (R2 0.69): res mod 100 in {61..62} | off (on 0) | - | same |
| L28 down c760 | 1% / 0 | **res** (R2 0.70): res in {50, 150, 170, 180} | off (on 0) | - | same |
| L28 down c794 | 4% / 7% | **res%100** (R2 0.69): res mod 100 in {21..24} | unexplained (best R2 0.46) | - | same |
| L28 down c903 | 2% / 0 | **res%100** (R2 0.71): res mod 100 in {72} | off (on 0) | - | same |
| L28 down c952 | 2% / 0 | **res%100** (R2 0.89): res mod 100 in {69} | off (on 0) | - | same |
| L28 down c965 | 2% / 0 | **res%100** (R2 0.58): res mod 100 in {71, 91} | off (on 0) | - | same |
| L28 down c979 | 5% / 0 | **res** (R2 0.83): res in {144..152} | off (on 0) | - | same |
| L28 down c990 | 1% / 3% | **res** (R2 0.72): res in {24..28} | unexplained (best R2 0.49) | - | same |
| L28 down c1010 | 2% / 3% | **res** (R2 0.85): res in {7..9, 11..20} | **res** (R2 0.51): res in {15..17} | - | same |

</details>

<details><summary>gate: 80</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L28 gate c10 | 11% / 8% | **res//10** (R2 0.91): (tens) res in {10..19, 110..119} [coarser: res mod 100 in {10..19}, R2 1.00] | **res//10** (R2 0.76): (tens) res in {11..19} | (reads) | same |
| L28 gate c12 | 27% / 10% | **res//10** (R2 0.86): (tens) res in {2, 8..18, 20, 100..128, 198..200} [coarser: res mod 100 in {0..2, 6, 8..26, 99}, R2 0.81] | **res** (R2 0.81): res in {8..18} | (reads) | same |
| L28 gate c22 | 15% / 15% | **res%100** (R2 0.77): res mod 100 in {0..1, 90..99} | **tens(a,b)** (R2 0.62): a//10 in {0} -> b//10 in {9,10}; a//10 in {1,2,3,4,8} -> b//10 in {9}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,9,10}; a//10 in {10} -> b//10 in {0,10} | (reads) | same |
| L28 gate c23 | 42% / 21% | **res%10** (R2 0.74): res mod 10 in {0, 2, 4, 6, 8} | **units(a,b)** (R2 0.64): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} | (reads) | same |
| L28 gate c24 | 14% / 7% | **res** (R2 0.76): res in {16..18, 26, 36, 76..78, 96, 116..118, 126, 136, 156, 166..186, 188, 196..198, 200} | **res** (R2 0.52): res in {16..18, 26, 36, 76..78, 96} | (reads) | same |
| L28 gate c36 | 12% / 14% | **res%10** (R2 0.82): res mod 10 in {2} | **units(a,b)** (R2 0.50): a%10 in {0} -> b%10 in {8}; a%10 in {2} -> b%10 in {0,2,4,8}; a%10 in {3} -> b%10 in {1}; a%10 in {4} -> b%10 in {2}; a%10 in {5} -> b%10 in {3,7}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {5}; a%10 in {8} -> b%10 in {6}; a%10 in {9} -> b%10 in {7} | (reads) | same |
| L28 gate c46 | 10% / 9% | **res%100** (R2 0.70): res mod 100 in {28..35, 80} | unexplained (best R2 0.46) | (reads) | same |
| L28 gate c49 | 24% / 75% | **res** (R2 0.72): res in {2, 6, 8..16, 18, 26, 33..66, 68, 76, 81, 86, 96, 141, 146, 156, 161, 166, 176, 181, 196, 198} | unexplained (best R2 0.36) | (reads) | same |
| L28 gate c51 | 8% / 3% | **res%100** (R2 0.83): res mod 100 in {8, 18, 28, 38, 58, 68, 78, 88, 98} [coarser: res mod 20 in {8, 18}, R2 0.80] | **res** (R2 0.59): res in {8, 18} | (reads) | same |
| L28 gate c67 | 10% / 5% | **res%10** (R2 0.97): res mod 10 in {7} | **res** (R2 0.65): res in {7, 17, 27, 37, 77, 87, 97} | (reads) | same |
| L28 gate c69 | 10% / 8% | **res%10** (R2 0.96): res mod 10 in {6} | **units(a,b)** (R2 0.56): a%10 in {0} -> b%10 in {4}; a%10 in {1} -> b%10 in {5}; a%10 in {2} -> b%10 in {6}; a%10 in {4} -> b%10 in {8}; a%10 in {5} -> b%10 in {9}; a%10 in {6} -> b%10 in {0}; a%10 in {7} -> b%10 in {1}; a%10 in {8} -> b%10 in {2}; a%10 in {9} -> b%10 in {3} | (reads) | same |
| L28 gate c70 | 18% / 18% | **res//10** (R2 0.75): (tens) res in {34..59, 140..147} | **tens(a,b)** (R2 0.53): a//10 in {0,10} -> b//10 in {4,5}; a//10 in {4} -> b//10 in {0,4,5,9,10}; a//10 in {5} -> b//10 in {0,1,5,6,9,10}; a//10 in {6} -> b//10 in {1,2}; a//10 in {7} -> b//10 in {3}; a//10 in {8,9} -> b//10 in {4} | (reads) | same |
| L28 gate c71 | 19% / 5% | **res%100** (R2 0.89): res mod 100 in {6, 16, 26, 36, 45..46, 56, 60..68, 76, 86, 96} | **res** (R2 0.52): res in {6, 16, 26, 46, 86, 96} | (reads) | same |
| L28 gate c73 | 4% / 2% | **res%100** (R2 0.68): res mod 100 in {10, 50, 70} | **res%100** (R2 0.51): res mod 100 in {10} | (reads) | same |
| L28 gate c80 | 12% / 4% | **res%100** (R2 0.90): res mod 100 in {65..75} | unexplained (best R2 0.27) | (reads) | same |
| L28 gate c82 | 14% / 5% | **res%100** (R2 0.89): res mod 100 in {3, 13, 23, 33, 40..44, 53, 63, 73, 83, 93} [coarser: res mod 50 in {3, 13, 23, 33, 43}, R2 0.81] | **res** (R2 0.68): res in {3, 13, 23, 33, 43, 83, 93} | (reads) | same |
| L28 gate c83 | 9% / 4% | **res%10** (R2 0.89): res mod 10 in {8} | **res** (R2 0.63): res in {8, 18, 28, 88, 98} | (reads) | same |
| L28 gate c90 | 20% / 9% | **res%100** (R2 0.73): res mod 100 in {11..17, 30..32, 34, 36, 52, 54, 56, 71..72, 76, 91..92, 94, 96} | **res** (R2 0.69): res in {11..17, 36, 96} | (reads) | same |
| L28 gate c91 | 11% / 0 | **res//10** (R2 0.85): (tens) res in {156..200} | off (on 0) | (reads) | same |
| L28 gate c93 | 11% / 5% | **units(a,b)** (R2 0.91): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {4}; a%10 in {2} -> b%10 in {3}; a%10 in {3} -> b%10 in {2}; a%10 in {4} -> b%10 in {1}; a%10 in {6} -> b%10 in {9}; a%10 in {7} -> b%10 in {8}; a%10 in {8} -> b%10 in {7}; a%10 in {9} -> b%10 in {6} | **units(a,b)** (R2 0.56): a%10 in {0} -> b%10 in {5}; a%10 in {5} -> b%10 in {0,5} | (reads) | same |
| L28 gate c99 | 0 / 0 | off (on 0) | same | (reads) | same |
| L28 gate c112 | 2% / 0 | **res%100** (R2 0.80): res mod 100 in {61, 81} | off (on 0) | (reads) | same |
| L28 gate c115 | 16% / 6% | **res%100** (R2 0.91): res mod 100 in {18, 74..88} | **res** (R2 0.55): res in {18, 74..88} | (reads) | same |
| L28 gate c121 | 6% / 1% | **res%100** (R2 0.79): res mod 100 in {29..30, 59, 79, 89} | **res** (R2 0.54): res in {29..30, 99} | (reads) | same |
| L28 gate c124 | 4% / 0 | **res%100** (R2 0.83): res mod 100 in {53..55, 93} | off (on 0) | (reads) | same |
| L28 gate c125 | 2% / 0 | **res%100** (R2 0.86): res mod 100 in {54..55} | off (on 0) | (reads) | same |
| L28 gate c129 | 11% / 6% | **res%10** (R2 0.86): res mod 10 in {9} | **res** (R2 0.58): res in {-99, -11, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99} | (reads) | same |
| L28 gate c133 | 0 / 1% | off (on 0) | **res%100** (R2 0.76): res mod 100: no class above 0.5 (max 0.40) | (reads) | same |
| L28 gate c136 | 10% / 2% | **res//10** (R2 0.85): (tens) res in {50..60, 151..159} [coarser: res mod 100 in {51..59}, R2 0.99] | unexplained (best R2 0.26) | (reads) | same |
| L28 gate c140 | 10% / 4% | **res%10** (R2 0.95): res mod 10 in {4} | **res** (R2 0.60): res in {4, 14, 24, 34, 84, 94} | (reads) | same |
| L28 gate c160 | 10% / 7% | **res%10** (R2 0.94): res mod 10 in {7} | **res** (R2 0.67): res in {-97, -13, -3, 7, 17, 27, 37, 47, 57, 67, 77, 87, 97} | (reads) | same |
| L28 gate c165 | 12% / 5% | **res%100** (R2 0.83): res mod 100 in {24..31, 66..70} | **res** (R2 0.53): res in {25..31} | (reads) | same |
| L28 gate c176 | 5% / 2% | **res** (R2 0.81): res in {18, 38..39, 118, 136..141} [coarser: res mod 100 in {18, 37..40}, R2 0.81] | **res** (R2 0.53): res in {18} | (reads) | same |
| L28 gate c182 | 6% / 1% | **res%100** (R2 0.83): res mod 100 in {75..79} | unexplained (best R2 0.28) | (reads) | same |
| L28 gate c191 | 2% / 0 | **res%100** (R2 0.72): res mod 100 in {1, 51, 55} [coarser: res mod 50 in {1}, R2 0.84] | off (on 0) | (reads) | same |
| L28 gate c203 | 33% / 0 | **res//10** (R2 0.93): (tens) res in {120..190, 192..200} | off (on 0) | (reads) | same |
| L28 gate c211 | 10% / 4% | **res%100** (R2 0.72): res mod 100 in {8..12, 31, 51, 71, 91} | **res%100** (R2 0.58): res mod 100 in {10..11} | (reads) | same |
| L28 gate c218 | 1% / 0 | **res** (R2 0.70): res in {112..113} | off (on 0) | (reads) | same |
| L28 gate c259 | 1% / 0 | **res** (R2 0.67): res in {139..140} | off (on 0) | (reads) | same |
| L28 gate c268 | 11% / 2% | **res//10** (R2 0.80): (tens) res in {50..60, 151..159} [coarser: res mod 100 in {50..60}, R2 0.96] | unexplained (best R2 0.33) | (reads) | same |
| L28 gate c271 | 10% / 6% | **res%10** (R2 0.94): res mod 10 in {3} | **res** (R2 0.68): res in {3, 13, 23, 33, 43, 53, 63, 73, 83, 93} | (reads) | same |
| L28 gate c275 | 5% / 2% | **res%20** (R2 0.68): res mod 20 in {1} | **res** (R2 0.55): res in {1, 18, 41} | (reads) | same |
| L28 gate c291 | 5% / 1% | **res%100** (R2 0.83): res mod 100 in {27, 47, 57, 67, 77, 97} [coarser: res mod 50 in {27, 47}, R2 0.87] | unexplained (best R2 0.48) | (reads) | same |
| L28 gate c294 | 4% / 3% | **res%100** (R2 0.88): res mod 100 in {12..15} | **res** (R2 0.81): res in {12..15} | (reads) | same |
| L28 gate c296 | 7% / 3% | **res%100** (R2 0.77): res mod 100 in {0, 20, 40, 59..60, 80} [coarser: res mod 20 in {0}, R2 0.87] | **res%100** (R2 0.54): res mod 100 in {0} | (reads) | same |
| L28 gate c297 | 6% / 1% | **res%100** (R2 0.91): res mod 100 in {69..74} | unexplained (best R2 0.17) | (reads) | same |
| L28 gate c298 | 11% / 0 | **res** (R2 0.86): res in {66, 157..190, 198..200} | off (on 0) | (reads) | same |
| L28 gate c301 | 4% / 0 | **res%100** (R2 0.82): res mod 100 in {54..55, 65..66} | off (on 0) | (reads) | same |
| L28 gate c304 | 8% / 23% | **res%100** (R2 0.80): res mod 100 in {32..39} | unexplained (best R2 0.48) | (reads) | same |
| L28 gate c312 | 9% / 3% | **res//10** (R2 0.85): (tens) res in {70..79, 171..179} [coarser: res mod 100 in {70..79}, R2 0.98] | unexplained (best R2 0.33) | (reads) | same |
| L28 gate c315 | 1% / 1% | **res%100** (R2 0.94): res mod 100 in {18} | **res** (R2 0.68): res in {18} | (reads) | same |
| L28 gate c316 | 2% / 0 | **res%100** (R2 0.81): res mod 100 in {54..55} | off (on 0) | (reads) | same |
| L28 gate c338 | 1% / 1% | **res** (R2 0.61): res in {112..113} | **res** (R2 0.81): res in {10} | (reads) | same |
| L28 gate c357 | 10% / 3% | **res%10** (R2 0.91): res mod 10 in {2} | **res** (R2 0.55): res in {2, 22} | (reads) | same |
| L28 gate c359 | 12% / 11% | **units(a,b)** (R2 0.97): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {9}; a%10 in {2} -> b%10 in {8}; a%10 in {3} -> b%10 in {7}; a%10 in {4} -> b%10 in {6}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {3}; a%10 in {8} -> b%10 in {2}; a%10 in {9} -> b%10 in {1} | **units(a,b)** (R2 0.62): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {1}; a%10 in {3} -> b%10 in {3}; a%10 in {4} -> b%10 in {4}; a%10 in {6} -> b%10 in {6}; a%10 in {8} -> b%10 in {8}; a%10 in {9} -> b%10 in {9} | (reads) | same |
| L28 gate c377 | 2% / 0 | **res** (R2 0.86): res in {130, 134..135} | off (on 0) | (reads) | same |
| L28 gate c416 | 0 / 1% | off (on 0) | **res** (R2 0.75): res in {10} | (reads) | same |
| L28 gate c433 | 11% / 9% | **res%100** (R2 0.89): res mod 100 in {0, 89..99} | **tens(a,b)** (R2 0.60): a//10 in {0} -> b//10 in {9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5}; a//10 in {10} -> b//10 in {0} | (reads) | same |
| L28 gate c434 | 9% / 3% | **res%100** (R2 0.78): res mod 100 in {44..52} | unexplained (best R2 0.29) | (reads) | same |
| L28 gate c435 | 9% / 3% | **res%10** (R2 0.93): res mod 10 in {8} | **res** (R2 0.59): res in {8, 18, 98} | (reads) | same |
| L28 gate c473 | 1% / 0 | **res%100** (R2 0.90): res mod 100 in {30} | off (on 0) | (reads) | same |
| L28 gate c517 | 2% / 0 | **res%100** (R2 0.87): res mod 100 in {54..55} | off (on 0) | (reads) | same |
| L28 gate c522 | 2% / 0 | **res** (R2 0.70): res in {47, 147..149} | off (on 0) | (reads) | same |
| L28 gate c526 | 6% / 13% | **res** (R2 0.61): res in {2..12, 14..15, 17, 19..21, 25, 104, 106, 200} | **res%100** (R2 0.60): res mod 100 in {4..9} | (reads) | same |
| L28 gate c543 | 9% / 1% | **res%10** (R2 0.83): res mod 10 in {6} | unexplained (best R2 0.14) | (reads) | same |
| L28 gate c588 | 0 / 1% | off (on 0) | **res%100** (R2 0.70): res mod 100: no class above 0.5 (max 0.46) | (reads) | same |
| L28 gate c593 | 7% / 3% | **res%100** (R2 0.96): res mod 100 in {38..44} | unexplained (best R2 0.46) | (reads) | same |
| L28 gate c602 | 1% / 6% | **res** (R2 0.77): res in {4..5, 18, 30} | **res%100** (R2 0.59): res mod 100: no class above 0.5 (max 0.48) | (reads) | same |
| L28 gate c612 | 1% / 1% | **res%100** (R2 0.93): res mod 100 in {18} | **res** (R2 0.69): res in {18} | (reads) | same |
| L28 gate c638 | 61% / 2% | **tens(a,b)** (R2 0.81): a//10 in {0} -> b//10 in {8,9,10}; a//10 in {1} -> b//10 in {7,8,9,10}; a//10 in {2} -> b//10 in {6,7,8,9,10}; a//10 in {3} -> b//10 in {5,6,7,8,9,10}; a//10 in {4} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {1,2,3,4,5,6,7,8}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,7,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | unexplained (best R2 0.34) | (reads) | same |
| L28 gate c660 | 10% / 5% | **res%20** (R2 0.78): res mod 20 in {1..2} | **res** (R2 0.58): res in {1..2, 21..22, 81} | (reads) | same |
| L28 gate c706 | 2% / 0 | **res%100** (R2 0.79): res mod 100 in {54..55} | off (on 0) | (reads) | same |
| L28 gate c719 | 9% / 0 | **res** (R2 0.52): res in {112, 116, 132, 134, 136, 138, 142, 144, 146, 152, 156, 176, 182, 192, 196} | off (on 0) | (reads) | same |
| L28 gate c758 | 2% / 0 | **res%100** (R2 0.86): res mod 100 in {54..55} | off (on 0) | (reads) | same |
| L28 gate c760 | 0 / 1% | off (on 0) | **res%100** (R2 0.68): res mod 100: no class above 0.5 (max 0.42) | (reads) | same |
| L28 gate c788 | 1% / 1% | **res%100** (R2 0.94): res mod 100 in {18} | **res** (R2 0.68): res in {18} | (reads) | same |
| L28 gate c794 | 12% / 4% | **res** (R2 0.84): res in {18, 22..24, 29..30, 118, 121..131, 133..135} | **res** (R2 0.54): res in {18, 21..24, 29..30} | (reads) | same |
| L28 gate c798 | 2% / 0 | **res** (R2 0.72): res in {170..174} | off (on 0) | (reads) | same |
| L28 gate c867 | 1% / 0 | **res** (R2 0.77): res in {113} | off (on 0) | (reads) | same |
| L28 gate c903 | 10% / 3% | **res%10** (R2 0.93): res mod 10 in {2} | **res** (R2 0.55): res in {12, 22, 32} | (reads) | same |

</details>

<details><summary>up: 65</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L28 up c10 | 11% / 9% | **res//10** (R2 0.85): (tens) res in {10..19, 110..119} [coarser: res mod 100 in {10..19}, R2 0.99] | **res//10** (R2 0.74): (tens) res in {10..18} | (reads) | same |
| L28 up c12 | 38% / 11% | **res** (R2 0.83): res in {10..13, 99..137, 139..141, 151, 153, 156..157, 159..160, 163, 176, 198..200} | **res** (R2 0.69): res in {-99, 6..18, 99} | (reads) | same |
| L28 up c22 | 15% / 15% | **res%100** (R2 0.83): res mod 100 in {0..1, 90..99} | **tens(a,b)** (R2 0.60): a//10 in {0} -> b//10 in {9,10}; a//10 in {1,2,3,4,8} -> b//10 in {9}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,9,10}; a//10 in {10} -> b//10 in {0,10} | (reads) | same |
| L28 up c23 | 28% / 0 | **res//10** (R2 0.76): (tens) res in {86, 88, 126..176, 179..188, 190..198, 200} | off (on 0) | (reads) | same |
| L28 up c24 | 12% / 21% | **res%10** (R2 0.81): res mod 10 in {6} | unexplained (best R2 0.39) | (reads) | same |
| L28 up c36 | 96% / 5% | always | unexplained (best R2 0.27) | (reads) | same |
| L28 up c46 | 10% / 5% | **res%100** (R2 0.88): res mod 100 in {0, 20, 30..31, 40, 50, 60, 70, 80, 90} [coarser: res mod 50 in {0, 10, 20, 30..31, 40}, R2 0.83] | unexplained (best R2 0.43) | (reads) | same |
| L28 up c51 | 0 / 1% | off (on 0) | **res** (R2 0.60): res in {18} | (reads) | same |
| L28 up c67 | 13% / 11% | **res%50** (R2 0.80): res mod 50 in {7, 17, 26..27, 37, 47} [coarser: res mod 10 in {7}, R2 0.80] | **res** (R2 0.63): res in {-97, -23, -13, -3, 7, 17, 25..28, 37, 47, 57, 67, 77, 87, 97} | (reads) | same |
| L28 up c69 | 1% / 0 | **res%100** (R2 0.73): res mod 100 in {76} | off (on 0) | (reads) | same |
| L28 up c71 | 14% / 3% | **res%100** (R2 0.85): res mod 100 in {26, 45..46, 60..69} | unexplained (best R2 0.25) | (reads) | same |
| L28 up c73 | 0 / 0 | off (on 0) | same | (reads) | same |
| L28 up c74 | 0 / 3% | off (on 0) | **res%100** (R2 0.74): res mod 100: no class above 0.5 (max 0.45) | (reads) | same |
| L28 up c80 | 13% / 4% | **res%100** (R2 0.93): res mod 100 in {67..79} | unexplained (best R2 0.31) | (reads) | same |
| L28 up c82 | 12% / 8% | **res%10** (R2 0.79): res mod 10 in {3} | **res** (R2 0.61): res in {-17, -7, 3, 13, 23, 33, 43, 53, 63, 73, 83, 93} | (reads) | same |
| L28 up c83 | 10% / 1% | **res%100** (R2 0.74): res mod 100 in {8, 18, 28, 38, 58, 68, 77..79, 98} | unexplained (best R2 0.18) | (reads) | same |
| L28 up c89 | 100% / 100% | always | same | (reads) | same |
| L28 up c90 | 1% / 0 | **res%100** (R2 0.75): res mod 100 in {43} | off (on 0) | (reads) | same |
| L28 up c91 | 10% / 0 | **res//10** (R2 0.80): (tens) res in {153..155, 158..200} | off (on 0) | (reads) | same |
| L28 up c93 | 13% / 9% | **units(a,b)** (R2 0.91): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {4}; a%10 in {2} -> b%10 in {3}; a%10 in {3} -> b%10 in {2}; a%10 in {4} -> b%10 in {1}; a%10 in {6} -> b%10 in {9}; a%10 in {7} -> b%10 in {8}; a%10 in {8} -> b%10 in {7}; a%10 in {9} -> b%10 in {6} | **units(a,b)** (R2 0.65): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {6}; a%10 in {4} -> b%10 in {9}; a%10 in {6} -> b%10 in {1}; a%10 in {8} -> b%10 in {3}; a%10 in {9} -> b%10 in {4} | (reads) | same |
| L28 up c98 | 13% / 27% | **res** (R2 0.76): res in {26..41, 126..133} [coarser: res mod 100 in {26..38}, R2 0.87] | **tens(a,b)** (R2 0.51): a//10 in {0} -> b//10 in {3}; a//10 in {1,6} -> b//10 in {2,3}; a//10 in {2} -> b//10 in {3,4,5}; a//10 in {3} -> b//10 in {0,3,4,5,6,9,10}; a//10 in {4} -> b//10 in {0,1,2,5,6}; a//10 in {5} -> b//10 in {2,6}; a//10 in {7} -> b//10 in {4}; a//10 in {8} -> b//10 in {5}; a//10 in {9} -> b//10 in {6}; a//10 in {10} -> b//10 in {6,7} | (reads) | same |
| L28 up c99 | 13% / 5% | **res%50** (R2 0.80): res mod 50 in {8, 18, 28, 38, 48} [coarser: res mod 10 in {8}, R2 0.82] | **res** (R2 0.58): res in {8, 18, 28, 38, 48, 78, 88, 98} | (reads) | same |
| L28 up c101 | 11% / 5% | **units(a,b)** (R2 0.53): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {4}; a%10 in {9} -> b%10 in {6} | **units(a,b)** (R2 0.70): a%10 in {0,5} -> b%10 in {0,5} | (reads) | same |
| L28 up c114 | 3% / 0 | **res%100** (R2 0.88): res mod 100 in {94, 96..98} | off (on 0) | (reads) | same |
| L28 up c115 | 13% / 4% | **res%100** (R2 0.94): res mod 100 in {77..89} | **tens(a,b)** (R2 0.54): a//10 in {8} -> b//10 in {0,1,10}; a//10 in {9,10} -> b//10 in {1} | (reads) | same |
| L28 up c121 | 4% / 2% | **res%100** (R2 0.81): res mod 100 in {15..16, 76} | **res** (R2 0.75): res in {15..16} | (reads) | same |
| L28 up c129 | 9% / 4% | **res%10** (R2 0.87): res mod 10 in {9} | **res** (R2 0.57): res in {9, 19, 29, 79, 89, 99} | (reads) | same |
| L28 up c136 | 13% / 1% | **res** (R2 0.74): res in {45, 55, 65, 75, 85, 95, 105, 125, 134..135, 142, 144..151, 154..155, 165, 174..175, 185, 194..195, 198} | unexplained (best R2 0.14) | (reads) | same |
| L28 up c165 | 6% / 1% | **res%100** (R2 0.81): res mod 100 in {28, 38, 58, 68, 78} | unexplained (best R2 0.38) | (reads) | same |
| L28 up c176 | 7% / 3% | **res** (R2 0.82): res in {39, 79, 89, 119, 129, 137..140, 159, 179, 189} | unexplained (best R2 0.43) | (reads) | same |
| L28 up c182 | 7% / 3% | **res%100** (R2 0.81): res mod 100 in {74..80} | unexplained (best R2 0.35) | (reads) | same |
| L28 up c191 | 14% / 4% | **res%100** (R2 0.76): res mod 100 in {0..2, 6, 50..53} | **res%100** (R2 0.52): res mod 100 in {1} | (reads) | same |
| L28 up c203 | 6% / 4% | **res%100** (R2 0.82): res mod 100 in {12..16} | **res** (R2 0.75): res in {12..16} | (reads) | same |
| L28 up c218 | 8% / 4% | **res** (R2 0.82): res in {18..20, 116..123, 160, 178, 180} | **res** (R2 0.61): res in {18..22} | (reads) | same |
| L28 up c268 | 12% / 3% | **res%100** (R2 0.83): res mod 100 in {50..60} | unexplained (best R2 0.38) | (reads) | same |
| L28 up c271 | 1% / 0 | **res** (R2 0.73): res in {113} | off (on 0) | (reads) | same |
| L28 up c277 | 5% / 2% | **res%50** (R2 0.72): res mod 50 in {0, 49} | unexplained (best R2 0.39) | (reads) | same |
| L28 up c291 | 2% / 0 | **res%50** (R2 0.89): res mod 50 in {47} | off (on 0) | (reads) | same |
| L28 up c294 | 14% / 5% | **res%100** (R2 0.83): res mod 100 in {0, 8..18} | **res** (R2 0.74): res in {10..15} | (reads) | same |
| L28 up c296 | 7% / 3% | **res** (R2 0.75): res in {30, 40, 60, 80, 120, 130, 140, 157, 159..163, 180} [coarser: res mod 100 in {20, 30, 40, 57, 59..60, 80}, R2 0.89] | **res%100** (R2 0.52): res mod 100 in {0} | (reads) | same |
| L28 up c298 | 1% / 0 | **res%100** (R2 0.75): res mod 100 in {43} | off (on 0) | (reads) | same |
| L28 up c300 | 9% / 3% | **res%10** (R2 0.92): res mod 10 in {4} | **res** (R2 0.64): res in {4, 14, 24, 34, 94} | (reads) | same |
| L28 up c303 | 1% / 0 | **res** (R2 0.74): res in {143} [coarser: res mod 100 in {43}, R2 0.87] | off (on 0) | (reads) | same |
| L28 up c310 | 3% / 0 | **res** (R2 0.80): res in {50, 147..152} | off (on 0) | (reads) | same |
| L28 up c312 | 4% / 0 | **res%100** (R2 0.77): res mod 100 in {70..74} | off (on 0) | (reads) | same |
| L28 up c338 | 6% / 0 | **res** (R2 0.85): res in {107..114} | off (on 0) | (reads) | same |
| L28 up c359 | 3% / 0 | **res%100** (R2 0.72): res mod 100 in {96..97} | off (on 0) | (reads) | same |
| L28 up c377 | 9% / 2% | **res%10** (R2 0.84): res mod 10 in {5} | **res** (R2 0.53): res in {15, 35, 85, 95} | (reads) | same |
| L28 up c416 | 4% / 7% | **res%100** (R2 0.55): res mod 100 in {8, 32} | **res%100** (R2 0.64): res mod 100: no class above 0.5 (max 0.45) | (reads) | same |
| L28 up c434 | 1% / 0 | **res** (R2 0.88): res in {129} | off (on 0) | (reads) | same |
| L28 up c435 | 5% / 20% | **res** (R2 0.66): res in {2..12, 20, 103, 106..109, 200} | **res//10** (R2 0.52): (tens) res in {-5, 0..13, 99} | (reads) | same |
| L28 up c452 | 1% / 0 | **res%100** (R2 0.79): res mod 100 in {76} | off (on 0) | (reads) | same |
| L28 up c462 | 1% / 0 | **res** (R2 0.76): res in {192..200} | off (on 0) | (reads) | same |
| L28 up c517 | 8% / 46% | **res%10** (R2 0.71): res mod 10 in {5} | unexplained (best R2 0.35) | (reads) | same |
| L28 up c522 | 10% / 7% | **res%10** (R2 0.97): res mod 10 in {7} | **res** (R2 0.69): res in {-13, -3, 7, 17, 27, 37, 47, 57, 67, 77, 87, 97} | (reads) | same |
| L28 up c588 | 3% / 2% | **res%100** (R2 0.56): res mod 100 in {5, 55, 95} | **res%100** (R2 0.61): res mod 100 in {95} | (reads) | same |
| L28 up c595 | 12% / 5% | **res%100** (R2 0.91): res mod 100 in {38..48} | **res** (R2 0.51): res in {39..47} | (reads) | same |
| L28 up c666 | 0 / 0 | off (on 0) | same | (reads) | same |
| L28 up c667 | 4% / 2% | **res%100** (R2 0.79): res mod 100 in {30, 60, 80, 90} [coarser: res mod 50 in {30}, R2 0.84] | unexplained (best R2 0.39) | (reads) | same |
| L28 up c729 | 1% / 0 | **res%100** (R2 0.78): res mod 100 in {29} | off (on 0) | (reads) | same |
| L28 up c733 | 16% / 3% | **res** (R2 0.78): res in {14, 24, 34, 44, 54, 74, 104, 113..115, 124..125, 130, 133..135, 142..145, 153..155, 174, 194} | unexplained (best R2 0.48) | (reads) | same |
| L28 up c746 | 3% / 2% | **res%100** (R2 0.88): res mod 100 in {30, 80, 90} [coarser: res mod 50 in {30}, R2 0.83] | **res** (R2 0.57): res in {0, 30, 90} | (reads) | same |
| L28 up c749 | 1% / 0 | **res%100** (R2 0.74): res mod 100 in {29} | off (on 0) | (reads) | same |
| L28 up c760 | 1% / 0 | **res%100** (R2 0.82): res mod 100 in {30} | off (on 0) | (reads) | same |
| L28 up c848 | 1% / 0 | **res%100** (R2 0.74): res mod 100 in {76} | off (on 0) | (reads) | same |

</details>

<details><summary>v: 2</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L28 v c42 (kv4) | 0 / 0 | off (on 0) | same | (reads) | same |
| L28 v c236 (kv4) | 0 / 0 | off (on 0) | same | (reads) | same |

</details>

</details>

<details><summary>layer 29: 358 components with main position `=`</summary>

<details><summary>down: 116</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L29 down c4 | 100% / 100% | always | same | - | same |
| L29 down c6 | 95% / 98% | unexplained (best R2 0.20) | always | - | same |
| L29 down c11 | 74% / 2% | unexplained (best R2 0.43) | unexplained (best R2 0.17) | a: mod100 +3% | - |
| L29 down c12 | 6% / 92% | unexplained (best R2 0.23) | unexplained (best R2 0.29) | - | same |
| L29 down c14 | 5% / 0 | **res** (R2 0.65): res in {126, 130, 140, 142, 146..148, 160, 170, 176..178, 180, 182} | off (on 0) | - | same |
| L29 down c28 | 6% / 0 | **res** (R2 0.74): res in {106..107, 117, 146..147, 155..157, 160, 166..167, 176..177, 186..187, 196} | off (on 0) | - | same |
| L29 down c35 | 4% / 33% | **res** (R2 0.70): res in {2, 30..31, 40..41, 60..61, 70, 90} | unexplained (best R2 0.31) | - | same |
| L29 down c42 | 17% / 0 | **res** (R2 0.85): res in {30, 33, 120, 122..142} | off (on 0) | - | same |
| L29 down c48 | 12% / 2% | **res%100** (R2 0.75): res mod 100 in {1, 10..11, 50, 70} | **res%100** (R2 0.68): res mod 100 in {10..11} | - | same |
| L29 down c51 | 14% / 2% | **res** (R2 0.88): res in {20..23, 116..130} | **res** (R2 0.62): res in {20..22} | - | same |
| L29 down c53 | 2% / 0 | **res** (R2 0.61): res in {83, 147, 180, 182..183} | off (on 0) | - | same |
| L29 down c64 | 10% / 6% | **res%100** (R2 0.72): res mod 100 in {35..41, 58, 88} | unexplained (best R2 0.47) | - | same |
| L29 down c67 | 6% / 16% | **res** (R2 0.72): res in {4, 6..28, 37, 41..42} | **res** (R2 0.59): res in {1, 4, 9, 12..26} | - | same |
| L29 down c68 | 10% / 2% | **res%100** (R2 0.75): res mod 100 in {6..7, 9, 46, 56, 66, 86} | **res%100** (R2 0.66): res mod 100 in {6} | - | same |
| L29 down c71 | 9% / 2% | **res%100** (R2 0.71): res mod 100 in {14, 34, 44..45, 54, 93..95} | unexplained (best R2 0.22) | - | same |
| L29 down c72 | 13% / 4% | **res%50** (R2 0.74): res mod 50 in {20, 22..23, 27, 30, 32} | unexplained (best R2 0.47) | - | same |
| L29 down c90 | 8% / 1% | **res%100** (R2 0.74): res mod 100 in {66..70} | unexplained (best R2 0.17) | - | same |
| L29 down c92 | 11% / 1% | **res%100** (R2 0.78): res mod 100 in {33, 51..58, 63, 73, 93} | unexplained (best R2 0.23) | - | same |
| L29 down c94 | 1% / 2% | unexplained (best R2 0.44) | **res%100** (R2 0.55): res mod 100 in {7} | - | same |
| L29 down c103 | 0 / 2% | off (on 0) | **res%100** (R2 0.70): res mod 100: no class above 0.5 (max 0.46) | - | same |
| L29 down c115 | 1% / 25% | unexplained (best R2 0.36) | **tens(a,b)** (R2 0.53): a//10 in {0} -> b//10 in {0}; a//10 in {1} -> b//10 in {1}; a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9,10}; a//10 in {8} -> b//10 in {6,7,8,9,10}; a//10 in {9} -> b//10 in {7,9,10}; a//10 in {10} -> b//10 in {6,7,8,9} | - | res: mod100 +2% |
| L29 down c122 | 1% / 2% | **res%100** (R2 0.85): res mod 100 in {6} | **res%100** (R2 0.54): res mod 100 in {6} | - | same |
| L29 down c128 | 9% / 0 | **res%100** (R2 0.85): res mod 100: no class above 0.5 (max 0.50) | off (on 0) | - | same |
| L29 down c130 | 12% / 4% | **res%100** (R2 0.81): res mod 100 in {4, 14, 24, 34, 44, 54, 63..65, 74, 84, 94} [coarser: res mod 10 in {4}, R2 0.81] | **res** (R2 0.65): res in {4, 14, 24, 34, 44, 64, 84, 94} | - | same |
| L29 down c132 | 11% / 7% | **res%10** (R2 0.90): res mod 10 in {4} | **res** (R2 0.53): res in {-26, -16, -6, 14, 24, 34, 44, 54, 64, 74, 84, 94} [coarser: res mod 100 in {14, 24, 34, 54, 64, 74, 84, 94}, R2 0.82] | res: mod10 +2%, mod5 +3%, mod2 +2% | - |
| L29 down c134 | 1% / 2% | unexplained (best R2 0.23) | unexplained (best R2 0.49) | - | same |
| L29 down c136 | 13% / 7% | **res%100** (R2 0.76): res mod 100 in {1, 11, 21, 31, 39..43, 45, 51, 61, 71, 81, 91} [coarser: res mod 50 in {1, 11, 21, 31, 41}, R2 0.82] | unexplained (best R2 0.40) | - | same |
| L29 down c140 | 6% / 0 | **res%100** (R2 0.81): res mod 100 in {83..89} | off (on 0) | - | same |
| L29 down c144 | 8% / 11% | **res//10** (R2 0.84): (tens) res in {70..79} | unexplained (best R2 0.49) | - | same |
| L29 down c145 | 8% / 2% | **res%20** (R2 0.78): res mod 20 in {6, 16} [coarser: res mod 10 in {6}, R2 0.82] | unexplained (best R2 0.29) | - | same |
| L29 down c146 | 8% / 2% | **res%100** (R2 0.74): res mod 100 in {73..79} | unexplained (best R2 0.23) | - | same |
| L29 down c150 | 8% / 3% | **res%100** (R2 0.81): res mod 100 in {81..87} | unexplained (best R2 0.42) | - | same |
| L29 down c159 | 11% / 19% | unexplained (best R2 0.41) | unexplained (best R2 0.27) | a: mod2 +2%; b: mod4 +7%, mod2 +2%; res: mod4 +4% | a: mod4 +10%, mod2 +5%; b: mod4 +11%, mod2 +5%; res: mod4 +5% |
| L29 down c160 | 2% / 0 | **res** (R2 0.72): res in {129, 135..136} | off (on 0) | - | same |
| L29 down c171 | 33% / 1% | **res** (R2 0.81): res in {47, 57, 59, 61, 63, 67, 79, 81, 83, 85, 87, 89, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183, 185, 187, 189, 191, 193, 195, 197} | unexplained (best R2 0.15) | res: mod2 +7% | - |
| L29 down c172 | 7% / 1% | **res%100** (R2 0.69): res mod 100 in {45, 47, 49, 83, 85, 87, 89} | unexplained (best R2 0.20) | res: mod4 +3% | - |
| L29 down c174 | 14% / 1% | **res%100** (R2 0.74): res mod 100 in {1, 97} | unexplained (best R2 0.45) | - | same |
| L29 down c177 | 0 / 1% | off (on 0) | **res%100** (R2 0.65): res mod 100: no class above 0.5 (max 0.34) | - | same |
| L29 down c178 | 4% / 3% | **res%100** (R2 0.61): res mod 100 in {27..29, 68} | unexplained (best R2 0.43) | - | same |
| L29 down c183 | 2% / 0 | **res%100** (R2 0.89): res mod 100 in {34, 94} | off (on 0) | - | same |
| L29 down c187 | 6% / 7% | **res** (R2 0.65): res in {30..37, 39, 53, 132..135} [coarser: res mod 100 in {31..35}, R2 0.85] | unexplained (best R2 0.30) | - | same |
| L29 down c193 | 9% / 1% | **res%20** (R2 0.83): res mod 20 in {6, 8} | unexplained (best R2 0.19) | - | same |
| L29 down c209 | 12% / 3% | **res%20** (R2 0.82): res mod 20 in {2, 12} [coarser: res mod 10 in {2}, R2 0.84] | **res** (R2 0.65): res in {12, 22, 32} | - | same |
| L29 down c213 | 3% / 6% | **res%100** (R2 0.64): res mod 100: no class above 0.5 (max 0.50) | **tens(a,b)** (R2 0.62): a//10 in {9} -> b//10 in {0,2,3,4,5}; a//10 in {10} -> b//10 in {0} | - | same |
| L29 down c217 | 7% / 1% | **res%100** (R2 0.63): res mod 100 in {39, 49, 69, 79, 89} [coarser: res mod 50 in {39}, R2 0.80] | unexplained (best R2 0.36) | - | same |
| L29 down c219 | 8% / 0 | **res** (R2 0.74): res in {147..162, 170, 175} | off (on 0) | - | same |
| L29 down c230 | 15% / 4% | **res%20** (R2 0.69): res mod 20 in {12..13} | **res** (R2 0.59): res in {12..13, 33, 92..93} | res: mod4 +2% | - |
| L29 down c232 | 21% / 0 | **res** (R2 0.83): res in {104..106, 109, 114..116, 118..119, 123..126, 129, 134..136, 143..146, 149, 154..156, 158..159, 163..166, 169, 174..176, 178..179, 184..186, 189, 194..196} | off (on 0) | - | same |
| L29 down c238 | 3% / 0 | **res%100** (R2 0.86): res mod 100 in {77..79} | off (on 0) | - | same |
| L29 down c263 | 2% / 2% | **res%100** (R2 0.66): res mod 100 in {13} | unexplained (best R2 0.39) | - | same |
| L29 down c279 | 3% / 6% | **res%100** (R2 0.63): res mod 100 in {0, 19..22} | **res** (R2 0.51): res in {-20, 19..23} | - | same |
| L29 down c280 | 3% / 1% | **res%100** (R2 0.60): res mod 100 in {16} | **res** (R2 0.63): res in {16..17} | - | same |
| L29 down c282 | 10% / 3% | **res%100** (R2 0.83): res mod 100 in {22, 40..43, 52, 62, 82, 92} | **res** (R2 0.51): res in {22, 41..43} | - | same |
| L29 down c293 | 12% / 11% | **units(a,b)** (R2 0.95): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {9}; a%10 in {2} -> b%10 in {8}; a%10 in {3} -> b%10 in {7}; a%10 in {4} -> b%10 in {6}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {3}; a%10 in {8} -> b%10 in {2}; a%10 in {9} -> b%10 in {1} | **units(a,b)** (R2 0.51): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {1}; a%10 in {2} -> b%10 in {2}; a%10 in {4} -> b%10 in {4}; a%10 in {8} -> b%10 in {8}; a%10 in {9} -> b%10 in {9} | - | same |
| L29 down c300 | 1% / 0 | **res** (R2 0.73): res in {59, 149, 159} [coarser: res mod 100 in {59}, R2 0.86] | off (on 0) | - | same |
| L29 down c301 | 0 / 0 | off (on 0) | same | - | same |
| L29 down c308 | 4% / 1% | **res%100** (R2 0.72): res mod 100 in {46..49} | unexplained (best R2 0.24) | - | same |
| L29 down c318 | 1% / 2% | **res%100** (R2 0.77): res mod 100 in {12} | unexplained (best R2 0.50) | - | same |
| L29 down c321 | 8% / 1% | **res%100** (R2 0.72): res mod 100 in {5, 15, 25, 35, 45, 64..66, 85, 95} | **res** (R2 0.53): res in {5} | - | same |
| L29 down c325 | 5% / 5% | unexplained (best R2 0.27) | unexplained (best R2 0.21) | - | same |
| L29 down c326 | 1% / 0 | **res** (R2 0.63): res in {122, 131..132} | off (on 0) | - | same |
| L29 down c331 | 5% / 5% | **res** (R2 0.74): res in {60..67} | unexplained (best R2 0.37) | - | same |
| L29 down c343 | 2% / 2% | **res%100** (R2 0.67): res mod 100 in {17} | unexplained (best R2 0.47) | - | same |
| L29 down c358 | 9% / 3% | **res%100** (R2 0.83): res mod 100 in {1, 95..97, 99} | unexplained (best R2 0.48) | - | same |
| L29 down c376 | 2% / 2% | **res%100** (R2 0.65): res mod 100 in {26} | unexplained (best R2 0.30) | - | same |
| L29 down c390 | 0 / 1% | off (on 0) | **res%100** (R2 0.80): res mod 100: no class above 0.5 (max 0.47) | - | same |
| L29 down c414 | 1% / 0 | **res%100** (R2 0.76): res mod 100 in {94} | off (on 0) | - | same |
| L29 down c415 | 3% / 8% | **res** (R2 0.59): res in {30..32, 41, 91, 131} | unexplained (best R2 0.31) | - | same |
| L29 down c420 | 1% / 0 | **res%100** (R2 0.61): res mod 100 in {90} | off (on 0) | - | same |
| L29 down c426 | 1% / 0 | **res** (R2 0.80): res in {92, 152, 192} [coarser: res mod 100 in {52, 92}, R2 0.88] | off (on 0) | - | same |
| L29 down c433 | 3% / 2% | **res%100** (R2 0.70): res mod 100 in {51..53} | unexplained (best R2 0.26) | - | same |
| L29 down c434 | 3% / 1% | **res%100** (R2 0.90): res mod 100 in {1, 51, 81} | unexplained (best R2 0.25) | - | same |
| L29 down c435 | 10% / 9% | **res//10** (R2 0.80): (tens) res in {61..69, 161..169} [coarser: res mod 100 in {61..69}, R2 1.00] | unexplained (best R2 0.40) | - | same |
| L29 down c437 | 4% / 2% | **res** (R2 0.84): res in {59..64, 160} | unexplained (best R2 0.18) | - | same |
| L29 down c438 | 1% / 1% | **res%100** (R2 0.61): res mod 100 in {62} | unexplained (best R2 0.09) | - | same |
| L29 down c444 | 8% / 1% | **res%100** (R2 0.69): res mod 100 in {90..93, 97..98} | unexplained (best R2 0.25) | - | same |
| L29 down c445 | 1% / 2% | **res%100** (R2 0.67): res mod 100 in {11} | **res%100** (R2 0.49): res mod 100 in {11} | - | same |
| L29 down c448 | 3% / 4% | **res** (R2 0.62): res in {24..26, 45, 75, 125} [coarser: res mod 100 in {25..26, 75}, R2 0.81] | unexplained (best R2 0.46) | - | same |
| L29 down c449 | 7% / 0 | **tens(a,b)** (R2 0.71): a//10 in {5} -> b//10 in {10}; a//10 in {6} -> b//10 in {9,10}; a//10 in {7} -> b//10 in {8,9,10}; a//10 in {8} -> b//10 in {7,8}; a//10 in {9} -> b//10 in {6}; a//10 in {10} -> b//10 in {5,6} | off (on 0) | - | same |
| L29 down c467 | 1% / 1% | **res%100** (R2 0.74): res mod 100 in {15} | **res** (R2 0.57): res in {15} | - | same |
| L29 down c469 | 0 / 2% | off (on 0) | **res%100** (R2 0.61): res mod 100: no class above 0.5 (max 0.48) | - | same |
| L29 down c477 | 2% / 1% | **res%100** (R2 0.63): res mod 100 in {16, 96} | **res** (R2 0.60): res in {16, 96} | - | same |
| L29 down c480 | 7% / 1% | **res** (R2 0.90): res in {61, 71, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191} [coarser: res mod 100 in {1, 21, 31, 41, 61, 71, 91}, R2 0.82] | unexplained (best R2 0.30) | - | same |
| L29 down c497 | 1% / 0 | **res%100** (R2 0.69): res mod 100 in {1} | off (on 0) | - | same |
| L29 down c502 | 5% / 3% | **res//10** (R2 0.78): (tens) res in {40..49} | unexplained (best R2 0.37) | - | same |
| L29 down c514 | 0 / 3% | off (on 0) | **res%100** (R2 0.59): res mod 100: no class above 0.5 (max 0.49) | - | same |
| L29 down c549 | 7% / 0 | **res** (R2 0.88): res in {112..119, 125} | off (on 0) | - | same |
| L29 down c560 | 0 / 0 | off (on 0) | same | - | same |
| L29 down c566 | 1% / 0 | **res%100** (R2 0.64): res mod 100 in {35} | off (on 0) | - | same |
| L29 down c577 | 7% / 3% | **res%50** (R2 0.72): res mod 50 in {19, 29, 39, 49} [coarser: res mod 10 in {9}, R2 0.83] | **res** (R2 0.52): res in {19, 29, 39, 79, 99} | - | same |
| L29 down c584 | 11% / 11% | unexplained (best R2 0.50) | **tens(a,b)** (R2 0.54): a//10 in {0} -> b//10 in {0,1,2,10}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {2}; a//10 in {9} -> b//10 in {9,10} | - | same |
| L29 down c595 | 5% / 1% | **res%100** (R2 0.67): res mod 100 in {87..89} | unexplained (best R2 0.29) | - | same |
| L29 down c604 | 10% / 3% | **res%10** (R2 0.77): res mod 10 in {1} | **res** (R2 0.50): res in {11, 21, 31, 51, 71} | - | same |
| L29 down c618 | 1% / 1% | **res%100** (R2 0.87): res mod 100 in {27} | **res** (R2 0.76): res in {27} | - | same |
| L29 down c635 | 2% / 2% | **res** (R2 0.51): res in {9, 69, 89, 149} | unexplained (best R2 0.47) | - | same |
| L29 down c653 | 3% / 0 | **res%100** (R2 0.77): res mod 100 in {70..71} | off (on 0) | - | same |
| L29 down c662 | 9% / 0 | **res** (R2 0.82): res in {102..103, 112..113, 122..123, 132..133, 142..143, 152..153, 163, 172..173, 182..183, 192..193} | off (on 0) | - | same |
| L29 down c686 | 3% / 0 | **res** (R2 0.71): res in {150..154, 156..157} | off (on 0) | - | same |
| L29 down c736 | 2% / 1% | **res%100** (R2 0.60): res mod 100: no class above 0.5 (max 0.47) | **res** (R2 0.61): res in {19, 99} | - | same |
| L29 down c739 | 2% / 1% | **res** (R2 0.66): res in {14, 114, 144, 194} | **res** (R2 0.66): res in {14} | - | same |
| L29 down c761 | 0 / 3% | off (on 0) | **res%100** (R2 0.59): res mod 100: no class above 0.5 (max 0.45) | - | same |
| L29 down c762 | 1% / 2% | **res** (R2 0.56): res in {15, 25, 35, 65} | **res** (R2 0.55): res in {15, 25, 35} | - | same |
| L29 down c794 | 6% / 0 | **res%100** (R2 0.83): res mod 100 in {1} | off (on 0) | - | same |
| L29 down c824 | 2% / 0 | **res%100** (R2 0.84): res mod 100 in {33, 83, 93} | off (on 0) | - | same |
| L29 down c842 | 5% / 2% | **res%100** (R2 0.74): res mod 100 in {21, 41, 51, 61, 81} | unexplained (best R2 0.28) | - | same |
| L29 down c865 | 2% / 2% | **res%100** (R2 0.75): res mod 100 in {13, 63} [coarser: res mod 50 in {13}, R2 0.85] | **res** (R2 0.62): res in {13, 23} | - | same |
| L29 down c871 | 1% / 0 | unexplained (best R2 0.48) | off (on 0) | - | same |
| L29 down c880 | 6% / 0 | **res** (R2 0.86): res in {114..120, 140} | off (on 0) | - | same |
| L29 down c893 | 11% / 5% | **res//10** (R2 0.79): (tens) res in {30..31, 35..58} | unexplained (best R2 0.41) | - | same |
| L29 down c904 | 4% / 0 | **res** (R2 0.80): res in {88, 98, 183..200} | off (on 0) | - | same |
| L29 down c905 | 0 / 0 | off (on 0) | same | - | same |
| L29 down c927 | 4% / 0 | **res** (R2 0.62): res in {96..98, 186..200} | off (on 0) | - | same |
| L29 down c928 | 0 / 1% | off (on 0) | **res%100** (R2 0.79): res mod 100: no class above 0.5 (max 0.43) | - | same |
| L29 down c930 | 7% / 2% | **res%100** (R2 0.66): res mod 100 in {29, 59, 66, 69, 96, 99} | unexplained (best R2 0.24) | - | same |
| L29 down c964 | 2% / 0 | **res%100** (R2 0.76): res mod 100 in {49..50} | off (on 0) | - | same |
| L29 down c1009 | 3% / 1% | **res%100** (R2 0.62): res mod 100 in {90} | unexplained (best R2 0.41) | - | same |

</details>

<details><summary>gate: 136</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L29 gate c6 | 10% / 6% | **res%100** (R2 0.69): res mod 100 in {14, 34, 44, 64, 74..75, 79, 84, 94} | unexplained (best R2 0.43) | (reads) | same |
| L29 gate c42 | 17% / 0 | **res** (R2 0.92): res in {118..141} | off (on 0) | (reads) | same |
| L29 gate c48 | 12% / 3% | **res%100** (R2 0.77): res mod 100 in {10..13} | **res%100** (R2 0.76): res mod 100 in {11} | (reads) | same |
| L29 gate c51 | 14% / 4% | **res%100** (R2 0.81): res mod 100 in {17, 19..29} | **res** (R2 0.67): res in {20..23, 25} | (reads) | same |
| L29 gate c58 | 0 / 1% | off (on 0) | **res** (R2 0.87): res in {12} | (reads) | same |
| L29 gate c64 | 14% / 7% | **res** (R2 0.88): res in {34..42, 129..143} [coarser: res mod 100 in {30, 33..43}, R2 0.86] | **res** (R2 0.51): res in {34..42} | (reads) | same |
| L29 gate c68 | 10% / 11% | **res%100** (R2 0.89): res mod 100 in {1..10} | **res%100** (R2 0.66): res mod 100 in {6..10} | (reads) | same |
| L29 gate c71 | 14% / 4% | **res%100** (R2 0.78): res mod 100 in {41..42, 44..45, 47, 91..96} [coarser: res mod 50 in {41..47}, R2 0.84] | unexplained (best R2 0.42) | (reads) | same |
| L29 gate c72 | 33% / 13% | **res%50** (R2 0.73): res mod 50 in {19..34} | unexplained (best R2 0.45) | (reads) | same |
| L29 gate c90 | 10% / 3% | **res%100** (R2 0.75): res mod 100 in {7, 17, 27, 37..38, 47, 57, 67..68, 77, 87, 97} [coarser: res mod 20 in {7, 17}, R2 0.80] | **res** (R2 0.51): res in {7, 17} | (reads) | same |
| L29 gate c92 | 11% / 3% | **res//10** (R2 0.79): (tens) res in {51..60, 63, 150..159} [coarser: res mod 100 in {50..60}, R2 0.99] | unexplained (best R2 0.32) | (reads) | same |
| L29 gate c115 | 0 / 1% | off (on 0) | **res%100** (R2 0.84): res mod 100: no class above 0.5 (max 0.45) | (reads) | same |
| L29 gate c122 | 1% / 2% | unexplained (best R2 0.35) | **res%100** (R2 0.59): res mod 100 in {6} | (reads) | same |
| L29 gate c128 | 9% / 0 | **res%100** (R2 0.84): res mod 100: no class above 0.5 (max 0.49) | off (on 0) | (reads) | same |
| L29 gate c130 | 12% / 2% | **res%100** (R2 0.86): res mod 100 in {4, 24, 44, 54, 60..65, 74, 84, 94} | unexplained (best R2 0.49) | (reads) | same |
| L29 gate c132 | 11% / 7% | **res%10** (R2 0.92): res mod 10 in {4} | **res** (R2 0.66): res in {-16, -6, 4, 14, 24, 34, 44, 54, 64, 74, 84, 94} | (reads) | same |
| L29 gate c134 | 6% / 3% | **res%100** (R2 0.90): res mod 100 in {8, 28, 38, 48, 68, 88} [coarser: res mod 20 in {8}, R2 0.87] | **res** (R2 0.52): res in {8, 28, 88} | (reads) | same |
| L29 gate c136 | 15% / 7% | **res%100** (R2 0.77): res mod 100 in {1, 11, 21, 31, 38..45, 51, 61, 71, 81, 91} | unexplained (best R2 0.45) | (reads) | same |
| L29 gate c140 | 11% / 2% | **res//10** (R2 0.86): (tens) res in {80..90, 180..190} [coarser: res mod 100 in {80..90}, R2 1.00] | **res//10** (R2 0.46): (tens) res in {80..90} | (reads) | same |
| L29 gate c144 | 2% / 0 | **res%100** (R2 0.58): res mod 100 in {94..95} [coarser: res mod 50 in {44}, R2 0.86] | off (on 0) | (reads) | same |
| L29 gate c145 | 10% / 7% | **res%10** (R2 0.98): res mod 10 in {6} | **res** (R2 0.63): res in {-4, 6, 16, 26, 36, 46, 56, 66, 76, 86, 96} | (reads) | same |
| L29 gate c146 | 4% / 1% | **res%100** (R2 0.78): res mod 100 in {74..76} | unexplained (best R2 0.37) | (reads) | same |
| L29 gate c147 | 0 / 2% | off (on 0) | **res%100** (R2 0.73): res mod 100 in {8} | (reads) | same |
| L29 gate c150 | 8% / 2% | **res%100** (R2 0.89): res mod 100 in {79..86} | unexplained (best R2 0.48) | (reads) | same |
| L29 gate c153 | 6% / 2% | **res%100** (R2 0.76): res mod 100 in {56..60} | unexplained (best R2 0.20) | (reads) | same |
| L29 gate c159 | 8% / 21% | unexplained (best R2 0.30) | unexplained (best R2 0.27) | (reads) | same |
| L29 gate c166 | 6% / 5% | **res** (R2 0.76): res in {63..69} | unexplained (best R2 0.38) | (reads) | same |
| L29 gate c168 | 2% / 0 | **res%100** (R2 0.77): res mod 100: no class above 0.5 (max 0.45) | off (on 0) | (reads) | same |
| L29 gate c171 | 42% / 7% | **res%10** (R2 0.70): res mod 10 in {0, 2, 4, 6, 8} | unexplained (best R2 0.34) | (reads) | same |
| L29 gate c172 | 22% / 3% | **res%100** (R2 0.80): res mod 100 in {40..51, 80..89} | unexplained (best R2 0.29) | (reads) | same |
| L29 gate c174 | 18% / 4% | **res%100** (R2 0.86): res mod 100 in {0..7, 9..11, 95..99} | unexplained (best R2 0.46) | (reads) | same |
| L29 gate c178 | 6% / 3% | **res%100** (R2 0.76): res mod 100 in {27..29, 48, 68, 88} | **res** (R2 0.60): res in {8, 28..29, 88} | (reads) | same |
| L29 gate c183 | 2% / 0 | **res** (R2 0.62): res in {34, 149..152} | off (on 0) | (reads) | same |
| L29 gate c187 | 39% / 0 | **tens(a,b)** (R2 0.69): a//10 in {1} -> b//10 in {10}; a//10 in {2} -> b//10 in {8,10}; a//10 in {3} -> b//10 in {7,10}; a//10 in {4} -> b//10 in {6,7,9,10}; a//10 in {5} -> b//10 in {5,6,7,8,9,10}; a//10 in {6} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {1,2,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,4,5,6,7,8,9,10} | off (on 0) | (reads) | same |
| L29 gate c193 | 3% / 2% | **res** (R2 0.74): res in {106..108, 168} | **res%100** (R2 0.85): res mod 100 in {8} | (reads) | same |
| L29 gate c209 | 12% / 4% | **res%10** (R2 0.83): res mod 10 in {2} | **res** (R2 0.60): res in {12, 22, 32, 92} | (reads) | same |
| L29 gate c213 | 3% / 0 | **res%100** (R2 0.60): res mod 100 in {94} | off (on 0) | (reads) | same |
| L29 gate c217 | 10% / 4% | **res%10** (R2 0.90): res mod 10 in {9} | **res** (R2 0.62): res in {9, 19, 29, 39, 79, 89, 99} | (reads) | same |
| L29 gate c219 | 11% / 1% | **res** (R2 0.88): res in {51..54, 142..159} | unexplained (best R2 0.24) | (reads) | same |
| L29 gate c230 | 10% / 3% | **res%20** (R2 0.91): res mod 20 in {12..13} | **res** (R2 0.64): res in {12..13, 33, 93} | (reads) | same |
| L29 gate c232 | 23% / 0 | **res** (R2 0.74): res in {119, 122..126, 129, 134, 138..139, 141..179, 182..187} | off (on 0) | (reads) | same |
| L29 gate c238 | 1% / 0 | **res%100** (R2 0.73): res mod 100 in {79} | off (on 0) | (reads) | same |
| L29 gate c263 | 10% / 3% | **res%10** (R2 0.97): res mod 10 in {3} | **res** (R2 0.61): res in {3, 13, 33, 93} | (reads) | same |
| L29 gate c279 | 4% / 5% | **res%100** (R2 0.79): res mod 100 in {19..23} | **res** (R2 0.63): res in {19..23} | (reads) | same |
| L29 gate c280 | 7% / 4% | **res%20** (R2 0.74): res mod 20 in {6, 16} [coarser: res mod 10 in {6}, R2 0.88] | **res** (R2 0.62): res in {6, 16..17, 26, 96} | (reads) | same |
| L29 gate c281 | 0 / 1% | off (on 0) | **res** (R2 0.87): res in {12} | (reads) | same |
| L29 gate c282 | 10% / 5% | **res%100** (R2 0.77): res mod 100 in {22, 39..45, 62, 82} | **res** (R2 0.55): res in {12, 22, 40..44} | (reads) | same |
| L29 gate c284 | 1% / 0 | **res%100** (R2 0.76): res mod 100: no class above 0.5 (max 0.47) | off (on 0) | (reads) | same |
| L29 gate c286 | 7% / 79% | **res** (R2 0.62): res in {2..11, 16..17, 19..21, 25..37, 39..41} | unexplained (best R2 0.18) | (reads) | same |
| L29 gate c293 | 12% / 13% | **units(a,b)** (R2 0.95): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {9}; a%10 in {2} -> b%10 in {8}; a%10 in {3} -> b%10 in {7}; a%10 in {4} -> b%10 in {6}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {3}; a%10 in {8} -> b%10 in {2}; a%10 in {9} -> b%10 in {1} | **units(a,b)** (R2 0.55): a%10 in {0} -> b%10 in {0,5,9}; a%10 in {1} -> b%10 in {1}; a%10 in {2} -> b%10 in {2}; a%10 in {3} -> b%10 in {3}; a%10 in {4} -> b%10 in {4}; a%10 in {5} -> b%10 in {0,5}; a%10 in {6} -> b%10 in {6}; a%10 in {8} -> b%10 in {8}; a%10 in {9} -> b%10 in {9} | (reads) | same |
| L29 gate c300 | 4% / 1% | **res%100** (R2 0.74): res mod 100 in {57..60, 62} | unexplained (best R2 0.12) | (reads) | same |
| L29 gate c303 | 1% / 0 | **res%100** (R2 0.83): res mod 100: no class above 0.5 (max 0.47) | off (on 0) | (reads) | same |
| L29 gate c308 | 5% / 1% | **res%100** (R2 0.75): res mod 100 in {47..49, 67..68} | unexplained (best R2 0.29) | (reads) | same |
| L29 gate c316 | 2% / 22% | **res** (R2 0.58): res in {2..15, 18} | **res//10** (R2 0.56): (tens) res in {-99, 0..19, 99} | (reads) | same |
| L29 gate c318 | 1% / 0 | **res%100** (R2 0.52): res mod 100: no class above 0.5 (max 0.48) | off (on 0) | (reads) | same |
| L29 gate c320 | 0 / 11% | off (on 0) | unexplained (best R2 0.49) | (reads) | same |
| L29 gate c321 | 4% / 2% | **res%100** (R2 0.67): res mod 100 in {15, 45, 65, 85} | unexplained (best R2 0.46) | (reads) | same |
| L29 gate c324 | 0 / 1% | off (on 0) | **res** (R2 0.77): res in {13} | (reads) | same |
| L29 gate c331 | 1% / 0 | **res%100** (R2 0.66): res mod 100: no class above 0.5 (max 0.36) | off (on 0) | (reads) | same |
| L29 gate c343 | 11% / 4% | **res%100** (R2 0.83): res mod 100 in {7, 17, 27, 37, 47, 57, 67, 72..73, 77, 87, 97} [coarser: res mod 20 in {7, 17}, R2 0.81] | **res** (R2 0.58): res in {7, 17, 27} | (reads) | same |
| L29 gate c356 | 2% / 0 | **res%100** (R2 0.77): res mod 100 in {64} | off (on 0) | (reads) | same |
| L29 gate c358 | 9% / 4% | **res%100** (R2 0.76): res mod 100 in {0..6, 90, 97..99} | unexplained (best R2 0.41) | (reads) | same |
| L29 gate c365 | 2% / 1% | **res%100** (R2 0.87): res mod 100 in {38..39} | unexplained (best R2 0.47) | (reads) | same |
| L29 gate c370 | 2% / 0 | **res%100** (R2 0.82): res mod 100 in {94..95} | off (on 0) | (reads) | same |
| L29 gate c376 | 2% / 3% | **res** (R2 0.76): res in {6, 25..26, 126} [coarser: res mod 100 in {26}, R2 0.81] | **res** (R2 0.64): res in {6, 16, 25..26} | (reads) | same |
| L29 gate c379 | 2% / 0 | **res%100** (R2 0.88): res mod 100 in {75..76} | off (on 0) | (reads) | same |
| L29 gate c409 | 1% / 0 | **res%100** (R2 0.67): res mod 100 in {41} | off (on 0) | (reads) | same |
| L29 gate c415 | 4% / 2% | **res%100** (R2 0.71): res mod 100 in {11, 31, 71} | **res** (R2 0.59): res in {11, 31} | (reads) | same |
| L29 gate c419 | 5% / 2% | **res%100** (R2 0.66): res mod 100 in {9, 39, 69, 89, 99} | **res** (R2 0.60): res in {9, 99} | (reads) | same |
| L29 gate c420 | 4% / 6% | **res//10** (R2 0.67): (tens) res in {3..10, 12..19, 21..28} | **res** (R2 0.61): res in {9, 12..15, 25} | (reads) | same |
| L29 gate c426 | 2% / 1% | **res%100** (R2 0.59): res mod 100 in {62, 92} | **res** (R2 0.75): res in {12} | (reads) | same |
| L29 gate c433 | 7% / 2% | **res%100** (R2 0.80): res mod 100 in {32..34, 51..54} | unexplained (best R2 0.45) | (reads) | same |
| L29 gate c434 | 10% / 3% | **res%10** (R2 0.94): res mod 10 in {1} | **res** (R2 0.59): res in {11, 31, 41, 51, 91} | (reads) | same |
| L29 gate c435 | 12% / 9% | **res//10** (R2 0.85): (tens) res in {60..70, 160..170} [coarser: res mod 100 in {60..70}, R2 0.99] | unexplained (best R2 0.43) | (reads) | same |
| L29 gate c437 | 5% / 1% | **res** (R2 0.76): res in {57..62, 159, 162} [coarser: res mod 100 in {57..62}, R2 0.88] | unexplained (best R2 0.22) | (reads) | same |
| L29 gate c438 | 1% / 0 | **res%100** (R2 0.57): res mod 100: no class above 0.5 (max 0.39) | off (on 0) | (reads) | same |
| L29 gate c444 | 13% / 2% | **res%100** (R2 0.88): res mod 100 in {89..98} | unexplained (best R2 0.49) | (reads) | same |
| L29 gate c445 | 3% / 1% | **res%100** (R2 0.76): res mod 100 in {11, 51, 71} | **res** (R2 0.76): res in {11} | (reads) | same |
| L29 gate c448 | 7% / 3% | **res** (R2 0.75): res in {25..26, 65, 75, 121..127} | **res** (R2 0.55): res in {15, 21, 25..26} | (reads) | same |
| L29 gate c449 | 5% / 0 | **res** (R2 0.69): res in {149..160} | off (on 0) | (reads) | same |
| L29 gate c465 | 18% / 1% | **res** (R2 0.86): res in {104..109, 114..129} | unexplained (best R2 0.41) | (reads) | same |
| L29 gate c467 | 11% / 7% | **res%10** (R2 0.91): res mod 10 in {5} | **res%100** (R2 0.62): res mod 100 in {5, 15, 25, 35, 55, 65, 75, 85, 95} [coarser: res mod 50 in {5, 15, 25, 35, 45}, R2 0.80] | (reads) | same |
| L29 gate c469 | 6% / 2% | **res%100** (R2 0.76): res mod 100 in {5, 15, 45, 55, 65, 75, 95} [coarser: res mod 50 in {5, 15, 45}, R2 0.83] | **res%100** (R2 0.57): res mod 100 in {5} | (reads) | same |
| L29 gate c480 | 10% / 8% | **res%10** (R2 0.97): res mod 10 in {1} | **res** (R2 0.52): res in {-99, -29, -19, -9, 11, 21, 31, 41, 51, 61, 71, 81, 91} | (reads) | same |
| L29 gate c481 | 0 / 1% | off (on 0) | **res** (R2 0.75): res in {13} | (reads) | same |
| L29 gate c493 | 0 / 1% | off (on 0) | **res** (R2 0.89): res in {12} | (reads) | same |
| L29 gate c502 | 6% / 3% | **res** (R2 0.80): res in {40..49, 141..143} | unexplained (best R2 0.44) | (reads) | same |
| L29 gate c505 | 2% / 0 | **res%100** (R2 0.64): res mod 100 in {64, 84} | off (on 0) | (reads) | same |
| L29 gate c512 | 100% / 100% | always | same | (reads) | same |
| L29 gate c534 | 0 / 1% | off (on 0) | **res%100** (R2 0.78): res mod 100: no class above 0.5 (max 0.46) | (reads) | same |
| L29 gate c546 | 0 / 1% | off (on 0) | **res%100** (R2 0.75): res mod 100 in {6} | (reads) | same |
| L29 gate c566 | 10% / 8% | **res//10** (R2 0.85): (tens) res in {31..39, 130..139} [coarser: res mod 100 in {30..39}, R2 1.00] | unexplained (best R2 0.48) | (reads) | same |
| L29 gate c577 | 1% / 2% | **res** (R2 0.78): res in {7, 17, 67, 167} | **res%100** (R2 0.61): res mod 100 in {7} | (reads) | same |
| L29 gate c578 | 1% / 2% | **res%100** (R2 0.76): res mod 100 in {13} | **res** (R2 0.81): res in {13..14} | (reads) | same |
| L29 gate c581 | 1% / 1% | **res%100** (R2 0.74): res mod 100 in {33} | unexplained (best R2 0.47) | (reads) | same |
| L29 gate c591 | 0 / 1% | off (on 0) | **res%100** (R2 0.73): res mod 100: no class above 0.5 (max 0.45) | (reads) | same |
| L29 gate c592 | 0 / 2% | off (on 0) | **res%100** (R2 0.71): res mod 100 in {7} | (reads) | same |
| L29 gate c595 | 7% / 2% | **res%100** (R2 0.83): res mod 100 in {87..92} | **res** (R2 0.53): res in {9, 87..91} | (reads) | same |
| L29 gate c604 | 12% / 3% | **res%100** (R2 0.89): res mod 100 in {1, 11, 21, 31, 41, 51..53, 61, 71, 81, 91} [coarser: res mod 50 in {1, 11, 21, 31, 41}, R2 0.85] | unexplained (best R2 0.49) | (reads) | same |
| L29 gate c605 | 0 / 0 | off (on 0) | same | (reads) | same |
| L29 gate c612 | 0 / 2% | off (on 0) | **res** (R2 0.84): res in {12..13} | (reads) | same |
| L29 gate c614 | 0 / 1% | off (on 0) | **res%100** (R2 0.79): res mod 100: no class above 0.5 (max 0.46) | (reads) | same |
| L29 gate c618 | 4% / 1% | **res%100** (R2 0.78): res mod 100 in {27..28, 67..68} | unexplained (best R2 0.43) | (reads) | same |
| L29 gate c635 | 1% / 1% | **res%50** (R2 0.65): res mod 50 in {39} | **res** (R2 0.72): res in {9} | (reads) | same |
| L29 gate c653 | 12% / 8% | **res//10** (R2 0.83): (tens) res in {69..79, 169..179} [coarser: res mod 100 in {69..79}, R2 1.00] | unexplained (best R2 0.48) | (reads) | same |
| L29 gate c662 | 20% / 6% | **res%10** (R2 0.98): res mod 10 in {2..3} | **res** (R2 0.68): res in {3, 12..13, 22..23, 32..33} | (reads) | same |
| L29 gate c668 | 9% / 3% | **res%10** (R2 0.86): res mod 10 in {8} | **res** (R2 0.51): res in {8, 28, 88} | (reads) | same |
| L29 gate c681 | 0 / 1% | off (on 0) | **res** (R2 0.84): res in {13} | (reads) | same |
| L29 gate c684 | 0 / 1% | off (on 0) | **res** (R2 0.85): res in {13} | (reads) | same |
| L29 gate c686 | 5% / 0 | **res** (R2 0.73): res in {44, 54, 64, 104, 144, 150..154, 164, 194} | off (on 0) | (reads) | same |
| L29 gate c711 | 0 / 1% | off (on 0) | **res%100** (R2 0.76): res mod 100: no class above 0.5 (max 0.46) | (reads) | same |
| L29 gate c714 | 0 / 0 | off (on 0) | same | (reads) | same |
| L29 gate c725 | 6% / 1% | **res** (R2 0.80): res in {67..68, 70, 72..76} | unexplained (best R2 0.35) | (reads) | same |
| L29 gate c726 | 0 / 1% | off (on 0) | **res%100** (R2 0.69): res mod 100 in {7} | (reads) | same |
| L29 gate c733 | 0 / 1% | off (on 0) | **res** (R2 0.82): res in {12} | (reads) | same |
| L29 gate c736 | 4% / 1% | **res%50** (R2 0.48): res mod 50 in {39} | **res** (R2 0.54): res in {9, 99} | (reads) | same |
| L29 gate c738 | 2% / 0 | **res%100** (R2 0.74): res mod 100: no class above 0.5 (max 0.47) | off (on 0) | (reads) | same |
| L29 gate c739 | 10% / 4% | **res%10** (R2 0.97): res mod 10 in {4} | **res** (R2 0.68): res in {4, 14, 24, 34, 84, 94} | (reads) | same |
| L29 gate c762 | 0 / 3% | off (on 0) | **res%100** (R2 0.66): res mod 100: no class above 0.5 (max 0.48) | (reads) | same |
| L29 gate c774 | 5% / 3% | **res%20** (R2 0.89): res mod 20 in {6} | **res%100** (R2 0.51): res mod 100 in {6} | (reads) | same |
| L29 gate c792 | 1% / 0 | **res%100** (R2 0.92): res mod 100 in {34} | off (on 0) | (reads) | same |
| L29 gate c794 | 6% / 0 | **res%100** (R2 0.92): res mod 100 in {1} | off (on 0) | (reads) | same |
| L29 gate c821 | 0 / 1% | off (on 0) | **res%100** (R2 0.73): res mod 100: no class above 0.5 (max 0.48) | (reads) | same |
| L29 gate c823 | 2% / 0 | **res%100** (R2 0.75): res mod 100 in {64} | off (on 0) | (reads) | same |
| L29 gate c824 | 3% / 1% | **res%100** (R2 0.66): res mod 100 in {13, 33, 93} | **res** (R2 0.72): res in {13, 33} | (reads) | same |
| L29 gate c828 | 3% / 0 | **res** (R2 0.76): res in {41, 131, 137..138, 141} | off (on 0) | (reads) | same |
| L29 gate c835 | 7% / 4% | **res%100** (R2 0.89): res mod 100 in {29..35} | **res** (R2 0.63): res in {30..35} | (reads) | same |
| L29 gate c842 | 10% / 5% | **res%10** (R2 0.96): res mod 10 in {1} | **res** (R2 0.61): res in {11, 21, 31, 41, 51, 81, 91} | (reads) | same |
| L29 gate c856 | 0 / 1% | off (on 0) | **res** (R2 0.91): res in {12} | (reads) | same |
| L29 gate c865 | 0 / 3% | off (on 0) | **res%100** (R2 0.62): res mod 100: no class above 0.5 (max 0.44) | (reads) | same |
| L29 gate c868 | 0 / 1% | off (on 0) | **res** (R2 0.75): res in {13} | (reads) | same |
| L29 gate c871 | 3% / 2% | **res%100** (R2 0.77): res mod 100 in {7, 17, 67} | **res%100** (R2 0.56): res mod 100 in {7} | (reads) | same |
| L29 gate c883 | 0 / 1% | off (on 0) | **res** (R2 0.83): res in {13} | (reads) | same |
| L29 gate c889 | 2% / 0 | **res%100** (R2 0.81): res mod 100: no class above 0.5 (max 0.47) | off (on 0) | (reads) | same |
| L29 gate c904 | 8% / 1% | **res%100** (R2 0.68): res mod 100 in {67..70, 72..73, 83, 85, 88} | unexplained (best R2 0.27) | (reads) | same |
| L29 gate c910 | 0 / 0 | off (on 0) | same | (reads) | same |

</details>

<details><summary>o: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L29 o c73 (H30) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>up: 104</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L29 up c0 | 37% / 2% | **res** (R2 0.69): res in {19, 29, 34, 39, 44..45, 49, 54, 59, 69, 71, 73..76, 79..87, 89, 94..95, 99, 102..105, 109, 114..116, 118..120, 123..125, 129, 134..135, 139, 143..145, 149, 154, 159, 163..164, 169, 174..175, 179, 183..185, 189, 194..195, 199} | unexplained (best R2 0.34) | (reads) | same |
| L29 up c22 | 0 / 2% | off (on 0) | **res** (R2 0.66): res in {9, 13, 15} | (reads) | same |
| L29 up c38 | 100% / 100% | always | same | (reads) | same |
| L29 up c42 | 16% / 5% | **res** (R2 0.83): res in {9, 13, 33, 125..141, 143..144, 147, 186..187, 189} | **res** (R2 0.69): res in {6..7, 9, 11, 13} | (reads) | same |
| L29 up c48 | 11% / 3% | **res%100** (R2 0.76): res mod 100 in {1, 10..11, 20, 51, 60, 70..71, 90} | **res** (R2 0.64): res in {10..11, 20} | (reads) | same |
| L29 up c51 | 17% / 1% | **res** (R2 0.89): res in {28, 117..138} | **res** (R2 0.53): res in {21, 28} | (reads) | same |
| L29 up c64 | 15% / 5% | **res%100** (R2 0.77): res mod 100 in {18, 35..41, 48, 58, 87..89, 98} | unexplained (best R2 0.45) | (reads) | same |
| L29 up c68 | 20% / 8% | **res%20** (R2 0.63): res mod 20 in {6..7, 16} [coarser: res mod 10 in {6..7}, R2 0.83] | unexplained (best R2 0.40) | (reads) | same |
| L29 up c69 | 1% / 0 | **res%100** (R2 0.79): res mod 100 in {36} | off (on 0) | (reads) | same |
| L29 up c71 | 19% / 7% | **res%10** (R2 0.84): res mod 10 in {4..5} | **res** (R2 0.59): res in {4, 14, 24, 34, 44, 54, 74, 84..85, 94..95} | (reads) | same |
| L29 up c72 | 26% / 6% | **res%100** (R2 0.66): res mod 100 in {17, 22..23, 27, 32..33, 42..43, 47, 57, 63, 72..78, 82..84, 87, 92..93} [coarser: res mod 50 in {22..25, 27..28, 32..34, 37, 42..43}, R2 0.84] | unexplained (best R2 0.40) | (reads) | same |
| L29 up c90 | 11% / 2% | **res%100** (R2 0.74): res mod 100 in {64..70} | unexplained (best R2 0.24) | (reads) | same |
| L29 up c92 | 3% / 0 | **res%100** (R2 0.52): res mod 100 in {61, 67..68} | off (on 0) | (reads) | same |
| L29 up c116 | 1% / 0 | **res** (R2 0.89): res in {138} | off (on 0) | (reads) | same |
| L29 up c128 | 12% / 9% | **res//10** (R2 0.83): (tens) res in {2..10, 100..110, 200} [coarser: res mod 100 in {0..10}, R2 1.00] | **res%100** (R2 0.77): res mod 100 in {1, 6..9} | (reads) | same |
| L29 up c130 | 11% / 6% | **res%10** (R2 0.89): res mod 10 in {4} | **res** (R2 0.69): res in {4, 14, 24, 34, 44, 54, 64, 74, 84, 94} | (reads) | same |
| L29 up c132 | 0 / 0 | off (on 0) | same | (reads) | same |
| L29 up c136 | 2% / 1% | **res%100** (R2 0.74): res mod 100 in {41..42} | unexplained (best R2 0.33) | (reads) | same |
| L29 up c140 | 8% / 1% | **res** (R2 0.59): res in {26, 86, 123..126, 145..146, 156, 163..166, 174..176, 186} | unexplained (best R2 0.43) | (reads) | same |
| L29 up c144 | 12% / 4% | **res%100** (R2 0.91): res mod 100 in {59..69} | unexplained (best R2 0.35) | (reads) | same |
| L29 up c145 | 9% / 2% | **res%10** (R2 0.83): res mod 10 in {6} | unexplained (best R2 0.47) | (reads) | same |
| L29 up c146 | 6% / 1% | **res%100** (R2 0.81): res mod 100 in {25, 35, 45, 65, 85} | unexplained (best R2 0.48) | (reads) | same |
| L29 up c149 | 3% / 1% | **res%100** (R2 0.62): res mod 100: no class above 0.5 (max 0.41) | unexplained (best R2 0.48) | (reads) | same |
| L29 up c150 | 1% / 0 | **res%100** (R2 0.84): res mod 100 in {68} | off (on 0) | (reads) | same |
| L29 up c153 | 5% / 1% | **res%100** (R2 0.76): res mod 100 in {67..68, 88} | unexplained (best R2 0.23) | (reads) | same |
| L29 up c171 | 11% / 0 | **res** (R2 0.63): res in {101, 106..107, 110..111, 117, 121, 131, 137, 147, 150..151, 156..157, 161, 167, 170..171, 177, 187} | off (on 0) | (reads) | same |
| L29 up c172 | 11% / 7% | **res%10** (R2 0.88): res mod 10 in {5} | **res%100** (R2 0.62): res mod 100 in {5, 15, 25, 35, 45, 55, 65, 75, 85, 95} [coarser: res mod 10 in {5}, R2 0.80] | (reads) | same |
| L29 up c174 | 18% / 1% | **res%100** (R2 0.78): res mod 100 in {1..10, 95..99} | unexplained (best R2 0.38) | (reads) | same |
| L29 up c178 | 8% / 2% | **res%100** (R2 0.71): res mod 100 in {8, 27..28, 38, 48, 68, 87..88} | unexplained (best R2 0.37) | (reads) | same |
| L29 up c187 | 4% / 1% | **res%100** (R2 0.77): res mod 100 in {33, 53, 83} | unexplained (best R2 0.48) | (reads) | same |
| L29 up c189 | 4% / 4% | **res** (R2 0.75): res in {27..31, 127..129} [coarser: res mod 100 in {27..31}, R2 0.89] | **res** (R2 0.53): res in {27..31} | (reads) | same |
| L29 up c193 | 16% / 3% | **res%100** (R2 0.80): res mod 100 in {8, 26, 28, 46, 48, 66, 68, 82..84, 86, 88} | **res** (R2 0.52): res in {8, 82..83, 86, 88} | (reads) | same |
| L29 up c197 | 2% / 1% | **res%100** (R2 0.83): res mod 100 in {38..39} | unexplained (best R2 0.41) | (reads) | same |
| L29 up c209 | 12% / 3% | **res%100** (R2 0.86): res mod 100 in {12, 22, 32, 42, 52, 62, 67..68, 72, 82, 92} [coarser: res mod 50 in {2, 12, 22, 32, 42}, R2 0.84] | **res** (R2 0.56): res in {12, 22} | (reads) | same |
| L29 up c211 | 0 / 1% | off (on 0) | **res%100** (R2 0.78): res mod 100: no class above 0.5 (max 0.43) | (reads) | same |
| L29 up c217 | 11% / 3% | **res%100** (R2 0.83): res mod 100 in {38..44, 87..91} [coarser: res mod 50 in {37..42}, R2 0.82] | unexplained (best R2 0.44) | (reads) | same |
| L29 up c219 | 14% / 1% | **res** (R2 0.78): res in {49..54, 147..164, 170, 172..175, 182, 192, 194, 196, 198, 200} | unexplained (best R2 0.20) | (reads) | same |
| L29 up c230 | 26% / 7% | **res%100** (R2 0.87): res mod 100 in {12..13, 22..23, 32..33, 42..43, 52..54, 62..63, 71..74, 82..83, 91..94} [coarser: res mod 20 in {2..3, 12..14}, R2 0.83] | **res** (R2 0.63): res in {3, 12..13, 22..23, 33, 43, 53, 73, 92..93} | (reads) | same |
| L29 up c232 | 34% / 4% | **res** (R2 0.70): res in {49, 58..60, 69..71, 78..79, 88..92, 96..102, 108..111, 118..121, 128..131, 138..139, 148..152, 157..162, 168..172, 178..182, 188..193, 196..199} | unexplained (best R2 0.35) | (reads) | same |
| L29 up c244 | 0 / 1% | off (on 0) | **res** (R2 0.81): res in {11} | (reads) | same |
| L29 up c263 | 5% / 6% | **res%100** (R2 0.84): res mod 100 in {13..17} | **res** (R2 0.79): res in {9, 11..17} | (reads) | same |
| L29 up c264 | 0 / 1% | off (on 0) | **res%100** (R2 0.68): res mod 100: no class above 0.5 (max 0.44) | (reads) | same |
| L29 up c280 | 4% / 3% | **res%100** (R2 0.83): res mod 100 in {15..17} | **res** (R2 0.67): res in {15..17} | (reads) | same |
| L29 up c282 | 21% / 5% | **res%100** (R2 0.73): res mod 100 in {1, 21..22, 32, 40..43, 51..52, 71..72, 80..83, 91..92} | unexplained (best R2 0.49) | (reads) | same |
| L29 up c293 | 10% / 14% | **res%10** (R2 0.95): res mod 10 in {0} | unexplained (best R2 0.50) | (reads) | same |
| L29 up c300 | 1% / 0 | **res%100** (R2 0.80): res mod 100 in {36} | off (on 0) | (reads) | same |
| L29 up c308 | 9% / 3% | **res** (R2 0.78): res in {40..52, 146..149} | unexplained (best R2 0.27) | (reads) | same |
| L29 up c318 | 0 / 2% | off (on 0) | **res** (R2 0.80): res in {9, 11, 13} | (reads) | same |
| L29 up c321 | 4% / 0 | **res%100** (R2 0.79): res mod 100 in {45, 53..54, 65} | off (on 0) | (reads) | same |
| L29 up c325 | 3% / 4% | unexplained (best R2 0.16) | same | (reads) | same |
| L29 up c331 | 4% / 0 | **res%100** (R2 0.82): res mod 100 in {62..65} | off (on 0) | (reads) | same |
| L29 up c343 | 3% / 4% | **res%100** (R2 0.77): res mod 100 in {13, 15..17} | **res** (R2 0.73): res in {13, 15..17} | (reads) | same |
| L29 up c344 | 2% / 0 | **res%100** (R2 0.66): res mod 100 in {64, 67..68} | off (on 0) | (reads) | same |
| L29 up c358 | 14% / 6% | **res%100** (R2 0.89): res mod 100 in {1, 91..97, 99} | unexplained (best R2 0.40) | (reads) | same |
| L29 up c372 | 0 / 1% | off (on 0) | **res** (R2 0.72): res in {9, 13} | (reads) | same |
| L29 up c375 | 8% / 4% | **res%100** (R2 0.83): res mod 100 in {10..15} | **res** (R2 0.75): res in {10..13, 15} | (reads) | same |
| L29 up c376 | 3% / 4% | **res%100** (R2 0.82): res mod 100 in {25..26, 28} | unexplained (best R2 0.43) | (reads) | same |
| L29 up c415 | 2% / 7% | **res%100** (R2 0.64): res mod 100 in {31..32} | unexplained (best R2 0.29) | (reads) | same |
| L29 up c419 | 0 / 1% | off (on 0) | **res%100** (R2 0.76): res mod 100: no class above 0.5 (max 0.42) | (reads) | same |
| L29 up c433 | 4% / 1% | **res%100** (R2 0.76): res mod 100 in {52..54, 72} | unexplained (best R2 0.23) | (reads) | same |
| L29 up c434 | 2% / 0 | **res%100** (R2 0.59): res mod 100 in {80..81, 83} | off (on 0) | (reads) | same |
| L29 up c435 | 9% / 2% | **res//10** (R2 0.84): (tens) res in {61..69, 161..169} [coarser: res mod 100 in {61..69}, R2 1.00] | unexplained (best R2 0.34) | (reads) | same |
| L29 up c437 | 4% / 1% | **res** (R2 0.78): res in {59..64} | unexplained (best R2 0.18) | (reads) | same |
| L29 up c444 | 2% / 0 | **res%100** (R2 0.67): res mod 100 in {67..68, 94} | off (on 0) | (reads) | same |
| L29 up c445 | 3% / 31% | **res** (R2 0.61): res in {2..19, 23..24} | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {0,1}; a//10 in {1,2} -> b//10 in {0,1,2}; a//10 in {3} -> b//10 in {1,2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {4,5,6,7}; a//10 in {7,8} -> b//10 in {5,6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {5,6,7,8,9} | (reads) | same |
| L29 up c448 | 7% / 1% | **res%100** (R2 0.69): res mod 100 in {45, 67..68, 85} | **res** (R2 0.54): res in {25} | (reads) | same |
| L29 up c449 | 8% / 0 | **tens(a,b)** (R2 0.75): a//10 in {6} -> b//10 in {9,10}; a//10 in {7} -> b//10 in {8,9,10}; a//10 in {8} -> b//10 in {7,8,9,10}; a//10 in {9} -> b//10 in {7,8,10}; a//10 in {10} -> b//10 in {6,7,8,10} | off (on 0) | (reads) | same |
| L29 up c467 | 8% / 9% | **res%100** (R2 0.84): res mod 100 in {13..19} | **res** (R2 0.70): res in {9, 12..19, 21} | (reads) | same |
| L29 up c488 | 0 / 0 | off (on 0) | same | (reads) | same |
| L29 up c497 | 3% / 0 | **res%100** (R2 0.85): res mod 100 in {1, 99} | off (on 0) | (reads) | same |
| L29 up c502 | 6% / 2% | **res%100** (R2 0.73): res mod 100 in {10..12} | **res%100** (R2 0.69): res mod 100 in {11} | (reads) | same |
| L29 up c529 | 2% / 0 | **res%100** (R2 0.75): res mod 100 in {53..54} | off (on 0) | (reads) | same |
| L29 up c549 | 11% / 8% | **res//10** (R2 0.89): (tens) res in {3, 20..29, 120..129} [coarser: res mod 100 in {20..29}, R2 0.96] | **res//10** (R2 0.75): (tens) res in {20..29} | (reads) | same |
| L29 up c553 | 0 / 1% | off (on 0) | **res** (R2 0.61): res in {13} | (reads) | same |
| L29 up c566 | 9% / 2% | **res** (R2 0.64): res in {35, 45, 49, 85, 95, 125, 135, 145, 148..156, 175, 185, 195} [coarser: res mod 100 in {25, 35, 45, 49, 51, 55, 75, 85, 95}, R2 0.84] | unexplained (best R2 0.46) | (reads) | same |
| L29 up c574 | 2% / 0 | **res** (R2 0.69): res in {148..152} | off (on 0) | (reads) | same |
| L29 up c595 | 1% / 0 | **res%100** (R2 0.53): res mod 100: no class above 0.5 (max 0.36) | off (on 0) | (reads) | same |
| L29 up c600 | 50% / 0 | **res>=100** (R2 0.91): res>=100=0: 0.01, res>=100=1: 0.96 | off (on 0) | (reads) | same |
| L29 up c618 | 1% / 1% | **res%100** (R2 0.93): res mod 100 in {28} | **res** (R2 0.62): res in {28} | (reads) | same |
| L29 up c627 | 1% / 0 | **res%100** (R2 0.71): res mod 100: no class above 0.5 (max 0.38) | off (on 0) | (reads) | same |
| L29 up c635 | 0 / 8% | off (on 0) | **res%100** (R2 0.54): res mod 100: no class above 0.5 (max 0.45) | (reads) | same |
| L29 up c653 | 1% / 0 | **res%100** (R2 0.71): res mod 100 in {62} | off (on 0) | (reads) | same |
| L29 up c668 | 0 / 1% | off (on 0) | **res%100** (R2 0.66): res mod 100: no class above 0.5 (max 0.37) | (reads) | same |
| L29 up c672 | 2% / 1% | **res%100** (R2 0.79): res mod 100 in {45, 85} | unexplained (best R2 0.38) | (reads) | same |
| L29 up c686 | 10% / 1% | **res//10** (R2 0.83): (tens) res in {51..58, 150..159, 175} [coarser: res mod 100 in {50..59}, R2 0.95] | unexplained (best R2 0.24) | (reads) | same |
| L29 up c714 | 3% / 0 | **res** (R2 0.74): res in {122..125} | off (on 0) | (reads) | same |
| L29 up c719 | 2% / 0 | **res%100** (R2 0.78): res mod 100: no class above 0.5 (max 0.43) | off (on 0) | (reads) | same |
| L29 up c736 | 8% / 2% | **res%100** (R2 0.65): res mod 100 in {20, 50, 70, 74..76} | unexplained (best R2 0.43) | (reads) | same |
| L29 up c739 | 2% / 0 | **res%100** (R2 0.58): res mod 100 in {67, 94} | off (on 0) | (reads) | same |
| L29 up c762 | 4% / 2% | **res%20** (R2 0.74): res mod 20 in {5} | **res** (R2 0.61): res in {5, 25} | (reads) | same |
| L29 up c778 | 0 / 1% | off (on 0) | **res%100** (R2 0.81): res mod 100: no class above 0.5 (max 0.42) | (reads) | same |
| L29 up c799 | 2% / 0 | **res%100** (R2 0.85): res mod 100: no class above 0.5 (max 0.46) | off (on 0) | (reads) | same |
| L29 up c804 | 1% / 0 | **res%100** (R2 0.83): res mod 100 in {62} | off (on 0) | (reads) | same |
| L29 up c817 | 0 / 0 | off (on 0) | same | (reads) | same |
| L29 up c824 | 8% / 2% | **res%100** (R2 0.86): res mod 100 in {22, 42, 52, 62, 72, 82, 92} [coarser: res mod 50 in {2, 12, 22, 42}, R2 0.84] | **res** (R2 0.57): res in {22, 42} | (reads) | same |
| L29 up c835 | 9% / 3% | **res** (R2 0.73): res in {28, 30..31, 60, 70, 90, 120, 128..134, 150, 169..170} | **res** (R2 0.50): res in {20, 28, 30..31} | (reads) | same |
| L29 up c836 | 0 / 1% | off (on 0) | **res** (R2 0.68): res in {13} | (reads) | same |
| L29 up c842 | 11% / 7% | **res%10** (R2 0.84): res mod 10 in {1} | **res** (R2 0.54): res in {-99, -29, -19, -9, 11, 21, 31, 41, 51, 61, 81, 91} | (reads) | same |
| L29 up c853 | 2% / 1% | **res%100** (R2 0.78): res mod 100 in {38..39} | unexplained (best R2 0.48) | (reads) | same |
| L29 up c865 | 0 / 1% | off (on 0) | **res%100** (R2 0.79): res mod 100: no class above 0.5 (max 0.41) | (reads) | same |
| L29 up c871 | 2% / 1% | **res%100** (R2 0.86): res mod 100 in {7} | **res%100** (R2 0.52): res mod 100: no class above 0.5 (max 0.30) | (reads) | same |
| L29 up c875 | 1% / 0 | **res%100** (R2 0.66): res mod 100 in {62} | off (on 0) | (reads) | same |
| L29 up c880 | 8% / 0 | **res%100** (R2 0.59): res mod 100 in {44..45, 54, 94..95} | off (on 0) | (reads) | same |
| L29 up c904 | 1% / 0 | **res%100** (R2 0.64): res mod 100: no class above 0.5 (max 0.34) | off (on 0) | (reads) | same |

</details>

<details><summary>v: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L29 v c214 (kv6) | 100% / 100% | always | same | (reads) | same |

</details>

</details>

<details><summary>layer 30: 217 components with main position `=`</summary>

<details><summary>down: 79</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L30 down c8 | 5% / 0 | **res** (R2 0.72): res in {118, 131, 135..139, 171, 174, 178..179} | off (on 0) | - | same |
| L30 down c17 | 4% / 0 | **res** (R2 0.76): res in {112..115, 131, 171, 181} | off (on 0) | - | same |
| L30 down c21 | 88% / 1% | unexplained (best R2 0.27) | unexplained (best R2 0.37) | a: mod100 +4%, mod50 +3%, mod25 +2%; b: mod100 +3%, mod25 +3% | - |
| L30 down c47 | 63% / 71% | unexplained (best R2 0.28) | unexplained (best R2 0.35) | - | same |
| L30 down c107 | 23% / 1% | unexplained (best R2 0.28) | unexplained (best R2 0.05) | - | same |
| L30 down c121 | 9% / 0 | **res** (R2 0.87): res in {113, 128..138, 188} | off (on 0) | - | same |
| L30 down c133 | 9% / 5% | **res%50** (R2 0.88): res mod 50 in {8, 18, 28, 48} | **res** (R2 0.61): res in {-2, 8, 18, 28, 38, 48, 98} | - | same |
| L30 down c159 | 9% / 4% | **res//10** (R2 0.82): (tens) res in {23..29, 120..129} [coarser: res mod 100 in {22..29}, R2 0.93] | **res** (R2 0.71): res in {23..29} | - | same |
| L30 down c177 | 10% / 1% | **res%100** (R2 0.71): res mod 100 in {26, 56..58, 66, 95..98} | unexplained (best R2 0.25) | - | same |
| L30 down c181 | 8% / 2% | **res%100** (R2 0.59): res mod 100 in {16..20} | **res** (R2 0.51): res in {17..19} | - | same |
| L30 down c183 | 6% / 1% | **res%100** (R2 0.62): res mod 100 in {31..33, 82} | unexplained (best R2 0.48) | - | same |
| L30 down c187 | 10% / 5% | **res%10** (R2 0.82): res mod 10 in {7} | **res** (R2 0.59): res in {17, 27, 37, 47, 57, 77, 87} | - | same |
| L30 down c192 | 1% / 1% | **res** (R2 0.61): res in {29..30, 130} | unexplained (best R2 0.25) | - | same |
| L30 down c199 | 8% / 1% | **res%100** (R2 0.76): res mod 100 in {40..47} | unexplained (best R2 0.25) | - | same |
| L30 down c201 | 0 / 7% | off (on 0) | **tens(a,b)** (R2 0.85): a//10 in {0} -> b//10 in {2,3,4,5,6,7,8,9,10} | - | same |
| L30 down c205 | 11% / 2% | **res** (R2 0.80): res in {43..49, 140..151} [coarser: res mod 100 in {42..50}, R2 0.86] | unexplained (best R2 0.45) | - | same |
| L30 down c207 | 8% / 3% | **res%100** (R2 0.80): res mod 100 in {26..33} | **res** (R2 0.65): res in {28..31} | - | same |
| L30 down c209 | 10% / 4% | **res%10** (R2 0.91): res mod 10 in {4} | **res** (R2 0.59): res in {14, 24, 34, 44, 74, 84, 94} | - | same |
| L30 down c214 | 7% / 1% | **res%100** (R2 0.77): res mod 100 in {17, 27, 37, 57, 77..78, 97} | **res** (R2 0.60): res in {17} | - | same |
| L30 down c223 | 2% / 3% | **res** (R2 0.60): res in {39..41, 140} [coarser: res mod 100 in {40}, R2 0.80] | unexplained (best R2 0.27) | - | same |
| L30 down c230 | 10% / 6% | **res%10** (R2 0.89): res mod 10 in {8} | unexplained (best R2 0.44) | res: mod5 +2% | - |
| L30 down c232 | 9% / 1% | **res%100** (R2 0.74): res mod 100 in {77, 81..84, 88} | unexplained (best R2 0.31) | - | same |
| L30 down c239 | 19% / 3% | **res%100** (R2 0.83): res mod 100 in {33, 42..43, 52..53, 62..63, 72..73, 82..83, 92..93} | unexplained (best R2 0.41) | res: mod2 +2% | - |
| L30 down c249 | 8% / 1% | **res%100** (R2 0.71): res mod 100 in {33, 60, 62..65, 83} | unexplained (best R2 0.22) | - | same |
| L30 down c251 | 9% / 0 | **res** (R2 0.80): res in {61, 81..82, 111..112, 121..123, 141..142, 161..162, 181..182} | off (on 0) | - | same |
| L30 down c270 | 0 / 0 | off (on 0) | same | - | same |
| L30 down c283 | 2% / 5% | **res** (R2 0.56): res in {21, 121, 171} [coarser: res mod 100 in {21}, R2 0.89] | unexplained (best R2 0.28) | - | same |
| L30 down c293 | 6% / 0 | **res** (R2 0.74): res in {59, 108..109, 119, 129, 158..160, 169} | off (on 0) | - | same |
| L30 down c305 | 2% / 0 | **res%50** (R2 0.72): res mod 50 in {1} | off (on 0) | - | same |
| L30 down c309 | 3% / 1% | **res%100** (R2 0.81): res mod 100 in {4..5} | **res%100** (R2 0.52): res mod 100: no class above 0.5 (max 0.42) | - | same |
| L30 down c317 | 3% / 0 | **res** (R2 0.67): res in {103..104, 143, 163, 183} | off (on 0) | - | same |
| L30 down c320 | 22% / 0 | **res** (R2 0.76): res in {102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 182, 184, 186, 188, 192, 194, 196, 198} | off (on 0) | res: mod2 +2% | - |
| L30 down c322 | 9% / 1% | **res//10** (R2 0.81): (tens) res in {70..79, 170..173} | unexplained (best R2 0.22) | - | same |
| L30 down c326 | 3% / 0 | **res%100** (R2 0.74): res mod 100 in {56..57} | off (on 0) | - | same |
| L30 down c357 | 4% / 0 | **res%100** (R2 0.73): res mod 100 in {85..87} | off (on 0) | - | same |
| L30 down c359 | 8% / 3% | **res%100** (R2 0.82): res mod 100 in {23..26, 44, 64, 74, 84} | **res** (R2 0.71): res in {23..26} | - | same |
| L30 down c364 | 8% / 1% | **res//10** (R2 0.77): (tens) res in {81..89, 181..186} [coarser: res mod 100 in {81..87}, R2 0.95] | unexplained (best R2 0.41) | - | same |
| L30 down c370 | 8% / 1% | **res%100** (R2 0.71): res mod 100 in {90..95} | unexplained (best R2 0.39) | - | same |
| L30 down c377 | 100% / 100% | always | same | - | same |
| L30 down c379 | 4% / 1% | **res** (R2 0.76): res in {72..74, 123, 133, 173..174} [coarser: res mod 100 in {73..74}, R2 0.84] | unexplained (best R2 0.20) | - | same |
| L30 down c381 | 4% / 99% | unexplained (best R2 0.41) | always | - | same |
| L30 down c394 | 3% / 0 | **res%100** (R2 0.77): res mod 100 in {67, 87} | off (on 0) | - | same |
| L30 down c399 | 11% / 0 | **res//10** (R2 0.73): (tens) res in {101..109, 198..200} | off (on 0) | - | same |
| L30 down c407 | 6% / 2% | **res%100** (R2 0.76): res mod 100 in {25..29} | unexplained (best R2 0.50) | - | same |
| L30 down c414 | 2% / 0 | **res%50** (R2 0.84): res mod 50 in {42} | off (on 0) | - | same |
| L30 down c452 | 1% / 0 | **res%100** (R2 0.84): res mod 100 in {71} | off (on 0) | - | same |
| L30 down c465 | 7% / 1% | **res//10** (R2 0.64): (tens) res in {130..140} | unexplained (best R2 0.17) | - | same |
| L30 down c475 | 1% / 0 | **res%100** (R2 0.68): res mod 100 in {58, 68} | off (on 0) | - | same |
| L30 down c486 | 3% / 1% | **res%100** (R2 0.87): res mod 100 in {79..81} | unexplained (best R2 0.21) | - | same |
| L30 down c501 | 5% / 1% | **res%100** (R2 0.76): res mod 100 in {51, 61, 90..92} | unexplained (best R2 0.19) | - | same |
| L30 down c519 | 1% / 1% | **res%100** (R2 0.76): res mod 100 in {14} | **res** (R2 0.62): res in {14} | - | same |
| L30 down c535 | 5% / 4% | **res%100** (R2 0.76): res mod 100 in {0, 94..99} | **a** (R2 0.69): a in {97..99} | - | same |
| L30 down c551 | 2% / 2% | **res** (R2 0.76): res in {23..25, 124} [coarser: res mod 100 in {23..24}, R2 0.82] | **res** (R2 0.59): res in {23..24} | - | same |
| L30 down c580 | 2% / 6% | **res** (R2 0.59): res in {32, 52, 132} | unexplained (best R2 0.21) | - | same |
| L30 down c594 | 5% / 0 | **res%100** (R2 0.60): res mod 100: no class above 0.5 (max 0.44) | off (on 0) | - | same |
| L30 down c603 | 3% / 1% | **res%100** (R2 0.68): res mod 100 in {69..70} | unexplained (best R2 0.14) | - | same |
| L30 down c628 | 2% / 1% | **res%100** (R2 0.77): res mod 100 in {23..24} | **res** (R2 0.60): res in {23..24} | - | same |
| L30 down c633 | 0 / 0 | off (on 0) | same | - | same |
| L30 down c642 | 1% / 1% | **res%100** (R2 0.59): res mod 100 in {18} | **res** (R2 0.58): res in {18} | - | same |
| L30 down c660 | 4% / 5% | **res** (R2 0.65): res in {24..27, 75, 85, 125} [coarser: res mod 100 in {24..25, 75, 85}, R2 0.84] | unexplained (best R2 0.29) | - | same |
| L30 down c661 | 2% / 0 | **res** (R2 0.63): res in {47, 147..149} | off (on 0) | - | same |
| L30 down c700 | 1% / 0 | **res** (R2 0.64): res in {152} | off (on 0) | - | same |
| L30 down c718 | 0 / 0 | off (on 0) | same | - | same |
| L30 down c739 | 1% / 1% | **res%100** (R2 0.84): res mod 100 in {19} | **res** (R2 0.66): res in {19} | - | same |
| L30 down c796 | 3% / 1% | **res%100** (R2 0.73): res mod 100 in {20, 40, 60} | **res** (R2 0.54): res in {20} | - | same |
| L30 down c800 | 0 / 1% | off (on 0) | **res** (R2 0.73): res in {20} | - | same |
| L30 down c808 | 5% / 0 | **res%100** (R2 0.83): res mod 100 in {67..70} | off (on 0) | - | same |
| L30 down c809 | 12% / 0 | **res** (R2 0.89): res in {146..172, 196, 198} | off (on 0) | - | same |
| L30 down c810 | 3% / 1% | **res%100** (R2 0.75): res mod 100 in {52, 82} | **res%100** (R2 0.59): res mod 100 in {2} | - | same |
| L30 down c831 | 2% / 0 | **res** (R2 0.84): res in {57, 59, 157..159} [coarser: res mod 100 in {57..59}, R2 0.89] | off (on 0) | - | same |
| L30 down c879 | 1% / 1% | **res%100** (R2 0.64): res mod 100 in {16} | **res** (R2 0.65): res in {16} | - | same |
| L30 down c882 | 2% / 3% | **res** (R2 0.77): res in {25..29, 127} | **res** (R2 0.58): res in {26..29} | - | same |
| L30 down c918 | 0 / 0 | off (on 0) | same | - | same |
| L30 down c941 | 3% / 0 | **res** (R2 0.73): res in {64, 161..165, 167..169} | off (on 0) | - | same |
| L30 down c956 | 0 / 0 | off (on 0) | same | - | same |
| L30 down c961 | 5% / 1% | **res%100** (R2 0.80): res mod 100 in {50..54} | unexplained (best R2 0.26) | - | same |
| L30 down c997 | 1% / 2% | **res%100** (R2 0.58): res mod 100 in {22} | unexplained (best R2 0.39) | - | same |
| L30 down c1003 | 0 / 1% | off (on 0) | **cmp(a,b)** (R2 0.74): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.92, cmp(a,b)=1: 0.00 | - | same |
| L30 down c1012 | 2% / 0 | **res%100** (R2 0.80): res mod 100 in {79..80} | off (on 0) | - | same |

</details>

<details><summary>gate: 80</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L30 gate c7 | 99% / 93% | always | unexplained (best R2 0.15) | (reads) | same |
| L30 gate c30 | 91% / 2% | unexplained (best R2 0.29) | unexplained (best R2 0.21) | (reads) | same |
| L30 gate c35 | 42% / 0 | **res>=100** (R2 0.62): res>=100=0: 0.02, res>=100=1: 0.79 | off (on 0) | (reads) | same |
| L30 gate c44 | 100% / 100% | always | same | (reads) | same |
| L30 gate c70 | 5% / 0 | **res** (R2 0.62): res in {118, 128, 131, 137..138, 171, 177..178, 188} | off (on 0) | (reads) | same |
| L30 gate c87 | 30% / 91% | unexplained (best R2 0.21) | unexplained (best R2 0.36) | (reads) | same |
| L30 gate c133 | 10% / 7% | **res%10** (R2 0.86): res mod 10 in {8} | **res** (R2 0.63): res in {-12, -2, 8, 18, 28, 38, 48, 58, 68, 78, 88, 98} | (reads) | same |
| L30 gate c159 | 9% / 4% | **res//10** (R2 0.82): (tens) res in {22..27, 121..129} [coarser: res mod 100 in {21..28}, R2 0.95] | **res** (R2 0.79): res in {22..27} | (reads) | same |
| L30 gate c170 | 0 / 0 | off (on 0) | same | (reads) | same |
| L30 gate c177 | 13% / 1% | **res%100** (R2 0.76): res mod 100 in {55..60, 66, 92..97, 99} | unexplained (best R2 0.26) | (reads) | same |
| L30 gate c181 | 11% / 6% | **res%100** (R2 0.81): res mod 100 in {15..23} | **res** (R2 0.70): res in {16..22} | (reads) | same |
| L30 gate c183 | 4% / 1% | **res%100** (R2 0.67): res mod 100 in {32, 82} | unexplained (best R2 0.32) | (reads) | same |
| L30 gate c187 | 10% / 6% | **res%10** (R2 0.94): res mod 10 in {7} | **res** (R2 0.62): res in {-3, 17, 27, 37, 47, 57, 67, 77, 87, 97} | (reads) | same |
| L30 gate c192 | 1% / 1% | **res** (R2 0.68): res in {30, 40} | unexplained (best R2 0.30) | (reads) | same |
| L30 gate c199 | 12% / 5% | **res//10** (R2 0.82): (tens) res in {39..49, 139..149} [coarser: res mod 100 in {39..49}, R2 1.00] | **res** (R2 0.50): res in {40..47} | (reads) | same |
| L30 gate c205 | 10% / 2% | **res%100** (R2 0.78): res mod 100 in {42..50} | **res** (R2 0.50): res in {45..48} | (reads) | same |
| L30 gate c207 | 9% / 5% | **res%100** (R2 0.93): res mod 100 in {25..33} | **res** (R2 0.72): res in {26..33} | (reads) | same |
| L30 gate c209 | 10% / 6% | **res%10** (R2 0.97): res mod 10 in {4} | **res** (R2 0.64): res in {14, 24, 34, 44, 54, 64, 74, 84, 94} | (reads) | same |
| L30 gate c214 | 10% / 2% | **res%100** (R2 0.79): res mod 100 in {17, 27, 37, 57, 77..78, 87, 97} | **res** (R2 0.58): res in {17, 37, 57, 77} | (reads) | same |
| L30 gate c223 | 4% / 4% | **res** (R2 0.66): res in {38..42, 44, 139..140} [coarser: res mod 100 in {39..40}, R2 0.81] | unexplained (best R2 0.34) | (reads) | same |
| L30 gate c230 | 11% / 9% | **res%10** (R2 0.89): res mod 10 in {8} | **res%10** (R2 0.51): res mod 10 in {8} | (reads) | same |
| L30 gate c232 | 10% / 2% | **res%100** (R2 0.60): res mod 100 in {28, 38, 48, 58, 68, 78, 83, 88, 98} | unexplained (best R2 0.34) | (reads) | same |
| L30 gate c249 | 12% / 3% | **res%100** (R2 0.83): res mod 100 in {23, 33, 60..65, 67, 83} | unexplained (best R2 0.42) | (reads) | same |
| L30 gate c251 | 6% / 2% | **res%100** (R2 0.68): res mod 100 in {21..23, 81} | **res** (R2 0.74): res in {21..23} | (reads) | same |
| L30 gate c261 | 1% / 0 | **res%100** (R2 0.71): res mod 100 in {63} | off (on 0) | (reads) | same |
| L30 gate c268 | 1% / 0 | **res%100** (R2 0.90): res mod 100 in {63} | off (on 0) | (reads) | same |
| L30 gate c283 | 10% / 5% | **res%10** (R2 0.74): res mod 10 in {1} | unexplained (best R2 0.37) | (reads) | same |
| L30 gate c293 | 6% / 0 | **res%100** (R2 0.80): res mod 100 in {58..60, 98} | off (on 0) | (reads) | same |
| L30 gate c295 | 11% / 4% | **res//10** (R2 0.84): (tens) res in {80..90, 180..187, 189} [coarser: res mod 100 in {80..90}, R2 0.95] | unexplained (best R2 0.41) | (reads) | same |
| L30 gate c309 | 3% / 3% | **res%100** (R2 0.83): res mod 100 in {4..5} | **res%100** (R2 0.57): res mod 100: no class above 0.5 (max 0.43) | (reads) | same |
| L30 gate c317 | 11% / 3% | **res%10** (R2 0.83): res mod 10 in {3} | **res** (R2 0.60): res in {3, 23, 33, 83, 93} | (reads) | same |
| L30 gate c320 | 17% / 1% | **res** (R2 0.56): res in {48, 52, 58, 68, 78, 82, 88, 96, 98, 102, 108, 118, 122, 124, 126, 128, 132, 138, 144, 148, 152, 154, 156, 158, 164, 168, 172, 178, 182, 188, 196, 198} | unexplained (best R2 0.16) | (reads) | same |
| L30 gate c322 | 12% / 5% | **res%100** (R2 0.93): res mod 100 in {69..80} | unexplained (best R2 0.44) | (reads) | same |
| L30 gate c326 | 7% / 1% | **res%100** (R2 0.59): res mod 100 in {54..59, 96} | unexplained (best R2 0.15) | (reads) | same |
| L30 gate c327 | 0 / 0 | off (on 0) | same | (reads) | same |
| L30 gate c357 | 7% / 1% | **res%100** (R2 0.86): res mod 100 in {83..89} | unexplained (best R2 0.40) | (reads) | same |
| L30 gate c359 | 11% / 2% | **res%100** (R2 0.78): res mod 100 in {14, 23..25, 34, 44, 54, 64, 74, 84, 94} [coarser: res mod 50 in {4, 14, 24, 34, 44}, R2 0.85] | **res** (R2 0.52): res in {23..24, 84} | (reads) | same |
| L30 gate c364 | 8% / 2% | **res%100** (R2 0.93): res mod 100 in {80..87} | unexplained (best R2 0.35) | (reads) | same |
| L30 gate c370 | 7% / 5% | **res%100** (R2 0.78): res mod 100 in {91..95} | **tens(a,b)** (R2 0.54): a//10 in {9} -> b//10 in {0,1,2,3,4,5,10} | (reads) | same |
| L30 gate c394 | 5% / 1% | **res%100** (R2 0.82): res mod 100 in {27, 37, 67, 87} | unexplained (best R2 0.49) | (reads) | same |
| L30 gate c399 | 13% / 7% | **res//10** (R2 0.73): (tens) res in {99..110, 198..200} | **res%100** (R2 0.73): res mod 100 in {2..8} | (reads) | same |
| L30 gate c407 | 4% / 1% | **res%100** (R2 0.82): res mod 100 in {26..29} | unexplained (best R2 0.39) | (reads) | same |
| L30 gate c414 | 7% / 1% | **res** (R2 0.86): res in {42, 44, 46, 92, 141..148, 192} [coarser: res mod 100 in {42..47, 92}, R2 0.84] | unexplained (best R2 0.25) | (reads) | same |
| L30 gate c424 | 2% / 0 | **res%100** (R2 0.87): res mod 100 in {79..80} | off (on 0) | (reads) | same |
| L30 gate c465 | 11% / 4% | **res%100** (R2 0.82): res mod 100 in {29..39} | **res//10** (R2 0.63): (tens) res in {31..37} | (reads) | same |
| L30 gate c486 | 5% / 2% | **res%100** (R2 0.85): res mod 100 in {40, 79..82} | unexplained (best R2 0.29) | (reads) | same |
| L30 gate c499 | 4% / 0 | **res%100** (R2 0.83): res mod 100 in {1, 51, 61, 91} | off (on 0) | (reads) | same |
| L30 gate c501 | 5% / 1% | **res%100** (R2 0.74): res mod 100 in {90..92} | unexplained (best R2 0.29) | (reads) | same |
| L30 gate c510 | 2% / 0 | **res%100** (R2 0.73): res mod 100 in {80..81} | off (on 0) | (reads) | same |
| L30 gate c519 | 8% / 2% | **res%100** (R2 0.76): res mod 100 in {13..14, 24, 34, 54, 74} | **res** (R2 0.50): res in {14, 34} | (reads) | same |
| L30 gate c535 | 9% / 6% | **res%100** (R2 0.83): res mod 100 in {0..1, 93, 95..99} | **a** (R2 0.61): a in {94..99} | (reads) | same |
| L30 gate c551 | 4% / 3% | **res%100** (R2 0.73): res mod 100 in {23..25, 74, 84} | **res** (R2 0.70): res in {23..25} | (reads) | same |
| L30 gate c580 | 5% / 5% | **res** (R2 0.67): res in {31..35, 52, 82, 131..133, 182} [coarser: res mod 100 in {31..33, 82}, R2 0.89] | unexplained (best R2 0.33) | (reads) | same |
| L30 gate c594 | 2% / 0 | **res%100** (R2 0.55): res mod 100: no class above 0.5 (max 0.49) | off (on 0) | (reads) | same |
| L30 gate c603 | 8% / 2% | **res%100** (R2 0.72): res mod 100 in {69..73, 80} | unexplained (best R2 0.22) | (reads) | same |
| L30 gate c628 | 2% / 1% | **res%100** (R2 0.95): res mod 100 in {22..23} | **res** (R2 0.72): res in {22..23} | (reads) | same |
| L30 gate c633 | 4% / 1% | **res%50** (R2 0.70): res mod 50 in {9, 19} | **res** (R2 0.75): res in {19} | (reads) | same |
| L30 gate c641 | 1% / 0 | **res%100** (R2 0.90): res mod 100 in {93} | off (on 0) | (reads) | same |
| L30 gate c642 | 4% / 3% | **res%100** (R2 0.82): res mod 100 in {17..19} | **res** (R2 0.74): res in {17..19} | (reads) | same |
| L30 gate c660 | 8% / 5% | **res%100** (R2 0.74): res mod 100 in {23..27, 75, 85} | **res** (R2 0.53): res in {24..27} | (reads) | same |
| L30 gate c661 | 10% / 1% | **res** (R2 0.80): res in {45..50, 143..156, 158, 200} | unexplained (best R2 0.40) | (reads) | same |
| L30 gate c669 | 2% / 0 | **res%100** (R2 0.71): res mod 100 in {48, 98..99} | off (on 0) | (reads) | same |
| L30 gate c700 | 6% / 1% | **res%100** (R2 0.86): res mod 100 in {50..55} | unexplained (best R2 0.25) | (reads) | same |
| L30 gate c706 | 7% / 6% | **res%100** (R2 0.84): res mod 100 in {23..29} | **res** (R2 0.71): res in {23..30} | (reads) | same |
| L30 gate c719 | 0 / 0 | off (on 0) | same | (reads) | same |
| L30 gate c727 | 0 / 0 | off (on 0) | same | (reads) | same |
| L30 gate c735 | 1% / 0 | **res%100** (R2 0.71): res mod 100 in {70} | off (on 0) | (reads) | same |
| L30 gate c739 | 10% / 6% | **res%10** (R2 0.87): res mod 10 in {9} | **res** (R2 0.53): res in {9, 19, 29, 39, 49, 59, 69, 79} | (reads) | same |
| L30 gate c763 | 1% / 0 | **res%100** (R2 0.91): res mod 100 in {63} | off (on 0) | (reads) | same |
| L30 gate c796 | 9% / 3% | **res%10** (R2 0.81): res mod 10 in {0} | unexplained (best R2 0.46) | (reads) | same |
| L30 gate c806 | 1% / 1% | **res%100** (R2 0.82): res mod 100: no class above 0.5 (max 0.43) | **res%100** (R2 0.70): res mod 100 in {2} | (reads) | same |
| L30 gate c809 | 16% / 0 | **res//10** (R2 0.86): (tens) res in {144..182, 184..187, 192, 194..200} | off (on 0) | (reads) | same |
| L30 gate c810 | 10% / 8% | **res%10** (R2 0.95): res mod 10 in {2} | **res** (R2 0.57): res in {-28, -18, -8, 2, 22, 32, 42, 52, 62, 72, 82, 92} | (reads) | same |
| L30 gate c831 | 6% / 1% | **res%100** (R2 0.73): res mod 100 in {57..60, 98} | unexplained (best R2 0.16) | (reads) | same |
| L30 gate c843 | 0 / 0 | off (on 0) | same | (reads) | same |
| L30 gate c851 | 2% / 0 | **res%100** (R2 0.85): res mod 100 in {63} | off (on 0) | (reads) | same |
| L30 gate c868 | 4% / 1% | **res** (R2 0.79): res in {40, 69..70, 73..74, 170, 174} [coarser: res mod 100 in {69..70, 73..74}, R2 0.83] | unexplained (best R2 0.22) | (reads) | same |
| L30 gate c879 | 1% / 1% | **res%100** (R2 0.87): res mod 100 in {16} | **res** (R2 0.75): res in {16} | (reads) | same |
| L30 gate c882 | 8% / 2% | **res%100** (R2 0.85): res mod 100 in {17, 27..28, 37, 57, 67, 77, 87} [coarser: res mod 50 in {17, 27, 37}, R2 0.82] | **res** (R2 0.55): res in {17, 27, 37} | (reads) | same |
| L30 gate c883 | 0 / 0 | off (on 0) | same | (reads) | same |

</details>

<details><summary>up: 58</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L30 up c16 | 23% / 0 | **res** (R2 0.77): res in {112..116, 118, 126..139, 144, 148, 152..154, 158, 162, 164, 166..200} | off (on 0) | (reads) | same |
| L30 up c92 | 0 / 0 | off (on 0) | same | (reads) | same |
| L30 up c133 | 11% / 11% | **res%10** (R2 0.90): res mod 10 in {8} | **res%100** (R2 0.48): res mod 100 in {2, 8, 18, 28, 38, 48, 58, 68, 78, 88, 98} | (reads) | same |
| L30 up c135 | 10% / 4% | **res//10** (R2 0.88): (tens) res in {80..89, 181..187} [coarser: res mod 100 in {80..89}, R2 0.96] | unexplained (best R2 0.32) | (reads) | same |
| L30 up c144 | 1% / 0 | **res%100** (R2 0.65): res mod 100: no class above 0.5 (max 0.33) | off (on 0) | (reads) | same |
| L30 up c159 | 9% / 3% | **res%100** (R2 0.80): res mod 100 in {22..28, 75} | **res** (R2 0.69): res in {24..27} | (reads) | same |
| L30 up c177 | 10% / 2% | **res%100** (R2 0.72): res mod 100 in {26, 56..57, 66, 77, 86, 95..98} | unexplained (best R2 0.41) | (reads) | same |
| L30 up c181 | 16% / 4% | **res** (R2 0.77): res in {16..21, 110..125, 157} | **res** (R2 0.67): res in {16..20} | (reads) | same |
| L30 up c183 | 10% / 4% | **res%100** (R2 0.77): res mod 100 in {29..36, 82} | **res** (R2 0.61): res in {30..34, 82} | (reads) | same |
| L30 up c187 | 15% / 6% | **res%100** (R2 0.80): res mod 100 in {7, 17, 27, 34..39, 47, 57, 67, 77, 87, 97} [coarser: res mod 50 in {7, 17, 27, 37, 47}, R2 0.83] | **res** (R2 0.52): res in {17, 27, 37..38, 47, 57, 77, 87} | (reads) | same |
| L30 up c199 | 9% / 2% | **res%100** (R2 0.81): res mod 100 in {40..47} | unexplained (best R2 0.38) | (reads) | same |
| L30 up c205 | 13% / 2% | **res** (R2 0.78): res in {43..49, 86, 139..151, 156, 186} [coarser: res mod 100 in {42..50, 86}, R2 0.88] | unexplained (best R2 0.45) | (reads) | same |
| L30 up c207 | 6% / 2% | **res%100** (R2 0.80): res mod 100 in {27..31} | **res** (R2 0.63): res in {28..31} | (reads) | same |
| L30 up c209 | 11% / 7% | **res%10** (R2 0.92): res mod 10 in {4} | **res** (R2 0.60): res in {-16, -6, 14, 24, 34, 44, 54, 64, 74, 84, 94} [coarser: res mod 100 in {24, 34, 44, 54, 64, 74, 84, 94}, R2 0.80] | (reads) | same |
| L30 up c214 | 20% / 3% | **res%100** (R2 0.73): res mod 100 in {16..18, 27, 37, 47, 57..58, 67, 74..79, 87, 97} | **res** (R2 0.52): res in {17..18, 57, 77} | (reads) | same |
| L30 up c223 | 3% / 1% | **res%100** (R2 0.75): res mod 100 in {20, 40, 70, 80} | **res** (R2 0.52): res in {20} | (reads) | same |
| L30 up c230 | 13% / 3% | **res%100** (R2 0.81): res mod 100 in {28, 38, 48, 58, 68, 70, 78..80, 88, 98} | unexplained (best R2 0.35) | (reads) | same |
| L30 up c232 | 3% / 0 | **res%100** (R2 0.80): res mod 100 in {80..81, 83} | off (on 0) | (reads) | same |
| L30 up c239 | 19% / 5% | **res%10** (R2 0.91): res mod 10 in {2..3} | unexplained (best R2 0.47) | (reads) | same |
| L30 up c249 | 15% / 2% | **res%100** (R2 0.73): res mod 100 in {23, 33, 43, 53, 60..65, 73, 83..84, 93} | unexplained (best R2 0.45) | (reads) | same |
| L30 up c251 | 21% / 5% | **res%100** (R2 0.71): res mod 100 in {1, 11..12, 21..22, 31, 41..42, 51, 61..62, 71..72, 77, 81..82, 91..92} [coarser: res mod 50 in {1, 11..12, 21..22, 31..32, 41..42}, R2 0.83] | unexplained (best R2 0.41) | (reads) | same |
| L30 up c293 | 6% / 0 | **res%100** (R2 0.66): res mod 100 in {9, 58..60} | off (on 0) | (reads) | same |
| L30 up c294 | 1% / 1% | **res%100** (R2 0.89): res mod 100 in {19} | **res** (R2 0.76): res in {19} | (reads) | same |
| L30 up c309 | 9% / 5% | **res%100** (R2 0.65): res mod 100 in {4..5, 25, 55, 65, 85} | **res%100** (R2 0.52): res mod 100 in {5} | (reads) | same |
| L30 up c317 | 8% / 3% | **res%100** (R2 0.83): res mod 100 in {0..4, 99} | **res%100** (R2 0.66): res mod 100 in {3} | (reads) | same |
| L30 up c322 | 8% / 1% | **res%100** (R2 0.68): res mod 100 in {71..77} | unexplained (best R2 0.17) | (reads) | same |
| L30 up c326 | 4% / 1% | **res** (R2 0.87): res in {55..57, 155..158, 177} [coarser: res mod 100 in {55..57, 77}, R2 0.87] | unexplained (best R2 0.13) | (reads) | same |
| L30 up c357 | 9% / 1% | **res%100** (R2 0.49): res mod 100 in {56, 66, 85..87} | unexplained (best R2 0.23) | (reads) | same |
| L30 up c359 | 9% / 4% | **res%100** (R2 0.83): res mod 100 in {23..26, 44, 64, 74, 84} | **res** (R2 0.67): res in {23..26} | (reads) | same |
| L30 up c364 | 8% / 1% | **res%100** (R2 0.85): res mod 100 in {80..87} | unexplained (best R2 0.46) | (reads) | same |
| L30 up c370 | 11% / 10% | **res//10** (R2 0.76): (tens) res in {89..99, 189..195} [coarser: res mod 100 in {89..97}, R2 0.93] | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {9,10}; a//10 in {8} -> b//10 in {9}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,9,10}; a//10 in {10} -> b//10 in {0} | (reads) | same |
| L30 up c379 | 4% / 1% | **res%100** (R2 0.76): res mod 100 in {23, 37, 73..74} | **res** (R2 0.59): res in {23, 37} | (reads) | same |
| L30 up c394 | 8% / 1% | **res%100** (R2 0.73): res mod 100 in {27, 37, 57, 67, 87, 97} | unexplained (best R2 0.36) | (reads) | same |
| L30 up c399 | 9% / 0 | **res//10** (R2 0.78): (tens) res in {101..109, 200} | off (on 0) | (reads) | same |
| L30 up c407 | 7% / 4% | **res%100** (R2 0.92): res mod 100 in {25..31} | **res** (R2 0.68): res in {25..30} | (reads) | same |
| L30 up c414 | 11% / 6% | **res%10** (R2 0.93): res mod 10 in {2} | **res** (R2 0.51): res in {-18, -8, 22, 32, 42, 52, 62, 72, 82, 92} [coarser: res mod 100 in {22, 32, 42, 62, 72, 82, 92}, R2 0.80] | (reads) | same |
| L30 up c486 | 10% / 2% | **res%100** (R2 0.78): res mod 100 in {75..83} | unexplained (best R2 0.24) | (reads) | same |
| L30 up c535 | 1% / 0 | **res%100** (R2 0.69): res mod 100 in {70} | off (on 0) | (reads) | same |
| L30 up c551 | 4% / 2% | **res%100** (R2 0.79): res mod 100 in {23..25, 84} | **res** (R2 0.80): res in {23..25} | (reads) | same |
| L30 up c565 | 0 / 0 | off (on 0) | same | (reads) | same |
| L30 up c580 | 4% / 1% | **res%100** (R2 0.78): res mod 100 in {31..33, 82} | **res** (R2 0.60): res in {32..33} | (reads) | same |
| L30 up c594 | 74% / 0 | **res** (R2 0.58): res in {45..50, 52..53, 55..69, 72, 75..77, 85..86, 89..90, 92..200} | off (on 0) | (reads) | same |
| L30 up c603 | 4% / 2% | **res%100** (R2 0.78): res mod 100 in {68..71} | unexplained (best R2 0.13) | (reads) | same |
| L30 up c628 | 9% / 3% | **res%10** (R2 0.91): res mod 10 in {3} | unexplained (best R2 0.48) | (reads) | same |
| L30 up c642 | 2% / 0 | **res** (R2 0.56): res in {117..118} | off (on 0) | (reads) | same |
| L30 up c649 | 0 / 0 | off (on 0) | same | (reads) | same |
| L30 up c660 | 0 / 0 | off (on 0) | same | (reads) | same |
| L30 up c687 | 0 / 0 | off (on 0) | same | (reads) | same |
| L30 up c706 | 0 / 0 | off (on 0) | same | (reads) | same |
| L30 up c739 | 4% / 3% | **res%100** (R2 0.81): res mod 100 in {18..20} | **res** (R2 0.66): res in {18..20} | (reads) | same |
| L30 up c787 | 1% / 0 | **res%100** (R2 0.64): res mod 100 in {70} | off (on 0) | (reads) | same |
| L30 up c796 | 5% / 1% | **res%100** (R2 0.84): res mod 100 in {20, 40, 60, 70, 80} | **res** (R2 0.58): res in {20} | (reads) | same |
| L30 up c806 | 1% / 0 | **res%100** (R2 0.59): res mod 100: no class above 0.5 (max 0.32) | off (on 0) | (reads) | same |
| L30 up c809 | 1% / 0 | **res** (R2 0.82): res in {157..159} | off (on 0) | (reads) | same |
| L30 up c810 | 6% / 8% | **res%100** (R2 0.58): res mod 100 in {0, 2..4, 32, 52, 82} | **res%100** (R2 0.56): res mod 100 in {2} | (reads) | same |
| L30 up c855 | 0 / 0 | off (on 0) | same | (reads) | same |
| L30 up c882 | 5% / 5% | **res%100** (R2 0.80): res mod 100 in {25..29} | **res** (R2 0.66): res in {25..30} | (reads) | same |
| L30 up c883 | 1% / 0 | **res** (R2 0.55): res in {146, 156} | off (on 0) | (reads) | same |

</details>

</details>

<details><summary>layer 31: 308 components with main position `=`</summary>

<details><summary>down: 162</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L31 down c2 | 100% / 100% | always | same | - | same |
| L31 down c6 | 100% / 100% | always | same | - | same |
| L31 down c7 | 100% / 100% | always | same | - | same |
| L31 down c9 | 100% / 100% | always | same | - | same |
| L31 down c10 | 0 / 6% | off (on 0) | **a** (R2 0.53): a in {1..2, 6..9} | - | same |
| L31 down c11 | 97% / 45% | always | unexplained (best R2 0.35) | - | same |
| L31 down c13 | 0 / 1% | off (on 0) | unexplained (best R2 0.38) | - | same |
| L31 down c15 | 7% / 90% | unexplained (best R2 0.19) | unexplained (best R2 0.27) | - | same |
| L31 down c17 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c18 | 100% / 100% | always | same | - | same |
| L31 down c22 | 100% / 100% | always | same | - | same |
| L31 down c23 | 100% / 100% | always | same | - | same |
| L31 down c24 | 0 / 3% | off (on 0) | unexplained (best R2 0.42) | - | same |
| L31 down c27 | 13% / 1% | **res//10** (R2 0.60): (tens) res in {142..143, 147..154, 157..164, 167..199} | unexplained (best R2 0.15) | - | same |
| L31 down c29 | 99% / 45% | always | unexplained (best R2 0.42) | - | same |
| L31 down c33 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c38 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c41 | 65% / 17% | **tens(a,b)** (R2 0.63): a//10 in {0,1,2} -> b//10 in {6,7,8,9,10}; a//10 in {3} -> b//10 in {5,6,7,8,9,10}; a//10 in {4,6} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {0,2,3,4,5,6,7,8,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | unexplained (best R2 0.36) | - | same |
| L31 down c42 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c44 | 10% / 0 | **res** (R2 0.84): res in {99, 117..120, 122..129} | off (on 0) | - | same |
| L31 down c45 | 5% / 70% | unexplained (best R2 0.45) | **tens(a,b)** (R2 0.52): a//10 in {0,9} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1,2} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {4} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {7,8,9}; a//10 in {7} -> b//10 in {0,1,2,3,4,7,8,9,10}; a//10 in {8} -> b//10 in {1,2,3,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,10} | - | res: mod100 +2% |
| L31 down c47 | 100% / 100% | always | same | - | same |
| L31 down c48 | 2% / 0 | unexplained (best R2 0.42) | off (on 0) | - | same |
| L31 down c51 | 7% / 0 | **res** (R2 0.68): res in {38..39, 71, 111, 129, 131, 138..139, 143, 159, 161, 163, 171} | off (on 0) | - | same |
| L31 down c52 | 0 / 3% | off (on 0) | **a** (R2 0.68): a in {1..3} | - | same |
| L31 down c53 | 1% / 0 | **res** (R2 0.59): res in {97, 194..198} | off (on 0) | - | same |
| L31 down c54 | 12% / 0 | **res** (R2 0.77): res in {76, 84, 86, 106, 114..116, 126, 134..136, 146, 154..156, 164..167, 175..176, 184..186, 196} | off (on 0) | - | same |
| L31 down c56 | 1% / 0 | **res%100** (R2 0.61): res mod 100: no class above 0.5 (max 0.39) | off (on 0) | - | same |
| L31 down c61 | 100% / 100% | always | same | - | same |
| L31 down c64 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c65 | 4% / 0 | **res%100** (R2 0.56): res mod 100 in {97} | off (on 0) | - | same |
| L31 down c66 | 1% / 0 | unexplained (best R2 0.36) | off (on 0) | - | same |
| L31 down c72 | 0 / 1% | off (on 0) | unexplained (best R2 0.29) | - | same |
| L31 down c73 | 1% / 0 | **res%100** (R2 0.57): res mod 100 in {86} | off (on 0) | - | same |
| L31 down c79 | 2% / 0 | **res** (R2 0.68): res in {163..167, 169} | off (on 0) | - | same |
| L31 down c80 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c83 | 0 / 1% | off (on 0) | **res** (R2 0.52): res in {20} | - | same |
| L31 down c85 | 2% / 1% | **res** (R2 0.57): res in {5, 150, 155, 160, 170, 175} | **res%100** (R2 0.63): res mod 100: no class above 0.5 (max 0.47) | - | same |
| L31 down c87 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c90 | 0 / 53% | off (on 0) | unexplained (best R2 0.47) | - | same |
| L31 down c100 | 2% / 10% | unexplained (best R2 0.20) | unexplained (best R2 0.30) | - | same |
| L31 down c103 | 2% / 3% | unexplained (best R2 0.47) | **res%100** (R2 0.49): res mod 100 in {9} | - | same |
| L31 down c104 | 48% / 1% | unexplained (best R2 0.44) | unexplained (best R2 0.21) | - | same |
| L31 down c106 | 1% / 0 | unexplained (best R2 0.43) | off (on 0) | - | same |
| L31 down c108 | 3% / 0 | **res** (R2 0.85): res in {140..145} | off (on 0) | - | same |
| L31 down c122 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c125 | 1% / 0 | **res** (R2 0.51): res in {151..153} | off (on 0) | - | same |
| L31 down c126 | 1% / 0 | **res%100** (R2 0.77): res mod 100 in {83} | off (on 0) | - | same |
| L31 down c127 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c128 | 1% / 0 | **res%100** (R2 0.66): res mod 100: no class above 0.5 (max 0.38) | off (on 0) | - | same |
| L31 down c132 | 9% / 0 | **res** (R2 0.80): res in {91, 149..162, 170..171, 175, 181, 191} | off (on 0) | - | same |
| L31 down c134 | 32% / 10% | **tens(a,b)** (R2 0.56): a//10 in {0} -> b//10 in {8,9}; a//10 in {1} -> b//10 in {7,8}; a//10 in {2} -> b//10 in {6,7}; a//10 in {3} -> b//10 in {5,6}; a//10 in {4} -> b//10 in {4,5}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {2,3}; a//10 in {7} -> b//10 in {1,2,9}; a//10 in {8} -> b//10 in {0,1,8,9}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,8,9} | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {9,10}; a//10 in {8} -> b//10 in {0}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,9,10}; a//10 in {10} -> b//10 in {0,1,2} | - | same |
| L31 down c137 | 2% / 0 | **res** (R2 0.57): res in {109, 189, 192..199} | off (on 0) | - | same |
| L31 down c147 | 3% / 0 | **res** (R2 0.62): res in {174..179, 182, 184, 187..189, 194..198} | off (on 0) | - | same |
| L31 down c160 | 16% / 0 | **res** (R2 0.80): res in {74, 83, 87..88, 107..108, 113..114, 117..118, 127..128, 133, 137..138, 147..148, 157..158, 163, 167..168, 173..175, 177..179, 183..184, 187..188} | off (on 0) | - | same |
| L31 down c165 | 6% / 0 | **res** (R2 0.77): res in {146..159} | off (on 0) | - | same |
| L31 down c174 | 43% / 7% | **tens(a,b)** (R2 0.53): a//10 in {0} -> b//10 in {4,5,6,7,8}; a//10 in {1} -> b//10 in {4,5,6,7}; a//10 in {2} -> b//10 in {3,4,5,6}; a//10 in {3} -> b//10 in {1,2,3,4,5}; a//10 in {4} -> b//10 in {0,1,2,3,4}; a//10 in {5} -> b//10 in {0,1,2,3}; a//10 in {6,7} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1}; a//10 in {9} -> b//10 in {7} | unexplained (best R2 0.45) | - | same |
| L31 down c176 | 6% / 1% | **res** (R2 0.80): res in {39, 59, 79, 99, 119, 139, 149, 159, 179, 189, 192, 194..195, 197..199} | unexplained (best R2 0.32) | - | same |
| L31 down c181 | 87% / 92% | unexplained (best R2 0.13) | unexplained (best R2 0.25) | - | same |
| L31 down c184 | 3% / 1% | **res** (R2 0.70): res in {66..68} [coarser: res mod 100 in {66..68}, R2 0.84] | unexplained (best R2 0.19) | - | same |
| L31 down c187 | 13% / 5% | **res%100** (R2 0.58): res mod 100 in {25, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90} | unexplained (best R2 0.49) | - | same |
| L31 down c188 | 2% / 0 | **res** (R2 0.62): res in {146, 148..149, 192, 194..199} | off (on 0) | - | same |
| L31 down c194 | 1% / 0 | **res** (R2 0.67): res in {124} | off (on 0) | - | same |
| L31 down c196 | 1% / 0 | **res%100** (R2 0.72): res mod 100 in {1} | off (on 0) | - | same |
| L31 down c201 | 13% / 0 | **res//10** (R2 0.81): (tens) res in {147, 151..188, 191, 193..196} | off (on 0) | - | same |
| L31 down c211 | 1% / 0 | **res%100** (R2 0.80): res mod 100 in {34} | off (on 0) | - | same |
| L31 down c213 | 18% / 16% | **res//10** (R2 0.60): (tens) res in {3..6, 8, 18, 26..29, 41..42, 44..68} | **tens(a,b)** (R2 0.63): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {1,2,4,5,6}; a//10 in {5,6} -> b//10 in {0}; a//10 in {10} -> b//10 in {10} | - | a: mod50 +3%, mod25 +3%, mod20 +2% |
| L31 down c219 | 5% / 1% | **res%100** (R2 0.80): res mod 100 in {81..86} | **res** (R2 0.52): res in {81..86} | - | same |
| L31 down c234 | 0 / 1% | off (on 0) | **res%100** (R2 0.73): res mod 100: no class above 0.5 (max 0.40) | - | same |
| L31 down c236 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c239 | 4% / 2% | **res%100** (R2 0.73): res mod 100 in {61, 81, 91} | unexplained (best R2 0.20) | - | same |
| L31 down c242 | 5% / 1% | **res** (R2 0.78): res in {87..89, 176..180, 185..200} | unexplained (best R2 0.34) | - | same |
| L31 down c244 | 4% / 0 | **res** (R2 0.90): res in {174..200} | off (on 0) | - | same |
| L31 down c247 | 3% / 0 | **res%100** (R2 0.71): res mod 100 in {84..85, 89} | off (on 0) | - | same |
| L31 down c250 | 2% / 0 | **res** (R2 0.77): res in {46..49} | off (on 0) | - | same |
| L31 down c256 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c259 | 47% / 2% | **res** (R2 0.50): res in {2..70, 75, 79..80, 82..83, 89..90, 92..95, 100, 109, 139, 155, 159..161, 163..165, 167..169, 180..183, 185, 189..190, 193..196, 200} | unexplained (best R2 0.15) | - | same |
| L31 down c271 | 1% / 0 | **res** (R2 0.63): res in {154..156} | off (on 0) | - | same |
| L31 down c273 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c293 | 6% / 0 | **res** (R2 0.65): res in {131..134, 136..139, 141} | off (on 0) | - | same |
| L31 down c296 | 3% / 1% | **res%100** (R2 0.82): res mod 100 in {81..83} | unexplained (best R2 0.31) | - | same |
| L31 down c297 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c301 | 3% / 0 | **res** (R2 0.72): res in {144..149} | off (on 0) | - | same |
| L31 down c306 | 6% / 19% | unexplained (best R2 0.33) | unexplained (best R2 0.42) | - | same |
| L31 down c307 | 11% / 44% | **res//10** (R2 0.64): (tens) res in {2..19, 21..41} | **tens(a,b)** (R2 0.67): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,3,4,5,6,7,8,9}; a//10 in {2} -> b//10 in {0,1,2,3,4,5,6}; a//10 in {3} -> b//10 in {0,3,4,5,6}; a//10 in {4} -> b//10 in {0,4,5,6}; a//10 in {5} -> b//10 in {0,1,6}; a//10 in {6} -> b//10 in {0,7} | - | a: mod100 +9%, mod50 +8%, mod25 +7%, mod20 +6%, mod10 +4%; b: mod50 +2%, mod25 +3%, mod20 +2% |
| L31 down c314 | 4% / 0 | **res** (R2 0.76): res in {113..114, 133..134, 152..154} | off (on 0) | - | same |
| L31 down c315 | 5% / 0 | **res** (R2 0.77): res in {97..99, 149..150, 187..199} | off (on 0) | - | same |
| L31 down c319 | 1% / 0 | **res** (R2 0.82): res in {118} | off (on 0) | - | same |
| L31 down c321 | 4% / 1% | **res** (R2 0.66): res in {80..84} | unexplained (best R2 0.34) | - | same |
| L31 down c328 | 1% / 3% | **res** (R2 0.64): res in {13..15} | **res** (R2 0.70): res in {13..15} | - | same |
| L31 down c351 | 0 / 1% | off (on 0) | unexplained (best R2 0.47) | - | same |
| L31 down c355 | 6% / 3% | **res%100** (R2 0.72): res mod 100 in {66..67, 86..89} | unexplained (best R2 0.32) | - | same |
| L31 down c365 | 12% / 1% | unexplained (best R2 0.26) | unexplained (best R2 0.39) | - | same |
| L31 down c375 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c386 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c387 | 4% / 0 | **res** (R2 0.73): res in {105, 110, 135, 155, 160, 165, 170, 180, 185, 195} | off (on 0) | - | same |
| L31 down c406 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c423 | 1% / 0 | **res** (R2 0.72): res in {168..169} | off (on 0) | - | same |
| L31 down c426 | 17% / 0 | **res** (R2 0.81): res in {34, 36, 116..118, 124..139, 156, 164, 166..167, 174..179} | unexplained (best R2 0.19) | - | same |
| L31 down c430 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c444 | 5% / 0 | **res** (R2 0.82): res in {81, 111, 121, 131, 141, 151, 161, 171, 181} | off (on 0) | - | same |
| L31 down c455 | 1% / 0 | **res** (R2 0.68): res in {141..142} | off (on 0) | - | same |
| L31 down c465 | 2% / 0 | unexplained (best R2 0.19) | off (on 0) | - | same |
| L31 down c466 | 4% / 0 | **res** (R2 0.90): res in {115..119} | off (on 0) | - | same |
| L31 down c474 | 13% / 13% | **res** (R2 0.76): res in {7, 68..81} | **tens(a,b)** (R2 0.53): a//10 in {0} -> b//10 in {7,8}; a//10 in {7} -> b//10 in {0,1,2,3,10}; a//10 in {8} -> b//10 in {0,1,2}; a//10 in {9,10} -> b//10 in {2} | - | same |
| L31 down c477 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c479 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c482 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c484 | 1% / 0 | **res%100** (R2 0.67): res mod 100: no class above 0.5 (max 0.42) | off (on 0) | - | same |
| L31 down c493 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c494 | 2% / 7% | unexplained (best R2 0.47) | **a** (R2 0.66): a in {1..6} | - | same |
| L31 down c495 | 3% / 1% | **res%100** (R2 0.90): res mod 100 in {76..78} | unexplained (best R2 0.29) | - | same |
| L31 down c500 | 2% / 0 | **res** (R2 0.78): res in {111, 131, 151} | off (on 0) | - | same |
| L31 down c522 | 2% / 0 | **res%100** (R2 0.74): res mod 100 in {71, 81} | off (on 0) | - | same |
| L31 down c527 | 2% / 0 | **res%100** (R2 0.84): res mod 100: no class above 0.5 (max 0.46) | off (on 0) | - | same |
| L31 down c531 | 3% / 0 | **res** (R2 0.91): res in {121..124} | off (on 0) | - | same |
| L31 down c533 | 5% / 68% | unexplained (best R2 0.37) | unexplained (best R2 0.47) | - | same |
| L31 down c544 | 6% / 13% | **tens(a,b)** (R2 0.54): a//10 in {0} -> b//10 in {1}; a//10 in {1,2} -> b//10 in {0,1} | unexplained (best R2 0.41) | - | same |
| L31 down c546 | 6% / 0 | **res** (R2 0.76): res in {85, 114..115, 135, 145, 155, 165, 175, 185, 195} | off (on 0) | - | same |
| L31 down c580 | 5% / 0 | **res** (R2 0.81): res in {123, 127..128, 132..133, 138, 142..143} | off (on 0) | - | same |
| L31 down c587 | 35% / 1% | unexplained (best R2 0.47) | unexplained (best R2 0.13) | - | same |
| L31 down c595 | 5% / 52% | **res** (R2 0.63): res in {75, 85, 115, 125, 135, 155, 175} | unexplained (best R2 0.49) | - | res: mod100 +3% |
| L31 down c598 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c612 | 3% / 0 | **res** (R2 0.63): res in {68, 162..164, 167..169, 182..184, 192, 194, 196..198} | off (on 0) | - | same |
| L31 down c622 | 11% / 79% | unexplained (best R2 0.38) | **tens(a,b)** (R2 0.52): a//10 in {0,1} -> b//10 in {0,1,2,3}; a//10 in {2} -> b//10 in {1,2,3,4,5,6}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,7,8,9}; a//10 in {4} -> b//10 in {1,2,3,4,5,6,7,8,9}; a//10 in {5,6,7,8,9} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,4,5,6,7,8,9,10} | - | a: mod100 +5%, mod50 +5%; b: mod50 +3%, mod25 +3%; res: mod100 +7%, mod50 +3% |
| L31 down c623 | 4% / 1% | **res%100** (R2 0.86): res mod 100 in {77..79, 81} | unexplained (best R2 0.34) | - | same |
| L31 down c629 | 0 / 5% | off (on 0) | **a** (R2 0.65): a in {94..99} | - | same |
| L31 down c640 | 23% / 0 | **res%100** (R2 0.76): res mod 100 in {1, 96..98} | off (on 0) | - | same |
| L31 down c641 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c644 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c645 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c652 | 6% / 1% | **res** (R2 0.87): res in {91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191} | unexplained (best R2 0.18) | - | same |
| L31 down c655 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c674 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c675 | 3% / 0 | **res** (R2 0.86): res in {112..114} | off (on 0) | - | same |
| L31 down c680 | 31% / 1% | **res** (R2 0.72): res in {6..31, 33..48, 51, 53..56, 58..59, 61..64, 66..68, 71, 73..74, 76..78, 80..88} | unexplained (best R2 0.15) | - | same |
| L31 down c690 | 0 / 1% | off (on 0) | **a** (R2 0.63): a in {99} | - | same |
| L31 down c695 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c700 | 2% / 0 | **res** (R2 0.61): res in {85, 135, 155, 175, 185} | off (on 0) | - | same |
| L31 down c748 | 10% / 1% | **res** (R2 0.72): res in {59..60, 79..80, 110, 119..120, 130, 140, 150, 155, 159..160, 165, 169..170, 175..180, 190, 195} | unexplained (best R2 0.20) | - | same |
| L31 down c754 | 26% / 62% | unexplained (best R2 0.46) | unexplained (best R2 0.39) | - | same |
| L31 down c756 | 99% / 98% | always | same | - | same |
| L31 down c760 | 2% / 0 | **res//10** (R2 0.82): (tens) res in {182..200} | off (on 0) | - | same |
| L31 down c773 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c776 | 0 / 3% | off (on 0) | **res%100** (R2 0.53): res mod 100: no class above 0.5 (max 0.45) | - | same |
| L31 down c805 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c813 | 9% / 0 | **res** (R2 0.79): res in {101..109, 188..189, 192..199} | off (on 0) | - | same |
| L31 down c819 | 3% / 1% | **res%50** (R2 0.71): res mod 50 in {36} | unexplained (best R2 0.35) | - | same |
| L31 down c844 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c847 | 5% / 0 | **res** (R2 0.74): res in {63, 68, 156, 162..164, 166..169, 174, 182..184} | off (on 0) | - | same |
| L31 down c859 | 1% / 0 | **res** (R2 0.72): res in {141} | off (on 0) | - | same |
| L31 down c867 | 4% / 0 | **res** (R2 0.76): res in {74, 137..138, 147..148, 173..174, 177} | off (on 0) | - | same |
| L31 down c895 | 0 / 0 | off (on 0) | same | - | same |
| L31 down c899 | 25% / 0 | **res** (R2 0.88): res in {105..133} | off (on 0) | - | same |
| L31 down c902 | 4% / 1% | **res** (R2 0.79): res in {60..65} | unexplained (best R2 0.30) | - | same |
| L31 down c921 | 8% / 0 | **res** (R2 0.72): res in {97..98, 102, 117..124} | off (on 0) | - | same |
| L31 down c926 | 6% / 0 | **res** (R2 0.83): res in {139..146, 149} | off (on 0) | - | same |
| L31 down c927 | 3% / 0 | **res** (R2 0.81): res in {113, 115, 117} | off (on 0) | - | same |
| L31 down c983 | 0 / 7% | off (on 0) | unexplained (best R2 0.38) | - | same |
| L31 down c992 | 4% / 0 | **res** (R2 0.85): res in {114..117, 155} | off (on 0) | - | same |
| L31 down c998 | 3% / 0 | **res%100** (R2 0.75): res mod 100: no class above 0.5 (max 0.45) | off (on 0) | - | same |
| L31 down c1023 | 8% / 4% | **res** (R2 0.76): res in {13..15, 71, 90..91, 169..174, 179..200} | unexplained (best R2 0.36) | - | same |

</details>

<details><summary>gate: 83</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L31 gate c5 | 100% / 100% | always | same | (reads) | same |
| L31 gate c9 | 11% / 8% | **res%100** (R2 0.58): res mod 100 in {2..7, 83, 97..98} | unexplained (best R2 0.49) | (reads) | same |
| L31 gate c14 | 100% / 100% | always | same | (reads) | same |
| L31 gate c18 | 0 / 19% | off (on 0) | **a//10** (R2 0.69): (tens) a in {1..19} | (reads) | same |
| L31 gate c20 | 3% / 0 | **res** (R2 0.66): res in {145..149} | off (on 0) | (reads) | same |
| L31 gate c23 | 94% / 98% | unexplained (best R2 0.10) | always | (reads) | same |
| L31 gate c31 | 4% / 0 | **res** (R2 0.67): res in {120, 122..124} | off (on 0) | (reads) | same |
| L31 gate c32 | 15% / 85% | unexplained (best R2 0.40) | unexplained (best R2 0.28) | (reads) | same |
| L31 gate c38 | 100% / 99% | always | same | (reads) | same |
| L31 gate c43 | 1% / 0 | unexplained (best R2 0.13) | off (on 0) | (reads) | same |
| L31 gate c47 | 54% / 5% | **tens(a,b)** (R2 0.56): a//10 in {0} -> b//10 in {8,9,10}; a//10 in {1} -> b//10 in {7,8,9,10}; a//10 in {2} -> b//10 in {6,7,8,9}; a//10 in {3,4} -> b//10 in {5,6,7,8,9,10}; a//10 in {5} -> b//10 in {3,4,7,8,9,10}; a//10 in {6} -> b//10 in {2,3,4,5,8,9,10}; a//10 in {7} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {8,9} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8,9} | unexplained (best R2 0.47) | (reads) | same |
| L31 gate c48 | 0 / 0 | off (on 0) | same | (reads) | same |
| L31 gate c57 | 13% / 2% | **res** (R2 0.74): res in {61, 71, 81, 91, 101, 111, 119..121, 131, 141, 151, 159..161, 170..171, 179..181, 191, 200} | unexplained (best R2 0.16) | (reads) | same |
| L31 gate c58 | 0 / 3% | off (on 0) | **a** (R2 0.79): a in {2..4} | (reads) | same |
| L31 gate c62 | 0 / 0 | off (on 0) | same | (reads) | same |
| L31 gate c63 | 1% / 2% | unexplained (best R2 0.33) | unexplained (best R2 0.24) | (reads) | same |
| L31 gate c69 | 1% / 0 | **res%100** (R2 0.83): res mod 100 in {81} | off (on 0) | (reads) | same |
| L31 gate c72 | 16% / 6% | **res%100** (R2 0.71): res mod 100 in {58..59, 68..80} | unexplained (best R2 0.43) | (reads) | same |
| L31 gate c77 | 100% / 100% | always | same | (reads) | same |
| L31 gate c78 | 15% / 1% | **res** (R2 0.72): res in {34, 36, 66, 76, 84..86, 106, 115..116, 124..126, 131, 134..136, 146, 154..156, 164..166, 175..176, 185..186} | unexplained (best R2 0.26) | (reads) | same |
| L31 gate c80 | 5% / 0 | **res** (R2 0.65): res in {113, 132..134, 136..138, 153, 163} | off (on 0) | (reads) | same |
| L31 gate c82 | 3% / 0 | **res** (R2 0.68): res in {176..200} | off (on 0) | (reads) | same |
| L31 gate c95 | 19% / 9% | **res//10** (R2 0.72): (tens) res in {71..89, 171, 173, 177..178, 183, 188} [coarser: res mod 100 in {71..89}, R2 0.83] | **tens(a,b)** (R2 0.53): a//10 in {0} -> b//10 in {7,8}; a//10 in {7} -> b//10 in {0}; a//10 in {8} -> b//10 in {0,1,10}; a//10 in {9} -> b//10 in {1,2,10}; a//10 in {10} -> b//10 in {1,2} | (reads) | same |
| L31 gate c105 | 1% / 0 | **res** (R2 0.55): res in {189, 192, 194..199} | off (on 0) | (reads) | same |
| L31 gate c106 | 0 / 0 | off (on 0) | same | (reads) | same |
| L31 gate c107 | 18% / 46% | unexplained (best R2 0.15) | unexplained (best R2 0.28) | (reads) | same |
| L31 gate c112 | 11% / 46% | unexplained (best R2 0.44) | unexplained (best R2 0.39) | (reads) | same |
| L31 gate c115 | 9% / 1% | **res** (R2 0.64): res in {85, 105, 110, 115, 120, 125, 130, 135, 145, 150, 155, 160, 165, 175, 180, 185, 195} | unexplained (best R2 0.13) | (reads) | same |
| L31 gate c117 | 1% / 11% | unexplained (best R2 0.43) | **tens(a,b)** (R2 0.67): a//10 in {0} -> b//10 in {0,1,2,3,5}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2,3} -> b//10 in {0} | (reads) | same |
| L31 gate c119 | 0 / 0 | off (on 0) | same | (reads) | same |
| L31 gate c123 | 10% / 1% | **res** (R2 0.77): res in {71, 101, 111, 131, 136..143, 151, 161, 171} | unexplained (best R2 0.11) | (reads) | same |
| L31 gate c125 | 0 / 0 | off (on 0) | same | (reads) | same |
| L31 gate c128 | 6% / 1% | **res** (R2 0.65): res in {83..86, 179..200} | unexplained (best R2 0.21) | (reads) | same |
| L31 gate c137 | 2% / 0 | **res%100** (R2 0.86): res mod 100 in {97..98} | off (on 0) | (reads) | same |
| L31 gate c138 | 0 / 0 | off (on 0) | same | (reads) | same |
| L31 gate c143 | 9% / 1% | **res%100** (R2 0.67): res mod 100 in {59, 61..69} | unexplained (best R2 0.32) | (reads) | same |
| L31 gate c145 | 9% / 44% | unexplained (best R2 0.47) | unexplained (best R2 0.33) | (reads) | same |
| L31 gate c146 | 9% / 27% | unexplained (best R2 0.40) | **tens(a,b)** (R2 0.53): a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {2}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4,5,10}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9,10}; a//10 in {8} -> b//10 in {6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {4} | (reads) | same |
| L31 gate c153 | 4% / 0 | **res%100** (R2 0.54): res mod 100 in {1, 81} | off (on 0) | (reads) | same |
| L31 gate c158 | 3% / 2% | **units(a,b)** (R2 0.62): a%10 in {0} -> b%10 in {0}; a%10 in {5} -> b%10 in {0,5} | unexplained (best R2 0.32) | (reads) | same |
| L31 gate c161 | 0 / 0 | off (on 0) | same | (reads) | same |
| L31 gate c164 | 7% / 0 | **res** (R2 0.73): res in {49, 59, 79, 89, 99, 119..120, 129, 139, 149, 159, 179, 189} | off (on 0) | (reads) | same |
| L31 gate c171 | 5% / 0 | **res** (R2 0.66): res in {68, 107..109, 161..164, 167..169} | off (on 0) | (reads) | same |
| L31 gate c182 | 7% / 1% | **res//10** (R2 0.69): (tens) res in {81..89} | unexplained (best R2 0.37) | (reads) | same |
| L31 gate c192 | 10% / 0 | unexplained (best R2 0.41) | off (on 0) | (reads) | same |
| L31 gate c205 | 2% / 4% | unexplained (best R2 0.13) | **a** (R2 0.71): a in {2..4} | (reads) | same |
| L31 gate c212 | 2% / 0 | **res** (R2 0.70): res in {97..98, 193..199} | off (on 0) | (reads) | same |
| L31 gate c233 | 2% / 0 | **res%100** (R2 0.76): res mod 100 in {69} | off (on 0) | (reads) | same |
| L31 gate c238 | 100% / 100% | always | same | (reads) | same |
| L31 gate c244 | 0 / 0 | off (on 0) | same | (reads) | same |
| L31 gate c249 | 0 / 3% | off (on 0) | **a** (R2 0.80): a in {1..4} | (reads) | same |
| L31 gate c251 | 0 / 0 | off (on 0) | same | (reads) | same |
| L31 gate c252 | 7% / 95% | unexplained (best R2 0.31) | unexplained (best R2 0.28) | (reads) | same |
| L31 gate c257 | 14% / 19% | **res//10** (R2 0.83): (tens) res in {2..51, 200} | **tens(a,b)** (R2 0.57): a//10 in {0} -> b//10 in {0,1,2,3,4}; a//10 in {1} -> b//10 in {0,1,2,3}; a//10 in {2} -> b//10 in {0,2,3}; a//10 in {3} -> b//10 in {0,3,4}; a//10 in {4} -> b//10 in {0,4}; a//10 in {5} -> b//10 in {0} | (reads) | same |
| L31 gate c260 | 37% / 2% | **res%100** (R2 0.49): res mod 100 in {1..4, 7, 17, 27, 37..38, 47, 50, 57..58, 73..75, 77..79, 83, 87..90, 92..98} | unexplained (best R2 0.24) | (reads) | same |
| L31 gate c268 | 14% / 0 | **res//10** (R2 0.94): (tens) res in {149..199} | off (on 0) | (reads) | same |
| L31 gate c279 | 1% / 0 | unexplained (best R2 0.35) | off (on 0) | (reads) | same |
| L31 gate c288 | 100% / 100% | always | same | (reads) | same |
| L31 gate c292 | 0 / 0 | off (on 0) | same | (reads) | same |
| L31 gate c321 | 3% / 0 | **res** (R2 0.68): res in {107, 117, 127, 137, 147} | off (on 0) | (reads) | same |
| L31 gate c328 | 3% / 7% | **res//10** (R2 0.71): (tens) res in {19..29} | unexplained (best R2 0.36) | (reads) | same |
| L31 gate c334 | 12% / 0 | **res** (R2 0.91): res in {107..119} | off (on 0) | (reads) | same |
| L31 gate c336 | 20% / 0 | **res** (R2 0.68): res in {58, 68, 74, 78, 88, 114, 116..118, 127..128, 133..134, 136..138, 147..148, 153..160, 162..164, 166..168, 170, 173..179, 183..184, 187..188} | off (on 0) | (reads) | same |
| L31 gate c343 | 0 / 0 | off (on 0) | same | (reads) | same |
| L31 gate c346 | 3% / 0 | **res** (R2 0.70): res in {98, 101, 111, 151} | off (on 0) | (reads) | same |
| L31 gate c350 | 3% / 0 | **res%100** (R2 0.89): res mod 100 in {97..98} | off (on 0) | (reads) | same |
| L31 gate c360 | 3% / 0 | **res%100** (R2 0.70): res mod 100 in {93..96} | off (on 0) | (reads) | same |
| L31 gate c368 | 1% / 33% | unexplained (best R2 0.19) | unexplained (best R2 0.41) | (reads) | same |
| L31 gate c376 | 33% / 0 | unexplained (best R2 0.30) | off (on 0) | (reads) | same |
| L31 gate c389 | 3% / 0 | **res** (R2 0.59): res in {61, 111, 131, 161, 171} | off (on 0) | (reads) | same |
| L31 gate c391 | 12% / 0 | **res** (R2 0.75): res in {34, 83..85, 114..117, 134..135, 154..155, 164..167, 174..175, 183..185} | off (on 0) | (reads) | same |
| L31 gate c398 | 13% / 0 | **res** (R2 0.67): res in {81, 111, 115..116, 119..121, 125..126, 130..131, 135, 151, 155..156, 159..161, 166, 170..171, 175..176, 179..181} | off (on 0) | (reads) | same |
| L31 gate c407 | 2% / 0 | **res** (R2 0.69): res in {111..113} | off (on 0) | (reads) | same |
| L31 gate c409 | 54% / 1% | unexplained (best R2 0.40) | unexplained (best R2 0.36) | (reads) | same |
| L31 gate c417 | 23% / 1% | **res** (R2 0.59): res in {51, 61, 71, 81..86, 91, 121, 141..143, 145..158, 161..165, 171, 173..174, 180..186, 191} | unexplained (best R2 0.21) | (reads) | same |
| L31 gate c418 | 33% / 1% | **res//10** (R2 0.58): (tens) res in {122..124, 126..200} | unexplained (best R2 0.35) | (reads) | same |
| L31 gate c428 | 13% / 0 | **res** (R2 0.61): res in {81..83, 121, 141..143, 146..164, 181..183} | off (on 0) | (reads) | same |
| L31 gate c447 | 13% / 46% | **res** (R2 0.68): res in {3, 5..6, 8..9, 28..39, 51..68} | unexplained (best R2 0.48) | (reads) | same |
| L31 gate c469 | 25% / 45% | unexplained (best R2 0.18) | unexplained (best R2 0.32) | (reads) | same |
| L31 gate c481 | 0 / 0 | off (on 0) | same | (reads) | same |
| L31 gate c507 | 38% / 2% | **res//10** (R2 0.74): (tens) res in {4..90} | unexplained (best R2 0.28) | (reads) | same |
| L31 gate c659 | 0 / 5% | off (on 0) | **a** (R2 0.67): a in {94..99} | (reads) | same |
| L31 gate c735 | 9% / 0 | **res%100** (R2 0.71): res mod 100 in {96..97} | off (on 0) | (reads) | same |

</details>

<details><summary>k: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L31 k c1 (kv3) | 89% / 3% | unexplained (best R2 0.18) | unexplained (best R2 0.32) | (reads) | same |

</details>

<details><summary>o: 6</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L31 o c7 (H14) | 100% / 100% | always | same | - | same |
| L31 o c18 (H14) | 0 / 0 | off (on 0) | same | - | same |
| L31 o c58 (H14) | 0 / 0 | off (on 0) | same | - | same |
| L31 o c77 (H12) | 0 / 0 | off (on 0) | same | - | same |
| L31 o c135 (H12) | 100% / 100% | always | same | - | same |
| L31 o c165 (H15) | 98% / 33% | always | **tens(a,b)** (R2 0.52): a//10 in {0,1} -> b//10 in {0,10}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3,4} -> b//10 in {0}; a//10 in {5} -> b//10 in {0,4,5}; a//10 in {6} -> b//10 in {5,6}; a//10 in {7} -> b//10 in {0,6}; a//10 in {8} -> b//10 in {0,1,4,6,7,8}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | - | same |

</details>

<details><summary>q: 2</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L31 q c0 (H2) | 100% / 100% | always | same | (reads) | same |
| L31 q c1 (H14) | 100% / 100% | always | same | (reads) | same |

</details>

<details><summary>up: 53</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L31 up c7 | 19% / 100% | unexplained (best R2 0.39) | always | (reads) | same |
| L31 up c10 | 99% / 99% | always | same | (reads) | same |
| L31 up c13 | 0 / 0 | off (on 0) | same | (reads) | same |
| L31 up c16 | 100% / 100% | always | same | (reads) | same |
| L31 up c19 | 88% / 93% | unexplained (best R2 0.12) | unexplained (best R2 0.25) | (reads) | same |
| L31 up c29 | 0 / 0 | off (on 0) | same | (reads) | same |
| L31 up c36 | 100% / 100% | always | same | (reads) | same |
| L31 up c50 | 100% / 100% | always | same | (reads) | same |
| L31 up c66 | 0 / 0 | off (on 0) | same | (reads) | same |
| L31 up c83 | 16% / 0 | **res//10** (R2 0.84): (tens) res in {101..118} | off (on 0) | (reads) | same |
| L31 up c108 | 17% / 0 | **res** (R2 0.65): res in {61, 68, 71, 74, 83..84, 114, 134, 138, 146..149, 151..169, 171, 173..174, 183..184} | off (on 0) | (reads) | same |
| L31 up c111 | 3% / 1% | unexplained (best R2 0.40) | unexplained (best R2 0.30) | (reads) | same |
| L31 up c140 | 0 / 7% | off (on 0) | **a** (R2 0.77): a in {1..6} | (reads) | same |
| L31 up c147 | 5% / 0 | **res** (R2 0.71): res in {113, 131..134, 136..139} | off (on 0) | (reads) | same |
| L31 up c162 | 14% / 0 | **res** (R2 0.62): res in {81..83, 117..119, 121..124, 143, 147..149, 161..164, 167..169, 174, 179, 181..184, 192, 197..198} | off (on 0) | (reads) | same |
| L31 up c169 | 15% / 0 | **res** (R2 0.60): res in {59, 61, 71, 79, 81, 85, 105, 109, 111, 115, 135, 145, 149..151, 155, 159..161, 165, 171, 175, 179..181, 185, 189, 195} | off (on 0) | (reads) | same |
| L31 up c176 | 1% / 0 | unexplained (best R2 0.35) | off (on 0) | (reads) | same |
| L31 up c186 | 10% / 0 | **res%100** (R2 0.66): res mod 100 in {1, 94, 96..99} | off (on 0) | (reads) | same |
| L31 up c197 | 94% / 26% | unexplained (best R2 0.46) | unexplained (best R2 0.48) | (reads) | same |
| L31 up c217 | 6% / 0 | **res** (R2 0.80): res in {111..117} | off (on 0) | (reads) | same |
| L31 up c237 | 5% / 20% | unexplained (best R2 0.41) | unexplained (best R2 0.50) | (reads) | same |
| L31 up c247 | 13% / 1% | **res** (R2 0.67): res in {85..89, 150..151, 154..158, 160, 167, 170..190, 194..198, 200} | unexplained (best R2 0.32) | (reads) | same |
| L31 up c307 | 0 / 0 | off (on 0) | same | (reads) | same |
| L31 up c317 | 0 / 0 | off (on 0) | same | (reads) | same |
| L31 up c345 | 3% / 0 | **res** (R2 0.68): res in {115..118} | off (on 0) | (reads) | same |
| L31 up c376 | 0 / 3% | off (on 0) | **a** (R2 0.82): a in {2..3} | (reads) | same |
| L31 up c419 | 24% / 0 | **res** (R2 0.86): res in {96..120, 188..190, 192, 194..199} | off (on 0) | (reads) | same |
| L31 up c436 | 75% / 4% | unexplained (best R2 0.22) | unexplained (best R2 0.43) | (reads) | same |
| L31 up c466 | 37% / 1% | **res//10** (R2 0.50): (tens) res in {106..107, 109..119, 121..139, 141..149, 151..188, 194..200} | unexplained (best R2 0.28) | (reads) | same |
| L31 up c472 | 2% / 0 | **res** (R2 0.69): res in {98, 117..118} | off (on 0) | (reads) | same |
| L31 up c497 | 0 / 0 | off (on 0) | same | (reads) | same |
| L31 up c534 | 100% / 100% | always | same | (reads) | same |
| L31 up c535 | 1% / 0 | **res%100** (R2 0.66): res mod 100 in {68} | off (on 0) | (reads) | same |
| L31 up c538 | 0 / 0 | off (on 0) | same | (reads) | same |
| L31 up c559 | 3% / 0 | **res** (R2 0.67): res in {175..182, 184, 186..188, 190, 194..200} | off (on 0) | (reads) | same |
| L31 up c568 | 84% / 80% | unexplained (best R2 0.21) | unexplained (best R2 0.20) | (reads) | same |
| L31 up c570 | 57% / 3% | **res//10** (R2 0.67): (tens) res in {87..116, 118..200} | unexplained (best R2 0.42) | (reads) | same |
| L31 up c572 | 14% / 32% | **tens(a,b)** (R2 0.51): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,10}; a//10 in {1} -> b//10 in {0,1}; a//10 in {2,3} -> b//10 in {0}; a//10 in {10} -> b//10 in {10} | **tens(a,b)** (R2 0.57): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,3,4,5,6,7,8,10}; a//10 in {2} -> b//10 in {0,1,2,3,4,10}; a//10 in {3} -> b//10 in {0,4,10}; a//10 in {4,5} -> b//10 in {0}; a//10 in {6} -> b//10 in {0,6,7}; a//10 in {10} -> b//10 in {10} | (reads) | same |
| L31 up c640 | 39% / 1% | **res** (R2 0.63): res in {2..50, 55..56, 63..67, 83..86, 111..117, 121, 123..136, 145..146, 155..156, 165..166, 185} | unexplained (best R2 0.09) | (reads) | same |
| L31 up c662 | 17% / 0 | **res%100** (R2 0.68): res mod 100 in {1, 92..99} | off (on 0) | (reads) | same |
| L31 up c664 | 12% / 0 | **res** (R2 0.77): res in {61, 81..84, 111, 117..124, 161, 181..182} | off (on 0) | (reads) | same |
| L31 up c681 | 3% / 0 | unexplained (best R2 0.45) | off (on 0) | (reads) | same |
| L31 up c731 | 100% / 100% | always | same | (reads) | same |
| L31 up c734 | 7% / 2% | **units(a,b)** (R2 0.57): a%10 in {0,5} -> b%10 in {0,5} | unexplained (best R2 0.48) | (reads) | same |
| L31 up c746 | 10% / 0 | **res** (R2 0.74): res in {111, 123, 127..134, 137..143} | off (on 0) | (reads) | same |
| L31 up c773 | 14% / 97% | unexplained (best R2 0.38) | always | (reads) | same |
| L31 up c787 | 44% / 10% | **res//10** (R2 0.67): (tens) res in {2..13, 15..18, 20..25, 27, 30, 32, 50..100, 190} | **tens(a,b)** (R2 0.53): a//10 in {6,7} -> b//10 in {0}; a//10 in {8} -> b//10 in {0,1}; a//10 in {9} -> b//10 in {0,1,3,10}; a//10 in {10} -> b//10 in {0,1,2,3,4} | (reads) | same |
| L31 up c834 | 23% / 0 | **res** (R2 0.71): res in {68, 113..114, 131..144, 146..184, 188} | off (on 0) | (reads) | same |
| L31 up c839 | 66% / 3% | **tens(a,b)** (R2 0.67): a//10 in {0,1,2} -> b//10 in {6,7,8,9,10}; a//10 in {3} -> b//10 in {5,6,7,8,9,10}; a//10 in {4} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {6,7} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {0,2,3,4,5,6,7,8,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | unexplained (best R2 0.25) | (reads) | same |
| L31 up c845 | 21% / 10% | **res%100** (R2 0.67): res mod 100 in {73..93, 95..96} | **tens(a,b)** (R2 0.51): a//10 in {8,10} -> b//10 in {0,1,2}; a//10 in {9} -> b//10 in {0,1,2,3,10} | (reads) | same |
| L31 up c864 | 9% / 1% | **res%100** (R2 0.52): res mod 100 in {34, 36..39, 43, 63, 83, 93} | unexplained (best R2 0.22) | (reads) | same |
| L31 up c878 | 4% / 0 | **res** (R2 0.61): res in {123, 127..130, 133} | off (on 0) | (reads) | same |
| L31 up c909 | 0 / 0 | off (on 0) | same | (reads) | same |

</details>

<details><summary>v: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L31 v c139 (kv7) | 67% / 93% | unexplained (best R2 0.32) | unexplained (best R2 0.21) | (reads) | same |

</details>

</details>

