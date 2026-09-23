# §5 — complete lists for the operator

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

## L19-L31 `=` components: does the same component respond to the same residues of res on both operations?

<details><summary>all 1995, by decreasing correlation</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) | corr of on-rate over res mod 100, add vs sub |
|---|---|---|---|---|---|---|
| L23 o c311 (H15) | 0 / 0 | off (on 0) | same | - | same | +1.00 |
| L26 down c137 | 0 / 0 | off (on 0) | same | - | same | +1.00 |
| L29 gate c605 | 0 / 0 | off (on 0) | same | (reads) | same | +1.00 |
| L29 gate c733 | 0 / 1% | off (on 0) | **res** (R2 0.82): res in {12} | (reads) | same | +1.00 |
| L29 gate c856 | 0 / 1% | off (on 0) | **res** (R2 0.91): res in {12} | (reads) | same | +1.00 |
| L29 gate c910 | 0 / 0 | off (on 0) | same | (reads) | same | +1.00 |
| L29 up c211 | 0 / 1% | off (on 0) | **res%100** (R2 0.78): res mod 100: no class above 0.5 (max 0.43) | (reads) | same | +1.00 |
| L29 up c488 | 0 / 0 | off (on 0) | same | (reads) | same | +1.00 |
| L29 up c653 | 1% / 0 | **res%100** (R2 0.71): res mod 100 in {62} | off (on 0) | (reads) | same | +1.00 |
| L29 up c804 | 1% / 0 | **res%100** (R2 0.83): res mod 100 in {62} | off (on 0) | (reads) | same | +1.00 |
| L29 up c817 | 0 / 0 | off (on 0) | same | (reads) | same | +1.00 |
| L30 up c687 | 0 / 0 | off (on 0) | same | (reads) | same | +1.00 |
| L30 up c787 | 1% / 0 | **res%100** (R2 0.64): res mod 100 in {70} | off (on 0) | (reads) | same | +1.00 |
| L30 up c855 | 0 / 0 | off (on 0) | same | (reads) | same | +1.00 |
| L28 up c666 | 0 / 0 | off (on 0) | same | (reads) | same | +1.00 |
| L30 up c649 | 0 / 0 | off (on 0) | same | (reads) | same | +1.00 |
| L30 up c535 | 1% / 0 | **res%100** (R2 0.69): res mod 100 in {70} | off (on 0) | (reads) | same | +1.00 |
| L29 down c560 | 0 / 0 | off (on 0) | same | - | same | +1.00 |
| L28 up c848 | 1% / 0 | **res%100** (R2 0.74): res mod 100 in {76} | off (on 0) | (reads) | same | +1.00 |
| L30 gate c170 | 0 / 0 | off (on 0) | same | (reads) | same | +1.00 |
| L29 up c865 | 0 / 1% | off (on 0) | **res%100** (R2 0.79): res mod 100: no class above 0.5 (max 0.41) | (reads) | same | +1.00 |
| L26 down c779 | 0 / 1% | off (on 0) | **res** (R2 0.84): res in {10} | - | same | +1.00 |
| L27 gate c461 | 1% / 0 | **res%100** (R2 0.81): res mod 100 in {56} | off (on 0) | (reads) | same | +1.00 |
| L29 gate c493 | 0 / 1% | off (on 0) | **res** (R2 0.89): res in {12} | (reads) | same | +1.00 |
| L27 gate c142 | 1% / 0 | **res%100** (R2 0.76): res mod 100 in {49} | off (on 0) | (reads) | same | +1.00 |
| L29 up c875 | 1% / 0 | **res%100** (R2 0.66): res mod 100 in {62} | off (on 0) | (reads) | same | +1.00 |
| L29 gate c281 | 0 / 1% | off (on 0) | **res** (R2 0.87): res in {12} | (reads) | same | +1.00 |
| L29 up c778 | 0 / 1% | off (on 0) | **res%100** (R2 0.81): res mod 100: no class above 0.5 (max 0.42) | (reads) | same | +1.00 |
| L29 up c116 | 1% / 0 | **res** (R2 0.89): res in {138} | off (on 0) | (reads) | same | +1.00 |
| L28 up c90 | 1% / 0 | **res%100** (R2 0.75): res mod 100 in {43} | off (on 0) | (reads) | same | +1.00 |
| L24 gate c694 | 1% / 1% | **res%100** (R2 0.68): res mod 100: no class above 0.5 (max 0.37) | **res%100** (R2 0.66): res mod 100: no class above 0.5 (max 0.37) | (reads) | same | +1.00 |
| L30 gate c763 | 1% / 0 | **res%100** (R2 0.91): res mod 100 in {63} | off (on 0) | (reads) | same | +1.00 |
| L29 gate c318 | 1% / 0 | **res%100** (R2 0.52): res mod 100: no class above 0.5 (max 0.48) | off (on 0) | (reads) | same | +1.00 |
| L28 up c298 | 1% / 0 | **res%100** (R2 0.75): res mod 100 in {43} | off (on 0) | (reads) | same | +1.00 |
| L29 down c905 | 0 / 0 | off (on 0) | same | - | same | +1.00 |
| L29 gate c868 | 0 / 1% | off (on 0) | **res** (R2 0.75): res in {13} | (reads) | same | +1.00 |
| L29 gate c58 | 0 / 1% | off (on 0) | **res** (R2 0.87): res in {12} | (reads) | same | +1.00 |
| L30 gate c268 | 1% / 0 | **res%100** (R2 0.90): res mod 100 in {63} | off (on 0) | (reads) | same | +1.00 |
| L29 up c836 | 0 / 1% | off (on 0) | **res** (R2 0.68): res in {13} | (reads) | same | +1.00 |
| L30 down c739 | 1% / 1% | **res%100** (R2 0.84): res mod 100 in {19} | **res** (R2 0.66): res in {19} | - | same | +1.00 |
| L29 up c69 | 1% / 0 | **res%100** (R2 0.79): res mod 100 in {36} | off (on 0) | (reads) | same | +1.00 |
| L28 up c51 | 0 / 1% | off (on 0) | **res** (R2 0.60): res in {18} | (reads) | same | +1.00 |
| L28 down c316 | 1% / 0 | **res** (R2 0.71): res in {154} [coarser: res mod 100 in {54}, R2 0.86] | off (on 0) | - | same | +1.00 |
| L29 down c618 | 1% / 1% | **res%100** (R2 0.87): res mod 100 in {27} | **res** (R2 0.76): res in {27} | - | same | +1.00 |
| L28 up c434 | 1% / 0 | **res** (R2 0.88): res in {129} | off (on 0) | (reads) | same | +1.00 |
| L27 gate c839 | 1% / 0 | **res%100** (R2 0.87): res mod 100 in {69} | off (on 0) | (reads) | same | +1.00 |
| L29 up c300 | 1% / 0 | **res%100** (R2 0.80): res mod 100 in {36} | off (on 0) | (reads) | same | +1.00 |
| L28 up c749 | 1% / 0 | **res%100** (R2 0.74): res mod 100 in {29} | off (on 0) | (reads) | same | +1.00 |
| L29 gate c409 | 1% / 0 | **res%100** (R2 0.67): res mod 100 in {41} | off (on 0) | (reads) | same | +1.00 |
| L29 gate c438 | 1% / 0 | **res%100** (R2 0.57): res mod 100: no class above 0.5 (max 0.39) | off (on 0) | (reads) | same | +1.00 |
| L29 gate c684 | 0 / 1% | off (on 0) | **res** (R2 0.85): res in {13} | (reads) | same | +1.00 |
| L28 up c729 | 1% / 0 | **res%100** (R2 0.78): res mod 100 in {29} | off (on 0) | (reads) | same | +1.00 |
| L29 up c618 | 1% / 1% | **res%100** (R2 0.93): res mod 100 in {28} | **res** (R2 0.62): res in {28} | (reads) | same | +1.00 |
| L25 up c479 | 0 / 2% | off (on 0) | **res** (R2 0.82): res in {10..11} | (reads) | same | +1.00 |
| L27 up c745 | 1% / 0 | **res%100** (R2 0.77): res mod 100 in {35} | off (on 0) | (reads) | same | +1.00 |
| L26 down c486 | 0 / 1% | off (on 0) | **res%100** (R2 0.75): res mod 100: no class above 0.5 (max 0.42) | - | same | +1.00 |
| L31 down c83 | 0 / 1% | off (on 0) | **res** (R2 0.52): res in {20} | - | same | +1.00 |
| L29 gate c238 | 1% / 0 | **res%100** (R2 0.73): res mod 100 in {79} | off (on 0) | (reads) | same | +1.00 |
| L29 gate c883 | 0 / 1% | off (on 0) | **res** (R2 0.83): res in {13} | (reads) | same | +1.00 |
| L28 gate c612 | 1% / 1% | **res%100** (R2 0.93): res mod 100 in {18} | **res** (R2 0.69): res in {18} | (reads) | same | +1.00 |
| L28 up c760 | 1% / 0 | **res%100** (R2 0.82): res mod 100 in {30} | off (on 0) | (reads) | same | +1.00 |
| L28 gate c315 | 1% / 1% | **res%100** (R2 0.94): res mod 100 in {18} | **res** (R2 0.68): res in {18} | (reads) | same | +1.00 |
| L27 up c320 | 1% / 0 | **res%100** (R2 0.73): res mod 100 in {84} | off (on 0) | (reads) | same | +1.00 |
| L28 gate c788 | 1% / 1% | **res%100** (R2 0.94): res mod 100 in {18} | **res** (R2 0.68): res in {18} | (reads) | same | +1.00 |
| L29 up c150 | 1% / 0 | **res%100** (R2 0.84): res mod 100 in {68} | off (on 0) | (reads) | same | +1.00 |
| L29 gate c534 | 0 / 1% | off (on 0) | **res%100** (R2 0.78): res mod 100: no class above 0.5 (max 0.46) | (reads) | same | +1.00 |
| L30 down c879 | 1% / 1% | **res%100** (R2 0.64): res mod 100 in {16} | **res** (R2 0.65): res in {16} | - | same | +1.00 |
| L29 up c244 | 0 / 1% | off (on 0) | **res** (R2 0.81): res in {11} | (reads) | same | +1.00 |
| L29 gate c681 | 0 / 1% | off (on 0) | **res** (R2 0.84): res in {13} | (reads) | same | +1.00 |
| L25 up c873 | 1% / 0 | **res%100** (R2 0.71): res mod 100: no class above 0.5 (max 0.39) | off (on 0) | (reads) | same | +1.00 |
| L29 gate c711 | 0 / 1% | off (on 0) | **res%100** (R2 0.76): res mod 100: no class above 0.5 (max 0.46) | (reads) | same | +1.00 |
| L29 gate c546 | 0 / 1% | off (on 0) | **res%100** (R2 0.75): res mod 100 in {6} | (reads) | same | +1.00 |
| L26 gate c813 | 1% / 1% | **res%100** (R2 0.68): res mod 100 in {20} | **res** (R2 0.57): res in {20} | (reads) | same | +1.00 |
| L29 up c197 | 2% / 1% | **res%100** (R2 0.83): res mod 100 in {38..39} | unexplained (best R2 0.41) | (reads) | same | +1.00 |
| L27 up c360 | 3% / 2% | **res%100** (R2 0.92): res mod 100 in {35..37} | **res** (R2 0.53): res in {35..37} | (reads) | same | +1.00 |
| L28 gate c473 | 1% / 0 | **res%100** (R2 0.90): res mod 100 in {30} | off (on 0) | (reads) | same | +1.00 |
| L26 down c205 | 5% / 5% | **res%100** (R2 0.83): res mod 100 in {10..15} | **res** (R2 0.79): res in {10..15} | - | same | +1.00 |
| L29 gate c481 | 0 / 1% | off (on 0) | **res** (R2 0.75): res in {13} | (reads) | same | +1.00 |
| L30 gate c879 | 1% / 1% | **res%100** (R2 0.87): res mod 100 in {16} | **res** (R2 0.75): res in {16} | (reads) | same | +1.00 |
| L28 down c752 | 2% / 0 | **res%100** (R2 0.69): res mod 100 in {61..62} | off (on 0) | - | same | +0.99 |
| L30 down c800 | 0 / 1% | off (on 0) | **res** (R2 0.73): res in {20} | - | same | +0.99 |
| L26 gate c182 | 3% / 2% | **res%100** (R2 0.88): res mod 100 in {31..33} | **res** (R2 0.57): res in {31..33} | (reads) | same | +0.99 |
| L30 up c294 | 1% / 1% | **res%100** (R2 0.89): res mod 100 in {19} | **res** (R2 0.76): res in {19} | (reads) | same | +0.99 |
| L30 down c642 | 1% / 1% | **res%100** (R2 0.59): res mod 100 in {18} | **res** (R2 0.58): res in {18} | - | same | +0.99 |
| L27 up c336 | 2% / 1% | **res%100** (R2 0.75): res mod 100 in {35..36} | unexplained (best R2 0.32) | (reads) | same | +0.99 |
| L26 up c237 | 1% / 0 | **res%100** (R2 0.52): res mod 100 in {31} | off (on 0) | (reads) | same | +0.99 |
| L29 gate c324 | 0 / 1% | off (on 0) | **res** (R2 0.77): res in {13} | (reads) | same | +0.99 |
| L30 down c414 | 2% / 0 | **res%50** (R2 0.84): res mod 50 in {42} | off (on 0) | - | same | +0.99 |
| L28 down c990 | 1% / 3% | **res** (R2 0.72): res in {24..28} | unexplained (best R2 0.49) | - | same | +0.99 |
| L29 gate c792 | 1% / 0 | **res%100** (R2 0.92): res mod 100 in {34} | off (on 0) | (reads) | same | +0.99 |
| L26 up c875 | 1% / 1% | **res%100** (R2 0.88): res mod 100 in {31} | **res** (R2 0.51): res in {31} | (reads) | same | +0.99 |
| L28 gate c758 | 2% / 0 | **res%100** (R2 0.86): res mod 100 in {54..55} | off (on 0) | (reads) | same | +0.99 |
| L29 down c414 | 1% / 0 | **res%100** (R2 0.76): res mod 100 in {94} | off (on 0) | - | same | +0.99 |
| L24 up c158 | 5% / 1% | **res%100** (R2 0.76): res mod 100 in {49..54} | unexplained (best R2 0.28) | (reads) | same | +0.99 |
| L30 down c628 | 2% / 1% | **res%100** (R2 0.77): res mod 100 in {23..24} | **res** (R2 0.60): res in {23..24} | - | same | +0.99 |
| L24 up c482 | 1% / 0 | **res%100** (R2 0.84): res mod 100 in {34} | off (on 0) | (reads) | same | +0.99 |
| L23 gate c15 | 17% / 12% | **res%100** (R2 0.90): res mod 100 in {26..41} | **res** (R2 0.64): res in {26..41} | (reads) | same | +0.99 |
| L22 up c27 | 1% / 0 | **res%100** (R2 0.82): res mod 100 in {22} | off (on 0) | (reads) | same | +0.99 |
| L30 gate c806 | 1% / 1% | **res%100** (R2 0.82): res mod 100: no class above 0.5 (max 0.43) | **res%100** (R2 0.70): res mod 100 in {2} | (reads) | same | +0.99 |
| L30 down c831 | 2% / 0 | **res** (R2 0.84): res in {57, 59, 157..159} [coarser: res mod 100 in {57..59}, R2 0.89] | off (on 0) | - | same | +0.99 |
| L28 gate c294 | 4% / 3% | **res%100** (R2 0.88): res mod 100 in {12..15} | **res** (R2 0.81): res in {12..15} | (reads) | same | +0.99 |
| L29 up c136 | 2% / 1% | **res%100** (R2 0.74): res mod 100 in {41..42} | unexplained (best R2 0.33) | (reads) | same | +0.99 |
| L30 gate c327 | 0 / 0 | off (on 0) | same | (reads) | same | +0.99 |
| L30 up c565 | 0 / 0 | off (on 0) | same | (reads) | same | +0.99 |
| L29 gate c821 | 0 / 1% | off (on 0) | **res%100** (R2 0.73): res mod 100: no class above 0.5 (max 0.48) | (reads) | same | +0.99 |
| L29 gate c726 | 0 / 1% | off (on 0) | **res%100** (R2 0.69): res mod 100 in {7} | (reads) | same | +0.99 |
| L27 gate c254 | 0 / 0 | off (on 0) | same | (reads) | same | +0.99 |
| L30 down c475 | 1% / 0 | **res%100** (R2 0.68): res mod 100 in {58, 68} | off (on 0) | - | same | +0.99 |
| L30 up c407 | 7% / 4% | **res%100** (R2 0.92): res mod 100 in {25..31} | **res** (R2 0.68): res in {25..30} | (reads) | same | +0.99 |
| L29 gate c150 | 8% / 2% | **res%100** (R2 0.89): res mod 100 in {79..86} | unexplained (best R2 0.48) | (reads) | same | +0.99 |
| L31 down c211 | 1% / 0 | **res%100** (R2 0.80): res mod 100 in {34} | off (on 0) | - | same | +0.99 |
| L30 gate c706 | 7% / 6% | **res%100** (R2 0.84): res mod 100 in {23..29} | **res** (R2 0.71): res in {23..30} | (reads) | same | +0.99 |
| L29 gate c591 | 0 / 1% | off (on 0) | **res%100** (R2 0.73): res mod 100: no class above 0.5 (max 0.45) | (reads) | same | +0.99 |
| L26 up c4 | 12% / 9% | **res%100** (R2 0.90): res mod 100 in {20..30} | **res//10** (R2 0.56): (tens) res in {19..30} | (reads) | same | +0.99 |
| L26 up c205 | 2% / 2% | **res%100** (R2 0.79): res mod 100 in {12} | **res** (R2 0.80): res in {11..13} | (reads) | same | +0.99 |
| L30 gate c628 | 2% / 1% | **res%100** (R2 0.95): res mod 100 in {22..23} | **res** (R2 0.72): res in {22..23} | (reads) | same | +0.99 |
| L30 gate c207 | 9% / 5% | **res%100** (R2 0.93): res mod 100 in {25..33} | **res** (R2 0.72): res in {26..33} | (reads) | same | +0.99 |
| L28 gate c112 | 2% / 0 | **res%100** (R2 0.80): res mod 100 in {61, 81} | off (on 0) | (reads) | same | +0.99 |
| L30 up c739 | 4% / 3% | **res%100** (R2 0.81): res mod 100 in {18..20} | **res** (R2 0.66): res in {18..20} | (reads) | same | +0.99 |
| L28 up c203 | 6% / 4% | **res%100** (R2 0.82): res mod 100 in {12..16} | **res** (R2 0.75): res in {12..16} | (reads) | same | +0.99 |
| L28 up c10 | 11% / 9% | **res//10** (R2 0.85): (tens) res in {10..19, 110..119} [coarser: res mod 100 in {10..19}, R2 0.99] | **res//10** (R2 0.74): (tens) res in {10..18} | (reads) | same | +0.99 |
| L27 up c254 | 6% / 3% | **res%100** (R2 0.90): res mod 100 in {38..43} | unexplained (best R2 0.45) | (reads) | same | +0.98 |
| L28 gate c316 | 2% / 0 | **res%100** (R2 0.81): res mod 100 in {54..55} | off (on 0) | (reads) | same | +0.98 |
| L25 up c110 | 9% / 7% | **res%100** (R2 0.84): res mod 100 in {13..19} | **res** (R2 0.79): res in {12..20} | (reads) | same | +0.98 |
| L28 gate c593 | 7% / 3% | **res%100** (R2 0.96): res mod 100 in {38..44} | unexplained (best R2 0.46) | (reads) | same | +0.98 |
| L29 up c553 | 0 / 1% | off (on 0) | **res** (R2 0.61): res in {13} | (reads) | same | +0.98 |
| L21 down c1 | 50% / 57% | **res%2** (R2 0.99): res mod 2 in {1} | **res%2** (R2 0.65): res mod 2 in {1} | res: mod2 +33% | a: mod2 +9%; res: mod2 +34% | +0.98 |
| L28 down c182 | 5% / 2% | **res%100** (R2 0.69): res mod 100 in {76..79} | unexplained (best R2 0.30) | - | same | +0.98 |
| L22 down c101 | 8% / 4% | **res%100** (R2 0.74): res mod 100 in {20..21, 23..27} | **res** (R2 0.53): res in {21, 23..26} | - | same | +0.98 |
| L26 gate c205 | 4% / 4% | **res%100** (R2 0.82): res mod 100 in {11..13, 15} | **res** (R2 0.78): res in {11..15} | (reads) | same | +0.98 |
| L28 up c312 | 4% / 0 | **res%100** (R2 0.77): res mod 100 in {70..74} | off (on 0) | (reads) | same | +0.98 |
| L23 up c387 | 4% / 4% | **units(a,b)** (R2 0.98): a%10 in {0,5} -> b%10 in {0,5} | **units(a,b)** (R2 0.94): a%10 in {0,5} -> b%10 in {0,5} | (reads) | same | +0.98 |
| L25 down c66 | 13% / 5% | **res%100** (R2 0.89): res mod 100 in {41..47, 81..86} | unexplained (best R2 0.45) | - | same | +0.98 |
| L23 down c202 | 9% / 9% | **res%100** (R2 0.86): res mod 100 in {13..20} | **res** (R2 0.63): res in {13..20} | - | same | +0.98 |
| L30 down c452 | 1% / 0 | **res%100** (R2 0.84): res mod 100 in {71} | off (on 0) | - | same | +0.98 |
| L24 gate c99 | 7% / 2% | **res%100** (R2 0.81): res mod 100 in {26..27, 29..32, 34} | unexplained (best R2 0.44) | (reads) | same | +0.98 |
| L28 gate c125 | 2% / 0 | **res%100** (R2 0.86): res mod 100 in {54..55} | off (on 0) | (reads) | same | +0.98 |
| L28 gate c760 | 0 / 1% | off (on 0) | **res%100** (R2 0.68): res mod 100: no class above 0.5 (max 0.42) | (reads) | same | +0.98 |
| L23 up c638 | 3% / 1% | **res%100** (R2 0.82): res mod 100 in {69..71} | unexplained (best R2 0.20) | (reads) | same | +0.98 |
| L30 up c209 | 11% / 7% | **res%10** (R2 0.92): res mod 10 in {4} | **res** (R2 0.60): res in {-16, -6, 14, 24, 34, 44, 54, 64, 74, 84, 94} [coarser: res mod 100 in {24, 34, 44, 54, 64, 74, 84, 94}, R2 0.80] | (reads) | same | +0.98 |
| L30 gate c209 | 10% / 6% | **res%10** (R2 0.97): res mod 10 in {4} | **res** (R2 0.64): res in {14, 24, 34, 44, 54, 64, 74, 84, 94} | (reads) | same | +0.98 |
| L24 gate c8 | 16% / 4% | **res%100** (R2 0.76): res mod 100 in {56..70} | unexplained (best R2 0.37) | (reads) | same | +0.98 |
| L21 up c14 | 10% / 8% | **res%10** (R2 0.96): res mod 10 in {5} | **res%10** (R2 0.65): res mod 10 in {5} | (reads) | same | +0.98 |
| L29 down c300 | 1% / 0 | **res** (R2 0.73): res in {59, 149, 159} [coarser: res mod 100 in {59}, R2 0.86] | off (on 0) | - | same | +0.98 |
| L25 gate c748 | 9% / 9% | **res%100** (R2 0.87): res mod 100 in {8, 10..14, 16, 18} | **res** (R2 0.78): res in {8..14, 16..18} | (reads) | same | +0.98 |
| L29 up c130 | 11% / 6% | **res%10** (R2 0.89): res mod 10 in {4} | **res** (R2 0.69): res in {4, 14, 24, 34, 44, 54, 64, 74, 84, 94} | (reads) | same | +0.98 |
| L28 up c303 | 1% / 0 | **res** (R2 0.74): res in {143} [coarser: res mod 100 in {43}, R2 0.87] | off (on 0) | (reads) | same | +0.98 |
| L29 up c331 | 4% / 0 | **res%100** (R2 0.82): res mod 100 in {62..65} | off (on 0) | (reads) | same | +0.98 |
| L27 gate c200 | 4% / 1% | **res%100** (R2 0.68): res mod 100 in {48..51} | unexplained (best R2 0.23) | (reads) | same | +0.98 |
| L29 gate c592 | 0 / 2% | off (on 0) | **res%100** (R2 0.71): res mod 100 in {7} | (reads) | same | +0.98 |
| L29 gate c132 | 11% / 7% | **res%10** (R2 0.92): res mod 10 in {4} | **res** (R2 0.66): res in {-16, -6, 4, 14, 24, 34, 44, 54, 64, 74, 84, 94} | (reads) | same | +0.98 |
| L29 up c419 | 0 / 1% | off (on 0) | **res%100** (R2 0.76): res mod 100: no class above 0.5 (max 0.42) | (reads) | same | +0.98 |
| L29 up c264 | 0 / 1% | off (on 0) | **res%100** (R2 0.68): res mod 100: no class above 0.5 (max 0.44) | (reads) | same | +0.98 |
| L26 gate c33 | 12% / 11% | **res%100** (R2 0.87): res mod 100 in {10..21} | **res** (R2 0.72): res in {10..22} | (reads) | same | +0.98 |
| L29 up c172 | 11% / 7% | **res%10** (R2 0.88): res mod 10 in {5} | **res%100** (R2 0.62): res mod 100 in {5, 15, 25, 35, 45, 55, 65, 75, 85, 95} [coarser: res mod 10 in {5}, R2 0.80] | (reads) | same | +0.98 |
| L22 gate c257 | 10% / 3% | **res%100** (R2 0.84): res mod 100 in {41..49} | unexplained (best R2 0.43) | (reads) | same | +0.98 |
| L26 down c813 | 4% / 3% | **res%100** (R2 0.77): res mod 100 in {14..17} | **res** (R2 0.74): res in {14..17} | - | same | +0.98 |
| L29 gate c835 | 7% / 4% | **res%100** (R2 0.89): res mod 100 in {29..35} | **res** (R2 0.63): res in {30..35} | (reads) | same | +0.98 |
| L30 gate c735 | 1% / 0 | **res%100** (R2 0.71): res mod 100 in {70} | off (on 0) | (reads) | same | +0.98 |
| L30 up c882 | 5% / 5% | **res%100** (R2 0.80): res mod 100 in {25..29} | **res** (R2 0.66): res in {25..30} | (reads) | same | +0.98 |
| L25 down c248 | 6% / 1% | **res%100** (R2 0.86): res mod 100 in {75..79} | unexplained (best R2 0.27) | - | same | +0.98 |
| L26 gate c219 | 4% / 1% | **res%100** (R2 0.95): res mod 100 in {60..63} | unexplained (best R2 0.14) | (reads) | same | +0.98 |
| L26 gate c226 | 12% / 3% | **res%100** (R2 0.79): res mod 100 in {40..52} | unexplained (best R2 0.33) | (reads) | same | +0.98 |
| L24 gate c419 | 5% / 0 | **res%100** (R2 0.93): res mod 100 in {53, 63, 73, 83, 93} | off (on 0) | (reads) | same | +0.98 |
| L29 gate c370 | 2% / 0 | **res%100** (R2 0.82): res mod 100 in {94..95} | off (on 0) | (reads) | same | +0.98 |
| L27 down c168 | 7% / 4% | **res%100** (R2 0.76): res mod 100 in {40..47} | unexplained (best R2 0.47) | - | same | +0.98 |
| L24 gate c361 | 6% / 1% | **res%100** (R2 0.84): res mod 100 in {55..61} | unexplained (best R2 0.18) | (reads) | same | +0.98 |
| L20 down c31 | 29% / 25% | **res//10** (R2 0.79): (tens) res in {2..13, 89..117, 188..200} [coarser: res mod 100 in {0..15, 89..99}, R2 0.98] | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {0,1,9,10}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {4,5}; a//10 in {6} -> b//10 in {5,6,7,8}; a//10 in {7} -> b//10 in {6,7,8,9}; a//10 in {8} -> b//10 in {7,8,9,10}; a//10 in {9,10} -> b//10 in {0,8,9,10} | res: mod100 +4%, mod25 +6% | res: mod100 +3%, mod50 +3%, mod25 +4% | +0.98 |
| L25 gate c243 | 5% / 0 | **res%100** (R2 0.88): res mod 100 in {90..93} | off (on 0) | (reads) | same | +0.98 |
| L30 up c364 | 8% / 1% | **res%100** (R2 0.85): res mod 100 in {80..87} | unexplained (best R2 0.46) | (reads) | same | +0.98 |
| L26 down c176 | 7% / 1% | **res%100** (R2 0.60): res mod 100 in {84..88} | unexplained (best R2 0.38) | - | same | +0.98 |
| L26 down c226 | 4% / 1% | **res%100** (R2 0.80): res mod 100 in {45..49} | unexplained (best R2 0.22) | - | same | +0.98 |
| L27 down c254 | 5% / 3% | **res** (R2 0.76): res in {38..43, 139..141} [coarser: res mod 100 in {38..43}, R2 0.85] | unexplained (best R2 0.44) | - | same | +0.98 |
| L26 gate c513 | 6% / 1% | **res%100** (R2 0.69): res mod 100 in {85..89} | unexplained (best R2 0.49) | (reads) | same | +0.98 |
| L29 gate c437 | 5% / 1% | **res** (R2 0.76): res in {57..62, 159, 162} [coarser: res mod 100 in {57..62}, R2 0.88] | unexplained (best R2 0.22) | (reads) | same | +0.98 |
| L25 gate c66 | 14% / 5% | **res%100** (R2 0.84): res mod 100 in {41..46, 81..86} | unexplained (best R2 0.44) | (reads) | same | +0.98 |
| L24 down c101 | 10% / 11% | **res%100** (R2 0.91): res mod 100 in {15..24} | **res** (R2 0.63): res in {14..24} | - | same | +0.98 |
| L24 down c41 | 15% / 4% | **res%100** (R2 0.90): res mod 100 in {82..96} | unexplained (best R2 0.43) | res: mod25 +3% | - | +0.97 |
| L26 gate c113 | 6% / 2% | **res%100** (R2 0.84): res mod 100 in {44..49} | unexplained (best R2 0.29) | (reads) | same | +0.97 |
| L31 gate c182 | 7% / 1% | **res//10** (R2 0.69): (tens) res in {81..89} | unexplained (best R2 0.37) | (reads) | same | +0.97 |
| L29 up c549 | 11% / 8% | **res//10** (R2 0.89): (tens) res in {3, 20..29, 120..129} [coarser: res mod 100 in {20..29}, R2 0.96] | **res//10** (R2 0.75): (tens) res in {20..29} | (reads) | same | +0.97 |
| L30 gate c309 | 3% / 3% | **res%100** (R2 0.83): res mod 100 in {4..5} | **res%100** (R2 0.57): res mod 100: no class above 0.5 (max 0.43) | (reads) | same | +0.97 |
| L29 gate c145 | 10% / 7% | **res%10** (R2 0.98): res mod 10 in {6} | **res** (R2 0.63): res in {-4, 6, 16, 26, 36, 46, 56, 66, 76, 86, 96} | (reads) | same | +0.97 |
| L26 up c182 | 15% / 15% | **res%100** (R2 0.91): res mod 100 in {30..44} | **res** (R2 0.52): res in {30..44} | (reads) | same | +0.97 |
| L23 down c14 | 11% / 8% | **res%10** (R2 0.93): res mod 10 in {4} | **res%100** (R2 0.58): res mod 100 in {4, 14, 24, 34, 44, 54, 64, 74, 84, 94} [coarser: res mod 50 in {4, 14, 24, 34, 44}, R2 0.80] | res: mod10 +4%, mod5 +5%, mod2 +2% | res: mod10 +3%, mod5 +3% | +0.97 |
| L24 down c46 | 10% / 9% | **res%10** (R2 0.98): res mod 10 in {0} | **res%20** (R2 0.47): res mod 20 in {0, 10} [coarser: res mod 10 in {0}, R2 0.83] | - | same | +0.97 |
| L31 down c902 | 4% / 1% | **res** (R2 0.79): res in {60..65} | unexplained (best R2 0.30) | - | same | +0.97 |
| L23 down c387 | 4% / 4% | **units(a,b)** (R2 0.98): a%10 in {0,5} -> b%10 in {0,5} | **units(a,b)** (R2 0.88): a%10 in {0,5} -> b%10 in {0,5} | - | same | +0.97 |
| L30 gate c261 | 1% / 0 | **res%100** (R2 0.71): res mod 100 in {63} | off (on 0) | (reads) | same | +0.97 |
| L28 up c522 | 10% / 7% | **res%10** (R2 0.97): res mod 10 in {7} | **res** (R2 0.69): res in {-13, -3, 7, 17, 27, 37, 47, 57, 67, 77, 87, 97} | (reads) | same | +0.97 |
| L27 down c107 | 11% / 2% | **res%100** (R2 0.84): res mod 100 in {51..59, 62..63} | unexplained (best R2 0.36) | - | same | +0.97 |
| L29 gate c467 | 11% / 7% | **res%10** (R2 0.91): res mod 10 in {5} | **res%100** (R2 0.62): res mod 100 in {5, 15, 25, 35, 55, 65, 75, 85, 95} [coarser: res mod 50 in {5, 15, 25, 35, 45}, R2 0.80] | (reads) | same | +0.97 |
| L24 gate c50 | 12% / 8% | **res%100** (R2 0.83): res mod 100 in {21..32} | **res** (R2 0.60): res in {21..29} | (reads) | same | +0.97 |
| L24 down c17 | 12% / 9% | **res%100** (R2 0.86): res mod 100 in {46..56} | unexplained (best R2 0.35) | res: mod25 +3% | - | +0.97 |
| L28 gate c268 | 11% / 2% | **res//10** (R2 0.80): (tens) res in {50..60, 151..159} [coarser: res mod 100 in {50..60}, R2 0.96] | unexplained (best R2 0.33) | (reads) | same | +0.97 |
| L27 gate c269 | 10% / 5% | **res%10** (R2 0.95): res mod 10 in {1} | **res** (R2 0.64): res in {1, 11, 21, 31, 41, 71, 81, 91} | (reads) | same | +0.97 |
| L30 gate c851 | 2% / 0 | **res%100** (R2 0.85): res mod 100 in {63} | off (on 0) | (reads) | same | +0.97 |
| L29 down c467 | 1% / 1% | **res%100** (R2 0.74): res mod 100 in {15} | **res** (R2 0.57): res in {15} | - | same | +0.97 |
| L26 down c94 | 6% / 1% | **res%100** (R2 0.81): res mod 100 in {53..54, 56..59} | unexplained (best R2 0.19) | - | same | +0.97 |
| L24 down c12 | 10% / 11% | **res%10** (R2 0.96): res mod 10 in {5} | **units(a,b)** (R2 0.64): a%10 in {0} -> b%10 in {5}; a%10 in {1} -> b%10 in {6}; a%10 in {2} -> b%10 in {7}; a%10 in {3} -> b%10 in {8}; a%10 in {4} -> b%10 in {9}; a%10 in {5} -> b%10 in {0,5}; a%10 in {6} -> b%10 in {1}; a%10 in {7} -> b%10 in {2}; a%10 in {8} -> b%10 in {3}; a%10 in {9} -> b%10 in {4} | res: mod10 +3%, mod5 +4% | res: mod10 +4%, mod5 +5% | +0.97 |
| L30 down c486 | 3% / 1% | **res%100** (R2 0.87): res mod 100 in {79..81} | unexplained (best R2 0.21) | - | same | +0.97 |
| L23 up c14 | 20% / 12% | **res%10** (R2 0.98): res mod 10 in {2, 4} | **res** (R2 0.63): res in {2, 4, 12, 14, 22, 24, 32, 34, 42, 44, 52, 54, 62, 64, 72, 74, 82, 84, 92, 94} | (reads) | same | +0.97 |
| L27 gate c586 | 7% / 7% | **res%100** (R2 0.87): res mod 100 in {13..20} | **res** (R2 0.75): res in {13..21} | (reads) | same | +0.97 |
| L22 down c27 | 12% / 10% | **res%100** (R2 0.81): res mod 100 in {27..38} | **res** (R2 0.57): res in {27..38} | res: mod25 +3% | - | +0.97 |
| L28 down c129 | 9% / 5% | **res%10** (R2 0.87): res mod 10 in {9} | **res** (R2 0.57): res in {9, 19, 39, 49, 59, 69, 79, 89, 99} | - | same | +0.97 |
| L29 down c132 | 11% / 7% | **res%10** (R2 0.90): res mod 10 in {4} | **res** (R2 0.53): res in {-26, -16, -6, 14, 24, 34, 44, 54, 64, 74, 84, 94} [coarser: res mod 100 in {14, 24, 34, 54, 64, 74, 84, 94}, R2 0.82] | res: mod10 +2%, mod5 +3%, mod2 +2% | - | +0.97 |
| L29 up c189 | 4% / 4% | **res** (R2 0.75): res in {27..31, 127..129} [coarser: res mod 100 in {27..31}, R2 0.89] | **res** (R2 0.53): res in {27..31} | (reads) | same | +0.97 |
| L29 down c122 | 1% / 2% | **res%100** (R2 0.85): res mod 100 in {6} | **res%100** (R2 0.54): res mod 100 in {6} | - | same | +0.97 |
| L22 down c8 | 10% / 9% | **res%10** (R2 0.98): res mod 10 in {0} | **res%10** (R2 0.52): res mod 10 in {0} | res: mod10 +5%, mod5 +8% | res: mod10 +3%, mod5 +6% | +0.97 |
| L28 down c23 | 41% / 19% | **res%2** (R2 0.70): res mod 2 in {0} | **units(a,b)** (R2 0.65): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} | a: mod2 +15%; b: mod2 +11%; res: mod2 +11% | a: mod2 +7%; b: mod2 +7%; res: mod2 +4% | +0.97 |
| L28 gate c182 | 6% / 1% | **res%100** (R2 0.83): res mod 100 in {75..79} | unexplained (best R2 0.28) | (reads) | same | +0.97 |
| L23 gate c14 | 11% / 8% | **res%10** (R2 0.93): res mod 10 in {4} | **res%100** (R2 0.57): res mod 100 in {4, 14, 24, 34, 44, 64, 74, 84, 94} | (reads) | same | +0.97 |
| L27 up c142 | 3% / 0 | **res%100** (R2 0.89): res mod 100 in {49, 69} | off (on 0) | (reads) | same | +0.97 |
| L24 down c152 | 4% / 0 | **res%100** (R2 0.72): res mod 100 in {75..78} | off (on 0) | - | same | +0.97 |
| L29 gate c480 | 10% / 8% | **res%10** (R2 0.97): res mod 10 in {1} | **res** (R2 0.52): res in {-99, -29, -19, -9, 11, 21, 31, 41, 51, 61, 71, 81, 91} | (reads) | same | +0.97 |
| L31 gate c143 | 9% / 1% | **res%100** (R2 0.67): res mod 100 in {59, 61..69} | unexplained (best R2 0.32) | (reads) | same | +0.97 |
| L26 down c237 | 7% / 3% | **res%100** (R2 0.84): res mod 100 in {49..55} | unexplained (best R2 0.27) | - | same | +0.97 |
| L24 gate c13 | 17% / 7% | **res%100** (R2 0.94): res mod 100 in {49..64} | unexplained (best R2 0.37) | (reads) | same | +0.97 |
| L26 down c703 | 4% / 1% | **res%100** (R2 0.78): res mod 100 in {65, 67..69} | unexplained (best R2 0.21) | - | same | +0.97 |
| L25 down c110 | 3% / 3% | **res%100** (R2 0.75): res mod 100 in {16..18} | **res** (R2 0.70): res in {15..18} | - | same | +0.97 |
| L22 up c8 | 10% / 10% | **res%10** (R2 0.98): res mod 10 in {0} | **res%10** (R2 0.51): res mod 10 in {0} | (reads) | same | +0.97 |
| L28 gate c115 | 16% / 6% | **res%100** (R2 0.91): res mod 100 in {18, 74..88} | **res** (R2 0.55): res in {18, 74..88} | (reads) | same | +0.97 |
| L25 gate c15 | 18% / 17% | **res%100** (R2 0.91): res mod 100 in {0..14, 98..99} | **res%100** (R2 0.72): res mod 100 in {0..2, 4, 6..14, 99} | (reads) | same | +0.97 |
| L27 up c107 | 12% / 3% | **res//10** (R2 0.81): (tens) res in {60..70, 160..170} [coarser: res mod 100 in {60..70}, R2 0.99] | unexplained (best R2 0.34) | (reads) | same | +0.97 |
| L29 gate c279 | 4% / 5% | **res%100** (R2 0.79): res mod 100 in {19..23} | **res** (R2 0.63): res in {19..23} | (reads) | same | +0.97 |
| L28 gate c69 | 10% / 8% | **res%10** (R2 0.96): res mod 10 in {6} | **units(a,b)** (R2 0.56): a%10 in {0} -> b%10 in {4}; a%10 in {1} -> b%10 in {5}; a%10 in {2} -> b%10 in {6}; a%10 in {4} -> b%10 in {8}; a%10 in {5} -> b%10 in {9}; a%10 in {6} -> b%10 in {0}; a%10 in {7} -> b%10 in {1}; a%10 in {8} -> b%10 in {2}; a%10 in {9} -> b%10 in {3} | (reads) | same | +0.97 |
| L30 gate c364 | 8% / 2% | **res%100** (R2 0.93): res mod 100 in {80..87} | unexplained (best R2 0.35) | (reads) | same | +0.97 |
| L27 up c23 | 21% / 14% | **res//10** (R2 0.80): (tens) res in {10..22, 24, 110..134} [coarser: res mod 100 in {10..26, 28, 30..31}, R2 0.90] | **res//10** (R2 0.67): (tens) res in {10..25} | (reads) | same | +0.97 |
| L30 gate c642 | 4% / 3% | **res%100** (R2 0.82): res mod 100 in {17..19} | **res** (R2 0.74): res in {17..19} | (reads) | same | +0.97 |
| L28 gate c23 | 42% / 21% | **res%10** (R2 0.74): res mod 10 in {0, 2, 4, 6, 8} | **units(a,b)** (R2 0.64): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} | (reads) | same | +0.97 |
| L29 gate c293 | 12% / 13% | **units(a,b)** (R2 0.95): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {9}; a%10 in {2} -> b%10 in {8}; a%10 in {3} -> b%10 in {7}; a%10 in {4} -> b%10 in {6}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {3}; a%10 in {8} -> b%10 in {2}; a%10 in {9} -> b%10 in {1} | **units(a,b)** (R2 0.55): a%10 in {0} -> b%10 in {0,5,9}; a%10 in {1} -> b%10 in {1}; a%10 in {2} -> b%10 in {2}; a%10 in {3} -> b%10 in {3}; a%10 in {4} -> b%10 in {4}; a%10 in {5} -> b%10 in {0,5}; a%10 in {6} -> b%10 in {6}; a%10 in {8} -> b%10 in {8}; a%10 in {9} -> b%10 in {9} | (reads) | same | +0.97 |
| L30 gate c187 | 10% / 6% | **res%10** (R2 0.94): res mod 10 in {7} | **res** (R2 0.62): res in {-3, 17, 27, 37, 47, 57, 67, 77, 87, 97} | (reads) | same | +0.97 |
| L28 gate c10 | 11% / 8% | **res//10** (R2 0.91): (tens) res in {10..19, 110..119} [coarser: res mod 100 in {10..19}, R2 1.00] | **res//10** (R2 0.76): (tens) res in {11..19} | (reads) | same | +0.97 |
| L23 gate c202 | 11% / 13% | **res%100** (R2 0.84): res mod 100 in {11..21} | **res** (R2 0.71): res in {9..21} | (reads) | same | +0.97 |
| L29 down c293 | 12% / 11% | **units(a,b)** (R2 0.95): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {9}; a%10 in {2} -> b%10 in {8}; a%10 in {3} -> b%10 in {7}; a%10 in {4} -> b%10 in {6}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {3}; a%10 in {8} -> b%10 in {2}; a%10 in {9} -> b%10 in {1} | **units(a,b)** (R2 0.51): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {1}; a%10 in {2} -> b%10 in {2}; a%10 in {4} -> b%10 in {4}; a%10 in {8} -> b%10 in {8}; a%10 in {9} -> b%10 in {9} | - | same | +0.97 |
| L29 gate c140 | 11% / 2% | **res//10** (R2 0.86): (tens) res in {80..90, 180..190} [coarser: res mod 100 in {80..90}, R2 1.00] | **res//10** (R2 0.46): (tens) res in {80..90} | (reads) | same | +0.97 |
| L27 up c85 | 24% / 5% | **res//10** (R2 0.79): (tens) res in {70..95, 171..185} [coarser: res mod 100 in {70..91, 94}, R2 0.88] | **res//10** (R2 0.66): (tens) res in {70..91, 93..94} | (reads) | same | +0.97 |
| L24 gate c83 | 10% / 9% | **res%10** (R2 0.96): res mod 10 in {5} | **res%10** (R2 0.57): res mod 10 in {5} | (reads) | same | +0.97 |
| L30 gate c510 | 2% / 0 | **res%100** (R2 0.73): res mod 100 in {80..81} | off (on 0) | (reads) | same | +0.97 |
| L28 up c268 | 12% / 3% | **res%100** (R2 0.83): res mod 100 in {50..60} | unexplained (best R2 0.38) | (reads) | same | +0.97 |
| L27 up c117 | 6% / 2% | **res%100** (R2 0.95): res mod 100 in {79..84} | unexplained (best R2 0.25) | (reads) | same | +0.97 |
| L22 gate c5 | 18% / 12% | **res%100** (R2 0.77): res mod 100 in {39..54} | unexplained (best R2 0.46) | (reads) | same | +0.97 |
| L30 gate c181 | 11% / 6% | **res%100** (R2 0.81): res mod 100 in {15..23} | **res** (R2 0.70): res in {16..22} | (reads) | same | +0.97 |
| L30 down c519 | 1% / 1% | **res%100** (R2 0.76): res mod 100 in {14} | **res** (R2 0.62): res in {14} | - | same | +0.97 |
| L25 down c46 | 5% / 4% | **res%100** (R2 0.70): res mod 100 in {17..20, 79} | **res** (R2 0.57): res in {17..20} | - | same | +0.97 |
| L22 down c257 | 2% / 1% | **res%100** (R2 0.63): res mod 100 in {45..46} | unexplained (best R2 0.27) | - | same | +0.97 |
| L22 down c28 | 14% / 4% | **res%100** (R2 0.87): res mod 100 in {77..92} | **res** (R2 0.55): res in {77..90} | res: mod25 +6% | - | +0.97 |
| L28 down c545 | 1% / 0 | **res%100** (R2 0.53): res mod 100: no class above 0.5 (max 0.40) | off (on 0) | - | same | +0.97 |
| L26 up c360 | 7% / 2% | **res%100** (R2 0.81): res mod 100 in {52..59} | unexplained (best R2 0.32) | (reads) | same | +0.96 |
| L30 down c551 | 2% / 2% | **res** (R2 0.76): res in {23..25, 124} [coarser: res mod 100 in {23..24}, R2 0.82] | **res** (R2 0.59): res in {23..24} | - | same | +0.96 |
| L24 gate c17 | 18% / 10% | **res%100** (R2 0.86): res mod 100 in {46..62} | unexplained (best R2 0.42) | (reads) | same | +0.96 |
| L28 down c268 | 8% / 2% | **res** (R2 0.80): res in {49..60, 155..156} | unexplained (best R2 0.40) | - | same | +0.96 |
| L29 up c318 | 0 / 2% | off (on 0) | **res** (R2 0.80): res in {9, 11, 13} | (reads) | same | +0.96 |
| L30 up c603 | 4% / 2% | **res%100** (R2 0.78): res mod 100 in {68..71} | unexplained (best R2 0.13) | (reads) | same | +0.96 |
| L24 down c99 | 11% / 6% | **res%100** (R2 0.83): res mod 100 in {26..36} | **res** (R2 0.63): res in {26..34} | - | same | +0.96 |
| L31 down c623 | 4% / 1% | **res%100** (R2 0.86): res mod 100 in {77..79, 81} | unexplained (best R2 0.34) | - | same | +0.96 |
| L26 down c299 | 8% / 1% | **res%100** (R2 0.82): res mod 100 in {72..78} | unexplained (best R2 0.22) | - | same | +0.96 |
| L31 down c73 | 1% / 0 | **res%100** (R2 0.57): res mod 100 in {86} | off (on 0) | - | same | +0.96 |
| L24 up c13 | 19% / 7% | **res%100** (R2 0.85): res mod 100 in {52..68} | unexplained (best R2 0.38) | (reads) | same | +0.96 |
| L29 gate c153 | 6% / 2% | **res%100** (R2 0.76): res mod 100 in {56..60} | unexplained (best R2 0.20) | (reads) | same | +0.96 |
| L27 down c193 | 2% / 9% | **res** (R2 0.62): res in {12..20} | **res//10** (R2 0.66): (tens) res in {10..19} | - | same | +0.96 |
| L28 down c10 | 10% / 7% | **res//10** (R2 0.86): (tens) res in {11..17, 110..119} [coarser: res mod 100 in {10..19}, R2 0.95] | **res** (R2 0.78): res in {11..17} | - | same | +0.96 |
| L24 gate c501 | 3% / 3% | **res%100** (R2 0.75): res mod 100: no class above 0.5 (max 0.44) | **res%100** (R2 0.72): res mod 100: no class above 0.5 (max 0.45) | (reads) | same | +0.96 |
| L28 gate c434 | 9% / 3% | **res%100** (R2 0.78): res mod 100 in {44..52} | unexplained (best R2 0.29) | (reads) | same | +0.96 |
| L27 gate c835 | 9% / 3% | **res%100** (R2 0.87): res mod 100 in {29..37} | unexplained (best R2 0.40) | (reads) | same | +0.96 |
| L26 up c46 | 2% / 0 | **res%100** (R2 0.80): res mod 100 in {77..78} | off (on 0) | (reads) | same | +0.96 |
| L27 up c251 | 0 / 0 | off (on 0) | same | (reads) | same | +0.96 |
| L28 gate c93 | 11% / 5% | **units(a,b)** (R2 0.91): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {4}; a%10 in {2} -> b%10 in {3}; a%10 in {3} -> b%10 in {2}; a%10 in {4} -> b%10 in {1}; a%10 in {6} -> b%10 in {9}; a%10 in {7} -> b%10 in {8}; a%10 in {8} -> b%10 in {7}; a%10 in {9} -> b%10 in {6} | **units(a,b)** (R2 0.56): a%10 in {0} -> b%10 in {5}; a%10 in {5} -> b%10 in {0,5} | (reads) | same | +0.96 |
| L31 up c535 | 1% / 0 | **res%100** (R2 0.66): res mod 100 in {68} | off (on 0) | (reads) | same | +0.96 |
| L28 gate c297 | 6% / 1% | **res%100** (R2 0.91): res mod 100 in {69..74} | unexplained (best R2 0.17) | (reads) | same | +0.96 |
| L29 down c183 | 2% / 0 | **res%100** (R2 0.89): res mod 100 in {34, 94} | off (on 0) | - | same | +0.96 |
| L29 gate c213 | 3% / 0 | **res%100** (R2 0.60): res mod 100 in {94} | off (on 0) | (reads) | same | +0.96 |
| L28 up c93 | 13% / 9% | **units(a,b)** (R2 0.91): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {4}; a%10 in {2} -> b%10 in {3}; a%10 in {3} -> b%10 in {2}; a%10 in {4} -> b%10 in {1}; a%10 in {6} -> b%10 in {9}; a%10 in {7} -> b%10 in {8}; a%10 in {8} -> b%10 in {7}; a%10 in {9} -> b%10 in {6} | **units(a,b)** (R2 0.65): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {6}; a%10 in {4} -> b%10 in {9}; a%10 in {6} -> b%10 in {1}; a%10 in {8} -> b%10 in {3}; a%10 in {9} -> b%10 in {4} | (reads) | same | +0.96 |
| L24 up c46 | 10% / 9% | **res%10** (R2 0.98): res mod 10 in {0} | **res%100** (R2 0.54): res mod 100 in {0, 10, 20, 30, 50, 70, 80, 90} [coarser: res mod 10 in {0}, R2 0.82] | (reads) | same | +0.96 |
| L23 up c225 | 10% / 9% | **res%10** (R2 0.98): res mod 10 in {6} | **res%100** (R2 0.49): res mod 100 in {6, 16, 26, 36, 46, 56, 66, 76, 86, 96} [coarser: res mod 10 in {6}, R2 0.80] | (reads) | same | +0.96 |
| L25 down c42 | 9% / 8% | **res%100** (R2 0.86): res mod 100 in {7..8, 10..14} | **res%100** (R2 0.73): res mod 100 in {10..12} | - | res: mod25 +4% | +0.96 |
| L28 gate c160 | 10% / 7% | **res%10** (R2 0.94): res mod 10 in {7} | **res** (R2 0.67): res in {-97, -13, -3, 7, 17, 27, 37, 47, 57, 67, 77, 87, 97} | (reads) | same | +0.96 |
| L29 up c437 | 4% / 1% | **res** (R2 0.78): res in {59..64} | unexplained (best R2 0.18) | (reads) | same | +0.96 |
| L26 gate c237 | 10% / 4% | **res%100** (R2 0.89): res mod 100 in {45..54} | unexplained (best R2 0.33) | (reads) | same | +0.96 |
| L31 down c247 | 3% / 0 | **res%100** (R2 0.71): res mod 100 in {84..85, 89} | off (on 0) | - | same | +0.96 |
| L30 gate c230 | 11% / 9% | **res%10** (R2 0.89): res mod 10 in {8} | **res%10** (R2 0.51): res mod 10 in {8} | (reads) | same | +0.96 |
| L21 gate c390 | 12% / 7% | **res%10** (R2 0.80): res mod 10 in {2} | **res** (R2 0.64): res in {-18, 2, 12, 22, 32, 42, 52, 62, 72, 82, 92} | (reads) | same | +0.96 |
| L29 down c426 | 1% / 0 | **res** (R2 0.80): res in {92, 152, 192} [coarser: res mod 100 in {52, 92}, R2 0.88] | off (on 0) | - | same | +0.96 |
| L29 up c842 | 11% / 7% | **res%10** (R2 0.84): res mod 10 in {1} | **res** (R2 0.54): res in {-99, -29, -19, -9, 11, 21, 31, 41, 51, 61, 81, 91} | (reads) | same | +0.96 |
| L26 up c207 | 5% / 4% | **res%100** (R2 0.74): res mod 100 in {7..9} | **res%100** (R2 0.71): res mod 100 in {8} | (reads) | same | +0.96 |
| L21 down c42 | 10% / 4% | **res%10** (R2 0.91): res mod 10 in {0} | **res** (R2 0.56): res in {10, 20, 30, 40, 50, 70, 80} | - | same | +0.96 |
| L28 down c300 | 10% / 4% | **res%10** (R2 0.97): res mod 10 in {4} | **res** (R2 0.57): res in {14, 24, 34, 84, 94} | - | same | +0.96 |
| L29 up c280 | 4% / 3% | **res%100** (R2 0.83): res mod 100 in {15..17} | **res** (R2 0.67): res in {15..17} | (reads) | same | +0.96 |
| L26 gate c207 | 5% / 5% | **res%100** (R2 0.77): res mod 100 in {20..23} | **res** (R2 0.60): res in {20..23} | (reads) | same | +0.96 |
| L30 gate c199 | 12% / 5% | **res//10** (R2 0.82): (tens) res in {39..49, 139..149} [coarser: res mod 100 in {39..49}, R2 1.00] | **res** (R2 0.50): res in {40..47} | (reads) | same | +0.96 |
| L29 down c187 | 6% / 7% | **res** (R2 0.65): res in {30..37, 39, 53, 132..135} [coarser: res mod 100 in {31..35}, R2 0.85] | unexplained (best R2 0.30) | - | same | +0.96 |
| L22 gate c6 | 17% / 20% | **res%100** (R2 0.87): res mod 100 in {0..15} | **res//10** (R2 0.63): (tens) res in {0..16} | (reads) | same | +0.96 |
| L29 gate c612 | 0 / 2% | off (on 0) | **res** (R2 0.84): res in {12..13} | (reads) | same | +0.96 |
| L25 gate c220 | 4% / 5% | **res%100** (R2 0.83): res mod 100 in {4} | **res%100** (R2 0.63): res mod 100: no class above 0.5 (max 0.45) | (reads) | same | +0.96 |
| L26 gate c115 | 6% / 4% | **res** (R2 0.84): res in {25..35, 130..133} [coarser: res mod 100 in {29..33}, R2 0.80] | **res** (R2 0.51): res in {27..32} | (reads) | same | +0.96 |
| L25 up c489 | 1% / 1% | **res%100** (R2 0.86): res mod 100: no class above 0.5 (max 0.44) | **cmp(a,b)** (R2 0.80): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.85, cmp(a,b)=1: 0.00 | (reads) | same | +0.96 |
| L29 gate c284 | 1% / 0 | **res%100** (R2 0.76): res mod 100: no class above 0.5 (max 0.47) | off (on 0) | (reads) | same | +0.96 |
| L29 down c437 | 4% / 2% | **res** (R2 0.84): res in {59..64, 160} | unexplained (best R2 0.18) | - | same | +0.96 |
| L24 up c593 | 5% / 1% | **res%100** (R2 0.80): res mod 100 in {62, 82..84} | unexplained (best R2 0.33) | (reads) | same | +0.96 |
| L28 down c93 | 14% / 8% | **res%10** (R2 0.77): res mod 10 in {5} | **units(a,b)** (R2 0.62): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {6}; a%10 in {6} -> b%10 in {1}; a%10 in {9} -> b%10 in {4} | res: mod5 +3% | - | +0.96 |
| L24 down c31 | 19% / 15% | **res//10** (R2 0.82): (tens) res in {8..20, 102..119, 200} [coarser: res mod 100 in {0, 7..19}, R2 0.86] | **res** (R2 0.69): res in {4..19} | - | res: mod25 +2% | +0.96 |
| L24 down c13 | 14% / 10% | **res%100** (R2 0.89): res mod 100 in {52..64} | unexplained (best R2 0.40) | res: mod100 +3%, mod50 +2%, mod25 +4% | - | +0.96 |
| L22 down c35 | 15% / 4% | **res%100** (R2 0.85): res mod 100 in {47..61} | unexplained (best R2 0.39) | res: mod25 +4% | - | +0.96 |
| L30 down c209 | 10% / 4% | **res%10** (R2 0.91): res mod 10 in {4} | **res** (R2 0.59): res in {14, 24, 34, 44, 74, 84, 94} | - | same | +0.96 |
| L24 up c82 | 14% / 9% | **res%100** (R2 0.83): res mod 100 in {21, 31..42} | unexplained (best R2 0.47) | (reads) | same | +0.96 |
| L30 down c187 | 10% / 5% | **res%10** (R2 0.82): res mod 10 in {7} | **res** (R2 0.59): res in {17, 27, 37, 47, 57, 77, 87} | - | same | +0.96 |
| L29 up c343 | 3% / 4% | **res%100** (R2 0.77): res mod 100 in {13, 15..17} | **res** (R2 0.73): res in {13, 15..17} | (reads) | same | +0.96 |
| L24 gate c11 | 17% / 8% | **res%100** (R2 0.88): res mod 100 in {21..27, 61..68} | **res** (R2 0.54): res in {22..27, 62..66} | (reads) | same | +0.96 |
| L25 gate c53 | 10% / 6% | **res%10** (R2 0.99): res mod 10 in {5} | **res%100** (R2 0.61): res mod 100 in {5, 15, 25, 85, 95} | (reads) | same | +0.96 |
| L29 gate c502 | 6% / 3% | **res** (R2 0.80): res in {40..49, 141..143} | unexplained (best R2 0.44) | (reads) | same | +0.96 |
| L25 gate c491 | 1% / 1% | **res%100** (R2 0.82): res mod 100: no class above 0.5 (max 0.49) | **cmp(a,b)** (R2 0.74): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.88, cmp(a,b)=1: 0.00 | (reads) | same | +0.96 |
| L24 up c5 | 10% / 5% | **res%10** (R2 0.98): res mod 10 in {3} | **res** (R2 0.68): res in {3, 13, 23, 33, 43, 73, 83, 93} | (reads) | same | +0.96 |
| L22 gate c28 | 15% / 5% | **res%100** (R2 0.87): res mod 100 in {77..92} | **tens(a,b)** (R2 0.53): a//10 in {8} -> b//10 in {0,10}; a//10 in {9} -> b//10 in {0,1}; a//10 in {10} -> b//10 in {1} | (reads) | same | +0.96 |
| L30 up c199 | 9% / 2% | **res%100** (R2 0.81): res mod 100 in {40..47} | unexplained (best R2 0.38) | (reads) | same | +0.96 |
| L21 up c42 | 31% / 22% | **res%10** (R2 0.97): res mod 10 in {4..6} | **res** (R2 0.69): res in {-25, -15, -6..-4, 4..6, 14..16, 24..26, 34..36, 44..46, 54..56, 64..66, 74..76, 84..86, 94..96} | (reads) | same | +0.96 |
| L20 up c9 | 60% / 52% | **res%5** (R2 0.97): res mod 5 in {0..1, 4} | **res%100** (R2 0.51): res mod 100 in {0..1, 4..6, 9..11, 14..16, 19..21, 24..26, 29..31, 34..36, 39..41, 44..46, 49..51, 54..56, 59..61, 64..66, 69..71, 74..76, 79..81, 84..86, 89..91, 94..96, 99} [coarser: res mod 10 in {0..1, 4..6, 9}, R2 0.80] | (reads) | same | +0.96 |
| L31 gate c69 | 1% / 0 | **res%100** (R2 0.83): res mod 100 in {81} | off (on 0) | (reads) | same | +0.96 |
| L25 down c53 | 10% / 5% | **res%10** (R2 0.96): res mod 10 in {5} | **res%100** (R2 0.52): res mod 100 in {15, 25, 85, 95} | - | same | +0.96 |
| L29 gate c217 | 10% / 4% | **res%10** (R2 0.90): res mod 10 in {9} | **res** (R2 0.62): res in {9, 19, 29, 39, 79, 89, 99} | (reads) | same | +0.96 |
| L24 gate c37 | 8% / 1% | **res%100** (R2 0.87): res mod 100 in {84..91} | unexplained (best R2 0.36) | (reads) | same | +0.96 |
| L29 down c577 | 7% / 3% | **res%50** (R2 0.72): res mod 50 in {19, 29, 39, 49} [coarser: res mod 10 in {9}, R2 0.83] | **res** (R2 0.52): res in {19, 29, 39, 79, 99} | - | same | +0.96 |
| L21 gate c16 | 21% / 16% | **res%10** (R2 0.96): res mod 10 in {5..6} | **units(a,b)** (R2 0.57): a%10 in {0} -> b%10 in {4,5}; a%10 in {1} -> b%10 in {5,6}; a%10 in {2} -> b%10 in {6}; a%10 in {4} -> b%10 in {8,9}; a%10 in {5} -> b%10 in {0,9}; a%10 in {6} -> b%10 in {0,1}; a%10 in {7} -> b%10 in {1}; a%10 in {8} -> b%10 in {2,3}; a%10 in {9} -> b%10 in {3,4} | (reads) | same | +0.96 |
| L31 down c819 | 3% / 1% | **res%50** (R2 0.71): res mod 50 in {36} | unexplained (best R2 0.35) | - | same | +0.96 |
| L30 up c628 | 9% / 3% | **res%10** (R2 0.91): res mod 10 in {3} | unexplained (best R2 0.48) | (reads) | same | +0.95 |
| L22 gate c9 | 10% / 6% | **res%10** (R2 0.96): res mod 10 in {9} | **res** (R2 0.57): res in {9, 19, 29, 39, 49, 59, 69, 79, 89, 99} | (reads) | same | +0.95 |
| L21 down c0 | 33% / 26% | **res//10** (R2 0.84): (tens) res in {8..40, 108..140} [coarser: res mod 100 in {8..40}, R2 1.00] | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {2}; a//10 in {1} -> b//10 in {0,10}; a//10 in {2} -> b//10 in {0,1,9,10}; a//10 in {3} -> b//10 in {0,1,2,10}; a//10 in {4} -> b//10 in {1,2,3}; a//10 in {5} -> b//10 in {2,3,4}; a//10 in {6} -> b//10 in {3,4,5}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {5,6,7}; a//10 in {9,10} -> b//10 in {6,7,8} | res: mod100 +15%, mod50 +5% | a: mod50 +2%; res: mod100 +7%, mod50 +6% | +0.95 |
| L28 gate c67 | 10% / 5% | **res%10** (R2 0.97): res mod 10 in {7} | **res** (R2 0.65): res in {7, 17, 27, 37, 77, 87, 97} | (reads) | same | +0.95 |
| L26 up c113 | 10% / 4% | **res%100** (R2 0.81): res mod 100 in {42..50} | unexplained (best R2 0.33) | (reads) | same | +0.95 |
| L26 down c175 | 10% / 5% | **res%10** (R2 0.97): res mod 10 in {1} | **res** (R2 0.59): res in {11, 21, 31, 41, 51, 61, 71, 91} | - | same | +0.95 |
| L27 down c204 | 9% / 4% | **res%50** (R2 0.81): res mod 50 in {11, 21, 31, 41} [coarser: res mod 10 in {1}, R2 0.84] | **res** (R2 0.57): res in {11, 21, 31, 71, 91} | - | same | +0.95 |
| L22 up c28 | 22% / 10% | **res%100** (R2 0.84): res mod 100 in {77..99} | **tens(a,b)** (R2 0.55): a//10 in {5} -> b//10 in {7}; a//10 in {6} -> b//10 in {7,8}; a//10 in {7} -> b//10 in {8}; a//10 in {8} -> b//10 in {0,9,10}; a//10 in {9} -> b//10 in {0,1,9,10}; a//10 in {10} -> b//10 in {0,1,2,10} | (reads) | same | +0.95 |
| L29 gate c68 | 10% / 11% | **res%100** (R2 0.89): res mod 100 in {1..10} | **res%100** (R2 0.66): res mod 100 in {6..10} | (reads) | same | +0.95 |
| L23 down c125 | 6% / 3% | **res%20** (R2 0.83): res mod 20 in {11} | **res** (R2 0.52): res in {11, 31, 91} | - | same | +0.95 |
| L31 down c328 | 1% / 3% | **res** (R2 0.64): res in {13..15} | **res** (R2 0.70): res in {13..15} | - | same | +0.95 |
| L28 up c182 | 7% / 3% | **res%100** (R2 0.81): res mod 100 in {74..80} | unexplained (best R2 0.35) | (reads) | same | +0.95 |
| L29 down c238 | 3% / 0 | **res%100** (R2 0.86): res mod 100 in {77..79} | off (on 0) | - | same | +0.95 |
| L23 gate c16 | 34% / 29% | **res%100** (R2 0.91): res mod 100 in {58..91} | **tens(a,b)** (R2 0.61): a//10 in {0,5} -> b//10 in {7,8}; a//10 in {3,4} -> b//10 in {6,7}; a//10 in {6} -> b//10 in {0,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,7,10}; a//10 in {10} -> b//10 in {1,2,3,4} | (reads) | same | +0.95 |
| L28 up c452 | 1% / 0 | **res%100** (R2 0.79): res mod 100 in {76} | off (on 0) | (reads) | same | +0.95 |
| L28 up c595 | 12% / 5% | **res%100** (R2 0.91): res mod 100 in {38..48} | **res** (R2 0.51): res in {39..47} | (reads) | same | +0.95 |
| L23 down c55 | 14% / 22% | unexplained (best R2 0.49) | **units(a,b)** (R2 0.81): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} | a: mod2 +3%; b: mod2 +3% | a: mod2 +13%; b: mod2 +13%; res: mod2 +5% | +0.95 |
| L23 down c20 | 18% / 12% | **res//10** (R2 0.80): (tens) res in {25..36, 121..139} [coarser: res mod 100 in {21..37}, R2 0.91] | **res** (R2 0.56): res in {23..36} | res: mod100 +2%, mod50 +2%, mod25 +3% | - | +0.95 |
| L28 down c595 | 5% / 1% | **res%100** (R2 0.73): res mod 100 in {42..47} | unexplained (best R2 0.21) | - | same | +0.95 |
| L29 up c433 | 4% / 1% | **res%100** (R2 0.76): res mod 100 in {52..54, 72} | unexplained (best R2 0.23) | (reads) | same | +0.95 |
| L30 down c364 | 8% / 1% | **res//10** (R2 0.77): (tens) res in {81..89, 181..186} [coarser: res mod 100 in {81..87}, R2 0.95] | unexplained (best R2 0.41) | - | same | +0.95 |
| L27 down c461 | 2% / 0 | **res%100** (R2 0.75): res mod 100 in {56, 58..59} | off (on 0) | - | same | +0.95 |
| L27 gate c117 | 4% / 2% | **res%100** (R2 0.91): res mod 100 in {80..83} | unexplained (best R2 0.39) | (reads) | same | +0.95 |
| L29 up c853 | 2% / 1% | **res%100** (R2 0.78): res mod 100 in {38..39} | unexplained (best R2 0.48) | (reads) | same | +0.95 |
| L23 down c141 | 2% / 1% | **res%100** (R2 0.61): res mod 100 in {28, 88} | unexplained (best R2 0.40) | - | same | +0.95 |
| L27 up c193 | 4% / 0 | **res%100** (R2 0.88): res mod 100 in {80..82} | off (on 0) | (reads) | same | +0.95 |
| L27 down c167 | 2% / 9% | **res** (R2 0.69): res in {19..27} | **res** (R2 0.65): res in {17..27} | - | same | +0.95 |
| L26 down c113 | 8% / 4% | **res%100** (R2 0.81): res mod 100 in {44..50} | unexplained (best R2 0.32) | - | same | +0.95 |
| L22 down c5 | 16% / 13% | **res%100** (R2 0.76): res mod 100 in {39..51} | unexplained (best R2 0.45) | res: mod100 +5%, mod50 +4%, mod25 +15% | - | +0.95 |
| L31 down c522 | 2% / 0 | **res%100** (R2 0.74): res mod 100 in {71, 81} | off (on 0) | - | same | +0.95 |
| L27 down c224 | 16% / 8% | **res%100** (R2 0.69): res mod 100 in {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95} [coarser: res mod 10 in {0}, R2 0.96] | **units(a,b)** (R2 0.58): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {1} | - | same | +0.95 |
| L27 gate c107 | 4% / 0 | **res%100** (R2 0.70): res mod 100 in {55..59} | off (on 0) | (reads) | same | +0.95 |
| L24 down c52 | 9% / 3% | **res%100** (R2 0.75): res mod 100 in {14, 24, 34..35, 44, 54, 74, 84, 94} [coarser: res mod 50 in {14, 24, 34, 44}, R2 0.81] | unexplained (best R2 0.45) | res: mod4 +5% | - | +0.95 |
| L29 up c308 | 9% / 3% | **res** (R2 0.78): res in {40..52, 146..149} | unexplained (best R2 0.27) | (reads) | same | +0.95 |
| L23 down c15 | 12% / 10% | **res%100** (R2 0.78): res mod 100 in {30..35, 38..42} | unexplained (best R2 0.38) | res: mod25 +6% | - | +0.95 |
| L24 gate c755 | 10% / 6% | **res%10** (R2 0.98): res mod 10 in {5} | **res%100** (R2 0.61): res mod 100 in {5, 15, 25, 85, 95} | (reads) | same | +0.95 |
| L28 up c67 | 13% / 11% | **res%50** (R2 0.80): res mod 50 in {7, 17, 26..27, 37, 47} [coarser: res mod 10 in {7}, R2 0.80] | **res** (R2 0.63): res in {-97, -23, -13, -3, 7, 17, 25..28, 37, 47, 57, 67, 77, 87, 97} | (reads) | same | +0.95 |
| L23 gate c20 | 28% / 13% | **res%100** (R2 0.78): res mod 100 in {23..41, 83..91} | **res** (R2 0.54): res in {23..35, 83..89} | (reads) | same | +0.95 |
| L21 gate c175 | 31% / 19% | **res%10** (R2 0.96): res mod 10 in {7..9} | **res** (R2 0.61): res in {-97, -23, -13, -3..-2, 7..9, 17..19, 27..29, 37..39, 47..49, 57..59, 67..69, 77..79, 87..89, 97..99} | (reads) | same | +0.95 |
| L30 gate c465 | 11% / 4% | **res%100** (R2 0.82): res mod 100 in {29..39} | **res//10** (R2 0.63): (tens) res in {31..37} | (reads) | same | +0.95 |
| L27 gate c168 | 8% / 7% | **res%100** (R2 0.75): res mod 100 in {35..37, 39..44} | unexplained (best R2 0.50) | (reads) | same | +0.95 |
| L29 down c178 | 4% / 3% | **res%100** (R2 0.61): res mod 100 in {27..29, 68} | unexplained (best R2 0.43) | - | same | +0.95 |
| L31 gate c137 | 2% / 0 | **res%100** (R2 0.86): res mod 100 in {97..98} | off (on 0) | (reads) | same | +0.95 |
| L26 down c182 | 3% / 13% | **res** (R2 0.77): res in {20..30} | unexplained (best R2 0.48) | - | same | +0.95 |
| L20 down c13 | 42% / 39% | **res%10** (R2 0.86): res mod 10 in {5..8} | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {2,3,4,5}; a%10 in {1} -> b%10 in {3,4,5,6}; a%10 in {2} -> b%10 in {4,5,6,7}; a%10 in {3} -> b%10 in {5,6,7}; a%10 in {4} -> b%10 in {6,7,8,9}; a%10 in {5} -> b%10 in {0,8,9}; a%10 in {6} -> b%10 in {0,1,8,9}; a%10 in {7} -> b%10 in {0,1,2,3,9}; a%10 in {8} -> b%10 in {0,1,2,3,4}; a%10 in {9} -> b%10 in {1,2,3,4,5} | res: mod10 +17% | res: mod10 +17% | +0.95 |
| L27 down c849 | 5% / 2% | **res** (R2 0.78): res in {61..68} | unexplained (best R2 0.38) | - | same | +0.95 |
| L24 up c99 | 13% / 6% | **res%100** (R2 0.80): res mod 100 in {25..36} | **res** (R2 0.59): res in {26..33} | (reads) | same | +0.95 |
| L29 gate c725 | 6% / 1% | **res** (R2 0.80): res in {67..68, 70, 72..76} | unexplained (best R2 0.35) | (reads) | same | +0.95 |
| L20 down c47 | 43% / 28% | **res%2** (R2 0.74): res mod 2 in {0} | **units(a,b)** (R2 0.75): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} | a: mod2 -8%; b: mod2 -5%; res: mod2 +16% | a: mod2 -9%; b: mod2 -5%; res: mod2 +13% | +0.95 |
| L28 down c115 | 12% / 3% | **res%100** (R2 0.94): res mod 100 in {78..88} | unexplained (best R2 0.46) | - | same | +0.95 |
| L29 gate c300 | 4% / 1% | **res%100** (R2 0.74): res mod 100 in {57..60, 62} | unexplained (best R2 0.12) | (reads) | same | +0.95 |
| L28 down c82 | 16% / 8% | **res%100** (R2 0.84): res mod 100 in {2..3, 13, 23, 33, 39..45, 53, 63, 73, 83, 93} | **res** (R2 0.60): res in {3, 13, 23, 33, 40..44, 53, 63, 73, 83, 93} | - | same | +0.95 |
| L26 down c118 | 18% / 6% | **res%100** (R2 0.93): res mod 100 in {69..86} | unexplained (best R2 0.46) | - | same | +0.95 |
| L30 gate c739 | 10% / 6% | **res%10** (R2 0.87): res mod 10 in {9} | **res** (R2 0.53): res in {9, 19, 29, 39, 49, 59, 69, 79} | (reads) | same | +0.95 |
| L30 down c207 | 8% / 3% | **res%100** (R2 0.80): res mod 100 in {26..33} | **res** (R2 0.65): res in {28..31} | - | same | +0.95 |
| L24 gate c31 | 29% / 23% | **res%100** (R2 0.88): res mod 100 in {0..1, 4..24, 96..99} | **res%100** (R2 0.61): res mod 100 in {1..4, 6..22, 99} | (reads) | same | +0.95 |
| L26 up c320 | 4% / 1% | **res%100** (R2 0.60): res mod 100 in {84..87} | unexplained (best R2 0.23) | (reads) | same | +0.95 |
| L24 up c17 | 21% / 13% | **res//10** (R2 0.74): (tens) res in {38..58, 140..156} [coarser: res mod 100 in {39..56}, R2 0.97] | unexplained (best R2 0.46) | (reads) | same | +0.95 |
| L28 gate c903 | 10% / 3% | **res%10** (R2 0.93): res mod 10 in {2} | **res** (R2 0.55): res in {12, 22, 32} | (reads) | same | +0.95 |
| L23 gate c123 | 10% / 5% | **res%10** (R2 0.95): res mod 10 in {2} | **res** (R2 0.64): res in {2, 12, 22, 32, 72, 82, 92} | (reads) | same | +0.95 |
| L27 down c82 | 16% / 5% | **res%100** (R2 0.91): res mod 100 in {50..65} | unexplained (best R2 0.36) | - | same | +0.95 |
| L22 down c6 | 16% / 14% | **res%100** (R2 0.88): res mod 100 in {0..1, 3..15} | **res%100** (R2 0.74): res mod 100 in {1, 3..14} | res: mod100 +4%, mod50 +2%, mod25 +11% | res: mod100 +4%, mod50 +5%, mod25 +11%, mod20 +3% | +0.95 |
| L26 up c37 | 2% / 3% | **res%100** (R2 0.81): res mod 100 in {7..8} | **res%100** (R2 0.71): res mod 100 in {8} | (reads) | same | +0.95 |
| L30 down c230 | 10% / 6% | **res%10** (R2 0.89): res mod 10 in {8} | unexplained (best R2 0.44) | res: mod5 +2% | - | +0.95 |
| L29 gate c566 | 10% / 8% | **res//10** (R2 0.85): (tens) res in {31..39, 130..139} [coarser: res mod 100 in {30..39}, R2 1.00] | unexplained (best R2 0.48) | (reads) | same | +0.95 |
| L30 gate c159 | 9% / 4% | **res//10** (R2 0.82): (tens) res in {22..27, 121..129} [coarser: res mod 100 in {21..28}, R2 0.95] | **res** (R2 0.79): res in {22..27} | (reads) | same | +0.95 |
| L24 down c11 | 17% / 9% | **res%100** (R2 0.86): res mod 100 in {21..28, 61..69} | **res** (R2 0.50): res in {22..27, 62..66} | res: mod25 +3%, mod20 +3% | - | +0.95 |
| L31 gate c360 | 3% / 0 | **res%100** (R2 0.70): res mod 100 in {93..96} | off (on 0) | (reads) | same | +0.95 |
| L28 up c218 | 8% / 4% | **res** (R2 0.82): res in {18..20, 116..123, 160, 178, 180} | **res** (R2 0.61): res in {18..22} | (reads) | same | +0.95 |
| L20 up c13 | 43% / 33% | **res%10** (R2 0.88): res mod 10 in {5..8} | **res** (R2 0.59): res in {-97, -95, -35..-33, -25..-23, -15..-13, -6..-2, 5..8, 15..18, 25..28, 35..38, 45..48, 55..58, 65..68, 74..78, 85..88, 95..98} | (reads) | same | +0.95 |
| L23 gate c310 | 10% / 4% | **res%10** (R2 0.91): res mod 10 in {2} | **res** (R2 0.71): res in {2, 12, 22, 32, 42, 52, 72, 82, 92} | (reads) | same | +0.95 |
| L27 down c51 | 9% / 3% | **res%100** (R2 0.82): res mod 100 in {38..45} | unexplained (best R2 0.48) | - | same | +0.95 |
| L30 down c882 | 2% / 3% | **res** (R2 0.77): res in {25..29, 127} | **res** (R2 0.58): res in {26..29} | - | same | +0.94 |
| L21 down c16 | 11% / 10% | **res%10** (R2 0.93): res mod 10 in {6} | **units(a,b)** (R2 0.55): a%10 in {0} -> b%10 in {4}; a%10 in {1} -> b%10 in {5}; a%10 in {2} -> b%10 in {6}; a%10 in {4} -> b%10 in {8}; a%10 in {5} -> b%10 in {9}; a%10 in {6} -> b%10 in {0}; a%10 in {7} -> b%10 in {1}; a%10 in {8} -> b%10 in {2}; a%10 in {9} -> b%10 in {3} | res: mod10 +3%, mod5 +4% | res: mod5 +3% | +0.94 |
| L28 gate c706 | 2% / 0 | **res%100** (R2 0.79): res mod 100 in {54..55} | off (on 0) | (reads) | same | +0.94 |
| L28 down c275 | 5% / 4% | **res%20** (R2 0.80): res mod 20 in {1} | unexplained (best R2 0.42) | - | same | +0.94 |
| L30 gate c407 | 4% / 1% | **res%100** (R2 0.82): res mod 100 in {26..29} | unexplained (best R2 0.39) | (reads) | same | +0.94 |
| L21 up c0 | 35% / 24% | **res//10** (R2 0.83): (tens) res in {8..40, 108..140} [coarser: res mod 100 in {8..40}, R2 0.99] | **tens(a,b)** (R2 0.70): a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,1,10}; a//10 in {3} -> b//10 in {0,1,2}; a//10 in {4} -> b//10 in {1,2,3}; a//10 in {5} -> b//10 in {2,3,4}; a//10 in {6} -> b//10 in {3,4,5}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {5,6,7}; a//10 in {9,10} -> b//10 in {6,7,8} | (reads) | same | +0.94 |
| L24 gate c285 | 10% / 5% | **res%10** (R2 0.97): res mod 10 in {8} | **res** (R2 0.62): res in {8, 18, 28, 38, 48, 88, 98} | (reads) | same | +0.94 |
| L29 gate c209 | 12% / 4% | **res%10** (R2 0.83): res mod 10 in {2} | **res** (R2 0.60): res in {12, 22, 32, 92} | (reads) | same | +0.94 |
| L21 up c43 | 27% / 33% | **res%100** (R2 0.82): res mod 100 in {0..17, 88..99} | **res%100** (R2 0.61): res mod 100 in {0..19, 85..99} | (reads) | same | +0.94 |
| L30 gate c660 | 8% / 5% | **res%100** (R2 0.74): res mod 100 in {23..27, 75, 85} | **res** (R2 0.53): res in {24..27} | (reads) | same | +0.94 |
| L30 down c232 | 9% / 1% | **res%100** (R2 0.74): res mod 100 in {77, 81..84, 88} | unexplained (best R2 0.31) | - | same | +0.94 |
| L29 up c217 | 11% / 3% | **res%100** (R2 0.83): res mod 100 in {38..44, 87..91} [coarser: res mod 50 in {37..42}, R2 0.82] | unexplained (best R2 0.44) | (reads) | same | +0.94 |
| L30 gate c580 | 5% / 5% | **res** (R2 0.67): res in {31..35, 52, 82, 131..133, 182} [coarser: res mod 100 in {31..33, 82}, R2 0.89] | unexplained (best R2 0.33) | (reads) | same | +0.94 |
| L26 gate c299 | 15% / 2% | **res%100** (R2 0.93): res mod 100 in {66..80} | unexplained (best R2 0.41) | (reads) | same | +0.94 |
| L30 gate c205 | 10% / 2% | **res%100** (R2 0.78): res mod 100 in {42..50} | **res** (R2 0.50): res in {45..48} | (reads) | same | +0.94 |
| L31 down c495 | 3% / 1% | **res%100** (R2 0.90): res mod 100 in {76..78} | unexplained (best R2 0.29) | - | same | +0.94 |
| L24 down c50 | 6% / 3% | **res%100** (R2 0.75): res mod 100 in {25..27, 76, 86} | unexplained (best R2 0.42) | res: mod4 +3% | - | +0.94 |
| L23 down c24 | 10% / 7% | **res%10** (R2 0.96): res mod 10 in {7} | **res** (R2 0.67): res in {-97, -23, -13, -3, 7, 17, 27, 37, 47, 57, 67, 77, 87, 97} | res: mod10 +3%, mod5 +3% | res: mod5 +3% | +0.94 |
| L31 down c219 | 5% / 1% | **res%100** (R2 0.80): res mod 100 in {81..86} | **res** (R2 0.52): res in {81..86} | - | same | +0.94 |
| L29 up c144 | 12% / 4% | **res%100** (R2 0.91): res mod 100 in {59..69} | unexplained (best R2 0.35) | (reads) | same | +0.94 |
| L27 down c417 | 2% / 0 | **res** (R2 0.84): res in {111..112} | off (on 0) | - | same | +0.94 |
| L30 down c808 | 5% / 0 | **res%100** (R2 0.83): res mod 100 in {67..70} | off (on 0) | - | same | +0.94 |
| L21 down c14 | 11% / 10% | **res%10** (R2 0.89): res mod 10 in {5} | **res%10** (R2 0.60): res mod 10 in {5} | res: mod10 +5%, mod5 +5% | res: mod10 +3%, mod5 +4% | +0.94 |
| L26 up c118 | 3% / 0 | **res** (R2 0.66): res in {77..78, 87, 147, 167, 177..178} [coarser: res mod 100 in {47, 77..78}, R2 0.87] | off (on 0) | (reads) | same | +0.94 |
| L28 down c952 | 2% / 0 | **res%100** (R2 0.89): res mod 100 in {69} | off (on 0) | - | same | +0.94 |
| L30 down c603 | 3% / 1% | **res%100** (R2 0.68): res mod 100 in {69..70} | unexplained (best R2 0.14) | - | same | +0.94 |
| L20 up c223 | 10% / 6% | **res//10** (R2 0.77): (tens) res in {110..120} | **res** (R2 0.68): res in {11..18} | (reads) | same | +0.94 |
| L25 gate c139 | 10% / 1% | **res%100** (R2 0.87): res mod 100 in {88..95, 98} | unexplained (best R2 0.48) | (reads) | same | +0.94 |
| L27 down c835 | 5% / 1% | **res** (R2 0.88): res in {31..33, 129..134} [coarser: res mod 100 in {29, 31..34}, R2 0.86] | unexplained (best R2 0.40) | - | same | +0.94 |
| L29 up c434 | 2% / 0 | **res%100** (R2 0.59): res mod 100 in {80..81, 83} | off (on 0) | (reads) | same | +0.94 |
| L20 gate c47 | 40% / 26% | **units(a,b)** (R2 0.78): a%10 in {0} -> b%10 in {4,6}; a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9}; a%10 in {2} -> b%10 in {4,6,8}; a%10 in {4,6} -> b%10 in {0,2,4,6,8}; a%10 in {8} -> b%10 in {2,4,6,8} | **units(a,b)** (R2 0.71): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} | (reads) | same | +0.94 |
| L29 gate c842 | 10% / 5% | **res%10** (R2 0.96): res mod 10 in {1} | **res** (R2 0.61): res in {11, 21, 31, 41, 51, 81, 91} | (reads) | same | +0.94 |
| L25 gate c200 | 22% / 8% | **res%100** (R2 0.83): res mod 100 in {38..60} | unexplained (best R2 0.47) | (reads) | same | +0.94 |
| L25 down c15 | 9% / 8% | **res%100** (R2 0.87): res mod 100 in {0..7} | **res%100** (R2 0.68): res mod 100 in {1..7} | res: mod100 +2%, mod25 +4% | res: mod25 +5% | +0.94 |
| L22 gate c48 | 15% / 10% | **res%100** (R2 0.74): res mod 100 in {27..34, 36..38, 48, 68, 88} | **res** (R2 0.52): res in {26..36, 38} | (reads) | same | +0.94 |
| L24 down c82 | 8% / 5% | **res%100** (R2 0.81): res mod 100 in {11, 21, 31..32, 41..42, 81..82} | unexplained (best R2 0.50) | - | same | +0.94 |
| L27 up c51 | 13% / 5% | **res%100** (R2 0.83): res mod 100 in {37..48} | unexplained (best R2 0.48) | (reads) | same | +0.94 |
| L22 gate c8 | 10% / 11% | **res%10** (R2 0.98): res mod 10 in {0} | **res%100** (R2 0.52): res mod 100 in {0..1, 10, 20, 30, 40, 50, 60, 70, 80, 90} | (reads) | same | +0.94 |
| L30 up c232 | 3% / 0 | **res%100** (R2 0.80): res mod 100 in {80..81, 83} | off (on 0) | (reads) | same | +0.94 |
| L26 up c157 | 10% / 4% | **res%10** (R2 0.84): res mod 10 in {5} | **res** (R2 0.51): res in {15, 25, 35, 45, 85, 95} | (reads) | same | +0.94 |
| L26 down c4 | 19% / 8% | **res** (R2 0.90): res in {23..29, 119..129, 131..143} | **res** (R2 0.57): res in {22..29, 34..35} | res: mod25 +3% | - | +0.94 |
| L28 up c82 | 12% / 8% | **res%10** (R2 0.79): res mod 10 in {3} | **res** (R2 0.61): res in {-17, -7, 3, 13, 23, 33, 43, 53, 63, 73, 83, 93} | (reads) | same | +0.94 |
| L29 down c762 | 1% / 2% | **res** (R2 0.56): res in {15, 25, 35, 65} | **res** (R2 0.55): res in {15, 25, 35} | - | same | +0.94 |
| L30 down c407 | 6% / 2% | **res%100** (R2 0.76): res mod 100 in {25..29} | unexplained (best R2 0.50) | - | same | +0.94 |
| L25 gate c830 | 10% / 3% | **res%10** (R2 0.96): res mod 10 in {9} | **res** (R2 0.67): res in {9, 19, 29, 39, 79, 89, 99} | (reads) | same | +0.94 |
| L30 up c414 | 11% / 6% | **res%10** (R2 0.93): res mod 10 in {2} | **res** (R2 0.51): res in {-18, -8, 22, 32, 42, 52, 62, 72, 82, 92} [coarser: res mod 100 in {22, 32, 42, 62, 72, 82, 92}, R2 0.80] | (reads) | same | +0.94 |
| L24 down c593 | 4% / 0 | **res** (R2 0.67): res in {70, 90, 110, 130, 140, 150, 170} | off (on 0) | - | same | +0.94 |
| L24 down c191 | 8% / 3% | **res%100** (R2 0.91): res mod 100 in {76..83} | unexplained (best R2 0.33) | - | same | +0.94 |
| L26 up c33 | 7% / 8% | **res%100** (R2 0.76): res mod 100 in {9..15} | **res%100** (R2 0.55): res mod 100 in {10..14} | (reads) | same | +0.94 |
| L29 up c467 | 8% / 9% | **res%100** (R2 0.84): res mod 100 in {13..19} | **res** (R2 0.70): res in {9, 12..19, 21} | (reads) | same | +0.94 |
| L30 gate c700 | 6% / 1% | **res%100** (R2 0.86): res mod 100 in {50..55} | unexplained (best R2 0.25) | (reads) | same | +0.94 |
| L26 gate c782 | 0 / 2% | off (on 0) | **res%100** (R2 0.78): res mod 100: no class above 0.5 (max 0.43) | (reads) | same | +0.94 |
| L28 gate c271 | 10% / 6% | **res%10** (R2 0.94): res mod 10 in {3} | **res** (R2 0.68): res in {3, 13, 23, 33, 43, 53, 63, 73, 83, 93} | (reads) | same | +0.94 |
| L25 down c151 | 8% / 3% | **res%100** (R2 0.74): res mod 100 in {25..30, 68} | **res** (R2 0.54): res in {25..30} | - | same | +0.94 |
| L24 up c39 | 15% / 6% | **res%100** (R2 0.82): res mod 100 in {0, 50, 88..99} | unexplained (best R2 0.38) | (reads) | same | +0.94 |
| L26 down c8 | 7% / 9% | **res%100** (R2 0.84): res mod 100 in {19..24} | unexplained (best R2 0.49) | - | res: mod25 +2% | +0.94 |
| L29 down c150 | 8% / 3% | **res%100** (R2 0.81): res mod 100 in {81..87} | unexplained (best R2 0.42) | - | same | +0.94 |
| L26 down c363 | 6% / 1% | **res%20** (R2 0.77): res mod 20 in {12} | unexplained (best R2 0.36) | - | same | +0.94 |
| L30 gate c810 | 10% / 8% | **res%10** (R2 0.95): res mod 10 in {2} | **res** (R2 0.57): res in {-28, -18, -8, 2, 22, 32, 42, 52, 62, 72, 82, 92} | (reads) | same | +0.94 |
| L30 gate c133 | 10% / 7% | **res%10** (R2 0.86): res mod 10 in {8} | **res** (R2 0.63): res in {-12, -2, 8, 18, 28, 38, 48, 58, 68, 78, 88, 98} | (reads) | same | +0.94 |
| L28 down c203 | 1% / 4% | **res** (R2 0.68): res in {12..16} | **res** (R2 0.73): res in {13..16} | - | same | +0.94 |
| L28 gate c46 | 10% / 9% | **res%100** (R2 0.70): res mod 100 in {28..35, 80} | unexplained (best R2 0.46) | (reads) | same | +0.94 |
| L29 down c761 | 0 / 3% | off (on 0) | **res%100** (R2 0.59): res mod 100: no class above 0.5 (max 0.45) | - | same | +0.94 |
| L25 gate c11 | 11% / 11% | **res%10** (R2 0.89): res mod 10 in {7} | **units(a,b)** (R2 0.51): a%10 in {0} -> b%10 in {3}; a%10 in {1} -> b%10 in {4}; a%10 in {2} -> b%10 in {5}; a%10 in {3} -> b%10 in {6}; a%10 in {5} -> b%10 in {8}; a%10 in {6} -> b%10 in {9}; a%10 in {7} -> b%10 in {0}; a%10 in {8} -> b%10 in {1}; a%10 in {9} -> b%10 in {2} | (reads) | same | +0.94 |
| L29 up c435 | 9% / 2% | **res//10** (R2 0.84): (tens) res in {61..69, 161..169} [coarser: res mod 100 in {61..69}, R2 1.00] | unexplained (best R2 0.34) | (reads) | same | +0.94 |
| L28 down c121 | 10% / 4% | **res%100** (R2 0.85): res mod 100 in {9, 19, 28..29, 39, 49, 59, 69, 79, 89} [coarser: res mod 50 in {9, 19, 28..29, 39, 49}, R2 0.85] | unexplained (best R2 0.42) | - | same | +0.94 |
| L24 up c12 | 17% / 12% | **res%10** (R2 0.82): res mod 10 in {5, 8} | **res%100** (R2 0.61): res mod 100 in {5, 8, 15, 25, 35, 45, 55, 65, 75, 85, 95, 98} | (reads) | same | +0.94 |
| L29 up c375 | 8% / 4% | **res%100** (R2 0.83): res mod 100 in {10..15} | **res** (R2 0.75): res in {10..13, 15} | (reads) | same | +0.94 |
| L25 gate c30 | 24% / 7% | **res%100** (R2 0.86): res mod 100 in {51..74} | unexplained (best R2 0.40) | (reads) | same | +0.94 |
| L22 gate c33 | 20% / 4% | **res%100** (R2 0.80): res mod 100 in {1, 87..98} | unexplained (best R2 0.46) | (reads) | same | +0.94 |
| L28 down c160 | 10% / 6% | **res%10** (R2 0.93): res mod 10 in {7} | **res** (R2 0.66): res in {-97, -3, 7, 17, 27, 37, 47, 57, 67, 77, 87, 97} | - | same | +0.94 |
| L24 down c212 | 4% / 2% | **res%100** (R2 0.61): res mod 100 in {20, 40, 80} | **res** (R2 0.55): res in {20} | - | same | +0.94 |
| L28 down c40 | 3% / 1% | **res%100** (R2 0.73): res mod 100 in {1} | unexplained (best R2 0.32) | - | same | +0.94 |
| L24 down c83 | 10% / 4% | **res%10** (R2 0.96): res mod 10 in {9} | **res** (R2 0.69): res in {9, 19, 29, 39, 69, 79, 89, 99} | - | same | +0.94 |
| L20 up c180 | 43% / 44% | **res%100** (R2 0.66): res mod 100 in {11..13, 15..28, 30..33, 57, 60..63, 65..82} [coarser: res mod 50 in {10..13, 15..33}, R2 0.86] | **tens(a,b)** (R2 0.53): a//10 in {2} -> b//10 in {0,1,6}; a//10 in {3} -> b//10 in {0,1,2,6,7,8}; a//10 in {4} -> b//10 in {1,2,3,6,7,8}; a//10 in {5} -> b//10 in {2,3,4,7,8}; a//10 in {6} -> b//10 in {3,4,5,8,9}; a//10 in {7} -> b//10 in {0,4,5,6,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,7,8,10}; a//10 in {9} -> b//10 in {1,2,6,7,8}; a//10 in {10} -> b//10 in {2,3,4,6,7,8,9} | (reads) | same | +0.94 |
| L22 down c30 | 7% / 2% | **res%100** (R2 0.71): res mod 100 in {31..36} | unexplained (best R2 0.28) | - | same | +0.93 |
| L26 gate c94 | 16% / 3% | **res%100** (R2 0.88): res mod 100 in {52..66, 68} | unexplained (best R2 0.41) | (reads) | same | +0.93 |
| L26 gate c139 | 10% / 11% | **res%100** (R2 0.74): res mod 100 in {0..6, 95..99} | **res%100** (R2 0.50): res mod 100 in {0..4, 97..99} | (reads) | same | +0.93 |
| L29 up c529 | 2% / 0 | **res%100** (R2 0.75): res mod 100 in {53..54} | off (on 0) | (reads) | same | +0.93 |
| L30 down c309 | 3% / 1% | **res%100** (R2 0.81): res mod 100 in {4..5} | **res%100** (R2 0.52): res mod 100: no class above 0.5 (max 0.42) | - | same | +0.93 |
| L26 gate c137 | 0 / 2% | off (on 0) | **res%100** (R2 0.74): res mod 100: no class above 0.5 (max 0.41) | (reads) | same | +0.93 |
| L24 gate c12 | 21% / 17% | **res%10** (R2 0.95): res mod 10 in {5, 8} | **res%100** (R2 0.52): res mod 100 in {5, 8, 15, 18, 25, 35, 45, 55, 65, 75, 78, 85, 88, 95, 98} [coarser: res mod 50 in {5, 15, 25, 35, 38, 45, 48}, R2 0.81] | (reads) | same | +0.93 |
| L28 gate c517 | 2% / 0 | **res%100** (R2 0.87): res mod 100 in {54..55} | off (on 0) | (reads) | same | +0.93 |
| L23 up c340 | 7% / 2% | **res%100** (R2 0.62): res mod 100 in {31..34, 72} | unexplained (best R2 0.36) | (reads) | same | +0.93 |
| L21 gate c1 | 51% / 32% | **res%2** (R2 0.96): res mod 2 in {1} | **res** (R2 0.64): res in {-99, -97, -95, -21, -11, -9, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99} | (reads) | same | +0.93 |
| L21 down c2 | 36% / 28% | **res%20** (R2 0.89): res mod 20 in {10..16} | **res** (R2 0.55): res in {-10..-4, 10..17, 30..36, 50..56, 70..76, 90..96} | res: mod20 +17% | res: mod25 +6%, mod20 +14% | +0.93 |
| L20 up c100 | 30% / 23% | **res%10** (R2 0.95): res mod 10 in {3..5} | **res** (R2 0.63): res in {-6..-5, 3..5, 13..15, 23..25, 33..35, 43..45, 53..55, 63..65, 73..75, 83..85, 93..96, 99} | (reads) | same | +0.93 |
| L27 gate c82 | 17% / 3% | **res%100** (R2 0.94): res mod 100 in {50..65} | unexplained (best R2 0.39) | (reads) | same | +0.93 |
| L21 up c13 | 31% / 15% | **res%10** (R2 0.92): res mod 10 in {2..4} | **res** (R2 0.64): res in {2..4, 12..14, 22..24, 32..34, 43..44, 54, 64, 74, 83..84, 92..94} | (reads) | same | +0.93 |
| L23 up c20 | 12% / 14% | **res%100** (R2 0.76): res mod 100 in {27..34, 70..71} | unexplained (best R2 0.38) | (reads) | same | +0.93 |
| L26 gate c118 | 19% / 8% | **res%100** (R2 0.93): res mod 100 in {67..85} | **tens(a,b)** (R2 0.52): a//10 in {7} -> b//10 in {0,10}; a//10 in {8} -> b//10 in {0,1}; a//10 in {9} -> b//10 in {1,2}; a//10 in {10} -> b//10 in {2,3} | (reads) | same | +0.93 |
| L26 up c8 | 15% / 11% | **res%100** (R2 0.90): res mod 100 in {15..28} | **res** (R2 0.63): res in {16..26} | (reads) | same | +0.93 |
| L19 up c85 | 29% / 33% | unexplained (best R2 0.41) | unexplained (best R2 0.39) | (reads) | same | +0.93 |
| L30 down c700 | 1% / 0 | **res** (R2 0.64): res in {152} | off (on 0) | - | same | +0.93 |
| L23 gate c17 | 17% / 4% | **res%100** (R2 0.94): res mod 100 in {64..80} | unexplained (best R2 0.36) | (reads) | same | +0.93 |
| L29 gate c578 | 1% / 2% | **res%100** (R2 0.76): res mod 100 in {13} | **res** (R2 0.81): res in {13..14} | (reads) | same | +0.93 |
| L26 down c115 | 3% / 16% | **tens(a,b)** (R2 0.56): a//10 in {0} -> b//10 in {2,3}; a//10 in {1} -> b//10 in {2}; a//10 in {3} -> b//10 in {0} | **tens(a,b)** (R2 0.55): a//10 in {3} -> b//10 in {0,3,4,5,10}; a//10 in {4} -> b//10 in {0,1,5}; a//10 in {5} -> b//10 in {1,2}; a//10 in {6} -> b//10 in {2,3}; a//10 in {7} -> b//10 in {3,4}; a//10 in {8} -> b//10 in {4,5}; a//10 in {10} -> b//10 in {6,7} | - | same | +0.93 |
| L28 gate c312 | 9% / 3% | **res//10** (R2 0.85): (tens) res in {70..79, 171..179} [coarser: res mod 100 in {70..79}, R2 0.98] | unexplained (best R2 0.33) | (reads) | same | +0.93 |
| L30 gate c357 | 7% / 1% | **res%100** (R2 0.86): res mod 100 in {83..89} | unexplained (best R2 0.40) | (reads) | same | +0.93 |
| L29 down c308 | 4% / 1% | **res%100** (R2 0.72): res mod 100 in {46..49} | unexplained (best R2 0.24) | - | same | +0.93 |
| L22 up c9 | 10% / 8% | **res%10** (R2 0.94): res mod 10 in {9} | **units(a,b)** (R2 0.51): a%10 in {0} -> b%10 in {1}; a%10 in {1} -> b%10 in {2}; a%10 in {2} -> b%10 in {3}; a%10 in {5} -> b%10 in {6}; a%10 in {6} -> b%10 in {7}; a%10 in {7} -> b%10 in {8}; a%10 in {9} -> b%10 in {0} | (reads) | same | +0.93 |
| L29 up c174 | 18% / 1% | **res%100** (R2 0.78): res mod 100 in {1..10, 95..99} | unexplained (best R2 0.38) | (reads) | same | +0.93 |
| L22 down c9 | 10% / 8% | **res%10** (R2 0.94): res mod 10 in {9} | **units(a,b)** (R2 0.50): a%10 in {0} -> b%10 in {1}; a%10 in {1} -> b%10 in {2}; a%10 in {2} -> b%10 in {3}; a%10 in {5} -> b%10 in {6}; a%10 in {6} -> b%10 in {7}; a%10 in {7} -> b%10 in {8}; a%10 in {9} -> b%10 in {0} | res: mod10 +6%, mod5 +7%, mod2 +3% | res: mod10 +2%, mod5 +3% | +0.93 |
| L30 gate c394 | 5% / 1% | **res%100** (R2 0.82): res mod 100 in {27, 37, 67, 87} | unexplained (best R2 0.49) | (reads) | same | +0.93 |
| L28 down c271 | 10% / 5% | **res%10** (R2 0.96): res mod 10 in {3} | **res** (R2 0.67): res in {3, 13, 23, 33, 53, 63, 73, 83, 93} | - | same | +0.93 |
| L29 down c358 | 9% / 3% | **res%100** (R2 0.83): res mod 100 in {1, 95..97, 99} | unexplained (best R2 0.48) | - | same | +0.93 |
| L29 gate c136 | 15% / 7% | **res%100** (R2 0.77): res mod 100 in {1, 11, 21, 31, 38..45, 51, 61, 71, 81, 91} | unexplained (best R2 0.45) | (reads) | same | +0.93 |
| L27 gate c51 | 11% / 4% | **res//10** (R2 0.81): (tens) res in {40..49, 140..151} [coarser: res mod 100 in {40..50}, R2 0.95] | unexplained (best R2 0.41) | (reads) | same | +0.93 |
| L25 up c50 | 9% / 4% | **res%10** (R2 0.90): res mod 10 in {3} | **res** (R2 0.63): res in {3, 13, 23, 83, 93} | (reads) | same | +0.93 |
| L28 down c46 | 8% / 5% | **res%100** (R2 0.73): res mod 100 in {29..33, 70, 80, 90} | unexplained (best R2 0.37) | - | same | +0.93 |
| L21 gate c0 | 32% / 21% | **res//10** (R2 0.84): (tens) res in {10..35, 108..140} [coarser: res mod 100 in {8..40}, R2 0.96] | **res** (R2 0.82): res in {8..32} | (reads) | same | +0.93 |
| L30 gate c424 | 2% / 0 | **res%100** (R2 0.87): res mod 100 in {79..80} | off (on 0) | (reads) | same | +0.93 |
| L22 down c98 | 8% / 1% | **res%100** (R2 0.70): res mod 100 in {54..60} | unexplained (best R2 0.21) | res: mod25 +2% | - | +0.93 |
| L26 down c360 | 5% / 1% | **res** (R2 0.76): res in {41..42, 141..146} [coarser: res mod 100 in {41..45}, R2 0.82] | unexplained (best R2 0.25) | - | same | +0.93 |
| L21 down c4 | 41% / 40% | **res%10** (R2 0.85): res mod 10 in {0..3} | **units(a,b)** (R2 0.46): a%10 in {0} -> b%10 in {0,7,8,9}; a%10 in {1} -> b%10 in {0,1,8,9}; a%10 in {2} -> b%10 in {0,1,2,8,9}; a%10 in {3} -> b%10 in {0,1,2,3,9}; a%10 in {4} -> b%10 in {1,2,3,4}; a%10 in {5} -> b%10 in {2,3,4,5,7}; a%10 in {6} -> b%10 in {4,5,6}; a%10 in {7} -> b%10 in {4,5,6,7}; a%10 in {8} -> b%10 in {5,6,7,8}; a%10 in {9} -> b%10 in {6,7,8,9} | res: mod10 +13% | res: mod10 +15% | +0.93 |
| L30 up c133 | 11% / 11% | **res%10** (R2 0.90): res mod 10 in {8} | **res%100** (R2 0.48): res mod 100 in {2, 8, 18, 28, 38, 48, 58, 68, 78, 88, 98} | (reads) | same | +0.93 |
| L30 gate c486 | 5% / 2% | **res%100** (R2 0.85): res mod 100 in {40, 79..82} | unexplained (best R2 0.29) | (reads) | same | +0.93 |
| L27 gate c320 | 12% / 9% | **units(a,b)** (R2 0.91): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {9}; a%10 in {2} -> b%10 in {8}; a%10 in {3} -> b%10 in {7}; a%10 in {4} -> b%10 in {6}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {3}; a%10 in {8} -> b%10 in {2}; a%10 in {9} -> b%10 in {1} | **units(a,b)** (R2 0.61): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {1}; a%10 in {4} -> b%10 in {4}; a%10 in {9} -> b%10 in {9} | (reads) | same | +0.93 |
| L25 gate c216 | 11% / 11% | **res%10** (R2 0.86): res mod 10 in {1} | **res%100** (R2 0.47): res mod 100 in {1, 11, 21, 31, 41, 51, 61, 71, 81, 91} [coarser: res mod 50 in {1, 11, 21, 31, 41}, R2 0.80] | (reads) | same | +0.93 |
| L30 down c159 | 9% / 4% | **res//10** (R2 0.82): (tens) res in {23..29, 120..129} [coarser: res mod 100 in {22..29}, R2 0.93] | **res** (R2 0.71): res in {23..29} | - | same | +0.93 |
| L24 gate c52 | 6% / 1% | **res%100** (R2 0.72): res mod 100 in {32, 34, 54, 94} | unexplained (best R2 0.33) | (reads) | same | +0.93 |
| L27 down c222 | 4% / 0 | **res** (R2 0.89): res in {131..135} | off (on 0) | - | same | +0.93 |
| L30 down c326 | 3% / 0 | **res%100** (R2 0.74): res mod 100 in {56..57} | off (on 0) | - | same | +0.93 |
| L30 up c207 | 6% / 2% | **res%100** (R2 0.80): res mod 100 in {27..31} | **res** (R2 0.63): res in {28..31} | (reads) | same | +0.93 |
| L26 gate c194 | 10% / 1% | **res%100** (R2 0.69): res mod 100 in {54, 56..63, 78} | unexplained (best R2 0.19) | (reads) | same | +0.93 |
| L26 down c88 | 5% / 1% | **res%100** (R2 0.70): res mod 100 in {46, 48, 86, 88} | unexplained (best R2 0.28) | res: mod4 +3% | - | +0.93 |
| L29 gate c174 | 18% / 4% | **res%100** (R2 0.86): res mod 100 in {0..7, 9..11, 95..99} | unexplained (best R2 0.46) | (reads) | same | +0.93 |
| L28 down c247 | 1% / 5% | **res** (R2 0.69): res in {5..8, 107} | **res%100** (R2 0.65): res mod 100 in {7..8} | - | same | +0.93 |
| L30 up c551 | 4% / 2% | **res%100** (R2 0.79): res mod 100 in {23..25, 84} | **res** (R2 0.80): res in {23..25} | (reads) | same | +0.93 |
| L21 gate c14 | 10% / 9% | **res%10** (R2 0.94): res mod 10 in {5} | unexplained (best R2 0.50) | (reads) | same | +0.93 |
| L24 gate c101 | 29% / 18% | **res%100** (R2 0.80): res mod 100 in {26..42, 44..52} | **tens(a,b)** (R2 0.56): a//10 in {3} -> b//10 in {0,3,4,10}; a//10 in {4} -> b//10 in {0,1,4,5,10}; a//10 in {5} -> b//10 in {0,1,2}; a//10 in {6} -> b//10 in {2,3}; a//10 in {7} -> b//10 in {3,4}; a//10 in {8} -> b//10 in {4,5}; a//10 in {10} -> b//10 in {5,6,7} | (reads) | same | +0.93 |
| L24 down c108 | 10% / 5% | **res%10** (R2 0.96): res mod 10 in {6} | **res** (R2 0.59): res in {6, 16, 26, 36, 86, 96} | - | same | +0.93 |
| L26 down c139 | 9% / 7% | **res%100** (R2 0.71): res mod 100 in {0..5, 96..99} | unexplained (best R2 0.44) | - | same | +0.93 |
| L28 gate c36 | 12% / 14% | **res%10** (R2 0.82): res mod 10 in {2} | **units(a,b)** (R2 0.50): a%10 in {0} -> b%10 in {8}; a%10 in {2} -> b%10 in {0,2,4,8}; a%10 in {3} -> b%10 in {1}; a%10 in {4} -> b%10 in {2}; a%10 in {5} -> b%10 in {3,7}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {5}; a%10 in {8} -> b%10 in {6}; a%10 in {9} -> b%10 in {7} | (reads) | same | +0.93 |
| L30 up c239 | 19% / 5% | **res%10** (R2 0.91): res mod 10 in {2..3} | unexplained (best R2 0.47) | (reads) | same | +0.93 |
| L24 up c36 | 8% / 1% | **res%100** (R2 0.89): res mod 100 in {70..77} | unexplained (best R2 0.25) | (reads) | same | +0.93 |
| L22 up c446 | 25% / 18% | **res%100** (R2 0.78): res mod 100 in {1, 4, 6..9, 11, 14..16, 94..97, 99} | **res//10** (R2 0.57): (tens) res in {-99, -97, -95, 1..17, 94..99} | (reads) | same | +0.93 |
| L29 gate c505 | 2% / 0 | **res%100** (R2 0.64): res mod 100 in {64, 84} | off (on 0) | (reads) | same | +0.93 |
| L30 down c660 | 4% / 5% | **res** (R2 0.65): res in {24..27, 75, 85, 125} [coarser: res mod 100 in {24..25, 75, 85}, R2 0.84] | unexplained (best R2 0.29) | - | same | +0.93 |
| L29 gate c134 | 6% / 3% | **res%100** (R2 0.90): res mod 100 in {8, 28, 38, 48, 68, 88} [coarser: res mod 20 in {8}, R2 0.87] | **res** (R2 0.52): res in {8, 28, 88} | (reads) | same | +0.93 |
| L25 down c130 | 1% / 0 | **res%100** (R2 0.59): res mod 100 in {56} | off (on 0) | - | same | +0.93 |
| L24 up c15 | 11% / 4% | **res%100** (R2 0.86): res mod 100 in {81..90} | unexplained (best R2 0.42) | (reads) | same | +0.93 |
| L28 gate c70 | 18% / 18% | **res//10** (R2 0.75): (tens) res in {34..59, 140..147} | **tens(a,b)** (R2 0.53): a//10 in {0,10} -> b//10 in {4,5}; a//10 in {4} -> b//10 in {0,4,5,9,10}; a//10 in {5} -> b//10 in {0,1,5,6,9,10}; a//10 in {6} -> b//10 in {1,2}; a//10 in {7} -> b//10 in {3}; a//10 in {8,9} -> b//10 in {4} | (reads) | same | +0.93 |
| L30 down c379 | 4% / 1% | **res** (R2 0.76): res in {72..74, 123, 133, 173..174} [coarser: res mod 100 in {73..74}, R2 0.84] | unexplained (best R2 0.20) | - | same | +0.93 |
| L26 down c341 | 7% / 2% | **res%100** (R2 0.83): res mod 100 in {91..96} | unexplained (best R2 0.32) | - | same | +0.93 |
| L28 down c69 | 10% / 11% | **res%10** (R2 0.93): res mod 10 in {6} | **res%100** (R2 0.46): res mod 100 in {6, 16, 26, 36, 46, 56, 66, 76, 86, 96} [coarser: res mod 50 in {6, 16, 26, 36, 46}, R2 0.81] | - | res: mod5 +2% | +0.93 |
| L27 gate c195 | 2% / 0 | **res%100** (R2 0.88): res mod 100 in {68..69} | off (on 0) | (reads) | same | +0.93 |
| L25 down c11 | 11% / 11% | **res%10** (R2 0.92): res mod 10 in {7} | **units(a,b)** (R2 0.51): a%10 in {0} -> b%10 in {3}; a%10 in {1} -> b%10 in {4}; a%10 in {2} -> b%10 in {5}; a%10 in {3} -> b%10 in {6}; a%10 in {5} -> b%10 in {8}; a%10 in {6} -> b%10 in {9}; a%10 in {7} -> b%10 in {0}; a%10 in {8} -> b%10 in {1}; a%10 in {9} -> b%10 in {2} | res: mod10 +5%, mod5 +5%, mod2 +4% | res: mod10 +3%, mod5 +4% | +0.93 |
| L19 gate c149 | 24% / 10% | **res%10** (R2 0.81): res mod 10 in {2..3} | **res** (R2 0.65): res in {2..3, 12..13, 22..23, 32..33, 42..43, 82..83, 92..93} | (reads) | same | +0.93 |
| L28 down c165 | 6% / 3% | **res%100** (R2 0.79): res mod 100 in {25..28, 66..68} | unexplained (best R2 0.48) | - | same | +0.93 |
| L24 down c5 | 11% / 12% | **res%10** (R2 0.90): res mod 10 in {3} | unexplained (best R2 0.47) | res: mod10 +7%, mod5 +8%, mod2 +4% | res: mod10 +5%, mod5 +6% | +0.93 |
| L25 gate c248 | 1% / 0 | **res%100** (R2 0.69): res mod 100 in {75} | off (on 0) | (reads) | same | +0.93 |
| L30 gate c183 | 4% / 1% | **res%100** (R2 0.67): res mod 100 in {32, 82} | unexplained (best R2 0.32) | (reads) | same | +0.93 |
| L23 gate c196 | 1% / 11% | **res//10** (R2 0.59): (tens) res in {2..10} | **res//10** (R2 0.66): (tens) res in {1..10} | (reads) | same | +0.93 |
| L30 gate c322 | 12% / 5% | **res%100** (R2 0.93): res mod 100 in {69..80} | unexplained (best R2 0.44) | (reads) | same | +0.93 |
| L25 down c114 | 7% / 2% | **res%100** (R2 0.84): res mod 100 in {86, 94..98} | unexplained (best R2 0.22) | - | same | +0.93 |
| L30 up c326 | 4% / 1% | **res** (R2 0.87): res in {55..57, 155..158, 177} [coarser: res mod 100 in {55..57, 77}, R2 0.87] | unexplained (best R2 0.13) | (reads) | same | +0.93 |
| L27 down c24 | 7% / 5% | **res%100** (R2 0.72): res mod 100 in {33..38} | unexplained (best R2 0.42) | - | same | +0.93 |
| L24 up c43 | 21% / 12% | **res%100** (R2 0.75): res mod 100 in {8..11, 29..30, 47..52, 69..70, 86..92} | unexplained (best R2 0.48) | (reads) | same | +0.93 |
| L27 down c203 | 5% / 1% | **res%100** (R2 0.79): res mod 100 in {27, 47, 67, 87} | unexplained (best R2 0.42) | - | same | +0.93 |
| L26 gate c136 | 4% / 3% | **res%100** (R2 0.86): res mod 100 in {20..22, 95} | **res** (R2 0.67): res in {20..22} | (reads) | same | +0.93 |
| L28 up c176 | 7% / 3% | **res** (R2 0.82): res in {39, 79, 89, 119, 129, 137..140, 159, 179, 189} | unexplained (best R2 0.43) | (reads) | same | +0.93 |
| L20 gate c31 | 36% / 31% | **res//10** (R2 0.78): (tens) res in {2..23, 91..125, 191..200} [coarser: res mod 100 in {0..23, 25, 91..99}, R2 0.98] | **tens(a,b)** (R2 0.69): a//10 in {0} -> b//10 in {0,1,10}; a//10 in {1} -> b//10 in {0,1}; a//10 in {2} -> b//10 in {0,1,2}; a//10 in {3} -> b//10 in {1,2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4,5}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {0,7,8,9,10} | (reads) | same | +0.92 |
| L19 down c33 | 33% / 4% | **res%100** (R2 0.73): res mod 100 in {1, 75..98} | **res** (R2 0.58): res in {78..99} | res: mod100 +2% | - | +0.92 |
| L26 down c393 | 3% / 1% | **res** (R2 0.74): res in {134, 136..139} | unexplained (best R2 0.35) | - | same | +0.92 |
| L20 gate c2 | 43% / 22% | **res//10** (R2 0.76): (tens) res in {24..55, 116..164} [coarser: res mod 100 in {20..58}, R2 0.86] | **res//10** (R2 0.56): (tens) res in {21..50} | (reads) | same | +0.92 |
| L22 down c212 | 10% / 5% | **res%10** (R2 0.97): res mod 10 in {3} | **res** (R2 0.65): res in {3, 13, 23, 33, 43, 83, 93} | - | same | +0.92 |
| L22 down c17 | 10% / 5% | **res%20** (R2 0.84): res mod 20 in {2..3} | **res** (R2 0.58): res in {-18, 2..3, 22..23, 42, 82} | res: mod20 +3%, mod4 +45% | - | +0.92 |
| L24 up c444 | 6% / 13% | **res%100** (R2 0.68): res mod 100 in {2..11} | **res%100** (R2 0.75): res mod 100 in {10..12} | (reads) | same | +0.92 |
| L27 gate c24 | 8% / 5% | **res%100** (R2 0.80): res mod 100 in {32..39} | unexplained (best R2 0.44) | (reads) | same | +0.92 |
| L28 up c115 | 13% / 4% | **res%100** (R2 0.94): res mod 100 in {77..89} | **tens(a,b)** (R2 0.54): a//10 in {8} -> b//10 in {0,1,10}; a//10 in {9,10} -> b//10 in {1} | (reads) | same | +0.92 |
| L29 down c263 | 2% / 2% | **res%100** (R2 0.66): res mod 100 in {13} | unexplained (best R2 0.39) | - | same | +0.92 |
| L22 gate c11 | 32% / 26% | **res%20** (R2 0.86): res mod 20 in {0..3, 18..19} | unexplained (best R2 0.49) | (reads) | same | +0.92 |
| L30 up c135 | 10% / 4% | **res//10** (R2 0.88): (tens) res in {80..89, 181..187} [coarser: res mod 100 in {80..89}, R2 0.96] | unexplained (best R2 0.32) | (reads) | same | +0.92 |
| L29 up c128 | 12% / 9% | **res//10** (R2 0.83): (tens) res in {2..10, 100..110, 200} [coarser: res mod 100 in {0..10}, R2 1.00] | **res%100** (R2 0.77): res mod 100 in {1, 6..9} | (reads) | same | +0.92 |
| L21 gate c68 | 51% / 36% | **res%2** (R2 0.81): res mod 2 in {0} | unexplained (best R2 0.48) | (reads) | same | +0.92 |
| L30 up c187 | 15% / 6% | **res%100** (R2 0.80): res mod 100 in {7, 17, 27, 34..39, 47, 57, 67, 77, 87, 97} [coarser: res mod 50 in {7, 17, 27, 37, 47}, R2 0.83] | **res** (R2 0.52): res in {17, 27, 37..38, 47, 57, 77, 87} | (reads) | same | +0.92 |
| L30 gate c223 | 4% / 4% | **res** (R2 0.66): res in {38..42, 44, 139..140} [coarser: res mod 100 in {39..40}, R2 0.81] | unexplained (best R2 0.34) | (reads) | same | +0.92 |
| L28 gate c165 | 12% / 5% | **res%100** (R2 0.83): res mod 100 in {24..31, 66..70} | **res** (R2 0.53): res in {25..31} | (reads) | same | +0.92 |
| L26 down c33 | 8% / 8% | **res%100** (R2 0.81): res mod 100 in {12, 16..20} | **res** (R2 0.60): res in {12..13, 16..21} | - | same | +0.92 |
| L30 up c486 | 10% / 2% | **res%100** (R2 0.78): res mod 100 in {75..83} | unexplained (best R2 0.24) | (reads) | same | +0.92 |
| L28 down c70 | 11% / 24% | **res//10** (R2 0.75): (tens) res in {36..57} | **tens(a,b)** (R2 0.56): a//10 in {0} -> b//10 in {4,5}; a//10 in {1,3,10} -> b//10 in {5}; a//10 in {4} -> b//10 in {0,4,5,9,10}; a//10 in {5} -> b//10 in {0,1,5,6,9,10}; a//10 in {6} -> b//10 in {0,1,2}; a//10 in {7} -> b//10 in {1,2,3}; a//10 in {8} -> b//10 in {3,4}; a//10 in {9} -> b//10 in {4} | - | same | +0.92 |
| L28 down c99 | 9% / 4% | **res%100** (R2 0.89): res mod 100 in {8, 28, 38, 48, 58, 68, 87..88, 98} [coarser: res mod 10 in {8}, R2 0.81] | **res** (R2 0.54): res in {8, 28, 38, 48, 88, 98} | - | same | +0.92 |
| L21 up c4 | 42% / 34% | **res%10** (R2 0.86): res mod 10 in {0..3} | **res** (R2 0.54): res in {-99, -39..-38, -29..-28, -19..-18, -10..-8, 0..3, 10..13, 20..23, 30..33, 40..43, 50..53, 60..63, 70..73, 80..83, 90..93} | (reads) | same | +0.92 |
| L31 down c527 | 2% / 0 | **res%100** (R2 0.84): res mod 100: no class above 0.5 (max 0.46) | off (on 0) | - | same | +0.92 |
| L27 down c140 | 11% / 1% | **res//10** (R2 0.83): (tens) res in {70..79, 87, 170..178} [coarser: res mod 100 in {70..79, 87}, R2 0.98] | unexplained (best R2 0.34) | - | same | +0.92 |
| L25 gate c24 | 12% / 9% | **res%10** (R2 0.81): res mod 10 in {9} | **units(a,b)** (R2 0.53): a%10 in {0} -> b%10 in {1}; a%10 in {1} -> b%10 in {2}; a%10 in {2} -> b%10 in {3}; a%10 in {5} -> b%10 in {6}; a%10 in {6} -> b%10 in {7}; a%10 in {7} -> b%10 in {8}; a%10 in {8} -> b%10 in {9}; a%10 in {9} -> b%10 in {0} | (reads) | same | +0.92 |
| L21 down c53 | 10% / 6% | **res%10** (R2 0.96): res mod 10 in {9} | **res** (R2 0.60): res in {-99, -11, -1, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99} | res: mod5 +2% | - | +0.92 |
| L20 up c6 | 57% / 29% | unexplained (best R2 0.34) | unexplained (best R2 0.29) | (reads) | same | +0.92 |
| L25 gate c50 | 17% / 8% | **res%100** (R2 0.84): res mod 100 in {3, 13, 20, 22..28, 33, 43, 53, 63..64, 73, 83, 93} | **res** (R2 0.65): res in {3, 13, 20, 22..28, 33, 43, 83} | (reads) | same | +0.92 |
| L26 down c219 | 1% / 0 | **res%100** (R2 0.83): res mod 100 in {61} | off (on 0) | - | same | +0.92 |
| L25 down c30 | 19% / 5% | **res%100** (R2 0.78): res mod 100 in {52..57, 59..67, 69..73} | unexplained (best R2 0.30) | - | same | +0.92 |
| L25 down c24 | 11% / 9% | **res%10** (R2 0.86): res mod 10 in {9} | **units(a,b)** (R2 0.53): a%10 in {0} -> b%10 in {1}; a%10 in {1} -> b%10 in {2}; a%10 in {2} -> b%10 in {3}; a%10 in {5} -> b%10 in {6}; a%10 in {6} -> b%10 in {7}; a%10 in {7} -> b%10 in {8}; a%10 in {8} -> b%10 in {9}; a%10 in {9} -> b%10 in {0} | res: mod10 +3%, mod5 +3%, mod2 +2% | res: mod5 +3% | +0.92 |
| L23 up c17 | 28% / 20% | **res%100** (R2 0.89): res mod 100 in {59..85} | **tens(a,b)** (R2 0.59): a//10 in {0,4} -> b//10 in {7}; a//10 in {3} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {7,8}; a//10 in {6} -> b//10 in {0,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,9,10}; a//10 in {8} -> b//10 in {0,1,2}; a//10 in {9} -> b//10 in {1,2}; a//10 in {10} -> b//10 in {2,3} | (reads) | same | +0.92 |
| L29 up c71 | 19% / 7% | **res%10** (R2 0.84): res mod 10 in {4..5} | **res** (R2 0.59): res in {4, 14, 24, 34, 44, 54, 74, 84..85, 94..95} | (reads) | same | +0.92 |
| L29 up c293 | 10% / 14% | **res%10** (R2 0.95): res mod 10 in {0} | unexplained (best R2 0.50) | (reads) | same | +0.92 |
| L26 gate c4 | 20% / 12% | **res** (R2 0.92): res in {118..144} | **res//10** (R2 0.66): (tens) res in {21..38} | (reads) | same | +0.92 |
| L28 down c80 | 8% / 4% | **res%100** (R2 0.89): res mod 100 in {68..75} | unexplained (best R2 0.19) | - | same | +0.92 |
| L22 down c427 | 7% / 2% | **res%20** (R2 0.78): res mod 20 in {12} | unexplained (best R2 0.46) | res: mod4 +10% | - | +0.92 |
| L29 down c502 | 5% / 3% | **res//10** (R2 0.78): (tens) res in {40..49} | unexplained (best R2 0.37) | - | same | +0.92 |
| L30 gate c501 | 5% / 1% | **res%100** (R2 0.74): res mod 100 in {90..92} | unexplained (best R2 0.29) | (reads) | same | +0.92 |
| L28 down c312 | 8% / 1% | **res//10** (R2 0.80): (tens) res in {70..79, 171, 173, 175} | **tens(a,b)** (R2 0.58): a//10 in {7} -> b//10 in {0} | - | same | +0.92 |
| L29 gate c356 | 2% / 0 | **res%100** (R2 0.77): res mod 100 in {64} | off (on 0) | (reads) | same | +0.92 |
| L30 up c183 | 10% / 4% | **res%100** (R2 0.77): res mod 100 in {29..36, 82} | **res** (R2 0.61): res in {30..34, 82} | (reads) | same | +0.92 |
| L19 down c149 | 27% / 11% | **res%100** (R2 0.77): res mod 100 in {2..4, 12..14, 22..24, 32..34, 42..44, 52..53, 62..63, 72..73, 82..83, 92..93} [coarser: res mod 10 in {2..4}, R2 0.97] | **res** (R2 0.65): res in {2..3, 12..13, 22..24, 32..33, 42..43, 82..83, 92..93} | res: mod10 +7%, mod5 +3% | res: mod10 +3% | +0.92 |
| L25 gate c36 | 10% / 6% | **res%10** (R2 0.94): res mod 10 in {7} | **res** (R2 0.63): res in {-97, 7, 17, 27, 37, 47, 57, 67, 77, 87, 97} | (reads) | same | +0.92 |
| L30 down c181 | 8% / 2% | **res%100** (R2 0.59): res mod 100 in {16..20} | **res** (R2 0.51): res in {17..19} | - | same | +0.92 |
| L26 down c36 | 41% / 30% | **res%2** (R2 0.68): res mod 2 in {0} | **units(a,b)** (R2 0.67): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} | a: mod2 +5%; b: mod2 +3%; res: mod2 +8% | a: mod2 +8%; b: mod2 +7%; res: mod2 +10% | +0.92 |
| L24 up c108 | 15% / 9% | **res%50** (R2 0.80): res mod 50 in {6, 16, 25..27, 36, 46} | unexplained (best R2 0.45) | (reads) | same | +0.92 |
| L24 gate c16 | 23% / 12% | **res%100** (R2 0.77): res mod 100 in {16, 20..22, 24..27, 66..67, 70..77, 81..82} [coarser: res mod 50 in {16, 20..27}, R2 0.84] | unexplained (best R2 0.44) | (reads) | same | +0.92 |
| L23 up c16 | 33% / 28% | **res%100** (R2 0.69): res mod 100 in {1..11, 44..70} | unexplained (best R2 0.44) | (reads) | same | +0.92 |
| L24 gate c28 | 11% / 6% | **res%10** (R2 0.91): res mod 10 in {3} | **res** (R2 0.62): res in {3, 13, 23, 33, 43, 53, 73, 83, 93} | (reads) | same | +0.92 |
| L23 gate c55 | 23% / 25% | **units(a,b)** (R2 0.75): a%10 in {0} -> b%10 in {0,2,8}; a%10 in {1} -> b%10 in {1}; a%10 in {2,4,8} -> b%10 in {0,2,4,6,8}; a%10 in {3} -> b%10 in {9}; a%10 in {5} -> b%10 in {7}; a%10 in {6} -> b%10 in {0,2,6,8}; a%10 in {7} -> b%10 in {5}; a%10 in {9} -> b%10 in {3} | **units(a,b)** (R2 0.77): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8}; a%10 in {3} -> b%10 in {1} | (reads) | same | +0.92 |
| L29 down c595 | 5% / 1% | **res%100** (R2 0.67): res mod 100 in {87..89} | unexplained (best R2 0.29) | - | same | +0.92 |
| L21 down c35 | 14% / 3% | **res%100** (R2 0.85): res mod 100 in {57..70} | unexplained (best R2 0.33) | res: mod25 +10% | - | +0.92 |
| L26 gate c89 | 13% / 4% | **res%100** (R2 0.87): res mod 100 in {72..84} | unexplained (best R2 0.35) | (reads) | same | +0.92 |
| L28 down c593 | 2% / 0 | **res** (R2 0.81): res in {40..44} | off (on 0) | - | same | +0.92 |
| L25 gate c40 | 12% / 9% | **res%100** (R2 0.87): res mod 100 in {0..6, 95..99} | **res%100** (R2 0.65): res mod 100 in {0..5, 95..96} | (reads) | same | +0.92 |
| L26 down c12 | 9% / 4% | **res%100** (R2 0.89): res mod 100 in {57, 59..66} | unexplained (best R2 0.28) | res: mod25 +2% | - | +0.92 |
| L25 down c108 | 14% / 6% | **res%20** (R2 0.85): res mod 20 in {2, 12..13} [coarser: res mod 10 in {2}, R2 0.82] | **res** (R2 0.57): res in {12..13, 22, 32..33, 72, 92} | - | same | +0.92 |
| L22 up c56 | 31% / 20% | **res//10** (R2 0.80): (tens) res in {22..38, 111..145} [coarser: res mod 100 in {14..17, 19..41}, R2 0.81] | **res//10** (R2 0.65): (tens) res in {11..19, 21..38} | (reads) | same | +0.92 |
| L27 up c32 | 15% / 9% | **res%100** (R2 0.86): res mod 100 in {1..11, 99} | **res%100** (R2 0.72): res mod 100 in {3, 5..7, 10} | (reads) | same | +0.92 |
| L26 down c20 | 7% / 10% | **res%100** (R2 0.82): res mod 100 in {29..34} | unexplained (best R2 0.40) | - | same | +0.92 |
| L20 down c9 | 60% / 49% | **res%5** (R2 0.95): res mod 5 in {1..3} | **res** (R2 0.62): res in {-97, -93, -63, -58, -53..-52, -48..-47, -43..-42, -38..-37, -33..-32, -28..-27, -23..-22, -19..-17, -13..-12, -9..-7, -4..-2, 1..3, 6..8, 11..13, 16..18, 21..23, 26..28, 31..33, 36..38, 41..43, 46..48, 51..53, 56..58, 61..63, 66..68, 71..73, 76..78, 81..83, 86..88, 91..93, 96..98} | res: mod5 +25% | res: mod5 +22% | +0.92 |
| L21 up c16 | 31% / 18% | **res%10** (R2 0.89): res mod 10 in {6..8} | **res** (R2 0.62): res in {-23, -14..-13, -4..-2, 6..8, 16..18, 26..28, 36..38, 46..47, 56..58, 66..67, 76..78, 86..88, 96..98} | (reads) | same | +0.92 |
| L29 gate c64 | 14% / 7% | **res** (R2 0.88): res in {34..42, 129..143} [coarser: res mod 100 in {30, 33..43}, R2 0.86] | **res** (R2 0.51): res in {34..42} | (reads) | same | +0.92 |
| L27 gate c17 | 11% / 4% | **res//10** (R2 0.85): (tens) res in {79..89, 178..189} [coarser: res mod 100 in {79..89}, R2 0.99] | unexplained (best R2 0.37) | (reads) | same | +0.91 |
| L26 up c12 | 17% / 3% | **res%100** (R2 0.83): res mod 100 in {60..74, 77} | unexplained (best R2 0.38) | (reads) | same | +0.91 |
| L30 gate c295 | 11% / 4% | **res//10** (R2 0.84): (tens) res in {80..90, 180..187, 189} [coarser: res mod 100 in {80..90}, R2 0.95] | unexplained (best R2 0.41) | (reads) | same | +0.91 |
| L27 gate c745 | 9% / 4% | **res%100** (R2 0.85): res mod 100 in {1, 41..43, 62, 81..83} | **res** (R2 0.57): res in {1..2, 42, 81..83} | (reads) | same | +0.91 |
| L22 up c6 | 17% / 14% | **res%100** (R2 0.88): res mod 100 in {0..1, 4, 6..15} | **res%100** (R2 0.72): res mod 100 in {6..15} | (reads) | same | +0.91 |
| L29 up c376 | 3% / 4% | **res%100** (R2 0.82): res mod 100 in {25..26, 28} | unexplained (best R2 0.43) | (reads) | same | +0.91 |
| L23 down c77 | 6% / 4% | **res%100** (R2 0.69): res mod 100 in {1, 21, 41, 61, 81} | unexplained (best R2 0.43) | res: mod4 +4% | - | +0.91 |
| L28 down c435 | 1% / 2% | **res%100** (R2 0.89): res mod 100 in {8} | **res%100** (R2 0.51): res mod 100 in {8} | - | same | +0.91 |
| L19 gate c0 | 48% / 30% | **res%50** (R2 0.84): res mod 50 in {15..38} | **tens(a,b)** (R2 0.56): a//10 in {2} -> b//10 in {0,4,5,9,10}; a//10 in {3} -> b//10 in {0,1,5,6,7}; a//10 in {4} -> b//10 in {1,2,6,7}; a//10 in {5} -> b//10 in {2,3,7,8}; a//10 in {6} -> b//10 in {3,4,7,8}; a//10 in {7} -> b//10 in {4,5,9,10}; a//10 in {8} -> b//10 in {0,5,6,10}; a//10 in {9} -> b//10 in {6,7}; a//10 in {10} -> b//10 in {1,2,3,6,7,8} | (reads) | same | +0.91 |
| L23 down c205 | 10% / 4% | **res%10** (R2 0.96): res mod 10 in {8} | **res** (R2 0.61): res in {8, 18, 28, 38, 48, 88, 98} | - | same | +0.91 |
| L25 gate c174 | 6% / 2% | **res%100** (R2 0.85): res mod 100 in {18..19, 38, 58, 78, 98} | **res** (R2 0.59): res in {18..19} | (reads) | same | +0.91 |
| L29 up c64 | 15% / 5% | **res%100** (R2 0.77): res mod 100 in {18, 35..41, 48, 58, 87..89, 98} | unexplained (best R2 0.45) | (reads) | same | +0.91 |
| L28 up c71 | 14% / 3% | **res%100** (R2 0.85): res mod 100 in {26, 45..46, 60..69} | unexplained (best R2 0.25) | (reads) | same | +0.91 |
| L25 up c15 | 10% / 6% | **res%100** (R2 0.81): res mod 100 in {1..4, 23..24, 43, 63, 83} | **res%100** (R2 0.67): res mod 100: no class above 0.5 (max 0.46) | (reads) | same | +0.91 |
| L30 down c501 | 5% / 1% | **res%100** (R2 0.76): res mod 100 in {51, 61, 90..92} | unexplained (best R2 0.19) | - | same | +0.91 |
| L23 down c9 | 10% / 14% | **res%10** (R2 0.95): res mod 10 in {6} | **units(a,b)** (R2 0.53): a%10 in {0} -> b%10 in {4}; a%10 in {1} -> b%10 in {5}; a%10 in {2} -> b%10 in {6}; a%10 in {4} -> b%10 in {0,8}; a%10 in {5} -> b%10 in {9}; a%10 in {6} -> b%10 in {0}; a%10 in {7} -> b%10 in {1}; a%10 in {8} -> b%10 in {2,4}; a%10 in {9} -> b%10 in {3} | res: mod10 +3%, mod5 +5%, mod2 +2% | res: mod10 +3%, mod5 +5%, mod2 +2% | +0.91 |
| L29 down c448 | 3% / 4% | **res** (R2 0.62): res in {24..26, 45, 75, 125} [coarser: res mod 100 in {25..26, 75}, R2 0.81] | unexplained (best R2 0.46) | - | same | +0.91 |
| L27 up c82 | 13% / 2% | **res%100** (R2 0.86): res mod 100 in {49..61} | unexplained (best R2 0.33) | (reads) | same | +0.91 |
| L26 gate c12 | 11% / 5% | **res%100** (R2 0.92): res mod 100 in {56..66} | unexplained (best R2 0.28) | (reads) | same | +0.91 |
| L25 up c276 | 3% / 0 | **res%100** (R2 0.66): res mod 100 in {72..74} | off (on 0) | (reads) | same | +0.91 |
| L30 down c223 | 2% / 3% | **res** (R2 0.60): res in {39..41, 140} [coarser: res mod 100 in {40}, R2 0.80] | unexplained (best R2 0.27) | - | same | +0.91 |
| L30 gate c603 | 8% / 2% | **res%100** (R2 0.72): res mod 100 in {69..73, 80} | unexplained (best R2 0.22) | (reads) | same | +0.91 |
| L22 gate c98 | 7% / 0 | **res%100** (R2 0.80): res mod 100 in {83..89} | off (on 0) | (reads) | same | +0.91 |
| L25 gate c110 | 12% / 6% | **res%100** (R2 0.92): res mod 100 in {7, 16..18, 27, 37, 47, 57, 67, 77, 87, 97} [coarser: res mod 20 in {7, 17}, R2 0.83] | **res** (R2 0.65): res in {7, 16..18, 27, 37, 97} | (reads) | same | +0.91 |
| L28 gate c136 | 10% / 2% | **res//10** (R2 0.85): (tens) res in {50..60, 151..159} [coarser: res mod 100 in {51..59}, R2 0.99] | unexplained (best R2 0.26) | (reads) | same | +0.91 |
| L27 gate c723 | 7% / 1% | **res%100** (R2 0.87): res mod 100 in {80..85, 89} | unexplained (best R2 0.36) | (reads) | same | +0.91 |
| L30 gate c796 | 9% / 3% | **res%10** (R2 0.81): res mod 10 in {0} | unexplained (best R2 0.46) | (reads) | same | +0.91 |
| L24 gate c5 | 11% / 14% | **res%10** (R2 0.90): res mod 10 in {3} | unexplained (best R2 0.46) | (reads) | same | +0.91 |
| L30 gate c843 | 0 / 0 | off (on 0) | same | (reads) | same | +0.91 |
| L27 down c23 | 13% / 6% | **res** (R2 0.91): res in {14..17, 110..124} | **res** (R2 0.71): res in {11..17} | - | same | +0.91 |
| L20 up c3 | 49% / 60% | unexplained (best R2 0.33) | unexplained (best R2 0.22) | (reads) | same | +0.91 |
| L23 gate c439 | 5% / 1% | **res%100** (R2 0.67): res mod 100 in {38..39, 58..59} | unexplained (best R2 0.35) | (reads) | same | +0.91 |
| L30 down c357 | 4% / 0 | **res%100** (R2 0.73): res mod 100 in {85..87} | off (on 0) | - | same | +0.91 |
| L28 up c129 | 9% / 4% | **res%10** (R2 0.87): res mod 10 in {9} | **res** (R2 0.57): res in {9, 19, 29, 79, 89, 99} | (reads) | same | +0.91 |
| L29 down c103 | 0 / 2% | off (on 0) | **res%100** (R2 0.70): res mod 100: no class above 0.5 (max 0.46) | - | same | +0.91 |
| L28 down c24 | 12% / 15% | **res%100** (R2 0.81): res mod 100 in {6, 15..17, 26, 36, 46, 56, 66, 75..77, 86, 96} [coarser: res mod 20 in {6, 16}, R2 0.80] | unexplained (best R2 0.47) | res: mod4 +4% | res: mod4 +3% | +0.91 |
| L23 gate c3 | 19% / 27% | **res%100** (R2 0.81): res mod 100 in {0..7, 89..99} | **tens(a,b)** (R2 0.56): a//10 in {0} -> b//10 in {0,1,8,9,10}; a//10 in {1} -> b//10 in {1,10}; a//10 in {2} -> b//10 in {2}; a//10 in {4} -> b//10 in {4,9,10}; a//10 in {5} -> b//10 in {5,6,10}; a//10 in {6} -> b//10 in {6,7,8,9,10}; a//10 in {7} -> b//10 in {7,8,9,10}; a//10 in {8} -> b//10 in {8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,9,10}; a//10 in {10} -> b//10 in {0,9,10} | (reads) | same | +0.91 |
| L29 down c865 | 2% / 2% | **res%100** (R2 0.75): res mod 100 in {13, 63} [coarser: res mod 50 in {13}, R2 0.85] | **res** (R2 0.62): res in {13, 23} | - | same | +0.91 |
| L24 up c28 | 13% / 2% | **res%100** (R2 0.84): res mod 100 in {49..60} | unexplained (best R2 0.35) | (reads) | same | +0.91 |
| L22 down c11 | 33% / 25% | **res%20** (R2 0.89): res mod 20 in {0..3, 18..19} | **res** (R2 0.51): res in {-99..-98, -41..-38, -22..-18, 0..3, 18..24, 38..43, 58..62, 78..82, 97..99} | res: mod20 +11% | res: mod20 +10% | +0.91 |
| L28 gate c275 | 5% / 2% | **res%20** (R2 0.68): res mod 20 in {1} | **res** (R2 0.55): res in {1, 18, 41} | (reads) | same | +0.91 |
| L23 down c16 | 18% / 19% | **res%100** (R2 0.91): res mod 100 in {70..86} | **tens(a,b)** (R2 0.55): a//10 in {0,6} -> b//10 in {7,8}; a//10 in {4} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {7}; a//10 in {7} -> b//10 in {0,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,10}; a//10 in {9,10} -> b//10 in {1,2} | res: mod100 +3%, mod50 +3%, mod25 +5% | - | +0.91 |
| L31 gate c328 | 3% / 7% | **res//10** (R2 0.71): (tens) res in {19..29} | unexplained (best R2 0.36) | (reads) | same | +0.91 |
| L27 gate c110 | 9% / 2% | **res%100** (R2 0.87): res mod 100 in {76..83} | unexplained (best R2 0.23) | (reads) | same | +0.91 |
| L25 up c12 | 21% / 38% | **res%10** (R2 0.96): res mod 10 in {1, 7} | unexplained (best R2 0.49) | (reads) | same | +0.91 |
| L24 down c16 | 18% / 12% | **res%100** (R2 0.76): res mod 100 in {16..17, 21..22, 24..28, 70..72, 74..77, 81..82} [coarser: res mod 50 in {16, 20..22, 24..27}, R2 0.84] | **res** (R2 0.50): res in {15..17, 20..22, 24..27, 71..72} | res: mod25 +5% | res: mod25 +3% | +0.91 |
| L25 gate c535 | 9% / 3% | **res%10** (R2 0.88): res mod 10 in {9} | **res** (R2 0.62): res in {9, 19, 29, 39, 79, 99} | (reads) | same | +0.91 |
| L25 gate c63 | 9% / 4% | **res%10** (R2 0.82): res mod 10 in {0} | **res** (R2 0.59): res in {10, 20, 30, 90} | (reads) | same | +0.91 |
| L25 up c207 | 10% / 5% | **res%10** (R2 0.89): res mod 10 in {2} | **res** (R2 0.57): res in {2, 12..13, 22, 32, 72, 92} | (reads) | same | +0.91 |
| L23 up c77 | 12% / 8% | **res%20** (R2 0.70): res mod 20 in {1, 11} [coarser: res mod 10 in {1}, R2 0.81] | **res** (R2 0.54): res in {-19, 1..2, 11, 21..23, 41, 61, 81, 91} | (reads) | same | +0.91 |
| L23 down c4 | 11% / 16% | **res%10** (R2 0.91): res mod 10 in {2} | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {8}; a%10 in {1} -> b%10 in {9}; a%10 in {2} -> b%10 in {0,2,4,6,8}; a%10 in {3} -> b%10 in {1}; a%10 in {4} -> b%10 in {2}; a%10 in {5} -> b%10 in {3,7}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {5}; a%10 in {8} -> b%10 in {6}; a%10 in {9} -> b%10 in {7} | res: mod10 +6%, mod5 +6%, mod2 +3% | res: mod10 +5%, mod5 +8%, mod2 +3% | +0.91 |
| L30 up c883 | 1% / 0 | **res** (R2 0.55): res in {146, 156} | off (on 0) | (reads) | same | +0.91 |
| L27 up c461 | 3% / 3% | **res%100** (R2 0.80): res mod 100 in {35..36, 56} | unexplained (best R2 0.41) | (reads) | same | +0.91 |
| L27 gate c23 | 15% / 7% | **res%100** (R2 0.89): res mod 100 in {9..19} | **res** (R2 0.80): res in {10..17} | (reads) | same | +0.91 |
| L29 down c64 | 10% / 6% | **res%100** (R2 0.72): res mod 100 in {35..41, 58, 88} | unexplained (best R2 0.47) | - | same | +0.91 |
| L29 down c653 | 3% / 0 | **res%100** (R2 0.77): res mod 100 in {70..71} | off (on 0) | - | same | +0.91 |
| L29 down c145 | 8% / 2% | **res%20** (R2 0.78): res mod 20 in {6, 16} [coarser: res mod 10 in {6}, R2 0.82] | unexplained (best R2 0.29) | - | same | +0.91 |
| L21 gate c15 | 38% / 19% | **res%20** (R2 0.83): res mod 20 in {2..8} | **res** (R2 0.55): res in {2..9, 22..28, 44..45, 84..88} | (reads) | same | +0.91 |
| L29 down c420 | 1% / 0 | **res%100** (R2 0.61): res mod 100 in {90} | off (on 0) | - | same | +0.91 |
| L24 up c16 | 25% / 16% | **res%100** (R2 0.82): res mod 100 in {10..13, 20..23, 31..32, 41..42, 50..52, 70..73, 81..82, 90..92} | **res** (R2 0.56): res in {-19, 1..2, 10..13, 20..23, 31..32, 71..72, 82} | (reads) | same | +0.91 |
| L21 up c10 | 11% / 16% | **res%10** (R2 0.91): res mod 10 in {1} | unexplained (best R2 0.49) | (reads) | same | +0.91 |
| L29 gate c828 | 3% / 0 | **res** (R2 0.76): res in {41, 131, 137..138, 141} | off (on 0) | (reads) | same | +0.91 |
| L22 up c121 | 29% / 21% | **res%10** (R2 0.88): res mod 10 in {0..1, 9} | **res** (R2 0.52): res in {-99, -40..-39, -31..-30, -21..-19, -10, -1..2, 9..10, 19..21, 29..30, 39..41, 50, 59..61, 79..80, 99} | (reads) | same | +0.91 |
| L21 gate c2 | 40% / 23% | **res%20** (R2 0.86): res mod 20 in {10..17} | **res** (R2 0.53): res in {-9..-4, 10..17, 31..36, 51..56, 71..75, 91..96} | (reads) | same | +0.91 |
| L28 gate c794 | 12% / 4% | **res** (R2 0.84): res in {18, 22..24, 29..30, 118, 121..131, 133..135} | **res** (R2 0.54): res in {18, 21..24, 29..30} | (reads) | same | +0.91 |
| L27 gate c222 | 14% / 4% | **res%100** (R2 0.82): res mod 100 in {34..46} | unexplained (best R2 0.46) | (reads) | same | +0.91 |
| L29 gate c739 | 10% / 4% | **res%10** (R2 0.97): res mod 10 in {4} | **res** (R2 0.68): res in {4, 14, 24, 34, 84, 94} | (reads) | same | +0.91 |
| L27 down c480 | 4% / 1% | **res%100** (R2 0.81): res mod 100 in {86..89} | unexplained (best R2 0.37) | - | same | +0.91 |
| L21 down c7 | 42% / 32% | **res%50** (R2 0.81): res mod 50 in {0..12, 42..49} | **res%100** (R2 0.49): res mod 100 in {0..13, 45, 47, 49..53, 55, 90..99} | res: mod100 +2%, mod50 +10%, mod25 +21% | res: mod100 +2%, mod50 +10%, mod25 +4% | +0.90 |
| L31 gate c72 | 16% / 6% | **res%100** (R2 0.71): res mod 100 in {58..59, 68..80} | unexplained (best R2 0.43) | (reads) | same | +0.90 |
| L26 up c178 | 14% / 4% | **res%100** (R2 0.88): res mod 100 in {27..29, 47..49, 57..59, 77..79, 88..89} | unexplained (best R2 0.46) | (reads) | same | +0.90 |
| L28 gate c83 | 9% / 4% | **res%10** (R2 0.89): res mod 10 in {8} | **res** (R2 0.63): res in {8, 18, 28, 88, 98} | (reads) | same | +0.90 |
| L27 down c110 | 10% / 4% | **res%100** (R2 0.71): res mod 100 in {19, 29, 39, 58..59, 76..79, 89, 99} | unexplained (best R2 0.38) | - | same | +0.90 |
| L29 down c94 | 1% / 2% | unexplained (best R2 0.44) | **res%100** (R2 0.55): res mod 100 in {7} | - | same | +0.90 |
| L24 down c263 | 5% / 2% | **res%100** (R2 0.70): res mod 100: no class above 0.5 (max 0.44) | unexplained (best R2 0.35) | - | same | +0.90 |
| L28 down c36 | 13% / 13% | **res%10** (R2 0.75): res mod 10 in {2} | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {8}; a%10 in {2} -> b%10 in {0,2}; a%10 in {3} -> b%10 in {1}; a%10 in {4} -> b%10 in {2}; a%10 in {5} -> b%10 in {3}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {5}; a%10 in {8} -> b%10 in {6}; a%10 in {9} -> b%10 in {7} | res: mod10 +3%, mod5 +3% | res: mod5 +2% | +0.90 |
| L29 up c153 | 5% / 1% | **res%100** (R2 0.76): res mod 100 in {67..68, 88} | unexplained (best R2 0.23) | (reads) | same | +0.90 |
| L29 gate c653 | 12% / 8% | **res//10** (R2 0.83): (tens) res in {69..79, 169..179} [coarser: res mod 100 in {69..79}, R2 1.00] | unexplained (best R2 0.48) | (reads) | same | +0.90 |
| L29 down c279 | 3% / 6% | **res%100** (R2 0.63): res mod 100 in {0, 19..22} | **res** (R2 0.51): res in {-20, 19..23} | - | same | +0.90 |
| L19 down c6 | 66% / 66% | unexplained (best R2 0.26) | unexplained (best R2 0.21) | a: mod20 -3%; b: mod20 +14%, mod4 +8%; res: mod20 +47% | b: mod20 +11%, mod4 +4%; res: mod20 +33% | +0.90 |
| L21 gate c53 | 49% / 29% | **res%2** (R2 0.83): res mod 2 in {0} | **res** (R2 0.57): res in {-98, -96, -94, -92, -90, -88, -86, -84, -82, -80, -78, -76, -74, -64, -4, -2, 0, 2, 4, 6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 32, 34, 36, 38, 40, 42, 44, 46, 48, 54, 56, 58, 60, 62, 64, 66, 68, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98} [coarser: res mod 100 in {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 34, 36, 38, 40, 42, 44, 46, 76, 78, 82, 84, 86, 88, 90, 92, 94, 96, 98}, R2 0.83] | (reads) | same | +0.90 |
| L26 gate c83 | 3% / 2% | **res%100** (R2 0.72): res mod 100 in {9} | **res%100** (R2 0.81): res mod 100: no class above 0.5 (max 0.48) | (reads) | same | +0.90 |
| L26 gate c8 | 11% / 8% | **res%100** (R2 0.82): res mod 100 in {19..23, 47..48, 86..89} | **res** (R2 0.52): res in {19..23, 86..89} | (reads) | same | +0.90 |
| L25 gate c683 | 6% / 2% | **res%100** (R2 0.87): res mod 100 in {31, 41, 51, 61, 71, 81, 91} | unexplained (best R2 0.38) | (reads) | same | +0.90 |
| L21 down c63 | 8% / 12% | **res%100** (R2 0.70): res mod 100 in {23..30} | unexplained (best R2 0.49) | res: mod25 +2% | - | +0.90 |
| L28 down c67 | 13% / 7% | **res%100** (R2 0.85): res mod 100 in {7, 17, 27, 37, 47, 57, 67, 75..79, 87, 97} [coarser: res mod 50 in {7, 17, 27..28, 37, 47}, R2 0.88] | **res** (R2 0.66): res in {-23, -13, -3, 7, 17, 27, 37, 47, 57, 67, 77, 87, 97} | - | same | +0.90 |
| L31 down c321 | 4% / 1% | **res** (R2 0.66): res in {80..84} | unexplained (best R2 0.34) | - | same | +0.90 |
| L27 gate c224 | 13% / 5% | **res%100** (R2 0.52): res mod 100 in {0, 15, 20, 25, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 95} [coarser: res mod 10 in {0, 5}, R2 0.80] | **units(a,b)** (R2 0.74): a%10 in {0,5} -> b%10 in {0,5} | (reads) | same | +0.90 |
| L25 up c40 | 25% / 13% | **res%50** (R2 0.84): res mod 50 in {0..1, 9..11, 19..20, 29..30, 39..40, 49} [coarser: res mod 10 in {0..1, 9}, R2 0.85] | **res%100** (R2 0.47): res mod 100 in {0..1, 9..11, 19, 89..90, 99} | (reads) | same | +0.90 |
| L31 down c250 | 2% / 0 | **res** (R2 0.77): res in {46..49} | off (on 0) | - | same | +0.90 |
| L27 up c312 | 10% / 3% | **res%10** (R2 0.95): res mod 10 in {9} | **res** (R2 0.59): res in {9, 19, 29, 79, 89} | (reads) | same | +0.90 |
| L29 up c502 | 6% / 2% | **res%100** (R2 0.73): res mod 100 in {10..12} | **res%100** (R2 0.69): res mod 100 in {11} | (reads) | same | +0.90 |
| L24 down c37 | 9% / 2% | **res%100** (R2 0.69): res mod 100 in {47, 77, 85..89} | unexplained (best R2 0.30) | - | same | +0.90 |
| L25 down c48 | 16% / 12% | **res%100** (R2 0.86): res mod 100 in {0..1, 8..11, 13..14, 88..91, 93..94, 98..99} | **res%100** (R2 0.49): res mod 100 in {1, 8..11, 13..14, 89..91, 98..99} | - | same | +0.90 |
| L27 up c157 | 4% / 0 | **res%100** (R2 0.65): res mod 100 in {90, 92} | off (on 0) | (reads) | same | +0.90 |
| L24 gate c736 | 10% / 3% | **res%10** (R2 0.96): res mod 10 in {5} | **res** (R2 0.64): res in {5, 15, 25, 85, 95} | (reads) | same | +0.90 |
| L19 up c346 | 23% / 14% | unexplained (best R2 0.30) | unexplained (best R2 0.19) | (reads) | same | +0.90 |
| L30 down c199 | 8% / 1% | **res%100** (R2 0.76): res mod 100 in {40..47} | unexplained (best R2 0.25) | - | same | +0.90 |
| L31 down c184 | 3% / 1% | **res** (R2 0.70): res in {66..68} [coarser: res mod 100 in {66..68}, R2 0.84] | unexplained (best R2 0.19) | - | same | +0.90 |
| L20 up c15 | 59% / 39% | **res%5** (R2 0.86): res mod 5 in {2..4} | unexplained (best R2 0.45) | (reads) | same | +0.90 |
| L28 down c136 | 6% / 1% | **res** (R2 0.75): res in {54, 56..57, 151..158} [coarser: res mod 100 in {53..57}, R2 0.85] | unexplained (best R2 0.18) | - | same | +0.90 |
| L27 gate c157 | 2% / 1% | **res%100** (R2 0.76): res mod 100 in {1, 98..99} | **res%100** (R2 0.73): res mod 100 in {1, 99} | (reads) | same | +0.90 |
| L29 gate c92 | 11% / 3% | **res//10** (R2 0.79): (tens) res in {51..60, 63, 150..159} [coarser: res mod 100 in {50..60}, R2 0.99] | unexplained (best R2 0.32) | (reads) | same | +0.90 |
| L30 gate c831 | 6% / 1% | **res%100** (R2 0.73): res mod 100 in {57..60, 98} | unexplained (best R2 0.16) | (reads) | same | +0.90 |
| L29 gate c230 | 10% / 3% | **res%20** (R2 0.91): res mod 20 in {12..13} | **res** (R2 0.64): res in {12..13, 33, 93} | (reads) | same | +0.90 |
| L28 gate c416 | 0 / 1% | off (on 0) | **res** (R2 0.75): res in {10} | (reads) | same | +0.90 |
| L23 up c15 | 18% / 18% | **res** (R2 0.75): res in {19..21, 35..45, 47..50, 59..61, 79..81, 136..143} | unexplained (best R2 0.36) | (reads) | same | +0.90 |
| L30 down c1012 | 2% / 0 | **res%100** (R2 0.80): res mod 100 in {79..80} | off (on 0) | - | same | +0.90 |
| L25 up c297 | 20% / 3% | **res%100** (R2 0.75): res mod 100 in {1, 81, 83..98} | unexplained (best R2 0.40) | (reads) | same | +0.90 |
| L28 up c80 | 13% / 4% | **res%100** (R2 0.93): res mod 100 in {67..79} | unexplained (best R2 0.31) | (reads) | same | +0.90 |
| L28 gate c660 | 10% / 5% | **res%20** (R2 0.78): res mod 20 in {1..2} | **res** (R2 0.58): res in {1..2, 21..22, 81} | (reads) | same | +0.90 |
| L29 down c477 | 2% / 1% | **res%100** (R2 0.63): res mod 100 in {16, 96} | **res** (R2 0.60): res in {16, 96} | - | same | +0.90 |
| L25 gate c314 | 6% / 3% | **res%100** (R2 0.89): res mod 100 in {7, 17, 27, 37, 47, 97} | **res** (R2 0.77): res in {7, 17, 27, 97} | (reads) | same | +0.90 |
| L24 down c501 | 0 / 3% | off (on 0) | **res%100** (R2 0.81): res mod 100: no class above 0.5 (max 0.46) | - | same | +0.90 |
| L24 gate c627 | 19% / 6% | **res%100** (R2 0.79): res mod 100 in {9..10, 26..27, 29..30, 46..51, 66..70} | unexplained (best R2 0.44) | (reads) | same | +0.90 |
| L30 gate c326 | 7% / 1% | **res%100** (R2 0.59): res mod 100 in {54..59, 96} | unexplained (best R2 0.15) | (reads) | same | +0.90 |
| L27 down c818 | 2% / 0 | **res%100** (R2 0.64): res mod 100 in {61..62} | off (on 0) | - | same | +0.90 |
| L27 gate c38 | 10% / 5% | **res%10** (R2 0.96): res mod 10 in {9} | **res** (R2 0.64): res in {-99, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99} | (reads) | same | +0.90 |
| L21 down c57 | 18% / 17% | **res** (R2 0.76): res in {65..71, 106..119} | **res** (R2 0.55): res in {6..21} | res: mod25 +3% | res: mod50 +2% | +0.90 |
| L24 down c4 | 11% / 12% | **res%10** (R2 0.86): res mod 10 in {8} | unexplained (best R2 0.50) | res: mod10 +7%, mod5 +8%, mod2 +4% | res: mod10 +5%, mod5 +6% | +0.90 |
| L29 gate c444 | 13% / 2% | **res%100** (R2 0.88): res mod 100 in {89..98} | unexplained (best R2 0.49) | (reads) | same | +0.90 |
| L20 up c485 | 26% / 15% | unexplained (best R2 0.44) | unexplained (best R2 0.42) | (reads) | same | +0.90 |
| L27 up c843 | 1% / 0 | **res%100** (R2 0.75): res mod 100 in {44, 84} | off (on 0) | (reads) | same | +0.90 |
| L29 gate c823 | 2% / 0 | **res%100** (R2 0.75): res mod 100 in {64} | off (on 0) | (reads) | same | +0.90 |
| L25 down c40 | 9% / 3% | **res%100** (R2 0.80): res mod 100 in {0..1, 95..99} | **res%100** (R2 0.55): res mod 100 in {0..1, 99} | - | same | +0.90 |
| L21 down c10 | 11% / 18% | **res%10** (R2 0.93): res mod 10 in {1} | unexplained (best R2 0.50) | res: mod10 +5%, mod5 +6% | res: mod10 +8%, mod5 +10%, mod2 +3% | +0.90 |
| L25 gate c46 | 11% / 6% | **res%10** (R2 0.86): res mod 10 in {9} | **res** (R2 0.55): res in {-99, 9, 18..19, 29, 39, 79, 89, 99} | (reads) | same | +0.90 |
| L23 down c17 | 15% / 11% | **res%100** (R2 0.86): res mod 100 in {64..76} | unexplained (best R2 0.40) | res: mod100 +2%, mod50 +3%, mod25 +4% | - | +0.90 |
| L27 down c117 | 7% / 3% | **res** (R2 0.84): res in {79..86} | unexplained (best R2 0.46) | - | same | +0.90 |
| L23 up c24 | 20% / 12% | **res%10** (R2 0.78): res mod 10 in {7..8} | **res** (R2 0.64): res in {-97, -23, -13, -3, 7..8, 17..18, 27..28, 37..39, 47, 57, 67, 77, 87..88, 97..99} | (reads) | same | +0.90 |
| L25 gate c151 | 6% / 4% | **res%100** (R2 0.92): res mod 100 in {0..5} | **res%100** (R2 0.73): res mod 100 in {2} | (reads) | same | +0.90 |
| L30 down c249 | 8% / 1% | **res%100** (R2 0.71): res mod 100 in {33, 60, 62..65, 83} | unexplained (best R2 0.22) | - | same | +0.90 |
| L29 gate c668 | 9% / 3% | **res%10** (R2 0.86): res mod 10 in {8} | **res** (R2 0.51): res in {8, 28, 88} | (reads) | same | +0.90 |
| L24 gate c87 | 8% / 0 | **res%100** (R2 0.85): res mod 100 in {70..76} | off (on 0) | (reads) | same | +0.90 |
| L30 gate c519 | 8% / 2% | **res%100** (R2 0.76): res mod 100 in {13..14, 24, 34, 54, 74} | **res** (R2 0.50): res in {14, 34} | (reads) | same | +0.90 |
| L27 down c17 | 8% / 4% | **res%100** (R2 0.91): res mod 100 in {79..85} | unexplained (best R2 0.33) | res: mod25 +3% | - | +0.90 |
| L19 down c0 | 51% / 60% | **res%50** (R2 0.84): res mod 50 in {14..39} | **tens(a,b)** (R2 0.54): a//10 in {0,5} -> b//10 in {2,3,7,8}; a//10 in {1} -> b//10 in {3,4,7,8,9}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {1,2,6,7,8}; a//10 in {6} -> b//10 in {2,3,4,7,8}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,10}; a//10 in {9} -> b//10 in {1,2,3,6,7}; a//10 in {10} -> b//10 in {1,2,3,7,8} | a: mod50 +2%; res: mod50 +26%, mod25 +64% | res: mod50 +18%, mod25 +4% | +0.90 |
| L23 up c465 | 18% / 6% | **res%100** (R2 0.85): res mod 100 in {73..90} | unexplained (best R2 0.41) | (reads) | same | +0.90 |
| L25 up c42 | 19% / 8% | **res%100** (R2 0.83): res mod 100 in {10..14, 33, 53, 61..63, 70..73, 90..94} | **res** (R2 0.58): res in {10..14, 72..73, 90..94} | (reads) | same | +0.90 |
| L23 up c35 | 10% / 4% | **res%10** (R2 0.94): res mod 10 in {6} | **res** (R2 0.57): res in {6, 16, 26, 36, 86, 96} | (reads) | same | +0.90 |
| L21 gate c13 | 15% / 5% | **res%10** (R2 0.71): res mod 10 in {4} | **res** (R2 0.62): res in {4, 14, 24, 34, 44, 74, 84, 94} | (reads) | same | +0.89 |
| L21 gate c99 | 44% / 20% | **res%100** (R2 0.69): res mod 100 in {0..1, 30..58, 82..83, 89..99} | unexplained (best R2 0.48) | (reads) | same | +0.89 |
| L31 gate c95 | 19% / 9% | **res//10** (R2 0.72): (tens) res in {71..89, 171, 173, 177..178, 183, 188} [coarser: res mod 100 in {71..89}, R2 0.83] | **tens(a,b)** (R2 0.53): a//10 in {0} -> b//10 in {7,8}; a//10 in {7} -> b//10 in {0}; a//10 in {8} -> b//10 in {0,1,10}; a//10 in {9} -> b//10 in {1,2,10}; a//10 in {10} -> b//10 in {1,2} | (reads) | same | +0.89 |
| L21 gate c279 | 37% / 23% | **res%20** (R2 0.78): res mod 20 in {13..19} | **res** (R2 0.53): res in {-97, -23, -5..-3, 13..19, 33..39, 53..59, 74..79, 93..99} | (reads) | same | +0.89 |
| L19 gate c20 | 50% / 23% | **res%50** (R2 0.79): res mod 50 in {4..26} | **tens(a,b)** (R2 0.57): a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4,8}; a//10 in {6} -> b//10 in {0,4,5,9,10}; a//10 in {7} -> b//10 in {0,5,6}; a//10 in {8} -> b//10 in {6,7}; a//10 in {9} -> b//10 in {7,8}; a//10 in {10} -> b//10 in {3,4,8,9} | (reads) | same | +0.89 |
| L30 down c961 | 5% / 1% | **res%100** (R2 0.80): res mod 100 in {50..54} | unexplained (best R2 0.26) | - | same | +0.89 |
| L19 up c0 | 53% / 52% | **res%50** (R2 0.77): res mod 50 in {14..38} | **tens(a,b)** (R2 0.53): a//10 in {0} -> b//10 in {2,7}; a//10 in {1} -> b//10 in {2,3}; a//10 in {2} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {3,8} -> b//10 in {0,1,2,5,6,7,10}; a//10 in {4} -> b//10 in {1,2,3,6,7,8}; a//10 in {5} -> b//10 in {2,3,7,8}; a//10 in {6} -> b//10 in {2,3,4,7,8}; a//10 in {7} -> b//10 in {0,1,2,4,5,6,8,9,10}; a//10 in {9} -> b//10 in {2,6,7}; a//10 in {10} -> b//10 in {1,2,3,7,8} | (reads) | same | +0.89 |
| L23 down c196 | 1% / 8% | unexplained (best R2 0.28) | **res%100** (R2 0.76): res mod 100: no class above 0.5 (max 0.46) | - | same | +0.89 |
| L28 gate c140 | 10% / 4% | **res%10** (R2 0.95): res mod 10 in {4} | **res** (R2 0.60): res in {4, 14, 24, 34, 84, 94} | (reads) | same | +0.89 |
| L28 up c359 | 3% / 0 | **res%100** (R2 0.72): res mod 100 in {96..97} | off (on 0) | (reads) | same | +0.89 |
| L24 down c36 | 9% / 3% | **res%100** (R2 0.94): res mod 100 in {70..78} | unexplained (best R2 0.26) | res: mod25 +3% | - | +0.89 |
| L29 down c376 | 2% / 2% | **res%100** (R2 0.65): res mod 100 in {26} | unexplained (best R2 0.30) | - | same | +0.89 |
| L25 down c586 | 3% / 1% | **res%100** (R2 0.77): res mod 100 in {33, 53, 73} | unexplained (best R2 0.26) | - | same | +0.89 |
| L28 gate c359 | 12% / 11% | **units(a,b)** (R2 0.97): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {9}; a%10 in {2} -> b%10 in {8}; a%10 in {3} -> b%10 in {7}; a%10 in {4} -> b%10 in {6}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {3}; a%10 in {8} -> b%10 in {2}; a%10 in {9} -> b%10 in {1} | **units(a,b)** (R2 0.62): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {1}; a%10 in {3} -> b%10 in {3}; a%10 in {4} -> b%10 in {4}; a%10 in {6} -> b%10 in {6}; a%10 in {8} -> b%10 in {8}; a%10 in {9} -> b%10 in {9} | (reads) | same | +0.89 |
| L20 gate c20 | 48% / 31% | **res//10** (R2 0.76): (tens) res in {46..88, 141..188} [coarser: res mod 100 in {45..88}, R2 0.96] | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {7}; a//10 in {2} -> b//10 in {5,6,7,8}; a//10 in {3,4} -> b//10 in {7,8}; a//10 in {5} -> b//10 in {0,7,8,9,10}; a//10 in {6} -> b//10 in {0,1,2,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,3,10}; a//10 in {8} -> b//10 in {0,1,2,3}; a//10 in {9} -> b//10 in {1,2,3,4}; a//10 in {10} -> b//10 in {2,3,4,5} | (reads) | same | +0.89 |
| L30 down c133 | 9% / 5% | **res%50** (R2 0.88): res mod 50 in {8, 18, 28, 48} | **res** (R2 0.61): res in {-2, 8, 18, 28, 38, 48, 98} | - | same | +0.89 |
| L20 down c21 | 42% / 37% | **res%10** (R2 0.89): res mod 10 in {0, 7..9} | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {0,1,2,3,9}; a%10 in {1} -> b%10 in {0,1,2,3}; a%10 in {2} -> b%10 in {1,2,3,4}; a%10 in {3} -> b%10 in {3,4,5,6}; a%10 in {4} -> b%10 in {4,5,6}; a%10 in {5} -> b%10 in {6,7}; a%10 in {6} -> b%10 in {6,7,8}; a%10 in {7} -> b%10 in {0,7,8,9}; a%10 in {8} -> b%10 in {0,1,8,9}; a%10 in {9} -> b%10 in {0,1,2,8,9} | res: mod10 +17%, mod5 +3% | a: mod10 +2%; b: mod10 +2%; res: mod10 +15%, mod5 +2% | +0.89 |
| L29 down c136 | 13% / 7% | **res%100** (R2 0.76): res mod 100 in {1, 11, 21, 31, 39..43, 45, 51, 61, 71, 81, 91} [coarser: res mod 50 in {1, 11, 21, 31, 41}, R2 0.82] | unexplained (best R2 0.40) | - | same | +0.89 |
| L20 up c653 | 32% / 16% | **res%20** (R2 0.54): res mod 20 in {0, 14..19} | unexplained (best R2 0.34) | (reads) | same | +0.89 |
| L26 gate c88 | 5% / 0 | **res%100** (R2 0.77): res mod 100 in {45..48, 86, 88} | off (on 0) | (reads) | same | +0.89 |
| L28 down c291 | 6% / 2% | **res%100** (R2 0.61): res mod 100 in {17, 57, 67, 77} | unexplained (best R2 0.40) | - | same | +0.89 |
| L30 gate c551 | 4% / 3% | **res%100** (R2 0.73): res mod 100 in {23..25, 74, 84} | **res** (R2 0.70): res in {23..25} | (reads) | same | +0.89 |
| L24 gate c482 | 9% / 3% | **res%10** (R2 0.88): res mod 10 in {4} | **res** (R2 0.56): res in {4, 24, 34, 84} | (reads) | same | +0.89 |
| L24 up c191 | 11% / 6% | **res%100** (R2 0.88): res mod 100 in {75..85} | unexplained (best R2 0.38) | (reads) | same | +0.89 |
| L29 up c90 | 11% / 2% | **res%100** (R2 0.74): res mod 100 in {64..70} | unexplained (best R2 0.24) | (reads) | same | +0.89 |
| L28 gate c543 | 9% / 1% | **res%10** (R2 0.83): res mod 10 in {6} | unexplained (best R2 0.14) | (reads) | same | +0.89 |
| L25 down c297 | 8% / 3% | **res%100** (R2 0.84): res mod 100 in {82..89} | unexplained (best R2 0.41) | - | same | +0.89 |
| L28 up c101 | 11% / 5% | **units(a,b)** (R2 0.53): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {4}; a%10 in {9} -> b%10 in {6} | **units(a,b)** (R2 0.70): a%10 in {0,5} -> b%10 in {0,5} | (reads) | same | +0.89 |
| L24 up c210 | 18% / 10% | **res%100** (R2 0.79): res mod 100 in {16..19, 56..59, 75..79, 95..98} | unexplained (best R2 0.47) | (reads) | same | +0.89 |
| L23 down c5 | 13% / 21% | **res%100** (R2 0.79): res mod 100 in {32..43} | unexplained (best R2 0.46) | res: mod100 +2%, mod25 +6% | res: mod50 +2%, mod25 +3% | +0.89 |
| L21 down c68 | 20% / 10% | **res%10** (R2 0.96): res mod 10 in {0, 8} | **res** (R2 0.52): res in {0, 8, 10, 18, 20, 28, 88, 98} | - | same | +0.89 |
| L31 down c239 | 4% / 2% | **res%100** (R2 0.73): res mod 100 in {61, 81, 91} | unexplained (best R2 0.20) | - | same | +0.89 |
| L29 up c145 | 9% / 2% | **res%10** (R2 0.83): res mod 10 in {6} | unexplained (best R2 0.47) | (reads) | same | +0.89 |
| L19 down c24 | 44% / 19% | **res//10** (R2 0.77): (tens) res in {24, 26, 28..80, 142..174} [coarser: res mod 100 in {34..76}, R2 0.81] | **tens(a,b)** (R2 0.57): a//10 in {2} -> b//10 in {6,7,8}; a//10 in {3} -> b//10 in {7,8}; a//10 in {4} -> b//10 in {7,8,9}; a//10 in {5} -> b//10 in {0,5,8,9,10}; a//10 in {6} -> b//10 in {0,9,10}; a//10 in {8,9} -> b//10 in {3,4}; a//10 in {10} -> b//10 in {3,4,5,6} | res: mod100 +17% | - | +0.89 |
| L29 down c566 | 1% / 0 | **res%100** (R2 0.64): res mod 100 in {35} | off (on 0) | - | same | +0.89 |
| L29 gate c871 | 3% / 2% | **res%100** (R2 0.77): res mod 100 in {7, 17, 67} | **res%100** (R2 0.56): res mod 100 in {7} | (reads) | same | +0.89 |
| L30 down c205 | 11% / 2% | **res** (R2 0.80): res in {43..49, 140..151} [coarser: res mod 100 in {42..50}, R2 0.86] | unexplained (best R2 0.45) | - | same | +0.89 |
| L27 up c139 | 9% / 4% | **res%10** (R2 0.90): res mod 10 in {1} | **res** (R2 0.64): res in {1, 11, 21, 31} | (reads) | same | +0.89 |
| L28 down c719 | 4% / 1% | **res%100** (R2 0.81): res mod 100 in {32, 52, 72, 92} | unexplained (best R2 0.33) | - | same | +0.89 |
| L25 gate c31 | 16% / 12% | **res%100** (R2 0.93): res mod 100 in {0..15} | **res%100** (R2 0.77): res mod 100 in {8..14} | (reads) | same | +0.89 |
| L25 gate c670 | 9% / 4% | **res%10** (R2 0.94): res mod 10 in {7} | **res** (R2 0.64): res in {7, 17, 27, 37, 87, 97} | (reads) | same | +0.89 |
| L24 down c70 | 9% / 18% | **res** (R2 0.86): res in {35..49, 51..55} | **tens(a,b)** (R2 0.56): a//10 in {0} -> b//10 in {4,5}; a//10 in {3} -> b//10 in {0,4,5}; a//10 in {4} -> b//10 in {0,1,4,5,10}; a//10 in {5} -> b//10 in {0,1,2,6}; a//10 in {6} -> b//10 in {1,2}; a//10 in {7} -> b//10 in {3}; a//10 in {10} -> b//10 in {5,6} | - | same | +0.89 |
| L29 down c438 | 1% / 1% | **res%100** (R2 0.61): res mod 100 in {62} | unexplained (best R2 0.09) | - | same | +0.89 |
| L29 up c263 | 5% / 6% | **res%100** (R2 0.84): res mod 100 in {13..17} | **res** (R2 0.79): res in {9, 11..17} | (reads) | same | +0.89 |
| L27 up c791 | 5% / 10% | **res%100** (R2 0.78): res mod 100 in {4..10} | **res%100** (R2 0.66): res mod 100: no class above 0.5 (max 0.46) | (reads) | same | +0.89 |
| L29 down c130 | 12% / 4% | **res%100** (R2 0.81): res mod 100 in {4, 14, 24, 34, 44, 54, 63..65, 74, 84, 94} [coarser: res mod 10 in {4}, R2 0.81] | **res** (R2 0.65): res in {4, 14, 24, 34, 44, 64, 84, 94} | - | same | +0.89 |
| L23 gate c9 | 10% / 14% | **res%10** (R2 0.98): res mod 10 in {6} | **units(a,b)** (R2 0.55): a%10 in {0} -> b%10 in {4}; a%10 in {1} -> b%10 in {5}; a%10 in {2} -> b%10 in {6}; a%10 in {4} -> b%10 in {0,4,8}; a%10 in {5} -> b%10 in {9}; a%10 in {6} -> b%10 in {0}; a%10 in {7} -> b%10 in {1}; a%10 in {8} -> b%10 in {2,4}; a%10 in {9} -> b%10 in {3,5} | (reads) | same | +0.89 |
| L26 gate c320 | 10% / 2% | **res%100** (R2 0.78): res mod 100 in {0..1, 91..99} | unexplained (best R2 0.22) | (reads) | same | +0.89 |
| L24 up c37 | 9% / 2% | **res%100** (R2 0.81): res mod 100 in {26..27, 47, 67, 85..88} | unexplained (best R2 0.40) | (reads) | same | +0.89 |
| L26 gate c360 | 2% / 0 | unexplained (best R2 0.49) | off (on 0) | (reads) | same | +0.89 |
| L20 up c767 | 16% / 9% | unexplained (best R2 0.38) | unexplained (best R2 0.31) | (reads) | same | +0.89 |
| L30 up c359 | 9% / 4% | **res%100** (R2 0.83): res mod 100 in {23..26, 44, 64, 74, 84} | **res** (R2 0.67): res in {23..26} | (reads) | same | +0.89 |
| L29 down c193 | 9% / 1% | **res%20** (R2 0.83): res mod 20 in {6, 8} | unexplained (best R2 0.19) | - | same | +0.88 |
| L22 up c276 | 28% / 11% | **res%10** (R2 0.83): res mod 10 in {7..9} | **res** (R2 0.53): res in {-99, 8..9, 18..19, 27..29, 39, 49, 59, 67..69, 78..79, 87..89, 97..99} | (reads) | same | +0.88 |
| L30 up c394 | 8% / 1% | **res%100** (R2 0.73): res mod 100 in {27, 37, 57, 67, 87, 97} | unexplained (best R2 0.36) | (reads) | same | +0.88 |
| L27 down c188 | 2% / 4% | unexplained (best R2 0.37) | **units(a,b)** (R2 0.87): a%10 in {0,5} -> b%10 in {0,5} | - | same | +0.88 |
| L28 gate c176 | 5% / 2% | **res** (R2 0.81): res in {18, 38..39, 118, 136..141} [coarser: res mod 100 in {18, 37..40}, R2 0.81] | **res** (R2 0.53): res in {18} | (reads) | same | +0.88 |
| L26 up c341 | 8% / 3% | **res%100** (R2 0.90): res mod 100 in {90..95} | unexplained (best R2 0.48) | (reads) | same | +0.88 |
| L26 down c570 | 5% / 0 | **res%100** (R2 0.86): res mod 100 in {90..94} | off (on 0) | - | same | +0.88 |
| L29 up c736 | 8% / 2% | **res%100** (R2 0.65): res mod 100 in {20, 50, 70, 74..76} | unexplained (best R2 0.43) | (reads) | same | +0.88 |
| L26 up c348 | 9% / 4% | **res%10** (R2 0.88): res mod 10 in {6} | **res** (R2 0.55): res in {6, 16, 26, 36, 96} | (reads) | same | +0.88 |
| L31 up c845 | 21% / 10% | **res%100** (R2 0.67): res mod 100 in {73..93, 95..96} | **tens(a,b)** (R2 0.51): a//10 in {8,10} -> b//10 in {0,1,2}; a//10 in {9} -> b//10 in {0,1,2,3,10} | (reads) | same | +0.88 |
| L19 gate c55 | 24% / 16% | **res%10** (R2 0.82): res mod 10 in {0..1} | **res%100** (R2 0.49): res mod 100 in {0..2, 10..11, 20, 30, 81, 90..91} | (reads) | same | +0.88 |
| L23 gate c105 | 10% / 17% | **res%10** (R2 0.98): res mod 10 in {2} | **units(a,b)** (R2 0.51): a%10 in {0} -> b%10 in {8}; a%10 in {1} -> b%10 in {9}; a%10 in {2} -> b%10 in {0,4,6,8}; a%10 in {3} -> b%10 in {1}; a%10 in {4} -> b%10 in {2,6,8}; a%10 in {5} -> b%10 in {3,7}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {5}; a%10 in {8} -> b%10 in {6}; a%10 in {9} -> b%10 in {7} | (reads) | same | +0.88 |
| L22 up c35 | 36% / 6% | **res%100** (R2 0.79): res mod 100 in {22..26, 42, 60..88} | unexplained (best R2 0.43) | (reads) | same | +0.88 |
| L26 up c88 | 10% / 6% | **res%10** (R2 0.94): res mod 10 in {8} | unexplained (best R2 0.47) | (reads) | same | +0.88 |
| L26 up c142 | 7% / 1% | **res%100** (R2 0.68): res mod 100 in {0, 55, 94..99} | unexplained (best R2 0.26) | (reads) | same | +0.88 |
| L30 down c239 | 19% / 3% | **res%100** (R2 0.83): res mod 100 in {33, 42..43, 52..53, 62..63, 72..73, 82..83, 92..93} | unexplained (best R2 0.41) | res: mod2 +2% | - | +0.88 |
| L30 gate c317 | 11% / 3% | **res%10** (R2 0.83): res mod 10 in {3} | **res** (R2 0.60): res in {3, 23, 33, 83, 93} | (reads) | same | +0.88 |
| L22 gate c129 | 10% / 3% | **res%10** (R2 0.95): res mod 10 in {6} | **res** (R2 0.67): res in {6, 16, 26} | (reads) | same | +0.88 |
| L29 down c928 | 0 / 1% | off (on 0) | **res%100** (R2 0.79): res mod 100: no class above 0.5 (max 0.43) | - | same | +0.88 |
| L27 up c168 | 1% / 0 | **res** (R2 0.66): res in {44, 69, 169} [coarser: res mod 100 in {69}, R2 0.83] | off (on 0) | (reads) | same | +0.88 |
| L19 down c11 | 46% / 44% | **tens(a,b)** (R2 0.71): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {4,8,9,10}; a//10 in {2,7} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {3,8} -> b//10 in {1,2,3,6,7,8}; a//10 in {4} -> b//10 in {1,2,5,6,7}; a//10 in {5} -> b//10 in {4,5,6}; a//10 in {6} -> b//10 in {3,4,5,8,9,10}; a//10 in {9} -> b//10 in {1,2,6,7,10}; a//10 in {10} -> b//10 in {0,1,2,5,6,7,10} | **tens(a,b)** (R2 0.61): a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {0,1,2,3,6,7,8,10}; a//10 in {3} -> b//10 in {1,2,3,7,8}; a//10 in {4} -> b//10 in {2,3,7,8}; a//10 in {5} -> b//10 in {3,4,8}; a//10 in {6} -> b//10 in {4,5,6,8,9}; a//10 in {7} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {1,2,3,6,7,8,9}; a//10 in {9} -> b//10 in {7,8,9}; a//10 in {10} -> b//10 in {3,4,8,9} | a: mod50 +4%; res: mod50 +11%, mod25 +18% | b: mod50 +4%; res: mod100 +2%, mod50 +11%, mod25 +5% | +0.88 |
| L23 up c653 | 38% / 20% | **res%100** (R2 0.77): res mod 100 in {5, 7..39, 69} | **res** (R2 0.72): res in {5, 7..33} | (reads) | same | +0.88 |
| L29 gate c308 | 5% / 1% | **res%100** (R2 0.75): res mod 100 in {47..49, 67..68} | unexplained (best R2 0.29) | (reads) | same | +0.88 |
| L24 gate c4 | 12% / 14% | **res%10** (R2 0.81): res mod 10 in {8} | unexplained (best R2 0.48) | (reads) | same | +0.88 |
| L24 up c11 | 25% / 4% | **res%100** (R2 0.85): res mod 100 in {60..76, 78..84} | unexplained (best R2 0.35) | (reads) | same | +0.88 |
| L27 gate c365 | 8% / 2% | **res%20** (R2 0.83): res mod 20 in {7, 17} [coarser: res mod 10 in {7}, R2 0.87] | unexplained (best R2 0.41) | (reads) | same | +0.88 |
| L30 down c322 | 9% / 1% | **res//10** (R2 0.81): (tens) res in {70..79, 170..173} | unexplained (best R2 0.22) | - | same | +0.88 |
| L29 down c433 | 3% / 2% | **res%100** (R2 0.70): res mod 100 in {51..53} | unexplained (best R2 0.26) | - | same | +0.88 |
| L30 gate c868 | 4% / 1% | **res** (R2 0.79): res in {40, 69..70, 73..74, 170, 174} [coarser: res mod 100 in {69..70, 73..74}, R2 0.83] | unexplained (best R2 0.22) | (reads) | same | +0.88 |
| L25 down c335 | 9% / 2% | **res%100** (R2 0.75): res mod 100 in {13, 51..56, 93} | unexplained (best R2 0.27) | - | same | +0.88 |
| L30 up c159 | 9% / 3% | **res%100** (R2 0.80): res mod 100 in {22..28, 75} | **res** (R2 0.69): res in {24..27} | (reads) | same | +0.88 |
| L24 gate c91 | 17% / 10% | **res%20** (R2 0.87): res mod 20 in {3, 13, 16} [coarser: res mod 10 in {3, 6}, R2 0.84] | **res** (R2 0.59): res in {3, 6, 13, 16, 23, 33, 36, 43, 53, 63, 73, 83, 93, 96} | (reads) | same | +0.88 |
| L19 down c55 | 23% / 15% | **res%10** (R2 0.81): res mod 10 in {0..1} | **res%100** (R2 0.48): res mod 100 in {0..1, 10..11, 20, 30, 81, 90..91} | res: mod10 +7%, mod5 +3% | res: mod10 +6%, mod5 +3% | +0.88 |
| L29 gate c662 | 20% / 6% | **res%10** (R2 0.98): res mod 10 in {2..3} | **res** (R2 0.68): res in {3, 12..13, 22..23, 32..33} | (reads) | same | +0.88 |
| L28 up c300 | 9% / 3% | **res%10** (R2 0.92): res mod 10 in {4} | **res** (R2 0.64): res in {4, 14, 24, 34, 94} | (reads) | same | +0.88 |
| L23 gate c5 | 14% / 23% | **res%100** (R2 0.80): res mod 100 in {32..43} | unexplained (best R2 0.48) | (reads) | same | +0.88 |
| L20 gate c14 | 30% / 4% | **res//10** (R2 0.67): (tens) res in {81..99, 132..147, 174..200} | unexplained (best R2 0.40) | (reads) | same | +0.88 |
| L26 down c89 | 7% / 4% | **res%100** (R2 0.90): res mod 100 in {76..82} | unexplained (best R2 0.33) | - | same | +0.88 |
| L29 down c1009 | 3% / 1% | **res%100** (R2 0.62): res mod 100 in {90} | unexplained (best R2 0.41) | - | same | +0.88 |
| L30 gate c414 | 7% / 1% | **res** (R2 0.86): res in {42, 44, 46, 92, 141..148, 192} [coarser: res mod 100 in {42..47, 92}, R2 0.84] | unexplained (best R2 0.25) | (reads) | same | +0.88 |
| L29 up c146 | 6% / 1% | **res%100** (R2 0.81): res mod 100 in {25, 35, 45, 65, 85} | unexplained (best R2 0.48) | (reads) | same | +0.88 |
| L25 up c108 | 19% / 11% | **res%10** (R2 0.86): res mod 10 in {0, 9} | **units(a,b)** (R2 0.48): a%10 in {0} -> b%10 in {0,1}; a%10 in {1} -> b%10 in {1,2}; a%10 in {4} -> b%10 in {4}; a%10 in {7,8} -> b%10 in {8}; a%10 in {9} -> b%10 in {0,9} | (reads) | same | +0.88 |
| L20 down c223 | 5% / 7% | **res** (R2 0.73): res in {113..118} | **res//10** (R2 0.69): (tens) res in {10..19} | res: mod25 +2% | - | +0.88 |
| L29 gate c172 | 22% / 3% | **res%100** (R2 0.80): res mod 100 in {40..51, 80..89} | unexplained (best R2 0.29) | (reads) | same | +0.88 |
| L29 gate c282 | 10% / 5% | **res%100** (R2 0.77): res mod 100 in {22, 39..45, 62, 82} | **res** (R2 0.55): res in {12, 22, 40..44} | (reads) | same | +0.88 |
| L29 down c144 | 8% / 11% | **res//10** (R2 0.84): (tens) res in {70..79} | unexplained (best R2 0.49) | - | same | +0.88 |
| L29 gate c178 | 6% / 3% | **res%100** (R2 0.76): res mod 100 in {27..29, 48, 68, 88} | **res** (R2 0.60): res in {8, 28..29, 88} | (reads) | same | +0.88 |
| L29 down c842 | 5% / 2% | **res%100** (R2 0.74): res mod 100 in {21, 41, 51, 61, 81} | unexplained (best R2 0.28) | - | same | +0.88 |
| L31 down c296 | 3% / 1% | **res%100** (R2 0.82): res mod 100 in {81..83} | unexplained (best R2 0.31) | - | same | +0.88 |
| L20 down c15 | 61% / 46% | **res%5** (R2 0.83): res mod 5 in {0..2} | unexplained (best R2 0.44) | res: mod5 +18% | a: mod5 +2%; b: mod5 +3%; res: mod5 +14% | +0.88 |
| L24 down c28 | 16% / 3% | **res%50** (R2 0.70): res mod 50 in {2..5, 13..14, 43..44} | unexplained (best R2 0.25) | res: mod25 +3% | - | +0.88 |
| L25 gate c754 | 19% / 6% | **res%100** (R2 0.83): res mod 100 in {13, 15..16, 32..36, 52..56, 73, 75, 93, 95..96} [coarser: res mod 20 in {12..13, 15..16}, R2 0.85] | **res** (R2 0.62): res in {13, 15..16, 32..36, 93, 95..96} | (reads) | same | +0.88 |
| L26 up c175 | 9% / 3% | **res%10** (R2 0.89): res mod 10 in {1} | unexplained (best R2 0.49) | (reads) | same | +0.88 |
| L27 down c269 | 7% / 4% | **res%100** (R2 0.83): res mod 100 in {1, 21, 41, 51, 61, 71, 81, 91} [coarser: res mod 10 in {1}, R2 0.88] | **res** (R2 0.57): res in {1, 11, 31, 41} | - | same | +0.87 |
| L26 up c86 | 8% / 10% | **res//10** (R2 0.79): (tens) res in {2..4, 7..8, 101..109} [coarser: res mod 100 in {1..9}, R2 0.95] | **res//10** (R2 0.78): (tens) res in {0..9} | (reads) | same | +0.87 |
| L27 up c64 | 9% / 4% | **res%100** (R2 0.77): res mod 100 in {22..25, 50, 52..53} | **res** (R2 0.67): res in {20, 22..25} | (reads) | same | +0.87 |
| L29 up c497 | 3% / 0 | **res%100** (R2 0.85): res mod 100 in {1, 99} | off (on 0) | (reads) | same | +0.87 |
| L28 up c377 | 9% / 2% | **res%10** (R2 0.84): res mod 10 in {5} | **res** (R2 0.53): res in {15, 35, 85, 95} | (reads) | same | +0.87 |
| L24 up c22 | 16% / 9% | **res%100** (R2 0.79): res mod 100 in {13..15, 24..25, 34, 63..65, 73..75, 83..85, 94} [coarser: res mod 50 in {13..15, 23..25, 33..35, 44}, R2 0.82] | **res** (R2 0.52): res in {4, 13..15, 23..25, 34, 74, 84, 94} | (reads) | same | +0.87 |
| L27 gate c112 | 6% / 1% | **res%100** (R2 0.66): res mod 100 in {44..45, 84..85} | unexplained (best R2 0.30) | (reads) | same | +0.87 |
| L22 gate c17 | 18% / 7% | **res%10** (R2 0.81): res mod 10 in {2..3} | **res** (R2 0.60): res in {2..3, 12..13, 22..23, 43} | (reads) | same | +0.87 |
| L28 gate c82 | 14% / 5% | **res%100** (R2 0.89): res mod 100 in {3, 13, 23, 33, 40..44, 53, 63, 73, 83, 93} [coarser: res mod 50 in {3, 13, 23, 33, 43}, R2 0.81] | **res** (R2 0.68): res in {3, 13, 23, 33, 43, 83, 93} | (reads) | same | +0.87 |
| L31 down c187 | 13% / 5% | **res%100** (R2 0.58): res mod 100 in {25, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90} | unexplained (best R2 0.49) | - | same | +0.87 |
| L29 gate c147 | 0 / 2% | off (on 0) | **res%100** (R2 0.73): res mod 100 in {8} | (reads) | same | +0.87 |
| L26 down c37 | 12% / 7% | **res%100** (R2 0.82): res mod 100 in {7..8, 17..18, 27, 37..38, 57..58, 87, 97..98} | **res** (R2 0.66): res in {-97, 7..8, 17..18, 27, 37..38, 87, 97..98} | res: mod4 +3% | res: mod4 +4% | +0.87 |
| L25 up c66 | 16% / 3% | **res%100** (R2 0.81): res mod 100 in {1, 8..10, 89, 97..99} | **res** (R2 0.58): res in {-99, 8..10, 89, 97..99} | (reads) | same | +0.87 |
| L29 down c415 | 3% / 8% | **res** (R2 0.59): res in {30..32, 41, 91, 131} | unexplained (best R2 0.31) | - | same | +0.87 |
| L28 gate c129 | 11% / 6% | **res%10** (R2 0.86): res mod 10 in {9} | **res** (R2 0.58): res in {-99, -11, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99} | (reads) | same | +0.87 |
| L29 down c230 | 15% / 4% | **res%20** (R2 0.69): res mod 20 in {12..13} | **res** (R2 0.59): res in {12..13, 33, 92..93} | res: mod4 +2% | - | +0.87 |
| L21 up c53 | 20% / 8% | **res%10** (R2 0.94): res mod 10 in {6, 9} | **res** (R2 0.55): res in {6, 9, 16, 19, 26, 29, 89, 99} | (reads) | same | +0.87 |
| L23 down c106 | 9% / 2% | **res%10** (R2 0.84): res mod 10 in {9} | **res** (R2 0.66): res in {9, 19, 29, 89, 99} | - | same | +0.87 |
| L28 up c99 | 13% / 5% | **res%50** (R2 0.80): res mod 50 in {8, 18, 28, 38, 48} [coarser: res mod 10 in {8}, R2 0.82] | **res** (R2 0.58): res in {8, 18, 28, 38, 48, 78, 88, 98} | (reads) | same | +0.87 |
| L31 gate c233 | 2% / 0 | **res%100** (R2 0.76): res mod 100 in {69} | off (on 0) | (reads) | same | +0.87 |
| L26 up c194 | 5% / 1% | **res%100** (R2 0.83): res mod 100 in {48, 57..59, 78} | unexplained (best R2 0.18) | (reads) | same | +0.87 |
| L24 up c52 | 9% / 4% | **res%100** (R2 0.79): res mod 100 in {14, 24..25, 34, 64..65, 74, 84, 94} [coarser: res mod 50 in {14, 24, 34}, R2 0.88] | **res** (R2 0.54): res in {4, 14, 24..25, 34, 84} | (reads) | same | +0.87 |
| L21 down c96 | 12% / 10% | **res%100** (R2 0.56): res mod 100 in {18..20, 58, 76..80, 98} | unexplained (best R2 0.29) | - | same | +0.87 |
| L19 down c20 | 60% / 25% | **res%50** (R2 0.68): res mod 50 in {4..31} | **tens(a,b)** (R2 0.56): a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {2,3}; a//10 in {5} -> b//10 in {3,4,8}; a//10 in {6} -> b//10 in {0,4,5,9,10}; a//10 in {7} -> b//10 in {0,5,6}; a//10 in {8} -> b//10 in {0,6,7}; a//10 in {9} -> b//10 in {2,7,8}; a//10 in {10} -> b//10 in {2,3,4,8,9} | res: mod50 +9%, mod25 +4% | res: mod50 +4% | +0.87 |
| L30 up c379 | 4% / 1% | **res%100** (R2 0.76): res mod 100 in {23, 37, 73..74} | **res** (R2 0.59): res in {23, 37} | (reads) | same | +0.87 |
| L24 gate c36 | 17% / 9% | **res%100** (R2 0.89): res mod 100 in {68..84} | **tens(a,b)** (R2 0.59): a//10 in {0} -> b//10 in {7}; a//10 in {7,8} -> b//10 in {0,1,2}; a//10 in {9} -> b//10 in {1,2}; a//10 in {10} -> b//10 in {2} | (reads) | same | +0.87 |
| L19 gate c275 | 24% / 14% | unexplained (best R2 0.32) | unexplained (best R2 0.25) | (reads) | same | +0.87 |
| L24 down c87 | 8% / 3% | **res%100** (R2 0.83): res mod 100 in {80..86} | unexplained (best R2 0.49) | - | same | +0.87 |
| L24 gate c191 | 3% / 0 | **res** (R2 0.72): res in {76..77, 173..177} [coarser: res mod 100 in {74..77}, R2 0.88] | off (on 0) | (reads) | same | +0.87 |
| L20 up c21 | 40% / 40% | **res%10** (R2 0.90): res mod 10 in {0, 7..9} | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {0,1,2,3,9}; a%10 in {1} -> b%10 in {0,1,2,3}; a%10 in {2} -> b%10 in {1,2,3,4}; a%10 in {3} -> b%10 in {3,4,5,6}; a%10 in {4} -> b%10 in {4,5,6}; a%10 in {5} -> b%10 in {6,7}; a%10 in {6} -> b%10 in {6,7,8}; a%10 in {7} -> b%10 in {0,7,8,9}; a%10 in {8} -> b%10 in {0,1,2,7,8,9}; a%10 in {9} -> b%10 in {0,1,2,8,9} | (reads) | same | +0.87 |
| L30 gate c192 | 1% / 1% | **res** (R2 0.68): res in {30, 40} | unexplained (best R2 0.30) | (reads) | same | +0.87 |
| L30 gate c249 | 12% / 3% | **res%100** (R2 0.83): res mod 100 in {23, 33, 60..65, 67, 83} | unexplained (best R2 0.42) | (reads) | same | +0.87 |
| L29 gate c6 | 10% / 6% | **res%100** (R2 0.69): res mod 100 in {14, 34, 44, 64, 74..75, 79, 84, 94} | unexplained (best R2 0.43) | (reads) | same | +0.87 |
| L19 gate c41 | 37% / 25% | **units(a,b)** (R2 0.87): a%10 in {0} -> b%10 in {6,7,8,9}; a%10 in {1} -> b%10 in {5,6,7,8,9}; a%10 in {2} -> b%10 in {5,6,7}; a%10 in {3} -> b%10 in {5,6}; a%10 in {4} -> b%10 in {5}; a%10 in {5} -> b%10 in {1,2,3,4}; a%10 in {6} -> b%10 in {0,1,2,3}; a%10 in {7} -> b%10 in {0,1,2,8,9}; a%10 in {8} -> b%10 in {0,1,8,9}; a%10 in {9} -> b%10 in {0,7,8,9} | **units(a,b)** (R2 0.56): a%10 in {0} -> b%10 in {0,1,2,3,4}; a%10 in {1} -> b%10 in {1,2,3}; a%10 in {6} -> b%10 in {0,8,9}; a%10 in {7} -> b%10 in {0,1,8,9}; a%10 in {8} -> b%10 in {0,1,2,9}; a%10 in {9} -> b%10 in {0,1,2,3} | (reads) | same | +0.87 |
| L29 gate c72 | 33% / 13% | **res%50** (R2 0.73): res mod 50 in {19..34} | unexplained (best R2 0.45) | (reads) | same | +0.87 |
| L29 down c739 | 2% / 1% | **res** (R2 0.66): res in {14, 114, 144, 194} | **res** (R2 0.66): res in {14} | - | same | +0.87 |
| L29 up c344 | 2% / 0 | **res%100** (R2 0.66): res mod 100 in {64, 67..68} | off (on 0) | (reads) | same | +0.87 |
| L29 gate c904 | 8% / 1% | **res%100** (R2 0.68): res mod 100 in {67..70, 72..73, 83, 85, 88} | unexplained (best R2 0.27) | (reads) | same | +0.87 |
| L29 down c174 | 14% / 1% | **res%100** (R2 0.74): res mod 100 in {1, 97} | unexplained (best R2 0.45) | - | same | +0.87 |
| L30 down c359 | 8% / 3% | **res%100** (R2 0.82): res mod 100 in {23..26, 44, 64, 74, 84} | **res** (R2 0.71): res in {23..26} | - | same | +0.87 |
| L20 down c29 | 56% / 53% | **res%100** (R2 0.82): res mod 100 in {11..29, 55..86} | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {2,6,7,8}; a//10 in {1} -> b//10 in {7,8,9,10}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {1,2,6,7,8}; a//10 in {4,9} -> b//10 in {1,2,3,6,7,8}; a//10 in {5} -> b//10 in {2,3,4,7,8,9}; a//10 in {6} -> b//10 in {0,4,8,9,10}; a//10 in {7} -> b//10 in {0,1,5,6,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,5,6,7,8,10}; a//10 in {10} -> b//10 in {2,3,4,7,8} | a: mod50 +2%; res: mod50 +10% | b: mod50 +3%; res: mod50 +9% | +0.87 |
| L22 gate c212 | 14% / 4% | **res%20** (R2 0.85): res mod 20 in {0, 8, 10} | **res** (R2 0.52): res in {10, 20, 30} | (reads) | same | +0.87 |
| L29 down c343 | 2% / 2% | **res%100** (R2 0.67): res mod 100 in {17} | unexplained (best R2 0.47) | - | same | +0.87 |
| L30 down c370 | 8% / 1% | **res%100** (R2 0.71): res mod 100 in {90..95} | unexplained (best R2 0.39) | - | same | +0.86 |
| L29 gate c824 | 3% / 1% | **res%100** (R2 0.66): res mod 100 in {13, 33, 93} | **res** (R2 0.72): res in {13, 33} | (reads) | same | +0.86 |
| L24 gate c152 | 3% / 0 | **res%100** (R2 0.75): res mod 100 in {73..76} | off (on 0) | (reads) | same | +0.86 |
| L29 down c140 | 6% / 0 | **res%100** (R2 0.81): res mod 100 in {83..89} | off (on 0) | - | same | +0.86 |
| L21 up c36 | 38% / 12% | **res//10** (R2 0.74): (tens) res in {58, 60..91, 148..150, 152..198, 200} [coarser: res mod 100 in {0, 56..95, 97..98}, R2 0.86] | **tens(a,b)** (R2 0.60): a//10 in {5} -> b//10 in {7}; a//10 in {6} -> b//10 in {8}; a//10 in {7} -> b//10 in {0,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,10}; a//10 in {9} -> b//10 in {1,2}; a//10 in {10} -> b//10 in {1,2,3,4} | (reads) | same | +0.86 |
| L26 up c94 | 13% / 5% | **res%100** (R2 0.81): res mod 100 in {9, 17..19, 29, 39, 57..59, 69, 79, 89, 99} | **res** (R2 0.57): res in {9, 17..19, 29, 39, 69, 79, 89, 99} | (reads) | same | +0.86 |
| L29 down c282 | 10% / 3% | **res%100** (R2 0.83): res mod 100 in {22, 40..43, 52, 62, 82, 92} | **res** (R2 0.51): res in {22, 41..43} | - | same | +0.86 |
| L30 gate c251 | 6% / 2% | **res%100** (R2 0.68): res mod 100 in {21..23, 81} | **res** (R2 0.74): res in {21..23} | (reads) | same | +0.86 |
| L25 up c46 | 20% / 7% | **res** (R2 0.75): res in {9, 15..24, 91..95, 116..123, 147, 150..157, 165..167, 190..196} [coarser: res mod 100 in {16..23, 91..96}, R2 0.81] | **res** (R2 0.64): res in {16..21, 23, 91..95} | (reads) | same | +0.86 |
| L24 down c158 | 1% / 14% | **res** (R2 0.66): res in {6..15} | unexplained (best R2 0.44) | - | same | +0.86 |
| L30 up c230 | 13% / 3% | **res%100** (R2 0.81): res mod 100 in {28, 38, 48, 58, 68, 70, 78..80, 88, 98} | unexplained (best R2 0.35) | (reads) | same | +0.86 |
| L25 gate c190 | 10% / 0 | **res%100** (R2 0.77): res mod 100 in {53, 92..96, 98} | off (on 0) | (reads) | same | +0.86 |
| L29 up c282 | 21% / 5% | **res%100** (R2 0.73): res mod 100 in {1, 21..22, 32, 40..43, 51..52, 71..72, 80..83, 91..92} | unexplained (best R2 0.49) | (reads) | same | +0.86 |
| L29 gate c434 | 10% / 3% | **res%10** (R2 0.94): res mod 10 in {1} | **res** (R2 0.59): res in {11, 31, 41, 51, 91} | (reads) | same | +0.86 |
| L29 down c435 | 10% / 9% | **res//10** (R2 0.80): (tens) res in {61..69, 161..169} [coarser: res mod 100 in {61..69}, R2 1.00] | unexplained (best R2 0.40) | - | same | +0.86 |
| L23 gate c24 | 19% / 10% | **res%10** (R2 0.93): res mod 10 in {4, 7} | **res** (R2 0.64): res in {-97, -3, 4, 7, 14, 17, 24, 27, 37, 47, 57, 67, 77, 84, 87, 97} | (reads) | same | +0.86 |
| L21 down c43 | 28% / 24% | **res%100** (R2 0.88): res mod 100 in {0..5, 79..99} | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {9,10}; a//10 in {5} -> b//10 in {7}; a//10 in {6} -> b//10 in {6,7,8}; a//10 in {7} -> b//10 in {7,8,9,10}; a//10 in {8} -> b//10 in {0,8,9,10}; a//10 in {9} -> b//10 in {0,1,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,9,10} | res: mod100 +2% | - | +0.86 |
| L19 gate c11 | 34% / 23% | **tens(a,b)** (R2 0.68): a//10 in {1} -> b//10 in {8,9}; a//10 in {2} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {3} -> b//10 in {1,2,3,6,7,8}; a//10 in {4,9} -> b//10 in {1,2,6,7}; a//10 in {6} -> b//10 in {4}; a//10 in {7} -> b//10 in {2,3,4,7,8,9}; a//10 in {8} -> b//10 in {1,2,3,7,8}; a//10 in {10} -> b//10 in {1,2} | **tens(a,b)** (R2 0.56): a//10 in {2} -> b//10 in {0,1,2,7,8}; a//10 in {3} -> b//10 in {1,2,3,7,8}; a//10 in {4} -> b//10 in {2,3,8}; a//10 in {7} -> b//10 in {5,6,7,8}; a//10 in {8} -> b//10 in {2,3,6,7,8}; a//10 in {9} -> b//10 in {7,8}; a//10 in {10} -> b//10 in {3,4,8,9} | (reads) | same | +0.86 |
| L30 up c580 | 4% / 1% | **res%100** (R2 0.78): res mod 100 in {31..33, 82} | **res** (R2 0.60): res in {32..33} | (reads) | same | +0.86 |
| L21 down c36 | 35% / 7% | **res//10** (R2 0.79): (tens) res in {60..91, 153..195, 197} [coarser: res mod 100 in {56..95}, R2 0.92] | **tens(a,b)** (R2 0.65): a//10 in {8} -> b//10 in {0,1,2,10}; a//10 in {9} -> b//10 in {1,2}; a//10 in {10} -> b//10 in {1,2,3,4} | res: mod100 +3% | - | +0.86 |
| L27 up c17 | 9% / 2% | **res** (R2 0.79): res in {79..83, 101, 160..163, 170..173, 179..184} | unexplained (best R2 0.28) | (reads) | same | +0.86 |
| L26 down c789 | 1% / 0 | **res** (R2 0.54): res in {179..180} | off (on 0) | - | same | +0.86 |
| L25 down c63 | 10% / 2% | **res%100** (R2 0.72): res mod 100 in {30, 40, 50..51, 70, 80, 90..91} | unexplained (best R2 0.31) | - | same | +0.86 |
| L29 up c824 | 8% / 2% | **res%100** (R2 0.86): res mod 100 in {22, 42, 52, 62, 72, 82, 92} [coarser: res mod 50 in {2, 12, 22, 42}, R2 0.84] | **res** (R2 0.57): res in {22, 42} | (reads) | same | +0.86 |
| L30 down c183 | 6% / 1% | **res%100** (R2 0.62): res mod 100 in {31..33, 82} | unexplained (best R2 0.48) | - | same | +0.86 |
| L26 up c226 | 6% / 3% | **res%100** (R2 0.74): res mod 100 in {7, 27, 47, 67, 77, 87} | **res** (R2 0.51): res in {7, 27} | (reads) | same | +0.86 |
| L26 up c486 | 9% / 3% | **res%10** (R2 0.85): res mod 10 in {3} | **res** (R2 0.64): res in {3, 13, 33} | (reads) | same | +0.86 |
| L29 gate c433 | 7% / 2% | **res%100** (R2 0.80): res mod 100 in {32..34, 51..54} | unexplained (best R2 0.45) | (reads) | same | +0.86 |
| L25 gate c10 | 18% / 13% | **res%100** (R2 0.89): res mod 100 in {1, 85..99} | **tens(a,b)** (R2 0.53): a//10 in {0} -> b//10 in {9,10}; a//10 in {6} -> b//10 in {7}; a//10 in {7} -> b//10 in {8}; a//10 in {8} -> b//10 in {8,9}; a//10 in {9} -> b//10 in {0,1,3,4,5,9,10}; a//10 in {10} -> b//10 in {0,1,10} | (reads) | same | +0.86 |
| L23 gate c141 | 19% / 8% | **res%10** (R2 0.93): res mod 10 in {4, 7} | **res** (R2 0.61): res in {4, 7, 14, 17, 24, 27, 37, 87, 97} | (reads) | same | +0.86 |
| L25 up c98 | 8% / 2% | **res%100** (R2 0.86): res mod 100 in {14, 24, 54, 64, 74, 84, 94} [coarser: res mod 50 in {4, 14, 24, 44}, R2 0.86] | **res** (R2 0.64): res in {14, 24, 84} | (reads) | same | +0.86 |
| L19 up c55 | 23% / 15% | **res%10** (R2 0.81): res mod 10 in {0..1} | **res%100** (R2 0.47): res mod 100 in {0..2, 10..11, 20..21, 90..91} | (reads) | same | +0.86 |
| L28 down c760 | 1% / 0 | **res** (R2 0.70): res in {50, 150, 170, 180} | off (on 0) | - | same | +0.86 |
| L30 gate c232 | 10% / 2% | **res%100** (R2 0.60): res mod 100 in {28, 38, 48, 58, 68, 78, 83, 88, 98} | unexplained (best R2 0.34) | (reads) | same | +0.86 |
| L29 up c415 | 2% / 7% | **res%100** (R2 0.64): res mod 100 in {31..32} | unexplained (best R2 0.29) | (reads) | same | +0.86 |
| L29 gate c595 | 7% / 2% | **res%100** (R2 0.83): res mod 100 in {87..92} | **res** (R2 0.53): res in {9, 87..91} | (reads) | same | +0.86 |
| L25 up c553 | 2% / 8% | **res%100** (R2 0.68): res mod 100 in {8..9} | **res%100** (R2 0.67): res mod 100 in {8..10} | (reads) | same | +0.86 |
| L25 up c248 | 6% / 2% | **res%100** (R2 0.75): res mod 100 in {15, 25, 35, 55, 75, 95} [coarser: res mod 20 in {15}, R2 0.83] | **res** (R2 0.52): res in {15, 25, 95} | (reads) | same | +0.86 |
| L29 down c72 | 13% / 4% | **res%50** (R2 0.74): res mod 50 in {20, 22..23, 27, 30, 32} | unexplained (best R2 0.47) | - | same | +0.86 |
| L21 gate c43 | 20% / 15% | **res%100** (R2 0.73): res mod 100 in {0..2, 81..84, 94..99} | **tens(a,b)** (R2 0.53): a//10 in {5} -> b//10 in {7}; a//10 in {6} -> b//10 in {6,7,8}; a//10 in {7} -> b//10 in {7,8,9}; a//10 in {8} -> b//10 in {8,9,10}; a//10 in {9,10} -> b//10 in {9,10} | (reads) | same | +0.86 |
| L29 up c686 | 10% / 1% | **res//10** (R2 0.83): (tens) res in {51..58, 150..159, 175} [coarser: res mod 100 in {50..59}, R2 0.95] | unexplained (best R2 0.24) | (reads) | same | +0.86 |
| L21 gate c4 | 10% / 4% | **res%10** (R2 0.98): res mod 10 in {1} | **res** (R2 0.65): res in {1, 11, 21, 31} | (reads) | same | +0.86 |
| L23 down c3 | 15% / 17% | **res%100** (R2 0.88): res mod 100 in {0..1, 90..99} | unexplained (best R2 0.45) | res: mod100 +5%, mod50 +4%, mod25 +12% | - | +0.86 |
| L28 down c83 | 12% / 3% | **res%100** (R2 0.85): res mod 100 in {8, 18, 28, 38, 48, 58, 68, 75..79, 98} | unexplained (best R2 0.48) | - | same | +0.86 |
| L29 down c90 | 8% / 1% | **res%100** (R2 0.74): res mod 100 in {66..70} | unexplained (best R2 0.17) | - | same | +0.86 |
| L24 down c377 | 10% / 0 | **res%100** (R2 0.84): res mod 100 in {0..1, 93..99} | off (on 0) | - | same | +0.85 |
| L24 gate c82 | 20% / 11% | **res%10** (R2 0.89): res mod 10 in {1, 8} | **res** (R2 0.59): res in {1..2, 8, 11, 18, 21, 28, 31, 38, 41, 48, 78, 88, 91, 98} | (reads) | same | +0.85 |
| L29 down c146 | 8% / 2% | **res%100** (R2 0.74): res mod 100 in {73..79} | unexplained (best R2 0.23) | - | same | +0.85 |
| L24 gate c112 | 9% / 2% | **res%10** (R2 0.92): res mod 10 in {8} | unexplained (best R2 0.43) | (reads) | same | +0.85 |
| L28 down c359 | 11% / 7% | **units(a,b)** (R2 0.88): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {9}; a%10 in {2} -> b%10 in {8}; a%10 in {3} -> b%10 in {7}; a%10 in {4} -> b%10 in {6}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {3}; a%10 in {8} -> b%10 in {2}; a%10 in {9} -> b%10 in {1} | **units(a,b)** (R2 0.56): a%10 in {0,5} -> b%10 in {0,5}; a%10 in {1} -> b%10 in {1} | - | same | +0.85 |
| L23 gate c77 | 13% / 4% | **res%100** (R2 0.69): res mod 100 in {1, 21, 59..63, 77..82} | unexplained (best R2 0.35) | (reads) | same | +0.85 |
| L28 down c296 | 5% / 3% | **res%100** (R2 0.67): res mod 100 in {0, 20, 40, 60, 80} [coarser: res mod 20 in {0}, R2 0.88] | unexplained (best R2 0.48) | - | same | +0.85 |
| L29 up c358 | 14% / 6% | **res%100** (R2 0.89): res mod 100 in {1, 91..97, 99} | unexplained (best R2 0.40) | (reads) | same | +0.85 |
| L29 down c824 | 2% / 0 | **res%100** (R2 0.84): res mod 100 in {33, 83, 93} | off (on 0) | - | same | +0.85 |
| L29 up c178 | 8% / 2% | **res%100** (R2 0.71): res mod 100 in {8, 27..28, 38, 48, 68, 87..88} | unexplained (best R2 0.37) | (reads) | same | +0.85 |
| L30 up c205 | 13% / 2% | **res** (R2 0.78): res in {43..49, 86, 139..151, 156, 186} [coarser: res mod 100 in {42..50, 86}, R2 0.88] | unexplained (best R2 0.45) | (reads) | same | +0.85 |
| L19 up c423 | 48% / 27% | **res//10** (R2 0.75): (tens) res in {2..28, 81..127, 186..200} [coarser: res mod 100 in {0..27, 83..99}, R2 0.98] | **tens(a,b)** (R2 0.69): a//10 in {0} -> b//10 in {0,1,9,10}; a//10 in {1} -> b//10 in {0,1,2,10}; a//10 in {2} -> b//10 in {0,1,2}; a//10 in {3} -> b//10 in {2}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {4}; a//10 in {6} -> b//10 in {4,5,6,8}; a//10 in {7} -> b//10 in {5,6,7,8}; a//10 in {8} -> b//10 in {6,7,8,9}; a//10 in {9,10} -> b//10 in {0,8,9,10} | (reads) | same | +0.85 |
| L25 down c12 | 11% / 28% | **res%10** (R2 0.86): res mod 10 in {1} | unexplained (best R2 0.47) | res: mod10 +4%, mod5 +4%, mod2 +3% | res: mod10 +4%, mod5 +5%, mod2 +5% | +0.85 |
| L31 down c355 | 6% / 3% | **res%100** (R2 0.72): res mod 100 in {66..67, 86..89} | unexplained (best R2 0.32) | - | same | +0.85 |
| L19 gate c423 | 32% / 3% | **res//10** (R2 0.70): (tens) res in {36, 38, 40..74, 143..172} [coarser: res mod 100 in {42..72}, R2 0.96] | unexplained (best R2 0.33) | (reads) | same | +0.85 |
| L25 up c149 | 5% / 4% | **res%100** (R2 0.81): res mod 100 in {8..9, 88..89, 98..99} | unexplained (best R2 0.50) | (reads) | same | +0.85 |
| L25 down c43 | 7% / 17% | **res//10** (R2 0.68): (tens) res in {4..33, 35} | **a** (R2 0.53): a in {3, 5..16} | - | same | +0.85 |
| L24 down c43 | 22% / 21% | **res%100** (R2 0.79): res mod 100 in {15..19, 36..37, 55..59, 75..79, 95..99} [coarser: res mod 20 in {15..19}, R2 0.87] | unexplained (best R2 0.42) | res: mod20 +3% | a: mod20 +2%; b: mod20 +2%; res: mod20 +5% | +0.85 |
| L29 gate c379 | 2% / 0 | **res%100** (R2 0.88): res mod 100 in {75..76} | off (on 0) | (reads) | same | +0.85 |
| L25 up c151 | 22% / 7% | **res%100** (R2 0.80): res mod 100 in {6..9, 26..29, 46..49, 58, 66..68, 86..89, 96..99} | **res** (R2 0.62): res in {-97, 7..9, 26..29, 86..89, 96..99} | (reads) | same | +0.85 |
| L22 up c17 | 12% / 8% | **res%20** (R2 0.78): res mod 20 in {2..3} | **res%100** (R2 0.54): res mod 100 in {82} | (reads) | same | +0.85 |
| L31 gate c350 | 3% / 0 | **res%100** (R2 0.89): res mod 100 in {97..98} | off (on 0) | (reads) | same | +0.85 |
| L27 up c110 | 9% / 1% | **res%100** (R2 0.83): res mod 100 in {71..80} | unexplained (best R2 0.25) | (reads) | same | +0.85 |
| L27 down c133 | 17% / 3% | **res//10** (R2 0.77): (tens) res in {39, 41, 45..71} | unexplained (best R2 0.41) | - | same | +0.85 |
| L27 down c165 | 10% / 4% | **res%10** (R2 0.99): res mod 10 in {6} | **res** (R2 0.68): res in {6, 16, 26, 86, 96} | - | same | +0.85 |
| L28 down c903 | 2% / 0 | **res%100** (R2 0.71): res mod 100 in {72} | off (on 0) | - | same | +0.85 |
| L23 up c125 | 17% / 7% | **res%10** (R2 0.79): res mod 10 in {3..4} | **res** (R2 0.57): res in {3..4, 14, 23..24, 34, 84, 94} | (reads) | same | +0.85 |
| L26 up c89 | 20% / 6% | **res%100** (R2 0.84): res mod 100 in {16..19, 37, 47..48, 56..59, 66..68, 76..78, 87..88, 97} | **res** (R2 0.53): res in {16..19, 57, 77..78, 87..88, 97} | (reads) | same | +0.85 |
| L23 up c5 | 33% / 7% | **res//10** (R2 0.76): (tens) res in {43..51, 53..71, 140..190} [coarser: res mod 100 in {41..73}, R2 0.83] | unexplained (best R2 0.34) | (reads) | same | +0.85 |
| L28 up c98 | 13% / 27% | **res** (R2 0.76): res in {26..41, 126..133} [coarser: res mod 100 in {26..38}, R2 0.87] | **tens(a,b)** (R2 0.51): a//10 in {0} -> b//10 in {3}; a//10 in {1,6} -> b//10 in {2,3}; a//10 in {2} -> b//10 in {3,4,5}; a//10 in {3} -> b//10 in {0,3,4,5,6,9,10}; a//10 in {4} -> b//10 in {0,1,2,5,6}; a//10 in {5} -> b//10 in {2,6}; a//10 in {7} -> b//10 in {4}; a//10 in {8} -> b//10 in {5}; a//10 in {9} -> b//10 in {6}; a//10 in {10} -> b//10 in {6,7} | (reads) | same | +0.85 |
| L21 gate c133 | 31% / 4% | **res//10** (R2 0.84): (tens) res in {61..90, 159..195} [coarser: res mod 100 in {60..92}, R2 0.94] | unexplained (best R2 0.49) | (reads) | same | +0.85 |
| L30 up c293 | 6% / 0 | **res%100** (R2 0.66): res mod 100 in {9, 58..60} | off (on 0) | (reads) | same | +0.85 |
| L26 down c348 | 7% / 2% | **res%100** (R2 0.69): res mod 100 in {25, 34..35, 74..75, 94} | unexplained (best R2 0.36) | - | same | +0.85 |
| L30 down c580 | 2% / 6% | **res** (R2 0.59): res in {32, 52, 132} | unexplained (best R2 0.21) | - | same | +0.85 |
| L29 gate c358 | 9% / 4% | **res%100** (R2 0.76): res mod 100 in {0..6, 90, 97..99} | unexplained (best R2 0.41) | (reads) | same | +0.85 |
| L22 up c5 | 42% / 13% | **res%100** (R2 0.66): res mod 100 in {31..59, 61} | **tens(a,b)** (R2 0.51): a//10 in {4} -> b//10 in {0,4,5,9,10}; a//10 in {5} -> b//10 in {0,1,9,10}; a//10 in {6} -> b//10 in {1}; a//10 in {7} -> b//10 in {3}; a//10 in {8} -> b//10 in {3,4}; a//10 in {9} -> b//10 in {4,5}; a//10 in {10} -> b//10 in {4,5,6} | (reads) | same | +0.85 |
| L20 down c6 | 60% / 49% | **res%20** (R2 0.60): res mod 20 in {1..11} | unexplained (best R2 0.33) | res: mod20 +31% | b: mod20 +4%; res: mod25 +6%, mod20 +23% | +0.85 |
| L24 up c50 | 13% / 4% | **res%50** (R2 0.84): res mod 50 in {6, 16..17, 26, 36, 46} [coarser: res mod 10 in {6}, R2 0.82] | **res** (R2 0.52): res in {16..17, 26, 96} | (reads) | same | +0.85 |
| L29 up c209 | 12% / 3% | **res%100** (R2 0.86): res mod 100 in {12, 22, 32, 42, 52, 62, 67..68, 72, 82, 92} [coarser: res mod 50 in {2, 12, 22, 32, 42}, R2 0.84] | **res** (R2 0.56): res in {12, 22} | (reads) | same | +0.85 |
| L28 down c71 | 14% / 3% | **res%100** (R2 0.83): res mod 100 in {26, 45..46, 60..69, 86} | unexplained (best R2 0.32) | - | same | +0.85 |
| L28 gate c301 | 4% / 0 | **res%100** (R2 0.82): res mod 100 in {54..55, 65..66} | off (on 0) | (reads) | same | +0.85 |
| L28 down c304 | 6% / 21% | **res** (R2 0.69): res in {32..39, 133, 135..137} [coarser: res mod 100 in {33..39}, R2 0.90] | unexplained (best R2 0.46) | - | same | +0.85 |
| L22 down c33 | 22% / 10% | **res%100** (R2 0.82): res mod 100 in {1, 86..99} | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {9,10}; a//10 in {7} -> b//10 in {8}; a//10 in {8} -> b//10 in {9}; a//10 in {9} -> b//10 in {0,1,3,4,5,9,10}; a//10 in {10} -> b//10 in {0,1} | res: mod100 +2%, mod25 +4% | - | +0.85 |
| L29 up c835 | 9% / 3% | **res** (R2 0.73): res in {28, 30..31, 60, 70, 90, 120, 128..134, 150, 169..170} | **res** (R2 0.50): res in {20, 28, 30..31} | (reads) | same | +0.85 |
| L29 up c627 | 1% / 0 | **res%100** (R2 0.71): res mod 100: no class above 0.5 (max 0.38) | off (on 0) | (reads) | same | +0.84 |
| L24 down c22 | 14% / 7% | **res%100** (R2 0.71): res mod 100 in {13..17, 63..68, 70..71, 74} [coarser: res mod 50 in {13..17, 24}, R2 0.83] | **res** (R2 0.55): res in {13..17, 24} | res: mod25 +2% | - | +0.84 |
| L29 gate c435 | 12% / 9% | **res//10** (R2 0.85): (tens) res in {60..70, 160..170} [coarser: res mod 100 in {60..70}, R2 0.99] | unexplained (best R2 0.43) | (reads) | same | +0.84 |
| L30 gate c293 | 6% / 0 | **res%100** (R2 0.80): res mod 100 in {58..60, 98} | off (on 0) | (reads) | same | +0.84 |
| L26 gate c155 | 8% / 2% | **res%100** (R2 0.76): res mod 100 in {9, 69, 88..92} | unexplained (best R2 0.49) | (reads) | same | +0.84 |
| L23 gate c18 | 29% / 13% | **res** (R2 0.90): res in {73..104, 191..199} | **tens(a,b)** (R2 0.62): a//10 in {0} -> b//10 in {8,9,10}; a//10 in {8} -> b//10 in {0,1,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,9,10}; a//10 in {10} -> b//10 in {0,1,2} | (reads) | same | +0.84 |
| L22 down c48 | 11% / 3% | **res%100** (R2 0.69): res mod 100 in {8, 18, 27..30, 38, 48, 68, 88} | **res** (R2 0.54): res in {8, 27..29} | res: mod4 +3% | - | +0.84 |
| L28 up c24 | 12% / 21% | **res%10** (R2 0.81): res mod 10 in {6} | unexplained (best R2 0.39) | (reads) | same | +0.84 |
| L19 up c41 | 40% / 26% | **units(a,b)** (R2 0.85): a%10 in {0} -> b%10 in {6,7,8,9}; a%10 in {1} -> b%10 in {5,6,7,8,9}; a%10 in {2} -> b%10 in {5,6,7}; a%10 in {3} -> b%10 in {5,6}; a%10 in {5} -> b%10 in {1,2,3,4}; a%10 in {6} -> b%10 in {0,1,2,3,9}; a%10 in {7} -> b%10 in {0,1,2,7,8,9}; a%10 in {8} -> b%10 in {0,1,7,8,9}; a%10 in {9} -> b%10 in {0,6,7,8,9} | **units(a,b)** (R2 0.56): a%10 in {0} -> b%10 in {1,2,3,4}; a%10 in {1} -> b%10 in {2,3,4}; a%10 in {6} -> b%10 in {0,8,9}; a%10 in {7} -> b%10 in {0,1,2,8,9}; a%10 in {8} -> b%10 in {0,1,2,9}; a%10 in {9} -> b%10 in {0,1,2,3} | (reads) | same | +0.84 |
| L20 up c14 | 79% / 10% | **res//10** (R2 0.76): (tens) res in {26..53, 81..200} | unexplained (best R2 0.49) | (reads) | same | +0.84 |
| L28 down c98 | 6% / 27% | **res** (R2 0.77): res in {26..41} | **tens(a,b)** (R2 0.53): a//10 in {0} -> b//10 in {3}; a//10 in {1,6} -> b//10 in {2,3}; a//10 in {2} -> b//10 in {3,4,5}; a//10 in {3} -> b//10 in {0,3,4,5,6,9,10}; a//10 in {4} -> b//10 in {0,1,2,4,5,6}; a//10 in {5} -> b//10 in {2,6}; a//10 in {7} -> b//10 in {4}; a//10 in {8} -> b//10 in {5}; a//10 in {9,10} -> b//10 in {6} | - | same | +0.84 |
| L24 up c8 | 26% / 29% | **res//10** (R2 0.72): (tens) res in {52..85, 169} | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {6,7,8}; a//10 in {2,3,4} -> b//10 in {7,8}; a//10 in {5} -> b//10 in {8,9}; a//10 in {6} -> b//10 in {0,1,2,9,10}; a//10 in {7} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1,2,3}; a//10 in {9} -> b//10 in {1,2,3,4}; a//10 in {10} -> b//10 in {2,3,4} | (reads) | same | +0.84 |
| L30 gate c359 | 11% / 2% | **res%100** (R2 0.78): res mod 100 in {14, 23..25, 34, 44, 54, 64, 74, 84, 94} [coarser: res mod 50 in {4, 14, 24, 34, 44}, R2 0.85] | **res** (R2 0.52): res in {23..24, 84} | (reads) | same | +0.84 |
| L29 up c193 | 16% / 3% | **res%100** (R2 0.80): res mod 100 in {8, 26, 28, 46, 48, 66, 68, 82..84, 86, 88} | **res** (R2 0.52): res in {8, 82..83, 86, 88} | (reads) | same | +0.84 |
| L25 down c331 | 3% / 1% | **res%100** (R2 0.72): res mod 100 in {98} | unexplained (best R2 0.36) | - | same | +0.84 |
| L20 up c29 | 50% / 53% | **res%100** (R2 0.81): res mod 100 in {0..8, 12, 30..57, 86..99} [coarser: res mod 50 in {0..7, 34..49}, R2 0.83] | **tens(a,b)** (R2 0.57): a//10 in {0} -> b//10 in {0,1}; a//10 in {1} -> b//10 in {1,2,3}; a//10 in {2} -> b//10 in {2,3,4}; a//10 in {3} -> b//10 in {0,3,4,5,6,9}; a//10 in {4,9,10} -> b//10 in {0,4,5,6,9,10}; a//10 in {5} -> b//10 in {0,1,2,4,5,6,7,9,10}; a//10 in {6} -> b//10 in {0,1,2,3,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {2,3,4,6,7,8,9}; a//10 in {8} -> b//10 in {3,4,5,8,9,10} | (reads) | same | +0.84 |
| L28 gate c24 | 14% / 7% | **res** (R2 0.76): res in {16..18, 26, 36, 76..78, 96, 116..118, 126, 136, 156, 166..186, 188, 196..198, 200} | **res** (R2 0.52): res in {16..18, 26, 36, 76..78, 96} | (reads) | same | +0.84 |
| L27 gate c480 | 1% / 0 | **res** (R2 0.74): res in {89, 186..189} | off (on 0) | (reads) | same | +0.84 |
| L31 down c484 | 1% / 0 | **res%100** (R2 0.67): res mod 100: no class above 0.5 (max 0.42) | off (on 0) | - | same | +0.84 |
| L25 down c216 | 6% / 2% | **res%100** (R2 0.80): res mod 100 in {20, 30, 40, 60, 70, 80} | **res** (R2 0.54): res in {20} | - | same | +0.84 |
| L26 up c299 | 5% / 0 | **res%100** (R2 0.58): res mod 100 in {78, 83, 85, 88} | off (on 0) | (reads) | same | +0.84 |
| L30 up c309 | 9% / 5% | **res%100** (R2 0.65): res mod 100 in {4..5, 25, 55, 65, 85} | **res%100** (R2 0.52): res mod 100 in {5} | (reads) | same | +0.84 |
| L29 down c134 | 1% / 2% | unexplained (best R2 0.23) | unexplained (best R2 0.49) | - | same | +0.84 |
| L24 down c210 | 7% / 1% | **res%20** (R2 0.74): res mod 20 in {12} | unexplained (best R2 0.28) | - | same | +0.84 |
| L29 down c444 | 8% / 1% | **res%100** (R2 0.69): res mod 100 in {90..93, 97..98} | unexplained (best R2 0.25) | - | same | +0.84 |
| L22 gate c427 | 10% / 3% | **res%10** (R2 0.90): res mod 10 in {2} | **res** (R2 0.64): res in {2, 12, 22} | (reads) | same | +0.84 |
| L30 gate c882 | 8% / 2% | **res%100** (R2 0.85): res mod 100 in {17, 27..28, 37, 57, 67, 77, 87} [coarser: res mod 50 in {17, 27, 37}, R2 0.82] | **res** (R2 0.55): res in {17, 27, 37} | (reads) | same | +0.84 |
| L22 down c376 | 1% / 22% | **res** (R2 0.81): res in {2..15} | **res%100** (R2 0.62): res mod 100 in {6..14} | - | same | +0.84 |
| L25 down c207 | 6% / 2% | **res%100** (R2 0.78): res mod 100 in {12, 14, 52, 72, 92} [coarser: res mod 20 in {12}, R2 0.80] | **res** (R2 0.62): res in {12, 14} | res: mod4 +2% | - | +0.83 |
| L25 gate c538 | 1% / 2% | **res%100** (R2 0.65): res mod 100 in {15} | unexplained (best R2 0.47) | (reads) | same | +0.83 |
| L30 up c181 | 16% / 4% | **res** (R2 0.77): res in {16..21, 110..125, 157} | **res** (R2 0.67): res in {16..20} | (reads) | same | +0.83 |
| L28 gate c90 | 20% / 9% | **res%100** (R2 0.73): res mod 100 in {11..17, 30..32, 34, 36, 52, 54, 56, 71..72, 76, 91..92, 94, 96} | **res** (R2 0.69): res in {11..17, 36, 96} | (reads) | same | +0.83 |
| L26 gate c37 | 15% / 7% | **res%100** (R2 0.88): res mod 100 in {7..8, 17..18, 27..28, 37..38, 57..58, 67..68, 87, 97..98} [coarser: res mod 20 in {7..8, 17..18}, R2 0.81] | **res** (R2 0.67): res in {-97, 7..8, 17..18, 27..28, 37..38, 97..98} | (reads) | same | +0.83 |
| L30 up c796 | 5% / 1% | **res%100** (R2 0.84): res mod 100 in {20, 40, 60, 70, 80} | **res** (R2 0.58): res in {20} | (reads) | same | +0.83 |
| L31 down c65 | 4% / 0 | **res%100** (R2 0.56): res mod 100 in {97} | off (on 0) | - | same | +0.83 |
| L29 down c604 | 10% / 3% | **res%10** (R2 0.77): res mod 10 in {1} | **res** (R2 0.50): res in {11, 21, 31, 51, 71} | - | same | +0.83 |
| L27 up c80 | 11% / 3% | **res%100** (R2 0.71): res mod 100 in {9, 35..37, 49..51, 69, 89..92} | unexplained (best R2 0.49) | (reads) | same | +0.83 |
| L19 gate c28 | 43% / 17% | **tens(a,b)** (R2 0.69): a//10 in {0,10} -> b//10 in {6,7,8}; a//10 in {1} -> b//10 in {5,6,7}; a//10 in {2} -> b//10 in {4,5,6}; a//10 in {3} -> b//10 in {3,4,5}; a//10 in {4} -> b//10 in {2,3,4,7}; a//10 in {5} -> b//10 in {1,2,3,6,7,8}; a//10 in {6} -> b//10 in {0,1,2,5,6,7}; a//10 in {7} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {8} -> b//10 in {0,8,9,10}; a//10 in {9} -> b//10 in {7,8,9} | **tens(a,b)** (R2 0.58): a//10 in {3} -> b//10 in {6}; a//10 in {4} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {3,6,7,8}; a//10 in {6} -> b//10 in {3,4,7,8}; a//10 in {7} -> b//10 in {3,4,5,8,9,10}; a//10 in {8} -> b//10 in {10}; a//10 in {10} -> b//10 in {2,3,7} | (reads) | same | +0.83 |
| L19 gate c159 | 17% / 17% | **units(a,b)** (R2 0.62): a%10 in {0} -> b%10 in {4,8}; a%10 in {2} -> b%10 in {2,4,6,8}; a%10 in {4,6,8} -> b%10 in {0,2,4,6,8} | unexplained (best R2 0.28) | (reads) | same | +0.83 |
| L19 down c28 | 57% / 20% | **res//10** (R2 0.79): (tens) res in {14..52, 100..162, 164} [coarser: res mod 100 in {1, 10..56}, R2 0.81] | **tens(a,b)** (R2 0.65): a//10 in {2} -> b//10 in {0}; a//10 in {4} -> b//10 in {1}; a//10 in {5} -> b//10 in {1,2,3}; a//10 in {6} -> b//10 in {2,3,4}; a//10 in {7} -> b//10 in {3,4,5}; a//10 in {8} -> b//10 in {4,5,6}; a//10 in {9} -> b//10 in {6,7}; a//10 in {10} -> b//10 in {5,6,7,8} | res: mod100 +12% | - | +0.83 |
| L26 up c154 | 4% / 8% | unexplained (best R2 0.14) | unexplained (best R2 0.20) | (reads) | same | +0.83 |
| L29 down c930 | 7% / 2% | **res%100** (R2 0.66): res mod 100 in {29, 59, 66, 69, 96, 99} | unexplained (best R2 0.24) | - | same | +0.83 |
| L27 down c64 | 10% / 3% | **res%100** (R2 0.88): res mod 100 in {21..25, 51..55} | **res** (R2 0.63): res in {22..25} | - | same | +0.83 |
| L29 gate c263 | 10% / 3% | **res%10** (R2 0.97): res mod 10 in {3} | **res** (R2 0.61): res in {3, 13, 33, 93} | (reads) | same | +0.83 |
| L27 down c586 | 2% / 3% | **res%100** (R2 0.85): res mod 100 in {20..21} | **res** (R2 0.61): res in {17..18, 20..21} | - | same | +0.83 |
| L21 up c35 | 17% / 2% | **res%100** (R2 0.80): res mod 100 in {57..73} | unexplained (best R2 0.25) | (reads) | same | +0.83 |
| L27 gate c64 | 10% / 6% | **res%100** (R2 0.79): res mod 100 in {20..25, 49..54} | **res** (R2 0.64): res in {19..25} | (reads) | same | +0.83 |
| L26 down c187 | 2% / 4% | unexplained (best R2 0.09) | unexplained (best R2 0.22) | - | same | +0.83 |
| L20 up c157 | 14% / 5% | unexplained (best R2 0.37) | unexplained (best R2 0.22) | (reads) | same | +0.83 |
| L23 up c3 | 27% / 8% | **res%100** (R2 0.74): res mod 100 in {33..40, 55..58, 74..79, 93..99} | unexplained (best R2 0.42) | (reads) | same | +0.83 |
| L19 down c41 | 40% / 27% | **units(a,b)** (R2 0.85): a%10 in {0} -> b%10 in {6,7,8,9}; a%10 in {1} -> b%10 in {5,6,7,8}; a%10 in {2} -> b%10 in {5,6,7}; a%10 in {3} -> b%10 in {5,6}; a%10 in {5} -> b%10 in {1,2,3,4}; a%10 in {6} -> b%10 in {0,1,2,3,9}; a%10 in {7} -> b%10 in {0,1,2,8,9}; a%10 in {8} -> b%10 in {0,1,7,8,9}; a%10 in {9} -> b%10 in {0,6,7,8,9} | **units(a,b)** (R2 0.55): a%10 in {0} -> b%10 in {1,2,3,4}; a%10 in {1} -> b%10 in {2,3,4}; a%10 in {6} -> b%10 in {0,8,9}; a%10 in {7} -> b%10 in {0,1,2,8,9}; a%10 in {8} -> b%10 in {0,1,2,9}; a%10 in {9} -> b%10 in {0,1,2,3} | a: mod10 +3%; b: mod10 +3%; res: mod10 +13% | res: mod10 +7% | +0.83 |
| L29 gate c321 | 4% / 2% | **res%100** (R2 0.67): res mod 100 in {15, 45, 65, 85} | unexplained (best R2 0.46) | (reads) | same | +0.83 |
| L25 down c10 | 13% / 9% | **res%100** (R2 0.81): res mod 100 in {86..96, 98} | unexplained (best R2 0.47) | res: mod100 +3%, mod50 +2%, mod25 +6% | - | +0.83 |
| L29 gate c376 | 2% / 3% | **res** (R2 0.76): res in {6, 25..26, 126} [coarser: res mod 100 in {26}, R2 0.81] | **res** (R2 0.64): res in {6, 16, 25..26} | (reads) | same | +0.83 |
| L23 up c41 | 7% / 12% | **res%100** (R2 0.61): res mod 100 in {9} | **res%100** (R2 0.65): res mod 100: no class above 0.5 (max 0.49) | (reads) | same | +0.82 |
| L27 down c42 | 14% / 4% | **res%100** (R2 0.72): res mod 100 in {9, 19, 29, 39, 48..49, 65..70, 79, 89} | unexplained (best R2 0.48) | res: mod4 +3% | - | +0.82 |
| L29 down c445 | 1% / 2% | **res%100** (R2 0.67): res mod 100 in {11} | **res%100** (R2 0.49): res mod 100 in {11} | - | same | +0.82 |
| L29 gate c51 | 14% / 4% | **res%100** (R2 0.81): res mod 100 in {17, 19..29} | **res** (R2 0.67): res in {20..23, 25} | (reads) | same | +0.82 |
| L28 down c288 | 4% / 1% | **res** (R2 0.89): res in {15, 113..116} | **res** (R2 0.60): res in {15} | - | same | +0.82 |
| L23 gate c58 | 11% / 4% | **res%100** (R2 0.84): res mod 100 in {0..1, 92..99} | unexplained (best R2 0.23) | (reads) | same | +0.82 |
| L26 down c24 | 16% / 3% | **res%100** (R2 0.73): res mod 100 in {25, 32..36, 38, 53..56, 58, 83..86, 88} | unexplained (best R2 0.30) | - | same | +0.82 |
| L28 down c176 | 6% / 1% | **res** (R2 0.71): res in {38..39, 89, 136..141, 179, 188..189} | unexplained (best R2 0.39) | - | same | +0.82 |
| L28 down c90 | 14% / 4% | **res%100** (R2 0.58): res mod 100 in {12, 31..32, 52, 71..72, 90..92} | unexplained (best R2 0.29) | - | same | +0.82 |
| L30 gate c499 | 4% / 0 | **res%100** (R2 0.83): res mod 100 in {1, 51, 61, 91} | off (on 0) | (reads) | same | +0.82 |
| L22 up c98 | 25% / 4% | **res%100** (R2 0.72): res mod 100 in {50..77} | unexplained (best R2 0.35) | (reads) | same | +0.82 |
| L24 down c8 | 25% / 28% | **res//10** (R2 0.74): (tens) res in {52..85, 169} | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {6,7,8}; a//10 in {2,3,4} -> b//10 in {7,8}; a//10 in {5} -> b//10 in {8,9}; a//10 in {6} -> b//10 in {0,1,9,10}; a//10 in {7} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1,2,3}; a//10 in {9} -> b//10 in {1,2,3,4}; a//10 in {10} -> b//10 in {2,3,4} | res: mod100 +2% | - | +0.82 |
| L30 gate c214 | 10% / 2% | **res%100** (R2 0.79): res mod 100 in {17, 27, 37, 57, 77..78, 87, 97} | **res** (R2 0.58): res in {17, 37, 57, 77} | (reads) | same | +0.82 |
| L26 up c24 | 11% / 5% | **res%50** (R2 0.69): res mod 50 in {3, 5, 23, 25, 33, 35} | **res** (R2 0.60): res in {3, 5, 13, 23, 25, 33, 35, 85} | (reads) | same | +0.82 |
| L27 down c38 | 10% / 3% | **res%100** (R2 0.83): res mod 100 in {19, 29, 39, 49, 58..59, 79, 89, 97..99} | **res** (R2 0.53): res in {-99, 19, 29, 39, 79, 98..99} | res: mod4 +4% | - | +0.82 |
| L30 gate c661 | 10% / 1% | **res** (R2 0.80): res in {45..50, 143..156, 158, 200} | unexplained (best R2 0.40) | (reads) | same | +0.82 |
| L25 gate c797 | 5% / 1% | **res%20** (R2 0.81): res mod 20 in {13} | unexplained (best R2 0.34) | (reads) | same | +0.82 |
| L29 gate c469 | 6% / 2% | **res%100** (R2 0.76): res mod 100 in {5, 15, 45, 55, 65, 75, 95} [coarser: res mod 50 in {5, 15, 45}, R2 0.83] | **res%100** (R2 0.57): res mod 100 in {5} | (reads) | same | +0.82 |
| L26 up c813 | 5% / 2% | **res%100** (R2 0.81): res mod 100 in {2, 32, 52, 72, 92} | **res** (R2 0.58): res in {2, 32} | (reads) | same | +0.82 |
| L28 up c667 | 4% / 2% | **res%100** (R2 0.79): res mod 100 in {30, 60, 80, 90} [coarser: res mod 50 in {30}, R2 0.84] | unexplained (best R2 0.39) | (reads) | same | +0.82 |
| L29 down c92 | 11% / 1% | **res%100** (R2 0.78): res mod 100 in {33, 51..58, 63, 73, 93} | unexplained (best R2 0.23) | - | same | +0.82 |
| L28 gate c435 | 9% / 3% | **res%10** (R2 0.93): res mod 10 in {8} | **res** (R2 0.59): res in {8, 18, 98} | (reads) | same | +0.82 |
| L25 down c55 | 20% / 4% | **res** (R2 0.74): res in {42..50, 52..68, 85..89} | unexplained (best R2 0.25) | - | same | +0.82 |
| L27 down c227 | 0 / 15% | off (on 0) | **res%100** (R2 0.56): res mod 100: no class above 0.5 (max 0.48) | - | same | +0.82 |
| L30 gate c883 | 0 / 0 | off (on 0) | same | (reads) | same | +0.82 |
| L28 up c588 | 3% / 2% | **res%100** (R2 0.56): res mod 100 in {5, 55, 95} | **res%100** (R2 0.61): res mod 100 in {95} | (reads) | same | +0.82 |
| L25 down c50 | 11% / 4% | **res%100** (R2 0.79): res mod 100 in {13, 22..25, 43, 53, 63, 73, 83} | **res** (R2 0.60): res in {3, 13, 23..25} | res: mod4 +6% | - | +0.82 |
| L27 up c30 | 21% / 4% | **res** (R2 0.84): res in {31, 41..43, 61..63, 71, 80..83, 101..103, 121..123, 131..133, 141..143, 151..153, 161..164, 171..173, 180..183} [coarser: res mod 100 in {1, 21..23, 31..33, 41..43, 61..63, 71..73, 80..83}, R2 0.83] | unexplained (best R2 0.35) | (reads) | same | +0.82 |
| L23 up c562 | 15% / 5% | **res** (R2 0.74): res in {38..44, 49, 136..151, 159..161, 179..183, 190, 200} | unexplained (best R2 0.31) | (reads) | same | +0.82 |
| L24 gate c41 | 23% / 5% | **res%100** (R2 0.91): res mod 100 in {0..1, 81..99} | unexplained (best R2 0.48) | (reads) | same | +0.82 |
| L29 down c53 | 2% / 0 | **res** (R2 0.61): res in {83, 147, 180, 182..183} | off (on 0) | - | same | +0.82 |
| L31 gate c9 | 11% / 8% | **res%100** (R2 0.58): res mod 100 in {2..7, 83, 97..98} | unexplained (best R2 0.49) | (reads) | same | +0.82 |
| L27 down c112 | 8% / 1% | **res%100** (R2 0.71): res mod 100 in {24, 44..45, 48, 84..85} | unexplained (best R2 0.30) | res: mod4 +2% | - | +0.82 |
| L27 gate c165 | 10% / 3% | **res%10** (R2 0.98): res mod 10 in {6} | **res** (R2 0.67): res in {6, 16, 26, 96} | (reads) | same | +0.82 |
| L29 gate c343 | 11% / 4% | **res%100** (R2 0.83): res mod 100 in {7, 17, 27, 37, 47, 57, 67, 72..73, 77, 87, 97} [coarser: res mod 20 in {7, 17}, R2 0.81] | **res** (R2 0.58): res in {7, 17, 27} | (reads) | same | +0.82 |
| L28 gate c522 | 2% / 0 | **res** (R2 0.70): res in {47, 147..149} | off (on 0) | (reads) | same | +0.82 |
| L29 gate c762 | 0 / 3% | off (on 0) | **res%100** (R2 0.66): res mod 100: no class above 0.5 (max 0.48) | (reads) | same | +0.81 |
| L19 gate c143 | 25% / 11% | **units(a,b)** (R2 0.76): a%10 in {1} -> b%10 in {3,4,7,8,9}; a%10 in {2,7} -> b%10 in {2,3,7,8}; a%10 in {3} -> b%10 in {1,2,6,7}; a%10 in {6} -> b%10 in {3,8,9}; a%10 in {8} -> b%10 in {1,6,7} | unexplained (best R2 0.33) | (reads) | same | +0.81 |
| L23 gate c106 | 24% / 9% | **res%10** (R2 0.77): res mod 10 in {0, 9} | **res** (R2 0.57): res in {-99, 0, 9..10, 19..20, 29..30, 89..90, 99} | (reads) | same | +0.81 |
| L29 gate c166 | 6% / 5% | **res** (R2 0.76): res in {63..69} | unexplained (best R2 0.38) | (reads) | same | +0.81 |
| L24 gate c108 | 8% / 4% | **res%100** (R2 0.72): res mod 100 in {19, 29, 39, 59, 69, 79} | **res** (R2 0.53): res in {9, 19..20, 29, 79, 99} | (reads) | same | +0.81 |
| L29 down c434 | 3% / 1% | **res%100** (R2 0.90): res mod 100 in {1, 51, 81} | unexplained (best R2 0.25) | - | same | +0.81 |
| L23 up c231 | 15% / 8% | **res%100** (R2 0.69): res mod 100 in {31..37, 51, 53..55, 71, 73..75, 93..95} | unexplained (best R2 0.46) | (reads) | same | +0.81 |
| L29 down c480 | 7% / 1% | **res** (R2 0.90): res in {61, 71, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191} [coarser: res mod 100 in {1, 21, 31, 41, 61, 71, 91}, R2 0.82] | unexplained (best R2 0.30) | - | same | +0.81 |
| L30 up c357 | 9% / 1% | **res%100** (R2 0.49): res mod 100 in {56, 66, 85..87} | unexplained (best R2 0.23) | (reads) | same | +0.81 |
| L19 down c74 | 20% / 5% | **res%10** (R2 0.73): res mod 10 in {8..9} | **res** (R2 0.57): res in {8..9, 18..19, 28..29, 98..99} | res: mod10 +2% | - | +0.81 |
| L28 gate c121 | 6% / 1% | **res%100** (R2 0.79): res mod 100 in {29..30, 59, 79, 89} | **res** (R2 0.54): res in {29..30, 99} | (reads) | same | +0.81 |
| L31 down c748 | 10% / 1% | **res** (R2 0.72): res in {59..60, 79..80, 110, 119..120, 130, 140, 150, 155, 159..160, 165, 169..170, 175..180, 190, 195} | unexplained (best R2 0.20) | - | same | +0.81 |
| L29 up c72 | 26% / 6% | **res%100** (R2 0.66): res mod 100 in {17, 22..23, 27, 32..33, 42..43, 47, 57, 63, 72..78, 82..84, 87, 92..93} [coarser: res mod 50 in {22..25, 27..28, 32..34, 37, 42..43}, R2 0.84] | unexplained (best R2 0.40) | (reads) | same | +0.81 |
| L29 up c230 | 26% / 7% | **res%100** (R2 0.87): res mod 100 in {12..13, 22..23, 32..33, 42..43, 52..54, 62..63, 71..74, 82..83, 91..94} [coarser: res mod 20 in {2..3, 12..14}, R2 0.83] | **res** (R2 0.63): res in {3, 12..13, 22..23, 33, 43, 53, 73, 92..93} | (reads) | same | +0.81 |
| L29 gate c448 | 7% / 3% | **res** (R2 0.75): res in {25..26, 65, 75, 121..127} | **res** (R2 0.55): res in {15, 21, 25..26} | (reads) | same | +0.81 |
| L23 gate c459 | 17% / 5% | **res%10** (R2 0.84): res mod 10 in {4..5} | **res** (R2 0.63): res in {4, 14, 24, 34, 84, 94} | (reads) | same | +0.81 |
| L26 down c157 | 5% / 0 | **res%100** (R2 0.52): res mod 100 in {45, 94..95} | unexplained (best R2 0.23) | - | same | +0.81 |
| L27 up c165 | 9% / 3% | **res%10** (R2 0.92): res mod 10 in {6} | **res** (R2 0.67): res in {6, 16, 26, 96} | (reads) | same | +0.81 |
| L30 down c796 | 3% / 1% | **res%100** (R2 0.73): res mod 100 in {20, 40, 60} | **res** (R2 0.54): res in {20} | - | same | +0.81 |
| L29 down c318 | 1% / 2% | **res%100** (R2 0.77): res mod 100 in {12} | unexplained (best R2 0.50) | - | same | +0.81 |
| L29 gate c604 | 12% / 3% | **res%100** (R2 0.89): res mod 100 in {1, 11, 21, 31, 41, 51..53, 61, 71, 81, 91} [coarser: res mod 50 in {1, 11, 21, 31, 41}, R2 0.85] | unexplained (best R2 0.49) | (reads) | same | +0.81 |
| L29 gate c415 | 4% / 2% | **res%100** (R2 0.71): res mod 100 in {11, 31, 71} | **res** (R2 0.59): res in {11, 31} | (reads) | same | +0.81 |
| L24 up c212 | 15% / 3% | **res** (R2 0.72): res in {19..20, 40, 70, 79..80, 99..100, 110, 119..122, 129..130, 139..140, 149..150, 160, 169..170, 179..180} [coarser: res mod 100 in {19..22, 29..30, 40, 60, 70, 79..80, 99}, R2 0.80] | unexplained (best R2 0.50) | (reads) | same | +0.81 |
| L28 gate c211 | 10% / 4% | **res%100** (R2 0.72): res mod 100 in {8..12, 31, 51, 71, 91} | **res%100** (R2 0.58): res mod 100 in {10..11} | (reads) | same | +0.81 |
| L19 gate c797 | 23% / 14% | unexplained (best R2 0.30) | unexplained (best R2 0.29) | (reads) | same | +0.81 |
| L25 gate c12 | 12% / 42% | **res%10** (R2 0.84): res mod 10 in {1} | unexplained (best R2 0.42) | (reads) | same | +0.81 |
| L29 down c635 | 2% / 2% | **res** (R2 0.51): res in {9, 69, 89, 149} | unexplained (best R2 0.47) | - | same | +0.81 |
| L29 gate c48 | 12% / 3% | **res%100** (R2 0.77): res mod 100 in {10..13} | **res%100** (R2 0.76): res mod 100 in {11} | (reads) | same | +0.81 |
| L29 up c672 | 2% / 1% | **res%100** (R2 0.79): res mod 100 in {45, 85} | unexplained (best R2 0.38) | (reads) | same | +0.81 |
| L23 gate c606 | 10% / 4% | **res%10** (R2 0.99): res mod 10 in {6} | **res** (R2 0.67): res in {6, 16, 26, 36, 96} | (reads) | same | +0.81 |
| L28 down c51 | 5% / 2% | **res%100** (R2 0.75): res mod 100 in {18, 38, 58, 78, 98} [coarser: res mod 20 in {18}, R2 0.87] | unexplained (best R2 0.40) | - | res: mod4 +2% | +0.81 |
| L29 up c48 | 11% / 3% | **res%100** (R2 0.76): res mod 100 in {1, 10..11, 20, 51, 60, 70..71, 90} | **res** (R2 0.64): res in {10..11, 20} | (reads) | same | +0.81 |
| L28 up c294 | 14% / 5% | **res%100** (R2 0.83): res mod 100 in {0, 8..18} | **res** (R2 0.74): res in {10..15} | (reads) | same | +0.81 |
| L19 up c506 | 6% / 5% | unexplained (best R2 0.18) | unexplained (best R2 0.20) | (reads) | same | +0.81 |
| L21 gate c35 | 20% / 3% | **res** (R2 0.85): res in {57..70, 154..182, 184} [coarser: res mod 100 in {56..74, 78..80}, R2 0.81] | unexplained (best R2 0.36) | (reads) | same | +0.81 |
| L30 gate c535 | 9% / 6% | **res%100** (R2 0.83): res mod 100 in {0..1, 93, 95..99} | **a** (R2 0.61): a in {94..99} | (reads) | same | +0.81 |
| L19 gate c426 | 8% / 23% | unexplained (best R2 0.40) | **tens(a,b)** (R2 0.58): a//10 in {0} -> b//10 in {0,10}; a//10 in {4} -> b//10 in {9,10}; a//10 in {5} -> b//10 in {4,5,6,9,10}; a//10 in {6} -> b//10 in {6,7,8,9,10}; a//10 in {7} -> b//10 in {7,8,9,10}; a//10 in {8} -> b//10 in {8,9,10}; a//10 in {9} -> b//10 in {4,5,6,8,9,10}; a//10 in {10} -> b//10 in {0,4,5,6,9,10} | (reads) | same | +0.81 |
| L29 down c893 | 11% / 5% | **res//10** (R2 0.79): (tens) res in {30..31, 35..58} | unexplained (best R2 0.41) | - | same | +0.81 |
| L28 gate c304 | 8% / 23% | **res%100** (R2 0.80): res mod 100 in {32..39} | unexplained (best R2 0.48) | (reads) | same | +0.81 |
| L28 gate c51 | 8% / 3% | **res%100** (R2 0.83): res mod 100 in {8, 18, 28, 38, 58, 68, 78, 88, 98} [coarser: res mod 20 in {8, 18}, R2 0.80] | **res** (R2 0.59): res in {8, 18} | (reads) | same | +0.80 |
| L31 down c474 | 13% / 13% | **res** (R2 0.76): res in {7, 68..81} | **tens(a,b)** (R2 0.53): a//10 in {0} -> b//10 in {7,8}; a//10 in {7} -> b//10 in {0,1,2,3,10}; a//10 in {8} -> b//10 in {0,1,2}; a//10 in {9,10} -> b//10 in {2} | - | same | +0.80 |
| L31 down c51 | 7% / 0 | **res** (R2 0.68): res in {38..39, 71, 111, 129, 131, 138..139, 143, 159, 161, 163, 171} | off (on 0) | - | same | +0.80 |
| L27 gate c203 | 3% / 0 | **res%100** (R2 0.68): res mod 100 in {59, 79, 98..99} | off (on 0) | (reads) | same | +0.80 |
| L31 up c662 | 17% / 0 | **res%100** (R2 0.68): res mod 100 in {1, 92..99} | off (on 0) | (reads) | same | +0.80 |
| L28 gate c296 | 7% / 3% | **res%100** (R2 0.77): res mod 100 in {0, 20, 40, 59..60, 80} [coarser: res mod 20 in {0}, R2 0.87] | **res%100** (R2 0.54): res mod 100 in {0} | (reads) | same | +0.80 |
| L28 gate c80 | 12% / 4% | **res%100** (R2 0.90): res mod 100 in {65..75} | unexplained (best R2 0.27) | (reads) | same | +0.80 |
| L26 up c20 | 8% / 6% | **res%100** (R2 0.78): res mod 100 in {30..32, 60..62} | unexplained (best R2 0.39) | (reads) | same | +0.80 |
| L30 up c322 | 8% / 1% | **res%100** (R2 0.68): res mod 100 in {71..77} | unexplained (best R2 0.17) | (reads) | same | +0.80 |
| L28 up c121 | 4% / 2% | **res%100** (R2 0.81): res mod 100 in {15..16, 76} | **res** (R2 0.75): res in {15..16} | (reads) | same | +0.80 |
| L21 gate c96 | 13% / 17% | **res%100** (R2 0.55): res mod 100 in {18..21, 58..60, 77..80} | unexplained (best R2 0.33) | (reads) | same | +0.80 |
| L28 up c46 | 10% / 5% | **res%100** (R2 0.88): res mod 100 in {0, 20, 30..31, 40, 50, 60, 70, 80, 90} [coarser: res mod 50 in {0, 10, 20, 30..31, 40}, R2 0.83] | unexplained (best R2 0.43) | (reads) | same | +0.80 |
| L27 up c246 | 9% / 3% | **res%100** (R2 0.86): res mod 100 in {33..39, 73..74} | **res** (R2 0.53): res in {34..37} | (reads) | same | +0.80 |
| L25 gate c901 | 3% / 0 | **res** (R2 0.88): res in {97, 137, 147, 157, 197} | off (on 0) | (reads) | same | +0.80 |
| L23 down c216 | 13% / 1% | **res%100** (R2 0.70): res mod 100 in {60..70, 84..86, 88} | unexplained (best R2 0.25) | res: mod25 +2% | - | +0.80 |
| L29 gate c219 | 11% / 1% | **res** (R2 0.88): res in {51..54, 142..159} | unexplained (best R2 0.24) | (reads) | same | +0.80 |
| L31 gate c735 | 9% / 0 | **res%100** (R2 0.71): res mod 100 in {96..97} | off (on 0) | (reads) | same | +0.80 |
| L23 down c465 | 3% / 29% | unexplained (best R2 0.43) | **tens(a,b)** (R2 0.60): a//10 in {0} -> b//10 in {0,1,2,3,4,5,9,10}; a//10 in {1} -> b//10 in {0,1,2,3,10}; a//10 in {2} -> b//10 in {1,2,3}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3,4}; a//10 in {5} -> b//10 in {4,5}; a//10 in {6} -> b//10 in {5,6,7}; a//10 in {7} -> b//10 in {6,7,8}; a//10 in {8} -> b//10 in {7,8,9}; a//10 in {9} -> b//10 in {8,9,10}; a//10 in {10} -> b//10 in {9,10} | - | same | +0.80 |
| L19 up c426 | 53% / 24% | **tens(a,b)** (R2 0.65): a//10 in {0} -> b//10 in {1,2,3,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,6,7,8,10}; a//10 in {2} -> b//10 in {0,1,5,6,10}; a//10 in {3} -> b//10 in {0,3,4,5,6,9,10}; a//10 in {4} -> b//10 in {3,4,5,8,9,10}; a//10 in {5} -> b//10 in {2,3,4}; a//10 in {6} -> b//10 in {0,1,2,3}; a//10 in {7} -> b//10 in {0,1,2,10}; a//10 in {8} -> b//10 in {0,1,3,4,5,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,7,8,9,10}; a//10 in {10} -> b//10 in {2,3,7,8,9,10} | **tens(a,b)** (R2 0.64): a//10 in {5} -> b//10 in {7}; a//10 in {6} -> b//10 in {6,7,8}; a//10 in {7} -> b//10 in {0,7,8,9,10}; a//10 in {8} -> b//10 in {0,1,5,6,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,5,6,7,9,10}; a//10 in {10} -> b//10 in {0,1,2,6,7,10} | (reads) | same | +0.80 |
| L24 gate c706 | 8% / 3% | **res%10** (R2 0.81): res mod 10 in {5} | **res** (R2 0.64): res in {5, 15, 25, 35} | (reads) | same | +0.80 |
| L31 down c700 | 2% / 0 | **res** (R2 0.61): res in {85, 135, 155, 175, 185} | off (on 0) | - | same | +0.80 |
| L29 down c209 | 12% / 3% | **res%20** (R2 0.82): res mod 20 in {2, 12} [coarser: res mod 10 in {2}, R2 0.84] | **res** (R2 0.65): res in {12, 22, 32} | - | same | +0.80 |
| L28 up c435 | 5% / 20% | **res** (R2 0.66): res in {2..12, 20, 103, 106..109, 200} | **res//10** (R2 0.52): (tens) res in {-5, 0..13, 99} | (reads) | same | +0.80 |
| L25 down c76 | 15% / 3% | **res%100** (R2 0.74): res mod 100 in {17..18, 36..39, 57..59, 76..79, 97..98} [coarser: res mod 20 in {17..19}, R2 0.84] | unexplained (best R2 0.38) | res: mod4 +3% | - | +0.80 |
| L28 gate c71 | 19% / 5% | **res%100** (R2 0.89): res mod 100 in {6, 16, 26, 36, 45..46, 56, 60..68, 76, 86, 96} | **res** (R2 0.52): res in {6, 16, 26, 46, 86, 96} | (reads) | same | +0.80 |
| L26 gate c24 | 12% / 3% | **res%100** (R2 0.79): res mod 100 in {31..36, 53..56, 83..86} | **res** (R2 0.53): res in {31..35} | (reads) | same | +0.80 |
| L26 up c540 | 5% / 0 | **res** (R2 0.82): res in {134..139} | off (on 0) | (reads) | same | +0.80 |
| L25 gate c48 | 2% / 18% | **res** (R2 0.65): res in {2, 4, 8..12, 109, 189} | **res%100** (R2 0.54): res mod 100 in {7..14} | (reads) | same | +0.80 |
| L25 up c200 | 1% / 0 | **res%100** (R2 0.66): res mod 100 in {24, 64} | **res** (R2 0.53): res in {24} | (reads) | same | +0.80 |
| L30 gate c370 | 7% / 5% | **res%100** (R2 0.78): res mod 100 in {91..95} | **tens(a,b)** (R2 0.54): a//10 in {9} -> b//10 in {0,1,2,3,4,5,10} | (reads) | same | +0.80 |
| L29 gate c90 | 10% / 3% | **res%100** (R2 0.75): res mod 100 in {7, 17, 27, 37..38, 47, 57, 67..68, 77, 87, 97} [coarser: res mod 20 in {7, 17}, R2 0.80] | **res** (R2 0.51): res in {7, 17} | (reads) | same | +0.80 |
| L23 up c338 | 12% / 5% | **res** (R2 0.89): res in {2..12, 14..16, 18, 20, 22, 26, 36, 46, 56, 66, 76, 86, 96, 106, 116, 126, 136, 146, 156, 166, 176, 186, 196} | **res** (R2 0.60): res in {6, 16, 26, 36, 96} | (reads) | same | +0.80 |
| L24 gate c555 | 10% / 3% | **res%10** (R2 0.96): res mod 10 in {3} | **res** (R2 0.67): res in {3, 13, 23, 93} | (reads) | same | +0.80 |
| L28 gate c191 | 2% / 0 | **res%100** (R2 0.72): res mod 100 in {1, 51, 55} [coarser: res mod 50 in {1}, R2 0.84] | off (on 0) | (reads) | same | +0.80 |
| L31 down c176 | 6% / 1% | **res** (R2 0.80): res in {39, 59, 79, 99, 119, 139, 149, 159, 179, 189, 192, 194..195, 197..199} | unexplained (best R2 0.32) | - | same | +0.80 |
| L28 up c291 | 2% / 0 | **res%50** (R2 0.89): res mod 50 in {47} | off (on 0) | (reads) | same | +0.80 |
| L28 gate c124 | 4% / 0 | **res%100** (R2 0.83): res mod 100 in {53..55, 93} | off (on 0) | (reads) | same | +0.79 |
| L30 down c283 | 2% / 5% | **res** (R2 0.56): res in {21, 121, 171} [coarser: res mod 100 in {21}, R2 0.89] | unexplained (best R2 0.28) | - | same | +0.79 |
| L29 gate c419 | 5% / 2% | **res%100** (R2 0.66): res mod 100 in {9, 39, 69, 89, 99} | **res** (R2 0.60): res in {9, 99} | (reads) | same | +0.79 |
| L27 down c147 | 14% / 1% | **res** (R2 0.78): res in {60..67, 155..176} | unexplained (best R2 0.28) | - | same | +0.79 |
| L30 down c394 | 3% / 0 | **res%100** (R2 0.77): res mod 100 in {67, 87} | off (on 0) | - | same | +0.79 |
| L27 down c80 | 11% / 3% | **res%10** (R2 0.76): res mod 10 in {9} | **res** (R2 0.57): res in {9, 29, 89} | - | same | +0.79 |
| L23 down c459 | 7% / 3% | **res%50** (R2 0.74): res mod 50 in {5, 15, 25, 35} [coarser: res mod 10 in {5}, R2 0.85] | **res** (R2 0.66): res in {5, 15, 25} | - | same | +0.79 |
| L29 gate c122 | 1% / 2% | unexplained (best R2 0.35) | **res%100** (R2 0.59): res mod 100 in {6} | (reads) | same | +0.79 |
| L29 gate c146 | 4% / 1% | **res%100** (R2 0.78): res mod 100 in {74..76} | unexplained (best R2 0.37) | (reads) | same | +0.79 |
| L29 gate c774 | 5% / 3% | **res%20** (R2 0.89): res mod 20 in {6} | **res%100** (R2 0.51): res mod 100 in {6} | (reads) | same | +0.79 |
| L29 down c67 | 6% / 16% | **res** (R2 0.72): res in {4, 6..28, 37, 41..42} | **res** (R2 0.59): res in {1, 4, 9, 12..26} | - | same | +0.79 |
| L22 gate c22 | 36% / 13% | **res%50** (R2 0.71): res mod 50 in {1, 3, 7, 9, 11, 13, 17, 19, 21, 23, 27, 29, 31, 33, 39, 41, 43, 49} [coarser: res mod 10 in {1, 3, 7, 9}, R2 0.82] | **res** (R2 0.55): res in {1, 3, 5, 7, 9, 11, 13, 19, 21, 23, 29, 31, 39, 69, 79, 89, 99} | (reads) | same | +0.79 |
| L30 up c249 | 15% / 2% | **res%100** (R2 0.73): res mod 100 in {23, 33, 43, 53, 60..65, 73, 83..84, 93} | unexplained (best R2 0.45) | (reads) | same | +0.79 |
| L29 down c736 | 2% / 1% | **res%100** (R2 0.60): res mod 100: no class above 0.5 (max 0.47) | **res** (R2 0.61): res in {19, 99} | - | same | +0.79 |
| L28 up c114 | 3% / 0 | **res%100** (R2 0.88): res mod 100 in {94, 96..98} | off (on 0) | (reads) | same | +0.79 |
| L27 gate c657 | 13% / 1% | **res%100** (R2 0.85): res mod 100 in {27, 72..83, 87} | unexplained (best R2 0.29) | (reads) | same | +0.79 |
| L30 down c661 | 2% / 0 | **res** (R2 0.63): res in {47, 147..149} | off (on 0) | - | same | +0.79 |
| L25 up c11 | 21% / 3% | **res%100** (R2 0.78): res mod 100 in {1, 27, 31, 37, 41, 47, 51, 57, 61, 65, 67, 71, 77, 81, 87, 91, 97} | unexplained (best R2 0.27) | (reads) | same | +0.79 |
| L20 down c44 | 7% / 12% | unexplained (best R2 0.28) | unexplained (best R2 0.32) | a: mod4 +8%; b: mod4 +10%, mod2 +2% | a: mod4 +10%; b: mod4 +9% | +0.79 |
| L24 gate c744 | 8% / 2% | **res%100** (R2 0.90): res mod 100 in {18, 48, 58, 68, 78, 88, 98} | **res** (R2 0.64): res in {8, 18, 88, 98} | (reads) | same | +0.79 |
| L28 up c69 | 1% / 0 | **res%100** (R2 0.73): res mod 100 in {76} | off (on 0) | (reads) | same | +0.79 |
| L30 up c251 | 21% / 5% | **res%100** (R2 0.71): res mod 100 in {1, 11..12, 21..22, 31, 41..42, 51, 61..62, 71..72, 77, 81..82, 91..92} [coarser: res mod 50 in {1, 11..12, 21..22, 31..32, 41..42}, R2 0.83] | unexplained (best R2 0.41) | (reads) | same | +0.79 |
| L26 down c552 | 0 / 14% | off (on 0) | **res%100** (R2 0.55): res mod 100: no class above 0.5 (max 0.47) | - | same | +0.79 |
| L30 down c941 | 3% / 0 | **res** (R2 0.73): res in {64, 161..165, 167..169} | off (on 0) | - | same | +0.79 |
| L21 down c13 | 24% / 10% | **res%10** (R2 0.73): res mod 10 in {3..5} | **res** (R2 0.55): res in {-16, -6, 4, 14, 24..25, 34, 44, 54, 64, 74, 84, 94} | res: mod10 +6%, mod5 +3% | res: mod10 +3%, mod5 +2% | +0.78 |
| L28 gate c73 | 4% / 2% | **res%100** (R2 0.68): res mod 100 in {10, 50, 70} | **res%100** (R2 0.51): res mod 100 in {10} | (reads) | same | +0.78 |
| L27 down c142 | 5% / 3% | **res** (R2 0.67): res in {9, 19, 29, 39, 47..49, 59, 69, 79, 149} | **res** (R2 0.53): res in {9, 29} | - | same | +0.78 |
| L30 up c177 | 10% / 2% | **res%100** (R2 0.72): res mod 100 in {26, 56..57, 66, 77, 86, 95..98} | unexplained (best R2 0.41) | (reads) | same | +0.78 |
| L24 down c207 | 6% / 7% | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {9}; a//10 in {8} -> b//10 in {10}; a//10 in {9} -> b//10 in {0,8,9,10}; a//10 in {10} -> b//10 in {8,9,10} | **a** (R2 0.61): a in {93..99} | - | same | +0.78 |
| L24 gate c171 | 1% / 16% | **res** (R2 0.69): res in {2..7, 9, 104, 200} | **res//10** (R2 0.55): (tens) res in {-97, -3, 0..8, 99} | (reads) | same | +0.78 |
| L29 gate c130 | 12% / 2% | **res%100** (R2 0.86): res mod 100 in {4, 24, 44, 54, 60..65, 74, 84, 94} | unexplained (best R2 0.49) | (reads) | same | +0.78 |
| L26 up c36 | 12% / 22% | **res%100** (R2 0.81): res mod 100 in {1, 3, 7, 9, 11, 21, 29, 31, 41, 51, 61, 71, 81, 91} | unexplained (best R2 0.40) | (reads) | same | +0.78 |
| L29 up c762 | 4% / 2% | **res%20** (R2 0.74): res mod 20 in {5} | **res** (R2 0.61): res in {5, 25} | (reads) | same | +0.78 |
| L24 gate c444 | 2% / 0 | **res%100** (R2 0.73): res mod 100 in {36, 96} | off (on 0) | (reads) | same | +0.78 |
| L23 o c152 (H22) | 4% / 9% | unexplained (best R2 0.50) | unexplained (best R2 0.46) | - | same | +0.78 |
| L29 down c469 | 0 / 2% | off (on 0) | **res%100** (R2 0.61): res mod 100: no class above 0.5 (max 0.48) | - | same | +0.78 |
| L20 down c3 | 61% / 53% | **res%20** (R2 0.74): res mod 20 in {8..19} | unexplained (best R2 0.43) | a: mod10 +3%; res: mod20 +35% | b: mod20 +4%; res: mod25 +8%, mod20 +24% | +0.78 |
| L25 gate c61 | 2% / 1% | **res** (R2 0.51): res in {124, 142..143, 182..183} | unexplained (best R2 0.10) | (reads) | same | +0.78 |
| L31 down c776 | 0 / 3% | off (on 0) | **res%100** (R2 0.53): res mod 100: no class above 0.5 (max 0.45) | - | same | +0.78 |
| L29 gate c618 | 4% / 1% | **res%100** (R2 0.78): res mod 100 in {27..28, 67..68} | unexplained (best R2 0.43) | (reads) | same | +0.78 |
| L29 gate c686 | 5% / 0 | **res** (R2 0.73): res in {44, 54, 64, 104, 144, 150..154, 164, 194} | off (on 0) | (reads) | same | +0.78 |
| L29 down c280 | 3% / 1% | **res%100** (R2 0.60): res mod 100 in {16} | **res** (R2 0.63): res in {16..17} | - | same | +0.78 |
| L25 up c63 | 10% / 2% | **res%100** (R2 0.85): res mod 100 in {10, 40, 50, 90..94} | unexplained (best R2 0.38) | (reads) | same | +0.78 |
| L20 gate c44 | 6% / 12% | unexplained (best R2 0.25) | unexplained (best R2 0.32) | (reads) | same | +0.78 |
| L28 up c310 | 3% / 0 | **res** (R2 0.80): res in {50, 147..152} | off (on 0) | (reads) | same | +0.78 |
| L29 up c566 | 9% / 2% | **res** (R2 0.64): res in {35, 45, 49, 85, 95, 125, 135, 145, 148..156, 175, 185, 195} [coarser: res mod 100 in {25, 35, 45, 49, 51, 55, 75, 85, 95}, R2 0.84] | unexplained (best R2 0.46) | (reads) | same | +0.78 |
| L28 down c584 | 0 / 3% | off (on 0) | unexplained (best R2 0.38) | - | same | +0.78 |
| L31 up c237 | 5% / 20% | unexplained (best R2 0.41) | unexplained (best R2 0.50) | (reads) | same | +0.78 |
| L22 down c404 | 4% / 4% | **res//10** (R2 0.58): (tens) res in {180..200} | unexplained (best R2 0.39) | - | same | +0.78 |
| L26 down c207 | 5% / 3% | **res%100** (R2 0.70): res mod 100 in {9} | **res%100** (R2 0.81): res mod 100: no class above 0.5 (max 0.46) | - | same | +0.78 |
| L19 up c11 | 46% / 40% | **tens(a,b)** (R2 0.66): a//10 in {1} -> b//10 in {3,4,8,9}; a//10 in {2} -> b//10 in {1,2,3,4,7,8,9,10}; a//10 in {3,8} -> b//10 in {1,2,3,6,7,8}; a//10 in {4} -> b//10 in {1,2,6,7}; a//10 in {6} -> b//10 in {3,4,5,7,8,9,10}; a//10 in {7} -> b//10 in {1,2,3,4,6,7,8,9,10}; a//10 in {9} -> b//10 in {1,2,6,7,10}; a//10 in {10} -> b//10 in {5,6,7} | **tens(a,b)** (R2 0.60): a//10 in {2} -> b//10 in {0,1,2,3,6,7,8}; a//10 in {3} -> b//10 in {1,2,3,7,8}; a//10 in {4} -> b//10 in {3,7,8}; a//10 in {5} -> b//10 in {4,8}; a//10 in {6} -> b//10 in {5,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {1,2,3,6,7,8,9}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {3,4,5,6,7,8,9,10} | (reads) | same | +0.78 |
| L28 gate c12 | 27% / 10% | **res//10** (R2 0.86): (tens) res in {2, 8..18, 20, 100..128, 198..200} [coarser: res mod 100 in {0..2, 6, 8..26, 99}, R2 0.81] | **res** (R2 0.81): res in {8..18} | (reads) | same | +0.78 |
| L19 down c32 | 26% / 11% | **units(a,b)** (R2 0.82): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} | unexplained (best R2 0.31) | a: mod2 +18%; b: mod2 +18%; res: mod2 +13% | a: mod2 +4%; b: mod2 +3%; res: mod2 +3% | +0.78 |
| L29 up c321 | 4% / 0 | **res%100** (R2 0.79): res mod 100 in {45, 53..54, 65} | off (on 0) | (reads) | same | +0.78 |
| L28 down c965 | 2% / 0 | **res%100** (R2 0.58): res mod 100 in {71, 91} | off (on 0) | - | same | +0.78 |
| L21 down c15 | 50% / 29% | **res%20** (R2 0.61): res mod 20 in {0, 4, 13..19} | unexplained (best R2 0.44) | res: mod20 +7% | res: mod25 +3%, mod20 +7% | +0.78 |
| L29 gate c71 | 14% / 4% | **res%100** (R2 0.78): res mod 100 in {41..42, 44..45, 47, 91..96} [coarser: res mod 50 in {41..47}, R2 0.84] | unexplained (best R2 0.42) | (reads) | same | +0.77 |
| L25 gate c76 | 3% / 1% | **res%100** (R2 0.77): res mod 100 in {23, 63} | **res** (R2 0.51): res in {23} | (reads) | same | +0.77 |
| L27 gate c42 | 11% / 1% | **res%100** (R2 0.70): res mod 100 in {29, 49, 65..72} | unexplained (best R2 0.19) | (reads) | same | +0.77 |
| L24 gate c367 | 1% / 3% | unexplained (best R2 0.45) | **res%100** (R2 0.57): res mod 100 in {0, 5, 99} | (reads) | same | +0.77 |
| L25 gate c74 | 28% / 8% | **res%100** (R2 0.58): res mod 100 in {14..15, 20, 22..25, 60, 62..66, 68..80} | unexplained (best R2 0.32) | (reads) | same | +0.77 |
| L31 gate c20 | 3% / 0 | **res** (R2 0.66): res in {145..149} | off (on 0) | (reads) | same | +0.77 |
| L31 down c652 | 6% / 1% | **res** (R2 0.87): res in {91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191} | unexplained (best R2 0.18) | - | same | +0.77 |
| L25 up c30 | 30% / 3% | **res** (R2 0.78): res in {53, 59..64, 69..73, 99..104, 109..113, 142..144, 149..154, 158..165, 167..174, 181..183} | unexplained (best R2 0.15) | (reads) | same | +0.77 |
| L29 down c172 | 7% / 1% | **res%100** (R2 0.69): res mod 100 in {45, 47, 49, 83, 85, 87, 89} | unexplained (best R2 0.20) | res: mod4 +3% | - | +0.77 |
| L25 up c76 | 20% / 5% | **res%100** (R2 0.77): res mod 100 in {18, 36..39, 57..59, 76..79, 88, 94..99} | unexplained (best R2 0.36) | (reads) | same | +0.77 |
| L24 up c91 | 5% / 21% | **res%100** (R2 0.75): res mod 100 in {2..9} | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {0}; a//10 in {1} -> b//10 in {0,1}; a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {4,5}; a//10 in {6} -> b//10 in {5,6,7,8}; a//10 in {7} -> b//10 in {6,7,8,9}; a//10 in {8} -> b//10 in {7,8,9}; a//10 in {9} -> b//10 in {8,9,10}; a//10 in {10} -> b//10 in {5,8,9,10} | (reads) | same | +0.77 |
| L27 gate c80 | 4% / 1% | **res%100** (R2 0.77): res mod 100 in {9, 49, 89, 91} | **res** (R2 0.70): res in {9, 89} | (reads) | same | +0.77 |
| L27 gate c421 | 24% / 1% | **res//10** (R2 0.79): (tens) res in {96..119, 199} | unexplained (best R2 0.32) | (reads) | same | +0.77 |
| L29 down c871 | 1% / 0 | unexplained (best R2 0.48) | off (on 0) | - | same | +0.77 |
| L30 down c192 | 1% / 1% | **res** (R2 0.61): res in {29..30, 130} | unexplained (best R2 0.25) | - | same | +0.77 |
| L27 up c365 | 9% / 1% | **res%100** (R2 0.67): res mod 100 in {17, 36..37, 47, 56..57, 77, 87} | unexplained (best R2 0.33) | (reads) | same | +0.77 |
| L31 up c186 | 10% / 0 | **res%100** (R2 0.66): res mod 100 in {1, 94, 96..99} | off (on 0) | (reads) | same | +0.77 |
| L26 down c155 | 11% / 2% | **res** (R2 0.74): res in {89..91, 109..110, 129..130, 139, 149..150, 169..170, 180, 189..194, 200} | unexplained (best R2 0.18) | - | same | +0.77 |
| L27 gate c147 | 26% / 3% | **res//10** (R2 0.82): (tens) res in {55..78, 150..178} [coarser: res mod 100 in {52..78}, R2 0.98] | unexplained (best R2 0.39) | (reads) | same | +0.77 |
| L27 down c823 | 1% / 2% | **res%100** (R2 0.64): res mod 100: no class above 0.5 (max 0.36) | **cmp(a,b)** (R2 0.53): cmp(a,b)=-1: 0.01, cmp(a,b)=0: 0.96, cmp(a,b)=1: 0.00 | - | same | +0.77 |
| L25 down c139 | 12% / 1% | **res%100** (R2 0.82): res mod 100 in {0..1, 93..99} | unexplained (best R2 0.35) | - | same | +0.77 |
| L28 down c12 | 27% / 8% | **res//10** (R2 0.86): (tens) res in {9..17, 100..129, 199..200} | **res** (R2 0.81): res in {9..17} | - | same | +0.77 |
| L22 down c22 | 22% / 7% | **res** (R2 0.65): res in {3, 5, 7, 11, 21, 31, 41, 51, 61, 67, 71, 75, 77, 81, 91, 101, 107, 111, 113, 115, 117, 121, 123, 125, 127, 131, 133, 135, 137, 141, 143, 145, 147, 151, 153, 161, 163, 167, 171, 173, 175, 177, 181, 191} [coarser: res mod 100 in {1, 3, 5, 7, 11, 15, 17, 21, 27, 31, 41, 51, 61, 67, 71, 73, 75, 77, 81, 91}, R2 0.82] | **res** (R2 0.50): res in {1, 3, 5, 7, 11, 21, 31} | - | same | +0.77 |
| L30 up c810 | 6% / 8% | **res%100** (R2 0.58): res mod 100 in {0, 2..4, 32, 52, 82} | **res%100** (R2 0.56): res mod 100 in {2} | (reads) | same | +0.77 |
| L31 down c174 | 43% / 7% | **tens(a,b)** (R2 0.53): a//10 in {0} -> b//10 in {4,5,6,7,8}; a//10 in {1} -> b//10 in {4,5,6,7}; a//10 in {2} -> b//10 in {3,4,5,6}; a//10 in {3} -> b//10 in {1,2,3,4,5}; a//10 in {4} -> b//10 in {0,1,2,3,4}; a//10 in {5} -> b//10 in {0,1,2,3}; a//10 in {6,7} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1}; a//10 in {9} -> b//10 in {7} | unexplained (best R2 0.45) | - | same | +0.77 |
| L24 up c101 | 22% / 17% | **res** (R2 0.80): res in {2..7, 12..45, 49, 117..129} | **tens(a,b)** (R2 0.56): a//10 in {0} -> b//10 in {2}; a//10 in {2} -> b//10 in {0,2,10}; a//10 in {3} -> b//10 in {0,1}; a//10 in {4} -> b//10 in {1,2}; a//10 in {5} -> b//10 in {2,3}; a//10 in {6} -> b//10 in {3,4}; a//10 in {7} -> b//10 in {4,5}; a//10 in {8} -> b//10 in {5,6}; a//10 in {9} -> b//10 in {7}; a//10 in {10} -> b//10 in {7,8} | (reads) | same | +0.76 |
| L19 up c804 | 10% / 3% | **units(a,b)** (R2 0.71): a%10 in {1} -> b%10 in {8}; a%10 in {2} -> b%10 in {7}; a%10 in {3} -> b%10 in {6}; a%10 in {4} -> b%10 in {5}; a%10 in {5} -> b%10 in {3,4}; a%10 in {6} -> b%10 in {3}; a%10 in {7} -> b%10 in {2}; a%10 in {8} -> b%10 in {1} | unexplained (best R2 0.44) | (reads) | same | +0.76 |
| L29 gate c193 | 3% / 2% | **res** (R2 0.74): res in {106..108, 168} | **res%100** (R2 0.85): res mod 100 in {8} | (reads) | same | +0.76 |
| L24 gate c263 | 7% / 1% | **res%100** (R2 0.92): res mod 100 in {23, 33, 43, 53, 63, 73, 83} | unexplained (best R2 0.50) | (reads) | same | +0.76 |
| L28 down c140 | 5% / 3% | **res%100** (R2 0.82): res mod 100 in {4, 64, 74, 84, 94} | unexplained (best R2 0.49) | - | same | +0.76 |
| L29 up c871 | 2% / 1% | **res%100** (R2 0.86): res mod 100 in {7} | **res%100** (R2 0.52): res mod 100: no class above 0.5 (max 0.30) | (reads) | same | +0.76 |
| L29 up c739 | 2% / 0 | **res%100** (R2 0.58): res mod 100 in {67, 94} | off (on 0) | (reads) | same | +0.76 |
| L31 gate c47 | 54% / 5% | **tens(a,b)** (R2 0.56): a//10 in {0} -> b//10 in {8,9,10}; a//10 in {1} -> b//10 in {7,8,9,10}; a//10 in {2} -> b//10 in {6,7,8,9}; a//10 in {3,4} -> b//10 in {5,6,7,8,9,10}; a//10 in {5} -> b//10 in {3,4,7,8,9,10}; a//10 in {6} -> b//10 in {2,3,4,5,8,9,10}; a//10 in {7} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {8,9} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8,9} | unexplained (best R2 0.47) | (reads) | same | +0.76 |
| L28 up c733 | 16% / 3% | **res** (R2 0.78): res in {14, 24, 34, 44, 54, 74, 104, 113..115, 124..125, 130, 133..135, 142..145, 153..155, 174, 194} | unexplained (best R2 0.48) | (reads) | same | +0.76 |
| L28 gate c291 | 5% / 1% | **res%100** (R2 0.83): res mod 100 in {27, 47, 57, 67, 77, 97} [coarser: res mod 50 in {27, 47}, R2 0.87] | unexplained (best R2 0.48) | (reads) | same | +0.76 |
| L27 up c42 | 19% / 5% | **res%10** (R2 0.90): res mod 10 in {8..9} | **res** (R2 0.58): res in {-99, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99} | (reads) | same | +0.76 |
| L27 gate c140 | 4% / 0 | **res** (R2 0.69): res in {71, 77, 169, 171..177} | off (on 0) | (reads) | same | +0.76 |
| L28 up c74 | 0 / 3% | off (on 0) | **res%100** (R2 0.74): res mod 100: no class above 0.5 (max 0.45) | (reads) | same | +0.76 |
| L28 gate c526 | 6% / 13% | **res** (R2 0.61): res in {2..12, 14..15, 17, 19..21, 25, 104, 106, 200} | **res%100** (R2 0.60): res mod 100 in {4..9} | (reads) | same | +0.76 |
| L20 gate c13 | 47% / 21% | **res%10** (R2 0.78): res mod 10 in {0..3, 9} | **res** (R2 0.58): res in {-29, -19, -9, 0..3, 10..13, 20..23, 30..32, 41..42, 81..82, 91..92, 99} | (reads) | same | +0.76 |
| L29 down c331 | 5% / 5% | **res** (R2 0.74): res in {60..67} | unexplained (best R2 0.37) | - | same | +0.76 |
| L31 gate c57 | 13% / 2% | **res** (R2 0.74): res in {61, 71, 81, 91, 101, 111, 119..121, 131, 141, 151, 159..161, 170..171, 179..181, 191, 200} | unexplained (best R2 0.16) | (reads) | same | +0.76 |
| L26 down c2 | 11% / 1% | **res%100** (R2 0.72): res mod 100 in {66..74, 76..77} | unexplained (best R2 0.15) | - | same | +0.76 |
| L25 gate c297 | 2% / 0 | **res** (R2 0.84): res in {90, 186..193} | off (on 0) | (reads) | same | +0.76 |
| L30 gate c283 | 10% / 5% | **res%10** (R2 0.74): res mod 10 in {1} | unexplained (best R2 0.37) | (reads) | same | +0.76 |
| L25 up c256 | 12% / 3% | **res** (R2 0.77): res in {75..77, 86, 165..200} | **res** (R2 0.60): res in {74..80, 82, 84..89} | (reads) | same | +0.76 |
| L31 down c867 | 4% / 0 | **res** (R2 0.76): res in {74, 137..138, 147..148, 173..174, 177} | off (on 0) | - | same | +0.76 |
| L31 gate c128 | 6% / 1% | **res** (R2 0.65): res in {83..86, 179..200} | unexplained (best R2 0.21) | (reads) | same | +0.76 |
| L29 down c497 | 1% / 0 | **res%100** (R2 0.69): res mod 100 in {1} | off (on 0) | - | same | +0.75 |
| L25 down c145 | 2% / 2% | unexplained (best R2 0.36) | **units(a,b)** (R2 0.58): a%10 in {0} -> b%10 in {0} | - | same | +0.75 |
| L29 gate c280 | 7% / 4% | **res%20** (R2 0.74): res mod 20 in {6, 16} [coarser: res mod 10 in {6}, R2 0.88] | **res** (R2 0.62): res in {6, 16..17, 26, 96} | (reads) | same | +0.75 |
| L19 gate c513 | 18% / 3% | unexplained (best R2 0.26) | unexplained (best R2 0.13) | (reads) | same | +0.75 |
| L21 down c125 | 13% / 0 | **res** (R2 0.67): res in {38, 40, 47..49, 137..151, 158, 198} [coarser: res mod 100 in {37..42, 46..50}, R2 0.82] | off (on 0) | - | same | +0.75 |
| L21 down c231 | 6% / 1% | **res%20** (R2 0.68): res mod 20 in {11} | unexplained (best R2 0.50) | - | same | +0.75 |
| L28 down c73 | 3% / 3% | **res** (R2 0.75): res in {2, 8..11, 30, 50, 70, 110} | **res%100** (R2 0.52): res mod 100 in {10..11} | - | res: mod4 +2% | +0.75 |
| L19 down c143 | 22% / 9% | **units(a,b)** (R2 0.62): a%10 in {1,6} -> b%10 in {3,7,8,9}; a%10 in {2} -> b%10 in {2,7,8}; a%10 in {3,8} -> b%10 in {1,6,7}; a%10 in {7} -> b%10 in {2,3,7,8} | unexplained (best R2 0.24) | res: mod5 +3% | - | +0.75 |
| L23 down c18 | 27% / 18% | **res//10** (R2 0.75): (tens) res in {73..102, 190..191} | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {7,8,9,10}; a//10 in {7} -> b//10 in {8}; a//10 in {8} -> b//10 in {0,1,2,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,9,10}; a//10 in {10} -> b//10 in {0,1,2} | - | same | +0.75 |
| L31 down c134 | 32% / 10% | **tens(a,b)** (R2 0.56): a//10 in {0} -> b//10 in {8,9}; a//10 in {1} -> b//10 in {7,8}; a//10 in {2} -> b//10 in {6,7}; a//10 in {3} -> b//10 in {5,6}; a//10 in {4} -> b//10 in {4,5}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {2,3}; a//10 in {7} -> b//10 in {1,2,9}; a//10 in {8} -> b//10 in {0,1,8,9}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,8,9} | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {9,10}; a//10 in {8} -> b//10 in {0}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,9,10}; a//10 in {10} -> b//10 in {0,1,2} | - | same | +0.75 |
| L21 down c916 | 8% / 2% | **res%100** (R2 0.70): res mod 100 in {47..48, 67..68, 87..88} | **res** (R2 0.53): res in {7..8} | - | same | +0.75 |
| L23 down c21 | 12% / 0 | **res** (R2 0.60): res in {65, 85..90, 95, 105, 107, 115, 125, 145, 185} | off (on 0) | - | same | +0.75 |
| L21 up c7 | 58% / 23% | **res%50** (R2 0.82): res mod 50 in {18..45} | unexplained (best R2 0.47) | (reads) | same | +0.75 |
| L29 up c444 | 2% / 0 | **res%100** (R2 0.67): res mod 100 in {67..68, 94} | off (on 0) | (reads) | same | +0.75 |
| L28 up c165 | 6% / 1% | **res%100** (R2 0.81): res mod 100 in {28, 38, 58, 68, 78} | unexplained (best R2 0.38) | (reads) | same | +0.75 |
| L30 up c223 | 3% / 1% | **res%100** (R2 0.75): res mod 100 in {20, 40, 70, 80} | **res** (R2 0.52): res in {20} | (reads) | same | +0.75 |
| L29 down c51 | 14% / 2% | **res** (R2 0.88): res in {20..23, 116..130} | **res** (R2 0.62): res in {20..22} | - | same | +0.75 |
| L27 gate c73 | 6% / 1% | **res%100** (R2 0.79): res mod 100 in {4, 44, 84} | **res%100** (R2 0.53): res mod 100: no class above 0.5 (max 0.40) | (reads) | same | +0.75 |
| L20 down c80 | 10% / 4% | **tens(a,b)** (R2 0.73): a//10 in {0} -> b//10 in {8,9,10}; a//10 in {7} -> b//10 in {10}; a//10 in {8,10} -> b//10 in {0,8,9,10}; a//10 in {9} -> b//10 in {0,7,8,9,10} | unexplained (best R2 0.40) | - | same | +0.75 |
| L22 gate c35 | 24% / 7% | **res** (R2 0.73): res in {18, 20..28, 64..68, 74..78, 80..88, 122..128, 165..166, 183..186} [coarser: res mod 100 in {21..28, 64..68, 81..86, 88}, R2 0.83] | **res** (R2 0.52): res in {21..28} | (reads) | same | +0.75 |
| L26 gate c570 | 8% / 2% | **res%100** (R2 0.74): res mod 100 in {9, 19, 39, 59, 69, 89..91, 93} [coarser: res mod 50 in {9, 19, 39}, R2 0.81] | **res** (R2 0.69): res in {9, 19, 89} | (reads) | same | +0.74 |
| L26 gate c707 | 5% / 4% | **res** (R2 0.79): res in {76..79, 176..182, 196..200} | unexplained (best R2 0.22) | (reads) | same | +0.74 |
| L26 gate c20 | 13% / 10% | **res%100** (R2 0.78): res mod 100 in {28..34, 60..65} | unexplained (best R2 0.44) | (reads) | same | +0.74 |
| L19 gate c85 | 20% / 9% | **units(a,b)** (R2 0.88): a%10 in {0} -> b%10 in {5}; a%10 in {5} -> b%10 in {0,5,6,7,8,9}; a%10 in {6} -> b%10 in {5,6,7,8,9}; a%10 in {7,8} -> b%10 in {5,6,7}; a%10 in {9} -> b%10 in {5,6} | unexplained (best R2 0.32) | (reads) | same | +0.74 |
| L28 gate c588 | 0 / 1% | off (on 0) | **res%100** (R2 0.70): res mod 100: no class above 0.5 (max 0.46) | (reads) | same | +0.74 |
| L24 gate c210 | 9% / 2% | **res%100** (R2 0.80): res mod 100 in {32, 42, 52, 71..74, 92} | unexplained (best R2 0.44) | (reads) | same | +0.74 |
| L27 gate c204 | 7% / 2% | **res%100** (R2 0.76): res mod 100 in {1, 21, 31, 41, 61, 71, 81, 91} [coarser: res mod 10 in {1}, R2 0.84] | **res%100** (R2 0.53): res mod 100: no class above 0.5 (max 0.45) | (reads) | same | +0.74 |
| L24 gate c658 | 0 / 0 | **res** (R2 0.59): res in {174} | off (on 0) | (reads) | same | +0.74 |
| L21 down c49 | 16% / 2% | **res//10** (R2 0.70): (tens) res in {40, 44, 132..156, 158, 160} | unexplained (best R2 0.27) | res: mod25 +2% | - | +0.74 |
| L20 up c624 | 8% / 3% | unexplained (best R2 0.19) | unexplained (best R2 0.14) | (reads) | same | +0.74 |
| L23 up c4 | 34% / 12% | **res** (R2 0.80): res in {8, 12, 22, 28, 32, 38, 42, 48, 52, 58, 62, 68, 72, 78, 82, 88, 92, 94, 98, 102, 108, 112, 118, 122, 128, 132, 134, 138, 142..144, 146..200} | **units(a,b)** (R2 0.49): a%10 in {0} -> b%10 in {2,8}; a%10 in {2} -> b%10 in {0}; a%10 in {3} -> b%10 in {1}; a%10 in {4} -> b%10 in {2}; a%10 in {5} -> b%10 in {3,7}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {5}; a%10 in {8} -> b%10 in {6}; a%10 in {9} -> b%10 in {7} | (reads) | same | +0.74 |
| L27 up c38 | 7% / 4% | **res%100** (R2 0.61): res mod 100 in {0..1, 59, 95..99} | **res%100** (R2 0.53): res mod 100 in {0..1, 98..99} | (reads) | same | +0.74 |
| L20 gate c910 | 48% / 9% | unexplained (best R2 0.31) | unexplained (best R2 0.19) | (reads) | same | +0.74 |
| L21 up c2 | 36% / 9% | **res%20** (R2 0.83): res mod 20 in {11..17} | **res** (R2 0.54): res in {11..16, 33, 95} | (reads) | same | +0.74 |
| L29 gate c635 | 1% / 1% | **res%50** (R2 0.65): res mod 50 in {39} | **res** (R2 0.72): res in {9} | (reads) | same | +0.74 |
| L25 gate c149 | 5% / 1% | **res%50** (R2 0.55): res mod 50 in {1, 21, 31} | unexplained (best R2 0.40) | (reads) | same | +0.74 |
| L25 gate c42 | 34% / 8% | **res** (R2 0.72): res in {2..50, 52, 68, 76, 78..80, 82..90, 96, 126, 128..130, 138..140, 142..144, 146, 183, 186} | unexplained (best R2 0.38) | (reads) | same | +0.73 |
| L25 gate c79 | 6% / 0 | **res%100** (R2 0.89): res mod 100 in {1} | off (on 0) | (reads) | same | +0.73 |
| L31 up c570 | 57% / 3% | **res//10** (R2 0.67): (tens) res in {87..116, 118..200} | unexplained (best R2 0.42) | (reads) | same | +0.73 |
| L27 gate c125 | 7% / 1% | **res%100** (R2 0.78): res mod 100 in {48..52, 98..99} | unexplained (best R2 0.26) | (reads) | same | +0.73 |
| L26 down c840 | 2% / 0 | **res%100** (R2 0.79): res mod 100 in {53, 93} | off (on 0) | - | same | +0.73 |
| L27 up c62 | 16% / 4% | **res%50** (R2 0.69): res mod 50 in {5..7, 35..37, 45..46} | **res** (R2 0.56): res in {5..6, 35..36, 85..86, 95..97} | (reads) | same | +0.73 |
| L29 gate c144 | 2% / 0 | **res%100** (R2 0.58): res mod 100 in {94..95} [coarser: res mod 50 in {44}, R2 0.86] | off (on 0) | (reads) | same | +0.73 |
| L27 up c112 | 9% / 1% | **res** (R2 0.78): res in {24, 44, 48, 84, 113, 115..120, 144, 184} | unexplained (best R2 0.27) | (reads) | same | +0.73 |
| L29 gate c316 | 2% / 22% | **res** (R2 0.58): res in {2..15, 18} | **res//10** (R2 0.56): (tens) res in {-99, 0..19, 99} | (reads) | same | +0.73 |
| L28 down c543 | 7% / 0 | **res** (R2 0.78): res in {56, 85..88, 136..137, 146, 156..157, 185..188} [coarser: res mod 100 in {56, 85..88}, R2 0.83] | off (on 0) | - | same | +0.73 |
| L27 gate c32 | 10% / 0 | **res%100** (R2 0.76): res mod 100 in {87..89} | off (on 0) | (reads) | same | +0.73 |
| L23 gate c4 | 26% / 9% | **res%50** (R2 0.69): res mod 50 in {1..3, 11..13, 21..23, 31..32, 41..42} | **res** (R2 0.58): res in {-38, -28, -8, 2, 12, 22, 32, 42, 52, 62, 72, 82, 92} | (reads) | same | +0.73 |
| L30 down c305 | 2% / 0 | **res%50** (R2 0.72): res mod 50 in {1} | off (on 0) | - | same | +0.73 |
| L25 down c74 | 22% / 7% | **res** (R2 0.68): res in {20, 22..25, 30, 32..34, 60..85, 165, 173..174} | unexplained (best R2 0.32) | - | same | +0.73 |
| L31 gate c78 | 15% / 1% | **res** (R2 0.72): res in {34, 36, 66, 76, 84..86, 106, 115..116, 124..126, 131, 134..136, 146, 154..156, 164..166, 175..176, 185..186} | unexplained (best R2 0.26) | (reads) | same | +0.73 |
| L29 down c160 | 2% / 0 | **res** (R2 0.72): res in {129, 135..136} | off (on 0) | - | same | +0.73 |
| L25 up c396 | 13% / 2% | **res%100** (R2 0.73): res mod 100 in {1, 39, 49, 59, 69, 79, 89, 99} | unexplained (best R2 0.46) | (reads) | same | +0.73 |
| L24 gate c22 | 19% / 8% | **res%100** (R2 0.80): res mod 100 in {9..17, 64..67, 69..71} | **res** (R2 0.75): res in {9..17, 19} | (reads) | same | +0.73 |
| L28 gate c602 | 1% / 6% | **res** (R2 0.77): res in {4..5, 18, 30} | **res%100** (R2 0.59): res mod 100: no class above 0.5 (max 0.48) | (reads) | same | +0.73 |
| L27 gate c33 | 14% / 12% | **res//10** (R2 0.58): (tens) res in {91..99, 186..188, 191..197} [coarser: res mod 100 in {91..98}, R2 0.84] | **tens(a,b)** (R2 0.59): a//10 in {0} -> b//10 in {9,10}; a//10 in {1,2,4} -> b//10 in {9}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,9,10}; a//10 in {10} -> b//10 in {0,1,2,4} | (reads) | same | +0.73 |
| L30 up c370 | 11% / 10% | **res//10** (R2 0.76): (tens) res in {89..99, 189..195} [coarser: res mod 100 in {89..97}, R2 0.93] | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {9,10}; a//10 in {8} -> b//10 in {9}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,9,10}; a//10 in {10} -> b//10 in {0} | (reads) | same | +0.73 |
| L30 down c918 | 0 / 0 | off (on 0) | same | - | same | +0.73 |
| L25 down c73 | 27% / 4% | **res** (R2 0.67): res in {38, 41..42, 44..51, 56..58, 76..99} | unexplained (best R2 0.48) | - | same | +0.73 |
| L30 gate c669 | 2% / 0 | **res%100** (R2 0.71): res mod 100 in {48, 98..99} | off (on 0) | (reads) | same | +0.73 |
| L27 down c33 | 14% / 13% | **res//10** (R2 0.59): (tens) res in {91..99, 186, 188, 191..197} [coarser: res mod 100 in {91..98}, R2 0.84] | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {9,10}; a//10 in {1,2} -> b//10 in {9}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4} | - | same | +0.72 |
| L21 down c105 | 3% / 33% | **res//10** (R2 0.68): (tens) res in {2..25} | **tens(a,b)** (R2 0.72): a//10 in {0} -> b//10 in {0,1,2}; a//10 in {1} -> b//10 in {0,1,2,10}; a//10 in {2} -> b//10 in {0,1,2,3}; a//10 in {3} -> b//10 in {1,2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9}; a//10 in {9,10} -> b//10 in {7,8,9,10} | - | res: mod100 +4%, mod50 +2% | +0.72 |
| L29 up c219 | 14% / 1% | **res** (R2 0.78): res in {49..54, 147..164, 170, 172..175, 182, 192, 194, 196, 198, 200} | unexplained (best R2 0.20) | (reads) | same | +0.72 |
| L28 down c517 | 4% / 6% | **res** (R2 0.51): res in {12, 20, 102, 112, 132, 191..192, 198..200} | unexplained (best R2 0.38) | - | same | +0.72 |
| L29 up c22 | 0 / 2% | off (on 0) | **res** (R2 0.66): res in {9, 13, 15} | (reads) | same | +0.72 |
| L24 down c91 | 8% / 4% | **res%100** (R2 0.73): res mod 100 in {16, 36..37, 56, 95..96} | **res** (R2 0.54): res in {6, 16, 36, 95..97} | res: mod4 +3% | - | +0.72 |
| L25 down c220 | 5% / 30% | **res//10** (R2 0.68): (tens) res in {2, 4..28, 30} | **tens(a,b)** (R2 0.64): a//10 in {0,1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {0,1,2,3}; a//10 in {3} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {7,8,9} | - | res: mod100 +2% | +0.72 |
| L29 down c217 | 7% / 1% | **res%100** (R2 0.63): res mod 100 in {39, 49, 69, 79, 89} [coarser: res mod 50 in {39}, R2 0.80] | unexplained (best R2 0.36) | - | same | +0.72 |
| L28 up c136 | 13% / 1% | **res** (R2 0.74): res in {45, 55, 65, 75, 85, 95, 105, 125, 134..135, 142, 144..151, 154..155, 165, 174..175, 185, 194..195, 198} | unexplained (best R2 0.14) | (reads) | same | +0.72 |
| L31 gate c409 | 54% / 1% | unexplained (best R2 0.40) | unexplained (best R2 0.36) | (reads) | same | +0.72 |
| L25 gate c898 | 9% / 1% | **res%10** (R2 0.83): res mod 10 in {7} | unexplained (best R2 0.50) | (reads) | same | +0.72 |
| L27 gate c139 | 2% / 1% | **res%100** (R2 0.76): res mod 100 in {1, 61} | **res%100** (R2 0.80): res mod 100: no class above 0.5 (max 0.46) | (reads) | same | +0.72 |
| L27 down c69 | 16% / 3% | **res%10** (R2 0.77): res mod 10 in {7..8} | unexplained (best R2 0.49) | - | same | +0.72 |
| L21 up c105 | 19% / 8% | **res//10** (R2 0.74): (tens) res in {127, 130..154, 156..157, 187} | **tens(a,b)** (R2 0.58): a//10 in {4} -> b//10 in {10}; a//10 in {6} -> b//10 in {3}; a//10 in {7} -> b//10 in {3,4}; a//10 in {8} -> b//10 in {4,5}; a//10 in {9} -> b//10 in {6}; a//10 in {10} -> b//10 in {4,5,6,7,8} | (reads) | same | +0.72 |
| L27 up c149 | 6% / 0 | **res** (R2 0.76): res in {44, 84, 140..148, 184} | off (on 0) | (reads) | same | +0.72 |
| L28 down c377 | 8% / 1% | **res** (R2 0.80): res in {34, 103..104, 132..136, 144, 153..154, 174} | unexplained (best R2 0.45) | - | same | +0.72 |
| L19 gate c341 | 0 / 1% | off (on 0) | unexplained (best R2 0.11) | (reads) | same | +0.72 |
| L29 up c635 | 0 / 8% | off (on 0) | **res%100** (R2 0.54): res mod 100: no class above 0.5 (max 0.45) | (reads) | same | +0.72 |
| L28 gate c357 | 10% / 3% | **res%10** (R2 0.91): res mod 10 in {2} | **res** (R2 0.55): res in {2, 22} | (reads) | same | +0.72 |
| L23 gate c738 | 10% / 2% | **res%10** (R2 0.97): res mod 10 in {6} | **res** (R2 0.73): res in {6, 16, 26, 96} | (reads) | same | +0.71 |
| L31 down c423 | 1% / 0 | **res** (R2 0.72): res in {168..169} | off (on 0) | - | same | +0.71 |
| L20 down c100 | 71% / 20% | **res%100** (R2 0.54): res mod 100 in {1..5, 8..9, 13..15, 18..25, 29, 33..35, 38..45, 48..49, 52..65, 68..69, 73..75, 78..85, 88..89, 93..95, 97..99} | **res** (R2 0.50): res in {3..5, 9, 13..15, 19, 23..25, 33..34, 43..45, 54..55, 59, 63..65, 74..75, 79, 83..85, 89, 93..95, 98..99} | res: mod10 +3%, mod5 +6% | res: mod10 +2%, mod5 +3% | +0.71 |
| L28 gate c638 | 61% / 2% | **tens(a,b)** (R2 0.81): a//10 in {0} -> b//10 in {8,9,10}; a//10 in {1} -> b//10 in {7,8,9,10}; a//10 in {2} -> b//10 in {6,7,8,9,10}; a//10 in {3} -> b//10 in {5,6,7,8,9,10}; a//10 in {4} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {1,2,3,4,5,6,7,8}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,7,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | unexplained (best R2 0.34) | (reads) | same | +0.71 |
| L26 down c86 | 8% / 0 | **res//10** (R2 0.77): (tens) res in {101..109} | off (on 0) | - | same | +0.71 |
| L26 gate c529 | 6% / 7% | **res%100** (R2 0.71): res mod 100 in {1..3} | **res%100** (R2 0.69): res mod 100: no class above 0.5 (max 0.44) | (reads) | same | +0.71 |
| L31 gate c123 | 10% / 1% | **res** (R2 0.77): res in {71, 101, 111, 131, 136..143, 151, 161, 171} | unexplained (best R2 0.11) | (reads) | same | +0.71 |
| L31 down c103 | 2% / 3% | unexplained (best R2 0.47) | **res%100** (R2 0.49): res mod 100 in {9} | - | same | +0.71 |
| L25 down c98 | 3% / 1% | **res%100** (R2 0.73): res mod 100 in {24, 64, 84} | **res** (R2 0.53): res in {24} | - | same | +0.71 |
| L27 gate c62 | 5% / 2% | **res%100** (R2 0.60): res mod 100 in {55} | **res%100** (R2 0.73): res mod 100: no class above 0.5 (max 0.43) | (reads) | same | +0.71 |
| L19 gate c639 | 47% / 14% | **res%50** (R2 0.62): res mod 50 in {0..5, 33..49} | unexplained (best R2 0.41) | (reads) | same | +0.71 |
| L26 gate c363 | 4% / 1% | **res%100** (R2 0.76): res mod 100 in {32, 52, 72} | unexplained (best R2 0.43) | (reads) | same | +0.71 |
| L30 down c177 | 10% / 1% | **res%100** (R2 0.71): res mod 100 in {26, 56..58, 66, 95..98} | unexplained (best R2 0.25) | - | same | +0.71 |
| L30 down c465 | 7% / 1% | **res//10** (R2 0.64): (tens) res in {130..140} | unexplained (best R2 0.17) | - | same | +0.71 |
| L26 gate c703 | 3% / 0 | **res** (R2 0.90): res in {159..165} | off (on 0) | (reads) | same | +0.70 |
| L30 up c317 | 8% / 3% | **res%100** (R2 0.83): res mod 100 in {0..4, 99} | **res%100** (R2 0.66): res mod 100 in {3} | (reads) | same | +0.70 |
| L29 gate c445 | 3% / 1% | **res%100** (R2 0.76): res mod 100 in {11, 51, 71} | **res** (R2 0.76): res in {11} | (reads) | same | +0.70 |
| L25 up c31 | 8% / 1% | **res%100** (R2 0.69): res mod 100 in {18..19} | **res** (R2 0.67): res in {18..19} | (reads) | same | +0.70 |
| L24 down c348 | 0 / 2% | off (on 0) | **res%100** (R2 0.85): res mod 100: no class above 0.5 (max 0.46) | - | same | +0.70 |
| L27 down c73 | 17% / 5% | **res** (R2 0.73): res in {2..15, 22..24, 32..35, 42..44, 52..54, 62..64, 73, 83, 93..94, 103..104, 113, 123, 133, 163} | unexplained (best R2 0.48) | - | same | +0.70 |
| L29 up c880 | 8% / 0 | **res%100** (R2 0.59): res mod 100 in {44..45, 54, 94..95} | off (on 0) | (reads) | same | +0.70 |
| L30 down c997 | 1% / 2% | **res%100** (R2 0.58): res mod 100 in {22} | unexplained (best R2 0.39) | - | same | +0.70 |
| L28 up c12 | 38% / 11% | **res** (R2 0.83): res in {10..13, 99..137, 139..141, 151, 153, 156..157, 159..160, 163, 176, 198..200} | **res** (R2 0.69): res in {-99, 6..18, 99} | (reads) | same | +0.70 |
| L20 gate c6 | 44% / 23% | **units(a,b)** (R2 0.63): a%10 in {0} -> b%10 in {1,2,3,4}; a%10 in {1} -> b%10 in {0,1,2,3,4}; a%10 in {2} -> b%10 in {0,1,2,3,4,9}; a%10 in {3} -> b%10 in {0,1,2,3,4,8,9}; a%10 in {4} -> b%10 in {0,1,2,3,4,7,8,9}; a%10 in {7} -> b%10 in {4}; a%10 in {8} -> b%10 in {3,4}; a%10 in {9} -> b%10 in {2,3,4} | unexplained (best R2 0.35) | (reads) | same | +0.70 |
| L23 down c310 | 4% / 0 | **res** (R2 0.66): res in {83, 93, 123, 143, 153, 163, 183, 193} [coarser: res mod 100 in {83, 93}, R2 0.82] | off (on 0) | - | same | +0.70 |
| L24 down c678 | 0 / 6% | off (on 0) | **res%100** (R2 0.51): res mod 100 in {3} | - | same | +0.70 |
| L28 up c277 | 5% / 2% | **res%50** (R2 0.72): res mod 50 in {0, 49} | unexplained (best R2 0.39) | (reads) | same | +0.70 |
| L26 up c155 | 10% / 2% | **res** (R2 0.77): res in {50, 89..92, 100, 109..110, 129..130, 149..150, 160, 169..170, 179..180, 189..192, 200} | unexplained (best R2 0.26) | (reads) | same | +0.70 |
| L23 o c234 (H22) | 2% / 5% | unexplained (best R2 0.38) | same | - | same | +0.70 |
| L25 down c31 | 16% / 7% | **res%100** (R2 0.86): res mod 100 in {1, 13..14} | **res%100** (R2 0.62): res mod 100 in {6..7} | res: mod50 +2% | - | +0.70 |
| L26 gate c142 | 7% / 1% | **res%100** (R2 0.68): res mod 100 in {15, 55..56, 59, 75, 95..96} | unexplained (best R2 0.46) | (reads) | same | +0.70 |
| L29 gate c159 | 8% / 21% | unexplained (best R2 0.30) | unexplained (best R2 0.27) | (reads) | same | +0.70 |
| L27 up c147 | 6% / 0 | **res%100** (R2 0.63): res mod 100 in {63..69} | off (on 0) | (reads) | same | +0.70 |
| L19 down c53 | 1% / 2% | unexplained (best R2 0.20) | unexplained (best R2 0.23) | - | same | +0.70 |
| L27 down c125 | 7% / 3% | **res%100** (R2 0.69): res mod 100 in {48..49, 96..99} [coarser: res mod 50 in {47..49}, R2 0.84] | unexplained (best R2 0.34) | - | same | +0.70 |
| L23 up c205 | 42% / 15% | **res** (R2 0.71): res in {2..35, 37, 39..41, 43..45, 47, 49..51, 53, 55, 57, 59, 63, 65, 67, 73, 75, 77..95, 97..101, 103, 105, 107, 183..191, 193, 197} | **tens(a,b)** (R2 0.47): a//10 in {0} -> b//10 in {0}; a//10 in {1} -> b//10 in {1}; a//10 in {6} -> b//10 in {6,7}; a//10 in {7} -> b//10 in {7,8}; a//10 in {8} -> b//10 in {0,8,9,10}; a//10 in {9} -> b//10 in {0,9,10}; a//10 in {10} -> b//10 in {0,1,10} | (reads) | same | +0.70 |
| L29 up c719 | 2% / 0 | **res%100** (R2 0.78): res mod 100: no class above 0.5 (max 0.43) | off (on 0) | (reads) | same | +0.69 |
| L31 up c864 | 9% / 1% | **res%100** (R2 0.52): res mod 100 in {34, 36..39, 43, 63, 83, 93} | unexplained (best R2 0.22) | (reads) | same | +0.69 |
| L25 up c48 | 33% / 11% | **res%100** (R2 0.75): res mod 100 in {0, 2..7, 12..16, 25, 43..46, 55..57, 65..67, 75..76, 85..86, 92..97} | **res** (R2 0.61): res in {3..7, 13..17, 93..96} | (reads) | same | +0.69 |
| L28 down c211 | 5% / 6% | **res** (R2 0.60): res in {2, 6..14, 16..17, 21, 31, 51, 71, 111, 171, 191} | unexplained (best R2 0.26) | - | same | +0.69 |
| L20 up c564 | 12% / 15% | unexplained (best R2 0.43) | unexplained (best R2 0.47) | (reads) | same | +0.69 |
| L24 gate c893 | 9% / 2% | **res%10** (R2 0.86): res mod 10 in {3} | **res** (R2 0.57): res in {13, 23} | (reads) | same | +0.69 |
| L30 up c642 | 2% / 0 | **res** (R2 0.56): res in {117..118} | off (on 0) | (reads) | same | +0.69 |
| L28 up c191 | 14% / 4% | **res%100** (R2 0.76): res mod 100 in {0..2, 6, 50..53} | **res%100** (R2 0.52): res mod 100 in {1} | (reads) | same | +0.69 |
| L26 down c258 | 8% / 1% | **res%100** (R2 0.76): res mod 100 in {13, 33, 53, 63, 73..74, 83, 93} | unexplained (best R2 0.48) | - | same | +0.69 |
| L29 up c0 | 37% / 2% | **res** (R2 0.69): res in {19, 29, 34, 39, 44..45, 49, 54, 59, 69, 71, 73..76, 79..87, 89, 94..95, 99, 102..105, 109, 114..116, 118..120, 123..125, 129, 134..135, 139, 143..145, 149, 154, 159, 163..164, 169, 174..175, 179, 183..185, 189, 194..195, 199} | unexplained (best R2 0.34) | (reads) | same | +0.69 |
| L29 down c159 | 11% / 19% | unexplained (best R2 0.41) | unexplained (best R2 0.27) | a: mod2 +2%; b: mod4 +7%, mod2 +2%; res: mod4 +4% | a: mod4 +10%, mod2 +5%; b: mod4 +11%, mod2 +5%; res: mod4 +5% | +0.69 |
| L20 down c20 | 54% / 48% | **res//10** (R2 0.81): (tens) res in {2..50, 99..150, 198..200} [coarser: res mod 100 in {0..50, 98..99}, R2 0.99] | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {0,1,2,3}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {0,1,2,3,9,10}; a//10 in {3} -> b//10 in {0,1,2,3,4,6,10}; a//10 in {4} -> b//10 in {0,1,2,3,4,6}; a//10 in {5} -> b//10 in {1,2,3,4,5}; a//10 in {6} -> b//10 in {2,3,4,5,6}; a//10 in {7} -> b//10 in {3,4,5,6,9}; a//10 in {8} -> b//10 in {4,5,6,7,8,9}; a//10 in {9} -> b//10 in {6,7,8,9,10}; a//10 in {10} -> b//10 in {0,5,6,7,8,9} | res: mod100 +8% | res: mod100 +4% | +0.69 |
| L30 down c214 | 7% / 1% | **res%100** (R2 0.77): res mod 100 in {17, 27, 37, 57, 77..78, 97} | **res** (R2 0.60): res in {17} | - | same | +0.69 |
| L31 gate c117 | 1% / 11% | unexplained (best R2 0.43) | **tens(a,b)** (R2 0.67): a//10 in {0} -> b//10 in {0,1,2,3,5}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2,3} -> b//10 in {0} | (reads) | same | +0.69 |
| L29 gate c865 | 0 / 3% | off (on 0) | **res%100** (R2 0.62): res mod 100: no class above 0.5 (max 0.44) | (reads) | same | +0.69 |
| L28 down c62 | 13% / 12% | **res%100** (R2 0.52): res mod 100 in {80, 86..87, 90..92, 96} | **tens(a,b)** (R2 0.68): a//10 in {7,10} -> b//10 in {1}; a//10 in {8} -> b//10 in {0,1,2}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,10} | - | same | +0.68 |
| L19 gate c6 | 64% / 44% | unexplained (best R2 0.14) | unexplained (best R2 0.18) | (reads) | same | +0.68 |
| L31 down c53 | 1% / 0 | **res** (R2 0.59): res in {97, 194..198} | off (on 0) | - | same | +0.68 |
| L28 down c794 | 4% / 7% | **res%100** (R2 0.69): res mod 100 in {21..24} | unexplained (best R2 0.46) | - | same | +0.68 |
| L27 gate c339 | 20% / 3% | **res%10** (R2 0.96): res mod 10 in {8..9} | unexplained (best R2 0.41) | (reads) | same | +0.68 |
| L23 up c878 | 2% / 1% | **res%100** (R2 0.72): res mod 100 in {96} | **res%100** (R2 0.63): res mod 100: no class above 0.5 (max 0.45) | (reads) | same | +0.68 |
| L19 down c3 | 27% / 98% | **res//10** (R2 0.63): (tens) res in {2..64, 200} | always | - | same | +0.68 |
| L26 down c194 | 11% / 1% | **res%100** (R2 0.77): res mod 100 in {18, 28, 38, 48, 57..59, 68, 78..79, 98} | unexplained (best R2 0.33) | - | same | +0.68 |
| L20 o c401 (H2) | 1% / 6% | unexplained (best R2 0.10) | unexplained (best R2 0.36) | - | same | +0.68 |
| L19 gate c106 | 11% / 3% | unexplained (best R2 0.48) | unexplained (best R2 0.33) | (reads) | same | +0.68 |
| L29 up c372 | 0 / 1% | off (on 0) | **res** (R2 0.72): res in {9, 13} | (reads) | same | +0.68 |
| L31 down c54 | 12% / 0 | **res** (R2 0.77): res in {76, 84, 86, 106, 114..116, 126, 134..136, 146, 154..156, 164..167, 175..176, 184..186, 196} | off (on 0) | - | same | +0.68 |
| L29 up c68 | 20% / 8% | **res%20** (R2 0.63): res mod 20 in {6..7, 16} [coarser: res mod 10 in {6..7}, R2 0.83] | unexplained (best R2 0.40) | (reads) | same | +0.68 |
| L31 up c169 | 15% / 0 | **res** (R2 0.60): res in {59, 61, 71, 79, 81, 85, 105, 109, 111, 115, 135, 145, 149..151, 155, 159..161, 165, 171, 175, 179..181, 185, 189, 195} | off (on 0) | (reads) | same | +0.68 |
| L27 down c723 | 4% / 0 | **res** (R2 0.69): res in {83..84, 179..189} | off (on 0) | - | same | +0.68 |
| L29 gate c577 | 1% / 2% | **res** (R2 0.78): res in {7, 17, 67, 167} | **res%100** (R2 0.61): res mod 100 in {7} | (reads) | same | +0.68 |
| L20 up c58 | 14% / 0 | **res//10** (R2 0.81): (tens) res in {146..181} | off (on 0) | (reads) | same | +0.68 |
| L29 up c445 | 3% / 31% | **res** (R2 0.61): res in {2..19, 23..24} | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {0,1}; a//10 in {1,2} -> b//10 in {0,1,2}; a//10 in {3} -> b//10 in {1,2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {4,5,6,7}; a//10 in {7,8} -> b//10 in {5,6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {5,6,7,8,9} | (reads) | same | +0.68 |
| L29 up c92 | 3% / 0 | **res%100** (R2 0.52): res mod 100 in {61, 67..68} | off (on 0) | (reads) | same | +0.68 |
| L28 up c22 | 15% / 15% | **res%100** (R2 0.83): res mod 100 in {0..1, 90..99} | **tens(a,b)** (R2 0.60): a//10 in {0} -> b//10 in {9,10}; a//10 in {1,2,3,4,8} -> b//10 in {9}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,9,10}; a//10 in {10} -> b//10 in {0,10} | (reads) | same | +0.68 |
| L28 down c124 | 3% / 0 | **res%100** (R2 0.74): res mod 100 in {52..53, 93} | off (on 0) | - | same | +0.68 |
| L28 up c416 | 4% / 7% | **res%100** (R2 0.55): res mod 100 in {8, 32} | **res%100** (R2 0.64): res mod 100: no class above 0.5 (max 0.45) | (reads) | same | +0.67 |
| L22 up c33 | 35% / 9% | **res%100** (R2 0.68): res mod 100 in {0..1, 12..14, 32, 51..55, 71..74, 87..99} | unexplained (best R2 0.50) | (reads) | same | +0.67 |
| L31 down c297 | 0 / 0 | off (on 0) | same | - | same | +0.67 |
| L26 down c180 | 1% / 2% | **res%100** (R2 0.67): res mod 100: no class above 0.5 (max 0.42) | **res%100** (R2 0.66): res mod 100: no class above 0.5 (max 0.44) | - | same | +0.67 |
| L28 down c433 | 5% / 0 | **res%100** (R2 0.66): res mod 100: no class above 0.5 (max 0.46) | off (on 0) | - | same | +0.67 |
| L28 down c522 | 2% / 1% | **res%100** (R2 0.74): res mod 100 in {17, 47} | **res** (R2 0.61): res in {17} | - | same | +0.67 |
| L23 down c269 | 6% / 0 | **res** (R2 0.80): res in {112..118} | off (on 0) | - | same | +0.67 |
| L29 down c904 | 4% / 0 | **res** (R2 0.80): res in {88, 98, 183..200} | off (on 0) | - | same | +0.67 |
| L20 down c14 | 65% / 12% | **res** (R2 0.66): res in {39, 73..104, 114, 118..165, 167..200} | **tens(a,b)** (R2 0.55): a//10 in {8} -> b//10 in {0,4,5,9,10}; a//10 in {9} -> b//10 in {0,1,4,5,6}; a//10 in {10} -> b//10 in {0,1,2,5,6,7,10} | b: mod50 +2%; res: mod50 +6%, mod25 +15% | - | +0.67 |
| L30 gate c633 | 4% / 1% | **res%50** (R2 0.70): res mod 50 in {9, 19} | **res** (R2 0.75): res in {19} | (reads) | same | +0.67 |
| L31 gate c212 | 2% / 0 | **res** (R2 0.70): res in {97..98, 193..199} | off (on 0) | (reads) | same | +0.67 |
| L31 up c787 | 44% / 10% | **res//10** (R2 0.67): (tens) res in {2..13, 15..18, 20..25, 27, 30, 32, 50..100, 190} | **tens(a,b)** (R2 0.53): a//10 in {6,7} -> b//10 in {0}; a//10 in {8} -> b//10 in {0,1}; a//10 in {9} -> b//10 in {0,1,3,10}; a//10 in {10} -> b//10 in {0,1,2,3,4} | (reads) | same | +0.67 |
| L31 down c242 | 5% / 1% | **res** (R2 0.78): res in {87..89, 176..180, 185..200} | unexplained (best R2 0.34) | - | same | +0.67 |
| L21 up c125 | 17% / 17% | **res** (R2 0.71): res in {2..4, 38..40, 44, 48..50, 54, 134, 138..150, 154..155, 158..160} [coarser: res mod 100 in {38..45, 48..50, 54..55, 58..60}, R2 0.80] | **tens(a,b)** (R2 0.55): a//10 in {4} -> b//10 in {0,10}; a//10 in {5} -> b//10 in {0,1,10}; a//10 in {6} -> b//10 in {0,1,2,10}; a//10 in {7} -> b//10 in {0,1,2,3}; a//10 in {8} -> b//10 in {3,4}; a//10 in {10} -> b//10 in {4,5,6,10} | (reads) | same | +0.67 |
| L21 down c206 | 6% / 0 | **res//10** (R2 0.75): (tens) res in {17..18, 22..39} | off (on 0) | - | same | +0.67 |
| L29 up c187 | 4% / 1% | **res%100** (R2 0.77): res mod 100 in {33, 53, 83} | unexplained (best R2 0.48) | (reads) | same | +0.67 |
| L29 gate c320 | 0 / 11% | off (on 0) | unexplained (best R2 0.49) | (reads) | same | +0.66 |
| L31 down c306 | 6% / 19% | unexplained (best R2 0.33) | unexplained (best R2 0.42) | - | same | +0.66 |
| L29 gate c168 | 2% / 0 | **res%100** (R2 0.77): res mod 100: no class above 0.5 (max 0.45) | off (on 0) | (reads) | same | +0.66 |
| L30 down c535 | 5% / 4% | **res%100** (R2 0.76): res mod 100 in {0, 94..99} | **a** (R2 0.69): a in {97..99} | - | same | +0.66 |
| L27 up c24 | 8% / 3% | **res%100** (R2 0.74): res mod 100 in {38..43, 50, 60, 80} | unexplained (best R2 0.41) | (reads) | same | +0.66 |
| L28 down c1010 | 2% / 3% | **res** (R2 0.85): res in {7..9, 11..20} | **res** (R2 0.51): res in {15..17} | - | same | +0.66 |
| L20 up c305 | 37% / 10% | **res** (R2 0.66): res in {31..32, 36..37, 41..42, 50..52, 56..63, 66..68, 70..102, 171..172, 176..177, 180..182, 186..187, 190..192} | unexplained (best R2 0.33) | (reads) | same | +0.66 |
| L31 gate c391 | 12% / 0 | **res** (R2 0.75): res in {34, 83..85, 114..117, 134..135, 154..155, 164..167, 174..175, 183..185} | off (on 0) | (reads) | same | +0.66 |
| L31 down c546 | 6% / 0 | **res** (R2 0.76): res in {85, 114..115, 135, 145, 155, 165, 175, 185, 195} | off (on 0) | - | same | +0.66 |
| L31 down c444 | 5% / 0 | **res** (R2 0.82): res in {81, 111, 121, 131, 141, 151, 161, 171, 181} | off (on 0) | - | same | +0.66 |
| L29 down c213 | 3% / 6% | **res%100** (R2 0.64): res mod 100: no class above 0.5 (max 0.50) | **tens(a,b)** (R2 0.62): a//10 in {9} -> b//10 in {0,2,3,4,5}; a//10 in {10} -> b//10 in {0} | - | same | +0.66 |
| L22 down c44 | 18% / 0 | **res%100** (R2 0.67): res mod 100 in {52, 60..76} | off (on 0) | - | same | +0.66 |
| L26 down c23 | 5% / 5% | **res%100** (R2 0.75): res mod 100: no class above 0.5 (max 0.45) | **res%100** (R2 0.68): res mod 100: no class above 0.5 (max 0.44) | - | same | +0.66 |
| L30 up c214 | 20% / 3% | **res%100** (R2 0.73): res mod 100 in {16..18, 27, 37, 47, 57..58, 67, 74..79, 87, 97} | **res** (R2 0.52): res in {17..18, 57, 77} | (reads) | same | +0.66 |
| L29 gate c365 | 2% / 1% | **res%100** (R2 0.87): res mod 100 in {38..39} | unexplained (best R2 0.47) | (reads) | same | +0.66 |
| L27 up c125 | 6% / 1% | **res%100** (R2 0.73): res mod 100 in {48..49, 69, 89, 98..99} [coarser: res mod 50 in {48..49}, R2 0.85] | unexplained (best R2 0.24) | (reads) | same | +0.66 |
| L29 up c140 | 8% / 1% | **res** (R2 0.59): res in {26, 86, 123..126, 145..146, 156, 163..166, 174..176, 186} | unexplained (best R2 0.43) | (reads) | same | +0.65 |
| L27 gate c533 | 7% / 0 | **res%100** (R2 0.80): res mod 100 in {13, 33, 53, 63, 73, 93} [coarser: res mod 20 in {13}, R2 0.83] | off (on 0) | (reads) | same | +0.65 |
| L29 down c71 | 9% / 2% | **res%100** (R2 0.71): res mod 100 in {14, 34, 44..45, 54, 93..95} | unexplained (best R2 0.22) | - | same | +0.65 |
| L30 down c251 | 9% / 0 | **res** (R2 0.80): res in {61, 81..82, 111..112, 121..123, 141..142, 161..162, 181..182} | off (on 0) | - | same | +0.65 |
| L24 o c503 (H17) | 1% / 1% | **a** (R2 0.89): a in {90} | **a** (R2 0.82): a in {90} | - | same | +0.65 |
| L22 down c134 | 20% / 1% | **tens(a,b)** (R2 0.70): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {9,10}; a//10 in {2} -> b//10 in {8,9,10}; a//10 in {3} -> b//10 in {7,8,9,10}; a//10 in {4} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {6}; a//10 in {6} -> b//10 in {4,5}; a//10 in {7} -> b//10 in {3,4,5}; a//10 in {8} -> b//10 in {2,3}; a//10 in {9} -> b//10 in {1,2,3,10}; a//10 in {10} -> b//10 in {0,1,2,3} | unexplained (best R2 0.20) | - | same | +0.65 |
| L31 down c1023 | 8% / 4% | **res** (R2 0.76): res in {13..15, 71, 90..91, 169..174, 179..200} | unexplained (best R2 0.36) | - | same | +0.65 |
| L28 down c277 | 4% / 2% | **res%50** (R2 0.69): res mod 50 in {0, 49} | unexplained (best R2 0.20) | - | same | +0.65 |
| L29 down c964 | 2% / 0 | **res%100** (R2 0.76): res mod 100 in {49..50} | off (on 0) | - | same | +0.65 |
| L20 down c180 | 11% / 1% | **res** (R2 0.65): res in {121..137} | unexplained (best R2 0.24) | res: mod25 +4% | - | +0.65 |
| L30 gate c594 | 2% / 0 | **res%100** (R2 0.55): res mod 100: no class above 0.5 (max 0.49) | off (on 0) | (reads) | same | +0.65 |
| L29 up c904 | 1% / 0 | **res%100** (R2 0.64): res mod 100: no class above 0.5 (max 0.34) | off (on 0) | (reads) | same | +0.65 |
| L29 down c48 | 12% / 2% | **res%100** (R2 0.75): res mod 100 in {1, 10..11, 50, 70} | **res%100** (R2 0.68): res mod 100 in {10..11} | - | same | +0.64 |
| L22 down c129 | 10% / 1% | **res%10** (R2 0.94): res mod 10 in {6} | **res%100** (R2 0.60): res mod 100: no class above 0.5 (max 0.47) | - | same | +0.64 |
| L23 down c7 | 39% / 14% | **res//10** (R2 0.75): (tens) res in {7..8, 15, 18, 20..88} | unexplained (best R2 0.41) | - | same | +0.64 |
| L20 o c396 (H2) | 2% / 11% | unexplained (best R2 0.48) | unexplained (best R2 0.49) | - | same | +0.64 |
| L31 down c544 | 6% / 13% | **tens(a,b)** (R2 0.54): a//10 in {0} -> b//10 in {1}; a//10 in {1,2} -> b//10 in {0,1} | unexplained (best R2 0.41) | - | same | +0.64 |
| L20 up c641 | 4% / 4% | unexplained (best R2 0.45) | unexplained (best R2 0.25) | (reads) | same | +0.64 |
| L28 gate c22 | 15% / 15% | **res%100** (R2 0.77): res mod 100 in {0..1, 90..99} | **tens(a,b)** (R2 0.62): a//10 in {0} -> b//10 in {9,10}; a//10 in {1,2,3,4,8} -> b//10 in {9}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,9,10}; a//10 in {10} -> b//10 in {0,10} | (reads) | same | +0.64 |
| L21 down c133 | 15% / 1% | **res** (R2 0.70): res in {65..75, 161..192} | unexplained (best R2 0.16) | - | same | +0.64 |
| L27 up c200 | 2% / 0 | **res%50** (R2 0.78): res mod 50 in {49} | off (on 0) | (reads) | same | +0.64 |
| L30 gate c177 | 13% / 1% | **res%100** (R2 0.76): res mod 100 in {55..60, 66, 92..97, 99} | unexplained (best R2 0.26) | (reads) | same | +0.63 |
| L31 down c430 | 0 / 0 | off (on 0) | same | - | same | +0.63 |
| L23 down c60 | 4% / 0 | **res** (R2 0.88): res in {120, 130, 140, 150, 160, 170, 180, 190, 200} | off (on 0) | - | same | +0.63 |
| L30 gate c399 | 13% / 7% | **res//10** (R2 0.73): (tens) res in {99..110, 198..200} | **res%100** (R2 0.73): res mod 100 in {2..8} | (reads) | same | +0.63 |
| L30 gate c320 | 17% / 1% | **res** (R2 0.56): res in {48, 52, 58, 68, 78, 82, 88, 96, 98, 102, 108, 118, 122, 124, 126, 128, 132, 138, 144, 148, 152, 154, 156, 158, 164, 168, 172, 178, 182, 188, 196, 198} | unexplained (best R2 0.16) | (reads) | same | +0.63 |
| L31 gate c260 | 37% / 2% | **res%100** (R2 0.49): res mod 100 in {1..4, 7, 17, 27, 37..38, 47, 50, 57..58, 73..75, 77..79, 83, 87..90, 92..98} | unexplained (best R2 0.24) | (reads) | same | +0.63 |
| L29 down c68 | 10% / 2% | **res%100** (R2 0.75): res mod 100 in {6..7, 9, 46, 56, 66, 86} | **res%100** (R2 0.66): res mod 100 in {6} | - | same | +0.63 |
| L20 gate c3 | 53% / 24% | **units(a,b)** (R2 0.88): a%10 in {0} -> b%10 in {5,6,7,8,9}; a%10 in {1} -> b%10 in {5,6,7,8}; a%10 in {2} -> b%10 in {5,6,7}; a%10 in {3} -> b%10 in {5,6}; a%10 in {4} -> b%10 in {5}; a%10 in {5} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {6} -> b%10 in {0,1,2,3,5,6,7,8,9}; a%10 in {7} -> b%10 in {0,1,2,5,6,7,8,9}; a%10 in {8} -> b%10 in {0,1,5,6,7,8,9}; a%10 in {9} -> b%10 in {0,5,6,7,8,9} | unexplained (best R2 0.41) | (reads) | same | +0.63 |
| L22 down c123 | 7% / 0 | **res%100** (R2 0.78): res mod 100 in {0..1, 96..99} | off (on 0) | - | same | +0.63 |
| L29 up c448 | 7% / 1% | **res%100** (R2 0.69): res mod 100 in {45, 67..68, 85} | **res** (R2 0.54): res in {25} | (reads) | same | +0.63 |
| L31 down c500 | 2% / 0 | **res** (R2 0.78): res in {111, 131, 151} | off (on 0) | - | same | +0.63 |
| L28 down c58 | 36% / 3% | **res** (R2 0.73): res in {15, 24..25, 35, 45, 65, 74..75, 84..85, 90, 94..95, 99, 104..105, 107..111, 114..115, 117, 119..127, 131, 134..135, 139..141, 144..145, 149..151, 155, 165, 169..171, 174..175, 179..181, 184..191, 194..195} | unexplained (best R2 0.37) | - | same | +0.62 |
| L19 down c106 | 10% / 2% | **units(a,b)** (R2 0.59): a%10 in {0} -> b%10 in {4}; a%10 in {1} -> b%10 in {3,4}; a%10 in {3} -> b%10 in {1}; a%10 in {5} -> b%10 in {9}; a%10 in {6} -> b%10 in {8}; a%10 in {7} -> b%10 in {7}; a%10 in {8} -> b%10 in {6}; a%10 in {9} -> b%10 in {5} | unexplained (best R2 0.49) | - | same | +0.62 |
| L24 down c209 | 0 / 4% | off (on 0) | unexplained (best R2 0.27) | - | same | +0.62 |
| L24 gate c71 | 83% / 93% | **res%100** (R2 0.62): res mod 100 in {0, 2..35, 38, 40, 43..85, 90, 93, 95, 99} | unexplained (best R2 0.22) | (reads) | same | +0.62 |
| L29 up c232 | 34% / 4% | **res** (R2 0.70): res in {49, 58..60, 69..71, 78..79, 88..92, 96..102, 108..111, 118..121, 128..131, 138..139, 148..152, 157..162, 168..172, 178..182, 188..193, 196..199} | unexplained (best R2 0.35) | (reads) | same | +0.62 |
| L23 o c187 (H7) | 1% / 4% | **a** (R2 0.70): a in {98} | **a** (R2 0.64): a in {97..99} | - | same | +0.62 |
| L22 gate c56 | 27% / 4% | **res%100** (R2 0.60): res mod 100 in {16, 51..61, 63..66, 71..77} | unexplained (best R2 0.24) | (reads) | same | +0.62 |
| L22 down c38 | 0 / 2% | off (on 0) | **tens(a,b)** (R2 0.50): a//10 in {4} -> b//10 in {0} | - | same | +0.62 |
| L27 gate c6 | 62% / 2% | **res//10** (R2 0.86): (tens) res in {86..87, 89..199} | unexplained (best R2 0.41) | (reads) | same | +0.62 |
| L31 gate c171 | 5% / 0 | **res** (R2 0.66): res in {68, 107..109, 161..164, 167..169} | off (on 0) | (reads) | same | +0.62 |
| L28 down c430 | 0 / 1% | off (on 0) | **res%100** (R2 0.77): res mod 100: no class above 0.5 (max 0.44) | - | same | +0.61 |
| L25 gate c92 | 84% / 6% | **res** (R2 0.84): res in {21, 31, 41, 45, 51, 55, 58, 60..66, 68..76, 78, 80..200} | unexplained (best R2 0.40) | (reads) | same | +0.61 |
| L23 down c32 | 10% / 1% | unexplained (best R2 0.42) | unexplained (best R2 0.16) | - | same | +0.61 |
| L29 down c35 | 4% / 33% | **res** (R2 0.70): res in {2, 30..31, 40..41, 60..61, 70, 90} | unexplained (best R2 0.31) | - | same | +0.61 |
| L28 down c22 | 17% / 15% | **res%100** (R2 0.72): res mod 100 in {0..1, 7, 17, 37, 47, 57, 87, 91..99} | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {9,10}; a//10 in {1,2,3,4,8} -> b//10 in {9}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,9,10}; a//10 in {10} -> b//10 in {0} | - | same | +0.61 |
| L21 up c341 | 1% / 3% | unexplained (best R2 0.09) | unexplained (best R2 0.38) | (reads) | same | +0.61 |
| L28 down c191 | 10% / 2% | **res%100** (R2 0.84): res mod 100 in {0..1, 51..53} | unexplained (best R2 0.47) | - | same | +0.61 |
| L23 gate c784 | 10% / 2% | **res%10** (R2 0.96): res mod 10 in {6} | **res** (R2 0.74): res in {6, 16, 96} | (reads) | same | +0.61 |
| L24 down c65 | 3% / 8% | unexplained (best R2 0.23) | unexplained (best R2 0.25) | - | same | +0.61 |
| L31 up c247 | 13% / 1% | **res** (R2 0.67): res in {85..89, 150..151, 154..158, 160, 167, 170..190, 194..198, 200} | unexplained (best R2 0.32) | (reads) | same | +0.60 |
| L29 down c321 | 8% / 1% | **res%100** (R2 0.72): res mod 100 in {5, 15, 25, 35, 45, 64..66, 85, 95} | **res** (R2 0.53): res in {5} | - | same | +0.60 |
| L24 gate c65 | 2% / 6% | unexplained (best R2 0.29) | unexplained (best R2 0.27) | (reads) | same | +0.60 |
| L25 up c74 | 37% / 7% | **res//10** (R2 0.67): (tens) res in {14..16, 18, 20..90, 123} | unexplained (best R2 0.32) | (reads) | same | +0.60 |
| L31 up c419 | 24% / 0 | **res** (R2 0.86): res in {96..120, 188..190, 192, 194..199} | off (on 0) | (reads) | same | +0.60 |
| L28 down c588 | 1% / 3% | unexplained (best R2 0.40) | **res%100** (R2 0.50): res mod 100: no class above 0.5 (max 0.48) | - | same | +0.60 |
| L19 gate c32 | 25% / 5% | **units(a,b)** (R2 0.79): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} | unexplained (best R2 0.22) | (reads) | same | +0.60 |
| L28 down c298 | 11% / 5% | **res%100** (R2 0.71): res mod 100 in {0, 49, 89..99} | unexplained (best R2 0.44) | - | same | +0.60 |
| L27 down c6 | 61% / 2% | **res//10** (R2 0.87): (tens) res in {90..199} | unexplained (best R2 0.39) | a: mod100 +2%, mod50 +3%; b: mod100 +3%, mod50 +2%; res: mod100 +3% | - | +0.60 |
| L31 down c612 | 3% / 0 | **res** (R2 0.63): res in {68, 162..164, 167..169, 182..184, 192, 194, 196..198} | off (on 0) | - | same | +0.60 |
| L26 up c258 | 10% / 3% | **res%100** (R2 0.69): res mod 100 in {13..14, 34, 53..54, 73..75, 84, 93..94} | unexplained (best R2 0.40) | (reads) | same | +0.60 |
| L28 down c310 | 6% / 0 | **res** (R2 0.67): res in {107, 117, 127, 147, 157..158, 160..164, 167} | off (on 0) | - | same | +0.60 |
| L21 gate c7 | 1% / 3% | unexplained (best R2 0.46) | unexplained (best R2 0.50) | (reads) | same | +0.60 |
| L31 down c754 | 26% / 62% | unexplained (best R2 0.46) | unexplained (best R2 0.39) | - | same | +0.60 |
| L30 down c293 | 6% / 0 | **res** (R2 0.74): res in {59, 108..109, 119, 129, 158..160, 169} | off (on 0) | - | same | +0.60 |
| L31 down c273 | 0 / 0 | off (on 0) | same | - | same | +0.60 |
| L31 gate c153 | 4% / 0 | **res%100** (R2 0.54): res mod 100 in {1, 81} | off (on 0) | (reads) | same | +0.60 |
| L22 up c11 | 33% / 6% | **res%20** (R2 0.90): res mod 20 in {0..4, 18..19} | unexplained (best R2 0.44) | (reads) | same | +0.60 |
| L27 up c269 | 3% / 1% | **res%100** (R2 0.67): res mod 100 in {1, 61, 71} | **res%100** (R2 0.76): res mod 100: no class above 0.5 (max 0.47) | (reads) | same | +0.59 |
| L27 gate c823 | 2% / 1% | **res%100** (R2 0.76): res mod 100 in {97..99} | **cmp(a,b)** (R2 0.57): cmp(a,b)=-1: 0.01, cmp(a,b)=0: 0.89, cmp(a,b)=1: 0.00 | (reads) | same | +0.59 |
| L28 up c517 | 8% / 46% | **res%10** (R2 0.71): res mod 10 in {5} | unexplained (best R2 0.35) | (reads) | same | +0.59 |
| L27 down c599 | 3% / 0 | **res** (R2 0.69): res in {119..121, 125} | off (on 0) | - | same | +0.59 |
| L29 gate c171 | 42% / 7% | **res%10** (R2 0.70): res mod 10 in {0, 2, 4, 6, 8} | unexplained (best R2 0.34) | (reads) | same | +0.59 |
| L20 down c378 | 18% / 6% | **units(a,b)** (R2 0.64): a%10 in {0} -> b%10 in {3}; a%10 in {5} -> b%10 in {7,8,9}; a%10 in {6} -> b%10 in {6,7,8,9}; a%10 in {7,8} -> b%10 in {5,6,7}; a%10 in {9} -> b%10 in {5,6} | unexplained (best R2 0.41) | - | same | +0.59 |
| L22 o c246 (H3) | 3% / 6% | unexplained (best R2 0.49) | **a** (R2 0.69): a in {37, 47, 57, 67, 97} | - | same | +0.59 |
| L31 gate c145 | 9% / 44% | unexplained (best R2 0.47) | unexplained (best R2 0.33) | (reads) | same | +0.59 |
| L19 down c85 | 6% / 3% | unexplained (best R2 0.15) | unexplained (best R2 0.08) | - | same | +0.59 |
| L31 down c315 | 5% / 0 | **res** (R2 0.77): res in {97..99, 149..150, 187..199} | off (on 0) | - | same | +0.58 |
| L25 up c55 | 0 / 26% | off (on 0) | **tens(a,b)** (R2 0.53): a//10 in {1,2} -> b//10 in {1}; a//10 in {3} -> b//10 in {2}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {4,7}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7,8} -> b//10 in {6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {7,8,9} | (reads) | same | +0.58 |
| L27 down c139 | 10% / 2% | **res%100** (R2 0.76): res mod 100 in {1, 21, 31, 41, 51, 61, 71, 81} | **res%100** (R2 0.59): res mod 100: no class above 0.5 (max 0.46) | - | same | +0.58 |
| L22 down c333 | 10% / 0 | **res%10** (R2 0.91): res mod 10 in {4} | off (on 0) | - | same | +0.58 |
| L21 gate c247 | 1% / 2% | unexplained (best R2 0.40) | **res%100** (R2 0.66): res mod 100 in {0} | (reads) | same | +0.58 |
| L29 down c171 | 33% / 1% | **res** (R2 0.81): res in {47, 57, 59, 61, 63, 67, 79, 81, 83, 85, 87, 89, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183, 185, 187, 189, 191, 193, 195, 197} | unexplained (best R2 0.15) | res: mod2 +7% | - | +0.58 |
| L26 gate c393 | 9% / 4% | **res//10** (R2 0.86): (tens) res in {2..41} | unexplained (best R2 0.38) | (reads) | same | +0.58 |
| L24 down c1 | 75% / 90% | **res** (R2 0.57): res in {2..16, 18, 21..25, 29..85, 89..90, 93, 100..136, 138, 140, 143..145, 150, 154, 160, 163..166, 170, 173..176, 180, 190, 199..200} | unexplained (best R2 0.17) | - | same | +0.58 |
| L26 up c552 | 0 / 19% | off (on 0) | **tens(a,b)** (R2 0.57): a//10 in {5} -> b//10 in {5,6,7}; a//10 in {6} -> b//10 in {6,7,8,9,10}; a//10 in {7} -> b//10 in {7,8,9,10}; a//10 in {8} -> b//10 in {8,9,10}; a//10 in {9,10} -> b//10 in {9,10} | (reads) | same | +0.58 |
| L20 up c153 | 10% / 0 | **res//10** (R2 0.71): (tens) res in {145, 149..170, 173..174} | off (on 0) | (reads) | same | +0.57 |
| L31 up c664 | 12% / 0 | **res** (R2 0.77): res in {61, 81..84, 111, 117..124, 161, 181..182} | off (on 0) | (reads) | same | +0.57 |
| L28 gate c377 | 2% / 0 | **res** (R2 0.86): res in {130, 134..135} | off (on 0) | (reads) | same | +0.57 |
| L30 down c317 | 3% / 0 | **res** (R2 0.67): res in {103..104, 143, 163, 183} | off (on 0) | - | same | +0.57 |
| L28 up c83 | 10% / 1% | **res%100** (R2 0.74): res mod 100 in {8, 18, 28, 38, 58, 68, 77..79, 98} | unexplained (best R2 0.18) | (reads) | same | +0.57 |
| L24 down c313 | 0 / 27% | off (on 0) | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {0,1}; a//10 in {1} -> b//10 in {1}; a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {4,5,7}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {5,6,7,8,9,10} | - | same | +0.57 |
| L30 gate c70 | 5% / 0 | **res** (R2 0.62): res in {118, 128, 131, 137..138, 171, 177..178, 188} | off (on 0) | (reads) | same | +0.56 |
| L27 down c32 | 8% / 1% | **res%100** (R2 0.79): res mod 100: no class above 0.5 (max 0.49) | unexplained (best R2 0.18) | - | same | +0.56 |
| L31 gate c164 | 7% / 0 | **res** (R2 0.73): res in {49, 59, 79, 89, 99, 119..120, 129, 139, 149, 159, 179, 189} | off (on 0) | (reads) | same | +0.56 |
| L29 gate c581 | 1% / 1% | **res%100** (R2 0.74): res mod 100 in {33} | unexplained (best R2 0.47) | (reads) | same | +0.56 |
| L31 down c307 | 11% / 44% | **res//10** (R2 0.64): (tens) res in {2..19, 21..41} | **tens(a,b)** (R2 0.67): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,3,4,5,6,7,8,9}; a//10 in {2} -> b//10 in {0,1,2,3,4,5,6}; a//10 in {3} -> b//10 in {0,3,4,5,6}; a//10 in {4} -> b//10 in {0,4,5,6}; a//10 in {5} -> b//10 in {0,1,6}; a//10 in {6} -> b//10 in {0,7} | - | a: mod100 +9%, mod50 +8%, mod25 +7%, mod20 +6%, mod10 +4%; b: mod50 +2%, mod25 +3%, mod20 +2% | +0.56 |
| L29 gate c183 | 2% / 0 | **res** (R2 0.62): res in {34, 149..152} | off (on 0) | (reads) | same | +0.56 |
| L31 gate c469 | 25% / 45% | unexplained (best R2 0.18) | unexplained (best R2 0.32) | (reads) | same | +0.56 |
| L24 up c65 | 98% / 67% | always | unexplained (best R2 0.24) | (reads) | same | +0.56 |
| L31 up c217 | 6% / 0 | **res** (R2 0.80): res in {111..117} | off (on 0) | (reads) | same | +0.56 |
| L27 up c657 | 6% / 0 | **res%100** (R2 0.79): res mod 100: no class above 0.5 (max 0.47) | off (on 0) | (reads) | same | +0.56 |
| L27 down c365 | 3% / 0 | **res** (R2 0.66): res in {97, 107, 147, 157..158} | off (on 0) | - | same | +0.56 |
| L31 down c85 | 2% / 1% | **res** (R2 0.57): res in {5, 150, 155, 160, 170, 175} | **res%100** (R2 0.63): res mod 100: no class above 0.5 (max 0.47) | - | same | +0.55 |
| L30 down c47 | 63% / 71% | unexplained (best R2 0.28) | unexplained (best R2 0.35) | - | same | +0.55 |
| L28 up c296 | 7% / 3% | **res** (R2 0.75): res in {30, 40, 60, 80, 120, 130, 140, 157, 159..163, 180} [coarser: res mod 100 in {20, 30, 40, 57, 59..60, 80}, R2 0.89] | **res%100** (R2 0.52): res mod 100 in {0} | (reads) | same | +0.55 |
| L25 gate c98 | 2% / 10% | **res%100** (R2 0.72): res mod 100 in {91, 93} | unexplained (best R2 0.39) | (reads) | same | +0.55 |
| L26 o c379 (H14) | 3% / 4% | **a** (R2 0.61): a in {92..94} | **a** (R2 0.81): a in {92..94} | - | same | +0.55 |
| L28 gate c433 | 11% / 9% | **res%100** (R2 0.89): res mod 100 in {0, 89..99} | **tens(a,b)** (R2 0.60): a//10 in {0} -> b//10 in {9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5}; a//10 in {10} -> b//10 in {0} | (reads) | same | +0.55 |
| L27 up c69 | 20% / 2% | **res%100** (R2 0.72): res mod 100 in {8, 18, 27..28, 38, 47..49, 57..58, 67..69, 78, 87..88} | unexplained (best R2 0.37) | (reads) | same | +0.55 |
| L21 down c212 | 1% / 8% | unexplained (best R2 0.20) | **tens(a,b)** (R2 0.58): a//10 in {7} -> b//10 in {0,1,2,3}; a//10 in {8} -> b//10 in {3} | - | same | +0.54 |
| L24 o c23 (H13) | 68% / 83% | **tens(a,b)** (R2 0.56): a//10 in {0,1} -> b//10 in {6,7,8,9}; a//10 in {2,3} -> b//10 in {5,6,7,8,9}; a//10 in {4} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {1,2,3,4,5,7,8,9,10}; a//10 in {7,8,9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | unexplained (best R2 0.40) | - | same | +0.54 |
| L27 up c835 | 5% / 23% | **res%100** (R2 0.86): res mod 100 in {80..84} | unexplained (best R2 0.49) | (reads) | same | +0.54 |
| L20 o c292 (H2) | 3% / 15% | unexplained (best R2 0.16) | unexplained (best R2 0.37) | - | a: mod4 +3%; b: mod4 +2% | +0.54 |
| L31 down c72 | 0 / 1% | off (on 0) | unexplained (best R2 0.29) | - | same | +0.54 |
| L28 gate c49 | 24% / 75% | **res** (R2 0.72): res in {2, 6, 8..16, 18, 26, 33..66, 68, 76, 81, 86, 96, 141, 146, 156, 161, 166, 176, 181, 196, 198} | unexplained (best R2 0.36) | (reads) | same | +0.54 |
| L24 up c76 | 5% / 82% | unexplained (best R2 0.44) | unexplained (best R2 0.26) | (reads) | same | +0.54 |
| L24 gate c46 | 13% / 12% | **res%50** (R2 0.79): res mod 50 in {3, 13, 23, 33, 43} | **res** (R2 0.58): res in {-98, 0, 3, 8, 10, 13, 18, 20, 23, 28, 30, 33, 38, 43, 83, 88, 93, 98..99} | (reads) | same | +0.54 |
| L27 up c823 | 10% / 1% | **res** (R2 0.80): res in {73, 108..117, 133, 173} | unexplained (best R2 0.36) | (reads) | same | +0.54 |
| L23 up c29 | 71% / 9% | **res//10** (R2 0.91): (tens) res in {79..200} | **tens(a,b)** (R2 0.55): a//10 in {0,1,2} -> b//10 in {9,10}; a//10 in {9} -> b//10 in {0,1,9}; a//10 in {10} -> b//10 in {0,1,2,10} | (reads) | same | +0.54 |
| L28 up c746 | 3% / 2% | **res%100** (R2 0.88): res mod 100 in {30, 80, 90} [coarser: res mod 50 in {30}, R2 0.83] | **res** (R2 0.57): res in {0, 30, 90} | (reads) | same | +0.54 |
| L29 down c115 | 1% / 25% | unexplained (best R2 0.36) | **tens(a,b)** (R2 0.53): a//10 in {0} -> b//10 in {0}; a//10 in {1} -> b//10 in {1}; a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9,10}; a//10 in {8} -> b//10 in {6,7,8,9,10}; a//10 in {9} -> b//10 in {7,9,10}; a//10 in {10} -> b//10 in {6,7,8,9} | - | res: mod100 +2% | +0.54 |
| L21 down c279 | 7% / 0 | **res** (R2 0.64): res in {133..138, 157, 174..177, 193..196} | off (on 0) | - | same | +0.53 |
| L27 down c68 | 26% / 1% | **tens(a,b)** (R2 0.50): a//10 in {0} -> b//10 in {6}; a//10 in {1} -> b//10 in {4,5,6,7}; a//10 in {2} -> b//10 in {3,4,5,6}; a//10 in {3} -> b//10 in {1,2,3,4,5}; a//10 in {4,5} -> b//10 in {1,2,3}; a//10 in {6} -> b//10 in {1,2}; a//10 in {7} -> b//10 in {1} | unexplained (best R2 0.45) | - | same | +0.53 |
| L26 gate c123 | 2% / 4% | unexplained (best R2 0.09) | unexplained (best R2 0.22) | (reads) | same | +0.53 |
| L29 gate c794 | 6% / 0 | **res%100** (R2 0.92): res mod 100 in {1} | off (on 0) | (reads) | same | +0.53 |
| L22 down c297 | 1% / 19% | unexplained (best R2 0.18) | **tens(a,b)** (R2 0.61): a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {3,4,5}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {4,5,6,7}; a//10 in {9} -> b//10 in {7,8}; a//10 in {10} -> b//10 in {3,4,5,6,7,8,9} | - | same | +0.53 |
| L19 up c28 | 46% / 18% | **res//10** (R2 0.73): (tens) res in {100..171, 174..176, 178, 196, 198, 200} | **tens(a,b)** (R2 0.67): a//10 in {4} -> b//10 in {1}; a//10 in {5} -> b//10 in {1,2,3,4}; a//10 in {6} -> b//10 in {2,3,4,5}; a//10 in {7} -> b//10 in {3,4,5}; a//10 in {8} -> b//10 in {4,5,6}; a//10 in {9} -> b//10 in {6,7}; a//10 in {10} -> b//10 in {5,6,7,8} | (reads) | same | +0.53 |
| L28 down c416 | 1% / 9% | **res%100** (R2 0.77): res mod 100 in {9} | **res%100** (R2 0.63): res mod 100: no class above 0.5 (max 0.47) | - | same | +0.53 |
| L29 down c14 | 5% / 0 | **res** (R2 0.65): res in {126, 130, 140, 142, 146..148, 160, 170, 176..178, 180, 182} | off (on 0) | - | same | +0.53 |
| L31 up c572 | 14% / 32% | **tens(a,b)** (R2 0.51): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,10}; a//10 in {1} -> b//10 in {0,1}; a//10 in {2,3} -> b//10 in {0}; a//10 in {10} -> b//10 in {10} | **tens(a,b)** (R2 0.57): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,3,4,5,6,7,8,10}; a//10 in {2} -> b//10 in {0,1,2,3,4,10}; a//10 in {3} -> b//10 in {0,4,10}; a//10 in {4,5} -> b//10 in {0}; a//10 in {6} -> b//10 in {0,6,7}; a//10 in {10} -> b//10 in {10} | (reads) | same | +0.53 |
| L31 up c472 | 2% / 0 | **res** (R2 0.69): res in {98, 117..118} | off (on 0) | (reads) | same | +0.53 |
| L21 down c267 | 7% / 0 | **tens(a,b)** (R2 0.69): a//10 in {4} -> b//10 in {4}; a//10 in {7} -> b//10 in {10}; a//10 in {8} -> b//10 in {9,10}; a//10 in {9} -> b//10 in {8,9,10}; a//10 in {10} -> b//10 in {7,8,9,10} | off (on 0) | - | same | +0.52 |
| L31 down c847 | 5% / 0 | **res** (R2 0.74): res in {63, 68, 156, 162..164, 166..169, 174, 182..184} | off (on 0) | - | same | +0.52 |
| L31 gate c417 | 23% / 1% | **res** (R2 0.59): res in {51, 61, 71, 81..86, 91, 121, 141..143, 145..158, 161..165, 171, 173..174, 180..186, 191} | unexplained (best R2 0.21) | (reads) | same | +0.52 |
| L24 gate c357 | 3% / 0 | **res%100** (R2 0.56): res mod 100 in {55, 75, 95} | off (on 0) | (reads) | same | +0.52 |
| L29 down c42 | 17% / 0 | **res** (R2 0.85): res in {30, 33, 120, 122..142} | off (on 0) | - | same | +0.52 |
| L31 gate c146 | 9% / 27% | unexplained (best R2 0.40) | **tens(a,b)** (R2 0.53): a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {2}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4,5,10}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9,10}; a//10 in {8} -> b//10 in {6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {4} | (reads) | same | +0.52 |
| L23 down c931 | 5% / 0 | **res** (R2 0.81): res in {151..158, 161..162} | off (on 0) | - | same | +0.52 |
| L24 up c234 | 5% / 1% | **res** (R2 0.75): res in {66..67, 116..117, 166..171} | unexplained (best R2 0.45) | (reads) | same | +0.52 |
| L26 gate c187 | 1% / 1% | unexplained (best R2 0.05) | unexplained (best R2 0.08) | (reads) | same | +0.52 |
| L27 up c73 | 12% / 2% | **res** (R2 0.72): res in {2..35, 38..47, 53, 143} | unexplained (best R2 0.23) | (reads) | same | +0.52 |
| L28 down c150 | 2% / 0 | **res%100** (R2 0.64): res mod 100 in {59} | off (on 0) | - | same | +0.51 |
| L23 o c308 (H15) | 96% / 91% | always | unexplained (best R2 0.41) | - | same | +0.51 |
| L31 down c640 | 23% / 0 | **res%100** (R2 0.76): res mod 100 in {1, 96..98} | off (on 0) | - | same | +0.51 |
| L30 down c810 | 3% / 1% | **res%100** (R2 0.75): res mod 100 in {52, 82} | **res%100** (R2 0.59): res mod 100 in {2} | - | same | +0.51 |
| L31 down c293 | 6% / 0 | **res** (R2 0.65): res in {131..134, 136..139, 141} | off (on 0) | - | same | +0.51 |
| L22 down c423 | 1% / 10% | unexplained (best R2 0.11) | unexplained (best R2 0.37) | - | same | +0.51 |
| L19 up c143 | 6% / 1% | unexplained (best R2 0.46) | unexplained (best R2 0.11) | (reads) | same | +0.50 |
| L19 up c347 | 10% / 42% | unexplained (best R2 0.39) | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {0,1}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {1,2,3,4,6}; a//10 in {4} -> b//10 in {3,4,6}; a//10 in {5} -> b//10 in {3,4,5,6,7}; a//10 in {6} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {7} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10}; a//10 in {10} -> b//10 in {5,7,8,9,10} | (reads) | same | +0.50 |
| L26 down c72 | 1% / 7% | unexplained (best R2 0.11) | unexplained (best R2 0.42) | - | same | +0.50 |
| L23 down c65 | 0 / 19% | off (on 0) | unexplained (best R2 0.26) | - | same | +0.50 |
| L25 down c1005 | 3% / 12% | unexplained (best R2 0.33) | unexplained (best R2 0.48) | - | same | +0.50 |
| L26 down c189 | 4% / 4% | unexplained (best R2 0.30) | unexplained (best R2 0.33) | - | same | +0.50 |
| L29 gate c420 | 4% / 6% | **res//10** (R2 0.67): (tens) res in {3..10, 12..19, 21..28} | **res** (R2 0.61): res in {9, 12..15, 25} | (reads) | same | +0.50 |
| L29 gate c128 | 9% / 0 | **res%100** (R2 0.84): res mod 100: no class above 0.5 (max 0.49) | off (on 0) | (reads) | same | +0.50 |
| L31 down c426 | 17% / 0 | **res** (R2 0.81): res in {34, 36, 116..118, 124..139, 156, 164, 166..167, 174..179} | unexplained (best R2 0.19) | - | same | +0.50 |
| L27 down c122 | 6% / 0 | **res%100** (R2 0.70): res mod 100: no class above 0.5 (max 0.48) | off (on 0) | - | same | +0.50 |
| L20 o c328 (H2) | 5% / 19% | unexplained (best R2 0.19) | unexplained (best R2 0.40) | a: mod4 +3%; b: mod4 +3% | a: mod4 +10%; b: mod4 +6% | +0.49 |
| L29 down c927 | 4% / 0 | **res** (R2 0.62): res in {96..98, 186..200} | off (on 0) | - | same | +0.49 |
| L26 down c142 | 6% / 1% | **res%100** (R2 0.57): res mod 100 in {55, 75, 94..99} | **res** (R2 0.50): res in {15, 99} | - | same | +0.49 |
| L23 down c105 | 8% / 1% | **res%20** (R2 0.73): res mod 20 in {3, 13} | **res%100** (R2 0.48): res mod 100: no class above 0.5 (max 0.38) | - | same | +0.49 |
| L23 down c41 | 15% / 30% | **res//10** (R2 0.79): (tens) res in {6..59} | **res//10** (R2 0.53): (tens) res in {2..31} | - | res: mod100 +2% | +0.49 |
| L21 down c390 | 6% / 1% | **res** (R2 0.80): res in {42, 62, 102, 122, 132, 142, 152, 162, 172, 182, 192, 198, 200} | **res%100** (R2 0.53): res mod 100: no class above 0.5 (max 0.44) | - | same | +0.49 |
| L29 gate c426 | 2% / 1% | **res%100** (R2 0.59): res mod 100 in {62, 92} | **res** (R2 0.75): res in {12} | (reads) | same | +0.49 |
| L21 gate c10 | 12% / 55% | **res%10** (R2 0.81): res mod 10 in {1} | **res%10** (R2 0.51): res mod 10 in {1, 3, 5, 7, 9} [coarser: res mod 2 in {1}, R2 0.91] | (reads) | same | +0.49 |
| L29 down c219 | 8% / 0 | **res** (R2 0.74): res in {147..162, 170, 175} | off (on 0) | - | same | +0.48 |
| L31 up c436 | 75% / 4% | unexplained (best R2 0.22) | unexplained (best R2 0.43) | (reads) | same | +0.48 |
| L28 down c979 | 5% / 0 | **res** (R2 0.83): res in {144..152} | off (on 0) | - | same | +0.48 |
| L23 o c456 (H22) | 1% / 3% | unexplained (best R2 0.21) | unexplained (best R2 0.39) | - | same | +0.48 |
| L31 gate c158 | 3% / 2% | **units(a,b)** (R2 0.62): a%10 in {0} -> b%10 in {0}; a%10 in {5} -> b%10 in {0,5} | unexplained (best R2 0.32) | (reads) | same | +0.48 |
| L31 gate c447 | 13% / 46% | **res** (R2 0.68): res in {3, 5..6, 8..9, 28..39, 51..68} | unexplained (best R2 0.48) | (reads) | same | +0.48 |
| L31 v c139 (kv7) | 67% / 93% | unexplained (best R2 0.32) | unexplained (best R2 0.21) | (reads) | same | +0.47 |
| L27 up c723 | 8% / 0 | **res** (R2 0.83): res in {44, 84, 144, 164..198} | off (on 0) | (reads) | same | +0.47 |
| L31 down c45 | 5% / 70% | unexplained (best R2 0.45) | **tens(a,b)** (R2 0.52): a//10 in {0,9} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1,2} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {4} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {7,8,9}; a//10 in {7} -> b//10 in {0,1,2,3,4,7,8,9,10}; a//10 in {8} -> b//10 in {1,2,3,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,10} | - | res: mod100 +2% | +0.47 |
| L25 down c200 | 17% / 0 | **res//10** (R2 0.81): (tens) res in {137..170, 185..188, 195, 198} | off (on 0) | - | same | +0.47 |
| L28 gate c719 | 9% / 0 | **res** (R2 0.52): res in {112, 116, 132, 134, 136, 138, 142, 144, 146, 152, 156, 176, 182, 192, 196} | off (on 0) | (reads) | same | +0.46 |
| L30 down c17 | 4% / 0 | **res** (R2 0.76): res in {112..115, 131, 171, 181} | off (on 0) | - | same | +0.46 |
| L26 gate c95 | 7% / 0 | **res** (R2 0.70): res in {120..121, 130..131, 160..161, 170..171, 176..177, 179..182, 190, 200} | off (on 0) | (reads) | same | +0.46 |
| L26 gate c7 | 6% / 0 | **res** (R2 0.92): res in {120..127} | off (on 0) | (reads) | same | +0.46 |
| L19 up c595 | 16% / 24% | unexplained (best R2 0.42) | unexplained (best R2 0.32) | (reads) | same | +0.46 |
| L28 up c338 | 6% / 0 | **res** (R2 0.85): res in {107..114} | off (on 0) | (reads) | same | +0.46 |
| L21 up c15 | 24% / 6% | **res%20** (R2 0.56): res mod 20 in {4..8} | unexplained (best R2 0.39) | (reads) | same | +0.46 |
| L31 down c160 | 16% / 0 | **res** (R2 0.80): res in {74, 83, 87..88, 107..108, 113..114, 117..118, 127..128, 133, 137..138, 147..148, 157..158, 163, 167..168, 173..175, 177..179, 183..184, 187..188} | off (on 0) | - | same | +0.46 |
| L30 up c594 | 74% / 0 | **res** (R2 0.58): res in {45..50, 52..53, 55..69, 72, 75..77, 85..86, 89..90, 92..200} | off (on 0) | (reads) | same | +0.46 |
| L19 down c86 | 25% / 1% | **res//10** (R2 0.92): (tens) res in {132..200} | **tens(a,b)** (R2 0.62): a//10 in {6} -> b//10 in {10}; a//10 in {10} -> b//10 in {3,4,5,6} | a: mod100 +3%; res: mod100 +5% | - | +0.45 |
| L20 down c266 | 3% / 29% | unexplained (best R2 0.48) | **tens(a,b)** (R2 0.74): a//10 in {0} -> b//10 in {3,4}; a//10 in {1} -> b//10 in {2,3,4}; a//10 in {2} -> b//10 in {2,3,4,5}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,10}; a//10 in {4} -> b//10 in {0,1,2,3,4,5,6}; a//10 in {5,6} -> b//10 in {2,3} | - | same | +0.45 |
| L25 down c92 | 17% / 0 | **res//10** (R2 0.81): (tens) res in {111..129, 135} | off (on 0) | - | same | +0.45 |
| L21 down c92 | 10% / 0 | **res** (R2 0.91): res in {130..143} | off (on 0) | res: mod25 +4% | - | +0.45 |
| L29 up c51 | 17% / 1% | **res** (R2 0.89): res in {28, 117..138} | **res** (R2 0.53): res in {21, 28} | (reads) | same | +0.45 |
| L26 gate c486 | 0 / 1% | off (on 0) | **res%100** (R2 0.82): res mod 100: no class above 0.5 (max 0.44) | (reads) | same | +0.45 |
| L29 gate c465 | 18% / 1% | **res** (R2 0.86): res in {104..109, 114..129} | unexplained (best R2 0.41) | (reads) | same | +0.45 |
| L19 up c20 | 4% / 0 | **tens(a,b)** (R2 0.61): a//10 in {4} -> b//10 in {4}; a//10 in {5} -> b//10 in {3,4} | off (on 0) | (reads) | same | +0.45 |
| L26 gate c180 | 0 / 1% | off (on 0) | **res%100** (R2 0.75): res mod 100: no class above 0.5 (max 0.45) | (reads) | same | +0.45 |
| L31 up c147 | 5% / 0 | **res** (R2 0.71): res in {113, 131..134, 136..139} | off (on 0) | (reads) | same | +0.45 |
| L29 down c128 | 9% / 0 | **res%100** (R2 0.85): res mod 100: no class above 0.5 (max 0.50) | off (on 0) | - | same | +0.45 |
| L20 gate c587 | 0 / 20% | off (on 0) | unexplained (best R2 0.40) | (reads) | same | +0.44 |
| L31 gate c389 | 3% / 0 | **res** (R2 0.59): res in {61, 111, 131, 161, 171} | off (on 0) | (reads) | same | +0.44 |
| L26 down c106 | 0 / 1% | off (on 0) | **res%100** (R2 0.79): res mod 100: no class above 0.5 (max 0.41) | - | same | +0.44 |
| L26 gate c260 | 0 / 1% | off (on 0) | **res%100** (R2 0.79): res mod 100: no class above 0.5 (max 0.41) | (reads) | same | +0.44 |
| L29 gate c736 | 4% / 1% | **res%50** (R2 0.48): res mod 50 in {39} | **res** (R2 0.54): res in {9, 99} | (reads) | same | +0.44 |
| L31 gate c336 | 20% / 0 | **res** (R2 0.68): res in {58, 68, 74, 78, 88, 114, 116..118, 127..128, 133..134, 136..138, 147..148, 153..160, 162..164, 166..168, 170, 173..179, 183..184, 187..188} | off (on 0) | (reads) | same | +0.44 |
| L20 o c192 (H2) | 6% / 12% | **units(a,b)** (R2 0.54): a%10 in {0,5} -> b%10 in {0,5} | **units(a,b)** (R2 0.56): a%10 in {0} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {5} -> b%10 in {0,5} | - | same | +0.44 |
| L29 down c11 | 74% / 2% | unexplained (best R2 0.43) | unexplained (best R2 0.17) | a: mod100 +3% | - | +0.43 |
| L31 up c734 | 7% / 2% | **units(a,b)** (R2 0.57): a%10 in {0,5} -> b%10 in {0,5} | unexplained (best R2 0.48) | (reads) | same | +0.43 |
| L31 down c314 | 4% / 0 | **res** (R2 0.76): res in {113..114, 133..134, 152..154} | off (on 0) | - | same | +0.43 |
| L31 down c494 | 2% / 7% | unexplained (best R2 0.47) | **a** (R2 0.66): a in {1..6} | - | same | +0.43 |
| L21 down c124 | 0 / 29% | off (on 0) | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {0}; a//10 in {1} -> b//10 in {1}; a//10 in {2} -> b//10 in {6}; a//10 in {3} -> b//10 in {4,5,6,7,8}; a//10 in {4} -> b//10 in {6,7,8,9}; a//10 in {5} -> b//10 in {5,6,7,8,9,10}; a//10 in {6} -> b//10 in {6,7,8,9,10}; a//10 in {7} -> b//10 in {7,8,9,10}; a//10 in {8} -> b//10 in {8,9,10}; a//10 in {10} -> b//10 in {10} | - | same | +0.43 |
| L31 gate c125 | 0 / 0 | off (on 0) | same | (reads) | same | +0.43 |
| L31 down c921 | 8% / 0 | **res** (R2 0.72): res in {97..98, 102, 117..124} | off (on 0) | - | same | +0.43 |
| L31 gate c192 | 10% / 0 | unexplained (best R2 0.41) | off (on 0) | (reads) | same | +0.43 |
| L23 gate c257 | 7% / 2% | **res** (R2 0.77): res in {57, 97, 117, 137, 143, 147, 153, 157, 163, 167, 173, 177, 183, 187, 193, 197} | **res** (R2 0.61): res in {7, 17} | (reads) | same | +0.43 |
| L29 down c880 | 6% / 0 | **res** (R2 0.86): res in {114..120, 140} | off (on 0) | - | same | +0.43 |
| L26 gate c157 | 7% / 2% | **res%100** (R2 0.65): res mod 100 in {14, 44..47, 64, 74, 94} | **res%100** (R2 0.47): res mod 100: no class above 0.5 (max 0.41) | (reads) | same | +0.42 |
| L27 down c200 | 7% / 0 | **res%100** (R2 0.64): res mod 100 in {48..50, 89, 94} | off (on 0) | - | same | +0.42 |
| L27 down c62 | 11% / 2% | **res%100** (R2 0.67): res mod 100 in {45, 55, 65} | **res%100** (R2 0.68): res mod 100: no class above 0.5 (max 0.44) | - | same | +0.42 |
| L28 gate c798 | 2% / 0 | **res** (R2 0.72): res in {170..174} | off (on 0) | (reads) | same | +0.42 |
| L28 gate c298 | 11% / 0 | **res** (R2 0.86): res in {66, 157..190, 198..200} | off (on 0) | (reads) | same | +0.41 |
| L30 down c399 | 11% / 0 | **res//10** (R2 0.73): (tens) res in {101..109, 198..200} | off (on 0) | - | same | +0.41 |
| L23 down c123 | 4% / 0 | **res** (R2 0.54): res in {92, 101..102, 151..152} | off (on 0) | - | same | +0.41 |
| L31 down c301 | 3% / 0 | **res** (R2 0.72): res in {144..149} | off (on 0) | - | same | +0.41 |
| L20 o c135 (H2) | 1% / 67% | unexplained (best R2 0.06) | unexplained (best R2 0.34) | - | a: mod4 +7%; b: mod4 +2% | +0.41 |
| L28 gate c338 | 1% / 1% | **res** (R2 0.61): res in {112..113} | **res** (R2 0.81): res in {10} | (reads) | same | +0.40 |
| L29 gate c449 | 5% / 0 | **res** (R2 0.69): res in {149..160} | off (on 0) | (reads) | same | +0.40 |
| L19 down c82 | 0 / 2% | off (on 0) | **tens(a,b)** (R2 0.60): a//10 in {0} -> b//10 in {0} | - | same | +0.40 |
| L31 up c108 | 17% / 0 | **res** (R2 0.65): res in {61, 68, 71, 74, 83..84, 114, 134, 138, 146..149, 151..169, 171, 173..174, 183..184} | off (on 0) | (reads) | same | +0.40 |
| L26 up c95 | 15% / 7% | **res** (R2 0.77): res in {2..10, 12..14, 17, 110, 120, 129..141, 150, 160..161, 170..171, 180..181} | **res%100** (R2 0.63): res mod 100: no class above 0.5 (max 0.49) | (reads) | same | +0.39 |
| L20 up c484 | 0 / 2% | off (on 0) | unexplained (best R2 0.27) | (reads) | same | +0.39 |
| L31 gate c398 | 13% / 0 | **res** (R2 0.67): res in {81, 111, 115..116, 119..121, 125..126, 130..131, 135, 151, 155..156, 159..161, 166, 170..171, 175..176, 179..181} | off (on 0) | (reads) | same | +0.39 |
| L31 down c533 | 5% / 68% | unexplained (best R2 0.37) | unexplained (best R2 0.47) | - | same | +0.39 |
| L19 down c17 | 0 / 11% | off (on 0) | **tens(a,b)** (R2 0.65): a//10 in {4,5,6,7} -> b//10 in {0}; a//10 in {8,9} -> b//10 in {0,1}; a//10 in {10} -> b//10 in {0,1,2} | - | same | +0.39 |
| L27 down c312 | 5% / 0 | **res** (R2 0.77): res in {58, 68, 78, 88, 118, 128, 138, 148, 158, 168, 178} | off (on 0) | - | same | +0.39 |
| L27 gate c246 | 3% / 0 | **res** (R2 0.77): res in {73, 113, 133, 173..174} | off (on 0) | (reads) | same | +0.39 |
| L22 o c352 (H3) | 1% / 2% | **a** (R2 0.54): a in {97} | **a** (R2 0.64): a in {97} | - | same | +0.39 |
| L31 up c773 | 14% / 97% | unexplained (best R2 0.38) | always | (reads) | same | +0.39 |
| L23 up c407 | 3% / 36% | **res%100** (R2 0.49): res mod 100 in {22} | **tens(a,b)** (R2 0.61): a//10 in {3} -> b//10 in {2,5,6,7}; a//10 in {4} -> b//10 in {3,6,7}; a//10 in {5} -> b//10 in {3,4,5,6,7,8}; a//10 in {6} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10}; a//10 in {10} -> b//10 in {5,6,7,8,9} | (reads) | same | +0.38 |
| L21 up c111 | 0 / 32% | off (on 0) | unexplained (best R2 0.46) | (reads) | same | +0.38 |
| L26 gate c552 | 5% / 3% | **res%100** (R2 0.79): res mod 100 in {43, 53, 63, 73, 93} | **res%100** (R2 0.58): res mod 100: no class above 0.5 (max 0.45) | (reads) | same | +0.38 |
| L19 down c278 | 2% / 54% | unexplained (best R2 0.25) | **tens(a,b)** (R2 0.67): a//10 in {0} -> b//10 in {7,8,9,10}; a//10 in {2} -> b//10 in {8,9,10}; a//10 in {3,4} -> b//10 in {6,7,8,9,10}; a//10 in {5} -> b//10 in {3,4,6,7,8,9,10}; a//10 in {6} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {8,9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | - | a: mod100 +2%; b: mod100 +3% | +0.38 |
| L20 down c184 | 4% / 3% | unexplained (best R2 0.22) | unexplained (best R2 0.31) | - | same | +0.38 |
| L20 down c106 | 3% / 0 | unexplained (best R2 0.48) | off (on 0) | - | same | +0.38 |
| L29 gate c42 | 17% / 0 | **res** (R2 0.92): res in {118..141} | off (on 0) | (reads) | same | +0.38 |
| L24 o c501 (H22) | 0 / 2% | off (on 0) | **a** (R2 0.63): a in {27..29} | - | same | +0.38 |
| L20 down c153 | 19% / 0 | **res//10** (R2 0.91): (tens) res in {140..196, 198} | off (on 0) | - | same | +0.37 |
| L31 down c104 | 48% / 1% | unexplained (best R2 0.44) | unexplained (best R2 0.21) | - | same | +0.37 |
| L23 o c394 (H22) | 6% / 9% | unexplained (best R2 0.35) | unexplained (best R2 0.34) | - | same | +0.37 |
| L20 up c416 | 35% / 11% | **res%100** (R2 0.49): res mod 100 in {1, 46..48, 50..68, 96, 98..99} | **res%100** (R2 0.63): res mod 100 in {6, 8, 10} | (reads) | same | +0.37 |
| L21 up c68 | 0 / 5% | off (on 0) | unexplained (best R2 0.34) | (reads) | same | +0.37 |
| L21 gate c63 | 0 / 7% | off (on 0) | unexplained (best R2 0.35) | (reads) | same | +0.37 |
| L21 down c48 | 1% / 10% | unexplained (best R2 0.11) | unexplained (best R2 0.36) | - | b: mod4 +2% | +0.37 |
| L24 down c79 | 0 / 32% | off (on 0) | **a** (R2 0.51): a in {2..24, 97} | - | a: mod100 +2% | +0.37 |
| L25 gate c33 | 6% / 0 | **res** (R2 0.77): res in {102..104, 118..120, 123..124} | off (on 0) | (reads) | same | +0.37 |
| L26 up c189 | 1% / 0 | unexplained (best R2 0.38) | off (on 0) | (reads) | same | +0.36 |
| L31 up c834 | 23% / 0 | **res** (R2 0.71): res in {68, 113..114, 131..144, 146..184, 188} | off (on 0) | (reads) | same | +0.36 |
| L29 down c584 | 11% / 11% | unexplained (best R2 0.50) | **tens(a,b)** (R2 0.54): a//10 in {0} -> b//10 in {0,1,2,10}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {2}; a//10 in {9} -> b//10 in {9,10} | - | same | +0.36 |
| L20 gate c5 | 23% / 64% | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {6,7,8,9,10}; a//10 in {1} -> b//10 in {6,7,8}; a//10 in {2} -> b//10 in {6,7}; a//10 in {6} -> b//10 in {1,2}; a//10 in {7} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1,9,10}; a//10 in {9,10} -> b//10 in {0,8,9,10} | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {2,5,6,7,8,9,10}; a//10 in {3,4} -> b//10 in {7,8,9}; a//10 in {5} -> b//10 in {5,6,7,8,9,10}; a//10 in {6} -> b//10 in {0,1,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,7,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,7,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,9,10} | (reads) | same | +0.36 |
| L23 o c370 (H22) | 1% / 1% | **a** (R2 0.82): a in {55} | **a** (R2 0.67): a in {55} | - | same | +0.36 |
| L29 down c6 | 95% / 98% | unexplained (best R2 0.20) | always | - | same | +0.36 |
| L22 down c55 | 1% / 32% | unexplained (best R2 0.08) | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {10}; a//10 in {2} -> b//10 in {9,10}; a//10 in {3} -> b//10 in {0,9,10}; a//10 in {4} -> b//10 in {1,2,9,10}; a//10 in {5} -> b//10 in {2,9,10}; a//10 in {6} -> b//10 in {1,2,3,10}; a//10 in {7} -> b//10 in {2,3,4}; a//10 in {8} -> b//10 in {1,2,3,4,5}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7}; a//10 in {10} -> b//10 in {0,1,2,3,5,6,7,10} | - | same | +0.36 |
| L23 down c10 | 19% / 97% | **res//10** (R2 0.63): (tens) res in {2..4, 6, 8..59, 61..63} | always | - | same | +0.36 |
| L19 up c6 | 99% / 63% | always | unexplained (best R2 0.31) | (reads) | same | +0.35 |
| L30 down c8 | 5% / 0 | **res** (R2 0.72): res in {118, 131, 135..139, 171, 174, 178..179} | off (on 0) | - | same | +0.35 |
| L26 down c119 | 0 / 0 | off (on 0) | same | - | same | +0.35 |
| L31 gate c368 | 1% / 33% | unexplained (best R2 0.19) | unexplained (best R2 0.41) | (reads) | same | +0.34 |
| L29 down c794 | 6% / 0 | **res%100** (R2 0.83): res mod 100 in {1} | off (on 0) | - | same | +0.34 |
| L21 down c17 | 78% / 2% | **res** (R2 0.60): res in {4, 8, 14, 16, 18, 20, 24..30, 32..40, 42..50, 54, 56, 58, 60, 63..70, 72, 74, 76..80, 82..90, 94, 96, 98, 100, 104..110, 114, 116, 118..120, 122, 124..200} | unexplained (best R2 0.23) | - | same | +0.34 |
| L25 down c32 | 0 / 11% | off (on 0) | unexplained (best R2 0.47) | - | same | +0.34 |
| L21 gate c57 | 10% / 0 | **res** (R2 0.77): res in {116..127} | off (on 0) | (reads) | same | +0.33 |
| L26 gate c72 | 1% / 1% | unexplained (best R2 0.06) | unexplained (best R2 0.09) | (reads) | same | +0.33 |
| L30 up c806 | 1% / 0 | **res%100** (R2 0.59): res mod 100: no class above 0.5 (max 0.32) | off (on 0) | (reads) | same | +0.33 |
| L20 o c288 (H2) | 4% / 13% | **units(a,b)** (R2 0.78): a%10 in {0} -> b%10 in {5}; a%10 in {5} -> b%10 in {0,5} | **units(a,b)** (R2 0.71): a%10 in {0,7,8,9} -> b%10 in {5}; a%10 in {5} -> b%10 in {0,1,2,3,4,5,6,7,8,9} | - | same | +0.33 |
| L21 up c206 | 19% / 1% | **res%100** (R2 0.51): res mod 100 in {1, 47..49, 57, 96..99} | unexplained (best R2 0.23) | (reads) | same | +0.33 |
| L26 up c115 | 4% / 1% | **res//10** (R2 0.53): (tens) res in {30, 32, 34..39, 180..190, 192..200} | unexplained (best R2 0.21) | (reads) | same | +0.32 |
| L28 down c218 | 8% / 0 | **res** (R2 0.93): res in {115..123, 179} | off (on 0) | - | same | +0.32 |
| L25 down c51 | 11% / 0 | **res** (R2 0.76): res in {62, 122, 152..178, 182..186} | off (on 0) | - | same | +0.32 |
| L26 down c95 | 17% / 0 | **res** (R2 0.78): res in {120..121, 125..141, 150..151, 160..161, 170..171, 175..176, 179..181} | off (on 0) | - | same | +0.32 |
| L31 down c90 | 0 / 53% | off (on 0) | unexplained (best R2 0.47) | - | same | +0.32 |
| L31 down c926 | 6% / 0 | **res** (R2 0.83): res in {139..146, 149} | off (on 0) | - | same | +0.32 |
| L21 gate c36 | 8% / 2% | **tens(a,b)** (R2 0.69): a//10 in {6} -> b//10 in {9,10}; a//10 in {7} -> b//10 in {8,9,10}; a//10 in {8} -> b//10 in {7,8,10}; a//10 in {9} -> b//10 in {6,7}; a//10 in {10} -> b//10 in {6,7,10} | unexplained (best R2 0.40) | (reads) | same | +0.32 |
| L31 down c188 | 2% / 0 | **res** (R2 0.62): res in {146, 148..149, 192, 194..199} | off (on 0) | - | same | +0.32 |
| L31 gate c428 | 13% / 0 | **res** (R2 0.61): res in {81..83, 121, 141..143, 146..164, 181..183} | off (on 0) | (reads) | same | +0.32 |
| L24 down c350 | 9% / 13% | **res//10** (R2 0.71): (tens) res in {2..43, 200} | unexplained (best R2 0.49) | - | same | +0.32 |
| L26 down c641 | 1% / 0 | **res//10** (R2 0.51): (tens) res in {181..182, 186, 192, 194..200} | off (on 0) | - | same | +0.31 |
| L25 down c79 | 20% / 0 | **tens(a,b)** (R2 0.52): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {9}; a//10 in {2} -> b//10 in {8,9}; a//10 in {3} -> b//10 in {7,8,9}; a//10 in {4} -> b//10 in {6,8}; a//10 in {6} -> b//10 in {4,5}; a//10 in {7} -> b//10 in {3,4,5}; a//10 in {8} -> b//10 in {2,3,4}; a//10 in {9} -> b//10 in {1,2,3}; a//10 in {10} -> b//10 in {0,1,2,3} | off (on 0) | - | same | +0.31 |
| L24 down c112 | 16% / 0 | **res//10** (R2 0.84): (tens) res in {120..142} | off (on 0) | - | same | +0.31 |
| L31 gate c82 | 3% / 0 | **res** (R2 0.68): res in {176..200} | off (on 0) | (reads) | same | +0.31 |
| L30 gate c87 | 30% / 91% | unexplained (best R2 0.21) | unexplained (best R2 0.36) | (reads) | same | +0.30 |
| L31 down c595 | 5% / 52% | **res** (R2 0.63): res in {75, 85, 115, 125, 135, 155, 175} | unexplained (best R2 0.49) | - | res: mod100 +3% | +0.30 |
| L28 up c23 | 28% / 0 | **res//10** (R2 0.76): (tens) res in {86, 88, 126..176, 179..188, 190..198, 200} | off (on 0) | (reads) | same | +0.30 |
| L23 down c8 | 0 / 11% | off (on 0) | **tens(a,b)** (R2 0.62): a//10 in {5,6} -> b//10 in {0,1}; a//10 in {7} -> b//10 in {0,1,2,3}; a//10 in {8} -> b//10 in {0,1,3}; a//10 in {10} -> b//10 in {2,3} | - | same | +0.29 |
| L31 o c165 (H15) | 98% / 33% | always | **tens(a,b)** (R2 0.52): a//10 in {0,1} -> b//10 in {0,10}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3,4} -> b//10 in {0}; a//10 in {5} -> b//10 in {0,4,5}; a//10 in {6} -> b//10 in {5,6}; a//10 in {7} -> b//10 in {0,6}; a//10 in {8} -> b//10 in {0,1,4,6,7,8}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | - | same | +0.29 |
| L31 down c147 | 3% / 0 | **res** (R2 0.62): res in {174..179, 182, 184, 187..189, 194..198} | off (on 0) | - | same | +0.29 |
| L22 up c151 | 96% / 96% | always | same | (reads) | same | +0.29 |
| L31 down c213 | 18% / 16% | **res//10** (R2 0.60): (tens) res in {3..6, 8, 18, 26..29, 41..42, 44..68} | **tens(a,b)** (R2 0.63): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {1,2,4,5,6}; a//10 in {5,6} -> b//10 in {0}; a//10 in {10} -> b//10 in {10} | - | a: mod50 +3%, mod25 +3%, mod20 +2% | +0.29 |
| L26 down c529 | 9% / 0 | **res%100** (R2 0.71): res mod 100 in {1} | off (on 0) | - | same | +0.29 |
| L26 down c154 | 1% / 37% | unexplained (best R2 0.09) | unexplained (best R2 0.37) | - | a: mod4 +5%; b: mod4 +5%; res: mod4 +5% | +0.29 |
| L20 o c479 (H2) | 6% / 15% | **units(a,b)** (R2 0.66): a%10 in {0,5} -> b%10 in {0,5} | **units(a,b)** (R2 0.61): a%10 in {0} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {5} -> b%10 in {0,5} | - | same | +0.29 |
| L29 down c28 | 6% / 0 | **res** (R2 0.74): res in {106..107, 117, 146..147, 155..157, 160, 166..167, 176..177, 186..187, 196} | off (on 0) | - | same | +0.29 |
| L30 up c399 | 9% / 0 | **res//10** (R2 0.78): (tens) res in {101..109, 200} | off (on 0) | (reads) | same | +0.29 |
| L19 up c94 | 0 / 15% | off (on 0) | **tens(a,b)** (R2 0.68): a//10 in {1,2,3,4,5,6} -> b//10 in {0}; a//10 in {7,8,9} -> b//10 in {0,1}; a//10 in {10} -> b//10 in {0,1,2,3,4,5} | (reads) | same | +0.29 |
| L22 down c122 | 0 / 1% | off (on 0) | unexplained (best R2 0.09) | - | same | +0.28 |
| L27 up c133 | 2% / 0 | **res//10** (R2 0.81): (tens) res in {180..200} | off (on 0) | (reads) | same | +0.28 |
| L25 down c464 | 0 / 0 | off (on 0) | same | - | same | +0.28 |
| L30 down c121 | 9% / 0 | **res** (R2 0.87): res in {113, 128..138, 188} | off (on 0) | - | same | +0.28 |
| L31 gate c115 | 9% / 1% | **res** (R2 0.64): res in {85, 105, 110, 115, 120, 125, 130, 135, 145, 150, 155, 160, 165, 175, 180, 185, 195} | unexplained (best R2 0.13) | (reads) | same | +0.28 |
| L23 down c130 | 28% / 0 | **res//10** (R2 0.83): (tens) res in {128..200} | off (on 0) | - | same | +0.27 |
| L25 up c73 | 5% / 0 | **res//10** (R2 0.78): (tens) res in {169..200} | off (on 0) | (reads) | same | +0.27 |
| L19 down c94 | 13% / 9% | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {0,2,3,7,8,9,10}; a//10 in {1,2,3,4,5,6,7,8,9} -> b//10 in {0}; a//10 in {10} -> b//10 in {0,10} | **tens(a,b)** (R2 0.58): a//10 in {0} -> b//10 in {0,1,2,10}; a//10 in {1,3,7,8,10} -> b//10 in {0}; a//10 in {9} -> b//10 in {0,9,10} | - | same | +0.27 |
| L31 down c629 | 0 / 5% | off (on 0) | **a** (R2 0.65): a in {94..99} | - | same | +0.27 |
| L25 up c851 | 4% / 0 | **res//10** (R2 0.79): (tens) res in {171..200} | off (on 0) | (reads) | same | +0.26 |
| L30 up c16 | 23% / 0 | **res** (R2 0.77): res in {112..116, 118, 126..139, 144, 148, 152..154, 158, 162, 164, 166..200} | off (on 0) | (reads) | same | +0.26 |
| L30 down c21 | 88% / 1% | unexplained (best R2 0.27) | unexplained (best R2 0.37) | a: mod100 +4%, mod50 +3%, mod25 +2%; b: mod100 +3%, mod25 +3% | - | +0.26 |
| L19 down c8 | 4% / 87% | unexplained (best R2 0.14) | unexplained (best R2 0.46) | - | res: mod100 +7%, mod25 +2% | +0.26 |
| L31 down c100 | 2% / 10% | unexplained (best R2 0.20) | unexplained (best R2 0.30) | - | same | +0.26 |
| L20 q c131 (H2) | 8% / 79% | unexplained (best R2 0.37) | unexplained (best R2 0.49) | (reads) | same | +0.25 |
| L31 gate c292 | 0 / 0 | off (on 0) | same | (reads) | same | +0.25 |
| L31 up c559 | 3% / 0 | **res** (R2 0.67): res in {175..182, 184, 186..188, 190, 194..200} | off (on 0) | (reads) | same | +0.25 |
| L31 gate c346 | 3% / 0 | **res** (R2 0.70): res in {98, 101, 111, 151} | off (on 0) | (reads) | same | +0.24 |
| L24 down c51 | 28% / 0 | **tens(a,b)** (R2 0.80): a//10 in {2} -> b//10 in {9,10}; a//10 in {3} -> b//10 in {8,9,10}; a//10 in {4,5} -> b//10 in {6,7,8,9,10}; a//10 in {6} -> b//10 in {5,6,7,8,9,10}; a//10 in {7} -> b//10 in {4,5,6,7}; a//10 in {8} -> b//10 in {3,4,5,6}; a//10 in {9,10} -> b//10 in {2,3,4,5,10} | off (on 0) | - | same | +0.24 |
| L29 down c662 | 9% / 0 | **res** (R2 0.82): res in {102..103, 112..113, 122..123, 132..133, 142..143, 152..153, 163, 172..173, 182..183, 192..193} | off (on 0) | - | same | +0.24 |
| L21 down c99 | 21% / 0 | **res** (R2 0.70): res in {50..53, 96, 136..159, 187..200} | off (on 0) | res: mod25 +3% | - | +0.24 |
| L29 gate c232 | 23% / 0 | **res** (R2 0.74): res in {119, 122..126, 129, 134, 138..139, 141..179, 182..187} | off (on 0) | (reads) | same | +0.24 |
| L31 up c640 | 39% / 1% | **res** (R2 0.63): res in {2..50, 55..56, 63..67, 83..86, 111..117, 121, 123..136, 145..146, 155..156, 165..166, 185} | unexplained (best R2 0.09) | (reads) | same | +0.24 |
| L29 up c171 | 11% / 0 | **res** (R2 0.63): res in {101, 106..107, 110..111, 117, 121, 131, 137, 147, 150..151, 156..157, 161, 167, 170..171, 177, 187} | off (on 0) | (reads) | same | +0.23 |
| L29 up c449 | 8% / 0 | **tens(a,b)** (R2 0.75): a//10 in {6} -> b//10 in {9,10}; a//10 in {7} -> b//10 in {8,9,10}; a//10 in {8} -> b//10 in {7,8,9,10}; a//10 in {9} -> b//10 in {7,8,10}; a//10 in {10} -> b//10 in {6,7,8,10} | off (on 0) | (reads) | same | +0.23 |
| L31 down c132 | 9% / 0 | **res** (R2 0.80): res in {91, 149..162, 170..171, 175, 181, 191} | off (on 0) | - | same | +0.23 |
| L31 k c1 (kv3) | 89% / 3% | unexplained (best R2 0.18) | unexplained (best R2 0.32) | (reads) | same | +0.23 |
| L31 down c680 | 31% / 1% | **res** (R2 0.72): res in {6..31, 33..48, 51, 53..56, 58..59, 61..64, 66..68, 71, 73..74, 76..78, 80..88} | unexplained (best R2 0.15) | - | same | +0.23 |
| L20 up c446 | 0 / 33% | off (on 0) | **tens(a,b)** (R2 0.53): a//10 in {3} -> b//10 in {7}; a//10 in {4} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {4,7,8,9}; a//10 in {6,7,8,10} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10} | (reads) | same | +0.23 |
| L31 gate c257 | 14% / 19% | **res//10** (R2 0.83): (tens) res in {2..51, 200} | **tens(a,b)** (R2 0.57): a//10 in {0} -> b//10 in {0,1,2,3,4}; a//10 in {1} -> b//10 in {0,1,2,3}; a//10 in {2} -> b//10 in {0,2,3}; a//10 in {3} -> b//10 in {0,3,4}; a//10 in {4} -> b//10 in {0,4}; a//10 in {5} -> b//10 in {0} | (reads) | same | +0.23 |
| L19 down c454 | 1% / 1% | **res//10** (R2 0.62): (tens) res in {2..9} | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {0} | - | same | +0.23 |
| L20 o c252 (H2) | 4% / 12% | unexplained (best R2 0.15) | unexplained (best R2 0.45) | - | same | +0.23 |
| L27 gate c270 | 7% / 0 | **res** (R2 0.79): res in {156..177} | off (on 0) | (reads) | same | +0.23 |
| L27 v c239 (kv3) | 2% / 60% | unexplained (best R2 0.08) | **tens(a,b)** (R2 0.51): a//10 in {0} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,6,7,8,9}; a//10 in {6,10} -> b//10 in {1,2,8}; a//10 in {7} -> b//10 in {0,1,2,8,9}; a//10 in {8} -> b//10 in {1,2}; a//10 in {9} -> b//10 in {1,2,3,4,5} | (reads) | same | +0.23 |
| L31 down c244 | 4% / 0 | **res** (R2 0.90): res in {174..200} | off (on 0) | - | same | +0.22 |
| L21 down c341 | 0 / 1% | off (on 0) | unexplained (best R2 0.08) | - | same | +0.22 |
| L25 up c735 | 7% / 4% | **res** (R2 0.65): res in {4, 6, 56..60, 100, 156..164} | **res%100** (R2 0.68): res mod 100 in {0, 2, 99} | (reads) | same | +0.22 |
| L20 down c73 | 0 / 12% | off (on 0) | **tens(a,b)** (R2 0.64): a//10 in {4} -> b//10 in {0}; a//10 in {5,8} -> b//10 in {0,1}; a//10 in {6,7} -> b//10 in {0,1,2}; a//10 in {10} -> b//10 in {0,1,2,3,4} | - | same | +0.22 |
| L27 up c25 | 28% / 0 | **res//10** (R2 0.86): (tens) res in {126, 128..200} | off (on 0) | (reads) | same | +0.21 |
| L19 gate c703 | 0 / 6% | off (on 0) | unexplained (best R2 0.26) | (reads) | same | +0.21 |
| L29 down c514 | 0 / 3% | off (on 0) | **res%100** (R2 0.59): res mod 100: no class above 0.5 (max 0.49) | - | same | +0.21 |
| L31 gate c252 | 7% / 95% | unexplained (best R2 0.31) | unexplained (best R2 0.28) | (reads) | same | +0.21 |
| L24 gate c86 | 0 / 1% | off (on 0) | **res%100** (R2 0.59): res mod 100: no class above 0.5 (max 0.39) | (reads) | same | +0.21 |
| L20 o c259 (H2) | 0 / 0 | off (on 0) | same | - | same | +0.20 |
| L31 up c83 | 16% / 0 | **res//10** (R2 0.84): (tens) res in {101..118} | off (on 0) | (reads) | same | +0.20 |
| L20 down c32 | 0 / 35% | off (on 0) | **tens(a,b)** (R2 0.59): a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {1,2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {3,4,5,6,7,8}; a//10 in {7,10} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10} | - | same | +0.20 |
| L21 up c63 | 87% / 17% | **res** (R2 0.63): res in {16..20, 26, 28..30, 33..80, 86..121, 126..130, 132..170, 172..180, 186..200} | unexplained (best R2 0.41) | (reads) | same | +0.20 |
| L25 down c715 | 0 / 1% | off (on 0) | **units(a,b)** (R2 0.60): a%10 in {0} -> b%10 in {0} | - | same | +0.20 |
| L28 down c1 | 100% / 100% | always | same | - | same | +0.20 |
| L21 gate c105 | 17% / 0 | **res** (R2 0.80): res in {126, 130, 132, 134, 136, 140, 142, 144, 146, 152, 154, 156, 160, 162, 164..198, 200} | off (on 0) | (reads) | same | +0.20 |
| L22 down c267 | 6% / 7% | unexplained (best R2 0.37) | **tens(a,b)** (R2 0.59): a//10 in {0,1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {2} | - | same | +0.20 |
| L31 down c813 | 9% / 0 | **res** (R2 0.79): res in {101..109, 188..189, 192..199} | off (on 0) | - | same | +0.19 |
| L28 gate c203 | 33% / 0 | **res//10** (R2 0.93): (tens) res in {120..190, 192..200} | off (on 0) | (reads) | same | +0.19 |
| L31 gate c659 | 0 / 5% | off (on 0) | **a** (R2 0.67): a in {94..99} | (reads) | same | +0.19 |
| L31 gate c112 | 11% / 46% | unexplained (best R2 0.44) | unexplained (best R2 0.39) | (reads) | same | +0.19 |
| L27 gate c25 | 13% / 0 | **res** (R2 0.88): res in {135..159} | off (on 0) | (reads) | same | +0.19 |
| L26 gate c712 | 99% / 3% | always | unexplained (best R2 0.31) | (reads) | same | +0.19 |
| L30 gate c30 | 91% / 2% | unexplained (best R2 0.29) | unexplained (best R2 0.21) | (reads) | same | +0.18 |
| L31 gate c48 | 0 / 0 | off (on 0) | same | (reads) | same | +0.18 |
| L31 down c181 | 87% / 92% | unexplained (best R2 0.13) | unexplained (best R2 0.25) | - | same | +0.18 |
| L31 down c259 | 47% / 2% | **res** (R2 0.50): res in {2..70, 75, 79..80, 82..83, 89..90, 92..95, 100, 109, 139, 155, 159..161, 163..165, 167..169, 180..183, 185, 189..190, 193..196, 200} | unexplained (best R2 0.15) | - | same | +0.18 |
| L20 down c181 | 0 / 13% | off (on 0) | **tens(a,b)** (R2 0.64): a//10 in {4} -> b//10 in {6}; a//10 in {5} -> b//10 in {6,7,8}; a//10 in {6} -> b//10 in {7,8,9}; a//10 in {7} -> b//10 in {8,9,10}; a//10 in {8} -> b//10 in {9,10} | - | same | +0.18 |
| L29 up c149 | 3% / 1% | **res%100** (R2 0.62): res mod 100: no class above 0.5 (max 0.41) | unexplained (best R2 0.48) | (reads) | same | +0.17 |
| L19 down c36 | 0 / 17% | off (on 0) | unexplained (best R2 0.45) | - | res: mod25 +2% | +0.17 |
| L25 down c33 | 42% / 0 | **res//10** (R2 0.94): (tens) res in {110..200} | off (on 0) | b: mod100 +2% | - | +0.17 |
| L19 down c34 | 14% / 0 | **res//10** (R2 0.82): (tens) res in {2..53} | off (on 0) | - | same | +0.17 |
| L27 up c93 | 12% / 0 | **res//10** (R2 0.88): (tens) res in {152..199} | off (on 0) | (reads) | same | +0.16 |
| L25 up c33 | 47% / 2% | **res//10** (R2 0.88): (tens) res in {105..198, 200} | unexplained (best R2 0.34) | (reads) | same | +0.16 |
| L30 gate c809 | 16% / 0 | **res//10** (R2 0.86): (tens) res in {144..182, 184..187, 192, 194..200} | off (on 0) | (reads) | same | +0.16 |
| L31 gate c38 | 100% / 99% | always | same | (reads) | same | +0.16 |
| L28 down c91 | 11% / 1% | **res//10** (R2 0.79): (tens) res in {17, 157..200} | **res** (R2 0.63): res in {17} | - | same | +0.16 |
| L22 down c969 | 3% / 0 | **tens(a,b)** (R2 0.46): a//10 in {8,9} -> b//10 in {9,10}; a//10 in {10} -> b//10 in {8,10} | off (on 0) | - | same | +0.16 |
| L30 down c320 | 22% / 0 | **res** (R2 0.76): res in {102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 182, 184, 186, 188, 192, 194, 196, 198} | off (on 0) | res: mod2 +2% | - | +0.15 |
| L23 o c280 (H7) | 1% / 3% | unexplained (best R2 0.41) | **a** (R2 0.56): a in {8, 88} | - | same | +0.15 |
| L20 o c316 (H2) | 0 / 0 | off (on 0) | same | - | same | +0.15 |
| L30 down c107 | 23% / 1% | unexplained (best R2 0.28) | unexplained (best R2 0.05) | - | same | +0.15 |
| L20 up c378 | 27% / 7% | **tens(a,b)** (R2 0.63): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,3,9,10}; a//10 in {2} -> b//10 in {0,1,2,10}; a//10 in {3,9} -> b//10 in {0,1}; a//10 in {4,5,6,7,10} -> b//10 in {0} | unexplained (best R2 0.29) | (reads) | same | +0.15 |
| L23 o c402 (H22) | 0 / 0 | off (on 0) | same | - | same | +0.15 |
| L31 down c11 | 97% / 45% | always | unexplained (best R2 0.35) | - | same | +0.14 |
| L22 up c106 | 1% / 95% | unexplained (best R2 0.08) | always | (reads) | same | +0.14 |
| L27 gate c149 | 5% / 0 | **res//10** (R2 0.78): (tens) res in {168..199} | off (on 0) | (reads) | same | +0.14 |
| L31 up c568 | 84% / 80% | unexplained (best R2 0.21) | unexplained (best R2 0.20) | (reads) | same | +0.14 |
| L27 down c157 | 5% / 0 | **res%100** (R2 0.77): res mod 100: no class above 0.5 (max 0.46) | off (on 0) | - | same | +0.14 |
| L31 up c197 | 94% / 26% | unexplained (best R2 0.46) | unexplained (best R2 0.48) | (reads) | same | +0.14 |
| L28 up c36 | 96% / 5% | always | unexplained (best R2 0.27) | (reads) | same | +0.14 |
| L25 gate c256 | 11% / 0 | **res//10** (R2 0.81): (tens) res in {152..154, 156..200} | off (on 0) | (reads) | same | +0.13 |
| L22 down c160 | 0 / 37% | off (on 0) | **res//10** (R2 0.63): (tens) res in {-99, -60, -52..0} | - | same | +0.13 |
| L30 down c381 | 4% / 99% | unexplained (best R2 0.41) | always | - | same | +0.13 |
| L23 up c9 | 22% / 98% | **res** (R2 0.62): res in {2..29, 32, 34..39, 41, 46, 56, 66, 76, 85..86, 95..96, 106, 116, 126, 136, 146, 156, 166, 176, 186, 196} | always | (reads) | same | +0.13 |
| L23 down c110 | 0 / 22% | off (on 0) | **tens(a,b)** (R2 0.56): a//10 in {3,4} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {6,7,8}; a//10 in {6} -> b//10 in {6,7,8,9,10}; a//10 in {7} -> b//10 in {7,8,9,10}; a//10 in {8} -> b//10 in {8,9,10}; a//10 in {9} -> b//10 in {7,9,10} | - | same | +0.13 |
| L23 o c435 (H22) | 1% / 2% | **a** (R2 0.54): a in {65} | unexplained (best R2 0.48) | - | same | +0.13 |
| L31 gate c107 | 18% / 46% | unexplained (best R2 0.15) | unexplained (best R2 0.28) | (reads) | same | +0.12 |
| L24 gate c51 | 18% / 0 | **res//10** (R2 0.88): (tens) res in {140..190, 192, 194..198, 200} | off (on 0) | (reads) | same | +0.12 |
| L26 up c7 | 9% / 1% | **res//10** (R2 0.72): (tens) res in {161, 163..200} | unexplained (best R2 0.19) | (reads) | same | +0.12 |
| L19 down c147 | 9% / 0 | **tens(a,b)** (R2 0.64): a//10 in {1,9} -> b//10 in {9,10}; a//10 in {2,3} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {10} | off (on 0) | - | same | +0.12 |
| L31 down c33 | 0 / 0 | off (on 0) | same | - | same | +0.12 |
| L21 v c105 (kv5) | 1% / 54% | unexplained (best R2 0.05) | **tens(a,b)** (R2 0.52): a//10 in {0,1} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {5,6,10} -> b//10 in {1,2}; a//10 in {7} -> b//10 in {0,1,2,3}; a//10 in {8} -> b//10 in {1,2,3}; a//10 in {9} -> b//10 in {1,2,3,4,5} | (reads) | same | +0.11 |
| L26 up c23 | 3% / 2% | unexplained (best R2 0.42) | unexplained (best R2 0.40) | (reads) | same | +0.11 |
| L29 gate c187 | 39% / 0 | **tens(a,b)** (R2 0.69): a//10 in {1} -> b//10 in {10}; a//10 in {2} -> b//10 in {8,10}; a//10 in {3} -> b//10 in {7,10}; a//10 in {4} -> b//10 in {6,7,9,10}; a//10 in {5} -> b//10 in {5,6,7,8,9,10}; a//10 in {6} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {1,2,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,4,5,6,7,8,9,10} | off (on 0) | (reads) | same | +0.11 |
| L31 gate c507 | 38% / 2% | **res//10** (R2 0.74): (tens) res in {4..90} | unexplained (best R2 0.28) | (reads) | same | +0.11 |
| L21 gate c653 | 9% / 1% | **res** (R2 0.57): res in {2..8, 10, 12, 18, 20..25, 41..42, 60..65} | unexplained (best R2 0.15) | (reads) | same | +0.11 |
| L31 down c44 | 10% / 0 | **res** (R2 0.84): res in {99, 117..120, 122..129} | off (on 0) | - | same | +0.11 |
| L31 down c365 | 12% / 1% | unexplained (best R2 0.26) | unexplained (best R2 0.39) | - | same | +0.11 |
| L25 down c256 | 7% / 0 | **res//10** (R2 0.83): (tens) res in {165..200} | off (on 0) | - | same | +0.10 |
| L20 down c136 | 8% / 0 | **tens(a,b)** (R2 0.67): a//10 in {5} -> b//10 in {7}; a//10 in {6} -> b//10 in {6,7,8}; a//10 in {7} -> b//10 in {6,7} | off (on 0) | - | same | +0.10 |
| L25 up c572 | 7% / 11% | **res//10** (R2 0.72): (tens) res in {2..33, 35} | **tens(a,b)** (R2 0.62): a//10 in {0,1,2} -> b//10 in {0,1,2} | (reads) | same | +0.10 |
| L19 down c367 | 0 / 20% | off (on 0) | **a//10** (R2 0.77): (tens) a in {1..19} | - | a: mod100 +2%, mod50 +2% | +0.10 |
| L24 gate c70 | 0 / 0 | off (on 0) | same | (reads) | same | +0.09 |
| L19 up c106 | 0 / 0 | off (on 0) | same | (reads) | same | +0.09 |
| L23 gate c216 | 76% / 8% | **res** (R2 0.67): res in {45..50, 55..61, 63..71, 73..81, 83..91, 98, 100..198, 200} | unexplained (best R2 0.40) | (reads) | same | +0.09 |
| L19 up c24 | 13% / 100% | unexplained (best R2 0.45) | always | (reads) | same | +0.09 |
| L28 down c11 | 2% / 98% | unexplained (best R2 0.30) | always | - | same | +0.09 |
| L26 down c7 | 52% / 0 | **res>=100** (R2 0.75): res>=100=0: 0.07, res>=100=1: 0.94 | off (on 0) | a: mod100 +3%; b: mod100 +3% | - | +0.09 |
| L29 up c600 | 50% / 0 | **res>=100** (R2 0.91): res>=100=0: 0.01, res>=100=1: 0.96 | off (on 0) | (reads) | same | +0.08 |
| L29 up c42 | 16% / 5% | **res** (R2 0.83): res in {9, 13, 33, 125..141, 143..144, 147, 186..187, 189} | **res** (R2 0.69): res in {6..7, 9, 11, 13} | (reads) | same | +0.08 |
| L31 up c176 | 1% / 0 | unexplained (best R2 0.35) | off (on 0) | (reads) | same | +0.08 |
| L19 down c151 | 9% / 0 | **tens(a,b)** (R2 0.72): a//10 in {3,4,8} -> b//10 in {9,10}; a//10 in {5} -> b//10 in {9}; a//10 in {9} -> b//10 in {3,4,5,8,9,10}; a//10 in {10} -> b//10 in {4,8,9} | off (on 0) | - | same | +0.08 |
| L23 down c29 | 58% / 3% | **res//10** (R2 0.85): (tens) res in {92..199} | unexplained (best R2 0.45) | res: mod100 +2% | - | +0.08 |
| L31 gate c376 | 33% / 0 | unexplained (best R2 0.30) | off (on 0) | (reads) | same | +0.08 |
| L26 up c318 | 22% / 0 | **res//10** (R2 0.77): (tens) res in {131, 137, 139..200} | off (on 0) | (reads) | same | +0.08 |
| L27 down c93 | 12% / 0 | **res//10** (R2 0.88): (tens) res in {150..198} | off (on 0) | - | same | +0.07 |
| L27 up c244 | 99% / 99% | always | same | (reads) | same | +0.07 |
| L31 down c15 | 7% / 90% | unexplained (best R2 0.19) | unexplained (best R2 0.27) | - | same | +0.07 |
| L26 down c75 | 1% / 27% | unexplained (best R2 0.23) | **tens(a,b)** (R2 0.62): a//10 in {0} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,9}; a//10 in {3,4,5,6,7,8,10} -> b//10 in {0}; a//10 in {9} -> b//10 in {0,1,2,3,4,5} | - | same | +0.07 |
| L31 up c7 | 19% / 100% | unexplained (best R2 0.39) | always | (reads) | same | +0.06 |
| L24 down c34 | 28% / 0 | **res//10** (R2 0.89): (tens) res in {126..200} | off (on 0) | a: mod100 +3%; b: mod100 +3% | - | +0.06 |
| L22 down c232 | 1% / 76% | unexplained (best R2 0.05) | **tens(a,b)** (R2 0.52): a//10 in {0,1,2} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {0,1,2,7,8,9}; a//10 in {7} -> b//10 in {0,1,2,3,8,9}; a//10 in {8} -> b//10 in {0,1,2,3,9}; a//10 in {9} -> b//10 in {0,1,2,3,4,5}; a//10 in {10} -> b//10 in {1} | - | a: mod100 +3%, mod50 +2% | +0.06 |
| L27 down c278 | 0 / 0 | off (on 0) | same | - | same | +0.06 |
| L19 gate c255 | 2% / 0 | unexplained (best R2 0.08) | off (on 0) | (reads) | same | +0.06 |
| L28 down c357 | 0 / 1% | off (on 0) | **res%100** (R2 0.68): res mod 100 in {2} | - | same | +0.06 |
| L31 up c19 | 88% / 93% | unexplained (best R2 0.12) | unexplained (best R2 0.25) | (reads) | same | +0.05 |
| L26 gate c10 | 99% / 100% | always | same | (reads) | same | +0.05 |
| L31 down c756 | 99% / 98% | always | same | - | same | +0.05 |
| L31 down c10 | 0 / 6% | off (on 0) | **a** (R2 0.53): a in {1..2, 6..9} | - | same | +0.05 |
| L30 gate c7 | 99% / 93% | always | unexplained (best R2 0.15) | (reads) | same | +0.04 |
| L23 down c58 | 15% / 25% | **res//10** (R2 0.83): (tens) res in {3..56, 200} | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {0,1,2,3}; a//10 in {1,2,3} -> b//10 in {0,1,2,3,4}; a//10 in {4} -> b//10 in {0,1,2}; a//10 in {5} -> b//10 in {2} | - | same | +0.04 |
| L26 up c119 | 0 / 1% | off (on 0) | unexplained (best R2 0.06) | (reads) | same | +0.03 |
| L28 down c408 | 0 / 1% | off (on 0) | **res%100** (R2 0.66): res mod 100: no class above 0.5 (max 0.48) | - | same | +0.03 |
| L24 gate c34 | 41% / 9% | **res//10** (R2 0.86): (tens) res in {2..15, 17, 114..200} | **res%100** (R2 0.52): res mod 100 in {0, 10..11} | (reads) | same | +0.03 |
| L20 o c498 (H2) | 1% / 2% | unexplained (best R2 0.07) | unexplained (best R2 0.17) | - | same | +0.03 |
| L28 down c660 | 2% / 2% | **res%100** (R2 0.52): res mod 100 in {82} | **res%100** (R2 0.66): res mod 100 in {0} | - | same | +0.03 |
| L26 down c85 | 10% / 14% | **res//10** (R2 0.73): (tens) res in {2..45} | **tens(a,b)** (R2 0.71): a//10 in {0} -> b//10 in {0,1,2,3,4}; a//10 in {1,2} -> b//10 in {0,1,2}; a//10 in {3,4} -> b//10 in {0} | - | same | +0.02 |
| L29 gate c286 | 7% / 79% | **res** (R2 0.62): res in {2..11, 16..17, 19..21, 25..37, 39..41} | unexplained (best R2 0.18) | (reads) | same | +0.02 |
| L31 down c38 | 0 / 0 | off (on 0) | same | - | same | +0.02 |
| L31 gate c23 | 94% / 98% | unexplained (best R2 0.10) | always | (reads) | same | +0.01 |
| L20 o c243 (H2) | 0 / 2% | off (on 0) | **a** (R2 0.57): a in {45} | - | same | +0.01 |
| L25 down c748 | 0 / 1% | off (on 0) | **res%100** (R2 0.63): res mod 100: no class above 0.5 (max 0.45) | - | same | +0.01 |
| L29 up c325 | 3% / 4% | unexplained (best R2 0.16) | same | (reads) | same | +0.01 |
| L31 up c162 | 14% / 0 | **res** (R2 0.62): res in {81..83, 117..119, 121..124, 143, 147..149, 161..164, 167..169, 174, 179, 181..184, 192, 197..198} | off (on 0) | (reads) | same | +0.00 |
| L30 down c594 | 5% / 0 | **res%100** (R2 0.60): res mod 100: no class above 0.5 (max 0.44) | off (on 0) | - | same | +0.00 |
| L19 down c423 | 11% / 0 | **tens(a,b)** (R2 0.74): a//10 in {1} -> b//10 in {2,3}; a//10 in {2,3} -> b//10 in {1,2,3} | off (on 0) | - | same | -0.00 |
| L22 down c107 | 13% / 8% | **res//10** (R2 0.76): (tens) res in {2..49, 200} | **tens(a,b)** (R2 0.73): a//10 in {0,1,2} -> b//10 in {0,1,2} | - | same | -0.00 |
| L29 down c325 | 5% / 5% | unexplained (best R2 0.27) | unexplained (best R2 0.21) | - | same | -0.01 |
| L31 down c844 | 0 / 0 | off (on 0) | same | - | same | -0.01 |
| L23 down c257 | 0 / 0 | off (on 0) | same | - | same | -0.01 |
| L22 o c147 (H28) | 0 / 0 | off (on 0) | same | - | same | -0.01 |
| L31 down c375 | 0 / 0 | off (on 0) | same | - | same | -0.01 |
| L20 down c529 | 1% / 0 | unexplained (best R2 0.37) | off (on 0) | - | same | -0.01 |
| L20 o c196 (H2) | 0 / 0 | off (on 0) | same | - | same | -0.01 |
| L19 gate c731 | 0 / 0 | off (on 0) | same | (reads) | same | -0.01 |
| L21 down c439 | 0 / 1% | off (on 0) | **tens(a,b)** (R2 0.66): a//10 in {10} -> b//10 in {3,4,5,6,7,8,9} | - | same | -0.01 |
| L28 gate c99 | 0 / 0 | off (on 0) | same | (reads) | same | -0.02 |
| L24 gate c560 | 0 / 1% | off (on 0) | unexplained (best R2 0.48) | (reads) | same | -0.02 |
| L31 gate c18 | 0 / 19% | off (on 0) | **a//10** (R2 0.69): (tens) a in {1..19} | (reads) | same | -0.02 |
| L31 down c66 | 1% / 0 | unexplained (best R2 0.36) | off (on 0) | - | same | -0.02 |
| L31 gate c279 | 1% / 0 | unexplained (best R2 0.35) | off (on 0) | (reads) | same | -0.02 |
| L31 gate c62 | 0 / 0 | off (on 0) | same | (reads) | same | -0.02 |
| L31 gate c407 | 2% / 0 | **res** (R2 0.69): res in {111..113} | off (on 0) | (reads) | same | -0.02 |
| L31 down c998 | 3% / 0 | **res%100** (R2 0.75): res mod 100: no class above 0.5 (max 0.45) | off (on 0) | - | same | -0.02 |
| L31 o c18 (H14) | 0 / 0 | off (on 0) | same | - | same | -0.02 |
| L31 gate c106 | 0 / 0 | off (on 0) | same | (reads) | same | -0.02 |
| L31 down c6 | 100% / 100% | always | same | - | same | -0.02 |
| L31 up c66 | 0 / 0 | off (on 0) | same | (reads) | same | -0.02 |
| L31 o c58 (H14) | 0 / 0 | off (on 0) | same | - | same | -0.02 |
| L26 down c271 | 0 / 1% | off (on 0) | **cmp(a,b)** (R2 0.64): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.90, cmp(a,b)=1: 0.00 | - | same | -0.02 |
| L30 down c1003 | 0 / 1% | off (on 0) | **cmp(a,b)** (R2 0.74): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.92, cmp(a,b)=1: 0.00 | - | same | -0.02 |
| L21 gate c33 | 0 / 0 | off (on 0) | same | (reads) | same | -0.02 |
| L31 gate c43 | 1% / 0 | unexplained (best R2 0.13) | off (on 0) | (reads) | same | -0.02 |
| L31 down c13 | 0 / 1% | off (on 0) | unexplained (best R2 0.38) | - | same | -0.03 |
| L31 gate c63 | 1% / 2% | unexplained (best R2 0.33) | unexplained (best R2 0.24) | (reads) | same | -0.03 |
| L31 up c307 | 0 / 0 | off (on 0) | same | (reads) | same | -0.03 |
| L20 gate c585 | 0 / 0 | off (on 0) | same | (reads) | same | -0.03 |
| L31 down c256 | 0 / 0 | off (on 0) | same | - | same | -0.03 |
| L23 down c13 | 7% / 2% | **tens(a,b)** (R2 0.64): a//10 in {0,1,2,5,7} -> b//10 in {10}; a//10 in {8} -> b//10 in {9,10}; a//10 in {9} -> b//10 in {8,9,10}; a//10 in {10} -> b//10 in {0,6,7,8,9,10} | unexplained (best R2 0.45) | - | same | -0.03 |
| L22 o c10 (H15) | 29% / 19% | **tens(a,b)** (R2 0.78): a//10 in {0,1,2} -> b//10 in {8,9,10}; a//10 in {3,4,5,6,7} -> b//10 in {9,10}; a//10 in {8} -> b//10 in {0,2,3,7,8,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.82): a//10 in {0,1,2,3,4} -> b//10 in {9,10}; a//10 in {5,6,8} -> b//10 in {10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | - | same | -0.03 |
| L25 up c218 | 100% / 100% | always | same | (reads) | same | -0.04 |
| L25 down c6 | 100% / 100% | always | same | - | same | -0.04 |
| L26 down c14 | 100% / 100% | always | same | - | same | -0.04 |
| L21 down c9 | 18% / 93% | unexplained (best R2 0.46) | **res%100** (R2 0.46): res mod 100 in {0..2, 4..98} | - | res: mod100 +3% | -0.04 |
| L31 down c64 | 0 / 0 | off (on 0) | same | - | same | -0.04 |
| L27 gate c650 | 5% / 0 | **res** (R2 0.62): res in {137, 140..149} | off (on 0) | (reads) | same | -0.04 |
| L31 down c236 | 0 / 0 | off (on 0) | same | - | same | -0.04 |
| L27 down c381 | 2% / 6% | unexplained (best R2 0.10) | unexplained (best R2 0.29) | - | same | -0.04 |
| L22 gate c160 | 0 / 49% | off (on 0) | **tens(a,b)** (R2 0.57): a//10 in {1} -> b//10 in {1,2,3}; a//10 in {2} -> b//10 in {2,4,5,6}; a//10 in {3} -> b//10 in {2,3,4,5,6,7,8}; a//10 in {4} -> b//10 in {3,4,5,6,7,8}; a//10 in {5,6,10} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {7,8} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10} | (reads) | same | -0.04 |
| L30 gate c35 | 42% / 0 | **res>=100** (R2 0.62): res>=100=0: 0.02, res>=100=1: 0.79 | off (on 0) | (reads) | same | -0.04 |
| L31 down c760 | 2% / 0 | **res//10** (R2 0.82): (tens) res in {182..200} | off (on 0) | - | same | -0.04 |
| L31 up c13 | 0 / 0 | off (on 0) | same | (reads) | same | -0.04 |
| L31 up c746 | 10% / 0 | **res** (R2 0.74): res in {111, 123, 127..134, 137..143} | off (on 0) | (reads) | same | -0.05 |
| L20 o c317 (H2) | 4% / 8% | unexplained (best R2 0.15) | unexplained (best R2 0.35) | - | same | -0.05 |
| L31 up c538 | 0 / 0 | off (on 0) | same | (reads) | same | -0.06 |
| L31 gate c418 | 33% / 1% | **res//10** (R2 0.58): (tens) res in {122..124, 126..200} | unexplained (best R2 0.35) | (reads) | same | -0.06 |
| L28 down c145 | 28% / 0 | **res//10** (R2 0.76): (tens) res in {121..170, 172..175, 178, 180..186, 194, 198, 200} | off (on 0) | - | same | -0.06 |
| L19 gate c23 | 79% / 28% | **res//10** (R2 0.90): (tens) res in {67..200} | **tens(a,b)** (R2 0.71): a//10 in {0,1,2} -> b//10 in {7,8,9,10}; a//10 in {3} -> b//10 in {9,10}; a//10 in {4} -> b//10 in {10}; a//10 in {7,10} -> b//10 in {0,1,2,3}; a//10 in {8} -> b//10 in {0,1,2,3,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5} | (reads) | same | -0.06 |
| L26 up c284 | 21% / 94% | **res** (R2 0.54): res in {3..33, 57, 59..65, 120..121, 160..163, 200} | unexplained (best R2 0.25) | (reads) | same | -0.06 |
| L31 down c52 | 0 / 3% | off (on 0) | **a** (R2 0.68): a in {1..3} | - | same | -0.06 |
| L31 down c983 | 0 / 7% | off (on 0) | unexplained (best R2 0.38) | - | same | -0.06 |
| L31 up c10 | 99% / 99% | always | same | (reads) | same | -0.06 |
| L28 gate c91 | 11% / 0 | **res//10** (R2 0.85): (tens) res in {156..200} | off (on 0) | (reads) | same | -0.07 |
| L31 down c406 | 0 / 0 | off (on 0) | same | - | same | -0.07 |
| L26 up c72 | 99% / 13% | always | **res** (R2 0.51): res in {4..15, 88..89} | (reads) | same | -0.08 |
| L28 up c91 | 10% / 0 | **res//10** (R2 0.80): (tens) res in {153..155, 158..200} | off (on 0) | (reads) | same | -0.09 |
| L31 up c140 | 0 / 7% | off (on 0) | **a** (R2 0.77): a in {1..6} | (reads) | same | -0.10 |
| L31 gate c205 | 2% / 4% | unexplained (best R2 0.13) | **a** (R2 0.71): a in {2..4} | (reads) | same | -0.10 |
| L31 gate c58 | 0 / 3% | off (on 0) | **a** (R2 0.79): a in {2..4} | (reads) | same | -0.10 |
| L23 gate c41 | 16% / 50% | **res//10** (R2 0.80): (tens) res in {2..59} | **res//10** (R2 0.59): (tens) res in {-29, -27..31} | (reads) | same | -0.10 |
| L31 up c376 | 0 / 3% | off (on 0) | **a** (R2 0.82): a in {2..3} | (reads) | same | -0.10 |
| L31 gate c268 | 14% / 0 | **res//10** (R2 0.94): (tens) res in {149..199} | off (on 0) | (reads) | same | -0.10 |
| L31 down c201 | 13% / 0 | **res//10** (R2 0.81): (tens) res in {147, 151..188, 191, 193..196} | off (on 0) | - | same | -0.11 |
| L20 down c79 | 5% / 64% | unexplained (best R2 0.26) | **tens(a,b)** (R2 0.54): a//10 in {0,1} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,1,3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,2,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,4,5,6,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,6,8,9}; a//10 in {6,7,10} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1,2,3}; a//10 in {9} -> b//10 in {0,1,2,3,4,5} | - | a: mod100 +3% | -0.12 |
| L31 down c587 | 35% / 1% | unexplained (best R2 0.47) | unexplained (best R2 0.13) | - | same | -0.13 |
| L31 gate c249 | 0 / 3% | off (on 0) | **a** (R2 0.80): a in {1..4} | (reads) | same | -0.13 |
| L31 up c111 | 3% / 1% | unexplained (best R2 0.40) | unexplained (best R2 0.30) | (reads) | same | -0.15 |
| L21 down c85 | 16% / 4% | **tens(a,b)** (R2 0.58): a//10 in {0} -> b//10 in {0,1,2,3,4,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,10}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3,9,10} -> b//10 in {0} | unexplained (best R2 0.40) | - | same | -0.16 |
| L23 up c18 | 8% / 0 | **res//10** (R2 0.77): (tens) res in {160, 163..200} | unexplained (best R2 0.31) | (reads) | same | -0.16 |
| L26 down c10 | 0 / 96% | off (on 0) | always | - | same | -0.19 |
| L21 down c51 | 1% / 61% | unexplained (best R2 0.10) | **tens(a,b)** (R2 0.68): a//10 in {0,1} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {3,4} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {5,6,7,8,9}; a//10 in {6} -> b//10 in {6,7,8,9}; a//10 in {7} -> b//10 in {1,2,7,8,9,10}; a//10 in {8} -> b//10 in {1,2,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,10}; a//10 in {10} -> b//10 in {0,1,2,10} | - | a: mod100 +2% | -0.21 |
| L26 down c605 | 15% / 0 | **res//10** (R2 0.83): (tens) res in {147, 149..200} | off (on 0) | - | same | -0.21 |
| L29 down c12 | 6% / 92% | unexplained (best R2 0.23) | unexplained (best R2 0.29) | - | same | -0.22 |
| L22 down c56 | 10% / 5% | **res** (R2 0.61): res in {125..133, 135..137} | **tens(a,b)** (R2 0.61): a//10 in {6,7,8} -> b//10 in {0}; a//10 in {10} -> b//10 in {2,3,4,7} | - | same | -0.22 |
| L23 down c38 | 6% / 8% | unexplained (best R2 0.16) | unexplained (best R2 0.29) | - | same | -0.25 |
| L31 up c466 | 37% / 1% | **res//10** (R2 0.50): (tens) res in {106..107, 109..119, 121..139, 141..149, 151..188, 194..200} | unexplained (best R2 0.28) | (reads) | same | -0.25 |
| L22 o c200 (H15) | 13% / 1% | **tens(a,b)** (R2 0.59): a//10 in {0,1,2,3,4,8} -> b//10 in {9,10}; a//10 in {5,6,7} -> b//10 in {10}; a//10 in {9} -> b//10 in {3,4,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.62): a//10 in {0,1,2,3,4,5,6} -> b//10 in {10}; a//10 in {10} -> b//10 in {1,10} | - | same | -0.26 |
| L23 gate c38 | 3% / 5% | unexplained (best R2 0.13) | unexplained (best R2 0.21) | (reads) | same | -0.28 |
| L31 down c29 | 99% / 45% | always | unexplained (best R2 0.42) | - | same | -0.33 |
| L31 up c839 | 66% / 3% | **tens(a,b)** (R2 0.67): a//10 in {0,1,2} -> b//10 in {6,7,8,9,10}; a//10 in {3} -> b//10 in {5,6,7,8,9,10}; a//10 in {4} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {6,7} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {0,2,3,4,5,6,7,8,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | unexplained (best R2 0.25) | (reads) | same | -0.36 |
| L20 down c172 | 5% / 3% | **tens(a,b)** (R2 0.66): a//10 in {9} -> b//10 in {1,2,3,6}; a//10 in {10} -> b//10 in {1,2,3} | **a** (R2 0.79): a in {99..100} | - | same | -0.36 |
| L23 up c38 | 6% / 7% | unexplained (best R2 0.17) | unexplained (best R2 0.27) | (reads) | same | -0.38 |
| L22 gate c29 | 48% / 1% | **res//10** (R2 0.82): (tens) res in {2, 107..200} | unexplained (best R2 0.43) | (reads) | same | -0.40 |
| L31 gate c32 | 15% / 85% | unexplained (best R2 0.40) | unexplained (best R2 0.28) | (reads) | same | -0.51 |
| L22 down c29 | 45% / 1% | **res//10** (R2 0.89): (tens) res in {108..200} | **tens(a,b)** (R2 0.51): a//10 in {1,2,3,4,5} -> b//10 in {10} | a: mod100 +4%, mod50 +2%; b: mod100 +5%, mod50 +2% | - | -0.52 |
| L31 down c27 | 13% / 1% | **res//10** (R2 0.60): (tens) res in {142..143, 147..154, 157..164, 167..199} | unexplained (best R2 0.15) | - | same | -0.53 |
| L19 gate c24 | 27% / 8% | **res//10** (R2 0.79): (tens) res in {2..69} | **tens(a,b)** (R2 0.67): a//10 in {0,1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {2} | (reads) | same | -0.55 |
| L31 down c622 | 11% / 79% | unexplained (best R2 0.38) | **tens(a,b)** (R2 0.52): a//10 in {0,1} -> b//10 in {0,1,2,3}; a//10 in {2} -> b//10 in {1,2,3,4,5,6}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,7,8,9}; a//10 in {4} -> b//10 in {1,2,3,4,5,6,7,8,9}; a//10 in {5,6,7,8,9} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,4,5,6,7,8,9,10} | - | a: mod100 +5%, mod50 +5%; b: mod50 +3%, mod25 +3%; res: mod100 +7%, mod50 +3% | -0.65 |
| L31 down c41 | 65% / 17% | **tens(a,b)** (R2 0.63): a//10 in {0,1,2} -> b//10 in {6,7,8,9,10}; a//10 in {3} -> b//10 in {5,6,7,8,9,10}; a//10 in {4,6} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {0,2,3,4,5,6,7,8,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | unexplained (best R2 0.36) | - | same | -0.68 |

</details>

