# §3 — complete lists for the result computation

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

## L16-L18 MLP components at `=`

<details><summary>units(a,b): 114 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 down c76 | 26% / 15% | **units(a,b)** (R2 0.92): a%10 in {2,7} -> b%10 in {0,1,5,6}; a%10 in {3,8} -> b%10 in {0,1,4,5,6,9}; a%10 in {4} -> b%10 in {0,4,5,9}; a%10 in {9} -> b%10 in {0,4,5} | **units(a,b)** (R2 0.52): a%10 in {2} -> b%10 in {0,4,5}; a%10 in {3} -> b%10 in {0,1,4,5,6}; a%10 in {7,8} -> b%10 in {0,5} | a: mod5 +3%; res: mod5 +20% | res: mod5 +7% |
| L16 down c79 | 33% / 28% | **units(a,b)** (R2 0.93): a%10 in {0,1,5,6} -> b%10 in {0,1,4,5,6,9}; a%10 in {4,9} -> b%10 in {0,1,5,6} | **units(a,b)** (R2 0.76): a%10 in {0,1,5,6} -> b%10 in {0,1,4,5,6,9}; a%10 in {4,9} -> b%10 in {0,4,5,9} | a: mod5 +5%; b: mod5 +2%; res: mod5 +11% | a: mod5 +4%; b: mod5 +3%; res: mod5 +8% |
| L16 down c94 | 21% / 22% | **units(a,b)** (R2 0.89): a%10 in {1,2,6,7} -> b%10 in {0,4,5,9}; a%10 in {3,8} -> b%10 in {4,9} | unexplained (best R2 0.47) | res: mod5 +16% | res: mod5 +9% |
| L16 down c107 | 22% / 15% | **units(a,b)** (R2 0.90): a%10 in {3,4,5,6,7,8} -> b%10 in {3,4,5}; a%10 in {9} -> b%10 in {3,4} | unexplained (best R2 0.50) | a: mod10 +2%; b: mod10 +4% | b: mod10 +3%; res: mod10 +3% |
| L16 down c127 | 22% / 18% | **units(a,b)** (R2 0.90): a%10 in {0,5} -> b%10 in {4,9}; a%10 in {1} -> b%10 in {0,3,4,5,8,9}; a%10 in {2,6,7} -> b%10 in {3,4,8,9} | **units(a,b)** (R2 0.69): a%10 in {0,5} -> b%10 in {1,6}; a%10 in {1} -> b%10 in {0,1,2,5,6,7}; a%10 in {2} -> b%10 in {1,2,7}; a%10 in {6,7} -> b%10 in {1,2,6,7} | res: mod5 +17% | res: mod5 +16% |
| L16 down c275 | 21% / 14% | **units(a,b)** (R2 0.89): a%10 in {3,8} -> b%10 in {0,1,2,5,6,7}; a%10 in {4,9} -> b%10 in {0,1,5,6} | **units(a,b)** (R2 0.58): a%10 in {3} -> b%10 in {0,3,4,8,9}; a%10 in {4,9} -> b%10 in {0,4,5,9}; a%10 in {8} -> b%10 in {3,4,8,9} | res: mod5 +9% | res: mod5 +7% |
| L16 gate c76 | 46% / 27% | **units(a,b)** (R2 0.85): a%10 in {0,1,5} -> b%10 in {2,3,4,7,8,9}; a%10 in {2,3,7,8} -> b%10 in {2,3,7,8}; a%10 in {4} -> b%10 in {3,8}; a%10 in {6} -> b%10 in {2,3,7,8,9}; a%10 in {9} -> b%10 in {2,3,8} | **units(a,b)** (R2 0.51): a%10 in {0,5} -> b%10 in {1,2,3,6,7,8}; a%10 in {1,6} -> b%10 in {2,3,7,8}; a%10 in {2,3,7,8,9} -> b%10 in {2,7}; a%10 in {4} -> b%10 in {2} | (reads) | same |
| L16 gate c94 | 50% / 35% | **units(a,b)** (R2 0.89): a%10 in {0,1,5} -> b%10 in {1,2,3,6,7,8}; a%10 in {2,7} -> b%10 in {2,6,7}; a%10 in {3,4,6,8,9} -> b%10 in {1,2,6,7,8} | **units(a,b)** (R2 0.54): a%10 in {0,1,4,5,9} -> b%10 in {2,3,4,8,9}; a%10 in {2} -> b%10 in {3,8}; a%10 in {3,6,8} -> b%10 in {3,4,8,9}; a%10 in {7} -> b%10 in {3} | (reads) | same |
| L16 gate c127 | 32% / 20% | **units(a,b)** (R2 0.85): a%10 in {0,3,5,8} -> b%10 in {4,9}; a%10 in {1,2,6,7} -> b%10 in {0,3,4,5,8,9} | **units(a,b)** (R2 0.54): a%10 in {0,5,7} -> b%10 in {1,6}; a%10 in {1,6} -> b%10 in {0,1,2,5,6,7}; a%10 in {2} -> b%10 in {1,2,6,7} | (reads) | same |
| L16 gate c392 | 30% / 16% | **units(a,b)** (R2 0.88): a%10 in {0,5,9} -> b%10 in {4,9}; a%10 in {1,2,3,6,7,8} -> b%10 in {3,4,8,9}; a%10 in {4} -> b%10 in {9} | unexplained (best R2 0.42) | (reads) | same |
| L16 gate c527 | 15% / 13% | **units(a,b)** (R2 0.81): a%10 in {4} -> b%10 in {3,4,5,6,7}; a%10 in {5,6} -> b%10 in {3,4,5,6} | unexplained (best R2 0.35) | (reads) | same |
| L16 up c76 | 47% / 31% | **units(a,b)** (R2 0.89): a%10 in {1,6} -> b%10 in {5}; a%10 in {2,4,7,9} -> b%10 in {0,1,4,5,6,9}; a%10 in {3,8} -> b%10 in {0,1,2,3,4,5,6,7,8,9} | unexplained (best R2 0.45) | (reads) | same |
| L16 up c127 | 28% / 19% | **units(a,b)** (R2 0.87): a%10 in {0,5} -> b%10 in {4,9}; a%10 in {1,6,7} -> b%10 in {0,3,4,5,8,9}; a%10 in {2} -> b%10 in {3,4,5,8,9} | unexplained (best R2 0.44) | (reads) | same |
| L16 up c275 | 23% / 6% | **units(a,b)** (R2 0.69): a%10 in {3,4} -> b%10 in {0,1,5,6}; a%10 in {7} -> b%10 in {6}; a%10 in {8} -> b%10 in {0,1,2,4,5,6,7,9}; a%10 in {9} -> b%10 in {0,1,4,5,6,9} | unexplained (best R2 0.23) | (reads) | same |
| L16 up c389 | 18% / 12% | **units(a,b)** (R2 0.71): a%10 in {0} -> b%10 in {0,5,6,7}; a%10 in {4,6,8,9} -> b%10 in {5}; a%10 in {5} -> b%10 in {0,1,2,3,4,5,6,7,8,9} | **a** (R2 0.49): a in {40, 45, 50, 55, 65, 75, 80, 85, 90, 95, 100} [coarser: a mod 5 in {0}, R2 0.82] | (reads) | same |
| L16 up c392 | 28% / 13% | **units(a,b)** (R2 0.79): a%10 in {0} -> b%10 in {4}; a%10 in {3} -> b%10 in {0,1,4,5,6,9}; a%10 in {4,9} -> b%10 in {0,1,3,4,5,6,9}; a%10 in {8} -> b%10 in {0,1,4,5,6} | unexplained (best R2 0.37) | (reads) | same |
| L17 down c22 | 25% / 25% | **units(a,b)** (R2 0.98): a%10 in {1,3,5,7,9} -> b%10 in {0,2,4,6,8} | **units(a,b)** (R2 0.81): a%10 in {1,3,5,7,9} -> b%10 in {0,2,4,6,8} | a: mod2 +8%; res: mod2 +42% | a: mod2 +13%; res: mod2 +45% |
| L17 down c34 | 35% / 21% | **units(a,b)** (R2 0.93): a%10 in {0} -> b%10 in {5,6,7}; a%10 in {1} -> b%10 in {5,6}; a%10 in {5} -> b%10 in {0,1,2,5,6,7,8,9}; a%10 in {6} -> b%10 in {0,1,5,6,7,8,9}; a%10 in {7} -> b%10 in {0,5,6,7,8,9}; a%10 in {8} -> b%10 in {5,6,7,8,9}; a%10 in {9} -> b%10 in {5,6,7,8} | **units(a,b)** (R2 0.55): a%10 in {0} -> b%10 in {4}; a%10 in {5,6} -> b%10 in {0,1,2,3,4,9}; a%10 in {7} -> b%10 in {0,1,2,3,4}; a%10 in {8} -> b%10 in {2,3,4}; a%10 in {9} -> b%10 in {3,4} | a: mod10 +8%, mod5 +2%; b: mod10 +6%; res: mod10 +25% | a: mod10 +3%; b: mod10 +4%; res: mod10 +16% |
| L17 down c37 | 43% / 31% | **units(a,b)** (R2 0.92): a%10 in {0,8,9} -> b%10 in {0,1,2,3,4,8,9}; a%10 in {1,2} -> b%10 in {0,1,2,3,8,9}; a%10 in {3} -> b%10 in {0,1,8,9}; a%10 in {4} -> b%10 in {0,8,9}; a%10 in {5} -> b%10 in {8,9} | **units(a,b)** (R2 0.68): a%10 in {0,9} -> b%10 in {0,1,2,7,8,9}; a%10 in {1,2,8} -> b%10 in {0,1,2,8,9}; a%10 in {3} -> b%10 in {0,1,2,9}; a%10 in {4} -> b%10 in {0,1} | a: mod10 +5%; b: mod10 +7%; res: mod10 +6% | a: mod10 +4%; b: mod10 +5%; res: mod10 +4% |
| L17 down c39 | 39% / 24% | **units(a,b)** (R2 0.87): a%10 in {0,4} -> b%10 in {5,6,7,8,9}; a%10 in {1} -> b%10 in {5,6,7,8}; a%10 in {2} -> b%10 in {7,8}; a%10 in {3} -> b%10 in {0,6,7,8,9}; a%10 in {5} -> b%10 in {6,7,8}; a%10 in {7} -> b%10 in {6,7,8,9}; a%10 in {8,9} -> b%10 in {0,5,6,7,8,9} | **units(a,b)** (R2 0.54): a%10 in {0,3,4,8,9} -> b%10 in {1,2,3,4}; a%10 in {1} -> b%10 in {2,3,4}; a%10 in {5} -> b%10 in {3,4}; a%10 in {7} -> b%10 in {1,2,3} | a: mod10 +6%; res: mod10 +19% | a: mod10 +3%; res: mod10 +14% |
| L17 down c47 | 33% / 19% | **units(a,b)** (R2 0.95): a%10 in {0,4,5,9} -> b%10 in {0,3,4,5,8,9}; a%10 in {3,8} -> b%10 in {0,4,5,9} | **units(a,b)** (R2 0.55): a%10 in {0,5} -> b%10 in {0,1,2,5,6,7}; a%10 in {3} -> b%10 in {0}; a%10 in {4} -> b%10 in {0,1,5,6}; a%10 in {8} -> b%10 in {5}; a%10 in {9} -> b%10 in {0,6} | a: mod5 +5%; b: mod5 +5%; res: mod5 +10% | a: mod5 +3%; res: mod5 +3% |
| L17 down c49 | 38% / 27% | **units(a,b)** (R2 0.90): a%10 in {5} -> b%10 in {0,1,2,3,5,6,7,9}; a%10 in {6} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {7} -> b%10 in {0,1,2,3,5,6,7,8,9}; a%10 in {8} -> b%10 in {0,1,2,5,6,8,9}; a%10 in {9} -> b%10 in {0,1,5,9} | **a** (R2 0.50): a in {1, 26..27, 36..37, 46..47, 55..58, 65..68, 75..78, 85..88, 95..97} [coarser: a mod 20 in {6..7, 15..17}, R2 0.80] | b: mod10 +2%; res: mod10 +15% | res: mod10 +12% |
| L17 down c52 | 26% / 17% | **units(a,b)** (R2 0.87): a%10 in {0} -> b%10 in {2,3,4}; a%10 in {1} -> b%10 in {1,2,3,4}; a%10 in {2} -> b%10 in {0,1,2,3,4}; a%10 in {3,4} -> b%10 in {0,1,2,3,4,9} | unexplained (best R2 0.49) | a: mod10 +5%; b: mod10 +4%; res: mod10 +16% | a: mod10 +2%; res: mod10 +8% |
| L17 down c61 | 40% / 15% | **units(a,b)** (R2 0.88): a%10 in {0,5} -> b%10 in {3,4,8,9}; a%10 in {1,6} -> b%10 in {3,8}; a%10 in {3,4,9} -> b%10 in {0,3,4,5,8,9}; a%10 in {8} -> b%10 in {0,1,3,4,5,6,7,8,9} | unexplained (best R2 0.26) | a: mod5 +4%; b: mod5 +4%; res: mod5 +10% | res: mod5 +3% |
| L17 down c67 | 25% / 20% | **units(a,b)** (R2 0.98): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} | **units(a,b)** (R2 0.73): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} | a: mod2 +18%; b: mod2 +15%; res: mod2 +29% | a: mod2 +6%; b: mod2 +6%; res: mod2 +17% |
| L17 down c70 | 18% / 4% | **units(a,b)** (R2 0.79): a%10 in {2} -> b%10 in {1,2,3,4,7,8,9}; a%10 in {4} -> b%10 in {2,7}; a%10 in {7} -> b%10 in {1,2,3,4,6,7,8,9}; a%10 in {9} -> b%10 in {2} | unexplained (best R2 0.22) | a: mod5 +4% | - |
| L17 down c86 | 31% / 16% | **units(a,b)** (R2 0.85): a%10 in {0} -> b%10 in {6,7,8}; a%10 in {1} -> b%10 in {5,6,7,8,9}; a%10 in {2} -> b%10 in {5,6,7,8}; a%10 in {3} -> b%10 in {0,1,2,6,7,8,9}; a%10 in {4} -> b%10 in {0,1,2,8,9}; a%10 in {5} -> b%10 in {0,1,8,9}; a%10 in {6} -> b%10 in {0,8,9} | unexplained (best R2 0.48) | a: mod10 +2%; res: mod10 +11% | res: mod10 +8% |
| L17 down c116 | 25% / 10% | **units(a,b)** (R2 0.90): a%10 in {0} -> b%10 in {1,2,7}; a%10 in {1,6} -> b%10 in {1,2,6,7}; a%10 in {2,7} -> b%10 in {0,1,2,5,6,7}; a%10 in {5} -> b%10 in {2,7} | unexplained (best R2 0.34) | res: mod5 +7% | res: mod5 +3% |
| L17 down c118 | 10% / 4% | **units(a,b)** (R2 0.81): a%10 in {6} -> b%10 in {8}; a%10 in {7} -> b%10 in {6,7,8,9}; a%10 in {8} -> b%10 in {6,7,8}; a%10 in {9} -> b%10 in {6,7} | unexplained (best R2 0.36) | - | same |
| L17 gate c22 | 26% / 26% | **units(a,b)** (R2 0.95): a%10 in {1,3,5,7,9} -> b%10 in {0,2,4,6,8} | **units(a,b)** (R2 0.79): a%10 in {1,3,5,7,9} -> b%10 in {0,2,4,6,8} | (reads) | same |
| L17 gate c34 | 35% / 24% | **units(a,b)** (R2 0.88): a%10 in {0} -> b%10 in {5,6}; a%10 in {5,6} -> b%10 in {0,1,2,3,5,6,7,8,9}; a%10 in {7} -> b%10 in {0,1,5,6,7,8,9}; a%10 in {8} -> b%10 in {5,6,7,8}; a%10 in {9} -> b%10 in {5,6,7} | unexplained (best R2 0.45) | (reads) | same |
| L17 gate c37 | 29% / 20% | **units(a,b)** (R2 0.83): a%10 in {0,1} -> b%10 in {0,1,2,8,9}; a%10 in {2,3,4,8,9} -> b%10 in {0,1,8,9} | **units(a,b)** (R2 0.52): a%10 in {0,1} -> b%10 in {0,1,2,9}; a%10 in {2} -> b%10 in {0,1,2}; a%10 in {8,9} -> b%10 in {0,1,9} | (reads) | same |
| L17 gate c39 | 38% / 26% | **units(a,b)** (R2 0.92): a%10 in {0,1,2} -> b%10 in {6,7,8}; a%10 in {3,4,5,6,7,8} -> b%10 in {6,7,8,9}; a%10 in {9} -> b%10 in {5,6,7,8,9} | **b** (R2 0.56): b in {1..4, 11..14, 21..24, 31..34, 41..44, 52..54, 62..63, 73, 82..83} [coarser: b mod 10 in {1..4}, R2 0.87] | (reads) | same |
| L17 gate c49 | 34% / 18% | **units(a,b)** (R2 0.91): a%10 in {5} -> b%10 in {0,1,2,3,6,7}; a%10 in {6,7} -> b%10 in {0,1,2,3,5,6,7,8,9}; a%10 in {8} -> b%10 in {0,1,2,5,6,7,8,9}; a%10 in {9} -> b%10 in {5,6} | unexplained (best R2 0.47) | (reads) | same |
| L17 gate c52 | 40% / 22% | **units(a,b)** (R2 0.88): a%10 in {0,1,3,7,8,9} -> b%10 in {1,2,3,4}; a%10 in {2} -> b%10 in {0,1,2,3,4}; a%10 in {4} -> b%10 in {0,1,2,3,4,9}; a%10 in {5} -> b%10 in {3,4}; a%10 in {6} -> b%10 in {2,3,4} | unexplained (best R2 0.28) | (reads) | same |
| L17 gate c67 | 25% / 18% | **units(a,b)** (R2 0.99): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} | **units(a,b)** (R2 0.62): a%10 in {1,5,7,9} -> b%10 in {1,3,5,7,9}; a%10 in {3} -> b%10 in {1,3,7,9} | (reads) | same |
| L17 gate c86 | 26% / 10% | **units(a,b)** (R2 0.85): a%10 in {5} -> b%10 in {2,3,4}; a%10 in {6} -> b%10 in {1,2,3,4}; a%10 in {7} -> b%10 in {0,1,2,3,4,9}; a%10 in {8} -> b%10 in {0,1,2,3,4,8,9}; a%10 in {9} -> b%10 in {0,1,2,3,9} | unexplained (best R2 0.37) | (reads) | same |
| L17 gate c116 | 25% / 11% | **units(a,b)** (R2 0.91): a%10 in {0} -> b%10 in {1,2,7}; a%10 in {1} -> b%10 in {0,1,2,6,7}; a%10 in {2,7} -> b%10 in {0,1,2,5,6,7}; a%10 in {5} -> b%10 in {2,7}; a%10 in {6} -> b%10 in {1,2,6,7} | unexplained (best R2 0.34) | (reads) | same |
| L17 up c34 | 64% / 17% | **units(a,b)** (R2 0.54): a%10 in {0} -> b%10 in {0,1,3,5,6,7,8,9}; a%10 in {1,5,6} -> b%10 in {0,1,5,6,7,8,9}; a%10 in {2} -> b%10 in {0,6,7,8,9}; a%10 in {3} -> b%10 in {0,6,8,9}; a%10 in {4} -> b%10 in {0,8,9}; a%10 in {7} -> b%10 in {0,5,6,7,8,9}; a%10 in {8} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {9} -> b%10 in {0,1,3,4,5,6,7,8,9} | unexplained (best R2 0.26) | (reads) | same |
| L17 up c37 | 35% / 17% | **units(a,b)** (R2 0.83): a%10 in {0,1,2,3,4,9} -> b%10 in {5,6,7}; a%10 in {5} -> b%10 in {0,4,5,6,7}; a%10 in {6,7} -> b%10 in {4,5,6,7}; a%10 in {8} -> b%10 in {5,6} | unexplained (best R2 0.39) | (reads) | same |
| L17 up c39 | 20% / 15% | **units(a,b)** (R2 0.87): a%10 in {1} -> b%10 in {8}; a%10 in {2} -> b%10 in {6,7,8,9}; a%10 in {3} -> b%10 in {0,5,6,7,8,9}; a%10 in {4} -> b%10 in {5,6,7,8,9}; a%10 in {5} -> b%10 in {6,7,8} | **units(a,b)** (R2 0.61): a%10 in {1} -> b%10 in {2}; a%10 in {2} -> b%10 in {1,2,3}; a%10 in {3} -> b%10 in {0,1,2,3,4}; a%10 in {4} -> b%10 in {1,2,3,4}; a%10 in {5} -> b%10 in {2,4} | (reads) | same |
| L17 up c47 | 32% / 18% | **units(a,b)** (R2 0.95): a%10 in {0,4,5,9} -> b%10 in {0,3,4,5,8,9}; a%10 in {3,8} -> b%10 in {0,4,5,9} | **units(a,b)** (R2 0.54): a%10 in {0} -> b%10 in {0,1,2,5,6,7}; a%10 in {3} -> b%10 in {0}; a%10 in {4} -> b%10 in {0,1,5,6}; a%10 in {5} -> b%10 in {0,1,2,5,6}; a%10 in {8} -> b%10 in {5}; a%10 in {9} -> b%10 in {0,6} | (reads) | same |
| L17 up c49 | 41% / 26% | **units(a,b)** (R2 0.84): a%10 in {0} -> b%10 in {0,1}; a%10 in {1,3} -> b%10 in {0,1,2}; a%10 in {2,4} -> b%10 in {0,1,2,9}; a%10 in {5,6} -> b%10 in {0,1,2,3,9}; a%10 in {7,8} -> b%10 in {0,1,2,5,9}; a%10 in {9} -> b%10 in {0,1,9} | unexplained (best R2 0.47) | (reads) | same |
| L17 up c52 | 25% / 25% | **units(a,b)** (R2 0.85): a%10 in {5} -> b%10 in {0,1,5,6,7,8,9}; a%10 in {6} -> b%10 in {0,5,6,7,8,9}; a%10 in {7} -> b%10 in {5,6,7,8,9}; a%10 in {8} -> b%10 in {5,6,7,8}; a%10 in {9} -> b%10 in {5,6,7} | unexplained (best R2 0.32) | (reads) | same |
| L17 up c61 | 32% / 11% | **units(a,b)** (R2 0.94): a%10 in {0,5} -> b%10 in {3,4,8,9}; a%10 in {3} -> b%10 in {0,4,5,8,9}; a%10 in {4,8,9} -> b%10 in {0,3,4,5,8,9} | unexplained (best R2 0.29) | (reads) | same |
| L17 up c67 | 25% / 20% | **units(a,b)** (R2 0.99): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} | **units(a,b)** (R2 0.72): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} | (reads) | same |
| L17 up c70 | 5% / 2% | **units(a,b)** (R2 0.72): a%10 in {2,7} -> b%10 in {4,9} | unexplained (best R2 0.23) | (reads) | same |
| L17 up c86 | 50% / 13% | **units(a,b)** (R2 0.64): a%10 in {0} -> b%10 in {2,3,4}; a%10 in {1} -> b%10 in {1,2,3,4}; a%10 in {2} -> b%10 in {0,1,2,3,4}; a%10 in {3} -> b%10 in {0,1,2,3,4,9}; a%10 in {4,5,6,7} -> b%10 in {0,1,2,3,4,8,9}; a%10 in {8} -> b%10 in {0,1,2,3,9}; a%10 in {9} -> b%10 in {0,1,2} | unexplained (best R2 0.20) | (reads) | same |
| L17 up c118 | 34% / 16% | **units(a,b)** (R2 0.73): a%10 in {0,1} -> b%10 in {2,3,4}; a%10 in {2} -> b%10 in {0,1,2,3,4,9}; a%10 in {3} -> b%10 in {0,1,2,3,4,7,8,9}; a%10 in {4} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {5,9} -> b%10 in {4}; a%10 in {6} -> b%10 in {3,4} | unexplained (best R2 0.30) | (reads) | same |
| L17 up c454 | 7% / 4% | **units(a,b)** (R2 0.57): a%10 in {1} -> b%10 in {1,6,8,9}; a%10 in {6} -> b%10 in {1,6} | unexplained (best R2 0.32) | (reads) | same |
| L17 up c508 | 27% / 11% | **units(a,b)** (R2 0.89): a%10 in {6} -> b%10 in {3,4,7,8,9}; a%10 in {7} -> b%10 in {2,3,4,6,7,8,9}; a%10 in {8} -> b%10 in {1,2,3,4,6,7,8,9}; a%10 in {9} -> b%10 in {2,3,4,6,7,8} | unexplained (best R2 0.35) | (reads) | same |
| L17 up c636 | 23% / 14% | **units(a,b)** (R2 0.91): a%10 in {3} -> b%10 in {0,1,8,9}; a%10 in {4} -> b%10 in {0,1,6,7,8,9}; a%10 in {5} -> b%10 in {0,1,5,6,7,8,9}; a%10 in {6} -> b%10 in {0,5,6,7,8,9} | unexplained (best R2 0.50) | (reads) | same |
| L18 down c4 | 41% / 40% | **units(a,b)** (R2 0.74): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8}; a%10 in {1,5} -> b%10 in {3,5,7}; a%10 in {3,7,9} -> b%10 in {1,3,5,7,9} | **units(a,b)** (R2 0.52): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} | a: mod2 +54%; b: mod2 +48%; res: mod2 +61% | a: mod2 +48%; b: mod2 +48%; res: mod2 +64% |
| L18 down c24 | 57% / 51% | **units(a,b)** (R2 0.87): a%10 in {0,5} -> b%10 in {1,2,6,7}; a%10 in {1} -> b%10 in {0,1,5,6,9}; a%10 in {2,7} -> b%10 in {0,3,4,5,8,9}; a%10 in {3} -> b%10 in {0,2,3,4,7,8,9}; a%10 in {4,9} -> b%10 in {1,2,3,6,7,8}; a%10 in {6} -> b%10 in {0,1,4,5,6,9}; a%10 in {8} -> b%10 in {2,3,4,7,8,9} | unexplained (best R2 0.49) | res: mod5 +16% | res: mod5 +12% |
| L18 down c26 | 69% / 42% | **units(a,b)** (R2 0.80): a%10 in {0,5} -> b%10 in {1,2,3,4,6,7,8,9}; a%10 in {1,6} -> b%10 in {0,1,2,3,5,6,7,8}; a%10 in {2,7} -> b%10 in {0,1,4,5,6,9}; a%10 in {3,8} -> b%10 in {0,3,4,5,8,9}; a%10 in {4,9} -> b%10 in {0,2,3,4,5,7,8,9} | **res** (R2 0.51): res in {-97, -48, -43..-42, -38..-37, -33, -28..-27, -23..-22, -18..-17, -13..-12, -8..-7, -3..-1, 1..3, 6..8, 11..13, 16..18, 21..23, 26..28, 31..33, 37..38, 42..43, 47..48, 52..53, 57..58, 62..63, 67..68, 72..73, 77..78, 82..83, 87..88, 92..93, 96..98} | res: mod5 +22% | res: mod5 +14% |
| L18 down c32 | 46% / 30% | **units(a,b)** (R2 0.85): a%10 in {0} -> b%10 in {3,4,5,6}; a%10 in {1,3,9} -> b%10 in {3,4,5,6,7}; a%10 in {2} -> b%10 in {5,6,7}; a%10 in {4,5} -> b%10 in {2,3,4,5,6,7}; a%10 in {6} -> b%10 in {1,2,3,4,5,6}; a%10 in {7} -> b%10 in {1,2,3,4}; a%10 in {8} -> b%10 in {4,5,6} | unexplained (best R2 0.50) | res: mod10 +19% | res: mod10 +13% |
| L18 down c62 | 25% / 28% | **units(a,b)** (R2 0.89): a%10 in {0} -> b%10 in {0,1,2,8,9}; a%10 in {1} -> b%10 in {0,1,7,8,9}; a%10 in {2} -> b%10 in {0,7,8,9}; a%10 in {7} -> b%10 in {1,2}; a%10 in {8} -> b%10 in {0,1,2}; a%10 in {9} -> b%10 in {0,1,2,9} | **units(a,b)** (R2 0.68): a%10 in {0} -> b%10 in {0,1,2,8,9}; a%10 in {1} -> b%10 in {0,1,2,3,8,9}; a%10 in {2} -> b%10 in {0,1,2,9}; a%10 in {7} -> b%10 in {8,9}; a%10 in {8} -> b%10 in {0,8,9}; a%10 in {9} -> b%10 in {0,1,7,8,9} | res: mod10 +9% | a: mod10 +2%; b: mod10 +2%; res: mod10 +13% |
| L18 down c65 | 36% / 29% | **units(a,b)** (R2 0.88): a%10 in {0,5} -> b%10 in {7,8,9}; a%10 in {1,6} -> b%10 in {6,7,8,9}; a%10 in {2,7} -> b%10 in {5,6,7,8}; a%10 in {3} -> b%10 in {4,5,6,7}; a%10 in {4} -> b%10 in {4,5,6}; a%10 in {8} -> b%10 in {5,6,7}; a%10 in {9} -> b%10 in {5,6,9} | unexplained (best R2 0.49) | res: mod10 +7% | res: mod10 +8% |
| L18 down c80 | 51% / 24% | **units(a,b)** (R2 0.82): a%10 in {0,5} -> b%10 in {0,1,5,6}; a%10 in {1,6} -> b%10 in {0,4,5,9}; a%10 in {2} -> b%10 in {3,4,7,8,9}; a%10 in {3,8} -> b%10 in {2,3,4,7,8,9}; a%10 in {4} -> b%10 in {1,2,6,7}; a%10 in {7} -> b%10 in {0,2,3,4,7,8,9}; a%10 in {9} -> b%10 in {1,2,6,7,8} | unexplained (best R2 0.46) | res: mod5 +7% | res: mod5 +8% |
| L18 down c88 | 30% / 19% | **units(a,b)** (R2 0.89): a%10 in {1} -> b%10 in {4,5,6}; a%10 in {2} -> b%10 in {3,4,5,6}; a%10 in {3} -> b%10 in {2,3,4,5,6}; a%10 in {4} -> b%10 in {1,2,3,4,5}; a%10 in {5} -> b%10 in {1,2,3,4,5,6,7}; a%10 in {6} -> b%10 in {1,5,6,7}; a%10 in {7} -> b%10 in {5,6} | unexplained (best R2 0.46) | res: mod10 +8% | res: mod10 +5% |
| L18 down c98 | 34% / 27% | **units(a,b)** (R2 0.81): a%10 in {0,1} -> b%10 in {4,5,6,7}; a%10 in {2} -> b%10 in {0,4,5,6,8,9}; a%10 in {3} -> b%10 in {0,4,5,8,9}; a%10 in {4} -> b%10 in {7,8}; a%10 in {7} -> b%10 in {8,9}; a%10 in {8} -> b%10 in {6,7,8,9}; a%10 in {9} -> b%10 in {5,6,7,8} | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {2,3,4,5,6}; a%10 in {1} -> b%10 in {1,3,4,5,6}; a%10 in {2} -> b%10 in {0,1,2,4,5,6}; a%10 in {3} -> b%10 in {0,1,2,3}; a%10 in {4} -> b%10 in {1,2,3}; a%10 in {8} -> b%10 in {2,3,4}; a%10 in {9} -> b%10 in {2,3,4,5} | res: mod10 +7% | res: mod10 +8% |
| L18 down c101 | 42% / 24% | **units(a,b)** (R2 0.85): a%10 in {1,6} -> b%10 in {4,8,9}; a%10 in {2,7} -> b%10 in {2,3,4,7,8,9}; a%10 in {3,8} -> b%10 in {1,2,3,4,6,7,8,9}; a%10 in {4,9} -> b%10 in {1,2,6,7,8} | unexplained (best R2 0.45) | res: mod5 +8% | res: mod5 +6% |
| L18 down c103 | 4% / 0 | **units(a,b)** (R2 0.75): a%10 in {2,7} -> b%10 in {2,7} | off (on 0) | - | same |
| L18 down c120 | 19% / 16% | **units(a,b)** (R2 0.85): a%10 in {0} -> b%10 in {3}; a%10 in {5} -> b%10 in {5,6,7}; a%10 in {6} -> b%10 in {4,5,6}; a%10 in {7} -> b%10 in {3,4,5,6}; a%10 in {8} -> b%10 in {2,3,4,5}; a%10 in {9} -> b%10 in {2,3,4} | **units(a,b)** (R2 0.51): a%10 in {0} -> b%10 in {7,8}; a%10 in {5} -> b%10 in {4,5}; a%10 in {6} -> b%10 in {4,5,6}; a%10 in {7} -> b%10 in {5,6,7}; a%10 in {8} -> b%10 in {5,6,7,8}; a%10 in {9} -> b%10 in {6,7,8} | res: mod10 +3% | res: mod10 +4% |
| L18 down c142 | 54% / 21% | **units(a,b)** (R2 0.88): a%10 in {0,5} -> b%10 in {0,3,4,5,9}; a%10 in {1,6} -> b%10 in {2,3,4,7,8,9}; a%10 in {2,7} -> b%10 in {1,2,3,6,7,8}; a%10 in {3,8} -> b%10 in {1,2,6,7}; a%10 in {4} -> b%10 in {0,1,5,6}; a%10 in {9} -> b%10 in {0,1,4,5,6} | **res** (R2 0.52): res in {0, 3..5, 8..10, 13..15, 19..20, 23..25, 29..30, 34..35, 39, 44..45, 59, 69, 74..75, 79, 84..85, 89..90, 94..95, 99} | res: mod5 +8% | res: mod5 +4% |
| L18 down c261 | 6% / 5% | **units(a,b)** (R2 0.93): a%10 in {5} -> b%10 in {5,6,7}; a%10 in {6} -> b%10 in {5,6}; a%10 in {7} -> b%10 in {5} | unexplained (best R2 0.48) | - | same |
| L18 down c375 | 37% / 26% | **units(a,b)** (R2 0.87): a%10 in {0,5} -> b%10 in {2,3,4,7,8,9}; a%10 in {1,6} -> b%10 in {2,3,7,8}; a%10 in {3,8} -> b%10 in {0,5}; a%10 in {4,9} -> b%10 in {0,3,4,5,8,9} | unexplained (best R2 0.46) | res: mod5 +5% | res: mod5 +6% |
| L18 down c474 | 18% / 9% | **units(a,b)** (R2 0.92): a%10 in {0} -> b%10 in {2,3,4}; a%10 in {1} -> b%10 in {1,2,3,4}; a%10 in {2} -> b%10 in {1,2,3}; a%10 in {3} -> b%10 in {0,1,2}; a%10 in {4} -> b%10 in {0,1,9}; a%10 in {9} -> b%10 in {4,5} | unexplained (best R2 0.38) | res: mod10 +3% | - |
| L18 gate c4 | 26% / 40% | **units(a,b)** (R2 0.94): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} | **units(a,b)** (R2 0.58): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} | (reads) | same |
| L18 gate c24 | 39% / 23% | **units(a,b)** (R2 0.90): a%10 in {1,6} -> b%10 in {1,2,3,6,7,8}; a%10 in {2} -> b%10 in {1,2,3,5,6,7,8}; a%10 in {3,8} -> b%10 in {1,2,6,7}; a%10 in {5} -> b%10 in {2,7}; a%10 in {7} -> b%10 in {0,1,2,3,5,6,7,8} | unexplained (best R2 0.48) | (reads) | same |
| L18 gate c26 | 61% / 25% | **units(a,b)** (R2 0.78): a%10 in {0,5} -> b%10 in {2,3,7,8,9}; a%10 in {2,7} -> b%10 in {2,3,4,7,8,9}; a%10 in {3,8,9} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {4} -> b%10 in {0,1,2,3,4,6,7,8,9} | unexplained (best R2 0.38) | (reads) | same |
| L18 gate c32 | 57% / 16% | **units(a,b)** (R2 0.85): a%10 in {0,3,9} -> b%10 in {2,3,4,5,6,7}; a%10 in {1,2,8} -> b%10 in {3,4,5,6,7}; a%10 in {4,5,6} -> b%10 in {1,2,3,4,5,6,7}; a%10 in {7} -> b%10 in {3,4,5,6} | unexplained (best R2 0.39) | (reads) | same |
| L18 gate c62 | 32% / 21% | **units(a,b)** (R2 0.88): a%10 in {0,1} -> b%10 in {2,3,4}; a%10 in {2} -> b%10 in {1,2,3,4}; a%10 in {3} -> b%10 in {0,1,2,3,4}; a%10 in {4} -> b%10 in {0,1,2,3,4,5,9}; a%10 in {5} -> b%10 in {0,1,2,3,4,5}; a%10 in {6} -> b%10 in {3,4}; a%10 in {9} -> b%10 in {4} | unexplained (best R2 0.42) | (reads) | same |
| L18 gate c65 | 28% / 16% | **units(a,b)** (R2 0.88): a%10 in {0} -> b%10 in {6,7}; a%10 in {1,2,7} -> b%10 in {5,6,7,8}; a%10 in {3,8} -> b%10 in {5,6,7}; a%10 in {5,6} -> b%10 in {6,7,8}; a%10 in {9} -> b%10 in {5,6} | unexplained (best R2 0.37) | (reads) | same |
| L18 gate c80 | 26% / 16% | **units(a,b)** (R2 0.75): a%10 in {0} -> b%10 in {0,1,5,6,9}; a%10 in {1,6} -> b%10 in {0,4,5,9}; a%10 in {2} -> b%10 in {9}; a%10 in {5} -> b%10 in {0,1,5,6}; a%10 in {7} -> b%10 in {0,4,9}; a%10 in {9} -> b%10 in {1,6} | **units(a,b)** (R2 0.65): a%10 in {0,5} -> b%10 in {0,4,5,9}; a%10 in {1,6} -> b%10 in {0,1,5,6} | (reads) | same |
| L18 gate c88 | 44% / 37% | **units(a,b)** (R2 0.90): a%10 in {0,1,2,6} -> b%10 in {3,4,5,6}; a%10 in {3} -> b%10 in {2,3,4,5,6,7}; a%10 in {4,5} -> b%10 in {1,2,3,4,5,6,7}; a%10 in {7,8} -> b%10 in {5}; a%10 in {9} -> b%10 in {4,5,6} | **units(a,b)** (R2 0.53): a%10 in {0,1,2,3,6} -> b%10 in {3,4,5,6,7}; a%10 in {4,5} -> b%10 in {3,4,5,6,7,9}; a%10 in {7} -> b%10 in {5,6}; a%10 in {8} -> b%10 in {5}; a%10 in {9} -> b%10 in {4,5,6} | (reads) | same |
| L18 gate c98 | 27% / 23% | **units(a,b)** (R2 0.86): a%10 in {0} -> b%10 in {5,6,7}; a%10 in {1} -> b%10 in {0,4,5,6,7,9}; a%10 in {2} -> b%10 in {0,4,5,6,7,8,9}; a%10 in {3} -> b%10 in {0,5,6,7,8,9}; a%10 in {4} -> b%10 in {6,7,8}; a%10 in {9} -> b%10 in {6,7} | **units(a,b)** (R2 0.56): a%10 in {0} -> b%10 in {3,4,5}; a%10 in {1,2} -> b%10 in {0,1,2,3,4,5,6}; a%10 in {3} -> b%10 in {1,2,3,4}; a%10 in {4} -> b%10 in {2,3,4}; a%10 in {9} -> b%10 in {3,4} | (reads) | same |
| L18 gate c101 | 55% / 30% | **units(a,b)** (R2 0.91): a%10 in {0,5} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {1,6} -> b%10 in {0,1,2,3,5,6,7,8}; a%10 in {2,7} -> b%10 in {0,1,5,6}; a%10 in {3} -> b%10 in {0,1,5}; a%10 in {4,8,9} -> b%10 in {0,5} | unexplained (best R2 0.50) | (reads) | same |
| L18 gate c120 | 34% / 32% | **units(a,b)** (R2 0.81): a%10 in {0,1} -> b%10 in {3,4}; a%10 in {3,4} -> b%10 in {3,4,5,6}; a%10 in {5} -> b%10 in {3,4,5}; a%10 in {6,7,8,9} -> b%10 in {2,3,4,5} | **units(a,b)** (R2 0.56): a%10 in {0,3,4,7,8,9} -> b%10 in {5,6,7,8}; a%10 in {1} -> b%10 in {6,7,8}; a%10 in {2} -> b%10 in {7}; a%10 in {5,6} -> b%10 in {4,5,6,7,8} | (reads) | same |
| L18 gate c130 | 8% / 4% | **units(a,b)** (R2 0.83): a%10 in {0} -> b%10 in {1,2}; a%10 in {1} -> b%10 in {0,1,2}; a%10 in {2} -> b%10 in {0,1} | unexplained (best R2 0.34) | (reads) | same |
| L18 gate c142 | 32% / 23% | **units(a,b)** (R2 0.88): a%10 in {1,6} -> b%10 in {2,7}; a%10 in {2,7} -> b%10 in {1,2,3,4,6,7,8,9}; a%10 in {3} -> b%10 in {2,3,4,7,8,9}; a%10 in {8} -> b%10 in {2,3,7,8,9} | unexplained (best R2 0.46) | (reads) | same |
| L18 gate c261 | 26% / 16% | **units(a,b)** (R2 0.84): a%10 in {0} -> b%10 in {6}; a%10 in {1} -> b%10 in {5,6}; a%10 in {2} -> b%10 in {4,5}; a%10 in {3} -> b%10 in {4}; a%10 in {4} -> b%10 in {1,2,3,7}; a%10 in {5} -> b%10 in {0,1,2,5,6,7}; a%10 in {6} -> b%10 in {0,1,5,6,7,9}; a%10 in {7} -> b%10 in {0,5,6} | unexplained (best R2 0.40) | (reads) | same |
| L18 gate c375 | 28% / 19% | **units(a,b)** (R2 0.85): a%10 in {0,5} -> b%10 in {0,2,3,4,5,7,8,9}; a%10 in {3,8} -> b%10 in {0}; a%10 in {4,9} -> b%10 in {0,4,5,9} | **units(a,b)** (R2 0.62): a%10 in {0} -> b%10 in {0,1,2,3,5,6,7,8,9}; a%10 in {5} -> b%10 in {0,1,2,5,6,7}; a%10 in {9} -> b%10 in {0} | (reads) | same |
| L18 gate c474 | 29% / 21% | **units(a,b)** (R2 0.92): a%10 in {0,5,6} -> b%10 in {8,9}; a%10 in {1,2,7} -> b%10 in {7,8,9}; a%10 in {3} -> b%10 in {7,8}; a%10 in {8} -> b%10 in {6,7,8,9}; a%10 in {9} -> b%10 in {0,1,5,6,7,8,9} | **units(a,b)** (R2 0.53): a%10 in {0} -> b%10 in {0,1,2}; a%10 in {1,9} -> b%10 in {0,1,2,3}; a%10 in {2,7,8} -> b%10 in {1,2,3}; a%10 in {3} -> b%10 in {2,3}; a%10 in {5,6} -> b%10 in {1} | (reads) | same |
| L18 gate c556 | 7% / 3% | **units(a,b)** (R2 0.84): a%10 in {0} -> b%10 in {2}; a%10 in {1} -> b%10 in {1,2,3}; a%10 in {2} -> b%10 in {1,2}; a%10 in {3} -> b%10 in {1} | unexplained (best R2 0.27) | (reads) | same |
| L18 gate c664 | 21% / 0 | **units(a,b)** (R2 0.79): a%10 in {0} -> b%10 in {2,4,6,8}; a%10 in {2,4,6,8} -> b%10 in {0,2,4,6,8} | off (on 0) | (reads) | same |
| L18 up c24 | 56% / 48% | **units(a,b)** (R2 0.84): a%10 in {0,5} -> b%10 in {1,2,6,7}; a%10 in {2} -> b%10 in {0,2,3,4,6,7,8,9}; a%10 in {3,8} -> b%10 in {1,2,3,4,6,7,8,9}; a%10 in {4,9} -> b%10 in {1,2,3,6,7,8}; a%10 in {6} -> b%10 in {4,6}; a%10 in {7} -> b%10 in {0,2,3,4,5,7,8,9} | unexplained (best R2 0.49) | (reads) | same |
| L18 up c26 | 37% / 43% | **units(a,b)** (R2 0.90): a%10 in {0,3,4,8,9} -> b%10 in {0,3,4,5,8,9}; a%10 in {1} -> b%10 in {3}; a%10 in {5} -> b%10 in {0,3,4,8,9} | **units(a,b)** (R2 0.57): a%10 in {0,5} -> b%10 in {0,1,2,3,5,6,7,8}; a%10 in {2,7} -> b%10 in {0}; a%10 in {3,4,8,9} -> b%10 in {0,1,2,5,6,7} | (reads) | same |
| L18 up c32 | 28% / 42% | **units(a,b)** (R2 0.88): a%10 in {2} -> b%10 in {5,6,7}; a%10 in {3} -> b%10 in {3,4,5,6,7}; a%10 in {4} -> b%10 in {2,3,4,5,6,7}; a%10 in {5} -> b%10 in {1,2,3,4,5,6,7}; a%10 in {6} -> b%10 in {1,2,3,4,5,6} | unexplained (best R2 0.47) | (reads) | same |
| L18 up c62 | 27% / 17% | **units(a,b)** (R2 0.83): a%10 in {0} -> b%10 in {6,7}; a%10 in {1} -> b%10 in {5,6}; a%10 in {4} -> b%10 in {7}; a%10 in {5} -> b%10 in {6,7,8}; a%10 in {6,7,8} -> b%10 in {5,6,7,8,9}; a%10 in {9} -> b%10 in {5,6,7,8} | unexplained (best R2 0.41) | (reads) | same |
| L18 up c65 | 40% / 34% | **units(a,b)** (R2 0.87): a%10 in {0} -> b%10 in {5,6}; a%10 in {4,9} -> b%10 in {5,6,7}; a%10 in {5} -> b%10 in {0,3,4,5,6,7,8,9}; a%10 in {6} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {7} -> b%10 in {0,2,3,4,5,6,7,8,9}; a%10 in {8} -> b%10 in {5,6,7,8} | **units(a,b)** (R2 0.63): a%10 in {4} -> b%10 in {4,5}; a%10 in {5} -> b%10 in {0,1,2,3,4,5,6,7}; a%10 in {6} -> b%10 in {0,1,2,3,4,5,6,7,8}; a%10 in {7} -> b%10 in {1,2,3,4,5,6,7,8}; a%10 in {8} -> b%10 in {2,3,4,5,6}; a%10 in {9} -> b%10 in {3,4} | (reads) | same |
| L18 up c80 | 25% / 58% | **units(a,b)** (R2 0.81): a%10 in {0,1} -> b%10 in {0,3,4,5,8,9}; a%10 in {4} -> b%10 in {0,9}; a%10 in {5} -> b%10 in {0,4,5,8,9}; a%10 in {9} -> b%10 in {0,1,4,5,9} | unexplained (best R2 0.28) | (reads) | same |
| L18 up c88 | 32% / 17% | **units(a,b)** (R2 0.90): a%10 in {0,4} -> b%10 in {1,2,3,4}; a%10 in {1,2} -> b%10 in {1,2,3,4,5,6,7}; a%10 in {3} -> b%10 in {1,2,3,4,5,6}; a%10 in {5} -> b%10 in {1,2,3} | unexplained (best R2 0.37) | (reads) | same |
| L18 up c98 | 31% / 24% | **units(a,b)** (R2 0.83): a%10 in {0,8,9} -> b%10 in {4,5,6,7}; a%10 in {1} -> b%10 in {3,4,5,6,7}; a%10 in {2,3,6,7} -> b%10 in {4,5,6}; a%10 in {4,5} -> b%10 in {5,6} | **units(a,b)** (R2 0.55): a%10 in {0,8,9} -> b%10 in {3,4,5,6}; a%10 in {1} -> b%10 in {3,4,5,6,7}; a%10 in {2,7} -> b%10 in {4,5,6}; a%10 in {6} -> b%10 in {5} | (reads) | same |
| L18 up c120 | 26% / 14% | **units(a,b)** (R2 0.88): a%10 in {2} -> b%10 in {4,5,6,7}; a%10 in {3,4} -> b%10 in {2,3,4,5,6,7}; a%10 in {5} -> b%10 in {2,3,4,5,6}; a%10 in {6} -> b%10 in {3,4,5}; a%10 in {7} -> b%10 in {3} | unexplained (best R2 0.48) | (reads) | same |
| L18 up c123 | 5% / 2% | **units(a,b)** (R2 0.87): a%10 in {5} -> b%10 in {2,3,4}; a%10 in {6} -> b%10 in {2,3} | unexplained (best R2 0.25) | (reads) | same |
| L18 up c156 | 26% / 11% | **units(a,b)** (R2 0.93): a%10 in {1} -> b%10 in {1,2,3,7,8,9}; a%10 in {2} -> b%10 in {1,2,3,6,7,8}; a%10 in {3} -> b%10 in {1,2,6,7}; a%10 in {4,9} -> b%10 in {1}; a%10 in {6} -> b%10 in {2}; a%10 in {7} -> b%10 in {1,2,3,7}; a%10 in {8} -> b%10 in {1,2} | unexplained (best R2 0.37) | (reads) | same |
| L18 up c178 | 2% / 0 | **units(a,b)** (R2 0.69): a%10 in {5} -> b%10 in {5} | off (on 0) | (reads) | same |
| L18 up c213 | 9% / 5% | **units(a,b)** (R2 0.76): a%10 in {0,9} -> b%10 in {4,5,6}; a%10 in {8} -> b%10 in {5,6} | unexplained (best R2 0.35) | (reads) | same |
| L18 up c336 | 10% / 9% | **units(a,b)** (R2 0.86): a%10 in {2,7} -> b%10 in {9}; a%10 in {3,4,8,9} -> b%10 in {8,9} | unexplained (best R2 0.45) | (reads) | same |
| L18 up c346 | 13% / 6% | **units(a,b)** (R2 0.85): a%10 in {0} -> b%10 in {3,4,5,6}; a%10 in {1} -> b%10 in {3,4,5}; a%10 in {8} -> b%10 in {4,5}; a%10 in {9} -> b%10 in {3,4,5,6,9} | unexplained (best R2 0.49) | (reads) | same |
| L18 up c359 | 16% / 6% | **units(a,b)** (R2 0.89): a%10 in {1} -> b%10 in {2,7}; a%10 in {2,7} -> b%10 in {1,2,6,7}; a%10 in {3} -> b%10 in {1}; a%10 in {6} -> b%10 in {2,7,8}; a%10 in {8} -> b%10 in {1,6} | unexplained (best R2 0.31) | (reads) | same |
| L18 up c368 | 2% / 1% | **units(a,b)** (R2 0.53): a%10 in {7} -> b%10 in {2} | unexplained (best R2 0.18) | (reads) | same |
| L18 up c375 | 44% / 29% | **units(a,b)** (R2 0.84): a%10 in {0} -> b%10 in {2,3,7,8}; a%10 in {2,7} -> b%10 in {3,4,8,9}; a%10 in {3,8} -> b%10 in {0,2,3,4,5,7,8,9}; a%10 in {4,9} -> b%10 in {2,3,4,7,8,9}; a%10 in {5} -> b%10 in {2,3,4,7,8} | unexplained (best R2 0.45) | (reads) | same |
| L18 up c381 | 3% / 1% | **units(a,b)** (R2 0.54): a%10 in {4,5} -> b%10 in {4} | unexplained (best R2 0.15) | (reads) | same |
| L18 up c446 | 8% / 8% | **units(a,b)** (R2 0.82): a%10 in {0} -> b%10 in {7,8,9}; a%10 in {1,2} -> b%10 in {6,7,8} | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {1,2,3}; a%10 in {1} -> b%10 in {1,2,3,4}; a%10 in {2} -> b%10 in {2,3} | (reads) | same |
| L18 up c474 | 24% / 18% | **units(a,b)** (R2 0.85): a%10 in {3} -> b%10 in {4,6}; a%10 in {4,5} -> b%10 in {0,3,4,5,6,7,8,9}; a%10 in {6} -> b%10 in {0,3,4,5,6,8,9} | **units(a,b)** (R2 0.53): a%10 in {4} -> b%10 in {0,1,2,3,4,5,6}; a%10 in {5} -> b%10 in {0,1,2,4,5,6}; a%10 in {6} -> b%10 in {4,5,6} | (reads) | same |
| L18 up c619 | 16% / 6% | **units(a,b)** (R2 0.79): a%10 in {3,8} -> b%10 in {3,4,8,9}; a%10 in {4} -> b%10 in {3,8}; a%10 in {7} -> b%10 in {4}; a%10 in {9} -> b%10 in {2,3,4,7,8,9} | unexplained (best R2 0.33) | (reads) | same |
| L18 up c630 | 1% / 1% | **units(a,b)** (R2 0.62): a%10 in {5} -> b%10 in {3} | unexplained (best R2 0.19) | (reads) | same |
| L18 up c695 | 20% / 8% | **units(a,b)** (R2 0.75): a%10 in {0} -> b%10 in {2,7}; a%10 in {1,2,6,7} -> b%10 in {1,2,6,7}; a%10 in {5} -> b%10 in {2,6,7} | unexplained (best R2 0.25) | (reads) | same |
| L18 up c819 | 5% / 2% | **units(a,b)** (R2 0.81): a%10 in {3,4} -> b%10 in {3,4} | unexplained (best R2 0.22) | (reads) | same |
| L18 up c841 | 7% / 9% | **units(a,b)** (R2 0.58): a%10 in {0} -> b%10 in {6,7}; a%10 in {1} -> b%10 in {5,6,7}; a%10 in {2} -> b%10 in {5}; a%10 in {9} -> b%10 in {7} | unexplained (best R2 0.39) | (reads) | same |
| L18 up c870 | 24% / 17% | **units(a,b)** (R2 0.83): a%10 in {0,2} -> b%10 in {0,1,9}; a%10 in {1} -> b%10 in {0,1,8,9}; a%10 in {3,7,8} -> b%10 in {1}; a%10 in {4,5,6} -> b%10 in {0,1}; a%10 in {9} -> b%10 in {0,1,2,9} | **units(a,b)** (R2 0.62): a%10 in {0,2} -> b%10 in {0,1,9}; a%10 in {1} -> b%10 in {0,1,2,9}; a%10 in {4,5,8} -> b%10 in {9}; a%10 in {9} -> b%10 in {0,1,8,9} | (reads) | same |
| L18 up c886 | 5% / 2% | **units(a,b)** (R2 0.69): a%10 in {5} -> b%10 in {2,3,8}; a%10 in {6} -> b%10 in {2} | unexplained (best R2 0.23) | (reads) | same |

</details>

<details><summary>tens(a,b): 74 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 down c10 | 48% / 47% | **tens(a,b)** (R2 0.80): a//10 in {0} -> b//10 in {2,3,5,6,7,8}; a//10 in {1} -> b//10 in {6}; a//10 in {5} -> b//10 in {3,4,5,7,8,9,10}; a//10 in {6,10} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {8,9} -> b//10 in {1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.73): a//10 in {3} -> b//10 in {2,6}; a//10 in {4} -> b//10 in {3,6,7}; a//10 in {5} -> b//10 in {3,4,6,7}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,8,9}; a//10 in {7,8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {1,2,3,4,5,6,7,8,9} | a: mod100 +2%; b: mod100 +6%; res: mod100 +33% | b: mod100 +5%; res: mod100 +4%, mod50 +2% |
| L16 down c18 | 86% / 0 | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {5,6,7,8,9,10}; a//10 in {1} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {3,4,5,9} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {6,7,8,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | off (on 0) | - | same |
| L16 down c25 | 34% / 25% | **tens(a,b)** (R2 0.83): a//10 in {3} -> b//10 in {5,6,7}; a//10 in {4} -> b//10 in {4,5,6,7,8,9}; a//10 in {5,6} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {3,4,5,6,7,8}; a//10 in {8} -> b//10 in {4,5,6,7}; a//10 in {9} -> b//10 in {6} | **tens(a,b)** (R2 0.66): a//10 in {3} -> b//10 in {2}; a//10 in {4,5} -> b//10 in {1,2,3,4,5,6}; a//10 in {6} -> b//10 in {2,3,4,5,6}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {3,4,5} | a: mod100 +3%; b: mod100 +3%; res: mod100 +10% | b: mod100 +2%; res: mod100 +2% |
| L16 down c29 | 50% / 31% | **tens(a,b)** (R2 0.80): a//10 in {0,9,10} -> b//10 in {0,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {8}; a//10 in {5} -> b//10 in {5,6,7}; a//10 in {6} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.74): a//10 in {0} -> b//10 in {0,1,10}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {1}; a//10 in {5} -> b//10 in {3}; a//10 in {6} -> b//10 in {2,3,4,5}; a//10 in {7} -> b//10 in {1,2,3,4,5,6}; a//10 in {8} -> b//10 in {2,3,4,5,6,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | b: mod100 +2%; res: mod100 +19% | - |
| L16 down c43 | 52% / 11% | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {4,5,6,7,8}; a//10 in {1} -> b//10 in {3,4,5,6,7,8}; a//10 in {2} -> b//10 in {2,3,4,5,6,7,8}; a//10 in {3} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {1,2,3,4,5,7,8,9,10}; a//10 in {6} -> b//10 in {1,2,8,9,10}; a//10 in {7} -> b//10 in {1,8}; a//10 in {8} -> b//10 in {6}; a//10 in {9} -> b//10 in {5,6,7}; a//10 in {10} -> b//10 in {5,6} | unexplained (best R2 0.40) | b: mod100 +2%; res: mod100 +9% | - |
| L16 down c49 | 15% / 0 | **tens(a,b)** (R2 0.70): a//10 in {0,3} -> b//10 in {1,2}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2,4,5,6,7,8} -> b//10 in {1} | off (on 0) | - | same |
| L16 down c106 | 35% / 16% | **tens(a,b)** (R2 0.77): a//10 in {0} -> b//10 in {2,3,4}; a//10 in {1,4} -> b//10 in {1,2,3}; a//10 in {2} -> b//10 in {2,3}; a//10 in {5} -> b//10 in {1,2,3,4}; a//10 in {6,7} -> b//10 in {0,1,2,3,4,10}; a//10 in {8} -> b//10 in {0,1,2,3,10}; a//10 in {9} -> b//10 in {1,2} | **tens(a,b)** (R2 0.70): a//10 in {5} -> b//10 in {7,8}; a//10 in {6} -> b//10 in {5,6,7,8}; a//10 in {7} -> b//10 in {0,6,7,8,9}; a//10 in {8} -> b//10 in {0,7,8,9} | a: mod100 +2%; res: mod100 +9% | - |
| L16 down c130 | 25% / 12% | **tens(a,b)** (R2 0.82): a//10 in {0} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {1} -> b//10 in {3,4,5,6,7,8}; a//10 in {2} -> b//10 in {5,6}; a//10 in {8} -> b//10 in {5,6,7,8,9}; a//10 in {9,10} -> b//10 in {4,5,6,7,8,9} | **tens(a,b)** (R2 0.81): a//10 in {8} -> b//10 in {0,1,2,3,4}; a//10 in {9} -> b//10 in {0,1,2,3,4,5}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6} | res: mod100 +6% | - |
| L16 gate c25 | 38% / 27% | **tens(a,b)** (R2 0.77): a//10 in {0,1} -> b//10 in {0,1}; a//10 in {2} -> b//10 in {0,1,2}; a//10 in {3,8} -> b//10 in {5,6,7}; a//10 in {4} -> b//10 in {5,6,7,8}; a//10 in {5} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {6} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {3,4,5,6,7,8}; a//10 in {9} -> b//10 in {6}; a//10 in {10} -> b//10 in {5,6} | **tens(a,b)** (R2 0.67): a//10 in {3} -> b//10 in {2}; a//10 in {4} -> b//10 in {1,2,3,4,5,6}; a//10 in {5} -> b//10 in {1,2,3,4,6}; a//10 in {6} -> b//10 in {1,2,3,4,5,6,7}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {3,4,5} | (reads) | same |
| L16 gate c29 | 40% / 17% | **tens(a,b)** (R2 0.82): a//10 in {0} -> b//10 in {2,3,4,5,6,7,8}; a//10 in {1} -> b//10 in {3,4,5,6,7}; a//10 in {6} -> b//10 in {4,5,6,7}; a//10 in {7} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {8,9,10} -> b//10 in {2,3,4,5,6,7,8,9} | **tens(a,b)** (R2 0.75): a//10 in {6} -> b//10 in {3,4}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {2,3,4,5,6}; a//10 in {9,10} -> b//10 in {1,2,3,4,5,6,7} | (reads) | same |
| L16 gate c43 | 22% / 23% | **tens(a,b)** (R2 0.76): a//10 in {2} -> b//10 in {1}; a//10 in {3,9} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {1,2,3}; a//10 in {5,6} -> b//10 in {0,1,2,3,10}; a//10 in {7} -> b//10 in {0,1,2,10}; a//10 in {8} -> b//10 in {0,1,2} | **tens(a,b)** (R2 0.72): a//10 in {3} -> b//10 in {7}; a//10 in {4} -> b//10 in {6,7,8}; a//10 in {5,6,7,8} -> b//10 in {6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {7,8,9} | (reads) | same |
| L16 gate c493 | 13% / 9% | **tens(a,b)** (R2 0.64): a//10 in {4,5} -> b//10 in {4,5,6,7,8}; a//10 in {6} -> b//10 in {6} | **tens(a,b)** (R2 0.52): a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3,4}; a//10 in {5} -> b//10 in {2,3,4}; a//10 in {10} -> b//10 in {4} | (reads) | same |
| L16 gate c872 | 8% / 10% | **tens(a,b)** (R2 0.62): a//10 in {1} -> b//10 in {5,6,7}; a//10 in {9} -> b//10 in {5,6,7,8}; a//10 in {10} -> b//10 in {4,5,6,7,8} | **tens(a,b)** (R2 0.71): a//10 in {9} -> b//10 in {0,1,2,3,4,5,8}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8,9} | (reads) | same |
| L16 up c29 | 63% / 31% | **tens(a,b)** (R2 0.67): a//10 in {0} -> b//10 in {0,3,4,5,6}; a//10 in {1,2} -> b//10 in {3,4,5,6}; a//10 in {3} -> b//10 in {3,4,5,6,7}; a//10 in {4} -> b//10 in {2,3,4,5,6,7}; a//10 in {5,10} -> b//10 in {2,3,4,5,6,7,8}; a//10 in {6} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {1,2,3,4,5,6,7,8,9}; a//10 in {8,9} -> b//10 in {1,2,3,4,5,6,7,8} | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {0,1,2,10}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {3,4,5,6}; a//10 in {7} -> b//10 in {3,4,5,6,7,10}; a//10 in {8} -> b//10 in {0,3,4,5,6,7,10}; a//10 in {9} -> b//10 in {0,1,3,4,5,6,7,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,10} | (reads) | same |
| L16 up c43 | 50% / 12% | **tens(a,b)** (R2 0.75): a//10 in {0,9} -> b//10 in {2,3,4,5,6,7,8,9}; a//10 in {1,2} -> b//10 in {2,3,4,5,6,7,8}; a//10 in {3,10} -> b//10 in {2,3,4,5,6,7}; a//10 in {4} -> b//10 in {3,4,5,6}; a//10 in {5,7} -> b//10 in {3,4,5}; a//10 in {8} -> b//10 in {3,4,5,6,8} | **tens(a,b)** (R2 0.55): a//10 in {8} -> b//10 in {4,5,6}; a//10 in {9} -> b//10 in {3,4,5,6,7}; a//10 in {10} -> b//10 in {2,3,4,5,6,7} | (reads) | same |
| L16 up c420 | 32% / 1% | **tens(a,b)** (R2 0.73): a//10 in {0} -> b//10 in {2}; a//10 in {1} -> b//10 in {2,3}; a//10 in {2} -> b//10 in {1,3}; a//10 in {3} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {1,2,3}; a//10 in {5,6,9,10} -> b//10 in {1,2,3,4}; a//10 in {7,8} -> b//10 in {0,1,2,3,4} | unexplained (best R2 0.09) | (reads) | same |
| L16 up c757 | 10% / 12% | **tens(a,b)** (R2 0.64): a//10 in {5} -> b//10 in {0}; a//10 in {6} -> b//10 in {0,1,2,6}; a//10 in {7,8} -> b//10 in {0,1} | **tens(a,b)** (R2 0.56): a//10 in {6} -> b//10 in {0,3,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,7,8,10} | (reads) | same |
| L17 down c11 | 46% / 27% | **tens(a,b)** (R2 0.79): a//10 in {2} -> b//10 in {6,7}; a//10 in {3} -> b//10 in {6,7,9,10}; a//10 in {4,6} -> b//10 in {0,1,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {0,5,6,7,8,9,10}; a//10 in {9,10} -> b//10 in {4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.61): a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {10}; a//10 in {5} -> b//10 in {0,1,2,8,9,10}; a//10 in {6} -> b//10 in {0,1,2,9,10}; a//10 in {7} -> b//10 in {0,1,2,3,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,10}; a//10 in {9} -> b//10 in {1,3}; a//10 in {10} -> b//10 in {1,2,3,4,5,10} | a: mod100 +9%; b: mod100 +6%; res: mod100 +22%, mod50 +5% | - |
| L17 down c15 | 44% / 9% | **tens(a,b)** (R2 0.62): a//10 in {3} -> b//10 in {6}; a//10 in {4} -> b//10 in {1,2,3,5,6,7}; a//10 in {5} -> b//10 in {2,3,6}; a//10 in {6,9} -> b//10 in {1,2,3,4,5,6,7}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,7,10}; a//10 in {10} -> b//10 in {1,2,3,4,5,6,7,8} | **tens(a,b)** (R2 0.67): a//10 in {7} -> b//10 in {0,1,2,3,4,5}; a//10 in {8} -> b//10 in {3}; a//10 in {10} -> b//10 in {0,1,2,3,4} | a: mod100 +3% | - |
| L17 down c23 | 65% / 22% | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {1,4,8,9}; a//10 in {1} -> b//10 in {1,3,4,5,6,8,9,10}; a//10 in {2,3,6} -> b//10 in {1,2,3,4,5,6,8,9,10}; a//10 in {4} -> b//10 in {1,3,8,9}; a//10 in {5} -> b//10 in {1,3,4,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,8,9,10}; a//10 in {9} -> b//10 in {1,3,4,6,8}; a//10 in {10} -> b//10 in {3,4,8,9} | **tens(a,b)** (R2 0.58): a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {1,3}; a//10 in {6} -> b//10 in {0,5}; a//10 in {7} -> b//10 in {0,1,3,4,5,6}; a//10 in {8} -> b//10 in {3,4,5,6,8,9} | b: mod50 +13%; res: mod50 +48% | b: mod50 +2% |
| L17 down c40 | 47% / 15% | **tens(a,b)** (R2 0.78): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7}; a//10 in {1,2} -> b//10 in {2,3,4,5,6,7}; a//10 in {3,4} -> b//10 in {1,2,3,4,5,6}; a//10 in {5} -> b//10 in {1,2,3,4,5}; a//10 in {6} -> b//10 in {1,2,3,4}; a//10 in {8} -> b//10 in {2}; a//10 in {9,10} -> b//10 in {0,1,2,3,4} | unexplained (best R2 0.44) | res: mod100 +10% | - |
| L17 down c107 | 24% / 15% | **tens(a,b)** (R2 0.81): a//10 in {0,1} -> b//10 in {2,3,4,5}; a//10 in {2} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {4,5}; a//10 in {7} -> b//10 in {4,5,6}; a//10 in {8,9} -> b//10 in {3,4,5,6}; a//10 in {10} -> b//10 in {2,3,4,5,6} | **tens(a,b)** (R2 0.76): a//10 in {6} -> b//10 in {4,5}; a//10 in {7,8} -> b//10 in {3,4,5,6}; a//10 in {9,10} -> b//10 in {3,4,5,6,7} | res: mod100 +2% | - |
| L17 down c119 | 28% / 9% | **tens(a,b)** (R2 0.60): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1,2} -> b//10 in {0,1,2,3,10}; a//10 in {3} -> b//10 in {0,3}; a//10 in {4} -> b//10 in {0,4}; a//10 in {5,6,7,8} -> b//10 in {0}; a//10 in {9} -> b//10 in {0,9}; a//10 in {10} -> b//10 in {0,10} | unexplained (best R2 0.41) | a: mod100 +2%, mod25 +2% | res: mod25 +3%, mod20 +3% |
| L17 down c122 | 19% / 0 | **tens(a,b)** (R2 0.72): a//10 in {2,3,4,7,8} -> b//10 in {6,7}; a//10 in {5,6} -> b//10 in {5,6,7} | off (on 0) | - | same |
| L17 down c235 | 17% / 0 | **tens(a,b)** (R2 0.80): a//10 in {0,1,2} -> b//10 in {6,7,8,9,10}; a//10 in {3} -> b//10 in {6,7}; a//10 in {9} -> b//10 in {9,10}; a//10 in {10} -> b//10 in {9} | off (on 0) | res: mod100 +2% | - |
| L17 gate c23 | 7% / 5% | **tens(a,b)** (R2 0.57): a//10 in {2,3,8} -> b//10 in {1}; a//10 in {7} -> b//10 in {1,6,9,10} | unexplained (best R2 0.36) | (reads) | same |
| L17 gate c40 | 21% / 2% | **tens(a,b)** (R2 0.67): a//10 in {0,3,4,5,9,10} -> b//10 in {2,3,4}; a//10 in {1,2} -> b//10 in {3,4}; a//10 in {6,8} -> b//10 in {3} | unexplained (best R2 0.43) | (reads) | same |
| L17 up c11 | 57% / 31% | **tens(a,b)** (R2 0.78): a//10 in {0} -> b//10 in {8}; a//10 in {2} -> b//10 in {10}; a//10 in {3} -> b//10 in {1,9,10}; a//10 in {4} -> b//10 in {0,1,2,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,6,7,8,9,10}; a//10 in {6} -> b//10 in {0,1,2,4,5,6,7,8,9,10}; a//10 in {7,8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {9,10} -> b//10 in {1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.69): a//10 in {2,3} -> b//10 in {2}; a//10 in {4} -> b//10 in {0,10}; a//10 in {5} -> b//10 in {0,1,2,8,9,10}; a//10 in {6} -> b//10 in {0,1,2,3,9,10}; a//10 in {7,8} -> b//10 in {0,1,2,3,4,10}; a//10 in {9} -> b//10 in {0,1,2,3,4}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,10} | (reads) | same |
| L17 up c33 | 63% / 24% | **tens(a,b)** (R2 0.69): a//10 in {1,5} -> b//10 in {6,7}; a//10 in {2,3,7,8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {5,6,7}; a//10 in {9} -> b//10 in {1,2,3,4,5,6,7,8,10}; a//10 in {10} -> b//10 in {1,2,5,6,7} | **tens(a,b)** (R2 0.58): a//10 in {2,3} -> b//10 in {1,2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {7} -> b//10 in {3,4,5,6,7,8}; a//10 in {8} -> b//10 in {0,3,4,5,6,7,8,9}; a//10 in {9,10} -> b//10 in {8} | (reads) | same |
| L17 up c40 | 41% / 17% | **tens(a,b)** (R2 0.74): a//10 in {0} -> b//10 in {0,1,2,3,4,5,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,3,4,9,10}; a//10 in {2} -> b//10 in {0,1,2,3,9,10}; a//10 in {3} -> b//10 in {0,1,2,10}; a//10 in {4} -> b//10 in {0}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,9,10} | **tens(a,b)** (R2 0.67): a//10 in {0} -> b//10 in {0}; a//10 in {7} -> b//10 in {6}; a//10 in {8,9} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,4,5,6,7,8,9,10} | (reads) | same |
| L17 up c193 | 32% / 11% | **tens(a,b)** (R2 0.63): a//10 in {1} -> b//10 in {3,4,5,8}; a//10 in {2} -> b//10 in {3,4,5,8,9}; a//10 in {3} -> b//10 in {3,4,8}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6,8} -> b//10 in {2,3,4,5}; a//10 in {7} -> b//10 in {2,3,4,5,9}; a//10 in {9} -> b//10 in {2,3,4}; a//10 in {10} -> b//10 in {3,4,5} | **tens(a,b)** (R2 0.62): a//10 in {2} -> b//10 in {0,1}; a//10 in {6} -> b//10 in {5}; a//10 in {7} -> b//10 in {0,5,6}; a//10 in {8,10} -> b//10 in {5,6,7} | (reads) | same |
| L18 down c21 | 70% / 36% | **tens(a,b)** (R2 0.57): a//10 in {0} -> b//10 in {1,2,3,4,6,7,8,9}; a//10 in {1,6} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,1,2,5,6,7,10}; a//10 in {3} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {4} -> b//10 in {2,3,4,8,9,10}; a//10 in {5} -> b//10 in {1,2,3,4,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,4,5,6,7,9,10}; a//10 in {8} -> b//10 in {0,1,4,5,9,10}; a//10 in {9} -> b//10 in {3,4,8,9,10}; a//10 in {10} -> b//10 in {2,3,4,7,8,9} | **tens(a,b)** (R2 0.57): a//10 in {0} -> b//10 in {1}; a//10 in {1} -> b//10 in {1,2,6,7}; a//10 in {2} -> b//10 in {2}; a//10 in {3} -> b//10 in {6}; a//10 in {4,9} -> b//10 in {1,6,7}; a//10 in {5} -> b//10 in {1,2,3,6,7}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,8,10}; a//10 in {7} -> b//10 in {3,4,7,8,10}; a//10 in {8} -> b//10 in {5}; a//10 in {10} -> b//10 in {0,1,2,5,6,7} | res: mod50 +18% | res: mod50 +5% |
| L18 down c25 | 78% / 32% | **tens(a,b)** (R2 0.52): a//10 in {0} -> b//10 in {3,4,5,7,8,9,10}; a//10 in {1} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {2} -> b//10 in {1,2,3,4,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,2,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,4,5,6,8,9,10}; a//10 in {5} -> b//10 in {0,3,4,5,7,8,9,10}; a//10 in {6} -> b//10 in {0,2,3,4,7,8,9,10}; a//10 in {7,8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {0,1,3,4,5,6,8,9,10}; a//10 in {10} -> b//10 in {0,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.54): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {0,1}; a//10 in {2} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {0,5,10}; a//10 in {5} -> b//10 in {0,1,5,6,9,10}; a//10 in {6} -> b//10 in {0,1,2,5,6,7,9,10}; a//10 in {7} -> b//10 in {1,2,3,6,7}; a//10 in {8} -> b//10 in {4}; a//10 in {9} -> b//10 in {0,4,5,9}; a//10 in {10} -> b//10 in {0,1,4,5,6,9,10} | a: mod50 +3%; b: mod50 +2%; res: mod50 +14% | res: mod50 +5% |
| L18 down c27 | 33% / 23% | **tens(a,b)** (R2 0.69): a//10 in {2} -> b//10 in {4,8,9,10}; a//10 in {3} -> b//10 in {3,4,7,8,9,10}; a//10 in {4} -> b//10 in {2,3,7,8,9}; a//10 in {5} -> b//10 in {8}; a//10 in {7} -> b//10 in {3,4,5,8,9,10}; a//10 in {8} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {9} -> b//10 in {2,3,4,7,8}; a//10 in {10} -> b//10 in {2,3,7,8} | **tens(a,b)** (R2 0.62): a//10 in {2} -> b//10 in {0,1,10}; a//10 in {3} -> b//10 in {0,1,2,6,7}; a//10 in {4} -> b//10 in {1,2,7}; a//10 in {7} -> b//10 in {0,5,6,10}; a//10 in {8} -> b//10 in {0,1,2,5,6,7,10}; a//10 in {9,10} -> b//10 in {1,2,6,7} | a: mod50 +3%; b: mod50 +2%; res: mod50 +9% | res: mod50 +4% |
| L18 down c30 | 26% / 9% | **tens(a,b)** (R2 0.51): a//10 in {2,4} -> b//10 in {5}; a//10 in {3} -> b//10 in {0,5,10}; a//10 in {6} -> b//10 in {0,4,5}; a//10 in {7,8} -> b//10 in {0,3,4,5,6,10}; a//10 in {9} -> b//10 in {3,4,5}; a//10 in {10} -> b//10 in {0,3,4,5} | **tens(a,b)** (R2 0.55): a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {4,5} | - | same |
| L18 down c36 | 58% / 23% | **tens(a,b)** (R2 0.54): a//10 in {0} -> b//10 in {4,5,6,10}; a//10 in {1} -> b//10 in {4,5,10}; a//10 in {2,3} -> b//10 in {1,2,3,6,7}; a//10 in {4,9} -> b//10 in {0,1,2,5,6,7,10}; a//10 in {5} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {6} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {7,8} -> b//10 in {1,2,3,5,6,7,8}; a//10 in {10} -> b//10 in {0,1,2,4,5,6,7,10} | **tens(a,b)** (R2 0.55): a//10 in {3} -> b//10 in {2,3,8}; a//10 in {4} -> b//10 in {3,8}; a//10 in {5} -> b//10 in {0,4,5,8,9,10}; a//10 in {6} -> b//10 in {0,9,10}; a//10 in {8} -> b//10 in {3,4,7,8}; a//10 in {9} -> b//10 in {3,4,7,8,9}; a//10 in {10} -> b//10 in {0,3,4,5,7,8,9,10} | b: mod50 +2%, mod25 +3%; res: mod50 +8% | res: mod50 +5% |
| L18 down c54 | 2% / 0 | **tens(a,b)** (R2 0.71): a//10 in {0,1,2,3,4,5,6,7,8,9} -> b//10 in {10}; a//10 in {10} -> b//10 in {5,6,7,10} | off (on 0) | - | same |
| L18 down c61 | 33% / 13% | **tens(a,b)** (R2 0.65): a//10 in {0} -> b//10 in {0,1,4,5,6,7,10}; a//10 in {1,6} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {2} -> b//10 in {0,4,5,10}; a//10 in {4} -> b//10 in {0,1,6}; a//10 in {5,10} -> b//10 in {0,1,5,6,10}; a//10 in {7} -> b//10 in {0,9,10}; a//10 in {9} -> b//10 in {1,6} | **tens(a,b)** (R2 0.61): a//10 in {5} -> b//10 in {3,4,5,8,9,10}; a//10 in {6} -> b//10 in {0,4,5,9,10}; a//10 in {10} -> b//10 in {3,4,8,9} | res: mod50 +6% | - |
| L18 down c66 | 32% / 37% | **tens(a,b)** (R2 0.77): a//10 in {0} -> b//10 in {0,1,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,8,9,10}; a//10 in {2} -> b//10 in {9,10}; a//10 in {3,4,5,6} -> b//10 in {10}; a//10 in {7} -> b//10 in {0,9,10}; a//10 in {8} -> b//10 in {0,1,2,7,8,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.71): a//10 in {0,1} -> b//10 in {0,1,2,7,8,9,10}; a//10 in {2} -> b//10 in {8,9,10}; a//10 in {3,4,5} -> b//10 in {9}; a//10 in {6,7} -> b//10 in {8,9}; a//10 in {8} -> b//10 in {0,1,2,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,8,9,10} | - | same |
| L18 down c79 | 7% / 0 | **tens(a,b)** (R2 0.84): a//10 in {2,3,4,5,6,7,8,9,10} -> b//10 in {0} | off (on 0) | - | same |
| L18 down c91 | 27% / 3% | **tens(a,b)** (R2 0.77): a//10 in {0} -> b//10 in {3,4}; a//10 in {1} -> b//10 in {2,3,4,5}; a//10 in {2} -> b//10 in {1,2,3,4,5,6}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6}; a//10 in {4} -> b//10 in {1,2,3,4,5}; a//10 in {5} -> b//10 in {2,3,4} | unexplained (best R2 0.22) | res: mod100 +2% | - |
| L18 down c111 | 13% / 36% | **tens(a,b)** (R2 0.58): a//10 in {0} -> b//10 in {3}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {0,1,2,3}; a//10 in {4} -> b//10 in {0,1,2}; a//10 in {5} -> b//10 in {0} | **tens(a,b)** (R2 0.70): a//10 in {0} -> b//10 in {2,3,4,5,6}; a//10 in {1} -> b//10 in {3,4,5}; a//10 in {2} -> b//10 in {0,3,4,5}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,7,9,10}; a//10 in {4} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,5,6}; a//10 in {6} -> b//10 in {2} | - | same |
| L18 down c118 | 6% / 48% | **tens(a,b)** (R2 0.56): a//10 in {9} -> b//10 in {0,1,2,7,8,9}; a//10 in {10} -> b//10 in {1,2} | **tens(a,b)** (R2 0.59): a//10 in {0} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {5,6,7,8,9}; a//10 in {2} -> b//10 in {7,8,9,10}; a//10 in {3} -> b//10 in {8,9}; a//10 in {4} -> b//10 in {0,1,2,8,9}; a//10 in {5} -> b//10 in {0,1,2,9}; a//10 in {6} -> b//10 in {2}; a//10 in {7} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1,2,3,4,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,7,10} | - | a: mod50 +2% |
| L18 down c132 | 34% / 0 | **tens(a,b)** (R2 0.89): a//10 in {3} -> b//10 in {9}; a//10 in {4} -> b//10 in {6,7,8,9,10}; a//10 in {5,6} -> b//10 in {5,6,7,8,9,10}; a//10 in {7,8,9,10} -> b//10 in {4,5,6,7,8,9,10} | off (on 0) | a: mod100 +2%; b: mod100 +2%; res: mod100 +3% | - |
| L18 down c147 | 16% / 29% | **tens(a,b)** (R2 0.80): a//10 in {0} -> b//10 in {0,1,2,3,10}; a//10 in {1} -> b//10 in {0,1,2,3,9,10}; a//10 in {2} -> b//10 in {0,1,2,9,10}; a//10 in {3} -> b//10 in {0,1}; a//10 in {10} -> b//10 in {1,10} | **tens(a,b)** (R2 0.82): a//10 in {0,4} -> b//10 in {0,1,2,3,4}; a//10 in {1} -> b//10 in {0,1,2,3,4,8,9,10}; a//10 in {2} -> b//10 in {0,1,2,3,4,9,10}; a//10 in {3} -> b//10 in {0,1,2,3,4,10}; a//10 in {10} -> b//10 in {7,8,9,10} | - | same |
| L18 down c196 | 9% / 0 | **tens(a,b)** (R2 0.71): a//10 in {0,1} -> b//10 in {7,8}; a//10 in {2} -> b//10 in {6,7,8}; a//10 in {3} -> b//10 in {6,7} | off (on 0) | - | same |
| L18 down c625 | 15% / 1% | **tens(a,b)** (R2 0.60): a//10 in {0,1,2} -> b//10 in {10}; a//10 in {3,4,5,6,7} -> b//10 in {9,10}; a//10 in {8} -> b//10 in {8,9,10}; a//10 in {9,10} -> b//10 in {4,5,6,7,8,9,10} | unexplained (best R2 0.49) | - | same |
| L18 gate c23 | 57% / 49% | **tens(a,b)** (R2 0.76): a//10 in {0,1} -> b//10 in {3,4,5,6}; a//10 in {2} -> b//10 in {4,5}; a//10 in {3} -> b//10 in {0,1,4,5,6,10}; a//10 in {4,5} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {0,1,3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {4,5,6,7}; a//10 in {9,10} -> b//10 in {3,4,5,6,7} | **tens(a,b)** (R2 0.70): a//10 in {0,1,2} -> b//10 in {4,5}; a//10 in {3} -> b//10 in {0,3,4,5,6,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,3,4,5,6,8,9,10}; a//10 in {6} -> b//10 in {0,1,2,4,5,9,10}; a//10 in {7} -> b//10 in {1,2,3,4,5}; a//10 in {8} -> b//10 in {3,4,5}; a//10 in {9,10} -> b//10 in {3,4,5,6} | (reads) | same |
| L18 gate c25 | 55% / 19% | **tens(a,b)** (R2 0.65): a//10 in {0} -> b//10 in {2,3,4,7,8,9}; a//10 in {1,6} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2,7} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {8}; a//10 in {4} -> b//10 in {9}; a//10 in {5} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {9} -> b//10 in {10}; a//10 in {10} -> b//10 in {3,4,7,8,9,10} | **tens(a,b)** (R2 0.55): a//10 in {1} -> b//10 in {0,1}; a//10 in {2} -> b//10 in {1,2}; a//10 in {5} -> b//10 in {1,6}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,10}; a//10 in {7} -> b//10 in {5,6,7}; a//10 in {10} -> b//10 in {1,5,6,7} | (reads) | same |
| L18 gate c27 | 44% / 13% | **tens(a,b)** (R2 0.69): a//10 in {0,1} -> b//10 in {8}; a//10 in {2} -> b//10 in {3,4,7,8,9,10}; a//10 in {3} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {4,9} -> b//10 in {2,3,4,7,8,9}; a//10 in {5,6} -> b//10 in {3,8,9}; a//10 in {7} -> b//10 in {3,4,5,7,8,9,10}; a//10 in {8} -> b//10 in {2,3,4,5,7,8,9,10}; a//10 in {10} -> b//10 in {2,3,4,7,8} | **tens(a,b)** (R2 0.60): a//10 in {2,3} -> b//10 in {1}; a//10 in {7} -> b//10 in {6}; a//10 in {8} -> b//10 in {0,1,2,5,6,7}; a//10 in {9} -> b//10 in {6,7} | (reads) | same |
| L18 gate c36 | 45% / 25% | **tens(a,b)** (R2 0.66): a//10 in {0,6,7} -> b//10 in {1,2,6,7}; a//10 in {1} -> b//10 in {6,7}; a//10 in {2} -> b//10 in {1,6,7}; a//10 in {3} -> b//10 in {1,2,6,7,8}; a//10 in {4,9,10} -> b//10 in {1,2,5,6,7,8,10}; a//10 in {5} -> b//10 in {1,2,5,6,7}; a//10 in {8} -> b//10 in {1,2,3,5,6,7,8} | **tens(a,b)** (R2 0.59): a//10 in {3,4} -> b//10 in {2,3,7,8}; a//10 in {5} -> b//10 in {3,4,7,8}; a//10 in {6,7} -> b//10 in {8}; a//10 in {8} -> b//10 in {2,3,7,8,9}; a//10 in {9} -> b//10 in {3,7,8,9,10}; a//10 in {10} -> b//10 in {2,3,4,5,6,7,8,9,10} | (reads) | same |
| L18 gate c61 | 51% / 26% | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1,3} -> b//10 in {0,1,5,6,9,10}; a//10 in {2} -> b//10 in {0,1,10}; a//10 in {4,5} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {6,8} -> b//10 in {0,1,5,6,10}; a//10 in {7} -> b//10 in {0,10}; a//10 in {9,10} -> b//10 in {0,1,3,4,5,6,9,10} | **tens(a,b)** (R2 0.60): a//10 in {0} -> b//10 in {0,9,10}; a//10 in {3} -> b//10 in {0,9}; a//10 in {4,5} -> b//10 in {0,3,4,5,9,10}; a//10 in {6} -> b//10 in {4,9,10}; a//10 in {8} -> b//10 in {0,4,5,9}; a//10 in {9} -> b//10 in {0,4,5,8,9,10}; a//10 in {10} -> b//10 in {0,3,4,5,6,8,9,10} | (reads) | same |
| L18 gate c66 | 31% / 35% | **tens(a,b)** (R2 0.80): a//10 in {0} -> b//10 in {0,1,8,9,10}; a//10 in {1} -> b//10 in {0,1,7,8,9,10}; a//10 in {2} -> b//10 in {8,9,10}; a//10 in {3,4,5,6,7} -> b//10 in {9,10}; a//10 in {8} -> b//10 in {0,1,2,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.75): a//10 in {0} -> b//10 in {0,1,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,7,8,9,10}; a//10 in {2,5} -> b//10 in {9,10}; a//10 in {3,4} -> b//10 in {9}; a//10 in {6,7} -> b//10 in {8,9}; a//10 in {8} -> b//10 in {0,1,2,8,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | (reads) | same |
| L18 gate c91 | 30% / 13% | **tens(a,b)** (R2 0.70): a//10 in {0,10} -> b//10 in {3}; a//10 in {2} -> b//10 in {0,3,4,9}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,3,4,8,9}; a//10 in {7,8,9} -> b//10 in {3,4} | **tens(a,b)** (R2 0.55): a//10 in {2} -> b//10 in {0}; a//10 in {3} -> b//10 in {0,1,2,3,6,10}; a//10 in {4} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {5,6}; a//10 in {9,10} -> b//10 in {6} | (reads) | same |
| L18 gate c111 | 17% / 11% | **tens(a,b)** (R2 0.74): a//10 in {0} -> b//10 in {3,4}; a//10 in {2,3,4} -> b//10 in {0,1,2,3,4} | **tens(a,b)** (R2 0.50): a//10 in {0} -> b//10 in {4}; a//10 in {3} -> b//10 in {0,1,3,4,10}; a//10 in {4} -> b//10 in {0,1,2,4,10} | (reads) | same |
| L18 gate c147 | 19% / 27% | **tens(a,b)** (R2 0.82): a//10 in {0,3} -> b//10 in {0,1,2,3}; a//10 in {1} -> b//10 in {0,1,2,3,10}; a//10 in {2} -> b//10 in {0,1,2,3,9,10}; a//10 in {4} -> b//10 in {0} | **tens(a,b)** (R2 0.84): a//10 in {0,4} -> b//10 in {0,1,2,3,4}; a//10 in {1,2} -> b//10 in {0,1,2,3,4,9,10}; a//10 in {3} -> b//10 in {0,1,2,3,4,10} | (reads) | same |
| L18 gate c543 | 16% / 6% | **tens(a,b)** (R2 0.65): a//10 in {0,1,2} -> b//10 in {10}; a//10 in {3,6,7} -> b//10 in {9,10}; a//10 in {4,5} -> b//10 in {8,9,10}; a//10 in {8} -> b//10 in {4,9,10}; a//10 in {9,10} -> b//10 in {4,8,9,10} | unexplained (best R2 0.46) | (reads) | same |
| L18 gate c677 | 13% / 7% | **tens(a,b)** (R2 0.60): a//10 in {3} -> b//10 in {1,2,3,7,8}; a//10 in {4} -> b//10 in {1,2,3,6,7}; a//10 in {8} -> b//10 in {2,7}; a//10 in {9} -> b//10 in {2} | **tens(a,b)** (R2 0.54): a//10 in {3} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {3}; a//10 in {8} -> b//10 in {6,7}; a//10 in {9} -> b//10 in {7} | (reads) | same |
| L18 up c21 | 50% / 34% | **tens(a,b)** (R2 0.71): a//10 in {0,1,5,6,10} -> b//10 in {1,2,3,6,7,8}; a//10 in {2} -> b//10 in {1,2,3,6,7}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4,9} -> b//10 in {2,3,7,8}; a//10 in {7} -> b//10 in {1,2,6,7}; a//10 in {8} -> b//10 in {2} | **tens(a,b)** (R2 0.65): a//10 in {1} -> b//10 in {1,2,3}; a//10 in {2} -> b//10 in {2}; a//10 in {3} -> b//10 in {6,7}; a//10 in {4,9} -> b//10 in {1,2,3,6,7}; a//10 in {5,6,10} -> b//10 in {1,2,3,6,7,8}; a//10 in {7} -> b//10 in {3,7}; a//10 in {8} -> b//10 in {7} | (reads) | same |
| L18 up c23 | 37% / 27% | **tens(a,b)** (R2 0.81): a//10 in {2} -> b//10 in {5,6,7}; a//10 in {3} -> b//10 in {4,5,6,7,8,9}; a//10 in {4} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {5} -> b//10 in {3,4,5,6,7,8}; a//10 in {6} -> b//10 in {2,3,4,5,6}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {3,4,5}; a//10 in {9,10} -> b//10 in {3,4} | **tens(a,b)** (R2 0.73): a//10 in {3} -> b//10 in {2,3,4,5,6}; a//10 in {4,5,6} -> b//10 in {3,4,5,6}; a//10 in {7} -> b//10 in {4,5,6}; a//10 in {8} -> b//10 in {4,5,6,7} | (reads) | same |
| L18 up c25 | 52% / 17% | **tens(a,b)** (R2 0.77): a//10 in {0,1} -> b//10 in {2,3,4,7,8,9}; a//10 in {2} -> b//10 in {2,3,7,8,9}; a//10 in {3} -> b//10 in {8,9,10}; a//10 in {4,5,6,9,10} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {7} -> b//10 in {2,3,7,8,9,10}; a//10 in {8} -> b//10 in {3,4,8,9,10} | **tens(a,b)** (R2 0.51): a//10 in {1,2,4} -> b//10 in {1}; a//10 in {5} -> b//10 in {1,2,6,7}; a//10 in {6,9} -> b//10 in {1,6,7}; a//10 in {10} -> b//10 in {0,1,2,5,6,7} | (reads) | same |
| L18 up c36 | 35% / 26% | **tens(a,b)** (R2 0.65): a//10 in {0,1,5,6} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {2} -> b//10 in {2,3,7,8}; a//10 in {7} -> b//10 in {2,3,4,8}; a//10 in {10} -> b//10 in {3,4,8} | **tens(a,b)** (R2 0.65): a//10 in {0} -> b//10 in {0,1}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {1,2}; a//10 in {5} -> b//10 in {0,1,2,5,6,7,10}; a//10 in {6} -> b//10 in {0,1,2,3,5,6,7,10}; a//10 in {7} -> b//10 in {1,6,7}; a//10 in {10} -> b//10 in {0,1,5,6,7} | (reads) | same |
| L18 up c61 | 49% / 36% | **tens(a,b)** (R2 0.69): a//10 in {0,5} -> b//10 in {3,8}; a//10 in {2} -> b//10 in {3,4,8,9,10}; a//10 in {3,8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {4,9} -> b//10 in {0,2,3,4,5,7,8,9,10}; a//10 in {7} -> b//10 in {3,4,9,10}; a//10 in {10} -> b//10 in {2,3,4,7,8,9} | **tens(a,b)** (R2 0.67): a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {0,1,2,3,6,7,10}; a//10 in {4} -> b//10 in {0,1,2,6,7,8,10}; a//10 in {7} -> b//10 in {0,5,6}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,8,10}; a//10 in {10} -> b//10 in {0,1,2,5,6,7} | (reads) | same |
| L18 up c91 | 31% / 9% | **tens(a,b)** (R2 0.82): a//10 in {0} -> b//10 in {3,4}; a//10 in {1} -> b//10 in {2,3,4}; a//10 in {2,3,4} -> b//10 in {0,1,2,3,4,5,6}; a//10 in {5} -> b//10 in {2,3,4,5}; a//10 in {6} -> b//10 in {4} | **tens(a,b)** (R2 0.52): a//10 in {3} -> b//10 in {2,3,4}; a//10 in {4} -> b//10 in {3,4}; a//10 in {5} -> b//10 in {4,5} | (reads) | same |
| L18 up c111 | 1% / 10% | **tens(a,b)** (R2 0.51): a//10 in {9} -> b//10 in {9,10}; a//10 in {10} -> b//10 in {10} | **tens(a,b)** (R2 0.67): a//10 in {7,8,9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {5,6,7,8,9,10} | (reads) | same |
| L18 up c130 | 17% / 1% | **tens(a,b)** (R2 0.73): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1,2,3,4,5,6,7,8,9} -> b//10 in {0}; a//10 in {10} -> b//10 in {0,1} | **tens(a,b)** (R2 0.56): a//10 in {0} -> b//10 in {0} | (reads) | same |
| L18 up c390 | 13% / 8% | **tens(a,b)** (R2 0.57): a//10 in {1} -> b//10 in {2}; a//10 in {2} -> b//10 in {1,2,3,7,8}; a//10 in {3} -> b//10 in {2,7}; a//10 in {7} -> b//10 in {2,7,8}; a//10 in {8} -> b//10 in {7} | **tens(a,b)** (R2 0.51): a//10 in {1} -> b//10 in {2}; a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {7,8} -> b//10 in {7} | (reads) | same |
| L18 up c400 | 33% / 17% | **tens(a,b)** (R2 0.73): a//10 in {0} -> b//10 in {0,1,2,6,10}; a//10 in {1,6} -> b//10 in {0,1,10}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {0,1,2}; a//10 in {4,5,7} -> b//10 in {0,1,2,10}; a//10 in {8} -> b//10 in {0,1,2,6,7,10}; a//10 in {9,10} -> b//10 in {0,1,2,5,6,7,10} | **tens(a,b)** (R2 0.67): a//10 in {3,4,6} -> b//10 in {8}; a//10 in {5,7} -> b//10 in {8,9}; a//10 in {8} -> b//10 in {3,4,7,8,9}; a//10 in {9} -> b//10 in {3,7,8,9,10}; a//10 in {10} -> b//10 in {2,3,4,7,8,9,10} | (reads) | same |
| L18 up c402 | 13% / 0 | **tens(a,b)** (R2 0.70): a//10 in {6} -> b//10 in {7,8,9}; a//10 in {7,8} -> b//10 in {6,7,8,9}; a//10 in {9} -> b//10 in {6,7,8} | off (on 0) | (reads) | same |
| L18 up c435 | 34% / 13% | **tens(a,b)** (R2 0.65): a//10 in {0,1} -> b//10 in {0,4,5,6,9,10}; a//10 in {2} -> b//10 in {4,5,9,10}; a//10 in {4,9} -> b//10 in {0,5,10}; a//10 in {5,6,10} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {7} -> b//10 in {0,4,5,9,10} | **tens(a,b)** (R2 0.58): a//10 in {2} -> b//10 in {0}; a//10 in {5,6,10} -> b//10 in {0,4,5,9,10}; a//10 in {7} -> b//10 in {0,5} | (reads) | same |
| L18 up c505 | 9% / 15% | **tens(a,b)** (R2 0.61): a//10 in {6} -> b//10 in {0,3,4,8}; a//10 in {7} -> b//10 in {2,3,7,8} | **tens(a,b)** (R2 0.67): a//10 in {2} -> b//10 in {7}; a//10 in {6} -> b//10 in {0,1,2,6,7,10}; a//10 in {7} -> b//10 in {0,1,2,3,6,7} | (reads) | same |
| L18 up c513 | 28% / 15% | **tens(a,b)** (R2 0.62): a//10 in {0,1} -> b//10 in {0,1,3,4,5,8,9,10}; a//10 in {2,10} -> b//10 in {0,4,9,10}; a//10 in {3,4,5,6,9} -> b//10 in {0,9,10}; a//10 in {7,8} -> b//10 in {9,10} | **tens(a,b)** (R2 0.58): a//10 in {0,1} -> b//10 in {0,1,10}; a//10 in {2} -> b//10 in {0,1}; a//10 in {4,7} -> b//10 in {0}; a//10 in {5,6} -> b//10 in {0,10}; a//10 in {9} -> b//10 in {0,9,10}; a//10 in {10} -> b//10 in {0,1,4,5,6,9,10} | (reads) | same |
| L18 up c518 | 19% / 10% | **tens(a,b)** (R2 0.77): a//10 in {3} -> b//10 in {5,6,7}; a//10 in {4,5,6} -> b//10 in {4,5,6,7}; a//10 in {7} -> b//10 in {5,6}; a//10 in {9,10} -> b//10 in {6} | **tens(a,b)** (R2 0.65): a//10 in {3} -> b//10 in {3}; a//10 in {4,5,10} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {3,4,5,6} | (reads) | same |
| L18 up c779 | 10% / 4% | **tens(a,b)** (R2 0.62): a//10 in {4} -> b//10 in {4,5,6,10}; a//10 in {5,10} -> b//10 in {4,5,10}; a//10 in {6} -> b//10 in {4,5}; a//10 in {9} -> b//10 in {5,10} | **tens(a,b)** (R2 0.53): a//10 in {4} -> b//10 in {10}; a//10 in {5} -> b//10 in {4,5,10}; a//10 in {6} -> b//10 in {5}; a//10 in {10} -> b//10 in {4,5} | (reads) | same |

</details>

<details><summary>unexplained: 44 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 down c45 | 1% / 98% | unexplained (best R2 0.08) | always | - | a: mod100 +2%; res: mod100 +8%, mod50 +3% |
| L16 down c53 | 4% / 33% | unexplained (best R2 0.37) | **tens(a,b)** (R2 0.64): a//10 in {3} -> b//10 in {6,7,10}; a//10 in {4} -> b//10 in {4,6,7,8,9,10}; a//10 in {5,8} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {6,7,10} -> b//10 in {5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10} | - | b: mod100 +3% |
| L16 down c67 | 6% / 11% | unexplained (best R2 0.24) | unexplained (best R2 0.47) | - | res: mod50 +3%, mod25 +4%, mod20 +4%, mod10 +3% |
| L16 down c90 | 1% / 2% | unexplained (best R2 0.32) | unexplained (best R2 0.39) | - | res: mod2 +3% |
| L16 down c175 | 1% / 19% | unexplained (best R2 0.30) | unexplained (best R2 0.46) | - | a: mod5 +3% |
| L16 down c176 | 1% / 7% | unexplained (best R2 0.15) | unexplained (best R2 0.21) | - | same |
| L16 down c287 | 4% / 0 | unexplained (best R2 0.28) | off (on 0) | - | same |
| L16 down c470 | 2% / 4% | unexplained (best R2 0.39) | unexplained (best R2 0.45) | - | same |
| L16 down c681 | 4% / 2% | unexplained (best R2 0.30) | unexplained (best R2 0.32) | - | same |
| L16 gate c42 | 2% / 0 | unexplained (best R2 0.34) | off (on 0) | (reads) | same |
| L16 gate c107 | 8% / 5% | unexplained (best R2 0.19) | unexplained (best R2 0.28) | (reads) | same |
| L16 up c90 | 2% / 2% | unexplained (best R2 0.14) | unexplained (best R2 0.22) | (reads) | same |
| L16 up c94 | 3% / 5% | unexplained (best R2 0.39) | unexplained (best R2 0.33) | (reads) | same |
| L17 down c20 | 1% / 2% | unexplained (best R2 0.08) | unexplained (best R2 0.15) | - | same |
| L17 down c29 | 1% / 35% | unexplained (best R2 0.10) | **tens(a,b)** (R2 0.62): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8}; a//10 in {1} -> b//10 in {0,6,7}; a//10 in {3} -> b//10 in {0}; a//10 in {4,5,6} -> b//10 in {0,1,2}; a//10 in {7,8} -> b//10 in {0,1,2,3}; a//10 in {9} -> b//10 in {0,1,2,3,4,5} | - | b: mod100 +4%, mod50 +3%, mod25 +3%; res: mod100 +3% |
| L17 down c38 | 11% / 0 | unexplained (best R2 0.38) | off (on 0) | - | same |
| L17 down c90 | 1% / 0 | unexplained (best R2 0.35) | off (on 0) | - | same |
| L17 gate c47 | 8% / 7% | unexplained (best R2 0.23) | unexplained (best R2 0.38) | (reads) | same |
| L17 gate c89 | 1% / 99% | unexplained (best R2 0.08) | always | (reads) | same |
| L17 up c15 | 2% / 0 | unexplained (best R2 0.46) | off (on 0) | (reads) | same |
| L17 up c173 | 6% / 0 | unexplained (best R2 0.36) | off (on 0) | (reads) | same |
| L18 down c12 | 78% / 59% | unexplained (best R2 0.46) | **tens(a,b)** (R2 0.51): a//10 in {0} -> b//10 in {0,1,4,5,9,10}; a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,8,9}; a//10 in {3} -> b//10 in {0,1,2,6,7,8}; a//10 in {4} -> b//10 in {0,1,2,3,4,6,7,8,9,10}; a//10 in {5,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {0,3,4,5,6,8,9,10}; a//10 in {7} -> b//10 in {0,4,5,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,5,6,7,8,10}; a//10 in {9} -> b//10 in {0,1,2,3,5,6,7,8,9,10} | b: mod50 +5%; res: mod50 +27% | b: mod50 +3%; res: mod50 +13% |
| L18 down c48 | 12% / 4% | unexplained (best R2 0.41) | unexplained (best R2 0.32) | - | same |
| L18 down c53 | 4% / 0 | unexplained (best R2 0.48) | off (on 0) | - | same |
| L18 down c63 | 11% / 20% | unexplained (best R2 0.38) | unexplained (best R2 0.49) | - | same |
| L18 down c99 | 1% / 0 | unexplained (best R2 0.27) | off (on 0) | - | same |
| L18 down c130 | 2% / 0 | unexplained (best R2 0.18) | off (on 0) | - | same |
| L18 down c213 | 1% / 20% | unexplained (best R2 0.21) | **tens(a,b)** (R2 0.58): a//10 in {0} -> b//10 in {6,7}; a//10 in {1} -> b//10 in {7}; a//10 in {2} -> b//10 in {7,8}; a//10 in {6} -> b//10 in {0,1,2}; a//10 in {7,8} -> b//10 in {0,1,2,3} | - | same |
| L18 down c230 | 1% / 30% | unexplained (best R2 0.07) | **tens(a,b)** (R2 0.55): a//10 in {4} -> b//10 in {7}; a//10 in {5} -> b//10 in {7,9,10}; a//10 in {6,7} -> b//10 in {0,6,7,8,9,10}; a//10 in {8} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | - | same |
| L18 down c518 | 2% / 0 | unexplained (best R2 0.32) | off (on 0) | - | same |
| L18 gate c267 | 25% / 11% | unexplained (best R2 0.33) | unexplained (best R2 0.21) | (reads) | same |
| L18 gate c400 | 3% / 1% | unexplained (best R2 0.31) | unexplained (best R2 0.27) | (reads) | same |
| L18 gate c483 | 2% / 0 | unexplained (best R2 0.43) | off (on 0) | (reads) | same |
| L18 gate c544 | 3% / 2% | unexplained (best R2 0.43) | unexplained (best R2 0.19) | (reads) | same |
| L18 gate c702 | 7% / 2% | unexplained (best R2 0.43) | unexplained (best R2 0.15) | (reads) | same |
| L18 up c16 | 19% / 15% | unexplained (best R2 0.31) | unexplained (best R2 0.34) | (reads) | same |
| L18 up c18 | 25% / 25% | unexplained (best R2 0.33) | unexplained (best R2 0.44) | (reads) | same |
| L18 up c59 | 13% / 6% | unexplained (best R2 0.24) | unexplained (best R2 0.18) | (reads) | same |
| L18 up c109 | 4% / 6% | unexplained (best R2 0.29) | unexplained (best R2 0.20) | (reads) | same |
| L18 up c405 | 2% / 5% | unexplained (best R2 0.34) | unexplained (best R2 0.38) | (reads) | same |
| L18 up c511 | 11% / 0 | unexplained (best R2 0.42) | off (on 0) | (reads) | same |
| L18 up c549 | 3% / 1% | unexplained (best R2 0.46) | unexplained (best R2 0.12) | (reads) | same |
| L18 up c708 | 5% / 2% | unexplained (best R2 0.18) | unexplained (best R2 0.13) | (reads) | same |
| L18 up c753 | 1% / 0 | unexplained (best R2 0.19) | off (on 0) | (reads) | same |

</details>

<details><summary>sub only: tens(a,b): 19 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 down c16 | 0 / 28% | off (on 0) | **tens(a,b)** (R2 0.61): a//10 in {3} -> b//10 in {0,1}; a//10 in {4,5} -> b//10 in {0,1,2,3}; a//10 in {6} -> b//10 in {1,2,3}; a//10 in {7,8} -> b//10 in {0,1,2,3,4}; a//10 in {9} -> b//10 in {1,2,3,4,5} | - | same |
| L16 down c77 | 0 / 23% | off (on 0) | **tens(a,b)** (R2 0.70): a//10 in {0,1,2,3} -> b//10 in {0}; a//10 in {4,5} -> b//10 in {0,1}; a//10 in {6,7,8,10} -> b//10 in {0,1,2}; a//10 in {9} -> b//10 in {0,1,2,3,4,5} | - | b: mod100 +3%, mod50 +3%, mod25 +3% |
| L16 down c152 | 0 / 15% | off (on 0) | **tens(a,b)** (R2 0.61): a//10 in {3} -> b//10 in {4}; a//10 in {4} -> b//10 in {5,6,7,8}; a//10 in {5} -> b//10 in {6,7,8}; a//10 in {6} -> b//10 in {7,8,9}; a//10 in {7} -> b//10 in {8,9,10}; a//10 in {8} -> b//10 in {9,10}; a//10 in {10} -> b//10 in {8,9} | - | same |
| L16 down c288 | 0 / 11% | off (on 0) | **tens(a,b)** (R2 0.70): a//10 in {0} -> b//10 in {6,7,8,9}; a//10 in {1} -> b//10 in {5,6,7,8,9}; a//10 in {2} -> b//10 in {7,8,9} | - | same |
| L16 up c77 | 0 / 47% | off (on 0) | **tens(a,b)** (R2 0.68): a//10 in {0,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,9,10}; a//10 in {4,5} -> b//10 in {0,1,10}; a//10 in {6,7,8} -> b//10 in {0,1,2}; a//10 in {9} -> b//10 in {0,1,2,3,4,5} | (reads) | same |
| L17 down c13 | 0 / 40% | off (on 0) | **tens(a,b)** (R2 0.61): a//10 in {2} -> b//10 in {3,4,6}; a//10 in {3} -> b//10 in {3,4,5,6,7}; a//10 in {4} -> b//10 in {4,5,6,7,8}; a//10 in {5} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {6,10} -> b//10 in {4,5,6,7,8,9}; a//10 in {7} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {5,6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10} | - | a: mod100 +3%; b: mod100 +4%; res: mod100 +5%, mod25 +2% |
| L17 down c26 | 0 / 5% | off (on 0) | **tens(a,b)** (R2 0.62): a//10 in {8} -> b//10 in {7,8,10}; a//10 in {9} -> b//10 in {8,9,10}; a//10 in {10} -> b//10 in {4,5,6,7,8,9,10} | - | same |
| L17 down c96 | 0 / 39% | off (on 0) | **tens(a,b)** (R2 0.80): a//10 in {0} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {5,6,7,8,9,10}; a//10 in {5} -> b//10 in {8} | - | a: mod100 +5% |
| L17 down c503 | 0 / 13% | off (on 0) | **tens(a,b)** (R2 0.63): a//10 in {0,1} -> b//10 in {4,5,6,7,8,9}; a//10 in {2} -> b//10 in {9}; a//10 in {10} -> b//10 in {0} | - | same |
| L17 gate c13 | 0 / 25% | off (on 0) | **tens(a,b)** (R2 0.62): a//10 in {3,4} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {6,7,8}; a//10 in {6,10} -> b//10 in {6,7,8,9}; a//10 in {7,8} -> b//10 in {6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10} | (reads) | same |
| L17 gate c504 | 0 / 14% | off (on 0) | **tens(a,b)** (R2 0.77): a//10 in {0} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {8,9} | (reads) | same |
| L17 up c13 | 0 / 55% | off (on 0) | **tens(a,b)** (R2 0.64): a//10 in {1} -> b//10 in {2,3}; a//10 in {2} -> b//10 in {1,2,3,4,5,6,7}; a//10 in {3} -> b//10 in {1,2,3,4,5,6,7,8,9}; a//10 in {4} -> b//10 in {3,4,5,6,7,8}; a//10 in {5,6,7,8} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {6,7,8,9,10}; a//10 in {10} -> b//10 in {1,2,3,4,5,6,7,8,9,10} | (reads) | same |
| L17 up c107 | 0 / 17% | off (on 0) | **tens(a,b)** (R2 0.59): a//10 in {2} -> b//10 in {1}; a//10 in {3} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {1,2,3}; a//10 in {5} -> b//10 in {1,2,3,4,5}; a//10 in {6} -> b//10 in {1,3,4,5,6}; a//10 in {10} -> b//10 in {9} | (reads) | same |
| L17 up c116 | 0 / 14% | off (on 0) | **tens(a,b)** (R2 0.77): a//10 in {0} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {1} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {8,9} | (reads) | same |
| L18 down c245 | 0 / 7% | off (on 0) | **tens(a,b)** (R2 0.69): a//10 in {8,10} -> b//10 in {5,6,7,8}; a//10 in {9} -> b//10 in {6,7,8} | - | same |
| L18 down c434 | 0 / 22% | off (on 0) | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {0,10}; a//10 in {1,4,6,7,8,9} -> b//10 in {0,1}; a//10 in {2,3} -> b//10 in {0}; a//10 in {5} -> b//10 in {0,1,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,9} | - | b: mod50 +2%, mod25 +2% |
| L18 up c85 | 0 / 3% | off (on 0) | **tens(a,b)** (R2 0.54): a//10 in {9} -> b//10 in {6}; a//10 in {10} -> b//10 in {0,1,5,6,7} | (reads) | same |
| L18 up c118 | 0 / 61% | off (on 0) | **tens(a,b)** (R2 0.57): a//10 in {0,1,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,1,4,5,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,4,5,8,9,10}; a//10 in {5} -> b//10 in {0,1,2}; a//10 in {6,7} -> b//10 in {0,1,2,3}; a//10 in {8} -> b//10 in {0,1,2,3,4}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,10} | (reads) | same |
| L18 up c542 | 0 / 2% | off (on 0) | **tens(a,b)** (R2 0.69): a//10 in {9} -> b//10 in {9,10}; a//10 in {10} -> b//10 in {4,5,6,7,8,9} | (reads) | same |

</details>

<details><summary>rarely on (< 0.5 % of prompts): 16 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 down c86 | 0 / 0 | off (on 0) | same | - | same |
| L16 down c96 | 0 / 0 | off (on 0) | same | - | same |
| L16 down c284 | 0 / 0 | off (on 0) | same | - | same |
| L16 down c757 | 0 / 0 | off (on 0) | same | - | same |
| L16 gate c90 | 0 / 0 | off (on 0) | same | (reads) | same |
| L17 down c25 | 0 / 0 | off (on 0) | same | - | same |
| L17 down c115 | 0 / 0 | off (on 0) | same | - | same |
| L18 down c20 | 0 / 0 | off (on 0) | same | - | same |
| L18 down c677 | 0 / 0 | off (on 0) | same | - | same |
| L18 gate c165 | 0 / 0 | off (on 0) | same | (reads) | same |
| L18 gate c801 | 0 / 0 | off (on 0) | same | (reads) | same |
| L18 up c73 | 0 / 0 | off (on 0) | same | (reads) | same |
| L18 up c261 | 0 / 0 | off (on 0) | same | (reads) | same |
| L18 up c573 | 0 / 0 | off (on 0) | same | (reads) | same |
| L18 up c657 | 0 / 0 | off (on 0) | same | (reads) | same |
| L18 up c676 | 0 / 0 | off (on 0) | same | (reads) | same |

</details>

<details><summary>b: 15 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 down c33 | 14% / 0 | **b** (R2 0.60): b in {1..11} | off (on 0) | - | same |
| L16 down c42 | 48% / 2% | **b** (R2 0.84): b in {25..53, 81..100} [coarser: b mod 50 in {0, 27..49}, R2 0.86] | unexplained (best R2 0.30) | b: mod50 +10%; res: mod50 +8% | - |
| L16 down c161 | 8% / 16% | **b** (R2 0.87): b in {1..7} | **tens(a,b)** (R2 0.86): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1,2,3,4,5,6,7,8,9,10} -> b//10 in {0} | - | b: mod25 +2% |
| L16 gate c501 | 7% / 23% | **b** (R2 0.64): b in {2..9} | **tens(a,b)** (R2 0.72): a//10 in {0,1,2,3} -> b//10 in {0}; a//10 in {4,5,6,7,8} -> b//10 in {0,1,2}; a//10 in {9} -> b//10 in {0,1,2,3}; a//10 in {10} -> b//10 in {0,1} | (reads) | same |
| L16 up c130 | 28% / 80% | **b** (R2 0.68): b in {49, 53..79, 81} | **tens(a,b)** (R2 0.62): a//10 in {0,1,2,3,4} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,3,4,5,9}; a//10 in {6,7,8,10} -> b//10 in {0,1,2,3,4,5}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7} | (reads) | same |
| L16 up c161 | 8% / 10% | **b** (R2 0.91): b in {1..8} | **b//10** (R2 0.78): (tens) b in {1..9} | (reads) | same |
| L17 down c170 | 9% / 0 | **b** (R2 0.70): b in {5..11, 86, 88} | off (on 0) | - | same |
| L17 gate c11 | 42% / 12% | **b** (R2 0.76): b in {57..100} | unexplained (best R2 0.44) | (reads) | same |
| L17 gate c107 | 36% / 34% | **b** (R2 0.75): b in {25..60, 64} | unexplained (best R2 0.50) | (reads) | same |
| L17 up c23 | 55% / 13% | **b** (R2 0.63): b in {3, 5, 27..58, 75, 77..100} [coarser: b mod 50 in {0..5, 27..49}, R2 0.96] | **tens(a,b)** (R2 0.67): a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {1}; a//10 in {6} -> b//10 in {0,5}; a//10 in {7} -> b//10 in {0,5,6}; a//10 in {8,10} -> b//10 in {5,6} | (reads) | same |
| L17 up c136 | 16% / 1% | **b** (R2 0.84): b in {5..21} | unexplained (best R2 0.17) | (reads) | same |
| L17 up c170 | 6% / 1% | **b** (R2 0.91): b in {15..20} | unexplained (best R2 0.19) | (reads) | same |
| L18 up c12 | 55% / 38% | **b** (R2 0.67): b in {10..33, 58..86} [coarser: b mod 50 in {9..36}, R2 0.98] | **tens(a,b)** (R2 0.68): a//10 in {2} -> b//10 in {8}; a//10 in {3,4,8} -> b//10 in {1,2,3,6,7,8}; a//10 in {5} -> b//10 in {2,3,7,8}; a//10 in {6} -> b//10 in {3,7,8}; a//10 in {7} -> b//10 in {3,8}; a//10 in {9} -> b//10 in {1,2,3,6,7,8,10}; a//10 in {10} -> b//10 in {1,2,3,4,6,7,8} | (reads) | same |
| L18 up c297 | 19% / 4% | **b** (R2 0.60): b in {36..38, 40..48, 50..53, 55..56} | unexplained (best R2 0.35) | (reads) | same |
| L18 up c543 | 27% / 12% | **b** (R2 0.65): b in {22..47} | **tens(a,b)** (R2 0.55): a//10 in {3,7,8} -> b//10 in {6}; a//10 in {4,5,6,9} -> b//10 in {6,7}; a//10 in {10} -> b//10 in {5,6,7} | (reads) | same |

</details>

<details><summary>always: 12 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 down c1 | 100% / 100% | always | same | - | same |
| L16 gate c1 | 99% / 14% | always | **tens(a,b)** (R2 0.54): a//10 in {2} -> b//10 in {1}; a//10 in {3} -> b//10 in {2}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {3,4,5}; a//10 in {7} -> b//10 in {4,5,6} | (reads) | same |
| L16 gate c4 | 100% / 100% | always | same | (reads) | same |
| L16 gate c37 | 100% / 100% | always | same | (reads) | same |
| L16 up c4 | 100% / 100% | always | same | (reads) | same |
| L17 down c3 | 100% / 100% | always | same | - | same |
| L17 down c5 | 100% / 1% | always | unexplained (best R2 0.10) | - | same |
| L17 gate c5 | 99% / 0 | always | off (on 0) | (reads) | same |
| L17 up c9 | 100% / 100% | always | same | (reads) | same |
| L18 down c0 | 100% / 100% | always | same | - | same |
| L18 gate c306 | 100% / 100% | always | same | (reads) | same |
| L18 up c142 | 95% / 9% | always | unexplained (best R2 0.26) | (reads) | same |

</details>

<details><summary>sub only: unexplained: 12 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 down c17 | 0 / 5% | off (on 0) | unexplained (best R2 0.46) | - | same |
| L16 down c71 | 0 / 62% | off (on 0) | unexplained (best R2 0.42) | - | a: mod100 +3%, mod50 +3%; b: mod10 +11%; res: mod100 +3%, mod50 +3%, mod25 +3%, mod20 +3% |
| L16 gate c79 | 0 / 4% | off (on 0) | unexplained (best R2 0.25) | (reads) | same |
| L16 gate c175 | 0 / 3% | off (on 0) | unexplained (best R2 0.47) | (reads) | same |
| L16 up c71 | 0 / 23% | off (on 0) | unexplained (best R2 0.37) | (reads) | same |
| L17 down c136 | 0 / 11% | off (on 0) | unexplained (best R2 0.42) | - | b: mod20 +3% |
| L17 up c5 | 0 / 4% | off (on 0) | unexplained (best R2 0.35) | (reads) | same |
| L18 down c167 | 0 / 2% | off (on 0) | unexplained (best R2 0.39) | - | same |
| L18 up c63 | 0 / 2% | off (on 0) | unexplained (best R2 0.36) | (reads) | same |
| L18 up c77 | 0 / 12% | off (on 0) | unexplained (best R2 0.43) | (reads) | same |
| L18 up c101 | 0 / 1% | off (on 0) | unexplained (best R2 0.21) | (reads) | same |
| L18 up c227 | 0 / 1% | off (on 0) | unexplained (best R2 0.28) | (reads) | same |

</details>

<details><summary>b%20: 7 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 down c24 | 50% / 42% | **b%20** (R2 0.99): b mod 20 in {0..4, 15..19} | **b%20** (R2 0.69): b mod 20 in {0..4, 15..19} | b: mod20 +43%, mod4 +26% | b: mod20 +25%, mod4 +11% |
| L16 up c24 | 49% / 29% | **b%20** (R2 0.96): b mod 20 in {0..4, 15..19} | unexplained (best R2 0.42) | (reads) | same |
| L16 up c821 | 49% / 25% | **b%20** (R2 0.95): b mod 20 in {5..14} | unexplained (best R2 0.37) | (reads) | same |
| L18 down c16 | 37% / 21% | **b%20** (R2 0.59): b mod 20 in {0..4, 15..19} | unexplained (best R2 0.27) | a: mod20 +10%, mod4 +4%; b: mod20 +8%, mod4 +5%; res: mod20 +24% | a: mod20 +4%; res: mod25 +2%, mod20 +12% |
| L18 down c18 | 35% / 28% | **b%20** (R2 0.60): b mod 20 in {0..4, 17..19} | unexplained (best R2 0.38) | a: mod20 +11%, mod4 +5%; res: mod20 +46% | a: mod20 +4%; res: mod20 +25% |
| L18 gate c16 | 38% / 19% | **b%20** (R2 0.65): b mod 20 in {0..4, 15..19} | unexplained (best R2 0.27) | (reads) | same |
| L18 gate c18 | 41% / 32% | **b%20** (R2 0.70): b mod 20 in {0..4, 15..19} | unexplained (best R2 0.47) | (reads) | same |

</details>

<details><summary>a//10: 5 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 gate c10 | 49% / 40% | **a//10** (R2 0.78): (tens) a in {2..4, 56..100} | **a//10** (R2 0.71): (tens) a in {58..97, 100} | (reads) | same |
| L16 gate c130 | 45% / 24% | **a//10** (R2 0.80): (tens) a in {1..25, 81..100} | **tens(a,b)** (R2 0.71): a//10 in {0} -> b//10 in {0,1,2,9,10}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {1,2}; a//10 in {8} -> b//10 in {0,3,4,5,6,8,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | (reads) | same |
| L16 gate c832 | 38% / 14% | **a//10** (R2 0.82): (tens) a in {11, 13..49} | **tens(a,b)** (R2 0.62): a//10 in {1} -> b//10 in {0,1}; a//10 in {2} -> b//10 in {0,1,2}; a//10 in {3,4} -> b//10 in {0,1,2,3} | (reads) | same |
| L16 up c106 | 41% / 23% | **a//10** (R2 0.78): (tens) a in {1..42, 100} | **a//10** (R2 0.61): (tens) a in {10..11, 13..32, 35, 100} | (reads) | same |
| L18 gate c153 | 22% / 18% | **a//10** (R2 0.81): (tens) a in {59..79} | **tens(a,b)** (R2 0.76): a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,10}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,6,7} | (reads) | same |

</details>

<details><summary>a%20: 4 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 down c73 | 49% / 31% | **a%20** (R2 0.95): a mod 20 in {0..4, 15..19} | **a** (R2 0.61): a in {21, 35..41, 43, 55..64, 75..84, 99..100} | a: mod20 +19%, mod4 +8% | a: mod20 +9%, mod4 +4% |
| L16 gate c73 | 50% / 31% | **a%20** (R2 0.98): a mod 20 in {5..14} | **a** (R2 0.63): a in {27..32, 45..47, 49..53, 65..74, 85..94} | (reads) | same |
| L18 down c59 | 17% / 8% | **a%20** (R2 0.52): a mod 20 in {15..17} | unexplained (best R2 0.24) | b: mod20 +3%; res: mod20 +14% | res: mod20 +3% |
| L18 gate c59 | 16% / 6% | **a%20** (R2 0.58): a mod 20 in {15..18} | unexplained (best R2 0.25) | (reads) | same |

</details>

<details><summary>b//10: 4 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 gate c106 | 47% / 20% | **b//10** (R2 0.86): (tens) b in {1..47, 49} | **tens(a,b)** (R2 0.66): a//10 in {4,9} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {6,7,8}; a//10 in {6} -> b//10 in {5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9,10}; a//10 in {8} -> b//10 in {5,6,7,8,9}; a//10 in {10} -> b//10 in {6,7,8,9} | (reads) | same |
| L16 up c10 | 52% / 43% | **b//10** (R2 0.79): (tens) b in {53..100} | **tens(a,b)** (R2 0.74): a//10 in {2} -> b//10 in {6}; a//10 in {3} -> b//10 in {3,5,6,7}; a//10 in {4} -> b//10 in {3,4,6,7,8}; a//10 in {5,6,7} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {8,9,10} -> b//10 in {4,5,6,7,8,9,10} | (reads) | same |
| L16 up c25 | 35% / 17% | **b//10** (R2 0.75): (tens) b in {51..81, 83..85} | **tens(a,b)** (R2 0.53): a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4,9} -> b//10 in {3}; a//10 in {5,6,7,8} -> b//10 in {3,4}; a//10 in {10} -> b//10 in {1,2,3,4,5,6,7,8} | (reads) | same |
| L16 up c42 | 51% / 0 | **b//10** (R2 0.77): (tens) b in {1..49} | off (on 0) | (reads) | same |

</details>

<details><summary>b%50: 4 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L17 down c10 | 65% / 46% | **b%50** (R2 0.84): b mod 50 in {11..43} | **tens(a,b)** (R2 0.69): a//10 in {1} -> b//10 in {1,2,3}; a//10 in {2} -> b//10 in {1,2,3,6,7}; a//10 in {3,4,5,6,7,8,9,10} -> b//10 in {1,2,3,6,7,8} | b: mod50 +15% | b: mod50 +11% |
| L17 up c10 | 64% / 40% | **b%50** (R2 0.85): b mod 50 in {11..42} | **tens(a,b)** (R2 0.65): a//10 in {1} -> b//10 in {2}; a//10 in {2} -> b//10 in {1,2,3,7}; a//10 in {3,4,5,6,7,8,9,10} -> b//10 in {1,2,3,6,7,8} | (reads) | same |
| L18 up c22 | 57% / 23% | **b%50** (R2 0.77): b mod 50 in {0..26, 48..49} | **tens(a,b)** (R2 0.59): a//10 in {3} -> b//10 in {3}; a//10 in {4,8} -> b//10 in {3,4,8}; a//10 in {5,6} -> b//10 in {3,4,5,8,9,10}; a//10 in {7} -> b//10 in {3,4,9}; a//10 in {9} -> b//10 in {3,8,9,10}; a//10 in {10} -> b//10 in {3,4,5,7,8,9,10} | (reads) | same |
| L18 up c27 | 61% / 50% | **b%50** (R2 0.63): b mod 50 in {0..9, 11, 31..49} | **tens(a,b)** (R2 0.65): a//10 in {0} -> b//10 in {0,1,4,5,8,9,10}; a//10 in {1} -> b//10 in {0,9,10}; a//10 in {2,7} -> b//10 in {0,5,9,10}; a//10 in {3,4} -> b//10 in {0,1,4,5,9,10}; a//10 in {5} -> b//10 in {0,1,5,6,9,10}; a//10 in {6} -> b//10 in {0,1,5,9,10}; a//10 in {8,10} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,8,9,10} | (reads) | same |

</details>

<details><summary>a%50: 4 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L17 down c33 | 60% / 26% | **a%50** (R2 0.72): a mod 50 in {19..47} | **tens(a,b)** (R2 0.59): a//10 in {2} -> b//10 in {0,1,2,3}; a//10 in {3} -> b//10 in {0,1,2,3,4}; a//10 in {4} -> b//10 in {3}; a//10 in {7} -> b//10 in {3,4,5,6,7,8}; a//10 in {8} -> b//10 in {0,3,4,5,6,7,8,9}; a//10 in {9} -> b//10 in {8}; a//10 in {10} -> b//10 in {8,9} | a: mod50 +15% | a: mod50 +6% |
| L18 gate c12 | 56% / 38% | **a%50** (R2 0.85): a mod 50 in {14..41} | **a** (R2 0.63): a in {20..36, 66..89} [coarser: a mod 50 in {17..39}, R2 0.88] | (reads) | same |
| L18 gate c21 | 57% / 38% | **a%50** (R2 0.76): a mod 50 in {0..25, 46..49} | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {0,1,10}; a//10 in {1} -> b//10 in {0,1,2,5,6}; a//10 in {2} -> b//10 in {2}; a//10 in {3} -> b//10 in {6}; a//10 in {4,7} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {0,1,2,3,4,5,6,7,10}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {7}; a//10 in {10} -> b//10 in {0,1,2,3,5,6,7,8,10} | (reads) | same |
| L18 gate c22 | 55% / 47% | **a%50** (R2 0.66): a mod 50 in {0..11, 34..49} | **a** (R2 0.68): a in {1, 34..61, 81..100} | (reads) | same |

</details>

<details><summary>sub only: a//10: 3 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 down c164 | 0 / 15% | off (on 0) | **a//10** (R2 0.70): (tens) a in {1..16} | - | a: mod100 +3%, mod50 +4%, mod25 +2% |
| L16 up c107 | 0 / 1% | off (on 0) | **a//10** (R2 0.81): (tens) a in {100} | (reads) | same |
| L18 down c290 | 0 / 25% | off (on 0) | **a//10** (R2 0.69): (tens) a in {2..24} | - | a: mod100 +3%, mod50 +3% |

</details>

<details><summary>res//10: 2 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 down c61 | 40% / 2% | **res//10** (R2 0.78): (tens) res in {2..90} | unexplained (best R2 0.50) | res: mod100 +2% | - |
| L18 down c23 | 52% / 35% | **res//10** (R2 0.71): (tens) res in {2..29, 31, 83..130, 179..200} [coarser: res mod 100 in {0..31, 81..99}, R2 0.99] | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {0,9,10}; a//10 in {1} -> b//10 in {0,1,10}; a//10 in {2} -> b//10 in {0,1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3,4,6}; a//10 in {5} -> b//10 in {3,4,5,6}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9}; a//10 in {9} -> b//10 in {0,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,8,9,10} | res: mod100 +22% | res: mod100 +6%, mod50 +5% |

</details>

<details><summary>a: 1 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 down c214 | 4% / 1% | **a** (R2 0.63): a in {44, 64, 84} [coarser: a mod 20 in {4}, R2 0.82] | **a** (R2 0.51): a in {64, 84} | - | same |

</details>

<details><summary>b%10: 1 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 down c295 | 23% / 19% | **b%10** (R2 0.80): b mod 10 in {6..7} | **b%10** (R2 0.54): b mod 10 in {3..4} | b: mod10 +4% | b: mod10 +5%, mod5 +2% |

</details>

<details><summary>sub only: b: 1 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 down c501 | 0 / 14% | off (on 0) | **b** (R2 0.69): b in {1..10, 12, 14, 16, 18} | - | same |

</details>

<details><summary>sub only: b%10: 1 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 gate c71 | 0 / 29% | off (on 0) | **b%10** (R2 0.51): b mod 10 in {6..9} | (reads) | same |

</details>

<details><summary>a%10: 1 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L16 gate c470 | 11% / 14% | **a%10** (R2 0.49): a mod 10 in {0} | unexplained (best R2 0.41) | (reads) | same |

</details>

<details><summary>sub only: res//10: 1 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L17 down c89 | 0 / 19% | off (on 0) | **res//10** (R2 0.53): (tens) res in {0..22, 99} | - | res: mod100 +3%, mod50 +5%, mod25 +3%, mod20 +3% |

</details>

<details><summary>res%50: 1 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L18 down c22 | 65% / 37% | **res%50** (R2 0.65): res mod 50 in {0, 18..49} | **tens(a,b)** (R2 0.64): a//10 in {2} -> b//10 in {10}; a//10 in {3} -> b//10 in {0,1,5,6,9,10}; a//10 in {4} -> b//10 in {0,1,2,5,6,7,9,10}; a//10 in {5} -> b//10 in {0,1,2,6,7}; a//10 in {6} -> b//10 in {3}; a//10 in {7} -> b//10 in {3,4,5,10}; a//10 in {8} -> b//10 in {0,4,5,6,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,10}; a//10 in {10} -> b//10 in {0,1,2,5,6,7,10} | b: mod50 +2%; res: mod50 +15% | res: mod50 +5% |

</details>

<details><summary>res: 1 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L18 down c34 | 1% / 4% | **res** (R2 0.72): res in {2..7, 9, 13} | **res%100** (R2 0.50): res mod 100: no class above 0.5 (max 0.45) | - | same |

</details>

<details><summary>sub only: always: 1 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L18 down c75 | 0 / 96% | off (on 0) | always | - | same |

</details>

<details><summary>res%2: 1 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L18 gate c299 | 46% / 40% | **res%2** (R2 0.80): res mod 2 in {0} | **units(a,b)** (R2 0.55): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} | (reads) | same |

</details>

## Result-code writers, L19-L31

<details><summary>`mlp_in.20`, add: who writes the a+b code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.98 | L19 MLP +0.38, L16 MLP +0.23, L18 MLP +0.20, L17 MLP +0.15 |
| mod 50 | 1.05 | L18 MLP +0.52, L19 MLP +0.49, L17 MLP +0.03 |
| mod 25 | 0.96 | L19 MLP +0.87, L18 MLP +0.05 |
| mod 20 | 0.90 | L19 MLP +0.48, L18 MLP +0.41 |
| mod 10 | 0.99 | L18 MLP +0.38, L19 MLP +0.32, L17 MLP +0.28 |
| mod 5 | 0.93 | L18 MLP +0.56, L16 MLP +0.17, L19 MLP +0.12, L17 MLP +0.06 |
| mod 4 | 0.67 | L19 MLP +0.25, L20 attn +0.17, L18 MLP +0.17, L18 attn +0.03 |
| mod 2 | 0.87 | L18 MLP +0.51, L17 MLP +0.23, L19 MLP +0.13 |

<details><summary>period 100: 11 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L19 down c24 | +0.169 | 44% | **res//10** (R2 0.77): (tens) res in {24, 26, 28..80, 142..174} [coarser: res mod 100 in {34..76}, R2 0.81] |
| L18 down c23 | +0.135 | 52% | **res//10** (R2 0.71): (tens) res in {2..29, 31, 83..130, 179..200} [coarser: res mod 100 in {0..31, 81..99}, R2 0.99] |
| L19 down c28 | +0.118 | 57% | **res//10** (R2 0.79): (tens) res in {14..52, 100..162, 164} [coarser: res mod 100 in {1, 10..56}, R2 0.81] |
| L16 down c10 | +0.093 | 48% | **tens(a,b)** (R2 0.80): a//10 in {0} -> b//10 in {2,3,5,6,7,8}; a//10 in {1} -> b//10 in {6}; a//10 in {5} -> b//10 in {3,4,5,7,8,9,10}; a//10 in {6,10} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {8,9} -> b//10 in {1,2,3,4,5,6,7,8,9,10} |
| L17 down c11 | +0.085 | 46% | **tens(a,b)** (R2 0.79): a//10 in {2} -> b//10 in {6,7}; a//10 in {3} -> b//10 in {6,7,9,10}; a//10 in {4,6} -> b//10 in {0,1,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {0,5,6,7,8,9,10}; a//10 in {9,10} -> b//10 in {4,5,6,7,8,9,10} |
| L16 down c29 | +0.048 | 50% | **tens(a,b)** (R2 0.80): a//10 in {0,9,10} -> b//10 in {0,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {8}; a//10 in {5} -> b//10 in {5,6,7}; a//10 in {6} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {2,3,4,5,6,7,8,9,10} |
| L19 down c86 | +0.042 | 25% | **res//10** (R2 0.92): (tens) res in {132..200} |
| L17 down c40 | +0.036 | 47% | **tens(a,b)** (R2 0.78): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7}; a//10 in {1,2} -> b//10 in {2,3,4,5,6,7}; a//10 in {3,4} -> b//10 in {1,2,3,4,5,6}; a//10 in {5} -> b//10 in {1,2,3,4,5}; a//10 in {6} -> b//10 in {1,2,3,4}; a//10 in {8} -> b//10 in {2}; a//10 in {9,10} -> b//10 in {0,1,2,3,4} |
| L19 down c33 | +0.023 | 33% | **res%100** (R2 0.73): res mod 100 in {1, 75..98} |
| L16 down c106 | +0.023 | 35% | **tens(a,b)** (R2 0.77): a//10 in {0} -> b//10 in {2,3,4}; a//10 in {1,4} -> b//10 in {1,2,3}; a//10 in {2} -> b//10 in {2,3}; a//10 in {5} -> b//10 in {1,2,3,4}; a//10 in {6,7} -> b//10 in {0,1,2,3,4,10}; a//10 in {8} -> b//10 in {0,1,2,3,10}; a//10 in {9} -> b//10 in {1,2} |
| L16 down c43 | +0.022 | 52% | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {4,5,6,7,8}; a//10 in {1} -> b//10 in {3,4,5,6,7,8}; a//10 in {2} -> b//10 in {2,3,4,5,6,7,8}; a//10 in {3} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {1,2,3,4,5,7,8,9,10}; a//10 in {6} -> b//10 in {1,2,8,9,10}; a//10 in {7} -> b//10 in {1,8}; a//10 in {8} -> b//10 in {6}; a//10 in {9} -> b//10 in {5,6,7}; a//10 in {10} -> b//10 in {5,6} |

</details>

<details><summary>period 50: 11 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L19 down c0 | +0.257 | 51% | **res%50** (R2 0.84): res mod 50 in {14..39} |
| L18 down c12 | +0.137 | 78% | unexplained (best R2 0.46) |
| L19 down c11 | +0.111 | 46% | **tens(a,b)** (R2 0.71): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {4,8,9,10}; a//10 in {2,7} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {3,8} -> b//10 in {1,2,3,6,7,8}; a//10 in {4} -> b//10 in {1,2,5,6,7}; a//10 in {5} -> b//10 in {4,5,6}; a//10 in {6} -> b//10 in {3,4,5,8,9,10}; a//10 in {9} -> b//10 in {1,2,6,7,10}; a//10 in {10} -> b//10 in {0,1,2,5,6,7,10} |
| L18 down c21 | +0.097 | 70% | **tens(a,b)** (R2 0.57): a//10 in {0} -> b//10 in {1,2,3,4,6,7,8,9}; a//10 in {1,6} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,1,2,5,6,7,10}; a//10 in {3} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {4} -> b//10 in {2,3,4,8,9,10}; a//10 in {5} -> b//10 in {1,2,3,4,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,4,5,6,7,9,10}; a//10 in {8} -> b//10 in {0,1,4,5,9,10}; a//10 in {9} -> b//10 in {3,4,8,9,10}; a//10 in {10} -> b//10 in {2,3,4,7,8,9} |
| L19 down c20 | +0.086 | 60% | **res%50** (R2 0.68): res mod 50 in {4..31} |
| L18 down c22 | +0.077 | 65% | **res%50** (R2 0.65): res mod 50 in {0, 18..49} |
| L18 down c25 | +0.076 | 78% | **tens(a,b)** (R2 0.52): a//10 in {0} -> b//10 in {3,4,5,7,8,9,10}; a//10 in {1} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {2} -> b//10 in {1,2,3,4,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,2,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,4,5,6,8,9,10}; a//10 in {5} -> b//10 in {0,3,4,5,7,8,9,10}; a//10 in {6} -> b//10 in {0,2,3,4,7,8,9,10}; a//10 in {7,8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {0,1,3,4,5,6,8,9,10}; a//10 in {10} -> b//10 in {0,3,4,5,6,7,8,9,10} |
| L18 down c36 | +0.047 | 58% | **tens(a,b)** (R2 0.54): a//10 in {0} -> b//10 in {4,5,6,10}; a//10 in {1} -> b//10 in {4,5,10}; a//10 in {2,3} -> b//10 in {1,2,3,6,7}; a//10 in {4,9} -> b//10 in {0,1,2,5,6,7,10}; a//10 in {5} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {6} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {7,8} -> b//10 in {1,2,3,5,6,7,8}; a//10 in {10} -> b//10 in {0,1,2,4,5,6,7,10} |
| L18 down c27 | +0.043 | 33% | **tens(a,b)** (R2 0.69): a//10 in {2} -> b//10 in {4,8,9,10}; a//10 in {3} -> b//10 in {3,4,7,8,9,10}; a//10 in {4} -> b//10 in {2,3,7,8,9}; a//10 in {5} -> b//10 in {8}; a//10 in {7} -> b//10 in {3,4,5,8,9,10}; a//10 in {8} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {9} -> b//10 in {2,3,4,7,8}; a//10 in {10} -> b//10 in {2,3,7,8} |
| L17 down c23 | +0.033 | 65% | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {1,4,8,9}; a//10 in {1} -> b//10 in {1,3,4,5,6,8,9,10}; a//10 in {2,3,6} -> b//10 in {1,2,3,4,5,6,8,9,10}; a//10 in {4} -> b//10 in {1,3,8,9}; a//10 in {5} -> b//10 in {1,3,4,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,8,9,10}; a//10 in {9} -> b//10 in {1,3,4,6,8}; a//10 in {10} -> b//10 in {3,4,8,9} |
| L18 down c61 | +0.032 | 33% | **tens(a,b)** (R2 0.65): a//10 in {0} -> b//10 in {0,1,4,5,6,7,10}; a//10 in {1,6} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {2} -> b//10 in {0,4,5,10}; a//10 in {4} -> b//10 in {0,1,6}; a//10 in {5,10} -> b//10 in {0,1,5,6,10}; a//10 in {7} -> b//10 in {0,9,10}; a//10 in {9} -> b//10 in {1,6} |

</details>

<details><summary>period 25: 3 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L19 down c0 | +0.652 | 51% | **res%50** (R2 0.84): res mod 50 in {14..39} |
| L19 down c11 | +0.177 | 46% | **tens(a,b)** (R2 0.71): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {4,8,9,10}; a//10 in {2,7} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {3,8} -> b//10 in {1,2,3,6,7,8}; a//10 in {4} -> b//10 in {1,2,5,6,7}; a//10 in {5} -> b//10 in {4,5,6}; a//10 in {6} -> b//10 in {3,4,5,8,9,10}; a//10 in {9} -> b//10 in {1,2,6,7,10}; a//10 in {10} -> b//10 in {0,1,2,5,6,7,10} |
| L19 down c20 | +0.034 | 60% | **res%50** (R2 0.68): res mod 50 in {4..31} |

</details>

<details><summary>period 20: 4 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L19 down c6 | +0.468 | 66% | unexplained (best R2 0.26) |
| L18 down c18 | +0.194 | 35% | **b%20** (R2 0.60): b mod 20 in {0..4, 17..19} |
| L18 down c16 | +0.167 | 37% | **b%20** (R2 0.59): b mod 20 in {0..4, 15..19} |
| L18 down c59 | +0.049 | 17% | **a%20** (R2 0.52): a mod 20 in {15..17} |

</details>

<details><summary>period 10: 15 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L19 down c41 | +0.131 | 40% | **units(a,b)** (R2 0.85): a%10 in {0} -> b%10 in {6,7,8,9}; a%10 in {1} -> b%10 in {5,6,7,8}; a%10 in {2} -> b%10 in {5,6,7}; a%10 in {3} -> b%10 in {5,6}; a%10 in {5} -> b%10 in {1,2,3,4}; a%10 in {6} -> b%10 in {0,1,2,3,9}; a%10 in {7} -> b%10 in {0,1,2,8,9}; a%10 in {8} -> b%10 in {0,1,7,8,9}; a%10 in {9} -> b%10 in {0,6,7,8,9} |
| L18 down c32 | +0.125 | 46% | **units(a,b)** (R2 0.85): a%10 in {0} -> b%10 in {3,4,5,6}; a%10 in {1,3,9} -> b%10 in {3,4,5,6,7}; a%10 in {2} -> b%10 in {5,6,7}; a%10 in {4,5} -> b%10 in {2,3,4,5,6,7}; a%10 in {6} -> b%10 in {1,2,3,4,5,6}; a%10 in {7} -> b%10 in {1,2,3,4}; a%10 in {8} -> b%10 in {4,5,6} |
| L17 down c34 | +0.074 | 35% | **units(a,b)** (R2 0.93): a%10 in {0} -> b%10 in {5,6,7}; a%10 in {1} -> b%10 in {5,6}; a%10 in {5} -> b%10 in {0,1,2,5,6,7,8,9}; a%10 in {6} -> b%10 in {0,1,5,6,7,8,9}; a%10 in {7} -> b%10 in {0,5,6,7,8,9}; a%10 in {8} -> b%10 in {5,6,7,8,9}; a%10 in {9} -> b%10 in {5,6,7,8} |
| L19 down c55 | +0.074 | 23% | **res%10** (R2 0.81): res mod 10 in {0..1} |
| L19 down c149 | +0.071 | 27% | **res%100** (R2 0.77): res mod 100 in {2..4, 12..14, 22..24, 32..34, 42..44, 52..53, 62..63, 72..73, 82..83, 92..93} [coarser: res mod 10 in {2..4}, R2 0.97] |
| L17 down c49 | +0.061 | 38% | **units(a,b)** (R2 0.90): a%10 in {5} -> b%10 in {0,1,2,3,5,6,7,9}; a%10 in {6} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {7} -> b%10 in {0,1,2,3,5,6,7,8,9}; a%10 in {8} -> b%10 in {0,1,2,5,6,8,9}; a%10 in {9} -> b%10 in {0,1,5,9} |
| L18 down c88 | +0.059 | 30% | **units(a,b)** (R2 0.89): a%10 in {1} -> b%10 in {4,5,6}; a%10 in {2} -> b%10 in {3,4,5,6}; a%10 in {3} -> b%10 in {2,3,4,5,6}; a%10 in {4} -> b%10 in {1,2,3,4,5}; a%10 in {5} -> b%10 in {1,2,3,4,5,6,7}; a%10 in {6} -> b%10 in {1,5,6,7}; a%10 in {7} -> b%10 in {5,6} |
| L17 down c39 | +0.057 | 39% | **units(a,b)** (R2 0.87): a%10 in {0,4} -> b%10 in {5,6,7,8,9}; a%10 in {1} -> b%10 in {5,6,7,8}; a%10 in {2} -> b%10 in {7,8}; a%10 in {3} -> b%10 in {0,6,7,8,9}; a%10 in {5} -> b%10 in {6,7,8}; a%10 in {7} -> b%10 in {6,7,8,9}; a%10 in {8,9} -> b%10 in {0,5,6,7,8,9} |
| L18 down c62 | +0.053 | 25% | **units(a,b)** (R2 0.89): a%10 in {0} -> b%10 in {0,1,2,8,9}; a%10 in {1} -> b%10 in {0,1,7,8,9}; a%10 in {2} -> b%10 in {0,7,8,9}; a%10 in {7} -> b%10 in {1,2}; a%10 in {8} -> b%10 in {0,1,2}; a%10 in {9} -> b%10 in {0,1,2,9} |
| L18 down c65 | +0.050 | 36% | **units(a,b)** (R2 0.88): a%10 in {0,5} -> b%10 in {7,8,9}; a%10 in {1,6} -> b%10 in {6,7,8,9}; a%10 in {2,7} -> b%10 in {5,6,7,8}; a%10 in {3} -> b%10 in {4,5,6,7}; a%10 in {4} -> b%10 in {4,5,6}; a%10 in {8} -> b%10 in {5,6,7}; a%10 in {9} -> b%10 in {5,6,9} |
| L18 down c98 | +0.045 | 34% | **units(a,b)** (R2 0.81): a%10 in {0,1} -> b%10 in {4,5,6,7}; a%10 in {2} -> b%10 in {0,4,5,6,8,9}; a%10 in {3} -> b%10 in {0,4,5,8,9}; a%10 in {4} -> b%10 in {7,8}; a%10 in {7} -> b%10 in {8,9}; a%10 in {8} -> b%10 in {6,7,8,9}; a%10 in {9} -> b%10 in {5,6,7,8} |
| L17 down c86 | +0.040 | 31% | **units(a,b)** (R2 0.85): a%10 in {0} -> b%10 in {6,7,8}; a%10 in {1} -> b%10 in {5,6,7,8,9}; a%10 in {2} -> b%10 in {5,6,7,8}; a%10 in {3} -> b%10 in {0,1,2,6,7,8,9}; a%10 in {4} -> b%10 in {0,1,2,8,9}; a%10 in {5} -> b%10 in {0,1,8,9}; a%10 in {6} -> b%10 in {0,8,9} |
| L17 down c52 | +0.035 | 26% | **units(a,b)** (R2 0.87): a%10 in {0} -> b%10 in {2,3,4}; a%10 in {1} -> b%10 in {1,2,3,4}; a%10 in {2} -> b%10 in {0,1,2,3,4}; a%10 in {3,4} -> b%10 in {0,1,2,3,4,9} |
| L19 down c74 | +0.024 | 20% | **res%10** (R2 0.73): res mod 10 in {8..9} |
| L18 down c120 | +0.021 | 19% | **units(a,b)** (R2 0.85): a%10 in {0} -> b%10 in {3}; a%10 in {5} -> b%10 in {5,6,7}; a%10 in {6} -> b%10 in {4,5,6}; a%10 in {7} -> b%10 in {3,4,5,6}; a%10 in {8} -> b%10 in {2,3,4,5}; a%10 in {9} -> b%10 in {2,3,4} |

</details>

<details><summary>period 5: 15 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L18 down c26 | +0.177 | 69% | **units(a,b)** (R2 0.80): a%10 in {0,5} -> b%10 in {1,2,3,4,6,7,8,9}; a%10 in {1,6} -> b%10 in {0,1,2,3,5,6,7,8}; a%10 in {2,7} -> b%10 in {0,1,4,5,6,9}; a%10 in {3,8} -> b%10 in {0,3,4,5,8,9}; a%10 in {4,9} -> b%10 in {0,2,3,4,5,7,8,9} |
| L18 down c24 | +0.142 | 57% | **units(a,b)** (R2 0.87): a%10 in {0,5} -> b%10 in {1,2,6,7}; a%10 in {1} -> b%10 in {0,1,5,6,9}; a%10 in {2,7} -> b%10 in {0,3,4,5,8,9}; a%10 in {3} -> b%10 in {0,2,3,4,7,8,9}; a%10 in {4,9} -> b%10 in {1,2,3,6,7,8}; a%10 in {6} -> b%10 in {0,1,4,5,6,9}; a%10 in {8} -> b%10 in {2,3,4,7,8,9} |
| L18 down c142 | +0.068 | 54% | **units(a,b)** (R2 0.88): a%10 in {0,5} -> b%10 in {0,3,4,5,9}; a%10 in {1,6} -> b%10 in {2,3,4,7,8,9}; a%10 in {2,7} -> b%10 in {1,2,3,6,7,8}; a%10 in {3,8} -> b%10 in {1,2,6,7}; a%10 in {4} -> b%10 in {0,1,5,6}; a%10 in {9} -> b%10 in {0,1,4,5,6} |
| L18 down c101 | +0.062 | 42% | **units(a,b)** (R2 0.85): a%10 in {1,6} -> b%10 in {4,8,9}; a%10 in {2,7} -> b%10 in {2,3,4,7,8,9}; a%10 in {3,8} -> b%10 in {1,2,3,4,6,7,8,9}; a%10 in {4,9} -> b%10 in {1,2,6,7,8} |
| L18 down c80 | +0.057 | 51% | **units(a,b)** (R2 0.82): a%10 in {0,5} -> b%10 in {0,1,5,6}; a%10 in {1,6} -> b%10 in {0,4,5,9}; a%10 in {2} -> b%10 in {3,4,7,8,9}; a%10 in {3,8} -> b%10 in {2,3,4,7,8,9}; a%10 in {4} -> b%10 in {1,2,6,7}; a%10 in {7} -> b%10 in {0,2,3,4,7,8,9}; a%10 in {9} -> b%10 in {1,2,6,7,8} |
| L16 down c76 | +0.046 | 26% | **units(a,b)** (R2 0.92): a%10 in {2,7} -> b%10 in {0,1,5,6}; a%10 in {3,8} -> b%10 in {0,1,4,5,6,9}; a%10 in {4} -> b%10 in {0,4,5,9}; a%10 in {9} -> b%10 in {0,4,5} |
| L16 down c127 | +0.044 | 22% | **units(a,b)** (R2 0.90): a%10 in {0,5} -> b%10 in {4,9}; a%10 in {1} -> b%10 in {0,3,4,5,8,9}; a%10 in {2,6,7} -> b%10 in {3,4,8,9} |
| L18 down c375 | +0.044 | 37% | **units(a,b)** (R2 0.87): a%10 in {0,5} -> b%10 in {2,3,4,7,8,9}; a%10 in {1,6} -> b%10 in {2,3,7,8}; a%10 in {3,8} -> b%10 in {0,5}; a%10 in {4,9} -> b%10 in {0,3,4,5,8,9} |
| L16 down c94 | +0.037 | 21% | **units(a,b)** (R2 0.89): a%10 in {1,2,6,7} -> b%10 in {0,4,5,9}; a%10 in {3,8} -> b%10 in {4,9} |
| L19 down c143 | +0.033 | 22% | **units(a,b)** (R2 0.62): a%10 in {1,6} -> b%10 in {3,7,8,9}; a%10 in {2} -> b%10 in {2,7,8}; a%10 in {3,8} -> b%10 in {1,6,7}; a%10 in {7} -> b%10 in {2,3,7,8} |
| L19 down c55 | +0.030 | 23% | **res%10** (R2 0.81): res mod 10 in {0..1} |
| L17 down c116 | +0.029 | 25% | **units(a,b)** (R2 0.90): a%10 in {0} -> b%10 in {1,2,7}; a%10 in {1,6} -> b%10 in {1,2,6,7}; a%10 in {2,7} -> b%10 in {0,1,2,5,6,7}; a%10 in {5} -> b%10 in {2,7} |
| L16 down c79 | +0.025 | 33% | **units(a,b)** (R2 0.93): a%10 in {0,1,5,6} -> b%10 in {0,1,4,5,6,9}; a%10 in {4,9} -> b%10 in {0,1,5,6} |
| L19 down c149 | +0.024 | 27% | **res%100** (R2 0.77): res mod 100 in {2..4, 12..14, 22..24, 32..34, 42..44, 52..53, 62..63, 72..73, 82..83, 92..93} [coarser: res mod 10 in {2..4}, R2 0.97] |
| L16 down c275 | +0.021 | 21% | **units(a,b)** (R2 0.89): a%10 in {3,8} -> b%10 in {0,1,2,5,6,7}; a%10 in {4,9} -> b%10 in {0,1,5,6} |

</details>

<details><summary>period 4: 8 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L19 down c6 | +0.179 | 66% | unexplained (best R2 0.26) |
| L20 o c328 (H2) | +0.117 | 5% | unexplained (best R2 0.19) |
| L18 down c18 | +0.107 | 35% | **b%20** (R2 0.60): b mod 20 in {0..4, 17..19} |
| L18 down c16 | +0.079 | 37% | **b%20** (R2 0.59): b mod 20 in {0..4, 15..19} |
| L19 down c85 | +0.059 | 6% | unexplained (best R2 0.15) |
| L20 o c292 (H2) | +0.043 | 3% | unexplained (best R2 0.16) |
| L18 o c210 (H18) | +0.026 | 4% | unexplained (best R2 0.23) |
| L18 down c59 | -0.042 | 17% | **a%20** (R2 0.52): a mod 20 in {15..17} |

</details>

<details><summary>period 2: 4 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L18 down c4 | +0.504 | 41% | **units(a,b)** (R2 0.74): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8}; a%10 in {1,5} -> b%10 in {3,5,7}; a%10 in {3,7,9} -> b%10 in {1,3,5,7,9} |
| L19 down c32 | +0.124 | 26% | **units(a,b)** (R2 0.82): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} |
| L17 down c67 | +0.123 | 25% | **units(a,b)** (R2 0.98): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} |
| L17 down c22 | +0.109 | 25% | **units(a,b)** (R2 0.98): a%10 in {1,3,5,7,9} -> b%10 in {0,2,4,6,8} |

</details>

</details>

<details><summary>`mlp_in.20`, sub: who writes the a−b code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.92 | L18 MLP +0.19, L19 MLP +0.18, L16 MLP +0.16, L17 MLP +0.12, L15 MLP +0.08, L15 attn +0.04, L14 MLP +0.03, L13 attn +0.03, L13 MLP +0.02 |
| mod 50 | 0.91 | L19 MLP +0.39, L18 MLP +0.35, L17 MLP +0.05, L16 MLP +0.04, L15 MLP +0.02 |
| mod 25 | 0.71 | L19 MLP +0.21, L18 MLP +0.18, L17 MLP +0.09, L16 MLP +0.07, L15 MLP +0.06, L15 attn +0.03, L13 attn +0.02 |
| mod 20 | 0.83 | L19 MLP +0.40, L18 MLP +0.30, L17 MLP +0.03, L16 MLP +0.03, L15 MLP +0.02 |
| mod 10 | 0.90 | L18 MLP +0.41, L17 MLP +0.26, L19 MLP +0.19 |
| mod 5 | 0.82 | L18 MLP +0.46, L16 MLP +0.21, L19 MLP +0.09, L17 MLP +0.03 |
| mod 4 | 0.55 | L18 MLP +0.13, L19 MLP +0.11, L20 attn +0.07, L16 MLP +0.05, L15 MLP +0.05, L17 MLP +0.05, L13 attn +0.03, L15 attn +0.02 |
| mod 2 | 0.76 | L18 MLP +0.53, L17 MLP +0.18, L19 MLP +0.03 |

<details><summary>period 100: 10 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L19 down c8 | +0.070 | 87% | unexplained (best R2 0.46) |
| L18 down c23 | +0.052 | 35% | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {0,9,10}; a//10 in {1} -> b//10 in {0,1,10}; a//10 in {2} -> b//10 in {0,1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3,4,6}; a//10 in {5} -> b//10 in {3,4,5,6}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9}; a//10 in {9} -> b//10 in {0,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,8,9,10} |
| L15 down c10 | +0.044 | 93% | unexplained (best R2 0.34) |
| L16 down c45 | +0.037 | 98% | always |
| L16 down c10 | +0.034 | 47% | **tens(a,b)** (R2 0.73): a//10 in {3} -> b//10 in {2,6}; a//10 in {4} -> b//10 in {3,6,7}; a//10 in {5} -> b//10 in {3,4,6,7}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,8,9}; a//10 in {7,8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {1,2,3,4,5,6,7,8,9} |
| L19 down c11 | +0.026 | 44% | **tens(a,b)** (R2 0.61): a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {0,1,2,3,6,7,8,10}; a//10 in {3} -> b//10 in {1,2,3,7,8}; a//10 in {4} -> b//10 in {2,3,7,8}; a//10 in {5} -> b//10 in {3,4,8}; a//10 in {6} -> b//10 in {4,5,6,8,9}; a//10 in {7} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {1,2,3,6,7,8,9}; a//10 in {9} -> b//10 in {7,8,9}; a//10 in {10} -> b//10 in {3,4,8,9} |
| L15 down c90 | +0.024 | 63% | **tens(a,b)** (R2 0.72): a//10 in {1} -> b//10 in {1,2,3,4}; a//10 in {2} -> b//10 in {1,2,3,4,5,6,7,9}; a//10 in {3} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {4,5,6,7,8,9,10} -> b//10 in {3,4,5,6,7,8,9,10} |
| L14 down c72 | +0.024 | 69% | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,3,9,10}; a//10 in {6} -> b//10 in {0,1,2,3,4,10}; a//10 in {7,8} -> b//10 in {0,1,2,3,4,5}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,8}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,7,8,10} |
| L17 down c13 | +0.022 | 40% | **tens(a,b)** (R2 0.61): a//10 in {2} -> b//10 in {3,4,6}; a//10 in {3} -> b//10 in {3,4,5,6,7}; a//10 in {4} -> b//10 in {4,5,6,7,8}; a//10 in {5} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {6,10} -> b//10 in {4,5,6,7,8,9}; a//10 in {7} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {5,6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10} |
| L17 down c89 | +0.021 | 19% | **res//10** (R2 0.53): (tens) res in {0..22, 99} |

</details>

<details><summary>period 50: 10 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L19 down c0 | +0.187 | 60% | **tens(a,b)** (R2 0.54): a//10 in {0,5} -> b//10 in {2,3,7,8}; a//10 in {1} -> b//10 in {3,4,7,8,9}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {1,2,6,7,8}; a//10 in {6} -> b//10 in {2,3,4,7,8}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,10}; a//10 in {9} -> b//10 in {1,2,3,6,7}; a//10 in {10} -> b//10 in {1,2,3,7,8} |
| L19 down c11 | +0.110 | 44% | **tens(a,b)** (R2 0.61): a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {0,1,2,3,6,7,8,10}; a//10 in {3} -> b//10 in {1,2,3,7,8}; a//10 in {4} -> b//10 in {2,3,7,8}; a//10 in {5} -> b//10 in {3,4,8}; a//10 in {6} -> b//10 in {4,5,6,8,9}; a//10 in {7} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {1,2,3,6,7,8,9}; a//10 in {9} -> b//10 in {7,8,9}; a//10 in {10} -> b//10 in {3,4,8,9} |
| L18 down c12 | +0.091 | 59% | **tens(a,b)** (R2 0.51): a//10 in {0} -> b//10 in {0,1,4,5,9,10}; a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,8,9}; a//10 in {3} -> b//10 in {0,1,2,6,7,8}; a//10 in {4} -> b//10 in {0,1,2,3,4,6,7,8,9,10}; a//10 in {5,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {0,3,4,5,6,8,9,10}; a//10 in {7} -> b//10 in {0,4,5,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,5,6,7,8,10}; a//10 in {9} -> b//10 in {0,1,2,3,5,6,7,8,9,10} |
| L19 down c20 | +0.043 | 25% | **tens(a,b)** (R2 0.56): a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {2,3}; a//10 in {5} -> b//10 in {3,4,8}; a//10 in {6} -> b//10 in {0,4,5,9,10}; a//10 in {7} -> b//10 in {0,5,6}; a//10 in {8} -> b//10 in {0,6,7}; a//10 in {9} -> b//10 in {2,7,8}; a//10 in {10} -> b//10 in {2,3,4,8,9} |
| L18 down c22 | +0.041 | 37% | **tens(a,b)** (R2 0.64): a//10 in {2} -> b//10 in {10}; a//10 in {3} -> b//10 in {0,1,5,6,9,10}; a//10 in {4} -> b//10 in {0,1,2,5,6,7,9,10}; a//10 in {5} -> b//10 in {0,1,2,6,7}; a//10 in {6} -> b//10 in {3}; a//10 in {7} -> b//10 in {3,4,5,10}; a//10 in {8} -> b//10 in {0,4,5,6,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,10}; a//10 in {10} -> b//10 in {0,1,2,5,6,7,10} |
| L18 down c25 | +0.040 | 32% | **tens(a,b)** (R2 0.54): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {0,1}; a//10 in {2} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {0,5,10}; a//10 in {5} -> b//10 in {0,1,5,6,9,10}; a//10 in {6} -> b//10 in {0,1,2,5,6,7,9,10}; a//10 in {7} -> b//10 in {1,2,3,6,7}; a//10 in {8} -> b//10 in {4}; a//10 in {9} -> b//10 in {0,4,5,9}; a//10 in {10} -> b//10 in {0,1,4,5,6,9,10} |
| L18 down c21 | +0.039 | 36% | **tens(a,b)** (R2 0.57): a//10 in {0} -> b//10 in {1}; a//10 in {1} -> b//10 in {1,2,6,7}; a//10 in {2} -> b//10 in {2}; a//10 in {3} -> b//10 in {6}; a//10 in {4,9} -> b//10 in {1,6,7}; a//10 in {5} -> b//10 in {1,2,3,6,7}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,8,10}; a//10 in {7} -> b//10 in {3,4,7,8,10}; a//10 in {8} -> b//10 in {5}; a//10 in {10} -> b//10 in {0,1,2,5,6,7} |
| L18 down c36 | +0.037 | 23% | **tens(a,b)** (R2 0.55): a//10 in {3} -> b//10 in {2,3,8}; a//10 in {4} -> b//10 in {3,8}; a//10 in {5} -> b//10 in {0,4,5,8,9,10}; a//10 in {6} -> b//10 in {0,9,10}; a//10 in {8} -> b//10 in {3,4,7,8}; a//10 in {9} -> b//10 in {3,4,7,8,9}; a//10 in {10} -> b//10 in {0,3,4,5,7,8,9,10} |
| L18 down c27 | +0.029 | 23% | **tens(a,b)** (R2 0.62): a//10 in {2} -> b//10 in {0,1,10}; a//10 in {3} -> b//10 in {0,1,2,6,7}; a//10 in {4} -> b//10 in {1,2,7}; a//10 in {7} -> b//10 in {0,5,6,10}; a//10 in {8} -> b//10 in {0,1,2,5,6,7,10}; a//10 in {9,10} -> b//10 in {1,2,6,7} |
| L18 down c23 | +0.027 | 35% | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {0,9,10}; a//10 in {1} -> b//10 in {0,1,10}; a//10 in {2} -> b//10 in {0,1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3,4,6}; a//10 in {5} -> b//10 in {3,4,5,6}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9}; a//10 in {9} -> b//10 in {0,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,8,9,10} |

</details>

<details><summary>period 25: 4 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L19 down c11 | +0.056 | 44% | **tens(a,b)** (R2 0.61): a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {0,1,2,3,6,7,8,10}; a//10 in {3} -> b//10 in {1,2,3,7,8}; a//10 in {4} -> b//10 in {2,3,7,8}; a//10 in {5} -> b//10 in {3,4,8}; a//10 in {6} -> b//10 in {4,5,6,8,9}; a//10 in {7} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {1,2,3,6,7,8,9}; a//10 in {9} -> b//10 in {7,8,9}; a//10 in {10} -> b//10 in {3,4,8,9} |
| L19 down c0 | +0.043 | 60% | **tens(a,b)** (R2 0.54): a//10 in {0,5} -> b//10 in {2,3,7,8}; a//10 in {1} -> b//10 in {3,4,7,8,9}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {1,2,6,7,8}; a//10 in {6} -> b//10 in {2,3,4,7,8}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,10}; a//10 in {9} -> b//10 in {1,2,3,6,7}; a//10 in {10} -> b//10 in {1,2,3,7,8} |
| L19 down c36 | +0.022 | 17% | unexplained (best R2 0.45) |
| L19 down c8 | +0.021 | 87% | unexplained (best R2 0.46) |

</details>

<details><summary>period 20: 3 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L19 down c6 | +0.344 | 66% | unexplained (best R2 0.21) |
| L18 down c18 | +0.138 | 28% | unexplained (best R2 0.38) |
| L18 down c16 | +0.106 | 21% | unexplained (best R2 0.27) |

</details>

<details><summary>period 10: 14 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L18 down c32 | +0.102 | 30% | unexplained (best R2 0.50) |
| L18 down c62 | +0.093 | 28% | **units(a,b)** (R2 0.68): a%10 in {0} -> b%10 in {0,1,2,8,9}; a%10 in {1} -> b%10 in {0,1,2,3,8,9}; a%10 in {2} -> b%10 in {0,1,2,9}; a%10 in {7} -> b%10 in {8,9}; a%10 in {8} -> b%10 in {0,8,9}; a%10 in {9} -> b%10 in {0,1,7,8,9} |
| L19 down c41 | +0.066 | 27% | **units(a,b)** (R2 0.55): a%10 in {0} -> b%10 in {1,2,3,4}; a%10 in {1} -> b%10 in {2,3,4}; a%10 in {6} -> b%10 in {0,8,9}; a%10 in {7} -> b%10 in {0,1,2,8,9}; a%10 in {8} -> b%10 in {0,1,2,9}; a%10 in {9} -> b%10 in {0,1,2,3} |
| L17 down c34 | +0.066 | 21% | **units(a,b)** (R2 0.55): a%10 in {0} -> b%10 in {4}; a%10 in {5,6} -> b%10 in {0,1,2,3,4,9}; a%10 in {7} -> b%10 in {0,1,2,3,4}; a%10 in {8} -> b%10 in {2,3,4}; a%10 in {9} -> b%10 in {3,4} |
| L19 down c55 | +0.064 | 15% | **res%100** (R2 0.48): res mod 100 in {0..1, 10..11, 20, 30, 81, 90..91} |
| L18 down c98 | +0.063 | 27% | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {2,3,4,5,6}; a%10 in {1} -> b%10 in {1,3,4,5,6}; a%10 in {2} -> b%10 in {0,1,2,4,5,6}; a%10 in {3} -> b%10 in {0,1,2,3}; a%10 in {4} -> b%10 in {1,2,3}; a%10 in {8} -> b%10 in {2,3,4}; a%10 in {9} -> b%10 in {2,3,4,5} |
| L18 down c65 | +0.061 | 29% | unexplained (best R2 0.49) |
| L17 down c39 | +0.057 | 24% | **units(a,b)** (R2 0.54): a%10 in {0,3,4,8,9} -> b%10 in {1,2,3,4}; a%10 in {1} -> b%10 in {2,3,4}; a%10 in {5} -> b%10 in {3,4}; a%10 in {7} -> b%10 in {1,2,3} |
| L17 down c49 | +0.055 | 27% | **a** (R2 0.50): a in {1, 26..27, 36..37, 46..47, 55..58, 65..68, 75..78, 85..88, 95..97} [coarser: a mod 20 in {6..7, 15..17}, R2 0.80] |
| L18 down c88 | +0.040 | 19% | unexplained (best R2 0.46) |
| L17 down c86 | +0.039 | 16% | unexplained (best R2 0.48) |
| L19 down c149 | +0.035 | 11% | **res** (R2 0.65): res in {2..3, 12..13, 22..24, 32..33, 42..43, 82..83, 92..93} |
| L18 down c120 | +0.029 | 16% | **units(a,b)** (R2 0.51): a%10 in {0} -> b%10 in {7,8}; a%10 in {5} -> b%10 in {4,5}; a%10 in {6} -> b%10 in {4,5,6}; a%10 in {7} -> b%10 in {5,6,7}; a%10 in {8} -> b%10 in {5,6,7,8}; a%10 in {9} -> b%10 in {6,7,8} |
| L17 down c52 | +0.028 | 17% | unexplained (best R2 0.49) |

</details>

<details><summary>period 5: 12 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L18 down c26 | +0.123 | 42% | **res** (R2 0.51): res in {-97, -48, -43..-42, -38..-37, -33, -28..-27, -23..-22, -18..-17, -13..-12, -8..-7, -3..-1, 1..3, 6..8, 11..13, 16..18, 21..23, 26..28, 31..33, 37..38, 42..43, 47..48, 52..53, 57..58, 62..63, 67..68, 72..73, 77..78, 82..83, 87..88, 92..93, 96..98} |
| L18 down c24 | +0.111 | 51% | unexplained (best R2 0.49) |
| L16 down c127 | +0.070 | 18% | **units(a,b)** (R2 0.69): a%10 in {0,5} -> b%10 in {1,6}; a%10 in {1} -> b%10 in {0,1,2,5,6,7}; a%10 in {2} -> b%10 in {1,2,7}; a%10 in {6,7} -> b%10 in {1,2,6,7} |
| L18 down c80 | +0.066 | 24% | unexplained (best R2 0.46) |
| L18 down c101 | +0.052 | 24% | unexplained (best R2 0.45) |
| L18 down c375 | +0.047 | 26% | unexplained (best R2 0.46) |
| L18 down c142 | +0.042 | 21% | **res** (R2 0.52): res in {0, 3..5, 8..10, 13..15, 19..20, 23..25, 29..30, 34..35, 39, 44..45, 59, 69, 74..75, 79, 84..85, 89..90, 94..95, 99} |
| L16 down c94 | +0.037 | 22% | unexplained (best R2 0.47) |
| L16 down c79 | +0.036 | 28% | **units(a,b)** (R2 0.76): a%10 in {0,1,5,6} -> b%10 in {0,1,4,5,6,9}; a%10 in {4,9} -> b%10 in {0,4,5,9} |
| L19 down c55 | +0.029 | 15% | **res%100** (R2 0.48): res mod 100 in {0..1, 10..11, 20, 30, 81, 90..91} |
| L16 down c76 | +0.029 | 15% | **units(a,b)** (R2 0.52): a%10 in {2} -> b%10 in {0,4,5}; a%10 in {3} -> b%10 in {0,1,4,5,6}; a%10 in {7,8} -> b%10 in {0,5} |
| L16 down c275 | +0.029 | 14% | **units(a,b)** (R2 0.58): a%10 in {3} -> b%10 in {0,3,4,8,9}; a%10 in {4,9} -> b%10 in {0,4,5,9}; a%10 in {8} -> b%10 in {3,4,8,9} |

</details>

<details><summary>period 4: 4 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 o c328 (H2) | +0.045 | 19% | unexplained (best R2 0.40) |
| L18 down c24 | +0.033 | 51% | unexplained (best R2 0.49) |
| L15 down c18 | +0.022 | 11% | unexplained (best R2 0.31) |
| L19 down c74 | +0.021 | 5% | **res** (R2 0.57): res in {8..9, 18..19, 28..29, 98..99} |

</details>

<details><summary>period 2: 4 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L18 down c4 | +0.527 | 40% | **units(a,b)** (R2 0.52): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} |
| L17 down c22 | +0.108 | 25% | **units(a,b)** (R2 0.81): a%10 in {1,3,5,7,9} -> b%10 in {0,2,4,6,8} |
| L17 down c67 | +0.072 | 20% | **units(a,b)** (R2 0.73): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} |
| L19 down c32 | +0.026 | 11% | unexplained (best R2 0.31) |

</details>

</details>

<details><summary>`mlp_in.21`, add: who writes the a+b code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.98 | L20 MLP +0.33, L19 MLP +0.25, L16 MLP +0.17, L18 MLP +0.13, L17 MLP +0.09 |
| mod 50 | 1.03 | L18 MLP +0.41, L19 MLP +0.36, L20 MLP +0.23, L17 MLP +0.03 |
| mod 25 | 0.63 | L20 MLP +0.30, L19 MLP +0.30, L18 MLP +0.02 |
| mod 20 | 1.00 | L20 MLP +0.70, L18 MLP +0.16, L19 MLP +0.14 |
| mod 10 | 1.01 | L20 MLP +0.38, L18 MLP +0.26, L19 MLP +0.19, L17 MLP +0.18 |
| mod 5 | 1.03 | L20 MLP +0.54, L18 MLP +0.32, L16 MLP +0.09, L19 MLP +0.05, L17 MLP +0.03 |
| mod 4 | 0.77 | L20 MLP +0.58, L19 MLP +0.09, L20 attn +0.07 |
| mod 2 | 0.93 | L18 MLP +0.45, L17 MLP +0.21, L20 MLP +0.17, L19 MLP +0.09 |

<details><summary>period 100: 12 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c2 | +0.118 | 100% | always |
| L19 down c24 | +0.097 | 44% | **res//10** (R2 0.77): (tens) res in {24, 26, 28..80, 142..174} [coarser: res mod 100 in {34..76}, R2 0.81] |
| L19 down c28 | +0.096 | 57% | **res//10** (R2 0.79): (tens) res in {14..52, 100..162, 164} [coarser: res mod 100 in {1, 10..56}, R2 0.81] |
| L20 down c20 | +0.082 | 54% | **res//10** (R2 0.81): (tens) res in {2..50, 99..150, 198..200} [coarser: res mod 100 in {0..50, 98..99}, R2 0.99] |
| L18 down c23 | +0.082 | 52% | **res//10** (R2 0.71): (tens) res in {2..29, 31, 83..130, 179..200} [coarser: res mod 100 in {0..31, 81..99}, R2 0.99] |
| L16 down c10 | +0.057 | 48% | **tens(a,b)** (R2 0.80): a//10 in {0} -> b//10 in {2,3,5,6,7,8}; a//10 in {1} -> b//10 in {6}; a//10 in {5} -> b//10 in {3,4,5,7,8,9,10}; a//10 in {6,10} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {8,9} -> b//10 in {1,2,3,4,5,6,7,8,9,10} |
| L17 down c11 | +0.050 | 46% | **tens(a,b)** (R2 0.79): a//10 in {2} -> b//10 in {6,7}; a//10 in {3} -> b//10 in {6,7,9,10}; a//10 in {4,6} -> b//10 in {0,1,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {0,5,6,7,8,9,10}; a//10 in {9,10} -> b//10 in {4,5,6,7,8,9,10} |
| L16 down c29 | +0.042 | 50% | **tens(a,b)** (R2 0.80): a//10 in {0,9,10} -> b//10 in {0,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {8}; a//10 in {5} -> b//10 in {5,6,7}; a//10 in {6} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {2,3,4,5,6,7,8,9,10} |
| L20 down c31 | +0.040 | 29% | **res//10** (R2 0.79): (tens) res in {2..13, 89..117, 188..200} [coarser: res mod 100 in {0..15, 89..99}, R2 0.98] |
| L17 down c40 | +0.027 | 47% | **tens(a,b)** (R2 0.78): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7}; a//10 in {1,2} -> b//10 in {2,3,4,5,6,7}; a//10 in {3,4} -> b//10 in {1,2,3,4,5,6}; a//10 in {5} -> b//10 in {1,2,3,4,5}; a//10 in {6} -> b//10 in {1,2,3,4}; a//10 in {8} -> b//10 in {2}; a//10 in {9,10} -> b//10 in {0,1,2,3,4} |
| L20 down c5 | +0.025 | 71% | **tens(a,b)** (R2 0.74): a//10 in {0,1,2} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,10}; a//10 in {4,5} -> b//10 in {0,1,2,3,4,5,6}; a//10 in {6} -> b//10 in {0,1,2,3,4}; a//10 in {7} -> b//10 in {0,1,2,3}; a//10 in {8} -> b//10 in {0,1,2,9}; a//10 in {9} -> b//10 in {0,1,2,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,6,8,9,10} |
| L19 down c86 | +0.021 | 25% | **res//10** (R2 0.92): (tens) res in {132..200} |

</details>

<details><summary>period 50: 14 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L19 down c0 | +0.182 | 51% | **res%50** (R2 0.84): res mod 50 in {14..39} |
| L18 down c12 | +0.114 | 78% | unexplained (best R2 0.46) |
| L20 down c29 | +0.105 | 56% | **res%100** (R2 0.82): res mod 100 in {11..29, 55..86} |
| L19 down c11 | +0.085 | 46% | **tens(a,b)** (R2 0.71): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {4,8,9,10}; a//10 in {2,7} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {3,8} -> b//10 in {1,2,3,6,7,8}; a//10 in {4} -> b//10 in {1,2,5,6,7}; a//10 in {5} -> b//10 in {4,5,6}; a//10 in {6} -> b//10 in {3,4,5,8,9,10}; a//10 in {9} -> b//10 in {1,2,6,7,10}; a//10 in {10} -> b//10 in {0,1,2,5,6,7,10} |
| L18 down c21 | +0.077 | 70% | **tens(a,b)** (R2 0.57): a//10 in {0} -> b//10 in {1,2,3,4,6,7,8,9}; a//10 in {1,6} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,1,2,5,6,7,10}; a//10 in {3} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {4} -> b//10 in {2,3,4,8,9,10}; a//10 in {5} -> b//10 in {1,2,3,4,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,4,5,6,7,9,10}; a//10 in {8} -> b//10 in {0,1,4,5,9,10}; a//10 in {9} -> b//10 in {3,4,8,9,10}; a//10 in {10} -> b//10 in {2,3,4,7,8,9} |
| L19 down c20 | +0.068 | 60% | **res%50** (R2 0.68): res mod 50 in {4..31} |
| L20 down c14 | +0.063 | 65% | **res** (R2 0.66): res in {39, 73..104, 114, 118..165, 167..200} |
| L18 down c22 | +0.059 | 65% | **res%50** (R2 0.65): res mod 50 in {0, 18..49} |
| L18 down c25 | +0.059 | 78% | **tens(a,b)** (R2 0.52): a//10 in {0} -> b//10 in {3,4,5,7,8,9,10}; a//10 in {1} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {2} -> b//10 in {1,2,3,4,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,2,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,4,5,6,8,9,10}; a//10 in {5} -> b//10 in {0,3,4,5,7,8,9,10}; a//10 in {6} -> b//10 in {0,2,3,4,7,8,9,10}; a//10 in {7,8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {0,1,3,4,5,6,8,9,10}; a//10 in {10} -> b//10 in {0,3,4,5,6,7,8,9,10} |
| L18 down c27 | +0.033 | 33% | **tens(a,b)** (R2 0.69): a//10 in {2} -> b//10 in {4,8,9,10}; a//10 in {3} -> b//10 in {3,4,7,8,9,10}; a//10 in {4} -> b//10 in {2,3,7,8,9}; a//10 in {5} -> b//10 in {8}; a//10 in {7} -> b//10 in {3,4,5,8,9,10}; a//10 in {8} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {9} -> b//10 in {2,3,4,7,8}; a//10 in {10} -> b//10 in {2,3,7,8} |
| L18 down c36 | +0.033 | 58% | **tens(a,b)** (R2 0.54): a//10 in {0} -> b//10 in {4,5,6,10}; a//10 in {1} -> b//10 in {4,5,10}; a//10 in {2,3} -> b//10 in {1,2,3,6,7}; a//10 in {4,9} -> b//10 in {0,1,2,5,6,7,10}; a//10 in {5} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {6} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {7,8} -> b//10 in {1,2,3,5,6,7,8}; a//10 in {10} -> b//10 in {0,1,2,4,5,6,7,10} |
| L17 down c23 | +0.028 | 65% | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {1,4,8,9}; a//10 in {1} -> b//10 in {1,3,4,5,6,8,9,10}; a//10 in {2,3,6} -> b//10 in {1,2,3,4,5,6,8,9,10}; a//10 in {4} -> b//10 in {1,3,8,9}; a//10 in {5} -> b//10 in {1,3,4,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,8,9,10}; a//10 in {9} -> b//10 in {1,3,4,6,8}; a//10 in {10} -> b//10 in {3,4,8,9} |
| L20 down c2 | +0.028 | 100% | always |
| L18 down c61 | +0.025 | 33% | **tens(a,b)** (R2 0.65): a//10 in {0} -> b//10 in {0,1,4,5,6,7,10}; a//10 in {1,6} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {2} -> b//10 in {0,4,5,10}; a//10 in {4} -> b//10 in {0,1,6}; a//10 in {5,10} -> b//10 in {0,1,5,6,10}; a//10 in {7} -> b//10 in {0,9,10}; a//10 in {9} -> b//10 in {1,6} |

</details>

<details><summary>period 25: 6 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L19 down c0 | +0.212 | 51% | **res%50** (R2 0.84): res mod 50 in {14..39} |
| L20 down c14 | +0.147 | 65% | **res** (R2 0.66): res in {39, 73..104, 114, 118..165, 167..200} |
| L19 down c11 | +0.062 | 46% | **tens(a,b)** (R2 0.71): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {4,8,9,10}; a//10 in {2,7} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {3,8} -> b//10 in {1,2,3,6,7,8}; a//10 in {4} -> b//10 in {1,2,5,6,7}; a//10 in {5} -> b//10 in {4,5,6}; a//10 in {6} -> b//10 in {3,4,5,8,9,10}; a//10 in {9} -> b//10 in {1,2,6,7,10}; a//10 in {10} -> b//10 in {0,1,2,5,6,7,10} |
| L20 down c31 | +0.061 | 29% | **res//10** (R2 0.79): (tens) res in {2..13, 89..117, 188..200} [coarser: res mod 100 in {0..15, 89..99}, R2 0.98] |
| L20 down c180 | +0.040 | 11% | **res** (R2 0.65): res in {121..137} |
| L20 down c223 | +0.024 | 5% | **res** (R2 0.73): res in {113..118} |

</details>

<details><summary>period 20: 6 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c3 | +0.347 | 61% | **res%20** (R2 0.74): res mod 20 in {8..19} |
| L20 down c6 | +0.310 | 60% | **res%20** (R2 0.60): res mod 20 in {1..11} |
| L19 down c6 | +0.132 | 66% | unexplained (best R2 0.26) |
| L18 down c18 | +0.075 | 35% | **b%20** (R2 0.60): b mod 20 in {0..4, 17..19} |
| L18 down c16 | +0.056 | 37% | **b%20** (R2 0.59): b mod 20 in {0..4, 15..19} |
| L18 down c59 | +0.025 | 17% | **a%20** (R2 0.52): a mod 20 in {15..17} |

</details>

<details><summary>period 10: 16 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c21 | +0.170 | 42% | **res%10** (R2 0.89): res mod 10 in {0, 7..9} |
| L20 down c13 | +0.167 | 42% | **res%10** (R2 0.86): res mod 10 in {5..8} |
| L18 down c32 | +0.087 | 46% | **units(a,b)** (R2 0.85): a%10 in {0} -> b%10 in {3,4,5,6}; a%10 in {1,3,9} -> b%10 in {3,4,5,6,7}; a%10 in {2} -> b%10 in {5,6,7}; a%10 in {4,5} -> b%10 in {2,3,4,5,6,7}; a%10 in {6} -> b%10 in {1,2,3,4,5,6}; a%10 in {7} -> b%10 in {1,2,3,4}; a%10 in {8} -> b%10 in {4,5,6} |
| L19 down c41 | +0.084 | 40% | **units(a,b)** (R2 0.85): a%10 in {0} -> b%10 in {6,7,8,9}; a%10 in {1} -> b%10 in {5,6,7,8}; a%10 in {2} -> b%10 in {5,6,7}; a%10 in {3} -> b%10 in {5,6}; a%10 in {5} -> b%10 in {1,2,3,4}; a%10 in {6} -> b%10 in {0,1,2,3,9}; a%10 in {7} -> b%10 in {0,1,2,8,9}; a%10 in {8} -> b%10 in {0,1,7,8,9}; a%10 in {9} -> b%10 in {0,6,7,8,9} |
| L17 down c34 | +0.048 | 35% | **units(a,b)** (R2 0.93): a%10 in {0} -> b%10 in {5,6,7}; a%10 in {1} -> b%10 in {5,6}; a%10 in {5} -> b%10 in {0,1,2,5,6,7,8,9}; a%10 in {6} -> b%10 in {0,1,5,6,7,8,9}; a%10 in {7} -> b%10 in {0,5,6,7,8,9}; a%10 in {8} -> b%10 in {5,6,7,8,9}; a%10 in {9} -> b%10 in {5,6,7,8} |
| L18 down c88 | +0.039 | 30% | **units(a,b)** (R2 0.89): a%10 in {1} -> b%10 in {4,5,6}; a%10 in {2} -> b%10 in {3,4,5,6}; a%10 in {3} -> b%10 in {2,3,4,5,6}; a%10 in {4} -> b%10 in {1,2,3,4,5}; a%10 in {5} -> b%10 in {1,2,3,4,5,6,7}; a%10 in {6} -> b%10 in {1,5,6,7}; a%10 in {7} -> b%10 in {5,6} |
| L19 down c149 | +0.039 | 27% | **res%100** (R2 0.77): res mod 100 in {2..4, 12..14, 22..24, 32..34, 42..44, 52..53, 62..63, 72..73, 82..83, 92..93} [coarser: res mod 10 in {2..4}, R2 0.97] |
| L17 down c49 | +0.039 | 38% | **units(a,b)** (R2 0.90): a%10 in {5} -> b%10 in {0,1,2,3,5,6,7,9}; a%10 in {6} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {7} -> b%10 in {0,1,2,3,5,6,7,8,9}; a%10 in {8} -> b%10 in {0,1,2,5,6,8,9}; a%10 in {9} -> b%10 in {0,1,5,9} |
| L19 down c55 | +0.037 | 23% | **res%10** (R2 0.81): res mod 10 in {0..1} |
| L17 down c39 | +0.037 | 39% | **units(a,b)** (R2 0.87): a%10 in {0,4} -> b%10 in {5,6,7,8,9}; a%10 in {1} -> b%10 in {5,6,7,8}; a%10 in {2} -> b%10 in {7,8}; a%10 in {3} -> b%10 in {0,6,7,8,9}; a%10 in {5} -> b%10 in {6,7,8}; a%10 in {7} -> b%10 in {6,7,8,9}; a%10 in {8,9} -> b%10 in {0,5,6,7,8,9} |
| L18 down c65 | +0.035 | 36% | **units(a,b)** (R2 0.88): a%10 in {0,5} -> b%10 in {7,8,9}; a%10 in {1,6} -> b%10 in {6,7,8,9}; a%10 in {2,7} -> b%10 in {5,6,7,8}; a%10 in {3} -> b%10 in {4,5,6,7}; a%10 in {4} -> b%10 in {4,5,6}; a%10 in {8} -> b%10 in {5,6,7}; a%10 in {9} -> b%10 in {5,6,9} |
| L18 down c62 | +0.035 | 25% | **units(a,b)** (R2 0.89): a%10 in {0} -> b%10 in {0,1,2,8,9}; a%10 in {1} -> b%10 in {0,1,7,8,9}; a%10 in {2} -> b%10 in {0,7,8,9}; a%10 in {7} -> b%10 in {1,2}; a%10 in {8} -> b%10 in {0,1,2}; a%10 in {9} -> b%10 in {0,1,2,9} |
| L18 down c98 | +0.032 | 34% | **units(a,b)** (R2 0.81): a%10 in {0,1} -> b%10 in {4,5,6,7}; a%10 in {2} -> b%10 in {0,4,5,6,8,9}; a%10 in {3} -> b%10 in {0,4,5,8,9}; a%10 in {4} -> b%10 in {7,8}; a%10 in {7} -> b%10 in {8,9}; a%10 in {8} -> b%10 in {6,7,8,9}; a%10 in {9} -> b%10 in {5,6,7,8} |
| L17 down c86 | +0.027 | 31% | **units(a,b)** (R2 0.85): a%10 in {0} -> b%10 in {6,7,8}; a%10 in {1} -> b%10 in {5,6,7,8,9}; a%10 in {2} -> b%10 in {5,6,7,8}; a%10 in {3} -> b%10 in {0,1,2,6,7,8,9}; a%10 in {4} -> b%10 in {0,1,2,8,9}; a%10 in {5} -> b%10 in {0,1,8,9}; a%10 in {6} -> b%10 in {0,8,9} |
| L20 down c100 | +0.027 | 71% | **res%100** (R2 0.54): res mod 100 in {1..5, 8..9, 13..15, 18..25, 29, 33..35, 38..45, 48..49, 52..65, 68..69, 73..75, 78..85, 88..89, 93..95, 97..99} |
| L17 down c52 | +0.024 | 26% | **units(a,b)** (R2 0.87): a%10 in {0} -> b%10 in {2,3,4}; a%10 in {1} -> b%10 in {1,2,3,4}; a%10 in {2} -> b%10 in {0,1,2,3,4}; a%10 in {3,4} -> b%10 in {0,1,2,3,4,9} |

</details>

<details><summary>period 5: 12 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c9 | +0.249 | 60% | **res%5** (R2 0.95): res mod 5 in {1..3} |
| L20 down c15 | +0.184 | 61% | **res%5** (R2 0.83): res mod 5 in {0..2} |
| L18 down c26 | +0.099 | 69% | **units(a,b)** (R2 0.80): a%10 in {0,5} -> b%10 in {1,2,3,4,6,7,8,9}; a%10 in {1,6} -> b%10 in {0,1,2,3,5,6,7,8}; a%10 in {2,7} -> b%10 in {0,1,4,5,6,9}; a%10 in {3,8} -> b%10 in {0,3,4,5,8,9}; a%10 in {4,9} -> b%10 in {0,2,3,4,5,7,8,9} |
| L18 down c24 | +0.082 | 57% | **units(a,b)** (R2 0.87): a%10 in {0,5} -> b%10 in {1,2,6,7}; a%10 in {1} -> b%10 in {0,1,5,6,9}; a%10 in {2,7} -> b%10 in {0,3,4,5,8,9}; a%10 in {3} -> b%10 in {0,2,3,4,7,8,9}; a%10 in {4,9} -> b%10 in {1,2,3,6,7,8}; a%10 in {6} -> b%10 in {0,1,4,5,6,9}; a%10 in {8} -> b%10 in {2,3,4,7,8,9} |
| L20 down c100 | +0.059 | 71% | **res%100** (R2 0.54): res mod 100 in {1..5, 8..9, 13..15, 18..25, 29, 33..35, 38..45, 48..49, 52..65, 68..69, 73..75, 78..85, 88..89, 93..95, 97..99} |
| L18 down c142 | +0.038 | 54% | **units(a,b)** (R2 0.88): a%10 in {0,5} -> b%10 in {0,3,4,5,9}; a%10 in {1,6} -> b%10 in {2,3,4,7,8,9}; a%10 in {2,7} -> b%10 in {1,2,3,6,7,8}; a%10 in {3,8} -> b%10 in {1,2,6,7}; a%10 in {4} -> b%10 in {0,1,5,6}; a%10 in {9} -> b%10 in {0,1,4,5,6} |
| L18 down c101 | +0.034 | 42% | **units(a,b)** (R2 0.85): a%10 in {1,6} -> b%10 in {4,8,9}; a%10 in {2,7} -> b%10 in {2,3,4,7,8,9}; a%10 in {3,8} -> b%10 in {1,2,3,4,6,7,8,9}; a%10 in {4,9} -> b%10 in {1,2,6,7,8} |
| L18 down c80 | +0.032 | 51% | **units(a,b)** (R2 0.82): a%10 in {0,5} -> b%10 in {0,1,5,6}; a%10 in {1,6} -> b%10 in {0,4,5,9}; a%10 in {2} -> b%10 in {3,4,7,8,9}; a%10 in {3,8} -> b%10 in {2,3,4,7,8,9}; a%10 in {4} -> b%10 in {1,2,6,7}; a%10 in {7} -> b%10 in {0,2,3,4,7,8,9}; a%10 in {9} -> b%10 in {1,2,6,7,8} |
| L20 down c21 | +0.027 | 42% | **res%10** (R2 0.89): res mod 10 in {0, 7..9} |
| L16 down c76 | +0.025 | 26% | **units(a,b)** (R2 0.92): a%10 in {2,7} -> b%10 in {0,1,5,6}; a%10 in {3,8} -> b%10 in {0,1,4,5,6,9}; a%10 in {4} -> b%10 in {0,4,5,9}; a%10 in {9} -> b%10 in {0,4,5} |
| L18 down c375 | +0.025 | 37% | **units(a,b)** (R2 0.87): a%10 in {0,5} -> b%10 in {2,3,4,7,8,9}; a%10 in {1,6} -> b%10 in {2,3,7,8}; a%10 in {3,8} -> b%10 in {0,5}; a%10 in {4,9} -> b%10 in {0,3,4,5,8,9} |
| L16 down c127 | +0.023 | 22% | **units(a,b)** (R2 0.90): a%10 in {0,5} -> b%10 in {4,9}; a%10 in {1} -> b%10 in {0,3,4,5,8,9}; a%10 in {2,6,7} -> b%10 in {3,4,8,9} |

</details>

<details><summary>period 4: 9 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c44 | +0.215 | 7% | unexplained (best R2 0.28) |
| L20 down c6 | +0.150 | 60% | **res%20** (R2 0.60): res mod 20 in {1..11} |
| L20 down c3 | +0.138 | 61% | **res%20** (R2 0.74): res mod 20 in {8..19} |
| L19 down c6 | +0.073 | 66% | unexplained (best R2 0.26) |
| L20 o c328 (H2) | +0.050 | 5% | unexplained (best R2 0.19) |
| L18 down c16 | +0.044 | 37% | **b%20** (R2 0.59): b mod 20 in {0..4, 15..19} |
| L20 down c15 | +0.042 | 61% | **res%5** (R2 0.83): res mod 5 in {0..2} |
| L20 down c9 | +0.036 | 60% | **res%5** (R2 0.95): res mod 5 in {1..3} |
| L18 down c59 | -0.045 | 17% | **a%20** (R2 0.52): a mod 20 in {15..17} |

</details>

<details><summary>period 2: 5 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L18 down c4 | +0.451 | 41% | **units(a,b)** (R2 0.74): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8}; a%10 in {1,5} -> b%10 in {3,5,7}; a%10 in {3,7,9} -> b%10 in {1,3,5,7,9} |
| L20 down c47 | +0.165 | 43% | **res%2** (R2 0.74): res mod 2 in {0} |
| L17 down c67 | +0.107 | 25% | **units(a,b)** (R2 0.98): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} |
| L17 down c22 | +0.102 | 25% | **units(a,b)** (R2 0.98): a%10 in {1,3,5,7,9} -> b%10 in {0,2,4,6,8} |
| L19 down c32 | +0.085 | 26% | **units(a,b)** (R2 0.82): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} |

</details>

</details>

<details><summary>`mlp_in.21`, sub: who writes the a−b code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.95 | L20 MLP +0.18, L18 MLP +0.16, L19 MLP +0.15, L16 MLP +0.14, L17 MLP +0.10, L15 MLP +0.07, L15 attn +0.03, L14 MLP +0.02, L13 attn +0.02 |
| mod 50 | 0.92 | L18 MLP +0.29, L19 MLP +0.29, L20 MLP +0.21, L17 MLP +0.04, L16 MLP +0.03 |
| mod 25 | 0.80 | L20 MLP +0.29, L18 MLP +0.16, L19 MLP +0.14, L17 MLP +0.07, L16 MLP +0.05, L15 MLP +0.04, L15 attn +0.02 |
| mod 20 | 0.93 | L20 MLP +0.54, L19 MLP +0.18, L18 MLP +0.17 |
| mod 10 | 0.95 | L20 MLP +0.36, L18 MLP +0.28, L17 MLP +0.17, L19 MLP +0.11 |
| mod 5 | 0.92 | L20 MLP +0.44, L18 MLP +0.29, L16 MLP +0.12, L19 MLP +0.04 |
| mod 4 | 0.70 | L20 MLP +0.25, L18 MLP +0.11, L19 MLP +0.07, L20 attn +0.07, L16 MLP +0.05, L15 MLP +0.04, L17 MLP +0.03, L13 attn +0.02 |
| mod 2 | 0.80 | L18 MLP +0.46, L17 MLP +0.16, L20 MLP +0.14 |

<details><summary>period 100: 10 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L19 down c8 | +0.057 | 87% | unexplained (best R2 0.46) |
| L18 down c23 | +0.045 | 35% | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {0,9,10}; a//10 in {1} -> b//10 in {0,1,10}; a//10 in {2} -> b//10 in {0,1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3,4,6}; a//10 in {5} -> b//10 in {3,4,5,6}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9}; a//10 in {9} -> b//10 in {0,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,8,9,10} |
| L20 down c20 | +0.037 | 48% | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {0,1,2,3}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {0,1,2,3,9,10}; a//10 in {3} -> b//10 in {0,1,2,3,4,6,10}; a//10 in {4} -> b//10 in {0,1,2,3,4,6}; a//10 in {5} -> b//10 in {1,2,3,4,5}; a//10 in {6} -> b//10 in {2,3,4,5,6}; a//10 in {7} -> b//10 in {3,4,5,6,9}; a//10 in {8} -> b//10 in {4,5,6,7,8,9}; a//10 in {9} -> b//10 in {6,7,8,9,10}; a//10 in {10} -> b//10 in {0,5,6,7,8,9} |
| L15 down c10 | +0.037 | 93% | unexplained (best R2 0.34) |
| L16 down c45 | +0.031 | 98% | always |
| L16 down c10 | +0.030 | 47% | **tens(a,b)** (R2 0.73): a//10 in {3} -> b//10 in {2,6}; a//10 in {4} -> b//10 in {3,6,7}; a//10 in {5} -> b//10 in {3,4,6,7}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,8,9}; a//10 in {7,8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {1,2,3,4,5,6,7,8,9} |
| L20 down c31 | +0.029 | 25% | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {0,1,9,10}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {4,5}; a//10 in {6} -> b//10 in {5,6,7,8}; a//10 in {7} -> b//10 in {6,7,8,9}; a//10 in {8} -> b//10 in {7,8,9,10}; a//10 in {9,10} -> b//10 in {0,8,9,10} |
| L20 down c2 | +0.029 | 100% | always |
| L19 down c11 | +0.021 | 44% | **tens(a,b)** (R2 0.61): a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {0,1,2,3,6,7,8,10}; a//10 in {3} -> b//10 in {1,2,3,7,8}; a//10 in {4} -> b//10 in {2,3,7,8}; a//10 in {5} -> b//10 in {3,4,8}; a//10 in {6} -> b//10 in {4,5,6,8,9}; a//10 in {7} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {1,2,3,6,7,8,9}; a//10 in {9} -> b//10 in {7,8,9}; a//10 in {10} -> b//10 in {3,4,8,9} |
| L14 down c72 | +0.021 | 69% | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,3,9,10}; a//10 in {6} -> b//10 in {0,1,2,3,4,10}; a//10 in {7,8} -> b//10 in {0,1,2,3,4,5}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,8}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,7,8,10} |

</details>

<details><summary>period 50: 13 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L19 down c0 | +0.135 | 60% | **tens(a,b)** (R2 0.54): a//10 in {0,5} -> b//10 in {2,3,7,8}; a//10 in {1} -> b//10 in {3,4,7,8,9}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {1,2,6,7,8}; a//10 in {6} -> b//10 in {2,3,4,7,8}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,10}; a//10 in {9} -> b//10 in {1,2,3,6,7}; a//10 in {10} -> b//10 in {1,2,3,7,8} |
| L20 down c29 | +0.088 | 53% | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {2,6,7,8}; a//10 in {1} -> b//10 in {7,8,9,10}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {1,2,6,7,8}; a//10 in {4,9} -> b//10 in {1,2,3,6,7,8}; a//10 in {5} -> b//10 in {2,3,4,7,8,9}; a//10 in {6} -> b//10 in {0,4,8,9,10}; a//10 in {7} -> b//10 in {0,1,5,6,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,5,6,7,8,10}; a//10 in {10} -> b//10 in {2,3,4,7,8} |
| L19 down c11 | +0.084 | 44% | **tens(a,b)** (R2 0.61): a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {0,1,2,3,6,7,8,10}; a//10 in {3} -> b//10 in {1,2,3,7,8}; a//10 in {4} -> b//10 in {2,3,7,8}; a//10 in {5} -> b//10 in {3,4,8}; a//10 in {6} -> b//10 in {4,5,6,8,9}; a//10 in {7} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {1,2,3,6,7,8,9}; a//10 in {9} -> b//10 in {7,8,9}; a//10 in {10} -> b//10 in {3,4,8,9} |
| L18 down c12 | +0.079 | 59% | **tens(a,b)** (R2 0.51): a//10 in {0} -> b//10 in {0,1,4,5,9,10}; a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,8,9}; a//10 in {3} -> b//10 in {0,1,2,6,7,8}; a//10 in {4} -> b//10 in {0,1,2,3,4,6,7,8,9,10}; a//10 in {5,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {0,3,4,5,6,8,9,10}; a//10 in {7} -> b//10 in {0,4,5,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,5,6,7,8,10}; a//10 in {9} -> b//10 in {0,1,2,3,5,6,7,8,9,10} |
| L19 down c20 | +0.036 | 25% | **tens(a,b)** (R2 0.56): a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {2,3}; a//10 in {5} -> b//10 in {3,4,8}; a//10 in {6} -> b//10 in {0,4,5,9,10}; a//10 in {7} -> b//10 in {0,5,6}; a//10 in {8} -> b//10 in {0,6,7}; a//10 in {9} -> b//10 in {2,7,8}; a//10 in {10} -> b//10 in {2,3,4,8,9} |
| L18 down c21 | +0.034 | 36% | **tens(a,b)** (R2 0.57): a//10 in {0} -> b//10 in {1}; a//10 in {1} -> b//10 in {1,2,6,7}; a//10 in {2} -> b//10 in {2}; a//10 in {3} -> b//10 in {6}; a//10 in {4,9} -> b//10 in {1,6,7}; a//10 in {5} -> b//10 in {1,2,3,6,7}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,8,10}; a//10 in {7} -> b//10 in {3,4,7,8,10}; a//10 in {8} -> b//10 in {5}; a//10 in {10} -> b//10 in {0,1,2,5,6,7} |
| L18 down c22 | +0.032 | 37% | **tens(a,b)** (R2 0.64): a//10 in {2} -> b//10 in {10}; a//10 in {3} -> b//10 in {0,1,5,6,9,10}; a//10 in {4} -> b//10 in {0,1,2,5,6,7,9,10}; a//10 in {5} -> b//10 in {0,1,2,6,7}; a//10 in {6} -> b//10 in {3}; a//10 in {7} -> b//10 in {3,4,5,10}; a//10 in {8} -> b//10 in {0,4,5,6,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,10}; a//10 in {10} -> b//10 in {0,1,2,5,6,7,10} |
| L18 down c25 | +0.031 | 32% | **tens(a,b)** (R2 0.54): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {0,1}; a//10 in {2} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {0,5,10}; a//10 in {5} -> b//10 in {0,1,5,6,9,10}; a//10 in {6} -> b//10 in {0,1,2,5,6,7,9,10}; a//10 in {7} -> b//10 in {1,2,3,6,7}; a//10 in {8} -> b//10 in {4}; a//10 in {9} -> b//10 in {0,4,5,9}; a//10 in {10} -> b//10 in {0,1,4,5,6,9,10} |
| L20 down c31 | +0.030 | 25% | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {0,1,9,10}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {4,5}; a//10 in {6} -> b//10 in {5,6,7,8}; a//10 in {7} -> b//10 in {6,7,8,9}; a//10 in {8} -> b//10 in {7,8,9,10}; a//10 in {9,10} -> b//10 in {0,8,9,10} |
| L18 down c36 | +0.027 | 23% | **tens(a,b)** (R2 0.55): a//10 in {3} -> b//10 in {2,3,8}; a//10 in {4} -> b//10 in {3,8}; a//10 in {5} -> b//10 in {0,4,5,8,9,10}; a//10 in {6} -> b//10 in {0,9,10}; a//10 in {8} -> b//10 in {3,4,7,8}; a//10 in {9} -> b//10 in {3,4,7,8,9}; a//10 in {10} -> b//10 in {0,3,4,5,7,8,9,10} |
| L20 down c2 | +0.025 | 100% | always |
| L18 down c27 | +0.024 | 23% | **tens(a,b)** (R2 0.62): a//10 in {2} -> b//10 in {0,1,10}; a//10 in {3} -> b//10 in {0,1,2,6,7}; a//10 in {4} -> b//10 in {1,2,7}; a//10 in {7} -> b//10 in {0,5,6,10}; a//10 in {8} -> b//10 in {0,1,2,5,6,7,10}; a//10 in {9,10} -> b//10 in {1,2,6,7} |
| L18 down c23 | +0.021 | 35% | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {0,9,10}; a//10 in {1} -> b//10 in {0,1,10}; a//10 in {2} -> b//10 in {0,1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3,4,6}; a//10 in {5} -> b//10 in {3,4,5,6}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9}; a//10 in {9} -> b//10 in {0,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,8,9,10} |

</details>

<details><summary>period 25: 5 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c3 | +0.088 | 53% | unexplained (best R2 0.43) |
| L20 down c6 | +0.065 | 49% | unexplained (best R2 0.33) |
| L20 down c31 | +0.044 | 25% | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {0,1,9,10}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {4,5}; a//10 in {6} -> b//10 in {5,6,7,8}; a//10 in {7} -> b//10 in {6,7,8,9}; a//10 in {8} -> b//10 in {7,8,9,10}; a//10 in {9,10} -> b//10 in {0,8,9,10} |
| L19 down c11 | +0.034 | 44% | **tens(a,b)** (R2 0.61): a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {0,1,2,3,6,7,8,10}; a//10 in {3} -> b//10 in {1,2,3,7,8}; a//10 in {4} -> b//10 in {2,3,7,8}; a//10 in {5} -> b//10 in {3,4,8}; a//10 in {6} -> b//10 in {4,5,6,8,9}; a//10 in {7} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {1,2,3,6,7,8,9}; a//10 in {9} -> b//10 in {7,8,9}; a//10 in {10} -> b//10 in {3,4,8,9} |
| L19 down c0 | +0.023 | 60% | **tens(a,b)** (R2 0.54): a//10 in {0,5} -> b//10 in {2,3,7,8}; a//10 in {1} -> b//10 in {3,4,7,8,9}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {1,2,6,7,8}; a//10 in {6} -> b//10 in {2,3,4,7,8}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,10}; a//10 in {9} -> b//10 in {1,2,3,6,7}; a//10 in {10} -> b//10 in {1,2,3,7,8} |

</details>

<details><summary>period 20: 5 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c3 | +0.238 | 53% | unexplained (best R2 0.43) |
| L20 down c6 | +0.234 | 49% | unexplained (best R2 0.33) |
| L19 down c6 | +0.160 | 66% | unexplained (best R2 0.21) |
| L18 down c18 | +0.082 | 28% | unexplained (best R2 0.38) |
| L18 down c16 | +0.055 | 21% | unexplained (best R2 0.27) |

</details>

<details><summary>period 10: 15 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c13 | +0.173 | 39% | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {2,3,4,5}; a%10 in {1} -> b%10 in {3,4,5,6}; a%10 in {2} -> b%10 in {4,5,6,7}; a%10 in {3} -> b%10 in {5,6,7}; a%10 in {4} -> b%10 in {6,7,8,9}; a%10 in {5} -> b%10 in {0,8,9}; a%10 in {6} -> b%10 in {0,1,8,9}; a%10 in {7} -> b%10 in {0,1,2,3,9}; a%10 in {8} -> b%10 in {0,1,2,3,4}; a%10 in {9} -> b%10 in {1,2,3,4,5} |
| L20 down c21 | +0.156 | 37% | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {0,1,2,3,9}; a%10 in {1} -> b%10 in {0,1,2,3}; a%10 in {2} -> b%10 in {1,2,3,4}; a%10 in {3} -> b%10 in {3,4,5,6}; a%10 in {4} -> b%10 in {4,5,6}; a%10 in {5} -> b%10 in {6,7}; a%10 in {6} -> b%10 in {6,7,8}; a%10 in {7} -> b%10 in {0,7,8,9}; a%10 in {8} -> b%10 in {0,1,8,9}; a%10 in {9} -> b%10 in {0,1,2,8,9} |
| L18 down c32 | +0.071 | 30% | unexplained (best R2 0.50) |
| L18 down c62 | +0.060 | 28% | **units(a,b)** (R2 0.68): a%10 in {0} -> b%10 in {0,1,2,8,9}; a%10 in {1} -> b%10 in {0,1,2,3,8,9}; a%10 in {2} -> b%10 in {0,1,2,9}; a%10 in {7} -> b%10 in {8,9}; a%10 in {8} -> b%10 in {0,8,9}; a%10 in {9} -> b%10 in {0,1,7,8,9} |
| L18 down c98 | +0.045 | 27% | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {2,3,4,5,6}; a%10 in {1} -> b%10 in {1,3,4,5,6}; a%10 in {2} -> b%10 in {0,1,2,4,5,6}; a%10 in {3} -> b%10 in {0,1,2,3}; a%10 in {4} -> b%10 in {1,2,3}; a%10 in {8} -> b%10 in {2,3,4}; a%10 in {9} -> b%10 in {2,3,4,5} |
| L19 down c41 | +0.044 | 27% | **units(a,b)** (R2 0.55): a%10 in {0} -> b%10 in {1,2,3,4}; a%10 in {1} -> b%10 in {2,3,4}; a%10 in {6} -> b%10 in {0,8,9}; a%10 in {7} -> b%10 in {0,1,2,8,9}; a%10 in {8} -> b%10 in {0,1,2,9}; a%10 in {9} -> b%10 in {0,1,2,3} |
| L17 down c34 | +0.043 | 21% | **units(a,b)** (R2 0.55): a%10 in {0} -> b%10 in {4}; a%10 in {5,6} -> b%10 in {0,1,2,3,4,9}; a%10 in {7} -> b%10 in {0,1,2,3,4}; a%10 in {8} -> b%10 in {2,3,4}; a%10 in {9} -> b%10 in {3,4} |
| L18 down c65 | +0.042 | 29% | unexplained (best R2 0.49) |
| L17 down c39 | +0.037 | 24% | **units(a,b)** (R2 0.54): a%10 in {0,3,4,8,9} -> b%10 in {1,2,3,4}; a%10 in {1} -> b%10 in {2,3,4}; a%10 in {5} -> b%10 in {3,4}; a%10 in {7} -> b%10 in {1,2,3} |
| L17 down c49 | +0.035 | 27% | **a** (R2 0.50): a in {1, 26..27, 36..37, 46..47, 55..58, 65..68, 75..78, 85..88, 95..97} [coarser: a mod 20 in {6..7, 15..17}, R2 0.80] |
| L19 down c55 | +0.033 | 15% | **res%100** (R2 0.48): res mod 100 in {0..1, 10..11, 20, 30, 81, 90..91} |
| L18 down c88 | +0.027 | 19% | unexplained (best R2 0.46) |
| L17 down c86 | +0.026 | 16% | unexplained (best R2 0.48) |
| L20 down c100 | +0.022 | 20% | **res** (R2 0.50): res in {3..5, 9, 13..15, 19, 23..25, 33..34, 43..45, 54..55, 59, 63..65, 74..75, 79, 83..85, 89, 93..95, 98..99} |
| L17 down c52 | +0.020 | 17% | unexplained (best R2 0.49) |

</details>

<details><summary>period 5: 12 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c9 | +0.224 | 49% | **res** (R2 0.62): res in {-97, -93, -63, -58, -53..-52, -48..-47, -43..-42, -38..-37, -33..-32, -28..-27, -23..-22, -19..-17, -13..-12, -9..-7, -4..-2, 1..3, 6..8, 11..13, 16..18, 21..23, 26..28, 31..33, 36..38, 41..43, 46..48, 51..53, 56..58, 61..63, 66..68, 71..73, 76..78, 81..83, 86..88, 91..93, 96..98} |
| L20 down c15 | +0.144 | 46% | unexplained (best R2 0.44) |
| L18 down c26 | +0.083 | 42% | **res** (R2 0.51): res in {-97, -48, -43..-42, -38..-37, -33, -28..-27, -23..-22, -18..-17, -13..-12, -8..-7, -3..-1, 1..3, 6..8, 11..13, 16..18, 21..23, 26..28, 31..33, 37..38, 42..43, 47..48, 52..53, 57..58, 62..63, 67..68, 72..73, 77..78, 82..83, 87..88, 92..93, 96..98} |
| L18 down c24 | +0.071 | 51% | unexplained (best R2 0.49) |
| L16 down c127 | +0.043 | 18% | **units(a,b)** (R2 0.69): a%10 in {0,5} -> b%10 in {1,6}; a%10 in {1} -> b%10 in {0,1,2,5,6,7}; a%10 in {2} -> b%10 in {1,2,7}; a%10 in {6,7} -> b%10 in {1,2,6,7} |
| L18 down c80 | +0.041 | 24% | unexplained (best R2 0.46) |
| L18 down c101 | +0.032 | 24% | unexplained (best R2 0.45) |
| L20 down c100 | +0.030 | 20% | **res** (R2 0.50): res in {3..5, 9, 13..15, 19, 23..25, 33..34, 43..45, 54..55, 59, 63..65, 74..75, 79, 83..85, 89, 93..95, 98..99} |
| L18 down c375 | +0.029 | 26% | unexplained (best R2 0.46) |
| L18 down c142 | +0.026 | 21% | **res** (R2 0.52): res in {0, 3..5, 8..10, 13..15, 19..20, 23..25, 29..30, 34..35, 39, 44..45, 59, 69, 74..75, 79, 84..85, 89..90, 94..95, 99} |
| L20 down c21 | +0.022 | 37% | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {0,1,2,3,9}; a%10 in {1} -> b%10 in {0,1,2,3}; a%10 in {2} -> b%10 in {1,2,3,4}; a%10 in {3} -> b%10 in {3,4,5,6}; a%10 in {4} -> b%10 in {4,5,6}; a%10 in {5} -> b%10 in {6,7}; a%10 in {6} -> b%10 in {6,7,8}; a%10 in {7} -> b%10 in {0,7,8,9}; a%10 in {8} -> b%10 in {0,1,8,9}; a%10 in {9} -> b%10 in {0,1,2,8,9} |
| L16 down c94 | +0.021 | 22% | unexplained (best R2 0.47) |

</details>

<details><summary>period 4: 5 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c44 | +0.127 | 12% | unexplained (best R2 0.32) |
| L20 o c328 (H2) | +0.045 | 19% | unexplained (best R2 0.40) |
| L20 down c3 | +0.043 | 53% | unexplained (best R2 0.43) |
| L18 down c24 | +0.035 | 51% | unexplained (best R2 0.49) |
| L20 down c15 | +0.021 | 46% | unexplained (best R2 0.44) |

</details>

<details><summary>period 2: 4 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L18 down c4 | +0.458 | 40% | **units(a,b)** (R2 0.52): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} |
| L20 down c47 | +0.134 | 28% | **units(a,b)** (R2 0.75): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} |
| L17 down c22 | +0.098 | 25% | **units(a,b)** (R2 0.81): a%10 in {1,3,5,7,9} -> b%10 in {0,2,4,6,8} |
| L17 down c67 | +0.061 | 20% | **units(a,b)** (R2 0.73): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} |

</details>

</details>

<details><summary>`mlp_in.23`, add: who writes the a+b code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.97 | L21 MLP +0.23, L22 MLP +0.18, L20 MLP +0.17, L19 MLP +0.14, L16 MLP +0.10, L18 MLP +0.08, L17 MLP +0.05 |
| mod 50 | 1.02 | L18 MLP +0.30, L19 MLP +0.26, L21 MLP +0.18, L20 MLP +0.14, L22 MLP +0.12 |
| mod 25 | 0.78 | L22 MLP +0.51, L21 MLP +0.21, L20 MLP +0.05 |
| mod 20 | 1.01 | L20 MLP +0.41, L21 MLP +0.23, L22 MLP +0.18, L18 MLP +0.10, L19 MLP +0.08 |
| mod 10 | 0.98 | L21 MLP +0.34, L20 MLP +0.17, L22 MLP +0.15, L18 MLP +0.13, L19 MLP +0.09, L17 MLP +0.09 |
| mod 5 | 1.02 | L20 MLP +0.32, L18 MLP +0.21, L21 MLP +0.19, L22 MLP +0.18, L16 MLP +0.06, L19 MLP +0.03 |
| mod 4 | 0.70 | L22 MLP +0.59, L21 MLP +0.05, L20 MLP +0.03 |
| mod 2 | 0.97 | L21 MLP +0.35, L18 MLP +0.27, L17 MLP +0.13, L20 MLP +0.11, L22 MLP +0.05, L19 MLP +0.05 |

<details><summary>period 100: 14 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L21 down c0 | +0.116 | 33% | **res//10** (R2 0.84): (tens) res in {8..40, 108..140} [coarser: res mod 100 in {8..40}, R2 1.00] |
| L20 down c2 | +0.066 | 100% | always |
| L19 down c24 | +0.057 | 44% | **res//10** (R2 0.77): (tens) res in {24, 26, 28..80, 142..174} [coarser: res mod 100 in {34..76}, R2 0.81] |
| L19 down c28 | +0.054 | 57% | **res//10** (R2 0.79): (tens) res in {14..52, 100..162, 164} [coarser: res mod 100 in {1, 10..56}, R2 0.81] |
| L18 down c23 | +0.049 | 52% | **res//10** (R2 0.71): (tens) res in {2..29, 31, 83..130, 179..200} [coarser: res mod 100 in {0..31, 81..99}, R2 0.99] |
| L20 down c20 | +0.046 | 54% | **res//10** (R2 0.81): (tens) res in {2..50, 99..150, 198..200} [coarser: res mod 100 in {0..50, 98..99}, R2 0.99] |
| L22 down c5 | +0.045 | 16% | **res%100** (R2 0.76): res mod 100 in {39..51} |
| L22 down c6 | +0.037 | 16% | **res%100** (R2 0.88): res mod 100 in {0..1, 3..15} |
| L16 down c10 | +0.035 | 48% | **tens(a,b)** (R2 0.80): a//10 in {0} -> b//10 in {2,3,5,6,7,8}; a//10 in {1} -> b//10 in {6}; a//10 in {5} -> b//10 in {3,4,5,7,8,9,10}; a//10 in {6,10} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {8,9} -> b//10 in {1,2,3,4,5,6,7,8,9,10} |
| L17 down c11 | +0.028 | 46% | **tens(a,b)** (R2 0.79): a//10 in {2} -> b//10 in {6,7}; a//10 in {3} -> b//10 in {6,7,9,10}; a//10 in {4,6} -> b//10 in {0,1,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {0,5,6,7,8,9,10}; a//10 in {9,10} -> b//10 in {4,5,6,7,8,9,10} |
| L21 down c36 | +0.025 | 35% | **res//10** (R2 0.79): (tens) res in {60..91, 153..195, 197} [coarser: res mod 100 in {56..95}, R2 0.92] |
| L16 down c29 | +0.024 | 50% | **tens(a,b)** (R2 0.80): a//10 in {0,9,10} -> b//10 in {0,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {8}; a//10 in {5} -> b//10 in {5,6,7}; a//10 in {6} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {2,3,4,5,6,7,8,9,10} |
| L22 down c33 | +0.022 | 22% | **res%100** (R2 0.82): res mod 100 in {1, 86..99} |
| L20 down c31 | +0.020 | 29% | **res//10** (R2 0.79): (tens) res in {2..13, 89..117, 188..200} [coarser: res mod 100 in {0..15, 89..99}, R2 0.98] |

</details>

<details><summary>period 50: 15 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L19 down c0 | +0.132 | 51% | **res%50** (R2 0.84): res mod 50 in {14..39} |
| L21 down c7 | +0.088 | 42% | **res%50** (R2 0.81): res mod 50 in {0..12, 42..49} |
| L18 down c12 | +0.079 | 78% | unexplained (best R2 0.46) |
| L20 down c29 | +0.067 | 56% | **res%100** (R2 0.82): res mod 100 in {11..29, 55..86} |
| L19 down c11 | +0.064 | 46% | **tens(a,b)** (R2 0.71): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {4,8,9,10}; a//10 in {2,7} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {3,8} -> b//10 in {1,2,3,6,7,8}; a//10 in {4} -> b//10 in {1,2,5,6,7}; a//10 in {5} -> b//10 in {4,5,6}; a//10 in {6} -> b//10 in {3,4,5,8,9,10}; a//10 in {9} -> b//10 in {1,2,6,7,10}; a//10 in {10} -> b//10 in {0,1,2,5,6,7,10} |
| L18 down c21 | +0.058 | 70% | **tens(a,b)** (R2 0.57): a//10 in {0} -> b//10 in {1,2,3,4,6,7,8,9}; a//10 in {1,6} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,1,2,5,6,7,10}; a//10 in {3} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {4} -> b//10 in {2,3,4,8,9,10}; a//10 in {5} -> b//10 in {1,2,3,4,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,4,5,6,7,9,10}; a//10 in {8} -> b//10 in {0,1,4,5,9,10}; a//10 in {9} -> b//10 in {3,4,8,9,10}; a//10 in {10} -> b//10 in {2,3,4,7,8,9} |
| L18 down c22 | +0.045 | 65% | **res%50** (R2 0.65): res mod 50 in {0, 18..49} |
| L19 down c20 | +0.044 | 60% | **res%50** (R2 0.68): res mod 50 in {4..31} |
| L18 down c25 | +0.044 | 78% | **tens(a,b)** (R2 0.52): a//10 in {0} -> b//10 in {3,4,5,7,8,9,10}; a//10 in {1} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {2} -> b//10 in {1,2,3,4,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,2,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,4,5,6,8,9,10}; a//10 in {5} -> b//10 in {0,3,4,5,7,8,9,10}; a//10 in {6} -> b//10 in {0,2,3,4,7,8,9,10}; a//10 in {7,8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {0,1,3,4,5,6,8,9,10}; a//10 in {10} -> b//10 in {0,3,4,5,6,7,8,9,10} |
| L20 down c14 | +0.040 | 65% | **res** (R2 0.66): res in {39, 73..104, 114, 118..165, 167..200} |
| L21 down c0 | +0.036 | 33% | **res//10** (R2 0.84): (tens) res in {8..40, 108..140} [coarser: res mod 100 in {8..40}, R2 1.00] |
| L22 down c5 | +0.036 | 16% | **res%100** (R2 0.76): res mod 100 in {39..51} |
| L18 down c36 | +0.026 | 58% | **tens(a,b)** (R2 0.54): a//10 in {0} -> b//10 in {4,5,6,10}; a//10 in {1} -> b//10 in {4,5,10}; a//10 in {2,3} -> b//10 in {1,2,3,6,7}; a//10 in {4,9} -> b//10 in {0,1,2,5,6,7,10}; a//10 in {5} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {6} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {7,8} -> b//10 in {1,2,3,5,6,7,8}; a//10 in {10} -> b//10 in {0,1,2,4,5,6,7,10} |
| L22 down c6 | +0.024 | 16% | **res%100** (R2 0.88): res mod 100 in {0..1, 3..15} |
| L18 down c27 | +0.024 | 33% | **tens(a,b)** (R2 0.69): a//10 in {2} -> b//10 in {4,8,9,10}; a//10 in {3} -> b//10 in {3,4,7,8,9,10}; a//10 in {4} -> b//10 in {2,3,7,8,9}; a//10 in {5} -> b//10 in {8}; a//10 in {7} -> b//10 in {3,4,5,8,9,10}; a//10 in {8} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {9} -> b//10 in {2,3,4,7,8}; a//10 in {10} -> b//10 in {2,3,7,8} |

</details>

<details><summary>period 25: 10 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L22 down c5 | +0.150 | 16% | **res%100** (R2 0.76): res mod 100 in {39..51} |
| L22 down c6 | +0.111 | 16% | **res%100** (R2 0.88): res mod 100 in {0..1, 3..15} |
| L21 down c7 | +0.083 | 42% | **res%50** (R2 0.81): res mod 50 in {0..12, 42..49} |
| L22 down c28 | +0.060 | 14% | **res%100** (R2 0.87): res mod 100 in {77..92} |
| L22 down c35 | +0.041 | 15% | **res%100** (R2 0.85): res mod 100 in {47..61} |
| L21 down c35 | +0.036 | 14% | **res%100** (R2 0.85): res mod 100 in {57..70} |
| L22 down c33 | +0.034 | 22% | **res%100** (R2 0.82): res mod 100 in {1, 86..99} |
| L22 down c27 | +0.033 | 12% | **res%100** (R2 0.81): res mod 100 in {27..38} |
| L22 down c98 | +0.025 | 8% | **res%100** (R2 0.70): res mod 100 in {54..60} |
| L20 down c14 | +0.021 | 65% | **res** (R2 0.66): res in {39, 73..104, 114, 118..165, 167..200} |

</details>

<details><summary>period 20: 9 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c3 | +0.216 | 61% | **res%20** (R2 0.74): res mod 20 in {8..19} |
| L20 down c6 | +0.169 | 60% | **res%20** (R2 0.60): res mod 20 in {1..11} |
| L21 down c2 | +0.141 | 36% | **res%20** (R2 0.89): res mod 20 in {10..16} |
| L22 down c11 | +0.105 | 33% | **res%20** (R2 0.89): res mod 20 in {0..3, 18..19} |
| L19 down c6 | +0.080 | 66% | unexplained (best R2 0.26) |
| L21 down c15 | +0.063 | 50% | **res%20** (R2 0.61): res mod 20 in {0, 4, 13..19} |
| L18 down c18 | +0.050 | 35% | **b%20** (R2 0.60): b mod 20 in {0..4, 17..19} |
| L18 down c16 | +0.032 | 37% | **b%20** (R2 0.59): b mod 20 in {0..4, 15..19} |
| L22 down c17 | +0.026 | 10% | **res%20** (R2 0.84): res mod 20 in {2..3} |

</details>

<details><summary>period 10: 15 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L21 down c4 | +0.111 | 41% | **res%10** (R2 0.85): res mod 10 in {0..3} |
| L20 down c21 | +0.074 | 42% | **res%10** (R2 0.89): res mod 10 in {0, 7..9} |
| L20 down c13 | +0.074 | 42% | **res%10** (R2 0.86): res mod 10 in {5..8} |
| L22 down c9 | +0.055 | 10% | **res%10** (R2 0.94): res mod 10 in {9} |
| L22 down c8 | +0.051 | 10% | **res%10** (R2 0.98): res mod 10 in {0} |
| L21 down c13 | +0.050 | 24% | **res%10** (R2 0.73): res mod 10 in {3..5} |
| L21 down c10 | +0.045 | 11% | **res%10** (R2 0.93): res mod 10 in {1} |
| L21 down c14 | +0.044 | 11% | **res%10** (R2 0.89): res mod 10 in {5} |
| L18 down c32 | +0.043 | 46% | **units(a,b)** (R2 0.85): a%10 in {0} -> b%10 in {3,4,5,6}; a%10 in {1,3,9} -> b%10 in {3,4,5,6,7}; a%10 in {2} -> b%10 in {5,6,7}; a%10 in {4,5} -> b%10 in {2,3,4,5,6,7}; a%10 in {6} -> b%10 in {1,2,3,4,5,6}; a%10 in {7} -> b%10 in {1,2,3,4}; a%10 in {8} -> b%10 in {4,5,6} |
| L19 down c41 | +0.042 | 40% | **units(a,b)** (R2 0.85): a%10 in {0} -> b%10 in {6,7,8,9}; a%10 in {1} -> b%10 in {5,6,7,8}; a%10 in {2} -> b%10 in {5,6,7}; a%10 in {3} -> b%10 in {5,6}; a%10 in {5} -> b%10 in {1,2,3,4}; a%10 in {6} -> b%10 in {0,1,2,3,9}; a%10 in {7} -> b%10 in {0,1,2,8,9}; a%10 in {8} -> b%10 in {0,1,7,8,9}; a%10 in {9} -> b%10 in {0,6,7,8,9} |
| L21 down c16 | +0.028 | 11% | **res%10** (R2 0.93): res mod 10 in {6} |
| L17 down c34 | +0.022 | 35% | **units(a,b)** (R2 0.93): a%10 in {0} -> b%10 in {5,6,7}; a%10 in {1} -> b%10 in {5,6}; a%10 in {5} -> b%10 in {0,1,2,5,6,7,8,9}; a%10 in {6} -> b%10 in {0,1,5,6,7,8,9}; a%10 in {7} -> b%10 in {0,5,6,7,8,9}; a%10 in {8} -> b%10 in {5,6,7,8,9}; a%10 in {9} -> b%10 in {5,6,7,8} |
| L18 down c88 | +0.021 | 30% | **units(a,b)** (R2 0.89): a%10 in {1} -> b%10 in {4,5,6}; a%10 in {2} -> b%10 in {3,4,5,6}; a%10 in {3} -> b%10 in {2,3,4,5,6}; a%10 in {4} -> b%10 in {1,2,3,4,5}; a%10 in {5} -> b%10 in {1,2,3,4,5,6,7}; a%10 in {6} -> b%10 in {1,5,6,7}; a%10 in {7} -> b%10 in {5,6} |
| L17 down c49 | +0.021 | 38% | **units(a,b)** (R2 0.90): a%10 in {5} -> b%10 in {0,1,2,3,5,6,7,9}; a%10 in {6} -> b%10 in {0,1,2,3,4,5,6,7,8,9}; a%10 in {7} -> b%10 in {0,1,2,3,5,6,7,8,9}; a%10 in {8} -> b%10 in {0,1,2,5,6,8,9}; a%10 in {9} -> b%10 in {0,1,5,9} |
| L19 down c149 | +0.020 | 27% | **res%100** (R2 0.77): res mod 100 in {2..4, 12..14, 22..24, 32..34, 42..44, 52..53, 62..63, 72..73, 82..83, 92..93} [coarser: res mod 10 in {2..4}, R2 0.97] |

</details>

<details><summary>period 5: 14 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c9 | +0.149 | 60% | **res%5** (R2 0.95): res mod 5 in {1..3} |
| L20 down c15 | +0.114 | 61% | **res%5** (R2 0.83): res mod 5 in {0..2} |
| L22 down c8 | +0.080 | 10% | **res%10** (R2 0.98): res mod 10 in {0} |
| L22 down c9 | +0.068 | 10% | **res%10** (R2 0.94): res mod 10 in {9} |
| L18 down c26 | +0.062 | 69% | **units(a,b)** (R2 0.80): a%10 in {0,5} -> b%10 in {1,2,3,4,6,7,8,9}; a%10 in {1,6} -> b%10 in {0,1,2,3,5,6,7,8}; a%10 in {2,7} -> b%10 in {0,1,4,5,6,9}; a%10 in {3,8} -> b%10 in {0,3,4,5,8,9}; a%10 in {4,9} -> b%10 in {0,2,3,4,5,7,8,9} |
| L18 down c24 | +0.054 | 57% | **units(a,b)** (R2 0.87): a%10 in {0,5} -> b%10 in {1,2,6,7}; a%10 in {1} -> b%10 in {0,1,5,6,9}; a%10 in {2,7} -> b%10 in {0,3,4,5,8,9}; a%10 in {3} -> b%10 in {0,2,3,4,7,8,9}; a%10 in {4,9} -> b%10 in {1,2,3,6,7,8}; a%10 in {6} -> b%10 in {0,1,4,5,6,9}; a%10 in {8} -> b%10 in {2,3,4,7,8,9} |
| L21 down c10 | +0.051 | 11% | **res%10** (R2 0.93): res mod 10 in {1} |
| L21 down c14 | +0.040 | 11% | **res%10** (R2 0.89): res mod 10 in {5} |
| L20 down c100 | +0.037 | 71% | **res%100** (R2 0.54): res mod 100 in {1..5, 8..9, 13..15, 18..25, 29, 33..35, 38..45, 48..49, 52..65, 68..69, 73..75, 78..85, 88..89, 93..95, 97..99} |
| L21 down c16 | +0.030 | 11% | **res%10** (R2 0.93): res mod 10 in {6} |
| L18 down c142 | +0.026 | 54% | **units(a,b)** (R2 0.88): a%10 in {0,5} -> b%10 in {0,3,4,5,9}; a%10 in {1,6} -> b%10 in {2,3,4,7,8,9}; a%10 in {2,7} -> b%10 in {1,2,3,6,7,8}; a%10 in {3,8} -> b%10 in {1,2,6,7}; a%10 in {4} -> b%10 in {0,1,5,6}; a%10 in {9} -> b%10 in {0,1,4,5,6} |
| L18 down c101 | +0.023 | 42% | **units(a,b)** (R2 0.85): a%10 in {1,6} -> b%10 in {4,8,9}; a%10 in {2,7} -> b%10 in {2,3,4,7,8,9}; a%10 in {3,8} -> b%10 in {1,2,3,4,6,7,8,9}; a%10 in {4,9} -> b%10 in {1,2,6,7,8} |
| L18 down c80 | +0.022 | 51% | **units(a,b)** (R2 0.82): a%10 in {0,5} -> b%10 in {0,1,5,6}; a%10 in {1,6} -> b%10 in {0,4,5,9}; a%10 in {2} -> b%10 in {3,4,7,8,9}; a%10 in {3,8} -> b%10 in {2,3,4,7,8,9}; a%10 in {4} -> b%10 in {1,2,6,7}; a%10 in {7} -> b%10 in {0,2,3,4,7,8,9}; a%10 in {9} -> b%10 in {1,2,6,7,8} |
| L21 down c53 | +0.020 | 10% | **res%10** (R2 0.96): res mod 10 in {9} |

</details>

<details><summary>period 4: 3 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L22 down c17 | +0.452 | 10% | **res%20** (R2 0.84): res mod 20 in {2..3} |
| L22 down c427 | +0.095 | 7% | **res%20** (R2 0.78): res mod 20 in {12} |
| L22 down c48 | +0.030 | 11% | **res%100** (R2 0.69): res mod 100 in {8, 18, 27..30, 38, 48, 68, 88} |

</details>

<details><summary>period 2: 7 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L21 down c1 | +0.305 | 50% | **res%2** (R2 0.99): res mod 2 in {1} |
| L18 down c4 | +0.268 | 41% | **units(a,b)** (R2 0.74): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8}; a%10 in {1,5} -> b%10 in {3,5,7}; a%10 in {3,7,9} -> b%10 in {1,3,5,7,9} |
| L20 down c47 | +0.107 | 43% | **res%2** (R2 0.74): res mod 2 in {0} |
| L17 down c22 | +0.065 | 25% | **units(a,b)** (R2 0.98): a%10 in {1,3,5,7,9} -> b%10 in {0,2,4,6,8} |
| L17 down c67 | +0.063 | 25% | **units(a,b)** (R2 0.98): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} |
| L19 down c32 | +0.048 | 26% | **units(a,b)** (R2 0.82): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} |
| L22 down c9 | +0.025 | 10% | **res%10** (R2 0.94): res mod 10 in {9} |

</details>

</details>

<details><summary>`mlp_in.23`, sub: who writes the a−b code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.98 | L21 MLP +0.22, L22 MLP +0.14, L20 MLP +0.12, L18 MLP +0.11, L19 MLP +0.10, L16 MLP +0.09, L17 MLP +0.06, L15 MLP +0.04 |
| mod 50 | 0.94 | L21 MLP +0.23, L18 MLP +0.20, L19 MLP +0.20, L20 MLP +0.12, L22 MLP +0.11, L17 MLP +0.03 |
| mod 25 | 0.88 | L22 MLP +0.22, L21 MLP +0.20, L20 MLP +0.16, L18 MLP +0.10, L19 MLP +0.07, L17 MLP +0.03, L16 MLP +0.03 |
| mod 20 | 0.93 | L20 MLP +0.32, L21 MLP +0.22, L22 MLP +0.16, L18 MLP +0.10, L19 MLP +0.10 |
| mod 10 | 0.90 | L21 MLP +0.31, L20 MLP +0.19, L18 MLP +0.16, L17 MLP +0.10, L22 MLP +0.07, L19 MLP +0.07 |
| mod 5 | 0.92 | L20 MLP +0.27, L18 MLP +0.20, L21 MLP +0.20, L22 MLP +0.11, L16 MLP +0.09, L19 MLP +0.03 |
| mod 4 | 0.74 | L22 MLP +0.39, L21 MLP +0.10, L20 MLP +0.08, L20 attn +0.06, L18 MLP +0.03 |
| mod 2 | 0.89 | L21 MLP +0.36, L18 MLP +0.30, L17 MLP +0.11, L20 MLP +0.10 |

<details><summary>period 100: 9 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L21 down c0 | +0.066 | 26% | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {2}; a//10 in {1} -> b//10 in {0,10}; a//10 in {2} -> b//10 in {0,1,9,10}; a//10 in {3} -> b//10 in {0,1,2,10}; a//10 in {4} -> b//10 in {1,2,3}; a//10 in {5} -> b//10 in {2,3,4}; a//10 in {6} -> b//10 in {3,4,5}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {5,6,7}; a//10 in {9,10} -> b//10 in {6,7,8} |
| L22 down c6 | +0.046 | 14% | **res%100** (R2 0.74): res mod 100 in {1, 3..14} |
| L21 down c105 | +0.037 | 33% | **tens(a,b)** (R2 0.72): a//10 in {0} -> b//10 in {0,1,2}; a//10 in {1} -> b//10 in {0,1,2,10}; a//10 in {2} -> b//10 in {0,1,2,3}; a//10 in {3} -> b//10 in {1,2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9}; a//10 in {9,10} -> b//10 in {7,8,9,10} |
| L18 down c23 | +0.034 | 35% | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {0,9,10}; a//10 in {1} -> b//10 in {0,1,10}; a//10 in {2} -> b//10 in {0,1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3,4,6}; a//10 in {5} -> b//10 in {3,4,5,6}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9}; a//10 in {9} -> b//10 in {0,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,8,9,10} |
| L19 down c8 | +0.034 | 87% | unexplained (best R2 0.46) |
| L20 down c20 | +0.027 | 48% | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {0,1,2,3}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {0,1,2,3,9,10}; a//10 in {3} -> b//10 in {0,1,2,3,4,6,10}; a//10 in {4} -> b//10 in {0,1,2,3,4,6}; a//10 in {5} -> b//10 in {1,2,3,4,5}; a//10 in {6} -> b//10 in {2,3,4,5,6}; a//10 in {7} -> b//10 in {3,4,5,6,9}; a//10 in {8} -> b//10 in {4,5,6,7,8,9}; a//10 in {9} -> b//10 in {6,7,8,9,10}; a//10 in {10} -> b//10 in {0,5,6,7,8,9} |
| L21 down c9 | +0.025 | 93% | **res%100** (R2 0.46): res mod 100 in {0..2, 4..98} |
| L16 down c10 | +0.023 | 47% | **tens(a,b)** (R2 0.73): a//10 in {3} -> b//10 in {2,6}; a//10 in {4} -> b//10 in {3,6,7}; a//10 in {5} -> b//10 in {3,4,6,7}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,8,9}; a//10 in {7,8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {1,2,3,4,5,6,7,8,9} |
| L15 down c10 | +0.022 | 93% | unexplained (best R2 0.34) |

</details>

<details><summary>period 50: 13 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L19 down c0 | +0.091 | 60% | **tens(a,b)** (R2 0.54): a//10 in {0,5} -> b//10 in {2,3,7,8}; a//10 in {1} -> b//10 in {3,4,7,8,9}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {1,2,6,7,8}; a//10 in {6} -> b//10 in {2,3,4,7,8}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,10}; a//10 in {9} -> b//10 in {1,2,3,6,7}; a//10 in {10} -> b//10 in {1,2,3,7,8} |
| L21 down c7 | +0.087 | 32% | **res%100** (R2 0.49): res mod 100 in {0..13, 45, 47, 49..53, 55, 90..99} |
| L19 down c11 | +0.060 | 44% | **tens(a,b)** (R2 0.61): a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {0,1,2,3,6,7,8,10}; a//10 in {3} -> b//10 in {1,2,3,7,8}; a//10 in {4} -> b//10 in {2,3,7,8}; a//10 in {5} -> b//10 in {3,4,8}; a//10 in {6} -> b//10 in {4,5,6,8,9}; a//10 in {7} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {1,2,3,6,7,8,9}; a//10 in {9} -> b//10 in {7,8,9}; a//10 in {10} -> b//10 in {3,4,8,9} |
| L18 down c12 | +0.051 | 59% | **tens(a,b)** (R2 0.51): a//10 in {0} -> b//10 in {0,1,4,5,9,10}; a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,8,9}; a//10 in {3} -> b//10 in {0,1,2,6,7,8}; a//10 in {4} -> b//10 in {0,1,2,3,4,6,7,8,9,10}; a//10 in {5,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {0,3,4,5,6,8,9,10}; a//10 in {7} -> b//10 in {0,4,5,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,5,6,7,8,10}; a//10 in {9} -> b//10 in {0,1,2,3,5,6,7,8,9,10} |
| L20 down c29 | +0.051 | 53% | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {2,6,7,8}; a//10 in {1} -> b//10 in {7,8,9,10}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {1,2,6,7,8}; a//10 in {4,9} -> b//10 in {1,2,3,6,7,8}; a//10 in {5} -> b//10 in {2,3,4,7,8,9}; a//10 in {6} -> b//10 in {0,4,8,9,10}; a//10 in {7} -> b//10 in {0,1,5,6,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,5,6,7,8,10}; a//10 in {10} -> b//10 in {2,3,4,7,8} |
| L22 down c6 | +0.050 | 14% | **res%100** (R2 0.74): res mod 100 in {1, 3..14} |
| L21 down c0 | +0.046 | 26% | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {2}; a//10 in {1} -> b//10 in {0,10}; a//10 in {2} -> b//10 in {0,1,9,10}; a//10 in {3} -> b//10 in {0,1,2,10}; a//10 in {4} -> b//10 in {1,2,3}; a//10 in {5} -> b//10 in {2,3,4}; a//10 in {6} -> b//10 in {3,4,5}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {5,6,7}; a//10 in {9,10} -> b//10 in {6,7,8} |
| L18 down c21 | +0.024 | 36% | **tens(a,b)** (R2 0.57): a//10 in {0} -> b//10 in {1}; a//10 in {1} -> b//10 in {1,2,6,7}; a//10 in {2} -> b//10 in {2}; a//10 in {3} -> b//10 in {6}; a//10 in {4,9} -> b//10 in {1,6,7}; a//10 in {5} -> b//10 in {1,2,3,6,7}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,8,10}; a//10 in {7} -> b//10 in {3,4,7,8,10}; a//10 in {8} -> b//10 in {5}; a//10 in {10} -> b//10 in {0,1,2,5,6,7} |
| L18 down c22 | +0.023 | 37% | **tens(a,b)** (R2 0.64): a//10 in {2} -> b//10 in {10}; a//10 in {3} -> b//10 in {0,1,5,6,9,10}; a//10 in {4} -> b//10 in {0,1,2,5,6,7,9,10}; a//10 in {5} -> b//10 in {0,1,2,6,7}; a//10 in {6} -> b//10 in {3}; a//10 in {7} -> b//10 in {3,4,5,10}; a//10 in {8} -> b//10 in {0,4,5,6,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,10}; a//10 in {10} -> b//10 in {0,1,2,5,6,7,10} |
| L19 down c20 | +0.023 | 25% | **tens(a,b)** (R2 0.56): a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {2,3}; a//10 in {5} -> b//10 in {3,4,8}; a//10 in {6} -> b//10 in {0,4,5,9,10}; a//10 in {7} -> b//10 in {0,5,6}; a//10 in {8} -> b//10 in {0,6,7}; a//10 in {9} -> b//10 in {2,7,8}; a//10 in {10} -> b//10 in {2,3,4,8,9} |
| L18 down c25 | +0.022 | 32% | **tens(a,b)** (R2 0.54): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {0,1}; a//10 in {2} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {0,5,10}; a//10 in {5} -> b//10 in {0,1,5,6,9,10}; a//10 in {6} -> b//10 in {0,1,2,5,6,7,9,10}; a//10 in {7} -> b//10 in {1,2,3,6,7}; a//10 in {8} -> b//10 in {4}; a//10 in {9} -> b//10 in {0,4,5,9}; a//10 in {10} -> b//10 in {0,1,4,5,6,9,10} |
| L21 down c105 | +0.021 | 33% | **tens(a,b)** (R2 0.72): a//10 in {0} -> b//10 in {0,1,2}; a//10 in {1} -> b//10 in {0,1,2,10}; a//10 in {2} -> b//10 in {0,1,2,3}; a//10 in {3} -> b//10 in {1,2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9}; a//10 in {9,10} -> b//10 in {7,8,9,10} |
| L18 down c36 | +0.020 | 23% | **tens(a,b)** (R2 0.55): a//10 in {3} -> b//10 in {2,3,8}; a//10 in {4} -> b//10 in {3,8}; a//10 in {5} -> b//10 in {0,4,5,8,9,10}; a//10 in {6} -> b//10 in {0,9,10}; a//10 in {8} -> b//10 in {3,4,7,8}; a//10 in {9} -> b//10 in {3,4,7,8,9}; a//10 in {10} -> b//10 in {0,3,4,5,7,8,9,10} |

</details>

<details><summary>period 25: 7 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L22 down c6 | +0.116 | 14% | **res%100** (R2 0.74): res mod 100 in {1, 3..14} |
| L20 down c3 | +0.055 | 53% | unexplained (best R2 0.43) |
| L21 down c2 | +0.044 | 28% | **res** (R2 0.55): res in {-10..-4, 10..17, 30..36, 50..56, 70..76, 90..96} |
| L20 down c6 | +0.041 | 49% | unexplained (best R2 0.33) |
| L21 down c7 | +0.031 | 32% | **res%100** (R2 0.49): res mod 100 in {0..13, 45, 47, 49..53, 55, 90..99} |
| L20 down c31 | +0.024 | 25% | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {0,1,9,10}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {4,5}; a//10 in {6} -> b//10 in {5,6,7,8}; a//10 in {7} -> b//10 in {6,7,8,9}; a//10 in {8} -> b//10 in {7,8,9,10}; a//10 in {9,10} -> b//10 in {0,8,9,10} |
| L21 down c15 | +0.021 | 29% | unexplained (best R2 0.44) |

</details>

<details><summary>period 20: 9 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c3 | +0.148 | 53% | unexplained (best R2 0.43) |
| L20 down c6 | +0.133 | 49% | unexplained (best R2 0.33) |
| L21 down c2 | +0.117 | 28% | **res** (R2 0.55): res in {-10..-4, 10..17, 30..36, 50..56, 70..76, 90..96} |
| L22 down c11 | +0.098 | 25% | **res** (R2 0.51): res in {-99..-98, -41..-38, -22..-18, 0..3, 18..24, 38..43, 58..62, 78..82, 97..99} |
| L19 down c6 | +0.094 | 66% | unexplained (best R2 0.21) |
| L21 down c15 | +0.060 | 29% | unexplained (best R2 0.44) |
| L18 down c18 | +0.053 | 28% | unexplained (best R2 0.38) |
| L18 down c16 | +0.031 | 21% | unexplained (best R2 0.27) |
| L22 down c6 | +0.027 | 14% | **res%100** (R2 0.74): res mod 100 in {1, 3..14} |

</details>

<details><summary>period 10: 16 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L21 down c4 | +0.130 | 40% | **units(a,b)** (R2 0.46): a%10 in {0} -> b%10 in {0,7,8,9}; a%10 in {1} -> b%10 in {0,1,8,9}; a%10 in {2} -> b%10 in {0,1,2,8,9}; a%10 in {3} -> b%10 in {0,1,2,3,9}; a%10 in {4} -> b%10 in {1,2,3,4}; a%10 in {5} -> b%10 in {2,3,4,5,7}; a%10 in {6} -> b%10 in {4,5,6}; a%10 in {7} -> b%10 in {4,5,6,7}; a%10 in {8} -> b%10 in {5,6,7,8}; a%10 in {9} -> b%10 in {6,7,8,9} |
| L20 down c13 | +0.088 | 39% | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {2,3,4,5}; a%10 in {1} -> b%10 in {3,4,5,6}; a%10 in {2} -> b%10 in {4,5,6,7}; a%10 in {3} -> b%10 in {5,6,7}; a%10 in {4} -> b%10 in {6,7,8,9}; a%10 in {5} -> b%10 in {0,8,9}; a%10 in {6} -> b%10 in {0,1,8,9}; a%10 in {7} -> b%10 in {0,1,2,3,9}; a%10 in {8} -> b%10 in {0,1,2,3,4}; a%10 in {9} -> b%10 in {1,2,3,4,5} |
| L20 down c21 | +0.078 | 37% | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {0,1,2,3,9}; a%10 in {1} -> b%10 in {0,1,2,3}; a%10 in {2} -> b%10 in {1,2,3,4}; a%10 in {3} -> b%10 in {3,4,5,6}; a%10 in {4} -> b%10 in {4,5,6}; a%10 in {5} -> b%10 in {6,7}; a%10 in {6} -> b%10 in {6,7,8}; a%10 in {7} -> b%10 in {0,7,8,9}; a%10 in {8} -> b%10 in {0,1,8,9}; a%10 in {9} -> b%10 in {0,1,2,8,9} |
| L21 down c10 | +0.067 | 18% | unexplained (best R2 0.50) |
| L18 down c32 | +0.039 | 30% | unexplained (best R2 0.50) |
| L18 down c62 | +0.035 | 28% | **units(a,b)** (R2 0.68): a%10 in {0} -> b%10 in {0,1,2,8,9}; a%10 in {1} -> b%10 in {0,1,2,3,8,9}; a%10 in {2} -> b%10 in {0,1,2,9}; a%10 in {7} -> b%10 in {8,9}; a%10 in {8} -> b%10 in {0,8,9}; a%10 in {9} -> b%10 in {0,1,7,8,9} |
| L21 down c14 | +0.028 | 10% | **res%10** (R2 0.60): res mod 10 in {5} |
| L18 down c98 | +0.028 | 27% | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {2,3,4,5,6}; a%10 in {1} -> b%10 in {1,3,4,5,6}; a%10 in {2} -> b%10 in {0,1,2,4,5,6}; a%10 in {3} -> b%10 in {0,1,2,3}; a%10 in {4} -> b%10 in {1,2,3}; a%10 in {8} -> b%10 in {2,3,4}; a%10 in {9} -> b%10 in {2,3,4,5} |
| L21 down c13 | +0.028 | 10% | **res** (R2 0.55): res in {-16, -6, 4, 14, 24..25, 34, 44, 54, 64, 74, 84, 94} |
| L22 down c8 | +0.027 | 9% | **res%10** (R2 0.52): res mod 10 in {0} |
| L19 down c41 | +0.026 | 27% | **units(a,b)** (R2 0.55): a%10 in {0} -> b%10 in {1,2,3,4}; a%10 in {1} -> b%10 in {2,3,4}; a%10 in {6} -> b%10 in {0,8,9}; a%10 in {7} -> b%10 in {0,1,2,8,9}; a%10 in {8} -> b%10 in {0,1,2,9}; a%10 in {9} -> b%10 in {0,1,2,3} |
| L18 down c65 | +0.024 | 29% | unexplained (best R2 0.49) |
| L17 down c34 | +0.022 | 21% | **units(a,b)** (R2 0.55): a%10 in {0} -> b%10 in {4}; a%10 in {5,6} -> b%10 in {0,1,2,3,4,9}; a%10 in {7} -> b%10 in {0,1,2,3,4}; a%10 in {8} -> b%10 in {2,3,4}; a%10 in {9} -> b%10 in {3,4} |
| L22 down c9 | +0.022 | 8% | **units(a,b)** (R2 0.50): a%10 in {0} -> b%10 in {1}; a%10 in {1} -> b%10 in {2}; a%10 in {2} -> b%10 in {3}; a%10 in {5} -> b%10 in {6}; a%10 in {6} -> b%10 in {7}; a%10 in {7} -> b%10 in {8}; a%10 in {9} -> b%10 in {0} |
| L17 down c39 | +0.022 | 24% | **units(a,b)** (R2 0.54): a%10 in {0,3,4,8,9} -> b%10 in {1,2,3,4}; a%10 in {1} -> b%10 in {2,3,4}; a%10 in {5} -> b%10 in {3,4}; a%10 in {7} -> b%10 in {1,2,3} |
| L17 down c49 | +0.021 | 27% | **a** (R2 0.50): a in {1, 26..27, 36..37, 46..47, 55..58, 65..68, 75..78, 85..88, 95..97} [coarser: a mod 20 in {6..7, 15..17}, R2 0.80] |

</details>

<details><summary>period 5: 13 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c9 | +0.141 | 49% | **res** (R2 0.62): res in {-97, -93, -63, -58, -53..-52, -48..-47, -43..-42, -38..-37, -33..-32, -28..-27, -23..-22, -19..-17, -13..-12, -9..-7, -4..-2, 1..3, 6..8, 11..13, 16..18, 21..23, 26..28, 31..33, 36..38, 41..43, 46..48, 51..53, 56..58, 61..63, 66..68, 71..73, 76..78, 81..83, 86..88, 91..93, 96..98} |
| L20 down c15 | +0.092 | 46% | unexplained (best R2 0.44) |
| L21 down c10 | +0.091 | 18% | unexplained (best R2 0.50) |
| L22 down c8 | +0.059 | 9% | **res%10** (R2 0.52): res mod 10 in {0} |
| L18 down c26 | +0.055 | 42% | **res** (R2 0.51): res in {-97, -48, -43..-42, -38..-37, -33, -28..-27, -23..-22, -18..-17, -13..-12, -8..-7, -3..-1, 1..3, 6..8, 11..13, 16..18, 21..23, 26..28, 31..33, 37..38, 42..43, 47..48, 52..53, 57..58, 62..63, 67..68, 72..73, 77..78, 82..83, 87..88, 92..93, 96..98} |
| L18 down c24 | +0.049 | 51% | unexplained (best R2 0.49) |
| L22 down c9 | +0.033 | 8% | **units(a,b)** (R2 0.50): a%10 in {0} -> b%10 in {1}; a%10 in {1} -> b%10 in {2}; a%10 in {2} -> b%10 in {3}; a%10 in {5} -> b%10 in {6}; a%10 in {6} -> b%10 in {7}; a%10 in {7} -> b%10 in {8}; a%10 in {9} -> b%10 in {0} |
| L21 down c14 | +0.032 | 10% | **res%10** (R2 0.60): res mod 10 in {5} |
| L18 down c80 | +0.030 | 24% | unexplained (best R2 0.46) |
| L16 down c127 | +0.029 | 18% | **units(a,b)** (R2 0.69): a%10 in {0,5} -> b%10 in {1,6}; a%10 in {1} -> b%10 in {0,1,2,5,6,7}; a%10 in {2} -> b%10 in {1,2,7}; a%10 in {6,7} -> b%10 in {1,2,6,7} |
| L21 down c16 | +0.023 | 10% | **units(a,b)** (R2 0.55): a%10 in {0} -> b%10 in {4}; a%10 in {1} -> b%10 in {5}; a%10 in {2} -> b%10 in {6}; a%10 in {4} -> b%10 in {8}; a%10 in {5} -> b%10 in {9}; a%10 in {6} -> b%10 in {0}; a%10 in {7} -> b%10 in {1}; a%10 in {8} -> b%10 in {2}; a%10 in {9} -> b%10 in {3} |
| L18 down c101 | +0.022 | 24% | unexplained (best R2 0.45) |
| L18 down c375 | +0.021 | 26% | unexplained (best R2 0.46) |

</details>

<details><summary>period 4: 5 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L22 down c17 | +0.269 | 5% | **res** (R2 0.58): res in {-18, 2..3, 22..23, 42, 82} |
| L20 down c44 | +0.047 | 12% | unexplained (best R2 0.32) |
| L22 down c427 | +0.038 | 2% | unexplained (best R2 0.46) |
| L20 o c328 (H2) | +0.035 | 19% | unexplained (best R2 0.40) |
| L22 down c48 | +0.024 | 3% | **res** (R2 0.54): res in {8, 27..29} |

</details>

<details><summary>period 2: 6 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L21 down c1 | +0.332 | 57% | **res%2** (R2 0.65): res mod 2 in {1} |
| L18 down c4 | +0.294 | 40% | **units(a,b)** (R2 0.52): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} |
| L20 down c47 | +0.095 | 28% | **units(a,b)** (R2 0.75): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} |
| L17 down c22 | +0.069 | 25% | **units(a,b)** (R2 0.81): a%10 in {1,3,5,7,9} -> b%10 in {0,2,4,6,8} |
| L17 down c67 | +0.039 | 20% | **units(a,b)** (R2 0.73): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} |
| L21 down c10 | +0.025 | 18% | unexplained (best R2 0.50) |

</details>

</details>

<details><summary>`mlp_in.26`, add: who writes the a+b code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.93 | L25 MLP +0.18, L24 MLP +0.16, L23 MLP +0.15, L21 MLP +0.10, L22 MLP +0.09, L20 MLP +0.08, L19 MLP +0.06, L16 MLP +0.05, L18 MLP +0.04, L17 MLP +0.02 |
| mod 50 | 0.99 | L18 MLP +0.18, L19 MLP +0.15, L24 MLP +0.15, L25 MLP +0.13, L23 MLP +0.11, L21 MLP +0.10, L20 MLP +0.08, L22 MLP +0.06 |
| mod 25 | 0.84 | L24 MLP +0.29, L25 MLP +0.26, L23 MLP +0.14, L22 MLP +0.09, L21 MLP +0.04 |
| mod 20 | 0.98 | L20 MLP +0.29, L21 MLP +0.16, L24 MLP +0.13, L25 MLP +0.12, L22 MLP +0.11, L18 MLP +0.07, L19 MLP +0.06, L23 MLP +0.05 |
| mod 10 | 0.98 | L24 MLP +0.19, L21 MLP +0.19, L25 MLP +0.15, L23 MLP +0.12, L20 MLP +0.09, L22 MLP +0.08, L18 MLP +0.07, L19 MLP +0.05, L17 MLP +0.05 |
| mod 5 | 0.99 | L24 MLP +0.20, L25 MLP +0.17, L20 MLP +0.15, L23 MLP +0.14, L18 MLP +0.10, L21 MLP +0.09, L22 MLP +0.08, L16 MLP +0.03 |
| mod 4 | 0.63 | L25 MLP +0.24, L22 MLP +0.19, L24 MLP +0.11, L23 MLP +0.04, L21 MLP +0.02 |
| mod 2 | 0.99 | L21 MLP +0.26, L18 MLP +0.20, L25 MLP +0.11, L17 MLP +0.10, L24 MLP +0.09, L20 MLP +0.09, L23 MLP +0.08, L19 MLP +0.03, L22 MLP +0.03 |

<details><summary>period 100: 10 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L21 down c0 | +0.047 | 33% | **res//10** (R2 0.84): (tens) res in {8..40, 108..140} [coarser: res mod 100 in {8..40}, R2 1.00] |
| L23 down c3 | +0.030 | 15% | **res%100** (R2 0.88): res mod 100 in {0..1, 90..99} |
| L20 down c2 | +0.030 | 100% | always |
| L25 down c10 | +0.029 | 13% | **res%100** (R2 0.81): res mod 100 in {86..96, 98} |
| L19 down c24 | +0.027 | 44% | **res//10** (R2 0.77): (tens) res in {24, 26, 28..80, 142..174} [coarser: res mod 100 in {34..76}, R2 0.81] |
| L19 down c28 | +0.024 | 57% | **res//10** (R2 0.79): (tens) res in {14..52, 100..162, 164} [coarser: res mod 100 in {1, 10..56}, R2 0.81] |
| L18 down c23 | +0.022 | 52% | **res//10** (R2 0.71): (tens) res in {2..29, 31, 83..130, 179..200} [coarser: res mod 100 in {0..31, 81..99}, R2 0.99] |
| L20 down c20 | +0.021 | 54% | **res//10** (R2 0.81): (tens) res in {2..50, 99..150, 198..200} [coarser: res mod 100 in {0..50, 98..99}, R2 0.99] |
| L25 down c15 | +0.021 | 9% | **res%100** (R2 0.87): res mod 100 in {0..7} |
| L24 down c13 | +0.021 | 14% | **res%100** (R2 0.89): res mod 100 in {52..64} |

</details>

<details><summary>period 50: 15 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L19 down c0 | +0.076 | 51% | **res%50** (R2 0.84): res mod 50 in {14..39} |
| L21 down c7 | +0.050 | 42% | **res%50** (R2 0.81): res mod 50 in {0..12, 42..49} |
| L18 down c12 | +0.048 | 78% | unexplained (best R2 0.46) |
| L20 down c29 | +0.039 | 56% | **res%100** (R2 0.82): res mod 100 in {11..29, 55..86} |
| L19 down c11 | +0.039 | 46% | **tens(a,b)** (R2 0.71): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {4,8,9,10}; a//10 in {2,7} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {3,8} -> b//10 in {1,2,3,6,7,8}; a//10 in {4} -> b//10 in {1,2,5,6,7}; a//10 in {5} -> b//10 in {4,5,6}; a//10 in {6} -> b//10 in {3,4,5,8,9,10}; a//10 in {9} -> b//10 in {1,2,6,7,10}; a//10 in {10} -> b//10 in {0,1,2,5,6,7,10} |
| L18 down c21 | +0.035 | 70% | **tens(a,b)** (R2 0.57): a//10 in {0} -> b//10 in {1,2,3,4,6,7,8,9}; a//10 in {1,6} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,1,2,5,6,7,10}; a//10 in {3} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {4} -> b//10 in {2,3,4,8,9,10}; a//10 in {5} -> b//10 in {1,2,3,4,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,4,5,6,7,9,10}; a//10 in {8} -> b//10 in {0,1,4,5,9,10}; a//10 in {9} -> b//10 in {3,4,8,9,10}; a//10 in {10} -> b//10 in {2,3,4,7,8,9} |
| L19 down c20 | +0.028 | 60% | **res%50** (R2 0.68): res mod 50 in {4..31} |
| L18 down c22 | +0.027 | 65% | **res%50** (R2 0.65): res mod 50 in {0, 18..49} |
| L18 down c25 | +0.025 | 78% | **tens(a,b)** (R2 0.52): a//10 in {0} -> b//10 in {3,4,5,7,8,9,10}; a//10 in {1} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {2} -> b//10 in {1,2,3,4,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,2,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,4,5,6,8,9,10}; a//10 in {5} -> b//10 in {0,3,4,5,7,8,9,10}; a//10 in {6} -> b//10 in {0,2,3,4,7,8,9,10}; a//10 in {7,8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {0,1,3,4,5,6,8,9,10}; a//10 in {10} -> b//10 in {0,3,4,5,6,7,8,9,10} |
| L23 down c3 | +0.025 | 15% | **res%100** (R2 0.88): res mod 100 in {0..1, 90..99} |
| L20 down c14 | +0.023 | 65% | **res** (R2 0.66): res in {39, 73..104, 114, 118..165, 167..200} |
| L25 down c10 | +0.022 | 13% | **res%100** (R2 0.81): res mod 100 in {86..96, 98} |
| L25 down c31 | +0.022 | 16% | **res%100** (R2 0.86): res mod 100 in {1, 13..14} |
| L23 down c16 | +0.021 | 18% | **res%100** (R2 0.91): res mod 100 in {70..86} |
| L21 down c0 | +0.020 | 33% | **res//10** (R2 0.84): (tens) res in {8..40, 108..140} [coarser: res mod 100 in {8..40}, R2 1.00] |

</details>

<details><summary>period 25: 7 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L25 down c10 | +0.057 | 13% | **res%100** (R2 0.81): res mod 100 in {86..96, 98} |
| L25 down c15 | +0.042 | 9% | **res%100** (R2 0.87): res mod 100 in {0..7} |
| L23 down c3 | +0.041 | 15% | **res%100** (R2 0.88): res mod 100 in {0..1, 90..99} |
| L24 down c16 | +0.031 | 18% | **res%100** (R2 0.76): res mod 100 in {16..17, 21..22, 24..28, 70..72, 74..77, 81..82} [coarser: res mod 50 in {16, 20..22, 24..27}, R2 0.84] |
| L22 down c5 | +0.026 | 16% | **res%100** (R2 0.76): res mod 100 in {39..51} |
| L24 down c13 | +0.025 | 14% | **res%100** (R2 0.89): res mod 100 in {52..64} |
| L24 down c17 | +0.024 | 12% | **res%100** (R2 0.86): res mod 100 in {46..56} |

</details>

<details><summary>period 20: 9 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c3 | +0.148 | 61% | **res%20** (R2 0.74): res mod 20 in {8..19} |
| L20 down c6 | +0.123 | 60% | **res%20** (R2 0.60): res mod 20 in {1..11} |
| L21 down c2 | +0.098 | 36% | **res%20** (R2 0.89): res mod 20 in {10..16} |
| L22 down c11 | +0.069 | 33% | **res%20** (R2 0.89): res mod 20 in {0..3, 18..19} |
| L19 down c6 | +0.056 | 66% | unexplained (best R2 0.26) |
| L21 down c15 | +0.046 | 50% | **res%20** (R2 0.61): res mod 20 in {0, 4, 13..19} |
| L18 down c18 | +0.032 | 35% | **b%20** (R2 0.60): b mod 20 in {0..4, 17..19} |
| L24 down c43 | +0.026 | 22% | **res%100** (R2 0.79): res mod 100 in {15..19, 36..37, 55..59, 75..79, 95..99} [coarser: res mod 20 in {15..19}, R2 0.87] |
| L18 down c16 | +0.023 | 37% | **b%20** (R2 0.59): b mod 20 in {0..4, 15..19} |

</details>

<details><summary>period 10: 20 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L24 down c4 | +0.059 | 11% | **res%10** (R2 0.86): res mod 10 in {8} |
| L21 down c4 | +0.057 | 41% | **res%10** (R2 0.85): res mod 10 in {0..3} |
| L24 down c5 | +0.055 | 11% | **res%10** (R2 0.90): res mod 10 in {3} |
| L25 down c11 | +0.046 | 11% | **res%10** (R2 0.92): res mod 10 in {7} |
| L20 down c13 | +0.040 | 42% | **res%10** (R2 0.86): res mod 10 in {5..8} |
| L25 down c12 | +0.037 | 11% | **res%10** (R2 0.86): res mod 10 in {1} |
| L20 down c21 | +0.037 | 42% | **res%10** (R2 0.89): res mod 10 in {0, 7..9} |
| L23 down c4 | +0.032 | 11% | **res%10** (R2 0.91): res mod 10 in {2} |
| L22 down c9 | +0.031 | 10% | **res%10** (R2 0.94): res mod 10 in {9} |
| L21 down c10 | +0.029 | 11% | **res%10** (R2 0.93): res mod 10 in {1} |
| L21 down c13 | +0.028 | 24% | **res%10** (R2 0.73): res mod 10 in {3..5} |
| L21 down c14 | +0.027 | 11% | **res%10** (R2 0.89): res mod 10 in {5} |
| L24 down c12 | +0.026 | 10% | **res%10** (R2 0.96): res mod 10 in {5} |
| L22 down c8 | +0.025 | 10% | **res%10** (R2 0.98): res mod 10 in {0} |
| L25 down c24 | +0.025 | 11% | **res%10** (R2 0.86): res mod 10 in {9} |
| L23 down c14 | +0.024 | 11% | **res%10** (R2 0.93): res mod 10 in {4} |
| L18 down c32 | +0.023 | 46% | **units(a,b)** (R2 0.85): a%10 in {0} -> b%10 in {3,4,5,6}; a%10 in {1,3,9} -> b%10 in {3,4,5,6,7}; a%10 in {2} -> b%10 in {5,6,7}; a%10 in {4,5} -> b%10 in {2,3,4,5,6,7}; a%10 in {6} -> b%10 in {1,2,3,4,5,6}; a%10 in {7} -> b%10 in {1,2,3,4}; a%10 in {8} -> b%10 in {4,5,6} |
| L19 down c41 | +0.022 | 40% | **units(a,b)** (R2 0.85): a%10 in {0} -> b%10 in {6,7,8,9}; a%10 in {1} -> b%10 in {5,6,7,8}; a%10 in {2} -> b%10 in {5,6,7}; a%10 in {3} -> b%10 in {5,6}; a%10 in {5} -> b%10 in {1,2,3,4}; a%10 in {6} -> b%10 in {0,1,2,3,9}; a%10 in {7} -> b%10 in {0,1,2,8,9}; a%10 in {8} -> b%10 in {0,1,7,8,9}; a%10 in {9} -> b%10 in {0,6,7,8,9} |
| L23 down c9 | +0.021 | 10% | **res%10** (R2 0.95): res mod 10 in {6} |
| L23 down c24 | +0.020 | 10% | **res%10** (R2 0.96): res mod 10 in {7} |

</details>

<details><summary>period 5: 18 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c9 | +0.068 | 60% | **res%5** (R2 0.95): res mod 5 in {1..3} |
| L24 down c4 | +0.064 | 11% | **res%10** (R2 0.86): res mod 10 in {8} |
| L24 down c5 | +0.058 | 11% | **res%10** (R2 0.90): res mod 10 in {3} |
| L25 down c11 | +0.055 | 11% | **res%10** (R2 0.92): res mod 10 in {7} |
| L20 down c15 | +0.054 | 61% | **res%5** (R2 0.83): res mod 5 in {0..2} |
| L25 down c12 | +0.042 | 11% | **res%10** (R2 0.86): res mod 10 in {1} |
| L22 down c8 | +0.034 | 10% | **res%10** (R2 0.98): res mod 10 in {0} |
| L23 down c4 | +0.034 | 11% | **res%10** (R2 0.91): res mod 10 in {2} |
| L22 down c9 | +0.033 | 10% | **res%10** (R2 0.94): res mod 10 in {9} |
| L25 down c24 | +0.033 | 11% | **res%10** (R2 0.86): res mod 10 in {9} |
| L24 down c12 | +0.032 | 10% | **res%10** (R2 0.96): res mod 10 in {5} |
| L18 down c26 | +0.031 | 69% | **units(a,b)** (R2 0.80): a%10 in {0,5} -> b%10 in {1,2,3,4,6,7,8,9}; a%10 in {1,6} -> b%10 in {0,1,2,3,5,6,7,8}; a%10 in {2,7} -> b%10 in {0,1,4,5,6,9}; a%10 in {3,8} -> b%10 in {0,3,4,5,8,9}; a%10 in {4,9} -> b%10 in {0,2,3,4,5,7,8,9} |
| L23 down c9 | +0.027 | 10% | **res%10** (R2 0.95): res mod 10 in {6} |
| L23 down c14 | +0.027 | 11% | **res%10** (R2 0.93): res mod 10 in {4} |
| L18 down c24 | +0.026 | 57% | **units(a,b)** (R2 0.87): a%10 in {0,5} -> b%10 in {1,2,6,7}; a%10 in {1} -> b%10 in {0,1,5,6,9}; a%10 in {2,7} -> b%10 in {0,3,4,5,8,9}; a%10 in {3} -> b%10 in {0,2,3,4,7,8,9}; a%10 in {4,9} -> b%10 in {1,2,3,6,7,8}; a%10 in {6} -> b%10 in {0,1,4,5,6,9}; a%10 in {8} -> b%10 in {2,3,4,7,8,9} |
| L21 down c10 | +0.025 | 11% | **res%10** (R2 0.93): res mod 10 in {1} |
| L23 down c24 | +0.024 | 10% | **res%10** (R2 0.96): res mod 10 in {7} |
| L21 down c14 | +0.020 | 11% | **res%10** (R2 0.89): res mod 10 in {5} |

</details>

<details><summary>period 4: 7 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L22 down c17 | +0.144 | 10% | **res%20** (R2 0.84): res mod 20 in {2..3} |
| L25 down c50 | +0.059 | 11% | **res%100** (R2 0.79): res mod 100 in {13, 22..25, 43, 53, 63, 73, 83} |
| L22 down c427 | +0.035 | 7% | **res%20** (R2 0.78): res mod 20 in {12} |
| L25 down c76 | +0.031 | 15% | **res%100** (R2 0.74): res mod 100 in {17..18, 36..39, 57..59, 76..79, 97..98} [coarser: res mod 20 in {17..19}, R2 0.84] |
| L24 down c52 | +0.029 | 9% | **res%100** (R2 0.75): res mod 100 in {14, 24, 34..35, 44, 54, 74, 84, 94} [coarser: res mod 50 in {14, 24, 34, 44}, R2 0.81] |
| L24 down c50 | +0.022 | 6% | **res%100** (R2 0.75): res mod 100 in {25..27, 76, 86} |
| L25 down c207 | +0.021 | 6% | **res%100** (R2 0.78): res mod 100 in {12, 14, 52, 72, 92} [coarser: res mod 20 in {12}, R2 0.80] |

</details>

<details><summary>period 2: 13 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L21 down c1 | +0.231 | 50% | **res%2** (R2 0.99): res mod 2 in {1} |
| L18 down c4 | +0.196 | 41% | **units(a,b)** (R2 0.74): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8}; a%10 in {1,5} -> b%10 in {3,5,7}; a%10 in {3,7,9} -> b%10 in {1,3,5,7,9} |
| L20 down c47 | +0.083 | 43% | **res%2** (R2 0.74): res mod 2 in {0} |
| L17 down c22 | +0.050 | 25% | **units(a,b)** (R2 0.98): a%10 in {1,3,5,7,9} -> b%10 in {0,2,4,6,8} |
| L17 down c67 | +0.045 | 25% | **units(a,b)** (R2 0.98): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} |
| L25 down c11 | +0.043 | 11% | **res%10** (R2 0.92): res mod 10 in {7} |
| L24 down c4 | +0.039 | 11% | **res%10** (R2 0.86): res mod 10 in {8} |
| L19 down c32 | +0.035 | 26% | **units(a,b)** (R2 0.82): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} |
| L25 down c12 | +0.028 | 11% | **res%10** (R2 0.86): res mod 10 in {1} |
| L24 down c5 | +0.026 | 11% | **res%10** (R2 0.90): res mod 10 in {3} |
| L23 down c4 | +0.024 | 11% | **res%10** (R2 0.91): res mod 10 in {2} |
| L25 down c24 | +0.022 | 11% | **res%10** (R2 0.86): res mod 10 in {9} |
| L23 down c9 | +0.021 | 10% | **res%10** (R2 0.95): res mod 10 in {6} |

</details>

</details>

<details><summary>`mlp_in.26`, sub: who writes the a−b code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.93 | L21 MLP +0.14, L25 MLP +0.12, L23 MLP +0.10, L24 MLP +0.10, L22 MLP +0.09, L20 MLP +0.08, L18 MLP +0.07, L19 MLP +0.06, L16 MLP +0.06, L17 MLP +0.04, L15 MLP +0.03 |
| mod 50 | 0.90 | L21 MLP +0.14, L18 MLP +0.13, L19 MLP +0.13, L23 MLP +0.11, L24 MLP +0.09, L25 MLP +0.09, L20 MLP +0.08, L22 MLP +0.07 |
| mod 25 | 0.84 | L25 MLP +0.20, L24 MLP +0.17, L23 MLP +0.10, L22 MLP +0.09, L21 MLP +0.09, L20 MLP +0.07, L18 MLP +0.04, L19 MLP +0.03 |
| mod 20 | 0.88 | L20 MLP +0.22, L21 MLP +0.16, L24 MLP +0.10, L22 MLP +0.10, L25 MLP +0.08, L19 MLP +0.07, L18 MLP +0.07, L23 MLP +0.05 |
| mod 10 | 0.91 | L21 MLP +0.19, L24 MLP +0.15, L25 MLP +0.12, L20 MLP +0.11, L18 MLP +0.10, L23 MLP +0.10, L17 MLP +0.06, L22 MLP +0.04, L19 MLP +0.04 |
| mod 5 | 0.91 | L24 MLP +0.17, L25 MLP +0.14, L20 MLP +0.13, L23 MLP +0.13, L18 MLP +0.11, L21 MLP +0.10, L22 MLP +0.06, L16 MLP +0.05 |
| mod 4 | 0.65 | L25 MLP +0.17, L22 MLP +0.14, L24 MLP +0.14, L21 MLP +0.04, L23 MLP +0.04, L20 MLP +0.04, L20 attn +0.03 |
| mod 2 | 0.87 | L21 MLP +0.26, L18 MLP +0.22, L23 MLP +0.10, L25 MLP +0.08, L17 MLP +0.08, L20 MLP +0.08, L24 MLP +0.04 |

<details><summary>period 100: 6 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L21 down c0 | +0.037 | 26% | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {2}; a//10 in {1} -> b//10 in {0,10}; a//10 in {2} -> b//10 in {0,1,9,10}; a//10 in {3} -> b//10 in {0,1,2,10}; a//10 in {4} -> b//10 in {1,2,3}; a//10 in {5} -> b//10 in {2,3,4}; a//10 in {6} -> b//10 in {3,4,5}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {5,6,7}; a//10 in {9,10} -> b//10 in {6,7,8} |
| L22 down c6 | +0.027 | 14% | **res%100** (R2 0.74): res mod 100 in {1, 3..14} |
| L21 down c105 | +0.025 | 33% | **tens(a,b)** (R2 0.72): a//10 in {0} -> b//10 in {0,1,2}; a//10 in {1} -> b//10 in {0,1,2,10}; a//10 in {2} -> b//10 in {0,1,2,3}; a//10 in {3} -> b//10 in {1,2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9}; a//10 in {9,10} -> b//10 in {7,8,9,10} |
| L25 down c220 | +0.023 | 30% | **tens(a,b)** (R2 0.64): a//10 in {0,1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {0,1,2,3}; a//10 in {3} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {7,8,9} |
| L18 down c23 | +0.022 | 35% | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {0,9,10}; a//10 in {1} -> b//10 in {0,1,10}; a//10 in {2} -> b//10 in {0,1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3,4,6}; a//10 in {5} -> b//10 in {3,4,5,6}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9}; a//10 in {9} -> b//10 in {0,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,8,9,10} |
| L19 down c8 | +0.022 | 87% | unexplained (best R2 0.46) |

</details>

<details><summary>period 50: 7 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L19 down c0 | +0.059 | 60% | **tens(a,b)** (R2 0.54): a//10 in {0,5} -> b//10 in {2,3,7,8}; a//10 in {1} -> b//10 in {3,4,7,8,9}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {1,2,6,7,8}; a//10 in {6} -> b//10 in {2,3,4,7,8}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,10}; a//10 in {9} -> b//10 in {1,2,3,6,7}; a//10 in {10} -> b//10 in {1,2,3,7,8} |
| L21 down c7 | +0.054 | 32% | **res%100** (R2 0.49): res mod 100 in {0..13, 45, 47, 49..53, 55, 90..99} |
| L19 down c11 | +0.041 | 44% | **tens(a,b)** (R2 0.61): a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {0,1,2,3,6,7,8,10}; a//10 in {3} -> b//10 in {1,2,3,7,8}; a//10 in {4} -> b//10 in {2,3,7,8}; a//10 in {5} -> b//10 in {3,4,8}; a//10 in {6} -> b//10 in {4,5,6,8,9}; a//10 in {7} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {1,2,3,6,7,8,9}; a//10 in {9} -> b//10 in {7,8,9}; a//10 in {10} -> b//10 in {3,4,8,9} |
| L20 down c29 | +0.035 | 53% | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {2,6,7,8}; a//10 in {1} -> b//10 in {7,8,9,10}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {1,2,6,7,8}; a//10 in {4,9} -> b//10 in {1,2,3,6,7,8}; a//10 in {5} -> b//10 in {2,3,4,7,8,9}; a//10 in {6} -> b//10 in {0,4,8,9,10}; a//10 in {7} -> b//10 in {0,1,5,6,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,5,6,7,8,10}; a//10 in {10} -> b//10 in {2,3,4,7,8} |
| L18 down c12 | +0.035 | 59% | **tens(a,b)** (R2 0.51): a//10 in {0} -> b//10 in {0,1,4,5,9,10}; a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,8,9}; a//10 in {3} -> b//10 in {0,1,2,6,7,8}; a//10 in {4} -> b//10 in {0,1,2,3,4,6,7,8,9,10}; a//10 in {5,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {0,3,4,5,6,8,9,10}; a//10 in {7} -> b//10 in {0,4,5,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,5,6,7,8,10}; a//10 in {9} -> b//10 in {0,1,2,3,5,6,7,8,9,10} |
| L22 down c6 | +0.031 | 14% | **res%100** (R2 0.74): res mod 100 in {1, 3..14} |
| L21 down c0 | +0.028 | 26% | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {2}; a//10 in {1} -> b//10 in {0,10}; a//10 in {2} -> b//10 in {0,1,9,10}; a//10 in {3} -> b//10 in {0,1,2,10}; a//10 in {4} -> b//10 in {1,2,3}; a//10 in {5} -> b//10 in {2,3,4}; a//10 in {6} -> b//10 in {3,4,5}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {5,6,7}; a//10 in {9,10} -> b//10 in {6,7,8} |

</details>

<details><summary>period 25: 5 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L25 down c15 | +0.046 | 8% | **res%100** (R2 0.68): res mod 100 in {1..7} |
| L22 down c6 | +0.045 | 14% | **res%100** (R2 0.74): res mod 100 in {1, 3..14} |
| L25 down c42 | +0.037 | 8% | **res%100** (R2 0.73): res mod 100 in {10..12} |
| L24 down c16 | +0.023 | 12% | **res** (R2 0.50): res in {15..17, 20..22, 24..27, 71..72} |
| L20 down c3 | +0.023 | 53% | unexplained (best R2 0.43) |

</details>

<details><summary>period 20: 9 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c3 | +0.101 | 53% | unexplained (best R2 0.43) |
| L20 down c6 | +0.098 | 49% | unexplained (best R2 0.33) |
| L21 down c2 | +0.080 | 28% | **res** (R2 0.55): res in {-10..-4, 10..17, 30..36, 50..56, 70..76, 90..96} |
| L19 down c6 | +0.066 | 66% | unexplained (best R2 0.21) |
| L22 down c11 | +0.064 | 25% | **res** (R2 0.51): res in {-99..-98, -41..-38, -22..-18, 0..3, 18..24, 38..43, 58..62, 78..82, 97..99} |
| L21 down c15 | +0.045 | 29% | unexplained (best R2 0.44) |
| L24 down c43 | +0.040 | 21% | unexplained (best R2 0.42) |
| L18 down c18 | +0.034 | 28% | unexplained (best R2 0.38) |
| L18 down c16 | +0.022 | 21% | unexplained (best R2 0.27) |

</details>

<details><summary>period 10: 14 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L21 down c4 | +0.075 | 40% | **units(a,b)** (R2 0.46): a%10 in {0} -> b%10 in {0,7,8,9}; a%10 in {1} -> b%10 in {0,1,8,9}; a%10 in {2} -> b%10 in {0,1,2,8,9}; a%10 in {3} -> b%10 in {0,1,2,3,9}; a%10 in {4} -> b%10 in {1,2,3,4}; a%10 in {5} -> b%10 in {2,3,4,5,7}; a%10 in {6} -> b%10 in {4,5,6}; a%10 in {7} -> b%10 in {4,5,6,7}; a%10 in {8} -> b%10 in {5,6,7,8}; a%10 in {9} -> b%10 in {6,7,8,9} |
| L20 down c13 | +0.055 | 39% | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {2,3,4,5}; a%10 in {1} -> b%10 in {3,4,5,6}; a%10 in {2} -> b%10 in {4,5,6,7}; a%10 in {3} -> b%10 in {5,6,7}; a%10 in {4} -> b%10 in {6,7,8,9}; a%10 in {5} -> b%10 in {0,8,9}; a%10 in {6} -> b%10 in {0,1,8,9}; a%10 in {7} -> b%10 in {0,1,2,3,9}; a%10 in {8} -> b%10 in {0,1,2,3,4}; a%10 in {9} -> b%10 in {1,2,3,4,5} |
| L20 down c21 | +0.045 | 37% | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {0,1,2,3,9}; a%10 in {1} -> b%10 in {0,1,2,3}; a%10 in {2} -> b%10 in {1,2,3,4}; a%10 in {3} -> b%10 in {3,4,5,6}; a%10 in {4} -> b%10 in {4,5,6}; a%10 in {5} -> b%10 in {6,7}; a%10 in {6} -> b%10 in {6,7,8}; a%10 in {7} -> b%10 in {0,7,8,9}; a%10 in {8} -> b%10 in {0,1,8,9}; a%10 in {9} -> b%10 in {0,1,2,8,9} |
| L21 down c10 | +0.045 | 18% | unexplained (best R2 0.50) |
| L24 down c4 | +0.040 | 12% | unexplained (best R2 0.50) |
| L25 down c12 | +0.039 | 28% | unexplained (best R2 0.47) |
| L24 down c5 | +0.038 | 12% | unexplained (best R2 0.47) |
| L24 down c12 | +0.033 | 11% | **units(a,b)** (R2 0.64): a%10 in {0} -> b%10 in {5}; a%10 in {1} -> b%10 in {6}; a%10 in {2} -> b%10 in {7}; a%10 in {3} -> b%10 in {8}; a%10 in {4} -> b%10 in {9}; a%10 in {5} -> b%10 in {0,5}; a%10 in {6} -> b%10 in {1}; a%10 in {7} -> b%10 in {2}; a%10 in {8} -> b%10 in {3}; a%10 in {9} -> b%10 in {4} |
| L23 down c4 | +0.033 | 16% | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {8}; a%10 in {1} -> b%10 in {9}; a%10 in {2} -> b%10 in {0,2,4,6,8}; a%10 in {3} -> b%10 in {1}; a%10 in {4} -> b%10 in {2}; a%10 in {5} -> b%10 in {3,7}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {5}; a%10 in {8} -> b%10 in {6}; a%10 in {9} -> b%10 in {7} |
| L25 down c11 | +0.032 | 11% | **units(a,b)** (R2 0.51): a%10 in {0} -> b%10 in {3}; a%10 in {1} -> b%10 in {4}; a%10 in {2} -> b%10 in {5}; a%10 in {3} -> b%10 in {6}; a%10 in {5} -> b%10 in {8}; a%10 in {6} -> b%10 in {9}; a%10 in {7} -> b%10 in {0}; a%10 in {8} -> b%10 in {1}; a%10 in {9} -> b%10 in {2} |
| L18 down c32 | +0.024 | 30% | unexplained (best R2 0.50) |
| L23 down c9 | +0.023 | 14% | **units(a,b)** (R2 0.53): a%10 in {0} -> b%10 in {4}; a%10 in {1} -> b%10 in {5}; a%10 in {2} -> b%10 in {6}; a%10 in {4} -> b%10 in {0,8}; a%10 in {5} -> b%10 in {9}; a%10 in {6} -> b%10 in {0}; a%10 in {7} -> b%10 in {1}; a%10 in {8} -> b%10 in {2,4}; a%10 in {9} -> b%10 in {3} |
| L18 down c62 | +0.021 | 28% | **units(a,b)** (R2 0.68): a%10 in {0} -> b%10 in {0,1,2,8,9}; a%10 in {1} -> b%10 in {0,1,2,3,8,9}; a%10 in {2} -> b%10 in {0,1,2,9}; a%10 in {7} -> b%10 in {8,9}; a%10 in {8} -> b%10 in {0,8,9}; a%10 in {9} -> b%10 in {0,1,7,8,9} |
| L21 down c14 | +0.021 | 10% | **res%10** (R2 0.60): res mod 10 in {5} |

</details>

<details><summary>period 5: 15 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c9 | +0.069 | 49% | **res** (R2 0.62): res in {-97, -93, -63, -58, -53..-52, -48..-47, -43..-42, -38..-37, -33..-32, -28..-27, -23..-22, -19..-17, -13..-12, -9..-7, -4..-2, 1..3, 6..8, 11..13, 16..18, 21..23, 26..28, 31..33, 36..38, 41..43, 46..48, 51..53, 56..58, 61..63, 66..68, 71..73, 76..78, 81..83, 86..88, 91..93, 96..98} |
| L24 down c4 | +0.051 | 12% | unexplained (best R2 0.50) |
| L24 down c5 | +0.048 | 12% | unexplained (best R2 0.47) |
| L20 down c15 | +0.046 | 46% | unexplained (best R2 0.44) |
| L23 down c4 | +0.045 | 16% | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {8}; a%10 in {1} -> b%10 in {9}; a%10 in {2} -> b%10 in {0,2,4,6,8}; a%10 in {3} -> b%10 in {1}; a%10 in {4} -> b%10 in {2}; a%10 in {5} -> b%10 in {3,7}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {5}; a%10 in {8} -> b%10 in {6}; a%10 in {9} -> b%10 in {7} |
| L25 down c12 | +0.045 | 28% | unexplained (best R2 0.47) |
| L21 down c10 | +0.044 | 18% | unexplained (best R2 0.50) |
| L25 down c11 | +0.043 | 11% | **units(a,b)** (R2 0.51): a%10 in {0} -> b%10 in {3}; a%10 in {1} -> b%10 in {4}; a%10 in {2} -> b%10 in {5}; a%10 in {3} -> b%10 in {6}; a%10 in {5} -> b%10 in {8}; a%10 in {6} -> b%10 in {9}; a%10 in {7} -> b%10 in {0}; a%10 in {8} -> b%10 in {1}; a%10 in {9} -> b%10 in {2} |
| L24 down c12 | +0.042 | 11% | **units(a,b)** (R2 0.64): a%10 in {0} -> b%10 in {5}; a%10 in {1} -> b%10 in {6}; a%10 in {2} -> b%10 in {7}; a%10 in {3} -> b%10 in {8}; a%10 in {4} -> b%10 in {9}; a%10 in {5} -> b%10 in {0,5}; a%10 in {6} -> b%10 in {1}; a%10 in {7} -> b%10 in {2}; a%10 in {8} -> b%10 in {3}; a%10 in {9} -> b%10 in {4} |
| L23 down c9 | +0.030 | 14% | **units(a,b)** (R2 0.53): a%10 in {0} -> b%10 in {4}; a%10 in {1} -> b%10 in {5}; a%10 in {2} -> b%10 in {6}; a%10 in {4} -> b%10 in {0,8}; a%10 in {5} -> b%10 in {9}; a%10 in {6} -> b%10 in {0}; a%10 in {7} -> b%10 in {1}; a%10 in {8} -> b%10 in {2,4}; a%10 in {9} -> b%10 in {3} |
| L18 down c26 | +0.029 | 42% | **res** (R2 0.51): res in {-97, -48, -43..-42, -38..-37, -33, -28..-27, -23..-22, -18..-17, -13..-12, -8..-7, -3..-1, 1..3, 6..8, 11..13, 16..18, 21..23, 26..28, 31..33, 37..38, 42..43, 47..48, 52..53, 57..58, 62..63, 67..68, 72..73, 77..78, 82..83, 87..88, 92..93, 96..98} |
| L22 down c8 | +0.028 | 9% | **res%10** (R2 0.52): res mod 10 in {0} |
| L25 down c24 | +0.026 | 9% | **units(a,b)** (R2 0.53): a%10 in {0} -> b%10 in {1}; a%10 in {1} -> b%10 in {2}; a%10 in {2} -> b%10 in {3}; a%10 in {5} -> b%10 in {6}; a%10 in {6} -> b%10 in {7}; a%10 in {7} -> b%10 in {8}; a%10 in {8} -> b%10 in {9}; a%10 in {9} -> b%10 in {0} |
| L18 down c24 | +0.025 | 51% | unexplained (best R2 0.49) |
| L23 down c24 | +0.022 | 7% | **res** (R2 0.67): res in {-97, -23, -13, -3, 7, 17, 27, 37, 47, 57, 67, 77, 87, 97} |

</details>

<details><summary>period 4: 7 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L22 down c17 | +0.096 | 5% | **res** (R2 0.58): res in {-18, 2..3, 22..23, 42, 82} |
| L25 down c50 | +0.040 | 4% | **res** (R2 0.60): res in {3, 13, 23..25} |
| L25 down c46 | +0.028 | 4% | **res** (R2 0.57): res in {17..20} |
| L24 down c22 | +0.025 | 7% | **res** (R2 0.55): res in {13..17, 24} |
| L20 down c44 | +0.022 | 12% | unexplained (best R2 0.32) |
| L24 down c91 | +0.021 | 4% | **res** (R2 0.54): res in {6, 16, 36, 95..97} |
| L25 down c207 | +0.021 | 2% | **res** (R2 0.62): res in {12, 14} |

</details>

<details><summary>period 2: 8 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L21 down c1 | +0.244 | 57% | **res%2** (R2 0.65): res mod 2 in {1} |
| L18 down c4 | +0.215 | 40% | **units(a,b)** (R2 0.52): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} |
| L20 down c47 | +0.072 | 28% | **units(a,b)** (R2 0.75): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} |
| L17 down c22 | +0.054 | 25% | **units(a,b)** (R2 0.81): a%10 in {1,3,5,7,9} -> b%10 in {0,2,4,6,8} |
| L23 down c55 | +0.046 | 22% | **units(a,b)** (R2 0.81): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} |
| L25 down c12 | +0.046 | 28% | unexplained (best R2 0.47) |
| L17 down c67 | +0.027 | 20% | **units(a,b)** (R2 0.73): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} |
| L23 down c4 | +0.023 | 16% | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {8}; a%10 in {1} -> b%10 in {9}; a%10 in {2} -> b%10 in {0,2,4,6,8}; a%10 in {3} -> b%10 in {1}; a%10 in {4} -> b%10 in {2}; a%10 in {5} -> b%10 in {3,7}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {5}; a%10 in {8} -> b%10 in {6}; a%10 in {9} -> b%10 in {7} |

</details>

</details>

<details><summary>`mlp_in.29`, add: who writes the a+b code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.85 | L27 MLP +0.14, L28 MLP +0.13, L25 MLP +0.09, L24 MLP +0.09, L26 MLP +0.09, L23 MLP +0.08, L21 MLP +0.05, L22 MLP +0.05, L20 MLP +0.04, L19 MLP +0.03, L16 MLP +0.02 |
| mod 50 | 0.88 | L28 MLP +0.14, L27 MLP +0.11, L18 MLP +0.10, L26 MLP +0.09, L24 MLP +0.08, L19 MLP +0.08, L25 MLP +0.07, L23 MLP +0.06, L21 MLP +0.05, L20 MLP +0.04, L22 MLP +0.03 |
| mod 25 | 0.79 | L28 MLP +0.18, L27 MLP +0.15, L24 MLP +0.12, L26 MLP +0.11, L25 MLP +0.10, L23 MLP +0.06, L22 MLP +0.04 |
| mod 20 | 0.88 | L20 MLP +0.17, L28 MLP +0.12, L27 MLP +0.11, L21 MLP +0.10, L24 MLP +0.08, L26 MLP +0.08, L25 MLP +0.07, L22 MLP +0.07, L18 MLP +0.04, L19 MLP +0.04, L23 MLP +0.03 |
| mod 10 | 0.93 | L28 MLP +0.16, L21 MLP +0.13, L24 MLP +0.12, L25 MLP +0.11, L23 MLP +0.09, L20 MLP +0.06, L22 MLP +0.05, L27 MLP +0.05, L18 MLP +0.05, L26 MLP +0.05, L19 MLP +0.03, L17 MLP +0.03 |
| mod 5 | 0.95 | L28 MLP +0.17, L24 MLP +0.13, L25 MLP +0.12, L20 MLP +0.10, L23 MLP +0.10, L21 MLP +0.07, L18 MLP +0.07, L22 MLP +0.06, L27 MLP +0.05, L26 MLP +0.04, L16 MLP +0.02 |
| mod 4 | 0.61 | L28 MLP +0.20, L27 MLP +0.10, L26 MLP +0.09, L25 MLP +0.08, L22 MLP +0.07, L24 MLP +0.04 |
| mod 2 | 0.94 | L28 MLP +0.20, L21 MLP +0.17, L18 MLP +0.12, L26 MLP +0.08, L25 MLP +0.07, L20 MLP +0.06, L17 MLP +0.06, L24 MLP +0.06, L23 MLP +0.05, L27 MLP +0.02, L19 MLP +0.02 |

<details><summary>period 100: 2 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L27 down c6 | +0.026 | 61% | **res//10** (R2 0.87): (tens) res in {90..199} |
| L21 down c0 | +0.024 | 33% | **res//10** (R2 0.84): (tens) res in {8..40, 108..140} [coarser: res mod 100 in {8..40}, R2 1.00] |

</details>

<details><summary>period 50: 4 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L19 down c0 | +0.041 | 51% | **res%50** (R2 0.84): res mod 50 in {14..39} |
| L21 down c7 | +0.026 | 42% | **res%50** (R2 0.81): res mod 50 in {0..12, 42..49} |
| L18 down c12 | +0.026 | 78% | unexplained (best R2 0.46) |
| L20 down c29 | +0.021 | 56% | **res%100** (R2 0.82): res mod 100 in {11..29, 55..86} |

</details>

<details><summary>period 25: 1 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L25 down c10 | +0.021 | 13% | **res%100** (R2 0.81): res mod 100 in {86..96, 98} |

</details>

<details><summary>period 20: 6 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c3 | +0.082 | 61% | **res%20** (R2 0.74): res mod 20 in {8..19} |
| L20 down c6 | +0.073 | 60% | **res%20** (R2 0.60): res mod 20 in {1..11} |
| L21 down c2 | +0.056 | 36% | **res%20** (R2 0.89): res mod 20 in {10..16} |
| L22 down c11 | +0.041 | 33% | **res%20** (R2 0.89): res mod 20 in {0..3, 18..19} |
| L19 down c6 | +0.034 | 66% | unexplained (best R2 0.26) |
| L21 down c15 | +0.027 | 50% | **res%20** (R2 0.61): res mod 20 in {0, 4, 13..19} |

</details>

<details><summary>period 10: 10 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L21 down c4 | +0.039 | 41% | **res%10** (R2 0.85): res mod 10 in {0..3} |
| L24 down c4 | +0.035 | 11% | **res%10** (R2 0.86): res mod 10 in {8} |
| L24 down c5 | +0.035 | 11% | **res%10** (R2 0.90): res mod 10 in {3} |
| L25 down c11 | +0.032 | 11% | **res%10** (R2 0.92): res mod 10 in {7} |
| L23 down c4 | +0.027 | 11% | **res%10** (R2 0.91): res mod 10 in {2} |
| L20 down c13 | +0.027 | 42% | **res%10** (R2 0.86): res mod 10 in {5..8} |
| L28 down c36 | +0.026 | 13% | **res%10** (R2 0.75): res mod 10 in {2} |
| L20 down c21 | +0.025 | 42% | **res%10** (R2 0.89): res mod 10 in {0, 7..9} |
| L25 down c12 | +0.025 | 11% | **res%10** (R2 0.86): res mod 10 in {1} |
| L22 down c9 | +0.021 | 10% | **res%10** (R2 0.94): res mod 10 in {9} |

</details>

<details><summary>period 5: 14 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c9 | +0.049 | 60% | **res%5** (R2 0.95): res mod 5 in {1..3} |
| L24 down c4 | +0.042 | 11% | **res%10** (R2 0.86): res mod 10 in {8} |
| L24 down c5 | +0.038 | 11% | **res%10** (R2 0.90): res mod 10 in {3} |
| L25 down c11 | +0.037 | 11% | **res%10** (R2 0.92): res mod 10 in {7} |
| L20 down c15 | +0.035 | 61% | **res%5** (R2 0.83): res mod 5 in {0..2} |
| L25 down c12 | +0.029 | 11% | **res%10** (R2 0.86): res mod 10 in {1} |
| L28 down c36 | +0.029 | 13% | **res%10** (R2 0.75): res mod 10 in {2} |
| L23 down c4 | +0.028 | 11% | **res%10** (R2 0.91): res mod 10 in {2} |
| L28 down c93 | +0.026 | 14% | **res%10** (R2 0.77): res mod 10 in {5} |
| L22 down c8 | +0.025 | 10% | **res%10** (R2 0.98): res mod 10 in {0} |
| L24 down c12 | +0.022 | 10% | **res%10** (R2 0.96): res mod 10 in {5} |
| L25 down c24 | +0.022 | 11% | **res%10** (R2 0.86): res mod 10 in {9} |
| L22 down c9 | +0.021 | 10% | **res%10** (R2 0.94): res mod 10 in {9} |
| L18 down c26 | +0.020 | 69% | **units(a,b)** (R2 0.80): a%10 in {0,5} -> b%10 in {1,2,3,4,6,7,8,9}; a%10 in {1,6} -> b%10 in {0,1,2,3,5,6,7,8}; a%10 in {2,7} -> b%10 in {0,1,4,5,6,9}; a%10 in {3,8} -> b%10 in {0,3,4,5,8,9}; a%10 in {4,9} -> b%10 in {0,2,3,4,5,7,8,9} |

</details>

<details><summary>period 4: 3 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L22 down c17 | +0.047 | 10% | **res%20** (R2 0.84): res mod 20 in {2..3} |
| L28 down c24 | +0.041 | 12% | **res%100** (R2 0.81): res mod 100 in {6, 15..17, 26, 36, 46, 56, 66, 75..77, 86, 96} [coarser: res mod 20 in {6, 16}, R2 0.80] |
| L27 down c38 | +0.027 | 10% | **res%100** (R2 0.83): res mod 100 in {19, 29, 39, 49, 58..59, 79, 89, 97..99} |

</details>

<details><summary>period 2: 10 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L21 down c1 | +0.155 | 50% | **res%2** (R2 0.99): res mod 2 in {1} |
| L18 down c4 | +0.117 | 41% | **units(a,b)** (R2 0.74): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8}; a%10 in {1,5} -> b%10 in {3,5,7}; a%10 in {3,7,9} -> b%10 in {1,3,5,7,9} |
| L28 down c23 | +0.105 | 41% | **res%2** (R2 0.70): res mod 2 in {0} |
| L20 down c47 | +0.055 | 43% | **res%2** (R2 0.74): res mod 2 in {0} |
| L26 down c36 | +0.054 | 41% | **res%2** (R2 0.68): res mod 2 in {0} |
| L17 down c22 | +0.032 | 25% | **units(a,b)** (R2 0.98): a%10 in {1,3,5,7,9} -> b%10 in {0,2,4,6,8} |
| L17 down c67 | +0.026 | 25% | **units(a,b)** (R2 0.98): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} |
| L25 down c11 | +0.025 | 11% | **res%10** (R2 0.92): res mod 10 in {7} |
| L24 down c4 | +0.024 | 11% | **res%10** (R2 0.86): res mod 10 in {8} |
| L19 down c32 | +0.023 | 26% | **units(a,b)** (R2 0.82): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} |

</details>

</details>

<details><summary>`mlp_in.29`, sub: who writes the a−b code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.81 | L21 MLP +0.09, L28 MLP +0.08, L25 MLP +0.08, L23 MLP +0.07, L24 MLP +0.06, L26 MLP +0.06, L22 MLP +0.06, L27 MLP +0.05, L20 MLP +0.05, L18 MLP +0.05, L19 MLP +0.04, L16 MLP +0.04, L17 MLP +0.02 |
| mod 50 | 0.82 | L28 MLP +0.10, L21 MLP +0.09, L18 MLP +0.09, L19 MLP +0.09, L23 MLP +0.08, L26 MLP +0.06, L24 MLP +0.06, L25 MLP +0.06, L20 MLP +0.06, L27 MLP +0.05, L22 MLP +0.04 |
| mod 25 | 0.77 | L28 MLP +0.14, L25 MLP +0.11, L26 MLP +0.10, L24 MLP +0.10, L27 MLP +0.07, L23 MLP +0.06, L22 MLP +0.05, L21 MLP +0.05, L20 MLP +0.03, L18 MLP +0.02 |
| mod 20 | 0.82 | L20 MLP +0.15, L21 MLP +0.11, L28 MLP +0.08, L24 MLP +0.07, L26 MLP +0.07, L22 MLP +0.07, L25 MLP +0.06, L19 MLP +0.05, L27 MLP +0.05, L18 MLP +0.05, L23 MLP +0.04 |
| mod 10 | 0.86 | L21 MLP +0.14, L28 MLP +0.11, L24 MLP +0.11, L25 MLP +0.09, L20 MLP +0.08, L23 MLP +0.08, L18 MLP +0.08, L17 MLP +0.04, L22 MLP +0.04, L19 MLP +0.03, L26 MLP +0.03 |
| mod 5 | 0.87 | L28 MLP +0.14, L24 MLP +0.13, L25 MLP +0.11, L23 MLP +0.10, L20 MLP +0.10, L21 MLP +0.08, L18 MLP +0.08, L22 MLP +0.04, L16 MLP +0.03, L27 MLP +0.02 |
| mod 4 | 0.61 | L28 MLP +0.22, L26 MLP +0.11, L25 MLP +0.06, L22 MLP +0.06, L24 MLP +0.05, L27 MLP +0.04 |
| mod 2 | 0.79 | L21 MLP +0.19, L18 MLP +0.15, L28 MLP +0.10, L26 MLP +0.09, L23 MLP +0.07, L20 MLP +0.06, L17 MLP +0.06, L25 MLP +0.05, L24 MLP +0.02 |

<details><summary>period 100: 1 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L21 down c0 | +0.024 | 26% | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {2}; a//10 in {1} -> b//10 in {0,10}; a//10 in {2} -> b//10 in {0,1,9,10}; a//10 in {3} -> b//10 in {0,1,2,10}; a//10 in {4} -> b//10 in {1,2,3}; a//10 in {5} -> b//10 in {2,3,4}; a//10 in {6} -> b//10 in {3,4,5}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {5,6,7}; a//10 in {9,10} -> b//10 in {6,7,8} |

</details>

<details><summary>period 50: 5 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L19 down c0 | +0.041 | 60% | **tens(a,b)** (R2 0.54): a//10 in {0,5} -> b//10 in {2,3,7,8}; a//10 in {1} -> b//10 in {3,4,7,8,9}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {1,2,6,7,8}; a//10 in {6} -> b//10 in {2,3,4,7,8}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,10}; a//10 in {9} -> b//10 in {1,2,3,6,7}; a//10 in {10} -> b//10 in {1,2,3,7,8} |
| L21 down c7 | +0.035 | 32% | **res%100** (R2 0.49): res mod 100 in {0..13, 45, 47, 49..53, 55, 90..99} |
| L19 down c11 | +0.026 | 44% | **tens(a,b)** (R2 0.61): a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {0,1,2,3,6,7,8,10}; a//10 in {3} -> b//10 in {1,2,3,7,8}; a//10 in {4} -> b//10 in {2,3,7,8}; a//10 in {5} -> b//10 in {3,4,8}; a//10 in {6} -> b//10 in {4,5,6,8,9}; a//10 in {7} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {1,2,3,6,7,8,9}; a//10 in {9} -> b//10 in {7,8,9}; a//10 in {10} -> b//10 in {3,4,8,9} |
| L20 down c29 | +0.025 | 53% | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {2,6,7,8}; a//10 in {1} -> b//10 in {7,8,9,10}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {1,2,6,7,8}; a//10 in {4,9} -> b//10 in {1,2,3,6,7,8}; a//10 in {5} -> b//10 in {2,3,4,7,8,9}; a//10 in {6} -> b//10 in {0,4,8,9,10}; a//10 in {7} -> b//10 in {0,1,5,6,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,5,6,7,8,10}; a//10 in {10} -> b//10 in {2,3,4,7,8} |
| L18 down c12 | +0.025 | 59% | **tens(a,b)** (R2 0.51): a//10 in {0} -> b//10 in {0,1,4,5,9,10}; a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,8,9}; a//10 in {3} -> b//10 in {0,1,2,6,7,8}; a//10 in {4} -> b//10 in {0,1,2,3,4,6,7,8,9,10}; a//10 in {5,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {0,3,4,5,6,8,9,10}; a//10 in {7} -> b//10 in {0,4,5,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,5,6,7,8,10}; a//10 in {9} -> b//10 in {0,1,2,3,5,6,7,8,9,10} |

</details>

<details><summary>period 25: 3 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L25 down c15 | +0.024 | 8% | **res%100** (R2 0.68): res mod 100 in {1..7} |
| L22 down c6 | +0.023 | 14% | **res%100** (R2 0.74): res mod 100 in {1, 3..14} |
| L25 down c42 | +0.020 | 8% | **res%100** (R2 0.73): res mod 100 in {10..12} |

</details>

<details><summary>period 20: 8 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c6 | +0.068 | 49% | unexplained (best R2 0.33) |
| L20 down c3 | +0.066 | 53% | unexplained (best R2 0.43) |
| L21 down c2 | +0.054 | 28% | **res** (R2 0.55): res in {-10..-4, 10..17, 30..36, 50..56, 70..76, 90..96} |
| L19 down c6 | +0.048 | 66% | unexplained (best R2 0.21) |
| L22 down c11 | +0.045 | 25% | **res** (R2 0.51): res in {-99..-98, -41..-38, -22..-18, 0..3, 18..24, 38..43, 58..62, 78..82, 97..99} |
| L21 down c15 | +0.031 | 29% | unexplained (best R2 0.44) |
| L24 down c43 | +0.029 | 21% | unexplained (best R2 0.42) |
| L18 down c18 | +0.022 | 28% | unexplained (best R2 0.38) |

</details>

<details><summary>period 10: 10 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L21 down c4 | +0.054 | 40% | **units(a,b)** (R2 0.46): a%10 in {0} -> b%10 in {0,7,8,9}; a%10 in {1} -> b%10 in {0,1,8,9}; a%10 in {2} -> b%10 in {0,1,2,8,9}; a%10 in {3} -> b%10 in {0,1,2,3,9}; a%10 in {4} -> b%10 in {1,2,3,4}; a%10 in {5} -> b%10 in {2,3,4,5,7}; a%10 in {6} -> b%10 in {4,5,6}; a%10 in {7} -> b%10 in {4,5,6,7}; a%10 in {8} -> b%10 in {5,6,7,8}; a%10 in {9} -> b%10 in {6,7,8,9} |
| L20 down c13 | +0.042 | 39% | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {2,3,4,5}; a%10 in {1} -> b%10 in {3,4,5,6}; a%10 in {2} -> b%10 in {4,5,6,7}; a%10 in {3} -> b%10 in {5,6,7}; a%10 in {4} -> b%10 in {6,7,8,9}; a%10 in {5} -> b%10 in {0,8,9}; a%10 in {6} -> b%10 in {0,1,8,9}; a%10 in {7} -> b%10 in {0,1,2,3,9}; a%10 in {8} -> b%10 in {0,1,2,3,4}; a%10 in {9} -> b%10 in {1,2,3,4,5} |
| L20 down c21 | +0.035 | 37% | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {0,1,2,3,9}; a%10 in {1} -> b%10 in {0,1,2,3}; a%10 in {2} -> b%10 in {1,2,3,4}; a%10 in {3} -> b%10 in {3,4,5,6}; a%10 in {4} -> b%10 in {4,5,6}; a%10 in {5} -> b%10 in {6,7}; a%10 in {6} -> b%10 in {6,7,8}; a%10 in {7} -> b%10 in {0,7,8,9}; a%10 in {8} -> b%10 in {0,1,8,9}; a%10 in {9} -> b%10 in {0,1,2,8,9} |
| L21 down c10 | +0.034 | 18% | unexplained (best R2 0.50) |
| L25 down c12 | +0.030 | 28% | unexplained (best R2 0.47) |
| L23 down c4 | +0.029 | 16% | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {8}; a%10 in {1} -> b%10 in {9}; a%10 in {2} -> b%10 in {0,2,4,6,8}; a%10 in {3} -> b%10 in {1}; a%10 in {4} -> b%10 in {2}; a%10 in {5} -> b%10 in {3,7}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {5}; a%10 in {8} -> b%10 in {6}; a%10 in {9} -> b%10 in {7} |
| L24 down c4 | +0.029 | 12% | unexplained (best R2 0.50) |
| L24 down c5 | +0.027 | 12% | unexplained (best R2 0.47) |
| L24 down c12 | +0.026 | 11% | **units(a,b)** (R2 0.64): a%10 in {0} -> b%10 in {5}; a%10 in {1} -> b%10 in {6}; a%10 in {2} -> b%10 in {7}; a%10 in {3} -> b%10 in {8}; a%10 in {4} -> b%10 in {9}; a%10 in {5} -> b%10 in {0,5}; a%10 in {6} -> b%10 in {1}; a%10 in {7} -> b%10 in {2}; a%10 in {8} -> b%10 in {3}; a%10 in {9} -> b%10 in {4} |
| L25 down c11 | +0.024 | 11% | **units(a,b)** (R2 0.51): a%10 in {0} -> b%10 in {3}; a%10 in {1} -> b%10 in {4}; a%10 in {2} -> b%10 in {5}; a%10 in {3} -> b%10 in {6}; a%10 in {5} -> b%10 in {8}; a%10 in {6} -> b%10 in {9}; a%10 in {7} -> b%10 in {0}; a%10 in {8} -> b%10 in {1}; a%10 in {9} -> b%10 in {2} |

</details>

<details><summary>period 5: 14 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c9 | +0.054 | 49% | **res** (R2 0.62): res in {-97, -93, -63, -58, -53..-52, -48..-47, -43..-42, -38..-37, -33..-32, -28..-27, -23..-22, -19..-17, -13..-12, -9..-7, -4..-2, 1..3, 6..8, 11..13, 16..18, 21..23, 26..28, 31..33, 36..38, 41..43, 46..48, 51..53, 56..58, 61..63, 66..68, 71..73, 76..78, 81..83, 86..88, 91..93, 96..98} |
| L23 down c4 | +0.038 | 16% | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {8}; a%10 in {1} -> b%10 in {9}; a%10 in {2} -> b%10 in {0,2,4,6,8}; a%10 in {3} -> b%10 in {1}; a%10 in {4} -> b%10 in {2}; a%10 in {5} -> b%10 in {3,7}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {5}; a%10 in {8} -> b%10 in {6}; a%10 in {9} -> b%10 in {7} |
| L24 down c4 | +0.036 | 12% | unexplained (best R2 0.50) |
| L25 down c12 | +0.034 | 28% | unexplained (best R2 0.47) |
| L24 down c5 | +0.034 | 12% | unexplained (best R2 0.47) |
| L21 down c10 | +0.034 | 18% | unexplained (best R2 0.50) |
| L25 down c11 | +0.033 | 11% | **units(a,b)** (R2 0.51): a%10 in {0} -> b%10 in {3}; a%10 in {1} -> b%10 in {4}; a%10 in {2} -> b%10 in {5}; a%10 in {3} -> b%10 in {6}; a%10 in {5} -> b%10 in {8}; a%10 in {6} -> b%10 in {9}; a%10 in {7} -> b%10 in {0}; a%10 in {8} -> b%10 in {1}; a%10 in {9} -> b%10 in {2} |
| L20 down c15 | +0.033 | 46% | unexplained (best R2 0.44) |
| L24 down c12 | +0.033 | 11% | **units(a,b)** (R2 0.64): a%10 in {0} -> b%10 in {5}; a%10 in {1} -> b%10 in {6}; a%10 in {2} -> b%10 in {7}; a%10 in {3} -> b%10 in {8}; a%10 in {4} -> b%10 in {9}; a%10 in {5} -> b%10 in {0,5}; a%10 in {6} -> b%10 in {1}; a%10 in {7} -> b%10 in {2}; a%10 in {8} -> b%10 in {3}; a%10 in {9} -> b%10 in {4} |
| L22 down c8 | +0.024 | 9% | **res%10** (R2 0.52): res mod 10 in {0} |
| L28 down c36 | +0.022 | 13% | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {8}; a%10 in {2} -> b%10 in {0,2}; a%10 in {3} -> b%10 in {1}; a%10 in {4} -> b%10 in {2}; a%10 in {5} -> b%10 in {3}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {5}; a%10 in {8} -> b%10 in {6}; a%10 in {9} -> b%10 in {7} |
| L23 down c9 | +0.022 | 14% | **units(a,b)** (R2 0.53): a%10 in {0} -> b%10 in {4}; a%10 in {1} -> b%10 in {5}; a%10 in {2} -> b%10 in {6}; a%10 in {4} -> b%10 in {0,8}; a%10 in {5} -> b%10 in {9}; a%10 in {6} -> b%10 in {0}; a%10 in {7} -> b%10 in {1}; a%10 in {8} -> b%10 in {2,4}; a%10 in {9} -> b%10 in {3} |
| L18 down c26 | +0.022 | 42% | **res** (R2 0.51): res in {-97, -48, -43..-42, -38..-37, -33, -28..-27, -23..-22, -18..-17, -13..-12, -8..-7, -3..-1, 1..3, 6..8, 11..13, 16..18, 21..23, 26..28, 31..33, 37..38, 42..43, 47..48, 52..53, 57..58, 62..63, 67..68, 72..73, 77..78, 82..83, 87..88, 92..93, 96..98} |
| L28 down c69 | +0.022 | 11% | **res%100** (R2 0.46): res mod 100 in {6, 16, 26, 36, 46, 56, 66, 76, 86, 96} [coarser: res mod 50 in {6, 16, 26, 36, 46}, R2 0.81] |

</details>

<details><summary>period 4: 6 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L22 down c17 | +0.035 | 5% | **res** (R2 0.58): res in {-18, 2..3, 22..23, 42, 82} |
| L26 down c154 | +0.033 | 37% | unexplained (best R2 0.37) |
| L28 down c24 | +0.030 | 15% | unexplained (best R2 0.47) |
| L28 down c51 | +0.024 | 2% | unexplained (best R2 0.40) |
| L26 down c37 | +0.024 | 7% | **res** (R2 0.66): res in {-97, 7..8, 17..18, 27, 37..38, 87, 97..98} |
| L28 down c73 | +0.022 | 3% | **res%100** (R2 0.52): res mod 100 in {10..11} |

</details>

<details><summary>period 2: 8 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L21 down c1 | +0.175 | 57% | **res%2** (R2 0.65): res mod 2 in {1} |
| L18 down c4 | +0.145 | 40% | **units(a,b)** (R2 0.52): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} |
| L26 down c36 | +0.080 | 30% | **units(a,b)** (R2 0.67): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} |
| L20 down c47 | +0.052 | 28% | **units(a,b)** (R2 0.75): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} |
| L28 down c23 | +0.038 | 19% | **units(a,b)** (R2 0.65): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} |
| L17 down c22 | +0.037 | 25% | **units(a,b)** (R2 0.81): a%10 in {1,3,5,7,9} -> b%10 in {0,2,4,6,8} |
| L23 down c55 | +0.033 | 22% | **units(a,b)** (R2 0.81): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} |
| L25 down c12 | +0.031 | 28% | unexplained (best R2 0.47) |

</details>

</details>

<details><summary>`mlp_in.31`, add: who writes the a+b code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.74 | L29 MLP +0.13, L30 MLP +0.10, L28 MLP +0.08, L27 MLP +0.08, L25 MLP +0.06, L26 MLP +0.05, L24 MLP +0.05, L23 MLP +0.05, L21 MLP +0.03, L22 MLP +0.03, L20 MLP +0.02 |
| mod 50 | 0.77 | L29 MLP +0.14, L30 MLP +0.11, L28 MLP +0.09, L27 MLP +0.07, L18 MLP +0.05, L24 MLP +0.05, L26 MLP +0.05, L25 MLP +0.05, L19 MLP +0.05, L23 MLP +0.04, L21 MLP +0.03, L20 MLP +0.03, L22 MLP +0.02 |
| mod 25 | 0.70 | L29 MLP +0.15, L30 MLP +0.13, L28 MLP +0.10, L27 MLP +0.08, L24 MLP +0.07, L26 MLP +0.06, L25 MLP +0.05, L23 MLP +0.03 |
| mod 20 | 0.77 | L29 MLP +0.12, L20 MLP +0.10, L30 MLP +0.10, L28 MLP +0.08, L27 MLP +0.07, L21 MLP +0.06, L26 MLP +0.05, L24 MLP +0.05, L25 MLP +0.05, L22 MLP +0.04, L18 MLP +0.02, L19 MLP +0.02 |
| mod 10 | 0.82 | L28 MLP +0.11, L30 MLP +0.09, L29 MLP +0.09, L24 MLP +0.09, L21 MLP +0.09, L25 MLP +0.07, L23 MLP +0.05, L27 MLP +0.04, L22 MLP +0.04, L26 MLP +0.04, L20 MLP +0.04, L18 MLP +0.03, L19 MLP +0.02 |
| mod 5 | 0.82 | L28 MLP +0.12, L30 MLP +0.10, L24 MLP +0.10, L29 MLP +0.09, L25 MLP +0.08, L20 MLP +0.06, L23 MLP +0.06, L21 MLP +0.05, L18 MLP +0.04, L27 MLP +0.04, L22 MLP +0.04, L26 MLP +0.03 |
| mod 4 | 0.59 | L29 MLP +0.16, L30 MLP +0.11, L28 MLP +0.10, L27 MLP +0.06, L26 MLP +0.05, L25 MLP +0.04, L22 MLP +0.03, L24 MLP +0.02 |
| mod 2 | 0.83 | L28 MLP +0.15, L29 MLP +0.12, L21 MLP +0.11, L30 MLP +0.09, L18 MLP +0.07, L26 MLP +0.05, L25 MLP +0.05, L24 MLP +0.04, L20 MLP +0.04, L17 MLP +0.04, L23 MLP +0.03 |

<details><summary>period 50: 1 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L19 down c0 | +0.023 | 51% | **res%50** (R2 0.84): res mod 50 in {14..39} |

</details>

<details><summary>period 20: 4 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c3 | +0.049 | 61% | **res%20** (R2 0.74): res mod 20 in {8..19} |
| L20 down c6 | +0.043 | 60% | **res%20** (R2 0.60): res mod 20 in {1..11} |
| L21 down c2 | +0.033 | 36% | **res%20** (R2 0.89): res mod 20 in {10..16} |
| L22 down c11 | +0.025 | 33% | **res%20** (R2 0.89): res mod 20 in {0..3, 18..19} |

</details>

<details><summary>period 10: 3 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L24 down c4 | +0.026 | 11% | **res%10** (R2 0.86): res mod 10 in {8} |
| L24 down c5 | +0.025 | 11% | **res%10** (R2 0.90): res mod 10 in {3} |
| L21 down c4 | +0.024 | 41% | **res%10** (R2 0.85): res mod 10 in {0..3} |

</details>

<details><summary>period 5: 7 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L24 down c4 | +0.032 | 11% | **res%10** (R2 0.86): res mod 10 in {8} |
| L24 down c5 | +0.030 | 11% | **res%10** (R2 0.90): res mod 10 in {3} |
| L20 down c9 | +0.028 | 60% | **res%5** (R2 0.95): res mod 5 in {1..3} |
| L20 down c15 | +0.025 | 61% | **res%5** (R2 0.83): res mod 5 in {0..2} |
| L25 down c11 | +0.022 | 11% | **res%10** (R2 0.92): res mod 10 in {7} |
| L30 down c230 | +0.021 | 10% | **res%10** (R2 0.89): res mod 10 in {8} |
| L25 down c12 | +0.021 | 11% | **res%10** (R2 0.86): res mod 10 in {1} |

</details>

<details><summary>period 4: 3 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L29 down c159 | +0.029 | 11% | unexplained (best R2 0.41) |
| L22 down c17 | +0.024 | 10% | **res%20** (R2 0.84): res mod 20 in {2..3} |
| L28 down c24 | +0.020 | 12% | **res%100** (R2 0.81): res mod 100 in {6, 15..17, 26, 36, 46, 56, 66, 75..77, 86, 96} [coarser: res mod 20 in {6, 16}, R2 0.80] |

</details>

<details><summary>period 2: 7 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L21 down c1 | +0.097 | 50% | **res%2** (R2 0.99): res mod 2 in {1} |
| L28 down c23 | +0.086 | 41% | **res%2** (R2 0.70): res mod 2 in {0} |
| L18 down c4 | +0.071 | 41% | **units(a,b)** (R2 0.74): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8}; a%10 in {1,5} -> b%10 in {3,5,7}; a%10 in {3,7,9} -> b%10 in {1,3,5,7,9} |
| L29 down c171 | +0.057 | 33% | **res** (R2 0.81): res in {47, 57, 59, 61, 63, 67, 79, 81, 83, 85, 87, 89, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183, 185, 187, 189, 191, 193, 195, 197} |
| L20 down c47 | +0.036 | 43% | **res%2** (R2 0.74): res mod 2 in {0} |
| L26 down c36 | +0.034 | 41% | **res%2** (R2 0.68): res mod 2 in {0} |
| L30 down c320 | +0.021 | 22% | **res** (R2 0.76): res in {102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 152, 154, 156, 158, 160, 162, 164, 166, 168, 170, 172, 174, 176, 178, 182, 184, 186, 188, 192, 194, 196, 198} |

</details>

</details>

<details><summary>`mlp_in.31`, sub: who writes the a−b code</summary>

| period | total explained | groups supplying ≥ 2 % |
|---|---|---|
| mod 100 | 0.66 | L29 MLP +0.09, L21 MLP +0.05, L28 MLP +0.05, L25 MLP +0.05, L30 MLP +0.05, L23 MLP +0.04, L26 MLP +0.04, L24 MLP +0.04, L22 MLP +0.04, L27 MLP +0.03, L20 MLP +0.03, L18 MLP +0.03, L19 MLP +0.02, L16 MLP +0.02 |
| mod 50 | 0.67 | L29 MLP +0.09, L28 MLP +0.07, L21 MLP +0.06, L18 MLP +0.06, L19 MLP +0.05, L23 MLP +0.05, L26 MLP +0.05, L24 MLP +0.04, L25 MLP +0.04, L20 MLP +0.04, L30 MLP +0.04, L27 MLP +0.04, L22 MLP +0.03 |
| mod 25 | 0.64 | L29 MLP +0.13, L28 MLP +0.08, L25 MLP +0.07, L26 MLP +0.06, L24 MLP +0.06, L30 MLP +0.05, L27 MLP +0.04, L23 MLP +0.03, L22 MLP +0.03, L21 MLP +0.03 |
| mod 20 | 0.67 | L20 MLP +0.10, L29 MLP +0.07, L21 MLP +0.07, L28 MLP +0.06, L24 MLP +0.05, L26 MLP +0.05, L22 MLP +0.05, L25 MLP +0.04, L27 MLP +0.04, L19 MLP +0.03, L30 MLP +0.03, L18 MLP +0.03, L23 MLP +0.03 |
| mod 10 | 0.70 | L21 MLP +0.10, L24 MLP +0.08, L28 MLP +0.08, L25 MLP +0.07, L20 MLP +0.06, L23 MLP +0.06, L18 MLP +0.06, L29 MLP +0.05, L17 MLP +0.03, L30 MLP +0.03, L22 MLP +0.03, L26 MLP +0.02, L19 MLP +0.02 |
| mod 5 | 0.72 | L28 MLP +0.10, L24 MLP +0.10, L25 MLP +0.08, L23 MLP +0.07, L20 MLP +0.07, L21 MLP +0.06, L29 MLP +0.06, L18 MLP +0.05, L22 MLP +0.03, L30 MLP +0.03, L16 MLP +0.02 |
| mod 4 | 0.60 | L29 MLP +0.19, L28 MLP +0.12, L26 MLP +0.07, L30 MLP +0.07, L25 MLP +0.04, L22 MLP +0.03, L24 MLP +0.03, L27 MLP +0.02 |
| mod 2 | 0.64 | L21 MLP +0.13, L18 MLP +0.11, L28 MLP +0.08, L26 MLP +0.07, L23 MLP +0.05, L25 MLP +0.04, L20 MLP +0.04, L17 MLP +0.04, L29 MLP +0.04, L24 MLP +0.02 |

<details><summary>period 50: 2 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L19 down c0 | +0.025 | 60% | **tens(a,b)** (R2 0.54): a//10 in {0,5} -> b//10 in {2,3,7,8}; a//10 in {1} -> b//10 in {3,4,7,8,9}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {1,2,6,7,8}; a//10 in {6} -> b//10 in {2,3,4,7,8}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,10}; a//10 in {9} -> b//10 in {1,2,3,6,7}; a//10 in {10} -> b//10 in {1,2,3,7,8} |
| L21 down c7 | +0.021 | 32% | **res%100** (R2 0.49): res mod 100 in {0..13, 45, 47, 49..53, 55, 90..99} |

</details>

<details><summary>period 20: 6 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c6 | +0.045 | 49% | unexplained (best R2 0.33) |
| L20 down c3 | +0.044 | 53% | unexplained (best R2 0.43) |
| L21 down c2 | +0.036 | 28% | **res** (R2 0.55): res in {-10..-4, 10..17, 30..36, 50..56, 70..76, 90..96} |
| L19 down c6 | +0.032 | 66% | unexplained (best R2 0.21) |
| L22 down c11 | +0.030 | 25% | **res** (R2 0.51): res in {-99..-98, -41..-38, -22..-18, 0..3, 18..24, 38..43, 58..62, 78..82, 97..99} |
| L21 down c15 | +0.020 | 29% | unexplained (best R2 0.44) |

</details>

<details><summary>period 10: 8 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L21 down c4 | +0.038 | 40% | **units(a,b)** (R2 0.46): a%10 in {0} -> b%10 in {0,7,8,9}; a%10 in {1} -> b%10 in {0,1,8,9}; a%10 in {2} -> b%10 in {0,1,2,8,9}; a%10 in {3} -> b%10 in {0,1,2,3,9}; a%10 in {4} -> b%10 in {1,2,3,4}; a%10 in {5} -> b%10 in {2,3,4,5,7}; a%10 in {6} -> b%10 in {4,5,6}; a%10 in {7} -> b%10 in {4,5,6,7}; a%10 in {8} -> b%10 in {5,6,7,8}; a%10 in {9} -> b%10 in {6,7,8,9} |
| L20 down c13 | +0.029 | 39% | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {2,3,4,5}; a%10 in {1} -> b%10 in {3,4,5,6}; a%10 in {2} -> b%10 in {4,5,6,7}; a%10 in {3} -> b%10 in {5,6,7}; a%10 in {4} -> b%10 in {6,7,8,9}; a%10 in {5} -> b%10 in {0,8,9}; a%10 in {6} -> b%10 in {0,1,8,9}; a%10 in {7} -> b%10 in {0,1,2,3,9}; a%10 in {8} -> b%10 in {0,1,2,3,4}; a%10 in {9} -> b%10 in {1,2,3,4,5} |
| L20 down c21 | +0.025 | 37% | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {0,1,2,3,9}; a%10 in {1} -> b%10 in {0,1,2,3}; a%10 in {2} -> b%10 in {1,2,3,4}; a%10 in {3} -> b%10 in {3,4,5,6}; a%10 in {4} -> b%10 in {4,5,6}; a%10 in {5} -> b%10 in {6,7}; a%10 in {6} -> b%10 in {6,7,8}; a%10 in {7} -> b%10 in {0,7,8,9}; a%10 in {8} -> b%10 in {0,1,8,9}; a%10 in {9} -> b%10 in {0,1,2,8,9} |
| L21 down c10 | +0.024 | 18% | unexplained (best R2 0.50) |
| L25 down c12 | +0.023 | 28% | unexplained (best R2 0.47) |
| L24 down c4 | +0.022 | 12% | unexplained (best R2 0.50) |
| L24 down c12 | +0.020 | 11% | **units(a,b)** (R2 0.64): a%10 in {0} -> b%10 in {5}; a%10 in {1} -> b%10 in {6}; a%10 in {2} -> b%10 in {7}; a%10 in {3} -> b%10 in {8}; a%10 in {4} -> b%10 in {9}; a%10 in {5} -> b%10 in {0,5}; a%10 in {6} -> b%10 in {1}; a%10 in {7} -> b%10 in {2}; a%10 in {8} -> b%10 in {3}; a%10 in {9} -> b%10 in {4} |
| L24 down c5 | +0.020 | 12% | unexplained (best R2 0.47) |

</details>

<details><summary>period 5: 9 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L20 down c9 | +0.036 | 49% | **res** (R2 0.62): res in {-97, -93, -63, -58, -53..-52, -48..-47, -43..-42, -38..-37, -33..-32, -28..-27, -23..-22, -19..-17, -13..-12, -9..-7, -4..-2, 1..3, 6..8, 11..13, 16..18, 21..23, 26..28, 31..33, 36..38, 41..43, 46..48, 51..53, 56..58, 61..63, 66..68, 71..73, 76..78, 81..83, 86..88, 91..93, 96..98} |
| L24 down c4 | +0.030 | 12% | unexplained (best R2 0.50) |
| L24 down c5 | +0.026 | 12% | unexplained (best R2 0.47) |
| L25 down c12 | +0.026 | 28% | unexplained (best R2 0.47) |
| L23 down c4 | +0.026 | 16% | **units(a,b)** (R2 0.52): a%10 in {0} -> b%10 in {8}; a%10 in {1} -> b%10 in {9}; a%10 in {2} -> b%10 in {0,2,4,6,8}; a%10 in {3} -> b%10 in {1}; a%10 in {4} -> b%10 in {2}; a%10 in {5} -> b%10 in {3,7}; a%10 in {6} -> b%10 in {4}; a%10 in {7} -> b%10 in {5}; a%10 in {8} -> b%10 in {6}; a%10 in {9} -> b%10 in {7} |
| L21 down c10 | +0.025 | 18% | unexplained (best R2 0.50) |
| L20 down c15 | +0.024 | 46% | unexplained (best R2 0.44) |
| L25 down c11 | +0.023 | 11% | **units(a,b)** (R2 0.51): a%10 in {0} -> b%10 in {3}; a%10 in {1} -> b%10 in {4}; a%10 in {2} -> b%10 in {5}; a%10 in {3} -> b%10 in {6}; a%10 in {5} -> b%10 in {8}; a%10 in {6} -> b%10 in {9}; a%10 in {7} -> b%10 in {0}; a%10 in {8} -> b%10 in {1}; a%10 in {9} -> b%10 in {2} |
| L24 down c12 | +0.023 | 11% | **units(a,b)** (R2 0.64): a%10 in {0} -> b%10 in {5}; a%10 in {1} -> b%10 in {6}; a%10 in {2} -> b%10 in {7}; a%10 in {3} -> b%10 in {8}; a%10 in {4} -> b%10 in {9}; a%10 in {5} -> b%10 in {0,5}; a%10 in {6} -> b%10 in {1}; a%10 in {7} -> b%10 in {2}; a%10 in {8} -> b%10 in {3}; a%10 in {9} -> b%10 in {4} |

</details>

<details><summary>period 4: 3 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L29 down c159 | +0.039 | 19% | unexplained (best R2 0.27) |
| L22 down c17 | +0.020 | 5% | **res** (R2 0.58): res in {-18, 2..3, 22..23, 42, 82} |
| L26 down c154 | +0.020 | 37% | unexplained (best R2 0.37) |

</details>

<details><summary>period 2: 8 writers with |share| ≥ 2 %</summary>

| component | share | on | on-set |
|---|---|---|---|
| L21 down c1 | +0.126 | 57% | **res%2** (R2 0.65): res mod 2 in {1} |
| L18 down c4 | +0.104 | 40% | **units(a,b)** (R2 0.52): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} |
| L26 down c36 | +0.059 | 30% | **units(a,b)** (R2 0.67): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} |
| L20 down c47 | +0.038 | 28% | **units(a,b)** (R2 0.75): a%10 in {1,3,5,7,9} -> b%10 in {1,3,5,7,9} |
| L28 down c23 | +0.033 | 19% | **units(a,b)** (R2 0.65): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} |
| L17 down c22 | +0.026 | 25% | **units(a,b)** (R2 0.81): a%10 in {1,3,5,7,9} -> b%10 in {0,2,4,6,8} |
| L23 down c55 | +0.025 | 22% | **units(a,b)** (R2 0.81): a%10 in {0,2,4,6,8} -> b%10 in {0,2,4,6,8} |
| L25 down c12 | +0.025 | 28% | unexplained (best R2 0.47) |

</details>

</details>

## Tens and carry

<details><summary>all 175</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L15 down c9 | 19% / 0 | **tens(a,b)** (R2 0.62): a//10 in {0} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5} | off (on 0) | a: mod100 +4%, mod50 +4% | - |
| L16 down c10 | 48% / 47% | **tens(a,b)** (R2 0.80): a//10 in {0} -> b//10 in {2,3,5,6,7,8}; a//10 in {1} -> b//10 in {6}; a//10 in {5} -> b//10 in {3,4,5,7,8,9,10}; a//10 in {6,10} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {8,9} -> b//10 in {1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.73): a//10 in {3} -> b//10 in {2,6}; a//10 in {4} -> b//10 in {3,6,7}; a//10 in {5} -> b//10 in {3,4,6,7}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,8,9}; a//10 in {7,8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {1,2,3,4,5,6,7,8,9} | a: mod100 +2%; b: mod100 +6%; res: mod100 +33% | b: mod100 +5%; res: mod100 +4%, mod50 +2% |
| L16 down c18 | 86% / 0 | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {5,6,7,8,9,10}; a//10 in {1} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {3,4,5,9} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {6,7,8,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | off (on 0) | - | same |
| L16 down c25 | 34% / 25% | **tens(a,b)** (R2 0.83): a//10 in {3} -> b//10 in {5,6,7}; a//10 in {4} -> b//10 in {4,5,6,7,8,9}; a//10 in {5,6} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {3,4,5,6,7,8}; a//10 in {8} -> b//10 in {4,5,6,7}; a//10 in {9} -> b//10 in {6} | **tens(a,b)** (R2 0.66): a//10 in {3} -> b//10 in {2}; a//10 in {4,5} -> b//10 in {1,2,3,4,5,6}; a//10 in {6} -> b//10 in {2,3,4,5,6}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {3,4,5} | a: mod100 +3%; b: mod100 +3%; res: mod100 +10% | b: mod100 +2%; res: mod100 +2% |
| L16 down c29 | 50% / 31% | **tens(a,b)** (R2 0.80): a//10 in {0,9,10} -> b//10 in {0,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {8}; a//10 in {5} -> b//10 in {5,6,7}; a//10 in {6} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.74): a//10 in {0} -> b//10 in {0,1,10}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {1}; a//10 in {5} -> b//10 in {3}; a//10 in {6} -> b//10 in {2,3,4,5}; a//10 in {7} -> b//10 in {1,2,3,4,5,6}; a//10 in {8} -> b//10 in {2,3,4,5,6,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | b: mod100 +2%; res: mod100 +19% | - |
| L16 down c43 | 52% / 11% | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {4,5,6,7,8}; a//10 in {1} -> b//10 in {3,4,5,6,7,8}; a//10 in {2} -> b//10 in {2,3,4,5,6,7,8}; a//10 in {3} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {1,2,3,4,5,7,8,9,10}; a//10 in {6} -> b//10 in {1,2,8,9,10}; a//10 in {7} -> b//10 in {1,8}; a//10 in {8} -> b//10 in {6}; a//10 in {9} -> b//10 in {5,6,7}; a//10 in {10} -> b//10 in {5,6} | unexplained (best R2 0.40) | b: mod100 +2%; res: mod100 +9% | - |
| L16 down c49 | 15% / 0 | **tens(a,b)** (R2 0.70): a//10 in {0,3} -> b//10 in {1,2}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2,4,5,6,7,8} -> b//10 in {1} | off (on 0) | - | same |
| L16 down c61 | 40% / 2% | **res//10** (R2 0.78): (tens) res in {2..90} | unexplained (best R2 0.50) | res: mod100 +2% | - |
| L16 down c106 | 35% / 16% | **tens(a,b)** (R2 0.77): a//10 in {0} -> b//10 in {2,3,4}; a//10 in {1,4} -> b//10 in {1,2,3}; a//10 in {2} -> b//10 in {2,3}; a//10 in {5} -> b//10 in {1,2,3,4}; a//10 in {6,7} -> b//10 in {0,1,2,3,4,10}; a//10 in {8} -> b//10 in {0,1,2,3,10}; a//10 in {9} -> b//10 in {1,2} | **tens(a,b)** (R2 0.70): a//10 in {5} -> b//10 in {7,8}; a//10 in {6} -> b//10 in {5,6,7,8}; a//10 in {7} -> b//10 in {0,6,7,8,9}; a//10 in {8} -> b//10 in {0,7,8,9} | a: mod100 +2%; res: mod100 +9% | - |
| L16 down c130 | 25% / 12% | **tens(a,b)** (R2 0.82): a//10 in {0} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {1} -> b//10 in {3,4,5,6,7,8}; a//10 in {2} -> b//10 in {5,6}; a//10 in {8} -> b//10 in {5,6,7,8,9}; a//10 in {9,10} -> b//10 in {4,5,6,7,8,9} | **tens(a,b)** (R2 0.81): a//10 in {8} -> b//10 in {0,1,2,3,4}; a//10 in {9} -> b//10 in {0,1,2,3,4,5}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6} | res: mod100 +6% | - |
| L16 gate c25 | 38% / 27% | **tens(a,b)** (R2 0.77): a//10 in {0,1} -> b//10 in {0,1}; a//10 in {2} -> b//10 in {0,1,2}; a//10 in {3,8} -> b//10 in {5,6,7}; a//10 in {4} -> b//10 in {5,6,7,8}; a//10 in {5} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {6} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {3,4,5,6,7,8}; a//10 in {9} -> b//10 in {6}; a//10 in {10} -> b//10 in {5,6} | **tens(a,b)** (R2 0.67): a//10 in {3} -> b//10 in {2}; a//10 in {4} -> b//10 in {1,2,3,4,5,6}; a//10 in {5} -> b//10 in {1,2,3,4,6}; a//10 in {6} -> b//10 in {1,2,3,4,5,6,7}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {3,4,5} | (reads) | same |
| L16 gate c29 | 40% / 17% | **tens(a,b)** (R2 0.82): a//10 in {0} -> b//10 in {2,3,4,5,6,7,8}; a//10 in {1} -> b//10 in {3,4,5,6,7}; a//10 in {6} -> b//10 in {4,5,6,7}; a//10 in {7} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {8,9,10} -> b//10 in {2,3,4,5,6,7,8,9} | **tens(a,b)** (R2 0.75): a//10 in {6} -> b//10 in {3,4}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {2,3,4,5,6}; a//10 in {9,10} -> b//10 in {1,2,3,4,5,6,7} | (reads) | same |
| L16 gate c43 | 22% / 23% | **tens(a,b)** (R2 0.76): a//10 in {2} -> b//10 in {1}; a//10 in {3,9} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {1,2,3}; a//10 in {5,6} -> b//10 in {0,1,2,3,10}; a//10 in {7} -> b//10 in {0,1,2,10}; a//10 in {8} -> b//10 in {0,1,2} | **tens(a,b)** (R2 0.72): a//10 in {3} -> b//10 in {7}; a//10 in {4} -> b//10 in {6,7,8}; a//10 in {5,6,7,8} -> b//10 in {6,7,8,9,10}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {7,8,9} | (reads) | same |
| L16 gate c493 | 13% / 9% | **tens(a,b)** (R2 0.64): a//10 in {4,5} -> b//10 in {4,5,6,7,8}; a//10 in {6} -> b//10 in {6} | **tens(a,b)** (R2 0.52): a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3,4}; a//10 in {5} -> b//10 in {2,3,4}; a//10 in {10} -> b//10 in {4} | (reads) | same |
| L16 gate c872 | 8% / 10% | **tens(a,b)** (R2 0.62): a//10 in {1} -> b//10 in {5,6,7}; a//10 in {9} -> b//10 in {5,6,7,8}; a//10 in {10} -> b//10 in {4,5,6,7,8} | **tens(a,b)** (R2 0.71): a//10 in {9} -> b//10 in {0,1,2,3,4,5,8}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8,9} | (reads) | same |
| L16 up c29 | 63% / 31% | **tens(a,b)** (R2 0.67): a//10 in {0} -> b//10 in {0,3,4,5,6}; a//10 in {1,2} -> b//10 in {3,4,5,6}; a//10 in {3} -> b//10 in {3,4,5,6,7}; a//10 in {4} -> b//10 in {2,3,4,5,6,7}; a//10 in {5,10} -> b//10 in {2,3,4,5,6,7,8}; a//10 in {6} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {1,2,3,4,5,6,7,8,9}; a//10 in {8,9} -> b//10 in {1,2,3,4,5,6,7,8} | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {0,1,2,10}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {3,4,5,6}; a//10 in {7} -> b//10 in {3,4,5,6,7,10}; a//10 in {8} -> b//10 in {0,3,4,5,6,7,10}; a//10 in {9} -> b//10 in {0,1,3,4,5,6,7,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,10} | (reads) | same |
| L16 up c43 | 50% / 12% | **tens(a,b)** (R2 0.75): a//10 in {0,9} -> b//10 in {2,3,4,5,6,7,8,9}; a//10 in {1,2} -> b//10 in {2,3,4,5,6,7,8}; a//10 in {3,10} -> b//10 in {2,3,4,5,6,7}; a//10 in {4} -> b//10 in {3,4,5,6}; a//10 in {5,7} -> b//10 in {3,4,5}; a//10 in {8} -> b//10 in {3,4,5,6,8} | **tens(a,b)** (R2 0.55): a//10 in {8} -> b//10 in {4,5,6}; a//10 in {9} -> b//10 in {3,4,5,6,7}; a//10 in {10} -> b//10 in {2,3,4,5,6,7} | (reads) | same |
| L16 up c420 | 32% / 1% | **tens(a,b)** (R2 0.73): a//10 in {0} -> b//10 in {2}; a//10 in {1} -> b//10 in {2,3}; a//10 in {2} -> b//10 in {1,3}; a//10 in {3} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {1,2,3}; a//10 in {5,6,9,10} -> b//10 in {1,2,3,4}; a//10 in {7,8} -> b//10 in {0,1,2,3,4} | unexplained (best R2 0.09) | (reads) | same |
| L16 up c757 | 10% / 12% | **tens(a,b)** (R2 0.64): a//10 in {5} -> b//10 in {0}; a//10 in {6} -> b//10 in {0,1,2,6}; a//10 in {7,8} -> b//10 in {0,1} | **tens(a,b)** (R2 0.56): a//10 in {6} -> b//10 in {0,3,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,7,8,10} | (reads) | same |
| L17 down c11 | 46% / 27% | **tens(a,b)** (R2 0.79): a//10 in {2} -> b//10 in {6,7}; a//10 in {3} -> b//10 in {6,7,9,10}; a//10 in {4,6} -> b//10 in {0,1,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {0,5,6,7,8,9,10}; a//10 in {9,10} -> b//10 in {4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.61): a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {10}; a//10 in {5} -> b//10 in {0,1,2,8,9,10}; a//10 in {6} -> b//10 in {0,1,2,9,10}; a//10 in {7} -> b//10 in {0,1,2,3,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,10}; a//10 in {9} -> b//10 in {1,3}; a//10 in {10} -> b//10 in {1,2,3,4,5,10} | a: mod100 +9%; b: mod100 +6%; res: mod100 +22%, mod50 +5% | - |
| L17 down c15 | 44% / 9% | **tens(a,b)** (R2 0.62): a//10 in {3} -> b//10 in {6}; a//10 in {4} -> b//10 in {1,2,3,5,6,7}; a//10 in {5} -> b//10 in {2,3,6}; a//10 in {6,9} -> b//10 in {1,2,3,4,5,6,7}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,7,10}; a//10 in {10} -> b//10 in {1,2,3,4,5,6,7,8} | **tens(a,b)** (R2 0.67): a//10 in {7} -> b//10 in {0,1,2,3,4,5}; a//10 in {8} -> b//10 in {3}; a//10 in {10} -> b//10 in {0,1,2,3,4} | a: mod100 +3% | - |
| L17 down c23 | 65% / 22% | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {1,4,8,9}; a//10 in {1} -> b//10 in {1,3,4,5,6,8,9,10}; a//10 in {2,3,6} -> b//10 in {1,2,3,4,5,6,8,9,10}; a//10 in {4} -> b//10 in {1,3,8,9}; a//10 in {5} -> b//10 in {1,3,4,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,8,9,10}; a//10 in {9} -> b//10 in {1,3,4,6,8}; a//10 in {10} -> b//10 in {3,4,8,9} | **tens(a,b)** (R2 0.58): a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {1,3}; a//10 in {6} -> b//10 in {0,5}; a//10 in {7} -> b//10 in {0,1,3,4,5,6}; a//10 in {8} -> b//10 in {3,4,5,6,8,9} | b: mod50 +13%; res: mod50 +48% | b: mod50 +2% |
| L17 down c40 | 47% / 15% | **tens(a,b)** (R2 0.78): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7}; a//10 in {1,2} -> b//10 in {2,3,4,5,6,7}; a//10 in {3,4} -> b//10 in {1,2,3,4,5,6}; a//10 in {5} -> b//10 in {1,2,3,4,5}; a//10 in {6} -> b//10 in {1,2,3,4}; a//10 in {8} -> b//10 in {2}; a//10 in {9,10} -> b//10 in {0,1,2,3,4} | unexplained (best R2 0.44) | res: mod100 +10% | - |
| L17 down c107 | 24% / 15% | **tens(a,b)** (R2 0.81): a//10 in {0,1} -> b//10 in {2,3,4,5}; a//10 in {2} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {4,5}; a//10 in {7} -> b//10 in {4,5,6}; a//10 in {8,9} -> b//10 in {3,4,5,6}; a//10 in {10} -> b//10 in {2,3,4,5,6} | **tens(a,b)** (R2 0.76): a//10 in {6} -> b//10 in {4,5}; a//10 in {7,8} -> b//10 in {3,4,5,6}; a//10 in {9,10} -> b//10 in {3,4,5,6,7} | res: mod100 +2% | - |
| L17 down c119 | 28% / 9% | **tens(a,b)** (R2 0.60): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1,2} -> b//10 in {0,1,2,3,10}; a//10 in {3} -> b//10 in {0,3}; a//10 in {4} -> b//10 in {0,4}; a//10 in {5,6,7,8} -> b//10 in {0}; a//10 in {9} -> b//10 in {0,9}; a//10 in {10} -> b//10 in {0,10} | unexplained (best R2 0.41) | a: mod100 +2%, mod25 +2% | res: mod25 +3%, mod20 +3% |
| L17 down c122 | 19% / 0 | **tens(a,b)** (R2 0.72): a//10 in {2,3,4,7,8} -> b//10 in {6,7}; a//10 in {5,6} -> b//10 in {5,6,7} | off (on 0) | - | same |
| L17 down c235 | 17% / 0 | **tens(a,b)** (R2 0.80): a//10 in {0,1,2} -> b//10 in {6,7,8,9,10}; a//10 in {3} -> b//10 in {6,7}; a//10 in {9} -> b//10 in {9,10}; a//10 in {10} -> b//10 in {9} | off (on 0) | res: mod100 +2% | - |
| L17 gate c23 | 7% / 5% | **tens(a,b)** (R2 0.57): a//10 in {2,3,8} -> b//10 in {1}; a//10 in {7} -> b//10 in {1,6,9,10} | unexplained (best R2 0.36) | (reads) | same |
| L17 gate c40 | 21% / 2% | **tens(a,b)** (R2 0.67): a//10 in {0,3,4,5,9,10} -> b//10 in {2,3,4}; a//10 in {1,2} -> b//10 in {3,4}; a//10 in {6,8} -> b//10 in {3} | unexplained (best R2 0.43) | (reads) | same |
| L17 up c11 | 57% / 31% | **tens(a,b)** (R2 0.78): a//10 in {0} -> b//10 in {8}; a//10 in {2} -> b//10 in {10}; a//10 in {3} -> b//10 in {1,9,10}; a//10 in {4} -> b//10 in {0,1,2,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,6,7,8,9,10}; a//10 in {6} -> b//10 in {0,1,2,4,5,6,7,8,9,10}; a//10 in {7,8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {9,10} -> b//10 in {1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.69): a//10 in {2,3} -> b//10 in {2}; a//10 in {4} -> b//10 in {0,10}; a//10 in {5} -> b//10 in {0,1,2,8,9,10}; a//10 in {6} -> b//10 in {0,1,2,3,9,10}; a//10 in {7,8} -> b//10 in {0,1,2,3,4,10}; a//10 in {9} -> b//10 in {0,1,2,3,4}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,10} | (reads) | same |
| L17 up c33 | 63% / 24% | **tens(a,b)** (R2 0.69): a//10 in {1,5} -> b//10 in {6,7}; a//10 in {2,3,7,8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {5,6,7}; a//10 in {9} -> b//10 in {1,2,3,4,5,6,7,8,10}; a//10 in {10} -> b//10 in {1,2,5,6,7} | **tens(a,b)** (R2 0.58): a//10 in {2,3} -> b//10 in {1,2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {7} -> b//10 in {3,4,5,6,7,8}; a//10 in {8} -> b//10 in {0,3,4,5,6,7,8,9}; a//10 in {9,10} -> b//10 in {8} | (reads) | same |
| L17 up c40 | 41% / 17% | **tens(a,b)** (R2 0.74): a//10 in {0} -> b//10 in {0,1,2,3,4,5,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,3,4,9,10}; a//10 in {2} -> b//10 in {0,1,2,3,9,10}; a//10 in {3} -> b//10 in {0,1,2,10}; a//10 in {4} -> b//10 in {0}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,9,10} | **tens(a,b)** (R2 0.67): a//10 in {0} -> b//10 in {0}; a//10 in {7} -> b//10 in {6}; a//10 in {8,9} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,4,5,6,7,8,9,10} | (reads) | same |
| L17 up c193 | 32% / 11% | **tens(a,b)** (R2 0.63): a//10 in {1} -> b//10 in {3,4,5,8}; a//10 in {2} -> b//10 in {3,4,5,8,9}; a//10 in {3} -> b//10 in {3,4,8}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6,8} -> b//10 in {2,3,4,5}; a//10 in {7} -> b//10 in {2,3,4,5,9}; a//10 in {9} -> b//10 in {2,3,4}; a//10 in {10} -> b//10 in {3,4,5} | **tens(a,b)** (R2 0.62): a//10 in {2} -> b//10 in {0,1}; a//10 in {6} -> b//10 in {5}; a//10 in {7} -> b//10 in {0,5,6}; a//10 in {8,10} -> b//10 in {5,6,7} | (reads) | same |
| L18 down c21 | 70% / 36% | **tens(a,b)** (R2 0.57): a//10 in {0} -> b//10 in {1,2,3,4,6,7,8,9}; a//10 in {1,6} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,1,2,5,6,7,10}; a//10 in {3} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {4} -> b//10 in {2,3,4,8,9,10}; a//10 in {5} -> b//10 in {1,2,3,4,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,4,5,6,7,9,10}; a//10 in {8} -> b//10 in {0,1,4,5,9,10}; a//10 in {9} -> b//10 in {3,4,8,9,10}; a//10 in {10} -> b//10 in {2,3,4,7,8,9} | **tens(a,b)** (R2 0.57): a//10 in {0} -> b//10 in {1}; a//10 in {1} -> b//10 in {1,2,6,7}; a//10 in {2} -> b//10 in {2}; a//10 in {3} -> b//10 in {6}; a//10 in {4,9} -> b//10 in {1,6,7}; a//10 in {5} -> b//10 in {1,2,3,6,7}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,8,10}; a//10 in {7} -> b//10 in {3,4,7,8,10}; a//10 in {8} -> b//10 in {5}; a//10 in {10} -> b//10 in {0,1,2,5,6,7} | res: mod50 +18% | res: mod50 +5% |
| L18 down c23 | 52% / 35% | **res//10** (R2 0.71): (tens) res in {2..29, 31, 83..130, 179..200} [coarser: res mod 100 in {0..31, 81..99}, R2 0.99] | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {0,9,10}; a//10 in {1} -> b//10 in {0,1,10}; a//10 in {2} -> b//10 in {0,1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3,4,6}; a//10 in {5} -> b//10 in {3,4,5,6}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9}; a//10 in {9} -> b//10 in {0,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,8,9,10} | res: mod100 +22% | res: mod100 +6%, mod50 +5% |
| L18 down c25 | 78% / 32% | **tens(a,b)** (R2 0.52): a//10 in {0} -> b//10 in {3,4,5,7,8,9,10}; a//10 in {1} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {2} -> b//10 in {1,2,3,4,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,2,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,4,5,6,8,9,10}; a//10 in {5} -> b//10 in {0,3,4,5,7,8,9,10}; a//10 in {6} -> b//10 in {0,2,3,4,7,8,9,10}; a//10 in {7,8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {0,1,3,4,5,6,8,9,10}; a//10 in {10} -> b//10 in {0,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.54): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {0,1}; a//10 in {2} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {0,5,10}; a//10 in {5} -> b//10 in {0,1,5,6,9,10}; a//10 in {6} -> b//10 in {0,1,2,5,6,7,9,10}; a//10 in {7} -> b//10 in {1,2,3,6,7}; a//10 in {8} -> b//10 in {4}; a//10 in {9} -> b//10 in {0,4,5,9}; a//10 in {10} -> b//10 in {0,1,4,5,6,9,10} | a: mod50 +3%; b: mod50 +2%; res: mod50 +14% | res: mod50 +5% |
| L18 down c27 | 33% / 23% | **tens(a,b)** (R2 0.69): a//10 in {2} -> b//10 in {4,8,9,10}; a//10 in {3} -> b//10 in {3,4,7,8,9,10}; a//10 in {4} -> b//10 in {2,3,7,8,9}; a//10 in {5} -> b//10 in {8}; a//10 in {7} -> b//10 in {3,4,5,8,9,10}; a//10 in {8} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {9} -> b//10 in {2,3,4,7,8}; a//10 in {10} -> b//10 in {2,3,7,8} | **tens(a,b)** (R2 0.62): a//10 in {2} -> b//10 in {0,1,10}; a//10 in {3} -> b//10 in {0,1,2,6,7}; a//10 in {4} -> b//10 in {1,2,7}; a//10 in {7} -> b//10 in {0,5,6,10}; a//10 in {8} -> b//10 in {0,1,2,5,6,7,10}; a//10 in {9,10} -> b//10 in {1,2,6,7} | a: mod50 +3%; b: mod50 +2%; res: mod50 +9% | res: mod50 +4% |
| L18 down c30 | 26% / 9% | **tens(a,b)** (R2 0.51): a//10 in {2,4} -> b//10 in {5}; a//10 in {3} -> b//10 in {0,5,10}; a//10 in {6} -> b//10 in {0,4,5}; a//10 in {7,8} -> b//10 in {0,3,4,5,6,10}; a//10 in {9} -> b//10 in {3,4,5}; a//10 in {10} -> b//10 in {0,3,4,5} | **tens(a,b)** (R2 0.55): a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {4,5} | - | same |
| L18 down c36 | 58% / 23% | **tens(a,b)** (R2 0.54): a//10 in {0} -> b//10 in {4,5,6,10}; a//10 in {1} -> b//10 in {4,5,10}; a//10 in {2,3} -> b//10 in {1,2,3,6,7}; a//10 in {4,9} -> b//10 in {0,1,2,5,6,7,10}; a//10 in {5} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {6} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {7,8} -> b//10 in {1,2,3,5,6,7,8}; a//10 in {10} -> b//10 in {0,1,2,4,5,6,7,10} | **tens(a,b)** (R2 0.55): a//10 in {3} -> b//10 in {2,3,8}; a//10 in {4} -> b//10 in {3,8}; a//10 in {5} -> b//10 in {0,4,5,8,9,10}; a//10 in {6} -> b//10 in {0,9,10}; a//10 in {8} -> b//10 in {3,4,7,8}; a//10 in {9} -> b//10 in {3,4,7,8,9}; a//10 in {10} -> b//10 in {0,3,4,5,7,8,9,10} | b: mod50 +2%, mod25 +3%; res: mod50 +8% | res: mod50 +5% |
| L18 down c54 | 2% / 0 | **tens(a,b)** (R2 0.71): a//10 in {0,1,2,3,4,5,6,7,8,9} -> b//10 in {10}; a//10 in {10} -> b//10 in {5,6,7,10} | off (on 0) | - | same |
| L18 down c61 | 33% / 13% | **tens(a,b)** (R2 0.65): a//10 in {0} -> b//10 in {0,1,4,5,6,7,10}; a//10 in {1,6} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {2} -> b//10 in {0,4,5,10}; a//10 in {4} -> b//10 in {0,1,6}; a//10 in {5,10} -> b//10 in {0,1,5,6,10}; a//10 in {7} -> b//10 in {0,9,10}; a//10 in {9} -> b//10 in {1,6} | **tens(a,b)** (R2 0.61): a//10 in {5} -> b//10 in {3,4,5,8,9,10}; a//10 in {6} -> b//10 in {0,4,5,9,10}; a//10 in {10} -> b//10 in {3,4,8,9} | res: mod50 +6% | - |
| L18 down c66 | 32% / 37% | **tens(a,b)** (R2 0.77): a//10 in {0} -> b//10 in {0,1,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,8,9,10}; a//10 in {2} -> b//10 in {9,10}; a//10 in {3,4,5,6} -> b//10 in {10}; a//10 in {7} -> b//10 in {0,9,10}; a//10 in {8} -> b//10 in {0,1,2,7,8,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.71): a//10 in {0,1} -> b//10 in {0,1,2,7,8,9,10}; a//10 in {2} -> b//10 in {8,9,10}; a//10 in {3,4,5} -> b//10 in {9}; a//10 in {6,7} -> b//10 in {8,9}; a//10 in {8} -> b//10 in {0,1,2,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,8,9,10} | - | same |
| L18 down c79 | 7% / 0 | **tens(a,b)** (R2 0.84): a//10 in {2,3,4,5,6,7,8,9,10} -> b//10 in {0} | off (on 0) | - | same |
| L18 down c91 | 27% / 3% | **tens(a,b)** (R2 0.77): a//10 in {0} -> b//10 in {3,4}; a//10 in {1} -> b//10 in {2,3,4,5}; a//10 in {2} -> b//10 in {1,2,3,4,5,6}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6}; a//10 in {4} -> b//10 in {1,2,3,4,5}; a//10 in {5} -> b//10 in {2,3,4} | unexplained (best R2 0.22) | res: mod100 +2% | - |
| L18 down c111 | 13% / 36% | **tens(a,b)** (R2 0.58): a//10 in {0} -> b//10 in {3}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {0,1,2,3}; a//10 in {4} -> b//10 in {0,1,2}; a//10 in {5} -> b//10 in {0} | **tens(a,b)** (R2 0.70): a//10 in {0} -> b//10 in {2,3,4,5,6}; a//10 in {1} -> b//10 in {3,4,5}; a//10 in {2} -> b//10 in {0,3,4,5}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,7,9,10}; a//10 in {4} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,5,6}; a//10 in {6} -> b//10 in {2} | - | same |
| L18 down c118 | 6% / 48% | **tens(a,b)** (R2 0.56): a//10 in {9} -> b//10 in {0,1,2,7,8,9}; a//10 in {10} -> b//10 in {1,2} | **tens(a,b)** (R2 0.59): a//10 in {0} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {5,6,7,8,9}; a//10 in {2} -> b//10 in {7,8,9,10}; a//10 in {3} -> b//10 in {8,9}; a//10 in {4} -> b//10 in {0,1,2,8,9}; a//10 in {5} -> b//10 in {0,1,2,9}; a//10 in {6} -> b//10 in {2}; a//10 in {7} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1,2,3,4,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,7,10} | - | a: mod50 +2% |
| L18 down c132 | 34% / 0 | **tens(a,b)** (R2 0.89): a//10 in {3} -> b//10 in {9}; a//10 in {4} -> b//10 in {6,7,8,9,10}; a//10 in {5,6} -> b//10 in {5,6,7,8,9,10}; a//10 in {7,8,9,10} -> b//10 in {4,5,6,7,8,9,10} | off (on 0) | a: mod100 +2%; b: mod100 +2%; res: mod100 +3% | - |
| L18 down c147 | 16% / 29% | **tens(a,b)** (R2 0.80): a//10 in {0} -> b//10 in {0,1,2,3,10}; a//10 in {1} -> b//10 in {0,1,2,3,9,10}; a//10 in {2} -> b//10 in {0,1,2,9,10}; a//10 in {3} -> b//10 in {0,1}; a//10 in {10} -> b//10 in {1,10} | **tens(a,b)** (R2 0.82): a//10 in {0,4} -> b//10 in {0,1,2,3,4}; a//10 in {1} -> b//10 in {0,1,2,3,4,8,9,10}; a//10 in {2} -> b//10 in {0,1,2,3,4,9,10}; a//10 in {3} -> b//10 in {0,1,2,3,4,10}; a//10 in {10} -> b//10 in {7,8,9,10} | - | same |
| L18 down c196 | 9% / 0 | **tens(a,b)** (R2 0.71): a//10 in {0,1} -> b//10 in {7,8}; a//10 in {2} -> b//10 in {6,7,8}; a//10 in {3} -> b//10 in {6,7} | off (on 0) | - | same |
| L18 down c625 | 15% / 1% | **tens(a,b)** (R2 0.60): a//10 in {0,1,2} -> b//10 in {10}; a//10 in {3,4,5,6,7} -> b//10 in {9,10}; a//10 in {8} -> b//10 in {8,9,10}; a//10 in {9,10} -> b//10 in {4,5,6,7,8,9,10} | unexplained (best R2 0.49) | - | same |
| L18 gate c23 | 57% / 49% | **tens(a,b)** (R2 0.76): a//10 in {0,1} -> b//10 in {3,4,5,6}; a//10 in {2} -> b//10 in {4,5}; a//10 in {3} -> b//10 in {0,1,4,5,6,10}; a//10 in {4,5} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {0,1,3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {4,5,6,7}; a//10 in {9,10} -> b//10 in {3,4,5,6,7} | **tens(a,b)** (R2 0.70): a//10 in {0,1,2} -> b//10 in {4,5}; a//10 in {3} -> b//10 in {0,3,4,5,6,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,3,4,5,6,8,9,10}; a//10 in {6} -> b//10 in {0,1,2,4,5,9,10}; a//10 in {7} -> b//10 in {1,2,3,4,5}; a//10 in {8} -> b//10 in {3,4,5}; a//10 in {9,10} -> b//10 in {3,4,5,6} | (reads) | same |
| L18 gate c25 | 55% / 19% | **tens(a,b)** (R2 0.65): a//10 in {0} -> b//10 in {2,3,4,7,8,9}; a//10 in {1,6} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2,7} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {8}; a//10 in {4} -> b//10 in {9}; a//10 in {5} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {9} -> b//10 in {10}; a//10 in {10} -> b//10 in {3,4,7,8,9,10} | **tens(a,b)** (R2 0.55): a//10 in {1} -> b//10 in {0,1}; a//10 in {2} -> b//10 in {1,2}; a//10 in {5} -> b//10 in {1,6}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,10}; a//10 in {7} -> b//10 in {5,6,7}; a//10 in {10} -> b//10 in {1,5,6,7} | (reads) | same |
| L18 gate c27 | 44% / 13% | **tens(a,b)** (R2 0.69): a//10 in {0,1} -> b//10 in {8}; a//10 in {2} -> b//10 in {3,4,7,8,9,10}; a//10 in {3} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {4,9} -> b//10 in {2,3,4,7,8,9}; a//10 in {5,6} -> b//10 in {3,8,9}; a//10 in {7} -> b//10 in {3,4,5,7,8,9,10}; a//10 in {8} -> b//10 in {2,3,4,5,7,8,9,10}; a//10 in {10} -> b//10 in {2,3,4,7,8} | **tens(a,b)** (R2 0.60): a//10 in {2,3} -> b//10 in {1}; a//10 in {7} -> b//10 in {6}; a//10 in {8} -> b//10 in {0,1,2,5,6,7}; a//10 in {9} -> b//10 in {6,7} | (reads) | same |
| L18 gate c36 | 45% / 25% | **tens(a,b)** (R2 0.66): a//10 in {0,6,7} -> b//10 in {1,2,6,7}; a//10 in {1} -> b//10 in {6,7}; a//10 in {2} -> b//10 in {1,6,7}; a//10 in {3} -> b//10 in {1,2,6,7,8}; a//10 in {4,9,10} -> b//10 in {1,2,5,6,7,8,10}; a//10 in {5} -> b//10 in {1,2,5,6,7}; a//10 in {8} -> b//10 in {1,2,3,5,6,7,8} | **tens(a,b)** (R2 0.59): a//10 in {3,4} -> b//10 in {2,3,7,8}; a//10 in {5} -> b//10 in {3,4,7,8}; a//10 in {6,7} -> b//10 in {8}; a//10 in {8} -> b//10 in {2,3,7,8,9}; a//10 in {9} -> b//10 in {3,7,8,9,10}; a//10 in {10} -> b//10 in {2,3,4,5,6,7,8,9,10} | (reads) | same |
| L18 gate c61 | 51% / 26% | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1,3} -> b//10 in {0,1,5,6,9,10}; a//10 in {2} -> b//10 in {0,1,10}; a//10 in {4,5} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {6,8} -> b//10 in {0,1,5,6,10}; a//10 in {7} -> b//10 in {0,10}; a//10 in {9,10} -> b//10 in {0,1,3,4,5,6,9,10} | **tens(a,b)** (R2 0.60): a//10 in {0} -> b//10 in {0,9,10}; a//10 in {3} -> b//10 in {0,9}; a//10 in {4,5} -> b//10 in {0,3,4,5,9,10}; a//10 in {6} -> b//10 in {4,9,10}; a//10 in {8} -> b//10 in {0,4,5,9}; a//10 in {9} -> b//10 in {0,4,5,8,9,10}; a//10 in {10} -> b//10 in {0,3,4,5,6,8,9,10} | (reads) | same |
| L18 gate c66 | 31% / 35% | **tens(a,b)** (R2 0.80): a//10 in {0} -> b//10 in {0,1,8,9,10}; a//10 in {1} -> b//10 in {0,1,7,8,9,10}; a//10 in {2} -> b//10 in {8,9,10}; a//10 in {3,4,5,6,7} -> b//10 in {9,10}; a//10 in {8} -> b//10 in {0,1,2,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.75): a//10 in {0} -> b//10 in {0,1,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,7,8,9,10}; a//10 in {2,5} -> b//10 in {9,10}; a//10 in {3,4} -> b//10 in {9}; a//10 in {6,7} -> b//10 in {8,9}; a//10 in {8} -> b//10 in {0,1,2,8,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | (reads) | same |
| L18 gate c91 | 30% / 13% | **tens(a,b)** (R2 0.70): a//10 in {0,10} -> b//10 in {3}; a//10 in {2} -> b//10 in {0,3,4,9}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,3,4,8,9}; a//10 in {7,8,9} -> b//10 in {3,4} | **tens(a,b)** (R2 0.55): a//10 in {2} -> b//10 in {0}; a//10 in {3} -> b//10 in {0,1,2,3,6,10}; a//10 in {4} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {5,6}; a//10 in {9,10} -> b//10 in {6} | (reads) | same |
| L18 gate c111 | 17% / 11% | **tens(a,b)** (R2 0.74): a//10 in {0} -> b//10 in {3,4}; a//10 in {2,3,4} -> b//10 in {0,1,2,3,4} | **tens(a,b)** (R2 0.50): a//10 in {0} -> b//10 in {4}; a//10 in {3} -> b//10 in {0,1,3,4,10}; a//10 in {4} -> b//10 in {0,1,2,4,10} | (reads) | same |
| L18 gate c147 | 19% / 27% | **tens(a,b)** (R2 0.82): a//10 in {0,3} -> b//10 in {0,1,2,3}; a//10 in {1} -> b//10 in {0,1,2,3,10}; a//10 in {2} -> b//10 in {0,1,2,3,9,10}; a//10 in {4} -> b//10 in {0} | **tens(a,b)** (R2 0.84): a//10 in {0,4} -> b//10 in {0,1,2,3,4}; a//10 in {1,2} -> b//10 in {0,1,2,3,4,9,10}; a//10 in {3} -> b//10 in {0,1,2,3,4,10} | (reads) | same |
| L18 gate c543 | 16% / 6% | **tens(a,b)** (R2 0.65): a//10 in {0,1,2} -> b//10 in {10}; a//10 in {3,6,7} -> b//10 in {9,10}; a//10 in {4,5} -> b//10 in {8,9,10}; a//10 in {8} -> b//10 in {4,9,10}; a//10 in {9,10} -> b//10 in {4,8,9,10} | unexplained (best R2 0.46) | (reads) | same |
| L18 gate c677 | 13% / 7% | **tens(a,b)** (R2 0.60): a//10 in {3} -> b//10 in {1,2,3,7,8}; a//10 in {4} -> b//10 in {1,2,3,6,7}; a//10 in {8} -> b//10 in {2,7}; a//10 in {9} -> b//10 in {2} | **tens(a,b)** (R2 0.54): a//10 in {3} -> b//10 in {1,2}; a//10 in {4} -> b//10 in {3}; a//10 in {8} -> b//10 in {6,7}; a//10 in {9} -> b//10 in {7} | (reads) | same |
| L18 o c66 (H30) | 30% / 36% | **tens(a,b)** (R2 0.77): a//10 in {0} -> b//10 in {0,1,2,3,4}; a//10 in {1} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,1,2,3,4,10}; a//10 in {3,4} -> b//10 in {0,1,2}; a//10 in {5} -> b//10 in {1,2}; a//10 in {10} -> b//10 in {1} | **tens(a,b)** (R2 0.82): a//10 in {0,3} -> b//10 in {0,1,2,3,4}; a//10 in {1} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,1,2,3,4,5,6,7,10}; a//10 in {4} -> b//10 in {0,1,2}; a//10 in {5} -> b//10 in {1,2}; a//10 in {6,7,8} -> b//10 in {1} | - | same |
| L18 o c73 (H30) | 34% / 33% | **tens(a,b)** (R2 0.66): a//10 in {0,1,2} -> b//10 in {5,6,7}; a//10 in {3,4,8,9,10} -> b//10 in {6}; a//10 in {5} -> b//10 in {0,5,6,7}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,6,7} | **tens(a,b)** (R2 0.66): a//10 in {0,1,5} -> b//10 in {5,6,7}; a//10 in {2,3,4} -> b//10 in {6,7}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,2,3,6,7}; a//10 in {9,10} -> b//10 in {6} | - | same |
| L18 o c82 (H30) | 18% / 27% | **tens(a,b)** (R2 0.71): a//10 in {0,1,2,5,9,10} -> b//10 in {3}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,10}; a//10 in {4} -> b//10 in {0,1,3} | **tens(a,b)** (R2 0.73): a//10 in {0,1,2} -> b//10 in {3,4}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,3,4,5,9}; a//10 in {5,6,7} -> b//10 in {3} | - | same |
| L18 o c107 (H30) | 44% / 42% | **tens(a,b)** (R2 0.83): a//10 in {0,1,7} -> b//10 in {7,8,9,10}; a//10 in {2,3,4,5,6} -> b//10 in {8,9,10}; a//10 in {8,9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.74): a//10 in {0,1,2,3} -> b//10 in {7,8,9,10}; a//10 in {4,5,6,7} -> b//10 in {8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,7,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,10} | - | b: mod100 +3%, mod50 +2% |
| L18 o c120 (H30) | 2% / 3% | **tens(a,b)** (R2 0.62): a//10 in {0,1,2,3,4,5,6,7,8,9} -> b//10 in {10}; a//10 in {10} -> b//10 in {0,10} | unexplained (best R2 0.49) | - | same |
| L18 o c137 (H30) | 13% / 11% | **tens(a,b)** (R2 0.63): a//10 in {0,3} -> b//10 in {3,4}; a//10 in {1,2} -> b//10 in {4}; a//10 in {4} -> b//10 in {0,1,2,3,4,5} | **tens(a,b)** (R2 0.57): a//10 in {0,1,2,3,8} -> b//10 in {4}; a//10 in {4} -> b//10 in {0,1,2,3,4,7,8,9,10} | - | same |
| L18 o c166 (H18) | 18% / 19% | **tens(a,b)** (R2 0.73): a//10 in {0} -> b//10 in {4,5}; a//10 in {1,2,3,6,9,10} -> b//10 in {5}; a//10 in {4} -> b//10 in {0,4,5}; a//10 in {5} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.74): a//10 in {0,1,2,3,4,6} -> b//10 in {5}; a//10 in {5} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {4,5} | - | same |
| L18 o c324 (H18) | 21% / 18% | **tens(a,b)** (R2 0.57): a//10 in {0,1,2,3,4,5,6,8} -> b//10 in {9,10}; a//10 in {7} -> b//10 in {10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.55): a//10 in {0,4,5,6,8} -> b//10 in {9,10}; a//10 in {1,2,3} -> b//10 in {10}; a//10 in {7} -> b//10 in {9}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | - | same |
| L18 o c355 (H30) | 4% / 13% | **tens(a,b)** (R2 0.59): a//10 in {0} -> b//10 in {7}; a//10 in {7} -> b//10 in {0,1,7} | **tens(a,b)** (R2 0.70): a//10 in {0,1,2,3,4,5} -> b//10 in {7}; a//10 in {7} -> b//10 in {0,1,2,3,4,7,10} | - | same |
| L18 o c402 (H30) | 11% / 15% | **tens(a,b)** (R2 0.52): a//10 in {0,1,2,3,5} -> b//10 in {4}; a//10 in {4} -> b//10 in {0,1,2,4} | unexplained (best R2 0.49) | - | same |
| L18 o c427 (H30) | 5% / 6% | **tens(a,b)** (R2 0.51): a//10 in {0,1} -> b//10 in {2}; a//10 in {2} -> b//10 in {0,1,2} | unexplained (best R2 0.37) | - | same |
| L18 up c21 | 50% / 34% | **tens(a,b)** (R2 0.71): a//10 in {0,1,5,6,10} -> b//10 in {1,2,3,6,7,8}; a//10 in {2} -> b//10 in {1,2,3,6,7}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4,9} -> b//10 in {2,3,7,8}; a//10 in {7} -> b//10 in {1,2,6,7}; a//10 in {8} -> b//10 in {2} | **tens(a,b)** (R2 0.65): a//10 in {1} -> b//10 in {1,2,3}; a//10 in {2} -> b//10 in {2}; a//10 in {3} -> b//10 in {6,7}; a//10 in {4,9} -> b//10 in {1,2,3,6,7}; a//10 in {5,6,10} -> b//10 in {1,2,3,6,7,8}; a//10 in {7} -> b//10 in {3,7}; a//10 in {8} -> b//10 in {7} | (reads) | same |
| L18 up c23 | 37% / 27% | **tens(a,b)** (R2 0.81): a//10 in {2} -> b//10 in {5,6,7}; a//10 in {3} -> b//10 in {4,5,6,7,8,9}; a//10 in {4} -> b//10 in {3,4,5,6,7,8,9}; a//10 in {5} -> b//10 in {3,4,5,6,7,8}; a//10 in {6} -> b//10 in {2,3,4,5,6}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {3,4,5}; a//10 in {9,10} -> b//10 in {3,4} | **tens(a,b)** (R2 0.73): a//10 in {3} -> b//10 in {2,3,4,5,6}; a//10 in {4,5,6} -> b//10 in {3,4,5,6}; a//10 in {7} -> b//10 in {4,5,6}; a//10 in {8} -> b//10 in {4,5,6,7} | (reads) | same |
| L18 up c25 | 52% / 17% | **tens(a,b)** (R2 0.77): a//10 in {0,1} -> b//10 in {2,3,4,7,8,9}; a//10 in {2} -> b//10 in {2,3,7,8,9}; a//10 in {3} -> b//10 in {8,9,10}; a//10 in {4,5,6,9,10} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {7} -> b//10 in {2,3,7,8,9,10}; a//10 in {8} -> b//10 in {3,4,8,9,10} | **tens(a,b)** (R2 0.51): a//10 in {1,2,4} -> b//10 in {1}; a//10 in {5} -> b//10 in {1,2,6,7}; a//10 in {6,9} -> b//10 in {1,6,7}; a//10 in {10} -> b//10 in {0,1,2,5,6,7} | (reads) | same |
| L18 up c36 | 35% / 26% | **tens(a,b)** (R2 0.65): a//10 in {0,1,5,6} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {2} -> b//10 in {2,3,7,8}; a//10 in {7} -> b//10 in {2,3,4,8}; a//10 in {10} -> b//10 in {3,4,8} | **tens(a,b)** (R2 0.65): a//10 in {0} -> b//10 in {0,1}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {1,2}; a//10 in {5} -> b//10 in {0,1,2,5,6,7,10}; a//10 in {6} -> b//10 in {0,1,2,3,5,6,7,10}; a//10 in {7} -> b//10 in {1,6,7}; a//10 in {10} -> b//10 in {0,1,5,6,7} | (reads) | same |
| L18 up c61 | 49% / 36% | **tens(a,b)** (R2 0.69): a//10 in {0,5} -> b//10 in {3,8}; a//10 in {2} -> b//10 in {3,4,8,9,10}; a//10 in {3,8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {4,9} -> b//10 in {0,2,3,4,5,7,8,9,10}; a//10 in {7} -> b//10 in {3,4,9,10}; a//10 in {10} -> b//10 in {2,3,4,7,8,9} | **tens(a,b)** (R2 0.67): a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {0,1,2,3,6,7,10}; a//10 in {4} -> b//10 in {0,1,2,6,7,8,10}; a//10 in {7} -> b//10 in {0,5,6}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,8,10}; a//10 in {10} -> b//10 in {0,1,2,5,6,7} | (reads) | same |
| L18 up c91 | 31% / 9% | **tens(a,b)** (R2 0.82): a//10 in {0} -> b//10 in {3,4}; a//10 in {1} -> b//10 in {2,3,4}; a//10 in {2,3,4} -> b//10 in {0,1,2,3,4,5,6}; a//10 in {5} -> b//10 in {2,3,4,5}; a//10 in {6} -> b//10 in {4} | **tens(a,b)** (R2 0.52): a//10 in {3} -> b//10 in {2,3,4}; a//10 in {4} -> b//10 in {3,4}; a//10 in {5} -> b//10 in {4,5} | (reads) | same |
| L18 up c111 | 1% / 10% | **tens(a,b)** (R2 0.51): a//10 in {9} -> b//10 in {9,10}; a//10 in {10} -> b//10 in {10} | **tens(a,b)** (R2 0.67): a//10 in {7,8,9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {5,6,7,8,9,10} | (reads) | same |
| L18 up c130 | 17% / 1% | **tens(a,b)** (R2 0.73): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1,2,3,4,5,6,7,8,9} -> b//10 in {0}; a//10 in {10} -> b//10 in {0,1} | **tens(a,b)** (R2 0.56): a//10 in {0} -> b//10 in {0} | (reads) | same |
| L18 up c390 | 13% / 8% | **tens(a,b)** (R2 0.57): a//10 in {1} -> b//10 in {2}; a//10 in {2} -> b//10 in {1,2,3,7,8}; a//10 in {3} -> b//10 in {2,7}; a//10 in {7} -> b//10 in {2,7,8}; a//10 in {8} -> b//10 in {7} | **tens(a,b)** (R2 0.51): a//10 in {1} -> b//10 in {2}; a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {7,8} -> b//10 in {7} | (reads) | same |
| L18 up c400 | 33% / 17% | **tens(a,b)** (R2 0.73): a//10 in {0} -> b//10 in {0,1,2,6,10}; a//10 in {1,6} -> b//10 in {0,1,10}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3} -> b//10 in {0,1,2}; a//10 in {4,5,7} -> b//10 in {0,1,2,10}; a//10 in {8} -> b//10 in {0,1,2,6,7,10}; a//10 in {9,10} -> b//10 in {0,1,2,5,6,7,10} | **tens(a,b)** (R2 0.67): a//10 in {3,4,6} -> b//10 in {8}; a//10 in {5,7} -> b//10 in {8,9}; a//10 in {8} -> b//10 in {3,4,7,8,9}; a//10 in {9} -> b//10 in {3,7,8,9,10}; a//10 in {10} -> b//10 in {2,3,4,7,8,9,10} | (reads) | same |
| L18 up c402 | 13% / 0 | **tens(a,b)** (R2 0.70): a//10 in {6} -> b//10 in {7,8,9}; a//10 in {7,8} -> b//10 in {6,7,8,9}; a//10 in {9} -> b//10 in {6,7,8} | off (on 0) | (reads) | same |
| L18 up c435 | 34% / 13% | **tens(a,b)** (R2 0.65): a//10 in {0,1} -> b//10 in {0,4,5,6,9,10}; a//10 in {2} -> b//10 in {4,5,9,10}; a//10 in {4,9} -> b//10 in {0,5,10}; a//10 in {5,6,10} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {7} -> b//10 in {0,4,5,9,10} | **tens(a,b)** (R2 0.58): a//10 in {2} -> b//10 in {0}; a//10 in {5,6,10} -> b//10 in {0,4,5,9,10}; a//10 in {7} -> b//10 in {0,5} | (reads) | same |
| L18 up c505 | 9% / 15% | **tens(a,b)** (R2 0.61): a//10 in {6} -> b//10 in {0,3,4,8}; a//10 in {7} -> b//10 in {2,3,7,8} | **tens(a,b)** (R2 0.67): a//10 in {2} -> b//10 in {7}; a//10 in {6} -> b//10 in {0,1,2,6,7,10}; a//10 in {7} -> b//10 in {0,1,2,3,6,7} | (reads) | same |
| L18 up c513 | 28% / 15% | **tens(a,b)** (R2 0.62): a//10 in {0,1} -> b//10 in {0,1,3,4,5,8,9,10}; a//10 in {2,10} -> b//10 in {0,4,9,10}; a//10 in {3,4,5,6,9} -> b//10 in {0,9,10}; a//10 in {7,8} -> b//10 in {9,10} | **tens(a,b)** (R2 0.58): a//10 in {0,1} -> b//10 in {0,1,10}; a//10 in {2} -> b//10 in {0,1}; a//10 in {4,7} -> b//10 in {0}; a//10 in {5,6} -> b//10 in {0,10}; a//10 in {9} -> b//10 in {0,9,10}; a//10 in {10} -> b//10 in {0,1,4,5,6,9,10} | (reads) | same |
| L18 up c518 | 19% / 10% | **tens(a,b)** (R2 0.77): a//10 in {3} -> b//10 in {5,6,7}; a//10 in {4,5,6} -> b//10 in {4,5,6,7}; a//10 in {7} -> b//10 in {5,6}; a//10 in {9,10} -> b//10 in {6} | **tens(a,b)** (R2 0.65): a//10 in {3} -> b//10 in {3}; a//10 in {4,5,10} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {3,4,5,6} | (reads) | same |
| L18 up c779 | 10% / 4% | **tens(a,b)** (R2 0.62): a//10 in {4} -> b//10 in {4,5,6,10}; a//10 in {5,10} -> b//10 in {4,5,10}; a//10 in {6} -> b//10 in {4,5}; a//10 in {9} -> b//10 in {5,10} | **tens(a,b)** (R2 0.53): a//10 in {4} -> b//10 in {10}; a//10 in {5} -> b//10 in {4,5,10}; a//10 in {6} -> b//10 in {5}; a//10 in {10} -> b//10 in {4,5} | (reads) | same |
| L19 down c3 | 27% / 98% | **res//10** (R2 0.63): (tens) res in {2..64, 200} | always | - | same |
| L19 down c11 | 46% / 44% | **tens(a,b)** (R2 0.71): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {4,8,9,10}; a//10 in {2,7} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {3,8} -> b//10 in {1,2,3,6,7,8}; a//10 in {4} -> b//10 in {1,2,5,6,7}; a//10 in {5} -> b//10 in {4,5,6}; a//10 in {6} -> b//10 in {3,4,5,8,9,10}; a//10 in {9} -> b//10 in {1,2,6,7,10}; a//10 in {10} -> b//10 in {0,1,2,5,6,7,10} | **tens(a,b)** (R2 0.61): a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {0,1,2,3,6,7,8,10}; a//10 in {3} -> b//10 in {1,2,3,7,8}; a//10 in {4} -> b//10 in {2,3,7,8}; a//10 in {5} -> b//10 in {3,4,8}; a//10 in {6} -> b//10 in {4,5,6,8,9}; a//10 in {7} -> b//10 in {0,1,2,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {1,2,3,6,7,8,9}; a//10 in {9} -> b//10 in {7,8,9}; a//10 in {10} -> b//10 in {3,4,8,9} | a: mod50 +4%; res: mod50 +11%, mod25 +18% | b: mod50 +4%; res: mod100 +2%, mod50 +11%, mod25 +5% |
| L19 down c24 | 44% / 19% | **res//10** (R2 0.77): (tens) res in {24, 26, 28..80, 142..174} [coarser: res mod 100 in {34..76}, R2 0.81] | **tens(a,b)** (R2 0.57): a//10 in {2} -> b//10 in {6,7,8}; a//10 in {3} -> b//10 in {7,8}; a//10 in {4} -> b//10 in {7,8,9}; a//10 in {5} -> b//10 in {0,5,8,9,10}; a//10 in {6} -> b//10 in {0,9,10}; a//10 in {8,9} -> b//10 in {3,4}; a//10 in {10} -> b//10 in {3,4,5,6} | res: mod100 +17% | - |
| L19 down c28 | 57% / 20% | **res//10** (R2 0.79): (tens) res in {14..52, 100..162, 164} [coarser: res mod 100 in {1, 10..56}, R2 0.81] | **tens(a,b)** (R2 0.65): a//10 in {2} -> b//10 in {0}; a//10 in {4} -> b//10 in {1}; a//10 in {5} -> b//10 in {1,2,3}; a//10 in {6} -> b//10 in {2,3,4}; a//10 in {7} -> b//10 in {3,4,5}; a//10 in {8} -> b//10 in {4,5,6}; a//10 in {9} -> b//10 in {6,7}; a//10 in {10} -> b//10 in {5,6,7,8} | res: mod100 +12% | - |
| L19 down c34 | 14% / 0 | **res//10** (R2 0.82): (tens) res in {2..53} | off (on 0) | - | same |
| L19 down c86 | 25% / 1% | **res//10** (R2 0.92): (tens) res in {132..200} | **tens(a,b)** (R2 0.62): a//10 in {6} -> b//10 in {10}; a//10 in {10} -> b//10 in {3,4,5,6} | a: mod100 +3%; res: mod100 +5% | - |
| L19 down c94 | 13% / 9% | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {0,2,3,7,8,9,10}; a//10 in {1,2,3,4,5,6,7,8,9} -> b//10 in {0}; a//10 in {10} -> b//10 in {0,10} | **tens(a,b)** (R2 0.58): a//10 in {0} -> b//10 in {0,1,2,10}; a//10 in {1,3,7,8,10} -> b//10 in {0}; a//10 in {9} -> b//10 in {0,9,10} | - | same |
| L19 down c147 | 9% / 0 | **tens(a,b)** (R2 0.64): a//10 in {1,9} -> b//10 in {9,10}; a//10 in {2,3} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {10} | off (on 0) | - | same |
| L19 down c151 | 9% / 0 | **tens(a,b)** (R2 0.72): a//10 in {3,4,8} -> b//10 in {9,10}; a//10 in {5} -> b//10 in {9}; a//10 in {9} -> b//10 in {3,4,5,8,9,10}; a//10 in {10} -> b//10 in {4,8,9} | off (on 0) | - | same |
| L19 down c423 | 11% / 0 | **tens(a,b)** (R2 0.74): a//10 in {1} -> b//10 in {2,3}; a//10 in {2,3} -> b//10 in {1,2,3} | off (on 0) | - | same |
| L19 down c454 | 1% / 1% | **res//10** (R2 0.62): (tens) res in {2..9} | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {0} | - | same |
| L19 gate c11 | 34% / 23% | **tens(a,b)** (R2 0.68): a//10 in {1} -> b//10 in {8,9}; a//10 in {2} -> b//10 in {2,3,4,7,8,9,10}; a//10 in {3} -> b//10 in {1,2,3,6,7,8}; a//10 in {4,9} -> b//10 in {1,2,6,7}; a//10 in {6} -> b//10 in {4}; a//10 in {7} -> b//10 in {2,3,4,7,8,9}; a//10 in {8} -> b//10 in {1,2,3,7,8}; a//10 in {10} -> b//10 in {1,2} | **tens(a,b)** (R2 0.56): a//10 in {2} -> b//10 in {0,1,2,7,8}; a//10 in {3} -> b//10 in {1,2,3,7,8}; a//10 in {4} -> b//10 in {2,3,8}; a//10 in {7} -> b//10 in {5,6,7,8}; a//10 in {8} -> b//10 in {2,3,6,7,8}; a//10 in {9} -> b//10 in {7,8}; a//10 in {10} -> b//10 in {3,4,8,9} | (reads) | same |
| L19 gate c23 | 79% / 28% | **res//10** (R2 0.90): (tens) res in {67..200} | **tens(a,b)** (R2 0.71): a//10 in {0,1,2} -> b//10 in {7,8,9,10}; a//10 in {3} -> b//10 in {9,10}; a//10 in {4} -> b//10 in {10}; a//10 in {7,10} -> b//10 in {0,1,2,3}; a//10 in {8} -> b//10 in {0,1,2,3,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5} | (reads) | same |
| L19 gate c24 | 27% / 8% | **res//10** (R2 0.79): (tens) res in {2..69} | **tens(a,b)** (R2 0.67): a//10 in {0,1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {2} | (reads) | same |
| L19 gate c28 | 43% / 17% | **tens(a,b)** (R2 0.69): a//10 in {0,10} -> b//10 in {6,7,8}; a//10 in {1} -> b//10 in {5,6,7}; a//10 in {2} -> b//10 in {4,5,6}; a//10 in {3} -> b//10 in {3,4,5}; a//10 in {4} -> b//10 in {2,3,4,7}; a//10 in {5} -> b//10 in {1,2,3,6,7,8}; a//10 in {6} -> b//10 in {0,1,2,5,6,7}; a//10 in {7} -> b//10 in {0,1,4,5,6,9,10}; a//10 in {8} -> b//10 in {0,8,9,10}; a//10 in {9} -> b//10 in {7,8,9} | **tens(a,b)** (R2 0.58): a//10 in {3} -> b//10 in {6}; a//10 in {4} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {3,6,7,8}; a//10 in {6} -> b//10 in {3,4,7,8}; a//10 in {7} -> b//10 in {3,4,5,8,9,10}; a//10 in {8} -> b//10 in {10}; a//10 in {10} -> b//10 in {2,3,7} | (reads) | same |
| L19 gate c423 | 32% / 3% | **res//10** (R2 0.70): (tens) res in {36, 38, 40..74, 143..172} [coarser: res mod 100 in {42..72}, R2 0.96] | unexplained (best R2 0.33) | (reads) | same |
| L19 up c11 | 46% / 40% | **tens(a,b)** (R2 0.66): a//10 in {1} -> b//10 in {3,4,8,9}; a//10 in {2} -> b//10 in {1,2,3,4,7,8,9,10}; a//10 in {3,8} -> b//10 in {1,2,3,6,7,8}; a//10 in {4} -> b//10 in {1,2,6,7}; a//10 in {6} -> b//10 in {3,4,5,7,8,9,10}; a//10 in {7} -> b//10 in {1,2,3,4,6,7,8,9,10}; a//10 in {9} -> b//10 in {1,2,6,7,10}; a//10 in {10} -> b//10 in {5,6,7} | **tens(a,b)** (R2 0.60): a//10 in {2} -> b//10 in {0,1,2,3,6,7,8}; a//10 in {3} -> b//10 in {1,2,3,7,8}; a//10 in {4} -> b//10 in {3,7,8}; a//10 in {5} -> b//10 in {4,8}; a//10 in {6} -> b//10 in {5,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,5,6,7,8,9,10}; a//10 in {8} -> b//10 in {1,2,3,6,7,8,9}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {3,4,5,6,7,8,9,10} | (reads) | same |
| L19 up c20 | 4% / 0 | **tens(a,b)** (R2 0.61): a//10 in {4} -> b//10 in {4}; a//10 in {5} -> b//10 in {3,4} | off (on 0) | (reads) | same |
| L19 up c28 | 46% / 18% | **res//10** (R2 0.73): (tens) res in {100..171, 174..176, 178, 196, 198, 200} | **tens(a,b)** (R2 0.67): a//10 in {4} -> b//10 in {1}; a//10 in {5} -> b//10 in {1,2,3,4}; a//10 in {6} -> b//10 in {2,3,4,5}; a//10 in {7} -> b//10 in {3,4,5}; a//10 in {8} -> b//10 in {4,5,6}; a//10 in {9} -> b//10 in {6,7}; a//10 in {10} -> b//10 in {5,6,7,8} | (reads) | same |
| L19 up c423 | 48% / 27% | **res//10** (R2 0.75): (tens) res in {2..28, 81..127, 186..200} [coarser: res mod 100 in {0..27, 83..99}, R2 0.98] | **tens(a,b)** (R2 0.69): a//10 in {0} -> b//10 in {0,1,9,10}; a//10 in {1} -> b//10 in {0,1,2,10}; a//10 in {2} -> b//10 in {0,1,2}; a//10 in {3} -> b//10 in {2}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {4}; a//10 in {6} -> b//10 in {4,5,6,8}; a//10 in {7} -> b//10 in {5,6,7,8}; a//10 in {8} -> b//10 in {6,7,8,9}; a//10 in {9,10} -> b//10 in {0,8,9,10} | (reads) | same |
| L19 up c426 | 53% / 24% | **tens(a,b)** (R2 0.65): a//10 in {0} -> b//10 in {1,2,3,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,6,7,8,10}; a//10 in {2} -> b//10 in {0,1,5,6,10}; a//10 in {3} -> b//10 in {0,3,4,5,6,9,10}; a//10 in {4} -> b//10 in {3,4,5,8,9,10}; a//10 in {5} -> b//10 in {2,3,4}; a//10 in {6} -> b//10 in {0,1,2,3}; a//10 in {7} -> b//10 in {0,1,2,10}; a//10 in {8} -> b//10 in {0,1,3,4,5,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,7,8,9,10}; a//10 in {10} -> b//10 in {2,3,7,8,9,10} | **tens(a,b)** (R2 0.64): a//10 in {5} -> b//10 in {7}; a//10 in {6} -> b//10 in {6,7,8}; a//10 in {7} -> b//10 in {0,7,8,9,10}; a//10 in {8} -> b//10 in {0,1,5,6,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,5,6,7,9,10}; a//10 in {10} -> b//10 in {0,1,2,6,7,10} | (reads) | same |
| L20 down c5 | 71% / 100% | **tens(a,b)** (R2 0.74): a//10 in {0,1,2} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,10}; a//10 in {4,5} -> b//10 in {0,1,2,3,4,5,6}; a//10 in {6} -> b//10 in {0,1,2,3,4}; a//10 in {7} -> b//10 in {0,1,2,3}; a//10 in {8} -> b//10 in {0,1,2,9}; a//10 in {9} -> b//10 in {0,1,2,6,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,6,8,9,10} | always | a: mod100 +3%; b: mod100 +2%; res: mod100 +3% | - |
| L20 down c20 | 54% / 48% | **res//10** (R2 0.81): (tens) res in {2..50, 99..150, 198..200} [coarser: res mod 100 in {0..50, 98..99}, R2 0.99] | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {0,1,2,3}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {0,1,2,3,9,10}; a//10 in {3} -> b//10 in {0,1,2,3,4,6,10}; a//10 in {4} -> b//10 in {0,1,2,3,4,6}; a//10 in {5} -> b//10 in {1,2,3,4,5}; a//10 in {6} -> b//10 in {2,3,4,5,6}; a//10 in {7} -> b//10 in {3,4,5,6,9}; a//10 in {8} -> b//10 in {4,5,6,7,8,9}; a//10 in {9} -> b//10 in {6,7,8,9,10}; a//10 in {10} -> b//10 in {0,5,6,7,8,9} | res: mod100 +8% | res: mod100 +4% |
| L20 down c31 | 29% / 25% | **res//10** (R2 0.79): (tens) res in {2..13, 89..117, 188..200} [coarser: res mod 100 in {0..15, 89..99}, R2 0.98] | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {0,1,9,10}; a//10 in {1} -> b//10 in {0,1,2}; a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {4,5}; a//10 in {6} -> b//10 in {5,6,7,8}; a//10 in {7} -> b//10 in {6,7,8,9}; a//10 in {8} -> b//10 in {7,8,9,10}; a//10 in {9,10} -> b//10 in {0,8,9,10} | res: mod100 +4%, mod25 +6% | res: mod100 +3%, mod50 +3%, mod25 +4% |
| L20 down c80 | 10% / 4% | **tens(a,b)** (R2 0.73): a//10 in {0} -> b//10 in {8,9,10}; a//10 in {7} -> b//10 in {10}; a//10 in {8,10} -> b//10 in {0,8,9,10}; a//10 in {9} -> b//10 in {0,7,8,9,10} | unexplained (best R2 0.40) | - | same |
| L20 down c101 | 14% / 0 | **res//10** (R2 0.77): (tens) res in {148..188, 190, 192..198, 200} | off (on 0) | - | same |
| L20 down c136 | 8% / 0 | **tens(a,b)** (R2 0.67): a//10 in {5} -> b//10 in {7}; a//10 in {6} -> b//10 in {6,7,8}; a//10 in {7} -> b//10 in {6,7} | off (on 0) | - | same |
| L20 down c153 | 19% / 0 | **res//10** (R2 0.91): (tens) res in {140..196, 198} | off (on 0) | - | same |
| L20 down c172 | 5% / 3% | **tens(a,b)** (R2 0.66): a//10 in {9} -> b//10 in {1,2,3,6}; a//10 in {10} -> b//10 in {1,2,3} | **a** (R2 0.79): a in {99..100} | - | same |
| L20 gate c2 | 43% / 22% | **res//10** (R2 0.76): (tens) res in {24..55, 116..164} [coarser: res mod 100 in {20..58}, R2 0.86] | **res//10** (R2 0.56): (tens) res in {21..50} | (reads) | same |
| L20 gate c5 | 23% / 64% | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {6,7,8,9,10}; a//10 in {1} -> b//10 in {6,7,8}; a//10 in {2} -> b//10 in {6,7}; a//10 in {6} -> b//10 in {1,2}; a//10 in {7} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1,9,10}; a//10 in {9,10} -> b//10 in {0,8,9,10} | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {2,5,6,7,8,9,10}; a//10 in {3,4} -> b//10 in {7,8,9}; a//10 in {5} -> b//10 in {5,6,7,8,9,10}; a//10 in {6} -> b//10 in {0,1,6,7,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,7,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,3,7,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,7,8,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,9,10} | (reads) | same |
| L20 gate c14 | 30% / 4% | **res//10** (R2 0.67): (tens) res in {81..99, 132..147, 174..200} | unexplained (best R2 0.40) | (reads) | same |
| L20 gate c20 | 48% / 31% | **res//10** (R2 0.76): (tens) res in {46..88, 141..188} [coarser: res mod 100 in {45..88}, R2 0.96] | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {7}; a//10 in {2} -> b//10 in {5,6,7,8}; a//10 in {3,4} -> b//10 in {7,8}; a//10 in {5} -> b//10 in {0,7,8,9,10}; a//10 in {6} -> b//10 in {0,1,2,8,9,10}; a//10 in {7} -> b//10 in {0,1,2,3,10}; a//10 in {8} -> b//10 in {0,1,2,3}; a//10 in {9} -> b//10 in {1,2,3,4}; a//10 in {10} -> b//10 in {2,3,4,5} | (reads) | same |
| L20 gate c31 | 36% / 31% | **res//10** (R2 0.78): (tens) res in {2..23, 91..125, 191..200} [coarser: res mod 100 in {0..23, 25, 91..99}, R2 0.98] | **tens(a,b)** (R2 0.69): a//10 in {0} -> b//10 in {0,1,10}; a//10 in {1} -> b//10 in {0,1}; a//10 in {2} -> b//10 in {0,1,2}; a//10 in {3} -> b//10 in {1,2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4,5}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9}; a//10 in {9} -> b//10 in {7,8,9,10}; a//10 in {10} -> b//10 in {0,7,8,9,10} | (reads) | same |
| L20 up c14 | 79% / 10% | **res//10** (R2 0.76): (tens) res in {26..53, 81..200} | unexplained (best R2 0.49) | (reads) | same |
| L20 up c58 | 14% / 0 | **res//10** (R2 0.81): (tens) res in {146..181} | off (on 0) | (reads) | same |
| L20 up c153 | 10% / 0 | **res//10** (R2 0.71): (tens) res in {145, 149..170, 173..174} | off (on 0) | (reads) | same |
| L20 up c223 | 10% / 6% | **res//10** (R2 0.77): (tens) res in {110..120} | **res** (R2 0.68): res in {11..18} | (reads) | same |
| L20 up c378 | 27% / 7% | **tens(a,b)** (R2 0.63): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,3,9,10}; a//10 in {2} -> b//10 in {0,1,2,10}; a//10 in {3,9} -> b//10 in {0,1}; a//10 in {4,5,6,7,10} -> b//10 in {0} | unexplained (best R2 0.29) | (reads) | same |
| L21 down c0 | 33% / 26% | **res//10** (R2 0.84): (tens) res in {8..40, 108..140} [coarser: res mod 100 in {8..40}, R2 1.00] | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {2}; a//10 in {1} -> b//10 in {0,10}; a//10 in {2} -> b//10 in {0,1,9,10}; a//10 in {3} -> b//10 in {0,1,2,10}; a//10 in {4} -> b//10 in {1,2,3}; a//10 in {5} -> b//10 in {2,3,4}; a//10 in {6} -> b//10 in {3,4,5}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {5,6,7}; a//10 in {9,10} -> b//10 in {6,7,8} | res: mod100 +15%, mod50 +5% | a: mod50 +2%; res: mod100 +7%, mod50 +6% |
| L21 down c36 | 35% / 7% | **res//10** (R2 0.79): (tens) res in {60..91, 153..195, 197} [coarser: res mod 100 in {56..95}, R2 0.92] | **tens(a,b)** (R2 0.65): a//10 in {8} -> b//10 in {0,1,2,10}; a//10 in {9} -> b//10 in {1,2}; a//10 in {10} -> b//10 in {1,2,3,4} | res: mod100 +3% | - |
| L21 down c49 | 16% / 2% | **res//10** (R2 0.70): (tens) res in {40, 44, 132..156, 158, 160} | unexplained (best R2 0.27) | res: mod25 +2% | - |
| L21 down c85 | 16% / 4% | **tens(a,b)** (R2 0.58): a//10 in {0} -> b//10 in {0,1,2,3,4,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,10}; a//10 in {2} -> b//10 in {0,1}; a//10 in {3,9,10} -> b//10 in {0} | unexplained (best R2 0.40) | - | same |
| L21 down c105 | 3% / 33% | **res//10** (R2 0.68): (tens) res in {2..25} | **tens(a,b)** (R2 0.72): a//10 in {0} -> b//10 in {0,1,2}; a//10 in {1} -> b//10 in {0,1,2,10}; a//10 in {2} -> b//10 in {0,1,2,3}; a//10 in {3} -> b//10 in {1,2,3}; a//10 in {4} -> b//10 in {3}; a//10 in {5} -> b//10 in {3,4}; a//10 in {6} -> b//10 in {4,5,6,7,8}; a//10 in {7} -> b//10 in {5,6,7,8,9}; a//10 in {8} -> b//10 in {6,7,8,9}; a//10 in {9,10} -> b//10 in {7,8,9,10} | - | res: mod100 +4%, mod50 +2% |
| L21 down c206 | 6% / 0 | **res//10** (R2 0.75): (tens) res in {17..18, 22..39} | off (on 0) | - | same |
| L21 down c267 | 7% / 0 | **tens(a,b)** (R2 0.69): a//10 in {4} -> b//10 in {4}; a//10 in {7} -> b//10 in {10}; a//10 in {8} -> b//10 in {9,10}; a//10 in {9} -> b//10 in {8,9,10}; a//10 in {10} -> b//10 in {7,8,9,10} | off (on 0) | - | same |
| L21 gate c0 | 32% / 21% | **res//10** (R2 0.84): (tens) res in {10..35, 108..140} [coarser: res mod 100 in {8..40}, R2 0.96] | **res** (R2 0.82): res in {8..32} | (reads) | same |
| L21 gate c36 | 8% / 2% | **tens(a,b)** (R2 0.69): a//10 in {6} -> b//10 in {9,10}; a//10 in {7} -> b//10 in {8,9,10}; a//10 in {8} -> b//10 in {7,8,10}; a//10 in {9} -> b//10 in {6,7}; a//10 in {10} -> b//10 in {6,7,10} | unexplained (best R2 0.40) | (reads) | same |
| L21 gate c133 | 31% / 4% | **res//10** (R2 0.84): (tens) res in {61..90, 159..195} [coarser: res mod 100 in {60..92}, R2 0.94] | unexplained (best R2 0.49) | (reads) | same |
| L21 up c0 | 35% / 24% | **res//10** (R2 0.83): (tens) res in {8..40, 108..140} [coarser: res mod 100 in {8..40}, R2 0.99] | **tens(a,b)** (R2 0.70): a//10 in {1} -> b//10 in {0}; a//10 in {2} -> b//10 in {0,1,10}; a//10 in {3} -> b//10 in {0,1,2}; a//10 in {4} -> b//10 in {1,2,3}; a//10 in {5} -> b//10 in {2,3,4}; a//10 in {6} -> b//10 in {3,4,5}; a//10 in {7} -> b//10 in {3,4,5,6}; a//10 in {8} -> b//10 in {5,6,7}; a//10 in {9,10} -> b//10 in {6,7,8} | (reads) | same |
| L21 up c36 | 38% / 12% | **res//10** (R2 0.74): (tens) res in {58, 60..91, 148..150, 152..198, 200} [coarser: res mod 100 in {0, 56..95, 97..98}, R2 0.86] | **tens(a,b)** (R2 0.60): a//10 in {5} -> b//10 in {7}; a//10 in {6} -> b//10 in {8}; a//10 in {7} -> b//10 in {0,8,9,10}; a//10 in {8} -> b//10 in {0,1,2,10}; a//10 in {9} -> b//10 in {1,2}; a//10 in {10} -> b//10 in {1,2,3,4} | (reads) | same |
| L21 up c105 | 19% / 8% | **res//10** (R2 0.74): (tens) res in {127, 130..154, 156..157, 187} | **tens(a,b)** (R2 0.58): a//10 in {4} -> b//10 in {10}; a//10 in {6} -> b//10 in {3}; a//10 in {7} -> b//10 in {3,4}; a//10 in {8} -> b//10 in {4,5}; a//10 in {9} -> b//10 in {6}; a//10 in {10} -> b//10 in {4,5,6,7,8} | (reads) | same |
| L22 down c29 | 45% / 1% | **res//10** (R2 0.89): (tens) res in {108..200} | **tens(a,b)** (R2 0.51): a//10 in {1,2,3,4,5} -> b//10 in {10} | a: mod100 +4%, mod50 +2%; b: mod100 +5%, mod50 +2% | - |
| L22 down c107 | 13% / 8% | **res//10** (R2 0.76): (tens) res in {2..49, 200} | **tens(a,b)** (R2 0.73): a//10 in {0,1,2} -> b//10 in {0,1,2} | - | same |
| L22 down c134 | 20% / 1% | **tens(a,b)** (R2 0.70): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {9,10}; a//10 in {2} -> b//10 in {8,9,10}; a//10 in {3} -> b//10 in {7,8,9,10}; a//10 in {4} -> b//10 in {6,7}; a//10 in {5} -> b//10 in {6}; a//10 in {6} -> b//10 in {4,5}; a//10 in {7} -> b//10 in {3,4,5}; a//10 in {8} -> b//10 in {2,3}; a//10 in {9} -> b//10 in {1,2,3,10}; a//10 in {10} -> b//10 in {0,1,2,3} | unexplained (best R2 0.20) | - | same |
| L22 down c404 | 4% / 4% | **res//10** (R2 0.58): (tens) res in {180..200} | unexplained (best R2 0.39) | - | same |
| L22 down c969 | 3% / 0 | **tens(a,b)** (R2 0.46): a//10 in {8,9} -> b//10 in {9,10}; a//10 in {10} -> b//10 in {8,10} | off (on 0) | - | same |
| L22 gate c29 | 48% / 1% | **res//10** (R2 0.82): (tens) res in {2, 107..200} | unexplained (best R2 0.43) | (reads) | same |
| L22 o c10 (H15) | 29% / 19% | **tens(a,b)** (R2 0.78): a//10 in {0,1,2} -> b//10 in {8,9,10}; a//10 in {3,4,5,6,7} -> b//10 in {9,10}; a//10 in {8} -> b//10 in {0,2,3,7,8,9,10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.82): a//10 in {0,1,2,3,4} -> b//10 in {9,10}; a//10 in {5,6,8} -> b//10 in {10}; a//10 in {9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | - | same |
| L22 o c200 (H15) | 13% / 1% | **tens(a,b)** (R2 0.59): a//10 in {0,1,2,3,4,8} -> b//10 in {9,10}; a//10 in {5,6,7} -> b//10 in {10}; a//10 in {9} -> b//10 in {3,4,9,10}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | **tens(a,b)** (R2 0.62): a//10 in {0,1,2,3,4,5,6} -> b//10 in {10}; a//10 in {10} -> b//10 in {1,10} | - | same |
| L22 up c56 | 31% / 20% | **res//10** (R2 0.80): (tens) res in {22..38, 111..145} [coarser: res mod 100 in {14..17, 19..41}, R2 0.81] | **res//10** (R2 0.65): (tens) res in {11..19, 21..38} | (reads) | same |
| L23 down c7 | 39% / 14% | **res//10** (R2 0.75): (tens) res in {7..8, 15, 18, 20..88} | unexplained (best R2 0.41) | - | same |
| L23 down c10 | 19% / 97% | **res//10** (R2 0.63): (tens) res in {2..4, 6, 8..59, 61..63} | always | - | same |
| L23 down c13 | 7% / 2% | **tens(a,b)** (R2 0.64): a//10 in {0,1,2,5,7} -> b//10 in {10}; a//10 in {8} -> b//10 in {9,10}; a//10 in {9} -> b//10 in {8,9,10}; a//10 in {10} -> b//10 in {0,6,7,8,9,10} | unexplained (best R2 0.45) | - | same |
| L23 down c18 | 27% / 18% | **res//10** (R2 0.75): (tens) res in {73..102, 190..191} | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {7,8,9,10}; a//10 in {7} -> b//10 in {8}; a//10 in {8} -> b//10 in {0,1,2,8,9,10}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,9,10}; a//10 in {10} -> b//10 in {0,1,2} | - | same |
| L23 down c20 | 18% / 12% | **res//10** (R2 0.80): (tens) res in {25..36, 121..139} [coarser: res mod 100 in {21..37}, R2 0.91] | **res** (R2 0.56): res in {23..36} | res: mod100 +2%, mod50 +2%, mod25 +3% | - |
| L23 down c29 | 58% / 3% | **res//10** (R2 0.85): (tens) res in {92..199} | unexplained (best R2 0.45) | res: mod100 +2% | - |
| L23 down c41 | 15% / 30% | **res//10** (R2 0.79): (tens) res in {6..59} | **res//10** (R2 0.53): (tens) res in {2..31} | - | res: mod100 +2% |
| L23 down c58 | 15% / 25% | **res//10** (R2 0.83): (tens) res in {3..56, 200} | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {0,1,2,3}; a//10 in {1,2,3} -> b//10 in {0,1,2,3,4}; a//10 in {4} -> b//10 in {0,1,2}; a//10 in {5} -> b//10 in {2} | - | same |
| L23 down c130 | 28% / 0 | **res//10** (R2 0.83): (tens) res in {128..200} | off (on 0) | - | same |
| L23 gate c41 | 16% / 50% | **res//10** (R2 0.80): (tens) res in {2..59} | **res//10** (R2 0.59): (tens) res in {-29, -27..31} | (reads) | same |
| L23 gate c196 | 1% / 11% | **res//10** (R2 0.59): (tens) res in {2..10} | **res//10** (R2 0.66): (tens) res in {1..10} | (reads) | same |
| L23 up c5 | 33% / 7% | **res//10** (R2 0.76): (tens) res in {43..51, 53..71, 140..190} [coarser: res mod 100 in {41..73}, R2 0.83] | unexplained (best R2 0.34) | (reads) | same |
| L23 up c18 | 8% / 0 | **res//10** (R2 0.77): (tens) res in {160, 163..200} | unexplained (best R2 0.31) | (reads) | same |
| L23 up c29 | 71% / 9% | **res//10** (R2 0.91): (tens) res in {79..200} | **tens(a,b)** (R2 0.55): a//10 in {0,1,2} -> b//10 in {9,10}; a//10 in {9} -> b//10 in {0,1,9}; a//10 in {10} -> b//10 in {0,1,2,10} | (reads) | same |
| L24 down c8 | 25% / 28% | **res//10** (R2 0.74): (tens) res in {52..85, 169} | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {6,7,8}; a//10 in {2,3,4} -> b//10 in {7,8}; a//10 in {5} -> b//10 in {8,9}; a//10 in {6} -> b//10 in {0,1,9,10}; a//10 in {7} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1,2,3}; a//10 in {9} -> b//10 in {1,2,3,4}; a//10 in {10} -> b//10 in {2,3,4} | res: mod100 +2% | - |
| L24 down c31 | 19% / 15% | **res//10** (R2 0.82): (tens) res in {8..20, 102..119, 200} [coarser: res mod 100 in {0, 7..19}, R2 0.86] | **res** (R2 0.69): res in {4..19} | - | res: mod25 +2% |
| L24 down c34 | 28% / 0 | **res//10** (R2 0.89): (tens) res in {126..200} | off (on 0) | a: mod100 +3%; b: mod100 +3% | - |
| L24 down c51 | 28% / 0 | **tens(a,b)** (R2 0.80): a//10 in {2} -> b//10 in {9,10}; a//10 in {3} -> b//10 in {8,9,10}; a//10 in {4,5} -> b//10 in {6,7,8,9,10}; a//10 in {6} -> b//10 in {5,6,7,8,9,10}; a//10 in {7} -> b//10 in {4,5,6,7}; a//10 in {8} -> b//10 in {3,4,5,6}; a//10 in {9,10} -> b//10 in {2,3,4,5,10} | off (on 0) | - | same |
| L24 down c112 | 16% / 0 | **res//10** (R2 0.84): (tens) res in {120..142} | off (on 0) | - | same |
| L24 down c207 | 6% / 7% | **tens(a,b)** (R2 0.55): a//10 in {0} -> b//10 in {9}; a//10 in {8} -> b//10 in {10}; a//10 in {9} -> b//10 in {0,8,9,10}; a//10 in {10} -> b//10 in {8,9,10} | **a** (R2 0.61): a in {93..99} | - | same |
| L24 down c350 | 9% / 13% | **res//10** (R2 0.71): (tens) res in {2..43, 200} | unexplained (best R2 0.49) | - | same |
| L24 gate c34 | 41% / 9% | **res//10** (R2 0.86): (tens) res in {2..15, 17, 114..200} | **res%100** (R2 0.52): res mod 100 in {0, 10..11} | (reads) | same |
| L24 gate c51 | 18% / 0 | **res//10** (R2 0.88): (tens) res in {140..190, 192, 194..198, 200} | off (on 0) | (reads) | same |
| L24 o c23 (H13) | 68% / 83% | **tens(a,b)** (R2 0.56): a//10 in {0,1} -> b//10 in {6,7,8,9}; a//10 in {2,3} -> b//10 in {5,6,7,8,9}; a//10 in {4} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {5} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {6} -> b//10 in {1,2,3,4,5,7,8,9,10}; a//10 in {7,8,9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | unexplained (best R2 0.40) | - | same |
| L24 up c8 | 26% / 29% | **res//10** (R2 0.72): (tens) res in {52..85, 169} | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {6,7,8}; a//10 in {2,3,4} -> b//10 in {7,8}; a//10 in {5} -> b//10 in {8,9}; a//10 in {6} -> b//10 in {0,1,2,9,10}; a//10 in {7} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1,2,3}; a//10 in {9} -> b//10 in {1,2,3,4}; a//10 in {10} -> b//10 in {2,3,4} | (reads) | same |
| L24 up c17 | 21% / 13% | **res//10** (R2 0.74): (tens) res in {38..58, 140..156} [coarser: res mod 100 in {39..56}, R2 0.97] | unexplained (best R2 0.46) | (reads) | same |

</details>

