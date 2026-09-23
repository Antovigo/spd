# Appendix C1 — components whose main position is `=`, L0-14

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

<details><summary>layer 0: 8 components with main position `=`</summary>

<details><summary>down: 2</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L0 down c4 | 100% / 100% | always | same | a: mod25 +2%, mod10 +2%; b: mod50 +2%, mod2 -3% | a: mod10 +3%; b: mod2 -3% |
| L0 down c37 | 100% / 100% | always | same | a: mod50 +2%, mod20 +3% | - |

</details>

<details><summary>gate: 2</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L0 gate c136 | 100% / 100% | always | same | (reads) | same |
| L0 gate c254 | 100% / 100% | always | same | (reads) | same |

</details>

<details><summary>up: 3</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L0 up c4 | 100% / 100% | always | same | (reads) | same |
| L0 up c206 | 100% / 100% | always | same | (reads) | same |
| L0 up c413 | 100% / 100% | always | same | (reads) | same |

</details>

<details><summary>v: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L0 v c34 (kv2) | 0 / 100% | off (on 0) | always | (reads) | same |

</details>

</details>

<details><summary>layer 1: 10 components with main position `=`</summary>

<details><summary>down: 2</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L1 down c9 | 100% / 100% | always | same | - | same |
| L1 down c20 | 100% / 100% | always | same | - | same |

</details>

<details><summary>gate: 4</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L1 gate c27 | 100% / 0 | always | off (on 0) | (reads) | same |
| L1 gate c38 | 100% / 100% | always | same | (reads) | same |
| L1 gate c147 | 100% / 100% | always | same | (reads) | same |
| L1 gate c579 | 100% / 100% | always | same | (reads) | same |

</details>

<details><summary>o: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L1 o c370 (H6) | 100% / 0 | always | off (on 0) | a: mod20 +2% | - |

</details>

<details><summary>q: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L1 q c66 (H5) | 100% / 100% | always | same | (reads) | same |

</details>

<details><summary>up: 2</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L1 up c14 | 100% / 100% | always | same | (reads) | same |
| L1 up c61 | 100% / 100% | always | same | (reads) | same |

</details>

</details>

<details><summary>layer 2: 12 components with main position `=`</summary>

<details><summary>down: 3</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L2 down c3 | 100% / 0 | always | off (on 0) | a: mod5 +3% | - |
| L2 down c4 | 100% / 100% | always | same | - | same |
| L2 down c266 | 0 / 100% | off (on 0) | always | - | same |

</details>

<details><summary>gate: 2</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L2 gate c4 | 100% / 100% | always | same | (reads) | same |
| L2 gate c5 | 100% / 0 | always | off (on 0) | (reads) | same |

</details>

<details><summary>o: 5</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L2 o c2 (H2) | 100% / 0 | always | off (on 0) | a: mod100 +3%, mod50 +3%, mod25 +4%, mod20 +9%, mod10 +12%, mod5 +24%, mod4 +6%, mod2 +14%; b: mod4 +3% | - |
| L2 o c3 (H2) | 100% / 100% | always | same | a: mod100 +7%, mod50 +4%, mod25 +4%, mod20 +4%, mod10 +7%, mod5 +11%, mod4 +6%, mod2 +10% | a: mod25 +4%, mod10 +8%, mod4 +3% |
| L2 o c6 (H2) | 0 / 100% | off (on 0) | always | - | a: mod100 +12%, mod50 +15%, mod25 +5%, mod20 +17%, mod10 +8%, mod4 +8% |
| L2 o c14 (H2) | 0 / 100% | off (on 0) | always | - | a: mod100 +4%, mod25 +3%, mod10 +8%, mod4 +3%; b: mod100 +3%, mod50 +3%, mod20 +2% |
| L2 o c217 (H2) | 15% / 0 | **a** (R2 0.87): a in {1..14} | off (on 0) | a: mod100 +13%, mod50 +15%, mod25 +11%, mod20 +11% | - |

</details>

<details><summary>q: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L2 q c20 (H15) | 100% / 100% | always | same | (reads) | same |

</details>

<details><summary>up: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L2 up c4 | 100% / 100% | always | same | (reads) | same |

</details>

</details>

<details><summary>layer 3: 8 components with main position `=`</summary>

<details><summary>down: 2</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L3 down c5 | 100% / 100% | always | same | - | same |
| L3 down c28 | 0 / 100% | off (on 0) | always | - | same |

</details>

<details><summary>gate: 3</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L3 gate c9 | 0 / 100% | off (on 0) | always | (reads) | same |
| L3 gate c15 | 100% / 100% | always | same | (reads) | same |
| L3 gate c58 | 100% / 0 | always | off (on 0) | (reads) | same |

</details>

<details><summary>o: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L3 o c154 (H27) | 100% / 0 | always | off (on 0) | - | same |

</details>

<details><summary>q: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L3 q c109 (H5) | 0 / 63% | off (on 0) | **tens(a,b)** (R2 0.85): a//10 in {1,5} -> b//10 in {5,6,7,8,9,10}; a//10 in {2,3,4} -> b//10 in {6,7,8,9,10}; a//10 in {6} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {7,8,9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | (reads) | same |

</details>

<details><summary>up: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L3 up c15 | 100% / 100% | always | same | (reads) | same |

</details>

</details>

<details><summary>layer 4: 9 components with main position `=`</summary>

<details><summary>down: 3</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L4 down c0 | 100% / 100% | always | same | a: mod10 +4% | res: mod50 +3%, mod25 +4%, mod20 +3% |
| L4 down c4 | 12% / 8% | **tens(a,b)** (R2 0.70): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1}; a//10 in {2} -> b//10 in {0} | **a** (R2 0.65): a in {1..7} | a: mod100 +5%, mod50 +5%, mod25 +5%, mod20 +5% | a: mod100 +6%, mod50 +9%, mod25 +8%, mod20 +9%, mod4 +3% |
| L4 down c16 | 0 / 100% | off (on 0) | always | - | same |

</details>

<details><summary>gate: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L4 gate c0 | 100% / 100% | always | same | (reads) | same |

</details>

<details><summary>o: 2</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L4 o c14 (H24) | 100% / 0 | always | off (on 0) | - | same |
| L4 o c23 (H25) | 100% / 100% | always | same | - | same |

</details>

<details><summary>q: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L4 q c3 (H3) | 100% / 100% | always | same | (reads) | same |

</details>

<details><summary>up: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L4 up c265 | 100% / 100% | always | same | (reads) | same |

</details>

<details><summary>v: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L4 v c69 (kv0) | 100% / 100% | always | same | (reads) | same |

</details>

</details>

<details><summary>layer 5: 16 components with main position `=`</summary>

<details><summary>down: 5</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L5 down c0 | 100% / 0 | always | off (on 0) | - | same |
| L5 down c3 | 100% / 0 | always | off (on 0) | a: mod10 +3%, mod5 +2% | - |
| L5 down c9 | 100% / 100% | always | same | - | a: mod4 +2%; res: mod50 +2%, mod25 +3%, mod20 +3% |
| L5 down c10 | 0 / 29% | off (on 0) | **tens(a,b)** (R2 0.83): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {8,9,10}; a//10 in {4} -> b//10 in {9,10} | - | a: mod100 +20%, mod50 +19%, mod25 +9%, mod20 +15%, mod4 +4%; res: mod100 +3% |
| L5 down c82 | 0 / 74% | off (on 0) | **tens(a,b)** (R2 0.73): a//10 in {1,2,3,4} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {5,6,7} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {8,9,10} -> b//10 in {3,4,5,6,7,8,9,10} | - | a: mod100 +3%, mod50 +5%, mod25 +3%, mod20 +3%; b: mod100 +21%, mod50 +15%, mod25 +8%, mod20 +11%, mod10 +3%, mod4 +3%; res: mod100 +9%, mod50 +6%, mod25 +3%, mod20 +3%, mod5 +2% |

</details>

<details><summary>gate: 2</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L5 gate c8 | 100% / 100% | always | same | (reads) | same |
| L5 gate c9 | 100% / 0 | always | off (on 0) | (reads) | same |

</details>

<details><summary>o: 6</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L5 o c22 (H31) | 0 / 99% | off (on 0) | always | - | a: mod100 +3%, mod50 +3%, mod4 +2%; b: mod100 +2%, mod50 +3%; res: mod100 +3% |
| L5 o c75 (H31) | 0 / 49% | off (on 0) | **cmp(a,b)** (R2 0.94): cmp(a,b)=-1: 0.98, cmp(a,b)=0: 0.00, cmp(a,b)=1: 0.01 | - | a: mod100 +9%, mod50 +7%, mod25 +3%, mod20 +4%; b: mod100 +5%, mod50 +3%; res: mod100 +6%, mod50 +5%, mod25 +3%, mod20 +3% |
| L5 o c141 (H30) | 98% / 1% | always | unexplained (best R2 0.16) | b: mod100 +2% | - |
| L5 o c310 (H31) | 0 / 33% | off (on 0) | **res//10** (R2 0.78): (tens) res in {21..99} | - | a: mod100 +3%; b: mod100 +8%, mod50 +5%, mod25 +4%, mod20 +5%; res: mod100 +8% |
| L5 o c423 (H14) | 100% / 0 | always | off (on 0) | - | same |
| L5 o c450 (H31) | 0 / 5% | off (on 0) | **tens(a,b)** (R2 0.62): a//10 in {0} -> b//10 in {0,1,2}; a//10 in {1} -> b//10 in {2} | - | same |

</details>

<details><summary>up: 2</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L5 up c25 | 100% / 100% | always | same | (reads) | same |
| L5 up c254 | 0 / 19% | off (on 0) | **tens(a,b)** (R2 0.85): a//10 in {0} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {7,8,9,10} | (reads) | same |

</details>

<details><summary>v: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L5 v c136 (kv0) | 100% / 100% | always | same | (reads) | same |

</details>

</details>

<details><summary>layer 6: 11 components with main position `=`</summary>

<details><summary>down: 4</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L6 down c11 | 0 / 100% | off (on 0) | always | - | same |
| L6 down c17 | 100% / 100% | always | same | - | b: mod20 +2% |
| L6 down c47 | 0 / 25% | off (on 0) | **tens(a,b)** (R2 0.87): a//10 in {0} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {9,10} | - | a: mod100 +14%, mod50 +13%, mod25 +7%, mod20 +11%, mod4 +3%; res: mod50 +3% |
| L6 down c264 | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>gate: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L6 gate c246 | 100% / 100% | always | same | (reads) | same |

</details>

<details><summary>k: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L6 k c60 (kv7) | 0 / 100% | off (on 0) | always | (reads) | same |

</details>

<details><summary>o: 2</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L6 o c290 (H7) | 0 / 51% | off (on 0) | **cmp(a,b)** (R2 0.98): cmp(a,b)=-1: 0.01, cmp(a,b)=0: 1.00, cmp(a,b)=1: 1.00 | - | a: mod100 +4%, mod50 +3%, mod25 +2%; b: mod100 +4%, mod50 +2%; res: mod100 +6%, mod50 +5%, mod25 +4%, mod20 +4%, mod10 +3% |
| L6 o c310 (H7) | 0 / 35% | off (on 0) | **tens(a,b)** (R2 0.79): a//10 in {0} -> b//10 in {1}; a//10 in {1} -> b//10 in {2,3}; a//10 in {2} -> b//10 in {3,4,5,6,7}; a//10 in {3} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {4} -> b//10 in {5,6,7,8,9,10}; a//10 in {5,6} -> b//10 in {6,7,8,9,10}; a//10 in {7} -> b//10 in {7,8,9,10}; a//10 in {8,9} -> b//10 in {9,10} | - | b: mod100 +4%, mod50 +2%; res: mod100 +10%, mod50 +9%, mod25 +6%, mod20 +6%, mod10 +4%, mod5 +3% |

</details>

<details><summary>up: 2</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L6 up c200 | 0 / 100% | off (on 0) | always | (reads) | same |
| L6 up c416 | 100% / 0 | always | off (on 0) | (reads) | same |

</details>

<details><summary>v: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L6 v c150 (kv6) | 100% / 100% | always | same | (reads) | same |

</details>

</details>

<details><summary>layer 7: 13 components with main position `=`</summary>

<details><summary>down: 5</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L7 down c1 | 100% / 100% | always | same | - | a: mod50 +2%; b: mod20 +2% |
| L7 down c23 | 0 / 100% | off (on 0) | always | - | a: mod20 +2% |
| L7 down c28 | 50% / 2% | **tens(a,b)** (R2 0.63): a//10 in {0,1,2} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,2,3,4,5,6,10}; a//10 in {4} -> b//10 in {0,1,3,4,5,10}; a//10 in {5} -> b//10 in {0,4,5}; a//10 in {6,7,8,9} -> b//10 in {0}; a//10 in {10} -> b//10 in {0,10} | unexplained (best R2 0.33) | a: mod100 +14%, mod50 +7%, mod25 +7%, mod20 +10%, mod10 +6%; b: mod100 +6%, mod50 +6%, mod25 +3%, mod20 +6%, mod10 +2%, mod4 +3% | - |
| L7 down c42 | 0 / 37% | off (on 0) | **tens(a,b)** (R2 0.75): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,9,10}; a//10 in {4} -> b//10 in {0,10}; a//10 in {5,6,7,8} -> b//10 in {0}; a//10 in {9} -> b//10 in {0,1,2}; a//10 in {10} -> b//10 in {0,1,2,3,4,10} | - | a: mod100 +13%, mod50 +13%, mod25 +5%, mod20 +6%; b: mod100 +3%, mod50 +7%, mod25 +5%, mod20 +6%, mod4 +3%; res: mod100 +4% |
| L7 down c186 | 0 / 14% | off (on 0) | unexplained (best R2 0.49) | - | b: mod100 +5%, mod50 +2%, mod25 +2%, mod20 +3%; res: mod100 +10%, mod50 +11%, mod25 +9%, mod20 +10%, mod10 +6%, mod5 +3% |

</details>

<details><summary>gate: 2</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L7 gate c87 | 100% / 100% | always | same | (reads) | same |
| L7 gate c906 | 0 / 43% | off (on 0) | **tens(a,b)** (R2 0.82): a//10 in {0,1} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {3,4} -> b//10 in {5,6,7,8,9,10}; a//10 in {5,6} -> b//10 in {6,7,8,9,10}; a//10 in {7} -> b//10 in {7,8,9,10}; a//10 in {8,9} -> b//10 in {9,10} | (reads) | same |

</details>

<details><summary>k: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L7 k c140 (kv0) | 100% / 100% | always | same | (reads) | same |

</details>

<details><summary>o: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L7 o c11 (H23) | 99% / 89% | always | unexplained (best R2 0.34) | - | same |

</details>

<details><summary>up: 3</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L7 up c28 | 39% / 2% | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,3,4,5,6,8,9,10}; a//10 in {2} -> b//10 in {0,1,2,3,4,5,6,9,10}; a//10 in {3} -> b//10 in {0,1,2,3,4,10}; a//10 in {4} -> b//10 in {0,4}; a//10 in {5,6,7,8,9} -> b//10 in {0}; a//10 in {10} -> b//10 in {10} | unexplained (best R2 0.40) | (reads) | same |
| L7 up c42 | 0 / 100% | off (on 0) | always | (reads) | same |
| L7 up c768 | 0 / 92% | off (on 0) | **tens(a,b)** (R2 0.70): a//10 in {0} -> b//10 in {0,1,2}; a//10 in {1} -> b//10 in {0,1,2,3,4,5,6,7,9}; a//10 in {2,3,4,5,6,7,8,9,10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | (reads) | same |

</details>

<details><summary>v: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L7 v c10 (kv6) | 100% / 100% | always | same | (reads) | same |

</details>

</details>

<details><summary>layer 8: 14 components with main position `=`</summary>

<details><summary>down: 5</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L8 down c1 | 100% / 100% | always | same | b: mod50 +2% | - |
| L8 down c8 | 0 / 40% | off (on 0) | **tens(a,b)** (R2 0.65): a//10 in {0} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,8,9,10}; a//10 in {4} -> b//10 in {0,9,10}; a//10 in {5,6,8} -> b//10 in {0}; a//10 in {7} -> b//10 in {0,1}; a//10 in {9} -> b//10 in {0,1,2,3}; a//10 in {10} -> b//10 in {0,1,2,3,4,10} | - | a: mod100 +14%, mod50 +14%, mod25 +6%, mod20 +6%; b: mod50 +5%, mod25 +3%, mod20 +4%, mod4 +6%; res: mod100 +7% |
| L8 down c33 | 0 / 46% | off (on 0) | **cmp(a,b)** (R2 0.76): cmp(a,b)=-1: 0.02, cmp(a,b)=0: 0.64, cmp(a,b)=1: 0.89 | - | a: mod100 +3%; b: mod100 +18%, mod50 +11%, mod25 +9%, mod20 +9%, mod10 +3%; res: mod100 +8%, mod50 +4%, mod25 +3%, mod20 +2% |
| L8 down c84 | 35% / 0 | **tens(a,b)** (R2 0.63): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,3,4,5,6,9,10}; a//10 in {2} -> b//10 in {0,1,2,3,4,5,10}; a//10 in {3} -> b//10 in {0,1,3,10}; a//10 in {4} -> b//10 in {0,4}; a//10 in {5,6,7,8,9} -> b//10 in {0}; a//10 in {10} -> b//10 in {0,10} | off (on 0) | a: mod100 +14%, mod50 +9%, mod25 +5%, mod20 +7%, mod10 +5%; b: mod100 +9%, mod50 +9%, mod25 +5%, mod20 +9%, mod10 +5%, mod5 +2%; res: mod100 -3%, mod2 +4% | - |
| L8 down c928 | 0 / 100% | off (on 0) | always | - | a: mod50 +2%, mod20 +2%; b: mod100 +2%, mod20 +3% |

</details>

<details><summary>gate: 3</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L8 gate c8 | 100% / 100% | always | same | (reads) | same |
| L8 gate c512 | 0 / 100% | off (on 0) | always | (reads) | same |
| L8 gate c834 | 100% / 100% | always | same | (reads) | same |

</details>

<details><summary>k: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L8 k c25 (kv1) | 91% / 94% | unexplained (best R2 0.43) | unexplained (best R2 0.31) | (reads) | same |

</details>

<details><summary>o: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L8 o c16 (H23) | 0 / 87% | off (on 0) | unexplained (best R2 0.43) | - | res: mod100 +3% |

</details>

<details><summary>up: 3</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L8 up c1 | 100% / 100% | always | same | (reads) | same |
| L8 up c8 | 26% / 33% | unexplained (best R2 0.50) | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,8,9,10}; a//10 in {4} -> b//10 in {0,9,10}; a//10 in {5,6,7} -> b//10 in {0}; a//10 in {9} -> b//10 in {0,1}; a//10 in {10} -> b//10 in {0,1,2,3,4,10} | (reads) | same |
| L8 up c604 | 0 / 100% | off (on 0) | always | (reads) | same |

</details>

<details><summary>v: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L8 v c236 (kv1) | 0 / 87% | off (on 0) | unexplained (best R2 0.44) | (reads) | same |

</details>

</details>

<details><summary>layer 9: 13 components with main position `=`</summary>

<details><summary>down: 5</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L9 down c1 | 100% / 100% | always | same | - | a: mod25 +2%, mod4 +2% |
| L9 down c6 | 0 / 23% | off (on 0) | **tens(a,b)** (R2 0.85): a//10 in {0} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {8,9,10} | - | a: mod100 +5%, mod50 +5%, mod25 +2%, mod20 +2%; b: mod100 +3%, mod50 +2% |
| L9 down c7 | 81% / 0 | **tens(a,b)** (R2 0.53): a//10 in {0} -> b//10 in {7,8}; a//10 in {1} -> b//10 in {2,3,4,5,6,7,8,9}; a//10 in {2} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {3,4,5,6,9} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {7,8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {10} -> b//10 in {1,2,3,4,5,6,7,8,9} | off (on 0) | a: mod100 +16%, mod50 +8%, mod25 +3%, mod20 +5%, mod10 +3%, mod5 +2%, mod2 +4%; b: mod100 +13%, mod50 +13%, mod25 +7%, mod20 +7%, mod10 +7%, mod5 +3%, mod4 +2%; res: mod100 +5%, mod50 +9% | - |
| L9 down c10 | 0 / 37% | off (on 0) | **tens(a,b)** (R2 0.70): a//10 in {0} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,7,8,9,10}; a//10 in {3} -> b//10 in {0,10}; a//10 in {4,5,6} -> b//10 in {0}; a//10 in {7,8} -> b//10 in {0,1,2}; a//10 in {9} -> b//10 in {0,1,2,3,4,5}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,10} | - | a: mod100 +7%, mod50 +7%, mod25 +2%; b: mod100 +5%, mod50 +9%, mod25 +8%, mod20 +10%, mod10 +3%, mod4 +5%; res: mod100 +8%, mod50 +3% |
| L9 down c50 | 0 / 100% | off (on 0) | always | - | a: mod50 +3%, mod20 +3% |

</details>

<details><summary>gate: 4</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L9 gate c80 | 100% / 100% | always | same | (reads) | same |
| L9 gate c332 | 0 / 35% | off (on 0) | **res//10** (R2 0.74): (tens) res in {20..99} | (reads) | same |
| L9 gate c539 | 0 / 17% | off (on 0) | **tens(a,b)** (R2 0.83): a//10 in {0} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {8,9,10} | (reads) | same |
| L9 gate c702 | 100% / 100% | always | same | (reads) | same |

</details>

<details><summary>o: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L9 o c409 (H22) | 97% / 0 | always | off (on 0) | a: mod100 +2%; b: mod100 +3%, mod50 +3%, mod25 +2% | - |

</details>

<details><summary>up: 3</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L9 up c1 | 100% / 100% | always | same | (reads) | same |
| L9 up c286 | 0 / 16% | off (on 0) | **tens(a,b)** (R2 0.82): a//10 in {0} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {8,9,10} | (reads) | same |
| L9 up c567 | 100% / 0 | always | off (on 0) | (reads) | same |

</details>

</details>

<details><summary>layer 10: 13 components with main position `=`</summary>

<details><summary>down: 5</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L10 down c9 | 100% / 100% | always | same | - | same |
| L10 down c252 | 0 / 98% | off (on 0) | always | - | a: mod100 +3%; res: mod100 +4%, mod50 +2% |
| L10 down c255 | 88% / 0 | unexplained (best R2 0.40) | off (on 0) | a: mod100 +10%, mod50 +6%, mod25 +3%, mod5 +3%, mod2 +4%; b: mod100 +11%, mod50 +11%, mod25 +8%, mod20 +6%, mod10 +7%, mod5 +3%, mod4 +2%; res: mod50 +8% | - |
| L10 down c385 | 0 / 53% | off (on 0) | **tens(a,b)** (R2 0.78): a//10 in {2} -> b//10 in {1,2}; a//10 in {3} -> b//10 in {1,2,3}; a//10 in {4} -> b//10 in {1,2,3,4}; a//10 in {5} -> b//10 in {1,2,3,4,5}; a//10 in {6} -> b//10 in {0,1,2,3,4,5,6,7,8}; a//10 in {7} -> b//10 in {0,1,2,3,4,5,6,7,8,9}; a//10 in {8} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {9,10} -> b//10 in {1,2,3,4,5,6,7,8,9} | - | a: mod100 +8%, mod50 +5%, mod25 +3%, mod20 +3%, mod4 +2%; b: mod100 +3%, mod50 +3%; res: mod100 +7%, mod50 +7%, mod25 +5%, mod20 +4%, mod10 +2% |
| L10 down c611 | 0 / 91% | off (on 0) | **tens(a,b)** (R2 0.57): a//10 in {0} -> b//10 in {0,1,2}; a//10 in {1,4,5,10} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {2,3,6,7,8,9} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | - | a: mod100 +5%, mod50 +6%, mod25 +3%, mod20 +3%; b: mod100 +13%, mod50 +10%, mod25 +9%, mod20 +11%, mod10 +3%, mod4 +4%; res: mod100 +10%, mod50 +5%, mod25 +3%, mod20 +3%, mod10 +3%, mod5 +5% |

</details>

<details><summary>gate: 2</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L10 gate c498 | 1% / 0 | unexplained (best R2 0.32) | off (on 0) | (reads) | same |
| L10 gate c712 | 0 / 99% | off (on 0) | always | (reads) | same |

</details>

<details><summary>k: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L10 k c58 (kv7) | 100% / 100% | always | same | (reads) | same |

</details>

<details><summary>o: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L10 o c54 (H15) | 100% / 100% | always | same | - | same |

</details>

<details><summary>up: 3</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L10 up c17 | 0 / 37% | off (on 0) | **tens(a,b)** (R2 0.70): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,9,10}; a//10 in {4} -> b//10 in {0,10}; a//10 in {5,6,7,8} -> b//10 in {0}; a//10 in {9} -> b//10 in {0,1,2,3}; a//10 in {10} -> b//10 in {0,1,2,3,4,10} | (reads) | same |
| L10 up c397 | 100% / 100% | always | same | (reads) | same |
| L10 up c793 | 100% / 0 | always | off (on 0) | (reads) | same |

</details>

<details><summary>v: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L10 v c142 (kv1) | 100% / 100% | always | same | (reads) | same |

</details>

</details>

<details><summary>layer 11: 11 components with main position `=`</summary>

<details><summary>down: 6</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L11 down c11 | 99% / 0 | always | off (on 0) | a: mod100 +3%, mod5 +3%, mod2 +3%; b: mod100 +4%, mod50 +5%, mod25 +3%, mod20 +2%; res: mod50 +2% | - |
| L11 down c34 | 0 / 20% | off (on 0) | **tens(a,b)** (R2 0.82): a//10 in {0} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {5,6,7,8,9,10}; a//10 in {3} -> b//10 in {10} | - | a: mod100 +6%, mod50 +7%, mod25 +3%, mod20 +5% |
| L11 down c75 | 0 / 36% | off (on 0) | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,7,8,9,10}; a//10 in {3} -> b//10 in {0,10}; a//10 in {4,5,6} -> b//10 in {0}; a//10 in {7} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1,2,3}; a//10 in {9} -> b//10 in {0,1,2,3,4,5}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | - | a: mod100 +3%, mod10 +2%; b: mod50 +4%, mod25 +4%, mod20 +5%, mod10 +3%, mod4 +2%; res: mod100 +9% |
| L11 down c176 | 0 / 10% | off (on 0) | **tens(a,b)** (R2 0.70): a//10 in {0} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {5,7,8,10} | - | same |
| L11 down c437 | 35% / 1% | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,3,4,9,10}; a//10 in {2} -> b//10 in {0,1,2,3,4,10}; a//10 in {3} -> b//10 in {0,1,3,4,10}; a//10 in {4} -> b//10 in {0,4}; a//10 in {5} -> b//10 in {0,5}; a//10 in {6,7,8,9} -> b//10 in {0}; a//10 in {10} -> b//10 in {0,10} | unexplained (best R2 0.10) | a: mod100 +8%, mod50 +6%, mod25 +4%, mod20 +4%, mod10 +4%; b: mod100 +7%, mod50 +7%, mod25 +5%, mod20 +7%, mod10 +4%, mod5 +2%, mod4 +2%; res: mod50 +5% | - |
| L11 down c468 | 0 / 14% | off (on 0) | **tens(a,b)** (R2 0.63): a//10 in {6} -> b//10 in {6,7,8}; a//10 in {7,8} -> b//10 in {6,7,8,9}; a//10 in {9} -> b//10 in {9,10}; a//10 in {10} -> b//10 in {6,7,9} | - | b: mod100 +5%, mod50 +2%, mod25 +2%; res: mod100 +5% |

</details>

<details><summary>gate: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L11 gate c366 | 0 / 21% | off (on 0) | **tens(a,b)** (R2 0.74): a//10 in {0} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {5,6,7,8,9,10}; a//10 in {3} -> b//10 in {10}; a//10 in {10} -> b//10 in {0,2} | (reads) | same |

</details>

<details><summary>o: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L11 o c428 (H4) | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>up: 3</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L11 up c140 | 93% / 0 | unexplained (best R2 0.45) | off (on 0) | (reads) | same |
| L11 up c335 | 0 / 0 | off (on 0) | same | (reads) | same |
| L11 up c431 | 0 / 100% | off (on 0) | always | (reads) | same |

</details>

</details>

<details><summary>layer 12: 10 components with main position `=`</summary>

<details><summary>down: 5</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L12 down c1 | 1% / 0 | unexplained (best R2 0.30) | off (on 0) | - | same |
| L12 down c4 | 86% / 0 | unexplained (best R2 0.46) | off (on 0) | a: mod100 +11%, mod50 +6%, mod25 +4%, mod20 +4%, mod10 +3%, mod2 +3%; b: mod100 +9%, mod50 +9%, mod25 +6%, mod20 +7%, mod10 +5%, mod5 +3%, mod4 +4%; res: mod100 +4%, mod50 +6% | - |
| L12 down c14 | 100% / 100% | always | same | - | same |
| L12 down c16 | 0 / 42% | off (on 0) | **tens(a,b)** (R2 0.62): a//10 in {0} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3,4} -> b//10 in {0,8,9,10}; a//10 in {5,8} -> b//10 in {0,1}; a//10 in {6} -> b//10 in {0}; a//10 in {7} -> b//10 in {0,1,2}; a//10 in {9} -> b//10 in {0,1,2,3,4,5}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8,10} | - | a: mod100 +13%, mod50 +11%, mod25 +3%, mod20 +3%, mod5 +2%; b: mod50 +6%, mod25 +4%, mod20 +4%, mod10 +3%, mod5 +2%, mod4 +9%, mod2 +2%; res: mod100 +13%, mod5 +2% |
| L12 down c56 | 0 / 100% | off (on 0) | always | - | a: mod50 +3%, mod20 +2%; b: mod5 +3% |

</details>

<details><summary>gate: 2</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L12 gate c108 | 95% / 0 | unexplained (best R2 0.20) | off (on 0) | (reads) | same |
| L12 gate c294 | 0 / 100% | off (on 0) | always | (reads) | same |

</details>

<details><summary>up: 2</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L12 up c16 | 0 / 14% | off (on 0) | **tens(a,b)** (R2 0.75): a//10 in {0} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {8} | (reads) | same |
| L12 up c813 | 0 / 43% | off (on 0) | **tens(a,b)** (R2 0.63): a//10 in {0} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,4,5,6,7,8,9,10}; a//10 in {3,4} -> b//10 in {0,8,9,10}; a//10 in {5} -> b//10 in {0,1}; a//10 in {6} -> b//10 in {0}; a//10 in {7} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1,2,3}; a//10 in {9} -> b//10 in {0,1,2,3,4,5}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10} | (reads) | same |

</details>

<details><summary>v: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L12 v c145 (kv0) | 0 / 0 | off (on 0) | same | (reads) | same |

</details>

</details>

<details><summary>layer 13: 26 components with main position `=`</summary>

<details><summary>down: 5</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L13 down c4 | 100% / 100% | always | same | a: mod100 +2%, mod50 +3%, mod25 +2%; b: mod100 +2%, mod50 +2%, mod25 +2% | - |
| L13 down c5 | 43% / 3% | **tens(a,b)** (R2 0.57): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,3,4,5,6,8,9,10}; a//10 in {2} -> b//10 in {0,1,2,3,4,5,6,9,10}; a//10 in {3} -> b//10 in {0,1,3,4,10}; a//10 in {4} -> b//10 in {0,1,4,10}; a//10 in {5} -> b//10 in {0,5,10}; a//10 in {6,7,8} -> b//10 in {0}; a//10 in {9} -> b//10 in {0,9,10}; a//10 in {10} -> b//10 in {0,1,2,10} | unexplained (best R2 0.29) | a: mod100 +11%, mod50 +8%, mod25 +7%, mod20 +7%, mod10 +7%, mod4 +2%; b: mod100 +9%, mod50 +10%, mod25 +6%, mod20 +9%, mod10 +5%, mod5 +3%, mod4 +3%; res: mod50 +4% | - |
| L13 down c22 | 0 / 19% | off (on 0) | **res//10** (R2 0.56): (tens) res in {0..24, 99} | - | a: mod100 +2%; b: mod4 +2%; res: mod100 +7%, mod50 +10%, mod25 +6%, mod20 +6%, mod10 +3% |
| L13 down c32 | 0 / 54% | off (on 0) | **tens(a,b)** (R2 0.69): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,9,10}; a//10 in {5,6,7} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1,2,3,4}; a//10 in {9} -> b//10 in {0,1,2,3,4,5}; a//10 in {10} -> b//10 in {0,1,2,3,10} | - | a: mod100 +9%, mod50 +9%, mod25 +2%, mod20 +3%; b: mod100 +5%, mod50 +5%, mod25 +4%, mod20 +6%, mod5 +3%, mod4 +3%; res: mod100 +10% |
| L13 down c127 | 0 / 100% | off (on 0) | always | - | a: mod50 +3%; b: mod100 +2%, mod50 +2% |

</details>

<details><summary>gate: 3</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L13 gate c71 | 100% / 0 | always | off (on 0) | (reads) | same |
| L13 gate c227 | 0 / 100% | off (on 0) | always | (reads) | same |
| L13 gate c828 | 0 / 0 | off (on 0) | same | (reads) | same |

</details>

<details><summary>k: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L13 k c34 (kv5) | 0 / 99% | off (on 0) | always | (reads) | same |

</details>

<details><summary>o: 9</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L13 o c1 (H7) | 100% / 100% | always | same | a: mod50 +2%, mod25 +2%; res: mod50 +3%, mod2 +2% | - |
| L13 o c72 (H7) | 5% / 0 | **a** (R2 0.59): a in {1..4} | off (on 0) | a: mod50 +2%, mod25 +2% | - |
| L13 o c145 (H7) | 0 / 100% | off (on 0) | always | - | a: mod100 +2%, mod50 +4%, mod2 +2%; b: mod100 +4%, mod50 +2%, mod20 +2%, mod5 +4%; res: mod100 +2%, mod25 +3%, mod20 +3%, mod10 +3%, mod5 +2% |
| L13 o c186 (H7) | 0 / 1% | off (on 0) | unexplained (best R2 0.36) | - | same |
| L13 o c213 (H7) | 0 / 60% | off (on 0) | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,9,10}; a//10 in {6} -> b//10 in {0,1,2,3,10}; a//10 in {7,8} -> b//10 in {0,1,2,3,4}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7}; a//10 in {10} -> b//10 in {0,1,2,3} | - | a: mod25 +2%, mod4 +5%; b: mod100 +3%; res: mod100 +9%, mod5 +3% |
| L13 o c231 (H7) | 10% / 6% | unexplained (best R2 0.41) | **res%100** (R2 0.65): res mod 100 in {0} | - | same |
| L13 o c304 (H7) | 8% / 21% | unexplained (best R2 0.33) | **res//10** (R2 0.55): (tens) res in {-1..20, 99} | - | res: mod100 +3%, mod50 +5%, mod25 +5%, mod20 +5%, mod10 +3% |
| L13 o c379 (H7) | 1% / 1% | **cmp(a,b)** (R2 0.92): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 0.96, cmp(a,b)=1: 0.00 | **cmp(a,b)** (R2 0.97): cmp(a,b)=-1: 0.00, cmp(a,b)=0: 1.00, cmp(a,b)=1: 0.00 | - | same |
| L13 o c414 (H7) | 26% / 0 | **tens(a,b)** (R2 0.78): a//10 in {0} -> b//10 in {10}; a//10 in {1} -> b//10 in {0,10}; a//10 in {2} -> b//10 in {0,1,10}; a//10 in {3,4} -> b//10 in {0,1}; a//10 in {5,6,7} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1,2,3}; a//10 in {9} -> b//10 in {0,1,2,3,4}; a//10 in {10} -> b//10 in {0,1,2,3,4,5} | off (on 0) | b: mod100 +11%, mod50 +7%, mod25 +3%, mod4 +2%; res: mod100 +4% | - |

</details>

<details><summary>up: 6</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L13 up c5 | 38% / 1% | **tens(a,b)** (R2 0.58): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,3,4,5,6,8,9,10}; a//10 in {2} -> b//10 in {0,1,2,3,4,9,10}; a//10 in {3} -> b//10 in {0,1,3,10}; a//10 in {4} -> b//10 in {0,4,10}; a//10 in {5} -> b//10 in {0,5}; a//10 in {6,7,8,9} -> b//10 in {0}; a//10 in {10} -> b//10 in {0,1,10} | unexplained (best R2 0.23) | (reads) | same |
| L13 up c22 | 0 / 16% | off (on 0) | **res//10** (R2 0.53): (tens) res in {0..14, 18..19, 99} | (reads) | same |
| L13 up c29 | 0 / 100% | off (on 0) | always | (reads) | same |
| L13 up c367 | 0 / 0 | off (on 0) | same | (reads) | same |
| L13 up c375 | 0 / 0 | off (on 0) | same | (reads) | same |
| L13 up c529 | 0 / 51% | off (on 0) | **tens(a,b)** (R2 0.68): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,9,10}; a//10 in {5,6,7} -> b//10 in {0,1,2}; a//10 in {8} -> b//10 in {0,1,2,3}; a//10 in {9} -> b//10 in {0,1,2,3,4,5}; a//10 in {10} -> b//10 in {0,1,2,3,10} | (reads) | same |

</details>

<details><summary>v: 2</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L13 v c1 (kv4) | 100% / 100% | always | same | (reads) | same |
| L13 v c31 (kv1) | 0 / 100% | off (on 0) | always | (reads) | same |

</details>

</details>

<details><summary>layer 14: 16 components with main position `=`</summary>

<details><summary>down: 5</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L14 down c1 | 100% / 0 | always | off (on 0) | a: mod100 +4%, mod50 +5%, mod25 +5%, mod20 +2%, mod10 +2%, mod5 +2%, mod4 +4%; b: mod100 +3%, mod50 +4%, mod25 +4%, mod20 +3%; res: mod50 +3% | - |
| L14 down c6 | 100% / 100% | always | same | - | same |
| L14 down c23 | 38% / 2% | **tens(a,b)** (R2 0.61): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,2,3,4,5,8,9,10}; a//10 in {2} -> b//10 in {0,1,2,3,4,5,9,10}; a//10 in {3} -> b//10 in {0,1,3,10}; a//10 in {4} -> b//10 in {0,4,10}; a//10 in {5,6,7,8,9} -> b//10 in {0}; a//10 in {10} -> b//10 in {0,1,10} | unexplained (best R2 0.33) | a: mod100 +9%, mod50 +8%, mod25 +7%, mod20 +7%, mod10 +4%, mod4 +3%; b: mod100 +6%, mod50 +8%, mod25 +5%, mod20 +6%, mod10 +2%; res: mod50 +3% | - |
| L14 down c43 | 0 / 23% | off (on 0) | **tens(a,b)** (R2 0.83): a//10 in {0,1} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {3,4} -> b//10 in {8,9,10} | - | a: mod100 +6%, mod50 +5%, mod20 +3%; b: mod50 +3%; res: mod100 +2% |
| L14 down c72 | 1% / 69% | unexplained (best R2 0.07) | **tens(a,b)** (R2 0.64): a//10 in {0} -> b//10 in {0,1,2,3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {0,1,3,4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {0,1,4,5,6,7,8,9,10}; a//10 in {3} -> b//10 in {0,1,7,8,9,10}; a//10 in {4} -> b//10 in {0,1,2,8,9,10}; a//10 in {5} -> b//10 in {0,1,2,3,9,10}; a//10 in {6} -> b//10 in {0,1,2,3,4,10}; a//10 in {7,8} -> b//10 in {0,1,2,3,4,5}; a//10 in {9} -> b//10 in {0,1,2,3,4,5,6,7,8}; a//10 in {10} -> b//10 in {0,1,2,3,4,5,7,8,10} | - | a: mod100 +4%, mod50 +4%; b: mod100 +7%, mod50 +5%, mod25 +5%, mod20 +5%, mod10 +2%; res: mod100 +11% |

</details>

<details><summary>gate: 4</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L14 gate c1 | 100% / 0 | always | off (on 0) | (reads) | same |
| L14 gate c79 | 100% / 100% | always | same | (reads) | same |
| L14 gate c128 | 89% / 0 | unexplained (best R2 0.43) | off (on 0) | (reads) | same |
| L14 gate c145 | 0 / 18% | off (on 0) | **tens(a,b)** (R2 0.84): a//10 in {0} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {1} -> b//10 in {4,5,6,7,8,9,10}; a//10 in {2} -> b//10 in {6,7,8,9,10}; a//10 in {3} -> b//10 in {10} | (reads) | same |

</details>

<details><summary>o: 2</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L14 o c84 (H31) | 100% / 100% | always | same | a: mod100 +2%, mod50 +4%, mod25 +3%, mod20 +3%, mod4 +4%; b: mod50 +3%, mod25 +3%; res: mod50 +3%, mod2 +5% | - |
| L14 o c101 (H18) | 0 / 11% | off (on 0) | **a//10** (R2 0.83): (tens) a in {90..100} | - | a: mod100 +2%, mod50 +3%, mod25 +3%, mod20 +5% |

</details>

<details><summary>q: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L14 q c115 (H23) | 99% / 0 | always | off (on 0) | (reads) | same |

</details>

<details><summary>up: 3</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L14 up c1 | 0 / 72% | off (on 0) | **tens(a,b)** (R2 0.66): a//10 in {0} -> b//10 in {0,1,2}; a//10 in {1} -> b//10 in {1,2,3}; a//10 in {2} -> b//10 in {1,2,3,4,5,6,7,8,9}; a//10 in {3,4,5} -> b//10 in {1,2,3,4,5,6,7,8,9,10}; a//10 in {6,8} -> b//10 in {2,3,4,5,6,7,8,9,10}; a//10 in {7,10} -> b//10 in {3,4,5,6,7,8,9,10}; a//10 in {9} -> b//10 in {3,5,6,7,8,9,10} | (reads) | same |
| L14 up c158 | 0 / 100% | off (on 0) | always | (reads) | same |
| L14 up c254 | 0 / 0 | off (on 0) | same | (reads) | same |

</details>

<details><summary>v: 1</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L14 v c129 (kv7) | 100% / 100% | always | same | (reads) | same |

</details>

</details>

