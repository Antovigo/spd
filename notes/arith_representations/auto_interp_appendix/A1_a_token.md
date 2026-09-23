# Appendix A1 — every component whose main position is `a`

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

## By on-set class

<details><summary>always: 265 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L0 down c3 | 100% / 100% | always | same | - | same |
| L0 down c9 | 100% / 100% | always | same | - | same |
| L0 down c21 | 100% / 100% | always | same | - | same |
| L0 gate c3 | 100% / 100% | always | same | (reads) | same |
| L0 gate c31 | 100% / 100% | always | same | (reads) | same |
| L0 gate c227 | 100% / 100% | always | same | (reads) | same |
| L0 up c3 | 100% / 100% | always | same | (reads) | same |
| L0 up c17 | 100% / 100% | always | same | (reads) | same |
| L0 o c7 (H0) | 100% / 100% | always | same | - | same |
| L0 q c21 (H2) | 100% / 100% | always | same | (reads) | same |
| L0 v c28 (kv7) | 100% / 100% | always | same | (reads) | same |
| L1 v c4 (kv2) | 100% / 100% | always | same | (reads) | same |
| L1 down c39 | 100% / 100% | always | same | - | same |
| L1 down c17 | 100% / 100% | always | same | a: mod100 +3% | a: mod100 +3% |
| L1 o c231 (H26) | 100% / 100% | always | same | - | same |
| L1 q c0 (H23) | 100% / 100% | always | same | (reads) | same |
| L1 q c2 (H6) | 100% / 100% | always | same | (reads) | same |
| L1 gate c39 | 100% / 100% | always | same | (reads) | same |
| L1 k c71 (kv4) | 100% / 100% | always | same | (reads) | same |
| L1 k c10 (kv4) | 100% / 100% | always | same | (reads) | same |
| L1 gate c45 | 100% / 100% | always | same | (reads) | same |
| L1 up c31 | 100% / 100% | always | same | (reads) | same |
| L1 down c13 | 100% / 100% | always | same | - | same |
| L2 gate c12 | 100% / 100% | always | same | (reads) | same |
| L2 o c89 (H23) | 100% / 100% | always | same | - | same |
| L2 up c10 | 100% / 100% | always | same | (reads) | same |
| L2 v c210 (kv1) | 100% / 100% | always | same | (reads) | same |
| L2 k c123 (kv3) | 100% / 100% | always | same | (reads) | same |
| L2 k c8 (kv2) | 100% / 100% | always | same | (reads) | same |
| L2 q c26 (H21) | 100% / 100% | always | same | (reads) | same |
| L3 gate c36 | 100% / 100% | always | same | (reads) | same |
| L3 gate c0 | 100% / 100% | always | same | (reads) | same |
| L3 k c2 (kv2) | 100% / 100% | always | same | (reads) | same |
| L3 k c0 (kv0) | 100% / 100% | always | same | (reads) | same |
| L3 up c397 | 100% / 100% | always | same | (reads) | same |
| L3 up c47 | 100% / 100% | always | same | (reads) | same |
| L3 up c11 | 100% / 100% | always | same | (reads) | same |
| L3 v c70 (kv1) | 100% / 100% | always | same | (reads) | same |
| L3 o c354 (H5) | 100% / 100% | always | same | - | same |
| L3 q c1 (H19) | 100% / 100% | always | same | (reads) | same |
| L3 o c128 (H5) | 100% / 100% | always | same | - | same |
| L3 gate c49 | 100% / 100% | always | same | (reads) | same |
| L3 down c0 | 100% / 100% | always | same | - | same |
| L4 down c32 | 100% / 100% | always | same | - | same |
| L4 q c1 (H18) | 100% / 100% | always | same | (reads) | same |
| L4 gate c17 | 100% / 100% | always | same | (reads) | same |
| L4 v c6 (kv6) | 100% / 100% | always | same | (reads) | same |
| L4 k c68 (kv0) | 100% / 100% | always | same | (reads) | same |
| L4 down c1 | 100% / 100% | always | same | - | same |
| L4 k c13 (kv1) | 100% / 100% | always | same | (reads) | same |
| L4 q c0 (H20) | 100% / 100% | always | same | (reads) | same |
| L4 up c88 | 100% / 100% | always | same | (reads) | same |
| L5 o c6 (H26) | 100% / 100% | always | same | - | same |
| L5 down c1 | 100% / 100% | always | same | - | same |
| L5 up c216 | 100% / 100% | always | same | (reads) | same |
| L5 up c184 | 100% / 100% | always | same | (reads) | same |
| L5 down c7 | 100% / 100% | always | same | - | same |
| L5 k c56 (kv5) | 100% / 100% | always | same | (reads) | same |
| L5 k c0 (kv4) | 100% / 100% | always | same | (reads) | same |
| L5 gate c104 | 99% / 99% | always | same | (reads) | same |
| L5 gate c1 | 100% / 100% | always | same | (reads) | same |
| L5 q c6 (H10) | 100% / 100% | always | same | (reads) | same |
| L6 q c123 (H0) | 100% / 100% | always | same | (reads) | same |
| L6 k c137 (kv4) | 100% / 100% | always | same | (reads) | same |
| L6 down c0 | 100% / 100% | always | same | - | same |
| L6 down c15 | 100% / 100% | always | same | - | same |
| L6 k c2 (kv1) | 100% / 100% | always | same | (reads) | same |
| L6 v c1 (kv6) | 100% / 100% | always | same | (reads) | same |
| L6 up c650 | 100% / 100% | always | same | (reads) | same |
| L6 gate c37 | 100% / 100% | always | same | (reads) | same |
| L6 up c15 | 100% / 100% | always | same | (reads) | same |
| L7 gate c117 | 100% / 100% | always | same | (reads) | same |
| L7 up c0 | 100% / 100% | always | same | (reads) | same |
| L7 gate c367 | 100% / 100% | always | same | (reads) | same |
| L7 down c0 | 100% / 100% | always | same | - | same |
| L7 down c2 | 100% / 100% | always | same | - | same |
| L7 gate c352 | 100% / 100% | always | same | (reads) | same |
| L7 v c230 (kv0) | 100% / 100% | always | same | (reads) | same |
| L7 o c3 (H5) | 100% / 100% | always | same | - | same |
| L7 q c87 (H22) | 100% / 100% | always | same | (reads) | same |
| L7 up c2 | 100% / 100% | always | same | (reads) | same |
| L7 o c14 (H20) | 100% / 100% | always | same | - | same |
| L8 up c81 | 100% / 100% | always | same | (reads) | same |
| L8 gate c38 | 100% / 100% | always | same | (reads) | same |
| L8 down c0 | 100% / 100% | always | same | - | same |
| L8 k c30 (kv3) | 100% / 100% | always | same | (reads) | same |
| L8 down c69 | 100% / 100% | always | same | - | same |
| L8 q c141 (H1) | 100% / 100% | always | same | (reads) | same |
| L8 v c204 (kv5) | 100% / 100% | always | same | (reads) | same |
| L9 o c183 (H11) | 100% / 100% | always | same | - | same |
| L9 o c9 (H11) | 100% / 100% | always | same | - | same |
| L9 up c185 | 100% / 100% | always | same | (reads) | same |
| L9 gate c91 | 100% / 100% | always | same | (reads) | same |
| L9 down c38 | 100% / 100% | always | same | - | same |
| L9 up c847 | 100% / 100% | always | same | (reads) | same |
| L9 v c212 (kv3) | 100% / 100% | always | same | (reads) | same |
| L9 v c1 (kv1) | 100% / 100% | always | same | (reads) | same |
| L9 q c79 (H7) | 100% / 100% | always | same | (reads) | same |
| L10 v c241 (kv1) | 100% / 100% | always | same | (reads) | same |
| L10 down c2 | 100% / 100% | always | same | - | same |
| L10 gate c21 | 100% / 100% | always | same | (reads) | same |
| L10 q c0 (H4) | 100% / 100% | always | same | (reads) | same |
| L10 up c509 | 100% / 100% | always | same | (reads) | same |
| L10 up c482 | 100% / 100% | always | same | (reads) | same |
| L11 q c2 (H29) | 100% / 100% | always | same | (reads) | same |
| L11 down c0 | 100% / 100% | always | same | - | same |
| L11 up c805 | 100% / 100% | always | same | (reads) | same |
| L11 up c17 | 100% / 100% | always | same | (reads) | same |
| L11 gate c481 | 100% / 100% | always | same | (reads) | same |
| L11 v c16 (kv5) | 100% / 100% | always | same | (reads) | same |
| L11 down c1 | 100% / 100% | always | same | - | same |
| L11 gate c350 | 100% / 100% | always | same | (reads) | same |
| L11 o c4 (H20) | 100% / 100% | always | same | - | same |
| L12 q c142 (H20) | 100% / 100% | always | same | (reads) | same |
| L12 up c0 | 100% / 100% | always | same | (reads) | same |
| L12 v c200 (kv0) | 100% / 100% | always | same | (reads) | same |
| L12 up c752 | 100% / 100% | always | same | (reads) | same |
| L12 k c67 (kv3) | 100% / 100% | always | same | (reads) | same |
| L12 o c41 (H20) | 100% / 100% | always | same | - | same |
| L12 o c209 (H20) | 100% / 100% | always | same | - | same |
| L12 gate c2 | 100% / 100% | always | same | (reads) | same |
| L12 gate c19 | 100% / 100% | always | same | (reads) | same |
| L12 down c0 | 100% / 100% | always | same | - | same |
| L12 up c36 | 100% / 100% | always | same | (reads) | same |
| L12 down c8 | 100% / 100% | always | same | - | same |
| L13 q c121 (H7) | 100% / 100% | always | same | (reads) | same |
| L13 down c0 | 100% / 100% | always | same | - | same |
| L13 v c68 (kv6) | 100% / 100% | always | same | (reads) | same |
| L13 o c0 (H16) | 100% / 100% | always | same | - | same |
| L13 up c629 | 100% / 100% | always | same | (reads) | same |
| L13 up c0 | 100% / 100% | always | same | (reads) | same |
| L13 gate c86 | 100% / 100% | always | same | (reads) | same |
| L13 down c2 | 100% / 100% | always | same | - | same |
| L14 up c151 | 100% / 100% | always | same | (reads) | same |
| L14 q c3 (H31) | 100% / 100% | always | same | (reads) | same |
| L14 v c9 (kv7) | 100% / 100% | always | same | (reads) | same |
| L14 up c2 | 100% / 100% | always | same | (reads) | same |
| L14 gate c5 | 100% / 100% | always | same | (reads) | same |
| L14 down c0 | 100% / 100% | always | same | a: mod100 +2% | a: mod100 +2% |
| L15 gate c69 | 100% / 100% | always | same | (reads) | same |
| L15 down c1 | 100% / 100% | always | same | - | same |
| L15 o c3 (H5) | 100% / 100% | always | same | - | same |
| L15 q c142 (H12) | 100% / 100% | always | same | (reads) | same |
| L15 up c1 | 100% / 100% | always | same | (reads) | same |
| L15 k c13 (kv3) | 100% / 100% | always | same | (reads) | same |
| L15 gate c91 | 100% / 100% | always | same | (reads) | same |
| L16 down c0 | 100% / 100% | always | same | - | same |
| L16 down c31 | 100% / 100% | always | same | - | same |
| L16 down c4 | 100% / 100% | always | same | - | same |
| L16 k c5 (kv5) | 100% / 100% | always | same | (reads) | same |
| L16 o c207 (H30) | 100% / 100% | always | same | - | same |
| L16 q c119 (H6) | 100% / 100% | always | same | (reads) | same |
| L16 up c37 | 100% / 100% | always | same | (reads) | same |
| L16 v c2 (kv3) | 100% / 100% | always | same | (reads) | same |
| L16 gate c5 | 100% / 100% | always | same | (reads) | same |
| L16 gate c40 | 100% / 100% | always | same | (reads) | same |
| L17 k c9 (kv2) | 100% / 100% | always | same | (reads) | same |
| L17 gate c502 | 100% / 100% | always | same | (reads) | same |
| L17 down c4 | 100% / 100% | always | same | - | same |
| L17 v c1 (kv4) | 100% / 100% | always | same | (reads) | same |
| L17 up c16 | 100% / 100% | always | same | (reads) | same |
| L17 q c127 (H8) | 100% / 100% | always | same | (reads) | same |
| L18 up c50 | 100% / 100% | always | same | (reads) | same |
| L18 down c3 | 100% / 100% | always | same | - | same |
| L18 down c1 | 100% / 100% | always | same | - | same |
| L18 gate c456 | 100% / 100% | always | same | (reads) | same |
| L18 k c95 (kv1) | 100% / 100% | always | same | (reads) | same |
| L18 up c138 | 100% / 100% | always | same | (reads) | same |
| L18 gate c164 | 100% / 100% | always | same | (reads) | same |
| L18 q c92 (H7) | 100% / 100% | always | same | (reads) | same |
| L19 down c9 | 100% / 100% | always | same | - | same |
| L19 down c45 | 100% / 100% | always | same | - | same |
| L19 v c92 (kv3) | 100% / 100% | always | same | (reads) | same |
| L19 up c7 | 100% / 100% | always | same | (reads) | same |
| L19 o c23 (H7) | 100% / 100% | always | same | - | same |
| L19 k c65 (kv6) | 100% / 100% | always | same | (reads) | same |
| L19 gate c448 | 100% / 100% | always | same | (reads) | same |
| L19 q c3 (H29) | 100% / 100% | always | same | (reads) | same |
| L20 down c0 | 100% / 100% | always | same | - | same |
| L20 gate c542 | 100% / 100% | always | same | (reads) | same |
| L20 up c5 | 100% / 100% | always | same | (reads) | same |
| L20 up c22 | 100% / 100% | always | same | (reads) | same |
| L20 k c26 (kv5) | 100% / 100% | always | same | (reads) | same |
| L20 q c25 (H13) | 100% / 100% | always | same | (reads) | same |
| L20 v c223 (kv4) | 100% / 100% | always | same | (reads) | same |
| L21 down c11 | 100% / 100% | always | same | - | same |
| L21 gate c21 | 100% / 100% | always | same | (reads) | same |
| L21 up c199 | 100% / 100% | always | same | (reads) | same |
| L21 q c3 (H28) | 100% / 100% | always | same | (reads) | same |
| L21 v c249 (kv4) | 100% / 100% | always | same | (reads) | same |
| L22 k c1 (kv7) | 100% / 100% | always | same | (reads) | same |
| L22 q c1 (H22) | 100% / 100% | always | same | (reads) | same |
| L22 v c74 (kv1) | 99% / 99% | always | same | (reads) | same |
| L22 up c12 | 100% / 100% | always | same | (reads) | same |
| L22 down c0 | 100% / 100% | always | same | - | same |
| L22 down c2 | 99% / 99% | always | same | - | same |
| L22 gate c37 | 100% / 100% | always | same | (reads) | same |
| L23 q c3 (H6) | 100% / 100% | always | same | (reads) | same |
| L23 k c2 (kv1) | 99% / 99% | always | same | (reads) | same |
| L23 v c166 (kv0) | 99% / 99% | always | same | (reads) | same |
| L23 up c69 | 100% / 100% | always | same | (reads) | same |
| L23 down c6 | 100% / 100% | always | same | - | same |
| L23 down c22 | 99% / 99% | always | same | - | same |
| L23 gate c88 | 100% / 100% | always | same | (reads) | same |
| L24 up c0 | 100% / 100% | always | same | (reads) | same |
| L24 v c211 (kv6) | 100% / 100% | always | same | (reads) | same |
| L24 gate c3 | 99% / 99% | always | same | (reads) | same |
| L24 gate c271 | 99% / 99% | always | same | (reads) | same |
| L24 k c6 (kv5) | 100% / 100% | always | same | (reads) | same |
| L24 down c3 | 100% / 100% | always | same | - | same |
| L24 o c39 (H14) | 100% / 100% | always | same | - | same |
| L24 q c3 (H25) | 100% / 100% | always | same | (reads) | same |
| L24 down c0 | 100% / 100% | always | same | - | same |
| L25 up c89 | 100% / 100% | always | same | (reads) | same |
| L25 down c0 | 100% / 100% | always | same | - | same |
| L25 gate c84 | 100% / 100% | always | same | (reads) | same |
| L25 o c394 (H23) | 100% / 100% | always | same | - | same |
| L25 k c2 (kv1) | 100% / 100% | always | same | (reads) | same |
| L25 q c2 (H14) | 100% / 100% | always | same | (reads) | same |
| L25 v c18 (kv5) | 100% / 100% | always | same | (reads) | same |
| L26 k c1 (kv3) | 100% / 100% | always | same | (reads) | same |
| L26 down c18 | 100% / 100% | always | same | - | same |
| L26 gate c54 | 100% / 100% | always | same | (reads) | same |
| L26 gate c41 | 100% / 100% | always | same | (reads) | same |
| L26 q c0 (H15) | 100% / 100% | always | same | (reads) | same |
| L26 down c47 | 100% / 100% | always | same | - | same |
| L26 o c19 (H10) | 100% / 100% | always | same | - | same |
| L26 v c32 (kv2) | 100% / 100% | always | same | (reads) | same |
| L27 v c155 (kv1) | 100% / 100% | always | same | (reads) | same |
| L27 up c45 | 100% / 100% | always | same | (reads) | same |
| L27 gate c1 | 100% / 100% | always | same | (reads) | same |
| L27 k c5 (kv0) | 100% / 100% | always | same | (reads) | same |
| L27 down c1 | 100% / 100% | always | same | - | same |
| L27 down c2 | 100% / 100% | always | same | - | same |
| L27 q c6 (H25) | 100% / 100% | always | same | (reads) | same |
| L27 o c5 (H14) | 100% / 100% | always | same | - | same |
| L28 gate c0 | 100% / 100% | always | same | (reads) | same |
| L28 v c138 (kv6) | 100% / 100% | always | same | (reads) | same |
| L28 down c0 | 100% / 100% | always | same | - | same |
| L28 k c1 (kv3) | 100% / 100% | always | same | (reads) | same |
| L28 q c0 (H24) | 100% / 100% | always | same | (reads) | same |
| L28 up c260 | 100% / 100% | always | same | (reads) | same |
| L28 gate c50 | 100% / 100% | always | same | (reads) | same |
| L29 q c22 (H12) | 100% / 100% | always | same | (reads) | same |
| L29 v c8 (kv5) | 100% / 100% | always | same | (reads) | same |
| L29 down c3 | 100% / 100% | always | same | - | same |
| L29 up c787 | 100% / 100% | always | same | (reads) | same |
| L29 k c1 (kv5) | 100% / 100% | always | same | (reads) | same |
| L29 down c9 | 100% / 100% | always | same | - | same |
| L29 up c5 | 100% / 100% | always | same | (reads) | same |
| L29 gate c261 | 100% / 100% | always | same | (reads) | same |
| L29 gate c131 | 100% / 100% | always | same | (reads) | same |
| L30 q c2 (H3) | 100% / 100% | always | same | (reads) | same |
| L30 gate c124 | 100% / 100% | always | same | (reads) | same |
| L30 up c367 | 100% / 100% | always | same | (reads) | same |
| L30 gate c378 | 100% / 100% | always | same | (reads) | same |
| L30 down c480 | 100% / 100% | always | same | - | same |
| L30 v c134 (kv1) | 100% / 100% | always | same | (reads) | same |
| L30 down c1 | 100% / 100% | always | same | - | same |
| L30 down c32 | 100% / 100% | always | same | - | same |
| L30 k c53 (kv7) | 100% / 100% | always | same | (reads) | same |
| L30 gate c191 | 100% / 100% | always | same | (reads) | same |
| L31 v c43 (kv3) | 100% / 100% | always | same | (reads) | same |
| L31 k c4 (kv3) | 100% / 100% | always | same | (reads) | same |
| L31 v c28 (kv3) | 100% / 100% | always | same | (reads) | same |

</details>

<details><summary>rarely on (< 0.5 %): 8 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L1 k c74 (kv5) | 0 / 0 | off (on 0) | same | (reads) | same |
| L1 k c119 (kv5) | 0 / 0 | off (on 0) | same | (reads) | same |
| L2 k c80 (kv2) | 0 / 0 | off (on 0) | same | (reads) | same |
| L5 v c121 (kv5) | 0 / 0 | off (on 0) | same | (reads) | same |
| L20 v c81 (kv0) | 0 / 0 | off (on 0) | same | (reads) | same |
| L20 v c253 (kv0) | 0 / 0 | off (on 0) | same | (reads) | same |
| L29 q c56 (H27) | 0 / 0 | off (on 0) | same | (reads) | same |
| L31 down c25 | 0 / 0 | off (on 0) | same | - | same |

</details>

<details><summary>residue mod 10: 118 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L0 down c36 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {1} | same | a: mod10 +9%, mod5 +8%, mod2 +9% | a: mod10 +9%, mod5 +8%, mod2 +9% |
| L0 down c38 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {9} | same | a: mod10 +7%, mod5 +7%, mod2 +7% | a: mod10 +7%, mod5 +7%, mod2 +7% |
| L0 down c44 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {7} | same | a: mod10 +5%, mod5 +6%, mod2 +8% | a: mod10 +5%, mod5 +6%, mod2 +8% |
| L0 down c45 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {5} | same | a: mod10 +7%, mod5 +7%, mod2 +8% | a: mod10 +7%, mod5 +7%, mod2 +8% |
| L0 down c50 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {8} | same | a: mod10 +6%, mod5 +6%, mod2 +9% | a: mod10 +6%, mod5 +6%, mod2 +9% |
| L0 down c52 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {3} | same | a: mod10 +7%, mod5 +6%, mod2 +10% | a: mod10 +7%, mod5 +6%, mod2 +10% |
| L0 down c55 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {6} | same | a: mod10 +6%, mod5 +6%, mod2 +7% | a: mod10 +6%, mod5 +6%, mod2 +7% |
| L0 down c62 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {4} | same | a: mod10 +6%, mod5 +7%, mod2 +7% | a: mod10 +6%, mod5 +7%, mod2 +7% |
| L0 down c81 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {2} | same | a: mod10 +6%, mod5 +6%, mod2 +5% | a: mod10 +6%, mod5 +6%, mod2 +5% |
| L0 down c145 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {0} | same | a: mod10 +6%, mod5 +4%, mod2 +6% | a: mod10 +6%, mod5 +4%, mod2 +6% |
| L0 gate c38 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {9} | same | (reads) | same |
| L0 gate c44 | 20% / 20% | **a%10** (R2 1.00): a mod 10 in {4, 7} | same | (reads) | same |
| L0 gate c55 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {6} | same | (reads) | same |
| L0 gate c81 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {2} | same | (reads) | same |
| L0 gate c349 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {8} | same | (reads) | same |
| L0 up c36 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {1} | same | (reads) | same |
| L0 up c45 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {5} | same | (reads) | same |
| L0 up c145 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {0} | same | (reads) | same |
| L1 v c43 (kv5) | 11% / 12% | **a%10** (R2 0.89): a mod 10 in {5} | **a%10** (R2 0.86): a mod 10 in {5} | (reads) | same |
| L1 v c154 (kv5) | 10% / 9% | **a%10** (R2 0.90): a mod 10 in {4} | **a%10** (R2 0.82): a mod 10 in {4} | (reads) | same |
| L3 gate c63 | 30% / 30% | **a%10** (R2 1.00): a mod 10 in {5..7} | same | (reads) | same |
| L3 gate c68 | 29% / 29% | **a%10** (R2 0.96): a mod 10 in {3..4, 9} | same | (reads) | same |
| L3 down c63 | 30% / 30% | **a%10** (R2 1.00): a mod 10 in {5..7} | same | a: mod10 +9%, mod5 +4%, mod2 +4% | a: mod10 +9%, mod5 +4%, mod2 +4% |
| L3 gate c34 | 20% / 20% | **a%10** (R2 1.00): a mod 10 in {1, 9} | same | (reads) | same |
| L3 down c136 | 37% / 37% | **a%10** (R2 0.91): a mod 10 in {2, 7..9} | same | a: mod10 +9%, mod5 +3%, mod2 +2% | a: mod10 +9%, mod5 +3%, mod2 +2% |
| L3 down c54 | 20% / 20% | **a%10** (R2 1.00): a mod 10 in {1, 9} | same | a: mod10 +4%, mod5 +4%, mod2 +11% | a: mod10 +4%, mod5 +4%, mod2 +11% |
| L4 down c66 | 39% / 39% | **a%10** (R2 0.96): a mod 10 in {6..9} | same | a: mod10 +13% | a: mod10 +13% |
| L4 gate c66 | 39% / 39% | **a%10** (R2 0.96): a mod 10 in {6..9} | same | (reads) | same |
| L5 v c33 (kv5) | 14% / 9% | **a%10** (R2 0.77): a mod 10 in {0} | **a** (R2 0.62): a in {20, 25, 30, 40, 50, 60, 70, 80, 100} [coarser: a mod 10 in {0}, R2 0.82] | (reads) | same |
| L11 down c36 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {7} | same | a: mod10 +2%, mod5 +2% | a: mod10 +2%, mod5 +2% |
| L11 down c63 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {6} | same | a: mod10 +2%, mod2 +3% | a: mod10 +2%, mod2 +3% |
| L11 down c42 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {0} | same | a: mod10 +3%, mod5 +4%, mod2 +3% | a: mod10 +3%, mod5 +4%, mod2 +3% |
| L11 gate c27 | 19% / 18% | **a%10** (R2 0.94): a mod 10 in {5..6} | **a%50** (R2 0.93): a mod 50 in {5, 15..16, 25..26, 35..36, 45..46} [coarser: a mod 10 in {5..6}, R2 0.90] | (reads) | same |
| L11 gate c36 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {7} | same | (reads) | same |
| L11 gate c63 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {6} | same | (reads) | same |
| L11 gate c20 | 28% / 28% | **a%10** (R2 0.91): a mod 10 in {1..3} | same | (reads) | same |
| L12 down c15 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {1} | same | a: mod10 +4%, mod5 +6%, mod2 +3% | a: mod10 +4%, mod5 +6%, mod2 +3% |
| L12 down c31 | 30% / 27% | **a%10** (R2 0.98): a mod 10 in {5..7} | **a%10** (R2 0.91): a mod 10 in {5..7} | a: mod10 +4%, mod5 +2% | a: mod10 +4%, mod5 +2% |
| L12 gate c15 | 11% / 11% | **a%10** (R2 0.91): a mod 10 in {1} | same | (reads) | same |
| L12 down c7 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {9} | same | a: mod10 +5%, mod5 +5%, mod2 +3% | a: mod10 +5%, mod5 +5%, mod2 +3% |
| L12 gate c32 | 50% / 48% | **a%10** (R2 1.00): a mod 10 in {5..9} | **a%10** (R2 0.94): a mod 10 in {5..9} | (reads) | same |
| L13 down c9 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {7} | same | a: mod10 +6%, mod5 +5%, mod2 +4% | a: mod10 +6%, mod5 +5%, mod2 +4% |
| L13 down c15 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {3} | same | a: mod10 +4%, mod5 +5%, mod2 +4% | a: mod10 +4%, mod5 +5%, mod2 +4% |
| L13 down c13 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {5} | same | a: mod10 +4%, mod5 +6%, mod2 +3% | a: mod10 +4%, mod5 +6%, mod2 +3% |
| L13 down c51 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {4} | same | a: mod10 +2%, mod5 +2% | a: mod10 +2%, mod5 +2% |
| L13 down c92 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {2} | same | - | same |
| L13 down c105 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {1} | same | - | same |
| L13 down c38 | 20% / 20% | **a%10** (R2 1.00): a mod 10 in {0, 9} | same | a: mod10 +4%, mod5 +3% | a: mod10 +4%, mod5 +3% |
| L13 gate c38 | 20% / 20% | **a%10** (R2 1.00): a mod 10 in {0, 9} | same | (reads) | same |
| L13 gate c51 | 20% / 20% | **a%10** (R2 1.00): a mod 10 in {4, 7} | same | (reads) | same |
| L13 gate c105 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {1} | same | (reads) | same |
| L13 up c105 | 19% / 19% | **a%10** (R2 0.94): a mod 10 in {1..2} | **a%10** (R2 0.92): a mod 10 in {1..2} | (reads) | same |
| L13 up c460 | 10% / 8% | **a%10** (R2 1.00): a mod 10 in {9} | **a** (R2 0.93): a in {9, 19, 29, 39, 49, 59, 69, 99} [coarser: a mod 10 in {9}, R2 0.83] | (reads) | same |
| L13 gate c9 | 20% / 20% | **a%10** (R2 1.00): a mod 10 in {4, 7} | same | (reads) | same |
| L14 gate c33 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {6} | same | (reads) | same |
| L14 gate c21 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {9} | same | (reads) | same |
| L14 down c33 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {6} | same | a: mod10 +2%, mod5 +2%, mod2 +3% | a: mod10 +2%, mod5 +2%, mod2 +3% |
| L14 down c71 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {1} | same | a: mod10 +2%, mod5 +2%, mod2 +2% | a: mod10 +2%, mod5 +2%, mod2 +2% |
| L14 down c226 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {5} | same | - | same |
| L14 gate c14 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {4} | same | (reads) | same |
| L14 gate c15 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {2} | same | (reads) | same |
| L14 down c14 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {4} | same | a: mod10 +4%, mod5 +4%, mod2 +4% | a: mod10 +4%, mod5 +4%, mod2 +4% |
| L14 down c15 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {2} | same | a: mod10 +3%, mod5 +3%, mod2 +4% | a: mod10 +3%, mod5 +3%, mod2 +4% |
| L14 down c21 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {9} | same | a: mod10 +3%, mod5 +3%, mod2 +4% | a: mod10 +3%, mod5 +3%, mod2 +4% |
| L14 up c310 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {3} | same | (reads) | same |
| L14 up c268 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {9} | same | (reads) | same |
| L14 up c226 | 9% / 10% | **a%10** (R2 0.93): a mod 10 in {5} | **a%10** (R2 1.00): a mod 10 in {5} | (reads) | same |
| L14 up c154 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {0} | same | (reads) | same |
| L14 up c55 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {1} | same | (reads) | same |
| L14 up c33 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {6} | same | (reads) | same |
| L14 up c14 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {4} | same | (reads) | same |
| L14 gate c226 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {5} | same | (reads) | same |
| L14 gate c71 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {1} | same | (reads) | same |
| L14 v c173 (kv4) | 3% / 10% | **a** (R2 0.92): a in {70, 80, 90} | **a%10** (R2 1.00): a mod 10 in {0} | (reads) | same |
| L15 gate c218 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {9} | same | (reads) | same |
| L15 gate c96 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {1} | same | (reads) | same |
| L15 up c310 | 10% / 10% | **a%10** (R2 0.99): a mod 10 in {9} | **a%10** (R2 1.00): a mod 10 in {9} | (reads) | same |
| L16 down c34 | 10% / 11% | **a%10** (R2 1.00): a mod 10 in {8} | **a%10** (R2 0.93): a mod 10 in {8} | a: mod10 +4%, mod5 +3%, mod2 +4% | a: mod10 +4%, mod5 +3%, mod2 +4% |
| L16 gate c34 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {8} | same | (reads) | same |
| L16 down c74 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {7} | same | a: mod10 +2%, mod5 +2% | a: mod10 +2%, mod5 +2% |
| L16 v c106 (kv5) | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {4} | **a%10** (R2 0.98): a mod 10 in {4} | (reads) | same |
| L16 v c119 (kv5) | 20% / 19% | **a%10** (R2 1.00): a mod 10 in {2, 9} | **a%10** (R2 0.97): a mod 10 in {2, 9} | (reads) | same |
| L16 v c121 (kv5) | 30% / 27% | **a%10** (R2 0.99): a mod 10 in {7..9} | **a%10** (R2 0.87): a mod 10 in {7..9} | (reads) | same |
| L16 v c122 (kv0) | 10% / 7% | **a%10** (R2 0.95): a mod 10 in {5} | **a%10** (R2 0.71): a mod 10 in {5} | (reads) | same |
| L16 v c185 (kv5) | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {8} | same | (reads) | same |
| L16 v c229 (kv5) | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {1} | **a%10** (R2 0.96): a mod 10 in {1} | (reads) | same |
| L16 v c79 (kv0) | 10% / 6% | **a%10** (R2 0.97): a mod 10 in {8} | **a** (R2 0.68): a in {28, 58, 68, 78, 88, 98} [coarser: a mod 10 in {8}, R2 0.81] | (reads) | same |
| L16 up c74 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {7} | same | (reads) | same |
| L16 v c16 (kv0) | 20% / 20% | **a%10** (R2 0.99): a mod 10 in {2, 5} | **a%10** (R2 0.98): a mod 10 in {2, 5} | (reads) | same |
| L16 v c34 (kv0) | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {9} | same | (reads) | same |
| L16 v c39 (kv0) | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {0} | **a%10** (R2 0.99): a mod 10 in {0} | (reads) | same |
| L16 v c40 (kv5) | 22% / 16% | **a%10** (R2 0.85): a mod 10 in {6, 8} | **a%20** (R2 0.74): a mod 20 in {6, 8, 16} [coarser: a mod 10 in {6, 8}, R2 0.90] | (reads) | same |
| L16 v c61 (kv5) | 30% / 30% | **a%10** (R2 1.00): a mod 10 in {3, 6, 9} | same | (reads) | same |
| L16 v c78 (kv5) | 20% / 20% | **a%10** (R2 1.00): a mod 10 in {3..4} | same | (reads) | same |
| L16 v c80 (kv5) | 28% / 25% | **a%10** (R2 0.94): a mod 10 in {0, 3, 7} | **a%10** (R2 0.86): a mod 10 in {0, 3, 7} | (reads) | same |
| L17 down c120 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {9} | same | - | same |
| L17 gate c120 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {9} | same | (reads) | same |
| L17 up c167 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {4} | same | (reads) | same |
| L18 v c175 (kv7) | 10% / 10% | **a%10** (R2 0.97): a mod 10 in {4} | **a%10** (R2 1.00): a mod 10 in {4} | (reads) | same |
| L18 v c213 (kv7) | 10% / 11% | **a%10** (R2 1.00): a mod 10 in {8} | **a%10** (R2 0.91): a mod 10 in {8} | (reads) | same |
| L18 v c24 (kv7) | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {5} | same | (reads) | same |
| L18 gate c40 | 9% / 10% | **a%20** (R2 0.90): a mod 20 in {9, 19} [coarser: a mod 10 in {9}, R2 0.89] | **a%10** (R2 1.00): a mod 10 in {9} | (reads) | same |
| L18 down c40 | 9% / 9% | **a%10** (R2 0.90): a mod 10 in {9} | **a%20** (R2 0.90): a mod 20 in {9, 19} [coarser: a mod 10 in {9}, R2 0.89] | - | same |
| L18 v c158 (kv7) | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {7} | same | (reads) | same |
| L18 v c231 (kv7) | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {3} | same | (reads) | same |
| L19 gate c182 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {9} | same | (reads) | same |
| L19 up c93 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {3} | same | (reads) | same |
| L20 v c172 (kv0) | 4% / 10% | unexplained (best R2 0.44) | **a%10** (R2 0.92): a mod 10 in {5} | (reads) | same |
| L21 gate c104 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {7} | same | (reads) | same |
| L21 gate c81 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {3} | same | (reads) | same |
| L21 down c104 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {7} | same | a: mod2 +2% | a: mod2 +2% |
| L21 down c81 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {3} | same | a: mod2 +2% | a: mod2 +2% |
| L21 gate c256 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {5} | same | (reads) | same |
| L21 down c194 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {5} | same | - | same |
| L22 v c15 (kv3) | 1% / 10% | **a** (R2 0.70): a in {6} | **a%10** (R2 1.00): a mod 10 in {6} | (reads) | same |
| L23 down c46 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {9} | same | - | same |
| L23 gate c46 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {9} | same | (reads) | same |
| L28 gate c174 | 10% / 10% | **a%10** (R2 1.00): a mod 10 in {2} | same | (reads) | same |

</details>

<details><summary>residue mod 20: 26 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L3 v c13 (kv1) | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {1, 11} [coarser: a mod 10 in {1}, R2 0.89] | **a%10** (R2 0.87): a mod 10 in {1} | (reads) | same |
| L3 gate c136 | 34% / 34% | **a%20** (R2 0.91): a mod 20 in {7..9, 12, 17..19} [coarser: a mod 10 in {7..9}, R2 0.89] | same | (reads) | same |
| L11 down c38 | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {3, 13} [coarser: a mod 10 in {3}, R2 0.89] | same | - | same |
| L11 down c40 | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {8, 18} [coarser: a mod 10 in {8}, R2 0.89] | same | a: mod10 +2% | a: mod10 +2% |
| L11 gate c40 | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {8, 18} [coarser: a mod 10 in {8}, R2 0.89] | same | (reads) | same |
| L11 up c40 | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {8, 18} [coarser: a mod 10 in {8}, R2 0.89] | same | (reads) | same |
| L12 down c9 | 25% / 25% | **a%20** (R2 1.00): a mod 20 in {10..14} | same | a: mod20 +10%, mod10 +3%, mod4 +4% | a: mod20 +10%, mod10 +3%, mod4 +4% |
| L13 down c263 | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {0, 10} [coarser: a mod 10 in {0}, R2 0.89] | same | - | same |
| L14 gate c135 | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {2, 12} [coarser: a mod 10 in {2}, R2 0.89] | same | (reads) | same |
| L15 up c771 | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {2, 12} [coarser: a mod 10 in {2}, R2 0.89] | same | (reads) | same |
| L15 down c374 | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {9, 19} [coarser: a mod 10 in {9}, R2 0.89] | same | - | same |
| L15 down c77 | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {1, 11} [coarser: a mod 10 in {1}, R2 0.89] | same | - | same |
| L16 v c200 (kv5) | 5% / 5% | **a%20** (R2 1.00): a mod 20 in {4} | **a%20** (R2 0.99): a mod 20 in {4} | (reads) | same |
| L16 v c43 (kv5) | 26% / 23% | **a%20** (R2 0.95): a mod 20 in {15..19} | **a** (R2 0.93): a in {15, 35..39, 55..59, 75..79, 95..100} [coarser: a mod 20 in {15..19}, R2 0.86] | (reads) | same |
| L16 up c272 | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {3, 13} [coarser: a mod 10 in {3}, R2 0.89] | same | (reads) | same |
| L18 v c54 (kv7) | 9% / 9% | **a%10** (R2 0.88): a mod 10 in {1} | **a%20** (R2 0.90): a mod 20 in {1, 11} [coarser: a mod 10 in {1}, R2 0.89] | (reads) | same |
| L19 down c93 | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {3, 13} [coarser: a mod 10 in {3}, R2 0.89] | same | - | same |
| L20 down c41 | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {2, 12} [coarser: a mod 10 in {2}, R2 0.89] | same | a: mod10 +2%, mod2 +2% | a: mod10 +2%, mod2 +2% |
| L20 gate c41 | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {2, 12} [coarser: a mod 10 in {2}, R2 0.89] | same | (reads) | same |
| L20 gate c22 | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {1, 11} [coarser: a mod 10 in {1}, R2 0.89] | same | (reads) | same |
| L22 k c92 (kv0) | 6% / 9% | **a** (R2 0.93): a in {47, 57, 67, 77, 87, 97} | **a%20** (R2 0.90): a mod 20 in {7, 17} [coarser: a mod 10 in {7}, R2 0.89] | (reads) | same |
| L22 up c39 | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {1, 11} [coarser: a mod 10 in {1}, R2 0.89] | same | (reads) | same |
| L28 down c174 | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {2, 12} [coarser: a mod 10 in {2}, R2 0.89] | same | - | same |
| L29 down c107 | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {1, 11} [coarser: a mod 10 in {1}, R2 0.89] | same | a: mod10 +2% | a: mod10 +2% |
| L29 gate c107 | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {1, 11} [coarser: a mod 10 in {1}, R2 0.89] | same | (reads) | same |
| L30 up c155 | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {1, 11} [coarser: a mod 10 in {1}, R2 0.89] | same | (reads) | same |

</details>

<details><summary>residue mod 25: 4 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L19 up c42 | 20% / 20% | **a%25** (R2 0.91): a mod 25 in {0, 5, 10, 15, 20} [coarser: a mod 5 in {0}, R2 0.88] | same | (reads) | same |
| L22 down c116 | 4% / 4% | **a%25** (R2 1.00): a mod 25 in {0} | same | - | same |
| L24 down c174 | 18% / 18% | **a%50** (R2 0.93): a mod 50 in {0, 10, 15, 20, 25, 30, 35, 40} [coarser: a mod 5 in {0}, R2 0.88] | **a%25** (R2 0.90): a mod 25 in {0, 5, 10, 15, 20} [coarser: a mod 5 in {0}, R2 0.89] | a: mod5 +3% | a: mod5 +3% |
| L29 down c877 | 19% / 19% | **a%25** (R2 0.89): a mod 25 in {0, 10, 15, 20} [coarser: a mod 5 in {0}, R2 0.86] | **a%25** (R2 0.90): a mod 25 in {0, 10, 15, 20} [coarser: a mod 5 in {0}, R2 0.85] | a: mod5 +4% | a: mod5 +4% |

</details>

<details><summary>residue mod 5: 21 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L0 up c162 | 20% / 20% | **a%5** (R2 1.00): a mod 5 in {0} | same | (reads) | same |
| L1 v c230 (kv4) | 20% / 20% | **a%5** (R2 1.00): a mod 5 in {0} | same | (reads) | same |
| L2 down c473 | 20% / 20% | **a%5** (R2 1.00): a mod 5 in {0} | same | a: mod5 +12% | a: mod5 +12% |
| L2 up c35 | 20% / 20% | **a%5** (R2 1.00): a mod 5 in {0} | same | (reads) | same |
| L3 gate c111 | 20% / 20% | **a%5** (R2 1.00): a mod 5 in {0} | same | (reads) | same |
| L3 down c615 | 19% / 19% | **a%5** (R2 0.94): a mod 5 in {0} | same | a: mod5 +8% | a: mod5 +8% |
| L14 down c18 | 20% / 20% | **a%5** (R2 1.00): a mod 5 in {0} | same | a: mod5 +5% | a: mod5 +5% |
| L14 up c18 | 21% / 21% | **a%5** (R2 0.94): a mod 5 in {0} | same | (reads) | same |
| L15 down c70 | 20% / 20% | **a%5** (R2 1.00): a mod 5 in {0} | same | a: mod5 +6% | a: mod5 +6% |
| L15 up c70 | 20% / 20% | **a%5** (R2 1.00): a mod 5 in {0} | same | (reads) | same |
| L17 up c8 | 20% / 20% | **a%5** (R2 1.00): a mod 5 in {0} | same | (reads) | same |
| L17 down c8 | 20% / 20% | **a%5** (R2 1.00): a mod 5 in {0} | same | a: mod5 +12% | a: mod5 +12% |
| L18 v c246 (kv7) | 15% / 19% | **a%10** (R2 0.81): a mod 10 in {0, 5} [coarser: a mod 5 in {0}, R2 0.84] | **a%5** (R2 0.92): a mod 5 in {0} | (reads) | same |
| L18 up c51 | 18% / 18% | **a%5** (R2 0.91): a mod 5 in {0} | **a%25** (R2 0.90): a mod 25 in {0, 5, 10, 15, 20} [coarser: a mod 5 in {0}, R2 0.88] | (reads) | same |
| L20 down c85 | 19% / 19% | **a%5** (R2 0.94): a mod 5 in {0} | same | a: mod5 +4% | a: mod5 +4% |
| L20 up c35 | 20% / 20% | **a%5** (R2 1.00): a mod 5 in {0} | same | (reads) | same |
| L23 up c181 | 19% / 19% | **a%5** (R2 0.94): a mod 5 in {0} | same | (reads) | same |
| L25 up c145 | 20% / 20% | **a%5** (R2 1.00): a mod 5 in {0} | same | (reads) | same |
| L26 up c286 | 19% / 19% | **a%5** (R2 0.94): a mod 5 in {0} | same | (reads) | same |
| L26 down c109 | 19% / 19% | **a%5** (R2 0.94): a mod 5 in {0} | same | a: mod5 +3% | a: mod5 +3% |
| L28 up c239 | 20% / 20% | **a%5** (R2 1.00): a mod 5 in {0} | same | (reads) | same |

</details>

<details><summary>residue mod 50: 58 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L0 gate c49 | 2% / 2% | **a%50** (R2 1.00): a mod 50 in {0} | **a%50** (R2 0.99): a mod 50 in {0} | (reads) | same |
| L1 gate c631 | 2% / 2% | **a%50** (R2 1.00): a mod 50 in {11} | same | (reads) | same |
| L1 v c116 (kv5) | 8% / 11% | **a%50** (R2 0.84): a mod 50 in {1, 11, 21, 31, 41} [coarser: a mod 10 in {1}, R2 0.85] | **a%50** (R2 0.85): a mod 50 in {1, 9, 11, 21, 31, 41} [coarser: a mod 10 in {1}, R2 0.88] | (reads) | same |
| L1 v c213 (kv6) | 1% / 2% | **a** (R2 1.00): a in {1} | **a%50** (R2 0.80): a mod 50 in {1} | (reads) | same |
| L3 down c68 | 38% / 35% | **a%50** (R2 0.91): a mod 50 in {4, 9, 13..14, 17, 19, 23..24, 27, 29, 33..34, 37, 39, 43..44, 49} [coarser: a mod 10 in {3..4, 7, 9}, R2 0.86] | **a%50** (R2 0.92): a mod 50 in {4, 9, 13..14, 17, 19, 23..24, 27, 29, 33..34, 37, 39, 43..44, 49} [coarser: a mod 10 in {3..4, 7, 9}, R2 0.84] | a: mod10 +2%, mod5 +8% | a: mod10 +2%, mod5 +7% |
| L3 v c90 (kv1) | 11% / 11% | **a%50** (R2 0.95): a mod 50 in {0, 10, 20, 25, 30, 40} | same | (reads) | same |
| L4 down c24 | 2% / 2% | **a%50** (R2 1.00): a mod 50 in {0} | same | - | same |
| L9 down c64 | 36% / 36% | **a%50** (R2 0.91): a mod 50 in {9, 11, 13, 17, 19, 21, 23, 29, 31, 33, 37, 39, 41, 43, 47, 49} [coarser: a mod 10 in {1, 3, 7, 9}, R2 0.85] | **a%50** (R2 0.91): a mod 50 in {9, 11, 13, 17, 19, 21, 23, 27, 29, 31, 33, 37, 39, 41, 43, 47, 49} [coarser: a mod 10 in {1, 3, 7, 9}, R2 0.85] | a: mod2 +13% | a: mod2 +13% |
| L9 gate c64 | 36% / 36% | **a%50** (R2 0.91): a mod 50 in {9, 11, 13, 17, 19, 21, 23, 29, 31, 33, 37, 39, 41, 43, 47, 49} [coarser: a mod 10 in {1, 3, 7, 9}, R2 0.85] | **a%50** (R2 0.91): a mod 50 in {9, 11, 13, 17, 19, 21, 23, 27, 29, 31, 33, 37, 39, 41, 43, 47, 49} [coarser: a mod 10 in {1, 3, 7, 9}, R2 0.85] | (reads) | same |
| L11 down c20 | 27% / 27% | **a%50** (R2 0.92): a mod 50 in {2, 11..12, 21..23, 31..33, 41..43} [coarser: a mod 10 in {1..3}, R2 0.87] | same | a: mod10 +8%, mod5 +4% | a: mod10 +8%, mod5 +4% |
| L12 up c428 | 2% / 2% | **a%50** (R2 1.00): a mod 50 in {0} | same | (reads) | same |
| L12 gate c7 | 27% / 26% | **a%50** (R2 0.92): a mod 50 in {6, 9, 12, 16, 19, 26, 29, 32, 36, 39, 42, 46, 49} [coarser: a mod 10 in {2, 6, 9}, R2 0.89] | **a%50** (R2 0.90): a mod 50 in {6, 9, 16, 19, 26, 29, 32, 36, 39, 42, 46, 49} [coarser: a mod 10 in {2, 6, 9}, R2 0.86] | (reads) | same |
| L13 up c433 | 2% / 2% | **a%50** (R2 1.00): a mod 50 in {25} | same | (reads) | same |
| L15 down c532 | 2% / 2% | **a%50** (R2 1.00): a mod 50 in {43} | **a%50** (R2 0.99): a mod 50 in {43} | - | same |
| L16 down c820 | 1% / 2% | **a** (R2 1.00): a in {75} | **a%50** (R2 0.88): a mod 50 in {25} | - | same |
| L16 down c120 | 12% / 12% | **a%50** (R2 0.98): a mod 50 in {0, 10, 20, 25, 30, 40} [coarser: a mod 10 in {0}, R2 0.84] | **a%50** (R2 1.00): a mod 50 in {0, 10, 20, 25, 30, 40} [coarser: a mod 10 in {0}, R2 0.85] | a: mod5 +3% | a: mod5 +3% |
| L16 up c524 | 2% / 2% | **a%50** (R2 1.00): a mod 50 in {23} | same | (reads) | same |
| L16 up c820 | 2% / 2% | **a%50** (R2 1.00): a mod 50 in {25} | same | (reads) | same |
| L16 v c22 (kv0) | 12% / 10% | **a%50** (R2 0.90): a mod 50 in {1, 11, 21, 31, 41} [coarser: a mod 10 in {1}, R2 0.84] | **a%10** (R2 0.89): a mod 10 in {1} | (reads) | same |
| L16 gate c50 | 15% / 16% | **a** (R2 1.00): a in {1, 11, 16, 21, 26, 31, 41, 46, 51, 61, 71, 76, 81, 86, 91} [coarser: a mod 10 in {1, 6}, R2 0.81] | **a%50** (R2 0.92): a mod 50 in {1, 11, 16, 21, 26, 31, 41, 46} [coarser: a mod 10 in {1, 6}, R2 0.83] | (reads) | same |
| L17 gate c834 | 8% / 8% | **a%50** (R2 1.00): a mod 50 in {17, 27, 37, 47} | same | (reads) | same |
| L17 down c704 | 8% / 8% | **a%50** (R2 1.00): a mod 50 in {17, 27, 37, 47} | same | - | same |
| L18 down c281 | 4% / 4% | **a%50** (R2 1.00): a mod 50 in {33, 43} | same | - | same |
| L18 down c51 | 11% / 11% | **a%50** (R2 0.95): a mod 50 in {0, 10, 20, 25, 30} | same | a: mod5 +2% | a: mod5 +2% |
| L18 v c130 (kv7) | 10% / 12% | **a%10** (R2 1.00): a mod 10 in {9} | **a%50** (R2 0.90): a mod 50 in {9, 19, 29, 39, 49} [coarser: a mod 10 in {9}, R2 0.83] | (reads) | same |
| L18 v c180 (kv7) | 0 / 2% | off (on 0) | **a%50** (R2 1.00): a mod 50 in {16} | (reads) | same |
| L18 v c239 (kv7) | 0 / 7% | off (on 0) | **a%50** (R2 0.92): a mod 50 in {23..25} | (reads) | same |
| L18 up c281 | 4% / 4% | **a%50** (R2 1.00): a mod 50 in {33, 43} | same | (reads) | same |
| L19 down c455 | 2% / 2% | **a%50** (R2 1.00): a mod 50 in {49} | same | - | same |
| L19 down c47 | 19% / 19% | **a%50** (R2 0.90): a mod 50 in {0, 15, 20, 25, 30, 35, 40, 45} [coarser: a mod 5 in {0}, R2 0.82] | same | a: mod5 +4% | a: mod5 +4% |
| L20 v c108 (kv0) | 7% / 12% | **a** (R2 0.54): a in {30, 60, 90} | **a%50** (R2 0.75): a mod 50 in {10, 20, 30, 40, 45} [coarser: a mod 10 in {0}, R2 0.86] | (reads) | same |
| L20 down c932 | 2% / 2% | **a%50** (R2 1.00): a mod 50 in {49} | same | - | same |
| L21 down c19 | 12% / 12% | **a%50** (R2 0.90): a mod 50 in {0, 10, 20, 30, 40} [coarser: a mod 10 in {0}, R2 0.85] | same | a: mod10 +3%, mod5 +4%, mod2 +3% | a: mod10 +3%, mod5 +4%, mod2 +3% |
| L21 gate c19 | 14% / 14% | **a%50** (R2 0.92): a mod 50 in {0, 10, 20, 25, 30, 40} [coarser: a mod 10 in {0}, R2 0.80] | same | (reads) | same |
| L21 up c19 | 16% / 16% | **a%50** (R2 0.91): a mod 50 in {0, 10, 20, 25, 30, 40, 45} | **a%50** (R2 0.93): a mod 50 in {0, 10, 20, 25, 30, 40, 45} | (reads) | same |
| L22 down c723 | 2% / 2% | **a%50** (R2 1.00): a mod 50 in {49} | same | - | same |
| L22 gate c7 | 13% / 13% | **a%50** (R2 0.95): a mod 50 in {0, 10, 20, 25, 30, 40} | **a%50** (R2 0.96): a mod 50 in {0, 10, 20, 25, 30, 40} | (reads) | same |
| L22 gate c99 | 2% / 2% | **a%50** (R2 1.00): a mod 50 in {49} | same | (reads) | same |
| L22 up c7 | 13% / 13% | **a%50** (R2 0.96): a mod 50 in {0, 10, 20, 25, 30, 40} [coarser: a mod 10 in {0}, R2 0.81] | same | (reads) | same |
| L22 gate c183 | 2% / 2% | **a%50** (R2 1.00): a mod 50 in {45} | same | (reads) | same |
| L22 up c416 | 7% / 7% | **a%50** (R2 0.92): a mod 50 in {0, 47, 49} | same | (reads) | same |
| L23 down c101 | 13% / 13% | **a%50** (R2 0.96): a mod 50 in {0, 10, 20, 25, 30, 40} [coarser: a mod 10 in {0}, R2 0.81] | same | - | same |
| L23 up c176 | 2% / 2% | **a%50** (R2 1.00): a mod 50 in {38} | same | (reads) | same |
| L24 gate c117 | 18% / 18% | **a%50** (R2 0.93): a mod 50 in {0, 10, 15, 20, 25, 30, 35, 40} [coarser: a mod 5 in {0}, R2 0.88] | same | (reads) | same |
| L25 gate c5 | 27% / 27% | **a%50** (R2 0.92): a mod 50 in {1..3, 21..23, 31..33, 41..43} [coarser: a mod 10 in {1..3}, R2 0.86] | same | (reads) | same |
| L25 gate c129 | 7% / 7% | **a%50** (R2 0.92): a mod 50 in {20, 30, 40} | same | (reads) | same |
| L25 down c5 | 27% / 27% | **a%50** (R2 0.92): a mod 50 in {1..3, 21..23, 31..33, 41..43} [coarser: a mod 10 in {1..3}, R2 0.86] | same | a: mod10 +5% | a: mod10 +5% |
| L26 down c737 | 2% / 2% | **a%50** (R2 1.00): a mod 50 in {0} | same | - | same |
| L27 down c5 | 18% / 18% | **a%50** (R2 0.93): a mod 50 in {2..3, 22..23, 32..33, 42..43} [coarser: a mod 10 in {2..3}, R2 0.88] | same | a: mod10 +7%, mod5 +2% | a: mod10 +7%, mod5 +2% |
| L27 gate c5 | 21% / 21% | **a%50** (R2 0.91): a mod 50 in {2..3, 21..23, 32..33, 42..43} | same | (reads) | same |
| L27 up c123 | 2% / 2% | **a%50** (R2 1.00): a mod 50 in {49} | same | (reads) | same |
| L28 down c915 | 7% / 7% | **a%50** (R2 0.92): a mod 50 in {0, 30, 40} | same | - | same |
| L28 gate c27 | 2% / 2% | **a%50** (R2 1.00): a mod 50 in {49} | same | (reads) | same |
| L28 up c839 | 2% / 2% | **a%50** (R2 1.00): a mod 50 in {49} | same | (reads) | same |
| L29 up c2 | 2% / 2% | **a%50** (R2 1.00): a mod 50 in {0} | same | (reads) | same |
| L29 down c133 | 2% / 2% | **a%50** (R2 1.00): a mod 50 in {0} | same | - | same |
| L30 gate c238 | 11% / 11% | **a%50** (R2 0.95): a mod 50 in {0, 20, 25, 30, 40} | same | (reads) | same |
| L30 down c994 | 11% / 11% | **a%50** (R2 0.93): a mod 50 in {0, 10, 20, 25, 30, 40} | **a%50** (R2 0.94): a mod 50 in {0, 20, 25, 30, 40} | a: mod5 +4% | a: mod5 +4% |

</details>

<details><summary>scattered values: 772 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L0 down c49 | 2% / 2% | **a** (R2 1.00): a in {91, 93} | same | - | same |
| L0 down c54 | 3% / 3% | **a** (R2 1.00): a in {60, 80, 90} | same | - | same |
| L0 down c90 | 10% / 10% | **a** (R2 1.00): a in {57..65, 67} | same | a: mod20 +2% | a: mod20 +2% |
| L0 down c109 | 10% / 10% | **a** (R2 1.00): a in {18, 24, 36, 42, 45, 48, 54, 60, 63, 72} | same | - | same |
| L0 up c127 | 10% / 10% | **a** (R2 1.00): a in {16, 24, 32, 36, 48, 56, 63..64, 72, 96} | same | (reads) | same |
| L0 up c187 | 40% / 40% | **a** (R2 1.00): a in {32..59, 89..100} | same | (reads) | same |
| L0 gate c7 | 56% / 56% | **a** (R2 1.00): a in {43..45, 47..49, 51..100} | same | (reads) | same |
| L0 gate c36 | 10% / 10% | **a** (R2 1.00): a in {11, 21, 31, 38, 41, 51, 61, 71, 81, 91} [coarser: a mod 10 in {1}, R2 0.80] | same | (reads) | same |
| L0 gate c45 | 3% / 3% | **a** (R2 1.00): a in {45, 55, 65} | same | (reads) | same |
| L0 gate c85 | 10% / 10% | **a** (R2 1.00): a in {57..65, 67} | same | (reads) | same |
| L0 gate c109 | 7% / 7% | **a** (R2 1.00): a in {24, 36, 42, 45, 48, 54, 72} | same | (reads) | same |
| L0 gate c184 | 9% / 9% | **a** (R2 1.00): a in {8, 18, 28, 38, 48, 55, 58, 78, 98} [coarser: a mod 50 in {8, 28, 48}, R2 0.82] | same | (reads) | same |
| L0 gate c583 | 8% / 8% | **a** (R2 1.00): a in {20, 30, 40, 50, 60, 70, 80, 90} [coarser: a mod 50 in {20, 30, 40}, R2 0.86] | same | (reads) | same |
| L0 up c50 | 3% / 3% | **a** (R2 1.00): a in {28, 58, 88} | same | (reads) | same |
| L0 down c163 | 2% / 2% | **a** (R2 1.00): a in {32, 64} | same | - | same |
| L0 down c165 | 4% / 4% | **a** (R2 1.00): a in {35, 37..39} | same | - | same |
| L0 down c273 | 2% / 1% | **a** (R2 1.00): a in {9, 90} | **a** (R2 0.89): a in {9} | - | same |
| L0 down c317 | 2% / 2% | **a** (R2 1.00): a in {40, 80} | same | - | same |
| L0 down c318 | 2% / 2% | **a** (R2 1.00): a in {44, 88} | same | - | same |
| L0 down c321 | 2% / 2% | **a** (R2 1.00): a in {24, 52} | same | - | same |
| L0 down c376 | 2% / 2% | **a** (R2 1.00): a in {7, 77} | same | - | same |
| L0 down c467 | 3% / 3% | **a** (R2 1.00): a in {66, 77, 88} | same | - | same |
| L0 up c318 | 4% / 4% | **a** (R2 1.00): a in {33, 44, 55, 88} | same | (reads) | same |
| L0 up c273 | 3% / 3% | **a** (R2 1.00): a in {9, 89..90} | same | (reads) | same |
| L0 up c266 | 3% / 3% | **a** (R2 1.00): a in {6, 26, 66} | same | (reads) | same |
| L0 k c43 (kv6) | 5% / 0 | **a** (R2 1.00): a in {18, 50, 55, 90, 94} | off (on 0) | (reads) | same |
| L0 k c25 (kv6) | 6% / 0 | **a** (R2 1.00): a in {18, 50, 55, 65, 90, 94} | off (on 0) | (reads) | same |
| L0 up c376 | 2% / 2% | **a** (R2 1.00): a in {7, 77} | same | (reads) | same |
| L0 up c351 | 2% / 2% | **a** (R2 1.00): a in {8, 88} | same | (reads) | same |
| L0 up c348 | 2% / 2% | **a** (R2 1.00): a in {4, 94} | same | (reads) | same |
| L1 up c452 | 11% / 11% | **a** (R2 1.00): a in {1..4, 21, 31, 41, 51, 61, 81, 91} | same | (reads) | same |
| L1 up c546 | 13% / 13% | **a** (R2 1.00): a in {19, 29, 39, 47..50, 59, 69, 79, 89, 98..99} [coarser: a mod 50 in {19, 29, 39, 48..49}, R2 0.87] | same | (reads) | same |
| L1 v c183 (kv6) | 4% / 5% | **a** (R2 0.99): a in {10..11, 50, 100} | **a** (R2 1.00): a in {10..12, 50, 100} | (reads) | same |
| L1 v c157 (kv5) | 14% / 16% | **a** (R2 0.94): a in {6, 56, 60..69, 86} | **a** (R2 0.89): a in {6, 56, 59..69, 86} | (reads) | same |
| L1 v c118 (kv5) | 9% / 9% | **a** (R2 0.98): a in {9, 19, 29, 39, 49, 79, 89, 98..99} [coarser: a mod 50 in {9, 29, 39, 49}, R2 0.82] | **a** (R2 0.99): a in {9, 19, 29, 39, 49, 79, 89, 98..99} [coarser: a mod 50 in {9, 29, 39, 49}, R2 0.82] | (reads) | same |
| L1 v c117 (kv5) | 8% / 7% | **a** (R2 0.95): a in {2, 12, 32, 52, 62, 72, 82, 92} [coarser: a mod 20 in {2, 12}, R2 0.81] | **a%50** (R2 0.88): a mod 50 in {2, 12, 32} | (reads) | same |
| L1 v c94 (kv5) | 11% / 7% | **a** (R2 0.86): a in {6, 13, 16, 23, 43, 53, 56, 66, 73, 93, 96} [coarser: a mod 50 in {6, 13, 16, 23, 43, 46}, R2 0.82] | **a** (R2 0.67): a in {6, 13, 16, 23, 43, 53, 56} | (reads) | same |
| L1 v c82 (kv5) | 14% / 16% | **a** (R2 0.98): a in {76..89} | **a** (R2 0.98): a in {8, 75..89} | (reads) | same |
| L1 v c78 (kv5) | 17% / 14% | **a** (R2 0.96): a in {1..4, 21, 31..32, 41, 51, 61..62, 71..72, 81, 91..92, 98} | **a** (R2 0.88): a in {1..4, 21, 31..32, 41, 51, 61..62, 81, 91..92} | (reads) | same |
| L1 v c241 (kv6) | 2% / 2% | **a** (R2 1.00): a in {84, 86} | **a** (R2 0.99): a in {84, 86} | (reads) | same |
| L1 v c71 (kv6) | 15% / 17% | **a** (R2 0.96): a in {1..12, 18, 21} | **a** (R2 0.99): a in {1..12, 18..22} | (reads) | same |
| L1 v c28 (kv5) | 16% / 16% | **a** (R2 0.98): a in {7, 66..80} | **a** (R2 0.97): a in {7, 66..80} | (reads) | same |
| L1 v c53 (kv6) | 91% / 91% | **a** (R2 0.98): a in {1..23, 33..100} | **a** (R2 1.00): a in {1..23, 33..100} | (reads) | same |
| L1 v c56 (kv5) | 4% / 3% | **a** (R2 0.87): a in {35, 37..38} | **a** (R2 0.90): a in {35, 37..38} | (reads) | same |
| L1 v c60 (kv5) | 5% / 5% | **a** (R2 0.98): a in {5, 15, 25, 65, 75} [coarser: a mod 50 in {5, 15, 25}, R2 0.90] | **a** (R2 0.94): a in {5, 15, 25, 65, 75} [coarser: a mod 50 in {5, 15, 25}, R2 0.90] | (reads) | same |
| L1 v c61 (kv5) | 6% / 6% | **a** (R2 0.86): a in {7..10, 98} | **a** (R2 0.82): a in {7..10, 98} | (reads) | same |
| L1 v c64 (kv5) | 8% / 7% | **a** (R2 0.99): a in {86, 88..94} | **a** (R2 0.95): a in {86, 89..94} | (reads) | same |
| L1 v c65 (kv5) | 6% / 6% | **a** (R2 0.94): a in {7, 17, 27, 37, 87, 97} | **a** (R2 0.96): a in {7, 17, 27, 37, 87, 97} | (reads) | same |
| L1 down c80 | 2% / 2% | **a** (R2 1.00): a in {42, 52} | same | - | same |
| L1 up c271 | 14% / 17% | **a** (R2 1.00): a in {50..52, 55, 80..89} | **a** (R2 1.00): a in {49..52, 54..55, 79..89} | (reads) | same |
| L1 down c617 | 11% / 11% | **a** (R2 1.00): a in {1..4, 21, 31, 41, 51, 61, 81, 91} | same | - | same |
| L1 down c481 | 11% / 11% | **a** (R2 1.00): a in {20, 30, 40, 50, 60, 70, 80, 90, 98..100} [coarser: a mod 50 in {0, 20, 30, 40}, R2 0.85] | same | - | same |
| L1 down c337 | 15% / 20% | **a** (R2 0.98): a in {83, 86..99} | **a** (R2 1.00): a in {78, 81..99} | - | same |
| L1 down c335 | 15% / 15% | **a** (R2 1.00): a in {30, 40, 60, 70, 80, 89..98} | **a** (R2 0.99): a in {30, 40, 60, 70, 80, 89..98} | - | same |
| L1 down c271 | 12% / 13% | **a** (R2 0.98): a in {55, 79..89} | **a** (R2 1.00): a in {52, 55, 79..89} | a: mod20 +2% | a: mod20 +2% |
| L1 down c166 | 10% / 10% | **a** (R2 1.00): a in {1..9, 18} | same | - | same |
| L1 o c70 (H26) | 78% / 80% | **a** (R2 1.00): a in {8..12, 21..32, 40..100} | **a** (R2 0.98): a in {7..12, 20..32, 40..100} | - | same |
| L2 up c247 | 8% / 8% | **a** (R2 1.00): a in {9..14, 98..99} | same | (reads) | same |
| L2 down c6 | 18% / 20% | **a** (R2 1.00): a in {62..79} | **a** (R2 1.00): a in {59, 61..79} | - | same |
| L2 gate c313 | 3% / 3% | **a** (R2 1.00): a in {16, 32, 64} | same | (reads) | same |
| L2 gate c282 | 3% / 2% | **a** (R2 1.00): a in {5, 55, 65} [coarser: a mod 50 in {5}, R2 0.83] | **a%50** (R2 0.99): a mod 50 in {5} | (reads) | same |
| L2 gate c155 | 10% / 10% | **a** (R2 1.00): a in {1..2, 21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 10 in {1}, R2 0.80] | same | (reads) | same |
| L2 gate c147 | 2% / 2% | **a** (R2 1.00): a in {90, 100} | same | (reads) | same |
| L2 down c934 | 1% / 2% | **a** (R2 1.00): a in {42} | **a** (R2 0.95): a in {42, 44} | - | same |
| L2 v c69 (kv6) | 18% / 21% | **a//10** (R2 0.89): (tens) a in {81..84, 86..99} | **a** (R2 1.00): a in {78..79, 81..99} | (reads) | same |
| L2 up c324 | 2% / 2% | **a** (R2 1.00): a in {32, 64} | same | (reads) | same |
| L2 up c363 | 4% / 4% | **a** (R2 1.00): a in {31, 42, 51..52} | **a** (R2 0.99): a in {31, 42, 51..52} | (reads) | same |
| L2 up c369 | 4% / 4% | **a** (R2 1.00): a in {18..19, 78, 88} | same | (reads) | same |
| L2 up c520 | 23% / 23% | **a** (R2 1.00): a in {12, 16, 18, 24, 32, 36, 42, 44, 46, 48, 52, 54, 56, 64, 72, 74, 76, 84, 86, 88, 92, 94, 96} | **a** (R2 0.99): a in {12, 16, 18, 24, 32, 36, 42, 44, 46, 48, 52, 54, 56, 64, 72, 74, 76, 84, 86, 88, 92, 94, 96} | (reads) | same |
| L2 k c62 (kv2) | 10% / 2% | **a** (R2 1.00): a in {11, 16..18, 21, 55, 86, 90..92} | **a** (R2 0.65): a in {18, 70} | (reads) | same |
| L2 k c114 (kv2) | 3% / 0 | **a** (R2 1.00): a in {11, 18, 21} | off (on 0) | (reads) | same |
| L2 k c125 (kv2) | 2% / 0 | **a** (R2 1.00): a in {11, 18} | off (on 0) | (reads) | same |
| L2 v c207 (kv6) | 16% / 16% | **a** (R2 1.00): a in {8, 10..24} | **a** (R2 0.99): a in {8, 10..24} | (reads) | same |
| L2 v c172 (kv6) | 13% / 13% | **a** (R2 1.00): a in {3..14, 16} | **a** (R2 0.98): a in {3..14, 16} | (reads) | same |
| L2 v c173 (kv2) | 2% / 1% | **a** (R2 0.99): a in {18, 21} | **a** (R2 0.85): a in {18} | (reads) | same |
| L2 v c177 (kv2) | 3% / 4% | **a** (R2 0.91): a in {70, 74, 80} | **a** (R2 0.83): a in {70, 74, 80} | (reads) | same |
| L2 v c247 (kv2) | 2% / 1% | **a** (R2 0.99): a in {21, 70} | **a** (R2 0.82): a in {70} | (reads) | same |
| L2 up c213 | 6% / 6% | **a** (R2 1.00): a in {45, 89..93} | **a** (R2 0.99): a in {45, 89..93} | (reads) | same |
| L2 up c275 | 4% / 4% | **a** (R2 1.00): a in {3, 33, 83, 93} | same | (reads) | same |
| L2 up c292 | 7% / 7% | **a** (R2 1.00): a in {1..4, 60..62} | same | (reads) | same |
| L2 up c98 | 6% / 6% | **a** (R2 1.00): a in {15, 25, 35, 50, 65, 75} [coarser: a mod 50 in {15, 25}, R2 0.82] | same | (reads) | same |
| L2 up c102 | 10% / 10% | **a** (R2 1.00): a in {12, 18, 24, 36, 45, 48, 54, 60, 72, 90} | same | (reads) | same |
| L2 up c118 | 6% / 6% | **a** (R2 1.00): a in {24, 48..49, 72, 96, 99} | same | (reads) | same |
| L2 up c149 | 4% / 4% | **a** (R2 1.00): a in {23..25, 48} | same | (reads) | same |
| L2 up c199 | 3% / 3% | **a** (R2 1.00): a in {7, 70, 77} | **a** (R2 0.99): a in {7, 70, 77} | (reads) | same |
| L2 gate c729 | 16% / 16% | **a** (R2 1.00): a in {10, 20, 30, 40, 50, 60, 80, 90, 93..100} | same | (reads) | same |
| L2 up c58 | 13% / 13% | **a** (R2 1.00): a in {76, 78..89} | same | (reads) | same |
| L2 up c61 | 3% / 3% | **a** (R2 1.00): a in {14, 21, 42} | **a** (R2 0.97): a in {14, 21, 42} | (reads) | same |
| L2 down c213 | 3% / 3% | **a** (R2 1.00): a in {44..45, 56} | **a** (R2 0.99): a in {44..45, 56} | - | same |
| L2 down c199 | 3% / 3% | **a** (R2 1.00): a in {7, 70, 77} | same | - | same |
| L2 down c147 | 12% / 13% | **a%50** (R2 0.90): a mod 50 in {0, 10, 20, 30, 40} [coarser: a mod 10 in {0}, R2 0.83] | **a** (R2 1.00): a in {10, 20, 30, 40, 50, 60, 70, 80, 90, 97..100} [coarser: a mod 50 in {0, 10, 20, 30, 40}, R2 0.87] | a: mod10 +2%, mod2 +2% | a: mod10 +2%, mod2 +2% |
| L2 down c118 | 19% / 19% | **a** (R2 1.00): a in {16, 24, 32, 36, 44, 46, 48..49, 52, 54, 56, 64, 72, 74, 76, 84, 86, 88, 96} | same | a: mod4 +6%, mod2 +4% | a: mod4 +6%, mod2 +4% |
| L2 down c102 | 11% / 13% | **a** (R2 1.00): a in {12, 18, 24, 36, 42, 45, 48, 60, 72, 90, 96} | **a** (R2 0.99): a in {12, 18, 24, 30, 36, 42, 45, 48, 54, 60, 72, 90, 96} | - | same |
| L2 down c98 | 6% / 6% | **a** (R2 1.00): a in {15, 25, 35, 50, 65, 75} [coarser: a mod 50 in {15, 25}, R2 0.82] | same | - | same |
| L2 down c61 | 4% / 5% | **a** (R2 1.00): a in {14, 21, 28, 42} | **a** (R2 0.97): a in {14, 21..22, 28, 42} | - | same |
| L2 down c9 | 10% / 10% | **a** (R2 1.00): a in {1..2, 21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 10 in {1}, R2 0.80] | same | a: mod10 +4%, mod5 +4%, mod2 +3% | a: mod10 +4%, mod5 +4%, mod2 +3% |
| L2 down c864 | 4% / 4% | **a** (R2 1.00): a in {95, 97..99} | same | - | same |
| L2 v c251 (kv2) | 3% / 3% | **a** (R2 1.00): a in {83..84, 86} | **a** (R2 0.92): a in {83..84, 86} | (reads) | same |
| L2 down c448 | 7% / 7% | **a** (R2 1.00): a in {45, 60, 89..93} | same | - | same |
| L2 down c369 | 2% / 2% | **a** (R2 1.00): a in {18, 88} | same | - | same |
| L2 down c363 | 3% / 3% | **a** (R2 1.00): a in {31, 51..52} | same | - | same |
| L2 down c313 | 6% / 7% | **a** (R2 1.00): a in {15..18, 32, 64} | **a** (R2 0.96): a in {14..18, 32, 64} | - | same |
| L2 down c271 | 12% / 10% | **a** (R2 1.00): a in {50..59, 62, 65} | **a** (R2 0.99): a in {51..59, 62} | - | same |
| L2 down c254 | 2% / 2% | **a** (R2 1.00): a in {10, 20} | same | - | same |
| L3 v c228 (kv1) | 17% / 17% | **a** (R2 0.98): a in {15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100} [coarser: a mod 5 in {0}, R2 0.81] | **a** (R2 0.93): a in {20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100} [coarser: a mod 5 in {0}, R2 0.81] | (reads) | same |
| L3 gate c325 | 10% / 10% | **a** (R2 1.00): a in {12, 22, 32, 42, 48, 52, 62, 72, 82, 92} [coarser: a mod 10 in {2}, R2 0.80] | same | (reads) | same |
| L3 k c133 (kv1) | 2% / 0 | **a** (R2 1.00): a in {50, 55} | off (on 0) | (reads) | same |
| L3 k c89 (kv3) | 3% / 0 | **a** (R2 1.00): a in {11, 21, 55} | off (on 0) | (reads) | same |
| L3 k c80 (kv1) | 0 / 8% | off (on 0) | **a** (R2 0.90): a in {20, 41..42, 61..62, 70, 81} | (reads) | same |
| L3 up c156 | 21% / 21% | **a** (R2 1.00): a in {23, 33, 39..49, 53, 62..65, 67, 74, 83} | same | (reads) | same |
| L3 up c153 | 6% / 6% | **a** (R2 1.00): a in {12, 24, 36, 48, 72, 96} | same | (reads) | same |
| L3 v c230 (kv1) | 0 / 12% | off (on 0) | **a** (R2 1.00): a in {63, 67..74, 77..79} | (reads) | same |
| L3 down c361 | 2% / 2% | **a** (R2 1.00): a in {64, 83} | same | - | same |
| L3 down c466 | 7% / 7% | **a** (R2 1.00): a in {33, 44, 55, 66, 77, 88, 99} | same | - | same |
| L3 down c665 | 8% / 8% | **a** (R2 1.00): a in {26, 33, 46, 53, 63, 76, 86, 96} | same | - | same |
| L3 gate c79 | 10% / 10% | **a** (R2 1.00): a in {36, 38..46} | same | (reads) | same |
| L3 gate c139 | 6% / 6% | **a** (R2 1.00): a in {49, 95..99} | same | (reads) | same |
| L3 down c156 | 23% / 23% | **a** (R2 1.00): a in {23, 33, 39..49, 53, 62..65, 67, 73..74, 83, 93} | same | a: mod20 +2% | a: mod20 +2% |
| L3 down c175 | 2% / 2% | **a** (R2 1.00): a in {42, 52} | same | - | same |
| L3 down c360 | 2% / 2% | **a** (R2 1.00): a in {6, 46} | same | - | same |
| L3 down c11 | 53% / 53% | **a** (R2 1.00): a in {47..49, 51..100} | same | a: mod100 +3% | a: mod100 +3% |
| L3 down c29 | 12% / 12% | **a** (R2 0.99): a in {60..70, 74} | same | - | same |
| L3 down c44 | 23% / 23% | **a** (R2 1.00): a in {2..23, 25} | same | a: mod100 +3%, mod50 +4% | a: mod100 +3%, mod50 +4% |
| L3 down c79 | 10% / 10% | **a** (R2 1.00): a in {36, 38..46} | same | - | same |
| L3 down c153 | 6% / 6% | **a** (R2 1.00): a in {12, 24, 36, 48, 72, 96} | same | a: mod4 +5% | a: mod4 +5% |
| L3 gate c466 | 7% / 7% | **a** (R2 1.00): a in {33, 44, 55, 66, 77, 88, 99} | same | (reads) | same |
| L3 v c187 (kv1) | 1% / 2% | **a** (R2 1.00): a in {24} | **a** (R2 0.75): a in {24, 48} | (reads) | same |
| L3 v c191 (kv1) | 2% / 2% | **a** (R2 1.00): a in {86, 90} | same | (reads) | same |
| L3 v c207 (kv1) | 6% / 12% | **a** (R2 0.99): a in {95..100} | **a** (R2 0.90): a in {10..13, 15, 93..100} | (reads) | same |
| L3 v c75 (kv1) | 1% / 2% | **a** (R2 1.00): a in {55} | **a** (R2 0.99): a in {55, 77} | (reads) | same |
| L3 v c112 (kv1) | 6% / 10% | **a** (R2 1.00): a in {1..4, 11, 13} | **a** (R2 0.98): a in {1..4, 10..15} | (reads) | same |
| L3 v c131 (kv1) | 1% / 7% | **a** (R2 1.00): a in {62} | **a** (R2 0.96): a in {58..63, 71} | (reads) | same |
| L3 v c149 (kv1) | 1% / 3% | **a** (R2 0.97): a in {74} | **a** (R2 0.99): a in {70, 73..74} | (reads) | same |
| L3 v c179 (kv1) | 1% / 2% | **a** (R2 0.98): a in {51} | **a** (R2 1.00): a in {51, 53} | (reads) | same |
| L3 k c135 (kv1) | 9% / 1% | **a** (R2 1.00): a in {21, 50..52, 55..57, 62, 65} | **a** (R2 0.96): a in {55} | (reads) | same |
| L3 o c85 (H19) | 2% / 2% | **a** (R2 0.99): a in {52, 74} | **a** (R2 0.96): a in {52, 74} | - | same |
| L3 v c54 (kv1) | 18% / 18% | **a** (R2 1.00): a in {50..58, 60..68} | same | (reads) | same |
| L3 v c195 (kv1) | 13% / 17% | **a** (R2 0.98): a in {86, 88..99} | **a** (R2 0.95): a in {10..13, 86..99} | (reads) | same |
| L3 v c180 (kv1) | 3% / 12% | **a** (R2 0.99): a in {11, 51, 53} | **a** (R2 0.99): a in {11, 31, 41, 43, 46..47, 49, 51..53, 61, 81} | (reads) | same |
| L4 down c57 | 5% / 5% | **a** (R2 1.00): a in {12, 30..32, 52} | same | - | same |
| L4 gate c625 | 5% / 5% | **a** (R2 1.00): a in {33, 55, 77, 88, 99} | same | (reads) | same |
| L4 up c57 | 2% / 2% | **a** (R2 1.00): a in {31, 52} | same | (reads) | same |
| L4 o c13 (H26) | 31% / 31% | **a** (R2 1.00): a in {21, 31, 48..49, 52, 69, 71..93, 96, 99} | **a** (R2 0.99): a in {21, 31, 48..49, 52, 69, 71..93, 96, 99} | - | same |
| L4 up c622 | 13% / 13% | **a** (R2 1.00): a in {30..41, 52} | same | (reads) | same |
| L4 down c408 | 2% / 2% | **a** (R2 1.00): a in {42, 83} | same | - | same |
| L4 down c361 | 4% / 4% | **a** (R2 1.00): a in {32, 48..49, 72} | same | - | same |
| L4 down c132 | 23% / 23% | **a** (R2 1.00): a in {2..23, 25} | same | a: mod100 +3%, mod50 +3% | a: mod100 +3%, mod50 +3% |
| L4 down c625 | 5% / 5% | **a** (R2 1.00): a in {33, 55, 77, 88, 99} | same | - | same |
| L4 down c97 | 10% / 10% | **a** (R2 1.00): a in {12, 18, 24, 36, 45, 48, 54, 60, 72, 90} | **a** (R2 0.99): a in {12, 18, 24, 36, 45, 48, 54, 60, 72, 90} | - | same |
| L4 gate c57 | 2% / 2% | **a** (R2 1.00): a in {31, 52} | same | (reads) | same |
| L4 up c315 | 3% / 3% | **a** (R2 1.00): a in {48, 52, 72} | same | (reads) | same |
| L4 gate c453 | 3% / 3% | **a** (R2 1.00): a in {73..74, 76} | same | (reads) | same |
| L4 gate c97 | 10% / 10% | **a** (R2 1.00): a in {12, 18, 24, 36, 45, 48, 54, 60, 72, 90} | same | (reads) | same |
| L4 down c12 | 4% / 4% | **a** (R2 1.00): a in {76, 91..93} | same | - | same |
| L5 gate c73 | 3% / 3% | **a** (R2 1.00): a in {52, 74, 83} | same | (reads) | same |
| L5 down c332 | 4% / 4% | **a** (R2 1.00): a in {24, 36, 48, 72} | **a** (R2 0.99): a in {24, 36, 48, 72} | a: mod4 +2% | a: mod4 +2% |
| L5 v c201 (kv5) | 39% / 28% | **a** (R2 0.78): a in {12, 16, 22, 24, 26, 28, 32, 34, 36, 38, 40, 42, 44, 46, 48, 52, 54, 56, 58, 60, 62, 64, 66, 68, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98} [coarser: a mod 10 in {2, 4, 6, 8}, R2 0.82] | unexplained (best R2 0.48) | (reads) | same |
| L5 down c28 | 18% / 18% | **a** (R2 0.99): a in {10, 15, 20, 25, 30, 35, 40, 45, 50, 52, 55, 60, 70, 75, 80, 90, 99..100} [coarser: a mod 50 in {0, 5, 10, 20, 25, 30, 35, 40}, R2 0.80] | **a** (R2 1.00): a in {10, 15, 20, 25, 30, 35, 40, 45, 50, 52, 55, 60, 70, 75, 80, 90, 99..100} | a: mod5 +7% | a: mod5 +7% |
| L5 up c62 | 17% / 17% | **a** (R2 1.00): a in {10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 75, 80, 85, 90, 100} [coarser: a mod 5 in {0}, R2 0.82] | same | (reads) | same |
| L5 gate c218 | 8% / 9% | **a** (R2 0.99): a in {10, 30, 40, 50, 52, 60, 99..100} | **a** (R2 0.94): a in {10, 20, 30, 40, 50, 52, 60, 99..100} | (reads) | same |
| L5 down c198 | 2% / 2% | **a** (R2 1.00): a in {61, 91} | same | - | same |
| L5 down c139 | 2% / 2% | **a** (R2 1.00): a in {74, 76} | same | - | same |
| L5 down c83 | 2% / 2% | **a** (R2 1.00): a in {42, 52} | same | - | same |
| L5 up c332 | 4% / 4% | **a** (R2 1.00): a in {24, 36, 48, 72} | same | (reads) | same |
| L5 up c108 | 2% / 2% | **a** (R2 1.00): a in {61, 91} | same | (reads) | same |
| L5 v c46 (kv5) | 18% / 18% | **a** (R2 0.77): a in {41..54, 56..59} | **a** (R2 0.56): a in {37, 39, 41..59} | (reads) | same |
| L6 down c423 | 6% / 6% | **a** (R2 1.00): a in {92..93, 95, 97..99} | same | - | same |
| L6 down c96 | 3% / 3% | **a** (R2 1.00): a in {24, 42, 83} | same | - | same |
| L6 up c96 | 4% / 4% | **a** (R2 1.00): a in {24, 42, 52, 83} | same | (reads) | same |
| L6 up c187 | 2% / 2% | **a** (R2 1.00): a in {61, 91} | same | (reads) | same |
| L6 down c62 | 2% / 2% | **a** (R2 1.00): a in {61, 91} | same | - | same |
| L6 gate c605 | 5% / 5% | **a** (R2 1.00): a in {21, 24, 42, 52, 83} | same | (reads) | same |
| L6 gate c423 | 4% / 4% | **a** (R2 1.00): a in {50, 52, 99..100} | same | (reads) | same |
| L7 down c38 | 2% / 2% | **a** (R2 1.00): a in {88, 99} | same | - | same |
| L7 down c5 | 28% / 28% | **a** (R2 1.00): a in {3..23, 25..31} | same | a: mod100 +4%, mod50 +3% | a: mod100 +4%, mod50 +3% |
| L7 down c13 | 6% / 6% | **a** (R2 1.00): a in {53, 56..59, 62} | same | - | same |
| L7 up c402 | 28% / 28% | **a** (R2 1.00): a in {3..23, 25..31} | same | (reads) | same |
| L7 down c150 | 3% / 4% | **a** (R2 0.99): a in {19, 22..23} | **a** (R2 0.92): a in {18..19, 22..23} | - | same |
| L7 up c13 | 6% / 6% | **a** (R2 1.00): a in {53, 56..59, 62} | same | (reads) | same |
| L7 up c38 | 2% / 2% | **a** (R2 1.00): a in {88, 99} | same | (reads) | same |
| L8 up c572 | 7% / 7% | **a** (R2 0.99): a in {92..95, 97..99} | **a** (R2 1.00): a in {92..95, 97..99} | (reads) | same |
| L8 down c100 | 7% / 7% | **a** (R2 1.00): a in {92..95, 97..99} | same | - | same |
| L9 down c62 | 4% / 4% | **a** (R2 1.00): a in {91..93, 99} | same | - | same |
| L9 down c18 | 2% / 2% | **a** (R2 1.00): a in {91, 93} | same | - | same |
| L9 gate c467 | 4% / 4% | **a** (R2 1.00): a in {91..93, 99} | same | (reads) | same |
| L9 down c529 | 3% / 3% | **a** (R2 1.00): a in {50, 52, 100} [coarser: a mod 50 in {0}, R2 0.83] | same | - | same |
| L10 gate c54 | 10% / 10% | **a** (R2 1.00): a in {49, 51..59} | same | (reads) | same |
| L10 gate c517 | 70% / 71% | **a** (R2 1.00): a in {24, 32..100} | **a** (R2 0.97): a in {24, 32..100} | (reads) | same |
| L10 gate c158 | 2% / 2% | **a** (R2 1.00): a in {1, 91} | same | (reads) | same |
| L10 down c937 | 3% / 3% | **a** (R2 1.00): a in {50, 52, 100} [coarser: a mod 50 in {0}, R2 0.83] | same | - | same |
| L10 down c54 | 10% / 10% | **a** (R2 1.00): a in {49, 51..59} | same | - | same |
| L11 up c26 | 13% / 13% | **a** (R2 1.00): a in {4, 14, 23..24, 34, 42..44, 54, 64, 74, 84, 94} [coarser: a mod 20 in {4, 14}, R2 0.82] | same | (reads) | same |
| L11 down c52 | 11% / 11% | **a** (R2 1.00): a in {78..87, 89} | same | - | same |
| L11 down c26 | 13% / 13% | **a** (R2 1.00): a in {4, 14, 23..24, 34, 42..44, 54, 64, 74, 84, 94} [coarser: a mod 20 in {4, 14}, R2 0.82] | same | a: mod10 +4%, mod2 +3% | a: mod10 +4%, mod2 +3% |
| L11 gate c223 | 2% / 2% | **a** (R2 1.00): a in {74, 76} | same | (reads) | same |
| L11 gate c52 | 11% / 11% | **a** (R2 1.00): a in {78..87, 89} | same | (reads) | same |
| L11 gate c50 | 2% / 2% | **a** (R2 1.00): a in {86, 88} | **a** (R2 0.99): a in {86, 88} | (reads) | same |
| L11 down c223 | 2% / 1% | **a** (R2 0.99): a in {74, 76} | **a** (R2 1.00): a in {76} | - | same |
| L11 up c223 | 2% / 2% | **a** (R2 1.00): a in {74, 76} | same | (reads) | same |
| L12 down c69 | 2% / 2% | **a** (R2 1.00): a in {34, 36} | same | - | same |
| L12 up c109 | 2% / 2% | **a** (R2 1.00): a in {74, 76} | same | (reads) | same |
| L12 up c214 | 5% / 5% | **a** (R2 1.00): a in {24, 36, 42, 48, 72} | same | (reads) | same |
| L12 down c364 | 3% / 3% | **a** (R2 1.00): a in {70..71, 73} | same | - | same |
| L12 gate c76 | 7% / 7% | **a** (R2 1.00): a in {33, 44, 55, 66, 77, 88, 99} | same | (reads) | same |
| L12 down c214 | 5% / 5% | **a** (R2 1.00): a in {24, 36, 42, 48, 72} | same | - | same |
| L12 down c76 | 7% / 7% | **a** (R2 0.96): a in {33, 44, 55, 66, 77, 88, 99} | **a** (R2 1.00): a in {33, 44, 55, 66, 77, 88, 99} | - | same |
| L12 down c47 | 3% / 3% | **a** (R2 1.00): a in {76, 84, 86} | same | - | same |
| L13 down c187 | 2% / 2% | **a** (R2 1.00): a in {33, 66} | same | - | same |
| L13 down c185 | 3% / 3% | **a** (R2 1.00): a in {80, 90..91} | same | - | same |
| L13 down c96 | 6% / 6% | **a** (R2 1.00): a in {68..69, 86..89} | **a** (R2 0.99): a in {68..69, 86..89} | - | same |
| L13 down c59 | 10% / 10% | **a** (R2 0.98): a in {12, 18, 24, 36, 42, 45, 48, 60, 72, 90} | **a** (R2 0.99): a in {12, 18, 24, 36, 42, 45, 48, 60, 72, 90} | - | same |
| L13 down c54 | 8% / 8% | **a** (R2 1.00): a in {30..32, 60..64} | same | - | same |
| L13 down c46 | 7% / 7% | **a** (R2 1.00): a in {32..34, 52..54, 74} | same | - | same |
| L13 down c361 | 2% / 2% | **a** (R2 1.00): a in {44, 55} | same | - | same |
| L13 down c545 | 1% / 2% | **a** (R2 0.83): a in {52} | **a** (R2 0.97): a in {32, 52} | - | same |
| L13 up c46 | 5% / 6% | **a** (R2 1.00): a in {31..34, 52} | **a** (R2 0.95): a in {31..35, 52} | (reads) | same |
| L13 gate c297 | 9% / 14% | **a** (R2 0.94): a in {93..100} | **a** (R2 0.94): a in {74..77, 79, 92..100} | (reads) | same |
| L13 gate c246 | 5% / 5% | **a** (R2 1.00): a in {3, 43, 53, 63, 93} [coarser: a mod 50 in {3, 43}, R2 0.89] | same | (reads) | same |
| L13 gate c54 | 8% / 8% | **a** (R2 1.00): a in {30..32, 60..64} | same | (reads) | same |
| L13 gate c15 | 13% / 13% | **a** (R2 1.00): a in {3, 13, 22..23, 33, 43, 53, 63, 73, 76, 83, 92..93} [coarser: a mod 50 in {3, 13, 23, 33, 43}, R2 0.87] | same | (reads) | same |
| L13 down c795 | 4% / 4% | **a** (R2 1.00): a in {36, 56, 76, 96} | same | - | same |
| L13 up c59 | 10% / 10% | **a** (R2 1.00): a in {12, 18, 24, 36, 42, 45, 48, 60, 72, 90} | same | (reads) | same |
| L13 down c16 | 13% / 13% | **a** (R2 1.00): a in {75..78, 92..100} | same | a: mod20 +2% | a: mod20 +2% |
| L13 up c187 | 2% / 2% | **a** (R2 0.99): a in {33, 66} | **a** (R2 1.00): a in {33, 66} | (reads) | same |
| L14 v c125 (kv4) | 18% / 24% | **a** (R2 0.87): a in {1..14, 20, 100} | **a** (R2 0.98): a in {1..16, 20, 30, 40, 50, 60, 80, 90, 100} | (reads) | same |
| L14 v c148 (kv4) | 2% / 3% | **a** (R2 0.99): a in {85, 87} | **a** (R2 1.00): a in {24, 85, 87} | (reads) | same |
| L14 v c131 (kv7) | 29% / 47% | unexplained (best R2 0.49) | **a** (R2 0.88): a in {4, 31..49, 51..55, 61..63, 65, 74, 81..94, 96} | (reads) | same |
| L14 down c154 | 6% / 6% | **a** (R2 1.00): a in {30, 40, 60, 70, 80, 90} [coarser: a mod 50 in {30, 40}, R2 0.82] | same | - | same |
| L14 down c165 | 2% / 2% | **a** (R2 1.00): a in {37, 74} | same | - | same |
| L14 down c176 | 5% / 5% | **a** (R2 1.00): a in {28, 36, 56..58} | **a** (R2 0.94): a in {28, 36, 56..58} | - | same |
| L14 v c221 (kv4) | 7% / 10% | **a** (R2 0.98): a in {11, 41, 51, 61, 71, 81, 91} | **a** (R2 1.00): a in {11, 21, 31, 41, 51, 53, 61, 71, 81, 91} [coarser: a mod 10 in {1}, R2 0.80] | (reads) | same |
| L14 down c215 | 5% / 5% | **a** (R2 1.00): a in {25, 49..52} | same | - | same |
| L14 down c393 | 2% / 4% | **a** (R2 0.76): a in {21, 41} | **a** (R2 1.00): a in {21, 41..43} | - | same |
| L14 down c398 | 2% / 2% | **a** (R2 1.00): a in {35, 45} | same | - | same |
| L14 down c400 | 2% / 2% | **a** (R2 1.00): a in {10, 100} | same | - | same |
| L14 down c625 | 3% / 3% | **a** (R2 1.00): a in {48, 96, 98} [coarser: a mod 50 in {48}, R2 0.83] | same | - | same |
| L14 down c787 | 1% / 2% | **a** (R2 0.93): a in {5} | **a** (R2 1.00): a in {5, 95} | - | same |
| L14 gate c51 | 3% / 3% | **a** (R2 1.00): a in {64, 76, 96} | same | (reads) | same |
| L14 down c201 | 4% / 4% | **a** (R2 1.00): a in {7..8, 17..18} | same | - | same |
| L14 down c148 | 3% / 3% | **a** (R2 1.00): a in {15, 75, 99} | same | - | same |
| L14 down c142 | 7% / 7% | **a** (R2 1.00): a in {9, 19..20, 39..40, 79..80} | same | - | same |
| L14 down c137 | 5% / 5% | **a** (R2 1.00): a in {27, 84..87} | same | - | same |
| L14 down c129 | 3% / 5% | **a** (R2 0.97): a in {44, 55..56} | **a** (R2 1.00): a in {22, 35, 44, 55..56} | - | same |
| L14 gate c215 | 5% / 5% | **a** (R2 1.00): a in {25, 49..52} | same | (reads) | same |
| L14 gate c248 | 4% / 5% | **a** (R2 0.99): a in {21, 23..25} | **a** (R2 1.00): a in {21, 23..26} | (reads) | same |
| L14 gate c448 | 4% / 5% | **a** (R2 0.95): a in {73..76} | **a** (R2 1.00): a in {25, 73..76} | (reads) | same |
| L14 up c42 | 16% / 16% | **a** (R2 1.00): a in {18, 22, 38, 42..43, 57..60, 62..64, 78, 82..83, 98} | same | (reads) | same |
| L14 up c51 | 9% / 9% | **a** (R2 1.00): a in {12, 32, 48, 52..54, 72, 92..93} | same | (reads) | same |
| L14 down c121 | 7% / 7% | **a** (R2 1.00): a in {23, 26, 43, 46, 53, 73, 93} | same | - | same |
| L14 down c55 | 8% / 8% | **a** (R2 1.00): a in {21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 50 in {21, 31, 41}, R2 0.86] | same | - | same |
| L14 down c42 | 15% / 17% | **a** (R2 0.99): a in {12, 18, 22, 32, 38, 42..43, 48, 58, 62, 72, 78, 82..83, 98} | **a** (R2 1.00): a in {12, 18, 22, 32, 38, 42..43, 48, 52, 58, 62..63, 72, 78, 82..83, 98} | a: mod4 +18% | a: mod4 +20% |
| L14 down c104 | 8% / 8% | **a** (R2 1.00): a in {85..90, 92..93} | same | - | same |
| L14 down c65 | 12% / 12% | **a** (R2 1.00): a in {42..43, 45..54} | same | - | same |
| L14 down c96 | 3% / 4% | **a** (R2 0.93): a in {16..17, 64} | **a** (R2 1.00): a in {16..17, 32, 64} | - | same |
| L14 gate c65 | 12% / 12% | **a** (R2 1.00): a in {42..43, 45..54} | same | (reads) | same |
| L14 gate c104 | 10% / 10% | **a** (R2 1.00): a in {80, 82..90} | same | (reads) | same |
| L14 gate c140 | 10% / 10% | **a** (R2 0.98): a in {2, 22..23, 32, 42..43, 62, 72..73, 82} | **a** (R2 1.00): a in {2, 22..23, 32, 42..43, 62, 72..73, 82} | (reads) | same |
| L14 gate c154 | 8% / 8% | **a** (R2 1.00): a in {20, 30, 40, 50, 60, 70, 80, 90} [coarser: a mod 50 in {20, 30, 40}, R2 0.86] | same | (reads) | same |
| L14 gate c201 | 9% / 10% | **a** (R2 1.00): a in {7..8, 18, 28, 38, 48, 58, 78, 98} [coarser: a mod 50 in {8, 28, 48}, R2 0.82] | **a** (R2 1.00): a in {7..8, 17..18, 28, 38, 48, 58, 78, 98} | (reads) | same |
| L14 up c393 | 2% / 2% | **a** (R2 1.00): a in {21, 42} | same | (reads) | same |
| L14 up c561 | 2% / 2% | **a** (R2 1.00): a in {60, 90} | same | (reads) | same |
| L14 v c46 (kv4) | 5% / 13% | **a** (R2 0.96): a in {78, 82, 84, 86, 88} | **a** (R2 0.99): a in {78, 80..90, 92} | (reads) | same |
| L14 v c81 (kv4) | 4% / 4% | **a** (R2 0.99): a in {7, 17, 27, 97} | **a** (R2 1.00): a in {7, 17, 27, 97} | (reads) | same |
| L14 up c165 | 4% / 4% | **a** (R2 1.00): a in {37, 47, 67, 87} | same | (reads) | same |
| L14 up c104 | 6% / 6% | **a** (R2 1.00): a in {28, 38, 48, 58, 68, 88} | same | (reads) | same |
| L14 up c290 | 10% / 9% | **a** (R2 1.00): a in {32, 34..42} | **a** (R2 0.98): a in {32, 35..42} | (reads) | same |
| L15 down c426 | 3% / 6% | **a** (R2 0.98): a in {42, 52, 72} | **a** (R2 0.95): a in {32, 42, 52, 72, 82, 92} | - | same |
| L15 down c73 | 9% / 9% | **a** (R2 1.00): a in {51..58, 95} | same | - | same |
| L15 down c42 | 15% / 15% | **a** (R2 1.00): a in {12, 16, 24, 32, 36, 40, 44, 48, 56, 64, 72, 76, 80, 84, 96} | same | a: mod4 +20%, mod2 +3% | a: mod4 +20%, mod2 +3% |
| L15 down c14 | 5% / 5% | **a** (R2 1.00): a in {9, 16..18, 27} | same | - | same |
| L15 down c5 | 46% / 46% | **a** (R2 0.99): a in {38, 43, 46..49, 53..54, 56..59, 61..69, 71..74, 76..79, 81..89, 91..98} | **a** (R2 1.00): a in {38, 43, 46..49, 53..54, 56..59, 61..69, 71..74, 76..79, 81..89, 91..98} | - | same |
| L15 gate c104 | 9% / 9% | **a** (R2 1.00): a in {40, 57..63, 80} | same | (reads) | same |
| L15 gate c110 | 9% / 9% | **a** (R2 0.99): a in {42, 45..52} | **a** (R2 1.00): a in {42, 45..52} | (reads) | same |
| L15 gate c181 | 4% / 4% | **a** (R2 1.00): a in {80, 97..99} | same | (reads) | same |
| L15 gate c184 | 7% / 7% | **a** (R2 1.00): a in {30, 45, 50, 60, 70, 80, 90} | same | (reads) | same |
| L15 k c81 (kv7) | 10% / 0 | **a** (R2 0.60): a in {86..88, 92..93, 96..98} | off (on 0) | (reads) | same |
| L15 up c894 | 6% / 6% | **a** (R2 1.00): a in {14, 34, 54, 64, 74, 94} [coarser: a mod 20 in {14}, R2 0.86] | same | (reads) | same |
| L15 up c840 | 4% / 4% | **a** (R2 1.00): a in {13, 33, 53, 93} | same | (reads) | same |
| L15 up c797 | 4% / 4% | **a** (R2 1.00): a in {18, 58, 78, 98} | same | (reads) | same |
| L15 gate c840 | 10% / 10% | **a** (R2 1.00): a in {22..23, 33, 43, 53, 63, 73, 83, 92..93} | same | (reads) | same |
| L15 gate c797 | 2% / 2% | **a** (R2 1.00): a in {28, 88} | same | (reads) | same |
| L15 gate c470 | 3% / 3% | **a** (R2 1.00): a in {22, 44, 88} | same | (reads) | same |
| L15 gate c267 | 4% / 4% | **a** (R2 1.00): a in {14, 24, 74, 84} | same | (reads) | same |
| L15 up c495 | 6% / 6% | **a** (R2 1.00): a in {20, 40, 60, 70, 80, 90} [coarser: a mod 50 in {20, 40}, R2 0.82] | same | (reads) | same |
| L15 up c426 | 2% / 2% | **a** (R2 1.00): a in {42, 52} | same | (reads) | same |
| L15 up c73 | 2% / 2% | **a** (R2 1.00): a in {45, 55} | same | (reads) | same |
| L15 up c42 | 15% / 15% | **a** (R2 1.00): a in {12, 16, 24, 32, 36, 40, 44, 48, 56, 64, 72, 76, 80, 84, 96} | same | (reads) | same |
| L15 down c96 | 2% / 2% | **a** (R2 1.00): a in {11, 31} | same | - | same |
| L15 down c104 | 6% / 7% | **a** (R2 1.00): a in {40, 60..61, 70, 80, 90} | **a** (R2 1.00): a in {30, 40, 60..61, 70, 80, 90} | - | same |
| L15 down c184 | 3% / 3% | **a** (R2 1.00): a in {45..46, 90} | same | - | same |
| L15 down c188 | 4% / 4% | **a** (R2 1.00): a in {14, 28..29, 74} | same | - | same |
| L15 gate c73 | 10% / 10% | **a** (R2 1.00): a in {51..59, 95} | same | (reads) | same |
| L15 gate c42 | 7% / 7% | **a** (R2 1.00): a in {16, 32, 36, 56, 64, 76, 96} | same | (reads) | same |
| L15 down c775 | 2% / 2% | **a** (R2 1.00): a in {13, 26} | same | - | same |
| L16 down c22 | 14% / 14% | **a** (R2 1.00): a in {17, 19..31} | **a** (R2 0.99): a in {17, 19..31} | - | same |
| L16 down c30 | 7% / 8% | **a** (R2 0.99): a in {24, 32, 36, 48, 64, 72, 96} | **a** (R2 0.99): a in {16, 24, 32, 36, 48, 64, 72, 96} | a: mod4 +9%, mod2 +3% | a: mod4 +10%, mod2 +3% |
| L16 down c50 | 13% / 15% | **a** (R2 0.99): a in {16, 21, 26, 31, 41, 46, 51, 61, 71, 76, 81, 86, 91} | **a** (R2 0.97): a in {11, 16, 21, 26, 31, 41, 46, 51, 61, 71, 76, 81, 86, 91, 96} [coarser: a mod 50 in {11, 16, 21, 26, 31, 41, 46}, R2 0.87] | a: mod5 +5% | a: mod5 +5% |
| L16 up c259 | 2% / 2% | **a** (R2 1.00): a in {62, 99} | same | (reads) | same |
| L16 up c250 | 3% / 3% | **a** (R2 1.00): a in {61, 64..65} | same | (reads) | same |
| L16 up c151 | 5% / 5% | **a** (R2 1.00): a in {22, 42..44, 82} | same | (reads) | same |
| L16 v c238 (kv0) | 3% / 1% | **a** (R2 0.91): a in {45, 65, 85} | **a** (R2 0.57): a in {65, 85} | (reads) | same |
| L16 v c103 (kv5) | 27% / 21% | **a** (R2 0.94): a in {7..14, 30..34, 50..54, 70..74, 90..94} [coarser: a mod 20 in {10..14}, R2 0.89] | **a%20** (R2 0.78): a mod 20 in {10..14} | (reads) | same |
| L16 v c100 (kv0) | 20% / 19% | **a** (R2 0.93): a in {10, 20, 40, 58..73, 80, 90} | **a** (R2 0.90): a in {40, 58..73, 80, 90} | (reads) | same |
| L16 v c90 (kv0) | 2% / 1% | **a** (R2 0.98): a in {4, 64} | **a** (R2 0.89): a in {64} | (reads) | same |
| L16 v c70 (kv5) | 39% / 39% | **a** (R2 1.00): a in {15..44, 76..84} | same | (reads) | same |
| L16 v c68 (kv5) | 19% / 15% | **a** (R2 0.95): a in {1..4, 55..65, 80..84} | **a** (R2 0.85): a in {1, 55..65, 80..84} | (reads) | same |
| L16 v c66 (kv5) | 8% / 8% | **a** (R2 0.94): a in {92, 94..100} | **a** (R2 0.94): a in {94..100} | (reads) | same |
| L16 v c30 (kv0) | 24% / 22% | **a** (R2 0.97): a in {57..79, 81} | **a//10** (R2 0.87): (tens) a in {58..79} | (reads) | same |
| L16 v c29 (kv0) | 25% / 23% | **a** (R2 1.00): a in {1..4, 32..49, 82..84} | **a** (R2 0.95): a in {4, 32..49, 82..84} | (reads) | same |
| L16 v c28 (kv5) | 4% / 3% | **a** (R2 0.73): a in {10, 50, 98, 100} | **a** (R2 0.54): a in {50, 100} [coarser: a mod 50 in {0}, R2 0.84] | (reads) | same |
| L16 v c26 (kv0) | 4% / 2% | **a** (R2 0.72): a in {4, 14, 84, 94} | unexplained (best R2 0.30) | (reads) | same |
| L16 v c25 (kv5) | 40% / 37% | **a** (R2 0.99): a in {4, 24..52, 65..67, 85..91} | **a** (R2 0.94): a in {24..52, 85..91} | (reads) | same |
| L16 o c17 (H30) | 24% / 25% | **a** (R2 0.98): a in {14, 28, 30..31, 35, 40, 42, 49..52, 54, 56..58, 60, 70, 74..75, 90, 92..93, 99..100} | **a** (R2 0.97): a in {14, 28, 30..31, 35, 40, 42, 49..52, 54, 56..58, 60, 70, 74..75, 90, 92..93, 99..100} | - | same |
| L16 up c557 | 2% / 2% | **a** (R2 1.00): a in {16, 96} | same | (reads) | same |
| L16 up c513 | 4% / 4% | **a** (R2 1.00): a in {32, 52, 72, 92} | same | (reads) | same |
| L16 up c137 | 7% / 7% | **a** (R2 1.00): a in {34..37, 74..76} | same | (reads) | same |
| L16 gate c807 | 5% / 5% | **a** (R2 1.00): a in {26..27, 46, 76, 86} | same | (reads) | same |
| L16 gate c499 | 3% / 3% | **a** (R2 1.00): a in {44, 88, 99} | same | (reads) | same |
| L16 gate c443 | 4% / 4% | **a** (R2 1.00): a in {30..32, 60} | same | (reads) | same |
| L16 gate c388 | 12% / 12% | **a** (R2 1.00): a in {76..85, 88..89} | same | (reads) | same |
| L16 gate c336 | 9% / 9% | **a** (R2 1.00): a in {1..2, 22, 32, 42, 52, 62, 82, 92} [coarser: a mod 50 in {2, 32, 42}, R2 0.82] | same | (reads) | same |
| L16 gate c278 | 2% / 2% | **a** (R2 1.00): a in {1, 7} | same | (reads) | same |
| L16 gate c252 | 7% / 8% | **a** (R2 1.00): a in {23..27, 48, 72} | **a** (R2 1.00): a in {23..28, 48, 72} | (reads) | same |
| L16 gate c209 | 5% / 5% | **a** (R2 1.00): a in {18, 35..36, 45, 90} | same | (reads) | same |
| L16 gate c120 | 16% / 16% | **a** (R2 1.00): a in {5, 10, 15, 20, 25, 30, 35, 40, 49..50, 55, 60, 70, 75, 80, 100} [coarser: a mod 50 in {0, 5, 10, 20, 25, 30}, R2 0.85] | **a** (R2 1.00): a in {5, 10, 15, 20, 25, 30, 35, 40, 49..50, 55, 60, 70, 75, 80, 100} [coarser: a mod 50 in {0, 5, 10, 20, 25, 30, 40}, R2 0.85] | (reads) | same |
| L16 gate c57 | 4% / 4% | **a** (R2 1.00): a in {40, 60, 80, 90} | same | (reads) | same |
| L16 gate c30 | 10% / 10% | **a** (R2 1.00): a in {16, 24, 32, 36, 48, 64, 72, 76, 84, 96} | same | (reads) | same |
| L16 down c929 | 3% / 3% | **a** (R2 1.00): a in {42, 52, 92} [coarser: a mod 50 in {42}, R2 0.83] | same | - | same |
| L16 down c814 | 3% / 3% | **a** (R2 1.00): a in {26, 46, 76} [coarser: a mod 50 in {26}, R2 0.83] | same | - | same |
| L16 down c762 | 2% / 2% | **a** (R2 1.00): a in {64, 84} | same | - | same |
| L16 down c557 | 2% / 2% | **a** (R2 1.00): a in {16, 96} | same | - | same |
| L16 down c524 | 7% / 7% | **a** (R2 1.00): a in {23, 43, 53, 63, 73, 83, 93} | same | - | same |
| L16 down c513 | 3% / 3% | **a** (R2 0.92): a in {32, 82, 92} [coarser: a mod 50 in {32}, R2 0.80] | **a** (R2 0.98): a in {32, 82, 92} [coarser: a mod 50 in {32}, R2 0.83] | - | same |
| L16 down c354 | 4% / 4% | **a** (R2 1.00): a in {32, 63..65} | same | - | same |
| L16 down c327 | 4% / 4% | **a** (R2 1.00): a in {60, 76, 80, 90} | same | - | same |
| L16 down c272 | 3% / 3% | **a** (R2 1.00): a in {3, 33, 93} | same | - | same |
| L16 down c271 | 3% / 3% | **a** (R2 1.00): a in {13, 26..27} | same | - | same |
| L16 down c248 | 4% / 4% | **a** (R2 1.00): a in {86..88, 90} | same | - | same |
| L16 down c211 | 2% / 2% | **a** (R2 1.00): a in {78, 82} | same | - | same |
| L16 down c209 | 4% / 4% | **a** (R2 1.00): a in {18, 36, 45, 90} | same | - | same |
| L16 down c151 | 5% / 5% | **a** (R2 1.00): a in {22, 42..43, 52..53} | same | - | same |
| L16 down c137 | 9% / 9% | **a** (R2 1.00): a in {34..38, 74..76, 95} | same | - | same |
| L16 down c114 | 2% / 2% | **a** (R2 1.00): a in {5, 10} | same | - | same |
| L17 gate c229 | 4% / 4% | **a** (R2 1.00): a in {20, 40, 60, 80} | same | (reads) | same |
| L17 gate c265 | 3% / 3% | **a** (R2 1.00): a in {32..33, 64} | same | (reads) | same |
| L17 gate c279 | 6% / 6% | **a** (R2 1.00): a in {28..31, 60, 90} | same | (reads) | same |
| L17 gate c336 | 4% / 4% | **a** (R2 1.00): a in {14, 54, 74, 94} | same | (reads) | same |
| L17 gate c429 | 6% / 6% | **a** (R2 1.00): a in {18, 38..39, 68, 88, 98} [coarser: a mod 50 in {18, 38}, R2 0.82] | same | (reads) | same |
| L17 gate c560 | 6% / 6% | **a** (R2 1.00): a in {37..40, 42..43} | same | (reads) | same |
| L17 gate c474 | 7% / 7% | **a** (R2 1.00): a in {32..35, 55, 65, 95} | same | (reads) | same |
| L17 gate c475 | 4% / 4% | **a** (R2 1.00): a in {21, 41, 51, 91} | same | (reads) | same |
| L17 gate c212 | 5% / 5% | **a** (R2 1.00): a in {60..63, 82} | same | (reads) | same |
| L17 gate c165 | 5% / 5% | **a** (R2 1.00): a in {13, 33, 53, 63, 93} | same | (reads) | same |
| L17 gate c186 | 3% / 3% | **a** (R2 1.00): a in {14..15, 75} | same | (reads) | same |
| L17 gate c720 | 2% / 2% | **a** (R2 1.00): a in {14, 54} | same | (reads) | same |
| L17 gate c823 | 2% / 2% | **a** (R2 1.00): a in {45, 55} | same | (reads) | same |
| L17 gate c849 | 25% / 25% | **a** (R2 1.00): a in {74, 76..99} | same | (reads) | same |
| L17 down c186 | 3% / 3% | **a** (R2 1.00): a in {14..15, 75} | same | - | same |
| L17 down c429 | 3% / 3% | **a** (R2 1.00): a in {19, 38..39} | same | - | same |
| L17 down c212 | 4% / 4% | **a** (R2 1.00): a in {61..63, 82} | same | - | same |
| L17 down c265 | 8% / 8% | **a** (R2 1.00): a in {31..37, 64} | same | - | same |
| L17 down c299 | 3% / 3% | **a** (R2 0.99): a in {46, 65..66} | **a** (R2 1.00): a in {46, 65..66} | - | same |
| L17 down c303 | 5% / 5% | **a** (R2 1.00): a in {24, 36, 48, 72, 96} | same | - | same |
| L17 down c356 | 3% / 3% | **a** (R2 1.00): a in {49, 51..52} | same | - | same |
| L17 down c357 | 6% / 6% | **a** (R2 1.00): a in {18, 38, 68, 78, 88, 98} [coarser: a mod 50 in {18, 38}, R2 0.82] | same | - | same |
| L17 down c418 | 5% / 5% | **a** (R2 1.00): a in {28..31, 90} | same | - | same |
| L17 up c106 | 7% / 7% | **a** (R2 1.00): a in {22, 40..45} | same | (reads) | same |
| L17 up c787 | 3% / 3% | **a** (R2 1.00): a in {18, 86, 88} | same | (reads) | same |
| L17 down c165 | 3% / 3% | **a** (R2 1.00): a in {13, 33, 63} [coarser: a mod 50 in {13}, R2 0.83] | same | - | same |
| L17 down c458 | 4% / 4% | **a** (R2 1.00): a in {32, 52, 72, 92} | same | - | same |
| L17 down c465 | 2% / 2% | **a** (R2 1.00): a in {44, 55} | same | - | same |
| L17 down c474 | 4% / 4% | **a** (R2 1.00): a in {34..35, 55, 65} | same | - | same |
| L17 down c172 | 7% / 7% | **a** (R2 1.00): a in {34..39, 49} | same | - | same |
| L17 down c144 | 5% / 5% | **a** (R2 1.00): a in {20, 25, 49..51} | same | - | same |
| L17 down c161 | 3% / 3% | **a** (R2 1.00): a in {5, 10, 15} | same | - | same |
| L17 down c54 | 10% / 10% | **a** (R2 1.00): a in {6, 26, 36, 46, 56, 60, 66, 76, 86, 96} [coarser: a mod 10 in {6}, R2 0.80] | same | a: mod2 +2% | a: mod2 +2% |
| L17 down c66 | 6% / 6% | **a** (R2 1.00): a in {12, 24, 36, 42, 48, 72} | same | a: mod4 +2% | a: mod4 +2% |
| L17 down c497 | 2% / 2% | **a** (R2 1.00): a in {16, 46} | same | - | same |
| L17 down c542 | 2% / 2% | **a** (R2 1.00): a in {60, 80} | same | - | same |
| L17 down c778 | 2% / 2% | **a** (R2 1.00): a in {76, 86} | same | - | same |
| L17 down c823 | 2% / 2% | **a** (R2 1.00): a in {45, 55} | same | - | same |
| L17 down c837 | 2% / 2% | **a** (R2 1.00): a in {51, 91} | same | - | same |
| L17 gate c8 | 3% / 3% | **a** (R2 1.00): a in {25, 50, 100} [coarser: a mod 50 in {0}, R2 0.83] | same | (reads) | same |
| L17 gate c54 | 14% / 14% | **a** (R2 1.00): a in {1..4, 6, 26, 36, 46, 56, 60, 66, 76, 86, 96} | same | (reads) | same |
| L17 gate c56 | 13% / 13% | **a** (R2 1.00): a in {10..11, 44, 53..60, 64, 67} | same | (reads) | same |
| L17 gate c66 | 9% / 9% | **a** (R2 1.00): a in {12, 24, 32, 36, 42, 48, 52, 72, 92} | same | (reads) | same |
| L17 gate c110 | 6% / 7% | **a** (R2 0.97): a in {10, 50, 97..100} | **a** (R2 1.00): a in {10, 25, 50, 97..100} | (reads) | same |
| L17 down c72 | 12% / 12% | **a** (R2 1.00): a in {54..60, 62..64, 66..67} | same | - | same |
| L17 down c110 | 3% / 3% | **a** (R2 1.00): a in {50, 99..100} [coarser: a mod 50 in {0}, R2 0.83] | same | - | same |
| L17 up c303 | 4% / 4% | **a** (R2 1.00): a in {24, 36, 48, 96} | same | (reads) | same |
| L17 up c299 | 6% / 6% | **a** (R2 1.00): a in {45..47, 65..67} | same | (reads) | same |
| L17 up c278 | 5% / 5% | **a** (R2 1.00): a in {26..29, 31} | same | (reads) | same |
| L17 up c212 | 3% / 3% | **a** (R2 1.00): a in {61..62, 82} | same | (reads) | same |
| L17 up c166 | 5% / 5% | **a** (R2 1.00): a in {1..2, 21, 51, 91} | same | (reads) | same |
| L17 up c161 | 6% / 6% | **a** (R2 1.00): a in {5, 45, 55, 65, 75, 85} | same | (reads) | same |
| L17 up c357 | 9% / 9% | **a** (R2 1.00): a in {18, 32, 38, 52, 58, 68, 78, 88, 98} | same | (reads) | same |
| L18 v c35 (kv1) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {47, 49..51, 53} | (reads) | same |
| L18 v c78 (kv7) | 1% / 9% | unexplained (best R2 0.13) | **a** (R2 0.98): a in {16, 24, 32, 36, 48, 52, 56, 64, 72} | (reads) | same |
| L18 v c68 (kv7) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {41, 61} | (reads) | same |
| L18 v c98 (kv7) | 9% / 22% | **a%10** (R2 0.88): a mod 10 in {6} | **a** (R2 0.99): a in {6, 16..18, 24, 26..28, 36..37, 45..48, 56..57, 66..67, 76, 86..87, 96} | (reads) | same |
| L18 v c49 (kv7) | 2% / 11% | **a** (R2 0.54): a in {62, 82} | **a** (R2 0.99): a in {12, 22, 32, 42, 52, 61..62, 72, 82..83, 92} [coarser: a mod 50 in {12, 22, 32, 42}, R2 0.85] | (reads) | same |
| L18 v c50 (kv7) | 0 / 2% | off (on 0) | **a** (R2 0.99): a in {52, 72} | (reads) | same |
| L18 v c90 (kv7) | 12% / 16% | **a//10** (R2 0.87): (tens) a in {39..49} | **a** (R2 1.00): a in {34, 36..50} | (reads) | same |
| L18 v c60 (kv7) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {24, 48, 64} | (reads) | same |
| L18 v c89 (kv7) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {20, 70, 80} [coarser: a mod 50 in {20}, R2 0.83] | (reads) | same |
| L18 v c146 (kv1) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {65, 70..71, 73..75} | (reads) | same |
| L18 v c159 (kv7) | 23% / 24% | **a** (R2 0.99): a in {41..63} | **a** (R2 1.00): a in {41..63, 66} | (reads) | same |
| L18 gate c11 | 14% / 14% | **a** (R2 1.00): a in {80, 82..83, 85..95} | same | (reads) | same |
| L18 gate c417 | 3% / 3% | **a** (R2 1.00): a in {30..31, 33} | same | (reads) | same |
| L18 gate c345 | 2% / 2% | **a** (R2 1.00): a in {35, 65} | same | (reads) | same |
| L18 down c188 | 7% / 7% | **a** (R2 1.00): a in {24, 34, 49, 64, 74, 84, 94} | same | - | same |
| L18 down c31 | 8% / 8% | **a** (R2 1.00): a in {21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 50 in {21, 31, 41}, R2 0.86] | same | - | same |
| L18 down c13 | 56% / 56% | **a** (R2 1.00): a in {2..52, 55, 60, 75, 90, 100} | same | a: mod100 +3% | a: mod100 +3% |
| L18 down c28 | 2% / 2% | **a** (R2 1.00): a in {5, 10} | same | - | same |
| L18 down c365 | 18% / 18% | **a** (R2 1.00): a in {80, 82..83, 85..99} | same | - | same |
| L18 down c419 | 6% / 6% | **a** (R2 1.00): a in {49, 51..55} | same | - | same |
| L18 down c274 | 3% / 3% | **a** (R2 1.00): a in {52, 71..72} | same | - | same |
| L18 down c345 | 9% / 9% | **a** (R2 1.00): a in {35, 40, 45, 55, 60, 65, 70, 75, 95} | same | - | same |
| L18 down c293 | 4% / 4% | **a** (R2 1.00): a in {32, 40, 48, 64} | same | - | same |
| L18 gate c152 | 5% / 5% | **a** (R2 0.99): a in {35, 38..40, 42} | **a** (R2 1.00): a in {35, 38..40, 42} | (reads) | same |
| L18 down c417 | 3% / 3% | **a** (R2 1.00): a in {30..31, 60} | same | - | same |
| L18 gate c811 | 2% / 2% | **a** (R2 1.00): a in {60, 90} | same | (reads) | same |
| L18 gate c839 | 6% / 6% | **a** (R2 1.00): a in {24, 34, 64, 74, 84, 94} [coarser: a mod 50 in {24, 34}, R2 0.82] | same | (reads) | same |
| L18 gate c243 | 11% / 11% | **a** (R2 1.00): a in {10, 20, 25, 30, 40, 50, 52, 60, 80, 90, 100} [coarser: a mod 50 in {0, 10, 30, 40}, R2 0.85] | same | (reads) | same |
| L18 gate c274 | 2% / 2% | **a** (R2 1.00): a in {52, 72} | same | (reads) | same |
| L18 gate c188 | 5% / 5% | **a** (R2 0.99): a in {9, 29, 39, 49, 99} | **a** (R2 1.00): a in {9, 29, 39, 49, 99} | (reads) | same |
| L18 gate c202 | 8% / 8% | **a** (R2 1.00): a in {21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 50 in {21, 31, 41}, R2 0.86] | same | (reads) | same |
| L18 gate c293 | 3% / 3% | **a** (R2 1.00): a in {32, 40, 64} | same | (reads) | same |
| L18 v c251 (kv7) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {25, 74..77} | (reads) | same |
| L18 v c242 (kv7) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {18, 78} | (reads) | same |
| L18 v c230 (kv7) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {19..20, 78..81} | (reads) | same |
| L18 gate c354 | 2% / 2% | **a** (R2 1.00): a in {48, 72} | same | (reads) | same |
| L18 v c97 (kv1) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {51, 53} | (reads) | same |
| L18 v c91 (kv7) | 0 / 11% | off (on 0) | **a** (R2 0.99): a in {68..77, 79} | (reads) | same |
| L18 v c157 (kv7) | 11% / 15% | **a** (R2 0.80): a in {23..34} | **a** (R2 0.98): a in {22..35, 37} | (reads) | same |
| L18 v c211 (kv1) | 0 / 8% | off (on 0) | **a** (R2 1.00): a in {49, 51..57} | (reads) | same |
| L18 v c188 (kv7) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {24, 47, 49, 51, 53} | (reads) | same |
| L18 v c107 (kv7) | 1% / 15% | unexplained (best R2 0.39) | **a** (R2 1.00): a in {12, 20..25, 32, 42..44, 52, 62, 72, 82} | (reads) | same |
| L18 v c124 (kv7) | 0 / 17% | off (on 0) | **a** (R2 1.00): a in {9, 19, 21, 29, 31, 39, 41, 47, 49, 51, 59, 61, 69, 79, 81, 89, 99} [coarser: a mod 20 in {1, 9, 19}, R2 0.80] | (reads) | same |
| L18 v c234 (kv7) | 2% / 9% | **a** (R2 0.65): a in {50} | **a** (R2 1.00): a in {25, 46..53} | (reads) | same |
| L18 v c21 (kv7) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {15, 35, 55, 65, 75} | (reads) | same |
| L18 up c275 | 6% / 6% | **a** (R2 1.00): a in {5, 35, 45, 55, 65, 95} [coarser: a mod 50 in {5, 45}, R2 0.82] | same | (reads) | same |
| L18 up c340 | 10% / 10% | **a** (R2 1.00): a in {90, 92..100} | same | (reads) | same |
| L18 up c839 | 10% / 10% | **a** (R2 1.00): a in {21..29, 31} | same | (reads) | same |
| L18 v c128 (kv7) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {15..19, 52} | (reads) | same |
| L18 v c28 (kv7) | 10% / 13% | **a** (R2 0.88): a in {79..86, 89} | **a** (R2 1.00): a in {76, 78..89} | (reads) | same |
| L19 down c572 | 2% / 2% | **a** (R2 1.00): a in {75, 95} | same | - | same |
| L19 gate c820 | 8% / 8% | **a** (R2 1.00): a in {55, 57..63} | same | (reads) | same |
| L19 down c13 | 67% / 67% | **a** (R2 1.00): a in {32..49, 51..99} | same | a: mod100 +3% | a: mod100 +3% |
| L19 down c166 | 2% / 2% | **a** (R2 1.00): a in {14, 21} | same | - | same |
| L19 down c654 | 2% / 2% | **a** (R2 1.00): a in {48, 72} | same | - | same |
| L19 down c627 | 4% / 4% | **a** (R2 1.00): a in {28, 38, 58, 78} | same | - | same |
| L19 down c584 | 2% / 2% | **a** (R2 1.00): a in {13, 17} | same | - | same |
| L19 gate c51 | 13% / 13% | **a** (R2 1.00): a in {9, 88..99} | same | (reads) | same |
| L19 gate c52 | 10% / 10% | **a** (R2 1.00): a in {1..2, 21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 10 in {1}, R2 0.80] | same | (reads) | same |
| L19 down c52 | 10% / 10% | **a** (R2 1.00): a in {1..2, 21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 10 in {1}, R2 0.80] | same | a: mod10 +3% | a: mod10 +3% |
| L19 down c103 | 7% / 7% | **a** (R2 1.00): a in {45, 47..52} | same | - | same |
| L19 down c23 | 4% / 4% | **a** (R2 1.00): a in {90, 92..94} | same | - | same |
| L19 down c31 | 12% / 12% | **a** (R2 1.00): a in {1, 5..15} | same | - | same |
| L19 down c44 | 5% / 5% | **a** (R2 1.00): a in {31..34, 42} | same | - | same |
| L19 down c51 | 2% / 2% | **a** (R2 1.00): a in {9, 90} | same | - | same |
| L19 down c69 | 4% / 4% | **a** (R2 1.00): a in {50, 52, 60, 100} | same | - | same |
| L19 down c206 | 6% / 6% | **a** (R2 1.00): a in {8, 16, 32, 40, 64, 80} | same | a: mod4 +4% | a: mod4 +4% |
| L19 up c206 | 6% / 6% | **a** (R2 1.00): a in {8, 16, 32, 40, 64, 80} | same | (reads) | same |
| L19 down c677 | 2% / 2% | **a** (R2 1.00): a in {24, 48} | **a** (R2 0.99): a in {24, 48} | - | same |
| L19 down c72 | 3% / 3% | **a** (R2 1.00): a in {47, 49, 55} | same | - | same |
| L19 gate c166 | 3% / 3% | **a** (R2 1.00): a in {7, 14, 21} | same | (reads) | same |
| L19 gate c677 | 3% / 3% | **a** (R2 1.00): a in {24, 48, 72} | same | (reads) | same |
| L19 up c72 | 8% / 8% | **a** (R2 1.00): a in {45, 47, 49..52, 55, 99} | same | (reads) | same |
| L19 up c103 | 4% / 4% | **a** (R2 1.00): a in {48..49, 52, 72} | same | (reads) | same |
| L20 gate c131 | 5% / 5% | **a** (R2 1.00): a in {4..6, 16, 76} | same | (reads) | same |
| L20 up c607 | 9% / 9% | **a** (R2 1.00): a in {90, 92..94, 96..100} | same | (reads) | same |
| L20 o c0 (H17) | 89% / 90% | **a** (R2 1.00): a in {1..4, 6, 9, 11..14, 16..19, 21..29, 31..39, 41..49, 51..79, 81..99} | **a** (R2 0.96): a in {1..4, 6, 9, 11..14, 16..29, 31..39, 41..49, 51..79, 81..99} | - | same |
| L20 up c520 | 3% / 3% | **a** (R2 1.00): a in {18, 95, 99} | same | (reads) | same |
| L20 up c194 | 9% / 9% | **a** (R2 1.00): a in {12, 18, 24, 36, 42, 48, 60, 72, 96} | same | (reads) | same |
| L20 up c81 | 14% / 14% | **a** (R2 1.00): a in {17..29, 31} | same | (reads) | same |
| L20 up c72 | 2% / 2% | **a** (R2 1.00): a in {80, 90} | same | (reads) | same |
| L20 up c149 | 2% / 2% | **a** (R2 1.00): a in {88, 98} | same | (reads) | same |
| L20 up c131 | 4% / 4% | **a** (R2 1.00): a in {38..40, 42} | same | (reads) | same |
| L20 gate c331 | 6% / 6% | **a** (R2 1.00): a in {40, 50, 60, 70, 80, 90} | same | (reads) | same |
| L20 up c66 | 3% / 3% | **a** (R2 1.00): a in {24, 48, 72} | **a** (R2 0.99): a in {24, 48, 72} | (reads) | same |
| L20 gate c245 | 55% / 55% | **a** (R2 1.00): a in {43..44, 46..47, 49, 51..100} | same | (reads) | same |
| L20 gate c320 | 2% / 2% | **a** (R2 1.00): a in {6, 76} | same | (reads) | same |
| L20 gate c167 | 3% / 3% | **a** (R2 1.00): a in {40, 80, 88} | same | (reads) | same |
| L20 gate c71 | 9% / 9% | **a** (R2 1.00): a in {60, 62..69} | same | (reads) | same |
| L20 gate c33 | 2% / 2% | **a** (R2 1.00): a in {9, 99} | same | (reads) | same |
| L20 down c643 | 2% / 2% | **a** (R2 1.00): a in {86, 88} | same | - | same |
| L20 down c575 | 3% / 3% | **a** (R2 1.00): a in {16, 76, 96} | same | - | same |
| L20 down c331 | 8% / 7% | **a** (R2 0.97): a in {30, 40, 50, 60, 70, 80, 88, 90} | **a** (R2 0.99): a in {40, 50, 60, 70, 80, 88, 90} | - | same |
| L20 down c320 | 2% / 2% | **a** (R2 1.00): a in {6, 76} | same | - | same |
| L20 down c78 | 6% / 6% | **a** (R2 1.00): a in {30, 89..93} | **a** (R2 0.99): a in {30, 89..93} | - | same |
| L20 down c71 | 2% / 2% | **a** (R2 1.00): a in {65, 67} | same | - | same |
| L20 down c66 | 9% / 9% | **a** (R2 1.00): a in {60, 62..69} | same | - | same |
| L20 down c50 | 10% / 10% | **a** (R2 1.00): a in {5, 7..13, 15, 20} | same | - | same |
| L20 down c46 | 9% / 9% | **a** (R2 1.00): a in {40, 42..49} | same | - | same |
| L20 down c35 | 9% / 9% | **a** (R2 1.00): a in {12, 18, 24, 36, 42, 48, 60, 72, 96} | **a** (R2 0.99): a in {12, 18, 24, 36, 42, 48, 60, 72, 96} | - | same |
| L20 down c22 | 8% / 8% | **a** (R2 1.00): a in {21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 50 in {21, 31, 41}, R2 0.86] | same | - | same |
| L20 down c7 | 86% / 86% | **a** (R2 1.00): a in {13..14, 16..99} | same | a: mod100 +3% | a: mod100 +3% |
| L20 v c85 (kv0) | 1% / 6% | **a** (R2 0.74): a in {52} | **a** (R2 0.82): a in {32, 48, 52..53, 92} | (reads) | same |
| L20 v c73 (kv0) | 1% / 8% | **a** (R2 0.81): a in {31} | **a** (R2 0.96): a in {21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 50 in {21, 31, 41}, R2 0.83] | (reads) | same |
| L20 gate c78 | 6% / 6% | **a** (R2 1.00): a in {30, 60, 90..93} | same | (reads) | same |
| L20 v c221 (kv0) | 2% / 6% | **a** (R2 0.77): a in {9, 99} | **a** (R2 0.88): a in {9, 49, 69, 97..99} | (reads) | same |
| L20 v c244 (kv0) | 7% / 13% | **a** (R2 0.56): a in {12, 24, 48, 96} | **a** (R2 0.76): a in {6, 8..9, 12, 16, 24, 32, 36, 48, 64, 72, 96} | (reads) | same |
| L20 v c206 (kv0) | 4% / 11% | **a** (R2 0.56): a in {32, 64} | **a** (R2 0.69): a in {8, 16, 20, 32, 40, 64, 80, 88, 96} | (reads) | same |
| L20 v c25 (kv4) | 5% / 5% | **a** (R2 1.00): a in {1..4, 6} | **a** (R2 0.99): a in {1..4, 6} | (reads) | same |
| L21 gate c18 | 20% / 20% | **a** (R2 1.00): a in {26, 28..46} | same | (reads) | same |
| L21 down c573 | 5% / 5% | **a** (R2 1.00): a in {49, 95..97, 99} | same | - | same |
| L21 gate c47 | 9% / 9% | **a** (R2 1.00): a in {12, 14, 18, 22, 24, 36, 42, 48, 72} | same | (reads) | same |
| L21 gate c50 | 4% / 4% | **a** (R2 1.00): a in {84..85, 87..88} | same | (reads) | same |
| L21 down c524 | 3% / 3% | **a** (R2 1.00): a in {45, 75, 95} [coarser: a mod 50 in {45}, R2 0.83] | same | - | same |
| L21 down c457 | 2% / 2% | **a** (R2 1.00): a in {9, 11} | same | - | same |
| L21 down c243 | 3% / 3% | **a** (R2 1.00): a in {20, 38, 40} | same | - | same |
| L21 down c272 | 7% / 7% | **a** (R2 1.00): a in {39..44, 49} | same | - | same |
| L21 down c177 | 7% / 7% | **a** (R2 1.00): a in {55, 59..62, 64..65} | same | - | same |
| L21 down c203 | 4% / 4% | **a** (R2 1.00): a in {33, 73, 83, 93} | same | - | same |
| L21 down c419 | 8% / 8% | **a** (R2 1.00): a in {10, 20, 25, 30, 40, 50, 60, 100} | same | - | same |
| L21 down c302 | 5% / 5% | **a** (R2 1.00): a in {16..17, 32, 64, 96} | same | - | same |
| L21 down c116 | 7% / 7% | **a** (R2 1.00): a in {11..12, 14, 17..18, 21..22} | same | - | same |
| L21 down c109 | 4% / 4% | **a** (R2 1.00): a in {85, 87..89} | same | - | same |
| L21 down c106 | 5% / 5% | **a** (R2 1.00): a in {18..20, 26, 76} | same | - | same |
| L21 down c98 | 3% / 3% | **a** (R2 1.00): a in {8..9, 48} | same | - | same |
| L21 down c87 | 8% / 8% | **a** (R2 1.00): a in {21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 50 in {21, 31, 41}, R2 0.86] | same | - | same |
| L21 down c72 | 4% / 4% | **a** (R2 1.00): a in {37..39, 59} | same | - | same |
| L21 down c66 | 3% / 3% | **a** (R2 1.00): a in {30, 60, 90} | same | - | same |
| L21 down c47 | 6% / 6% | **a** (R2 1.00): a in {3, 5, 7, 15, 17, 21} | same | - | same |
| L21 down c33 | 14% / 14% | **a** (R2 1.00): a in {8, 12, 16, 24, 32, 36, 48, 52, 56, 64, 72, 88, 92, 96} | same | a: mod4 +10%, mod2 +3% | a: mod4 +10%, mod2 +3% |
| L21 down c8 | 39% / 39% | **a** (R2 1.00): a in {2..36, 42..43, 48, 64} | same | - | same |
| L21 gate c272 | 2% / 2% | **a** (R2 1.00): a in {9, 39} | same | (reads) | same |
| L21 gate c144 | 5% / 5% | **a** (R2 1.00): a in {4, 14, 24, 74, 94} | same | (reads) | same |
| L21 down c154 | 9% / 9% | **a** (R2 1.00): a in {35, 55, 79..85} | same | - | same |
| L21 down c144 | 8% / 8% | **a** (R2 1.00): a in {2..4, 14, 54, 74, 84, 94} | same | - | same |
| L21 down c156 | 3% / 3% | **a** (R2 1.00): a in {14..15, 21} | same | - | same |
| L21 down c128 | 2% / 2% | **a** (R2 1.00): a in {11, 23} | same | - | same |
| L21 up c321 | 5% / 5% | **a** (R2 1.00): a in {9, 29, 39, 49, 99} | same | (reads) | same |
| L21 up c312 | 4% / 4% | **a** (R2 1.00): a in {31, 51, 61, 91} | same | (reads) | same |
| L21 up c98 | 7% / 7% | **a** (R2 1.00): a in {8..9, 38, 48, 68, 88, 98} | same | (reads) | same |
| L21 up c106 | 3% / 3% | **a** (R2 1.00): a in {19, 76, 79} | same | (reads) | same |
| L21 gate c312 | 8% / 8% | **a** (R2 1.00): a in {21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 50 in {21, 31, 41}, R2 0.86] | same | (reads) | same |
| L21 up c33 | 14% / 14% | **a** (R2 1.00): a in {8, 12, 16, 24, 32, 36, 48, 52, 56, 64, 72, 88, 92, 96} | same | (reads) | same |
| L21 gate c302 | 3% / 3% | **a** (R2 1.00): a in {16, 32, 64} | same | (reads) | same |
| L21 up c144 | 7% / 7% | **a** (R2 1.00): a in {4, 14, 34, 54, 74, 84, 94} [coarser: a mod 20 in {14}, R2 0.82] | same | (reads) | same |
| L21 up c166 | 2% / 2% | **a** (R2 1.00): a in {50, 52} | same | (reads) | same |
| L21 up c364 | 4% / 4% | **a** (R2 1.00): a in {4, 44, 64, 74} | same | (reads) | same |
| L21 gate c98 | 2% / 2% | **a** (R2 1.00): a in {8, 88} | same | (reads) | same |
| L22 gate c875 | 3% / 3% | **a** (R2 1.00): a in {68, 88, 98} | same | (reads) | same |
| L22 gate c148 | 5% / 5% | **a** (R2 1.00): a in {54..57, 60} | same | (reads) | same |
| L22 gate c120 | 10% / 10% | **a** (R2 1.00): a in {70, 72..80} | same | (reads) | same |
| L22 gate c119 | 3% / 3% | **a** (R2 1.00): a in {24, 72, 92} | same | (reads) | same |
| L22 gate c524 | 3% / 3% | **a** (R2 1.00): a in {25, 49..50} | same | (reads) | same |
| L22 down c39 | 8% / 8% | **a** (R2 1.00): a in {21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 50 in {21, 31, 41}, R2 0.86] | same | - | same |
| L22 down c26 | 10% / 10% | **a** (R2 0.98): a in {12, 24, 32, 36, 48, 52, 64, 72, 76, 96} | **a** (R2 0.99): a in {12, 24, 32, 36, 48, 52, 64, 72, 76, 96} | a: mod4 +3% | a: mod4 +3% |
| L22 down c21 | 13% / 13% | **a** (R2 1.00): a in {32, 38..49} | same | - | same |
| L22 up c667 | 27% / 27% | **a** (R2 1.00): a in {72..80, 82..99} | same | (reads) | same |
| L22 up c308 | 9% / 9% | **a** (R2 1.00): a in {20, 30, 32, 40, 60, 64, 70, 80, 90} [coarser: a mod 50 in {20, 30, 40}, R2 0.82] | same | (reads) | same |
| L22 down c70 | 2% / 2% | **a** (R2 1.00): a in {32, 64} | same | - | same |
| L22 down c113 | 11% / 11% | **a** (R2 1.00): a in {5, 10, 15, 20, 25, 30, 40, 49..50, 99..100} | same | - | same |
| L22 gate c73 | 7% / 7% | **a** (R2 1.00): a in {80, 85..90} | same | (reads) | same |
| L22 gate c54 | 2% / 2% | **a** (R2 1.00): a in {32, 64} | same | (reads) | same |
| L22 gate c39 | 8% / 8% | **a** (R2 1.00): a in {21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 50 in {21, 31, 41}, R2 0.86] | same | (reads) | same |
| L22 gate c21 | 7% / 7% | **a** (R2 1.00): a in {42..43, 45..49} | same | (reads) | same |
| L22 down c308 | 12% / 12% | **a** (R2 1.00): a in {20, 30, 40, 45, 50, 60, 65, 70, 75, 80, 90, 100} [coarser: a mod 50 in {0, 20, 30, 40}, R2 0.81] | same | - | same |
| L22 down c186 | 6% / 6% | **a** (R2 1.00): a in {33, 38..39, 43, 88, 93} [coarser: a mod 50 in {38, 43}, R2 0.82] | same | - | same |
| L22 v c191 (kv3) | 1% / 6% | unexplained (best R2 0.09) | **a** (R2 1.00): a in {79..83, 85} | (reads) | same |
| L22 v c172 (kv3) | 10% / 17% | **a** (R2 0.85): a in {24..31} | **a** (R2 1.00): a in {21..31, 33..37, 45} | (reads) | same |
| L22 down c52 | 9% / 9% | **a** (R2 1.00): a in {82, 86..93} | same | - | same |
| L22 v c143 (kv6) | 2% / 2% | **a** (R2 0.94): a in {60, 90} | **a** (R2 0.91): a in {60, 90} | (reads) | same |
| L22 v c165 (kv3) | 9% / 11% | **a** (R2 0.92): a in {92..100} | **a** (R2 0.99): a in {25, 50, 75, 77, 94..100} | (reads) | same |
| L22 v c194 (kv3) | 0 / 7% | off (on 0) | **a** (R2 1.00): a in {26..29, 87..89} | (reads) | same |
| L22 v c192 (kv3) | 6% / 14% | **a** (R2 0.56): a in {34, 36..38} | **a** (R2 1.00): a in {30..42, 45} | (reads) | same |
| L22 v c199 (kv3) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {17..18, 88} | (reads) | same |
| L22 v c220 (kv0) | 1% / 6% | unexplained (best R2 0.10) | **a** (R2 1.00): a in {72, 76..77, 79..81} | (reads) | same |
| L22 v c101 (kv3) | 4% / 10% | **a** (R2 0.63): a in {52, 55} | **a** (R2 1.00): a in {45, 50..57, 60} | (reads) | same |
| L22 v c89 (kv3) | 5% / 9% | **a** (R2 0.96): a in {4, 24, 48, 72, 96} | **a** (R2 0.97): a in {12, 16, 23..24, 26, 36, 48, 72, 96} | (reads) | same |
| L22 v c78 (kv3) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {10, 20, 40, 50} | (reads) | same |
| L22 v c59 (kv3) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {21, 24} | (reads) | same |
| L22 v c41 (kv3) | 13% / 14% | **a** (R2 0.92): a in {4, 6..17} | **a** (R2 0.97): a in {4, 6..18} | (reads) | same |
| L22 v c18 (kv0) | 0 / 8% | off (on 0) | **a** (R2 1.00): a in {22..23, 25..29, 33} | (reads) | same |
| L22 o c6 (H25) | 9% / 9% | **a** (R2 1.00): a in {16..19, 21..24, 31} | same | - | same |
| L22 v c239 (kv3) | 1% / 11% | **a** (R2 0.50): a in {31} | **a** (R2 0.98): a in {28..33, 52, 55..57, 85} | (reads) | same |
| L22 v c231 (kv3) | 5% / 6% | **a** (R2 0.90): a in {31, 41, 51, 91} | **a** (R2 0.99): a in {21, 31, 41, 51, 81, 91} [coarser: a mod 50 in {31, 41}, R2 0.82] | (reads) | same |
| L22 up c26 | 12% / 12% | **a** (R2 1.00): a in {12, 24, 32, 36, 48, 52, 60, 64, 72, 76, 90, 96} | same | (reads) | same |
| L22 up c263 | 5% / 5% | **a** (R2 1.00): a in {9, 39, 49, 69, 99} | same | (reads) | same |
| L22 up c186 | 8% / 8% | **a** (R2 1.00): a in {33, 38..40, 42..43, 88, 93} | same | (reads) | same |
| L22 v c130 (kv3) | 10% / 16% | **a** (R2 0.79): a in {15, 25, 30, 35, 40, 45, 55, 65, 75, 85} | **a** (R2 0.95): a in {15, 25, 30, 32..38, 40, 45, 50, 55, 85} | (reads) | same |
| L22 v c114 (kv3) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {21, 41..42} | (reads) | same |
| L22 v c110 (kv3) | 1% / 5% | **a** (R2 0.82): a in {3} | **a** (R2 0.99): a in {13, 23, 33, 83, 93} | (reads) | same |
| L22 v c109 (kv3) | 13% / 15% | **a** (R2 0.92): a in {86..98} | **a** (R2 0.97): a in {80, 85..97} | (reads) | same |
| L22 v c167 (kv3) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {17, 19..23} | (reads) | same |
| L23 down c148 | 4% / 4% | **a** (R2 1.00): a in {24, 48, 72, 74} | same | - | same |
| L23 gate c72 | 2% / 2% | **a** (R2 1.00): a in {24, 26} | same | (reads) | same |
| L23 gate c176 | 3% / 3% | **a** (R2 1.00): a in {80, 88, 90} | same | (reads) | same |
| L23 gate c74 | 3% / 3% | **a** (R2 1.00): a in {30, 60, 90} | same | (reads) | same |
| L23 gate c239 | 4% / 4% | **a** (R2 1.00): a in {14, 17..19} | same | (reads) | same |
| L23 up c39 | 3% / 3% | **a** (R2 1.00): a in {16, 32, 64} | same | (reads) | same |
| L23 up c46 | 5% / 5% | **a** (R2 1.00): a in {9, 49, 97..99} | same | (reads) | same |
| L23 gate c182 | 3% / 3% | **a** (R2 1.00): a in {50, 80, 100} [coarser: a mod 50 in {0}, R2 0.83] | same | (reads) | same |
| L23 up c288 | 10% / 10% | **a** (R2 1.00): a in {24, 26..33, 64} | same | (reads) | same |
| L23 up c612 | 10% / 10% | **a** (R2 1.00): a in {1..3, 21, 31, 41, 51, 61, 81, 91} | same | (reads) | same |
| L23 k c90 (kv1) | 0 / 2% | off (on 0) | **a** (R2 0.99): a in {88, 98} | (reads) | same |
| L23 k c129 (kv1) | 2% / 6% | **a** (R2 0.92): a in {88, 98} | **a** (R2 1.00): a in {28, 58, 68, 78, 88, 98} | (reads) | same |
| L23 k c142 (kv1) | 1% / 4% | **a** (R2 0.91): a in {98} | **a** (R2 1.00): a in {58, 68, 88, 98} | (reads) | same |
| L23 down c37 | 8% / 8% | **a** (R2 1.00): a in {1, 21, 31, 41, 51, 61, 81, 91} [coarser: a mod 20 in {1, 11}, R2 0.84] | same | - | same |
| L23 down c40 | 14% / 14% | **a** (R2 1.00): a in {30, 55, 57..63, 65..68, 90} | same | - | same |
| L23 down c44 | 11% / 11% | **a** (R2 1.00): a in {38..40, 42..44, 46..49, 99} | same | - | same |
| L23 v c216 (kv3) | 1% / 2% | **a** (R2 0.86): a in {50} | **a** (R2 0.95): a in {25, 50} | (reads) | same |
| L23 down c67 | 14% / 14% | **a** (R2 1.00): a in {30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100} | same | - | same |
| L23 down c114 | 7% / 7% | **a** (R2 1.00): a in {13..14, 16..19, 21} | same | - | same |
| L23 down c122 | 2% / 2% | **a** (R2 1.00): a in {80, 88} | same | - | same |
| L23 down c412 | 4% / 4% | **a** (R2 1.00): a in {36, 48, 52, 72} | same | - | same |
| L23 down c283 | 3% / 3% | **a** (R2 1.00): a in {16, 32, 64} | same | - | same |
| L23 down c493 | 9% / 9% | **a** (R2 1.00): a in {5..12, 15} | same | - | same |
| L23 up c182 | 25% / 25% | **a** (R2 1.00): a in {74, 76..99} | same | (reads) | same |
| L23 down c741 | 4% / 4% | **a** (R2 1.00): a in {49..50, 52, 55} | same | - | same |
| L23 v c246 (kv1) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {55, 58} | (reads) | same |
| L23 gate c56 | 4% / 4% | **a** (R2 1.00): a in {40, 42..44} | same | (reads) | same |
| L23 gate c1 | 84% / 84% | **a** (R2 1.00): a in {14, 16..19, 21..99} | same | (reads) | same |
| L23 v c91 (kv1) | 0 / 2% | off (on 0) | **a** (R2 0.99): a in {68, 78} | (reads) | same |
| L23 v c201 (kv1) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {55, 58} | (reads) | same |
| L23 v c29 (kv3) | 6% / 8% | **a** (R2 0.90): a in {20, 25, 50, 75, 100} [coarser: a mod 25 in {0}, R2 0.82] | **a** (R2 0.93): a in {20, 25, 40, 50, 75, 80, 100} [coarser: a mod 50 in {0, 20, 25, 30, 40}, R2 0.80] | (reads) | same |
| L24 up c109 | 24% / 24% | **a** (R2 1.00): a in {70, 74, 76, 78..80, 82..99} | **a** (R2 0.99): a in {70, 74, 76, 78..80, 82..99} | (reads) | same |
| L24 down c45 | 6% / 6% | **a** (R2 1.00): a in {1, 41, 51, 61, 81, 91} [coarser: a mod 50 in {1, 41}, R2 0.82] | same | - | same |
| L24 up c62 | 6% / 6% | **a** (R2 1.00): a in {1..2, 41, 51, 61, 91} [coarser: a mod 50 in {1, 41}, R2 0.82] | same | (reads) | same |
| L24 up c97 | 2% / 2% | **a** (R2 1.00): a in {52, 55} | same | (reads) | same |
| L24 v c233 (kv5) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {31, 61} | (reads) | same |
| L24 up c174 | 8% / 8% | **a** (R2 1.00): a in {20, 40, 50, 60, 75, 80, 99..100} | same | (reads) | same |
| L24 v c185 (kv5) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {24, 64} | (reads) | same |
| L24 v c232 (kv5) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {24, 61..63} | (reads) | same |
| L24 up c117 | 2% / 2% | **a** (R2 1.00): a in {60, 90} | same | (reads) | same |
| L24 gate c9 | 13% / 13% | **a** (R2 1.00): a in {12, 16, 18, 22, 24, 32, 36, 42, 48, 52, 64, 72, 96} | same | (reads) | same |
| L24 down c9 | 15% / 15% | **a** (R2 1.00): a in {12, 16, 18, 22, 24, 32, 36, 42, 44, 46, 48, 52, 64, 72, 96} | same | a: mod4 +3%, mod2 +2% | a: mod4 +3%, mod2 +2% |
| L24 down c49 | 4% / 4% | **a** (R2 1.00): a in {60, 80, 88, 90} | same | - | same |
| L24 gate c626 | 2% / 2% | **a** (R2 1.00): a in {4, 8} | same | (reads) | same |
| L24 down c334 | 4% / 4% | **a** (R2 1.00): a in {95, 97..99} | same | - | same |
| L24 down c61 | 7% / 7% | **a** (R2 1.00): a in {74, 76, 78, 80, 82..84} | same | - | same |
| L24 down c95 | 5% / 5% | **a** (R2 1.00): a in {37..39, 59, 99} | same | - | same |
| L24 down c310 | 7% / 7% | **a** (R2 1.00): a in {24, 26..31} | same | - | same |
| L24 gate c651 | 6% / 6% | **a** (R2 1.00): a in {1, 41, 51, 61, 81, 91} [coarser: a mod 50 in {1, 41}, R2 0.82] | same | (reads) | same |
| L24 gate c95 | 5% / 5% | **a** (R2 1.00): a in {37..39, 59, 99} | same | (reads) | same |
| L25 down c176 | 6% / 6% | **a** (R2 1.00): a in {49..50, 95, 97..99} | same | - | same |
| L25 up c5 | 10% / 10% | **a** (R2 1.00): a in {1..2, 4, 31, 41, 51, 61, 71, 81, 91} | same | (reads) | same |
| L25 up c125 | 10% / 10% | **a** (R2 1.00): a in {47, 49..50, 93..99} | same | (reads) | same |
| L25 down c39 | 4% / 4% | **a** (R2 1.00): a in {24, 36, 48, 72} | same | - | same |
| L25 gate c878 | 16% / 16% | **a** (R2 1.00): a in {12, 16, 18, 22, 24, 32, 36, 38, 40, 42, 46, 48, 52, 64, 72, 96} | same | (reads) | same |
| L25 down c47 | 11% / 11% | **a** (R2 1.00): a in {12, 16, 18, 22, 32, 36, 42, 46, 48, 52, 72} | same | - | same |
| L25 down c360 | 6% / 6% | **a** (R2 1.00): a in {1, 31, 41, 51, 61, 91} [coarser: a mod 50 in {1, 41}, R2 0.82] | same | - | same |
| L25 down c293 | 8% / 8% | **a** (R2 1.00): a in {30, 56..60, 62..63} | same | - | same |
| L25 down c338 | 33% / 33% | **a** (R2 1.00): a in {1..32, 34} | same | - | same |
| L25 gate c39 | 6% / 6% | **a** (R2 1.00): a in {24, 36, 48, 72, 74, 96} | same | (reads) | same |
| L25 gate c156 | 11% / 11% | **a** (R2 1.00): a in {52..60, 62..63} | same | (reads) | same |
| L25 gate c125 | 8% / 8% | **a** (R2 1.00): a in {23..24, 26..31} | same | (reads) | same |
| L25 gate c142 | 10% / 10% | **a** (R2 1.00): a in {88, 90..98} | same | (reads) | same |
| L25 down c65 | 5% / 5% | **a** (R2 1.00): a in {45, 88, 90, 92..93} | same | - | same |
| L25 up c36 | 6% / 6% | **a** (R2 1.00): a in {8, 16, 32, 64, 80, 88} | same | (reads) | same |
| L25 down c97 | 11% / 11% | **a** (R2 1.00): a in {5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100} | same | - | same |
| L25 down c119 | 17% / 17% | **a** (R2 1.00): a in {20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100} [coarser: a mod 5 in {0}, R2 0.82] | same | a: mod5 +3% | a: mod5 +3% |
| L25 down c28 | 6% / 6% | **a** (R2 1.00): a in {92..94, 96..98} | same | - | same |
| L25 down c36 | 6% / 6% | **a** (R2 1.00): a in {8, 16, 32, 64, 80, 88} | same | - | same |
| L25 gate c144 | 37% / 37% | **a** (R2 1.00): a in {1..32, 34..37, 40} | same | (reads) | same |
| L26 down c38 | 6% / 6% | **a** (R2 1.00): a in {29, 49, 69, 76, 79, 89} | same | - | same |
| L26 down c5 | 21% / 21% | **a** (R2 1.00): a in {2..19, 21, 23..24} | same | - | same |
| L26 up c28 | 4% / 4% | **a** (R2 1.00): a in {39, 49, 69, 99} | same | (reads) | same |
| L26 up c114 | 2% / 2% | **a** (R2 1.00): a in {18, 36} | same | (reads) | same |
| L26 up c200 | 8% / 8% | **a** (R2 1.00): a in {21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 50 in {21, 31, 41}, R2 0.86] | same | (reads) | same |
| L26 down c65 | 4% / 4% | **a** (R2 1.00): a in {24, 36, 48, 72} | same | - | same |
| L26 gate c120 | 6% / 6% | **a** (R2 1.00): a in {30, 45, 55, 59..60, 90} | same | (reads) | same |
| L26 gate c38 | 5% / 5% | **a** (R2 1.00): a in {49..50, 69, 76, 99} | same | (reads) | same |
| L26 gate c30 | 7% / 7% | **a** (R2 1.00): a in {1..3, 22, 32, 42, 52} | **a** (R2 0.99): a in {1..3, 22, 32, 42, 52} | (reads) | same |
| L26 gate c338 | 8% / 8% | **a** (R2 1.00): a in {10, 20, 40, 50, 52, 70, 80, 100} | same | (reads) | same |
| L26 down c82 | 8% / 8% | **a** (R2 1.00): a in {21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 50 in {21, 31, 41}, R2 0.86] | same | - | same |
| L26 gate c496 | 2% / 2% | **a** (R2 1.00): a in {76, 96} | same | (reads) | same |
| L26 gate c417 | 7% / 7% | **a** (R2 1.00): a in {70, 80..84, 90} | same | (reads) | same |
| L26 up c606 | 7% / 7% | **a** (R2 1.00): a in {24, 26..31} | same | (reads) | same |
| L26 up c630 | 4% / 4% | **a** (R2 1.00): a in {42, 46..47, 49} | same | (reads) | same |
| L26 up c669 | 3% / 3% | **a** (R2 1.00): a in {84..85, 88} | same | (reads) | same |
| L26 down c54 | 6% / 6% | **a** (R2 1.00): a in {60, 70, 80, 88, 90, 100} | same | - | same |
| L26 down c453 | 5% / 5% | **a** (R2 1.00): a in {49..50, 97..99} | same | - | same |
| L26 gate c5 | 7% / 7% | **a** (R2 1.00): a in {9, 14, 18, 34, 54, 74, 84} | same | (reads) | same |
| L26 down c587 | 4% / 4% | **a** (R2 1.00): a in {40, 50, 70, 80} | same | - | same |
| L26 down c562 | 25% / 25% | **a** (R2 1.00): a in {73..74, 76..79, 81..99} | same | - | same |
| L26 gate c16 | 36% / 36% | **a** (R2 1.00): a in {6..31, 90..99} | same | (reads) | same |
| L26 down c506 | 5% / 5% | **a** (R2 1.00): a in {5..8, 10} | same | - | same |
| L26 down c200 | 6% / 6% | **a** (R2 1.00): a in {41, 51, 61, 71, 81, 91} | same | - | same |
| L26 down c369 | 9% / 9% | **a** (R2 1.00): a in {60, 62..69} | same | - | same |
| L26 down c173 | 6% / 6% | **a** (R2 1.00): a in {15, 20, 30, 40, 60, 90} | same | - | same |
| L26 up c26 | 48% / 48% | **a** (R2 1.00): a in {2..48, 50} | same | (reads) | same |
| L26 down c107 | 5% / 5% | **a** (R2 1.00): a in {42..43, 46..47, 49} | same | - | same |
| L26 down c114 | 6% / 6% | **a** (R2 1.00): a in {14, 18, 34..36, 54} | same | - | same |
| L27 up c63 | 10% / 10% | **a** (R2 1.00): a in {12, 16, 24, 32, 36, 42, 48, 64, 72, 96} | same | (reads) | same |
| L27 gate c885 | 3% / 3% | **a** (R2 1.00): a in {50, 52, 100} [coarser: a mod 50 in {0}, R2 0.83] | same | (reads) | same |
| L27 gate c523 | 2% / 2% | **a** (R2 1.00): a in {88, 92} | same | (reads) | same |
| L27 up c188 | 6% / 6% | **a** (R2 1.00): a in {30, 40, 45, 60, 80, 90} [coarser: a mod 50 in {30, 40}, R2 0.82] | same | (reads) | same |
| L27 up c411 | 28% / 28% | **a** (R2 1.00): a in {67, 74..100} | same | (reads) | same |
| L27 up c414 | 17% / 17% | **a** (R2 1.00): a in {5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99..100} [coarser: a mod 50 in {0, 10, 20, 25, 30, 40}, R2 0.82] | same | (reads) | same |
| L27 gate c197 | 5% / 5% | **a** (R2 1.00): a in {8..9, 88..89, 99} | same | (reads) | same |
| L27 down c690 | 16% / 16% | **a** (R2 1.00): a in {20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100} [coarser: a mod 50 in {0, 20, 25, 30, 35, 40}, R2 0.85] | same | a: mod5 +3% | a: mod5 +3% |
| L27 gate c425 | 7% / 7% | **a** (R2 1.00): a in {24, 32, 36, 48, 64, 72, 96} | same | (reads) | same |
| L27 down c608 | 3% / 3% | **a** (R2 1.00): a in {30, 60, 90} | same | - | same |
| L27 down c472 | 6% / 6% | **a** (R2 1.00): a in {5, 7, 10..11, 15, 20} | same | - | same |
| L27 down c327 | 11% / 11% | **a** (R2 1.00): a in {1..10, 12} | same | - | same |
| L27 down c552 | 5% / 5% | **a** (R2 1.00): a in {88, 90..93} | same | - | same |
| L27 up c739 | 8% / 8% | **a** (R2 1.00): a in {21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 50 in {21, 31, 41}, R2 0.86] | same | (reads) | same |
| L27 v c152 (kv1) | 0 / 12% | off (on 0) | **a** (R2 1.00): a in {3..10, 12, 16, 24, 26} | (reads) | same |
| L27 up c419 | 42% / 42% | **a** (R2 1.00): a in {2..36, 38, 40, 42..43, 46, 48..49} | same | (reads) | same |
| L27 down c109 | 2% / 2% | **a** (R2 1.00): a in {41, 51} | same | - | same |
| L27 down c144 | 5% / 5% | **a** (R2 1.00): a in {21..23, 28..29} | same | - | same |
| L27 down c36 | 5% / 5% | **a** (R2 1.00): a in {50, 75, 95, 99..100} | same | - | same |
| L27 gate c625 | 16% / 16% | **a** (R2 1.00): a in {3, 5..13, 15..20} | same | (reads) | same |
| L27 gate c690 | 11% / 11% | **a** (R2 1.00): a in {20, 30, 40, 45, 50, 60, 70, 75, 80, 90, 100} [coarser: a mod 50 in {0, 20, 30, 40}, R2 0.85] | same | (reads) | same |
| L27 gate c353 | 5% / 5% | **a** (R2 1.00): a in {95, 97..100} | same | (reads) | same |
| L27 down c277 | 7% / 7% | **a** (R2 1.00): a in {7..12, 15} | same | - | same |
| L27 down c141 | 8% / 8% | **a** (R2 1.00): a in {21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 50 in {21, 31, 41}, R2 0.86] | same | - | same |
| L27 down c103 | 7% / 7% | **a** (R2 1.00): a in {24, 32, 36, 48, 64, 72, 96} | same | a: mod4 +2% | a: mod4 +2% |
| L27 down c63 | 8% / 8% | **a** (R2 1.00): a in {12, 16, 32, 36, 42, 48, 72, 96} | same | - | same |
| L28 down c589 | 38% / 38% | **a** (R2 1.00): a in {1..36, 40, 50} | same | a: mod100 +4%, mod50 +2% | a: mod100 +4%, mod50 +2% |
| L28 down c424 | 10% / 10% | **a** (R2 1.00): a in {21..23, 25..31} | same | - | same |
| L28 gate c179 | 2% / 2% | **a** (R2 1.00): a in {52, 55} | same | (reads) | same |
| L28 down c405 | 8% / 9% | **a** (R2 1.00): a in {92..99} | **a** (R2 1.00): a in {88, 92..99} | - | same |
| L28 down c350 | 5% / 5% | **a** (R2 1.00): a in {49, 51..54} | same | - | same |
| L28 down c135 | 16% / 16% | **a** (R2 1.00): a in {32..46, 48} | same | - | same |
| L28 down c9 | 14% / 14% | **a** (R2 1.00): a in {3, 5..13, 15, 20, 25, 50} | same | - | same |
| L28 down c79 | 5% / 5% | **a** (R2 0.99): a in {14, 18, 24, 36, 48} | **a** (R2 1.00): a in {14, 18, 24, 36, 48} | - | same |
| L28 down c17 | 17% / 17% | **a** (R2 1.00): a in {20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100} [coarser: a mod 5 in {0}, R2 0.82] | same | a: mod5 +4% | a: mod5 +4% |
| L28 down c18 | 2% / 2% | **a** (R2 1.00): a in {80, 100} | same | - | same |
| L28 down c63 | 3% / 3% | **a** (R2 1.00): a in {5, 7, 9} | same | - | same |
| L28 gate c68 | 17% / 17% | **a** (R2 1.00): a in {20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100} [coarser: a mod 5 in {0}, R2 0.82] | same | (reads) | same |
| L28 down c741 | 2% / 2% | **a** (R2 1.00): a in {12, 48} | same | - | same |
| L28 gate c360 | 38% / 38% | **a** (R2 1.00): a in {1..36, 40, 50} | same | (reads) | same |
| L28 gate c907 | 3% / 3% | **a** (R2 1.00): a in {5, 7, 9} | same | (reads) | same |
| L28 up c786 | 20% / 20% | **a** (R2 1.00): a in {11, 13..31} | same | (reads) | same |
| L28 up c326 | 4% / 4% | **a** (R2 1.00): a in {35, 37..39} | same | (reads) | same |
| L28 gate c213 | 6% / 7% | **a** (R2 0.96): a in {95..100} | **a** (R2 1.00): a in {88, 95..100} | (reads) | same |
| L28 gate c736 | 3% / 3% | **a** (R2 1.00): a in {24, 36, 48} | same | (reads) | same |
| L29 down c457 | 4% / 4% | **a** (R2 1.00): a in {88, 97..99} | same | - | same |
| L29 down c212 | 10% / 10% | **a** (R2 1.00): a in {8, 12, 16, 24, 32, 36, 48, 64, 72, 96} | same | a: mod4 +4% | a: mod4 +4% |
| L29 down c214 | 4% / 4% | **a** (R2 1.00): a in {36, 48, 72, 96} | same | - | same |
| L29 up c214 | 4% / 4% | **a** (R2 1.00): a in {24, 36, 48, 72} | same | (reads) | same |
| L29 up c508 | 8% / 8% | **a** (R2 1.00): a in {88, 90, 95..100} | same | (reads) | same |
| L29 up c340 | 5% / 5% | **a** (R2 1.00): a in {1..4, 6} | same | (reads) | same |
| L29 up c242 | 9% / 9% | **a** (R2 1.00): a in {23, 73, 83, 90..95} | same | (reads) | same |
| L29 up c557 | 17% / 17% | **a** (R2 1.00): a in {10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100} [coarser: a mod 5 in {0}, R2 0.82] | same | (reads) | same |
| L29 gate c386 | 10% / 10% | **a** (R2 1.00): a in {38, 40, 42..49} | same | (reads) | same |
| L29 gate c485 | 62% / 62% | **a** (R2 1.00): a in {3..60, 63, 65, 99..100} | same | (reads) | same |
| L29 down c63 | 4% / 4% | **a** (R2 1.00): a in {32..34, 64} | same | - | same |
| L29 down c203 | 4% / 4% | **a** (R2 1.00): a in {24, 36, 48, 72} | same | - | same |
| L29 down c242 | 10% / 10% | **a** (R2 1.00): a in {39, 91..99} | same | - | same |
| L29 down c81 | 44% / 44% | **a** (R2 1.00): a in {53..54, 56..59, 61..98} | same | a: mod100 +3% | a: mod100 +3% |
| L29 down c30 | 2% / 2% | **a** (R2 1.00): a in {14, 18} | same | - | same |
| L29 down c22 | 2% / 2% | **a** (R2 1.00): a in {60, 80} | same | - | same |
| L29 down c574 | 3% / 3% | **a** (R2 1.00): a in {12, 36, 48} | same | - | same |
| L29 down c838 | 4% / 4% | **a** (R2 1.00): a in {8..9, 11..12} | same | - | same |
| L29 down c576 | 12% / 12% | **a** (R2 1.00): a in {22, 38, 40..49} | same | - | same |
| L29 down c1005 | 4% / 4% | **a** (R2 1.00): a in {20, 30, 40, 50} | same | - | same |
| L29 gate c214 | 9% / 9% | **a** (R2 1.00): a in {12, 16, 24, 32, 36, 48, 64, 72, 96} | same | (reads) | same |
| L29 gate c154 | 17% / 17% | **a** (R2 1.00): a in {14, 16..31} | same | (reads) | same |
| L29 gate c160 | 11% / 11% | **a** (R2 0.98): a in {20, 30, 40, 45, 50, 60, 70, 75, 80, 90, 100} [coarser: a mod 50 in {0, 20, 30, 40}, R2 0.84] | **a** (R2 1.00): a in {20, 30, 40, 45, 50, 60, 70, 75, 80, 90, 100} [coarser: a mod 50 in {0, 20, 30, 40}, R2 0.85] | (reads) | same |
| L29 gate c212 | 7% / 7% | **a** (R2 1.00): a in {8, 16, 32..34, 64, 74} | **a** (R2 0.98): a in {8, 16, 32..34, 64, 74} | (reads) | same |
| L29 gate c576 | 5% / 5% | **a** (R2 1.00): a in {22, 42, 62, 82, 92} | same | (reads) | same |
| L29 down c111 | 2% / 2% | **a** (R2 1.00): a in {10, 12} | same | - | same |
| L29 up c634 | 12% / 12% | **a** (R2 1.00): a in {2..12, 15} | same | (reads) | same |
| L30 v c196 (kv7) | 2% / 2% | **a** (R2 0.99): a in {24, 52} | same | (reads) | same |
| L30 gate c476 | 2% / 2% | **a** (R2 0.98): a in {47, 49} | **a** (R2 1.00): a in {47, 49} | (reads) | same |
| L30 up c147 | 33% / 33% | **a** (R2 1.00): a in {3..31, 35, 40, 50, 100} | same | (reads) | same |
| L30 up c259 | 7% / 7% | **a** (R2 1.00): a in {22..24, 26..29} | same | (reads) | same |
| L30 down c759 | 7% / 7% | **a** (R2 1.00): a in {21, 41, 51, 61, 71, 81, 91} | same | - | same |
| L30 down c768 | 6% / 6% | **a** (R2 0.98): a in {28..32, 40} | **a** (R2 1.00): a in {28..32, 40} | - | same |
| L30 down c948 | 2% / 1% | **a** (R2 1.00): a in {3, 33} | **a** (R2 0.88): a in {3} | - | same |
| L30 gate c155 | 8% / 8% | **a** (R2 1.00): a in {21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 50 in {21, 31, 41}, R2 0.86] | same | (reads) | same |
| L30 gate c397 | 6% / 6% | **a** (R2 1.00): a in {29..33, 40} | same | (reads) | same |
| L30 down c155 | 7% / 7% | **a** (R2 1.00): a in {31, 41, 51, 61, 71, 81, 91} | same | - | same |
| L30 down c189 | 4% / 4% | **a** (R2 1.00): a in {60, 70, 80, 90} | same | - | same |
| L30 down c259 | 14% / 13% | **a** (R2 0.99): a in {14, 16, 18..20, 22..30} | **a** (R2 1.00): a in {14, 16, 18..20, 22..29} | - | same |
| L31 v c219 (kv5) | 9% / 10% | **a%20** (R2 0.90): a mod 20 in {0, 10} [coarser: a mod 10 in {0}, R2 0.89] | **a** (R2 1.00): a in {20, 30, 40, 45, 50, 60, 70, 80, 90, 100} [coarser: a mod 10 in {0}, R2 0.80] | (reads) | same |

</details>

<details><summary>single value: 444 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L0 down c17 | 1% / 1% | **a** (R2 1.00): a in {74} | same | - | same |
| L0 down c67 | 1% / 1% | **a** (R2 1.00): a in {62} | same | - | same |
| L0 down c112 | 1% / 1% | **a** (R2 1.00): a in {1} | same | - | same |
| L0 down c131 | 1% / 1% | **a** (R2 1.00): a in {3} | same | - | same |
| L0 down c133 | 1% / 1% | **a** (R2 1.00): a in {10} | same | - | same |
| L0 down c167 | 1% / 1% | **a** (R2 1.00): a in {56} | same | - | same |
| L0 down c169 | 1% / 1% | **a** (R2 1.00): a in {11} | same | - | same |
| L0 down c171 | 1% / 1% | **a** (R2 1.00): a in {26} | same | - | same |
| L0 down c184 | 1% / 1% | **a** (R2 1.00): a in {55} | same | - | same |
| L0 down c207 | 1% / 1% | **a** (R2 1.00): a in {5} | same | - | same |
| L0 down c237 | 1% / 1% | **a** (R2 1.00): a in {31} | same | - | same |
| L0 down c245 | 1% / 1% | **a** (R2 1.00): a in {50} | same | - | same |
| L0 down c687 | 1% / 1% | **a** (R2 1.00): a in {83} | same | - | same |
| L0 down c697 | 1% / 1% | **a** (R2 1.00): a in {29} | same | - | same |
| L0 down c738 | 1% / 1% | **a** (R2 1.00): a in {86} | same | - | same |
| L0 down c795 | 1% / 1% | **a** (R2 1.00): a in {68} | same | - | same |
| L0 down c946 | 1% / 1% | **a** (R2 1.00): a in {81} | same | - | same |
| L0 down c948 | 1% / 1% | **a** (R2 1.00): a in {20} | same | - | same |
| L0 down c984 | 1% / 1% | **a** (R2 1.00): a in {33} | same | - | same |
| L0 down c1020 | 1% / 1% | **a** (R2 1.00): a in {91} | same | - | same |
| L0 gate c52 | 1% / 1% | **a** (R2 1.00): a in {13} | same | (reads) | same |
| L0 gate c86 | 1% / 1% | **a** (R2 1.00): a in {41} | same | (reads) | same |
| L0 gate c112 | 1% / 1% | **a** (R2 1.00): a in {1} | same | (reads) | same |
| L0 gate c269 | 1% / 1% | **a** (R2 1.00): a in {44} | same | (reads) | same |
| L0 gate c460 | 1% / 1% | **a** (R2 1.00): a in {74} | same | (reads) | same |
| L0 gate c462 | 1% / 1% | **a** (R2 1.00): a in {5} | same | (reads) | same |
| L0 gate c539 | 1% / 1% | **a** (R2 1.00): a in {21} | same | (reads) | same |
| L0 gate c738 | 1% / 1% | **a** (R2 1.00): a in {86} | same | (reads) | same |
| L0 gate c744 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L0 gate c834 | 1% / 1% | **a** (R2 1.00): a in {90} | same | (reads) | same |
| L0 up c29 | 1% / 1% | **a** (R2 1.00): a in {68} | same | (reads) | same |
| L0 up c44 | 1% / 1% | **a** (R2 1.00): a in {87} | same | (reads) | same |
| L0 up c48 | 1% / 1% | **a** (R2 1.00): a in {54} | same | (reads) | same |
| L0 up c55 | 1% / 1% | **a** (R2 1.00): a in {46} | same | (reads) | same |
| L0 up c62 | 1% / 1% | **a** (R2 1.00): a in {34} | same | (reads) | same |
| L0 up c63 | 1% / 1% | **a** (R2 1.00): a in {38} | same | (reads) | same |
| L0 up c73 | 1% / 1% | **a** (R2 1.00): a in {98} | same | (reads) | same |
| L0 up c75 | 1% / 1% | **a** (R2 1.00): a in {77} | same | (reads) | same |
| L0 up c90 | 1% / 1% | **a** (R2 1.00): a in {62} | same | (reads) | same |
| L0 up c133 | 1% / 1% | **a** (R2 1.00): a in {10} | same | (reads) | same |
| L0 up c148 | 1% / 1% | **a** (R2 1.00): a in {22} | same | (reads) | same |
| L0 up c167 | 1% / 1% | **a** (R2 1.00): a in {56} | same | (reads) | same |
| L0 up c207 | 1% / 1% | **a** (R2 1.00): a in {5} | same | (reads) | same |
| L0 up c237 | 1% / 1% | **a** (R2 1.00): a in {31} | same | (reads) | same |
| L0 up c415 | 1% / 1% | **a** (R2 1.00): a in {42} | same | (reads) | same |
| L0 up c423 | 1% / 1% | **a** (R2 1.00): a in {1} | same | (reads) | same |
| L0 up c467 | 1% / 1% | **a** (R2 1.00): a in {99} | same | (reads) | same |
| L0 up c477 | 1% / 1% | **a** (R2 1.00): a in {74} | same | (reads) | same |
| L0 up c592 | 1% / 1% | **a** (R2 1.00): a in {2} | same | (reads) | same |
| L0 k c115 (kv6) | 1% / 0 | **a** (R2 1.00): a in {90} | off (on 0) | (reads) | same |
| L0 down c572 | 1% / 1% | **a** (R2 1.00): a in {42} | same | - | same |
| L0 down c586 | 1% / 1% | **a** (R2 1.00): a in {99} | same | - | same |
| L0 down c679 | 1% / 1% | **a** (R2 1.00): a in {44} | same | - | same |
| L0 down c309 | 1% / 1% | **a** (R2 1.00): a in {1} | same | - | same |
| L0 down c266 | 1% / 1% | **a** (R2 1.00): a in {6} | same | - | same |
| L0 down c330 | 1% / 1% | **a** (R2 1.00): a in {89} | same | - | same |
| L0 down c426 | 1% / 1% | **a** (R2 1.00): a in {38} | same | - | same |
| L0 down c406 | 1% / 1% | **a** (R2 1.00): a in {46} | **a** (R2 0.96): a in {46} | - | same |
| L0 down c469 | 1% / 1% | **a** (R2 1.00): a in {34} | same | - | same |
| L0 down c387 | 1% / 1% | **a** (R2 1.00): a in {46} | same | - | same |
| L0 down c352 | 1% / 1% | **a** (R2 1.00): a in {2} | same | - | same |
| L0 down c351 | 1% / 1% | **a** (R2 1.00): a in {8} | same | - | same |
| L0 down c349 | 1% / 1% | **a** (R2 1.00): a in {68} | same | - | same |
| L0 down c348 | 1% / 1% | **a** (R2 1.00): a in {4} | same | - | same |
| L0 down c515 | 1% / 1% | **a** (R2 1.00): a in {70} | same | - | same |
| L0 down c539 | 1% / 1% | **a** (R2 1.00): a in {21} | same | - | same |
| L1 down c128 | 1% / 1% | **a** (R2 1.00): a in {91} | same | - | same |
| L1 down c331 | 1% / 1% | **a** (R2 1.00): a in {70} | same | - | same |
| L1 down c468 | 1% / 1% | **a** (R2 1.00): a in {55} | same | - | same |
| L1 down c546 | 1% / 1% | **a** (R2 1.00): a in {48} | same | - | same |
| L1 down c663 | 1% / 1% | **a** (R2 1.00): a in {99} | same | - | same |
| L1 gate c372 | 1% / 1% | **a** (R2 1.00): a in {72} | same | (reads) | same |
| L1 gate c525 | 1% / 1% | **a** (R2 1.00): a in {42} | same | (reads) | same |
| L1 gate c904 | 1% / 1% | **a** (R2 1.00): a in {82} | same | (reads) | same |
| L1 v c45 (kv6) | 1% / 1% | **a** (R2 1.00): a in {24} | same | (reads) | same |
| L1 v c52 (kv6) | 1% / 1% | **a** (R2 0.93): a in {8} | **a** (R2 0.79): a in {8} | (reads) | same |
| L1 v c90 (kv6) | 1% / 1% | **a** (R2 1.00): a in {1} | same | (reads) | same |
| L1 v c93 (kv6) | 1% / 1% | **a** (R2 0.90): a in {80} | **a** (R2 0.88): a in {80} | (reads) | same |
| L1 v c147 (kv6) | 0 / 1% | off (on 0) | **a** (R2 0.62): a in {28} | (reads) | same |
| L1 v c255 (kv6) | 1% / 1% | **a** (R2 1.00): a in {99} | same | (reads) | same |
| L2 v c198 (kv2) | 1% / 0 | **a** (R2 1.00): a in {18} | off (on 0) | (reads) | same |
| L2 down c74 | 1% / 1% | **a** (R2 1.00): a in {88} | same | - | same |
| L2 down c282 | 1% / 1% | **a** (R2 1.00): a in {5} | same | - | same |
| L2 down c643 | 1% / 1% | **a** (R2 1.00): a in {15} | same | - | same |
| L2 down c474 | 1% / 1% | **a** (R2 1.00): a in {17} | same | - | same |
| L2 up c254 | 1% / 1% | **a** (R2 1.00): a in {20} | same | (reads) | same |
| L2 up c162 | 1% / 1% | **a** (R2 1.00): a in {19} | same | (reads) | same |
| L2 gate c643 | 1% / 1% | **a** (R2 1.00): a in {15} | same | (reads) | same |
| L2 down c717 | 1% / 1% | **a** (R2 1.00): a in {92} | same | - | same |
| L2 gate c58 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L2 gate c162 | 1% / 1% | **a** (R2 1.00): a in {19} | same | (reads) | same |
| L2 gate c602 | 1% / 1% | **a** (R2 1.00): a in {33} | same | (reads) | same |
| L2 v c92 (kv2) | 1% / 1% | **a** (R2 1.00): a in {18} | **a** (R2 0.51): a in {18} | (reads) | same |
| L2 down c612 | 1% / 1% | **a** (R2 1.00): a in {22} | same | - | same |
| L2 down c544 | 1% / 1% | **a** (R2 1.00): a in {2} | same | - | same |
| L2 down c543 | 1% / 1% | **a** (R2 1.00): a in {74} | same | - | same |
| L2 v c76 (kv2) | 1% / 1% | **a** (R2 1.00): a in {11} | same | (reads) | same |
| L3 down c510 | 1% / 1% | **a** (R2 1.00): a in {65} | same | - | same |
| L3 up c360 | 1% / 1% | **a** (R2 1.00): a in {6} | same | (reads) | same |
| L3 down c271 | 1% / 1% | **a** (R2 1.00): a in {24} | same | - | same |
| L3 down c17 | 1% / 1% | **a** (R2 1.00): a in {1} | same | - | same |
| L3 v c151 (kv1) | 1% / 0 | **a** (R2 1.00): a in {55} | off (on 0) | (reads) | same |
| L3 v c83 (kv3) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {70} | (reads) | same |
| L3 down c257 | 1% / 1% | **a** (R2 1.00): a in {1} | same | - | same |
| L3 v c222 (kv1) | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L3 v c232 (kv1) | 1% / 1% | **a** (R2 1.00): a in {75} | same | (reads) | same |
| L3 v c198 (kv3) | 1% / 1% | **a** (R2 1.00): a in {67} | same | (reads) | same |
| L3 k c114 (kv1) | 1% / 0 | **a** (R2 1.00): a in {55} | off (on 0) | (reads) | same |
| L3 v c63 (kv3) | 1% / 1% | **a** (R2 1.00): a in {77} | same | (reads) | same |
| L3 v c59 (kv1) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {70} | (reads) | same |
| L3 v c27 (kv3) | 1% / 1% | **a** (R2 1.00): a in {21} | same | (reads) | same |
| L3 v c176 (kv1) | 1% / 0 | **a** (R2 1.00): a in {55} | off (on 0) | (reads) | same |
| L4 down c768 | 1% / 1% | **a** (R2 1.00): a in {83} | same | - | same |
| L4 down c864 | 1% / 1% | **a** (R2 1.00): a in {74} | same | - | same |
| L4 gate c100 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L4 gate c247 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L4 up c365 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L4 down c561 | 1% / 1% | **a** (R2 1.00): a in {1} | same | - | same |
| L4 down c317 | 1% / 1% | **a** (R2 1.00): a in {24} | same | - | same |
| L4 down c110 | 1% / 1% | **a** (R2 1.00): a in {61} | same | - | same |
| L4 up c134 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L4 gate c249 | 1% / 1% | **a** (R2 1.00): a in {88} | same | (reads) | same |
| L4 gate c248 | 1% / 1% | **a** (R2 1.00): a in {55} | same | (reads) | same |
| L4 up c583 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L4 gate c134 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L4 up c387 | 1% / 1% | **a** (R2 1.00): a in {42} | same | (reads) | same |
| L4 up c408 | 1% / 1% | **a** (R2 1.00): a in {42} | same | (reads) | same |
| L4 down c735 | 1% / 1% | **a** (R2 1.00): a in {31} | same | - | same |
| L5 down c14 | 1% / 1% | **a** (R2 1.00): a in {3} | same | - | same |
| L5 down c207 | 1% / 1% | **a** (R2 1.00): a in {83} | same | - | same |
| L5 down c568 | 1% / 1% | **a** (R2 1.00): a in {52} | same | - | same |
| L6 down c187 | 1% / 1% | **a** (R2 1.00): a in {1} | same | - | same |
| L6 down c522 | 1% / 1% | **a** (R2 1.00): a in {91} | same | - | same |
| L6 gate c859 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L6 down c997 | 1% / 1% | **a** (R2 1.00): a in {52} | same | - | same |
| L6 down c862 | 0 / 1% | off (on 0) | **a** (R2 0.98): a in {24} | - | same |
| L6 up c528 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L7 up c199 | 1% / 1% | **a** (R2 0.99): a in {52} | **a** (R2 1.00): a in {52} | (reads) | same |
| L7 down c199 | 1% / 1% | **a** (R2 1.00): a in {52} | same | - | same |
| L7 down c112 | 1% / 1% | **a** (R2 1.00): a in {1} | same | - | same |
| L7 up c159 | 1% / 1% | **a** (R2 1.00): a in {5} | **a** (R2 0.99): a in {5} | (reads) | same |
| L7 down c698 | 1% / 1% | **a** (R2 1.00): a in {2} | same | - | same |
| L7 gate c192 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L7 gate c450 | 1% / 1% | **a** (R2 1.00): a in {83} | same | (reads) | same |
| L7 down c578 | 1% / 1% | **a** (R2 1.00): a in {24} | same | - | same |
| L7 down c450 | 1% / 1% | **a** (R2 1.00): a in {83} | same | - | same |
| L8 down c370 | 1% / 1% | **a** (R2 1.00): a in {52} | same | - | same |
| L8 down c612 | 1% / 1% | **a** (R2 1.00): a in {1} | same | - | same |
| L8 up c829 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L9 down c165 | 1% / 1% | **a** (R2 1.00): a in {52} | same | - | same |
| L9 down c738 | 1% / 1% | **a** (R2 1.00): a in {99} | same | - | same |
| L9 down c841 | 1% / 1% | **a** (R2 1.00): a in {52} | same | - | same |
| L9 gate c360 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L9 gate c529 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L9 gate c664 | 1% / 1% | **a** (R2 1.00): a in {1} | same | (reads) | same |
| L9 up c369 | 1% / 1% | **a** (R2 1.00): a in {1} | same | (reads) | same |
| L9 up c452 | 1% / 1% | **a** (R2 1.00): a in {99} | same | (reads) | same |
| L10 down c138 | 1% / 1% | **a** (R2 1.00): a in {2} | same | - | same |
| L10 down c423 | 1% / 1% | **a** (R2 1.00): a in {83} | same | - | same |
| L10 gate c35 | 1% / 1% | **a** (R2 1.00): a in {1} | same | (reads) | same |
| L10 up c423 | 1% / 1% | **a** (R2 1.00): a in {83} | same | (reads) | same |
| L10 down c86 | 1% / 1% | **a** (R2 1.00): a in {52} | same | - | same |
| L11 down c74 | 1% / 1% | **a** (R2 1.00): a in {42} | **a** (R2 0.99): a in {42} | - | same |
| L11 down c35 | 1% / 1% | **a** (R2 1.00): a in {93} | same | - | same |
| L11 down c43 | 1% / 1% | **a** (R2 1.00): a in {1} | same | - | same |
| L11 up c74 | 1% / 1% | **a** (R2 1.00): a in {42} | **a** (R2 0.97): a in {42} | (reads) | same |
| L11 up c453 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L12 down c932 | 1% / 1% | **a** (R2 1.00): a in {1} | same | - | same |
| L12 down c870 | 1% / 1% | **a** (R2 1.00): a in {62} | same | - | same |
| L12 down c327 | 1% / 1% | **a** (R2 1.00): a in {52} | same | - | same |
| L12 up c821 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L12 gate c419 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L12 gate c212 | 1% / 1% | **a** (R2 1.00): a in {1} | same | (reads) | same |
| L13 down c429 | 1% / 1% | **a** (R2 1.00): a in {51} | same | - | same |
| L13 down c332 | 1% / 1% | **a** (R2 1.00): a in {1} | same | - | same |
| L13 down c208 | 1% / 1% | **a** (R2 1.00): a in {59} | same | - | same |
| L13 gate c572 | 1% / 1% | **a** (R2 1.00): a in {16} | same | (reads) | same |
| L13 gate c309 | 1% / 1% | **a** (R2 1.00): a in {66} | same | (reads) | same |
| L13 gate c133 | 1% / 1% | **a** (R2 1.00): a in {66} | same | (reads) | same |
| L13 gate c46 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L13 down c560 | 1% / 1% | **a** (R2 1.00): a in {49} | same | - | same |
| L14 down c476 | 1% / 1% | **a** (R2 1.00): a in {32} | same | - | same |
| L14 v c109 (kv4) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {24} | (reads) | same |
| L14 up c415 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L14 up c902 | 1% / 1% | **a** (R2 1.00): a in {13} | same | (reads) | same |
| L14 gate c238 | 1% / 1% | **a** (R2 1.00): a in {88} | same | (reads) | same |
| L14 down c105 | 1% / 1% | **a** (R2 1.00): a in {49} | same | - | same |
| L14 down c310 | 1% / 1% | **a** (R2 1.00): a in {3} | same | - | same |
| L14 gate c707 | 1% / 1% | **a** (R2 1.00): a in {13} | same | (reads) | same |
| L14 gate c459 | 1% / 1% | **a** (R2 1.00): a in {83} | same | (reads) | same |
| L14 down c551 | 1% / 1% | **a** (R2 1.00): a in {52} | same | - | same |
| L14 down c637 | 1% / 1% | **a** (R2 1.00): a in {10} | same | - | same |
| L14 down c1002 | 1% / 1% | **a** (R2 1.00): a in {12} | same | - | same |
| L14 gate c588 | 1% / 1% | **a** (R2 1.00): a in {99} | same | (reads) | same |
| L14 gate c320 | 1% / 1% | **a** (R2 1.00): a in {74} | same | (reads) | same |
| L14 down c441 | 1% / 1% | **a** (R2 1.00): a in {52} | same | - | same |
| L15 down c878 | 1% / 1% | **a** (R2 1.00): a in {75} | same | - | same |
| L15 gate c590 | 1% / 1% | **a** (R2 1.00): a in {18} | same | (reads) | same |
| L15 down c853 | 1% / 1% | **a** (R2 1.00): a in {42} | same | - | same |
| L15 down c584 | 1% / 1% | **a** (R2 1.00): a in {21} | same | - | same |
| L15 gate c152 | 1% / 1% | **a** (R2 1.00): a in {74} | same | (reads) | same |
| L15 gate c258 | 1% / 1% | **a** (R2 1.00): a in {20} | same | (reads) | same |
| L15 gate c212 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L15 gate c204 | 1% / 1% | **a** (R2 1.00): a in {20} | same | (reads) | same |
| L15 down c888 | 1% / 1% | **a** (R2 1.00): a in {34} | **a** (R2 0.81): a in {34} | - | same |
| L15 gate c106 | 1% / 1% | **a** (R2 1.00): a in {18} | same | (reads) | same |
| L15 gate c105 | 1% / 1% | **a** (R2 0.71): a in {96} | **a** (R2 1.00): a in {96} | (reads) | same |
| L15 down c927 | 1% / 1% | **a** (R2 1.00): a in {18} | same | - | same |
| L15 down c557 | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {53} | - | same |
| L15 gate c287 | 1% / 1% | **a** (R2 1.00): a in {37} | same | (reads) | same |
| L15 up c470 | 1% / 1% | **a** (R2 1.00): a in {44} | same | (reads) | same |
| L15 up c188 | 1% / 1% | **a** (R2 1.00): a in {28} | same | (reads) | same |
| L15 down c425 | 1% / 1% | **a** (R2 0.56): a in {88} | **a** (R2 1.00): a in {88} | - | same |
| L15 up c532 | 1% / 1% | **a** (R2 1.00): a in {43} | same | (reads) | same |
| L15 gate c415 | 1% / 1% | **a** (R2 1.00): a in {11} | same | (reads) | same |
| L15 gate c328 | 1% / 1% | **a** (R2 1.00): a in {28} | same | (reads) | same |
| L15 gate c366 | 1% / 1% | **a** (R2 1.00): a in {74} | same | (reads) | same |
| L16 down c259 | 1% / 1% | **a** (R2 1.00): a in {99} | same | - | same |
| L16 down c406 | 1% / 1% | **a** (R2 1.00): a in {7} | same | - | same |
| L16 down c443 | 1% / 1% | **a** (R2 1.00): a in {32} | same | - | same |
| L16 down c484 | 1% / 1% | **a** (R2 1.00): a in {41} | same | - | same |
| L16 down c554 | 1% / 1% | **a** (R2 1.00): a in {30} | same | - | same |
| L16 down c250 | 1% / 1% | **a** (R2 1.00): a in {60} | **a** (R2 0.88): a in {60} | - | same |
| L16 down c238 | 1% / 1% | **a** (R2 1.00): a in {51} | same | - | same |
| L16 v c176 (kv5) | 1% / 1% | **a** (R2 0.98): a in {1} | **a** (R2 1.00): a in {1} | (reads) | same |
| L16 down c290 | 1% / 1% | **a** (R2 1.00): a in {1} | same | - | same |
| L16 up c554 | 1% / 1% | **a** (R2 1.00): a in {30} | same | (reads) | same |
| L16 gate c509 | 1% / 1% | **a** (R2 1.00): a in {1} | same | (reads) | same |
| L16 gate c271 | 1% / 1% | **a** (R2 1.00): a in {13} | same | (reads) | same |
| L16 up c864 | 1% / 1% | **a** (R2 1.00): a in {70} | same | (reads) | same |
| L16 down c917 | 0 / 1% | off (on 0) | **a** (R2 0.94): a in {33} | - | same |
| L16 down c598 | 1% / 1% | **a** (R2 1.00): a in {35} | same | - | same |
| L16 down c720 | 1% / 1% | **a** (R2 1.00): a in {74} | same | - | same |
| L17 down c720 | 1% / 1% | **a** (R2 1.00): a in {34} | same | - | same |
| L17 down c411 | 1% / 1% | **a** (R2 1.00): a in {20} | same | - | same |
| L17 down c475 | 1% / 1% | **a** (R2 1.00): a in {41} | same | - | same |
| L17 down c490 | 1% / 1% | **a** (R2 1.00): a in {80} | same | - | same |
| L17 down c864 | 1% / 1% | **a** (R2 1.00): a in {21} | same | - | same |
| L17 down c731 | 1% / 1% | **a** (R2 1.00): a in {49} | same | - | same |
| L17 down c779 | 1% / 1% | **a** (R2 1.00): a in {33} | same | - | same |
| L17 down c270 | 1% / 1% | **a** (R2 1.00): a in {51} | same | - | same |
| L17 down c331 | 1% / 1% | **a** (R2 1.00): a in {24} | same | - | same |
| L17 down c12 | 1% / 1% | **a** (R2 1.00): a in {36} | same | - | same |
| L17 down c249 | 1% / 1% | **a** (R2 1.00): a in {54} | same | - | same |
| L17 down c167 | 1% / 1% | **a** (R2 0.97): a in {4} | **a** (R2 1.00): a in {4} | - | same |
| L17 up c865 | 1% / 1% | **a** (R2 1.00): a in {99} | same | (reads) | same |
| L17 gate c736 | 1% / 1% | **a** (R2 1.00): a in {12} | same | (reads) | same |
| L17 up c66 | 1% / 1% | **a** (R2 1.00): a in {14} | same | (reads) | same |
| L17 gate c570 | 1% / 1% | **a** (R2 1.00): a in {13} | same | (reads) | same |
| L17 down c857 | 1% / 1% | **a** (R2 1.00): a in {74} | same | - | same |
| L17 gate c321 | 1% / 1% | **a** (R2 1.00): a in {45} | same | (reads) | same |
| L17 gate c169 | 1% / 1% | **a** (R2 1.00): a in {12} | same | (reads) | same |
| L17 gate c864 | 1% / 1% | **a** (R2 1.00): a in {74} | same | (reads) | same |
| L17 up c195 | 1% / 1% | **a** (R2 1.00): a in {74} | same | (reads) | same |
| L17 up c245 | 1% / 1% | **a** (R2 1.00): a in {14} | same | (reads) | same |
| L17 up c331 | 1% / 1% | **a** (R2 1.00): a in {24} | same | (reads) | same |
| L17 gate c609 | 1% / 1% | **a** (R2 1.00): a in {70} | same | (reads) | same |
| L17 up c498 | 1% / 1% | **a** (R2 1.00): a in {42} | same | (reads) | same |
| L18 v c7 (kv7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {61} | (reads) | same |
| L18 up c274 | 1% / 1% | **a** (R2 1.00): a in {74} | same | (reads) | same |
| L18 v c39 (kv7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {18} | (reads) | same |
| L18 v c243 (kv7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {80} | (reads) | same |
| L18 v c207 (kv7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {24} | (reads) | same |
| L18 down c81 | 1% / 1% | **a** (R2 1.00): a in {38} | same | - | same |
| L18 v c43 (kv4) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {74} | (reads) | same |
| L18 v c126 (kv7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {52} | (reads) | same |
| L18 v c148 (kv7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {55} | (reads) | same |
| L18 v c165 (kv7) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {62} | (reads) | same |
| L18 v c183 (kv1) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {70} | (reads) | same |
| L18 v c196 (kv4) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {70} | (reads) | same |
| L18 down c8 | 1% / 1% | **a** (R2 1.00): a in {1} | same | - | same |
| L18 gate c281 | 1% / 1% | **a** (R2 1.00): a in {33} | same | (reads) | same |
| L18 down c420 | 1% / 1% | **a** (R2 1.00): a in {60} | same | - | same |
| L18 down c1011 | 1% / 1% | **a** (R2 1.00): a in {38} | same | - | same |
| L18 gate c94 | 1% / 1% | **a** (R2 1.00): a in {33} | same | (reads) | same |
| L18 gate c875 | 1% / 1% | **a** (R2 1.00): a in {65} | same | (reads) | same |
| L19 gate c305 | 1% / 1% | **a** (R2 1.00): a in {80} | same | (reads) | same |
| L19 down c360 | 1% / 1% | **a** (R2 1.00): a in {50} | same | - | same |
| L19 gate c572 | 1% / 1% | **a** (R2 1.00): a in {75} | same | (reads) | same |
| L19 gate c627 | 1% / 1% | **a** (R2 1.00): a in {28} | same | (reads) | same |
| L19 down c895 | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {15} | - | same |
| L20 v c174 (kv0) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {70} | (reads) | same |
| L20 v c181 (kv0) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {98} | (reads) | same |
| L20 v c232 (kv0) | 0 / 1% | off (on 0) | **a** (R2 0.86): a in {54} | (reads) | same |
| L20 gate c348 | 1% / 1% | **a** (R2 1.00): a in {7} | same | (reads) | same |
| L20 down c870 | 1% / 1% | **a** (R2 1.00): a in {9} | same | - | same |
| L20 down c839 | 1% / 1% | **a** (R2 1.00): a in {1} | same | - | same |
| L20 down c605 | 1% / 1% | **a** (R2 1.00): a in {10} | same | - | same |
| L20 down c710 | 1% / 1% | **a** (R2 1.00): a in {40} | same | - | same |
| L20 down c520 | 1% / 1% | **a** (R2 1.00): a in {18} | same | - | same |
| L20 down c322 | 1% / 1% | **a** (R2 1.00): a in {7} | same | - | same |
| L20 down c247 | 1% / 1% | **a** (R2 1.00): a in {27} | same | - | same |
| L20 down c149 | 1% / 1% | **a** (R2 1.00): a in {98} | same | - | same |
| L20 gate c642 | 1% / 1% | **a** (R2 1.00): a in {45} | same | (reads) | same |
| L20 gate c618 | 1% / 1% | **a** (R2 0.66): a in {27} | **a** (R2 1.00): a in {27} | (reads) | same |
| L20 up c839 | 1% / 1% | **a** (R2 1.00): a in {1} | same | (reads) | same |
| L20 v c237 (kv0) | 0 / 1% | off (on 0) | **a** (R2 0.99): a in {45} | (reads) | same |
| L20 gate c50 | 1% / 1% | **a** (R2 1.00): a in {10} | same | (reads) | same |
| L20 down c986 | 1% / 1% | **a** (R2 1.00): a in {14} | same | - | same |
| L20 down c915 | 1% / 1% | **a** (R2 1.00): a in {9} | same | - | same |
| L20 v c129 (kv0) | 0 / 1% | off (on 0) | **a** (R2 0.98): a in {45} | (reads) | same |
| L20 v c173 (kv0) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {98} | (reads) | same |
| L20 v c164 (kv0) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {54} | (reads) | same |
| L20 v c158 (kv0) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {42} | (reads) | same |
| L21 up c740 | 1% / 1% | **a** (R2 1.00): a in {51} | same | (reads) | same |
| L21 up c426 | 1% / 1% | **a** (R2 1.00): a in {27} | same | (reads) | same |
| L21 up c47 | 1% / 1% | **a** (R2 1.00): a in {4} | same | (reads) | same |
| L21 gate c364 | 1% / 1% | **a** (R2 1.00): a in {4} | same | (reads) | same |
| L21 gate c109 | 1% / 1% | **a** (R2 1.00): a in {99} | same | (reads) | same |
| L21 gate c82 | 1% / 1% | **a** (R2 1.00): a in {20} | same | (reads) | same |
| L21 gate c203 | 1% / 1% | **a** (R2 1.00): a in {29} | same | (reads) | same |
| L21 gate c152 | 1% / 1% | **a** (R2 1.00): a in {45} | same | (reads) | same |
| L21 up c236 | 1% / 1% | **a** (R2 1.00): a in {10} | same | (reads) | same |
| L21 up c347 | 1% / 1% | **a** (R2 1.00): a in {99} | same | (reads) | same |
| L21 up c76 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L21 up c128 | 1% / 1% | **a** (R2 1.00): a in {11} | same | (reads) | same |
| L21 down c515 | 1% / 1% | **a** (R2 1.00): a in {74} | same | - | same |
| L21 down c432 | 1% / 1% | **a** (R2 1.00): a in {28} | same | - | same |
| L21 down c166 | 1% / 1% | **a** (R2 1.00): a in {52} | same | - | same |
| L21 down c298 | 1% / 1% | **a** (R2 1.00): a in {18} | same | - | same |
| L21 down c312 | 1% / 1% | **a** (R2 1.00): a in {31} | same | - | same |
| L21 down c346 | 1% / 1% | **a** (R2 1.00): a in {70} | same | - | same |
| L21 down c426 | 1% / 1% | **a** (R2 1.00): a in {27} | same | - | same |
| L21 down c430 | 1% / 1% | **a** (R2 1.00): a in {88} | same | - | same |
| L21 down c44 | 1% / 1% | **a** (R2 1.00): a in {36} | same | - | same |
| L21 down c86 | 1% / 1% | **a** (R2 1.00): a in {52} | same | - | same |
| L22 v c75 (kv3) | 1% / 0 | **a** (R2 1.00): a in {4} | off (on 0) | (reads) | same |
| L22 v c52 (kv3) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {31} | (reads) | same |
| L22 down c197 | 1% / 1% | **a** (R2 1.00): a in {1} | same | - | same |
| L22 down c264 | 1% / 1% | **a** (R2 1.00): a in {80} | same | - | same |
| L22 v c177 (kv3) | 1% / 0 | **a** (R2 1.00): a in {4} | off (on 0) | (reads) | same |
| L22 v c34 (kv3) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {15} | (reads) | same |
| L22 down c54 | 1% / 1% | **a** (R2 1.00): a in {35} | same | - | same |
| L22 v c203 (kv3) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {80} | (reads) | same |
| L22 up c229 | 1% / 1% | **a** (R2 1.00): a in {35} | same | (reads) | same |
| L22 k c27 (kv7) | 1% / 1% | **a** (R2 1.00): a in {93} | same | (reads) | same |
| L22 gate c418 | 1% / 1% | **a** (R2 1.00): a in {60} | same | (reads) | same |
| L22 up c120 | 1% / 1% | **a** (R2 1.00): a in {74} | same | (reads) | same |
| L22 down c495 | 1% / 1% | **a** (R2 1.00): a in {60} | same | - | same |
| L22 gate c242 | 1% / 1% | **a** (R2 1.00): a in {3} | same | (reads) | same |
| L22 v c122 (kv3) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {24} | (reads) | same |
| L22 v c213 (kv3) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {90} | (reads) | same |
| L22 down c162 | 1% / 1% | **a** (R2 1.00): a in {2} | same | - | same |
| L23 down c932 | 1% / 1% | **a** (R2 1.00): a in {30} | same | - | same |
| L23 v c231 (kv1) | 1% / 1% | **a** (R2 1.00): a in {88} | same | (reads) | same |
| L23 v c236 (kv1) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {58} | (reads) | same |
| L23 v c33 (kv3) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {88} | (reads) | same |
| L23 gate c350 | 1% / 1% | **a** (R2 1.00): a in {35} | same | (reads) | same |
| L23 gate c22 | 1% / 1% | **a** (R2 1.00): a in {1} | same | (reads) | same |
| L23 down c975 | 1% / 1% | **a** (R2 1.00): a in {2} | same | - | same |
| L23 v c151 (kv1) | 1% / 1% | **a** (R2 1.00): a in {88} | **a** (R2 0.99): a in {88} | (reads) | same |
| L23 v c61 (kv1) | 1% / 1% | **a** (R2 1.00): a in {98} | **a** (R2 0.95): a in {98} | (reads) | same |
| L23 v c175 (kv1) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {58} | (reads) | same |
| L23 down c81 | 1% / 1% | **a** (R2 1.00): a in {5} | same | - | same |
| L23 v c230 (kv1) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {58} | (reads) | same |
| L23 v c221 (kv1) | 1% / 1% | **a** (R2 1.00): a in {88} | same | (reads) | same |
| L23 v c154 (kv1) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {88} | (reads) | same |
| L23 v c164 (kv1) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {98} | (reads) | same |
| L23 v c96 (kv1) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {88} | (reads) | same |
| L23 down c292 | 1% / 1% | **a** (R2 1.00): a in {90} | same | - | same |
| L24 down c109 | 1% / 1% | **a** (R2 1.00): a in {99} | same | - | same |
| L24 down c208 | 1% / 1% | **a** (R2 1.00): a in {52} | same | - | same |
| L24 down c97 | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {52} | - | same |
| L24 down c292 | 1% / 1% | **a** (R2 1.00): a in {90} | **a** (R2 0.95): a in {90} | - | same |
| L24 gate c155 | 1% / 1% | **a** (R2 1.00): a in {88} | same | (reads) | same |
| L24 gate c859 | 1% / 1% | **a** (R2 1.00): a in {55} | same | (reads) | same |
| L24 up c155 | 1% / 1% | **a** (R2 1.00): a in {99} | same | (reads) | same |
| L24 v c219 (kv4) | 1% / 1% | **a** (R2 1.00): a in {90} | same | (reads) | same |
| L24 v c203 (kv5) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {63} | (reads) | same |
| L24 v c225 (kv5) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {24} | (reads) | same |
| L24 v c155 (kv5) | 0 / 1% | off (on 0) | **a** (R2 1.00): a in {24} | (reads) | same |
| L25 gate c360 | 1% / 1% | **a** (R2 1.00): a in {1} | same | (reads) | same |
| L25 gate c176 | 1% / 1% | **a** (R2 1.00): a in {99} | same | (reads) | same |
| L25 down c458 | 1% / 1% | **a** (R2 1.00): a in {50} | same | - | same |
| L25 down c592 | 1% / 1% | **a** (R2 1.00): a in {95} | same | - | same |
| L25 up c101 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L25 gate c258 | 1% / 1% | **a** (R2 1.00): a in {99} | same | (reads) | same |
| L25 gate c363 | 1% / 1% | **a** (R2 1.00): a in {5} | same | (reads) | same |
| L25 up c156 | 1% / 1% | **a** (R2 1.00): a in {64} | same | (reads) | same |
| L25 down c369 | 1% / 1% | **a** (R2 1.00): a in {31} | same | - | same |
| L25 down c101 | 1% / 1% | **a** (R2 1.00): a in {52} | same | - | same |
| L26 up c134 | 1% / 1% | **a** (R2 0.99): a in {32} | **a** (R2 1.00): a in {32} | (reads) | same |
| L26 gate c70 | 1% / 1% | **a** (R2 1.00): a in {99} | same | (reads) | same |
| L26 gate c586 | 1% / 1% | **a** (R2 1.00): a in {38} | same | (reads) | same |
| L26 down c582 | 1% / 1% | **a** (R2 1.00): a in {74} | same | - | same |
| L26 down c210 | 1% / 1% | **a** (R2 1.00): a in {50} | same | - | same |
| L26 down c249 | 1% / 1% | **a** (R2 1.00): a in {55} | same | - | same |
| L26 down c117 | 1% / 1% | **a** (R2 1.00): a in {52} | same | - | same |
| L26 down c574 | 1% / 1% | **a** (R2 1.00): a in {31} | same | - | same |
| L26 gate c45 | 1% / 1% | **a** (R2 1.00): a in {60} | same | (reads) | same |
| L26 gate c454 | 1% / 1% | **a** (R2 1.00): a in {1} | same | (reads) | same |
| L27 gate c779 | 1% / 1% | **a** (R2 1.00): a in {4} | same | (reads) | same |
| L27 gate c367 | 1% / 1% | **a** (R2 1.00): a in {74} | same | (reads) | same |
| L27 gate c273 | 1% / 1% | **a** (R2 1.00): a in {1} | same | (reads) | same |
| L27 down c923 | 1% / 1% | **a** (R2 1.00): a in {99} | same | - | same |
| L27 gate c414 | 1% / 1% | **a** (R2 1.00): a in {99} | same | (reads) | same |
| L27 up c760 | 1% / 1% | **a** (R2 1.00): a in {5} | same | (reads) | same |
| L27 gate c484 | 1% / 1% | **a** (R2 1.00): a in {74} | same | (reads) | same |
| L27 down c131 | 1% / 1% | **a** (R2 1.00): a in {42} | same | - | same |
| L27 up c277 | 1% / 1% | **a** (R2 1.00): a in {60} | same | (reads) | same |
| L27 down c852 | 1% / 1% | **a** (R2 1.00): a in {2} | same | - | same |
| L27 down c273 | 1% / 1% | **a** (R2 1.00): a in {1} | same | - | same |
| L27 down c559 | 1% / 1% | **a** (R2 1.00): a in {74} | same | - | same |
| L27 down c199 | 1% / 1% | **a** (R2 1.00): a in {80} | same | - | same |
| L27 down c779 | 1% / 1% | **a** (R2 1.00): a in {4} | same | - | same |
| L28 gate c501 | 1% / 1% | **a** (R2 1.00): a in {55} | same | (reads) | same |
| L28 gate c432 | 1% / 1% | **a** (R2 1.00): a in {48} | same | (reads) | same |
| L28 gate c467 | 1% / 1% | **a** (R2 1.00): a in {1} | same | (reads) | same |
| L28 up c792 | 1% / 1% | **a** (R2 1.00): a in {3} | same | (reads) | same |
| L28 gate c839 | 1% / 1% | **a** (R2 1.00): a in {99} | same | (reads) | same |
| L28 gate c669 | 1% / 1% | **a** (R2 1.00): a in {80} | same | (reads) | same |
| L28 gate c502 | 1% / 1% | **a** (R2 1.00): a in {24} | same | (reads) | same |
| L28 down c6 | 1% / 1% | **a** (R2 1.00): a in {1} | same | - | same |
| L28 gate c29 | 1% / 1% | **a** (R2 1.00): a in {1} | same | (reads) | same |
| L28 down c497 | 1% / 1% | **a** (R2 1.00): a in {1} | same | - | same |
| L28 q c109 (H17) | 1% / 1% | **a** (R2 0.99): a in {8} | **a** (R2 1.00): a in {8} | (reads) | same |
| L28 gate c279 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L29 up c317 | 1% / 1% | **a** (R2 1.00): a in {93} | same | (reads) | same |
| L29 up c350 | 1% / 1% | **a** (R2 1.00): a in {1} | same | (reads) | same |
| L29 up c689 | 1% / 1% | **a** (R2 1.00): a in {74} | same | (reads) | same |
| L29 gate c877 | 1% / 1% | **a** (R2 1.00): a in {90} | same | (reads) | same |
| L29 up c200 | 1% / 1% | **a** (R2 1.00): a in {74} | same | (reads) | same |
| L29 gate c367 | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L29 up c421 | 1% / 1% | **a** (R2 1.00): a in {99} | same | (reads) | same |
| L29 down c991 | 1% / 1% | **a** (R2 1.00): a in {99} | same | - | same |
| L29 down c108 | 1% / 1% | **a** (R2 1.00): a in {74} | **a** (R2 0.99): a in {74} | - | same |
| L29 down c603 | 1% / 1% | **a** (R2 1.00): a in {90} | same | - | same |
| L29 down c920 | 1% / 1% | **a** (R2 1.00): a in {45} | same | - | same |
| L29 gate c267 | 1% / 1% | **a** (R2 1.00): a in {99} | same | (reads) | same |
| L29 gate c164 | 1% / 1% | **a** (R2 1.00): a in {64} | same | (reads) | same |
| L29 gate c406 | 1% / 1% | **a** (R2 1.00): a in {80} | same | (reads) | same |
| L30 gate c902 | 1% / 1% | **a** (R2 1.00): a in {1} | same | (reads) | same |
| L30 down c65 | 1% / 1% | **a** (R2 1.00): a in {1} | same | - | same |
| L30 up c686 | 1% / 1% | **a** (R2 1.00): a in {1} | same | (reads) | same |
| L30 down c260 | 1% / 1% | **a** (R2 1.00): a in {2} | same | - | same |
| L30 down c114 | 1% / 1% | **a** (R2 1.00): a in {1} | same | - | same |
| L30 down c968 | 1% / 1% | **a** (R2 1.00): a in {74} | same | - | same |
| L30 gate c497 | 1% / 1% | **a** (R2 1.00): a in {1} | same | (reads) | same |
| L30 gate c667 | 1% / 1% | **a** (R2 0.97): a in {99} | **a** (R2 1.00): a in {99} | (reads) | same |
| L30 down c274 | 1% / 1% | **a** (R2 1.00): a in {99} | same | - | same |
| L31 v c58 (kv5) | 1% / 1% | **a** (R2 1.00): a in {88} | same | (reads) | same |
| L31 v c143 (kv5) | 1% / 1% | **a** (R2 1.00): a in {52} | same | (reads) | same |
| L31 v c165 (kv5) | 1% / 1% | **a** (R2 1.00): a in {74} | same | (reads) | same |
| L31 v c120 (kv5) | 1% / 1% | **a** (R2 1.00): a in {88} | same | (reads) | same |

</details>

<details><summary>tens-digit set: 130 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L0 down c6 | 50% / 50% | **a//10** (R2 0.96): (tens) a in {51..100} | same | a: mod100 +13%, mod50 +7%, mod25 +3%, mod20 +3% | a: mod100 +13%, mod50 +7%, mod25 +3%, mod20 +3% |
| L0 down c22 | 19% / 19% | **a//10** (R2 0.94): (tens) a in {61..79} | same | a: mod100 +2%, mod50 +3% | a: mod100 +2%, mod50 +3% |
| L0 down c23 | 39% / 39% | **a//10** (R2 1.00): (tens) a in {1..39} | **a//10** (R2 0.99): (tens) a in {1..39} | a: mod100 +7%, mod50 +5% | a: mod100 +7%, mod50 +5% |
| L0 down c28 | 32% / 32% | **a//10** (R2 0.90): (tens) a in {1..32} | same | - | same |
| L0 down c123 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | - | same |
| L0 down c252 | 9% / 9% | **a//10** (R2 1.00): (tens) a in {1..9} | same | a: mod50 +2%, mod25 +3%, mod20 +3% | a: mod50 +2%, mod25 +3%, mod20 +3% |
| L0 down c405 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | - | same |
| L0 gate c262 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | (reads) | same |
| L0 gate c406 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | (reads) | same |
| L0 gate c697 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | (reads) | same |
| L0 gate c764 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | (reads) | same |
| L0 up c123 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | (reads) | same |
| L0 up c130 | 46% / 46% | **a//10** (R2 0.90): (tens) a in {54..99} | same | (reads) | same |
| L0 up c216 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | (reads) | same |
| L0 up c265 | 27% / 28% | **a** (R2 1.00): a in {10..36} | **a//10** (R2 0.92): (tens) a in {10..37} | (reads) | same |
| L0 v c14 (kv0) | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | (reads) | same |
| L1 down c96 | 27% / 27% | **a//10** (R2 0.92): (tens) a in {1..27} | same | a: mod100 +4%, mod50 +3% | a: mod100 +4%, mod50 +3% |
| L1 down c212 | 19% / 19% | **a//10** (R2 0.94): (tens) a in {61..79} | same | - | same |
| L1 gate c159 | 9% / 9% | **a//10** (R2 1.00): (tens) a in {1..9} | same | (reads) | same |
| L1 up c42 | 27% / 27% | **a//10** (R2 0.92): (tens) a in {1..27} | same | (reads) | same |
| L1 up c212 | 19% / 19% | **a//10** (R2 0.94): (tens) a in {61..79} | same | (reads) | same |
| L1 v c75 (kv2) | 89% / 89% | **a//10** (R2 1.00): (tens) a in {1..89} | same | (reads) | same |
| L1 v c86 (kv6) | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | (reads) | same |
| L2 gate c28 | 87% / 88% | **a** (R2 0.98): a in {1..66, 68..69, 71..89} | **a//10** (R2 0.91): (tens) a in {1..69, 71..89} | (reads) | same |
| L2 down c92 | 31% / 31% | **a//10** (R2 1.00): (tens) a in {70..100} | same | a: mod100 +4%, mod50 +3% | a: mod100 +4%, mod50 +3% |
| L2 down c111 | 10% / 10% | **a//10** (R2 1.00): (tens) a in {10..19} | same | - | same |
| L2 v c194 (kv6) | 30% / 31% | **a//10** (R2 1.00): (tens) a in {20..49} | **a//10** (R2 0.95): (tens) a in {20..50} | (reads) | same |
| L2 v c160 (kv2) | 10% / 10% | **a//10** (R2 1.00): (tens) a in {50..59} | **a//10** (R2 0.94): (tens) a in {50, 52..59} | (reads) | same |
| L2 up c330 | 9% / 9% | **a//10** (R2 1.00): (tens) a in {1..9} | same | (reads) | same |
| L2 up c271 | 11% / 10% | **a//10** (R2 0.90): (tens) a in {50..59, 62} | **a** (R2 0.99): a in {51..59, 62} | (reads) | same |
| L2 up c90 | 10% / 10% | **a//10** (R2 1.00): (tens) a in {20..29} | same | (reads) | same |
| L3 down c100 | 9% / 9% | **a//10** (R2 1.00): (tens) a in {1..9} | same | a: mod20 +2% | a: mod20 +2% |
| L3 v c247 (kv1) | 48% / 48% | **a//10** (R2 0.92): (tens) a in {51, 54..100} | same | (reads) | same |
| L3 v c48 (kv1) | 21% / 23% | **a//10** (R2 1.00): (tens) a in {80..100} | **a//10** (R2 0.91): (tens) a in {78..100} | (reads) | same |
| L3 v c28 (kv1) | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | (reads) | same |
| L3 v c26 (kv1) | 20% / 20% | **a//10** (R2 0.94): (tens) a in {1..20} | same | (reads) | same |
| L3 v c12 (kv1) | 68% / 68% | **a//10** (R2 0.93): (tens) a in {32..99} | same | (reads) | same |
| L3 up c122 | 11% / 11% | **a//10** (R2 1.00): (tens) a in {90..100} | same | (reads) | same |
| L3 gate c736 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | (reads) | same |
| L3 gate c177 | 10% / 10% | **a//10** (R2 1.00): (tens) a in {20..29} | same | (reads) | same |
| L3 down c177 | 10% / 10% | **a//10** (R2 1.00): (tens) a in {20..29} | same | - | same |
| L3 down c126 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | - | same |
| L3 down c122 | 11% / 12% | **a//10** (R2 0.99): (tens) a in {90..100} | **a//10** (R2 0.92): (tens) a in {88, 90..100} | - | same |
| L4 up c313 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | (reads) | same |
| L4 down c127 | 9% / 9% | **a//10** (R2 1.00): (tens) a in {1..9} | same | - | same |
| L4 down c808 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | - | same |
| L4 gate c211 | 19% / 19% | **a//10** (R2 0.94): (tens) a in {80..98} | same | (reads) | same |
| L5 down c387 | 10% / 10% | **a//10** (R2 1.00): (tens) a in {90..99} | same | - | same |
| L5 v c11 (kv5) | 25% / 17% | **a//10** (R2 0.80): (tens) a in {53..54, 56..79} | unexplained (best R2 0.48) | (reads) | same |
| L5 down c936 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | - | same |
| L5 v c13 (kv5) | 27% / 16% | **a//10** (R2 0.87): (tens) a in {74..100} | unexplained (best R2 0.48) | (reads) | same |
| L6 gate c133 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | (reads) | same |
| L10 up c138 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | (reads) | same |
| L11 up c493 | 9% / 9% | **a//10** (R2 0.97): (tens) a in {1..9} | **a//10** (R2 1.00): (tens) a in {1..9} | (reads) | same |
| L11 down c294 | 6% / 9% | **a** (R2 0.99): a in {1..6} | **a//10** (R2 0.99): (tens) a in {1..9} | - | same |
| L12 gate c256 | 10% / 10% | **a//10** (R2 1.00): (tens) a in {40..49} | same | (reads) | same |
| L12 down c154 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | - | same |
| L12 down c59 | 11% / 11% | **a//10** (R2 0.91): (tens) a in {60..70} | same | - | same |
| L13 down c733 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | - | same |
| L14 gate c34 | 20% / 20% | **a//10** (R2 0.94): (tens) a in {80, 82..100} | same | (reads) | same |
| L14 up c255 | 11% / 11% | **a//10** (R2 1.00): (tens) a in {90..100} | same | (reads) | same |
| L14 down c8 | 46% / 46% | **a//10** (R2 0.92): (tens) a in {1..40, 42, 44..48} | same | a: mod100 +7%, mod50 +2% | a: mod100 +7%, mod50 +2% |
| L14 gate c8 | 46% / 46% | **a//10** (R2 0.92): (tens) a in {1..40, 42, 44..48} | same | (reads) | same |
| L15 gate c36 | 9% / 9% | **a//10** (R2 1.00): (tens) a in {1..9} | same | (reads) | same |
| L15 gate c237 | 21% / 21% | **a//10** (R2 0.90): (tens) a in {1..20, 25} | same | (reads) | same |
| L15 gate c244 | 11% / 11% | **a//10** (R2 0.91): (tens) a in {69..79} | same | (reads) | same |
| L15 down c36 | 9% / 9% | **a//10** (R2 1.00): (tens) a in {1..9} | same | - | same |
| L15 down c193 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | - | same |
| L16 v c192 (kv0) | 28% / 27% | **a//10** (R2 0.90): (tens) a in {40..67} | **a//10** (R2 0.90): (tens) a in {40..66} | (reads) | same |
| L16 up c3 | 30% / 30% | **a//10** (R2 0.96): (tens) a in {70, 72..100} | same | (reads) | same |
| L16 down c9 | 20% / 20% | **a//10** (R2 0.94): (tens) a in {80, 82..100} | same | a: mod100 +4%, mod50 +3% | a: mod100 +4%, mod50 +3% |
| L16 v c163 (kv0) | 10% / 10% | **a//10** (R2 0.89): (tens) a in {1..10} | same | (reads) | same |
| L17 up c110 | 11% / 11% | **a//10** (R2 1.00): (tens) a in {90..100} | same | (reads) | same |
| L17 gate c261 | 31% / 31% | **a//10** (R2 0.93): (tens) a in {1..31} | same | (reads) | same |
| L17 down c58 | 21% / 21% | **a//10** (R2 0.90): (tens) a in {1..21} | same | a: mod100 +2%, mod50 +2% | a: mod100 +2%, mod50 +2% |
| L17 gate c497 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | (reads) | same |
| L18 v c25 (kv7) | 5% / 11% | **a** (R2 0.78): a in {64..67} | **a//10** (R2 0.91): (tens) a in {56, 60..69} | (reads) | same |
| L18 v c38 (kv7) | 26% / 28% | **a//10** (R2 0.86): (tens) a in {2..26} | **a//10** (R2 0.96): (tens) a in {2..29} | (reads) | same |
| L19 gate c93 | 9% / 9% | **a//10** (R2 1.00): (tens) a in {1..9} | same | (reads) | same |
| L19 gate c78 | 27% / 27% | **a//10** (R2 0.92): (tens) a in {3..29} | same | (reads) | same |
| L19 up c52 | 10% / 10% | **a//10** (R2 1.00): (tens) a in {10..19} | same | (reads) | same |
| L19 down c700 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | - | same |
| L19 up c10 | 45% / 45% | **a//10** (R2 0.90): (tens) a in {56..100} | same | (reads) | same |
| L20 down c23 | 50% / 50% | **a//10** (R2 0.96): (tens) a in {1..50} | same | a: mod100 +4% | a: mod100 +4% |
| L20 up c46 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | (reads) | same |
| L20 gate c7 | 50% / 50% | **a//10** (R2 0.90): (tens) a in {2..50, 52} | same | (reads) | same |
| L20 down c33 | 22% / 22% | **a//10** (R2 0.94): (tens) a in {78, 80..100} | same | - | same |
| L20 down c81 | 11% / 11% | **a//10** (R2 0.91): (tens) a in {19..29} | same | - | same |
| L20 gate c46 | 11% / 11% | **a//10** (R2 0.91): (tens) a in {40..50} | same | (reads) | same |
| L20 gate c66 | 18% / 18% | **a//10** (R2 0.90): (tens) a in {82..99} | **a** (R2 1.00): a in {82..99} | (reads) | same |
| L21 up c58 | 31% / 31% | **a//10** (R2 0.93): (tens) a in {1..31} | same | (reads) | same |
| L21 down c58 | 21% / 21% | **a//10** (R2 0.90): (tens) a in {1..21} | same | - | same |
| L21 down c141 | 11% / 11% | **a//10** (R2 0.91): (tens) a in {89..99} | **a//10** (R2 0.89): (tens) a in {89..99} | - | same |
| L21 up c39 | 11% / 11% | **a//10** (R2 0.91): (tens) a in {19..29} | same | (reads) | same |
| L22 gate c268 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | (reads) | same |
| L22 down c14 | 32% / 32% | **a//10** (R2 0.90): (tens) a in {1..31, 34} | same | a: mod100 +2% | a: mod100 +2% |
| L22 v c37 (kv3) | 7% / 10% | **a** (R2 0.91): a in {82..88} | **a//10** (R2 1.00): (tens) a in {80..89} | (reads) | same |
| L22 up c338 | 31% / 31% | **a//10** (R2 0.93): (tens) a in {1..31} | same | (reads) | same |
| L22 gate c328 | 31% / 31% | **a//10** (R2 0.93): (tens) a in {1..31} | same | (reads) | same |
| L22 v c63 (kv3) | 0 / 1% | off (on 0) | **a//10** (R2 1.00): (tens) a in {100} | (reads) | same |
| L22 down c268 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | - | same |
| L22 down c416 | 30% / 30% | **a//10** (R2 0.91): (tens) a in {69..70, 72..99} | same | - | same |
| L23 down c322 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | - | same |
| L23 down c126 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | - | same |
| L23 gate c76 | 10% / 10% | **a//10** (R2 1.00): (tens) a in {90..99} | same | (reads) | same |
| L23 up c114 | 42% / 42% | **a//10** (R2 0.91): (tens) a in {1..40, 42, 49} | same | (reads) | same |
| L23 down c1 | 69% / 69% | **a//10** (R2 0.93): (tens) a in {32..100} | same | a: mod100 +4% | a: mod100 +4% |
| L23 down c533 | 19% / 19% | **a//10** (R2 0.94): (tens) a in {80, 82..99} | same | - | same |
| L24 gate c174 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | (reads) | same |
| L24 up c450 | 49% / 49% | **a//10** (R2 0.93): (tens) a in {2..50} | same | (reads) | same |
| L25 up c384 | 33% / 33% | **a//10** (R2 0.93): (tens) a in {67..68, 70..100} | same | (reads) | same |
| L25 down c638 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | - | same |
| L25 down c35 | 31% / 31% | **a//10** (R2 1.00): (tens) a in {70..100} | same | - | same |
| L26 down c766 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | - | same |
| L26 gate c766 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | (reads) | same |
| L26 up c570 | 10% / 10% | **a//10** (R2 1.00): (tens) a in {90..99} | same | (reads) | same |
| L26 up c562 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | (reads) | same |
| L26 down c69 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | - | same |
| L26 down c286 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | - | same |
| L27 up c862 | 9% / 9% | **a//10** (R2 1.00): (tens) a in {1..9} | same | (reads) | same |
| L27 down c336 | 10% / 10% | **a//10** (R2 1.00): (tens) a in {90..99} | same | - | same |
| L28 gate c627 | 9% / 9% | **a//10** (R2 1.00): (tens) a in {1..9} | same | (reads) | same |
| L28 gate c385 | 19% / 19% | **a//10** (R2 0.94): (tens) a in {80, 82..99} | same | (reads) | same |
| L29 gate c114 | 21% / 21% | **a//10** (R2 0.90): (tens) a in {1..21} | same | (reads) | same |
| L29 down c332 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | - | same |
| L30 down c476 | 28% / 28% | **a//10** (R2 0.92): (tens) a in {32..59} | same | - | same |
| L30 down c925 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | **a//10** (R2 0.98): (tens) a in {100} | - | same |
| L30 down c63 | 52% / 51% | **a//10** (R2 0.90): (tens) a in {1..50, 54, 92} | **a//10** (R2 0.94): (tens) a in {1..50, 54} | a: mod100 +8%, mod50 +4%, mod20 +2% | a: mod100 +8%, mod50 +4%, mod20 +2% |
| L30 gate c132 | 31% / 31% | **a//10** (R2 0.93): (tens) a in {1..31} | same | (reads) | same |
| L30 down c405 | 1% / 1% | **a//10** (R2 1.00): (tens) a in {100} | same | - | same |

</details>

<details><summary>unexplained: 11 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L1 k c139 (kv5) | 1% / 1% | unexplained (best R2 0.03) | same | (reads) | same |
| L1 k c142 (kv5) | 2% / 2% | unexplained (best R2 0.03) | unexplained (best R2 0.04) | (reads) | same |
| L5 v c27 (kv5) | 4% / 2% | unexplained (best R2 0.42) | unexplained (best R2 0.17) | (reads) | same |
| L5 v c39 (kv5) | 3% / 5% | unexplained (best R2 0.19) | unexplained (best R2 0.30) | (reads) | same |
| L5 v c109 (kv5) | 1% / 4% | unexplained (best R2 0.14) | unexplained (best R2 0.37) | (reads) | same |
| L5 v c75 (kv5) | 7% / 10% | unexplained (best R2 0.40) | unexplained (best R2 0.29) | (reads) | same |
| L5 v c152 (kv5) | 12% / 15% | unexplained (best R2 0.47) | unexplained (best R2 0.20) | (reads) | same |
| L5 v c165 (kv5) | 3% / 8% | unexplained (best R2 0.19) | unexplained (best R2 0.30) | (reads) | same |
| L5 v c255 (kv5) | 1% / 1% | unexplained (best R2 0.09) | unexplained (best R2 0.11) | (reads) | same |
| L8 k c72 (kv4) | 1% / 23% | unexplained (best R2 0.04) | unexplained (best R2 0.37) | (reads) | same |
| L23 v c52 (kv3) | 3% / 4% | unexplained (best R2 0.35) | unexplained (best R2 0.46) | (reads) | same |

</details>

<details><summary>window (one run): 708 components</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) | writes (add) | writes (sub) |
|---|---|---|---|---|---|
| L0 down c5 | 84% / 84% | **a** (R2 1.00): a in {13..96} | same | a: mod100 +12% | a: mod100 +12% |
| L0 down c30 | 9% / 9% | **a** (R2 1.00): a in {23..31} | same | a: mod50 +3%, mod25 +3%, mod20 +4% | a: mod50 +3%, mod25 +3%, mod20 +4% |
| L0 gate c795 | 2% / 2% | **a** (R2 1.00): a in {68..69} | same | (reads) | same |
| L0 gate c813 | 5% / 5% | **a** (R2 1.00): a in {25..29} | same | (reads) | same |
| L0 up c5 | 74% / 74% | **a** (R2 1.00): a in {26..99} | same | (reads) | same |
| L0 up c6 | 18% / 18% | **a** (R2 1.00): a in {82..99} | same | (reads) | same |
| L0 up c41 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | (reads) | same |
| L0 up c43 | 3% / 3% | **a** (R2 1.00): a in {7..9} | same | (reads) | same |
| L0 up c53 | 2% / 2% | **a** (R2 1.00): a in {13..14} | same | (reads) | same |
| L0 up c72 | 8% / 8% | **a** (R2 1.00): a in {73..80} | same | (reads) | same |
| L0 up c88 | 11% / 11% | **a** (R2 1.00): a in {77..87} | same | (reads) | same |
| L0 up c97 | 7% / 7% | **a** (R2 1.00): a in {85..91} | same | (reads) | same |
| L0 up c101 | 4% / 4% | **a** (R2 1.00): a in {16..19} | same | (reads) | same |
| L0 up c105 | 2% / 2% | **a** (R2 1.00): a in {66..67} | same | (reads) | same |
| L0 up c119 | 7% / 7% | **a** (R2 1.00): a in {35..41} | same | (reads) | same |
| L0 up c121 | 5% / 5% | **a** (R2 1.00): a in {10..14} | same | (reads) | same |
| L0 up c135 | 4% / 4% | **a** (R2 1.00): a in {49..52} | same | (reads) | same |
| L0 up c170 | 5% / 5% | **a** (R2 1.00): a in {41..45} | same | (reads) | same |
| L0 up c171 | 9% / 9% | **a** (R2 1.00): a in {23..31} | same | (reads) | same |
| L0 up c182 | 3% / 3% | **a** (R2 1.00): a in {1..3} | same | (reads) | same |
| L0 up c391 | 5% / 5% | **a** (R2 1.00): a in {13..17} | same | (reads) | same |
| L0 v c1 (kv0) | 24% / 24% | **a** (R2 1.00): a in {1..24} | same | (reads) | same |
| L0 down c170 | 8% / 8% | **a** (R2 1.00): a in {40..47} | same | a: mod20 +2% | a: mod20 +2% |
| L0 down c164 | 7% / 7% | **a** (R2 1.00): a in {29..35} | same | - | same |
| L0 down c253 | 2% / 2% | **a** (R2 1.00): a in {68..69} | same | - | same |
| L0 down c244 | 3% / 3% | **a** (R2 1.00): a in {74..76} | same | - | same |
| L0 down c220 | 2% / 2% | **a** (R2 1.00): a in {30..31} | same | - | same |
| L0 down c213 | 3% / 3% | **a** (R2 1.00): a in {34..36} | same | - | same |
| L0 down c189 | 3% / 3% | **a** (R2 1.00): a in {40..42} | same | - | same |
| L0 down c182 | 3% / 3% | **a** (R2 1.00): a in {1..3} | same | - | same |
| L0 down c313 | 3% / 3% | **a** (R2 1.00): a in {23..25} | same | - | same |
| L0 down c287 | 4% / 4% | **a** (R2 1.00): a in {25..28} | same | - | same |
| L0 down c58 | 6% / 6% | **a** (R2 1.00): a in {95..100} | same | - | same |
| L0 down c48 | 8% / 8% | **a** (R2 1.00): a in {52..59} | same | a: mod50 +2%, mod20 +3% | a: mod50 +2%, mod20 +3% |
| L0 down c60 | 5% / 5% | **a** (R2 1.00): a in {45..49} | same | - | same |
| L0 down c59 | 13% / 13% | **a** (R2 1.00): a in {15..27} | same | a: mod50 +3%, mod25 +2%, mod20 +3% | a: mod50 +3%, mod25 +2%, mod20 +3% |
| L0 down c61 | 2% / 2% | **a** (R2 1.00): a in {86..87} | same | - | same |
| L0 down c63 | 8% / 8% | **a** (R2 1.00): a in {34..41} | same | a: mod20 +3% | a: mod20 +3% |
| L0 down c72 | 7% / 7% | **a** (R2 1.00): a in {75..81} | same | a: mod20 +2% | a: mod20 +2% |
| L0 down c69 | 7% / 7% | **a** (R2 1.00): a in {55..61} | same | a: mod20 +2% | a: mod20 +2% |
| L0 down c101 | 4% / 4% | **a** (R2 1.00): a in {16..19} | same | - | same |
| L0 down c108 | 5% / 5% | **a** (R2 1.00): a in {69..73} | same | - | same |
| L0 down c100 | 4% / 4% | **a** (R2 1.00): a in {90..93} | same | - | same |
| L0 down c94 | 8% / 8% | **a** (R2 1.00): a in {69..76} | same | a: mod25 +2%, mod20 +2% | a: mod25 +2%, mod20 +2% |
| L0 down c75 | 7% / 7% | **a** (R2 1.00): a in {73..79} | same | - | same |
| L0 down c73 | 7% / 7% | **a** (R2 1.00): a in {93..99} | same | a: mod20 +2% | a: mod20 +2% |
| L0 down c88 | 9% / 9% | **a** (R2 1.00): a in {79..87} | same | - | same |
| L0 down c87 | 8% / 8% | **a** (R2 1.00): a in {90..97} | same | a: mod20 +2% | a: mod20 +2% |
| L0 gate c87 | 9% / 9% | **a** (R2 1.00): a in {89..97} | same | (reads) | same |
| L0 gate c77 | 2% / 2% | **a** (R2 1.00): a in {83..84} | same | (reads) | same |
| L0 gate c72 | 4% / 4% | **a** (R2 1.00): a in {75..78} | same | (reads) | same |
| L0 gate c65 | 2% / 2% | **a** (R2 1.00): a in {10..11} | same | (reads) | same |
| L0 gate c59 | 8% / 8% | **a** (R2 1.00): a in {17..24} | same | (reads) | same |
| L0 gate c19 | 88% / 88% | **a** (R2 1.00): a in {13..100} | same | (reads) | same |
| L0 down c148 | 4% / 4% | **a** (R2 1.00): a in {22..25} | same | - | same |
| L0 gate c108 | 7% / 7% | **a** (R2 1.00): a in {69..75} | same | (reads) | same |
| L0 gate c148 | 7% / 7% | **a** (R2 1.00): a in {21..27} | same | (reads) | same |
| L0 gate c167 | 11% / 12% | **a//10** (R2 0.91): (tens) a in {49..59} | **a** (R2 1.00): a in {49..60} | (reads) | same |
| L0 gate c213 | 8% / 8% | **a** (R2 1.00): a in {32..39} | same | (reads) | same |
| L0 gate c237 | 8% / 8% | **a** (R2 1.00): a in {27..34} | same | (reads) | same |
| L0 gate c387 | 2% / 2% | **a** (R2 1.00): a in {46..47} | same | (reads) | same |
| L0 gate c396 | 26% / 26% | **a** (R2 1.00): a in {75..100} | same | (reads) | same |
| L0 gate c456 | 3% / 3% | **a** (R2 1.00): a in {15..17} | same | (reads) | same |
| L0 gate c477 | 5% / 5% | **a** (R2 1.00): a in {45..49} | same | (reads) | same |
| L0 down c31 | 10% / 10% | **a** (R2 1.00): a in {1..10} | same | - | same |
| L0 down c43 | 8% / 8% | **a** (R2 1.00): a in {3..10} | same | a: mod25 +2%, mod20 +2% | a: mod25 +2%, mod20 +2% |
| L0 down c110 | 6% / 6% | **a** (R2 1.00): a in {45..50} | same | - | same |
| L0 down c135 | 4% / 4% | **a** (R2 1.00): a in {50..53} | same | - | same |
| L0 down c141 | 4% / 4% | **a** (R2 1.00): a in {32..35} | same | - | same |
| L0 down c143 | 7% / 7% | **a** (R2 1.00): a in {48..54} | same | a: mod25 +2%, mod20 +2% | a: mod25 +2%, mod20 +2% |
| L1 v c228 (kv5) | 4% / 4% | **a** (R2 1.00): a in {97..100} | same | (reads) | same |
| L1 v c233 (kv6) | 2% / 3% | **a** (R2 0.95): a in {49..50} | **a** (R2 0.76): a in {48..50} | (reads) | same |
| L1 v c21 (kv5) | 11% / 13% | **a//10** (R2 0.89): (tens) a in {49..59} | **a** (R2 0.99): a in {47..59} | (reads) | same |
| L1 v c22 (kv6) | 5% / 5% | **a** (R2 1.00): a in {13..17} | same | (reads) | same |
| L1 v c50 (kv5) | 4% / 4% | **a** (R2 0.83): a in {37..39} | **a** (R2 0.82): a in {37..39} | (reads) | same |
| L1 v c72 (kv6) | 2% / 2% | **a** (R2 1.00): a in {11..12} | same | (reads) | same |
| L1 v c85 (kv6) | 1% / 2% | **a** (R2 0.57): a in {24} | **a** (R2 0.91): a in {24..25} | (reads) | same |
| L1 v c89 (kv6) | 2% / 2% | **a** (R2 1.00): a in {32..33} | same | (reads) | same |
| L1 v c131 (kv6) | 13% / 13% | **a** (R2 1.00): a in {19..31} | **a** (R2 0.98): a in {19..31} | (reads) | same |
| L1 v c138 (kv6) | 11% / 14% | **a** (R2 0.96): a in {26..36} | **a** (R2 0.97): a in {24..37} | (reads) | same |
| L1 v c146 (kv5) | 5% / 5% | **a** (R2 0.99): a in {45..49} | **a** (R2 1.00): a in {45..49} | (reads) | same |
| L1 v c156 (kv6) | 4% / 4% | **a** (R2 1.00): a in {20..23} | **a** (R2 0.97): a in {20..23} | (reads) | same |
| L1 v c171 (kv6) | 25% / 27% | **a** (R2 0.98): a in {4..28} | **a** (R2 1.00): a in {4..30} | (reads) | same |
| L1 v c215 (kv5) | 8% / 8% | **a** (R2 0.99): a in {57..64} | **a** (R2 1.00): a in {57..64} | (reads) | same |
| L1 v c217 (kv6) | 6% / 6% | **a** (R2 1.00): a in {8..13} | same | (reads) | same |
| L1 v c221 (kv6) | 5% / 5% | **a** (R2 1.00): a in {40..44} | same | (reads) | same |
| L1 gate c218 | 3% / 3% | **a** (R2 1.00): a in {41..43} | same | (reads) | same |
| L1 gate c210 | 3% / 3% | **a** (R2 1.00): a in {87..89} | same | (reads) | same |
| L1 gate c205 | 2% / 2% | **a** (R2 1.00): a in {82..83} | same | (reads) | same |
| L1 gate c108 | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | (reads) | same |
| L1 gate c69 | 14% / 15% | **a** (R2 1.00): a in {86..99} | **a** (R2 0.99): a in {85..99} | (reads) | same |
| L1 down c885 | 4% / 4% | **a** (R2 1.00): a in {75..78} | same | - | same |
| L1 down c631 | 4% / 4% | **a** (R2 1.00): a in {60..63} | same | - | same |
| L1 down c264 | 9% / 9% | **a** (R2 1.00): a in {23..31} | same | - | same |
| L1 down c37 | 3% / 3% | **a** (R2 1.00): a in {8..10} | same | - | same |
| L1 down c118 | 2% / 2% | **a** (R2 1.00): a in {82..83} | same | - | same |
| L1 up c668 | 10% / 10% | **a** (R2 1.00): a in {22..31} | same | (reads) | same |
| L1 up c663 | 2% / 2% | **a** (R2 1.00): a in {99..100} | same | (reads) | same |
| L1 gate c592 | 8% / 8% | **a** (R2 1.00): a in {42..49} | same | (reads) | same |
| L1 gate c402 | 2% / 2% | **a** (R2 1.00): a in {63..64} | same | (reads) | same |
| L1 gate c366 | 3% / 3% | **a** (R2 1.00): a in {76..78} | same | (reads) | same |
| L1 gate c314 | 9% / 9% | **a** (R2 1.00): a in {55..63} | same | (reads) | same |
| L1 down c251 | 2% / 2% | **a** (R2 1.00): a in {8..9} | same | - | same |
| L1 down c163 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | - | same |
| L2 up c476 | 5% / 6% | **a** (R2 1.00): a in {70..74} | **a** (R2 1.00): a in {70..75} | (reads) | same |
| L2 up c65 | 15% / 15% | **a** (R2 1.00): a in {1..15} | same | (reads) | same |
| L2 up c111 | 4% / 4% | **a** (R2 1.00): a in {12..15} | same | (reads) | same |
| L2 up c128 | 23% / 23% | **a** (R2 1.00): a in {10..32} | same | (reads) | same |
| L2 v c157 (kv6) | 2% / 2% | **a** (R2 1.00): a in {98..99} | same | (reads) | same |
| L2 up c617 | 2% / 2% | **a** (R2 1.00): a in {60..61} | same | (reads) | same |
| L2 gate c617 | 6% / 7% | **a** (R2 0.99): a in {60..65} | **a** (R2 1.00): a in {59..65} | (reads) | same |
| L2 up c11 | 23% / 24% | **a** (R2 1.00): a in {27..49} | **a** (R2 1.00): a in {27..50} | (reads) | same |
| L2 down c639 | 14% / 14% | **a** (R2 1.00): a in {2..15} | same | a: mod50 +2% | a: mod50 +2% |
| L2 down c275 | 3% / 3% | **a** (R2 1.00): a in {2..4} | same | - | same |
| L2 down c351 | 3% / 3% | **a** (R2 1.00): a in {15..17} | same | - | same |
| L2 down c385 | 2% / 2% | **a** (R2 1.00): a in {10..11} | same | - | same |
| L2 down c777 | 4% / 4% | **a** (R2 1.00): a in {30..33} | same | - | same |
| L2 down c617 | 6% / 6% | **a** (R2 1.00): a in {60..65} | same | - | same |
| L2 down c162 | 6% / 6% | **a** (R2 1.00): a in {15..20} | same | - | same |
| L2 gate c265 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | (reads) | same |
| L2 down c79 | 14% / 14% | **a** (R2 1.00): a in {1..14} | same | a: mod100 +3%, mod50 +3%, mod25 +2%, mod20 +2% | a: mod100 +3%, mod50 +3%, mod25 +2%, mod20 +2% |
| L2 down c149 | 3% / 3% | **a** (R2 1.00): a in {23..25} | same | - | same |
| L2 down c90 | 22% / 22% | **a** (R2 1.00): a in {11..32} | same | a: mod100 +3%, mod50 +3% | a: mod100 +3%, mod50 +3% |
| L2 down c140 | 1% / 2% | **a** (R2 1.00): a in {83} | **a** (R2 0.96): a in {83..84} | - | same |
| L2 down c11 | 23% / 24% | **a** (R2 1.00): a in {27..49} | **a** (R2 1.00): a in {27..50} | a: mod100 +3% | a: mod100 +3% |
| L2 down c58 | 22% / 22% | **a** (R2 1.00): a in {71..92} | same | - | same |
| L2 gate c9 | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | (reads) | same |
| L2 down c428 | 7% / 7% | **a** (R2 1.00): a in {25..31} | same | - | same |
| L2 gate c74 | 7% / 7% | **a** (R2 1.00): a in {26..32} | same | (reads) | same |
| L2 gate c187 | 11% / 11% | **a** (R2 1.00): a in {1..11} | same | (reads) | same |
| L2 gate c351 | 12% / 12% | **a** (R2 1.00): a in {12..23} | same | (reads) | same |
| L2 gate c428 | 4% / 4% | **a** (R2 1.00): a in {26..29} | same | (reads) | same |
| L2 gate c29 | 3% / 3% | **a** (R2 1.00): a in {2..4} | same | (reads) | same |
| L3 up c210 | 9% / 9% | **a** (R2 1.00): a in {56..64} | same | (reads) | same |
| L3 gate c176 | 6% / 6% | **a** (R2 1.00): a in {26..31} | same | (reads) | same |
| L3 gate c29 | 18% / 19% | **a** (R2 0.99): a in {56..72, 74} | **a** (R2 1.00): a in {56..74} | (reads) | same |
| L3 down c113 | 17% / 17% | **a** (R2 1.00): a in {46..62} | same | a: mod50 +2% | a: mod50 +2% |
| L3 v c159 (kv1) | 2% / 4% | **a** (R2 0.97): a in {52..53} | **a** (R2 0.97): a in {51..54} | (reads) | same |
| L3 up c357 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | (reads) | same |
| L3 v c101 (kv1) | 8% / 9% | **a** (R2 0.98): a in {13..19, 21} | **a** (R2 1.00): a in {13..21} | (reads) | same |
| L3 gate c316 | 15% / 15% | **a** (R2 1.00): a in {1..15} | same | (reads) | same |
| L3 gate c849 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | (reads) | same |
| L3 down c121 | 15% / 15% | **a** (R2 1.00): a in {78..92} | same | - | same |
| L3 gate c61 | 14% / 14% | **a** (R2 1.00): a in {27..40} | **a** (R2 0.99): a in {27..40} | (reads) | same |
| L3 down c77 | 14% / 14% | **a** (R2 1.00): a in {70..83} | same | - | same |
| L3 down c76 | 3% / 3% | **a** (R2 1.00): a in {1..3} | same | - | same |
| L3 v c237 (kv1) | 8% / 8% | **a** (R2 1.00): a in {1..8} | same | (reads) | same |
| L3 v c225 (kv1) | 2% / 6% | **a** (R2 0.99): a in {48, 50} | **a** (R2 1.00): a in {45..50} | (reads) | same |
| L3 v c211 (kv1) | 11% / 17% | **a** (R2 0.99): a in {16..24, 26..27} | **a** (R2 0.99): a in {13..29} | (reads) | same |
| L3 v c205 (kv1) | 1% / 2% | **a** (R2 0.99): a in {55} | **a** (R2 0.99): a in {43..44} | (reads) | same |
| L3 v c160 (kv1) | 4% / 5% | **a** (R2 0.99): a in {41..44} | **a** (R2 1.00): a in {41..45} | (reads) | same |
| L3 v c133 (kv1) | 1% / 3% | **a** (R2 1.00): a in {33} | **a** (R2 0.98): a in {32..34} | (reads) | same |
| L3 down c210 | 9% / 9% | **a** (R2 1.00): a in {56..64} | same | - | same |
| L3 down c176 | 6% / 6% | **a** (R2 1.00): a in {26..31} | same | - | same |
| L3 gate c41 | 3% / 3% | **a** (R2 1.00): a in {48..50} | same | (reads) | same |
| L3 gate c77 | 9% / 9% | **a** (R2 1.00): a in {73..81} | same | (reads) | same |
| L3 gate c121 | 15% / 15% | **a** (R2 1.00): a in {78..92} | same | (reads) | same |
| L4 up c390 | 14% / 14% | **a** (R2 1.00): a in {76..89} | same | (reads) | same |
| L4 up c317 | 2% / 2% | **a** (R2 1.00): a in {23..24} | same | (reads) | same |
| L4 gate c80 | 17% / 17% | **a** (R2 1.00): a in {41..57} | same | (reads) | same |
| L4 down c80 | 17% / 17% | **a** (R2 1.00): a in {41..57} | same | a: mod50 +2% | a: mod50 +2% |
| L4 up c69 | 6% / 7% | **a** (R2 1.00): a in {53..58} | **a** (R2 1.00): a in {53..59} | (reads) | same |
| L4 up c63 | 3% / 3% | **a** (R2 0.99): a in {98..100} | **a** (R2 1.00): a in {98..100} | (reads) | same |
| L4 gate c382 | 4% / 4% | **a** (R2 1.00): a in {23..26} | same | (reads) | same |
| L4 gate c175 | 6% / 6% | **a** (R2 1.00): a in {39..44} | same | (reads) | same |
| L4 gate c110 | 18% / 18% | **a** (R2 1.00): a in {53..70} | same | (reads) | same |
| L4 up c673 | 44% / 44% | **a** (R2 1.00): a in {1..44} | same | (reads) | same |
| L4 down c62 | 23% / 23% | **a** (R2 1.00): a in {11..33} | same | a: mod100 +2% | a: mod100 +2% |
| L4 down c530 | 7% / 7% | **a** (R2 1.00): a in {53..59} | same | - | same |
| L4 down c487 | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | - | same |
| L4 down c390 | 14% / 14% | **a** (R2 1.00): a in {76..89} | same | - | same |
| L4 down c237 | 6% / 6% | **a** (R2 1.00): a in {17..22} | same | - | same |
| L4 down c211 | 18% / 18% | **a** (R2 1.00): a in {53..70} | same | a: mod50 +2% | a: mod50 +2% |
| L4 down c181 | 14% / 14% | **a** (R2 1.00): a in {30..43} | same | - | same |
| L4 down c158 | 17% / 17% | **a** (R2 1.00): a in {83..99} | **a** (R2 0.99): a in {83..99} | a: mod50 +2% | a: mod50 +2% |
| L4 down c622 | 4% / 4% | **a** (R2 1.00): a in {73..76} | same | - | same |
| L5 v c190 (kv5) | 14% / 4% | **a** (R2 0.88): a in {4..18} | unexplained (best R2 0.21) | (reads) | same |
| L5 v c55 (kv5) | 18% / 12% | **a** (R2 0.90): a in {12..28} | **a//10** (R2 0.53): (tens) a in {18, 20..29} | (reads) | same |
| L5 down c27 | 30% / 30% | **a** (R2 1.00): a in {2..31} | same | a: mod100 +5%, mod50 +4% | a: mod100 +5%, mod50 +4% |
| L5 down c16 | 35% / 35% | **a** (R2 1.00): a in {15..49} | same | a: mod100 +3% | a: mod100 +3% |
| L5 down c45 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | - | same |
| L5 down c124 | 12% / 12% | **a** (R2 1.00): a in {1..12} | same | a: mod50 +3%, mod25 +2%, mod20 +3% | a: mod50 +3%, mod25 +2%, mod20 +3% |
| L5 down c177 | 2% / 2% | **a** (R2 1.00): a in {82..83} | **a** (R2 0.99): a in {82..83} | - | same |
| L5 gate c838 | 2% / 2% | **a** (R2 1.00): a in {2..3} | same | (reads) | same |
| L5 up c27 | 26% / 26% | **a** (R2 1.00): a in {2..27} | same | (reads) | same |
| L5 down c838 | 3% / 3% | **a** (R2 1.00): a in {2..4} | same | - | same |
| L5 gate c902 | 4% / 4% | **a** (R2 0.99): a in {1..4} | **a** (R2 1.00): a in {1..4} | (reads) | same |
| L5 up c16 | 10% / 10% | **a** (R2 1.00): a in {1..10} | same | (reads) | same |
| L5 v c23 (kv5) | 24% / 22% | **a** (R2 0.88): a in {24..48} | **a//10** (R2 0.64): (tens) a in {24..49} | (reads) | same |
| L5 up c820 | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | (reads) | same |
| L5 v c48 (kv5) | 24% / 15% | **a** (R2 0.91): a in {66..89} | unexplained (best R2 0.46) | (reads) | same |
| L6 down c848 | 2% / 2% | **a** (R2 1.00): a in {99..100} | same | - | same |
| L6 gate c309 | 17% / 17% | **a** (R2 1.00): a in {83..99} | same | (reads) | same |
| L6 down c28 | 17% / 17% | **a** (R2 1.00): a in {83..99} | same | - | same |
| L6 down c220 | 10% / 10% | **a** (R2 1.00): a in {2..11} | same | - | same |
| L6 down c25 | 30% / 30% | **a** (R2 1.00): a in {2..31} | same | a: mod100 +5%, mod50 +3% | a: mod100 +5%, mod50 +3% |
| L6 up c28 | 8% / 8% | **a** (R2 1.00): a in {92..99} | same | (reads) | same |
| L6 up c212 | 13% / 13% | **a** (R2 1.00): a in {3..15} | same | (reads) | same |
| L6 up c358 | 2% / 2% | **a** (R2 1.00): a in {1..2} | **a** (R2 0.99): a in {1..2} | (reads) | same |
| L6 up c460 | 6% / 6% | **a** (R2 1.00): a in {1..6} | same | (reads) | same |
| L6 down c191 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | - | same |
| L6 gate c212 | 29% / 29% | **a** (R2 1.00): a in {3..31} | same | (reads) | same |
| L7 up c698 | 3% / 3% | **a** (R2 1.00): a in {2..4} | same | (reads) | same |
| L7 up c420 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | (reads) | same |
| L7 gate c286 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | (reads) | same |
| L7 gate c287 | 3% / 3% | **a** (R2 1.00): a in {5..7} | same | (reads) | same |
| L7 gate c478 | 3% / 3% | **a** (R2 0.99): a in {92..94} | **a** (R2 1.00): a in {92..94} | (reads) | same |
| L7 v c30 (kv0) | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | (reads) | same |
| L7 down c478 | 2% / 2% | **a** (R2 0.97): a in {92..93} | **a** (R2 1.00): a in {92..93} | - | same |
| L7 gate c141 | 11% / 11% | **a** (R2 0.98): a in {2..12} | **a** (R2 0.99): a in {2..12} | (reads) | same |
| L7 gate c150 | 7% / 7% | **a** (R2 1.00): a in {17..23} | **a** (R2 0.99): a in {17..23} | (reads) | same |
| L7 gate c203 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | (reads) | same |
| L7 down c15 | 12% / 12% | **a** (R2 1.00): a in {1..12} | same | - | same |
| L7 down c141 | 13% / 13% | **a** (R2 1.00): a in {87..99} | same | - | same |
| L7 down c287 | 3% / 3% | **a** (R2 1.00): a in {5..7} | same | - | same |
| L7 down c455 | 3% / 3% | **a** (R2 1.00): a in {7..9} | same | - | same |
| L8 down c23 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | - | same |
| L8 down c12 | 9% / 9% | **a** (R2 1.00): a in {2..10} | same | - | same |
| L8 down c48 | 8% / 8% | **a** (R2 1.00): a in {2..9} | same | - | same |
| L8 o c3 (H23) | 93% / 93% | **a** (R2 0.98): a in {8..100} | **a** (R2 0.97): a in {8..100} | - | same |
| L8 up c21 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | (reads) | same |
| L8 gate c27 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | (reads) | same |
| L8 up c58 | 6% / 6% | **a** (R2 1.00): a in {2..7} | same | (reads) | same |
| L8 gate c21 | 5% / 5% | **a** (R2 1.00): a in {3..7} | **a** (R2 0.97): a in {3..7} | (reads) | same |
| L8 v c248 (kv6) | 85% / 91% | **a** (R2 0.64): a in {9, 11, 13..49, 51, 53..99} | **a** (R2 0.79): a in {8..99} | (reads) | same |
| L9 up c38 | 8% / 8% | **a** (R2 1.00): a in {3..10} | same | (reads) | same |
| L9 down c289 | 8% / 8% | **a** (R2 1.00): a in {3..10} | same | - | same |
| L9 up c289 | 3% / 3% | **a** (R2 1.00): a in {1..3} | same | (reads) | same |
| L9 gate c399 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | (reads) | same |
| L9 down c369 | 3% / 3% | **a** (R2 1.00): a in {1..3} | same | - | same |
| L9 down c158 | 3% / 3% | **a** (R2 1.00): a in {2..4} | same | - | same |
| L10 down c493 | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | - | same |
| L10 down c203 | 8% / 8% | **a** (R2 1.00): a in {3..10} | same | - | same |
| L10 down c75 | 2% / 2% | **a** (R2 1.00): a in {42..43} | **a** (R2 0.90): a in {42..43} | - | same |
| L10 gate c391 | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | (reads) | same |
| L11 gate c750 | 7% / 7% | **a** (R2 1.00): a in {3..9} | same | (reads) | same |
| L11 up c113 | 4% / 4% | **a** (R2 1.00): a in {31..34} | same | (reads) | same |
| L11 up c50 | 10% / 11% | **a** (R2 1.00): a in {83, 85..93} | **a** (R2 1.00): a in {83..93} | (reads) | same |
| L11 gate c782 | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | (reads) | same |
| L11 up c294 | 3% / 3% | **a** (R2 1.00): a in {98..100} | same | (reads) | same |
| L11 gate c61 | 13% / 13% | **a** (R2 1.00): a in {19..31} | same | (reads) | same |
| L11 gate c89 | 8% / 8% | **a** (R2 1.00): a in {31..38} | same | (reads) | same |
| L11 gate c117 | 2% / 2% | **a** (R2 1.00): a in {48..49} | same | (reads) | same |
| L11 gate c294 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | (reads) | same |
| L11 down c117 | 4% / 4% | **a** (R2 1.00): a in {48..51} | same | - | same |
| L11 down c113 | 3% / 3% | **a** (R2 1.00): a in {31..33} | same | - | same |
| L11 gate c47 | 11% / 11% | **a** (R2 1.00): a in {21..31} | same | (reads) | same |
| L11 gate c30 | 12% / 12% | **a** (R2 1.00): a in {68..79} | same | (reads) | same |
| L11 down c50 | 10% / 11% | **a** (R2 1.00): a in {83, 85..93} | **a** (R2 1.00): a in {83..93} | - | same |
| L11 down c89 | 8% / 9% | **a** (R2 1.00): a in {31..38} | **a** (R2 1.00): a in {31..39} | - | same |
| L11 down c47 | 13% / 13% | **a** (R2 1.00): a in {19..31} | same | - | same |
| L11 down c31 | 16% / 15% | **a** (R2 1.00): a in {8..23} | **a** (R2 1.00): a in {9..23} | a: mod50 +2% | a: mod50 +2% |
| L11 down c22 | 14% / 14% | **a** (R2 1.00): a in {87..100} | same | a: mod20 +2% | a: mod20 +2% |
| L11 down c30 | 12% / 12% | **a** (R2 1.00): a in {68..79} | same | a: mod20 +2% | a: mod20 +2% |
| L11 gate c18 | 13% / 13% | **a** (R2 1.00): a in {50..62} | same | (reads) | same |
| L11 down c465 | 2% / 2% | **a** (R2 1.00): a in {99..100} | same | - | same |
| L11 down c18 | 13% / 13% | **a** (R2 1.00): a in {50..62} | same | a: mod20 +3% | a: mod20 +3% |
| L11 down c13 | 5% / 5% | **a** (R2 1.00): a in {5..9} | **a** (R2 0.99): a in {5..9} | - | same |
| L12 down c138 | 2% / 2% | **a** (R2 1.00): a in {92..93} | same | - | same |
| L12 down c17 | 17% / 16% | **a** (R2 0.99): a in {69..85} | **a** (R2 0.97): a in {70..85} | a: mod50 +2% | a: mod50 +2% |
| L12 gate c693 | 7% / 7% | **a** (R2 1.00): a in {52..58} | same | (reads) | same |
| L12 down c633 | 15% / 15% | **a** (R2 1.00): a in {1..15} | same | a: mod100 +2% | a: mod100 +2% |
| L12 up c50 | 15% / 15% | **a** (R2 1.00): a in {1..15} | **a** (R2 0.99): a in {1..15} | (reads) | same |
| L12 up c32 | 3% / 3% | **a** (R2 0.90): a in {1..3} | **a** (R2 1.00): a in {1..3} | (reads) | same |
| L12 gate c224 | 9% / 9% | **a** (R2 1.00): a in {92..100} | same | (reads) | same |
| L12 gate c26 | 17% / 17% | **a** (R2 1.00): a in {25..41} | same | (reads) | same |
| L12 down c224 | 9% / 9% | **a** (R2 1.00): a in {92..100} | same | - | same |
| L12 down c168 | 2% / 2% | **a** (R2 0.89): a in {1..2} | **a** (R2 1.00): a in {1..2} | - | same |
| L12 down c18 | 14% / 14% | **a** (R2 1.00): a in {19..32} | same | - | same |
| L12 down c110 | 5% / 5% | **a** (R2 1.00): a in {78..82} | same | - | same |
| L12 up c295 | 7% / 8% | **a** (R2 1.00): a in {92..98} | **a** (R2 0.98): a in {92..99} | (reads) | same |
| L12 down c550 | 2% / 4% | **a** (R2 0.92): a in {57..58} | **a** (R2 0.86): a in {55..58} | - | same |
| L13 down c957 | 2% / 2% | **a** (R2 1.00): a in {78..79} | same | - | same |
| L13 down c999 | 5% / 5% | **a** (R2 0.99): a in {73..77} | **a** (R2 1.00): a in {73..77} | - | same |
| L13 up c332 | 6% / 6% | **a** (R2 1.00): a in {11..16} | **a** (R2 0.96): a in {11..16} | (reads) | same |
| L13 up c623 | 3% / 3% | **a** (R2 1.00): a in {86..88} | same | (reads) | same |
| L13 gate c20 | 15% / 15% | **a** (R2 1.00): a in {52..66} | same | (reads) | same |
| L13 gate c23 | 15% / 15% | **a** (R2 1.00): a in {31..45} | **a** (R2 0.99): a in {31..45} | (reads) | same |
| L13 gate c108 | 19% / 21% | **a** (R2 0.97): a in {14..31} | **a** (R2 1.00): a in {11..31} | (reads) | same |
| L13 gate c332 | 3% / 3% | **a** (R2 1.00): a in {1..3} | same | (reads) | same |
| L13 gate c877 | 5% / 5% | **a** (R2 1.00): a in {82..86} | same | (reads) | same |
| L13 up c24 | 4% / 4% | **a** (R2 1.00): a in {25..28} | same | (reads) | same |
| L13 up c48 | 2% / 2% | **a** (R2 1.00): a in {92..93} | same | (reads) | same |
| L13 up c74 | 3% / 3% | **a** (R2 1.00): a in {82..84} | same | (reads) | same |
| L13 up c76 | 10% / 10% | **a** (R2 1.00): a in {1..10} | same | (reads) | same |
| L13 up c96 | 4% / 4% | **a** (R2 1.00): a in {86..89} | same | (reads) | same |
| L13 up c133 | 7% / 7% | **a** (R2 1.00): a in {63..69} | **a** (R2 0.99): a in {63..69} | (reads) | same |
| L13 up c195 | 4% / 4% | **a** (R2 1.00): a in {21..24} | same | (reads) | same |
| L13 down c23 | 15% / 15% | **a** (R2 1.00): a in {32..46} | same | - | same |
| L13 down c774 | 4% / 5% | **a** (R2 1.00): a in {14..17} | **a** (R2 1.00): a in {13..17} | - | same |
| L13 down c133 | 6% / 6% | **a** (R2 1.00): a in {63..68} | same | - | same |
| L13 down c7 | 21% / 21% | **a** (R2 1.00): a in {11..31} | same | - | same |
| L13 down c291 | 8% / 8% | **a** (R2 1.00): a in {2..9} | same | - | same |
| L13 down c526 | 2% / 2% | **a** (R2 1.00): a in {23..24} | same | - | same |
| L13 down c24 | 14% / 14% | **a** (R2 1.00): a in {1..14} | same | - | same |
| L13 down c108 | 3% / 3% | **a** (R2 1.00): a in {83..85} | same | - | same |
| L13 down c161 | 2% / 2% | **a** (R2 1.00): a in {99..100} | same | - | same |
| L14 up c238 | 5% / 5% | **a** (R2 1.00): a in {45..49} | same | (reads) | same |
| L14 gate c213 | 6% / 6% | **a** (R2 1.00): a in {69..74} | **a** (R2 0.98): a in {69..74} | (reads) | same |
| L14 up c150 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | (reads) | same |
| L14 up c148 | 3% / 3% | **a** (R2 0.93): a in {18..20} | **a** (R2 1.00): a in {18..20} | (reads) | same |
| L14 up c96 | 8% / 8% | **a** (R2 1.00): a in {12..19} | same | (reads) | same |
| L14 up c34 | 12% / 12% | **a** (R2 0.99): a in {63..74} | **a** (R2 0.98): a in {63..74} | (reads) | same |
| L14 up c21 | 2% / 2% | **a** (R2 1.00): a in {56..57} | same | (reads) | same |
| L14 gate c833 | 5% / 5% | **a** (R2 1.00): a in {65..69} | same | (reads) | same |
| L14 gate c627 | 7% / 7% | **a** (R2 1.00): a in {63..69} | same | (reads) | same |
| L14 down c182 | 7% / 7% | **a** (R2 1.00): a in {75..81} | same | - | same |
| L14 down c151 | 9% / 9% | **a** (R2 1.00): a in {2..10} | same | - | same |
| L14 down c150 | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | - | same |
| L14 down c131 | 2% / 2% | **a** (R2 1.00): a in {13..14} | same | - | same |
| L14 down c70 | 13% / 13% | **a** (R2 0.99): a in {52..64} | **a** (R2 1.00): a in {52..64} | - | same |
| L14 down c48 | 7% / 7% | **a** (R2 1.00): a in {25..31} | same | - | same |
| L14 down c41 | 13% / 13% | **a** (R2 1.00): a in {32..44} | same | - | same |
| L14 down c213 | 5% / 5% | **a** (R2 1.00): a in {68..72} | same | - | same |
| L14 down c2 | 5% / 5% | **a** (R2 1.00): a in {1..5} | same | - | same |
| L14 gate c70 | 14% / 14% | **a** (R2 1.00): a in {52..65} | same | (reads) | same |
| L14 gate c48 | 7% / 7% | **a** (R2 1.00): a in {25..31} | same | (reads) | same |
| L14 gate c41 | 14% / 14% | **a** (R2 1.00): a in {32..45} | same | (reads) | same |
| L14 up c248 | 8% / 8% | **a** (R2 1.00): a in {22..29} | same | (reads) | same |
| L14 up c677 | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {24..26} | (reads) | same |
| L14 v c54 (kv4) | 8% / 8% | **a** (R2 1.00): a in {93..100} | same | (reads) | same |
| L14 v c67 (kv4) | 1% / 2% | **a** (R2 0.94): a in {28} | **a** (R2 1.00): a in {28..29} | (reads) | same |
| L14 down c547 | 4% / 4% | **a** (R2 1.00): a in {34..37} | same | - | same |
| L14 down c238 | 2% / 2% | **a** (R2 1.00): a in {45..46} | same | - | same |
| L14 down c248 | 2% / 2% | **a** (R2 1.00): a in {23..24} | same | - | same |
| L14 down c320 | 5% / 5% | **a** (R2 1.00): a in {71..75} | same | - | same |
| L14 down c671 | 3% / 3% | **a** (R2 1.00): a in {52..54} | same | - | same |
| L14 down c588 | 6% / 6% | **a** (R2 1.00): a in {95..100} | same | - | same |
| L14 down c615 | 3% / 3% | **a** (R2 1.00): a in {14..16} | same | - | same |
| L14 down c644 | 4% / 4% | **a** (R2 1.00): a in {81..84} | same | - | same |
| L14 down c833 | 4% / 4% | **a** (R2 1.00): a in {66..69} | same | - | same |
| L14 down c701 | 5% / 5% | **a** (R2 1.00): a in {63..67} | same | - | same |
| L14 down c1021 | 2% / 2% | **a** (R2 1.00): a in {63..64} | same | - | same |
| L15 up c233 | 11% / 11% | **a** (R2 1.00): a in {75..85} | same | (reads) | same |
| L15 up c45 | 14% / 14% | **a** (R2 1.00): a in {5..18} | same | (reads) | same |
| L15 up c201 | 16% / 16% | **a** (R2 1.00): a in {85..100} | same | (reads) | same |
| L15 up c110 | 13% / 13% | **a** (R2 1.00): a in {33..45} | same | (reads) | same |
| L15 up c104 | 3% / 3% | **a** (R2 1.00): a in {1..3} | same | (reads) | same |
| L15 down c762 | 2% / 2% | **a** (R2 1.00): a in {98..99} | same | - | same |
| L15 gate c53 | 15% / 15% | **a** (R2 1.00): a in {86..100} | same | (reads) | same |
| L15 down c946 | 3% / 3% | **a** (R2 1.00): a in {80..82} | same | - | same |
| L15 down c904 | 6% / 6% | **a** (R2 1.00): a in {64..69} | same | - | same |
| L15 down c894 | 3% / 3% | **a** (R2 1.00): a in {62..64} | same | - | same |
| L15 up c98 | 5% / 5% | **a** (R2 1.00): a in {19..23} | same | (reads) | same |
| L15 up c36 | 8% / 8% | **a** (R2 1.00): a in {2..9} | same | (reads) | same |
| L15 gate c854 | 3% / 3% | **a** (R2 1.00): a in {83..85} | same | (reads) | same |
| L15 gate c233 | 7% / 7% | **a** (R2 1.00): a in {76..82} | same | (reads) | same |
| L15 gate c772 | 3% / 3% | **a** (R2 1.00): a in {32..34} | same | (reads) | same |
| L15 gate c451 | 3% / 3% | **a** (R2 1.00): a in {20..22} | same | (reads) | same |
| L15 gate c246 | 4% / 4% | **a** (R2 1.00): a in {86..89} | same | (reads) | same |
| L15 gate c144 | 5% / 5% | **a** (R2 1.00): a in {57..61} | same | (reads) | same |
| L15 gate c186 | 7% / 7% | **a** (R2 1.00): a in {63..69} | same | (reads) | same |
| L15 gate c137 | 6% / 6% | **a** (R2 1.00): a in {40..45} | same | (reads) | same |
| L15 gate c77 | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | (reads) | same |
| L15 up c854 | 2% / 2% | **a** (R2 1.00): a in {84..85} | same | (reads) | same |
| L15 up c483 | 3% / 3% | **a** (R2 1.00): a in {18..20} | same | (reads) | same |
| L15 up c477 | 4% / 4% | **a** (R2 1.00): a in {50..53} | same | (reads) | same |
| L15 up c236 | 13% / 13% | **a** (R2 1.00): a in {21..33} | same | (reads) | same |
| L15 down c110 | 7% / 7% | **a** (R2 1.00): a in {46..52} | same | - | same |
| L15 down c854 | 6% / 7% | **a** (R2 1.00): a in {81..86} | **a** (R2 1.00): a in {80..86} | - | same |
| L15 down c814 | 4% / 4% | **a** (R2 1.00): a in {73..76} | same | - | same |
| L15 down c772 | 2% / 2% | **a** (R2 1.00): a in {32..33} | same | - | same |
| L15 down c590 | 3% / 3% | **a** (R2 1.00): a in {17..19} | same | - | same |
| L15 down c31 | 16% / 16% | **a** (R2 1.00): a in {85..100} | same | - | same |
| L15 down c48 | 13% / 13% | **a** (R2 1.00): a in {1..13} | same | - | same |
| L15 down c98 | 3% / 3% | **a** (R2 1.00): a in {21..23} | same | - | same |
| L15 down c451 | 2% / 2% | **a** (R2 1.00): a in {20..21} | same | - | same |
| L15 down c366 | 8% / 8% | **a** (R2 1.00): a in {68..75} | same | - | same |
| L15 down c364 | 5% / 5% | **a** (R2 1.00): a in {95..99} | same | - | same |
| L15 down c310 | 2% / 2% | **a** (R2 1.00): a in {8..9} | same | - | same |
| L15 down c17 | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | - | same |
| L15 down c144 | 6% / 6% | **a** (R2 1.00): a in {57..62} | same | - | same |
| L15 down c233 | 10% / 10% | **a** (R2 1.00): a in {75..84} | same | - | same |
| L15 down c236 | 3% / 3% | **a** (R2 1.00): a in {23..25} | same | - | same |
| L15 down c597 | 3% / 3% | **a** (R2 1.00): a in {87..89} | same | - | same |
| L15 down c470 | 6% / 6% | **a** (R2 1.00): a in {40..45} | same | - | same |
| L15 down c483 | 4% / 4% | **a** (R2 1.00): a in {18..21} | same | - | same |
| L15 down c477 | 3% / 3% | **a** (R2 1.00): a in {50..52} | same | - | same |
| L15 down c599 | 2% / 2% | **a** (R2 1.00): a in {77..78} | same | - | same |
| L15 down c743 | 6% / 6% | **a** (R2 1.00): a in {35..40} | same | - | same |
| L15 up c834 | 4% / 4% | **a** (R2 1.00): a in {61..64} | same | (reads) | same |
| L16 down c807 | 2% / 2% | **a** (R2 1.00): a in {25..26} | same | - | same |
| L16 gate c484 | 4% / 4% | **a** (R2 1.00): a in {39..42} | **a** (R2 0.99): a in {39..42} | (reads) | same |
| L16 gate c449 | 4% / 4% | **a** (R2 1.00): a in {30..33} | same | (reads) | same |
| L16 gate c438 | 8% / 8% | **a** (R2 1.00): a in {67..74} | **a** (R2 0.99): a in {67..74} | (reads) | same |
| L16 gate c260 | 3% / 3% | **a** (R2 1.00): a in {60..62} | **a** (R2 0.99): a in {60..62} | (reads) | same |
| L16 down c438 | 5% / 5% | **a** (R2 1.00): a in {68..72} | **a** (R2 0.99): a in {68..72} | - | same |
| L16 down c449 | 3% / 3% | **a** (R2 1.00): a in {16..18} | same | - | same |
| L16 down c461 | 5% / 5% | **a** (R2 1.00): a in {55..59} | same | - | same |
| L16 down c570 | 2% / 2% | **a** (R2 1.00): a in {44..45} | same | - | same |
| L16 up c238 | 2% / 2% | **a** (R2 1.00): a in {51..52} | same | (reads) | same |
| L16 up c282 | 15% / 15% | **a** (R2 1.00): a in {1..15} | same | (reads) | same |
| L16 up c298 | 6% / 6% | **a** (R2 1.00): a in {95..100} | same | (reads) | same |
| L16 up c449 | 8% / 8% | **a** (R2 1.00): a in {47..54} | same | (reads) | same |
| L16 up c461 | 5% / 5% | **a** (R2 1.00): a in {55..59} | same | (reads) | same |
| L16 up c570 | 19% / 19% | **a** (R2 1.00): a in {32..50} | same | (reads) | same |
| L16 up c665 | 3% / 3% | **a** (R2 1.00): a in {60..62} | same | (reads) | same |
| L16 v c20 (kv0) | 36% / 36% | **a** (R2 0.99): a in {65..100} | same | (reads) | same |
| L16 v c23 (kv0) | 13% / 12% | **a** (R2 0.93): a in {18..30} | **a//10** (R2 0.85): (tens) a in {19..30} | (reads) | same |
| L16 v c31 (kv5) | 14% / 15% | **a** (R2 0.99): a in {65..78} | **a** (R2 1.00): a in {65..79} | (reads) | same |
| L16 v c59 (kv0) | 17% / 17% | **a** (R2 1.00): a in {84..100} | **a** (R2 0.99): a in {84..100} | (reads) | same |
| L16 v c72 (kv0) | 34% / 34% | **a** (R2 0.99): a in {1..34} | **a** (R2 1.00): a in {1..34} | (reads) | same |
| L16 down c864 | 6% / 6% | **a** (R2 1.00): a in {70..75} | same | - | same |
| L16 gate c9 | 15% / 15% | **a** (R2 1.00): a in {86..100} | same | (reads) | same |
| L16 gate c44 | 8% / 8% | **a** (R2 1.00): a in {26..33} | same | (reads) | same |
| L16 gate c151 | 2% / 2% | **a** (R2 1.00): a in {52..53} | same | (reads) | same |
| L16 down c610 | 2% / 2% | **a** (R2 1.00): a in {23..24} | same | - | same |
| L16 down c665 | 2% / 2% | **a** (R2 1.00): a in {61..62} | same | - | same |
| L16 down c692 | 2% / 2% | **a** (R2 1.00): a in {88..89} | same | - | same |
| L16 down c38 | 2% / 2% | **a** (R2 1.00): a in {99..100} | same | - | same |
| L16 down c298 | 6% / 6% | **a** (R2 1.00): a in {95..100} | same | - | same |
| L16 down c59 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | - | same |
| L16 down c57 | 10% / 10% | **a** (R2 1.00): a in {76..85} | same | - | same |
| L16 down c44 | 5% / 5% | **a** (R2 1.00): a in {29..33} | **a** (R2 0.97): a in {29..33} | - | same |
| L16 up c120 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | (reads) | same |
| L16 up c57 | 10% / 10% | **a** (R2 1.00): a in {76..85} | same | (reads) | same |
| L16 up c20 | 10% / 10% | **a** (R2 1.00): a in {13..22} | same | (reads) | same |
| L16 gate c784 | 3% / 3% | **a** (R2 1.00): a in {35..37} | same | (reads) | same |
| L16 v c151 (kv0) | 13% / 12% | **a** (R2 0.95): a in {80..92} | **a** (R2 0.79): a in {80..92} | (reads) | same |
| L16 down c260 | 9% / 9% | **a** (R2 1.00): a in {56..64} | same | - | same |
| L16 down c370 | 3% / 3% | **a** (R2 1.00): a in {39..41} | same | - | same |
| L16 down c256 | 2% / 2% | **a** (R2 1.00): a in {27..28} | same | - | same |
| L16 down c78 | 15% / 15% | **a** (R2 1.00): a in {1..15} | same | a: mod50 +2% | a: mod50 +2% |
| L16 down c20 | 12% / 12% | **a** (R2 1.00): a in {11..22} | same | - | same |
| L16 down c3 | 27% / 27% | **a** (R2 1.00): a in {32..58} | same | a: mod100 +3% | a: mod100 +3% |
| L16 v c82 (kv0) | 16% / 13% | **a** (R2 0.93): a in {9..25} | **a** (R2 0.88): a in {12..24} | (reads) | same |
| L16 v c245 (kv5) | 4% / 4% | **a** (R2 0.99): a in {34..37} | **a** (R2 0.95): a in {34..37} | (reads) | same |
| L16 v c194 (kv0) | 10% / 10% | **a** (R2 1.00): a in {25..34} | same | (reads) | same |
| L16 v c74 (kv0) | 10% / 10% | **a** (R2 1.00): a in {45..54} | same | (reads) | same |
| L16 v c131 (kv0) | 14% / 14% | **a** (R2 1.00): a in {87..100} | same | (reads) | same |
| L17 down c56 | 9% / 9% | **a** (R2 1.00): a in {5..13} | same | - | same |
| L17 down c43 | 3% / 3% | **a** (R2 1.00): a in {96..98} | same | - | same |
| L17 down c48 | 19% / 19% | **a** (R2 1.00): a in {13..31} | same | - | same |
| L17 down c195 | 3% / 3% | **a** (R2 1.00): a in {91..93} | same | - | same |
| L17 down c865 | 3% / 3% | **a** (R2 1.00): a in {97..99} | same | - | same |
| L17 down c75 | 3% / 3% | **a** (R2 1.00): a in {1..3} | same | - | same |
| L17 down c85 | 24% / 24% | **a** (R2 1.00): a in {76..99} | same | - | same |
| L17 down c106 | 2% / 2% | **a** (R2 1.00): a in {42..43} | same | - | same |
| L17 down c138 | 7% / 7% | **a** (R2 1.00): a in {82..88} | same | - | same |
| L17 down c171 | 13% / 13% | **a** (R2 1.00): a in {64..76} | same | - | same |
| L17 down c229 | 7% / 7% | **a** (R2 1.00): a in {38..44} | same | - | same |
| L17 down c609 | 3% / 3% | **a** (R2 1.00): a in {70..72} | same | - | same |
| L17 down c565 | 2% / 2% | **a** (R2 1.00): a in {22..23} | same | - | same |
| L17 down c442 | 3% / 3% | **a** (R2 1.00): a in {56..58} | same | - | same |
| L17 down c415 | 4% / 4% | **a** (R2 1.00): a in {92..95} | same | - | same |
| L17 down c383 | 2% / 2% | **a** (R2 1.00): a in {39..40} | same | - | same |
| L17 down c359 | 5% / 5% | **a** (R2 1.00): a in {75..79} | same | - | same |
| L17 down c321 | 5% / 5% | **a** (R2 1.00): a in {45..49} | same | - | same |
| L17 down c278 | 3% / 3% | **a** (R2 1.00): a in {25..27} | same | - | same |
| L17 down c245 | 2% / 2% | **a** (R2 1.00): a in {13..14} | same | - | same |
| L17 up c779 | 2% / 2% | **a** (R2 1.00): a in {32..33} | same | (reads) | same |
| L17 gate c125 | 3% / 3% | **a** (R2 1.00): a in {56..58} | same | (reads) | same |
| L17 gate c138 | 5% / 5% | **a** (R2 1.00): a in {84..88} | same | (reads) | same |
| L17 gate c154 | 8% / 8% | **a** (R2 1.00): a in {16..23} | same | (reads) | same |
| L17 gate c245 | 8% / 8% | **a** (R2 1.00): a in {75..82} | same | (reads) | same |
| L17 gate c356 | 6% / 6% | **a** (R2 1.00): a in {46..51} | same | (reads) | same |
| L17 gate c383 | 16% / 16% | **a** (R2 1.00): a in {34..49} | same | (reads) | same |
| L17 gate c453 | 4% / 4% | **a** (R2 1.00): a in {92..95} | same | (reads) | same |
| L17 gate c687 | 3% / 3% | **a** (R2 1.00): a in {30..32} | same | (reads) | same |
| L17 gate c750 | 7% / 7% | **a** (R2 1.00): a in {49..55} | same | (reads) | same |
| L17 gate c857 | 13% / 13% | **a** (R2 1.00): a in {64..76} | same | (reads) | same |
| L17 up c48 | 23% / 23% | **a** (R2 1.00): a in {9..31} | same | (reads) | same |
| L17 up c54 | 11% / 11% | **a** (R2 1.00): a in {2..12} | same | (reads) | same |
| L17 up c144 | 8% / 8% | **a** (R2 1.00): a in {46..53} | same | (reads) | same |
| L17 up c249 | 2% / 2% | **a** (R2 1.00): a in {54..55} | same | (reads) | same |
| L17 up c270 | 10% / 10% | **a** (R2 1.00): a in {31..40} | same | (reads) | same |
| L17 gate c48 | 3% / 3% | **a** (R2 1.00): a in {98..100} | same | (reads) | same |
| L17 down c849 | 2% / 2% | **a** (R2 1.00): a in {44..45} | same | - | same |
| L17 down c834 | 2% / 2% | **a** (R2 1.00): a in {47..48} | same | - | same |
| L17 down c750 | 2% / 2% | **a** (R2 1.00): a in {52..53} | same | - | same |
| L17 down c687 | 2% / 2% | **a** (R2 1.00): a in {31..32} | same | - | same |
| L17 down c679 | 3% / 3% | **a** (R2 1.00): a in {17..19} | same | - | same |
| L17 down c664 | 2% / 2% | **a** (R2 1.00): a in {30..31} | same | - | same |
| L18 up c206 | 35% / 35% | **a** (R2 1.00): a in {65..99} | same | (reads) | same |
| L18 up c419 | 5% / 5% | **a** (R2 1.00): a in {49..53} | same | (reads) | same |
| L18 v c233 (kv7) | 2% / 12% | unexplained (best R2 0.32) | **a** (R2 1.00): a in {18..29} | (reads) | same |
| L18 v c245 (kv7) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {60..62} | (reads) | same |
| L18 v c23 (kv7) | 2% / 8% | **a** (R2 0.67): a in {70} | **a** (R2 1.00): a in {67..74} | (reads) | same |
| L18 v c61 (kv7) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {42..45} | (reads) | same |
| L18 v c237 (kv7) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {30..31} | (reads) | same |
| L18 up c440 | 2% / 2% | **a** (R2 1.00): a in {11..12} | **a** (R2 0.99): a in {11..12} | (reads) | same |
| L18 v c121 (kv7) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {13..17} | (reads) | same |
| L18 v c139 (kv7) | 8% / 14% | **a** (R2 0.93): a in {72..79} | **a** (R2 1.00): a in {69..82} | (reads) | same |
| L18 v c147 (kv7) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {58..63} | (reads) | same |
| L18 v c176 (kv7) | 8% / 13% | **a** (R2 0.83): a in {36..42} | **a** (R2 1.00): a in {32..44} | (reads) | same |
| L18 v c202 (kv7) | 1% / 8% | unexplained (best R2 0.10) | **a** (R2 1.00): a in {33..40} | (reads) | same |
| L18 v c212 (kv7) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {20..23} | (reads) | same |
| L18 v c80 (kv7) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {60..62} | (reads) | same |
| L18 v c86 (kv7) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {30..33} | (reads) | same |
| L18 down c58 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | - | same |
| L18 down c243 | 2% / 2% | **a** (R2 1.00): a in {99..100} | same | - | same |
| L18 gate c373 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | (reads) | same |
| L18 down c354 | 9% / 9% | **a** (R2 1.00): a in {68..76} | same | - | same |
| L18 down c693 | 8% / 8% | **a** (R2 1.00): a in {92..99} | same | - | same |
| L18 down c41 | 15% / 15% | **a** (R2 1.00): a in {4..18} | same | - | same |
| L18 down c43 | 19% / 19% | **a** (R2 1.00): a in {13..31} | same | - | same |
| L18 down c55 | 16% / 16% | **a** (R2 1.00): a in {1..16} | same | a: mod50 +2% | a: mod50 +2% |
| L18 gate c55 | 11% / 11% | **a** (R2 1.00): a in {2..12} | same | (reads) | same |
| L18 down c811 | 4% / 4% | **a** (R2 1.00): a in {90..93} | same | - | same |
| L18 down c340 | 4% / 4% | **a** (R2 1.00): a in {39..42} | same | - | same |
| L18 gate c324 | 9% / 9% | **a** (R2 1.00): a in {68..76} | same | (reads) | same |
| L18 v c226 (kv7) | 1% / 5% | unexplained (best R2 0.16) | **a** (R2 1.00): a in {25..29} | (reads) | same |
| L18 v c227 (kv7) | 9% / 13% | **a** (R2 0.80): a in {30..38} | **a** (R2 1.00): a in {28..40} | (reads) | same |
| L19 up c192 | 12% / 12% | **a** (R2 1.00): a in {58..69} | same | (reads) | same |
| L19 up c702 | 6% / 6% | **a** (R2 1.00): a in {95..100} | same | (reads) | same |
| L19 down c189 | 6% / 6% | **a** (R2 1.00): a in {94..99} | same | - | same |
| L19 down c219 | 2% / 2% | **a** (R2 1.00): a in {34..35} | same | - | same |
| L19 down c241 | 9% / 9% | **a** (R2 1.00): a in {23..31} | same | - | same |
| L19 gate c219 | 6% / 6% | **a** (R2 1.00): a in {32..37} | same | (reads) | same |
| L19 down c291 | 3% / 3% | **a** (R2 1.00): a in {11..13} | same | - | same |
| L19 gate c103 | 6% / 6% | **a** (R2 1.00): a in {47..52} | same | (reads) | same |
| L19 down c820 | 6% / 6% | **a** (R2 1.00): a in {58..63} | same | - | same |
| L19 down c619 | 3% / 3% | **a** (R2 1.00): a in {35..37} | same | - | same |
| L19 down c25 | 11% / 11% | **a** (R2 1.00): a in {1..11} | same | - | same |
| L19 down c54 | 7% / 7% | **a** (R2 1.00): a in {77..83} | same | - | same |
| L19 down c64 | 3% / 3% | **a** (R2 1.00): a in {17..19} | same | - | same |
| L19 up c78 | 17% / 17% | **a** (R2 1.00): a in {1..17} | same | (reads) | same |
| L19 up c241 | 11% / 11% | **a** (R2 1.00): a in {21..31} | same | (reads) | same |
| L19 up c162 | 3% / 3% | **a** (R2 1.00): a in {1..3} | same | (reads) | same |
| L19 down c78 | 5% / 5% | **a** (R2 1.00): a in {5..9} | same | - | same |
| L19 down c10 | 37% / 37% | **a** (R2 1.00): a in {64..100} | same | a: mod100 +3% | a: mod100 +3% |
| L20 gate c499 | 3% / 3% | **a** (R2 1.00): a in {49..51} | same | (reads) | same |
| L20 gate c687 | 4% / 4% | **a** (R2 1.00): a in {28..31} | same | (reads) | same |
| L20 gate c393 | 5% / 5% | **a** (R2 1.00): a in {1..5} | same | (reads) | same |
| L20 up c7 | 15% / 15% | **a** (R2 1.00): a in {3..17} | same | (reads) | same |
| L20 up c85 | 5% / 5% | **a** (R2 1.00): a in {48..52} | same | (reads) | same |
| L20 k c106 (kv4) | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | (reads) | same |
| L20 v c123 (kv0) | 0 / 2% | off (on 0) | **a** (R2 0.92): a in {97..98} | (reads) | same |
| L20 up c679 | 2% / 2% | **a** (R2 1.00): a in {31..32} | same | (reads) | same |
| L20 up c399 | 7% / 7% | **a** (R2 1.00): a in {26..32} | same | (reads) | same |
| L20 up c110 | 2% / 2% | **a** (R2 1.00): a in {65..66} | same | (reads) | same |
| L20 v c240 (kv0) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {29..31} | (reads) | same |
| L20 gate c77 | 4% / 4% | **a** (R2 1.00): a in {74..77} | same | (reads) | same |
| L20 gate c75 | 9% / 9% | **a** (R2 1.00): a in {52..60} | same | (reads) | same |
| L20 gate c72 | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | (reads) | same |
| L20 down c630 | 6% / 6% | **a** (R2 1.00): a in {38..43} | same | - | same |
| L20 down c77 | 4% / 4% | **a** (R2 1.00): a in {74..77} | **a** (R2 0.99): a in {74..77} | - | same |
| L20 gate c194 | 4% / 4% | **a** (R2 1.00): a in {30..33} | same | (reads) | same |
| L20 gate c81 | 3% / 3% | **a** (R2 1.00): a in {22..24} | same | (reads) | same |
| L20 down c69 | 7% / 7% | **a** (R2 1.00): a in {13..19} | same | - | same |
| L20 down c38 | 7% / 7% | **a** (R2 1.00): a in {25..31} | same | - | same |
| L20 down c75 | 9% / 9% | **a** (R2 1.00): a in {52..60} | same | - | same |
| L20 down c607 | 2% / 2% | **a** (R2 1.00): a in {99..100} | same | - | same |
| L20 down c399 | 4% / 4% | **a** (R2 1.00): a in {30..33} | same | - | same |
| L20 down c440 | 3% / 3% | **a** (R2 1.00): a in {1..3} | same | - | same |
| L20 down c131 | 12% / 12% | **a** (R2 1.00): a in {1..12} | same | - | same |
| L20 down c110 | 2% / 2% | **a** (R2 1.00): a in {65..66} | same | - | same |
| L20 gate c248 | 6% / 6% | **a** (R2 1.00): a in {95..100} | same | (reads) | same |
| L20 o c8 (H17) | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | - | same |
| L20 down c98 | 5% / 5% | **a** (R2 1.00): a in {48..52} | same | - | same |
| L20 down c349 | 3% / 3% | **a** (R2 1.00): a in {2..4} | same | - | same |
| L21 up c346 | 7% / 7% | **a** (R2 1.00): a in {68..74} | same | (reads) | same |
| L21 up c282 | 6% / 6% | **a** (R2 1.00): a in {45..50} | same | (reads) | same |
| L21 down c539 | 2% / 2% | **a** (R2 1.00): a in {91..92} | same | - | same |
| L21 down c313 | 2% / 2% | **a** (R2 0.99): a in {1..2} | **a** (R2 1.00): a in {1..2} | - | same |
| L21 up c272 | 4% / 4% | **a** (R2 1.00): a in {40..43} | same | (reads) | same |
| L21 up c229 | 3% / 3% | **a** (R2 1.00): a in {38..40} | same | (reads) | same |
| L21 up c177 | 4% / 4% | **a** (R2 1.00): a in {59..62} | same | (reads) | same |
| L21 up c69 | 9% / 9% | **a** (R2 1.00): a in {92..100} | same | (reads) | same |
| L21 up c52 | 9% / 9% | **a** (R2 1.00): a in {86..94} | same | (reads) | same |
| L21 gate c326 | 2% / 2% | **a** (R2 1.00): a in {13..14} | same | (reads) | same |
| L21 gate c282 | 11% / 11% | **a** (R2 1.00): a in {2..12} | same | (reads) | same |
| L21 gate c243 | 3% / 3% | **a** (R2 1.00): a in {38..40} | same | (reads) | same |
| L21 gate c219 | 2% / 2% | **a** (R2 1.00): a in {2..3} | same | (reads) | same |
| L21 gate c160 | 9% / 9% | **a** (R2 1.00): a in {23..31} | same | (reads) | same |
| L21 gate c132 | 6% / 6% | **a** (R2 1.00): a in {72..77} | same | (reads) | same |
| L21 gate c116 | 3% / 3% | **a** (R2 1.00): a in {10..12} | same | (reads) | same |
| L21 gate c87 | 2% / 2% | **a** (R2 1.00): a in {51..52} | same | (reads) | same |
| L21 gate c76 | 7% / 7% | **a** (R2 1.00): a in {64..70} | same | (reads) | same |
| L21 down c213 | 2% / 2% | **a** (R2 1.00): a in {98..99} | same | - | same |
| L21 down c160 | 6% / 6% | **a** (R2 1.00): a in {26..31} | same | - | same |
| L21 down c132 | 5% / 5% | **a** (R2 1.00): a in {73..77} | same | - | same |
| L21 down c229 | 6% / 6% | **a** (R2 1.00): a in {67..72} | same | - | same |
| L21 down c219 | 2% / 2% | **a** (R2 1.00): a in {2..3} | same | - | same |
| L21 down c422 | 2% / 2% | **a** (R2 1.00): a in {30..31} | same | - | same |
| L21 gate c34 | 10% / 10% | **a** (R2 1.00): a in {45..54} | same | (reads) | same |
| L21 down c326 | 2% / 2% | **a** (R2 1.00): a in {13..14} | same | - | same |
| L21 down c110 | 11% / 11% | **a** (R2 1.00): a in {21..31} | same | - | same |
| L21 down c37 | 3% / 3% | **a** (R2 1.00): a in {5..7} | same | - | same |
| L21 down c34 | 14% / 14% | **a** (R2 1.00): a in {45..58} | same | - | same |
| L21 down c152 | 3% / 3% | **a** (R2 1.00): a in {45..47} | same | - | same |
| L21 down c18 | 19% / 19% | **a** (R2 1.00): a in {28..46} | **a** (R2 0.99): a in {28..46} | - | same |
| L22 v c56 (kv0) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {56..57} | (reads) | same |
| L22 v c84 (kv3) | 0 / 7% | off (on 0) | **a** (R2 1.00): a in {90..96} | (reads) | same |
| L22 v c82 (kv0) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {56..57} | (reads) | same |
| L22 v c38 (kv3) | 1% / 5% | unexplained (best R2 0.15) | **a** (R2 1.00): a in {15..19} | (reads) | same |
| L22 up c176 | 8% / 8% | **a** (R2 1.00): a in {16..23} | same | (reads) | same |
| L22 gate c617 | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | (reads) | same |
| L22 gate c453 | 2% / 2% | **a** (R2 1.00): a in {92..93} | same | (reads) | same |
| L22 gate c292 | 9% / 9% | **a** (R2 1.00): a in {5..13} | same | (reads) | same |
| L22 gate c263 | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | (reads) | same |
| L22 gate c229 | 5% / 5% | **a** (R2 1.00): a in {32..36} | same | (reads) | same |
| L22 down c640 | 2% / 2% | **a** (R2 1.00): a in {8..9} | same | - | same |
| L22 down c242 | 4% / 4% | **a** (R2 1.00): a in {2..5} | same | - | same |
| L22 down c176 | 11% / 11% | **a** (R2 1.00): a in {13..23} | same | - | same |
| L22 down c148 | 4% / 4% | **a** (R2 1.00): a in {54..57} | same | - | same |
| L22 down c120 | 5% / 5% | **a** (R2 1.00): a in {72..76} | same | - | same |
| L22 down c93 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | - | same |
| L22 down c74 | 15% / 15% | **a** (R2 1.00): a in {1..15} | same | - | same |
| L22 down c49 | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | - | same |
| L22 down c24 | 8% / 8% | **a** (R2 0.99): a in {91..98} | **a** (R2 1.00): a in {91..98} | - | same |
| L22 v c212 (kv3) | 0 / 8% | off (on 0) | **a** (R2 1.00): a in {20..27} | (reads) | same |
| L22 v c196 (kv3) | 6% / 11% | **a** (R2 0.84): a in {16, 18, 20..22} | **a** (R2 0.94): a in {14..23} | (reads) | same |
| L22 v c140 (kv3) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {22..24} | (reads) | same |
| L22 v c138 (kv3) | 0 / 2% | off (on 0) | **a** (R2 1.00): a in {22..23} | (reads) | same |
| L22 v c113 (kv3) | 3% / 12% | unexplained (best R2 0.42) | **a** (R2 1.00): a in {40..51} | (reads) | same |
| L23 gate c237 | 2% / 2% | **a** (R2 1.00): a in {11..12} | same | (reads) | same |
| L23 gate c78 | 11% / 11% | **a** (R2 1.00): a in {1..11} | same | (reads) | same |
| L23 gate c79 | 2% / 2% | **a** (R2 1.00): a in {20..21} | same | (reads) | same |
| L23 down c239 | 2% / 2% | **a** (R2 1.00): a in {18..19} | same | - | same |
| L23 down c237 | 2% / 2% | **a** (R2 1.00): a in {11..12} | same | - | same |
| L23 down c284 | 6% / 6% | **a** (R2 1.00): a in {26..31} | same | - | same |
| L23 down c554 | 4% / 4% | **a** (R2 1.00): a in {97..100} | same | - | same |
| L23 gate c96 | 5% / 5% | **a** (R2 1.00): a in {85..89} | same | (reads) | same |
| L23 up c185 | 2% / 2% | **a** (R2 1.00): a in {99..100} | same | (reads) | same |
| L23 v c131 (kv3) | 9% / 9% | **a** (R2 1.00): a in {91..99} | same | (reads) | same |
| L23 v c55 (kv3) | 5% / 9% | **a** (R2 0.93): a in {88..92} | **a** (R2 0.94): a in {85..92} | (reads) | same |
| L23 up c56 | 15% / 15% | **a** (R2 1.00): a in {35..49} | same | (reads) | same |
| L23 gate c288 | 6% / 6% | **a** (R2 1.00): a in {26..31} | same | (reads) | same |
| L23 down c150 | 14% / 14% | **a** (R2 1.00): a in {1..14} | same | - | same |
| L23 down c96 | 2% / 3% | **a** (R2 0.90): a in {88..89} | **a** (R2 1.00): a in {87..89} | - | same |
| L23 down c79 | 2% / 2% | **a** (R2 1.00): a in {20..21} | same | - | same |
| L23 down c49 | 7% / 7% | **a** (R2 1.00): a in {32..38} | same | - | same |
| L23 down c47 | 3% / 3% | **a** (R2 1.00): a in {1..3} | same | - | same |
| L23 down c31 | 8% / 8% | **a** (R2 1.00): a in {92..99} | same | - | same |
| L23 v c51 (kv3) | 7% / 8% | **a** (R2 0.98): a in {62..68} | **a** (R2 0.95): a in {62..68} | (reads) | same |
| L23 up c74 | 6% / 6% | **a** (R2 1.00): a in {47..52} | same | (reads) | same |
| L23 up c73 | 4% / 4% | **a** (R2 1.00): a in {90..93} | same | (reads) | same |
| L24 down c32 | 29% / 29% | **a** (R2 1.00): a in {3..31} | same | a: mod100 +2% | a: mod100 +2% |
| L24 down c33 | 19% / 19% | **a** (R2 1.00): a in {32..50} | same | - | same |
| L24 gate c97 | 8% / 8% | **a** (R2 1.00): a in {49..56} | same | (reads) | same |
| L24 up c589 | 8% / 8% | **a** (R2 1.00): a in {45..52} | same | (reads) | same |
| L24 up c61 | 11% / 11% | **a** (R2 1.00): a in {21..31} | same | (reads) | same |
| L24 gate c32 | 30% / 30% | **a** (R2 1.00): a in {2..31} | same | (reads) | same |
| L24 down c227 | 4% / 4% | **a** (R2 1.00): a in {90..93} | same | - | same |
| L24 down c35 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | - | same |
| L24 down c115 | 17% / 17% | **a** (R2 1.00): a in {83..99} | same | - | same |
| L24 down c213 | 3% / 3% | **a** (R2 1.00): a in {3..5} | same | - | same |
| L24 down c222 | 17% / 17% | **a** (R2 1.00): a in {1..17} | same | - | same |
| L24 down c141 | 7% / 7% | **a** (R2 1.00): a in {53..59} | same | - | same |
| L24 down c170 | 6% / 6% | **a** (R2 1.00): a in {49..54} | same | - | same |
| L24 gate c213 | 3% / 3% | **a** (R2 1.00): a in {3..5} | same | (reads) | same |
| L24 gate c61 | 6% / 6% | **a** (R2 1.00): a in {26..31} | same | (reads) | same |
| L24 gate c53 | 6% / 6% | **a** (R2 1.00): a in {95..100} | same | (reads) | same |
| L25 down c59 | 16% / 16% | **a** (R2 1.00): a in {2..17} | same | - | same |
| L25 down c70 | 12% / 12% | **a** (R2 1.00): a in {32..43} | same | - | same |
| L25 down c363 | 8% / 8% | **a** (R2 1.00): a in {1..8} | same | - | same |
| L25 down c121 | 3% / 3% | **a** (R2 1.00): a in {72..74} | same | - | same |
| L25 down c120 | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | - | same |
| L25 up c283 | 17% / 17% | **a** (R2 1.00): a in {1..17} | same | (reads) | same |
| L25 up c35 | 9% / 9% | **a** (R2 1.00): a in {32..40} | same | (reads) | same |
| L25 up c20 | 19% / 19% | **a** (R2 1.00): a in {13..31} | same | (reads) | same |
| L25 gate c20 | 4% / 4% | **a** (R2 1.00): a in {9..12} | same | (reads) | same |
| L25 down c607 | 5% / 5% | **a** (R2 1.00): a in {47..51} | same | - | same |
| L26 down c204 | 7% / 7% | **a** (R2 1.00): a in {92..98} | same | - | same |
| L26 down c59 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | - | same |
| L26 down c16 | 24% / 24% | **a** (R2 1.00): a in {8..31} | same | - | same |
| L26 gate c501 | 15% / 15% | **a** (R2 1.00): a in {1..15} | same | (reads) | same |
| L26 gate c59 | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | (reads) | same |
| L26 down c586 | 9% / 9% | **a** (R2 1.00): a in {32..40} | same | - | same |
| L26 down c407 | 5% / 5% | **a** (R2 1.00): a in {80..84} | same | - | same |
| L26 down c252 | 5% / 5% | **a** (R2 1.00): a in {3..7} | same | - | same |
| L26 up c59 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | (reads) | same |
| L26 up c263 | 11% / 11% | **a** (R2 1.00): a in {3..13} | same | (reads) | same |
| L26 up c270 | 3% / 3% | **a** (R2 1.00): a in {26..28} | same | (reads) | same |
| L26 up c215 | 3% / 3% | **a** (R2 1.00): a in {73..75} | same | (reads) | same |
| L26 up c876 | 3% / 3% | **a** (R2 1.00): a in {88..90} | same | (reads) | same |
| L26 up c564 | 2% / 2% | **a** (R2 1.00): a in {98..99} | same | (reads) | same |
| L27 down c43 | 21% / 21% | **a** (R2 1.00): a in {32..52} | same | - | same |
| L27 down c22 | 34% / 34% | **a** (R2 1.00): a in {2..35} | same | a: mod100 +3% | a: mod100 +3% |
| L27 down c303 | 4% / 4% | **a** (R2 1.00): a in {1..4} | same | - | same |
| L27 down c197 | 7% / 7% | **a** (R2 1.00): a in {83..89} | same | - | same |
| L27 down c149 | 4% / 4% | **a** (R2 1.00): a in {49..52} | same | - | same |
| L27 gate c220 | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | (reads) | same |
| L27 gate c329 | 21% / 21% | **a** (R2 1.00): a in {11..31} | same | (reads) | same |
| L27 gate c43 | 20% / 20% | **a** (R2 1.00): a in {33..52} | same | (reads) | same |
| L27 gate c36 | 2% / 2% | **a** (R2 1.00): a in {99..100} | same | (reads) | same |
| L27 gate c26 | 13% / 13% | **a** (R2 1.00): a in {86..98} | same | (reads) | same |
| L28 up c27 | 15% / 15% | **a** (R2 1.00): a in {1..15} | same | (reads) | same |
| L28 gate c665 | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | (reads) | same |
| L28 up c4 | 15% / 15% | **a** (R2 1.00): a in {85..99} | same | (reads) | same |
| L28 gate c350 | 7% / 7% | **a** (R2 1.00): a in {49..55} | same | (reads) | same |
| L28 down c639 | 17% / 17% | **a** (R2 1.00): a in {82..98} | same | - | same |
| L28 down c627 | 10% / 10% | **a** (R2 1.00): a in {1..10} | same | - | same |
| L28 down c501 | 7% / 7% | **a** (R2 1.00): a in {53..59} | same | - | same |
| L28 down c477 | 2% / 2% | **a** (R2 1.00): a in {2..3} | same | - | same |
| L28 down c4 | 9% / 9% | **a** (R2 1.00): a in {90..98} | same | - | same |
| L28 down c96 | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | - | same |
| L28 down c77 | 3% / 3% | **a** (R2 1.00): a in {49..51} | same | - | same |
| L29 down c123 | 23% / 23% | **a** (R2 1.00): a in {9..31} | same | - | same |
| L29 down c143 | 3% / 4% | **a** (R2 1.00): a in {1..3} | **a** (R2 0.98): a in {1..4} | - | same |
| L29 down c228 | 4% / 4% | **a** (R2 1.00): a in {2..5} | same | - | same |
| L29 down c421 | 6% / 6% | **a** (R2 1.00): a in {34..39} | same | - | same |
| L29 down c114 | 24% / 24% | **a** (R2 1.00): a in {1..24} | same | a: mod100 +2%, mod50 +3% | a: mod100 +2%, mod50 +3% |
| L29 gate c242 | 14% / 14% | **a** (R2 1.00): a in {86..99} | same | (reads) | same |
| L29 up c703 | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | (reads) | same |
| L29 down c829 | 7% / 7% | **a** (R2 1.00): a in {4..10} | same | - | same |
| L29 up c123 | 26% / 26% | **a** (R2 1.00): a in {6..31} | same | (reads) | same |
| L29 up c386 | 18% / 18% | **a** (R2 1.00): a in {32..49} | same | (reads) | same |
| L30 down c991 | 14% / 14% | **a** (R2 0.98): a in {86..99} | **a** (R2 1.00): a in {86..99} | - | same |
| L30 down c142 | 2% / 2% | **a** (R2 1.00): a in {1..2} | same | - | same |
| L30 gate c241 | 2% / 2% | **a** (R2 1.00): a in {2..3} | same | (reads) | same |
| L30 up c306 | 4% / 4% | **a** (R2 0.99): a in {1..4} | **a** (R2 1.00): a in {1..4} | (reads) | same |
| L30 up c238 | 18% / 18% | **a** (R2 1.00): a in {82..99} | same | (reads) | same |
| L30 up c422 | 13% / 13% | **a** (R2 1.00): a in {3..15} | same | (reads) | same |
| L30 v c23 (kv7) | 16% / 16% | **a** (R2 1.00): a in {3..18} | same | (reads) | same |
| L30 down c55 | 17% / 15% | **a** (R2 0.96): a in {1..16} | **a** (R2 1.00): a in {1..15} | a: mod50 +3% | a: mod50 +3% |
| L31 v c111 (kv5) | 2% / 2% | **a** (R2 1.00): a in {66..67} | same | (reads) | same |

</details>

## On-set depends on the operator

<details><summary>all 268</summary>

| component | on (add/sub) | on-set (add) | on-set (sub) |
|---|---|---|---|
| L8 k c72 (kv4) | 1% / 23% | unexplained (best R2 0.04) | unexplained (best R2 0.37) |
| L14 v c131 (kv7) | 29% / 47% | unexplained (best R2 0.49) | **a** (R2 0.88): a in {4, 31..49, 51..55, 61..63, 65, 74, 81..94, 96} |
| L18 v c124 (kv7) | 0 / 17% | off (on 0) | **a** (R2 1.00): a in {9, 19, 21, 29, 31, 39, 41, 47, 49, 51, 59, 61, 69, 79, 81, 89, 99} [coarser: a mod 20 in {1, 9, 19}, R2 0.80] |
| L18 v c107 (kv7) | 1% / 15% | unexplained (best R2 0.39) | **a** (R2 1.00): a in {12, 20..25, 32, 42..44, 52, 62, 72, 82} |
| L18 v c98 (kv7) | 9% / 22% | **a%10** (R2 0.88): a mod 10 in {6} | **a** (R2 0.99): a in {6, 16..18, 24, 26..28, 36..37, 45..48, 56..57, 66..67, 76, 86..87, 96} |
| L3 v c230 (kv1) | 0 / 12% | off (on 0) | **a** (R2 1.00): a in {63, 67..74, 77..79} |
| L27 v c152 (kv1) | 0 / 12% | off (on 0) | **a** (R2 1.00): a in {3..10, 12, 16, 24, 26} |
| L5 v c201 (kv5) | 39% / 28% | **a** (R2 0.78): a in {12, 16, 22, 24, 26, 28, 32, 34, 36, 38, 40, 42, 44, 46, 48, 52, 54, 56, 58, 60, 62, 64, 66, 68, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98} [coarser: a mod 10 in {2, 4, 6, 8}, R2 0.82] | unexplained (best R2 0.48) |
| L5 v c190 (kv5) | 14% / 4% | **a** (R2 0.88): a in {4..18} | unexplained (best R2 0.21) |
| L18 v c91 (kv7) | 0 / 11% | off (on 0) | **a** (R2 0.99): a in {68..77, 79} |
| L5 v c13 (kv5) | 27% / 16% | **a//10** (R2 0.87): (tens) a in {74..100} | unexplained (best R2 0.48) |
| L22 v c239 (kv3) | 1% / 11% | **a** (R2 0.50): a in {31} | **a** (R2 0.98): a in {28..33, 52, 55..57, 85} |
| L15 k c81 (kv7) | 10% / 0 | **a** (R2 0.60): a in {86..88, 92..93, 96..98} | off (on 0) |
| L18 v c233 (kv7) | 2% / 12% | unexplained (best R2 0.32) | **a** (R2 1.00): a in {18..29} |
| L18 v c49 (kv7) | 2% / 11% | **a** (R2 0.54): a in {62, 82} | **a** (R2 0.99): a in {12, 22, 32, 42, 52, 61..62, 72, 82..83, 92} [coarser: a mod 50 in {12, 22, 32, 42}, R2 0.85] |
| L22 v c113 (kv3) | 3% / 12% | unexplained (best R2 0.42) | **a** (R2 1.00): a in {40..51} |
| L3 v c180 (kv1) | 3% / 12% | **a** (R2 0.99): a in {11, 51, 53} | **a** (R2 0.99): a in {11, 31, 41, 43, 46..47, 49, 51..53, 61, 81} |
| L5 v c48 (kv5) | 24% / 15% | **a** (R2 0.91): a in {66..89} | unexplained (best R2 0.46) |
| L22 v c15 (kv3) | 1% / 10% | **a** (R2 0.70): a in {6} | **a%10** (R2 1.00): a mod 10 in {6} |
| L22 v c192 (kv3) | 6% / 14% | **a** (R2 0.56): a in {34, 36..38} | **a** (R2 1.00): a in {30..42, 45} |
| L18 v c78 (kv7) | 1% / 9% | unexplained (best R2 0.13) | **a** (R2 0.98): a in {16, 24, 32, 36, 48, 52, 56, 64, 72} |
| L3 k c135 (kv1) | 9% / 1% | **a** (R2 1.00): a in {21, 50..52, 55..57, 62, 65} | **a** (R2 0.96): a in {55} |
| L18 v c211 (kv1) | 0 / 8% | off (on 0) | **a** (R2 1.00): a in {49, 51..57} |
| L3 k c80 (kv1) | 0 / 8% | off (on 0) | **a** (R2 0.90): a in {20, 41..42, 61..62, 70, 81} |
| L22 v c212 (kv3) | 0 / 8% | off (on 0) | **a** (R2 1.00): a in {20..27} |
| L22 v c18 (kv0) | 0 / 8% | off (on 0) | **a** (R2 1.00): a in {22..23, 25..29, 33} |
| L2 k c62 (kv2) | 10% / 2% | **a** (R2 1.00): a in {11, 16..18, 21, 55, 86, 90..92} | **a** (R2 0.65): a in {18, 70} |
| L14 v c46 (kv4) | 5% / 13% | **a** (R2 0.96): a in {78, 82, 84, 86, 88} | **a** (R2 0.99): a in {78, 80..90, 92} |
| L5 v c11 (kv5) | 25% / 17% | **a//10** (R2 0.80): (tens) a in {53..54, 56..79} | unexplained (best R2 0.48) |
| L18 v c202 (kv7) | 1% / 8% | unexplained (best R2 0.10) | **a** (R2 1.00): a in {33..40} |
| L22 v c172 (kv3) | 10% / 17% | **a** (R2 0.85): a in {24..31} | **a** (R2 1.00): a in {21..31, 33..37, 45} |
| L18 v c234 (kv7) | 2% / 9% | **a** (R2 0.65): a in {50} | **a** (R2 1.00): a in {25, 46..53} |
| L22 v c84 (kv3) | 0 / 7% | off (on 0) | **a** (R2 1.00): a in {90..96} |
| L22 v c194 (kv3) | 0 / 7% | off (on 0) | **a** (R2 1.00): a in {26..29, 87..89} |
| L14 v c173 (kv4) | 3% / 10% | **a** (R2 0.92): a in {70, 80, 90} | **a%10** (R2 1.00): a mod 10 in {0} |
| L18 v c239 (kv7) | 0 / 7% | off (on 0) | **a%50** (R2 0.92): a mod 50 in {23..25} |
| L14 v c125 (kv4) | 18% / 24% | **a** (R2 0.87): a in {1..14, 20, 100} | **a** (R2 0.98): a in {1..16, 20, 30, 40, 50, 60, 80, 90, 100} |
| L20 v c244 (kv0) | 7% / 13% | **a** (R2 0.56): a in {12, 24, 48, 96} | **a** (R2 0.76): a in {6, 8..9, 12, 16, 24, 32, 36, 48, 64, 72, 96} |
| L20 v c172 (kv0) | 4% / 10% | unexplained (best R2 0.44) | **a%10** (R2 0.92): a mod 10 in {5} |
| L20 v c206 (kv0) | 4% / 11% | **a** (R2 0.56): a in {32, 64} | **a** (R2 0.69): a in {8, 16, 20, 32, 40, 64, 80, 88, 96} |
| L8 v c248 (kv6) | 85% / 91% | **a** (R2 0.64): a in {9, 11, 13..49, 51, 53..99} | **a** (R2 0.79): a in {8..99} |
| L3 v c131 (kv1) | 1% / 7% | **a** (R2 1.00): a in {62} | **a** (R2 0.96): a in {58..63, 71} |
| L18 v c23 (kv7) | 2% / 8% | **a** (R2 0.67): a in {70} | **a** (R2 1.00): a in {67..74} |
| L22 v c101 (kv3) | 4% / 10% | **a** (R2 0.63): a in {52, 55} | **a** (R2 1.00): a in {45, 50..57, 60} |
| L20 v c73 (kv0) | 1% / 8% | **a** (R2 0.81): a in {31} | **a** (R2 0.96): a in {21, 31, 41, 51, 61, 71, 81, 91} [coarser: a mod 50 in {21, 31, 41}, R2 0.83] |
| L16 v c40 (kv5) | 22% / 16% | **a%10** (R2 0.85): a mod 10 in {6, 8} | **a%20** (R2 0.74): a mod 20 in {6, 8, 16} [coarser: a mod 10 in {6, 8}, R2 0.90] |
| L18 v c25 (kv7) | 5% / 11% | **a** (R2 0.78): a in {64..67} | **a//10** (R2 0.91): (tens) a in {56, 60..69} |
| L0 k c25 (kv6) | 6% / 0 | **a** (R2 1.00): a in {18, 50, 55, 65, 90, 94} | off (on 0) |
| L18 v c147 (kv7) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {58..63} |
| L18 v c146 (kv1) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {65, 70..71, 73..75} |
| L18 v c230 (kv7) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {19..20, 78..81} |
| L5 v c55 (kv5) | 18% / 12% | **a** (R2 0.90): a in {12..28} | **a//10** (R2 0.53): (tens) a in {18, 20..29} |
| L18 v c128 (kv7) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {15..19, 52} |
| L22 v c167 (kv3) | 0 / 6% | off (on 0) | **a** (R2 1.00): a in {17, 19..23} |
| L16 v c103 (kv5) | 27% / 21% | **a** (R2 0.94): a in {7..14, 30..34, 50..54, 70..74, 90..94} [coarser: a mod 20 in {10..14}, R2 0.89] | **a%20** (R2 0.78): a mod 20 in {10..14} |
| L3 v c211 (kv1) | 11% / 17% | **a** (R2 0.99): a in {16..24, 26..27} | **a** (R2 0.99): a in {13..29} |
| L18 v c139 (kv7) | 8% / 14% | **a** (R2 0.93): a in {72..79} | **a** (R2 1.00): a in {69..82} |
| L20 v c108 (kv0) | 7% / 12% | **a** (R2 0.54): a in {30, 60, 90} | **a%50** (R2 0.75): a mod 50 in {10, 20, 30, 40, 45} [coarser: a mod 10 in {0}, R2 0.86] |
| L22 v c191 (kv3) | 1% / 6% | unexplained (best R2 0.09) | **a** (R2 1.00): a in {79..83, 85} |
| L22 v c130 (kv3) | 10% / 16% | **a** (R2 0.79): a in {15, 25, 30, 35, 40, 45, 55, 65, 75, 85} | **a** (R2 0.95): a in {15, 25, 30, 32..38, 40, 45, 50, 55, 85} |
| L3 v c207 (kv1) | 6% / 12% | **a** (R2 0.99): a in {95..100} | **a** (R2 0.90): a in {10..13, 15, 93..100} |
| L13 gate c297 | 9% / 14% | **a** (R2 0.94): a in {93..100} | **a** (R2 0.94): a in {74..77, 79, 92..100} |
| L5 v c165 (kv5) | 3% / 8% | unexplained (best R2 0.19) | unexplained (best R2 0.30) |
| L18 v c35 (kv1) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {47, 49..51, 53} |
| L0 k c43 (kv6) | 5% / 0 | **a** (R2 1.00): a in {18, 50, 55, 90, 94} | off (on 0) |
| L18 v c21 (kv7) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {15, 35, 55, 65, 75} |
| L18 v c121 (kv7) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {13..17} |
| L18 v c188 (kv7) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {24, 47, 49, 51, 53} |
| L18 v c251 (kv7) | 0 / 5% | off (on 0) | **a** (R2 1.00): a in {25, 74..77} |
| L22 v c220 (kv0) | 1% / 6% | unexplained (best R2 0.10) | **a** (R2 1.00): a in {72, 76..77, 79..81} |
| L18 v c176 (kv7) | 8% / 13% | **a** (R2 0.83): a in {36..42} | **a** (R2 1.00): a in {32..44} |
| L1 down c337 | 15% / 20% | **a** (R2 0.98): a in {83, 86..99} | **a** (R2 1.00): a in {78, 81..99} |
| L18 v c90 (kv7) | 12% / 16% | **a//10** (R2 0.87): (tens) a in {39..49} | **a** (R2 1.00): a in {34, 36..50} |
| L5 v c33 (kv5) | 14% / 9% | **a%10** (R2 0.77): a mod 10 in {0} | **a** (R2 0.62): a in {20, 25, 30, 40, 50, 60, 70, 80, 100} [coarser: a mod 10 in {0}, R2 0.82] |
| L18 v c157 (kv7) | 11% / 15% | **a** (R2 0.80): a in {23..34} | **a** (R2 0.98): a in {22..35, 37} |
| L22 v c196 (kv3) | 6% / 11% | **a** (R2 0.84): a in {16, 18, 20..22} | **a** (R2 0.94): a in {14..23} |
| L22 v c38 (kv3) | 1% / 5% | unexplained (best R2 0.15) | **a** (R2 1.00): a in {15..19} |
| L20 v c85 (kv0) | 1% / 6% | **a** (R2 0.74): a in {52} | **a** (R2 0.82): a in {32, 48, 52..53, 92} |
| L20 v c221 (kv0) | 2% / 6% | **a** (R2 0.77): a in {9, 99} | **a** (R2 0.88): a in {9, 49, 69, 97..99} |
| L18 v c226 (kv7) | 1% / 5% | unexplained (best R2 0.16) | **a** (R2 1.00): a in {25..29} |
| L3 v c195 (kv1) | 13% / 17% | **a** (R2 0.98): a in {86, 88..99} | **a** (R2 0.95): a in {10..13, 86..99} |
| L1 v c94 (kv5) | 11% / 7% | **a** (R2 0.86): a in {6, 13, 16, 23, 43, 53, 56, 66, 73, 93, 96} [coarser: a mod 50 in {6, 13, 16, 23, 43, 46}, R2 0.82] | **a** (R2 0.67): a in {6, 13, 16, 23, 43, 53, 56} |
| L22 v c89 (kv3) | 5% / 9% | **a** (R2 0.96): a in {4, 24, 48, 72, 96} | **a** (R2 0.97): a in {12, 16, 23..24, 26, 36, 48, 72, 96} |
| L24 v c232 (kv5) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {24, 61..63} |
| L3 v c225 (kv1) | 2% / 6% | **a** (R2 0.99): a in {48, 50} | **a** (R2 1.00): a in {45..50} |
| L18 v c61 (kv7) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {42..45} |
| L16 v c79 (kv0) | 10% / 6% | **a%10** (R2 0.97): a mod 10 in {8} | **a** (R2 0.68): a in {28, 58, 68, 78, 88, 98} [coarser: a mod 10 in {8}, R2 0.81] |
| L18 v c86 (kv7) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {30..33} |
| L22 v c78 (kv3) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {10, 20, 40, 50} |
| L18 v c212 (kv7) | 0 / 4% | off (on 0) | **a** (R2 1.00): a in {20..23} |
| L16 v c68 (kv5) | 19% / 15% | **a** (R2 0.95): a in {1..4, 55..65, 80..84} | **a** (R2 0.85): a in {1, 55..65, 80..84} |
| L23 k c129 (kv1) | 2% / 6% | **a** (R2 0.92): a in {88, 98} | **a** (R2 1.00): a in {28, 58, 68, 78, 88, 98} |
| L22 v c110 (kv3) | 1% / 5% | **a** (R2 0.82): a in {3} | **a** (R2 0.99): a in {13, 23, 33, 83, 93} |
| L3 v c112 (kv1) | 6% / 10% | **a** (R2 1.00): a in {1..4, 11, 13} | **a** (R2 0.98): a in {1..4, 10..15} |
| L18 v c227 (kv7) | 9% / 13% | **a** (R2 0.80): a in {30..38} | **a** (R2 1.00): a in {28..40} |
| L16 v c80 (kv5) | 28% / 25% | **a%10** (R2 0.94): a mod 10 in {0, 3, 7} | **a%10** (R2 0.86): a mod 10 in {0, 3, 7} |
| L18 v c28 (kv7) | 10% / 13% | **a** (R2 0.88): a in {79..86, 89} | **a** (R2 1.00): a in {76, 78..89} |
| L1 v c138 (kv6) | 11% / 14% | **a** (R2 0.96): a in {26..36} | **a** (R2 0.97): a in {24..37} |
| L18 v c246 (kv7) | 15% / 19% | **a%10** (R2 0.81): a mod 10 in {0, 5} [coarser: a mod 5 in {0}, R2 0.84] | **a%5** (R2 0.92): a mod 5 in {0} |
| L16 v c121 (kv5) | 30% / 27% | **a%10** (R2 0.99): a mod 10 in {7..9} | **a%10** (R2 0.87): a mod 10 in {7..9} |
| L16 v c43 (kv5) | 26% / 23% | **a%20** (R2 0.95): a mod 20 in {15..19} | **a** (R2 0.93): a in {15, 35..39, 55..59, 75..79, 95..100} [coarser: a mod 20 in {15..19}, R2 0.86] |
| L16 v c82 (kv0) | 16% / 13% | **a** (R2 0.93): a in {9..25} | **a** (R2 0.88): a in {12..24} |
| L23 v c55 (kv3) | 5% / 9% | **a** (R2 0.93): a in {88..92} | **a** (R2 0.94): a in {85..92} |
| L1 v c78 (kv5) | 17% / 14% | **a** (R2 0.96): a in {1..4, 21, 31..32, 41, 51, 61..62, 71..72, 81, 91..92, 98} | **a** (R2 0.88): a in {1..4, 21, 31..32, 41, 51, 61..62, 81, 91..92} |
| L16 v c25 (kv5) | 40% / 37% | **a** (R2 0.99): a in {4, 24..52, 65..67, 85..91} | **a** (R2 0.94): a in {24..52, 85..91} |
| L1 up c271 | 14% / 17% | **a** (R2 1.00): a in {50..52, 55, 80..89} | **a** (R2 1.00): a in {49..52, 54..55, 79..89} |
| L18 v c80 (kv7) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {60..62} |
| L18 v c60 (kv7) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {24, 48, 64} |
| L18 v c245 (kv7) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {60..62} |
| L3 k c89 (kv3) | 3% / 0 | **a** (R2 1.00): a in {11, 21, 55} | off (on 0) |
| L22 v c140 (kv3) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {22..24} |
| L22 v c37 (kv3) | 7% / 10% | **a** (R2 0.91): a in {82..88} | **a//10** (R2 1.00): (tens) a in {80..89} |
| L14 up c677 | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {24..26} |
| L18 v c89 (kv7) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {20, 70, 80} [coarser: a mod 50 in {20}, R2 0.83] |
| L23 k c142 (kv1) | 1% / 4% | **a** (R2 0.91): a in {98} | **a** (R2 1.00): a in {58, 68, 88, 98} |
| L14 v c221 (kv4) | 7% / 10% | **a** (R2 0.98): a in {11, 41, 51, 61, 71, 81, 91} | **a** (R2 1.00): a in {11, 21, 31, 41, 51, 53, 61, 71, 81, 91} [coarser: a mod 10 in {1}, R2 0.80] |
| L22 v c199 (kv3) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {17..18, 88} |
| L11 down c294 | 6% / 9% | **a** (R2 0.99): a in {1..6} | **a//10** (R2 0.99): (tens) a in {1..9} |
| L2 v c69 (kv6) | 18% / 21% | **a//10** (R2 0.89): (tens) a in {81..84, 86..99} | **a** (R2 1.00): a in {78..79, 81..99} |
| L2 k c114 (kv2) | 3% / 0 | **a** (R2 1.00): a in {11, 18, 21} | off (on 0) |
| L5 v c39 (kv5) | 3% / 5% | unexplained (best R2 0.19) | unexplained (best R2 0.30) |
| L3 down c68 | 38% / 35% | **a%50** (R2 0.91): a mod 50 in {4, 9, 13..14, 17, 19, 23..24, 27, 29, 33..34, 37, 39, 43..44, 49} [coarser: a mod 10 in {3..4, 7, 9}, R2 0.86] | **a%50** (R2 0.92): a mod 50 in {4, 9, 13..14, 17, 19, 23..24, 27, 29, 33..34, 37, 39, 43..44, 49} [coarser: a mod 10 in {3..4, 7, 9}, R2 0.84] |
| L22 v c114 (kv3) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {21, 41..42} |
| L5 v c152 (kv5) | 12% / 15% | unexplained (best R2 0.47) | unexplained (best R2 0.20) |
| L15 down c426 | 3% / 6% | **a** (R2 0.98): a in {42, 52, 72} | **a** (R2 0.95): a in {32, 42, 52, 72, 82, 92} |
| L20 v c240 (kv0) | 0 / 3% | off (on 0) | **a** (R2 1.00): a in {29..31} |
| L5 v c23 (kv5) | 24% / 22% | **a** (R2 0.88): a in {24..48} | **a//10** (R2 0.64): (tens) a in {24..49} |
| L1 v c116 (kv5) | 8% / 11% | **a%50** (R2 0.84): a mod 50 in {1, 11, 21, 31, 41} [coarser: a mod 10 in {1}, R2 0.85] | **a%50** (R2 0.85): a mod 50 in {1, 9, 11, 21, 31, 41} [coarser: a mod 10 in {1}, R2 0.88] |
| L22 k c92 (kv0) | 6% / 9% | **a** (R2 0.93): a in {47, 57, 67, 77, 87, 97} | **a%20** (R2 0.90): a mod 20 in {7, 17} [coarser: a mod 10 in {7}, R2 0.89] |
| L5 v c109 (kv5) | 1% / 4% | unexplained (best R2 0.14) | unexplained (best R2 0.37) |
| L18 v c38 (kv7) | 26% / 28% | **a//10** (R2 0.86): (tens) a in {2..26} | **a//10** (R2 0.96): (tens) a in {2..29} |
| L1 o c70 (H26) | 78% / 80% | **a** (R2 1.00): a in {8..12, 21..32, 40..100} | **a** (R2 0.98): a in {7..12, 20..32, 40..100} |
| L5 v c75 (kv5) | 7% / 10% | unexplained (best R2 0.40) | unexplained (best R2 0.29) |
| L13 gate c108 | 19% / 21% | **a** (R2 0.97): a in {14..31} | **a** (R2 1.00): a in {11..31} |
| L16 v c29 (kv0) | 25% / 23% | **a** (R2 1.00): a in {1..4, 32..49, 82..84} | **a** (R2 0.95): a in {4, 32..49, 82..84} |
| L16 v c22 (kv0) | 12% / 10% | **a%50** (R2 0.90): a mod 50 in {1, 11, 21, 31, 41} [coarser: a mod 10 in {1}, R2 0.84] | **a%10** (R2 0.89): a mod 10 in {1} |
| L16 v c122 (kv0) | 10% / 7% | **a%10** (R2 0.95): a mod 10 in {5} | **a%10** (R2 0.71): a mod 10 in {5} |
| L1 v c71 (kv6) | 15% / 17% | **a** (R2 0.96): a in {1..12, 18, 21} | **a** (R2 0.99): a in {1..12, 18..22} |
| L16 v c30 (kv0) | 24% / 22% | **a** (R2 0.97): a in {57..79, 81} | **a//10** (R2 0.87): (tens) a in {58..79} |
| L16 v c26 (kv0) | 4% / 2% | **a** (R2 0.72): a in {4, 14, 84, 94} | unexplained (best R2 0.30) |
| L12 down c31 | 30% / 27% | **a%10** (R2 0.98): a mod 10 in {5..7} | **a%10** (R2 0.91): a mod 10 in {5..7} |
| L13 up c460 | 10% / 8% | **a%10** (R2 1.00): a mod 10 in {9} | **a** (R2 0.93): a in {9, 19, 29, 39, 49, 59, 69, 99} [coarser: a mod 10 in {9}, R2 0.83] |
| L3 v c159 (kv1) | 2% / 4% | **a** (R2 0.97): a in {52..53} | **a** (R2 0.97): a in {51..54} |
| L3 v c133 (kv1) | 1% / 3% | **a** (R2 1.00): a in {33} | **a** (R2 0.98): a in {32..34} |
| L22 v c109 (kv3) | 13% / 15% | **a** (R2 0.92): a in {86..98} | **a** (R2 0.97): a in {80, 85..97} |
| L23 v c29 (kv3) | 6% / 8% | **a** (R2 0.90): a in {20, 25, 50, 75, 100} [coarser: a mod 25 in {0}, R2 0.82] | **a** (R2 0.93): a in {20, 25, 40, 50, 75, 80, 100} [coarser: a mod 50 in {0, 20, 25, 30, 40}, R2 0.80] |
| L3 v c149 (kv1) | 1% / 3% | **a** (R2 0.97): a in {74} | **a** (R2 0.99): a in {70, 73..74} |
| L22 v c165 (kv3) | 9% / 11% | **a** (R2 0.92): a in {92..100} | **a** (R2 0.99): a in {25, 50, 75, 77, 94..100} |
| L18 v c130 (kv7) | 10% / 12% | **a%10** (R2 1.00): a mod 10 in {9} | **a%50** (R2 0.90): a mod 50 in {9, 19, 29, 39, 49} [coarser: a mod 10 in {9}, R2 0.83] |
| L3 v c48 (kv1) | 21% / 23% | **a//10** (R2 1.00): (tens) a in {80..100} | **a//10** (R2 0.91): (tens) a in {78..100} |
| L14 down c393 | 2% / 4% | **a** (R2 0.76): a in {21, 41} | **a** (R2 1.00): a in {21, 41..43} |
| L2 down c6 | 18% / 20% | **a** (R2 1.00): a in {62..79} | **a** (R2 1.00): a in {59, 61..79} |
| L14 down c129 | 3% / 5% | **a** (R2 0.97): a in {44, 55..56} | **a** (R2 1.00): a in {22, 35, 44, 55..56} |
| L2 down c271 | 12% / 10% | **a** (R2 1.00): a in {50..59, 62, 65} | **a** (R2 0.99): a in {51..59, 62} |
| L14 down c42 | 15% / 17% | **a** (R2 0.99): a in {12, 18, 22, 32, 38, 42..43, 48, 58, 62, 72, 78, 82..83, 98} | **a** (R2 1.00): a in {12, 18, 22, 32, 38, 42..43, 48, 52, 58, 62..63, 72, 78, 82..83, 98} |
| L12 down c550 | 2% / 4% | **a** (R2 0.92): a in {57..58} | **a** (R2 0.86): a in {55..58} |
| L1 v c21 (kv5) | 11% / 13% | **a//10** (R2 0.89): (tens) a in {49..59} | **a** (R2 0.99): a in {47..59} |
| L2 down c102 | 11% / 13% | **a** (R2 1.00): a in {12, 18, 24, 36, 42, 45, 48, 60, 72, 90, 96} | **a** (R2 0.99): a in {12, 18, 24, 30, 36, 42, 45, 48, 54, 60, 72, 90, 96} |
| L1 v c157 (kv5) | 14% / 16% | **a** (R2 0.94): a in {6, 56, 60..69, 86} | **a** (R2 0.89): a in {6, 56, 59..69, 86} |
| L30 down c55 | 17% / 15% | **a** (R2 0.96): a in {1..16} | **a** (R2 1.00): a in {1..15} |
| L1 v c82 (kv5) | 14% / 16% | **a** (R2 0.98): a in {76..89} | **a** (R2 0.98): a in {8, 75..89} |
| L1 v c171 (kv6) | 25% / 27% | **a** (R2 0.98): a in {4..28} | **a** (R2 1.00): a in {4..30} |
| L22 v c231 (kv3) | 5% / 6% | **a** (R2 0.90): a in {31, 41, 51, 91} | **a** (R2 0.99): a in {21, 31, 41, 51, 81, 91} [coarser: a mod 50 in {31, 41}, R2 0.82] |
| L16 v c238 (kv0) | 3% / 1% | **a** (R2 0.91): a in {45, 65, 85} | **a** (R2 0.57): a in {65, 85} |
| L16 v c100 (kv0) | 20% / 19% | **a** (R2 0.93): a in {10, 20, 40, 58..73, 80, 90} | **a** (R2 0.90): a in {40, 58..73, 80, 90} |
| L16 down c50 | 13% / 15% | **a** (R2 0.99): a in {16, 21, 26, 31, 41, 46, 51, 61, 71, 76, 81, 86, 91} | **a** (R2 0.97): a in {11, 16, 21, 26, 31, 41, 46, 51, 61, 71, 76, 81, 86, 91, 96} [coarser: a mod 50 in {11, 16, 21, 26, 31, 41, 46}, R2 0.87] |
| L1 v c85 (kv6) | 1% / 2% | **a** (R2 0.57): a in {24} | **a** (R2 0.91): a in {24..25} |
| L5 gate c218 | 8% / 9% | **a** (R2 0.99): a in {10, 30, 40, 50, 52, 60, 99..100} | **a** (R2 0.94): a in {10, 20, 30, 40, 50, 52, 60, 99..100} |
| L20 o c0 (H17) | 89% / 90% | **a** (R2 1.00): a in {1..4, 6, 9, 11..14, 16..19, 21..29, 31..39, 41..49, 51..79, 81..99} | **a** (R2 0.96): a in {1..4, 6, 9, 11..14, 16..29, 31..39, 41..49, 51..79, 81..99} |
| L1 v c233 (kv6) | 2% / 3% | **a** (R2 0.95): a in {49..50} | **a** (R2 0.76): a in {48..50} |
| L2 v c194 (kv6) | 30% / 31% | **a//10** (R2 1.00): (tens) a in {20..49} | **a//10** (R2 0.95): (tens) a in {20..50} |
| L16 v c28 (kv5) | 4% / 3% | **a** (R2 0.73): a in {10, 50, 98, 100} | **a** (R2 0.54): a in {50, 100} [coarser: a mod 50 in {0}, R2 0.84] |
| L2 down c61 | 4% / 5% | **a** (R2 1.00): a in {14, 21, 28, 42} | **a** (R2 0.97): a in {14, 21..22, 28, 42} |
| L16 v c23 (kv0) | 13% / 12% | **a** (R2 0.93): a in {18..30} | **a//10** (R2 0.85): (tens) a in {19..30} |
| L30 down c259 | 14% / 13% | **a** (R2 0.99): a in {14, 16, 18..20, 22..30} | **a** (R2 1.00): a in {14, 16, 18..20, 22..29} |
| L2 up c271 | 11% / 10% | **a//10** (R2 0.90): (tens) a in {50..59, 62} | **a** (R2 0.99): a in {51..59, 62} |
| L3 v c205 (kv1) | 1% / 2% | **a** (R2 0.99): a in {55} | **a** (R2 0.99): a in {43..44} |
| L11 up c50 | 10% / 11% | **a** (R2 1.00): a in {83, 85..93} | **a** (R2 1.00): a in {83..93} |
| L3 v c75 (kv1) | 1% / 2% | **a** (R2 1.00): a in {55} | **a** (R2 0.99): a in {55, 77} |
| L2 up c11 | 23% / 24% | **a** (R2 1.00): a in {27..49} | **a** (R2 1.00): a in {27..50} |
| L30 down c63 | 52% / 51% | **a//10** (R2 0.90): (tens) a in {1..50, 54, 92} | **a//10** (R2 0.94): (tens) a in {1..50, 54} |
| L11 down c89 | 8% / 9% | **a** (R2 1.00): a in {31..38} | **a** (R2 1.00): a in {31..39} |
| L28 down c405 | 8% / 9% | **a** (R2 1.00): a in {92..99} | **a** (R2 1.00): a in {88, 92..99} |
| L13 down c774 | 4% / 5% | **a** (R2 1.00): a in {14..17} | **a** (R2 1.00): a in {13..17} |
| L15 down c104 | 6% / 7% | **a** (R2 1.00): a in {40, 60..61, 70, 80, 90} | **a** (R2 1.00): a in {30, 40, 60..61, 70, 80, 90} |
| L15 down c854 | 6% / 7% | **a** (R2 1.00): a in {81..86} | **a** (R2 1.00): a in {80..86} |
| L4 up c69 | 6% / 7% | **a** (R2 1.00): a in {53..58} | **a** (R2 1.00): a in {53..59} |
| L18 gate c40 | 9% / 10% | **a%20** (R2 0.90): a mod 20 in {9, 19} [coarser: a mod 10 in {9}, R2 0.89] | **a%10** (R2 1.00): a mod 10 in {9} |
| L2 up c476 | 5% / 6% | **a** (R2 1.00): a in {70..74} | **a** (R2 1.00): a in {70..75} |
| L0 up c265 | 27% / 28% | **a** (R2 1.00): a in {10..36} | **a//10** (R2 0.92): (tens) a in {10..37} |
| L11 down c31 | 16% / 15% | **a** (R2 1.00): a in {8..23} | **a** (R2 1.00): a in {9..23} |
| L2 down c11 | 23% / 24% | **a** (R2 1.00): a in {27..49} | **a** (R2 1.00): a in {27..50} |
| L14 gate c201 | 9% / 10% | **a** (R2 1.00): a in {7..8, 18, 28, 38, 48, 58, 78, 98} [coarser: a mod 50 in {8, 28, 48}, R2 0.82] | **a** (R2 1.00): a in {7..8, 17..18, 28, 38, 48, 58, 78, 98} |
| L1 v c183 (kv6) | 4% / 5% | **a** (R2 0.99): a in {10..11, 50, 100} | **a** (R2 1.00): a in {10..12, 50, 100} |
| L11 down c50 | 10% / 11% | **a** (R2 1.00): a in {83, 85..93} | **a** (R2 1.00): a in {83..93} |
| L12 down c17 | 17% / 16% | **a** (R2 0.99): a in {69..85} | **a** (R2 0.97): a in {70..85} |
| L14 gate c248 | 4% / 5% | **a** (R2 0.99): a in {21, 23..25} | **a** (R2 1.00): a in {21, 23..26} |
| L18 v c159 (kv7) | 23% / 24% | **a** (R2 0.99): a in {41..63} | **a** (R2 1.00): a in {41..63, 66} |
| L2 gate c617 | 6% / 7% | **a** (R2 0.99): a in {60..65} | **a** (R2 1.00): a in {59..65} |
| L11 down c223 | 2% / 1% | **a** (R2 0.99): a in {74, 76} | **a** (R2 1.00): a in {76} |
| L2 gate c282 | 3% / 2% | **a** (R2 1.00): a in {5, 55, 65} [coarser: a mod 50 in {5}, R2 0.83] | **a%50** (R2 0.99): a mod 50 in {5} |
| L3 v c179 (kv1) | 1% / 2% | **a** (R2 0.98): a in {51} | **a** (R2 1.00): a in {51, 53} |
| L0 gate c167 | 11% / 12% | **a//10** (R2 0.91): (tens) a in {49..59} | **a** (R2 1.00): a in {49..60} |
| L2 down c147 | 12% / 13% | **a%50** (R2 0.90): a mod 50 in {0, 10, 20, 30, 40} [coarser: a mod 10 in {0}, R2 0.83] | **a** (R2 1.00): a in {10, 20, 30, 40, 50, 60, 70, 80, 90, 97..100} [coarser: a mod 50 in {0, 10, 20, 30, 40}, R2 0.87] |
| L16 gate c252 | 7% / 8% | **a** (R2 1.00): a in {23..27, 48, 72} | **a** (R2 1.00): a in {23..28, 48, 72} |
| L3 v c160 (kv1) | 4% / 5% | **a** (R2 0.99): a in {41..44} | **a** (R2 1.00): a in {41..45} |
| L14 v c148 (kv4) | 2% / 3% | **a** (R2 0.99): a in {85, 87} | **a** (R2 1.00): a in {24, 85, 87} |
| L31 v c219 (kv5) | 9% / 10% | **a%20** (R2 0.90): a mod 20 in {0, 10} [coarser: a mod 10 in {0}, R2 0.89] | **a** (R2 1.00): a in {20, 30, 40, 45, 50, 60, 70, 80, 90, 100} [coarser: a mod 10 in {0}, R2 0.80] |
| L11 gate c27 | 19% / 18% | **a%10** (R2 0.94): a mod 10 in {5..6} | **a%50** (R2 0.93): a mod 50 in {5, 15..16, 25..26, 35..36, 45..46} [coarser: a mod 10 in {5..6}, R2 0.90] |
| L14 v c67 (kv4) | 1% / 2% | **a** (R2 0.94): a in {28} | **a** (R2 1.00): a in {28..29} |
| L29 down c143 | 3% / 4% | **a** (R2 1.00): a in {1..3} | **a** (R2 0.98): a in {1..4} |
| L2 down c140 | 1% / 2% | **a** (R2 1.00): a in {83} | **a** (R2 0.96): a in {83..84} |
| L14 down c787 | 1% / 2% | **a** (R2 0.93): a in {5} | **a** (R2 1.00): a in {5, 95} |
| L23 v c216 (kv3) | 1% / 2% | **a** (R2 0.86): a in {50} | **a** (R2 0.95): a in {25, 50} |
| L3 gate c29 | 18% / 19% | **a** (R2 0.99): a in {56..72, 74} | **a** (R2 1.00): a in {56..74} |
| L16 down c30 | 7% / 8% | **a** (R2 0.99): a in {24, 32, 36, 48, 64, 72, 96} | **a** (R2 0.99): a in {16, 24, 32, 36, 48, 64, 72, 96} |
| L16 gate c50 | 15% / 16% | **a** (R2 1.00): a in {1, 11, 16, 21, 26, 31, 41, 46, 51, 61, 71, 76, 81, 86, 91} [coarser: a mod 10 in {1, 6}, R2 0.81] | **a%50** (R2 0.92): a mod 50 in {1, 11, 16, 21, 26, 31, 41, 46} [coarser: a mod 10 in {1, 6}, R2 0.83] |
| L3 down c122 | 11% / 12% | **a//10** (R2 0.99): (tens) a in {90..100} | **a//10** (R2 0.92): (tens) a in {88, 90..100} |
| L2 down c934 | 1% / 2% | **a** (R2 1.00): a in {42} | **a** (R2 0.95): a in {42, 44} |
| L16 v c31 (kv5) | 14% / 15% | **a** (R2 0.99): a in {65..78} | **a** (R2 1.00): a in {65..79} |
| L16 v c90 (kv0) | 2% / 1% | **a** (R2 0.98): a in {4, 64} | **a** (R2 0.89): a in {64} |
| L0 down c273 | 2% / 1% | **a** (R2 1.00): a in {9, 90} | **a** (R2 0.89): a in {9} |
| L12 up c295 | 7% / 8% | **a** (R2 1.00): a in {92..98} | **a** (R2 0.98): a in {92..99} |
| L30 down c948 | 2% / 1% | **a** (R2 1.00): a in {3, 33} | **a** (R2 0.88): a in {3} |
| L12 gate c7 | 27% / 26% | **a%50** (R2 0.92): a mod 50 in {6, 9, 12, 16, 19, 26, 29, 32, 36, 39, 42, 46, 49} [coarser: a mod 10 in {2, 6, 9}, R2 0.89] | **a%50** (R2 0.90): a mod 50 in {6, 9, 16, 19, 26, 29, 32, 36, 39, 42, 46, 49} [coarser: a mod 10 in {2, 6, 9}, R2 0.86] |
| L14 up c290 | 10% / 9% | **a** (R2 1.00): a in {32, 34..42} | **a** (R2 0.98): a in {32, 35..42} |
| L17 gate c110 | 6% / 7% | **a** (R2 0.97): a in {10, 50, 97..100} | **a** (R2 1.00): a in {10, 25, 50, 97..100} |
| L2 v c173 (kv2) | 2% / 1% | **a** (R2 0.99): a in {18, 21} | **a** (R2 0.85): a in {18} |
| L3 v c101 (kv1) | 8% / 9% | **a** (R2 0.98): a in {13..19, 21} | **a** (R2 1.00): a in {13..21} |
| L2 v c247 (kv2) | 2% / 1% | **a** (R2 0.99): a in {21, 70} | **a** (R2 0.82): a in {70} |
| L2 gate c28 | 87% / 88% | **a** (R2 0.98): a in {1..66, 68..69, 71..89} | **a//10** (R2 0.91): (tens) a in {1..69, 71..89} |
| L16 down c820 | 1% / 2% | **a** (R2 1.00): a in {75} | **a%50** (R2 0.88): a mod 50 in {25} |
| L3 v c187 (kv1) | 1% / 2% | **a** (R2 1.00): a in {24} | **a** (R2 0.75): a in {24, 48} |
| L1 gate c69 | 14% / 15% | **a** (R2 1.00): a in {86..99} | **a** (R2 0.99): a in {85..99} |
| L1 down c271 | 12% / 13% | **a** (R2 0.98): a in {55, 79..89} | **a** (R2 1.00): a in {52, 55, 79..89} |
| L1 v c64 (kv5) | 8% / 7% | **a** (R2 0.99): a in {86, 88..94} | **a** (R2 0.95): a in {86, 89..94} |
| L14 gate c448 | 4% / 5% | **a** (R2 0.95): a in {73..76} | **a** (R2 1.00): a in {25, 73..76} |
| L16 v c192 (kv0) | 28% / 27% | **a//10** (R2 0.90): (tens) a in {40..67} | **a//10** (R2 0.90): (tens) a in {40..66} |
| L22 v c41 (kv3) | 13% / 14% | **a** (R2 0.92): a in {4, 6..17} | **a** (R2 0.97): a in {4, 6..18} |
| L5 v c46 (kv5) | 18% / 18% | **a** (R2 0.77): a in {41..54, 56..59} | **a** (R2 0.56): a in {37, 39, 41..59} |
| L16 v c66 (kv5) | 8% / 8% | **a** (R2 0.94): a in {92, 94..100} | **a** (R2 0.94): a in {94..100} |
| L1 v c213 (kv6) | 1% / 2% | **a** (R2 1.00): a in {1} | **a%50** (R2 0.80): a mod 50 in {1} |
| L13 up c46 | 5% / 6% | **a** (R2 1.00): a in {31..34, 52} | **a** (R2 0.95): a in {31..35, 52} |
| L14 down c96 | 3% / 4% | **a** (R2 0.93): a in {16..17, 64} | **a** (R2 1.00): a in {16..17, 32, 64} |
| L7 down c150 | 3% / 4% | **a** (R2 0.99): a in {19, 22..23} | **a** (R2 0.92): a in {18..19, 22..23} |
| L20 down c331 | 8% / 7% | **a** (R2 0.97): a in {30, 40, 50, 60, 70, 80, 88, 90} | **a** (R2 0.99): a in {40, 50, 60, 70, 80, 88, 90} |
| L2 down c313 | 6% / 7% | **a** (R2 1.00): a in {15..18, 32, 64} | **a** (R2 0.96): a in {14..18, 32, 64} |
| L3 v c228 (kv1) | 17% / 17% | **a** (R2 0.98): a in {15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100} [coarser: a mod 5 in {0}, R2 0.81] | **a** (R2 0.93): a in {20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 100} [coarser: a mod 5 in {0}, R2 0.81] |
| L28 gate c213 | 6% / 7% | **a** (R2 0.96): a in {95..100} | **a** (R2 1.00): a in {88, 95..100} |
| L23 down c96 | 2% / 3% | **a** (R2 0.90): a in {88..89} | **a** (R2 1.00): a in {87..89} |
| L18 up c51 | 18% / 18% | **a%5** (R2 0.91): a mod 5 in {0} | **a%25** (R2 0.90): a mod 25 in {0, 5, 10, 15, 20} [coarser: a mod 5 in {0}, R2 0.88] |
| L13 down c545 | 1% / 2% | **a** (R2 0.83): a in {52} | **a** (R2 0.97): a in {32, 52} |
| L2 v c160 (kv2) | 10% / 10% | **a//10** (R2 1.00): (tens) a in {50..59} | **a//10** (R2 0.94): (tens) a in {50, 52..59} |
| L1 v c117 (kv5) | 8% / 7% | **a** (R2 0.95): a in {2, 12, 32, 52, 62, 72, 82, 92} [coarser: a mod 20 in {2, 12}, R2 0.81] | **a%50** (R2 0.88): a mod 50 in {2, 12, 32} |
| L16 down c513 | 3% / 3% | **a** (R2 0.92): a in {32, 82, 92} [coarser: a mod 50 in {32}, R2 0.80] | **a** (R2 0.98): a in {32, 82, 92} [coarser: a mod 50 in {32}, R2 0.83] |
| L29 gate c160 | 11% / 11% | **a** (R2 0.98): a in {20, 30, 40, 45, 50, 60, 70, 75, 80, 90, 100} [coarser: a mod 50 in {0, 20, 30, 40}, R2 0.84] | **a** (R2 1.00): a in {20, 30, 40, 45, 50, 60, 70, 75, 80, 90, 100} [coarser: a mod 50 in {0, 20, 30, 40}, R2 0.85] |
| L29 down c877 | 19% / 19% | **a%25** (R2 0.89): a mod 25 in {0, 10, 15, 20} [coarser: a mod 5 in {0}, R2 0.86] | **a%25** (R2 0.90): a mod 25 in {0, 10, 15, 20} [coarser: a mod 5 in {0}, R2 0.85] |
| L20 gate c66 | 18% / 18% | **a//10** (R2 0.90): (tens) a in {82..99} | **a** (R2 1.00): a in {82..99} |
| L16 down c120 | 12% / 12% | **a%50** (R2 0.98): a mod 50 in {0, 10, 20, 25, 30, 40} [coarser: a mod 10 in {0}, R2 0.84] | **a%50** (R2 1.00): a mod 50 in {0, 10, 20, 25, 30, 40} [coarser: a mod 10 in {0}, R2 0.85] |
| L18 down c40 | 9% / 9% | **a%10** (R2 0.90): a mod 10 in {9} | **a%20** (R2 0.90): a mod 20 in {9, 19} [coarser: a mod 10 in {9}, R2 0.89] |
| L3 v c13 (kv1) | 9% / 9% | **a%20** (R2 0.90): a mod 20 in {1, 11} [coarser: a mod 10 in {1}, R2 0.89] | **a%10** (R2 0.87): a mod 10 in {1} |
| L30 down c994 | 11% / 11% | **a%50** (R2 0.93): a mod 50 in {0, 10, 20, 25, 30, 40} | **a%50** (R2 0.94): a mod 50 in {0, 20, 25, 30, 40} |
| L24 down c174 | 18% / 18% | **a%50** (R2 0.93): a mod 50 in {0, 10, 15, 20, 25, 30, 35, 40} [coarser: a mod 5 in {0}, R2 0.88] | **a%25** (R2 0.90): a mod 25 in {0, 5, 10, 15, 20} [coarser: a mod 5 in {0}, R2 0.89] |
| L18 v c54 (kv7) | 9% / 9% | **a%10** (R2 0.88): a mod 10 in {1} | **a%20** (R2 0.90): a mod 20 in {1, 11} [coarser: a mod 10 in {1}, R2 0.89] |
| L16 gate c120 | 16% / 16% | **a** (R2 1.00): a in {5, 10, 15, 20, 25, 30, 35, 40, 49..50, 55, 60, 70, 75, 80, 100} [coarser: a mod 50 in {0, 5, 10, 20, 25, 30}, R2 0.85] | **a** (R2 1.00): a in {5, 10, 15, 20, 25, 30, 35, 40, 49..50, 55, 60, 70, 75, 80, 100} [coarser: a mod 50 in {0, 5, 10, 20, 25, 30, 40}, R2 0.85] |
| L5 down c28 | 18% / 18% | **a** (R2 0.99): a in {10, 15, 20, 25, 30, 35, 40, 45, 50, 52, 55, 60, 70, 75, 80, 90, 99..100} [coarser: a mod 50 in {0, 5, 10, 20, 25, 30, 35, 40}, R2 0.80] | **a** (R2 1.00): a in {10, 15, 20, 25, 30, 35, 40, 45, 50, 52, 55, 60, 70, 75, 80, 90, 99..100} |
| L9 down c64 | 36% / 36% | **a%50** (R2 0.91): a mod 50 in {9, 11, 13, 17, 19, 21, 23, 29, 31, 33, 37, 39, 41, 43, 47, 49} [coarser: a mod 10 in {1, 3, 7, 9}, R2 0.85] | **a%50** (R2 0.91): a mod 50 in {9, 11, 13, 17, 19, 21, 23, 27, 29, 31, 33, 37, 39, 41, 43, 47, 49} [coarser: a mod 10 in {1, 3, 7, 9}, R2 0.85] |
| L9 gate c64 | 36% / 36% | **a%50** (R2 0.91): a mod 50 in {9, 11, 13, 17, 19, 21, 23, 29, 31, 33, 37, 39, 41, 43, 47, 49} [coarser: a mod 10 in {1, 3, 7, 9}, R2 0.85] | **a%50** (R2 0.91): a mod 50 in {9, 11, 13, 17, 19, 21, 23, 27, 29, 31, 33, 37, 39, 41, 43, 47, 49} [coarser: a mod 10 in {1, 3, 7, 9}, R2 0.85] |

</details>

