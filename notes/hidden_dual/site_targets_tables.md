# addsub-L18-11-* readouts (steps: 15000, 15000, 15000, 15000, 15000)
- **baseline** trained on `['model.layers.18.*']`
- **module-out** trained on `['*.self_attn.o_proj', '*.mlp.down_proj']`
- **resid** trained on `['resid_*']`
- **down-only** trained on `['*.mlp.down_proj']`
- **mlp-only** trained on `['*.mlp.*']`

### Alive components per locus — output CI net (absolute, threshold 0.1)

| locus | C | baseline | module-out | resid | down-only | mlp-only |
|---|---|---|---|---|---|---|
| mlp.gate_proj | 1024 | 201 | 232 | 285 | 376 | 362 |
| mlp.up_proj | 1024 | 240 | 207 | 238 | 353 | 416 |
| mlp.down_proj | 1024 | 264 | 536 | 570 | 720 | 479 |
| self_attn.q_proj | 512 | 72 | 37 | 34 | 6 | 4 |
| self_attn.k_proj | 512 | 51 | 33 | 16 | 6 | 5 |
| self_attn.v_proj | 1024 | 175 | 157 | 89 | 26 | 22 |
| self_attn.o_proj | 1024 | 226 | 286 | 189 | 45 | 41 |
| **MLP subtotal** | **3072** | **705** | **975** | **1093** | **1449** | **1257** |
| **attention subtotal** | **3072** | **524** | **513** | **328** | **83** | **72** |
| **total** | **6144** | **1229** | **1488** | **1421** | **1532** | **1329** |

### Alive components per locus — hidden CI net (absolute, threshold 0.1)

| locus | C | baseline | module-out | resid | down-only | mlp-only |
|---|---|---|---|---|---|---|
| mlp.gate_proj | 1024 | 263 | 258 | 298 | 439 | 494 |
| mlp.up_proj | 1024 | 319 | 232 | 248 | 416 | 545 |
| mlp.down_proj | 1024 | 292 | 601 | 636 | 836 | 552 |
| self_attn.q_proj | 512 | 172 | 53 | 38 | 6 | 4 |
| self_attn.k_proj | 512 | 158 | 48 | 20 | 5 | 5 |
| self_attn.v_proj | 1024 | 364 | 214 | 106 | 26 | 24 |
| self_attn.o_proj | 1024 | 338 | 554 | 278 | 37 | 30 |
| **MLP subtotal** | **3072** | **874** | **1091** | **1182** | **1691** | **1591** |
| **attention subtotal** | **3072** | **1032** | **869** | **442** | **74** | **63** |
| **total** | **6144** | **1906** | **1960** | **1624** | **1765** | **1654** |

### Active components per position per locus — output CI net (absolute, CI_L0)

| locus | C | baseline | module-out | resid | down-only | mlp-only |
|---|---|---|---|---|---|---|
| mlp.gate_proj | 1024 | 5.3 | 5.8 | 6.2 | 6.9 | 6.2 |
| mlp.up_proj | 1024 | 5.8 | 5.3 | 6.0 | 6.4 | 6.3 |
| mlp.down_proj | 1024 | 6.7 | 7.9 | 8.0 | 8.4 | 7.4 |
| self_attn.q_proj | 512 | 1.5 | 1.7 | 1.5 | 1.1 | 0.9 |
| self_attn.k_proj | 512 | 1.7 | 2.0 | 1.9 | 1.0 | 0.9 |
| self_attn.v_proj | 1024 | 2.1 | 2.1 | 2.1 | 1.4 | 1.3 |
| self_attn.o_proj | 1024 | 2.3 | 2.4 | 2.4 | 1.5 | 1.3 |
| **MLP subtotal** | **3072** | **17.8** | **19.0** | **20.3** | **21.7** | **20.0** |
| **attention subtotal** | **3072** | **7.6** | **8.2** | **7.9** | **5.0** | **4.3** |
| **total** | **6144** | **25.4** | **27.2** | **28.2** | **26.7** | **24.3** |

### Active components per position per locus — hidden CI net (absolute, CI_L0)

| locus | C | baseline | module-out | resid | down-only | mlp-only |
|---|---|---|---|---|---|---|
| mlp.gate_proj | 1024 | 8.3 | 8.9 | 10.4 | 12.7 | 12.9 |
| mlp.up_proj | 1024 | 10.7 | 9.1 | 11.0 | 13.3 | 15.7 |
| mlp.down_proj | 1024 | 10.6 | 17.9 | 18.7 | 25.7 | 15.5 |
| self_attn.q_proj | 512 | 6.9 | 4.5 | 2.4 | 1.2 | 0.9 |
| self_attn.k_proj | 512 | 7.0 | 3.6 | 2.3 | 1.0 | 0.8 |
| self_attn.v_proj | 1024 | 9.5 | 6.9 | 4.4 | 1.6 | 1.3 |
| self_attn.o_proj | 1024 | 9.8 | 20.0 | 8.4 | 1.7 | 1.4 |
| **MLP subtotal** | **3072** | **29.6** | **35.9** | **40.1** | **51.7** | **44.1** |
| **attention subtotal** | **3072** | **33.2** | **35.1** | **17.5** | **5.5** | **4.4** |
| **total** | **6144** | **62.8** | **71.0** | **57.6** | **57.2** | **48.5** |

### Saturation (alive / available)

| arm | output | hidden | worst matrix (hidden) |
|---|---|---|---|
| baseline | 20.0% | 31.0% | self_attn.v_proj 35.5% |
| module-out | 24.2% | 31.9% | mlp.down_proj 58.7% |
| resid | 23.1% | 26.4% | mlp.down_proj 62.1% |
| down-only | 24.9% | 28.7% | mlp.down_proj 81.6% |
| mlp-only | 21.6% | 26.9% | mlp.down_proj 53.9% |

### Selection metric: output-PGD nats per total alive component

| arm | PGDRecon (nats) | alive output | alive hidden | alive **either** | **nats / alive-either** | recovered / alive-either |
|---|---|---|---|---|---|---|
| baseline | 0.00547 | 93 | 374 | **374** | **1.4624e-05** | 6.3284e-04 |
| module-out | 0.00602 | 78 | 406 | **406** | **1.4820e-05** | 5.8162e-04 |
| resid | 0.00692 | 91 | 367 | **367** | **1.8854e-05** | 6.4096e-04 |
| down-only | 0.00627 | 86 | 315 | **315** | **1.9889e-05** | 7.4885e-04 |
| mlp-only | 0.00596 | 82 | 298 | **298** | **1.9995e-05** | 7.9260e-04 |

Denominator is the union over both CI nets, not the sum — the two nets score the same subcomponent pool. Lower `nats / alive-either` is better.

#### Alive-either per locus (union over both CI nets)

| locus | C | baseline | module-out | resid | down-only | mlp-only |
|---|---|---|---|---|---|---|
| mlp.gate_proj | 1024 | 46 | 44 | 57 | 69 | 81 |
| mlp.up_proj | 1024 | 59 | 64 | 76 | 86 | 99 |
| mlp.down_proj | 1024 | 77 | 121 | 130 | 146 | 112 |
| self_attn.q_proj | 512 | 33 | 16 | 11 | 2 | 1 |
| self_attn.k_proj | 512 | 23 | 1 | 1 | 0 | 0 |
| self_attn.v_proj | 1024 | 40 | 4 | 1 | 0 | 0 |
| self_attn.o_proj | 1024 | 96 | 156 | 91 | 12 | 5 |
| **total** | **6144** | **374** | **406** | **367** | **315** | **298** |

### Anomaly census (magenta = output-active, hidden-inactive; cut 0.5)

| arm | magenta cells | green cells | both | magenta % of active | anomalous comps | output-only comps |
|---|---|---|---|---|---|---|
| baseline | 21854 | 1444261 | 720683 | 1.00% | 6 | 0 |
| module-out | 7750 | 1999937 | 786035 | 0.28% | 2 | 1 |
| resid | 6012 | 1328258 | 826561 | 0.28% | 2 | 1 |
| down-only | 3931 | 1390306 | 801794 | 0.18% | 1 | 0 |
| mlp-only | 9370 | 982164 | 730447 | 0.54% | 1 | 1 |

#### Magenta cells by matrix

| matrix | baseline | module-out | resid | down-only | mlp-only |
|---|---|---|---|---|---|
| mlp.gate_proj | 5299 | 1626 | 1680 | 1034 | 1901 |
| mlp.up_proj | 4093 | 2024 | 1346 | 675 | 1937 |
| mlp.down_proj | 12375 | 4061 | 2838 | 1715 | 4921 |
| self_attn.q_proj | 4 | 3 | 24 | 6 | 374 |
| self_attn.k_proj | 0 | 0 | 0 | 0 | 0 |
| self_attn.v_proj | 0 | 0 | 0 | 0 | 0 |
| self_attn.o_proj | 83 | 36 | 124 | 501 | 237 |

### Hidden reconstruction, both CI nets (relative error)

| probe | baseline | module-out | resid | down-only | mlp-only |
|---|---|---|---|---|---|
| CIHiddenActsRecon_outputCI | 0.2625 | 0.4418 | 0.459 | 0.6627 | 0.6016 |
| CIHiddenActsRecon_hiddenCI | 0.0334 | 0.3113 | 0.373 | 0.6473 | 0.5759 |
| CIHiddenActsRecon_outputCI_resid | 5.838e-05 | 5.385e-05 | 5.599e-05 | 8.981e-05 | 8.857e-05 |
| CIHiddenActsRecon_hiddenCI_resid | 1.983e-05 | 1.173e-05 | 1.511e-05 | 7.05e-05 | 7.234e-05 |

### Output quality (the ranking criterion)

| metric | baseline | module-out | resid | down-only | mlp-only |
|---|---|---|---|---|---|
| ce_kl/ce_difference_ci_masked | -0.0027893 | 0.0051285 | -0.0022003 | 0.0035446 | -0.00023956 |
| ce_kl/ce_difference_random_masked | 0.11413 | 0.11748 | 0.11945 | 0.11868 | 0.12187 |
| ce_kl/ce_difference_rounded_masked | -0.0039276 | 0.0021774 | -0.0039383 | 0.0015686 | -0.0012039 |
| ce_kl/ce_difference_stoch_masked | 0.0016861 | 0.0029312 | 0.0013184 | 0.0058136 | 0.0031281 |
| ce_kl/ce_difference_unmasked | -0.00060883 | 0.0043106 | 0.0037735 | 0.0014618 | 0.0044601 |
| ce_kl/ce_unrecovered_ci_masked | -0.022942 | 0.036156 | -0.019019 | 0.024365 | -0.005533 |
| ce_kl/ce_unrecovered_random_masked | 0.82672 | 0.85317 | 0.86711 | 0.86159 | 0.88498 |
| ce_kl/ce_unrecovered_rounded_masked | -0.03134 | 0.013984 | -0.031501 | 0.009095 | -0.011835 |
| ce_kl/ce_unrecovered_stoch_masked | 0.010105 | 0.020059 | 0.0086454 | 0.041627 | 0.021761 |
| ce_kl/ce_unrecovered_unmasked | -0.0065218 | 0.029679 | 0.026348 | 0.0091716 | 0.031179 |
| ce_kl/kl_ci_masked | 0.0039531 | 0.0036362 | 0.0038499 | 0.0039759 | 0.0041212 |
| ce_kl/kl_random_masked | 0.18252 | 0.17999 | 0.18315 | 0.18338 | 0.18116 |
| ce_kl/kl_rounded_masked | 0.0037523 | 0.0034087 | 0.0036974 | 0.0037633 | 0.0038566 |
| ce_kl/kl_stoch_masked | 0.0026263 | 0.0024487 | 0.0027341 | 0.0025234 | 0.0025604 |
| ce_kl/kl_unmasked | 0.0015459 | 0.0013281 | 0.0019832 | 0.0013102 | 0.0013748 |
| ce_kl/kl_zero_masked | 0.24215 | 0.24215 | 0.24215 | 0.24215 | 0.24215 |
| loss/PGDReconLoss | 0.0054695 | 0.0060169 | 0.0069193 | 0.0062651 | 0.0059585 |
| loss/PersistentPGDReconLoss/hidden_acts | 0.021843 | 0.069227 | 0.069888 | 0.09294 | 0.10096 |
| loss/PersistentPGDReconLoss/hidden_acts/model.layers.18.mlp.down_proj | 0.00058743 | 0.00054624 | 0.00064861 | 0.00053997 | 0.00056064 |
| loss/PersistentPGDReconLoss/hidden_acts/model.layers.18.mlp.gate_proj | 0.0073432 | 0.02279 | 0.021689 | 0.017003 | 0.0064348 |
| loss/PersistentPGDReconLoss/hidden_acts/model.layers.18.mlp.up_proj | 0.0038058 | 0.016307 | 0.019141 | 0.01447 | 0.0037503 |
| loss/PersistentPGDReconLoss/hidden_acts/model.layers.18.self_attn.k_proj | 0.40891 | 0.6042 | 0.69371 | 1.3429 | 1.3145 |
| loss/PersistentPGDReconLoss/hidden_acts/model.layers.18.self_attn.o_proj | 0.00023992 | 0.00021241 | 0.00021647 | 0.00045038 | 0.00044865 |
| loss/PersistentPGDReconLoss/hidden_acts/model.layers.18.self_attn.q_proj | 0.084 | 0.43375 | 0.41293 | 0.51863 | 0.68648 |
| loss/PersistentPGDReconLoss/hidden_acts/model.layers.18.self_attn.v_proj | 0.013114 | 0.017942 | 0.014798 | 0.041466 | 0.033161 |
| loss/PersistentPGDReconLoss/output_recon | 0.0038397 | 0.0034916 | 0.0043161 | 0.0038248 | 0.0039127 |
| loss/UnmaskedReconLoss | 0.0015459 | 0.0013281 | 0.0019832 | 0.0013102 | 0.0013748 |
