# Retrospective benchmark: brader_valentino_suhay_2008

Brader, T., Valentino, N. A., & Suhay, E. (2008). What triggers public opposition to immigration? Anxiety, group cues, and immigration threat. American Journal of Political Science, 52(4), 959-978.

Seeds per arm: 100 (first seed 1000). Analyst-stated d: 0.3. Code commit: `6dc5f9efd9`. Human data SHA-256: `15afc45d23909f5f...`. Python 3.13.17, numpy 2.5.3, pandas 3.0.5.

## Human data (n = 265, focal cell n = 68)

| DV | mean | SD | focal g [95% CI] | focal p |
|---|---|---|---|---|
| immigr | 3.03 | 0.97 | 0.46 [0.18, 0.74] | 0.0007 |
| cong_mesg | 0.33 | 0.47 | 0.23 [-0.05, 0.50] | 0.1206 |
| anti_info | 0.11 | 0.31 | -0.09 [-0.37, 0.18] | 0.4938 |

Focal contrast: negative tone x Latino immigrant against the other three cells.

## Behavioral forecasting: mean over seeds (SD over seeds in parentheses)

### immigr

| Arm | mean error (human SD) | SD ratio | Wasserstein | TV | focal g | sign agrees | CI covers human g | focal decision agrees |
|---|---|---|---|---|---|---|---|---|
| Engine, nothing configured | -0.28 (0.06) | 0.99 | 0.089 | 0.154 | -0.02 (0.14) | 0.47 | 0.09 | 0.03 |
| Engine, direction declared | -0.27 (0.07) | 1.02 | 0.088 | 0.139 | 0.16 (0.16) | 0.86 | 0.40 | 0.21 |
| Engine, effects off | -0.28 (0.06) | 0.99 | 0.089 | 0.154 | -0.02 (0.14) | 0.47 | 0.09 | 0.03 |
| Engine, analyst d | -0.36 (0.07) | 1.03 | 0.116 | 0.165 | 0.30 (0.15) | 0.99 | 0.80 | 0.53 |
| Naive simulation, no effect | -0.54 (0.07) | 1.15 | 0.175 | 0.196 | -0.01 (0.15) | 0.49 | 0.07 | 0.06 |
| Naive simulation, analyst d | -0.46 (0.07) | 1.16 | 0.148 | 0.170 | 0.27 (0.15) | 0.94 | 0.73 | 0.48 |

### cong_mesg

| Arm | mean error (human SD) | SD ratio | Wasserstein | TV | focal g | sign agrees | CI covers human g | focal decision agrees |
|---|---|---|---|---|---|---|---|---|
| Engine, nothing configured | 0.36 (0.07) | 1.06 | 0.170 | 0.170 | -0.02 (0.13) | 0.45 | 0.58 | 0.96 |
| Engine, direction declared | 0.36 (0.07) | 1.06 | 0.172 | 0.172 | 0.16 (0.16) | 0.84 | 0.89 | 0.74 |
| Engine, effects off | 0.36 (0.07) | 1.06 | 0.170 | 0.170 | -0.02 (0.13) | 0.45 | 0.58 | 0.96 |
| Engine, analyst d | 0.27 (0.09) | 1.05 | 0.130 | 0.130 | 0.33 (0.19) | 0.98 | 0.89 | 0.44 |
| Naive simulation, no effect | 0.36 (0.07) | 1.06 | 0.168 | 0.168 | 0.01 (0.14) | 0.58 | 0.71 | 0.94 |
| Naive simulation, analyst d | 0.42 (0.07) | 1.06 | 0.197 | 0.197 | 0.24 (0.14) | 0.95 | 0.94 | 0.64 |

### anti_info

| Arm | mean error (human SD) | SD ratio | Wasserstein | TV | focal g | sign agrees | CI covers human g | focal decision agrees |
|---|---|---|---|---|---|---|---|---|
| Engine, nothing configured | 1.28 (0.10) | 1.60 | 0.400 | 0.400 | -0.00 (0.13) | 0.52 | 0.90 | 0.98 |
| Engine, direction declared | 1.29 (0.11) | 1.60 | 0.402 | 0.402 | 0.15 (0.15) | 0.19 | 0.63 | 0.77 |
| Engine, effects off | 1.28 (0.10) | 1.60 | 0.400 | 0.400 | -0.00 (0.13) | 0.52 | 0.90 | 0.98 |
| Engine, analyst d | 1.15 (0.11) | 1.59 | 0.358 | 0.358 | 0.33 (0.16) | 0.01 | 0.20 | 0.40 |
| Naive simulation, no effect | 1.24 (0.10) | 1.60 | 0.388 | 0.388 | 0.01 (0.14) | 0.51 | 0.89 | 0.99 |
| Naive simulation, analyst d | 1.34 (0.10) | 1.60 | 0.418 | 0.418 | 0.24 (0.13) | 0.06 | 0.33 | 0.59 |

## Correlation structure: mean absolute difference in off-diagonal Spearman correlations

| Arm | difference |
|---|---|
| Engine, nothing configured | 0.322 (0.038) |
| Engine, direction declared | 0.309 (0.036) |
| Engine, effects off | 0.322 (0.038) |
| Engine, analyst d | 0.305 (0.042) |
| Naive simulation, no effect | 0.329 (0.035) |
| Naive simulation, analyst d | 0.316 (0.035) |

## Design sensitivity: implied raw minimum detectable effect, synthetic divided by human

| Arm | immigr | cong_mesg | anti_info |
|---|---|---|---|
| Engine, nothing configured | 1.00 | 1.07 | 1.61 |
| Engine, direction declared | 1.03 | 1.07 | 1.61 |
| Engine, effects off | 1.00 | 1.07 | 1.61 |
| Engine, analyst d | 1.04 | 1.07 | 1.61 |
| Naive simulation, no effect | 1.15 | 1.06 | 1.60 |
| Naive simulation, analyst d | 1.16 | 1.06 | 1.60 |

## Null calibration: share of seeds with a significant focal contrast when no effect was configured

| DV | Engine, effects off | Naive simulation |
|---|---|---|
| immigr | 0.03 | 0.06 |
| cong_mesg | 0.04 | 0.06 |
| anti_info | 0.02 | 0.01 |

Expected value under a true null: 0.05.

## Pipeline rehearsal (one engine run, nothing configured)

| DV | column present | finite | inside human range | integer coded | levels observed |
|---|---|---|---|---|---|
| immigr | True | True | True | True | [1.0, 2.0, 3.0, 4.0] |
| cong_mesg | True | True | True | True | [0.0, 1.0] |
| anti_info | True | True | True | True | [0.0, 1.0] |

Provenance columns present: True. Engine-detected domains: ['behavioral_economics', 'consumer_behavior'].

## What the adapter had to do

- the parser reported one factor with four levels, not the 2x2 structure; the engine received the parser's factors unchanged (found: ['European'])

