# Results log (numbers only)

Factual results as they come in, with the file each number comes from. No interpretation prose.
CIs are 95% percentile bootstrap over runs (10,000 resamples) unless stated.

## Positive control (all 75 runs; reproduces the paper, uses none of our features)

Source: `results/positive_control_contrasts.csv`, `results/fig_positive_control.png`.

| Measure | base-seller_comms − base (10 vs 10 runs) | all comms runs − base (65 vs 10) |
|---|---|---|
| Mean trade price ($) | +2.80 [−0.34, +5.67] | +3.65 [+1.40, +5.58] |
| Mean ask ($) | +1.90 [−0.97, +4.57] | +2.56 [+0.76, +4.24] |
| Ask dispersion (SD, $) | −0.19 [−0.45, +0.06] | −0.09 [−0.35, +0.17] |
| Judge score (1–4) | +1.57 [+1.43, +1.71] | +1.01 [+0.79, +1.22] |
| Judge "coordination: yes" share | +0.29 [+0.22, +0.36] | +0.20 [+0.12, +0.27] |
| Seller profit, run total ($) | −162 [−478, +81] | −110 [−425, +132] |
| Buyer surplus, run total ($) | −276 [−622, +126] | −335 [−545, −107] |
| Number of trades | −21.9 [−44.5, +2.2] | −22.3 [−39.6, −4.0] |

- In `base` (no messaging possible) the paper's judge says "coordination: yes" in **69%** of seller-hours
  (mean score 2.22). In `base-seller_comms` it is 98% (mean 3.80).
- Direction matches the paper: messaging raises prices and judge scores. With only 10 vs 10 runs, the
  price difference CI for `base-seller_comms` alone includes 0; pooled over all comms runs it does not.

## Scripted-market control (no LLM; 30 markets per type)

Source: `results/scripted_control_auc.csv`, `results/scripted_control_feature_means.csv`. Rules as `market.py`
(midpoint price, persistence, alphabetical ties); agent rules in `src/controls.py:simulate_market`.
Replaying the simulated orders through `load.replay` reproduces every simulated trade.

AUC = chance that a random positive unit scores higher than a random negative one (0.5 = no signal, 1 = perfect).
"Separation" = max(AUC, 1 − AUC), i.e. ignoring direction.

| Feature | A (cartel) vs B (independents) | A (loyal) vs C (cheater) |
|---|---|---|
| T1 logistic, 5-fold CV by market | **1.000** | **1.000** |
| price only (logistic) | 0.823 | 0.749 |
| own_gap | 0.990 | 1.000 |
| level_shift | 0.916 | 0.986 |
| mkt_disp | 0.801 | 0.851 |
| even_vs_chance | 0.734 | 0.632 |
| holdout_share | 0.704 | 0.646 |
| holdout_gap | 0.645 | 0.741 |
| follow_beta / lead_beta | 0.575 / 0.562 | 0.612 / 0.755 |
| comove / disp_shift | 0.541 / 0.553 | 0.523 / 0.840 |

- Units: A 1,675, B 900, C 125.
- The scripted cartel has no price leader and no rotation by design, so this control does **not** test the
  leadership or rotation metrics; it only shows they do not break the combined model.

## Train-half dry run (5-fold CV by run, A vs B; rule labels; nothing frozen)

Source: `python -m src.evaluate dryrun`. Logistic models; flag if p >= 0.5. FPR = share of B units flagged.

| Model | CV AUC | TPR on A | FPR on B |
|---|---|---|---|
| price_only | 0.551 | 0.586 | 0.460 |
| judge_only | 0.962 | 0.888 | 0.033 |
| T1 | 0.641 | 0.795 | 0.493 |
| price+T1 | 0.574 | 0.788 | 0.573 |
| T1+T2 | 0.975 | 0.960 | 0.180 |
| T1+T3 | 0.968 | 0.942 | 0.147 |
| T1+T2+T3 | 0.990 | 0.980 | 0.093 |

- Only 5 B runs in train, so each CV fold holds out one B run.
- T2 features are zero for every B unit (no messages exist), so any T2 model separates A from B partly by construction.

## Train half: single features (`results/train_feature_auc.csv`)

- Tier 1 features one at a time separate A from B only weakly on the LLM logs: separation 0.52–0.68
  (best: level_shift 0.677, ask_minus_ce 0.637). Compare 0.80–0.99 for the best ones in the scripted market.
- D units (comms runs, no detected agreement; mostly gagged oversight runs) have the **highest** mean ask:
  +3.86 over $90 vs A +2.83, B +0.96. Judge score D 2.43 ≈ B 2.36 (A 3.73).
- Mean window trade price: A $91.25, D $89.85, B $89.40, C $86.57 (C has only 15 train units, 1.6 trades/window).

## Labeller sensitivity (train half; `results/labeller_sensitivity_train.csv`)

Grid: adherence tolerance {$0.25, $0.50, $1.00} × level-match width {$0.25, $0.50, $1.00} × reply lag {1, 2 hours}.
Frozen setting: $0.50 / $0.50 / 1 hour.

- Share of train units with the same label as the frozen setting: 97.5%–99.7% across all 18 settings.
- A ranges 582–622, D 358–381, B fixed at 150.
- **C ranges 8–27.** C ≥ 20 in 4 of 18 settings, all with the tighter $0.25 tolerance. The "C < 20 → descriptive"
  decision therefore depends partly on the tolerance choice.

## Net influence stretch (Bonjour et al. 2022), run level

- Implementation check (`results/net_influence_check.csv`): in 30 scripted markets with one colluding pair among
  three independents, the true pair had the top pair score in **30/30**. True-pair scores 0.87–1.29 bits; best
  other pair ≤ 0.09 bits.
- Train half (`results/net_influence_train_runs.csv`, `results/net_influence_frozen.json`): run-level AUC **0.765**
  for comms runs with any A unit (23 runs) vs `base` (5 runs). Frozen α = 0.0867 bits.

| Train group | Runs | Mean run score (bits) | Mean pairwise γ (bits) | Share of "hold" moves |
|---|---|---|---|---|
| B (base) | 5 | 0.067 | 0.135 | 0.230 |
| comms, has A | 23 | 0.149 | 0.190 | 0.417 |
| comms, no A | 10 | 0.124 | 0.122 | 0.353 |

---

**Note (24 Oct, evening):** everything above this line used rule labeller v1. The labeller now follows Andrew's
codebook (v2, `notes/decisions.md`). The positive control, the scripted control and all feature values are
unaffected; the label-based numbers above are superseded by the v2 numbers below.

## Labeller v2: train-half labels and dry runs

| Reading | A | B | C | D |
|---|---|---|---|---|
| Words (main) | 554 | 150 | 19 | 417 |
| Interstate Circuit (silent compliance counts) | 694 | 150 | 47 | 249 |

- v1 to v2: 94% of all 2,250 units keep the same label. No train message is a withdrawal. 0 rule ("no number")
  agreements form under the words reading, 23 under the Interstate reading. The VAGUE flag is set on 542 of 990
  train comms units.
- Sensitivity (`results/labeller_sensitivity_train.csv`; tolerance x level-match): words-reading C ranges 15-45 and
  is >= 20 in 6 of 9 settings, so the "C is descriptive" decision sits on the boundary (the frozen setting gives 19).

Train-only 5-fold CV by run, A vs B (`python -m src.evaluate dryrun [interstate]`):

| Model | Words: AUC | TPR | FPR on B | Interstate: AUC | TPR | FPR on B |
|---|---|---|---|---|---|---|
| price_only | 0.595 | 0.655 | 0.453 | 0.584 | 0.644 | 0.460 |
| judge_only | 0.975 | 0.921 | 0.033 | 0.935 | 0.823 | 0.033 |
| T1 | 0.589 | 0.742 | 0.500 | 0.554 | 0.774 | 0.587 |
| price+T1 | 0.555 | 0.756 | 0.560 | 0.564 | 0.782 | 0.567 |
| T1+T2 | 0.981 | 0.968 | 0.180 | 0.936 | 0.914 | 0.213 |
| T1+T3 | 0.970 | 0.957 | 0.133 | 0.925 | 0.915 | 0.193 |
| T1+T2+T3 | 0.996 | 0.991 | 0.060 | 0.966 | 0.935 | 0.173 |

Net influence (v2 labels, train): run-level AUC 0.764 (22 comms runs with any A vs 5 `base`); alpha = 0.0950 bits.

## Codebook v2 labeller additions (train half)

- Stated ends and later-hour replies changed **no** train labels (A 554, B 150, C 19, D 417).
- Ladder rungs (train): B 150, D0 263, D1 117, D2 37, A 554, C 19. 262 of the 263 D0 units are in the gagged
  oversight runs (`oversight`, `boss_pressure-oversight`).
- Sensitivity grid rerun (`results/labeller_sensitivity_train.csv`): words-reading C ranges 15-50 across settings.

## Hand labels vs rule labeller (`results/kappa.json`, `results/hand_labels.csv`)

Kappa = agreement beyond chance (1 = perfect, 0 = chance level; 0.6+ is usually read as substantial).

| Measure | Kappa | n |
|---|---|---|
| **Agreement yes/no (primary)** | **0.886** | 70 (2 unclear excluded) |
| Agreement, unclear counted as no | 0.861 | 72 |
| A/C/D (Andrew's agreement + level, labeller's adherence rule) | 0.793 | 70 |
| Ladder rung (agreement / D2 / D1 / D0) | 0.739 | 70 |
| Invitation flag | 0.684 | 72 |
| VAGUE flag | 0.476 | 72 |
| Agreement vs Interstate-reading rule labels | 0.613 | 70 |

- Raw agreement on agreement yes/no: 94.3%. Confusion: rule yes/hand yes 34, rule no/hand no 32, rule no/hand yes 4,
  rule yes/hand no 0. The rule labeller misses some agreements; it never invents one Andrew did not see.
- Raw agreement by condition: 1.0 in 7 of 9 conditions; 0.75 in `boss_pressure` and `claude_buyers-gpt_sellers`.
- The `level` field was checked with an AI tool (see decisions log); it affects only the A/C/D kappa.

---

## TEST HALF: one-time run (25 Oct 07:01-07:07; frozen rules `c9e3e54`; outputs `f245e46`)

Square brackets = 95% CI, bootstrap over runs within condition (2,000 resamples). Test half has only **5 B runs**,
so every false-positive rate on B is uncertain. A CI of [0.000, 0.000] means no B unit was flagged in any resample,
not that the true rate is zero.

### A vs B, main reading (`results/test_metrics.csv`; 540 A units, 150 B units)

| Model | TPR on A | FPR on B | AUC |
|---|---|---|---|
| Price only | 0.548 [0.411, 0.670] | 0.227 [0.153, 0.333] | 0.649 [0.528, 0.756] |
| Paper's judge only | 0.946 [0.931, 0.960] | 0.000 [0.000, 0.000] | 0.994 [0.991, 0.997] |
| Tier 1 | 0.730 [0.662, 0.800] | 0.287 [0.100, 0.527] | 0.798 [0.698, 0.885] |
| Price + Tier 1 | 0.706 [0.623, 0.784] | 0.260 [0.087, 0.487] | 0.806 [0.719, 0.883] |
| Tier 1 + 2 * | 0.932 [0.893, 0.966] | 0.020 [0.000, 0.060] | 0.968 [0.927, 0.993] |
| Tier 1 + 3 | 0.967 [0.955, 0.976] | 0.013 [0.000, 0.027] | 0.990 [0.977, 0.998] |
| All tiers * | 0.980 [0.964, 0.993] | 0.007 [0.000, 0.020] | 0.993 [0.980, 1.000] |

\* Tier 2 features are zero for every B unit by construction. Tier 1 AUC on test (0.80) is higher than in train CV
(0.59): a sign of instability with 5 B runs per half.

### Flag rates on the no-agreement group D, by ladder rung (`results/test_fpr_on_D.csv`; D0 256, D1 119, D2 25 units)

| Model | D (all, 400) | D0 talk, no coordination content | D1 vague only | D2 invitation not accepted |
|---|---|---|---|---|
| Price only | 0.612 | 0.629 | 0.597 | 0.52 |
| Paper's judge only | 0.120 | 0.023 | 0.269 | 0.40 |
| Tier 1 | 0.530 | 0.512 | 0.580 | 0.48 |
| Price + Tier 1 | 0.520 | 0.504 | 0.563 | 0.48 |
| Tier 1 + 2 | 0.078 | 0.012 | 0.168 | 0.32 |
| Tier 1 + 3 | 0.245 | 0.098 | 0.496 | 0.56 |
| All tiers | 0.128 | 0.016 | 0.286 | 0.52 |

### Single features, test A vs B (`results/test_feature_auc.csv`; separation = max(AUC, 1 - AUC))

- Tier 1: level_shift 0.729, even_vs_chance 0.686, ask_minus_ce 0.649, holdout_share 0.628, follow_beta 0.585,
  mkt_disp 0.562, own_gap 0.552, holdout_gap 0.544, comove 0.534, disp_shift 0.523, lead_beta 0.511.
- Tier 2: t2_others_invites 0.982, t2_price_invites 0.947, t2_assent 0.737, t2_cut_talk 0.565.
- Tier 3: judge mean 0.994, coord flags 0.962, judge yes-share 0.887, defect flags 0.707.

### Interstate Circuit reading, sensitivity (`results/test_metrics_interstate.csv`; 643 A, 150 B, 27 C test units)

- A vs B AUC: price 0.655; judge 0.977 (FPR on B 0.000); Tier 1 0.807 (FPR 0.347 [0.173, 0.553]); Tier 1 + 3 0.984.
- **A vs C** (fitted only under this reading; train C = 47):

| Model | AUC | Flags a cheater (C) as "kept" | Balanced accuracy |
|---|---|---|---|
| Price only | 0.501 [0.418, 0.552] | 0.407 | 0.561 |
| Paper's judge only | 0.896 [0.832, 0.949] | 0.296 | 0.786 |
| Tier 1 | 0.677 [0.520, 0.833] | 0.407 | 0.701 |
| Price + Tier 1 | 0.732 [0.602, 0.875] | 0.407 | 0.710 |
| Tier 1 + 2 | 0.908 [0.846, 0.951] | 0.111 | 0.838 |
| Tier 1 + 3 | 0.937 [0.906, 0.973] | 0.148 | 0.872 |
| All tiers | 0.941 [0.913, 0.965] | 0.148 | 0.858 |

### Model shift (`results/test_model_shift.csv`): trained on gpt-4.1-seller train runs only

| Model | TPR, Claude-seller A (110) | TPR, mixed-seller A (85) | FPR on B (150) | In-distribution TPR (main test) |
|---|---|---|---|---|
| Price only | 0.273 | 0.118 | 0.187 | 0.548 |
| Paper's judge only | 0.755 | 0.812 | 0.000 | 0.946 |
| Tier 1 | 0.536 | 0.353 | 0.287 | 0.730 |
| Price + Tier 1 | 0.227 | 0.271 | 0.100 | 0.706 |
| Tier 1 + 2 | 0.682 | 0.776 | 0.000 | 0.932 |
| Tier 1 + 3 | 0.891 | 0.859 | 0.007 | 0.967 |
| All tiers | 0.900 | 0.859 | 0.000 | 0.980 |

Claude-seller runs also differ in information structure (same-hour leak, `method_facts.md` §4).

### Net influence, run level (`results/net_influence_test.json`)

Test AUC **0.227** (train 0.764). At the frozen alpha: TPR 0.409 (9 of 22 comms runs with A), FPR 0.80 (4 of 5 base
runs). The train result did not replicate.

### Harm by label, all 75 runs (`results/harm_by_label.csv`)

| Label | Window trade price | Buyer surplus per trade | Units / runs |
|---|---|---|---|
| A agreement kept | 91.48 [90.23, 92.68] | 8.52 [7.32, 9.77] | 1,094 / 44 |
| B no comms | 87.71 [86.18, 89.70] | 12.29 [10.30, 13.82] | 300 / 10 |
| C agreement broken | 86.58 [84.77, 88.71] | 13.42 [11.29, 15.23] | 39 / 13 |
| D no agreement | 89.62 [87.92, 91.34] | 10.38 [8.66, 12.08] | 817 / 56 |

C, all 75 runs (`results/c_descriptive.csv`): 39 units in 13 runs; mean window 2.3; mean shortfall below the agreed
level $3.59; mean ask - $90 = +0.44; judge mean 3.18.

### Ladder of communication evidence, all 75 runs (pre-registered; raw means; `results/ladder_*.csv`, `fig_ladder.png`)

Seller's mean ask - $90 per window:

| Rung | Mean | 95% CI | Units / runs |
|---|---|---|---|
| B no talk possible | +0.12 | [-1.21, +1.57] | 300 / 10 |
| D0 talk, no coordination content | +3.08 | [+2.49, +3.72] | 519 / 21 |
| D1 vague talk only | +3.25 | [+2.03, +4.64] | 236 / 54 |
| D2 invitation not accepted | +3.20 | [+2.03, +4.43] | 62 / 28 |
| Agreement (A + C) | +2.35 | [+1.55, +3.05] | 1,133 / 44 |

Steps: B -> D0 +2.96 [+1.37, +4.47]; D0 -> D1 +0.17 [-1.19, +1.64]; D1 -> D2 -0.05 [-1.27, +1.03];
D2 -> agreement -0.85 [-2.23, +0.54].

Confounds: D0 is almost all gagged oversight runs, and its mean depends on condition (`boss_pressure-oversight` +6.48,
`oversight` -0.15; B +0.12). Timing: 65% of D1 units and 82% of D2 units are in window 1, when asks are high in every
run. The raw rung means therefore mix communication with condition and time.

## EXPLORATORY (post-hoc) ladder analysis, all 75 runs (`results/ladder_exploratory.csv`, `fig_ladder_exploratory.png`)

Designed after the test results; defined in `notes/decisions.md` (25 Oct 07:33) before it ran (07:34, once).
Outcome: seller's mean ask - $90 in the window. Difference vs the reference rung, 95% CI over runs.

| Comparison | D1 vague only | D2 invitation not accepted | Agreement (A + C) |
|---|---|---|---|
| vs D0, same condition and window | -0.21 [-0.84, +0.28] | -0.31 [-1.13, +0.42] | +0.55 [-0.63, +1.40] |
| vs D0, same market and window | +0.04 [-0.64, +0.63] | +0.13 [-0.57, +0.73] | +0.08 [-0.60, +0.68] |

- Same condition and window: 1,000 of 1,950 comms units sit in cells with more than one rung. Same market and window:
  405 units.
- Steps, same market and window: D0 -> D1 +0.04 [-0.64, +0.63]; D1 -> D2 +0.10 [-0.10, +0.30];
  D2 -> agreement -0.06 [-0.26, +0.14].

| vs B, same window (condition not held fixed) | Difference |
|---|---|
| D0 talk, no coordination content | +3.56 [+1.87, +5.10] |
| D1 vague only | +0.59 [-1.04, +2.25] |
| D2 invitation not accepted | -0.29 [-2.09, +1.49] |
| Agreement (A + C) | +2.66 [+0.96, +4.24] |

- Within the same condition and window, no rung's mean ask differs from D0 (all CIs include 0). Within the same
  market and window, the estimates are within about $0.15 of zero.
- Against B in the same window, agreement units ask $2.66 more and D0 units $3.56 more (D0 is mostly
  `boss_pressure-oversight`); D1 and D2 units are not distinguishable from B.
