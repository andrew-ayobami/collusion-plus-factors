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
