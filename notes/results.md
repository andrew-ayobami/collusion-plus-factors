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
