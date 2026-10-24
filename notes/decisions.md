# Decisions log

Every analysis choice, with the time (WAT). Metric definitions go here
**before** any results are seen.

| Time (WAT) | Decision | Reason |
|---|---|---|
| 2026-10-10 15:30 | Data pinned to `andyjliu/llm-agent-collusion` commit `2bb2a7e`, cloned into `data/raw/` (gitignored). | Reproducibility; repo has no LICENSE, so raw data stays out of git. |
| 2026-10-10 15:45 | Run count is **75**, not 76 as first written in `CLAUDE.md` (brief corrected 10 Oct with Andrew's OK). Units = 75 × 5 × 6 = **2,250**. Run-count tests should expect 75. | Counted in `data/`; the brief's own table sums to 75; the paper's 11,250 seller-hours = 75 × 5 × 30. See `method_facts.md` §1. |
| 2026-10-10 15:45 | Python venv at `~/.venvs/collusion-plus-factors` (outside OneDrive), Python 3.13.3, versions pinned in `requirements.txt`. | Stops OneDrive syncing thousands of venv files. |
| 2026-10-10 16:00 | Join judge scores on the `round` field in `llm_eval_full.json`, not on list position in `collusion_metrics.json`. | Positional lists would shift if a judge call failed. `method_facts.md` §6. |
| 2026-10-10 16:00 | Do not use `gini_coefficient_profit` from `collusion_metrics.json`. | Formula bug at `metrics.py:70`. `method_facts.md` §7. |
| 2026-10-24 03:40 | Seller's ask in an hour = the ask **in force** at matching time (rebuilt by replaying the book), not just asks posted that hour. "Best bid seen" = best standing bid at the start of the hour, before anyone posts. | Matches `market.py`; replay reproduces all logged trades. The same-hour leak in Claude runs is ignored here (noted in `method_facts.md` §4). |
| 2026-10-24 03:50 | Split by run, stratified by condition, seed 20261024 (`src/split.py`). 10-run conditions 5/5; the five 5-run conditions alternate 3/2 and 2/3 train/test in sorted order. Result: 38 train runs, 37 test runs; `base` 5/5. Committed before any outcome was computed. | Brief's analysis plan step 1. |
| 2026-10-24 03:50 | Hand-labelled units will be drawn from the **train half only**. | Keeps the test half untouched until the rules are frozen. |
| 2026-10-24 04:20 | Positive control uses all 75 runs (not just train). | It reproduces the paper's published result and uses none of our features or thresholds, so it cannot leak into the frozen rules. |
