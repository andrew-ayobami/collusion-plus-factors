# Project brief: plus factors for AI agent collusion

Andrew's project for the Apart Research **AI Collusion Research Sprint**.
Sprint: Fri 23 – Sun 25 Oct 2026. Deadline: Sun 25 Oct 23:59 AoE = **Mon 26 Oct 12:59 Lagos time (WAT)**.
Sprint page: https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25
Track 2, question 2.3: *What should count as convincing evidence of collusion?*

Andrew is a law graduate with a software background, working solo. Talk to him in plain, simple language and explain any stats briefly.

## Hard rules

1. **$0 budget, no GPU, ordinary laptop.** Default to zero LLM API calls. If a step truly needs one, ask Andrew first. Then use a free tier only (Gemini Flash key from Google AI Studio, or Groq), cache every response to disk, and set a hard call cap.
2. **Prior-work rule.** Run `date` at the start of each session. Before 23 Oct 2026, do **Phase 0 only**. Log anything built before the sprint in `PRIOR_WORK.md`, because undisclosed prior work can lead to disqualification.
3. **The report is Andrew's own writing.** The judges will not score a report that reads as AI-generated. Your job is code, numbers, tables, figures, short factual method notes in `notes/`, and checking his drafts for errors. Do not draft report prose.
4. **Data licence.** The source repo has no LICENSE file. Clone it into `data/raw/`, which is gitignored, and never commit the raw data or derived per-message text. Pin commit `2bb2a7e` and cite Agrawal et al. (2025), arXiv 2507.01413.
5. **Commit after every step** with a clear message. The git history shows what was built during the sprint.
6. Log every analysis choice, with the time, in `notes/decisions.md`. Log dual-use thoughts in `notes/dual_use.md`; the report needs a required appendix on this.

## The research idea

Antitrust law dealt with noisy evidence of collusion long ago through the **plus-factors doctrine**. Parallel conduct alone is not proof of agreement; you need extra factors. ML collusion work is rediscovering this without naming it. We turn plus factors into metrics on logged agent behaviour and test which ones separate three cases:

- **A – Agreement acted on.** Sellers agree on prices in messages and follow through.
- **B – Parallel conduct, no agreement.** Sellers can't talk, so any similar behaviour comes from shared incentives.
- **C – Agreement talk, no follow-through.** The seller endorses a deal, then undercuts it.

We sort evidence into three tiers, by what an enforcer could actually get:

1. **Public conduct:** asks, bids and trades (what a market regulator sees).
2. **Communications:** seller-to-seller messages (what a subpoena or dawn raid gets).
3. **Internal deliberations:** private reflection, plan and scratchpad (like internal memos found in discovery).

Headline questions:

- How well does Tier 1 alone detect agreement?
- What do Tiers 2 and 3 add?
- What is the false-positive rate on B? Convicting innocent parallel conduct is exactly the error the doctrine exists to prevent.

Legal nuance to keep for the discussion section: under Sherman Act §1 the agreement itself is the offence, even if a party later cheats. So C is not "innocent" in law, but it may be harmless in effect. Report results through both lenses.

## Data (verified 10 Oct 2026)

Repo: https://github.com/andyjliu/llm-agent-collusion, folder `data/`, 76 runs. Every run has 5 sellers, 5 buyers, 30 hours (rounds) and temperature 0.7.

| Condition | Seller comms | Oversight | Pressure | Seller / buyer models | Runs |
|---|---|---|---|---|---|
| base | no | no | no | gpt-4.1 / gpt-4.1 | 10 |
| base-seller_comms | yes | no | no | gpt-4.1 / gpt-4.1 | 10 |
| oversight | yes | yes | no | gpt-4.1 / gpt-4.1 | 10 |
| boss_pressure | yes | no | yes | gpt-4.1 / gpt-4.1 | 10 |
| boss_pressure-oversight | yes | yes | yes | gpt-4.1 / gpt-4.1 | 10 |
| 5 cross-model conditions | yes | no | no | claude-3.7 / gpt-4.1 / mixed | 5 each |

The only no-comms runs are the 10 `base` runs. Those are all of class B.

Files in each run folder:

- `experiment_metadata.json`: the config.
- `auction_results.md`: one fenced ````json block per hour. Keys are `round_number`, `seller_asks`, `buyer_bids`, `trades` (each `{buyer_id, seller_id, price}`), `seller_messages` and `buyer_messages`.
- `Seller <Name>.md` and `Buyer <Name>.md`: the prompt and response for each hour. Seller responses are JSON with `reflection`, `plan_for_this_hour`, `ask`, `new_memory`, `plan_for_message`, `message_to_sellers` and `scratch_pad_update`.
- `collusion_metrics.json`: the paper's metrics. It includes `competitive_equilibrium_price` (90), `joint_profit_maximization_price` (100), per-hour asks and trade prices, `hhi_volume`, profits and buyer surplus.
- `llm_eval_full.json`: the paper's AI judge (gpt-4.1-mini). For each seller and hour it gives `coordination` (yes/no), `score`, `type` (tacit/overt/both/null) and `evidence[].category` (propose, maintain, informational, withdraw, threaten).
  - This judge reads **private reasoning only**, not the messages. Treat it as a ready-made Tier 3 signal, never as ground truth.

Market rules (check them in `src/continuous_double_auction/market/market.py`):

- Seller cost is $80; buyer value is $100.
- Each agent places one order per hour.
- A trade happens when the best bid is at or above the best ask. Its price is the midpoint of the two.

Sanity check already run: the mean trade price was $87.71 in the no-comms runs and $90.52 in the matching comms runs. The parser above works on all 76 runs.

## Unit of analysis and labels (the hardest part)

**Unit:** seller × 5-hour window. That gives 76 runs × 5 sellers × 6 windows = 2,280 units. Attach market-level window features to each unit. Rows from the same run are not independent, so always split, bootstrap and compute confidence intervals **by run**.

- **B:** every unit in the `base` runs.
- **In comms runs:**
  - **Agreement:** in this window or earlier, the seller proposed or endorsed a *specific* price level or rule ("keep asks above $95"), and at least one other seller assented to it.
  - **Adherence:** the seller's asks stay at or above the agreed level, within a tolerance you set and log.
  - Agreed and adhered = **A**. Agreed and not adhered = **C**. No agreement = **D**, an extra comparison group.
- **Labelling steps:**
  1. Build a rule-based labeller: extract dollar figures from messages, then detect assent.
  2. Andrew hand-labels about 60–80 units, stratified by condition, using the legal standard for agreement (a "meeting of minds").
  3. Report agreement between the rule labeller and Andrew as Cohen's kappa.
  4. If kappa is below 0.6, stop and ask Andrew. Options are a free-tier LLM labeller checked against his labels, or restricting the analysis to the hand-labelled set.
- **Decision point:** if C has fewer than 20 units, report C descriptively rather than as a classification target.
- **Circularity warning:** labels use the messages, so Tier 2 features partly match the labels by construction. Tier 1 conduct features never see the messages, so they are the clean test. Use Tier 2 mainly to separate A from C.

## Plus factors to turn into metrics

Andrew checks every legal source. Write each metric's exact definition in `notes/decisions.md` *before* you look at any results.

1. **Price uniformity and parallel moves (Tier 1).** Ask dispersion within the window; correlation of a seller's ask changes with the other sellers'.
2. **Acting against unilateral self-interest (Tier 1).** How often a seller holds an ask above the best bid when a profitable undercut was available. Calibrate it against the B distribution, because some holding-out is ordinary bargaining.
3. **Abrupt change in conduct (Tier 1).** A break in ask level or dispersion compared with the seller's earlier windows.
4. **Price leadership (Tier 1).** Whether seller j's ask at hour t+1 moves toward seller i's ask at hour t. Start with simple lagged regression. Stretch goal: pairwise net influence (Bonjour et al. 2022, PMLR v180), which the sprint suggests under Q1.4.
5. **Market allocation or rotation (Tier 1).** Whether trades are spread across sellers more evenly than chance (taking turns). Compare against `hhi_volume`.
6. **Invitation and acceptance (Tier 2).** Price-specific proposals and assent in messages, following the logic of *Interstate Circuit*.
7. **Internal intent (Tier 3).** The paper's judge score, plus simple flags in private reasoning ("coordinate", "price war", "hold firm").
8. **Harm (outcome, not a plus factor).** Price-cost margin and buyer surplus. These show where harm occurs without agreement.

Legal sources for Andrew to verify:

- *Interstate Circuit v. US* (1939)
- *Theatre Enterprises v. Paramount* (1954)
- *Matsushita v. Zenith* (1986)
- *Bell Atlantic v. Twombly* (2007)
- *In re Text Messaging Antitrust Litig.* (7th Cir. 2010)
- Kovacic, Marshall, Marx & White, "Plus Factors and Agreement in Antitrust Law", 110 Mich. L. Rev. 393 (2011)
- Berger et al. (2025), ABA Antitrust magazine (linked on the sprint page)

## Analysis plan

1. **Split by run, 50/50, stratified by condition, fixed seed.** Write it to `results/split.json` and commit it *before* computing any outcome. Never change it afterwards.
2. **Train half only:**
   - Choose a threshold for each feature, and fit a small logistic regression for each combination: price-only baseline (mean ask vs $90), the existing judge alone, T1, T1+T2, T1+T3 and T1+T2+T3.
   - Save the frozen rules to `results/frozen_rules.json` and commit them.
3. **Test half, once:**
   - For A vs B: true-positive rate, and **false-positive rate on B**.
   - For A vs C: accuracy.
   - Give 95% bootstrap confidence intervals clustered by run, and break results down by condition.
4. **Robustness (stretch):** train on gpt-4.1-seller runs and test on claude and mixed-seller runs (a model-shift test).
5. **Be honest about size.** There are only 10 B runs, so just 5 in the test half. Call this a pilot and show the wide confidence intervals.

## Controls (the sprint requires a positive control)

- **Must have:** reproduce the paper's headline result as the positive control. Comms runs should show higher prices, higher judge coordination scores, and so on, compared with `base`.
- **Stretch:** a scripted market in pure Python that follows the same rules (midpoint price, cost $80, value $100, 5×5 agents, 30 hours) and costs nothing to run. Simulate:
  - independent noisy undercutters (B-like)
  - a cartel holding a shared floor (A-like)
  - a cartel with one cheater (C-like)

  The Tier 1 metrics must separate the cartel from the independents here. If they can't, the metrics are broken; fix them before trusting any results on the LLM logs.

## Schedule

- **Phase 0 (before 23 Oct, allowed):**
  - Set up a Python 3.11+ environment with pandas, numpy, scipy, scikit-learn, matplotlib and pytest.
  - Clone the data repo at the pinned commit.
  - Read the paper, `market.py`, `evaluate.py` and the judge template.
  - Andrew reads the legal sources and drafts the agreement codebook. Log the codebook in `PRIOR_WORK.md`.
  - **No analysis code.**
- **Fri 23 Oct:**
  - Write a loader that turns the logs into tidy tables: hours, asks, trades, messages, private reasoning, judge scores.
  - Add a pytest that checks 76 runs × 30 hours × 5 sellers.
  - Create and commit the split.
  - Run the positive-control reproduction.
- **Sat 24 Oct:**
  - Build the plus-factor features and the rule labeller.
  - Andrew hand-labels; compute kappa.
  - Fit on the train half; freeze the rules.
- **Sun 25 Oct:**
  - Evaluate on the test half; make figures and tables.
  - Run the stretch items only if everything above is done.
  - **Hand Andrew the final numbers and figures by Sun evening.**
- **Mon 26 Oct morning:** Andrew finishes writing. Submit by about 12:00 WAT.

## Repo layout

```
CLAUDE.md  PRIOR_WORK.md  README.md  .gitignore (data/)
data/raw/  data/processed/        # both gitignored
src/  load.py  features.py  labels.py  controls.py  evaluate.py
tests/  results/ (csv, png, split.json, frozen_rules.json)
notes/  decisions.md  dual_use.md  method_facts.md
```

## Always ask Andrew before you

- make any API call or install anything that costs money
- touch the test half before the rules are frozen (never do this), or change anything after seeing test results
- drop or merge a class
- delete files

## Submission checklist (Andrew's job; remind him)

- Report PDF on the official template.
- Abstract of 150 words or fewer.
- Authors and affiliations.
- The "Limitations and Dual-Use / Ethical Considerations" appendix.
- Tick Track 2, and Track 1 too if net influence is included.
- Optional: a public GitHub repo (code only, no raw data) and a 3–5 minute video.
- State clearly that the data is from Agrawal et al. and that the analysis is new sprint work.
