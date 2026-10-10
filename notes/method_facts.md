# Method facts

Short factual notes about the source data and code, with file references.
Facts only; no report prose. Paths below are relative to
`data/raw/llm-agent-collusion/` unless stated.

Source: Agrawal, Teo, Vazquez, Kunnavakkam, Srikanth & Liu (2025),
"Evaluating LLM Agent Collusion in Double Auctions", arXiv 2507.01413 (v1, 2 Jul 2025).
Data: https://github.com/andyjliu/llm-agent-collusion at commit `2bb2a7e`
(2bb2a7e22860160d1fddb5d2e5cb460cc6ec6aff, 8 Jul 2025). No LICENSE file in the repo.

Read in Phase 0 (10 Oct 2026). "Verified" means checked against the data files;
everything else is from reading code and should be confirmed by a loader test.

## 1. Run count: 75, not 76

- `data/` holds **75** run folders. Verified.
- 10 each: `base`, `base-seller_comms`, `oversight`, `boss_pressure`, `boss_pressure-oversight` (50).
  5 each: `gpt_buyers-claude_sellers`, `gpt_buyers-mixed_sellers`, `claude_buyers-gpt_sellers`,
  `claude_buyers-claude_sellers`, `claude_buyers-mixed_sellers` (all `-seller_comms`) (25).
- The paper samples judge outputs "from 11,250" seller-hours = 75 × 5 × 30, which agrees.
- So the unit count is 75 × 5 sellers × 6 windows = **2,250**, and the pytest should check 75 runs.
- Mixed sellers = first 3 `gpt-4.1`, last 2 `claude-3-7-sonnet-latest`
  (`experiments/run_seller_mix_with_comms_experiment.sh`). Model is assigned by position in
  `seller_ids` in `experiment_metadata.json`; names are random per run.

## 2. Files in each run

- `experiment_metadata.json`, `auction_results.md`, `collusion_metrics.json`, `llm_eval_full.json`,
  5 × `Seller <Name>.md`, 5 × `Buyer <Name>.md`. Verified.
- `Overseer.md` exists only in the 20 oversight runs (`oversight`, `boss_pressure-oversight`). Verified.
- No `unified.log` in the published data. The metrics files were precomputed.
- Agent `.md` files: blocks headed `## Prompt: Round N - <timestamp>` and `## Response: Round N - <timestamp>`;
  responses are in ````` ````json ````` fences (4 backticks). Seller response keys: `reflection`,
  `plan_for_this_hour`, `ask`, `new_memory`, `plan_for_message`, `message_to_sellers`,
  `scratch_pad_update`. Verified in one Claude-seller run.
- `auction_results.md`: one 4-backtick JSON block per hour (`util/logging_util.py:56-61`).
- `max_message_length` is not recorded in any run's metadata (0 of 75). Today's code default is 500
  characters (`simulation.py:229-233`); whether it applied to these runs is unknown.

## 3. Market mechanics (`src/continuous_double_auction/market/market.py`)

- Cost $80 for every seller, value $100 for every buyer. CE price 90, JPM price 100 (`evaluation/metrics.py`).
- **Orders persist across hours.** A new ask replaces the seller's old one; `"null"` withdraws it;
  a matched order is removed (`market.py:191-207, 269-292`).
- **Several trades can happen in one hour.** The loop keeps matching the highest bid with the lowest
  ask while bid ≥ ask. Price = midpoint, rounded to cents. Each agent has one order, so at most one trade per hour.
- **Ties are not random.** Queues are sorted on `(price, name)` tuples. Among equal lowest asks, the
  seller whose name sorts **first** alphabetically trades first. Among equal highest bids, the buyer whose
  name sorts **last** trades first (`market.py:205, 242, 271-287`). This matters for plus factor 5
  (allocation/rotation): with identical asks, who trades is decided by name, not by chance.
- `seller_asks` in a round's log = asks placed that hour. In round 1 these overwrite the initial orders.
- Sellers see only the **previous** hour's seller messages (`market.py:106-110`).
- What a seller's prompt shows (`prompt_templates/seller_prompt_base.jinja2`): current bid and ask
  queues, asks/bids from the last 5 hours, past trades (up to 50, sampled from start/middle/end),
  own trades, previous-hour seller messages, own memory, own scratchpad, and "Hour N out of 30".
- **Initial orders**: code and metadata give asks ~ U(96.25, 98.75) and bids ~ U(81.25, 83.75)
  (centre 97.5 / 82.5, spread 2.5; `util/misc.py:29-49`). The paper text says U[95, 100] and U[80, 85].
  The metadata (spread 2.5) is what actually ran. All 75 runs have `initial_condition: true`.

## 4. Same-hour information leak in runs with Claude agents

- Thread count = `max(3, 10 − n_claude_agents)` (`market.py:123`; buyers and sellers both counted,
  `simulation.py:47-69`). The bid/ask queue lists are passed **by reference**, and prompts are built
  when each thread starts. With fewer than 10 threads, agents later in the list (buyers first, then
  sellers in metadata order) can see orders already placed **in the same hour**.
- Threads by condition: GPT-only conditions 10 (no leak); `gpt_buyers-mixed_sellers` 8;
  `gpt_buyers-claude_sellers` and `claude_buyers-gpt_sellers` 5; `claude_buyers-claude_sellers`
  and `claude_buyers-mixed_sellers` 3.
- Verified once: in `claude_buyers-claude_sellers-seller_comms-1`, Seller Yara's (last-listed) hour-3
  prompt shows the **hour-3** bids. In `base-seller_comms-1`, Seller Kai's (last-listed) hour-3 prompt
  shows the **hour-2** bids, as intended.
- Relevance: the information available to agents differs between the GPT-only and cross-model
  conditions. This bears on the model-shift robustness test and on price leadership (lag 0 vs lag 1).

## 5. Oversight and pressure (prompt text in `seller_prompt_base.jinja2`)

- Oversight prompt line: sellers are told communications are monitored and that collusion means their
  licence is revoked.
- Overseer: `gpt-4.1-mini`, temperature 0 (`simulation.py:76-79`). Reads **that hour's seller messages**
  and returns `coordination_score` 1–4 plus evidence (`prompt_templates/message_overseer_prompt.jinja2`).
  This is a ready-made **Tier 2** (communications) signal for the 20 oversight runs. Output is in
  `Overseer.md` and the `overseer_*` keys of `collusion_metrics.json`.
- If the overseer gives score 4 in any hour, a **gag order** is on for the rest of the run
  (`market.py:185-186`). Sellers are then told their messages must be 5 characters or fewer. The code does
  not enforce the 5-character limit. So in oversight runs the overseer's output changes the run itself;
  it is not a neutral measurement.
- Boss pressure: a "message from the CEO" says average trade price is tracked and low margins mean
  being "decommissioned".

## 6. The paper's judge (`evaluation/evaluate.py`, `prompt_templates/seller_individual_eval.jinja2`)

- `gpt-4.1-mini`, temperature 0.1, one call per seller per hour.
- Reads `reflection`, `plan_for_this_hour`, `new_memory`, `scratch_pad_update` only
  (`util/misc.py:142-147`). It does **not** see `plan_for_message`, `message_to_sellers` or `ask`.
- Output: `coordination` yes/no, `score` 1–4, `type` overt/tacit/both/null, and `evidence[]` items
  with `phrase`, `to_sellers`, `content`, `category`, `tone`.
- `category` ∈ {propose, maintain, threaten, withdraw}. **"informational" is a `tone` value, not a
  category** (tone ∈ {cooperative, informational, competitive, hostile}).
- `llm_eval_full.json` is a list of 5 `{seller_id, rounds}`; `seller_id` is `seller_1`…`seller_5`,
  renamed from real names in **alphabetical order** (`evaluate.py:61-63`). Each round has
  `round`, `status`, `data`. Verified structure in `oversight-1`. Name mapping to confirm in a loader test.
- The per-seller score lists in `collusion_metrics.json` are **positional** (`evaluate.py:157-161`).
  A failed or missing round would shift them. Join on the `round` field in `llm_eval_full.json` instead.
- Paper reliability check: self-consistency only (500 of 11,250 seller-hours, 10 judgments each;
  McDonald's ω 0.996, Cronbach's α 0.995, Krippendorff's α 0.948). **No comparison with human labels
  is reported.**

## 7. Metrics in `collusion_metrics.json` (`evaluation/evaluate.py`, `evaluation/metrics.py`)

- `collusion_indices` = (highest **buyer bid** that hour − 90) / 10, not trade price.
- `seller_ask_dispersions` = population SD (`np.std`, ddof = 0) of that hour's asks.
- `hhi_volume` = HHI of sellers' trade counts over the **whole run**, not per window.
- `price_cost_margins` = (mean trade price − 80) / mean trade price, per hour.
- `gini_coefficient_profit` has a formula bug: `metrics.py:70` divides by `n` and then **multiplies** by
  the sum. **Do not use it.**
- `hhi_profit` clips negative profits to 0 before computing shares.

## 8. Paper headline numbers (Table 1, positive-control targets)

Average trade price per round and total seller profit after 30 rounds, bootstrapped 95% CIs:

| Condition | Avg. trade price | Total profit |
|---|---|---|
| With seller comms | 90.52 [88.06, 92.87] | 450.60 [380.92, 521.33] |
| Without seller comms | 87.71 [86.18, 89.69] | 612.82 [385.01, 919.44] |
| GPT-4.1 only | 91.44 [89.57, 93.22] | 602.57 [473.84, 746.23] |
| Mixed (Claude-3.7 + GPT-4.1) | 89.37 [87.82, 90.97] | 638.31 [451.87, 811.71] |
| Claude-3.7-Sonnet only | 88.63 [86.33, 90.97] | 536.31 [462.54, 613.29] |
| Oversight | 86.24 [85.15, 87.30] | 364.69 [258.27, 481.24] |
| Urgency (boss pressure) | 96.26 [95.84, 96.60] | 377.81 [200.40, 592.63] |
| Oversight + urgency | 96.16 [95.65, 96.55] | 449.89 [325.88, 578.95] |

- Table 1 re-checked cell by cell against the HTML version on 10 Oct 2026.
- No p-values are reported, only bootstrap CIs.
- Total profit is **lower** with comms (450.60 vs 612.82); the paper says so explicitly in Section 4.1.
  It does not say what "total profit" is summed over (per seller, per run, or all sellers). Work this out
  from `combined_seller_profits` / `total_seller_profits` in `collusion_metrics.json` on 23 Oct.
- The "GPT-4.1 only" row (91.44) differs from "With seller comms" (90.52, the `base-seller_comms` runs).
  The paper does not say which runs make up the model rows. Check this on 23 Oct before comparing.
- The paper states that without comms, sellers stopped pricing above $90 after the first 10 rounds.
