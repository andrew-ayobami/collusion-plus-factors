# Dual-use notes

Running log of dual-use and ethical thoughts, for the required report appendix
"Limitations and Dual-Use / Ethical Considerations".

| Date | Note |
|---|---|
| 2026-10-10 | Detection thresholds are evasion targets. Publishing frozen Tier 1 thresholds tells an agent designer which conduct patterns to avoid. In the source data, agents told they were monitored already changed behaviour (paper: lower prices under oversight), and the overseer's score-4 trigger fed back into the run via the gag order. Consider how much threshold detail to publish. |
| 2026-10-10 | Data handling: agent names are synthetic, so there is no personal data. The source repo has no licence, so raw logs and per-message text are not redistributed; only code and aggregate results go in this repo. |
| 2026-10-24 | Train-half finding (pre-test): conduct-only plus factors separate LLM cartels from independent sellers only weakly, while private-reasoning evidence (the paper's judge) separates them well. If this holds on test, a regulator watching only public market data would miss most agent collusion, which argues for logging agent reasoning (audit trails). The flip side: agents optimised against a reasoning monitor may learn to hide intent in their reasoning, and the oversight runs show that a known monitor changes behaviour. |
| 2026-10-24 | The paper's judge says "coordination: yes" in 69% of seller-hours in `base` runs, where sellers could not talk. Used as a standalone enforcement trigger, a judge like this would accuse innocent parallel conduct often unless it is calibrated against a no-communication baseline. |
| 2026-10-24 | If the LLM fallback labeller is used: it sends raw transcripts from an unlicensed dataset to a third-party API. Free tiers may allow the provider to use inputs to improve its products. Check the provider's data-use terms before any call and mention it in the limitations. |
