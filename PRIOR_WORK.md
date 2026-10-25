# Prior work (before the sprint)

The AI Collusion Research Sprint runs Fri 23 – Sun 25 Oct 2026.
Everything listed here was done **before** 23 Oct 2026 and is disclosed so the
judges can see what is new sprint work.

Rule for Phase 0: setup, reading and planning only. No analysis code.

## Log

| Date (WAT) | What | Where |
|---|---|---|
| 2026-10-10 (before this log) | Project brief written (`CLAUDE.md`): research question, plus-factor list, labelling plan, analysis plan. | `CLAUDE.md` |
| 2026-10-10 (before this log) | Data repo inspected. A quick parser was run on all runs (75; the brief then said 76) as a sanity check: mean trade price $87.71 (no-comms `base`) vs $90.52 (matching comms runs). That parser is **not** in this repo. See note below. | `CLAUDE.md`, "Data" section |
| 2026-10-10 | Git repo created; folders and notes files set up. | repo root |
| 2026-10-10 | Python environment set up (versions pinned). | `requirements.txt` |
| 2026-10-10 | Data repo cloned at commit `2bb2a7e` into `data/raw/` (not committed). | `data/raw/` |
| 2026-10-10 | Read the paper, `market.py`, `evaluate.py`, `metrics.py`, the judge, overseer and seller prompt templates. Short factual notes written. Light inspection of file formats and one prompt-vs-log check; no analysis code written or kept. | `notes/method_facts.md` |
| 2026-10-10 | Phase-0 decisions logged (run count 75, judge join key, unused metric). | `notes/decisions.md` |

## Note on the pre-sprint sanity check

`CLAUDE.md` records that the mean trade price was already computed before the
sprint. This is a small piece of pre-sprint analysis and must be disclosed.
The sprint loader in `src/load.py` was written from scratch during the sprint (24 Oct, commit `2452c1d`).

## Agreement codebook (Andrew)

Not prior work. The codebook was planned for Phase 0 but was written during the sprint, on 24 Oct 2026:
`notes/codebook.md`, v1 draft commit `95b9a8f`, v2 commit `bc278ee`.

## Summary

Pre-sprint work was limited to setup and reading (four commits on 10 Oct 2026, ending `42f201f`). Every analysis
step, from the loader (`2452c1d`, 24 Oct) onward, was done during the sprint.
