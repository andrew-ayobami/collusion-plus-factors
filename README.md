# Plus factors for AI agent collusion

Apart Research AI Collusion Research Sprint, 23–25 Oct 2026. Track 2, question 2.3.

Data: Agrawal et al. (2025), arXiv 2507.01413,
https://github.com/andyjliu/llm-agent-collusion at commit `2bb2a7e`.
The data is not included in this repo.

**What was done before the sprint, and what during it.** Before the sprint (10 Oct 2026) only minimal setup was
done: the project brief, the Python environment, a clone of the source data, reading the paper and its code (short
factual notes in `notes/method_facts.md`), and one quick sanity check of mean trade prices. No analysis code was
written or kept. All the research work was done during the sprint (24–25 Oct): the loader and tests, the train/test
split, the controls, the plus-factor metrics, the agreement codebook, the rule labeller, the hand labels, the models
and every result. The git history shows this timeline; `PRIOR_WORK.md` lists the pre-sprint items.

## Setup (Windows, Git Bash)

The virtual environment lives outside OneDrive so OneDrive does not try to sync it.

```bash
python -m venv ~/.venvs/collusion-plus-factors
~/.venvs/collusion-plus-factors/Scripts/python -m pip install -r requirements.txt
```

Get the data at the pinned commit:

```bash
git clone https://github.com/andyjliu/llm-agent-collusion.git data/raw/llm-agent-collusion
git -C data/raw/llm-agent-collusion checkout 2bb2a7e
```

## Layout

```
src/       load.py  features.py  labels.py  controls.py  evaluate.py
tests/     pytest checks
results/   csv, png, split.json, frozen_rules.json
notes/     decisions.md  dual_use.md  method_facts.md
data/      raw/ and processed/ (gitignored)
```

## Running the pipeline

Use the venv's Python (`~/.venvs/collusion-plus-factors/Scripts/python`) from the repo root.

```bash
python -m src.load                 # parse 75 runs -> data/processed/tables.pkl
python -m pytest -q tests          # 75 runs x 30 hours x 5 sellers, replay check, features
python -m src.controls positive    # paper Table 1 reproduction (positive control)
python -m src.controls scripted    # scripted-market control (no LLM)
python -m src.labels               # rule labeller v2 (codebook), both readings
python -m src.labels sensitivity   # train-half label sensitivity grid
python -m src.features             # plus-factor features -> data/processed/units.pkl
python -m src.handlabel sheet      # hand-label sheet (local HTML, gitignored)
python -m src.handlabel kappa      # kappa vs rule labels, after hand labelling
python -m src.evaluate dryrun [interstate]   # train-half CV only
python -m src.evaluate freeze [interstate]   # fit on train -> results/frozen_rules[_interstate].json (once)
python -m src.evaluate test [interstate]     # test half, once
python -m src.evaluate shift       # model-shift stretch test
python -m src.figures              # figures and descriptive tables
python -m src.influence check|train|test     # net influence stretch (Bonjour et al. 2022)
python -m src.ladder_explore       # EXPLORATORY ladder analysis (post-hoc)
```

`src/split.py` made `results/split.json` once; it refuses to overwrite it.
