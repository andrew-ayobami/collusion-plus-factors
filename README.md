# Plus factors for AI agent collusion

Apart Research AI Collusion Research Sprint, 23–25 Oct 2026. Track 2, question 2.3.

Data: Agrawal et al. (2025), arXiv 2507.01413,
https://github.com/andyjliu/llm-agent-collusion at commit `2bb2a7e`.
The data is not included in this repo. The analysis code here is new sprint work
(see `PRIOR_WORK.md` for what existed before the sprint).

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
