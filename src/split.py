"""Make the run-level 50/50 train/test split, stratified by condition.

Run once: python -m src.split   -> results/split.json (commit it; never change it).
"""
from __future__ import annotations

import json

import numpy as np

from src.load import ROOT, load

SEED = 20261024
OUT = ROOT / "results" / "split.json"


def make_split(runs) -> dict:
    rng = np.random.default_rng(SEED)
    train, test = [], []
    odd_turn = 0
    for cond in sorted(runs.condition.unique()):
        ids = sorted(runs.run_id[runs.condition == cond])
        ids = list(rng.permutation(ids))
        n = len(ids)
        if n % 2:  # 5-run conditions alternate 3/2 and 2/3 so the halves stay balanced
            n_train = n // 2 + (1 if odd_turn % 2 == 0 else 0)
            odd_turn += 1
        else:
            n_train = n // 2
        train += ids[:n_train]
        test += ids[n_train:]
    return {"seed": SEED, "method": "per condition: sort run_ids, numpy default_rng(seed).permutation, "
                                    "first half -> train; 5-run conditions alternate 3/2 train in sorted condition order",
            "train": sorted(train), "test": sorted(test)}


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists; the split must never change.")
    runs = load()["runs"]
    split = make_split(runs)
    OUT.write_text(json.dumps(split, indent=2), encoding="utf-8")
    cond = runs.set_index("run_id").condition
    for half in ["train", "test"]:
        print(half, len(split[half]), cond[split[half]].value_counts().sort_index().to_dict())
