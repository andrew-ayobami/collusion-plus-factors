"""Fit on the train half, freeze, then evaluate once on the test half.

  python -m src.evaluate dryrun   train half only: 5-fold CV by run (nothing written to results/)
  python -m src.evaluate freeze   fit on train half -> results/frozen_rules.json (commit it)
  python -m src.evaluate test     apply frozen rules to the test half ONCE -> results/test_*.csv

Plan pre-registered in notes/decisions.md (2026-10-24 03:19).
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.features import PRICE_ONLY, T1, T2, T3
from src.load import PROCESSED, ROOT

RESULTS = ROOT / "results"
FROZEN = RESULTS / "frozen_rules.json"
MIN_C_TRAIN = 20
N_BOOT = 2000

FEATURE_SETS = {
    "price_only": PRICE_ONLY,
    "judge_only": ["t3_judge_mean"],
    "T1": T1,
    "price+T1": PRICE_ONLY + T1,
    "T1+T2": T1 + T2,
    "T1+T3": T1 + T3,
    "T1+T2+T3": T1 + T2 + T3,
}
ALL_FEATURES = PRICE_ONLY + T1 + T2 + T3


def units(labels_override: pd.DataFrame | None = None) -> pd.DataFrame:
    u = pd.read_pickle(PROCESSED / "units.pkl")
    if labels_override is not None:
        u = u.drop(columns="label").merge(labels_override, on=["run_id", "seller", "window"])
    return u


def fit_lr(X: np.ndarray, y: np.ndarray) -> dict:
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(sc.transform(X), y)
    return dict(mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_[0].tolist(),
                intercept=float(lr.intercept_[0]), threshold=0.5)


def predict_lr(m: dict, X: np.ndarray) -> np.ndarray:
    z = (X - np.array(m["mean"])) / np.array(m["scale"])
    return 1 / (1 + np.exp(-(z @ np.array(m["coef"]) + m["intercept"])))


def youden_rule(x: np.ndarray, y: np.ndarray) -> dict:
    """Direction and cut maximising TPR - FPR. Rule flags x >= cut (direction +1) or x <= cut (-1)."""
    best = dict(j=-1.0)
    for direction in (1, -1):
        xs = direction * x
        for cut in np.unique(xs):
            flag = xs >= cut
            j = flag[y == 1].mean() - flag[y == 0].mean()
            if j > best["j"]:
                best = dict(j=float(j), direction=direction, cut=float(direction * cut))
    return best


def apply_rule(r: dict, x: np.ndarray) -> np.ndarray:
    return (x >= r["cut"]) if r["direction"] == 1 else (x <= r["cut"])


def task_data(u: pd.DataFrame, pos: str, neg: str):
    d = u[u.label.isin([pos, neg])].copy()
    return d, (d.label == pos).astype(int).to_numpy()


def dryrun(u=None):
    u = units() if u is None else u
    tr = u[u.half == "train"].copy()
    med = tr[ALL_FEATURES].median()
    tr[ALL_FEATURES] = tr[ALL_FEATURES].fillna(med)
    d, y = task_data(tr, "A", "B")
    rows = []
    for name, cols in FEATURE_SETS.items():
        p = np.zeros(len(d))
        for fit_ix, val_ix in GroupKFold(5).split(d, y, groups=d.run_id):
            m = fit_lr(d[cols].to_numpy()[fit_ix], y[fit_ix])
            p[val_ix] = predict_lr(m, d[cols].to_numpy()[val_ix])
        flag = p >= 0.5
        rows.append(dict(model=name, cv_auc=roc_auc_score(y, p), cv_tpr=flag[y == 1].mean(), cv_fpr_B=flag[y == 0].mean()))
    print("TRAIN ONLY, 5-fold CV by run, A vs B:")
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def freeze(u=None, note: str = ""):
    if FROZEN.exists():
        raise SystemExit(f"{FROZEN} exists. Rules are frozen; never refit after seeing test results.")
    u = units() if u is None else u
    tr = u[u.half == "train"].copy()
    med = tr[ALL_FEATURES].median()
    tr[ALL_FEATURES] = tr[ALL_FEATURES].fillna(med)
    counts = tr.label.value_counts().to_dict()
    tasks = {"A_vs_B": ("A", "B")}
    if counts.get("C", 0) >= MIN_C_TRAIN:
        tasks["A_vs_C"] = ("A", "C")
    frozen = dict(
        frozen_at=datetime.now().isoformat(timespec="seconds"),
        git_commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                  cwd=ROOT).stdout.strip(),
        note=note, train_label_counts=counts, medians=med.to_dict(), tasks={})
    for task, (pos, neg) in tasks.items():
        d, y = task_data(tr, pos, neg)
        models = {name: dict(features=cols, **fit_lr(d[cols].to_numpy(), y)) for name, cols in FEATURE_SETS.items()}
        rules = {f: youden_rule(d[f].to_numpy(), y) for f in ALL_FEATURES}
        frozen["tasks"][task] = dict(positive=pos, negative=neg, n_train=dict(pos=int(y.sum()), neg=int((1 - y).sum())),
                                     models=models, single_feature_rules=rules)
    FROZEN.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"Frozen -> {FROZEN}: tasks {list(tasks)}; train counts {counts}")


# ------------------------------------------------------------------ test (run once, after freezing)

def _scores(frozen, task, d):
    t = frozen["tasks"][task]
    out = {f"model:{n}": predict_lr(m, d[m["features"]].to_numpy()) for n, m in t["models"].items()}
    out.update({f"rule:{f}": apply_rule(r, d[f].to_numpy()).astype(float) for f, r in t["single_feature_rules"].items()})
    return out


def _metrics(y, s, is_model: bool):
    flag = s >= 0.5
    m = dict(tpr=flag[y == 1].mean() if (y == 1).any() else np.nan,
             fpr=flag[y == 0].mean() if (y == 0).any() else np.nan)
    m["auc"] = roc_auc_score(y, s) if is_model and len(np.unique(y)) == 2 else np.nan
    m["accuracy"] = (flag == y).mean()
    m["balanced_accuracy"] = np.nanmean([m["tpr"], 1 - m["fpr"]])
    return m


def _boot(d, y, s, is_model, rng):
    """Resample runs with replacement within each condition."""
    groups = {c: g.run_id.unique() for c, g in d.groupby("condition")}
    idx_by_run = {r: np.flatnonzero(d.run_id.to_numpy() == r) for r in d.run_id.unique()}
    reps = []
    for _ in range(N_BOOT):
        ix = np.concatenate([idx_by_run[r] for runs in groups.values() for r in rng.choice(runs, len(runs))])
        reps.append(_metrics(y[ix], s[ix], is_model))
    return pd.DataFrame(reps)


def test(u=None):
    frozen = json.loads(FROZEN.read_text())
    u = units() if u is None else u
    te = u[u.half == "test"].copy()
    te[ALL_FEATURES] = te[ALL_FEATURES].fillna(pd.Series(frozen["medians"]))
    rng = np.random.default_rng(20261025)
    rows, cond_rows, d_rows = [], [], []
    for task, t in frozen["tasks"].items():
        d, y = task_data(te, t["positive"], t["negative"])
        sc = _scores(frozen, task, d)
        dd = te[te.label == "D"]
        sd = _scores(frozen, task, dd) if task == "A_vs_B" else {}
        for name, s in sc.items():
            is_model = name.startswith("model:")
            m = _metrics(y, s, is_model)
            b = _boot(d, y, s, is_model, rng)
            row = dict(task=task, scorer=name, n_pos=int(y.sum()), n_neg=int((1 - y).sum()), **m)
            for k in m:
                row[f"{k}_lo"], row[f"{k}_hi"] = (np.nanquantile(b[k], [0.025, 0.975]) if b[k].notna().any()
                                                  else (np.nan, np.nan))
            rows.append(row)
            for c, g in d.groupby("condition"):
                gy, gs = (g.label == t["positive"]).astype(int).to_numpy(), s[d.index.get_indexer(g.index)]
                flag = gs >= 0.5
                cond_rows.append(dict(task=task, scorer=name, condition=c, n=len(g),
                                      flag_rate=flag.mean(), label=t["positive"] if gy.all() else
                                      (t["negative"] if not gy.any() else "mixed")))
            if name in sd:
                flag = sd[name] >= 0.5
                d_rows.append(dict(task=task, scorer=name, n_D=len(dd), fpr_D=flag.mean()))
    pd.DataFrame(rows).round(4).to_csv(RESULTS / "test_metrics.csv", index=False)
    pd.DataFrame(cond_rows).round(4).to_csv(RESULTS / "test_by_condition.csv", index=False)
    pd.DataFrame(d_rows).round(4).to_csv(RESULTS / "test_fpr_on_D.csv", index=False)
    print(pd.DataFrame(rows)[["task", "scorer", "n_pos", "n_neg", "tpr", "fpr", "auc"]].round(3).to_string(index=False))


if __name__ == "__main__":
    {"dryrun": dryrun, "freeze": freeze, "test": test}[sys.argv[1]]()
