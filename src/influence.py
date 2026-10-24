"""Pairwise net influence (Bonjour, Aggarwal & Bhargava 2022, UAI; PMLR 180:223-232), stretch analysis.

Pre-registered in notes/decisions.md (2026-10-24, "Net influence (stretch)"). Run level, not seller x window:
there are only 29 ask moves per seller per run.

  python -m src.influence check   scripted colluding-pair markets: does the true pair score highest?
  python -m src.influence train   run scores on all runs; alpha chosen on train; train AUC
  python -m src.influence test    test-half run-level TPR/FPR (only after the main test)
"""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.load import N_HOURS, ROOT, load

RESULTS = ROOT / "results"
HOLD_BAND = 0.005
N_CHECK_MARKETS = 30


def actions(asks: np.ndarray) -> np.ndarray:
    """asks: hours x sellers (in force). Returns (hours-1) x sellers in {0 down, 1 hold, 2 up}."""
    d = np.diff(asks, axis=0)
    d = np.nan_to_num(d, nan=0.0)
    return np.where(d < -HOLD_BAND, 0, np.where(d > HOLD_BAND, 2, 1))


def gamma(ai: np.ndarray, aj: np.ndarray, s: np.ndarray) -> float:
    """Plug-in conditional mutual information I(a_i; a_j | s) in bits (Bonjour Eq. 6)."""
    total = 0.0
    n = len(s)
    for sv in np.unique(s):
        m = s == sv
        x, y = ai[m], aj[m]
        joint = np.zeros((3, 3))
        np.add.at(joint, (x, y), 1)
        joint /= m.sum()
        px, py = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
        nz = joint > 0
        total += (m.sum() / n) * float((joint[nz] * np.log2(joint[nz] / (px @ py)[nz])).sum())
    return total


def net_influence(A: np.ndarray, s: np.ndarray):
    """A: actions (T x n). Returns gamma matrix, net influence matrix Gamma[i, j] = Γ(i; j), pair scores."""
    n = A.shape[1]
    g = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        g[i, j] = g[j, i] = gamma(A[:, i], A[:, j], s)
    G = np.full((n, n), np.nan)
    for i, j in itertools.permutations(range(n), 2):
        G[i, j] = g[i, j] - max(g[k, j] for k in range(n) if k not in (i, j))
    pairs = {(i, j): min(G[i, j], G[j, i]) for i, j in itertools.combinations(range(n), 2)}
    return g, G, pairs


def run_scores(orders: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    traded = set(map(tuple, trades[["run_id", "hour"]].drop_duplicates().to_numpy()))
    for run_id, o in orders.groupby("run_id"):
        P = o.pivot(index="hour", columns="seller", values="ask").sort_index()
        sellers = list(P.columns)
        A = actions(P.to_numpy(dtype=float))
        s = np.array([int((run_id, h - 1) in traded) for h in range(2, N_HOURS + 1)])
        g, G, pairs = net_influence(A, s)
        (bi, bj), best = max(pairs.items(), key=lambda kv: kv[1])
        rows.append(dict(run_id=run_id, run_score=best, best_pair=f"{sellers[bi]} & {sellers[bj]}",
                         mean_gamma=g[np.triu_indices(len(sellers), 1)].mean(),
                         share_hold=(A == 1).mean()))
    return pd.DataFrame(rows)


def check():
    """Implementation check on scripted markets with one colluding pair."""
    from src.controls import SIM_SELLERS, simulate_market
    rows = []
    for k in range(N_CHECK_MARKETS):
        a, b, tr, pair, _ = simulate_market("pair", 10_000 + k)
        P = a.pivot(index="hour", columns="seller", values="ask_logged").reindex(columns=SIM_SELLERS).ffill()
        A = actions(P.to_numpy(dtype=float))
        hours_traded = set(tr.hour) if len(tr) else set()
        s = np.array([int(h - 1 in hours_traded) for h in range(2, N_HOURS + 1)])
        g, G, pairs = net_influence(A, s)
        (bi, bj), best = max(pairs.items(), key=lambda kv: kv[1])
        true = (SIM_SELLERS.index(pair[0]), SIM_SELLERS.index(pair[1]))
        rows.append(dict(market=k, true_pair=" & ".join(pair), top_pair_correct=(bi, bj) == true,
                         true_pair_score=pairs[true], best_other_score=max(v for p, v in pairs.items() if p != true)))
    res = pd.DataFrame(rows)
    res.round(4).to_csv(RESULTS / "net_influence_check.csv", index=False)
    print(res.round(3).to_string(index=False))
    print("top pair correct in", res.top_pair_correct.sum(), "of", len(res), "markets")


def _labelled_runs():
    t = load()
    sc = run_scores(t["orders"], t["trades"])
    u = pd.read_pickle(ROOT / "data" / "processed" / "units.pkl")
    has_a = u.groupby("run_id").label.apply(lambda s: (s == "A").any())
    runs = t["runs"][["run_id", "condition", "seller_comms"]].merge(sc, on="run_id")
    runs["has_A"] = runs.run_id.map(has_a)
    split = json.loads((RESULTS / "split.json").read_text())
    runs["half"] = np.where(runs.run_id.isin(split["train"]), "train", "test")
    runs["group"] = np.where(~runs.seller_comms, "B (base)", np.where(runs.has_A, "comms, has A", "comms, no A"))
    return runs


def train():
    from src.evaluate import youden_rule
    runs = _labelled_runs()
    tr = runs[(runs.half == "train") & runs.group.isin(["B (base)", "comms, has A"])]
    y = (tr.group == "comms, has A").astype(int).to_numpy()
    rule = youden_rule(tr.run_score.to_numpy(), y)
    out = dict(alpha_rule=rule, train_auc=roc_auc_score(y, tr.run_score), n_train_pos=int(y.sum()),
               n_train_neg=int((1 - y).sum()))
    (RESULTS / "net_influence_frozen.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    trd = runs[runs.half == "train"]
    trd.drop(columns="half").round(4).to_csv(RESULTS / "net_influence_train_runs.csv", index=False)
    print(json.dumps(out, indent=2))
    print(trd.groupby("group")[["run_score", "mean_gamma", "share_hold"]].agg(["mean", "count"]).round(3))


def test():
    from src.evaluate import apply_rule
    if not (RESULTS / "test_metrics.csv").exists():
        raise SystemExit("Run the main test first.")
    frozen = json.loads((RESULTS / "net_influence_frozen.json").read_text())
    runs = _labelled_runs()
    te = runs[(runs.half == "test") & runs.group.isin(["B (base)", "comms, has A"])]
    y = (te.group == "comms, has A").astype(int).to_numpy()
    flag = apply_rule(frozen["alpha_rule"], te.run_score.to_numpy())
    out = dict(test_auc=roc_auc_score(y, te.run_score), tpr=flag[y == 1].mean(), fpr=flag[y == 0].mean(),
               n_pos=int(y.sum()), n_neg=int((1 - y).sum()))
    (RESULTS / "net_influence_test.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    runs[runs.half == "test"].drop(columns="half").round(4).to_csv(RESULTS / "net_influence_test_runs.csv", index=False)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    {"check": check, "train": train, "test": test}[sys.argv[1]]()
