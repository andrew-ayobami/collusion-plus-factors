"""Controls.

Positive control: reproduce the paper's headline result (Agrawal et al. 2025, Table 1) and show that
comms runs differ from `base` (no comms) in prices and judge coordination scores.

Usage:  python -m src.controls positive
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.load import CE_PRICE, N_HOURS, ROOT, load
from src.stats import boot_diff_ci, boot_mean_ci

RESULTS = ROOT / "results"

# Reference palette (dataviz skill, light mode)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

# Paper Table 1 rows -> the run conditions that reproduce them (found 24 Oct; see method_facts.md §10)
TABLE1_GROUPS = {
    "With seller comms": ["base-seller_comms"],
    "Without seller comms": ["base"],
    "GPT-4.1 sellers": ["base-seller_comms", "claude_buyers-gpt_sellers-seller_comms"],
    "Mixed sellers": ["gpt_buyers-mixed_sellers-seller_comms", "claude_buyers-mixed_sellers-seller_comms"],
    "Claude-3.7 sellers": ["gpt_buyers-claude_sellers-seller_comms", "claude_buyers-claude_sellers-seller_comms"],
    "Oversight": ["oversight"],
    "Urgency (boss pressure)": ["boss_pressure"],
    "Oversight + urgency": ["boss_pressure-oversight"],
}
PAPER_TABLE1 = {  # avg trade price, total profit (point estimates as printed)
    "With seller comms": (90.52, 450.60), "Without seller comms": (87.71, 612.82),
    "GPT-4.1 sellers": (91.44, 602.57), "Mixed sellers": (89.37, 638.31), "Claude-3.7 sellers": (88.63, 536.31),
    "Oversight": (86.24, 364.69), "Urgency (boss pressure)": (96.26, 377.81), "Oversight + urgency": (96.16, 449.89),
}


def run_level(t) -> pd.DataFrame:
    """One row per run with outcome measures used by the positive control."""
    runs = t["runs"].merge(t["paper"], on="run_id")
    o = t["orders"]
    hourly = o.groupby(["run_id", "hour"]).ask.agg(["mean", lambda s: s.std(ddof=0)])
    hourly.columns = ["mean_ask", "ask_sd"]
    j = t["judge"]
    rl = pd.DataFrame({
        "mean_ask": hourly.mean_ask.groupby("run_id").mean(),
        "ask_dispersion": hourly.ask_sd.groupby("run_id").mean(),
        "judge_score": j.groupby("run_id").score.mean(),
        "judge_yes_share": j.groupby("run_id").coordination.mean(),
    })
    runs = runs.join(rl, on="run_id")
    return runs.rename(columns={"avg_trade_price_overall": "trade_price", "combined_seller_profits": "seller_profit",
                                "combined_buyer_surplus": "buyer_surplus"})


def positive_control():
    t = load()
    rl = run_level(t)

    rows = []
    for name, conds in TABLE1_GROUPS.items():
        g = rl[rl.condition.isin(conds)]
        p, plo, phi = boot_mean_ci(g.trade_price)
        q, qlo, qhi = boot_mean_ci(g.seller_profit)
        rows.append(dict(row=name, n_runs=len(g), trade_price=p, trade_price_lo=plo, trade_price_hi=phi,
                         paper_trade_price=PAPER_TABLE1[name][0], total_profit=q, total_profit_lo=qlo,
                         total_profit_hi=qhi, paper_total_profit=PAPER_TABLE1[name][1]))
    table1 = pd.DataFrame(rows).round(2)
    table1.to_csv(RESULTS / "positive_control_table1.csv", index=False)

    comms, base = rl[rl.condition == "base-seller_comms"], rl[rl.condition == "base"]
    allcomms = rl[rl.seller_comms]
    rows = []
    for m in ["trade_price", "mean_ask", "ask_dispersion", "judge_score", "judge_yes_share", "seller_profit",
              "buyer_surplus", "num_trades"]:
        for label, g in [("base-seller_comms", comms), ("all comms runs", allcomms)]:
            d, lo, hi = boot_diff_ci(g[m], base[m])
            rows.append(dict(measure=m, comparison=f"{label} minus base", group_mean=g[m].mean(),
                             base_mean=base[m].mean(), diff=d, diff_lo=lo, diff_hi=hi, n_group=len(g),
                             n_base=len(base), ci_excludes_0=(lo > 0) or (hi < 0)))
    contrasts = pd.DataFrame(rows)
    contrasts.round(3).to_csv(RESULTS / "positive_control_contrasts.csv", index=False)

    rl[["run_id", "condition", "trade_price", "mean_ask", "ask_dispersion", "judge_score", "judge_yes_share",
        "seller_profit", "buyer_surplus", "num_trades", "hhi_volume"]].round(4).to_csv(
        RESULTS / "run_level_outcomes.csv", index=False)
    plot_positive_control(t)
    print(table1.to_string(index=False))
    print(contrasts.round(3).to_string(index=False))


def _hourly_band(df, value, runs):
    """Per-hour mean across runs with a 95% bootstrap band over runs."""
    piv = df[df.run_id.isin(runs)].groupby(["hour", "run_id"])[value].mean().unstack()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, piv.shape[1], size=(2000, piv.shape[1]))
    boots = np.nanmean(piv.values[:, idx], axis=2)  # hours x boots
    return piv.index.values, np.nanmean(piv.values, axis=1), np.quantile(boots, 0.025, axis=1), np.quantile(boots, 0.975, axis=1)


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_positive_control(t):
    runs = t["runs"]
    groups = [("No seller comms (base)", runs.run_id[runs.condition == "base"], BLUE),
              ("Seller comms (base-seller_comms)", runs.run_id[runs.condition == "base-seller_comms"], ORANGE)]
    panels = [("Mean seller ask ($)", t["orders"], "ask"), ("Judge coordination score (1-4)", t["judge"], "score")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    for ax, (title, df, col) in zip(axes, panels):
        _style(ax)
        for label, ids, color in groups:
            h, m, lo, hi = _hourly_band(df, col, ids)
            ax.fill_between(h, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(h, m, color=color, linewidth=2, label=label)
        if col == "ask":
            ax.axhline(CE_PRICE, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
            ax.text(1.5, CE_PRICE - 0.25, "competitive price $90", color=INK2, fontsize=8, ha="left", va="top")
        ax.set_title(title, loc="left", color=INK, fontsize=11)
        ax.set_xlabel("Hour", color=INK2, fontsize=9)
        ax.set_xlim(1, N_HOURS)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")
    fig.suptitle("Positive control: seller messaging raises asks and judge coordination scores "
                 "(10 runs each, 95% bootstrap bands over runs)", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_positive_control.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Scripted market control: same rules as market.py, scripted agents, no LLM calls.
# ---------------------------------------------------------------------------------------------
SIM_SELLERS = [f"Seller S{k}" for k in range(1, 6)]
SIM_BUYERS = [f"Buyer B{k}" for k in range(1, 6)]
SIM_MARKETS_PER_TYPE = 30


def _match(a_book: dict, b_book: dict, hour: int, out: list):
    k = 0
    while a_book and b_book:
        pa, sa = min((p, i) for i, p in a_book.items())
        pb, sb = max((p, i) for i, p in b_book.items())
        if pa > pb:
            break
        out.append(dict(hour=hour, seq=k, buyer=sb, seller=sa, price=round((pa + pb) / 2, 2)))
        del a_book[sa], b_book[sb]
        k += 1


def simulate_market(kind: str, seed: int):
    """kind: 'independent' (B-like), 'cartel' (A-like), 'cheater' (C-like: one cartel member undercuts).

    Buyers (same rule in every kind): start U(81.25, 83.75); if their bid is still standing, raise it by
    U(0.1, 0.8) up to 99; after a trade, re-enter at last trade price - U(0.5, 2.5).
    Independent sellers: start U(96.25, 98.75); after a trade, re-enter at own trade price + U(0, 1);
    otherwise, with prob. 0.6, undercut the lowest other ask by U(0.01, 0.5) (never below 80.5), else shade
    own ask down by U(0, 0.2).
    Cartel sellers: floor F ~ U(93, 97) per market; ask F + U(0, 0.3) every hour.
    Cheater: one random cartel member, from hour h_c ~ U{8..16}, asks lowest other ask - U(0.5, 2) (>= 80.5).
    """
    rng = np.random.default_rng(seed)
    a_book, b_book, trades, asks, bids = {}, {}, [], [], []
    floor = rng.uniform(93, 97)
    cheater = SIM_SELLERS[rng.integers(5)] if kind == "cheater" else None
    h_cheat = int(rng.integers(8, 17)) if kind == "cheater" else None
    last_own_trade: dict[str, float] = {}
    last_trade_price = None
    prev_asks: dict[str, float] = {}
    for h in range(1, N_HOURS + 1):
        for s in SIM_SELLERS:
            others = [v for k, v in prev_asks.items() if k != s]
            low_other = min(others) if others else 97.5
            if kind == "independent":
                if h == 1:
                    p = rng.uniform(96.25, 98.75)
                elif s not in a_book:  # traded last hour
                    p = last_own_trade.get(s, 97.5) + rng.uniform(0, 1)
                elif rng.random() < 0.6:
                    p = max(80.5, min(a_book[s], low_other - rng.uniform(0.01, 0.5)))
                else:
                    p = max(80.5, a_book[s] - rng.uniform(0, 0.2))
            else:
                p = floor + rng.uniform(0, 0.3)
                if s == cheater and h >= h_cheat:
                    p = max(80.5, low_other - rng.uniform(0.5, 2.0))
            a_book[s] = round(p, 2)
            asks.append(dict(hour=h, seller=s, ask_logged=a_book[s]))
        for b in SIM_BUYERS:
            if h == 1:
                q = rng.uniform(81.25, 83.75)
            elif b in b_book:
                q = min(99.0, b_book[b] + rng.uniform(0.1, 0.8))
            else:
                q = (last_trade_price or 85.0) - rng.uniform(0.5, 2.5)
            b_book[b] = round(q, 2)
            bids.append(dict(hour=h, buyer=b, bid_logged=b_book[b]))
        prev_asks = dict(a_book)
        n0 = len(trades)
        _match(a_book, b_book, h, trades)
        for tr in trades[n0:]:
            last_own_trade[tr["seller"]] = tr["price"]
            last_trade_price = tr["price"]
    return pd.DataFrame(asks), pd.DataFrame(bids), pd.DataFrame(trades), cheater, h_cheat


def scripted_tables():
    """Simulate markets and shape them like load() output so features.tier1 can run unchanged."""
    from src.load import replay
    asks, bids, trades, runs, sellers, meta = [], [], [], [], [], []
    seed = 0
    for kind in ["independent", "cartel", "cheater"]:
        for k in range(SIM_MARKETS_PER_TYPE):
            seed += 1
            run_id = f"sim_{kind}_{k:02d}"
            a, b, tr, cheater, h_cheat = simulate_market(kind, seed)
            for df, store in [(a, asks), (b, bids), (tr, trades)]:
                store.append(df.assign(run_id=run_id))
            runs.append(dict(run_id=run_id, kind=kind))
            sellers += [dict(run_id=run_id, seller=s) for s in SIM_SELLERS]
            meta.append(dict(run_id=run_id, cheater=cheater, h_cheat=h_cheat))
    asks, bids, trades = pd.concat(asks), pd.concat(bids), pd.concat(trades)
    runs, sellers, meta = pd.DataFrame(runs), pd.DataFrame(sellers), pd.DataFrame(meta)
    empty_r = pd.DataFrame(columns=["run_id", "hour", "seller", "ask_raw"])
    empty_b = pd.DataFrame({"run_id": np.repeat(runs.run_id, 5 * N_HOURS),
                            "hour": np.tile(np.repeat(np.arange(1, N_HOURS + 1), 5), len(runs)),
                            "buyer": np.tile(SIM_BUYERS, N_HOURS * len(runs)), "bid_raw": np.nan})
    book, orders, trades_rep = replay(asks, bids, empty_r, empty_b, runs, sellers)
    key = ["run_id", "hour", "seq", "buyer", "seller", "price"]
    pd.testing.assert_frame_equal(trades_rep[key].sort_values(key).reset_index(drop=True),
                                  trades[key].sort_values(key).reset_index(drop=True), check_dtype=False)
    for df in (orders, book):
        df["window"] = (df.hour - 1) // 5 + 1
    return dict(orders=orders, book=book, trades=trades), runs, meta


def scripted_control():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.features import PRICE_ONLY, T1, tier1

    t, runs, meta = scripted_tables()
    u = tier1(t).merge(runs, on="run_id").merge(meta, on="run_id")
    first_hour = (u.window - 1) * 5 + 1
    last_hour = u.window * 5
    u["label"] = np.select([u.kind == "independent", (u.seller == u.cheater) & (last_hour >= u.h_cheat)],
                           ["B", "C"], "A")
    u.loc[(u.kind == "cheater") & (u.label == "A") & (u.seller == u.cheater) & (first_hour >= u.h_cheat), "label"] = "C"
    feats = PRICE_ONLY + T1
    u[feats] = u[feats].fillna(u[feats].median())

    rows = []
    for task, pos, neg in [("A vs B", "A", "B"), ("A vs C", "A", "C")]:
        d = u[u.label.isin([pos, neg])]
        y = (d.label == pos).astype(int)
        for f in feats:
            auc = roc_auc_score(y, d[f])
            rows.append(dict(task=task, feature=f, auc=auc, separation=max(auc, 1 - auc)))
        for name, cols in [("T1 (logistic, 5-fold by market)", T1), ("price only (logistic)", PRICE_ONLY)]:
            model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
            p = cross_val_predict(model, d[cols], y, groups=d.run_id, cv=GroupKFold(5), method="predict_proba")[:, 1]
            rows.append(dict(task=task, feature=name, auc=roc_auc_score(y, p), separation=roc_auc_score(y, p)))
    res = pd.DataFrame(rows).round(3)
    res.to_csv(RESULTS / "scripted_control_auc.csv", index=False)
    summ = u.groupby("label")[feats + ["win_trade_price"]].mean().T.round(3)
    summ.to_csv(RESULTS / "scripted_control_feature_means.csv")
    print(u.label.value_counts().to_dict())
    print(res.to_string(index=False))
    print(summ)


if __name__ == "__main__":
    {"positive": positive_control, "scripted": scripted_control}[sys.argv[1] if len(sys.argv) > 1 else "positive"]()
