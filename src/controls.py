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


if __name__ == "__main__":
    {"positive": positive_control}[sys.argv[1] if len(sys.argv) > 1 else "positive"]()
