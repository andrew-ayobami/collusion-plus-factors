"""Figures and descriptive tables, made after the one-time test run.

  python -m src.figures      -> results/fig_test_ab.png, fig_feature_separation.png, fig_harm_by_label.png,
                                results/c_descriptive.csv, results/harm_by_label.csv
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.controls import BLUE, GRID, INK, INK2, ORANGE, SURFACE, _style
from src.evaluate import FEATURE_SETS, RESULTS, units
from src.features import HARM, PRICE_ONLY, T1, T2, T3
from src.stats import boot_mean_ci

MODEL_LABELS = {"price_only": "Price only", "judge_only": "Paper's judge only", "T1": "Tier 1 (conduct)",
                "price+T1": "Price + Tier 1", "T1+T2": "Tier 1 + 2 (messages)*",
                "T1+T3": "Tier 1 + 3 (private reasoning)", "T1+T2+T3": "All tiers*"}


def fig_test_ab():
    m = pd.read_csv(RESULTS / "test_metrics.csv")
    m = m[(m.task == "A_vs_B") & m.scorer.str.startswith("model:")].copy()
    m["model"] = m.scorer.str.replace("model:", "")
    m = m.set_index("model").loc[list(FEATURE_SETS)[::-1]]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True, facecolor=SURFACE)
    for ax, (col, title) in zip(axes, [("tpr", "Detects agreement (TPR on A)"),
                                       ("fpr", "Flags innocent parallel conduct (FPR on B)"),
                                       ("auc", "Overall separation (AUC)")]):
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        y = np.arange(len(m))
        ax.errorbar(m[col], y, xerr=[m[col] - m[f"{col}_lo"], m[f"{col}_hi"] - m[col]], fmt="o", color=BLUE,
                    ecolor=BLUE, elinewidth=2, capsize=0, markersize=7)
        for yi, v in zip(y, m[col]):
            ax.text(v, yi + 0.28, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_xlim(0, 1.02)
        if col == "auc":
            ax.axvline(0.5, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_title(title, loc="left", color=INK, fontsize=10)
    axes[0].set_yticks(np.arange(len(m)), [MODEL_LABELS[k] for k in m.index], color=INK)
    fig.suptitle("Test half, A (agreement acted on) vs B (no communication): 95% CIs by bootstrap over runs",
                 x=0.01, ha="left", color=INK, fontsize=10)
    fig.text(0.01, 0.01, "* Tier 2 features are zero for every B unit (B runs had no messaging), so these models "
                         "separate A from B partly by construction.", color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(RESULTS / "fig_test_ab.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_feature_separation():
    sim = pd.read_csv(RESULTS / "scripted_control_auc.csv")
    sim = sim[(sim.task == "A vs B") & sim.feature.isin(PRICE_ONLY + T1)].set_index("feature").separation
    llm = pd.read_csv(RESULTS / "test_feature_auc.csv").set_index("feature").auc_AvB
    llm = np.maximum(llm, 1 - llm)
    feats = (PRICE_ONLY + T1)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(feats))
    for yi, f in zip(y, feats):
        ax.plot([llm[f], sim[f]], [yi, yi], color=GRID, linewidth=2, zorder=1)
    ax.scatter(sim[feats], y, color=ORANGE, s=50, zorder=2, label="Scripted market (cartel vs independents)")
    ax.scatter(llm[feats], y, color=BLUE, s=50, zorder=3, label="LLM agents, test half (A vs B)")
    ax.axvline(0.5, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_yticks(y, feats, color=INK, fontsize=9)
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel("Separation = max(AUC, 1 − AUC); 0.5 = no signal", color=INK2, fontsize=9)
    ax.set_title("Single plus factors: strong on a scripted cartel, weak on LLM sellers", loc="left", color=INK,
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_feature_separation.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def harm_and_c():
    u = units()
    rows = []
    for lab, g in u.groupby("label"):
        run_means = g.groupby("run_id")[HARM].mean()
        for h in HARM:
            mean, lo, hi = boot_mean_ci(run_means[h])
            rows.append(dict(label=lab, measure=h, n_units=len(g), n_runs=g.run_id.nunique(), mean=mean, lo=lo, hi=hi))
    harm = pd.DataFrame(rows)
    harm.round(4).to_csv(RESULTS / "harm_by_label.csv", index=False)

    c = u[u.label == "C"]
    desc = pd.DataFrame({
        "n_units": c.groupby("condition").size(),
        "n_runs": c.groupby("condition").run_id.nunique(),
        "mean_window": c.groupby("condition").window.mean(),
        "mean_shortfall_below_level": (c.agreed_level - c.min_ask_after).groupby(c.condition).mean(),
        "mean_ask_minus_ce": c.groupby("condition").ask_minus_ce.mean(),
        "mean_judge": c.groupby("condition").t3_judge_mean.mean(),
        "mean_t2_cut_talk": c.groupby("condition").t2_cut_talk.mean(),
        "mean_t3_defect_flags": c.groupby("condition").t3_defect_flags.mean(),
    })
    desc.loc["ALL"] = [len(c), c.run_id.nunique(), c.window.mean(), (c.agreed_level - c.min_ask_after).mean(),
                       c.ask_minus_ce.mean(), c.t3_judge_mean.mean(), c.t2_cut_talk.mean(), c.t3_defect_flags.mean()]
    desc.round(3).to_csv(RESULTS / "c_descriptive.csv")
    comp = u[u.label.isin(["A", "C"])].groupby("label")[PRICE_ONLY + T1 + T2 + T3].mean().T.round(3)
    comp.to_csv(RESULTS / "c_vs_a_feature_means.csv")

    order = ["B", "D", "A", "C"]
    names = {"B": "B: no comms", "D": "D: comms, no agreement", "A": "A: agreement kept", "C": "C: agreement broken"}
    h = harm[harm.measure == "win_trade_price"].set_index("label").loc[order]
    fig, ax = plt.subplots(figsize=(6.5, 3.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(order))[::-1]
    ax.errorbar(h["mean"], y, xerr=[h["mean"] - h.lo, h.hi - h["mean"]], fmt="o", color=BLUE, elinewidth=2,
                capsize=0, markersize=7)
    for yi, (lab, r) in zip(y, h.iterrows()):
        ax.text(r.hi + 0.15, yi, f"${r['mean']:.2f}  ({int(r.n_units)} units, {int(r.n_runs)} runs)", va="center",
                fontsize=8, color=INK2)
    ax.axvline(90, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_yticks(y, [names[k] for k in order], color=INK, fontsize=9)
    ax.set_xlabel("Mean trade price in the window ($); dashed = competitive price $90", color=INK2, fontsize=9)
    ax.set_title("Harm by label: window trade price (all 75 runs, 95% CI over runs)", loc="left", color=INK,
                 fontsize=10)
    lo = np.nanmin(h.lo) - 1
    ax.set_xlim(lo, np.nanmax(h.hi) + 4)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_harm_by_label.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(harm[harm.measure == "win_trade_price"].round(2).to_string(index=False))
    print(desc.round(2))


if __name__ == "__main__":
    fig_test_ab()
    fig_feature_separation()
    harm_and_c()
