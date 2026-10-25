"""EXPLORATORY (post-hoc) ladder analysis, defined in notes/decisions.md on 25 Oct after the test results were seen.

Do asks rise with communication evidence when time and setting are held fixed?
  Spec 1  comms runs, rungs vs D0, condition x window fixed effects (same condition, same 5-hour window)
  Spec 2  comms runs, rungs vs D0, run x window fixed effects (same market, same window)
  Spec 3  all runs, rungs vs B, window fixed effects (B exists only in `base`)
Outcome: ask_minus_ce. 95% CIs: 2,000 bootstrap resamples of runs within condition.

Usage:  python -m src.ladder_explore   -> results/ladder_exploratory.csv, results/fig_ladder_exploratory.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.controls import BLUE, GRID, INK, INK2, ORANGE, SURFACE, _style
from src.evaluate import RESULTS, units

N_BOOT = 2000
SEED = 20261027
NAMES = {"B": "B: no talk possible", "D0": "D0: talk, no coordination content", "D1": "D1: vague talk only",
         "D2": "D2: invitation not accepted", "AC": "Agreement (A + C)"}
SPECS = {  # name: (sample, reference rung, other rungs, fixed-effect column)
    "condition x window": ("comms", "D0", ["D1", "D2", "AC"], "cw"),
    "run x window": ("comms", "D0", ["D1", "D2", "AC"], "rw"),
    "B vs comms, window": ("all", "B", ["D0", "D1", "D2", "AC"], "window"),
}


def prep() -> pd.DataFrame:
    u = units()
    u = u[u.ask_minus_ce.notna()].copy()
    u["rung"] = u.ladder.replace({"A": "AC", "C": "AC"})
    u["cw"] = u.condition + "|" + u.window.astype(str)
    u["rw"] = u.run_id + "|" + u.window.astype(str)
    return u.reset_index(drop=True)


def _demean(a: np.ndarray, codes: np.ndarray) -> np.ndarray:
    return a - (np.bincount(codes, weights=a) / np.bincount(codes))[codes]


def fe_ols(d: pd.DataFrame, ref: str, rungs: list[str], fe: str) -> dict[str, float]:
    """Rung coefficients vs `ref` with fixed effects for `fe` (within transformation)."""
    codes = pd.factorize(d[fe])[0]
    y = _demean(d.ask_minus_ce.to_numpy(float), codes)
    X = np.column_stack([_demean((d.rung == r).to_numpy(float), codes) for r in rungs])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    identified = X.var(axis=0) > 1e-12
    return {r: (b if ok else np.nan) for r, b, ok in zip(rungs, beta, identified)}


def _steps(coef: dict, ref: str, rungs: list[str]) -> dict[str, float]:
    chain = [ref] + rungs
    val = {ref: 0.0, **coef}
    return {f"{a} -> {b}": val[b] - val[a] for a, b in zip(chain[:-1], chain[1:])}


def main():
    u = prep()
    rng = np.random.default_rng(SEED)
    rows = []
    for spec, (sample, ref, rungs, fe) in SPECS.items():
        d = u if sample == "all" else u[u.seller_comms]
        d = d[d.rung.isin([ref] + rungs)].reset_index(drop=True)
        mixed = d.groupby(fe).rung.transform("nunique") > 1
        point = fe_ols(d, ref, rungs, fe)
        by_cond = {c: g.run_id.unique() for c, g in d.groupby("condition")}
        idx = {r: np.flatnonzero(d.run_id.to_numpy() == r) for r in d.run_id.unique()}
        boots = []
        for _ in range(N_BOOT):
            ix = np.concatenate([idx[r] for runs in by_cond.values() for r in rng.choice(runs, len(runs))])
            bd = d.iloc[ix].copy()
            bd["rw"] = bd.run_id + "|" + bd.window.astype(str)  # duplicated runs merge into one cell
            c = fe_ols(bd, ref, rungs, fe)
            boots.append({**c, **_steps(c, ref, rungs)})
        boots = pd.DataFrame(boots)
        estimates = {**point, **_steps(point, ref, rungs)}
        for k, v in estimates.items():
            is_step = "->" in k
            rows.append(dict(spec=spec, contrast=k if is_step else f"{k} vs {ref}", kind="step" if is_step else "vs reference",
                             estimate=v, lo=np.nanquantile(boots[k], 0.025), hi=np.nanquantile(boots[k], 0.975),
                             n_units=len(d), n_units_in_mixed_cells=int(mixed.sum()),
                             n_rung_in_mixed_cells=(int((mixed & (d.rung == k)).sum()) if not is_step else np.nan)))
    res = pd.DataFrame(rows)
    res.round(4).to_csv(RESULTS / "ladder_exploratory.csv", index=False)
    plot(res)
    print(res.round(3).to_string(index=False))


def plot(res: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
    panels = [(axes[0], [("B vs comms, window", BLUE, "same window")], "B", ["D0", "D1", "D2", "AC"],
               "vs B (no talk possible), same window"),
              (axes[1], [("condition x window", ORANGE, "same condition and window"),
                         ("run x window", BLUE, "same market and window")], "D0", ["D1", "D2", "AC"],
               "vs D0 (talk without coordination content)")]
    for ax, series, ref, rungs, title in panels:
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.grid(axis="y", visible=False)
        y = np.arange(len(rungs))[::-1]
        for k, (spec, color, label) in enumerate(series):
            r = res[(res.spec == spec) & (res.kind == "vs reference")].set_index("contrast")
            r = r.loc[[f"{x} vs {ref}" for x in rungs]]
            off = (k - (len(series) - 1) / 2) * 0.22
            ax.errorbar(r.estimate, y + off, xerr=[r.estimate - r.lo, r.hi - r.estimate], fmt="o", color=color,
                        elinewidth=2, capsize=0, markersize=6, label=label)
        ax.axvline(0, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_yticks(y, [NAMES[x] for x in rungs], color=INK, fontsize=9)
        ax.set_title(f"Difference in mean ask ($)\n{title}", loc="left", color=INK, fontsize=10)
        if len(series) > 1:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.1),
                      ncol=2)
    fig.suptitle("EXPLORATORY (post-hoc): ladder rungs compared within the same window "
                 "(95% CI, bootstrap over runs)", x=0.01, ha="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_ladder_exploratory.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["plot"]:  # redraw from the saved CSV without recomputing
        plot(pd.read_csv(RESULTS / "ladder_exploratory.csv"))
    else:
        main()
