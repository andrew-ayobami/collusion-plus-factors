"""Rule-based agreement labeller (pre-registered in notes/decisions.md, 2026-10-24 04:40).

Unit = (run, seller, 5-hour window). Labels: B (base runs), and in comms runs
A (agreed + adhered), C (agreed + not adhered), D (no agreement).

Usage:  python -m src.labels      (labels all units; prints TRAIN-half counts only)
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from src.load import N_WINDOWS, ROOT, WINDOW, load

LEVEL_MATCH = 0.50  # two price statements match if within $0.50
ADHERE_TOL = 0.50   # asks must stay >= agreed level - $0.50
PRICE_MIN, PRICE_MAX = 80.0, 120.0

SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\s*[—–]\s*|\s+-\s+")
PRICE = re.compile(r"\$\s?(\d{2,3}(?:\.\d{1,2})?)")
BID_CTX = re.compile(r"bid|buyer")
ASK_CTX = re.compile(r"ask|hold|floor|price|level")
COLLECTIVE = re.compile(
    r"\b(let'?s|let us|we should|we all|all of us|everyone|everybody|together|unified|unite|united|"
    r"collective(?:ly)?|coordinat\w*|join\w*|match\w*|floor|hold (?:the line|firm|at|our|steady)|"
    r"stay (?:at|above)|keep (?:our )?asks?|agree\w*|as (?:planned|discussed|agreed|proposed)|"
    r"propose\w*|suggest\w*|urge\w*)\b")
ASSENT = re.compile(
    r"\b(agree\w*|join\w*|as (?:planned|discussed|agreed|proposed)|on board|i'?m in|count me in|"
    r"sounds good|i support|match\w*)\b")
CUT = re.compile(r"\b(?:i'?ve|i have|i will|i'?ll|i am|i'?m)\s+(?:\w+\s+){0,3}?"
                 r"(?:undercut\w*|lower\w*|reduc\w*|cut\w*|drop\w*)\b")


def norm(text: str) -> str:
    return (text or "").replace("’", "'").replace("‘", "'").lower()


def sentences(text: str) -> list[str]:
    return [s for s in SENT_SPLIT.split(norm(text)) if s.strip()]


def sentence_prices(s: str) -> list[float]:
    out = []
    for m in PRICE.finditer(s):
        p = float(m.group(1))
        if not (PRICE_MIN < p <= PRICE_MAX):
            continue
        ctx = s[max(0, m.start() - 25):m.start()]
        if BID_CTX.search(ctx) and not ASK_CTX.search(ctx):
            continue
        out.append(p)
    return out


def price_invites(text: str) -> list[float]:
    """Prices that appear in a sentence with a collective cue."""
    return [p for s in sentences(text) if COLLECTIVE.search(s) for p in sentence_prices(s)]


def has_assent(text: str) -> bool:
    return bool(ASSENT.search(norm(text)))


def has_cut_talk(text: str) -> bool:
    return bool(CUT.search(norm(text)))


def price_statements(messages: pd.DataFrame) -> pd.DataFrame:
    rows = [dict(run_id=r.run_id, hour=r.hour, seller=r.seller, price=p)
            for r in messages.itertuples(index=False) for p in price_invites(r.text)]
    return pd.DataFrame(rows, columns=["run_id", "hour", "seller", "price"])


def agreement_events(statements: pd.DataFrame) -> pd.DataFrame:
    """Per-seller agreement events: (run, seller, hour, level, partner)."""
    rows = []
    for run_id, g in statements.groupby("run_id"):
        st = g[["hour", "seller", "price"]].to_numpy(dtype=object)
        for hi, si, pi in st:
            for hj, sj, pj in st:
                if sj == si or hj not in (hi, hi + 1) or abs(pi - pj) > LEVEL_MATCH:
                    continue
                hour, level = max(hi, hj), min(pi, pj)
                rows.append(dict(run_id=run_id, seller=si, hour=hour, level=level, partner=sj))
                rows.append(dict(run_id=run_id, seller=sj, hour=hour, level=level, partner=si))
    ev = pd.DataFrame(rows, columns=["run_id", "seller", "hour", "level", "partner"])
    return ev.drop_duplicates().sort_values(["run_id", "seller", "hour", "level"]).reset_index(drop=True)


def unit_labels(t: dict) -> pd.DataFrame:
    runs = t["runs"].set_index("run_id")
    orders = t["orders"]
    ev = agreement_events(price_statements(t["messages"]))
    ev_by = {k: g for k, g in ev.groupby(["run_id", "seller"])}
    ask = orders.set_index(["run_id", "seller", "hour"]).ask.sort_index()

    rows = []
    for s in t["sellers"].itertuples(index=False):
        base = not runs.seller_comms[s.run_id]
        e = ev_by.get((s.run_id, s.seller))
        for w in range(1, N_WINDOWS + 1):
            first, last = WINDOW * (w - 1) + 1, WINDOW * w
            row = dict(run_id=s.run_id, seller=s.seller, window=w, label="B" if base else "D",
                       agreed_level=np.nan, event_hour=np.nan, min_ask_after=np.nan)
            if not base and e is not None and (e.hour <= last).any():
                past = e[e.hour <= last]
                h = past.hour.max()
                level = past[past.hour == h].level.min()  # ties at the latest hour: lowest level (lenient)
                asks = ask.loc[(s.run_id, s.seller)].loc[max(h, first):last].dropna()
                m = asks.min() if len(asks) else np.nan
                adhered = (not len(asks)) or m >= level - ADHERE_TOL
                row.update(label="A" if adhered else "C", agreed_level=level, event_hour=h, min_ask_after=m)
            rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    t = load()
    lab = unit_labels(t)
    lab.to_pickle(ROOT / "data" / "processed" / "labels.pkl")
    train = set(json.loads((ROOT / "results" / "split.json").read_text())["train"])
    tr = lab[lab.run_id.isin(train)].merge(t["runs"][["run_id", "condition"]], on="run_id")
    print("TRAIN label counts:", tr.label.value_counts().to_dict())
    print(pd.crosstab(tr.condition, tr.label))
    print(pd.crosstab(tr.window, tr.label))
