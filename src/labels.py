"""Rule-based agreement labeller, v2: follows Andrew's codebook (notes/codebook.md, v1).

v1 (pre-registered 2026-10-24 03:19, commit 2bd2867) matched any two price statements within one hour.
v2 (2026-10-24, before any hand label was seen) applies the codebook:
  Rule 1  invitation = a sentence with an invitation cue AND a price, or a concrete no-undercut / turn-taking rule
  Rule 2  assent only in the NEXT hour (sellers see only the previous hour's messages); a re-post of the same
          level, a matching price in an assent sentence, or explicit assent words with no other price
  Rule 3  VAGUE flag: vague cue in a message with no invitation and no assent
  Rule 4  words reading (main) vs Interstate Circuit reading (sensitivity: silent compliance also counts)
  Rule 5  carry-over until replaced or withdrawn (stated end times are NOT implemented)
Adherence: price agreement -> asks >= level - $0.50; "no number" rule -> asks >= lowest other ask
in the same hour - $0.50.

The v1 cue lists below are kept unchanged because the pre-registered Tier 2 features use them.

Usage:  python -m src.labels               (labels all units; prints TRAIN-half counts only)
        python -m src.labels sensitivity   (train-half sensitivity grid -> results/labeller_sensitivity_train.csv)
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from src.load import N_HOURS, N_WINDOWS, ROOT, WINDOW, load

LEVEL_MATCH = 0.50  # a reply matches an invited level if within $0.50
ADHERE_TOL = 0.50   # price: asks >= level - $0.50; rule: asks >= lowest other ask - $0.50
COMPLY_HOURS = 2    # Interstate reading: asks conform in the first 2 hours after the invitation is seen
PRICE_MIN, PRICE_MAX = 80.0, 120.0

SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\s*[—–]\s*|\s+-\s+")
PRICE = re.compile(r"\$\s?(\d{2,3}(?:\.\d{1,2})?)")
BID_CTX = re.compile(r"bid|buyer")
ASK_CTX = re.compile(r"ask|hold|floor|price|level")

# ---- v1 cue lists: used by the pre-registered Tier 2 features (src/features.py). Do not change. ----
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

# ---- v2 cue lists: used by the labeller only ----
INVITE = re.compile(
    r"\b(let'?s|let us|we should|we need to|we must|we can all|we all|we each|all of us|each of us|everyone|"
    r"everybody|all sellers|every seller|together|unified|unite|united|collective(?:ly)?|"
    r"i (?:propose|suggest|urge|recommend|encourage)|propose|suggest|urge|join me|stick (?:to|with)|"
    r"if (?:everyone|everybody|all of you|others|the rest|you all|we all))\b")
RULE = re.compile(
    r"(?:not|n't|no|never|stop|avoid|without)\s+(?:\w+\s+){0,2}?undercut\w*\s+(?:each other|one another)|"
    r"no (?:one|seller|body) (?:undercuts|goes below)|no undercutting|take turns|taking turns|rotate (?:fills|trades|turns)|"
    r"alternate (?:fills|trades|turns)")
ASSENT2 = re.compile(
    r"\b(agreed|i agree|we agree|i'?m in|count me in|on board|sounds good|works for me|i support|"
    r"as (?:planned|agreed|discussed|proposed)|aligned|(?:i'?ve |i have |i'?m |i am )?join(?:ed|ing) "
    r"(?:you|in|the|at|this|everyone|all)|i'?ll (?:join|follow|match))\b")
WITHDRAW = re.compile(
    r"(?:set(?:ting)?|pric(?:e|ing)) (?:my|our) own (?:price|ask)|my own pric|independent(?:ly)? pric|"
    r"no longer (?:part of|participating|bound|following|holding to|committed)|opt(?:ing)? out|"
    r"going (?:my own way|it alone)|not bound by|won'?t (?:follow|be bound)")
VAGUE = re.compile(
    r"price war|race to the bottom|healthy (?:market|margins?|prices|competition)|sustainab|disciplin|cooperat|"
    r"collaborat|work(?:ing)? together|mutual(?:ly)? benefi|benefit (?:us )?all|stay (?:strong|united|firm|patient)|"
    r"hold (?:the line|firm|steady)|(?:avoid|no) (?:aggressive|unnecessary|large|big|deep|panic) "
    r"(?:cuts?|undercut\w*|reductions?)|(?:united|unified) front|collective|together")


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


# ---- v1 helpers (Tier 2 features) ----
def price_invites(text: str) -> list[float]:
    """v1: prices that appear in a sentence with a collective cue."""
    return [p for s in sentences(text) if COLLECTIVE.search(s) for p in sentence_prices(s)]


def has_assent(text: str) -> bool:
    return bool(ASSENT.search(norm(text)))


def has_cut_talk(text: str) -> bool:
    return bool(CUT.search(norm(text)))


# ---- v2 message classification ----
def classify(text: str) -> dict:
    inv_prices, inv_rule, assent_prices = [], False, []
    for s in sentences(text):
        ps = sentence_prices(s)
        if INVITE.search(s):
            inv_prices += ps
            if not ps and RULE.search(s):
                inv_rule = True
        if ASSENT2.search(s):
            assent_prices += ps
    n = norm(text)
    assent_words = bool(ASSENT2.search(n))
    definite = bool(inv_prices) or inv_rule or assent_words
    return dict(inv_prices=inv_prices, inv_rule=inv_rule, assent_words=assent_words, assent_prices=assent_prices,
                withdraw=bool(WITHDRAW.search(n)), vague=bool(VAGUE.search(n)) and not definite)


def classify_messages(messages: pd.DataFrame) -> pd.DataFrame:
    cols = ["inv_prices", "inv_rule", "assent_words", "assent_prices", "withdraw", "vague"]
    c = pd.DataFrame([classify(x) for x in messages.text], index=messages.index, columns=cols)
    c[["inv_rule", "assent_words", "withdraw", "vague"]] = c[["inv_rule", "assent_words", "withdraw", "vague"]].astype(bool)
    return pd.concat([messages[["run_id", "hour", "seller"]], c], axis=1)


def _conforms(asks: dict, run_id, seller, hour, kind, level, others_min, tol) -> bool:
    a = asks.get((run_id, seller, hour), np.nan)
    if np.isnan(a):
        return True
    ref = level if kind == "price" else others_min.get((run_id, seller, hour), np.nan)
    return np.isnan(ref) or a >= ref - tol


def agreement_events(cm: pd.DataFrame, t: dict | None = None, reading: str = "words",
                     level_match: float = LEVEL_MATCH, adhere_tol: float = ADHERE_TOL) -> pd.DataFrame:
    """Per-seller agreement events (run_id, seller, hour, kind, level, partner, via).

    An invitation at hour h is accepted by a different seller's message at hour h+1 (Rule 2). Both become
    parties from hour h+1. Under reading='interstate', a seller who sends no counter-proposal at h+1 and whose
    asks conform at h+1..h+COMPLY_HOURS also becomes a party (silent compliance, Rule 4 sensitivity).
    """
    rows = []
    if reading == "interstate":
        asks, others_min = _ask_lookups(t)
        sellers_by_run = t["sellers"].groupby("run_id").seller.apply(list).to_dict()
    by_key = {(r.run_id, r.hour, r.seller): r for r in cm.itertuples(index=False)}
    by_hour: dict[tuple, list] = {}
    for r in cm.itertuples(index=False):
        by_hour.setdefault((r.run_id, r.hour), []).append(r)
    for r in cm.itertuples(index=False):
        invites = [("price", p) for p in r.inv_prices] + ([("rule", np.nan)] if r.inv_rule else [])
        if not invites:
            continue
        h, i = r.hour, r.seller
        replies = [x for x in by_hour.get((r.run_id, h + 1), []) if x.seller != i]
        replied = set()
        for j in replies:
            j_prices = j.inv_prices + j.assent_prices
            for kind, level in invites:
                if kind == "price":
                    ok = any(abs(p - level) <= level_match for p in j_prices) or (j.assent_words and not j_prices)
                else:
                    ok = j.inv_rule or (j.assent_words and not j_prices)
                if ok:
                    replied.add(j.seller)
                    for a, b in [(i, j.seller), (j.seller, i)]:
                        rows.append(dict(run_id=r.run_id, seller=a, hour=h + 1, kind=kind, level=level, partner=b,
                                         via="words"))
        if reading == "interstate":
            for s in sellers_by_run[r.run_id]:
                if s == i or s in replied or h + 1 > N_HOURS:
                    continue
                j = by_key.get((r.run_id, h + 1, s))
                hours = range(h + 1, min(h + COMPLY_HOURS, N_HOURS) + 1)
                for kind, level in invites:
                    counter = j is not None and kind == "price" and any(abs(p - level) > level_match for p in j.inv_prices) \
                        and not any(abs(p - level) <= level_match for p in j.inv_prices)
                    if counter:
                        continue
                    if all(_conforms(asks, r.run_id, s, hh, kind, level, others_min, adhere_tol) for hh in hours):
                        for a, b in [(i, s), (s, i)]:
                            rows.append(dict(run_id=r.run_id, seller=a, hour=h + 1, kind=kind, level=level, partner=b,
                                             via="conduct"))
    ev = pd.DataFrame(rows, columns=["run_id", "seller", "hour", "kind", "level", "partner", "via"])
    return ev.drop_duplicates().sort_values(["run_id", "seller", "hour"]).reset_index(drop=True)


def _ask_lookups(t):
    o = t["orders"][["run_id", "hour", "seller", "ask"]]
    asks = o.set_index(["run_id", "seller", "hour"]).ask.to_dict()
    others_min = {}
    for (run_id, hour), g in o.groupby(["run_id", "hour"]):
        vals = dict(zip(g.seller, g.ask))
        for s in vals:
            rest = [v for k, v in vals.items() if k != s and not np.isnan(v)]
            others_min[(run_id, s, hour)] = min(rest) if rest else np.nan
    return asks, others_min


def labels_from_events(t: dict, ev: pd.DataFrame, withdrawals: pd.DataFrame | None = None,
                       vague: pd.DataFrame | None = None, adhere_tol: float = ADHERE_TOL) -> pd.DataFrame:
    """A/C/D/B unit labels from agreement events (run_id, seller, hour, kind, level), any source.

    Party at hour x: the latest event at or before x exists and the seller has not withdrawn since (Rule 5).
    Agreed in a window if a party at any hour of it. Adherence is checked over the window's hours under the
    latest active agreement. Ties at the same hour: price agreements before rules, then the lowest level.
    """
    runs = t["runs"].set_index("run_id")
    asks, others_min = _ask_lookups(t)
    if "kind" not in ev:
        ev = ev.assign(kind=np.where(ev.level.isna(), "rule", "price"))
    ev = ev.assign(_rank=(ev.kind == "rule").astype(int)).sort_values(["hour", "_rank", "level"],
                                                                      ascending=[True, False, False])
    ev_by = {k: g for k, g in ev.groupby(["run_id", "seller"])}  # within an hour, the LAST row is the preferred one
    wd_by = {} if withdrawals is None else withdrawals.groupby(["run_id", "seller"]).hour.apply(sorted).to_dict()
    vg = set() if vague is None else set(map(tuple, vague[["run_id", "seller", "window"]].to_numpy()))

    rows = []
    for s in t["sellers"].itertuples(index=False):
        base = not runs.seller_comms[s.run_id]
        e = ev_by.get((s.run_id, s.seller))
        wds = wd_by.get((s.run_id, s.seller), [])
        for w in range(1, N_WINDOWS + 1):
            first, last = WINDOW * (w - 1) + 1, WINDOW * w
            row = dict(run_id=s.run_id, seller=s.seller, window=w, label="B" if base else "D", agreed=False,
                       agreement_kind=None, agreed_level=np.nan, event_hour=np.nan, withdrawn_hour=np.nan,
                       min_margin=np.nan, vague=(s.run_id, s.seller, w) in vg)
            if not base and e is not None:
                active = {}
                for x in range(first, last + 1):
                    past = e[e.hour <= x]
                    if not len(past):
                        continue
                    cur = past.iloc[-1]
                    out = [wh for wh in wds if cur.hour <= wh <= x]
                    if out:
                        row["withdrawn_hour"] = out[0]
                        continue
                    active[x] = cur
                if active:
                    last_x = max(active)
                    cur = active[last_x]
                    hours = [x for x, c in active.items() if c.hour == cur.hour and c.kind == cur.kind
                             and (c.level == cur.level or (np.isnan(c.level) and np.isnan(cur.level)))]
                    margins = []
                    for x in hours:
                        a = asks.get((s.run_id, s.seller, x), np.nan)
                        ref = cur.level if cur.kind == "price" else others_min.get((s.run_id, s.seller, x), np.nan)
                        if not (np.isnan(a) or np.isnan(ref)):
                            margins.append(a - ref)
                    m = min(margins) if margins else np.nan
                    adhered = np.isnan(m) or m >= -adhere_tol
                    row.update(label="A" if adhered else "C", agreed=True, agreement_kind=cur.kind,
                               agreed_level=cur.level, event_hour=cur.hour, min_margin=m)
            rows.append(row)
    return pd.DataFrame(rows)


def unit_labels(t: dict, reading: str = "words", adhere_tol: float = ADHERE_TOL,
                level_match: float = LEVEL_MATCH, cm: pd.DataFrame | None = None) -> pd.DataFrame:
    cm = classify_messages(t["messages"]) if cm is None else cm
    ev = agreement_events(cm, t, reading=reading, level_match=level_match, adhere_tol=adhere_tol)
    wd = cm.loc[cm.withdraw, ["run_id", "seller", "hour"]]
    vg = cm.loc[cm.vague, ["run_id", "seller", "hour"]].assign(window=lambda d: (d.hour - 1) // WINDOW + 1)
    return labels_from_events(t, ev, wd, vg, adhere_tol)


def sensitivity():
    """Train half only: how labels move with the adherence tolerance and level-match width (reply lag fixed at
    one hour by the codebook). The frozen labels are not changed."""
    import itertools
    t = load()
    train = set(json.loads((ROOT / "results" / "split.json").read_text())["train"])
    cm = classify_messages(t["messages"])
    key = ["run_id", "seller", "window"]
    frozen = unit_labels(t, cm=cm).set_index(key).label
    rows = []
    for reading, tol, match in itertools.product(["words", "interstate"], [0.25, 0.50, 1.00], [0.25, 0.50, 1.00]):
        lab = unit_labels(t, reading=reading, adhere_tol=tol, level_match=match, cm=cm)
        lab = lab[lab.run_id.isin(train)].set_index(key).label
        c = lab.value_counts()
        rows.append(dict(reading=reading, adhere_tol=tol, level_match=match, A=c.get("A", 0), C=c.get("C", 0),
                         D=c.get("D", 0), B=c.get("B", 0),
                         same_as_frozen_words=round((lab == frozen.loc[lab.index]).mean(), 3),
                         frozen=(reading, tol, match) == ("words", 0.5, 0.5)))
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "labeller_sensitivity_train.csv", index=False)
    print(res.to_string(index=False))


def main():
    t = load()
    cm = classify_messages(t["messages"])
    lab = unit_labels(t, cm=cm)
    lab_ic = unit_labels(t, reading="interstate", cm=cm)
    lab.to_pickle(ROOT / "data" / "processed" / "labels.pkl")
    lab_ic.to_pickle(ROOT / "data" / "processed" / "labels_interstate.pkl")
    train = set(json.loads((ROOT / "results" / "split.json").read_text())["train"])
    for name, L in [("words (main)", lab), ("Interstate Circuit reading", lab_ic)]:
        tr = L[L.run_id.isin(train)].merge(t["runs"][["run_id", "condition"]], on="run_id")
        print(f"TRAIN, {name}:", tr.label.value_counts().to_dict(),
              "| rule agreements:", int((tr.agreement_kind == "rule").sum()), "| vague units:", int(tr.vague.sum()))
    tr = lab[lab.run_id.isin(train)].merge(t["runs"][["run_id", "condition"]], on="run_id")
    print(pd.crosstab(tr.condition, tr.label))
    print("train messages flagged withdraw:", int(cm[cm.run_id.isin(train)].withdraw.sum()))


if __name__ == "__main__":
    import sys
    sensitivity() if sys.argv[1:] == ["sensitivity"] else main()
