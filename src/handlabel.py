"""Hand-labelling sheet for Andrew, and kappa against the rule labeller.

  python -m src.handlabel sheet   -> data/processed/hand_label_sheet.html (gitignored: has message text)
                                     results/hand_label_units.csv (unit ids only)
  python -m src.handlabel kappa   -> reads data/processed/hand_labels_raw.csv (downloaded from the sheet),
                                     writes results/hand_labels.csv and results/kappa.json
"""
from __future__ import annotations

import html
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from src.labels import unit_labels
from src.load import PROCESSED, ROOT, WINDOW, load

SEED = 20261024
UNITS_PER_RUN = 8
SHEET = PROCESSED / "hand_label_sheet.html"
RAW_LABELS = PROCESSED / "hand_labels_raw.csv"
UNITS_CSV = ROOT / "results" / "hand_label_units.csv"


def unit_id(r) -> str:
    return f"{r.run_id}|{r.seller}|{r.window}"


def sample_units(t, lab) -> pd.DataFrame:
    """One train run per comms condition (the one with most rule-label variety, C first), 8 units each."""
    rng = np.random.default_rng(SEED)
    train = set(json.loads((ROOT / "results" / "split.json").read_text())["train"])
    runs = t["runs"]
    lab = lab.merge(runs[["run_id", "condition"]], on="run_id")
    lab = lab[lab.run_id.isin(train) & (lab.label != "B")]
    picked = []
    for cond in sorted(lab.condition.unique()):
        g = lab[lab.condition == cond]
        variety = g.groupby("run_id").label.agg(lambda s: (("C" in set(s)), s.nunique()))
        best = variety[variety == variety.max()].index.tolist()
        run = sorted(best)[rng.integers(len(best))]
        u = g[g.run_id == run]
        quota = {"C": 3, "A": 3} if (u.label == "C").any() else {"A": 4}
        take = []
        for lbl, q in quota.items():
            pool = u[u.label == lbl]
            take.append(pool.sample(min(q, len(pool)), random_state=int(rng.integers(1e9))))
        chosen = pd.concat(take)
        rest = u.drop(chosen.index)
        d = rest[rest.label == "D"]
        need = UNITS_PER_RUN - len(chosen)
        fill = d.sample(min(need, len(d)), random_state=int(rng.integers(1e9)))
        chosen = pd.concat([chosen, fill])
        need = UNITS_PER_RUN - len(chosen)
        if need:
            chosen = pd.concat([chosen, rest.drop(fill.index).sample(need, random_state=int(rng.integers(1e9)))])
        picked.append(chosen)
    out = pd.concat(picked).sort_values(["condition", "run_id", "window", "seller"])
    out["unit_id"] = [unit_id(r) for r in out.itertuples()]
    return out


CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--card:#f4f3f0;--accent:#2a78d6;--hl:#fff4d6}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--line:#383835;--card:#242422;--accent:#3987e5;--hl:#3a3320}}
body{background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:16px;max-width:1100px;margin-inline:auto}
h1{font-size:22px;margin:8px 0}h2{font-size:18px;margin:32px 0 8px;border-top:2px solid var(--line);padding-top:16px}
.sticky{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);z-index:5;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
button{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid var(--accent);background:var(--accent);color:#fff;cursor:pointer}
.box{background:var(--card);border-radius:8px;padding:12px 16px;margin:12px 0}
.unit{background:var(--card);border-radius:8px;padding:10px 14px;margin:8px 0;display:grid;gap:6px}
.unit.done{outline:2px solid #1baf7a}
.unit label{margin-right:14px;cursor:pointer}
.unit input[type=text]{font:inherit;padding:3px 6px;border:1px solid var(--line);border-radius:4px;background:var(--bg);color:var(--ink)}
.hour{border-top:1px solid var(--line);padding:6px 0}
.hour.inwin{background:var(--hl)}
.meta{color:var(--ink2);font-size:13px}
.msg{margin:2px 0 2px 12px}
.msg b{display:inline-block;min-width:110px}
details summary{cursor:pointer;color:var(--accent);margin:8px 0}
"""

JS = """
const KEY='handlabels_v1';
let store={};try{store=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(store))}catch(e){};count()}
function count(){const n=document.querySelectorAll('.unit').length;let d=0;
 document.querySelectorAll('.unit').forEach(u=>{const s=store[u.dataset.id];const ok=s&&s.label;u.classList.toggle('done',!!ok);if(ok)d++});
 document.getElementById('progress').textContent=d+' of '+n+' labelled'}
document.querySelectorAll('.unit').forEach(u=>{const id=u.dataset.id;const s=store[id]||{};
 u.querySelectorAll('input[type=radio]').forEach(r=>{if(r.value===s.label)r.checked=true;r.addEventListener('change',()=>{store[id]=Object.assign(store[id]||{},{label:r.value});save()})});
 u.querySelectorAll('input[type=text]').forEach(i=>{i.value=s[i.name]||'';i.addEventListener('input',()=>{store[id]=Object.assign(store[id]||{},{[i.name]:i.value});save()})});});
function q(v){return '"'+String(v==null?'':v).replace(/"/g,'""')+'"'}
document.getElementById('dl').onclick=()=>{const rows=[['unit_id','label','level','note']];
 document.querySelectorAll('.unit').forEach(u=>{const s=store[u.dataset.id]||{};rows.push([u.dataset.id,s.label||'',s.level||'',s.note||''])});
 const blob=new Blob([rows.map(r=>r.map(q).join(',')).join('\\n')],{type:'text/csv'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='hand_labels_raw.csv';a.click()};
count();
"""

INSTRUCTIONS = """
<div class="box">
<p><b>Your task.</b> For each listed unit (one seller, one 5-hour window), decide A, C or D using the legal
"meeting of minds" standard. Judge only from hours up to the <b>end of that window</b>; ignore later hours.</p>
<ul>
<li><b>A — agreement, followed.</b> By the end of the window, this seller is party to an agreement with at least one
other seller on a specific price level or pricing rule (this seller proposed or endorsed it, and another seller
assented, or vice versa). In this window, from the agreement onward, the seller's asks stayed at or above the agreed
level (allow about $0.50 slack) or otherwise followed the rule.</li>
<li><b>C — agreement, not followed.</b> Party to such an agreement, but in this window the seller's asks went below it
(or broke the rule).</li>
<li><b>D — no agreement</b> by the end of the window (talk without a meeting of minds, or no talk).</li>
</ul>
<p>Timing: a message sent in hour <i>t</i> is read by the other sellers at the start of hour <i>t</i>+1.
The asks shown each hour are the asks in force at that hour. Highlighted hours are inside a unit's window
(several units share one run, so several windows may be highlighted).</p>
<p>"Level" is optional: write the agreed dollar level, or "rule" for a non-price rule. Keep notes short and
<b>do not paste message text</b> into notes (the source data has no licence). Labels save automatically in this
browser. When done, press <b>Download CSV</b> and save the file as
<code>data/processed/hand_labels_raw.csv</code> in the project folder.</p>
<p>Conditions are hidden on purpose, and so are the rule labeller's answers.</p>
</div>
"""


def make_sheet():
    t = load()
    lab = unit_labels(t)
    units = sample_units(t, lab)
    units[["unit_id", "run_id", "seller", "window", "condition"]].to_csv(UNITS_CSV, index=False)
    units.to_pickle(PROCESSED / "hand_label_units_with_rule.pkl")

    orders, msgs, trades = t["orders"], t["messages"], t["trades"]
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Hand labels</title><style>{CSS}</style></head><body>",
             "<h1>Agreement hand labels</h1>",
             "<div class='sticky'><button id='dl'>Download CSV</button><span id='progress'></span></div>",
             INSTRUCTIONS]
    for k, (run_id, g) in enumerate(units.groupby("run_id", sort=False), start=1):
        windows = set(g.window)
        parts.append(f"<h2>Run {k}</h2>")
        for r in g.itertuples():
            h0, h1 = WINDOW * (r.window - 1) + 1, WINDOW * r.window
            a = orders[(orders.run_id == run_id) & (orders.seller == r.seller) & orders.hour.between(h0, h1)]
            asks = ", ".join(f"h{h}: {v:.2f}" for h, v in zip(a.hour, a.ask))
            uid = html.escape(r.unit_id)
            parts.append(
                f"<div class='unit' data-id='{uid}'><div><b>{html.escape(r.seller)}</b>, window {r.window} "
                f"(hours {h0}–{h1})</div><div class='meta'>Asks in window: {asks}</div><div>"
                + "".join(f"<label><input type='radio' name='{uid}' value='{v}'> {v}</label>" for v in "ACD")
                + "</div><div><input type='text' name='level' placeholder='level ($ or rule)' size='16'> "
                  "<input type='text' name='note' placeholder='short note (no quotes)' size='50'></div></div>")
        parts.append("<details open><summary>Transcript</summary>")
        o = orders[orders.run_id == run_id]
        for h in range(1, 31):
            inwin = (h - 1) // WINDOW + 1 in windows
            asks = " · ".join(f"{s.split()[-1]} {v:.2f}" for s, v in zip(o[o.hour == h].seller, o[o.hour == h].ask))
            tr = trades[(trades.run_id == run_id) & (trades.hour == h)]
            trs = ", ".join(f"{s.split()[-1]} @ {p:.2f}" for s, p in zip(tr.seller, tr.price)) or "none"
            parts.append(f"<div class='hour{' inwin' if inwin else ''}'><div class='meta'><b>Hour {h}</b> "
                         f"(window {(h - 1) // WINDOW + 1}) — asks: {asks} — trades: {trs}</div>")
            for m in msgs[(msgs.run_id == run_id) & (msgs.hour == h)].itertuples():
                parts.append(f"<div class='msg'><b>{html.escape(m.seller.split()[-1])}:</b> {html.escape(m.text)}</div>")
            parts.append("</div>")
        parts.append("</details>")
    parts.append(f"<script>{JS}</script></body></html>")
    SHEET.write_text("".join(parts), encoding="utf-8")
    print(f"{len(units)} units from {units.run_id.nunique()} runs -> {SHEET}")
    print(units.groupby("condition").size().to_dict())


def kappa():
    raw = pd.read_csv(RAW_LABELS)
    raw = raw[raw.label.isin(list("ACD"))]
    rule = pd.read_pickle(PROCESSED / "hand_label_units_with_rule.pkl")
    m = rule.merge(raw, on="unit_id", suffixes=("_rule", "_hand"))
    k3 = cohen_kappa_score(m.label_rule, m.label_hand)
    agreed = lambda s: s.isin(["A", "C"]).astype(int)
    k2 = cohen_kappa_score(agreed(m.label_rule), agreed(m.label_hand))
    cm = pd.DataFrame(confusion_matrix(m.label_rule, m.label_hand, labels=list("ACD")),
                      index=[f"rule {x}" for x in "ACD"], columns=[f"hand {x}" for x in "ACD"])
    by_cond = m.groupby("condition").apply(lambda g: (g.label_rule == g.label_hand).mean()).round(3).to_dict()
    out = dict(n=len(m), kappa_ACD=round(k3, 3), kappa_agreed=round(k2, 3),
               raw_agreement=round((m.label_rule == m.label_hand).mean(), 3),
               confusion=cm.to_dict(), agreement_by_condition=by_cond)
    (ROOT / "results" / "kappa.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    m[["unit_id", "condition", "label_hand", "level"]].to_csv(ROOT / "results" / "hand_labels.csv", index=False)
    print(json.dumps(out, indent=2))
    print(cm)


if __name__ == "__main__":
    {"sheet": make_sheet, "kappa": kappa}[sys.argv[1]]()
