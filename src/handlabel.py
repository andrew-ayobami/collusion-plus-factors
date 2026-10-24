"""Hand-labelling sheet for Andrew (codebook fields), and kappa against the rule labeller v2.

  python -m src.handlabel sheet   -> data/processed/hand_label_sheet.html (gitignored: has message text)
  python -m src.handlabel kappa   -> reads data/processed/hand_labels_raw.csv (downloaded from the sheet),
                                     writes results/hand_labels.csv and results/kappa.json

The 72 units were sampled once (2026-10-24 03:23, commit d6f60bf, using rule labeller v1 for stratification) and are
fixed in results/hand_label_units.csv. `sample_units` is kept to document how; it is not rerun.
Fields follow notes/codebook.md: agreement, level, vague, notes, original label. The sheet shows messages only:
codebook Rule 4 says prices never decide a label.
"""
from __future__ import annotations

import html
import json
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from src.labels import ADHERE_TOL, _ask_lookups, classify_messages, unit_labels
from src.load import PROCESSED, ROOT, WINDOW, load

SEED = 20261024
UNITS_PER_RUN = 8
SHEET = PROCESSED / "hand_label_sheet.html"
RAW_LABELS = PROCESSED / "hand_labels_raw.csv"
UNITS_CSV = ROOT / "results" / "hand_label_units.csv"
KAPPA_MIN = 0.6


def unit_id(r) -> str:
    return f"{r.run_id}|{r.seller}|{r.window}"


def sample_units(t, lab) -> pd.DataFrame:
    """How the fixed 72 units were drawn (run once with labeller v1). One train run per comms condition (most
    rule-label variety, C first), 8 units each: up to 3 C and 3 A (4 A if no C), rest D."""
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
.field{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.field b{min-width:90px}
.unit input[type=text]{font:inherit;padding:3px 6px;border:1px solid var(--line);border-radius:4px;background:var(--bg);color:var(--ink);flex:1;min-width:180px}
.hour{border-top:1px solid var(--line);padding:6px 0}
.hour.inwin{background:var(--hl)}
.meta{color:var(--ink2);font-size:13px}
.msg{margin:2px 0 2px 12px}
.msg b{display:inline-block;min-width:110px}
details summary{cursor:pointer;color:var(--accent);margin:8px 0}
"""

JS = """
const KEY='handlabels_v2';
let store={};try{store=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(store))}catch(e){};count()}
function count(){const n=document.querySelectorAll('.unit').length;let d=0;
 document.querySelectorAll('.unit').forEach(u=>{const s=store[u.dataset.id]||{};const ok=s.agreement&&s.vague;u.classList.toggle('done',!!ok);if(ok)d++});
 document.getElementById('progress').textContent=d+' of '+n+' labelled (agreement and vague both set)'}
document.querySelectorAll('.unit').forEach(u=>{const id=u.dataset.id;const s=store[id]||{};
 u.querySelectorAll('input[type=radio]').forEach(r=>{if(s[r.dataset.f]===r.value)r.checked=true;
   r.addEventListener('change',()=>{store[id]=Object.assign(store[id]||{},{[r.dataset.f]:r.value});save()})});
 u.querySelectorAll('input[type=text]').forEach(i=>{i.value=s[i.dataset.f]||'';
   i.addEventListener('input',()=>{store[id]=Object.assign(store[id]||{},{[i.dataset.f]:i.value});save()})});});
function q(v){return '"'+String(v==null?'':v).replace(/"/g,'""')+'"'}
document.getElementById('dl').onclick=()=>{const F=['agreement','level','vague','notes','original_label'];
 const rows=[['unit_id'].concat(F)];
 document.querySelectorAll('.unit').forEach(u=>{const s=store[u.dataset.id]||{};rows.push([u.dataset.id].concat(F.map(f=>s[f]||'')))});
 const blob=new Blob([rows.map(r=>r.map(q).join(',')).join('\\n')],{type:'text/csv'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='hand_labels_raw.csv';a.click()};
count();
"""

INSTRUCTIONS = """
<div class="box">
<p><b>Apply <code>notes/codebook.md</code> (v1).</b> One unit = one seller, one 5-hour window. Read every seller's
messages from hour 1 to the <b>end of that window</b>; ignore later hours. Messages sent in hour <i>t</i> are read by
the other sellers at the start of hour <i>t</i>+1.</p>
<ul>
<li><b>agreement</b>: yes if the seller was a party to an agreement (Rules 1, 2, 5) at any hour in the window.</li>
<li><b>level</b>: the agreed price (e.g. 95), "no number" for a rule like "don't undercut", blank if no agreement.
If the level changed, give the latest one and say so in notes.</li>
<li><b>vague</b>: yes if this seller sent VAGUE talk (Rule 3) in this window.</li>
<li><b>notes</b>: one line for every close call, and the hour of any withdrawal. Keep it short and do not paste
message text (the source data has no licence).</li>
<li><b>original label</b>: only if you change a label after discussing it: your first call and the reason.</li>
</ul>
<p>Asks and trades are hidden on purpose (Rule 4: prices never decide a label). Conditions and the rule labeller's
answers are hidden too. Labels save automatically in this browser. When done, press <b>Download CSV</b> and save the
file as <code>data/processed/hand_labels_raw.csv</code>. When unsure, label no and explain in notes (Rule 6).</p>
</div>
"""


def _radio(uid: str, field: str, options) -> str:
    name = html.escape(f"{uid}::{field}")
    return "".join(f"<label><input type='radio' name='{name}' data-f='{field}' value='{v}'> {v}</label>" for v in options)


def make_sheet():
    t = load()
    units = pd.read_csv(UNITS_CSV)
    msgs = t["messages"]
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
            uid = html.escape(r.unit_id)
            parts.append(
                f"<div class='unit' data-id='{uid}'><div><b>{html.escape(r.seller)}</b>, window {r.window} "
                f"(hours {h0}–{h1})</div>"
                f"<div class='field'><b>agreement</b>{_radio(r.unit_id, 'agreement', ['yes', 'no'])}</div>"
                f"<div class='field'><b>level</b><input type='text' data-f='level' placeholder='e.g. 95, or no number'></div>"
                f"<div class='field'><b>vague</b>{_radio(r.unit_id, 'vague', ['yes', 'no'])}</div>"
                f"<div class='field'><b>notes</b><input type='text' data-f='notes' placeholder='close calls, withdrawal hour (no quotes)'></div>"
                f"<div class='field'><b>original label</b><input type='text' data-f='original_label' placeholder='only if changed after discussion'></div>"
                f"</div>")
        parts.append("<details open><summary>Messages</summary>")
        for h in range(1, 31):
            w = (h - 1) // WINDOW + 1
            parts.append(f"<div class='hour{' inwin' if w in windows else ''}'><div class='meta'><b>Hour {h}</b> (window {w})</div>")
            for m in msgs[(msgs.run_id == run_id) & (msgs.hour == h)].itertuples():
                parts.append(f"<div class='msg'><b>{html.escape(m.seller.split()[-1])}:</b> {html.escape(m.text)}</div>")
            parts.append("</div>")
        parts.append("</details>")
    parts.append(f"<script>{JS}</script></body></html>")
    SHEET.write_text("".join(parts), encoding="utf-8")
    print(f"{len(units)} units from {units.run_id.nunique()} runs -> {SHEET}")


def _parse_level(x) -> tuple[str | None, float]:
    s = str(x).strip().lower() if pd.notna(x) else ""
    m = re.search(r"\d{2,3}(?:\.\d{1,2})?", s)
    if m:
        return "price", float(m.group())
    if "no number" in s or "rule" in s:
        return "rule", np.nan
    return None, np.nan


def _hand_acd(t, hand: pd.DataFrame) -> pd.Series:
    """A/C/D from Andrew's agreement + level, with the same adherence rule as the labeller.
    His labels carry no agreement hour, so adherence is checked over the whole window (approximation)."""
    asks, others_min = _ask_lookups(t)
    out = {}
    for r in hand.itertuples():
        if r.agreement != "yes":
            out[r.unit_id] = "D"
            continue
        kind, level = _parse_level(r.level)
        if kind is None:
            out[r.unit_id] = np.nan
            continue
        run_id, seller, w = r.unit_id.split("|")
        w = int(w)
        margins = []
        for x in range(WINDOW * (w - 1) + 1, WINDOW * w + 1):
            a = asks.get((run_id, seller, x), np.nan)
            ref = level if kind == "price" else others_min.get((run_id, seller, x), np.nan)
            if not (np.isnan(a) or np.isnan(ref)):
                margins.append(a - ref)
        out[r.unit_id] = "A" if (not margins or min(margins) >= -ADHERE_TOL) else "C"
    return pd.Series(out)


def kappa():
    t = load()
    raw = pd.read_csv(RAW_LABELS, dtype=str)
    for c in ["agreement", "vague"]:
        raw[c] = raw[c].fillna("").str.strip().str.lower()
    hand = raw[raw.agreement.isin(["yes", "no"])].copy()
    units = pd.read_csv(UNITS_CSV)
    cm = classify_messages(t["messages"])
    key = ["run_id", "seller", "window"]
    rule = unit_labels(t, cm=cm)
    rule_ic = unit_labels(t, reading="interstate", cm=cm)
    for df in (rule, rule_ic):
        df["unit_id"] = df.run_id + "|" + df.seller + "|" + df.window.astype(str)
    m = units.merge(hand.rename(columns={"vague": "vague_hand"}), on="unit_id").merge(
        rule[["unit_id", "label", "agreed", "vague"]].rename(columns={"agreed": "agreed_rule", "vague": "vague_rule"}),
        on="unit_id")
    m = m.merge(rule_ic[["unit_id", "agreed"]].rename(columns={"agreed": "agreed_ic"}), on="unit_id")
    m["agreed_hand"] = m.agreement == "yes"
    m["acd_hand"] = m.unit_id.map(_hand_acd(t, m))
    k_agree = cohen_kappa_score(m.agreed_rule, m.agreed_hand)
    acd = m.dropna(subset=["acd_hand"])
    vg = m[m.vague_hand.isin(["yes", "no"])]
    out = dict(
        n_labelled=len(m), n_units=len(units),
        kappa_agreement=round(k_agree, 3),
        raw_agreement=round((m.agreed_rule == m.agreed_hand).mean(), 3),
        confusion_agreement=pd.crosstab(m.agreed_rule.map({True: "rule yes", False: "rule no"}),
                                        m.agreed_hand.map({True: "hand yes", False: "hand no"})).to_dict(),
        kappa_ACD=round(cohen_kappa_score(acd.label, acd.acd_hand), 3), n_ACD=len(acd),
        kappa_vague=round(cohen_kappa_score(vg.vague_rule, vg.vague_hand == "yes"), 3), n_vague=len(vg),
        kappa_agreement_vs_interstate_rule=round(cohen_kappa_score(m.agreed_ic, m.agreed_hand), 3),
        raw_agreement_by_condition=m.groupby("condition").apply(
            lambda g: round((g.agreed_rule == g.agreed_hand).mean(), 3)).to_dict(),
        changed_after_discussion=int(m.original_label.fillna("").str.strip().ne("").sum()),
        threshold=KAPPA_MIN, passes=bool(k_agree >= KAPPA_MIN),
    )
    (ROOT / "results" / "kappa.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    m.assign(changed_after_discussion=m.original_label.fillna("").str.strip().ne(""))[
        ["unit_id", "condition", "agreement", "level", "vague_hand", "changed_after_discussion"]].rename(
        columns={"vague_hand": "vague"}).to_csv(ROOT / "results" / "hand_labels.csv", index=False)
    print(json.dumps(out, indent=2, default=str))
    if not out["passes"]:
        print(f"\nkappa {k_agree:.3f} < {KAPPA_MIN}: STOP and ask Andrew (brief, labelling step 4).")


if __name__ == "__main__":
    {"sheet": make_sheet, "kappa": kappa}[sys.argv[1]]()
