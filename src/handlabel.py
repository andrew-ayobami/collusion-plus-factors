"""Hand-labelling sheets for Andrew (codebook fields), and kappa against the rule labeller v2.

  python -m src.handlabel sheet   -> data/processed/hand_label_sheet.html and hand_label_recheck.html (gitignored:
                                     they contain message text); results/hand_label_recheck_units.csv (ids only)
  python -m src.handlabel kappa   -> reads data/processed/hand_labels_raw.csv (and hand_labels_recheck_raw.csv if
                                     present), writes results/hand_labels.csv and results/kappa.json

The 72 units were sampled once (2026-10-24 03:23, commit d6f60bf, using rule labeller v1 for stratification) and are
fixed in results/hand_label_units.csv. `sample_units` is kept to document how; it is not rerun.
Fields follow notes/codebook.md v2: agreement (yes/no/unclear), level, vague, invitation, notes, original label.
The sheets show messages only: codebook Rule 4 says prices never decide a label.
"""
from __future__ import annotations

import html
import json
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.labels import ADHERE_TOL, _ask_lookups, classify_messages, unit_labels
from src.load import PROCESSED, ROOT, WINDOW, load

SEED = 20261024
UNITS_PER_RUN = 8
SHEET = PROCESSED / "hand_label_sheet.html"
RAW_LABELS = PROCESSED / "hand_labels_raw.csv"
UNITS_CSV = ROOT / "results" / "hand_label_units.csv"
RECHECK_SHEET = PROCESSED / "hand_label_recheck.html"
RECHECK_RAW = PROCESSED / "hand_labels_recheck_raw.csv"
RECHECK_UNITS = ROOT / "results" / "hand_label_recheck_units.csv"
N_RECHECK = 10
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
const KEY='__KEY__';
let store={};try{store=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(store))}catch(e){};count()}
function count(){const n=document.querySelectorAll('.unit').length;let d=0;
 document.querySelectorAll('.unit').forEach(u=>{const s=store[u.dataset.id]||{};const ok=s.agreement&&s.vague&&s.invitation;u.classList.toggle('done',!!ok);if(ok)d++});
 document.getElementById('progress').textContent=d+' of '+n+' labelled (agreement, vague and invitation set)'}
document.querySelectorAll('.unit').forEach(u=>{const id=u.dataset.id;const s=store[id]||{};
 u.querySelectorAll('input[type=radio]').forEach(r=>{if(s[r.dataset.f]===r.value)r.checked=true;
   r.addEventListener('change',()=>{store[id]=Object.assign(store[id]||{},{[r.dataset.f]:r.value});save()})});
 u.querySelectorAll('input[type=text]').forEach(i=>{i.value=s[i.dataset.f]||'';
   i.addEventListener('input',()=>{store[id]=Object.assign(store[id]||{},{[i.dataset.f]:i.value});save()})});});
function q(v){return '"'+String(v==null?'':v).replace(/"/g,'""')+'"'}
document.getElementById('dl').onclick=()=>{const F=['agreement','level','vague','invitation','notes','original_label'];
 const rows=[['unit_id'].concat(F)];
 document.querySelectorAll('.unit').forEach(u=>{const s=store[u.dataset.id]||{};rows.push([u.dataset.id].concat(F.map(f=>s[f]||'')))});
 const blob=new Blob([rows.map(r=>r.map(q).join(',')).join('\\n')],{type:'text/csv'});
 const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='__CSV__';a.click()};
count();
"""

INSTRUCTIONS = """
<div class="box">
<p><b>Apply <code>notes/codebook.md</code> (v2).</b> One unit = one seller, one 5-hour window. Read every seller's
messages from hour 1 to the <b>end of that window</b>; ignore later hours. Messages sent in hour <i>t</i> are read by
the other sellers at the start of hour <i>t</i>+1, and each seller sees only the previous hour's messages.</p>
<ul>
<li><b>agreement</b>: yes if the seller was a party to an agreement (Rules 1, 2, 5) at any hour in the window;
<b>unclear</b> when unsure, with a note.</li>
<li><b>level</b>: the agreed price (e.g. 95), "no number" for a rule like "don't undercut", "ceiling X" for a ceiling,
blank if no agreement. If the level changed, give the latest one and say so in notes. Write "conditional" in notes for
a conditional plan.</li>
<li><b>vague</b>: yes if this seller sent VAGUE talk (Rule 3) in this window.</li>
<li><b>invitation</b>: yes if this seller made an invitation (Rule 1) in this window, accepted or not (Rule 6).</li>
<li><b>notes</b>: one line for every close call, and the hour of any withdrawal. Keep it short and do not paste
message text (the source data has no licence).</li>
<li><b>original label</b>: only if you change a label after discussing it: your first call and the reason.</li>
</ul>
<p>Asks and trades are hidden on purpose (Rule 4: prices never decide a label). Conditions and the rule labeller's
answers are hidden too. Labels save automatically in this browser. When done, press <b>Download CSV</b> and save the
file in <code>data/processed/</code> as <code>hand_labels_raw.csv</code>.</p>
</div>
"""

RECHECK_INTRO = """
<div class="box">
<p><b>Consistency recheck (codebook, How to label, step 7).</b> Do this a few hours after finishing the main sheet.
These 10 units were drawn at random from the 72 you labelled; run numbers are reshuffled. Label them again from
scratch with <code>notes/codebook.md</code> v2, <b>without looking at your first answers</b>. Then press
<b>Download CSV</b> and save it in <code>data/processed/</code> as <code>hand_labels_recheck_raw.csv</code>.</p>
</div>
"""


def _radio(uid: str, field: str, options) -> str:
    name = html.escape(f"{uid}::{field}")
    return "".join(f"<label><input type='radio' name='{name}' data-f='{field}' value='{v}'> {v}</label>" for v in options)


def _write_sheet(units: pd.DataFrame, path, key: str, csv_name: str, title: str, intro: str, run_word: str,
                 with_original: bool, msgs: pd.DataFrame):
    js = JS.replace("__KEY__", key).replace("__CSV__", csv_name)
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>{title}</title><style>{CSS}</style></head><body>",
             f"<h1>{title}</h1>",
             "<div class='sticky'><button id='dl'>Download CSV</button><span id='progress'></span></div>",
             intro]
    for k, (run_id, g) in enumerate(units.groupby("run_id", sort=False), start=1):
        windows = set(g.window)
        parts.append(f"<h2>{run_word} {k}</h2>")
        for r in g.itertuples():
            h0, h1 = WINDOW * (r.window - 1) + 1, WINDOW * r.window
            uid = html.escape(r.unit_id)
            parts.append(
                f"<div class='unit' data-id='{uid}'><div><b>{html.escape(r.seller)}</b>, window {r.window} "
                f"(hours {h0}–{h1})</div>"
                f"<div class='field'><b>agreement</b>{_radio(r.unit_id, 'agreement', ['yes', 'no', 'unclear'])}</div>"
                f"<div class='field'><b>level</b><input type='text' data-f='level' placeholder='e.g. 95, no number, ceiling 97'></div>"
                f"<div class='field'><b>vague</b>{_radio(r.unit_id, 'vague', ['yes', 'no'])}</div>"
                f"<div class='field'><b>invitation</b>{_radio(r.unit_id, 'invitation', ['yes', 'no'])}</div>"
                f"<div class='field'><b>notes</b><input type='text' data-f='notes' placeholder='close calls, withdrawal hour, conditional (no quotes)'></div>"
                + ("<div class='field'><b>original label</b><input type='text' data-f='original_label' "
                   "placeholder='only if changed after discussion'></div>" if with_original else "")
                + "</div>")
        parts.append("<details open><summary>Messages</summary>")
        for h in range(1, 31):
            w = (h - 1) // WINDOW + 1
            parts.append(f"<div class='hour{' inwin' if w in windows else ''}'><div class='meta'><b>Hour {h}</b> (window {w})</div>")
            for m in msgs[(msgs.run_id == run_id) & (msgs.hour == h)].itertuples():
                parts.append(f"<div class='msg'><b>{html.escape(m.seller.split()[-1])}:</b> {html.escape(m.text)}</div>")
            parts.append("</div>")
        parts.append("</details>")
    parts.append(f"<script>{js}</script></body></html>")
    path.write_text("".join(parts), encoding="utf-8")
    print(f"{len(units)} units from {units.run_id.nunique()} runs -> {path}")


def make_sheet():
    msgs = load()["messages"]
    units = pd.read_csv(UNITS_CSV)
    _write_sheet(units, SHEET, "handlabels_v2", "hand_labels_raw.csv", "Agreement hand labels", INSTRUCTIONS,
                 "Run", True, msgs)
    if not RECHECK_UNITS.exists():  # drawn once, then fixed
        rng = np.random.default_rng(SEED + 1)
        pick = units.iloc[np.sort(rng.choice(len(units), N_RECHECK, replace=False))]
        order = {r: i for i, r in enumerate(rng.permutation(pick.run_id.unique()))}
        pick = pick.assign(_o=pick.run_id.map(order)).sort_values(["_o", "window"]).drop(columns="_o")
        pick[["unit_id", "run_id", "seller", "window", "condition"]].to_csv(RECHECK_UNITS, index=False)
    _write_sheet(pd.read_csv(RECHECK_UNITS), RECHECK_SHEET, "handlabels_recheck_v1", "hand_labels_recheck_raw.csv",
                 "Recheck: hand labels", RECHECK_INTRO, "Recheck run", False, msgs)


def _parse_level(x) -> tuple[str | None, float]:
    s = str(x).strip().lower() if pd.notna(x) else ""
    if "ceiling" in s:
        return None, np.nan  # adherence is only defined for floors and exact prices
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


def _rung_hand(r):
    if r.agreement == "yes":
        return "agreement"
    if r.agreement != "no":
        return np.nan
    return "D2" if r.invitation == "yes" else "D1" if r.vague == "yes" else "D0"


def _k(a, b) -> float:
    a, b = list(a), list(b)
    return round(float(cohen_kappa_score(a, b)), 3) if len(set(a) | set(b)) > 1 else float("nan")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["agreement", "vague", "invitation"]:
        df[c] = (df[c] if c in df else pd.Series("", index=df.index)).fillna("").str.strip().str.lower()
    return df


def kappa():
    t = load()
    raw = _clean(pd.read_csv(RAW_LABELS, dtype=str))
    units = pd.read_csv(UNITS_CSV)
    cm = classify_messages(t["messages"])
    rule = unit_labels(t, cm=cm)
    rule_ic = unit_labels(t, reading="interstate", cm=cm)
    for df in (rule, rule_ic):
        df["unit_id"] = df.run_id + "|" + df.seller + "|" + df.window.astype(str)
    m = units.merge(raw[raw.agreement.isin(["yes", "no", "unclear"])], on="unit_id").merge(
        rule[["unit_id", "label", "agreed", "vague", "invited", "ladder"]].rename(
            columns={"agreed": "agreed_rule", "vague": "vague_rule", "invited": "invited_rule", "ladder": "ladder_rule"}),
        on="unit_id")
    m = m.merge(rule_ic[["unit_id", "agreed"]].rename(columns={"agreed": "agreed_ic"}), on="unit_id")
    if "original_label" not in m:  # column absent from the CSV = no label changed after discussion
        m["original_label"] = ""
    yn = m[m.agreement.isin(["yes", "no"])]
    agreed_hand = yn.agreement == "yes"
    acd = yn.assign(acd_hand=yn.unit_id.map(_hand_acd(t, yn))).dropna(subset=["acd_hand"])
    lad = m.assign(rung_hand=[_rung_hand(r) for r in m.itertuples()],
                   rung_rule=m.ladder_rule.replace({"A": "agreement", "C": "agreement"})).dropna(subset=["rung_hand"])
    vg = m[m.vague.isin(["yes", "no"])]
    iv = m[m.invitation.isin(["yes", "no"])]
    k_agree = _k(yn.agreed_rule, agreed_hand)
    out = dict(
        n_labelled=len(m), n_units=len(units), n_unclear=int((m.agreement == "unclear").sum()),
        kappa_agreement=k_agree, n_agreement=len(yn),
        raw_agreement=round(float((yn.agreed_rule == agreed_hand).mean()), 3),
        confusion_agreement=pd.crosstab(yn.agreed_rule.map({True: "rule yes", False: "rule no"}),
                                        agreed_hand.map({True: "hand yes", False: "hand no"})).to_dict(),
        kappa_agreement_unclear_as_no=_k(m.agreed_rule, m.agreement == "yes"),
        kappa_ACD=_k(acd.label, acd.acd_hand), n_ACD=len(acd),
        kappa_vague=_k(vg.vague_rule, vg.vague == "yes"), n_vague=len(vg),
        kappa_invitation=_k(iv.invited_rule, iv.invitation == "yes"), n_invitation=len(iv),
        kappa_ladder=_k(lad.rung_rule, lad.rung_hand), n_ladder=len(lad),
        kappa_agreement_vs_interstate_rule=_k(yn.agreed_ic, agreed_hand),
        raw_agreement_by_condition=yn.assign(ok=yn.agreed_rule == agreed_hand).groupby("condition").ok.mean()
        .round(3).to_dict(),
        changed_after_discussion=int(m.original_label.fillna("").str.strip().ne("").sum()),
        threshold=KAPPA_MIN, passes=bool(k_agree >= KAPPA_MIN),
    )
    if RECHECK_RAW.exists():
        rc = _clean(pd.read_csv(RECHECK_RAW, dtype=str))
        both = m.merge(rc, on="unit_id", suffixes=("", "_2"))
        out["intra_rater"] = dict(n=len(both), agreement_same=round(float((both.agreement == both.agreement_2).mean()), 3),
                                  kappa_agreement=_k(both.agreement, both.agreement_2),
                                  kappa_vague=_k(both.vague, both.vague_2),
                                  kappa_invitation=_k(both.invitation, both.invitation_2))
    (ROOT / "results" / "kappa.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    m.assign(changed_after_discussion=m.original_label.fillna("").str.strip().ne(""))[
        ["unit_id", "condition", "agreement", "level", "vague", "invitation", "changed_after_discussion"]].to_csv(
        ROOT / "results" / "hand_labels.csv", index=False)
    print(json.dumps(out, indent=2, default=str))
    if not out["passes"]:
        print(f"\nkappa {k_agree} < {KAPPA_MIN}: STOP and ask Andrew (brief, labelling step 4).")


if __name__ == "__main__":
    {"sheet": make_sheet, "kappa": kappa}[sys.argv[1]]()
