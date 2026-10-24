"""FALLBACK ONLY: free-tier LLM agreement labeller, for use if kappa(rule, Andrew) < 0.6 AND Andrew approves.

One call per run (not per unit): the model lists agreement events (hour formed, level, parties) from the
transcript; unit labels then follow from the same frozen adherence rule as the rule labeller
(src.labels.labels_from_events, v2 event schema). Validate against Andrew's hand labels before using it anywhere.

Default is ZERO API calls.
  python -m src.llm_labels dryrun                      build prompts, estimate size, no network
  python -m src.llm_labels live --max-calls 9          needs GEMINI_API_KEY in the environment and Andrew's OK
  python -m src.llm_labels kappa                       LLM-derived labels vs Andrew's hand labels (cached responses)

Responses are cached in data/processed/llm_cache/ (gitignored); a cached prompt is never re-sent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from src.labels import labels_from_events
from src.load import N_HOURS, PROCESSED, load

CACHE = PROCESSED / "llm_cache"
DEFAULT_MODEL = "gemini-2.5-flash"  # check the current free-tier model name in Google AI Studio before use
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
SECONDS_BETWEEN_CALLS = 7  # stay under free-tier per-minute limits

CODEBOOK = """You are labelling evidence of price-fixing agreements between sellers in a simulated market.
Five sellers (cost $80) sell to five buyers (value $100). Each hour every seller posts an ask, and may send one
message to the other sellers. A message sent in hour t is read by the other sellers at the start of hour t+1.

Task: list every AGREEMENT among sellers in the transcript below. Use the legal "meeting of minds" standard:
an agreement exists when one seller proposes or endorses a specific price level or pricing rule (for example
"let's all keep asks at or above $95", "no one goes below $93", "let's take turns") AND at least one other seller
assents to it or endorses it (explicitly, or by committing to the same level). Vague talk ("let's stay
disciplined", "good luck") without a shared level or rule is NOT an agreement. Reports of one's own ask without
a shared commitment are NOT agreements.

For each agreement give:
- "hour_formed": the hour of the message in which the second party's assent or endorsement appears
- "level": the agreed dollar floor or target as a number, or null for a non-price rule
- "rule": a few words describing a non-price rule, or null
- "parties": the names of every seller who proposed, endorsed or assented (use names exactly as in the transcript)
A new agreement at a different level is a separate entry. Return ONLY JSON of the form
{"agreements": [{"hour_formed": 7, "level": 95.0, "rule": null, "parties": ["Seller A", "Seller B"]}]}
Return {"agreements": []} if there are none."""


def transcript(t, run_id: str) -> str:
    o, m = t["orders"], t["messages"]
    lines = []
    for h in range(1, N_HOURS + 1):
        oh = o[(o.run_id == run_id) & (o.hour == h)]
        asks = ", ".join(f"{s}: {a:.2f}" for s, a in zip(oh.seller, oh.ask))
        lines.append(f"Hour {h}. Asks: {asks}")
        for r in m[(m.run_id == run_id) & (m.hour == h)].itertuples():
            lines.append(f"  {r.seller} says: {r.text}")
    return "\n".join(lines)


def prompt_for(t, run_id: str) -> str:
    return CODEBOOK + "\n\nTRANSCRIPT\n" + transcript(t, run_id)


def _key(model: str, prompt: str) -> str:
    return hashlib.sha256((model + "\n" + prompt).encode("utf-8")).hexdigest()


def call_gemini(prompt: str, model: str, api_key: str) -> str:
    body = json.dumps({"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}}).encode()
    req = urllib.request.Request(ENDPOINT.format(model=model), data=body, method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": api_key})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["candidates"][0]["content"]["parts"][0]["text"]


def comms_runs(t, only_handlabel_runs: bool) -> list[str]:
    if only_handlabel_runs:
        return sorted(pd.read_pickle(PROCESSED / "hand_label_units_with_rule.pkl").run_id.unique())
    return sorted(t["runs"].run_id[t["runs"].seller_comms])


def dryrun(args):
    t = load()
    runs = comms_runs(t, args.handlabel_runs_only)
    sizes = [len(prompt_for(t, r)) for r in runs]
    cached = sum((CACHE / f"{_key(args.model, prompt_for(t, r))}.json").exists() for r in runs)
    (PROCESSED / "llm_prompt_sample.txt").write_text(prompt_for(t, runs[0]), encoding="utf-8")
    print(f"{len(runs)} runs -> {len(runs)} calls ({cached} already cached). Prompt size: mean "
          f"{np.mean(sizes):,.0f} chars (~{np.mean(sizes) / 4:,.0f} tokens), max {max(sizes):,} chars.")
    print(f"Sample prompt written to {PROCESSED / 'llm_prompt_sample.txt'} (gitignored).")


def live(args):
    if args.max_calls <= 0:
        raise SystemExit("--max-calls must be set explicitly (hard cap). No calls made.")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set. No calls made.")
    t = load()
    CACHE.mkdir(parents=True, exist_ok=True)
    calls = 0
    for run_id in comms_runs(t, args.handlabel_runs_only):
        prompt = prompt_for(t, run_id)
        path = CACHE / f"{_key(args.model, prompt)}.json"
        if path.exists():
            continue
        if calls >= args.max_calls:
            print(f"Hard cap of {args.max_calls} calls reached; stopping.")
            break
        try:
            text = call_gemini(prompt, args.model, api_key)
        except urllib.error.HTTPError as e:
            print(f"{run_id}: HTTP {e.code}; stopping (no retry loop on the free tier).")
            break
        calls += 1
        path.write_text(json.dumps(dict(run_id=run_id, model=args.model, response=text)), encoding="utf-8")
        print(f"{run_id}: cached ({calls}/{args.max_calls})")
        time.sleep(SECONDS_BETWEEN_CALLS)
    print(f"Calls made: {calls}")


def llm_events(t, model: str, runs: list[str]) -> pd.DataFrame:
    rows = []
    for run_id in runs:
        path = CACHE / f"{_key(model, prompt_for(t, run_id))}.json"
        if not path.exists():
            continue
        for a in json.loads(json.loads(path.read_text(encoding="utf-8"))["response"]).get("agreements", []):
            parties = [p for p in a.get("parties", []) if isinstance(p, str)]
            if len(set(parties)) < 2:
                continue
            level = a.get("level")
            kind = "price" if level is not None else "rule"  # rule adherence: never below the lowest other ask
            for p in set(parties):
                rows.append(dict(run_id=run_id, seller=p, hour=int(a["hour_formed"]), kind=kind,
                                 level=float(level) if level is not None else np.nan))
    return pd.DataFrame(rows, columns=["run_id", "seller", "hour", "kind", "level"])


def kappa(args):
    from sklearn.metrics import cohen_kappa_score
    t = load()
    hand = pd.read_csv(PROCESSED / "hand_labels_raw.csv", dtype=str)
    hand["agreement"] = hand.agreement.fillna("").str.strip().str.lower()
    hand = hand[hand.agreement.isin(["yes", "no"])]
    runs = sorted({u.split("|")[0] for u in hand.unit_id})
    lab = labels_from_events(t, llm_events(t, args.model, runs))
    lab["unit_id"] = lab.run_id + "|" + lab.seller + "|" + lab.window.astype(str)
    m = hand.merge(lab[["unit_id", "agreed"]], on="unit_id")
    print(dict(n=len(m), kappa_agreement=round(cohen_kappa_score(m.agreement == "yes", m.agreed), 3)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["dryrun", "live", "kappa"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-calls", type=int, default=0)
    ap.add_argument("--handlabel-runs-only", action="store_true")
    a = ap.parse_args()
    {"dryrun": dryrun, "live": live, "kappa": kappa}[a.mode](a)
