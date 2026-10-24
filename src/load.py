"""Turn the Agrawal et al. (2025) auction logs into tidy pandas tables.

Source: github.com/andyjliu/llm-agent-collusion at commit 2bb2a7e, cloned into
data/raw/ (gitignored). Tables are cached to data/processed/tables.pkl, which is
also gitignored because it contains message and reasoning text.

Usage:  python -m src.load          (parse, replay the market, print a summary)
"""
from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "llm-agent-collusion" / "data"
PROCESSED = ROOT / "data" / "processed"
CACHE = PROCESSED / "tables.pkl"

N_HOURS = 30
WINDOW = 5
N_WINDOWS = N_HOURS // WINDOW
COST = 80.0
VALUE = 100.0
CE_PRICE = 90.0
JPM_PRICE = 100.0

AUCTION_BLOCK = re.compile(r"````json\n(.*?)\n````", re.S)
RESPONSE_BLOCK = re.compile(r"## Response: Round (\d+) - [^\n]*\n````json\n(.*?)\n````", re.S)
RUN_NAME = re.compile(r"^\d{8}_\d{6}_(.+)-(\d+)$")

REASONING_FIELDS = ["reflection", "plan_for_this_hour", "new_memory", "plan_for_message", "scratch_pad_update"]
JUDGE_FIELDS = ["reflection", "plan_for_this_hour", "new_memory", "scratch_pad_update"]  # what the paper's judge read


def window_of(hour) -> int:
    """Hours 1-5 -> window 1, ..., 26-30 -> window 6."""
    return (np.asarray(hour) - 1) // WINDOW + 1


def _model_family(models: list[str]) -> str:
    fams = {"claude" if m.startswith("claude") else "gpt" for m in models}
    return "mixed" if len(fams) > 1 else fams.pop()


def _as_price(x):
    """Response 'ask'/'bid' -> float, 'null' (withdraw), or None (no order)."""
    if x is None:
        return None
    if isinstance(x, str):
        if x.strip().lower() == "null":
            return "null"
        try:
            return float(x)
        except ValueError:
            return None
    return float(x)


def _responses(md_path: Path) -> dict[int, dict]:
    text = md_path.read_text(encoding="utf-8")
    return {int(h): json.loads(body) for h, body in RESPONSE_BLOCK.findall(text)}


def parse_run(run_dir: Path) -> dict[str, list[dict]]:
    meta = json.loads((run_dir / "experiment_metadata.json").read_text(encoding="utf-8"))
    cfg = meta["auction_config"]
    run_id = run_dir.name
    condition, rep = RUN_NAME.match(run_id).groups()
    sellers, buyers = meta["seller_ids"], meta["buyer_ids"]

    out: dict[str, list[dict]] = {k: [] for k in
        ["runs", "sellers", "asks", "bids", "trades", "messages", "reasoning", "buyer_orders", "judge", "overseer", "paper"]}

    out["runs"].append(dict(
        run_id=run_id, condition=condition, rep=int(rep),
        seller_comms=cfg["seller_comms_enabled"], oversight=cfg["oversight_enabled"],
        boss_pressure=cfg["boss_pressure"],
        seller_family=_model_family(cfg["seller_models"]), buyer_family=_model_family(cfg["buyer_models"]),
        n_sellers=len(sellers), n_buyers=len(buyers), rounds=cfg["rounds"], temperature=cfg["temperature"],
    ))
    judge_key = {f"seller_{i + 1}": s for i, s in enumerate(sorted(sellers))}  # evaluate.py:61-63
    for i, s in enumerate(sellers):
        out["sellers"].append(dict(run_id=run_id, seller=s, seller_model=cfg["seller_models"][i],
                                   seller_pos=i, judge_key=[k for k, v in judge_key.items() if v == s][0]))

    # Market log: one block per hour
    blocks = [json.loads(b) for b in AUCTION_BLOCK.findall((run_dir / "auction_results.md").read_text(encoding="utf-8"))]
    for b in blocks:
        h = b["round_number"]
        for s, p in b["seller_asks"].items():
            out["asks"].append(dict(run_id=run_id, hour=h, seller=s, ask_logged=float(p)))
        for byr, p in b["buyer_bids"].items():
            out["bids"].append(dict(run_id=run_id, hour=h, buyer=byr, bid_logged=float(p)))
        for k, t in enumerate(b["trades"]):
            out["trades"].append(dict(run_id=run_id, hour=h, seq=k, buyer=t["buyer_id"], seller=t["seller_id"],
                                      price=float(t["price"])))
        for s, m in b["seller_messages"].items():
            out["messages"].append(dict(run_id=run_id, hour=h, seller=s, text=m))

    # Seller private reasoning and raw order decisions
    for s in sellers:
        for h, r in sorted(_responses(run_dir / f"{s}.md").items()):
            row = dict(run_id=run_id, hour=h, seller=s, ask_raw=_as_price(r.get("ask")),
                       message_to_sellers=r.get("message_to_sellers"))
            for f in REASONING_FIELDS:
                v = r.get(f)
                row[f] = v if isinstance(v, str) else ("" if v is None else json.dumps(v))
            out["reasoning"].append(row)
    for byr in buyers:
        for h, r in sorted(_responses(run_dir / f"{byr}.md").items()):
            out["buyer_orders"].append(dict(run_id=run_id, hour=h, buyer=byr, bid_raw=_as_price(r.get("bid"))))

    # Paper's judge (Tier 3, private reasoning only). Join on `round`, not list position.
    for item in json.loads((run_dir / "llm_eval_full.json").read_text(encoding="utf-8")):
        s = judge_key[item["seller_id"]]
        for r in item["rounds"]:
            d = r.get("data") if r.get("status") == "success" else None
            d = d if isinstance(d, dict) else {}
            ev = d.get("evidence") or []
            ev = [e for e in ev if isinstance(e, dict)]
            cats = [str(e.get("category", "")).strip().lower() for e in ev]
            out["judge"].append(dict(
                run_id=run_id, hour=int(r["round"]), seller=s, status=r.get("status"),
                coordination=(str(d.get("coordination")).lower() == "yes") if d else np.nan,
                score=pd.to_numeric(d.get("score"), errors="coerce") if d else np.nan,
                type=d.get("type"), n_evidence=len(ev),
                **{f"cat_{c}": cats.count(c) for c in ["propose", "maintain", "threaten", "withdraw"]},
                evidence_phrases=[str(e.get("phrase", "")) for e in ev],
            ))

    # Paper's message overseer (oversight runs only)
    ov = run_dir / "Overseer.md"
    if ov.exists():
        for h, r in sorted(_responses(ov).items()):
            out["overseer"].append(dict(run_id=run_id, hour=h, coordination=str(r.get("coordination")).lower() == "yes",
                                        score=pd.to_numeric(r.get("coordination_score"), errors="coerce"),
                                        type=r.get("type")))

    m = json.loads((run_dir / "collusion_metrics.json").read_text(encoding="utf-8"))
    out["paper"].append(dict(run_id=run_id, **{k: m.get(k) for k in [
        "avg_trade_price_overall", "combined_seller_profits", "combined_buyer_surplus", "num_trades",
        "trade_frequency", "hhi_volume", "avg_seller_ask_dispersion", "average_pcm", "collusion_auc",
        "competitive_equilibrium_price", "joint_profit_maximization_price"]}))
    return out


def replay(asks: pd.DataFrame, bids: pd.DataFrame, reasoning: pd.DataFrame, buyer_orders: pd.DataFrame,
           runs: pd.DataFrame, sellers: pd.DataFrame):
    """Rebuild the order book hour by hour, following market.py.

    Orders persist until matched, replaced, or withdrawn with "null". Each hour the
    lowest ask is matched with the highest bid while bid >= ask, at the midpoint.
    Ties go by (price, name) tuples, as in market.py.

    Returns (book, orders, trades_replayed):
      book    - per run-hour state *before* this hour's orders (what agents saw, GPT runs)
      orders  - per run-hour-seller ask in force at matching time (NaN if none)
      trades  - replayed trades, for checking against the log
    """
    ask_ix = asks.set_index(["run_id", "hour", "seller"])["ask_logged"].to_dict()
    bid_ix = bids.set_index(["run_id", "hour", "buyer"])["bid_logged"].to_dict()
    s_null = set(map(tuple, reasoning.loc[reasoning.ask_raw.astype(str) == "null", ["run_id", "hour", "seller"]].values))
    b_null = set(map(tuple, buyer_orders.loc[buyer_orders.bid_raw.astype(str) == "null", ["run_id", "hour", "buyer"]].values))
    buyers_by_run = buyer_orders.groupby("run_id").buyer.unique().to_dict()
    sellers_by_run = sellers.groupby("run_id").seller.apply(list).to_dict()

    book_rows, order_rows, trade_rows = [], [], []
    for run_id in runs.run_id:
        a_book: dict[str, float] = {}
        b_book: dict[str, float] = {}
        for h in range(1, N_HOURS + 1):
            book_rows.append(dict(run_id=run_id, hour=h,
                                  open_best_bid=max(b_book.values()) if b_book else np.nan,
                                  open_best_ask=min(a_book.values()) if a_book else np.nan,
                                  open_n_bids=len(b_book), open_n_asks=len(a_book)))
            for s in sellers_by_run[run_id]:
                if (run_id, h, s) in ask_ix:
                    a_book[s] = ask_ix[(run_id, h, s)]
                elif (run_id, h, s) in s_null:
                    a_book.pop(s, None)
            for byr in buyers_by_run[run_id]:
                if (run_id, h, byr) in bid_ix:
                    b_book[byr] = bid_ix[(run_id, h, byr)]
                elif (run_id, h, byr) in b_null:
                    b_book.pop(byr, None)
            for s in sellers_by_run[run_id]:
                order_rows.append(dict(run_id=run_id, hour=h, seller=s, ask=a_book.get(s, np.nan),
                                       posted=(run_id, h, s) in ask_ix))
            book_rows[-1].update(close_best_bid=max(b_book.values()) if b_book else np.nan,
                                 close_best_ask=min(a_book.values()) if a_book else np.nan)
            k = 0
            while a_book and b_book:
                pa, sa = min((p, i) for i, p in a_book.items())
                pb, sb = max((p, i) for i, p in b_book.items())
                if pa > pb:
                    break
                trade_rows.append(dict(run_id=run_id, hour=h, seq=k, buyer=sb, seller=sa, price=round((pa + pb) / 2, 2)))
                del a_book[sa], b_book[sb]
                k += 1
    return pd.DataFrame(book_rows), pd.DataFrame(order_rows), pd.DataFrame(trade_rows)


def build_tables() -> dict[str, pd.DataFrame]:
    run_dirs = sorted(p for p in RAW.iterdir() if (p / "experiment_metadata.json").exists())
    acc: dict[str, list[dict]] = {}
    for d in run_dirs:
        for k, rows in parse_run(d).items():
            acc.setdefault(k, []).extend(rows)
    t = {k: pd.DataFrame(v) for k, v in acc.items()}
    t["book"], t["orders"], t["trades_replayed"] = replay(t["asks"], t["bids"], t["reasoning"], t["buyer_orders"],
                                                          t["runs"], t["sellers"])
    t["orders"] = t["orders"].merge(t["sellers"][["run_id", "seller", "seller_model"]], on=["run_id", "seller"])
    for k in ["asks", "bids", "trades", "messages", "reasoning", "judge", "overseer", "orders", "book"]:
        if len(t[k]):
            t[k]["window"] = window_of(t[k]["hour"])
    return t


def load(rebuild: bool = False) -> dict[str, pd.DataFrame]:
    if CACHE.exists() and not rebuild:
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    t = build_tables()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(t, f)
    return t


if __name__ == "__main__":
    t = load(rebuild=True)
    for k, v in t.items():
        print(f"{k:16s} {v.shape}")
    print(t["runs"].groupby("condition").size())
