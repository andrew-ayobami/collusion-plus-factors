import numpy as np
import pandas as pd
import pytest

from src.load import JUDGE_FIELDS, N_HOURS, load


@pytest.fixture(scope="module")
def t():
    return load()


def test_75_runs_by_condition(t):
    runs = t["runs"]
    assert len(runs) == 75 and runs.run_id.is_unique
    counts = runs.condition.value_counts().to_dict()
    for c in ["base", "base-seller_comms", "oversight", "boss_pressure", "boss_pressure-oversight"]:
        assert counts[c] == 10
    assert sum(v for k, v in counts.items() if k.endswith("_sellers-seller_comms")) == 25
    assert (runs.n_sellers == 5).all() and (runs.n_buyers == 5).all() and (runs.rounds == 30).all()


def test_75_runs_x_30_hours_x_5_sellers(t):
    for name in ["reasoning", "judge", "orders"]:
        df = t[name]
        assert len(df) == 75 * N_HOURS * 5, name
        assert not df.duplicated(["run_id", "hour", "seller"]).any(), name
        per_run = df.groupby("run_id").agg(h=("hour", "nunique"), s=("seller", "nunique"))
        assert (per_run.h == N_HOURS).all() and (per_run.s == 5).all(), name
    assert t["book"].groupby("run_id").size().eq(N_HOURS).all()


def test_only_base_runs_lack_seller_messages(t):
    msg_runs = set(t["messages"].run_id)
    runs = t["runs"]
    assert not msg_runs & set(runs.run_id[runs.condition == "base"])
    assert set(runs.run_id[runs.seller_comms]) <= msg_runs


def test_replayed_market_matches_logged_trades(t):
    key = ["run_id", "hour", "seq", "buyer", "seller", "price"]
    logged = t["trades"][key].sort_values(key).reset_index(drop=True)
    replayed = t["trades_replayed"][key].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(logged, replayed)


def test_judge_names_map_alphabetically(t):
    """Evidence phrases the judge quoted should appear in the mapped seller's own reasoning."""
    j = t["judge"].explode("evidence_phrases").dropna(subset=["evidence_phrases"])
    j = j[j.evidence_phrases.str.len() >= 12]
    r = t["reasoning"].copy()
    r["text"] = r[JUDGE_FIELDS].agg(" ".join, axis=1).str.lower()
    text = r.set_index(["run_id", "hour", "seller"]).text.to_dict()
    sellers = t["sellers"].groupby("run_id").seller.apply(list).to_dict()
    sample = j.sample(min(2000, len(j)), random_state=0)
    own = np.mean([p.lower() in text[(ru, h, s)] for ru, h, s, p in
                   sample[["run_id", "hour", "seller", "evidence_phrases"]].itertuples(index=False)])
    other = np.mean([any(p.lower() in text[(ru, h, o)] for o in sellers[ru] if o != s) for ru, h, s, p in
                     sample[["run_id", "hour", "seller", "evidence_phrases"]].itertuples(index=False)])
    assert own > 0.8 and own > 3 * other, (own, other)


def test_judge_mostly_succeeded(t):
    assert (t["judge"].status == "success").mean() > 0.99
    assert t["judge"].score.dropna().between(1, 4).all()


def test_overseer_only_in_oversight_runs(t):
    runs = t["runs"]
    ov_runs = set(t["overseer"].run_id)
    assert ov_runs == set(runs.run_id[runs.oversight])
    assert len(ov_runs) == 20 and t["overseer"].groupby("run_id").size().eq(N_HOURS).all()
