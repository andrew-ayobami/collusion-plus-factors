import pandas as pd
import pytest

from src.features import PRICE_ONLY, T1, T2, T3
from src.load import PROCESSED


@pytest.fixture(scope="module")
def u():
    return pd.read_pickle(PROCESSED / "units.pkl")


def test_one_row_per_unit(u):
    assert len(u) == 75 * 5 * 6
    assert not u.duplicated(["run_id", "seller", "window"]).any()


def test_labels_follow_condition(u):
    assert set(u.label[u.condition == "base"]) == {"B"}
    assert "B" not in set(u.label[u.condition != "base"])
    assert set(u.label) <= set("ABCD")


def test_tier2_zero_without_messages(u):
    assert (u.loc[u.condition == "base", T2] == 0).all().all()


def test_only_holdout_features_can_be_missing(u):
    missing = u[PRICE_ONLY + T1 + T2 + T3].isna().any()
    assert set(missing[missing].index) <= {"holdout_share", "holdout_gap"}


def test_halves_match_split(u):
    assert u.groupby("run_id").half.nunique().eq(1).all()
    assert u.drop_duplicates("run_id").half.value_counts().to_dict() == {"train": 38, "test": 37}
