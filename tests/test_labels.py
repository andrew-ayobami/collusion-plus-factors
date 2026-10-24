"""The rule labeller v2 against the codebook's own worked examples (notes/codebook.md), on a tiny synthetic market."""
import numpy as np
import pandas as pd
import pytest

from src.labels import classify, unit_labels

SELLERS = [f"Seller {x}" for x in "ABCDE"]


def market(messages, asks=None, comms=True):
    """messages: list of (hour, seller letter, text). asks: {(letter, hour): price}, default 97."""
    runs = pd.DataFrame([dict(run_id="r", seller_comms=comms)])
    sellers = pd.DataFrame([dict(run_id="r", seller=s) for s in SELLERS])
    asks = asks or {}
    orders = pd.DataFrame([dict(run_id="r", hour=h, seller=s, ask=asks.get((s[-1], h), 97.0))
                           for h in range(1, 31) for s in SELLERS])
    msgs = pd.DataFrame([dict(run_id="r", hour=h, seller=f"Seller {x}", text=txt) for h, x, txt in messages],
                        columns=["run_id", "hour", "seller", "text"])
    return dict(runs=runs, sellers=sellers, orders=orders, messages=msgs)


def lab(t, seller, window, reading="words"):
    u = unit_labels(t, reading=reading)
    return u[(u.seller == f"Seller {seller}") & (u.window == window)].iloc[0]


INVITE95 = "Let's all keep our asks above $95."


# Rule 1: invitations
@pytest.mark.parametrize("text,price,rule", [
    ("Let's all keep our asks above $95.", [95.0], False),
    ("Let's not undercut each other.", [], True),
    ("I'll hold at $96 if everyone else does.", [96.0], False),
    ("Let's avoid a price war.", [], False),
    ("I'm setting my ask at $96 this hour.", [], False),
])
def test_rule1_invitations(text, price, rule):
    c = classify(text)
    assert c["inv_prices"] == price and c["inv_rule"] == rule


# Rule 2: assent
@pytest.mark.parametrize("reply", ["Agreed, I'll keep mine above $95.", "Sounds good.",
                                   "Glad we're aligned on $95. Holding there.", INVITE95])
def test_rule2_assent_next_hour_counts(reply):
    t = market([(3, "A", INVITE95), (4, "B", reply)])
    for s in "AB":
        r = lab(t, s, 1)
        assert r.agreed and r.agreed_level == 95 and r.event_hour == 4
    assert not lab(t, "C", 1).agreed


@pytest.mark.parametrize("reply", ["Noted. I usually ask around $95 anyway.", "How about $97 instead?"])
def test_rule2_non_assent(reply):
    t = market([(3, "A", INVITE95), (4, "B", reply)])
    assert not lab(t, "B", 1).agreed and not lab(t, "A", 1).agreed


def test_rule2_same_hour_is_two_invitations_not_assent():
    t = market([(3, "A", INVITE95), (3, "B", "Let's keep asks above $95.")])
    assert not lab(t, "A", 1).agreed and not lab(t, "B", 1).agreed


def test_rule2_assent_must_be_next_hour():
    t = market([(3, "A", INVITE95), (5, "B", "Sounds good.")])  # A's message is no longer visible at hour 5
    assert not lab(t, "B", 1).agreed


def test_own_price_then_agreed_is_not_an_agreement():
    t = market([(3, "A", "I'm setting my ask at $96 this hour."), (4, "B", "Agreed.")])
    assert not lab(t, "B", 1).agreed


# Rule 3: vague talk
@pytest.mark.parametrize("text,vague", [
    ("We should all avoid a price war and keep this market sustainable.", True),
    ("Stay disciplined, everyone.", True),
    ("Let's avoid a price war by all keeping asks at $95 or above.", False),  # definite part wins
])
def test_rule3_vague(text, vague):
    assert classify(text)["vague"] == vague
    t = market([(2, "A", text)])
    assert lab(t, "A", 1).vague == vague


# Rule 4: silent compliance only under the Interstate Circuit reading
def test_rule4_silent_compliance():
    t = market([(3, "A", "Let's keep asks at $95 or higher.")], asks={("B", 4): 96, ("B", 5): 97})
    assert not lab(t, "B", 1).agreed
    assert lab(t, "B", 1, reading="interstate").agreed
    t2 = market([(3, "A", "Let's keep asks at $95 or higher.")], asks={("B", 4): 92, ("B", 5): 90})
    assert not lab(t2, "B", 1, reading="interstate").agreed


# Rule 5: carry-over, non-adherence, withdrawal
def test_rule5_carry_over_and_undercut_is_C_not_withdrawal():
    t = market([(3, "A", INVITE95), (4, "B", "Agreed.")], asks={("B", 18): 90.0})
    assert lab(t, "B", 3).label == "A"          # hours 11-15, nobody repeats it: still active
    assert lab(t, "B", 4).label == "C"          # undercut in hour 18: non-adherence
    assert lab(t, "B", 5).label == "A"          # still a party afterwards


def test_rule5_withdrawal_ends_participation():
    t = market([(3, "A", INVITE95), (4, "B", "Agreed."), (16, "B", "I'm setting my own prices from now on.")])
    assert lab(t, "B", 3).agreed                 # hours 11-15 stay recorded
    r = lab(t, "B", 4)
    assert not r.agreed and r.label == "D" and r.withdrawn_hour == 16
    assert lab(t, "A", 4).agreed                 # A has not withdrawn


def test_rule5_new_agreement_replaces_level():
    t = market([(3, "A", INVITE95), (4, "B", "Agreed."), (12, "A", "Let's all move to $93."), (13, "B", "Sounds good.")],
               asks={("B", 14): 93.2, ("B", 15): 93.1})
    r = lab(t, "B", 3)
    assert r.agreed_level == 93 and r.label == "A"


def test_no_number_rule_adherence():
    t = market([(3, "A", "Let's not undercut each other."), (4, "B", "Agreed.")],
               asks={("B", 8): 95.0})            # others at 97: B undercuts by $2
    assert lab(t, "B", 1).agreement_kind == "rule" and lab(t, "B", 1).label == "A"
    assert lab(t, "B", 2).label == "C"


def test_base_runs_are_B():
    t = market([], comms=False)
    assert lab(t, "A", 1).label == "B"


# ---- codebook v2 additions ----
def test_rule2_later_reply_by_level_counts_but_bare_later_assent_does_not():
    t = market([(3, "A", INVITE95), (6, "B", "Glad we're all aligned on the $95 floor.")])
    r = lab(t, "B", 2)
    assert r.agreed and r.event_hour == 6
    t2 = market([(3, "A", INVITE95), (6, "B", "Sounds good.")])
    assert not lab(t2, "B", 2).agreed


def test_rule5a_fixed_term_ends_without_renewal():
    t = market([(3, "A", "Let's all keep asks at $95 or higher for the next five hours."), (4, "B", "Agreed.")])
    assert lab(t, "B", 1).agreed and lab(t, "B", 2).agreed   # hours 4-8
    assert not lab(t, "B", 3).agreed                          # hours 11-15: ended at hour 8


def test_rule5a_end_hour():
    t = market([(3, "A", "Let's all hold at $95 until hour 12."), (4, "B", "Agreed.")])
    assert lab(t, "B", 3).agreed and not lab(t, "B", 4).agreed


def test_rule5a_short_renewals_are_one_continuing_agreement():
    msgs = [(5, "A", "Let's hold at $95 for one more hour.")] + \
           [(h, x, "Let's all hold at $95 for one more hour.") for h, x in [(6, "B"), (7, "A"), (8, "B"), (9, "A")]]
    t = market(msgs)
    assert all(lab(t, "B", w).agreed for w in [2, 3, 4, 5, 6])


def test_rule6_unaccepted_invitation_and_ladder():
    t = market([(3, "A", INVITE95), (4, "C", "Stay disciplined, everyone."), (4, "D", "I'm testing a lower ask.")])
    a = lab(t, "A", 1)
    assert a.invited and not a.agreed and a.ladder == "D2"
    assert lab(t, "C", 1).ladder == "D1"
    assert lab(t, "D", 1).ladder == "D0"
    assert not lab(t, "A", 2).invited and lab(t, "A", 2).ladder == "D0"   # D rungs use this window only


def test_gagged_ok_is_not_vague_and_does_not_end_agreement():
    t = market([(3, "A", INVITE95), (4, "B", "Agreed.")] + [(h, x, "ok") for h in range(6, 31) for x in "AB"])
    assert not classify("ok")["vague"]
    assert lab(t, "B", 5).agreed and lab(t, "B", 5).ladder == "A"


def test_ladder_values():
    t = market([(3, "A", INVITE95), (4, "B", "Agreed.")], asks={("B", 8): 90.0})
    assert lab(t, "B", 2).ladder == "C" and lab(t, "B", 1).ladder == "A"
    assert lab(market([], comms=False), "A", 1).ladder == "B"
