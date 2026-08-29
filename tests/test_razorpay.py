"""Pulling a merchant's own settlements, in test mode only.

The recon endpoint returns every settled line with the `settlement_id` it was
paid out under, which is this project's hard case with real money. These tests
inject the transport, so the guards, the parsing and the error messages are all
exercised without a network or anybody's credentials.
"""

from __future__ import annotations

import pytest

from allocation_agent.adapters.razorpay import (
    RazorpayError,
    ReconItem,
    fetch_recon,
    group_into_settlements,
)


def reply(*items):
    return lambda url, auth: {"entity": "collection", "count": len(items),
                              "items": list(items)}


def line(settlement="setl_A", entity="pay_1", credit=10_000, debit=0, fee=236):
    return {"entity_id": entity, "settlement_id": settlement, "type": "payment",
            "credit": credit, "debit": debit, "fee": fee, "tax": 36,
            "currency": "INR", "settled_at": 1_700_000_000}


# --------------------------------------------------------------------------- #
# the guard that matters most
# --------------------------------------------------------------------------- #

def test_a_live_key_is_refused_and_told_why():
    """A live key authorises reads against real settled money. No demo needs
    that, and refusing it is itself the safety argument."""
    with pytest.raises(RazorpayError, match="test-mode"):
        fetch_recon("rzp_live_abc123", "secret", year=2026, month=3,
                    fetch=reply(line()))


@pytest.mark.parametrize("key", ["rzp_liv_x", "acc_x", "", "RZP_TEST_X", "rzp_tes_x"])
def test_only_an_exact_test_prefix_is_accepted(key):
    with pytest.raises(RazorpayError, match="test-mode"):
        fetch_recon(key, "secret", year=2026, month=3, fetch=reply(line()))


def test_a_test_key_is_accepted():
    items = fetch_recon("rzp_test_abc", "secret", year=2026, month=3,
                        fetch=reply(line()))
    assert len(items) == 1


def test_the_secret_is_required():
    with pytest.raises(RazorpayError, match="secret"):
        fetch_recon("rzp_test_abc", "", year=2026, month=3, fetch=reply(line()))


def test_the_secret_never_appears_in_the_url():
    """Credentials belong in the Authorization header, never a query string —
    URLs are logged by proxies and end up in browser history."""
    seen = {}

    def spy(url, auth):
        seen["url"], seen["auth"] = url, auth
        return {"items": [line()]}

    fetch_recon("rzp_test_abc", "topsecret", year=2026, month=3, fetch=spy)
    assert "topsecret" not in seen["url"]
    assert "rzp_test_abc" not in seen["url"]
    assert seen["auth"].startswith("Basic ")


# --------------------------------------------------------------------------- #
# request shape
# --------------------------------------------------------------------------- #

def test_the_period_is_zero_padded_as_the_api_requires():
    seen = {}

    def spy(url, auth):
        seen["url"] = url
        return {"items": [line()]}

    fetch_recon("rzp_test_a", "s", year=2026, month=3, day=7, fetch=spy)
    assert "year=2026" in seen["url"]
    assert "month=03" in seen["url"]
    assert "day=07" in seen["url"]


def test_an_impossible_month_is_refused_before_any_request():
    called = []
    with pytest.raises(RazorpayError, match="month"):
        fetch_recon("rzp_test_a", "s", year=2026, month=13,
                    fetch=lambda u, a: called.append(u) or {"items": []})
    assert not called, "a request went out for a month that cannot exist"


def test_the_fetch_is_capped():
    many = [line(entity=f"pay_{i}") for i in range(5_000)]
    items = fetch_recon("rzp_test_a", "s", year=2026, month=3,
                        fetch=lambda u, a: {"items": many})
    assert len(items) <= 1_000


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

def test_amounts_stay_integer_paise():
    items = fetch_recon("rzp_test_a", "s", year=2026, month=3,
                        fetch=reply(line(credit=12_345)))
    assert items[0].credit_minor == 12_345
    assert isinstance(items[0].credit_minor, int)


def test_the_net_of_a_line_is_credit_minus_debit():
    items = fetch_recon("rzp_test_a", "s", year=2026, month=3,
                        fetch=reply(line(credit=10_000, debit=2_500)))
    assert items[0].net_minor == 7_500


def test_a_line_not_yet_settled_is_skipped():
    """No settlement id means it has not been paid out; there is nothing to
    reconcile it against."""
    unsettled = {**line(), "settlement_id": None}
    items = fetch_recon("rzp_test_a", "s", year=2026, month=3,
                        fetch=reply(unsettled, line()))
    assert len(items) == 1


def test_a_reply_without_items_is_an_error_not_an_empty_result():
    with pytest.raises(RazorpayError, match="items"):
        fetch_recon("rzp_test_a", "s", year=2026, month=3,
                    fetch=lambda u, a: {"entity": "collection"})


# --------------------------------------------------------------------------- #
# grouping — this is the part the solver consumes
# --------------------------------------------------------------------------- #

def test_lines_sharing_a_settlement_become_one_bank_credit():
    items = [
        ReconItem("pay_1", "setl_A", "payment", 10_000, 0, 236, "INR", 1),
        ReconItem("pay_2", "setl_A", "payment", 25_000, 0, 590, "INR", 1),
        ReconItem("pay_3", "setl_B", "payment", 7_000, 0, 165, "INR", 1),
    ]
    settlements = group_into_settlements(items)
    assert len(settlements) == 2
    a = next(s for s in settlements if s["settlement_id"] == "setl_A")
    assert a["amount_minor"] == 35_000
    assert sorted(a["truth"]) == ["pay_1", "pay_2"]


def test_a_refund_reduces_its_settlement_rather_than_adding_to_it():
    items = [
        ReconItem("pay_1", "setl_A", "payment", 10_000, 0, 236, "INR", 1),
        ReconItem("rfnd_1", "setl_A", "refund", 0, 2_500, 0, "INR", 1),
    ]
    assert group_into_settlements(items)[0]["amount_minor"] == 7_500


def test_a_settlement_that_nets_to_nothing_is_not_offered():
    """A payout of zero is not a credit anybody has to reconcile."""
    items = [
        ReconItem("pay_1", "setl_A", "payment", 5_000, 0, 0, "INR", 1),
        ReconItem("rfnd_1", "setl_A", "refund", 0, 5_000, 0, "INR", 1),
    ]
    assert group_into_settlements(items) == []


def test_settlements_come_back_largest_first():
    items = [
        ReconItem("p1", "small", "payment", 100, 0, 0, "INR", 1),
        ReconItem("p2", "big", "payment", 90_000, 0, 0, "INR", 1),
    ]
    assert [s["settlement_id"] for s in group_into_settlements(items)] == ["big", "small"]
