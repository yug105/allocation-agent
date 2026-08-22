"""Blocking: cut the candidate space without losing the answer.

Recall is the only metric that matters here. A key dropped at this stage can
never be recovered downstream, so every accuracy number afterwards is capped by
it. Candidate-set size is the cost we pay for that recall.
"""

import pytest

from allocation_agent.match.blocker import BlockingConfig, block
from allocation_agent.stores.keys import KeyIndex, KeyRow
from allocation_agent.types import BankRecord


def key(k: str, account: str = "ACC1", minor: int = 10_000, day: int = 100) -> KeyRow:
    return KeyRow(key=k, account=account, amount_minor=minor, day=day)


def rec(account: str = "ACC1", minor: int = 10_000, day: int = 100) -> BankRecord:
    return BankRecord(record_id="b1", account=account, amount_minor=minor, day=day)


# --------------------------------------------------------------------------- #
# amount predicate
# --------------------------------------------------------------------------- #

def test_exact_amount_and_account_is_a_candidate():
    idx = KeyIndex([key("K1")])
    assert block(rec(), idx, BlockingConfig(date_slack_days=0)) == {"K1"}


def test_same_amount_different_account_is_not_a_candidate():
    idx = KeyIndex([key("K1", account="OTHER")])
    assert block(rec(), idx, BlockingConfig(date_slack_days=0)) == set()


def test_amount_off_by_one_minor_unit_does_not_match_the_amount_predicate():
    """Amounts are integers. Near-misses are the date predicate's job, not this one."""
    idx = KeyIndex([key("K1", minor=10_001, day=999)])
    assert block(rec(), idx, BlockingConfig(date_slack_days=0)) == set()


def test_several_keys_sharing_an_amount_all_survive():
    idx = KeyIndex([key("K1"), key("K2"), key("K3")])
    assert block(rec(), idx, BlockingConfig(date_slack_days=0)) == {"K1", "K2", "K3"}


# --------------------------------------------------------------------------- #
# date predicate
# --------------------------------------------------------------------------- #

def test_date_predicate_catches_a_key_the_amount_predicate_missed():
    """The 9% whose amount does not match any single key exactly."""
    idx = KeyIndex([key("K1", minor=99_999, day=103)])
    assert block(rec(day=100), idx, BlockingConfig(date_slack_days=7)) == {"K1"}


def test_date_slack_is_inclusive_at_both_edges():
    idx = KeyIndex([key("EARLY", minor=1, day=97), key("LATE", minor=2, day=103)])
    got = block(rec(day=100), idx, BlockingConfig(date_slack_days=3))
    assert got == {"EARLY", "LATE"}


def test_key_just_outside_the_window_is_excluded():
    idx = KeyIndex([key("K1", minor=1, day=108)])
    assert block(rec(day=100), idx, BlockingConfig(date_slack_days=7)) == set()


def test_zero_slack_means_same_day_only():
    idx = KeyIndex([key("SAME", minor=1, day=100), key("NEXT", minor=2, day=101)])
    assert block(rec(day=100), idx, BlockingConfig(date_slack_days=0)) == {"SAME"}


# --------------------------------------------------------------------------- #
# union, not cascade
# --------------------------------------------------------------------------- #

def test_predicates_union_rather_than_short_circuit():
    """A hit on amount must not stop the date predicate from contributing."""
    idx = KeyIndex([key("BY_AMOUNT", minor=10_000, day=900),
                    key("BY_DATE", minor=55_555, day=100)])
    got = block(rec(minor=10_000, day=100), idx, BlockingConfig(date_slack_days=1))
    assert got == {"BY_AMOUNT", "BY_DATE"}


def test_a_key_matching_both_predicates_appears_once():
    idx = KeyIndex([key("BOTH", minor=10_000, day=100)])
    assert block(rec(), idx, BlockingConfig(date_slack_days=3)) == {"BOTH"}


def test_disabling_a_predicate_shrinks_the_candidate_set():
    idx = KeyIndex([key("BY_DATE", minor=77_777, day=100)])
    assert block(rec(), idx, BlockingConfig(use_date=False)) == set()
    assert block(rec(), idx, BlockingConfig(use_date=True, date_slack_days=0)) == {"BY_DATE"}


# --------------------------------------------------------------------------- #
# degenerate input must not crash the batch
# --------------------------------------------------------------------------- #

def test_empty_index_returns_empty_set():
    assert block(rec(), KeyIndex([]), BlockingConfig()) == set()


def test_record_with_no_account_yields_nothing_rather_than_everything():
    """Missing must never widen the search to the whole index."""
    idx = KeyIndex([key("K1")])
    assert block(BankRecord("b1", account=None, amount_minor=10_000, day=100),
                 idx, BlockingConfig()) == set()


def test_record_with_no_date_still_uses_the_amount_predicate():
    idx = KeyIndex([key("K1")])
    got = block(BankRecord("b1", account="ACC1", amount_minor=10_000, day=None),
                idx, BlockingConfig(date_slack_days=7))
    assert got == {"K1"}


def test_negative_date_slack_is_rejected_at_construction():
    with pytest.raises(ValueError, match="date_slack_days"):
        BlockingConfig(date_slack_days=-1)


def test_amount_must_be_an_integer_minor_unit():
    """Floats invite rounding drift into a key that must be exact."""
    with pytest.raises(TypeError, match="amount_minor"):
        KeyRow(key="K", account="A", amount_minor=100.5, day=1)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# index construction
# --------------------------------------------------------------------------- #

def test_index_reports_its_size():
    idx = KeyIndex([key("K1"), key("K2"), key("K1")])
    assert idx.n_keys == 2


def test_one_key_spanning_several_rows_is_indexed_under_each():
    """A key aggregating many ledger rows must be reachable from any of them."""
    idx = KeyIndex([key("K1", minor=100, day=10), key("K1", minor=200, day=20)])
    assert block(rec(minor=100, day=999), idx, BlockingConfig(date_slack_days=0)) == {"K1"}
    assert block(rec(minor=200, day=999), idx, BlockingConfig(date_slack_days=0)) == {"K1"}
