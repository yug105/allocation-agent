"""BenchRec loader and blocking-recall measurement.

The loader is where money becomes integers and dates become day numbers. Both
conversions are lossy if done carelessly, and both are load-bearing: the whole
matching layer assumes exact integer arithmetic.
"""

import pandas as pd
import pytest

from allocation_agent.adapters.benchrec import (
    ParseError,
    load_benchrec,
    parse_day,
    parse_minor,
)
from allocation_agent.eval.blocking_recall import measure_blocking
from allocation_agent.match.blocker import BlockingConfig


# --------------------------------------------------------------------------- #
# money -> integer minor units
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw,expected", [
    ("100", 10_000),
    ("100.00", 10_000),
    ("34,233.97", 3_423_397),
    ("0.01", 1),
    ("0", 0),
    ("-5.50", -550),
    (" 1,000.5 ", 100_050),
])
def test_amount_parses_to_exact_minor_units(raw, expected):
    assert parse_minor(raw) == expected


def test_blank_amount_is_none_not_zero():
    """Zero is a real amount. Absent is not."""
    assert parse_minor("") is None
    assert parse_minor(None) is None
    assert parse_minor("   ") is None


def test_third_decimal_place_is_rejected_rather_than_rounded():
    """Silent rounding turns an exact-match problem into a fuzzy one."""
    with pytest.raises(ParseError, match="precision"):
        parse_minor("10.005")


def test_unparseable_amount_raises_rather_than_becoming_nan():
    with pytest.raises(ParseError):
        parse_minor("abc")


def test_float_input_is_refused():
    """Floats have already lost the precision we are trying to preserve."""
    with pytest.raises(ParseError, match="str"):
        parse_minor(10.5)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# dates -> day numbers
# --------------------------------------------------------------------------- #

def test_two_digit_year_parses_to_a_stable_day_number():
    assert parse_day("9/11/13") == parse_day("09/11/13")


def test_one_day_apart_differs_by_exactly_one():
    assert parse_day("9/12/13") - parse_day("9/11/13") == 1


def test_year_boundary_is_one_day():
    assert parse_day("1/1/14") - parse_day("12/31/13") == 1


def test_blank_date_is_none():
    assert parse_day("") is None
    assert parse_day(None) is None


def test_impossible_date_raises_rather_than_guessing():
    with pytest.raises(ParseError):
        parse_day("13/45/99")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def _frame() -> pd.DataFrame:
    """Two ledger rows sharing a key, plus two bank records."""
    return pd.DataFrame([
        {"matchId": "1", "A_amount": "100.00", "A_valueDate": "9/11/13",
         "A_account": "ACC1", "A_currencyCode": "USD", "A_transactionAttributes": "x",
         "generatorAllocation": "USD_9/11/13_ACC1_x", "B_amount": "", "B_valueDate": "",
         "B_account": "", "targetAllocation": "USD_9/11/13_ACC1_x"},
        {"matchId": "1", "A_amount": "50.00", "A_valueDate": "9/11/13",
         "A_account": "ACC1", "A_currencyCode": "USD", "A_transactionAttributes": "x",
         "generatorAllocation": "USD_9/11/13_ACC1_x", "B_amount": "150.00",
         "B_valueDate": "9/12/13", "B_account": "ACC1",
         "targetAllocation": "USD_9/11/13_ACC1_x"},
        {"matchId": "2", "A_amount": "", "A_valueDate": "", "A_account": "",
         "A_currencyCode": "", "A_transactionAttributes": "",
         "generatorAllocation": "", "B_amount": "77.00", "B_valueDate": "9/13/13",
         "B_account": "ACC2", "targetAllocation": "MULT"},
    ])


def test_ledger_rows_and_bank_records_are_separated():
    ds = load_benchrec(_frame())
    assert len(ds.key_rows) == 2
    assert len(ds.records) == 2


def test_labels_align_with_records():
    ds = load_benchrec(_frame())
    assert len(ds.labels) == len(ds.records)


def test_multi_key_records_are_flagged_not_dropped():
    ds = load_benchrec(_frame())
    assert ds.is_mult.sum() == 1
    assert len(ds.records) == 2, "MULT records are part of the batch, not excluded"


def test_a_key_spanning_two_rows_produces_two_index_entries():
    ds = load_benchrec(_frame())
    assert {r.key for r in ds.key_rows} == {"USD_9/11/13_ACC1_x"}
    assert sorted(r.amount_minor for r in ds.key_rows) == [5_000, 10_000]


def test_outcome_columns_are_not_carried_into_the_dataset():
    ds = load_benchrec(_frame())
    assert not hasattr(ds, "match_rule")
    assert not hasattr(ds, "matched_by")


# --------------------------------------------------------------------------- #
# recall measurement
# --------------------------------------------------------------------------- #

def test_recall_is_one_when_every_answer_survives_blocking():
    ds = load_benchrec(_frame())
    r = measure_blocking(ds, BlockingConfig(date_slack_days=7))
    assert r.recall == pytest.approx(1.0)


def test_recall_falls_when_blocking_is_too_tight():
    ds = load_benchrec(_frame())
    tight = measure_blocking(ds, BlockingConfig(use_date=False))
    assert tight.recall < 1.0, "150.00 matches no single ledger row exactly"


def test_report_carries_candidate_size_statistics():
    ds = load_benchrec(_frame())
    r = measure_blocking(ds, BlockingConfig())
    assert r.n_evaluated == 1, "only single-key records are scoreable for recall"
    assert r.mean_candidates >= 0
    assert r.p95_candidates >= r.median_candidates
