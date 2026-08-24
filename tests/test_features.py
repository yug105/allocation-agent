"""Candidate features.

One row per (bank record, candidate key) pair. The model never sees the label,
the key string, or anything recording how a match was resolved.

Several features the design assumed are absent from this data: references,
counterparty names, and direction are all unpopulated, and currency has a single
value. What remains is amount, date, key shape, and ambiguity.
"""

import numpy as np
import pytest

from allocation_agent.match.features import FEATURE_NAMES, KeyStats, featurise
from allocation_agent.types import BankRecord


def stats(amounts=(10_000,), days=(100,)) -> KeyStats:
    return KeyStats(amounts=frozenset(amounts), days=tuple(days), n_rows=len(amounts))


def rec(minor=10_000, day=100) -> BankRecord:
    return BankRecord("b1", account="ACC1", amount_minor=minor, day=day)


def f(record, key_stats, n_candidates=1) -> dict[str, float]:
    vec = featurise(record, key_stats, n_candidates=n_candidates)
    return dict(zip(FEATURE_NAMES, vec, strict=False))


# --------------------------------------------------------------------------- #
# shape
# --------------------------------------------------------------------------- #

def test_vector_length_matches_the_declared_names():
    assert len(featurise(rec(), stats(), n_candidates=1)) == len(FEATURE_NAMES)


def test_feature_names_are_unique():
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_output_is_finite_for_degenerate_input():
    """No NaN or inf may reach the model, whatever the input."""
    v = featurise(BankRecord("b", "A", 0, None), KeyStats(frozenset(), (), 0), n_candidates=0)
    assert np.all(np.isfinite(v))


# --------------------------------------------------------------------------- #
# amount
# --------------------------------------------------------------------------- #

def test_exact_amount_match_is_flagged():
    assert f(rec(10_000), stats(amounts=(10_000, 5_000)))["amount_exact"] == 1.0


def test_no_exact_amount_match_is_flagged():
    assert f(rec(10_000), stats(amounts=(9_999,)))["amount_exact"] == 0.0


def test_delta_is_to_the_nearest_row_not_the_key_total():
    """A key may span many rows; distance to the closest is what matters."""
    got = f(rec(10_000), stats(amounts=(50_000, 10_050, 90_000)))
    assert got["amount_delta_abs"] == 0.5, "50 minor units = 0.50 major"


def test_deltas_are_reported_in_major_units():
    """Feature importances are read by humans; paise-scale numbers are not."""
    assert f(rec(10_000), stats(amounts=(11_000,)))["amount_delta_abs"] == 10.0


def test_delta_against_key_total_is_a_separate_feature():
    got = f(rec(30_000), stats(amounts=(10_000, 20_000)))
    assert got["amount_exact"] == 0.0
    assert got["total_delta_abs"] == 0.0, "record equals the key's total, not any single row"


def test_relative_delta_is_scale_free():
    small = f(rec(100), stats(amounts=(110,)))["amount_delta_rel"]
    large = f(rec(100_000), stats(amounts=(110_000,)))["amount_delta_rel"]
    assert small == pytest.approx(large)


def test_zero_amount_does_not_divide_by_zero():
    assert np.all(np.isfinite(featurise(rec(0), stats(amounts=(500,)), n_candidates=1)))


# --------------------------------------------------------------------------- #
# date
# --------------------------------------------------------------------------- #

def test_gap_is_to_the_nearest_row_date():
    assert f(rec(day=100), stats(days=(90, 103, 200)))["date_gap_abs"] == 3.0


def test_sign_distinguishes_early_from_late():
    later = f(rec(day=105), stats(days=(100,)))["date_gap_signed"]
    earlier = f(rec(day=95), stats(days=(100,)))["date_gap_signed"]
    assert later > 0 > earlier


def test_missing_date_is_marked_rather_than_imputed():
    got = f(BankRecord("b", "A", 10_000, None), stats())
    assert got["date_missing"] == 1.0
    assert np.isfinite(got["date_gap_abs"])


def test_key_with_no_dates_is_marked():
    got = f(rec(), KeyStats(frozenset({10_000}), (), 1))
    assert got["date_missing"] == 1.0


# --------------------------------------------------------------------------- #
# key shape and ambiguity
# --------------------------------------------------------------------------- #

def test_row_count_is_exposed():
    assert f(rec(), stats(amounts=(1, 2, 3)))["key_n_rows"] == 3.0


def test_candidate_count_is_exposed():
    assert f(rec(), stats(), n_candidates=42)["n_candidates"] == 42.0


def test_single_row_key_is_flagged():
    assert f(rec(), stats(amounts=(10_000,)))["key_is_single_row"] == 1.0
    assert f(rec(), stats(amounts=(1, 2)))["key_is_single_row"] == 0.0


# --------------------------------------------------------------------------- #
# the model must not be able to see the answer
# --------------------------------------------------------------------------- #

def test_no_feature_is_named_after_an_outcome_column():
    forbidden = {"generatorallocation", "matchrule", "matchedby", "matchdate", "targetallocation"}
    assert not ({n.lower() for n in FEATURE_NAMES} & forbidden)


def test_featurise_takes_no_label_argument():
    import inspect

    params = set(inspect.signature(featurise).parameters)
    assert not (params & {"label", "y", "target", "is_match"})
