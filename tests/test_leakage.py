"""Leakage guard.

Four columns in the source data silently hand over the answer. Any feature set
containing them (or anything derived from them) produces a near-perfect score
that means nothing. This must fail loudly, not warn.
"""

import numpy as np
import pandas as pd
import pytest

from allocation_agent.eval.leakage import (
    FORBIDDEN_COLUMNS,
    LeakageError,
    assert_group_split,
    assert_no_leakage,
    find_suspicious_columns,
)

# --------------------------------------------------------------------------- #
# forbidden columns
# --------------------------------------------------------------------------- #

def test_forbidden_set_is_exactly_the_four_known_leaks():
    assert frozenset(
        {"generatorAllocation", "matchRule", "matchedBy", "matchDate"}
    ) == FORBIDDEN_COLUMNS


def test_clean_feature_frame_passes():
    df = pd.DataFrame({"amount_delta": [1, 2], "date_gap": [0, 3]})
    assert_no_leakage(df, y=pd.Series([0, 1]))  # must not raise


@pytest.mark.parametrize("col", sorted(FORBIDDEN_COLUMNS))
def test_each_forbidden_column_raises(col):
    df = pd.DataFrame({"amount_delta": [1, 2], col: ["a", "b"]})
    with pytest.raises(LeakageError, match=col):
        assert_no_leakage(df, y=pd.Series([0, 1]))


def test_error_names_every_offending_column_not_just_the_first():
    df = pd.DataFrame({"matchRule": [1, 2], "matchedBy": [3, 4]})
    with pytest.raises(LeakageError) as exc:
        assert_no_leakage(df, y=pd.Series([0, 1]))
    assert "matchRule" in str(exc.value)
    assert "matchedBy" in str(exc.value)


def test_forbidden_check_is_case_insensitive():
    """Column casing varies between sources; the guard must not be fooled."""
    df = pd.DataFrame({"MATCHRULE": [1, 2]})
    with pytest.raises(LeakageError):
        assert_no_leakage(df, y=pd.Series([0, 1]))


# --------------------------------------------------------------------------- #
# derived leaks — a renamed copy of the label is still the label
# --------------------------------------------------------------------------- #

def test_column_that_perfectly_predicts_the_label_is_flagged():
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1] * 4)
    df = pd.DataFrame({"innocent_name": y.values, "noise": np.arange(len(y))})
    suspicious = find_suspicious_columns(df, y)
    assert "innocent_name" in suspicious
    assert "noise" not in suspicious


def test_assert_no_leakage_raises_on_a_derived_leak():
    y = pd.Series([0, 1] * 16)
    df = pd.DataFrame({"sneaky": y.values})
    with pytest.raises(LeakageError, match="sneaky"):
        assert_no_leakage(df, y)


def test_genuinely_predictive_feature_is_not_flagged():
    """A strong feature is not a leak. Only near-perfect determination is."""
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, 200))
    noisy = y.values ^ rng.binomial(1, 0.25, 200)  # ~75% agreement
    df = pd.DataFrame({"strong_feature": noisy})
    assert "strong_feature" not in find_suspicious_columns(df, y)


# --------------------------------------------------------------------------- #
# group split — half a match in train and half in test is the same leak
# --------------------------------------------------------------------------- #

def test_disjoint_groups_pass():
    train = pd.DataFrame({"matchId": [1, 1, 2]})
    test = pd.DataFrame({"matchId": [3, 4]})
    assert_group_split(train, test, group_col="matchId")


def test_shared_group_raises_and_names_the_id():
    train = pd.DataFrame({"matchId": [1, 2]})
    test = pd.DataFrame({"matchId": [2, 3]})
    with pytest.raises(LeakageError, match="2"):
        assert_group_split(train, test, group_col="matchId")


def test_group_split_reports_how_many_overlap():
    train = pd.DataFrame({"matchId": [1, 2, 3]})
    test = pd.DataFrame({"matchId": [1, 2, 3]})
    with pytest.raises(LeakageError, match="3"):
        assert_group_split(train, test, group_col="matchId")


def test_missing_group_column_is_an_error_not_a_silent_pass():
    train = pd.DataFrame({"other": [1]})
    test = pd.DataFrame({"other": [2]})
    with pytest.raises(LeakageError, match="matchId"):
        assert_group_split(train, test, group_col="matchId")
