"""Temporal splitting.

A random split leaks the future into the past and inflates every number. On a
published fraud benchmark the same model scored 0.925 average precision under a
random split and 0.537 under a forward-in-time one. The random number was not a
better result; it was a false one.

Two invariants: test is strictly later than train, and no match group straddles
the boundary.
"""

import numpy as np
import pytest

from allocation_agent.eval.splits import SplitError, temporal_split


def days(n: int) -> list[int]:
    return list(range(n))


def test_proportions_are_respected():
    s = temporal_split(days(100), groups=[str(i) for i in range(100)],
                       train_frac=0.7, val_frac=0.1)
    assert len(s.train) == pytest.approx(70, abs=2)
    assert len(s.val) == pytest.approx(10, abs=2)
    assert len(s.test) == pytest.approx(20, abs=2)


def test_every_index_lands_in_exactly_one_split():
    s = temporal_split(days(100), groups=[str(i) for i in range(100)])
    combined = np.concatenate([s.train, s.val, s.test])
    assert sorted(combined.tolist()) == list(range(100))


def test_test_is_strictly_later_than_train():
    d = days(100)
    s = temporal_split(d, groups=[str(i) for i in range(100)])
    assert max(d[i] for i in s.train) <= min(d[i] for i in s.test)


def test_val_sits_between_train_and_test():
    d = days(100)
    s = temporal_split(d, groups=[str(i) for i in range(100)])
    assert max(d[i] for i in s.train) <= min(d[i] for i in s.val)
    assert max(d[i] for i in s.val) <= min(d[i] for i in s.test)


# --------------------------------------------------------------------------- #
# groups must not straddle the boundary
# --------------------------------------------------------------------------- #

def test_a_group_spanning_the_cut_is_kept_whole():
    """Half a match in train and half in test leaks as surely as an outcome column."""
    d = list(range(20))
    groups = ["G"] * 20 if False else [("SPAN" if 12 <= i <= 16 else f"g{i}") for i in range(20)]
    s = temporal_split(d, groups=groups, train_frac=0.7, val_frac=0.0)
    train_g = {groups[i] for i in s.train}
    test_g = {groups[i] for i in s.test}
    assert not (train_g & test_g)


def test_unsorted_input_is_handled():
    d = [50, 10, 90, 30, 70, 20, 80, 40, 60, 100]
    s = temporal_split(d, groups=[str(i) for i in range(10)], train_frac=0.6, val_frac=0.0)
    assert max(d[i] for i in s.train) <= min(d[i] for i in s.test)


def test_missing_dates_are_placed_in_train_not_dropped():
    """A record with no date cannot be placed in time, but must not vanish."""
    d = [1, 2, None, 4, 5, 6, None, 8, 9, 10]
    s = temporal_split(d, groups=[str(i) for i in range(10)], train_frac=0.6, val_frac=0.0)
    total = len(s.train) + len(s.val) + len(s.test)
    assert total == 10
    assert 2 in s.train and 6 in s.train


# --------------------------------------------------------------------------- #
# refuse rather than return something useless
# --------------------------------------------------------------------------- #

def test_mismatched_lengths_raise():
    with pytest.raises(SplitError, match="length"):
        temporal_split(days(10), groups=["a", "b"])


def test_fractions_summing_above_one_raise():
    with pytest.raises(SplitError, match="frac"):
        temporal_split(days(10), groups=[str(i) for i in range(10)],
                       train_frac=0.8, val_frac=0.3)


def test_empty_input_raises_rather_than_returning_empty_splits():
    with pytest.raises(SplitError, match="empty"):
        temporal_split([], groups=[])


def test_single_group_cannot_be_split_and_says_so():
    """Everything in one group means no honest split exists."""
    with pytest.raises(SplitError, match="group"):
        temporal_split(days(10), groups=["only"] * 10)
