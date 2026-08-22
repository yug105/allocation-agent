"""Ranker contract. Learning quality is measured on real data, not asserted here."""

import numpy as np
import pytest

from allocation_agent.match.features import FEATURE_NAMES
from allocation_agent.match.ranker import Ranker, RankerConfig


def toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    y = (X[:, 0] > 0).astype(int)          # first feature decides
    X[:, 0] += rng.normal(0, 0.2, n)
    return X, y


def test_unfitted_ranker_refuses_to_score():
    with pytest.raises(RuntimeError, match="not fitted"):
        Ranker().score(np.zeros((1, len(FEATURE_NAMES))))


def test_scores_are_probabilities():
    X, y = toy()
    s = Ranker(RankerConfig(n_estimators=30)).fit(X, y).score(X)
    assert s.shape == (len(X),)
    assert np.all((s >= 0) & (s <= 1))


def test_learns_a_separable_signal():
    X, y = toy()
    s = Ranker(RankerConfig(n_estimators=60)).fit(X, y).score(X)
    assert ((s > 0.5).astype(int) == y).mean() > 0.9


def test_importances_cover_every_feature_and_sum_to_one():
    X, y = toy()
    imp = Ranker(RankerConfig(n_estimators=30)).fit(X, y).importances
    assert set(imp) == set(FEATURE_NAMES)
    assert sum(imp.values()) == pytest.approx(1.0)


def test_calibration_keeps_scores_in_range():
    X, y = toy()
    r = Ranker(RankerConfig(n_estimators=40)).fit(X[:300], y[:300], X[300:], y[300:])
    s = r.score(X)
    assert np.all((s >= 0) & (s <= 1))


def test_same_seed_gives_identical_scores():
    """Reproducibility is a requirement, not a nicety."""
    X, y = toy()
    a = Ranker(RankerConfig(n_estimators=30)).fit(X, y).score(X)
    b = Ranker(RankerConfig(n_estimators=30)).fit(X, y).score(X)
    assert np.allclose(a, b)
