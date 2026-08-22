"""Multiplicity detector contract and the leakage risk in its account prior."""

import numpy as np
import pytest

from allocation_agent.match.multiplicity import (
    MULT_FEATURE_NAMES, AccountPrior, MultiplicityDetector, featurise_multiplicity,
)
from allocation_agent.types import BankRecord


def prior() -> AccountPrior:
    return AccountPrior.fit(["A", "A", "B", "B"], [100, 300, 1000, 5000], [0, 1, 0, 0])


def rec(minor=200, acct="A") -> BankRecord:
    return BankRecord("b", acct, minor, 100)


def f(**kw):
    base = dict(n_candidates=5, has_exact=True, min_delta_minor=0.0, prior=prior())
    base.update(kw)
    return dict(zip(MULT_FEATURE_NAMES, featurise_multiplicity(rec(), **base)))


def test_vector_length_matches_names():
    assert len(featurise_multiplicity(rec(), n_candidates=1, has_exact=True,
                                      min_delta_minor=0, prior=prior())) == len(MULT_FEATURE_NAMES)


def test_exact_amount_availability_is_exposed():
    assert f(has_exact=True)["has_exact_amount_candidate"] == 1.0
    assert f(has_exact=False)["has_exact_amount_candidate"] == 0.0


def test_account_prior_is_used_when_the_account_is_known():
    assert f()["account_mult_rate"] == pytest.approx(0.5)


def test_unknown_account_falls_back_to_the_global_rate_not_zero():
    v = featurise_multiplicity(rec(acct="NEVER_SEEN"), n_candidates=1, has_exact=True,
                               min_delta_minor=0, prior=prior())
    assert dict(zip(MULT_FEATURE_NAMES, v))["account_mult_rate"] == pytest.approx(0.25)


def test_features_are_finite_for_degenerate_input():
    v = featurise_multiplicity(BankRecord("b", None, 0, None), n_candidates=0,
                               has_exact=False, min_delta_minor=1e18, prior=prior())
    assert np.all(np.isfinite(v))


def test_zero_amount_does_not_divide_by_zero():
    v = featurise_multiplicity(rec(minor=0), n_candidates=1, has_exact=False,
                               min_delta_minor=500, prior=prior())
    assert np.all(np.isfinite(v))


def test_unfitted_detector_refuses_to_predict():
    with pytest.raises(RuntimeError, match="not fitted"):
        MultiplicityDetector().predict_proba(np.zeros((1, len(MULT_FEATURE_NAMES))))


def test_learns_a_separable_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, len(MULT_FEATURE_NAMES)))
    y = (X[:, 0] > 0).astype(int)
    p = MultiplicityDetector(n_estimators=50).fit(X, y).predict_proba(X)
    assert ((p > 0.5).astype(int) == y).mean() > 0.9


def test_probabilities_are_in_range():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, len(MULT_FEATURE_NAMES)))
    y = rng.integers(0, 2, 200)
    p = MultiplicityDetector(n_estimators=20).fit(X, y).predict_proba(X)
    assert np.all((p >= 0) & (p <= 1))
