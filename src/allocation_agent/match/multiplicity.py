"""Multiplicity detection — does one bank record span several allocation keys?

11.3% of records in the batch (21,549 of 190,717). At the Tier-1 institution
this data came from, **none of them were resolved automatically**: every grouped
match carries `matchRule == MANUAL`. This is the decision that consumes the
human hours, and no product on the market attempts it.

Measured separators (median unless noted):

===================================  ==========  ==========  =======
signal                                     MULT      single    ratio
===================================  ==========  ==========  =======
**has an exact-amount candidate**       **10.9%**  **91.9%**    ---
blocked candidates                            62          22     2.8x
amount (minor units)                   1,124,239     664,703     1.7x
round to Rs 1,000                           0.0%        0.0%      ---
===================================  ==========  ==========  =======

The design predicted amount size would be the strongest signal. It is not.
Whether *any single ledger row matches the amount exactly* separates the classes
far more sharply, and it follows directly from what a grouped payment is: a sum
of several rows matches no single row.

Round numbers were predicted to help and do not: zero records in either class are
round to Rs 1,000. Recorded rather than quietly dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MULT_FEATURE_NAMES: tuple[str, ...] = (
    "has_exact_amount_candidate",
    "n_candidates",
    "log_amount",
    "amount_vs_account_median",
    "account_mult_rate",
    "min_amount_delta_major",
    "min_amount_delta_rel",
)


@dataclass(frozen=True, slots=True)
class AccountPrior:
    """Per-account statistics. **Computed from training rows only.**

    Fitting these on the evaluation split would leak the label: an account's
    MULT rate is a summary of the very thing being predicted.
    """

    median_amount_minor: dict[str, float]
    mult_rate: dict[str, float]
    global_median: float
    global_mult_rate: float

    @classmethod
    def fit(cls, accounts, amounts, is_mult) -> AccountPrior:
        med: dict[str, list[float]] = {}
        rate: dict[str, list[int]] = {}
        for a, amt, m in zip(accounts, amounts, is_mult, strict=False):
            med.setdefault(a, []).append(abs(float(amt)))
            rate.setdefault(a, []).append(int(m))
        return cls(
            median_amount_minor={k: float(np.median(v)) for k, v in med.items()},
            mult_rate={k: float(np.mean(v)) for k, v in rate.items()},
            global_median=float(np.median([abs(float(a)) for a in amounts])) or 1.0,
            global_mult_rate=float(np.mean([int(m) for m in is_mult])),
        )


def featurise_multiplicity(
    record,
    *,
    n_candidates: int,
    has_exact: bool,
    min_delta_minor: float,
    prior: AccountPrior,
) -> np.ndarray:
    """Record-level features. No candidate identity, no label."""
    acct = record.account or "?"
    amt = abs(float(record.amount_minor))

    med = prior.median_amount_minor.get(acct, prior.global_median) or 1.0
    rate = prior.mult_rate.get(acct, prior.global_mult_rate)
    delta = min(min_delta_minor, 1e12)

    return np.array(
        [
            1.0 if has_exact else 0.0,
            float(n_candidates),
            math.log10(amt + 1.0),
            amt / med,
            rate,
            delta / 100.0,
            min(delta / max(amt, 1.0), 1e6),
        ],
        dtype=np.float64,
    )


class MultiplicityDetector:
    """Binary classifier. Reports probability, never decides."""

    def __init__(self, n_estimators: int = 200, random_state: int = 0) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> MultiplicityDetector:
        from lightgbm import LGBMClassifier

        pos = max(int(y.sum()), 1)
        self._model = LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=0.08,
            num_leaves=31,
            random_state=self.random_state,
            scale_pos_weight=(len(y) - pos) / pos,
            verbose=-1,
        )
        self._model.fit(X, y, feature_name=list(MULT_FEATURE_NAMES))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("MultiplicityDetector is not fitted")
        return self._model.predict_proba(X)[:, 1]

    @property
    def importances(self) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("MultiplicityDetector is not fitted")
        total = self._model.feature_importances_.sum() or 1
        return dict(sorted(zip(MULT_FEATURE_NAMES, self._model.feature_importances_ / total, strict=False),
                           key=lambda kv: -kv[1]))
