"""Candidate ranker.

Scores one (bank record, candidate key) pair. Ranking is a supervised problem:
169,168 records carry a known correct key, and blocking supplies the wrong
answers that were plausible enough to consider. Those are the negatives worth
training on -- random keys teach nothing, because blocking already excluded them.

The model ranks. It never commits: the gate decides whether a score is high
enough to post, and that is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from allocation_agent.match.features import FEATURE_NAMES


@dataclass
class RankerConfig:
    n_estimators: int = 300
    learning_rate: float = 0.08
    num_leaves: int = 63
    min_child_samples: int = 40
    max_negatives_per_record: int = 24
    """Hard negatives from the same block. Capped to bound memory and to stop
    records with huge candidate sets dominating the training set."""
    random_state: int = 0
    objective: str = "rank"
    """``rank`` uses LambdaRank with the record as the group -- the problem is
    choose-best-of-N, not classify-each-pair. ``binary`` keeps the simpler
    formulation for comparison."""


class Ranker:
    """Thin wrapper over gradient-boosted trees, with calibration."""

    def __init__(self, config: RankerConfig | None = None) -> None:
        self.config = config or RankerConfig()
        self._model = None
        self._calibrator = None
        self._ranking = False

    def fit(self, X, y, X_cal=None, y_cal=None, group=None, group_cal=None) -> Ranker:
        """Fit the scorer.

        Args:
            group: sizes of each record's candidate set, required for ``rank``.
                Ranking optimises the ordering *within* a record, which is the
                actual task; binary classification treats every pair as
                independent and wastes capacity on features that are constant
                across a record's candidates.
        """
        cfg = self.config
        common = dict(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            min_child_samples=cfg.min_child_samples,
            random_state=cfg.random_state,
            verbose=-1,
        )

        if cfg.objective == "rank":
            from lightgbm import LGBMRanker

            if group is None:
                raise ValueError("objective='rank' needs group sizes")
            self._model = LGBMRanker(objective="lambdarank", **common)
            self._model.fit(X, y, group=group, feature_name=list(FEATURE_NAMES))
            self._ranking = True
        else:
            from lightgbm import LGBMClassifier

            pos = max(int(y.sum()), 1)
            self._model = LGBMClassifier(scale_pos_weight=(len(y) - pos) / pos, **common)
            self._model.fit(X, y, feature_name=list(FEATURE_NAMES))
            self._ranking = False

        if X_cal is not None and len(X_cal):
            from sklearn.isotonic import IsotonicRegression

            raw = self._model.predict_proba(X_cal)[:, 1]
            self._calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw, y_cal)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Score each pair. Higher is better.

        Under ``rank`` these are relevance scores, not probabilities: they order
        candidates within a record but carry no absolute meaning. Calibration
        (needed by the gate) is applied separately, per record, from the score
        margin.
        """
        if self._model is None:
            raise RuntimeError("Ranker is not fitted")
        if getattr(self, "_ranking", False):
            return self._model.predict(X)
        raw = self._model.predict_proba(X)[:, 1]
        return self._calibrator.predict(raw) if self._calibrator is not None else raw

    @property
    def importances(self) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("Ranker is not fitted")
        total = self._model.feature_importances_.sum() or 1
        return dict(
            sorted(
                zip(FEATURE_NAMES, self._model.feature_importances_ / total, strict=False),
                key=lambda kv: -kv[1],
            )
        )
