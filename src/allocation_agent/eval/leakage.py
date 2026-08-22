"""Leakage guard — run before anything else.

Four columns in the source data are outcomes, not inputs. A feature set that
contains them (or a renamed copy of the label) scores near-perfectly and means
nothing. Splitting one match across train and test is the same leak by another
route.

These raise rather than warn. A silent leak is worse than a crash: the crash is
found in seconds, the leak is found in the panel interview.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

#: Columns that record how a match was resolved. All are known only after the
#: fact; ``generatorAllocation`` is the answer itself in a different format.
FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    {"generatorAllocation", "matchRule", "matchedBy", "matchDate"}
)

#: Above this normalised mutual information with the label, a column is treated
#: as a renamed copy of the answer rather than a strong feature. Deliberately
#: high: real features correlate, only leaks determine.
LEAK_MI_THRESHOLD: float = 0.95

#: Below this many rows the MI estimate is too noisy to act on.
MIN_ROWS_FOR_MI: int = 20


class LeakageError(AssertionError):
    """Raised when a feature set or split would leak the answer."""


def _discretise(s: pd.Series, bins: int = 20) -> np.ndarray:
    """Map any column to integer codes so MI can be computed uniformly."""
    if s.dtype.kind in "biufc":
        if s.nunique(dropna=False) <= bins:
            return pd.factorize(s, use_na_sentinel=False)[0]
        return pd.factorize(
            pd.qcut(s.rank(method="first"), q=bins, duplicates="drop"),
            use_na_sentinel=False,
        )[0]
    return pd.factorize(s.astype("string"), use_na_sentinel=False)[0]


def find_suspicious_columns(
    features: pd.DataFrame,
    y: pd.Series,
    threshold: float = LEAK_MI_THRESHOLD,
) -> list[str]:
    """Columns that determine the label almost exactly.

    Catches leaks that survived a rename. A genuinely strong feature correlates
    with the label; a leak *is* the label, so its normalised mutual information
    sits at or near 1.0.
    """
    if len(features) < MIN_ROWS_FOR_MI:
        return []

    y_codes = _discretise(y)
    suspicious: list[str] = []
    for col in features.columns:
        try:
            mi = normalized_mutual_info_score(y_codes, _discretise(features[col]))
        except ValueError:  # degenerate column, nothing to learn from it
            continue
        if mi >= threshold:
            suspicious.append(col)
    return suspicious


def assert_no_leakage(
    features: pd.DataFrame,
    y: pd.Series,
    *,
    check_derived: bool = True,
) -> None:
    """Raise if *features* contains a forbidden column or a renamed label.

    Args:
        features: the frame about to be handed to a model.
        y: the label, aligned to ``features``.
        check_derived: also run the mutual-information check. Disable only for
            speed on a frame already known to be clean.

    Raises:
        LeakageError: naming every offending column, not just the first.
    """
    lowered = {c.lower(): c for c in features.columns}
    forbidden = sorted(
        lowered[f.lower()] for f in FORBIDDEN_COLUMNS if f.lower() in lowered
    )
    if forbidden:
        raise LeakageError(
            f"Outcome columns present in features: {forbidden}. "
            "These record how a match was resolved and are unknown at prediction time."
        )

    if check_derived:
        derived = find_suspicious_columns(features, y)
        if derived:
            raise LeakageError(
                f"Columns determine the label almost exactly (NMI >= "
                f"{LEAK_MI_THRESHOLD}): {derived}. "
                "A renamed copy of the answer is still the answer."
            )


def assert_group_split(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    group_col: str = "matchId",
) -> None:
    """Raise if any group appears on both sides of a split.

    Half a match in training and half in test leaks just as surely as an
    outcome column.
    """
    for name, frame in (("train", train), ("test", test)):
        if group_col not in frame.columns:
            raise LeakageError(f"{name} has no '{group_col}' column; cannot verify the split.")

    overlap = set(train[group_col]) & set(test[group_col])
    if overlap:
        sample = sorted(map(str, overlap))[:5]
        raise LeakageError(
            f"{len(overlap)} '{group_col}' value(s) appear in both train and test, "
            f"e.g. {sample}. Split by group, never by row."
        )
