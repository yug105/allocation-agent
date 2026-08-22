"""Temporal, group-respecting splits.

Two rules, both non-negotiable:

**Split by time, not at random.** A random split lets the model see the future
while predicting the past. Published measurement on a fraud benchmark: the same
model scored 0.925 average precision on a random split and 0.537 on a
forward-in-time one. The random figure was not better, it was false.

**Never split a match group.** Half a match in train and half in test leaks the
answer exactly as an outcome column would. Where a group straddles the cut, the
whole group goes to the earlier side.

Test is frozen. Touch it once, at the end.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


class SplitError(ValueError):
    """Raised when no honest split is possible."""


@dataclass(frozen=True, slots=True)
class Split:
    """Row indices for each partition."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def __str__(self) -> str:
        n = len(self.train) + len(self.val) + len(self.test)
        return (
            f"train {len(self.train):,} ({len(self.train)/n:.0%})  "
            f"val {len(self.val):,} ({len(self.val)/n:.0%})  "
            f"test {len(self.test):,} ({len(self.test)/n:.0%})"
        )


def temporal_split(
    days: Sequence[int | None],
    groups: Sequence[str],
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
) -> Split:
    """Partition rows by time, keeping every group intact.

    Args:
        days: day number per row. ``None`` cannot be placed in time and is put
            in train rather than dropped -- a record that vanishes silently is
            worse than one evaluated conservatively.
        groups: group identifier per row, typically ``matchId``.
        train_frac, val_frac: the remainder becomes test.

    Raises:
        SplitError: on empty input, mismatched lengths, impossible fractions, or
            a single group covering everything.
    """
    if len(days) != len(groups):
        raise SplitError(f"days and groups differ in length: {len(days)} vs {len(groups)}")
    if not days:
        raise SplitError("cannot split an empty dataset")
    if train_frac <= 0 or val_frac < 0 or train_frac + val_frac >= 1.0:
        raise SplitError(
            f"train_frac + val_frac must be < 1.0 and positive, got {train_frac} + {val_frac}"
        )
    if len(set(groups)) < 2:
        raise SplitError("all rows share one group; no split can separate them")

    # A group is placed by its earliest dated row, so a group spanning the cut
    # falls entirely on the earlier side.
    group_day: dict[str, int] = {}
    for day, group in zip(days, groups):
        if day is None:
            continue
        prev = group_day.get(group)
        if prev is None or day < prev:
            group_day[group] = day

    undated = [g for g in set(groups) if g not in group_day]
    ordered = sorted(group_day, key=lambda g: (group_day[g], g))

    n = len(ordered)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    assignment: dict[str, int] = {g: 0 for g in undated}  # undated -> train
    for i, group in enumerate(ordered):
        assignment[group] = 0 if i < train_end else (1 if i < val_end else 2)

    buckets: list[list[int]] = [[], [], []]
    for i, group in enumerate(groups):
        buckets[assignment[group]].append(i)

    return Split(
        train=np.array(buckets[0], dtype=int),
        val=np.array(buckets[1], dtype=int),
        test=np.array(buckets[2], dtype=int),
    )
