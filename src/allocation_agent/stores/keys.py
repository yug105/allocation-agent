"""Allocation-key index.

Two inverted indexes built once over the ledger side, so blocking is a hash
lookup rather than a scan. Both are point lookups: no ordering, no traversal.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KeyRow:
    """One ledger row contributing to an allocation key.

    A key may span many rows. Each row is indexed separately, so the key is
    reachable from any of its constituent amounts or dates.
    """

    key: str
    account: str | None
    amount_minor: int
    day: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.amount_minor, int) or isinstance(self.amount_minor, bool):
            raise TypeError(
                f"amount_minor must be int minor units, got {type(self.amount_minor).__name__}"
            )


class KeyIndex:
    """Inverted indexes over ``(account, amount)`` and ``(account, day)``."""

    __slots__ = ("by_amount", "by_day", "_keys")

    def __init__(self, rows: Iterable[KeyRow]) -> None:
        self.by_amount: dict[tuple[str, int], set[str]] = defaultdict(set)
        self.by_day: dict[tuple[str, int], set[str]] = defaultdict(set)
        self._keys: set[str] = set()

        for r in rows:
            self._keys.add(r.key)
            if r.account is None:
                continue  # unindexable: cannot be reached without widening to everything
            self.by_amount[(r.account, r.amount_minor)].add(r.key)
            if r.day is not None:
                self.by_day[(r.account, r.day)].add(r.key)

    @property
    def n_keys(self) -> int:
        return len(self._keys)

    def __len__(self) -> int:
        return len(self._keys)
