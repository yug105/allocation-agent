"""Core data contracts.

Money is always an integer count of minor units (paise). Never a float — a
match either balances exactly or it does not, and floating point makes that
question unanswerable.

Absent fields are ``None``, never a sentinel. A string that looks like an
identifier ("UNRESOLVED", "") is an equality hazard: downstream code compares it
and silently treats unknown as known.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BankRecord:
    """One external record awaiting an allocation key."""

    record_id: str
    account: str | None
    amount_minor: int
    day: int | None
    """Days since a fixed epoch. Integer so date windows are cheap range scans."""

    def __post_init__(self) -> None:
        if not isinstance(self.amount_minor, int) or isinstance(self.amount_minor, bool):
            raise TypeError(
                f"amount_minor must be int minor units, got {type(self.amount_minor).__name__}"
            )
