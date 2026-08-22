"""Features for one (bank record, candidate key) pair.

What the design assumed and what this data actually has diverged sharply.
Measured field population on the BenchRec batch:

=========================  ==========  =========================================
field                       populated   consequence
=========================  ==========  =========================================
transactionReferences            0.0%   no reference-overlap feature
orderingPartyInfo                0.0%   **no counterparty names at all**
receivingPartyInfo               0.0%   identity resolution is untestable here
currencyCode                     100%   one distinct value: no signal
debitOrCredit                    100%   one distinct value ("NONE"): no signal
transactionAttributes            100%   different vocabularies per side
amount, valueDate, account       100%   the usable signal
=========================  ==========  =========================================

So the feature set is thinner than designed, and honestly so: amount, date, key
shape, ambiguity. The absence of names means the alias layer cannot be evaluated
on this dataset -- that belongs in the limitations, not in a silent omission.

The bar to beat is a trivial baseline (exact amount, tiebreak nearest date)
scoring **90.22%** top-1, against a blocking ceiling of **98.94%**. Roughly 8.7
points of headroom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from allocation_agent.types import BankRecord

#: Order is part of the contract: the trained model indexes by position.
FEATURE_NAMES: tuple[str, ...] = (
    "amount_exact",       # record equals some single ledger row of this key
    "amount_delta_abs",   # distance to the nearest row, in major units
    "amount_delta_rel",   # same, scaled by record size
    "total_delta_abs",    # distance to the key's total across all its rows
    "total_delta_rel",
    "date_gap_abs",       # days to the nearest row date
    "date_gap_signed",    # negative = record precedes the ledger row
    "date_missing",       # either side has no date
    "key_n_rows",
    "key_is_single_row",
    "key_total_major",
    "n_candidates",       # ambiguity of the block this pair came from
)

_NO_DATE_GAP = 999.0
_MAJOR = 100.0


@dataclass(frozen=True, slots=True)
class KeyStats:
    """Everything about a candidate key the scorer is allowed to see.

    Deliberately excludes the key string, so the model cannot memorise
    identifiers instead of learning the relationship.
    """

    amounts: frozenset[int]
    days: tuple[int, ...]
    n_rows: int

    @property
    def total_minor(self) -> int:
        return sum(self.amounts)


def featurise(
    record: BankRecord,
    key: KeyStats,
    *,
    n_candidates: int,
) -> np.ndarray:
    """Build the feature vector for one candidate pair.

    Every value is finite. Missing inputs are flagged with an indicator rather
    than imputed -- an imputed zero is indistinguishable from a real zero, and
    absence is itself signal.
    """
    amt = record.amount_minor

    if key.amounts:
        nearest = min(key.amounts, key=lambda a: abs(a - amt))
        delta_abs = abs(amt - nearest) / _MAJOR
        exact = 1.0 if amt in key.amounts else 0.0
    else:
        delta_abs, exact = _NO_DATE_GAP, 0.0

    total = key.total_minor
    total_delta = abs(amt - total) / _MAJOR

    scale = max(abs(amt), 1) / _MAJOR
    delta_rel = min(delta_abs / scale, 1e6)
    total_delta_rel = min(total_delta / scale, 1e6)

    if record.day is not None and key.days:
        signed = min((record.day - d for d in key.days), key=abs)
        gap_abs, gap_signed, missing = float(abs(signed)), float(signed), 0.0
    else:
        gap_abs, gap_signed, missing = _NO_DATE_GAP, 0.0, 1.0

    return np.array(
        [
            exact,
            delta_abs,
            delta_rel,
            total_delta,
            total_delta_rel,
            gap_abs,
            gap_signed,
            missing,
            float(key.n_rows),
            1.0 if key.n_rows == 1 else 0.0,
            total / _MAJOR,
            float(n_candidates),
        ],
        dtype=np.float64,
    )


def build_key_stats(key_rows) -> dict[str, KeyStats]:
    """Aggregate ledger rows into per-key statistics, once per batch."""
    amounts: dict[str, set[int]] = {}
    days: dict[str, list[int]] = {}
    counts: dict[str, int] = {}

    for row in key_rows:
        amounts.setdefault(row.key, set()).add(row.amount_minor)
        counts[row.key] = counts.get(row.key, 0) + 1
        if row.day is not None:
            days.setdefault(row.key, []).append(row.day)

    return {
        k: KeyStats(
            amounts=frozenset(v),
            days=tuple(sorted(days.get(k, ()))),
            n_rows=counts[k],
        )
        for k, v in amounts.items()
    }
