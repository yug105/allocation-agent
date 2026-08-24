"""BenchRec loader.

Obfuscated ledger and bank records from a Tier-1 institution's production
reconciliation, released for the ICAIF 2023 benchmark. Labels are real analyst
decisions.

Two conversions happen here and both are load-bearing:

* **money -> integer minor units.** Everything downstream assumes exact integer
  arithmetic. A float here quietly turns an exact-match problem into a fuzzy one.
* **dates -> day numbers.** Integer days make the blocking window a cheap range.

Both refuse rather than coerce. A silent NaN propagates; a raised error does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from allocation_agent.stores.keys import KeyRow
from allocation_agent.types import BankRecord

#: Columns recording how a match was resolved. Never loaded. See eval.leakage.
_OUTCOME_COLUMNS = frozenset({"generatorAllocation", "matchRule", "matchedBy", "matchDate"})

_EPOCH = date(2000, 1, 1)
_MULT = "MULT"


class ParseError(ValueError):
    """Raised when a value cannot be converted without losing information."""


def parse_minor(raw: str | None) -> int | None:
    """Parse a money string into an exact integer count of minor units."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    if not isinstance(raw, str):
        raise ParseError(f"expected str, got {type(raw).__name__}: floats have already lost precision")

    text = raw.strip().replace(",", "")
    if not text:
        return None

    neg = text.startswith("-")
    if neg:
        text = text[1:]

    whole, _, frac = text.partition(".")
    if not whole.isdigit() or (frac and not frac.isdigit()):
        raise ParseError(f"not a money amount: {raw!r}")
    if len(frac) > 2:
        raise ParseError(f"more precision than minor units allow: {raw!r}")

    minor = int(whole) * 100 + int(frac.ljust(2, "0") or 0)
    return -minor if neg else minor


def parse_day(raw: str | None) -> int | None:
    """Parse ``m/d/yy`` into days since the epoch."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        parsed = datetime.strptime(text, "%m/%d/%y").date()
    except ValueError as exc:
        raise ParseError(f"not a date: {raw!r}") from exc
    return (parsed - _EPOCH).days


@dataclass(frozen=True, slots=True)
class Dataset:
    """One loaded batch. Records, their labels, and the ledger rows to match against."""

    records: list[BankRecord]
    labels: list[str]
    """Allocation key, or ``MULT`` where the record spans several."""
    is_mult: np.ndarray
    key_rows: list[KeyRow]
    match_ids: list[str]
    """Group identifier. Never split across train and test."""

    @property
    def n_records(self) -> int:
        return len(self.records)


def load_benchrec(df: pd.DataFrame, *, strict: bool = False) -> Dataset:
    """Split a BenchRec frame into ledger rows and bank records.

    Args:
        df: raw frame, string dtype.
        strict: raise on an unparseable row rather than skipping it. Off by
            default so one bad row cannot fail a 190k batch; skipped rows are
            simply absent, which shows up as reduced recall rather than as a
            silent wrong answer.
    """
    key_rows: list[KeyRow] = []
    records: list[BankRecord] = []
    labels: list[str] = []
    match_ids: list[str] = []
    mult: list[bool] = []

    get = lambda row, col: row[col] if col in row and pd.notna(row[col]) else None

    for i, row in enumerate(df.to_dict("records")):
        try:
            a_amt = parse_minor(get(row, "A_amount"))
            b_amt = parse_minor(get(row, "B_amount"))
        except ParseError:
            if strict:
                raise
            continue

        alloc = get(row, "generatorAllocation")
        if a_amt is not None and alloc:
            key_rows.append(
                KeyRow(
                    key=str(alloc),
                    account=get(row, "A_account"),
                    amount_minor=a_amt,
                    day=parse_day(get(row, "A_valueDate")),
                )
            )

        if b_amt is not None:
            target = get(row, "targetAllocation")
            records.append(
                BankRecord(
                    record_id=f"b{i}",
                    account=get(row, "B_account") or get(row, "A_account"),
                    amount_minor=b_amt,
                    day=parse_day(get(row, "B_valueDate")),
                )
            )
            labels.append(str(target) if target else _MULT)
            mult.append(target == _MULT or not target)
            match_ids.append(str(get(row, "matchId") or i))

    return Dataset(
        records=records,
        labels=labels,
        is_mult=np.array(mult, dtype=bool),
        key_rows=key_rows,
        match_ids=match_ids,
    )
