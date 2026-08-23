"""Read an arbitrary bank/ledger CSV pair well enough to reconcile it.

The demo runs on a dataset the visitor has never seen, scored against labels
they cannot inspect. That is exactly the situation where a reasonable person
withholds belief. This module exists so they can put in a file whose answers
they already know.

Two rules pull in opposite directions and both are kept:

* **Forgiving about names.** Nobody's export calls a column ``amount_minor``.
  Real ones say ``Txn Amt (INR)``, ``Val Dt``, ``A/c No``. Columns are found by
  looking, not by demanding.
* **Unforgiving about money.** A third decimal place is refused, never rounded.
  Silently altering someone's figures is the one unrecoverable failure here --
  every other error announces itself, that one balances and lies.

Errors name the row number. "Could not parse file" sends someone to open a
10,000-line CSV and guess.
"""

from __future__ import annotations

import csv
import io
import re
from datetime import date, datetime

from allocation_agent.stores.keys import KeyRow
from allocation_agent.types import BankRecord

_EPOCH = date(2000, 1, 1)


class UploadError(ValueError):
    """A problem with the supplied file, phrased for the person who supplied it."""


# Ordered: the first pattern that hits wins, so exact names beat lookalikes.
_PATTERNS: dict[str, list[str]] = {
    "amount": [r"^amount$", r"^amt$", r"^value$",
               r"\bamount\b", r"\bamt\b", r"\bcredit\b", r"\bdebit\b", r"\bvalue\b"],
    "date":   [r"^date$", r"^day$",
               r"value.?date", r"\bdate\b", r"\bdt\b", r"posted", r"booked"],
    "account": [r"^account$", r"^acct$",
                r"account.?(no|number|id)", r"\bacc?t?\b", r"\ba/c\b", r"\biban\b"],
    "key":    [r"^key$", r"^id$",
               r"invoice", r"\bref(erence)?\b", r"allocation", r"\bkey\b", r"\bid\b"],
}


def sniff_columns(headers: list[str]) -> dict[str, str]:
    """Map canonical field -> the column in *headers* that carries it.

    Two passes per field so that a file containing both ``amount`` and
    ``settled_amount`` binds the plain one. Fields with no match are absent
    from the result; the caller decides which of those are fatal.
    """
    found: dict[str, str] = {}
    for field, patterns in _PATTERNS.items():
        for pattern in patterns:
            for header in headers:
                if header is not None and re.search(pattern, header.strip().lower()):
                    found[field] = header
                    break
            if field in found:
                break
    return found


def _rows(text: str) -> tuple[list[dict[str, str]], list[str]]:
    reader = csv.DictReader(io.StringIO(text))
    headers = [h for h in (reader.fieldnames or []) if h]
    if not headers:
        raise UploadError("the file has no header row")
    rows = [r for r in reader if any((v or "").strip() for v in r.values())]
    if not rows:
        raise UploadError("the file has no rows below the header")
    return rows, headers


def _minor(raw: str | None, row_no: int) -> int:
    """Money, as an exact integer count of minor units."""
    text = (raw or "").strip().replace(",", "").replace(" ", "")
    text = re.sub(r"^[^\d\-+.]+|[^\d]*$", "", text) if text else text
    if not text:
        raise UploadError(f"row {row_no}: the amount is blank")

    neg = text.startswith("-") or (raw or "").strip().startswith("(")
    text = text.lstrip("-+")

    whole, _, frac = text.partition(".")
    whole = whole or "0"
    if not whole.isdigit() or (frac and not frac.isdigit()):
        raise UploadError(f"row {row_no}: {raw!r} is not an amount")
    if len(frac) > 2:
        raise UploadError(
            f"row {row_no}: {raw!r} has more precision than minor units allow. "
            "Refusing rather than rounding someone's money."
        )
    minor = int(whole) * 100 + int(frac.ljust(2, "0") or 0)
    return -minor if neg else minor


# Ambiguous layouts first so a file that fits both is settled by the whole-file
# vote below rather than by whichever row happens to come first.
_DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y",
    "%d/%m/%y", "%m/%d/%y", "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y",
    "%d.%m.%Y", "%Y%m%d",
]


def _pick_format(samples: list[str]) -> str:
    """Choose one layout for the whole file.

    ``03/01/2026`` is readable two ways and so is ``05/02/2026``. Deciding per
    row mixes day-first and month-first inside a single file and shifts dates by
    weeks with nothing visibly wrong. So the format that parses the most rows
    wins, ties broken by the order above.
    """
    best, best_n = None, 0
    for fmt in _DATE_FORMATS:
        n = 0
        for s in samples:
            try:
                datetime.strptime(s, fmt)
                n += 1
            except ValueError:
                pass
        if n > best_n:
            best, best_n = fmt, n
        if best_n == len(samples):
            break
    if best is None:
        raise UploadError(
            f"could not read the date column: {samples[0]!r} matches no layout "
            "this accepts (try YYYY-MM-DD)"
        )
    return best


def _parse(text: str, *, key_field: str, id_prefix: str):
    rows, headers = _rows(text)
    cols = sniff_columns(headers)

    missing = [f for f in ("account", "amount", "date") if f not in cols]
    if missing:
        raise UploadError(
            f"could not find a column for: {', '.join(missing)}. "
            f"The file has: {', '.join(headers)}"
        )

    stamps = [(r.get(cols["date"]) or "").strip() for r in rows]
    blank = next((i for i, s in enumerate(stamps) if not s), None)
    if blank is not None:
        raise UploadError(f"row {blank + 2}: the date is blank")
    fmt = _pick_format(stamps)

    out = []
    for i, row in enumerate(rows):
        row_no = i + 2                     # header is row 1
        try:
            day = (datetime.strptime(stamps[i], fmt).date() - _EPOCH).days
        except ValueError as exc:
            raise UploadError(
                f"row {row_no}: {stamps[i]!r} is not a date in this file's "
                f"layout ({fmt})"
            ) from exc

        account = (row.get(cols["account"]) or "").strip() or None
        ident = (row.get(cols[key_field]) or "").strip() if key_field in cols else ""
        out.append((ident or f"{id_prefix}{row_no}", account,
                    _minor(row.get(cols["amount"]), row_no), day))
    return out


def parse_bank_csv(text: str) -> list[BankRecord]:
    """Money that arrived: the side being explained."""
    return [BankRecord(i, a, m, d) for i, a, m, d in
            _parse(text, key_field="key", id_prefix="row-")]


def parse_ledger_csv(text: str) -> list[KeyRow]:
    """What the books say it should be: the side being matched against."""
    return [KeyRow(k, a, m, d) for k, a, m, d in
            _parse(text, key_field="key", id_prefix="row-")]
