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


def _llm_sniff_columns(headers: list[str], backend) -> dict[str, str]:
    """Ask an LLM to map headers when regex doesn't recognise them.

    Real-world files call their columns anything: ``Txn Amt (INR)``,
    ``Val Dt``, ``A/c No``, ``Inv Ref``. A regex vocabulary has edges; a model
    that has read thousands of financial exports can interpolate. The result is
    validated the same way the regex result is — if a required field is missing
    from the response, the caller raises rather than guessing.
    """
    import json as _json

    prompt = (
        "TASK: You are mapping CSV column headers from a financial file to canonical fields.\n\n"
        "CONTEXT: In bank/ledger reconciliation, every file has these concepts:\n"
        "- 'amount': the monetary value (may be called Txn Amt, Value, Credit, Debit, etc)\n"
        "- 'date': the transaction or value date (may be called Val Dt, Posted, Booked, etc)\n"
        "- 'account': the account identifier (may be called Acct No, A/c, IBAN, etc)\n"
        "- 'key': the allocation key, invoice, or reference (may be called Ref, Invoice, ID, etc)\n\n"
        f"HEADERS: {headers}\n\n"
        "Return a JSON object mapping each canonical field to the exact header string that "
        "carries it. Only include fields you can confidently identify. Example:\n"
        '{"amount": "Txn Amt (INR)", "date": "Val Dt", "account": "A/c No"}\n\n'
        "Return ONLY the JSON object, no prose."
    )
    try:
        raw = backend.complete(prompt)
        # Strip markdown fences if present.
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        mapping = _json.loads(text)
        if not isinstance(mapping, dict):
            return {}
        # Only keep mappings whose values are actual headers in the file.
        header_set = {h.strip() for h in headers if h}
        return {
            field: header
            for field, header in mapping.items()
            if field in ("amount", "date", "account", "key") and header in header_set
        }
    except Exception:  # noqa: BLE001 — a broken LLM must not break parsing
        return {}


def sniff_columns(headers: list[str], *, backend=None) -> dict[str, str]:
    """Map canonical field -> the column in *headers* that carries it.

    Two passes per field so that a file containing both ``amount`` and
    ``settled_amount`` binds the plain one. Fields with no match are absent
    from the result; the caller decides which of those are fatal.

    When *backend* is provided and regex leaves required fields unmapped, the
    LLM is consulted. Any LLM failure degrades silently to the regex result.
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

    # If regex missed required fields, ask the LLM.
    required = {"amount", "date", "account"}
    if backend is not None and not required.issubset(found):
        llm_result = _llm_sniff_columns(headers, backend)
        # LLM fills gaps; regex wins on fields it already found.
        for field, header in llm_result.items():
            if field not in found:
                found[field] = header

    return found


#: Delimiter -> the decimal mark that convention pairs with it. A European
#: Excel writes `a;b;c` *and* `1250,00`; reading that file with a
#: comma-stripping parser yields 125,000.00, a hundredfold error in money with
#: nothing visibly wrong.
_DELIMITERS = {",": ".", ";": ",", "\t": ".", "|": "."}


def _sniff_delimiter(header: str) -> str:
    """Whichever candidate splits the header into the most fields."""
    best, best_n = ",", 0
    for d in _DELIMITERS:
        # An empty file yields no row at all; default rather than raise, so the
        # caller reaches the readable "no header row" message below.
        row = next(csv.reader(io.StringIO(header), delimiter=d), [])
        if len(row) > best_n:
            best, best_n = d, len(row)
    return best


def _rows(text: str) -> tuple[list[dict[str, str]], list[str], str]:
    # Excel writes a UTF-8 BOM; it would otherwise become part of the first
    # column's name and hide it from the sniffer.
    text = text.lstrip("\ufeff")
    first = text.split("\n", 1)[0]
    delimiter = _sniff_delimiter(first)
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    headers = [h for h in (reader.fieldnames or []) if h]
    if not headers:
        raise UploadError("the file has no header row")
    rows = [r for r in reader if any((v or "").strip() for v in r.values())]
    if not rows:
        raise UploadError("the file has no rows below the header")
    return rows, headers, _DELIMITERS[delimiter]


def _minor(raw: str | None, row_no: int, decimal: str = ".") -> int:
    """Money, as an exact integer count of minor units.

    *decimal* is the mark this file uses for the fractional part -- a comma in
    a European export. Reading `1250,00` with a comma-stripping parser yields
    125,000.00: a hundredfold error, in money, with nothing visibly wrong.
    """
    original = (raw or "").strip()
    if not original:
        raise UploadError(f"row {row_no}: the amount is blank")

    text = original.replace(" ", "")
    if decimal == ",":
        text = text.replace(".", "").replace(",", ".")
    else:
        text = text.replace(",", "")
    text = re.sub(r"^[^\d\-+.]+|[^\d]*$", "", text)
    if not text:
        # Present but unreadable. Calling it "blank" sends someone hunting
        # for an empty cell that is not empty.
        raise UploadError(f"row {row_no}: {original!r} is not an amount")

    neg = text.startswith("-") or original.startswith("(")
    text = text.lstrip("-+")

    whole, _, frac = text.partition(".")
    whole = whole or "0"
    if not whole.isdigit() or (frac and not frac.isdigit()):
        raise UploadError(f"row {row_no}: {original!r} is not an amount")
    if len(frac) > 2:
        raise UploadError(
            f"row {row_no}: {original!r} has more precision than minor units allow. "
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


#: How each accepted layout reads to a person, for reporting back.
_LAYOUT_NAMES = {
    "%Y-%m-%d": "YYYY-MM-DD", "%Y/%m/%d": "YYYY/MM/DD", "%d/%m/%Y": "DD/MM/YYYY",
    "%m/%d/%Y": "MM/DD/YYYY", "%d-%m-%Y": "DD-MM-YYYY", "%m-%d-%Y": "MM-DD-YYYY",
    "%d/%m/%y": "DD/MM/YY", "%m/%d/%y": "MM/DD/YY", "%d %b %Y": "DD Mon YYYY",
    "%d %B %Y": "DD Month YYYY", "%b %d %Y": "Mon DD YYYY", "%B %d %Y": "Month DD YYYY",
    "%d.%m.%Y": "DD.MM.YYYY", "%Y%m%d": "YYYYMMDD",
}


def _parse(text: str, *, key_field: str, id_prefix: str, backend=None):
    rows, headers, decimal = _rows(text)
    cols = sniff_columns(headers, backend=backend)

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
                    _minor(row.get(cols["amount"]), row_no, decimal), day))
    return out, _LAYOUT_NAMES.get(fmt, fmt)


def parse_bank_csv(text: str, *, report_layout: bool = False, backend=None):
    """Money that arrived: the side being explained.

    With *report_layout*, also returns which date layout was chosen.
    ``03/01/2026`` is the third of January or the first of March depending on
    who wrote the file; one reading is picked for the whole file, and a caller
    never told which has no way to notice the wrong one.
    """
    parsed, layout = _parse(text, key_field="key", id_prefix="row-", backend=backend)
    recs = [BankRecord(i, a, m, d) for i, a, m, d in parsed]
    return (recs, layout) if report_layout else recs


def parse_ledger_csv(text: str, *, report_layout: bool = False, backend=None):
    """What the books say it should be: the side being matched against."""
    parsed, layout = _parse(text, key_field="key", id_prefix="row-", backend=backend)
    rows = [KeyRow(k, a, m, d) for k, a, m, d in parsed]
    return (rows, layout) if report_layout else rows
