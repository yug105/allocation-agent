"""Pull a merchant's own settlements from Razorpay, in test mode only.

`GET /v1/settlements/recon/combined` returns every payment, refund, transfer and
adjustment settled in a period, each carrying the `settlement_id` it was paid
out under. Group by that id and the result is exactly this project's hard case,
with real money: several payments landing as one bank credit.

So the integration is not a logo. It feeds the solver the one thing ReconRiver
could only imitate — a merchant's actual settlement batches — and the batch id
is withheld from the solver in the same way, so recovering it is a measurement
rather than a lookup.

**Test mode only, and refused loudly otherwise.** A live key would authorise
reads against real production money, and no demo needs that. Nothing here
stores, logs or returns the secret; it is used once, for one request, to
`api.razorpay.com` and nowhere else.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

API = "https://api.razorpay.com/v1/settlements/recon/combined"

#: Beyond this the demo is fetching more than it can usefully show.
MAX_ITEMS = 1_000

#: A slow upstream must not hold a request open indefinitely.
TIMEOUT_SECONDS = 20.0


class RazorpayError(RuntimeError):
    """A problem talking to Razorpay, phrased for the person who connected."""


@dataclass(frozen=True, slots=True)
class ReconItem:
    """One settled transaction. Amounts stay in integer paise throughout."""

    entity_id: str
    settlement_id: str
    type: str
    credit_minor: int
    debit_minor: int
    fee_minor: int
    currency: str
    settled_at: int

    @property
    def net_minor(self) -> int:
        """What this line contributed to the bank credit."""
        return self.credit_minor - self.debit_minor


def _default_fetch(url: str, auth_header: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={
        "Authorization": auth_header,
        "Accept": "application/json",
        "User-Agent": "allocation-agent",
    })
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise RazorpayError(
                "Razorpay rejected those credentials. Check the key id and "
                "secret are a matching test-mode pair."
            ) from exc
        if exc.code == 400:
            raise RazorpayError(
                "Razorpay refused the request — usually a year or month with "
                "no settlement activity on this account."
            ) from exc
        raise RazorpayError(f"Razorpay returned HTTP {exc.code}.") from exc
    except urllib.error.URLError as exc:
        raise RazorpayError(f"Could not reach Razorpay: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RazorpayError("Razorpay returned something that is not JSON.") from exc


def fetch_recon(
    key_id: str,
    key_secret: str,
    *,
    year: int,
    month: int,
    day: int | None = None,
    count: int = MAX_ITEMS,
    fetch: Callable[[str, str], dict[str, Any]] | None = None,
) -> list[ReconItem]:
    """Settled transactions for a period, grouped later by settlement.

    *fetch* is injectable so the parsing, the guards and the error messages are
    all testable without a network or anybody's credentials.
    """
    if not key_id.startswith("rzp_test_"):
        raise RazorpayError(
            "Only test-mode keys are accepted. A live key would read against "
            "real settled money, and this demo never needs that — create a "
            "test key in the Razorpay dashboard instead."
        )
    if not key_secret:
        raise RazorpayError("The key secret is required.")
    if not 1 <= month <= 12:
        raise RazorpayError(f"month must be 1-12, got {month}")
    if day is not None and not 1 <= day <= 31:
        raise RazorpayError(f"day must be 1-31, got {day}")

    params = {"year": f"{year:04d}", "month": f"{month:02d}",
              "count": str(min(count, MAX_ITEMS))}
    if day is not None:
        params["day"] = f"{day:02d}"

    token = base64.b64encode(f"{key_id}:{key_secret}".encode()).decode()
    payload = (fetch or _default_fetch)(
        f"{API}?{urllib.parse.urlencode(params)}", f"Basic {token}")

    items = payload.get("items")
    if items is None:
        raise RazorpayError("Razorpay's reply carried no items.")

    out: list[ReconItem] = []
    for raw in items[:MAX_ITEMS]:
        settlement_id = raw.get("settlement_id")
        if not settlement_id:
            continue                    # not settled yet; nothing to reconcile
        out.append(ReconItem(
            entity_id=str(raw.get("entity_id") or raw.get("payment_id") or ""),
            settlement_id=str(settlement_id),
            type=str(raw.get("type") or "unknown"),
            credit_minor=int(raw.get("credit") or 0),
            debit_minor=int(raw.get("debit") or 0),
            fee_minor=int(raw.get("fee") or 0),
            currency=str(raw.get("currency") or "INR"),
            settled_at=int(raw.get("settled_at") or 0),
        ))
    return out


def group_into_settlements(items: list[ReconItem]) -> list[dict[str, Any]]:
    """Turn settled lines into the shape the solver takes.

    Each settlement becomes a bank credit whose amount is the net of its lines,
    and the lines themselves become the candidate pool. The `settlement_id` is
    kept only as the answer to check against — it never reaches the solver.
    """
    by_settlement: dict[str, list[ReconItem]] = {}
    for item in items:
        by_settlement.setdefault(item.settlement_id, []).append(item)

    out = []
    for settlement_id, lines in by_settlement.items():
        total = sum(line.net_minor for line in lines)
        if total <= 0:
            continue                    # a net-zero or refund-only payout
        out.append({
            "settlement_id": settlement_id,
            "amount_minor": total,
            "currency": lines[0].currency,
            "booked_at": lines[0].settled_at,
            "truth": [line.entity_id for line in lines],
            "n_lines": len(lines),
        })
    return sorted(out, key=lambda s: -s["amount_minor"])
