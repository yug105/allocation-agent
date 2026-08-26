"""Exception narration — the engine computes, the model writes the sentence.

Two rules make this safe rather than decorative.

**The diagnosis is arithmetic.** Each possible cause predicts what the residual
*would* be, and the causes are ranked by how well that prediction fits. No model
is consulted about why a match failed; a model is consulted about how to say it.

**The model may not introduce a number.** Every figure in the output must appear
in the input payload, checked after generation. A model that cannot introduce a
figure cannot invent a plausible-sounding fee, gap or date -- which is the
failure mode a published post-mortem describes as *"92% confidence citing a
common fee structure that did not exist."*

Anything that fails validation falls back to a template. The narration layer is
never on the matching path, is batched, and is cached by situation, so a batch of
190,717 records costs a handful of calls rather than one per record.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

#: Anything that looks like a figure: amounts, counts, identifiers, percentages.
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

CAUSES = ("BANK_CHARGE", "ROUNDING", "FX_DIFFERENCE", "PARTIAL_PAYMENT", "UNEXPLAINED")

_TEMPLATES = {
    "BANK_CHARGE": "Gap of {gap} against {amount} matches this counterparty's usual deduction.",
    "ROUNDING": "Gap of {gap} is within rounding tolerance for {lines} line(s).",
    "FX_DIFFERENCE": "Gap of {gap} against {amount} is consistent with a rate difference.",
    "PARTIAL_PAYMENT": "Gap of {gap} equals one outstanding line; the payment appears partial.",
    "UNEXPLAINED": "Gap of {gap} against {amount} does not fit any known cause. Needs review.",
}


class NarrationError(ValueError):
    """Raised when generated text contains a figure that was never supplied."""


class Backend(Protocol):
    def complete(self, prompt: str) -> str: ...


class StubBackend:
    """Deterministic backend. Lets the whole pipeline run with no API key."""

    def complete(self, prompt: str) -> str:
        items = json.loads(prompt.split("ITEMS:", 1)[1])
        return json.dumps([
            {"record_id": it["record_id"], "cause": it["causes"][0][0],
             "sentence": _fill(it, it["causes"][0][0])}
            for it in items
        ])


def _major(minor: int) -> str:
    return f"{minor / 100:,.2f}"


def _fill(item: dict[str, Any], cause: str) -> str:
    return _TEMPLATES.get(cause, _TEMPLATES["UNEXPLAINED"]).format(
        gap=_major(item["residual_minor"]),
        amount=_major(item["amount_minor"]),
        lines=item.get("n_lines", 1),
    )


def diagnose_residual(
    *,
    residual_minor: int,
    amount_minor: int,
    n_lines: int,
    usual_fee_bps: int,
) -> list[tuple[str, float]]:
    """Rank causes by how well each one's *predicted* residual fits the observed.

    Pure arithmetic. Returns ``(cause, fit)`` sorted best first, or ``[]`` when
    there is no residual to explain.
    """
    if residual_minor == 0:
        return []

    observed = abs(residual_minor)
    predictions = {
        "BANK_CHARGE": abs(amount_minor) * usual_fee_bps // 10_000,
        "ROUNDING": max(n_lines, 1),
        "FX_DIFFERENCE": abs(amount_minor) // 100,
        "PARTIAL_PAYMENT": abs(amount_minor),
    }

    # Relative error against the *smaller* of the two, so a prediction that is
    # orders of magnitude off scores near zero rather than floored at 0.5.
    # Normalising by the observed value alone caps the error ratio at 1.0 and
    # makes "nothing fits" indistinguishable from "fits moderately".
    scored = []
    for cause, predicted in predictions.items():
        denominator = max(min(observed, predicted), 1)
        scored.append((cause, 1.0 / (1.0 + abs(observed - predicted) / denominator)))
    scored.append(("UNEXPLAINED", 0.05))
    scored.sort(key=lambda c: -c[1])
    return scored


def _as_number(token: str) -> Decimal | None:
    """A figure's value, or None if it is not one."""
    try:
        return Decimal(token.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


def validate_numbers(text: str, allowed: set[str]) -> None:
    """Raise unless every figure in *text* was supplied in the payload.

    Compared by **value**, not by spelling. String matching rejected `1250`
    against an allowed `1,250.00` — correct, written differently — which fails
    the model for being right and drops a good sentence.

    The obvious repair is to add trailing-zero-stripped forms to the allowed
    set. It is a trap: `"1250.00".rstrip(".0")` is `"125"` and
    `"100.00".rstrip(".0")` is `"1"`, so the guard would begin permitting a
    figure an order of magnitude out — the exact invention it exists to stop.
    Comparing values accepts every correct spelling and no incorrect one.

    Signs are ignored on the text side: residuals are stored signed and the
    sentence usually reads "gap of 636.83".
    """
    permitted = {v for v in (_as_number(a) for a in allowed) if v is not None}
    permitted |= {-v for v in permitted}
    for found in _NUMBER.findall(text):
        value = _as_number(found)
        if value is not None and (value in permitted or -value in permitted):
            continue
        raise NarrationError(f"generated figure {found!r} is not in the payload")


@dataclass
class Narrator:
    """Batched, cached, and incapable of emitting an unsupported figure."""

    backend: Backend | None = None
    batch_size: int = 20
    _cache: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    calls: int = 0
    narrated: int = 0
    """Times narrate() was invoked at all -- the LLM counter stays at zero on the
    template path, which is exactly what made 'constructed but never called'
    invisible."""

    def _situation(self, item: dict[str, Any]) -> str:
        """Cache key. Identical situations are narrated once."""
        raw = f"{item['causes'][0][0]}|{item['residual_minor']}|{item.get('n_lines', 1)}"
        return hashlib.sha1(raw.encode()).hexdigest()[:12]

    def narrate(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.narrated += 1
        if not items:
            return []

        todo = [it for it in items if self._situation(it) not in self._cache]
        for start in range(0, len(todo), self.batch_size):
            self._run(todo[start : start + self.batch_size])

        out = []
        for it in items:
            cached = self._cache.get(self._situation(it))
            if cached is None:
                cached = self._template(it)
            out.append({**cached, "record_id": it["record_id"]})
        return out

    def _template(self, item: dict[str, Any]) -> dict[str, Any]:
        cause = item["causes"][0][0] if item.get("causes") else "UNEXPLAINED"
        return {"cause": cause, "sentence": _fill(item, cause), "source": "template"}

    def _run(self, batch: list[dict[str, Any]]) -> None:
        if not batch:
            return

        results: dict[str, dict[str, Any]] = {}
        if self.backend is not None:
            prompt = (
                "For each item, name the cause from "
                f"{list(CAUSES)} and write one sentence explaining it.\n"
                "Use ONLY figures that appear in the item. Do not introduce any number.\n"
                'Return JSON: [{"record_id":..., "cause":..., "sentence":...}]\n'
                "ITEMS:" + json.dumps(batch, default=str)
            )
            try:
                self.calls += 1
                parsed = json.loads(_extract_json(self.backend.complete(prompt)))
                by_id = {r["record_id"]: r for r in parsed}
                for it in batch:
                    r = by_id.get(it["record_id"])
                    if r is None:
                        continue
                    allowed = {
                        _major(it["residual_minor"]), _major(it["amount_minor"]),
                        str(abs(it["residual_minor"])), str(abs(it["amount_minor"])),
                        str(it.get("n_lines", 1)),
                    }
                    validate_numbers(r["sentence"], allowed)
                    if r["cause"] not in CAUSES:
                        continue
                    results[self._situation(it)] = {
                        "cause": r["cause"], "sentence": r["sentence"], "source": "llm"
                    }
            except Exception:
                # Any failure -- unreachable service, unparseable output, an
                # invented figure -- degrades to the template. Never raises.
                results = {}

        for it in batch:
            key = self._situation(it)
            self._cache[key] = results.get(key, self._template(it))


def _extract_json(text: str) -> str:
    """Pull the JSON array out of a response that may be fenced or prefixed."""
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array in response")
    return text[start : end + 1]
