"""The gate — post, or send to a human.

Deterministic by construction. The model produces a number; this decides what
that number is allowed to do, and the decision must be identical on re-run,
explainable to a controller, and defensible to an auditor.

**The asymmetry.** A wrong match closes a real exception, writes into the ledger
that two unrelated things are the same economic event, removes both from review,
and surfaces months later as a reporting break. A missed match costs a reviewer
ten minutes. Every product on the market uses one confidence threshold for every
amount, which prices those two outcomes identically. They are not.

So the threshold rises with the money at stake::

    threshold(a) = clip(base + slope * log10(max(|a|, ref) / ref), base, cap)

Logarithmic because consequence scales with orders of magnitude, not rupees.
``slope`` is a policy dial: zero reproduces the industry's single threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Outcome(str, Enum):
    POST = "post"
    QUEUE = "queue"
    NO_CANDIDATE = "no_candidate"


@dataclass(frozen=True, slots=True)
class GateConfig:
    """Approved before a run, never changed during one."""

    base: float = 0.85
    """Confidence required at the reference amount."""
    slope: float = 0.02
    """Extra confidence per tenfold increase in amount. 0.0 = flat threshold."""
    cap: float = 0.995
    """Ceiling. Above this, nothing would ever post."""
    reference_minor: int = 10_000
    """Amount at which the threshold equals ``base``. Rs 100 by default."""
    review_all: bool = False
    """Queue everything regardless of confidence. For a first run against a new
    source, or any period under heightened control."""
    policy_version: str = "v0.1"

    def __post_init__(self) -> None:
        if not 0.0 < self.base <= 1.0:
            raise ValueError(f"base must be in (0, 1], got {self.base}")
        if self.base > self.cap:
            raise ValueError(f"base {self.base} exceeds cap {self.cap}")
        if self.slope < 0:
            raise ValueError(f"slope must be >= 0, got {self.slope}")

    def threshold_for(self, amount_minor: int) -> float:
        """Confidence required to post this amount without review."""
        magnitude = max(abs(amount_minor), self.reference_minor)
        decades = math.log10(magnitude / self.reference_minor)
        return min(max(self.base + self.slope * decades, self.base), self.cap)


@dataclass(frozen=True, slots=True)
class GateDecision:
    """What was decided, what it had to clear, and under which rules.

    Frozen: a decision editable after the fact is not an audit trail.
    """

    outcome: Outcome
    confidence: float | None
    threshold_required: float
    amount_minor: int
    reason: str
    policy_version: str


def decide(
    *,
    confidence: float | None,
    amount_minor: int,
    config: GateConfig | None = None,
) -> GateDecision:
    """Apply the gate to one scored record.

    Args:
        confidence: calibrated probability for the best candidate, or ``None``
            when blocking produced nothing. Absent is not the same as low, and
            is reported as its own outcome.
    """
    cfg = config or GateConfig()

    if confidence is not None and not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")

    threshold = cfg.threshold_for(amount_minor)
    common = dict(
        confidence=confidence,
        threshold_required=threshold,
        amount_minor=amount_minor,
        policy_version=cfg.policy_version,
    )

    if confidence is None:
        return GateDecision(
            outcome=Outcome.NO_CANDIDATE,
            reason="no candidate survived blocking",
            **common,
        )

    if cfg.review_all:
        return GateDecision(
            outcome=Outcome.QUEUE,
            reason="review_all is set: every record is reviewed regardless of confidence",
            **common,
        )

    if confidence >= threshold:
        return GateDecision(
            outcome=Outcome.POST,
            reason=f"confidence {confidence:.4f} >= threshold {threshold:.4f}",
            **common,
        )

    return GateDecision(
        outcome=Outcome.QUEUE,
        reason=f"confidence {confidence:.4f} below threshold {threshold:.4f} "
               f"for amount {amount_minor / 100:,.2f}",
        **common,
    )
