"""Failure routing — the load-bearing half of learning from a correction.

Four symptoms look identical from outside: a record went to a human and the
human named a different answer. The causes are not identical, and neither are
the fixes.

===============  ==========================================  =====================
locus            what actually happened                      fix
===============  ==========================================  =====================
``blocking``     the right key was never a candidate         widen the window
``multiplicity`` wrongly routed as grouped, or missed        labelled example
``ranking``      considered, but ranked below another        labelled example
``threshold``    chosen correctly, then refused by the gate  settings change
===============  ==========================================  =====================

Only two of the four are model problems. Routing every correction into one store
learns none of them: the blocking failures get fed to a ranker that never saw the
key, and the threshold failures get fed to a model that was already right.

Order matters. A record routed away as grouped was never ranked, so blaming the
ranker would be wrong; a key that was never a candidate cannot have been
mis-ranked.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class FailureLocus(str, Enum):
    NONE = "none"
    BLOCKING = "blocking"
    MULTIPLICITY = "multiplicity"
    RANKING = "ranking"
    THRESHOLD = "threshold"


@dataclass(frozen=True, slots=True)
class Diagnosis:
    locus: FailureLocus
    detail: str


def diagnose(
    *,
    correct_keys: list[str],
    candidates: set[str],
    ranked_keys: list[str],
    posted: bool,
    routed_multiple: bool,
    truly_multiple: bool,
) -> Diagnosis:
    """Attribute a correction to the stage that caused it."""
    # 1. multiplicity first: a misrouted record never reached the ranker.
    if routed_multiple != truly_multiple:
        if routed_multiple:
            return Diagnosis(
                FailureLocus.MULTIPLICITY,
                "routed to review as grouped, but the record maps to a single key",
            )
        return Diagnosis(
            FailureLocus.MULTIPLICITY,
            "record spans several keys and was not detected as grouped",
        )

    # 2. blocking: a key that was never a candidate cannot have been mis-ranked.
    missing = [k for k in correct_keys if k not in candidates]
    if missing:
        return Diagnosis(
            FailureLocus.BLOCKING,
            f"correct key(s) {missing} never entered the candidate set "
            f"({len(candidates)} candidates considered)",
        )

    # 3. ranking: present, but something else came first.
    if ranked_keys and ranked_keys[0] not in correct_keys:
        position = next(
            (i + 1 for i, k in enumerate(ranked_keys) if k in correct_keys), None
        )
        return Diagnosis(
            FailureLocus.RANKING,
            f"correct key ranked at position {position} behind {ranked_keys[0]!r}",
        )

    # 4. threshold: right answer, refused.
    if not posted:
        return Diagnosis(
            FailureLocus.THRESHOLD,
            "correct key was ranked first but confidence fell below the gate",
        )

    return Diagnosis(FailureLocus.NONE, "decision was correct")
