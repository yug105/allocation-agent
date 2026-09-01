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
    # A trail that did not record what was ranked cannot say whether the key
    # was passed over or never offered. Naming that is the point: guessing
    # "ranking" feeds a ranker examples it may never have seen, which is the
    # exact failure this module exists to prevent.
    UNATTRIBUTED = "unattributed"


@dataclass(frozen=True, slots=True)
class Diagnosis:
    locus: FailureLocus
    detail: str


def diagnose(
    *,
    correct_keys: list[str],
    candidates: set[str] | None,
    ranked_keys: list[str],
    posted: bool,
    routed_multiple: bool,
    truly_multiple: bool,
) -> Diagnosis:
    """Attribute a correction to the stage that caused it.

    ``candidates`` is what blocking actually offered, or ``None`` where the
    trail does not record it. ``None`` is not an empty set: an empty set means
    blocking offered nothing, which *is* a blocking failure, while ``None``
    means we cannot tell and must not claim either way.
    """
    # 0. routed to a person as grouped, and the reviewer says it is grouped.
    #    Nothing was matched and nothing needed to be: the decision was to
    #    decline, and declining was right.
    if routed_multiple and truly_multiple:
        return Diagnosis(
            FailureLocus.NONE, "correctly routed for review as grouped"
        )

    # 1. multiplicity: a misrouted record never reached the ranker.
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
    if candidates is not None:
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
        where = f"position {position}" if position else "outside the recorded ranking"
        return Diagnosis(
            FailureLocus.RANKING,
            f"correct key ranked at {where} behind {ranked_keys[0]!r}",
        )

    # 4. threshold: right answer, refused. Only a record that was *ranked* can
    #    have been refused by the gate -- one routed away earlier never reached
    #    it, and calling that a threshold failure sends someone to change a
    #    setting that had no part in it.
    if ranked_keys and not posted:
        return Diagnosis(
            FailureLocus.THRESHOLD,
            "correct key was ranked first but confidence fell below the gate",
        )

    if not ranked_keys and candidates is None:
        return Diagnosis(
            FailureLocus.UNATTRIBUTED,
            "the trail does not record what was ranked, so the failing stage "
            "cannot be named",
        )

    return Diagnosis(FailureLocus.NONE, "decision was correct")
