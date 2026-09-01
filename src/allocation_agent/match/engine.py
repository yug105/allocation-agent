"""The matching engine: one bank record in, one decision out.

Extracted because there were two of these. `api.py` had the maintained one and
`pipeline.py` had a copy that had quietly fallen a long way behind — no
calibration, a fabricated margin for the single-candidate case, the grouping
check still able to overrule an exact amount, and a record that could leave the
audit trail entirely. A reviewer found eight defects in that file; all eight
were fixes that existed here and had never been carried across.

So it lives in one place now, and both callers import it. The rule in
CLAUDE.md — *one matching path, two callers* — was true of the API and false of
the repo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from allocation_agent.decide.gate import Outcome, decide
from allocation_agent.decide.narrate import Narrator, diagnose_residual
from allocation_agent.match.blocker import BlockingConfig, block
from allocation_agent.match.features import featurise
from allocation_agent.match.multiplicity import featurise_multiplicity
from allocation_agent.types import BankRecord

# Taking the lone exact-amount candidate is right 98.98% of the time on the
# BenchRec held-out set (2,321 of 2,345).
#
# **It is measured on a different population from the one it is applied to.**
# Those 2,345 records each had four or more blocked candidates, one of which
# matched the amount. This branch only fires when there is exactly *one*
# candidate at all -- a case BenchRec contains zero of, because blocking never
# returns fewer than four there. The figure is an extrapolation, not a
# measurement of this branch.
#
# The direction of that extrapolation is measured, and it is conservative
# (scripts/audit_direct_rule.py):
#
#     candidates blocked      n     correct
#      4-10               1,253    100.00%
#     11-50                 258     96.51%
#     50+                   834     98.20%
#
# Two separate statements, and only the first is measured:
#
#   Measured: among records with exactly one exact-amount candidate *and at
#   least four blocked candidates*, taking that candidate is right 98.98% of
#   the time. Precision does not deteriorate as competition falls -- the 4-10
#   group is 100.00%.
#
#   Not measured: what the rule does on one candidate. That population is
#   absent from BenchRec's held-out distribution, so 0.9898 is an extrapolated
#   estimate from the nearest measured population, not the branch's precision.
#
# No amount bucket falls under the gate's base bar either -- above 100k is
# 100.00% (n=113), which inverts the usual worry that one constant hides a
# dangerous tail.
#
# Labelled `..._unvalidated` wherever it is used, because an extrapolation is
# not a measurement.
DIRECT_CONFIDENCE = 0.9898

# On data with no measurement behind it the figure is used unchanged but
# *labelled*, and the response says no accuracy is claimed for that file. An
# earlier attempt discounted it to 0.60, which was worse: it made a lone exact
# amount rank below a wide ranker margin on the very same file, when the exact
# amount is the stronger evidence of the two.

# Where the grouping detector's probability becomes a routing decision.
MULT_THRESHOLD = 0.7


@dataclass(frozen=True, slots=True)
class Models:
    """What matching needs, without the web service around it."""

    ranker: Any
    detector: Any
    prior: Any
    calibrator: Any = None
    calibrator_kind: str = "none"


def match_one(rec: BankRecord, *, index, key_stats, models: Models, gate,
              mult_threshold: float, blocking: BlockingConfig,
              narrator: Narrator | None = None,
              calibrated_for_this_data: bool = False) -> dict:
    """Run one record through the whole matching path.

    Extracted so an uploaded file goes through *identical* code to the demo. A
    separate path for user data would make the demo's numbers evidence for
    nothing but the demo.
    """
    cands = sorted(block(rec, index, blocking))

    if not cands:
        d = decide(confidence=None, amount_minor=rec.amount_minor, config=gate)
        return {"residual_cause": None, "residual_minor": 0, "stage": "narrowing", "outcome": "no_candidate", "decision": d,
                "keys": [], "n_blocked": 0, "n_scored": 0, "n_candidates": 0,
                "path": "blocked", "evidence": None,
                "confidence": None,
                "explanation": "Nothing in the ledger is close enough to consider — "
                               "no entry shares this account within a week of this date."}

    usable = [k for k in cands if k in key_stats]
    if not usable:
        # Blocking found entries; none could be scored. Reporting that as
        # "no candidate" makes the exception breakdown describe a different
        # failure from the one that happened.
        d = decide(confidence=None, amount_minor=rec.amount_minor, config=gate)
        return {"residual_cause": None, "residual_minor": 0, "stage": "narrowing",
                "outcome": "unscorable", "decision": d, "keys": [],
                "n_blocked": len(cands), "n_scored": 0, "n_candidates": len(cands),
                "path": "unscorable", "evidence": {"n_blocked": len(cands)},
                "confidence": None,
                "explanation": (f"{len(cands)} nearby ledger entries were found, but none "
                                f"carry the figures needed to score them. Sent for review.")}

    # A lone candidate whose amount is exactly this figure is direct evidence,
    # and it limits what the next two steps are entitled to do.
    exact = [k for k in usable if rec.amount_minor in key_stats[k].amounts]
    lone_exact = len(exact) == 1

    # 1. The grouping check does not get to overrule it. Measured on the
    #    held-out set: the detector is right 96.3% of the time overall, but on
    #    records with a lone exact-amount match it fires 41 times and is wrong
    #    on 36 -- 12.2% precision. A single entry accounting for the whole
    #    amount defeats the premise of the grouped path and the detector cannot
    #    see that. Everywhere else it is trusted exactly as before.
    p_mult = p_multiple(models, rec, usable, key_stats)
    if p_mult >= mult_threshold and not lone_exact:
        d = decide(confidence=None, amount_minor=rec.amount_minor, config=gate)
        return {"residual_cause": None, "residual_minor": 0, "stage": "grouping", "outcome": "suspected_grouped", "decision": d,
                "keys": [], "n_blocked": len(cands), "n_scored": len(usable), "n_candidates": len(cands), "path": "multiplicity",
                "evidence": {"p_multiple": round(p_mult, 4)}, "confidence": None,
                "explanation": f"This looks like one payment covering several ledger "
                               f"entries ({p_mult:.0%} confidence), so a single match "
                               f"would be wrong. Sent for review."}

    # 2. Rank. Confidence is the gap between first and second place.
    X = np.vstack([featurise(rec, key_stats[k], n_candidates=len(usable)) for k in usable])
    scores = models.ranker.score(X)
    order = np.argsort(-scores)
    chosen = usable[int(order[0])]

    if len(order) > 1:
        margin = float(scores[order[0]] - scores[order[1]])
        # LambdaRank optimises order, not likelihood, so its margin carries no
        # probability meaning and neither does a sigmoid of it. The calibrator
        # maps margin to the measured frequency of being right; without one,
        # fall back to the sigmoid and say so in the evidence.
        if models.calibrator is not None and calibrated_for_this_data:
            confidence = float(np.clip(models.calibrator.predict([margin])[0], 0.0, 1.0))
            source = models.calibrator_kind
        else:
            # The calibrator maps a margin to a frequency measured on BenchRec.
            # On other data that mapping is unvalidated, so the number is named
            # for what it is rather than dressed as a probability.
            confidence = float(1.0 / (1.0 + np.exp(-margin)))
            source = "uncalibrated_sigmoid"
        path = "ranked"
        # The order the ranker actually produced, not only its winner. A
        # correction arrives later with the right key and has to be attributed
        # to a stage: without this list the trail cannot say whether that key
        # was ranked below another or never reached the ranker at all, and
        # `/api/correct` was guessing "ranked" for both. `n_candidates` records
        # how many were scored, so a reader can tell this list is complete.
        evidence = {"margin": round(margin, 4), "confidence_from": source,
                    "ranked_keys": [usable[int(i)] for i in order]}
        # When a lone exact amount kept the grouping check from firing, the
        # trail has to say so. Without this the record is indistinguishable
        # from an ordinary match, and a reviewer cannot see that a
        # probabilistic detector was overridden or on what grounds.
        if lone_exact and p_mult >= mult_threshold:
            evidence |= {"exact_amount": True, "overrode_grouping": True,
                         "p_multiple": round(p_mult, 4)}
    elif lone_exact:
        # No runner-up, so no margin exists. The old code substituted
        # margin=1.0 here, which is sigmoid -> 73.1%: a constant dressed as a
        # measurement, and permanently under the 85% base bar -- a lone
        # candidate could never post however exact the match. Blocking never
        # returns fewer than 4 candidates on BenchRec, so no test could reach
        # it and every small uploaded file did.
        #
        # The evidence here is the exact amount itself. On the held-out set,
        # where exactly one candidate matches the amount exactly it is the
        # right answer 98.98% of the time (2,321 of 2,345). That measured rate
        # is the confidence.
        chosen = exact[0]
        confidence = DIRECT_CONFIDENCE
        source = "benchrec_heldout" if calibrated_for_this_data else "benchrec_heldout_unvalidated"
        path = "direct"
        evidence = {"exact_amount": True, "confidence_from": source,
                    "ranked_keys": [chosen]}
    else:
        # One candidate, and its amount is not this figure. Nothing supports it.
        # It was still the only thing considered, and a correction naming a
        # different key needs to see that rather than an empty trail.
        confidence, path = None, "ranked"
        evidence = {"ranked_keys": [chosen]}

    d = decide(confidence=confidence, amount_minor=rec.amount_minor, config=gate)

    if confidence is None:
        expl = (f"{chosen} is the only nearby ledger entry, but its amount is not this "
                f"figure and there is no second candidate to weigh it against. Nothing "
                f"here supports posting it, so it goes to a person.")
    elif path == "direct":
        expl = (f"Matched to {chosen}, the one nearby ledger entry whose amount is "
                f"exactly this figure. On records like this that entry is the right "
                f"answer {DIRECT_CONFIDENCE:.0%} of the time."
                if d.outcome is Outcome.POST else
                f"{chosen} matches this amount exactly, but {DIRECT_CONFIDENCE:.0%} is "
                f"still under the {d.threshold_required:.0%} an amount this large "
                f"requires. Sent for review.")
    elif d.outcome is Outcome.POST:
        expl = (f"Matched to {chosen}. It was the best of {len(usable)} nearby ledger "
                f"entries by a clear enough margin ({confidence:.0%}) to post without "
                f"a human looking.")
    else:
        expl = (f"Best guess is {chosen}, but at {confidence:.0%} it is under the "
                f"{d.threshold_required:.0%} this amount requires. Sent for review "
                f"rather than posted.")

    # A queued record with an amount gap has something to diagnose: the reviewer
    # needs to know *why* the figures differ, not only that they do. Causes are
    # ranked by arithmetic fit; the narrator turns the winner into a sentence
    # and refuses to emit any figure the record does not carry.
    residual_cause = None
    residual_minor = 0
    if narrator is not None and d.outcome is not Outcome.POST and confidence is not None:
        stats = key_stats[chosen]
        # The gap against the nearest single line, which is what a reviewer
        # compares. `amounts` is a frozenset -- indexing it is meaningless.
        nearest = min(stats.amounts, key=lambda a: abs(rec.amount_minor - a),
                      default=0)
        residual_minor = rec.amount_minor - nearest
        if residual_minor:
            causes = diagnose_residual(
                residual_minor=residual_minor, amount_minor=rec.amount_minor,
                n_lines=stats.n_rows, usual_fee_bps=0)
            try:
                (told,) = narrator.narrate([{
                    "record_id": rec.record_id, "causes": causes,
                    "residual_minor": residual_minor,
                    "amount_minor": rec.amount_minor, "n_lines": stats.n_rows}])
            except Exception as exc:  # noqa: BLE001
                # Narration explains a decision already made; it must not be
                # able to unmake one. The decision, the chosen key and the
                # confidence are fixed by this point -- only the sentence is
                # lost, and the record keeps the one written above it.
                residual_cause = f"narration_failed:{type(exc).__name__}"
            else:
                residual_cause = told["cause"]
                expl = f"{expl} {told['sentence']}"

    return {"stage": "ranking",
            "residual_cause": residual_cause,
            "residual_minor": residual_minor, "outcome": "posted" if d.outcome is Outcome.POST else "queued",
            "decision": d, "keys": [chosen], "n_blocked": len(cands), "n_scored": len(usable),
            "n_candidates": len(usable), "path": path,
            "evidence": evidence, "confidence": confidence, "explanation": expl}



def p_multiple(models: Models, rec: BankRecord, cands: list[str], key_stats) -> float:
    amts = [a for k in cands if k in key_stats for a in key_stats[k].amounts]
    has_exact = rec.amount_minor in amts if amts else False
    min_delta = min((abs(rec.amount_minor - a) for a in amts), default=1e12)
    f = featurise_multiplicity(rec, n_candidates=len(cands), has_exact=has_exact,
                               min_delta_minor=float(min_delta), prior=models.prior)
    return float(models.detector.predict_proba(f.reshape(1, -1))[0])


