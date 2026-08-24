"""Learning from corrections.

Two questions, and the first decides everything else.

**What broke?** The right key was never considered (blocking), or it was
considered and ranked below another (ranking), or the record was wrongly routed
as grouped (multiplicity), or it was chosen correctly but refused by the gate
(threshold). Four symptoms that look identical from the outside and need four
different fixes. Routing every correction into one store learns none of them.

**How do we remember it?** A case for explanation, a training example for
accuracy, a rule proposal for permanence.
"""

import numpy as np

from allocation_agent.learn.casebase import Case, CaseBase
from allocation_agent.learn.router import FailureLocus, diagnose


def diag(*, correct="K_TRUE", candidates=("K_TRUE", "K1"), ranked=("K_TRUE", "K1"),
         posted=False, routed_multiple=False, was_multiple=False):
    return diagnose(
        correct_keys=[correct] if isinstance(correct, str) else list(correct),
        candidates=set(candidates),
        ranked_keys=list(ranked),
        posted=posted,
        routed_multiple=routed_multiple,
        truly_multiple=was_multiple,
    )


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #

def test_key_never_considered_is_a_blocking_failure():
    """Not a model problem. Widen the window; no amount of scoring recovers it."""
    d = diag(candidates=("K1", "K2"), ranked=("K1", "K2"))
    assert d.locus is FailureLocus.BLOCKING


def test_key_considered_but_ranked_below_another_is_a_ranking_failure():
    d = diag(ranked=("K1", "K_TRUE"))
    assert d.locus is FailureLocus.RANKING


def test_correct_and_top_ranked_but_not_posted_is_a_threshold_failure():
    """The system was right and refused to act. That is a settings problem."""
    d = diag(ranked=("K_TRUE", "K1"), posted=False)
    assert d.locus is FailureLocus.THRESHOLD


def test_single_key_record_routed_as_grouped_is_a_multiplicity_failure():
    d = diag(routed_multiple=True, was_multiple=False)
    assert d.locus is FailureLocus.MULTIPLICITY


def test_grouped_record_not_routed_is_also_a_multiplicity_failure():
    d = diag(correct=["K_TRUE", "K1"], routed_multiple=False, was_multiple=True)
    assert d.locus is FailureLocus.MULTIPLICITY


def test_a_correct_posted_decision_is_not_a_failure():
    assert diag(ranked=("K_TRUE",), posted=True).locus is FailureLocus.NONE


def test_multiplicity_is_diagnosed_before_ranking():
    """A record routed away as grouped was never ranked; blaming the ranker is wrong."""
    d = diag(routed_multiple=True, was_multiple=False, ranked=())
    assert d.locus is FailureLocus.MULTIPLICITY


def test_blocking_is_diagnosed_before_ranking():
    """A key that was never a candidate cannot have been mis-ranked."""
    d = diag(candidates=("K1",), ranked=("K1",))
    assert d.locus is FailureLocus.BLOCKING


def test_diagnosis_explains_itself():
    assert diag(candidates=("K1",), ranked=("K1",)).detail


def test_locus_values_match_the_pipeline_stages():
    assert {l.value for l in FailureLocus} >= {
        "blocking", "ranking", "multiplicity", "threshold", "none"}


# --------------------------------------------------------------------------- #
# case base: retrieval
# --------------------------------------------------------------------------- #

def case(cid="c1", vec=(1.0, 0.0), locus="ranking", keys=("K1",)) -> Case:
    return Case(case_id=cid, situation=np.array(vec), locus=locus, resolution=list(keys))


def test_similar_case_in_the_same_locus_is_retrieved():
    cb = CaseBase(similarity_threshold=0.9)
    cb.retain(case(vec=(1.0, 0.0)))
    assert cb.retrieve(np.array([0.99, 0.05]), locus="ranking")


def test_dissimilar_case_is_not_retrieved():
    cb = CaseBase(similarity_threshold=0.9)
    cb.retain(case(vec=(1.0, 0.0)))
    assert not cb.retrieve(np.array([-1.0, 0.0]), locus="ranking")


def test_a_case_from_a_different_locus_is_never_retrieved():
    """A blocking failure teaches nothing about a threshold failure."""
    cb = CaseBase(similarity_threshold=0.5)
    cb.retain(case(vec=(1.0, 0.0), locus="blocking"))
    assert not cb.retrieve(np.array([1.0, 0.0]), locus="threshold")


def test_retrieval_is_ordered_by_similarity():
    cb = CaseBase(similarity_threshold=0.0)
    cb.retain(case(cid="far", vec=(0.0, 1.0)))
    cb.retain(case(cid="near", vec=(1.0, 0.0)))
    got = cb.retrieve(np.array([1.0, 0.0]), locus="ranking", k=2)
    assert got[0].case_id == "near"


def test_empty_base_retrieves_nothing_rather_than_failing():
    assert CaseBase().retrieve(np.array([1.0, 0.0]), locus="ranking") == []


# --------------------------------------------------------------------------- #
# case base: selective retention
# --------------------------------------------------------------------------- #

def test_a_near_duplicate_bumps_a_counter_instead_of_being_stored():
    """Otherwise the base fills with copies and 'learning' is just growth."""
    cb = CaseBase(duplicate_threshold=0.95)
    cb.retain(case(cid="a", vec=(1.0, 0.0)))
    assert not cb.retain(case(cid="b", vec=(1.0, 0.001)))
    assert len(cb) == 1
    assert cb.cases[0].confirmations == 1


def test_a_genuinely_new_case_is_stored():
    cb = CaseBase(duplicate_threshold=0.95)
    cb.retain(case(cid="a", vec=(1.0, 0.0)))
    assert cb.retain(case(cid="b", vec=(0.0, 1.0)))
    assert len(cb) == 2


def test_an_uncertain_human_label_is_not_retained():
    cb = CaseBase()
    assert not cb.retain(case(), human_certain=False)
    assert len(cb) == 0


def test_the_base_is_size_capped():
    cb = CaseBase(max_cases=3, duplicate_threshold=1.1)
    for i in range(10):
        cb.retain(case(cid=f"c{i}", vec=(float(i), 1.0)))
    assert len(cb) <= 3


def test_a_case_that_stops_working_is_retired():
    cb = CaseBase(min_applications=2, retire_below_accuracy=0.9)
    cb.retain(case(cid="bad"))
    for _ in range(3):
        cb.record_outcome("bad", correct=False)
    assert not cb.retrieve(np.array([1.0, 0.0]), locus="ranking")


def test_a_case_that_keeps_working_survives():
    cb = CaseBase(min_applications=2, retire_below_accuracy=0.9)
    cb.retain(case(cid="good"))
    for _ in range(3):
        cb.record_outcome("good", correct=True)
    assert cb.retrieve(np.array([1.0, 0.0]), locus="ranking")


def test_retirement_is_recorded_not_silent():
    """A decaying case is a signal the world changed."""
    cb = CaseBase(min_applications=1, retire_below_accuracy=0.9)
    cb.retain(case(cid="bad"))
    cb.record_outcome("bad", correct=False)
    cb.record_outcome("bad", correct=False)
    assert "bad" in cb.retired
