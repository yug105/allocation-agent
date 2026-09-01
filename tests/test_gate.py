"""The gate: post, or send to a human.

A wrong match closes a real exception, asserts in the ledger that two unrelated
things are the same event, and surfaces months later as a reporting break. A
missed match costs someone ten minutes. Those are not symmetric, so the
confidence required must rise with the amount at stake.

Every product on the market uses one threshold for every amount.
"""

import pytest

from allocation_agent.decide.gate import Absent, GateConfig, Outcome, decide


def cfg(**kw) -> GateConfig:
    return GateConfig(**kw)


# --------------------------------------------------------------------------- #
# the threshold curve
# --------------------------------------------------------------------------- #

def test_threshold_rises_with_amount():
    c = cfg()
    small = c.threshold_for(10_000)        # Rs 100
    large = c.threshold_for(50_000_000)    # Rs 5,00,000
    assert small < large


def test_threshold_never_exceeds_the_cap():
    c = cfg(cap=0.995)
    assert c.threshold_for(10**12) <= 0.995


def test_threshold_never_falls_below_the_base():
    c = cfg(base=0.85)
    assert c.threshold_for(1) >= 0.85
    assert c.threshold_for(0) >= 0.85


def test_curve_is_monotonic_across_the_range():
    c = cfg()
    amounts = [1, 100, 10_000, 1_000_000, 100_000_000]
    ts = [c.threshold_for(a) for a in amounts]
    assert ts == sorted(ts)


def test_negative_amount_uses_its_magnitude():
    """A refund of Rs 5,00,000 is exactly as consequential as a payment of it."""
    c = cfg()
    assert c.threshold_for(-5_000_000) == c.threshold_for(5_000_000)


# --------------------------------------------------------------------------- #
# decisions
# --------------------------------------------------------------------------- #

def test_confident_small_amount_posts():
    d = decide(confidence=0.95, amount_minor=10_000, config=cfg())
    assert d.outcome is Outcome.POST


def test_same_confidence_on_a_large_amount_is_queued():
    c = cfg()
    small = decide(confidence=0.90, amount_minor=10_000, config=c)
    large = decide(confidence=0.90, amount_minor=100_000_000, config=c)
    assert small.outcome is Outcome.POST
    assert large.outcome is Outcome.QUEUE


def test_no_candidate_is_its_own_outcome_not_a_low_score():
    d = decide(confidence=None, amount_minor=10_000, config=cfg())
    assert d.outcome is Outcome.NO_CANDIDATE


def test_decision_reports_the_threshold_it_had_to_clear():
    d = decide(confidence=0.9, amount_minor=1_000_000, config=cfg())
    assert 0.0 < d.threshold_required <= 1.0
    assert d.confidence == 0.9


def test_confidence_exactly_at_the_threshold_posts():
    c = cfg()
    t = c.threshold_for(10_000)
    assert decide(confidence=t, amount_minor=10_000, config=c).outcome is Outcome.POST


def test_a_hair_below_the_threshold_is_queued():
    c = cfg()
    t = c.threshold_for(10_000)
    assert decide(confidence=t - 1e-9, amount_minor=10_000, config=c).outcome is Outcome.QUEUE


# --------------------------------------------------------------------------- #
# the gate is a control, so it must be inspectable and hard to misconfigure
# --------------------------------------------------------------------------- #

def test_decision_carries_the_policy_version():
    """An audit asks which rules were in force. The decision must answer."""
    d = decide(confidence=0.99, amount_minor=1, config=cfg(policy_version="v1.2"))
    assert d.policy_version == "v1.2"


def test_review_everything_mode_queues_even_perfect_confidence():
    d = decide(confidence=1.0, amount_minor=1, config=cfg(review_all=True))
    assert d.outcome is Outcome.QUEUE
    assert "review_all" in d.reason


def test_confidence_outside_zero_to_one_is_rejected():
    with pytest.raises(ValueError, match="confidence"):
        decide(confidence=1.5, amount_minor=100, config=cfg())


def test_base_above_cap_is_rejected_at_construction():
    with pytest.raises(ValueError, match="cap"):
        GateConfig(base=0.99, cap=0.9)


def test_queued_decision_states_why():
    d = decide(confidence=0.5, amount_minor=100_000_000, config=cfg())
    assert d.outcome is Outcome.QUEUE
    assert "below threshold" in d.reason.lower()


def test_decision_is_frozen():
    """A decision that can be edited after the fact is not an audit trail."""
    d = decide(confidence=0.99, amount_minor=100, config=cfg())
    with pytest.raises(Exception):
        d.outcome = Outcome.QUEUE  # type: ignore[misc]


def test_gate_is_pure_same_input_same_output():
    c = cfg()
    a = decide(confidence=0.9, amount_minor=12_345, config=c)
    b = decide(confidence=0.9, amount_minor=12_345, config=c)
    assert a == b


# --------------------------------------------------------------------------- #
# the curve is a policy choice, so it must be tunable without code changes
# --------------------------------------------------------------------------- #

def test_steeper_slope_demands_more_of_large_amounts():
    flat = GateConfig(slope=0.0)
    steep = GateConfig(slope=0.05)
    assert flat.threshold_for(10**8) < steep.threshold_for(10**8)


def test_zero_slope_gives_one_threshold_for_every_amount():
    c = GateConfig(slope=0.0, base=0.9)
    assert c.threshold_for(1) == pytest.approx(c.threshold_for(10**9))


# --------------------------------------------------------------------------- #
# Why there is no confidence. `decide()` documents `None` as "blocking produced
# nothing", and three of its four callers in the engine pass `None` for records
# that had plenty of candidates — a record routed away as grouped, one whose
# candidates could not be scored, and one lone candidate whose amount does not
# match. All three were written into the append-only log as
# `outcome=no_candidate, reason="no candidate survived blocking"` beside an
# `n_candidates` of 55, which is a row that contradicts itself.
# --------------------------------------------------------------------------- #

def test_no_candidates_is_the_only_no_candidate_outcome():
    d = decide(confidence=None, amount_minor=5_000, absent=Absent.NO_CANDIDATES)
    assert d.outcome is Outcome.NO_CANDIDATE
    assert "blocking" in d.reason


@pytest.mark.parametrize("absent", [
    Absent.SUSPECTED_GROUPED, Absent.UNSCORABLE, Absent.NO_SUPPORT,
])
def test_a_record_with_candidates_is_queued_not_called_no_candidate(absent):
    """It went to a person. That is a queue, and the reason must say which."""
    d = decide(confidence=None, amount_minor=5_000, absent=absent)
    assert d.outcome is Outcome.QUEUE
    assert "no candidate survived blocking" not in d.reason
    assert d.reason


def test_every_absent_reason_states_a_distinct_cause():
    reasons = {a: decide(confidence=None, amount_minor=5_000, absent=a).reason
               for a in Absent}
    assert len(set(reasons.values())) == len(Absent), reasons


@pytest.mark.parametrize("absent", [Absent.NO_RANKER, Absent.MODEL_ERROR])
def test_a_missing_model_is_not_reported_as_a_blocking_failure(absent):
    """No ranker configured, or one that raised, says nothing about blocking.
    Both logged 'no candidate survived blocking' and queued under a cause that
    had not happened."""
    d = decide(confidence=None, amount_minor=5_000, absent=absent)
    assert d.outcome is Outcome.QUEUE
    assert "blocking" not in d.reason
