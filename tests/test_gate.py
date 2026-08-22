"""The gate: post, or send to a human.

A wrong match closes a real exception, asserts in the ledger that two unrelated
things are the same event, and surfaces months later as a reporting break. A
missed match costs someone ten minutes. Those are not symmetric, so the
confidence required must rise with the amount at stake.

Every product on the market uses one threshold for every amount.
"""

import pytest

from allocation_agent.decide.gate import GateConfig, GateDecision, Outcome, decide


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
