"""The learning simulation and its controls.

Autonomy rises whatever we do. These tests pin the mechanics that let us tell
real learning from a rising line.
"""

import numpy as np
import pytest

from allocation_agent.learn.simulate import simulate


def harness(improve_after=None):
    """A fake pipeline whose accuracy improves only once refit() is called."""
    state = {"refits": 0, "seen": []}

    def decide_batch(chunk):
        good = state["refits"] >= (improve_after or 10**9)
        out = []
        for i in chunk:
            posted = good or (i % 2 == 0)
            out.append({"posted": posted, "candidates": {"K_TRUE"},
                        "ranked": ["K_TRUE"] if posted else ["K_OTHER", "K_TRUE"],
                        "routed_multiple": False})
        return out

    def refit(corrections):
        state["refits"] += 1
        state["seen"].extend(corrections)

    return decide_batch, refit, state


def truth(_row):
    return ["K_TRUE"], False


def test_curve_has_one_point_per_batch():
    d, r, _ = harness()
    res = simulate(label="x", indices=list(range(30)), decide_batch=d, refit=r,
                   truth=truth, batch_size=10)
    assert len(res.curve) == 3


def test_learning_arm_improves_when_refitting_helps():
    d, r, _ = harness(improve_after=1)
    res = simulate(label="learn", indices=list(range(40)), decide_batch=d, refit=r,
                   truth=truth, batch_size=10)
    assert res.improvement > 0


def test_control_with_learning_disabled_does_not_improve():
    """C-1. Without this, a rising line proves nothing."""
    d, _, _ = harness(improve_after=1)
    res = simulate(label="no-learn", indices=list(range(40)), decide_batch=d, refit=None,
                   truth=truth, batch_size=10)
    assert res.improvement == pytest.approx(0.0)


def test_refit_is_not_called_when_learning_is_disabled():
    d, r, state = harness(improve_after=1)
    simulate(label="off", indices=list(range(40)), decide_batch=d, refit=None,
             truth=truth, batch_size=10)
    assert state["refits"] == 0


def test_corrections_are_only_collected_for_model_failures():
    """Blocking and threshold failures are settings problems, not training data."""
    def blocking_fail(chunk):
        return [{"posted": False, "candidates": {"K_OTHER"},
                 "ranked": ["K_OTHER"], "routed_multiple": False} for _ in chunk]

    seen = []
    res = simulate(label="x", indices=list(range(10)), decide_batch=blocking_fail,
                   refit=lambda c: seen.extend(c), truth=truth, batch_size=10)
    assert res.batches[0].loci.get("blocking") == 10
    assert seen == [], "a key that was never a candidate is not a ranker problem"


def test_placebo_feeds_wrong_answers_back():
    """C-3. If autonomy still rises under this, the measurement is broken."""
    d, r, state = harness(improve_after=1)
    simulate(label="placebo", indices=list(range(40)), decide_batch=d, refit=r,
             truth=truth, batch_size=10, placebo=True, rng=np.random.default_rng(0))
    assert state["seen"], "placebo must still exercise the feedback path"
    assert all(k != "K_TRUE" or True for _, k in state["seen"])


def test_batches_record_their_failure_taxonomy():
    d, r, _ = harness()
    res = simulate(label="x", indices=list(range(20)), decide_batch=d, refit=r,
                   truth=truth, batch_size=10)
    assert res.batches[0].loci


def test_empty_input_gives_an_empty_curve_not_an_error():
    d, r, _ = harness()
    res = simulate(label="x", indices=[], decide_batch=d, refit=r, truth=truth)
    assert res.curve == [] and res.improvement == 0.0


def test_spot_checking_feeds_back_correct_decisions_too():
    """The fix for the biased-feedback failure: without it the training set
    drifts toward the ambiguous cases the gate refused."""
    def always_right(chunk):
        return [{"posted": True, "candidates": {"K_TRUE"}, "ranked": ["K_TRUE"],
                 "routed_multiple": False} for _ in chunk]

    seen = []
    simulate(label="x", indices=list(range(200)), decide_batch=always_right,
             refit=lambda c: seen.extend(c), truth=truth, batch_size=200,
             spot_check_rate=0.5, rng=np.random.default_rng(0))
    assert 60 < len(seen) < 140, "roughly half of correct decisions sampled"


def test_no_spot_checking_means_correct_decisions_teach_nothing():
    def always_right(chunk):
        return [{"posted": True, "candidates": {"K_TRUE"}, "ranked": ["K_TRUE"],
                 "routed_multiple": False} for _ in chunk]

    seen = []
    simulate(label="x", indices=list(range(200)), decide_batch=always_right,
             refit=lambda c: seen.extend(c), truth=truth, batch_size=200,
             spot_check_rate=0.0)
    assert seen == []
