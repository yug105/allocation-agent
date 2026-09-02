"""Resolving a grouped credit on the upload path.

The multiplicity detector says "this one payment covers several entries" and
stops there. The solver can say *which* entries. Both were built and, until
this, never introduced: `solve_subset` was reachable only through
`/api/settlements` and `/api/connect`, so a visitor who uploaded their own file
got the diagnosis without the answer, even when the answer was two rows in a
pool of three.

The rule these tests exist to hold is that resolving is **evidence, not a
posting**. A subset that balances is not proof it is the right subset — this
project measured 51.3% wrong sets on a pool of ~100 — so a proposal is attached
to a review item and the record stays in the queue. Nothing here may turn a
grouped record into an auto-post.
"""

from __future__ import annotations

import numpy as np
import pytest

from allocation_agent.decide.gate import GateConfig, Outcome
from allocation_agent.match.blocker import BlockingConfig
from allocation_agent.match.engine import Models, match_one
from allocation_agent.match.features import build_key_stats
from allocation_agent.match.multiplicity import AccountPrior
from allocation_agent.match.solver import SolverConfig
from allocation_agent.stores.keys import KeyIndex, KeyRow
from allocation_agent.types import BankRecord

BLOCKING = BlockingConfig(date_slack_days=7)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from allocation_agent.api import create_app
    return TestClient(create_app())


class AlwaysGrouped:
    """Stands in for the detector so these tests measure the resolution step.

    Whether the real detector fires on a given record is `test_multiplicity`'s
    subject; what happens *after* it fires is this file's.
    """

    def predict_proba(self, X):
        return np.ones(len(X), dtype=float)


class NeverRanks:
    """The ranker must not be reached on a grouped record."""

    def score(self, X):
        raise AssertionError("ranking ran on a record routed as grouped")


def models() -> Models:
    prior = AccountPrior(median_amount_minor={}, mult_rate={},
                         global_median=100_000.0, global_mult_rate=0.1)
    return Models(ranker=NeverRanks(), detector=AlwaysGrouped(), prior=prior)


def ledger(*rows: tuple[str, int]) -> tuple[KeyIndex, dict]:
    key_rows = [KeyRow(k, "ACC-1", a, 100) for k, a in rows]
    return KeyIndex(key_rows), build_key_stats(key_rows)


def credit(amount_minor: int) -> BankRecord:
    return BankRecord(record_id="B1", account="ACC-1",
                      amount_minor=amount_minor, day=100)


def run(rec, index, key_stats, *, solver=SolverConfig(max_candidates=64)):
    return match_one(rec, index=index, key_stats=key_stats, models=models(),
                     gate=GateConfig(), mult_threshold=0.7, blocking=BLOCKING,
                     narrator=None, calibrated_for_this_data=False,
                     group_solver=solver)


# --------------------------------------------------------------------------- #
# the thing being built
# --------------------------------------------------------------------------- #

def test_a_grouped_credit_is_told_which_entries_make_it_up():
    """20,000 covering 12,000 + 8,000, from a pool of three."""
    index, key_stats = ledger(("INV-1", 1_200_000), ("INV-2", 800_000),
                              ("INV-3", 330_000))
    r = run(credit(2_000_000), index, key_stats)

    assert r["outcome"] == "suspected_grouped"
    assert r["proposal"] is not None
    assert sorted(r["proposal"]["keys"]) == ["INV-1", "INV-2"]
    assert sum(r["proposal"]["amounts_minor"]) == 2_000_000


def test_the_proposal_is_named_in_the_explanation_a_person_reads():
    index, key_stats = ledger(("INV-1", 1_200_000), ("INV-2", 800_000),
                              ("INV-3", 330_000))
    r = run(credit(2_000_000), index, key_stats)
    assert "INV-1" in r["explanation"] and "INV-2" in r["explanation"]


# --------------------------------------------------------------------------- #
# the rule: a proposal is evidence, never a posting
# --------------------------------------------------------------------------- #

def test_a_resolved_group_is_still_sent_to_a_person():
    """A balancing subset is not proof of membership. On a pool of ~100 this
    project measured 51.3% of balancing subsets to be the wrong set, so the
    answer goes to a reviewer with its working shown, not into the ledger."""
    index, key_stats = ledger(("INV-1", 1_200_000), ("INV-2", 800_000),
                              ("INV-3", 330_000))
    r = run(credit(2_000_000), index, key_stats)

    assert r["outcome"] == "suspected_grouped"
    assert r["decision"].outcome is not Outcome.POST
    assert r["keys"] == [], "a proposal must not be recorded as a match"
    assert r["confidence"] is None


def test_two_subsets_of_the_same_size_produce_no_proposal():
    """5,000 = 3,000 + 2,000 and also = 4,000 + 1,000. The amounts do not
    distinguish them, so neither is offered."""
    index, key_stats = ledger(("INV-1", 300_000), ("INV-2", 200_000),
                              ("INV-3", 400_000), ("INV-4", 100_000))
    r = run(credit(500_000), index, key_stats)

    assert r["outcome"] == "suspected_grouped"
    assert r["proposal"] is None
    assert "more than one" in r["explanation"].lower()


def test_no_combination_reaching_the_total_is_said_plainly():
    index, key_stats = ledger(("INV-1", 300_000), ("INV-2", 200_000))
    r = run(credit(999_999), index, key_stats)
    assert r["proposal"] is None
    assert r["outcome"] == "suspected_grouped"


# --------------------------------------------------------------------------- #
# it is opt-in, so nothing else changes
# --------------------------------------------------------------------------- #

def test_without_a_solver_config_the_behaviour_is_exactly_as_before():
    """`/api/run` and `pipeline.run_batch` share this function. Wiring the
    solver for uploads must not silently alter what the benchmark reports."""
    index, key_stats = ledger(("INV-1", 1_200_000), ("INV-2", 800_000))
    off = run(credit(2_000_000), index, key_stats, solver=None)

    assert off["outcome"] == "suspected_grouped"
    assert off["proposal"] is None
    assert "INV-1" not in off["explanation"]


def test_the_explanation_without_a_solver_is_unchanged():
    index, key_stats = ledger(("INV-1", 1_200_000), ("INV-2", 800_000))
    off = run(credit(2_000_000), index, key_stats, solver=None)
    assert off["explanation"].endswith("Sent for review.")


# --------------------------------------------------------------------------- #
# failure must not cost a decision, the way narration must not
# --------------------------------------------------------------------------- #

def test_a_solver_that_throws_cannot_unmake_the_grouping_decision():
    """Resolution runs after the record is already routed. Like narration, it
    explains a decision rather than making one, so its failure is recorded and
    the decision stands."""
    import allocation_agent.match.engine as engine

    index, key_stats = ledger(("INV-1", 1_200_000), ("INV-2", 800_000))
    good = run(credit(2_000_000), index, key_stats)

    def boom(**kwargs):
        raise RuntimeError("solver exploded")

    real = engine.solve_subset
    engine.solve_subset = boom
    try:
        bad = run(credit(2_000_000), index, key_stats)
    except Exception:  # noqa: BLE001
        pytest.fail("a failing solver propagated out of matching")
    finally:
        engine.solve_subset = real

    assert bad["outcome"] == good["outcome"] == "suspected_grouped"
    assert bad["proposal"] is None
    assert bad["evidence"].get("resolution_failed") is True


def test_a_pool_larger_than_the_solver_accepts_is_refused_not_truncated():
    """Silently dropping candidates to fit would let the answer be excluded and
    a wrong subset offered in its place."""
    rows = [(f"INV-{i}", 1_000 + i) for i in range(80)]
    index, key_stats = ledger(*rows)
    r = run(credit(2_050), index, key_stats,
            solver=SolverConfig(max_candidates=8))

    assert r["proposal"] is None
    assert r["outcome"] == "suspected_grouped"


# --------------------------------------------------------------------------- #
# money
# --------------------------------------------------------------------------- #

def test_the_proposed_amounts_sum_to_the_credit_exactly():
    """Reported in minor units and checked as integers — a proposal that is a
    penny out is a wrong proposal, and floats would hide it."""
    index, key_stats = ledger(("INV-1", 1_234_567), ("INV-2", 765_433),
                              ("INV-3", 111_111))
    r = run(credit(2_000_000), index, key_stats)

    assert r["proposal"] is not None
    amounts = r["proposal"]["amounts_minor"]
    assert all(isinstance(a, int) for a in amounts)
    assert sum(amounts) == 2_000_000


# --------------------------------------------------------------------------- #
# the wiring: an uploaded file gets this, the benchmark run does not
# --------------------------------------------------------------------------- #

BANK = ("date,description,amount,account\n"
        "2026-03-05,UMBRELLA LTD SETTLEMENT,20000.00,ACC-1002\n")
LEDGER = ("date,reference,amount,account\n"
          "2026-03-05,INV-4474 UMBRELLA,12000.00,ACC-1002\n"
          "2026-03-05,INV-4475 UMBRELLA,8000.00,ACC-1002\n"
          "2026-03-04,INV-4473 INITECH,975.25,ACC-1002\n")


def _upload(client, bank=BANK, ledger=LEDGER):
    import io
    return client.post("/api/reconcile", files={
        "bank": ("b.csv", io.BytesIO(bank.encode()), "text/csv"),
        "ledger": ("l.csv", io.BytesIO(ledger.encode()), "text/csv"),
    }).json()


def test_an_uploaded_grouped_credit_comes_back_with_its_members_named(client):
    """The case the whole project is about, on the one screen a visitor
    reaches with their own data. Before this, the answer stopped at the
    diagnosis: `solve_subset` was reachable only through /api/settlements."""
    body = _upload(client)
    row = body["results"][0]

    assert row["outcome"] == "suspected_grouped"
    assert row["proposal"] is not None
    assert sorted(row["proposal"]["keys"]) == ["INV-4474 UMBRELLA", "INV-4475 UMBRELLA"]
    assert sorted(row["proposal"]["amounts"]) == [8000.00, 12000.00]
    assert sum(row["proposal"]["amounts"]) == row["amount"]


def test_the_response_says_how_many_groups_it_resolved(client):
    body = _upload(client)
    assert body["groups_found"] == 1
    assert body["groups_resolved"] == 1


def test_a_resolved_upload_is_still_not_a_match(client):
    """`matched_key` stays null and the record stays in the queue. A split that
    balances is working shown to a reviewer, not a posting."""
    row = _upload(client)["results"][0]
    assert row["matched_key"] is None
    assert row["confidence"] is None
    assert row["outcome"] == "suspected_grouped"


def test_no_accuracy_is_claimed_for_a_resolved_upload(client):
    """An uploaded file has no answer key. The count of resolved groups is a
    count; whether a split is *right* is the reviewer's call."""
    body = _upload(client)
    assert body["confidence_validated_for_this_data"] is False
    assert "no precision is reported" in body["caveat"]
    assert not any(k.startswith("precision") for k in body)


def test_the_benchmark_run_is_unchanged_by_this(client):
    """/api/run publishes figures a reader checks against the README. Wiring
    the solver for uploads must not move them, so the demo path passes no
    solver and its grouped records carry no proposal."""
    body = client.post("/api/run", json={"limit": 200}).json()
    grouped = [e for e in body["exceptions"] if e["reason"] == "suspected_grouped"]
    assert grouped, "no grouped records in the sample; the test proves nothing"
    assert all("proposal" not in e for e in grouped)
    assert "groups_resolved" not in body
