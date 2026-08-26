"""End-to-end runner: every record accounted for, nothing silently dropped."""

import pytest

from allocation_agent.decide.gate import GateConfig
from allocation_agent.match.features import KeyStats
from allocation_agent.pipeline import run_batch
from allocation_agent.report.audit import AuditLog, RunConfig
from allocation_agent.stores.keys import KeyIndex, KeyRow
from allocation_agent.types import BankRecord


@pytest.fixture()
def audit(tmp_path):
    return AuditLog(tmp_path / "a.db")


def cfg() -> RunConfig:
    return RunConfig(approved_by="test", blocking={}, gate={})


def setup(n_matchable=3, n_orphan=1):
    rows, records, stats = [], [], {}
    for i in range(n_matchable):
        k = f"K{i}"
        rows.append(KeyRow(key=k, account="ACC", amount_minor=1000 + i, day=100))
        stats[k] = KeyStats(frozenset({1000 + i}), (100,), 1)
        records.append(BankRecord(f"b{i}", "ACC", 1000 + i, 100))
    for j in range(n_orphan):
        records.append(BankRecord(f"orphan{j}", "NOWHERE", 999_999, 500))
    return KeyIndex(rows), stats, records


def test_every_record_produces_exactly_one_decision(audit):
    idx, stats, records = setup()
    r = run_batch(records, idx, stats, audit, run_config=cfg())
    assert r.posted + r.queued + r.no_candidate + r.suspected_multiple == len(records)
    assert len(audit.decisions(r.run_id)) == len(records)


def test_orphan_records_are_reported_not_dropped(audit):
    idx, stats, records = setup(n_orphan=2)
    r = run_batch(records, idx, stats, audit, run_config=cfg())
    assert r.no_candidate == 2
    assert r.exceptions["no_candidate"] == 2


def test_runs_without_a_ranker(audit):
    """No model available must degrade to rules, not halt."""
    idx, stats, records = setup()
    r = run_batch(records, idx, stats, audit, run_config=cfg(), ranker=None)
    assert r.posted > 0
    assert all(d["path"] in ("fallback_rules", "blocked") for d in audit.decisions(r.run_id))


def test_fallback_path_is_named_in_the_audit_trail(audit):
    """A reviewer must be able to see the model was not involved."""
    idx, stats, records = setup(n_orphan=0)
    r = run_batch(records, idx, stats, audit, run_config=cfg())
    assert {d["path"] for d in audit.decisions(r.run_id)} == {"fallback_rules"}


def test_no_llm_call_on_the_matching_path(audit):
    idx, stats, records = setup()
    r = run_batch(records, idx, stats, audit, run_config=cfg())
    assert r.llm_calls == 0


def test_strict_gate_queues_everything(audit):
    idx, stats, records = setup(n_orphan=0)
    r = run_batch(records, idx, stats, audit, run_config=cfg(),
                  gate=GateConfig(review_all=True))
    assert r.posted == 0 and r.queued == len(records)


def test_result_reports_throughput(audit):
    idx, stats, records = setup()
    r = run_batch(records, idx, stats, audit, run_config=cfg())
    assert r.records_per_second > 0
    assert 0.0 <= r.straight_through_rate <= 1.0


def test_run_is_recorded_with_its_approver(audit):
    idx, stats, records = setup()
    r = run_batch(records, idx, stats, audit,
                  run_config=RunConfig(approved_by="alice", blocking={}, gate={}))
    assert audit.get_run(r.run_id)["approved_by"] == "alice"


def test_empty_batch_is_a_valid_run_not_an_error(audit):
    idx, stats, _ = setup()
    r = run_batch([], idx, stats, audit, run_config=cfg())
    assert r.n_records == 0 and r.straight_through_rate == 0.0


def test_rerun_is_deterministic(audit, tmp_path):
    idx, stats, records = setup()
    a = run_batch(records, idx, stats, audit, run_config=cfg())
    b = run_batch(records, idx, stats, AuditLog(tmp_path / "b.db"), run_config=cfg())
    assert (a.posted, a.queued, a.no_candidate) == (b.posted, b.queued, b.no_candidate)


def test_a_tie_on_the_deciding_evidence_is_refused_not_posted():
    """Three ledger entries of the same amount on the same day are a tie, not
    a match. Returning the first at 0.90 would auto-post an arbitrary pick --
    the exact behaviour the rest of the system refuses."""
    from allocation_agent.match.features import KeyStats
    from allocation_agent.pipeline import _fallback_choice
    from allocation_agent.types import BankRecord

    rec = BankRecord("b1", "A", 1_000_000, 100)
    stats = {k: KeyStats(amounts=frozenset({1_000_000}), days=(100,), n_rows=1)
             for k in ("K1", "K2", "K3")}
    chosen, confidence = _fallback_choice(rec, ["K1", "K2", "K3"], stats)
    assert chosen is None
    assert confidence is None


def test_a_clear_winner_is_still_chosen():
    from allocation_agent.match.features import KeyStats
    from allocation_agent.pipeline import _fallback_choice
    from allocation_agent.types import BankRecord

    rec = BankRecord("b1", "A", 1_000_000, 100)
    stats = {
        "K1": KeyStats(amounts=frozenset({1_000_000}), days=(100,), n_rows=1),
        "K2": KeyStats(amounts=frozenset({9_999_999}), days=(100,), n_rows=1),
    }
    chosen, confidence = _fallback_choice(rec, ["K1", "K2"], stats)
    assert chosen == "K1"
    assert confidence == 0.90


def test_scores_and_selection_index_the_same_list():
    """X is built from candidates present in key_stats; selecting from the
    unfiltered list shifts every position after the first missing key, so a
    correct ranking still picks the wrong ledger entry."""
    import inspect

    from allocation_agent import pipeline
    src = inspect.getsource(pipeline.run_batch)
    assert "chosen = scored[int(order[0])]" in src
    assert "chosen = candidates[int(order[0])]" not in src
