"""End-to-end runner: every record accounted for, nothing silently dropped."""

import pytest

from allocation_agent.decide.gate import GateConfig
from allocation_agent.match.features import KeyStats, build_key_stats
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


class _FlatRanker:
    """Scores everything equally. Enough to reach the stages after ranking
    without asserting anything about which key a real model would pick."""

    def score(self, X):
        import numpy as np
        return np.zeros(len(X), dtype=float)


def test_orphan_records_are_reported_not_dropped(audit):
    idx, stats, records = setup(n_orphan=2)
    r = run_batch(records, idx, stats, audit, run_config=cfg(), ranker=_FlatRanker())
    assert r.no_candidate == 2
    assert r.exceptions["no_candidate"] == 2




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





# --------------------------------------------------------------------------- #
# This module used to carry its own copy of the matching logic, and the copy
# fell behind: no calibration, a fabricated margin for the single-candidate
# case, the grouping check still able to overrule an exact amount, and a record
# that could leave the audit trail entirely when blocking found candidates that
# key_stats did not cover. A review found eight defects; every one was a fix
# that already existed in the API and had never been carried across.
#
# It now calls match_one, so there is one implementation to fix.
# --------------------------------------------------------------------------- #

def test_the_batch_runner_uses_the_shared_engine(audit, monkeypatch):
    """One matching path: this file used to carry its own copy of it.

    This grepped `run_batch`'s source for `match_one(` and broke the moment the
    loop moved into a helper — while still calling the shared engine on every
    record. Calling it is the property; where the call is written is not.
    """
    import inspect

    from allocation_agent import pipeline

    seen = []
    real = pipeline.match_one
    monkeypatch.setattr(pipeline, "match_one",
                        lambda rec, **kw: seen.append(rec) or real(rec, **kw))

    idx, stats, records = setup()
    pipeline.run_batch(records, idx, stats, audit, run_config=cfg(),
                       ranker=_ScoresInOrder())
    assert len(seen) == len(records), "a record bypassed the shared engine"

    src = inspect.getsource(pipeline)
    assert "featurise(" not in src, "still scoring locally"
    assert "np.argsort" not in src, "still ranking locally"


class _ScoresInOrder:
    def score(self, X):
        import numpy as np
        return np.linspace(1.0, 0.0, len(X))


def test_without_a_ranker_everything_goes_to_a_person(tmp_path):
    """The rules pick returned 0.90 for an exact amount and 0.55 otherwise --
    a different scale from the calibrated model, handed to the same gate. Not
    scoring is the honest degradation, and every record still gets a row."""
    from allocation_agent.match.blocker import BlockingConfig
    from allocation_agent.pipeline import run_batch
    from allocation_agent.report.audit import AuditLog, RunConfig
    from allocation_agent.stores.keys import KeyIndex, KeyRow
    from allocation_agent.types import BankRecord

    records = [BankRecord(f"b{i}", "A", 1000, 10) for i in range(4)]
    audit = AuditLog(tmp_path / "n.db")
    result = run_batch(records, KeyIndex([KeyRow("K1", "A", 1000, 10)]), {}, audit,
                       run_config=RunConfig(approved_by="t", blocking={}, gate={}),
                       blocking=BlockingConfig(date_slack_days=7), ranker=None)
    assert result.posted == 0
    assert result.queued == 4
    rows = audit.decisions(run_id=result.run_id)
    assert len(rows) == 4
    assert all(r["path"] == "no_ranker" for r in rows)


def test_a_model_failure_becomes_an_exception_not_the_end_of_the_batch(tmp_path):
    """'Degrades rather than halts' has to hold for the models too."""
    from allocation_agent.decide.gate import GateConfig
    from allocation_agent.match.blocker import BlockingConfig
    from allocation_agent.pipeline import run_batch
    from allocation_agent.report.audit import AuditLog, RunConfig
    from allocation_agent.stores.keys import KeyIndex, KeyRow
    from allocation_agent.types import BankRecord

    class Exploding:
        def score(self, X):
            raise RuntimeError("model blew up")

    records = [BankRecord(f"b{i}", "A", 1000, 10) for i in range(5)]
    rows = [KeyRow("K1", "A", 1000, 10), KeyRow("K2", "A", 1200, 11)]
    audit = AuditLog(tmp_path / "a.db")
    result = run_batch(records, KeyIndex(rows), {}, audit,
                       run_config=RunConfig(approved_by="t", blocking={}, gate={}),
                       blocking=BlockingConfig(date_slack_days=7),
                       gate=GateConfig(), ranker=Exploding())

    assert result.n_records == 5
    assert result.posted + result.queued + result.no_candidate \
        + result.suspected_multiple == 5
    assert len(audit.decisions(run_id=result.run_id)) == 5, \
        "a record left the audit trail"


def test_the_batch_applies_the_calibrator_it_is_given(tmp_path):
    """`calibrator` and `calibrator_kind` were accepted and never used.

    `match_one` applies a calibrator only when told the data is the population
    it was fitted on, and this file never passed that flag — so the batch
    scored on `sigmoid(margin)` whatever it was handed, and the script that
    produces the README's end-to-end figures measured a configuration the
    service does not run.
    """
    import json

    import numpy as np

    from allocation_agent.decide.gate import GateConfig
    from allocation_agent.match.blocker import BlockingConfig
    from allocation_agent.match.multiplicity import AccountPrior
    from allocation_agent.pipeline import run_batch
    from allocation_agent.report.audit import AuditLog, RunConfig
    from allocation_agent.stores.keys import KeyIndex, KeyRow
    from allocation_agent.types import BankRecord

    class Ranker:
        def score(self, X):
            return np.linspace(1.0, 0.0, len(X))

    class Detector:
        def predict_proba(self, X):
            return np.zeros(len(X))

    class Calibrator:
        """Maps any margin to a number no sigmoid can produce."""

        def predict(self, margins):
            return np.full(len(margins), 0.123)

    prior = AccountPrior(median_amount_minor={}, global_median=1000.0,
                         mult_rate={}, global_mult_rate=0.0)

    rows = [KeyRow(f"K{i}", "A", 1000 + i, 10) for i in range(3)]
    records = [BankRecord("b0", "A", 1000, 10)]
    audit = AuditLog(tmp_path / "c.db")
    result = run_batch(
        records, KeyIndex(rows), build_key_stats(rows), audit,
        run_config=RunConfig(approved_by="t", blocking={}, gate={}),
        blocking=BlockingConfig(date_slack_days=7), gate=GateConfig(),
        ranker=Ranker(), multiplicity=Detector(), prior=prior,
        calibrator=Calibrator(), calibrator_kind="isotonic",
        calibrated_for_this_data=True)

    row = audit.decisions(run_id=result.run_id)[0]
    assert row["confidence"] == pytest.approx(0.123), \
        "the calibrator was passed and ignored"
    assert json.loads(row["evidence"])["confidence_from"] == "isotonic"


def test_a_result_can_be_printed(audit):
    """`__str__` read `self.llm_calls`, which is not a field — so printing a
    result raised, and the script that prints one crashed before its figures."""
    idx, stats, records = setup()
    r = run_batch(records, idx, stats, audit, run_config=cfg())
    text = str(r)
    assert f"{len(records):,} records" in text
    assert "posted" in text


def test_a_finished_batch_is_committed_and_marked_finished(tmp_path):
    """The API path calls commit() and finish_run(); this one called neither.

    So a batch wrote 37,398 decisions that were discarded when the process
    exited, and left its run row at status='running' forever — the exact
    failure the audit design claims to prevent, inverted: a run that finished
    looks like one still going, and its trail is gone.
    """
    idx, stats, records = setup()
    audit = AuditLog(tmp_path / "b.db")
    r = run_batch(records, idx, stats, audit, run_config=cfg())

    # A second connection sees only what was committed.
    reread = AuditLog(tmp_path / "b.db")
    assert len(reread.decisions(r.run_id)) == len(records), \
        "the decisions were never committed"
    assert reread.get_run(r.run_id)["status"] == "completed"
