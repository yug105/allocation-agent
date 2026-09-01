"""The audit log.

Not a report generated afterwards — it is what the run emits as it goes. An
auditor asks three things: what was decided, why, and under which rules. All
three must be answerable from the log alone, months later, without rerunning
anything.

Append-only. A row that can be edited after the fact is not evidence.
"""

import json

import pytest

from allocation_agent.decide.gate import GateConfig, Outcome, decide
from allocation_agent.report.audit import AuditLog, ImmutableError, RunConfig


@pytest.fixture()
def log(tmp_path):
    return AuditLog(tmp_path / "audit.db")


def run_cfg(**kw) -> RunConfig:
    base = dict(approved_by="yug", blocking={"date_slack_days": 7},
                gate={"base": 0.85, "slope": 0.02}, policy_version="v0.1")
    base.update(kw)
    return RunConfig(**base)


# --------------------------------------------------------------------------- #
# a run must be authorised before it can record anything
# --------------------------------------------------------------------------- #

def test_recording_without_starting_a_run_is_refused(log):
    d = decide(confidence=0.99, amount_minor=100, config=GateConfig())
    with pytest.raises(RuntimeError, match="run"):
        log.record("b1", d, keys=["K1"], n_candidates=3, path="ranked")


def test_starting_a_run_returns_an_identifier(log):
    run_id = log.start_run(run_cfg())
    assert isinstance(run_id, str) and run_id


def test_run_stores_who_approved_it_and_the_settings_they_approved(log):
    run_id = log.start_run(run_cfg(approved_by="alice"))
    meta = log.get_run(run_id)
    assert meta["approved_by"] == "alice"
    assert json.loads(meta["blocking"])["date_slack_days"] == 7


def test_a_run_without_an_approver_is_refused(log):
    """Settings approved by nobody is the failure mode the control exists for."""
    with pytest.raises(ValueError, match="approved_by"):
        log.start_run(run_cfg(approved_by=""))


# --------------------------------------------------------------------------- #
# what a decision row must carry
# --------------------------------------------------------------------------- #

def test_decision_is_retrievable_with_its_reasoning(log):
    log.start_run(run_cfg())
    d = decide(confidence=0.97, amount_minor=50_000, config=GateConfig())
    log.record("b1", d, keys=["K1"], n_candidates=7, path="ranked")

    (row,) = log.decisions()
    assert row["record_id"] == "b1"
    assert row["outcome"] == Outcome.POST.value
    assert row["confidence"] == pytest.approx(0.97)
    assert row["threshold_required"] == pytest.approx(d.threshold_required)
    assert row["n_candidates"] == 7
    assert row["path"] == "ranked"
    assert row["reason"]


def test_policy_version_is_recorded_on_every_row(log):
    log.start_run(run_cfg(policy_version="v9.9"))
    d = decide(confidence=0.99, amount_minor=100, config=GateConfig(policy_version="v9.9"))
    log.record("b1", d, keys=["K"], n_candidates=1, path="direct")
    assert log.decisions()[0]["policy_version"] == "v9.9"


def test_several_keys_are_recorded_for_a_grouped_match(log):
    log.start_run(run_cfg())
    d = decide(confidence=0.9, amount_minor=1000, config=GateConfig())
    log.record("b1", d, keys=["K1", "K2", "K3"], n_candidates=9, path="solved")
    assert json.loads(log.decisions()[0]["chosen_keys"]) == ["K1", "K2", "K3"]


def test_evidence_survives_the_round_trip(log):
    log.start_run(run_cfg())
    d = decide(confidence=0.9, amount_minor=1000, config=GateConfig())
    ev = {"amount_delta": 0, "date_gap": 3, "runner_up_margin": 0.41}
    log.record("b1", d, keys=["K"], n_candidates=2, path="ranked", evidence=ev)
    assert json.loads(log.decisions()[0]["evidence"]) == ev


def test_every_row_carries_a_timestamp(log):
    log.start_run(run_cfg())
    d = decide(confidence=0.9, amount_minor=1, config=GateConfig())
    log.record("b1", d, keys=["K"], n_candidates=1, path="direct")
    assert log.decisions()[0]["decided_at"]


# --------------------------------------------------------------------------- #
# append-only
# --------------------------------------------------------------------------- #

def test_a_recorded_decision_cannot_be_updated(log):
    log.start_run(run_cfg())
    d = decide(confidence=0.9, amount_minor=1, config=GateConfig())
    log.record("b1", d, keys=["K"], n_candidates=1, path="ranked")
    with pytest.raises(ImmutableError):
        log.raw_execute("UPDATE decisions SET outcome = 'post' WHERE record_id = 'b1'")


def test_a_recorded_decision_cannot_be_deleted(log):
    log.start_run(run_cfg())
    d = decide(confidence=0.9, amount_minor=1, config=GateConfig())
    log.record("b1", d, keys=["K"], n_candidates=1, path="ranked")
    with pytest.raises(ImmutableError):
        log.raw_execute("DELETE FROM decisions WHERE record_id = 'b1'")


def test_a_correction_is_a_new_row_not_an_edit(log):
    """A reviewer overturning a decision must leave both visible."""
    run_id = log.start_run(run_cfg())
    d = decide(confidence=0.6, amount_minor=1_000_000, config=GateConfig())
    log.record("b1", d, keys=["K1"], n_candidates=5, path="ranked")
    log.record_correction("b1", run_id=run_id, correct_keys=["K2"], locus="ranking",
                          detail="the right key was offered and ranked second",
                          reviewer="alice",
                          reviewer_notes="settlement split across two batches")

    rows = log.decisions()
    assert len(rows) == 2
    assert rows[0]["outcome"] == Outcome.QUEUE.value
    assert rows[1]["path"] == "correction"
    assert rows[1]["reviewer"] == "alice", "the trail does not say who overturned it"
    assert json.loads(rows[1]["chosen_keys"]) == ["K2"]
    assert json.loads(rows[1]["evidence"])["locus"] == "ranking"


def test_a_correction_takes_the_amount_from_its_own_run(log):
    """Scoped by run_id: without it a record corrected in one run inherits the
    amount and policy of its last decision in another."""
    d_small = decide(confidence=0.6, amount_minor=100, config=GateConfig())
    d_big = decide(confidence=0.6, amount_minor=9_000_000, config=GateConfig())

    first = log.start_run(run_cfg())
    log.record("b1", d_small, keys=["K1"], n_candidates=2, path="ranked", run_id=first)
    log.finish_run(first)

    second = log.start_run(run_cfg())
    log.record("b1", d_big, keys=["K1"], n_candidates=9, path="ranked", run_id=second)
    log.record_correction("b1", run_id=second, correct_keys=["K2"], locus="ranking",
                          detail="", reviewer="bob")
    log.commit()

    correction = log.decisions(run_id=second)[-1]
    assert correction["amount_minor"] == 9_000_000
    assert correction["n_candidates"] == 9


# --------------------------------------------------------------------------- #
# the log is the source for every reported number
# --------------------------------------------------------------------------- #

def test_summary_counts_outcomes(log):
    log.start_run(run_cfg())
    for i, (conf, amt) in enumerate([(0.99, 100), (0.99, 100), (0.10, 10_000_000), (None, 500)]):
        d = decide(confidence=conf, amount_minor=amt, config=GateConfig())
        log.record(f"b{i}", d, keys=["K"], n_candidates=1, path="ranked")

    s = log.summary()
    assert s["post"] == 2
    assert s["queue"] == 1
    assert s["no_candidate"] == 1


def test_summary_covers_only_the_current_run(log):
    first = log.start_run(run_cfg())
    d = decide(confidence=0.99, amount_minor=1, config=GateConfig())
    log.record("b1", d, keys=["K"], n_candidates=1, path="ranked")
    log.finish_run(first)

    log.start_run(run_cfg())
    log.record("b2", d, keys=["K"], n_candidates=1, path="ranked")
    assert log.summary()["post"] == 1


def test_reopening_the_database_preserves_history(tmp_path):
    path = tmp_path / "audit.db"
    a = AuditLog(path)
    run_id = a.start_run(run_cfg())
    d = decide(confidence=0.99, amount_minor=1, config=GateConfig())
    a.record("b1", d, keys=["K"], n_candidates=1, path="ranked")
    a.close()

    b = AuditLog(path)
    assert len(b.decisions(run_id=run_id)) == 1


def test_the_log_is_usable_from_several_threads(tmp_path):
    """A web server handles requests on a thread pool. SQLite binds a connection
    to its creating thread by default, so this fails without an explicit flag
    and a lock -- and fails in production, not in development."""
    import threading

    from allocation_agent.decide.gate import GateConfig, decide

    log = AuditLog(tmp_path / "threads.db")
    log.start_run(run_cfg())
    errors: list[Exception] = []

    def worker(n: int) -> None:
        try:
            d = decide(confidence=0.99, amount_minor=100, config=GateConfig())
            for i in range(20):
                log.record(f"t{n}-{i}", d, keys=["K"], n_candidates=1, path="ranked")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    log.commit()

    assert not errors, errors
    assert len(log.decisions()) == 80


# --------------------------------------------------------------------------- #
# A run has a lifecycle, and "incomplete" has to be a state it can be in.
#
# Until now `finish_run` was called after the loop, so a run that crashed
# halfway left rows behind with `finished_at` null — indistinguishable from a
# run still in progress, a killed process, or a failed audit writer. For a
# reconciliation system somebody has to be able to tell those apart months
# later, from the log alone.
# --------------------------------------------------------------------------- #

def test_a_new_run_is_running(log):
    run_id = log.start_run(run_cfg())
    assert log.get_run(run_id)["status"] == "running"


def test_a_finished_run_is_completed(log):
    run_id = log.start_run(run_cfg())
    log.finish_run(run_id)
    assert log.get_run(run_id)["status"] == "completed"


def test_a_failed_run_says_so_and_why(log):
    run_id = log.start_run(run_cfg())
    d = decide(confidence=0.99, amount_minor=100, config=GateConfig())
    log.record("b1", d, keys=["K"], n_candidates=1, path="ranked")
    log.fail_run(run_id, "MemoryError")

    meta = log.get_run(run_id)
    assert meta["status"] == "failed"
    assert "MemoryError" in meta["failure"]
    assert meta["finished_at"], "a failed run still ended at a known time"


def test_the_rows_a_failed_run_wrote_are_kept(log):
    """A partial trail is evidence. Discarding it loses what did happen."""
    run_id = log.start_run(run_cfg())
    d = decide(confidence=0.99, amount_minor=100, config=GateConfig())
    for i in range(3):
        log.record(f"b{i}", d, keys=["K"], n_candidates=1, path="ranked")
    log.fail_run(run_id, "KeyboardInterrupt")
    assert len(log.decisions(run_id=run_id)) == 3


def test_a_completed_run_cannot_be_marked_failed_afterwards(log):
    run_id = log.start_run(run_cfg())
    log.finish_run(run_id)
    with pytest.raises(ValueError, match="completed"):
        log.fail_run(run_id, "too late")


def test_reopening_the_database_preserves_the_status(tmp_path):
    path = tmp_path / "status.db"
    a = AuditLog(path)
    run_id = a.start_run(run_cfg())
    a.fail_run(run_id, "OSError")
    a.close()
    assert AuditLog(path).get_run(run_id)["status"] == "failed"


def test_an_older_database_without_the_column_still_opens(tmp_path):
    """runs.db already exists in deployment. A schema change that cannot read
    it discards the audit history it was written to protect."""
    import sqlite3
    path = tmp_path / "legacy.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE runs (run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL,
            finished_at TEXT, approved_by TEXT NOT NULL, blocking TEXT NOT NULL,
            gate TEXT NOT NULL, policy_version TEXT NOT NULL, notes TEXT);
        INSERT INTO runs VALUES ('old1','2026-01-01','2026-01-01','yug','{}','{}','v0.1','');
    """)
    con.commit()
    con.close()

    log = AuditLog(path)
    assert log.get_run("old1")["status"] in {"completed", "unknown"}


# --------------------------------------------------------------------------- #
# One AuditLog is shared by the whole service and `_run_id` was a single
# mutable slot. Two overlapping requests meant the second start_run overwrote
# the first, so one batch's decisions landed under the other's run — and when
# the first finish_run cleared the slot, the batch still going inserted a NULL
# and died. Measured with four concurrent writers before the fix: three raised
# IntegrityError and lost every row they had written.
#
# For a system whose central claim is a complete audit trail, that is the worst
# available failure: it destroys the evidence rather than the answer.
# --------------------------------------------------------------------------- #

def test_concurrent_runs_keep_their_own_decisions(tmp_path):
    import threading

    log = AuditLog(tmp_path / "concurrent.db")
    d = decide(confidence=0.99, amount_minor=100, config=GateConfig())
    errors: list[str] = []
    runs: dict[str, str] = {}

    def writer(name: str) -> None:
        try:
            run_id = log.start_run(run_cfg())
            runs[name] = run_id
            for i in range(40):
                log.record(f"{name}-{i}", d, keys=["K"], n_candidates=1,
                           path="ranked", run_id=run_id)
            log.finish_run(run_id)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=writer, args=(f"user{u}",)) for u in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log.commit()

    assert not errors, errors
    assert len(runs) == 4
    for name, run_id in runs.items():
        rows = log.decisions(run_id=run_id)
        assert len(rows) == 40, f"{name} lost rows"
        assert all(r["record_id"].startswith(name) for r in rows), \
            f"{name}'s run contains another run's decisions"


def test_finishing_one_run_does_not_break_another_in_flight(tmp_path):
    """The specific crash: finish_run cleared the shared slot and the batch
    still going inserted a NULL run_id."""
    log = AuditLog(tmp_path / "inflight.db")
    d = decide(confidence=0.99, amount_minor=100, config=GateConfig())
    a = log.start_run(run_cfg())
    b = log.start_run(run_cfg())
    log.record("a1", d, keys=["K"], n_candidates=1, path="ranked", run_id=a)
    log.finish_run(a)
    log.record("b1", d, keys=["K"], n_candidates=1, path="ranked", run_id=b)
    log.commit()
    assert [r["record_id"] for r in log.decisions(run_id=a)] == ["a1"]
    assert [r["record_id"] for r in log.decisions(run_id=b)] == ["b1"]


def test_a_single_threaded_caller_still_needs_no_run_id(tmp_path):
    """Scripts have exactly one run; making them name it would be noise."""
    log = AuditLog(tmp_path / "single.db")
    run_id = log.start_run(run_cfg())
    d = decide(confidence=0.99, amount_minor=100, config=GateConfig())
    log.record("b1", d, keys=["K"], n_candidates=1, path="ranked")
    log.commit()
    assert len(log.decisions(run_id=run_id)) == 1


def test_the_suite_does_not_write_to_the_runtime_audit_log():
    """The tests used to append to `artifacts/runs.db` — the file the deployed
    service writes to. It reached 901 MB, and two overlapping runs deadlocked
    each other on it. conftest.py points AUDIT_DB at a temp file."""
    import os
    from pathlib import Path

    from allocation_agent.api import ARTIFACTS

    configured = os.environ.get("AUDIT_DB")
    assert configured, "AUDIT_DB is unset: an app built here writes to the real log"
    assert Path(configured).resolve() != (ARTIFACTS / "runs.db").resolve()


def _one_decision(log):
    from allocation_agent.decide.gate import GateConfig, decide
    d = decide(confidence=0.99, amount_minor=5_000, config=GateConfig())
    rid = log.start_run(RunConfig(approved_by="t", blocking={}, gate={}))
    log.record("r1", d, keys=["K1"], n_candidates=4, path="ranked", run_id=rid)
    log.commit()
    return rid


def test_a_correction_is_not_logged_as_an_auto_post(tmp_path):
    """`record_correction` hardcoded outcome='post', so a reviewer overturning
    the machine was recorded with the same outcome as a machine auto-post. Any
    count of posted decisions included every correction — including the ones
    saying the machine had been wrong."""
    log = AuditLog(tmp_path / "c.db")
    rid = _one_decision(log)
    log.record_correction("r1", rid, correct_keys=["K2"], locus="ranking",
                          detail="d", reviewer="yug")
    row = log.decisions(run_id=rid)[-1]
    assert row["path"] == "correction"
    assert row["outcome"] != "post", "a reviewer's correction counts as an auto-post"
    assert log.summary(rid).get("post", 0) == 1, "corrections inflate the posted count"


def test_a_multi_key_correction_is_logged_as_several_keys(tmp_path):
    """Joining them into one string makes ["KEY-A, KEY-B"] — a single key whose
    name contains a comma. Any real key containing one becomes unparseable."""
    import json

    log = AuditLog(tmp_path / "m.db")
    rid = _one_decision(log)
    log.record_correction("r1", rid, correct_keys=["KEY-A", "KEY-B"],
                          locus="multiplicity", detail="d", reviewer="yug")
    row = log.decisions(run_id=rid)[-1]
    assert json.loads(row["chosen_keys"]) == ["KEY-A", "KEY-B"]
