"""Append-only audit log.

Not a report assembled afterwards -- this is what the run emits as it goes. An
auditor asks what was decided, why, and under which rules; all three are
answerable from the log alone, months later, without rerunning anything.

**Append-only is enforced by the database, not by convention.** SQLite triggers
reject `UPDATE` and `DELETE` on the decisions table, so a row cannot be quietly
revised even by code that means to. A reviewer overturning a decision writes a
*new* row; both remain visible, which is the point.

**A run must be authorised before it can record anything.** Settings are approved
first and stored with the run, so a later reviewer can tell whether a tolerance
was widened to make the numbers work. Approving results after the fact cannot
answer that question.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from allocation_agent.decide.gate import GateDecision


class ImmutableError(RuntimeError):
    """Raised when something tries to change a recorded decision."""


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Settings approved before the run, stored with it."""

    approved_by: str
    blocking: dict[str, Any]
    gate: dict[str, Any]
    policy_version: str = "v0.1"
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.approved_by.strip():
            raise ValueError("approved_by is required: settings approved by nobody is not a control")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    status        TEXT NOT NULL DEFAULT 'running',
    failure       TEXT,
    approved_by   TEXT NOT NULL,
    blocking      TEXT NOT NULL,
    gate          TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS decisions (
    seq           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES runs(run_id),
    record_id     TEXT NOT NULL,
    decided_at    TEXT NOT NULL,
    path          TEXT NOT NULL,
    outcome       TEXT NOT NULL,
    chosen_keys   TEXT NOT NULL,
    confidence    REAL,
    threshold_required REAL NOT NULL,
    amount_minor  INTEGER NOT NULL,
    n_candidates  INTEGER NOT NULL,
    reason        TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    evidence      TEXT,
    reviewer      TEXT
);

CREATE INDEX IF NOT EXISTS idx_dec_run ON decisions(run_id);
CREATE INDEX IF NOT EXISTS idx_dec_rec ON decisions(record_id);

CREATE TRIGGER IF NOT EXISTS decisions_no_update
BEFORE UPDATE ON decisions
BEGIN
    SELECT RAISE(ABORT, 'decisions are append-only: record a correction instead');
END;

CREATE TRIGGER IF NOT EXISTS decisions_no_delete
BEFORE DELETE ON decisions
BEGIN
    SELECT RAISE(ABORT, 'decisions are append-only: record a correction instead');
END;
"""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class AuditLog:
    """One SQLite file per environment. Cheap, embedded, and survives a restart."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # A web server handles requests on a thread pool, and SQLite binds a
        # connection to its creating thread by default. Sharing one connection
        # across threads needs both the flag *and* a lock -- the flag alone
        # removes the guard without making writes safe.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._migrate()
            self._conn.commit()
        self._run_id: str | None = None

    def _migrate(self) -> None:
        """Add columns an older file predates.

        `runs.db` already exists in deployment. A schema change that cannot
        read it would discard exactly the history the log was built to keep, so
        the columns are added in place and existing rows are given a status
        inferred from whether they ever finished.
        """
        have = {r["name"] for r in self._conn.execute("PRAGMA table_info(runs)")}
        if "status" not in have:
            self._conn.execute("ALTER TABLE runs ADD COLUMN status TEXT")
            self._conn.execute(
                "UPDATE runs SET status = CASE WHEN finished_at IS NULL "
                "THEN 'unknown' ELSE 'completed' END"
            )
        if "failure" not in have:
            self._conn.execute("ALTER TABLE runs ADD COLUMN failure TEXT")

    # -- runs ---------------------------------------------------------------- #

    def start_run(self, config: RunConfig) -> str:
        run_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute(
            "INSERT INTO runs (run_id, started_at, status, approved_by, blocking, "
            "gate, policy_version, notes) VALUES (?,?,?,?,?,?,?,?)",
                (run_id, _now(), "running", config.approved_by,
                 json.dumps(config.blocking), json.dumps(config.gate),
                 config.policy_version, config.notes),
            )
            self._conn.commit()
        self._run_id = run_id
        return run_id

    def finish_run(self, run_id: str | None = None) -> None:
        """Mark a run completed. The batch got to the end."""
        rid = run_id or self._run_id
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET finished_at = ?, status = 'completed' WHERE run_id = ?",
                (_now(), rid),
            )
            self._conn.commit()
        if rid == self._run_id:
            self._run_id = None

    def fail_run(self, run_id: str | None = None, reason: str = "") -> None:
        """Mark a run failed, keeping whatever it managed to write.

        Without this a crashed run is indistinguishable from one still in
        progress, a killed process, or a broken audit writer -- all of them
        rows with a null `finished_at`. A partial trail is evidence; the thing
        that must not be ambiguous is whether it is complete.
        """
        rid = run_id or self._run_id
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM runs WHERE run_id = ?", (rid,)).fetchone()
            if row is None:
                raise KeyError(rid)
            if row["status"] == "completed":
                raise ValueError(f"run {rid} is already completed; it cannot fail")
            self._conn.execute(
                "UPDATE runs SET finished_at = ?, status = 'failed', failure = ? "
                "WHERE run_id = ?",
                (_now(), reason or "unspecified", rid),
            )
            self._conn.commit()
        if rid == self._run_id:
            self._run_id = None

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return dict(row)

    # -- decisions ----------------------------------------------------------- #

    def record(
        self,
        record_id: str,
        decision: GateDecision,
        *,
        keys: list[str],
        n_candidates: int,
        path: str,
        evidence: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> None:
        """Append one decision. Never overwrites.

        **Pass `run_id` from any concurrent caller.** One `AuditLog` is shared
        by the whole service, and `self._run_id` is a single mutable slot: two
        requests overlapping means the second `start_run` overwrites the first,
        so one batch's decisions land under the other's run — and when the first
        `finish_run` clears the slot, the batch still going inserts a NULL and
        dies. Measured with four concurrent writers: three raised
        `IntegrityError` and lost every row they had written.

        The attribute remains for single-threaded scripts, where there is only
        ever one run and passing it would be noise.
        """
        rid = run_id or self._run_id
        if rid is None:
            raise RuntimeError("no run in progress: call start_run() with approved settings first")
        with self._lock:
            self._conn.execute(
            "INSERT INTO decisions (run_id, record_id, decided_at, path, outcome, chosen_keys, "
            "confidence, threshold_required, amount_minor, n_candidates, reason, policy_version, "
            "evidence, reviewer) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rid, record_id, _now(), path, decision.outcome.value,
             json.dumps(keys), decision.confidence, decision.threshold_required,
             decision.amount_minor, n_candidates, decision.reason,
                 decision.policy_version, json.dumps(evidence) if evidence else None, None),
            )

    def record_correction(
        self,
        record_id: str,
        run_id: str,
        correct_keys: list[str],
        locus: str,
        detail: str,
        reviewer: str = "",
        reviewer_notes: str = "",
    ) -> None:
        """Append a human correction as a new row.

        The original decision stays visible. An audit trail that hides what the
        machine originally proposed cannot show that a human changed anything —
        and one that does not name *who* overturned it cannot show who did.

        `locus` is why the machine was wrong, attributed by `learn.router`: a
        correction that does not say which stage failed cannot be routed to the
        fix, because widening blocking will not repair a ranking miss.

        `correct_keys` is a list because a reviewer saying a credit covers three
        invoices is naming three keys. The caller used to join them with ", "
        into a single string, which the log stored as `["KEY-A, KEY-B"]` — one
        key whose name contains a comma, and unparseable back into two.
        """
        with self._lock:
            # Scoped to the run. Without it a record corrected in one run
            # inherits the amount and policy of its last decision in another,
            # which is the same shared-state defect `record()` already had.
            prior = self._conn.execute(
                "SELECT amount_minor, n_candidates, policy_version FROM decisions "
                "WHERE record_id = ? AND run_id = ? ORDER BY seq DESC LIMIT 1",
                (record_id, run_id),
            ).fetchone()
        
        if prior is None:
            raise KeyError(f"no prior decision for {record_id}")

        evidence = {"locus": locus, "detail": detail, "reviewer_notes": reviewer_notes}
        
        with self._lock:
            self._conn.execute(
                "INSERT INTO decisions (run_id, record_id, decided_at, path, outcome, chosen_keys, "
                "confidence, threshold_required, amount_minor, n_candidates, reason, policy_version, "
                "evidence, reviewer) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                # `outcome` is "correction", not "post". It was "post", so a
                # reviewer overturning the machine was recorded with the same
                # outcome as a machine auto-post, and every count of posted
                # decisions silently included the corrections that said the
                # machine had been wrong.
                (run_id, record_id, _now(), "correction", "correction",
                 json.dumps(list(correct_keys)),
                 None, 0.0, prior["amount_minor"], prior["n_candidates"],
                 reviewer_notes or "corrected by reviewer", prior["policy_version"],
                 json.dumps(evidence), reviewer or None),
            )
            self._conn.commit()


    def commit(self) -> None:
        with self._lock:
            self._conn.commit()

    # -- reading ------------------------------------------------------------- #

    def decisions(self, run_id: str | None = None) -> list[dict[str, Any]]:
        rid = run_id or self._run_id
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM decisions WHERE run_id = ? ORDER BY seq", (rid,)
            ).fetchall()
        return [dict(r) for r in rows]

    def last_decision(self, record_id: str, *, run_id: str,
                      machine_only: bool = False) -> dict[str, Any] | None:
        """The most recent decision recorded for one record in one run.

        Exists so callers do not reach through `_conn` and `_lock` to read the
        log. Those are private because every write goes through the lock; a
        caller that borrows them is one refactor away from writing without it.

        `machine_only` skips correction rows. A correction is appended like any
        other decision, so the most recent row for a corrected record is the
        correction — and a caller asking "what did the machine decide?" got a
        reviewer's note back, with no ranking on it and no candidate count. Any
        second correction was then diagnosed against the first.
        """
        sql = "SELECT * FROM decisions WHERE record_id = ? AND run_id = ? "
        if machine_only:
            sql += "AND path != 'correction' "
        with self._lock:
            row = self._conn.execute(
                sql + "ORDER BY seq DESC LIMIT 1", (record_id, run_id)).fetchone()
        return dict(row) if row else None

    def summary(self, run_id: str | None = None) -> dict[str, int]:
        rid = run_id or self._run_id
        with self._lock:
            rows = self._conn.execute(
                "SELECT outcome, COUNT(*) n FROM decisions WHERE run_id = ? GROUP BY outcome", (rid,)
            ).fetchall()
        return {r["outcome"]: r["n"] for r in rows}

    def raw_execute(self, sql: str) -> None:
        """Escape hatch for tests. Append-only triggers still apply."""
        try:
            with self._lock:
                self._conn.execute(sql)
                self._conn.commit()
        except sqlite3.IntegrityError as exc:
            raise ImmutableError(str(exc)) from exc
        except sqlite3.OperationalError as exc:
            if "append-only" in str(exc):
                raise ImmutableError(str(exc)) from exc
            raise

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()
