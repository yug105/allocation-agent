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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
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
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
            self._conn.commit()
        self._run_id: str | None = None

    # -- runs ---------------------------------------------------------------- #

    def start_run(self, config: RunConfig) -> str:
        run_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._conn.execute(
            "INSERT INTO runs (run_id, started_at, approved_by, blocking, gate, "
            "policy_version, notes) VALUES (?,?,?,?,?,?,?)",
                (run_id, _now(), config.approved_by, json.dumps(config.blocking),
                 json.dumps(config.gate), config.policy_version, config.notes),
            )
            self._conn.commit()
        self._run_id = run_id
        return run_id

    def finish_run(self, run_id: str | None = None) -> None:
        rid = run_id or self._run_id
        with self._lock:
            self._conn.execute("UPDATE runs SET finished_at = ? WHERE run_id = ?", (_now(), rid))
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
    ) -> None:
        """Append one decision. Never overwrites."""
        if self._run_id is None:
            raise RuntimeError("no run in progress: call start_run() with approved settings first")
        with self._lock:
            self._conn.execute(
            "INSERT INTO decisions (run_id, record_id, decided_at, path, outcome, chosen_keys, "
            "confidence, threshold_required, amount_minor, n_candidates, reason, policy_version, "
            "evidence, reviewer) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (self._run_id, record_id, _now(), path, decision.outcome.value,
             json.dumps(keys), decision.confidence, decision.threshold_required,
             decision.amount_minor, n_candidates, decision.reason,
                 decision.policy_version, json.dumps(evidence) if evidence else None, None),
            )

    def record_correction(
        self,
        record_id: str,
        *,
        corrected_keys: list[str],
        reviewer: str,
        note: str = "",
    ) -> None:
        """Append a human correction as a new row.

        The original decision stays visible. An audit trail that hides what the
        machine originally proposed cannot show that a human changed anything.
        """
        if self._run_id is None:
            raise RuntimeError("no run in progress")
        with self._lock:
            prior = self._conn.execute(
                "SELECT amount_minor, n_candidates, policy_version FROM decisions "
                "WHERE record_id = ? ORDER BY seq DESC LIMIT 1",
                (record_id,),
            ).fetchone()
        if prior is None:
            raise KeyError(f"no prior decision for {record_id}")

        with self._lock:
            self._conn.execute(
            "INSERT INTO decisions (run_id, record_id, decided_at, path, outcome, chosen_keys, "
            "confidence, threshold_required, amount_minor, n_candidates, reason, policy_version, "
            "evidence, reviewer) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (self._run_id, record_id, _now(), "human", "post", json.dumps(corrected_keys),
                 None, 0.0, prior["amount_minor"], prior["n_candidates"],
                 note or "corrected by reviewer", prior["policy_version"], None, reviewer),
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
