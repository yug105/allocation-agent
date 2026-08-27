"""End-to-end batch runner.

One record at a time: block, rank, detect multiplicity, gate, record. Every
decision lands in the audit log as it is made, so the reported numbers and the
audit trail cannot disagree -- they are the same rows.

**Degrades rather than halts.** With no trained models the runner falls back to
deterministic rules (nearest exact amount, then nearest date) and keeps going.
"The AI can fail, the payment system cannot" is a design constraint here, not a
slogan: the models are advisory and the pipeline must produce a complete,
explainable batch without them.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from allocation_agent.decide.gate import GateConfig, decide
from allocation_agent.match.blocker import BlockingConfig
from allocation_agent.match.engine import MULT_THRESHOLD, Models, match_one
from allocation_agent.match.features import KeyStats
from allocation_agent.report.audit import AuditLog, RunConfig
from allocation_agent.stores.keys import KeyIndex
from allocation_agent.types import BankRecord

#: Why a record could not be resolved. Drives the exception taxonomy, which is
#: reported rather than hidden.
EXCEPTION_REASONS = (
    "no_candidate",          # blocking found nothing
    "below_threshold",       # scored, but not confidently enough for the amount
    "suspected_multiple",    # looks like it spans several keys
)


@dataclass
class RunResult:
    run_id: str
    n_records: int
    posted: int = 0
    queued: int = 0
    no_candidate: int = 0
    suspected_multiple: int = 0
    exceptions: Counter = field(default_factory=Counter)
    seconds: float = 0.0

    @property
    def straight_through_rate(self) -> float:
        return self.posted / self.n_records if self.n_records else 0.0

    @property
    def records_per_second(self) -> float:
        return self.n_records / self.seconds if self.seconds else 0.0

    def __str__(self) -> str:
        return (
            f"{self.n_records:,} records in {self.seconds:.1f}s "
            f"({self.records_per_second:,.0f}/sec)\n"
            f"  posted            {self.posted:>8,}  ({self.straight_through_rate*100:5.1f}% STP)\n"
            f"  queued            {self.queued:>8,}\n"
            f"  no candidate      {self.no_candidate:>8,}\n"
            f"  suspected grouped {self.suspected_multiple:>8,}\n"
            f"  LLM calls         {self.llm_calls:>8,}"
        )




def run_batch(
    records: list[BankRecord],
    key_index: KeyIndex,
    key_stats: dict[str, KeyStats],
    audit: AuditLog,
    *,
    run_config: RunConfig,
    blocking: BlockingConfig | None = None,
    gate: GateConfig | None = None,
    ranker: Any = None,
    multiplicity: Any = None,
    prior: Any = None,
    calibrator: Any = None,
    calibrator_kind: str = "none",
    mult_threshold: float = MULT_THRESHOLD,
) -> RunResult:
    """Reconcile a batch, recording every decision as it is made.

Without a ranker every record is queued. This used to fall back to a rules
    pick returning 0.90 for an exact amount and 0.55 otherwise — numbers on a
    different scale from the model's, handed to the same gate as though they
    were comparable. Now that the model's confidence is a calibrated
    probability, that mixture would make one threshold mean two things. Not
    scoring is the honest degradation: it cannot judge, so a person does.
    """

    bcfg = blocking or BlockingConfig()
    gcfg = gate or GateConfig()

    run_id = audit.start_run(run_config)
    result = RunResult(run_id=run_id, n_records=len(records))
    started = time.perf_counter()

    models = Models(ranker, multiplicity, prior, calibrator, calibrator_kind)

    for record in records:
        if ranker is None:
            d = decide(confidence=None, amount_minor=record.amount_minor, config=gcfg)
            audit.record(record.record_id, d, keys=[], n_candidates=0,
                         path="no_ranker", run_id=run_id)
            result.queued += 1
            result.exceptions["no_ranker"] += 1
            continue

        # One matching path. This file used to carry its own copy, which fell
        # behind on calibration, the single-candidate case, the grouping
        # override, and a record that could leave the audit trail entirely.
        try:
            r = match_one(record, index=key_index, key_stats=key_stats,
                          models=models, gate=gcfg, mult_threshold=mult_threshold,
                          blocking=bcfg)
        except Exception as exc:  # noqa: BLE001
            # "Degrades rather than halts" has to be true of the models too.
            # A record that breaks featurising or scoring becomes an exception
            # with a reason, not the end of the batch.
            d = decide(confidence=None, amount_minor=record.amount_minor, config=gcfg)
            audit.record(record.record_id, d, keys=[], n_candidates=0,
                         path="model_error", evidence={"error": type(exc).__name__},
                         run_id=run_id)
            result.queued += 1
            result.exceptions["model_error"] += 1
            continue

        audit.record(record.record_id, r["decision"], keys=r["keys"],
                     n_candidates=r["n_candidates"], path=r["path"],
                     evidence=r["evidence"], run_id=run_id)

        outcome = r["outcome"]
        if outcome == "posted":
            result.posted += 1
        elif outcome == "no_candidate":
            result.no_candidate += 1
            result.exceptions["no_candidate"] += 1
        elif outcome == "suspected_grouped":
            result.suspected_multiple += 1
            result.exceptions["suspected_multiple"] += 1
        else:
            result.queued += 1
            result.exceptions["below_threshold"] += 1

    result.seconds = time.perf_counter() - started
    return result
