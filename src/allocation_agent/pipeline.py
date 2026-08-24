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

import numpy as np

from allocation_agent.decide.gate import GateConfig, Outcome, decide
from allocation_agent.match.blocker import BlockingConfig, block
from allocation_agent.match.features import KeyStats, featurise
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
    llm_calls: int = 0

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


def _fallback_choice(record: BankRecord, candidates: list[str],
                     key_stats: dict[str, KeyStats]) -> tuple[str | None, float]:
    """Deterministic pick used when no ranker is available.

    Exact amount first, then nearest date. Confidence is a coarse rule-derived
    value, never a model score, and it is deliberately capped below the
    auto-post threshold for large amounts.
    """
    best, best_score = None, None
    for k in candidates:
        st = key_stats.get(k)
        if st is None:
            continue
        exact = 0 if record.amount_minor in st.amounts else 1
        gap = (min((abs(record.day - d) for d in st.days), default=999)
               if record.day is not None else 999)
        score = (exact, gap)
        if best_score is None or score < best_score:
            best, best_score = k, score
    if best is None:
        return None, 0.0
    confidence = 0.90 if best_score[0] == 0 else 0.55
    return best, confidence


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
    mult_features: Any = None,
    mult_threshold: float = 0.7,
) -> RunResult:
    """Reconcile a batch, recording every decision as it is made."""
    bcfg = blocking or BlockingConfig()
    gcfg = gate or GateConfig()

    run_id = audit.start_run(run_config)
    result = RunResult(run_id=run_id, n_records=len(records))
    started = time.perf_counter()

    for i, record in enumerate(records):
        candidates = sorted(block(record, key_index, bcfg))
        n = len(candidates)

        if not candidates:
            d = decide(confidence=None, amount_minor=record.amount_minor, config=gcfg)
            audit.record(record.record_id, d, keys=[], n_candidates=0, path="blocked")
            result.no_candidate += 1
            result.exceptions["no_candidate"] += 1
            continue

        # does this record look like it spans several keys?
        if multiplicity is not None and mult_features is not None:
            p_mult = float(multiplicity.predict_proba(mult_features[i : i + 1])[0])
            if p_mult >= mult_threshold:
                d = decide(confidence=None, amount_minor=record.amount_minor, config=gcfg)
                audit.record(record.record_id, d, keys=[], n_candidates=n,
                             path="multiplicity",
                             evidence={"p_multiple": round(p_mult, 4)})
                result.suspected_multiple += 1
                result.exceptions["suspected_multiple"] += 1
                continue

        if ranker is not None:
            X = np.vstack([featurise(record, key_stats[k], n_candidates=n)
                           for k in candidates if k in key_stats])
            scores = ranker.score(X)
            order = np.argsort(-scores)
            chosen = candidates[int(order[0])]
            margin = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 1.0
            confidence = float(1.0 / (1.0 + np.exp(-margin)))
            path = "ranked"
            evidence = {"margin": round(margin, 4), "n_scored": len(scores)}
        else:
            chosen, confidence = _fallback_choice(record, candidates, key_stats)
            path = "fallback_rules"
            evidence = {"note": "no ranker available; deterministic rules used"}

        d = decide(confidence=confidence, amount_minor=record.amount_minor, config=gcfg)
        audit.record(record.record_id, d, keys=[chosen] if chosen else [],
                     n_candidates=n, path=path, evidence=evidence)

        if d.outcome is Outcome.POST:
            result.posted += 1
        else:
            result.queued += 1
            result.exceptions["below_threshold"] += 1

    audit.commit()
    audit.finish_run(run_id)
    result.seconds = time.perf_counter() - started
    return result
