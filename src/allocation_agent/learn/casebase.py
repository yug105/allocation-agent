"""Case base — precedent a reviewer and an auditor can read.

Retrieval by similarity within the same failure locus. A retrieved case is
*evidence*, not a rule: it says "a reviewer resolved a situation like this, on
this date, this way". A model weight cannot answer "why did you do that"; a
cited precedent can.

**Selective retention is the part people skip.** Store every correction and the
base fills with near-duplicates, retrieval degrades, and the autonomy curve
becomes a measure of how big the base got rather than of anything learned. A
near-duplicate bumps a counter; an uncertain human label is refused; a case that
stops working is retired and the retirement is recorded, because a decaying case
is a signal that the world changed.

Honest note: retraining alone would probably carry the accuracy. This layer earns
its place on explainability. If something must be cut for time, cut this first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Case:
    case_id: str
    situation: np.ndarray
    locus: str
    resolution: list[str]
    confirmations: int = 0
    applications: int = 0
    correct: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.applications if self.applications else 1.0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class CaseBase:
    """In-memory during a run; persisted to the state store between runs."""

    def __init__(
        self,
        *,
        similarity_threshold: float = 0.85,
        duplicate_threshold: float = 0.95,
        max_cases: int = 50_000,
        min_applications: int = 3,
        retire_below_accuracy: float = 0.9,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.duplicate_threshold = duplicate_threshold
        self.max_cases = max_cases
        self.min_applications = min_applications
        self.retire_below_accuracy = retire_below_accuracy
        self.cases: list[Case] = []
        self.retired: set[str] = set()

    def __len__(self) -> int:
        return len(self.cases)

    def retain(self, case: Case, *, human_certain: bool = True) -> bool:
        """Store a case, or decline and say so by returning ``False``."""
        if not human_certain:
            return False

        for existing in self.cases:
            if existing.locus != case.locus:
                continue
            if _cosine(existing.situation, case.situation) >= self.duplicate_threshold:
                existing.confirmations += 1
                return False

        self.cases.append(case)
        if len(self.cases) > self.max_cases:
            # evict the least-confirmed; retention is competitive, not first-come.
            self.cases.sort(key=lambda c: (c.confirmations, c.applications))
            dropped = self.cases.pop(0)
            self.retired.add(dropped.case_id)
        return True

    def retrieve(self, situation: np.ndarray, *, locus: str, k: int = 5) -> list[Case]:
        """Cases similar enough to be worth showing, most similar first."""
        scored = [
            (_cosine(c.situation, situation), c)
            for c in self.cases
            if c.locus == locus and c.case_id not in self.retired
        ]
        hits = [(s, c) for s, c in scored if s >= self.similarity_threshold]
        hits.sort(key=lambda sc: -sc[0])
        return [c for _, c in hits[:k]]

    def record_outcome(self, case_id: str, *, correct: bool) -> None:
        """Record whether applying a case turned out right, and retire it if not."""
        for c in self.cases:
            if c.case_id != case_id:
                continue
            c.applications += 1
            c.correct += int(correct)
            if c.applications >= self.min_applications and c.accuracy < self.retire_below_accuracy:
                self.retired.add(case_id)
            return
