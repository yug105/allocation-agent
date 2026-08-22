"""Blocking recall — measure this before building anything downstream.

A key dropped during blocking cannot be recovered by any amount of scoring, so
this number is a ceiling on every accuracy figure that follows. Candidate-set
size is what the recall costs.

Only single-key records are scoreable here: a ``MULT`` record has no single
correct key to look for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from allocation_agent.adapters.benchrec import Dataset
from allocation_agent.match.blocker import BlockingConfig, block
from allocation_agent.stores.keys import KeyIndex


@dataclass(frozen=True, slots=True)
class BlockingReport:
    recall: float
    n_evaluated: int
    n_hit: int
    mean_candidates: float
    median_candidates: float
    p95_candidates: float
    max_candidates: int
    config: BlockingConfig

    def __str__(self) -> str:
        return (
            f"recall {self.recall * 100:6.2f}%  "
            f"({self.n_hit:,}/{self.n_evaluated:,})   "
            f"candidates median {self.median_candidates:6.0f} "
            f"p95 {self.p95_candidates:7.0f} mean {self.mean_candidates:7.1f}"
        )


def measure_blocking(
    dataset: Dataset,
    config: BlockingConfig | None = None,
    index: KeyIndex | None = None,
) -> BlockingReport:
    """Fraction of single-key records whose true key survives blocking."""
    cfg = config or BlockingConfig()
    idx = index or KeyIndex(dataset.key_rows)

    hits = 0
    evaluated = 0
    sizes: list[int] = []

    for record, label, is_mult in zip(dataset.records, dataset.labels, dataset.is_mult):
        candidates = block(record, idx, cfg)
        sizes.append(len(candidates))
        if is_mult:
            continue
        evaluated += 1
        if label in candidates:
            hits += 1

    arr = np.array(sizes) if sizes else np.array([0])
    return BlockingReport(
        recall=hits / evaluated if evaluated else 0.0,
        n_evaluated=evaluated,
        n_hit=hits,
        mean_candidates=float(arr.mean()),
        median_candidates=float(np.median(arr)),
        p95_candidates=float(np.percentile(arr, 95)),
        max_candidates=int(arr.max()),
        config=cfg,
    )
