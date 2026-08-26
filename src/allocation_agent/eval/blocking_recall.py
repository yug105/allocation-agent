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
    """Two populations, kept apart.

    `recall` can only be measured on single-key records — a grouped one has no
    single true key to find. Candidate counts are a property of every call, so
    the `_evaluated` variants report the same statistic over the rows recall
    actually covers. Reporting one median beside `n_evaluated` described two
    different sets of rows in one sentence: 38 against 28 on the held-out set.
    """

    recall: float
    n_records: int
    n_evaluated: int
    n_hit: int
    mean_candidates: float
    median_candidates: float
    mean_candidates_evaluated: float
    median_candidates_evaluated: float
    p95_candidates: float
    max_candidates: int
    config: BlockingConfig

    def __str__(self) -> str:
        return (
            f"recall {self.recall * 100:6.2f}%  "
            f"({self.n_hit:,}/{self.n_evaluated:,} single-key)   "
            f"candidates median {self.median_candidates_evaluated:6.0f} "
            f"on those, {self.median_candidates:.0f} over all "
            f"{self.n_records:,}   p95 {self.p95_candidates:.0f} "
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
    sizes: list[int] = []           # every record: what blocking does per call
    eval_sizes: list[int] = []      # only the rows recall is measured on

    for record, label, is_mult in zip(dataset.records, dataset.labels,
                                      dataset.is_mult, strict=False):
        candidates = block(record, idx, cfg)
        sizes.append(len(candidates))
        if is_mult:
            # A grouped record has no single true key, so it cannot hit or
            # miss. It still costs candidates, which is why it stays in
            # `sizes` and is kept out of `eval_sizes`.
            continue
        evaluated += 1
        eval_sizes.append(len(candidates))
        if label in candidates:
            hits += 1

    arr = np.array(sizes) if sizes else np.array([0])
    ev = np.array(eval_sizes) if eval_sizes else np.array([0])
    return BlockingReport(
        recall=hits / evaluated if evaluated else 0.0,
        n_records=len(sizes),
        n_evaluated=evaluated,
        n_hit=hits,
        median_candidates_evaluated=float(np.median(ev)),
        mean_candidates_evaluated=float(ev.mean()),
        mean_candidates=float(arr.mean()),
        median_candidates=float(np.median(arr)),
        p95_candidates=float(np.percentile(arr, 95)),
        max_candidates=int(arr.max()),
        config=cfg,
    )
