"""Blocking — cut the candidate space without losing the answer.

Comparing every bank record against every allocation key is 190,717 x 103,191 =
~19.7 billion pairs. Blocking never creates those pairs: it looks up the small
set of keys that could plausibly match and ignores the rest.

Two predicates, unioned rather than cascaded:

* **amount**   ``(account, amount_minor)`` -- 91.0% recall, ~3.7 candidates.
  Most bank records match one ledger row exactly.
* **date**     ``(account, day +/- slack)`` -- catches records whose amount
  matches nothing exactly: fees deducted, partial payments, rounding.

Measured on 169,168 labelled records:

===========================  ========  ================
strategy                       recall    mean candidates
===========================  ========  ================
account only                   100.0%             2804.4
amount only                     90.9%                3.7
amount + date +/-3d             97.6%               20.9
**amount + date +/-7d**       **98.9%**          **41.4**
amount + date +/-14d            99.3%               75.9
===========================  ========  ================

The default is +/-7 days: a 68x reduction against account-only, for 1.1% of
recall. Widen it if recall matters more than throughput; the cost is roughly
linear in the window.

Recall here is a ceiling on everything downstream. A key dropped at this stage
cannot be recovered by any amount of scoring.
"""

from __future__ import annotations

from dataclasses import dataclass

from allocation_agent.stores.keys import KeyIndex
from allocation_agent.types import BankRecord


@dataclass(frozen=True, slots=True)
class BlockingConfig:
    """Blocking parameters. Approved before a run, never changed during one."""

    date_slack_days: int = 7
    use_amount: bool = True
    use_date: bool = True

    def __post_init__(self) -> None:
        if self.date_slack_days < 0:
            raise ValueError(f"date_slack_days must be >= 0, got {self.date_slack_days}")


def block(
    record: BankRecord,
    index: KeyIndex,
    config: BlockingConfig | None = None,
) -> set[str]:
    """Return the allocation keys worth scoring against *record*.

    Returns an empty set rather than the whole index when the record cannot be
    blocked. Widening the search on missing data would turn one unusable record
    into a full scan.
    """
    cfg = config or BlockingConfig()

    if record.account is None:
        return set()

    candidates: set[str] = set()

    if cfg.use_amount:
        candidates |= index.by_amount.get((record.account, record.amount_minor), set())

    if cfg.use_date and record.day is not None:
        for day in range(record.day - cfg.date_slack_days, record.day + cfg.date_slack_days + 1):
            candidates |= index.by_day.get((record.account, day), set())

    return candidates
