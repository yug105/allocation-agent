"""Is one number enough for the lone exact-amount rule?

An aggregate can hide a subgroup. `DIRECT_CONFIDENCE` is a single constant
applied to every record the rule fires on, so this breaks its accuracy down by
amount, date gap and how many candidates were competing — the axes on which a
reconciliation rule would plausibly fail differently.

It also reports the gap that matters most: the population the figure was
measured on is not the population the branch applies it to.
"""

from __future__ import annotations

import collections

from allocation_agent.api import BLOCKING, _State
from allocation_agent.match.blocker import block

BUCKETS = [(0, 100_000, "< 1,000"), (100_000, 1_000_000, "1k - 10k"),
           (1_000_000, 10_000_000, "10k - 100k"), (10_000_000, 10**12, "> 100k")]


def main() -> None:
    state = _State()
    state.load()
    if not state.ready:
        raise SystemExit(state.error)

    by_amount = collections.defaultdict(lambda: [0, 0])
    by_gap = collections.defaultdict(lambda: [0, 0])
    by_ncand = collections.defaultdict(lambda: [0, 0])
    measured = correct = fires = 0

    for rec in state.records:
        usable = [k for k in sorted(block(rec, state.index, BLOCKING))
                  if k in state.key_stats]
        if not usable:
            continue
        exact = [k for k in usable if rec.amount_minor in state.key_stats[k].amounts]
        if len(exact) != 1:
            continue
        truth, is_mult = state.truth.get(rec.record_id, ("", False))
        ok = int((not is_mult) and exact[0] == truth)
        measured += 1
        correct += ok
        fires += len(usable) == 1

        amount = abs(rec.amount_minor)
        for lo, hi, label in BUCKETS:
            if lo <= amount < hi:
                by_amount[label][0] += 1
                by_amount[label][1] += ok
                break

        days = state.key_stats[exact[0]].days
        gap = (min((abs(rec.day - d) for d in days), default=99)
               if rec.day is not None else 99)
        key = "same day" if gap == 0 else ("1-3 days" if gap <= 3 else "4-7 days")
        by_gap[key][0] += 1
        by_gap[key][1] += ok

        n = len(usable)
        key = ("1 (this branch)" if n == 1 else
               "4-10" if n <= 10 else "11-50" if n <= 50 else "50+")
        by_ncand[key][0] += 1
        by_ncand[key][1] += ok

    print(f"taking the lone exact-amount candidate: {correct:,}/{measured:,} "
          f"= {correct / measured * 100:.2f}%\n")
    for title, data, order in (
        ("by amount", by_amount, [b[2] for b in BUCKETS]),
        ("by date gap", by_gap, ["same day", "1-3 days", "4-7 days"]),
        ("by candidates blocked", by_ncand,
         ["1 (this branch)", "4-10", "11-50", "50+"]),
    ):
        print(f"  {title}:")
        for label in order:
            n, ok = data[label]
            if n:
                flag = "   <-- below the 0.85 base bar" if ok / n < 0.85 else ""
                print(f"    {label:<16}{n:>6,}  {ok / n * 100:>6.2f}%{flag}")
        print()

    print(f"records the direct branch actually fires on: {fires:,}")
    if not fires:
        print("  none — it needs exactly one candidate, and blocking never returns")
        print("  fewer than four here. So the constant is an extrapolation from the")
        print("  trend above rather than a measurement of this branch.")


if __name__ == "__main__":
    main()
