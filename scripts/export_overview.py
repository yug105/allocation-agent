"""Precompute the figures the page shows before anyone clicks anything.

A judge scans above the fold for about thirty seconds and forms an opinion.
Until now that surface held a title, a paragraph and three dataset counts --
nothing had happened yet, and nothing could until they pressed a button.

Running a batch at startup instead is not an option: 4,000 records take ~90s on
the deployed free instance and the health check would fail before the container
was ready. So it is computed here, over the **whole** held-out set rather than
whatever slice a live run would reach, and written to `artifacts/overview.json`.

Everything is in value, not counts. Every reconciliation product measures
unreconciled balance and value at risk, because "1,535 records" is a statistic
and "6.3M sitting in review" is a reason to care. The amounts are BenchRec's
obfuscated units, which is stated on the page rather than implied to be a
currency.
"""

from __future__ import annotations

import collections
import json
import time
from pathlib import Path

from allocation_agent.api import BLOCKING, MULT_THRESHOLD, _match_one, _State
from allocation_agent.decide.gate import GateConfig

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    state = _State()
    state.load()
    if not state.ready:
        raise SystemExit(f"artifacts not loaded: {state.error}")

    gate = GateConfig()
    value = collections.defaultdict(float)
    count = collections.Counter()
    posted_correct = 0
    exceptions: list[tuple[float, str, str]] = []
    started = time.perf_counter()

    for rec in state.records:
        r = _match_one(state, rec, state.index, state.key_stats, gate, MULT_THRESHOLD)
        amount = abs(rec.amount_minor) / 100
        outcome = r["outcome"]
        value[outcome] += amount
        count[outcome] += 1
        if outcome == "posted":
            truth, is_mult = state.truth.get(rec.record_id, ("", False))
            posted_correct += (not is_mult) and r["keys"][0] == truth
        else:
            exceptions.append((amount, rec.record_id, outcome))

    total = sum(value.values())
    queue = total - value["posted"]
    exceptions.sort(reverse=True)
    top10 = sum(a for a, _, _ in exceptions[:10])

    overview = {
        "n_records": len(state.records),
        "seconds": round(time.perf_counter() - started, 1),
        "blocking_date_slack_days": BLOCKING.date_slack_days,
        "mult_threshold": MULT_THRESHOLD,
        "total_value": round(total, 2),
        "posted_value": round(value["posted"], 2),
        "queue_value": round(queue, 2),
        "queue_share_of_value": queue / total if total else 0.0,
        "counts": dict(count),
        "value_by_outcome": {k: round(v, 2) for k, v in value.items()},
        # The whole argument, in money: the grouped case is the one the bank's
        # own rules engine resolved 0% of, and it is where the queue's value is.
        "grouped_share_of_queue": value["suspected_grouped"] / queue if queue else 0.0,
        "precision_of_posted": posted_correct / count["posted"] if count["posted"] else 0.0,
        "straight_through_rate": count["posted"] / len(state.records),
        # A controller works the queue top-down by value, so how concentrated it
        # is decides whether that is ten clicks or four hundred.
        "top10_share_of_queue": top10 / queue if queue else 0.0,
        "largest_exception": round(exceptions[0][0], 2) if exceptions else 0.0,
        "median_exception": round(exceptions[len(exceptions) // 2][0], 2) if exceptions else 0.0,
    }

    out = ROOT / "artifacts" / "overview.json"
    out.write_text(json.dumps(overview, indent=2) + "\n")

    print(f"{overview['n_records']:,} held-out records in {overview['seconds']}s\n")
    print(f"  posted automatically   {overview['posted_value']:>16,.2f}   "
          f"{overview['straight_through_rate'] * 100:.1f}% of records, "
          f"{overview['precision_of_posted'] * 100:.1f}% of them right")
    print(f"  needs a person         {overview['queue_value']:>16,.2f}   "
          f"{overview['queue_share_of_value'] * 100:.1f}% of the value")
    for k, v in sorted(overview["value_by_outcome"].items(), key=lambda kv: -kv[1]):
        if k != "posted":
            print(f"    {k:<20}{count[k]:>5} items {v:>16,.2f}")
    print(f"\n  grouped share of the queue's value: "
          f"{overview['grouped_share_of_queue'] * 100:.1f}%")
    print(f"  top 10 exceptions are {overview['top10_share_of_queue'] * 100:.0f}% "
          f"of the queue's value")
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
