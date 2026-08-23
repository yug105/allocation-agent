"""Ablation for the group solver, on ReconRiver.

Three variants, each the whole algorithm rather than a flag on one algorithm.
The point of the table is that *coverage alone is a misleading number*: the
first variant answers more credits than the third and is wrong far more often,
and only reporting both columns makes that visible.

Ground truth is the recorded settlement batch. The batch identifier is never
passed to the solver -- it sees an amount and a pool.
"""

from __future__ import annotations

import json
import statistics as stx
import time
from pathlib import Path

from allocation_agent.match.solver import SolverConfig, SolverStatus, solve_subset

ROOT = Path(__file__).resolve().parents[1]


def legacy_reachability(target: int, amounts: list[int]) -> list[int]:
    """What shipped first: one bitset of reachable sums, backtracked by index.

    Finds *a* subset. Nothing in it prefers a likely subset over an unlikely
    one, which is the whole defect -- it answered a single-payment credit with
    five unrelated payments that happened to add up.
    """
    mask = (1 << (target + 1)) - 1
    reach, states = 1, [1]
    for a in amounts:
        if 0 < a <= target:
            reach = (reach | (reach << a)) & mask
        states.append(reach)
    if not (reach >> target) & 1:
        return []
    out, s = [], target
    for i in range(len(amounts) - 1, -1, -1):
        if s == 0:
            break
        if (states[i] >> s) & 1:
            continue
        a = amounts[i]
        if a == 0 or a > s:
            continue
        out.append(i)
        s -= a
    return sorted(out) if s == 0 else []


def main() -> None:
    data = json.loads((ROOT / "artifacts" / "reconriver.json").read_text())
    pay = {p["payment_id"]: p for p in data["payments"]}
    sett = data["settlements"]

    print(f"{len(sett)} settlements | pool median "
          f"{stx.median(len(s['pool']) for s in sett):.0f} | batch size median "
          f"{stx.median(len(s['truth']) for s in sett):.0f} "
          f"max {max(len(s['truth']) for s in sett)}\n")
    print(f"{'':32}{'coverage':>10}{'precision':>11}{'wrong':>8}{'ambig':>7}{'p50ms':>8}{'p99ms':>8}")

    variants = [
        ("reachability DP, first subset", None),
        ("smallest subset", SolverConfig(max_candidates=128, require_unique=False)),
        ("smallest, refuse ties (shipped)", SolverConfig(max_candidates=128)),
    ]

    for label, cfg in variants:
        exact = solved = ambiguous = wrong = 0
        lat: list[float] = []
        for s in sett:
            ids = [p for p in s["pool"] if p in pay]
            amounts = [pay[p]["amount_minor"] for p in ids]
            t0 = time.perf_counter()
            if cfg is None:
                idx = legacy_reachability(s["amount_minor"], amounts)
                status = SolverStatus.SOLVED if idx else SolverStatus.INFEASIBLE
            else:
                r = solve_subset(target_minor=s["amount_minor"],
                                 candidates_minor=amounts, config=cfg)
                idx, status = r.indices, r.status
            lat.append((time.perf_counter() - t0) * 1000)

            if status is SolverStatus.SOLVED:
                solved += 1
                if sorted(ids[i] for i in idx) == sorted(s["truth"]):
                    exact += 1
                else:
                    wrong += 1
            elif status is SolverStatus.AMBIGUOUS:
                ambiguous += 1

        n = len(sett)
        lat.sort()
        print(f"{label:32}{exact / n * 100:9.1f}%"
              f"{(exact / solved * 100 if solved else 0):10.1f}%"
              f"{wrong / n * 100:7.1f}%{ambiguous / n * 100:6.1f}%"
              f"{lat[len(lat) // 2]:8.1f}{lat[int(len(lat) * 0.99)]:8.1f}")

    print("\ncoverage  = credits given an answer that was the recorded batch")
    print("precision = of the answers it gave, how many were the recorded batch")
    print("wrong     = answers that balance but are not the recorded batch")


if __name__ == "__main__":
    main()
