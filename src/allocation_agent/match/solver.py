"""Subset-sum group solver.

Many payments settle into one bank credit. Given the credit and a pool of
candidate payments, recover which subset produced it.

Follows the Subset Sum Matching Problem formulation (J.P. Morgan AI Research,
ECAI 2025), which proves the problem NP-complete and finds dynamic programming
outperforms both integer programming and tree search at these instance sizes --
CPLEX times out on all but the smallest, and search dies past ~30 elements.

DP is *pseudo*-polynomial: ``O(n * target)`` in the target's magnitude. That is
usually a caveat and here it is the point, because **money is an integer count of
minor units**. The paper says so directly: "the positive integer case is
important as it directly applies to financial reconciliation."

Three properties this must have, none of which is optional:

* **Deterministic.** Same input, same subset, every time.
* **Refuses rather than guesses.** ``INFEASIBLE`` is a real answer. An
  unresolved record belongs in the exception list, not in the ledger under a
  plausible-looking subset.
* **Bounded.** A pool too large or a target too big is refused up front rather
  than allocating an enormous table and hanging.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SolverStatus(str, Enum):
    SOLVED = "solved"
    INFEASIBLE = "infeasible"
    TOO_LARGE = "too_large"


@dataclass(frozen=True, slots=True)
class SolverConfig:
    tolerance_minor: int = 0
    """Accepted gap between the subset sum and the target. Real settlements
    carry fees and rounding; an exact match is still preferred when one exists."""

    max_candidates: int = 64
    """Pool size beyond which the instance is refused. Refused instances appear
    in the exception list rather than being silently truncated to fit."""

    max_target_minor: int = 5_000_000_000
    """Guards the DP table. The table is ``target + tolerance`` wide."""


@dataclass(frozen=True, slots=True)
class SolverResult:
    status: SolverStatus
    indices: list[int]
    """Positions in the input pool. Empty unless solved."""
    residual_minor: int
    n_considered: int
    detail: str = ""


def solve_subset(
    *,
    target_minor: int,
    candidates_minor: list[int],
    config: SolverConfig | None = None,
) -> SolverResult:
    """Find a subset of *candidates_minor* summing to *target_minor*.

    Returns the exact subset where one exists, otherwise the closest admissible
    one within tolerance, otherwise ``INFEASIBLE``.
    """
    cfg = config or SolverConfig()

    if target_minor < 0:
        raise ValueError(f"target must not be negative, got {target_minor}")
    if any(a < 0 for a in candidates_minor):
        raise ValueError(
            "candidate amounts must not be negative: a refund belongs on the "
            "other side of the equation, not as a negative row"
        )

    n = len(candidates_minor)

    if target_minor == 0:
        return SolverResult(SolverStatus.SOLVED, [], 0, n, "target is zero")

    if n > cfg.max_candidates:
        return SolverResult(
            SolverStatus.TOO_LARGE, [], target_minor, n,
            f"{n} candidates exceeds the cap of {cfg.max_candidates}",
        )
    if target_minor > cfg.max_target_minor:
        return SolverResult(
            SolverStatus.TOO_LARGE, [], target_minor, n,
            f"target {target_minor} exceeds the cap of {cfg.max_target_minor}",
        )
    if not candidates_minor:
        return SolverResult(SolverStatus.INFEASIBLE, [], target_minor, 0, "empty pool")

    ceiling = target_minor + cfg.tolerance_minor

    # Bitset DP. Reachable sums are bits of one big integer, so an entire DP row
    # is a single shift-or: `reach |= reach << amount`. Python's arbitrary-
    # precision ints make this run at C speed on machine words instead of one
    # Python loop iteration per (candidate, sum) pair -- which was ~40M
    # interpreted steps and several seconds per instance.
    mask = (1 << (ceiling + 1)) - 1
    reach = 1                      # only sum 0 is reachable before any candidate
    states: list[int] = [reach]    # snapshot after each candidate, for backtracking

    for amount in candidates_minor:
        if 0 < amount <= ceiling:
            reach = (reach | (reach << amount)) & mask
        states.append(reach)

    # Prefer an exact hit, then the smallest admissible residual.
    best_sum = None
    if (reach >> target_minor) & 1:
        best_sum = target_minor
    elif cfg.tolerance_minor:
        for gap in range(1, cfg.tolerance_minor + 1):
            for s in (target_minor - gap, target_minor + gap):
                if 0 <= s <= ceiling and (reach >> s) & 1:
                    best_sum = s
                    break
            if best_sum is not None:
                break

    if best_sum is None:
        return SolverResult(
            SolverStatus.INFEASIBLE, [], target_minor, n,
            "no subset sums to the target within tolerance",
        )

    # Walk backwards: candidate i was needed iff its sum was not already
    # reachable without it.
    indices: list[int] = []
    s = best_sum
    for i in range(n - 1, -1, -1):
        if s == 0:
            break
        if (states[i] >> s) & 1:
            continue                      # reachable without candidate i
        amount = candidates_minor[i]
        if amount == 0 or amount > s:
            continue
        indices.append(i)
        s -= amount

    if s != 0:
        return SolverResult(
            SolverStatus.INFEASIBLE, [], target_minor, n, "reconstruction failed"
        )

    return SolverResult(
        status=SolverStatus.SOLVED,
        indices=sorted(indices),
        residual_minor=abs(best_sum - target_minor),
        n_considered=n,
        detail=f"{len(indices)} of {n} candidates",
    )
