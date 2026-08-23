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

Finding *a* balancing subset is the easy half, and on its own it is close to
useless. Measured on ReconRiver, a plain reachability DP solved 89.3% of
settlements and **51.3% of those were the wrong set** -- including credits whose
true batch was a single payment, answered with five unrelated ones that happened
to sum to the same figure. Balancing is not evidence of membership.

Two priors close most of that gap, and both come from how settlement works
rather than from tuning against the answer:

* **Fewest components wins.** Batch sizes are small; pools are not. Each extra
  component multiplies the coincidental subsets available, so the smallest
  balancing subset is by a wide margin the likeliest one. This is a search over
  cardinality, not a heuristic applied afterwards.
* **A tie is not an answer.** If two different smallest subsets both balance,
  the amounts do not distinguish them. That is reported as ``AMBIGUOUS`` and
  routed to review. Resolving it by index order would look like a match and
  carry none of the evidence of one.

Three properties this must have, none of which is optional:

* **Deterministic.** Same input, same subset, every time.
* **Refuses rather than guesses.** ``INFEASIBLE`` and ``AMBIGUOUS`` are real
  answers. An unresolved record belongs in the exception list, not in the ledger
  under a plausible-looking subset.
* **Bounded.** A pool too large or a target too big is refused up front rather
  than allocating an enormous table and hanging.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SolverStatus(str, Enum):
    SOLVED = "solved"
    AMBIGUOUS = "ambiguous"
    """A subset balances, but another subset of the same size balances too."""
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

    max_subset_size: int = 8
    """Largest batch the solver will claim. Beyond this, a balancing subset is
    weak enough evidence that returning it does more harm than returning
    nothing -- the count of coincidental subsets grows combinatorially while
    real batch sizes do not."""

    require_unique: bool = True
    """Refuse when a second subset of the same size reaches the same sum."""


@dataclass(frozen=True, slots=True)
class SolverResult:
    status: SolverStatus
    indices: list[int]
    """Positions in the input pool. Empty unless solved."""
    residual_minor: int
    n_considered: int
    detail: str = ""
    subset_size: int = 0
    """How many payments the answer uses -- and for ``AMBIGUOUS``, how many each
    of the tied answers uses. Zero when there is no answer at all."""


def _reach_by_size(
    amounts: list[int], ceiling: int, max_k: int, mask: int, *, snapshots: bool
) -> tuple[list[int], list[tuple[int, ...]]]:
    """Reachable sums split by how many candidates they use.

    ``layers[k]`` is a bitset whose set bits are the sums reachable using
    exactly *k* candidates. One candidate advances every layer with a single
    shift-or, so a whole DP row is a handful of big-integer ops rather than one
    interpreted step per (candidate, sum) pair.
    """
    layers = [0] * (max_k + 1)
    layers[0] = 1                       # sum 0, using nothing
    snaps: list[tuple[int, ...]] = [tuple(layers)] if snapshots else []

    for amount in amounts:
        if 0 < amount <= ceiling:
            # Descending, so each candidate contributes to at most one layer.
            for k in range(max_k, 0, -1):
                layers[k] = (layers[k] | (layers[k - 1] << amount)) & mask
        if snapshots:
            snaps.append(tuple(layers))

    return layers, snaps


def _pick(layers: list[int], target: int, tolerance: int, ceiling: int) -> tuple[int, int] | None:
    """Choose (sum, size): exact before approximate, then fewest components."""
    for k in range(1, len(layers)):
        if (layers[k] >> target) & 1:
            return target, k
    for gap in range(1, tolerance + 1):
        for s in (target - gap, target + gap):
            if 0 <= s <= ceiling:
                for k in range(1, len(layers)):
                    if (layers[k] >> s) & 1:
                        return s, k
    return None


def solve_subset(
    *,
    target_minor: int,
    candidates_minor: list[int],
    config: SolverConfig | None = None,
) -> SolverResult:
    """Find the smallest subset of *candidates_minor* summing to *target_minor*.

    Returns that subset when it is the only one of its size reaching that sum,
    ``AMBIGUOUS`` when it is not, and ``INFEASIBLE`` when none exists.
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
    mask = (1 << (ceiling + 1)) - 1
    max_k = min(cfg.max_subset_size, n)

    layers, snaps = _reach_by_size(candidates_minor, ceiling, max_k, mask, snapshots=True)
    chosen = _pick(layers, target_minor, cfg.tolerance_minor, ceiling)

    if chosen is None:
        return SolverResult(
            SolverStatus.INFEASIBLE, [], target_minor, n,
            f"no subset of {max_k} or fewer sums to the target within tolerance",
        )
    best_sum, best_k = chosen

    # Walk backwards through the snapshots. Candidate i-1 was needed iff the sum
    # was not already reachable at this size without it.
    indices: list[int] = []
    s, k = best_sum, best_k
    for i in range(n, 0, -1):
        if k == 0:
            break
        if (snaps[i - 1][k] >> s) & 1:
            continue                       # reachable without candidate i-1
        s -= candidates_minor[i - 1]
        k -= 1
        indices.append(i - 1)

    if s != 0 or k != 0:
        return SolverResult(
            SolverStatus.INFEASIBLE, [], target_minor, n, "reconstruction failed"
        )
    indices.sort()

    # Uniqueness. Any competing subset of the same size must omit at least one
    # member of this one, so dropping each member in turn and re-checking is a
    # complete test -- not a sample of one.
    if cfg.require_unique:
        for j in indices:
            without = candidates_minor[:j] + candidates_minor[j + 1:]
            rival, _ = _reach_by_size(without, ceiling, best_k, mask, snapshots=False)
            if (rival[best_k] >> best_sum) & 1:
                return SolverResult(
                    SolverStatus.AMBIGUOUS, [], abs(best_sum - target_minor), n,
                    f"at least two different sets of {best_k} reach this amount",
                    subset_size=best_k,
                )

    return SolverResult(
        status=SolverStatus.SOLVED,
        indices=indices,
        residual_minor=abs(best_sum - target_minor),
        n_considered=n,
        detail=f"{len(indices)} of {n} candidates",
        subset_size=len(indices),
    )
