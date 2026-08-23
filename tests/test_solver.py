"""Subset-sum group solver.

Many payments settle into one bank credit. Given the credit and a pool of
candidate payments, recover which subset produced it.

The formulation follows the Subset Sum Matching Problem (J.P. Morgan AI
Research, ECAI 2025), which proves it NP-complete and finds dynamic programming
beats both integer programming and tree search at these instance sizes. DP is
pseudo-polynomial in the target value — which is genuinely polynomial here,
because money is an integer count of minor units.

Deterministic. Returns *infeasible* rather than a guess: an unresolved record
belongs in the exception list, not in the ledger under a plausible-looking
answer.
"""

import pytest

from allocation_agent.match.solver import SolverConfig, SolverStatus, solve_subset


def solve(target, amounts, **kw):
    return solve_subset(target_minor=target, candidates_minor=list(amounts),
                        config=SolverConfig(**kw))


# --------------------------------------------------------------------------- #
# finding the subset
# --------------------------------------------------------------------------- #

def test_single_candidate_equal_to_the_target():
    r = solve(1000, [1000])
    assert r.status is SolverStatus.SOLVED
    assert r.indices == [0]


def test_two_candidates_summing_to_the_target():
    r = solve(1500, [1000, 500, 9999])
    assert r.status is SolverStatus.SOLVED
    assert sorted(r.indices) == [0, 1]


def test_picks_the_summing_subset_out_of_a_larger_pool():
    r = solve(3300, [1200, 700, 2100, 5000, 800])
    assert r.status is SolverStatus.SOLVED
    assert sum([1200, 700, 2100, 5000, 800][i] for i in r.indices) == 3300


def test_reports_infeasible_rather_than_guessing():
    r = solve(777, [100, 200, 300])
    assert r.status is SolverStatus.INFEASIBLE
    assert r.indices == []


def test_zero_target_selects_nothing():
    r = solve(0, [100, 200])
    assert r.status is SolverStatus.SOLVED
    assert r.indices == []


def test_empty_pool_with_a_nonzero_target_is_infeasible():
    assert solve(100, []).status is SolverStatus.INFEASIBLE


# --------------------------------------------------------------------------- #
# tolerance — real settlements carry fees and rounding
# --------------------------------------------------------------------------- #

def test_a_gap_within_tolerance_is_accepted():
    r = solve(1002, [1000], tolerance_minor=5)
    assert r.status is SolverStatus.SOLVED
    assert r.residual_minor == 2


def test_a_gap_outside_tolerance_is_not():
    assert solve(1050, [1000], tolerance_minor=5).status is SolverStatus.INFEASIBLE


def test_an_exact_match_is_preferred_over_one_within_tolerance():
    """Both are admissible; the exact one is the right answer."""
    r = solve(1000, [1000, 998], tolerance_minor=5)
    assert r.indices == [0]
    assert r.residual_minor == 0


def test_residual_is_reported_so_it_can_be_diagnosed():
    r = solve(1003, [1000], tolerance_minor=10)
    assert r.residual_minor == 3


# --------------------------------------------------------------------------- #
# guards — an unbounded solver on a large pool is a hang, not a feature
# --------------------------------------------------------------------------- #

def test_a_pool_larger_than_the_cap_is_refused_not_silently_truncated():
    r = solve(100, list(range(1, 60)), max_candidates=40)
    assert r.status is SolverStatus.TOO_LARGE
    assert r.indices == []


def test_the_cap_is_inclusive():
    r = solve(3, [1, 2] + [99] * 38, max_candidates=40)
    assert r.status is SolverStatus.SOLVED


def test_an_oversized_target_is_refused_rather_than_allocating_a_huge_table():
    r = solve(10**12, [1, 2, 3], max_target_minor=10**7)
    assert r.status is SolverStatus.TOO_LARGE


def test_negative_amounts_are_rejected():
    """Refunds belong on the other side of the equation, not as negative rows."""
    with pytest.raises(ValueError, match="negative"):
        solve(100, [50, -50])


def test_negative_target_is_rejected():
    with pytest.raises(ValueError, match="negative"):
        solve(-100, [50])


# --------------------------------------------------------------------------- #
# determinism and reporting
# --------------------------------------------------------------------------- #

def test_same_input_gives_the_same_subset():
    pool = [300, 700, 1000, 200, 800]
    a = solve(1000, pool)
    b = solve(1000, pool)
    assert a.indices == b.indices


def test_result_reports_how_many_candidates_it_considered():
    r = solve(1500, [1000, 500, 9999])
    assert r.n_considered == 3


def test_solved_result_is_arithmetically_consistent():
    pool = [1200, 700, 2100, 5000, 800]
    r = solve(3300, pool)
    assert abs(sum(pool[i] for i in r.indices) - 3300) == r.residual_minor


def test_indices_are_unique():
    pool = [500, 500, 500]
    r = solve(1000, pool)
    assert len(set(r.indices)) == len(r.indices)


def test_a_candidate_is_used_at_most_once():
    """One payment cannot settle twice."""
    r = solve(1000, [500])
    assert r.status is SolverStatus.INFEASIBLE
