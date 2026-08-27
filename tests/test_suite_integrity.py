"""The test suite's own defects.

A test that does not run is worse than a missing one: it reports success. Two
kinds have already happened in this repo and both were invisible.
"""

from __future__ import annotations

import ast
import collections
import pathlib

TESTS = sorted(pathlib.Path(__file__).parent.glob("test_*.py"))


def test_no_test_name_is_defined_twice():
    """A second definition silently replaces the first, so the first never runs
    and never fails. `test_the_narrator_is_actually_invoked` existed twice --
    the shadowed copy was the one asserting the real behaviour.
    """
    dupes = {}
    for path in TESTS:
        names = [n.name for n in ast.parse(path.read_text()).body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name.startswith("test_")]
        repeated = [n for n, c in collections.Counter(names).items() if c > 1]
        if repeated:
            dupes[path.name] = repeated
    assert not dupes, f"shadowed test definitions: {dupes}"


def test_every_test_asserts_something():
    """A test body with no assert passes unconditionally.

    Calling a function whose contract is to raise counts -- `assert_no_leakage`
    and `validate_numbers` are assertions with a different spelling, and the
    tests that call them bare are checking they stay silent on valid input.
    """
    barren = []
    for path in TESTS:
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test_")):
                continue
            has = any(isinstance(n, ast.Assert) for n in ast.walk(node))
            calls = {getattr(n.func, "attr", getattr(n.func, "id", ""))
                     for n in ast.walk(node) if isinstance(n, ast.Call)}
            raising = {c for c in calls
                       if c.startswith("assert_") or c.startswith("validate_")}
            if not has and not ({"raises", "warns", "approx"} & calls) and not raising:
                barren.append(f"{path.name}::{node.name}")
    assert not barren, f"tests with no assertion: {barren}"


# --------------------------------------------------------------------------- #
# The README's guard table is a list of claims about the code. This repo's
# recurring defect is a claim nothing checks, and a table of them at the top of
# the README is the most-read place for one to rot.
# --------------------------------------------------------------------------- #

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Each guard the README promises, and the code that has to exist for it to hold.
GUARDS = {
    "outcome columns refused": ("src/allocation_agent/adapters/benchrec.py", "_OUTCOME_COLUMNS"),
    "leakage gate in training": ("scripts/train_ranker.py", "assert_no_leakage"),
    "renamed-label check": ("src/allocation_agent/eval/leakage.py", "LEAK_MI_THRESHOLD"),
    "calibrated confidence": ("src/allocation_agent/match/engine.py", "calibrator"),
    "amount-scaled threshold": ("src/allocation_agent/decide/gate.py", "log10"),
    "uniqueness test": ("src/allocation_agent/match/solver.py", "require_unique"),
    "model failure degrades": ("src/allocation_agent/api.py", "_match_or_degrade"),
    "narration cannot unmake": ("src/allocation_agent/match/engine.py", "narration_failed"),
    "figures by value": ("src/allocation_agent/decide/narrate.py", "_as_number"),
    "append-only": ("src/allocation_agent/report/audit.py", "decisions_no_update"),
    "failed run state": ("src/allocation_agent/report/audit.py", "def fail_run"),
    "integer money": ("src/allocation_agent/types.py", "isinstance"),
    "precision refused": ("src/allocation_agent/adapters/csv_upload.py", "precision"),
    "decimal comma": ("src/allocation_agent/adapters/csv_upload.py", "_DELIMITERS"),
}


def test_every_guard_the_readme_promises_still_exists():
    missing = [name for name, (path, needle) in GUARDS.items()
               if needle not in (ROOT / path).read_text()]
    assert not missing, f"README claims guards that are gone: {missing}"


def test_the_gate_figures_quoted_in_the_readme_are_current():
    """The table states 0.90 posts at 10,000 and queues at 1,000,000."""
    from allocation_agent.decide.gate import GateConfig
    g = GateConfig()
    assert g.threshold_for(1_000_000) <= 0.90, "0.90 no longer posts at 10,000"
    assert g.threshold_for(100_000_000) > 0.90, "0.90 now posts at 1,000,000"


def test_the_calibration_figures_quoted_in_the_readme_are_current():
    """The table cites sigmoid claiming 74.8% where the truth was 21.5%."""
    readme = (ROOT / "README.md").read_text()
    assert "74.8%" in readme and "21.5%" in readme
    # Both must also appear in the section that produced them, or the headline
    # figure has drifted from its own evidence.
    assert readme.count("74.8%") >= 2, "the guard table cites a figure the results no longer show"
