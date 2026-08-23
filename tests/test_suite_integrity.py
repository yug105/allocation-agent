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
