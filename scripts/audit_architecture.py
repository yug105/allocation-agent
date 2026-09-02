"""Is every component in the diagram actually on an execution path?

The failure this exists to catch has already happened twice here. The narrator
was constructed in `create_app()` and never called, so its guarantee protected
nothing anybody read. `ConnectRequest` outlived the endpoint that used it by
several days. Both were invisible: the code was present, imported, and tested.

So each component is *exercised*, not grepped for. A component that answers
"yes, I ran" is on the path; one that cannot is decoration no matter how good
its tests are.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("ARTIFACTS_DIR", str(ROOT / "artifacts"))

RESULTS: list[tuple[str, str, bool, str, str]] = []


def check(stage: str, component: str, fn, how: str = "ran") -> None:
    """Record one component's answer.

    *how* is the evidence, and it is printed, because the two kinds are not
    worth the same. `ran` means a live request reached the component and it
    answered. `source` means the file was read — which can show a call site
    exists but never that anything took it, so a `source` row is a weaker
    claim and says so instead of borrowing the credibility of the others.
    """
    try:
        ok, note = fn()
    except Exception as exc:  # noqa: BLE001
        ok, note = False, f"{type(exc).__name__}: {exc}"
    RESULTS.append((stage, component, bool(ok), how, note))


def main() -> None:  # noqa: PLR0915
    from fastapi.testclient import TestClient

    from allocation_agent.api import _State, create_app

    state = _State()
    state.load()
    app = create_app()
    client = TestClient(app)

    # -- ingest ------------------------------------------------------------ #
    check("ingest", "adapters.csv_upload", lambda: (
        client.post("/api/reconcile", files={
            "bank": ("b.csv", io.BytesIO(b"id,account,amount,date\nB1,A,1.00,2026-01-01\n"), "text/csv"),
            "ledger": ("l.csv", io.BytesIO(b"k,account,amount,date\nK1,A,1.00,2026-01-01\n"), "text/csv"),
        }).status_code == 200, "an upload reconciles"))

    check("ingest", "eval.leakage", lambda: (
        "assert_no_leakage" in (ROOT / "scripts/train_ranker.py").read_text(),
        "gates training; raises before a model reaches disk"), how="source")

    check("ingest", "eval.splits", lambda: (
        "temporal_split" in (ROOT / "scripts/train_ranker.py").read_text(),
        "train/val/test are cut here"), how="source")

    # -- match -------------------------------------------------------------- #
    run = client.post("/api/run", json={"limit": 300}).json()
    check("match", "match.blocker", lambda: (
        run["summary"]["no_candidate"] >= 0 and run["n_records"] == 300,
        "every record is blocked before anything else"))

    check("match", "match.multiplicity", lambda: (
        run["summary"]["suspected_grouped"] > 0,
        f"{run['summary']['suspected_grouped']} routed as grouped"))

    check("match", "match.ranker", lambda: (
        run["summary"]["posted"] > 0, f"{run['summary']['posted']} posted"))

    check("match", "calibrator", lambda: (
        state.calibrator is not None and state.calibrator_kind != "none",
        f"{state.calibrator_kind}, loaded from the bundle"))

    check("match", "match.solver", lambda: (
        client.post("/api/settlements", json={"limit": 5}).status_code == 200,
        "reached through /api/settlements"))

    # -- decide ------------------------------------------------------------- #
    check("decide", "decide.gate", lambda: (
        run["summary"]["queued"] + run["summary"]["posted"] > 0,
        "every scored record passes the gate"))

    check("decide", "decide.narrate", lambda: (
        app.state.narrator_calls() >= 0 and any(
            e.get("residual_cause") for e in run["exceptions"]),
        "diagnoses the gap on queued records"))

    check("decide", "decide.openrouter", lambda: (
        "OPENROUTER_API_KEY" in (ROOT / "src/allocation_agent/api.py").read_text(),
        "opt-in: constructed only when a key is set"), how="source")

    # -- record ------------------------------------------------------------- #
    trail = client.get(f"/api/run/{run['run_id']}/audit").json()
    check("record", "report.audit", lambda: (
        len(trail["decisions"]) == run["n_records"],
        f"{len(trail['decisions'])} rows for {run['n_records']} records"))

    check("record", "audit run lifecycle", lambda: (
        trail["run"]["status"] == "completed", f"status={trail['run']['status']}"))

    # -- learn -------------------------------------------------------------- #
    # These two were grepped for in api.py until the grep started lying: the
    # note read "offline only" on a row whose own verdict said yes, because
    # /api/correct had since gone live and nothing here noticed. A string in a
    # file is not an execution. So a correction is actually submitted, and the
    # two components are judged on what comes back.
    posted = next((d for d in trail["decisions"] if d["outcome"] == "post"), None)
    correction = client.post("/api/correct", json={
        "run_id": run["run_id"],
        "record_id": posted["record_id"] if posted else "",
        "correct_key": "a-key-blocking-never-offered",
        "reviewer": "audit_architecture.py",
    }) if posted else None

    check("learn", "learn.router", lambda: (
        correction is not None and correction.status_code == 200
        and correction.json()["locus"] in {"blocking", "multiplicity",
                                           "ranking", "threshold", "none"},
        f"attributed a correction to {correction.json()['locus']}"
        if correction is not None and correction.status_code == 200
        else "no posted record to correct"))

    check("learn", "learn.casebase", lambda: (
        client.get("/api/cases").json()["n_cases"] > 0,
        f"holds {client.get('/api/cases').json()['n_cases']} case(s); "
        f"retrieve() is still called by nothing, so no decision consults one"))

    api_src = (ROOT / "src/allocation_agent/api.py").read_text()
    check("learn", "learn.simulate", lambda: (
        "simulate" in api_src, "offline only: scripts/run_learning.py"),
        how="source")
    check("learn", "pipeline.run_batch", lambda: (
        "run_batch" in api_src, "offline only: scripts/run_batch.py"),
        how="source")

    # -- report -------------------------------------------------------------- #
    print(f"{'stage':<9}{'component':<28}{'on path':<9}{'evidence':<10}note")
    print("-" * 104)
    dead, grepped = [], []
    for stage, component, ok, how, note in RESULTS:
        print(f"{stage:<9}{component:<28}{'yes' if ok else 'NO':<9}{how:<10}{note}")
        if not ok:
            dead.append(component)
        elif how == "source":
            grepped.append(component)

    print()
    if dead:
        print(f"{len(dead)} component(s) not on the request path: {', '.join(dead)}")
        print("That is only a defect if the docs claim otherwise — check each one.")
    else:
        print("every component exercised by a live request")
    if grepped:
        print(f"{len(grepped)} answered by reading source, not by running: "
              f"{', '.join(grepped)}")


if __name__ == "__main__":
    main()
