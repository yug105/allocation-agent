"""The deployed service.

A judge opens a link and must see a real result without configuring anything.
That is the whole requirement, so these tests pin it: the demo runs on held-out
records, every record is accounted for, and nothing crashes on bad input.
"""

import io

import pytest
from fastapi.testclient import TestClient

from allocation_agent.api import create_app


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


def test_root_serves_a_page(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_health_is_cheap_and_says_whether_models_loaded(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert "models_loaded" in r.json()


def test_meta_reports_the_data_provenance(client):
    m = client.get("/api/meta").json()
    assert "BenchRec" in m["source"]
    assert m["n_demo_records"] > 0


# --------------------------------------------------------------------------- #
# the demo run
# --------------------------------------------------------------------------- #

def test_demo_run_returns_a_result(client):
    r = client.post("/api/run", json={"limit": 200})
    assert r.status_code == 200
    body = r.json()
    assert body["n_records"] == 200
    assert body["run_id"]


def test_every_record_is_accounted_for(client):
    b = client.post("/api/run", json={"limit": 300}).json()
    s = b["summary"]
    assert s["posted"] + s["queued"] + s["no_candidate"] + s["suspected_grouped"] == b["n_records"]


def test_result_reports_precision_against_held_out_truth(client):
    b = client.post("/api/run", json={"limit": 500}).json()
    assert 0.0 <= b["precision_of_posted"] <= 1.0
    assert b["posted_correct"] <= b["summary"]["posted"]


def test_exceptions_carry_a_reason_and_an_explanation(client):
    b = client.post("/api/run", json={"limit": 400}).json()
    if b["exceptions"]:
        e = b["exceptions"][0]
        assert e["reason"] and e["explanation"]
        assert "record_id" in e


def test_review_all_mode_posts_nothing(client):
    b = client.post("/api/run", json={"limit": 100, "review_all": True}).json()
    assert b["summary"]["posted"] == 0


def test_throughput_is_reported(client):
    b = client.post("/api/run", json={"limit": 200}).json()
    assert b["records_per_second"] > 0


def test_zero_llm_calls_on_the_matching_path(client):
    b = client.post("/api/run", json={"limit": 200}).json()
    assert b["llm_calls_on_matching_path"] == 0


def test_audit_trail_is_retrievable_for_a_run(client):
    rid = client.post("/api/run", json={"limit": 100}).json()["run_id"]
    rows = client.get(f"/api/run/{rid}/audit").json()
    assert len(rows["decisions"]) == 100
    assert rows["decisions"][0]["policy_version"]


def test_audit_for_an_unknown_run_is_404(client):
    assert client.get("/api/run/does-not-exist/audit").status_code == 404


# --------------------------------------------------------------------------- #
# it must not fall over on bad input
# --------------------------------------------------------------------------- #

def test_limit_is_clamped_rather_than_exploding(client):
    b = client.post("/api/run", json={"limit": 10**9}).json()
    assert b["n_records"] <= 4000


def test_zero_limit_is_rejected_clearly(client):
    r = client.post("/api/run", json={"limit": 0})
    assert r.status_code == 422


def test_upload_rejects_a_non_csv(client):
    r = client.post("/api/upload", files={"file": ("x.exe", io.BytesIO(b"MZ"), "application/octet-stream")})
    assert r.status_code == 400
    assert "csv" in r.json()["detail"].lower()


def test_upload_rejects_a_csv_missing_required_columns(client):
    csv = b"foo,bar\n1,2\n"
    r = client.post("/api/upload", files={"file": ("x.csv", io.BytesIO(csv), "text/csv")})
    assert r.status_code == 400
    assert "column" in r.json()["detail"].lower()


def test_a_live_razorpay_key_is_refused(client):
    """A judge pasting a production key into a hackathon demo is a real risk."""
    r = client.post("/api/connect", json={"key_id": "rzp_live_abc123"})
    assert r.status_code == 400
    assert "test" in r.json()["detail"].lower()


def test_a_test_mode_key_is_accepted_in_shape(client):
    r = client.post("/api/connect", json={"key_id": "rzp_test_abc123"})
    assert r.status_code in (200, 501)


def test_artifacts_directory_is_overridable(monkeypatch, tmp_path):
    """A packaged install puts the code in site-packages, where a path relative
    to __file__ no longer points at the artifacts. The deployment must be able
    to say where they are."""
    import importlib

    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    import allocation_agent.api as api

    importlib.reload(api)
    assert api.ARTIFACTS == tmp_path
    monkeypatch.delenv("ARTIFACTS_DIR")
    importlib.reload(api)


def test_missing_artifacts_degrade_to_503_rather_than_crashing_at_import(monkeypatch, tmp_path):
    """A container that cannot find its models should say so on request, not
    fail to boot -- a crashed process gives a judge a blank page."""
    import importlib

    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "nope"))
    import allocation_agent.api as api

    importlib.reload(api)
    c = TestClient(api.create_app())
    assert c.get("/api/health").json()["models_loaded"] is False
    assert c.post("/api/run", json={"limit": 10}).status_code == 503
    monkeypatch.delenv("ARTIFACTS_DIR")
    importlib.reload(api)


def test_a_corrupt_model_file_does_not_kill_the_process(monkeypatch, tmp_path):
    """Artifact loading happens at process start. An exception there means the
    container never serves anything and a visitor gets a blank page rather than
    a message."""
    import importlib

    (tmp_path / "demo.json").write_text('{"records": [], "key_rows": []}')
    (tmp_path / "models.pkl").write_bytes(b"not a pickle")
    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path))
    import allocation_agent.api as api

    importlib.reload(api)
    c = TestClient(api.create_app())          # must not raise
    h = c.get("/api/health").json()
    assert h["models_loaded"] is False
    assert h["error"]
    assert c.post("/api/run", json={"limit": 5}).status_code == 503
    monkeypatch.delenv("ARTIFACTS_DIR")
    importlib.reload(api)


# --------------------------------------------------------------------------- #
# grouped settlements: the solver, wired in
# --------------------------------------------------------------------------- #

def test_settlement_run_recovers_grouped_payments(client):
    """BenchRec has no subset-sum structure -- measured. ReconRiver does, so the
    solver is demonstrated on the dataset that can actually exercise it."""
    r = client.post("/api/settlements", json={"limit": 40})
    assert r.status_code == 200
    b = r.json()
    assert b["n_settlements"] == 40
    assert b["solved"] + b["ambiguous"] + b["unresolved"] == b["n_settlements"]


def test_a_solved_settlement_shows_its_arithmetic(client):
    b = client.post("/api/settlements", json={"limit": 40}).json()
    solved = [s for s in b["results"] if s["status"] == "solved"]
    assert solved, "expected at least one recoverable settlement"
    s = solved[0]
    assert abs(sum(p["amount"] for p in s["components"]) - s["amount"]) < 0.011
    assert len(s["components"]) >= 1


def test_exact_recovery_is_reported_against_ground_truth(client):
    b = client.post("/api/settlements", json={"limit": 60}).json()
    assert 0.0 <= b["exact_recovery_rate"] <= 1.0
    assert b["wrong_set_rate"] >= 0.0


def test_a_balancing_but_wrong_subset_is_not_reported_as_correct(client):
    """A subset that sums correctly is not evidence it is the right subset.
    Conflating the two is the worst failure this system could produce."""
    b = client.post("/api/settlements", json={"limit": 60}).json()
    assert b["exact_recovery_rate"] <= (b["solved"] / max(b["n_settlements"], 1)) + 1e-9


def test_settlement_results_carry_an_explanation(client):
    b = client.post("/api/settlements", json={"limit": 20}).json()
    assert all(r["explanation"] for r in b["results"])


# --------------------------------------------------------------------------- #
# Where the solver fails matters more than how often. Measured on ReconRiver,
# every wrong answer and almost every refusal is a *single-payment* credit --
# which is not a grouping problem at all and belongs on the matching path. On
# genuine multi-payment batches the solver has not yet returned a wrong set.
# One aggregate number hides that completely, so the split is reported.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def settled(client):
    return client.post("/api/settlements", json={"limit": 150}).json()


def test_results_are_broken_down_by_how_many_payments_the_batch_really_had(settled):
    rows = settled["by_batch_size"]
    assert {r["size"] for r in rows} >= {1, 2, 3}
    for r in rows:
        assert r["recovered"] + r["wrong"] + r["ambiguous"] + r["unresolved"] == r["n"]


def test_the_breakdown_separates_single_payment_credits_from_real_batches(settled):
    """A one-payment 'batch' is an exact match, not a subset-sum problem."""
    rows = {r["size"]: r for r in settled["by_batch_size"]}
    assert rows[1]["n"] > 0 and rows[2]["n"] > 0
    multi_wrong = sum(r["wrong"] for s, r in rows.items() if s > 1)
    assert multi_wrong <= rows[1]["wrong"]


def test_a_batch_that_is_not_inside_the_candidate_pool_is_counted_as_unreachable(settled):
    """No solver can recover a payment that blocking never offered it. Counting
    those as solver failures would blame the wrong component."""
    body = settled
    assert body["unreachable"] >= 0
    assert body["unreachable"] <= body["n_settlements"]
    for r in body["by_batch_size"]:
        assert r["unreachable"] <= r["n"]


# --------------------------------------------------------------------------- #
# Bringing your own data.
#
# The previous version of /api/upload validated a CSV and returned
# "reconciliation of uploaded files is not wired yet". So a judge could not
# check the system against a file whose answers they already knew -- which is
# the only way to actually believe a demo that scores itself.
# --------------------------------------------------------------------------- #

BANK_CSV = ("id,account,amount,date\n"
            "b1,ACC-1,1250.00,2026-03-01\n"
            "b2,ACC-2,80.50,2026-03-02\n"
            "b3,ACC-9,9999.00,2026-03-02\n")

LEDGER_CSV = ("invoice,account,amount,date\n"
              "INV-1,ACC-1,1250.00,2026-03-01\n"
              "INV-2,ACC-2,80.50,2026-03-03\n"
              "INV-3,ACC-2,4000.00,2026-03-03\n")


def _files(bank=BANK_CSV, ledger=LEDGER_CSV):
    return {"bank": ("bank.csv", io.BytesIO(bank.encode()), "text/csv"),
            "ledger": ("ledger.csv", io.BytesIO(ledger.encode()), "text/csv")}


def test_an_uploaded_pair_is_actually_reconciled(client):
    body = client.post("/api/reconcile", files=_files()).json()
    assert body["n_records"] == 3
    assert len(body["results"]) == 3


def test_the_obvious_match_is_found(client):
    """Same account, same amount, same day. If this does not match, nothing does."""
    rows = {r["record_id"]: r for r in client.post(
        "/api/reconcile", files=_files()).json()["results"]}
    assert rows["b1"]["matched_key"] == "INV-1"


def test_a_record_with_nothing_to_match_says_so_rather_than_guessing(client):
    rows = {r["record_id"]: r for r in client.post(
        "/api/reconcile", files=_files()).json()["results"]}
    assert rows["b3"]["matched_key"] is None
    assert "nothing" in rows["b3"]["explanation"].lower()


def test_every_uploaded_record_is_accounted_for(client):
    body = client.post("/api/reconcile", files=_files()).json()
    assert sum(body["summary"].values()) == body["n_records"]


def test_no_precision_is_claimed_on_data_with_no_answer_key(client):
    """The demo can report precision because BenchRec is labelled. An uploaded
    file is not, and inventing a score for it would be the dishonest part."""
    body = client.post("/api/reconcile", files=_files()).json()
    assert "precision" not in body
    assert "ground truth" in body["caveat"].lower() or "no labels" in body["caveat"].lower()


def test_each_result_explains_itself_in_plain_language(client):
    for r in client.post("/api/reconcile", files=_files()).json()["results"]:
        assert r["explanation"] and not any(
            j in r["explanation"] for j in ("no_candidate", "below_threshold", "MULT"))


def test_a_broken_file_reports_the_row_rather_than_failing_opaquely(client):
    bad = "id,account,amount,date\nb1,ACC-1,1.005,2026-03-01\n"
    r = client.post("/api/reconcile", files=_files(bank=bad))
    assert r.status_code == 400
    assert "row 2" in r.json()["detail"]


def test_a_missing_column_is_named(client):
    r = client.post("/api/reconcile", files=_files(bank="id,account,amount\nb1,A,1.00\n"))
    assert r.status_code == 400
    assert "date" in r.json()["detail"]


def test_the_uploaded_run_is_written_to_the_same_audit_log(client):
    """An uploaded run is a real run. It gets a run id and an audit trail, or
    the audit story only holds for the demo."""
    body = client.post("/api/reconcile", files=_files()).json()
    trail = client.get(f"/api/run/{body['run_id']}/audit").json()
    assert len(trail["decisions"]) == body["n_records"]


# --------------------------------------------------------------------------- #
# The page is what a judge actually meets. It gets the same treatment as the
# API: the fields it reads must exist, and the words it shows must be words.
# --------------------------------------------------------------------------- #

def test_the_page_only_reads_fields_the_meta_endpoint_returns(client):
    """The page once read m.n_keys and m.n_train, neither of which exists, so
    two of the three dataset figures silently rendered as nothing."""
    import re
    page = client.get("/").text
    meta = client.get("/api/meta").json()
    for field in set(re.findall(r"\bm\.([a-z_]+)", page)):
        assert field in meta, f"page reads m.{field}, which /api/meta does not return"


def test_no_internal_outcome_name_is_shown_to_a_visitor(client):
    """suspected_grouped is a good variable name and a terrible thing to show
    somebody who has only read the problem statement."""
    page = client.get("/").text
    shown = page.split("<script>")[0]
    for enum in ("no_candidate", "suspected_grouped", "below_threshold", "straight-through"):
        assert enum not in shown, f"{enum} appears in the visible page"


def test_the_page_can_reach_every_endpoint_a_visitor_needs(client):
    page = client.get("/").text
    for route in ("/api/meta", "/api/run", "/api/settlements", "/api/reconcile"):
        assert route in page, f"{route} is built but unreachable from the page"
