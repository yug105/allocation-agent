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


# --------------------------------------------------------------------------- #
# The direct path.
#
# Two defects found by running the sample pair on the upload tab:
#
#   * With one candidate, `margin` fell back to 1.0, so confidence was always
#     sigmoid(1.0) = 73.1% -- permanently below the 85% base bar. A lone
#     candidate could never post, however exact the match. BenchRec never
#     returns fewer than 4 candidates, so no existing test could reach it.
#   * The grouping check ran before ranking and short-circuited it, letting a
#     63% guess overrule an exact amount match the ranker was certain of.
#
# Measured on the held-out set: where exactly one candidate matches the amount
# exactly, it is the right answer 98.98% of the time (2,321 of 2,345), and only
# 0.68% of those records were actually grouped. So an exact amount match is
# direct evidence and is treated as such -- which is what the design specified
# for the direct-key component all along.
# --------------------------------------------------------------------------- #

def _pair(bank, ledger):
    return {"bank": ("b.csv", io.BytesIO(bank.encode()), "text/csv"),
            "ledger": ("l.csv", io.BytesIO(ledger.encode()), "text/csv")}


def _one(client, bank, ledger):
    return client.post("/api/reconcile", files=_pair(bank, ledger)).json()["results"][0]


def test_a_lone_exact_amount_match_is_posted_not_left_at_73_percent(client):
    r = _one(client,
             "id,account,amount,date\nB1,A-1,80.50,2026-03-02\n",
             "invoice,account,amount,date\nINV-1,A-1,80.50,2026-03-03\n")
    assert r["matched_key"] == "INV-1"
    assert r["outcome"] == "posted"
    assert r["confidence"] > 0.9


def test_a_lone_candidate_that_does_not_match_the_amount_is_not_posted(client):
    """There is no margin and no exact-amount evidence, so there is nothing to
    be confident from. Inventing a number here is what produced the bug."""
    r = _one(client,
             "id,account,amount,date\nB1,A-1,80.50,2026-03-02\n",
             "invoice,account,amount,date\nINV-1,A-1,4000.00,2026-03-03\n")
    assert r["outcome"] != "posted"


def test_an_exact_amount_match_is_not_overruled_by_the_grouping_guess(client):
    """A single ledger entry accounting for the whole amount defeats the
    premise of the grouped path, which is that no single entry explains it."""
    r = _one(client,
             "id,account,amount,date\nB1,A-1,4300.00,2026-03-04\n",
             "invoice,account,amount,date\n"
             "INV-1,A-1,1250.00,2026-03-01\nINV-2,A-1,4300.00,2026-03-03\n"
             "INV-3,A-1,120.00,2026-03-06\n")
    assert r["matched_key"] == "INV-2"
    assert r["outcome"] == "posted"


def test_two_candidates_with_the_same_exact_amount_do_not_take_the_direct_path(client):
    """Two entries of 500 both explain a 500 credit. The amount cannot choose
    between them, so this is a ranking question, not a direct one."""
    r = _one(client,
             "id,account,amount,date\nB1,A-1,500.00,2026-03-02\n",
             "invoice,account,amount,date\n"
             "INV-1,A-1,500.00,2026-03-01\nINV-2,A-1,500.00,2026-03-03\n")
    assert r["outcome"] != "posted" or r["confidence"] < 0.99


def test_the_direct_path_is_named_in_the_audit_log(client):
    body = client.post("/api/reconcile", files=_pair(
        "id,account,amount,date\nB1,A-1,80.50,2026-03-02\n",
        "invoice,account,amount,date\nINV-1,A-1,80.50,2026-03-03\n")).json()
    trail = client.get(f"/api/run/{body['run_id']}/audit").json()
    assert trail["decisions"][0]["path"] == "direct"


def test_a_large_exact_match_still_answers_to_the_amount_scaled_bar(client):
    """The direct path supplies evidence; it does not bypass the gate.

    The bar rises 2 points per decade above a 100.00 reference, so 98.98%
    evidence carries an exact match up to about a billion and no further. Above
    that the required confidence exceeds what this evidence is worth, and even
    a perfect amount match goes to a person.
    """
    r = _one(client,
             "id,account,amount,date\nB1,A-1,5000000000.00,2026-03-02\n",
             "invoice,account,amount,date\nINV-1,A-1,5000000000.00,2026-03-03\n")
    assert r["outcome"] == "queued"


def test_the_direct_match_says_in_plain_words_why_it_was_certain(client):
    r = _one(client,
             "id,account,amount,date\nB1,A-1,80.50,2026-03-02\n",
             "invoice,account,amount,date\nINV-1,A-1,80.50,2026-03-03\n")
    assert "exact" in r["explanation"].lower()


# --------------------------------------------------------------------------- #
# The badge said "Found the group" above a sentence explaining it was not the
# recorded batch. The label was derived from status alone, which is 'solved'
# for a subset that balances whether or not it is the right one. A verdict is a
# judgement about the answer, so the server makes it and the page displays it.
# --------------------------------------------------------------------------- #

def test_a_balancing_but_wrong_subset_is_never_labelled_found(settled):
    """Asserted on the exact string. The first version of this test accepted
    any verdict containing "not", which matches "Nothing found" and "Notable" --
    it would have passed on the bug it was written to catch."""
    for r in settled["results"]:
        if r["status"] == "solved" and not r["exact"]:
            assert r["verdict"] == "Wrong group — sent to review"
            assert r["tone"] == "bad"


def test_a_correct_subset_is_labelled_found(settled):
    hits = [r for r in settled["results"] if r["exact"]]
    assert hits
    for r in hits:
        assert r["verdict"] == "Found the group"
        assert r["tone"] == "good"


def test_every_result_carries_a_verdict_and_a_severity(settled):
    for r in settled["results"]:
        assert r["verdict"]
        assert r["tone"] in {"good", "warn", "bad"}


def test_a_record_with_no_answer_is_never_badged_as_found(settled):
    """`is_exact` is `sorted([]) == sorted(truth)`, which is True whenever the
    truth is empty -- so an unresolved record could be badged green with zero
    components. Only a solved record can have found anything."""
    for r in settled["results"]:
        if r["status"] != "solved":
            assert r["verdict"] != "Found the group"
            assert r["tone"] != "good"
        if not r["components"]:
            assert r["verdict"] != "Found the group"


# --------------------------------------------------------------------------- #
# Rates over a handful of records are arithmetic, not evidence -- and the thing
# being rated decides what "a handful" counts. Precision is a rate over
# *answers*, not over records, and the first version of this guard checked the
# record count, so 20 records yielding 3 answers still printed "0.0%" in green.
# --------------------------------------------------------------------------- #

def test_the_recovery_rate_is_withheld_when_there_are_too_few_records(client):
    body = client.post("/api/settlements", json={"limit": 1}).json()
    assert body["recovery_rate_meaningful"] is False


def test_the_precision_rate_is_withheld_when_there_are_too_few_answers(client):
    """20 records, 3 answers. The record count clears the bar and the answer
    count does not, and precision is a rate over answers."""
    body = client.post("/api/settlements", json={"limit": 20}).json()
    assert body["solved"] < body["min_for_rates"]
    assert body["precision_meaningful"] is False


def test_both_rates_are_reported_once_there_are_enough_of_each(client):
    body = client.post("/api/settlements", json={"limit": 150}).json()
    assert body["recovery_rate_meaningful"] is True
    assert body["precision_meaningful"] is True


def test_the_threshold_is_sent_to_the_page_rather_than_duplicated_in_it(client):
    """The page hardcoded 20 while the server owned the constant. Change one
    and the other silently lies."""
    from allocation_agent.api import MIN_FOR_RATES
    body = client.post("/api/settlements", json={"limit": 1}).json()
    assert body["min_for_rates"] == MIN_FOR_RATES
    assert "${MIN" not in client.get("/").text
    assert "above 20" not in client.get("/").text


def test_the_counts_behind_each_rate_are_returned_not_recomputed(client):
    """The page recovered `exact` by multiplying a float rate back by its
    denominator. The server had the integer all along."""
    body = client.post("/api/settlements", json={"limit": 150}).json()
    assert body["exact"] == sum(1 for r in body["results"] if r["exact"])
    assert body["wrong"] == sum(
        1 for r in body["results"] if r["status"] == "solved" and not r["exact"])


# --------------------------------------------------------------------------- #
# How many payments an answer uses decides how much it is worth.
# --------------------------------------------------------------------------- #

def test_no_answer_uses_more_payments_than_the_solver_permits(client):
    """Reads the cap rather than repeating it, so moving the cap moves the
    test instead of quietly making it vacuous."""
    from allocation_agent.match.solver import SolverConfig
    cap = SolverConfig().max_subset_size
    body = client.post("/api/settlements", json={"limit": 150}).json()
    assert any(r["status"] == "solved" for r in body["results"])
    for r in body["results"]:
        assert len(r["components"]) <= cap


def test_the_refusal_sentence_states_the_cap_the_solver_actually_uses(client):
    """It said "six payments" while the cap was four. The sentence and the
    constant must come from the same place."""
    from allocation_agent.match.solver import SolverConfig
    cap = SolverConfig().max_subset_size
    body = client.post("/api/settlements", json={"limit": 150}).json()
    seen = False
    for r in body["results"]:
        if "adds up to this credit" in r["explanation"]:
            seen = True
            assert f"{cap} payments or fewer" in r["explanation"]
            assert f"{cap + 1} payments" in r["explanation"]
    assert seen, "no refusal sentence exercised"


def test_refusing_on_the_size_cap_does_not_claim_nothing_adds_up(settled):
    for r in settled["results"]:
        if "adds up to this credit" in r["explanation"]:
            assert "deliberately not claimed" in r["explanation"]


def test_a_pool_refusal_and_a_target_refusal_do_not_share_one_sentence(client):
    """The solver returns TOO_LARGE for an oversized target as well as an
    oversized pool. One sentence naming the pool cap is false for the other."""
    from allocation_agent.match.solver import SolverConfig, SolverStatus, solve_subset
    r = solve_subset(target_minor=10**13, candidates_minor=[1, 2, 3],
                     config=SolverConfig(max_target_minor=10**7))
    assert r.status is SolverStatus.TOO_LARGE
    assert "candidates" not in r.detail


# --------------------------------------------------------------------------- #
# The narrator was constructed in create_app() and never called once. Every
# explanation on the page was an f-string, so its central guarantee -- that it
# cannot emit a figure absent from the payload -- protected nothing a visitor
# read.
#
# Its actual job is residual diagnosis: naming *why* two amounts differ. That
# applies to a record queued against a best candidate, where a real gap exists.
# It is not wired to cases with no residual to explain.
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def demo_run(client):
    return client.post("/api/run", json={"limit": 500}).json()


def test_a_queued_record_with_an_amount_gap_gets_a_diagnosed_cause(demo_run):
    """A reviewer needs to know *why* the figures differ, not only that they
    do. Causes are ranked by arithmetic fit against the actual gap."""
    causes = {e.get("residual_cause") for e in demo_run["exceptions"]
              if e["reason"] == "queued"}
    named = causes - {None}
    assert named, "no queued record was diagnosed"
    assert named <= {"BANK_CHARGE", "ROUNDING", "FX_DIFFERENCE",
                     "PARTIAL_PAYMENT", "UNEXPLAINED"}


def test_the_diagnosis_is_written_into_what_the_reviewer_reads(demo_run):
    diagnosed = [e for e in demo_run["exceptions"]
                 if e["reason"] == "queued" and e.get("residual_cause")]
    assert diagnosed
    for e in diagnosed[:5]:
        assert "Gap of" in e["explanation"]


def test_the_narrator_is_actually_invoked(client):
    """The defect was that it was constructed and never called, so its
    guarantee protected nothing anyone read."""
    from allocation_agent.api import create_app
    app = create_app()
    before = app.state.narrator_calls()
    TestClient(app).post("/api/run", json={"limit": 500})
    assert app.state.narrator_calls() > before


def test_a_diagnosed_gap_matches_the_arithmetic_it_claims(demo_run):
    """The sentence quotes a gap. That figure must be the record's real one."""
    for e in demo_run["exceptions"]:
        if e.get("residual_cause") and "Gap of" in e["explanation"]:
            quoted = e["explanation"].split("Gap of ")[1].split()[0].rstrip(",")
            assert quoted.replace(",", "") == f"{e['residual']:.2f}"


def test_a_record_with_no_gap_is_not_given_a_residual_cause(client):
    """An exact match has nothing to diagnose. Naming a cause anyway would be
    inventing one."""
    r = _one(client,
             "id,account,amount,date\nB1,A-1,80.50,2026-03-02\n",
             "invoice,account,amount,date\nINV-1,A-1,80.50,2026-03-03\n")
    assert r["residual"] == 0
    assert r["residual_cause"] is None


def test_a_record_with_nothing_to_match_is_not_given_a_residual_cause(client):
    r = _one(client,
             "id,account,amount,date\nB1,NOBODY,777.00,2026-03-05\n",
             "invoice,account,amount,date\nINV-1,A-1,80.50,2026-03-03\n")
    assert r["outcome"] == "no_candidate"
    assert r["residual_cause"] is None




def test_no_generated_figure_escapes_that_was_not_in_the_payload(client):
    """The guarantee, applied to live output: every number in an explanation
    must be one the record actually carries."""
    import re
    body = client.post("/api/reconcile", files=_pair(
        "id,account,amount,date\nB1,A-1,80.50,2026-03-02\n",
        "invoice,account,amount,date\n"
        "INV-1,A-1,100.00,2026-03-03\nINV-2,A-1,100.00,2026-03-04\n")).json()
    r = body["results"][0]
    allowed = {"80.50", "100.00", "19.50", "1", "2", "0", "85", "73", "99"}
    figures = set(re.findall(r"\d+(?:[.,]\d+)*", r["explanation"]))
    unexplained = {f for f in figures if f not in allowed and f.rstrip("%") not in allowed}
    assert not unexplained, f"explanation contains unsupported figures: {unexplained}"


# --------------------------------------------------------------------------- #
# Stubs removed. Both returned a placeholder; /api/upload was superseded by
# /api/reconcile and /api/connect never did anything at all.
# --------------------------------------------------------------------------- #

def test_the_superseded_upload_stub_is_gone(client):
    assert client.post("/api/upload", files={"file": ("f.csv", io.BytesIO(b"a\n"), "text/csv")}
                       ).status_code == 404


def test_the_unimplemented_connect_stub_is_gone(client):
    assert client.post("/api/connect", json={"key_id": "rzp_test_x",
                                             "key_secret": "y"}).status_code == 404


# --------------------------------------------------------------------------- #
# Asking for a few credits returned the first few, and ReconRiver front-loads
# every hard case: 11 of the first 15 are unsolvable and 0 of credits 30-149
# are. So a visitor sampling small saw nothing but failures and concluded the
# component does not work.
#
# The fix is an evenly spaced sample -- systematic, chosen by position and
# never by outcome, and labelled on the page. Taking the best N would be the
# cherry-picking this project already got wrong once.
# --------------------------------------------------------------------------- #

def test_a_small_request_samples_across_the_file_not_off_the_front(client):
    body = client.post("/api/settlements", json={"limit": 4}).json()
    ids = [r["settlement_id"] for r in body["results"]]
    assert ids != sorted(ids)[:4] or len({i[-3:] for i in ids}) == 4
    numbers = sorted(int(i.split("-")[-1]) for i in ids)
    assert numbers[-1] - numbers[0] > 50, f"sample is bunched at the front: {numbers}"


def test_the_sample_is_evenly_spaced_and_says_so(client):
    body = client.post("/api/settlements", json={"limit": 5}).json()
    assert body["sampling"] == "evenly spaced"
    numbers = sorted(int(r["settlement_id"].split("-")[-1]) for r in body["results"])
    gaps = [b - a for a, b in zip(numbers, numbers[1:])]
    assert max(gaps) - min(gaps) <= 1, f"uneven spacing: {gaps}"


def test_asking_for_everything_returns_everything_in_order(client):
    body = client.post("/api/settlements", json={"limit": 150}).json()
    assert body["sampling"] == "all"
    assert body["n_settlements"] == 150


def test_the_sample_is_chosen_by_position_never_by_outcome(client):
    """Two runs of the same size return the same credits, and a small sample
    still contains failures -- if it did not, it would be a curated one."""
    a = client.post("/api/settlements", json={"limit": 10}).json()
    b = client.post("/api/settlements", json={"limit": 10}).json()
    assert [r["settlement_id"] for r in a["results"]] == [r["settlement_id"] for r in b["results"]]
    assert any(r["status"] != "solved" for r in a["results"]), \
        "a sample with no failures in it has been curated"



# --------------------------------------------------------------------------- #
# Ten credits carry a flat 2.00 charge, so the credit never equals the sum of
# its batch; twelve have pools of 765-777 against a 128 cap, and the two groups
# overlap. Three fixes were tried and measured, and all three made it worse:
#
#   tolerance 2.00                123 answers, 113 right -- every extra one wrong
#   fewest-payments-before-exact  135 answers,  76 right -- far worse
#   solve at target minus 2.00    121 answers, 115 right -- 2 of 10 recovered,
#                                                           precision 98.3 -> 95.0
#
# So they stay unrecovered and the refusal says only what is known. A near-miss
# probe was built and then removed: it reported gaps of 0.11 and 3.17 as
# "consistent with a flat bank charge" when the real charge is exactly 2.00 --
# finding the nearest coincidental subset and narrating it as evidence, which
# is the defect this project keeps having to correct.
# --------------------------------------------------------------------------- #

def test_no_near_miss_is_narrated_as_a_probable_charge(client):
    body = client.post("/api/settlements", json={"limit": 150}).json()
    for r in body["results"]:
        assert "near_miss_minor" not in r
        assert "bank charge" not in r["explanation"].lower()


def test_an_unresolved_credit_says_only_what_is_known(client):
    body = client.post("/api/settlements", json={"limit": 150}).json()
    unresolved = [r for r in body["results"] if r["status"] == "unresolved"]
    assert unresolved
    for r in unresolved:
        assert not r["components"]
        assert r["tone"] != "good"
        assert r["verdict"] == "Could not resolve"


# --------------------------------------------------------------------------- #
# The page renders the arithmetic as its headline, then itemises the payments.
# The explanation restated the same equation a third time, so a recovered
# credit read as three copies of one fact and the sentence that actually
# carried a judgement was buried at the end of the repetition.
# --------------------------------------------------------------------------- #

def test_the_explanation_does_not_restate_the_arithmetic(settled):
    """`3,989.47 = 1,899.61 + 2,089.86` is already the headline and the list."""
    solved = [r for r in settled["results"] if r["components"]]
    assert solved
    for r in solved:
        equation = " + ".join(f"{c['amount']:,.2f}" for c in r["components"])
        assert equation not in r["explanation"], (
            f"{r['settlement_id']} repeats its own arithmetic")


def test_a_recovered_credit_says_what_the_arithmetic_cannot(settled):
    """That the subset is the recorded batch, not merely that it adds up."""
    hits = [r for r in settled["results"] if r["exact"]]
    assert hits
    for r in hits:
        assert "record" in r["explanation"].lower()


def test_a_wrong_group_keeps_the_sentence_that_matters(settled):
    for r in settled["results"]:
        if r["status"] == "solved" and not r["exact"]:
            assert "not the recorded batch" in r["explanation"]


def test_the_date_layout_that_was_chosen_is_reported_back(client):
    """03/01/2026 is read one way for the whole file. Saying which is the only
    way the person who wrote it can catch a wrong guess."""
    body = client.post("/api/reconcile", files=_pair(
        "id,account,amount,date\nB1,A-1,80.50,03/01/2026\n",
        "invoice,account,amount,date\nINV-1,A-1,80.50,2026-03-03\n")).json()
    assert body["date_layout"]["bank"] in {"DD/MM/YYYY", "MM/DD/YYYY"}
    assert body["date_layout"]["ledger"] == "YYYY-MM-DD"


# --------------------------------------------------------------------------- #
# Measured on the deployed free instance: ~40 records/second, so the page's old
# default of 500 was a 13-second wait behind a button that only says
# "running...", and its old maximum of 4,000 was around 100 seconds -- long
# enough that a visitor concludes it has hung. The controls now offer what the
# deployed box can actually do.
# --------------------------------------------------------------------------- #

def test_the_demo_control_defaults_to_a_batch_that_returns_quickly(client):
    import re
    page = client.get("/").text
    field = re.search(r'<input id=n type=number value=(\d+) min=\d+ max=(\d+)', page)
    assert field, "record-count control not found"
    default, maximum = int(field.group(1)), int(field.group(2))
    assert default <= 250, f"default of {default} is a {default / 40:.0f}s wait"
    assert maximum <= 2000, f"maximum of {maximum} is a {maximum / 40:.0f}s wait"


def test_the_page_states_the_deployed_throughput(client):
    """A visitor who knows it is six seconds waits; one who does not reloads."""
    page = client.get("/").text
    assert "40 records a second" in page
