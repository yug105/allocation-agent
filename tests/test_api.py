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
    from allocation_agent import api

    importlib.reload(api)
    assert tmp_path == api.ARTIFACTS
    monkeypatch.delenv("ARTIFACTS_DIR")
    importlib.reload(api)


def test_missing_artifacts_degrade_to_503_rather_than_crashing_at_import(monkeypatch, tmp_path):
    """A container that cannot find its models should say so on request, not
    fail to boot -- a crashed process gives a judge a blank page."""
    import importlib

    monkeypatch.setenv("ARTIFACTS_DIR", str(tmp_path / "nope"))
    from allocation_agent import api

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
    from allocation_agent import api

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
    gaps = [b - a for a, b in zip(numbers, numbers[1:], strict=False)]
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
# The previous attempt at this was prose plus an HTML attribute and enforced
# nothing: `max=2000` is advisory, the click handler read `.value` regardless,
# and the server default stayed at 500. Typing 4000 into the box produced
# exactly the run the change claimed to prevent.
#
# The number also had no owner. Four different figures were live at once -- 40
# on the page, 45 in the README, 495 and 524 elsewhere in the same README --
# which is the defect `test_the_threshold_is_sent_to_the_page_rather_than_
# duplicated_in_it` already exists to stop.
# --------------------------------------------------------------------------- #

def test_the_measured_rates_come_from_the_server(client):
    from allocation_agent.api import FREE_TIER_RECORDS_PER_SECOND
    meta = client.get("/api/meta").json()
    assert meta["free_tier_records_per_second"] == FREE_TIER_RECORDS_PER_SECOND


def test_the_page_does_not_hardcode_a_throughput_figure(client):
    """It reads the rate from /api/meta and derives the estimate from it."""
    page = client.get("/").text
    for stale in ("40 records a second", "45 records a second",
                  "roughly six seconds", "495", "524"):
        assert stale not in page, f"page hardcodes {stale!r}"
    assert "free_tier_records_per_second" in page


def test_the_record_count_control_offers_the_whole_held_out_set(client):
    """Section 1 advertises 4,000 held-out records. Refusing to run more than
    half of them, with no sentence saying why, is worse than showing the cost."""
    import re
    meta = client.get("/api/meta").json()
    field = re.search(r'<input id=n type=number value=(\d+) min=(\d+) max=(\d+)',
                      client.get("/").text)
    assert field
    default, minimum, maximum = (int(g) for g in field.groups())
    assert maximum == meta["n_demo_records"]
    assert minimum >= 1
    assert minimum <= default <= maximum, "the control's own default is out of range"
    assert default <= 250, "the default should return quickly"


def test_the_page_clamps_before_sending_rather_than_trusting_the_attribute(client):
    """`max` on an input outside a form does not stop JS reading `.value`."""
    page = client.get("/").text
    assert "clamp" in page, "no clamping helper in the page"
    assert "limit:+$('#n').value" not in page, "still sends the raw field value"


def test_a_validation_error_is_shown_as_words_not_object_object(client):
    """FastAPI returns `detail` as a list of dicts for a 422. Passing that to
    `new Error()` renders the literal text [object Object]."""
    page = client.get("/").text
    assert "Array.isArray" in page or ".map(" in page, \
        "the error path does not handle a list-shaped detail"


def test_the_note_a_visitor_must_read_is_not_the_faintest_text(client):
    """It was styled `.faint` -- the page's lowest contrast, ~4.1:1, under the
    4.5:1 floor -- for the one sentence that has to be read before clicking."""
    page = client.get("/").text
    est = page[page.index("id=estimate") - 200:page.index("id=estimate") + 200]
    assert "class=faint" not in est


# --------------------------------------------------------------------------- #
# The demo endpoint defaulted to 0.7 and the upload endpoint hardcoded 0.5, so
# the same record could be called grouped by one and matched by the other. At
# those two points the detector's test precision is 68.4% and 57.6% -- the
# choice is worth 11 points and neither place owned it.
# --------------------------------------------------------------------------- #

def test_both_endpoints_use_the_same_grouping_threshold(client):
    from allocation_agent.api import MULT_THRESHOLD, RunRequest
    assert RunRequest().mult_threshold == MULT_THRESHOLD
    page_default = client.post("/api/run", json={"limit": 1}).json()
    assert page_default["mult_threshold"] == MULT_THRESHOLD


def test_the_run_reports_the_threshold_it_used(client):
    """A run whose grouping cut cannot be read back cannot be reproduced."""
    body = client.post("/api/run", json={"limit": 1, "mult_threshold": 0.9}).json()
    assert body["mult_threshold"] == 0.9


# --------------------------------------------------------------------------- #
# Above the fold.
#
# Judges scan the top of the page for about thirty seconds and form an opinion.
# Until now that surface was a title, a paragraph and three dataset counts --
# nothing had happened, and nothing could until a button was pressed. And every
# figure on the page was a record count, while every reconciliation product
# measures unreconciled *value*: "1,535 records" is a statistic, "15.6M sitting
# in review" is a reason to care.
#
# Computed at export time over the whole held-out set, not at startup: 4,000
# records take ~90s on the deployed free instance and the health check would
# fail before the container was ready.
# --------------------------------------------------------------------------- #

def test_the_overview_is_served_without_running_anything(client):
    body = client.get("/api/overview").json()
    assert body["n_records"] == 4000
    assert body["posted_value"] > 0 and body["queue_value"] > 0


def test_the_overview_is_measured_in_value_not_only_counts(client):
    body = client.get("/api/overview").json()
    assert body["posted_value"] + body["queue_value"] == pytest.approx(body["total_value"], rel=1e-6)
    assert 0 < body["queue_share_of_value"] < 1


def test_the_overview_names_where_the_queue_value_actually_sits(client):
    """The grouped case is the one the bank's rules engine resolved 0% of, and
    it is where the money needing a person is. That is the whole argument."""
    body = client.get("/api/overview").json()
    assert body["grouped_share_of_queue"] > 0.5
    assert body["value_by_outcome"]["suspected_grouped"] > body["value_by_outcome"]["queued"]


def test_the_overview_agrees_with_a_live_run(client):
    """A cached figure that drifts from what the buttons produce is worse than
    no figure. Same code path, so the rates must line up."""
    overview = client.get("/api/overview").json()
    live = client.post("/api/run", json={"limit": 4000}).json()
    assert live["straight_through_rate"] == pytest.approx(
        overview["straight_through_rate"], abs=0.005)
    assert live["precision_of_posted"] == pytest.approx(
        overview["precision_of_posted"], abs=0.005)


def test_the_page_shows_a_result_before_any_click(client):
    page = client.get("/")
    assert "/api/overview" in page.text
    assert "id=hero" in page.text


def test_the_page_states_the_amounts_are_obfuscated_units(client):
    """BenchRec's amounts are not real currency. Rendering them with a symbol
    would imply a real bank is short 17.9 million.

    Checks the whole document: the caveat is rendered into the hero, so it sits
    in the script block rather than in the static markup.
    """
    page = client.get("/").text.lower()
    assert "obfuscated" in page
    assert "not a real currency" in page


def test_a_run_reports_the_value_it_moved_not_only_the_record_count(client):
    body = client.post("/api/run", json={"limit": 200}).json()
    assert body["posted_value"] >= 0
    assert body["queue_value"] >= 0
    for e in body["exceptions"]:
        assert "amount" in e


def test_the_exception_queue_is_ordered_by_value_at_risk(client):
    """A controller works it top-down, and the top ten are a tenth of the
    queue's value. Record order wastes that."""
    body = client.post("/api/run", json={"limit": 500}).json()
    amounts = [abs(e["amount"]) for e in body["exceptions"]]
    assert amounts == sorted(amounts, reverse=True)


def test_the_reported_exception_total_covers_every_exception_not_the_first_page(client):
    """`exceptions` is capped at 100 for payload size. A total computed from
    that list would silently describe a fraction of the queue."""
    body = client.post("/api/run", json={"limit": 2000}).json()
    assert body["n_exceptions"] > len(body["exceptions"])
    assert body["queue_value"] > sum(abs(e["amount"]) for e in body["exceptions"])


# --------------------------------------------------------------------------- #
# Exception aging.
#
# Every reconciliation product reports unreconciled balance by age, so the
# obvious move was to add it. Measured first: the held-out set spans 27 days,
# every record falls in one 30-day bucket, and the auto-post rate across weekly
# buckets is 74.9 / 80.2 / 81.3 / 80.0 -- flat. Median age is 14 days for
# posted against 11 for queued.
#
# Age carries no signal here because aging measures how long an item has sat
# unresolved in a *running* system, and this is a 27-day snapshot resolved in
# one batch. A real uploaded ledger spanning months is a different matter. So
# it is computed when the span supports it and refused with a reason when it
# does not -- the same rule as rates_meaningful.
# --------------------------------------------------------------------------- #

def test_aging_is_withheld_when_the_data_is_too_short_a_window(client):
    body = client.post("/api/run", json={"limit": 500}).json()
    assert body["aging"]["meaningful"] is False
    assert body["aging"]["span_days"] < 60
    assert "snapshot" in body["aging"]["note"].lower() or "window" in body["aging"]["note"].lower()


def test_aging_is_reported_when_an_uploaded_ledger_spans_months(client):
    bank = "id,account,amount,date\n" + "".join(
        f"B{i},A-1,{100 + i}.00,2026-{m:02d}-05\n"
        for i, m in enumerate([1, 2, 3, 5, 7, 9, 11]))
    ledger = "invoice,account,amount,date\nINV-1,A-1,9999.00,2026-01-05\n"
    body = client.post("/api/reconcile", files=_pair(bank, ledger)).json()
    assert body["aging"]["meaningful"] is True
    assert body["aging"]["span_days"] > 60
    assert body["aging"]["buckets"], "no buckets returned"


def test_aging_buckets_carry_value_not_only_counts(client):
    bank = "id,account,amount,date\n" + "".join(
        f"B{i},A-9,{1000 * (i + 1)}.00,2026-{m:02d}-05\n"
        for i, m in enumerate([1, 3, 6, 9, 12]))
    ledger = "invoice,account,amount,date\nINV-1,A-1,5.00,2026-01-05\n"
    body = client.post("/api/reconcile", files=_pair(bank, ledger)).json()
    for b in body["aging"]["buckets"]:
        assert {"label", "count", "value"} <= set(b)
    assert sum(b["value"] for b in body["aging"]["buckets"]) > 0


def test_only_unresolved_records_are_aged(client):
    """Aging an item that was reconciled is meaningless -- it is not waiting."""
    bank = "id,account,amount,date\n" + "".join(
        f"B{i},A-1,{100 + i}.00,2026-{m:02d}-05\n" for i, m in enumerate([1, 4, 8, 12]))
    ledger = "invoice,account,amount,date\nINV-1,A-1,9999.00,2026-01-05\n"
    body = client.post("/api/reconcile", files=_pair(bank, ledger)).json()
    aged = sum(b["count"] for b in body["aging"]["buckets"])
    s = body["summary"]
    assert aged == s["queued"] + s["suspected_grouped"] + s["no_candidate"]


# --------------------------------------------------------------------------- #
# Working the queue.
#
# Each exception already says why it stopped. The question a reviewer actually
# has is which of 824 to open first, and how much of the backlog clearing the
# top few would remove. Measured on the held-out set: the ten largest are 12%
# of the queue's value, so that answer is worth stating.
# --------------------------------------------------------------------------- #

def test_each_exception_carries_its_share_of_the_queue(client):
    body = client.post("/api/run", json={"limit": 500}).json()
    assert body["exceptions"]
    for e in body["exceptions"]:
        assert 0 <= e["share_of_queue"] <= 1


def test_the_shares_are_taken_against_the_whole_queue_not_the_returned_page(client):
    """`exceptions` is capped at 100. Shares computed against that list would
    sum to 1 while describing a fraction of the backlog."""
    body = client.post("/api/run", json={"limit": 2000}).json()
    assert body["n_exceptions"] > len(body["exceptions"])
    assert sum(e["share_of_queue"] for e in body["exceptions"]) < 0.999


def test_the_running_total_says_what_clearing_the_top_n_would_clear(client):
    body = client.post("/api/run", json={"limit": 500}).json()
    running = [e["cumulative_share"] for e in body["exceptions"]]
    assert running == sorted(running), "cumulative share must not decrease"
    first = body["exceptions"][0]
    assert first["cumulative_share"] == pytest.approx(first["share_of_queue"])


def test_the_page_tells_a_reviewer_where_to_start(client):
    page = client.get("/").text
    assert "cumulative_share" in page


# --------------------------------------------------------------------------- #
# A lone exact amount overrules the grouping check. On the held-out set the
# detector fires 535 times and is overruled on 9 of them -- and those 9 were
# recorded as ordinary ranked matches, so nothing in the trail said a
# probabilistic detector had been overridden. An audit log that cannot show
# that is not showing why the decision was made.
# --------------------------------------------------------------------------- #

def test_overruling_the_grouping_check_is_recorded_as_evidence(client):
    """Nine held-out records hit this: the detector fires, a lone exact amount
    overrules it, and the record posts. Found by scanning the real data rather
    than staged, because the detector will not fire on a three-row fixture."""
    from allocation_agent.api import BLOCKING, MULT_THRESHOLD, _p_multiple_with, _State
    from allocation_agent.match.blocker import block

    state = _State()
    state.load()
    target = None
    for rec in state.records:
        usable = [k for k in sorted(block(rec, state.index, BLOCKING))
                  if k in state.key_stats]
        if not usable:
            continue
        exact = [k for k in usable if rec.amount_minor in state.key_stats[k].amounts]
        if len(exact) == 1 and _p_multiple_with(state, rec, usable,
                                                state.key_stats) >= MULT_THRESHOLD:
            target = rec.record_id
            break
    assert target, "no record exercises the override"

    body = client.post("/api/run", json={"limit": 4000}).json()
    trail = client.get(f"/api/run/{body['run_id']}/audit").json()
    row = next(d for d in trail["decisions"] if d["record_id"] == target)
    import json as _json
    evidence = _json.loads(row["evidence"] or "{}")
    assert evidence.get("overrode_grouping") is True
    assert evidence.get("exact_amount") is True
    assert "p_multiple" in evidence, "the overruled probability must be on the record"



def test_an_ordinary_match_carries_no_override_evidence(client):
    body = client.post("/api/run", json={"limit": 200}).json()
    trail = client.get(f"/api/run/{body['run_id']}/audit").json()
    plain = [d for d in trail["decisions"] if d["path"] == "ranked"]
    assert plain
    assert any("overrode_grouping" not in (d["evidence"] or "") for d in plain)


def test_the_llm_call_count_is_measured_not_asserted(client):
    """Reporting a hardcoded 0 answers 'how do you know?' with 'I typed it'."""
    import inspect

    from allocation_agent import api
    src = inspect.getsource(api.create_app)
    assert '"llm_calls_on_matching_path": 0' not in src
    assert "narrator.calls - llm_before" in src
    assert client.post("/api/run", json={"limit": 50}).json()[
        "llm_calls_on_matching_path"] == 0


# --------------------------------------------------------------------------- #
# Calibration.
#
# `sigmoid(top - second)` is a monotone transform of a LambdaRank margin.
# LambdaRank optimises order, not likelihood, so neither the scores nor a
# sigmoid of their difference carry probability meaning. Measured on test, it
# claimed 74.8% where the truth was 21.5% and 84.9% where it was 46.0% -- and
# the gate compares that number against 0.85.
#
# An isotonic fit on validation: ECE 0.0920 -> 0.0116, Brier 0.0774 -> 0.0407.
# End to end it costs 2.55pp of straight-through and removes 23% of the wrong
# auto-posts, because the gate now means what it says.
# --------------------------------------------------------------------------- #

def test_a_calibrator_is_loaded_and_used(client):
    from allocation_agent.api import _State
    state = _State()
    state.load()
    assert state.calibrator is not None
    assert state.calibrator_kind in {"isotonic", "platt"}


def test_the_confidence_source_is_recorded_on_every_ranked_decision(client):
    """Which scale a number came from decides what it means. A trail that does
    not say cannot be re-read later."""
    import json as _json
    body = client.post("/api/run", json={"limit": 200}).json()
    trail = client.get(f"/api/run/{body['run_id']}/audit").json()
    ranked = [d for d in trail["decisions"] if d["path"] == "ranked"]
    assert ranked
    for d in ranked:
        assert _json.loads(d["evidence"])["confidence_from"] in {"isotonic", "platt"}


def test_confidence_is_no_longer_the_sigmoid_of_the_margin(client):
    """The specific failure: sigmoid(margin) never returns below 0.5, because a
    margin cannot be negative. A real probability can and must."""
    body = client.post("/api/run", json={"limit": 2000}).json()
    trail = client.get(f"/api/run/{body['run_id']}/audit").json()
    conf = [d["confidence"] for d in trail["decisions"]
            if d["path"] == "ranked" and d["confidence"] is not None]
    assert conf
    assert min(conf) < 0.5, "no record scored below 50% — still on the sigmoid scale"
