"""HTTP service.

The whole requirement: someone opens a link and sees a real result without
configuring anything. So the demo runs on **held-out records the models never
saw during training**, and reports precision against ground truth rather than
asking to be believed.

Three ways in, in order of how many people will use them:

1. **Demo** -- bundled slice of the held-out split. One request, no setup.
2. **Upload** -- a CSV of your own.
3. **Connect** -- a Razorpay *test-mode* key. A live key is refused: a judge
   pasting a production credential into a hackathon project is a real risk, and
   refusing it costs nothing.
"""

from __future__ import annotations

import asyncio
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from allocation_agent.adapters.csv_upload import (
    UploadError,
    parse_bank_csv,
    parse_ledger_csv,
)
from allocation_agent.adapters.razorpay import (
    RazorpayError,
    fetch_recon,
    group_into_settlements,
)
from allocation_agent.adapters.razorpay import _default_fetch as _rzp_fetch
from allocation_agent.decide.gate import GateConfig, decide
from allocation_agent.decide.narrate import Narrator
from allocation_agent.match.blocker import BlockingConfig
from allocation_agent.match.engine import MULT_THRESHOLD, Models, match_one
from allocation_agent.match.features import FEATURE_NAMES, build_key_stats
from allocation_agent.match.solver import SolverConfig, SolverStatus, solve_subset
from allocation_agent.report.audit import AuditLog, RunConfig
from allocation_agent.stores.keys import KeyIndex, KeyRow
from allocation_agent.types import BankRecord


def _artifacts_dir() -> Path:
    """Locate the artifacts directory.

    ``parents[2]`` is correct for an editable install from a src layout and
    wrong for a normal one, where the package lives in site-packages. Env var
    first so a deployment can be explicit, then the plausible locations.
    """
    if env := os.environ.get("ARTIFACTS_DIR"):
        return Path(env)
    here = Path(__file__).resolve()
    for candidate in (here.parents[2] / "artifacts",   # editable, src layout
                      Path.cwd() / "artifacts",        # container workdir
                      here.parents[3] / "artifacts"):  # site-packages install
        if candidate.exists():
            return candidate
    return here.parents[2] / "artifacts"


ARTIFACTS = _artifacts_dir()
REQUIRED_COLUMNS = {"account", "amount", "date"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_ROWS = 20_000
# A ledger is legitimately larger than a bank file, but KeyIndex and
# build_key_stats are built from it on every request, so it needs its own cap.
MAX_LEDGER_ROWS = 100_000

# Measured on the held-out set, not asserted: see _match_one.

# Below this many records, report counts rather than rates.
MIN_FOR_RATES = 20

# Blocking is configured once. It was written in four places -- a local that
# nothing read, two audit records, and the matcher itself -- so widening the
# window would have silently disagreed with what the audit log claimed was used.
BLOCKING = BlockingConfig(date_slack_days=7)


# An aging report needs a book that spans time. Measured on the held-out set:
# it covers 27 days, every record lands in one 30-day bucket, and the auto-post
# rate across weekly buckets is 74.9 / 80.2 / 81.3 / 80.0 -- flat. Aging counts
# how long an item has sat unresolved in a *running* system; a one-month
# snapshot resolved in a single batch has nothing to measure. An uploaded
# ledger spanning months does, so it is computed there and refused here.
MIN_SPAN_FOR_AGING_DAYS = 60
AGING_BUCKETS = ((0, 30), (30, 60), (60, 90), (90, None))

# Measured, and owned here so exactly one place states it. Median of three warm
# runs each: 2,000 records on an 8-core arm64 laptop, 500 on the deployed free
# instance. Four different figures were live at once before this constant
# existed -- 40 on the page, 45 in the README, and 495 and 524 elsewhere in the
# same README -- because each was typed by hand where it was needed.
FREE_TIER_RECORDS_PER_SECOND = 45
LAPTOP_RECORDS_PER_SECOND = 795


class RunRequest(BaseModel):
    # 200 returns in about four seconds on the deployed free instance. The
    # default was 500 -- a thirteen-second wait for any caller that omits the
    # field, which the page's own control had already stopped offering.
    # Anything larger is clamped to the size of the held-out set.
    limit: int = Field(default=200, gt=0, le=10**9)
    review_all: bool = False
    mult_threshold: float = Field(default=MULT_THRESHOLD, ge=0.0, le=1.0)


class ConnectRequest(BaseModel):
    """A merchant's own test-mode credentials, used once and never stored."""

    key_id: str
    key_secret: str
    year: int = Field(ge=2000, le=2100)
    month: int = Field(ge=1, le=12)
    day: int | None = Field(default=None, ge=1, le=31)


class SettlementRequest(BaseModel):
    limit: int = Field(default=50, gt=0, le=10**6)
    max_pool: int = Field(default=128, ge=2, le=512)




class _State:
    """Loaded once at startup. Holding the models in memory is the point: the
    matching path must not touch a network or a database."""

    def __init__(self) -> None:
        self.ready = False
        self.records: list[BankRecord] = []
        self.truth: dict[str, tuple[str, bool]] = {}
        self.index: KeyIndex | None = None
        self.key_stats: dict = {}
        self.ranker = None
        self.detector = None
        self.prior = None
        self.meta: dict[str, Any] = {}
        self.error: str | None = None
        self.calibrator = None
        self.calibrator_kind = "none"
        self.bundle_meta: dict[str, Any] = {}
        self.models: Models | None = None
        self.overview: dict[str, Any] = {}
        self.training: dict[str, Any] = {}
        self.settlements: list[dict] = []
        self.payments: dict[str, dict] = {}

    def load(self) -> None:
        """Load artifacts, or record why not.

        Never raises. A failure here happens at process start, so an exception
        kills the container before it serves anything and a visitor sees a blank
        page instead of a message. Degrade to a reported 503 instead.
        """
        try:
            self._load()
        except Exception as exc:  # noqa: BLE001
            self.error = f"{type(exc).__name__}: {exc}"
            self.ready = False

    def _load(self) -> None:
        demo_path, model_path = ARTIFACTS / "demo.json", ARTIFACTS / "models.pkl"
        if not (demo_path.exists() and model_path.exists()):
            self.error = f"artifacts not found in {ARTIFACTS}"
            return
        demo = json.loads(demo_path.read_text())
        self.records = [
            BankRecord(r["record_id"], r["account"], r["amount_minor"], r["day"])
            for r in demo["records"]
        ]
        self.truth = {r["record_id"]: (r["truth"], r["is_mult"]) for r in demo["records"]}
        rows = [KeyRow(k["key"], k["account"], k["amount_minor"], k["day"]) for k in demo["key_rows"]]
        self.index = KeyIndex(rows)
        self.key_stats = build_key_stats(rows)
        bundle = pickle.loads(model_path.read_bytes())
        self.ranker, self.detector, self.prior = (
            bundle["ranker"], bundle["detector"], bundle["prior"]
        )
        # Maps the ranker's margin to the frequency with which the top
        # candidate is actually right. Fitted on validation, scored on test:
        # sigmoid(margin) claimed 74.8% where the truth was 21.5%.
        # The bundle is a contract. Unpickling proves only that bytes were
        # written by some version of this code, so what it must contain is
        # checked here rather than discovered on the first request.
        missing = [k for k in ("ranker", "detector", "prior") if bundle.get(k) is None]
        if missing:
            self.error = f"models.pkl is missing: {', '.join(missing)}"
            return
        declared = tuple(bundle.get("feature_names", FEATURE_NAMES))
        if declared != tuple(FEATURE_NAMES):
            # A reordered feature list is the dangerous case: the model scores
            # the wrong columns and raises nothing at all.
            self.error = ("models.pkl was trained against a different feature "
                          f"order ({len(declared)} names, first differing at "
                          f"{next((i for i, (a, b) in enumerate(zip(declared, FEATURE_NAMES, strict=False)) if a != b), 0)})")
            return
        self.bundle_meta = {
            "feature_names": list(declared),
            "calibrator": bundle.get("calibrator_kind", "none"),
        }
        self.calibrator = bundle.get("calibrator")
        self.calibrator_kind = bundle.get("calibrator_kind", "none")
        self.models = Models(self.ranker, self.detector, self.prior,
                             self.calibrator, self.calibrator_kind)
        self.meta = json.loads((ARTIFACTS / "meta.json").read_text())

        ov = ARTIFACTS / "overview.json"
        if ov.exists():
            self.overview = json.loads(ov.read_text())

        tr = ARTIFACTS / "training.json"
        if tr.exists():
            self.training = json.loads(tr.read_text())

        rr = ARTIFACTS / "reconriver.json"
        if rr.exists():
            data = json.loads(rr.read_text())
            self.settlements = data["settlements"]
            self.payments = {p["payment_id"]: p for p in data["payments"]}

        self.ready = True


def create_app() -> FastAPI:
    app = FastAPI(title="Allocation Agent", version="0.1.0")
    state = _State()
    state.load()
    # AUDIT_DB overrides the location so a persistent disk or volume is a
    # deploy setting rather than a code change. Left unset it writes beside the
    # artifacts, which on the free tier is **ephemeral**: the file is excluded
    # from both git and the image, so every deploy starts with an empty log and
    # a spin-down discards it. Fine for a demo whose runs are minutes old;
    # stated on /api/health rather than left for someone to discover.
    audit_path = Path(os.environ.get("AUDIT_DB") or (ARTIFACTS / "runs.db"))
    audit = AuditLog(audit_path)
    # Templates by default -- no key, no cost, no network on any path. A key in
    # the environment switches on the model backend, which was previously
    # impossible: `openrouter.py` read OPENROUTER_API_KEY and nothing ever
    # constructed it, so the README's "add a key for narration" did nothing.
    backend = None
    if os.environ.get("OPENROUTER_API_KEY"):
        try:
            from allocation_agent.decide.openrouter import OpenRouterBackend
            backend = OpenRouterBackend()
        except Exception:  # noqa: BLE001 -- never let narration break matching
            backend = None
    narrator = Narrator(backend=backend)
    app.state.narrator_calls = lambda: narrator.narrated

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _page()

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "models_loaded": state.ready,
                "feature_version": len(state.bundle_meta.get("feature_names", [])),
                "audit_db": str(audit_path),
                "audit_persistent": bool(os.environ.get("AUDIT_DB")),
                "calibrator": state.bundle_meta.get("calibrator", "none"),
                "n_demo_records": len(state.records), "error": state.error}

    @app.get("/api/meta")
    def meta() -> dict:
        return {**state.meta, "n_demo_records": len(state.records),
                "source": state.meta.get("source", "BenchRec (ICAIF 2023)"),
                "free_tier_records_per_second": FREE_TIER_RECORDS_PER_SECOND,
                "laptop_records_per_second": LAPTOP_RECORDS_PER_SECOND}

    @app.get("/api/overview")
    def overview() -> dict:
        """What the page shows before anyone presses anything.

        Precomputed by scripts/export_overview.py over the whole held-out set.
        Running it at startup would take ~90s on the deployed free instance and
        the health check would fail before the container was ready.
        """
        if not state.overview:
            raise HTTPException(503, "overview not exported; run scripts/export_overview.py")
        return state.overview

    @app.get("/api/training")
    def training() -> dict:
        """How the models were trained, and what that produced.

        Precomputed by scripts/export_training_evidence.py, which needs the
        60 MB training CSV the deployed image does not carry. Until this
        existed the evidence for every headline number lived in a terminal.
        """
        if not state.training:
            raise HTTPException(503, "training evidence not exported; "
                                     "run scripts/export_training_evidence.py")
        return state.training

    @app.post("/api/run")
    def run(req: RunRequest) -> dict:
        if not state.ready:
            raise HTTPException(503, f"models not loaded: {state.error or 'unknown'}")

        records = state.records[: min(req.limit, len(state.records))]
        gate = GateConfig(review_all=req.review_all, policy_version="v0.1")

        run_id = audit.start_run(RunConfig(
            approved_by="demo", blocking={"date_slack_days": BLOCKING.date_slack_days},
            gate={"base": gate.base, "slope": gate.slope, "review_all": req.review_all},
            policy_version="v0.1", notes="held-out demo slice"))

        summary = {"posted": 0, "queued": 0, "no_candidate": 0,
                   "suspected_grouped": 0, "unscorable": 0, "model_error": 0}
        # Value, not only counts. A reconciliation queue is worked by amount at
        # risk, so the money is the reportable figure and the count is context.
        value = dict.fromkeys(summary, 0.0)
        unresolved: list[tuple[int, float]] = []
        exceptions: list[dict] = []
        posted_correct = 0
        # Measured, not asserted. "No LLM on the matching path" is the design
        # claim; this is the counter that would show it false. The narrator
        # runs after decide(), on queued records only, and writes prose -- it
        # cannot change which key was chosen.
        llm_before = narrator.calls
        started = time.perf_counter()

        try:
            for rec in records:
                truth_key, truly_mult = state.truth.get(rec.record_id, ("", False))
                # Same function the upload path calls. One matching path, so the
                # measured demo numbers say something about uploaded files too.
                r = _match_or_degrade(rec, index=state.index, key_stats=state.key_stats,
                                      models=state.models, gate=gate,
                                      mult_threshold=req.mult_threshold,
                                      blocking=BLOCKING, narrator=narrator,
                                      calibrated=True)
                audit.record(rec.record_id, r["decision"], keys=r["keys"],
                             n_candidates=r["n_candidates"], path=r["path"],
                             evidence=r["evidence"], run_id=run_id)
                summary[r["outcome"]] += 1
                value[r["outcome"]] += abs(rec.amount_minor) / 100
                if r["outcome"] != "posted" and rec.day is not None:
                    unresolved.append((rec.day, abs(rec.amount_minor) / 100))

                if r["outcome"] == "posted":
                    posted_correct += (not truly_mult and r["keys"][0] == truth_key)
                else:
                    exceptions.append({**_exception(rec, r["outcome"], r["explanation"],
                                                    r["n_candidates"]),
                                       "stage": r["stage"],
                                       "residual": round(r["residual_minor"] / 100, 2),
                                       "residual_cause": r["residual_cause"]})
        except Exception as exc:  # noqa: BLE001
            # A run that stops halfway must not look like one still
            # going. Whatever it wrote is kept and the row says why.
            audit.commit()
            audit.fail_run(run_id, f"{type(exc).__name__}: {exc}")
            raise HTTPException(500, f"run {run_id} failed: {type(exc).__name__}") from exc

        audit.commit()
        audit.finish_run(run_id)
        elapsed = time.perf_counter() - started
        # Biggest first: a controller works the queue top-down, and the ten
        # largest are about a tenth of its value. Record order wastes that.
        exceptions.sort(key=lambda e: -abs(e["amount"]))
        # What a reviewer needs on top of "why it stopped": which to open first,
        # and how much of the backlog the first few would clear. Shares are
        # taken against the *whole* queue, not the 100 returned below, or they
        # would sum to 1 while describing a quarter of it.
        queue_total = sum(v for k, v in value.items() if k != "posted")
        running = 0.0
        for e in exceptions:
            share = abs(e["amount"]) / queue_total if queue_total else 0.0
            running += share
            e["share_of_queue"] = round(share, 6)
            e["cumulative_share"] = round(running, 6)

        return {
            "run_id": run_id,
            "n_records": len(records),
            "summary": summary,
            "posted_correct": posted_correct,
            "precision_of_posted": posted_correct / summary["posted"] if summary["posted"] else 0.0,
            "straight_through_rate": summary["posted"] / len(records) if records else 0.0,
            "records_per_second": len(records) / elapsed if elapsed else 0.0,
            "seconds": round(elapsed, 3),
            "llm_calls_on_matching_path": narrator.calls - llm_before,
            "mult_threshold": req.mult_threshold,
            "posted_value": round(value["posted"], 2),
            # Summed over every exception, not over the 100 returned below --
            # a total taken from the truncated list describes a fraction of it.
            "queue_value": round(sum(v for k, v in value.items() if k != "posted"), 2),
            "value_by_outcome": {k: round(v, 2) for k, v in value.items()},
            "aging": _aging(unresolved, _span([r.day for r in records])),
            "exceptions": exceptions[:100],
            "n_exceptions": len(exceptions),
        }

    @app.post("/api/connect")
    def connect(req: ConnectRequest) -> dict:
        """Recover a merchant's own settlement batches from their Razorpay data.

        `GET /v1/settlements/recon/combined` returns every settled line with the
        `settlement_id` it was paid out under. Grouped, that is this project's
        hard case with real money: several payments arriving as one bank credit.

        **The settlement id is withheld from the solver.** It gets the credit
        and a pool of that period's payments and has to recover the subset, so
        the result is a measurement rather than a lookup — the same contract
        the synthetic set runs under.

        Test-mode keys only. The secret is used for one request to Razorpay and
        is never stored, logged or returned.
        """
        try:
            lines = fetch_recon(req.key_id, req.key_secret, year=req.year,
                                month=req.month, day=req.day, fetch=_rzp_fetch)
        except RazorpayError as exc:
            raise HTTPException(400, str(exc)) from None

        settlements = group_into_settlements(lines)
        if not settlements:
            return {"n_settlements": 0, "n_lines": len(lines), "results": [],
                    "note": ("Connected, but this account has no settlements in "
                             f"{req.year}-{req.month:02d}. Test-mode settlements "
                             "appear once test payments have been captured and "
                             "settled — try a period with activity.")}

        # The candidate pool is every settled line in the period, which is what
        # a reconciler would actually be holding: the batch is not marked.
        pool = [(item.entity_id, item.net_minor) for item in lines
                if item.net_minor > 0]
        amounts = [amount for _, amount in pool]
        ids = [entity for entity, _ in pool]
        cfg = SolverConfig(max_candidates=128)

        out, exact = [], 0
        for st in settlements[:50]:
            r = solve_subset(target_minor=st["amount_minor"],
                             candidates_minor=amounts, config=cfg)
            chosen = [ids[i] for i in r.indices]
            is_exact = sorted(chosen) == sorted(st["truth"])
            exact += is_exact
            if is_exact:
                verdict, tone = "Recovered your batch", "good"
            elif r.status is SolverStatus.SOLVED:
                verdict, tone = "Balances but is not the batch", "bad"
            elif r.status is SolverStatus.AMBIGUOUS:
                verdict, tone = "Two groups fit — refused to guess", "warn"
            else:
                verdict, tone = "Could not resolve", "warn"
            out.append({
                "settlement_id": st["settlement_id"], "verdict": verdict, "tone": tone,
                "amount": round(st["amount_minor"] / 100, 2),
                "currency": st["currency"], "exact": is_exact,
                "true_size": st["n_lines"], "pool_size": len(pool),
                "components": [{"payment_id": p,
                                "amount": round(dict(pool)[p] / 100, 2)}
                               for p in chosen],
            })

        return {
            "n_settlements": len(settlements), "n_lines": len(lines),
            "n_solved": len(out), "exact": exact,
            "period": f"{req.year}-{req.month:02d}" + (f"-{req.day:02d}" if req.day else ""),
            "note": ("Your own settlements, recovered from the payment pool "
                     "without the batch id. Amounts are in rupees."),
            "results": out,
        }

    @app.get("/api/stream")
    def stream(limit: int = 200, mult_threshold: float = MULT_THRESHOLD):
        """The same reconciliation, one decision at a time.

        A batch that computes for six seconds and then prints a table asks a
        viewer to take the pipeline on trust. This emits each payment's verdict
        as it is made, so the counters move and the queue fills while the work
        happens — and there is a test asserting it reaches the same verdicts as
        the batch endpoint, because a demo path that diverged from the measured
        one would make the measurement evidence for nothing.
        """
        if not state.ready:
            raise HTTPException(503, f"models not loaded: {state.error or 'unknown'}")

        records = state.records[: max(1, min(limit, len(state.records)))]
        gate = GateConfig(policy_version="v0.1")
        run_id = audit.start_run(RunConfig(
            approved_by="demo", blocking={"date_slack_days": BLOCKING.date_slack_days},
            gate={"base": gate.base, "slope": gate.slope},
            policy_version="v0.1", notes="streamed"))

        def events():
            summary = {"posted": 0, "queued": 0, "no_candidate": 0,
                       "suspected_grouped": 0, "unscorable": 0, "model_error": 0}
            value = dict.fromkeys(summary, 0.0)
            correct = 0
            started = time.perf_counter()
            try:
                for i, rec in enumerate(records, 1):
                    r = _match_or_degrade(rec, index=state.index,
                                          key_stats=state.key_stats,
                                          models=state.models, gate=gate,
                                          mult_threshold=mult_threshold,
                                          blocking=BLOCKING, narrator=narrator,
                                          calibrated=True)
                    audit.record(rec.record_id, r["decision"], keys=r["keys"],
                                 n_candidates=r["n_candidates"], path=r["path"],
                                 evidence=r["evidence"], run_id=run_id)
                    summary[r["outcome"]] += 1
                    amount = abs(rec.amount_minor) / 100
                    value[r["outcome"]] += amount
                    truth, is_mult = state.truth.get(rec.record_id, ("", False))
                    if r["outcome"] == "posted":
                        correct += (not is_mult) and r["keys"][0] == truth
                    yield "data: " + json.dumps({
                        "type": "record", "i": i,
                        "record_id": rec.record_id,
                        "amount": round(amount, 2),
                        "outcome": r["outcome"],
                        "stage": r["stage"],
                        "matched_key": r["keys"][0] if r["keys"] else None,
                        "confidence": r["confidence"],
                        "explanation": r["explanation"],
                    }) + "\n\n"

                audit.commit()
                audit.finish_run(run_id)
                elapsed = time.perf_counter() - started
                yield "data: " + json.dumps({
                    "type": "done", "run_id": run_id, "n_records": len(records),
                    "summary": summary,
                    "value_by_outcome": {k: round(v, 2) for k, v in value.items()},
                    "precision_of_posted": correct / summary["posted"] if summary["posted"] else 0.0,
                    "straight_through_rate": summary["posted"] / len(records),
                    "records_per_second": len(records) / elapsed if elapsed else 0.0,
                    "seconds": round(elapsed, 3),
                }) + "\n\n"
            except Exception as exc:  # noqa: BLE001
                audit.commit()
                audit.fail_run(run_id, f"{type(exc).__name__}: {exc}")
                yield "data: " + json.dumps({
                    "type": "error", "run_id": run_id,
                    "detail": f"{type(exc).__name__}"}) + "\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    @app.post("/api/settlements")
    def settlements(req: SettlementRequest) -> dict:
        """Recover which payments produced each bank credit.

        The batch identifier is never shown to the solver: it gets the credit
        amount and a pool of plausible payments and must find the subset.

        A subset that sums correctly is **not** evidence it is the right subset --
        with a pool of ~100 payments many subsets hit any given total. The first
        version of this endpoint reported "solved" for all of them; 51.3% were
        the wrong set. So two numbers are reported, never one: **coverage** (how
        many credits got an answer) and **precision** (how many of those answers
        were the recorded batch). Ties are returned as ambiguous, not resolved.
        """
        if not state.settlements:
            raise HTTPException(503, "settlement demo data not loaded")

        # Evenly spaced, not the first N. ReconRiver front-loads its hard
        # cases -- 11 of the first 15 credits are unsolvable and none of
        # credits 30-149 are -- so taking the head is systematically the worst
        # possible sample and made the component look broken to anyone who
        # asked for a few. Spacing is by position and never by outcome:
        # selecting the ones that succeed is the cherry-picking this project
        # already got wrong once.
        total = len(state.settlements)
        want = min(req.limit, total)
        if want >= total:
            items, sampling = state.settlements, "all"
        else:
            step = total / want
            items = [state.settlements[int(i * step)] for i in range(want)]
            sampling = "evenly spaced"
        cfg = SolverConfig(tolerance_minor=0, max_candidates=req.max_pool)
        out, solved, exact, wrong, ambiguous, unresolved = [], 0, 0, 0, 0, 0
        # Per true batch size. Which credits it gets wrong turns out to matter
        # more than how many, and one aggregate number hides it entirely.
        sizes: dict[int, dict[str, int]] = {}
        unreachable = 0
        started = time.perf_counter()

        for st in items:
            pool_ids = [p for p in st["pool"] if p in state.payments]
            amounts = [state.payments[p]["amount_minor"] for p in pool_ids]
            r = solve_subset(target_minor=st["amount_minor"], candidates_minor=amounts, config=cfg)

            chosen = [pool_ids[i] for i in r.indices]
            is_exact = sorted(chosen) == sorted(st["truth"])

            true_size = len(st["truth"])
            bucket = sizes.setdefault(true_size, {"n": 0, "recovered": 0, "wrong": 0,
                                                  "ambiguous": 0, "unresolved": 0,
                                                  "unreachable": 0})
            bucket["n"] += 1
            # Blocking never offered the answer, so no solver could find it.
            # Counting this against the solver blames the wrong component.
            if not set(st["truth"]).issubset(pool_ids) or len(pool_ids) > req.max_pool:
                bucket["unreachable"] += 1
                unreachable += 1

            if r.status is SolverStatus.SOLVED:
                solved += 1
                exact += is_exact
                wrong += (not is_exact)
                bucket["recovered" if is_exact else "wrong"] += 1
                # The arithmetic is the page's headline and its itemised list.
                # Restating it here made a recovered credit read as three copies
                # of one fact, with the sentence carrying the actual judgement
                # buried at the end. This says only what the sum cannot.
                status = "solved"
                if is_exact:
                    expl = (f"These {len(chosen)} payments are the ones the settlement "
                            f"file records for this credit, found from a pool of "
                            f"{len(pool_ids)} without being told the batch number.")
                else:
                    expl = ("This balances but is not the recorded batch — a summing "
                            "subset is not proof of the right subset, so it goes to "
                            "review.")
            elif r.status is SolverStatus.AMBIGUOUS:
                ambiguous += 1
                bucket["ambiguous"] += 1
                status, expl = ("ambiguous",
                    f"Two different sets of {r.subset_size} payments both reach "
                    f"{st['amount_minor'] / 100:,.2f} exactly. The amounts do not choose "
                    "between them, so neither is claimed.")
            elif r.status is SolverStatus.TOO_LARGE:
                unresolved += 1
                bucket["unresolved"] += 1
                # TOO_LARGE covers an oversized pool *and* an oversized target.
                # One sentence naming the pool cap is simply false for the other.
                if r.detail.startswith("pool"):
                    expl = (f"{r.n_considered} candidate payments exceeds the "
                            f"{req.max_pool} cap. Refused rather than truncated to fit.")
                else:
                    expl = (f"This credit is larger than the solver will search over "
                            f"({r.detail}). Refused rather than approximated.")
                status = "unresolved"
            else:
                unresolved += 1
                bucket["unresolved"] += 1
                cap = min(cfg.max_subset_size, r.n_considered)
                nxt = cap + 1
                status, expl = ("unresolved",
                    f"No group of {cap} payments or fewer adds up to this credit. "
                    f"Longer combinations may exist and are deliberately not claimed — "
                    f"there are far more ways to pick {nxt} payments out of "
                    f"{r.n_considered} than {cap}, so a longer sum that happens to hit "
                    f"the total is a coincidence rather than evidence.")

            # The label is a judgement about the answer, so it is made here.
            # Deriving it from `status` alone in the page produced a badge
            # reading "Found the group" above a sentence explaining that the
            # group was wrong -- `solved` means a subset balances, not that it
            # is the right subset.
            # `is_exact` compares two sorted lists, so an empty answer against
            # an empty truth is True. Testing it before `status` badged a
            # record with no answer at all as "Found the group".
            if status == "solved" and is_exact:
                verdict, tone = "Found the group", "good"
            elif status == "solved":
                verdict, tone = "Wrong group — sent to review", "bad"
            elif status == "ambiguous":
                verdict, tone = "Two groups fit — refused to guess", "warn"
            else:
                verdict, tone = "Could not resolve", "warn"

            out.append({
                "verdict": verdict, "tone": tone,
                "settlement_id": st["settlement_id"],
                "amount": round(st["amount_minor"] / 100, 2),
                "currency": st["currency"], "booked_at": st["booked_at"],
                "status": status, "exact": is_exact, "pool_size": len(pool_ids),
                "components": [{"payment_id": p, "amount": round(state.payments[p]["amount_minor"] / 100, 2),
                                "order_id": state.payments[p].get("order_id", "")} for p in chosen],
                "true_size": len(st["truth"]),
                "explanation": expl,
            })

        n = len(items)
        return {
            "sampling": sampling,
            "n_settlements": n, "solved": solved, "ambiguous": ambiguous,
            "unresolved": unresolved,
            # coverage: got an answer at all. precision: that answer was right.
            # Reporting either one alone is how a solver looks good and is not.
            "exact_recovery_rate": exact / n if n else 0.0,
            "precision": exact / solved if solved else 0.0,
            "wrong_set_rate": wrong / n if n else 0.0,
            "unreachable": unreachable,
            # A percentage over a handful is arithmetic, not evidence -- and
            # *what* is being rated decides what counts as a handful. Recovery
            # is a rate over records; precision is a rate over answers. Gating
            # both on the record count let 20 records yielding 3 answers print
            # "0.0% of the groups it named were right" as though measured.
            "min_for_rates": MIN_FOR_RATES,
            "recovery_rate_meaningful": n >= MIN_FOR_RATES,
            "precision_meaningful": solved >= MIN_FOR_RATES,
            # The counts behind each rate, so the page never has to recover an
            # integer by multiplying a float back by its denominator.
            "exact": exact,
            "wrong": wrong,
            "by_batch_size": [dict(size=k, **v) for k, v in sorted(sizes.items())],
            "seconds": round(time.perf_counter() - started, 3),
            "results": out,
        }

    @app.get("/api/run/{run_id}/audit")
    def audit_trail(run_id: str) -> dict:
        try:
            meta = audit.get_run(run_id)
        except KeyError:
            raise HTTPException(404, f"no run {run_id}") from None
        # The run's own row comes back with the decisions. Whether the batch
        # completed is part of reading the trail, not a separate lookup -- a
        # partial trail with no status is indistinguishable from a whole one.
        return {"run_id": run_id, "run": meta,
                "decisions": audit.decisions(run_id=run_id)}

    @app.post("/api/reconcile")
    async def reconcile(bank: UploadFile = File(...),
                        ledger: UploadFile = File(...)) -> dict:
        """Reconcile a bank CSV against a ledger CSV the visitor supplies.

        Same matching path as the demo -- deliberately, because a separate code
        path for user data would make the demo's measured numbers evidence for
        nothing but the demo.

        **No precision is reported here and that is not an omission.** The demo
        can score itself because BenchRec is labelled; an uploaded file has no
        answer key, so any accuracy figure would be invented. What is reported
        is what it decided and why, for the visitor to check against what they
        already know about their own data.
        """
        if not state.ready:
            raise HTTPException(503, f"models not loaded: {state.error or 'unknown'}")

        async def read(f: UploadFile, label: str) -> str:
            raw = await f.read()
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    400, f"the {label} file exceeds "
                         f"{MAX_UPLOAD_BYTES // 1024 // 1024} MB")
            return raw.decode("utf-8", "replace")

        try:
            records, bank_layout = parse_bank_csv(await read(bank, "bank"),
                                                  report_layout=True)
            rows, ledger_layout = parse_ledger_csv(await read(ledger, "ledger"),
                                                   report_layout=True)
        except UploadError as exc:
            raise HTTPException(400, str(exc)) from exc

        if len(rows) > MAX_LEDGER_ROWS:
            raise HTTPException(400, f"the ledger file has {len(rows):,} rows; "
                                     f"this demo caps at {MAX_LEDGER_ROWS:,}")
        if len(records) > MAX_UPLOAD_ROWS:
            raise HTTPException(400, f"the bank file has {len(records)} rows; "
                                     f"this demo caps at {MAX_UPLOAD_ROWS}")

        index, key_stats = KeyIndex(rows), build_key_stats(rows)
        gate = GateConfig(policy_version="v0.1")
        run_id = audit.start_run(RunConfig(
            approved_by="uploader", blocking={"date_slack_days": BLOCKING.date_slack_days},
            gate={"base": gate.base, "slope": gate.slope},
            policy_version="v0.1", notes=f"uploaded: {bank.filename} x {ledger.filename}"))

        summary = {"posted": 0, "queued": 0, "no_candidate": 0,
                   "suspected_grouped": 0, "unscorable": 0, "model_error": 0}
        unresolved: list[tuple[int, float]] = []
        llm_before = narrator.calls
        results = []
        started = time.perf_counter()

        def work() -> None:
            """The matching loop, off the event loop.

            This endpoint is `async def` because it awaits the uploads, and
            everything after that is CPU-bound: blocking, featurising and
            scoring up to 20,000 rows. Running it inline stalls every other
            request for the duration. It is also where the narrator runs, and
            with a key configured that reaches a synchronous HTTP call with a
            long timeout -- so one slow model response would freeze the server
            rather than one request. `/api/run` never had the problem because a
            sync endpoint is already dispatched to a worker thread.
            """
            for rec in records:
                # calibrated=False: the calibrator and DIRECT_CONFIDENCE were both
                # measured on BenchRec. Sharing a code path with this file does not
                # transfer those measurements to it.
                r = _match_or_degrade(rec, index=index, key_stats=key_stats,
                                      models=state.models, gate=gate,
                                      mult_threshold=MULT_THRESHOLD, blocking=BLOCKING,
                                      narrator=narrator, calibrated=False)
                audit.record(rec.record_id, r["decision"], keys=r["keys"],
                             n_candidates=r["n_candidates"], path=r["path"],
                             evidence=r["evidence"], run_id=run_id)
                summary[r["outcome"]] += 1
                if r["outcome"] != "posted" and rec.day is not None:
                    unresolved.append((rec.day, abs(rec.amount_minor) / 100))
                results.append({
                    "record_id": rec.record_id,
                    "account": rec.account,
                    "amount": round(rec.amount_minor / 100, 2),
                    "outcome": r["outcome"],
                    "matched_key": r["keys"][0] if r["keys"] else None,
                    "confidence": r["confidence"],
                    "n_candidates": r["n_candidates"],
                    "residual": round(r["residual_minor"] / 100, 2),
                    "residual_cause": r["residual_cause"],
                    "explanation": r["explanation"],
                })

        try:
            await asyncio.to_thread(work)
        except Exception as exc:  # noqa: BLE001
            audit.commit()
            audit.fail_run(run_id, f"{type(exc).__name__}: {exc}")
            raise HTTPException(500, f"run {run_id} failed: {type(exc).__name__}") from exc

        audit.commit()
        audit.finish_run(run_id)
        return {
            "run_id": run_id,
            "n_records": len(records),
            "n_ledger_rows": len(rows),
            "summary": summary,
            "seconds": round(time.perf_counter() - started, 3),
            "llm_calls_on_matching_path": narrator.calls - llm_before,
            # 03/01/2026 is the third of January or the first of March. One
            # reading is chosen for the whole file; saying which is the only way
            # the person who wrote it can catch a wrong guess.
            "date_layout": {"bank": bank_layout, "ledger": ledger_layout},
            "aging": _aging(unresolved, _span([r.day for r in records])),
            # Every confidence on this response was derived from measurements
            # taken on BenchRec. Sharing a code path does not transfer them.
            "confidence_validated_for_this_data": False,
            "caveat": "Your file has no ground truth, so no precision is reported — "
                      "any accuracy number here would be invented. Check the matches "
                      "against what you already know about this data.",
            "results": results,
        }

    return app





def _match_or_degrade(rec, *, index, key_stats, models, gate, mult_threshold,
                      blocking, narrator, calibrated: bool) -> dict:
    """match_one, but a model failure becomes an exception rather than a 500.

    "The AI can fail, the payment system cannot" is the project's central
    claim, and until now it was enforced in the batch runner and nowhere else:
    a ranker or detector that raised took the whole HTTP request with it.
    """
    try:
        return match_one(rec, index=index, key_stats=key_stats, models=models,
                         gate=gate, mult_threshold=mult_threshold,
                         blocking=blocking, narrator=narrator,
                         calibrated_for_this_data=calibrated)
    except Exception as exc:  # noqa: BLE001
        d = decide(confidence=None, amount_minor=rec.amount_minor, config=gate)
        return {"residual_cause": None, "residual_minor": 0, "stage": "model",
                "outcome": "model_error", "decision": d, "keys": [],
                "n_blocked": 0, "n_scored": 0, "n_candidates": 0,
                "path": "model_error", "evidence": {"error": type(exc).__name__},
                "confidence": None,
                "explanation": (f"A model failed on this record "
                                f"({type(exc).__name__}). Routed to a person "
                                f"rather than guessed at.")}


def _span(days: list[int | None]) -> int:
    known = [d for d in days if d is not None]
    return max(known) - min(known) if known else 0


def _aging(unresolved: list[tuple[int, float]], span: int) -> dict:
    """Unresolved value by how long it has been waiting.

    *unresolved* is (day, amount) for records that did **not** reconcile --
    aging something already matched is meaningless, it is not waiting for
    anyone.
    """
    if span < MIN_SPAN_FOR_AGING_DAYS:
        return {
            "meaningful": False, "span_days": span, "buckets": [],
            "note": (f"This data covers {span} days, so every item falls in one "
                     f"bucket. Aging measures how long something has sat "
                     f"unresolved in a running book; a snapshot this short has "
                     f"nothing to age. Shown for a ledger spanning "
                     f"{MIN_SPAN_FOR_AGING_DAYS}+ days."),
        }

    latest = max((d for d, _ in unresolved), default=0)
    buckets = []
    for lo, hi in AGING_BUCKETS:
        chosen = [(d, a) for d, a in unresolved
                  if lo <= latest - d and (hi is None or latest - d < hi)]
        buckets.append({
            "label": f"{lo}-{hi} days" if hi else f"{lo}+ days",
            "count": len(chosen),
            "value": round(sum(a for _, a in chosen), 2),
        })
    return {"meaningful": True, "span_days": span, "buckets": buckets,
            "note": "Measured from the most recent entry in the file."}




def _exception(rec: BankRecord, reason: str, explanation: str, n_candidates: int) -> dict:
    return {"record_id": rec.record_id, "reason": reason, "explanation": explanation,
            "amount": round(rec.amount_minor / 100, 2), "account": rec.account,
            "n_candidates": n_candidates}


_PAGE_PATH = Path(__file__).resolve().parent / "static" / "index.html"


def _page() -> str:
    """Read the page from disk each request.

    Costs a few microseconds on a route that runs once per visitor, and means
    the demo page can be corrected without a redeploy of the matching code.
    """
    try:
        return _PAGE_PATH.read_text()
    except OSError as exc:
        return f"<pre>demo page missing: {exc}</pre>"
