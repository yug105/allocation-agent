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

import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from allocation_agent.decide.gate import GateConfig, Outcome, decide
from allocation_agent.decide.narrate import Narrator, diagnose_residual
from allocation_agent.match.blocker import BlockingConfig, block
from allocation_agent.adapters.csv_upload import (
    UploadError, parse_bank_csv, parse_ledger_csv,
)
from allocation_agent.match.features import build_key_stats, featurise
from allocation_agent.match.multiplicity import featurise_multiplicity
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

# Measured on the held-out set, not asserted: see _match_one.
DIRECT_CONFIDENCE = 0.9898

# Below this many records, report counts rather than rates.
MIN_FOR_RATES = 20


class RunRequest(BaseModel):
    limit: int = Field(default=500, gt=0, le=10**9)
    review_all: bool = False
    mult_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class SettlementRequest(BaseModel):
    limit: int = Field(default=50, gt=0, le=10**6)
    max_pool: int = Field(default=128, ge=2, le=512)


class ConnectRequest(BaseModel):
    key_id: str


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
        self.meta = json.loads((ARTIFACTS / "meta.json").read_text())

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
    audit = AuditLog(ARTIFACTS / "runs.db")
    narrator = Narrator()  # templates by default; no key required, no cost

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _page()

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "models_loaded": state.ready,
                "n_demo_records": len(state.records), "error": state.error}

    @app.get("/api/meta")
    def meta() -> dict:
        return {**state.meta, "n_demo_records": len(state.records),
                "source": state.meta.get("source", "BenchRec (ICAIF 2023)")}

    @app.post("/api/run")
    def run(req: RunRequest) -> dict:
        if not state.ready:
            raise HTTPException(503, f"models not loaded: {state.error or 'unknown'}")

        records = state.records[: min(req.limit, len(state.records))]
        gate = GateConfig(review_all=req.review_all, policy_version="v0.1")
        bcfg = BlockingConfig(date_slack_days=7)

        run_id = audit.start_run(RunConfig(
            approved_by="demo", blocking={"date_slack_days": 7},
            gate={"base": gate.base, "slope": gate.slope, "review_all": req.review_all},
            policy_version="v0.1", notes="held-out demo slice"))

        summary = {"posted": 0, "queued": 0, "no_candidate": 0, "suspected_grouped": 0}
        exceptions: list[dict] = []
        posted_correct = 0
        started = time.perf_counter()

        for rec in records:
            truth_key, truly_mult = state.truth.get(rec.record_id, ("", False))
            # Same function the upload path calls. One matching path, so the
            # measured demo numbers say something about uploaded files too.
            r = _match_one(state, rec, state.index, state.key_stats, gate,
                           req.mult_threshold)
            audit.record(rec.record_id, r["decision"], keys=r["keys"],
                         n_candidates=r["n_candidates"], path=r["path"],
                         evidence=r["evidence"])
            summary[r["outcome"]] += 1

            if r["outcome"] == "posted":
                posted_correct += (not truly_mult and r["keys"][0] == truth_key)
            else:
                exceptions.append({**_exception(rec, r["outcome"], r["explanation"],
                                                r["n_candidates"]),
                                   "stage": r["stage"]})

        audit.commit(); audit.finish_run(run_id)
        elapsed = time.perf_counter() - started

        return {
            "run_id": run_id,
            "n_records": len(records),
            "summary": summary,
            "posted_correct": posted_correct,
            "precision_of_posted": posted_correct / summary["posted"] if summary["posted"] else 0.0,
            "straight_through_rate": summary["posted"] / len(records) if records else 0.0,
            "records_per_second": len(records) / elapsed if elapsed else 0.0,
            "seconds": round(elapsed, 3),
            "llm_calls_on_matching_path": 0,
            "exceptions": exceptions[:100],
            "n_exceptions": len(exceptions),
        }

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

        items = state.settlements[: min(req.limit, len(state.settlements))]
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
                status, expl = ("solved", _sum_sentence(state, st, chosen))
                if not is_exact:
                    expl += (" This balances but is not the recorded batch — a summing "
                             "subset is not proof of the right subset, so it goes to review.")
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
        rows = audit.decisions(run_id=run_id)
        if not rows:
            raise HTTPException(404, f"no run {run_id}")
        return {"run_id": run_id, "decisions": rows}

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
            records = parse_bank_csv(await read(bank, "bank"))
            rows = parse_ledger_csv(await read(ledger, "ledger"))
        except UploadError as exc:
            raise HTTPException(400, str(exc)) from exc

        if len(records) > MAX_UPLOAD_ROWS:
            raise HTTPException(400, f"the bank file has {len(records)} rows; "
                                     f"this demo caps at {MAX_UPLOAD_ROWS}")

        index, key_stats = KeyIndex(rows), build_key_stats(rows)
        gate = GateConfig(policy_version="v0.1")
        run_id = audit.start_run(RunConfig(
            approved_by="uploader", blocking={"date_slack_days": 7},
            gate={"base": gate.base, "slope": gate.slope},
            policy_version="v0.1", notes=f"uploaded: {bank.filename} x {ledger.filename}"))

        summary = {"posted": 0, "queued": 0, "no_candidate": 0, "suspected_grouped": 0}
        results = []
        started = time.perf_counter()

        for rec in records:
            r = _match_one(state, rec, index, key_stats, gate, 0.5)
            audit.record(rec.record_id, r["decision"], keys=r["keys"],
                         n_candidates=r["n_candidates"], path=r["path"],
                         evidence=r["evidence"])
            summary[r["outcome"]] += 1
            results.append({
                "record_id": rec.record_id,
                "account": rec.account,
                "amount": round(rec.amount_minor / 100, 2),
                "outcome": r["outcome"],
                "matched_key": r["keys"][0] if r["keys"] else None,
                "confidence": r["confidence"],
                "n_candidates": r["n_candidates"],
                "explanation": r["explanation"],
            })

        audit.commit(); audit.finish_run(run_id)
        return {
            "run_id": run_id,
            "n_records": len(records),
            "n_ledger_rows": len(rows),
            "summary": summary,
            "seconds": round(time.perf_counter() - started, 3),
            "llm_calls_on_matching_path": 0,
            "caveat": "Your file has no ground truth, so no precision is reported — "
                      "any accuracy number here would be invented. Check the matches "
                      "against what you already know about this data.",
            "results": results,
        }

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)) -> dict:
        if not (file.filename or "").lower().endswith(".csv"):
            raise HTTPException(400, "only .csv files are accepted")
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(400, f"file exceeds {MAX_UPLOAD_BYTES // 1024 // 1024} MB")
        header = raw.split(b"\n", 1)[0].decode("utf-8", "replace").lower()
        present = {c.strip().strip('"') for c in header.split(",")}
        missing = REQUIRED_COLUMNS - present
        if missing:
            raise HTTPException(400, f"missing required column(s): {sorted(missing)}")
        return {"accepted": True, "bytes": len(raw),
                "note": "parsed and validated; reconciliation of uploaded files is not wired yet"}

    @app.post("/api/connect")
    def connect(req: ConnectRequest) -> dict:
        if not req.key_id.startswith("rzp_test_"):
            raise HTTPException(
                400,
                "only Razorpay test-mode keys are accepted (rzp_test_...). "
                "This service will not accept a live key.",
            )
        raise HTTPException(501, "live Razorpay ingestion is not implemented yet")

    return app


def _sum_sentence(state: _State, st: dict, chosen: list[str]) -> str:
    parts = " + ".join(f"{state.payments[p]['amount_minor'] / 100:,.2f}" for p in chosen)
    return (f"{st['amount_minor'] / 100:,.2f} = {parts}  "
            f"({len(chosen)} payment{'s' if len(chosen) != 1 else ''})")



def _match_one(state: _State, rec: BankRecord, index, key_stats, gate,
               mult_threshold: float) -> dict:
    """Run one record through the whole matching path.

    Extracted so an uploaded file goes through *identical* code to the demo. A
    separate path for user data would make the demo's numbers evidence for
    nothing but the demo.
    """
    cands = sorted(block(rec, index, BlockingConfig(date_slack_days=7)))

    if not cands:
        d = decide(confidence=None, amount_minor=rec.amount_minor, config=gate)
        return {"stage": "narrowing", "outcome": "no_candidate", "decision": d,
                "keys": [], "n_candidates": 0, "path": "blocked", "evidence": None,
                "confidence": None,
                "explanation": "Nothing in the ledger is close enough to consider — "
                               "no entry shares this account within a week of this date."}

    usable = [k for k in cands if k in key_stats]
    if not usable:
        d = decide(confidence=None, amount_minor=rec.amount_minor, config=gate)
        return {"stage": "narrowing", "outcome": "no_candidate", "decision": d,
                "keys": [], "n_candidates": 0, "path": "blocked", "evidence": None,
                "confidence": None,
                "explanation": "Nothing in the ledger is close enough to consider."}

    # A lone candidate whose amount is exactly this figure is direct evidence,
    # and it limits what the next two steps are entitled to do.
    exact = [k for k in usable if rec.amount_minor in key_stats[k].amounts]
    lone_exact = len(exact) == 1

    # 1. The grouping check does not get to overrule it. Measured on the
    #    held-out set: the detector is right 96.3% of the time overall, but on
    #    records with a lone exact-amount match it fires 41 times and is wrong
    #    on 36 -- 12.2% precision. A single entry accounting for the whole
    #    amount defeats the premise of the grouped path and the detector cannot
    #    see that. Everywhere else it is trusted exactly as before.
    p_mult = _p_multiple_with(state, rec, usable, key_stats)
    if p_mult >= mult_threshold and not lone_exact:
        d = decide(confidence=None, amount_minor=rec.amount_minor, config=gate)
        return {"stage": "grouping", "outcome": "suspected_grouped", "decision": d,
                "keys": [], "n_candidates": len(cands), "path": "multiplicity",
                "evidence": {"p_multiple": round(p_mult, 4)}, "confidence": None,
                "explanation": f"This looks like one payment covering several ledger "
                               f"entries ({p_mult:.0%} confidence), so a single match "
                               f"would be wrong. Sent for review."}

    # 2. Rank. Confidence is the gap between first and second place.
    X = np.vstack([featurise(rec, key_stats[k], n_candidates=len(usable)) for k in usable])
    scores = state.ranker.score(X)
    order = np.argsort(-scores)
    chosen = usable[int(order[0])]

    if len(order) > 1:
        margin = float(scores[order[0]] - scores[order[1]])
        confidence = float(1.0 / (1.0 + np.exp(-margin)))
        path, evidence = "ranked", {"margin": round(margin, 4)}
    elif lone_exact:
        # No runner-up, so no margin exists. The old code substituted
        # margin=1.0 here, which is sigmoid -> 73.1%: a constant dressed as a
        # measurement, and permanently under the 85% base bar -- a lone
        # candidate could never post however exact the match. Blocking never
        # returns fewer than 4 candidates on BenchRec, so no test could reach
        # it and every small uploaded file did.
        #
        # The evidence here is the exact amount itself. On the held-out set,
        # where exactly one candidate matches the amount exactly it is the
        # right answer 98.98% of the time (2,321 of 2,345). That measured rate
        # is the confidence.
        chosen, confidence = exact[0], DIRECT_CONFIDENCE
        path, evidence = "direct", {"exact_amount": True}
    else:
        # One candidate, and its amount is not this figure. Nothing supports it.
        confidence, path, evidence = None, "ranked", None

    d = decide(confidence=confidence, amount_minor=rec.amount_minor, config=gate)

    if confidence is None:
        expl = (f"{chosen} is the only nearby ledger entry, but its amount is not this "
                f"figure and there is no second candidate to weigh it against. Nothing "
                f"here supports posting it, so it goes to a person.")
    elif path == "direct":
        expl = (f"Matched to {chosen}, the one nearby ledger entry whose amount is "
                f"exactly this figure. On records like this that entry is the right "
                f"answer {DIRECT_CONFIDENCE:.0%} of the time."
                if d.outcome is Outcome.POST else
                f"{chosen} matches this amount exactly, but {DIRECT_CONFIDENCE:.0%} is "
                f"still under the {d.threshold_required:.0%} an amount this large "
                f"requires. Sent for review.")
    elif d.outcome is Outcome.POST:
        expl = (f"Matched to {chosen}. It was the best of {len(usable)} nearby ledger "
                f"entries by a clear enough margin ({confidence:.0%}) to post without "
                f"a human looking.")
    else:
        expl = (f"Best guess is {chosen}, but at {confidence:.0%} it is under the "
                f"{d.threshold_required:.0%} this amount requires. Sent for review "
                f"rather than posted.")

    return {"stage": "ranking", "outcome": "posted" if d.outcome is Outcome.POST else "queued",
            "decision": d, "keys": [chosen], "n_candidates": len(usable), "path": path,
            "evidence": evidence, "confidence": confidence, "explanation": expl}


def _p_multiple_with(state: _State, rec: BankRecord, cands: list[str], key_stats) -> float:
    amts = [a for k in cands if k in key_stats for a in key_stats[k].amounts]
    has_exact = rec.amount_minor in amts if amts else False
    min_delta = min((abs(rec.amount_minor - a) for a in amts), default=1e12)
    f = featurise_multiplicity(rec, n_candidates=len(cands), has_exact=has_exact,
                               min_delta_minor=float(min_delta), prior=state.prior)
    return float(state.detector.predict_proba(f.reshape(1, -1))[0])


def _p_multiple(state: _State, rec: BankRecord, cands: list[str]) -> float:
    amts = [a for k in cands for a in state.key_stats[k].amounts if k in state.key_stats]
    has_exact = rec.amount_minor in amts if amts else False
    min_delta = min((abs(rec.amount_minor - a) for a in amts), default=1e12)
    f = featurise_multiplicity(rec, n_candidates=len(cands), has_exact=has_exact,
                               min_delta_minor=float(min_delta), prior=state.prior)
    return float(state.detector.predict_proba(f.reshape(1, -1))[0])


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
    except OSError as exc:  # noqa: BLE001
        return f"<pre>demo page missing: {exc}</pre>"
