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
from allocation_agent.match.features import build_key_stats, featurise
from allocation_agent.match.multiplicity import featurise_multiplicity
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


class RunRequest(BaseModel):
    limit: int = Field(default=500, gt=0, le=10**9)
    review_all: bool = False
    mult_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


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
        self.ready = True


def create_app() -> FastAPI:
    app = FastAPI(title="Allocation Agent", version="0.1.0")
    state = _State()
    state.load()
    audit = AuditLog(ARTIFACTS / "runs.db")
    narrator = Narrator()  # templates by default; no key required, no cost

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _PAGE

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
            cands = sorted(block(rec, state.index, bcfg))
            truth_key, truly_mult = state.truth.get(rec.record_id, ("", False))

            if not cands:
                d = decide(confidence=None, amount_minor=rec.amount_minor, config=gate)
                audit.record(rec.record_id, d, keys=[], n_candidates=0, path="blocked")
                summary["no_candidate"] += 1
                exceptions.append(_exception(rec, "no_candidate",
                                             "Nothing in the ledger is close enough to consider.", 0))
                continue

            p_mult = _p_multiple(state, rec, cands)
            if p_mult >= req.mult_threshold:
                d = decide(confidence=None, amount_minor=rec.amount_minor, config=gate)
                audit.record(rec.record_id, d, keys=[], n_candidates=len(cands),
                             path="multiplicity", evidence={"p_multiple": round(p_mult, 4)})
                summary["suspected_grouped"] += 1
                exceptions.append(_exception(
                    rec, "suspected_grouped",
                    f"Looks like one payment covering several ledger entries "
                    f"(confidence {p_mult:.0%}). Routed for review.", len(cands)))
                continue

            X = np.vstack([featurise(rec, state.key_stats[k], n_candidates=len(cands))
                           for k in cands if k in state.key_stats])
            scores = state.ranker.score(X)
            order = np.argsort(-scores)
            chosen = cands[int(order[0])]
            margin = float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else 1.0
            confidence = float(1.0 / (1.0 + np.exp(-margin)))

            d = decide(confidence=confidence, amount_minor=rec.amount_minor, config=gate)
            audit.record(rec.record_id, d, keys=[chosen], n_candidates=len(cands),
                         path="ranked", evidence={"margin": round(margin, 4)})

            if d.outcome is Outcome.POST:
                summary["posted"] += 1
                posted_correct += (not truly_mult and chosen == truth_key)
            else:
                summary["queued"] += 1
                exceptions.append(_exception(
                    rec, "below_threshold",
                    f"Best candidate scored {confidence:.0%}, below the "
                    f"{d.threshold_required:.0%} required for this amount.", len(cands)))

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

    @app.get("/api/run/{run_id}/audit")
    def audit_trail(run_id: str) -> dict:
        rows = audit.decisions(run_id=run_id)
        if not rows:
            raise HTTPException(404, f"no run {run_id}")
        return {"run_id": run_id, "decisions": rows}

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


_PAGE = """<!doctype html><meta charset=utf-8><title>Allocation Agent</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0d1117;--fg:#e6edf3;--dim:#8b949e;--line:#21262d;--card:#161b22;--ok:#3fb950;--warn:#d29922}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;padding:2rem 1.25rem}
.w{max-width:60rem;margin:0 auto}h1{font-size:1.3rem;margin:0 0 .3rem}
p.sub{color:var(--dim);margin:0 0 1.5rem}
button{background:#238636;color:#fff;border:0;padding:.6rem 1.1rem;border-radius:6px;
cursor:pointer;font:inherit}button:disabled{opacity:.5;cursor:default}
label{color:var(--dim);margin-right:1rem}input,select{background:var(--card);color:var(--fg);
border:1px solid var(--line);border-radius:6px;padding:.4rem .6rem;font:inherit}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));gap:1px;
background:var(--line);border:1px solid var(--line);border-radius:8px;overflow:hidden;margin:1.5rem 0}
.c{background:var(--card);padding:.9rem 1rem}.c .n{font-size:1.5rem;font-weight:600}
.c .l{color:var(--dim);font-size:.8rem}
table{width:100%;border-collapse:collapse;font-size:.85rem;margin-top:.5rem}
th{text-align:left;color:var(--dim);font-weight:500;border-bottom:1px solid var(--line);padding:.5rem .6rem .5rem 0}
td{padding:.5rem .6rem .5rem 0;border-bottom:1px solid var(--line);vertical-align:top}
.tag{font-size:.72rem;padding:.15rem .45rem;border-radius:4px;border:1px solid currentColor;white-space:nowrap}
.g{color:var(--ok)}.y{color:var(--warn)}
</style>
<div class=w>
<h1>Allocation Agent</h1>
<p class=sub>Matches bank records to ledger allocation keys. Demo runs on
<b>held-out records the models never saw in training</b>, so the precision below is
measured against ground truth, not asserted.</p>

<div>
  <label>records <input id=n type=number value=500 min=1 max=4000 style=width:6rem></label>
  <label><input id=all type=checkbox> review everything</label>
  <button id=go>Run</button>
</div>

<div id=out></div>
</div>
<script>
const $=s=>document.querySelector(s), fmt=n=>n.toLocaleString();
$('#go').onclick=async()=>{
  const b=$('#go'); b.disabled=true; b.textContent='running...';
  $('#out').innerHTML='';
  try{
    const r=await fetch('/api/run',{method:'POST',headers:{'content-type':'application/json'},
      body:JSON.stringify({limit:+$('#n').value, review_all:$('#all').checked})});
    if(!r.ok) throw new Error((await r.json()).detail||r.statusText);
    const d=await r.json(); render(d);
  }catch(e){ $('#out').innerHTML='<p class=y>'+e.message+'</p>'; }
  b.disabled=false; b.textContent='Run';
};
function render(d){
  const s=d.summary, pct=x=>(x*100).toFixed(1)+'%';
  $('#out').innerHTML=`
  <div class=grid>
    <div class=c><div class=n>${fmt(d.n_records)}</div><div class=l>records</div></div>
    <div class="c"><div class="n g">${pct(d.precision_of_posted)}</div><div class=l>precision of posted</div></div>
    <div class=c><div class=n>${pct(d.straight_through_rate)}</div><div class=l>straight-through</div></div>
    <div class=c><div class=n>${fmt(Math.round(d.records_per_second))}</div><div class=l>records/sec</div></div>
    <div class=c><div class=n>${d.llm_calls_on_matching_path}</div><div class=l>LLM calls</div></div>
  </div>
  <p class=sub>posted ${fmt(s.posted)} &middot; queued ${fmt(s.queued)} &middot;
  suspected grouped ${fmt(s.suspected_grouped)} &middot; no candidate ${fmt(s.no_candidate)}
  &middot; <b>${fmt(s.posted+s.queued+s.suspected_grouped+s.no_candidate)} of ${fmt(d.n_records)} accounted for</b>
  &middot; run <code>${d.run_id}</code></p>
  <h3 style="font-size:1rem;margin:1.5rem 0 0">Exceptions (${fmt(d.n_exceptions)})</h3>
  <table><tr><th>record</th><th>amount</th><th>reason</th><th>explanation</th></tr>
  ${d.exceptions.slice(0,40).map(e=>`<tr><td>${e.record_id}</td>
   <td style=text-align:right>${e.amount.toLocaleString(undefined,{minimumFractionDigits:2})}</td>
   <td><span class="tag ${e.reason==='suspected_grouped'?'y':''}">${e.reason}</span></td>
   <td>${e.explanation}</td></tr>`).join('')}
  </table>`;
}
</script>"""
