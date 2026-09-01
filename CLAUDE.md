# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Bank-to-ledger reconciliation. Money arrives in a bank account; the books say
what it was for; this matches them — and takes on the case a real bank's rules
engine left entirely manual: one payment covering several ledger entries.

## Commands

```bash
uv run pytest -q -p no:warnings              # full suite, ~24s
uv run pytest tests/test_solver.py -q        # one file
uv run pytest -q -k "ambiguous"              # one test by name
uv run ruff check src tests scripts          # baseline is zero errors
uv run ruff check src tests scripts --fix

ARTIFACTS_DIR=$PWD/artifacts uv run uvicorn allocation_agent.api:create_app \
  --factory --port 8077                      # serve locally
```

Regenerating things (each reads `data/`, writes `artifacts/`):

```bash
uv run python scripts/export_artifacts.py    # -> demo.json, models.pkl, meta.json
uv run python scripts/export_reconriver.py   # -> reconriver.json
uv run python scripts/export_overview.py     # -> overview.json (the hero figures)
uv run python scripts/export_training_evidence.py  # -> training.json (the Trust screen)
uv run python scripts/train_ranker.py        # retrains; the leakage gate runs here
uv run python scripts/eval_solver.py         # regenerates the README solver table
uv run python scripts/run_learning.py        # the learning experiment + its controls
uv run python scripts/model_diagnostics.py   # train/val/test gap, F1, recall, confusion
uv run python scripts/calibrate_ranker.py    # refits the calibrator into models.pkl
```

Live at https://allocation-agent.onrender.com — free tier, ~45 rec/sec, sleeps
after ~15 min idle and takes ~50s to wake. Docker builds have taken 8+ minutes,
so do not push shortly before a demo.

## Architecture

**One matching path, three callers.** `match_one()` in `match/engine.py` is the
whole per-record pipeline. `/api/run`, `/api/reconcile` and `pipeline.run_batch`
all call it. Keep it that way — this rule was written while `pipeline.py` still
carried its own copy, and that copy silently fell behind on calibration, the
single-candidate case, the grouping override, and a record that could leave the
audit trail. A reviewer found eight defects in it; all eight were fixes that
existed in the API and had never been carried across.

Without a ranker, `run_batch` queues everything rather than falling back to
rules. The old fallback returned 0.90 for an exact amount and 0.55 otherwise —
a different scale from the calibrated model, handed to the same gate.

The four stages, in order, each able to end the record:

1. **Narrow** (`match/blocker.py`) — union of two hash lookups, `(account,
   amount)` and `(account, day ± slack)`. 98.9% recall, ~44 candidates from
   103k. No candidates → `no_candidate`.
2. **Direct** — two separate things, and only the first is common. Whenever
   exactly one candidate's amount equals the record's, **stage 3 is skipped**:
   on that subpopulation the multiplicity detector is right 12.2% of the time
   against 96.3% overall, so it does not get to overrule an exact amount. The
   record still goes on to be *ranked*. Only when that lone exact candidate is
   also the **only** scorable candidate — no runner-up, so no margin exists —
   is it taken at `DIRECT_CONFIDENCE` (0.9898). Blocking never returns fewer
   than four candidates on BenchRec, so that branch fires 6 times in 74,796
   records there; it is the small-uploaded-file case.
3. **Group check** (`match/multiplicity.py`) — a separate GBDT asking "is this
   one payment covering several entries?" Fires → `suspected_grouped`, no
   single match is claimed.
4. **Rank** (`match/ranker.py`) — LightGBM LambdaRank over 12 features.
   Confidence is an **isotonic calibrator** mapping the first-to-second margin
   to the measured frequency of being right — `sigmoid(margin)` claimed 74.8%
   where the truth was 21.5%, and the gate compares against 0.85. It lives in
   `models.pkl` under `bundle["calibrator"]`;
   with no runner-up there is no margin, so confidence is `None` and the
   record is queued rather than given a fabricated number.

Then `decide/gate.py` compares confidence to a bar that **rises with the
amount**, and every decision is appended to `report/audit.py`.

**The solver is a separate problem** (`match/solver.py`), reached through
`/api/settlements`, not through `_match_one`. Given a credit and ~100 candidate
payments it recovers the subset — cardinality-layered bitset DP, smallest
subset wins, and a second subset of the same size makes it `AMBIGUOUS` rather
than picking. Ties and oversized pools are refused, not guessed.

**The hero renders before any click.** `/api/overview` serves
`artifacts/overview.json`, precomputed over the whole held-out set by
`scripts/export_overview.py`. It is not computed at startup: 4,000 records take
~90s on the deployed free instance and the health check would fail first. It
must agree with a live run — a test pins that — because a cached figure that
drifts from what the buttons produce is worse than no figure.

**Aging is conditional.** `_aging()` refuses below a 60-day span and says why.
Measured on the held-out set: it covers 27 days and the auto-post rate across
weekly buckets is 74.9 / 80.2 / 81.3 / 80.0 — flat. Aging counts how long an
item has sat unresolved in a *running* book; a one-month snapshot resolved in a
single batch has nothing to age. Uploaded ledgers spanning months do.

**Report value, not only counts.** Reconciliation is worked by amount at risk,
so every figure has a money form and the queue is ordered biggest-first. Totals
are summed over *all* exceptions, never over the 100 the API returns. BenchRec
amounts are obfuscated units and the page says so rather than implying a
currency.

**`artifacts/` mixes two kinds of file.** `demo.json`, `models.pkl`,
`overview.json` and `meta.json` are build inputs, committed, produced by the scripts above and
loaded once at startup into `_State`. `runs.db` is runtime state — the audit
log — gitignored and dockerignored, so on the free tier it is **ephemeral**:
every deploy starts empty. `AUDIT_DB` overrides the path for a persistent
volume; `/api/health` reports `audit_persistent`. `ARTIFACTS_DIR` overrides the location;
`_State.load()` never raises, it records the failure and serves a 503.

**`models.pkl` couples three things.** It unpickles
`allocation_agent.match.ranker.Ranker` and the two multiplicity classes, so
those modules must stay importable even though `api.py` never imports them by
name — a static import graph will call them dead. `FEATURE_NAMES` in
`match/features.py` is an ordering contract with the pickled model: change the
order and the model silently scores the wrong columns.

**The page is a real dependency.** `static/index.html` is served by `_page()`
and read from disk per request. Tests assert it can reach every endpoint, reads
only fields `/api/meta` actually returns, and shows no internal enum name. The
server owns every number the page displays (`FREE_TIER_RECORDS_PER_SECOND`,
`MIN_FOR_RATES`, `BLOCKING`) and ships them through `/api/meta`.

**`/api/connect` is real.** It pulls a merchant's settlements from Razorpay in
test mode and runs the solver on them with the settlement id withheld. The
transport is injectable (`_rzp_fetch`) so every test runs without a key or a
network. A live key is refused; the secret is never stored, logged or returned.

**Partly on a live path, and the line matters.** `/api/correct` calls
`learn/router.py` to attribute a correction and `learn/casebase.py` to retain
it, and `/api/cases` reports what is held — so those two are live. What is
*not*: `casebase.retrieve()` is called by nothing outside its own tests, so no
decision consults a precedent, and nothing retrains. `learn/simulate.py` and
the retraining experiment are `scripts/run_learning.py` only. `pipeline.py` is
used only by `scripts/run_batch.py` and its test. Do not describe retraining or
precedent-driven matching as product features; `scripts/audit_architecture.py`
prints the current answer.

## The rule everything else follows from

**A claim that nothing checks is not a claim.** Every defect found here was
something asserted and unenforced:

| claimed | what actually held |
|---|---|
| `max=2000` on an input | advisory; the handler read `.value` anyway |
| a badge reading "Found the group" | derived from `status`, which is `solved` for wrong answers too |
| "rates withheld below 20" | guarded record count; precision is a rate over *answers* |
| "the narrator writes explanations" | constructed in `create_app()`, never called |
| "we checked for leakage" | a module with tests that nothing ran |
| a cap of 4 payments | admitted exactly one answer, and it was wrong |

When you write a rule, write the test that fails without it. When you state a
number, have code produce it.

## Testing

Make the failing test fail *for the right reason*. Several here passed while
testing nothing: `assert "not" in verdict` matches *Nothing* and *Notable*;
`pytest.raises(match="7.50")` treats `.` as a wildcard; a duplicate test
definition shadowed the first so it never ran. `tests/test_suite_integrity.py`
now catches duplicate names and assertion-free bodies, and ruff's `F811` and
`RUF043` catch the other two. Assert on exact strings or a field, never a
substring a dozen other strings contain.

## Numbers and evaluation

**One owner per number, and code produces it.** Four throughput figures were
live at once (40, 45, 495, 524) because each was typed where it was needed.
Constants live in `api.py` and reach the page through `/api/meta`. Never
hand-carry a measurement into prose — regenerate the table instead; doing that
caught a figure written as 10.7% that was 9.3%.

**Report coverage and precision separately.** One aggregate hid that 51% of
"solved" settlements were the wrong set.

Splits are temporal and group-respecting; the test set is frozen. Demo samples
are systematic — every Nth by position, never by outcome, because ReconRiver
front-loads its hard cases and an exporter that filtered to easy instances once
produced a fake 100%. `assert_no_leakage` runs inside `train_ranker.py` and
raises. No accuracy figure is reported for uploaded files: there is no answer
key, so any number would be invented.

## Money

Integer minor units everywhere; `BankRecord` refuses floats. Parsers refuse
more than two decimals rather than rounding — every other error announces
itself, that one balances and lies. The decimal mark travels with the
delimiter: a semicolon file uses a comma decimal, and reading `1250,00` as
comma-thousands makes it 125,000.00.

## When something fails

Prefer refusing to guessing. `INFEASIBLE`, `AMBIGUOUS` and "sent to a person"
are real answers; a wrong auto-post balances the books against the wrong
invoices, which is worse than an absent one. Explain in the words of someone
who has read only the problem statement — `suspected_grouped` is a good
variable name and a terrible label.

Before claiming anything is done: run the suite, then check the thing you
changed actually behaves differently — grepping an attribute is not exercising
a behaviour. Re-run the *whole* battery after a fix, not the cases you touched;
fixing three CSV defects introduced a 500 on empty files, caught only by the
full re-run.
