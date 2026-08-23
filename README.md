# Allocation Agent

**Money lands in a bank account. Somewhere in the books there is a record of what it was for. Connect them — and when one payment covers *several* records, notice that too.**

Built for the Razorpay Buildathon, Track 04 — AI Finance Controller.

---

## The finding this is built on

The data is [BenchRec](https://www.kaggle.com/datasets/benchmarkteam/benchrec-real-world-cash-reconciliation-dataset): 223,937 rows covering 172,023 reconciliations of obfuscated ledger and bank records, released from a **Tier-1 financial institution's production system**. The labels are decisions real analysts made about real money.

Split those reconciliations by shape and one number stands out:

| Match shape | Count | Resolved automatically |
|---|---|---|
| one-to-one | 153,557 | **94%** |
| many-to-one | 11,692 | 34% |
| one-to-many | 3,911 | **0%** |
| many-to-many | 2,707 | **0%** |

Their rules engine handled almost every simple match and **none** of the grouped ones. Not few — none. All 6,618 went to a person, and the data says so directly: every one carries `matchRule == MANUAL`.

That is the gap this fills.

---

## Results

Held-out temporal split, test frozen, models trained on earlier records only.

### Matching

| | |
|---|---|
| blocking recall | **98.94%** — the ceiling on everything downstream |
| top-1 accuracy | **93.53%** |
| trivial baseline | 90.63% (exact amount, tiebreak nearest date) |

### End-to-end, 37,398 records

| | |
|---|---|
| straight-through rate | 79.3% |
| **precision of auto-posted matches** | **99.2%** (29,426 / 29,649) |
| grouped records routed to review | 87.3% |
| throughput | 524 rec/sec on a laptop (8-core M-series); **~45/sec on free-tier hosting**, which is a fraction of a CPU |
| **LLM calls on the matching path** | **0** |
| records unaccounted for | **0** |

Review volume falls from 37,398 records to 7,749 — a **79.3% reduction in what a human must look at.**

### The learning loop

Cold start on 3,000 records, then learning from what a reviewer says. Ten batches of 4,000, all controls run.

| arm | autonomy | precision | **wrong auto-posts** |
|---|---|---|---|
| no learning | 82.1% | 97.3% | 2.22% |
| corrections only | 79.0% | 99.4% | 0.47% |
| **+ spot-check 25%** | **81.4%** | **99.2%** | **0.65%  (−71%)** |
| placebo (control) | 39.9% | 73.7% | 10.49% |

**The loop cuts wrong auto-posts by 71% at a cost of 0.7 pp of straight-through rate.** On a 4,000-record batch that is 89 wrong posts falling to 26.

It does not make the agent post *more* — it makes it post *better*, which is the right direction when a wrong match writes a false claim into the ledger and a missed one costs ten minutes of review.

Getting here required discovering that autonomy alone is a gameable metric: feeding the loop deliberately wrong answers *improved* it, because posting is gated on score margin and easy examples widen margins regardless of correctness. That is in [`BUILD_JOURNAL.md`](BUILD_JOURNAL.md).

### Multiplicity detection — the 11.3% nobody automates

| flag top | precision | recall |
|---|---|---|
| **5%** | **96.3%** | 47.3% |
| 10% | 77.4% | 76.1% |
| 15% | 62.1% | 91.6% |

PR-AUC 87.2% against a 10.2% positive rate. Beats the obvious rule (no exact-amount match) on F1: 76.7% vs 72.2%.

---

## What is *not* an LLM, and why

| Stage | Kind | Reason |
|---|---|---|
| blocking | rules | a hash lookup; a model would be slower and worse |
| ranking | gradient-boosted trees | 169,168 labelled examples exist. That is supervised learning. |
| multiplicity | gradient-boosted trees | same |
| gate | rules | anything deciding where money goes must be reproducible |
| residual diagnosis | **arithmetic** | each cause predicts a residual; rank by fit |
| column mapping | **LLM** | no deterministic parser generalises across formats |
| narration | **LLM** | language is where the ambiguity is |

**The model ranks. The engine decides. The person commits.**

**And it declines to guess.** Where two ledger keys carry identical amounts in
the same account and window, no feature can separate them: the model scores them
equally, the margin is zero, confidence is exactly 50%, and the record goes to a
human. Breaking that tie arbitrarily would post a wrong match with false
confidence on half of them.

Two constraints hold that in place:

- **The narrator may not introduce a number.** Every figure in generated text must appear in the input payload, checked after generation. There is a test that feeds it a lying backend and asserts the invented figure never reaches output.
- **The audit log is append-only, enforced by the database.** SQLite triggers reject `UPDATE` and `DELETE` on decisions. A reviewer overturning a decision writes a *new* row; both stay visible.

---

## What broke

The full account is in [`BUILD_JOURNAL.md`](BUILD_JOURNAL.md), written as it happened. Three that mattered:

**I nearly recorded a working learning loop as a failure.** It appeared to cost 3.15 pp of autonomy, reproducing under three shuffled orderings. Then the placebo control improved by 3.72 pp — feeding deliberately wrong answers made autonomy go *up*. That exposed the metric: posting is gated on score margin, easy examples widen margins, so autonomy measures *decisiveness*, not correctness. Adding precision to every arm inverted the conclusion — the loop trades 0.7 pp of straight-through for **71% fewer wrong auto-posts**.

**I nearly built blocking on a false assumption.** First measurement said amount was useless as a signal — 29% exact match, median residual 6.2 million. Wrong: I was comparing one bank record against the sum of *every* ledger row sharing a key, and a key can span 624 rows. Measured properly, **90.9% of records match some individual row to the paisa.**

**Graceful degradation got demonstrated by accident.** The LLM model I first configured does not exist — 404. I found out not from an error but because the narrator quietly produced complete, correct output and I checked why `source` said `template`. The batch finished; every exception got an accurate cause; nothing failed.

---

## Limitations

Stated plainly, because the alternative is being asked about them.

**The identity layer cannot be evaluated on this data.** `orderingPartyInfo` and `receivingPartyInfo` are **0% populated** — there are no counterparty names at all. So the five-layer resolver and transitive alias clustering in the design have nothing to run on here.

**Direction is not a usable constraint here.** `debitOrCredit` has one distinct value (`NONE`) across all 211,744 ledger rows. The design treats direction as a hard constraint, following a published post-mortem; there is nothing to constrain.

**1.06% of records are lost at blocking** and cannot be recovered downstream. Widening the window to ±14 days recovers 0.4% of that for double the candidates.

**223 auto-posted matches are wrong.** That is what 99.2% precision leaves. Each closes a real exception and writes a false claim into the ledger.

**1,535 single-key records were wrongly sent for review** — 4.1% of the batch doing unnecessary work, the false-positive cost of catching 87.3% of the grouped ones.

**Straight-through is below vendor claims.** HighRadius publishes 95–98% auto-match; this posts 79.3%. Some of the gap is definitional — routing a grouped record to a human counts against us and may not count against them — but the comparison is written the way that does not assume our favour.

**Throughput is 1.26× the published commercial figure, not more** — and that is on a laptop. On the free-tier host the live demo runs at ~45 rec/sec, roughly 11× slower, because the instance is a fraction of a CPU. Blocking alone does 6,720/sec; the pipeline is still a per-record Python loop. A throughput number without the machine attached is not a number.

**The grouped-settlement numbers come from synthetic data.** BenchRec has no subset-sum structure — measured, not assumed: 0 of 4,000 grouped records form a closed batch, because `matchId` is a batch identifier rather than an accounting unit. So the solver is evaluated on [ReconRiver](https://huggingface.co/datasets/heybadrinath/reconriver-synthetic-reconciliation), which does have it. Synthetic amounts are drawn independently, which likely makes coincidental subsets *rarer* than in production — so 96.6% precision should be read as an upper bound, not a forecast.

**[APEX-Accounting](https://huggingface.co/datasets/sadcasticme/apex-accounting) was not used.** It is 10 developer tasks scored by rubric, not a labelled reconciliation set — a good benchmark for a coding agent and the wrong shape for measuring match precision against ground truth. Named here because it is the obvious dataset to ask about.

**The free LLM's prose is worse than the templates it replaces.** The architecture is sound; the output is not yet an improvement. Templates are the default and the model is the optional upgrade — the reverse of what the design assumed.

**The learning loop is an offline experiment, not a live feature.** `scripts/run_learning.py` produces the −71% result above with all its controls, and it is real. But no API path touches it: a reviewer correcting a decision in the deployed demo does not retrain anything. The architecture diagram draws learning as a layer of the running system; today it is a script.

**The case base is built and tested but wired to nothing.** Twelve tests cover retrieval, near-duplicate collapse, size capping and retirement. Nothing calls it — not even the learning experiment that produced the number above. It is retained rather than deleted because the design for it was requested deliberately, but it is not part of any measured result.

---

## Live demo

Deployed on Render's free tier. The first request after a period of inactivity
takes ~50 seconds to wake the container; subsequent ones are immediate. The
instance is a fraction of a CPU, so throughput there is ~45 rec/sec against 495
on a laptop — same code, different hardware.

The demo runs on **held-out records the models never saw during training**, so
the precision it reports is measured against ground truth rather than asserted.

## Running it

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest                              # 240 tests

uv run python scripts/measure_blocking.py  # recall/size tradeoff
uv run python scripts/train_ranker.py      # top-1 vs baselines
uv run python scripts/train_multiplicity.py
uv run python scripts/run_batch.py         # end-to-end + audit trail
uv run python scripts/run_learning.py      # learning loop + all controls
uv run python scripts/eval_solver.py       # group solver on ReconRiver
```

### Grouped settlements

One bank credit, many payments. The solver gets the amount and a pool of ~100
candidate payments and must recover which subset produced it -- **the batch
identifier is never passed to it**. Evaluated on ReconRiver, whose settlement
structure BenchRec does not have (measured; see the journal).

| variant | coverage | precision | balanced-but-wrong | ties refused |
|---|---|---|---|---|
| reachability DP, first subset | 38.0% | 39.0% | 59.3% | -- |
| smallest subset | 83.3% | 93.3% | 6.0% | -- |
| smallest, refuse ties | 75.3% | 96.6% | 2.7% | 11.3% |
| + cap the answer at 3 payments (shipped) | **75.3%** | **98.3%** | **1.3%** | 9.3% |

`coverage` = credits whose recorded batch it recovered. `precision` = of the
answers it gave, how many were the recorded batch. Reporting either alone is how
a solver looks good and is not: the first row answers plenty and is wrong more
often than right.

The shipped variant gives up 8pp of coverage to cut wrong answers by half. A
credit it refuses goes to a reviewer; a credit it gets wrong balances the books
against the wrong invoices, which is the more expensive of the two.

Split by how many payments the batch really had, that aggregate turns out to
cover two different problems:

| true batch | credits | recovered | wrong | ties refused | unresolved | unreachable |
|---|---|---|---|---|---|---|
| 1 payment | 30 | 7 | **4** | 3 | 16 | 21 |
| 2 payments | 62 | **62** | 0 | 0 | 0 | 0 |
| 3 payments | 58 | 44 | 0 | 14 | 0 | 0 |

**On genuine multi-payment batches: 106 of 120 recovered, zero wrong.** Every
wrong answer is a single-payment credit — an exact match that should never have
reached a subset-sum solver, and for 21 of the 30 the true payment was not in
the candidate pool at all. `unreachable` is reported separately because that is
a blocking limit, not a solver one.

```

uv run uvicorn allocation_agent.api:create_app --factory --reload   # the service
```

Everything runs on CPU. No GPU, no external services required — the LLM layer is optional and falls back to templates without a key.

```bash
cp .env.example .env    # optional: add an OpenRouter key for narration
```

---

## Design

[`docs/DESIGN.md`](docs/DESIGN.md) — HLD and LLD, including the parts not yet built and the decisions still open.

Where the code and the design disagree, the journal explains which measurement changed my mind.
