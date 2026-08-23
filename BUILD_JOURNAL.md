# Build journal

What broke, what I got wrong, and what I did about it. Written as it happens.

---

## Editable install before the package existed

**What broke:** `uv pip install -e .` ran cleanly, then `import allocation_agent`
failed with `ModuleNotFoundError`. The install had reported success.

**Root cause:** hatchling resolves `packages = ["src/allocation_agent"]` at install
time. The directory was created *after* the install, so hatchling recorded an
empty package and exited 0. A successful exit code meant nothing here.

**Fix:** create the package tree and `__init__.py` files first, then install.

**Kept because:** a zero exit code is not evidence that a thing worked. The same
mistake in the reconciliation pipeline would be a run that "succeeds" having
matched nothing.

---

## Leakage guard: case sensitivity

**Caught before it shipped.** The first version compared column names exactly.
Column casing varies between sources — a bank export may well ship `MATCHRULE`.
An exact-match guard would have passed it and every downstream metric would have
been fiction.

**Fix:** case-insensitive comparison, and the error names *every* offending
column rather than the first.

---

## Derived-leak check found nothing — recorded as a negative result

Ran the mutual-information check over the 26 remaining columns on a 20,000-row
sample after dropping the four known outcome columns. Nothing above NMI 0.95.

Worth writing down: it confirms the four known columns were the only leaks,
rather than leaving that as an assumption. A negative result checked is worth
more than a positive result assumed.

---

## Nearly built blocking on a false assumption about amounts

**What happened:** The design assumed a bank record's amount would sit close to
its allocation key's total, so blocking would use an amount *band*. First
measurement on real data said otherwise: only 29.2% of true pairs matched the
key total exactly, and the median residual was **6.2 million**. On the face of
it, amount looked useless as a blocking signal.

**Root cause:** I was comparing one bank record against the sum of *every*
ledger row carrying that key. A key can span up to 624 rows and be claimed by up
to 624 bank records, so of course one record does not equal the whole total.
The comparison was wrong, not the data.

**What is actually true, measured:**

- 87% of keys are claimed by exactly one bank record
- 83.5% of keys are one ledger row and one bank record, and those balance 100%
  exactly
- 90.2% of keys balance at *key* level: sum(ledger) == sum(bank)
- **90.9% of bank records match some individual ledger row to the paisa**

So amount is not a weak signal. It is very nearly a hard key -- just at row
level, not key level.

**Fix:** index every ledger *row* separately under `(account, amount_minor)`, so
a key is reachable from any of its constituent amounts. Blocking on account plus
exact amount alone gets 91.0% recall with a mean of 3.4 candidates.

**Kept because:** the first number was not a data problem, it was a wrong join.
Had I trusted it, blocking would have used a wide amount band, dragged in
thousands of candidates, and buried the fact that the amount is nearly exact.
Measure the join before concluding anything about the column.

---

## Blocking recall, measured (build step 2)

Recall is a ceiling on every downstream number, so this had to be established
before the ranker existed. 169,168 labelled single-key records:

| strategy | recall | median cands | mean cands |
|---|---|---|---|
| account only | 100.00% | 371 | 2804.4 |
| amount only | 90.95% | 1 | 3.4 |
| amount + date +/-3d | 97.64% | 18 | 22.2 |
| **amount + date +/-7d** | **98.94%** | **38** | **44.5** |
| amount + date +/-14d | 99.34% | 71 | 82.1 |

Chose +/-7 days: a 63x reduction against account-only for 1.06% of recall.
190,717 records blocked in **1.1 seconds**; index build 0.2s.

The 1.06% that is lost is a hard ceiling and belongs in the exception taxonomy,
not hidden. Widening to +/-14d recovers 0.4% of it for double the candidates --
a tradeoff to revisit once the ranker's cost per candidate is known.

**Also worth noting:** parsing is 6.8s of the run, six times longer than the
matching it feeds. It is a row-wise Python loop. Not optimised yet because it is
a one-time cost per batch, but it is now the slowest stage.

---

## Several designed features do not exist in this data

Measured field population before writing the feature extractor:

| field | populated | consequence |
|---|---|---|
| `transactionReferences` | **0.0%** | no reference-overlap feature |
| `orderingPartyInfo` | **0.0%** | **no counterparty names at all** |
| `receivingPartyInfo` | **0.0%** | identity resolution is untestable here |
| `currencyCode` | 100% | one distinct value: no signal |
| `debitOrCredit` | 100% | one distinct value (`NONE`): no signal |
| `transactionAttributes` | 100% | different vocabularies per side |

Two design decisions die here. **Direction as a hard constraint** -- lifted from a
published post-mortem where it prevented a whole class of false match -- has
nothing to constrain, because every row says `NONE`. And the **five-layer identity
resolver** has no names to resolve.

Neither is wrong as a design. Both are simply unevaluable on this dataset, and
that belongs in the limitations rather than in a quiet omission. Recording it
here so the README says so too.

---

## TDD caught a unit confusion in the feature extractor

`amount_delta_abs` is reported in major units; my test asserted minor. The test
failed, the code was right. Added a second test pinning the unit explicitly, so
the next person does not have to re-derive it from the divisor.

Small, but it is the kind of thing that survives to production as a feature
scaled 100x differently from its neighbours.

---

## First real ranker result

Temporal split (70/10/20), group-respecting, test frozen.

| | top-1 |
|---|---|
| trivial baseline (exact amount, tiebreak nearest date) | 90.63% |
| **ranker** | **93.50%** |
| blocking ceiling | 98.94% |

+2.87 pp over baseline; 5.44 pp of headroom remains.

1.77M training pairs, 23s to train, 33,274 test records scored in 10.4s.

**But the feature importances say the formulation is wrong.** The top two are
`key_total_major` (23.3%) and `n_candidates` (21.5%), and `n_candidates` is
*constant across every candidate of a given record*. It cannot discriminate
within a record at all -- it can only act through interactions. A feature that
cannot separate candidates taking a fifth of the model's attention means the
binary-classification framing is leaking effort.

The problem is a **ranking** problem: choose the best of N, not classify each pair
independently. Next step is a LambdaRank objective with the record as the group.

---

## Ranking objective beats binary classification

Same features, same split, same data:

| objective | top-1 | train |
|---|---|---|
| binary classification | 92.73% | 19.6s |
| **LambdaRank** | **93.53%** | 22.1s |
| trivial baseline | 90.63% | — |
| blocking ceiling | 98.94% | — |

+0.80 pp for the formulation change alone, and +2.90 pp over baseline.

The task is choose-best-of-N within a record, not classify-each-pair
independently. Binary treats every candidate as its own yes/no question and
spends capacity learning an absolute decision boundary that is never used --
only the ordering within a record matters.

**But I was half wrong about `n_candidates`.** It is still 23.2% of splits under
LambdaRank, and I had written that a group-constant feature "cannot discriminate
within a record". That is true of its *direct* effect and irrelevant to its real
use: it gates the other features. The tree learns *"when this record has many
candidates, weight the amount features differently than when it has few."* That
is a legitimate interaction, not wasted capacity. Correcting the earlier entry
rather than deleting it.

## Calibration can silently change which candidate wins

The binary run scored 93.50% before the LambdaRank change and 92.73% after, on
identical code paths. The difference is the isotonic calibrator.

Isotonic regression is monotonic, so it cannot reorder candidates -- but it is a
*step* function, and it maps ranges of raw scores onto the same calibrated value.
Distinct raw scores become tied, and `argmax` then picks whichever came first in
the candidate set, which is arbitrary.

**Rule:** rank on raw scores, calibrate the winner afterwards. Calibration exists
to make the gate's threshold meaningful, and the gate only ever sees one score
per record. Applying it before selection buys nothing and quietly costs accuracy.

---

## Multiplicity detection — the 11.3% nobody automates

21,549 of 190,717 records span several allocation keys. Every one carries
`matchRule == MANUAL`: the institution's own engine automated **none** of them.

**The design's prediction about the strongest signal was wrong.** It expected
amount size (grouped payments being larger). Measured:

| signal | MULT | single |
|---|---|---|
| **has an exact-amount candidate** | **10.9%** | **91.9%** |
| blocked candidates (median) | 62 | 22 |
| amount (median, minor) | 1,124,239 | 664,703 |
| round to Rs 1,000 | 0.0% | 0.0% |

Amount is a 1.7x separator. Exact-amount availability is near-categorical, and
it follows from what a grouped payment *is*: a sum of several rows matches no
single row. Round numbers were predicted to help and contribute nothing at all --
recorded rather than quietly dropped.

**Results, temporal split, test frozen:**

| | precision | recall | F1 |
|---|---|---|---|
| baseline: no exact-amount match | 58.8% | 93.6% | 72.2% |
| model @ 0.5 | 57.6% | 93.2% | 71.2% |
| **model @ 0.7** | **68.4%** | **87.3%** | **76.7%** |
| model @ 0.9 | 80.9% | 72.0% | 76.1% |

PR-AUC 87.2% against a 10.2% positive rate.

**At a fixed alert budget**, which is how a review queue actually works:

| flag top | precision | recall |
|---|---|---|
| 5% (1,869 records) | **96.3%** | 47.3% |
| 10% (3,739) | 77.4% | 76.1% |
| 15% (5,609) | 62.1% | 91.6% |

Note the model **loses to the baseline at threshold 0.5** (F1 71.2 vs 72.2). It
only wins from 0.7 upward. Reporting the losing threshold too: a model that needs
its operating point chosen carefully is a fact about the model, not a detail to
omit.

## Split-count importance is not predictive value

`has_exact_amount_candidate` is the sharpest separator in the data and does not
appear in the top five importances. That is not a contradiction -- LightGBM's
default importance counts *splits*, and a feature that resolves most of the
problem once at the root is used once. Features that carve up the residue are
used constantly and score higher.

Read gain, not split count, when asking what a model relies on. Left as-is with
this note rather than silently swapping the metric.

---

## First end-to-end run on the held-out split

37,398 records, temporal split, test frozen, models trained on train only.

```
posted              29,649   (79.3% straight-through)
queued               2,895
suspected grouped    4,854
no candidate             0
```

| | |
|---|---|
| straight-through rate | **79.3%** |
| **precision of auto-posted matches** | **99.2%** (29,426 / 29,649) |
| grouped records routed to review | 87.3% (3,319 / 3,804) |
| single records wrongly routed | **1,535** |
| throughput | 524 rec/sec |
| LLM calls on the matching path | **0** |
| records unaccounted for | **0** |

**Where this is weaker than it looks.**

*223 auto-posted matches are wrong.* At 99.2% precision that is the residue, and
it is not zero. Each one closes a real exception and writes a false claim into
the ledger. Raising the gate's base threshold trades straight-through rate for
precision; the curve has not been swept yet and should be, against cost rather
than accuracy.

*1,535 single-key records were wrongly sent to review* by the multiplicity
detector at threshold 0.7. That is 4.1% of the batch doing unnecessary human
work -- the false-positive cost of catching 87.3% of the grouped ones. The
tradeoff is explicit and tunable, and reporting only the 87.3% would hide half
of it.

*Straight-through is below what vendors claim.* HighRadius publishes 95-98%
auto-match. This run posts 79.3%. Some of that gap is real and some is
definitional -- routing a grouped record to a human counts against us here and
may not count against them -- but the honest comparison is the one that does not
assume their favour.

**Throughput: 524 rec/sec against a published commercial figure of 417/sec.**
Only 1.26x, and blocking alone runs at 6,720/sec. The pipeline is a per-record
Python loop doing featurisation and scoring one record at a time; vectorising it
would move this a lot. Recorded now so the number is honest rather than
flattering.

**Review volume.** Without the system all 37,398 records need a human. With it,
7,749 do -- a 79.3% reduction. That is the number a finance lead actually buys.

## Keyword-only arguments caught a wrong call site

`temporal_split(days, groups, 0.7, 0.1)` raised `TypeError` rather than silently
interpreting the fractions as something else. The signature makes everything
after `groups` keyword-only, deliberately, because `(0.7, 0.1)` and `(0.1, 0.7)`
are both plausible-looking and only one is right.

Cost: one failed run. Alternative: a split that looked fine and was 10/70.

---

## The learning loop made the system WORSE. Recording it before fixing it.

Three arms over the same 37,398 held-out records, cold start (3,000-record seed),
autonomy measured per batch of 4,000:

```
learning on          81.9  81.3  81.4  79.0  77.2  77.3  79.4  73.3  72.0  79.0   (-2.93 pp)
C-1 learning off     81.9  81.5  82.2  82.4  81.7  82.8  84.0  79.1  79.3  82.1   (+0.22 pp)
C-3 placebo          81.9  81.8  80.4  78.0  79.8  78.7  72.3  61.0  57.1  59.1  (-22.82 pp)
```

C-2, three shuffled orderings: **-4.96, -7.31, -5.57**. Consistently negative.

**Learning is 3.15 pp *worse* than not learning.** Not noise -- it reproduces
under every ordering.

**The placebo control passes**, and that is what makes the result trustworthy
rather than a bug: feeding random corrections collapses autonomy by 22.8 pp, so
the feedback path demonstrably works. It is working, and what it does is harmful.

### Why

Corrections are collected **only from records the gate refused**. Those are, by
construction, the ambiguous ones -- the cases where the top two candidates scored
close together. Refits therefore see a training set drifting steadily toward the
hardest examples in the distribution.

Confidence here is derived from the **margin** between the top two scores. Train
on progressively more ambiguous cases and the model learns to separate them less
sharply; margins compress; confidence falls; fewer records clear the gate. Lower
autonomy is not a symptom of a worse model -- it is the direct arithmetic
consequence of a less decisive one.

So the loop is a negative feedback cycle: refuse the hard ones, train on the hard
ones, become less decisive, refuse more.

### What this says about the design

The design said "corrections become labelled examples, upweighted, refit." That
is wrong on its own, and wrong in a way that looks right on paper. **A system
that only ever learns from its failures learns a biased view of the world.**

The production analogue is exact and worth stating: queued records get reviewed
so their truth becomes known; posted records are never reviewed, so theirs never
does. The bias is not an artefact of the simulation. It is what the deployment
would actually do.

**Next: sample a fraction of auto-posted records for review as well**, which is
what a finance function already does under the name spot-checking. If the
diagnosis is right, restoring the easy cases to the training set should restore
the curve. Testing that rather than assuming it.

---

## Graceful degradation demonstrated by accident

The model I first configured, `google/gemini-2.0-flash-exp:free`, returns
**404 -- no endpoints found**. It does not exist on OpenRouter.

I did not discover this from an error. I discovered it because the narrator
quietly produced complete, correct output via templates and I checked *why* the
`source` field said `template` instead of `llm`.

That is the intended behaviour working: the batch finished, every exception got
an accurate cause and a readable sentence, and nothing failed. The diagnosis is
arithmetic and needs no model at all -- BANK_CHARGE, ROUNDING and UNEXPLAINED
were all correctly identified with the language layer entirely unavailable.

Better demonstration of "the AI can fail, the system cannot" than the deliberate
chaos test would have been, because nobody staged it.

## Live LLM: it works, and the prose is poor

Switched to `nvidia/nemotron-3.5-lightning:free`.

```
1 call, 4 items, 34.9s
  [llm] BANK_CHARGE   BANK_CHARGE on amount 20000 with 1 line and residual 200.
  [llm] ROUNDING      ROUNDING affects amount 500000 with 4 lines and residual 3.
  [llm] UNEXPLAINED   UNEXPLAINED discrepancy on amount 1000 with 1 line and residual 123456.
second pass: 1 total call, 0.000s (cache hit)
```

**The architecture works.** One call for four items, every figure validated
against the payload, second pass served entirely from cache.

**The sentences are bad.** The model is essentially restating the payload fields
in a sentence-shaped way -- "BANK_CHARGE on amount 20000 with 1 line" is not what
a reviewer wants to read. A stronger model would write better prose; this one is
free and slow (34.9s for one call).

Recording rather than dressing it up: the constraint that makes the layer safe
(no figure may be introduced) is satisfied, and the output quality is currently
worse than the hand-written templates it falls back to. On this evidence the
template path is the better default, and the model is the optional upgrade --
which inverts what the design assumed.

---

## The placebo control caught that my headline metric is gameable

Ran the spot-check fix. The dose-response is exactly as the diagnosis predicted:

```
C-1 learning off     81.9 ... 82.1   (+0.22 pp)
corrections only     81.9 ... 79.0   (-2.93 pp)   -3.15 vs C-1
+ spot-check 10%     81.9 ... 80.3   (-1.57 pp)   -1.79 vs C-1
+ spot-check 25%     81.9 ... 81.4   (-0.50 pp)   -0.72 vs C-1
```

Monotonic. Restoring easy cases to the training distribution reduces the harm in
proportion to how many are restored. The mechanism was right.

**But the placebo arm improved: +3.72 pp.** Feeding deliberately wrong keys made
autonomy go *up*, and by more than any honest arm.

Two things are true and both matter.

### 1. The control itself is mis-specified

`placebo` corrupts the keys taken from *corrections*, but the spot-check branch
appends `correct_keys[0]` untouched. So the placebo arm receives garbage
corrections **plus 25% genuine easy examples** -- and still improves. My control
does not isolate what it claims to.

That is a bug and it is being fixed: placebo must corrupt everything fed back.

### 2. What the broken control accidentally proved is worse

Garbage corrections plus easy examples beat *everything*, including honest
corrections plus the same easy examples. Which means the corrections are not
contributing positively at all. The gain is coming entirely from the easy
examples -- and easy examples make the model **more decisive**, not more correct.

Autonomy here is `posted / total`, and posting is gated on the margin between the
top two scores. **A model trained on easy, well-separated examples produces wide
margins, high confidence, and posts more -- whether or not it is right.**

So the metric I have been reporting as "the learning curve" measures
decisiveness, not correctness. It is gameable, and the placebo gamed it.

### What this changes

Autonomy alone is not a valid success measure for this loop. Every arm must
report **precision of what it posts** alongside autonomy, and a rising autonomy
curve with falling precision is a regression, not progress.

This is precisely why the control exists. Had I reported the +25% spot-check arm
as "the fix, -0.72 pp and closing", the number would have been real and the
conclusion drawn from it wrong. An hour of control cost less than that would
have.

---

## The learning loop works. I was measuring the wrong thing.

Re-ran with the placebo corrected and precision tracked per batch. The picture
inverts completely.

```
                       autonomy          precision
C-1 learning off       81.9 -> 82.1      96.7 -> 97.3   (+0.60 pp)
corrections only       81.9 -> 79.0      96.7 -> 99.4   (+2.66 pp)
+ spot-check 10%       81.9 -> 80.3      96.7 -> 99.0   (+2.32 pp)
+ spot-check 25%       81.9 -> 81.4      96.7 -> 99.2   (+2.51 pp)
C-3 placebo            81.9 -> 39.9      96.7 -> 73.7   (-23.05 pp)
```

**The placebo now fails on both axes** -- autonomy -41.99 pp, precision -23.05 pp.
Nonsense hurts everything. The control is finally isolating what it claims to.

### What the loop actually does

It does not make the agent post more. **It makes the agent post better.** Every
learning arm gains 2.3-2.7 pp of precision against 0.6 pp for not learning at
all, and the autonomy it gives up is the cost of that caution.

Which is the correct direction for this domain. A wrong match closes a real
exception and writes a false claim into the ledger; a missed match costs a
reviewer ten minutes. Learning to be more careful is not a regression here, it is
the whole point -- and I nearly recorded it as a failure because autonomy was the
only number on the chart.

### Combining the two axes

| arm | correct auto-posts | **wrong auto-posts** |
|---|---|---|
| no learning | 79.88% | 2.22% |
| corrections only | 78.53% | **0.47%  (-78.6%)** |
| + spot-check 10% | 79.50% | 0.80%  (-63.8%) |
| **+ spot-check 25%** | **80.75%** | **0.65%  (-70.6%)** |
| placebo | 29.41% | 10.49%  (+373%) |

**Spot-check 25% dominates the no-learning arm on both counts**: slightly more
correct posts (80.75% vs 79.88%) and **71% fewer wrong ones**.

On a batch of 4,000 records that is **89 wrong auto-posts falling to 26**.

### The chain this took

1. Loop appears to make things worse: -3.15 pp autonomy, reproducing under three
   orderings
2. Diagnosed the mechanism: corrections come only from refused records, training
   drifts toward ambiguous cases, margins compress
3. Fix predicted and confirmed: spot-checking posted records produces a clean
   monotonic dose-response
4. Placebo caught that autonomy is gameable -- garbage improved it
5. Added precision; the result inverts and the loop is doing the right thing

Steps 4 and 5 are the ones that mattered. Without the control I would have
reported "the learning loop reduces autonomy by 0.72 pp and needs more work",
which is true, measured, and the wrong conclusion.

**Headline, stated correctly:** the learning loop cuts wrong auto-posts by 71%
at a cost of 0.7 pp of straight-through rate.

---

## The group solver cannot be evaluated on this data. The subset-sum premise does not hold.

Before building the solver I checked whether MULT records are actually subset
sums. Over 4,000 MULT match groups:

```
B equals a single A row          10.2%
B equals a SUM of A rows          0.0%      <- zero
B is neither                     89.8%
group balances (sumA == sumB)     8.2%
```

Not one record in the sample. With tolerance, over 2,916 records:

```
best achievable |B - subset(A)|     exact  7.7%   (all single-row, no sums)
                                  <= 1.00 10.1%
                                  <= 100  22.5%
median gap                        Rs 691.35
```

**Yet 99.6% of MULT amounts do appear as an A-side amount somewhere in the
dataset** -- just not inside their own match group.

### What that means

`matchId` is not a closed accounting unit. It is a reconciliation batch
identifier, and the A and B rows inside one do not sum to each other. So there is
no subset to find, because the grouping I would solve over is the wrong grouping.

And the decisive problem is the labels: **`targetAllocation` for a grouped record
is the literal string `MULT` and nothing more.** The dataset never records which
keys such a record maps to. Even a perfect solver could not be scored, because
the ground truth does not exist in the file.

### What I was about to build

The design put the subset-sum DP from the J.P. Morgan ECAI 2025 paper at the
centre and called it the differentiator. The paper is sound and the problem it
formalises is real -- Hyperswitch, HighRadius and a competing entry all describe
grouped matching as the hard case. **BenchRec simply does not contain that
pattern**, and I would have discovered this only after building the solver and
finding nothing to measure it against.

Third design assumption overturned by measurement, after the amount-aggregation
mistake and the autonomy metric.

### What is still true

The multiplicity detector stands. `MULT` is a real, labelled class covering 11.3%
of records, **100% of which were resolved manually** by the source institution,
and detecting it at 96.3% precision on a 5% alert budget is a genuine result. What
changes is the claim: **detect and route**, not **detect and solve**.

### Decision

Build the solver, validate it on generated data with subset structure injected by
construction, and state in the README that BenchRec cannot exercise it. A
capability that is tested but unexercised by the available data is honest. A
capability claimed on data that cannot demonstrate it is not.

---

## Group solver built and evaluated on ReconRiver

BenchRec has no subset-sum structure (measured earlier), so the solver is
evaluated on **ReconRiver** -- a synthetic 3-way reconciliation set that does:
`sum(net_amount) == bank credited_amount` holds exactly for **99.3%** of 1,499
settlement batches, median 3 payments per batch, up to 22.

Batch ids are **hidden** from the solver. It gets a bank credit and a pool of
plausible payments (same currency, settlement day -2 .. day 0) and must recover
the membership.

### Two evaluation bugs before any real number

**Same-day pool contained none of the answer.** Payments book 1-2 days *before*
settlement; **zero** land on the settlement day. My first pool was `day == day`,
so the true batch was never present and 68.6% came back infeasible. The solver
was right and the question was wrong.

**A stale cap in the eval script** refused 71% of instances at `max_candidates=40`
while the solver default was 64.

Both found by asking "is the answer even in the pool?" -- which should have been
the first check, not the third.

### Result

Pool median 98, p90 775. True batch present in the pool for **97%** of
settlements.

| cap | exact set | right sum, wrong set | infeasible | too large | exact among attempted |
|---|---|---|---|---|---|
| 64 | 1.2% | 0.9% | 0.0% | 97.9% | **59.3%** |
| 128 | 28.9% | 34.9% | 0.5% | 35.8% | **45.0%** |
| 256 | 28.9% | 34.9% | 0.5% | 35.8% | 45.0% |

Median solve 1.5 ms, p99 6 ms.

### What this actually shows

**The arithmetic is solved. The identification is not.**

34.9% of the time the solver returns a subset that sums to the target exactly and
is **not the true batch**. With 98 candidates and a 3-element answer there are
astronomically many subsets hitting any given total, and amount alone cannot
distinguish them. This is not a solver defect -- it is the problem being
genuinely under-determined at these pool sizes.

Smaller pools help sharply: **59.3% exact at <=64 candidates against 45.0% at
<=128**. Tighter blocking is worth more here than a better solver.

### The design consequence

A subset that sums correctly is **not** evidence it is the right subset. So the
solver must not auto-post on a single solution -- it should either confirm the
solution is *unique*, or hand the reviewer the candidate subsets ranked by a
second signal.

Posting a wrong-but-balancing subset is the worst failure this system can
produce: the books balance, the audit trail looks clean, and the money is
attributed to the wrong invoices. Left as detect-and-route.

---

## The database bug the design predicted, hit in the deployment step

Wiring the API, ten of sixteen tests failed at once:

```
sqlite3.ProgrammingError: SQLite objects created in a thread can only be used
in that same thread.
```

A web server handles requests on a thread pool. SQLite binds a connection to its
creating thread by default, so every write from a request handler raised.

**The design document warned about exactly this** -- it rejected DuckDB for state
on the grounds that a single-writer database "breaks on concurrent users". I then
built the audit log on SQLite and did not apply the same reasoning, because in a
single-threaded script it worked perfectly.

**Fix:** `check_same_thread=False` **and** a re-entrant lock around every
statement. The flag alone is the trap -- it removes the guard without making
writes safe, so the failure would move from a loud exception in development to
silent corruption under load.

Added `test_the_log_is_usable_from_several_threads`: four threads, twenty writes
each, assert eighty rows and no errors. It fails without the lock.

**Kept because** the bug was predicted in writing, in the same document, and
still shipped. Knowing a class of bug is not the same as checking for it in the
place you have just written.

---

## Deploy failed on a missing shared library, and took the process with it

Render built the image cleanly and the container died at start:

```
OSError: libgomp.so.1: cannot open shared object file: No such file or directory
  lightgbm/libpath.py -> ctypes.cdll.LoadLibrary(...)
```

LightGBM links against the GNU OpenMP runtime. `python:3.11-slim` does not ship
it. Locally it worked because macOS has its own OpenMP and my venv was never a
slim Debian image -- the classic "works on my machine" gap, in its most literal
form.

**Fix:** `apt-get install libgomp1` in the image, plus a build-time smoke import:

```dockerfile
RUN python -c "import lightgbm, allocation_agent.api; print('imports ok')"
```

That line turns a deploy-time crash into a build-time failure, which is cheaper
to see and impossible to miss.

### The worse half

The library was missing *and* the process died. Artifact loading runs inside
`create_app()`, so any exception there kills the container before it can serve a
single request -- a visitor gets a blank page, not a message.

I had already written a test asserting that *missing* artifacts degrade to a 503
rather than crashing. It passed. It did not cover artifacts that exist and fail
to load, which is what actually happened.

`load()` now catches, records the reason, and reports it on `/api/health`, with a
test that feeds it a deliberately corrupt pickle and asserts the app still boots.

**Kept because** the test I wrote for this exact failure mode was too narrow, and
the narrowness was invisible until reality picked the case I had not covered.

### Also worth noting

I could not verify the container locally: Docker CLI was installed but the daemon
was not running, so `docker build` failed and I pushed unverified. The build-time
import check is the compensating control for not being able to run the image.

---

## Live, and two things the deployment taught me

### The 50%-confidence cluster is the calibration working

The exception list showed a run of records all scoring *exactly* 50%. Exactly
one half is suspicious -- it means the margin between the top two candidates was
precisely zero.

It was. Twelve of the first 500 records, and in every one the top two allocation
keys had **identical amounts, in the same account, inside the same date window**:

```
b179260   5,402.59   160 candidates
   top-2 key amounts: [[5402.59], [5402.59]]
```

No feature the model has can separate those. It scores them identically, the
margin is zero, confidence is 0.5, and the gate queues them.

**That is correct behaviour, not a defect.** A system that broke the tie
arbitrarily would post one with false confidence and be wrong half the time on
this population. Saying "I cannot tell these apart" is the right answer, and it
falls out of deriving confidence from the margin rather than from a raw score.

### Throughput on free-tier hosting is 11x slower, and the README was wrong

Local: **495 records/sec**. Deployed on a free instance: **45/sec**.

Render's free tier is a fraction of a CPU. Nothing about the code changed. But
"524 rec/sec against a published commercial figure of 417" was written from a
laptop measurement and presented without qualification, and anyone opening the
live demo sees a number an order of magnitude lower.

Both figures are now stated, with the hardware attached to each. A throughput
claim without the machine it was measured on is not a claim.

---

## I cherry-picked my own demo, and the number that exposed it was 100%

Wiring the solver into the demo, I wrote an exporter that pulled 150
settlements out of ReconRiver. The endpoint came back **100% exact recovery**.

That is not a good result, it is a broken measurement. Two lines of the
exporter did it:

```python
if 2 <= len(mem) <= 8:            # only these batch sizes
if len(pool) > 60:                # and shrink the pool, keeping the answer
    pool = pd.concat([true_batch, pool.sample(55)])
```

The filter dropped every hard instance, and the subsample guaranteed the answer
was present in a pool small enough to be easy. I had built a demo dataset that
could only produce a good number. Nothing about it was deliberate -- I wrote
both lines to make the instances tractable and did not notice that "tractable"
and "easy" were the same edit.

Deleted both. Representative export: pool median **97**, p90 777, all batch
sizes. Exact recovery fell from 100% to **38.0%**, and **59.3%** of the answers
it gave were subsets that balanced and were not the recorded batch.

**A demo dataset needs the same split discipline as a training set.** I had been
careful about temporal leakage in the ranker for weeks and then hand-picked the
solver's evaluation set without registering it as the same class of mistake.

### The defect the honest number exposed

One case made it obvious:

```
SYNTH-BANK-000001   773.33   pool 97   true batch size 1
  773.33 = 144.98 + 170.34 + 45.36 + 367.82 + 44.83   (5 payments)
```

The true answer was a **single payment of 773.33 sitting in the pool**. The
solver walked past it and returned five unrelated payments that happened to add
up. The bitset DP answers "is this total reachable", and the backtrack then
returns whichever subset falls out of descending index order. Nothing anywhere
in it prefers a likely subset over an unlikely one.

I had recorded this as "the problem is genuinely under-determined" (above). That
was half right and it let me stop early. The problem *is* under-determined, and
the solver was also making it much worse than it needed to be.

### Two priors, both from how settlement works

**Fewest components wins.** Batches are small; pools are not. Each additional
component multiplies the coincidental subsets available, so the smallest
balancing subset is by a wide margin the likeliest one. This has to be a search
over cardinality, not a filter applied afterwards -- so reachability is split by
subset size, `layers[k]` holding the sums reachable using exactly *k* payments,
and the answer is read out of the lowest non-empty layer.

**A tie is not an answer.** If two different smallest subsets both balance, the
amounts do not distinguish them, and picking by index order produces something
that looks exactly like a match. Any rival subset of the same size must omit at
least one member of the one found -- so dropping each member in turn and
re-solving is a *complete* uniqueness test, not a sample. It costs k extra
passes, k typically 2.

| variant | coverage | precision | balanced-but-wrong | ties refused | p50 | p99 |
|---|---|---|---|---|---|---|
| reachability DP, first subset | 38.0% | 39.0% | 59.3% | -- | 2.5 ms | 14 ms |
| smallest subset | 83.3% | 93.3% | 6.0% | -- | 16 ms | 37 ms |
| smallest, refuse ties (shipped) | **75.3%** | **96.6%** | **2.7%** | 11.3% | 23 ms | 70 ms |

`coverage` = credits whose recorded batch it recovered. `precision` = of the
answers it gave, how many were the recorded batch.

The shipped variant is **not** the top row on coverage, and that is the choice.
It gives up 8pp of coverage to cut wrong answers from 6.0% to 2.7%; the 11.3% it
refuses go to a reviewer as ties rather than into the ledger under a subset that
merely balances. For a system that posts to a general ledger, an answer that is
wrong is far more expensive than an answer that is absent -- the books balance,
the audit trail looks clean, and the money sits against the wrong invoices.

`max_subset_size` turned out **not** to be an accuracy lever: 8 and 64 give
identical results, because min-cardinality search finds the real batch (median
size 2) long before the cap matters. It is a latency guard and is documented as
one rather than claimed as part of the gain.

### What I would have shipped

The version before this one reported a single number -- "solved" -- for 89.3% of
settlements, of which more than half were wrong. Splitting one metric into
**coverage and precision** is what made the defect visible; the aggregate hid it
completely, and the cherry-picked export hid it twice over.

### Where it fails matters more than how often

`75.3% coverage` is still one number covering two different problems. Split by
how many payments the batch really had:

| true batch | credits | recovered | wrong | ties refused | unresolved | unreachable |
|---|---|---|---|---|---|---|
| 1 payment | 30 | 7 | **4** | 3 | 16 | 21 |
| 2 payments | 62 | **62** | 0 | 0 | 0 | 0 |
| 3 payments | 58 | 44 | 0 | 14 | 0 | 0 |

**On genuine multi-payment batches: 106 of 120 recovered, zero wrong, 14 refused
as ties.** Every wrong answer in the whole run is a *single-payment* credit.

Which is not a grouping problem at all. A credit whose amount equals one payment
is an exact match and belongs on the matching path; it reached the solver only
because the demo routes every settlement there. And it fails there for a reason
that has nothing to do with subset-sum: for 21 of those 30 the true payment is
**not in the candidate pool** — 10 missing outright, 12 in pools of up to 765
that blow the 128 cap. With no k=1 available the solver does what it is asked
and finds a larger subset that balances.

So the `unreachable` column is reported separately. Blocking never offered the
answer; charging that to the solver would send me optimising the component that
is working. This is the same failure-locus argument the learning loop already
routes on, applied to my own headline metric.

**What this changes.** The next fix is not a better solver — it is routing
single-payment credits to the matcher and widening the settlement pool, in that
order. The solver's own remaining weakness is the 14 three-payment ties, and
those need a second signal beyond amount (order reference, merchant, timing) to
break, not more search.
