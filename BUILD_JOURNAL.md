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

---

## Reading my own demo as a judge who has not read anything

Asked to look at the deployed page knowing only the problem statement, the
verdict was that it is incomprehensible: what data is going in is not stated,
your own data cannot go in, the workflow is invisible, and some things say
"resolved" and others "unresolved" with no explanation of either.

Checking rather than arguing, one of those is not a wording problem:

```
endpoints built and tested : 7
endpoints the page can reach: 2   (/api/run, /api/settlements)
```

`/api/meta` carries the dataset provenance and the page never called it.
`/api/upload` **validated a CSV and returned "reconciliation of uploaded files
is not wired yet"**. `/api/connect` returned 501. So "I cannot feed it my own
data" was not a usability complaint — the capability did not exist. I had built
the validation, tested it, documented it in the README, and never noticed that
nothing behind it did anything.

**Every one of those endpoints had passing tests.** The tests asserted that
upload rejects a non-CSV, rejects an oversized file, rejects missing columns.
All true, all passing, and all of it testing a door with no room behind it. A
test suite confirms the code does what it says; it cannot notice that what it
says is not worth doing.

### What changed

**Uploads actually reconcile now.** Two CSVs — bank side and ledger side —
parsed by column *discovery* rather than by demanding our names (`Txn Amt
(INR)`, `Val Dt`, `A/c No` all resolve), then run through `_match_one`, the
same function the demo calls. A separate path for user data would have made the
demo's measured numbers evidence for nothing but the demo.

Two rules pull opposite ways in that parser and both are kept: forgiving about
column names, unforgiving about money. A third decimal place is refused with the
row number rather than rounded, because every other error announces itself and
that one balances and lies. Date layout is chosen **once for the whole file** by
vote — deciding per row silently mixes day-first and month-first inside one file
and shifts dates by weeks with nothing visibly wrong.

**No precision is reported for uploads.** The demo can score itself because
BenchRec is labelled; an uploaded file has no answer key, so any accuracy figure
would be invented. The endpoint returns that sentence rather than a number.

**The workflow is on the page.** Four steps — narrow, group check, rank, decide
— each with what it does in one sentence, and after a run, how many records left
the pipeline at that step. The counts are the explanation.

**No internal name is shown to a visitor.** `suspected_grouped` is a good
variable name and a terrible thing to put in front of someone who has read only
the problem statement. It now reads "One payment, several invoices", with a
sentence saying why matching it to a single entry would be wrong. Three tests
hold the line: no enum in the visible page, no page field absent from the API,
no built endpoint unreachable from the page.

The last one exists because the page had been reading `m.n_keys` and `m.n_train`
against a meta endpoint that returns `n_demo_keys` and `trained_on` — two of the
three dataset figures rendered as nothing at all, and nothing failed.

---

## The sample pair I shipped got 3 of 4 wrong, and both causes were real

The "use a sample pair" button exists so a judge sees the upload path work in
one click. Run against four rows whose answers are obvious, it matched one:

```
B-1002    80.50  ->  INV-502   73.1%   under the 85% bar, sent for review
B-1003  4,300.00 ->  —         63% "one payment covering several entries"
```

Both of those have an exact-amount, same-account entry sitting in the ledger.
Two distinct defects, and the sample data only made them visible.

### The 73.1% was a constant wearing a measurement's clothes

```python
margin = float(scores[order[0]] - scores[order[1]])  if len(order) > 1 else 1.0
confidence = 1 / (1 + exp(-margin))
```

With one candidate there is no runner-up, so `margin` fell back to `1.0` and
confidence became `sigmoid(1.0) = 0.731` — **every time, for every record, at
any amount**. The base bar is 0.85. So a lone candidate could never post, no
matter how perfect the match. Not "rarely": never.

Why no test caught it: blocking never returns fewer than four candidates on
BenchRec.

```
candidates returned, held-out set:
    4 candidates :    390
   5+ candidates :   3610
```

**Zero records can reach the branch.** Every test I had ran on that
distribution, so the branch was unreachable in test and hit constantly by small
uploaded files — the exact case the feature was built for. A dataset that never
exercises a path is not coverage of it.

### The grouping check was overruling evidence it cannot read

`B-1003` had an exact match the ranker scored at 100%, and a 63% "looks
grouped" guess ran first and discarded it. Rather than argue about which should
win, I measured the detector on that subpopulation:

| records with exactly one exact-amount candidate | |
|---|---|
| grouping check fired | 41 |
| …of which actually grouped | **5 (12.2%)** |
| …so it was wrong on | **36 (87.8%)** |
| check stayed quiet | 2,304 (11 actually grouped) |

The detector is right **96.3%** of the time overall and **12.2%** here. A single
entry accounting for the whole amount defeats the premise of the grouped path,
and the detector has no feature that can see it.

### Fixing it twice, the second time properly

First attempt: give a lone exact match its own path at measured confidence
(98.98%, from 2,321 of 2,345 held-out records) and skip the grouping check.
That worked on the sample and cost real money on the measured set:

| | posted | precision | straight-through |
|---|---|---|---|
| before | 3,115 | **99.45%** | 77.88% |
| bypass the check | 3,172 | 98.99% | 79.30% |

+57 posted, +42 right, **+15 wrong** — marginal quality 74%, against a ranker
measured at 99.45%. I had conflated two things: *the grouping check should not
overrule an exact match* (justified, 87.8% wrong) and *0.9898 should replace the
ranker's judgement* (not justified at all — the ranker has a real margin
whenever there are several candidates).

Separating them: the grouping check is skipped only on a lone exact match; the
ranker still decides; and the measured 98.98% is used **only when there is no
runner-up to form a margin from** — the actual bug.

| | posted | precision | straight-through |
|---|---|---|---|
| before | 3,115 | 99.45% | 77.88% |
| bypass the check | 3,172 | 98.99% | 79.30% |
| **shipped** | **3,176** | **99.31%** | **79.40%** |

Better than the bypass on *both* axes — more posted at higher precision. Against
the original, +61 auto-posts for −0.14pp precision: +56 right, +5 wrong,
marginal quality 92%.

**The lesson is about the first version, not the third.** It made the sample
look right and the measured set worse, and I would have shipped it if I had
stopped at the sample. A demo that starts working is not evidence the change was
correct — the held-out set is, and it disagreed.

---

## A badge that said "Found the group" over a sentence saying it was wrong

Asked for one credit on the settlement tab, the page returned:

```
SYNTH-BANK-000001 · GBP 773.33      [ Found the group ]
773.33 = 144.98 + 170.34 + 45.36 + 367.82 + 44.83
"This balances but is not the recorded batch — a summing subset is not
 proof of the right subset, so it goes to review."
```

The badge and the sentence directly contradict each other. The label came from
`status`, and `status` is `solved` whenever a subset *balances* — which says
nothing about whether it is the right subset. That distinction is the entire
point of the component and the label erased it.

Fixed by moving the verdict server-side. It is a judgement about the answer, so
the code that knows both `status` and `exact` makes it, and the page displays a
string. Four tests now assert a balancing-but-wrong subset is never labelled
"found" and never rendered as good news.

### And the answer itself should not have been offered

The deeper question is why a five-payment sum was claimed at all. Measured
across the 150 settlements, by how many payments the answer used:

| payments used | answers | right | precision | subsets of that size available |
|---|---|---|---|---|
| 1 | 7 | 7 | 100.0% | ~93 |
| 2 | 62 | 62 | 100.0% | ~4,371 |
| 3 | 46 | 44 | 95.7% | ~138,415 |
| 4 | 1 | 0 | **0.0%** | ~3,612,280 |
| 5 | 1 | 0 | **0.0%** | ~64,446,024 |

The right-hand column is the mechanism. There are about four thousand ways to
pick two payments out of a pool and sixty-four million ways to pick five. A
two-payment sum that lands exactly on the target is evidence; a five-payment one
is a coincidence that was always going to be available.

Lowering `max_subset_size` from 8 to 4:

| cap | answers | right | wrong | precision | coverage |
|---|---|---|---|---|---|
| 8 | 117 | 113 | 4 | 96.6% | 75.3% |
| 4 | 116 | 113 | 3 | 97.4% | 75.3% |
| 3 | 115 | 113 | 2 | 98.3% | 75.3% |

**Coverage is identical at every cap.** Every correct answer used three payments
or fewer, so tightening removed only wrong ones — it cost nothing.

Two things this cannot show, both recorded in the config docstring rather than
buried: the 0% above three payments is *two records*, not a rate; and
ReconRiver's own batches never exceed three, so nothing here validates any
particular larger cap. The default sits one above the largest batch the data
contains rather than exactly at it, so it is not simply fitted to the observed
maximum. A book with genuinely larger batches needs it raised and needs its own
measurement to say where.

### Percentages over one record

The same screen reported `0.0% of the groups it named were right` — arithmetic
over a single credit, reading as a measured property of the system. Rates are
now withheld below twenty records and shown as counts, with a line saying the
credits run in file order so a small number is the first few and not a sample.

### The refusal message was also false

With the cap in place the credit came back "No combination of the 97 candidate
payments sums to this credit." A five-payment combination sums to it exactly —
that is how the wrong answer arose. The honest sentence is that longer
combinations may exist and are deliberately not claimed, which is what it now
says.

---

## An independent review of the last two commits found thirteen defects

Every one was in work I had described as finished. Six I verified by running
the code before touching it; the rest were confirmed by reading. The pattern is
worth more than the list: **most were shipped alongside a test that was written
to catch exactly that defect and did not.**

| | defect | the test that should have caught it |
|---|---|---|
| F1 | `rates_meaningful` guarded on the record count while precision's denominator is the *answer* count. 20 records → 3 answers → `0.0%` printed in green | two tests, at `limit=1` and `limit=150`, that pass for any threshold between 2 and 150 and never touch the boundary |
| F2 | the refusal sentence said "six payments" while the cap was 4 | a test asserting `"not" in expl and "claimed" in expl` — true of the contradictory sentence |
| F6 | `is_exact` tested before `status`, so a record with **no answer** badges green "Found the group" | a test asserting `"wrong" in v or "not" in v`, which passes for "Nothing found" |
| F9 | the cap test hardcoded `<= 4` instead of reading the config | — |

`"not" in verdict` matches *Nothing*, *Notable*, *Another*. An assertion that
loose is not a weaker test, it is not a test.

### The one that was a judgement error

I capped claimed answers at 4 payments while the measurement said precision
collapses to 0% above 3, reasoning that capping at 3 would be "fitting to the
largest batch the data happens to contain". The result:

```
answers by size, cap = 4:   1 payment  7 (7 right)
                            2 payments 62 (62 right)
                            3 payments 46 (44 right)
                            4 payments  1 (0 right)   <- the only one the cap admitted
```

**The cap of 4 admitted exactly one answer and that answer was wrong.**
Declining to fit the data is not a virtue when the only thing the extra room
admits is a mistake. Now 3: 98.3% precision against 96.6%, at identical
coverage, because every correct answer used three payments or fewer.

Worse than the cap itself was the paragraph I wrote defending it, which cited
sizes 1, 2, 3 and 5 — **skipping 4**, the only size the decision turned on. The
prose argued for a cap of 3 while the code shipped 4, and I did not notice
because I had left the inconvenient row out of the sentence.

### Two documents that disagreed with the code

`DESIGN.md` still said `max_subset_size = 8` and "NOT an accuracy lever" — the
exact claim the change reversed — two commits after I had updated that file
specifically to match what shipped. And `scripts/eval_solver.py`, which
generates the README table, no longer reproduced it: both configured variants
inherited the new default, so the 96.6% row was unreachable and "(shipped)" sat
on the wrong row. Every variant now states its cap explicitly.

Regenerating that table immediately caught a stale figure I had hand-carried
into README.md and DESIGN.md: ties refused was 9.3%, not 10.7%. A number typed
by hand is a number nobody checked.

---

## Auditing what is real: six of twenty-one modules were decoration

The review covered two commits. The instruction that followed was harder: *only
keep things that are 100% correct and working, and do not assume anything.* So
I traced every endpoint and every module rather than reasoning about them.

**Endpoints, by whether they do work:**

```
REAL       GET  /            GET /api/health   GET /api/meta
REAL       POST /api/run     POST /api/settlements   POST /api/reconcile
STUB/DEAD  POST /api/upload    -> {"note": "reconciliation ... is not wired yet"}
STUB/DEAD  POST /api/connect   -> 501 not implemented
```

**Modules, by whether any live path reaches them.** A static import graph gave
one false positive worth recording: `match.ranker` appeared unreachable because
`api.py` never imports it — the class arrives through `pickle`, which needs it
importable at runtime. Checking the pickle's contents rather than the imports
corrected that. The genuinely unreachable ones were `decide.openrouter`,
`eval.leakage` and `learn.casebase` — each referenced **only by its own test**.

### The worst of it: the narrator was never called

`narrator` appears exactly once in `api.py`: on the line that constructs it.
Every explanation on the page was an f-string written inline. So the component's
whole point — *it cannot emit a figure that is not in the payload, and there is
a test that feeds it a lying backend to prove it* — protected nothing anybody
read. The README describes narration as an LLM component of the system. It was
a component of the test suite.

Worse, `openrouter.py` reads `OPENROUTER_API_KEY` and **nothing ever constructed
it**, so README line 217 — "optional: add an OpenRouter key for narration" — was
false in effect. Setting the key did nothing at all.

Now wired where it belongs. The narrator's real job is residual diagnosis:
naming *why* two amounts differ. It runs on a queued record with a genuine gap,
which is where a reviewer needs the reason rather than "below threshold":

```
b179218  5,635.03
  Best guess is USD_2/26/16_3445812260_195 AAB, but at 58% it is under the
  89% this amount requires. Sent for review rather than posted.
  Gap of -636.83 equals one outstanding line; the payment appears partial.
```

14 invocations across 500 records — the 36 queued records include 22 with no
gap to explain, and inventing a cause for those would be the same defect in a
new place. A key in the environment now switches the model backend on for real.

### A test that was not running

Fixing this surfaced a second kind of invisible failure: `test_the_narrator_
is_actually_invoked` was defined **twice** in `test_api.py`. Python keeps the
last definition, so the shadowed copy — the one asserting the real behaviour —
never ran and never failed. `tests/test_suite_integrity.py` now walks the AST of
every test file and fails on a duplicate name, and on any test whose body
contains no assertion at all.

That second check found four tests with no `assert`. All four were fine —
they call `assert_no_leakage` and `validate_numbers`, functions whose contract
*is* to raise, so the bare call is an assertion with a different spelling. The
check now recognises that rather than forcing four correct tests to be rewritten
to satisfy it.

### The leakage gate now runs

`eval.leakage` caught four outcome columns during development and was then never
run again — "we checked for leakage" was a claim about the past rather than a
property of the artefact. It is now a gate inside `train_ranker.py`, raising
rather than warning, so a model trained on a leaked feature cannot reach disk.

### Two things kept and labelled rather than deleted

The **learning loop** is a real measured experiment and no API path touches it:
correcting a decision in the deployed demo retrains nothing. The **case base**
is correct, has twelve tests, and is called by nothing — not even the learning
experiment. Deleting it was the obvious reading of "no decoration", and I did
delete it, then put it back: the design for it was requested deliberately, and
quietly removing requested work is not my call to make. Both are now labelled as
what they are, in the README and in the architecture diagram.

---

## Asking the demo for four credits returned four failures

Set the settlement count to 1 and everything fails. Set it to 4 and everything
fails. The component looked broken because, for anyone sampling small, it was.

The cause is not the solver:

```
where the unsolvable credits sit in file order
  credits   0- 14: 11 of 15 unsolvable  XXXXXXXXXXX....
  credits  15- 29:  3 of 15 unsolvable  .....X...X..X..
  credits  30-149:  0 of 120 unsolvable ........................
```

**ReconRiver front-loads every hard case, and the demo took the first N.** Of
150 credits, 14 are unsolvable and 11 of them are in the first fifteen. The
worst possible sample, taken by default.

Fixed by sampling evenly across the file — every Nth credit, chosen by position
and never by outcome, and labelled as such on the page. Selecting the ones that
succeed would be the cherry-picking this project already got wrong once with the
solver export. Asking for four now returns two recovered groups with their
arithmetic, one refused tie, and one genuine failure.

### Why those fourteen fail, and three fixes that did not work

Two overlapping causes. Ten credits are exactly **2.00 more** than the sum of
their batch — a flat bank charge, so no exact subset can ever reach the credit.
Twelve have candidate pools of **765-777** against a 128 cap, all of them USD,
because the pool is built from currency and date and the export carries no
account or merchant field to narrow it further.

The solver has a `tolerance_minor` setting built for exactly this, and the API
passes 0. Turning it on looked like the obvious fix. It is not:

| attempt | answers | right | wrong | precision |
|---|---|---|---|---|
| exact only (shipped) | 115 | 113 | 2 | **98.3%** |
| tolerance 2.00 | 123 | 113 | 10 | 91.1% |
| fewest-payments before exact, tolerance 2.00 | 135 | 76 | 59 | 56.3% |
| exact at *target − 2.00* when the plain solve fails | 121 | 115 | 6 | 95.0% |

**Tolerance never recovered a single fee credit.** It added eight answers and
all eight were wrong: opening a ±2.00 window across ~138,000 three-subsets finds
coincidences far faster than truth. Reordering to prefer the fewest payments
before exactness was worse still — at size 1 with a window, many single payments
land near any target. Even solving at the *known* charge recovered only 2 of 10,
because most fee credits are also the oversized-pool ones and are refused before
any solving happens, and it cost 4 wrong answers to get 2 right ones.

So they stay unrecovered. The refusal says what is known and nothing more.

### The diagnostic I built and deleted the same hour

Between those attempts I added a near-miss probe: when nothing matches exactly,
search within a 5.00 window and tell the reviewer how far short the closest
group falls. It shipped this sentence:

> The closest group of 3 comes to **0.11** short of the credit, which is
> consistent with a flat bank charge.

The real charge is 2.00. The probe had found the nearest *coincidental* subset —
with a 10.00-wide window and thousands of three-subsets, something always lands
close — and narrated it as a diagnosis. For `SYNTH-BANK-000001` it did not even
find the true batch, which is a single payment 2.00 away.

I removed it. It is the same defect as the badge that said "Found the group"
over a wrong answer and the paragraph that skipped size 4: presenting a
coincidence with the grammar of evidence. That it appeared in the very work
written to fix that pattern is the part worth remembering.

---

## Three copies of one fact

A recovered credit rendered like this:

```
3,989.47 = 1,899.61 + 2,089.86            <- headline
  SYNTH-PROC-000970      1,899.61         <- itemised, with provenance
  SYNTH-PROC-001280      2,089.86
3,989.47 = 1,899.61 + 2,089.86 (2 payments)   <- explanation, saying it again
```

The explanation was `_sum_sentence`, written back when the arithmetic was the
thing that needed showing. Once the page rendered the equation as a headline and
itemised the payments beneath it, the sentence became a third copy — and on a
wrong group it pushed the one line that carried a judgement to the end of two
lines of repetition.

Each line now says something the others cannot. The headline is the arithmetic,
the list is which payments and from which orders, and the sentence is the part
no sum can state:

> These 2 payments are the ones the settlement file records for this credit,
> found from a pool of 65 without being told the batch number.

`_sum_sentence` had no callers left and was deleted rather than kept for later.
A test now asserts no explanation restates its own arithmetic.

---

## Thirty-one files a real person might upload

The upload tab had been tested with files I wrote to demonstrate it working.
Testing it with files people actually have found four defects, three of them
invisible because the response looked reasonable.

**A European Excel export was refused outright.** Excel on a machine with a
comma decimal separator writes `a;b;c`, not `a,b,c`. The whole row parsed as one
column and the error read "could not find a column for: account" — technically
true, uselessly so.

Supporting it is not just a delimiter, and this is the part that matters: the
same locale writes **`1250,00`** for one thousand two hundred and fifty. The
existing parser strips commas as thousands separators, so that file would have
been read as **125,000.00** — a hundredfold error, in money, with nothing
visibly wrong anywhere. The decimal mark now travels with the delimiter:
semicolon implies comma-decimal, which is the convention Excel itself follows.

**`N/A` in the amount column reported "the amount is blank".** It is not blank;
it is unreadable. The message sent someone hunting for an empty cell that does
not exist. The cell's actual contents are now quoted back.

**An ambiguous date was resolved silently.** `03/01/2026` is the third of
January or the first of March depending on who wrote the file. One layout is
chosen for the whole file — deliberately, since deciding per row mixes both
inside one file — but nothing said which. A US bank export was being read
day-first and could sit eleven months from where its author meant it. The chosen
layout now comes back in the response and is shown on the page: *"Dates read as
DD/MM/YYYY in the bank file."*

**And I broke empty files while fixing the others.** Sniffing the delimiter
reads the first row; on an empty file there is none, and `next()` without a
default raised `StopIteration` — a 500 where there had been a clean 400. Caught
only because I re-ran the whole battery after the fix rather than testing the
three cases I had just changed.

Things that already worked and are now pinned: UTF-8 BOM from Excel, CRLF line
endings, tab separation, `$` and thousands separators, parenthesised negatives,
quoted fields containing commas, ragged rows, headers in any case or with
padding, files with no id column, 1,000 rows, and eight distinct ways of being
malformed.

The other two tabs took 30 requests — zero and negative limits, limits past the
end of the dataset, every boundary of every threshold, wrong types, arrays where
objects belong, missing multipart parts. No crashes, no 500s, and **every record
accounted for in every run**: `posted + queued + grouped + none == n_records`
held in all of them.

---

## A fix that enforced nothing

The previous entry describes lowering the demo's record-count control so a
visitor could not trigger a hundred-second run. A review of that commit found
thirteen defects in it, and the summary was the part worth keeping:

> the dominant theme is that this change is prose plus an HTML attribute, with
> no enforcement anywhere.

Correct on both counts.

**`max=2000` stops nothing.** The input is not inside a `<form>`, and the click
handler reads `+$('#n').value` regardless. Typing 4000 marks the field
`:invalid` and returns "4000" anyway, the page POSTs it, and the server accepts
`le=10**9`. The run I had claimed to prevent was one keystroke away. The test I
wrote passed because it grepped the attribute and never exercised the behaviour.

**The server default was untouched.** `RunRequest.limit` still defaulted to 500,
so any caller omitting the field got exactly the thirteen-second run the commit
message says was removed. The change lived entirely in a static HTML attribute.

### The number had no owner

Worse, and more embarrassing given this repo already has a test named
`test_the_threshold_is_sent_to_the_page_rather_than_duplicated_in_it`:

```
page says          40 records a second
README says        ~45/sec        (three places)
README also says   524 rec/sec on a laptop   (line 47)
README also says   495                       (line 155)
```

Four figures for two machines, each typed by hand where it was needed. The page
had also claimed "roughly six seconds" for 200 records — at its own stated 40/sec
that is five. On a page whose entire argument is that its numbers check out, the
first division a reader performs fails.

Measured properly, median of three warm runs each: **795 rec/sec** on an 8-core
arm64 laptop, **45 rec/sec** on the deployed free instance. Both laptop figures
in the README were stale — the direct path had made the pipeline faster and
nothing re-measured. Both now live in `api.py` as constants, served through
`/api/meta`, and the page states no rate of its own.

### The estimate is now derived, not written

A two-point prose estimate says nothing about the values between its endpoints:
someone typing 1,500 still waited thirty-seven seconds behind a button reading
"running…". So the page computes it live from the number in the box and the best
rate known for the machine, and **the maximum went back up to 4,000** — the full
held-out set the page advertises in its own first section. Refusing to run half
of an advertised dataset, with no sentence explaining why, was a worse answer
than showing the cost:

```
   200 -> About 4 seconds — 45 records/sec on the deployed free instance.
  4000 -> About 89 seconds — 45 records/sec on the deployed free instance.
```

After a run it re-derives from what it actually observed, so the same page on a
laptop stops quoting the free tier at you — the previous copy asserted "this
runs on a free instance … about 40 records a second" while the tile two inches
below reported 795.

Three smaller ones from the same review: the 422 path rendered FastAPI's
list-shaped `detail` through `new Error()` as the literal text
**`[object Object]`**, which is what a visitor saw for an empty or decimal
record count; the cold-start wake-up — the dominant wait, and documented in the
README — went unmentioned in the very paragraph written to set expectations; and
the sentence that must be read before clicking was styled `.faint`, the page's
lowest-contrast text at ~4.1:1, below the 4.5:1 floor.

**The pattern across this whole session.** Every one of these was a claim
enforced by nothing: a badge derived from the wrong field, a guard on the wrong
denominator, a cap justified by prose that skipped the row it turned on, a
component constructed and never called, and now a limit that was only a
suggestion. The fix each time was the same shape — move the thing being claimed
to where it can be checked, then check it.

---

## Writing the rules down, then making something check them

The repo had no `CLAUDE.md`, no linter in the loop, and no gate on committing.
Every defect in the entries above was a claim enforced by nothing, so the
correction is not another document — it is machinery.

**`CLAUDE.md`** now carries the rules that were being re-derived every session,
each with the failure that earned it: one owner per number, assert on exact
strings rather than substrings, sample by position never by outcome, report
coverage and precision separately, refuse rather than guess.

**A pre-commit gate** runs the suite and blocks on failure (exit 2). Verified
both ways before wiring: green tree passes, and a tree with one deliberately
failing test is refused with the failure printed.

**Ruff on every edited file**, with the rule set pinned in `pyproject.toml`.

### The linter found two things 370 tests did not

```
api.py:32  F811  Redefinition of unused `diagnose_residual` from line 32
api.py:203 F841  Local variable `bcfg` is assigned to but never used
```

The first is a sed replacement that ran twice — harmless. The second is not.
`bcfg = BlockingConfig(date_slack_days=7)` was built in the demo endpoint and
never read, because extracting `_match_one` moved blocking inside it. So
`date_slack_days` was written in **four** places: a dead local, two audit
records, and the matcher. Widening the window would have left the audit log
claiming a value the run did not use — the same no-single-owner defect as the
four throughput figures, sitting in the audit trail this time. There is now one
`BLOCKING` constant and the audit records read from it.

**F811 is the rule that catches a duplicate test definition** — the exact
failure where `test_the_narrator_is_actually_invoked` existed twice and the
shadowed copy, the one asserting real behaviour, never ran. It is now enforced
by a linter rather than by a test I remembered to write afterwards.

Ruff's last finding was the same species as `"not" in verdict`:

```
test_narrate.py:70  RUF043  Pattern passed to `match=` contains metacharacters
```

`pytest.raises(NarrationError, match="7.50")` treats `.` as a wildcard, so the
test passes on `7X50`. An assertion looser than it looks, found by a tool rather
than by reading. Now `match=r"7\.50"`.

Baseline is zero ruff errors, so the hook has something meaningful to enforce; a
linter that always prints 58 lines enforces nothing, because nobody reads it.

**What I did not copy.** The setup this came from runs fourteen MCP servers and
fourteen plugins. That guide's own advice is that context is precious and to
keep under ten enabled — and none of those servers touch this problem. Two
hooks, one rules file, one pinned lint config.


---

## Researching what a reconciliation product actually shows, and what I built instead

Two searches, and they indicted the same thing from opposite directions.

On demo pages: judges scan above the fold for about thirty seconds and form an
opinion that rarely changes. My hero was a title, a paragraph and three dataset
counts — **nothing had happened, and nothing could until a button was pressed.**

On reconciliation products: every one of them measures **unreconciled value**,
**value at risk** and **exception aging**, with queues ordered by amount and
each item tagged by failure reason. Every figure on my page was a record count.
"1,535 records" is a statistic. "15.6M sitting in review" is a reason to care.

### The number the page was hiding from itself

Running all 4,000 held-out records and summing by value rather than counting:

```
posted automatically      76,469,898.05    79.4% of records, 99.3% of them right
needs a person            17,904,105.92    19.0% of the value
  suspected_grouped   526 items  15,581,523.14
  queued              298 items   2,322,582.78
```

**87% of the money needing a human is the grouped case** — the one this bank's
own rules engine resolved 0% of. That is the entire argument for the project,
in one figure, and the page had never said it.

It is precomputed at export time rather than at startup: 4,000 records take
~90s on the deployed free instance and the health check would fail before the
container was ready. A test pins it against a live run, because a cached figure
that drifts from what the buttons produce is worse than no figure.

Summing it caught one more thing. `exceptions` is capped at 100 for payload
size, and my first pass computed the queue total from that list — describing a
quarter of the queue as the whole of it. Totals now sum over every exception,
with a test asserting the reported total exceeds what the payload returns.

### The KPI I researched, measured, and did not build

Exception aging is on every product's list, so I went to add it. Measured
first:

```
age bucket   records   auto-posted   needs a human
  0-6  days      802         74.9%           25.1%
  7-13 days     1071         80.2%           19.8%
 14-20 days     1056         81.3%           18.7%
 21-27 days     1071         80.0%           20.0%

median age, auto-posted   14 days
median age, needs a human 11 days
```

Flat. The held-out set spans **27 days**, so every record lands in one 30-day
bucket and the standard chart would be a single bar.

The reason is not a data quirk, it is a category error: **aging counts how long
an item has sat unresolved in a running book**, and this is a one-month
snapshot resolved in a single batch. There is nothing to age.

A real uploaded ledger spanning months is a different matter, so it is computed
when the span supports it and refused with the reason when it does not — the
same rule as `rates_meaningful` and the evenly-spaced sampling. On the demo the
page says why there is no chart; on a year-long upload it draws one.

Adding the chart anyway would have looked more like a finance product and been
worth nothing, which is the failure mode this whole journal is about.

---

## Which of the 824 do I open first?

Each exception already said why it stopped. That answers a question a reviewer
does not have. Facing a queue, they want to know **which to open first** and how
much of the backlog the first few clear.

Every exception now carries its share of the queue's value and a running total:

```
  # record           amount  of queue  running  stopped at
  1 b179363      210,500.05      2.9%     2.9%  suspected_grouped
  2 b179391      176,402.57      2.4%     5.2%  suspected_grouped
  3 b181079      158,025.92      2.1%     7.4%  suspected_grouped
  ...
  clearing the top 10 clears 17.9% of the queue
  clearing the top 40 clears 44.8%
```

**Forty of 380 exceptions — a tenth of them — hold 45% of the value.** The
order is worth more than the count, which is exactly what a queue sorted by
record id destroys.

The shares are taken against the whole queue rather than the 100 exceptions the
payload returns; computed against the returned list they would sum to 1 while
describing a quarter of the backlog. There is a test for that, because this is
the third time the 100-item cap has nearly produced a wrong number.

One thing fell out of the ordering that no metric had shown: **every one of the
eight largest exceptions is `suspected_grouped`.** The hero says 87% of the
queue's value is the grouped case; sorting by size says the same thing again at
the top of the list, without being asked to.

### The design document had drifted again

Checking before writing: four of eight claims in `DESIGN.md` no longer matched
the code — the grouping threshold, the direct path, the overview endpoint and
aging. That is the third drift, and the earlier two were caught by a reviewer
rather than by me.

The check itself is eight lines of Python comparing strings in the doc against
strings in the source. It should not be a thing I remember to run. Written down
here so the next drift is caught by something rather than noticed by someone.

`B4.1 Direct key` was the worst of it: it still described a *reference* match at
confidence 1.0, which was never built — BenchRec's reference fields are 0%
populated — while the code matches on amount at a measured 0.9898 and runs
before the grouping check rather than after.

---

## A review that found a bug I had already fixed everywhere except one file

Fifteen points, and the top one was right.

**Scores and selection indexed different lists.** In `pipeline.py`:

```python
X = np.vstack([featurise(...) for k in candidates if k in key_stats])
scores = ranker.score(X)
chosen = candidates[int(order[0])]      # the unfiltered list
```

`X` is built from the *filtered* candidates and `chosen` reads the *unfiltered*
one. Every position after the first key missing from `key_stats` is shifted, so
a correct ranking still returns the wrong ledger entry.

The live path does not have it — `_match_one` reads `usable[int(order[0])]`,
and it does so because extracting that function forced the filtered list to be
named. The bug survived in the file the extraction did not touch. **The fix was
a side effect of a refactor rather than a decision, which is why it did not
propagate.**

**A tie was being auto-posted at 90%.** `_fallback_choice` scored candidates on
`(exact_amount, date_gap)` and returned the best. Three ledger entries of the
same amount on the same day score identically; it returned whichever sorted
first and labelled it 0.90. The README says ties are refused and this path did
the opposite. It now returns `(None, None)` when the top two tie on the
deciding evidence, and `decide(confidence=None)` cannot post at any amount.

**Every non-POST was counted as `below_threshold`.** A record with no
confidence at all is `no_candidate`; counting it as a threshold miss makes the
exception breakdown describe something that did not happen. It now asks the
gate what it decided.

**`llm_calls_on_matching_path: 0` was a typed constant.** "How do you know
there were zero?" — "I wrote zero." It is now the narrator's own counter
measured across the loop.

### The leakage question, which was the one worth being afraid of

> *A temporal split alone doesn't prevent feature-level leakage if statistics
> were computed globally.*

Correct in general, so I measured rather than reasoned:

```
keys 5,869   rows per key: median 1
a key's own date span: median 0d   p90 0d   max 0d
```

**Every allocation key's rows fall on a single day.** `KeyStats` is therefore
that key's own rows and never a pooled statistic — there is nothing aggregated
across records to leak. The one genuine corpus statistic is `AccountPrior`, an
account's historical MULT rate, which is a summary of the label; both training
scripts fit it on `sp.train` only, and its docstring says why.

A related finding that is *not* leakage: 100% of records have a candidate key
dated after them. That is the blocker offering a ±7 day window on purpose,
because payments and bookings do not align, and this reconciles a period rather
than making real-time decisions.

### Points I agree with and have not acted on

Calibration is the real one. `sigmoid(top - second)` is a monotone transform of
a LambdaRank margin, not a probability, and calling it *confidence* while
comparing it to 0.85 borrows a meaning it has not earned. The end-to-end number
that matters — 99.3% of what it posts is correct — is measured directly and does
not depend on the name. But the gate's thresholds are tuned against a quantity
whose units are arbitrary, and that should be Platt or isotonic on the
validation split. Recorded rather than done.

---

## Fifteen more, one of which was mine and wrong

**The one I got to check rather than fix.** The review's top item was that
`predict_proba(X)[0]` reads a row, not a probability — sklearn returns
`[[p0, p1]]`, so `float(...)` of that is a type error and the matching path
would be broken outright. Verified before touching anything:
`MultiplicityDetector.predict_proba` is a wrapper that already does `[:, 1]`,
returns shape `(3,)`, and `float(out[0])` is `0.59`. A correct reading of the
sklearn contract, applied to a class that is not sklearn.

**The one that was a genuine conceptual error.** `DIRECT_CONFIDENCE = 0.9898`
is BenchRec's measured rate, and `match_one` handed it to uploaded files too,
on the strength of a comment reading *"same matching path, so the measured demo
numbers say something about uploaded files too."* They do not. **Same code path
is not the same data distribution** — an uploaded ledger can have duplicate
amounts, a different date discipline, another account structure entirely.

The first fix was worse than the problem: discount it to 0.60 on uncalibrated
data. That made a lone exact amount rank *below* a wide ranker margin on the
same file, when the exact amount is the stronger of the two. The inconsistency
came from patching one path and not the other.

What ships instead: the figure is used unchanged and **labelled**. Calibrated
data records `benchrec_heldout`; anything else records
`benchrec_heldout_unvalidated`, the calibrator is bypassed for
`uncalibrated_sigmoid`, and the response carries
`confidence_validated_for_this_data: false`. The number a caller sees now says
where it came from.

**Two claims that lived only in prose.** *"The AI can fail, the payment system
cannot"* was enforced in the batch runner and nowhere else — a ranker that
raised inside the API returned a 500 and took the request with it. And `usable
== []` was reported as `no_candidate` when blocking had in fact found entries
that could not be scored, which is a different failure. Both now have code and
a test; `n_blocked` and `n_scored` are separate fields, because one name meant
"blocked" on one path and "scored" on another.

Also: the ledger had no row cap while `KeyIndex` and `build_key_stats` are
rebuilt from it on every request, and `ConnectRequest` was dead code left over
from an endpoint deleted days ago.

### The closing point is the one worth keeping

> Every strong architectural claim should have a corresponding test.

That is the whole failure mode of this project stated in one line. The comments
here have repeatedly been more confident than the code, and a reader trusts a
comment. Six of this batch's tests exist only to make a sentence enforceable.
