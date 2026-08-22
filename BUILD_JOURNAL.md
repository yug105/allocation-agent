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
