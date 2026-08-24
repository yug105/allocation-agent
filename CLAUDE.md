# Allocation Agent — working rules

Bank-to-ledger reconciliation. Deterministic where money is decided; a model
only writes explanations.

## Commands

```bash
uv run pytest -q -p no:warnings          # full suite, ~24s
uv run python scripts/eval_solver.py     # regenerates the README solver table
uv run python scripts/train_ranker.py    # retrains; the leakage gate runs here
ARTIFACTS_DIR=$PWD/artifacts uv run uvicorn allocation_agent.api:create_app \
  --factory --port 8077                  # serve locally
```

Live: https://allocation-agent.onrender.com — free tier, sleeps after ~15 min
idle and takes ~50s to wake. Docker builds have taken up to 8 minutes.

## The one rule everything else follows from

**A claim that nothing checks is not a claim.** Every defect found in this
repo, without exception, was something asserted and unenforced:

| what was claimed | what actually held |
|---|---|
| `max=2000` on the record input | advisory; the handler read `.value` anyway |
| a badge reading "Found the group" | derived from `status`, which is `solved` for wrong answers too |
| "rates are withheld below 20" | guarded the record count; precision is a rate over *answers* |
| "the narrator writes explanations" | constructed in `create_app()` and never called |
| "we checked for leakage" | a module with tests that nothing ran |
| a cap of 4 payments | admitted exactly one answer, and it was wrong |

When you write a rule, write the test that fails without it. When you state a
number, have code produce it.

## TDD, in this repo's sense

Write the failing test first — and make it fail *for the right reason*. Several
tests here passed while testing nothing:

- `assert "not" in verdict` matches *Nothing*, *Notable*, *Another*.
- `test_the_narrator_is_actually_invoked` was defined twice; Python kept the
  second, so the one asserting real behaviour never ran.
- Two rate tests bracketed `limit=1` and `limit=150` and passed for any
  threshold between 2 and 150.

`tests/test_suite_integrity.py` now catches duplicate names and assertion-free
bodies. Assert on exact strings or on a field, never a substring that a dozen
other strings contain.

## Numbers

**One owner per number, and code produces it.** Four different throughput
figures were live at once (40, 45, 495, 524) because each was typed where it
was needed. Constants live in `api.py` and reach the page through `/api/meta`.

Never hand-carry a measurement into prose. Regenerate the table
(`scripts/eval_solver.py`) — doing that caught a ties-refused figure that was
9.3% and had been written as 10.7%.

**Report coverage and precision separately.** One aggregate hid that 51% of
"solved" settlements were the wrong set.

## Evaluation

- Splits are temporal and group-respecting. The test set is frozen.
- Demo and sample data are **systematic** samples — every Nth by position,
  never by outcome. An exporter that filtered to easy instances once produced
  100% exact recovery, which is how it was caught.
- `assert_no_leakage` runs inside `train_ranker.py` and raises. A model trained
  on a leaked feature must not reach disk.
- No accuracy figure for uploaded files: there is no answer key, so any number
  would be invented.

## Money

Integer minor units everywhere. `BankRecord` refuses floats. Parsers refuse
more than two decimals rather than rounding — every other error announces
itself, that one balances and lies.

The decimal mark travels with the delimiter: a semicolon file uses a comma
decimal. Read `1250,00` as comma-thousands and it becomes 125,000.00.

## When something fails

Prefer refusing to guessing. `INFEASIBLE`, `AMBIGUOUS`, and "sent to a person"
are real answers. A wrong auto-post balances the books against the wrong
invoices, which is worse than an absent one.

Explain in the words of someone who has read only the problem statement. No
enum names on the page — `suspected_grouped` is a good variable and a terrible
label.

## Before claiming anything is done

Run the suite. Then check the thing you changed actually behaves differently —
grepping an attribute is not exercising a behaviour. Re-run the *whole* battery
after a fix, not the cases you touched: fixing three CSV defects introduced a
500 on empty files, caught only by the full re-run.
