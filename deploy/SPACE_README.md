---
title: Allocation Agent
emoji: 🧮
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Reconciliation agent — matches bank records to ledger keys
---

# Allocation Agent

Matches bank records to ledger allocation keys, and detects when one payment
covers several.

**The demo runs on held-out records the models never saw during training**, so
the precision shown is measured against ground truth rather than asserted.

## What it does

| | |
|---|---|
| precision of auto-posted matches | **99.2%** |
| straight-through rate | ~80% |
| grouped-payment detection | **96.3%** precision at a 5% review budget |
| LLM calls on the matching path | **0** |
| records unaccounted for | **0** |

Every record reaches exactly one outcome: posted, queued for review, flagged as
covering several ledger entries, or reported as having no candidate at all.

## The finding it is built on

Data is [BenchRec](https://www.kaggle.com/datasets/benchmarkteam/benchrec-real-world-cash-reconciliation-dataset)
(ICAIF 2023) — 172,023 reconciliations from a Tier-1 institution's production
system, labelled by real analysts.

Split by match shape, their rules engine automated **94% of one-to-one matches
and 0% of the grouped ones**. All 6,618 went to a person.

## Where a language model is and is not used

Blocking, ranking, grouped-payment detection and the posting gate are
deterministic or classical ML — anything deciding where money goes must be
reproducible and defensible. A language model is used only for reading messy
column names and writing exception explanations, and it is forbidden from
introducing a number that did not come from the engine.

Source and full method: see the linked repository.
