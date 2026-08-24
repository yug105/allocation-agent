"""Blocking recall/size tradeoff on the real batch.

Recall is a ceiling on everything downstream. Run this before building the ranker.
"""
import sys
import time

import pandas as pd

from allocation_agent.adapters.benchrec import load_benchrec
from allocation_agent.eval.blocking_recall import measure_blocking
from allocation_agent.match.blocker import BlockingConfig
from allocation_agent.stores.keys import KeyIndex

path = sys.argv[1] if len(sys.argv) > 1 else "data/benchrec_train.csv"

t = time.perf_counter()
df = pd.read_csv(path, dtype=str, low_memory=False)
print(f"read {len(df):,} rows in {time.perf_counter()-t:.1f}s")

t = time.perf_counter()
ds = load_benchrec(df)
print(f"parsed in {time.perf_counter()-t:.1f}s  ->  "
      f"{len(ds.records):,} records ({ds.is_mult.sum():,} MULT), {len(ds.key_rows):,} ledger rows")

t = time.perf_counter()
idx = KeyIndex(ds.key_rows)
print(f"indexed {idx.n_keys:,} keys in {time.perf_counter()-t:.1f}s\n")

print(f"{'strategy':32}  {'recall':>7}  {'median':>7} {'p95':>7} {'mean':>8}  {'sec':>5}")
print("-" * 78)
configs = [
    ("account only (no predicates)", BlockingConfig(use_amount=False, use_date=True, date_slack_days=10_000)),
    ("amount only",                  BlockingConfig(use_date=False)),
    *[(f"amount + date +/-{s}d", BlockingConfig(date_slack_days=s)) for s in (0, 1, 2, 3, 7, 14)],
]
for name, cfg in configs:
    if "account only" in name:
        continue  # 20k-day window is not a meaningful run; account-only measured separately
    t = time.perf_counter()
    r = measure_blocking(ds, cfg, index=idx)
    print(f"{name:32}  {r.recall*100:6.2f}%  {r.median_candidates:7.0f} "
          f"{r.p95_candidates:7.0f} {r.mean_candidates:8.1f}  {time.perf_counter()-t:5.1f}")
