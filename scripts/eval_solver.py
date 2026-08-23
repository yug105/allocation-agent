"""Group solver on ReconRiver: recover batch membership with the batch id hidden.

Many payments settle into one bank credit. The solver is given the credit and a
pool of plausible payments -- same currency, nearby date -- and must find which
subset produced it. Ground truth is the true batch membership, which it never sees.
"""
from __future__ import annotations
import time, sys
import numpy as np, pandas as pd
from allocation_agent.match.solver import SolverConfig, SolverStatus, solve_subset

SCEN = sys.argv[1] if len(sys.argv) > 1 else "month-end-close"
DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
D = f"data/reconriver/{SCEN}"

bank = pd.read_csv(f"{D}/bank_settlements.csv")
proc = pd.read_csv(f"{D}/processor_transactions.csv")
proc = proc.dropna(subset=["net_amount", "settlement_batch_id"]).copy()
proc["cents"] = (proc.net_amount * 100).round().astype("int64")
proc["day"] = pd.to_datetime(proc.processor_event_time, errors="coerce").dt.floor("D")
bank["cents"] = (bank.credited_amount * 100).round().astype("int64")
bank["day"] = pd.to_datetime(bank.booked_at, errors="coerce").dt.floor("D")

print(f"scenario {SCEN}   {len(bank):,} settlements, {len(proc):,} payments")
print(f"candidate pool = same currency, booked_at +/- {DAYS} day(s). batch id HIDDEN.\n")

CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 64
cfg = SolverConfig(tolerance_minor=0, max_candidates=CAP)
truth = proc.groupby("settlement_batch_id").cents.apply(list).to_dict()

stats = {"exact_set": 0, "right_sum_wrong_set": 0, "infeasible": 0, "too_large": 0}
pool_sizes, batch_sizes, times = [], [], []

for _, b in bank.iterrows():
    true_members = sorted(truth.get(b.settlement_batch_id, []))
    if not true_members:
        continue
    batch_sizes.append(len(true_members))

    pool = proc[(proc.currency == b.currency)
                & (proc.day >= b.day - pd.Timedelta(days=DAYS))
                & (proc.day <= b.day + pd.Timedelta(days=DAYS))]
    amounts = pool.cents.tolist()
    pool_sizes.append(len(amounts))

    t = time.perf_counter()
    r = solve_subset(target_minor=int(b.cents), candidates_minor=amounts, config=cfg)
    times.append(time.perf_counter() - t)

    if r.status is SolverStatus.TOO_LARGE:
        stats["too_large"] += 1
    elif r.status is SolverStatus.INFEASIBLE:
        stats["infeasible"] += 1
    elif sorted(amounts[i] for i in r.indices) == true_members:
        stats["exact_set"] += 1
    else:
        stats["right_sum_wrong_set"] += 1

n = sum(stats.values())
print(f"{'outcome':26} {'count':>8} {'share':>8}")
print("-" * 46)
for k in ["exact_set", "right_sum_wrong_set", "infeasible", "too_large"]:
    print(f"{k:26} {stats[k]:>8,} {stats[k]/max(n,1)*100:7.1f}%")
print(f"{'TOTAL':26} {n:>8,}")

ps, bs, ts = np.array(pool_sizes), np.array(batch_sizes), np.array(times)
print(f"\npool size    median {np.median(ps):.0f}  p90 {np.percentile(ps,90):.0f}  max {ps.max()}")
print(f"batch size   median {np.median(bs):.0f}  p90 {np.percentile(bs,90):.0f}  max {bs.max()}")
print(f"solve time   median {np.median(ts)*1000:.2f} ms  p99 {np.percentile(ts,99)*1000:.1f} ms  total {ts.sum():.1f}s")
print(f"\nexact recovery among solvable (excl. too_large): "
      f"{stats['exact_set']/max(n-stats['too_large'],1)*100:.1f}%")
