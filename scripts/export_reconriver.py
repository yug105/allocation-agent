"""Export a ReconRiver demo slice.

BenchRec has no subset-sum structure -- measured, and recorded in the journal.
ReconRiver does: sum(net_amount) == bank credit for 99.3% of batches. Two demo
datasets so each shows the thing it can actually show, rather than one dataset
pretending to show both.
"""
import json
from pathlib import Path
import pandas as pd

SRC = Path("data/reconriver/month-end-close")
OUT = Path("artifacts"); OUT.mkdir(exist_ok=True)

bank = pd.read_csv(SRC/"bank_settlements.csv")
proc = pd.read_csv(SRC/"processor_transactions.csv").dropna(subset=["net_amount","settlement_batch_id"])
proc = proc.copy()
proc["cents"] = (proc.net_amount*100).round().astype("int64")
proc["day"]   = pd.to_datetime(proc.processor_event_time, errors="coerce").dt.floor("D")
bank = bank.copy()
bank["cents"] = (bank.credited_amount*100).round().astype("int64")
bank["day"]   = pd.to_datetime(bank.booked_at, errors="coerce").dt.floor("D")
proc = proc[proc.cents > 0]

# Representative sample. An earlier version kept only batches of 2-8 payments and
# subsampled the candidate pool, which produced 100% exact recovery -- because it
# had quietly removed every hard instance. The real distribution runs to 22
# payments per batch with a median pool near 98, so that is what ships.
keep = []
for _, b in bank.iterrows():
    mem = proc[proc.settlement_batch_id == b.settlement_batch_id]
    if len(mem) >= 1:
        keep.append(b)
    if len(keep) >= 150:
        break

settlements, payments = [], []
seen = set()
for b in keep:
    lo, hi = b.day - pd.Timedelta(days=2), b.day
    pool = proc[(proc.currency == b.currency) & (proc.day >= lo) & (proc.day <= hi)]
    settlements.append({
        "settlement_id": b.bank_entry_id, "amount_minor": int(b.cents),
        "currency": b.currency, "booked_at": str(b.booked_at)[:10],
        "truth": sorted(proc[proc.settlement_batch_id == b.settlement_batch_id]
                        .processor_transaction_id.tolist()),
        "pool": pool.processor_transaction_id.tolist(),
    })
    for _, p in pool.iterrows():
        if p.processor_transaction_id in seen: continue
        seen.add(p.processor_transaction_id)
        payments.append({"payment_id": p.processor_transaction_id,
                         "amount_minor": int(p.cents), "order_id": p.merchant_order_id,
                         "booked_at": str(p.processor_event_time)[:10]})

(OUT/"reconriver.json").write_text(json.dumps({"settlements": settlements, "payments": payments}))
sizes = [len(s["truth"]) for s in settlements]
pools = [len(s["pool"]) for s in settlements]
print(f"  {len(settlements)} settlements, {len(payments):,} payments")
print(f"  batch size  median {sorted(sizes)[len(sizes)//2]}  max {max(sizes)}")
print(f"  pool size   median {sorted(pools)[len(pools)//2]}  max {max(pools)}")
print(f"  {(OUT/'reconriver.json').stat().st_size/1e6:.2f} MB")
