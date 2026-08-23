"""Exactly what data goes where. Provenance, splits, and the loss funnel."""
from __future__ import annotations
import hashlib, os
import numpy as np, pandas as pd

from allocation_agent.adapters.benchrec import load_benchrec
from allocation_agent.eval.splits import temporal_split
from allocation_agent.match.blocker import BlockingConfig, block
from allocation_agent.stores.keys import KeyIndex

PATH = "data/benchrec_train.csv"
BCFG = BlockingConfig(date_slack_days=7)

print("="*78); print("1. PROVENANCE"); print("="*78)
size = os.path.getsize(PATH)
h = hashlib.sha256(open(PATH,'rb').read()).hexdigest()[:16]
raw = pd.read_csv(PATH, dtype=str, low_memory=False)
print(f"  file          {PATH}")
print(f"  size          {size/1e6:.1f} MB    sha256 {h}...")
print(f"  raw rows      {len(raw):,}   columns {len(raw.columns)}")
print(f"  source        BenchRec (ICAIF 2023) - obfuscated ledger+bank records,")
print(f"                Tier-1 institution production reconciliation")
print(f"  labels        real analyst decisions (targetAllocation)")
dt = pd.to_datetime(raw.A_valueDate, format='%m/%d/%y', errors='coerce')
print(f"  date range    {dt.min():%Y-%m-%d} to {dt.max():%Y-%m-%d}")

print("\n  columns DROPPED before any model sees them (outcome leaks):")
for c in ["generatorAllocation","matchRule","matchedBy","matchDate"]:
    print(f"    - {c:22} {raw[c].notna().mean()*100:5.1f}% populated")

ds = load_benchrec(raw)
print("\n"+"="*78); print("2. WHAT THE LOADER PRODUCES"); print("="*78)
print(f"  bank records (unit of work) {len(ds.records):>10,}")
print(f"    single-key                {(~ds.is_mult).sum():>10,}  ({(~ds.is_mult).mean()*100:.1f}%)")
print(f"    MULT (spans several)      {ds.is_mult.sum():>10,}  ({ds.is_mult.mean()*100:.1f}%)")
print(f"  ledger rows                 {len(ds.key_rows):>10,}")
idx = KeyIndex(ds.key_rows)
print(f"  distinct allocation keys    {idx.n_keys:>10,}   <- the label space")
print(f"  rows dropped by loader      {len(raw)*2 - len(ds.records) - len(ds.key_rows):>10,}  (blank amount on both sides)")

sp = temporal_split([r.day for r in ds.records], ds.match_ids, train_frac=0.7, val_frac=0.1)
print("\n"+"="*78); print("3. THE SPLIT  (temporal, group-respecting, test frozen)"); print("="*78)
days = np.array([r.day if r.day is not None else -1 for r in ds.records])
for name, ix in [("train", sp.train), ("val", sp.val), ("test", sp.test)]:
    d = days[ix]; d = d[d >= 0]
    print(f"  {name:6} {len(ix):>8,} records  ({len(ix)/len(ds.records)*100:4.1f}%)   "
          f"days {d.min():>6} .. {d.max():>6}   MULT {ds.is_mult[ix].mean()*100:4.1f}%")
overlap = set(np.array(ds.match_ids)[sp.train]) & set(np.array(ds.match_ids)[sp.test])
print(f"\n  matchId overlap train/test: {len(overlap)}   (must be 0)")

print("\n"+"="*78); print("4. WHAT EACH MODEL IS TRAINED ON"); print("="*78)
print(f"""  ranker (LightGBM LambdaRank)
    trained on   train split, single-key only, blocking hit only
                 -> ~1,774,128 (record, candidate) pairs, ~117,181 records
    positives    the true key                     negatives  other blocked candidates
    features     12, all derived from amount/date/key-shape. NO key identity.
    evaluated on test split, never seen during fit

  multiplicity detector (LightGBM binary)
    trained on   train split, ALL records ({len(sp.train):,})
    label        is_mult (targetAllocation == 'MULT')
    features     7, incl. account prior FITTED ON TRAIN ONLY
                 (an account's MULT rate summarises the label -> leak if fitted on test)
    evaluated on test split

  learning loop
    seed         first 3,000 TRAIN records (cold-start simulation)
    learns from  test split, in 10 batches of 4,000, revealed by a simulated reviewer
    NOTE         this reuses test as an online stream. Legitimate for measuring the
                 loop (the model never sees a record before deciding on it) but it is
                 NOT an additional held-out evaluation.""")

print("="*78); print("5. THE LOSS FUNNEL  (where records stop being resolvable)"); print("="*78)
single = [(i,l) for i,(l,m) in enumerate(zip(ds.labels, ds.is_mult)) if not m]
n_single = len(single)
no_cand = miss = 0
for i, lab in single:
    c = block(ds.records[i], idx, BCFG)
    if not c: no_cand += 1
    elif lab not in c: miss += 1
survive = n_single - no_cand - miss
rows = [
    ("bank records in file",            len(ds.records), None),
    ("  of which single-key",           n_single,        len(ds.records)),
    ("  blocking produced no candidate", -no_cand,       n_single),
    ("  true key not among candidates",  -miss,          n_single),
    ("= resolvable ceiling",             survive,        n_single),
]
for label, v, base in rows:
    pct = f"{abs(v)/base*100:6.2f}%" if base else ""
    print(f"  {label:36} {v:>9,}  {pct}")
print(f"\n  ceiling on top-1 accuracy: {survive/n_single*100:.2f}%   (measured 93.53%)")
print(f"  gap still on the table:    {survive/n_single*100 - 93.53:.2f} pp")
