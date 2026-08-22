"""Multiplicity detector: does one bank record span several allocation keys?

11.3% of the batch. Zero were automated by the institution's own engine.
Baseline is the obvious rule; the model must beat it or be dropped.
"""
from __future__ import annotations
import time
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_fscore_support

from allocation_agent.adapters.benchrec import load_benchrec
from allocation_agent.eval.splits import temporal_split
from allocation_agent.match.blocker import BlockingConfig, block
from allocation_agent.match.multiplicity import (
    AccountPrior, MultiplicityDetector, MULT_FEATURE_NAMES, featurise_multiplicity)
from allocation_agent.stores.keys import KeyIndex

CFG = BlockingConfig(date_slack_days=7)

ds = load_benchrec(pd.read_csv("data/benchrec_train.csv", dtype=str, low_memory=False))
idx = KeyIndex(ds.key_rows)
kamts: dict[str, set[int]] = {}
for r in ds.key_rows:
    kamts.setdefault(r.key, set()).add(r.amount_minor)

t = time.perf_counter()
has_exact, ncand, mindelta = [], [], []
for r in ds.records:
    c = block(r, idx, CFG)
    amts = [a for k in c for a in kamts.get(k, ())]
    ncand.append(len(c))
    has_exact.append(r.amount_minor in amts if amts else False)
    mindelta.append(min((abs(r.amount_minor - a) for a in amts), default=1e12))
has_exact = np.array(has_exact); ncand = np.array(ncand); mindelta = np.array(mindelta, float)
print(f"blocked all records in {time.perf_counter()-t:.1f}s")

sp = temporal_split([r.day for r in ds.records], ds.match_ids, train_frac=0.7, val_frac=0.1)
y = ds.is_mult.astype(int)
print(f"split: {sp}   MULT rate train {y[sp.train].mean()*100:.1f}%  test {y[sp.test].mean()*100:.1f}%\n")

# prior fitted on TRAIN ONLY -- an account's MULT rate summarises the label
prior = AccountPrior.fit([ds.records[i].account or "?" for i in sp.train],
                         [ds.records[i].amount_minor for i in sp.train],
                         y[sp.train])

def build(ix):
    return np.vstack([featurise_multiplicity(ds.records[i], n_candidates=int(ncand[i]),
                      has_exact=bool(has_exact[i]), min_delta_minor=float(mindelta[i]),
                      prior=prior) for i in ix])

Xtr, Xte = build(sp.train), build(sp.test)
ytr, yte = y[sp.train], y[sp.test]

# --- baseline: no exact-amount candidate => MULT
base_pred = ~has_exact[sp.test]
bp, br, bf, _ = precision_recall_fscore_support(yte, base_pred, average="binary", zero_division=0)

t = time.perf_counter()
det = MultiplicityDetector().fit(Xtr, ytr)
p = det.predict_proba(Xte)
print(f"trained + scored in {time.perf_counter()-t:.1f}s\n")

print(f"{'':34} {'prec':>7} {'recall':>7} {'F1':>7}")
print("-" * 60)
print(f"{'baseline: no exact-amount match':34} {bp*100:6.1f}% {br*100:6.1f}% {bf*100:6.1f}%")
for th in (0.3, 0.5, 0.7, 0.9):
    pr, rc, f1, _ = precision_recall_fscore_support(yte, p >= th, average="binary", zero_division=0)
    print(f"{'model @ ' + str(th):34} {pr*100:6.1f}% {rc*100:6.1f}% {f1*100:6.1f}%")
print(f"\nPR-AUC (average precision): {average_precision_score(yte, p)*100:.1f}%")
print(f"   positive rate {yte.mean()*100:.1f}%  ->  a random model scores that")

print("\n=== precision at a fixed alert budget ===")
for budget in (0.05, 0.10, 0.15):
    k = int(len(p) * budget)
    top = np.argsort(-p)[:k]
    print(f"  flag top {budget*100:4.0f}% ({k:,} records): precision {yte[top].mean()*100:5.1f}%  "
          f"recall {yte[top].sum()/yte.sum()*100:5.1f}%")

print("\ntop features:")
for k, v in list(det.importances.items())[:5]:
    print(f"  {k:28} {v*100:5.1f}%")
