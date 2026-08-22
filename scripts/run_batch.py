"""End-to-end run on the held-out split, with the numbers that go in the README."""
from __future__ import annotations
import json, time
import numpy as np, pandas as pd

from allocation_agent.adapters.benchrec import load_benchrec
from allocation_agent.decide.gate import GateConfig
from allocation_agent.eval.splits import temporal_split
from allocation_agent.match.blocker import BlockingConfig, block
from allocation_agent.match.features import build_key_stats, featurise
from allocation_agent.match.multiplicity import (
    AccountPrior, MultiplicityDetector, featurise_multiplicity)
from allocation_agent.match.ranker import Ranker, RankerConfig
from allocation_agent.pipeline import run_batch
from allocation_agent.report.audit import AuditLog, RunConfig
from allocation_agent.stores.keys import KeyIndex

BCFG = BlockingConfig(date_slack_days=7)
GCFG = GateConfig(base=0.85, slope=0.02, cap=0.995, policy_version="v0.1")
RNG = np.random.default_rng(0)

t0 = time.perf_counter()
ds = load_benchrec(pd.read_csv("data/benchrec_train.csv", dtype=str, low_memory=False))
idx = KeyIndex(ds.key_rows); kstats = build_key_stats(ds.key_rows)
kamts = {k: st.amounts for k, st in kstats.items()}
sp = temporal_split([r.day for r in ds.records], ds.match_ids, 0.7, 0.1)
print(f"loaded {ds.n_records:,} records / {idx.n_keys:,} keys in {time.perf_counter()-t0:.1f}s")
print(f"split: {sp}\n")

# ---- blocking pass (shared) ------------------------------------------------
t = time.perf_counter()
cands = [sorted(block(r, idx, BCFG)) for r in ds.records]
has_exact = np.array([any(r.amount_minor in kamts.get(k, ()) for k in c)
                      for r, c in zip(ds.records, cands)])
mindelta = np.array([min((abs(r.amount_minor - a) for k in c for a in kamts.get(k, ())),
                         default=1e12) for r, c in zip(ds.records, cands)], float)
ncand = np.array([len(c) for c in cands])
print(f"blocked in {time.perf_counter()-t:.1f}s  ({ds.n_records/(time.perf_counter()-t):,.0f} rec/s)")

# ---- train ranker ----------------------------------------------------------
def pairs(ix, max_neg=24):
    X, y, g = [], [], []
    for i in ix:
        if ds.is_mult[i]: continue
        lab = ds.labels[i]; c = cands[i]
        if lab not in c: continue
        sel = c
        if len(c) - 1 > max_neg:
            neg = [k for k in c if k != lab]; RNG.shuffle(neg); sel = [lab] + neg[:max_neg]
        rows = [featurise(ds.records[i], kstats[k], n_candidates=len(c)) for k in sel if k in kstats]
        if not rows: continue
        X.extend(rows); y.extend(1 if k == lab else 0 for k in sel if k in kstats); g.append(len(rows))
    return np.asarray(X, np.float32), np.asarray(y, np.int8), g

t = time.perf_counter()
Xtr, ytr, gtr = pairs(sp.train)
ranker = Ranker(RankerConfig(objective="rank")).fit(Xtr, ytr, group=gtr)
print(f"ranker trained on {len(Xtr):,} pairs in {time.perf_counter()-t:.1f}s")

# ---- train multiplicity ----------------------------------------------------
prior = AccountPrior.fit([ds.records[i].account or "?" for i in sp.train],
                         [ds.records[i].amount_minor for i in sp.train],
                         ds.is_mult[sp.train])
mfeat = np.vstack([featurise_multiplicity(r, n_candidates=int(ncand[i]),
                   has_exact=bool(has_exact[i]), min_delta_minor=float(mindelta[i]),
                   prior=prior) for i, r in enumerate(ds.records)])
det = MultiplicityDetector().fit(mfeat[sp.train], ds.is_mult[sp.train].astype(int))
print("multiplicity detector trained\n")

# ---- run the batch ---------------------------------------------------------
test_records = [ds.records[i] for i in sp.test]
audit = AuditLog("data/audit.db")
res = run_batch(test_records, idx, kstats, audit,
                run_config=RunConfig(approved_by="yug",
                                     blocking={"date_slack_days": 7},
                                     gate={"base": 0.85, "slope": 0.02},
                                     policy_version="v0.1",
                                     notes="held-out temporal split"),
                blocking=BCFG, gate=GCFG, ranker=ranker,
                multiplicity=det, mult_features=mfeat[sp.test], mult_threshold=0.7)
print(res, "\n")

# ---- was it right? ---------------------------------------------------------
rows = {d["record_id"]: d for d in audit.decisions(res.run_id)}
truth = {ds.records[i].record_id: (ds.labels[i], bool(ds.is_mult[i])) for i in sp.test}
post_ok = post_n = q_n = 0
mult_caught = mult_total = mult_wrong = 0
for rid, d in rows.items():
    lab, is_mult = truth[rid]
    keys = json.loads(d["chosen_keys"])
    if is_mult: mult_total += 1
    if d["path"] == "multiplicity":
        mult_caught += is_mult; mult_wrong += (not is_mult)
    elif d["outcome"] == "post":
        post_n += 1; post_ok += (not is_mult and keys and keys[0] == lab)
    else:
        q_n += 1

print("=" * 62)
print(f"{'straight-through rate':38} {res.straight_through_rate*100:7.1f}%")
print(f"{'precision of auto-posted matches':38} {post_ok/max(post_n,1)*100:7.1f}%   ({post_ok:,}/{post_n:,})")
print(f"{'grouped records routed to review':38} {mult_caught:,}/{mult_total:,} "
      f"({mult_caught/max(mult_total,1)*100:.1f}%)")
print(f"{'single records wrongly routed':38} {mult_wrong:,}")
print(f"{'throughput':38} {res.records_per_second:7,.0f} rec/sec")
print(f"{'LLM calls on the matching path':38} {res.llm_calls:7,}")
print("=" * 62)
print("\nexception taxonomy:")
for k, v in res.exceptions.most_common():
    print(f"  {k:26} {v:7,}  ({v/res.n_records*100:5.1f}%)")
print(f"\naudit rows written: {len(rows):,}  (one per record, append-only)")
