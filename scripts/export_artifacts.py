"""Train once, export everything the deployed app needs.

The deployed service must not carry the 58 MB source file or retrain on boot.
It gets a fitted model, a key index, and a small demo slice.
"""
from __future__ import annotations
import json, pickle, time
from pathlib import Path
import numpy as np, pandas as pd

from allocation_agent.adapters.benchrec import load_benchrec
from allocation_agent.eval.splits import temporal_split
from allocation_agent.match.blocker import BlockingConfig, block
from allocation_agent.match.features import build_key_stats, featurise
from allocation_agent.match.multiplicity import AccountPrior, MultiplicityDetector, featurise_multiplicity
from allocation_agent.match.ranker import Ranker, RankerConfig
from allocation_agent.stores.keys import KeyIndex

OUT = Path("artifacts"); OUT.mkdir(exist_ok=True)
BCFG = BlockingConfig(date_slack_days=7)
DEMO_N = 4_000
RNG = np.random.default_rng(0)

t0=time.perf_counter()
ds = load_benchrec(pd.read_csv("data/benchrec_train.csv", dtype=str, low_memory=False))
idx = KeyIndex(ds.key_rows); kstats = build_key_stats(ds.key_rows)
sp = temporal_split([r.day for r in ds.records], ds.match_ids, train_frac=0.7, val_frac=0.1)
print(f"loaded in {time.perf_counter()-t0:.1f}s")

cands = {i: sorted(block(ds.records[i], idx, BCFG)) for i in list(sp.train)+list(sp.test)}
kamts = {k: st.amounts for k, st in kstats.items()}

# --- ranker
X,y,g = [],[],[]
for i in sp.train:
    if ds.is_mult[i]: continue
    lab=ds.labels[i]; c=cands[i]
    if lab not in c: continue
    sel=c
    if len(c)-1>24:
        neg=[k for k in c if k!=lab]; RNG.shuffle(neg); sel=[lab]+neg[:24]
    rows=[(k,featurise(ds.records[i],kstats[k],n_candidates=len(c))) for k in sel if k in kstats]
    if not rows: continue
    X.extend(v for _,v in rows); y.extend(1 if k==lab else 0 for k,_ in rows); g.append(len(rows))
ranker = Ranker(RankerConfig(objective="rank")).fit(np.asarray(X,np.float32), np.asarray(y,np.int8), group=g)
print(f"ranker trained on {len(X):,} pairs")

# --- multiplicity
he=np.array([any(ds.records[i].amount_minor in kamts.get(k,()) for k in cands[i]) for i in sp.train])
md=np.array([min((abs(ds.records[i].amount_minor-a) for k in cands[i] for a in kamts.get(k,())),default=1e12)
             for i in sp.train],float)
nc=np.array([len(cands[i]) for i in sp.train])
prior=AccountPrior.fit([ds.records[i].account or "?" for i in sp.train],
                       [ds.records[i].amount_minor for i in sp.train], ds.is_mult[sp.train])
mf=np.vstack([featurise_multiplicity(ds.records[i],n_candidates=int(nc[j]),has_exact=bool(he[j]),
              min_delta_minor=float(md[j]),prior=prior) for j,i in enumerate(sp.train)])
det=MultiplicityDetector().fit(mf, ds.is_mult[sp.train].astype(int))
print("multiplicity trained")

# --- demo slice: held-out test records + only the keys they can reach
demo_ix = list(sp.test[:DEMO_N])
needed = {k for i in demo_ix for k in cands[i]}
demo = {
    "records": [{"record_id": ds.records[i].record_id, "account": ds.records[i].account,
                 "amount_minor": ds.records[i].amount_minor, "day": ds.records[i].day,
                 "truth": ds.labels[i], "is_mult": bool(ds.is_mult[i])} for i in demo_ix],
    "key_rows": [{"key": r.key, "account": r.account, "amount_minor": r.amount_minor, "day": r.day}
                 for r in ds.key_rows if r.key in needed],
}
(OUT/"demo.json").write_text(json.dumps(demo))
pickle.dump({"ranker": ranker, "detector": det, "prior": prior}, open(OUT/"models.pkl","wb"))

meta = {"n_demo_records": len(demo_ix), "n_demo_keys": len(needed),
        "trained_on": int(len(sp.train)), "source": "BenchRec (ICAIF 2023)",
        "note": "held-out temporal split; these records were never seen in training"}
(OUT/"meta.json").write_text(json.dumps(meta, indent=2))
for f in OUT.iterdir(): print(f"  {f.name:16} {f.stat().st_size/1e6:6.2f} MB")
