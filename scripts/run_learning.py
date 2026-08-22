"""Does the learning loop actually work? Three arms, one answer.

A model already fitted on 134k records will not visibly improve from another two
thousand. A real deployment starts cold, so this does too: a deliberately small
initial fit, then learning from what the reviewer says.
"""
from __future__ import annotations
import time
import numpy as np, pandas as pd

from allocation_agent.adapters.benchrec import load_benchrec
from allocation_agent.decide.gate import GateConfig, Outcome, decide
from allocation_agent.eval.splits import temporal_split
from allocation_agent.learn.simulate import simulate
from allocation_agent.match.blocker import BlockingConfig, block
from allocation_agent.match.features import build_key_stats, featurise
from allocation_agent.match.ranker import Ranker, RankerConfig
from allocation_agent.stores.keys import KeyIndex

BCFG = BlockingConfig(date_slack_days=7)
GCFG = GateConfig(base=0.85, slope=0.02)
SEED_SIZE = 3_000          # deliberately small: a cold deployment
BATCH = 4_000

ds = load_benchrec(pd.read_csv("data/benchrec_train.csv", dtype=str, low_memory=False))
idx = KeyIndex(ds.key_rows); kstats = build_key_stats(ds.key_rows)
sp = temporal_split([r.day for r in ds.records], ds.match_ids, train_frac=0.7, val_frac=0.1)

t = time.perf_counter()
cands = {i: sorted(block(ds.records[i], idx, BCFG))
         for i in list(sp.train[:SEED_SIZE]) + list(sp.test)}
print(f"blocked {len(cands):,} records in {time.perf_counter()-t:.1f}s")

FEAT = {}
def feats(i):
    if i not in FEAT:
        c = cands[i]
        rows = [(k, featurise(ds.records[i], kstats[k], n_candidates=len(c)))
                for k in c if k in kstats]
        FEAT[i] = rows
    return FEAT[i]

def pairs_for(rows_and_keys):
    X, y, g = [], [], []
    for i, key in rows_and_keys:
        rows = feats(i)
        if not rows or key not in {k for k, _ in rows}:
            continue
        X.extend(v for _, v in rows); y.extend(1 if k == key else 0 for k, _ in rows)
        g.append(len(rows))
    if not X:
        return None
    return np.asarray(X, np.float32), np.asarray(y, np.int8), g

def make_arm(seed_pairs):
    """Fresh model + a refit closure that accumulates corrections."""
    store = list(seed_pairs)
    model = {"r": None}
    def fit():
        p = pairs_for(store)
        if p is None: return
        X, y, g = p
        model["r"] = Ranker(RankerConfig(objective="rank", n_estimators=150)).fit(X, y, group=g)
    fit()
    def refit(corrections):
        store.extend(corrections); fit()
    return model, refit, store

seed = [(i, ds.labels[i]) for i in sp.train[:SEED_SIZE] if not ds.is_mult[i]]
print(f"seed training set: {len(seed):,} records\n")

def decide_factory(model):
    def decide_batch(chunk):
        out = []
        for i in chunk:
            rows = feats(i)
            if not rows or model["r"] is None:
                out.append({"posted": False, "candidates": set(cands[i]),
                            "ranked": [], "routed_multiple": False}); continue
            keys = [k for k, _ in rows]
            X = np.vstack([v for _, v in rows])
            s = model["r"].score(X)
            order = np.argsort(-s)
            ranked = [keys[j] for j in order]
            margin = float(s[order[0]] - s[order[1]]) if len(order) > 1 else 1.0
            conf = float(1/(1+np.exp(-margin)))
            d = decide(confidence=conf, amount_minor=ds.records[i].amount_minor, config=GCFG)
            out.append({"posted": d.outcome is Outcome.POST, "candidates": set(cands[i]),
                        "ranked": ranked, "routed_multiple": False})
        return out
    return decide_batch

def truth(i): return ([ds.labels[i]], bool(ds.is_mult[i]))

order = list(sp.test)
results = []
for label, learn, placebo in [("learning on", True, False),
                              ("C-1 learning off", False, False),
                              ("C-3 placebo", True, True)]:
    t = time.perf_counter()
    model, refit, _ = make_arm(seed)
    res = simulate(label=label, indices=order, decide_batch=decide_factory(model),
                   refit=refit if learn else None, truth=truth,
                   batch_size=BATCH, placebo=placebo, rng=np.random.default_rng(0))
    results.append(res)
    print(f"{res}   [{time.perf_counter()-t:.0f}s]")

print("\n=== C-2 shuffled order (learning on) ===")
imps = []
for s in range(3):
    rng = np.random.default_rng(s)
    shuffled = list(rng.permutation(order))
    model, refit, _ = make_arm(seed)
    r = simulate(label=f"  order seed {s}", indices=shuffled,
                 decide_batch=decide_factory(model), refit=refit, truth=truth,
                 batch_size=BATCH, rng=rng)
    imps.append(r.improvement); print(r)
print(f"\n  improved under all {len(imps)} orderings: {all(i > 0 for i in imps)}")

print("\n" + "="*72)
learn, off, plac = results
print(f"learning effect (C-1)  : {learn.improvement - off.improvement:+.2f} pp over the no-learning arm")
print(f"placebo check (C-3)    : {plac.improvement:+.2f} pp  ->  "
      f"{'BROKEN: nonsense also improves' if plac.improvement > 1.0 else 'passes: nonsense does not help'}")
