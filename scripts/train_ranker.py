"""Train the candidate ranker and measure top-1 against baselines.

Temporal split, group-respecting. The bar is a trivial baseline (exact amount,
tiebreak nearest date) and the ceiling is blocking recall.
"""
from __future__ import annotations
import sys, time
import numpy as np, pandas as pd

from allocation_agent.adapters.benchrec import load_benchrec
from allocation_agent.eval.splits import temporal_split
from allocation_agent.match.blocker import BlockingConfig, block
from allocation_agent.match.features import build_key_stats, featurise, FEATURE_NAMES
from allocation_agent.match.ranker import Ranker, RankerConfig
from allocation_agent.stores.keys import KeyIndex

RNG = np.random.default_rng(0)
CFG = BlockingConfig(date_slack_days=7)


def build_pairs(ds, idx, kstats, indices, max_neg=None):
    """Return (X, y, group_starts, truth) for the given row indices."""
    X, y, groups, truth = [], [], [], []
    for i in indices:
        if ds.is_mult[i]:
            continue
        rec, lab = ds.records[i], ds.labels[i]
        cands = block(rec, idx, CFG)
        if lab not in cands:
            continue                      # blocking miss: unrecoverable, excluded from top-1
        cands = list(cands)
        n = len(cands)
        if max_neg is not None and n - 1 > max_neg:
            negs = [c for c in cands if c != lab]
            RNG.shuffle(negs)
            cands = [lab] + negs[:max_neg]
        start = len(X)
        for k in cands:
            st = kstats.get(k)
            if st is None:
                continue
            X.append(featurise(rec, st, n_candidates=n))
            y.append(1 if k == lab else 0)
        groups.append((start, len(X), lab, cands))
        truth.append(lab)
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8), groups, truth)


def top1(scores, groups):
    hit = 0
    for (s, e, lab, cands) in groups:
        if e <= s:
            continue
        best = cands[int(np.argmax(scores[s:e]))]
        hit += best == lab
    return hit / len(groups) if groups else 0.0


def baseline_top1(ds, idx, kstats, indices):
    hit = n = 0
    for i in indices:
        if ds.is_mult[i]:
            continue
        rec, lab = ds.records[i], ds.labels[i]
        cands = block(rec, idx, CFG)
        if lab not in cands:
            continue
        n += 1
        best, bs = None, None
        for k in cands:
            st = kstats.get(k)
            if st is None:
                continue
            amt_miss = 0 if rec.amount_minor in st.amounts else 1
            gap = min((abs(rec.day - d) for d in st.days), default=999) if rec.day is not None else 999
            s = (amt_miss, gap)
            if bs is None or s < bs:
                best, bs = k, s
        hit += best == lab
    return hit / n if n else 0.0


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/benchrec_train.csv"
    t0 = time.perf_counter()
    ds = load_benchrec(pd.read_csv(path, dtype=str, low_memory=False))
    idx = KeyIndex(ds.key_rows)
    kstats = build_key_stats(ds.key_rows)
    print(f"loaded {ds.n_records:,} records, {idx.n_keys:,} keys  ({time.perf_counter()-t0:.1f}s)")

    days = [r.day for r in ds.records]
    sp = temporal_split(days, ds.match_ids, train_frac=0.7, val_frac=0.1)
    print(f"temporal split: {sp}\n")

    t = time.perf_counter()
    Xtr, ytr, gtr, _ = build_pairs(ds, idx, kstats, sp.train, max_neg=RankerConfig().max_negatives_per_record)
    Xva, yva, gva, _ = build_pairs(ds, idx, kstats, sp.val, max_neg=RankerConfig().max_negatives_per_record)
    Xte, yte, gte, _ = build_pairs(ds, idx, kstats, sp.test)
    print(f"pairs  train {len(Xtr):,}  val {len(Xva):,}  test {len(Xte):,}   "
          f"({time.perf_counter()-t:.1f}s)")
    print(f"scoreable records  train {len(gtr):,}  test {len(gte):,}\n")

    t = time.perf_counter()
    r = Ranker().fit(Xtr, ytr, Xva, yva)
    print(f"trained in {time.perf_counter()-t:.1f}s")

    t = time.perf_counter()
    s = r.score(Xte)
    model_acc = top1(s, gte)
    dur = time.perf_counter() - t

    base = baseline_top1(ds, idx, kstats, sp.test)
    print(f"\n{'':28} {'top-1':>8}")
    print("-" * 40)
    print(f"{'trivial baseline':28} {base*100:7.2f}%")
    print(f"{'ranker (temporal, held-out)':28} {model_acc*100:7.2f}%")
    print(f"{'blocking ceiling':28} {98.94:7.2f}%")
    print(f"\nlift over baseline: {(model_acc-base)*100:+.2f} pp")
    print(f"scoring {len(gte):,} records in {dur:.1f}s")

    print("\ntop features:")
    for k, v in list(r.importances.items())[:6]:
        print(f"  {k:20} {v*100:5.1f}%")


if __name__ == "__main__":
    main()
