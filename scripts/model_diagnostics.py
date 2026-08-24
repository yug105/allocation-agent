"""Overfitting and classification diagnostics for both trained models.

The README reported test-set top-1 for the ranker and precision-at-budget for
the detector. Neither of those answers "is it overfitting?", which is a
*comparison* between what the model scores on data it was fitted on and data it
was not. This script makes that comparison, on the same temporal split the
models were trained under, and adds the classification metrics that were
missing: recall, F1, ROC-AUC, and a confusion matrix at the operating point.

Reads the trained artefacts rather than refitting, so the numbers describe the
models that are actually deployed.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from allocation_agent.adapters.benchrec import load_benchrec
from allocation_agent.eval.splits import temporal_split
from allocation_agent.match.blocker import BlockingConfig, block
from allocation_agent.match.features import build_key_stats, featurise
from allocation_agent.match.multiplicity import featurise_multiplicity
from allocation_agent.stores.keys import KeyIndex

ROOT = Path(__file__).resolve().parents[1]
CFG = BlockingConfig(date_slack_days=7)
RNG = np.random.default_rng(0)


def ranker_pairs(ds, idx, kstats, indices, max_neg=None):
    """Candidate rows for scoreable records, grouped per record."""
    X, groups = [], []
    for i in indices:
        if ds.is_mult[i]:
            continue
        rec, lab = ds.records[i], ds.labels[i]
        cands = block(rec, idx, CFG)
        if lab not in cands:
            continue                     # blocking miss: no model can recover it
        cands = list(cands)
        n = len(cands)
        if max_neg is not None and n - 1 > max_neg:
            negs = [c for c in cands if c != lab]
            RNG.shuffle(negs)
            cands = [lab] + negs[:max_neg]
        start = len(X)
        kept = []
        for k in cands:
            st = kstats.get(k)
            if st is None:
                continue
            X.append(featurise(rec, st, n_candidates=n))
            kept.append(k)
        if len(kept) > 0:
            groups.append((start, len(X), lab, kept))
    return np.asarray(X, dtype=np.float32), groups


def top1(scores, groups):
    hit = sum(cands[int(np.argmax(scores[s:e]))] == lab
              for (s, e, lab, cands) in groups if e > s)
    return hit / len(groups) if groups else 0.0


def mrr(scores, groups):
    """Mean reciprocal rank of the true key. Falls before top-1 does."""
    total = 0.0
    for (s, e, lab, cands) in groups:
        if e <= s:
            continue
        order = np.argsort(-scores[s:e])
        rank = next(r for r, j in enumerate(order, 1) if cands[j] == lab)
        total += 1.0 / rank
    return total / len(groups) if groups else 0.0


def mult_rows(ds, idx, kstats, indices, prior):
    X, y = [], []
    for i in indices:
        rec = ds.records[i]
        cands = list(block(rec, idx, CFG))
        amts = [a for k in cands if k in kstats for a in kstats[k].amounts]
        has_exact = rec.amount_minor in amts if amts else False
        min_delta = min((abs(rec.amount_minor - a) for a in amts), default=1e12)
        X.append(featurise_multiplicity(rec, n_candidates=len(cands),
                                        has_exact=has_exact,
                                        min_delta_minor=float(min_delta),
                                        prior=prior))
        y.append(int(ds.is_mult[i]))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8)


def pr_curve(y, p):
    """Precision/recall at every threshold, plus average precision."""
    order = np.argsort(-p)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(y.sum(), 1)
    ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    return precision, recall, ap, p[order]


def roc_auc(y, p):
    """Rank-based AUC; ties averaged."""
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    pos, neg = y.sum(), len(y) - y.sum()
    if pos == 0 or neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def at_threshold(y, p, t):
    pred = p >= t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/benchrec_train.csv"
    t0 = time.perf_counter()
    ds = load_benchrec(pd.read_csv(path, dtype=str, low_memory=False))
    idx = KeyIndex(ds.key_rows)
    kstats = build_key_stats(ds.key_rows)
    # models.pkl is produced by scripts/export_artifacts.py in this repo and is
    # not user-supplied; the deployed API loads the same file the same way.
    bundle = pickle.loads((ROOT / "artifacts" / "models.pkl").read_bytes())
    ranker, detector, prior = bundle["ranker"], bundle["detector"], bundle["prior"]
    print(f"loaded {ds.n_records:,} records in {time.perf_counter() - t0:.0f}s\n")

    days = [r.day for r in ds.records]
    sp = temporal_split(days, ds.match_ids, train_frac=0.7, val_frac=0.1)
    splits = [("train", sp.train), ("val", sp.val), ("test", sp.test)]

    # ---- ranker: the overfitting check is train vs test on the same metric --
    print("RANKER — choose the right key among blocked candidates")
    print(f"  {'split':<7}{'records':>10}{'top-1':>9}{'MRR':>8}")
    scores = {}
    for name, ids in splits:
        # Same negative cap as training, so the three rows are comparable.
        X, groups = ranker_pairs(ds, idx, kstats, ids, max_neg=24)
        s = ranker.score(X)
        scores[name] = (top1(s, groups), mrr(s, groups), len(groups))
        print(f"  {name:<7}{len(groups):>10,}{scores[name][0] * 100:>8.2f}%{scores[name][1]:>8.3f}")
    gap = (scores["train"][0] - scores["test"][0]) * 100
    print(f"\n  train - test gap: {gap:+.2f} pp", end="  ")
    print("(a large positive gap is memorisation)")

    # ---- multiplicity detector: the metrics that were missing ---------------
    print("\n\nMULTIPLICITY DETECTOR — is this one payment covering several keys?")
    per_split = {}
    for name, ids in splits:
        X, y = mult_rows(ds, idx, kstats, ids, prior)
        p = detector.predict_proba(X)
        _, _, ap, _ = pr_curve(y, p)
        per_split[name] = (y, p, ap)
        base = y.mean()
        print(f"\n  {name}  n={len(y):,}  positives={y.sum():,} ({base * 100:.1f}%)"
              f"  PR-AUC={ap:.3f}  ROC-AUC={roc_auc(y, p):.3f}")
        for t in (0.3, 0.5, 0.7):
            m = at_threshold(y, p, t)
            print(f"    t={t:.1f}  precision {m['precision'] * 100:5.1f}%  "
                  f"recall {m['recall'] * 100:5.1f}%  F1 {m['f1']:.3f}   "
                  f"tp={m['tp']:<5} fp={m['fp']:<5} fn={m['fn']:<5} tn={m['tn']}")
    tr_ap, te_ap = per_split["train"][2], per_split["test"][2]
    print(f"\n  train - test PR-AUC gap: {tr_ap - te_ap:+.3f}")

    print("\n  confusion matrix at the shipped threshold (0.5), test split:")
    y, p, _ = per_split["test"]
    m = at_threshold(y, p, 0.5)
    print(f"                  predicted grouped   predicted single")
    print(f"    actually grouped   {m['tp']:>10,}        {m['fn']:>10,}")
    print(f"    actually single    {m['fp']:>10,}        {m['tn']:>10,}")


if __name__ == "__main__":
    main()
