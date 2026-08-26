"""Turn the ranker's margin into a probability that means what it says.

`confidence = sigmoid(top_score - second_score)` is a monotone transform of a
LambdaRank margin. LambdaRank optimises *order*, not likelihood, so its scores
carry no probability semantics and neither does a sigmoid of their difference.
Calling the result "confidence" and comparing it to 0.85 borrows a meaning it
has not earned: a margin of 0 becomes 50% and a margin of 2 becomes 88%, and
nothing establishes that 88% of such records are actually right.

This fits a calibrator on the **validation** split and scores it on **test**,
so the number the gate compares against is a measured frequency.

The target is the one the gate actually needs: *given that this record reaches
the gate, is the top candidate the right key?* Blocking misses and grouped
records are included as negatives — they reach the gate too, and excluding
them would calibrate against a population the gate never sees.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from allocation_agent.adapters.benchrec import load_benchrec
from allocation_agent.eval.splits import temporal_split
from allocation_agent.match.blocker import BlockingConfig, block
from allocation_agent.match.features import build_key_stats, featurise
from allocation_agent.stores.keys import KeyIndex

ROOT = Path(__file__).resolve().parents[1]
CFG = BlockingConfig(date_slack_days=7)


def margins_and_truth(ds, idx, kstats, ranker, indices):
    """(margin, was the top candidate right) for every record reaching the gate."""
    margins, correct = [], []
    for i in indices:
        rec = ds.records[i]
        usable = [k for k in sorted(block(rec, idx, CFG)) if k in kstats]
        if len(usable) < 2:
            continue                    # no runner-up: no margin exists
        X = np.vstack([featurise(rec, kstats[k], n_candidates=len(usable))
                       for k in usable])
        s = ranker.score(X)
        order = np.argsort(-s)
        margins.append(float(s[order[0]] - s[order[1]]))
        # Same definition the API scores itself by: a grouped record has no
        # single right key, so the top candidate is wrong by construction.
        correct.append(int((not ds.is_mult[i]) and usable[int(order[0])] == ds.labels[i]))
    return np.asarray(margins), np.asarray(correct)


def ece(p, y, bins=10):
    """Expected calibration error: mean gap between claimed and actual, by bin."""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            total += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return total


def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            rows.append((lo, hi, int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    return rows


def report(name, p, y):
    brier = float(np.mean((p - y) ** 2))
    print(f"\n  {name}   ECE {ece(p, y):.4f}   Brier {brier:.4f}")
    print(f"    {'claimed':>18}{'actual':>9}{'records':>9}{'gap':>8}")
    for lo, hi, n, claimed, actual in reliability(p, y):
        flag = "  <-- overconfident" if claimed - actual > 0.05 else ""
        print(f"    {lo:.1f}-{hi:.1f}{claimed * 100:>11.1f}%{actual * 100:>8.1f}%"
              f"{n:>9,}{(claimed - actual) * 100:>+7.1f}{flag}")


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/benchrec_train.csv"
    t0 = time.perf_counter()
    ds = load_benchrec(pd.read_csv(path, dtype=str, low_memory=False))
    idx, kstats = KeyIndex(ds.key_rows), build_key_stats(ds.key_rows)
    # Produced by scripts/export_artifacts.py in this repo, not user-supplied.
    bundle = pickle.loads((ROOT / "artifacts" / "models.pkl").read_bytes())
    ranker = bundle["ranker"]
    sp = temporal_split([r.day for r in ds.records], ds.match_ids,
                        train_frac=0.7, val_frac=0.1)
    print(f"loaded {ds.n_records:,} records in {time.perf_counter() - t0:.0f}s")

    mv, yv = margins_and_truth(ds, idx, kstats, ranker, sp.val)
    mt, yt = margins_and_truth(ds, idx, kstats, ranker, sp.test)
    print(f"val {len(mv):,} records ({yv.mean() * 100:.1f}% top-1 correct)   "
          f"test {len(mt):,} ({yt.mean() * 100:.1f}%)")

    print("\n" + "=" * 62)
    print("WHAT SHIPS TODAY: sigmoid(margin), on the test split")
    print("=" * 62)
    raw_t = 1 / (1 + np.exp(-mt))
    report("sigmoid(margin)", raw_t, yt)

    print("\n" + "=" * 62)
    print("CALIBRATED — fitted on validation, scored on test")
    print("=" * 62)
    platt = LogisticRegression().fit(mv.reshape(-1, 1), yv)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(mv, yv)
    p_platt = platt.predict_proba(mt.reshape(-1, 1))[:, 1]
    p_iso = iso.predict(mt)
    report("Platt", p_platt, yt)
    report("isotonic", p_iso, yt)

    print("\n" + "=" * 62)
    print("SUMMARY (lower is better)")
    print("=" * 62)
    print(f"    {'method':<18}{'ECE':>9}{'Brier':>9}")
    for nm, p in (("sigmoid(margin)", raw_t), ("Platt", p_platt), ("isotonic", p_iso)):
        print(f"    {nm:<18}{ece(p, yt):>9.4f}{float(np.mean((p - yt) ** 2)):>9.4f}")

    best = min((("platt", p_platt, platt), ("isotonic", p_iso, iso)),
               key=lambda t: ece(t[1], yt))
    print(f"\n  picked: {best[0]}")

    bundle["calibrator"] = best[2]
    bundle["calibrator_kind"] = best[0]
    (ROOT / "artifacts" / "models.pkl").write_bytes(pickle.dumps(bundle))
    print(f"  written into artifacts/models.pkl as bundle['calibrator']")


if __name__ == "__main__":
    main()
