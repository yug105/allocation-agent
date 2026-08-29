"""Everything about how the models were trained, in one artifact.

`model_diagnostics.py` and `calibrate_ranker.py` measure the things a reader
should want — the train/test gap, recall and F1 at the operating point, how far
the confidence was from a real probability before calibration. All of it needs
the 60 MB training CSV, which the deployed image does not carry, so none of it
ever reached the page. The evidence for the numbers lived only in a terminal.

This computes it locally and writes `artifacts/training.json`, which the service
serves and the Trust screen renders. Nothing here is hand-copied: every figure
is produced by the same code that reports it on the command line.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibrate_ranker import ece, margins_and_truth  # noqa: E402
from model_diagnostics import mult_rows, ranker_pairs, top1  # noqa: E402

from allocation_agent.adapters.benchrec import load_benchrec  # noqa: E402
from allocation_agent.eval.splits import temporal_split  # noqa: E402
from allocation_agent.match.blocker import BlockingConfig  # noqa: E402
from allocation_agent.match.features import build_key_stats  # noqa: E402
from allocation_agent.stores.keys import KeyIndex  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CFG = BlockingConfig(date_slack_days=7)


def pr_auc(y, p):
    order = np.argsort(-p)
    y = y[order]
    tp, fp = np.cumsum(y), np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(y.sum(), 1)
    return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))


def at(y, p, t):
    pred = p >= t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
            "false_alarms": fp, "missed": fn}


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/benchrec_train.csv"
    t0 = time.perf_counter()
    ds = load_benchrec(pd.read_csv(path, dtype=str, low_memory=False))
    idx, kstats = KeyIndex(ds.key_rows), build_key_stats(ds.key_rows)
    # Produced by scripts/export_artifacts.py in this repo, not user-supplied.
    bundle = pickle.loads((ROOT / "artifacts" / "models.pkl").read_bytes())
    ranker, detector, prior = bundle["ranker"], bundle["detector"], bundle["prior"]
    sp = temporal_split([r.day for r in ds.records], ds.match_ids,
                        train_frac=0.7, val_frac=0.1)

    out: dict = {
        "n_records": ds.n_records,
        "n_keys": idx.n_keys,
        "split": {"train": len(sp.train), "val": len(sp.val), "test": len(sp.test)},
        "features": list(bundle.get("feature_names", [])),
        "calibrator": bundle.get("calibrator_kind", "none"),
    }

    # ranker: the train/test gap is the overfitting question
    ranker_scores = {}
    for name, ids in (("train", sp.train), ("val", sp.val), ("test", sp.test)):
        X, groups = ranker_pairs(ds, idx, kstats, ids, max_neg=24)
        ranker_scores[name] = round(top1(ranker.score(X), groups), 4)
    out["ranker_top1"] = ranker_scores
    out["ranker_gap"] = round(ranker_scores["train"] - ranker_scores["test"], 4)

    # detector: the metrics at the threshold that actually runs
    detector_stats = {}
    for name, ids in (("train", sp.train), ("test", sp.test)):
        X, y = mult_rows(ds, idx, kstats, ids, prior)
        p = detector.predict_proba(X)
        detector_stats[name] = {"pr_auc": round(pr_auc(y, p), 4),
                                "positive_rate": round(float(y.mean()), 4),
                                "at_0.7": at(y, p, 0.7)}
    out["detector"] = detector_stats
    out["detector_gap"] = round(detector_stats["train"]["pr_auc"]
                                - detector_stats["test"]["pr_auc"], 4)

    # calibration: what the confidence claimed against what happened
    mv, yv = margins_and_truth(ds, idx, kstats, ranker, sp.val)
    mt, yt = margins_and_truth(ds, idx, kstats, ranker, sp.test)
    raw = 1 / (1 + np.exp(-mt))
    cal = bundle.get("calibrator")
    out["calibration"] = {
        "ece_before": round(ece(raw, yt), 4),
        "ece_after": round(ece(np.clip(cal.predict(mt), 0, 1), yt), 4) if cal else None,
        "n_val": len(mv), "n_test": len(mt),
    }

    # what the model leans on
    out["importances"] = {k: round(v, 4) for k, v in
                          sorted(ranker.importances.items(), key=lambda kv: -kv[1])}

    out["seconds"] = round(time.perf_counter() - t0, 1)
    (ROOT / "artifacts" / "training.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"{out['n_records']:,} records, {out['n_keys']:,} keys, {out['seconds']}s\n")
    print(f"  ranker top-1   train {ranker_scores['train']:.1%}  "
          f"val {ranker_scores['val']:.1%}  test {ranker_scores['test']:.1%}"
          f"   gap {out['ranker_gap']:+.2%}")
    print(f"  detector PR-AUC train {detector_stats['train']['pr_auc']:.3f}  "
          f"test {detector_stats['test']['pr_auc']:.3f}   gap {out['detector_gap']:+.3f}")
    d = detector_stats["test"]["at_0.7"]
    print(f"  detector at 0.7  precision {d['precision']:.1%}  recall {d['recall']:.1%}  "
          f"F1 {d['f1']:.3f}")
    print(f"  calibration ECE  {out['calibration']['ece_before']:.4f} -> "
          f"{out['calibration']['ece_after']:.4f}")
    print(f"\n  top features: "
          + ", ".join(list(out['importances'])[:4]))
    print(f"\nwrote artifacts/training.json")


if __name__ == "__main__":
    main()
