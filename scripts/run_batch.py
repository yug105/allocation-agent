"""End-to-end run on the held-out split, with the numbers that go in the README.

**This runs the shipped bundle.** It used to train a ranker inline and pass no
calibrator, so it measured `sigmoid(margin)` -- a configuration the service does
not run -- and the README's headline figures came from it. Calibration costs
straight-through, so the number it produced was the flattering one. It also
crashed: `mult_features=` was removed from `pipeline.run_batch` when the
multiplicity check moved inside `match_one`, and nothing had regenerated these
figures since, which is how they went stale without anyone noticing.

`artifacts/models.pkl` is trained by `export_artifacts.py` on `sp.train` of this
same split, so scoring it on `sp.test` is honest, and it is the artifact that
actually serves traffic.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from allocation_agent.adapters.benchrec import load_benchrec
from allocation_agent.decide.gate import GateConfig
from allocation_agent.eval.splits import temporal_split
from allocation_agent.match.blocker import BlockingConfig
from allocation_agent.match.features import build_key_stats
from allocation_agent.pipeline import run_batch
from allocation_agent.report.audit import AuditLog, RunConfig
from allocation_agent.stores.keys import KeyIndex

ROOT = Path(__file__).resolve().parents[1]
BCFG = BlockingConfig(date_slack_days=7)
GCFG = GateConfig(base=0.85, slope=0.02, cap=0.995, policy_version="v0.1")

t0 = time.perf_counter()
raw = pd.read_csv("data/benchrec_train.csv", dtype=str, low_memory=False)
ds = load_benchrec(raw)
idx = KeyIndex(ds.key_rows); kstats = build_key_stats(ds.key_rows)
sp = temporal_split([r.day for r in ds.records], ds.match_ids, train_frac=0.7, val_frac=0.1)
print(f"loaded {ds.n_records:,} records / {idx.n_keys:,} keys in {time.perf_counter()-t0:.1f}s")
print(f"split: {sp}\n")

# This project's own committed build artifact, produced by export_artifacts.py
# and loaded the same way by api.py. Not third-party input.
bundle = pickle.loads((ROOT / "artifacts" / "models.pkl").read_bytes())
print(f"shipped bundle: calibrator={bundle['calibrator_kind']}, "
      f"{len(bundle['feature_names'])} features\n")

# ---- run the batch ---------------------------------------------------------
test_records = [ds.records[i] for i in sp.test]
audit = AuditLog("data/audit.db")
res = run_batch(test_records, idx, kstats, audit,
                run_config=RunConfig(approved_by="yug",
                                     blocking={"date_slack_days": 7},
                                     gate={"base": 0.85, "slope": 0.02},
                                     policy_version="v0.1",
                                     notes="held-out temporal split, shipped bundle"),
                blocking=BCFG, gate=GCFG,
                ranker=bundle["ranker"], multiplicity=bundle["detector"],
                prior=bundle["prior"], calibrator=bundle["calibrator"],
                calibrator_kind=bundle["calibrator_kind"],
                # BenchRec is the population the calibrator was fitted on.
                calibrated_for_this_data=True,
                mult_threshold=0.7)
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
print(f"{'precision of auto-posted matches':38} {post_ok/max(post_n,1)*100:7.2f}%   ({post_ok:,}/{post_n:,})")
print(f"{'wrong auto-posts':38} {post_n-post_ok:7,}")
print(f"{'grouped records routed to review':38} {mult_caught:,}/{mult_total:,} "
      f"({mult_caught/max(mult_total,1)*100:.1f}%)")
print(f"{'single records wrongly routed':38} {mult_wrong:,}")
print(f"{'throughput':38} {res.records_per_second:7,.0f} rec/sec")
# No narrator is constructed on this path, so this is structural, not counted.
print(f"{'LLM on the matching path':38} {'none constructed':>7}")
print("=" * 62)

# ---- against the engine whose gap this fills -------------------------------
#
# The comparison the README was missing. BenchRec's labels *are* the incumbent
# system's resolutions, so `matchRule == MANUAL` marks every reconciliation a
# person had to touch. That makes the baseline auto-resolution rate computable
# on exactly this population -- and it is the first thing anyone buying a
# reconciliation tool asks. Reporting the vendor comparison and omitting this
# one was picking the flattering baseline.
rule = raw.groupby("matchId")["matchRule"].agg(
    lambda s: "MANUAL" if (s == "MANUAL").any() else s.iloc[0])
mids = np.asarray(ds.match_ids, dtype=object)
incumbent_auto = np.array([rule.get(m, "MANUAL") != "MANUAL" for m in mids[sp.test]])

print("\nagainst the rules engine that produced these labels:")
print(f"  {'incumbent auto-resolved':34} {incumbent_auto.mean()*100:7.2f}%")
print(f"  {'this system posted':34} {res.straight_through_rate*100:7.2f}%")
print(f"  {'difference':34} {(res.straight_through_rate-incumbent_auto.mean())*100:+7.2f} pp")

# Split by what the incumbent already managed. The aggregate hides that almost
# every wrong auto-post comes from the population this project exists for.
print(f"\n{'population':44} {'n':>7} {'posted':>9} {'correct':>9}")
for label, mask in (("incumbent auto-resolved it", incumbent_auto),
                    ("incumbent sent it to a human (the gap)", ~incumbent_auto)):
    ids = [ds.records[i].record_id for i, m in zip(sp.test, mask, strict=True) if m]
    sub = [rows[r] for r in ids if r in rows]
    posted = [d for d in sub if d["outcome"] == "post" and d["path"] != "multiplicity"]
    ok = sum(not truth[d["record_id"]][1]
             and json.loads(d["chosen_keys"])[:1] == [truth[d["record_id"]][0]]
             for d in posted)
    rate = len(posted) / len(sub) * 100 if sub else 0.0
    prec = ok / len(posted) * 100 if posted else 0.0
    print(f"  {label:42} {len(sub):7,} {rate:8.2f}% {prec:8.2f}%")

print("\nexception taxonomy:")
for k, v in res.exceptions.most_common():
    print(f"  {k:26} {v:7,}  ({v/res.n_records*100:5.1f}%)")
print(f"\naudit rows written: {len(rows):,}  (one per record, append-only)")
