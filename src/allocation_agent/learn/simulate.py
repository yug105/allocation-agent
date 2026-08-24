"""Learning simulation, with the controls that stop it fooling us.

Records arrive in batches. The agent decides; whatever the gate refuses goes to
a "reviewer" who reveals the true answer. Corrections are attributed to the
stage that caused them, and the ones that are model problems become labelled
examples for the next refit.

**Autonomy will rise whatever we do**, which is why the controls exist:

* **C-1 learning off** -- the same sequence with no refitting. The gap between
  the curves is the learning; without this, a rising line only shows that later
  batches were easier.
* **C-2 shuffled order** -- several orderings. Improvement in all of them is
  real; improvement in one is luck.
* **C-3 placebo** -- feed *random* corrections instead of true ones. If autonomy
  still rises, the measurement is broken, and we would rather find that out for
  the price of one run.

**The starting point matters.** A model already fitted on 134,000 records will
not visibly improve from another two thousand. A real deployment starts cold, so
the simulation does too: a deliberately small initial fit, then learning from
what the reviewer says. That is the honest scenario, and also the only one where
the question "does the loop work" has a measurable answer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from allocation_agent.learn.router import FailureLocus, diagnose


@dataclass
class BatchOutcome:
    index: int
    n_records: int
    autonomous: int
    corrections: int
    posted_correct: int = 0
    loci: dict[str, int] = field(default_factory=dict)

    @property
    def autonomy(self) -> float:
        """Share posted without a human. **Gameable on its own** -- see precision."""
        return self.autonomous / self.n_records if self.n_records else 0.0

    @property
    def precision(self) -> float:
        """Share of posted decisions that were right.

        Autonomy measures decisiveness, not correctness: a model trained on easy,
        well-separated examples produces wide margins and posts more whether or
        not it is right. Rising autonomy with falling precision is a regression.
        """
        return self.posted_correct / self.autonomous if self.autonomous else 0.0


@dataclass
class SimulationResult:
    label: str
    batches: list[BatchOutcome] = field(default_factory=list)

    @property
    def curve(self) -> list[float]:
        return [b.autonomy for b in self.batches]

    @property
    def first(self) -> float:
        return self.batches[0].autonomy if self.batches else 0.0

    @property
    def last(self) -> float:
        return self.batches[-1].autonomy if self.batches else 0.0

    @property
    def improvement(self) -> float:
        """Percentage points gained from the first batch to the last."""
        return (self.last - self.first) * 100

    @property
    def precision_curve(self) -> list[float]:
        return [b.precision for b in self.batches]

    @property
    def precision_change(self) -> float:
        if not self.batches:
            return 0.0
        return (self.batches[-1].precision - self.batches[0].precision) * 100

    def __str__(self) -> str:
        a = "  ".join(f"{x * 100:5.1f}" for x in self.curve)
        p = "  ".join(f"{x * 100:5.1f}" for x in self.precision_curve)
        return (f"{self.label:22} autonomy  {a}   ({self.improvement:+5.2f} pp)\n"
                f"{'':22} precision {p}   ({self.precision_change:+5.2f} pp)")


def simulate(
    *,
    label: str,
    indices: Sequence[int],
    decide_batch: Callable[[Sequence[int]], list[dict]],
    refit: Callable[[list[tuple[int, str]]], None] | None,
    truth: Callable[[int], tuple[list[str], bool]],
    batch_size: int = 4_000,
    placebo: bool = False,
    spot_check_rate: float = 0.0,
    rng: np.random.Generator | None = None,
) -> SimulationResult:
    """Run one arm of the experiment.

    Args:
        decide_batch: runs the pipeline over row indices, returning one dict per
            record with ``posted``, ``candidates``, ``ranked``, ``routed_multiple``.
        refit: consumes ``(row_index, correct_key)`` pairs. ``None`` disables
            learning entirely, which is control C-1.
        truth: the reviewer. Returns the correct keys and whether the record is
            genuinely grouped.
        placebo: corrupt every correction before feeding it back (control C-3).
            Autonomy must *not* improve.
        spot_check_rate: fraction of *auto-posted* records also sent for review.

            Without this the loop only ever learns from records the gate
            refused, which are the ambiguous ones by construction. The training
            set drifts toward the hard cases, the model separates them less
            sharply, margins compress, confidence falls, and fewer records clear
            the gate -- measured at -3.15 pp against not learning at all.

            Sampling posted records restores the easy cases to the distribution.
            A finance function already does this and calls it spot-checking.
    """
    rng = rng or np.random.default_rng(0)
    result = SimulationResult(label=label)
    all_keys: list[str] = []

    for b, start in enumerate(range(0, len(indices), batch_size)):
        chunk = list(indices[start : start + batch_size])
        if not chunk:
            break

        decisions = decide_batch(chunk)
        autonomous = sum(1 for d in decisions if d["posted"])

        corrections: list[tuple[int, str]] = []
        loci: dict[str, int] = {}
        posted_correct = 0

        for row, d in zip(chunk, decisions, strict=False):
            correct_keys, truly_multiple = truth(row)
            all_keys.extend(correct_keys)

            dg = diagnose(
                correct_keys=correct_keys,
                candidates=d["candidates"],
                ranked_keys=d["ranked"],
                posted=d["posted"],
                routed_multiple=d["routed_multiple"],
                truly_multiple=truly_multiple,
            )
            if dg.locus is FailureLocus.NONE and d["posted"]:
                posted_correct += 1
            if dg.locus is FailureLocus.NONE:
                # Correct and posted. Learn from it only if this record was
                # spot-checked, mirroring what a reviewer would actually see.
                if (spot_check_rate > 0 and d["posted"] and correct_keys
                        and rng.random() < spot_check_rate):
                    key = correct_keys[0]
                    if placebo and all_keys:
                        # The control must corrupt *everything* fed back. An
                        # earlier version corrupted only the corrections, so the
                        # placebo arm still received genuine easy examples -- and
                        # improved, which made the control useless.
                        key = str(rng.choice(all_keys))
                    corrections.append((row, key))
                continue
            loci[dg.locus.value] = loci.get(dg.locus.value, 0) + 1

            # Only ranking and multiplicity failures are model problems.
            # Blocking and threshold failures are settings, handled by rule proposal.
            if dg.locus in (FailureLocus.RANKING, FailureLocus.MULTIPLICITY) and correct_keys:
                key = correct_keys[0]
                if placebo and all_keys:
                    key = str(rng.choice(all_keys))  # a plausible but wrong answer
                corrections.append((row, key))

        result.batches.append(
            BatchOutcome(index=b, n_records=len(chunk), autonomous=autonomous,
                         corrections=len(corrections), posted_correct=posted_correct,
                         loci=loci)
        )

        if refit is not None and corrections:
            refit(corrections)

    return result
