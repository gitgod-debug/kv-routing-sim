"""Is the recommended configuration at the edge of what the fleet can carry?

A replica count is read off a latency curve, and near saturation that curve is steep: a small
error in the timing model becomes a large error in p95. Two measurements say how large.

  * C3 (2026-10-05): two runs of `kvsim calibrate` on one engine, 25 minutes apart, differed by
    3–5% in their coefficients — and that put one p95 4.2× off.
  * T1 (2026-10-05): on the two held-out runs at the knee of a T4's load curve the simulator's
    per-token latency was 1.54× and 1.71× the engine's; on the three off it, within 1.07×.

So a count printed without saying how close to the edge it sits claims more than it can support.

THE CHECK. Run the chosen configuration once more with every timing coefficient `SLOWDOWN` (5%)
larger — what a second calibration of the same GPU could give — and compare:

  knee   a p95 rises by more than `KNEE_FACTOR` (1.5×). Queueing has taken over: 5% in, 50% or
         more out. No latency the simulator prints at this point can be trusted.
  edge   the configuration no longer meets the SLO. The count depends on a difference that two
         calibrations of one GPU have between them: allow one replica more.
  clear  neither. With room to spare a 5% slowdown is a 5% slowdown.

⚠️ WHERE THE 1.5 COMES FROM — AND WHAT IT IS NOT. It was chosen after looking (post hoc), on the
five held-out runs of T1 replayed on one replica with profile A: the 5% slowdown multiplies p95
TTFT by 5.3 and 5.5 on the two runs where the model was 1.5–1.7× off, and by 1.13–1.14 on the
three where it was within 1.07× (p95 time per token: 1.13–1.21 on all five). Any threshold
between 1.21 and 5.33 separates them; 1.5 is a round number in that gap. That is five runs of
three workloads on one GPU: an indication that the check sees the knee, not a validated threshold
(scripts/edge_posthoc_t1.py → artifacts/results/t1_itl/posthoc_edge.md).

The check costs one extra run per seed, of one configuration.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from ..sim.cost_model import TIMING_FIELDS, CostProfile
from ..traces.schema import NormalizedRequest
from .cost import CostModel
from .pareto import Point
from .sweep import sweep

SLOWDOWN = 1.05     # C3: two calibrations of one engine differ by 3–5% in their coefficients
KNEE_FACTOR = 1.5   # see the module docstring: chosen post hoc, on five runs


def slowed(profile: CostProfile, factor: float = SLOWDOWN) -> CostProfile:
    """The profile with every timing coefficient multiplied by `factor`. Capacity, block size and
    the calibrated flag are untouched: it is the same engine, measured a little slower."""
    return replace(profile, name=f"{profile.name}×{factor:g}",
                   **{k: getattr(profile, k) * factor for k in TIMING_FIELDS})


def _factor(base: float, after: float) -> float | None:
    if math.isnan(base) or math.isnan(after) or base <= 0:
        return None  # no multi-token request, or nothing to compare
    return after / base


def edge_check(
    trace: Sequence[NormalizedRequest],
    *,
    chosen: Point,
    seeds: Sequence[int],
    profile: CostProfile,
    cost: CostModel,
    trace_info: dict[str, Any] | None = None,
    slowdown: float = SLOWDOWN,
    knee_factor: float = KNEE_FACTOR,
) -> dict[str, Any]:
    """`chosen` (a point of the sweep that meets the SLO) again, on timings `slowdown` slower.

    Returns the two p95s before and after, their factors, whether the SLO still holds, and the
    verdict: "knee", "edge" or "clear" (module docstring) — or "not_met" for a point that did not
    meet the SLO before the slowdown either and is not at the knee.
    """
    points, _ = sweep(trace, policies=[chosen.policy], replica_counts=[chosen.n_replicas],
                      seeds=seeds, profile=slowed(profile, slowdown), cost=cost,
                      trace_info=trace_info)
    after = points[0]
    ttft = _factor(chosen.p95_ttft_ms, after.p95_ttft_ms)
    tpot = _factor(chosen.p95_tpot_ms, after.p95_tpot_ms)
    worst = max(f for f in (ttft, tpot, 0.0) if f is not None)
    if worst > knee_factor:
        verdict = "knee"
    elif not chosen.meets_slo:
        verdict = "not_met"  # it did not meet the SLO to begin with: there is no edge to be at
    elif not after.meets_slo:
        verdict = "edge"
    else:
        verdict = "clear"

    def r(x: float) -> float | None:
        return None if math.isnan(x) else round(x, 3)

    return {
        "policy": chosen.policy, "n_replicas": chosen.n_replicas, "slowdown": slowdown,
        "knee_factor": knee_factor,
        "p95_ttft_ms": [r(chosen.p95_ttft_ms), r(after.p95_ttft_ms)],
        "p95_tpot_ms": [r(chosen.p95_tpot_ms), r(after.p95_tpot_ms)],
        "ttft_factor": None if ttft is None else round(ttft, 3),
        "tpot_factor": None if tpot is None else round(tpot, 3),
        "still_meets_slo": bool(after.meets_slo),
        "verdict": verdict,
    }
