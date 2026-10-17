"""Whether a recommended configuration sits at the edge of what the fleet can carry.

Near saturation the latency curve is steep, so a small error in the timing model becomes a large
error in p95. The check reruns the chosen configuration with every timing coefficient `SLOWDOWN`
larger, the difference between two calibrations of the same GPU, and classifies the result:

    knee    a p95 grows by more than `KNEE_FACTOR`: queueing dominates, and no latency the
            simulator reports at this point can be trusted.
    edge    the configuration stops meeting the SLO: allow one replica more.
    clear   neither.

`KNEE_FACTOR` was chosen after inspecting five held-out live runs: it separates the two runs the
timing model got wrong from the three it got right (any value between 1.21 and 5.33 would), so it
is an indication rather than a validated threshold (scripts/edge_posthoc_t1.py).
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
from .sweep import RunCallback, sweep

SLOWDOWN = 1.05
KNEE_FACTOR = 1.5


def slowed(profile: CostProfile, factor: float = SLOWDOWN) -> CostProfile:
    """Return the profile with every timing coefficient multiplied by `factor`.

    Capacity, block size and the calibrated flag are unchanged: it models the same engine,
    measured slightly slower.
    """
    return replace(profile, name=f"{profile.name}×{factor:g}",
                   **{k: getattr(profile, k) * factor for k in TIMING_FIELDS})


def _factor(base: float, after: float) -> float | None:
    if math.isnan(base) or math.isnan(after) or base <= 0:
        return None
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
    on_run: RunCallback | None = None,
) -> dict[str, Any]:
    """Rerun the chosen configuration on timings `slowdown` slower and classify the change.

    Returns:
        The p95 first-token and per-token latencies before and after, their ratios, whether the
        SLO still holds, and a verdict: "knee", "edge", "clear", or "not_met" for a configuration
        that did not meet the SLO to begin with and is not at the knee.
    """
    points, _ = sweep(trace, policies=[chosen.policy], replica_counts=[chosen.n_replicas],
                      seeds=seeds, profile=slowed(profile, slowdown), cost=cost,
                      trace_info=trace_info, on_run=on_run)
    after = points[0]
    ttft = _factor(chosen.p95_ttft_ms, after.p95_ttft_ms)
    tpot = _factor(chosen.p95_tpot_ms, after.p95_tpot_ms)
    worst = max(f for f in (ttft, tpot, 0.0) if f is not None)
    if worst > knee_factor:
        verdict = "knee"
    elif not chosen.meets_slo:
        verdict = "not_met"
    elif not after.meets_slo:
        verdict = "edge"
    else:
        verdict = "clear"

    def rounded(x: float) -> float | None:
        return None if math.isnan(x) else round(x, 3)

    return {
        "policy": chosen.policy, "n_replicas": chosen.n_replicas, "slowdown": slowdown,
        "knee_factor": knee_factor,
        "p95_ttft_ms": [rounded(chosen.p95_ttft_ms), rounded(after.p95_ttft_ms)],
        "p95_tpot_ms": [rounded(chosen.p95_tpot_ms), rounded(after.p95_tpot_ms)],
        "ttft_factor": None if ttft is None else round(ttft, 3),
        "tpot_factor": None if tpot is None else round(tpot, 3),
        "still_meets_slo": bool(after.meets_slo),
        "verdict": verdict,
    }
