"""Grid sweeps: every (policy, replica count) cell over several seeds, averaged into points.

Several seeds per cell matter because routing policies break ties at random, and the differences
between policies can be as small as the variation between seeds (docs/06-methodology.md).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..sim.cost_model import UNCALIBRATED_DEFAULT, CostProfile
from ..sim.engine import run
from ..sim.metrics import RunResult
from ..traces.generator import AgenticWorkload
from ..traces.schema import NormalizedRequest
from .cost import CostModel
from .pareto import Point, to_point

RunCallback = Callable[[], None]
"""Called once after each simulation run finishes; used to report progress."""


def sweep(
    trace: Sequence[NormalizedRequest],
    *,
    policies: Sequence[str],
    replica_counts: Sequence[int],
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    profile: CostProfile = UNCALIBRATED_DEFAULT,
    cost: CostModel | None = None,
    trace_info: dict[str, Any] | None = None,
    on_run: RunCallback | None = None,
) -> tuple[list[Point], list[RunResult]]:
    """Run every (policy, replica count, seed) combination.

    Returns:
        One point per (policy, replica count) cell, averaged over seeds, and every raw result.
    """
    results: list[RunResult] = []
    points: list[Point] = []
    for policy in policies:
        for n in replica_counts:
            cell = []
            for seed in seeds:
                cell.append(run(trace, policy=policy, n_replicas=n, seed=seed, profile=profile,
                                trace_info=trace_info))
                if on_run is not None:
                    on_run()
            results.extend(cell)
            points.append(average_points([to_point(r, cost) for r in cell]))
    return points, results


def average_points(pts: list[Point]) -> Point:
    """Average one cell's per-seed points into a single point.

    Percentiles are averaged across seeds rather than pooled: the result is the typical p95 of a
    run, not the p95 of a mixture of runs, which would understate run-to-run variation.
    """
    n = len(pts)
    finite = [p.inr_per_1k for p in pts if p.inr_per_1k != float("inf")]
    return Point(
        policy=pts[0].policy,
        n_replicas=pts[0].n_replicas,
        hit_rate=sum(p.hit_rate for p in pts) / n,
        p95_ttft_ms=sum(p.p95_ttft_ms for p in pts) / n,
        p50_ttft_ms=sum(p.p50_ttft_ms for p in pts) / n,
        inr_per_1k=(sum(finite) / len(finite)) if finite else float("inf"),
        throughput_rps=sum(p.throughput_rps for p in pts) / n,
        load_imbalance=sum(p.load_imbalance for p in pts) / n,
        # A cell meets the SLO only if every seed does; None means no SLO was defined.
        meets_slo=(
            None if any(p.meets_slo is None for p in pts) else all(p.meets_slo for p in pts)
        ),
        calibrated=pts[0].calibrated,
        p95_tpot_ms=sum(p.p95_tpot_ms for p in pts) / n,
    )


def crossover_sweep(
    *,
    policies: Sequence[str],
    sharing_ratios: Sequence[float],
    arrival_rates: Sequence[float],
    n_replicas: int = 4,
    seeds: Sequence[int] = (0, 1, 2),
    n_sessions: int = 150,
    profile: CostProfile = UNCALIBRATED_DEFAULT,
) -> list[dict[str, Any]]:
    """Find the policy with the lowest p95 TTFT for each (prefix sharing, arrival rate) pair.

    The sharing axis reports the cross-session sharing the generator achieved, because reuse
    within a session is captured by session-sticky routing anyway and would hide the effect.
    """
    rows: list[dict[str, Any]] = []
    for ratio in sharing_ratios:
        for rate in arrival_rates:
            trace, report = AgenticWorkload(
                n_sessions=n_sessions,
                prefix_sharing_ratio=ratio,
                arrival_rate=rate,
                seed=0,
            ).generate()
            cell: dict[str, float] = {}
            for policy in policies:
                p95s = [
                    run(
                        trace, policy=policy, n_replicas=n_replicas, seed=s, profile=profile
                    ).ttft_percentiles()["p95_ms"]
                    for s in seeds
                ]
                cell[policy] = sum(p95s) / len(p95s)
            winner = min(cell, key=lambda k: cell[k])
            rows.append(
                {
                    "requested_sharing": ratio,
                    "cross_session_sharing": report["achieved_cross_session_sharing"],
                    "arrival_rate": rate,
                    "n_replicas": n_replicas,
                    "p95_by_policy_ms": {k: round(v, 2) for k, v in cell.items()},
                    "winner_p95": winner,
                }
            )
    return rows
