"""The sweep: policy x replicas x seeds -> points, and the crossover sweep that tests H2.

Multi-seed is not optional here. A single seed hides the variance that policy tie-breaking
introduces, and the differences between policies are sometimes the same size as that variance —
which is precisely the case where a single number would mislead (docs/06-methodology.md §1).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..sim.cost_model import UNCALIBRATED_DEFAULT, CostProfile
from ..sim.engine import run
from ..sim.metrics import RunResult
from ..traces.generator import AgenticWorkload
from ..traces.schema import NormalizedRequest
from .cost import CostModel
from .pareto import Point, to_point


def sweep(
    trace: Sequence[NormalizedRequest],
    *,
    policies: Sequence[str],
    replica_counts: Sequence[int],
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    profile: CostProfile = UNCALIBRATED_DEFAULT,
    cost: CostModel | None = None,
    trace_info: dict[str, Any] | None = None,
) -> tuple[list[Point], list[RunResult]]:
    """Run the full grid. Returns (one averaged Point per cell, every raw RunResult)."""
    results: list[RunResult] = []
    points: list[Point] = []
    for policy in policies:
        for n in replica_counts:
            cell = [
                run(
                    trace,
                    policy=policy,
                    n_replicas=n,
                    seed=s,
                    profile=profile,
                    trace_info=trace_info,
                )
                for s in seeds
            ]
            results.extend(cell)
            points.append(_average_cell(cell, cost))
    return points, results


def _average_cell(cell: list[RunResult], cost: CostModel | None) -> Point:
    """Average a cell's seeds into one point.

    ⚠️ Percentiles are averaged ACROSS SEEDS, not pooled. Pooling every seed's TTFTs into one list
    and taking p95 of that would answer a different question — "the p95 of a mixture of runs" rather
    than "the typical p95 of a run" — and would systematically understate run-to-run variation.
    """
    pts = [to_point(r, cost) for r in cell]
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
        # A cell meets the SLO only if EVERY seed did. "Met it on average" is not met.
        # None propagates: with no cost model there is no SLO to meet or miss.
        meets_slo=(
            None if any(p.meets_slo is None for p in pts) else all(p.meets_slo for p in pts)
        ),
        calibrated=pts[0].calibrated,
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
    """⭐ The H2 experiment: which policy wins on p95 TTFT across (sharing x load)?

    This is the finding, and it is a REPORTING AXIS on machinery that already exists — not new
    work. The x-axis is CROSS-SESSION sharing (the measured value, not the requested knob), because
    within-session reuse is captured by session-stickiness for free and would hide the effect.
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
