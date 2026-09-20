"""The Pareto frontier over (hit rate, p95 TTFT, cost) and the recommendation that comes out of it.

The frontier is the honest output: there is no single best configuration, only a set where nothing
is better on every axis at once. The RECOMMENDATION is what turns that into something actionable —
apply the SLO as a hard constraint, then minimise cost among what is left. That step is the thing no
competitor produces (docs/04-prior-art.md): Vidur sizes without seeing the cache, Frontier sees the
cache but hands you a configuration space rather than an answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..sim.metrics import RunResult
from .cost import CostModel, successful_requests


@dataclass(frozen=True, slots=True)
class Point:
    """One (policy, replicas) configuration, with everything needed to rank it."""

    policy: str
    n_replicas: int
    hit_rate: float
    p95_ttft_ms: float
    p50_ttft_ms: float
    inr_per_1k: float
    throughput_rps: float
    load_imbalance: float
    meets_slo: bool | None  # None = no SLO was defined (no cost model)
    calibrated: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "n_replicas": self.n_replicas,
            "hit_rate": round(self.hit_rate, 6),
            "p50_ttft_ms": round(self.p50_ttft_ms, 3),
            "p95_ttft_ms": round(self.p95_ttft_ms, 3),
            "inr_per_1k_requests": (
                round(self.inr_per_1k, 4) if self.inr_per_1k != float("inf") else None
            ),
            "throughput_rps": round(self.throughput_rps, 4),
            "load_imbalance": round(self.load_imbalance, 4),
            "meets_slo": self.meets_slo,
            "calibrated": self.calibrated,
        }


def to_point(result: RunResult, cost: CostModel | None) -> Point:
    pct = result.ttft_percentiles()
    p95 = pct["p95_ms"]
    # None, not False: "no SLO was defined" is a different statement from "failed the SLO".
    meets = (p95 <= cost.ttft_p95_slo_ms) if cost is not None else None
    inr = float("inf")
    if cost is not None and result.makespan_s > 0:
        ok = successful_requests(result.ttft_ms, cost.ttft_p95_slo_ms)
        per_hour = ok / result.makespan_s * 3600.0
        inr = cost.inr_per_1k_requests(
            n_replicas=result.n_replicas, successful_requests_per_hour=per_hour
        )
    return Point(
        policy=result.policy,
        n_replicas=result.n_replicas,
        hit_rate=result.hit_rate,
        p95_ttft_ms=p95,
        p50_ttft_ms=pct["p50_ms"],
        inr_per_1k=inr,
        throughput_rps=result.throughput_rps(),
        load_imbalance=result.load_imbalance(),
        meets_slo=meets,
        calibrated=result.calibrated,
    )


def frontier(points: list[Point]) -> list[Point]:
    """Non-dominated set: maximise hit rate, minimise p95 TTFT, minimise cost.

    A point is dominated if another is at least as good on all three and strictly better on one.
    """

    def dominates(a: Point, b: Point) -> bool:
        at_least = (
            a.hit_rate >= b.hit_rate
            and a.p95_ttft_ms <= b.p95_ttft_ms
            and a.inr_per_1k <= b.inr_per_1k
        )
        strictly = (
            a.hit_rate > b.hit_rate
            or a.p95_ttft_ms < b.p95_ttft_ms
            or a.inr_per_1k < b.inr_per_1k
        )
        return at_least and strictly

    return [p for p in points if not any(dominates(q, p) for q in points if q is not p)]
