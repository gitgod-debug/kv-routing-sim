"""Turn sweep points and an SLO into one policy, one replica count and a cost delta.

A recommendation from an uncalibrated profile carries a warning that its latency and rupee figures
are not predictions (ADR-010); only a calibrated profile supports absolute figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .pareto import Point, frontier


@dataclass(frozen=True, slots=True)
class Recommendation:
    chosen: Point | None
    baseline: Point | None
    reason: str
    inr_delta_pct: float | None
    replica_delta: int | None
    calibrated: bool
    frontier_size: int
    # Other policies that meet the SLO on the same number of replicas. The cost gap between them is
    # throughput noise, so they are reported as ties rather than hidden behind the pick.
    tied_with: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "recommended": self.chosen.as_dict() if self.chosen else None,
            "baseline": self.baseline.as_dict() if self.baseline else None,
            "reason": self.reason,
            "inr_delta_pct_vs_baseline": (
                round(self.inr_delta_pct, 2) if self.inr_delta_pct is not None else None
            ),
            "replica_delta_vs_baseline": self.replica_delta,
            "frontier_size": self.frontier_size,
            "calibrated": self.calibrated,
            "tied_with": list(self.tied_with),
        }
        if not self.calibrated:
            out["WARNING"] = (
                "UNCALIBRATED. The latency and rupee figures are shapes from an unmeasured cost "
                "profile, not predictions for any real fleet. The recommended policy and replica "
                "count depend on the hardware too: the full ranking of policies changes with it. "
                "Measure your own profile with kvsim calibrate."
            )
        return out


def recommend(
    points: list[Point], *, baseline_policy: str = "round-robin", slo: str = "p95 TTFT SLO"
) -> Recommendation:
    """Pick the cheapest configuration that meets the SLO, else the one with the lowest p95 TTFT.

    Args:
        slo: The SLO's name as used in the reason, normally `CostModel.slo_label`.
    """
    front = frontier(points)
    calibrated = all(p.calibrated for p in points) and bool(points)

    if not points:
        return Recommendation(None, None, "no results", None, None, False, 0)

    # Three different cases: "no SLO was supplied" must not be reported as "nothing met the SLO".
    no_slo_defined = all(p.meets_slo is None for p in points)
    feasible = [p for p in points if p.meets_slo and p.inr_per_1k != float("inf")]
    ties: tuple[str, ...] = ()

    if no_slo_defined:
        chosen = min(points, key=lambda p: (p.p95_ttft_ms, p.n_replicas))
        reason = (
            "no cost model supplied, so no SLO was applied and no rupee figure was computed — "
            "ranked by p95 TTFT alone. Pass --cost-model to get a cost-ranked recommendation."
        )
    elif feasible:
        chosen = min(feasible, key=lambda p: (p.inr_per_1k, p.n_replicas, p.p95_ttft_ms))
        reason = f"cheapest configuration meeting the {slo}"
        ties = tuple(sorted({p.policy for p in feasible
                             if p.n_replicas == chosen.n_replicas and p.policy != chosen.policy}))
    else:
        # An SLO was given and nothing met it: say so rather than relax the constraint.
        chosen = min(points, key=lambda p: (p.p95_ttft_ms, p.n_replicas))
        reason = (
            f"NO configuration met the {slo} — showing the lowest-p95 option. "
            "Either the SLO is too tight for this workload or the fleet needs more replicas "
            "than were swept."
        )

    base_candidates = [p for p in points if p.policy == baseline_policy]
    baseline = (
        min(base_candidates, key=lambda p: (not p.meets_slo, p.inr_per_1k, p.n_replicas))
        if base_candidates
        else None
    )

    inr_delta = None
    replica_delta = None
    base_ok = baseline is not None and baseline.inr_per_1k not in (0.0, float("inf"))
    if base_ok and chosen.inr_per_1k != float("inf"):
        inr_delta = (chosen.inr_per_1k - baseline.inr_per_1k) / baseline.inr_per_1k * 100.0
        replica_delta = chosen.n_replicas - baseline.n_replicas

    return Recommendation(
        chosen=chosen,
        baseline=baseline,
        reason=reason,
        inr_delta_pct=inr_delta,
        replica_delta=replica_delta,
        calibrated=calibrated,
        frontier_size=len(front),
        tied_with=ties,
    )
