"""Frontier + an SLO -> ONE policy, ONE replica count, and a rupee delta against the baseline.

⛔ THE GUARD THAT MATTERS MOST IN THIS FILE: a recommendation built from an UNCALIBRATED profile is
never presented as a prediction. ADR-010 splits the claim into tiers, and this is where the split is
enforced in code rather than in a README sentence someone might forget to write:

    calibrated   -> policy, replica count, and absolute latency/cost figures
    uncalibrated -> policy and replica count ONLY, carrying an explicit warning that the
                    latency and rupee numbers are shapes, not predictions

H6 is what licenses even the first line: it predicts the policy RANKING survives a change of
hardware. Until H6 is tested, the uncalibrated recommendation is a hypothesis about a hypothesis,
and it says so.
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
    # Other policies that hold the SLO on the SAME number of replicas. Between those, the rupee gap
    # is throughput noise (H6 found one choice made on a 0.1% difference), so they are reported as
    # ties instead of being hidden behind the pick.
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
                "profile and are NOT predictions for any real fleet. The recommended POLICY and "
                "REPLICA COUNT depend on the hardware too: H6 found the full ranking moves with it "
                "(only the direction of the prompt-count flip held). Measure your own profile."
            )
        return out


def recommend(
    points: list[Point], *, baseline_policy: str = "round-robin", slo: str = "p95 TTFT SLO"
) -> Recommendation:
    """Cheapest configuration that meets the SLO; fall back to the lowest p95 if none does.

    `slo` names the SLO in the reason given — `CostModel.slo_label`, when it has two halves."""
    front = frontier(points)
    calibrated = all(p.calibrated for p in points) and bool(points)

    if not points:
        return Recommendation(None, None, "no results", None, None, False, 0)

    # Three genuinely different situations. Collapsing the first two — reporting "nothing met the
    # SLO" when no SLO was ever supplied — would be a false statement about the result.
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
        # A real failure: an SLO existed and nothing met it. Say so rather than quietly relaxing
        # the constraint and presenting the result as though it had passed.
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
