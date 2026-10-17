"""How firm a replica count is: the same sizing, repeated on each repeat of the calibration.

`kvsim calibrate` measures every timing cell several times and stores the fit of each repeat
alone. Sizing the fleet on each of those fits shows how much the replica count moves within one
calibration. That range is a floor on the uncertainty, not a confidence interval: it says nothing
about another day, another software image, or an error in the model's form.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..sim.cost_model import CostProfile
from ..traces.schema import NormalizedRequest
from .cost import CostModel
from .pareto import Point
from .sweep import RunCallback, sweep


def required_replicas(points: Sequence[Point]) -> dict[str, int | None]:
    """Return, per policy, the fewest swept replicas that meet the SLO, or None if none does."""
    out: dict[str, int | None] = {}
    for p in points:
        out.setdefault(p.policy, None)
        if p.meets_slo and (out[p.policy] is None or p.n_replicas < out[p.policy]):
            out[p.policy] = p.n_replicas
    return out


def replica_spread(
    trace: Sequence[NormalizedRequest],
    *,
    policies: Sequence[str],
    replica_counts: Sequence[int],
    seeds: Sequence[int],
    cost: CostModel,
    pooled_points: Sequence[Point],
    replicates: Sequence[CostProfile],
    trace_info: dict[str, Any] | None = None,
    on_run: RunCallback | None = None,
) -> dict[str, Any]:
    """Size each policy on the pooled fit and on each repeat's fit alone.

    Returns:
        {"repeats": k, "swept_up_to": n, "replicas": {policy: row}}, where each row is
        {"pooled": n, "repeats": [n, ...], "min": a, "max": b, "firm": bool}. None means the SLO
        was not met on any swept count and sorts above every count.
    """
    pooled = required_replicas(pooled_points)
    per_repeat = []
    for prof in replicates:
        points, _ = sweep(trace, policies=policies, replica_counts=replica_counts, seeds=seeds,
                          profile=prof, cost=cost, trace_info=trace_info, on_run=on_run)
        per_repeat.append(required_replicas(points))
    not_met = max(replica_counts) + 1

    def order(n: int | None) -> int:
        return not_met if n is None else n

    table = {}
    for policy in policies:
        vals = [r.get(policy) for r in per_repeat]
        every = [pooled.get(policy), *vals]
        lo, hi = min(every, key=order), max(every, key=order)
        table[policy] = {"pooled": pooled.get(policy), "repeats": vals, "min": lo, "max": hi,
                         "firm": order(lo) == order(hi)}
    return {"repeats": len(replicates), "swept_up_to": max(replica_counts), "replicas": table}
