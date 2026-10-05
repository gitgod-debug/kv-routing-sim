"""How firm is a replica count? The same sizing, on each repeat of the calibration alone.

A profile is a fit, and a fit moves when the engine is measured again. C3 put a number on that:
two runs of `kvsim calibrate` on one engine, 25 minutes apart, differed by 3–5% in their
coefficients — and across three calibrations of the same GPU, 102 of 144 replica counts were
identical and 138 within one (docs/09-hypotheses.md, 2026-10-05). So a single replica count,
printed alone, claims more than the measurement behind it can support.

`kvsim calibrate` therefore measures every timing cell several times and keeps the fit of each
repeat alone (`_source.replicates`). This module re-runs the sizing on each of those, and reports
the counts side by side.

⚠️ What the range is, and is not. It is what ONE calibration's own repeats disagree by: a floor on
the uncertainty. It is not a confidence interval — it knows nothing about another day, another
software image, or an error in the model's form.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..sim.cost_model import CostProfile
from ..traces.schema import NormalizedRequest
from .cost import CostModel
from .pareto import Point
from .sweep import sweep


def required_replicas(points: Sequence[Point]) -> dict[str, int | None]:
    """Per policy: the fewest replicas, of those swept, that meet the SLO. None if none does."""
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
) -> dict[str, Any]:
    """The replica count each policy needs on the pooled fit, and on each repeat's fit alone.

    `replicas` per policy: {"pooled": n, "repeats": [n, …], "min": a, "max": b, "firm": bool}, where
    None means "not met on the replica counts swept" and sorts above every count.
    """
    pooled = required_replicas(pooled_points)
    per_repeat = []
    for prof in replicates:
        points, _ = sweep(trace, policies=policies, replica_counts=replica_counts, seeds=seeds,
                          profile=prof, cost=cost, trace_info=trace_info)
        per_repeat.append(required_replicas(points))
    top = max(replica_counts) + 1  # "not met": above every count that was swept

    def key(n: int | None) -> int:
        return top if n is None else n

    table = {}
    for policy in policies:
        vals = [r.get(policy) for r in per_repeat]
        every = [pooled.get(policy), *vals]
        lo, hi = min(every, key=key), max(every, key=key)
        table[policy] = {"pooled": pooled.get(policy), "repeats": vals, "min": lo, "max": hi,
                         "firm": key(lo) == key(hi)}
    return {"repeats": len(replicates), "swept_up_to": max(replica_counts), "replicas": table}
