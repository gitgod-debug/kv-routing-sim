"""Statistics, cost, Pareto curves, and the recommendation.

  * `bootstrap.py`  — percentile bootstrap, and a PAIRED bootstrap for policy-vs-policy deltas.
                      ⚠️ The resampling unit is a SESSION or trace segment, never an individual
                      request: an agentic session's turns share prefixes by construction, so
                      request-level resampling breaks the dependence that is the phenomenon under
                      study and produces intervals that are far too tight.
  * `cost.py`       — reads `configs/cost_model.yaml`. NO constant is defined in Python (ADR-006),
                      and it refuses to run while a `<TBD>` remains rather than treating it as 0.
  * `pareto.py`     — the non-dominated frontier over (hit rate, p95 TTFT, ₹/1k requests).
  * `rank.py`       — Kendall's tau-b, with ties handled: the statistic H6 is decided on.
  * `recommend.py`  — frontier + an SLO → ONE policy, ONE replica count, and the ₹ delta vs
                      round-robin. (Not unique: NVIDIA's `aisimulate recommend` returns a GPU
                      count under an SLO. What this adds is the routing policy —
                      docs/04-prior-art.md §2a.)
  * `spread.py`     — how firm a replica count is: the same sizing on each repeat of the
                      calibration alone, side by side with the pooled answer.
  * `sweep.py`      — policy × replicas × seeds → points, and `crossover_sweep`, the H2 grid with
                      the sharing ratio as its own axis.
"""

from .bootstrap import CI, bootstrap_ci_grouped, paired_bootstrap_delta
from .cost import CostModel, UnfilledCostModel
from .pareto import Point, frontier, to_point
from .rank import kendall_tau_b
from .recommend import Recommendation, recommend
from .spread import replica_spread, required_replicas
from .sweep import crossover_sweep, sweep

__all__ = [
    "CI", "CostModel", "Point", "Recommendation", "UnfilledCostModel",
    "bootstrap_ci_grouped", "crossover_sweep", "frontier", "kendall_tau_b",
    "paired_bootstrap_delta", "recommend", "replica_spread", "required_replicas", "sweep",
    "to_point",
]
