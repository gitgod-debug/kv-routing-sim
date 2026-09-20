"""Statistics, cost, Pareto curves, and the recommendation. 📐 designed, not built.

  * `bootstrap.py`  — percentile bootstrap, and a PAIRED bootstrap for policy-vs-policy deltas.
                      ⚠️ The resampling unit is a SESSION or trace segment, never an individual
                      request: an agentic session's turns share prefixes by construction, so
                      request-level resampling breaks the dependence that is the phenomenon under
                      study and produces intervals that are far too tight.
  * `cost.py`       — reads `configs/cost_model.yaml`. NO constant is defined in Python (ADR-006),
                      and it refuses to run while a `<TBD>` remains rather than treating it as 0.
  * `pareto.py`     — the non-dominated frontier over (hit rate, p95 TTFT, ₹/1k requests).
  * `recommend.py`  — frontier + an SLO → ONE policy, ONE replica count, and the ₹ delta vs
                      round-robin. This output is the thing no competitor produces.
  * `crossover.py`  — the H2 finding: sharing ratio as its own axis, the flip boundary published
                      as a curve with CIs.
"""

from .bootstrap import CI, bootstrap_ci_grouped, paired_bootstrap_delta
from .cost import CostModel, UnfilledCostModel
from .pareto import Point, frontier, to_point
from .recommend import Recommendation, recommend
from .sweep import crossover_sweep, sweep

__all__ = [
    "CI", "CostModel", "Point", "Recommendation", "UnfilledCostModel",
    "bootstrap_ci_grouped", "crossover_sweep", "frontier", "paired_bootstrap_delta",
    "recommend", "sweep", "to_point",
]
