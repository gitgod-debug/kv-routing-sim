"""Evaluation of runs: statistics, cost, Pareto frontiers and the policy recommendation.

The submodules cover session-level bootstrap intervals (`bootstrap`), the cost model (`cost`), the
non-dominated frontier (`pareto`), Kendall's tau-b (`rank`), the recommendation (`recommend`), how
firm a replica count is (`spread`) and the policy x replica x seed grid (`sweep`).
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
