"""Power-of-two-choices: sample two replicas, take the less loaded.

Why it exists in the grid: it gets most of least-loaded's balance while needing load information
about only two replicas instead of all N — which is why real load balancers use it at scale. H6
predicts it sits BETWEEN round-robin and least-loaded, and that its crossover with prefix-affinity
happens at a higher load than least-loaded's.
"""

from __future__ import annotations

import random

from ..traces.schema import NormalizedRequest
from .base import FleetState, RoutingPolicy


class PowerOfTwo(RoutingPolicy):
    name = "power-of-two"

    def choose(self, state: FleetState, request: NormalizedRequest, rng: random.Random) -> int:
        if self.n_replicas == 1:
            return 0
        a, b = rng.sample(range(self.n_replicas), 2)
        la, lb = state.outstanding(a), state.outstanding(b)
        if la < lb:
            return a
        if lb < la:
            return b
        return a if rng.random() < 0.5 else b  # tie: coin flip, never "the lower index"
