"""Least-loaded (LOR / join-shortest-queue): optimises load, ignores the cache completely.

The interesting policy in this project, because it is the one prefix-affinity has to beat — and the
one that wins back above the H2 crossover, when concentrating a hot prefix costs more in queueing
than it saves in prefill.
"""

from __future__ import annotations

import random

from ..traces.schema import NormalizedRequest
from .base import FleetState, RoutingPolicy, argmin_random


class LeastLoaded(RoutingPolicy):
    name = "least-loaded"

    def choose(self, state: FleetState, request: NormalizedRequest, rng: random.Random) -> int:
        return argmin_random(state.all_outstanding(), rng)
