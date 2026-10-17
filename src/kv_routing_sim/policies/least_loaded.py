"""Least-loaded routing (join-shortest-queue): balances load and ignores the cache."""

from __future__ import annotations

import random

from ..traces.schema import NormalizedRequest
from .base import FleetState, RoutingPolicy, argmin_random


class LeastLoaded(RoutingPolicy):
    name = "least-loaded"

    def choose(self, state: FleetState, request: NormalizedRequest, rng: random.Random) -> int:
        return argmin_random(state.all_outstanding(), rng)
