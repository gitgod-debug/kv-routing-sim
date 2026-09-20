"""Round-robin: the cache-blind baseline every other policy is measured against."""

from __future__ import annotations

import random

from ..traces.schema import NormalizedRequest
from .base import FleetState, RoutingPolicy


class RoundRobin(RoutingPolicy):
    name = "round-robin"

    def __init__(self, n_replicas: int, **kwargs: object) -> None:
        super().__init__(n_replicas)
        self._cursor = 0

    def choose(self, state: FleetState, request: NormalizedRequest, rng: random.Random) -> int:
        i = self._cursor
        self._cursor = (self._cursor + 1) % self.n_replicas
        return i
