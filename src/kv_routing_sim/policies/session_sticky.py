"""Session-sticky routing: a session's requests all go to the replica chosen for its first.

Stickiness follows session identity, not shared-prefix identity: sessions that send the same system
prompt are spread over the fleet, and each replica caches its own copy. A session is placed once,
before its length is known, so a replica that receives several long sessions can overload;
`smetric_local.py` adds a rule that moves such a session.
"""

from __future__ import annotations

import random

from ..traces.schema import NormalizedRequest
from .base import FleetState, RoutingPolicy, argmin_random


class SessionSticky(RoutingPolicy):
    name = "session-sticky"

    def __init__(self, n_replicas: int, *, max_sessions: int = 1_000_000, **kwargs: object) -> None:
        super().__init__(n_replicas)
        self._assigned: dict[str, int] = {}
        self.max_sessions = max_sessions

    def choose(self, state: FleetState, request: NormalizedRequest, rng: random.Random) -> int:
        sid = request.session_id
        if sid is None:
            return argmin_random(state.all_outstanding(), rng)
        existing = self._assigned.get(sid)
        if existing is not None:
            return existing
        # A new session goes to the least-loaded replica (sticky least-outstanding-requests).
        chosen = argmin_random(state.all_outstanding(), rng)
        if len(self._assigned) < self.max_sessions:
            self._assigned[sid] = chosen
        return chosen
