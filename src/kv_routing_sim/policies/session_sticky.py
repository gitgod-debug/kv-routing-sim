"""Session-sticky: same session -> same replica. Locality BY PROXY.

⭐ THE DISTINCTION THIS PROJECT TURNS ON, and the one most often blurred:

Sticky routing sees SESSION identity. Prefix-affinity sees SHARED-PREFIX identity. They coincide for
a session re-sending its own history — which is why sticky looks competitive on agentic traffic —
and they diverge completely on the part that matters:

    if a thousand DIFFERENT users send the SAME system prompt,
      sticky   -> spreads them across the fleet; every replica caches its own copy
      affinity -> concentrates them; one cached copy serves all thousand

That is why `cross_session_sharing_ratio` is the x-axis of H2 and not total block reuse, and it is
the capability gap against `NetX-lab/Frontier`, which registers sticky-round-robin and sticky-LOR
and has NO prefix-affinity router (docs/04-prior-art.md).

A request with no session id falls back to least-loaded — real traffic has anonymous requests, and
routing them all to one replica would be an artifact of the model rather than of the policy.

⚠️ ITS KNOWN WEAKNESS, WHICH GENERATED TRAFFIC DOES NOT SHOW (added 2026-10-05): a session is
placed once, when nothing is known about how long it will get. Real sessions differ widely in
size, and a replica that happens to receive several long ones overloads. SMetric (arXiv
2607.08565) measures a load 2.17× out of balance for routing by session id on production traces,
and adds a rule that moves a session whose replica would miss its deadline. Every session
`kvsim generate` makes has the same number of turns.
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
        # A new session is placed where there is room — the standard sticky-LOR behaviour.
        chosen = argmin_random(state.all_outstanding(), rng)
        if len(self._assigned) < self.max_sessions:
            self._assigned[sid] = chosen
        return chosen
