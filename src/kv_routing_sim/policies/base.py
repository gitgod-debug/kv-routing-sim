"""The routing-policy interface. One file per policy; a new policy is never a core change (ADR-003).

⛔ EVERY TIE IS BROKEN FROM THE INJECTED `rng`. `min()` over a list silently prefers low indices,
which concentrates traffic on replica 0 and MANUFACTURES cache locality the real engine does not
have. That is a correctness bug that looks exactly like a result: hit rate goes up, nothing errors,
and no downstream check catches it. Every policy here takes the rng and uses it.

⭐ A FIDELITY DECISION THAT MATTERS, AND THAT A REVIEWER WILL PROBE: a policy may NOT read the
replicas' actual caches. `FleetState` exposes load only. A real prefix-aware router (SGLang's
router, Dynamo) does not have oracle knowledge of what is resident on each worker — it maintains its
OWN approximate view of what it dispatched, and that view drifts from reality as blocks are evicted
behind it. Handing the policy the true cache would overstate prefix-affinity's benefit by exactly
the amount the drift costs, which is the number we would most want to be honest about.

⚠️ CORRECTED 2026-10-05: THAT DESCRIBES ONE KIND OF PRODUCTION ROUTER, NOT ALL OF THEM. Dynamo's
default router keeps "a prefix tree built from worker-reported KV events", vLLM publishes such
events, and the LMetric authors' router follows the engine's step events: for those the true cache
is what the router knows, late by an event. Routing from the dispatch record alone is Dynamo's
*approximate* mode (`--no-router-kv-events`). This simulator models only that dispatch view; a
policy with the event view is not built (docs/04-prior-art.md §2a). The rule stands for the
policies here: a dispatch-view policy must not peek.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..traces.schema import NormalizedRequest


@dataclass(slots=True)
class FleetState:
    """What a router can see. Deliberately load-only — see the module docstring."""

    n_replicas: int
    running: list[int]  # sequences currently in each replica's batch
    queued: list[int]  # sequences waiting for a slot

    def outstanding(self, i: int) -> int:
        return self.running[i] + self.queued[i]

    def all_outstanding(self) -> list[int]:
        return [self.running[i] + self.queued[i] for i in range(self.n_replicas)]


class RoutingPolicy(ABC):
    """Pick a replica for each request. Never mutates the fleet; may keep its own state."""

    name: str = "base"

    def __init__(self, n_replicas: int, **kwargs: object) -> None:
        if n_replicas < 1:
            raise ValueError(f"n_replicas must be >= 1, got {n_replicas}")
        self.n_replicas = n_replicas

    @abstractmethod
    def choose(
        self, state: FleetState, request: NormalizedRequest, rng: random.Random
    ) -> int: ...

    def observe(self, replica: int, request: NormalizedRequest) -> None:
        """Called after a routing decision, so a policy can update its own view.

        This is how a prefix-aware router learns where it sent things WITHOUT being told what is
        actually resident. Most policies ignore it.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_replicas={self.n_replicas})"


def argmin_random(values: list[int], rng: random.Random) -> int:
    """Index of the minimum, ties broken UNIFORMLY AT RANDOM. See the module docstring."""
    lo = min(values)
    candidates = [i for i, v in enumerate(values) if v == lo]
    return candidates[0] if len(candidates) == 1 else rng.choice(candidates)
