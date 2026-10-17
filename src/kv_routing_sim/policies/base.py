"""The routing-policy interface: what a router can see of the fleet, and the base class (ADR-003).

Every tie is broken with the injected `rng`. `min()` silently prefers low indices, which
concentrates traffic on replica 0 and fabricates cache locality the engine would not have.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from ..traces.schema import NormalizedRequest


@dataclass(slots=True)
class FleetState:
    """What a router can see: load always, and the caches only if it asks.

    By default a policy keeps its own record of what it dispatched where, which drifts as replicas
    evict blocks (Dynamo's router with `--no-router-kv-events`). For a policy with
    `wants_cache_view`, `hit_blocks` reports each replica's cache at the moment of the decision, as
    an event-driven router sees it with no delay or loss (vLLM KV events, Dynamo's KvIndexer).
    """

    n_replicas: int
    running: list[int]  # sequences currently in each replica's batch
    queued: list[int]  # sequences waiting for a slot
    # Only for a policy with `wants_cache_view`: (replica, request) -> how many leading blocks of
    # the request's prompt that replica holds now. None for everyone else.
    hit_blocks: Callable[[int, NormalizedRequest], int] | None = None

    def outstanding(self, i: int) -> int:
        return self.running[i] + self.queued[i]

    def all_outstanding(self) -> list[int]:
        return [self.running[i] + self.queued[i] for i in range(self.n_replicas)]


class RoutingPolicy(ABC):
    """Pick a replica for each request. Never mutates the fleet; may keep its own state.

    A policy that keeps its own dispatch view must not read the true caches: the two views are
    different routers, and a result names the one it used. With `wants_events`, a policy is told
    when a request it routed emits its first and last token, as a router sees them on the response
    stream it proxies.
    """

    name: str = "base"
    wants_events: bool = False  # deliver on_first_token / on_finish
    wants_cache_view: bool = False  # fill FleetState.hit_blocks (the event view)

    def __init__(self, n_replicas: int, **kwargs: object) -> None:
        if n_replicas < 1:
            raise ValueError(f"n_replicas must be >= 1, got {n_replicas}")
        self.n_replicas = n_replicas

    @abstractmethod
    def choose(
        self, state: FleetState, request: NormalizedRequest, rng: random.Random
    ) -> int: ...

    def observe(self, replica: int, request: NormalizedRequest) -> None:
        """Record a routing decision, so the policy can update its own view of the fleet.

        A dispatch-view router learns here where it sent each request, without being told what is
        resident. Most policies ignore it.
        """

    def on_first_token(self, replica: int, request: NormalizedRequest) -> None:
        """Note that a request on `replica` emitted its first token, so its prefill is done.

        Called only when `wants_events` is set.
        """

    def on_finish(self, replica: int, request: NormalizedRequest) -> None:
        """Note that a request on `replica` emitted its last token; only with `wants_events`."""

    def stats(self) -> dict[str, object]:
        """Return counts of the policy's own decisions, such as how often it moved a session.

        The run result stores them as `router`; empty for a policy with nothing to add.
        """
        return {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_replicas={self.n_replicas})"


def argmin_random(values: list[int] | list[float], rng: random.Random) -> int:
    """Return the index of the minimum, breaking ties uniformly at random."""
    lo = min(values)
    candidates = [i for i, v in enumerate(values) if v == lo]
    return candidates[0] if len(candidates) == 1 else rng.choice(candidates)
