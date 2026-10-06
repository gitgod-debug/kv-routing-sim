"""The routing-policy interface. One file per policy; a new policy is never a core change (ADR-003).

⛔ EVERY TIE IS BROKEN FROM THE INJECTED `rng`. `min()` over a list silently prefers low indices,
which concentrates traffic on replica 0 and MANUFACTURES cache locality the real engine does not
have. That is a correctness bug that looks exactly like a result: hit rate goes up, nothing errors,
and no downstream check catches it. Every policy here takes the rng and uses it.

⭐ WHAT A ROUTER MAY KNOW ABOUT THE CACHES. There are two kinds of prefix-aware router in
production, and a policy here says which one it is:

  * THE DISPATCH VIEW — the default, and what every policy written before 2026-10-05 uses.
    `FleetState` exposes load only; the policy keeps its OWN map of what it sent where. That view
    drifts from reality as blocks are evicted behind it. This is the approximate mode of Dynamo's
    router (`--no-router-kv-events`) and the tree most small open-source routers keep.
  * THE EVENT VIEW — `wants_cache_view = True`. The engine tells the router which blocks it stores
    and removes: vLLM publishes KV events, Dynamo's KvIndexer is "a prefix tree built from
    worker-reported KV events", and the LMetric authors' router follows the engine's step events.
    `FleetState.hit_blocks(i, request)` then answers from replica i's cache as it is at the moment
    of the decision — an event stream with no delay and no loss, the BEST case for such a router.

⚠️ Until 2026-10-05 this docstring said no production router has the second kind, and that reading
the true cache "would overstate prefix-affinity's benefit". For event-driven routers that was
wrong (docs/04-prior-art.md §2a). What stays true: a dispatch-view policy must not peek — the two
views are different routers, and a result names the one it used.

EVENTS. A policy with `wants_events = True` is told when a request it routed emits its first
token and when it finishes, at the simulated time that happens — what a router sees on the
response stream it proxies. Policies that keep their own account of the work they have queued
(LMetric's prefill tokens, SMetric's queued prefill cost) need exactly that.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from ..traces.schema import NormalizedRequest


@dataclass(slots=True)
class FleetState:
    """What a router can see: load always, the caches only if it asks — see the module docstring."""

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
    """Pick a replica for each request. Never mutates the fleet; may keep its own state."""

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
        """Called after a routing decision, so a policy can update its own view.

        This is how a prefix-aware router learns where it sent things WITHOUT being told what is
        actually resident. Most policies ignore it.
        """

    def on_first_token(self, replica: int, request: NormalizedRequest) -> None:
        """A request routed to `replica` has emitted its first token: its prefill is done.
        Called only when `wants_events` is set."""

    def on_finish(self, replica: int, request: NormalizedRequest) -> None:
        """A request routed to `replica` has emitted its last token. Only with `wants_events`."""

    def stats(self) -> dict[str, object]:
        """What the policy can say about its own decisions (how often it stuck, moved a session,
        ...). Goes into the run's result as `router`; empty for a policy with nothing to add."""
        return {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_replicas={self.n_replicas})"


def argmin_random(values: list[int] | list[float], rng: random.Random) -> int:
    """Index of the minimum, ties broken UNIFORMLY AT RANDOM. See the module docstring."""
    lo = min(values)
    candidates = [i for i, v in enumerate(values) if v == lo]
    return candidates[0] if len(candidates) == 1 else rng.choice(candidates)
