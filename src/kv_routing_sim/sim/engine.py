"""The fleet and the event loop. `run()` is the one function the sweep calls.

A discrete-event simulation over two event types: a request arriving, and a replica finishing an
iteration. Everything else falls out of those.

⭐ THE ORDERING RULE THAT MAKES RESULTS REPRODUCIBLE: events are popped from a heap keyed by
`(time, sequence_number)`. Two events at the identical timestamp are therefore resolved in insertion
order, deterministically — never by whichever object happens to compare first. Without that
tie-break the heap would try to order the payloads themselves, and a sweep would stop being
re-derivable, which would make every Pareto curve unfalsifiable (docs/06-methodology.md §1).
"""

from __future__ import annotations

import heapq
import itertools
import random
from collections.abc import Iterable
from typing import Any

from ..policies import build as build_policy
from ..policies.base import FleetState
from ..traces.schema import NormalizedRequest
from .cost_model import UNCALIBRATED_DEFAULT, CostProfile
from .metrics import RunResult
from .replica import Replica

_ARRIVAL = 0
_ITER_DONE = 1


def run(
    trace: Iterable[NormalizedRequest],
    *,
    policy: str,
    n_replicas: int,
    seed: int = 0,
    profile: CostProfile = UNCALIBRATED_DEFAULT,
    policy_kwargs: dict[str, Any] | None = None,
    warmup_requests: int = 0,
    trace_info: dict[str, Any] | None = None,
) -> RunResult:
    """Replay a trace through one policy on N replicas.

    `warmup_requests` discards the first N requests from the METRICS while still letting them warm
    the caches — a cold cache would otherwise drag every policy's hit rate toward zero and compress
    exactly the differences we are trying to measure (docs/06-methodology.md §1).
    """
    rng = random.Random(seed)
    router = build_policy(policy, n_replicas, **(policy_kwargs or {}))
    replicas = [Replica(i, profile) for i in range(n_replicas)]
    result = RunResult(policy=policy, n_replicas=n_replicas, seed=seed)
    result.profile = profile.provenance()
    result.trace = dict(trace_info or {})

    counter = itertools.count()
    events: list[tuple[float, int, int, Any]] = []
    for req in trace:
        heapq.heappush(events, (req.arrival_ts, next(counter), _ARRIVAL, req))
        result.arrived += 1

    scheduled: set[int] = set()  # replicas with an iteration already in flight
    per_replica_requests = [0] * n_replicas
    routed_index = 0
    now = 0.0

    def state() -> FleetState:
        return FleetState(
            n_replicas=n_replicas,
            running=[len(r.running) for r in replicas],
            queued=[len(r.queue) for r in replicas],
        )

    def schedule(idx: int, at: float) -> None:
        if idx in scheduled:
            return
        scheduled.add(idx)
        heapq.heappush(events, (at, next(counter), _ITER_DONE, idx))

    while events:
        now, _, kind, payload = heapq.heappop(events)

        if kind == _ARRIVAL:
            req: NormalizedRequest = payload
            idx = router.choose(state(), req, rng)
            if not 0 <= idx < n_replicas:
                raise RuntimeError(
                    f"policy {policy!r} returned replica {idx}, outside 0..{n_replicas - 1}"
                )
            router.observe(idx, req)
            replicas[idx].enqueue(req)
            per_replica_requests[idx] += 1
            routed_index += 1
            schedule(idx, now)
            continue

        # _ITER_DONE: this replica's iteration boundary. Run the next one from here.
        idx = payload
        scheduled.discard(idx)
        replica = replicas[idx]
        if not replica.has_work:
            continue

        duration, finished = replica.step(now)
        end = now + duration
        for seq in finished:
            # Counted only once warm-up is past. `arrival_ts` ordering is the trace's own.
            if seq.request.arrival_ts >= 0 and result.completed + 1 > warmup_requests:
                result.ttft_ms.append((seq.first_token_ts - seq.arrival_ts) * 1000.0)
            result.completed += 1
            result.makespan_s = max(result.makespan_s, end)

        if replica.has_work:
            schedule(idx, end)

    for r in replicas:
        result.cache_queries += r.cache.queries
        result.cache_hits += r.cache.hits
        result.evictions += r.cache.evictions
        result.iterations += r.iterations
        result.per_replica_hit_rate.append(round(r.cache.hit_rate, 6))
    result.per_replica_requests = per_replica_requests
    return result
