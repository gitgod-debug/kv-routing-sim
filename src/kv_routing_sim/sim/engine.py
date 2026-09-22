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
    cache_model: str = "vllm",
    track_block_metrics: bool = False,
) -> RunResult:
    """Replay a trace through one policy on N replicas.

    `cache_model` — `"vllm"` (default): the block pool of cache/block_pool.py, every behaviour
    cited to vLLM's source. `"lru"`: the original model, kept as the ablation baseline.

    `track_block_metrics` records vLLM-style eviction events (for H4) — off by default because a
    long run produces one per recycled block.

    `warmup_requests` discards the first N requests from the METRICS while still letting them warm
    the caches — a cold cache would otherwise drag every policy's hit rate toward zero and compress
    exactly the differences we are trying to measure (docs/06-methodology.md §1).
    """
    rng = random.Random(seed)
    router = build_policy(policy, n_replicas, **(policy_kwargs or {}))
    replicas = [
        Replica(i, profile, cache_model=cache_model, track_block_metrics=track_block_metrics)
        for i in range(n_replicas)
    ]
    result = RunResult(policy=policy, n_replicas=n_replicas, seed=seed, cache_model=cache_model)
    result.profile = profile.provenance()
    result.trace = dict(trace_info or {})

    counter = itertools.count()
    events: list[tuple[float, int, int, Any]] = []
    for req in trace:
        # vLLM refuses to START if one request of max_model_len cannot fit in the KV cache — the
        # v1 spike hit exactly this ("4.38 GiB KV cache is needed ... larger than the available").
        # The simulator refuses too, before any work, rather than deadlocking mid-run on a request
        # that could never be scheduled.
        if cache_model == "vllm":
            need = -(-(req.prompt_tokens + req.output_tokens) // profile.block_size)
            if need > profile.kv_cache_blocks:
                raise ValueError(
                    f"request {req.request_id!r} needs {need} KV blocks "
                    f"({req.prompt_tokens} prompt + {req.output_tokens} output tokens) but a "
                    f"replica only has {profile.kv_cache_blocks}. vLLM would refuse to start "
                    "with this max_model_len; so does the simulator."
                )
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
        if cache_model == "vllm":
            c = r.pool
            result.evictions += c.real_evictions
            result.preempted_queries += c.preempted_queries
            result.preempted_hits += c.preempted_hits
            result.eviction_events.extend(c.eviction_events)
        else:
            c = r.cache
            result.evictions += c.evictions
        result.cache_queries += c.queries
        result.cache_hits += c.hits
        result.preemptions += r.preemptions
        result.iterations += r.iterations
        result.per_replica_hit_rate.append(round(c.hit_rate, 6))
    result.per_replica_requests = per_replica_requests
    return result
