"""Discrete-event simulation of a fleet of replicas behind one router.

Three events drive a run: a request arriving, a replica finishing an iteration, and, for routers
that listen for them, a notification of first tokens and completions at the time they happen.
Events are ordered by (time, sequence number), so simultaneous events are handled in the order
they were scheduled and every run is reproducible from its seed.

A replica runs one step at a time, as vLLM runs one forward pass at a time: a request that arrives
during a step waits for the step's end, and a sequence that finishes in a step counts as running
until that end. Release 0.1.2 broke both rules when a step finished all the work a replica had.
`engine_version="0.1.2"` keeps that engine, so that results recorded with it can be checked again.
"""

from __future__ import annotations

import heapq
import itertools
import random
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Any

from ..policies import build as build_policy
from ..policies.base import FleetState
from ..traces.schema import NormalizedRequest
from .cost_model import UNCALIBRATED_DEFAULT, CostProfile
from .metrics import RunResult
from .replica import InFlight, Replica

_ARRIVAL = 0
_ITERATION_DONE = 1
_NOTIFY = 2
_RECORD = 3

ENGINE_VERSIONS = ("0.2.0", "0.1.2")
"""The engines `run` can use: the current one first, then the ones kept for the record."""
_default_engine = ENGINE_VERSIONS[0]


@contextmanager
def use_engine(version: str) -> Iterator[None]:
    """Run every simulation started inside the block with the engine of an earlier release.

    For checking results and registered predictions that were made with that engine.
    """
    global _default_engine
    if version not in ENGINE_VERSIONS:
        raise ValueError(f"engine_version must be one of {ENGINE_VERSIONS}, got {version!r}")
    previous, _default_engine = _default_engine, version
    try:
        yield
    finally:
        _default_engine = previous


class Simulation:
    """One replay of a trace through one routing policy on a fixed number of replicas."""

    def __init__(
        self,
        *,
        policy: str,
        n_replicas: int,
        seed: int,
        profile: CostProfile,
        policy_kwargs: dict[str, Any] | None,
        warmup_requests: int,
        trace_info: dict[str, Any] | None,
        cache_model: str,
        track_block_metrics: bool,
        record_per_request: bool,
        engine_version: str | None = None,
    ) -> None:
        engine_version = engine_version or _default_engine
        if engine_version not in ENGINE_VERSIONS:
            raise ValueError(f"engine_version must be one of {ENGINE_VERSIONS}, "
                             f"got {engine_version!r}")
        self.engine_version = engine_version
        self._as_0_1_2 = engine_version == "0.1.2"
        self.policy = policy
        self.n_replicas = n_replicas
        self.profile = profile
        self.cache_model = cache_model
        self.warmup_requests = warmup_requests
        self.record_per_request = record_per_request

        self._rng = random.Random(seed)
        router_kwargs = {"profile": profile, **(policy_kwargs or {})}
        self.router = build_policy(policy, n_replicas, **router_kwargs)
        self.replicas = [
            Replica(i, profile, cache_model=cache_model, track_block_metrics=track_block_metrics)
            for i in range(n_replicas)
        ]
        if self.router.wants_cache_view and cache_model != "vllm":
            raise ValueError(
                f"policy {policy!r} routes on the engine's cache events, which only the 'vllm' "
                f"cache model can answer (got cache_model={cache_model!r})"
            )
        self._cache_probe = self._cached_blocks if self.router.wants_cache_view else None

        self.result = RunResult(policy=policy, n_replicas=n_replicas, seed=seed,
                                cache_model=cache_model)
        self.result.profile = profile.provenance()
        self.result.trace = dict(trace_info or {})

        self._events: list[tuple[float, int, int, Any]] = []
        self._sequence = itertools.count()
        self._scheduled: set[int] = set()
        self._requests_per_replica = [0] * n_replicas
        self._handlers = {
            _ARRIVAL: self._on_arrival,
            _ITERATION_DONE: self._on_iteration_done,
            _NOTIFY: self._on_notify,
            _RECORD: self._on_record,
        }

    def run(self, trace: Iterable[NormalizedRequest]) -> RunResult:
        """Replay the trace to completion and return the collected metrics."""
        seen: set[str] = set()
        for request in trace:
            # Per-request state is kept by id, so a repeated id would mix two requests.
            if request.request_id in seen:
                raise ValueError(f"request id {request.request_id!r} appears more than once. "
                                 "Ids must be unique: the trace readers rename repeated ones.")
            seen.add(request.request_id)
            self._check_fits(request)
            self._push(request.arrival_ts, _ARRIVAL, request)
            self.result.arrived += 1
        while self._events:
            now, _, kind, payload = heapq.heappop(self._events)
            self._handlers[kind](now, payload)
        self._collect()
        return self.result

    def _push(self, at: float, kind: int, payload: Any) -> None:
        heapq.heappush(self._events, (at, next(self._sequence), kind, payload))

    def _schedule_iteration(self, replica: int, at: float) -> None:
        """Schedule the replica's next iteration unless one is already pending."""
        if replica in self._scheduled:
            return
        self._scheduled.add(replica)
        if not self._as_0_1_2:
            at = max(at, self.replicas[replica].next_free_ts)  # one step at a time
        self._push(at, _ITERATION_DONE, replica)

    def _check_fits(self, request: NormalizedRequest) -> None:
        """Reject a request no replica could ever hold, as vLLM refuses to start in that case."""
        if self.cache_model != "vllm":
            return
        needed = -(-(request.prompt_tokens + request.output_tokens) // self.profile.block_size)
        usable = self.profile.kv_cache_blocks - 1  # vLLM reserves one block as the null block
        if needed > usable:
            raise ValueError(
                f"request {request.request_id!r} needs {needed} KV blocks "
                f"({request.prompt_tokens} prompt + {request.output_tokens} output tokens) but a "
                f"replica only has {usable} usable. vLLM would refuse to start "
                "with this max_model_len; so does the simulator."
            )

    def _cached_blocks(self, replica: int, request: NormalizedRequest) -> int:
        """How many of the request's blocks the replica holds; a read-only lookup."""
        return len(self.replicas[replica].pool.lookup(request.block_hashes,
                                                      request.prompt_tokens))

    def _fleet_state(self, now: float) -> FleetState:
        return FleetState(
            n_replicas=self.n_replicas,
            running=[len(r.running) + self._finishing(r, now) for r in self.replicas],
            queued=[len(r.queue) for r in self.replicas],
            hit_blocks=self._cache_probe,
        )

    def _finishing(self, replica: Replica, now: float) -> int:
        """Sequences in the replica's batch that finish at the end of the step in progress."""
        if self._as_0_1_2 or now >= replica.next_free_ts:
            return 0
        return replica.finishing

    def _on_arrival(self, now: float, request: NormalizedRequest) -> None:
        idx = self.router.choose(self._fleet_state(now), request, self._rng)
        if not 0 <= idx < self.n_replicas:
            raise RuntimeError(
                f"policy {self.policy!r} returned replica {idx}, outside 0..{self.n_replicas - 1}"
            )
        self.router.observe(idx, request)
        self.replicas[idx].enqueue(request)
        self._requests_per_replica[idx] += 1
        self._schedule_iteration(idx, now)

    def _on_notify(self, now: float, payload: tuple[int, list, list]) -> None:
        idx, first_tokens, finished = payload
        for request in first_tokens:
            self.router.on_first_token(idx, request)
        for request in finished:
            self.router.on_finish(idx, request)

    def _on_iteration_done(self, now: float, idx: int) -> None:
        self._scheduled.discard(idx)
        replica = self.replicas[idx]
        if not replica.has_work:
            return
        duration, finished = replica.step(now)
        end = now + duration
        if self.router.wants_events:
            # A sequence admitted in this step emits its first token at the step's end, which is
            # when the router can learn of it.
            first_tokens = [s.request for s in finished if s.tokens_emitted == 1]
            first_tokens += [s.request for s in replica.running if s.tokens_emitted == 1]
            if first_tokens or finished:
                self._push(end, _NOTIFY, (idx, first_tokens, [s.request for s in finished]))
        if self._as_0_1_2:
            for seq in finished:
                self._record(seq, idx, end)
        elif finished:
            # A request completes when its last token lands, at the step's end. Recording it then
            # keeps completions in time order, which the warm-up count relies on.
            self._push(end, _RECORD, (idx, finished))
        if replica.has_work:
            self._schedule_iteration(idx, end)

    def _on_record(self, now: float, payload: tuple[int, list[InFlight]]) -> None:
        idx, finished = payload
        for seq in finished:
            self._record(seq, idx, now)

    def _record(self, seq: InFlight, replica: int, end: float) -> None:
        """Record a finished request. Its last token lands at the end of the step."""
        result = self.result
        gaps = seq.tokens_emitted - 1
        decode_s = end - seq.first_token_ts
        tpot_ms = decode_s / gaps * 1000.0 if gaps > 0 else float("nan")
        if gaps > 0:
            result.itl_sum_s += decode_s
            result.itl_count += gaps
        if seq.request.arrival_ts >= 0 and result.completed + 1 > self.warmup_requests:
            result.ttft_ms.append((seq.first_token_ts - seq.arrival_ts) * 1000.0)
            result.tpot_ms.append(tpot_ms)
        result.completed += 1
        result.makespan_s = max(result.makespan_s, end)
        if self.record_per_request:
            result.per_request[seq.request.request_id] = {
                "cached_tokens": seq.matched_blocks * self.profile.block_size,
                "ttft_ms": (seq.first_token_ts - seq.arrival_ts) * 1000.0,
                "tpot_ms": tpot_ms if gaps > 0 else None,
                "preemptions": seq.num_preemptions,
                "replica": replica,
            }

    def _collect(self) -> None:
        """Fold the replicas' cache and scheduler counters into the result."""
        result = self.result
        for replica in self.replicas:
            if self.cache_model == "vllm":
                cache = replica.pool
                result.evictions += cache.real_evictions
                result.preempted_queries += cache.preempted_queries
                result.preempted_hits += cache.preempted_hits
                result.eviction_events.extend(cache.eviction_events)
            else:
                cache = replica.cache
                result.evictions += cache.evictions
            result.cache_queries += cache.queries
            result.cache_hits += cache.hits
            result.preemptions += replica.preemptions
            result.iterations += replica.iterations
            result.per_replica_hit_rate.append(round(cache.hit_rate, 6))
        result.per_replica_requests = self._requests_per_replica
        result.router = dict(self.router.stats())


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
    record_per_request: bool = False,
    engine_version: str | None = None,
) -> RunResult:
    """Replay a trace through one policy on `n_replicas` replicas.

    Args:
        cache_model: "vllm" for the block-pool model of cache/block_pool.py, or "lru" for the
            plain LRU model kept as an ablation.
        warmup_requests: completions excluded from the latency metrics while still warming the
            caches.
        track_block_metrics: record vLLM-style eviction events; off by default because a long run
            produces one per recycled block.
        record_per_request: keep cached tokens, latencies and the serving replica per request.
        engine_version: "0.1.2" runs the engine of that release, kept for the record (see the
            module docstring). By default the current engine, or the one `use_engine` selects.
    """
    simulation = Simulation(
        policy=policy,
        n_replicas=n_replicas,
        seed=seed,
        profile=profile,
        policy_kwargs=policy_kwargs,
        warmup_requests=warmup_requests,
        trace_info=trace_info,
        cache_model=cache_model,
        track_block_metrics=track_block_metrics,
        record_per_request=record_per_request,
        engine_version=engine_version,
    )
    return simulation.run(trace)
