"""Discrete-event simulation of a fleet of replicas behind one router.

Three events drive a run: a request arriving, a replica finishing an iteration, and, for routers
that listen for them, a notification of first tokens and completions at the time they happen.
Events are ordered by (time, sequence number), so simultaneous events are handled in the order
they were scheduled and every run is reproducible from its seed.
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
from .replica import InFlight, Replica

_ARRIVAL = 0
_ITERATION_DONE = 1
_NOTIFY = 2


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
    ) -> None:
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
        }

    def run(self, trace: Iterable[NormalizedRequest]) -> RunResult:
        """Replay the trace to completion and return the collected metrics."""
        for request in trace:
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

    def _fleet_state(self) -> FleetState:
        return FleetState(
            n_replicas=self.n_replicas,
            running=[len(r.running) for r in self.replicas],
            queued=[len(r.queue) for r in self.replicas],
            hit_blocks=self._cache_probe,
        )

    def _on_arrival(self, now: float, request: NormalizedRequest) -> None:
        idx = self.router.choose(self._fleet_state(), request, self._rng)
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
        for seq in finished:
            self._record(seq, idx, end)
        if replica.has_work:
            self._schedule_iteration(idx, end)

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
    )
    return simulation.run(trace)
