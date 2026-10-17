"""The result of one simulation run, with latency reported as percentiles rather than means.

A routing policy can improve mean TTFT while worsening the tail, so TTFT and time per output token
(TPOT) are reported as p50, p95 and p99, and there is deliberately no mean TTFT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def percentile(sorted_values: list[float], q: float) -> float:
    """Return the nearest-rank percentile of an already sorted list, without interpolation.

    An interpolated p99 can be a latency no request experienced; nearest rank always returns a value
    some request saw.
    """
    if not sorted_values:
        return float("nan")
    k = max(0, min(len(sorted_values) - 1, round(q * (len(sorted_values) - 1))))
    return sorted_values[k]


@dataclass(slots=True)
class RunResult:
    """One (policy, replicas, seed) run."""

    policy: str
    n_replicas: int
    seed: int

    cache_model: str = "vllm"

    # Cache counters, in tokens as vLLM counts them (ADR-008). `cache_queries` and `cache_hits`
    # exclude lookups by preempted requests, as vllm:prefix_cache_* do.
    cache_queries: int = 0
    cache_hits: int = 0
    evictions: int = 0  # real evictions: a block that held a cached prefix was reused
    preemptions: int = 0
    preempted_queries: int = 0
    preempted_hits: int = 0
    eviction_events: list[Any] = field(default_factory=list)  # only with track_block_metrics
    # request_id -> {cached_tokens, ttft_ms, tpot_ms, preemptions, replica}; only with
    # record_per_request. The engine reports the same `cached_tokens` per request
    # (`usage.prompt_tokens_details`), so a calibration can say which requests diverge, not only
    # whether the total does.
    per_request: dict[str, dict[str, Any]] = field(default_factory=dict)

    ttft_ms: list[float] = field(default_factory=list)
    # Time per output token in ms, one value per request, aligned with `ttft_ms`:
    #     (last token - first token) / (output tokens - 1)
    # as MLPerf's TPOT and vLLM's `mean_time_per_output_token` define it
    # (v1/metrics/stats.py:549-562 at v0.30.0), including any stall while preempted. NaN for a
    # request with one output token: it has no interval, so it is in no percentile and cannot miss
    # a per-token limit.
    tpot_ms: list[float] = field(default_factory=list)
    # The engine's own inter-token counters, as vllm:inter_token_latency_seconds_{sum,count}
    # accumulate them: one observation per token after a request's first (stats.py:498-504).
    # Whole-run, like `cache_queries`, since an engine does not know about warm-up.
    itl_sum_s: float = 0.0
    itl_count: int = 0
    completed: int = 0
    arrived: int = 0
    makespan_s: float = 0.0
    per_replica_requests: list[int] = field(default_factory=list)
    per_replica_hit_rate: list[float] = field(default_factory=list)
    iterations: int = 0
    profile: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)
    # The policy's own account of its decisions (RoutingPolicy.stats()); {} for most policies.
    router: dict[str, Any] = field(default_factory=dict)

    @property
    def hit_rate(self) -> float:
        """Cached tokens over queried tokens, derived from the two counters (ADR-008)."""
        return self.cache_hits / self.cache_queries if self.cache_queries else 0.0

    @property
    def calibrated(self) -> bool:
        return bool(self.profile.get("calibrated"))

    def ttft_percentiles(self) -> dict[str, float]:
        s = sorted(self.ttft_ms)
        return {
            "p50_ms": round(percentile(s, 0.50), 3),
            "p95_ms": round(percentile(s, 0.95), 3),
            "p99_ms": round(percentile(s, 0.99), 3),
            "max_ms": round(s[-1], 3) if s else float("nan"),
            # The request count behind the percentiles: a p99 over 100 requests is one request.
            "n": len(s),
        }

    def tpot_percentiles(self) -> dict[str, float]:
        """Percentiles of the per-request time per output token, over requests that have one."""
        s = sorted(t for t in self.tpot_ms if not math.isnan(t))  # NaN: one token, no interval
        return {
            "p50_ms": round(percentile(s, 0.50), 3),
            "p95_ms": round(percentile(s, 0.95), 3),
            "p99_ms": round(percentile(s, 0.99), 3),
            "max_ms": round(s[-1], 3) if s else float("nan"),
            "n": len(s),
        }

    def engine_itl_mean_ms(self) -> float:
        """Return the mean inter-token gap in ms, as `vllm:inter_token_latency_seconds` gives it.

        That is the counter's sum over its count, for comparison with a live engine only; targets
        and sizing use percentiles of per-request TPOT instead.
        """
        return self.itl_sum_s / self.itl_count * 1000.0 if self.itl_count else float("nan")

    def load_imbalance(self) -> float:
        """Return max over mean requests per replica; 1.0 is perfect balance.

        It shows how much a policy concentrates traffic, which is the cost of prefix locality.
        """
        if not self.per_replica_requests:
            return float("nan")
        mean = sum(self.per_replica_requests) / len(self.per_replica_requests)
        return max(self.per_replica_requests) / mean if mean else float("nan")

    def throughput_rps(self) -> float:
        return self.completed / self.makespan_s if self.makespan_s else 0.0

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "policy": self.policy,
            "n_replicas": self.n_replicas,
            "seed": self.seed,
            "hit_rate": round(self.hit_rate, 6),
            "cache_queries": self.cache_queries,
            "cache_hits": self.cache_hits,
            "cache_model": self.cache_model,
            "evictions": self.evictions,
            "preemptions": self.preemptions,
            "ttft": self.ttft_percentiles(),
            "tpot": self.tpot_percentiles(),
            "completed": self.completed,
            "arrived": self.arrived,
            "makespan_s": round(self.makespan_s, 4),
            "throughput_rps": round(self.throughput_rps(), 4),
            "load_imbalance": round(self.load_imbalance(), 4),
            "iterations": self.iterations,
            "calibrated": self.calibrated,
        }
        if not self.calibrated:
            out["WARNING"] = (
                "UNCALIBRATED: hit rate and load imbalance are properties of the routing model and "
                "are meaningful; TTFT and any cost derived from it are NOT predictions."
            )
        return out
