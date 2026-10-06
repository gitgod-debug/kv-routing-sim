"""What a run produces. Distributions, not means (docs/06-methodology.md §1).

⛔ TTFT IS REPORTED AS p50/p95/p99, NEVER AS A MEAN ± STD. The whole argument of this project lives
in the tail: a routing policy that improves the mean while blowing p99 has made the product worse,
and averaging is how that gets hidden. `mean_ttft_ms` is deliberately absent rather than merely
discouraged.

TWO LATENCIES, because a response has two: how long until the first token (TTFT), and how fast the
rest arrive — TPOT, the time per output token. The MLPerf row this project's 2-second TTFT limit
comes from sets both (configs/cost_model.yaml). Until 2026-10-05 only the first was measured here;
C3's live engine then took 134–521 ms per token on traffic that met, or nearly met, the first-token
limit (docs/09-hypotheses.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile on an ALREADY SORTED list. No interpolation.

    Nearest-rank is used because an interpolated p99 invents a latency no request experienced. Every
    number here is a value some request actually saw.
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

    # Cache — in TOKENS, as vLLM counts (ADR-008, confirmed on a live engine 2026-09-21).
    # `cache_queries`/`cache_hits` exclude preempted re-lookups, exactly like vllm:prefix_cache_*.
    cache_queries: int = 0
    cache_hits: int = 0
    evictions: int = 0  # REAL evictions: a block that held a cached prefix was reused
    preemptions: int = 0
    preempted_queries: int = 0
    preempted_hits: int = 0
    eviction_events: list[Any] = field(default_factory=list)  # only with track_block_metrics
    # request_id -> {cached_tokens, ttft_ms, tpot_ms, preemptions, replica}; only with
    # record_per_request. The engine reports the same `cached_tokens` per request
    # (`usage.prompt_tokens_details`), so a calibration can say WHICH requests diverge, not just
    # whether the total does.
    per_request: dict[str, dict[str, Any]] = field(default_factory=dict)

    ttft_ms: list[float] = field(default_factory=list)
    # Time per output token, one value per request and ALIGNED with `ttft_ms`:
    #     (last token − first token) ÷ (output tokens − 1), in ms.
    # MLPerf's TPOT ("the average interval between all the tokens generated") and vLLM's
    # `mean_time_per_output_token` (v1/metrics/stats.py:549-562 @ v0.30.0) — a stall while
    # preempted is inside it, as it is there. NaN for a request with ONE output token: it has no
    # interval, so it is in no percentile and cannot miss a per-token limit.
    tpot_ms: list[float] = field(default_factory=list)
    # The engine's own inter-token counters, as vllm:inter_token_latency_seconds_{sum,count}
    # accumulate them — one observation per token after a request's first (stats.py:498-504).
    # Whole-run, like `cache_queries`: an engine does not know about warm-up.
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
        """⛔ Derived from two counters. Never a stored gauge (ADR-008)."""
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
            # The request count behind every percentile — a p99 from 100 requests is ONE request,
            # and a reader cannot judge the number without knowing that.
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
        """Σ inter-token gaps ÷ their number — what Δsum ÷ Δcount of
        `vllm:inter_token_latency_seconds` gives on a live engine.

        ⚠️ For comparing with that counter, and nothing else. It is a mean: no target is set on
        it and nothing is sized on it — the per-token target is a percentile of per-request TPOT.
        It exists for the reason `hit_rate` does: the engine publishes two counters, and a model
        that cannot be compared with them cannot be checked.
        """
        return self.itl_sum_s / self.itl_count * 1000.0 if self.itl_count else float("nan")

    def load_imbalance(self) -> float:
        """max/mean requests per replica. 1.0 is perfect balance.

        This is how locality's cost shows up: prefix-affinity concentrates traffic, and this is the
        number that says by how much. It is the mechanism behind H2 made visible.
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
