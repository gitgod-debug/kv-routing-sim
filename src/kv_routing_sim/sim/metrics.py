"""What a run produces. Distributions, not means (docs/06-methodology.md §1).

⛔ TTFT IS REPORTED AS p50/p95/p99, NEVER AS A MEAN ± STD. The whole argument of this project lives
in the tail: a routing policy that improves the mean while blowing p99 has made the product worse,
and averaging is how that gets hidden. `mean_ttft_ms` is deliberately absent rather than merely
discouraged.
"""

from __future__ import annotations

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

    # Cache — counted in BLOCKS, mirroring vLLM's counters (ADR-008, and the unit note in
    # cache/block_trie.py that Phase 0 must confirm).
    cache_queries: int = 0
    cache_hits: int = 0
    evictions: int = 0

    ttft_ms: list[float] = field(default_factory=list)
    completed: int = 0
    arrived: int = 0
    makespan_s: float = 0.0
    per_replica_requests: list[int] = field(default_factory=list)
    per_replica_hit_rate: list[float] = field(default_factory=list)
    iterations: int = 0
    profile: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)

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
            "evictions": self.evictions,
            "ttft": self.ttft_percentiles(),
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
