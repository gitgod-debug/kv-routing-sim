"""Percentile bootstrap confidence intervals that resample whole sessions, not requests.

Requests in an agentic trace are not independent: a session's turns share prefixes by construction.
Resampling single requests ignores that dependence and gives intervals that are far too narrow, so
sessions are the resampling unit.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_RESAMPLES = 1_000
HEADLINE_RESAMPLES = 10_000  # anything in the README headline


@dataclass(frozen=True, slots=True)
class CI:
    point: float
    lo: float
    hi: float
    n_resamples: int
    unit: str  # "session" or "request", stated wherever the interval is reported

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}]"

    def as_dict(self) -> dict[str, object]:
        return {
            "point": round(self.point, 6),
            "ci_lo": round(self.lo, 6),
            "ci_hi": round(self.hi, 6),
            "n_resamples": self.n_resamples,
            "resample_unit": self.unit,
        }


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = max(0, min(len(sorted_vals) - 1, round(q * (len(sorted_vals) - 1))))
    return sorted_vals[k]


def bootstrap_ci_grouped(
    groups: Sequence[Sequence[Any]],
    statistic: Callable[[list[Any]], float],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
) -> CI:
    """Return a percentile bootstrap CI, resampling whole groups (sessions) with replacement.

    Items may be any type the statistic accepts, for example `(cached, prompt)` token pairs, so that
    a hit rate is resampled as a ratio of sums, as the engine's counters compute it.
    """
    flat = [v for g in groups for v in g]
    if not flat:
        return CI(float("nan"), float("nan"), float("nan"), n_resamples, "session")
    point = statistic(flat)

    rng = random.Random(seed)
    n = len(groups)
    stats: list[float] = []
    for _ in range(n_resamples):
        sample: list[float] = []
        for _ in range(n):
            sample.extend(groups[rng.randrange(n)])
        if sample:
            stats.append(statistic(sample))
    stats.sort()
    alpha = (1.0 - confidence) / 2.0
    return CI(
        point, _percentile(stats, alpha), _percentile(stats, 1 - alpha), n_resamples, "session"
    )


def paired_bootstrap_delta(
    groups_a: Sequence[Sequence[Any]],
    groups_b: Sequence[Sequence[Any]],
    statistic: Callable[[list[Any]], float],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
) -> CI:
    """Return a CI on the difference a - b, resampling the same group indices for both.

    Both policies ran on the same trace, so variance from the trace itself is shared and cancels in
    the difference. The interval on the delta is usually much narrower than two separate intervals
    suggest, and is the right test for a policy against a baseline.
    """
    if len(groups_a) != len(groups_b):
        raise ValueError(
            f"paired bootstrap needs matched groups, got {len(groups_a)} and {len(groups_b)}"
        )
    flat_a = [v for g in groups_a for v in g]
    flat_b = [v for g in groups_b for v in g]
    if not flat_a or not flat_b:
        return CI(float("nan"), float("nan"), float("nan"), n_resamples, "session")
    point = statistic(flat_a) - statistic(flat_b)

    rng = random.Random(seed)
    n = len(groups_a)
    deltas: list[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        sa = [v for i in idx for v in groups_a[i]]
        sb = [v for i in idx for v in groups_b[i]]
        if sa and sb:
            deltas.append(statistic(sa) - statistic(sb))
    deltas.sort()
    alpha = (1.0 - confidence) / 2.0
    return CI(
        point, _percentile(deltas, alpha), _percentile(deltas, 1 - alpha), n_resamples, "session"
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def p95(values: list[float]) -> float:
    return _percentile(sorted(values), 0.95)


def p99(values: list[float]) -> float:
    return _percentile(sorted(values), 0.99)
