"""Confidence intervals, resampled correctly for THIS data (docs/06-methodology.md §1).

⛔ THE RESAMPLING UNIT IS A SESSION, NOT A REQUEST. This is the one place this project's statistics
differ from the sibling's, and getting it wrong inflates confidence badly.

Requests in an agentic trace are NOT independent: a session's turns share prefixes by construction,
which is the entire phenomenon under study. Resampling individual requests pretends that dependence
away and produces intervals that are far too tight — a bootstrap that says "+-0.3%" about a quantity
whose real uncertainty is +-3%. Sessions are the independent unit, so sessions are what we resample.

`bootstrap_ci` (ungrouped) exists only for quantities that genuinely are independent. Anything that
goes in a README table uses `bootstrap_ci_grouped`, and every caption states the unit.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

DEFAULT_RESAMPLES = 1_000
HEADLINE_RESAMPLES = 10_000  # anything in the README headline


@dataclass(frozen=True, slots=True)
class CI:
    point: float
    lo: float
    hi: float
    n_resamples: int
    unit: str  # "session" or "request" — printed in every caption, never assumed

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
    groups: Sequence[Sequence[float]],
    statistic: Callable[[list[float]], float],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
) -> CI:
    """Percentile bootstrap, resampling whole GROUPS (sessions) with replacement."""
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
    groups_a: Sequence[Sequence[float]],
    groups_b: Sequence[Sequence[float]],
    statistic: Callable[[list[float]], float],
    *,
    n_resamples: int = DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    seed: int = 0,
) -> CI:
    """CI on the DIFFERENCE (a - b), resampling the same group indices for both.

    ⭐ Why this and not two separate intervals: both policies ran on the SAME trace, so
    trace-difficulty variance is shared and cancels. Comparing 0.41 [0.38, 0.44] against
    0.33 [0.30, 0.36] and noting the intervals overlap tells you almost nothing; the CI on the
    delta is usually far tighter and is the correct test. Every policy-vs-baseline row gets one.
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
