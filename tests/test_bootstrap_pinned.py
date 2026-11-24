"""Pinned outputs of the bootstrap module on fixed inputs.

Stored results were computed with this module, so its numbers must not move. The order in which
random group indices are drawn and the percentile rule are therefore part of its contract, and
these tests fail if either changes.

The inputs are whole numbers, so every sum is exact and the expected values do not depend on how
a Python version adds floats.
"""

from __future__ import annotations

import math

import pytest

from kv_routing_sim.eval.bootstrap import (
    CI,
    bootstrap_ci_grouped,
    mean,
    p95,
    p99,
    paired_bootstrap_delta,
)


def _latencies(n_sessions: int, base: int) -> list[list[float]]:
    """Sessions of two to five turns. The level differs by session and a little by turn."""
    return [
        [float(base + 37 * ((s * 7919) % 23) + 5 * t) for t in range(2 + s % 4)]
        for s in range(n_sessions)
    ]


def _slower(groups: list[list[float]]) -> list[list[float]]:
    """The same sessions under a second policy that adds 11 to 13 ms to every turn."""
    return [[v + 11.0 + (i % 3) for v in g] for i, g in enumerate(groups)]


def _token_pairs(n_sessions: int) -> list[list[tuple[int, int]]]:
    """(cached tokens, prompt tokens) per request, in whole 64-token blocks."""
    sessions = []
    for s in range(n_sessions):
        prompt = 1024 + 64 * ((s * 31) % 11)
        blocks = prompt // 64
        sessions.append([(64 * ((s * 13 + t * 5) % blocks), prompt) for t in range(2 + s % 3)])
    return sessions


def _hit_rate(items: list[tuple[int, int]]) -> float:
    """A ratio of sums, which is how the engine's counters give a hit rate."""
    return sum(cached for cached, _ in items) / sum(prompt for _, prompt in items)


def _values(ci: CI) -> tuple[float, float, float]:
    return (ci.point, ci.lo, ci.hi)


def test_grouped_interval_on_a_mean_is_pinned():
    ci = bootstrap_ci_grouped(_latencies(40, 300), mean, n_resamples=200, seed=7)
    assert _values(ci) == (721.0142857142857, 623.6330935251799, 800.8059701492538)
    assert (ci.n_resamples, ci.unit) == (200, "session")


def test_grouped_interval_on_a_p95_at_90_percent_is_pinned():
    ci = bootstrap_ci_grouped(_latencies(40, 300), p95, n_resamples=200, confidence=0.90, seed=3)
    assert _values(ci) == (1092.0, 1018.0, 1114.0)


def test_grouped_interval_on_a_ratio_of_sums_is_pinned():
    ci = bootstrap_ci_grouped(_token_pairs(30), _hit_rate, n_resamples=300, seed=5)
    assert _values(ci) == (0.488176563321072, 0.42105263157894735, 0.5338737695425594)


def test_paired_interval_on_a_mean_difference_is_pinned():
    baseline = _latencies(40, 300)
    ci = paired_bootstrap_delta(_slower(baseline), baseline, mean, n_resamples=200, seed=11)
    assert _values(ci) == (11.9785714285714, 11.730496453900741, 12.270833333333371)
    assert (ci.n_resamples, ci.unit) == (200, "session")


def test_paired_interval_on_a_p95_difference_is_pinned():
    baseline = _latencies(40, 300)
    ci = paired_bootstrap_delta(_slower(baseline), baseline, p95, n_resamples=200, seed=2)
    assert _values(ci) == (11.0, 11.0, 13.0)


@pytest.mark.parametrize(
    ("statistic", "n", "expected"),
    [
        (p95, 11, 11.0),  # 0.95 * 10 = 9.5, which rounds to index 10
        (p95, 20, 19.0),
        (p95, 31, 29.0),  # 0.95 * 30 = 28.5: a half goes to the even index, 28
        (p99, 10, 10.0),
        (p99, 51, 51.0),  # 0.99 * 50 = 49.5, which rounds to index 50
        (p99, 151, 149.0),  # 0.99 * 150 = 148.5: a half goes to the even index, 148
    ],
)
def test_percentile_is_nearest_rank_with_halves_to_even(statistic, n, expected):
    """The values 1..n are given in falling order, so this also checks that the helper sorts."""
    assert statistic([float(i) for i in range(n, 0, -1)]) == expected


def test_interval_prints_and_serialises_in_the_stored_format():
    ci = CI(0.123456789, 0.1, 0.15, 1000, "session")
    assert str(ci) == "0.1235 [0.1000, 0.1500]"
    assert ci.as_dict() == {
        "point": 0.123457,
        "ci_lo": 0.1,
        "ci_hi": 0.15,
        "n_resamples": 1000,
        "resample_unit": "session",
    }


def test_no_data_gives_nan_and_keeps_the_resample_count():
    for ci in (
        bootstrap_ci_grouped([[], []], mean, n_resamples=50),
        paired_bootstrap_delta([[]], [[1.0]], mean, n_resamples=50),
    ):
        assert all(math.isnan(v) for v in _values(ci))
        assert (ci.n_resamples, ci.unit) == (50, "session")
    assert math.isnan(mean([]))
    assert math.isnan(p95([]))
