"""The simulator end to end, including the property test the spec names by hand."""

from __future__ import annotations

import pytest

from kv_routing_sim.sim.cost_model import UNCALIBRATED_DEFAULT, CostProfile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces.generator import AgenticWorkload
from kv_routing_sim.traces.schema import NormalizedRequest

SMALL = CostProfile(
    name="test",
    prefill_base_s=0.001,
    prefill_per_token_s=0.00001,
    decode_base_s=0.002,
    decode_per_seq_s=0.0001,
    kv_cache_blocks=5_000,
    block_size=16,
    max_num_seqs=16,
)


def _trace(**kw):
    return AgenticWorkload(n_sessions=40, seed=0, **kw).generate()[0]


def test_every_request_completes():
    trace = _trace()
    r = run(trace, policy="round-robin", n_replicas=2, profile=SMALL)
    assert r.arrived == len(trace)
    assert r.completed == len(trace)
    assert len(r.ttft_ms) == len(trace)


def test_runs_are_reproducible_given_a_seed():
    """A sweep that cannot be re-derived makes its Pareto curves unfalsifiable."""
    trace = _trace()
    a = run(trace, policy="radix-affinity-balanced", n_replicas=4, seed=3, profile=SMALL)
    b = run(trace, policy="radix-affinity-balanced", n_replicas=4, seed=3, profile=SMALL)
    assert a.summary() == b.summary()


def test_prefix_affinity_never_loses_to_round_robin_on_perfectly_shared_prefixes():
    """The property test named in the spec's test matrix.

    When every request is the identical prompt, routing by prefix cannot do worse than round-robin
    on hit rate. If it ever does, the cache model or the policy is broken — and no calibration
    against a live engine would reveal it, because both would be wrong in the same direction.
    """
    blocks = tuple(range(1, 21))
    trace = [
        NormalizedRequest(f"r{i}", i * 0.05, blocks, 320, 8, f"s{i}") for i in range(200)
    ]
    rr = run(trace, policy="round-robin", n_replicas=4, seed=0, profile=SMALL)
    aff = run(trace, policy="radix-affinity", n_replicas=4, seed=0, profile=SMALL)
    assert aff.hit_rate >= rr.hit_rate


def test_more_replicas_dilute_the_cache_under_round_robin():
    """Scattering a shared prefix across more caches lowers the hit rate.

    This is the reason routing matters at all.
    """
    trace = _trace(prefix_sharing_ratio=0.8)
    one = run(trace, policy="round-robin", n_replicas=1, profile=SMALL)
    eight = run(trace, policy="round-robin", n_replicas=8, profile=SMALL)
    assert one.hit_rate > eight.hit_rate


def test_hit_rate_is_zero_on_a_cold_unique_workload():
    trace = [
        NormalizedRequest(f"r{i}", i * 0.1, tuple(range(i * 50, i * 50 + 10)), 160, 4, None)
        for i in range(30)
    ]
    r = run(trace, policy="round-robin", n_replicas=2, profile=SMALL)
    assert r.hit_rate == 0.0


def test_ttft_is_reported_as_a_distribution_never_a_mean():
    r = run(_trace(), policy="round-robin", n_replicas=2, profile=SMALL)
    pct = r.ttft_percentiles()
    assert set(pct) == {"p50_ms", "p95_ms", "p99_ms", "max_ms", "n"}
    assert "mean" not in r.summary()["ttft"]
    assert pct["p50_ms"] <= pct["p95_ms"] <= pct["p99_ms"] <= pct["max_ms"]
    assert pct["n"] == len(r.ttft_ms)  # the count behind every percentile


def test_uncalibrated_results_carry_a_warning():
    """A number from an unmeasured profile must never look like a prediction."""
    r = run(_trace(), policy="round-robin", n_replicas=2, profile=UNCALIBRATED_DEFAULT)
    assert r.calibrated is False
    assert "UNCALIBRATED" in r.summary()["WARNING"]
    assert "UNCALIBRATED" in r.profile["WARNING"]


def test_a_bad_policy_name_fails_before_any_work():
    with pytest.raises(ValueError, match="unknown policy"):
        run(_trace(), policy="nonsense", n_replicas=2, profile=SMALL)
