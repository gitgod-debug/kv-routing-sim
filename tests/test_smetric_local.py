"""SMetric's routing rule without its global tier (policies/smetric_local.py): stick a session's
follow-ups to the replica that holds its history, unless it would miss the first-token deadline."""

from __future__ import annotations

import dataclasses
import random

import pytest

from kv_routing_sim import policies
from kv_routing_sim.policies.base import FleetState
from kv_routing_sim.policies.smetric_local import SMetricLocal
from kv_routing_sim.sim.cost_model import CostProfile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces.generator import AgenticWorkload
from kv_routing_sim.traces.schema import NormalizedRequest

# 1 ms per new prompt token, nothing else: a prefill cost can be read off the token count.
PROFILE = CostProfile(
    name="test",
    prefill_base_s=0.0,
    prefill_per_token_s=0.001,
    decode_base_s=0.002,
    decode_per_seq_s=0.0001,
    kv_cache_blocks=5_000,
    block_size=16,
    max_num_seqs=16,
)

TURN1 = NormalizedRequest("s-t1", 0.0, tuple(range(100, 106)), 100, 8, "S")
TURN2 = NormalizedRequest("s-t2", 5.0, tuple(range(100, 112)), 200, 8, "S")  # extends turn 1


def _state(running, queued=None):
    return FleetState(len(running), list(running), list(queued or [0] * len(running)))


def _policy(n=4, **kw) -> SMetricLocal:
    p = policies.build("smetric-local", n, profile=PROFILE, index="dispatch", **kw)
    assert isinstance(p, SMetricLocal)
    return p


def _serve(p, replica, request, state=None):
    """Route `request` to `replica` by hand and let its prefill finish."""
    p.choose(state or _state([0] * p.n_replicas), request, random.Random(0))
    p.observe(replica, request)
    p.on_first_token(replica, request)


def test_a_first_turn_is_balanced_and_a_follow_up_sticks():
    p = _policy()
    # A first turn has no history: it goes by the balance score, here to the only idle replica.
    assert p.choose(_state([5, 5, 0, 5]), TURN1, random.Random(0)) == 2
    p.observe(2, TURN1)
    p.on_first_token(2, TURN1)
    # Its follow-up sticks to replica 2 although every other replica is now idle: the history is
    # there (96 of the 96 tokens it implies), and 104 ms of prefill is far inside 2 s.
    assert p.choose(_state([0, 0, 9, 0]), TURN2, random.Random(0)) == 2
    assert p.stats() == {"stuck": 1, "migrated": 0, "balanced": 1}


def test_a_follow_up_moves_when_its_replica_would_miss_the_deadline():
    p = _policy(ttft_slo_s=2.0)
    _serve(p, 2, TURN1)
    # 2,400 tokens of someone else's prefill are queued on replica 2 and not yet done: 2.4 s.
    other = NormalizedRequest("big", 1.0, tuple(range(900, 1050)), 2400, 8, "T")
    p.choose(_state([0, 0, 0, 0]), other, random.Random(0))
    p.observe(2, other)
    # Sticking would cost 2.4 s + 0.104 s > 2 s: the session migrates, to an idle replica.
    chosen = p.choose(_state([0, 0, 1, 0]), TURN2, random.Random(0))
    assert chosen != 2
    assert p.stats()["migrated"] == 1 and p.stats()["stuck"] == 0
    # With twice the slack the same request stays.
    q = _policy(ttft_slo_s=2.0, slack=2.0)
    _serve(q, 2, TURN1)
    q.choose(_state([0, 0, 0, 0]), other, random.Random(0))
    q.observe(2, other)
    assert q.choose(_state([0, 0, 1, 0]), TURN2, random.Random(0)) == 2


def test_a_follow_up_whose_history_is_cached_nowhere_is_placed_like_a_new_session():
    """HIT_RATIO: the best hit must cover more than that share of what the history implies."""
    p = _policy(hit_ratio=0.5)
    _serve(p, 2, TURN1)
    # A "follow-up" of session S that shares only two of the six blocks its history implies
    # (a compacted or rewritten history): 32 of 96 tokens, under half.
    rewritten = NormalizedRequest("s-t2b", 5.0, (100, 101, 7, 8, 9, 10, 11, 12), 136, 8, "S")
    assert p.choose(_state([3, 0, 3, 3]), rewritten, random.Random(0)) == 1  # balanced: idle wins
    assert p.stats() == {"stuck": 0, "migrated": 0, "balanced": 2}


def test_the_queued_cost_is_released_at_the_first_token():
    p = _policy()
    p.choose(_state([0, 0, 0, 0]), TURN1, random.Random(0))
    p.observe(1, TURN1)
    assert p._queued_cost[1] == pytest.approx(0.100)  # 100 new tokens × 1 ms
    p.on_first_token(1, TURN1)
    assert p._queued_cost == [0.0, 0.0, 0.0, 0.0] and p._queued_tokens == [0, 0, 0, 0]


def test_it_needs_a_timing_profile_and_sane_settings():
    with pytest.raises(ValueError, match="timing profile"):
        policies.build("smetric-local", 4)
    for bad in ({"slack": 0}, {"hit_ratio": 1.0}, {"ttft_slo_s": -1}):
        with pytest.raises(ValueError, match="need ttft_slo_s"):
            policies.build("smetric-local", 4, profile=PROFILE, **bad)


def _traffic(**kw):
    base = {"n_sessions": 60, "turns_per_session": 4, "session_context_tokens": 150,
            "user_turn_tokens": 40, "assistant_turn_tokens": 60, "output_tokens": 16,
            "think_time_s": 2.0, "prefix_sharing_ratio": 0.8, "seed": 0, "arrival_rate": 4.0}
    return AgenticWorkload(**{**base, **kw}).generate()[0]


SIM = dataclasses.replace(PROFILE, prefill_per_token_s=0.0005, decode_base_s=0.05)


def test_in_the_simulator_every_request_is_accounted_for_and_the_run_reproduces():
    trace = _traffic()
    a = run(trace, policy="smetric-local", n_replicas=4, seed=1, profile=SIM)
    b = run(trace, policy="smetric-local", n_replicas=4, seed=1, profile=SIM)
    assert a.completed == len(trace) and a.summary() == b.summary() and a.router == b.router
    assert sum(a.router.values()) == len(trace)
    assert a.router["balanced"] >= 60  # at least every session's first turn
    assert a.router["stuck"] > 0


def test_a_tight_deadline_makes_it_move_sessions_and_a_loose_one_does_not():
    trace = _traffic(arrival_rate=12.0)  # three times the load: prefill queues build up
    tight = run(trace, policy="smetric-local", n_replicas=3, seed=0, profile=SIM,
                policy_kwargs={"ttft_slo_s": 0.5})
    loose = run(trace, policy="smetric-local", n_replicas=3, seed=0, profile=SIM,
                policy_kwargs={"ttft_slo_s": 1e9})
    assert tight.router["migrated"] > 0
    assert loose.router["migrated"] == 0
    assert tight.completed == loose.completed == len(trace)


def test_other_policies_report_nothing_about_themselves():
    r = run(_traffic(n_sessions=4), policy="round-robin", n_replicas=2, profile=SIM)
    assert r.router == {} and "router" not in r.summary()
