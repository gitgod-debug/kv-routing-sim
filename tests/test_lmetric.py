"""Tests for LMetric (OSDI '26) and the two things it needs from the simulator: the engine's cache
events, and notice of when each request's first and last tokens happen."""

from __future__ import annotations

import dataclasses
import pathlib
import random
from typing import ClassVar

import pytest
import yaml

from kv_routing_sim import policies
from kv_routing_sim.policies import registry
from kv_routing_sim.policies.base import FleetState
from kv_routing_sim.policies.lmetric import LMetric
from kv_routing_sim.policies.round_robin import RoundRobin
from kv_routing_sim.sim.cost_model import CostProfile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces.generator import AgenticWorkload
from kv_routing_sim.traces.schema import NormalizedRequest

REPO = pathlib.Path(__file__).resolve().parents[1]

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

# 100 tokens: six full blocks and a tail, so even a fully cached prompt has tokens left to compute.
REQ = NormalizedRequest("r1", 0.0, (11, 22, 33, 44, 55, 66), 100, 20, "sess-A")


def _state(running, queued=None, hit_blocks=None):
    return FleetState(len(running), list(running), list(queued or [0] * len(running)), hit_blocks)


def _dispatch(n=4) -> LMetric:
    p = policies.build("lmetric", n, index="dispatch")
    assert isinstance(p, LMetric)
    return p


# The registry


def test_all_is_still_the_seven_policies_of_the_registered_sweeps():
    """`policies: all` in Phase 4's design, which H6 reruns against a stored table, must still mean
    the seven original policies."""
    assert len(policies.ALL) == 7 and "lmetric" not in policies.ALL
    assert "lmetric" in policies.REGISTRY
    design = yaml.safe_load((REPO / "configs" / "phase4_sweep.yaml").read_text())
    assert policies.resolve(design["policies"]) == list(policies.ALL)
    assert policies.resolve("lmetric,round-robin") == ["lmetric", "round-robin"]


# The score


def test_lmetric_is_the_authors_formula():
    """blitz-router: `Select min by prefill_tokens(req, sctx) * (sctx.bs + 1)`, with
    prefill_tokens = queued + new and bs = running + queued."""
    p = _dispatch(3)
    p._queued_tokens = [0, 500, 40]  # prefill already committed to each replica
    # Nobody holds the prefix: 100 new tokens everywhere.
    #   replica 0: (0 + 100) * (2 + 3 + 1) = 600
    #   replica 1: (500 + 100) * (0 + 0 + 1) = 600
    #   replica 2: (40 + 100) * (1 + 2 + 1) = 560   <- the minimum
    assert p.new_tokens(_state([2, 0, 1], [3, 0, 2]), REQ) == [100, 100, 100]
    assert p.choose(_state([2, 0, 1], [3, 0, 2]), REQ, random.Random(0)) == 2


def test_a_cached_prefix_lowers_the_score_and_load_can_outweigh_it():
    p = _dispatch()
    p.choose(_state([0, 0, 0, 0]), REQ, random.Random(0))
    p.observe(3, REQ)  # replica 3 now holds the prompt, and has its prefill queued
    p.on_first_token(3, REQ)  # ... which is done
    again = NormalizedRequest("r2", 1.0, REQ.block_hashes, 100, 20, "sess-B")
    # Six blocks are cached there, so 100 - 96 = 4 new tokens against 100 anywhere else.
    assert p.new_tokens(_state([0, 0, 0, 0]), again) == [100, 100, 100, 4]
    assert p.choose(_state([0, 0, 0, 0]), again, random.Random(0)) == 3
    # 4 * (bs + 1) passes 100 * 1 once that replica holds 25 requests and another is idle.
    assert p.choose(_state([0, 9, 9, 23]), again, random.Random(0)) == 3  # 4 * 24 = 96 < 100
    assert p.choose(_state([0, 9, 9, 25]), again, random.Random(0)) == 0  # 4 * 26 = 104 > 100


def test_the_last_token_is_never_counted_as_cached():
    """The engine recomputes at least the last token (`max_hit_blocks`); a router that believed a
    prompt was free to serve would score its replica 0 whatever its load."""
    exact = NormalizedRequest("e1", 0.0, (1, 2, 3, 4), 64, 8, "s")  # exactly four blocks
    p = _dispatch()
    p.choose(_state([0, 0, 0, 0]), exact, random.Random(0))
    p.observe(1, exact)
    p.on_first_token(1, exact)
    again = NormalizedRequest("e2", 1.0, exact.block_hashes, 64, 8, "s")
    assert p.new_tokens(_state([0, 0, 0, 0]), again)[1] == 16  # three blocks hit, one recomputed


def test_queued_prefill_is_held_until_the_first_token():
    p = _dispatch()
    a = NormalizedRequest("a", 0.0, (1, 2, 3), 60, 5, "s1")
    b = NormalizedRequest("b", 0.0, (7, 8, 9), 50, 5, "s2")
    for r in (a, b):
        p.choose(_state([0, 0, 0, 0]), r, random.Random(0))
        p.observe(2, r)
    assert p._queued_tokens == [0, 0, 110, 0]
    p.on_first_token(2, a)
    assert p._queued_tokens == [0, 0, 50, 0]
    p.on_first_token(2, a)  # told twice: counted once
    p.on_finish(2, a)
    assert p._queued_tokens == [0, 0, 50, 0]
    p.on_first_token(2, b)
    assert p._queued_tokens == [0, 0, 0, 0]


def test_idle_replicas_are_told_apart_by_their_queued_prefill():
    """The reason for the `+ 1`: without it every idle replica would score zero."""
    p = _dispatch(3)
    p._queued_tokens = [300, 0, 200]
    assert p.choose(_state([0, 0, 0]), REQ, random.Random(0)) == 1


def test_lmetric_has_no_weight_to_set():
    with pytest.raises(ValueError, match="index must be one of"):
        policies.build("lmetric", 4, index="oracle")
    assert policies.build("lmetric", 4, load_penalty=16).load_penalty == 0.0  # ignored, not used


# In the simulator


def _traffic(**kw):
    base = {"n_sessions": 60, "turns_per_session": 4, "session_context_tokens": 150,
            "user_turn_tokens": 40, "assistant_turn_tokens": 60, "output_tokens": 16,
            "think_time_s": 2.0, "prefix_sharing_ratio": 0.8, "seed": 0, "arrival_rate": 4.0}
    return AgenticWorkload(**{**base, **kw}).generate()[0]


@pytest.mark.parametrize("index", ["events", "dispatch"])
def test_lmetric_completes_every_request_and_is_reproducible(index):
    trace = _traffic()
    a = run(trace, policy="lmetric", n_replicas=4, seed=3, profile=SMALL,
            policy_kwargs={"index": index})
    b = run(trace, policy="lmetric", n_replicas=4, seed=3, profile=SMALL,
            policy_kwargs={"index": index})
    assert a.completed == a.arrived == len(trace)
    assert a.summary() == b.summary() and a.per_replica_requests == b.per_replica_requests


def test_lmetric_follows_the_cache_while_idle_and_spreads_under_load():
    """Like pure prefix-affinity (H2), LMetric follows the cached prompt while its replica is idle,
    but spreads once that replica's batch makes an idle replica score lower."""
    trace = _traffic()
    # Fast engine: every request is done before the next arrives. Following the cache is right.
    idle = run(trace, policy="lmetric", n_replicas=4, seed=0, profile=SMALL)
    assert max(idle.per_replica_requests) == len(trace)
    # The same traffic on an engine 50x slower: requests overlap, and batches build up.
    slow = dataclasses.replace(SMALL, prefill_per_token_s=0.0005, decode_base_s=0.05)
    pure = run(trace, policy="radix-affinity", n_replicas=4, seed=0, profile=slow)
    lm = run(trace, policy="lmetric", n_replicas=4, seed=0, profile=slow)
    assert max(pure.per_replica_requests) == len(trace)  # the collapse, as documented
    assert max(lm.per_replica_requests) < 0.5 * len(trace)
    assert min(lm.per_replica_requests) > 0
    assert lm.summary()["ttft"]["p95_ms"] < 0.5 * pure.summary()["ttft"]["p95_ms"]


def test_the_event_view_needs_the_vllm_cache_model():
    trace = _traffic(n_sessions=6)
    with pytest.raises(ValueError, match="cache events"):
        run(trace, policy="lmetric", n_replicas=2, profile=SMALL, cache_model="lru")
    r = run(trace, policy="lmetric", n_replicas=2, profile=SMALL, cache_model="lru",
            policy_kwargs={"index": "dispatch"})
    assert r.completed == len(trace)


# Events and cache views offered to policies that ask for them


class _Recorder(RoundRobin):
    """Round-robin that records every event and, at each arrival, how many first tokens it had
    heard of."""

    name = "test-recorder"
    wants_events = True
    wants_cache_view = True
    log: ClassVar[list] = []  # the last instance's events, for the test that built it by name

    def __init__(self, n_replicas, **kwargs):
        super().__init__(n_replicas)
        type(self).log = self.events = []
        self.firsts_heard = 0

    def choose(self, state, request, rng):
        probe = [state.hit_blocks(i, request) for i in range(self.n_replicas)]
        i = super().choose(state, request, rng)
        self.events.append(("arrive", request.request_id, self.firsts_heard, probe[i], i))
        return i

    def on_first_token(self, replica, request):
        self.firsts_heard += 1
        self.events.append(("first", request.request_id, replica))

    def on_finish(self, replica, request):
        self.events.append(("finish", request.request_id, replica))


@pytest.fixture
def recorder(monkeypatch):
    monkeypatch.setitem(registry.REGISTRY, _Recorder.name, _Recorder)
    return _Recorder


def test_a_listening_policy_hears_each_request_once_and_when_it_happens(recorder):
    trace = _traffic()
    r = run(trace, policy=recorder.name, n_replicas=3, seed=0, profile=SMALL,
            record_per_request=True)
    log = recorder.log
    firsts = [e for e in log if e[0] == "first"]
    finishes = [e for e in log if e[0] == "finish"]
    ids = {q.request_id for q in trace}
    assert sorted(e[1] for e in firsts) == sorted(ids) == sorted(e[1] for e in finishes)
    # ... on the replica that served it,
    assert all(r.per_request[e[1]]["replica"] == e[2] for e in firsts + finishes)
    # ... its first token before its last,
    order = {(e[0], e[1]): n for n, e in enumerate(log)}
    assert all(order[("first", i)] <= order[("finish", i)] for i in ids)
    # ... and at the time it happens: at each arrival the router has heard of exactly the first
    # tokens that came earlier, never of one still inside a running step.
    first_at = {q.request_id: q.arrival_ts + r.per_request[q.request_id]["ttft_ms"] / 1000.0
                for q in trace}
    arrivals = {q.request_id: q.arrival_ts for q in trace}
    for e in (e for e in log if e[0] == "arrive"):
        t = arrivals[e[1]]
        earlier = sum(1 for v in first_at.values() if v < t - 1e-9)
        not_later = sum(1 for v in first_at.values() if v <= t + 1e-9)
        assert earlier <= e[2] <= not_later


def test_the_event_view_is_the_engines_cache_and_reading_it_changes_nothing(recorder):
    # Sessions far apart and one turn at a time: each request is admitted the moment it arrives,
    # so what the router is told at the decision is what the engine then finds.
    trace = _traffic(n_sessions=12, arrival_rate=0.05, think_time_s=30.0)
    seen = run(trace, policy=recorder.name, n_replicas=2, seed=0, profile=SMALL,
               record_per_request=True)
    told = {e[1]: e[3] for e in recorder.log if e[0] == "arrive"}
    assert any(told.values())  # the scenario means something only if there were hits
    assert all(seen.per_request[i]["cached_tokens"] == told[i] * SMALL.block_size for i in told)
    plain = run(trace, policy="round-robin", n_replicas=2, seed=0, profile=SMALL,
                record_per_request=True)
    a, b = seen.summary(), plain.summary()
    assert {k: v for k, v in a.items() if k != "policy"} == {k: v for k, v in b.items()
                                                             if k != "policy"}
    assert seen.per_request == plain.per_request


def test_a_policy_that_does_not_ask_is_given_no_view_of_the_caches():
    seen = []

    class Peek(RoundRobin):
        name = "test-peek"

        def choose(self, state, request, rng):
            seen.append(state.hit_blocks)
            return super().choose(state, request, rng)

    registry.REGISTRY[Peek.name] = Peek
    try:
        run(_traffic(n_sessions=4), policy=Peek.name, n_replicas=2, profile=SMALL)
    finally:
        del registry.REGISTRY[Peek.name]
    assert seen and all(v is None for v in seen)
