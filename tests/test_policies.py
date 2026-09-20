"""Policy behaviour, including the property test the spec names explicitly."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from kv_routing_sim import policies
from kv_routing_sim.policies.base import FleetState
from kv_routing_sim.traces.schema import NormalizedRequest

REQ = NormalizedRequest("r1", 0.0, (11, 22, 33, 44), 64, 20, "sess-A")


def _state(running, queued=None):
    return FleetState(len(running), list(running), list(queued or [0] * len(running)))


@pytest.mark.parametrize("name", policies.ALL)
def test_every_policy_returns_a_valid_replica(name):
    p = policies.build(name, 4)
    rng = random.Random(0)
    for _ in range(50):
        idx = p.choose(_state([1, 2, 3, 4]), REQ, rng)
        assert 0 <= idx < 4
        p.observe(idx, REQ)


@pytest.mark.parametrize("name", policies.ALL)
def test_every_policy_is_deterministic_given_a_seed(name):
    def run():
        p = policies.build(name, 4)
        rng = random.Random(7)
        out = []
        for _ in range(30):
            i = p.choose(_state([2, 0, 5, 1]), REQ, rng)
            p.observe(i, REQ)
            out.append(i)
        return out

    assert run() == run()


def test_ties_are_broken_randomly_not_by_index():
    """A min() over a list would always pick replica 0 and fabricate locality that is not there."""
    ll = policies.build("least-loaded", 4)
    counts = Counter(ll.choose(_state([0, 0, 0, 0]), REQ, random.Random(s)) for s in range(400))
    assert len(counts) == 4
    assert min(counts.values()) > 400 / 4 * 0.5  # roughly uniform, generously bounded


def test_round_robin_cycles():
    rr = policies.build("round-robin", 3)
    rng = random.Random(0)
    assert [rr.choose(_state([0, 0, 0]), REQ, rng) for _ in range(6)] == [0, 1, 2, 0, 1, 2]


def test_least_loaded_picks_the_least_loaded():
    ll = policies.build("least-loaded", 4)
    assert ll.choose(_state([5, 5, 1, 5]), REQ, random.Random(0)) == 2


def test_radix_affinity_follows_the_prefix_over_the_load():
    """Locality wins over load — the behaviour that makes H2's crossover possible."""
    p = policies.build("radix-affinity", 4)
    p.observe(3, REQ)
    assert p.choose(_state([0, 0, 0, 99]), REQ, random.Random(0)) == 3


def test_radix_affinity_tracks_multiple_owners_per_block():
    """A block lives on every replica that served it. Exclusive ownership made the router thrash."""
    p = policies.build("radix-affinity", 4)
    p.observe(1, REQ)
    p.observe(2, REQ)
    chosen = {p.choose(_state([0, 0, 0, 0]), REQ, random.Random(s)) for s in range(40)}
    assert chosen <= {1, 2}
    assert len(chosen) == 2  # both are valid targets; it does not collapse onto one


def test_radix_affinity_spreads_unseen_prefixes():
    p = policies.build("radix-affinity", 4)
    unseen = NormalizedRequest("r2", 0.0, (777, 888), 64, 20, "sess-B")
    assert p.choose(_state([9, 0, 9, 9]), unseen, random.Random(0)) == 1


def test_session_sticky_is_sticky_and_falls_back_when_anonymous():
    p = policies.build("session-sticky", 4)
    rng = random.Random(0)
    first = p.choose(_state([0, 0, 0, 0]), REQ, rng)
    assert all(p.choose(_state([9, 9, 9, 9]), REQ, rng) == first for _ in range(10))

    anon = NormalizedRequest("r3", 0.0, (1, 2), 64, 20, None)
    assert p.choose(_state([5, 5, 0, 5]), anon, rng) == 2


def test_consistent_hash_sends_identical_prefixes_to_one_replica():
    p = policies.build("consistent-hash-prefix", 4)
    rng = random.Random(0)
    picks = {p.choose(_state([0, 0, 0, 0]), REQ, rng) for _ in range(20)}
    assert len(picks) == 1


def test_unknown_policy_fails_loudly():
    with pytest.raises(ValueError, match="unknown policy"):
        policies.build("does-not-exist", 4)
    with pytest.raises(ValueError, match="unknown policies"):
        policies.resolve(["round-robin", "nope"])
    assert policies.resolve("all") == list(policies.ALL)
