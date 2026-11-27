"""A replica runs one step at a time, and the router and the metrics see each step end when it ends.

Release 0.1.2 broke this when a step finished all the work a replica had: the next arrival
started a step at once, the router saw the replica idle while it computed, and completions were
counted when their last step started. `engine_version="0.1.2"` keeps that engine for the record.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from kv_routing_sim.sim.cost_model import CostProfile, resolve_profile
from kv_routing_sim.sim.engine import ENGINE_VERSIONS, run, use_engine
from kv_routing_sim.sim.replica import Replica
from kv_routing_sim.traces.generator import AgenticWorkload
from kv_routing_sim.traces.schema import NormalizedRequest

# Round numbers: prefill 1 ms per token, a decode step 10 ms plus 2 ms per sequence.
STEPPY = CostProfile(name="steppy", prefill_base_s=0.0, prefill_per_token_s=0.001,
                     decode_base_s=0.010, decode_per_seq_s=0.002, kv_cache_blocks=5_000,
                     block_size=16, max_num_seqs=16)
PROMPT = tuple(range(1, 63))  # the 62 full blocks of a 1,000-token prompt


def _a_and_b(a_output_tokens: int) -> list[NormalizedRequest]:
    """A: 1,000 cold tokens at 0 s, its first step lasting until 1.012 s. B: 32 tokens at 0.1 s."""
    return [NormalizedRequest("a", 0.0, PROMPT, 1000, a_output_tokens, "sa"),
            NormalizedRequest("b", 0.100, (901, 902), 32, 1, "sb")]


@pytest.mark.parametrize("a_output_tokens", [1, 2])
def test_an_arrival_during_a_step_waits_for_its_end(a_output_tokens):
    r = run(_a_and_b(a_output_tokens), policy="round-robin", n_replicas=1, profile=STEPPY,
            record_per_request=True)
    a_first = r.per_request["a"]["ttft_ms"] / 1000.0
    b_first = 0.100 + r.per_request["b"]["ttft_ms"] / 1000.0
    assert a_first == pytest.approx(1.012)
    assert b_first > a_first  # B's step starts when A's ends, whether or not A is finished


def test_a_replica_never_starts_a_step_before_its_last_one_ends(monkeypatch):
    steps: dict[int, list[tuple[float, float]]] = {}
    step = Replica.step

    def logged(self, now):
        duration, finished = step(self, now)
        steps.setdefault(self.index, []).append((now, now + duration))
        return duration, finished

    monkeypatch.setattr(Replica, "step", logged)
    # One-token answers: every step is a request's last, the case 0.1.2 got wrong.
    trace, _ = AgenticWorkload(n_sessions=200, turns_per_session=4, output_tokens=1,
                               arrival_rate=16.0, seed=0, prefix_sharing_ratio=0.8).generate()
    run(trace, policy="least-loaded", n_replicas=2,
        profile=resolve_profile("t4-qwen3-4b-vllm0.30.0"))
    for replica, spans in steps.items():
        assert all(b[0] >= a[1] - 1e-12 for a, b in pairwise(spans)), replica
        work = sum(end - start for start, end in spans)
        assert work <= spans[-1][1] - spans[0][0] + 1e-9, replica  # no more work than wall time


@pytest.mark.parametrize("policy", ["least-loaded", "power-of-two", "session-sticky",
                                    "radix-affinity", "radix-affinity-balanced", "lmetric",
                                    "smetric-local"])
def test_the_router_sees_a_replica_busy_until_its_last_step_ends(policy):
    for seed in range(40):
        r = run(_a_and_b(1), policy=policy, n_replicas=2, seed=seed, profile=STEPPY,
                record_per_request=True)
        assert r.per_request["a"]["replica"] != r.per_request["b"]["replica"], seed


def test_warmup_drops_the_first_completions_in_time():
    # A's only step ends at 1.012 s. B arrives at 0.5 s on the other replica and ends at 0.544 s.
    a = NormalizedRequest("a", 0.0, PROMPT, 1000, 1, "sa")
    b = NormalizedRequest("b", 0.5, (901, 902), 32, 1, "sb")
    r = run([a, b], policy="round-robin", n_replicas=2, profile=STEPPY, warmup_requests=1)
    assert r.ttft_ms == [pytest.approx(1012.0)]  # B finished first, so B is the one dropped


def test_a_repeated_request_id_is_refused():
    a = NormalizedRequest("same", 0.0, (1, 2, 3), 50, 4, "s")
    b = NormalizedRequest("same", 0.01, (1, 2, 3), 50, 4, "s")
    with pytest.raises(ValueError, match="appears more than once"):
        run([a, b], policy="round-robin", n_replicas=1, profile=STEPPY)


def test_the_engine_of_0_1_2_is_kept_for_the_record():
    r = run(_a_and_b(1), policy="round-robin", n_replicas=1, profile=STEPPY,
            record_per_request=True, engine_version="0.1.2")
    # The behaviour this release fixes: B started at once, on a replica busy until 1.012 s.
    assert 0.100 + r.per_request["b"]["ttft_ms"] / 1000.0 == pytest.approx(0.144)
    with use_engine("0.1.2"):
        assert run(_a_and_b(1), policy="round-robin", n_replicas=1, profile=STEPPY,
                   record_per_request=True).per_request == r.per_request
    fixed = run(_a_and_b(1), policy="round-robin", n_replicas=1, profile=STEPPY,
                record_per_request=True)
    assert fixed.per_request != r.per_request  # the block ends, and the current engine is back
    assert ENGINE_VERSIONS[0] == "0.2.0"
    with pytest.raises(ValueError, match="engine_version"):
        run([], policy="round-robin", n_replicas=1, engine_version="0.1.0")
    with pytest.raises(ValueError, match="engine_version"), use_engine("0.1.1"):
        pass


def test_short_answers_queue_on_a_busy_replica():
    trace, _ = AgenticWorkload(n_sessions=200, turns_per_session=4, output_tokens=1,
                               arrival_rate=16.0, seed=0, prefix_sharing_ratio=0.8).generate()
    profile = resolve_profile("t4-qwen3-4b-vllm0.30.0")
    now = run(trace, policy="round-robin", n_replicas=1, profile=profile)
    before = run(trace, policy="round-robin", n_replicas=1, profile=profile,
                 engine_version="0.1.2")
    # 800 one-token requests at 16 sessions a second are more than one T4 can serve: they queue.
    assert now.ttft_percentiles()["p95_ms"] > 100 * before.ttft_percentiles()["p95_ms"]
