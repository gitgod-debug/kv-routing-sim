"""End-to-end tests for the simulator, including the spec's prefix-affinity property test."""

from __future__ import annotations

import dataclasses
import math
import pathlib
import random

import pytest
import yaml

from kv_routing_sim.sim.cost_model import UNCALIBRATED_DEFAULT, CostProfile, load_profile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.sim.replica import Replica
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
    """The spec's property test: when every request is the same prompt, prefix routing cannot have
    a lower hit rate than round-robin."""
    blocks = tuple(range(1, 21))
    trace = [
        NormalizedRequest(f"r{i}", i * 0.05, blocks, 320, 8, f"s{i}") for i in range(200)
    ]
    rr = run(trace, policy="round-robin", n_replicas=4, seed=0, profile=SMALL)
    aff = run(trace, policy="radix-affinity", n_replicas=4, seed=0, profile=SMALL)
    assert aff.hit_rate >= rr.hit_rate


# A cache the traces below can never fill: the precondition the property needs.
ROOMY = dataclasses.replace(SMALL, name="test-roomy", kv_cache_blocks=20_000, max_num_seqs=256)


@pytest.mark.parametrize("case", range(24))
def test_prefix_affinity_never_loses_to_round_robin_without_memory_pressure(case):
    """Without memory pressure the property holds on any trace: prefix routing gets every hit an
    earlier request made possible, so it cannot fall below round-robin."""
    rng = random.Random(case)
    trace, _ = AgenticWorkload(
        n_sessions=rng.randint(4, 30),
        turns_per_session=rng.randint(1, 5),
        prefix_sharing_ratio=rng.choice((0.0, 0.2, 0.5, 0.8, 0.95)),
        n_shared_prefixes=rng.choice((1, 1, 4, 16)),
        arrival_rate=0.2,
        seed=case,
    ).generate()
    # Every distinct block the trace can ever put in a cache (its prompt blocks, plus each
    # request's own output blocks) fits on one replica at once, so nothing is ever evicted.
    ever = len({h for q in trace for h in q.block_hashes})
    ever += sum(q.output_tokens // ROOMY.block_size + 1 for q in trace)
    assert ever < ROOMY.kv_cache_blocks
    n = rng.randint(2, 6)
    rr = run(trace, policy="round-robin", n_replicas=n, seed=case, profile=ROOMY)
    aff = run(trace, policy="radix-affinity", n_replicas=n, seed=case, profile=ROOMY)
    assert rr.preemptions == aff.preemptions == 0  # the precondition really held
    assert aff.hit_rate >= rr.hit_rate


def test_the_property_needs_its_precondition():
    """Under memory pressure the property fails: on Phase 4's high-sharing trace at 4 sessions/s on
    five T4s, pure prefix-affinity overloads one replica and ends below round-robin's hit rate."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    p4 = yaml.safe_load((repo / "configs" / "phase4_sweep.yaml").read_text())
    trace, _ = AgenticWorkload(**p4["traces"]["high_sharing"], arrival_rate=4.0).generate()
    t4 = load_profile(repo / p4["profile"])

    def mean_hit(policy: str) -> float:
        runs = [run(trace, policy=policy, n_replicas=5, seed=s, profile=t4) for s in p4["seeds"]]
        return sum(r.hit_rate for r in runs) / len(runs)

    aff, rr = mean_hit("radix-affinity"), mean_hit("round-robin")
    assert aff == pytest.approx(0.815, abs=5e-4)
    assert rr == pytest.approx(0.832, abs=5e-4)
    assert aff < rr


def test_more_replicas_dilute_the_cache_under_round_robin():
    """Scattering a shared prefix across more caches lowers the hit rate."""
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


# The vLLM cache model under memory pressure: admission control and preemption, pinned to the
# scheduler rules of vLLM v0.30.0 (scheduler.py).


TINY = CostProfile(
    name="tiny",
    prefill_base_s=0.001,
    prefill_per_token_s=0.00001,
    decode_base_s=0.002,
    decode_per_seq_s=0.0001,
    kv_cache_blocks=24,
    block_size=16,
    max_num_seqs=8,
)


def _drive(replica: Replica, requests, max_steps=20_000):
    """Step a replica to completion, checking the pool's invariants after every step."""
    for r in requests:
        replica.enqueue(r)
    now, done, steps = 0.0, [], 0
    while replica.has_work:
        dur, fin = replica.step(now)
        now += dur
        done.extend(fin)
        replica.pool.check_consistency()
        assert replica.pool.stranded_hashes() == [], f"stranded at step {steps}"
        steps += 1
        assert steps < max_steps, "no progress — a scheduling deadlock"
    return done, now


def test_vllm_admission_waits_for_memory_instead_of_overcommitting():
    """A waiting request that does not fit stops admission (scheduler.py:1172-1178), so with 5
    blocks per request and 24 in the pool at most 4 run at once, not the 8 max_num_seqs allows."""
    rep = Replica(0, TINY, cache_model="vllm")
    reqs = [NormalizedRequest(f"r{i}", 0.0, tuple(range(i * 10, i * 10 + 4)), 70, 3, f"s{i}")
            for i in range(12)]
    for r in reqs:
        rep.enqueue(r)
    rep.step(0.0)
    assert len(rep.running) == 4  # 4 x 5 blocks = 20 of 24; a fifth would need 25
    assert len(rep.queue) == 8
    done, _ = _drive(rep, [])
    assert len(done) == 12  # including the four admitted above, which finish inside _drive


def test_vllm_preempts_under_decode_pressure_and_every_request_still_completes():
    """Long outputs outgrow the pool mid-decode, forcing FCFS preemption of the most recent
    request (scheduler.py:785). Every request must still finish, with a first token recorded."""
    rep = Replica(0, TINY, cache_model="vllm")
    reqs = [NormalizedRequest(f"r{i}", 0.0, tuple(range(i * 10, i * 10 + 2)), 40, 90, f"s{i}")
            for i in range(6)]
    done, _ = _drive(rep, reqs)
    assert len(done) == 6
    assert rep.preemptions > 0  # the scenario only means something if preemption really happened
    assert all(s.first_token_ts >= 0 for s in done)
    assert all(s.tokens_emitted == 90 for s in done)
    # A preempted request's later lookups go to separate counters, as in vLLM's stats.
    assert rep.pool.preempted_queries > 0
    assert rep.pool.queries == sum(r.prompt_tokens for r in reqs)  # first lookups only


def test_vllm_invariants_hold_under_concurrent_shared_prefix_traffic():
    """Agentic traffic on a pool far too small (shared prefixes, eviction, preemption) never leaves
    a block double-booked or stranded after any step."""
    trace, _ = AgenticWorkload(n_sessions=30, turns_per_session=4, seed=0,
                               prefix_sharing_ratio=0.6).generate()
    big = CostProfile(
        name="pressured", prefill_base_s=0.001, prefill_per_token_s=1e-5, decode_base_s=0.002,
        decode_per_seq_s=1e-4, kv_cache_blocks=max(-(-(r.prompt_tokens + r.output_tokens) // 16)
                                                  for r in trace) + 40,
        block_size=16, max_num_seqs=16,
    )
    rep = Replica(0, big, cache_model="vllm")
    done, _ = _drive(rep, trace)
    assert len(done) == len(trace)
    assert rep.pool.real_evictions > 0


def test_vllm_refuses_a_request_that_could_never_fit():
    """Like vLLM, the run refuses to start when one max-length request cannot fit, rather than
    deadlock mid-run."""
    huge = NormalizedRequest("huge", 0.0, tuple(range(100)), 1600, 200, "s")
    with pytest.raises(ValueError, match="vLLM would refuse to start"):
        run([huge], policy="round-robin", n_replicas=1, profile=TINY, cache_model="vllm")


def test_lru_model_never_preempts_and_never_blocks():
    """The LRU ablation baseline has no memory-based admission and no preemption."""
    rep = Replica(0, TINY, cache_model="lru")
    reqs = [NormalizedRequest(f"r{i}", 0.0, tuple(range(i * 10, i * 10 + 4)), 70, 3, f"s{i}")
            for i in range(12)]
    for r in reqs:
        rep.enqueue(r)
    rep.step(0.0)
    assert len(rep.running) == 8  # max_num_seqs, regardless of memory
    assert rep.preemptions == 0


@pytest.mark.parametrize("cache_model", ["vllm", "lru"])
def test_both_models_complete_every_request_on_agentic_traffic(cache_model):
    trace = _trace()
    r = run(trace, policy="radix-affinity-balanced", n_replicas=4, profile=SMALL,
            cache_model=cache_model)
    assert r.completed == len(trace)


def test_context_terms_follow_the_stated_formulas():
    """prefill(n, c) = base + per_token*n + attn*n*(c + n/2); decode(b, S) = base + per_seq*b +
    per_ctx*S. With both context terms at zero, this is the context-free model."""
    p = dataclasses.replace(SMALL, prefill_attn_s=1e-8, decode_per_ctx_token_s=2e-6)
    assert p.prefill_s(100, 400) == pytest.approx(0.001 + 0.00001 * 100 + 1e-8 * 100 * 450)
    assert p.decode_s(4, 2_000) == pytest.approx(0.002 + 0.0001 * 4 + 2e-6 * 2_000)
    assert p.context_aware and not SMALL.context_aware
    assert SMALL.prefill_s(100, 400) == SMALL.prefill_s(100)  # no context term: ignored
    assert SMALL.decode_s(4, 2_000) == SMALL.decode_s(4)


def test_long_contexts_cost_more_only_with_the_context_term():
    """The same requests with long prompts run slower than with short ones when the profile has a
    context term, and cost exactly the same per step without one (the decode side)."""
    def trace(prompt_tokens):
        blocks = prompt_tokens // 16
        return [NormalizedRequest(f"r{i}", 0.0, tuple(range(i * 10_000, i * 10_000 + blocks)),
                                  prompt_tokens, 32, f"s{i}") for i in range(8)]

    aware = dataclasses.replace(SMALL, decode_per_ctx_token_s=2e-6)
    short, long_ = trace(160), trace(1_600)
    flat = [run(t, policy="round-robin", n_replicas=1, profile=SMALL).makespan_s
            for t in (short, long_)]
    ctx = [run(t, policy="round-robin", n_replicas=1, profile=aware).makespan_s
           for t in (short, long_)]
    assert ctx[1] - ctx[0] > flat[1] - flat[0] + 0.01  # the context term adds real time
    assert ctx[0] > flat[0]


# Time per output token

# Round numbers, so every expected value below can be worked out by hand.
STEPPY = CostProfile(name="steppy", prefill_base_s=0.0, prefill_per_token_s=0.001,
                     decode_base_s=0.010, decode_per_seq_s=0.002, kv_cache_blocks=5_000,
                     block_size=16, max_num_seqs=16)


def test_a_lone_requests_time_per_token_is_its_decode_step():
    """32 cold tokens, 5 output tokens, alone: the first token after prefill + one step (44 ms),
    then four steps of 12 ms each. TPOT is the mean of those four gaps and nothing else."""
    req = NormalizedRequest("a", 0.0, (1, 2), 32, 5, "s")
    r = run([req], policy="round-robin", n_replicas=1, profile=STEPPY, record_per_request=True)
    assert r.ttft_ms == pytest.approx([44.0])
    assert r.tpot_ms == pytest.approx([12.0])
    assert r.per_request["a"]["tpot_ms"] == pytest.approx(12.0)
    assert (r.itl_count, r.itl_sum_s) == (4, pytest.approx(0.048))
    assert r.engine_itl_mean_ms() == pytest.approx(12.0)


def test_another_requests_prefill_stalls_the_tokens_of_everyone_decoding():
    """B's prefill (48 ms) shares a step with A's next token, so that gap of A's is 62 ms, not 12;
    a first-token SLO cannot see this.

        A: first token at 44 ms; gaps 12, 62 (B's prefill + a step of two), 14, 14 -> 25.5 ms/token
        B: sent at 50 ms, first token at 118 ms (TTFT 68); gaps 14, 14 -> 14 ms/token
    """
    a = NormalizedRequest("a", 0.0, (1, 2), 32, 5, "sa")
    b = NormalizedRequest("b", 0.050, (7, 8, 9), 48, 3, "sb")
    r = run([a, b], policy="round-robin", n_replicas=1, profile=STEPPY, record_per_request=True)
    assert r.per_request["a"]["ttft_ms"] == pytest.approx(44.0)
    assert r.per_request["b"]["ttft_ms"] == pytest.approx(68.0)
    assert r.per_request["a"]["tpot_ms"] == pytest.approx(25.5)
    assert r.per_request["b"]["tpot_ms"] == pytest.approx(14.0)
    # The engine-style counters: one observation per token after a request's first.
    assert r.itl_count == 4 + 2
    assert r.engine_itl_mean_ms() == pytest.approx((102.0 + 28.0) / 6)
    pct = r.tpot_percentiles()
    assert set(pct) == {"p50_ms", "p95_ms", "p99_ms", "max_ms", "n"} and pct["n"] == 2
    assert r.summary()["tpot"] == pct


def test_a_single_token_request_has_no_time_per_token():
    """One output token is one point in time: no interval. It is in no percentile, it adds
    nothing to the engine-style counters, and its place in the list keeps TPOT aligned with TTFT.
    """
    one = NormalizedRequest("one", 0.0, (1, 2), 32, 1, "s1")
    five = NormalizedRequest("five", 1.0, (5, 6), 32, 5, "s5")
    r = run([one, five], policy="round-robin", n_replicas=1, profile=STEPPY,
            record_per_request=True)
    assert len(r.tpot_ms) == len(r.ttft_ms) == 2
    assert math.isnan(r.tpot_ms[0])
    assert r.per_request["one"]["tpot_ms"] is None
    assert r.tpot_percentiles()["n"] == 1 and r.itl_count == 4
    alone = run([one], policy="round-robin", n_replicas=1, profile=STEPPY)
    assert alone.itl_count == 0 and math.isnan(alone.engine_itl_mean_ms())
    assert alone.tpot_percentiles()["n"] == 0


def test_a_stall_while_preempted_is_inside_the_time_per_token():
    """As in vLLM (stats.py:549-551), time spent preempted during decode counts toward the
    request's TPOT."""
    reqs = [NormalizedRequest(f"r{i}", 0.0, tuple(range(i * 10, i * 10 + 2)), 40, 90, f"s{i}")
            for i in range(6)]
    r = run(reqs, policy="round-robin", n_replicas=1, profile=TINY, record_per_request=True)
    assert r.preemptions > 0 and r.completed == 6
    hit = [v["tpot_ms"] for v in r.per_request.values() if v["preemptions"]]
    spared = [v["tpot_ms"] for v in r.per_request.values() if not v["preemptions"]]
    assert hit and spared
    # No single step here is longer than one full recomputation plus a step of six (~4 ms); a
    # preempted request's mean gap is still above every unpreempted one's.
    assert min(hit) > max(spared)
    # Two accountings agree: summed per-request TPOT x gaps equals the counters' sum.
    total = sum(v["tpot_ms"] * 89 for v in r.per_request.values()) / 1000.0
    assert total == pytest.approx(r.itl_sum_s) and r.itl_count == 6 * 89


def test_time_per_token_respects_warm_up_and_ttft_does_not_move():
    """Warm-up drops early requests' TPOT as it does their TTFT, while the engine-style counters,
    like the cache's, count the whole run."""
    trace = _trace()
    r = run(trace, policy="session-sticky", n_replicas=2, profile=SMALL, warmup_requests=25)
    assert len(r.tpot_ms) == len(r.ttft_ms) == len(trace) - 25
    assert r.itl_count == sum(q.output_tokens - 1 for q in trace)
