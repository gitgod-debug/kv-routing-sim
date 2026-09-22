"""The full simulator against numbers MEASURED on a live engine.

Source: artifacts/results/phase0_kaggle_t4/ (v1, vLLM 0.29.0) and phase0_v2_kaggle_t4/ (v2, vLLM
0.30.0) — Qwen/Qwen3-0.6B, block_size 16, Kaggle 2x Tesla T4.

These replay the exact workloads the notebooks sent, through `engine.run`, and demand the engine's
counters to the token. A failure means the simulator has drifted from real vLLM behaviour.

⚠️ What these do and do not prove: v1/v2 never filled the cache and sent requests one at a time. So
they pin ACCOUNTING — the token unit, the partial-block rule, the lookup cap, per-replica cold
prefixes, and the eviction-event quirk — not eviction under pressure and not batching.
"""

from __future__ import annotations

import pytest

from kv_routing_sim.sim.cost_model import CostProfile
from kv_routing_sim.sim.engine import run
from kv_routing_sim.traces.schema import NormalizedRequest

# Engine facts from the v2 logs: "GPU KV cache size: 94,496 tokens" (the caching-on arm) and
# 99,024 tokens (each replica). Timings are irrelevant to these counters; any valid profile will do.
BS = 16
T4_06B_ON = CostProfile(
    name="t4-qwen3-0.6b-v2-on", prefill_base_s=0.001, prefill_per_token_s=1e-5,
    decode_base_s=0.002, decode_per_seq_s=1e-4, kv_cache_blocks=94_496 // BS,
    block_size=BS, max_num_seqs=256,
)
T4_06B_REPLICA = CostProfile(
    name="t4-qwen3-0.6b-v2-replica", prefill_base_s=0.001, prefill_per_token_s=1e-5,
    decode_base_s=0.002, decode_per_seq_s=1e-4, kv_cache_blocks=99_024 // BS,
    block_size=BS, max_num_seqs=256,
)

PREFIX = tuple(range(1, 33))  # 512 tokens = 32 full blocks, shared by every request
GAP = 10.0  # seconds between arrivals: the notebooks sent each request after the previous returned


def _caching_on_arm() -> list[NormalizedRequest]:
    """1 cold probe (512 tokens, max_tokens=4); then 64 x (prefix + 1 token, max_tokens=8)."""
    reqs = [NormalizedRequest("probe", 0.0, PREFIX, 512, 4, "probe")]
    reqs += [NormalizedRequest(f"w{i}", GAP * (i + 1), PREFIX, 513, 8, f"w{i}") for i in range(64)]
    return reqs


def _round_robin_arm() -> list[NormalizedRequest]:
    """64 x (a FRESH prefix + 1 unique token, max_tokens=8), alternated across two replicas."""
    fresh = tuple(range(1001, 1033))
    return [NormalizedRequest(f"rr{i}", GAP * i, fresh, 513, 8, f"rr{i}") for i in range(64)]


@pytest.mark.parametrize("cache_model", ["vllm", "lru"])
def test_caching_on_arm_counters_match_the_engine_exactly(cache_model):
    """v1 AND v2 both measured 33,344 / 32,768 / 576 — on different vLLM versions and machines."""
    r = run(_caching_on_arm(), policy="round-robin", n_replicas=1, profile=T4_06B_ON,
            cache_model=cache_model)
    assert r.cache_queries == 33_344  # vllm:prefix_cache_queries_total
    assert r.cache_hits == 32_768  # vllm:prefix_cache_hits_total
    assert r.cache_queries - r.cache_hits == 576  # prompt_tokens_by_source{local_compute}
    assert r.completed == 65


@pytest.mark.parametrize("cache_model", ["vllm", "lru"])
def test_round_robin_two_replicas_matches_the_engine_exactly(cache_model):
    """v2 §7: predicted 1,088 / 0.966862 before the run; the engine measured exactly that."""
    r = run(_round_robin_arm(), policy="round-robin", n_replicas=2, profile=T4_06B_REPLICA,
            cache_model=cache_model)
    assert r.cache_queries - r.cache_hits == 1_088
    assert r.hit_rate == pytest.approx(0.966862, abs=1e-6)
    assert r.per_replica_requests == [32, 32]


def test_vllm_model_reproduces_the_engines_eviction_histograms():
    """⭐ The engine's odd histograms, reproduced by the model — quirk included.

    v2 measured kv_block_* counts of 64 (caching-on arm) with the cache never full: every one a
    request's partial tail block being recycled LIFO, which vLLM reports as an "eviction" because
    `on_block_evicted` runs before the hash check (block_pool.py:739 vs 741). Lifetime equal to
    idle time (the block is never hit), zero reuse gaps. The lru model cannot produce these at all.
    """
    r = run(_caching_on_arm(), policy="round-robin", n_replicas=1, profile=T4_06B_ON,
            cache_model="vllm", track_block_metrics=True)
    ev = r.eviction_events
    assert len(ev) == 64  # engine: kv_block_lifetime_seconds_count = 64
    assert all(not e.was_cached for e in ev)  # none is a real eviction
    assert all(e.idle_s == pytest.approx(e.lifetime_s) for e in ev)  # never touched
    assert sum(len(e.reuse_gaps_s) for e in ev) == 0  # engine: reuse_gap count = 0
    assert r.evictions == 0  # and no real eviction happened


def test_vllm_model_reproduces_the_per_replica_histograms():
    """v2 §7: 31 samples on EACH replica — every request but the first recycles its predecessor's
    partial block; the first starts cold and recycles nothing."""
    trace = _round_robin_arm()
    per_replica = []
    for rep in (0, 1):
        sub = [r for i, r in enumerate(trace) if i % 2 == rep]  # round-robin's split
        res = run(sub, policy="round-robin", n_replicas=1, profile=T4_06B_REPLICA,
                  cache_model="vllm", track_block_metrics=True)
        per_replica.append(len(res.eviction_events))
    assert per_replica == [31, 31]  # engine: 31 and 31


def test_lookup_cap_is_applied_by_both_models():
    """kv_cache_manager.py:295. A repeated, block-ALIGNED prompt recomputes its last block.
    v1/v2 never exercised this (their prompts were 513 tokens), so it is pinned separately."""
    reqs = [NormalizedRequest(f"a{i}", GAP * i, PREFIX, 512, 2, f"a{i}") for i in range(3)]
    for model in ("vllm", "lru"):
        r = run(reqs, policy="round-robin", n_replicas=1, profile=T4_06B_ON, cache_model=model)
        # first: cold. Then each repeat hits 31 blocks (496 tokens), not 32.
        assert r.cache_hits == 2 * 496, model
