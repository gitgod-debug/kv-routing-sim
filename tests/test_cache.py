"""The cache model. `docs/05-codebase-map.md`: "the tests that matter most live here"."""

from __future__ import annotations

import pytest

from kv_routing_sim.cache.block_trie import BlockCache
from kv_routing_sim.cache.capacity import SHAPES, kv_blocks_for


def test_known_sequence_gives_known_hit_miss_pattern():
    """These prompts are block-ALIGNED (no prompt_tokens given = exactly len x 16), so vLLM's
    lookup cap applies: a fully cached prompt recomputes its last block (kv_cache_manager.py:295).
    Before 2026-09-22 this test expected 4 and 6 — the model then ignored the cap."""
    c = BlockCache(capacity_blocks=100)
    a = (1, 2, 3, 4)
    b = (1, 2, 3, 4, 5, 6)
    assert c.query(a) == 0  # cold
    assert c.query(a) == 3  # fully cached — but the last block is always recomputed
    assert c.query(b) == 4  # shares a's four blocks; the cap (5) does not bite
    assert c.query((9, 9, 9)) == 0  # unrelated
    assert c.query(b) == 5  # fully cached again: 6 blocks, the last recomputed
    assert c.queries_blocks == 4 + 4 + 6 + 3 + 6
    assert c.hits_blocks == 0 + 3 + 4 + 0 + 5
    # Token counters are the block counters x block_size when prompts are block-aligned.
    assert c.queries == c.queries_blocks * 16
    assert c.hits == c.hits_blocks * 16


def test_stop_at_first_miss_is_the_physics():
    """An evicted block strands everything after it, even though those blocks are still resident.

    You cannot attend over a prefix with a hole in it, so a lookup that counted all matching
    blocks regardless of order would inflate every hit rate in the project.
    """
    c = BlockCache(capacity_blocks=3)
    c.query((10, 11, 12))  # fills the cache
    c.query((20,))  # evicts the LRU block, which is 10
    assert c.lookup((10, 11, 12)) == 0  # 11 and 12 are resident but unreachable
    assert len(c) == 3  # ...and they still occupy capacity


def test_lru_order_follows_the_chain():
    """A prefix is always at least as recently used as what follows it."""
    c = BlockCache(capacity_blocks=4)
    c.query((1, 2, 3, 4))
    c.query((1, 2))  # touches 1 and 2
    c.query((5, 6))  # must evict the two least-recently-used, i.e. 3 and 4
    assert c.lookup((1, 2)) == 2
    assert c.lookup((1, 2, 3)) == 2


def test_hit_rate_is_derived_from_two_counters():
    """ADR-008: never a stored gauge."""
    c = BlockCache(capacity_blocks=10)
    assert c.hit_rate == 0.0  # no division by zero on an untouched cache
    c.query((1, 2))  # 32 tokens, cold
    c.query((1, 2))  # fully cached: the cap leaves 1 of 2 blocks hittable
    assert c.hits == 16 and c.queries == 64
    assert c.hit_rate == pytest.approx(16 / 64)


def test_capacity_rejects_nonsense():
    with pytest.raises(ValueError):
        BlockCache(capacity_blocks=0)
    with pytest.raises(ValueError, match="no room"):
        kv_blocks_for(SHAPES["llama3-8b"], gpu_memory_gib=4.0)


def test_llama3_8b_kv_is_128kb_per_token():
    """An independent check against a published figure, not against our own arithmetic."""
    assert SHAPES["llama3-8b"].bytes_per_token() == 128 * 1024


# ─────────────────────────────────────────────────────────────────────────────────────────────
# Calibration regression tests: these reproduce COUNTERS MEASURED ON A LIVE ENGINE.
# Source: artifacts/results/phase0_kaggle_t4/ — vLLM 0.29.0, Qwen/Qwen3-0.6B, block_size 16, 2x T4.
# If one of these fails, the cache model has drifted from real vLLM behaviour.
# ─────────────────────────────────────────────────────────────────────────────────────────────


def test_counts_tokens_not_blocks_like_vllm():
    """vLLM 0.29.0: 512 tokens on a cold cache moved prefix_cache_queries by 512, not 32.

    Its own HELP text agrees: "Prefix cache queries, in terms of number of queried tokens."
    """
    c = BlockCache(capacity_blocks=1000, block_size=16)
    blocks = tuple(range(32))  # 512 tokens
    c.query(blocks, prompt_tokens=512)
    assert c.queries == 512
    assert c.queries_blocks == 32


def test_partial_trailing_block_is_in_the_denominator_but_never_hits():
    """513-token prompt, 32 full blocks all cached: vLLM reports 512/513, not 32/32.

    Before 2026-09-21 we counted full blocks on both sides and reported 1.0 — biased upward.
    """
    c = BlockCache(capacity_blocks=1000, block_size=16)
    blocks = tuple(range(32))
    c.query(blocks, prompt_tokens=512)  # warm the prefix
    c.reset_counters()
    c.query(blocks, prompt_tokens=513)  # same 32 blocks + 1 partial token
    assert c.hits == 512
    assert c.queries == 513
    assert c.hit_rate == pytest.approx(512 / 513)


def test_short_prompts_are_where_the_old_bias_was_worst():
    """20 tokens = one 16-token block + 4 partial. vLLM: 16/20. The old model said 16/16."""
    c = BlockCache(capacity_blocks=10, block_size=16)
    c.query((7,), prompt_tokens=20)
    c.reset_counters()
    c.query((7,), prompt_tokens=20)
    assert c.hit_rate == pytest.approx(16 / 20)


def test_reproduces_the_live_engine_counters_exactly():
    """⭐ The Phase-0 spike workload, replayed through our model, against vLLM's real counters.

    Measured on vLLM 0.29.0 (caching on): prefix_cache_queries_total = 33344,
    prefix_cache_hits_total = 32768, prompt_tokens_by_source{local_compute} = 576.

    The workload: 1 cold probe of 512 tokens, then 64 requests of the same 512-token prefix plus
    one unique token. Our model must produce the same three numbers to the token.

    ⚠️ What this does and does not prove: it validates the ACCOUNTING — unit, denominator, the
    partial-block rule — on a trivial workload with no eviction. It says nothing about eviction
    order, which is where H3/H4 live. It is the first calibration point, not the calibration.
    """
    c = BlockCache(capacity_blocks=1574, block_size=16)  # the engine's KV blocks at 0.42 util
    prefix = tuple(range(32))  # 512 tokens
    c.query(prefix, prompt_tokens=512)  # the cold probe
    for _ in range(64):
        c.query(prefix, prompt_tokens=513)  # shared prefix + 1 unique token (a partial block)

    assert c.queries == 33344  # vllm:prefix_cache_queries_total
    assert c.hits == 32768  # vllm:prefix_cache_hits_total
    assert c.queries - c.hits == 576  # vllm:prompt_tokens_by_source{source="local_compute"}
