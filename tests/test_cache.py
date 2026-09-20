"""The cache model. `docs/05-codebase-map.md`: "the tests that matter most live here"."""

from __future__ import annotations

import pytest

from kv_routing_sim.cache.block_trie import BlockCache
from kv_routing_sim.cache.capacity import SHAPES, kv_blocks_for


def test_known_sequence_gives_known_hit_miss_pattern():
    c = BlockCache(capacity_blocks=100)
    a = (1, 2, 3, 4)
    b = (1, 2, 3, 4, 5, 6)
    assert c.query(a) == 0  # cold
    assert c.query(a) == 4  # fully warm
    assert c.query(b) == 4  # shares a's prefix, computes 5 and 6
    assert c.query((9, 9, 9)) == 0  # unrelated
    assert c.query(b) == 6  # now fully warm
    assert c.queries == 4 + 4 + 6 + 3 + 6
    assert c.hits == 0 + 4 + 4 + 0 + 6


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
    c.query((1, 2))
    c.query((1, 2))
    assert c.hit_rate == pytest.approx(2 / 4)


def test_capacity_rejects_nonsense():
    with pytest.raises(ValueError):
        BlockCache(capacity_blocks=0)
    with pytest.raises(ValueError, match="no room"):
        kv_blocks_for(SHAPES["llama3-8b"], gpu_memory_gib=4.0)


def test_llama3_8b_kv_is_128kb_per_token():
    """An independent check against a published figure, not against our own arithmetic."""
    assert SHAPES["llama3-8b"].bytes_per_token() == 128 * 1024
