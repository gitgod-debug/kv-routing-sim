"""The vLLM-faithful block pool. Every test names the vLLM v0.30.0 line it pins.

If one of these fails, the pool has drifted from the engine's documented algorithm — which is a bug
in the pool, never a reason to edit the test.
"""

from __future__ import annotations

import random

import pytest

from kv_routing_sim.cache.block_pool import VllmBlockPool, max_hit_blocks

BS = 16

# ⚠️ vLLM reserves ONE block as the null block (block_pool.py:193), so `VllmBlockPool(n)` has n - 1
# usable. Pool sizes below are written as `usable + 1` so each test states the capacity it means.


def _admit(pool: VllmBlockPool, rid: str, hashes: tuple[int, ...], tokens: int, now: float = 0.0):
    hit = pool.lookup(pool.chain_for(rid, hashes), tokens)
    assert pool.admit(rid, hit, hashes, tokens, now)
    return hit


# ── lookup rules (shared by both cache models) ─────────────────────────────────────────────────


def test_fully_cached_aligned_prompt_recomputes_its_last_block():
    """kv_cache_manager.py:295 — max_cache_hit_length = num_tokens - 1, floored to a block."""
    assert max_hit_blocks(512, BS) == 31
    assert max_hit_blocks(513, BS) == 32
    assert max_hit_blocks(16, BS) == 0  # a single-block prompt can never hit
    assert max_hit_blocks(0, BS) == 0

    pool = VllmBlockPool(100, BS)
    prefix = tuple(range(32))
    _admit(pool, "a", prefix, 512)
    pool.free("a")
    assert len(pool.lookup(prefix, 512)) == 31  # not 32 — the whole last block is recomputed
    assert len(pool.lookup(prefix, 513)) == 32


def test_lookup_stops_at_first_miss_and_finds_free_blocks():
    """stm.py find_longest_cache_hit: longest PREFIX; cached blocks are found even when free."""
    pool = VllmBlockPool(100, BS)
    _admit(pool, "a", (1, 2, 3, 4), 4 * BS + 1)
    pool.free("a")  # ref 0, but hashes kept: freed is not evicted
    assert len(pool.lookup((1, 2, 3, 4), 4 * BS + 1)) == 4
    assert len(pool.lookup((1, 2, 9, 4), 4 * BS + 1)) == 2  # stops at the first miss


# ── the free queue ─────────────────────────────────────────────────────────────────────────────


def test_blocks_are_freed_tail_first_so_the_tail_is_evicted_first():
    """single_type_kv_cache_manager.py:580 — "Free blocks in reverse order so that the tail blocks
    are freed first." With the pool full of one request's freed blocks, the next allocation takes
    the TAIL, and the shared head survives — the opposite of plain LRU."""
    pool = VllmBlockPool(4 + 1, BS)
    _admit(pool, "a", (10, 11, 12, 13), 4 * BS)  # exactly 4 full blocks, all cached
    pool.free("a")
    _admit(pool, "b", (), BS)  # one unrelated block: must evict exactly one of a's
    assert pool.is_cached(10) and pool.is_cached(11) and pool.is_cached(12)
    assert not pool.is_cached(13)  # the tail went first
    assert pool.real_evictions == 1


def test_uncached_blocks_are_reused_first_lifo():
    """block_pool.py:789 — "LIFO reuse of non-cached blocks"; prepended to the front."""
    pool = VllmBlockPool(6 + 1, BS)
    _admit(pool, "a", (1, 2), 2 * BS + 5)  # 2 cached blocks + 1 partial (no hash)
    pool.free("a")
    # The partial block is at the FRONT, so the next allocation recycles it — not a cached block.
    _admit(pool, "b", (), BS)
    assert pool.is_cached(1) and pool.is_cached(2)
    assert pool.real_evictions == 0


def test_cached_blocks_are_evicted_fifo_by_free_time():
    """block_pool.py:792 — "FIFO reuse of cached blocks for LRU eviction behavior": the request
    freed FIRST is evicted first."""
    pool = VllmBlockPool(4 + 1, BS)
    _admit(pool, "old", (1, 2), 2 * BS)
    _admit(pool, "new", (3, 4), 2 * BS)
    pool.free("old")
    pool.free("new")
    _admit(pool, "c", (), 2 * BS)  # needs two blocks: both must come from "old"
    assert not pool.is_cached(1) and not pool.is_cached(2)
    assert pool.is_cached(3) and pool.is_cached(4)


def test_a_hit_moves_a_block_to_the_back():
    """block_pool.py `touch` removes a free block from the queue; freeing it again appends it."""
    pool = VllmBlockPool(4 + 1, BS)
    _admit(pool, "a", (1, 2), 2 * BS)
    _admit(pool, "b", (3, 4), 2 * BS)
    pool.free("a")
    pool.free("b")
    _admit(pool, "a2", (1, 2), 2 * BS + 1)  # hits 1, 2 (+ needs one more block for the extra token)
    pool.free("a2")
    _admit(pool, "c", (), 2 * BS)  # evicts from the front: 3/4's region, since 1/2 moved back
    assert pool.is_cached(1) and pool.is_cached(2)


# ── reference counts ───────────────────────────────────────────────────────────────────────────


def test_blocks_in_use_cannot_be_evicted():
    """Only ref_cnt == 0 blocks are in the free queue (block_pool.py `free_blocks`). A running
    request's blocks are pinned: allocation FAILS rather than evicting them."""
    pool = VllmBlockPool(4 + 1, BS)
    _admit(pool, "running", (1, 2, 3, 4), 4 * BS)  # all 4 blocks held
    assert not pool.can_admit([], BS)
    assert not pool.admit("x", [], (), BS, 0.0)
    assert all(pool.is_cached(h) for h in (1, 2, 3, 4))


def test_shared_block_is_freed_only_when_its_last_user_finishes():
    pool = VllmBlockPool(10, BS)
    _admit(pool, "a", (1, 2), 2 * BS + 1)
    _admit(pool, "b", (1, 2), 2 * BS + 1)  # hits a's blocks: ref 2
    free_before = pool.num_free
    pool.free("a")
    # a's shared blocks are still held by b; only a's own partial block came back
    assert pool.num_free == free_before + 1
    pool.check_consistency()


def test_admission_counts_free_hit_blocks_against_capacity():
    """Hitting a FREE block pulls it out of the free queue, so it cannot also back a new block."""
    pool = VllmBlockPool(3 + 1, BS)
    _admit(pool, "a", (1, 2), 2 * BS)
    pool.free("a")  # 3 free, two of them cached
    hit = pool.lookup((1, 2), 3 * BS + 1)  # hits both, needs 2 more new blocks (4 total)
    assert len(hit) == 2
    assert not pool.can_admit(hit, 3 * BS + 1)  # 2 new + 2 hit-from-free > 3 free


# ── capacity and decode ────────────────────────────────────────────────────────────────────────


def test_capacity_includes_partial_and_decode_blocks():
    """A 513-token prompt generating 8 tokens occupies 33 blocks, not 32."""
    pool = VllmBlockPool(100, BS)
    _admit(pool, "a", tuple(range(32)), 513)
    assert len(pool.blocks_of("a")) == 33
    for k in range(1, 8):
        assert pool.grow("a", 513 + k, 0.0)
    assert len(pool.blocks_of("a")) == 33  # 520 tokens still fit in 33 blocks
    assert pool.grow("a", 529, 0.0)
    assert len(pool.blocks_of("a")) == 34


def test_full_decode_blocks_are_cached():
    """kv_cache_manager.py:595-599 caches up to request.num_tokens, which INCLUDES output."""
    pool = VllmBlockPool(100, BS)
    _admit(pool, "a", tuple(range(2)), 2 * BS)
    cached_before = len(pool._by_hash)
    pool.grow("a", 3 * BS, 0.0)  # a whole block of generated tokens
    assert len(pool._by_hash) == cached_before + 1


def test_grow_fails_when_no_block_is_free():
    """The caller must then preempt (scheduler.py:726-800)."""
    pool = VllmBlockPool(2 + 1, BS)
    _admit(pool, "a", (1,), BS + 1)  # 2 blocks
    assert not pool.grow("a", 2 * BS + 1, 0.0)


# ── preemption accounting ──────────────────────────────────────────────────────────────────────


def test_preempted_lookups_go_to_separate_counters():
    """stats.py PrefixCacheStats.record: preempted requests never touch queries/hits."""
    pool = VllmBlockPool(100, BS)
    pool.record_lookup(513, 32, preempted=False)
    pool.record_lookup(520, 32, preempted=True)
    assert (pool.queries, pool.hits) == (513, 512)
    assert (pool.preempted_queries, pool.preempted_hits) == (520, 512)


def test_a_preempted_request_can_rehit_its_own_decode_blocks():
    """_preempt_request frees KV but keeps tokens; freed blocks keep their hashes."""
    pool = VllmBlockPool(100, BS)
    _admit(pool, "a", (1, 2), 2 * BS)
    pool.grow("a", 3 * BS + 1, 0.0)  # one full decode block, plus one more token
    pool.free("a")  # preempted — chain kept, not forgotten
    chain = pool.chain_for("a", (1, 2))
    assert len(chain) == 3  # two prompt blocks + one decode block
    assert len(pool.lookup(chain, 3 * BS + 1)) == 3


# ── eviction events, by vLLM's own rule ────────────────────────────────────────────────────────


def test_recycling_an_uncached_block_is_reported_as_an_eviction():
    """block_pool.py:739 fires the event before the hash check at 741 — the quirk we reproduce so
    H4 compares like with like."""
    pool = VllmBlockPool(10, BS, track_block_metrics=True)
    _admit(pool, "a", (1,), BS + 1, now=0.0)  # block 0 cached, block 1 partial
    pool.free("a")
    _admit(pool, "b", (), BS, now=1.0)  # recycles the partial block (LIFO)
    assert len(pool.eviction_events) == 1
    ev = pool.eviction_events[0]
    assert not ev.was_cached
    assert ev.lifetime_s == pytest.approx(1.0)
    assert ev.idle_s == pytest.approx(1.0)  # never touched: idle == lifetime
    assert ev.reuse_gaps_s == ()
    assert pool.real_evictions == 0


def test_reuse_gaps_keep_only_the_last_four_accesses():
    """kv_cache_metrics.py:24 — deque(maxlen=4): at most three gaps per block."""
    pool = VllmBlockPool(3 + 1, BS, track_block_metrics=True)
    _admit(pool, "a", (1,), BS, now=0.0)
    pool.free("a")
    for t in range(1, 7):  # six hits at t=1..6
        rid = f"r{t}"
        _admit(pool, rid, (1,), BS + 1, now=float(t))
        pool.free(rid)
    _admit(pool, "evictor", (), 3 * BS, now=10.0)  # takes every block, block 1's included
    ev = [e for e in pool.eviction_events if e.was_cached]
    assert len(ev) == 1
    assert ev[0].reuse_gaps_s == (1.0, 1.0, 1.0)  # only the LAST four accesses: t = 3, 4, 5, 6


# ── the invariant ──────────────────────────────────────────────────────────────────────────────


def test_no_block_is_ever_stranded_under_random_traffic():
    """⭐ Freeing tail-first + the prefix rule means a parent is never evicted before its child, so
    the pool never strands a block. A failure here is a bug in block_pool.py.

    For calibration of expectations: plain LRU on this same traffic strands at most ONE block of 64
    (measured 2026-09-22) — its head-first eviction strands briefly, not constantly, because a
    chain's tail is next in line once its head goes. The invariant is exact, but it is not by itself
    evidence of a large accuracy difference."""
    rng = random.Random(0)
    pool = VllmBlockPool(64, BS)
    prefixes = [tuple(rng.randrange(10**9) for _ in range(rng.randint(1, 12))) for _ in range(40)]
    live: list[str] = []
    for i in range(3000):
        if live and (rng.random() < 0.5 or len(live) > 5):
            pool.free(live.pop(rng.randrange(len(live))))
        base = rng.choice(prefixes)
        hashes = base[: rng.randint(1, len(base))]
        tokens = len(hashes) * BS + rng.randint(0, BS - 1)
        rid = f"q{i}"
        hit = pool.lookup(hashes, tokens)
        if pool.admit(rid, hit, hashes, tokens, float(i)):
            live.append(rid)
        assert pool.stranded_hashes() == [], f"stranded after step {i}"
    pool.check_consistency()
    assert pool.real_evictions > 100  # the test only means something if eviction really happened


def test_one_block_is_reserved_as_the_null_block():
    """block_pool.py:193 — `self.null_block = self.free_block_queue.popleft()`. With
    `--num-gpu-blocks-override 1024` the engine has 1,023 usable blocks, and so must we."""
    pool = VllmBlockPool(1024, BS)
    assert pool.usable_blocks == 1023
    assert pool.num_free == 1023
    assert pool.can_admit([], 1023 * BS)
    assert not pool.can_admit([], 1024 * BS)
    pool.check_consistency()
