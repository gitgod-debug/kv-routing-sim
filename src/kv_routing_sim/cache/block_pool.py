"""A model of vLLM's KV block pool, the default cache model (`cache_model="vllm"`).

It behaves differently from the plain LRU of `block_trie.py` once the cache fills, which is the
regime where routing matters. Each behaviour implements the vLLM source it cites, at the v0.30.0
tag, and is not tuned against measurements (ADR-002):

    frees a finished request's blocks tail first        single_type_kv_cache_manager.py:580
    only blocks with ref_cnt 0 are evictable            block_pool.py `free_blocks`, `touch`
    uncached blocks are reused first (prepended, LIFO)  block_pool.py:789, `prepend_n`
    cached blocks are reused last (appended, FIFO)      block_pool.py:792, `append_n`
    capacity includes partial and decode blocks         block_pool.py `get_new_blocks`
    full decode blocks are cached too                   kv_cache_manager.py:595-599
    a fully cached prompt still recomputes a block      kv_cache_manager.py:295
    an eviction event fires on any recycle              block_pool.py:739, before the check at 741
    one block is reserved as the null block             block_pool.py:193

Assumptions not yet checked against a live engine:
  * Hashes are committed at scheduling time (`cache_blocks` runs inside `allocate_slots`), so a
    request admitted later in the same step can hit blocks an earlier one is about to compute.
  * Decode-block hashes are unique per request. In an open-loop replay no later prompt contains a
    request's output tokens, so only the same request, after preemption, can hit those blocks.
  * There is no chunked prefill and no token budget: a prompt is allocated in one step. This affects
    TTFT under load more than hit rate.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field


def max_hit_blocks(num_tokens: int, block_size: int) -> int:
    """Return how many full blocks a lookup may hit, by vLLM's rule; both cache models apply it.

    When every prompt token is cached, vLLM still recomputes the last token to obtain logits, and
    because computed tokens must be block-aligned it recomputes that whole block
    (kv_cache_manager.py:290-295). A fully cached 512-token prompt hits 31 blocks, not 32.
    """
    return max(0, num_tokens - 1) // block_size


def _h64(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


@dataclass(slots=True)
class _BlockMetrics:
    """kv_cache_metrics.py `BlockMetricsState`: per-block metrics, in simulated time."""

    birth: float
    last_access: float
    # Only the last four accesses are kept (kv_cache_metrics.py:24), so at most three reuse gaps
    # per block, biased toward the end of its life.
    history: deque[float] = field(default_factory=lambda: deque(maxlen=4))


@dataclass(frozen=True, slots=True)
class EvictionEvent:
    """One `KVCacheEvictionEvent`, the record behind the three `vllm:kv_block_*` histograms."""

    lifetime_s: float
    idle_s: float
    reuse_gaps_s: tuple[float, ...]
    # True only for a real eviction; vLLM emits the event either way (block_pool.py:739).
    was_cached: bool


class VllmBlockPool:
    """One replica's KV block pool, with vLLM's free-queue order, reference counts and eviction.

    The free queue is an OrderedDict of block ids whose front is the next block to be reused. vLLM
    hand-rolls a doubly linked list for O(1) removal from the middle (kv_cache_utils.py:246); an
    OrderedDict gives the same O(1) popleft, append, prepend and remove.
    """

    def __init__(
        self, num_blocks: int, block_size: int = 16, track_block_metrics: bool = False
    ) -> None:
        """Create a pool with every usable block free.

        Args:
            num_blocks: vLLM's `num_gpu_blocks`, as set by `--num-gpu-blocks-override` and reported
                in the engine's log. One of them is the null block, so `num_blocks - 1` are usable.
        """
        if num_blocks <= 1:
            raise ValueError(f"num_blocks must be > 1 (one is the null block), got {num_blocks}")
        self.num_blocks = num_blocks
        self.block_size = block_size

        # The free queue starts ordered by block id (kv_cache_utils.py:255).
        self._free: OrderedDict[int, None] = OrderedDict((b, None) for b in range(num_blocks))
        # Block 0 is reserved as the null block and never allocated (block_pool.py:193). It only
        # shows with a small cache, where it moves every prediction by one block.
        self.null_block, _ = self._free.popitem(last=False)
        self._ref = [0] * num_blocks
        self._hash: list[int | None] = [None] * num_blocks
        # cached hash -> its parent's hash, for the no-stranding invariant
        self._parent: dict[int, int | None] = {}
        # hash -> the block ids holding it. vLLM allows duplicates and returns the first
        # (block_pool.py `get_cached_block`).
        self._by_hash: dict[int, dict[int, None]] = {}

        # Per request: its block table, how many of those blocks are committed (hashed), and the
        # hash chain it has built, kept across preemption so the request can hit it again.
        self._req_blocks: dict[str, list[int]] = {}
        self._req_committed: dict[str, int] = {}
        self._req_chain: dict[str, list[int]] = {}

        # Counters in tokens, as vLLM counts (ADR-008). Lookups by preempted requests are kept
        # separate, as vllm/v1/metrics/stats.py `PrefixCacheStats.record` does, because the
        # Prometheus prefix-cache counters exclude them.
        self.queries = 0
        self.hits = 0
        self.preempted_queries = 0
        self.preempted_hits = 0
        self.real_evictions = 0  # a popped block that held a hash

        self.track_block_metrics = track_block_metrics
        self._metrics: dict[int, _BlockMetrics] = {}
        self.eviction_events: list[EvictionEvent] = []

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def usable_blocks(self) -> int:
        return self.num_blocks - 1  # the null block is never allocated

    @property
    def hit_rate(self) -> float:
        return self.hits / self.queries if self.queries else 0.0

    def is_cached(self, h: int) -> bool:
        return h in self._by_hash

    def blocks_of(self, request_id: str) -> list[int]:
        return list(self._req_blocks.get(request_id, []))

    def chain_for(self, request_id: str, prompt_hashes: Sequence[int]) -> Sequence[int]:
        """Return the hashes to look up: a preempted request's own chain, else the prompt's.

        vLLM's `_preempt_request` keeps a request's tokens and frees only its KV, so after
        preemption the request looks up its prompt and decode blocks.
        """
        return self._req_chain.get(request_id) or prompt_hashes

    def lookup(self, block_hashes: Sequence[int], num_tokens: int) -> list[int]:
        """Return the longest cached prefix as block ids.

        Stops at the first miss, as `FullAttentionManager.find_longest_cache_hit` does in
        single_type_kv_cache_manager.py, and at the `max_hit_blocks` cap (kv_cache_manager.py:295).
        Blocks are found whether they are in use or free.
        """
        limit = min(len(block_hashes), max_hit_blocks(num_tokens, self.block_size))
        hit: list[int] = []
        for h in block_hashes[:limit]:
            holders = self._by_hash.get(h)
            if not holders:
                break
            hit.append(next(iter(holders)))  # the first, like get_one_block
        return hit

    def record_lookup(self, num_tokens: int, hit_blocks: int, preempted: bool) -> None:
        """Count a lookup, in tokens.

        Lookups by preempted requests go to separate counters (stats.py `PrefixCacheStats.record`).
        """
        hit_tokens = hit_blocks * self.block_size
        if preempted:
            self.preempted_queries += num_tokens
            self.preempted_hits += hit_tokens
        else:
            self.queries += num_tokens
            self.hits += hit_tokens

    def blocks_needed(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)  # ceil

    def can_admit(self, hit_blocks: Sequence[int], num_tokens: int) -> bool:
        """Return whether `admit` would succeed, as vLLM's `allocate_slots` checks it.

        A hit on a free block takes it out of the free queue (block_pool.py `touch`), so it counts
        against the free pool as well.
        """
        new = self.blocks_needed(num_tokens) - len(hit_blocks)
        evictable_hits = sum(1 for b in hit_blocks if self._ref[b] == 0)
        return new + evictable_hits <= self.num_free

    def admit(
        self,
        request_id: str,
        hit_blocks: Sequence[int],
        prompt_hashes: Sequence[int],
        num_tokens: int,
        now: float,
    ) -> bool:
        """Allocate a request's blocks for `num_tokens` and commit its full blocks.

        As kv_cache_manager.py `allocate_slots` does: touch the hit blocks, get new ones, then
        `cache_blocks` up to the tokens computed this step (lines 595-599). Returns False if there
        is no room.
        """
        if not self.can_admit(hit_blocks, num_tokens):
            return False
        self._touch(hit_blocks, now)
        new = self._get_new_blocks(self.blocks_needed(num_tokens) - len(hit_blocks), now)
        self._req_blocks[request_id] = list(hit_blocks) + new
        self._req_committed[request_id] = len(hit_blocks)
        chain = self._req_chain.setdefault(request_id, [])
        if not chain:
            chain.extend(prompt_hashes)
        self._commit(request_id, num_tokens)
        return True

    def grow(self, request_id: str, num_tokens: int, now: float) -> bool:
        """Grow a request to `num_tokens` for a decode step and commit newly full blocks.

        Returns False when no block is free; the caller must then preempt (scheduler.py:726-800).
        """
        blocks = self._req_blocks[request_id]
        need = self.blocks_needed(num_tokens) - len(blocks)
        if need > self.num_free:
            return False
        if need > 0:
            blocks.extend(self._get_new_blocks(need, now))
        self._commit(request_id, num_tokens)
        return True

    def free(self, request_id: str) -> None:
        """Release a finished or preempted request's blocks; their hashes stay cached.

        Blocks are freed tail first (single_type_kv_cache_manager.py:580). When ref_cnt reaches 0,
        an uncached block is prepended to the free queue and reused first, and a cached block is
        appended and evicted last (block_pool.py:770-799).
        """
        evict_first: list[int] = []
        evict_last: list[int] = []
        for b in reversed(self._req_blocks.pop(request_id, [])):
            self._ref[b] -= 1
            if self._ref[b] == 0:
                (evict_first if self._hash[b] is None else evict_last).append(b)
        # prepend_n keeps list order: evict_first[0] becomes the new front (kv_cache_utils.py:411)
        for b in reversed(evict_first):
            self._free[b] = None
            self._free.move_to_end(b, last=False)
        for b in evict_last:
            self._free[b] = None
        self._req_committed.pop(request_id, None)

    def forget(self, request_id: str) -> None:
        """Drop a finished request's hash chain; a preempted request keeps its chain."""
        self._req_chain.pop(request_id, None)

    def _touch(self, blocks: Sequence[int], now: float) -> None:
        """block_pool.py `touch`: a free block that is hit leaves the free queue; ref_cnt += 1."""
        for b in blocks:
            if self._ref[b] == 0:
                del self._free[b]
            self._ref[b] += 1
            m = self._metrics.get(b)
            if m is not None:  # kv_cache_metrics.py `on_block_accessed`
                m.last_access = now
                m.history.append(now)

    def _get_new_blocks(self, n: int, now: float) -> list[int]:
        """block_pool.py `get_new_blocks`: pop from the front, evict, allocate."""
        out: list[int] = []
        for _ in range(n):
            b, _ = self._free.popitem(last=False)
            self._maybe_evict(b, now)
            self._ref[b] = 1
            if self.track_block_metrics:  # sample rate 1.0: every block is tracked
                self._metrics[b] = _BlockMetrics(birth=now, last_access=now)
            out.append(b)
        return out

    def _maybe_evict(self, b: int, now: float) -> None:
        """block_pool.py `_maybe_evict_cached_block`, including its ordering quirk.

        Line 739 records the metrics event before line 741 checks whether the block held a hash, so
        recycling a block that never held a cached prefix is still reported as an eviction. The
        model does the same so that its events and the engine's histograms count by the same rule.
        vLLM pull request #47715 proposes the fix upstream.
        """
        m = self._metrics.pop(b, None)
        if m is not None:
            hist = list(m.history)
            gaps = tuple(hist[i] - hist[i - 1] for i in range(1, len(hist)))
            self.eviction_events.append(
                EvictionEvent(now - m.birth, now - m.last_access, gaps, self._hash[b] is not None)
            )
        h = self._hash[b]
        if h is not None:
            holders = self._by_hash.get(h)
            if holders is not None:
                holders.pop(b, None)
                if not holders:
                    del self._by_hash[h]
                    self._parent.pop(h, None)
            self._hash[b] = None
            self.real_evictions += 1

    def _commit(self, request_id: str, num_tokens: int) -> None:
        """kv_cache_manager.py:595-599 `cache_blocks`: hash every block that is now full.

        Prompt blocks take their hash from the trace. Decode blocks, and the block straddling the
        end of the prompt, get a hash that continues the chain and is unique to this request (see
        the module's assumptions). The chain is extended in place so a preempted request can hit it
        again.
        """
        full = num_tokens // self.block_size
        blocks = self._req_blocks[request_id]
        chain = self._req_chain[request_id]
        done = self._req_committed[request_id]
        upto = min(full, len(blocks))
        for i in range(done, upto):
            if i >= len(chain):
                parent = chain[i - 1] if i > 0 else 0
                payload = parent.to_bytes(8, "big") + request_id.encode() + i.to_bytes(4, "big")
                chain.append(_h64(payload))
            h = chain[i]
            b = blocks[i]
            if self._hash[b] is None:
                self._hash[b] = h
                self._by_hash.setdefault(h, {})[b] = None
                self._parent.setdefault(h, chain[i - 1] if i > 0 else None)
        self._req_committed[request_id] = max(done, upto)

    def stranded_hashes(self) -> list[int]:
        """Return cached hashes whose parent is not cached, which no lookup can reach.

        Tail-first freeing and the prefix rule keep a parent at or behind its children in the free
        queue, so a parent is never evicted before its child and this list should be empty.
        """
        return [h for h, p in self._parent.items() if p is not None and p not in self._by_hash]

    def check_consistency(self) -> None:
        """Assert that every block is either free (ref_cnt 0) or referenced, never both."""
        for b in range(self.num_blocks):
            if b == self.null_block:
                if b in self._free or self._ref[b] != 0:
                    raise AssertionError("the null block must never be allocated or freed")
                continue
            in_free = b in self._free
            if in_free and self._ref[b] != 0:
                raise AssertionError(f"block {b} is in the free queue with ref_cnt {self._ref[b]}")
            if not in_free and self._ref[b] <= 0:
                raise AssertionError(f"block {b} is neither free nor referenced")
