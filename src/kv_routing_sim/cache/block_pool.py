"""A faithful model of vLLM's KV block pool — the `cache_model="vllm"` path.

⭐ EVERY BEHAVIOUR BELOW CITES THE vLLM LINE IT IMPLEMENTS — that is the rule for editing this file.
Citations are to the `v0.30.0` tag. This module implements the engine's documented algorithm, read
from its source; it is NOT tuned against measurements. The difference matters: a model adjusted
until it matches the data proves nothing, while a model that implements the published algorithm and
THEN matches is evidence (ADR-002, amendment 2026-09-22).

Why it exists: `BlockCache` (`cache_model="lru"`, kept unchanged as the ablation baseline) is a
plain LRU touched head-to-tail. vLLM's pool differs in ways invisible until the cache fills —
which is exactly the regime where routing matters:

  vLLM does                                             source (v0.30.0)
  ----------------------------------------------------  ----------------------------------------
  frees a finished request's blocks TAIL-FIRST          single_type_kv_cache_manager.py:580
  only ref_cnt == 0 blocks are evictable (pinning)      block_pool.py `free_blocks`, `touch`
  uncached blocks reused first, LIFO (prepended)        block_pool.py:789, `prepend_n`
  cached blocks reused last, FIFO (appended)            block_pool.py:792, `append_n`
  capacity includes partial and decode blocks           block_pool.py `get_new_blocks`
  full DECODE blocks are cached too                     kv_cache_manager.py:595-599
  a fully cached prompt still recomputes a block        kv_cache_manager.py:295
  eviction events fire on ANY recycle                   block_pool.py:739 (before the check at 741)
  ONE block is reserved as the null block, never used   block_pool.py:193

⭐ AN INVARIANT THIS PRODUCES, checked by a property test: a cached block's parent is never evicted
before the block itself. Freeing is tail-first, and a hit on a child always touches its parent too
(the prefix rule), so a parent always sits at or behind its children in the free queue. vLLM
therefore does not strand blocks the way plain LRU does. A stranded block here means a bug here.

⚠️ ASSUMPTIONS — not verified against a live engine, named so Phase 3 can test them:
  * Hashes are committed at SCHEDULING time (`cache_blocks` runs inside `allocate_slots`), so a
    request admitted later in the same step can hit blocks an earlier one is about to compute. The
    lookup is a plain hash-map check (block_pool.py `get_cached_block`) and nothing found in the
    scheduler prevents it. A concurrent workload will confirm or refute this.
  * Decode-block hashes are unique per request. In vLLM two identical prompts at temperature 0
    produce identical output blocks, hence identical hashes. In an OPEN-LOOP replay no later prompt
    contains those tokens, so only the same request resuming after preemption can hit them — for
    which a per-request hash is exact.
  * No chunked prefill and no token budget: a prompt is allocated in one step. That affects TTFT
    under load more than it affects hit rate.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict, deque
from collections.abc import Sequence
from dataclasses import dataclass, field


def max_hit_blocks(num_tokens: int, block_size: int) -> int:
    """How many full blocks a lookup may hit — vLLM's rule, applied by BOTH cache models.

    kv_cache_manager.py:290-295: "When all tokens hit the cache, we must recompute the last token to
    obtain logits. Thus, set max_cache_hit_length to prompt_length - 1. This can trigger
    recomputation of an entire block, rather than just the single last token, because
    allocate_slots() requires num_computed_tokens to be block-size aligned."

    So a fully cached 512-token prompt hits 31 blocks (496 tokens), not 32.
    """
    return max(0, num_tokens - 1) // block_size


def _h64(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


@dataclass(slots=True)
class _BlockMetrics:
    """kv_cache_metrics.py `BlockMetricsState`, in simulated time."""

    birth: float
    last_access: float
    # kv_cache_metrics.py:24 — only the last FOUR accesses are kept ("Bounded to prevent
    # unbounded growth"), so at most three reuse gaps per block, biased toward the end of its life.
    history: deque[float] = field(default_factory=lambda: deque(maxlen=4))


@dataclass(frozen=True, slots=True)
class EvictionEvent:
    """One `KVCacheEvictionEvent` — what feeds the three `vllm:kv_block_*` histograms."""

    lifetime_s: float
    idle_s: float
    reuse_gaps_s: tuple[float, ...]
    # True only for a REAL eviction. vLLM emits the event either way (block_pool.py:739).
    was_cached: bool


class VllmBlockPool:
    """One replica's KV block pool, with vLLM's free-queue order, reference counts and eviction.

    The free queue is an OrderedDict of block ids, front = next to be reused. vLLM hand-rolls a
    doubly linked list for O(1) removal from the middle (kv_cache_utils.py:246); an OrderedDict
    gives the same O(1) popleft, append, prepend and remove.
    """

    def __init__(
        self, num_blocks: int, block_size: int = 16, track_block_metrics: bool = False
    ) -> None:
        """`num_blocks` is vLLM's `num_gpu_blocks` — what `--num-gpu-blocks-override` sets and what
        the engine's log reports. ONE of those is the null block (below), so `num_blocks - 1` are
        usable."""
        if num_blocks <= 1:
            raise ValueError(f"num_blocks must be > 1 (one is the null block), got {num_blocks}")
        self.num_blocks = num_blocks
        self.block_size = block_size

        # kv_cache_utils.py:255 — "The queue is ordered by block ID in the beginning."
        self._free: OrderedDict[int, None] = OrderedDict((b, None) for b in range(num_blocks))
        # block_pool.py:193 — `self.null_block = self.free_block_queue.popleft()`: block 0 is
        # reserved as a placeholder and never allocated. Invisible while the cache has room;
        # with a deliberately small cache it moves every prediction by a block.
        self.null_block, _ = self._free.popitem(last=False)
        self._ref = [0] * num_blocks
        self._hash: list[int | None] = [None] * num_blocks
        # cached hash -> its parent's hash, for the no-stranding invariant
        self._parent: dict[int, int | None] = {}
        # hash -> the block ids holding it. vLLM allows duplicates and returns the first
        # (block_pool.py `get_cached_block`: "If there are duplicated blocks, we return the first").
        self._by_hash: dict[int, dict[int, None]] = {}

        # Per request: its block table, how many of those blocks are committed (hashed), and the
        # hash chain it has built — kept across preemption so a resumed request can re-hit it.
        self._req_blocks: dict[str, list[int]] = {}
        self._req_committed: dict[str, int] = {}
        self._req_chain: dict[str, list[int]] = {}

        # Counters in TOKENS, as vLLM counts (ADR-008). Preempted re-lookups are kept separate,
        # exactly as vllm/v1/metrics/stats.py `PrefixCacheStats.record` does: the Prometheus
        # counters we calibrate against exclude them.
        self.queries = 0
        self.hits = 0
        self.preempted_queries = 0
        self.preempted_hits = 0
        self.real_evictions = 0  # a popped block that held a hash

        self.track_block_metrics = track_block_metrics
        self._metrics: dict[int, _BlockMetrics] = {}
        self.eviction_events: list[EvictionEvent] = []

    # ── views ─────────────────────────────────────────────────────────────────────────────────

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
        """The hashes to look up: a preempted request's own persisted chain (prompt + decode
        blocks — `_preempt_request` keeps the tokens and frees only the KV), else the prompt's."""
        return self._req_chain.get(request_id) or prompt_hashes

    # ── lookup ────────────────────────────────────────────────────────────────────────────────

    def lookup(self, block_hashes: Sequence[int], num_tokens: int) -> list[int]:
        """Longest cached prefix, as block ids. Stops at the first miss; capped by
        `max_hit_blocks` (stm.py `FullAttentionManager.find_longest_cache_hit`,
        kv_cache_manager.py:295). Blocks are found whether in use or free — the map holds both."""
        limit = min(len(block_hashes), max_hit_blocks(num_tokens, self.block_size))
        hit: list[int] = []
        for h in block_hashes[:limit]:
            holders = self._by_hash.get(h)
            if not holders:
                break
            hit.append(next(iter(holders)))  # the first, like get_one_block
        return hit

    def record_lookup(self, num_tokens: int, hit_blocks: int, preempted: bool) -> None:
        """stats.py `PrefixCacheStats.record`: preempted requests go to separate counters."""
        hit_tokens = hit_blocks * self.block_size
        if preempted:
            self.preempted_queries += num_tokens
            self.preempted_hits += hit_tokens
        else:
            self.queries += num_tokens
            self.hits += hit_tokens

    # ── allocation ────────────────────────────────────────────────────────────────────────────

    def blocks_needed(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)  # ceil

    def can_admit(self, hit_blocks: Sequence[int], num_tokens: int) -> bool:
        """Would `admit` succeed? vLLM's `allocate_slots` check.

        Hitting a block that is currently FREE pulls it out of the free queue (block_pool.py
        `touch`), so it can no longer serve a new allocation — it counts against the free pool too.
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

        kv_cache_manager.py `allocate_slots`: touch the hit blocks, get the new ones, then
        `cache_blocks` up to the tokens computed this step (lines 595-599). False if no room.
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
        """A decode step: make room for `num_tokens` total and commit newly full blocks.

        False means no free block: the caller must preempt (scheduler.py:726-800).
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
        """Return a finished or preempted request's blocks. Hashes are KEPT: freed is not evicted.

        single_type_kv_cache_manager.py:580 — "Free blocks in reverse order so that the tail blocks
        are freed first." block_pool.py:770-799 — ref_cnt drops; at zero an uncached block is
        PREPENDED (reused first, LIFO) and a cached block is APPENDED (evicted last, FIFO).
        """
        evict_first: list[int] = []
        evict_last: list[int] = []
        for b in reversed(self._req_blocks.pop(request_id, [])):
            self._ref[b] -= 1
            if self._ref[b] == 0:
                (evict_first if self._hash[b] is None else evict_last).append(b)
        # prepend_n keeps list order — evict_first[0] becomes the new front (kv_cache_utils.py:411)
        for b in reversed(evict_first):
            self._free[b] = None
            self._free.move_to_end(b, last=False)
        for b in evict_last:
            self._free[b] = None
        self._req_committed.pop(request_id, None)

    def forget(self, request_id: str) -> None:
        """Drop a FINISHED request's hash chain. A preempted one keeps it, to re-hit on resume."""
        self._req_chain.pop(request_id, None)

    # ── internals ─────────────────────────────────────────────────────────────────────────────

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
        """block_pool.py `get_new_blocks`: pop from the FRONT, evict, allocate."""
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
        """block_pool.py `_maybe_evict_cached_block`, INCLUDING its ordering quirk.

        Line 739 records the metrics event BEFORE line 741 checks whether the block held a hash, so
        recycling a block that never held a cached prefix is still reported as an "eviction". We do
        the same, deliberately: H4 compares our events against the engine's histograms, and that is
        only fair if both sides count by the same rule. (Logged upstream: the upstream log #1.)
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
        """kv_cache_manager.py:595-599 `cache_blocks`: hash every block that is now FULL.

        Prompt blocks take their hash from the trace. Decode blocks — and the block straddling the
        end of the prompt — get a hash continuing the chain, unique to this request (see the
        module's assumptions). The chain is extended in place so a preempted request can re-hit it.
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

    # ── invariant checks, for tests ───────────────────────────────────────────────────────────

    def stranded_hashes(self) -> list[int]:
        """Cached hashes whose parent is NOT cached — unreachable by any lookup. Should be none."""
        return [h for h, p in self._parent.items() if p is not None and p not in self._by_hash]

    def check_consistency(self) -> None:
        """Every block is either free (ref 0) or referenced — never both, never neither."""
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
