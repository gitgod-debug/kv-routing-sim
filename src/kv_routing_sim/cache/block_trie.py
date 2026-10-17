"""The plain-LRU prefix cache, kept as the ablation baseline (`cache_model="lru"`).

Resident blocks are a flat dict rather than a tree: a block's hash is computed from its
predecessor's, so it identifies the whole path from the root, and walking a request's hashes until
the first miss gives radix-trie semantics at O(1) per block. When a block is evicted, the blocks
after it stay resident but unreachable until LRU order evicts them; they are not counted, since that
would need a parent pointer per block.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

from .block_pool import max_hit_blocks


class BlockCache:
    """A single replica's prefix cache: LRU over a bounded pool of blocks.

    `queries` and `hits` count tokens, as vLLM's prefix-cache counters do (ADR-008). `queries` is
    every prompt token, including a partial trailing block, while `hits` can only be full blocks, so
    a 513-token prompt whose 32 full blocks are all cached scores 512/513, not 1.0. `queries_blocks`
    and `hits_blocks` are internal; comparisons with the engine use the token counters.
    """

    __slots__ = (
        "_resident",
        "block_size",
        "capacity_blocks",
        "evictions",
        "hits",
        "hits_blocks",
        "queries",
        "queries_blocks",
    )

    def __init__(self, capacity_blocks: int, block_size: int = 16) -> None:
        if capacity_blocks <= 0:
            raise ValueError(f"capacity_blocks must be > 0, got {capacity_blocks}")
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")
        self.capacity_blocks = capacity_blocks
        self.block_size = block_size
        # OrderedDict as an LRU: leftmost is least-recently-used.
        self._resident: OrderedDict[int, None] = OrderedDict()
        self.queries = 0  # tokens: every prompt token, as vLLM counts
        self.hits = 0  # tokens: cached full blocks x block_size
        self.queries_blocks = 0
        self.hits_blocks = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._resident)

    @property
    def hit_rate(self) -> float:
        """Cached tokens over all prompt tokens, derived from the counters as vLLM defines it."""
        return self.hits / self.queries if self.queries else 0.0

    def lookup(self, block_hashes: Sequence[int]) -> int:
        """Return the longest resident prefix in blocks, touching every hit.

        Stops at the first miss, since a prefix is usable only up to its first missing block.
        Insertion is a separate step (`admit`) because an engine looks up at admission and adds the
        newly computed blocks as the prefill runs.
        """
        matched = 0
        for h in block_hashes:
            if h in self._resident:
                self._resident.move_to_end(h)
                matched += 1
            else:
                break
        return matched

    def admit(self, block_hashes: Sequence[int], matched: int) -> None:
        """Insert the blocks this request computed, those after the matched prefix."""
        for h in block_hashes[matched:]:
            if h in self._resident:
                self._resident.move_to_end(h)
            else:
                self._resident[h] = None
                if len(self._resident) > self.capacity_blocks:
                    self._evict_one()

    def _evict_one(self) -> None:
        self._resident.popitem(last=False)  # drop the least-recently-used block
        self.evictions += 1

    def query(self, block_hashes: Sequence[int], prompt_tokens: int | None = None) -> int:
        """Look up, admit and count one request; return the number of matched blocks.

        Args:
            prompt_tokens: The full prompt length, including any partial trailing block that
                `block_hashes` omits, which vLLM counts as queried. Without it only full blocks are
                counted: exact for block-aligned prompts, biased upward otherwise.
        """
        tokens = prompt_tokens if prompt_tokens is not None else len(block_hashes) * self.block_size
        # vLLM's lookup cap (kv_cache_manager.py:295): a fully cached prompt still recomputes its
        # last block. It is a lookup rule, so both cache models apply it and the ablation between
        # them compares eviction alone.
        matched = self.lookup(block_hashes[: max_hit_blocks(tokens, self.block_size)])
        self.queries += tokens
        self.hits += matched * self.block_size
        self.queries_blocks += len(block_hashes)
        self.hits_blocks += matched
        self.admit(block_hashes, matched)
        return matched

    def reset_counters(self) -> None:
        """Zero the counters but keep residency, to discard a warm-up period."""
        self.queries = 0
        self.hits = 0
        self.queries_blocks = 0
        self.hits_blocks = 0
        self.evictions = 0
