"""The per-replica prefix cache: which blocks are resident, and how long a prefix matches.

⭐ A SIMPLIFICATION WORTH UNDERSTANDING, because it looks like a shortcut and is not.

`docs/05-codebase-map.md` calls this a trie. The implementation is a flat dict. Both are true: the
hash CHAIN already encodes the tree. Block i's hash is computed from block i-1's hash, so a given
hash value uniquely identifies one full path from the root. Walking a request's hashes in order and
stopping at the first one that is not resident gives exactly radix-trie semantics, with O(1) per
block and no node objects to allocate.

⛔ THE STOP-AT-FIRST-MISS RULE IS NOT AN OPTIMIZATION — IT IS THE PHYSICS. To attend over a prefix
you need every KV block in it. If block 3 was evicted, blocks 4+ are useless even if still resident,
because there is no way to skip a hole. A lookup that counted all matching blocks regardless of
order would inflate every hit rate in the project.

⭐ AND THE CONSEQUENCE THAT H4 IS ABOUT: when block 3 is evicted, blocks 4+ stay RESIDENT but become
unreachable. They still occupy capacity. They are not proactively dropped, because that is not what
the engine does — vLLM refcounts blocks and evicts by LRU, it does not cascade. So the cache can
hold blocks that can never be hit until LRU reaches them. This "stranded block" behaviour is one of
the concrete mechanisms behind the eviction-order residual H4 predicts, and modelling it is why the
residual might be explainable rather than merely narratable.

⚠️ We model stranding but do not COUNT it: knowing a block is stranded requires tracking parentage,
which this flat representation deliberately does not keep. If H4's investigation needs the number,
that is the change to make — and it is a real cost (a parent pointer per block), so it waits until
a measurement asks for it rather than being paid for speculatively.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence


class BlockCache:
    """A single replica's prefix cache: LRU over a bounded pool of blocks.

    ⭐ COUNTED IN TOKENS, EXACTLY AS vLLM COUNTS — corrected 2026-09-21 from a live engine.

    `queries` and `hits` mirror vLLM's own counters in name AND unit, so a predicted and a measured
    hit rate are literally the same quantity (ADR-008). The Phase-0 spike (vLLM 0.29.0, 2x T4)
    settled the unit two independent ways:

      * vLLM's own HELP text: "Prefix cache queries, in terms of number of queried TOKENS" and
        "Prefix cache hits, in terms of number of cached TOKENS".
      * Empirically: 512 tokens sent on a cold cache moved `queries` by exactly 512, not 32.

    And it exposed a SECOND discrepancy that a block->token conversion alone would not have fixed:

      * vLLM's `queries` is ALL prompt tokens, including the partial trailing block.
      * `hits` can only ever be FULL blocks (the partial-block rule).

    So for a 513-token prompt whose 32 full blocks are all cached, vLLM reports 512/513 = 0.99805,
    not 1.0. Before this fix we counted full blocks on BOTH sides and reported 1.0 — biased upward,
    and badly so on short prompts: a 20-token prompt has one 16-token block, so we would have said
    16/16 = 100% where vLLM says 16/20 = 80%.

    `queries_blocks` / `hits_blocks` are kept for internal use (eviction reasoning, capacity).
    Everything that is compared against the engine uses the token counters.
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
        self.queries = 0  # TOKENS — every prompt token, as vLLM counts
        self.hits = 0  # TOKENS — cached full blocks x block_size
        self.queries_blocks = 0
        self.hits_blocks = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._resident)

    @property
    def hit_rate(self) -> float:
        """⛔ DERIVED, never stored. Cached tokens / ALL prompt tokens — vLLM's own definition."""
        return self.hits / self.queries if self.queries else 0.0

    def lookup(self, block_hashes: Sequence[int]) -> int:
        """Longest resident prefix, in blocks. Stops at the first miss. Touches every hit.

        Does NOT mutate residency beyond LRU ordering — insertion is a separate step, because in a
        real engine the lookup happens at admission and the newly computed blocks are added as the
        prefill runs.
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
        """Insert the blocks this request computed (everything after the matched prefix).

        Blocks are inserted in order, so the LRU ordering reflects the chain: a prefix is always at
        least as recently used as what follows it.
        """
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
        """Lookup + admit + count, as one admission would do. Returns matched BLOCKS.

        `prompt_tokens` is the request's full prompt length, INCLUDING any partial trailing block
        that `block_hashes` omits. It is what vLLM counts as queried, so pass it whenever the real
        length is known — every real request has it. When omitted, it falls back to full blocks
        only (`len(block_hashes) * block_size`), which is exact for block-aligned prompts and
        biased upward otherwise.
        """
        matched = self.lookup(block_hashes)
        tokens = prompt_tokens if prompt_tokens is not None else len(block_hashes) * self.block_size
        self.queries += tokens
        self.hits += matched * self.block_size
        self.queries_blocks += len(block_hashes)
        self.hits_blocks += matched
        self.admit(block_hashes, matched)
        return matched

    def reset_counters(self) -> None:
        """Zero the counters but keep residency — for discarding a warm-up period."""
        self.queries = 0
        self.hits = 0
        self.queries_blocks = 0
        self.hits_blocks = 0
        self.evictions = 0
