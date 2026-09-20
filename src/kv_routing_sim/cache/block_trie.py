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

    Counter names deliberately mirror vLLM's own (`queries`, `hits`) so the predicted and the
    measured quantity are literally the same quantity rather than two things that sound alike
    (ADR-008).
    """

    __slots__ = ("_resident", "capacity_blocks", "evictions", "hits", "queries")

    def __init__(self, capacity_blocks: int) -> None:
        if capacity_blocks <= 0:
            raise ValueError(f"capacity_blocks must be > 0, got {capacity_blocks}")
        self.capacity_blocks = capacity_blocks
        # OrderedDict as an LRU: leftmost is least-recently-used.
        self._resident: OrderedDict[int, None] = OrderedDict()
        self.queries = 0
        self.hits = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._resident)

    @property
    def hit_rate(self) -> float:
        """⛔ DERIVED, never a stored gauge. hits / queries — the same shape as ADR-008 requires."""
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

    def query(self, block_hashes: Sequence[int]) -> int:
        """Lookup + admit + count, as one admission would do. Returns matched blocks.

        ⚠️ UNIT NOTE — a Phase-0 spike question, not a settled fact. We count `queries` and `hits`
        in BLOCKS. Whether vLLM's `vllm:prefix_cache_queries` counts blocks or tokens must be read
        off the live engine before any predicted-vs-measured comparison is trusted; if it counts
        tokens, multiply by block size. Comparing a block-rate against a token-rate would produce a
        confident, wrong calibration error. See `CHECKPOINTS.md` Phase 0.
        """
        matched = self.lookup(block_hashes)
        self.queries += len(block_hashes)
        self.hits += matched
        self.admit(block_hashes, matched)
        return matched

    def reset_counters(self) -> None:
        """Zero the counters but keep residency — for discarding a warm-up period."""
        self.queries = 0
        self.hits = 0
        self.evictions = 0
