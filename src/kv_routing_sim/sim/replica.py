"""One replica: a bounded batch, a waiting queue, and a prefix cache.

The continuous-batching model (docs/03-background-concepts.md §4): a request does not wait for the
batch ahead of it to FINISH, it waits for a SLOT. Each iteration admits whatever fits, runs one
prefill pass over the newly admitted, and advances every running sequence by one token.

⭐ WHERE THE CACHE ENTERS, AND WHY IT IS HERE AND NOT IN THE ROUTER: the lookup happens at
ADMISSION, not at arrival. A request routed to a replica with a warm prefix still has to wait for a
slot, and by the time it gets one the prefix may have been evicted by whatever ran in between. That
gap between "routed here because the cache was warm" and "actually found it warm" is a real source
of loss, it is what punishes locality under load, and it is a mechanism behind the H2 crossover.
Doing the lookup at arrival instead would quietly delete that effect.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..cache.block_trie import BlockCache
from ..traces.schema import NormalizedRequest
from .cost_model import CostProfile


@dataclass(slots=True)
class InFlight:
    request: NormalizedRequest
    arrival_ts: float
    admitted_ts: float
    first_token_ts: float
    tokens_emitted: int
    matched_blocks: int
    uncached_tokens: int


class Replica:
    __slots__ = ("cache", "index", "iterations", "next_free_ts", "profile", "queue", "running")

    def __init__(self, index: int, profile: CostProfile) -> None:
        self.index = index
        self.profile = profile
        self.cache = BlockCache(profile.kv_cache_blocks)
        self.running: list[InFlight] = []
        self.queue: deque[NormalizedRequest] = deque()
        self.next_free_ts = 0.0
        self.iterations = 0

    def enqueue(self, request: NormalizedRequest) -> None:
        self.queue.append(request)

    @property
    def has_work(self) -> bool:
        return bool(self.running or self.queue)

    def step(self, now: float) -> tuple[float, list[InFlight]]:
        """Run one iteration. Returns (duration, sequences that finished).

        Admission first, then one decode step for everything in the batch — the order a real
        scheduler uses, and the reason a newly admitted request emits its first token at the end of
        the same iteration it was admitted in.
        """
        p = self.profile
        prefill_s = 0.0

        while len(self.running) < p.max_num_seqs and self.queue:
            req = self.queue.popleft()
            matched = self.cache.query(req.block_hashes)
            cached_tokens = matched * p.block_size
            uncached = max(0, req.prompt_tokens - cached_tokens)
            prefill_s += p.prefill_s(uncached)
            self.running.append(
                InFlight(
                    request=req,
                    arrival_ts=req.arrival_ts,
                    admitted_ts=now,
                    first_token_ts=-1.0,  # set when this iteration ends
                    tokens_emitted=0,
                    matched_blocks=matched,
                    uncached_tokens=uncached,
                )
            )

        duration = prefill_s + p.decode_s(len(self.running))
        end = now + duration
        self.iterations += 1

        finished: list[InFlight] = []
        still_running: list[InFlight] = []
        for seq in self.running:
            if seq.tokens_emitted == 0:
                seq.first_token_ts = end
            seq.tokens_emitted += 1
            if seq.tokens_emitted >= seq.request.output_tokens:
                finished.append(seq)
            else:
                still_running.append(seq)
        self.running = still_running
        return duration, finished
