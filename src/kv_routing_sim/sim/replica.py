"""One replica: a bounded batch, a waiting queue and a prefix cache, with continuous batching.

A request waits for a slot, not for the batch ahead of it to finish. Each iteration admits what
fits, prefills the newly admitted and advances every running sequence by one token. The cache is
looked up at admission rather than arrival, so a prefix that was warm when the request was routed
may have been evicted by the time it is admitted, as on a real engine.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ..cache.block_pool import VllmBlockPool
from ..cache.block_trie import BlockCache
from ..traces.schema import NormalizedRequest
from .cost_model import CostProfile

CACHE_MODELS = ("vllm", "lru")


@dataclass(slots=True)
class InFlight:
    request: NormalizedRequest
    arrival_ts: float
    admitted_ts: float
    first_token_ts: float
    tokens_emitted: int
    matched_blocks: int
    uncached_tokens: int
    num_preemptions: int = 0


class Replica:
    """One replica, with one of two cache models chosen by `cache_model`.

    * `"vllm"` (default): the block pool of `cache/block_pool.py`, driven as vLLM's scheduler drives
      it. Running requests are scheduled first, and one that cannot grow preempts the most recently
      admitted; waiting requests are admitted only in a step with no preemption, and only while
      memory lasts. Each rule cites scheduler.py at v0.30.0.
    * `"lru"`: the plain-LRU ablation baseline, with no memory-based admission control, no
      preemption and no freeing.
    """

    __slots__ = (
        "cache",
        "cache_model",
        "index",
        "iterations",
        "next_free_ts",
        "pool",
        "preemptions",
        "profile",
        "queue",
        "running",
    )

    def __init__(
        self,
        index: int,
        profile: CostProfile,
        cache_model: str = "vllm",
        track_block_metrics: bool = False,
    ) -> None:
        if cache_model not in CACHE_MODELS:
            raise ValueError(f"cache_model must be one of {CACHE_MODELS}, got {cache_model!r}")
        self.index = index
        self.profile = profile
        self.cache_model = cache_model
        self.cache = BlockCache(profile.kv_cache_blocks, block_size=profile.block_size)
        self.pool = VllmBlockPool(
            profile.kv_cache_blocks, profile.block_size, track_block_metrics=track_block_metrics
        )
        self.running: list[InFlight] = []
        # Fresh requests wait as NormalizedRequest; a preempted one waits as its InFlight, so it
        # keeps the tokens it has already generated (scheduler.py `_preempt_request`).
        self.queue: deque[NormalizedRequest | InFlight] = deque()
        self.next_free_ts = 0.0
        self.iterations = 0
        self.preemptions = 0

    def enqueue(self, request: NormalizedRequest) -> None:
        self.queue.append(request)

    @property
    def has_work(self) -> bool:
        return bool(self.running or self.queue)

    def step(self, now: float) -> tuple[float, list[InFlight]]:
        """Run one iteration. Returns (duration, sequences that finished)."""
        if self.cache_model == "lru":
            return self._step_lru(now)
        return self._step_vllm(now)

    def _step_lru(self, now: float) -> tuple[float, list[InFlight]]:
        """Run one iteration of the LRU model: admit, then one decode step for the whole batch."""
        p = self.profile
        prefill_s = 0.0

        while len(self.running) < p.max_num_seqs and self.queue:
            req = self.queue.popleft()
            assert isinstance(req, NormalizedRequest)  # the lru model never preempts
            matched = self.cache.query(req.block_hashes, req.prompt_tokens)
            cached_tokens = matched * p.block_size
            uncached = max(0, req.prompt_tokens - cached_tokens)
            prefill_s += p.prefill_s(uncached, cached_tokens)
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

        duration = prefill_s + p.decode_s(len(self.running), self._context_tokens())
        return duration, self._emit(now + duration)

    def _step_vllm(self, now: float) -> tuple[float, list[InFlight]]:
        """Run one `Scheduler.schedule()` step in vLLM's order.

        A scheduled sequence computes KV up to `prompt + emitted_so_far` tokens and samples one more
        token. A fresh request therefore prefills its prompt, a decoding one computes one token, and
        a preempted one, once readmitted, recomputes its prompt and everything it had generated,
        minus cache hits.
        """
        p = self.profile
        pool = self.pool
        prefill_s = 0.0
        preempted_now = False

        # 1. Running requests first (scheduler.py:612). One that cannot grow preempts running[-1],
        #    the most recently admitted under FCFS (scheduler.py:785), until it fits or has
        #    preempted itself, which ends scheduling of running requests (scheduler.py:797).
        i = 0
        while i < len(self.running):
            seq = self.running[i]
            need = seq.request.prompt_tokens + seq.tokens_emitted
            preempted_self = False
            while not pool.grow(seq.request.request_id, need, now):
                victim = self.running.pop()
                self._preempt(victim)
                preempted_now = True
                if victim is seq:
                    preempted_self = True
                    break
            if preempted_self:
                break
            i += 1

        # 2. Waiting requests, only in a step with no preemption (scheduler.py:854) and while
        #    slots remain (scheduler.py:864). A request that does not fit stops admission; waiting
        #    requests never preempt anyone (scheduler.py:1172-1178).
        if not preempted_now:
            while self.queue and len(self.running) < p.max_num_seqs:
                item = self.queue[0]
                seq = item if isinstance(item, InFlight) else None
                req = seq.request if seq is not None else item
                emitted = seq.tokens_emitted if seq is not None else 0
                num_tokens = req.prompt_tokens + emitted  # vLLM's request.num_tokens
                rid = req.request_id
                hit = pool.lookup(pool.chain_for(rid, req.block_hashes), num_tokens)
                if not pool.admit(rid, hit, req.block_hashes, num_tokens, now):
                    break
                self.queue.popleft()
                pool.record_lookup(num_tokens, len(hit), preempted=seq is not None)
                uncached = max(0, num_tokens - len(hit) * p.block_size)
                prefill_s += p.prefill_s(uncached, num_tokens - uncached)
                if seq is None:
                    seq = InFlight(
                        request=req,
                        arrival_ts=req.arrival_ts,
                        admitted_ts=now,
                        first_token_ts=-1.0,
                        tokens_emitted=0,
                        matched_blocks=len(hit),
                        uncached_tokens=uncached,
                    )
                self.running.append(seq)

        # A step in which everything was preempted computes nothing, and costs nothing.
        duration = prefill_s + (
            p.decode_s(len(self.running), self._context_tokens()) if self.running else 0.0
        )
        end = now + duration
        finished = self._emit(end)
        for seq in finished:
            pool.free(seq.request.request_id)
            pool.forget(seq.request.request_id)
        return duration, finished

    def _context_tokens(self) -> int:
        """Sum the batch's prompt and generated tokens, or 0 if the profile has no context term."""
        if not self.profile.decode_per_ctx_token_s:
            return 0
        return sum(s.request.prompt_tokens + s.tokens_emitted for s in self.running)

    def _preempt(self, seq: InFlight) -> None:
        """Preempt a sequence as scheduler.py `_preempt_request` does.

        Its blocks are freed but their hashes survive, its generated tokens are kept, and it goes
        back to the front of the waiting queue.
        """
        self.pool.free(seq.request.request_id)  # not forget(): its chain can be hit again later
        seq.num_preemptions += 1
        self.preemptions += 1
        self.queue.appendleft(seq)

    def _emit(self, end: float) -> list[InFlight]:
        """Every scheduled sequence emits one token at the end of the step."""
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
        return finished
