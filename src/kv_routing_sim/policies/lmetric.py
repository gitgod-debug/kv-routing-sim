"""LMetric routing: minimise (queued plus new prefill tokens) times (active requests + 1).

Zhang et al., "Simple is Better: Multiplication May Be All You Need for LLM Request Scheduling",
OSDI '26 (arXiv 2603.15202). Ported from the authors' router, blitz-serving/blitz-router at
lmetric/camera-ready, `docs/dsl/policies.md`:

    policy lmetric-q:   Select min by prefill_tokens(req, sctx) · (sctx.bs + 1)

and `docs/dsl/schema.md` for the terms:

    sctx.bs                     "active batch size (running + queued)"
    prefill_tokens(req, sctx)   queued_tokens(sctx) + new_tokens(req, sctx)
    new_tokens(req, sctx)       req.tokens − hit_blocks(req, sctx) · block_size
    queued_tokens(sctx)         prefill tokens committed to the replica and not yet done:
                                += new_tokens(req, sctx) when a request is dispatched to it

The `+ 1` keeps idle replicas (bs == 0) distinguishable by their prefill load alone (their
`lmetric.rs`). A product needs no weight: in a weighted sum `cache_score - alpha * load` the right
alpha changes with the load, while in a product the exchange rate cancels when two replicas are
compared.

Choices this port makes, each of which can move a result:

1. The index. `index="events"` (the default) asks the engine what each replica holds, as the
   authors' router does by following the engine's step events; `index="dispatch"` uses this router's
   own record of what it sent where (see policies/base.py).
2. Queued prefill is released at the request's first token: in this simulator a request is prefilled
   in the step that admits it and emits its first token at that step's end.
3. Ties are broken with the injected rng; the authors' router takes the first minimum.
"""

from __future__ import annotations

import random

from ..cache.block_pool import max_hit_blocks
from ..traces.hashing import DEFAULT_BLOCK_SIZE
from ..traces.schema import NormalizedRequest
from .base import FleetState, argmin_random
from .radix_affinity import RadixAffinity

INDEXES = ("events", "dispatch")


class LMetric(RadixAffinity):
    name = "lmetric"
    wants_events = True

    def __init__(
        self,
        n_replicas: int,
        *,
        index: str = "events",
        block_size: int = DEFAULT_BLOCK_SIZE,
        **kwargs: object,
    ) -> None:
        if index not in INDEXES:
            raise ValueError(f"index must be one of {INDEXES}, got {index!r}")
        profile = kwargs.pop("profile", None)
        kwargs.pop("load_penalty", None)  # the product needs no weight
        super().__init__(n_replicas, **kwargs)
        self.index = index
        self.wants_cache_view = index == "events"
        # The engine's block size when it is known (the run's profile), else vLLM's default.
        self.block_size = int(getattr(profile, "block_size", block_size))
        self._queued_tokens = [0] * n_replicas  # prefill committed to each replica, not yet done
        self._pending: dict[str, tuple[int, int]] = {}  # request id -> (replica, its new tokens)
        self._last: tuple[str, list[int]] | None = None  # what choose() computed, for observe()

    def _hits(self, state: FleetState, request: NormalizedRequest) -> list[int]:
        """Leading blocks of the prompt each replica holds, by this router's index."""
        if self.index == "events":
            assert state.hit_blocks is not None  # the engine fills it for wants_cache_view
            return [state.hit_blocks(i, request) for i in range(self.n_replicas)]
        # The engine never reports a whole prompt as cached, since the last token is always
        # computed (cache/block_pool.py `max_hit_blocks`), so neither does this view of it.
        cap = max_hit_blocks(request.prompt_tokens, self.block_size)
        blocks = request.block_hashes[:cap]
        first = self._owners.get(blocks[0], 0) if blocks else 0
        return [
            self._match_len(i, blocks) if first & (1 << i) else 0 for i in range(self.n_replicas)
        ]

    def new_tokens(self, state: FleetState, request: NormalizedRequest) -> list[int]:
        """Per replica: the prompt tokens it would still have to prefill for this request."""
        return [
            max(0, request.prompt_tokens - h * self.block_size)
            for h in self._hits(state, request)
        ]

    def choose(self, state: FleetState, request: NormalizedRequest, rng: random.Random) -> int:
        new = self.new_tokens(state, request)
        self._last = (request.request_id, new)
        scores = [
            (self._queued_tokens[i] + new[i]) * (state.outstanding(i) + 1)
            for i in range(self.n_replicas)
        ]
        return argmin_random(scores, rng)

    def observe(self, replica: int, request: NormalizedRequest) -> None:
        assert self._last is not None and self._last[0] == request.request_id
        tokens = self._last[1][replica]
        self._queued_tokens[replica] += tokens
        self._pending[request.request_id] = (replica, tokens)
        if self.index == "dispatch":
            super().observe(replica, request)

    def on_first_token(self, replica: int, request: NormalizedRequest) -> None:
        entry = self._pending.pop(request.request_id, None)
        if entry is not None:
            self._queued_tokens[entry[0]] -= entry[1]
