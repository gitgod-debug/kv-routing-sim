"""Radix prefix-affinity: route by LONGEST ACTUAL PREFIX MATCH. The policy this project is about.

⭐ THE FIDELITY DECISION, AND IT IS THE MOST IMPORTANT ONE IN THE POLICY LAYER.

This router does NOT read the replicas' caches. It keeps its OWN map of block -> replica, built
only from what it has dispatched — the DISPATCH VIEW of policies/base.py: the approximate mode of
Dynamo's router, and the tree most small open-source routers keep. Two consequences, both
deliberate:

  * Its view DRIFTS. When a replica evicts a block, this router does not find out. It keeps routing
    to a replica that no longer has the prefix, and the hit it predicted does not happen. That drift
    is real for a router of this kind, and it is part of what the calibration has to reproduce.
  * ⛔ This policy must not peek at the true caches: its numbers would then be another router's.

⚠️ CORRECTED 2026-10-05. This docstring said the dispatch view is "exactly as SGLang's router and
Dynamo's KV-aware router do", and that a view of the true caches "would be pure fiction". Dynamo's
default is the opposite — its KvIndexer is "a prefix tree built from worker-reported KV events" —
and so is the LMetric authors' router. For those the true view is the real one, late by an event
(docs/04-prior-art.md §2a). `policies/lmetric.py` can use either; this policy uses only its own.

⭐ AND A BLOCK LIVES ON MANY REPLICAS AT ONCE. The map is block -> SET of replicas, not
block -> one replica. Each replica that serves a prefix caches its OWN copy, so after a shared
system prompt has been served by three replicas, all three hold it and all three are valid targets.

That is not a detail. Modelling ownership as exclusive makes `observe()` overwrite the owner on
every re-route, so the "who has the system prompt" answer flips constantly and the router thrashes —
measured here before the fix, it scored a LOWER hit rate than round-robin, i.e. worse than not
trying. The set is stored as a BITMASK in one int: compact enough for millions of blocks, and O(1)
to test.

The map is bounded, because a real router's is: a process with finite memory, not an index of the
fleet.
"""

from __future__ import annotations

import random
from collections import OrderedDict

from ..traces.schema import NormalizedRequest
from .base import FleetState, RoutingPolicy, argmin_random


class RadixAffinity(RoutingPolicy):
    name = "radix-affinity"

    def __init__(
        self,
        n_replicas: int,
        *,
        router_map_blocks: int = 500_000,
        load_penalty: float = 0.0,
        **kwargs: object,
    ) -> None:
        super().__init__(n_replicas)
        # block hash -> bitmask of replicas believed to hold it. LRU-bounded.
        self._owners: OrderedDict[int, int] = OrderedDict()
        self.router_map_blocks = router_map_blocks
        # 0.0 = pure locality. A weighted-sum router uses matched_blocks - alpha * backlog; leaving
        # the knob at 0 keeps this policy a CLEAN statement of locality, so the H2 crossover
        # measures locality-vs-load rather than a blend that has already been tuned against it.
        self.load_penalty = load_penalty

    def _match_len(self, replica: int, blocks: tuple[int, ...]) -> int:
        bit = 1 << replica
        n = 0
        for h in blocks:
            if not (self._owners.get(h, 0) & bit):
                break  # stop-at-first-miss: a gap makes everything after it worthless
            n += 1
        return n

    def choose(self, state: FleetState, request: NormalizedRequest, rng: random.Random) -> int:
        if not request.block_hashes:
            return argmin_random(state.all_outstanding(), rng)

        # Only replicas holding the FIRST block can match anything: stop-at-first-miss means a
        # match that does not start at block 0 is worth nothing.
        first_owners = self._owners.get(request.block_hashes[0], 0)
        if first_owners == 0:
            # Nobody has this prefix. Place it where it will be cheapest to serve, which also
            # spreads new prefixes instead of piling them on one replica.
            return argmin_random(state.all_outstanding(), rng)

        # Only score replicas that hold block 0; the rest are known to match nothing.
        scores = [
            self._match_len(i, request.block_hashes) if (first_owners & (1 << i)) else 0
            for i in range(self.n_replicas)
        ]
        best = max(scores)
        if best == 0:
            return argmin_random(state.all_outstanding(), rng)

        if self.load_penalty > 0.0:
            adjusted = [
                s - self.load_penalty * state.outstanding(i) for i, s in enumerate(scores)
            ]
            top = max(adjusted)
            candidates = [i for i, a in enumerate(adjusted) if a == top]
        else:
            candidates = [i for i, s in enumerate(scores) if s == best]

        if len(candidates) == 1:
            return candidates[0]
        # Among equally-good matches, prefer the least loaded — then coin-flip.
        loads = [state.outstanding(i) for i in candidates]
        return candidates[argmin_random(loads, rng)]

    def observe(self, replica: int, request: NormalizedRequest) -> None:
        """Record that `replica` now holds these blocks — ADDING to the set, never replacing it."""
        bit = 1 << replica
        for h in request.block_hashes:
            if h in self._owners:
                self._owners[h] |= bit
                self._owners.move_to_end(h)
            else:
                self._owners[h] = bit
                if len(self._owners) > self.router_map_blocks:
                    self._owners.popitem(last=False)
