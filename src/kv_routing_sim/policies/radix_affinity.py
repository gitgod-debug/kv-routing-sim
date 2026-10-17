"""Radix prefix affinity: route each request to the replica with the longest matching prefix.

The router does not read the replicas' caches. It keeps its own map of block hash to the set of
replicas believed to hold it, built from what it has dispatched, so its view drifts when a
replica evicts a block. A block may live on several replicas at once; the map stores that set
as a bitmask and is LRU-bounded, as a real router's memory is.
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
        # 0.0 is pure locality, so this policy measures locality alone. A positive value scores
        # matched_blocks - load_penalty * outstanding, as a weighted-sum router does.
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

        # Only replicas holding the first block can match anything: stop-at-first-miss means a
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
        # Among equally good matches, prefer the least loaded, then break ties at random.
        loads = [state.outstanding(i) for i in candidates]
        return candidates[argmin_random(loads, rng)]

    def observe(self, replica: int, request: NormalizedRequest) -> None:
        """Record that `replica` now holds these blocks, adding to each block's set of holders."""
        bit = 1 << replica
        for h in request.block_hashes:
            if h in self._owners:
                self._owners[h] |= bit
                self._owners.move_to_end(h)
            else:
                self._owners[h] = bit
                if len(self._owners) > self.router_map_blocks:
                    self._owners.popitem(last=False)
