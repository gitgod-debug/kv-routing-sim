"""Consistent hashing on the prompt prefix: stateless locality, zero load feedback.

How it works: take the first few blocks of the prompt as a key, hash it onto a ring of virtual
nodes, and send the request to the replica that owns that point. Identical prefixes always land on
the same replica, so the cache hits — with no coordination and no state anywhere.

⚠️ AND THE FAILURE MODE IS THE POINT, not a bug to fix: a hot prefix means a hot replica. If every
user on the fleet sends the same system prompt, this policy sends EVERY request to one replica and
the other N-1 sit idle. It is the purest form of "locality with no load feedback", which makes it
the clearest illustration of the tension H2 is about.

Virtual nodes (not `hash % n`) so that adding a replica reshuffles ~1/N of the keyspace instead of
almost all of it — the actual point of consistent hashing, and the reason a sizing sweep across
replica counts stays meaningful instead of being dominated by remapping churn.
"""

from __future__ import annotations

import bisect
import hashlib
import random

from ..traces.schema import NormalizedRequest
from .base import FleetState, RoutingPolicy


def _h(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "big")


class ConsistentHashPrefix(RoutingPolicy):
    name = "consistent-hash-prefix"

    def __init__(
        self, n_replicas: int, *, key_blocks: int = 4, vnodes: int = 160, **kwargs: object
    ) -> None:
        super().__init__(n_replicas)
        self.key_blocks = key_blocks
        self._ring: list[int] = []
        self._owner: dict[int, int] = {}
        for replica in range(n_replicas):
            for v in range(vnodes):
                point = _h(f"replica-{replica}-vnode-{v}".encode())
                self._ring.append(point)
                self._owner[point] = replica
        self._ring.sort()

    def choose(self, state: FleetState, request: NormalizedRequest, rng: random.Random) -> int:
        if not request.block_hashes:
            # No prompt blocks to key on (a very short prompt). Falling back to a random replica
            # rather than a fixed one keeps the no-prefix case from silently biasing replica 0.
            return rng.randrange(self.n_replicas)
        key = request.block_hashes[: self.key_blocks]
        point = _h(b"".join(h.to_bytes(8, "big") for h in key))
        idx = bisect.bisect(self._ring, point) % len(self._ring)
        return self._owner[self._ring[idx]]
