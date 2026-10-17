"""Consistent hashing on the prompt prefix: locality with no state and no load feedback.

The first few blocks of the prompt are hashed onto a ring of virtual nodes, so identical prefixes
always reach the same replica, and a hot prefix makes a hot replica; that is inherent to the policy,
not a bug. Virtual nodes, rather than `hash % n`, mean that adding a replica remaps about 1/N of the
keys, so a sweep across replica counts is not dominated by remapping.
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
