"""Radix affinity with a load term: score `matched_blocks - load_penalty * outstanding`.

Pure radix affinity sends every request that shares a first block to the replica holding it, so
under agentic traffic one replica takes nearly all the load; production prefix-aware routers balance
the two. The penalty converts queued requests into blocks of matched prefix and has no principled
default, so it should be swept.
"""

from __future__ import annotations

from .radix_affinity import RadixAffinity

# 1.0: one queued request weighs as much as one block of matched prefix; a placeholder.
DEFAULT_LOAD_PENALTY = 1.0


class RadixAffinityBalanced(RadixAffinity):
    name = "radix-affinity-balanced"

    def __init__(
        self, n_replicas: int, *, load_penalty: float = DEFAULT_LOAD_PENALTY, **kwargs: object
    ) -> None:
        kwargs.pop("load_penalty", None)
        super().__init__(n_replicas, load_penalty=load_penalty, **kwargs)
