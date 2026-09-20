"""Radix affinity WITH a load term — what production prefix-aware routers actually do.

⭐ WHY A SEVENTH POLICY EXISTS WHEN THE SPEC LISTS SIX. Discovered by running the sim, 2026-09-20:

Pure `radix-affinity` (load_penalty = 0) DEGENERATES. Under agentic traffic every request shares the
same system-prompt first block, so whichever replica owns that block wins the longest-prefix match
for every subsequent request, forever. Measured on a 900-request trace across 4 replicas:

    round-robin      [225, 225, 225, 225]    p95 TTFT    49 ms
    radix-affinity   [  0,   0,   0, 900]    p95 TTFT 22,218 ms      <- all on one replica

It buys a better hit rate (0.89 vs 0.74) and pays 450x in tail latency. That is a real property of
pure locality and worth keeping in the grid as the extreme — but it is NOT what anyone ships.

⛔ AND SHIPPING IT AS "THE PREFIX-AWARE POLICY" WOULD BE A STRAWMAN. SGLang's router balances.
Dynamo scores `matched_blocks - alpha * prefill_backlog`. `azrabano23/disagg-sim`'s `kv_aware` —
the closest published comparator — is explicitly that Dynamo-style blend (docs/04-prior-art.md). If
we compared our blend against their blend but published pure affinity as the prefix-aware number,
the comparison would be rigged in a direction that flatters the argument we are making. That is the
most damaging error available here.

So: `radix-affinity` stays pure and honest about what it is; this is the realistic one.

⚠️ ALPHA IS A GUESS AND MUST BE SWEPT. `matched_blocks - alpha * outstanding` compares blocks to
queued requests, two different units, so alpha carries the exchange rate between them. There is no
principled default. It is swept as its own axis in Phase 5 and the chosen value is published, the
same way every cost constant is (ADR-006).
"""

from __future__ import annotations

from .radix_affinity import RadixAffinity

# 1.0 => one queued request is "worth" one block of matched prefix. A placeholder, to be swept.
DEFAULT_LOAD_PENALTY = 1.0


class RadixAffinityBalanced(RadixAffinity):
    name = "radix-affinity-balanced"

    def __init__(
        self, n_replicas: int, *, load_penalty: float = DEFAULT_LOAD_PENALTY, **kwargs: object
    ) -> None:
        kwargs.pop("load_penalty", None)
        super().__init__(n_replicas, load_penalty=load_penalty, **kwargs)
