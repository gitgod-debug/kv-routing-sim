"""The seven routing policies. One file each.

Contract (ADR-003): `choose(fleet_state, request, rng) -> replica_index`. No global state; no
policy may mutate the fleet. A new policy is a NEW FILE, never a core change.

  round_robin · least_loaded · power_of_two · consistent_hash_prefix · radix_affinity ·
  radix_affinity_balanced · session_sticky
  (+ `base.py` for the interface, `registry.py` for name → class)

⚠️ EVERY TIE IS BROKEN FROM THE INJECTED `rng`. A `min()` over a dict silently prefers low replica
indices and so fabricates locality the real engine does not have — a correctness bug that looks
exactly like a result, and one that no amount of downstream calibration would surface.

Note on `session_sticky` vs `radix_affinity`, since they are the pair most often conflated: sticky
routes on SESSION identity, affinity on SHARED-PREFIX identity. If a thousand different users send
the same system prompt, sticky scatters them and every replica caches its own copy; affinity
concentrates them and one copy serves all. `NetX-lab/Frontier` has the former and not the latter,
which is the capability gap this project turns on (`docs/04-prior-art.md`).
"""

from .base import FleetState, RoutingPolicy
from .registry import ALL, BASELINE, REGISTRY, build, resolve

__all__ = ["ALL", "BASELINE", "REGISTRY", "FleetState", "RoutingPolicy", "build", "resolve"]
