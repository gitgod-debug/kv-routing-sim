"""Map policy names to classes; the single source of truth for `--policies` (ADR-003)."""

from __future__ import annotations

from .base import RoutingPolicy
from .consistent_hash_prefix import ConsistentHashPrefix
from .least_loaded import LeastLoaded
from .lmetric import LMetric
from .power_of_two import PowerOfTwo
from .radix_affinity import RadixAffinity
from .radix_affinity_balanced import RadixAffinityBalanced
from .round_robin import RoundRobin
from .session_sticky import SessionSticky
from .smetric_local import SMetricLocal

# Ordered from simplest to most sophisticated, the order results tables use.
REGISTRY: dict[str, type[RoutingPolicy]] = {
    RoundRobin.name: RoundRobin,
    LeastLoaded.name: LeastLoaded,
    PowerOfTwo.name: PowerOfTwo,
    ConsistentHashPrefix.name: ConsistentHashPrefix,
    RadixAffinity.name: RadixAffinity,
    RadixAffinityBalanced.name: RadixAffinityBalanced,
    SessionSticky.name: SessionSticky,
    # Published policies, requested by name rather than through `all`.
    LMetric.name: LMetric,
    SMetricLocal.name: SMetricLocal,
}

# `all` is a fixed set: stored experiment configs say `policies: all`, and growing the set would
# change what they run. A policy added later is named explicitly.
ALL = (
    RoundRobin.name,
    LeastLoaded.name,
    PowerOfTwo.name,
    ConsistentHashPrefix.name,
    RadixAffinity.name,
    RadixAffinityBalanced.name,
    SessionSticky.name,
)
BASELINE = RoundRobin.name  # what every delta is measured against


def build(name: str, n_replicas: int, **kwargs: object) -> RoutingPolicy:
    """Construct a policy by name; an unknown name raises ValueError (ADR-004)."""
    try:
        cls = REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown policy {name!r}; available: {list(REGISTRY)}") from None
    return cls(n_replicas, **kwargs)


def resolve(names: str | list[str]) -> list[str]:
    """Turn `all` or a comma-separated list into validated policy names."""
    if isinstance(names, str):
        names = [n.strip() for n in names.split(",") if n.strip()]
    if names == ["all"]:
        return list(ALL)
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"unknown policies {unknown}; available: {list(REGISTRY)}")
    return names
