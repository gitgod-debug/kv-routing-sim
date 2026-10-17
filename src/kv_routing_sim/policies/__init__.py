"""Routing policies, one per module, behind the interface in `base.py` (ADR-003).

A policy implements `choose(fleet_state, request, rng) -> replica_index`, keeps no global state and
never mutates the fleet. `registry.py` maps names to classes: `all` is a fixed set of seven, and the
published policies `lmetric` and `smetric-local` are requested by name.
"""

from .base import FleetState, RoutingPolicy
from .registry import ALL, BASELINE, REGISTRY, build, resolve

__all__ = ["ALL", "BASELINE", "REGISTRY", "FleetState", "RoutingPolicy", "build", "resolve"]
