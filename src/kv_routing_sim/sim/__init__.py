"""The simulated fleet: replicas, the timing profile, the event loop and run results.

Simulation is at request and block granularity with an iteration-level scheduler loop (ADR-001). Hit
rate depends on which blocks are resident where, and TTFT on queueing and uncached prefill, so
per-token simulation would add cost without changing either.
"""
