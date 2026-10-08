"""The replica fleet, the clock, and the numbers that come out.

Contract (ADR-001) — request-and-block granularity, with an iteration-level scheduler loop:
  * `replica.py` — one replica: continuous-batching admission bounded by `max_num_seqs`, a
                   prefill/decode cost function FITTED to the live engine (not derived from FLOP
                   counts), and a waiting queue.
  * `cost_model.py` — the timing profile: six coefficients, the KV capacity, and where they
                   were measured.
  * `metrics.py` — the collector. Emits DISTRIBUTIONS (p50/p95/p99), never a mean ± std for
                   latency.
  * `engine.py`  — N replicas, the router, the arrival stream and the event loop:
                   `run(trace, policy, n_replicas, seed) -> RunResult`. The one function the sweep
                   calls.

Why not per-token: the gradeable quantities are the cache hit rate and the TTFT distribution. Hit
rate is decided by which blocks are resident where, which is block-granular; TTFT by queueing plus
uncached prefill, which the iteration loop captures. Per-token fidelity would buy precision on an
axis nothing checks while making the sweep intractable.
"""
