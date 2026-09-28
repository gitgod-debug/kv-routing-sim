"""kv-routing-sim — which KV-cache routing policy should you actually run, and on how many replicas?

A trace-driven simulator of a multi-replica vLLM fleet, and a recommender on top of it. The cache
model follows vLLM 0.30.0's block pool and was checked against a live engine before anything was
built on it (Phase 3: exact on two sequential workloads under eviction pressure, +0.3% concurrent).

  traces/    adapters and the generator → requests as chains of block hashes (no text, ever)
  cache/     the block pool (vLLM's rules) and the plain-LRU ablation
  policies/  seven routers, one file each
  sim/       replicas, the event loop, the timing profile
  eval/      cost, Pareto, bootstrap, rank agreement, the recommendation
  calib/     `kvsim calibrate`: fit a profile to YOUR engine, and check the cache model on it
  cli.py     `kvsim replay | generate | crossover | calibrate`

The layout was written down as contracts before the code (ADR-001…004); the code implements them.
Read `docs/05-codebase-map.md` for the module-by-module contract.
"""

__version__ = "0.1.0"  # keep in step with pyproject.toml
